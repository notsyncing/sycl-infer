# 设计 04：引擎编排与图执行

覆盖 `src/engine/engine.{h,cpp}`、`engine_graph.cpp`、`sampler.{h,cpp}`；KV 池见
[05-kv-cache.md](05-kv-cache.md)，前缀缓存见 [06-prefix-cache.md](06-prefix-cache.md)。

---

## 1. 职责

`si::engine` 把静态权重（`model`）与 tokenizer 组合成一个可执行的推理运行时：

* 选择计算后端（GPU 或 CPU）并为（可能是多个）设备分配所有激活/状态/KV/图缓冲区；
* 把一次完整 forward 编码为 `seg_plan` 并在 GPU 上录制为 SYCL command graph；
* 提供 prefill / decode / 单序列 API；
* 管理动态 KV 池与三层前缀缓存（多设备时关闭，见下）。

`engine` 的核心数据结构是 `seg_plan`；核心函数是 `build_plan` 与 `record_forward`，二者必须保持
调用顺序一致。所有 kernel 调用都经过 `compute_backend`（见 §11），因此同一份前向逻辑在 GPU
（录制图）与 CPU/多设备（直接重放）上都成立。

---

## 2. 生命周期

### 2.1 构造（`engine.cpp:19-239`）

1. **模型/分词器/权重**：`m.load(path)` → `tk.load(m.gguf)`。队列由 `make_queue(device_req)` 选择：
   GPU（`sycl::gpu_selector_v`，默认）或 CPU（`sycl::cpu_selector_v`，`--device cpu`），都带
   `property::queue::in_order`（视觉阶段/CPU 分配依赖顺序）。权重：GPU `m.upload(q)` 拷贝整文件；
   CPU `m.upload(q, host=true)` 只把 `dev_ptr` 退化为 mmap 指针，不拷贝。
2. 若给了 `--layer-map`，`setup_multi_device` 建多个后端、每 GPU 一份整文件权重副本，并把
   `layer_dev_` / `layer_attn_local_` 填好（见 §11）。
3. `PF_META` → `build_meta32`（默认关）。
4. `PF_DP4A` → `pf8`、`pf8_dec`（默认都开）。GPU 在 `pf8` 时 `m.build_w8(q)`（约 700 MB）；
   CPU 读 GGUF 整数 block，**不建 w8**。
4. **前缀缓存配置**：`PF_DEC_SPLIT` clamp；计算 `pc_state_floats`（每检查点 float 数）；由
   `PF_PC_STATES` / `PF_PC_VRAM_MB` / `PF_PC_MEM_MB` 得 `pc_max_states`；RAM/磁盘预算；用
   `kv_cap_mb` 收缩三层预算；`pc_max_states == 0` 关闭整个缓存；打开 RAM/磁盘 store。
5. **oneDNN**：`use_dnnl = pf8 && dnnl_gemm_enabled()`；为所有 SI8 权重 `add_weight`；除
   `PF_DNNL_NOWARM` 外 `warmup()`。
6. **尺寸**：`ffn_stride = 2*n_ff`，`max_blocks = ceil(max_seq/kBlockSize)`。
7. **池预留**：`pool_cap = max(n_blocks_, cap_blocks)`，前缀缓存开启时至少 `pc_max_states`；
   `pool_initial = n_blocks_`；`PF_KV_GROW` → `pool_chunk`。
8. **关键**：`n_blocks = pool_cap`（指向预留而非已提交），因此 `kv_layer_stride` 与录制进图的每层
   基址在池增长时保持稳定。
9. `alloc_buffers()` → `build_graphs()`。

### 2.2 析构（`engine.cpp:241-301`）

1. `pc_flush_to_disk()`（在池/检查点仍存活时）；
2. 释放所有设备分配；
3. `pool_print`、`pc_print_stats`、磁盘/RAM 统计；
4. `kv_release_pool()`、`m.free_w8(q)`、释放权重 blob 与 `h_logits`。

### 2.3 状态重置

* `zero_slot(slot)`：memset 一个序列的 GDN + conv 状态。
* `reset_state()`：清零所有 slot 递归状态、memset `d_info`、**从 `hp.rope_sections` 恢复
  `mrope_sections`**（memset 会清掉这个模型常量）、清空所有 `pc_slot_`、释放 pending 检查点预留。
* `reset_single()`：先 `reset_state()`，再设 `n_rows=1, tpb=kMaxT, n_real=1, pos[0]=0, slot[0]=0,
  active[0]=1, tokens[0]=0`。`eval`/`generate` 使用的单序列状态。

---

## 3. 缓冲区布局（`engine.cpp:353-433`）

行数 `R = kMaxB*kMaxT = 512`：

| 缓冲 | 大小 |
|---|---|
| `d_x`, `d_xnorm` | `R * n_embd` |
| `d_qkv` | `R * 3*d_inner` |
| `d_z` | `R * d_inner` |
| `d_beta`, `d_alpha` | `R * dt_rank` |
| `d_conv_out` | `R * 3*d_inner` |
| `d_attn_pre`, `d_attn_merged` | `R * d_inner` |
| `d_qbuf` | `R * n_head * 2 * head_dim`（第二半是 gate） |
| `d_kbuf`, `d_vbuf` | `R * n_head_kv * head_dim` |
| `d_attn_out` | `R * n_head * head_dim` |
| `d_ffn` | `R * ffn_stride` |
| `d_partials` | `R * n_head * n_splits * (2+head_dim)` |
| `d_partials_dec` | `kMaxB * n_head * dec_splits * (2+head_dim)` |
| `d_logits` | `kMaxB * n_vocab` |
| `d_last_hidden` | `n_embd` |
| `d_img_embd` | `kMaxImgTokens * n_embd` |

* `kv_layer_stride = n_blocks * kv_block_bytes()`（`n_blocks` 是预留）。
* `d_gdn_state` / `d_conv_state` 按 **layer-major** 索引进分配（`[n_gdn][kMaxB][...]`），尽管头注释
  写作 `[kMaxB][n_gdn][...]`；实际索引见 `engine.cpp:432-433`。
* `d_info` 是 `sycl::malloc_host<step_info>`（host USM），清零。
* 段数组：`d_segs_dec`（1024，已废弃）、`d_segs_pf`（4096）、`d_segs_pf8`（4096）、`d_segs_aux`（8）。
* SI8 scratch：`d_x8`、`d_xmeta`（float2）、`d_xsumq`，行数 `kMaxB*kMaxT`。
* decode 桶 `{1,2,4,8,16}` 各带自己的 `d_segs`（1024）。
* 前缀缓存检查点池 `d_pc_states[pc_max_states][pc_state_floats]` + 空闲/时间戳/owner 向量。
* 最后 `reset_state()`。

---

## 4. `seg_plan` 与 `build_plan`

### 4.1 `seg_plan`（`engine.h:30-92`）

一次完整 forward 的全部 GEMV/GEMM 工作：

* `segs` — 扁平的 `gemv_seg` 列表。
* `call_offsets[i]` / `call_counts[i]` — call `i` 在 `segs` 中的切片。
* `call_total_rows[i]`、`call_tb[i]`（token 块宽）、`call_nsb[i]`（K/256 split 提示）。
* `groups[]` — call 内连续同 type 段的 `{type, off, n, rows}` 合并。
* `call_group_begin/count[i]` — 每 call 在 `groups` 中的范围。
* `call_xq[i]` — DP4A 路径的激活量化描述 `{x, up, x_stride, up_stride, K}`。

辅助：`begin_call(tb,nsb)`、`add(seg)`、`set_xq(...)`、`finalize()`。`finalize()` 对每个 call 的段按
`type` 稳定排序，再把连续同 type 合并成 group 并记录每 call 的 group 范围。

### 4.2 `build_plan(T, tb, head_batched, use_w8, with_head)`（`engine_graph.cpp:51-197`）

* `n_slices = use_w8 ? 1 : ceil(T/tb)`。fp32 路径把每个段复制到 `tb` token 切片；w8 路径是整块单段。
* `add8` 构造 w8 段（`type = w8.vals ? 12 : 0`，12 只是分组占位，真实量化类型在 `w8.type`），
  `x8=d_x8`、`xmeta=d_xmeta`、`xsumq=d_xsumq`。
* `mk` 构造 fp32 段，用 `m.dev_ptr(w.data)` 与 `meta32_of(w.data)`。

每层固定四个 call：

**GDN 层**

| call | 内容 |
|---|---|
| 1 | w8: `wqkv8→d_qkv`、`wgate8→d_z`，`set_xq(d_xnorm)`；另加 fp32 `ssm_beta→d_beta`、`ssm_alpha→d_alpha`。四个投影共享一次激活量化 |
| 2 | `ssm_out8`/`ssm_out`，residual `d_x→d_x` |
| 3 | `ffn_gate` + `ffn_up`（up 在 `d_ffn+n_ff`） |
| 4 | `ffn_down`；w8 时 `set_xq(d_ffn, up=d_ffn+n_ff, K=n_ff)`（量化时应用 SiLU）；fp32 时设 `act_up` |

**attention 层**

| call | 内容 |
|---|---|
| 1 | `wq/wk/wv` → `d_qbuf/d_kbuf/d_vbuf` |
| 2 | `wo` → `d_x`（residual `d_x`） |
| 3-4 | 与 GDN 相同的 FFN call |

**head**（仅当 `with_head || head_batched`）：`begin_call(head_batched ? tb : 1, n_embd/256)`，
`tok_embd` GEMV；`x = head_batched ? d_xnorm : d_last_hidden`，`out = d_logits`。仅 batched decode
（`use_w8 && head_batched && tok_embd8`）走 int8。prefill 保持 fp32（只需一行）。

因此每个 plan 有 `n_layer*4` 个 call，加可选的一个 head call。

### 4.3 已录制的 plan 矩阵（`build_graphs`）

| plan | 形状 | 用途 |
|---|---|---|
| `plan_pf_` | `build_plan(kMaxT, 8, false)` | fp32 prefill，1 行 × 32 token，8-token 切片 |
| `buckets_[b].plan` | `build_plan(b.tb, b.tb, true)` | decode，`T=tb`、头批处理、无 w8 |
| `plan_pf8_` | `build_plan(kMaxT, kMaxT, false, true, true)` | w8 prefill 带 head |
| `plan_pf8_nh` | `build_plan(kMaxT, kMaxT, false, true, false)` | w8 prefill 不带 head（非最终 chunk） |
| `plan_dec8_` | `build_plan(1, 1, true, true, true)` | w8 batch-1 decode |
| `plan_pfb_` | `build_plan(kMaxT, kMaxT, false, true, true)` | 共享的 w8 chunk-batched plan |

`head_batched=true`（decode）写 `d_logits[B][vocab]` 并读 `d_xnorm`；`false`（prefill）写一行、读
`d_last_hidden`（由 `copy_row` 产生）。

---

## 5. `record_forward`

`record_forward(mode, plan, d_segs, rows, d_segs_rows, at_nsp_hint)` 执行（或录制）一次完整 forward。

| mode | 含义 | `T` | `nrows` | `nreal` | `NCH` |
|---|---|---|---|---|---|
| 0 | decode | `rows`（批大小） | `rows` | 1 | 1 |
| 1 | chunked prefill（1 行 × ≤32 token） | `kMaxT` | 1 | `rows` | 1 |
| 2 | chunk-batched prefill（`rows` 个 token = NCH 个 chunk） | `rows` | `NCH` | `kMaxT` | `ceil(rows/kMaxT)` |

### 5.1 每个 plan call（`gemv_at`）

1. `tb = call_tb[idx]`，`single = (tb==1)`，`nb = single?1:NCH`，`tbm = (mode==2 && !single) ? rows : tb`。
2. **oneDNN 判定**：`use_dnnl && mode != 0 && !single`，call 有 xq 描述，且该 call 的每个 w8 段都有匹配
   `K` 的已转换权重时走 oneDNN。
3. **激活量化**：oneDNN → `dnnl->quantize`；否则 `call_xq[idx].x` 存在时 `xq_launch`（模式 2 一次覆盖
   `tbm` token，否则逐行 `r` 用 `TB=tb`）。
4. **段分发**（按 `s0.w8.vals`）：
   * w8 且 `mode != 0 && !single` → 每段一次 `dp4a_gemm_launch`（或 oneDNN `gemm`）；
   * 否则逐行 `r`：`tb==1` 用 `dp4a_gemv_launch`，否则 `dp4a_gemm_launch(..., tb)`；
   * fp32 模式 2 → 每 group 一次 `gemv_group_launch(..., tb, nsb, NCH)`（一次 dispatch 覆盖所有 chunk 行）；
   * fp32 其他 → 每行一次 `gemv_group_launch`。

`gemv()` 调用 `gemv_at(ci++)`，因此 call 顺序严格等于 plan 顺序。

### 5.2 kernel 序列

```
embed(tok_embd → d_x)
for il in 0..n_layer-1:
    rmsnorm(d_x, attn_norm → d_xnorm)
    if GDN:
        gemv()                                   # call1
        [conv_l2 + conv_state_update + gdn] 或融合路径
        gated_norm(d_attn_pre, d_z, ssm_norm → d_attn_merged)
        gemv()                                   # call2 ssm_out
    else:
        gemv()                                   # call1 wq/wk/wv
        qk_norm_rope(qbuf, kbuf, vbuf, pool, tables, info)
        attn(..., out = fused ? d_attn_out : nullptr)
        if !fused: attn_combine(partials, qbuf → d_attn_out)
        gemv()                                   # call2 wo
    rmsnorm(d_x, post_attn_norm → d_xnorm)
    gemv()                                       # call3 ffn_gate+up
    gemv()                                       # call4 ffn_down
rmsnorm(d_x, output_norm → d_xnorm)
if mode != 0: copy_row(d_xnorm → d_last_hidden, row=-1)
gemv_at(call_offsets.size()-1)                   # LM head
```

`record_forward` 的 call 顺序必须与 `build_plan` 同步。

### 5.3 GDN 融合路径

* 每个 GDN 层的 `pc_snap` 切片在录制时计算好，供 conv/GDN kernel 写检查点。
* 模式 2 且 `PF_GDN_FUSE>=1`（默认 2）时，整个批次一次性物化：`conv_l2(cross_row=true, NCH, kMaxT)` →
  `conv_state_update(last_row_only=true)` → 完全融合的 `gdn_launch`（`fuse_gdn>=2`）或逐行 GDN
  （`fuse_gdn==1`）。这把每行 launch 链从 48 降到 4。
* 否则走经典逐行循环。
* `gated_norm` 无行序依赖，所有 chunk 行一次 dispatch。

### 5.4 attention split 选择

* `PF_ATTN_SPLIT` 强制 split 数（clamp）。
* 模式 0 → `dec_splits`（默认 `kMaxSplits`，上限 `kMaxDecSplits`）。
* 模式 2 → `n_splits`，或录制变体的 `at_nsp_hint`，否则
  `ceil(max_nkv / PF_ATTN_SPLIT_KEYS)`（默认 512），clamp 到 `[1, n_splits]`。
* 模式 1 → `n_splits`。
* `at_fused = (nsp==1 && PF_ATTN_FUSE!=0)` 时 attention 直接写输出并跳过 `attn_combine`。

### 5.5 诊断开关

`PF_ABL_NOATTN` 跳过 attention 块但仍消费 `gemv()`（`wo`）；`PF_ABL_NOGDN` 跳过 GDN；
`STOP_AFTER_LAYER` 在 N 层后停；`PF_DBG_MID`（4=post-embed，1=mid-layer，2=post-attn-norm，
3=post-ffn-gate）提前退出。`PF_PROF` 需要 `PF_NOGRAPH`（`prof_on()` 要求两者），按 gemv/xq/head/
attn/gdn/norm/other 分组计时，并打印每 call 的 top-8 表。

---

## 6. SYCL command graph

### 6.1 捕获与重放（`build_graphs`）

* `capture_guard` 在整个构建期间置全局 `g_capturing`，禁用 prof 等待。
* 每张图：`command_graph<modifiable>(ctx, dev)` → `begin_recording(q)` → `record_forward(...)` →
  `end_recording()` → `finalize()` 成 `command_graph<executable>`。
* 重放：`q.ext_oneapi_graph(*e)` 然后 `q.wait()`。

### 6.2 decode 桶

`buckets_` 对应 `tb ∈ {1,2,4,8,16}`。`decode_batch` 选最小的 `tb >= n_rows` 的桶，否则最后一个。
超出 `n_rows` 的行标 `active=0`，kernel 跳过。

### 6.3 prefill 变体

| 图 | 条件 | 说明 |
|---|---|---|
| `e_pf` | `pf8 == false` | fp32 prefill |
| `e_pf8` | `pf8` | w8 prefill 带 head（最终 chunk） |
| `e_pf8_nh` | `pf8` | w8 prefill 不带 head（非最终 chunk） |
| `e_dec8` | `pf8_dec` | w8 batch-1 decode |
| `pfb_vars_` | `!use_dnnl` 且 `pf8` | `ntok ∈ {2,4,8,16}*kMaxT` 的 chunk-batched 图，每个尺寸一张 |

`use_dnnl` 时 `pfb_vars_` 只保留 `ntok`（oneDNN 不能录制），`batched_prefill_fit` 允许任意
`kMaxT` 倍数直到 `kMaxB*kMaxT`；DP4A 时只能选已录制尺寸。

### 6.4 图冻结与 `step_info` 不变量

command graph 记录的是 kernel 命令列表；录制时按值传入的主机标量（`rows`、`T`、`tb`、`nsp`、`mode`、
指针运算、`gemv_seg` 内容、env 派生的常量如 `at_split`、`fuse_gdn`）都被烘焙进可执行图。因此：

1. **所有逐步状态必须放进 host-USM 的 `step_info`，在 kernel 体内读取**，绝不能在图录制时于主机读取。
   图只保存 `d_info` 指针，kernel 在重放时看到当前内容。
2. `record_forward` 中唯一一次对 `d_info` 的主机读取（模式 2 从 `d_info->pos[r]` 推导 split）只在
   `at_nsp_hint == 0`（直接模式 2 路径）时发生，绝不在录制变体里。
3. 通过函数内 `static` 读取的 env 开关在首次使用时锁定，不会每次重放重读。

---

## 7. 入口点

### 7.1 批处理 API（调用者持有 `engine::mtx`）

* `prefill_chunk(toks, start, n, slot, with_head=true)`（`engine.cpp:483-505`）：
  `pc_capture_begin` → 填 `d_info`（`n_rows=1, tpb=kMaxT, n_real=n, pos[0]=start, slot, active`）→
  `PF_NOGRAPH&&pf8` 直接 `record_forward(1, plan_pf8_, ...)`，否则重放 `e_pf8`/`e_pf8_nh`/`e_pf` →
  `pc_active=0`。
* `prefill_batch(toks, start, n, slot, pos0)`（`engine.cpp:510-561`）：`n` 必须是 `kMaxT` 倍数；填 NCH 行；
  直接模式 2 或精确匹配 `pfb_variant.ntok`。头注释标注为实验性、仅直接路径正确。
* `decode_batch(tokens, poss, slots, n_rows)`（`engine.cpp:563-592`）：填 `n_rows/n_real/tpb=1` 与每行
  `pos/slot/active/tokens`；`pf8_dec && n_rows==1` 用 `e_dec8`，否则最小适配桶；总是 `q.wait()`。
* `fetch_logits(row, out)`：`d_logits` → `h_logits` → `out`。

### 7.2 无图诊断/回退

`forward_plain_pf`、`forward_plain_pf8`、`forward_plain_dec(rows)` 直接调用对应 `record_forward`。

### 7.3 单序列 API

* `eval(tokens)`（`engine.cpp:638-666`）：加锁、`reset_single`、按 `kBlockSize` 分配块、`set_table`、
  以 32 token 分块 `prefill_chunk`、`run_head`、释放块、返回最后一个 token 的 logits。无前缀缓存、无采样。
* `run_head()`（`engine.cpp:616-636`）：为 `tok_embd` 构造一个 aux `gemv_seg`（`d_last_hidden → d_logits`），
  用 `gemv_group_launch` 启动，拷回主机。注意它会重算最终 prefill chunk 已由 `e_pf8` 写好的 head。
* `generate(prompt, gp, cb, first_logits)` / `generate_mm(p, gp, cb, first_logits)`：加锁后调
  `generate_impl`。
* `generate_impl`（`engine.cpp:680-791`）：
  1. `reset_single`；seed `mt19937_64`（`gp.seed` 或 `random_device`）。
  2. 多模态：校验 `embd` 行数 ≤ `kMaxImgTokens`；若无设备指针则拷贝到 `d_img_embd`。
  3. 分配 prompt 的块并 `set_table(0, blocks)`。
  4. **prefill 循环**：按 32 token 分块；有图时设 `mrope_on=1`、`img_embd`，逐 chunk 填
     `d_info->mrope[s*(kMaxB*kMaxT)+i]` 与 `img_row[i]`，再 `prefill_chunk`。所有 chunk 都带 head。
  5. `run_head()` → `logits`；可选输出 `first_logits`。
  6. `next_pos = mm ? mm->pos_after : nprompt`；清 `img_embd`。
  7. **decode 循环**：`sample_token`（`out` 同时作为 `recent`）→ push token → EOS/回调/`max_tokens`/
     `pos >= max_seq-1` 终止 → 每 32 token 扩块表 → 有图时把四个 section 的 mrope 都设为 `next_pos`
     → `decode_batch` → `pos++`、`next_pos++` → `fetch_logits`。
  8. 释放块。

注意：KV slot 序号是 token 计数，RoPE 位置单独由 `mrope` 携带 —— 这正是多模态不变量
（[architecture.md](../architecture.md)）。

---

## 8. 采样器

见独立文档 [07-sampler.md](07-sampler.md)。

---

## 9. 相关环境变量

| 变量 | 作用 |
|---|---|
| `PF_DP4A` / `PF_DP4A_DEC` | int8 计算路径开关 |
| `PF_GEMM_DNNL` / `PF_DNNL_NOWARM` / `PF_DNNL_TIME` | oneDNN 路径 |
| `PF_META` / `PF_SI4` | 权重旁路数组 / 4-bit 重化 |
| `PF_ATTN_SPLIT` / `PF_ATTN_SPLIT_KEYS` / `PF_ATTN_FUSE` / `PF_DEC_SPLIT` / `PF_DEC_GROUP` / `PF_ATTN_VEC` | attention |
| `PF_GDN_COLS` / `PF_GDN_WG` / `PF_GDN_VEC` / `PF_GDN_FUSE` / `PF_GDN_DBG` | GDN |
| `PF_GEMV_*` / `PF_GEMM_*` / `PF_MT_*` / `GEMV_*` | GEMV/GEMM 调优 |
| `PF_NOGRAPH` / `PF_PROF` / `PF_TIME` / `PF_DBG_*` / `PF_PROF` | 诊断/回退 |
| `PF_KV_TYPE` / `PF_KV_F32` / `PF_KV_BF16` / `PF_KV_CAP_MB` / `PF_KV_GROW` | KV |
| `PF_PREFIX_CACHE` / `PF_PC_*` | 前缀缓存 |

---

## 10. 已知细节与注意点

* `pf8_dec` 默认 on（GPU 与 CPU 都走 int8 decode，除非 `PF_DP4A_DEC=0`）。
* 无 head 的 prefill 变体（`plan_pf8_nh` / `plan_pf_nh_slot`）由 `seg_plan::has_head=false` 标记，
  `record_forward` 只在 `plan.has_head` 时重放最后一个 call，因此不会重跑 `ffn_down`。
* `run_head` 会重复最终 chunk 已经写好的 head；`d_segs_pfb` 的行偏移副本 / `row_offset_seg` 在当前
  分发路由下处于休眠状态（模式 2 的 fp32 用 token-block 网格，w8/i8 用连续 `tbm=rows`）。

---

## 11. 后端抽象、设备选择与多设备

引擎的所有 kernel 调用都经过 `compute_backend`（`src/backend/backend.h`），它一一镜像
`src/backend/gpu/kernels/kernels.h` 的启动 API：

* `gpu_backend`（`backend/gpu/gpu_backend.cpp`）转发到 SYCL kernel；`record_forward` 在 command
  graph 录制期间调用它。
* `cpu_backend`（`backend/cpu/cpu_backend.cpp`）把 `step_info`/`gemv_seg` 转成无 SYCL 的镜像
  （`cpu_types.h`）后调用 `src/backend/cpu/kernels/`；主机 kernel 同步执行。

**选择与执行路径**

* `--device cpu|gpu|auto`（或 `PF_DEVICE`）在构造时决定 `dev_kind`/`cpu_mode` 与活跃 `be`。
* GPU 单设备：`build_graphs` 录制 command graph，`prefill_chunk`/`decode_batch` 重放图。
* CPU 或 `multi_dev`：`build_graphs` 只调用 `build_plans()`，`prefill_chunk`/`decode_batch` 直接调用
  `record_forward`（与 `PF_NOGRAPH` 相同的直放路径），`sync_all()` 同步所有后端。
* CPU 的 prefill 计划按 token 数分档（`plan_pf_slot[i] = build_plan((i+1)*8, 8, ...)`，含 head 与
  no-head 两套），`prefill_chunk` 用 `round_up(n,8)` 选最小的一份，短 prompt 不再算满 32 行。

**多设备（pipeline parallel）**

* `--layer-map 0-11:gpu,12-23:cpu`：`setup_multi_device` 解析闭区间、校验无缝隙覆盖 `[0,n_layer)`，
  建立 `backends_`（GPU 优先作为全局张量/LM head 的主设备）与 `layer_dev_`。
* `build_plan` 用 `wptr(dev, host)` 为每层解析该设备的权重指针；`record_forward` 用 `cur_be = &be_of(il)`
  逐层切换后端，并在层边界 `synchronize()` 上一个设备（激活在主机 USM，交接无需拷贝）。GPU 只上传
  分给它的层（`weight_maps_` 逐张量上传），不持有整份 GGUF 副本。
* `layer_attn_local_[il]` 给出该层在**所属设备**注意力层里的序号；`kv_setup` 为每个设备分配只含
  其注意力层的 paged KV 池，block id 全局一致（同一张 block table，见 [05-kv-cache.md](05-kv-cache.md)）。
* 多设备关闭前缀缓存（三层记录需要多池）、`pf8` 与 oneDNN；递归状态（GDN/conv）在共享主机 USM 中
  按全局 GDN 层序号索引，因此混合模型也可用。
