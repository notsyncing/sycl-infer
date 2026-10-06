# 设计 04：引擎编排与图执行

覆盖 `src/engine/engine.{h,cpp}`、`engine_graph.cpp`、`engine_kvpool.cpp`、`engine_mtp.cpp`、
`sampler.{h,cpp}`；KV 池的存储格式见 [05-kv-cache.md](05-kv-cache.md)，前缀缓存见
[06-prefix-cache.md](06-prefix-cache.md)，kernel 侧见 [03-kernels.md](03-kernels.md)。MTP / NextN
投机解码的引擎侧设计见 **§12**，它与前 11 节共用同一套 plan / `record_forward` / `step_info` 机制。

---

## 1. 职责

`si::engine` 把静态权重（`model`）与 tokenizer 组合成一个可执行的推理运行时：

* 选择计算后端（GPU 或 CPU）并为（可能是多个）设备分配所有激活/状态/KV/图缓冲区；
* 把一次完整 forward 编码为 `seg_plan` 并在 GPU 上录制为 SYCL command graph；
* 提供 prefill / decode / 单序列 API；
* 管理动态 KV 池与三层前缀缓存（多设备时同样启用，见 §11）；
* 在模型自带 NextN 头时提供 MTP 投机解码的单序列循环（`--mtp [N]`，见 §12）。

`engine` 的核心数据结构是 `seg_plan`；核心函数是 `build_plan` 与 `record_forward`，二者必须保持
调用顺序一致。所有 kernel 调用都经过 `compute_backend`（见 §11），因此同一份前向逻辑在 GPU
（录制图）与 CPU/多设备（直接重放）上都成立。

引擎按翻译单元分成四个文件：`engine.cpp`（生命周期、缓冲区、多设备装配、`prefill_*`/`decode_*`
入口、单序列 API）、`engine_graph.cpp`（`seg_plan` 构建、`record_forward`、图录制/重放、
prof/launch 诊断）、`engine_kvpool.cpp`（虚拟地址动态 KV 池与 `attn_layers()`）、`engine_mtp.cpp`
（§12 的草稿/校验循环）。

---

## 2. 生命周期

### 2.1 构造（`engine.cpp:91-512`）

1. **模型/分词器**：`m.load(path)` → `tk.load(m.gguf)`。队列由 `make_queue(device_req)`
   （`engine.cpp:38-52`）选择：GPU（`sycl::gpu_selector_v`，默认）或 CPU
   （`sycl::cpu_selector_v`，`--device cpu`），都带 `property::queue::in_order`（视觉阶段/CPU
   分配依赖顺序）。`resolve_device`（`engine.cpp:19-36`）还负责一件事：**一个不含 `gpu` 的
   `--layer-map` 直接解析成 CPU**，否则主队列是 GPU 队列而所有层都在 CPU 后端上，prefill 侥幸
   存活、单 token decode 静默出错。
2. **MTP 装配**（`engine.cpp:100-153`，细节见 §12）：`mtp_k` 取 `--mtp N`，`PF_MTP` 覆盖；
   GGUF 没有 `blk.<n_layer>.nextn.*` 就打印一行 `[mtp] model has no NextN ... - MTP disabled`
   并回落；`mtp_k` clamp 到 12（`n = k+1 <= kMaxB`，`d_logits` / `step_info` 的上限）；`PF_MTP_SPLITS`
   覆盖 `mtp_splits`（默认 `kMaxDecSplits`）；`PF_MTP_FORCE_INT8` 把 `PF_CB4`/`PF_K5` 强置 0。
3. **多设备装配**：若给了 `--layer-map`，`setup_multi_device`（`engine.cpp:1409-1676`）建多个
   后端、填 `layer_dev_` / `layer_attn_local_` / `layer_gdn_local_`、解析 prefill 流水线相位、
   决定 `md_xmx` / `md_int8`、逐设备上传权重（见 §11）。未给 `--layer-map` 时
   `be = cpu_mode ? make_cpu_backend() : make_gpu_backend(q)`（`engine.cpp:138-140`）。
4. **MTP 第二个门控**（`engine.cpp:141-149`）：`mtp_on` 需要 `multi_dev && md_xmx`，否则打印
   `[mtp] MTP needs a multi-device oneDNN int8 partition (...) - disabled` 并回落普通 decode。
   再校验 `--mtp-device` 在范围内（越界退回 0）。
5. **split 策略**（`engine.cpp:154-172`）：`cpu_mode` 强制 `n_splits = dec_splits = 1`；`multi_dev`
   默认也钉成 1，`PF_MD_SPLITS=1` 才选回 split 路径（MTP 草稿**不**受此门控，它有自己的
   `mtp_splits`）。
6. **权重上传**：`m.upload(q, /*host=*/cpu_mode || multi_dev)`——GPU 单设备拷整文件；CPU 只把
   `dev_ptr` 退化为 mmap 指针。
7. `PF_META` → `build_meta32`（默认关）；`PF_DP4A` → `pf8`、`pf8_dec`（默认都开，且 `pf8` 在
   `multi_dev` 下强制 false）。GPU 在 `pf8` 时 `m.build_w8(q)`（约 700 MB）；CPU 读 GGUF 整数
   block，**不建 w8**。
8. **decode split + 前缀缓存配置**（`engine.cpp:203-361`，同一块里顺序执行）：`PF_DEC_SPLIT` 覆盖，
   否则按设备 profile 推导 `dec_splits = max(8, ((warps_per_eu_x2*compute_units)/(2*n_head))/8*8)`，
   上限 `kMaxDecSplits`；`PF_PREFIX_CACHE`；算 `pc_state_floats`；由 `PF_PC_STATES` /
   `PF_PC_VRAM_MB` / `PF_PC_MEM_MB` 得 `pc_max_states`；RAM/磁盘预算；用 `kv_cap_mb` 收缩三层预算
   （先磁盘、再 RAM、最后 VRAM 检查点数）；`pc_max_states == 0` 关闭整个缓存；打开 RAM/磁盘 store。
9. **oneDNN**（`engine.cpp:368-465`）：单设备 `use_dnnl = pf8 && dnnl_gemm_enabled()`，为所有 SI8
   权重 `add_weight`，除 `PF_DNNL_NOWARM` 外 `warmup()`；多设备走另一条 `!md_int8 && !md_xmx`
   的分支，每 GPU 后端一份只装该设备层的 `dnnl_gemm`。
10. **尺寸与池预留**：`ffn_stride = 2*n_ff`（466），`max_blocks = ceil(max_seq/kBlockSize)`（467）；
    `pool_cap = max(n_blocks_, cap_blocks)`，前缀缓存开启时至少 `pc_max_states`（488-494）；
    `pool_initial = n_blocks_`；`PF_KV_GROW` → `pool_chunk`（默认 64）。
11. **关键**：`n_blocks = pool_cap`（506，指向预留而非已提交），因此 `kv_layer_stride` 与录制进图的
    每层基址在池增长时保持稳定。
12. `alloc_buffers()`（1678）→ `build_graphs()`（`engine_graph.cpp:1698`）→
    `release_host_weight_pages()`（`engine.cpp:1008-1073`，把 GGUF mmap 里 GPU 侧不再读的页丢掉；
    CPU 分区与其 tensor 保留，因为它们的 kernel 直接读 mmap）。

### 2.2 析构（`engine.cpp:514-660`）

1. `multi_dev` 时先 `prefill_flush()`（补完挂起的流水线 chunk）并 `sync_all()`，再动缓冲；
2. `pc_flush_to_disk()`（在池/检查点仍存活时）；
3. 释放所有设备分配（先逐个 `as_[dev]` 的 act_set，再单设备成员，避免双重 free）；
4. `pool_print`、`pc_print_stats`、磁盘/RAM 统计；
5. `kv_release_pool()`、`m.free_w8(q)`、每设备 `w8_dev_` 的 SIn 拷贝、`weight_maps_` 的设备权重、
   `m.dev_weights` 与 `h_logits`。

### 2.3 状态重置

* `zero_slot(slot)`（`engine.cpp:1911-1935`）：memset 一个序列的 GDN + conv 状态；多设备下状态在
  各自分区的 `as_[dev]` 里，索引用 `layer_gdn_local_[il]`，并走该设备的 queue。
* `reset_state()`（`engine.cpp:1936-1970`）：清零所有 slot 递归状态、memset `d_info`、**从
  `hp.rope_sections` 恢复 `mrope_sections`**（memset 会清掉这个模型常量）、清空所有 `pc_slot_`、
  释放 pending 检查点预留。
* `reset_single()`（`engine.cpp:1972-1984`）：先 `reset_state()`，再设 `n_rows=1, tpb=kMaxT,
  n_real=1, n_real_row[*]=1, pos[0]=0, slot[0]=0, active[0]=1, tokens[0]=0`。`eval`/`generate` 使用的
  单序列状态。

---

## 3. 缓冲区布局（`engine.cpp:1678-1909`）

行数 `R = kMaxB*kMaxT = 512`：

| 缓冲 | 大小 |
|---|---|
| `d_x`, `d_xnorm` | `R * n_embd` |
| `d_qkv` | `R * qkv_dim()`（`2*n_group*d_state + dt_rank*d_state`，**不是 `3*d_inner`**） |
| `d_z` | `R * d_inner` |
| `d_beta`, `d_alpha` | `R * dt_rank` |
| `d_conv_out` | `R * qkv_dim()` |
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

* `kv_layer_stride = n_blocks * kv_block_bytes()`（`n_blocks` 是预留；单设备在
  `engine.cpp:1717`，多设备在 `engine_kvpool.cpp:193`）。
* `d_gdn_state` / `d_conv_state` 按 **layer-major** 索引进分配（`[n_gdn][kMaxB][...]`），尽管头注释
  写作 `[kMaxB][n_gdn][...]`；实际索引是
  `state + layer_index*kMaxB*per`（`engine_graph.cpp:928-929`，单设备用
  `m.gdn_layer_index[il]`、多设备用 `layer_gdn_local_[il]`；`zero_slot` 里同样，
  `engine.cpp:1924-1930`）。多设备下这两块不进单设备成员，而是每个分区在 `as_[dev]` 里各有一份，
  状态**从不跨分区边界**，只交接隐状态（`engine.cpp:759-764`）。
* `d_info` 是 `sycl::malloc_host<step_info>`（host USM），清零；`d_info2_` 是多设备 prefill 流水线的
  第二份（同时在飞的 chunk）。
* 段数组：`d_segs_dec`（1024，**已废弃**：只分配和释放，无任何读写点，engine.cpp:1736 / 587）、
  `d_segs_pf`（4096）、`d_segs_pf8`（4096）、`d_segs_pf8_nh`、`d_segs_pfb`（`kMaxB * plan.segs`）、
  `d_segs_aux`（8）、`d_segs_vf`（verify plan）、`d_segs_dec8`、`d_segs_pf_slot[i]` /
  `d_segs_pf_nh_slot[i]`。
* SI8 scratch：`d_x8`、`d_xmeta`（float2）、`d_xsumq`，行数 `kMaxB*kMaxT`。
* decode 桶 `{1,2,4,8,16}` 各带自己的 `d_segs`（1024）。
* 前缀缓存检查点池 `d_pc_states[pc_max_states][pc_state_floats]` + 空闲/时间戳/owner 向量。
* MTP（`mtp_on`）另有一组 `d_mtp_*` 缓冲（`engine.cpp:1760-1891`），见 §12.4。
* 最后 `reset_state()`。

---

## 4. `seg_plan` 与 `build_plan`

### 4.1 `seg_plan`（`engine.h:31-113`）

一次完整 forward 的全部 GEMV/GEMM 工作：

* `segs` — 扁平的 `gemv_seg` 列表。
* `call_offsets[i]` / `call_counts[i]` — call `i` 在 `segs` 中的切片。
* `call_total_rows[i]`、`call_tb[i]`（token 块宽）、`call_nsb[i]`（K/256 split 提示）。
* `groups[]` — call 内连续同 type 段的 `{type, off, n, rows}` 合并。
* `call_group_begin/count[i]` — 每 call 在 `groups` 中的范围。
* `call_xq[i]` — DP4A 路径的激活量化描述 `{x, up, x_stride, up_stride, K}`。
* `layer_c0[il]` — 每层首个 call 的下标，末尾再附 head call 的下标；**plan 全局的 call 元数据按 call
  序索引，部分（按设备的）重放必须从这里起步**，见 §9.1 的不变量。
* `has_head` — 非最终 prefill 变体置 false，`record_forward` 因此不重放最后一个 call。

辅助：`begin_call(tb,nsb)`、`add(seg)`、`set_xq(...)`、`set_act_up(up_base, x_base)`、
`finalize()`。`set_act_up` 会就地给**当前 call 新加的**段打上 `act_up`（并保留每个切片的 x 偏移），
因为 `ffn_down` 的激活是 `silu(gate)*up`：量化路径从 `call_xq.up` 施加，fp32 GEMV 和 oneDNN 回落
路径从段的 `act_up` 施加。`finalize()` 对每个 call 的段按 `(dev, type)` 稳定排序，再把连续同 type
合并成 group 并记录每 call 的 group 范围。

### 4.2 `build_plan(T, tb, head_batched, use_w8, with_head)`（`engine_graph.cpp:67-313`）

* `n_slices = use_w8 ? 1 : ceil(T/tb)`。fp32 路径把每个段复制到 `tb` token 切片；w8 路径是整块单段。
* `add8` 构造 w8/int8 段：`type = w.type`（真实 ggml 类型，与 `mk` 相同——`w8` 是独立字段
  `w8t`，`s0.w8.vals` 才决定走不走 int8 路径），`c.w = w8v.vals ? w8v.vals : wkey(dev, w.data)`，
  `x8=d_x8`、`xmeta=d_xmeta`、`xsumq=d_xsumq`；`multi_dev` 时先查该设备自己的 `w8_dev_` 副本。
  `c.i8` 只对 CPU 分区（单设备 CPU，或多设备的 CPU 后端）置位——它让 CPU 的整数 kernel 直接读
  GGUF block，不建 w8。
* `mk` 构造 fp32 段：`w = wkey(dev, w.data)`、`meta32 = meta32_of(w.data)`。
* `wkey`（lambda，`engine_graph.cpp:101-111`）与 `engine::wkey`（`engine.h:177-187`）同义：
  已上传到该分区的用设备指针，否则用 host 指针——**后者正是该张量的 oneDNN key**，因为转换器跳过
  了每个被转换张量的原始上传。

每层固定四个 call（`engine_graph.cpp:175-245`，逐层先 `bind_acts(dev)` 把 plan 快照到该设备的
缓冲，并把 `layer_c0.push_back(call_tb.size())`）：

**GDN 层**

| call | 内容 |
|---|---|
| 1 | w8: `wqkv8→d_qkv`、`wgate8→d_z`，`set_xq(d_xnorm)`；另加 fp32 `ssm_beta→d_beta`、`ssm_alpha→d_alpha`。四个投影共享一次激活量化 |
| 2 | `ssm_out8`/`ssm_out`，residual `d_x→d_x` |
| 3 | `ffn_gate` + `ffn_up`（up 在 `d_ffn+n_ff`） |
| 4 | `ffn_down`；两条分支都设 `set_xq(d_ffn, up=d_ffn+n_ff, K=n_ff)`（w8，量化时应用 SiLU）**并**
   `set_act_up(d_ffn+n_ff, d_ffn)`（供 fp32/oneDNN 回落自己施加）；fp32 段直接 `s.act_up = d_ffn+n_ff` |

**attention 层**

| call | 内容 |
|---|---|
| 1 | `wq/wk/wv` → `d_qbuf/d_kbuf/d_vbuf` |
| 2 | `wo` → `d_x`（residual `d_x`） |
| 3-4 | 与 GDN 相同的 FFN call |

**head**（`engine_graph.cpp:249-311`，仅当 `with_head || head_batched`）：先 `layer_c0.push_back(...)`
再 `bind_acts(0)`，然后 `begin_call(head_batched ? tb : 1, n_embd/256)`，**`m.output`** GEMV（不是
`m.tok_embd`：27B 的 LM head 是独立的 `output.weight`）；`s.w = wkey(0, m.output.data)`，
`x = head_batched ? d_xnorm : d_last_hidden`，`out = d_logits`。两条 int8 入口各自注册 `set_xq`：
仅 batched decode（`use_w8 && head_batched && output8.vals`，即 head 有 SIn 副本时，282-300），以及
`md_xmx` 时的兜底（307-309，**缺了这条，head 在多设备上拿不到 `dnnl_call` 分支**，因为该分支要求
call 有 xq 描述）。

⚠️ **head 段必须在 `bind_acts(0)` 之后构建**（`build_plan` 里确实这么做了）。`build_plan` 的逐层循环对每层
调用 `bind_acts(layer_dev_[il])`，把 `d_xnorm`/`d_x8`/`d_last_hidden` 等成员指向**该层所在设备**
的缓冲；循环结束后如果没有重新绑回 primary，head 段就会捕获**最后一层设备**的缓冲，而运行时
`record_forward` 在 `bind_acts(0)` 之后把 rmsnorm 结果写进 **primary** 的 `d_xnorm`——于是 batch-1 decode
的 head 读到一块本 step 从未被写入的缓冲，logits（进而采样）全错，而 prefill 走 `run_head()`（读
`d_last_hidden`）却完全正常。症状是"decode 与前向不一致、只吐 1–2 个 token、且对所有内核开关免疫"。

因此每个 plan 有 `n_layer*4` 个 call，加可选的一个 head call。

### 4.3 已录制的 plan 矩阵（`build_graphs`）

| plan | 形状 | 用途 |
|---|---|---|
| `plan_pf_` | `build_plan(kMaxT, 8, false)` | fp32 prefill，1 行 × 32 token，8-token 切片 |
| `buckets_[b].plan` | `build_plan(b.tb, b.tb, true)` | decode，`T=tb`、头批处理、无 w8（多设备 `build_plans` 里 `use_w8 = multi_dev \|\| md_int8`） |
| `plan_pf8_` | `build_plan(kMaxT, kMaxT, false, true, true)` | w8 prefill 带 head |
| `plan_pf8_nh` | `build_plan(kMaxT, kMaxT, false, true, false)` | w8 prefill 不带 head（非最终 chunk） |
| `plan_dec8_` | `build_plan(1, 1, true, true, true)` | w8 batch-1 decode |
| `plan_pfb_` | `build_plan(kMaxT, kMaxT, false, true, true)` | 共享的 w8 chunk-batched plan |
| `plan_pf_slot[i]` / `plan_pf_nh_slot[i]` | `build_plan((i+1)*8, 8, false[, …,false])` | CPU / 多设备 host 重放用的 prefill 分档（带/不带 head） |
| `plan_vf_` | `build_plan(kMaxT, kMaxT, /*head_batched=*/true, true, true)` | MTP verify：mode 2 的 `k+1` 行 batch + **逐行** head（见 §12.5） |
| `plan_mtp_` | 手工构造，5 个 call | MTP 层自己的 4 个 call + 共享 LM head（见 §12.3） |

`head_batched=true`（decode）写 `d_logits[B][vocab]` 并读 `d_xnorm`；`false`（prefill）写一行、读
`d_last_hidden`（由 `copy_row` 产生）。

**plan 是"指针快照"**：`build_plan` 在构建时把当时的 `d_x*` 成员指针写进 `gemv_seg`。multi-device 下
`bind_acts(dev)` 会把这些成员切到各设备的缓冲（`as_[dev]`），因此**一个 plan 的段指针全部属于构建它时
最后绑定的那个设备**；`bind_acts()` 不会回头修改已构建的 plan。凡是"固定在 primary 上跑"的东西（目前
只有 head，`s.dev = 0`）都必须在构建其段之前 `bind_acts(0)`，否则会拿错设备的激活缓冲。

---

## 5. `record_forward`

`record_forward(mode, plan, d_segs, rows, d_segs_rows, at_nsp_hint, ph, info)`（`engine_graph.cpp:443`）
执行（或录制）一次完整 forward。`ph != nullptr` 表示多设备的分相录制（一段连续的同设备层区间），
`info` 是该 chunk 自己的 `step_info`（多设备 prefill 流水线有两份）。网格范围由
`engine_graph.cpp:835-839` 决定：

| mode | 含义 | `T` | `nrows` | `nreal` | `NCH` |
|---|---|---|---|---|---|
| 0 | decode | `rows`（批大小） | `rows` | 1 | 1 |
| 1 | chunked prefill（1 行 × ≤32 token） | `kMaxT` | 1 | `rows` | 1 |
| 2 | chunk-batched prefill（`rows` 个 token = NCH 个 chunk） | `rows` | `NCH` | `kMaxT` | `ceil(rows/kMaxT)` |

`step_info`（`kernels.h:76-119`）是唯一的逐步状态载体，**所有字段都在 kernel 体内读**：

| 字段 | 含义 |
|---|---|
| `n_rows` / `n_real` / `tpb` | 本步的行数、每行 token 数、行内 token 槽位（行 `r` 的第 `t` 个 token 在 `r*tpb+t`） |
| `pos[kMaxB]` / `slot[kMaxB]` / `active[kMaxB]` | 每行的首位置、状态 slot + KV block 表行、是否参与 |
| `n_real_row[kMaxB]` | mode 2 每行的**实际** token 数（最后一行可不足 `kMaxT`；`0` 表示用 `n_real`） |
| `tokens[kMaxB*kMaxT]` | 每行每槽的 token id |
| `pc_active` / `pc_stride` / `pc_base` / `pc_row_slot[kPcMapLen]` | 前缀缓存检查点：仅在被跟踪的 prefill 期间为 1；`pc_row_slot[b]` 是第 `b` 个完整块边界（= token `b*32` 之后）的检查点槽 |
| `mtp_dt` / `mtp_dry` | MTP：逐 token 快照（而非 32-token 边界）/ dry verify（算但不写回递归状态） |
| `mrope_on` / `mrope_sections[4]` / `mrope[4*kMaxB*kMaxT]` / `img_embd` / `img_row[]` | 多模态：4-section 位置（section-major）、图像 embedding 行映射 |

### 5.1 每个 plan call（`gemv_at`，`engine_graph.cpp:506-817`）

1. `tb = call_tb[idx]`，`single = (tb==1)`，`nb = single?1:NCH`，`tbm = (mode==2 && !single) ? rows : tb`。
2. **oneDNN 判定**（`dnnl_call`，542-582）：要 `dnnl_for(cur_dev)` 存在、**不是**"单设备 decode"
   （`mode == 0 && !multi_dev` 保留 dp4a GEMV；多设备 decode 也走 oneDNN，因为它的 M=1 路径就是
   `i8_row_gemv`）、call 有 xq 描述，且该 call 的每个需要转换的段都有匹配的权重（`has_weight` /
   `_w4` / `_k5` / `_cb4`）且 `K` 与 xq 一致。**single-token 的 call 也在覆盖范围内**——早先把
   `single` 排除在外，把 prefill 的 LM head 送进了 fp32 反量化 `gemv_group`（10.1 vs 3.3 ms），
   而那时 head 的原始设备拷贝已被跳过，回落路径会在设备上解引用一个 host 指针。
3. **激活量化**：oneDNN → `D->quantize(..., do_split = (mode == 0 || inf->mtp_dry != 0))`
   （589-590）——decode 与 dry verify 才产 even/odd 平面，原生存储的 GEMV 要读它；否则
   `call_xq[idx].x` 存在时 `cur_be->xq`（模式 2 一次覆盖 `tbm` token，否则逐行 `r` 用 `TB=tb`）。
4. **段分发**（按 `s0.i8` → `s0.w8.vals` → `multi_dev && dnnl_call` → fp32 的顺序）：
   * `i8`（CPU 分区）：mode≠0 且非 single 时逐段 `i8_gemm(sj, tbm)`，否则逐行 `i8_gemv` /
     `i8_gemm`（624-642）。
   * w8 且 `mode != 0 && !single` → 每段一次 `dp4a_gemm`（或 oneDNN `gemm`；gemm 返回 false 时补一次
     `xq` 再落 dp4a，643-662）。
   * 否则逐行 `r`：`tb==1` 用 `dp4a_gemv`，否则 `dp4a_gemm(..., tb)`（663-676）。
   * 多设备 GPU 组（无 SIn 视图）：mode 0 且激活就绪时整组一次 `i8_row_gemv_multi`；否则
     `tbm <= 13 && gr.n > 1` 的**窄 int8 组**走 `gemm_i8_group` 融合（GDN 的 `ssm_alpha`/`ssm_beta`
     两个 48 行段，709-742，`PF_NOFUSE=1` 关）；再否则逐段 `gemm_w4` / `gemm`，全失败才回落
     `gemv_group`（677-785）。
   * fp32 模式 2 → 每 group 一次 `gemv_group_launch(..., tb, nsb, NCH)`（一次 dispatch 覆盖所有 chunk 行）；
   * fp32 其他 → 每行一次 `gemv_group_launch`（786-797）。

`gemv()` 调用 `gemv_at(ci++)`，因此 call 顺序严格等于 plan 顺序。

### 5.2 kernel 序列

```
embed(tok_embd → d_x)
for il in 0..n_layer-1:
    rmsnorm(d_x, attn_norm → d_xnorm)
    if GDN:
        gemv()                                   # call1
        [conv_l2 + conv_state_update + gdn] 或融合路径
        mtp_capture(...)                         # 仅 inf->mtp_dt（见 §12.2）
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
mtp_capture(d_x → d_mtp_main_h)                 # 仅 mtp_on：output_norm **之前**的隐状态（§12.2）
if mode != 0: copy_row(d_xnorm → d_last_hidden, row=-1)
gemv_at(call_offsets.size()-1)                   # LM head（固定在 primary 设备）
```

`rmsnorm`/`copy_row`/head 都在 `cur_be = backend()`（primary）上执行，且在此之前的 `bind_acts(0)` 把成员
指针切回 primary；head 段的指针也必须来自那次绑定（见 §4.2 的 ⚠️）。decode（`mode == 0`）不写
`d_last_hidden`，所以 head 段读的是 `d_xnorm`（`head_batched=true`）。

`record_forward` 的 call 顺序必须与 `build_plan` 同步。`ci` 的起点由 `ph` 决定（非分相时为 0），
见 §9.1。

### 5.3 GDN 融合路径

* 每个 GDN 层的 `pc_snap` 切片在录制时计算好，供 conv/GDN kernel 写检查点；`inf->mtp_dt` 置位时改
  指向该设备的 `d_mtp_hist_`（每个 token 一个槽，`layer_off = gl*(gdn_per+conv_per)`，
  `engine_graph.cpp:948-958`）。
* 模式 2 且 `PF_GDN_FUSE>=1`（默认 2）时，整个批次一次性物化：`conv_l2(cross_row=true, NCH, kMaxT)` →
  `conv_state_update(last_row_only=true)` → 完全融合的 `gdn_launch`（`fuse_gdn>=2`，以
  `n_rows=1, row0=0, tpb_arg=T, nreal_arg=T` 一次走完整批）或逐行 GDN（`fuse_gdn==1`）。这把每行
  launch 链从 48 降到 4。
* ⚠️ **`nreal_arg` 是必须的**：两个 GPU kernel 现在都用
  `n_real = nreal_arg > 0 ? nreal_arg : row_nr(info, rr)`（`gdn.cpp:59`、`gdn.cpp:181`）。融合的
  mode-2 调用传一行 `T` 个 token，而 kernel 若用 `row_nr(info, 0) = kMaxT`，就**只有第 0 个 chunk 行**
  拿到递推、之后每行读陈旧状态——prompt ≥ 64 token 时 logits 错（63 token 还对，因为调度器那时
  对 `rem < 2*kMaxT` 仍走 mode 1），且单设备与多设备表现不同（CPU 的 `cpu_gdn` 一直尊重
  `nreal_arg`）。这也是 `PF_PFB_MAX_M`（曾默认把多设备 batch 压到 `kMaxT`）被移除的原因：它只是把
  这个 bug 藏起来，代价是 27B/2x A770 上 pp512 91 vs ~660 t/s。
* `PF_GDN_FUSE=0` 的逐行 mode-2 循环过不了图录制（重复的相同 kernel + USM 参数不会被逐节点重放），
  仅作诊断。
* 否则走经典逐行循环。`gated_norm` 无行序依赖，所有 chunk 行一次 dispatch。

### 5.4 attention split 选择

* `PF_ATTN_SPLIT` 强制 split 数（三种模式都生效，clamp 到各自的 `n_splits` / `dec_splits`）。
* 模式 0 → `dec_splits`（`PF_DEC_SPLIT` 覆盖；否则由设备 profile 推导
  `(warps_per_eu_x2/2)*compute_units/n_head`，向下取 8 的倍数、下限 8、上限 `kMaxDecSplits = 512`；
  27B（`n_head = 24`）在 512-EU A770 上是 160。**旧默认是 `kMaxSplits = 64`**，已废）。
* 模式 2 → `n_splits`，或录制变体的 `at_nsp_hint`，否则
  `ceil(max_nkv / PF_ATTN_SPLIT_KEYS)`（默认 512），clamp 到 `[1, n_splits]`。
* 模式 1 → `n_splits`。
* `at_fused = (nsp==1 && PF_ATTN_FUSE!=0)` 时 attention 直接写输出并跳过 `attn_combine`。

模式 2 那条 `ceil(max_nkv / …)` 需要在主机上读 `d_info->pos[r]`（1123-1128），所以它**只在
`at_nsp_hint == 0` 的直接路径**发生——录制变体必须带 hint（见 §6.4）。

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

`buckets_` 对应 `tb ∈ {1,2,4,8,16}`（`engine.cpp:1892`）。批式 decode 选最小的 `tb >= n_rows` 的桶，
否则最后一个；超出 `n_rows` 的行标 `active=0`，kernel 跳过。batch-1 的 int8 路径（单设备
`e_dec8`、多设备 `plan_dec8_` + 逐分区图）在选桶之前就短路了（§7.1）。

### 6.3 prefill 变体

| 图 | 条件 | 说明 |
|---|---|---|
| `e_pf` | 单设备 GPU，**无条件** | fp32 prefill（`build_plan(kMaxT, 8, false)`） |
| `e_pf8` | `pf8` | w8 prefill 带 head（最终 chunk） |
| `e_pf8_nh` | `pf8` | w8 prefill 不带 head（非最终 chunk） |
| `e_dec8` | `pf8_dec` | w8 batch-1 decode |
| `buckets_[b].e` | 单设备 GPU | 每个 `tb ∈ {1,2,4,8,16}` 一张 decode 图 |
| `pfb_vars_` | `pf8 && !use_dnnl` | `ntok ∈ {2,4,8,16}*kMaxT` 的 chunk-batched 图，每个尺寸一张；`use_dnnl` 时只登记 `ntok` 让 `batched_prefill_fit` 能选中同样的尺寸（oneDNN 不能录制），`batched_prefill_fit` 允许任意 `kMaxT` 倍数直到 `kMaxB*kMaxT`；DP4A 时只能选已录制尺寸 |

`cpu_mode || multi_dev` 时 `build_graphs` 只 `build_plans()`（不录任何图）；多设备另外走
`build_md_dec_graphs()`（§9.1），MTP 开启时再走 `build_md_verify_graphs()`（§12.5）。

### 6.4 图冻结与 `step_info` 不变量

command graph 记录的是 kernel 命令列表；录制时按值传入的主机标量（`rows`、`T`、`tb`、`nsp`、`mode`、
指针运算、`gemv_seg` 内容、env 派生的常量如 `at_split`、`fuse_gdn`）都被烘焙进可执行图。因此：

1. **所有逐步状态必须放进 host-USM 的 `step_info`，在 kernel 体内读取**，绝不能在图录制时于主机读取。
   图只保存 `d_info` 指针，kernel 在重放时看到当前内容。
2. `record_forward` 中对 `d_info` 的主机读取（模式 2 从 `d_info->pos[r]` 推导 split）只在
   `at_nsp_hint == 0`（直接模式 2 路径）时发生，绝不在录制变体里。
3. 同理还有 `inf->mtp_dry`：它决定激活量化是否产 even/odd 平面（§5.1 第 3 步），是一个主机标量。
   MTP verify 的图在 `mtp_dry = 1` 下录制（`engine_graph.cpp:1618`），所以重放保持 `do_split`。
   `attn_xmx_launch` 同理是按 key 数在主机侧选的，所以 verify 的图在超过 `xmx_min_keys` 后就被弃用
   （§12.5）。
4. 通过函数内 `static` 读取的 env 开关在首次使用时锁定，不会每次重放重读。

---

## 7. 入口点

### 7.1 批处理 API（调用者持有 `engine::mtx`）

* `prefill_chunk(toks, start, n, slot, with_head=true)`（`engine.cpp:1986-2042`）：
  先补完挂起的流水线 chunk（`prefill_flush` + `sync_all`）并清 `pf_info_`，再
  `pc_capture_begin` → 填 `d_info`（`n_rows=1, tpb=kMaxT, n_real=n, n_real_row[*]=n,
  pos[0]=start, slot, active, tokens`）→ 分三条路：`cpu_mode || multi_dev` 直接
  `record_forward(1, plan_pf8_ / plan_pf8_nh_ / plan_pf_slot[si], …)`（fp32 时按
  `round_up(n, kPfSlice=8)` 选最小的一份），`PF_NOGRAPH&&pf8` 直接 `record_forward`，否则重放
  `e_pf8`/`e_pf8_nh`/`e_pf` → `pc_active=0` → `sync_all()`。
* `prefill_text(toks, slot, n)`（`engine.cpp:2050-2064`）：**文本 prompt 的首选入口**，循环取
  `batched_prefill_fit(rem)`（CPU 恒为 0）喂 `prefill_batch`，只有 `< kMaxT` 的尾巴（或无可用
  批量变体时）落回 `prefill_chunk`。
* `prefill_batch(toks, start, n, slot, pos0)`（`engine.cpp:2123-2196`）：mode 2 的 chunk-batched
  prefill，`NCH = ceil(n/kMaxT)`，**最后一行可以不足 `kMaxT`**（写在 `n_real_row` 里）；`cpu_mode`
  直接 throw。三条路：`multi_dev && pf_pipe_ok_` 走三相位流水线（`pf_pipe_finish(false)` 让上一个
  chunk 的 device-1 相位落地，再用 `d_info`/`d_info2_` 交替录 device-0 相位并置
  `pf_pipe_pending_`），`nog || use_dnnl || multi_dev` 直接 `record_forward(2, plan_pfb_, …)`，
  否则重放 `ntok == n` 的 `pfb_variant` 图（没有就 throw）。
* `decode_batch(tokens, poss, slots, n_rows)`（`engine.cpp:2198-2254`）：先 `prefill_flush()`；
  填 `n_rows/n_real/tpb=1`、`pc_active=0` 与每行 `pos/slot/active/tokens/n_real_row=1`；然后依次
  尝试：单设备 `pf8_dec && n_rows==1` → `e_dec8`；CPU `pf8_dec && n_rows==1` → `plan_dec8_` 直放；
  多设备 int8 且 `n_rows==1` → `replay_md_dec_graphs()`（没有图则直放）；否则选最小的 `tb >= n_rows`
  的桶并直放（CPU/多设备）或重放它的图。直放路径统一 `sync_all()`。
* `fetch_logits(row, out)`：`d_logits[row]` → `h_logits` → `out`。

**部分（不足 `kMaxT` 的）mode-2 batch 需要 oneDNN 权重路径**：`batched_prefill_fit`
（`engine.h:365-413`）只在 `use_dnnl || (multi_dev && dnnl_any_dev())` 时允许最后一行
`n_real_row < kMaxT`；`md_int8` 回落的 dp4a GEMM（及其 per-device w8/x8 布局）只对完整
`kMaxT` 行的 batch 正确，让它吃部分 batch 会**静默破坏**隐状态（2 GPU md_int8 上的
decode-vs-prefill 不匹配即由此而来）。`PF_NO_PFB_PARTIAL=1` / `PF_PFB_MAX_M=N` 是 A/B 开关
（后者默认 0 = 无上限）。注意 `multi_dev` 下 `pf8 == false`，单设备 SIn w8 整模型不建，两条 int8 走
各自的分区机制（§11）。

### 7.2 无图诊断/回退

`forward_plain_pf`、`forward_plain_pf8`、`forward_plain_dec(rows)` 直接调用对应 `record_forward`。

### 7.3 单序列 API

* `eval(tokens)`（`engine.cpp:2323-2346`）：加锁、`reset_single`、按 `kBlockSize` 分配块、
  `set_table`、**`prefill_text`**、`run_head`、释放块、返回最后一个 token 的 logits。无前缀缓存、
  无采样。
* `run_head()`（`engine.cpp:2278-2321`）：先 `prefill_flush()`。优先走 **primary 设备的 int8 head**：
  `wkey(0, m.output.data)`（已转换的 head 用 host key，未转换用上传后的设备指针）→ `D->quantize`
  → `D->gemm_w4` / `D->gemm`（`d_last_hidden → d_logits`）→ 拷回主机。只有在根本没有 oneDNN head
  时才回落到构造 aux `gemv_seg` + `gemv_group_launch` 的 fp32 路径（此时 `wptr` 是有效的，因为未转换的
  head 一定被上传了）。GGUF 无 `output.weight` 时加载器把 `m.output` 指向 `tok_embd`（tied head）。
  注意它会重算最终 prefill chunk 已由图写好的 head。
* `generate(prompt, gp, cb, first_logits)` / `generate_mm(p, gp, cb, first_logits)`：加锁后调
  `generate_impl`。
* `generate_impl`（`engine.cpp:2360-2500`）：
  0. **MTP 分流**：`mtp_on && mm == nullptr && (temperature <= 0 || top_k == 1)` → 直接
     `generate_mtp`（§12）。只对 greedy 成立：接受判据是"目标自己的下一个 token"的相等比较，
     采样目标需要 rejection sampling 才精确；多模态 prompt 也不走。
  1. `reset_single`；seed `mt19937_64`（`gp.seed` 或 `random_device`）。
  2. 多模态：校验 `embd` 行数 ≤ `kMaxImgTokens`；若无设备指针则拷贝到 `d_img_embd`。
  3. 分配 prompt 的块并 `set_table(0, blocks)`。
  4. **prefill**：多模态按 ≤32 token 分块设 `mrope_on`/`img_embd`/`img_row` 后
     `prefill_chunk`（**每个 chunk 都带 head**，因为没有 head 的变体只在调度器路径用）；纯文本走
     `prefill_text`。
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
| `PF_META` / `PF_SI4` / `PF_W4` | 权重旁路数组 / 4-bit 重化 / 原生 4-bit（u4）权重路径 |
| `PF_CB4` / `PF_W4_ALL` / `PF_W4_RB` | IQ4_XS/IQ4_NL 码本 4-bit（默认开）/ 全类型重化到 u4（有损）/ 解码 GEMV 的 rows-per-workgroup（默认 16） |
| `PF_K5` / `PF_K5_NOCORR` | Q5_K 原生 5-bit（nibble + 第 5 位平面，默认开）/ 预填充去掉修正 epilogue（诊断二分，会算错） |
| `PF_ATTN_SPLIT` / `PF_ATTN_SPLIT_KEYS` / `PF_ATTN_FUSE` / `PF_DEC_SPLIT` / `PF_DEC_GROUP` / `PF_ATTN_VEC` | attention |
| `PF_GDN_COLS` / `PF_GDN_WG` / `PF_GDN_VEC` / `PF_GDN_FUSE` / `PF_GDN_DBG` | GDN |
| `PF_GEMV_*` / `PF_GEMM_*` / `PF_MT_*` / `GEMV_*` | GEMV/GEMM 调优 |
| `PF_NOGRAPH` / `PF_PROF` / `PF_PROF_ALL` / `PF_TIME` / `PF_HOSTPROF` / `PF_DBG_*` / `PF_ROWACT` | 诊断/回退（`PF_PROF_ALL` dump 全部 call group；`PF_ROWACT` 恢复已死的按行激活量化器；`PF_HOSTPROF` 是**不带**每戳设备等待的 `PF_PROF` 分桶，因此量的是提交开销——这正是找到 `wg_clamped` 回归的工具，见 §13） |
| `PF_MD_SPLITS` / `PF_MD_GRAPH_DEV` | 多设备：重新选入 attention K-split 路径（默认关）/ decode command graph 只给设备 N 录（`99` = 不录） |
| `PF_KV_TYPE` / `PF_KV_F32` / `PF_KV_BF16` / `PF_KV_CAP_MB` / `PF_KV_GROW` | KV |
| `PF_PREFIX_CACHE` / `PF_PC_*` | 前缀缓存 |
| `PF_MTP*` / `PF_MTP_DEV` / `PF_MTP_FORCE_INT8` | MTP / NextN 投机解码，见 §12.9（总表以 `AGENTS.md#environment-variables` 为准） |
| `PF_PF_PIPE=0` / `PF_NO_PFB_PARTIAL` / `PF_PFB_MAX_M` | 多设备 prefill 三相位流水线开关 / mode-2 只收 `kMaxT` 倍数（A/B）/ 可选 batch 上限（`0` = 无上限，默认） |

---

## 9.1 多设备 decode 的 command graph

多设备路径直接重放 `record_forward`，单 token decode 每步约 700 次 kernel 提交（实测约占 62 ms
一步里的 9 ms）。`build_md_dec_graphs`（`engine_graph.cpp:1427-1505`）按 layer 循环里**连续的设备
段**各录一张图（0-31/32-63 映射下 3 张：`[dev0 层+embed]`、`[dev1 层]`、`[dev0 out_norm+head]`），
回放时用现有的 host staging handoff 串起来（`replay_md_dec_graphs`，1507-1525）。前置条件：
`multi_dev && (md_int8 || md_xmx) && d_segs_dec8 && !PF_NOGRAPH`，且**所有相位的设备都必须是 GPU
分区**——含 CPU 分区的混合映射直接跳过整组录制并打印一行（CPU 分区跑的是主机代码，不是队列上的
kernel）。`PF_MD_GRAPH_DEV=N` 只给设备 N 录（`99` = 都不录，用于 A/B），`PF_NOGRAPH` 关闭。

**关键不变量**：`ci`（call 游标）索引 plan **全局**的 `call_tb`/`call_xq`/`call_group_*`，所以一个
phase 必须从它首层的 call 下标开始——用 `seg_plan::layer_c0`（层 → 首个 call，末尾附 head 的 call；
`engine_graph.cpp:883-887`）。从 0 重新计数会让后面的分区读到**第 0 层**的 call 元数据（激活指针、
K）而执行自己的 segment：`dnnl_call` 变假、整组退回 fp32 `gemv_group`、层静默不写（一个重复
token，慢 6 倍）。

实测收益只有 ~1-2 ms/token，说明**逐 kernel 提交开销不是层 GEMV 效率差距的主因**。

## 9.2 LM head 与 oneDNN（`wkey` / `dnnl_call`）

head 是固定在 backend 0 的全局张量，按 **untied / tied** 两条路走（`setup_md_dnnl`，
`engine.cpp:1258-1272`、`1661-1672`）：

* **untied**（`m.output.data != m.tok_embd.data`，27B）：在 `upload_device_weights` **之前**就
  `add(m.output)`，key 是 **host 指针**；转换器的存在让上传跳过那份 ~1.0 GB 的原始设备拷贝。
* **tied**（0.8B，共享 `tok_embd` 的存储）：原始 GGUF 行必须留着给 embed kernel，所以转换在
  **上传之后**做，key 是**上传后的设备指针**（`wkey(0, output.data)` 正好解析到它）。

两条路共同要求：`build_plan` 用 `wkey()`（不是 `wptr()`）给 head 段取权重，plan 的 head call 有
`xq` 描述（§4.2），且 `gemv_at` 的 oneDNN 分支覆盖 single-token call。`record_forward` 末尾还要把
`cur_dev` 重置为 0（`engine_graph.cpp:1225`）——层循环之后 `cur_dev` 是**最后一层**的设备，它的
oneDNN 表没有 primary 的 head 条目，会把 head 静默降级到 fp32 反量化 GEMV。漏掉任一项就是
**12.3 vs 3.4 ms/token**（27B、2x A770 实测）。

`run_head()` 也优先走 oneDNN（§7.3），因此 prefill head 与 decode head 数值一致
（pp512 1420 → ~1520 t/s）。

**head 的原始 GGUF 拷贝（dev0 上 ~1.0 GB）现在只对 tied head 存在**（embed kernel 要读），untied
head 已经不再上传它。`d_segs_aux` 的 fp32 回落仍然保留，但只在"根本没有 oneDNN head"时才走到；
那条路上 `wptr()` 有效（未转换 → 已上传），所以不能简单把回落删掉。

## 9.3 codebook prefill 的 scratch 与 embed 的 host USM

* codebook（§10 of 02-quantization.md）在 prefill 时把索引展开成 int8 到 `dnnl_gemm` 的**复用
  scratch**（按最大单张量 ~89 MB），再跑已有 int8 primitive。oneDNN 每次 execute 重读权重 memory
  （已验证），所以不需要持久 int8 拷贝——显存反而省 2.46 GB/card。
* `upload_device_weights` 把 `tok_embd` 用 `sycl::malloc_host` 放在 host USM（每 token 只读一行
  20 KB，PCIe ~3.5 us），省 0.72 GB/card；共享 context 内设备可直接访问。

## 10. 已知细节与注意点

* `pf8_dec` 默认 on（单设备 GPU 与 CPU 都走 int8 decode，除非 `PF_DP4A_DEC=0`）。**多设备下
  `pf8 == false`**（`engine.cpp:190`、`1547`），decode 的 int8 走分区机制（`w8_dev_` 或
  `dnnl_dev_`），不是这两个开关。
* 无 head 的 prefill 变体（`plan_pf8_nh` / `plan_pf_nh_slot`）由 `seg_plan::has_head=false` 标记，
  `record_forward` 只在 `plan.has_head` 时重放最后一个 call，因此不会重跑 `ffn_down`。
* `run_head` 会重复最终 chunk 已经写好的 head；`d_segs_pfb` 的行偏移副本 / `row_offset_seg` 在当前
  分发路由下处于休眠状态（模式 2 的 fp32 用 token-block 网格，w8/i8 用连续 `tbm=rows`）。
* **跨序列 prefill batching 未启用**：把多个 prompt 塞进同一次 mode-2 forward 会产出垃圾（不是
  另一个合理续写），而 `step_info` 本身已带 per-row 的 `slot`/`pos`/`n_real_row`/`active`，所以故障在
  某个读 plan 期状态的地方。prefill 因此一次只跑一条序列（每条 ~300 ms）；多请求的收益全部来自
  decode batching。修好它需要"同一批 N 个请求必须逐字节相同"的对照 harness，并先把 decode batch
  钉成 1 以隔离变量。
* 调度器 `loop` 在引擎调用期间**释放序列互斥量 `m`**（`scheduler.cpp:157`、`234`、`359` 的
  `lk2.unlock()`），否则 `submit()` 被饿死、请求只能一条接一条跑完。`e.mtx` 仍然串行化引擎本身。

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

* `--layer-map 0-11:gpu,12-23:cpu`：`setup_multi_device` 解析**闭区间**、校验无缝隙覆盖
  `[0,n_layer)`，建立 `backends_`（GPU 优先作为全局张量/LM head 的主设备）与 `layer_dev_`。
* `build_plan` 用 `wkey(dev, host)` 为每层解析该张量的权重句柄（设备拷贝，或 host 指针=oneDNN
  key）；`record_forward` 用 `cur_be = &be_of(il)` 逐层切换后端，并在层边界 `handoff_x(prev, dev,
  nrows*nreal)`。**交接确实要拷贝**：激活在 host USM（`host_act = true`），所以
  `handoff_x`（`engine.cpp:796-830`）先 `qf.wait()` 再 `D2H → host staging → H2D`，并只为**活着的
  行**拷（`nrows * nreal`，不是整块 `kMaxB*kMaxT`——旧实现每 token 两次交接白搬 10.5 MB）。
  层循环之后还有一次把最后分区的隐状态交回 primary 的 `handoff_x`（`engine_graph.cpp:1217-1226`）。
* `upload_device_weights(dev)`（`engine.cpp:866-995`）只上传**该分区层**的张量 + 落在 dev0 的全局
  张量（`tok_embd`/`output`/`output_norm`；`mtp_on` 时还有 MTP 层的 F32 norm 与线性），因此 GPU 只
  持有自己那份权重而不是整份 GGUF。两条例外：`tok_embd` 放在 **host USM**（每 token 只读一行、
  PCIe ~3.5 us，省 ~0.72 GB/card），以及已被 `w8_dev_` / `dnnl_dev_`（含 u4/k5/cb4/w2 副本）转换过
  的张量——它们跳过原始设备拷贝。
* `layer_attn_local_[il]` 给出该层在**所属设备**注意力层里的序号；`kv_setup` 为每个设备分配只含
  其注意力层的 paged KV 池，block id 全局一致（同一张 block table，见 [05-kv-cache.md](05-kv-cache.md)）。
* `bind_acts(dev)` 把引擎的激活成员（`d_x`/`d_xnorm`/`d_x8`/…）切到该设备的 `as_[dev]` 缓冲；
  plan 在构建时把这些成员"快照"进 `gemv_seg`，而 `bind_acts` 不会回头改已构建的 plan。因此固定在
  primary 上执行的段（LM head）必须在其段被构建前 `bind_acts(0)`，否则会捕获最后一层所在设备的缓冲
  （见 §4.2 的 ⚠️，这是 27B decode 出错的原因）。
* 多设备支持前缀缓存：block id 全局、递归状态检查点在共享主机 USM 中按全局 GDN 层序号索引；
  `pc_serialize_block`/`pc_deserialize_block` 经 `engine::kv_layer_ptrs` 把每个全局注意力层解析到所属
  设备的池（局部序号 `layer_attn_local_[il]`，块内偏移与单设备布局一致）。整模型 `pf8`（SIn w8 预留）
  不建，但两条 int8 都可用：
  - **prefill（按块）**：GPU 分区走 oneDNN int8（`dnnl_dev_`，每 GPU 后端一份，只转该设备的层；
    `PF_GEMM_DNNL=0` 关掉）；CPU 分区层走直读 GGUF 块的 i8 路径（与单设备 CPU 行为相同）。
  - **prefill（整块 mode-2）**：`build_plans` 额外产出 `plan_pfb_` + 行偏移副本 `d_segs_pfb`，
    `batched_prefill_fit` 对多设备返回 `kMaxT..kMaxB*kMaxT`，调度器走 `prefill_batch`（不再 throw）：
    `record_forward(2, ...)` 直接把一个 prompt 的所有 32-token 块合并进单次前向。GPU 分区可转换权
    重组走 oneDNN int8——`add_weight` 对 M=32..512 全部预建原语、`acc_cap` 覆盖最宽 N，`dnnl_call`
    成立且 `D->gemm` 在 M=批 token 数上命中；CPU 分区层走 `i8_gemm` 网格；其余没有 oneDNN 调用的
    小张量（非 K-quant / 无 call_xq）每组合并成一次 fp32 网格分发（`mode == 2 && !single` 分支），
    省掉 mode 1 按行重复调度的开销。整块 batch 内 int8 GEMM ~3.9s、fp32 旁路 ~0.27s（PF_PROF 实测）。
  - **prefill 流水线**：恰好是"dev0（含 embed）/ dev1 / dev0（含 head）"三个相位且全 GPU 分区时，
    `pf_pipe_ok_` 打开（`engine.cpp:1551-1603`），`prefill_batch` 让 chunk *i*+1 的 device-0 相位与
    chunk *i* 的 device-1 相位重叠（T0+T1 → max(T0,T1)）；chunk 间依赖是设备内的（递归状态、
    append-only 的 KV/block 表），所以这个重叠是合法的。`PF_PF_PIPE=0` 关掉。
  - **decode**：`plan_dec8_`（batch-1）走 `w8_dev_` 的 SIn 拷贝 + `dp4a_gemv`（CPU 分区层
    `i8_gemv`），或 oneDNN int8 GEMM（`md_xmx`）；`PF_DP4A_DEC=0` 关掉。`n_rows > 1` 走
    `buckets_` 的 plan（`use_w8 = multi_dev || md_int8`），此时段上没有 SIn 视图
    （`w8_dev_` 只在 `md_int8` 时建），`multi_dev && dnnl_call` 分支把可转换的段逐段送
    `D->gemm_w4` / `D->gemm`。

---

## 12. MTP / NextN 投机解码（引擎侧）

`--mtp [N]` 用模型自带的 NextN 头起草最多 N 个 token，再用主模型一次批量 forward 校验。设计上输出与
普通 greedy decode 等价（接受判据是"目标自己的下一个 token"的相等比较，所以接受只可能省掉工作、
不可能改变输出），但**不保证逐字节相同**——见 §12.7 的 caveat。实现全在
`src/engine/engine_mtp.cpp`（1609 行），缓冲在 `alloc_buffers` 的 `mtp_on` 分支
（`engine.cpp:1760-1891`），字段注释在 `engine.h:444-558`。kernel 侧（`mtp.cpp` 的
`mtp_concat`/`mtp_capture`、`mtp_argmax.cpp` 的 `mtp_argmax`/`mtp_cand`/`mtp_gather`、以及
`nat_gemm_launch` 的批量原生精度 GEMM）见 [03-kernels.md](03-kernels.md)，本文不重复。

> **DFlash2**（`--spec-type dflash2`）复用本节描述的 verify/rollback/accept
> （`mtp_verify`），只有 draft 那一半是新的：它一次非因果前向产出整块候选，由 selector
> lattice 在主机侧走出一条路径。引擎侧在 `src/engine/engine_dflash.cpp`，见
> [设计 14](14-dflash2.md)。

### 12.1 两个硬门控

MTP 需要 GGUF 里有一个额外的**全注意力块** `blk.<n_layer>.*`（由
`qwen35.nextn_predict_layers > 0` 触发并置 `m.has_mtp`，`qwen35.cpp:96-118`）：attention + FFN 线性、
`nextn.eh_proj` / `nextn.enorm` / `nextn.hnorm` / `nextn.shared_head_norm`，可选的
`nextn.shared_head_head`（没有就退回 `m.output`）。参考模型 `Qwen3.8-27B-UD-Q4_K_M.gguf` 有，
0.8B 没有。因此：

1. **没有 NextN 头**（`engine.cpp:108-111`）→ 打印
   `[mtp] model has no NextN (blk.%d.nextn.*) layer - MTP disabled`，`mtp_k = 0`。
2. **没有 multi-device + oneDNN int8 分区**（`engine.cpp:141-149`，条件 `multi_dev && md_xmx`）
   → 打印 `[mtp] MTP needs a multi-device oneDNN int8 partition (...) - disabled`，
   `mtp_on = false`。理由写在代码里：草稿头要走 oneDNN 的 int8 **行主序权重**路径（与多设备 decode
   同一条），其它任何形状都会掉进慢得多的逐 call GEMM 链。

所以单设备的 `--mtp 4` 在 0.8B 上（无头）和在一张卡放得下的 27B 上（单设备）**都是静默 no-op**，
只有一行 stderr 说明原因。

* draft 长度 clamp 到 **12**（`engine.cpp:112-114`）：`n = mtp_k+1 ≤ kMaxB = 16`，这是 `d_logits`
  行数与 `step_info` 的上限。
* `--mtp-device N`（默认 0）选草稿层跑在哪个分区上，越界退回 0（`engine.cpp:150-153`）。
* MTP 层是**多设备预算的一部分**：它是一个全注意力块，拥有自己的一份 paged KV 切片（全局注意力
  层序号 `attn_layers()-1`），因此计入 `--kv-cap-mb`、池预留（`engine.cpp:478`）与三层前缀缓存
  （`attn_layers()` 把它算进去，`engine_kvpool.cpp:102-115`，于是序列化/反序列化/分块 blob
  全部自动覆盖它，零额外管线）。

### 12.2 一个 cycle 的语义

对齐（`engine_mtp.cpp:11-14`）：

```
MTP 行 @ token 位置 p   <->   (emb(t_p), h_{p-1})   ->   t_{p+1}
```

也就是**右移**：一个 cycle 的第一行用主模型在上一个已提交位置的隐状态，之后每一行用 MTP 自己的隐
状态。这条对齐要求草稿头吃的是 **`output_norm` 之前**的 trunk 隐状态——所以主模型 forward 在
output norm 之后立刻把 `d_x` 抄进 `d_mtp_main_h`（`engine_graph.cpp:1233-1238`，`mtp_on` 时），
draft 链再从那里取 `h_{p-1}`。

一个 cycle（`generate_mtp`，`engine_mtp.cpp:1143-1602`）：

1. **draft**：`cand[0] = 上一次提交的 token`，然后 `mtp_forward` 跑 `k` 步，每步 = MTP 层一个
   token + 共享 LM head，`hprev` 依次取上一步的 `d_mtp_raw`（MTP 层**自己的、未过 shared head
   norm** 的隐状态，`engine_mtp.cpp:450`）。
2. **verify**：`mtp_verify(cand, k+1, …)` —— 一次 **mode 2** 的主模型 forward，输入
   `[last_committed, draft0..draft_{k-1}]`，位置连续（`setup_pf_info` 填一个 chunk 行，
   `n_real_row[0] = k+1`）。它同时把每个 token 的递归状态快照进 `d_mtp_hist_`。
3. **接受**：`target_argmax(row i) == cand[i+1]` 连续成立就 `j++`，再取 row `j` 的 argmax 作为
   bonus（`engine_mtp.cpp:1403-1466`）。因此每 cycle 发出 `j+1` 个 token、`pos += j+1`。
4. **commit**：用**已提交**的 token（`cand[0..j]` + bonus，共 `j+2` 个）重跑一次 `mtp_forward`
   （`with_head=false`）重建 MTP 自己的 KV（1481-1491）。注意不能用原始草稿重建：那样从第一个被
   拒绝的草稿起每个位置都由一个从未生成的 token 支撑，而只有 bonus 被提交。
5. **rollback**：`mtp_rollback(j)`（661-701）把**主模型**的 GDN 状态从 `d_mtp_hist_[dev]` 的第 `j`
   个 token 槽拷回活跃状态，并从 `d_mtp_qsave_` 里保存的**原始 qkv tap** 重建 conv 三行窗口
   （更早的行来自 verify 前保存的 `d_mtp_convsave_`）——因为 `conv_state_update` 只在 32-token
   边界快照。这对应 llama.cpp 的 `n_rs_seq`。

**verify 是 dry 的**（`step_info::mtp_dry = 1`，`engine_mtp.cpp:530-535`）：算 forward 但**不写**
活跃的 GDN/conv 状态（只写它自己的每 token 快照槽 + 这些行的 KV，下个 cycle 会重写）。这正是
commit 能把递归状态精确推进 `j+1` 个 token 的前提。副作用是激活量化必须带 even/odd 平面（§12.3）。

接受测试只要每行的 argmax，所以它在**设备**上做（`backend().mtp_argmax`，一次 256-lane work-group
per row + SLM 树归约，平票取**最小**下标，与主机的 `v[i] > v[best]` 逐位一致），只回传 `k+1` 个
int 而不是 `(k+1)*n_vocab` 个 float（k=6 时约 7 MB/cycle）。旧的纯主机扫描实测 **33.7 ms/cycle**
（整个 cycle 的 24%，`mtp_argmax.cpp:1-12`），设备化后 **1.2 ms/cycle（28x）**。
`PF_MTP_AMCHK` 打印设备/主机 argmax 的不一致行（实测 0 条）。注意：这一步**不带来整体加速**——
2000 token 生成实测 200.2/199.8 s（主机扫描）对 199.8/200.1 s（设备），user+sys CPU 时间两路都是
约 150 s（`engine_mtp.cpp:704-713`）：拷贝从来不在关键路径上（verify 自己的 sync 主导，拷贝与下一个
cycle 的 draft 重叠）。保留它是因为主机工作量严格更少且输出逐位相同。

### 12.3 权重存储与 `do_split` 规则

MTP 层的 plan 是手工构造的 5 个 call（`build_mtp_plan`，`engine_mtp.cpp:53-162`）：
`wq/wk/wv` → `wo(+residual)` → `ffn_gate+ffn_up` → `ffn_down(+residual)` → 共享 LM head
（`dev = 0`，因为共享 head 只存在于 primary 设备）。`eh_proj` 不在 plan 里，它在 `mtp_forward` 里
单独跑（`2*n_embd → n_embd`，375-389）。

| 开关 | 默认 | 作用与实测 |
|---|---|---|
| `PF_MTP_LAYER_W4` | **开** | MTP 层的线性（Q6_K/Q8_0，338 MB）也拿一份**原生 per-32 u4** 存储，`add_weight_w4(..., any_type=true, gemv_only=true)`。**叠加**在 int8 存储之上而不是替代它（1229-1247）：MTP 自己的 prefill（M 可到 `kMaxT`）与任何 `M=1` 之外仍读 int8，只有草稿的单行走原生平面。实测 acceptance 完全不变（int8 / u4 / 精确 fp32 三者 acc 都是 2.14），草稿 14.2 → 12.8 ms/cycle |
| `PF_MTP_LAYER_EXACT` | 关 | 让草稿层的 GEMV 走**精确 fp32 反量化参考**（`gemv_group` 读 `seg.w_raw`，需要 `build_mtp_plan` 在多设备下自建 7 份原始 GGUF 设备拷贝，约 290 MB，只为测量）。acceptance 与 int8 **完全相同**（2.14），所以**草稿精度不是约束，草稿的字节才是** |
| `PF_MTP_LAYER_W2` | 关 | 2-bit（0.375 B/w）草稿层存储：acc 2.14 → 1.75，只换 0.5 ms/cycle。**否决**：有损在读出端免费，在**递归**里会复利 |
| `PF_MTP_HEAD_W4` | **开** | 给草稿一份**独立的有损 u4** LM head 副本，key 是私有单字节 key `mtp_head_w4_key_`——注册在 head 的 key 下会把**目标**的 decode 也切到 u4。每起草一个 token 读一次 head（int8 下每次 1.35 GB），0.625 vs 1.0625 B/weight 把 k=6 的草稿从 38.4 降到 30.4 ms/cycle，而所有试过的 prompt 上 acceptance 到小数点后两位不变 |
| `PF_MTP_HEAD_W2` | 关 | 2-bit 草稿 head：精确算术、head 字节少 40%（k=4 草稿 18.9 → 17.4 ms/cycle），但这个 Q6_K head 的 2-bit 拟合代价是 **39.8% 相对 L2**，吃掉约 2% acceptance，端到端打平（32.4 vs 32.0 ms/token）。默认关 |
| `PF_MTP_HEAD_SPLIT` | **关** | 把 795 MB 的 head 读出摊到两张卡：device 1 取上半行、写进 device 0 的**同一行** logits（`out[n] = w[n].h` 按行独立），argmax 仍在 device 0 上一次扫完。草稿 17.2 → 13.1 ms/cycle 确定性变快，端到端打平：u4 GEMV 的分解依赖 N，N 减半会在近平票上翻掉草稿的 argmax（p1 128 token 逐位相同，p0 acceptance −6%、p2 −8%；三个 prompt 净 −0.3%） |
| `PF_MTP_FORCE_INT8` | 关 | 强置 `PF_CB4=0 PF_K5=0`（A/B 用；历史上原生存储会在 verify 里每次都把张量展开回 int8，现在 `nat_gemm` 直接吃 k5/cb4/u4 平面，`M ≤ 13`） |

⚠️ **`do_split` 规则：任何喂给原生存储 GEMV 的激活都必须用 `do_split=true` 量化。** u4/k5 GEMV 读的是
激活的 **even/odd 平面**；`do_split=false` 不会产生它们，于是内核读到的是**上一次调用**留下的陈旧
平面。`eh_proj` 曾漏掉，于是 `PF_MTP_LAYER_W4` 产出的是**错误**草稿（acc 2.14 → 0.08）而不是一个
仅仅有损的草稿（`engine.cpp:1205-1211`、`engine_mtp.cpp:377-381`）——这也正是
`PF_MTP_LAYER_W4_CALL=<ci>`（`-1` = eh_proj、0 = qkv、1 = wo、2 = ffn gate/up、3 = ffn down）
这个二分开关存在的理由。

同一规则在主模型侧由 `record_forward` 自己兜住：
`do_split = (mode == 0 || inf->mtp_dry != 0)`（`engine_graph.cpp:589-590`）。

⚠️ **早期把 MTP 层整体降到 4 bit 的那个负结果**：这些张量是 Q6_K/Q8_0，通用 per-32 u4 打包在 Q6_K
上要 2.6% 相对 L2（cos 0.9949、max|diff| 0.031 on `blk.64.nextn.eh_proj`，是 `PF_W4_ALL` 记录的 8
倍误差），草稿便宜 3 倍（5.4 vs 16.9 ms）但 acceptance 崩到 0.11，净 83 vs 36 ms/token
（`engine.cpp:1183-1191`）。原因不是精度而是**一次错草稿要赔掉整个 verify**。

### 12.4 草稿链

* **MTP 的 attention split 是 `PF_MTP_SPLITS`，与 `PF_MD_SPLITS` 无关**。`--layer-map` 路径把
  `n_splits`/`dec_splits` 钉成 1（它的 split 路径是 opt-in），而草稿是单 token decode，正是 1-split
  网格最糟的形状：`n_head` 个 workgroup 各循环整条 KV，128k 深度下 **2451 ms/cycle vs 183 ms**
  （`engine.h:507-512`）。所以 `mtp_forward` 自己按 KV 长度推 split 数
  （`min(max(ceil((pos0+n)/512), 1), mtp_splits)`，`engine_mtp.cpp:418-429`，即 prefill 用的
  ~512 键/split，上限 `mtp_splits`，默认 `kMaxDecSplits`），并自己分配几百 KB 的 partials
  （`max(mtp_splits, kMaxT*n_head) * (2+head_dim)`，只够这两种形状）。**短上下文与旧的单 split
  行为逐位一致**：固定 256-split 网格实测会降草稿的 acceptance（320-token prompt 上 1.27 vs 2.29
  per cycle），而没有 acceptance 可损失的普通 decode 用满上限正好。
  128k 深度下草稿相位 2451 → **35.7 ms/cycle（68x）**。
* **设备常驻的草稿链**（`mtp_dev == 0`，默认）：head 的 argmax 直接落到 `d_mtp_tok_[step]`，下一步
  的 `mtp_concat` 通过新增的 `tok_dev` 参数从设备内存读它，于是一个 cycle 连续发出 k 个草稿，
  省掉 k 次 sync + k 次 n_vocab float 的 D2H + k 次对 248320 个值的扫描，实测 **−1.2 ms/cycle**，
  输出逐字节相同（`engine_mtp.cpp:1155-1161`、`mtp.cpp:28-35`）。跨设备 `--mtp-device` 保留主机
  往返。
* **`PF_MTP_ADAPT`（默认开）**：一个 cycle 接受 ≤1 个草稿（`j <= 1`）就把 draft 长度减 1，全接受
  （`j == k`）再加回来，上限仍是 `--mtp`（缓冲覆盖得到），且**永不改变输出流**（acceptance 与 k
  无关）。实测低 acceptance 的 prompt 42.0 → 38.7 ms/token，高 acceptance 的自动把 k 升到上限、
  27.5 ms/token（`engine_mtp.cpp:1121-1132`、`1523-1530`）。
* **`PF_MTP_LAYER_EXACT` 的一个陷阱**：`exact` 判定**不能**做成函数内 `static`，否则它会锁住第一次
  调用（MTP prefill 的 `n = 13`）的 `ci`/`M`，之后每次都错——这个探针就是这样一度读到 acc = 0
  而看不见任何 exact GEMV（`engine_mtp.cpp:212-228`）。另外 fp32 反量化 GEMV 只为 `TB ∈ {1,8,16,32}`
  实例化，其它行数会静默跑 `TB=32` 的 kernel（污染后续草稿），所以 exact 只对 `M == 1` 生效、
  `ci == 4`（LM head）永远不走它（多设备下 head 的原始 GGUF 拷贝根本没上传）。

### 12.5 verify

* verify 用的是 `plan_vf_`（§4.3）：mode 2、一个 chunk 行、`head_batched = true`，所以 head 对
  **每一行**都算 logits（普通 mode 2 prefill 的 head 只算一行）。
* verify 现在是**录制的 command graph**（`build_md_verify_graphs`，`engine_graph.cpp:1550-1661`），
  相位切分与 decode 完全相同（`md_dec_` 的同一段代码）：形状固定（`k+1` 行）、每个 cycle 变化的
  东西（token、位置、slot、`n_real`、KV page 表、递归状态快照槽表）全在 kernel 内从 USM 读，所以
  它是"形状固定"而不是"状态固定"的载荷。k=6 时 −3.7 ms/cycle，输出逐字节相同。
  直接重放是每 cycle 约 **810** 次 SYCL 提交（`PF_MTP_SUBMIT`：86 ms 的 pass 里 82 ms 花在主机
  提交上）。
* 两个必须**检查而不是假设**的可录性守卫：
  * `dnnl_capture_guard()`（1600-1601）让每个 oneDNN `prim.execute` 在捕获期间抛异常。verify 的
    GEMM 全走 `nat_gemm_launch`，但一个 nat 服务不了的 shape 会在 `dnnl_gemm::gemm` 里静默回落到
    oneDNN，而 oneDNN 对着录制队列执行会录下一个**缺活的** pass。
  * `vf_graph_usable()`（1663-1681）在 context 超过 `xmx_min_keys` 之后拒绝用图，因为录进去的
    attention 路径（`attn_xmx_launch` 是主机侧按 key 数选的）会过期；此时交回直接重放（它每次
    调用重新决定）。录制时也刻意用一个"最长的仍有效的 context"（`pos = xmx_min - rows - kMaxT`）
    来填 `d_info`，因为偏大的 split 数是正确的（多出来的 split 自己 mask 掉），偏小的不对。
* **`split_valid` 守卫**：`nat_gemm_launch` 的 int8（FMT 3，`dnnl_gemm.cpp:1375`）与 cb4
  （FMT 2，1323）分支被 `p->split_valid` 门住，而 `p->split_valid = quantize(..., do_split)`
  （1246）。prefill 以 `do_split=false` 量化，读到的是不同的（陈旧的）分组激活视图——少了这个
  守卫，0.8B 的多设备 decode-vs-prefill 测试会挂、`PF_NAT=0` 又会过（注释在 `1370-1374`）。
  `PF_NAT=0` 是二分开关。verify 通过 `mtp_dry` 天然满足它。
* **窄 int8 组的融合**：GDN 把 `ssm_alpha`/`ssm_beta`（各 48 行）放进同一个 group，逐段 launch 的
  纯延迟就吃掉 verify 边际预算的一大块（27B 尺寸实测 K=5120 × 32 行 40.1 us、× 96 行 40.9 us、
  × 1024 行 13.5 us @ K=1024，launch 下限 5.8 us，`kernels.h:169-196`）。`gemm_i8_group`
  （`tbm <= 13 && gr.n > 1`，709-742；`PF_NOFUSE=1` 关、`PF_FUSEDBG=1` 追踪）把 27B 的权重一遍从
  68.3 降到 66.5 ms @M=5（53.9 → 53.4 @M=1），verify 94.0 → 91.7 ms。
* **一个必须写下来的负面教训**：`PF_LAUNCHCNT` 数出 verify 是 812 次提交（257 xq、395 GEMV 段、
  64 rmsnorm、96 GDN 家族），但**只有 75 次（9%）可融合**——每个 xq 读的是不同的源，因此整条链都在
  关键依赖上。把段描述符当主机指针传给设备那个版本因此"看起来快 1.9 ms"：设备读到的是零，整组
  GEMV 什么都没算。段描述符必须按值进 lambda 闭包。
* 诊断：`PF_MTP_SUBMIT`（提交时间 vs wall）、`PF_MTP_VERIFY_PAD=<m>` / `PF_MTP_VERIFY_M32=1`
  （在补齐的 M 上跑 GEMM，真实 token 数仍由 `n_real_row` 给出）、`PF_MTP_VERIFYN=<n>`（只校验前 n
  行）、`PF_MTP_DECCHK`（verify 的 row 0 vs 同一 token 的普通单 token decode）。

### 12.6 prompt prefill 与前缀缓存

草稿头需要"上一个 token 的主隐状态"，而那个隐状态只对已经 forward 过的 token 存在，所以 prefill
之后还要让 MTP 层跑一遍 prompt（填充它自己的 KV）：

1. `pc_admit(0, prompt, blocks)`（`engine_mtp.cpp:736`）：恢复匹配链的 KV **与**递归状态。
2. 按 `batched_prefill_fit(rem)` 取**能容纳的最大批量**，一次 `prefill_text` 走完，再
   `prefill_flush()`（多设备流水线把最后一个 chunk 的 device-1 + head 相位推迟到下一次调用，
   而每位置的隐状态与 capture 只由那个 head 相位写）。
3. 把 MTP 层跑在它**捕获到的**隐状态上，按 `kMaxT` 一片一片（823-832）：每片把自己的 capture
   基址当 `h`，把同 capture 里的前一个 token（第一批时是 `d_mtp_hprev`）当 `h_prev`。
4. `dev_queue(0).memcpy(d_mtp_hprev, d_mtp_main_h + (nb-1)*n_embd)`，进入生成循环。

⚠️ 旧实现按 `kMaxT` 分批走 prompt，每 32 个 token 付一次 `prefill_flush()` **加一次阻塞的
`d_mtp_hprev` 主机 memcpy/wait**，把三相位多设备流水线彻底串行化：131k 深度的 e2e_ttft 是 1475 s，
走调度器的同一 prompt 是 311 s；64k 是 697 vs 117 s（`engine_mtp.cpp:794-801`）。

**前缀缓存**：`generate_mtp` 调 `pc_admit`（消费命中：恢复 KV 与递归状态），prefill 循环**之后**调
`pc_commit(0, prompt, blocks, nprompt)`，与调度器同一套契约。顺序是承重的：`pc_commit` 会把
`pc_slot_[0].registered` 推到 `nprompt/32`，而每个 prefill chunk 里的 `pc_capture_begin` 要求
`ps.registered * kBlockSize == pos0`（`engine_prefix_cache.cpp:806-808`）——所以提前提交会让本次
prefill 一个检查点也捕不到（`pc_active` 恒 0），这条路径的缓存支持等于没有；而且节点会在其 KV 块
还没写完时就发布，prefill 中途抛错就会留下"声称常驻、实际是陈旧 KV"的记录。`pc_commit` 必须在
prefill 之后、生成之前。

实测（27B、`--layer-map 0-31:gpu.0,32-63:gpu.1`、`--mtp 4 gen --temp 0`、约 90 token 的 prompt、
`PF_PC_DEBUG=1`，2x A770）：把 `pc_commit` 挪到 prefill 之后，退出统计从
`nodes=2 (with state 0) ... captured=0` 变成 `nodes=2 (with state 2) ... captured=2`——节点从
"有链无状态"（`pc_admit` 只恢复到**有状态**的最深边界，所以这种节点下次根本接不上）变成带递归状态
检查点的可续接命中。改动前那两个节点是既不可续接、又是在 KV 写完之前发布的。

注意 CLI 的 `gen`（无论普通还是 MTP）**不走前缀缓存**：`pc_admit`/`pc_commit` 只有
`src/server/scheduler.cpp:91,259` 调用，`generate_impl` 直接 `alloc_block`。所以这条契约只在
`--temp 0 --mtp N`（`generate_impl` 分流到 `generate_mtp`，`engine.cpp:2365-2368`）与服务端
`mtp_direct`（`PF_MTP_SERVER=1`）下生效。

### 12.7 cycle 的时间分解，以及 2x 是 acceptance 决定的

27B / 2x A770、`--layer-map 0-31:gpu.0,32-63:gpu.1`、`k=4`（`PF_MTP_TIME` 的相计时）：

| 相 | ms/cycle | 说明 |
|---|---:|---|
| draft | 19.0 | 4 步 × 1.25 GB @ 263-297 GB/s —— 就是它的字节下限 |
| verify | 85.0 | 70.7（M=1 等价的那一遍）+ 4.2 / 额外行 |
| commit + rollback + emit | 1.2 | |
| **普通 decode** | **69.1** | 在自身结构下限（17.5 GB @ 325 GB/s + ~13 ms 非 GEMV）的 3% 以内 |

所以 `X = cycle - plain = 36 ms` 的一半是草稿字节、一半是 verify 的额外行，而两个大额固定成本在比
值里互相抵消。

**加速比由 acceptance 上限**：`speedup = n * T_plain / (T_plain + X)`，`n = 1 + 接受的草稿数`。
`PF_MTP_STEPS=1` 给出的 `reach[j] = 75/53/37/28/15/6 %`（k=6）是一条干净的几何衰减
**p ≈ 0.70**，**没有 step-0 崩塌**——说明草稿的隐状态是对的，衰减来自 MTP 层自身的精度，而精度又已
被 `PF_MTP_LAYER_EXACT` 证明不是可动的杠杆。最终实测（每格一个进程，普通 decode 69.10 ms/token）：
代码续写 **2.16x**（k=4）、技术解释 **1.97x**（k=3）、故事开头 1.35x（k=2；它的天花板就是
`n = 1 + acc = 2.23`）。k 的扫面（`main.cpp:253-259`，带 u4 草稿头，两个 prompt）给出
k=2/3/4/5/6/8 → 40.2/38.1/38.1/39.0/40.6/47.6 与 k=3/4/6/8 → 41.1/40.8/43.1/52.7 ms/token，
每个多一个草稿约 5.4 ms、每多一个 verify 行约 5.7 ms，而 k>4 的边际 acceptance 只有 0.1-0.2 ——
所以裸 `--mtp` 的默认值是 k=4。

**MTP 不与批处理复合**，所以 server 默认走调度器：一个批式 decode step 实测
`59.0 + 10.7 ms/行`（N=1/2/4/8 → 70.0/81.6/99.7/145.5 ms 拟合），而 MTP 每个
（`1+acc`）个输出 token 付 `k+1` 行（实测 acc 1.49 时是 **2.0 行/token**，普通 decode 是 1.0），
两条路付的是同样的每行成本，于是模型化地 `S=1/2/4/8` 时 plain 是 69.7/40.2/25.5/18.1 ms/token 而
MTP 是 50.4/36.0/28.8/25.1：`S<=2` 赢、`S>=4` 输（`server.cpp:895-906`）。`server.cpp` 的
`mtp_direct` 因此默认**不**把 greedy 请求路由进单序列 MTP 循环（那样会让整个生成期占住 engine 并
串行化一切：8 个并发 greedy 128-token 请求，MTP 17.5 tok/s vs 走调度器 55.0），
`PF_MTP_SERVER=1` 恢复旧路由。MTP 剩下的价值是**单请求延迟**：同一 server、同一 prompt、同一
greedy 设置，68.6 → **45.5 ms/token = 1.51x**（accept-argmax 修复前是 58.3 / 1.18x）。

**三个必须写下来的测量陷阱**：

* ⚠️ **"与普通 greedy decode 逐字节一致"没有保证**。实测（27B / 2x A770）有一个 prompt 的 ~130 token
  生成从**第 11 个 token** 起就与普通 decode 分叉，而且两种草稿头变体下都一样，`PF_MTP_NOACCEPT=1`、
  `PF_MTP_NORB=1`、`k=1` 也都复现——所以问题在 verify/rollback 的状态而不在草稿质量；其它 prompt
  逐字节相同。在查清之前，这条路径只保证"意图上的 greedy 等价"。
* **前缀缓存命中会改变数值，进而改变 acceptance**：同一 prompt 在同一进程里先以 k=2..5 跑过之后
  acc 1.9（冷）对 2.8。进程内扫 k 比较的是没有从同一状态出发的配置，所以扫 k 必须一个配置一个进程。
* **两个设备分区按必然性串行跑各自的层范围**：token 的 forward 是一条穿过层切分的严格链
  （card 0 的层 0-31 必须先交出隐藏状态，card 1 才能开始），而 verify 的 k+1 行是**一次** forward、
  不是 k+1 次（行 i+1 要 attend 行 i 的 KV）。decode 与 verify 里都没有跨设备重叠可回收。

**草稿的字节砍不动**：草稿头每起草一个 token 要读全部 248320 行（int8 下 1.35 GB）。候选集限制
（§12.8）是唯一想到的减法，实测是净负。

### 12.8 一个机制可行但净负的 idea：候选集草稿头（`PF_MTP_CAND`，默认关）

草稿只需要 head 的 argmax，却读了全部 248320 行。`mtp_cand_launch` 从目标刚产出的分布里收集候选集
（`ids[0]` 恒为那一行的精确 argmax，所以**同一行**上的受限 argmax 与全扫逐位相同），`mtp_gather_launch`
只在候选行上算 head，`mtp_gather_argmax_launch` 取 argmax。**数值精确，机制成立**，草稿 19.1 →
10.9 ms/cycle。但候选集只有 **25/17/7 %** 的概率包含**下一步**的 argmax（margin 8），即便 16384 行
（词表的 7%）也只有 81/77/83 %——分布一步之内跑得太远。净：一个 prompt 43.4 → 42.9 ms/token，另一个
48.4 → 50.4。**默认关**；`PF_MTP_CANDSRC=1` 从草稿自己的第一步播种（默认）、`=0` 从 verify 的
bonus 行播种，`PF_MTP_CANDDBG=1` 打印逐步命中率，`PF_MTP_CANDV=1` 额外回传选中候选的 logit。

### 12.9 MTP 环境变量

总表以 `AGENTS.md#environment-variables` 为准；下面是引擎侧分组（`0`/未设 = 关，除注明默认者）：

* 开关：`--mtp [N]` / `PF_MTP`（长度，默认 k=4）、`--mtp-device N` / `PF_MTP_DEV`、
  `PF_MTP_FORCE_INT8`、以及全局的 `PF_NOGRAPH`（关掉 verify 的录制图与各直放路径的图）。
* 权重存储：`PF_MTP_LAYER_W4`（**默认开**）/ `PF_MTP_LAYER_W4_CALL=<ci>` /
  `PF_MTP_LAYER_W2(_CALL)` / `PF_MTP_LAYER_EXACT` / `PF_MTP_HEAD_W4`（**默认开**）/
  `PF_MTP_HEAD_W2` / `PF_MTP_HEAD_SPLIT(_DEBUG)` / `PF_MTP_CAND` `_CANDM` `_CANDSRC` `_CANDDBG`
  `_CANDV`。
* 循环行为：`PF_MTP_ADAPT`（**默认开**）、`PF_MTP_SPLITS`、`PF_MTP_NORB` /
  `PF_MTP_NORBSYNC`、`PF_MTP_NOACCEPT`、`PF_MTP_NOMTPFWD`、`PF_MTP_DECODE_H`、
  `PF_MTP_VERIFYN`、`PF_MTP_VERIFY_PAD` / `PF_MTP_VERIFY_M32`。
* 诊断：`PF_MTP_TIME`（相计时 + `cycle-wall` 的闭合检查）、`PF_MTP_DSTEP`、`PF_MTP_SUBMIT`、
  `PF_MTP_AMCHK`、`PF_MTP_STEPS`、`PF_MTP_DEBUG`/`PF_MTP_DUMP`、`PF_MTP_DECCHK`、
  `PF_MTP_STATECHK`/`PF_MTP_STATEDUMP`/`PF_MTP_SNAPCHK`/`PF_MTP_SNAPDUMP`/`PF_MTP_STATESEQ`、
  `PF_MTP_INFOCHK`/`PF_MTP_HVEC`/`PF_MTP_HPROBE`/`PF_MTP_VPROBE`/`PF_MTP_HOSTCMP`/
  `PF_MTP_HDUMPS`、`PF_MTP_MEM`、`PF_MTP_EXACT_CALL`/`PF_MTP_EXACT_DEBUG`、
  `PF_MTP_AUTOTEST`/`PF_MTP_DRAFTTEST`/`PF_MTP_DIAG`、`PF_MTP_LSTAT`、以及全局的
  `PF_HOSTPROF` / `PF_LAUNCHCNT`。

---

## 13. 引擎侧不变量与陷阱

1. **绝不要把设备查询放进 kernel launcher**。`rmsnorm_launch` 每次 launch 都调
   `si::dev::wg_clamped()`，所以那里的 `sycl::device::get_devices()` + `max_work_group_size`
   查询**必须**缓存在函数内 `static` 里（`src/device/device_registry.cpp:134-142`，那里的注释记着
   实测值）。设备 profile 重构曾让它变成每次重跑：**3.2 ms/次**，一次 MTP verify 里的 65 次
   rmsnorm 就是 210 ms 纯主机时间，verify 221 → 85 ms、cycle 255 → 107 ms（修好后 MTP 从 0.72x
   变成 1.8-2.2x）。普通 decode 看不到它，因为 decode 是启动时录制一次的 command graph，而 verify
   是直接重放——`PF_HOSTPROF=1`（`PF_PROF` 的分桶但不做每戳设备等待）就是找到它的工具。
2. **`build_plan` 与 `record_forward` 的调用顺序必须逐条对齐**，且 plan 是**设备指针快照**：
   `bind_acts()` 不会回头修改已构建的 plan（§4.3 的 ⚠️）。
3. **部分（按设备的）重放必须从 `seg_plan::layer_c0[首层]` 起步 call 游标**（§9.1）。
4. **逐步状态只能在 kernel 体内从 `step_info` 读**（§6.4）；注意 `record_forward` 里为此做的两处主机
   判断都有对应的录制期处理（模式 2 的 split hint、`mtp_dry`）。
5. **mode-2 的部分 batch 需要 oneDNN 权重路径**（§7.1）；all-`cpu` 的 `--layer-map` 必须由
   `resolve_device` 选到 CPU queue，否则 decode 路径被破坏。
6. **完全融合的 mode-2 GDN 必须传 `nreal_arg`**（§5.3）。
7. **`nat_gemm_launch` 的 int8/cb4 路径需要 decode/verify 的激活形式**（`split_valid`，§12.5）。
8. **GDN 头配对是取模、attention GQA 是分块**：`gdn.cpp` 把 value 头 `h` 与 q/k 头 `h % n_group`
   配对（对应 `ggml_repeat_4d`），`attn.cpp` 用 `kvh = h*n_head_kv/n_head` 展开（HF `repeat_kv`）。
   两者只在头数相等时一致（恰好是 0.8B 的 `n_group == dt_rank == 16`），所以在 0.8B 上弄错不可见、
   在 27B 上致命。
9. **`qkv_dim()` 不是 `3*d_inner`，`m.output` 不总是 `m.tok_embd`**（前者见 §4.2，后者见 §9.2）。
10. **GGUF mmap 在权重到达设备后被逐出**：`setup_md_dnnl` 的 `add` 每转一个张量就
    `page_out_tensor`，`upload_device_weights` 每拷一个就 `page_out_host`，
    `release_host_weight_pages` 扫尾。映射仍然有效（设备指针只用 `map_base` 做算术），但**此后任何
    新的主机读取都会从磁盘重新缺页**——正确性保住了，吞吐没有。CPU 分区的页必须留在 keep 集合里
    （含 backend 0 是 CPU 时的 `tok_embd`/`output`/`output_norm`）。
