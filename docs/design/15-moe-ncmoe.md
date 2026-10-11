# 15 — MoE 架构加载与 ncmoe 专家切分（含三级专家缓存）

本文覆盖 routed-MoE（Qwen3.5 / Qwen3.6 的 `qwen35moe`）在引擎里的三件事：

1. **加载与执行**：`qwen35moe` 架构的绑定、每层混合量化、shared expert 与
   sigmoid 门控 epilogue（§1–§3）；
2. **ncmoe 切分**：设备只保留固定的一部分专家，另一部分每步由主机计算
   （§4–§6）；
3. **三级专家缓存**：disk → RAM → VRAM，VRAM 那级是**实测**频次冻结的 bank，
   不是索引截断（§7–§9）。

模型几何：35B A3B 为 40 层 trunk + 1 个 NextN 块，`n_expert=256`、
`n_expert_used=8`、`n_ff_exp=512`、`n_ff_shexp=512`，routed 专家占 18.22 GiB
（466.5 MiB/层）。

---

## 1. 架构注册与绑定

`src/model/qwen35moe.cpp` 定义 `arch::load_qwen35moe(model &)`，在
`src/model/model.cpp` 的 `kLoaders[]` 注册。除 `qwen35` 已有的 GDN / 全注意力层绑定外，
它额外绑定：

| 张量 | 含义 |
|---|---|
| `blk.N.ffn_gate_exps` | routed 专家的 gate，`[E, n_ff_exp, n_embd]` |
| `blk.N.ffn_up_exps` | routed 专家的 up |
| `blk.N.ffn_down_exps` | routed 专家的 down |
| `blk.N.exp_probs_b` | router 的 logits，fp32，`n_expert` 宽 |
| `blk.N.ffn_gate_inp` | shared expert 的门，`n_ff_shexp` 宽 |
| `blk.N.ffn_gate_shexp` / `up_shexp` / `down_shexp` | shared expert 本体 |

NextN 块（`m.mtp`）自己也有一套 `moe`（`blk.<n_layer>.nextn.*`），
`moe_t` 挂在 `mtp_layer_t` 上，因此 slot 表的最后一行就是 NextN 层
（见 `setup_moe_slots()`）。

`validate_hparams` 会拒绝几何不支持的模型。注意 `n_expert`、`n_expert_used`
不参与通用检查，它们是 MoE loader 自己的责任。

---

## 2. 每层混合量化：slot 步长必须逐层

35B A3B 的 gate/up 是 Q4_K，**down 有 37 层 Q5_K、3 层 Q6_K（34/38/39）**。
所以 `moe_slot_gb[] / moe_slot_ub[] / moe_slot_db[]` 是**逐层**的
（`src/engine/engine.cpp` 的 `setup_moe_slots()`），且另存一个 `_max` 供需要
"一个 bank"的场合取最大值。

这是本次踩过的最深的一个坑：早期实现把 stride 当成**每张量一个值**，
Q6_K 层读到 Q5_K 层的权重，症状是"模型能加载、48 层里 46 层对、
第 34/38/39 层错"——而且这三层的量化类型**只在这一个模型上**就这么排布。

自检方法：`setup_moe_slots()` 在启动时打印三种张量各自的层数与不同值的数量；
一个"静悄悄"的启动日志意味着混合量化没被检测到。

---

## 3. MoE 的 plan：四个 call

`record_forward` 的 MoE 分支（`engine_graph.cpp`）把一层 MoE 编成：

```
call 0  moe_routed     router + CSR + gate/up + down + reduce（无 gemv 段）
call 1  ffn_gate_shexp, ffn_up_shexp      （shared expert 的 dense FFN）
call 2  ffn_down_shexp                    （无 residual）
call 3  moe_epilogue     sigmoid(gate_inp @ x) * shared，并加上 routed 半
```

顺序是必需的：routed 半先落进 `d_x`，shared 的 down 再读 `d_moe_shout`，
否则 residual 会被读到一半。

### 3.1 f32 门不能污染整类 call（oneDNN 门槛是全 call）

GDN 的 `ssm_beta`/`ssm_alpha` 是 f32。它们原先和 `wqkv`/`wgate` 共用一个 plan
call，而 **oneDNN 资格判定是 per-call 且全有全无**：一个不可转换的 f32 张量
就让整个 call 退回 fp32 反量化 GEMV，把 `wqkv`/`wgate` 一起拖下水。
拆成独立 call 后恢复 `gemv()` 分派。

### 3.2 `moe_gate_add` 的子组约简

`moe_gate_add_impl` 用 256-thread work-group、256-stride 的点积，但只在
**一个 32-lane 子组**里做约简——算出来的是部分和，shared expert 的门和
sigmoid 因此偏大约 20 倍。现在是**每个 token 一个 32-lane 子组**、
stride 32（`src/backend/gpu/kernels/moe.cpp`）。

### 3.3 slot 表

`slot_map[layer][expert]` 给出该专家在设备 bank 里的槽位，`-1` 表示不驻留；
所有 MoE kernel（gate/up、down）第一件事就是 `if (slot < 0) return;`。
所以"哪些专家在设备上"是一个**纯主机侧**的决定，内核不需要改。

---

## 4. ncmoe 切分：固定部分在 GPU，可变部分在 CPU

参照 llama.cpp 的 ncmoe：每个 token 只激活 `n_expert_used` 个专家，
所以**驻留集小于 `n_expert`** 时，缺的专家由主机补算。CLI：

```
--moe-resident N     每层驻留 N 个专家（0 = 全驻留，默认）
--moe-vram-mb MB     设备专家 bank 的字节预算（与上一者取更小者）
--moe-ram-mb  MB     主机 RAM 层的预算（0 = 关闭）
```

`moe_host_split()` = `moe_cache_.active() && !moe_cache_.all_resident()`。

### 4.1 主机半必须在 reduce **之前**换填，不能在之后加

这是本次真正的 bug，也是它为什么长成那样的原因：

设备 `moe_down_launch` 对 `slot < 0` 的 pair 直接 `return`，
所以 `d_moe_acc` 里这些行还留着**上一层**的值。而 `moe_reduce_launch`
会对**所有行**加权求和。

* 主机 pass 放在 reduce **之后**再加（第一版）：stale 行被当成当前层输出
  加权计入，正确值又加了一遍。
* 症状极其有辨识度：**layer 0 精确**（缓冲区初始为零）、**layer 1 起
  routed 能量虚高 2–4 倍**。逐层 routed 指纹从 300% 相对差降到 1%，
  剩下的 1% 是 fp32 设备 / fp64 主机的累加舍入。
* 正确顺序：主机先读 `d_moe_acc`、只覆盖非驻留行、写回，**然后** reduce
  一次性加权求和。`moe_acc_add_launch`（一个"事后追加"的入口）随之删除——
  有了正确顺序它就是死代码。

### 4.2 主机半读的是 MoE 自己的输入缓冲

主机 half 原先硬编码 `d_xnorm`。MTP 的 draft 层跑在 `d_mtp_xnorm` 上
（见 `engine_mtp.cpp` 里 `mtp_moe` 的注释），读 `d_xnorm` 等于拿**另一层**的
激活去算专家。现在输入缓冲作为参数传入。

### 4.3 主机 pass 不能进 command graph

它是带阻塞 `memcpy`/`wait` 的主机 pass，所以**每一处**图复录都要让路：

| 位置 | 处理 |
|---|---|
| 普通 decode | `moe_host_split()` → `nog` |
| 多设备 decode | `md_dec_ok && !moe_host_split()` 才复录图 |
| MTP verify | `v_pad == 0 && !moe_host_split() && vf_graph_usable()` |

任何一处漏掉，recording 会"看起来正常"，只是每一步静默丢掉所有无槽专家。

### 4.4 边界（实测，未解决）

* verify 的 **anchor 行**是对的（`PF_MTP_NOACCEPT=1` 组合并输出正确的流）。
* verify 的**第 2 行及之后**（决定是否接受草稿的那些行）不对：
  `--mtp >= 2` + 切分会输出错误 token。
* `--mtp 1` + 切分实测正确。

引擎没有假装它没坏，也没有留一条静默输出错字的路径：切分 armed 时把 MTP 钳到
k=1 并打印一行说明和绕法。这是**容量限制**，不是调优——要 k>1 就先别切分。

---

## 5. 主机半的算子

`moe_cpu_experts()`（`src/engine/moe_cache.cpp`）：

1. 读回该 layer 的 `d_moe_ids` / `d_moe_wts`；
2. `row[e] < 0` 的行进 CPU 列表（与设备的 `slot_map[e] < 0` 判据一致）；
3. 从 RAM 层或 mmap 取这三个 expert 的 GGUF 块；
4. 逐 pair 反量化 + **fp64 累加**（与 CPU 后端一致），
   `h = silu(gate) * up`，再乘 down；
5. memcpy 回 `d_moe_acc` 的对应行。

Expert 平面寻址：GGUF 里 gate/up/down 是**三个独立张量**，各自的基址不同；
只有 RAM 层把三个平面**打包**在一个 entry 里。曾把"打包式"指针算到 mmap 上
（`up = gp + ng`），读到的其实是 expert e+1 的 gate 平面——RAM 层一开
输出就变成乱码，而 `--moe-resident`（不开 RAM 层）完全正常。

---

## 6. 诊断

| 变量 | 作用 |
|---|---|
| `PF_MOE_RESIDENT` / `PF_MOE_VRAM_MB` / `PF_MOE_RAM_MB` | CLI 三个开关的 env 形式 |
| `PF_MOE_CPUDBG` | 每个 MoE 层一行：`il / n_tok / spilled of n_pairs` |
| `PF_MOE_REFDUMP` | 逐层 routed 指纹（sum/sumsq/max） |
| `PF_MOE_REFDUMP_ALL=1` | 打开**每一层**（默认只 layer 0） |
| `PF_MOE_REFDUMP_LAYER=L` | 只看第 L 层 |

**逐层指纹是本次唯一有效的定位手段。** `routed_exact` 是"前减后"的元素级差
（d_x 在 reduce 前后之和不可加，只有差分是诚实的 per-token 量）。
一个只 dump layer 0 的开关会发现不了 layer 34/38/39 的混合量化问题——
因为它第一次跑的时候 layer 0 恰好是对的。

`test_moe_ref`（`tests/model/test_moe_ref.cpp`）是主机侧 oracle：
`TEST_TOKENS` / `TEST_SNAPS` / `TEST_ROWS` / `TEST_TOKROW` 决定快照哪些
中间量。它的 MoE 参考实现与 llama.cpp 在 40 层上一致到 0.4%。

---

## 7. 三级专家缓存

```
disk   GGUF mmap           无预算，ground truth。命中失败 = 一次缺页，不是一次读
RAM    (layer, expert) 的   主机侧字节缓存，LRU + 字节预算
       GGUF 块             —— 让主机半不再去碰被 MADV_DONTNEED 掉的页
VRAM   设备 bank           每层 resident_ 个槽；内容是**实测冻结**的热集
```

`src/engine/moe_cache.{h,cpp}` 只有 slot 表 / 频次表 / RAM LRU，不含设备内存。

### 7.1 VRAM：实测频次，不是索引截断

`--moe-vram-mb MB` 是**预算**：`per_expert = gate+up+down 的 plane_bytes`，
`resident_ = min(resident_mb / per_expert, n_expert)`。
启动打印 `55 of 256 experts resident per layer`（实测 `--moe-vram-mb 100`
在 35B A3B 上）。

冻结发生在 `moe_calibrate()`：用一个 prefill 期间 `moe_calib_ids_` 收集的
**真实路由**填充频次表，再按频次降序取前 `resident_` 个（tie 取低索引，
保证确定性——slot 表会被录进 command graph）。
调用点在 `decode_batch` 的开头（一次性）：此时每层的路由都已收齐，
且产生它的 forward 已经结束。

**为什么不在 prefill 的尾部做**：多设备 prefill 是流水的
（`prefill_batch` 提前 return），它的尾部跑不到。

### 7.2 RAM：为什么缓存"块"而不是"解量化后的行"

一个专家 512+512+2048 行 fp32 权重是 ~25 MB，而它的 GGUF 块是 ~0.6 MB。
主机半本来就是逐 pair 反量化的，缓存行没有意义；缓存块才能拿到真正的收益——
**不再去缺页**。

LRU 的淘汰单位是**一整个专家**（三个平面一起）。
`ram()` 是 `const` 且返回指向 entry 的指针；它的两个计数器是 `mutable`
（查询不该改策略）。命中率在 `--moe-ram-mb 150` 下实测 48–66%。

### 7.3 `release_host_weight_pages()` 必须延后

这个 sweep 在 `alloc_buffers()` 里就跑，而 `setup_moe_slots()` 也在
`alloc_buffers()` 里、且**位置更后**——它要读 `moe_resident` 才能决定 keep-set。
第一版在这里问 `moe_host_split()`，永远得到 false（cache 还没 init），
专家页被 MADV_DONTNEED 掉，主机半每行都缺一次页：不慢就算了，还不可观察。

现在 sweep 延后到 `setup_moe_slots()` 之后（`moe_host_weights_deferred_`），
实测 `--moe-resident 64` 时释放量从 21603 MB 降到 2396 MB——差额就是保住的主机侧
专家页。

---

## 8. 多设备与 CPU 分区

slot 表按 `n_layer + 1` 行构建（最后一行是 NextN），并 memcpy 到**每个**设备的
`moe_as_[d].slotmap`。`bind_moe_acts(dev)` 负责在换设备时换这套指针；
`record_forward` 的 MoE 分支用的 `dev` 来自该层所属分区。

主机半按"算它的那个设备"入队（`dev` 参数穿到 `moe_cpu_experts`），
因此双 GPU 的切分两侧各自在自己队列上做主机 pass。

---

## 9. 环境变量速查

| 变量 | 默认 | 说明 |
|---|---|---|
| `PF_MOE_RESIDENT=N` | 0 | 每层驻留 N 个专家；0 = 全驻留 |
| `PF_MOE_VRAM_MB=MB` | 0 | 设备 bank 预算；0 = 由 N 定 |
| `PF_MOE_RAM_MB=MB` | 0 | 主机 RAM 层预算；0 = 关闭该层 |
| `PF_MOE_CPUDBG=1` | off | 每层 spilled/总数 |
| `PF_MOE_REFDUMP_ALL=1` | off | 逐层 routed 指纹 |

测试侧：`TEST_MOE_RESIDENT`（`stage_env::make`）让 stage 测试也走切分路径；
`test_decode_vs_prefill`/`test_gpu_vs_ref`/`test_multimodal` 都认这个变量。
