# 设计 11：Qwen3.5 混合模型

覆盖 `src/model/qwen35.cpp` 以及它在引擎/内核中的执行路径。通用加载机制见
[01-model-loading.md](01-model-loading.md)，kernel 细节见 [03-kernels.md](03-kernels.md)。

---

## 1. 架构概述

Qwen3.5 是一种**混合架构**：大多数层是 **Gated DeltaNet（GDN）** 线性递归层，每隔
`full_attention_interval` 层插入一个 **full attention（GQA）** 层。

`hparams::is_recr(il)`（`model.h:33-35`）：

```cpp
bool is_recr(int il) const { return (il + 1) % full_attn_interval != 0; }
```

`full_attn_interval` 默认 4，因此 `il % 4 == 3`（也就是 `il+1` 能被 4 整除）是 full attention，其余是
GDN。参考模型 Qwen3.5-0.8B 用这一配置。

| 层类型 | 计算 | 状态 |
|---|---|---|
| GDN | depthwise causal conv → delta-rule 递归 → gated RMSNorm | 每序列固定大小的递归状态 + conv 滑窗 |
| Full attention | GQA + QK norm + RoPE | paged KV cache |

这种混合是前缀缓存必须携带递归状态检查点的根本原因，见
[06-prefix-cache.md](06-prefix-cache.md)。

此外还有**主干之外的一层**：MTP/NextN 草稿层（`--mtp`）。它是一个 **full attention** block，
既不在 `m.layers` 里、也不占 `gdn_layer_index` 的位置，而是独立的 `m.mtp`，并且自带一份
paged KV slice。见 §3.2。

---

## 2. 超参数

来自 Qwen3.5 的 GGUF 元数据（`qwen35.cpp:19-46`）。加载器用一个 `key()` 闭包把所有元数据键加上
`"qwen35."` 前缀（`qwen35.cpp:17`），所以下表里的短名都实际带前缀：

| `hparams` 字段 | GGUF key |
|---|---|
| `n_layer` | `block_count` **减去** `nextn_predict_layers`（见下） |
| `n_mtp` | `nextn_predict_layers`（默认 0） |
| `n_embd` / `n_ff` | `embedding_length` / `feed_forward_length` |
| `n_head` / `n_head_kv` / `head_dim` / `n_rot` | `attention.head_count` / `attention.head_count_kv` / `attention.key_length` / `rope.dimension_count` |
| `n_vocab` | `tokenizer.ggml.tokens` 长度 |
| `rope_base` / `rms_eps` | `rope.freq_base`（代码默认 10000）/ `attention.layer_norm_rms_epsilon`（默认 1e-6） |
| `attn_scale` | `1/sqrt(head_dim)`（派生） |
| `d_state` / `n_group` / `dt_rank` / `d_inner` / `conv_k` | `ssm.state_size` / `ssm.group_count` / `ssm.time_step_rank` / `ssm.inner_size` / `ssm.conv_kernel` |
| `full_attn_interval` | `full_attention_interval`（默认 4） |
| `rope_sections[4]` | `rope.dimension_sections`（如 `[11,11,10,0]`） |

`attn_scale = 1/sqrt(head_dim)`；`rope_sections` 全零表示普通 RoPE，非零时启用文本模型的交错 M-RoPE。

⚠️ **`block_count` 包含 NextN 层**。`load_qwen35` 算的是
`hp.n_layer = block_count - n_layer_nextn`（`qwen35.cpp:20-23`），所以：

* 0.8B 的 GGUF **没有** `qwen35.nextn_predict_layers`，`hp.n_mtp == 0`、`hp.n_layer == 24`，
  `--mtp` 在它上面是 no-op（`engine.cpp:108-111` 打一行 `[mtp] model has no NextN (blk.%d.nextn.*)
  layer - MTP disabled`）；
* 27B 的 GGUF 有 `qwen35.nextn_predict_layers == 1`，于是 `hp.n_layer = 65 - 1 = 64`，草稿头落在
  `blk.64.*`（与 `engine.cpp:1186` 注释里提到的 `blk.64.nextn.eh_proj` 一致），**主干里不存在第 65 层**。

`rope.freq_base` 在这两个参考 GGUF 里都是 `10000000.0`（1e7），不是代码里的默认值 10000——写死
10000 会在参考模型上直接跑出错误的 RoPE。

**派生宽度**（`hparams` 的成员函数，不要在别处手写）：

| 派生量 | 定义 | 0.8B | 27B |
|---|---|---|---|
| `qkv_dim()` | `2*n_group*d_state + dt_rank*d_state` | 6144 | **10240** |
| `d_inner` | `dt_rank*d_state`（= ssm_out 的 K） | 2048 | 6144 |
| Q 宽度 | `n_head*2*head_dim`（第二半是 gate） | 4096 | 12288 |
| K/V 宽度 | `n_head_kv*head_dim` | 512 | 1024 |
| RoPE 覆盖 | `n_rot` 对（`n_rot <= head_dim`，**部分旋转**） | 64 / 256 | 64 / 256 |

⚠️ **`n_group == dt_rank` 只在 0.8B 成立**（16 == 16）。27B 是 **16 vs 48**，于是出现两个"只在 0.8B
上恒等、在 27B 上崩坏"的陷阱，见 §4.1 与 [03-kernels.md §9](03-kernels.md)：

1. `qkv_dim()` **不等于** `3*d_inner`（27B：10240 vs 18432）——历史上曾用 `3*d_inner` 当 qkv/conv 宽度。
2. q/k 的 head 数（`n_group`）**不等于** v 的 head 数（`dt_rank`）——value head `h` 只能配对到
   **`h % n_group`**（取模），不是分块。

---

## 3. 加载器张量绑定（`qwen35.cpp`）

### 3.1 主干

全局（`qwen35.cpp:48-57`）：

| 张量 | 目标 |
|---|---|
| `token_embd.weight` | `m.tok_embd` |
| `output.weight`（存在时） | `m.output` |
| `output_norm.weight` (f32) | `m.output_norm` |
| `m.tok_embd_row_bytes` | `ggml_row_bytes(tok_embd.type, n_embd)` |

**没有 `output.weight` 时 `m.output = m.tok_embd`**（tied embeddings，0.8B）。这是一个条件分支而不是
恒等式，见 §7。

每层前缀 `blk.<il>.`，所有层公共：`attn_norm.weight`、`post_attention_norm.weight`（f32）、
`ffn_gate.weight`、`ffn_up.weight`、`ffn_down.weight`。

**GDN 层**（`L.recurrent`，即 `is_recr(il)` 为真）：

| 张量 | 目标 |
|---|---|
| `attn_qkv.weight` | `wqkv` |
| `attn_gate.weight` | `wgate` |
| `ssm_beta.weight` | `ssm_beta` |
| `ssm_alpha.weight` | `ssm_alpha` |
| `ssm_out.weight` | `ssm_out` |
| `ssm_a` (f32) | `ssm_a` |
| `ssm_dt.bias` (f32) | `ssm_dt` |
| `ssm_norm.weight` (f32) | `ssm_norm` |
| `ssm_conv1d.weight` (f32) | `ssm_conv1d` |

并令 `gdn_layer_index[il] = gdn_count++`（该层在 GDN 层中的顺序索引，供递归状态缓冲与 kernel 使用）。

**Full attention 层**：`attn_q.weight`、`attn_k.weight`、`attn_v.weight`、`attn_output.weight`，
以及 f32 `attn_q_norm.weight`、`attn_k_norm.weight`；`gdn_layer_index[il] = -1`。

注意 `ssm_beta` / `ssm_alpha` 是**量化**张量（0.8B 上是 Q8_0，形状 `[n_embd, dt_rank]`），
`ssm_a` / `ssm_dt` / `ssm_norm` 是 f32；`ssm_conv1d.weight` 在 GGUF 里是**转置**的 `[conv_k, qkv_dim]`
（0.8B：`[4, 6144]`），只能按裸 f32 数组读，见 [01-model-loading.md §2.5](01-model-loading.md)。

### 3.2 MTP / NextN 草稿层（`qwen35.cpp:96-118`）

`hp.n_mtp > 0` 时才绑定，前缀是 `blk.<hp.n_layer>.`——**注意它不在主干的层循环里**
（`m.layers` 只有 `hp.n_layer` 个元素，`gdn_layer_index` 也不含它），而是独立的 `m.mtp`
（`mtp_layer_t`，[01-model-loading.md §5.3](01-model-loading.md)）：

| 张量 | 目标 | 备注 |
|---|---|---|
| `attn_norm.weight` / `post_attention_norm.weight` (f32) | `M.attn_norm` / `M.post_attn_norm` | 与主干同名的普通 block norm |
| `attn_q/k/v/output.weight` | `M.wq/wk/wv/wo` | **full attention**，宽度用主干的 `n_head`/`n_head_kv`/`head_dim` |
| `attn_q_norm.weight` / `attn_k_norm.weight` (f32) | `M.q_norm` / `M.k_norm` | |
| `ffn_gate/up/down.weight` | `M.ffn_gate/up/down` | |
| `nextn.eh_proj.weight` | `M.eh_proj` | `[2*n_embd][K=2*n_embd] -> n_embd` |
| `nextn.enorm.weight` / `nextn.hnorm.weight` (f32) | `M.enorm` / `M.hnorm` | 分别归一化 `emb(t_p)` 与 `h_{p-1}` |
| `nextn.shared_head_norm.weight` (f32) | `M.shared_head_norm` | **无条件绑定**（缺失即抛 `missing tensor`） |
| `nextn.shared_head_head.weight` | `M.shared_head` | **存在才绑定**；为空时草稿回落用 `m.output` |

随后 `m.has_mtp = true`。它的输入语义（对齐 llama.cpp 的右移）：

```
eh_proj( concat( enorm(emb(t_p)), hnorm(h_{p-1}) ) ) -> 一个 full-attention Qwen3.5 block -> shared_head_norm -> LM head
```

即 **MTP 在位置 p 的行预测 token p+1**，其条件是主模型在 `p-1` 的 hidden（不是自己的上一层输出）。
执行侧的 plan/forward/verify/rollback 见 [04-engine.md](04-engine.md)，这里只强调两条与加载/资源分配
直接相关的事实：

* 它是一个 **full-attention** 层，因此**自带一份 paged KV slice**（`engine_kvpool.cpp:102-115` 的
  `attn_layers()` 在 `mtp_on` 时 `+1`，索引为 `attn_layers()-1`），共享 block table、KV 存储类型与
  前缀缓存，**并计入 `--kv-cap-mb` 与三层 prefix cache 预算**；
* 它的 f32 norm 上传到 `--mtp-device` 指定的那个分区，线性层（`eh_proj` 与 block 里的 7 个矩阵）的
  oneDNN/u4 转换也发生在该分区的权重表里；而草稿用的 shared LM head
  （`M.shared_head` 或 `m.output`）**永远在主设备 0 上**，因为它和主干 head 共用一张表
  （`engine.cpp:877-912`、`engine_mtp.cpp:128-137`）。

---

## 4. 一次 forward 中的 Qwen3.5 层

由 `engine::record_forward` 驱动（见 [04-engine.md](04-engine.md)）。每层先
`rmsnorm(d_x, attn_norm)`，然后按类型分支。

### 4.1 GDN 层

```
gemv: wqkv -> d_qkv, wgate -> d_z, ssm_beta -> d_beta, ssm_alpha -> d_alpha
conv_l2(d_qkv, conv_state, ssm_conv1d -> d_conv_out)        # depthwise 4-tap + silu + q/k L2 norm
conv_state_update(写回滑窗 + 可选检查点)
gdn(d_conv_out, alpha, dt_bias, ssm_a, beta, state -> d_attn_pre)
gated_norm(d_attn_pre, d_z, ssm_norm -> d_attn_merged)
gemv: ssm_out -> d_x (+ residual d_x)
```

* `d_qkv` / `d_conv_out` 的宽度是 `hp.qkv_dim()`（**不是 `3*d_inner`**，见 §2）。
* q/k/v 打包（**q/k 段用 `n_group` 个 head，v 段用 `dt_rank` 个 head**）：
  q 在偏移 0（`n_group*d_state`）、k 在 `n_group*d_state`、v 在 `2*n_group*d_state`；
  `conv_l2` 只对前 `2*n_group` 组（q 和 k，组宽 `d_state`）做 L2 归一化，v 段不做。
* **value head `h` 的 q/k 来自 head `h % n_group`（取模/交错）**。参考实现用 `ggml_repeat_4d` 展开 q/k
  的 head 轴，其平铺是取模；写成分块 `h*n_group/dt_rank` 会在 `n_group != dt_rank` 时彻底配错
  （0.8B 上两种写法等价，所以这个 bug 曾长期不可见——见 [12-build-and-testing.md](12-build-and-testing.md) §7 的教训）。
* `gdn_launch` 调用参数：`head_dim = d_state`、`n_k_heads = n_group`、`n_heads = dt_rank`、
  `scale = 1/sqrt(d_state)`；`gated_norm` 用 `n_heads = dt_rank`、`head_dim = d_state`（`ssm_norm`
  只有 `d_state` 个元素，按 value head 复用）。
* `wqkv` 的 N 是 `qkv_dim()`，`wgate`/`ssm_out` 的 N/K 是 `d_inner`。
* `gdn` 的 delta 规则（见 [03-kernels.md](03-kernels.md)）：
  `g = exp(A*softplus(alpha+dt_bias))`、`del = (v - g*(S·k))*sigmoid(beta)`、
  `S = g*S + k⊗del`、`out = (S·q)*scale`。
* `gated_norm`：`out = a * rsqrt(mean(a²)+eps) * ssm_norm * silu(z)`。
* 模式 2（chunk 批量 prefill）下 `PF_GDN_FUSE>=1`（默认 2）整批融合：**一次** `conv_l2` + **一次**
  `conv_state_update` + **一个** `gdn` dispatch（`engine_graph.cpp:978-1007`）。非融合路径是每行 3 次
  dispatch（`conv_l2` + `conv_state_update` + `gdn`，`engine_graph.cpp:1010-1030`），而 `gated_norm`
  两种路径都只调一次、覆盖整批（`engine_graph.cpp:1049-1053`）。
* 融合路径的 `gdn` 用 `kMaxT` 作为行数、`tpb` 作为 token 数（`engine_graph.cpp:994-996`）；非融合路径
  逐行走 `row0 = r0`。`nreal_arg` 是这里唯一的例外开关：融合路径用一行走完整批
  （`nreal_arg > 0 ? nreal_arg : row_nr(...)`，`gdn.cpp:53-59`）——没有它，kernel 只处理第 0 行的
  `kMaxT` 个 token，后续 chunk 行读到陈旧递归状态，这就是 mode-2 prefill 的那个 bug。

### 4.2 Full attention 层

```
gemv: wq -> d_qbuf, wk -> d_kbuf, wv -> d_vbuf
qk_norm_rope(d_qbuf, d_kbuf, d_vbuf, kpool/vpool, tables, info)
attn(...)  # 非融合时写 partials，再由 attn_combine 归约
gemv: wo -> d_x (+ residual d_x)
```

* Q buffer 每 head 占 `2*head_dim`，第二段是 attention gate。
* `qk_norm_rope` 对 Q/K 做 per-head RMSNorm + RoPE，并把 K/V 写进 paged KV 池。

### 4.3 公共 FFN

```
rmsnorm(post_attn_norm)
gemv: ffn_gate -> d_ffn, ffn_up -> d_ffn + n_ff
gemv: ffn_down (SiLU 门控)
```

最后：`rmsnorm(output_norm)` → `copy_row` → LM head。

---

## 5. 递归状态缓冲

* `d_gdn_state`：每 GDN 层一个 `[dt_rank][d_state][d_state]` 矩阵，每序列一份 slot。
* `d_conv_state`：每 GDN 层 `(conv_k-1) * qkv_dim()` 的滑窗（**不是 `3*d_inner`**）。
* 单设备布局：`[kMaxB][n_gdn][dt_rank][d_state][d_state]` 与
  `[kMaxB][n_gdn][conv_k-1][qkv_dim()]`，分配在 `engine.cpp:1722-1723`；`n_gdn` 只数 GDN 层
  （`attn_layers()` 数 attention 层，两者互斥，见 `engine_kvpool.cpp:102-124`）。
* 多设备布局：**每个分区只分配自己那几层**的状态（`engine.cpp:759-764`，用 per-device 的
  `n_gdn_dev_` 计数），并通过 `bind_acts(dev)` 在设备间切换（`engine.cpp:792-793`）。所以“层 n 的状态”
  在多设备下不是全局第 n 个 slot，而是“第 n 层所在分区的第 n_gdn_dev_[dev] 个”。
* 前缀缓存检查点按层切分为 `[GDN gdn_per][conv conv_per]`，见
  [06-prefix-cache.md](06-prefix-cache.md)。

---

## 6. M-RoPE

文本模型在 `rope_sections` 非零时使用**交错** M-RoPE：`qk_norm_rope` 按 lane（head 内维度对索引）在
`0,1,2` sector 模式中选择 temporal/row/col 位置，频率指数仍用全局 pair 索引。`rope_sections` 例如
`[11,11,10,0]` 表示 temporal/row/col/extra 各占多少对。

纯文本请求 `mrope_on=0`，`rpos = pos`，退化为普通 RoPE。多模态请求由
[10-multimodal.md](10-multimodal.md) 填充 `step_info::mrope`。

---

## 7. 权重与量化

* 线性层通常是 Q4_K/Q5_K/Q6_K，norm/SSM 标量为 F32，`ssm_beta`/`ssm_alpha` 是窄的量化行
  （0.8B 上是 Q8_0，形状 `[n_embd, dt_rank]`）。
* `PF_DP4A` 时 `build_w8` 为 **`output`（LM head）**、FFN 三线性、GDN 的 `wqkv/wgate/ssm_out`、
  attention 的 `wq/wk/wv/wo` 构建 SIn int8 副本；`ssm_beta`/`ssm_alpha`/norm 不复制
  （`model_w8.cpp:96-113`）。`output` 的 SIn 副本只有 8-bit 类型可精确表示，5/6-bit（Q5_K/Q6_K）与
  codebook 型（IQ*）的精度损失见 [02-quantization.md](02-quantization.md)。
* **`m.output` 与 `m.tok_embd` 未必是同一张量**：GGUF 没有 `output.weight` 时加载器把 `m.output`
  指向 `tok_embd`（0.8B），27B 则自带独立的 Q6_K `output.weight`。任何算 logits 的地方都必须用
  `m.output`——用 `m.tok_embd` 在 0.8B 上看不出差别，但 27B 的 logits 会全错（这正是
  `tests/common/cpu_ref.h` 曾经的问题，见 [12-build-and-testing.md](12-build-and-testing.md) §7）。
* 这个条件分支在**多设备**下还有一个后果：LM head 的 oneDNN key 取决于它是否 tied——untied 用 host
  指针（上传前转换），tied 用已上传的 device 指针（上传后转换，因为 embed kernel 还要读 GGUF 原始
  行）。用错 key 只会让 head 静默掉回 fp32 dequant GEMV，实测 12.3 → 3.7 ms/token（27B / 2x A770）。
  完整规则与两个陷阱见 [01-model-loading.md §4.4](01-model-loading.md)。
* MTP 草稿层的 SIn/u4/w2 副本与草稿头的第二份 u4 副本（`PF_MTP_HEAD_W4`，注册在私有 key 下，
  避免把 target 的 decode 也切到 u4）见 [04-engine.md](04-engine.md)。
* 见 [02-quantization.md](02-quantization.md)。

---

## 8. 相关测试

* `test_gpu_stages`：`conv_stage`、`gdn_stage`、`gated_norm_stage`、`qk_norm_rope_stage`、`attn_stage`
  等逐 kernel 对照 CPU 参考（强制 `PF_DP4A=0`）。
* `test_gpu_vs_ref`：整模型 GPU logits vs `tests/common/cpu_ref.h`。
* `test_forward`：端到端 logits/top-k。
* `test_cpuref`：CPU 参考头输出。
* `test_cpu_gdn`：**CPU 的 `cpu_gdn` 用非对称 head 数**（`n_k_heads=2, n_heads=6`）对照测试内标量参考，
  覆盖 `head % n_k_heads` 配对与状态写回。0.8B 的真实维度是恒等映射，测不出这个 bug，所以必须显式造
  非对称维度（该测试已做反向验证：改回分块映射会 `MISMATCH`）。
* `test_w4*` / `test_quant_audit`：u4/SIn 打包与权重路径的正确性、量化误差。
* `test_27b_prefill` / `test_w4_vs_cpuref`（配 `TEST_LAYER_MAP`）：27B 分卡 prefill 的 logits vs
  fp32 CPU 参考——**27B 专属的线数/宽度问题只在这里暴露**。
* `test_decode_vs_prefill`：单 token decode 与"同序列重新 prefill"必须给出同一预测。**纯 prefill 的
  测试对解码路径是盲区**，这个测试是发现 head 段缓冲绑定 bug 的唯一手段（见
  [12-build-and-testing.md](12-build-and-testing.md) §7）。
* tied-embedding 的多设备 LM head 只有 `test_decode_vs_prefill` 覆盖（0.8B 是唯一 tied 的模型）：
  `TEST_LAYER_MAP=0-11:gpu.0,12-23:gpu.1 ./build/test_decode_vs_prefill` 与
  `TEST_LAYER_MAP=0-11:gpu.0,12-23:cpu`。head 掉回 fp32 dequant GEMV **不会让输出变错**，只慢 3 倍以上，
  所以对这条路径而言 `test_decode_vs_prefill` 只验正确性，性能要靠启动日志里
  `[dev] tied LM head: oneDNN int8 conversion keyed by its device copy`（`engine.cpp:1670`）这行确认。
* MTP 路径目前**没有单元测试**（`tests/` 里只有 `test_w4_gemm.cpp` 引用了 MTP），验证靠 CLI 的
  `PF_MTP_TIME` / `PF_MTP_STEPS` 诊断与 `AGENTS.md` 里记录的测量。

详见 [12-build-and-testing.md](12-build-and-testing.md)。

---

## 9. Qwen3.8-27B：非退化情形

27B 与 0.8B 的结构参数几乎相同（`head_dim=256`、`n_rot=64`、`rope_sections=[11,11,10,0]`、
`conv_k=4`、`d_state=128`、`n_group=16`），差别在**规模与两处非退化**：

| 参数 | 0.8B | 27B | 影响 |
|---|---|---|---|
| `block_count` / `hp.n_layer` | 24 / 24（无 NextN） | 65 / **64**（`nextn_predict_layers=1`，草稿头在 `blk.64`） | 主干 full attention 层 = `(il+1)%4==0` → 3,7,…,63 共 16 层有 KV（0.8B 是 3,7,…,23 共 6 层） |
| `n_head` / `n_head_kv` | 8 / 2 | 24 / 4 | GQA 比例 4 vs 6 |
| `dt_rank` | 16 | 48 | **`n_group != dt_rank`**，见 §4.1 |
| `d_inner` | 2048 | 6144 | — |
| `qkv_dim()` | 6144 | 10240 | `qkv_dim() != 3*d_inner` |
| LM head | 与 `tok_embd` 共享（无 `output.weight`） | 独立 `output.weight`（Q6_K，裸拷贝约 1.0 GB，`engine.cpp:1258-1262`） | 见 §7 与 [01 §4.4](01-model-loading.md) |

27B 还有一处必然踩到的分支：`--layer-map` 是必需的（模型放不下一张卡），所以多设备 LM head 那条路
每次都在场——而 tied（0.8B）与 untied（27B）恰好各走一条 key 规则，见上表最后一行与 §7。

模型几何可从 GGUF 直接核对：`dev/gguf_dump.py <model.gguf>`（本地未跟踪的压测工具，`dev/` 在
.gitignore 中）会打印全部 kv 与张量形状。27B 上关键的几行是
`blk.0.attn_qkv.weight = [5120, 10240]`（= `2*16*128 + 48*128`）、`blk.0.ssm_out.weight = [6144, 5120]`、
`blk.3.attn_q.weight = [5120, 12288]`（= `24*2*256`）、`blk.0.ssm_norm.weight = [128]`。
0.8B 的对应值是 `blk.0.attn_qkv.weight = [1024, 6144]`（= `2*16*128 + 16*128`，两者相等正是
`n_group == dt_rank` 的表现）、`blk.3.attn_q.weight = [1024, 4096]`（= `8*2*256`）、
`blk.0.ssm_out.weight = [2048, 1024]`、`blk.0.ssm_norm.weight = [128]`。
核对张量形状是发现“宽度假设写错”最快的方式（[12-build-and-testing.md](12-build-and-testing.md)
§7 第 9 条）。
