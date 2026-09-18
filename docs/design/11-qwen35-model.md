# 设计 11：Qwen3.5 混合模型

覆盖 `src/model/qwen35.cpp` 以及它在引擎/内核中的执行路径。通用加载机制见
[01-model-loading.md](01-model-loading.md)，kernel 细节见 [03-kernels.md](03-kernels.md)。

---

## 1. 架构概述

Qwen3.5 是一种**混合架构**：大多数层是 **Gated DeltaNet（GDN）** 线性递归层，每隔
`full_attention_interval` 层插入一个 **full attention（GQA）** 层。

`hparams::is_recr(il)`（`model.h:30-32`）：

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

---

## 2. 超参数

来自 Qwen3.5 的 GGUF 元数据（`qwen35.cpp:18-39`）：

| `hparams` 字段 | GGUF key |
|---|---|
| `n_layer` / `n_embd` / `n_ff` | `block_count` / `embedding_length` / `feed_forward_length` |
| `n_head` / `n_head_kv` / `head_dim` / `n_rot` | `attention.head_count` / `attention.head_count_kv` / `attention.key_length` / `rope.dimension_count` |
| `n_vocab` | `tokenizer.ggml.tokens` 长度 |
| `rope_base` / `rms_eps` | `rope.freq_base` / `attention.layer_norm_rms_epsilon` |
| `attn_scale` | `1/sqrt(head_dim)`（派生） |
| `d_state` / `n_group` / `dt_rank` / `d_inner` / `conv_k` | `ssm.state_size` / `ssm.group_count` / `ssm.time_step_rank` / `ssm.inner_size` / `ssm.conv_kernel` |
| `full_attn_interval` | `full_attention_interval`（默认 4） |
| `rope_sections[4]` | `rope.dimension_sections`（如 `[11,11,10,0]`） |

`attn_scale = 1/sqrt(head_dim)`；`rope_sections` 全零表示普通 RoPE，非零时启用文本模型的交错 M-RoPE。

---

## 3. 加载器张量绑定（`qwen35.cpp`）

全局：

| 张量 | 目标 |
|---|---|
| `token_embd.weight` | `m.tok_embd` |
| `output_norm.weight` (f32) | `m.output_norm` |
| `m.tok_embd_row_bytes` | `ggml_row_bytes(tok_embd.type, n_embd)` |

每层前缀 `blk.<il>.`，所有层公共：`attn_norm.weight`、`post_attention_norm.weight`（f32）、
`ffn_gate.weight`、`ffn_up.weight`、`ffn_down.weight`。

**GDN 层**（`L.recurrent`）：

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

---

## 4. 一次 forward 中的 Qwen3.5 层

由 `engine::record_forward` 驱动（见 [04-engine.md](04-engine.md)）。每层先
`rmsnorm(d_x, attn_norm)`，然后按类型分支。

### 4.1 GDN 层

```
gemv: wqkv -> d_qkv, wgate -> d_z, ssm_beta -> d_beta, ssm_alpha -> d_alpha
conv_l2(d_qkv, conv_state, ssm_conv1d -> d_conv_out)        # depthwise 4-tap + q/k L2 norm
conv_state_update(写回滑窗 + 可选检查点)
gdn(d_conv_out, alpha, dt_bias, ssm_a, beta, state -> d_attn_pre)
gated_norm(d_attn_pre, d_z, ssm_norm -> d_attn_merged)
gemv: ssm_out -> d_x (+ residual d_x)
```

* q/k/v 打包进 `d_conv_out`：q 在偏移 0，k 在 `n_heads*head_dim`，v 在 `2*n_heads*head_dim`。
* `gdn_launch` 调用参数：`head_dim = d_state`、`n_heads = dt_rank`、`scale = 1/sqrt(d_state)`。
* `gdn` 的 delta 规则（见 [03-kernels.md](03-kernels.md)）：
  `g = exp(A*softplus(alpha+dt_bias))`、`del = (v - g*(S·k))*sigmoid(beta)`、
  `S = g*S + k⊗del`、`out = (S·q)*scale`。
* `gated_norm`：`out = a * rsqrt(mean(a²)+eps) * ssm_norm * silu(z)`。
* 模式 2 且 `PF_GDN_FUSE>=1` 时整批融合（一次 conv_l2 + 一次 state_update + 一个 gdn dispatch），
  把每行 launch 链从 48 降到 4。

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
* `d_conv_state`：每 GDN 层 `(conv_k-1) * 3*d_inner` 的滑窗。
* 分配按 layer-major（`engine.cpp:432-433`），索引 `[n_gdn][kMaxB][...]`。
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

* 线性层通常是 Q4_K/Q5_K/Q6_K，norm/SSM 标量为 F32。
* `PF_DP4A` 时 `build_w8` 为 token embedding、FFN 三线性、GDN 的 `wqkv/wgate/ssm_out`、attention 的
  `wq/wk/wv/wo` 构建 SIn int8 副本；`ssm_beta`/`ssm_alpha`/norm 不复制。
* 见 [02-quantization.md](02-quantization.md)。

---

## 8. 相关测试

* `test_gpu_stages`：`conv_stage`、`gdn_stage`、`gated_norm_stage`、`qk_norm_rope_stage`、`attn_stage`
  等逐 kernel 对照 CPU 参考（强制 `PF_DP4A=0`）。
* `test_gpu_vs_ref`：整模型 GPU logits vs `tests/common/cpu_ref.h`。
* `test_forward`：端到端 logits/top-k。
* `test_cpuref`：CPU 参考头输出。

详见 [12-build-and-testing.md](12-build-and-testing.md)。
