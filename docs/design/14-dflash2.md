# 设计 14：DFlash2 block drafter

> 配套阅读：[设计 04 §12（MTP 引擎侧）](04-engine.md)、[设计 03 §15（MTP kernel）](03-kernels.md)。
> DFlash2 与 MTP 的 **verify / rollback / accept** 那一半完全共用（`engine_mtp.cpp` 的
> `mtp_verify`），差别只在 **draft** 那一半：MTP 逐 token 前向，DFlash2 一次前向产出整块候选。

## 1. 概览

DFlash2 是一个**块扩散（block-diffusion）草稿器**，不是第二个语言模型：

* 它**没有**自己的 embedding 与 LM head（GGUF 里就没有），词表、embedding、LM head
  全部借目标模型的；
* 它读目标模型在若干层的 hidden state 作为输入；
* 它**一次非因果前向**产出整块 `block_size - 1` 个候选 token。

参考实现：llama.cpp `src/models/dflash.cpp` +
`common/speculative.cpp::common_speculative_impl_draft_draft_dflash`。

一个 cycle 的数据流（`src/engine/engine_dflash.cpp:933` `df_block`）：

```
目标 hidden（target_layers 处捕获）
    │  [n_tgt_layer][n_embd] 拼接 → fc 投影 → enc.output_norm
    ▼
   fc_out ──► 每层的 injection：在**已提交** token 上算出该层自己的 K/V
    │            （所以 drafter 不必跑目标的层就能看到对话历史）
    ▼
   block：一次前向 over [anchor, MASK × (n_max)]，注意力**非因果**
    │        attention_swa=2048；每层 attn_norm → conv → qkv → rope → attn
    │        → wo → conv1 → 残差；ffn_norm → conv → gate/up → down → conv1 → 残差
    ▼
   out_norm → 共享 LM head → logits[M][n_vocab]
    │
    ├─► top-k（每行 K 个候选 id/val）
    └─► selector：unary + <A[p]·gate(h_i), B[c]>，给出 (i-1 候选 p → i 候选 c) 的边权
    ▼
   主机沿 lattice 走**一条**连贯路径 → cand[0..M-1]
    ▼
   mtp_verify：目标模型一次 mode-2 前向 over [last_committed, cand…] → 接受/回滚
```

### 1.1 文件

| 文件 | 作用 |
|---|---|
| `src/model/dflash.{h,cpp}` | 草稿 GGUF 的 metadata、张量绑定、conv/selector 几何推导 |
| `src/backend/gpu/kernels/dflash.cpp` | `df_capture_launch` / `df_conv_launch` / `df_gemm_head` / `df_topk_launch` / `df_sel_launch` / `df_rowrms` 等设备 kernel |
| `src/engine/engine_dflash.cpp` | `df_block`（一次块前向）、`df_draft`、`generate_dflash`（投机循环） |
| `src/engine/engine_graph.cpp` | 目标 hidden 的捕获 hook（`df_capture`） |

## 2. 模型与几何

实测模型：`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`

| 项 | 值 |
|---|---|
| 层数 | 5 |
| `n_embd` | 5120 |
| `n_head` / `n_head_kv` | 32 / 8 |
| `head_dim` | 128 |
| `block_size` | 8（`n_max` 默认 5） |
| `attention.sliding_window` | 2048 |
| mask token id | 248070 |
| conv | `conv_kernel_size=2`、`conv_group_size=16` |
| selector | `sel_rank=256`、`sel_top_k=16` |
| `target_layers` | `[6, 20, 34, 48, 62]`（27B 有 64 层，取 5 层） |

派生量（`dflash.cpp:42-43,127-128`）：

```
n_groups = n_embd / conv_group          = 5120/16 = 320
conv_proj = 2 * conv_k * n_groups       = 2*2*320  = 1280
n_feat    = n_tgt_layer * 目标 n_embd    = 5*5120   = 25600
```

一个 cycle 的行数 `M = n_max + 1`：第 0 行是**已提交的 anchor**，第 1..M-1 行填 `mask_id`。

## 3. 卷积系数：两个镜像的轴序

DFlash2 在 attn/FFN 前面各有一个 grouped depthwise **动态**卷积，系数来自两个张量的组合：

| 张量 | 形状 | `(c, t, side)` 的线性下标 |
|---|---|---|
| `attn_conv_base` / `ffn_conv_base` | `[2][group_size][n_groups][conv_k]` | `side*width*conv_k + c + width*t` |
| `dyn`（= `conv_proj` 的输出 `[M][conv_proj]`） | `[n_groups][conv_k][2][M]` | `g + n_groups*t + side*n_groups*conv_k` |

base 那一侧是 **channel 最内**（`(c,t) = c + width*t`），dynamic 那一侧是 **token 最内**
（`(g,t,side) = g + n_groups*t + side*n_groups*conv_k`）。**两者的轴序是镜像的**，
而且两边都曾经写错过。`df_conv_launch` 里 `base` 通道内、`dyn` 通道外。

> 这两个下标是本项目里最容易写错的地方之一：base 写错时 anchor 行（只用 tap 0）仍能
> 跟参考对上（cos 0.994），只有 mask 行错，于是出现 **"anchor 接近、mask 行偏离"**
> 的特征签名；先修好 base 之后 anchor 行到 cos 0.999997，才把剩下的误差隔离到
> tap/side 系数上。

## 4. RoPE：旋转完整 head_dim

`dimension_sections` 是 `[64,0,0,0]`，**读 `sections[0]` 当旋转宽度会得到 64，参考旋转 128**。
必须旋转完整的 `head_dim=128`。这是一个固定的 ~0.7% 注意力偏差，在任何聚合量里都看不见，
但会累积到第 4 层 cos 0.79。实测（对 `DFLASH_REF_CUR`，pos0=204，row 0）：

```
n_rot = 32 / 64 / 96 / 128  →  cos = 0.9930 / 0.9932 / 0.9947 / 0.999991
```

## 5. attention_sinks

按 `ggml_soft_max_add_sinks` 的语义接入了：每个 query head 多一个 key，分数 `sinks[h]`、
value 为 0，整体效果是乘一个 `z / (z + exp(sink - max))` 因子。

**但本 GGUF 不带这个张量**——15 个 `blk.N.*` 张量里没有 `attn_sinks`，llama.cpp 那边也是
`TENSOR_NOT_REQUIRED` 建出来的。所以在这份模型上它是 no-op；接上而不是假定它不存在，
是因为一份**真的**带 sinks 的草稿模型会在每个 head 上都错。

## 6. selector lattice 与主机侧行走

selector 给每个 block 位置 i 打分：

```
score(p → c) = unary_i[c] + <A[p] * gate(h_i), B[c]>
```

`gate(h_i)` 是最终 norm 后的 hidden 经 `sel_hidden` 投影得到的 `sel_rank=256` 维向量。
`df_sel_launch` 输出 `[M][K + K*K]` 的 lattice（`K=16`）。

主机从位置 1 开始走，每步在 `pred` 那一行里对 16 个候选取 argmax 作为下一个 `pred`
（`engine_dflash.cpp` 的 `pred` 循环），**一次**产出整条路径——这是 DFlash2 与"每 token 独立
重抽"的关键差别。

> **候选顺序是模型定义的一部分。** ggml 的 `top_k` 排序后会交换前两个元素
> ("emphasize that the order is not important")，但 DFlash2 的 lattice 行走把候选下标 k
> 当作候选 k 的后继，所以参考实现的 `(1,0)` 次序必须原样复现，否则每条转移边都会配错前驱。

## 7. 性能（27B / 2×A770，greedy）

短 prompt，与 llama.cpp 在同一 `n_max` 下对比（均值 = 每 cycle 接受的 token 数）：

| 生成 token 数 | llama.cpp | 本实现 |
|---:|---:|---:|
| 32 | 4.43 | 4.75 |
| 116 | 3.96 | 3.90 |
| 227 | 4.13 | 3.92 |

`n_max` 扫描（本实现）：

| `n_max` | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| acc（每 cycle 接受数） | 0.98 | 1.44 | 2.18 | 2.54 | **2.90** | 2.90 |
| ms/token | 48.6 | 41.9 | 34.1 | 32.5 | **31.5** | 33.3 |

**默认 `n_max=5`**（2.90 drafts/cycle，31.5 ms/token，对 plain decode 的 ~68 ms/token 是
**2.17x**）。204-token 的长 prompt 上两边都掉（llama.cpp 0.239 / 均值 2.07，本实现
0.81 / 1.81），所以长上下文差距约 12%。

cycle 分解（`PF_DFLASH_TIME=1`，k=5）：

```
draft=26.9  verify=90.9  rb=1.0  inject=1.5  emit=1.5   cycle=121.8 ms
```

### 7.1 verify 在结构下限上

`n_max` 扫描给 verify 定价：

| k | 1 | 2 | 3 | 5 | 7 |
|---|---:|---:|---:|---:|---:|
| verify ms | 71.9 | 76.3 | 80.0 | 90.8 | 105.0 |

即 **72.2 ms 固定 + ~5.5 ms/额外行**。固定部分就是一次 plain decode 的权重流，而权重流模型
能把它预测到 0.1 ms：`53.93(流式) + 3.67×(k)(dp4a) + ~18.5(非 GEMM) = 90.8`。

那条在 MTP 上曾花掉 210 ms 的坑——**未图化的直接重放**——在这里不存在：DFlash 的 verify
走 `mtp_verify` 的已录制 command graph，且它是活的（`PF_DFLASH_VFCHK` 打印
`vf_dec_ok`、行数、图数：1、6、378）。所以只剩两个杠杆，都已经被定价并否决：
更少的权重字节（`PF_W4_K5` verify −5.7%，代价是 4.60% 平均相对 L2 权重误差），或者更快的
dp4a（DPAS 发行率高 16-24×，但真实 GEMM 里慢 37×）。**两条都别重新推导。**

### 7.2 draft 块前向：平坦，没有 82% 的热点

设备侧 7 段（`PF_DFLASH_SEGTIME`，全部 5 层，ms）：

```
anorm=0.36  acproj=1.71  aconv0=2.27  attn=7.05   (qkv+rope+attn+wo+conv1)
ffn: norm+cproj+cconv=1.89   gate_up=2.75   down+cconv=4.33
sum=20.35   devspan=20.37   hostwall=20.70
```

`sum ≈ devspan` 到 0.1% 才是这张表可信的原因：`devspan` 是第一层开始栅栏到最后层结束栅栏，
所以**前向是 device-bound**（20.37/20.70 ms），而 A/B 相减永远给不出这个结论。最大单项是
注意力路径 35%，其次 `ffn_down + conv1` 21%。

> 这一段历史值得留着：早先那张表说"RMSNorm 占 82%"，而真实值是 0.36 ms。三个 bug 叠在
> 同一处：累加在 5 层循环**之后**只跑一次（所以每列都是第 4 层的）、marker 的下标顺序与
> 程序顺序不一致（程序顺序是 0,6,5,1,3,4，于是差分了两个早 marker 和两个晚 marker，打印
> −3.60 ms）、以及 `seg_e[0]` 在循环**外**而其余都在循环内，于是第 0 段量的是
> "第 0 层之前 → 第 4 层的 norm"，几乎整个前向。**一个坏掉的 verifier 比没有更糟**。

### 7.3 top-k：唯一真正慢的 kernel

读出阶段对 `[M=6][n_vocab=248320]` 的 logits 做 top-16，5.96 MB，实测 **2.06 ms**
（≈2.8 GB/s，而地板约 300 GB/s）。它不是显而易见的嫌疑：把每 lane 的 top-K SLM 列表从
`[tid][k]` 转置成 `[k][tid]` 可以消掉一个真实的 16 路 bank conflict
（stride 16 个 float 让 256 个 lane 落在两个 bank 上，每次元素比较命中一次、每轮归约再命中
16 次），而实测 **2.06 → 2.11 ms，毫无变化**。真正的成本是 **512 个 EU 上只起了 6 个工作组**。

现在 `df_topk_launch` 把每行切成 `S` 片，各自归约成 `[M][S][K]` 的 partial，再用一个
workgroup/行合并。两点值得不要重新推导：

* 所有切片必须在**一次 launch** 里覆盖 `M*S` 个工作组、边界由 group 下标算。第一个版本
  每片单独 submit（好让边界是字面量），结果把工作串行化了（每次 submit 仍然只有 6 个
  工作组），实测 **8.0 ms，比原版差 4×**。
* 最优值**不是**占满机器的那个：`S=1/8/32/64/128/256` → `2.09/1.39/1.52/1.69/1.70/1.67 ms`，
  所以取 `S=8`。超过 8 之后合并本身和它要归约的 `S*K` 个候选比多出来的并行更贵——这说明
  kernel 卡在**每 lane SLM 插入链的延迟**上，而不是带宽上（1.39 ms 跑 5.96 MB 仍然只有
  ~4 GB/s）。真要重写应该是"每 lane 一个寄存器阈值 + warp ballot 合并"，而不是更多切片。

净效果：topk 2.06 → 1.39 ms，draft 28.0 → 26.9 ms，cycle 122.9 → 121.8 ms，输出逐字节不变。

## 8. 与参考实现逐元素对比

**逐元素比，永不看指纹。** 参考侧 `DFLASH_REF_*` 在 `DFLASH_REF_IL=<il>` 选中的那一层发布一个
张量，`DFLASH_REF_BIN=<path>` 原样写出；本侧 `PF_DFLASH_BIN=<path>` 做同样的事，
`PF_DFLASH_SIGALL` 加上每一层。

`rms`/`max`/`first`/`top-6` 只能说明偏差**多大**，永远说不出偏差**在哪**；而且 anchor 行的
能量比 mask 行高一个数量级，所以**整批聚合值和 llama.cpp 的逐位置打印根本不可比**——曾经有
一个聚合值说 `wo` 输出低了 30%，那是在比两个不同的东西。llama.cpp 的 copy-out 固定是
`n_embd` 宽，所以更窄的张量得在它那一侧补零（`DFLASH_REF_DYN` 做了这件事）。

### 8.1 诊断的名字必须带上触发时机

四次"自信但错误"的结论都出自这里，代价比那 15 个 bug 本身还大：

* 把 f16 ring 当 f32 读（1e12 rms 的垃圾数据，以及一个 104/205 的"覆盖缺口"）；
* 从 `d_df_h` 而不是 `d_df_c` 读层输出（于是造出了一条很干净的误差累积曲线）；
* dump 只按层命名，后面的块覆盖了前面的块，同一层的两次读数**正好差一层**——由此得出一个
  极有说服力的"第 4 层读错了张量"，而层号是在**另一个函数**里被设的（一个
  `replace(...,1)` 匹配了两个相同的 `const dflash_layer_t &` 声明中的第一个，和让一个探针
  段错误的是同一个错误）；
* `df_conv_check` 保留了修复前的下标推导，于是 conv 轴修好之后它在报告两个公式的差，同时
  还把自己当成 conv 的验证（第 0 层报 host 0.805 vs dev 1.545）。

## 9. CLI

```
--spec-type dflash2              开启 DFlash2 块草稿器
--spec-draft-model <gguf>       草稿 GGUF（必需）
--spec-draft-n-max N            每 cycle 的草稿 token 数（默认 5，PF_DFLASH_NMAX 可覆盖）
--spec-draft-device N           草稿跑在哪个设备分区（默认 0）
```

`--mtp [N]` / `--mtp-device N` 是 `--spec-type mtp` / `--spec-draft-device` 的别名。

## 10. 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `PF_DFLASH_NMAX` | 5 | 每 cycle 草稿长度（覆盖 `--spec-draft-n-max`） |
| `PF_DFLASH_TIME` | off | cycle 各阶段 ms |
| `PF_DFLASH_VFCHK` | off | verify 是否走已录制 command graph，行数与图数 |
| `PF_DFLASH_SEGTIME` | off | 设备侧 7 段分解 + `sum`/`devspan`/host wall |
| `PF_DFLASH_BTIME` | off | draft 块前向：embed+layers vs outnorm+head+topk+selector |
| `PF_DFLASH_TOPTIME` | off | 读出尾部的**设备侧**拆分：topk / selector_hidden / selector |
| `PF_DFLASH_DTTAIL` | off | 同上但从**主机侧**：块前向 / 阻塞拷贝 / lattice 行走 |
| `PF_DFLASH_SLICES` | 8 | top-k 每行的切片数（实测最优，非占满机器的值） |
| `PF_DFLASH_DEBUG` | off | 下面所有探针的总开关 |
| `PF_DFLASH_SIG` / `PF_DFLASH_BIN` / `PF_DFLASH_SIGALL` | off | 逐位置指纹 / 全张量 dump |
| `PF_DFLASH_GEMMCHK` / `PROJCHK` / `CONVCHK` / `CONVCHKALL` / `BASECHK` / `ROWRMS` / `EMBCHK` / `INJCHK` | off | 各自的 host/device 对拍探针 |

## 11. 已知未做的事

* **草稿权重格式**：49 个草稿张量全部注册为 u4（0.625 B/weight），`failed=0`。把
  `ffn_gate`+`ffn_up` 拼成一个 `[2*n_ff, n_embd]` 的 GEMM（逐位等价、一次 launch 代替两次）
  实测 28.0 → 28.3 ms，即**什么都没有**；换 int8 存储（1.7× 字节）也只有 28.0 → 29.2。
  所以草稿块前向 20.5 ms 对 4.67 ms 的带宽地板，既不是字节、也不是 launch 数、也不是算术
  （10 GMAC 在实测 7.6 T-MAC/s 下是 1.3-2.8 ms）。
* **长上下文**：204-token prompt 上均值 1.81 vs 参考 2.07，差约 12%。这一段是引擎解决不了的
  之外的问题——天花板是 `n = 1 + acc`。