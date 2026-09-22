# 设计 02：量化格式与 int8 计算路径

覆盖 `src/common/quant.h`、`src/common/w8.{h,cpp}`、`src/common/dp4a.h`、`src/backend/gpu/kernels/xq.cpp`、
`src/backend/dnnl_gemm.{h,cpp}`。kernel 侧的 GEMV/GEMM 实现见 [03-kernels.md](03-kernels.md)。

---

## 1. 概览

引擎有 GPU 的 SIn/DP4A、CPU 的 GGUF 整数 int8、可选 oneDNN，以及融合 fp32 回退：

| 路径 | 权重格式 | 激活 | 启用 | 适用 |
|---|---|---|---|---|
| **u4 原生 4-bit**（GPU，Q4_K） | `w4t`（原生 q + 每 32 组 (step, offset)） | 每 32 组对称 int8（`act_quant_grp_launch`） | `PF_W4=1`（默认，且张量是 Q4_K） | Q4_K 的 prefill GEMM + decode GEMV，见 §9 |
| **SIn + DP4A**（GPU） | `w8t`（GGUF 整数原值，按位宽打包） | 运行时对称 int8（`xq_launch`） | `PF_DP4A=1`（默认） | prefill GEMM + decode GEMV |
| **CPU 整数**（`backend/cpu/kernels/i8.cpp`） | 直接读 GGUF block，AVX2 提 4/5/6-bit | 运行时对称 int8（`cpu_xq`） | `PF_DP4A=1`（默认） | Q4_K/Q5_K/Q6_K 的 prefill/decode |
| **oneDNN int8** | 主机转成行主序 int8 `[N][K]` + 每行 scale | 每行对称 int8，按 call 量化一次 | `PF_GEMM_DNNL=1`（默认） | 模式 2 chunk-batched prefill，以及模式 1 |
| **fp32/融合反量化** | 直接读 GGUF block；GPU `dequant_sb_lane`，CPU `qgemv_sb_*`（SIMD 反量化+点积） | fp32 | `PF_DP4A=0`，或 `PF_DP4A_DEC=0` 的 decode | 回退/严格测试/Q8_0 |

`PF_DP4A_DEC=0` 只关闭 decode 的 int8 GEMV，prefill 仍走 int8。

CPU 后端默认走 **整数 int8** 路径，但**不构建 `w8t` 副本**（那是 GPU 的 SIn 格式）：`i8.cpp`
按 32 值组从 block 里取无符号权重、用 `maddubs`/`madd` 与 `x8` 做整数点积，再按组做
`sx*(d*sc*Σq·qx − dmin*m*Σqx)` 修正（Q6 为 `sx*d*sc*(Σq·qx − 32·Σqx)`，16 宽组映射到 x8 的
32 宽组两个半区）。Q8_0 与 `PF_CPU_ISA=scalar` 回退到融合 fp32（`qgemv_sb_*`）。见
[architecture.md §11](../architecture.md) 与 [03-kernels.md](03-kernels.md)。

---

## 2. ggml K-quant block 布局（`quant.h`）

```
QK_K = 256, QK8_0 = 32, K_SCALE_SIZE = 12

block_q4_K  { uint16 d; uint16 dmin; uint8 scales[12]; uint8 qs[128]; }   // 144 B
block_q5_K  { uint16 d; uint16 dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]; } // 176 B
block_q6_K  { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; uint16 d; }  // 210 B
block_q8_0  { uint16 d; int8 qs[32]; }                                   // 34 B
```

* `d` / `dmin` 是 fp16 的 super-block scale/min scale。
* Q4_K/Q5_K 的 8 个子块 `(scale,min)` 以 6 bit 打包在 `scales[12]`，用 `get_scale_min_k4` 解出。
* 每 256 元素一个 super-block，其中 8 个子块各 32 元素。

`quant.h` 提供：

* fp16↔fp32：`ggml_half_to_float` / `ggml_float_to_half`（手写位操作，主机/设备通用）。
* `dequantize_block_q4_K/q5_K/q6_K/q8_0` 与通用 `dequantize_row(type, data, y, n)`。
* `quant_row_bytes(type, n)`。

这些参考反量化用于主机端校验、oneDNN 权重转换和 `PF_SI4` 的重新量化。

---

## 3. SIn 权重格式（`w8.h` / `w8.cpp`）

### 3.1 动机

GGUF K-quant 布局是为“逐字节标量反量化”设计的。在 Iris Xe-LP 上这条路径受指令数和 load sector
限制，而硬件 DP4A 的原始吞吐是 FP32 FMA 的约 4 倍。SIn（"sycl-infer n-bit"）**保留 GGUF 中的整数
原值**（0..15 / 0..31 / 0..63），按原生位宽打包，只改变布局与 scale/min 的表示：

```
w        = scale * q - min
sum_k w*x = sx * (scale * sum_k (q*qx) - min * sum_k qx),   x ≈ sx*qx
```

### 3.2 布局：group-major + 行块

```
行块 RB = 128 行，组大小 G（Q4_K/Q5_K: 32，Q6_K: 16）
vals[((rb*MG + g)*RB + ri) * GB]    打包的 k-bit 无符号值
meta[((rb*MG + g)*RB + ri)]         uint32 {fp16 scale, fp16 min}
MG = K/G 每行组数,  GB = G*k/8 每组字节数
```

`kRB = 128`。`rb = r / 128`、`ri = r % 128`，目标偏移 `o = (rb*MG + g)*128 + ri`。**同一组的 128 个行
块连续存放**，因此一个 warp 读 32 个连续行的同一组时访问的是 32 个连续的组槽 —— 这就是 GEMV/GEMM
合并访问的基础。

每组字节：Q4_K = 16，Q5_K = 24（含 4 字节 padding，保持 8 字节对齐），Q6_K = 12。

### 3.3 打包细节

* `pack_nib16`：16 值 → 8 字节，字节 `b` 低半字节存值 `b`，高半字节存值 `b+8`。
* `pack4`：32 值 → 16 字节（两个 nibble 半块）。
* `plane1`：1-bit 平面，bit `i` = 值 `i` 的第 4 位。
* `pack5`：32 值 → 24 字节 = 16 字节 nibble + 4 字节 1-bit 平面 + 4 字节 padding。
* `pack6`：16 值 → 12 字节 = 8 字节 nibble + 4 字节 2-bit 平面（值 `i` 在 bit `2i,2i+1`）。

### 3.4 meta 平面与 Q6_K scale-only 优化

meta 项是 `uint32 = {fp16 scale 低 16 位, fp16 min 高 16 位}`。Q6_K 恒有 `min == 32*scale`，因此 min
冗余。`w8_q6_scale_only_ok`（`w8.cpp:174-194`）逐组检查 `fp16(32*sc) == 32*fp16(sc)` 是否严格成立
（防止 fp16 下溢导致重建权重改变）。成立且非 `PF_SI4` 时 meta 只存 2 字节 fp16 scale
（`meta_elem = 2`），kernel 用 `mw = 32*sw` 推导 min。注释记录：LM head 是最大的 Q6_K 张量，每 token
读 242 MB 其中 60 MB 是 meta，减半收益显著。

### 3.5 `PF_SI4`

`w8_force4()` 读取一次 `PF_SI4`。打开时：

* `w8_effective_type` 恒返回 12；
* 尺寸函数按 `G=32, gb=16, meta=4B` 计算；
* `w8_q6_scale_only_ok` 返回 false；
* `w8_repack` 先把整行反量化，再重化为 **非对称 4-bit 组（每组 32 值）**：
  `sc = (hi-lo)/15`、`min = -lo`、`q = clamp(round((v-lo)/sc), 0, 15)`，kernel 仍按 `w = scale*q - min`
  重建。默认（关闭）则按原生位宽精确拷贝 GGUF 整数。

### 3.6 `w8_repack` 与尺寸函数

`w8_repack(type, src, K, N, vals_out, meta_out, scale_only)`（`w8.cpp:196-276`）要求 `K % 32 == 0`
且 type ∈ {12,13,14}。逐行逐组提取 `q[32], scale, min` 并打包。

尺寸函数（`w8.h` / `w8.cpp:153-167`）：

* `w8_vals_bytes = N * (K/G) * gb`
* `w8_meta_count = N * (K/G)`
* `w8_meta_bytes = meta_count * (scale_only && !force4 && type==14 ? 2 : 4)`
* `w8_group_size(type)`：14 → 16，否则 32
* `w8_words_per_group = group_size / 4`

### 3.7 `w8t`（`w8.h:49-63`）

```cpp
struct w8t {
    uint8_t * vals; uint32_t * meta;
    int32_t K, N, bits;
    uint32_t type; int32_t meta_elem; // 2 或 4
    bool ok() const { return vals != nullptr; }
};
```

`meta_elem` 是逐张量的运行时值，kernel 用循环不变量分支处理，不需要为 scale-only 增加模板实例。

---

## 4. 激活量化（`xq_launch`，`xq.cpp`）

### 4.1 方案

对称 int8，**每 32 个值一个 scale**；同时记录每 16 值的整数和 `sum(qx)`，用于在整数域做权重 min
修正：

```
sum_k w_k x_k = sx * (sw * sum_k qw_k qx_k - mw * sum_k qx_k)
```

### 4.2 布局与发射

`G = 32`，`groups = K/32`，`TB` = 本次 call 的 token 数。`nd_range<1>(TB*groups*32)`，work-group 32，
`reqd_sub_group_size(32)`：一个子组负责一个 `(token t, group g)`，lane 是组内值索引。

```
o = (g*TB + t) * 32          // x8
xmeta[g*TB + t] = float2(scale, 0)
xsumq[g*TB + t] = int2(half0_sum, half1_sum)   // 每 16 值一个
```

注意：**激活 scale 是 fp32**（`xmeta` 是 `sycl::float2`），因为激活可能小到 ~1e-5，fp16 scale 会进入
次正规甚至下溢为零而破坏重建；权重 scale 保持 fp16（小权重只有小绝对误差）。

### 4.3 计算

* 读 `v = x[t*x_stride + g*32 + lane]`；若 `up` 非空则 `v = silu(v)*up[...]`（ffn_down）。
* 子组内求 `mx = sg_max(|v|)`（XOR butterfly），`scale = mx>0 ? mx/127 : 1`。
* `qv = round(v/scale)`，clamp 到 `[-127, 127]`（对称，不产生 -128）。
* 16-lane 半组内用 XOR mask 1,2,4,8 求 `sum(qv)`；`permute_group_by_xor(sg, s, 16)` 得到另一半的和；
  lane 0 写 `int2`。

`TB` 语义：chunked prefill 为 `kMaxT`，decode 为批大小，模式 2 为整个扁平 token 数。`xq_launch` 有意
不使用 `info->n_real`（注释解释：若按 `n_real` 门控，模式 2 里第一个 chunk 之后的 token 会保持未量化）。

---

## 5. DP4A 数学与辅助（`dp4a.h`）

DP4A 指令做 `int8 × int8 → int32` 的四路点积。项目用**有符号 x × 无符号 w** 的变体：

```
d = dp4a(x_signed, w_unsigned, acc)
```

`dp4a.h` 提供可移植的包装。SIn 的展开辅助（`kernel_utils.h`，`si::kd`）：

* `w8_sw_mw<QT>`：读 `(sw, mw)`；`QT==14 && melem==2` 时 `mw = 32*sw`。
* `w8_nib_half<GB>`：8 字节 → 4 个 word，值集合 {0..3, 8..11, 4..7, 12..15}；`GB%8==0` 时用 8 字节加载。
* `w8_expand_half<QT>`：为 Q5_K 补充 1-bit 平面、为 Q6_K 补充 2-bit 平面。
* `w8_group_expand<QT>`：展开整组。
* `w8_xword<QT>(j)`：权重 word 槽 `j` 对应的 x word 索引（`j==1↔2, j==2↔1, else j`）。

---

## 6. oneDNN 路径（`dnnl_gemm.{h,cpp}`）

### 6.1 启用与目的

`dnnl_gemm_enabled()` 除非 `PF_GEMM_DNNL=0`，否则为真。`use_dnnl = pf8 && dnnl_gemm_enabled()`。
它把模式 2（chunk-batched prefill）以及模式 1 的 GEMM 交给 oneDNN GPU int8 matmul。

### 6.2 权重转换

每个 GGUF K-quant 张量在主机侧逐行反量化，重化为**行主序 int8 `[N][K]`**，权重 scale 是
**每 32 值一组**的 f16 `[K/32][N]`（对应 oneDNN 的 grouped WEIGHTS scale，`mask=1, groups={32,1}`，
`"ba"` 权重布局 + f32 dst）——与 K-quant 原生网格一致，比原来的每行 scale 精度高得多。
`add_weight`（`dnnl_gemm.cpp:233-317`）：

* 拒绝非 12/13/14、`K<=0`、`N<=0`、`K%32!=0`、`K > kActMaxK=32768`（27B 的 `n_ff=17408`）；
* 用 `hardware_concurrency` 个线程并行反量化/量化；
* 上传设备并创建 `"ba"` memory；
* 预先为 `M = kMaxT..cap_M`（步长 `kMaxT`）创建所有 primitive，标记 `w.ok` 以便引擎在不支持时回退。

### 6.3 执行

`impl`（`dnnl_gemm.cpp:136-190`）持有：

* 激活 scratch `ax [cap_M][cap_K]` int8、`axs [cap_M]` fp32，`cap_M = kMaxB*kMaxT = 512`；
* 整数输出 `acc [M][N]` int32，容量 `cap_M*6144`（模型最宽的 N 是 wqkv）；
* `weights` map（key = `w8t.vals` 指针 → `{int8 [N][K], scales, memory}`）；
* `prims` map（key = `(M,K,N)` → `{matmul, src, dst}`）。

matmul 本身是 `src [M,K] s8 "ab"`、`weights [K,N] s8 "ba"`、`dst [M,N] s32 "ab"`，**不带 scales
attribute**：整数累加保持精确，每行 scale、residual 和 alpha 全部由 fp32 epilogue kernel 应用
（`out[m][n] = alpha*sx[m]*sw[n]*acc[m][n] + residual`）。N/out_stride/out/residual 对齐时用 float4
版本，否则标量。

`quantize()` 每个 call 调一次，激活在该 call 的所有张量间共享；ffn_down 的输入在量化时应用
`silu(gate)*up`。

### 6.4 为什么不能录制进图

oneDNN primitive 无法被 SYCL command graph 捕获。因此：

* `record_forward` 里有 `dnnl_call` 分支：当 `use_dnnl && mode != 0 && !single`、call 有 xq 计划、
  且该 call 的每个 w8 segment 都有匹配 K 的已转换权重时，逐 call `dnnl->quantize` + `dnnl->gemm`；
  不支持的形状回退为 `xq_launch` + `dp4a_gemm_launch`。
* `build_graphs` 在 `use_dnnl` 时**不**录制模式 2 的 `pfb_vars_` 命令图，只保留 `ntok` 供
  `batched_prefill_fit` 选择尺寸。模式 2 因此总是直接（非图）重放，见
  [04-engine.md](04-engine.md)。

`PF_GEMM_DNNL=0` 恢复录制的 DP4A chunk-batched 路径，被注释称为 “bit-identical dp4a validation”。

---

## 7. 相关环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `PF_DP4A` | on | `0` 强制 fp32；否则构建 SIn 副本并录制 SI8 图 |
| `PF_DP4A_DEC` | on | `0` 只把 decode 退回 fp32（prefill 仍 int8） |
| `PF_META` | off | 构建 Q4_K/Q5_K 的 fp32 `(scale,min)` 旁路数组（实测净损失，约 260 MB） |
| `PF_SI4` | off | 全部重化为 4-bit SIn，副本减半、精度略降 |
| `PF_W4` | on | Q4_K 走原生 4-bit 权重路径（§9）；`0` 恢复纯 int8（每 32 组权重 scale 仍生效） |
| `PF_GEMM_DNNL` | on | `0` 关闭 oneDNN，改用 DP4A chunk-batched |

### `PF_META` fp32 side array

`build_meta32`（`engine.cpp:303-351`）只为 Q4_K(12)/Q5_K(13) 构建 `sycl::float2` 设备数组
（`N*(K/32)` 项，144/176 字节 super-block 解码），按主机张量指针登记在 `meta32_`。它被挂到
`gemv_seg::meta32`，供 fp32 decode GEMV 跳过 6-bit packed scale 解码。实测为净损失：packed scale 与
权重共享 cache line，而独立数组增加一个内存流。

---

## 8. 精度与正确性说明

* **权重值**默认与 GGUF 整数逐位一致，只有 scale/min 的表示被重新量化到 fp16（u4 路径同样如此，
  见 §9：唯一的误差来源是 f16 元数据，实测每行相对 L2 **0.077%**，而 int8 的每行重化是 **0.98%**）。
* **激活**是对称 int8，每 32 值两个额外量：fp32 scale 与每 16 值整数和；round 为 half-to-away，
  clamp 到 `[-127,127]`。
* **DP4A vs oneDNN**：数值上不同（DP4A 用每 32 值激活 scale + 精确整数 dp4a 累加；oneDNN 用每行
  scale + 精确 s32 matmul + fp32 epilogue）。引擎把 `PF_GEMM_DNNL=0` 当作 DP4A 验证配置。
* 严格 kernel/端到端测试通过 `setenv("PF_DP4A","0",1)` 验证 fp32 路径。

---

## 9. 原生 4-bit（u4）权重路径（`w4.{h,cpp}`、`w4_gemv.cpp`）

### 9.1 动机

GGUF 的 K-quant 已经把每个权重放在**每 32 值一组**的均匀网格上：`w = step_g*q - offset_g`，
`q` 是该类型的原生位宽，`(step_g, offset_g) = (d*sc_j, dmin*m_j)` 直接来自 super-block。因此
**保留 `(q, step_g, offset_g)` 是无损的**，而且比 int8 小 1.6 倍（`K*N/2 + (K/32)*N*4` 字节 vs
`K*N`）。实测每行相对 L2（模型自身反量化值为基准）：

| 表示 | rel L2 |
|---|---|
| int8 per-row（原转换） | 0.98 % |
| u4 + 重算每组 min/max | 4.39 % |
| **u4 + 原生 `(step, offset)`，f16 元数据** | **0.077 %** |
| u4 + 原生 `(step, offset)`，f32 元数据 | 0 |

也就是说 4-bit 不只是省内存，**精度还严格优于 int8**：它不引入重化误差，只是把元数据舍入到 f16。
27B 的 Q4_K 占多数，这是它能塞进两张 A770 的关键。

### 9.2 布局（`w4t`，`w4.h`）

| 平面 | 布局 |
|---|---|
| `vals` | `vals[(n*K + k) >> 1]` 的 nibble（`k & 1`），u4、K 为内层、低 nibble 在前 |
| `scale` | `scale[g*N + n]` f16 = `step`，`g = k/32` |
| `off` | `off[g*N + n]` f16 = `offset`（修正系数） |

两个独立的 f16 平面：oneDNN 的 grouped-scale memory 必须是恰好这个顺序的连续 f16 张量（已实测验证），
而修正项把 `off` 当作 `{NG,N}` 的 f16 矩阵读取。

### 9.3 prefill：oneDNN u4 GEMM + 修正项

权重作为 `{K,N}` `format_tag::ba`，配 grouped f16 WEIGHTS scale（= `step`，mask 在 K、
`groups={32,1}`）与 grouped f16 SRC scale（mask 在 K、`groups={1,32}`），**dst 为 f32**：

```
out = oneDNN(u4 weights, grouped f16 scales = step) + correction
correction[m][n] = Σ_g off[n][g] * XS[m][g]        # XS[m][g] = 该组 32 个激活之和
```

因为 `w = offset + step*q`，零点被移进一个小的 `M × (K/32) × N` 项，而不必使用 oneDNN 的 grouped
zero-points（其布局未通过验证）。`w4_xs_launch` 算 `XS`，修正由 `w4_epilogue_launch` 完成。

### 9.4 decode：u4 GEMV（`w4_gemv.cpp`）

`w4_split_act_launch`（按偶/奇 k 拆分激活，供 4-bit 交错布局使用）+
`w4_gemv_launch`（把 `(g,n)` 的 `step`/`off` 分阶段进 SLM、16 字节 nibble 载入、`dp4a` 累加）。
`XS` 与偶/奇拆分不再由独立 kernel 产生：`act_quant_grp_launch` 在同一个 kernel 里一并写出
`axg`/`asa`/`XS`/`axe`/`axo`（每 32 lane 一个 (行,组)，max 用子组归约），所以一次 call 只发一个
激活 kernel 而不是三个。SLM 里的 scale 分块按 **g 外/行内** 索引（相邻 lane 读相邻 `n`），
对 u4 的 `ffn_down` 形状实测 `174 -> 288 GB/s`。

### 9.5 与 int8 / SIn 的关系

* **Q4_K 走 u4，IQ4_XS/IQ4_NL 走 codebook u4（见 §10）**。Q5_K/Q6_K 是 5/6-bit、IQ3_S 是
  codebook 但位宽不同，都无法无损进 u4/oneDNN，它们继续走 SIn/DP4A 或 fp32——27B 是混合量化，
  所以每个张量各自选路径（启动日志会打印 `device N weights: X u4, Y codebook, Z int8`）。
* **激活量化也按 32 值一组**（`act_quant_grp_launch`）：只有整行一个 scale 时，激活的 ~0.9% 误差会
  盖过 4-bit 权重的精度优势。int8 的 oneDNN matmul 只接受每行一个 SRC scale，所以每行版本仍然保留
  （一次 call 可能混有 u4 与 int8 段）。
* `PF_W4=0` 恢复纯 int8（每 32 组的权重 scale 仍然生效，那不是 u4 专有的）。
* 各类型的实际损失用 `test_quant_audit` 逐类型度量；u4 打包往返用 `test_w4`，u4 vs int8 的逐张量
  差异用 `test_w4_vs_i8`，u4 GEMM vs 真实反量化权重用 `test_w4_gemm`。

## 10. codebook 4-bit（IQ4_XS / IQ4_NL，`PF_CB4`，默认开）

### 10.1 动机

IQ4_XS / IQ4_NL **不是网格**：它们的值是

```
w = d * (ls - 32) * kvalues_iq4nl[q]        # q 是 4-bit 索引
```

其中 `kvalues_iq4nl[16]` 是固定的 int8 码本（`quant.h:218`），`d` 是每 256 值的 f16，
`ls` 是每 32 值的 6-bit scale。所以**只要存 4-bit 索引 + 每 (行,组) 的 f16 `d*(ls-32)`，
native 值就是精确的**——不是重化（requant），只是换存储。

| 表示 | 字节/权重 | dev0 占用 |
|---|---|---|
| int8（原转换） | 1.0625 | 12155 MiB |
| **codebook u4** | **0.5625** | **9638 MiB（−2.46 GB）** |

（dev1 11219 → 9354 MiB。）注：IQ4_XS 的 native 值本身就是 int8 网格上的精确点
（`dl*lut` 是 `dl` 的整数倍），所以原来的 int8 转换对它几乎无损（~0.05% f16 舍入）——
**这条路的收益是字节数，不是精度**（`test_w4_vs_cpuref` 的 `mean|diff|` 0.0375 → 0.0479，
两者 argmax 都 SAME）。

### 10.2 布局（`cb4t`，`w4.h`）

| 平面 | 布局 |
|---|---|
| `idx` | `idx[(n*(K/2)) + g*16 + b]`，每 32 值一组 16 字节；**native 元素序**：元素 `e<16` 是 byte `e` 的低 nibble，`e>=16` 是 byte `e-16` 的高 nibble |
| `scale` | `scale[g*N + n]` f16 = `d*(ls-32)`，与 u4 的 step 平面同构（oneDNN grouped scale 直接用） |

`cb4_pack` 只做重排（`memcpy` 每组的 16 字节 nibble + 抽取 scale），不碰任何数值。

### 10.3 decode：LUT 展开的 dp4a GEMV（`cb4_gemv_launch`）

结构同 `i8_grp_gemv`（g 外/行内 SLM staging、`asa`/`XS`/XOR 偏差校正），只是权重字由
nibble 经 16 项 SLM LUT 展开成 int8 再 `dp4a`：`w[j]` 取 byte `j*4..j*4+3` 的低 nibble、
`w[4+j]` 取它们的高 nibble（位移是 `8b` 与 `8b+4`，写错会被"全 nibble 相同"的测试掩盖）。
实测每张量比 int8 GEMV 快 11-20%（head 3.31→2.63 ms），但有效 GB/s 更低
（272 vs 408），因为它 read 一半字节却受 ALU/延迟限制更多。

### 10.4 prefill：展开到复用 scratch + 现有 int8 primitive

oneDNN 只认线性 `u4` 或 `s8`（`dnnl_common_types.h` 里有 `s4/u4/f4_e2m1/f4_e3m0`，**没有 u5/u6**），
所以 codebook 的 prefill 只能喂 int8。做法是**把该张量的索引经同一 LUT 展开成 int8，写进一块
按最大单张量分配的 scratch（~89 MB）**，再用**已有的 int8 primitive**：

* 前提已验证：覆盖权重缓冲后重新 `execute`，结果跟着变（2→64、3→96、4→128），
  即 **oneDNN 每次 execute 都重读用户权重 memory**，不做内部 reorder 缓存。
* 代价是每 pass 多一遍展开流量（读 0.5625 + 写 1.0 B/w，IQ4_XS+NL 合计 ~13.3 GB → ~31 ms/pass）。
  第一版逐字节标量写让 prefill 掉到 940 t/s，改成"16 字节载入 + 两个 16 字节写出"后回到 ~1340 t/s。
* `cb4_expand_launch` 的元素序必须与 §10.2 一致（`w[0..3]` 是元素 0..15、`w[4..7]` 是 16..31）。

### 10.5 实测

| | `PF_CB4=0` | `PF_CB4=1` |
|---|---:|---:|
| tg128 | 12.82 t/s | **13.49（+5.2 %）** |
| pp512 | ~1500 t/s | **~1290（−13 %）** |
| argmax vs fp32 参考 | SAME | SAME |

`PF_CB4=0` 退回 int8 转换（A/B 用）。tg 收益受限于 LUT kernel 是半 ALU/延迟受限，
以及 codebook 类型只占每 token 字节的 19 %。

## 11. 原生 5-bit（Q5_K，`PF_K5`，默认开）

Q5_K 的原生栅格是

```
q5 = (qs nibble) | (qh bit << 4)   ∈ [0,31]
w  = d·sc_j · q5 − dmin·m_j          （每 32 个一组，6-bit (sc,m) 由 get_scale_min_k4 取出）
```

它**不是** 4-bit 栅格：UD-Q4_K_M 里 Q5_K 占 7.03 G 权重（本模型权重的 35 %），
曾经只能 int8 转换（1.0625 B/w）。丢掉第 5 位（`PF_W4_ALL` 的 u4 重化）会损失该类型的
原生分辨率（~5 % 相对误差），所以这里**保留 5 位**，只换布局：

| 平面 | 布局 | 大小 |
|---|---|---|
| `vals` | 4-bit nibble，**交错**（byte k = 元素 2k 低 nibble / 2k+1 高 nibble） | K/2 |
| `hi` | 第 5 位，**按 split 元素序**：每 32 组 4 字节，bit 0-15 = 偶元素 2i→bit i，bit 16-31 = 奇元素 2i+1→bit i | K/8 |
| `scale` / `off` | `[g][n]` f16 step = d·sc、off = −dmin·m（与 u4 同一约定） | 各 K/32·N·2 |

即 0.5 + 0.125 + 0.125 = **0.75 B/w，native 值精确**（只有 step/off 的 f16 舍入，
与 u4 路径同量级）。

为什么 `hi` 用 split 序而不是元素序：解码 GEMV 复用 u4 的 **偶/奇激活平面**
（`axe`/`axo`），一个 dp4a 操作数的 4 个元素正好是同一 split 序的相邻 4 位——
于是 4 bit 经 16 项 SLM LUT 展开成 4 字节掩码后，`q5 = lo4 | (mask << 4)`
（OR 即加，lo4 < 16），代价只有每 32 值 8 次 LUT 读 + 8 次 OR + 8 次 shift，
**内核仍纯带宽受限**：实测每张量 400-443 GB/s（100-109 % 的 405 GB/s 上限），
而 int8 GEMV 只有 82-92 %；同形状比 int8 快 1.5x（`dev/bench_native56.cpp`）。

### 11.1 decode / prefill 两条路

* **decode**：`k5_gemv_launch`（`w4_gemv.cpp`），0.75 B/w 直接读，无展开。
* **prefill**：oneDNN 只认 `u4`/`s8`，所以 `k5_expand_launch` 把两个平面重排成元素序 int8
  写进与 cb4 共享的 scratch（~89 MB），再跑**已有的分组 scale int8 primitive**；
  `off` 项用 u4 的修正 epilogue（`w4_epilogue_launch`）加回。代价是每 pass
  多 0.625（读）+ 1.0（写）B/w 的**串行**流量（M=512 时 GEMM 自身的权重读被算力掩盖，
  展开流量则不能），实测 pp512 −15..−20 %；`PF_K5_NOCORR=1` 可去掉修正项做二分定位
  （约占其中 5 %）。
* 展开内核本身必须是**向量写**：早期版本每 work-item 8 次 32-bit 分散存储只能到
  200 GB/s（纯指令/存储受限），改成两个 `uint4` 存储后 425 GB/s（上限的 105 %）。
  注意 2D `range` 会打乱 work-item→数据映射，反而更慢（98 GB/s），保持一维。

### 11.2 实测（27B，2×A770，`--layer-map 0-31:gpu.0,32-63:gpu.1`）

| | `PF_K5=0` | `PF_K5=1` |
|---|---:|---:|
| tg128 | 13.48 t/s | **14.34（+6.4 %）** |
| pp512 | 1438 t/s | ~1140（−20 %） |
| dev0 dnnl 权重 | 9637.7 MiB | **9072.1** |
| dev1 dnnl 权重 | 9354.1 MiB | **7824.5** |
| vs fp32 参考 mean\|diff\| | 0.0375 | **0.0354**（更准） |

精度反而更好（保留 native 值），设备显存合计 −2.1 GB。pp 的代价是展开流量，
`--kv-type i4` 或后续把展开分块进 L2 才能收回。
