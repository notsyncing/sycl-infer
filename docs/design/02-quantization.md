# 设计 02：量化格式与 int8 计算路径

覆盖 `src/common/quant.h`、`src/common/w8.{h,cpp}`、`src/common/dp4a.h`、`src/backend/gpu/kernels/xq.cpp`、
`src/backend/dnnl_gemm.{h,cpp}`。kernel 侧的 GEMV/GEMM 实现见 [03-kernels.md](03-kernels.md)。

---

## 1. 概览

引擎有 GPU 的 SIn/DP4A、CPU 的 GGUF 整数 int8、可选 oneDNN，以及融合 fp32 回退：

| 路径 | 权重格式 | 激活 | 启用 | 适用 |
|---|---|---|---|---|
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

每个 GGUF K-quant 张量在主机侧逐行反量化，重化为**行主序 int8 `[N][K]`**，每输出行一个对称 scale，
对应 oneDNN 的 `"ba"` 权重布局（GPU int8 matmul 全速读取）。`add_weight`（`dnnl_gemm.cpp:233-317`）：

* 拒绝非 12/13/14、`K<=0`、`N<=0`、`K%32!=0`、`K > kActMaxK=4096`；
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
| `PF_GEMM_DNNL` | on | `0` 关闭 oneDNN，改用 DP4A chunk-batched |

### `PF_META` fp32 side array

`build_meta32`（`engine.cpp:303-351`）只为 Q4_K(12)/Q5_K(13) 构建 `sycl::float2` 设备数组
（`N*(K/32)` 项，144/176 字节 super-block 解码），按主机张量指针登记在 `meta32_`。它被挂到
`gemv_seg::meta32`，供 fp32 decode GEMV 跳过 6-bit packed scale 解码。实测为净损失：packed scale 与
权重共享 cache line，而独立数组增加一个内存流。

---

## 8. 精度与正确性说明

* **权重值**默认与 GGUF 整数逐位一致，只有 scale/min 的表示被重新量化到 fp16。
* **激活**是对称 int8，每 32 值两个额外量：fp32 scale 与每 16 值整数和；round 为 half-to-away，
  clamp 到 `[-127,127]`。
* **DP4A vs oneDNN**：数值上不同（DP4A 用每 32 值激活 scale + 精确整数 dp4a 累加；oneDNN 用每行
  scale + 精确 s32 matmul + fp32 epilogue）。引擎把 `PF_GEMM_DNNL=0` 当作 DP4A 验证配置。
* 严格 kernel/端到端测试通过 `setenv("PF_DP4A","0",1)` 验证 fp32 路径。
