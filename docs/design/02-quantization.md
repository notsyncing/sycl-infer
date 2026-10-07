# 设计 02：量化格式与 int8 计算路径

覆盖 `src/common/quant.h`、`src/common/w8.{h,cpp}`、`src/common/w4.{h,cpp}`、`src/common/dp4a.h`、
`src/common/cpu_isa.h`、`src/backend/gpu/kernels/xq.cpp`、`src/backend/dnnl_gemm.{h,cpp}`，
以及 CPU 侧的 `src/backend/cpu/kernels/{dp4a,i8,xq}.cpp`。kernel 侧的 GEMV/GEMM 发射几何与
SLM 分块见 [03-kernels.md](03-kernels.md)；KV 缓存的量化格式见
[05-kv-cache.md](05-kv-cache.md)。

---

## 1. 概览

引擎有 GPU 的 SIn/DP4A、CPU 的 GGUF 整数 int8、可选 oneDNN、原生位宽 4/5/2-bit 权重存储，以及融合
fp32 回退：

| 路径 | 权重格式 | 激活 | 启用 | 适用 |
|---|---|---|---|---|
| **u4 原生 4-bit**（GPU，Q4_K） | `w4t`（原生 q + 每 32 组 (step, offset)） | 每 32 组对称 int8（`act_quant_grp_launch`）+ 偶/奇 k 平面 | `PF_W4=1`（默认，且张量是 Q4_K） | `--layer-map` 的 decode GEMV + prefill GEMM，见 §9 |
| **k5 原生 5-bit**（GPU，Q5_K） | `k5t`（4-bit nibble + 1-bit 平面 + 每 32 组 (step, offset)） | 同上 | `PF_K5=1`（默认） | 同上，见 §11 |
| **codebook u4**（GPU，IQ4_XS/IQ4_NL） | `cb4t`（4-bit 索引 + 每 32 组 f16 scale） | 每 32 组对称 int8（无偶/奇平面） | `PF_CB4=1`（默认） | 同上，见 §10 |
| **w2 原生 2-bit**（GPU，仅 MTP draft） | `w2t`（2-bit 平面 + 每 32 组 (step, offset)） | 每 32 组对称 int8 | `PF_MTP_HEAD_W2` / `PF_MTP_LAYER_W2`（**默认关**） | draft 的 head readout / MTP 层，见 §12 |
| **SIn + DP4A**（GPU） | `w8t`（GGUF 整数原值，按位宽打包） | 运行时对称 int8（`xq_launch`） | `PF_DP4A=1`（默认；**仅单设备**） | prefill GEMM + decode GEMV |
| **CPU 整数**（`backend/cpu/kernels/i8.cpp`） | 直接读 GGUF block，AVX2 提 4/5/6-bit | 运行时对称 int8（`cpu_xq`） | CPU 分区的层（默认） | Q4_K/Q5_K/Q6_K 的 prefill/decode |
| **oneDNN int8** | 主机转成行主序 int8 `[N][K]` + **每 32 组** f16 scale `[K/32][N]` | 每 32 组对称 int8 + f16 组 scale + 组和 | `PF_GEMM_DNNL=1`（默认） | 单设备的模式 1/2 prefill；`--layer-map` 的全部层 GEMM（含 decode） |
| **fp32/融合反量化** | 直接读 GGUF block；GPU `dequant_sb_lane`，CPU `qgemv_sb_*`（SIMD 反量化+点积） | fp32 | `PF_DP4A=0`，或 `PF_DP4A_DEC=0` 的 decode | 回退/严格测试/Q8_0 |

`PF_DP4A_DEC=0` 只关闭 decode 的 int8 GEMV，prefill 仍走 int8。

**只有 `--layer-map` 的多设备路径才注册原生位宽存储**（`engine::setup_md_dnnl`，
`engine.cpp:1145-1168` 是 u4 → k5 → cb4 → int8 的注册链）。单设备 GPU 跑的是
`w8t` SIn + dp4a GEMV，prefill GEMM 交给单设备那一个 oneDNN 实例（`engine.cpp:368-398`），
所以 `PF_W4` / `PF_K5` / `PF_CB4` 在单设备下没有任何权重被转换。启动日志
`[dev] device N weights: … u4, … k5, … codebook, … int8`（`engine.cpp:1364`）就是这条注册链的计数。

CPU 后端默认走**整数 int8** 路径，但**不构建 `w8t` 副本**（那是 GPU 的 SIn 格式）：`i8.cpp`
按 32 值组从 block 里取无符号权重、用 `maddubs`/`madd` 与 `x8` 做整数点积，再按组做
`sx*(d*sc*Σq·qx − dmin*m*Σqx)` 修正（Q6 为 `sx*d*sc*(Σq·qx − 32*Σqx)`，16 宽组映射到 x8 的
32 宽组两个半区）。选哪条由 plan 的 `gemv_seg::i8` 决定（`engine_graph.cpp:147`：**只有 CPU 分区的
层**置位），于是同一个 CPU 后端同时暴露 `i8_gemv/gemm`（直读 GGUF）与 `dp4a_gemv/gemm`（吃 `w8t`）
两套入口，分派见 `engine_graph.cpp:624-669`。Q8_0 与 `PF_CPU_ISA=scalar` 回退到融合 fp32
（`qgemv_sb_*`）。见 [architecture.md §11](../architecture.md) 与
[03-kernels.md](03-kernels.md)。

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
* Q4_K/Q5_K 的 8 个子块 `(scale,min)` 以 6 bit 打包在 `scales[12]`，用 `get_scale_min_k4`（`quant.h:87-95`）解出。
* 每 256 元素一个 super-block，其中 8 个子块各 32 元素。
* **Q6_K 的 min 是冗余的**：值域是 `w = d*sc*(q-32)`，`min` 恒等于 `32*scale`（§3.4 的
  scale-only 优化就是靠这一条）。
* Q4_K/Q5_K 的 `qs` 是**分裂半区**序（低 nibble = 前 32 个元素，高 nibble = 后 32 个），而引擎所有
  原生位宽存储都把它**重排成交错序**（byte k = 元素 2k 低 nibble / 2k+1 高 nibble），§9/§10/§11 会
  反复用到这个约定。

引擎还会遇到 IQ/Q3_K 系列（`quant.h:181-214`）：

```
block_q3_K   { uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; uint16 d; }   // 110 B
block_iq3_s  { uint16 d; uint8 qs[64]; uint8 qh[8]; uint8 signs[32]; uint8 scales[4]; } // 110 B
block_iq4_nl { uint16 d; uint8 qs[16]; }                                      // 18 B
block_iq4_xs { uint16 d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]; } // 136 B
```

`quant.h` 提供：

* fp16↔fp32：`ggml_half_to_float` / `ggml_float_to_half`（手写位操作，主机/设备通用）。
* `dequantize_block_q4_K/q5_K/q6_K/q8_0` 与通用 `dequantize_row(type, data, y, n)`
  （`quant.h:369-422`，覆盖 type 0/8/11/12/13/14/20/21/23）。
* `quant_row_bytes(type, n)`（`quant.h:424-437`，同一张 type 表）。

这些参考反量化用于主机端校验、oneDNN 权重转换、`PF_SI4` 的重新量化，以及 `PF_W4_ALL` /
`w2_pack_any` 的通用重化。

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
读 242 MB 其中 60 MB 是 meta，减半收益显著（`w8.h:24-28`）。

### 3.5 `PF_SI4` 与"没有原生 SIn 打包"的类型

`w8_force4()`（`w8.cpp:141-147`）读取一次 `PF_SI4`。打开时：

* `w8_effective_type` 恒返回 12（`w8.cpp:149-151`）；
* 尺寸函数按 `G=32, gb=16, meta=4B` 计算（`w8.cpp:153-167`）；
* `w8_q6_scale_only_ok` 返回 false；
* `w8_repack` 先把整行反量化，再重化为 **非对称 4-bit 组（每组 32 值）**：
  `sc = (hi-lo)/15`、`min = -lo`、`q = clamp(lround((v-lo)/sc), 0, 15)`（`w8.cpp:231-249`），
  kernel 仍按 `w = scale*q - min` 重建。默认（关闭）则按原生位宽精确拷贝 GGUF 整数。

**IQ/Q3_K 类型与 `PF_SI4` 无关**：Q3_K(11)、IQ4_NL(20)、IQ3_S(21)、IQ4_XS(23) 没有原生 SIn 打包，
`w8_requant_type`（`w8.h:91-93`）把它们**无条件**映射到 effective type 12，走同一条 4-bit 重化路径
（`w8.cpp:150`、`w8.cpp:209`）。也就是说这些类型在任何配置下都不进 SIn 的 4/5/6-bit 打包；
有原生栅格的只有 Q4_K（以及 IQ4_XS/IQ4_NL/Q5_K，但它们各有 §9/§10/§11 自己的原生存储，不走 SIn）。

### 3.6 `w8_repack` 与尺寸函数

`w8_repack(type, src, K, N, vals_out, meta_out, scale_only)`（`w8.cpp:196-278`）要求 `K % 32 == 0`
且 type ∈ {12,13,14} ∪ `w8_requant_type`（`w8.cpp:201-203`）。逐行逐组提取 `q[32], scale, min` 并打包。

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
不使用 `info->n_real`（注释解释：若按 `n_real` 门控，模式 2 里第一个 chunk 之后的 token 会保持未量化；
另外 `n_real` 必须在 kernel 体内读，host 侧读会在图录制时被按值捕获，而那时 `step_info` 还是零，
`xq.cpp:26-32`）。

CPU 的 `cpu_xq`（`cpu/kernels/xq.cpp:10-45`）是同一格式的 host 版：同样的 group-major 布局
（`x8[(g*TB+t)*32+i]`）、`xmeta` 是交错的 `{scale, 0}` float 对、`xsumq` 两个 int32，同样用
`std::round` + `[-127,127]` clamp（所以两侧的量化结果逐位相同），也**同样不看 `n_real`**。

### 4.4 `do_split`：激活的两种形式

每 32 组的 int8 激活有两个派生的物理形式，`dnnl_gemm::quantize` 的 `do_split` 参数决定这次调用
**写不写**第二个：

| 形式 | 缓冲区 | 谁读 |
|---|---|---|
| 连续（group-major） | `axg [M][K]` + `asa`（f16 组 scale）+ `xs`（组内 int32 和写成 f32） | oneDNN 的 grouped SRC scales、所有分组 GEMV、w2/cb4 |
| 偶/奇 k 拆分 | `axe`/`axo`，`axe[j] = xq[2j]`、`axo[j] = xq[2j+1]` | u4/k5 的 GEMV 与 `nat_gemm`（它们的 nibble 把 k 与 k+1 放进一个字节） |

**核心不变量：任何喂给原生存储 GEMV 的激活都必须用 `do_split=true` 量化。** 少这一次拆分不会报错，
只会静默读到**上一次 call** 留下的 `axe`/`axo`——这就是 MTP 层 `eh_proj` 曾给出 acc 0.08（草稿全是
垃圾）的原因，`eh_proj` 现在的显式写法是 `mtp_layer_w4_ || mtp_head_w4_`
（`engine_mtp.cpp:382`，注释在 `engine_mtp.cpp:377-381`）。同理，prefill 故意用 `do_split=false`
（oneDNN 只读连续形式 + 组和，省掉两次额外的存储），而消费这些平面的路径必须由 `split_valid`
守卫，见 §6.5。

---

## 5. DP4A 数学与辅助（`dp4a.h`）

DP4A 指令做 `int8 × int8 → int32` 的四路点积。项目用**有符号 x × 无符号 w** 的变体：

```
d = dp4a(x_signed, w_unsigned, acc)
```

`dp4a.h` 提供可移植的包装 `si::dp4a_s8u8(a, b, c)`（`dp4a.h:16-21`）：它不是内联汇编，而是 4 次
移位 + 4 次乘法，编译器认得这个模式并发硬件 dp4a（注释记录实测 ~4.8 T-MAC/s，`dp4a.h:4-8`）。
**不要从多个 TU 包含 `sycl/ext/oneapi/dot_product.hpp`**：这个工具链里它的函数不是 `inline`，
会重复定义；自己的静态包装就是为了这个。

SIn 的展开辅助（`kernel_utils.h`，`si::kd`）：

* `w8_sw_mw<QT>`（`kernel_utils.h:149-160`）：读 `(sw, mw)`；`QT==14 && melem==2` 时 `mw = 32*sw`。
* `w8_nib_half<GB>`（`kernel_utils.h:564-578`）：8 字节 → 4 个 word，值集合 {0..3, 8..11, 4..7, 12..15}；`GB%8==0` 时用 8 字节加载。
* `w8_expand_half<QT>`（`kernel_utils.h:581-596`）：为 Q5_K 补充 1-bit 平面、为 Q6_K 补充 2-bit 平面。
* `w8_group_expand<QT>`（`kernel_utils.h:599-604`）：展开整组（Q6_K 只有一半）。
* `w8_xword<QT>(j)`（`kernel_utils.h:606-608`）：权重 word 槽 `j` 对应的 x word 索引（`j==1↔2, j==2↔1, else j`）。

`sg_max<N>`（`kernel_utils.h:545-550`）的 `N` 是**编译期**模板参数：运行时查
`sg.get_local_range()` 会让 shuffle 循环变成动态的，在 GDN 递推里实测慢 ~2 倍（注释在
`kernel_utils.h:541-544`）。

### 5.1 三个存储各自的"无符号"处理

`dp4a_s8u8` 要求权重字节是**无符号**的，所以每个存储要单独处理：

| 存储 | 权重字节 | 需要偏置修正吗 |
|---|---|---|
| SIn `w8t`（Q4_K/Q5_K/Q6_K） | 原生 `q ∈ 0..15 / 0..31 / 0..63` | 否；min 修正走 `sw*d − mw*c` |
| 分组 int8（oneDNN 的 int8 表） | `int8 ∈ [-127,127]`，每 32 组 f16 step | **要**：权重字先 XOR `0x80`（每字节 +128），再用组激活和减掉 `128*sum(x)`（`w4_gemv.cpp:439-445`） |
| 原生 4/5/2-bit（u4/k5/w2） | 原生非负值 | 否（k5 的第 5 位用一次 OR 合并，见 §11） |

分组 int8 的解码 GEMV 因此是 `y[n] = Σ_g asa[g] * step[g][n] * QDOT_g[n]`，**没有 min 项**；
`test_w4_vs_i8` 就是逐张量对这两条路径的。

### 5.2 CPU 侧：同一套格式的第二份实现

`cpu/kernels/dp4a.cpp` 是 `w8t` 的 host 版实现，**自带一份展开辅助**
（`nib_half`/`expand_half`/`expand_group`，`cpu/kernels/dp4a.cpp:20-65`），与设备端
`si::kd` 的版本必须逐位一致：`expand_group` 用同一个 `w8_xword` 置换
（`cpu/kernels/dp4a.cpp:56-63`）把整组展开成**激活序**的无符号字节，然后交给
`isa().dot_i8`。改一侧必须改另一侧，否则 CPU 与 GPU 的 `w8t` 路径会静默分叉。

整数点积按运行期 ISA 分派（`cpu/kernels/common.cpp:531-545`）：

| 条件 | 实现 | 每条指令处理 |
|---|---|---|
| `avx512` + `avx512vnni` | `dot_i8_avx512`（`dpbusd`） | 64 |
| `cpu_has_avxvnni()`（AVX-VNNI） | `dot_i8_vnni`（`dpbusd`） | 32 |
| 其它非 scalar | `dot_i8_avx2`（`maddubs` + `madd`） | 32 |
| `scalar` | 纯 C 循环 | 1 |

`PF_CPU_ISA=scalar|avx2|avxvnni|avx512`（`cpu_isa.h:9`）可强制某一变体。注意 `qgemv_sb` 在
scalar 下是 `nullptr`（`common.cpp:546`），而 `i8_supported` 要求
`isa().qgemv_sb != nullptr`（`i8.cpp:159-161`）——所以 `PF_CPU_ISA=scalar` 会把 CPU 的整数 int8
路径整体关掉、退回融合 fp32。

### 5.3 CPU 整数路径的 prefill 分块

`cpu_i8_gemv`（`i8.cpp:321-339`）是每 token 一行地扫完整个权重行；`TB >= 2` 时改走
`cpu_i8_gemm_tiled`（`i8.cpp:291-317`）：以 `kI8Rt=16` 输出行 × `kI8Tt=32` token 为 tile，
先把 (行块, k 块, 组) 的 `x8`/scale/两个 16 值和**一次性**收集进寄存器结构 `qx_t`
（`i8.cpp:182-186`），再对行 tile 里的每一行做同样的整数点积——权重每行读一次、激活每个行 tile
读一次。注释记录这样做的原因：per-(token,row) 的形式在 `TB=64..384` 时把 prefill 变成权重带宽
问题（一个 CPU 层 128-token batch 实测 ~121 ms，`i8.cpp:166-173`）。逐元素数学与 `i8_row` 完全
相同（组序一致），所以两者逐位相等。

---

## 6. oneDNN 路径（`dnnl_gemm.{h,cpp}`）

### 6.1 启用与目的

`dnnl_gemm_enabled()`（`dnnl_gemm.cpp:293-301`）= 环境变量 `PF_GEMM_DNNL` 未置 0 **且**设备 profile 的
`wt.gemm_dnnl` 为真（`src/device/profiles/arc_a770.cpp:85`、`iris_xe.cpp:98`，两张卡都是 1）——环境变量
是 A/B，profile 是实测默认。

单设备 GPU 走 `use_dnnl = !cpu_mode && !multi_dev && pf8 && dnnl_gemm_enabled()`
（`engine.cpp:368`），那一个 `dnnl_gemm` 实例负责模式 1/2 的 prefill GEMM。
`--layer-map` 的多设备路径另外为每个 GPU 分区建一个 `dnnl_gemm`（`engine.cpp:399-435`、
`engine.cpp:1115`），此时所有层 GEMM（含 decode）都由 oneDNN 服务，且 `pf8` 恒为 false
（`engine.cpp:190`），SIn/dp4a 只作为 oneDNN 不可用时的回退
（`md_xmx → md_int8 → fp32`，`engine.cpp:1611-1618`）。

### 6.2 权重转换

每个 GGUF 量化张量在主机侧逐行反量化，重化为**行主序 int8 `[N][K]`**，权重 scale 是
**每 32 值一组**的 f16 `[K/32][N]`（对应 oneDNN 的 grouped WEIGHTS scale，`mask=1, groups={32,1}`，
`"ba"` 权重布局 + f32 dst）——与 K-quant 原生网格一致，比原来的每行 scale 精度高得多。
`impl::w_entry` 的注释（`dnnl_gemm.cpp:336-340`）记录了这个改动的原因与代价：每行一个 scale 时
**所有类型**（含 Q8_0）的重化误差都是 ~1%（`test_quant_audit` 实测），改成每 32 组后降到 ~0.1%，
代价是 oneDNN 改用 f32-dst kernel（约 +20 % prefill）。

`add_weight`（`dnnl_gemm.cpp:644-763`）：

* 拒绝 type ∉ {8,11,12,13,14,20,21,23}（`quant.h` 能逐行反量化的全部格式）、`K<=0`、`N<=0`、
  `K%32!=0`、`K > kActMaxK=32768`（27B 的 `n_ff=17408`）；
* 用 `min(N, hardware_concurrency())` 个线程并行反量化/量化（每线程一行条带）；
* 权重 scale 用 `ggml_float_to_half`（`std::lround` + `[-127,127]` clamp，与 kernel 侧同一约定）；
* 上传设备并创建 `"ba"` 的 s8 memory 与 `{1, ng*N}` f16 的 scale memory（`PF_DNNL_BLOCKED=1/2/3`
  可强制 `Ab4a`/`Ab8a` 阻塞布局，**仅诊断用**：权重缓冲不会重排成它，结果是错的，
  `dnnl_gemm.cpp:509-512`）；
* 预先为 `M = kMaxT..cap_M`（步长 `kMaxT`）创建所有 primitive，标记 `w.ok` 以便引擎在不支持时回退。

### 6.3 执行

`impl`（`dnnl_gemm.cpp:304-540`）持有：

* 激活 scratch（`dnnl_gemm.cpp:547-564`，`cap_M = kMaxB*kMaxT = 512`、`cap_K = kActMaxK = 32768`）：
  每 32 组形式 `axg`（int8）+ `asa`（f16 组 scale）+ `xs`（组内 int32 和写成 f32）+ `axe`/`axo`
  （偶/奇 k 拆分），加上 f32 累加器 `accf`，以及**只有 `PF_ROWACT=1` 才发射**的每行形式
  `ax`/`axs`/`axsum`；
* 整数输出 `acc`（int32）与 f32 输出 `accf`，容量都是 `acc_cap = cap_M*32768`
  （最宽的 N 是 27B 的 `ffn_gate/up = 17408`，向上取整到 2 的幂，`dnnl_gemm.cpp:552-556`）；
* `weights` map（key = `w8t.vals` 指针 → `{int8 [N][K], scales, memory}`）、`w4weights`/`w2weights`/
  `k5weights`/`cb4weights` 四张原生位宽表（§9-§12）、共享的 `cb4_scratch`；
* `prims` map（key = `(M,K,N)` → `{matmul, src, dst, sscales}`）与 `prims4`（u4 版本）；
* `owned_` 注册表（`dev_alloc`/`dev_erase`）：所有长生命周期的设备分配在创建时登记，
  析构只扫表——新权重存储再也不会因漏改析构而泄漏（cb4/LUT 漏的就是这么补上的）。
  分配时机与数量不变，单函数内平衡的临时量（`dsc`、`dx/dsw/dout`）仍手动管理；
* 两条 oneDNN stream：`st`（稠密 GEMM）与 `st_a`（只服务 attention 的 int8 matmul）。
  两者包同一个 in-order 队列，所以提交仍有序，但在 `st_a` 上等不会把 `st` 上排队的稠密 GEMM
  一起抽干（`dnnl_gemm.cpp:308-312`）。

matmul 本身是 `src [M,K] s8 "ab"`、`weights [K,N] s8 "ba"`、`dst [M,N] **f32** "ab"`，**带两个
grouped scale attribute**：WEIGHTS `mask=1, groups={32,1}`（= step 平面）、SRC `mask=2,
groups={1,32}`（= `asa`），见 `make_prim`（`dnnl_gemm.cpp:497-539`）。因为 scale 由 matmul 在浮点里
应用，累加器只能是 f32；残留的只有 `alpha` 与 residual，由 `epilogue_f32_launch`
（`out = alpha*acc + residual`，`dnnl_gemm.cpp:229-241`）施加。u4/k5 的 prefill 走
`w4_epilogue_launch`（`out = alpha*(acc4 + Σ_g off[g][n]*xs[m][g]) + residual`，
`dnnl_gemm.cpp:161-224`），它同时做零点修正，因此不需要 oneDNN 的 grouped zero-point。

`quantize()` 每个 call 调一次，激活在该 call 的所有张量间共享；ffn_down 的输入在量化时应用
`silu(gate)*up`。**旧的每行 int8 形式已停用**：`act_quant_launch` 只在 `PF_ROWACT=1` 下跑，
理由记在 `dnnl_gemm.cpp:1222-1236`（两个权重路径现在都读每 32 组的形式，这个 kernel 只剩纯
per-call 延迟，27B decode 的 320 个 call 实测 ~13 ms/token）。

### 6.4 为什么不能录制进图

oneDNN primitive 无法被 SYCL command graph 捕获。除了靠 `use_dnnl` 不录图之外，还有三道保险：

* `dnnl_set_capturing(true)` 期间每个 `execute` 站点都先调 `dnnl_capture_guard()`
  （`dnnl_gemm.cpp:287-291`，调用点如 `dnnl_gemm.cpp:1408`），它直接抛异常，让调用方放弃录制并
  退回直接重放——否则会录下一条**悄悄少了工作**的 pass。
* `record_forward` 的 `dnnl_call` 判定（`engine_graph.cpp:542-582`）：必须有绑到本 call 计算后端的
  `dnnl_gemm *D`，且**不是**（`mode == 0 && !multi_dev`）——即单设备的 decode 保留 dp4a GEMV，
  其余（模式 1/2，以及多设备的 decode）走 oneDNN；call 必须有 xq 计划；call 的每个 w8 segment
  都必须在 `has_weight`/`has_weight_w4`/`has_weight_k5`/`has_weight_cb4` 里有 K 匹配的条目，
  多设备下**所有** segment 都要（它们没有 w8 视图）。任一条件不满足就退回
  `xq_launch` + `dp4a_gemm_launch`。
* `build_graphs` 在 `use_dnnl` 时**不**录制模式 2 的 `pfb_vars_` 命令图，只保留 `ntok`
  （`{2,4,8,16}×kMaxT`）供 `batched_prefill_fit` 选择尺寸（`engine_graph.cpp:1790-1808`）。
  模式 2 因此总是直接（非图）重放，见 [04-engine.md](04-engine.md)。

`PF_DNNL_NOWARM=1` 跳过 `warmup()`（`dnnl_gemm.cpp:396-398`、`408-409`），后者在引擎构造时把每个
primitive 与两个 SYCL kernel 各跑一次，把一次性 kernel 加载挡在第一个请求之前。

`PF_GEMM_DNNL=0` 恢复录制的 DP4A chunk-batched 路径，被注释称为 “bit-identical dp4a validation”。

### 6.5 各权重存储消费的激活形式与 `split_valid` 守卫

`quantize()` 把 `do_split` 记在 `p->split_valid`（`dnnl_gemm.cpp:1246`），而**每一个**读原生
激活视图的分支都必须先看它：

| 分支 | 需要的激活形式 | 守卫 | 位置 |
|---|---|---|---|
| `gemm_w4` → `nat_gemm_launch(FMT 0)` | `axe`/`axo` + `axg`/`asa`/`xs` | `M>=2 && p->split_valid && …` | `dnnl_gemm.cpp:1439-1443` |
| `gemm`（k5）→ `nat_gemm_launch(FMT 1)` | 同上 | `p->split_valid && p->axe && …` | `dnnl_gemm.cpp:1266-1270` |
| `gemm`（cb4）→ `nat_gemm_launch(FMT 2)` | `axg`/`asa`/`xs` | `p->split_valid && …` | `dnnl_gemm.cpp:1323-1327` |
| `gemm`（int8）→ `nat_gemm_launch(FMT 3)` | `axg`/`asa`/`xs` | `M >= 2 && p->split_valid && …` | `dnnl_gemm.cpp:1368-1380` |
| u4 / k5 / cb4 的 M=1 GEMV | `axe`/`axo`（u4、k5）、`axg`（cb4） | decode 恒 `do_split=true` | `dnnl_gemm.cpp:1487-1494`、`1259-1264`、`1316-1319` |
| `gemm_w2` | `axg`/`asa`/`xs` | `acts_valid`（w2 不需要拆分） | `dnnl_gemm.cpp:1116-1131` |

**为什么必须有守卫**：模式 1/2 的 prefill 用 `do_split=false` 量化，它只写 `axg`/`asa`/`xs`，
`axe`/`axo` 里留着**上一次 call** 的内容。缺守卫时 0.8B 的多设备 decode-vs-prefill 测试会失败
（`gen0/gen1/ref` 220/220/567 对 248068/271/271），而所有单设备测试都通过；`PF_NAT=0` 是对应的
二分开关（`dnnl_gemm.cpp:1370-1374`）。格式与守卫条件都在这里，kernel 侧的展开与发射几何见
[03-kernels.md §5.8](03-kernels.md)。

---

## 7. 相关环境变量

总表以 [`AGENTS.md#environment-variables`](../../AGENTS.md#environment-variables) 为准，这里只列与权重/激活量化直接相关的子集。
**默认值有两层**：设备 profile 的实测默认（`src/device/profiles/<card>.cpp` 的 `wt.*`）与环境变量，
环境变量是 A/B 开关、profile 是默认（`e ? atoi(e) : si::dev::active().wt.X`）。

| 变量 | 默认 | 作用 |
|---|---|---|
| `PF_DP4A` | on | `0` 强制 fp32；否则单设备构建 `w8t` 副本并录制 SI8 prefill/decode 图。**多设备（`--layer-map`）下 `pf8` 恒为 false**（`engine.cpp:190`），此时 0 只是把整条 int8 路径关掉 |
| `PF_DP4A_DEC` | on | `0` 只把 decode 退回 fp32（prefill 仍 int8，`engine.cpp:193`） |
| `PF_META` | off | 构建 Q4_K/Q5_K 的 fp32 `(scale,min)` 旁路数组（约 260 MB）。实测净损失，且只在单设备生效（`engine.cpp:181`） |
| `PF_SI4` | off | 全部重化为 4-bit SIn，副本减半、精度略降 |
| `PF_W4` | on | Q4_K 走原生 4-bit 权重路径（§9）；`0` 恢复纯 int8（每 32 组权重 scale 仍生效） |
| `PF_K5` | on | Q5_K 走**原生 5-bit** 存储（§11，0.75 B/w，无损）；`0` 回到 int8 转换 |
| `PF_CB4` | on | IQ4_XS/IQ4_NL 走 codebook 4-bit（§10，0.5625 B/w，无损）；`0` 回到 int8 转换 |
| `PF_W4_K5` | off | **只**把 Q5_K 重化到 u4 栅格（§9.6，有损）；与 `PF_K5` 是两回事 |
| `PF_W4_ALL` | off | 把其它所有量化类型也重化到 u4 栅格（有损，§9.6） |
| `PF_K5_NOCORR` | off | 诊断：k5 prefill 丢掉 offset 修正项，用于二分 prefill 的代价来源 |
| `PF_NAT` | on | `0` 关掉 `nat_gemm_launch`（批量原生 GEMM），退回 oneDNN matmul + epilogue |
| `PF_ROWACT` | off | `1` 恢复已停用的每行激活量化 `act_quant_launch`（A/B，它实测只增加 per-call 延迟） |
| `PF_DNNL_BLOCKED` | 0 | 诊断：强制 oneDNN 用阻塞（XMX 友好）权重布局；缓冲不重排，结果错 |
| `PF_DNNL_NOWARM` | off | `1` 跳过 `warmup()`，让第一个请求付一次性 kernel 加载 |
| `PF_GEMM_DNNL` | on | `0` 关闭 oneDNN，改用 DP4A chunk-batched |
| `PF_CPU_ISA` | 自动 | `scalar\|avx2\|avxvnni\|avx512` 强制 CPU 整数内核的 ISA 变体；`scalar` 会整体关掉 CPU 的整数 int8 路径（§5.2） |
| `PF_W2_PAIR` | off | `1` 让 w2 GEMV 一次载入两个 32 值组（实测更差，见 §12） |
| `PF_W2_INFO` / `PF_W2_DEBUG` | off | 打印 w2 张量的 B/weight 与 rel L2 / 追踪 GEMV 命中 |
| `PF_W4_INFO` / `PF_W4_DEBUG` | off | 打印每个 u4 权重被拒/被登记的原因与 B/weight / 追踪 GEMV 分支 |

下面这组是纯诊断开关，一行一条（语义见 [`AGENTS.md`](../../AGENTS.md#environment-variables)）：
`PF_W4_RB`（decode GEMV 每 work-group 的行数，默认 16，`w4_gemv.cpp:57-70`）、`PF_W4_GEMM_MAXM`
（`w4_gemm_launch` 的 M 上限，默认 1 = oneDNN matmul，`dnnl_gemm.cpp:1502-1512`）、
`PF_W4_NOCORR`、`PF_I8_NOGEMV`、`PF_DNNL_TIME`、`PF_MTP_HEAD_W4`、`PF_MTP_HEAD_W2`、
`PF_MTP_LAYER_W4`、`PF_MTP_LAYER_W2`、`PF_MTP_LAYER_W4_CALL`、`PF_MTP_LAYER_W2_CALL`、
`PF_MTP_LAYER_EXACT`。

### `PF_META` fp32 side array

`build_meta32`（`engine.cpp:666-710`）只为 Q4_K(12)/Q5_K(13) 构建 `sycl::float2` 设备数组
（`N*(K/32)` 项，144/176 字节 super-block 解码），按主机张量指针登记在 `meta32_`。它被挂到
`gemv_seg::meta32`，供 fp32 decode GEMV 跳过 6-bit packed scale 解码。实测为净损失：packed scale 与
权重共享 cache line，而独立数组增加一个内存流；打开还要求 `!cpu_mode && !multi_dev`
（`engine.cpp:181`）。

### KV 缓存的量化（另一套格式）

KV 的量化与权重/激活是**独立**的一套：`kv_dtype_t`（`kv_type.h:30`）的 `f32|bf16|f16|i8|i4`，
默认 `i8`。`i8` 是对称 int8 + **每 32 个 head dim 一个 f16 scale**（i8 与 i4 共用这个 scale 平面，
`kv_type.h:13-20`），每 token 每层 K+V 共 3 KB（bf16 6、f32 12），`i4` 每字节两个有符号 4-bit
（值域 `[-7,7]`、低 nibble 是偶数 head dim）再减半到 1.5 KB。`PF_KV_TYPE`（可写 `K:V` 分别选两侧）
与别名 `PF_KV_F32` / `PF_KV_BF16` 选择，`--kv-type` 覆盖环境
（`kv_type.cpp:86-102`；混用只允许两个带 scale 的类型互配或完全相同，`kv_type.cpp:43-52`）。
算术全程 fp32，只有存储变窄；布局与 attention kernel 侧见
[05-kv-cache.md](05-kv-cache.md) 与 [03-kernels.md §11](03-kernels.md)。

---

## 8. 精度与正确性说明

* **权重值**默认与 GGUF 整数逐位一致，只有 scale/min 的表示被重新量化到 fp16（u4 路径同样如此，
  见 §9：唯一的误差来源是 f16 元数据，实测每行相对 L2 **0.077%**）。
* **和 int8 比精度时要挑对基准**：现在 `add_weight` 的 int8 转换是**每 32 组**一个 scale，实测 ~0.1 %
  （`dnnl_gemm.cpp:336-340`）；**每行一个** scale 的旧转换才是 ~1 %（对每个类型都一样，Q8_0 也不例外，
  `test_quant_audit`）。所以 u4 的 0.077 % 只比**旧的** int8 转换好一个数量级，比现在的 ~0.1 % 略好；
  §9.1 的表里 0.98 % 那一行指的是旧转换（`w4.h:11` 原文就写着 "the previous conversion"）。
* **激活**是对称 int8，每 32 值两个额外量：fp32 scale 与每 16 值整数和；round 为 half-to-away
  （GPU `sycl::round`、CPU `std::round`），clamp 到 `[-127,127]`。激活误差 ~0.9 %（每 32 组一个
  scale，`dnnl_gemm.cpp:94-96`），这是 u4 路径必须用分组激活而不能只留每行 scale 的原因。
* **DP4A vs oneDNN**：数值上不同。DP4A 走 SIn 的原生整数 + 每 32 值激活 scale，整数域精确；
  oneDNN 走 int8 行主序 + grouped f16 scale，matmul 内就乘上两个 scale（f32 累加），epilogue 只剩
  `alpha` 与 residual。引擎把 `PF_GEMM_DNNL=0` 当作 DP4A 验证配置（注释里的 "bit-identical
  dp4a validation"）。
* 严格 kernel/端到端测试通过 `setenv("PF_DP4A","0",1)` 验证 fp32 路径。

---

## 9. 原生 4-bit（u4）权重路径（`w4.{h,cpp}`、`w4_gemv.cpp`）

### 9.1 动机

GGUF 的 K-quant 已经把每个权重放在**每 32 值一组**的均匀网格上：`w = offset_g + step_g*q`，
`q` 是该类型的原生位宽，`step_g = d*sc_j`、`offset_g = -dmin*m_j` 直接来自 super-block。因此
**保留 `(q, step_g, offset_g)` 是无损的**，而且比 int8 小 1.6 倍（`K*N/2 + (K/32)*N*4` 字节 =
0.625 B/w vs int8 的 `K*N + (K/32)*N*2` = 1.0625 B/w，两式与 `dnnl_gemm::weight_bytes`
（`dnnl_gemm.cpp:770-790`）一致）。实测每行相对 L2（模型自身反量化值为基准，`w4.h:8-13`）：

| 表示 | rel L2 |
|---|---|
| int8 per-row（**旧**转换） | 0.98 % |
| u4 + 重算每组 min/max | 4.39 % |
| **u4 + 原生 `(step, offset)`，f16 元数据** | **0.077 %** |
| u4 + 原生 `(step, offset)`，f32 元数据 | 0 |

也就是说 4-bit 不只是省内存，**精度还优于 int8 转换**：它不引入重化误差，只是把元数据舍入到 f16。
27B 的 Q4_K 占多数，这是它能塞进两张 A770 的关键。

注意 `off` 平面存的是**加性常数** `offset`（= `-dmin*m`），所以 §9.3 的修正是 `+offset`，不是
`-offset`（`w4.cpp:76-79`）。

### 9.2 布局（`w4t`，`w4.h:45-55`）

| 平面 | 布局 |
|---|---|
| `vals` | `vals[(n*K + k) >> 1]` 的 nibble（`k & 1`），u4、K 为内层、低 nibble 在前 |
| `scale` | `scale[g*N + n]` f16 = `step`，`g = k/32` |
| `off` | `off[g*N + n]` f16 = `offset`（加性常数/修正系数） |

两个独立的 f16 平面：oneDNN 的 grouped-scale memory 必须是恰好这个顺序的连续 f16 张量（已实测验证），
而修正项把 `off` 当作 `{NG,N}` 的 f16 矩阵读取。

**注册范围**：只有 `--layer-map` 的多设备路径会调用 `add_weight_w4`（`engine.cpp:1150`），且
`w4_supported`（`w4.cpp:32-49`）只放行 Q4_K(12)——除非开了 `PF_W4_K5`（再放行 13）或
`PF_W4_ALL`（§9.6）。单设备运行不注册任何 u4 权重（§1）。

**`w4_pack_any`**（`w4.cpp:327-332`）是绕过 `w4_supported` 的同一个有损打包，供单个显式选定的
张量用——目前只有 MTP draft 的 LM head（`PF_MTP_HEAD_W4`，`engine.cpp:1289`）。它注册在
**私有 key** 下：同一个 key 再存一份就会把 target 的 decode head 也切到 u4。

### 9.3 prefill：oneDNN u4 GEMM + 修正项

权重作为 `{K,N}` `format_tag::ba` 的 `u4`，配 grouped f16 WEIGHTS scale（= `step`，mask 在 K、
`groups={32,1}`）与 grouped f16 SRC scale（mask 在 K、`groups={1,32}`），**dst 为 f32**
（`make_prim4`，`dnnl_gemm.cpp:466-494`）：

```
out = oneDNN(u4 weights, grouped f16 scales = step) + correction
correction[m][n] = Σ_g off[n][g] * XS[m][g]        # XS[m][g] = 该组 32 个激活之和
```

因为 `w = offset + step*q`，零点被移进一个小的 `M × (K/32) × N` 项，而不必使用 oneDNN 的 grouped
zero-points（其布局未通过验证）。`act_quant_grp_launch` 顺带算出 `XS`，修正由 `w4_epilogue_launch`
完成（`dnnl_gemm.cpp:161-224`，寄存器分块 `TM=TN=4`、work-group 64×64，否则朴素的 per-(m,n) 循环
会把 offset 平面重读 M 次，实测慢 ~45 倍）。

### 9.4 decode：u4 GEMV（`w4_gemv.cpp`）

`w4_gemv_launch`（`w4_gemv.cpp:148-165`）把 `(g,n)` 的 `step`/`off` 分阶段进 SLM、16 字节 nibble
载入、`dp4a` 累加，行块 `RB` 默认 16（`PF_W4_RB` 覆盖，`w4_gemv.cpp:57-70`；SLM 装不下就退到 8/4）。
它读的激活是**偶/奇 k 平面**，所以它属于必须 `do_split=true` 的那批（§4.4）。

拆分与 `XS` 都**不再由独立 kernel 产生**：`act_quant_grp_launch` 在同一个 kernel 里一并写出
`axg`/`asa`/`xs`/`axe`/`axo`（每 32 lane 一个 (行,组)，max 用子组归约），所以一次 call 只发一个
激活 kernel 而不是三个（`dnnl_gemm.cpp:107-111`）。文件里那个单独的
`w4_split_act_launch`（`w4_gemv.cpp:46-55`）已经没有调用者。SLM 里的 scale 分块按 **g 外/行内**
索引（相邻 lane 读相邻 `n`），代码注释记录这个顺序值内核带宽的 10-25 %（A770 实测，
`w4_gemv.cpp:18-25`）；tile 按 f16 分块，所以 `RB=16` 就已用尽两个平面的 SLM 预算。

### 9.5 与 int8 / SIn 的关系

* **Q4_K 走 u4，IQ4_XS/IQ4_NL 走 codebook u4（见 §10），Q5_K 走原生 5-bit（见 §11）**。
  Q6_K/IQ3_S/Q3_K 没有对应的原生存储，继续走 int8（`add_weight`）——27B 是混合量化，所以每个张量
  各自选路径，注册顺序 u4 → k5 → cb4 → int8（`engine.cpp:1150-1168`），启动日志
  `[dev] device N weights: X u4, Y k5, Z codebook, W int8` 就是这条链的计数（`engine.cpp:1364`）。
* **激活量化也按 32 值一组**（`act_quant_grp_launch`）：只有整行一个 scale 时，激活的 ~0.9% 误差会
  盖过 4-bit 权重的精度优势（`dnnl_gemm.cpp:91-96`）。每行的 int8 形式仍然被产出但已无消费者
  （`PF_ROWACT`，§6.3）。
* `PF_W4=0` 恢复纯 int8（每 32 组的权重 scale 仍然生效，那不是 u4 专有的）。
* **喂给原生 store 的 GEMV，激活必须用 `do_split=true` 量化**——完整规则见 §4.4，守卫条件见 §6.5。
  少这一次拆分不会报错，只会读到上一次 call 的平面：MTP 层的 `eh_proj` 就这样给出 acc 0.08
  （draft 全是垃圾），修成 `do_split = mtp_layer_w4_ || mtp_head_w4_` 后恢复到 2.14
  （`engine_mtp.cpp:377-382`）。
* 测试分工：各类型（相对 fp32）的损失用 `test_quant_audit` 逐类型度量；u4 打包往返用
  `tests/model/test_w4`；u4 vs int8 的逐张量差异用 `test_w4_vs_i8`；u4 GEMM vs 真实反量化权重用
  `test_w4_gemm`；u4 logits vs fp32 CPU 参考用 `test_w4_vs_cpuref`。

### 9.6 `PF_W4_K5` / `PF_W4_ALL`：把其它类型重化到 u4 栅格（默认关）

`pack_generic`（`w4.cpp:108-151`）是 `PF_W4_ALL` 的通用重化：按该类型自己的参考反量化器解出一行，
再对每个 32 组按 min/max 拟合一个仿射 `(step, off)` 并 `lround` 到 nibble。`off` 存的是**加性**
常数（= `mn`），正好落进 u4 GEMV 的 `step*qdot + off*xs`（`w4.cpp:105-107`）；它与 `PF_SI4`
（§3.5）的通用重化是同一套数学。

| 开关 | 作用范围 | B/w | 有损？ |
|---|---|---|---|
| `PF_W4_K5=1` | **只有 Q5_K(13)**（`w4.cpp:43-45`） | 0.625 | 是：丢掉第 5 位 |
| `PF_W4_ALL=1` | 其余所有量化类型（排除 type 0/1，`w4.cpp:48`） | 0.625 | 是 |

`PF_W4_K5` 与**原生 5-bit 存储 `PF_K5`** 是两回事，别混：

| | `PF_K5`（§11，默认开） | `PF_W4_K5`（默认关） |
|---|---|---|
| 布局 | u4 nibble + **1-bit 第 5 位平面** | 只有 u4 nibble（第 5 位被丢掉） |
| B/w | 0.75 | 0.625 |
| 精度 | **无损**（native 值精确，只有 step/off 的 f16 舍入） | **有损**（重化） |
| decode GEMV | 一次 OR 合并两个平面 | 与 u4 完全相同 |

`PF_W4_K5` 的动机是 Q5_K 是 27B 权重流里最大的单一字节消费者（35 %，`w4.cpp:37-42`、`w4.h:81-83`），
0.75 → 0.625 的省下是值得的**字节**交易；**为什么默认关**是精度：
`test_w4` 的往返 rel L2 从 0.077 % 变成 **4.60 % 均值 / 10.83 % 最差**——与 `PF_W4_ALL` 同一
误差等级，160 token 的贪心生成虽然仍然连贯但在第 37 个词发散。这是一个精度决策，不是引擎缺陷，
所以它与 `PF_W4_ALL` 一样默认关闭。

## 10. codebook 4-bit（IQ4_XS / IQ4_NL，`PF_CB4`，默认开）

### 10.1 动机

IQ4_XS / IQ4_NL **不是网格**：它们的值是

```
w = d * (ls - 32) * kvalues_iq4nl[q]        # q 是 4-bit 索引
```

其中 `kvalues_iq4nl[16]` 是固定的 int8 码本（`quant.h:218`），`d` 是每 256 值的 f16，
`ls` 是每 32 值的 6-bit scale。所以**只要存 4-bit 索引 + 每 (行,组) 的 f16 `d*(ls-32)`，
native 值就是精确的**——不是重化（requant），只是换存储。

| 表示 | 字节/权重 |
|---|---|
| int8（`add_weight` 的转换） | 1.0625 |
| **codebook u4** | **0.5625** |

省下的就是 27B 每 token 约 2.5 GB 的设备权重（[`README.md`](../../README.md) 的
"Codebook 4-bit" 一节）。注：IQ4_XS 的 native 值本身
就是 int8 网格上的精确点（`dl*lut` 是 `dl` 的整数倍），所以原来的 int8 转换对它几乎无损
（~0.05% f16 舍入）——**这条路的收益是字节数，不是精度**（`test_w4_vs_cpuref` 的 `mean|diff|`
0.0375 → 0.0479，两者 argmax 都 SAME）。

### 10.2 布局（`cb4t`，`w4.h:125-131`）

| 平面 | 布局 |
|---|---|
| `idx` | `idx[(n*(K/2)) + g*16 + b]`，每 32 值一组 16 字节；**交错序**（与 u4/k5 的 nibble 平面同一约定）：byte `k` = 元素 `2k` 的低 nibble / `2k+1` 的高 nibble |
| `scale` | `scale[g*N + n]` f16 = `d*(ls-32)`，与 u4 的 step 平面同构（oneDNN grouped scale 直接用） |

`cb4_pack` 只做重排（`interleave_group`，`w4.cpp:161-173`：把 native 的"元素 `e<16` = byte `e`
低 nibble / `e>=16` = byte `e-16` 高 nibble"转成交错序）+ 抽取 scale，不碰任何数值。**为什么用
交错序**：一个索引字节的两个 nibble 正好是同一 dp4a 操作数里的两个相邻元素，于是 256 项 uint16 表
`lut16[b] = value(b&0xF) | value(b>>4)<<8` 一次查表就给出一半的操作数（2 个值），一个权重字（4 值）
只要 2 次查表 + 1 次 shift + 1 次 or；native 序则需要 4 次字节查表 + 3 shift + 3 or（每 32 值
32 次查表 + 24 次 ALU）。代码注释把改之前的形态记为**读带宽上限的 49-67 %，即指令受限**
（`w4_gemv.cpp:676-684`、`w4.h:119-124`）。

### 10.3 decode：LUT 展开的 dp4a GEMV（`cb4_gemv_launch`）

结构同 `i8_grp_gemv`（g 外/行内 SLM staging、`asa`/`XS`、0x80 偏置修正），只是权重字由索引经
**256 项 `lut16`（SLM）** 展开：`w[2j] = lut16[b_2j] | (lut16[b_2j+1] << 16)`，即一个权重字
（4 值）2 次查表 + 1 shift + 1 or（§10.2 的交错序是前提）。**没有偶/奇拆分**：cb4 的 GEMV 直接读
连续的 `axg`（`dnnl_gemm.cpp:1316-1319`），所以它不需要 `axe`/`axo`；但它的 `nat_gemm` 分支同样由
`split_valid` 守卫（§6.5），即 decode/verify 时这次 call 的 `do_split` 仍要为真。
（真正完全不受 `do_split` 约束的是 §12 的 w2，它连守卫都不需要。）

有效速率仍低于 int8 GEMV，[`AGENTS.md`](../../AGENTS.md) 的结论是它**受解码发射限制**
（~195 GB/s）：码本 LUT 解码（每个操作数集 16 次 `lt16` 查表 + shift/OR）就是全部的超出量，
而 256 项 uint32 表、预偏置码本、预移位 LUT 三种改法实测都更差。整步 tg 的收益因此只有 ~+5 %
（[`README.md`](../../README.md)）——两张卡合计的聚合带宽已经接近可达 DRAM 速率，单核再快也不在
关键路径上。

### 10.4 prefill：展开到复用 scratch + 现有 int8 primitive

oneDNN 只认线性 `u4` 或 `s8`：它的数据类型枚举里是 `s4`/`u4`/`f4_e2m1`/`f4_e3m0`（以及 s8/u8/f16…），
**没有 u5/u6**（`oneapi/dnnl/dnnl_common_types.h:88-112`，oneDNN 头文件树里的文件，不在本仓库），
所以 codebook 的 prefill 只能喂 int8。做法是**把该张量的索引经同一 LUT 展开成 int8，写进一块按
最大单张量分配的共享 scratch**（`cb4_scratch`，最大是 `ffn_gate/up`，~95 MB，
`dnnl_gemm.cpp:916-918`；k5 的 prefill 展开共用同一块，`dnnl_gemm.cpp:842-855`），再用**已有的 int8
primitive**：

* 前提已验证：覆盖权重缓冲后重新 `execute`，结果跟着变（2→64、3→96、4→128），即 **oneDNN 每次
  execute 都重读用户权重 memory**，不做内部 reorder 缓存（`dnnl_gemm.cpp:1328-1331`）。
* 代价是每 pass 多一遍**串行**展开流量（读 0.5625 + 写 1.0 B/w；M=512 时 GEMM 自身的权重读被算力
  掩盖，这段掩盖不住）。
* `cb4_expand_launch` 的元素序必须与 §10.2 一致：一个 `uint4` 索引（16 字节）→ 8 个权重字
  （每个 = 2 次 `lut16` 查表 + shift/or），两个 `uint4` 写出。**必须向量写**：逐字节标量写的形式让
  prefill 慢了 ~1.5 倍（`w4_gemv.cpp:691-693`），与 k5 展开同样的教训——分散的 32-bit 写会把内核
  压到带宽上限的一半。

### 10.5 净效果

`PF_CB4=0` 退回 int8 转换（A/B 用）。在 27B / 2×A770 上开着的净效果是 **tg128 ~+5 %**、
**每卡设备权重 −2.5 GB**，prefill 因为展开流量而变慢（[`README.md`](../../README.md)）。
两个 A/B 的 argmax 都与 fp32 参考一致（`test_w4_vs_cpuref`）。

## 11. 原生 5-bit（Q5_K，`PF_K5`，默认开）

Q5_K 的原生栅格是

```
q5 = (qs nibble) | (qh bit << 4)   ∈ [0,31]
w  = d·sc_j · q5 − dmin·m_j          （每 32 个一组，6-bit (sc,m) 由 get_scale_min_k4 取出）
```

它**不是** 4-bit 栅格：UD-Q4_K_M 里 Q5_K 占本模型权重的 35 %，曾经只能 int8 转换（1.0625 B/w）。
丢掉第 5 位（`PF_W4_K5`/`PF_W4_ALL` 的 u4 重化，见 §9.6）会损失该类型的原生分辨率，所以这里
**保留 5 位**，只换布局：

| 平面 | 布局 | 大小 |
|---|---|---|
| `vals` | 4-bit nibble，**交错**（byte k = 元素 2k 低 nibble / 2k+1 高 nibble） | K/2 |
| `hi` | 第 5 位，**按 split 元素序**：每 32 组 4 字节，bit 0-15 = 偶元素 2i→bit i，bit 16-31 = 奇元素 2i+1→bit i | K/8 |
| `scale` / `off` | `[g][n]` f16 step = d·sc、off = −dmin·m（与 u4 同一约定） | 各 K/32·N·2 |

即 0.5 + 0.125 + 0.125 = **0.75 B/w，native 值精确**（只有 step/off 的 f16 舍入，
与 u4 路径同量级；`w4_gemv.cpp:784-787` 的 `weight_bytes()` 与此一致）。

为什么 `hi` 用 split 序而不是元素序：解码 GEMV 复用 u4 的 **偶/奇激活平面**
（`axe`/`axo`），一个 dp4a 操作数的 4 个元素正好是同一 split 序的相邻 4 位——
于是 4 bit 经 16 项 SLM LUT 展开成 4 字节掩码后，`q5 = lo4 | (mask << 4)`
（OR 即加，lo4 < 16），代价只有每 32 值 8 次 SLM LUT 读 + 16 条 ALU（`w4_gemv.cpp:524-537`），
**内核仍纯带宽受限**：注释记录它在卡的 ~405 GB/s 读上限处，而 int8 GEMV 只到该上限的 92 %，
所以这次解包完全被隐藏（`w4_gemv.cpp:530-531`）。

### 11.1 decode / prefill 两条路

* **decode**：`k5_gemv_launch`（`w4_gemv.cpp:538-631`），0.75 B/w 直接读，无展开；读 `axe`/`axo`
  两个偶/奇平面，所以它属于必须 `do_split=true` 的那批（§4.4、§6.5）。
* **prefill**：oneDNN 只认 `u4`/`s8`（§10.4 的数据类型枚举），所以 `k5_expand_launch`
  （`w4_gemv.cpp:639-669`）把两个平面重排成元素序 int8 写进与 cb4 共享的 `cb4_scratch`
  （~95 MB，§10.4），再跑**已有的分组 scale int8 primitive**，最后用 u4 的修正 epilogue
  （`w4_epilogue_launch`）把 `off` 项加回（`dnnl_gemm.cpp:1271-1309`）。代价是每 pass 多
  0.75（读）+ 1.0（写）B/w 的**串行**流量（M=512 时 GEMM 自身的权重读被算力掩盖，展开流量
  则不能）；`PF_K5_NOCORR=1` 丢掉修正项做二分定位（`dnnl_gemm.cpp:1300-1308`）。
* 展开内核本身必须是**向量写**：每 work-item 一次 32-bit 载入 + 两个 `uint4` 存储。
  注释记录这里踩过的两个坑（`w4_gemv.cpp:650-667`、`:691-693`）：**64-bit 的 byte-spread**
  版本每 8 个值约 40 条指令，把内核压到 200 GB/s（上限的 49 %，纯指令受限），32-bit 形式只要
  ~12 条；逐 work-item 的 32-bit **分散存储**同样只到带宽上限的一半。
* **MTP verify 不走这条路**：k5 也被 `nat_gemm_launch(FMT 1)` 直接读原生两个平面
  （`dnnl_gemm.cpp:1265-1270`），所以多 token 的 verify 不付展开流量；prefill（M 可达 `kMaxT`）
  仍然要展开，这也是 §12.3 里 `PF_MTP_LAYER_W4` "与 int8 副本并存而不是取代"的原因。

### 11.2 净效果（27B / 2×A770，`--layer-map 0-31:gpu.0,32-63:gpu.1`）

`PF_K5=1` 相对 `PF_K5=0`：**tg128 +6 %、pp512 −15..−20 %、每卡设备权重 −2.1 GB**
（[`README.md`](../../README.md)），而精度**反而更好**——保留 native 值，没有任何张量被重化，
`test_w4_vs_cpuref` 的 `mean|diff|` 从 0.0375 降到 0.0354。
pp 的代价是展开流量，只能靠把展开分块进 L2 收回，而 **L2 分块已实测为负收益**：把 GEMM 按 N
切块让展开 tile 留在 L2，在 M=512 与 M=41 两种区间都比整张慢 22-326 %（M=512 的 GEMM 是
算力受限、tile 在不在 L2 无关，而每次 `execute` 有固定的 GPU 开销、展开本身也更慢）。

k5 的 GEMV 有专门的测试：`tests/backend/gpu/kernels/test_k5_gemv.cpp` 拿**原生** Q5_K 权重
（反量化器自己的 q5 抽取）构造精确的 host 参考，所以唯一的残差就是那两个 f16 元数据平面的舍入。

## 12. 2-bit 权重（`w2t`，MTP draft 专用，默认关）

### 12.1 布局与解码

`w2_pack_any`（`w4.cpp:337-383`）把任意量化张量重化到**每 32 值 2-bit** 的均匀网格：
`step = (hi-lo)/3`（4 个电平 0..3）、`off = lo`（加性常数），`q = clamp(lround((w-lo)/step), 0, 3)`。

| 平面 | 布局 |
|---|---|
| `vals` | `[N][K/4]`，**K 为内层的元素序**：元素 `k` 在字节 `k>>2` 的 bit `(k&3)*2`（低字段在前） |
| `scale` / `off` | `[K/32][N]` f16 step / offset（与 u4 同一约定） |

即 0.25 + 0.125 = **0.375 B/w**（u4 的 0.625 的 60 %，比 int8 转换的 1.0625 少 65 %），
与 `weight_bytes()`（`dnnl_gemm.cpp:778-780`）和 `w4.h:133-148` 一致。**与 u4/k5 的三个关键
差别**：

1. **没有偶/奇拆分**。2-bit 平面是元素序而非交错序，一个字节正好是 4 个**相邻**元素 = 一个 dp4a
   操作数，所以 GEMV 直接读连续的每 32 组 int8 激活 `axg`（`dnnl_gemm.cpp:1116-1131`）。
   于是 w2 不需要 `do_split`，也就躲开了 §4.4 那类静默失效——这是它比 u4 好用的地方之一。
2. **没有 0x80 偏置修正**。电平 0..3 本身就是合法的无符号 dp4a 操作数，不像 int8 路径那样要
   XOR `0x80` 再减 `128*sum(a)`（对比 `w4_gemv.cpp:174-176` 与 `:439-445`）。
3. **每组只有 8 字节**，所以一次 8 字节载入给一个 lane 16 个权重（u4 是 16 字节载入给 8 个）。
   展开本身每 4 个权重大约 9 条 ALU，代码注释的结论是这远低于发射预算，内核仍然带宽受限
   （`w4_gemv.cpp:167-182`）。`PF_W2_PAIR=1` 试过用一次 `uint4` 载入两个组，
   **实测更差**（head 形状 K=5120 N=65536：0.517 ms / 243 GB/s 对 0.467 ms / 270 GB/s），
   原因是活跃字更少胜过了更宽的载入，所以默认是窄载入（`w4_gemv.cpp:220-226`、`:279-288`）。

`w2t::rel_l2` 字段（`w4.h:155-159`）在打包时就顺带算好这个拟合相对源张量的相对 L2，
`PF_W2_INFO=1` 打印它——它是判断"这个存储能不能用"的唯一数字。

### 12.2 它服务谁、为什么默认关

它只为 MTP draft 的 head readout 存在（`PF_MTP_HEAD_W2=1`，需要 `--mtp` + 27B 的 NextN 头），
以及 `PF_MTP_LAYER_W2=1` 的 MTP 层。**两处默认都关**：

* head：27B 的 Q6_K head 上 rel L2 是 **39.8 %**（同一个 head 的 u4 副本是 0.077 %）。
  它是精确算术且少 40 % 的 head 字节，draft 18.9 → 17.4 ms/cycle（k=4），但 ~2 % 的 acceptance
  跟着一起走，端到端 32.4 vs 32.0 ms/token —— 打平（`engine.cpp:1291-1296`）。
  [`AGENTS.md`](../../AGENTS.md) 补了为什么字节只兑现一半：这个 GEMV **不是带宽受限**
  （270 对 u4 路径的 314 GB/s），所以 1.6 倍的字节削减只实现了 1.4 倍。
* 层：见 §12.3。
* 往返误差用 `dev/test_w2.cpp` 对拍（1.9e-07，精确到 fp32 参考）。**注意 `dev/` 在
  `.gitignore` 里、未被 git 跟踪**，那是一个本地压测工具，不是仓库的一部分。

### 12.3 关键的不对称：读出端有损无所谓，递推里有损会累积

head readout 只取 argmax，偶尔翻一个近似打平不致命（verify 会兜住）；但 MTP **层**的 2-bit
store（`PF_MTP_LAYER_W2=1`）误差会喂回下一个 draft step，**acc 2.14 → 1.75**，换来 0.5 ms/cycle，
被否决（`engine.cpp:1196-1198`）。反过来 `PF_MTP_LAYER_EXACT=1`（层的 GEMV 走精确 fp32 反量化）
与 int8 store 的 acceptance **完全相同（acc 2.14 两者一致）**，说明 draft 的精度余量很大，
位宽根本不是 acceptance 的限制项。

同一逻辑下 u4 是安全的、且**默认开**：MTP 层的 linears 是 Q6_K/Q8_0（338 MB），通用 per-32 u4
打包在 Q6_K 上要 2.6 % 相对 L2（`blk.64.nextn.eh_proj` 上 cos 0.9949、max|diff| 0.031），是
`PF_W4_ALL` 记录值的 8 倍——**但它对 acceptance 毫无影响**：int8 副本、u4 副本、精确 fp32 反量化
三条路的 acceptance 都是 2.14，而 draft 从 14.2 → 12.8 ms/cycle（`engine.cpp:1183-1198`）。
这里曾经出现过 acc **0.11** 的崩塌，但那是 §4.4 那个 `do_split` bug（读到了上一次的偶/奇平面），
不是精度——所以最终结论是：**MTP 层可以用 4-bit（无损于 acceptance），2-bit 不行；
head 的读出端两者都可以，但 2-bit 省下的字节不足以补偿它丢掉的 acceptance。**
`PF_MTP_HEAD_W4`（默认开）就是 head 的 u4 副本，注册在私有 key 下，target 的 decode head 仍是
精确 int8（`engine.cpp:1273-1290`）。

最后一层保险：**原生副本是"加上"而不是"取代"int8 副本**（`engine.cpp:1229-1247`）——MTP 层自己的
prefill（M 可达 `kMaxT`）与任何 M=1 GEMV 服务不了的形状仍读 int8；早期版本跳过 `add(t)` 导致
prefill 抛 "oneDNN GEMM failed for a layer tensor"。`PF_MTP_LAYER_W4_CALL=<ci>` /
`PF_MTP_LAYER_W2_CALL=<ci>`（`0` qkv、`1` wo、`2` ffn gate/up、`3` ffn down、`-1` eh_proj）是
把单个 call 放上原生栅格的二分开关。
