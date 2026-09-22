# 设计 03：SYCL Kernel 库与 CPU kernel 库

覆盖 `src/backend/gpu/kernels/` 下所有 SYCL kernel 与 `kernel_utils.h`；CPU 后端的同构主机实现
在 `src/backend/cpu/kernels/`（见 §14）。公开启动 API 见 `kernels.h`；量化数学与 SIn/DP4A
见 [02-quantization.md](02-quantization.md)；kernel 如何被编排见 [04-engine.md](04-engine.md)。

---

## 1. 组织约定

* **一个 kernel 一个编译单元**（`.cpp`），kernel 体不放进头文件。
* 启动函数声明在 `kernels.h`，命名 `*_launch`。
* 共享设备辅助在 `kernel_utils.h` 的 `si::kd` 命名空间。
* 典型前缀：

```cpp
#include "kernels.h"
#include "kernel_utils.h"
namespace si {
using namespace sycl;
using namespace si::kd;
...
}
```

* 所有 kernel 从 host-USM 的 `step_info*` 读逐步状态（图重放安全），见
  [04-engine.md](04-engine.md)。

### `step_info` 与 `gemv_seg`

```cpp
struct step_info {                 // kernels.h:57-86
    int32_t n_rows, n_real, tpb;   // 行数、每行 token 数、每行 buffer 的 token 槽数
    int32_t pos[kMaxB], slot[kMaxB], active[kMaxB];
    int32_t tokens[kMaxB * kMaxT];
    int32_t pc_active, pc_stride; float * pc_base;   // 前缀缓存快照
    int32_t pc_row_slot[kPcMapLen];
    int32_t mrope_on, mrope_sections[4];             // 多模态
    int32_t mrope[4*kMaxB*kMaxT];
    const float * img_embd; int32_t img_row[kMaxB*kMaxT];
};
```

`gemv_seg`（`kernels.h:33-54`）描述一个权重矩阵的应用：`w/type/K/n_rows`、输入 `x/x_stride`（可选
`act_up` 做 `silu(x)*up`）、输出 `out/out_stride`（可选 `residual`）、`alpha`、可选 fp32 `meta32`，
以及 DP4A 字段 `w8/x8/xmeta/xsumq`。

### `kernel_utils.h` 共享辅助

| 辅助 | 作用 |
|---|---|
| `silu_f` / `sigmoid_f` | 激活函数 |
| `vload<V>` / `vstore<V>` | `__builtin_assume_aligned` 对齐向量访问 |
| `kv_ld` / `kv_st` / `kv_ld4` | 按 KV 存储类型读/写元素（`kv_ld4` 读 float4） |
| `i8_quant` | round + clamp 到 ±127 |
| `kv_row_data` / `kv_row_scales` | `[block][kv head][token][head_dim]` 数据/scale 行指针 |
| `fast_h2f` | 快速 fp16→fp32（次正规/特殊值回退） |
| `w8_sw_mw` / `w8_group_expand` / `w8_xword` | SIn 权重读取/展开 |
| `dequant_sb_lane_typed` 等 | 256 元素 super-block 的按 lane 反量化 |
| `superblock_bytes` | ggml 类型 → super-block 字节数 |
| `sg_sum` / `sg_sum_i` / `sg_max` | 编译期宽度 32 的子组归约（运行时 `get_local_range()` 曾使 GDN 慢 2×） |
| `dot4` / `mul4` / `fma4` | float4 运算 |
| `gemm_ws(queue&, size_t)` | 进程级 split-K workspace（定义在 `dp4a_common.cpp`） |

---

## 2. `rmsnorm_launch`（`rmsnorm.cpp:12-38`）

```
out_i = x_i * (1/sqrt(mean(x²)+eps)) * w_i
```

`nd_range<1>(n_rows*256, 256)`：每行一个 256 work-item 的组。每线程跨步累加二次和，写入 256 float 的
SLM，再做 256 宽树状归约（`it.barrier()` 分隔各阶段），最后按相同公式写 `out`。归约覆盖全部 256 线程
而非单个子组，因此支持任意 `n`（不要求 `n % 256 == 0`）。

---

## 3. `embed_launch`（`embed.cpp:14-39`）

把每个 active token 的 embedding 行反量化到 fp32 输出，并做多模态替换。

* 网格固定为 `nd_range<1>(kMaxB*kMaxT*n_sb*256, 256)`，`n_sb = n_embd/256`。**故意固定**：`n_rows`/
  `n_real` 是设备侧值，主机计算 extent 会在图捕获时被冻结为 0。
* 组 `g` 分解：`t = g/n_sb`（扁平 token 槽），`sb = g%n_sb`（行内 256 元素 super-block）。
* 多模态：若 `img_embd != nullptr` 且 `img_row[t] >= 0`，直接拷贝
  `img_embd[img_row[t]*n_embd + sb*256 + tid]`，**不查 token 表**。
* 否则 `row = table + tokens[t]*row_bytes + sb*sb_bytes`，值 = `dequant_elem_sb(type, row, tid)`。
  支持 Q4_K/Q5_K/Q6_K/Q8_0/F32；未知类型返回 0。

`row_bytes` 由调用者提供，包含 padding（`sb_bytes*n_sb` 不一定等于 GGUF 行大小）。

---

## 4. `copy_row_launch`（`copy_row.cpp:11-20`）

把一个 activation 行（`n` floats）从 `src` 拷贝到 `dst`。`row < 0` 时取批次最后一个真实 token
（`n_rows*n_real - 1`）。网格是单个 256 线程组 + grid-stride 循环。唯一调用点：prefill 时把最终
post-RMSNorm hidden state 拷进连续的 `d_last_hidden`（模式 != 0）。

---

## 5. GEMV / GEMM

### 5.1 fp32 / 按需反量化 GEMV（`gemv.cpp`）

`gemv_multi_kernel<QT,TB,RPS,NSB,SGW>`（`gemv.cpp:17-147`）：

* `QT` = ggml 类型（12/13/14/8/0，0 = 原始 fp32）；`TB` = 每 segment token 数；`RPS` = 每子组行数；
  `NSB` = super-block 数（0 = `K/256`）；`SGW` = 每 work-group 子组数（默认 8）。
* `rows_per_wg = RPS*SGW`，`n_wg = ceil(total_rows/rows_per_wg)`，线程数 `SGW*32`；另有 `n_tb` 网格维
  覆盖多 token 块（模式 2 一次 dispatch 覆盖全部 chunk 行）。
* 每个 work-group 扫描 segment 列表定位自己的 `row_base`，把描述符缓存进寄存器。
* lane 是 super-block 内 k 位置，`dequant_sb_lane_typed<QT>` 产出 `w[8]`（8 个子块）。
* `TB==1` 直接读 `X`；`TB>1` 先把激活行（应用 `silu(gate)*up`）staging 进 SLM `xs[256*TB]` 并 barrier。
* 最后 `sg_sum` 归约 `(row,t)`，lane 0 写 `alpha*v (+residual)`。

`gemv_dec_vec_kernel<QT>`（`gemv.cpp:199-296`）：仅 Q4_K/Q5_K 的单 token 向量化 decode。`SGW=8`、
256 线程、每子组一行。lane 覆盖**一个子块内 8 个连续值**（`s=lane/4, m=lane%4`），因此每 lane 只需一个
`(scale,min)` 和一次 `uint2` 加载；激活是两次 `float4` 加载。注释记录约减少 1.5× 指令/值。

`gemv_group_launch`（`gemv.cpp:298-410`）：`n_tb = (TB>1 && n_tok_blocks>0) ? n_tok_blocks : 1`。
`GEMV_DEC_VEC`（默认开）、`GEMV_VEC12`、`GEMV_VEC13`（默认开）决定 `TB==1` 是否走向量化 kernel。
`GEMV_CFG1/8/16/32` 是实验性的 `RPS*100+SGW` 配置码。默认由 `GEMV_DISPATCH_TB` 选择
`NSB ∈ {4,8,14,0}` 与 `TB ∈ {1,8,16,32}`。

### 5.2 DP4A GEMV（`dp4a_gemv.cpp`）

`dp4a_gemv_impl<QT,SPLIT>`（`dp4a_gemv.cpp:17-75`）：

* `WG=128` = 4 子组 × 32 行；`n_wg = ceil(N/128)`，grid = `n_wg*n_split`。
* `row = (blk/n_split)*128 + local_id`：**一 lane 一输出行**，一个子组覆盖 32 连续行。因为 32 行同属一个
  128 行块且连续，warp 的权重加载是 `32*GB` 连续字节。
* 读取：`w8_group_expand<QT>` 展开 GBP 字节为 dp4a word；`w8_sw_mw<QT>` 取 `(sw,mw)`；激活 scale
  `sx = xmeta[gx].x()`；`gx = g`（G=32）或 `g>>1`（G=16，两个 Q6_K 组共享一个 32 值激活组）；
  `uint4 xw` 在子组内广播；内层 4 个 dp4a word。
* 累加 `acc += sx*(sw*dot - mw*c)`，`c = sum(qx)`。
* `SPLIT` 时写 partials，否则 epilogue 应用 `alpha*acc (+residual)`。
* `dp4a_gemv_launch`：`PF_GEMV_SPLIT` 默认开；`N <= 4096` 时 `S = ceil(2048/N)` 上限 8（decode 对小
  输出尺寸（4–8 个 work-group）是延迟受限，因此像 prefill 一样切 K）。

### 5.3 DP4A GEMM（`dp4a_gemm.cpp`）

kernel 家族（全部 int8、硬件 dp4a）：

| 家族 | 说明 |
|---|---|
| `dp4a_gemm_impl<QT,TP,RP>` | 寄存器 tiled，`WG=128`，lane 拥有 `TP×RP` 累加 tile；`PF_GEMM_TILE` 选择 (TP,RP) |
| `dp4a_row_gemm_impl<...>` | **一 lane 一输出行**，`TB_T` 个累加器，权重全合并、x/meta 广播；`MT` 时第二网格维切 token；主管道（`PF_GEMM_ROW` 默认开） |
| `dp4a_row2_mt_gemm_impl<...>` | 2 行/lane、`TB_T=16`，32 个累加器，两行共享 x/meta（每 cell 指令少约 25%），可选 SLM staging（`PF_MT_SLM`）与软件预取（`PF_MT_PF`） |
| `dp4a_gs_mt_gemm_impl<...>` | “gemmstone 风格” 4×4 tile（`PF_GEMM_ARCH=2`），约 3.1 MAC/instr |
| `dp4a_row2_gemm_impl<...>` | 子组覆盖 32 行 × 32 token，每 lane 两行相邻行 + 一个 16-token 半块 |
| `dp4a_row_gemm_slm_impl<...>` | SLM staging 版本（`PF_GEMM_XSLM`，默认较慢） |

**split-K workspace**：`gemm_ws(q, need)` 是进程级单例设备缓冲，按需增长。注释解释布局原因：chunk-batched
prefill 中多个行的 GEMM 同时在飞，command graph 不对 USM 访问排序，因此每个 row/split 必须有独立槽位。
partial 布局 `ws[((s*TB_T + t)*N) + row]`，reduce 内核求和后应用 alpha/residual，写
`out[t*out_stride + row]`。`n_rows`（张量行数）与 `out_stride`（可能更大，如 ffn_gate/up 共享 `2*n_ff`）
不同。

**TB 语义**：`TB` 是一次 call 的 token 数（模式 1 = 32，decode = 批大小，模式 2 = 整个扁平 token 数）。
`TB_T` 是编译期 tile，`tstride` 是激活 token 步长；dispatcher 只在 `TB == TB_T`（或 MT 时
`TB == tstride`）时选择对应变体，使 token 循环无 guard。

**调度与调优**（`dp4a_gemm_launch`，`dp4a_gemm.cpp:831-1305`）：M-tiled 路径在
`TB > 32 && TB % 32 == 0 && TB <= kMaxB*kMaxT` 时启用，由 `PF_MT_R`（默认 2）、`PF_MT_TB`、
`PF_MT_WG/WG2`、`PF_MT_PF`、`PF_MT_SLM` 选择；K-split row 路径用于 `TB==32 && N<=8192`，由
`PF_GEMM_SPLIT`（0 → `ceil(16384/N)` 上限 8）、`PF_GEMM_WG`、`PF_GEMM_XSLM`、`PF_GEMM_SG` 选择；
无 split row 路径用于 `N>=2048 && TB∈{32,16,8}`；tiled 回退由 `PF_GEMM_TILE` 选择。

注释中的实测：row 映射在同一数据上约 12 GB/s，tiled 约 4 GB/s；row kernel 需要约 16k 输出行才能填满
机器，因此小张量切 K；更小的 WG 提升常驻 warp 数。

### 5.4 分组 scale 解码 GEMV（`w4_gemv.cpp`）

multi-device 的单 token 解码走这两个 kernel（`dnnl_gemm::gemm_w4` / `gemm` 在 `M==1` 时调用；只
有它们才知道每个权重张量的 oneDNN 分组 scale）：

| kernel | 权重布局 | 公式 |
|---|---|---|
| `w4_gemv_launch` | u4 nibble 平面（K 内层、低 nibble 在前）+ `[g][n]` f16 step/off 平面 | `y[n]=Σ_g asa[g]·(step[g][n]·QDOT_g[n] + off[g][n]·XS[g])` |
| `i8_grp_gemv_launch` | int8 行（K 内层）+ `[g][n]` f16 scale 平面 | `y[n]=Σ_g asa[g]·step[g][n]·QDOT_g[n]` |

- 一个子组（32 lane）= 一个输出行 `n`；lane `l` 处理组 `g=l, l+32, …`。u4 的一个组恰好 16 连续
  字节（一次 `uint4` 载入），int8 是 32 字节（两次 `uint4`）；激活侧 int8 用
  `dp4a_s8u8(x, w^0x80808080, acc)`（XOR 把有符号权重变无符号，再用 `128·Σx` 校正，`Σx` 即
  `XS`，由 `w4_xs_launch` 一次算出、同一 call 的所有张量共享）。
- **scale staging 是这里的关键**：oneDNN 要求 scale 平面按 `[g][n]`（组在外），对逐行 GEMV 就是
  跨 `N` 的步长访问——每个 scale 一条 cache line。因此每个 workgroup 先把 `RB` 行的 scale
  stage 进 SLM，索引取 **g 外 / 行内**（`g=i/RB, r=i%RB`），让相邻 lane 读相邻 `n`（一次
  64 B line 覆盖 32 个 f16），SLM 存 f16（u4 两个平面也能用 `RB=16`）。实测：int8 +10-17 %，
  u4 最高 **-39 %**（`ffn_down` K=17408 N=5120：174 → 288 GB/s）；输出与改前逐位一致
  （`dev/bench_decgemv_eq.cpp`）。`RB` 依 `K` 在 16/8 间回退以守住 48 KB SLM。
- 调优细节与整机 tg128 的结果见 [`reports/tg128_20tps_evaluation.md`](../../reports/tg128_20tps_evaluation.md)。

### 5.5 codebook 4-bit（IQ4_XS / IQ4_NL，`cb4_*`，`w4_gemv.cpp`）

值不是网格而是 `scale[g][n] * kvalues_iq4nl[q]`（16 项 int8 码本 × 每 (行,组) f16 scale），
所以权重存 4-bit 索引 + scale（0.5625 B/w vs int8 1.0625），**native 值精确**。

* `cb4_gemv_launch`（decode，M=1）：结构与 §5.4 的 int8 GEMV 相同（g 外/行内 SLM staging、
  `asa`/`XS`/XOR 偏差校正），差别只在权重字由 16 字节 nibble 经 **16 项 SLM LUT** 展开：
  `w[j]` 取 byte `4j..4j+3` 的低 nibble，`w[4+j]` 取它们的高 nibble（位移 `8b` / `8b+4`）。
  实测每张量比 int8 GEMV 快 11-20 %（head 3.31→2.63 ms），但有效 GB/s 更低（272 vs 408）。
* `cb4_expand_launch`（prefill 用）：把索引展开成 int8 到 `dnnl_gemm` 的复用 scratch，再跑
  已有的 int8 primitive（oneDNN 每次 execute 重读权重 memory，已验证）。向量化：一个
  work-item 处理一组（16 字节载入 + 两个 16 字节写出）；标量版会让 prefill 从 ~1500 掉到 940 t/s。
* 元素序（§10.2 of 02-quantization.md）：每组 16 字节，`e<16` 是 byte `e` 的低 nibble、
  `e>=16` 是 byte `e-16` 的高 nibble——`cb4_pack`、GEMV、展开三处必须一致。
* 逐位对拍：`dev/bench_cb4.cpp` 同时算 host 公式、kernel 逻辑的 CPU 仿真、GPU 三者
  （随机 nibble，max rel 0.000000）。

---

## 6. `qk_norm_rope_launch`（`qk_norm_rope.cpp:14-171`）

对 Q/K 做 per-head RMSNorm + RoPE，并把 K/V 写入 paged KV 池。

* Q buffer 每行 `qstride = n_head*2*head_dim`；每个 head 的第二段 `head_dim` 是 **attention gate**
  （`attn_launch` 从 `qbuf + head_dim` 读它）。
* 子组角色：`n_sg = n_head + 2*n_head_kv`；`sg < n_head` 处理 Q head `sg`；中间处理 K；其余处理 V。
* 网格 `n_rows*n_real*n_sg*32`，每子组 32；`r = gid/n_real`，`t = gid%n_real`；`pos = pos[r]+t`，
  `row = r*tpb + t`，block table 行 = `slot[r]`。

### 6.1 交错 M-RoPE

默认 `rpos = pos`。`mrope_on` 时，lane（= head 内维度对索引）决定 section：

```
sect_dims = s0+s1+s2+s3
sector    = lane % sect_dims
sec = 1 if sector%3==1 && sector<3*s1
      2 if sector%3==2 && sector<3*s2
      0 if sector%3==0 && sector<3*s0
      else 3
rpos = mrope[sec*(kMaxB*kMaxT) + r*kMaxT + t]
```

这是 `0,1,2,0,1,2,…` 的交错模式（section 单位是“对”，对应 `rope.dimension_sections` 如
`[11,11,10,0]`）；频率指数仍用全局对索引 `lane`。

### 6.2 per-head RMSNorm 与 RoPE

* 每 lane 跨步累加 `head_dim/32` 个元素的二次和，`inv = 1/sqrt(sg_sum(ss)/head_dim + eps)`，
  然后 `qh[lane+32*i] *= inv * q_norm[lane+32*i]`（K 用 `k_norm`）。norm 向量长度 `head_dim`，按维度
  索引而非按 head。
* 仅 `lane < n_rot/2` 参与旋转：`ang = rpos * exp2(-2*lane/n_rot * log2(rope_base))`，对
  `(x[lane], x[lane+n_rot/2])` 做半分割旋转（GPT-J 风格）。`≥ n_rot` 的维度不动。

### 6.3 KV 写入

块内 token 偏移 `kb = table[pos/kBlockSize]`、`ko = pos%kBlockSize`。存储类型由 `kv_dtype()` 模板化。

* 非 int8/int4：K 写入 `pool + (((kb*n_head_kv + kh)*kBlockSize + ko)*head_dim) * elem`，布局
  `[block][kv head][token][head_dim]`；V 同理。
* int8：`unit = kb*n_head_kv + kh`；行指针 `krow = base + (unit*kBlockSize + ko)*head_dim`；scale 指针
  `ksc = base + (unit*kBlockSize+ko)*(head_dim/kI8Q)`（fp16）。每个 32 维块内 `m = sg_max(|v|)`、
  `sc = m>0 ? m/127 : 1`、存 `i8_quant(v, sc)`，lane 0 写 `ksc[i] = (half)sc`。
* int4：行指针按字节 `(unit*kBlockSize + ko)*(head_dim/2)`；每个 32 维块 `sc = m/7`、量化到 `[-7,7]`，
  偶数 lane 用 `permute_group_by_xor(sg, nib, 1)` 取相邻 lane 的高 nibble 打包成一个字节；scale 平面同 i8。

---

## 7. `attn_launch` 与 `attn_combine_launch`（`attn.cpp`）

### 7.1 调度

`HD = 256`，`qstride = n_head*2*head_dim`，`pstride = 2 + head_dim`（partial 记录：`m`、`l`、`head_dim`
个累加值）。`fuse = (out != nullptr && n_splits == 1)`。`grp` 来自参数或 `PF_DEC_GROUP`（**默认 off**，
`kernels.h:117` 的注释已过时）。分组 kernel 仅在 `grp && !kv_dtype_has_scales(kv_dtype()) && !fuse && head_dim==256 &&
n_head == 4*n_head_kv` 时选中。否则走经典 kernel，按 KV dtype 特化（i8/i4 无分组路径）。

### 7.2 经典 kernel（`attn.cpp:176-350`）

* grid `nd_range<1>(n_wg*32, 32)`，`n_wg = n_rows*n_real*n_head*n_splits`：一个 warp 对应
  `(row r, token t, query head h, split s)`。
* partial 指针 `partials + ((r*tpb + t)*n_head + h)*n_splits*pstride + s*pstride`；inactive 行写
  `m=-inf, l=0`。
* **因果**：`n_kv = pos+1`；键范围切成 `n_splits` 段，`t0 = s*chunk`，`t1 = min(t0+chunk, n_kv)`。
* **GQA**：`kvh = (h*n_head_kv)/n_head`（**分块**：连续 `n_head/n_head_kv` 个 query head 共享同一个 KV 头，
  对应 HF 的 `repeat_kv`/`repeat_interleave`）。注意 GDN 的 q/k↔value 配对是**取模**（见 §9）：同一个
  模型里两种约定并存，不要想当然地统一，它们只在 head 数相等时才一致。
* **向量化路径 `avec`**（默认，`PF_ATTN_VEC=0` 关闭）：每 lane 两个 4 维块 `d0=lane*4`、
  `d1=d0+HD/2`，Q/K/V 合并访问。非量化 `dot = dot4(qa,ka)+dot4(qb,kb4)`；int8/int4 时两块落在量化 block
  `lane/8` 与 `4+lane/8`，各自乘 scale（i4 用 `i4_ld4` 解包 nibble）。在线 softmax：
  `mnew=max(m,dot); e=exp(dot-mnew); corr=exp(m-mnew); l=l*corr+e`；V 用 `fma4` 累加。
* **融合输出**（`fuse`）：gate 指针是 head 槽的第二半，输出 `(a/l)*sigmoid(gb)`。
* 否则写 partial（`part[0]=m`、`part[1]=l`、后续累加值）。

### 7.3 分组 decode kernel（`attn_group_kernel<HPG>`，`attn.cpp:35-130`）

`PF_DEC_GROUP` 可选（默认 off；Iris Xe 上经典 kernel 实测更快）。一个 warp 对应 `(r,t,kvh,s)`，
`HPG=4` 个 query head 共享每次 K/V float4 加载；每个 head 的运算序列与经典向量化 kernel 完全相同，
因此 partial 逐位一致。无 gate、无 int8/int4。

### 7.4 `attn_combine_launch`（`attn.cpp:364-392`）

按 `(r,t,h)` 归约 `n_splits` 个 partial：`M = max_s m_s`；`sum = Σ l_s*exp(m_s-M)`，
`acc = Σ partial[s*2+d]*exp(m_s-M)`；乘以 `sigmoid(gate)` 后写出。融合/非融合路径都乘 Q-gate 的
sigmoid。

---

## 8. `conv_l2_launch` / `conv_state_update_launch`（`conv.cpp`）

GDN 的 depthwise 因果卷积 + q/k 部分的 L2 归一化。

### 8.1 `conv_l2_launch`（`conv.cpp:17-71`）

* 组维度 `group_dim = head_k_dim`，`n_groups = conv_dim/group_dim`，实现上 tap 循环 `#pragma unroll`
  固定为 4（即硬编码 GDN 的 4-tap 卷积）。
* `conv_state + slot*3*conv_dim` 是 `kernel_size-1 = 3` 行滑窗。
* `cross_row=true`：tap 索引是批次绝对 token `gp = rr*tpb + p`，可读前一行的 qkv，使一次调用覆盖整个
  chunk-batched prefill；`cross_row=false`：负索引读保存的 state。
* `silu_f` 后，前 `2*n_k_heads` 组（q 和 k 段）做 L2 归一化（SLM 归约 + `1/max(sqrt(red[0]),eps)`）。

### 8.2 `conv_state_update_launch`（`conv.cpp:76-116`）

把最后 3 个 qkv 行写回每 slot 状态，并在需要时写检查点：`end_tok = pos[rr]+n`，
`pc_active && end_tok % kBlockSize == 0` 时 `cap = pc_row_slot[end_tok/kBlockSize]`，写
`snap.base + cap*stride + layer_off + gdn_per + {0,1,2}*conv_dim + i`。`last_row_only` 用于完全融合的
prefill 模式。

---

## 9. `gdn_launch`（`gdn.cpp`）

Gated DeltaNet 递归。两个数学等价的实现：`gdn_kernel<C,WPW>`（标量）与 `gdn_f4_kernel<C,WPW>`
（float4）。

* `C` = 一个 warp 拥有的状态列（行）数；`col_groups = head_dim/C`；`total_warps = n_heads*col_groups`。
* q/k/v 在 `conv_out` 中打包（**q/k 用 key head 数，v 用 value head 数，两者一般不相等**）：
  `k_off = n_k_heads*head_dim`、`v_off = 2*n_k_heads*head_dim`。调用时 `head_dim = d_state`、
  `n_k_heads = n_group`、`n_heads = dt_rank`、`scale = 1/sqrt(d_state)`。
  `n_heads` 必须是 `n_k_heads` 的整数倍。
* **value head `h` 配对到 q/k 头 `h % n_k_heads`（取模/交错）**：
  `const int qk_head = head % n_k_heads;`。参考实现用 `ggml_repeat_4d` 展开 q/k 的 head 轴，其平铺语义
  是取模（`dst[i1*ne01 + k1] = src[k1]`），不是分块（`head*n_k_heads/n_heads`）。
  **两者的陷阱**：`n_group == dt_rank` 时（参考模型 Qwen3.5-0.8B，16 == 16）取模与分块**都是恒等映射**，
  错误实现完全不可见；Qwen3.8-27B 是 16 vs 48，分块会把 48 个 value 头与 q/k 配错，输出退化为重复
  （实测：模型只吐 1–2 个 token）。GPU 的 `gdn_kernel` 与 `gdn_f4_kernel`、CPU 的 `cpu_gdn`、
  以及 `tests/common/cpu_ref.h` 三处必须同步。
  注意这与 attention 的 GQA 展开**不同**：`attn.cpp` 用分块 `kvh = h*n_head_kv/n_head`（对应 HF 的
  `repeat_kv`/`repeat_interleave`）。两者不可互换。
* 状态布局 `state + slot*n_heads*head_dim*head_dim + (head*head_dim + col0)*head_dim`，即
  `[slot][value head][row][col]`，每 value head 一个 `head_dim×head_dim` 矩阵。warp 拥有 `C` 个连续行。
  `alpha`/`beta`/`ssm_a`/`ssm_dt` 都是**每 value head 一个标量**（共 `n_heads` 个）。
* 每 token 每 head：

```
bt   = sigmoid(beta)
sp   = softplus(alpha + dt_bias)
g    = exp(A * sp)
dots = S · k                      // warp 内归约
del  = (vval - g*dots) * bt
S    = g*S + k ⊗ del
out  = (S · q) * scale
```

* 状态在 token 循环前载入、循环后写回。
* 检查点：token 完成 32 块边界（`pc_on && (pbase+t+1) % kBlockSize == 0`）时，把该 warp 的 C 行状态写到
  `snap.base + st*stride + layer_off + (head*head_dim+col0)*head_dim`。
* 配置：`PF_GDN_COLS` ∈ {1,2,4,8}（默认 2）、`PF_GDN_WG` ∈ {1,2,4,8}（默认 8）、`PF_GDN_VEC` 默认开。
  `n_real >= 8` 且 `head_dim % cols == 0` 时才用列批处理（prefill），decode 保持 1 列/warp。
  `tpb_arg`/`nreal_arg` 允许融合调用覆盖 `info->tpb`/`info->n_real`。

---

## 10. `gated_norm_launch`（`gated_norm.cpp:14-44`）

每个 `(r,t,head)` 一个 warp，对 attention 输出做 gated RMSNorm：

```
inv = 1/sqrt(mean(a²)+eps)
out = a * inv * weight * silu(z)
```

`z` 是 GDN gate，`weight` 是 `ssm_norm`。调用参数 `n_heads = dt_rank`、`head_dim = d_state`。

---

## 11. KV 存储类型（`kv_type.{h,cpp}`）

```cpp
enum class kv_dtype_t : int { f32=0, bf16=1, f16=2, i8=3, i4=4 };
```

解析优先级（`kv_type.cpp`）：

1. `--kv-type T`（`kv_dtype_set`，在 engine 构造前生效）；
2. `PF_KV_F32 != 0` 或 `PF_KV_BF16 == 0` → f32；
3. `PF_KV_TYPE` 未设/空 → i8（默认）；
4. `PF_KV_TYPE = f32|fp32|0 / f16|fp16 / bf16 / i8|int8|q8 / i4|int4|q4`；未知 → 警告 + i8。

i8 几何：`[block][kv head]` 单元内是 `kBlockSize` 行 × `head_dim` int8，后接独立的
`kBlockSize × (head_dim/32)` fp16 scale 平面。成本 3 KB/token/layer（K+V）对比 bf16 的 6、f32 的 12。

i4 几何相同，但每个字节打包两个有符号 4-bit 值（低 nibble = 偶数 head dim，值域 `[-7,7]`，二补码），
一行是 `head_dim/2` 字节，scale 平面与 i8 一致；成本 1.5 KB/token/layer。`kv_dtype_bits` /
`kv_dtype_row_bytes` 给出每元素位数与行字节数，`kv_dtype_has_scales` 判断是否有独立 scale 平面。

`kv_ld_host` 是主机侧元素读取；i8/i4 返回反量化后的值（i4 用元素下标定位 nibble），
pool 的 scale 处理在 `engine::kv_read_vec`。

i4 读取细节：`i4_ld4` 用**一次对齐 16-bit 加载**取 4 个 nibble（`d` 是 4 的倍数，字节偏移
`d/2` 必为偶数），再做 4 次提取——两次标量字节加载会让 16k decode 明显变慢（该 kernel 在长
上下文是**指令**而非字节受限）。性能取舍：i4 端到端 decode/prefill 与 i8 持平，收益是容量
（同预算 2x 上下文）；短上下文 prefill attention 因解包 ALU 慢约 20%，但被 GEMM 稀释。测量见
[`reports/int4_kv.md`](../../reports/int4_kv.md)。

---

## 12. 视觉编码器 kernel（`vit.cpp`）

所有权重是 BF16（type 30）或 F32（type 0），行主序 `[N][K]`。

| kernel | 说明 |
|---|---|
| `vit_gemm_launch` | `out[t][n] = alpha*Σ W[n][k]*x[t][k] (+residual)`。128 线程 WG，64 行 × 32 token tile，lane `TP=4` token × `RP=4` 行，K 按 64 宽 SLM 分块，`KU=8`；bf16/f32 均用此 tiling |
| `vit_layernorm_launch` | 每子组一行，`sg_sum` 归约；`var = E[x²]-mean²`；可选 bias；支持输入/输出 stride |
| `vit_gelu_launch` | 原地 tanh 近似 GELU，与 `ggml_gelu` 一致 |
| `vit_add_bias_launch` / `vit_add_launch` | 二维逐元素加法，独立 stride |
| `vit_rope_launch` | 2D 视觉 RoPE（作用于 fused qkv 的 Q/K）：pair → section（每 `head_dim/4` 对一节），前 16 对用 patch 行、后 16 对用 patch 列，频率指数每节重置；从合并 token 反推 `(px,py)` |
| `vit_attn_launch` | **双向非因果** attention，`HD` 固定 64，`BQ=32` 查询块、`BK=64` 键块，K/V 经 SLM，在线 softmax，`1/sqrt(HD)` |

视觉 tower 的 RoPE 与文本模型的交错 M-RoPE 是两套不同的东西，见
[10-multimodal.md](10-multimodal.md)。

---

## 13. 新增 kernel

1. 新建 `src/backend/gpu/kernels/<name>.cpp`，定义启动函数、在同一 TU 内声明 kernel lambda、包含
   `"kernels.h"` 与 `"kernel_utils.h"`。
2. 在 `kernels.h` 声明启动函数。
3. 在 `CMakeLists.txt` 加入 `.cpp`（无 globbing）。
4. 新增 stage 测试：[12-build-and-testing.md](12-build-and-testing.md)。

---

## 14. CPU kernel 库（`src/backend/cpu/kernels/`）

CPU 后端是同一套算子的主机实现，**每个 kernel 一个 `.cpp`**，与 GPU 目录一一对应：

| 文件 | 内容 |
|---|---|
| `common.{h,cpp}` | 线程池（自旋+休眠混合、`ready` 握手）、ISA 分派表（AVX2 / AVX-VNNI / AVX-512）、融合反量化 GEMV（`gemv_row` / `qgemv_sb_*`）、RMSNorm 辅助 |
| `rmsnorm.cpp` `embed.cpp` `copy_row.cpp` `gemv.cpp` `qk_norm_rope.cpp` `attn.cpp` `conv.cpp` `gdn.cpp` `gated_norm.cpp` `xq.cpp` `dp4a.cpp` `i8.cpp` | 对应的 `cpu_*` 启动函数（声明在 `src/backend/cpu/cpu_types.h`） |

要点：

* 整个目录用 **`-fno-sycl`** 编译（CMake 的 `SI_CPU_SOURCES` 列表），因此 `<immintrin.h>` 与
  `__attribute__((target(...)))` 只在 host pass 生效，不进入 SYCL device pass。
* `cpu_types.h` 无 SYCL 依赖：`cpu_step_info` / `cpu_gemv_seg` 是 `step_info` / `gemv_seg` 的镜像；
  `src/backend/cpu/cpu_backend.cpp` 在调用边界做转换。
* 热点循环（点积、归约、整数 `maddubs`）按 `src/common/cpu_isa.h` 运行期选变体；
  `PF_CPU_ISA=scalar|avx2|avx512|avxvnni` 可强制。结构类 kernel（paged attention、conv/GDN、
  RoPE/M-RoPE）为标量/自动向量化，语义与 GPU kernel 一致。
* `gemv_seg.i8`（`cpu_gemv_seg.i8`）为 CPU 专有：为真时 `cpu_i8_gemv`/`cpu_i8_gemm` 从 `seg.w`
  指向的 GGUF block 直接做整数点积；Q8_0/未知格式与 scalar ISA 回退到融合 fp32。
* 新增 CPU kernel 的步骤见 [AGENTS.md](../../AGENTS.md) 的 “Add a CPU kernel”
  （同时加入 `CMakeLists.txt` 与 `SI_CPU_SOURCES`）。
