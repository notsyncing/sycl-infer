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
  [04-engine.md](04-engine.md)。**逐步状态一个都不许在主机侧读**：那是图捕获时会被冻结成 0 的量
  （`xq.cpp` 的注释与 `embed_launch` 的固定网格都是同一原因）。
* 启动函数里的设备查询一律走缓存（见 §1.3）。

### 1.1 `step_info` 与 `gemv_seg`

```cpp
struct step_info {                 // kernels.h:76-119
    int32_t n_rows, n_real, tpb;   // 行数、每行 token 数、每行 buffer 的 token 槽数
    int32_t n_real_row[kMaxB];     // 模式 2 每行真实 token 数（最后一行可不足 kMaxT）；0 = 用 n_real
    int32_t pos[kMaxB], slot[kMaxB], active[kMaxB];
    int32_t tokens[kMaxB * kMaxT];
    int32_t pc_active, pc_stride; float * pc_base;   // 前缀缓存快照
    int32_t pc_row_slot[kPcMapLen];
    int32_t mtp_dt, mtp_dry;       // MTP：逐 token 快照 / dry verify（算但不写回递归状态）
    int32_t mrope_on, mrope_sections[4];             // 多模态
    int32_t mrope[4*kMaxB*kMaxT];
    const float * img_embd; int32_t img_row[kMaxB*kMaxT];
};
```

**`n_real_row` 是模式 2 的一部分**：chunk-batched prefill 的最后一行可以是部分行，因此每个"这一行有多少
个真实 token"的判断都必须走 `row_nr(info, r)`（`kernel_utils.h:127-130`，`n_real_row[r] > 0 ? n_real_row[r]
: n_real`），而不是直接读 `info->n_real`。漏掉它就是 mode-2 prefill 的老 bug：`embed`/`attn`/`gdn`/`conv`
多算一段 token，行缓冲里留下一段陈旧状态。

`gemv_seg`（`kernels.h:36-73`）描述一个权重矩阵的应用：`w/type/K/n_rows/dev`、输入 `x/x_stride`（可选
`act_up` 做 `silu(x)*up`）、输出 `out/out_stride`（可选 `residual`）、`alpha`、可选 fp32 `meta32`
（预先抽好的 per-32 `(scale, min)`，让 decode GEMV 跳过 6-bit scale 解码），三条互斥的权重来源字段：
SIn int8 的 `w8`（配 `x8/xmeta/xsumq`，见 §5.3）、oneDNN 转换后的 `wi8/wsc`（行主序 int8 + 每行一个
fp32 scale，M=1 解码直接读，见 §5.2）、`w_raw`（原始 GGUF 字节，只给 `PF_MTP_LAYER_EXACT` 的 fp32
参考路径用，因为它必须跟着 segment 走：`seg_plan` 按 `(device, type)` 排序，按位置绑定会把张量接到
错的 segment 上），以及 CPU 专有的 `i8`（真值时 CPU 直接从 `w` 指向的 GGUF block 做整数点积）。

### 1.2 `kernel_utils.h` 共享辅助

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
| `row_nr(info, r)` | 模式 2 的真实 token 数（见 §1.1） |
| `gemm_ws(queue&, size_t)` | 进程级 split-K workspace（定义在 `dp4a_common.cpp`） |

### 1.3 设备 profile：launcher 的形状从哪里来

**规则：在一张卡上实测出来的常量不进使用点，进那张卡的 profile；使用点按名字问 `si::dev::active()`。**
`profile` 结构在 `src/device/device_profile.h:52-138`，一张卡一个文件（`src/device/profiles/<card>.cpp`）
加注册表一行（`src/device/device_registry.cpp:22-26`）——注册、选择与 fallback 机制见
[architecture.md](../architecture.md)，这里只讲 kernel 这一侧怎么取值。

| 组 | 字段 | 读点 | 作用 |
|---|---|---|---|
| `shape.*` | `rmsnorm_wg` | `rmsnorm.cpp:71` | 每行一个工作组；选 1024/512/256/128 哪个实例化 |
| | `gemv_rows_per_wg` | `gemv.cpp:218` | 向量化 decode GEMV 每组的行数 |
| | `gdn_cols` / `gdn_warps_per_wg` / `gdn_vec_max_rows` | `gdn.cpp:258` / `gdn.cpp:264` / `gdn.cpp:302` | GDN 每 warp 的状态行数、每组 warp 数、float4 变体的 `n_real` 门槛 |
| `occ.*` | `warps_per_eu_x2` | `engine.cpp:245` | decode attention 的 K-split：`dec_splits = (wpe/2)*EUs/n_head`，向下取 8 的倍数、下限 8、上限 `kMaxDecSplits` |
| `split.*` | `gemv_rows` / `max` | `dp4a_gemv.cpp:105-113` | decode GEMV：`N <= 2*gemv_rows` 时切 K，`S = ceil(gemv_rows/N)`，上限 `max` |
| | `gemm_rows` / `max` | `dp4a_gemm.cpp:1085-1088` | 同上，prefill row GEMM（仅 `TB==32 && N<=8192`） |
| `slm.*` | `budget_bytes` | `w4_gemv.cpp:153/277/509/625/788` | staging tile 的 `fits(RB)` 判据；48 KB 是在 64 KB 硬件上限下的**主动预留** |
| `attn.*` | `vec` / `dec_group` | `attn.cpp:20` / `attn.cpp:303` | 向量化经典 kernel / 分组 decode kernel 的默认（后者两卡都是 off，但理由相反，见 §7.4） |
| | `xmx` / `xmx_min_keys` | `attn_xmx.cpp:54` / `attn_xmx.cpp:72` | oneDNN prefill attention 开关与键数门槛 |
| | `xmx_gather_red` / `xmx_gather_red_max` | `attn_xmx.cpp:87-88` | gather 一级 max 归约的工作组数与上限 |
| | `split_keys` | 目前无读点：`engine_graph.cpp:1097-1101` 直接把 `PF_ATTN_SPLIT_KEYS` 默认成字面 512 | 字段已就位，engine 尚未改用它 |
| `wt.*` | `w4` / `k5` / `cb4` | `engine.cpp:1132` / `1138` / `1143` | 原生 u4 / 5-bit / codebook 存储的默认 |
| `hw.*` | `compute_units` | `engine.cpp:241` | SYCL 查不到 EU 数时的兜底 |
| | `max_work_group_size` | `device_registry.cpp:153` | `wg_clamped()` 的兜底 |

四条规矩：

1. **env 覆盖 profile，profile 值才是实测的那个**：一律写成 `e ? atoi(e) : dp.<field>`（如
   `attn.cpp:20`），顺序不要反——一个从未被实测过的 env 默认值就是 bug。
2. **`shape.rmsnorm_wg` 是硬件差异，不是调优**。A770 接受 1024 线程的工作组，Iris Xe 只接受 512，
   无条件 1024 的 kernel 在后者上**根本 launch 不了**。`si::dev::wg_clamped()`
   （`device_registry.cpp:134-163`）把 profile 想要的宽度减去设备真实上限再取整到 32 的倍数，于是未知卡
   拿到的是更小的 kernel 而不是一次 launch 失败。
3. **work-group 宽度必须是模板参数**（`rmsnorm_impl<WG>`，`rmsnorm.cpp:26-65`）：SYCL kernel 不能捕获
   运行时初始化的全局量，而宽度本来就是**每卡编译期**的量；模板化还让 strided 载入循环能被展开
   （`n_embd=5120` 时 1024 线程把串行载入从 20 次降到 5 次）。
4. **永远不要在 kernel launcher 里查设备。** `rmsnorm_launch` 每次 launch 都调 `wg_clamped()`，而
   `sycl::device::get_devices()` 要走驱动枚举每一张 GPU。缓存前实测 **3.2 ms/次**，一次 MTP verify 里的
   65 次 rmsnorm 就是 **210 ms 纯主机开销**；普通 decode 完全看不到（图把 launch 录一次再重放），
   所以它在 decode 数字里隐形、在推测 cycle 里主导。查询只做一次（函数内 static，
   `device_registry.cpp:143-159`），修好后 verify 221 → 85 ms、cycle 255 → 107 ms。

**每张卡必须自己写全，包括 provenance。** `arc_a770.cpp` 的 `.provenance` 串（19-26 行）与逐字段注释给出
了测量条件（1-2x A770 16 GB、512 EU、`--layer-map 0-31:gpu.0,32-63:gpu.1`、attn+combine 在 64k 深度下
nsp 64/128/144/160/168/176 → 3.05/2.18/1.75/1.70/1.74/2.68 ms 等）。`iris_xe.cpp` 里绝大多数字段标
**INHERITED**：两卡同属 Xe-LP，所以 8-warp/EU 的子组格子相同，但**最优点没有重扫**——原因写在
`iris_xe.cpp:24-32`，这类集成卡与 host 共享内存，本地未跟踪的压测工具（`dev/` 在 `.gitignore` 中）
`bench_attn27 --dec-only` 在有负载的机器上同一配置三次重复给出 2.1-4.4 ms，噪声大于效应。未测量要写出来，
不要假装测过。同一个值、两份相反的证据，正是这个拆分存在的理由（`attn.dec_group` 见 §7.4）。

### 1.4 kernel 相关环境变量（`AGENTS.md#environment-variables` 的子集）

只列与 kernel 形状/存储选择直接相关的；完整表以 `AGENTS.md` 为准。

| env | 默认 | 读点 |
|---|---|---|
| `PF_DEVICE_PROFILE` / `PF_DEVICE_INFO` | unset / off | 强制某个 profile（`PF_DEVICE_INFO` 打印全部字段与 provenance） |
| `PF_W4` / `PF_K5` / `PF_CB4` / `PF_DP4A` / `PF_GEMM_DNNL` | 来自 profile 的 `wt.*` | 存储格式选择，权重在 load 时定型 |
| `PF_W4_K5` / `PF_W4_ALL` | off | 把 Q5_K / 其它类型也放到 per-32 u4 网格上（精度决策，不是调优；见 `AGENTS.md`） |
| `GEMV_DEC_VEC` / `GEMV_VEC12` / `GEMV_VEC13` | on | fp32 GEMV 的向量化 decode 变体（§5.1） |
| `GEMV_CFG1` / `GEMV_CFG8` / `GEMV_CFG16` / `GEMV_CFG32` | unset | 实验性 `RPS*100+SGW` 配置码（§5.1） |
| `PF_GEMV_SPLIT` | on | decode DP4A GEMV 的 K-split（§5.2） |
| `PF_GEMM_ROW` / `PF_GEMM_ARCH` / `PF_GEMM_TILE` | on / 1 / 按类型 | prefill DP4A GEMM 的家族选择（§5.3） |
| `PF_GEMM_SPLIT` / `PF_GEMM_WG` / `PF_GEMM_SG` / `PF_GEMM_XSLM` | 0（自动）/ 128 / 16 / off | K-split row 路径（§5.3） |
| `PF_MT_R` / `PF_MT_R2` / `PF_MT_TB` / `PF_MT_WG` / `PF_MT_WG2` / `PF_MT_PF` / `PF_MT_SLM` | 2 / - / 16 / 128 / 128 / off / off | M-tiled 变体（§5.3） |
| `PF_DEC_R` | 1 | oneDNN int8 权重解码 GEMV 每子组行数（§5.2） |
| `PF_W4_RB` | 0（自动 16） | u4 / w2 / i8 各 GEMV 的 `RB` 覆盖（k5、cb4 只按 `fits()` 选） |
| `PF_W4_GEMM_MAXM` / `PF_W4_GEMM_U` / `PF_W4_GEMM_TN` | 1 / 1 / 0 | 旧版 `w4_gemm_launch` 的 A/B（§5.8 末尾） |
| `PF_NAT` | on | 关掉 `nat_gemm_launch`（§5.8） |
| `PF_ATTN_VEC` / `PF_ATTN_FUSE` / `PF_ATTN_FLASH` / `PF_ATTN_WAIT` | on / on / off / off | attention 变体（§7） |
| `PF_DEC_GROUP` / `PF_DEC_SPLIT` | profile 的 `attn.dec_group`（0） / `(wpe/2)*EUs/n_head` | 分组 decode kernel / decode K-split |
| `PF_ATTN_SPLIT` / `PF_ATTN_SPLIT_KEYS` / `PF_ATTN_DBG` | unset / 512 / off | prefill attention 的 split 数与每段键数 |
| `PF_ATTN_XMX` / `PF_ATTN_XMX_MIN` | profile 的 `attn.xmx`（1） / 2048 | oneDNN prefill attention 开关与门槛（§7.5） |
| `PF_XMX_GRED` / `PF_XMX_TAILBLK` / `PF_XMX_BREAKDOWN` / `PF_XMX_TIME` / `PF_XMX_DBG` | 64 / on / off / off / off | §7.5 的归约工作组数、尾块宽度与计时 |
| `PF_GDN_COLS` / `PF_GDN_COLS_MIN` / `PF_GDN_WG` / `PF_GDN_VEC` | profile（2 / 8 / 8） / 8 / 规则（见 §9） | GDN 形状 |
| `PF_CPU_ISA` / `PF_CPU_THREADS` | 自动 / 物理核 | 主机后端 ISA 变体与 worker 数（§14） |

---

## 2. `rmsnorm_launch`（`rmsnorm.cpp:68-81`，kernel 体 `rmsnorm.cpp:26-65`）

```
out_i = x_i * (1/sqrt(mean(x²)+eps)) * w_i
```

每行一个工作组，**工作组宽度 `WG` 是模板参数**，按 §1.3 取自 profile 并被设备上限夹住：
`rmsnorm_impl<128|256|512|1024>`。为什么必须是模板：宽度是每卡的量（Iris Xe 只接受 512），
而且 SYCL kernel 不能捕获运行时初始化的全局量；模板化还让 strided 载入循环能展开。

* 归约是**两级**的：先在子组内 `reduce_over_group`，每个子组写一个 SLM 槽 `red[NSG]`，barrier 之后由
  0 号子组再归约一次写回 `red[0]`，第二次 barrier 后所有线程读它——只有 2 个 barrier。旧的 256 线程
  版本是在 256 float 的 SLM 上做 256 宽树状归约（8 个 barrier）。
* 归约覆盖**整个工作组**而非单个子组，因此支持任意 `n`（不要求 `n % WG == 0`）。
* 发射量 `nd_range<1>(n_rows*WG, WG)` 里 `n_rows` 是主机知道的张量行数（每步一行），逐 token 状态
  不参与，所以这个 kernel 可以被录进命令图。

实测（`rmsnorm.cpp:12-24`）：256 线程 / `n_embd=5120` 是 20 次串行载入 + 8 级 SLM 树，测得 32 µs/call
= 27B 一次 85 ms 多设备 decode 里的 4.1 ms；1024 线程把串行载入降到 5 次、barrier 降到 2 个。

---

## 3. `embed_launch`（`embed.cpp:14-41`）

把每个 active token 的 embedding 行反量化到 fp32 输出，并做多模态替换。

* 网格固定为 `nd_range<1>(kMaxB*kMaxT*n_sb*256, 256)`，`n_sb = n_embd/256`。**故意固定**：`n_rows`/
  `n_real` 是设备侧值，主机计算 extent 会在图捕获时被冻结为 0。
* 组 `g` 分解：`t = g/n_sb`（扁平 token 槽），`sb = g%n_sb`（行内 256 元素 super-block），
  `row = t / info->tpb`；`row >= info->n_rows` 或该槽超过 `row_nr(info, row)` 就直接返回
  （模式 2 的最后一行可能不满）。
* 多模态：若 `img_embd != nullptr` 且 `img_row[t] >= 0`，直接拷贝
  `img_embd[img_row[t]*n_embd + sb*256 + tid]`，**不查 token 表**。
* 否则 `row = table + tokens[t]*row_bytes + sb*sb_bytes`，值 = `dequant_elem_sb(type, row, tid)`。
  支持 Q4_K/Q5_K/Q6_K/Q8_0/F32；未知类型返回 0。

`row_bytes` 由调用者提供，包含 padding（`sb_bytes*n_sb` 不一定等于 GGUF 行大小）。

---

## 4. `copy_row_launch`（`copy_row.cpp:11-26`）

把一个 activation 行（`n` floats）从 `src` 拷贝到 `dst`。网格是单个 256 线程组 + grid-stride 循环。
`row < 0` 时取批次最后一个真实 token：`rr = n_rows-1`、`r = row_nr(info, rr)`，行号 `rr*tpb + r - 1`
（`info->n_real_row` 存在正是为了这个——模式 2 的最后一行可能不满，用 `n_rows*n_real-1` 会越过行尾）。
唯一调用点：prefill 时把最终 post-RMSNorm hidden state 拷进连续的 `d_last_hidden`（模式 != 0）。

---

## 5. GEMV / GEMM

### 5.1 fp32 / 按需反量化 GEMV（`gemv.cpp`）

`gemv_multi_kernel<QT,TB,RPS,NSB,SGW>`（`gemv.cpp:18-148`）：

* `QT` = ggml 类型（12/13/14/8/0 为默认分派，另覆盖 11/20/21/23 = Q3_K/IQ4_NL/IQ3_S/IQ4_XS；
  0 = 原始 fp32）；`TB` = 每 segment token 数；`RPS` = 每子组行数；
  `NSB` = super-block 数（0 = `K/256`）；`SGW` = 每 work-group 子组数（默认 8）。
* `rows_per_wg = RPS*SGW`，`n_wg = ceil(total_rows/rows_per_wg)`，线程数 `SGW*32`；另有 `n_tb` 网格维
  覆盖多 token 块（模式 2 一次 dispatch 覆盖全部 chunk 行）。
* 每个 work-group 扫描 segment 列表定位自己的 `row_base`，把描述符缓存进寄存器（`gemv.cpp:52`：
  重新从设备内存读比从寄存器读贵）。
* lane 是 super-block 内 k 位置，`dequant_sb_lane_typed<QT>` 产出 `w[8]`（8 个子块）。
* `TB==1` 直接读 `X`；`TB>1` 先把激活行（应用 `silu(gate)*up`）staging 进 SLM `xs[256*TB]` 并 barrier
  （前后各一个 barrier，因为同一个 SLM tile 要被每个 `sb` 复用）。
* 最后 `sg_sum` 归约 `(row,t)`，lane 0 写 `alpha*v (+residual)`。

`gemv_dec_vec_kernel<QT>`（`gemv.cpp:214-312`）：仅 Q4_K/Q5_K 的单 token 向量化 decode。每工作组
256 线程、每子组一行，**行数来自 profile**（`shape.gemv_rows_per_wg`，两卡都是 8，`gemv.cpp:218`：
它是工作组大小，所以随“想让多少 warp 常驻”缩放）。lane 覆盖**一个子块内 8 个连续值**
（`s=lane/4, m=lane%4`），因此每 lane 只需一个 `(scale,min)` 和一次 `uint2` 加载；激活是两次 `float4`
加载。Q5_K 的第 5 位在这里是 8 字节 `qh` + 按 `2c+half` 取位，不需要 k5 的 SLM LUT。注释记录
约减少 1.5× 指令/值。

`gemv_group_launch`（`gemv.cpp:314-442`）：`n_tb = (TB>1 && n_tok_blocks>0) ? n_tok_blocks : 1`。
`GEMV_DEC_VEC`（默认开）、`GEMV_VEC12`、`GEMV_VEC13`（默认开）决定 `TB==1` 且类型为 12/13 时是否走向量化
kernel。`GEMV_CFG1/8/16/32` 是实验性的 `RPS*100+SGW` 配置码（104/204/208/216/404/408/416/808）。
默认由 `GEMV_DISPATCH_TB` 选择 `NSB ∈ {4,8,14,0}` 与 `TB ∈ {1,8,16,32}`。

### 5.2 DP4A GEMV（`dp4a_gemv.cpp`）

`dp4a_gemv_impl<QT,SPLIT>`（`dp4a_gemv.cpp:18-76`）：

* `WG=128` = 4 子组 × 32 行；`n_wg = ceil(N/128)`，grid = `n_wg*n_split`。
* `row = (blk/n_split)*128 + local_id`：**一 lane 一输出行**，一个子组覆盖 32 连续行。因为 32 行同属一个
  128 行块（`kRB`，`common/w8.h:47`）且连续，warp 的权重加载是 `32*GB` 连续字节。
* 读取：`w8_group_expand<QT>` 展开 GBP 字节为 dp4a word；`w8_sw_mw<QT>` 取 `(sw,mw)`；激活 scale
  `sx = xmeta[gx].x()`；`gx = g`（G=32）或 `g>>1`（G=16，两个 Q6_K 组共享一个 32 值激活组）；
  `uint4 xw` 在子组内广播；内层 4 个 dp4a word。
* 累加 `acc += sx*(sw*dot - mw*c)`，`c = sum(qx)`。
* `SPLIT` 时写 partials，否则 epilogue 应用 `alpha*acc (+residual)`。
* `dp4a_gemv_launch`（`dp4a_gemv.cpp:93-131`）：`PF_GEMV_SPLIT` 默认开；`N <= 2*split.gemv_rows` 时
  `S = ceil(split.gemv_rows/N)`、上限 `split.max`（两卡 2048 / 8，即 `N<=4096` 时切 2..8 份）。
  decode 对小输出尺寸（4–8 个 workgroup）同样是延迟受限，因此像 prefill 一样切 K。

**同一文件里的另外两条解码路径**（oneDNN int8 权重，不走 SIn）：

| kernel | 几何 | 说明 |
|---|---|---|
| `i8_row_gemv_launch`（`dp4a_gemv.cpp:139-171`） | 一个输出行一个 256 线程组，lane 跨步归约 K | 行主序 `[N][K]` int8 + 每行一个 fp32 scale（`sx_dev` 在 kernel 内从 USM 读）。oneDNN primitive 在 M=1 被 per-call 开销主导，所以解码直接读它的权重缓冲 |
| `i8_row_gemv_multi_impl<R>`（`dp4a_gemv.cpp:178-246`） | `WG=256`、`LPR=32/R` lane 一行、每组 `(WG/32)*R` 行 | 每 lane 一条更深的 K 循环（10 → 40 次迭代，R=4）才是隐藏 DRAM 延迟的关键；权重 XOR 到无符号后 `128*sum(x)` 每行校正一次。`PF_DEC_R` ∈ {1,2,4,8}（默认 1） |

### 5.3 DP4A GEMM（`dp4a_gemm.cpp`）

kernel 家族（全部 int8、硬件 dp4a）：

| 家族 | 说明 |
|---|---|
| `dp4a_gemm_impl<QT,TP,RP>` | 寄存器 tiled，`WG=128`，lane 拥有 `TP×RP` 累加 tile；`PF_GEMM_TILE` 选择 (TP,RP)，Q4_K 默认 12 = (1,2)、其它 42 = (4,2) |
| `dp4a_row_gemm_impl<...>` | **一 lane 一输出行**，`TB_T` 个累加器，权重全合并、x/meta 广播；`MT` 时第二网格维切 token；主管道（`PF_GEMM_ROW` 默认开） |
| `dp4a_row2_mt_gemm_impl<...>` | 2 行/lane、`TB_T=16`，32 个累加器，两行共享 x/meta（每 cell 指令少约 25%），可选 SLM staging（`PF_MT_SLM`）与软件预取（`PF_MT_PF`） |
| `dp4a_gs_mt_gemm_impl<...>` | “gemmstone 风格” 4×4 tile（`PF_GEMM_ARCH=2`），约 3.1 MAC/instr（对 R2 的 ~2.5） |
| `dp4a_row2_gemm_impl<...>` | 子组仍覆盖 32 行 × 32 token，但一 lane 拥有两个相邻行和一个 16-token 半块；把这些 x/meta 载入提出去实测快 45% |
| `dp4a_row_gemm_slm_impl<...>` | SLM staging 版本（`PF_GEMM_XSLM`，默认较慢：staging + 占用代价 > 省下的 L1 延迟） |

**split-K workspace**：`gemm_ws(q, need)` 是进程级单例设备缓冲，按需增长（`dp4a_common.cpp:10-25`）。
注释解释布局原因：chunk-batched prefill 中多个行的 GEMM 同时在飞，command graph 不对 USM 访问排序，
因此每个 row/split 必须有独立槽位。partial 布局 `ws[((s*TB_T + t)*N) + row]`，reduce 内核求和后应用
alpha/residual，写 `out[t*out_stride + row]`。注意 partials 用张量自己的 `N` 而非 `out_stride`——
后者可以更大（`ffn_gate`/`ffn_up` 共用 `2*n_ff` 缓冲），而 workspace 按 `S*TB*w.N` 定量。`n_rows`
（张量行数）与 `out_stride` 不同这一点在 §5.3 与 `dp4a_gemm.cpp:1095-1098` 都要注意。

**TB 语义**：`TB` 是一次 call 的 token 数（模式 1 = 32，decode = 批大小，模式 2 = 整个扁平 token 数）。
`TB_T` 是编译期 tile，`tstride` 是激活 token 步长；dispatcher 只在 `TB == TB_T`（或 MT 时
`TB == tstride`）时选择对应变体，使 token 循环无 guard。

**调度与调优**（`dp4a_gemm_launch`，`dp4a_gemm.cpp:832-1307`），从上到下第一个命中的生效：

1. **M-tiled 路径**：`TB > 32 && TB % 32 == 0 && TB <= kMaxB*kMaxT` 时启用（chunk-batched prefill）。
   `PF_GEMM_ARCH=2` 走 gemmstone 变体；否则 `PF_MT_R`（默认 2，`PF_MT_R2=0` 才回到 1）选
   `row_gemm_impl`（R=1，配 `PF_MT_TB` 的 8/16/32 宽 tile）或 `row2_mt_gemm_impl`
   （R=2/4，`PF_MT_TB`、`PF_MT_WG2=64`、`PF_MT_PF`、`PF_MT_SLM`）。
2. **K-split row 路径**：`TB == 32 && N <= 8192` 时启用。`S` 由 `PF_GEMM_SPLIT` 给定，否则
   `ceil(split.gemm_rows/N)`，上限 `split.max`（两卡 16384 / 8）；`PF_GEMM_WG` 选 128/64/32，
   `PF_GEMM_SG=32` 恢复旧的 SIMD32 码（默认 SIMD16，实测更快且块读需要），`PF_GEMM_XSLM` 才走 SLM
   staging 且只对 `type != 14 && N >= 2048`。
3. **无 split row 路径**：`N >= 2048 && TB ∈ {32,16,8}`。
4. **tiled 回退**：由 `PF_GEMM_TILE` 选择（12/14/24/42/44，默认按类型 12 或 42）。

注释中的实测：row 映射在同一数据上约 12 GB/s，tiled 约 4 GB/s；row kernel 需要约 16k 输出行才能填满
机器，因此小张量切 K；更小的 WG 提升常驻 warp 数。

**编译期开关（不是环境变量）**：`PF_MT_DEBUG` 是 `#ifdef` 宏（`dp4a_gemm.cpp:190`）。定义后给 MT 路径
加一段"在当前 group 的 32-token 主体运行时预取下一个 group 的打包权重字节"的实验块（载入有约 2400
个周期可以到达），并打印一行一次性的 `[mtk] K=… N=… TB=… tstride=… n_tt=… grid=…`。该实验块注释里写的
`PF_ROW_PF=0`（`dp4a_gemm.cpp:225`）用来关掉这个实验，但代码里没有任何地方读它，所以它同样是编译期的。
MT 路径真正的环境变量开关是上表列出的 `PF_MT_R`/`_R2`/`_TB`/`_WG`/`_WG2`/`_PF`/`_SLM`。

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
  64 B line 覆盖 32 个 f16），SLM 存 f16（u4 两个平面也能用 `RB=16`）。实测 **10-25 %**
  （A770，`w4_gemv.cpp:18-25`），输出与改前逐位一致（本地未跟踪的压测工具，
  `dev/bench_decgemv_eq.cpp`，`dev/` 在 `.gitignore` 中）。`RB` 依 `K` 在 16/8 间回退，
  判据是 `2*RB*ng*2 B` 是否装得进 `slm.budget_bytes`（`w4_gemv.cpp:153`）。整机 tg128
  的结果见 `AGENTS.md`。

**`i8_grp_gemv_rows_multi_launch`（`w4_gemv.cpp:308-429`）——把一个 call group 的窄 int8 段合成一次
launch。** GDN 的 `ssm_alpha` / `ssm_beta` 是两个同 K、同量化激活的 48 行 int8 段；每段一次 launch 的
代价是暴露的延迟而不是带宽（M=5 下 K=5120×32 行 40.1 µs、×96 行 40.9 µs，而 launch 地板本身是 5.8 µs），
27B 一次权重 pass 里 48 层 × 2 次 launch 就是 ~4 ms。融合后 68.3 → 66.5 ms（M=5）、
53.9 → 53.4 ms（M=1）。内核形状：

* `i8_grp_gemv_rows_multi_impl<M,NS>`（`w4_gemv.cpp:309-389`）：**一个输出行一个工作组、32 线程**
  （一个子组），`NS ≤ 4`；行 → `(segment, row)` 的映射是对捕获到的 NS 个描述符做一次扫描，
  所以没有跨工作组归约，子组树是唯一的归约。每行 stage 自己的 `ng` 个 f16 scale（该段的
  `[K/32][n_rows]` 平面）进同一个 SLM 槽，然后跑与 M=1 GEMV 相同的 group 循环——**每行与
  `i8_grp_gemv_launch` 逐位一致**（同样的 lane→group 映射与归约）。算术相同，只是 `M` 个累加器。
* 12 个参数 / 一个 local accessor（对比 `nat_gemm` 的 25 个 / 6 个）；`M = 2..13` 与
  `n_segs = 1..4` 都是模板化分派（`i8_grp_gemv_rows_launch` 就是 `n_segs=1` 的包装，
  `w4_gemv.cpp:431-436`）。端到端：verify **94.0 → 91.7 ms**（各三次重复，-2.4 ms，确定性）。
* **必须记录的坑**：`i8_grp_seg` 描述符要**按值**拷进 lambda 闭包（`w4_gemv.cpp:312-320`）。调用方把它
  们放在主机栈上，设备侧解引用读到的是零——一个看起来完全正常、**静默为空**的 GEMV，实测还"快"了
  1.9 ms。按值拷贝才会进 kernel 参数 blob。
* 默认开（`engine_graph.cpp:708` 的 `PF_NOFUSE` 关掉、`PF_FUSEDBG=1` 打 trace）。它用的是 M=1 GEMV 的
  累加顺序而非 `nat_gemm` 的，所以会重排 emit 流的接受率：一个 prompt 上 1.64 → 1.53，另两个不变
  （`-1.1 … +0.9 ms/token`）。

### 5.5 codebook 4-bit（IQ4_XS / IQ4_NL，`cb4_*`，`w4_gemv.cpp`）

值不是网格而是 `scale[g][n] * kvalues_iq4nl[q]`（16 项 int8 码本 × 每 (行,组) f16 scale），
所以权重存 4-bit 索引 + scale（0.5625 B/w vs int8 1.0625），**native 值精确**。

* `cb4_gemv_launch`（decode，M=1）：结构与 §5.4 的 int8 GEMV 相同（g 外/行内 SLM staging、
  `asa`/`XS`/XOR 偏差校正），差别只在权重字由 16 字节 nibble 经 **256 项 SLM LUT** 展开：
  `w[2j]` 取字节 `j`/`j+1` 的低 nibble（一次 256 项查表同时拿回两个字节，拼成一个 16 位的两个
  dp4a 半字节），`w[2j+1]` 取同样这两个字节的高 nibble。LUT 就是承重的部分：两个索引字节 → 4 个码本值
  （一个 dp4a 操作数）只需 2 次查表 + 1 移位 + 1 或；上一版形式每个操作数要 4 次字节查表 + 3 移位 +
  3 或（每 32 值 32 次 LUT 载入 + 24 条 ALU），实测只有卡读上限的 **49-67 %**，即**指令受限**而不是
  带宽受限。
* `cb4_expand_launch`（prefill 用）：把索引展开成 int8 到 `dnnl_gemm` 的复用 scratch，再跑
  已有的 int8 primitive（oneDNN 每次 execute 重读权重 memory，已验证）。一个 work-item 处理一组
  （16 字节载入 + 两个 16 字节写出）；标量逐字节的写法让 prefill 慢约 1.5×，而 k5 展开的
  “8 次 32-bit 分散写”形式只到上限的一半。
* 元素序（§10.2 of 02-quantization.md）：每组 16 字节**交错序**（byte k = 元素 2k 低 nibble /
  2k+1 高 nibble）——`cb4_pack`、GEMV、展开三处必须一致；native 序在 pack 里转成交错序。
  GEMV/展开都用 **256 项 `lut16`**（`value(b&0xF) | value(b>>4)<<8`，GEMV 里在 SLM、
  展开里用设备副本）：一个索引字节 → 半个 dp4a 操作数。
* 逐位对拍：本地未跟踪的压测工具 `dev/bench_cb4.cpp`（`dev/` 在 `.gitignore` 中）同时算 host 公式、
  kernel 逻辑的 CPU 仿真、GPU 三者（随机 nibble，max rel 0.000000）。

### 5.6 原生 5-bit（Q5_K，`k5_*`，`w4_gemv.cpp`）

* `k5_gemv_launch`（decode，M=1）：与 `w4_gemv_launch` 同构（同 `RB=16`、同 g 外/行内 f16
  SLM scale staging、同 `asa`/`XS`/修正公式），权重侧多一个平面：`vals` 的 uint4
  给出 32 个 nibble，`hi`（每 32 组 4 字节）按 split 序取第 5 位，`lt[16]` 把 4 bit 展开成
  4 字节 0/1，`l0 |= lt[…] << 4` 后与 §5.4 一样 8 次 `dp4a_s8u8`。4 条独立累加链
  （单链慢 ~3 %）。整个额外开销是每 32 值 8 次 SLM LUT 载入 + 16 条 ALU，换来 0.75 B/weight
  而不是 int8 转换的 1.125——它**完全藏在带宽里**：实测就贴着卡读上限 ~405 GB/s，而 int8 GEMV 只达到
  其中 92 %（`w4_gemv.cpp:530-531`）。作为速率参照，`AGENTS.md` 记录该形状（M=7、K=5120、N=17408）
  下 M=1 GEMV 的 u4/k5/cb4/int8 = 294/249/261/364 GB/s，对比同一次测量里 §5.8 的
  `nat_gemm` 223/188/173/323 GB/s。
* `k5_expand_launch`（prefill）：元素序 int8 = `lo4 | (bit << 4)`。偶/奇两半各自用
  `bit_lut` 展开成字节后按 4 字节字内交错（`p = (ev & 0x00FF00FF) | ((od & 0x00FF00FF) << 8)`，
  `t` 同理），每 8 值 2 个 `uint4` 存储。`bit_lut` 由 `dnnl_gemm` 持有（16×uint32，设备内存）。
  写必须是 `uint4`：64-bit 的 byte-spread 每 8 值约 40 条指令，把内核压到 **200 GB/s**
  （读上限的 49 %，指令受限）；32-bit 形式约 12 条。
* 元素序不变量（`k5_pack`、GEMV、展开三处）：`vals` 交错（byte k = 2k/2k+1），`hi` split
  （bit i = 偶元素 2i，bit 16+i = 奇元素 2i+1）。逐位校验：`test_k5_gemv`
  用真实模型的 Q5_K 张量，与 `dequantize_block_q5_K` 的 native q5 对拍，
  实测 max rel 5.7e-08（float32 舍入）。

### 5.7 2-bit draft 存储（`w2_gemv_launch`，`w4_gemv.cpp:167-301`）

给 MTP draft 的 LM head 用：0.375 B/weight（u4 的 0.625）。形状、循环顺序与 epilogue 都照抄 u4 GEMV，
三个差别：权重平面每权重 2 bit、**按元素序**存放，所以一个字节正好展开成一个 dp4a word，不需要
even/odd 激活拆分，`xq` 就是 int8 路径已有的连续 per-32 激活；level 是 0..3，本身就是合法**无符号**
dp4a 操作数，所以既没有 `0x80` 偏置也没有 `128*sum(a)` 校正；一组只有 8 字节，所以一 lane 一次
8 字节载入读 16 个权重（u4 是 16 字节读 8 个）。

实测（`w4_gemv.cpp:279-283`，head 形状 K=5120 N=65536）：每 4 个权重约 9 条 ALU 展开，在由此得到的
字节率上远低于发射预算，所以 kernel 仍和 u4 一样带宽受限。**两个组合的一组：**扩大成
`PAIR=1`（一次 `uint4` 取两组）实测**更差**——0.517 ms / 243 GB/s 对 0.467 ms / 270 GB/s，
两者字节数相同，窄形式只是活着的字更少、发出更多更小的载入，所以默认窄形式，`PF_W2_PAIR=1` 是 A/B。
实测结论是 draft 便宜 6 % 而接受率掉 2 %（`AGENTS.md`）。

### 5.8 `nat_gemm_launch`：M = 2..13 的批量原生精度 GEMM（`w4_gemv.cpp:1306-1892`）

MTP 推测 verify 的形状是 `M = k+1` 行、`K`/`N` 与解码同阶。§5.4 的 M=1 GEMV 每列都要重读一整遍激活行
（激活流量 ~`N*K*M` 字节，是权重流的 `N/RB` 倍），所以按行展开的 `w4_gemm_batched` 以 `1/M` 衰减
（M=5 时 43 GB/s）。`nat_gemm_launch` 保持同样的 `lane = 组 / 子组 = 输出列` 映射，但**把整个调用的
K-tile 激活一次 stage 进 SLM**，于是权重流对全部 M 行只读一遍、激活重读留在 SLM 里；scale/offset/
residual 的 epilogue 在同一个 kernel 内完成，`M <= 13` 时取代 `gemm_w4`/`gemm`。
它返回 `false` 表示调用方要回落：`M < 2 || M > 13 || K % 32 != 0`（`w4_gemv.cpp:1858-1860`），
`PF_NAT=0` 则整条路径关闭、回落 oneDNN 分组-scale matmul + epilogue。

**SLM 预算与模板参数**（`nat_gemm_impl<FMT,M,C,KT,SG,ORDER,TX,OPQ,TREE,ABL,VECST,VECA,NOX,DIRECT>`，
`w4_gemv.cpp:1370-1797`）：

* `NSG = TX/SG`，`RB = NSG*C`（一个工作组覆盖 `RB` 个输出列）。发射
  `nd_range<1>(ceil(N/RB)*TX, TX)`，每子组 `C` 列、lane = `g += SG`。
* 每 K-tile（`KT` 个权重组）stage：`act_s[M*KT*APAD]` int8、`asa_s[M*KT]` f16、`xs_s[M*KT]` f32、
  `sc_s[KT*RB]` 与 `of_s[KT*RB]` f16（cb4/int8 不用 off）、以及 256 项 `lut16`（cb4）或 16 项
  `bit_lut`（k5）。`FMT`：0 = u4（Q4_K）、1 = k5（Q5_K）、2 = cb4（IQ4_XS/NL）、3 = 分组 int8；
  u4/k5 读 even/odd 拆分（`axe/axo`），cb4/int8 读连续分组激活（`axg`）。
* **赢的形状**（`nat_gemm_pick`，`w4_gemv.cpp:1805-1820`，调在 27B verify 形状
  M=7 K=5120 N=17408 / A770 上）：`C=2, KT=32, SG=8, ORDER=1, TX=128, TREE=1, VECST=1, VECA=1,
  NOX=0`，即 `NSG=16`、`RB=32`；每个输出 `(m,n)` 都按组升序累加、子组归约也相同，所以**与 M=1 的
  GEMV 逐位一致**。

**循环顺序是承重的。** `ORDER=1`（m-outer / c-inner：先载入一行的激活组，再对该组跑每列的 8 个 dp4a）
是这整个设计里最要紧的一条：它让 IGC 为 u4 的 nibble 解码形成**硬件 dp4a**。反过来
`ORDER=0`（c-outer / m-inner，每列重载一次激活）会让 IGC 标量化整个内层循环——生成的 asm 里
**零个 dp4a**，kernel 跑在 106 GB/s。ORDER=1 下同一个形状的实测：

| 格式 | c-outer / m-inner | m-outer / c-inner | |
|---|---:|---:|---|
| u4 | 0.421 ms | **0.200 ms** | 2.08× |
| k5 | 0.342 | **0.237** | |
| cb4 | 0.380 | **0.258** | |
| int8 | 0.370 | **0.276** | |

其它承重项：

* **SIMD8 子组**（`SG=8`）。SIMD16/SIMD32 的一个 float 占 2/4 个 GRF，而 `M*C` 个累加器必须全部常驻；
  SIMD32 会把整套累加器溢出 18-23 KB/thread 到 scratch 并慢 4-5×（`w4_gemv.cpp:1329-1331`）。
* **16 字节拷贝** stage 激活（`VECA`）与 scale/offset tile（`VECST`），不是逐字节循环。
* **SLM 激活 stride `APAD`**：48 字节让 `M >= 8` 越过寄存器悬崖，因此取 **32**，只有 u4 在
  `M >= 12` 时例外取 48（`w4_gemv.cpp:1379-1386`）。实测（48 → 32）：M=8 u4 0.360 → 0.240、
  M=12 k5 0.685 → 0.497、cb4 0.639 → 0.426、int8 0.583 → 0.398；M=7 上 32 赢 u4 ~1 %、
  k5/cb4 ~4-7 %，而 u4 M=12 反向——那里 48 更好（0.572 → 0.549 ms），只有 u4 在 `M >= 12` 退化。
* 每格式的组循环展开不同（`w4_gemv.cpp:1659-1669`）：u4 用 `#pragma unroll 2` 让第二组的权重载入
  与第一组的 dp4a 链重叠（u4 的解码只有掩码，有富余寄存器）；k5/cb4/int8 保持 unroll 1，
  因为它们活着的解码字更多，展开反而更快（M=7：k5 0.248 → 0.237、cb4 0.276 → 0.259 ms）。
* `TREE=1` 是每列一条 dp4a 累加链（`TREE=0` 则拆成 4 条链再 `((q0+q1)+(q2+q3))` 合并）；选中的是前者。
* 输掉的形状：`C=4/8`、更大的 `TX`（更大的工作组更慢，多出的寄存器占用也没帮助——stage 过的激活本来
  就住在 L2）、以及 `ORDER=0`。

**硬限制：它服务不了 prefill。** `nat_gemm_impl` 把整个调用的 32 组激活 tile stage 进 SLM
（`M*KT*APAD` 字节，`KT=32`，即 `M × 1024` 字节），`APAD=32` 时约 `M <= 56`、`APAD=48` 时约
`M <= 37`，对照 DG2 每工作组 64 KB SLM（profile 预算 48 KB）。而 **prefill 的 GEMM M 就是总 token
数**（`record_forward` 里的 `rows`，`tbm = (mode == 2 && !single) ? rows : tb`），主配置下 prefill 永远
是模式 2，所以**没有任何 prefill 调用落进 `M <= 13` 的窗口**。这是硬限制，不是缺一个实例化：即使给
prefill 加上偶/奇激活拆分，也只对 < 13 token 的短 prompt 有用。prefill 的大 M 需要的是被阻塞的大 M
GEMM，`nat_gemm` 与 dp4a 都不是答案（见 `AGENTS.md` 的 prefill 成本分析）。

**旧版 `w4_gemm_launch`**（`w4_gemv.cpp:797-1304`，`dnnl_gemm.cpp:1502-1511` 门控）只服务 `2 <= M <= 8`
的小批量，且默认**不启用**（`PF_W4_GEMM_MAXM` 默认 1，所以默认仍走 oneDNN matmul；把它设成 2..8 才会把
u4 张量路由过来）。它是同一问题的行展开答案：`w4_gemm_batched<M,U,RB>`（运行时行索引会把累加器打到
local memory，20 GB/s；编译期 M 但经数组索引仍溢出，M=2 时 38.8 GB/s）、`w4_gemm_tn<M,TN>`（每子组
`TN` 列）与 `w4_gemm_c2<M,RB>`（每子组两列但不缩小网格）。`PF_W4_GEMM_U` ∈ {1,2,4}（独立累加器组数，
默认 1）、`PF_W4_GEMM_TN`（0 = 行展开，2/4 = `w4_gemm_tn`，3 = `w4_gemm_c2`；两个列分块变体实测慢
4-10×，“每子组覆盖 TN 列”的映射在这里并不合适）。它已被 §5.8 的 `nat_gemm_launch` 取代，保留只为 A/B。

### 5.9 `xq_launch`：DP4A 路径的激活量化（`xq.cpp:22-80`）

§5.2/§5.3 的权重是 SIn int8，激活必须在设备上量化：**每 32 值一个对称 int8 scale**（scale 是 **fp32**，
激活可以小到 ~1e-5，fp16 会掉进次正规区毁掉重建；权重 scale 保持 fp16），外加**每 16 值一半的
`sum(q)`**，这样整数点积能补上权重的 min/zero-point：

```
sum_k w_k x_k = sx * (sw * sum_k qw_k qx_k - mw * sum_k qx_k)
布局：x8[(g32*TB + t)*32 + i]  xmeta[(g32*TB + t)]  xsumq[(g16*TB + t)]（int2 = 两个半和）
```

* 网格 `nd_range<1>(TB*(K/32)*32, 32)`（每 32 值组一个子组），lane 取 `v/sg_max|v|` 求 scale、
  `round(v/sx)` 再 clamp 到 ±127，半和用 `permute_group_by_xor`（1,2,4,8 累加再 xor 16 一次）。
  **超预算的行写 0**，所以量化量不受 `n_real` 门控（门控会让模式 2 第一个 chunk 之后的所有 chunk
  保持未量化）。
* `up` 非空时按 `silu(x)*up` 量化，与 fp32 kernel 同序。
* `permute_group_by_xor` 是**子组集合操作**，必须所有 lane 都执行（发散的调用返回垃圾）。

---

## 6. `qk_norm_rope_launch`（`qk_norm_rope.cpp:14-224`）

对 Q/K 做 per-head RMSNorm + RoPE，并把 K/V 写入 paged KV 池。

* Q buffer 每行 `qstride = n_head*2*head_dim`；每个 head 的第二段 `head_dim` 是 **attention gate**
  （`attn_launch` 从 `qbuf + head_dim` 读它）。
* 子组角色：`n_sg = n_head + 2*n_head_kv`；`sg < n_head` 处理 Q head `sg`；中间处理 K；其余处理 V。
* 网格 `nd_range<1>(n_rows*n_real*n_sg*32, n_sg*32)`（一个工作组 `n_sg` 个子组），`r = gid/n_real`，
  `t = gid%n_real`；`pos = pos[r]+t`，`row = r*tpb + t`，block table 行 = `slot[r]`；行不过
  `row_nr(info, r)` 或 `!active[r]` 就返回。

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

块内 token 偏移 `kb = table[pos/kBlockSize]`、`ko = pos%kBlockSize`。存储类型由**池指针类型**模板化
（`launch` 是泛型 lambda，`I8`/`I4`/`V8`/`V4` 是编译期常量）：`with_k` 按 `kv_k_dtype()` 选 K 的实例化，
K 的实例化内部再按 `kv_v_dtype()` 分派 V——所以 `--kv-type K:V` 能混合两个带 scale 的类型（i4/i8）。

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
个累加值）。`fuse = (out != nullptr && n_splits == 1)`。四个候选按顺序试：`attn_xmx_launch`（§7.5，
`n_real > 1 && head_dim == 256` 时最先试）→ `attn_flash_kernel`（§7.3，`PF_ATTN_FLASH`）→ 分组 decode
kernel（§7.4）→ 经典 kernel（§7.2）。`grp` 来自参数或 `PF_DEC_GROUP`，默认取 profile 的
`attn.dec_group`（**两张卡都是 0**，理由相反；`kernels.h:263` 那句 “default on” 的注释已过时）。

分组 kernel 仅在 `grp && !kv_dtype_has_scales(kv_k_dtype()) && !kv_dtype_has_scales(kv_v_dtype())
&& !fuse && head_dim == 256 && n_head_kv > 0 && n_head % n_head_kv == 0 && n_head/n_head_kv == 4` 时
选中（`attn.cpp:359-360`；注意是 **K 和 V 都**不能带 scale 平面，i8/i4 因此没有分组路径）。
否则走经典 kernel，按 KV dtype 特化。

### 7.2 经典 kernel（`attn.cpp:379-602`，kernel 体 387-579）

* grid `nd_range<1>(n_wg*32, 32)`，`n_wg = n_rows*n_real*n_head*n_splits`：一个 warp 对应
  `(row r, token t, query head h, split s)`。
* partial 指针 `partials + ((r*tpb + t)*n_head + h)*n_splits*pstride + s*pstride`；inactive 行写
  `m=-inf, l=0`。
* **因果**：`n_kv = pos+1`；键范围切成 `n_splits` 段，`chunk = ceil(n_kv/n_splits)`，
  `t0 = s*chunk`，`t1 = min(t0+chunk, n_kv)`。
* **GQA**：`kvh = (h*n_head_kv)/n_head`（**分块**：连续 `n_head/n_head_kv` 个 query head 共享同一个 KV 头，
  对应 HF 的 `repeat_kv`/`repeat_interleave`）。注意 GDN 的 q/k↔value 配对是**取模**（见 §9）：同一个
  模型里两种约定并存，不要想当然地统一，它们只在 head 数相等时才一致。
* **向量化路径 `avec`**（默认，`attn_vec_env()` 默认值来自 profile 的 `attn.vec`，
  `PF_ATTN_VEC=0` 关闭）：每 lane 两个 4 维块 `d0=lane*4`、`d1=d0+HD/2`，Q/K/V 合并访问。
  非量化 `dot = dot4(qa,ka)+dot4(qb,kb4)`；int8/int4 时两块落在量化 block
  `lane/8` 与 `4+lane/8`，各自乘 scale（i4 用 `i4_ld4` 解包 nibble）。在线 softmax：
  `mnew=max(m,dot); e=exp(dot-mnew); corr=exp(m-mnew); l=l*corr+e`；V 用 `fma4` 累加
  （量化 V 同样按 `lane/8` 与 `4+lane/8` 取 scale）。
* **融合输出**（`fuse`）：gate 指针是 head 槽的第二半，输出 `(a/l)*sigmoid(gb)`。
* 否则写 partial（`part[0]=m`、`part[1]=l`、后续累加值）。
* 非向量的标量路径保留：`acc[HD/32]`，逐元素 `kv_ld` + 每个 32 维块一个 scale。
* 把 lane 扩到 16 维换 16 字节载入的写法**更慢**（4.45 vs 1.55 ms/层，`AGENTS.md`）：16 个累加器
  + 16 个 query + 16 个 K + 16 个 V float 会让 IGC 分配 128 个寄存器并溢出。8-warp/EU 的驻留波是
  架构性的，寄存器技巧买不到更多 warp。

### 7.3 flash 式 tiled prefill attention（`attn_flash_kernel<HPG,KV>`，`attn.cpp:146-273`）

`PF_ATTN_FLASH=1` 打开，**默认关**：它在 SLM 里 stage K/V 并做 FA-2 式子块，本意是消掉经典 kernel
每个 K/V 行按 query head（27B 上 HPG=6）和 query token 的冗余读（16k 下一个 512-token chunk 约
1.66 TB 的 K/V 读、~613 GB/s，确实带宽受限）。实测（2x A770，pp512 @16k）**96 vs 140 t/s**，因为
经典 kernel 受限的其实是每 score 的标量算术（32-lane 点积 + 子组归约 + exp 链），而不是 tiling 省掉的
那点 K/V 流量。要赢需要把点积放进 int8/dp4a 并改成每 lane 一个 key，而不是只做 tiling。
作用域：`!fuse && n_real > 1 && head_dim == 256 && HPG ∈ {4,6} && K 与 V 同型且带 scale`。

### 7.4 分组 decode kernel（`attn_group_kernel<HPG>`，`attn.cpp:37-132`）

`PF_DEC_GROUP` 可选（默认 off）。一个 warp 对应 `(r,t,kvh,s)`，`HPG=4` 个 query head 共享每次 K/V
float4 加载；每个 head 的运算序列与经典向量化 kernel 完全相同，因此 partial 逐位一致。无 gate、无
int8/int4。**两张卡都关，但理由相反**：A770 上它比经典 kernel 慢 5×（64k 深度）；Iris Xe 上是经典
kernel 更快（它有 4× 的 warp，K/V 冗余由 L2 承担）。同一个值、两份相反的证据——这正是按卡拆文件的
理由（见 §1.3）。

### 7.5 `attn_xmx_launch`：oneDNN int8 matmul 的 prefill attention（`attn_xmx.cpp`）

> **名字警告（务必保留）**：这里的 “XMX” 只是“用 oneDNN 的 int8 matmul”这个名字，**与 XMX 单元无关**。
> `ONEDNN_VERBOSE=2` 对引擎跑的每个 matmul 报的 impl 都是 `jit:gemm:any`，而这个 oneDNN 3.11.4 里
> `strings | grep -c xmx` 为 0（全部 498 处 `dpas` 命中都是 `DUMMY_DPAS_*` JIT 生成器占位符），
> 即**根本没有编译进任何 XMX GEMM kernel**。不要从 `PF_ATTN_XMX` 的收益推断 XMX 单元的性能。

经典 prefill kernel 受限在每 score 的标量算术：每个 key 一次 32-lane 点积 + 一次子组归约 + 一次 exp，
而且每个 K/V 行按 query head 重读一遍。这条路径把两个 attention GEMM 写成**普通的 oneDNN int8 matmul**，
KV 先 gather 进连续 scratch（`attn_xmx.cpp:21-46`）：

```
QK^T : [M, 256] s8  x  [256, blk] s8 -> [M, blk] s32
softmax + 因果 mask + 把 P 重量化成 u8（每行一个 scale）
PV   : [M, blk] u8   x  [blk, 256] s8 -> [M, 256] s32
在线 softmax 的 rescale + 累加，最后除以 l
```

四条承重约束（每一条都改对/改错过）：

1. **普通 int8 matmul 沿 k 求和，所以 per-key scale 不能在事后施加**：K 与 V 各自必须只有**一个
   block 级 scale**，源头的 32 组 fp16 scale 在 gather 时折进值里，再由 block max 定这一个 scale。
   这就是 gather 要两遍（先 max 后量化）的原因。
2. **query tile 把共享同一 kv head 的所有 `HPG` 个 query head 堆起来**，matmul 宽度是 `HPG*rows`，
   于是每个 kv head 的 KV 只被读一遍、每个 query token 只贡献一次 score。
3. **matmul 永远以固定宽度 `kBlk = 8192` 运行**（零填充）。一次 oneDNN primitive 创建约 15 ms，
   per-chunk 的 `N` 会让 primitive 永远不可复用——单这一条就是 **12× 回归**。
4. **gather 里的 block-max 归约必须摊到很多工作组，绝不能只有一个**：单个 256 线程工作组归约整个
   8192 键块曾占一层 attention 37.8 ms 中的 **18.1 ms**（QK matmul 6.7、softmax 6.1、PV matmul 4.3）。
   现在是 `kGatherRed = 64`（来自 profile `attn.xmx_gather_red`，`PF_XMX_GRED` 覆盖，上限
   `xmx_gather_red_max = 512`）个工作组各归约一片到私有槽 `part[]`，再由一个小 kernel 折叠；`fmax`
   精确且可结合，所以 block scale 与整个输出**逐位不变**。实测 **18.1 → 9.1 ms**，64k 深度下
   512-token chunk 的边际 **975 → 833 ms（-15 %）**。

一级归约本身还有两个实现细节（`attn_xmx.cpp:289-323`）：每 32 维一组，
`max|k*scale| = |scale| * max|k|` 是**精确**的，所以 256 次乘+取绝对值+取 max 可以塌成 8 次；
四个组各放一条独立累加链（标量形式是每 lane 一条 256 长的 fmax 依赖链，正是这一阶段剩下 9 ms 的
来源），数据载入走 16 字节向量。

**三个已测的负面结论，不要重试：**

* **把 M 分块让 101 MB 的 score 矩阵塞进 16 MB L2 是 +53 %**。L2 卡的是 tile 的**面积**，所以
  oneDNN execute 次数涨约 7×，每次约 35 µs。
* **固定宽度留下的 per-block 填充是免费的**（反正要裁掉）。`xmx_tail_blk` 把尾块收缩到能覆盖余数的
  最小 2 的幂宽度，浪费最大的地方是浅深度（结束在 4096 时需 4608 键 = 1 块 = 44 % 填充，16k 是
  31 %，64k 只有 0.8 %），但实测**中性**（16k 每边际 chunk 563.7/565.3/566.7 vs 563.3/565.9/567.3 ms；
  64k 971.0/972.7/975.8 vs 971.3/972.8/978.3 ms）。保留是因为它不可能亏，`PF_XMX_TAILBLK=0` 恢复旧行为。
* **只从 scale 平面算 block max 是精确的**（每个非零 32 宽组把极值量化成 `|k| = 127`，故
  `max|k*scale| == 127*max(scale[])`），该 pass 的流量降 16× 而实测**中性**（833 → 830 ms）——
  因为 gather 受**页表查找延迟**而非带宽限制。这正是"并行化它"（-15 %）赢过"缩小它"的原因。
  没有采用还有格式上的理由：这个恒等式要求全零组存 scale 0 而不是 1（否则错到 127×），还要给
  `i8_quant` 加一个零 scale 保护（`0/0` 是 NaN）——一个 KV 格式的改动不值得 0.3 %。

**其余实现要点：**

* **per-row 索引数组 `orow`/`oh`/`olim` 由 device kernel 从 `info` 构造**，不是 host `q.memcpy`
  （`attn_xmx.cpp:601-632`）。每个 kv head 三次 host→device 拷贝会把主机钉在 GPU 上（in-order 队列
  上的 host 指针拷贝把主机与设备串行化），还会与复用的 host vector 竞争，约 2×。
* **softmax 是每个 query 行一个 256 线程工作组**（`xmx_softmax`，`attn_xmx.cpp:381-460`）：先一遍
  向量化的 block max，再一遍 `exp` + P 重量化（直接对 block max 归一化，`pmax = exp(blo-mb)`、
  `e2 = exp(score-blo)`、`p8 = round(255*e2)`、`ps = pmax/255`），于是不需要第二遍 exp 或 pmax 扫描；
  因果 mask 用每行的 `olim` 裁剪。被遮住的 key 直接写 `P = 0`。
* **scratch 按 queue 区分**：`xmx_get` 以 **queue 指针**为 key（`attn_xmx.cpp:142-149`）。函数内
  static 会被两个 `--layer-map` 设备共享，device 1 的 matmul 去读写 device 0 的 USM（跨设备页抖动，
  约 12×）。只有 query-tile 容量 `Mcap` 决定重分配——键块恒为 `kBlk` 宽、`max_nkv` 增长**不许**
  重分配（per-chunk 重分配 `[M,kBlk]` scratch 要花掉几秒）。
* **matmul 走专用的 oneDNN stream `st_a`**（`dnnl_gemm.cpp:1621-1673`），它包着与 softmax/累加 kernel
  同一个 in-order 队列，所以后续 SYCL kernel 自动排在 matmul 之后，不用显式 wait；只等 `st_a` 不会
  把主 stream 上排队的稠密 GEMM 一起等掉（queue 级 wait 会把整个 prefill 串行化）。`PF_ATTN_WAIT=1`
  恢复每次 matmul 都等。
* **作用域**：i8 KV、head_dim 256、oneDNN 可用、`n_real > 1`；激活行必须共用同一张 page table
  （模式 1 / 2 都是整个 chunk 一个 slot，decode 的每行 slot 到不了这里），否则返回 `false` 让调用方
  回落经典 kernel。`xmx_min_keys` 以下回落（默认 2048，见 profile）。两种输出形态都支持：融合
  （`out != nullptr && n_splits == 1`，直接写 gate 后的输出）与 partials（写 split 0，其余 split 填
  `(-inf, 0, 0)` 让 `attn_combine` 合并掉）。
* **诊断**：`PF_XMX_BREAKDOWN=1` 打印块循环内各阶段的墙钟（gather / QK / softmax / PV）——**唯一能给
  模式 2 的 512-token prefill attention 归因的办法**，因为 `PF_PROF` 需要 `PF_NOGRAPH` 而那会强制走
  模式 1 的 32-token 路径；每个阶段都带 `q.wait()`，绝对值被串行化放大了，只有份额有意义。
  `PF_XMX_TIME=1` 整个 call 的墙钟，`PF_XMX_DBG` 打印 `M`、`max_nkv` 与每块宽度。

### 7.6 `attn_combine_launch`（`attn.cpp:604-632`）

按 `(r,t,h)` 归约 `n_splits` 个 partial：网格 `nd_range<1>(n_rows*n_real*n_head*head_dim, head_dim)`
（一个 head 一个 256 线程工作组，每线程一个 head 维）。`M = max_s m_s`；
`sum = Σ l_s*exp(m_s-M)`，`acc = Σ partial[s*2+d]*exp(m_s-M)`；乘以 `sigmoid(gate)` 后写出。
融合/非融合路径都乘 Q-gate 的 sigmoid。

---

## 8. `conv_l2_launch` / `conv_state_update_launch`（`conv.cpp`）

GDN 的 depthwise 因果卷积 + q/k 部分的 L2 归一化。

### 8.1 `conv_l2_launch`（`conv.cpp:17-71`）

* 组维度 `group_dim = head_k_dim`（调用方传 `hp.d_state`），`n_groups = conv_dim/group_dim`，实现上
  tap 循环 `#pragma unroll` 固定为 4（即硬编码 GDN 的 4-tap 卷积）。网格
  `nd_range<1>(n_rows*n_real*n_groups*128, 128)`：每组一个 conv 输出通道块、128 线程，
  `ch = grp*head_k_dim + lane`，L2 归约用 128 宽的 SLM 树。
* `conv_state + slot*3*conv_dim` 是 `kernel_size-1 = 3` 行滑窗。
* `cross_row=true`：tap 索引是批次绝对 token `gp = rr*tpb + p`，可读前一行的 qkv，使一次调用覆盖整个
  chunk-batched prefill；`cross_row=false`：负索引读保存的 state。
* `silu_f` 后，前 `n_norm_groups = 2*n_k_heads` 组（q 和 k 段）做 L2 归一化
  （SLM 归约 + `1/max(sqrt(red[0]),eps)`）。

### 8.2 `conv_state_update_launch`（`conv.cpp:76-126`）

网格 `nd_range<1>(n_rows*(conv_dim/256)*256, 256)`。把最后 3 个 qkv 行写回每 slot 状态，并在需要时写
检查点：`end_tok = pos[rr]+n`，`pc_active && end_tok % kBlockSize == 0` 时 `cap = pc_row_slot[end_tok/kBlockSize]`，
写 `snap.base + cap*stride + layer_off + gdn_per + {0,1,2}*conv_dim + i`。两个细节：

* 三个 tap 的来源用**绝对 token 索引** `a = rr*tpb + n-1`、`a-1`、`a-2`；落在本行内就从本次 forward 的
  `qkv_raw` 取（模式 2 的连续行），否则从保存的 state 取。这正是**部分**末行也正确的原因——短于
  `conv_k-1` 的行必须回上一行取，而不是读（尚未更新的）state。
* `last_row_only`（一次调用覆盖所有 chunk 行时）只有最后一行写常驻 state，但被标记了块边界的行始终
  写检查点池；`info->mtp_dry` 时一律不写 state（MTP 的 dry verify，见 §9）。

---

## 9. `gdn_launch`（`gdn.cpp:248-392`）

Gated DeltaNet 递归。两个数学等价的实现：`gdn_kernel<C,WPW>`（标量，`gdn.cpp:19-147`）与
`gdn_f4_kernel<C,WPW>`（float4，`gdn.cpp:152-246`）。

* `C` = 一个 warp 拥有的状态列（行）数；`col_groups = head_dim/C`；`total_warps = n_heads*col_groups`；
  网格 `nd_range<1>(ceil(total_warps*n_rows/WPW)*WPW*32, WPW*32)`（`WPW` 个 warp 一组）。
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

* 状态在 token 循环前载入、循环后写回。`step_info` 的字段在循环外 hoist 一次：它们别名自己的写，
  在 token 循环里重读 host USM 比算术还贵。
* **检查点有两种触发**（两个 kernel 都一样）：`pc_on && (pbase+t+1) % kBlockSize == 0` 时把该 warp 的
  C 行状态写到 `snap.base + st*stride + layer_off + (head*head_dim+col0)*head_dim`；而
  `info->mtp_dt` 置位时**每个 token** 都快照一次（边界索引直接用 `t`），这是 MTP verify 把递归状态回滚
  到“已接受的 draft 长度”的基础。
* `info->mtp_dry` 置位时**跳过最后的状态写回**：verify 算完整条 forward 但不碰常驻的 GDN/conv 状态，
  于是 commit 可以精确地只前滚已接受的那几个 token（见 [04-engine.md](04-engine.md)）。
* `tpb_arg`/`nreal_arg` 允许融合调用覆盖 `info->tpb`/每行 token 数：`nreal_arg` 是完全融合的
  chunk-batched prefill（`PF_GDN_FUSE=2`）下那一行的**整个 batch** 长度，一个 warp 走完整批并带着递归
  状态。没有这个覆盖时 kernel 只处理第 0 行（`kMaxT` 个 token），后面每个 chunk 行都读到陈旧状态
  ——这正是 mode-2 prefill 那个 bug（CPU 的 `cpu_gdn` 一直是对的）。
* 配置（默认值全部来自 profile，见 §1.3）：
  * `PF_GDN_COLS` ∈ {1,2,4,8}（默认 `shape.gdn_cols` = 2）、`PF_GDN_WG` ∈ {1,2,4,8}
    （默认 `shape.gdn_warps_per_wg` = 8）。
  * 只有 `n_real >= PF_GDN_COLS_MIN`（默认 8 = prefill 的行宽）且 `head_dim % cols == 0` 才做列批处理，
    否则 `c = 1`（decode 每行只有 1 个 token，列批处理省下的共享 q/k 载入换不回减半的 warp 数）。
    MTP verify 的 `n_real = k+1 = 5` 落在门槛之下，实测每个 `PF_GDN_COLS>1` 变体在那里慢 1-1.7 ms，
    所以门槛是对的。
  * `PF_GDN_VEC`：**未设不是“默认开”**，而是“按规则”——float4 变体只在
    `n_real < shape.gdn_vec_max_rows`（两卡都是 2）时用，因为 float4 的状态切片占 4× 寄存器。
    27B / 2x A770 上每个 pass 的 ms：`n_real` 1/2/4/8/12/16/32 时 float4 是
    4.64/7.66/7.68/10.24/12.38/14.63/22.03，标量是 4.89/5.23/5.23/5.92/6.91/8.26/12.17——即 float4
    在 `n_real=1`（普通 decode）赢 0.25 ms，从 2 起就输 1.46×，到 32 输 1.81×；边际成本 0.51 ms/token
    对标量的 0.25。MTP verify 在 `n_real=5`，那里标量快 2.5 ms。`PF_GDN_VEC=0/1` 强制标量/float4。
  * `PF_GDN_PD` 只剩一行注释（`gdn.cpp:280`，宣布"classic form：`at` 取更新后的状态"的 A/B），代码里
    没有对应的 `getenv`——两种 `at` 形式已不可切换，`gdn_launch` 的实际分派只看 `f4`、`c`（列数）与
    `wpw`（每工作组 warp 数）三者。

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

1. `--kv-type T`（`kv_dtype_set_kv`，在 engine 构造前生效）；`T` 可以是单个名字（K、V 同型）
   或 `K:V`（K、V 独立，如 `i4:i8`）；
2. `PF_KV_F32 != 0` 或 `PF_KV_BF16 == 0` → f32；
3. `PF_KV_TYPE` 未设/空 → i8（默认）；
4. `PF_KV_TYPE = f32|fp32|0 / f16|fp16 / bf16 / i8|int8|q8 / i4|int4|q4`，或 `K:V`；未知 → 警告 + i8。

K 与 V 由 `kv_k_dtype()` / `kv_v_dtype()` 分别给出（`kv_dtype()` 是 K 的兼容别名），
`kv_dtype_mix_ok` 只允许混合两种带 scale 的类型（i4/i8）——它们的行字节数不同
（`head_dim` vs `head_dim/2`）但共用同一个 per-32 fp16 scale 平面。engine 的 pool 因此有
`kv_block_bytes()`/`kv_v_block_bytes()`、`kv_layer_stride`/`kv_v_layer_stride` 两套几何，
虚拟内存映射和 prefix-cache blob 也按 K/V 分别计算；`--kv-cap-mb` 的换算用两者之和。

**精度**：attention 的误差由 V 主导，不是 K。27B 对 fp32 CPU 参考的 mean|diff|：i8 0.035、
`i4:i8`（K=i4/V=i8）0.061、i4 0.183、`i8:i4` 1.49。所以 `--kv-type i4:i8` 用 i8 的 75% 字节
拿到接近 i8 的精度，并能装下 262144 上下文。

i8 几何：`[block][kv head]` 单元内是 `kBlockSize` 行 × `head_dim` int8，后接独立的
`kBlockSize × (head_dim/32)` fp16 scale 平面。成本 3 KB/token/layer（K+V）对比 bf16 的 6、f32 的 12。

i4 几何相同，但每个字节打包两个有符号 4-bit 值（低 nibble = 偶数 head dim，值域 `[-7,7]`，二补码），
一行是 `head_dim/2` 字节，scale 平面与 i8 一致；成本 1.5 KB/token/layer。`kv_dtype_bits` /
`kv_dtype_row_bytes` 给出每元素位数与行字节数，`kv_dtype_has_scales` 判断是否有独立 scale 平面。
混合时两个 kernel 都按 K 的类型实例化、在内部按 V 的类型分派（`attn.cpp`/`qk_norm_rope.cpp`
的 `with_k` lambda）。

`kv_ld_host` 是主机侧元素读取，**返回存储值本身**（i8 是原始 int8，i4 是符号扩展后的 nibble
`(v^8)-8`），不乘 scale；scale 的处理在 `engine::kv_read_vec`，那里才知道行几何。

i4 读取细节：`i4_ld4` 用**一次对齐 16-bit 加载**取 4 个 nibble（`d` 是 4 的倍数，字节偏移
`d/2` 必为偶数），再做 4 次提取——两次标量字节加载会让 16k decode 明显变慢（该 kernel 在长
上下文是**指令**而非字节受限）。性能取舍：i4 端到端 decode/prefill 与 i8 持平，收益是容量
（同预算 2x 上下文）；短上下文 prefill attention 因解包 ALU 慢约 20%，但被 GEMM 稀释。

---

## 12. 视觉编码器 kernel（`vit.cpp`）

所有权重是 BF16（type 30）或 F32（type 0），行主序 `[N][K]`。

| kernel | 说明 |
|---|---|
| `vit_gemm_launch` | `out[t][n] = alpha*Σ W[n][k]*x[t][k] (+residual)`。128 线程 WG，64 行 × 32 token tile，lane `TP=4` token × `RP=4` 行，K 按 64 宽 SLM 分块，`KU=8`；bf16/f32 均用此 tiling |
| `vit_layernorm_launch` | 每子组一行，`sg_sum` 归约；`var = E[x²]-mean²`；可选 bias；支持输入/输出 stride |
| `vit_gelu_launch` | 原地 tanh 近似 GELU，与 `ggml_gelu` 一致 |
| `vit_add_bias_launch` / `vit_add_launch` | 二维逐元素加法，独立 stride |
| `vit_rope_launch` | 2D 视觉 RoPE（作用于 fused qkv 的 Q/K）：pair → section（`sec = pair/(head_dim/4)`，每节一节），`sec==0` 用 patch 行 `py`、否则用 patch 列 `px`，频率指数每节重置；从合并 token 反推 `(px,py)` |
| `vit_attn_launch` | **双向非因果** attention，`HD` 固定 64，`BQ=32` 查询块、`BK=64` 键块，K/V 经 SLM（`HDP = HD+4`、`SDP = BK+4` 避免 bank 冲突），128 线程/组，在线 softmax，`1/sqrt(HD)` |
| `vit_copy_launch` | `out[row][i] = x[row][i]`，独立 stride |

视觉 tower 的 RoPE 与文本模型的交错 M-RoPE 是两套不同的东西，见
[10-multimodal.md](10-multimodal.md)。

### 12.1 音频 tower 的两个 kernel（`at.cpp`）

AuT 风格的音频塔复用上面全部 `vit_*` kernel，只有两处不同，单独放在 `src/backend/gpu/kernels/at.cpp`：

| kernel | 说明 |
|---|---|
| `at_conv1d_launch` | 1D 因果卷积 stem（参考 `audio_model.cpp` 的 `conv1d_all`）：`out[t][o] = b[o] + Σ_tap Σ_i w[o][tap*x_in+i]·x[t*stride+tap-pad][i]`，零填充，按 `range<2>(y_frames, w_out)` 一线程一个输出 |
| `at_rope1d_launch` | fused qkv 的 Q/K 半部做 1D RoPE：每个 token 的每个 pair 用位置 `t` 和标准逆频率，作用在 `[n_tok][3*n_embd]` 的前两个三分段上 |

设计细节与取舍见 [13-audio-video.md](13-audio-video.md)。

---

## 13. 新增 kernel

1. 新建 `src/backend/gpu/kernels/<name>.cpp`，定义启动函数、在同一 TU 内声明 kernel lambda、包含
   `"kernels.h"` 与 `"kernel_utils.h"`。
2. 在 `kernels.h` 声明启动函数。
3. 在 `CMakeLists.txt` 加入 `.cpp`（无 globbing）。
4. **形状常量不进这个文件**：按 §1.3 把在卡上实测出来的宽度/SLM/split 判据加进那张卡的 profile，
   使用点按名字读；只有在**所有卡**都成立时才留字面量。
5. 新增 stage 测试：[12-build-and-testing.md](12-build-and-testing.md)。

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

---

## 15. MTP（NextN）相关 kernel（`mtp.cpp`、`mtp_argmax.cpp`）

MTP draft head 本身是一个普通的 full-attention Qwen3.5 block，复用 `rmsnorm` / GEMV / `qk_norm_rope` /
`attn`；只有输入准备与 verify 的 argmax 是新的。引擎侧（MTP plan / forward / verify / rollback / 接受
判定）见 [04-engine.md](04-engine.md)。

### 15.1 draft 输入：`mtp_concat_launch` / `mtp_capture_launch`（`mtp.cpp`）

`mtp_concat_launch`（`mtp.cpp:33-81`）一次 launch 产出 draft head 需要的 `2*n_embd` 宽行：

```
out[slot] = [ rmsnorm(enorm, emb(tok_slot)) ; rmsnorm(hnorm, h_{slot-1}) ]
```

* 网格固定 `nd_range<1>(kMaxB*kMaxT*256, 256)`（一个 token 一组），同样是**故意固定**——live 行数是设备
  侧状态。`h` 取本 forward 的主模型 hidden，按 `slot = r*tpb + t` 定位：`slot == 0`（批次的第一个
  token）取 `h_prev[r]`，否则取 `h[slot-1]`，即同一 flat 批次里的前一个 token（draft 链只有一个行，
  所以每个 step 的首 token 都落在 `slot == 0` 这条上）。
* 每线程持有每个 256 元素 super-block 的一个元素，所以反量化后的 embedding 与 hidden 行都留在寄存器
  里给第二遍用（不重读、不上 SLM）；两个二次和各自一次 `reduce_over_group`。
* **`tok_dev` 参数是这条链留在设备上的原因**：`tok_dev` 非空时 token 从设备内存读而不是
  `step_info::tokens`——head 的 argmax 直接写进 `d_mtp_tok[step]`，下一步的 `mtp_concat` 从这里读，
  于是一个 cycle 的 k 个 draft 背靠背下发，而不是 k 次设备 sync + k × 1 MB logits 拷贝 + k 次 host 扫描
  248320 个 float：**-1.2 ms/cycle 且输出逐字节相同**。它要求 MTP 层在主设备上（默认的
  `--mtp-device 0`）；跨设备仍走主机往返。prefill 与一条链的第一步传 `nullptr`。

`mtp_capture_launch`（`mtp.cpp:15-26`）把主模型 `output_norm` 之后的 hidden 行整块拷进专用缓冲，
这样 draft head 不会读共享的激活 scratch（后面会被别的 kernel 覆写）。

### 15.2 verify 的 accept argmax：`mtp_argmax_launch`（`mtp_argmax.cpp:170-216`）

对 `[M][n]` 的 logits 矩阵逐行 argmax，只把下标（可选地连值）写回——verify 每 cycle 要它做接受判定，
而把 `M*n_vocab` 个 float（M=5、n_vocab=248320 时约 7 MB）拷回主机再单核串行比较，实测 **33.7
ms/cycle**，占整个 MTP cycle 的 24 %。

* 现在是**每行一个 256 lane 工作组**：strided 合并访问（相邻 lane 读相邻地址），每 lane 在寄存器里
  留一对 `(value, index)`，然后 SLM 树归约。
* **平票打破到最小下标**——per-lane 扫描用严格 `>`，树归约里 `ov > sv[lid] || (ov == sv[lid] &&
  oi < si[lid])`。这与主机 `if (v[i] > best)` 的扫描**逐位一致**，所以输出的 token 不会变。
* 实测 **33.7 → 1.2 ms/cycle（28×）**。`PF_MTP_AMCHK` 做 device/host 对拍（0 失配）。
  它是唯一的入口（GPU backend 的 `mtp_argmax.cpp`，全 `cpu` 时是 `cpu_backend.cpp` 里的主机循环），
  旧的 `PF_MTP_ARGMAX_CPU` 主机扫描已删除。

### 15.3 候选受限的 draft head（默认关，`mtp_argmax.cpp:44-168`）

draft 每步只需要 head 的 argmax，而它的 head GEMV 要读**全部** 248320 行（794 MB 的 u4）。这条路径
改为在**候选集**上求值：候选集来自目标模型刚产生的分布（`mtp_cand_launch` 取 argmax 及其 `margin`
logits 内的 token，`ids[0]` 恒为源行的精确 argmax，所以受限 argmax 对该行逐位相同），
`mtp_gather_launch` 对每个候选行跑与解码 GEMV 相同的 int8 分组-scale 点积（一个候选行一个工作组），
`mtp_gather_argmax_launch` 取 argmax（同样平票到最小 id）。

**数值精确，实测不赚**：候选集只有 25/17/7 % 的概率包含下一步的 argmax（margin 8），即使放宽到
16384 行（词表的 7 %）也只有 81/77/83 %——分布一步之内跑得太远。净效果是一个 prompt 上
43.4 → 42.9 ms/token，另一个 48.4 → 50.4。**默认关**（`PF_MTP_CAND` = 候选上限、
`PF_MTP_CANDM` = margin、`PF_MTP_CANDSRC` = 1 从 draft 自己的第一步取种子、0 从 verify 的 bonus 行取、
`PF_MTP_CANDDBG` 打逐步命中率、`PF_MTP_CANDV` 顺带返回所选 logit）；这机制是给更好的候选来源
（低秩预筛、或者更粗的 draft head）准备的。
