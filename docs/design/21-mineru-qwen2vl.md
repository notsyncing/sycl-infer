# MinerU2.5-Pro-2605-1.2B（`qwen2vl`）与 MinerU 两步抽取

本文记录 **MinerU2.5-Pro-2605-1.2B** 在 sycl-infer 中的接入方式。它是一个
**Qwen2-VL** 解码器（`general.architecture = qwen2vl`），因此这里新增的是一条
独立的 dense 文本架构路径。视觉编码器是**同一个** `src/mm/vision.{h,cpp}`：
它通过模型 profile 的 `vision` 字段同时服务 Qwen3.5（`qwen3vl_merger`）和
Qwen2-VL（`qwen2vl_merger`），不再有第二个视觉塔实现。

## 1. 模型与文件

| 文件 | 说明 |
|---|---|
| `MinerU2.5-Pro-2605-1.2B.Q4_K_M.gguf` | 文本主干，~380 MiB |
| `MinerU2.5-Pro-2605-1.2B.mmproj-f16.gguf` | 视觉塔（32 层，1280 宽，16×80 头，patch 14，merge 2，投影 896） |

原始权重在 `opendatalab/MinerU2.5-Pro-2605-1.2B`，GGUF 量化在
`mradermacher/MinerU2.5-Pro-2605-1.2B-GGUF`。目标机器上 `hf-mirror.com` 与
`huggingface.co` 均不可达，实际使用的是同协议的镜像 `aifasthub.com`：

```bash
B=https://aifasthub.com/mradermacher/MinerU2.5-Pro-2605-1.2B-GGUF/resolve/main
curl -L -o MinerU2.5-Pro-2605-1.2B.Q4_K_M.gguf      $B/MinerU2.5-Pro-2605-1.2B.Q4_K_M.gguf
curl -L -o MinerU2.5-Pro-2605-1.2B.mmproj-f16.gguf  $B/MinerU2.5-Pro-2605-1.2B.mmproj-f16.gguf
```

文件放在 `bench-host` 的 `/data/models/MinerU2.5-Pro-2605-1.2B-GGUF/`。

## 2. 文本主干（`qwen2vl`）

几何（来自 GGUF 元数据）：24 层、`n_embd` 896、`n_ff` 4864、14 个 query 头 /
2 个 KV 头（`head_dim` 64）、RoPE sections `[8,12,12,0]`、base 1e6、每层都是
full attention、tied `output`。

**Q4_K_M 在这里不是 K-quant**：`n_embd = 896` 不是 256 的倍数，ggml 对这些张量
回退到 Q5_0/Q8_0。文本张量里 132 个 Q5_0、13 个 Q8_0、12 个 Q4_K、12 个 Q6_K。
因此：

* `src/common/quant.h` 增加了 `block_q5_0` 的 host 反量化；
* `src/backend/gpu/kernels/kernel_utils.h` 增加了设备端的 Q5_0 反量化
  （`dequant_sb_lane_typed<6>`、`dequant_elem_sb`、`dequant_lane_w`、
  `superblock_bytes`）；
* `src/backend/dnnl_gemm.cpp` 的 `add_weight` 接受 type 6。

模型的 `head_dim` 不是 256，这是第一次让 **classic attention** 真正走到
`head_dim != 256` 的路径：

* `attn.cpp` 的 classic kernel 把 `HD/32` 的循环改成 `head_dim/32`；两
  `float4` 的快速路径只在 `head_dim == 256` 时启用，其余走
  `lane + 32*i` 的标量路径。partials 的 stride 本来就是 `2 + head_dim`。
* `qk_norm_rope` / `attn` / `attn_combine` 支持 **没有 QK RMSNorm、没有
  attention 输出门** 的模型：q/k 的 `q_norm`/`k_norm` 为空时跳过归一化；q/k/v
  的投影 bias 在 rope 之前加；`hp.attn_gate = false` 时 attention 不做
  `sigmoid(gate)`。

### 2.1 两个致命细节（已修，勿回退）

1. **q 投影的 head stride 必须跟 gate 是否存在一致。**
   Qwen3.5 把 q 和输出门融合在 `attn_q` 里，布局是 `[head][q(64)|gate(64)]`；
   Qwen2-VL 没有门，投影只写 896 个值（`[head][q(64)]`）。attention 和
   `qk_norm_rope` 原来固定按 `2*head_dim` 取 head，导致 Qwen2-VL 的每个 head
   读到下一个 head 的 q（或空洞）。症状极具误导性：**单 token 的 teacher-forced
   输出正常**（单 key 的 attention 输出就是 v，q 不参与 softmax），但**第二个
   token 起全错**；一个 5 token 的纯文本 prompt 的 top-1 就和 llama.cpp 不同。
   现在 `qgate = gate ? 2 : 1`，`qstride = n_head*qgate*head_dim`，由
   `hp.attn_gate` 传入 `qk_norm_rope`。
2. **mmproj 的 `clip.vision.feed_forward_length` 是错的值（896）。** 真实的
   `ffn_up` 是 `[1280,5120]`，FFN 宽度 5120。视觉塔的 `n_ff` 现在从
   `layers[0].up.N` 推导，不再信元数据；否则 FFN 被截断到 896 列，视觉输出
   变成 0.87 的余弦（几乎全错）。

### 2.2 M-RoPE

Qwen2-VL 用 **blocked** sections（`[8,12,12,0]`，和 = `n_rot/2`），
Qwen3.5 用 interleaved（sections 和 32，`n_rot/2` = 128）。
`qk_norm_rope` 以 `sum(sections) == n_rot/2` 判别，采用 ggml 的 non-imrope
分支：pairs `[0,8)` 用 t、`[8,20)` 用 row、`[20,32)` 用 col。视觉塔自己的
4-section VISION RoPE 在 `vit_rope_launch` 里（每 20 对 row/col，指数按段重启）。

### 2.3 路径选择

* 该模型默认走 **per-tensor / multi-device 权重路径**（即使只有一张卡）：
  `policy.auto_layer_map = true` + `auto_layer_map_one_gpu = true`。单设备
  路径要求 SIn 打包，而 Q5_0 不在 SIn 打包列表里，且 raw GEMV 以 256 为一
  个 superblock，`K=896` 会被截断。per-tensor 路径的 oneDNN int8 转换通过
  `dequantize_row` 读 Q5_0，`i8_row_gemv_multi` 按 32 值组处理任意 K。
* `policy.prefix_cache = false`：MinerU 的每个 crop 都不重复，且没有 GDN
  状态需要恢复。
* `rep.sin_w8 = false`、`rep.dnnl_int8 = true`。
* `f32_gemm` 默认 **关**。`PF_F32=1` 会把文本线性层反量化成 f32 并用
  `vit_gemm_launch` 跑（精度 A/B）。**int8 在 q-stride 修复后与 f32 的布局
  结果一致**；这个开关只是留给对精度极端敏感的场景。

## 3. 视觉塔（`src/mm/vision.{h,cpp}`，一个模块两种变体）

`vision_model::kind` 来自文本模型 profile 的 `model_profile::vision`
（qwen2vl → `qwen2vl`，qwen35/qwen35moe → `qwen3vl`，无多模态的 qwen4exp →
`none`），并在加载时与 mmproj 的 `clip.projector_type` 交叉校验，不一致直接
报错。两种变体共享同一套 host/device 前向，差异只在这些参数上：

* **qwen2vl**：patch 14、temporal 2（两张静态图各用一半 conv，权重相加）、
  无 class token、无 learned position；32 层，1280 宽，16 个 80 维头，FFN
  5120，LayerNorm **带 bias**，q/k/v/out/up/down **全有 bias**，激活是
  **quick_gelu**；分离的 `attn_q/k/v` 在 oneDNN 注册时用 `add_weight_blocks`
  直接拼成一份融合的 `[3E][E]` int8 投影（bias 拼成 `[3E]`），因此前向只有
  一次融合投影；host 参考/`vit_gemm` 回退需要的 f32 视图按需构建
  （`PF_VISION_DEFER_QKV=0` 恢复加载时构建）；
* **qwen3vl**：patch 16、融合 `attn_qkv`、learned `v.position_embd`、
  gelu_tanh，head_dim 64/72；
* 两者共用的部分：2D RoPE（80 维头分 4 段 × 20 对，前 20 对用行号、后 20 对
  用列号，指数每段重启；`vit_rope_launch`）、全双向 attention、2×2 patch
  merge。`v.post_ln` 在 merge 之前逐 patch 应用（qwen3vl 原本就是这个顺序）；
  merger 激活 qwen3vl 用 gelu_tanh，qwen2vl 用精确 erf GELU。

设备端复用 `src/backend/gpu/kernels/vit.cpp` 的 GEMM / LayerNorm / RoPE /
attention 内核；为此增加了：

* `GGML_TYPE_F16` 的权重加载（`load_w`/`load_w8`，用
  `ggml_half_to_float`，不要用 `sycl::bit_cast<sycl::half>`）；
* `head_dim == 80` 的 attention instantiation，默认是保序 ILP
  `vit_attn_ilp<80,48,32,1024,16,2,2>`（`PF_VIT_ATTN=70`；47.6 KiB SLM；见 §5），
  旧的 `vit_attn_impl<80,32,32,1024>` 是 `PF_VIT_ATTN=2`；
* `vit_quick_gelu_launch`（视觉 block）和 `vit_gelu_errf_launch`（merger）；
* `vit_silu_mul_launch`（f32 文本路径的 `silu(gate)*up`）。

**线性层走文本模型的 oneDNN int8 矩阵乘**（`PF_VISION_DNNL`，qwen2vl 默认
开）。`vit_gemm` 按 32-token 一组切 M、每个 tile 重读一遍权重：5476 个 patch
就是每个线性层 172 遍权重。`vision_model::upload` 建一个私有 `dnnl_gemm`，
懒注册（`add_weight(..., eager=false)`）32 层的 qkv/out/up/down 与 merger 的
`mm.0`/`mm.2`，每次前向按 `kMaxB*kMaxT` 行分块量化 + 相乘。qwen2vl 的
q/k/v 是三个独立张量，用 `add_weight_blocks` 直接按 GGUF 转换（不必先建
3E×E 的 f32 副本）；合并的 f32 视图只在 host 参考前向和 `vit_gemm` 回退时
按需构建（`PF_VISION_DEFER_QKV=0` 恢复加载时构建）。patch embedding 仍是
`vit_gemm`（K=592 不是 32 的倍数，约占 1%），未被接受的张量逐调用回退。
F16/F32/BF16 源都支持（`add_weight` 的 switch 与 `dequantize_row`）。

**上传是紧凑的**：所有线性层都注册到 oneDNN 之后，设备内核真正读的只有
norms/biases（1.32 GB 文件里的 3.7 MB），所以 `PF_VISION_COMPACT=1`（默认）
只上传这些 span，`dev_ptr` 走 `blob_spans` 的二分查找；`PF_VISION_COMPACT=0`
恢复整文件 1:1 映射（回退/A/B）。

**验证**：对同一张 1036×1036 页面，和 llama.cpp `llama-mtmd-cli` 的
`MTMD_DEBUG_EMBEDDINGS` 输出逐元素比较：1369 个 merged token × 896 维，
cos **0.999997**，max|diff| 33.4（值域 ~69，约 0.05%），RMS 比值 1.0000。
这说明 patch 顺序、temporal 权重相加、2D RoPE、attention、merger 都是对的。
int8 线性层之后页面输出 md5 不变（`c79d2d222d6d90763f0d25c5ffb5269a`，2026-11 采样器
presence/frequency 惩罚符号修正后的新基线；旧值 `10eed240...` 已作废），
host-vs-device 自检 cos 0.9991（fp32 device 路径本身是 0.9994）。

## 4. MinerU 两步抽取（客户端工具）

**当前结构**：`tools/mineru/`（独立二进制 `sycl-infer-mineru`），见
[22-pipelines.md](22-pipelines.md) §9。`mineru.{h,cpp}` 是两步编排与图像几何，
`layout.{h,cpp}` 是纯 prompt/版面语法，`main.cpp` 是独立 CLI。`sycl-infer` 的
`mineru` 命令和 `--mineru-*` 参数已经移除。它**不是**服务端 pipeline：客户端流程
只需模型能力，按 22 的准入规则留在 `tools/`。

**来源与定位**：`tools/mineru/` 是 `opendatalab/mineru-vl-utils` 的
`MinerUClient.two_step_extract` 的 C++ 复刻（prompt、block 语法、JSON 形状均来自
上游，2026-11-02）。它不是模型能力，也不是引擎必须内置的文档流水线；默认应作为
**reference client** 由调用方驱动，服务端只提供 qwen2vl 的图片
`/v1/chat/completions`。是否把它注册成服务端 pipeline 需过
[22-pipelines.md](22-pipelines.md) §1.3 的准入规则（进程内状态复用 / 实测收益 /
资源账本），并且要与上游版本做 parity 测试。

MinerU 不是一个模型，而是构建在 `qwen2vl` 模型 + 视觉塔之上的**请求级编排**。
因此：

* `GET /v1/models` 只列 qwen2vl 模型，不列 `mineru`；
* `POST /v1/chat/completions` 仍按标准 OpenAI 语义工作（调用方自己给 prompt，
  拿模型原始输出），不暴露任何 MinerU 专用字段；
* 两步抽取由客户端调用 qwen2vl 的图片 `/v1/chat/completions`（或 `tools/mineru`
  直接驱动 engine）；服务端不内置 MinerU pipeline，也不加模型别名。

完全按 `opendatalab/mineru-vl-utils` 的 `MinerUClient.two_step_extract`：

1. 整页缩放到 1036×1036（PIL bicubic），prompt `"\nLayout Detection:"`；
2. 解析 `<|box_start|>x1 y1 x2 y2<|box_end|><|ref_start|>type<|ref_end|>...`，
   bbox 除以 1000，去掉 `inline_formula`，同类型+同角度 IoU>0.9 去重；
3. 每个 block 从**原始页面**裁剪、按 `<|rotate_*|>` 旋转、`resize_by_need`
   （比例 >50 补白、短边 <28 放大），再按模型 preprocessor 的
   `min_pixels=50176 / max_pixels=1605632` 处理；
4. 按类型用 `"\nText Recognition:"` / `"\nTable Recognition:"` /
   `"\nFormula Recognition:"` / `"\nImage Analysis:"`；
5. 采样：temperature 0、top_k 1；非 layout 任务 `presence=1.0`，
   `frequency=0.005`（table）/`0.05`（其余）；`no_repeat_ngram_size=100`
   （llama.cpp 后端同样丢弃它，这里也没有实现）。

Prompt 是手写的 ChatML，与 GGUF 的 chat template 完全一致：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{instruction}<|im_end|>
<|im_start|>assistant
```

CLI（独立参考工具，不进入 server，也不经过 pipeline registry）：

```bash
./build/sycl-infer-mineru \
  --model  /path/to/MinerU2.5-Pro-2605-1.2B.Q4_K_M.gguf \
  --mmproj /path/to/MinerU2.5-Pro-2605-1.2B.mmproj-f16.gguf \
  --ctx 8192 \
  --image page.png --max-tokens 1024 > page.json
```

可选：`--only-type T`（调试单个类型）、`--layout-side N`、`--device cpu|gpu|auto`、
`--layer-map`、`--kv-type`、`--debug`。layout 解码需要足够 token：在本页上
`--max-tokens 256` 会截断到 11 个 block，1024（或 4096）给出完整的 21 个；工具默认
是 4096。
image/chart block **总是**被裁剪分析（`\nImage Analysis:`），不再有
`--mineru-image-analysis` 开关；参考实现的默认 `false` 是有意不复刻的。

在示例页面 `layout_original_image_01.png` 上，layout 与 llama.cpp 的框几乎
逐项一致（个别坐标差 1-2/1000），text block 的内容正确
（"Where do you come from?"、"What do you study?" ...）。整页（layout + 21 个
block）在单张 A770 上现在是 **10.99/11.23 s 热缓存、11.89/12.28 s 冷缓存**
（AOT Release、2400 MHz、静默窗口；逐阶段拆分与 113.1 s -> 15.8 s -> 11.0 s
的历史见 §5）。

## 5. 性能与后续

### 5.1 历史：113.1 s -> 15.8 s -> 11.0 s

同一张 1036×1036 页面（`layout_original_image_01.png`，5476 patch，21 个
block）、单卡 A770、AOT Release、贪心、每次调用 `--max-tokens 1024`、每个
rep 一个进程，2026-11-02。两轮优化的输出始终逐字节不变（当时的页面 md5
`10eed240936226fd4696d943aaabc44b`，2026-11 采样器惩罚符号修正后重测为
`c79d2d222d6d90763f0d25c5ffb5269a`；纯文本 `gen` md5
`bfc55c543af1d1aecae5da0b1f9b99e5`；视觉自检 cos 0.999104）。

* **113.1 s -> 15.8 s（第一轮）。** 视觉线性层换 oneDNN int8、80 维头
  attention 从 16/64/128 换到 32/32/1024、多模态 prefill 走 mode 2。
* **15.8 s -> 11.0 s（第二轮）。** 盒子固定在 2400 MHz
  （`xpu-smi config -d 0 --standby never`），静默窗口，每 rep 一个进程；
  attention 换成保序的 ILP 形状，decode 融合了量化 + 整组 GEMV，加载路径
  去掉一次性开销。

### 5.2 每一阶段：113.06 s -> 15.82 s

| 阶段 | 优化前 | 第一轮后 |
|---|---:|---:|
| layout 视觉编码（5476 patch -> 1369 token） | 50.5 s | **3.64 s** |
| layout 文本 prefill（1394 token） | 1.92 s | **0.26 s** |
| layout decode（463 token） | 3.28 s | 3.27 s |
| 21 个 content block（视觉+文本合计） | 52.9 s | **3.30 s** |
| **整页 wall** | **113.06 s** | **15.82 s** |

第一轮的视觉分阶段（`PF_VISION_TIME=1`，每阶段一次 `q.wait()`，所以优化前
那一列的等待也算在总数里）：

| 阶段（32 层合计） | 优化前 | 第一轮后 |
|---|---:|---:|
| patch embedding（仍在 `vit_gemm`） | 26 ms | 23 ms |
| qkv | 3 725 ms | **77 ms** |
| attention | 11 237 ms | **3 262 ms** |
| attn out | 4 733 ms | **49 ms** |
| ffn up | 19 696 ms | **90 ms** |
| ffn activation | 73 ms | 16 ms |
| ffn down | 18 475 ms | **91 ms** |
| merger（mm.0 / mm.2） | 682 ms | **2.5 ms** |
| **视觉合计** | **58.7 s** | **3.64 s** |

### 5.3 第二轮：15.8 s -> 11.0 s

| | 15.8 s 那一版 | 最终 |
|---|---:|---:|
| 启动 + 文本模型加载 | 2.68 s | **1.65 s** |
| layout 视觉 | 3.65 s | **2.49 s** |
| - attention（32 层） | 3.26 s | **2.10 s** |
| layout 文本 prefill | 0.26 s | 0.26 s |
| layout decode（463 token） | 3.27 s | **3.16 s** |
| 21 个 content block（视觉+文本） | 3.30 s | **2.77 s**（视觉 1.31，文本 1.93） |
| 页面文本 decode（676 token） | 4.97 s | **4.60 s**（6.81 ms/token） |
| **整页 wall** | **15.8 s** | **10.99 / 11.23 s**（11.08/11.12/11.28） |
| 冷文件缓存 | 13.6 s | **11.89 / 12.28 s** |

### 5.4 什么推动了这些数字

1. **视觉线性层换 oneDNN int8**（见 §3）。`vit_gemm` 按 32-token 一组切 M，
   5476 patch 就是每个线性层约 172 遍权重；oneDNN 每个 ≤1024 行的 chunk 只
   读一遍。`add_weight` 接受 F16/F32/BF16 源并用 `eager=false` 跳过预填充
   primitive 阶梯（页面里每个 crop 的 M 都不同）。布局页视觉 **58.7 s ->
   3.64 s**。
2. **80 维头 attention 的 tile 演进**。旧 16/64/128 只用了 16 EU 核里的 4
   个：47.6 KiB SLM 允许一个 workgroup 占一个 Xe core，所以 workgroup 的
   warp 数就是 occupancy。布局页 attention 实测：16/64/128 **10 473 ms**、
   16/32/512 4 658、32/32/128 8 106、32/32/256 4 563、32/32/512 3 720、
   64/16/512 3 335、32/32/1024 **3 263**；再加查询行 ILP 的
   `BQ=48/BK=32/WG=1024` 是 2 865，默认的 ILP 形状（SG16、每线程 2 个查询
   行的 score dot、2 个 PV 行共享一个 V `float4`）是 **2 076 ms**。所有改动
   都保持原来的逐 score 求和顺序、BK=32 在线 softmax 分组和 PV 的 k 顺序，
   部分 tile 的填充行在每个阶段都有 `r < n_q` / `k < n_k` 守卫（56×56 自检
   是 16 token）。
3. **多模态 prefill 走 mode 2**（`PF_MM_BATCH_PREFILL`，默认开）。M-RoPE 与
   `img_row` 是按 token 的、内核按 `[row*kMaxT + t]` 索引，正好是
   `mm_write_step` 对 n 个 token 写出的连续布局，所以一次前向最多覆盖
   `kMaxB*kMaxT` 个 token（两卡流水线轮换两个 `step_info`，仍走 mode 1）。
   旧的每 32 token 一次 mode-1 前向，布局页 1394 token 就是 44 遍权重；
   layout 文本 prefill **1.92 s -> 0.26 s**。
4. **decode 的量化 + 整组 GEMV 融合**（`i8_grp_quant_gemv_launch`）。M=1
   的 decode 每次 call 原本是“激活量化 + 每个段一次 launch”；这个 kernel
   把整个 call 的量化（阶段 1，与 `act_quant_grp_launch` 逐位相同）和最多 4
   个 int8 段的 GEMV（阶段 3，与 `i8_grp_gemv_impl` 逐位相同）合成一次
   launch，权重 scale 按段自己的 `[ng][N_seg]` 平面 stage 进 SLM。它只服务
   `mode == 0 && tb == 1` 且段都是普通 int8 的 call。
5. **加载路径**：文本 oneDNN 的 primitive 阶梯改为惰性（`PF_DNNL_LAZY`，
   M=1 仍预建，因为它在录制的 decode 图里）；mmproj 只上传设备内核真正读的
   3.7 MB（norms/biases），不再是整个 1.32 GB（`PF_VISION_COMPACT`）；
   qwen2vl 的合并 q|k|v f32 副本按需构建（`PF_VISION_DEFER_QKV`）；int8
   转换的量化器向量化（`PF_DNNL_QCHK=1` 逐张量复算参考行：287/287
   个张量、0 mismatch）；转换输出走两个 pinned 暂存槽，H2D 与下一个张量的
   反量化重叠；tied embedding 的复制留在 CPU 侧。启动+文本加载从第一轮前的
   **5.03 s** 降到 **1.65 s**（15.8 s 那一版是 2.68 s）。

### 5.5 剩余成本、天花板与不采用的路径

现在是 **10.99/11.23 s 热缓存、11.89/12.28 s 冷缓存**。按当前保序数学的
per-phase 地板约 **5.0 s one-shot**（load ~1.0、attention 0.87 流量 /
0.55 计算、decode 0.9-1.5、vision GEMM 0.75、prefill 0.12、content ~0.5），
进程复用时约 3.7 s。**11 s 是 bit-exact 的天花板**；再往下需要重排数学或
批处理：

* **decode 4.60 s vs 0.9-1.5 s 地板**：676 token × 0.52 GB/token @
  76-88 GB/s。N=896 的窄投影只暴露 ~896 warp（512 EU 中的 ~112），这是形状
  的上限；split-K / 重排归约会改 token 流，且 M=1 的 oneDNN 不能进录制的
  decode 图。
* **attention 2.10 s vs 0.87 s 流量地板**：保序天花板。
* **content vision 1.31 s**：21 个不同的小 M 各自建 oneDNN primitive；把
  crop 批起来或做 M 分桶阶梯是下一个 bit-exact 步骤，约 0.5-1 s。
* **load 1.65 s vs ~1.0 s**：剩余是 int8 转换（总共 ~0.85 s）。

已量到但**不采用**的路径：

* `PF_VIT_ATTN=51`（f32 oneDNN QK/PV）：1.59 s，比手写 kernel 数值更好，但
  求和顺序不同，且 **run-to-run 非确定**，只能 opt-in。block 级检查
  （2026-11-03，两页、同机同 prompt）：
  - layout 页 5 次：4 次与默认逐字节相同；1 次只有 1 个 text block 的 bbox
    各差 0.001（crop +3/+1 px），21 个 block 的 type 和 content 全部相同。
  - `table_07.png` 3 次：分别有 **5 / 3 / 2 个（共 14 个）block 的 content 不同**
    （都是 image analysis），包括图注和识别文本：
    `2020年/2021年` -> `29/2020`，`铜箔/HM9000` -> `钢带/HMMHPL`；还有
    bbox 差 0.025。差异集合每次不同。
  因此重排路径**会造成最终识别结果的语义差异**，默认必须保序。
* int8 oneDNN QK/PV：1.38 s，页面 md5 改变且不可复现，关闭。
* 预转换 int8 缓存：key 面没有版本，存在错误输出风险，拒绝。
* split-K decode（`PF_I8_SPLITK`）：layout 与 table 两页各 2/2 bit-exact，
  本卡上中性，默认关（构造上不保证）。
* `i8_grp_gemv_rows_multi`：1 ulp 的差异改变了 emit 流，页面不用。

同页 `llama-mtmd-cli`（SYCL）的对比：layout 步 28.2 s wall。第一轮
（15.8 s 那版）我们的 layout-only 进程是 13.2 s（2.1x；去掉模型加载后
20.9 s vs 7.2 s，2.9x）；最终版是 **9.03 s（3.1x；去掉加载 ~7.4 s，2.8x）**。
整页 398 s（我们最终 11 s）。那个 CLI 每张图一个进程，每进程约 7.3 s
模型+mmproj 初始化 + 约 6 s SYCL JIT，且该驱动的 persistent JIT cache 会
SIGSEGV，所以整页差距远大于单步差距；单请求的保守数字引用 layout 步的
2.8-3.1x。

已量到的正确性基线（必须保持）：

* 文本 top-1：`"The capital of France is "` → 220，与 llama.cpp 相同；
* 视觉：cos 0.999997（对 `MTMD_DEBUG_EMBEDDINGS`）；
* layout + text：与 llama.cpp 的框和文本一致；
* 整页 md5 `c79d2d222d6d90763f0d25c5ffb5269a`、21 个 block（采样器惩罚符号修正后的
  新基线；旧值 `10eed240...` 已作废），`--max-tokens 1024`。

## 6. 诊断开关

| 环境变量 | 作用 |
|---|---|
| `PF_F32=1` | 文本线性层走 f32（默认 int8） |
| `PF_VISION_F32=1` | 视觉线性层反量化成 f32（qwen2vl 默认走 oneDNN int8；qwen3vl 保持文件里的 BF16/F32；合并的 qkv f32 视图只在 host/回退时按需构建） |
| `PF_VISION_DNNL=0` | 视觉线性层回退到 `vit_gemm`（qwen2vl 默认走 oneDNN int8） |
| `PF_VIT_ATTN=N` | 80 维头的 attention 形状（`70` 默认 = ILP 48/32/1024、SG16、RQ2/RQP2；0=32/32/512 回退；1=64/16/512，2=32/32/1024，3=16/32/512，4=32/32/256，5=64/16/1024，6=48/32/1024 非 ILP，7=16/64/1024，60=48/32/1024 SG16，71=SG32，72=RQ4，73=RQP4，76=BQ52，51=f32 oneDNN opt-in） |
| `PF_VISION_COMPACT=0` | 恢复整文件 mmproj 上传（默认只上传设备内核读的 3.7 MB） |
| `PF_VISION_DEFER_QKV=0` | 加载时构建合并的 q/k/v f32 副本（默认按需） |
| `PF_DNNL_LAZY` | 文本 oneDNN primitive 阶梯惰性化（M=1 仍预建） |
| `PF_LOAD_TIME=1` | 打印启动/上传各阶段耗时 |
| `PF_DNNL_QCHK=1` | 逐张量复算参考量化行，报告 mismatch 与总数 |
| `PF_I8_SPLITK` | 窄 int8 decode GEMV 的 split-K（默认 0 = 关；本卡中性） |
| `PF_MM_BATCH_PREFILL=0` | 多模态 prefill 回退到每 32 token 一个 mode-1 chunk |
| `PF_MINERU_TIME=1` | 每张图一行：prep / vision / prefill / decode / total ms，以及 mmproj 加载+上传时间 |
| `PF_VISION_TIME=1` | 每个视觉 encode 一行阶段耗时（patch / merger / post_ln + 32 层合计），带每阶段 `q.wait()` |
| `PF_MINERU_DEBUG=1` | 打印 layout raw、每个 block 的耗时/大小、host-vs-device 视觉自检 |
| `PF_MINERU_PROBE=N` | 自检裁剪边长（默认 56；56 是 16 patch / 4 merged token） |
| `PF_MINERU_NOMROPE=1` | 把所有 M-RoPE section 置为 t（A/B） |
| `PF_MINERU_ZERO_EMBD=1` | 把图像 embedding 清零（确认 embedding 确实被使用） |
| `PF_MINERU_DUMP_EMBD=<path>` | 按 llama.cpp 的 `[int32 n_tokens][int32 n_embd][f32]` 格式导出视觉输出 |
