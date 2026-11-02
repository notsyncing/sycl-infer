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
  **quick_gelu**；分离的 `attn_q/k/v` 在加载时反量化并拼成一份连续的
  `[3E][E]` f32 qkv（bias 拼成 `[3E]`），因此前向只有一次融合投影；
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
* `head_dim == 80` 的 attention instantiation，现在是 `vit_attn_impl<80,32,32,1024>`
  （47.6 KiB SLM；见 §5）；
* `vit_quick_gelu_launch`（视觉 block）和 `vit_gelu_errf_launch`（merger）；
* `vit_silu_mul_launch`（f32 文本路径的 `silu(gate)*up`）。

**线性层走文本模型的 oneDNN int8 矩阵乘**（`PF_VISION_DNNL`，qwen2vl 默认
开）。`vit_gemm` 按 32-token 一组切 M、每个 tile 重读一遍权重：5476 个 patch
就是每个线性层 172 遍权重。`vision_model::upload` 建一个私有 `dnnl_gemm`，
懒注册（`add_weight(..., eager=false)`）32 层的 qkv/out/up/down 与 merger 的
`mm.0`/`mm.2`，每次前向按 `kMaxB*kMaxT` 行分块量化 + 相乘。patch embedding 仍是
`vit_gemm`（K=592 不是 32 的倍数，约占 1%），未被接受的张量逐调用回退。
F16/F32/BF16 源都支持（`add_weight` 的 switch 与 `dequantize_row`）。

**验证**：对同一张 1036×1036 页面，和 llama.cpp `llama-mtmd-cli` 的
`MTMD_DEBUG_EMBEDDINGS` 输出逐元素比较：1369 个 merged token × 896 维，
cos **0.999997**，max|diff| 33.4（值域 ~69，约 0.05%），RMS 比值 1.0000。
这说明 patch 顺序、temporal 权重相加、2D RoPE、attention、merger 都是对的。
int8 线性层之后页面输出 md5 不变（`10eed240936226fd4696d943aaabc44b`），
host-vs-device 自检 cos 0.9991（fp32 device 路径本身是 0.9994）。

## 4. MinerU 两步抽取（`src/mineru.*`）

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

CLI：

```bash
./build-aot/sycl-infer \
  --model  /data/models/MinerU2.5-Pro-2605-1.2B-GGUF/MinerU2.5-Pro-2605-1.2B.Q4_K_M.gguf \
  --mmproj /data/models/MinerU2.5-Pro-2605-1.2B-GGUF/MinerU2.5-Pro-2605-1.2B.mmproj-f16.gguf \
  --ctx 8192 \
  mineru --image page.png --max-tokens 256 > page.json
```

可选：`--mineru-only-type T`（调试单个类型）、`--mineru-layout-side N`。
image/chart block **总是**被裁剪分析（`\nImage Analysis:`），不再有
`--mineru-image-analysis` 开关；参考实现的默认 `false` 是有意不复刻的。

在示例页面 `layout_original_image_01.png` 上，layout 与 llama.cpp 的框几乎
逐项一致（个别坐标差 1-2/1000），text block 的内容正确
（"Where do you come from?"、"What do you study?" ...）。整页（layout + 21 个
block）在单张 A770 上现在是 **15.8 s**（2026-11-02；视觉 4.94 s + 文本
prefill 0.73 s + decode 4.75 s + 加载/启动 7.3 s），优化前是 **113.1 s**。

## 5. 性能与后续

优化后的耗时拆分（同一张 1036×1036 页面、5476 patch、单卡 A770、AOT
Release、`PF_MINERU_TIME=1`）：

| 阶段 | 优化前 | 优化后 |
|---|---:|---:|
| layout 视觉编码 | 50.5 s | **3.64 s** |
| layout 文本 prefill（1394 token） | 1.92 s | **0.26 s** |
| layout decode（463 token） | 3.28 s | 3.27 s |
| 21 个 content block（视觉+文本合计） | 52.9 s | **3.30 s** |
| **整页 wall** | **113.06 s** | **15.82 s** |

三个改动，输出 md5 全部不变：

1. **视觉线性层换 oneDNN int8**（见 §3），布局页视觉 58.7 s → 3.64 s。
2. **80 维头的 attention tile 改成 `BQ=32, BK=32, WG=1024`**：47.6 KiB SLM
   允许一个 workgroup 占一个 Xe core，所以 workgroup 的 warp 数就是
   occupancy；旧的 16/64/128 只用了 16 EU 核里的 4 个。补零行在每个阶段都有
   `r < n_q` / `k < n_k` 守卫，所以任意 `n_tok` 都安全（56×56 探针是 16
   token，两个方向都是部分 tile）。布局页 attention：16/64/128 **10 473 ms**、
   16/32/512 4 658、32/32/128 8 106、32/32/256 4 563、32/32/512 3 720、
   64/16/512 3 335、**32/32/1024 3 263**。
3. **多模态 prefill 走 mode 2**（`PF_MM_BATCH_PREFILL`，默认开）：M-RoPE 与
   `img_row` 按 `[row*kMaxT + t]` 索引，正好是 `mm_write_step` 的连续布局，一次
   前向最多 `kMaxB*kMaxT` 个 token（两卡流水线轮换两个 step_info，仍走 mode 1）；
   旧的每 32 token 一次前向。

剩下的成本：attention 占视觉的 90%，layout 的 463-token decode 是另一个大
块；再快需要换 attention 算法（还没做的只有 oneDNN int8 QK），而不是更大的
tile。同页 `llama-mtmd-cli`（SYCL）layout 步 28.2 s wall（我们同样只跑 layout
的进程是 13.2 s，2.1x；各自去掉模型加载后 20.9 s vs 7.2 s，2.9x）、整页 398 s
（我们 15.75 s）。那个 CLI 每张图一个进程，每进程约 7.3 s 模型+mmproj 初始化 +
约 6 s SYCL JIT，且该驱动的 persistent JIT cache 会 SIGSEGV，所以整页差距远大
于单步差距。

已量到的正确性基线（必须保持）：

* 文本 top-1：`"The capital of France is "` → 220，与 llama.cpp 相同；
* 视觉：cos 0.999997（对 `MTMD_DEBUG_EMBEDDINGS`）；
* layout + text：与 llama.cpp 的框和文本一致；
* 优化后整页 md5 `10eed240936226fd4696d943aaabc44b`、21 个 block，与优化前逐字节相同。

## 6. 诊断开关

| 环境变量 | 作用 |
|---|---|
| `PF_F32=1` | 文本线性层走 f32（默认 int8） |
| `PF_VISION_F32=1` | 视觉线性层反量化成 f32（默认 f16；qwen2vl 的 qkv 拼合始终是 f32） |
| `PF_VISION_DNNL=0` | 视觉线性层回退到 `vit_gemm`（qwen2vl 默认走 oneDNN int8） |
| `PF_VIT_ATTN=N` | 80 维头的 attention tile 扫描（0=32/32/512，1=64/16/512，2=32/32/1024 默认，3=16/32/512，4=32/32/256，5=64/16/1024） |
| `PF_MM_BATCH_PREFILL=0` | 多模态 prefill 回退到每 32 token 一个 mode-1 chunk |
| `PF_MINERU_TIME=1` | 每张图一行：prep / vision / prefill / decode / total ms，以及 mmproj 加载+上传时间 |
| `PF_VISION_TIME=1` | 每个视觉 encode 一行阶段耗时（patch / merger / post_ln + 32 层合计），带每阶段 `q.wait()` |
| `PF_MINERU_DEBUG=1` | 打印 layout raw、每个 block 的耗时/大小、host-vs-device 视觉自检 |
| `PF_MINERU_PROBE=N` | 自检裁剪边长（默认 56；56 是 16 patch / 4 merged token） |
| `PF_MINERU_NOMROPE=1` | 把所有 M-RoPE section 置为 t（A/B） |
| `PF_MINERU_ZERO_EMBD=1` | 把图像 embedding 清零（确认 embedding 确实被使用） |
| `PF_MINERU_DUMP_EMBD=<path>` | 按 llama.cpp 的 `[int32 n_tokens][int32 n_embd][f32]` 格式导出视觉输出 |
