# 设计 13：音频与视频输入

> 配套阅读：[设计 10（视觉）](10-multimodal.md)。音频/视频复用视觉编码器的 transformer 块结构、
> 设备向量 `d_img_embd` 与 prompt 扩展机制，差异只在 **输入预处理**、**位置几何** 与 **嵌入宽度**。

## 1. 概览

在图像之外，`sycl-infer` 支持两类新的多模态输入：

* **视频**：解码为均匀采样的帧序列，每帧走视觉编码器，产生 `T × (nx·ny)` 个嵌入；
* **音频**：解码 + log-mel 谱，走独立的音频编码塔（AuT 风格），一个嵌入对应一个 mel 帧。

三种输入共享同一条下游链路：`mm_prompt` → `img_row` → `d_img_embd` → `generate_mm`。
一个请求可以**混合** image/video/audio，块按占位符出现顺序排列，由
`mm_build_prompt_mixed_device`（`multimodal.cpp`）统一装配。

### 1.1 文件

| 文件 | 作用 |
|---|---|
| `src/mm/audio.{h,cpp}` | 音频解码（WAV 原生 / ffmpeg 回退）+ log-mel 预处理 |
| `src/mm/audio_model.{h,cpp}` | 音频编码塔（host 参考 + device），独立 GGUF |
| `src/mm/video.{h,cpp}` | 视频解码（内置解码 + ffmpeg 回退）+ 均匀采样 |
| `src/backend/gpu/kernels/at.cpp` | `at_conv1d` / `at_rope1d` 设备 kernel |
| `src/mm/multimodal.{h,cpp}` | 三种媒体的块规划、M-RoPE、`mm_build_prompt_mixed*` |

## 2. 视频路径

### 2.1 解码（`video.cpp`）

`mm_video_decode_*`（`video.cpp`）把原始容器解码为 `mm_video{ frames[] }`：

* 内置解码器处理基本的运动 JPEG / 无交错容器；其余（MP4/H.264 等）交给 `ffmpeg` CLI
  （`PF_AV_FFMPEG`，与音频共用），逐帧读 RGB24。
* `mm_video_fmt{ max_frames, max_side }` 约束：**解码前**超大帧（`max_side`，默认 768）
  直接拒绝；**采样后**帧数不超过 `max_frames`（默认 16）。`kMaxVideoDecodeFrames=128`
  是单次解码的硬上限。
* `mm_video_subsample` 在时间轴上尽量均匀取 `n` 帧（从头等间隔丢弃），保证时序覆盖。

### 2.2 预处理

每一帧独立走 `mm_image_preprocess`（与图像完全一致：Qwen 智能缩放、bicubic、
归一化），cfg 由 `vision_cfg(vm)` 派生（`multimodal.cpp`）：`min_pixels = 8·patch_area`，
`max_pixels = kMaxImgTokens·patch_area`。**帧间网格必须一致**——第一帧决定 `out_w × out_h`，
其余帧按同一网格处理。

## 3. 音频路径

### 3.1 解码与 mel（`audio.cpp`）

* WAV 原生解码（PCM 8/16/24/32、IEEE float、uns 8，mono/stereo 下混）；MP3/OGG 等容器
  落到 ffmpeg CLI（`mm_audio_decode_ffmpeg`）。服务端内存中的非 WAV 数据由
  `mm_audio_decode_bytes` 先落临时文件再走 ffmpeg。
* 硬时长上限 `audio_preproc_cfg::max_seconds`（默认 60）；`sample_rate` 与模型对齐（16 kHz）。
* `mm_audio_preprocess`：25 ms 窗 / 10 ms hop / 128 mel 箱，HTK 滤波器组，
  `log(1 + power/floor)`，输出 `mel[n_frames][n_mel]`。

### 3.2 音频编码塔（`audio_model.{h,cpp}`）

独立的 `audio.`/`a.` 前缀 GGUF（`--audio-mmproj`）。结构（`audio_model.h:4-29`）：

```
log-mel [n_frames][n_mel]
  -> conv1 (k=3, s=1, p=1) -> GELU            n_mel -> C
  -> conv2 (k=3, s=2, p=1) -> GELU            C -> E     (n_out = ceil(n_frames/2))
  -> learned position embedding [n_pos][E]
  -> N x (LN + fused QKV + 1D RoPE + 双向 attn + FFN)      (vision_layer 复用)
  -> post_ln -> 可选 a.out 投影 [proj_dim][E]
```

* `audio.embedding_length` = 塔宽 `E`；`audio.projection_dim` = 输出宽度
  `out_width(am)`（`proj_dim>0 ? proj_dim : E`）。**必须等于文本 `n_embd`**，否则抛错。
* head_dim 固定 64；RoPE 1D 按位置 `pos`（learned 位置后的帧序）旋转。
* `encode_host` 是宿主参考；`encode_device` 与视觉一致——权重单 blob 上传、`dev_ptr` 翻译、
  scratch 增长至最大 `n_frames`、**in-order queue**。复用 `vit_*` kernel 与
  `at_conv1d`/`at_rope1d`（见 §5）。
* `n_frames` 上限 `kMaxImgTokens·2`（device 路径断言，`audio_model.cpp:416`）。

### 3.3 缺权重时的行为

`--audio`/`input_audio` 请求在 `audio_ready=false` 时收到明确错误
（`audio input requires --audio-mmproj`），服务不崩溃、视觉仍可用。

## 4. 混合 prompt 装配（`multimodal.cpp`）

### 4.1 块几何与位置（M-RoPE）

| 媒体 | kind | 块 token 数 | 位置消耗 |
|---|---|---|---|
| image | `MM_KIND_IMAGE` | `nx·ny` | `max(out_w, out_h)`，grid (row j, col k) |
| video | `MM_KIND_VIDEO` | `T·nx·ny` | `max(out_w, out_h)`，grid (frame i, row j, col k) |
| audio | `MM_KIND_AUDIO` | `n_out` | `n_out`，每嵌入一个时间步 (t, 0, 0) |

M-RoPE `dimension_sections` 复用 `[11,11,10,0]`（Qwen3.5 参考架构）：第 3 段恒等，
从而支持任意列数而不越界。视频的 `n_pos = max(out_w,out_h)`（与图像同源），音频
`n_pos = n_out`（纯时间流）。Image token 永不进入 `tok_embd`——embed kernel 按
`step_info::img_row` 拷贝 `d_img_embd` 行（与图像完全一致）。

### 4.2 入口

* `mm_build_prompt_mixed_device(vm, am, q, tk, rendered, images, vids, auds, order, n_embd, d_out, max_frames)`
  （`multimodal.cpp`）：按 `order`（每占位符一条 `mm_media_ref`）规划块，预编码各媒体
  （视频帧预处理、音频 mel），然后逐个 `encode_device` 写入 `d_out` 的块偏移，最后
  `mm_expand_prompt`。总行数超过 `kMaxImgTokens` 抛错。
* host 版本 `mm_build_prompt_video` / `mm_build_prompt_audio` 用于单媒体测试与主机路径。

## 5. 新增 GPU kernel（`at.cpp`）

* `at_conv1d`：1D 卷积 + 可选 bias（用法同 conv 但 1D），支持 stride 1/2；几何简单，
  O(n·k·cin)。
* `at_rope1d`：1D RoPE，position 为帧序。
上传整塔权重后复用 `vit_gemm`/`vit_layernorm`/`vit_gelu`/`vit_attn`/`vit_copy`。

## 6. 调用点

| 入口 | 代码 |
|---|---|
| CLI `gen --image/--video/--audio` | `main.cpp`（`mm_build_prompt_mixed_device`，`read_file` 读盘） |
| 服务端 `/v1/chat/completions` | `server.cpp` `parse_messages`（`image_url`/`video_url`/`input_audio`/`audio_url`）→ `mm_build_prompt_mixed_device` |

服务端 `chat_part` 扩展为带 `kind` 的判别联合（`chat.h`）：渲染时按 kind 替换
`<|vision_start|><|image_pad|><|vision_end|>` / `<|video_pad|>` / `<|audio_start|><|audio_pad|><|audio_end|>`
（Jinja 模板侧输出 `{type:image|video|input_audio}`，`chat_template.cpp`）。

## 7. 测试（`tests/mm/test_multimodal.cpp`）

* `test_prompt_video`：4 帧合成 96×96 视频，校验网格从预处理尺寸推导、
  `pos_after == tokens.size() - T·G + max(out_w,out_h)`、M-RoPE (t, row, col) 布局。
* `test_audio_geom` / `test_audio_prep`：WAV 解码 round-trip + mel 几何。
* `test_audio_kernels`：`at_conv1d` / `at_rope1d`（含 bias）对照宿主参考，`max|diff| ≈ 0`。
* `test_audio_encoder`：host/device 一致性；无音频 mmproj 参数时 skip。

无真实音频 mmproj 权重 → 音频端到端（真实语音）不能本地验证，测试以合成权重/skip 覆盖。

## 8. 环境变量

| 变量 | 作用 |
|---|---|
| `PF_AV_FFMPEG` | ffmpeg 可执行路径（音频/视频都读） |
| `PF_MM_URL_FETCH=0` | 禁 remote `http(s)` 媒体 URL（base64 `data:` 仍可用） |