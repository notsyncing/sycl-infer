# 设计 13：音频与视频输入

> 配套阅读：[设计 10（视觉）](10-multimodal.md)。音频/视频复用视觉编码器的 transformer 块结构、
> 设备向量 `d_img_embd` 与 prompt 扩展机制，差异只在 **输入预处理**、**位置几何** 与 **嵌入宽度**。

## 1. 概览

在图像之外，`sycl-infer` 支持两类新的多模态输入：

* **视频**：解码为均匀采样的帧序列，每帧走视觉编码器，产生 `T × (nx·ny)` 个嵌入；
* **音频**：解码 + log-mel 谱，走独立的音频编码塔（AuT 风格），一个嵌入对应一个 conv2 之后的帧。

三种输入共享同一条下游链路：`mm_prompt` → `img_row` → `d_img_embd` → `generate_mm`。
一个请求可以**混合** image/video/audio，块按占位符出现顺序排列，由
`mm_build_prompt_mixed_device`（`multimodal.cpp:422-541`）统一装配；位置几何与扩展规则见
[10-multimodal.md §4.3](10-multimodal.md)（三种 kind 的 `(t,h,w)` 公式在那里，本篇只列差异）。

### 1.1 文件

| 文件 | 作用 |
|---|---|
| `src/mm/audio.{h,cpp}` | 音频解码（WAV 原生 / ffmpeg 回退）+ log-mel 预处理 |
| `src/mm/audio_model.{h,cpp}` | 音频编码塔（host 参考 + device），独立 GGUF |
| `src/mm/video.{h,cpp}` | 视频解码（内置 AVI 解复用 + ffmpeg 回退）+ 均匀采样 |
| `src/backend/gpu/kernels/at.cpp` | `at_conv1d` / `at_rope1d` 设备 kernel |
| `src/mm/multimodal.{h,cpp}` | 三种媒体的块规划、M-RoPE、`mm_build_prompt_mixed*` |

## 2. 视频路径

### 2.1 解码（`video.cpp`）

`mm_video_decode_file` / `mm_video_decode_mem`（`video.cpp:543-597`）把原始容器解码为
`mm_video{ frames[] }`，每帧仍是原始分辨率的 RGB8（视觉预处理负责缩放）：

* **内置路径**：RIFF/AVI 解复用器（`parse_avi_header` + `collect_avi_frames`，`video.cpp:112-234`），
  只收集帧偏移，随后用 stb 解码——MJPG/MJPD 走 `stbi_load_from_memory`，原始帧支持 24/16/32 bpp
  （`decode_one`，`video.cpp:236-293`，16 bpp 是 RGB565，32 bpp 是 BGRA，且按 `strf` 的高度符号处理
  bottom-up）。够测试套件与简单的 MJPEG 片段用，不引入新的编译期依赖。
* **回退路径**：其余容器（MP4/H.264 等）交给 `ffmpeg` CLI（`PF_AV_FFMPEG`，与音频共用）。先
  `probe_video`（`video.cpp:393-446`）拿几何与时长：优先 `ffprobe -of csv=p=0`
  （`PF_AV_FFPROBE`，`video.cpp:344-347`），无 ffprobe 时退回解析 `ffmpeg -i` 的 stderr。然后按
  `fps = max_frames/duration`（上限 120，时长未知时取 10）加
  `scale=trunc(iw/2)*2:trunc(ih/2)*2` 管道输出 rgb24（`video.cpp:459-474`）。`fps` 过滤是近似的，所以
  收尾再用等间隔抽样压到恰好 `min(max_frames, 已收帧数)`（`video.cpp:514-519`）。
  in-memory 的输入先落到 `/tmp/opencode/video_input_<ptr>.bin`（`video.cpp:556`）再走同一条路径。
* **尺寸约束**：`max_side` 是**解码前**的硬拒绝——内置路径在识别出 AVI 头后立刻拒绝并且**不再回退**
  ffmpeg（`video.cpp:300-305` 的注释 “recognized but rejected”），ffmpeg 路径在 probe 之后拒绝
  （`video.cpp:451-458`）。帧数约束在采样之后：`mm_video_fmt::max_frames`（CLI `--max-video-frames`
  默认 16，`main.cpp:158`；结构体默认也是 16，`video.h:46`），`max_side` 的 CLI 默认是 768
  （`main.cpp:159`、`--max-video-side`），而 `mm_video_fmt` 结构体默认是 1024（`video.h:47`）——
  两个默认值不一样，实际生效的是 CLI 传下来的那个。
* `kMaxVideoDecodeFrames = 128`（`video.h:31`）是**内置 AVI 路径**扫描 `movi` 列表时的硬上限
  （`video.cpp:307`）；ffmpeg 路径另有一个 `max_frames * 2` 的收帧上限防内存爆掉
  （`video.cpp:487`）。
* `mm_video_subsample`（`video.cpp:525-541`）在时间轴上等间隔取 `n` 帧（`step = ceil(size/n)`，从头
  取），保证时序覆盖；帧数不足时原样返回。

### 2.2 预处理与网格

每一帧独立走 `mm_image_preprocess`（与图像完全一致：Qwen 智能缩放、bicubic、归一化），cfg 由
`vision_cfg(vm)` 派生（`multimodal.cpp:225-237`）：`patch_area = P²·merge²`，
`min_pixels = 8·patch_area`、`max_pixels = kMaxImgTokens·patch_area`，`mean/std` 取自视觉塔超参。

块的 token/位置几何**只由第 0 帧决定**：`plan_videos` 用 `make_input(vm, imgs[0])` 得到
`out_w/out_h`，然后 `n_tok = 选中帧数 × vi0.n_out`、`n_pos = max(out_w, out_h)`
（`multimodal.cpp:259-273`；混合路径同构，`multimodal.cpp:480-490`）。因此
**帧间网格必须一致**——这是前提而非检查：编码循环里每帧各自 `make_input` 并按 `vi.n_out` 递增行偏移
（`multimodal.cpp:289-296`、`525-532`），若某一帧因分辨率不同拿到不同的网格，`n_tok` 与行偏移就会
错位。实践中一个视频的所有帧分辨率相同，智能缩放结果一致，所以成立。总行数超 `kMaxImgTokens` 抛
`video tokens exceed the kMaxImgTokens budget` / `media tokens exceed ...`（`multimodal.cpp:275-277`、
`514-516`）——注意视频的总预算是**所有块共享**的 1024 行，`T` 帧 × 每帧 `nx·ny` 很容易吃掉它。

## 3. 音频路径

### 3.1 解码与 mel（`audio.cpp`）

* WAV 原生解码：`parse_wav` 走 RIFF chunk 链，读 `fmt `/`data`（`audio.cpp:31-71`），
  `WAVE_FORMAT_EXTENSIBLE` 从 SubFormat GUID 取真实 tag（`audio.cpp:52-59`）；`convert_pcm` 支持
  PCM 8/16/24/32 与 IEEE float 32/64，立体声下混成单声道（`audio.cpp:73-134`）。其它格式 tag
  （ADPCM 等）与位深一律拒绝，让调用方回退 ffmpeg（`audio.cpp:147-158`）。
* ffmpeg 回退：`mm_audio_decode_ffmpeg` 走
  `ffmpeg -v error -i <path> -f f32le -ac 1 -ar 16000 ... pipe:1`（`audio.cpp:198-199`），失败时把
  stderr 尾部（上限 600 字节）读进错误信息（`audio.cpp:214-232`；ffmpeg 的 stderr 落在
  `/tmp/opencode/ffmpeg_sycl_infer_err.log`，`audio.cpp:19`），并对非有限/越界样本做清洗
  （`audio.cpp:235-241`）。
* 内存中的非 WAV 数据：`mm_audio_decode_bytes` 先试原生解码，失败则 `mkstemp` 一个
  `/tmp/opencode/sycl_infer_audio_XXXXXX` 临时文件（扩展名留 `.tmp`，让 ffmpeg 嗅探 magic），
  写完交给 ffmpeg 再删（`audio.cpp:245-289`）。服务端 `input_audio` 的 base64 走的正是这条。
* 时长上限 `audio_preproc_cfg::max_seconds`（默认 60，`audio.h:31`）在 `mm_audio_decode_mem`
  （`audio.cpp:190-193`）和 `mm_audio_decode_bytes`（`audio.cpp:285-287`）里生效。
  `mm_audio_decode_ffmpeg` 自身**不**截断——CLI 的 `--audio` 先试 `mm_audio_decode_mem` 再直接调
  `mm_audio_decode_ffmpeg`（`main.cpp:490-495`），这条路径上没有 60 s 上限。
  真正兜底的是 `encode_device` 的断言：`n_frames <= kMaxImgTokens * 2 = 2048`
  （`audio_model.cpp:416-419`），按 16 kHz / hop 160 折算约 **20.5 s**，超出即抛
  `audio: input too long`。
* 重采样：`mm_audio_resample` 线性插值（`src = (i+0.5)·ratio - 0.5`，`audio.cpp:291-312`）。
* `mm_audio_preprocess`（`audio.cpp:314-393`）：`n_frames = max(1, (ns>=n_fft ? (ns-n_fft)/hop+1 : ns/hop))`
  （`audio.cpp:322`）；Hann 窗（`audio.cpp:326-330`）；HTK mel 滤波器组在 `f_min..f_max` 上取
  `n_mel+2` 个中心、折成 `n_bins_lim` 个 bin 的三角窗（`audio.cpp:332-352`，`n_bins_lim` 受
  `f_max·n_fft/sample_rate` 限制）；DFT twiddle 预计算一次给所有帧复用（`audio.cpp:354-363`），
  逐 bin 算 `r²+i²` 后过滤波器组，输出 `log(1 + power/floor)`（`audio.cpp:389`）。
  结果是 `[n_frames][n_mel]`。默认几何见 `audio.h:23-32`：16 kHz、`n_fft=400`（25 ms）、
  `hop=160`（10 ms）、`n_mel=128`、`floor=1e-8`。

### 3.2 音频编码塔（`audio_model.{h,cpp}`）

独立的 `audio.` / `a.` 前缀 GGUF（`--audio-mmproj`）。结构（文件头注释 `audio_model.h:11-28`）：

```
log-mel [n_frames][n_mel]
  -> conv1 (k=3, s=1, p=1) -> GELU            n_mel -> C
  -> conv2 (k=3, s=2, p=1) -> GELU            C -> E     (n_out = ceil(n_frames/2))
  -> learned position embedding [n_pos][E]    (第 t 行取 t % n_pos)
  -> N x (LN + fused QKV + 1D RoPE + 双向 attn + FFN)      (复用 vision_layer)
  -> post_ln -> 可选 a.out 投影 [proj_dim][E]
```

* `audio.embedding_length` = 塔宽 `E`；`audio.projection_dim` = 输出宽度
  `out_width(am) = proj_dim > 0 ? proj_dim : E`（`audio_model.cpp:131-133`）。
  **必须等于文本 `n_embd`**，否则入口抛 `audio tower output width != text n_embd`
  （`multimodal.cpp:348-350`、`387-389`；混合路径在 `any_audio` 时才检查，
  `multimodal.cpp:511-513`）。`proj_dim == 0` 时 device 路径直接把 `d_x` 拷进 `d_out`
  （`audio_model.cpp:519-520`）。
* head_dim 由 `n_embd / n_head` 派生，加载时**强制等于 64**（`audio_model.cpp:158-160`，共享注意力
  kernel 的固定 HD）；conv 形状也校验（`conv1.tap==3`、`conv1.in==n_mel`、`conv2.tap==3`、
  `conv2.in==C`、`conv2.out==E`，`audio_model.cpp:185-187`）。conv1 的通道数 `C` 在 forward 里由
  `c2_w.size() / (E*3)` 反推（`audio_model.cpp:273`、`412`）。
* `n_pos` 取 `a.position_embd.weight` 的行数，并与元数据 `audio.position_embd_length` 取小
  （`audio_model.cpp:206-212`）——注释说明这是为了让过大的元数据值不会把 `%` 索引带出张量。
* **1D RoPE 的位置是 conv2 之后的帧序 `t`**（`audio_model.cpp:317`：`theta = t * base^(-2·ic/HD)`；
  设备侧 `at.cpp:53` 写成 `exp2(-2·pair/head_dim · log2(base))`，两者等价）。注意这与
  learned position embedding 无关——后者是逐行**加上去的向量**（`audio_model.cpp:283-291` /
  `476-482`），不移动任何位置。
* `n_out = (n_frames+1)/2`（`audio_model::make_input`，`audio_model.cpp:259-265`），即 ceil(n_frames/2)。
* `encode_host` 是宿主参考；`encode_device` 与视觉一致——权重单 blob 上传、`dev_ptr` 翻译、
  scratch 增长至最大 `n_frames`、**in-order queue**。conv 权重/位置嵌入被打成一个 `d_cw` blob
  （`c1_w | c1_b | c2_w | c2_b | pos`，`audio_model.cpp:441-462`），并且**每次调用都重拷**
  （`audio_model.cpp:463-471`）——与视觉路径“只拷一次”的做法不同。
* device 前向复用 `vit_gemm`/`vit_layernorm`/`vit_gelu`/`vit_add_bias`/`vit_add`/`vit_attn`/`vit_copy`
  加 `at_conv1d`/`at_rope1d`（见 §5）。块顺序逐字镜像视觉塔：`x += out_b` → out GEMM 带 residual →
  ln2 → up → +bias → GELU → `x += down_b` → down GEMM 带 residual（`audio_model.cpp:493-510`，
  `500-501` 的注释明说是"mirroring the vision encoder's block order"）。

### 3.3 缺权重时的行为

`--audio` / `input_audio` 请求在音频塔未加载时收到明确错误：服务端返回 400
`audio input requires --audio-mmproj`（`server.cpp:1602-1611`），CLI 抛
`--audio requires --audio-mmproj <audio-mmproj.gguf>`（`main.cpp:448-450`）。服务不崩溃、视觉仍可用
（两者是独立的加载与 ready 标志，`server.cpp:1460-1497`）。同理
`image/video input requires --mmproj`（`server.cpp:1597-1601`）。

注意服务端的检查顺序：**任何**媒体 part 都先查视觉塔的 `mm.ready`
（`server.cpp:1597-1601`），所以纯音频请求在只加载了 `--audio-mmproj` 的部署上也会拿到
`image/video input requires --mmproj` —— 服务端实际需要**两个** mmproj，而 CLI 的 `--audio` 只需要
`--audio-mmproj`（`main.cpp:448-451`）。

## 4. 混合 prompt 装配（`multimodal.cpp`）

### 4.1 块几何与位置（M-RoPE）

| 媒体 | kind | 块 token 数 | 位置消耗 | `(t,h,w)` |
|---|---|---|---|---|
| image | `MM_KIND_IMAGE` | `nx·ny` | `max(out_w, out_h)` | `(base, base+j, base+k)`，grid (row j, col k) |
| video | `MM_KIND_VIDEO` | `T·nx·ny` | `max(out_w, out_h)` | `(base+i, base+j, base+k)`，grid (frame i, row j, col k) |
| audio | `MM_KIND_AUDIO` | `n_out` | `n_out` | `(base+i, 0, 0)`，纯时间流 |

视频的 `n_pos = max(out_w, out_h)`（与图像同源，只有时间轴多一维），音频 `n_pos = n_out`。
Image/video/audio token 永不进入 `tok_embd`——embed kernel 按 `step_info::img_row` 拷
`d_img_embd` 行（与图像完全一致，见 [10-multimodal.md §5](10-multimodal.md)）。
M-RoPE 的 section 划分来自 GGUF 的 `rope.dimension_sections`（Qwen3.5 参考值 `[11,11,10,0]`），
第 4 段 0 对，所以任意列数都不会越界；段 3 是 temporal 的副本（`multimodal.cpp:128`）。

### 4.2 入口

`mm_build_prompt_mixed_device(vm, am, q, tk, rendered, images, vids, auds, order, n_embd, d_out,
max_video_frames)`（`multimodal.cpp:422-541`，声明在 `multimodal.h:117-121`）分三趟：

1. **规划趟**（`multimodal.cpp:445-516`）：按 `order`（每个占位符一条 `mm_media_ref`，三个平行向量
   的下标）逐块填 `mm_block`——image 直接 `make_input`；video 先 `mm_video_subsample` 再逐帧
   `mm_image_preprocess`（结果缓存进 `vimgs[idx]` 供第二趟复用）并用第 0 帧定几何；audio 先
   `audio_prepare`（重采样 → `mm_audio_preprocess` → `make_input`，`multimodal.cpp:324-331`，
   cfg 来自 `cfg_of(am)`，`multimodal.cpp:333-342`）并把 mel 存进 `ains[idx]`。同时累加 `off[k]`。
   越界下标、非音频的宽度检查（`any_audio && audio_out_width(am) != n_embd`）与
   `rows > kMaxImgTokens` 都在这一趟抛错。
2. **编码趟**（`multimodal.cpp:518-536`）：按 `order` 把每个媒体编码进
   `d_out + off[k]*n_embd`——image/video 走 `vision_model::encode_device`（视频逐帧、每帧推进行
   偏移），audio 走 `audio_model::encode_device`。
3. **扩展趟**：`mm_expand_prompt(tk, rendered, blocks, rows)` 并置 `p.d_embd = d_out`
   （`multimodal.cpp:538-539`）。

因为三趟共用 `mm_expand_prompt`，混合装配与单媒体入口返回的 token/position 布局是同一套代码算出来的。
`audio_model` 只有在 `order` 里出现 `MM_KIND_AUDIO` 时才被用到。

单媒体的 host 版本 `mm_build_prompt_video` / `mm_build_prompt_audio`（`multimodal.cpp:283-299`、
`346-382`）用于单媒体测试与主机参考路径；device 版本 `..._video_device` /
`..._audio_device`（`multimodal.cpp:301-316`、`384-420`）与混合版等价，只是媒体种类单一。

## 5. 新增 GPU kernel（`at.cpp`）

只有两块与视觉塔不同，其余全部复用 `vit_*`（文件头注释 `at.cpp:1-7`）：

* `at_conv1d_launch`（`at.cpp:17-40`）：`y[t][o] = b[o] + Σ_tap Σ_i w[o][tap*xin+i] · x[t*stride+tap-pad][i]`，
  越界 tap 当零。`range<2>(y_frames, w_out)` 一线程一个输出元素，复杂度 `O(n·k·cin)`。用法与几何同
  `conv` 但只有一维；conv 茎用它跑两次（stride 1 与 stride 2）。
* `at_rope1d_launch`（`at.cpp:42-65`）：1D RoPE，`range<1>(n_tok*n_head*head_dim/2)`，位置就是 token
  序号（帧序），无分段。Q 与 K 在 fused qkv 行的前 2/3。

上传整塔权重后复用 `vit_gemm`/`vit_layernorm`/`vit_gelu`/`vit_add_bias`/`vit_add`/`vit_attn`/
`vit_copy`。

## 6. 调用点

| 入口 | 代码 |
|---|---|
| CLI `gen --image/--video/--audio` | `main.cpp:412-520`（解码 → `order` → `render_chat` → `mm_build_prompt_mixed_device` → `generate_mm`） |
| 服务端 `/v1/chat/completions` | `server.cpp:1597-1699`（`parse_messages` 收集 `image_url`/`video_url`/`input_audio`/`audio_url` → `load_media_bytes` → 按 kind 解码 → `mm_build_prompt_mixed_device`），细节见 [09-server.md §6](09-server.md) |

两个入口都在**媒体 part 全部就绪之后**才按顺序拼 `chat_part`，所以 `order` 与渲染结果里的占位符顺序
天然一致。CLI 的 `--image`/`--video` 共用 `--mmproj`（`main.cpp:431`），`--audio` 单独要
`--audio-mmproj`（`main.cpp:449`）。

服务端 `chat_part` 是带 `kind` 的判别联合（`chat.h:12-21`），两个渲染器各认一次：

* 内置 ChatML 回退（`chat.cpp:100-107`）：IMAGE → `<|vision_start|><|image_pad|><|vision_end|>`，
  VIDEO → `<|vision_start|><|video_pad|><|vision_end|>`，
  AUDIO → `<|audio_start|><|audio_pad|><|audio_end|>`。
* Jinja 模板路径（`chat_template.cpp:38-45`）：把每个 part 变成 `{"type": "image" | "video" |
  "input_audio"}` 交给 GGUF 的 `tokenizer.chat_template` 自己出占位符。

## 7. 测试（`tests/mm/test_multimodal.cpp`）

| 测试 | 校验 |
|---|---|
| `test_prompt_video` | 4 帧合成 96×96 视频；网格从预处理后的第 0 帧推导、`pos_after == tokens.size() - T·G + max(out_w,out_h)`、`img_row` 映射后清除、逐 token 的 M-RoPE `(base+frame, base+row, base+col)`（`test_multimodal.cpp:537-555`） |
| `test_audio_geom` | `n_out = ceil(n_frames/2)`（`test_multimodal.cpp:559-566`） |
| `test_audio_prep` | 合成 1 kHz 正弦的 16-bit WAV → `mm_audio_decode_wav` round-trip → 样本数/采样率、mel 帧数与有限非零、截断 buffer 必须被拒（`test_multimodal.cpp:592-645`） |
| `test_audio_kernels` | `at_conv1d`（stride 2 无 bias / stride 1 带 bias）与 `at_rope1d` 对照宿主参考，阈值 1e-5（`test_multimodal.cpp:647-753`） |
| `test_audio_encoder` | 合成 mel + `argv[3]` 的音频 mmproj：host 输出形状 `n_out * audio_out_width`、host 嵌入有限非零、host vs device 最大差 `<= 5e-3 * max(1, max\|host\|)`；未给路径则 skip（`test_multimodal.cpp:758-812`） |

没有真实的音频 mmproj 权重时，音频端到端（真实语音）不能本地验证，测试以合成权重/skip 覆盖。
视频解码本身（AVI 解复用 / ffmpeg）没有单元测试——`test_prompt_video` 直接构造 `mm_video_frame`
列表，只测 prompt 布局这一层。

## 8. 环境变量

总表见 [`../../AGENTS.md`](../../AGENTS.md#environment-variables)；这里只列本篇涉及的：

| 变量 | 作用 |
|---|---|
| `PF_AV_FFMPEG` | ffmpeg 可执行路径（音频/视频都读，`video.cpp:339-342`、`audio.cpp:21-24`），默认 `ffmpeg` |
| `PF_AV_FFPROBE` | ffprobe 路径（视频探测优先走它，`video.cpp:344-347`），默认 `ffprobe` |
| `PF_MM_URL_FETCH=0` | 禁 remote `http(s)` 媒体 URL（base64 `data:` 仍可用，`server.cpp:478-485`） |