# 设计 10：多模态（视觉）输入

覆盖 `src/mm/image.{h,cpp}`、`vision.{h,cpp}`、`multimodal.{h,cpp}`、`src/kernels/vit.cpp` 以及引擎侧的
`generate_mm` / `step_info` 多模态字段。

---

## 1. 概览

Qwen3.5 的视觉输入由独立的 **mmproj GGUF**（`clip` 架构）提供。流程：

```mermaid
flowchart LR
    IMG[图片文件 / base64] --> DEC[stb 解码 RGB8]
    DEC --> PRE[智能缩放 + bicubic + 归一化 CHW]
    PRE --> VIT[视觉编码器 host 或 device]
    VIT --> MERGE[合并 token 嵌入 n_out x n_embd]
    MERGE --> PROMPT[mm_build_prompt:<br/>展开 image_pad + M-RoPE]
    PROMPT --> ENG[engine::generate_mm]
```

视觉编码器在 GPU 上运行（`encode_device`），`encode_host` 是与之等价的参考实现。两条路径用同一套
输入构建函数，测试要求它们在约 2e-4（实测）到 5e-3（测试阈值）内一致。

---

## 2. 图像预处理（`image.cpp`）

### 2.1 数据模型

```cpp
struct mm_image { int width; int height; std::vector<float> chw; };   // image.h:19-23
struct image_preproc_cfg {                                            // image.h:25-32
    int patch_size = 16, merge = 2;
    int min_pixels = 0, max_pixels = 0;   // 0 = 关闭
    float mean[3] = {0.5,0.5,0.5}, std[3] = {0.5,0.5,0.5};
};
```

`chw` 是 plane-major（CHW）fp32，三个连续平面 R/G/B，已归一化。运行时配置由 mmproj 超参覆盖。

### 2.2 解码

stb 在本 TU 编译（`STB_IMAGE_IMPLEMENTATION` + `STBI_ONLY_JPEG/PNG/BMP/GIF` + `STBI_NO_STDIO`）。
`mm_image_decode_mem` 调 `stbi_load_from_memory(..., 3)` 强制 RGB8；`mm_image_decode_file` 读整个文件后
委托之（不把路径交给 stb）。

### 2.3 Qwen 智能缩放（`mm_image_target_size`，`image.cpp:123-147`）

对齐单元 `align = patch_size * merge`（Qwen3.5 为 32）。

```
w_bar = max(align, round(w));  h_bar = max(align, round(h))
if max_pixels > 0 && h_bar*w_bar > max_pixels:
    beta = sqrt((w*h) / max_pixels)
    h_bar = max(align, floor(h/beta)); w_bar = max(align, floor(w/beta))
else if min_pixels > 0 && h_bar*w_bar < min_pixels:
    beta = sqrt(min_pixels / (w*h))
    h_bar = ceil(h*beta); w_bar = ceil(w*beta)
```

* 两边永远对齐到 `patch_size*merge` 的倍数，保证合并网格精确。
* max 分支优先（`else if`），因此 `max_pixels` 压过 `min_pixels`。
* min 分支放大小图（`ceil`），max 分支缩小大图（`floor`），都带 `max(align,…)` 下界。

### 2.4 Pillow 兼容 bicubic 重采样

`resize_bicubic`（`image.cpp:50-119`）是可分离两趟重采样。核是 Pillow 的 bicubic（`a=-0.5`），注释
强调 ggml/PyTorch 用 `a=-0.75`，这里**有意对齐 Pillow**，因为 llama.cpp 的参考预处理也用 Pillow。

每轴：`scale = src/dst`、`filterscale = max(1, scale)`、`support = 2*filterscale`、
`center = (x+0.5)*scale`，tap 范围 `[center-support+0.5, center+support+0.5)` 裁剪到源尺寸，权重
`filter_w((i-center+0.5)/filterscale)`。水平趟存 float 中间结果并按权重和归一化；垂直趟预计算 tap
权重、按 `wsum` 归一化，输出经 `clip8`（round-half-away + `[0,255]` clamp）。

### 2.5 归一化与 patchify

`mm_image_preprocess`：算目标尺寸，必要时重采样，然后

```
v = src[p*3+c] / 255.0f;
chw[c*n + p] = (v - mean[c]) / std[c];
```

### 2.6 预算常量

`kMaxImgTokens = 1024`（单图最大合并 token），`kMaxImgPatches = 4*kMaxImgTokens = 4096`（patch token）。
前者约束 `d_img_embd` 与 `max_pixels`，后者约束每图 patch 网格与设备 scratch。

---

## 3. 视觉编码器

### 3.1 类型

* `vt`：行主序权重视图 `{data, type, K, N}`。
* `vision_hparams`：`image_size, patch_size, n_embd, n_ff, n_layer, n_head, head_dim, proj_dim, merge,
  n_pos_side, eps, rope_base, mean/std[3]`。
* `vision_layer`：`qkv/out/up/down` + 可选 f32 bias，`ln1/ln1_b, ln2/ln2_b`。
* `vision_input`：`chw`、`width/height`、`pw=w/P`、`ph=h/P`、`n_patches=pw*ph`、
  `out_w=pw/merge`、`out_h=ph/merge`、`n_out=out_w*out_h`。
* `vision_model`：权重 + 设备状态；`patch_w` 是两个 patch conv 相加后反量化的 f32 `[n_embd][3*P*P]`；
  `pos_embd_host`；`dev_weights`。

### 3.2 mmproj 加载

元数据来自 `clip.vision.*`（`image_size`、`patch_size`、`embedding_length`、`feed_forward_length`、
`block_count`、`attention.head_count`、`projection_dim`、`spatial_merge_size`、`layer_norm_epsilon`、
`rope_theta`、`image_mean/std`）。张量：

* `v.patch_embd.weight` + `v.patch_embd.weight.1`：两个 4-D 卷积 `[kw,kh,in,out]`，校验
  `N=n_embd`、`K=3P²`，逐元素相加反量化为 `patch_w`；`v.patch_embd.bias` 可选。
* `v.position_embd.weight`：要求是正方形（`n_side=round(sqrt(N))`），反量化并转置成 `[N][K]`。
* `v.post_ln.weight/bias`。
* 每层 `v.blk.{i}.`: `attn_qkv.{weight,bias}`、`attn_out.{weight,bias}`、
  `ffn_up.{weight,bias}`、`ffn_down.{weight,bias}`、`ln1.{weight,bias}`、`ln2.{weight,bias}`。
* `mm.0.{weight,bias}`、`mm.2.{weight,bias}`。

约定：**线性层/嵌入为 BF16，norm/bias/patch/position 为 F32**（随附的 mmproj 即如此）；`bind_f32_opt`
强制 F32，缺失返回 nullptr。

### 3.3 上传与指针翻译

`upload` 把整个 mmproj 映射拷到设备一次；`dev_ptr` 用与文本模型相同的偏移恒等。`patch_w` 是合成数据，
不在映射内，因此单独分配设备副本 `d_patch_w`。析构有意不释放 `dev_weights`（随模型生命周期）。

### 3.4 host 参考前向（`encode_host`）

* **输入构建**：`build_patch_input` 按合并 token 顺序，每 token 抽 `[c][kh][kw]` 顺序的 patch，用
  `patch_w`（+bias）投影，再加位置嵌入（`pw==ph==n_side` 时精确，否则 align_corners 双线性）。
* **合并顺序**：循环 `my, mx, dy, dx`，因此每个 2×2 合并组内 token 连续，合并输出行是 4 个组 token 的
  拼接。这是整个合并阶段依赖的顺序契约。
* **Transformer 层**：LayerNorm（double 求均值/方差）→ fused QKV → 2D RoPE（见下）→ 双向 attention
  （无因果掩码，`1/sqrt(HD)`）→ out 投影 + residual → ln2 → `up → gelu_tanh → down` + residual。
* **merger**：`C = merge² = 4`，`hidden = C*n_embd`；`n_out` 行直接重解释为 `n_out` 个长度 `hidden`
  的向量（2×2 拼接是 stride 重解释），然后 `mm.0 → gelu → mm.2`，输出 `n_out * proj_dim`。

### 3.5 device 前向（`encode_device`）

守卫：非空网格、`np <= kMaxImgPatches`、`HD == 64`（attention kernel 硬编码 64）。

迟初始化：首次 `upload`；`d_patch_w` 一次性拷贝；scratch 在 `np` 变大时整体重分配（8 个缓冲），
`scratch_patches` 记录上次尺寸，因此后续更小的图复用 —— 即“增长到见过的最大图”。缓冲：
`d_patch_in = np*3P²`、`d_pos = np*E`、`d_x/d_ln/d_attn = np*E`、`d_qkv = np*3E`、
`d_ffn = np*ff`、`d_mm0 = n_out*hidden`。

流水线：host 构建 raw patch + pos → memcpy → patch 投影 GEMM（`N=E, K=patch_elems, T=np`）→ +bias
→ += pos → 每层（ln1 → qkv GEMM → +bias → RoPE → attn → out residual → ln2 → up +bias → GELU →
down residual）→ post LN → merger（`mm.0` GEMM 的 `K=hidden`、`x_stride=hidden` 即 2×2 拼接重解释 →
bias → GELU → `mm.2` → bias）→ `q.wait()`。

**队列必须 in-order**：各阶段数据依赖但启动器不发事件，因此要求 in-order 队列（引擎队列即是）。

### 3.6 视觉 kernel

见 [03-kernels.md](03-kernels.md)。要点：`vit_gemm`（128 线程 WG，
64 行 × 32 token tile）、`vit_layernorm`、`vit_gelu`、`vit_add_bias/add`、`vit_rope`（2D 视觉 RoPE）、
`vit_attn`（双向在线 softmax，`HD=64`）。

### 视觉 RoPE vs 文本 M-RoPE

* 视觉 tower：pair 索引切四个 `head_dim/4` 节，**前 16 对用 patch 行、后 16 对用 patch 列**，频率指数
  **每节重置**；由合并 token 反推 `(px,py)`。
* 文本模型：**交错** M-RoPE，pair 索引按 `0,1,2` sector 模式选 temporal/row/col 位置，频率指数用全局
  pair 索引。二者不可混淆。

---

## 4. 提示词组装（`multimodal.cpp`）

### 4.1 `mm_prompt`（`multimodal.h:27-38`）

```cpp
struct mm_prompt {
    std::vector<int> tokens;        // 展开后的 token id
    std::vector<int32_t> mrope;     // 4*n，section-major（[s*n+i]）
    std::vector<int32_t> img_row;   // n，-1 或嵌入表行号
    std::vector<float> embd;        // host 路径：n_img_total * n_embd
    const float * d_embd;           // device 路径：调用者拥有
    int n_img, pos_after;
};
```

### 4.2 几何

`make_inputs` 对每张图建 `vision_input`；`n_out <= 0` 抛错；强制全局
`total_rows + n_out <= kMaxImgTokens`；`offset[k]` 是第 k 张图嵌入的起始行。

### 4.3 token 扩展与 M-RoPE（`build_meta`）

* 找到 `<|image_pad|>` id（缺失抛错），按 `parse_special=true` 分词 `rendered`，要求 pad 数等于图片数。
* 对第 k 张图的每个 pad（`nx = out_w`, `base = pos`）：
  * 追加 `n_out` 个 pad token；
  * `img_row = offset[k] + t`；
  * temporal `pt = base`（所有 t 相同）；
  * row `ph = base + t/nx`；
  * col `pw = base + t%nx`；
  * `pos = base + max(out_w, out_h)` —— **图像消耗 `max(nx,ny)` 个位置，而不是每 token 一个**。
* 文本 token 每 token 进一个位置，`t,h,w` 都等于 `pos`。
* mrope 按 section-major 写：section 0/1/2 = temporal/row/col，section 3 = temporal（未使用，0 对）。
* `pos_after = pos`。

### 4.4 host / device 入口

* `mm_build_prompt`：`make_inputs` → `build_meta` → 分配 `embd` → 逐图 `encode_host` 拷到
  `offset[k]*n_embd`。
* `mm_build_prompt_device`：`make_inputs` → 逐图 `encode_device` 写 `d_out + offset[k]*n_embd` →
  `build_meta` → `p.d_embd = d_out`。`vm` 必须存活；`d_out` 由调用者拥有（引擎传 `e.d_img_embd`）。

---

## 5. 引擎集成

* `d_img_embd`：`kMaxImgTokens * n_embd` f32，启动时分配一次。
* `step_info` 多模态字段（`mrope_on`、`mrope_sections[4]`、`mrope[4*kMaxB*kMaxT]`、`img_embd`、
  `img_row[]`）在 **kernel 体内**读取，故图重放能拿到每一步的值。
* `mrope_sections` 在 `reset_state` 里从 `hp.rope_sections` 恢复。
* **embed kernel**：`img_embd && img_row[t] >= 0` 时直接拷贝，图像 token 不经过 `tok_embd`。
* **qk_norm_rope kernel**：`mrope_on` 时按交错 sector 模式从 `mrope` 选 `rpos`；频率指数仍是全局 pair
  索引。
* `generate_impl`：见 [04-engine.md](04-engine.md)。KV slot 序号是 token 计数，RoPE 位置
  由 `mrope` 单独携带 —— 二者因图像而分离。

---

## 6. 调用点

* **CLI**：`--image` 需要 `--mmproj`；从视觉超参建 `cfg`（`min_pixels=8*patch_area`，
  `max_pixels=kMaxImgTokens*patch_area`），逐文件解码/预处理，渲染带图片 part 的 chat，
  `mm_build_prompt_device(..., e.d_img_embd)`，`e.generate_mm`。
* **服务器**：`mm_server` 持有 `vision_model` 与 cfg；OpenAI `image_url` `data:` part 经
  `decode_image_url` → `mm_image_decode_mem` → `mm_image_preprocess` → 渲染 →
  `mm_build_prompt_device`；全程 `mm_req` 串行化。

---

## 7. 测试

`tests/mm/test_multimodal.cpp` 依次运行：

| 测试 | 校验 |
|---|---|
| `test_target_size` | 768×768 不变；1069×893 → 1056×896（两边 `%32==0`）；16×16 放大到 ≥min_pixels 且 32 对齐 |
| `test_vision` | 96×96 图：尺寸不变、`n_patches=36`、`n_out=9`、`embd.size()=n_out*proj_dim`、有限且非零范数 |
| `test_prompt` | `img_row`/`mrope` 尺寸、`pos_after = 1 + max(out_w,out_h) + 2`、图像行映射后清除、逐 token 的 temporal/row/col 位置 |
| `test_device` | host vs device 最大差 `<= 5e-3 * max(1, max|host|)`；打印 host/cold/warm 计时与加速比 |
| `test_kernels` | 每个视觉 kernel vs host 参考：gemm 1e-5、layernorm 1e-4、rope 1e-5、attn 1e-4 |
| `bench_kernels` | 真实 Qwen3.5 形状（T=256,E=768,ff=3072,NH=12,HD=64）的 kernel 计时 |

模型缺失或 tokenizer 无 `<|image_pad|>` 时优雅跳过。
