# 设计 01：模型加载与 CLI

覆盖 `src/model/gguf.{h,cpp}`、`model.{h,cpp}`、`model_arch.h`、`qwen35.cpp`、`model_w8.cpp`、
`src/main.cpp`。相关：SIn 格式细节见 [02-quantization.md](02-quantization.md)，分词见
[08-tokenizer.md](08-tokenizer.md)。

---

## 1. 设计目标

* **零转换**：GGUF 直接读，不生成中间权重文件。
* **零主机拷贝**：权重在主机侧只有 `mmap`，不做堆上副本。
* **一次设备上传**：整个 GGUF 映射（含元数据）一次 `memcpy` 到设备，之后用“偏移恒等”推导所有设备指针。
* **架构无关**：`general.architecture` → 加载器函数指针，注册表分发。

---

## 2. GGUF 解析（`gguf.cpp` / `gguf.h`）

### 2.1 文件布局

```
偏移 0:  magic u32 = 0x46554747 ("GGUF")
        version u32
        n_tensors u64
        n_kv u64
        [kv 表] name(string) type(u32) value(可变)
        [tensor 信息表] name(string) nd(u32) dims[u64] type(u32) offset(u64)
        <对齐到 general.alignment，默认 32>
        [张量数据区]
```

`gguf_file::load`（`gguf.cpp:173-187`）：

1. `open(path, O_RDONLY)`；失败抛 `cannot open <path>`。
2. `fstat` 取大小，`mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0)` 映射整个文件；随后立即关闭 fd。
   映射在 `~gguf_file` 中 `munmap`（`gguf.cpp:167-171`）。
3. 用游标式 `reader`（`gguf.cpp:84-163`）解析；`reader::need(n)` 对每次读取做边界检查，越界抛
   `gguf: unexpected end of file`。所有解析错误都是 `std::runtime_error`。
4. magic 不匹配抛 `not a gguf file`（`gguf.cpp:191-193`）。

### 2.2 元数据类型

`reader::read_value(type)`（`gguf.cpp:107-162`）实现 GGUF 的全部 KV 类型 id（0..12，含 array 递归）。
数值统一存进 `gguf_kv` 的 `u64`/`i64`/`f64` 字段并保持两种整数解释同步，因此 `as_u32`/`as_i32` 均安全。
数组读取用 `reserve(min(n, 1<<20))` 限制恶意 count 造成的过度分配（`gguf.cpp:144`）。未知 id 抛
`gguf: unknown kv type`。

元数据存入 `std::map<std::string, gguf_kv> kv`，访问器 `meta` / `get_u32` / `get_f32` / `get_str`
（`gguf.h:99-114`）。

### 2.3 张量信息

每个张量（`gguf.cpp:204-216`）：`name`、`nd`、`dims[]`、`type`（ggml 类型 id）、`offset`（相对数据区
的字节偏移）。`tensor_index[name]` 提供 O(log n) 查找。

约定：`dims[0]` 是最内层（ggml `ne[0]`），即行宽；`n_rows() = product(dims[1:])`；
`n_elements() = product(dims)`；`nbytes() = ggml_row_bytes(type, n_elements())`（`gguf.h:63-80`）。

### 2.4 对齐与数据指针

`data_offset` = 张量信息表结束位置向上对齐到 `general.alignment`（默认 32）（`gguf.cpp:218-223`）。
每个张量数据指针 = `map_base + data_offset + offset`（`gguf.cpp:225-227`）。加载时不逐张量校验映射
边界，依赖文件本身合法。

### 2.5 ggml 类型表

`ggml_blck_size` / `ggml_type_size` / `ggml_row_bytes`（`gguf.cpp:11-80`）覆盖 F32、F16、Q4_0/Q4_1、
Q5_0/Q5_1、Q8_0/Q8_1、Q2_K..Q8_K、BF16。块大小：F32/F16/BF16 = 1；Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 = 32；
所有 K-quant = 256。

计算路径实际消费的子集：F32（norm/小张量）、Q4_K/Q5_K/Q6_K（线性层）、BF16/F32（视觉 encoder）。

---

## 3. 架构注册表

```cpp
// model_arch.h:17-30
struct loader { const char *name; void (*load)(model & m); };
const loader * find(const std::string & name);
void load_qwen35(model & m);
wt bind_tensor(const gguf_file & f, const std::string & name, uint32_t expect_type = 0xFFFFFFFF);
const float * bind_f32(const gguf_file & f, const std::string & name);
```

`model.cpp:41-52` 中的静态表：

```cpp
static const loader kLoaders[] = { {"qwen35", &load_qwen35} };
```

`model::load(path)`（`model.cpp:59-71`）：

1. `gguf.load(path)`；
2. 若存在 `tokenizer.chat_template` 则拷进 `model::chat_template`（空表示 GGUF 未携带）；
3. 读 `general.architecture`，`arch::find`；未知抛 `unsupported architecture: <name>`；
4. 调用加载器填充 `hp`、绑定张量、构建 `layers`。

### 新增模型架构

1. 新建 `src/model/<arch>.cpp`，定义 `si::arch::load_<arch>(model & m)`（照抄 `qwen35.cpp` 结构），
   用 `arch::bind_tensor` / `arch::bind_f32` 绑定张量。
2. 在 `model_arch.h` 声明，在 `model.cpp` 的 `kLoaders[]` 注册。
3. 在 `CMakeLists.txt` 加入该 `.cpp`。`model::load` 自动分发；未知架构仍抛异常。

---

## 4. 张量绑定与设备上传

### 4.1 张量视图

```cpp
struct wt { const void *data; uint32_t type; int32_t K; int32_t N; }; // model.h:13-18
```

`K = dims[0]`（输入维度），`N = n_rows()`（输出行数）。`bind_tensor`（`model.cpp:12-26`）查找张量，
可选的期望类型校验（缺失抛 `missing tensor: <name>`，类型不符抛 `unexpected type for <name>`）。
`bind_f32`（`model.cpp:28-37`）要求 `GGML_TYPE_F32`，返回直接指向 `mmap` 的 `const float*`。

### 4.2 一次性上传与 `dev_ptr`

`model::upload(q)`（`model.cpp:83-90`）：

```cpp
dev_weights_size = gguf.map_size;                  // 整个文件，含元数据和 padding
dev_weights = sycl::malloc_device(dev_weights_size, q);
q.memcpy(dev_weights, gguf.map_base, dev_weights_size).wait();
```

设备指针用**偏移恒等**推导（`model.h:77-82`）：

```cpp
const void * dev_ptr(const void * host_ptr) const {
    return (const char *)dev_weights + ((const char *)host_ptr - (const char *)gguf.map_base);
}
```

之所以要上传整个映射而不是仅张量数据：所有 `wt.data` / `bind_f32` 返回的都是 `map_base` 内部的
指针，一次线性拷贝才能保持每个偏移有效。kernel 在录制图时捕获 `dev_ptr` 值，因此只要
`dev_weights` 存活，录制的图就一直有效。

`model::upload(q, host)`（`model.h:60-63`）的 `host=true` 分支 **不做拷贝**：`dev_weights = nullptr`，
`dev_ptr` 退化为恒等（直接返回 mmap 指针），这就是 CPU 后端的模式——权重留在 mmap 里，主机内核
直接读（见 [architecture.md §11](../architecture.md)）。多设备（`--layer-map`）不复用单一
`dev_weights`：`engine::setup_multi_device` 只把每个 GPU 后端 **自己分区** 的张量（外加始终落在
主设备上的 `tok_embd` / `output` / `output_norm`）拷到该设备，`engine::wptr(dev, host)` 按层解析出
该设备的指针（CPU 后端仍返回 mmap 指针）。

### 4.3 释放主机的 mmap 常驻页

mmap 只在上传/转换阶段被读；GPU 分区一旦落到设备，主机就不再需要这些页，但内核会一直把它们算进
进程 RSS（27B 的文件有 15.7 GB）。`model::page_out_host` 对一段映射先 `madvise(MADV_DONTNEED)`
（把页从进程摘掉，映射本身仍有效，日后误读只会重新缺页），再 `posix_fadvise(POSIX_FADV_DONTNEED)`
（此时 PTE 已摘除，内核可真正回收 page cache；仍被映射的 CPU 分区页会被内核跳过）。

释放时机有两处：

* `engine::setup_md_dnnl` 的 `add` 回调在每个张量 **转换成功**（w4/k5/cb4/int8，主机读取已完成）
  后立刻释放该张量——上传阶段会跳过已转换的张量，所以不会再次读取；
* `engine::upload_device_weights` 在把每个原始张量拷到设备后释放它（只对 GPU 分区调用）。

构造函数末尾的 `engine::release_host_weight_pages` 是兜底：释放未被任何分区引用的张量（如未绑定的
`nextn` 张量）、以及没有 CPU 分区时的 GGUF 头。CPU 分区的张量永远保留，因为它们的宿主内核每 token
都直接读 mmap。效果（27B + 双 A770）：稳态 RSS 16.2 GB → 474 MB，加载峰值 16.2 GB → 2.7 GB。

`build_meta32`（`engine.cpp:303-351`）也用主机张量指针作为 `meta32_` 的 key（见
[02-quantization.md](02-quantization.md)）。

---

## 5. 超参数与层绑定

### 5.1 `hparams`（`model.h:20-33`）

| 字段 | GGUF key |
|---|---|
| `n_layer` | `block_count` |
| `n_embd` | `embedding_length` |
| `n_ff` | `feed_forward_length` |
| `n_head` / `n_head_kv` | `attention.head_count` / `attention.head_count_kv` |
| `head_dim` | `attention.key_length` |
| `n_rot` | `rope.dimension_count` |
| `n_vocab` | `tokenizer.ggml.tokens` 数组长度 |
| `rope_base` | `rope.freq_base`（默认 10000） |
| `rms_eps` | `attention.layer_norm_rms_epsilon`（默认 1e-6） |
| `attn_scale` | 派生 `1/sqrt(head_dim)` |
| `d_state` / `n_group` / `dt_rank` / `d_inner` / `conv_k` | `ssm.*` |
| `full_attn_interval` | `full_attention_interval`（默认 4） |
| `rope_sections[4]` | `rope.dimension_sections`（全零 = 普通 RoPE） |

`is_recr(il) = (il+1) % full_attn_interval != 0`（`model.h:30-32`）。含义是“不是每个 interval 的最后一层”
为卷积/GDN 层。Qwen3.5 典型 `full_attn_interval=4`：`il % 4 == 3` 是 full attention，其余是 GDN。

### 5.2 `layer_t`（`model.h:35-54`）

* 公共：`recurrent`、`attn_norm`、`post_attn_norm`、`ffn_gate/up/down`。
* attention 层：`wq, wk, wv, wo` + `q_norm, k_norm`。
* GDN 层：`wqkv, wgate, ssm_beta, ssm_alpha, ssm_out` + `ssm_a, ssm_dt, ssm_norm, ssm_conv1d`。
* SI8 副本：`ffn_gate8/up8/down8`，attention 的 `wq8/wk8/wv8/wo8`，GDN 的 `wqkv8/wgate8/ssm_out8`。

`model` 还持有 `gdn_layer_index`（层 id → 该层在 GDN 层中的顺序索引，attention 层为 -1）、`tok_embd`、
`output_norm`、`tok_embd_row_bytes`、`tok_embd8`。

### 5.3 `load_qwen35`（`qwen35.cpp`）

见 [11-qwen35-model.md](11-qwen35-model.md)。

---

## 6. SIn int8 权重副本

`PF_DP4A`（默认开）时 `engine` 构造中调用 `m.build_w8(q)`，构建约 700 MB 的 int8 副本。格式与数学见
[02-quantization.md](02-quantization.md)。这里只说明加载期行为：

* `build_w8_tensor`（`model_w8.cpp:19-59`）跳过非 Q4_K/Q5_K/Q6_K 张量；按 `w8_vals_bytes` /
  `w8_meta_bytes` 分配设备缓冲；以 **4096 行为一 slab** 在主机暂存区重排后 `memcpy` 到设备，每个 slab
  后 `q.wait()`（暂存区复用）。LM head 约 250 MB，slab 化用于限制主机暂存内存。
* 覆盖集合：`tok_embd` + 3 个 FFN 线性 + 每层的 `wqkv/wgate/ssm_out`（GDN）或 `wq/wk/wv/wo`
  （attention）。`ssm_beta`、`ssm_alpha` 和所有 f32 norm **不**复制。
* `PF_SI4` 会影响 `w8_vals_bytes` / `w8_meta_bytes`（全 4-bit），因此副本尺寸随环境变化。
* `PF_META` 是引擎级可选项（不是 `model_w8.cpp`），为 Q4_K/Q5_K 构建 fp32 `(scale,min)` 旁路数组，
  实测为净损失，默认关闭（`engine.cpp:26-34`）。

---

## 7. CLI（`src/main.cpp`）

### 7.1 子命令与参数

子命令是命令行中出现的第一个非 flag token，最后一个生效；空则打印 usage 并返回 1。
识别 `serve` 与 `gen`。

| flag | 目标 | 默认 |
|---|---|---|
| `--model <gguf>` | 模型文件 | 编译期硬编码参考模型路径 |
| `--mmproj <gguf>` | 视觉 projector | 空 |
| `--image <file>` | `gen` 图像（可重复） | 空 |
| `--prompt <text>` | `gen` 提示词 | 空 |
| `--ctx N` / `--ctx full` | 最大序列长度 | `PF_CTX` 或 20480 |
| `--blocks N` | 启动时提交的 KV 块 | 512（lazy）或按 ctx 计算 |
| `--kv-cap-mb N` | KV 池上限，同时限制三层缓存预算之和 | auto |
| `--kv-type T` | KV 存储类型 `i4\|i8\|bf16\|f16\|f32`，或 `K:V` 分别指定（如 `i4:i8`）（覆盖 `PF_KV_TYPE`） | i8 |
| `--device cpu\|gpu\|auto` | 计算后端（`auto` 读 `PF_DEVICE`，否则 gpu） | auto |
| `--cpu-threads N` | CPU worker 线程数（0 = 物理核，回退硬件并发） | auto |
| `--layer-map L:dev,...` | 多设备层放置（pipeline parallel，闭区间无缝隙覆盖） | 空 |
| `--host H` / `--port N` | serve 绑定 | 0.0.0.0 / 8080 |
| `--max-tokens` / `--temp` / `--top-p` / `--top-k` | `gen` 采样 | 256 / 0.7 / 0.95 / 40 |
| `--raw` | `gen` 原样发送（不套 chat 模板） | off |
| `--pc-vram-mb` / `--pc-ram-mb` / `--pc-dir` / `--pc-disk-mb` | 前缀缓存三层预算 | 见 [06](06-prefix-cache.md) |
| `--pc-mem-mb` | `--pc-vram-mb` 别名 | -1 |

### 7.2 上下文与块数计算（`main.cpp:156-215`）

* `kDefaultCtx = 20480`。`--ctx full` 调 `model_context_length` 读 GGUF 元数据；≤0 报错。
* `lazy_blocks = ctx_auto || ctx_full`；`need_blocks = ceil(ctx / kBlockSize)`。
* `--blocks` 未给时：lazy → 512，否则 `max(512, need_blocks)`；显式 `--kv-cap-mb` 保证至少
  `need_blocks`。
* 打印 `[ctx]` 行：`max_seq`、`kv_blocks`、`kv_pool`、`kv_cap`、`kv_virtual|fixed`、`kv_type`；并用
  `sycl::aspect::ext_intel_free_memory` 在 KV cap 超过空闲显存一半时告警。

### 7.3 `gen` 文本路径

1. `gen_params` 从 flags 构造。
2. `utf8_stream_buffer` 包裹 emit 回调，逐 token 输出并 flush（保证多字节 UTF-8 不被截断）。
3. `--raw`：`tk.encode(prompt, parse_special=true)`；否则套 `{"user", prompt}` 后
   `render_chat(chat_template, msgs, add_generation_prompt=true, false)` 再编码。
4. `e.generate(toks, gp, emit)`，最后 flush + 换行。

### 7.4 `gen` 多模态路径

`--image` 必须配 `--mmproj`。加载 `vision_model`，从视觉超参导出 `image_preproc_cfg`
（`min_pixels = 8*patch_area`，`max_pixels = kMaxImgTokens*patch_area`），逐图
`mm_image_decode_file` → `mm_image_preprocess`，渲染带图像 part 的 chat，调
`mm_build_prompt_device` 把合并嵌入写进 `e.d_img_embd`，最后 `e.generate_mm`。该路径绕过前缀缓存。

### 7.5 信号处理

`main` 不安装信号处理器；`serve()` 安装 `SIGINT`/`SIGTERM`（只置原子标志），watchdog 线程负责
正常停止服务，使 `~engine` 能 flush 前缀缓存。详见 [09-server.md](09-server.md)。
