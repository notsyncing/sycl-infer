# 设计 01：模型加载与 CLI

覆盖 `src/model/gguf.{h,cpp}`、`model.{h,cpp}`、`model_arch.h`、`qwen35.cpp`、`model_w8.cpp`、
`src/main.cpp`。相关：SIn 格式细节见 [02-quantization.md](02-quantization.md)，分词见
[08-tokenizer.md](08-tokenizer.md)。

---

## 1. 设计目标

* **零转换**：GGUF 直接读，不生成中间权重文件。
* **零主机拷贝**：权重在主机侧只有 `mmap`，不做堆上副本。
* **一次设备上传**：整个 GGUF 映射（含元数据）一次 `memcpy` 到设备，之后用“偏移恒等”推导所有设备指针。
* **权重上设备后回收主机页**：`madvise(MADV_DONTNEED)` + `posix_fadvise(POSIX_FADV_DONTNEED)` 把 27B 的
  15.7 GB 文件页从进程 RSS 里摘掉（见 §4.3）——映射仍然有效，所以这是纯吞吐优化、不是正确性依赖。
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

`gguf_file::load`（`gguf.cpp:195-252`）：

1. `open(path, O_RDONLY)`；失败抛 `cannot open <path>`（`gguf.cpp:196-199`）。
2. `fstat` 取大小，`mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0)` 映射整个文件。
   **fd 不会立刻关闭**——它被保留下来给 `gguf_file::drop_cache` 发 `posix_fadvise`
   （`gguf.cpp:211`、`gguf.cpp:188-193`），`~gguf_file` 才 `munmap` + `close`
   （`gguf.cpp:179-186`）。
3. 用游标式 `reader`（`gguf.cpp:96-175`）解析；`reader::need(n)`（`gguf.cpp:100-104`）对每次读取做边界
   检查，越界抛 `gguf: unexpected end of file`。所有解析错误都是 `std::runtime_error`。
4. magic 不匹配抛 `not a gguf file`（`gguf.cpp:214-217`）。

### 2.2 元数据类型

`reader::read_value(type)`（`gguf.cpp:119-174`）实现 GGUF 的全部 KV 类型 id（0..12，含 array 递归）。
数值统一存进 `gguf_kv` 的 `u64`/`i64`/`f64` 字段并保持两种整数解释同步，因此 `as_u32`/`as_i32` 均安全。
数组读取用 `reserve(min(n, 1<<20))` 限制恶意 count 造成的过度分配（`gguf.cpp:156`）。未知 id 抛
`gguf: unknown kv type`。

元数据存入 `std::map<std::string, gguf_kv> kv`，访问器 `meta` / `get_u32` / `get_f32` / `get_str`
（`gguf.h:110-125`）。三者都**带默认值**：key 缺失时返回 `def`，所以加载器里
`get_u32(key("full_attention_interval"), 4)` 这类写法不区分“缺键”和“值就是 4”。

### 2.3 张量信息

每个张量（`gguf.cpp:228-240`）：`name`、`nd`、`dims[]`、`type`（ggml 类型 id）、`offset`（相对数据区
的字节偏移）。`tensor_index[name]` 提供 O(log n) 查找。

约定：`dims[0]` 是最内层（ggml `ne[0]`），即行宽；`n_rows() = product(dims[1:])`；
`n_elements() = product(dims)`；`nbytes() = ggml_row_bytes(type, n_elements())`（`gguf.h:67-84`）。

### 2.4 对齐与数据指针

`data_offset` = 张量信息表结束位置向上对齐到 `general.alignment`（默认 32）；超出文件则抛
`gguf: bad data offset`（`gguf.cpp:242-247`）。
每个张量数据指针 = `map_base + data_offset + offset`（`gguf.cpp:249-251`）。加载时不逐张量校验映射
边界，依赖文件本身合法。

### 2.5 ggml 类型表

`ggml_type_name` / `ggml_blck_size` / `ggml_type_size` / `ggml_row_bytes`
（`gguf.cpp:11-34`、`gguf.cpp:36-59`、`gguf.cpp:61-84`、`gguf.cpp:86-92`）覆盖 F32、F16、BF16、
Q4_0/Q4_1、Q5_0/Q5_1、Q8_0/Q8_1、Q2_K..Q8_K，以及 IQ3_XXS / IQ4_NL / IQ3_S / IQ4_XS。

块大小（`ggml_blck_size`，即每块的元素数）：F32/F16/BF16 = 1；Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 = 32；
Q2_K..Q8_K 与 IQ3_XXS/IQ3_S/IQ4_XS = 256；**IQ4_NL = 32**（不是 256，它用的是 32 元素块）。
`ggml_row_bytes` 在 `n_elems % blck_size != 0` 时抛 `row size not divisible by block size`，所以
K-quant 线性层的 `K`（= `dims[0]`）必须是块大小的整数倍。反过来，像
`blk.0.ssm_conv1d.weight = [4, 6144]` 这种“转置读”的 conv 核一旦被 `bind_tensor` 绑成 `wt`，
得到的会是 `K=4, N=6144`（完全反的形状）——它必须用 `bind_f32` 按裸 f32 数组读，加载器正是这么做的
（`qwen35.cpp:81`）。

计算路径实际消费的子集：F32（norm/标量）、Q4_K/Q5_K/Q6_K（线性层主力）、Q8_0/Q8_1
（`ssm_beta`/`ssm_alpha` 这类窄 int8 行）、IQ4_XS/IQ4_NL（走 `PF_CB4` 的 codebook 4-bit 原生存储，
见 [02-quantization.md](02-quantization.md)）、BF16/F32（视觉/音频 encoder 的线性层）。

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

`model.cpp:44-48` 中的静态表（**目前只有 qwen35 一行**）：

```cpp
static const loader kLoaders[] = {
    {"qwen35", &load_qwen35},
};
```

`arch::find`（`model.cpp:50-57`）线性扫这张表，返回 `nullptr` 表示未知架构。

`model::load(path)`（`model.cpp:64-76`）：

1. `gguf.load(path)`；
2. 若存在 `tokenizer.chat_template` 则拷进 `model::chat_template`（空表示 GGUF 未携带）；
3. 读 `general.architecture`（缺失按空串处理），`arch::find`；未知抛 `unsupported architecture: <name>`；
4. 调用加载器填充 `hp`、绑定张量、构建 `layers`。

`model_context_length(path)`（`model.cpp:78-86`）是加载器之外唯一独立读 GGUF 的入口：它自己 `load()`
一遍，只取 `<arch>.context_length`，**不构造 `model`**，因此可以在引擎之前用来给 `--ctx full` 定大小
（见 §7.2）。它对没有该键的模型返回 0。

### 3.2 加载期的几何校验（`validate_hparams`）

`model::load` 在架构 loader 填完 `hp` 之后调用 `validate_hparams(hp, arch)`（`model.cpp:62-105`）。
**几何不满足 kernel 硬编码假设的 GGUF 在加载期被拒绝**，而不是"算错"或"算零"。

原因不是精度而是内存安全：`attn.cpp` 的通用（非特化）注意力 kernel 用**字面量**
`constexpr int HD = 256` 计算每个 split 的 partials 长度，却用 `pstride = 2 + head_dim` 定位。
所以 `head_dim = 128` 会让它往 130 个 float 的槽位里写 256 个 float，**越界写坏 partials 缓冲**。
特化 kernel（oneDNN 注意力、flash、grouped）全部以 `head_dim == HD` 为条件，不满足时**回落到同一个
通用 kernel**——因此除 256 以外没有任何受支持形状。`qk_norm_rope.cpp` 的 QK-norm 同理：RMSNorm 循环
是 `head_dim / 32` 配 32 宽 sub-group，非 32 的倍数会静默漏掉尾部维度。

当前校验项：`head_dim == 256`、`head_dim % 32 == 0`、
`n_head > 0 && n_head_kv > 0 && n_head % n_head_kv == 0`（GQA 比值被多处 kernel 整除）、
`n_layer/n_embd/n_vocab > 0`、`full_attn_interval > 0`（`is_recr` 对它取模）、以及存在
recurrent 层时 `d_state/n_group/dt_rank/conv_k > 0`。消息带架构名。

**新增架构自动继承这一关**：只要 loader 通过 `kLoaders[]` 注册，填出一个 kernel 不支持的几何时
报错发生在分配任何设备内存之前。测试 `tests/model/test_hparams.cpp` 不需要 GGUF 或 GPU，
直接驱动该函数。

### 新增模型架构

1. 新建 `src/model/<arch>.cpp`，定义 `si::arch::load_<arch>(model & m)`（照抄 `qwen35.cpp` 结构），
   用 `arch::bind_tensor` / `arch::bind_f32` 绑定张量。
2. 在 `model_arch.h` 声明，在 `model.cpp` 的 `kLoaders[]` 注册。
3. 在 `CMakeLists.txt` 加入该 `.cpp`。`model::load` 自动分发；未知架构仍抛异常。
4. 若该架构的几何超出 §3.2 的约束，先扩展 `validate_hparams` 并放宽对应 kernel——**不要**让一个
   不受支持的 head_dim 走到 kernel 里越界写内存。

---

## 4. 张量绑定与设备上传

### 4.1 张量视图

```cpp
struct wt { const void *data; uint32_t type; int32_t K; int32_t N; }; // model.h:13-18
```

`K = dims[0]`（输入维度），`N = n_rows()`（输出行数）。`bind_tensor`（`model.cpp:17-31`）查找张量，
可选的期望类型校验（缺失抛 `missing tensor: <name>`，类型不符抛 `unexpected type for <name>`）。
`bind_f32`（`model.cpp:33-42`）要求 `GGML_TYPE_F32`（否则抛 `expected f32 for <name>`），返回直接指向
`mmap` 的 `const float*`。两者都**不做边界/形状推断**：一个 `wt` 只带 `data/type/K/N`，宽度假设错写成
别处（`3*d_inner` 之类）不会有任何报错，见 [11-qwen35-model.md §2](11-qwen35-model.md)。

### 4.2 一次性上传与 `dev_ptr`

`model::upload(q)`（`model.cpp:88-100`）：

```cpp
dev_weights_size = gguf.map_size;                  // 整个文件，含元数据和 padding
dev_weights = sycl::malloc_device(dev_weights_size, q);
q.memcpy(dev_weights, gguf.map_base, dev_weights_size).wait();
```

设备指针用**偏移恒等**推导（`model.h:134-139`）：

```cpp
const void * dev_ptr(const void * host_ptr) const {
    if (!dev_weights) return host_ptr;             // CPU 后端 / 多设备
    return (const char *)dev_weights + ((const char *)host_ptr - (const char *)gguf.map_base);
}
```

之所以要上传整个映射而不是仅张量数据：所有 `wt.data` / `bind_f32` 返回的都是 `map_base` 内部的
指针，一次线性拷贝才能保持每个偏移有效。kernel 在录制图时捕获 `dev_ptr` 值，因此只要
`dev_weights` 存活，录制的图就一直有效。

`model::upload(q, host)`（`model.h:113-115`）的 `host=true` 分支 **不做拷贝**：`dev_weights = nullptr`，
`dev_ptr` 退化为恒等（直接返回 mmap 指针），这就是 CPU 后端的模式——权重留在 mmap 里，主机内核
直接读（见 [architecture.md §11](../architecture.md)）。`engine` 构造时对 CPU 后端和多设备都传
`host=true`（`engine.cpp:175`），所以多设备**不复用单一 `dev_weights`**：`engine::setup_multi_device`
只把每个 GPU 后端 **自己分区** 的张量（外加始终落在主设备上的 `tok_embd` / `output` / `output_norm`）
拷到该设备，`engine::wptr(dev, host)`（`engine.cpp:842-859`）按层解析出该设备的指针（CPU 后端仍返回
mmap 指针）。

### 4.3 释放主机的 mmap 常驻页

mmap 只在上传/转换阶段被读；GPU 分区一旦落到设备，主机就不再需要这些页，但内核会一直把它们算进
进程 RSS（27B 的文件有 15.7 GB，`engine.cpp:1001-1003`）。`model::page_out_host`（`model.cpp:106-128`）
对一段映射先 `madvise(MADV_DONTNEED)`（把页从进程摘掉，映射本身仍有效，日后误读只会重新缺页），再
`posix_fadvise(POSIX_FADV_DONTNEED)`（`gguf_file::drop_cache`，`gguf.cpp:188-193`；此时 PTE 已摘除，
内核可真正回收 page cache；仍被映射的 CPU 分区页会被内核跳过）。范围会先对齐到页边界再裁剪到映射内。

> **“重新缺页”是纯粹的吞吐惩罚，不是正确性问题。** `engine` 构造之后任何新增的 host 端权重读取都会
> 从磁盘重新 fault 进来——包括诊断代码。映射地址仍然有效，所以结果正确，但 27B 上一次 fault 是几百
> MB 的磁盘 IO。凡是“转换/上传之后再读 host 张量”的代码路径都要检查这一条。

释放时机有三处：

* `engine::setup_md_dnnl` 的 `add` 回调在每个张量 **转换成功**（w4/k5/cb4/int8，主机读取已完成）
  后立刻 `page_out_tensor`——上传阶段会跳过已转换的张量，所以不会再次读取（`engine.cpp:1145-1177`）；
* `engine::upload_device_weights` 在把每个原始张量拷到设备后释放它（`engine.cpp:983-990`，只对 GPU
  分区调用）。**唯一的例外**是 tied LM head：`md_xmx && m.output.data == m.tok_embd.data` 时
  `token_embd` 的页被保留，因为它的 oneDNN 转换必须在上传之后才做（见 §4.4）；
* 构造函数末尾的 `engine::release_host_weight_pages()`（`engine.cpp:511`、`engine.cpp:1008-1073`）是
  兜底：释放未被任何分区引用的张量（如未绑定的 `nextn` 张量）、以及 `keep` 为空时的 GGUF 头。

`release_host_weight_pages` 的 `keep` 集合（`engine.cpp:1012-1057`）有两类成员，**漏掉任何一类都会
让本该常驻的页被摘掉**：

* 每个 `dev_kind_ == 1`（CPU）分区的层张量——它们的宿主内核每 token 都直接读 mmap；
* 当**分区 0 本身是 CPU**（all-`cpu` 的 `--layer-map`）时的三个全局张量 `tok_embd` / `output` /
  `output_norm`：这条路径下它们**从不上传**（没有 GPU 拷贝），只能靠 `keep` 保住。

效果（27B，双 A770，`--layer-map 0-31:gpu.0,32-63:gpu.1`）：稳态 RSS 16.2 GB → 474 MB，加载峰值
16.2 GB → 2.7 GB，模型文件的 page cache 15.7 GB → 66 MB；同一次改动下的门禁不变（`test_w4_vs_cpuref`
argmax SAME、`test_decode_vs_prefill` OK、`test_gpu_stages` all stages OK）。运行时会打印
`[mem] released <MB> of host-resident GGUF pages (RSS)`，这是判断 `keep` 有没有写漏的第一手证据。

`build_meta32`（`engine.cpp:666-710`）也用主机张量指针作为 `meta32_` 的 key（见
[02-quantization.md](02-quantization.md)）。

### 4.4 多设备 LM head：key 取决于 tied 还是 untied

这是加载/绑定侧最容易“静默变慢”的一处。`setup_md_dnnl` 建立的 oneDNN int8/u4 权重表**以 host 指针为
key**（`engine.cpp:1150-1164`），而 `upload_device_weights` 会跳过任何已转换张量的裸设备拷贝
（`engine.cpp:955-964`）——于是 `engine::wkey(dev, host)`（`engine.h:177-187`）的返回值有两种，取决于
head 是否 tied：

| 情况 | `m.output.data` | head 的 oneDNN key | 转换时机 |
|---|---|---|---|
| **untied**（GGUF 有 `output.weight`，27B） | 与 `tok_embd` 不同 | **host 指针** | 上传**之前**，`engine.cpp:1258-1272` |
| **tied**（无 `output.weight`，0.8B） | `== m.tok_embd.data` | **已上传的 device 指针** | 上传**之后**，`engine.cpp:1667-1672` |

tied 之所以要反过来：它与 `token_embd` 共用同一份存储，而 embed kernel 必须读到 GGUF 量化后的原始行
（`token_embd.weight` 是 Q6_K），所以裸拷贝不能被 int8 head 顶替。于是先上传 embedding，再用
`wptr(0, m.output.data)` 取到它的 device 指针作为 key 做转换。

漏掉任何一个前提（key 用错、或 head 的 GEMV 段没有 `xq` 条目）都不会报错，只会让 head **静默掉回
fp32 dequant GEMV**：记录在 `engine_graph.cpp:301-306` 注释里的实测是 **27B / 2x A770 上 12.3 →
3.7 ms/token**。因此 plan 构建 head 段时必须用 `wkey` 而不是 `wptr`（`engine_graph.cpp:265-271`），
并且 `gemv_at` 的 oneDNN 分支要覆盖单 token 调用。engine 侧的其它相关不变量（`bind_acts(0)` 必须在
构建 head 段之前、`cur_dev` 复位）见 [04-engine.md](04-engine.md)。

---

## 5. 超参数与层绑定

### 5.1 `hparams`（`model.h:20-45`）

| 字段 | GGUF key |
|---|---|
| `n_layer` | `block_count`（架构 loader 会再减去 MTP 层数，见下） |
| `n_mtp` | `<arch>.nextn_predict_layers`（缺省 0） |
| `n_embd` | `embedding_length` |
| `n_ff` | `feed_forward_length` |
| `n_head` / `n_head_kv` | `attention.head_count` / `attention.head_count_kv` |
| `head_dim` | `attention.key_length` |
| `n_rot` | `rope.dimension_count` |
| `n_vocab` | `tokenizer.ggml.tokens` 数组长度 |
| `rope_base` | `rope.freq_base`（代码默认 10000；0.8B/27B 的 GGUF 里实际是 1e7） |
| `rms_eps` | `attention.layer_norm_rms_epsilon`（代码默认 1e-6） |
| `attn_scale` | 派生 `1/sqrt(head_dim)` |
| `d_state` / `n_group` / `dt_rank` / `d_inner` / `conv_k` | `ssm.*` |
| `full_attn_interval` | `full_attention_interval`（默认 4） |
| `rope_sections[4]` | `rope.dimension_sections`（全零 = 普通 RoPE） |

**注意 `block_count` 包含 MTP 层**：`load_qwen35` 算的是 `hp.n_layer = block_count - n_mtp`
（`qwen35.cpp:20-23`），所以 `hp.n_layer` 只数主干 block，NextN 头落在 `blk.<hp.n_layer>.*`。
0.8B 的 GGUF 没有 `nextn_predict_layers`，`hp.n_layer == block_count == 24`。

`is_recr(il) = (il+1) % full_attn_interval != 0`（`model.h:33-35`）。含义是“不是每个 interval 的最后一层”
为卷积/GDN 层。Qwen3.5 典型 `full_attn_interval=4`：`il % 4 == 3` 是 full attention，其余是 GDN。

### 5.2 `layer_t`（`model.h:47-66`）

* 公共：`recurrent`、`attn_norm`、`post_attn_norm`、`ffn_gate/up/down`。
* attention 层：`wq, wk, wv, wo` + `q_norm, k_norm`。
* GDN 层：`wqkv, wgate, ssm_beta, ssm_alpha, ssm_out` + `ssm_a, ssm_dt, ssm_norm, ssm_conv1d`。
* SI8 副本：`ffn_gate8/up8/down8`，attention 的 `wq8/wk8/wv8/wo8`，GDN 的 `wqkv8/wgate8/ssm_out8`。

`model` 还持有 `gdn_layer_index`（层 id → 该层在 GDN 层中的顺序索引，attention 层为 -1）、`tok_embd`、
`output`、`output_norm`、`tok_embd_row_bytes`、`output8`（LM head 的 SIn 副本）。

⚠️ **`m.output` 与 `m.tok_embd` 不是同一个字段**：`load_qwen35` 只在 GGUF 里**没有**
`output.weight` 时才让 `m.output = m.tok_embd`（`qwen35.cpp:48-55`）。任何算 logits 的地方必须用
`m.output`——两者的差别见 §4.4，以及 [11-qwen35-model.md §7](11-qwen35-model.md)。

### 5.3 `mtp_layer_t`（`model.h:73-88`）

> DFlash2 草稿器是**另一个 GGUF**，不在本节：`src/model/dflash.{h,cpp}` 加载
> `--spec-draft-model` 指向的草稿模型，绑定它自己的张量（词表/embedding/LM head 借目标）。
> 见 [设计 14](14-dflash2.md)。

MTP/NextN 草稿头，**只有 `hp.n_mtp > 0` 时才绑定**（`m.has_mtp = true`），否则所有字段为空、
`model::build_w8` / `setup_md_dnnl` 的 `if (mtp_on)` 分支根本不会走到它：

* 一个完整 full-attention block：`attn_norm`、`post_attn_norm`、`wq/wk/wv/wo`、`q_norm`、`k_norm`、
  `ffn_gate/up/down`——注意它是 **attention** block，不是 GDN；
* NextN 额外张量：`eh_proj`（`[2*n_embd][K=2*n_embd] -> n_embd`，把 `concat(enorm(emb(t_p)), hnorm(h_{p-1}))`
  投回 `n_embd`）、`enorm`、`hnorm`、可选的 `shared_head_norm`；
* `shared_head`：`blk.<n>.nextn.shared_head_head.weight` **存在才绑定**（`qwen35.cpp:114-116`），
  空则草稿头回落到 `m.output`。注意 `shared_head_norm` 相反——它是无条件 `bind_f32`，缺失会抛
  `missing tensor`，所以“可选”只对 `shared_head_head` 成立；
* 每设备的 SIn 副本：`wq8/wk8/wv8/wo8/ffn_gate8/ffn_up8/ffn_down8/eh_proj8`。

MTP 层的 KV 归属：它是一个 full-attention 层，因此**自己占一份 paged KV slice**
（`engine_kvpool.cpp:102-115`：`attn_layers()` 在 `mtp_on` 时 `+1`，索引为最后一层），共享 block table、
KV 存储类型与前缀缓存，并计入 `--kv-cap-mb` 与三层前缀缓存预算。MTP 的 plan/forward/verify/rollback
语义见 [04-engine.md](04-engine.md)。

### 5.4 `load_qwen35`（`qwen35.cpp`）

见 [11-qwen35-model.md](11-qwen35-model.md)。

---

## 6. SIn int8 权重副本

`PF_DP4A`（默认开）时 `engine` 构造中调用 `m.build_w8(q)`（`engine.cpp:194-196`），构建约 700 MB 的
int8 副本。格式与数学见 [02-quantization.md](02-quantization.md)。这里只说明加载期行为：

* `build_w8_tensor`（`model_w8.cpp:19-61`）跳过既不是 Q4_K/Q5_K/Q6_K、也不在 `w8_requant_type`
  列表里的张量；后者是 Q3_K / IQ4_NL / IQ3_S / IQ4_XS——它们没有原生 SIn 打包，被 dequant 到 fp32 再
  重量化到 per-32 的 4-bit 非对称组（`w8.h:89-93`）。之后按 `w8_vals_bytes` / `w8_meta_bytes` 分配
  设备缓冲；以 **4096 行为一 slab** 在主机暂存区重排后 `memcpy` 到设备，每个 slab 后 `q.wait()`
  （暂存区复用，否则异步拷贝会与下一 slab 覆写竞争）。LM head 的副本最大（27B 约 1 GB 的 GGUF 行），
  slab 化用于限制主机暂存内存。
* 覆盖集合（`model_w8.cpp:96-113`）：**LM head `m.output` → `m.output8`**（注意不是 `tok_embd`；
  tied head 上二者同源）、每层的 3 个 FFN 线性、以及 GDN 层的 `wqkv/wgate/ssm_out` 或 attention 层的
  `wq/wk/wv/wo`。`ssm_beta`、`ssm_alpha` 和所有 f32 norm **不**复制。
* CPU 后端不构建这些副本：`pf8` 只在非 CPU 且非多设备时为真（`engine.cpp:188-196`），CPU 的整数
  kernel 直接读 GGUF 块。
* `PF_SI4` 会影响 `w8_vals_bytes` / `w8_meta_bytes`（全 4-bit），因此副本尺寸随环境变化。
* `PF_META` 是引擎级可选项（不是 `model_w8.cpp`）：`use_meta32` 要求非 CPU 且非多设备
  （`engine.cpp:176-184`），`build_meta32` 为 Q4_K/Q5_K 构建 fp32 `(scale,min)` 旁路数组
  （`engine.cpp:666-710`），实测为净损失（多一条内存流），默认关闭。

---

## 7. CLI（`src/main.cpp`）

### 7.1 子命令与参数

子命令是命令行中出现的第一个非 flag token，最后一个生效；空则打印 usage 并返回 1。识别 `serve` 与
`gen`。缺值的 flag 是硬错误（`%s requires a value`），未知 flag 也直接 `usage` + 返回 1
（`main.cpp:174-185`、`main.cpp:290-306`）。

| flag | 目标 | 默认 |
|---|---|---|
| `--model <gguf>` | 模型文件 | 编译期硬编码参考模型路径（`main.cpp:43`） |
| `--device cpu\|host\|gpu\|auto` | 计算后端（`auto` 读 `PF_DEVICE`，否则 gpu） | auto |
| `--cpu-threads N` | CPU worker 线程数（0 = 物理核，回退硬件并发；env `PF_CPU_THREADS`） | auto |
| `--layer-map L:dev,...` | 多设备层放置（pipeline parallel，闭区间无缝隙覆盖） | 空 |
| `--ctx N` / `--ctx full` | 最大序列长度 | `PF_CTX` 或 20480 |
| `--blocks N` | 启动时提交的 KV 块 | 512（lazy）或按 ctx 计算 |
| `--kv-cap-mb N` | KV 池上限，同时限制三层缓存预算之和 | auto |
| `--kv-type T` | KV 存储类型 `i4\|i8\|bf16\|f16\|f32`，或 `K:V` 分别指定（如 `i4:i8`）（覆盖 `PF_KV_TYPE`） | i8 |
| `--mmproj <gguf>` | 视觉 projector（`--image` / `--video` 需要） | 空 |
| `--audio-mmproj <gguf>` | 音频塔（`--audio` 需要，单独加载第二个 GGUF） | 空 |
| `--spec-type T` | 投机解码类型 `none\|mtp\|dflash2`（llama.cpp 的拼法） | 关（env `PF_SPEC_TYPE`） |
| `--spec-draft-model <gguf>` | DFlash2 草稿 GGUF（`--spec-type dflash2` 需要） | 空 |
| `--spec-draft-n-max N` | 每 cycle 的草稿 token 数（DFlash2） | 5（env `PF_DFLASH_NMAX`） |
| `--spec-draft-device N` | 草稿跑在哪个设备分区 | 0（env `PF_MTP_DEV` / `PF_DFLASH_DEV`） |
| `--mtp [N]` | `--spec-type mtp --spec-draft-n-max N` 的别名；**长度可选**，裸 `--mtp` = 4（引擎侧再把 >12 截到 12） | 关（env `PF_MTP` 可覆盖） |
| `--mtp-device N` | `--spec-draft-device` 的别名 | 0（env `PF_MTP_DEV`） |
| `--host H` / `--port N` | serve 绑定 | 0.0.0.0 / 8080 |
| `--prompt <text>` / `--raw` | `gen` 提示词 / 原样发送（不套 chat 模板） | 空 / off |
| `--thinking`（别名 `--enable-thinking`） | `gen` 传给 chat 模板的 `enable_thinking` | off |
| `--image <file>` / `--video <file>` / `--audio <file>` | `gen` 多模态输入（各自可重复、可混用） | 空 |
| `--max-video-frames` / `--max-video-side` | 视频抽帧数 / 帧边长上限 | 16 / 768 |
| `--max-tokens` / `--temp` / `--top-p` / `--top-k` | `gen` 采样 | 256 / 0.7 / 0.95 / 40 |
| `--pc-vram-mb` | 设备层预算；换算成检查点数量 | 见 [06](06-prefix-cache.md) |
| `--pc-mem-mb` | `--pc-vram-mb` 别名 | -1（未设置） |
| `--pc-ram-mb` | 主机 RAM 层预算（0 关闭该层） | -1（未设置 → 引擎默认 512 MB） |
| `--pc-dir DIR` / `--pc-disk-mb` | 磁盘层目录 / 预算（0 = 无上限） | 空（关闭） / -1（未设置 → 1024 MB） |
| `-h` / `--help` | usage | - |

三类 prefix cache 预算的默认在引擎侧（`engine.cpp:275-303`）：VRAM 未给时先看 `PF_PC_STATES`、再按
`--pc-vram-mb`/`PF_PC_VRAM_MB` 除以每节点字节，最后回落成历史默认 8 个检查点；RAM 512 MB、disk
1024 MB。显式 `--kv-cap-mb` 会把这三个预算的和一起压住（先缩 disk，再 RAM，最后 VRAM 检查点数）。

`--mtp` 的可选长度是刻意设计的（`main.cpp:252-270`）：下一个 token 全是数字时才吞掉它，所以裸
`--mtp` 永远不会吃掉后面的 flag（`--mtp gen ...` 是合法的）。k=4 是实测最优——每个额外草稿约 +5.4 ms、
每个额外 verify 行约 +5.7 ms，而 k>4 的边际接受率只有 0.1-0.2（27B / 2x A770，u4 草稿头，两个提示词
的 k 扫描）。

### 7.2 上下文与块数计算（`main.cpp:42-43`、`main.cpp:308-345`）

* `kDefaultCtx = 20480`。`--ctx full` 调 `model_context_length(model_path)` 读 GGUF 元数据；读不到
  metadata 或结果 ≤0 都报错退出（`main.cpp:308-319`）。未给 `--ctx` 且没有 `PF_CTX` 时
  `ctx_auto = true`，回落到 20480（`main.cpp:321-324`）。
* `lazy_blocks = ctx_auto || ctx_full`；`need_blocks = ceil(ctx / kBlockSize)`。
* `--blocks` 未给时：lazy → 512，否则 `max(512, need_blocks)`；显式 `--kv-cap-mb` 保证至少
  `need_blocks`（`main.cpp:335-340`）。注意 `--ctx full` 也走 lazy：一个 262144 的上下文不会在启动时
  一次性提交 8192 个块。
* 打印 `[ctx]` 行：`max_seq`、`kv_blocks`、`kv_pool`、`kv_cap`、`kv_virtual|fixed`、`kv_type`
  （外加 KB/token）；并用 `sycl::aspect::ext_intel_free_memory` 在 KV cap 超过空闲显存一半时告警
  （`main.cpp:356-385`）。

### 7.3 `gen` 文本路径

1. `gen_params` 从 flags 构造。
2. `utf8_stream_buffer` 包裹 emit 回调，逐 token 输出并 flush（保证多字节 UTF-8 不被截断）。
3. `--raw`：`tk.encode(prompt, parse_special=true)`；否则套 `{"user", prompt}` 后
   `render_chat(e.m.chat_template, msgs, /*add_generation_prompt=*/true, thinking)` 再编码——第四个
   参数就是 `--thinking`（`main.cpp:521-527`）。
4. `e.generate(toks, gp, emit)`，最后 flush + 换行（`main.cpp:542-545`）。

`PF_DUMP_PROMPT` 会额外打印渲染后的 chat 文本和 token id，是区分“模板坏”与“前向坏”的第一手工具。

### 7.4 `gen` 多模态路径

任一 `--image`/`--video`/`--audio` 存在就走这条路径（`main.cpp:412-520`）：`--image`/`--video` 要求
`--mmproj`，`--audio` 要求 `--audio-mmproj`（两者分别抛清晰的错误信息，`main.cpp:428-460`）。视觉塔加载
后从超参导出 `image_preproc_cfg`（`min_pixels = 8*patch_area`，`max_pixels = kMaxImgTokens*patch_area`），
音频塔导出 `audio_preproc_cfg`（采样率/FFT/hop/mel/f_min/f_max）。三者解码后按 `mm_media_ref order`
的占位符顺序组装，渲染带媒体 part 的 chat，交给
`mm_build_prompt_mixed_device(...)` 把合并嵌入写进 `e.d_img_embd`，最后 `e.generate_mm`。该路径绕过
前缀缓存。图像/视频/音频的预处理与位置编码见 [10-multimodal.md](10-multimodal.md) 与
[13-audio-video.md](13-audio-video.md)。

### 7.5 信号处理

`main` 不安装信号处理器；`serve()` 安装 `SIGINT`/`SIGTERM`（`server.cpp:1916-1917`，处理函数只置原子
标志 `g_term_requested`），watchdog 线程（100 ms 轮询）负责调 `srv.stop()` 正常停止服务，使 `~engine`
能 flush 前缀缓存。详见 [09-server.md](09-server.md)。
