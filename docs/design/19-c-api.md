# C API 设计（`libsycl_infer`）

本文定义 `sycl-infer` 的嵌入式 C ABI：一个可以静态/动态链接进宿主程序、在进程内加载
GGUF 模型并执行批量推理的稳定接口。它是 C++ 引擎的**薄封装**，不是第二套推理实现：
模型加载、分页 KV、连续批处理、采样、前缀缓存全部复用现有组件。Python binding
（[20-python-binding.md](20-python-binding.md)）只建立在这一层之上。

> 状态：设计稿（v1）。本文先冻结 ABI 形状与语义，再进入实现。实现阶段按 §13 的
> 里程碑拆成独立改动，每个里程碑都有可运行的测试，不一次性重写引擎。

---

## 1. 目标与非目标

### 1.1 目标

1. **进程内嵌**：宿主程序以动态库（默认）或静态库链接，不需要 HTTP、不需要子进程，
   不依赖 CLI 参数或环境变量即可完成加载与推理。
2. **稳定 ABI**：C 头文件 + 不透明句柄；调用方用 C99/C11 也能使用；结构体按
   `struct_size` 扩展，新增字段不破坏旧二进制。
3. **批量推理**：一次提交多个请求，由引擎自动合并 decode；请求之间状态完全隔离
   （KV slot、采样器、RNG、停止串、取消、输出队列）。
4. **自动 batching 与排队**：请求在“批量窗口”内自动成组；窗口关闭后到达的请求
   进入 FIFO 队列，等 slot 释放后被接纳，不插队、不相互阻塞。
5. **多线程提交**：多个宿主线程可以同时提交请求；模型句柄可共享。流式读取按请求
   隔离，读取慢的请求只回压自己（引擎已有的背压语义），不拖住其他请求。
6. **可运维**：错误码、错误消息、队列/活跃/批大小统计、优雅停机、取消与超时。

### 1.2 非目标（v1）

* **多模态请求**：图像/视频/音频在 v1 不进入 C API。`engine::generate_mm` 走的是
  单序列、slot 0 的路径，和连续批处理不共用同一条 lane；v1.1 会以“独占 lane”
  （同一时刻只跑一个多模态请求）的形式加入，并复用 `mm_build_prompt_*`。模型加载
  仍可忽略 mmproj。
* **投机解码（MTP / DFlash2）**：两者都是单序列、greedy-only 的路径，会把整个
  batch 串行化。v1 不暴露；需要时用 CLI/server 或在 v2 加独立 lane。
* **跨模型共享权重**：每个 `syi_model` 持有自己的权重、KV 池和 SYCL context。
  多个模型可在同一进程共存，但各自付显存，且共享少量进程级设置（见 §11.3）。
* **Windows/macOS**：v1 只承诺 Linux x86_64 + Intel GPU/CPU。ABI 头文件保留
  `SYI_API` 的 Windows 分支，但不测试。
* **HTTP 服务**：C API 不包含 `server.cpp`；宿主自己决定暴露方式。
* **训练/微调/embedding-only 接口**：只做生成。

---

## 2. 设计原则

1. **一个模型一个调度器**。`syi_model` 内部拥有一个 `engine` 和一个
   `scheduler` 线程；所有请求都进同一个调度器，才能真正共享一次权重读取。
2. **不把 C++ 类型带过 ABI**。句柄是不透明指针；字符串是 `(指针, 长度)`；枚举
   固定为 `int`；布尔用 `int32_t`；不用 `std::string`、`std::vector`、`bool`。
3. **调用方拥有调用方的内存，库拥有库的内存**。输入字符串/数组在调用期间复制；
   输出视图的生命周期在文档里逐项写明；库分配的缓冲区用 `syi_free` 释放。
4. **引擎锁内不调用用户代码**。C API 的回调（v1 不提供）和日志函数都不在
   `engine::mtx` 内执行；v1 的流式读取是拉模式，天然避免这个问题。
5. **失败可诊断**。每个失败返回稳定错误码，并在当前线程留下人类可读的消息；异步
   失败在请求上保留 `error` 和 `finish_reason`。
6. **批处理是默认行为，不是附加功能**。`syi_request_submit` 本身就会进入批处理
   队列；`syi_submit_batch` 只是把“同一批”这件事显式化。

---

## 3. 与现有代码的映射

| C API 概念 | 现有实现 | 说明 |
|---|---|---|
| `syi_model` | `si::engine` | 加载 GGUF、上传权重、建 KV 池、前缀缓存、设备/分层 |
| 请求调度 | `si::scheduler` / `si::sequence` | 连续批处理、分块 prefill、批量 decode |
| 采样 | `si::sample_token` / `sampler_state` | 每序列独立 RNG、penalty、logit_bias、logprobs |
| tokenizer | `si::tokenizer` | `encode/decode/token_piece` |
| chat template | `si::render_chat` | GGUF 模板 + 内置 ChatML 回退 |
| 停止串 | 目前只在 `src/server/server.cpp` 的 `stop_filter` | v1 移入 `sequence`/调度器，见 §9.2 |
| 流式输出 | `sequence::out_q` | v1 为 API 请求增加“全量收集”模式，见 §9.3 |
| 多设备 | `engine_config::layer_map` | 原样映射 |
| 前缀缓存 | `engine::pc_*` | 原样映射，请求间共享只读 KV 块 |
| 设备/CPU | `engine_config::device` | 原样映射 |

### 3.1 当前代码里必须补的缺口

这些不是 C API 层的装饰，而是“进程内嵌”与“多线程”暴露出来的真实约束：

1. **KV 类型是进程级的**。`src/backend/gpu/kernels/kv_type.h` 的
   `kv_dtype_set_kv()` 在引擎构造前全局设置，36 处调用点读全局。C API 需要把
   K/V 类型变成 `engine_config` 的一部分，由每个引擎持有并显式传给 kernel
   launcher（`kv_layout {k, v}`）。v1 实现时先做这个重构；否则两个不同 KV 类型的
   模型在同一进程会互相污染。
2. **CPU 线程池是进程单例**。`cpu_thread_count()` 读
   `src/backend/cpu/kernels/common.cpp` 的全局 override。v1 的 `cpu_threads`
   是进程级设置：第一个 CPU 模型生效，后续模型若给不同的值，`syi_model_create`
   返回 `SYI_UNSUPPORTED` 并说明原因（或者按里程碑 M2 把池改为每 backend 一份）。
3. **调度器没有停止串/全量收集**。`sequence` 已经带 `stops` 字段但调度器不读；
   `out_q` 固定 256/512 上限，不适合“先 submit 再统一 wait”的批量用法。v1 给
   `sequence` 增加 `collect_all` 模式和停止串过滤，见 §9。
4. **调度器线程没有异常边界**。`prefill_batch`/`decode_batch` 内的 `throw`
   会直接终止进程。v1 在每个请求的调度步骤外包 `try/catch`，把异常变成该请求的
   `finish_reason=ERROR`，并把消息放进 `syi_request_error()`。
5. **`engine_config` 缺若干进程级/调优字段**：`cpu_threads`、KV 类型
   （`kv_k`/`kv_v`）、`dec_splits`、`prefix_cache` 开关。v1 补上；CLI 仍可用
   `--cpu-threads` / `--kv-type` / `PF_PREFIX_CACHE` 覆盖，行为不变。`n_splits`
   已经在 `engine_config` 里，`dec_splits` 目前由 device profile/env 决定，需要
   变成显式字段。
6. **C API 不应链接 HTTP/OpenSSL**。把核心源文件拆成“引擎核心 + 服务层”两个
   源集合；`sycl_infer_c` 只链接引擎核心、`chat.cpp`、`chat_template.cpp`、
   `scheduler.cpp`，不链接 `server.cpp`、`url_fetch.cpp`、`media_fetch.cpp`、
   `control.cpp`、`request.cpp`、`response.cpp`。这样 wheel 不需要打包/加载
   OpenSSL。

---

## 4. 对象模型与生命周期

```
syi_model (不透明)
├── engine                 模型权重、KV 池、前缀缓存、device/CPU backend
├── scheduler              一个后台线程，连续批处理
├── admission state        批量窗口、FIFO 队列、队列上限
├── tokenizer + chat tmpl  由 engine 持有
└── stats                  原子计数 + 读时加锁快照

syi_request (不透明)
├── prompt (text / token ids / chat messages，已复制)
├── gen_params (已复制)
├── stops / logit_bias (已复制)
├── shared_ptr<sequence>   提交后由 scheduler 持有；请求持弱引用/结果副本
├── output (token 记录 + 完整 text)
├── state / finish_reason / error
└── condition_variable     等待/流式读取
```

### 4.1 创建顺序

```c
syi_model_options opt;
syi_model_options_init(&opt);
opt.model_path = "/path/to/model.gguf";
syi_model *m = NULL;
syi_status st = syi_model_create(&opt, &m);
```

`model_path` 指向 split GGUF 时传第一个 shard（`-00001-of-00004.gguf`），loader
会按现有规则找到其余 shard。

### 4.2 请求状态机

```mermaid
stateDiagram-v2
    [*] --> CREATED: syi_request_create
    CREATED --> QUEUED: syi_request_submit / syi_submit_batch
    QUEUED --> PREFILL: scheduler::admit
    PREFILL --> DECODING: prompt 完成
    DECODING --> FINISHED: EOS / max_tokens / stop
    DECODING --> CANCELLED: cancel / request timeout
    QUEUED --> CANCELLED: cancel / request timeout
    PREFILL --> CANCELLED: cancel
    DECODING --> FAILED: 引擎异常
    PREFILL --> FAILED: 引擎异常
    FINISHED --> [*]: syi_request_destroy
    CANCELLED --> [*]: syi_request_destroy
    FAILED --> [*]: syi_request_destroy
```

* `FINISHED` 表示正常结束（`finish_reason` 为 `STOP` 或 `LENGTH`）。
* `CANCELLED` 包含用户取消和请求级超时（`finish_reason` 为 `CANCELLED` 或
  `TIMEOUT`）。
* `FAILED` 的 `syi_request_error()` 非空。
* 状态查询是只读快照，可能在下一次调度迭代后变化。

### 4.3 销毁顺序

推荐顺序：

```c
syi_request_destroy(req);
syi_model_destroy(model);
```

但库允许先销毁模型：

* `syi_model_destroy(model)` 会停止接收新请求、取消所有未完成请求、等待调度线程
  退出，然后释放引擎。
* 已经结束的请求，其结果（text/token ids）在 `syi_request_destroy` 之前仍然有效；
  它们不再引用模型。
* 未结束的请求会被取消并 finalize；之后 `syi_request_get_result` 返回
  `finish_reason=CANCELLED`，`syi_request_error` 为空。
* 模型销毁后，对旧模型句柄调用任何函数都是未定义行为；请求句柄仍然安全，直到
  被销毁。
* 已 `create` 但尚未 `submit` 的请求不在调度器里：模型销毁后它们不能再次提交
  （`syi_request_submit` 返回 `SYI_SHUTDOWN`），直接 `syi_request_destroy` 即可。

`syi_model_shutdown(model, timeout_ms)` 是优雅版本：停止接收新请求，等待活跃请求
自然结束；`timeout_ms=0` 立即返回，`SYI_TIMEOUT_INFINITE` 一直等到所有请求结束。
超时返回 `SYI_AGAIN`（模型仍处于 stopped，后续可再次调用 shutdown 或直接
destroy）。当前引擎的一次 forward 不可中断，所以超时只能等当前 forward 返回，
不能强杀线程。

---

## 5. 线程模型

### 5.1 谁在什么线程上运行

| 工作 | 线程 | 备注 |
|---|---|---|
| `syi_model_create` | 调用线程 | 加载/上传可能数十秒；可被多个宿主线程并发调用（不同模型） |
| 提交/取消/状态查询 | 任意线程 | 只锁短临界区；不等待 GPU |
| 分块 prefill / 批量 decode | `scheduler` 线程（每模型一个） | 现有 `scheduler::loop` |
| `syi_request_next` / `wait` | 调用线程 | 在请求的 CV 上等待；不持有引擎锁 |
| 日志函数 | 任意线程 | 必须线程安全、快速；不得回调同一模型 |
| `syi_model_destroy` | 调用线程 | 取消、join、释放；可能阻塞一个 forward |

### 5.2 线程安全规则

* **模型句柄**：`syi_model_*` 和 `syi_request_*` 的提交/统计/取消接口可从任意线程
  并发调用；内部用互斥量保护队列和统计。
* **请求句柄的消费端**：`syi_request_next`/`wait`/`get_result` 是**单消费者**；
  同一个请求不要由两个线程同时 `next`。`syi_request_cancel` 可以从任意线程调用。
* **构造/销毁**：`syi_request_destroy` 不得与同一请求的 `next`/`wait` 并发。
* **日志函数**：库可能在 `syi_model_create`、调度线程和提交线程调用；不要在里面
  做阻塞 I/O 或调用同一个模型。
* **Python binding**：在阻塞调用期间释放 GIL；见 Python 文档。

### 5.3 与引擎锁的关系

`scheduler::loop` 在调用 `engine` 时持有 `engine::mtx`，但在 prefill/decode 期间
释放 `scheduler::m`（现有实现，见 `src/server/scheduler.cpp` 的注释）。C API 的
提交路径只进入 `scheduler::submit` 的短锁，不会在 GPU 工作期间阻塞。请求的
`next`/`wait` 在 `sequence` 自己的条件变量上等待，不碰 `engine::mtx`。

---

## 6. 自动 batching 与排队

### 6.1 两级调度

```
调用线程
  │  syi_request_submit / syi_submit_batch
  ▼
┌──────────────────────────────────────────────┐
│ admission（每模型）                          │
│  · 批量窗口 batch_window_us                  │
│  · FIFO waiting 队列                         │
│  · 队列上限 max_queue_depth / max_queue_tokens│
└──────────────────┬───────────────────────────┘
                   │ 成组交给 scheduler::submit
                   ▼
┌──────────────────────────────────────────────┐
│ scheduler（每模型一个线程）                   │
│  · slot 分配（max_slots）                    │
│  · 逐序列分块 prefill（round-robin）          │
│  · 批量 decode（max_batch_rows，默认 kMaxB）  │
│  · 采样、停止、取消、前缀缓存                 │
└──────────────────────────────────────────────┘
```

**批量窗口**解决的是“空闲时突然来一串请求”的冷启动：第一个请求到达时开始计时，
`batch_window_us` 内到达的请求进入同一波（最多 `max_slots` 个或直到
`syi_submit_batch` 显式关窗）。窗口关闭后到达的请求进入 FIFO 队列，等下一次
slot 释放；不会因为“来晚了”而被插到已有请求前面。

具体算法：

1. `syi_request_submit`：tokenize/校验，进入 waiting；如果模型空闲且没有打开的
   窗口，开始计时。
2. 窗口在 `batch_window_us` 到期、达到 `max_slots` 个请求、或
   `syi_submit_batch` 显式关窗时结束。
3. 调度器按 FIFO 顺序接纳 waiting 中能拿到 slot 的请求；拿不到 slot 的留在
   队列头部（不跳过后面的请求，也不丢弃）。
4. 每一轮 decode 把当前所有 ready 的活跃序列合成一个 batch（上限
   `max_batch_rows`）；prefill 逐序列 round-robin 推进。
5. 窗口关闭后才到的请求从第 3 步开始排队；如果队列已满，提交返回
   `SYI_QUEUE_FULL`，不会静默丢弃。

**窗口只在模型空闲时等待**。如果已有活跃请求，新请求直接进入 waiting 队列，由
调度器在下一个迭代接纳；不再额外等一个窗口，避免给本已繁忙的系统增加延迟。

### 6.2 为什么 prefill 不是“一次 fused batch”

当前引擎的 `prefill_batch` 是**一个序列的多个 chunk**（mode 2），不是多个序列的
prompt 拼成一个 batch；跨序列 prefill batching 在 `AGENTS.md` 里明确记录为
“已实现但会产生垃圾输出”，因此没有启用。C API 不改变这个事实：

* **decode** 是真正融合的：一个 `decode_batch` 处理最多 `max_batch_rows` 行；
* **prefill** 是逐序列的：调度器每轮给一个活跃序列推进一个 chunk，round-robin；
* 因此“自动 batch”首先体现在 decode 的权重摊销上，prefill 只做到公平交错。

如果未来的跨序列 prefill batching 修好，C API 不需要改 ABI，只需在
`scheduler` 内部替换 prefill 策略。

### 6.3 队列、上限与超时

| 参数 | 默认 | 语义 |
|---|---:|---|
| `batch_window_us` | 2000 | 空闲时等待更多请求的窗口；0 = 不等待 |
| `max_batch_rows` | 0（= `kMaxB`） | 单次 decode 的最大行数；1 = 严格串行 |
| `max_slots` | 16 | 并发序列数，也是 recurrent state 的分配量 |
| `max_queue_depth` | 1024 | waiting 队列的请求数上限；超过返回 `SYI_QUEUE_FULL` |
| `max_queue_tokens` | 4,194,304 | waiting 队列的 prompt token 总数上限 |
| `request_timeout_ms` | 0 | 请求级总超时；0 = 不限。超时在下一个调度迭代生效 |

* **`max_slots` 是硬容量**：超过它的请求留在 waiting 队列，不会分配 slot/KV。
* **队列满时拒绝而不是无限增长**：`syi_request_submit` 返回 `SYI_QUEUE_FULL`，
  请求仍处于 `CREATED`，调用方可以稍后重试或销毁。`syi_submit_batch` 是原子的：
  如果放不下整批，整批都不提交。
* **超时**：`syi_request_next`/`wait` 的 `timeout_ms` 是调用方等待超时，不影响
  请求；`syi_generation_options.timeout_ms` 是请求级预算，超时后调度器在下一个可
  检查点取消该请求，`finish_reason=TIMEOUT`。
* **公平性**：waiting 是 FIFO；decode 对所有活跃序列每步各推进一步，不使用
  优先级。v1 不提供优先级，避免“高优先级请求饿死低优先级”的运维问题。

### 6.4 请求隔离

每个请求拥有：

* **独立 slot**：`engine::step_info::slot` / KV block table / GDN+conv 状态；
* **独立 `sampler_state` 与 seed**：penalty 历史、RNG 不共享；
* **独立 `gen_params`**：temperature/top-k/top-p/penalty/logit_bias/logprobs；
* **独立停止串集合与字节回退缓冲**；
* **独立输出队列与结果**；
* **独立取消标志**。

共享的只有：

* **权重、backend、KV 池**：只读/按 slot 分区；
* **前缀缓存的只读 KV 块**：`pc_node.refcount` 管理，多个请求命中同一前缀时只共享
  已经不可变的块；新写入的 token 落在各自的新块上；
* **GPU 队列**：in-order 队列按调度顺序执行，但请求之间没有数据依赖。

**数值上的“不干扰”有边界**：并发批处理会改变 GEMM 的 fp 累加顺序，greedy 输出
在 near-tie 上可能和单请求运行不同（`AGENTS.md` 已记录这一现象）。C API 保证的是
**状态隔离**，不是逐 bit 相同。需要严格复现的单请求可以：

```c
opt.max_batch_rows = 1;          /* decode 一次一行 */
opt.batch_window_us = 0;         /* 不等待 */
opt.prefix_cache = 0;            /* 不走前缀缓存 */
```

即使这样，GPU 上的 kernel 仍可能因驱动/编译版本不同产生细微差异。

### 6.5 背压

现有 scheduler 对 `out_q` 满的序列做“暂停该序列、不阻塞整个 loop”的背压。C API
的请求使用“全量收集”模式（§9.3），输出受 `max_tokens` 限制，不会因为调用方
暂时不读而丢 token；但 `max_tokens` 很大且 `top_logprobs` 很高时内存会增长，
文档要求在批量场景显式设置上限。慢消费者不会拖住其他请求。

---

## 7. ABI、版本与错误模型

### 7.1 版本

```c
#define SYI_VERSION_MAJOR 0
#define SYI_VERSION_MINOR 1
#define SYI_VERSION_PATCH 0
#define SYI_ABI_VERSION 1
```

* `syi_abi_version()` 返回运行时的 ABI 号；头文件里的 `SYI_ABI_VERSION` 是编译期
  期望值。`syi_model_create` 检查 `opts->abi_version`，不匹配返回
  `SYI_INVALID_ARGUMENT`。
* 语义化版本：新增函数/字段是 minor；删除/改变语义/布局是 major。ABI major
  不变时，旧二进制可以在新库上运行。
* 结构体扩展规则：**只允许在末尾追加字段**；每个输入/输出结构体第一个字段是
  `size_t struct_size`。库只读写 `min(caller_size, library_size)` 范围内的字节；
  调用方把 `struct_size` 设为自身编译时的 `sizeof`。输出结构体由库把
  `struct_size` 更新为实际写入的大小。

### 7.2 错误码

```c
typedef enum syi_status {
    SYI_OK = 0,
    SYI_END = 1,          /* next: 流结束（正常/取消/错误，见 result） */
    SYI_AGAIN = 2,        /* next/wait: 本次等待超时，请求仍可能继续 */
    SYI_INVALID_ARGUMENT = -1,
    SYI_INVALID_STATE = -2,
    SYI_UNSUPPORTED = -3,
    SYI_OUT_OF_MEMORY = -4,
    SYI_BACKEND = -5,     /* SYCL / oneDNN / 设备错误 */
    SYI_MODEL_LOAD = -6,  /* GGUF / 架构 / 几何校验失败 */
    SYI_IO = -7,
    SYI_INTERNAL = -8,
    SYI_QUEUE_FULL = -9,
    SYI_SHUTDOWN = -10,   /* 模型已 shutdown，不再接收提交 */
    SYI_CANCELLED = -11   /* 对已取消请求的提交/等待 */
} syi_status;
```

* 每个失败在当前线程写一条 `syi_last_error()` 消息（UTF-8，NUL 结尾，指针在
  下一次同线程 API 调用前有效）。成功不清空上一条消息；需要时调用
  `syi_clear_error()`。
* 异步失败不通过 `syi_status` 返回，而是落在请求上：
  `syi_request_state()==SYI_REQUEST_FAILED`，`syi_request_get_result().finish_reason
  ==SYI_FINISH_ERROR`，`syi_request_error()` 给原因。
* `SYI_END`/`SYI_AGAIN` 不是失败；不要用 `syi_last_error` 判断。
* 所有 `timeout_ms` 参数：`0` = 非阻塞轮询；`SYI_TIMEOUT_INFINITE` = 无限等待；
  其他值 = 毫秒。`SYI_CANCELLED` 只在“对已取消请求执行 submit/wait 以外的
  操作”这类同步路径上出现；终态请求的 `next`/`wait` 仍返回 `SYI_END`/`SYI_OK`，
  由 `finish_reason` 区分取消/超时。

### 7.3 内存与字符串

| 数据 | 谁分配 | 生命周期 |
|---|---|---|
| 输入 `model_path`/`layer_map`/`pc_dir`/`draft_path` | 调用方 | `syi_model_create` 内复制 |
| 输入 prompt/text/token ids/stop/logit_bias | 调用方 | `set_*` 内复制 |
| `syi_model_info` 里的字符串 | 库 | 模型存活期间 |
| `syi_token.text` / `.top` | 库 | 下一次 `syi_request_next` 或 `syi_request_destroy` 之前 |
| `syi_result.text` / `.token_ids` / `error` | 库（请求持有） | `syi_request_destroy` 之前 |
| `syi_model_tokenize`/`detokenize`/`render_chat` 的输出 | 库 | 调用方用 `syi_free` 释放 |
| `syi_free` | — | 只能释放库返回的指针 |

字符串统一是 UTF-8 字节序列；`syi_string.size` 不含结尾 NUL。库保证返回的
`syi_string.data` 非空指针（空字符串指向 `""`），并在 `size` 之后有一个 NUL
（方便 C 用户），但二进制内容以 `size` 为准。`syi_model_tokenize` 在 `n==0` 时
可以把 `*out_ids` 设为 NULL；`syi_free(NULL)` 是安全的。

### 7.4 符号与库

* 默认构建 `libsycl_infer.so.1`（SONAME），导出符号仅 `syi_*`；C++ 符号全部
  hidden（`-fvisibility=hidden` + version script）。
* 静态库 `libsycl_infer.a` 可选（`SI_C_API_STATIC=ON`），用于不方便带 DSO 的
  产品；静态库不保证跨编译器 ABI，只保证同一工具链。
* `pkg-config`：`sycl_infer.pc` 提供 `-I${includedir} -L${libdir} -lsycl_infer`。
* CMake：`find_package(sycl_infer CONFIG REQUIRED)`，target
  `sycl_infer::c`。
* 头文件只依赖 `<stddef.h>`、`<stdint.h>`，可被 C99/C11/C++17 包含。

---

## 8. C API 参考

以下是 v1 的完整头文件形状（实现时放在
`include/sycl_infer/sycl_infer.h`；本节是规范，不是已生成的文件）。

### 8.1 基础类型与版本

```c
#ifndef SYCL_INFER_H
#define SYCL_INFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SYI_VERSION_MAJOR 0
#define SYI_VERSION_MINOR 1
#define SYI_VERSION_PATCH 0
#define SYI_ABI_VERSION 1

/* 传给 next/wait/shutdown 的“无限等待”；0 表示非阻塞轮询。 */
#define SYI_TIMEOUT_INFINITE 0xFFFFFFFFu

#if defined(_WIN32)
#  if defined(SYI_BUILD)
#    define SYI_API __declspec(dllexport)
#  else
#    define SYI_API __declspec(dllimport)
#  endif
#else
#  define SYI_API __attribute__((visibility("default")))
#endif

typedef struct syi_model   syi_model;
typedef struct syi_request syi_request;

typedef struct syi_string {
    const char *data;   /* UTF-8, 不保证调用方可以修改 */
    size_t      size;   /* 不含结尾 NUL */
} syi_string;

typedef enum syi_status {
    SYI_OK = 0,
    SYI_END = 1,
    SYI_AGAIN = 2,
    SYI_INVALID_ARGUMENT = -1,
    SYI_INVALID_STATE = -2,
    SYI_UNSUPPORTED = -3,
    SYI_OUT_OF_MEMORY = -4,
    SYI_BACKEND = -5,
    SYI_MODEL_LOAD = -6,
    SYI_IO = -7,
    SYI_INTERNAL = -8,
    SYI_QUEUE_FULL = -9,
    SYI_SHUTDOWN = -10,
    SYI_CANCELLED = -11
} syi_status;

SYI_API uint32_t     syi_abi_version(void);
SYI_API const char  *syi_version_string(void);
SYI_API const char  *syi_status_string(syi_status status);
SYI_API const char  *syi_last_error(void);   /* thread-local */
SYI_API void         syi_clear_error(void);
SYI_API void         syi_free(void *p);      /* NULL 安全 */
```

### 8.2 模型配置

```c
typedef enum syi_device {
    SYI_DEVICE_AUTO = 0,
    SYI_DEVICE_GPU  = 1,
    SYI_DEVICE_CPU  = 2
} syi_device;

typedef enum syi_kv_type {
    SYI_KV_AUTO = 0,   /* 引擎默认：i8 */
    SYI_KV_I8   = 1,
    SYI_KV_I4   = 2,
    SYI_KV_F16  = 3,
    SYI_KV_BF16 = 4,
    SYI_KV_F32  = 5
} syi_kv_type;

typedef void (*syi_log_fn)(int32_t level, const char *message, void *user_data);

typedef struct syi_model_options {
    size_t      struct_size;       /* = sizeof(syi_model_options) */
    uint32_t    abi_version;       /* = SYI_ABI_VERSION */

    const char *model_path;        /* 必填；split GGUF 传第一个 shard */
    const char *layer_map;         /* 可空；"0-31:gpu.0,32-63:gpu.1" */

    syi_device  device;            /* AUTO/GPU/CPU */
    int32_t     max_seq;           /* 0 = 引擎默认 8192；-1 = 模型训练上下文；>0 显式 */
    int32_t     max_slots;         /* 0 = 16；上限 kMaxB=32 */
    int32_t     n_blocks;          /* 0 = 自动 */
    int32_t     kv_cap_mb;         /* -1 = 自动；0 = 不设上限；>0 = MB */
    syi_kv_type kv_type;           /* K 类型 */
    syi_kv_type kv_type_v;         /* V 类型；AUTO = 与 K 相同 */

    int32_t     cpu_threads;       /* 0 = 自动；CPU backend 的 worker 数（进程级） */
    int32_t     n_splits;          /* 0 = 自动；prefill attention K-split 上限 */
    int32_t     dec_splits;        /* 0 = 自动；decode attention K-split */

    /* 批处理/排队 */
    int32_t     batch_window_us;   /* 0 = 不等待；默认 2000 */
    int32_t     max_batch_rows;    /* 0 = kMaxB；1 = 严格串行 */
    int32_t     max_queue_depth;   /* 0 = 无限制；默认 1024 */
    int64_t     max_queue_tokens;  /* 0 = 无限制；默认 4,194,304 tokens */
    int32_t     request_timeout_ms;/* 0 = 不限 */

    /* 前缀缓存 */
    int32_t     prefix_cache;      /* -1 = 默认；0 = 关；1 = 开 */
    const char *pc_dir;            /* 非空 = 启用磁盘层 */
    int32_t     pc_vram_mb;        /* -1 = 默认 */
    int32_t     pc_ram_mb;         /* -1 = 默认 512；0 = 关 */
    int32_t     pc_disk_mb;        /* -1 = 默认 1024；0 = 不限 */

    /* MoE / qwen4exp */
    int32_t     moe_resident;      /* 0 = 全部常驻（默认） */
    int64_t     moe_vram_mb;
    int64_t     moe_ram_mb;
    int64_t     engram_ram_mb;     /* -1 = 默认 4096；0 = 关 */

    /* YaRN */
    int32_t     yarn;              /* 0/1 */
    int32_t     yarn_orig_ctx;
    float       yarn_factor;
    float       yarn_beta_fast;
    float       yarn_beta_slow;
    float       yarn_attn_factor;

    /* 其他 */
    int32_t     head_split;
    uint64_t    flags;             /* 预留，0 */
    syi_log_fn  log_fn;            /* 可空；默认静默 */
    void       *log_user_data;

    void       *reserved[8];       /* 必须为 NULL；为将来扩展保留 */
} syi_model_options;

SYI_API void        syi_model_options_init(syi_model_options *options);
SYI_API syi_status  syi_model_create(const syi_model_options *options, syi_model **out_model);
SYI_API void        syi_model_destroy(syi_model *model);   /* NULL 安全 */
SYI_API syi_status  syi_model_shutdown(syi_model *model, uint32_t timeout_ms);
```

字段语义说明：

* `max_seq`：`0` 用引擎默认（8192）；`-1` 用模型 `context_length`（可能很大，
  需要相应 KV 容量）；`>0` 显式。超过模型的训练上下文仍可运行，但通常应配合
  YaRN；C API 不替用户打开 YaRN。
* `kv_type_v`：只有 `i4`/`i8` 的比例型可以混用；其他组合返回
  `SYI_INVALID_ARGUMENT`。
* `max_slots` 同时决定 recurrent state 的显存；降低它是长上下文最便宜的
  headroom（见 `AGENTS.md` 的容量表）。取值 1..`kMaxB`（32），超出返回
  `SYI_INVALID_ARGUMENT`。
* `max_batch_rows`：0 表示用引擎的 `kMaxB` 上限；1..`kMaxB` 为显式上限。它只
  限制 decode 行数，不改变 `max_slots` 的容量。
* `max_queue_depth`/`max_queue_tokens`：0 表示不限制；队列满时提交返回
  `SYI_QUEUE_FULL`。
* `n_splits`/`dec_splits` 是高级调优；一般留 0，由 device profile 决定。
* `head_split` 对应 `--head-split`；默认 0。
* `pc_*` 的 `-1` 表示“使用 `PF_PC_*` 或引擎默认”，C API 不读环境变量也能工作。
* `log_fn` 的 level：0=error，1=warn，2=info，3=debug。默认不输出到 stderr；
  宿主需要时自己接日志。

### 8.3 模型信息与统计

```c
typedef struct syi_model_info {
    size_t      struct_size;       /* in/out */
    syi_string  model_name;
    syi_string  architecture;
    syi_string  device_name;       /* e.g. "gpu:0,1" 或 "cpu" */
    int32_t     n_vocab;
    int32_t     n_ctx;             /* 模型声明的训练上下文 */
    int32_t     n_embd;
    int32_t     n_layer;
    int32_t     n_head;
    int32_t     n_head_kv;
    int32_t     head_dim;
    int32_t     is_moe;
    int32_t     is_split_gguf;
    int32_t     max_seq;           /* 实际配置 */
    int32_t     max_slots;
    syi_kv_type kv_type;
    syi_kv_type kv_type_v;
    syi_device  device;
} syi_model_info;

typedef struct syi_model_stats {
    size_t   struct_size;          /* in/out */
    uint64_t submitted;
    uint64_t admitted;
    uint64_t completed;
    uint64_t cancelled;
    uint64_t failed;
    uint64_t generated_tokens;
    uint64_t prefill_steps;
    uint64_t decode_steps;
    double   avg_decode_batch;     /* decode_steps > 0 */
    uint32_t max_decode_batch;
    int32_t  queued;               /* waiting 队列 */
    int32_t  active;               /* 已占 slot */
    int32_t  free_slots;           /* max_slots - active */
} syi_model_stats;

SYI_API void       syi_model_info_init(syi_model_info *info);
SYI_API void       syi_model_stats_init(syi_model_stats *stats);
SYI_API syi_status syi_model_get_info(const syi_model *model, syi_model_info *info);
SYI_API syi_status syi_model_get_stats(const syi_model *model, syi_model_stats *stats);
```

`info`/`stats` 的调用方先调用 `*_init`（或自行设置 `struct_size = sizeof(...)`）；
库只写这个范围内。

### 8.4 Tokenizer 与 chat template

```c
SYI_API syi_status syi_model_tokenize(syi_model *model,
                                      const char *text, size_t text_len,
                                      int32_t parse_special,
                                      int32_t **out_ids, size_t *out_n);

SYI_API syi_status syi_model_detokenize(syi_model *model,
                                        const int32_t *ids, size_t n,
                                        char **out_text, size_t *out_len);

typedef struct syi_chat_message {
    size_t      struct_size;       /* = sizeof(syi_chat_message) */
    const char *role;              /* "system" / "user" / "assistant" / "tool" */
    const char *content;           /* UTF-8；可空（纯工具调用） */
    const char *name;              /* 可空 */
    const char *tool_call_id;      /* role=tool 时使用 */
    const char *reasoning_content; /* assistant 可选 */
} syi_chat_message;

SYI_API void syi_chat_message_init(syi_chat_message *message);

SYI_API syi_status syi_model_render_chat(syi_model *model,
                                         const syi_chat_message *messages, size_t n_messages,
                                         int32_t enable_thinking,
                                         const char *tools_json, /* 可空 */
                                         char **out_text, size_t *out_len);
```

`tokenize`/`detokenize`/`render_chat` 的输出由库分配，调用方用 `syi_free` 释放。
`tokenize` 的 `parse_special=0` 用于把特殊 token 当普通文本；默认聊天输入用 1。

### 8.5 生成参数

```c
typedef struct syi_generation_options {
    size_t   struct_size;          /* = sizeof(syi_generation_options) */
    uint32_t abi_version;          /* = SYI_ABI_VERSION */

    int32_t  max_tokens;           /* 默认 256；必须 >= 1 */
    int32_t  ignore_eos;           /* 0/1 */
    float    temperature;          /* 默认 0.7 */
    float    top_p;                /* 默认 0.95 */
    int32_t  top_k;                /* 默认 40；1 = greedy */
    float    min_p;                /* 默认 0 */
    float    repeat_penalty;       /* 默认 1.0 */
    int32_t  repeat_last_n;        /* 默认 64 */
    float    presence_penalty;     /* 默认 0 */
    float    frequency_penalty;    /* 默认 0 */
    uint64_t seed;                 /* 0 = 随机；非 0 = 每请求确定性 */

    int32_t  logprobs;             /* 0/1 */
    int32_t  top_logprobs;         /* 0..20 */
    int32_t  timeout_ms;           /* 0 = 用模型默认；-1 = 不限 */

    const int32_t *logit_bias_ids; /* 可为空 */
    const float   *logit_bias_values;
    size_t         n_logit_bias;

    const char *const *stop;       /* 停止串数组；可为空 */
    size_t             n_stop;     /* 上限 32；单个串上限 256 字节 */

    uint32_t flags;                /* 预留，0 */
    void    *reserved[8];
} syi_generation_options;

SYI_API void syi_generation_options_init(syi_generation_options *options);
```

默认值与 CLI 对齐（`--temp 0.7 --top-p 0.95 --top-k 40 --max-tokens 256`）。
`max_tokens < 1`、`top_logprobs < 0`、超过 20 的 `top_logprobs`、负的 `top_k` /
`repeat_last_n` 返回 `SYI_INVALID_ARGUMENT`；`stop` 超过 32 个或单串超过 256 字节
同样拒绝。
`temperature<=0` 或 `top_k==1` 表示 greedy；`seed` 非 0 时每请求独立 RNG，
不会被同批其他请求影响。

### 8.6 请求

```c
typedef enum syi_finish_reason {
    SYI_FINISH_NONE = 0,      /* 仍在生成 */
    SYI_FINISH_STOP = 1,      /* EOS 或停止串 */
    SYI_FINISH_LENGTH = 2,    /* max_tokens 或上下文上限 */
    SYI_FINISH_CANCELLED = 3, /* syi_request_cancel */
    SYI_FINISH_TIMEOUT = 4,   /* 请求级 timeout_ms */
    SYI_FINISH_ERROR = 5      /* 引擎/后端错误 */
} syi_finish_reason;

typedef enum syi_request_state {
    SYI_REQUEST_CREATED = 0,
    SYI_REQUEST_QUEUED = 1,
    SYI_REQUEST_PREFILL = 2,
    SYI_REQUEST_DECODING = 3,
    SYI_REQUEST_FINISHED = 4,
    SYI_REQUEST_FAILED = 5,
    SYI_REQUEST_CANCELLED = 6
} syi_request_state;

typedef struct syi_logprob {
    int32_t    id;
    float      logprob;
    syi_string text;
} syi_logprob;

typedef struct syi_token {
    size_t             struct_size;  /* in/out */
    int32_t            id;
    uint32_t           flags;        /* 预留，0 */
    float              logprob;      /* logprobs=0 时无意义 */
    syi_string         text;         /* 完整 UTF-8 piece；可能为空 */
    const syi_logprob *top;          /* 可为空；生命周期同 text */
    size_t             n_top;
    uint64_t           index;        /* 0-based completion token 序号 */
} syi_token;

typedef struct syi_result {
    size_t            struct_size;   /* in/out */
    syi_string        text;          /* 已过滤停止串的完整文本 */
    const int32_t    *token_ids;     /* 请求持有，直到 destroy */
    size_t            n_tokens;      /* 采样 token 数（含 EOS） */
    syi_finish_reason finish_reason;
    uint64_t          prompt_tokens;
    uint64_t          completion_tokens;
    uint64_t          cached_tokens; /* 前缀缓存命中 */
    double            ttft_ms;       /* submit -> 第一个 token */
    double            total_ms;      /* submit -> 终态 */
    syi_string        error;         /* 成功时 size == 0 */
} syi_result;

SYI_API void syi_token_init(syi_token *token);
SYI_API void syi_result_init(syi_result *result);

SYI_API syi_status syi_request_create(syi_model *model,
                                      const syi_generation_options *options,
                                      syi_request **out_request);
SYI_API void       syi_request_destroy(syi_request *request); /* NULL 安全 */

SYI_API syi_status syi_request_set_text(syi_request *request,
                                        const char *text, size_t text_len);
SYI_API syi_status syi_request_set_tokens(syi_request *request,
                                          const int32_t *ids, size_t n);
SYI_API syi_status syi_request_set_chat(syi_request *request,
                                        const syi_chat_message *messages, size_t n_messages,
                                        int32_t enable_thinking,
                                        const char *tools_json);

SYI_API syi_status syi_request_submit(syi_request *request);
SYI_API syi_status syi_submit_batch(syi_model *model,
                                    syi_request *const *requests, size_t n);
SYI_API syi_status syi_wait_batch(syi_request *const *requests, size_t n,
                                  uint32_t timeout_ms);

SYI_API syi_status syi_request_get_state(const syi_request *request,
                                         syi_request_state *state);
SYI_API syi_status syi_request_next(syi_request *request, syi_token *out,
                                    uint32_t timeout_ms);
SYI_API syi_status syi_request_wait(syi_request *request, uint32_t timeout_ms);
SYI_API syi_status syi_request_cancel(syi_request *request);
SYI_API syi_status syi_request_get_result(const syi_request *request,
                                          syi_result *out);
SYI_API const char *syi_request_error(const syi_request *request);
```

#### 语义细节

* `syi_request_create` 只创建对象，不提交；此时可以多次 `set_text`/`set_tokens`/
  `set_chat`，最后一次生效。提交后这些 setter 返回 `SYI_INVALID_STATE`。
* `syi_request_set_chat` 立即用模型的 chat template 渲染成文本并 tokenize；模板
  失败（例如模板语法不支持）返回 `SYI_MODEL_LOAD`/`SYI_INTERNAL`，不会提交。
  也可以用 `syi_model_render_chat` 先渲染、再 `set_text`。
* `syi_request_submit`：tokenize（如果需要）、校验 prompt 长度、入队；返回时请求
  可能仍在 `QUEUED`。失败时请求保持 `CREATED`（`SYI_QUEUE_FULL`）或进入
  `FAILED`（不可恢复的内部错误）。
* `syi_submit_batch`：所有请求必须属于同一个模型、都处于 `CREATED`；作为一个
  原子批次入队（窗口立即关闭）。任何一个不满足则整批不提交。
* `syi_request_next`：
  * `SYI_OK` + 一个 token；
  * `SYI_END`：请求已终态（正常、取消、超时、错误）；用
    `syi_request_get_result` 区分；
  * `SYI_AGAIN`：本次等待超时，请求仍可能继续；
  * 负值：调用参数错误等同步错误。
  * `timeout_ms==0` 表示不等待（非阻塞轮询）。
* `syi_request_wait`：等待到终态；`SYI_OK`（终态）、`SYI_AGAIN`（超时）或错误。
  不消费 token，所以可以和 `next` 混用；但单消费者规则仍然适用。
* `syi_request_cancel`：幂等；对 `CREATED` 直接标记取消，对 `QUEUED` 在下一个
  调度迭代移除，对 `DECODING` 在当前步结束后停止。对终态请求返回 `SYI_OK`。
* `syi_request_get_result`：仅在终态可调用；未完成时返回 `SYI_INVALID_STATE`。
  `text`/`token_ids` 在 destroy 前有效且不再变化。流式读取用 `next`。
* `syi_request_error`：异步错误的 UTF-8 消息；没有错误时返回 `""`；指针在
  `syi_request_destroy` 前有效。
* `syi_request_destroy`：释放请求；如果仍在运行，等价于 `cancel` 后等待终态。
  不得与 `next`/`wait` 并发。

### 8.7 C++ RAII 包装（头文件，非 ABI）

`include/sycl_infer/sycl_infer.hpp` 提供可选的 header-only 包装：

```cpp
namespace sycl_infer {

class Error : public std::runtime_error { /* ... */ };
class Model;
class Request;

class GenerationOptions {
public:
    int max_tokens = 256;
    double temperature = 0.7;
    double top_p = 0.95;
    int top_k = 40;
    std::uint64_t seed = 0;
    std::vector<std::string> stop;
    // ...
};

class Model {
public:
    explicit Model(const ModelOptions&);
    Model(Model&&) noexcept;
    ~Model();
    Request submit(std::string_view prompt, const GenerationOptions& = {});
    std::vector<Request> submit_batch(const std::vector<std::string>& prompts,
                                      const GenerationOptions& = {});
    Generation generate(std::string_view prompt, const GenerationOptions& = {});
    std::vector<Generation> generate_batch(const std::vector<std::string>& prompts,
                                           const GenerationOptions& = {});
    ModelInfo info() const;
    ModelStats stats() const;
    void shutdown(std::chrono::milliseconds timeout = {});
};

class Request {
public:
    bool next(Token&, std::chrono::milliseconds timeout = {});
    Generation wait(std::chrono::milliseconds timeout = {});
    void cancel();
    // 迭代器：for (auto tok : request) ...
};

} // namespace sycl_infer
```

包装层只做异常转换和 RAII，不改变语义；C 用户和 C++ 用户看到的是同一个库。

---

## 9. 实现设计

### 9.1 新增目录与构建目标

```
include/sycl_infer/sycl_infer.h        C ABI
include/sycl_infer/sycl_infer.hpp     可选 C++ RAII 包装
src/capi/model.cpp                    create/destroy/info/tokenize/chat
src/capi/request.cpp                  request 状态机、next/wait/result
src/capi/scheduler_bridge.cpp         提交、窗口、FIFO、队列上限
src/capi/errors.cpp                   状态码/线程本地错误
src/capi/stats.cpp                    统计
cmake/sycl_infer.map                  只导出 syi_*
cmake/sycl_inferConfig.cmake.in
```

CMake：

```cmake
option(SI_C_API "build libsycl_infer (C ABI)" ON)
option(SI_C_API_STATIC "also build libsycl_infer.a" OFF)

if(SI_C_API AND NOT SI_ROUTER_ONLY)
    add_library(sycl_infer_c SHARED ${SI_CAPI_SOURCES})
    target_link_libraries(sycl_infer_c PRIVATE sycl_infer_capi_core)
    set_target_properties(sycl_infer_c PROPERTIES
        OUTPUT_NAME sycl_infer
        VERSION 1.0.0
        SOVERSION 1
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON)
    target_link_options(sycl_infer_c PRIVATE
        -Wl,--version-script=${CMAKE_SOURCE_DIR}/cmake/sycl_infer.map)
    install(TARGETS sycl_infer_c EXPORT sycl_inferTargets ...)
    install(FILES include/sycl_infer/sycl_infer.h include/sycl_infer/sycl_infer.hpp
            DESTINATION include/sycl_infer)
endif()
```

`sycl_infer_capi_core` 是 `sycl_infer_core` 的源集合减去 HTTP 服务文件；两者共享
同一个源列表变量，避免两份文件列表漂移。

### 9.2 停止串与输出收集

在 `sequence` 上增加：

```cpp
struct sequence {
    // ...
    bool collect_all = false;        // C API 请求：保留全部 token
    std::vector<token_out> tokens;   // collect_all 时使用
    size_t read_pos = 0;
    std::string text;                // 已过滤停止串的完整文本
    std::string error;               // 异步错误
    uint64_t deadline_ms = 0;        // 0 = 无请求级超时
};
```

`sample_and_emit` 之后、`push` 之前调用 `sequence::stop_filter(piece)`：

* 保留最多 `max_stop_len - 1` 字节，避免停止串跨 token 时被提前发出；
* 命中停止串时只发出其前缀，设置 `stop_hit`，并让 `sample_outcome.done=true`
  （`finish_reason=stop`）；
* EOS/length 结束时 flush 保留字节；
* `collect_all` 模式下把每个 `token_out` 追加到 `tokens`，同时追加到 `text`；
  `push_token` 不再受 256/512 上限约束（上限由 `max_tokens`/`max_seq` 决定）。

服务器路径保持 `collect_all=false`，继续用现有有界 `out_q` 和 `stop_filter`；
等 C API 稳定后可以把服务器的过滤器也合并到 `sequence`，但不是 v1 的前置条件。

### 9.3 异常边界

`scheduler::loop` 的每个请求步骤包在：

```cpp
try {
    // prefill / decode / fetch_logits / sample
} catch (const std::exception &e) {
    s->error = e.what();
    retire(s, "error"); // retire 内部设置 finish_reason/状态
}
```

`retire` 必须保证 slot、KV block、prefix cache 引用都释放；异常路径和正常路径
走同一份清理代码，避免“错误请求泄漏 slot”。

### 9.4 请求级超时与取消

* `admit` 前检查 `deadline_ms`；过期直接 retire（`TIMEOUT`），不占 slot。
* 活跃请求在每一轮 decode/prefill 之间检查 `cancelled`/deadline；引擎的一次
  forward 不可中断，所以最多多跑一个 forward。
* `syi_request_cancel` 只设置 atomic 标志并 notify；清理由调度线程完成。

### 9.5 统计

* `scheduler` 维护原子计数：`submitted/admitted/completed/cancelled/failed`、
  `prefill_steps`、`decode_steps`、`sum_decode_rows`、`max_decode_rows`。
* `syi_model_get_stats` 在锁内取快照，`avg_decode_batch = sum/decode_steps`。
* `free_slots = max_slots - active`（active 由调度器维护的计数给出，不能靠遍历
  requesting 列表）。

### 9.6 队列上限与批量窗口

`syi_model` 的 admission 状态：

```cpp
struct admission {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::shared_ptr<request_impl>> waiting;
    size_t queued_tokens = 0;
    std::chrono::steady_clock::time_point window_start;
    bool window_open = false;
    // ...
};
```

* `submit()`：锁内检查队列上限，push，notify。如果模型空闲且窗口未开，打开窗口。
* 调度线程或独立的 admission 线程在窗口到期/满员时把一批请求交给
  `scheduler::submit`。v1 推荐**在 scheduler 线程内**做窗口等待（不新增线程）：
  `loop()` 在 `active.empty() && !waiting.empty()` 时 `cv.wait_for(window)`，然后
  一次性接纳；忙时不等。
* `syi_submit_batch`：锁内一次性 push 并关闭窗口，避免等待。
* 队列 token 计数在 submit 时加、在 admit/retire 时减；避免长时间的锁竞争。

**为什么窗口在 scheduler 线程内**：少一个线程、少一次上下文切换；窗口只影响
冷启动，`wait_for` 不持有 `engine::mtx`。服务器路径把 `batch_window_us=0`，行为
不变。

### 9.7 与现有 CLI/server 的关系

* CLI/server 继续链接 `sycl_infer_core`，不经过 C API。
* C API 与 CLI 共享 `engine`/`scheduler`，但使用不同的 `scheduler` 配置
  （窗口、收集模式、异常边界、统计）。
* 这样不会把“嵌入式 API 的稳定性”绑到 HTTP 服务的字段变化上，也不会为了 C API
  重写一条推理路径。

---

## 10. 确定性、隔离与已知边界

1. **状态隔离是保证，逐 bit 相同不是**。并发 batch 改变 GEMM 累加顺序，near-tie
   的 greedy argmax 可能翻转。需要单请求逐 bit 复现时，按 §6.4 设置
   `max_batch_rows=1`、`batch_window_us=0`、`prefix_cache=0`，并固定 seed/版本。
2. **前缀缓存共享只读块**。命中同一前缀的请求共享 `pc_node` 的 KV 块
   （refcount），新 token 写入各自的新块；不会把 A 的生成写进 B 的 KV。
3. **`qwen4exp` 的 decode 默认一行**。引擎在该模型上实测 2 行 0.85x，调度器
   强制 `dec_cap=1` 并串行化序列；C API 的 `max_batch_rows` 不会覆盖这个模型
   策略。统计里的 `avg_decode_batch` 会是 1。
4. **CPU backend 没有 GPU 并发**。`max_batch_rows` 对 CPU 仍然生效，但 CPU
   kernel 使用进程级线程池；`cpu_threads` 是进程级设置。
5. **多设备 `layer_map`**：每个请求仍需要 KV slot；每卡都有自己的池。前缀缓存
   和 batch 行为与 CLI 一致。
6. **一次 forward 不可中断**：cancel/timeout/shutdown 的响应延迟上限是一次
   prefill chunk 或一次 decode step。长 prompt 的 prefill 可能到秒级。

---

## 11. 部署、资源与安全

### 11.1 显存

C API 不改变引擎的容量模型。`max_seq`/`max_slots`/`kv_type`/`moe_vram_mb` 的
组合决定 VRAM；`max_slots` 的 recurrent state 是 27B 上最大的非权重分配。
`AGENTS.md` 的容量表就是 API 的容量表。模型加载失败（显存不足）返回
`SYI_OUT_OF_MEMORY`/`SYI_BACKEND`，不会留下半个模型。

### 11.2 进程与模型

* 推荐**一个进程一个模型**（与 router 的 worker 拓扑一致），避免多模型争抢
  同一张卡。
* 同一进程允许多个 `syi_model`，但：
  * 每个模型独立加载权重，显存相加；
  * M1 完成 per-engine KV 类型后，不同 KV 类型可以共存；在那之前（或对仍未
    per-backend 化的 CPU 线程数），创建时检查并返回 `SYI_UNSUPPORTED`，而不是
    静默串味。目标状态是 KV 类型完全 per-engine、CPU 线程池 per-backend；
  * `fork()` 在 SYCL 初始化后不安全；多进程请用 `spawn`/`forkserver`。

### 11.3 安全

* C API 本身不监听网络；安全性由宿主决定。
* 模型路径、`pc_dir` 是宿主提供的本地路径；库不做沙箱。
* 停止串/输入长度有上限（`max_stop` 32、单串 256 字节；prompt 受 `max_seq`
  约束），避免恶意输入无限占用内存。
* `syi_free` 只接受库返回的指针；不要用 `free()` 释放库分配的内存。

---

## 12. 测试计划

### 12.1 无需模型/GPU 的 hermetic 测试

* ABI：`test_capi_abi`——结构体大小/偏移、`struct_size` 兼容、旧版本头文件模拟、
  `syi_abi_version`、符号可见性（`nm -D` 只有 `syi_*`）。
* 生命周期：create/submit/cancel/destroy 顺序、空 prompt、超长 prompt、重复
  submit、跨模型 submit、队列满、超时。
* 错误路径：不存在的模型路径、非法 KV 组合、`max_slots` 越界、`logit_bias`
  越界、token id 越界。
* 线程：N 个线程同时 submit，统计计数守恒；取消与 next 并发；模型先于请求
  销毁。

### 12.2 模型/GPU 测试

* `test_capi_generate`：单请求 greedy，与 `engine::generate`/CLI 的 token 流
  对比（允许 near-tie 已知差异时改为 argmax 对比）。
* `test_capi_batch`：16 个请求同时提交，断言 `avg_decode_batch > 1`、
  `max_decode_batch >= 2`，所有请求都完成且每个结果非空；这是“自动 batch
  真的发生”的硬断言。
* `test_capi_isolation`：两个不同 prompt/seed 的请求并发，结果与各自单独运行
  的 greedy 结果在非 near-tie 上一致；取消 A 不影响 B。
* `test_capi_stop`：停止串跨 token、停止串为空、停止串在第一个 token、超时。
* `test_capi_shutdown`：活跃请求时 shutdown(timeout) 返回 `SYI_AGAIN`，再次
  shutdown 或 destroy 后所有请求进入终态。
* `test_capi_threads`：ThreadPoolExecutor 式提交 64 个请求，统计 `submitted`
  == `completed + cancelled + failed`。

### 12.3 压力与长稳

* 连续 10 万次 submit/cancel/destroy，RSS 与 VRAM 不增长（泄漏检测）。
* 混合长/短 prompt，验证 FIFO 公平性与队列上限。
* Sanitizer：`-fsanitize=thread` 跑 hermetic 线程测试；`-fsanitize=address`
  跑生命周期测试。GPU 部分不能用 ASan 时，至少跑 CPU backend。

### 12.4 CI 矩阵

| 维度 | 值 |
|---|---|
| 编译器 | icpx 2026.1/2026.2 |
| 构建 | Release JIT、Release AOT（acm-g10） |
| backend | CPU、单 GPU、双 GPU layer-map |
| 模型 | 0.8B（tied embedding）、27B（multi-device）、35B A3B（MoE）、qwen4exp（split） |
| 语言 | C99、C11、C++17 |
| Python | 3.10–3.13（wheel 测试） |

---

## 13. 实现里程碑

### M0：文档与 ABI 冻结（本次）

* 本文、Python 文档、两份使用说明。
* 评审通过后再写代码；ABI 变化必须回到本文更新。

### M1：C API 骨架与模型加载

* `include/sycl_infer/sycl_infer.h`、`src/capi/*`、CMake 目标、版本脚本。
* `engine_config` 增加 `cpu_threads`、`kv_k`、`kv_v`；KV 类型从进程级改为
  per-engine（36 处调用点的机械改造 + 测试）。
* `syi_model_create/destroy/info/tokenize/detokenize/render_chat`。
* hermetic 生命周期/ABI 测试。

### M2：调度桥与批量推理

* `sequence::collect_all`、停止串、异常边界、请求级超时。
* admission 窗口/FIFO/队列上限；`syi_request_*` 全套。
* `test_capi_generate`、`test_capi_batch`、`test_capi_isolation`。
* 统计与 `syi_model_get_stats`。

### M3：C++ 包装与 SDK 安装

* `sycl_infer.hpp`、`cmake --install`、pkg-config、CMake config。
* C99/C11/C++17 编译测试；`nm -D` 符号检查。

### M4：Python binding 与 wheel

* 见 [20-python-binding.md](20-python-binding.md)。
* Python 测试矩阵、wheel 自检（`sycl_infer.doctor`）。

### M5（v1.1）：多模态独占 lane

* `engine::generate_mm` + 一个独占锁；同一时刻只跑一个多模态请求，文本请求
  排队等待或走另一个模型实例。
* 复用 `mm_build_prompt_*`，不新建视觉/音频路径。

### M6（v2）：投机解码、每 backend CPU 线程池、跨序列 prefill batching

* 只有在引擎侧有实测收益/正确性保证后再暴露。

---

## 14. 备选方案与取舍

### 14.1 直接暴露 C++ 引擎

* 优点：零封装。
* 缺点：ABI 不稳定（`std::string`/`std::vector`/异常/模板），宿主必须用同一个
  编译器、同一份 STL、同一套 oneAPI；升级引擎就会破坏宿主。否决。

### 14.2 每个请求一个线程/一个引擎

* 优点：实现简单，天然隔离。
* 缺点：每个引擎复制权重，显存爆炸；没有 batch；GPU 队列竞争。否决。

### 14.3 只做同步 `generate()`，不做异步/流式

* 优点：C API 最小。
* 缺点：无法表达“多个请求在同一窗口内到达”，自动 batching 退化为调度器内部
  的偶然重叠；也无法流式输出。否决。

### 14.4 只做回调式流式，不做拉模式

* 优点：C 用户熟悉。
* 缺点：回调运行在调度线程上，阻塞会拖住所有请求；需要额外的交付线程和
  背压。v1 用拉模式（`next`/`wait`），回调留到 v2。

### 14.5 把 KV 类型继续留在进程级

* 优点：不改 36 处调用点。
* 缺点：嵌入场景下两个模型不能有不同的 KV 类型，且初始化顺序依赖全局状态；
  与“稳定、可组合的库”目标冲突。否决，M1 做 per-engine。

---

## 15. 附录：最小 C 示例

```c
#include <stdio.h>
#include <string.h>
#include <sycl_infer/sycl_infer.h>

int main(void) {
    syi_model_options mo;
    syi_model_options_init(&mo);
    mo.model_path = "/path/to/Qwen3.5-0.8B-Q4_K_M.gguf";
    mo.device = SYI_DEVICE_GPU;
    mo.max_seq = 8192;
    mo.max_slots = 8;
    mo.batch_window_us = 2000;

    syi_model *model = NULL;
    syi_status st = syi_model_create(&mo, &model);
    if (st != SYI_OK) {
        fprintf(stderr, "load: %s\n", syi_last_error());
        return 1;
    }

    syi_generation_options go;
    syi_generation_options_init(&go);
    go.max_tokens = 64;
    go.temperature = 0.0f;   /* greedy */

    const char *prompts[] = {"Hello", "The capital of France is", "1+1="};
    syi_request *reqs[3] = {0};

    for (int i = 0; i < 3; i++) {
        if (syi_request_create(model, &go, &reqs[i]) != SYI_OK ||
            syi_request_set_text(reqs[i], prompts[i], strlen(prompts[i])) != SYI_OK) {
            fprintf(stderr, "request: %s\n", syi_last_error());
            return 1;
        }
    }
    if (syi_submit_batch(model, reqs, 3) != SYI_OK) {
        fprintf(stderr, "submit: %s\n", syi_last_error());
        return 1;
    }
    if (syi_wait_batch(reqs, 3, 120000) != SYI_OK) {
        fprintf(stderr, "wait: %s\n", syi_last_error());
        return 1;
    }

    for (int i = 0; i < 3; i++) {
        syi_result r;
        syi_result_init(&r);
        syi_request_get_result(reqs[i], &r);
        printf("[%d] %.*s (%s)\n", i, (int)r.text.size, r.text.data,
               r.finish_reason == SYI_FINISH_STOP ? "stop" : "length");
        syi_request_destroy(reqs[i]);
    }
    syi_model_destroy(model);
    return 0;
}
```

编译（动态库）：

```bash
cc -std=c11 app.c -I/path/to/sycl-infer/include \
   -L/path/to/sycl-infer/lib -lsycl_infer -o app
export LD_LIBRARY_PATH=/path/to/sycl-infer/lib:$LD_LIBRARY_PATH
```
