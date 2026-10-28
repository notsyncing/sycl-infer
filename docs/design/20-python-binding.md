# Python binding 与 wheel 打包设计

> 实现状态（v0.1）：设计目标不变，但当前实现使用 **ctypes** 直接调用 C ABI，
> 因为构建机没有 pybind11 且无法联网；公开 API 与本文一致，wheel 用 setuptools
> 打包并内置 C 库与运行时闭包。pybind11 仍是后续的性能路径（见 §4.1）。
> 已验证：CPU 0.8B 生成、流式、停止串、取消、2 请求批处理、wheel 在无
> `LD_LIBRARY_PATH` 的环境下自加载。

本文定义 `sycl_infer` Python 包：构建在 [C API](19-c-api.md) 之上的 pybind11
扩展、Python 风格的异步/流式/批量接口，以及把 SYCL/oneDNN/oneAPI 运行时和本项目
一起打进 wheel 的构建方案。

> 状态：设计稿（v1）。C ABI 是唯一语义来源；Python 层不复制推理逻辑，只做
> 类型转换、GIL 管理、异常映射和生命周期包装。

---

## 1. 目标与非目标

### 1.1 目标

1. `pip install sycl_infer-*.whl` 后，`import sycl_infer` 即可加载 GGUF 并推理，
   不需要用户设置 `LD_LIBRARY_PATH`、不需要安装 oneAPI toolkit。
2. API 符合 Python 习惯：`Model`、`Request`、`Generation`、迭代器、上下文管理器、
   关键字参数、dataclass 式结果。
3. 多线程/批量：`Model` 可被多个 Python 线程共享；提交后由 C API 自动 batching；
   `generate_batch` 保证一批请求先全部提交再等待。
4. 阻塞调用释放 GIL，Python 线程可以真正并行提交/等待。
5. wheel 内含本项目 + SYCL 运行时 + oneDNN + oneAPI 编译器运行时；通过 RPATH
   解析，不依赖宿主环境变量；同时提供 `doctor` 自检。
6. 与 C API 版本绑定：Python 包版本 `0.1.x` 对应 ABI 1；wheel 元数据记录
   oneAPI/oneDNN 版本。

### 1.2 非目标（v1）

* 多模态请求（图像/视频/音频）——与 C API 同步，v1.1 再做。
* 投机解码（MTP/DFlash2）——C API v1 不暴露，Python 也不暴露。
* 在 Python 进程内加载多个不同 KV 类型的模型——受 C API §3.1 的进程级限制；
  推荐一进程一模型。
* Windows/macOS wheel。
* `asyncio` 原生接口——v1 提供同步 + 迭代器；`asyncio.to_thread` 即可接入。
* 训练/微调/embedding 接口。

---

## 2. 包结构

```
python/
├── pyproject.toml
├── CMakeLists.txt                 # pybind11_add_module(_sycl_infer)
├── src/
│   ├── module.cpp                 # 模块入口
│   ├── model.cpp                  # Model 构造参数、info/stats
│   ├── request.cpp                # Request / Token 迭代器
│   ├── types.cpp                  # dataclass 的 py::class_ 绑定
│   ├── errors.cpp                 # 异常映射
│   └── doctor.cpp                 # _doctor() 辅助
└── sycl_infer/
    ├── __init__.py                # 公开 API 再导出 + __version__
    ├── _types.py                  # ChatMessage 等纯 Python dataclass
    ├── _runtime.py                # 运行时定位/诊断（纯 Python）
    ├── doctor.py                  # python -m sycl_infer.doctor
    ├── py.typed
    ├── include/sycl_infer/        # C 头文件（供 SDK 用户）
    ├── _lib/                      # 随 wheel 打包的 DSO
    │   ├── libsycl_infer.so.1
    │   ├── libsycl.so.9
    │   ├── libdnnl.so.3
    │   ├── libtbb.so.12
    │   ├── libOpenCL.so.1
    │   ├── libze_loader.so.1
    │   ├── libimf.so / libsvml.so / libintlc.so.5 / libirng.so
    │   └── ...
    ├── _sycl_infer.cpython-3xx-x86_64-linux-gnu.so
    └── share/cmake/sycl_infer/    # 可选：SDK 的 CMake config
```

安装后：

```
site-packages/
├── sycl_infer/
│   ├── __init__.py
│   ├── _sycl_infer...so
│   ├── _lib/libsycl_infer.so.1
│   ├── _lib/libsycl.so.9
│   └── ...
└── sycl_infer-0.1.0.dist-info/
```

`_sycl_infer` 的 `DT_RUNPATH` 设为 `$ORIGIN/_lib`；`_lib` 内的 DSO 设为
`$ORIGIN`。这样无论 wheel 安装到哪个 `site-packages`，都不需要
`LD_LIBRARY_PATH`。

---

## 3. Python API

### 3.1 顶层导出

```python
# sycl_infer/__init__.py
from ._sycl_infer import (
    Model, ModelInfo, ModelStats,
    Request, RequestState, Generation, Token, Logprob,
    SyclInferError, InvalidArgumentError, ModelLoadError,
    BackendError, OutOfMemoryError, QueueFullError,
    TimeoutError, CancelledError, UnsupportedError, ClosedError,
    __version__, __abi_version__,
)
from ._types import ChatMessage
```

`__all__` 固定；新增 API 走 minor 版本。

### 3.2 `Model`

```python
class Model:
    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        device: Literal["auto", "gpu", "cpu"] = "auto",
        layer_map: str | None = None,
        max_seq: int = 0,            # 0 = 8192, -1 = 模型训练上下文
        max_slots: int = 0,          # 0 = 16
        n_blocks: int = 0,
        kv_cap_mb: int = -1,
        kv_type: str = "i8",         # "i8" | "i4" | "f16" | "bf16" | "f32" | "i4:i8"
        cpu_threads: int = 0,
        n_splits: int = 0,
        dec_splits: int = 0,
        batch_window_ms: float = 2.0,
        max_batch_rows: int = 0,
        max_queue_depth: int = 1024,
        max_queue_tokens: int = 4 * 1024 * 1024,
        request_timeout_s: float | None = None,
        prefix_cache: bool | None = None,
        pc_dir: str | None = None,
        pc_vram_mb: int = -1,
        pc_ram_mb: int = -1,
        pc_disk_mb: int = -1,
        moe_resident: int = 0,
        moe_vram_mb: int = 0,
        moe_ram_mb: int = 0,
        engram_ram_mb: int = -1,
        yarn: bool = False,
        yarn_factor: float = 0.0,
        yarn_orig_ctx: int = 0,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_attn_factor: float = 0.0,
        head_split: bool = False,
        log: Callable[[int, str], None] | None = None,
    ) -> None: ...

    @property
    def info(self) -> ModelInfo: ...
    @property
    def stats(self) -> ModelStats: ...
    @property
    def closed(self) -> bool: ...

    def tokenize(self, text: str, *, parse_special: bool = True) -> list[int]: ...
    def detokenize(self, ids: Sequence[int]) -> str: ...
    def apply_chat_template(
        self,
        messages: Sequence[ChatMessage | Mapping[str, str]],
        *,
        enable_thinking: bool = False,
        tools: Sequence[Mapping[str, Any]] | str | None = None,
    ) -> str: ...

    def submit(
        self,
        prompt: str | Sequence[int] | Sequence[ChatMessage | Mapping[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.0,
        repeat_penalty: float = 1.0,
        repeat_last_n: int = 64,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: int = 0,
        ignore_eos: bool = False,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Mapping[int, float] | None = None,
        stop: str | Sequence[str] | None = None,
        timeout: float | None = None,
        enable_thinking: bool = False,
        tools: Sequence[Mapping[str, Any]] | str | None = None,
    ) -> Request: ...

    def generate(self, prompt, *, timeout: float | None = None, **params) -> Generation: ...
    def stream(self, prompt, *, timeout: float | None = None, **params) -> Iterator[Token]: ...
    def generate_batch(
        self,
        prompts: Sequence[str | Sequence[int] | Sequence[ChatMessage]],
        *,
        timeout: float | None = None,
        return_exceptions: bool = False,
        **params,
    ) -> list[Generation | BaseException]: ...

    def chat(self, messages, *, timeout: float | None = None, **params) -> Generation: ...
    def chat_stream(self, messages, *, timeout: float | None = None, **params) -> Iterator[Token]: ...

    def close(self, *, timeout: float | None = None) -> None: ...
    def __enter__(self) -> Model: ...
    def __exit__(self, *exc) -> None: ...
```

设计说明：

* **字符串与 token 的自动识别**：`str` -> `syi_request_set_text`；
  `Sequence[int]` -> `syi_request_set_tokens`；`Sequence[ChatMessage]` ->
  `syi_request_set_chat`。`bytes` 不自动当 UTF-8，避免歧义；需要字节输入时用
  `list[int]`。
* **`kv_type` 接受 `"i4:i8"`**，内部拆成 K/V 两个枚举；其他组合由 C API 校验。
* **`batch_window_ms` -> `batch_window_us`**；保留毫秒对 Python 用户更自然。
* **`timeout`** 是请求级总预算（秒）：`None` = 使用模型默认；`0` = 不限；
  `>0` = 秒。`Request.next(timeout=...)` 是每次等待的超时（`None` = 无限等待，
  `0` = 非阻塞轮询）。
* **`log`** 回调在 C API 的 `log_fn` 上转发；它在调用线程/调度线程执行，Python
  侧必须快速返回，且不得在回调里调用同一 `Model` 的方法（会死锁）。默认不注册
  （库静默）。如果回调抛异常，绑定层吞掉并记一条 warning，不让 C 回调把异常
  带过 ABI。

### 3.3 `Request`

```python
class Request:
    @property
    def id(self) -> int: ...
    @property
    def state(self) -> RequestState: ...
    @property
    def prompt_tokens(self) -> int: ...

    def next(self, timeout: float | None = None) -> Token | None:
        """下一个 token；None 表示等待超时。终态时抛 StopIteration。"""

    def wait(self, timeout: float | None = None) -> Generation:
        """等待终态并返回完整结果；超时抛 TimeoutError（请求继续运行）。"""

    def result(self, timeout: float | None = None) -> Generation: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...
    def __iter__(self) -> Request: ...
    def __next__(self) -> Token: ...
    def __enter__(self) -> Request: ...
    def __exit__(self, *exc) -> None: ...
```

* 迭代器在内部调用 `syi_request_next`；每次 `next` 释放 GIL，返回前重新获取。
* 迭代结束（`SYI_END`）时自动 `close()` 并 `StopIteration`；如果
  `finish_reason == ERROR`，抛 `SyclInferError` 子类，而不是 StopIteration。
* `close()` 对运行中的请求执行 cancel；对已终态请求释放句柄。重复 close 幂等。
* 同一个 `Request` 只能有一个消费者线程；`cancel()` 可从任意线程调用。

### 3.4 结果类型

```python
@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None

@dataclass(frozen=True, slots=True)
class Logprob:
    id: int
    logprob: float
    text: str

@dataclass(frozen=True, slots=True)
class Token:
    id: int
    text: str
    index: int
    logprob: float | None
    top_logprobs: tuple[Logprob, ...]

@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    token_ids: tuple[int, ...]
    finish_reason: Literal["stop", "length", "cancelled", "timeout", "error"]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    ttft_ms: float
    total_ms: float
    error: str | None

@dataclass(frozen=True, slots=True)
class ModelInfo:
    model_name: str
    architecture: str
    device_name: str
    n_vocab: int
    n_ctx: int
    n_embd: int
    n_layer: int
    n_head: int
    n_head_kv: int
    head_dim: int
    is_moe: bool
    is_split_gguf: bool
    max_seq: int
    max_slots: int
    kv_type: str
    kv_type_v: str

@dataclass(frozen=True, slots=True)
class ModelStats:
    submitted: int
    admitted: int
    completed: int
    cancelled: int
    failed: int
    generated_tokens: int
    prefill_steps: int
    decode_steps: int
    avg_decode_batch: float
    max_decode_batch: int
    queued: int
    active: int
    free_slots: int
```

`Token.text` 是完整的 UTF-8 piece；可能为空（token 的字节还没拼完整）。拼接所有
非空 piece 应等于 `Generation.text`（停止串过滤后）。

`RequestState` 是固定字符串枚举：

```python
class RequestState(str, enum.Enum):
    CREATED = "created"
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### 3.5 异常

```
SyclInferError
├── InvalidArgumentError
├── InvalidStateError
├── UnsupportedError
├── ModelLoadError
├── BackendError
├── OutOfMemoryError
├── QueueFullError
├── TimeoutError            # 继承 builtins.TimeoutError + SyclInferError
├── CancelledError          # 继承 builtins.CancelledError? 见下
└── ClosedError
```

* `TimeoutError` 同时继承 `builtins.TimeoutError`，方便通用 `except TimeoutError`。
* `CancelledError` 继承 `SyclInferError`，不继承 `asyncio.CancelledError`（避免
  语义混淆）；`Request.cancel()` 后迭代会抛它。
* `QueueFullError` 对应 `SYI_QUEUE_FULL`；`generate_batch` 默认整批失败并抛它。
* 异步失败（`finish_reason=error`）在 `Generation.error` 里保留消息；迭代器在
  结束处抛 `SyclInferError`。

### 3.6 线程与 GIL

| Python 调用 | GIL | 说明 |
|---|---|---|
| `Model(...)` | 释放 | 加载可能数十秒 |
| `Model.submit` | 释放 | 内部只做短锁，但可能 tokenize/渲染模板 |
| `Request.next` | 释放 | 阻塞在 C CV 上 |
| `Request.wait/result` | 释放 | 同上 |
| `Model.stats/info/tokenize/detokenize` | 保持 | 短操作；tokenize 大文本可释放 |
| `Request.cancel` | 保持 | 原子操作 |
| 日志回调 | 获取 | 从 C 线程进入 Python，必须 `PyGILState_Ensure` |

规则：

* 任何可能阻塞超过 1 ms 的调用都释放 GIL；pybind11 用
  `py::gil_scoped_release`。
* 日志回调在 C 线程上执行，绑定层用 `PyGILState_Ensure` 获取 GIL；回调异常被
  捕获并丢弃，避免跨 ABI 传播。
* 不提供 Python 回调式 token 流：拉模式（迭代器）已经足够，且避免“回调阻塞
  调度线程”的坑。

### 3.7 并发示例

```python
import sycl_infer as si
from concurrent.futures import ThreadPoolExecutor

with si.Model("Qwen3.5-0.8B-Q4_K_M.gguf", max_slots=8) as model:
    prompts = [f"Question {i}: ..." for i in range(16)]

    # 方式 A：显式批量（一次提交，自动 batching）
    results = model.generate_batch(prompts, max_tokens=64, temperature=0.0)

    # 方式 B：多线程提交（每个线程一个请求；C API 自动成批）
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda p: model.generate(p, max_tokens=64), prompts))
```

### 3.8 流式与取消

```python
req = model.submit("Write a story", max_tokens=512)
try:
    for tok in req:
        print(tok.text, end="", flush=True)
except si.CancelledError:
    print("\n[cancelled]")
finally:
    req.close()
```

客户端断开/用户取消时调用 `req.cancel()`；C API 在下一个调度迭代停止该请求，
不影响同批其他请求。

### 3.9 `asyncio` 接入

v1 不内建 event loop 集成。阻塞 API 可以这样用：

```python
import asyncio

async def generate(model, prompt):
    return await asyncio.to_thread(model.generate, prompt, max_tokens=64)
```

不要在每个协程里创建新 `Model`；共享一个 `Model`，让 C API 做 batching。

### 3.10 上下文管理器与关闭

```python
with sycl_infer.Model(path) as model:
    ...
# __exit__ 调用 syi_model_shutdown(默认 5s)，超时则取消剩余请求并等待
```

`Model.close(timeout=...)`：

* 停止接收新请求；
* 取消未完成请求；
* 等待调度线程退出；
* 超时抛 `TimeoutError`，但模型进入 closed 状态，后续调用抛 `ClosedError`；
* 已完成的 `Generation` 不受影响。

`Model` 不在 `__del__` 里做阻塞清理（解释器退出顺序不可靠）；显式 close/with。

---

## 4. 绑定实现

### 4.1 pybind11 还是 CPython C API

选 **pybind11**：

* 项目已经是 C++17；pybind11 是 header-only，构建依赖清晰；
* 异常、GIL、`std::string`/`std::vector` 转换成熟；
* 比手写 `PyObject*` 少几个数量级的样板；
* 不引入运行时依赖（header-only）。

版本：`pybind11>=2.13,<3`。wheel 构建时通过 `[build-system].requires` 获取，
不把 pybind11 打进运行时 wheel。

### 4.2 绑定层职责

* `Model` 持有一个 `syi_model*`；构造时把 Python 参数翻译成
  `syi_model_options`，字符串用 UTF-8（`PyUnicode_AsUTF8AndSize`）。
* `Request` 持有 `syi_request*`；`__next__` 调用 `syi_request_next`，把
  `syi_token` 转成 Python `Token`（文本/日志概率复制成 `str`/tuple）。
* `Generation` 从 `syi_result` 复制；复制后 C 侧结果可以继续存活，Python 对象
  不依赖请求生命周期。
* 所有 C 字符串都按 `size` 复制，不假设 NUL 终止（虽然库保证有）。
* 错误映射集中在一个 `throw_status()` 函数里，按 `syi_status` 抛对应异常，
  消息取 `syi_last_error()`；异步错误取 `syi_request_error()`。
* `Model` 的 `close` 用 `syi_model_shutdown`；`__del__` 只 `syi_model_destroy`
  兜底（可能阻塞，但只在用户忘记 close 时发生）。
* `Request.__del__` 调 `syi_request_destroy`；不要在 `__del__` 里 cancel 后等待
  太久；`close()` 是显式路径。

### 4.3 类型转换细节

* 生成参数（`submit`/`generate` 的关键字参数）在绑定层填成
  `syi_generation_options`；`logit_bias` 的 `dict[int, float]` 转成两个 C 数组，
  在调用期间保持存活。
* `stop`：`str` -> 单元素列表；`Sequence[str]` -> `const char*[]`；长度/单串
  长度由 C API 校验。
* `top_logprobs`：Python `tuple[Logprob, ...]`；C 数组在 `next` 返回后失效，绑定
  层立即复制。
* `ChatMessage`：Python dataclass/`Mapping` -> `syi_chat_message` 数组；`tools`
  用 `json.dumps` 序列化成 `tools_json`（ensure_ascii=False）。
* `prompt` 为 `Sequence[int]` 时，校验每个元素 0..2^31-1；负数/越界抛
  `InvalidArgumentError`。
* `seed`：Python int -> `uint64_t`；负数拒绝（或按 2^64 取模？拒绝更清晰）。

### 4.4 模块初始化

```cpp
PYBIND11_MODULE(_sycl_infer, m) {
    m.doc() = "sycl-infer Python binding (C ABI " SYI_ABI_VERSION_STR ")";
    m.attr("__version__") = SYCL_INFER_VERSION;
    m.attr("__abi_version__") = syi_abi_version();
    // exceptions first, then classes
    register_errors(m);
    register_types(m);
    register_model(m);
    register_request(m);
}
```

`__init__.py` 只做 re-export 和 `__version__` 校验，避免在 import 时加载模型。

---

## 5. Wheel 打包

### 5.1 为什么不是纯 `pip install` 源码构建

用户要的是“装完就能用”，而不是“先装 oneAPI toolkit”。wheel 需要：

1. 编译好的 `libsycl_infer.so.1` + Python 扩展；
2. SYCL runtime（`libsycl.so.9` 等）；
3. oneDNN（`libdnnl.so.3`）及 TBB；
4. oneAPI 编译器运行时（`libimf.so`、`libsvml.so`、`libintlc.so.5`、`libirng.so`）；
5. OpenCL/Level Zero loader（`libOpenCL.so.1`、`libze_loader.so.1`）；
6. 可选：JIT 所需的 IGC（`libigc.so`、`libigdfcl.so`）或依赖
   `intel-opencl-rt`。

**不能打包**：Linux 内核 GPU 驱动（`i915`/`xe`）和固件。用户必须安装与 GPU 匹配
的驱动；`intel-opencl-rt` 提供用户态 OpenCL/IGC，可作为依赖或随 wheel vendoring。

### 5.2 参考 PyPI 上的 oneAPI 包

Intel 已经把这些运行时拆成 PyPI 包，构建时可以直接以它们为“运行时来源”，避免
从完整 oneAPI toolkit 里挑选文件：

| 包 | 内容 | 用途 |
|---|---|---|
| `intel-sycl-rt` | `libsycl.so.*`、SYCL runtime | 必需 |
| `intel-cmplr-lib-rt` | `libimf.so`、`libsvml.so`、`libintlc.so.5`、`libirng.so` | 必需 |
| `intel-opencl-rt` | OpenCL/Level Zero 用户态运行时、IGC | JIT/驱动；可选依赖 |
| `intel-pti` | PTI 工具 | 不需要 |

oneDNN 不在这些包里；从 oneAPI 的 `dnnl/<version>/lib/libdnnl.so.3` 取，或从
oneDNN 官方 release 构建。TBB 从系统/`tbb` PyPI 包取。

**版本必须绑定**：`libsycl.so`、`libdnnl.so`、编译器运行时、IGC 之间有兼容矩阵。
wheel 元数据记录 `oneapi_version` / `dnnl_version` / `igc_version`，`doctor` 打印。

### 5.3 打包策略

提供两种构建 profile：

#### 默认：`bundled`（推荐，自包含）

* 构建时从 `intel-sycl-rt`/`intel-cmplr-lib-rt` wheel（或本地 oneAPI）复制 DSO 到
  `sycl_infer/_lib/`。
* 复制 oneDNN、TBB；如果允许再分发 IGC，也复制（或把 `intel-opencl-rt` 作为
  runtime dependency）。
* 用 `patchelf`/CMake RPATH 修正：
  * `_sycl_infer*.so`：`$ORIGIN/_lib`
  * `_lib/*.so*`：`$ORIGIN`
  * 对 dlopen 的库（`libsycl` -> `libze_loader` / `libOpenCL`）也设置 `$ORIGIN`
    RUNPATH，避免它按系统路径找到不兼容版本。
* `auditwheel repair --plat manylinux_2_28_x86_64` 兜底：把仍被链接、但不在
  `_lib` 的第三方库复制到 wheel 的 `.libs` 并 patch RUNPATH。已经由
  `vendor_runtime.cmake` 放进 `_lib` 的库用 `--exclude` 排掉，避免同一份 DSO
  在 wheel 里出现两次（`--exclude libsycl.so.9 --exclude libdnnl.so.3 ...`）。
* 许可证文件：`THIRD_PARTY_NOTICES`，包含 Intel 的 oneAPI 运行时许可（ISSL/EULA，以发行包为准）、oneDNN
  Apache-2.0、TBB Apache-2.0、OpenCL/Level Zero 的许可证。不要在未确认再分发
  条款前打包 IGC/驱动。

#### 可选：`slim`

* wheel 只含 `libsycl_infer.so.1` + 扩展 + 头文件；
* `install_requires` 依赖 `intel-sycl-rt`、`intel-cmplr-lib-rt`、`intel-opencl-rt`；
* 体积小、与 Intel 官方运行时升级一致；适合 CI/容器。

两种 profile 使用同一个 Python API；用环境变量
`SYCL_INFER_BUNDLE_RUNTIME=0` 或构建选项 `-DSI_PY_BUNDLE_RUNTIME=OFF` 切换。
发布到 PyPI 的默认 wheel 用 `bundled`，因为用户明确要求“打包本项目及 SYCL
运行时”。

### 5.4 wheel 标签

* Python：`cp310`, `cp311`, `cp312`, `cp313`（2026 年仍维护的 CPython）。
* 平台：`manylinux_2_28_x86_64`（glibc >= 2.28；oneAPI 2026 的基线）。如果用
  `manylinux_2_31`，兼容性更窄；不推荐。
* 不使用 `abi3`：pybind11 可以使用 limited API，但 C++ 异常/类型和 SYCL 运行时
  的 ABI 太复杂，收益不值得。每个 Python 版本一个 wheel。
* 文件名示例：
  `sycl_infer-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl`。
* 体积预估：`bundled` 约 250–450 MB（IGC 占大头）；`slim` 约 10–20 MB。

### 5.5 JIT 与 AOT

* **默认 wheel 是 JIT 构建**：设备 kernel 在首次调用时由 IGC 编译，因此运行时
  需要 IGC 和可写的 `SYCL_CACHE_DIR`。首次调用有 1–3 s 的编译开销（见
  `AGENTS.md` 的 AOT 说明）。
* **AOT wheel**：用 `-DSYCL_INFER_AOT=ON -DSYCL_INFER_AOT_DEVICE=acm-g10`
  （A770/DG2）或 `adl-p`（Iris Xe）构建，设备镜像烘焙在二进制里；运行时不需要
  IGC，首次调用快。同一份 wheel 不能同时覆盖两种设备，因此按设备发布单独的
  distribution（例如 `sycl-infer-aot-acm-g10`），或用 `--extra-index-url`。
* 默认 JIT wheel 的 `doctor` 会提示：如果目标机器没有 IGC，改用 AOT wheel；
  如果 `SYCL_CACHE_PERSISTENT=1` 在驱动上崩溃（`AGENTS.md` 有记录），关闭缓存。
* wheel 的 `METADATA` 增加自定义字段（或 `sycl_infer/_build.json`）：
  `{"build": "jit", "aot_device": null, "oneapi": "2026.2", "dnnl": "3.13"}`。

### 5.6 runtime discovery 与 RPATH

构建/安装脚本：

```cmake
# 扩展
set_target_properties(_sycl_infer PROPERTIES
    INSTALL_RPATH "$ORIGIN/_lib"
    BUILD_WITH_INSTALL_RPATH ON)
# 打包的 DSO
set_target_properties(sycl_infer_c PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON)
```

`libsycl.so` 通过 `dlopen("libze_loader.so.1")` 找 Level Zero；需要给它打
`RUNPATH=$ORIGIN`，否则它会从系统路径加载，可能与 wheel 内的 SYCL 不匹配。
`libOpenCL.so.1`/ICD 的 vendor JSON 也需要相对路径；`doctor` 检查。

不要依赖 `LD_LIBRARY_PATH`：它会被宿主环境污染，也违反“装完即用”。RPATH 是
首选；`LD_LIBRARY_PATH` 只在调试时使用。

### 5.7 构建工具链

* 构建后端：`scikit-build-core>=0.10`（CMake 驱动，支持 `wheel.packages`、
  `cmake.targets`、`install-dir`）。
* 扩展：`pybind11>=2.13`。
* 修复/审计：`auditwheel>=6`（Linux），可选 `patchelf`。
* CMake：>= 3.26（项目已有）。
* 编译器：`icpx`；wheel 构建必须在 `source /opt/intel/oneapi/setvars.sh` 之后，
  否则 `-fsycl` 和目标设备代码不可用。
* CI：`cibuildwheel` 2.x，Linux 容器内安装 oneAPI runtime/compiler。

`pyproject.toml` 草案：

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "pybind11>=2.13"]
build-backend = "scikit_build_core.build"

[project]
name = "sycl-infer"
version = "0.1.0"
description = "SYCL LLM inference engine with a C ABI and Python binding"
requires-python = ">=3.10"
license = { text = "MIT" }
readme = "README.md"
classifiers = [
  "Programming Language :: Python :: 3",
  "Programming Language :: C++",
  "Operating System :: POSIX :: Linux",
]

[project.optional-dependencies]
test = ["pytest>=8", "numpy"]

[tool.scikit-build]
cmake.version = ">=3.26"
cmake.args = ["-DSI_C_API=ON", "-DSI_PYTHON=ON", "-DSI_PY_BUNDLE_RUNTIME=ON"]
wheel.packages = ["python/sycl_infer"]
wheel.install-dir = "sycl_infer"
build-dir = "build/{wheel_tag}"
```

CMake 侧：

```cmake
option(SI_PYTHON "build the Python extension" OFF)
option(SI_PY_BUNDLE_RUNTIME "vendor SYCL/oneDNN/oneAPI runtime into the wheel" ON)
if(SI_PYTHON AND NOT SI_ROUTER_ONLY)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_sycl_infer python/src/module.cpp ...)
    target_link_libraries(_sycl_infer PRIVATE sycl_infer_c)
    # 扩展只调用 C ABI，不需要 device pass：和 router 一样用 -fno-sycl
    set_source_files_properties(python/src/module.cpp python/src/model.cpp
        python/src/request.cpp python/src/types.cpp python/src/errors.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-sycl")
    target_link_options(_sycl_infer PRIVATE -fno-sycl)
    install(TARGETS _sycl_infer LIBRARY DESTINATION sycl_infer)
    install(TARGETS sycl_infer_c LIBRARY DESTINATION sycl_infer/_lib)
    if(SI_PY_BUNDLE_RUNTIME)
        include(cmake/vendor_runtime.cmake)
    endif()
endif()
```

`vendor_runtime.cmake` 负责：

1. 从 `$ENV{ONEAPI_ROOT}` 或 `python -m pip download intel-sycl-rt==...` 得到的
   目录里定位 DSO；
2. 复制到 `${CMAKE_INSTALL_PREFIX}/sycl_infer/_lib`；
3. 用 `file(RPATH_CHANGE ...)` 或 `patchelf --set-rpath '$ORIGIN'` 修正；
4. 复制 `LICENSE`/`THIRD_PARTY_NOTICES`；
5. 生成 `_build.json`。

### 5.8 构建步骤（从源码到 wheel）

```bash
# 0. 环境
source /opt/intel/oneapi/setvars.sh
cmake --version            # >= 3.26
icpx --version

# 1. 创建 venv 与构建依赖
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip build auditwheel pytest numpy
python -m pip install "scikit-build-core>=0.10" "pybind11>=2.13"

# 2. 直接构建 wheel（scikit-build-core 会调用 CMake）
python -m build --wheel

# 3. Linux 审计/修复（默认 bundled 构建可能已经 patch；auditwheel 兜底）
auditwheel repair dist/sycl_infer-0.1.0-*.whl \
  -w dist/fixed --plat manylinux_2_28_x86_64
auditwheel show dist/fixed/sycl_infer-0.1.0-*.whl

# 4. 安装并自检
python -m pip install --force-reinstall dist/fixed/sycl_infer-0.1.0-*.whl
python -m sycl_infer.doctor
python -m sycl_infer.doctor --model /path/to/Qwen3.5-0.8B-Q4_K_M.gguf

# 5. 冒烟测试
python - <<'PY'
import sycl_infer as si
with si.Model("/path/to/Qwen3.5-0.8B-Q4_K_M.gguf", max_slots=4) as m:
    print(m.generate("The capital of France is", max_tokens=16, temperature=0.0).text)
PY
```

开发模式（不构建 wheel，直接 `pip install -e .`）：

```bash
python -m pip install -e . --no-build-isolation
```

`--no-build-isolation` 需要当前环境已有 `scikit-build-core`/`pybind11`，但能复用
已安装的 oneAPI 和 CMake 缓存。

### 5.9 `doctor`

```bash
python -m sycl_infer.doctor
```

检查项：

1. `import sycl_infer`、`__version__`、`__abi_version__`；
2. `libsycl_infer.so.1` 的 `DT_RUNPATH` 与 `ldd` 结果；
3. 打包的 `libsycl`/`libdnnl`/`libOpenCL`/`libze_loader` 版本（从 `_build.json`）；
4. 可选：`--model` 尝试加载并打印 `ModelInfo`（会真实占显存）；
5. 可选：`--list-devices` 通过 `sycl-ls` 或一次最小 `Model` 构造列出设备；
6. 诊断建议：缺少 IGC -> 用 AOT wheel；`SYCL_CACHE_PERSISTENT` 崩溃 -> 关闭；
   驱动版本不匹配 -> 安装对应 `intel-opencl-rt`。

`doctor` 不默认加载模型，避免在 CI 中占用 GPU。

### 5.10 发布与版本

* Python 包版本与 C ABI 解耦：`0.1.x` 可以包含 ABI 1；ABI major 变化 -> Python
  包 major 变化。
* wheel 元数据：
  * `Requires-Dist: intel-sycl-rt==2026.2.0`（slim profile）；
  * `Provides-Extra: cuda` 不适用；
  * `sycl_infer/_build.json` 记录构建环境。
* 发布矩阵：
  * `sycl-infer`：JIT，bundled，cp310–cp313，manylinux_2_28；
  * `sycl-infer-aot-acm-g10`（可选）：A770/DG2 AOT；
  * `sycl-infer-aot-adl-p`（可选）：Iris Xe AOT；
  * `sycl-infer-sdk`（可选）：C 头文件 + `.so` + CMake config + 静态库。
* 不做 universal wheel；不同 oneAPI 小版本的运行时不要混用。

### 5.11 许可证与合规

* 本项目代码：MIT（与仓库 `LICENSE` 一致）。
* oneDNN：Apache-2.0。
* TBB：Apache-2.0。
* Intel oneAPI runtime/SYCL/编译器运行时：按 Intel 的 oneAPI 运行时许可（ISSL/EULA）再分发；
  wheel 必须包含对应许可证文本和 `THIRD_PARTY_NOTICES`。
* 不要打包 GPU 内核驱动/固件；在 `doctor` 和 README 中明确系统驱动要求。
* 发布前用 `pip-licenses`/`auditwheel show` 检查；不确定的库宁可不打包，改为
  `Requires-Dist`。

---

## 6. 测试计划

### 6.1 无模型测试

* `test_python_import`：导入、版本、ABI、`__all__`。
* `test_python_errors`：参数校验、异常类型、`ClosedError`。
* `test_python_lifecycle`：`with Model`、提前关闭、请求迭代器提前 break。
* `test_python_threads`：多线程 `submit`，统计守恒。
* `test_python_pickle`：确认 `Model`/`Request` 不可 pickle（抛 `TypeError`）。

### 6.2 有模型测试

* `test_python_generate`：greedy 单请求，与 CLI/`engine::generate` 的 token 流
  对比。
* `test_python_batch`：`generate_batch` 16 个 prompt，断言 `stats.avg_decode_batch
  > 1`，每个结果非空。
* `test_python_stream`：迭代器拼接文本 == `Generation.text`；提前 break 会取消。
* `test_python_chat`：`apply_chat_template` 与 `test_chat_template` 的参考输出
  一致；`chat()` 能生成。
* `test_python_threads_model`：8 个线程共享一个 `Model`，结果互不串。
* `test_python_close`：活跃请求时 close，迭代器抛 `CancelledError`，模型后续
  调用抛 `ClosedError`。

### 6.3 wheel 测试

* 在 `manylinux_2_28` 容器 + Intel GPU 直通中安装 wheel；
* 不设置 `LD_LIBRARY_PATH`，运行 `doctor` 和冒烟测试；
* 用 `ldd`/`readelf -d` 断言 RUNPATH；
* 用 `auditwheel show` 确认没有缺失依赖；
* 用 `python -X dev` 检查 ResourceWarning；
* 在无 GPU 的机器上确认 `doctor` 给出清晰错误而不是崩溃。

### 6.4 CI 矩阵

| 维度 | 值 |
|---|---|
| Python | 3.10, 3.11, 3.12, 3.13 |
| wheel | JIT bundled、AOT acm-g10（单独 job） |
| 平台 | manylinux_2_28_x86_64 |
| GPU | 无（hermetic）、Iris Xe、A770、双 A770 |
| 模型 | 0.8B（CI 快速）、27B/35B（夜间） |
| 测试 | pytest + 冒烟脚本 + doctor |

---

## 7. 实现里程碑

### P0：Python 绑定骨架

* pybind11 模块、`Model` 构造/关闭、`generate` 单请求、异常映射。
* 复用 C API M1（模型加载）和 M2（调度）。
* 无 wheel 打包，开发模式测试。

### P1：流式、批量、线程

* `Request` 迭代器、`stream`、`generate_batch`、GIL 释放。
* `test_python_batch`/`test_python_threads`。

### P2：wheel 打包

* `scikit-build-core`、CMake `SI_PYTHON`、RPATH、`vendor_runtime.cmake`。
* `auditwheel`、`doctor`、`_build.json`。
* 在干净的 manylinux 容器里验证“无 LD_LIBRARY_PATH”。

### P3：AOT 与发布

* AOT wheel 变体、发布矩阵、许可证/notice。
* 夜间模型测试、性能冒烟（batch 16 的 avg_decode_batch）。

---

## 8. 备选方案与取舍

### 8.1 纯 `ctypes` 包装，不写 C++ 扩展

* 优点：无编译扩展，wheel 只有 `.so`。
* 缺点：每次调用都要手工定义结构体/联合体，迭代器/异常/GIL 全要自己处理；
  性能和控制力都不如 pybind11。可以作为“无扩展”的备选（`sycl_infer.ctypes`），
  但不是主路径。

### 8.2 用 `cffi`

* 与 ctypes 类似；构建时需要 cffi，运行时仍有跨语言胶水。否决为主路径。

### 8.3 用 `nanobind`

* 更小更快，但生态/调试经验不如 pybind11；项目没有既有依赖。v1 选 pybind11，
  以后可评估。

### 8.4 只发源码包，让用户自己装 oneAPI

* 与“装完即用”目标冲突；wheel 必须自包含运行时（至少提供 bundled profile）。

### 8.5 把 oneDNN/IGC 全部静态链接进扩展

* 体积更大、许可证/符号冲突更复杂；动态库 + RPATH 更容易替换/诊断。否决。

---

## 9. 与 C API 的版本/兼容约定

* Python 包只调用公开的 `syi_*` 符号；不 include 任何 `src/` 内部头文件。
* C ABI 新增函数/字段 -> Python 包 minor；C ABI major -> Python 包 major。
* Python 层不重新解释 `finish_reason` 字符串；映射表集中在 `errors.cpp`/
  `types.cpp`。
* 如果 wheel 内的 `libsycl_infer.so` 与扩展的 ABI 不匹配，`import` 时
  `syi_abi_version()` 检查失败，抛 `ImportError` 并提示重装（不要静默降级）。
* wheel 的 `METADATA`/`_build.json` 记录 `abi_version`、oneAPI、oneDNN、构建类型
  （JIT/AOT），`doctor` 打印。

---

## 10. 附录：完整使用示例

```python
import sycl_infer as si

model = si.Model(
    "/path/to/Qwen3.5-0.8B-Q4_K_M.gguf",
    device="gpu",
    max_slots=8,
    batch_window_ms=2.0,
)

print(model.info)

# 1) 流式
for tok in model.stream("Explain paged attention in one paragraph.",
                        max_tokens=128, temperature=0.7, top_p=0.95):
    print(tok.text, end="", flush=True)

# 2) 批量
prompts = ["Name a color.", "Name a fruit.", "Name a country."]
for i, g in enumerate(model.generate_batch(prompts, max_tokens=16, temperature=0.0)):
    print(i, g.text, g.finish_reason)

# 3) 多线程共享一个模型
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as ex:
    for g in ex.map(lambda p: model.generate(p, max_tokens=32), prompts * 4):
        print(g.text)

# 4) Chat
msgs = [
    si.ChatMessage(role="system", content="You are concise."),
    si.ChatMessage(role="user", content="What is 2+2?"),
]
print(model.chat(msgs, max_tokens=32, temperature=0.0).text)

model.close()
```

---

## 11. 参考

* C ABI：[19-c-api.md](19-c-api.md)
* 构建/测试总览：[12-build-and-testing.md](12-build-and-testing.md)
* AOT 与运行时缓存：[../../AGENTS.md](../../AGENTS.md) 的 *AOT builds* 与
  *Benchmarking* 两节
* 连续批处理与调度器：[04-engine.md](04-engine.md)、[09-server.md](09-server.md)
