# Python binding 使用说明

`sycl_infer` 是 [C API](../design/19-c-api.md) 的 Python 封装。装好 wheel 后，
不需要 oneAPI toolkit、不需要 `LD_LIBRARY_PATH`、不需要 HTTP 服务，就能在进程内
加载 GGUF 并做单请求、流式、批量和多线程推理。

设计细节与 wheel 打包方案见
[design/20-python-binding.md](../design/20-python-binding.md)。

> 状态：v0.1 已实现并通过本地验证（CPU 0.8B、流式/停止串/取消/2 请求批处理）。
> v0.1 使用 ctypes 而不是 pybind11；wheel 由 setuptools 构建，内置 C 库和
> SYCL/oneDNN/oneAPI runtime 闭包。设计差异见
> [design/20-python-binding.md §4.1](../design/20-python-binding.md#41-pybind11-还是-cpython-c-api)。

---

## 1. 安装

### 1.1 从 wheel 安装（推荐）

```bash
python -m pip install sycl_infer-0.1.0-py3-none-linux_x86_64.whl
```

或者（发布到索引后）：

```bash
python -m pip install sycl-infer
```

要求：

* Linux x86_64，glibc >= 2.28；
* Python 3.10–3.13；
* Intel GPU（Iris Xe / Arc）及匹配的驱动；CPU backend 不需要 GPU。

wheel 内含本项目、SYCL runtime、oneDNN、oneAPI 编译器运行时、OpenCL/Level Zero
loader（`bundled` profile）。**GPU 内核驱动/固件不在 wheel 里**，必须由系统提供。

### 1.2 验证安装

```bash
python -m sycl_infer.doctor
```

输出示例（字段随构建变化）：

```
sycl_infer 0.1.0 (abi 1)
extension : /.../site-packages/sycl_infer/_sycl_infer.cpython-312-x86_64-linux-gnu.so
runtime   : /.../site-packages/sycl_infer/_lib
build     : jit, oneapi 2026.2, dnnl 3.13
runpath   : $ORIGIN/_lib
status    : ok
```

带模型做真实加载检查（会占显存）：

```bash
python -m sycl_infer.doctor --model /path/to/Qwen3.5-0.8B-Q4_K_M.gguf
```

### 1.3 从源码构建 wheel

v0.1 的 wheel 由 setuptools 打包，不含扩展模块；C ABI 共享库和运行时闭包
由 `python/vendor_runtime.py` 复制进 `python/sycl_infer/_lib`。完整打包设计见
[design/20-python-binding.md §5](../design/20-python-binding.md#5-wheel-打包)。

```bash
source /opt/intel/oneapi/setvars.sh

# 1. 构建 C ABI（首次或改了 src/ 后）
cmake -S . -B build-rel2 -DCMAKE_BUILD_TYPE=Release -DSI_C_API=ON
cmake --build build-rel2 -j2 --target sycl_infer_c

# 2. 把 libsycl_infer + libsycl/oneDNN/oneAPI 运行时复制进包
python3 python/vendor_runtime.py --build build-rel2

# 3. 构建 wheel（不需要 pybind11/build/scikit-build-core）
cd python
python3 -m pip wheel . --no-deps --no-build-isolation -w dist
cd ..

# 4. 安装并验证（不设置 LD_LIBRARY_PATH）
python3 -m pip install --force-reinstall python/dist/sycl_infer-*.whl
python3 -m sycl_infer.doctor
```

wheel 的标签由构建机的 glibc 决定：本机（glibc 2.43）得到
`py3-none-linux_x86_64`；发布 `manylinux_2_28` 需要在对应容器里构建。
`vendor_runtime.py` 复制的是构建机的 SYCL/oneDNN 版本，发布时固定该版本即可。

开发模式（不打包，直接指向构建目录）：

```bash
export PYTHONPATH="$PWD/python"
export SYCL_INFER_LIB="$PWD/build-rel2/libsycl_infer.so.1.0.0"
python3 -m sycl_infer.doctor
```

`LD_LIBRARY_PATH` 仍需包含 oneAPI 运行时；用 wheel 时不需要，因为包内
`_lib` 已被 `_capi._preload()` 加载。

---

## 2. 快速开始

```python
import sycl_infer as si

with si.Model(
    "/path/to/Qwen3.5-0.8B-Q4_K_M.gguf",
    device="gpu",
    max_slots=8,
) as model:
    print(model.info)

    out = model.generate(
        "The capital of France is",
        max_tokens=16,
        temperature=0.0,   # greedy
    )
    print(out.text)
```

`Model` 是上下文管理器；`__exit__` 会优雅关闭（默认 5 s），超时则取消剩余请求。

---

## 3. 流式输出

```python
with si.Model(path, max_slots=4) as model:
    for tok in model.stream("Write a short poem about GPUs.", max_tokens=128):
        print(tok.text, end="", flush=True)
```

`Token` 字段：

* `id`：token id；
* `text`：完整 UTF-8 piece（可能为空，表示字节还没拼完整）；
* `index`：0-based completion 序号；
* `logprob` / `top_logprobs`：请求了 `logprobs=True` 时才有值。

迭代器提前 `break` 会取消请求；也可以显式：

```python
req = model.submit("...", max_tokens=512)
try:
    for tok in req:
        ...
except si.CancelledError:
    pass
finally:
    req.close()
```

---

## 4. 批量推理

### 4.1 `generate_batch`

```python
prompts = [
    "Name a color.",
    "Name a fruit.",
    "Name a country.",
]
results = model.generate_batch(prompts, max_tokens=16, temperature=0.0)
for i, g in enumerate(results):
    print(i, g.text, g.finish_reason)
```

`generate_batch` 先把所有请求提交（一次 `syi_submit_batch`，窗口立即关闭），
再逐个等待；所以它们会被自动 batch，而不是“提交一个等一个”。

### 4.2 多线程提交

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(
        lambda p: model.generate(p, max_tokens=64, temperature=0.0),
        prompts * 4,
    ))
```

`Model` 可被多个线程共享；阻塞调用释放 GIL。不要在每个线程里 new 一个
`Model`（会复制权重、显存爆炸）。

### 4.3 确认 batch 真的发生

```python
s = model.stats
print(s.active, s.queued, s.avg_decode_batch, s.max_decode_batch)
```

如果 `avg_decode_batch` 总是 1：

* 提交和等待是否交错进行了；
* `max_batch_rows` 是否为 1；
* 模型是否是 `qwen4exp`（引擎实测 2 行 0.85x，强制一行）；
* 请求是否都太短，在 prefill 阶段就结束。

---

## 5. Chat

```python
msgs = [
    si.ChatMessage(role="system", content="You are concise."),
    si.ChatMessage(role="user", content="What is 2+2?"),
]
out = model.chat(msgs, max_tokens=32, temperature=0.0)
print(out.text)

# 只渲染模板
text = model.apply_chat_template(msgs, enable_thinking=False)
print(text)
```

`apply_chat_template` 使用 GGUF 的 `tokenizer.chat_template`；模板不支持时走
内置 ChatML 回退。`tools` 参数接受 OpenAI `tools` 数组或 JSON 字符串。

多轮对话需要把上一轮的 assistant 输出加回 `messages`；binding 不维护会话状态。

---

## 6. Tokenizer

```python
ids = model.tokenize("hello", parse_special=True)
print(ids)
print(model.detokenize(ids))
```

`tokenize` 的输入是 UTF-8 文本；`parse_special=True` 会把 `<|im_start|>` 等
特殊 token 识别为控制 token。`detokenize` 返回原始字节拼成的 `str`。

---

## 7. 参数

`Model` 的关键参数（完整列表见设计文档 §3.2）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `device` | `"auto"` | `"gpu"` / `"cpu"` |
| `max_seq` | 0（8192） | `-1` = 模型训练上下文 |
| `max_slots` | 0（16） | 并发序列数；决定 recurrent state 显存 |
| `kv_type` | `"i8"` | `"i4"`、`"i4:i8"`、`"f16"` 等 |
| `batch_window_ms` | 2.0 | 空闲时的自动 batch 窗口；0 = 不等待 |
| `max_batch_rows` | 0（kMaxB） | 1 = 严格串行/更可复现 |
| `max_queue_depth` | 1024 | 排队上限；超过抛 `QueueFullError` |
| `prefix_cache` | None | None = 引擎默认；False = 关 |
| `moe_vram_mb` | 0 | qwen4exp/35B 的专家 bank 预算 |
| `layer_map` | None | 多卡分层，如 `"0-31:gpu.0,32-63:gpu.1"` |

`generate`/`stream`/`submit` 的关键参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `max_tokens` | 256 | |
| `temperature` | 0.7 | `0` = greedy |
| `top_p` / `top_k` | 0.95 / 40 | |
| `seed` | 0 | 非 0 = 每请求确定性 |
| `stop` | None | 字符串或字符串列表；最多 32 个，单个 ≤256 字节 |
| `logprobs` / `top_logprobs` | False / 0 | |
| `timeout` | None | 请求级总预算（秒）；None = 模型默认，0 = 不限 |

同一批中的每个请求有自己的 seed、penalty、stop、logit_bias；不会互相串。

---

## 8. 关闭与取消

```python
model = si.Model(path)
req = model.submit("...")
...
req.cancel()       # 下一个调度迭代停止该请求；不影响其他请求
model.close()      # 停止接收新请求、取消未完成请求、等待调度线程
```

`Request.close()` 对运行中的请求等价于 cancel；`Model.close()` 后新提交抛
`ClosedError`。已完成的 `Generation` 是独立数据，关闭模型后仍可读。

不要在 `__del__` 里依赖清理；显式 `with`/`close()`。

---

## 9. 性能与调优

* **吞吐优先**：`max_slots` 设为显存允许的最大值，`batch_window_ms=2`（默认）。
* **低延迟单请求**：`batch_window_ms=0`、`max_batch_rows=1`；但这样没有 batch
  收益。
* **长上下文**：降低 `max_slots` 是最便宜的显存 headroom；`max_seq` 按需设置，
  不要默认开满训练上下文。
* **KV 位宽**：`i4:i8` 通常能在接近 i8 的精度下省 25% KV；`i4:i4` 更省但精度
  下降（见 `AGENTS.md` 的实测）。
* **JIT 首次调用**：默认 wheel 是 JIT，第一次推理有编译开销；生产环境可用
  AOT wheel。`SYCL_CACHE_DIR` 可复用编译结果；某些驱动上
  `SYCL_CACHE_PERSISTENT=1` 会崩溃（`AGENTS.md` 有记录），先别开。
* **CPU backend**：`device="cpu"` + `cpu_threads`（进程级）；适合没有 Intel GPU
  的机器，吞吐远低于 GPU。

---

## 10. 故障排查

**`ImportError: libsycl.so.9: cannot open shared object file`**

wheel 的 `_lib` 不完整或 RUNPATH 被破坏。运行 `doctor`；重新安装 wheel；不要
手工 `LD_LIBRARY_PATH` 指向不兼容的 oneAPI。

**`BackendError: no device` / `SYCL device not found`**

没有可用 Intel GPU，或驱动/Level Zero 未安装。检查 `sycl-ls`；CPU 用户改用
`device="cpu"`。

**`ModelLoadError: ... unsupported architecture`**

GGUF 架构不在注册表；或用错了 shard（split GGUF 传第一个分片）。

**`OutOfMemoryError` / `device alloc failed`**

降 `max_seq`/`max_slots`/`moe_vram_mb`，或换 KV 类型；27B 需要多卡分层。

**`QueueFullError`**

排队请求超过 `max_queue_depth`/`max_queue_tokens`；上游限流或增大上限。

**`TimeoutError`**

请求级预算用完；请求会在下一个检查点取消。若只是 `next()` 等待超时，用
`Request.next(timeout=...)` 并继续。

**输出与单请求不完全一样**

并发 batch 改变 fp 累加顺序，near-tie 的 greedy argmax 可能翻转；这是引擎
已知边界。需要严格复现时用 `max_batch_rows=1`、`batch_window_ms=0`、
`prefix_cache=False`。

**wheel 很大**

`bundled` 含 SYCL/oneDNN/编译器运行时，约 250–450 MB；`slim` profile 只装
本项目并依赖 Intel 的 PyPI runtime 包。构建选项见设计文档 §5.3。

---

## 11. 版本与兼容

```python
print(si.__version__)       # Python 包版本
print(si.__abi_version__)   # C ABI 版本
```

* Python 包 minor 版本可能新增 API；C ABI major 变化才会破坏兼容。
* wheel 与 oneAPI/oneDNN 版本绑定；不要混用不同小版本的 runtime。同一进程里
  不要同时加载另一个包自带的 `libsycl`/`libdnnl`（符号冲突通常表现为
  `undefined symbol` 或设备枚举失败）。
* `Model`/`Request` 不可 pickle；多进程用 `spawn`，每个进程各自加载模型。

---

## 12. 完整示例

```python
import sycl_infer as si
from concurrent.futures import ThreadPoolExecutor

MODEL = "/path/to/Qwen3.5-0.8B-Q4_K_M.gguf"

with si.Model(MODEL, device="gpu", max_slots=8, batch_window_ms=2.0) as model:
    print(model.info)

    # 1) 流式
    for tok in model.stream("Explain paged attention.", max_tokens=64):
        print(tok.text, end="", flush=True)
    print()

    # 2) 批量
    prompts = ["Name a color.", "Name a fruit.", "Name a country."]
    for g in model.generate_batch(prompts, max_tokens=16, temperature=0.0):
        print(g.text, g.finish_reason)

    # 3) 多线程
    with ThreadPoolExecutor(max_workers=4) as ex:
        for g in ex.map(lambda p: model.generate(p, max_tokens=32), prompts * 4):
            print(g.text)

    # 4) Chat
    msgs = [
        si.ChatMessage(role="system", content="You are concise."),
        si.ChatMessage(role="user", content="What is 2+2?"),
    ]
    print(model.chat(msgs, max_tokens=16, temperature=0.0).text)

    print(model.stats)
```
