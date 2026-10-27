# C API 使用说明

本指南面向把 `sycl-infer` 直接链接进成品程序的 C/C++ 开发者。ABI 的完整语义、
设计取舍见 [design/19-c-api.md](../design/19-c-api.md)；本文只讲怎么用。

> 状态：v1 目标用法。C API 的实现里程碑见
> [design/19-c-api.md §13](../design/19-c-api.md#13-实现里程碑)。在 M1/M2 落地前，
> 下列构建目标尚不存在；接口形状已经冻结。

---

## 1. 构建与安装

### 1.1 从源码构建 C 库

```bash
source /opt/intel/oneapi/setvars.sh

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSI_C_API=ON
cmake --build build --target sycl_infer_c -j"$(nproc)"
```

产物：

```
build/libsycl_infer.so.1
build/libsycl_infer.so
```

安装到自定义前缀：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSI_C_API=ON \
  -DCMAKE_INSTALL_PREFIX=/opt/sycl-infer
cmake --build build --target sycl_infer_c -j"$(nproc)"
cmake --install build
```

安装内容：

```
/opt/sycl-infer/
├── include/sycl_infer/sycl_infer.h
├── include/sycl_infer/sycl_infer.hpp   # 可选 C++ RAII 包装
├── lib/libsycl_infer.so -> libsycl_infer.so.1
├── lib/libsycl_infer.so.1
├── lib/pkgconfig/sycl_infer.pc
└── lib/cmake/sycl_infer/...
```

### 1.2 静态库（可选）

```bash
cmake -S . -B build -DSI_C_API=ON -DSI_C_API_STATIC=ON
cmake --build build --target sycl_infer_c_static -j"$(nproc)"
```

静态库只保证同一工具链/同一 oneAPI 版本；跨编译器请用动态库。

---

## 2. 链接

### 2.1 CMake

```cmake
find_package(sycl_infer CONFIG REQUIRED)
add_executable(my_app main.c)
target_link_libraries(my_app PRIVATE sycl_infer::c)
```

如果库安装在非标准前缀：

```bash
cmake -S . -B build -Dsycl_infer_DIR=/opt/sycl-infer/lib/cmake/sycl_infer
```

### 2.2 pkg-config

```bash
cc -std=c11 main.c $(pkg-config --cflags --libs sycl_infer) -o my_app
```

### 2.3 手工

```bash
cc -std=c11 main.c \
  -I/opt/sycl-infer/include \
  -L/opt/sycl-infer/lib -lsycl_infer \
  -o my_app

# 动态库：让 loader 找到 libsycl_infer 及其依赖
export LD_LIBRARY_PATH=/opt/sycl-infer/lib:/opt/intel/oneapi/2026.2/lib:$LD_LIBRARY_PATH
```

生产环境不要依赖 `LD_LIBRARY_PATH`：把库和 oneAPI runtime 放到 `RUNPATH` 里，
或使用打包好的 Python wheel / SDK 包。

---

## 3. 生命周期

```c
#include <sycl_infer/sycl_infer.h>

syi_model_options opt;
syi_model_options_init(&opt);
opt.model_path = "/path/to/Qwen3.5-0.8B-Q4_K_M.gguf";
opt.device = SYI_DEVICE_GPU;
opt.max_seq = 8192;
opt.max_slots = 8;

syi_model *model = NULL;
syi_status st = syi_model_create(&opt, &model);
if (st != SYI_OK) {
    fprintf(stderr, "model load failed: %s\n", syi_last_error());
    return 1;
}

/* ... 推理 ... */

syi_model_destroy(model);  /* 取消未完成请求、等待调度线程、释放 */
```

`max_seq=0` 用引擎默认 8192；`-1` 用模型训练上下文；显式值必须配合足够的 KV
容量。`max_slots` 同时决定 recurrent state 的显存，长上下文场景可以先降到 1
（见 `AGENTS.md` 的容量表）。

模型必须活到请求 `submit` 为止（set_text/set_chat 会读 tokenizer/chat template）；
请求提交后，即使先销毁模型，已生成/已取消请求的结果仍可读到
`syi_request_destroy`。推荐顺序仍是“先销毁所有请求，再销毁模型”。

---

## 4. 单请求

```c
syi_generation_options go;
syi_generation_options_init(&go);
go.max_tokens = 128;
go.temperature = 0.7f;
go.top_p = 0.95f;
go.top_k = 40;

syi_request *req = NULL;
if (syi_request_create(model, &go, &req) != SYI_OK ||
    syi_request_set_text(req, "Explain paged attention.", strlen("Explain paged attention.")) != SYI_OK ||
    syi_request_submit(req) != SYI_OK) {
    fprintf(stderr, "%s\n", syi_last_error());
    syi_request_destroy(req);
    syi_model_destroy(model);
    return 1;
}

/* 方式 A：流式读取 */
for (;;) {
    syi_token tok;
    syi_token_init(&tok);
    st = syi_request_next(req, &tok, 60000 /* ms */);
    if (st == SYI_AGAIN) {
        continue;  /* 等待超时，请求仍在跑 */
    }
    if (st == SYI_END) {
        break;
    }
    if (st != SYI_OK) {
        fprintf(stderr, "next: %s\n", syi_last_error());
        break;
    }
    fwrite(tok.text.data, 1, tok.text.size, stdout);
    fflush(stdout);
}

syi_result res;
syi_result_init(&res);
syi_request_get_result(req, &res);
fprintf(stderr, "\nfinish=%d tokens=%zu\n", (int)res.finish_reason, res.n_tokens);
syi_request_destroy(req);
```

`tok.text` 是完整 UTF-8 piece，可能为空（token 字节还没拼完整）；`res.text` 是
过滤停止串后的完整文本。`tok.text`/`tok.top` 在下一次 `next` 前有效，需要保留
就立即复制。`timeout_ms` 为 `0` 是非阻塞轮询，`SYI_TIMEOUT_INFINITE` 是一直等；
示例里的 60000 只是“最多等 60 秒后返回 `SYI_AGAIN`，请求继续跑”。

### 4.1 只等结果，不流式

```c
if (syi_request_submit(req) != SYI_OK) { /* ... */ }
if (syi_request_wait(req, 120000) != SYI_OK) {
    fprintf(stderr, "wait: %s\n", syi_last_error());
}

syi_result res;
syi_result_init(&res);
syi_request_get_result(req, &res);
printf("%.*s\n", (int)res.text.size, res.text.data);
```

批量场景推荐“先全部 submit，再逐个 wait”，这样请求进入同一个批处理窗口。

---

## 5. 批量推理

### 5.1 显式批量

```c
const char *prompts[] = {
    "Name a color.",
    "Name a fruit.",
    "Name a country.",
};
syi_request *reqs[3] = {0};

for (int i = 0; i < 3; i++) {
    syi_request_create(model, &go, &reqs[i]);
    syi_request_set_text(reqs[i], prompts[i], strlen(prompts[i]));
}

/* 原子提交：整批入队，窗口立即关闭 */
if (syi_submit_batch(model, reqs, 3) != SYI_OK) {
    fprintf(stderr, "submit: %s\n", syi_last_error());
    /* 整批都没有提交；销毁后重试 */
}

/* 等全部终态（或超时） */
syi_wait_batch(reqs, 3, 120000);

for (int i = 0; i < 3; i++) {
    syi_result r;
    syi_result_init(&r);
    syi_request_get_result(reqs[i], &r);
    printf("[%d] %.*s\n", i, (int)r.text.size, r.text.data);
    syi_request_destroy(reqs[i]);
}
```

`syi_submit_batch` 是原子的：如果队列上限放不下整批，返回 `SYI_QUEUE_FULL`，
一个都不提交。超过 `max_slots` 的请求留在 FIFO 队列，等 slot 释放后自动被接纳。
`syi_wait_batch` 只表示“所有请求都到了终态”，不表示每个请求都成功；逐个检查
`syi_result.finish_reason`/`error`。

### 5.2 自动批量

即使不调用 `syi_submit_batch`，多个线程/多个循环提交的请求也会在
`batch_window_us` 内自动成组：

```c
for (int i = 0; i < n; i++) {
    syi_request_create(model, &go, &reqs[i]);
    syi_request_set_text(reqs[i], prompts[i], strlen(prompts[i]));
    syi_request_submit(reqs[i]);
}
/* 先全部提交，再统一等待 */
for (int i = 0; i < n; i++) {
    syi_request_wait(reqs[i], 120000);
}
```

正确的用法是“提交和等待分开”；如果每提交一个就 `wait` 一个，后面的请求会错过
前面的 batch 窗口，但不会影响正确性，只是吞吐更低。

### 5.3 看 batch 是否真的发生

```c
syi_model_stats s;
s.struct_size = sizeof(s);
syi_model_get_stats(model, &s);
printf("active=%d queued=%d avg_batch=%.2f max_batch=%u\n",
       s.active, s.queued, s.avg_decode_batch, s.max_decode_batch);
```

如果 `avg_decode_batch` 一直是 1，检查：

* 是否每个请求都 `wait` 完才提交下一个；
* `max_slots`/`max_batch_rows` 是否为 1；
* 模型是否是 `qwen4exp`（引擎实测 2 行 0.85x，强制一行）；
* 是否请求都短到在 prefill 阶段就结束（decode 没机会合批）。

---

## 6. 多线程提交

```c
#include <pthread.h>

typedef struct { syi_model *m; const char *prompt; syi_request *req; } job;

void *worker(void *arg) {
    job *j = arg;
    syi_generation_options go;
    syi_generation_options_init(&go);
    go.max_tokens = 64;
    syi_request_create(j->m, &go, &j->req);
    syi_request_set_text(j->req, j->prompt, strlen(j->prompt));
    syi_request_submit(j->req);   /* 线程安全，不等待 GPU */
    return NULL;
}
```

`Model` 句柄可被任意多个线程共享；`submit` 只做短锁和队列插入。不要让两个线程
同时读同一个 `syi_request` 的 `next`（单消费者）；`cancel` 可以从任意线程调用。

---

## 7. Chat 模板与 tokenizer

```c
syi_chat_message msgs[2];
syi_chat_message_init(&msgs[0]);
msgs[0].role = "system";
msgs[0].content = "You are concise.";
syi_chat_message_init(&msgs[1]);
msgs[1].role = "user";
msgs[1].content = "What is 2+2?";

char *rendered = NULL;
size_t rendered_len = 0;
if (syi_model_render_chat(model, msgs, 2, 0 /* thinking */, NULL,
                          &rendered, &rendered_len) == SYI_OK) {
    printf("%.*s\n", (int)rendered_len, rendered);
    syi_free(rendered);
}
```

也可以直接：

```c
syi_request *req = NULL;
syi_request_create(model, &go, &req);
syi_request_set_chat(req, msgs, 2, 0, NULL);
syi_request_submit(req);
```

`set_chat` 会立即用 GGUF 的 `tokenizer.chat_template` 渲染并 tokenize；模板不支持
时返回错误，不会静默使用错误格式。`tools_json` 是 OpenAI `tools` 数组的 JSON
字符串（可空）。

Tokenizer 单独使用时：

```c
int32_t *ids = NULL;
size_t n = 0;
syi_model_tokenize(model, "hello", 5, 1, &ids, &n);
/* ... */
syi_free(ids);
```

---

## 8. 错误处理

```c
syi_status st = syi_request_submit(req);
switch (st) {
case SYI_OK:            break;
case SYI_QUEUE_FULL:    /* 队列满；请求仍是 CREATED，可稍后重试 */ break;
case SYI_SHUTDOWN:      /* 模型已 close */ break;
case SYI_INVALID_STATE: /* 已提交过，或 setter 在提交后调用 */ break;
default:                fprintf(stderr, "%s\n", syi_last_error()); break;
}
```

* `SYI_END`/`SYI_AGAIN` 不是错误。
* 异步失败：

```c
syi_request_state state;
syi_request_get_state(req, &state);
if (state == SYI_REQUEST_FAILED) {
    fprintf(stderr, "request failed: %s\n", syi_request_error(req));
}
```

* 每个线程有自己的 `syi_last_error()`；不要跨线程读取。
* `syi_free` 只能释放库返回的指针（tokenize/detokenize/render_chat）。

---

## 9. 调优批量窗口与队列

| 场景 | 建议 |
|---|---|
| 单请求低延迟 | `batch_window_us=0`；`max_batch_rows=1` 可获得更可复现的输出 |
| 多请求吞吐 | `batch_window_us=2000`（默认）；`max_slots` 设为显存允许的最大值 |
| 服务端持续负载 | 窗口可设 0；活跃请求本身会持续合批，窗口只影响冷启动 |
| 内存敏感 | 降低 `max_queue_depth`/`max_queue_tokens`，或让上游限流 |
| 严格隔离 | `max_batch_rows=1`、`batch_window_us=0`、`prefix_cache=0` |

`batch_window_us` 只在模型空闲时生效；繁忙时请求直接进入 FIFO，不会额外等待。

---

## 10. 部署检查

1. **GPU 驱动**：C API 依赖 Level Zero/OpenCL 用户态驱动。安装与 GPU 匹配的
   Intel 驱动（`intel-opencl-rt`、`intel-level-zero-gpu` 或发行版包）。
2. **oneAPI runtime**：`libsycl.so`、oneDNN、编译器运行时必须能被 loader 找到。
   用 `ldd libsycl_infer.so` 检查。
3. **JIT 缓存**：默认 JIT 构建首次调用会编译 kernel；设置
   `SYCL_CACHE_DIR` 可复用。某些驱动上 `SYCL_CACHE_PERSISTENT=1` 会崩溃
   （见 `AGENTS.md`），生产环境优先用 AOT 构建。
4. **显存**：`max_seq`/`max_slots`/`kv_type` 决定 KV 容量；加载失败时先降
   `max_slots` 或 KV 位宽。
5. **进程模型**：推荐一个进程一个模型；多模型会各自复制权重。

---

## 11. 完整示例

见 [design/19-c-api.md §15](../design/19-c-api.md#15-附录最小-c-示例)。
再给一个带流式和批量统计的版本：

```c
#include <stdio.h>
#include <string.h>
#include <sycl_infer/sycl_infer.h>

static void drain(syi_request *req) {
    for (;;) {
        syi_token tok;
        syi_token_init(&tok);
        syi_status st = syi_request_next(req, &tok, 60000);
        if (st == SYI_AGAIN) continue;
        if (st == SYI_END) break;
        if (st != SYI_OK) {
            fprintf(stderr, "next: %s\n", syi_last_error());
            break;
        }
        if (tok.text.size) {
            fwrite(tok.text.data, 1, tok.text.size, stdout);
            fflush(stdout);
        }
    }
    syi_result r;
    syi_result_init(&r);
    if (syi_request_get_result(req, &r) == SYI_OK && r.finish_reason == SYI_FINISH_ERROR) {
        fprintf(stderr, "\nrequest failed: %.*s\n", (int)r.error.size, r.error.data);
    }
}

int main(void) {
    syi_model_options mo;
    syi_model_options_init(&mo);
    mo.model_path = "/path/to/Qwen3.5-0.8B-Q4_K_M.gguf";
    mo.max_slots = 8;
    mo.batch_window_us = 2000;

    syi_model *m = NULL;
    if (syi_model_create(&mo, &m) != SYI_OK) {
        fprintf(stderr, "%s\n", syi_last_error());
        return 1;
    }

    syi_generation_options go;
    syi_generation_options_init(&go);
    go.max_tokens = 32;
    go.temperature = 0.0f;

    const char *ps[] = {"Hello", "1+1=", "The sky is"};
    syi_request *rs[3] = {0};
    for (int i = 0; i < 3; i++) {
        syi_request_create(m, &go, &rs[i]);
        syi_request_set_text(rs[i], ps[i], strlen(ps[i]));
        syi_request_submit(rs[i]);
    }

    for (int i = 0; i < 3; i++) {
        printf("--- %s ---\n", ps[i]);
        drain(rs[i]);
        syi_request_destroy(rs[i]);
    }

    syi_model_stats s;
    s.struct_size = sizeof(s);
    syi_model_get_stats(m, &s);
    printf("\navg_batch=%.2f max_batch=%u generated=%llu\n",
           s.avg_decode_batch, s.max_decode_batch,
           (unsigned long long)s.generated_tokens);

    syi_model_destroy(m);
    return 0;
}
```

---

## 12. 常见问题

**`model load failed: device alloc failed`**

显存不够。降低 `max_seq`、`max_slots`、`kv_type`（`i4:i8`/`i4:i4`），或把
`moe_vram_mb` 降低。27B 需要 `--layer-map` 式双卡分层。

**`SYI_UNSUPPORTED`：KV type / cpu_threads**

v1 的这两个设置是进程级；同一进程已有模型用了不同值。一个进程一个模型，或
让它们一致。

**`avg_decode_batch` 约为 1**

请求没有并发提交，或 `max_batch_rows=1`，或模型是 `qwen4exp`（强制一行）。
按 §5.3 检查。

**`next` 一直 `SYI_AGAIN`**

请求在排队/长 prefill；用 `syi_request_get_state` 区分 `QUEUED`/`PREFILL`。
如果 `state` 是 `FAILED`，读 `syi_request_error`。

**取消后还要等很久**

一次 forward 不可中断；cancel 在下一个 prefill chunk/ decode step 生效。长
prompt 的 prefill 可能到秒级。

**输出和单请求不完全一样**

并发 batch 改变 fp 累加顺序，near-tie 的 greedy argmax 可能翻转；这是引擎
已知边界。需要逐 bit 复现时按 §9 的“严格隔离”设置。
