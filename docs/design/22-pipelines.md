# 设计 22：Pipeline 机制（请求级多步编排）

覆盖结构 `src/pipeline/`。相关文档：
[09-server.md](09-server.md)（OpenAI HTTP/SSE）、[16-router.md](16-router.md)（多模型代理）、
[21-mineru-qwen2vl.md](21-mineru-qwen2vl.md)（MinerU 模型与客户端工具）。

> **状态（2026-11）**：机制已经落地（`src/pipeline/`、worker 的
> `/v1/pipelines` 端点、router 的专用代理与 `models[].pipelines`），
> **registry 默认是空的**：没有内置 pipeline。MinerU 已迁移到
> `tools/mineru/`，作为**独立的测试/参考工具**（`sycl-infer-mineru`），
> 不再属于服务端，也不再是本文的 pipeline 示例。迁移记录见 §11。

---

## 1. 背景与边界

`engine` 提供的是一次前向/一次生成的原子能力：prefill、decode、`generate`、
`generate_mm`。有些请求需要在进程内完成**多步编排**，而这些步骤无法或不适合通过
OpenAI 兼容边界表达，例如：

* 同一份 KV/前缀缓存要在多步之间复用；
* 逐步 verify / rollback（MTP、DFlash 类）；
* 需要独占 `engine::d_img_embd`、`vision_model` 或某个跨步状态；
* 需要在 server 的并发/取消/资源账本内完成多步，而不是由客户端多次往返。

`pipeline` 就是这类编排的扩展点。它不是模型，不是推理后端，也不是“把客户端脚本
搬进服务端”的容器。

### 1.1 与 OpenAI API 的边界

* **`GET /v1/models` 只列出模型**，不列出 pipeline，也不做 model alias。
* **`POST /v1/chat/completions` / `/v1/completions` 保持标准语义**：`model` 是模型
  原始 id（worker 会校验它，未知 id 返回 404，因此 pipeline id 不会意外成为别名）；
  调用方提供 prompt；返回模型原始输出。服务端不解析文档、不拼业务 JSON。
* pipeline **只通过专用端点** `GET /v1/pipelines` 与
  `POST /v1/pipelines/{id}/run` 暴露；它不是 OpenAI 标准的一部分。
* 模型路由与 pipeline 路由是两张独立的表；pipeline id 不会出现在 `model` 路由中。

### 1.2 图片输入不属于 pipeline

图片解码、预处理、视觉塔、M-RoPE 和 `engine::generate_mm` 都是**模型/mm 层**能力。
只要 `--mmproj` 已加载且模型 profile 声明 `vision = qwen2vl`、`spec.multimodal = true`，
`/v1/chat/completions` 就可以直接带 `image_url` 调用该模型，不经过任何 pipeline。

pipeline 可以**使用**这条图片路径，但它不拥有图片输入机制；把“裁图/换 prompt/多次
调用”当成 pipeline 的理由是不成立的。

---

## 2. Pipeline 的准入规则

一个多步流程只有满足至少一条，才值得放进服务端 `src/pipeline/`：

1. **需要进程内引擎状态**：KV/前缀缓存复用、逐 token verify、MTP/DFlash 状态、
   多模态 embedding 缓冲的独占访问等，无法通过 OpenAI 边界安全表达；
2. **有实测的服务端性能收益**：避免 base64/HTTP/重复图片解码或重复视觉编码，
   且该收益在目标负载上可测量；
3. **必须与 server 的并发/取消/资源账本共用同一套机制**，客户端方案无法保证。

### 2.1 反例：MinerU 不是 pipeline

`tools/mineru/`（原 `src/mineru.*`）是 `opendatalab/mineru-vl-utils` 的
`MinerUClient.two_step_extract` 的 C++ 复刻（prompt、block 语法、JSON 形状均来自
上游，2026-11-02，commit `955045d`）。它的全部工作是：

1. 把整页图 resize 到固定边长；
2. 用 layout prompt 调一次模型；
3. 解析版面 block；
4. 逐块裁剪/旋转，用类型 prompt 再调模型；
5. 汇总 JSON。

这些步骤只用了**公开的模型能力**：图片输入 + 文本输出 + 采样参数。它们不需要
KV/前缀缓存复用、不需要 verify、不需要 engine 内部状态。按 §2 的规则，MinerU
**默认不满足任何一条**，因此：

* 它应放在 `tools/mineru/`，作为一个**独立的测试/参考客户端**；
* 服务端只提供 qwen2vl 的 `/v1/chat/completions`；
* 文档流水线（RapidDoc / `mineru-vl-utils`）在客户端完成两步逻辑；
* 只有将来实测出“进程内复用视觉塔/避免 base64”的收益足够大，才重新评估是否把它
  注册成可选的服务端快路径；即便如此，也要固定上游版本并做 parity 测试。

---

## 3. 分层与目录

```
src/model/        qwen2vl loader / profile / hparams          （不动）
src/mm/           vision tower、image/video/audio 预处理       （补共享 runtime）
src/engine/       单次/批量 forward、generate_mm               （不动）
src/pipeline/     请求级编排机制（当前无内置 pipeline）
  pipeline.h        interface / request / sink / descriptor
  registry.{h,cpp}  id -> pipeline、availability、列表
src/server/       HTTP/SSE；只依赖 pipeline 接口，不知道具体流程
tools/mineru/     MinerU 参考客户端（独立二进制，不属于 server）
  mineru.h / mineru.cpp / layout.{h,cpp}
tests/pipeline/   registry 与路由的 hermetic 测试
```

依赖方向：

```
server ──> pipeline ──> engine ──> model
                    └─> mm ──────> model
tools/mineru ──> engine + mm（或可选地走 OpenAI HTTP）
model 不依赖 pipeline / engine / server / tools
```

`tools/` 不是库的一部分：它不进入 `sycl-infer` 主二进制，也不被 server 引用。
它可以链接 `sycl_infer_core` 直接调用 engine（当前做法，便于 GPU 基准和参考输出），
也可以作为 HTTP 客户端验证公开 API；两种模式共享同一套 prompt/解析代码。

---

## 4. Pipeline 接口

### 4.1 Descriptor

```cpp
struct pipeline_desc {
    const char * id;                 // 例如 "stateful_rag"
    const char * name;
    model_profile::vision_kind requires_vision; // none = 不需要视觉
    bool accepts_image = false;
    bool accepts_video = false;
    bool accepts_audio = false;
    const char * output_formats;     // "json,markdown,text"
    const char * options_schema;     // 可选：JSON schema 文本
};
```

server 不写 `if (id == "...")`；它只读 descriptor 做可用性过滤和输入校验。

### 4.2 请求与输出

pipeline 不依赖 httplib，也不依赖 OpenAI JSON：

```cpp
struct pipeline_request {
    std::vector<chat_msg> messages;    // 已解析的对话（可选）
    std::vector<media_part> media;     // 已下载的原始媒体（未解码）
    gen_params params;                 // max_tokens / temperature / ...
    std::string options_json;          // pipeline 私有选项
    bool stream = false;
};

struct pipeline_usage {
    long long prompt_tokens = 0;
    long long completion_tokens = 0;
    long long reasoning_tokens = 0;
    long long cached_tokens = 0;
};

struct pipeline_sink {
    std::function<void(std::string_view)> on_text;   // 内容增量
    std::function<void()> on_progress;               // 可选
    std::function<void(const pipeline_usage &, std::string finish_reason)> on_done;
};
```

### 4.3 对象与注册表

```cpp
class pipeline {
public:
    virtual ~pipeline() = default;
    virtual const pipeline_desc & desc() const = 0;
    virtual bool available(const model_profile & prof) const = 0;
    virtual void run(const pipeline_request &, pipeline_sink &) = 0;
};

class pipeline_registry {
public:
    bool add(std::unique_ptr<pipeline>); // false on duplicate/empty id
    pipeline * find(const std::string & id) const;
    std::vector<const pipeline *> available(const model_profile &) const;
};
```

* `available()` 只读模型 profile（`engine.m.prof`），不读 device profile，也不做逐卡
  判断。接口直接收 `model_profile`（而不是 `engine&`），这样 registry 与它的
  hermetic 测试不依赖 engine/SYCL。
* 注册在 `serve()` 启动时做一次。
* pipeline 如果使用视觉塔，必须持有 server 已加载的 `vision_model&`，不能重新
  `load/upload` 一份（1.3 GB VRAM，且与普通图片 chat 冲突）。

---

## 5. 生命周期与并发

1. **视觉塔共享**：server 已经把 `--mmproj` 加载到 `mm_server::vm`；pipeline 复用
   这个实例。
2. **媒体 gate**：使用 `engine::d_img_embd` 的 pipeline 在整个生命周期持有
   `media_permit`，与普通多模态请求互斥。
3. **engine 锁**：每次 `engine::generate_mm`/`generate` 自己会加 `engine::mtx`；
   pipeline 不得在两步之间持有它，否则会阻塞整条服务。
4. **资源**：按请求复用 pipeline 内的缓冲，不要每次重新分配塔权重。
5. **取消**：`generate` 的 token 回调是取消点；pipeline 至少在每一步边界检查
   SSE 断开/sequence 取消。

---

## 6. Worker 端点

### 6.1 `GET /v1/pipelines`

只列**可用**的 pipeline：

```json
{
  "object": "list",
  "data": [
    {
      "id": "<pipeline-id>",
      "object": "pipeline",
      "name": "...",
      "input": ["text"],
      "output": ["application/json"]
    }
  ]
}
```

当前没有内置 pipeline 时返回空列表。

### 6.2 `POST /v1/pipelines/{id}/run`

请求：

```json
{
  "input": {"image_url": "data:image/png;base64,..."},
  "options": {},
  "stream": false
}
```

非流式响应：

```json
{
  "id": "pipeline-...",
  "object": "pipeline.result",
  "pipeline": "<id>",
  "model": "<GGUF general.name>",
  "output": {"format": "json", "content": "..."},
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

流式响应：

```
data: {"object":"pipeline.delta","pipeline":"<id>","delta":{"content":"..."}}

data: {"object":"pipeline.done","pipeline":"<id>","finish_reason":"stop","usage":{}}

data: [DONE]
```

错误：未知 pipeline → 404 `pipeline_not_found`；输入不满足 descriptor → 400；
SSE 取消沿用 `serve_sse_chunked` 的断连检测。

### 6.3 明确不改的端点

* `GET /v1/models`、`GET /v1/models/{id}`：只有模型。
* `POST /v1/chat/completions`、`POST /v1/completions`：标准语义，不认识 pipeline id，
  也不暴露任何业务专用字段。

---

## 7. Router 端点

Router 不把 pipeline 当模型，也不加 alias。它只新增专用代理：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/v1/pipelines` | 聚合各 worker 的可用 pipeline；重复 id 是配置错误 |
| POST | `/v1/pipelines/{id}/run` | 找到 owning worker，原样转发（含 SSE） |

归属关系来自 router 配置的 `models[].pipelines`（不是模型别名）：

```json
{
  "models": [
    {
      "id": "some-model",
      "pipelines": ["<pipeline-id>"],
      "gguf": "/path/to/model.gguf"
    }
  ]
}
```

* `pipelines` 省略时该 worker 不提供 pipeline。
* 启动时只把声明的 id 加入路由表；worker 的 `GET /v1/pipelines` 没有列出该 id 时，
  建议启动失败并打印原因。
* pipeline 请求对 `model` 字段无意义；router 不改写它。
* 取消链与现有代理一致；远程 worker 走同一条路径。

---

## 8. 新增一条 pipeline

1. 先过 §2 的准入规则，并在设计文档里写明满足哪一条、测量条件是什么。
2. 在 `src/pipeline/<name>/` 实现 `pipeline` 接口和 descriptor。
3. 纯逻辑（语法、格式转换、prompt 模板）放进不依赖 engine 的 TU。
4. 在 `serve()` 的 registry 构造处注册。
5. 如果 router 要暴露，给对应模型配置加 `pipelines: ["<name>"]`。
6. 补 hermetic 测试（registry、纯逻辑、server 路由）；端到端测试放 GPU/model 档。
7. 更新 `docs/README.md`、`AGENTS.md` 的扩展步骤和本文。

---

## 9. MinerU：客户端工具

`src/mineru.*` 已迁移到 `tools/mineru/`，并构建成独立二进制
`sycl-infer-mineru`（不进入 `sycl-infer` 主二进制）：

```
tools/mineru/
  mineru.h / mineru.cpp   两步编排、图像几何、JSON 输出
  layout.{h,cpp}          prompt 与版面语法（纯函数）
  CMakeLists.txt 或主 CMake 的一个 SI_TOOLS target
```

* 保留现有能力并补齐运行参数：`--model --mmproj --image --ctx --device
  --layer-map --kv-type --layout-side --only-type --max-tokens --debug`。
* 默认直接链接 `sycl_infer_core`（参考输出、GPU 基准）。
* 可选 `--api <host:port>` 模式（尚未实现）：走 `/v1/chat/completions` 的图片输入，
  验证公开 API；两种模式共享 prompt/解析代码。
* 它不进入 `serve()` 的 pipeline registry，不出现在 `/v1/pipelines`，也不影响
  `/v1/models` 和 `/v1/chat/completions`。
* 由于是上游逻辑的复刻，固定 parity 基准：同一页面/模型/参数的输出与上游
  `mineru-vl-utils` 一致；上游更新时要显式复审，而不是自动跟随。

### 9.1 主 CLI 的变化

`sycl-infer mineru` 命令移除；`main.cpp` 只保留模型推理命令（`serve` / `gen`）。
需要 MinerU 的用户运行 `sycl-infer-mineru`，或在自己的客户端里调用 OpenAI API。

---

## 10. 测试

| 测试 | 层 | 覆盖 |
|---|---|---|
| `test_pipeline_registry` | hermetic | id 查找、重复 id、`available()` 过滤、空列表 |
| `test_pipeline_route` | hermetic/CPU | `/v1/pipelines` 列表、`/run` 非流式、SSE 拼接、404/400、`/v1/models` 不含 pipeline |
| `test_mineru_layout` | hermetic | `tools/mineru` 的 prompt、版面语法、bbox/角度/IoU/去重 |
| `test_pipeline_chat_unchanged` | CPU | 可选：pipeline id 在 `/v1/chat/completions` 上按普通模型处理（未知模型 404），不触发 pipeline |

---

## 11. 迁移计划

1. **移动 MinerU**（已完成）：`src/mineru.{h,cpp}` → `tools/mineru/`；拆出
   `layout.{h,cpp}`；CMake 新增 `sycl-infer-mineru`，从 `sycl-infer` 的源列表中删除
   `src/mineru.cpp`；`main.cpp` 删除 `mineru` 命令和 `--mineru-*` 参数。
2. **抽 `src/mm/vision_runtime`**（未做，仍是后续项）：统一 mmproj `load →
   runnable → proj_dim → upload → image_preproc_cfg`，server 与 tools 共用（见 03/10 文档）。
3. **建 `src/pipeline/`**（已完成）接口与 registry；不放任何内置 pipeline。
4. **Worker 端点**（已完成）：`GET /v1/pipelines`、`POST /v1/pipelines/{id}/run`。
5. **Router**（已完成）：`models[].pipelines` + 两条代理路由。
6. **测试**（已完成 hermetic 部分）：§10 的 registry/端点/layout 用例。MinerU
   的 model/GPU parity 用例仍待补。
7. 同步 `CMakeLists.txt`、`README.md`、`docs/architecture.md`、`AGENTS.md`、
   `docs/design/21-mineru-qwen2vl.md`。

### 11.1 非目标

* 不把 MinerU 放进 `src/pipeline/`、`src/server/` 或 `/v1/models`。
* 不加 model alias；pipeline id 只用于专用端点。
* 不把上游的 prompt/grammar/JSON 当成引擎职责；它们是工具/客户端逻辑。
* 不在 pipeline 里复制 OpenAI 响应整形；`chat.completion` 的 reasoning/tool_calls/
  usage 仍由 server 的现有代码生成。
