# 设计 09：HTTP 服务与连续批处理

覆盖 `src/server/server.{h,cpp}`、`scheduler.{h,cpp}`、`chat.{h,cpp}`、`chat_template.{h,cpp}`、
`response_parser.{h,cpp}`、`chat_util.h`，以及 `engine` 的采样接口。

---

## 1. 组件与数据流

| 层 | 文件 | 职责 |
|---|---|---|
| HTTP | `server.cpp` | httplib 服务与路由、SSE 发送、CORS、stop 过滤、视觉与音频 tower 调度、`mtp_direct` 路由 |
| 请求解析 | `request.{h,cpp}` | OpenAI 请求体 → 引擎入参（纯函数，见 §2.4） |
| 媒体获取 | `media_fetch.{h,cpp}` | 一个 media part → 字节：`data:`/内联 base64/url_fetch 策略 |
| JSON 序列化 | `json_util.h` | `dump_json`（容忍非法 UTF-8，编码为 U+FFFD 而非抛异常） |
| 批处理 | `scheduler.cpp` | `sequence` 生命周期、连续批处理、持有 `engine::mtx` |
| Chat 模板 | `chat.cpp` / `chat_template.cpp` | minja Jinja 渲染 + 内置 ChatML 回退 |
| 响应构建 | `response.{h,cpp}` | OpenAI 响应体与 SSE 帧（纯函数，见 §2.6/§2.7） |
| 响应拆分 | `response_parser.cpp` | `reasoning_content` / `content` / `tool_calls` 流式状态机 |
| 流式辅助 | `chat_util.h` | UTF-8 边界安全的 token piece 缓冲 |
| 采样 | `sampler.cpp` | 采样算法 |
| 引擎 | `engine.h` | prefill/decode/logits、前缀缓存 API、`generate`（含内部 MTP 分流）、`generate_mm` |

三条请求路径（`handle_chat` 的分派顺序，`server.cpp:1597`→`1717`→`1750`）：

* **纯文本**（无媒体 part，默认）：在 httplib 工作线程渲染模板并分词 → `scheduler::submit` → 连续批处理
  → 输出队列。`n > 1` 时每个 choice 一个生产者线程（`server.cpp:1784-1788`）。
* **多模态**（含任意媒体 part）：解码/预处理媒体 → 视觉/音频 tower → 在生产者线程 `engine::generate_mm`，
  **绕过调度器与前缀缓存**，`n` 个 choice 顺序生成。
* **MTP**（`PF_MTP_SERVER=1` 且引擎开了 `--mtp`）：在生产者线程 `engine::generate`，**同样绕过调度器**；
  见 §2.4 的 `mtp_direct` 门控与 §3.6。

后两条路径与批处理互斥的原因写在 §3.6：MTP 是单序列循环，运行它等于在整个生成期占住 `engine::mtx`。

---

## 2. HTTP 层

### 2.1 服务设置

`serve(engine&, server_config)`：

* `scheduler sched(e); sched.start();`（`server.cpp:1456-1457`），栈上对象，`serve` 返回时析构。
* `httplib::Server srv;`，读/写超时 3600 s，`set_payload_max_length(64 MiB)`（base64 图片需要）
  （`server.cpp:1513-1516`）。
* `server_config`（`server.h:8-20`）的 `n_threads`（`server.h:12`）**从未被读取**——全仓库唯一出现处是这行
  定义，`main.cpp:386-394` 也不填它；httplib 使用自己的默认线程池
  （`max(8, hardware_concurrency()-1)`，`third_party/httplib.h:183-190`，由
  `Server::Server()` 在 `httplib.h:12787-12788` 构造）。没有安装自定义 task queue。
* `model_id` 默认 `"qwen3.5-0.8b"`（`server.h:9`），但 `/v1/models` 实际用 GGUF `general.name` 覆盖
  （`server.cpp`）；`host` 默认 `127.0.0.1`、`port` 默认 8080（`server.h`，由
  `--host`/`--port` 填充）。

### 2.2 CORS

每个响应设置 `Access-Control-Allow-Origin: *`、`Allow-Headers: *`、
`Allow-Methods: GET, POST, OPTIONS`（`cors()`，`server.cpp:1518-1522`）；`srv.Options(".*", ...)` 返回 204
（`server.cpp:1523-1526`）。调用 `cors(res)` 的处理器：`/v1/models`（`server.cpp:1541`）、
`/v1/models/{id}`（`server.cpp:1545`）、`/v1/chat/completions`、`/v1/completions`（两个 handler 的第一句）。
**只有 `GET /health` 不设 CORS**（`server.cpp:1536-1538`）。

SSE 响应在此之上再加 `Cache-Control: no-cache` 与 `Connection: keep-alive`（`cors_sse`，
`server.cpp:1555-1558`），即流式响应同时带 CORS 与这两个头。

### 2.3 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | `{"status":"ok"}`；**唯一不带 CORS 的端点** |
| GET | `/v1/models` | OpenAI 列表；id 取 GGUF `general.name`（回退 `model_id`），附 `meta` |
| GET | `/v1/models/{id}` | 单个模型（`srv.Get(R"(/v1/models/(.+))")`，`server.cpp:1544`）；id 既可等于 GGUF `general.name` 也可等于 CLI 默认 `model_id`，否则 404 |
| POST | `/v1/chat/completions` | 流式 + 非流式；reasoning_content、工具调用/结果、`n`、stop、`stream_options.include_usage`、图片输入 |
| POST | `/v1/completions` | 纯文本补全；string/string[]/token 数组 prompt、`n`、`echo`、`suffix`、流式 + 非流式 |
| OPTIONS | `.*` | 204 + CORS |

无鉴权，无 `/v1/embeddings`、`/tokenize`。

### 2.4 请求解析

解析全部在 `request.{h,cpp}` 里，是**纯函数**（JSON 进、普通值出）：不碰 httplib、不碰 engine 状态、
不依赖全局量。这正是 `tests/server/test_request_parse.cpp`（hermetic，无模型无 socket）能存在的原因——
这些别名与优先级规则此前只能在真实 HTTP 请求里触达，等于没有测试。其中两个函数刻意只收它们真正需要
的字段：`parse_completion_prompts(tokenizer &, ...)`（不收 engine）与
`prompt_rejected_json(int max_seq, size_t)`（不收 engine，也不设状态码、不打日志）。

* JSON 解析失败 → 400 `{"error":{"message":"invalid json","type":"invalid_request_error"}}`
  （chat: `server.cpp:1576-1582`；completions: `server.cpp:1796-1802`）。
* `parse_messages` 把 `role` 默认 `"user"`；`content` 可以是字符串、`null` 或 part 数组。part 按
  `type` 分派（`server.cpp:532-571`）：`image_url`/`image` → IMAGE、`video_url`/`video` → VIDEO、
  `input_audio`（读 `data`+`format`）、`audio_url` → AUDIO、带 `text` 的 → TEXT（文本**追加**到
  `cm.content`）；未识别的 part 被静默丢弃。每个媒体 part 同时往 `media` 压一条 `media_part`（按出现
  顺序，这就是 §6 的 `order`），并往 `cm.parts` 压一个占位 `chat_part`。
  assistant 消息还可带 `reasoning_content`（或别名 `reasoning`）、`tool_calls`（OpenAI 形状，
  `function.arguments` 为字符串或对象）或旧式 `function_call`（解析成一个 tool_call）；
  `tool` 消息读取 `tool_call_id`/`name`（`server.cpp:582-630`）。
  若某消息**没有任何媒体 part**（`has_image`，`server.cpp:510`、该位对四种媒体都会置起），
  则清空 `parts` 走纯文本快速路径（`server.cpp:631-633`）。
* `thinking`（`parse_thinking`）：`chat_template_kwargs.enable_thinking` 优先，其次平铺的
  `enable_thinking`/`thinking`，最后 `reasoning_effort`（非 `"none"` 即开启）。
* `tools_json`（`parse_tools_json`）：把请求 `tools` 数组回传给模板；`tool_choice == "none"`
  时清空，指定函数对象时只保留该函数，其它字符串值（`auto`/`required`/`any`）原样放行全部工具。
  `parse_tools = !tools_json.empty()`（`parse_tools_json`，`request.cpp`）——它同时决定 response_parser 是否拆工具调用。
* `gen_params` 来自 `parse_params`，`stops` 来自 `parse_stop`（字符串或字符串数组，数组最多 16 条，
  `kMaxStops`），`n` 来自 `parse_n`（1..16，`kMaxN`）；两者与 `kMaxJobs` 都在 `request.h`。
* `/v1/completions` 另有 `parse_best_of`（默认等于 `n`，clamp 到 1..16）、
  `parse_include_usage`（只读 `stream_options.include_usage`）、
  `echo`、`suffix`（`server.cpp:1820-1821`）。
* `model` 只被回显进响应（`body.value("model", model_id)`，`server.cpp:1595`）；请求里写错 model 名
  **不报错**。`created` 是每次请求 `time(nullptr)`。
* **`seed`**：`gp.seed == 0` 表示“不指定”，此时每条序列用 `steady_clock` 计数播种
  （`scheduler.cpp:55`；单序列路径用 `std::random_device{}()`，`engine.cpp:2371`），即不可复现。
  非 0 时第 `c` 个 choice 用 `seed + c`（`gp_for_choice`，`server.cpp:738-744`）。
* 请求若满足 `mtp_direct(e, gp, n, gp.logprobs)` 则不进调度器，改走 §3.6 的 MTP 路径
  （`server.cpp:1717`）。

参数映射（`parse_params`）：

| JSON key | 字段 | 默认 |
|---|---|---|
| `max_tokens` / `max_completion_tokens` | `max_tokens` | 256（≤0 强制 256） |
| `temperature` / `top_p` / `top_k` / `min_p` | 同名 | 1.0 / 0.95 / 40 / 0.0 |
| `repetition_penalty` / `repeat_penalty` | `repeat_penalty` | 1.0 |
| `repeat_last_n` | `repeat_last_n` | 64 |
| `presence_penalty` / `frequency_penalty` | 同名 | 0.0 |
| `ignore_eos` | `ignore_eos` | false |
| `seed` | `seed` | 0 |
| `logit_bias` | `logit_bias`（map，clamp ±100） | 空 |
| `logprobs`（chat: bool / completion: int） + `top_logprobs` | `logprobs` / `top_logprobs`（≤20） | 关闭 |

`/v1/completions` 的 `prompt` 支持字符串、字符串数组、token id 数组、token id 数组的数组
（`parse_completion_prompts`，`request.cpp`；缺失/空/类型不识别都退化成一条空 prompt）；
`echo` 把 prompt 文本前置，`suffix` 追加到生成文本后。`best_of > n` 时每个 prompt 生成 `best_of`
条候选，按 chosen token 的 logprob 之和（`choice_out::score`）`stable_sort` 降序后取前 `n`
（`server.cpp:1850-1867`）；评分走 `gp.need_score`，它让调度器即使在客户端没要 logprobs 时也算
chosen token 的 logprob（`sampler.h:29-35`）。

### 2.5 长度拒绝

`request.cpp` 的 `prompt_rejected_json(max_seq, n)` 返回应 400 的响应体，或 `""` 表示接受；判定理由是
否则 `scheduler::admit` 会静默退役超长 prompt，返回空的 200 流。`server.cpp` 的 `reject_prompt` lambda
把状态码、content type 与 `[http] 400 context_length_exceeded` 日志留在传输层（策略与日志分开，
于是策略可被单测断言）。对 chat 文本、每个 completion prompt、以及**扩展后**的多模态 prompt 都调用
（三个 handler 内各一处）。

其它 400（`bad_request`，消息即 `error.message`，`type` 一律 `invalid_request_error`）：

| 条件 | 消息 | 位置 |
|---|---|---|
| 请求含任意媒体 part 但没有可用的 `--mmproj` | `image/video input requires --mmproj` | `server.cpp:1597-1601` |
| 请求含音频 part 但没有可用的 `--audio-mmproj` | `audio input requires --audio-mmproj` | `server.cpp:1602-1611` |
| 媒体下载/解码/预处理失败、prompt 装配失败（多模态 try 块） | 解码器给出的 `err` | `server.cpp:1616-1658` |
| `best_of < n` | `best_of must be >= n` | `server.cpp:1832-1835` |
| `stream && best_of > 1` | `best_of is not supported with stream` | `server.cpp:1836-1839` |
| `prompt 数 × max(n, best_of) > 64`（`kMaxJobs`） | `prompt count * n (or best_of) exceeds the server limit` | `server.cpp:1840-1844` |

最后一条在 `if (!stream)` **之前**，所以流式与非流式都受限；流式时 `best_of` 必为 1，上限实际就是
`prompt 数 × n <= 64`。

**音频请求同时需要两个 mmproj**：`!media.empty()` 就先查 `mm.ready`，所以只加载了
`--audio-mmproj` 而没加载 `--mmproj` 的部署，纯音频请求也会拿到
`image/video input requires --mmproj`。`input_audio.format` / `audio_url.format` 被解析并存进
`media_part::format`（`parse_messages` 填，`media_fetch.cpp` 不用），但解码端不用它——`mm_audio_decode_bytes` 自己嗅探
RIFF/WAV 头，所以写错 `format` 不会导致解码失败。

### 2.6 非流式响应

响应体与 SSE 帧的构建都在 `response.{h,cpp}`（纯函数，无 httplib），随它一起搬走的还有两个**响应状态**
类型：`lp_tracker`（把采样 token 的 logprob 配对到它最终落入的 parser 分片）与 `choice_out`
（一个 choice 产出的全部内容，流式与非流式共用）。因此 `tests/server/test_response_json.cpp`
（hermetic）能直接断言线上格式——`reasoning` 与 `content` 的分流、`tool_calls` 的 index 递增、
logprob 只挂载一次、usage 的各计数器——这些此前只能靠驱动一次真实生成来观察。两个 logprob 构建器需要
真 tokenizer，钉在 `test_tokenizer` 里。

* chat：`object:"chat.completion"`，每个 choice `{message:{role,content[,reasoning_content][,tool_calls]},finish_reason,logprobs}`，
  含 `usage`；`n` 个 choice 的 token 统计求和。仅有工具调用时 `content` 为 `null`；未请求 logprobs 时为
  `null`（`server.cpp:1756-1760`）。
* completion：`object:"text_completion"`，choice `{index,text,finish_reason,logprobs}`；多个 prompt/`n`
  展开为多个 choice，`index` 顺序编号（`server.cpp:1873-1874`）。
* 非流式路径同样先过 stop 过滤与 `response_parser`，因此与流式输出一致。
* **多模态与 MTP 路径的响应形状相同，但 `logprobs` 恒为 `null`**（`server.cpp:1672`、`1729`）——
  这两条路径不走调度器，拿不到 per-token 的采样细节；`mtp_direct` 因此也把 `logprobs` 列为排除条件。
* `usage` 由 `usage_json(prompt, completion, reasoning, cached)` 生成（`response.cpp`）：
  `prompt_tokens_details.cached_tokens` = 前缀缓存命中的 prompt token 数（`sequence::reused`）、
  `completion_tokens_details.reasoning_tokens` = 计入 `reasoning_content` 的生成片段数，
  外加 DeepSeek 风格的 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`（同义，miss = prompt -
  cached），`prompt_tokens_details.audio_tokens` 与
  `completion_tokens_details.{audio,accepted_prediction,rejected_prediction}_tokens` 恒为 0 占位。
  `/v1/completions` 不做 reasoning 拆分，所以那里 `reasoning_tokens` 传 0（`server.cpp:1886`）。

### 2.7 流式 SSE

传输与取消机制在 `src/server/sse.h`（`sse_queue` / `sse_session` / `serve_sse_chunked`），
单独成头文件是为了让取消契约**不依赖模型**即可测（见 `tests/server/test_sse_cancel.cpp`）：
它的失败症状（对着死 socket 继续解码、worker 线程卡在 `join()`）在生成测试里完全不可见，
因为两种情形的输出文本完全一样。

通用形态：

* 响应头 `text/event-stream`（`set_chunked_content_provider` 的第一个参数）加上 CORS、
  `Cache-Control: no-cache`、`Connection: keep-alive`（`cors_sse`）；
* 用 `set_chunked_content_provider`，内容 reader 从 `sse_queue` 取，队列结束调 `sink.done()`；
  资源释放回调先处理断连、再 join 生产者线程（`serve_sse_chunked`）。
* 每个事件是 `"data: " + dump_json(j) + "\n\n"`；`dump_json` 用 `error_handler_t::replace`
  防止坏 UTF-8 抛异常。
* 纯文本 chat 每 choice 一个生产者线程；completion 是 `prompt 数 × n` 个线程；
  多模态与 MTP 路径只有**一个**线程在同一个循环里顺序跑完 `n` 个 choice。最后一个调用
  `choice_done` 的 choice 负责发 usage chunk（若 `stream_options.include_usage`）与
  `data: [DONE]`，然后 `q->finish()`。usage chunk 的 JSON 由 `response.cpp` 的 `sse_usage_frame`
  经 `sse_session::usage_emitter` 注入——这样 `sse.h` 不必依赖那些 file-local 的 JSON helper。
* 请求 logprobs 时，每个 content/text chunk 附带 `choices[].logprobs`（chat 为
  `{content:[...],refusal:null}`，completion 为 `{tokens,token_logprobs,top_logprobs,text_offset}`）。
* **异常**：chat 的生产者线程 catch 后发一个
  `delta:{"error":{"message":"generation failed","type":"server_error"}}` 且 `finish_reason:"stop"`
  的 chunk（`sse_error`，多模态/MTP 是 `multimodal generation failed` /
  `mtp generation failed`）；completion 的 catch **不发任何事件**，直接 `choice_done(0,0)`，
  客户端只会看到流提前结束。

#### 背压：两个队列都有上限

`sequence::out_q` 与 `sse_queue::items` 的唯一消费者是 HTTP writer 线程，所以客户端读得比引擎慢时
它们是**无界缓冲**。两边都不能阻塞：调度器循环是所有序列**共享**的单线程（阻塞等于一个慢客户端
冻住全部请求），而 MM/MTP 生产者在整个 forward 期间持有 `engine::mtx`（阻塞等于拖慢全部请求）。
两边也都不能丢事件——丢掉一个 token 就是静默截断响应。所以：

* 调度器路径：`sequence::kMaxOutQueue = 256` 是**暂停阈值**，decode 组批时把满的序列排除在本轮
  batch 之外（它保住 slot 与 KV，drain 之后自动恢复，§3.4 组批处 `out_queue_full()`）。
  这个保证属于**调用点**，所以队列自身另有一个 `kHardOutCap = 2*kMaxOutQueue`，由 `push_token`
  强制：越过它就取消该序列（结束请求而不是截断它），这样将来任何漏掉暂停判断的新生产者都不会把
  无界缓冲重新引回来。
* 流式路径：`sse_queue::kMaxQueue = 4096`，越过即自取消（`backpressure_cancelled()` 可区分它与
  真正的断连）。drain 不掉 4096 个事件（几分钟 decode）的客户端与断连者同等对待。
  `sse_session::cancelled()` 因此**同时**查 `stop` 和队列状态，否则生产者在溢出时收不到信号。

`test_sse_cancel` 的 `test_backpressure_bounds` 钉住这些数字（到阈值时 full、越过硬上限后被取消、
drain 一个即恢复、溢出取消能传达到 session 的生产者）。

#### 断连取消

唯一的断连信号是 httplib 的 `ContentProviderResourceReleaser` 那个 `bool`：写失败与对端挂断
都会让它为 false（httplib 只在 `write_content_with_provider` 返回成功后置
`content_provider_success_`）。三处消费它：

1. **`sink.write` 返回 false** → `sc->cancel()` 并让 provider 返回 `false`（转成
   `Error::Canceled`），否则 httplib 会把这次断流记成成功；
2. **releaser 的 `ok == false`** → `sc->cancel()`。这是覆盖「客户端在两个 token 之间消失、
   一次 write 都没发生」的主路径；
3. `cancel()` 置 `sse_session::stop`、`sse_queue::cancel()`（唤醒可能阻塞在 `pop` 的 writer，
   并丢弃已排队的事件）、并 `cancel()` 本响应登记过的所有 `sequence`。

生产者侧每个 token 边界检查一次 `sc->cancelled()`：scheduler 路径靠 `pop_token` 返回 false
（`sequence::cancelled` 同时唤醒等待者），直连 engine 路径靠回调返回 `false`
（`!sf.stopped && !sc->cancelled()`）——回调返回值是一次 forward 内部**唯一**的取消点，
所以被取消的生成最多多跑完当前那一步 decode（27B 上约 65 ms）。

`sequence::cancelled` 让调度器在下一轮迭代就 `retire`（`admit`、prefill 轮、decode 组批三处），
slot 与 KV 块立刻归还，而不是等 `max_tokens` 跑完。`sse_session::track` 会在注册后复查 `stop`，
覆盖「取消先于 `submit` 到达」的时序。

**未覆盖**：非流式路径在 handler 线程内同步生成，httplib 在 handler 返回前不给任何断连信号，
因此那里无法取消（一次非流式请求必然跑完）。这是 httplib 的限制，不是本实现的取舍。


chat chunk schema：

```json
{"id":"chatcmpl-N","object":"chat.completion.chunk","created":T,"model":"...",
 "choices":[{"index":0,"delta":{"reasoning_content":"..."},"finish_reason":null}]}
{"choices":[{"index":0,"delta":{"content":"..."},"finish_reason":null}]}
{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_...","type":"function",
  "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":null}]}
```

每个 choice 先发一个 `delta:{"role":"assistant"}`（`server.cpp:1037-1039`）；随后是
reasoning/content/tool_calls 片段（stop 处截断，`tool_calls[].index` 从 0 递增）；再发带
`finish_reason` 的最终 chunk（chat 为 `delta:{}`，completion 为 `text:""`）；
最后（可选 usage +）`data: [DONE]`。usage chunk 的 `choices` 是**空数组**，只有 `usage`
（`server.cpp:1005-1010`）。completion 的流式没有 role chunk，`echo` 的 prompt 文本作为第一个
`text` chunk 发出（`server.cpp:1260-1262`），`suffix` 在生成结束后单独发一次
（`server.cpp:1281`）。

### 2.8 回复拆分（`response_parser`）

`response_parser(thinking, parse_tools, emit)` 把生成文本拆成三类片段：

* **reasoning**：`thinking` 为真时，初始处于推理状态，`</think>` 之前（并去掉前导 `<think>`）的文本
  作为 `reasoning_content`；遇到 `</think>` 后转入 content。`thinking` 为假则不产生 reasoning。
* **content**：其余文本；`<tool_call>` 标记不作为 content 输出。
* **tool_call**：解析
  `<tool_call><function=NAME><parameter=K>V</parameter>…</function></tool_call>`，参数值去掉
  首尾各一个换行后按 JSON 反序列化（失败则作字符串，所以 `"hello world"` 仍是字符串而非数字），
  并回退支持 `{"name":…,"arguments":…}`（或 `parameters`）的 JSON 形式；`arguments` 用
  `nlohmann::ordered_json::dump()` 序列化为字符串，`id` 由 `call_%08llx%04llx` 计数器生成
  （`response_parser.cpp:96-118`、`122-189`）。标记是**纯 ASCII** 的 `<tool_call>` /
  `</tool_call>`，与模板/内置渲染器发出的字节一致（`chat.cpp:54`、`chat.cpp:65`）。

`feed(piece)` 只发射完整片段：piece 末尾可能是 `</think>`/`<tool_call>` 的前缀时会被暂存
（`partial_marker_suffix`，`response_parser.cpp:82-90`），避免标记被拆到两个 chunk；推理末尾的空白
同样会被回退，以便与模板的 rstrip 一致（`trailing_whitespace_run`）。`</think>` 之后与一个工具块
之后的首个 content 片段会去掉前导换行（`strip_content_leading_`，`response_parser.cpp:222-228`）。
`finish()` 冲刷剩余缓冲，并在“有工具调用且 content 只有空白”时清空 content（分隔符噪声，
`response_parser.cpp:348-354`）；解析不出来的工具块会**原样退回成 content**
（`response_parser.cpp:312-314`、`327-329`），不会静默吞掉。parse_tools 为假（completion 端点）时
`<tool_call>` 直接当普通 content。以上行为由 `tests/server/test_response_parser.cpp` 覆盖
（逐字符流式与整段解析必须给出同一结果）。

`finish_reason` 分三条路径：

* **文本/调度器**（`server.cpp:820-824`、`1097-1098`）：有工具调用 → `"tool_calls"`；否则 stop 命中
  → `"stop"`；否则用调度器的 `seq->finish_reason`（只有 `"stop"`/`"length"`，`scheduler.h:27`）。
* **多模态**（`server.cpp:876-880`、`1165-1166`）与 **MTP**（`server.cpp:945-949`、`1237-1238`）：
  这两条路径没有调度器的 `finish_reason`，所以非 stop、非工具调用时**硬编码为 `"length"`**——
  即使是 EOS 或 `max_tokens` 结束也一样。这是两条路径与文本路径唯一的 `finish_reason` 差异。

### 2.9 线程模型

* httplib 每个连接派一个工作线程执行 handler。
* 非流式 handler 阻塞在 `seq->pop_token(t)`（`sequence::token_out`，`scheduler.h:76-85`）；流式 handler
  为每个 choice 起一个生产者线程后立即返回，由 chunked provider 驱动。handler 返回时它捕获的一切
  （prompt 副本、`shared_ptr<sse_session>`、多模态的共享许可）都由线程自己的 lambda 持有。
* `scheduler::submit` 多线程安全（锁 `scheduler::m`）；每个 `sequence` 的输出由自带 mutex+cv 保护。
* 所有引擎工作由 `engine::mtx` 串行化——调度器的两个阶段各自持 `e.mtx`（`scheduler.cpp:150`、
  `318`），`engine::generate` / `generate_mm` 也在入口加锁（`engine.cpp:2350`、`2356`）。
* 多模态额外用 `serve()` 的共享 `media_gate` 许可串行化，因为请求共享
  `engine::d_img_embd`。许可由 handler 线程获取，可由最后持有它的 SSE 生产者线程释放；
  内部 mutex 只在获取/释放许可的瞬间持有，**不跨线程转移 `std::mutex` 的所有权**。
  媒体设备编码在 `engine::mtx` 下提交，避免与调度器 forward 交错；之后
  `generate_mm` 自己获取同一锁。多模态的 `n` 个 choice 在一个线程里顺序生成；
  MTP 的 `n` 恒为 1（`mtp_direct` 门控）。

### 2.10 关闭与信号

`SIGINT`/`SIGTERM` 处理器只置原子 `g_term_requested`（`server.cpp:50-53`、注册在 `1916-1917`）；
watchdog 线程每 100 ms 轮询 `g_term_requested` 与 `listen_done` 并调用 `srv.stop()`
（`server.cpp:1918-1926`）。`srv.listen` 返回后 `listen_done` 置位让 watchdog 自己退出并 join，
`g_term_requested` 复位；listen 失败打印 `failed to listen on ...` 并返回 1，否则打印
`[srv] shutting down, flushing prefix cache` 并返回 0（`server.cpp:1927-1936`）。栈上 `scheduler`
析构调用 `shutdown()`（置 `stopping`、notify、join，`scheduler.cpp:34-43`），`serve` 返回后
`~engine` 把 RAM/disk 与驻留 VRAM 节点 flush 到磁盘层。

---

## 3. 连续批处理调度器

### 3.1 `sequence`（`scheduler.h:16-89`）

* 身份/位置：`id`、`slot`（引擎状态槽 `[0,kMaxB)`）、`prompt`、`prompt_pos`、`blocks`、`recent`。
* 生成状态：`gp`、`ss`（`sampler_state`，每序列独立的 RNG）、`n_generated`、
  `finish_reason`（默认 `"stop"`）、`stops`。
* 生命周期：`admitted`、`finished`。
* 诊断（`PF_SRV_TIME`）：`t_submit`、`wait_ms`、`admit_ms`、`pf_ms`、`n_chunks`、`reused`
  （`reused` 同时就是响应里 `usage.prompt_tokens_details.cached_tokens` 的来源）。
* 输出：mutex `m` + condvar `cv` + **`std::deque<token_out> out_q`**（不是 `deque<string>`）+ `int
  prompt_tokens`。`push(string)` 造一个只有 `text` 的 `token_out`，`push_token` 追加并 `notify_all`；
  `pop_token` 阻塞到有数据或 `finished`，排空时返回 false；`pop`/`pop_wait` 是只取 `text` 的薄封装
  （服务器实际只用 `pop_token`，因为 logprobs/best_of 需要 id 与 logprob）。
* `token_out`（`scheduler.h:43-48`）：`text`（UTF-8 安全片段，token 字节不完整时为空串）、
  `id`、`logprob`、`top`（`(token id, logprob)`，长度 = `top_logprobs`）。只有 `wants_logprobs()`
  为真时才填后三者。

序列存在两个 vector：`waiting`（已提交未准入）与 `active`（已准入）。`m` 保护两者，`cv` 在
submit/shutdown 时唤醒循环。

### 3.2 线程生命周期

* `start()` 恰好起一个 `loop()` 线程（`scheduler.cpp:30-32`）。
* `shutdown()` 置 `stopping`、通知、join（析构调用，`scheduler.cpp:34-43`）。
* `submit()`（`scheduler.cpp:45-63`）：锁 `m`，分配 `id`，拷贝 prompt/gp，设 `prompt_tokens`，
  `ss.seed(gp.seed ? gp.seed : steady_clock count)`，push 到 `waiting`，通知。
* 循环退出条件：`stopping && waiting.empty() && active.empty()`（`scheduler.cpp:140-145`）。

### 3.3 准入（`admit`，`scheduler.cpp:65-113`）

在 `loop()` 内、**同时持有** `e.mtx` 与 `m` 时调用：

1. 超长 prompt（`> e.max_seq`）→ `retire(s,"length")` 并返回 true（从 `waiting` 移除）。
2. 找空闲 slot（标记 active 占用的槽，取第一个空闲 `[0,kMaxB)`）；无 → 返回 false（队头阻塞，本循环
   停止准入）。
3. 前缀缓存 `e.pc_admit(slot, prompt, blocks)`；`matched <= 0` 时 `e.zero_slot(slot)`（清 GDN/conv
   递归状态）；`s->reused = matched`。
4. 为首个 chunk 预留块：`first_end = min(matched+kMaxT, prompt.size())`，`need = ceil(max(first_end,1)/32)`，
   `e.alloc_block()` 补足。失败则 `e.pc_retire(slot, blocks)` 并返回 false。
5. 安装 `slot`、`prompt_pos = matched`、`blocks`、`e.set_table`、`admitted = true`。

`e.alloc_block()` 不是简单出池：它先驱逐前缀缓存节点，再在 cap 内提交更多池内存。

`retire(s, reason)`（`scheduler.cpp:115-127`）设 `finish_reason`、**先 `e.prefill_flush()`**（多设备
流水线里还有排在 device 1 未跑的 prefill chunk，它还要写这些块，块不能在它前面释放）、再
`pc_retire` + 清空 `blocks`，最后在序列自己的锁下置 `finished` 并 `notify_all`。

### 3.4 主循环（`scheduler.cpp:134-419`）

两个阶段的锁顺序都是 **先 `e.mtx` 后 `m`**，但 **`m` 在每个 engine 调用前后显式 unlock/relock**
（`std::unique_lock<std::mutex> lk2(m)` + `lk2.unlock()` / `lk2.lock()`，`scheduler.cpp:157`、`234-241`、
`319`、`360-362`）。这是有测量支撑的**必需**行为，不是风格选择：

> 旧代码在 engine 调用期间一直持有 sequence mutex `m` 并立刻重锁，于是 `scheduler::submit()`
> 被饿死——每个请求只有在前一个生成**结束**后才被准入，跨序列批处理两条路径都成了死代码。
> 实测（27B、temperature 0、`ignore_eos`、每请求 128 token、8 个并发请求）：请求分别隔
> 9、18、27 … 71 s 结束，聚合 14.5 tok/s = 恰好一个请求的速率，188/188 次 decode pass 都是
> `nb=1 active=1`。改成在 `m` 释放状态下调用 engine（`e.mtx` 仍然串行化引擎本身）之后，
> N=1/2/4/8 → 14.3 / 24.5 / 40.1 / **55.0 tok/s**，即 N=8 时聚合 **3.85x**。
> （条件与数字同样记录在 `AGENTS.md` 的 “Invariants and gotchas”。）

**阶段 1 — 准入 + 一个 prefill chunk**（`scheduler.cpp:149-314`）：

* 从头到尾 drain `waiting` 直到 `admit` 失败，把准入的移入 `active`。
* 轮询 `active`：跳过 `finished`；第一个 `prompt_pos < prompt.size()` 的序列执行**恰好一个 chunk** 后
  break。这是公平性策略：active 的 prefill 严格 round-robin、每循环一个 chunk。
* chunk 大小：从 `e.batched_prefill_fit(rem)` 向下按 `kMaxT` 步尝试，取能容纳（按需分配块并
  `set_table`）的最大 chunk-batched 尺寸；都不行则回退 `n = min(kMaxT, rem)`。
  * `batched_prefill_fit`：`PF_GEMM_DNNL`（或多设备 GPU 分区有 oneDNN）时允许整个余量直到
    `kMaxB*kMaxT`；DP4A/md_int8 只允许已录制的 `kMaxT` 倍数图尺寸，且**不允许**部分末行。
  * 尾部也走 mode 2（不是 `>= 2*kMaxT` 门槛）：mode-1 的 chunk 每 32 token 要付约 0.3 ms 的
    handoff/sync，一个 41 token 的尾巴曾花掉 553 token prefill 的约 0.7 s
    （`scheduler.cpp:180-185` 的注释）。
* 饥饿守卫：`blocks.size()*32 < prompt_pos + n` 时无法写入 → `retire(s,"length")`。
* `last_chunk = (prompt_pos + n >= prompt.size())`。forward：mode 2 用 `prefill_batch`（**不传**
  `last_chunk`，它总是跑 head），否则 `prefill_chunk(..., last_chunk)`。
* 推进 `prompt_pos += n`，`pc_commit`。
* prompt 完成后：`prefill_flush()` → `fetch_logits(0, ...)` → `recent = prompt` →
  `sample_token` → 追加，`n_generated++`。首个片段走 `ubs[0]` 这个 UTF-8 缓冲（每循环只有一个序列会
  走这里，所以复用 row 0 是安全的；decode 阶段则按 batch row 用 `ubs[r]`，`scheduler.cpp:137`）。
  EOS 对客户端不可见（不推送）；`n_generated >= max_tokens` 以 `"length"` 退役，否则 EOS 以 `"stop"`
  （`scheduler.cpp:307-309`）。

**阶段 2 — 批量 decode**（每个外层循环都执行，因此 prefill 与 decode 交错；`scheduler.cpp:316-408`）：

* 收集最多 `kMaxB` 个“已 prefill 完成且 `recent` 非空”的 active 序列。
* `pos = recent.size()`（下一写入位置），按 `ceil(pos/32)` 扩块表。
* 有效性检查：decode 写入 `recent.back()` 于 `wpos = pos-1`；`wpos >= max_seq` 或
  `>= blocks.size()*32` 则 `retire(s,"length")` 且该行不进 batch。
* `e.decode_batch`，逐行 `fetch_logits(r, ...)` → `sample_token` → 追加 → EOS/`max_tokens` 处理 →
  push piece。退役原因判定与 prefill 阶段略有差别：decode 是
  `n_generated >= max_tokens && !eos ? "length" : "stop"`（`scheduler.cpp:400`），即**同时**撞上 EOS 与
  `max_tokens` 时 decode 报 `"stop"`、prefill 报 `"length"`。
* 已完成的从 `active` 擦除。

**空闲等待**：本轮无工作时 `cv.wait_for(lk, 2 ms, pred)`，谓词是
`stopping || !waiting.empty() || !active.empty()`（`scheduler.cpp:414-416`）。谓词让 submit/shutdown
立刻唤醒，2 ms 只是兜底——旧的固定 sleep 会给每个首 token 加 0-2 ms。

### 3.5 调度策略与边界

* **准入**：FIFO、贪心，遇到第一个不可准入者停止；容量受空闲槽（`kMaxB`=16）与空闲 KV 块限制。
* **prefill**：round-robin，每序列每循环一个 chunk，优先用最大可容纳的 chunk-batched 尺寸。
* **decode**：所有合格序列合并成一次 `decode_batch`。
* **无显式优先级、无 decode-first、不驱逐 active 序列**。块压力在 `alloc_block` 内解决；再失败则相关
  序列以 `"length"` 退役。
* **stop 字符串不在这里处理**，是服务器侧后处理。调度器只管 EOS/EOT、`max_tokens`、上下文/块耗尽、
  分配失败。
* **前缀缓存集成**：admit 时 `pc_admit` / `zero_slot`；每 chunk 后 `pc_commit`；退役时 `pc_retire`；
  准入失败平衡引用。多模态完全绕过。
* **跨序列 prefill 批处理没有启用**（会损坏输出）。把多个 prompt 打进一次 mode-2 forward
  （按拼接 token 的 row-major、每行各自的 `slot`/`pos`/`n_real_row`、每 batch 一个 `prefill_text`）
  实现过，产出的是**垃圾**——退化续写（`1000000000...`、`| 10 | 10 | 10 |`），不是“其它合理解法”，
  即使把 decode batch 限到 1 也一样；`step_info` 本来就带 per-row 的 `slot`/`pos`/`active`，所以故障
  在某个读取 **plan 期状态**的东西上，而不是 `info`。因此 prefill 一次只跑一个序列（27B 上各约
  300 ms），上面那个 3.85x **全部来自 decode 批处理**。N=8 时串行化的 prefill 约占 18.6 s 总墙钟里的
  2.4 s；要修它需要先有一个“N 个并发相同请求必须返回逐字节相同输出”的测试台，并把 decode batch
  限到 1 以隔离变量（条件与数字同 `AGENTS.md`）。
* **并发会破坏逐位一致性**。同一个 temperature-0 prompt 扇出到 N 个并发请求，返回的是连贯、确定、
  但**可能与单请求结果不同**的文本：翻转发生在近平票的 argmax 上，一旦某序列加入 batch 就发生
  （实测 N 行里 1 行逐字节相同，其余在同一处 "attention mechanism" / "**Scaled Dot-Product
  Attention**" 的平票上翻转，约 20 token 处；多次运行的输出多重集合相同，所以是累加顺序的数值问题，
  不是竞争）。先被准入、单独 decode 的序列保持早期 token 精确；batch 只改变批量 GEMM 里 fp 累加的
  顺序。需要逐位可复现的客户端应串行发请求。

### 3.6 MTP 路由（`mtp_direct`，`server.cpp:883-907`）

`mtp_direct(e, gp, n, logprobs)` 决定请求是否走 §1 的第三条路径，五个条件全满足才为真：

```cpp
on /* PF_MTP_SERVER */ && e.mtp_on && n == 1 && !logprobs && (gp.temperature <= 0.f || gp.top_k == 1)
```

* `PF_MTP_SERVER` **默认关**（`server.cpp:902-905`）；`e.mtp_on` 需要启动时 `--mtp N` 且模型带
  NextN 头 + 多设备 oneDNN int8 分区（见 `AGENTS.md` 的 MTP 一节）。
* `n == 1 && !logprobs` 与 `greedy` 是必须的：MTP 的接受判据是“目标自身的下一 token 的精确相等”，
  采样目标需要 rejection sampling 才精确；而且这条路径拿不到 per-token 采样细节。
* **默认关是测出来的**：MTP 是单序列循环，运行它等于在整个生成期占住 `engine::mtx`，一切都被串行化
  （实测 8 个并发 greedy 128-token 请求：MTP 17.5 tok/s vs 走调度器 55.0）。而且 MTP 与批处理不
  可组合——批量 decode 一步实测 `59.0 ms + 10.7 ms/row`，MTP 却要花 `k+1 = 5` 个 verify 行换
  `acc+1 ≈ 2.49` 个 token（2.0 行/token 对 plain 的 1.0），所以固定行数预算下 S≥4 起 MTP 反而更慢。
  `PF_MTP_SERVER=1` 恢复旧路由，只值得用在严格单请求的负载上。
* 它的价值是**单请求延迟**：同一 prompt、同一 greedy 设置，服务端 68.6 → **45.5 ms/token = 1.51x**，
  且强依赖 prompt（acc 1.49 vs 2.28）。引擎侧还有一道同样的 greedy 门
  （`generate_impl`，`engine.cpp:2365-2367`），多模态请求在那里被显式排除。

---

## 4. Chat 渲染

### 4.1 `render_chat`

`render_chat(tmpl, msgs, add_generation_prompt, enable_thinking, tools_json)`
（`chat.cpp:37-44`）先试 `render_chat_template`，任何异常或空模板都回退 `render_chat_builtin`。
`tmpl` 来自 GGUF 的 `tokenizer.chat_template`（`model.h:94`、`model.cpp:66-67`），服务器传的是
`e.m.chat_template`。`chat_msg` 除 role/content/parts 外还带 `reasoning_content`、`tool_calls`
（`{id,name,arguments(JSON 字符串)}`）、`tool_call_id`、`name`（`chat.h:30-49`）。

### 4.2 minja Jinja 封装（`chat_template.cpp`）

* `tmpl` 为空立即返回 false（`chat_template.cpp:23-25`）。
* `PF_CHAT_TMPL_DEBUG` 控制是否把异常写到 stderr（`chat_template.cpp:26`、`101-105`）；`tools_json`
  解析失败也只在 debug 下打印并把 `tools` 留空（`chat_template.cpp:84-94`）。
* 每次调用**新建** `minja::chat_template(tmpl, "", "")`（`chat_template.cpp:28`），模板无解析缓存。
* 消息转 `nlohmann::ordered_json`：文本消息 `{role, content:<string>}`；含 part 的消息
  `{role, content:[...]}`，part 的 `type` 按 kind 映射为 `"image"` / `"video"` / `"input_audio"` /
  `"text"`（`chat_template.cpp:38-45`）——注意**视频与音频在模板里也走 part 形式**，不是纯文本。
  assistant 带 `reasoning_content`；带 `tool_calls` 时把 `arguments` 解析成 JSON 对象交给模板
  （解析失败就原样给字符串），`tool` 带 `tool_call_id`/`name`。
* 输入：`messages`、`tools`、`add_generation_prompt`、`extra_context = {"enable_thinking": bool}`。
  `bos_token`/`eos_token` 有意留空，因为渲染结果是完整 prompt 字符串，由分词器按 `parse_special` 编码。
* `ct.apply` 抛出的任何异常都被捕获并返回 false，调用者回退（所以一个语法/运行时不支持的模板不会
  让请求失败，只会换一个 prompt——这就是 §8 里 `role:"function"` 那类输入的兜底路径）。

### 4.3 内置 ChatML 回退

* 空消息 → 空串。
* `content_of`：无 parts 返回 `content`；否则按顺序拼接，媒体 part 替换为占位符——图片
  `<|vision_start|><|image_pad|><|vision_end|>`、视频 `<|vision_start|><|video_pad|><|vision_end|>`、
  音频 `<|audio_start|><|audio_pad|><|audio_end|>`（`chat.cpp:95-109`）。
* 有 `tools_json` 时注入 Qwen 风格的 `# Tools` 说明块：首条是 system 就附在它后面，否则自己合成一条
  `system` 消息（`chat.cpp:120-127`）。
* user/assistant/tool 各按 ChatML 渲染：user 是 `user\n<content>\n`；assistant 优先用
  `reasoning_content`，否则从 `content` 里找 `</think>`（再往前找 `<think>`）拆出 reasoning，且**只在
  “最后一个 user 之后的 assistant 轮”**保留 `<think>...</think>`（`chat.cpp:151-155`）；`tool_calls`
  渲染成 `<tool_call>\n<function=...>\n<parameter=...>\n...</parameter>\n</function>\n</tool_call>`
  块（`chat.cpp:64-83`）；`role:"tool"` 渲染成一条 `user\n<tool_response>\n...\n</tool_response>\n`
  （`chat.cpp:166-168`）。
* **其它 role（含旧式 `role:"function"`）不匹配任何分支，被静默丢弃**（`chat.cpp:129-168` 的 if/else
  链没有 else）——不报错。
* `add_generation_prompt` 时追加 `assistant\n`；`enable_thinking` 追加 `<think>\n`，否则
  追加一个空的 `<think>\n\n</think>\n\n`。

### 4.4 `utf8_stream_buffer`（`chat_util.h`）

`push(piece)` 按 UTF-8 首字节推算序列长度，只返回“完整且合法续字节”的安全前缀，把尾部不完整/非法
序列留在 `pending`；`flush()` 返回剩余部分（生成结束时避免丢尾部）。被调度器（每行一个）、多模态
生成、CLI 流式输出使用。

---

## 5. 停止字符串

stop 字符串由**服务器侧**在生成文本上匹配（`stop_filter`）：

* 每个片段先进入 stop 过滤，命中任意 stop 即截断、置 `stopped`，其余文本丢弃；为检测跨片段的 stop，
  末尾会暂存 `max(len(stop))-1` 字节，结束时冲刷。
* 过滤后的文本才送入 `response_parser`（或直接作为 completion 文本），因此 stop 标记不会泄漏。
* `finish_reason` 优先级：有工具调用 → `"tool_calls"`；否则 stop 命中 → `"stop"`；否则沿用序列的 `finish_reason`（文本路径即 `seq->finish_reason`；两条旁路则硬编码，见 §2.8）。

因为不是采样期停止，stop 之后的 token（以及 mm 路径 EOS 之后）可能已在内部生成。

---

## 6. 多模态请求处理

* 启动时若有 `mmproj_path` 则 `mm.vm.load()`，从视觉超参导出 `image_preproc_cfg`
  （`patch_size`/`merge`/`mean`/`std`，`max_pixels = kMaxImgTokens * patch²`），失败非致命（打日志，
  后续媒体请求 400）；若有 `audio_mmproj_path` 则 `mm.am.load()` 导出 `audio_preproc_cfg`
  （`sample_rate`/`n_fft`/`hop`/`n_mel`/`f_min`/`f_max`），失败同样非致命（音频请求 400）
  （`server.cpp:1460-1497`）。`max_video_frames`/`max_video_side` 从 `server_config` 拷进 `mm_server`
  （`server.cpp:1498-1499`）。
* `parse_messages` 按顺序收集媒体的 `media_part`（`image_url`/`image`、`video_url`/`video`、
  `input_audio`（带 `data`+`format`）、`audio_url`），保留文本/媒体交错（`cm.parts` 的顺序就是模板
  看到的顺序），`chat_part` 带 `kind` 标记。
* `load_media_bytes` 接受 `data:`（要求 `;base64`；`b64_decode` 容忍空白、遇到 `=` 停止）、
  内联 `input_audio` base64 与 `http(s)://`。其它 scheme 一律 400。远程抓取的策略见
  [§5 远程媒体抓取与 SSRF 边界](#5-远程媒体抓取与-ssrf-边界)（`src/server/url_fetch.{h,cpp}`，
  默认关闭，开启时只允许公网可达地址且逐跳复检跳转）。
* 图片 `mm_image_decode_mem` → `mm_image_preprocess`；视频 `mm_video_decode_mem`（`max_frames`/
  `max_side` 由上面两个字段控制，探测/解码分别走 `PF_AV_FFPROBE`/`PF_AV_FFMPEG`）；音频
  `mm_audio_decode_bytes`（WAV 原生解析 PCM 8/16/24/32 与 IEEE float 32/64、立体声下混，其它容器落
  临时文件走 ffmpeg CLI），失败均 400。
* `render_chat(..., add_generation_prompt=true, thinking, tools_json)` → `mm_build_prompt_mixed_device`
  （按 `mm_media_ref order` 把 image/video/audio 分别经视觉/音频塔编码进 `e.d_img_embd`）→ 用
  **扩展后**的 token 数检查长度 → `run_mm_choice`（非流式）或 `stream_chat_mm_choices`（流式，逐个
  choice 顺序生成）。`out.prompt_tokens` 取 `mp.tokens.size()`，所以 `usage.prompt_tokens` 是含媒体
  token 的值。
* 全程持有 `mm_req`（`serve()` 的函数局部 mutex，见 §2.9）；`generate_mm` 内部持有 `engine::mtx`
  （`engine.cpp:2356`）。
* 注意：mm 路径 `finish_reason` 只在 stop 命中时为 `"stop"`，其余情况（EOS、`max_tokens`、块耗尽）
  一律 `"length"`，有工具调用时为 `"tool_calls"`；`n>1` 顺序生成；绕过前缀缓存与连续批处理；
  `logprobs` 恒为 `null`（§2.6）。多模态请求也**不会**走 MTP——`handle_chat` 先判媒体分支再判
  `mtp_direct`，引擎侧也显式排除（`engine.cpp:2365`）。

### 6.1 远程媒体抓取与 SSRF 边界（`src/server/url_fetch.{h,cpp}`）

HTTP API **没有任何鉴权**，而绑定地址一旦不是 loopback 就等于把引擎交给所有能连到端口的人。
在这个前提下，「让服务端 GET 任意 URL」就是一把 SSRF 枪：它能碰到调用方碰不到的一切——
云实例元数据（`169.254.169.254`）、集群内管理接口、同机上没加保护的推理端口。抓取发生在持有
`mm_req` 期间，每个 URL 最多 10 s，因此一个慢 URL 还能串行化其它所有多模态请求。

三层防御，默认全部关闭：

| 层 | 默认 | 开启方式 | 挡住什么 |
|---|---|---|---|
| 远程抓取总开关 | **关** | `PF_MM_URL_FETCH=1` | 默认配置下不存在任何出网请求 |
| 目的地址过滤 | 开（仅在抓取开启时生效） | `PF_MM_URL_ALLOW_PRIVATE=1` 放行内网 | 直接写 `http://169.254.169.254/...` |
| 逐跳跳转复检 | 开（仅在抓取开启时生效） | 无 | 「一个公网 URL 302 跳到内网地址」 |

**地址分类**（`addr_is_public`）拒绝：unspecified、`127.0.0.0/8`、`10/8`、`172.16/12`、
`192.168/16`、`169.254/16`（含元数据地址）、CGNAT `100.64/10`、multicast、reserved，
以及 special-use/文档段（`192.0.0.0/24`、`192.0.2.0/24`、`192.88.99.0/24`、
`198.18.0.0/15`、`198.51.100.0/24`、`203.0.113.0/24`）；IPv6 侧拒绝 `::`、`::1`、
`fe80::/10`、`fc00::/7`、`ff00::/8`，以及 6to4 包裹非公网 v4 的 `2002::/16`。

两个容易写错、测试里专门盯住的点：

* **用解析而不是字符串匹配**。`host_is_public` 走 `getaddrinfo`，所以十进制
  （`2130706433`）、八进制（`0177.0.0.1`）写法的 loopback 会被挡住，任何文本过滤都会漏。
  一个名字返回多条 A 记录时，**只要有一条非公网就整体拒绝**。
* **IPv4-mapped IPv6 按内嵌的 IPv4 判定**，否则 `::ffff:127.0.0.1` 直接绕过整个 IPv4 表
  （`v6_is_public` 里先查 `IN6_IS_ADDR_V4MAPPED`）。同理 `2002::/16` 拆出被包裹的 v4 再判。

**跳转必须自己走**。原实现用 `cli.set_follow_location(true)`，而 httplib 的跳转循环自己
重新解析并重新连接，**不给调用方任何钩子**——一个校验过的 URL 因此可以变成任意 URL，这正是
最常见的绕过方式。现在 `set_follow_location(false)`，`url_fetch` 自己走至多 5 跳，每跳重新
解析 + 重新分类；`resolve_redirect` 支持绝对 / scheme-relative / 根相对 / 路径相对，并拒绝
`https → http` 降级（否则允许它的理由——传输保证——被悄悄丢掉）。重定向响应体不进 payload。

**已知残留**：`host_is_public` 在校验时解析一次，httplib 连接时再解析一次，恶意 DNS 服务器仍能
赢下这个竞争（DNS rebinding）。彻底关掉需要把连接钉在校验过的 IP 上，httplib 没有这个钩子，
本轮不做。默认关闭意味着要触发它必须先显式开启远程抓取——所以这里如实记录为残留，而不是宣称
已经消除。上限 10 MB 与各 10 s 超时不变。

测试 `tests/server/test_url_fetch.cpp`（无模型、无 GPU）：地址分类逐条列出被拒范围并检查
172.15/172.32 等相邻公网段**没有**被过度拦截；`url_parse`/`resolve_redirect` 的接受与拒绝；
以及用进程内 httplib 服务做的集成测试——它监听 `127.0.0.1`，所以既是「调用方控制的端点」
也是「跳转目标」，整条链不离开本机。策略由 `url_fetch_mode()` 的静态缓存决定，故三种策略各跑
一个进程（`test_url_fetch_default|public|private`），测试自己 setenv/unsetenv，shell 里已有的
`PF_MM_URL_FETCH` 影响不到 default 那档。

---

## 7. 诊断环境变量

| 变量 | 作用 |
|---|---|
| `PF_SRV_TIME` | 端点侧打印 `tokenize=.. ms chars=.. tokens=..`；调度器侧每 chunk 一行 `chunk pos=.. n=.. ms=..`，每个序列结束时一行 `prefill/fetch/sample/to_first_tok (wait/admit/prefill_total/chunks/reused/total)`（`server.cpp:1705-1712`、`1803-1814`、`scheduler.cpp:242-250`、`285-290`） |
| `SCHED_DEBUG` | 准入/prefill/decode 决策日志，含每行 decode 的 `seq/slot/pos/tok`（`scheduler.cpp:129-132`、`252-255`、`363-369`、`378-380`） |
| `PF_CHAT_TMPL_DEBUG` | 记录导致回退 ChatML 的模板异常（也记录 `tools_json` 解析失败，`chat_template.cpp:26`、`90-92`、`102-104`） |
| `PF_MTP_SERVER` | `1` 让 `mtp_direct` 把 greedy 单请求路由进单序列 MTP 循环，绕过调度器；**默认关**，理由与实测见 §3.6（`server.cpp:901-907`） |
| `PF_MM_URL_FETCH` | 允许下载远程 `http(s)://` 图片/视频/音频（base64 `data:` 不受影响），**默认关闭**；设 `1` 开启且只允许公网可达地址 |
| `PF_MM_URL_ALLOW_PRIVATE` | 仅在 `PF_MM_URL_FETCH=1` 时有意义：`1` 放行 loopback/私有/link-local 目标（内网媒体服务器场景，等于重新暴露 SSRF） |
| `PF_AV_FFMPEG` | 音频/视频解码的 ffmpeg 可执行路径（默认 `ffmpeg`，`video.cpp:340`、`audio.cpp:22`） |
| `PF_AV_FFPROBE` | 视频探测（时长/帧率）的 ffprobe 可执行路径（默认 `ffprobe`，`video.cpp:345`） |
| `PF_GEMM_DNNL` | 不是服务端旋钮，但通过 `engine::batched_prefill_fit` 决定 prefill chunk 尺寸（见 §3.4） |

完整的环境变量总表见 [`../../AGENTS.md`](../../AGENTS.md#environment-variables)；本表只列 `src/server/` 读到的。

---

## 8. OpenAI 字段支持矩阵

图例：✅ 已实现；⚠️ 部分实现/仅近似；❌ 未实现（静默忽略或返回 400）。本表以当前 `server.cpp`
的解析与响应代码为准。

### 8.1 `POST /v1/chat/completions` 请求

| 字段 | 状态 | 说明 |
|---|---|---|
| `model` | ✅ | 仅回显/用于选择（单模型）；未知 id 不报错 |
| `messages` | ✅ | 见 8.4 |
| `stream` | ✅ | SSE；纯文本路径每 choice 一个生产者线程，多模态/MTP 路径一个线程顺序生成全部 choice（§2.7） |
| `stream_options.include_usage` | ✅ | 最后一个 choice 发 usage chunk |
| `stream_options.continuous_usage_stats` | ❌ | 忽略 |
| `max_tokens` / `max_completion_tokens` | ✅ | 默认 256，`≤0` 强制 256 |
| `n` | ✅ | 1..16（超出截断）；choice index 0..n-1 |
| `temperature` | ✅ | |
| `top_p` | ✅ | |
| `top_k` / `min_p` | ✅ | llama.cpp 扩展（OpenAI 无此字段） |
| `presence_penalty` / `frequency_penalty` | ✅ | |
| `repeat_penalty` / `repetition_penalty` / `repeat_last_n` | ✅ | llama.cpp 扩展 |
| `stop` | ✅ | 字符串或数组，服务器侧匹配，最多 16 条 |
| `seed` | ⚠️ | `0`/缺省 = 每条序列用时钟/随机源播种（不可复现）；非 0 时第 `c` 个 choice 用 `seed+c`；并发批处理还会改变 GEMM 的 fp 累加顺序，所以**不保证逐位复现**（§3.5） |
| `ignore_eos` | ✅ | llama.cpp 扩展 |
| `logprobs` / `top_logprobs` | ✅ | 生成 token 的 logprob（惩罚/偏置、温度缩放后的完整词表 log-softmax）；chat 返回 `logprobs.content[]`；多模态与 MTP 路径恒为 `null`（拿不到 per-token 细节；`logprobs` 也会让 `mtp_direct` 退出） |
| `logit_bias` | ✅ | token id -> 加性偏置，clamp 到 [-100,100]，在惩罚之后、softmax 之前应用 |
| `response_format`（`json_object` / `json_schema`） | ❌ | 无语法约束，忽略 |
| `tools` | ✅ | 传给 Jinja 模板（Qwen 原生支持） |
| `tool_choice` | ⚠️ | `none` 清空工具；字符串/指定函数对象按名过滤；`required` 只透传不强制 |
| `parallel_tool_calls` | ❌ | 忽略；始终允许模型并行调用 |
| `functions` / `function_call`（旧式） | ❌ | 请求级旧接口未实现；消息级旧 `function_call` 仅用于输入解析 |
| `chat_template_kwargs.enable_thinking` | ✅ | 控制模板 `<think>`；也是推理拆分开关 |
| `enable_thinking` / `thinking` | ✅ | 平铺别名 |
| `reasoning_effort` | ⚠️ | 仅映射为“开/关思考”（`"none"` 关闭），无等级 |
| `metadata` / `store` / `user` / `service_tier` | ❌ | 忽略 |
| `modalities` / `audio` / `prediction` | ❌ | 忽略 |
| `web_search_options` | ❌ | 忽略 |
| `best_of`（chat 端） | ❌ | 忽略；只有 `/v1/completions` 实现（§8.3） |
| 图片 part `{"type":"image_url","image_url":{"url":...}}` | ✅ | `data:` base64 或 `http(s)://`（远程**默认关闭**，`PF_MM_URL_FETCH=1` 开启且限公网地址）；需要 `--mmproj` |
| 视频 part `{"type":"video_url","video_url":{"url":...}}` | ✅ | 同 `image_url`，需 `--mmproj`，由 `max_video_frames`/`max_video_side` 控制采样 |
| 音频 part `{"type":"input_audio","input_audio":{"data":...,"format":"wav"}}` | ✅ | `data` 为 base64 内联音频（WAV 原生，MP3/OGG 等经 ffmpeg CLI）；需 `--audio-mmproj`，且因为媒体分支先查视觉塔，**实际还需要 `--mmproj`**（§2.5）；`format` 只被记录、不参与解码 |
| 音频 part `{"type":"audio_url","audio_url":{"url":...,"format":...}}` | ✅ | `data:` 或 `http(s)://`，同上 |

### 8.2 `POST /v1/chat/completions` 响应

| 字段 | 状态 | 说明 |
|---|---|---|
| `id` / `object` / `created` / `model` | ✅ | `object:"chat.completion"`（流式为 `.chunk`）；`id` 是进程内自增的 `chatcmpl-N` |
| `choices[].index` | ✅ | |
| `choices[].message.role` | ✅ | `"assistant"` |
| `choices[].message.content` | ✅ | 仅有工具调用且无文本时为 `null` |
| `choices[].message.reasoning_content` | ✅ | 仅当请求开启了 thinking 且确实解析到推理时才出现 |
| `choices[].message.tool_calls[]` | ✅ | `{id,type,function:{name,arguments}}`，`arguments` 为 JSON 字符串 |
| `choices[].finish_reason` | ✅ | `stop` / `length` / `tool_calls`；多模态与 MTP 路径在非 stop 时**硬编码** `"length"`（§2.8） |
| `choices[].logprobs` | ✅ | chat 为 `{content:[{token,logprob,bytes,top_logprobs}],refusal:null}`；未请求、多模态、MTP 三种情况为 `null` |
| `usage.{prompt,completion,total}_tokens` | ✅ | 多 choice 求和；多模态的 `prompt_tokens` 含媒体 token |
| `usage.completion_tokens_details.reasoning_tokens` | ✅ | 计入 `reasoning_content` 的生成片段数（按解析出的推理片段计，不是 token 数） |
| `usage.completion_tokens_details.{audio,accepted_prediction,rejected_prediction}_tokens` | ✅ | 恒为 0（占位） |
| `usage.prompt_tokens_details.cached_tokens` | ✅ | 前缀缓存命中的 prompt token 数（`sequence::reused`）；多模态/MTP 路径恒为 0 |
| `usage.prompt_tokens_details.audio_tokens` | ✅ | 恒为 0（占位） |
| `usage.prompt_cache_hit_tokens` / `usage.prompt_cache_miss_tokens` | ✅ | DeepSeek 风格的同义字段（hit = cached，miss = prompt - cached） |
| `system_fingerprint` / `service_tier` | ❌ | |

### 8.3 `POST /v1/completions` 请求/响应

| 字段 | 状态 | 说明 |
|---|---|---|
| `model` | ✅ | 回显 |
| `prompt` | ✅ | `string` / `string[]` / `int[]`（token id）/ `int[][]` |
| `n` | ✅ | 1..16；多 prompt × n 展开为多个 choice，`index` 顺序编号 |
| `stream` | ✅ | 流式 completion chunk |
| `stream_options.include_usage` | ✅ | |
| `echo` | ✅ | 生成文本前置 prompt 文本 |
| `suffix` | ⚠️ | 追加到生成文本末尾（非真正的 insert 语义） |
| `best_of` | ✅ | 非流式；每个 prompt 生成 `best_of` 条，按 chosen token 的 logprob 之和 `stable_sort` 降序后取 top-`n`；`best_of < n` 或流式且 `best_of>1` 返回 400 |
| `logprobs` / `logit_bias` | ✅ | completion 用 legacy 数组 `{tokens,token_logprobs,top_logprobs,text_offset}`；prompt 位置（`echo`）不产出 logprob |
| `temperature` / `top_p` / `top_k` / `min_p` / 惩罚 / `seed` / `ignore_eos` / `stop` | ✅ | 同 chat |
| 响应 `choices[].{index,text,finish_reason}`、`usage` | ✅ | `object:"text_completion"`；该端点不做 reasoning 拆分，故 `reasoning_tokens` 恒为 0 |
| 请求规模上限 | ⚠️ | `prompt 数 × max(n, best_of) > 64` 返回 400，**流式与非流式都受限**（检查在 `if (!stream)` 之前） |

### 8.4 `messages` 元素

| 字段 | 状态 | 说明 |
|---|---|---|
| `role` | ✅ | `system` / `user` / `assistant` / `tool` |
| `role:"function"`（旧式） | ❌ | 内置 ChatML 回退里不匹配任何分支、整条消息被静默丢弃（§4.3）；Jinja 模板路径则原样交给模板 |
| `content` | ✅ | 字符串、`null`、或 part 数组（`text` / `image_url` / `image` / `video_url` / `video` / `input_audio` / `audio_url`）；未识别的 part 被静默丢弃 |
| `reasoning_content` / `reasoning` | ✅ | assistant 输入，回填模板 `<think>` |
| `tool_calls[]` | ✅ | 输入；`function.arguments` 支持字符串或对象 |
| `function_call`（旧式单调用） | ✅ | 作为输入解析为一个 tool_call |
| `tool_call_id` | ✅ | `tool` 消息 |
| `name` | ✅ | `tool` 消息（及 assistant） |

### 8.5 `GET /v1/models`

| 字段 | 状态 | 说明 |
|---|---|---|
| `object:"list"` / `data[]` | ✅ | |
| `data[].id` | ✅ | GGUF `general.name`，回退 `model_id` |
| `data[].object` / `created` / `owned_by` | ✅ | `"model"` / 启动时间 / `"local"` |
| `data[].meta.{n_ctx,n_vocab,n_layer}` | ✅ | 扩展信息 |
| `data[].permission[]` / `root` / `parent` | ❌ | 忽略 |
| `GET /v1/models/{id}` | ✅ | 未知 id 返回 404；`general.name` 与 CLI 默认 `model_id` 两个 id 都接受（§2.3） |

### 8.6 未提供的端点

`/v1/embeddings`、`/v1/tokenize`、`/v1/detokenize`、`/v1/audio/*`、`/v1/images/*`、
`/v1/moderations`、`/v1/files`、`/v1/batches` 均未实现。
