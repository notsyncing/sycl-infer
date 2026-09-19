# 设计 09：HTTP 服务与连续批处理

覆盖 `src/server/server.{h,cpp}`、`scheduler.{h,cpp}`、`chat.{h,cpp}`、`chat_template.{h,cpp}`、
`response_parser.{h,cpp}`、`chat_util.h`，以及 `engine` 的采样接口。

---

## 1. 组件与数据流

| 层 | 文件 | 职责 |
|---|---|---|
| HTTP | `server.cpp` | httplib 服务、路由、JSON、SSE、CORS、图像处理、视觉 tower 调度 |
| 批处理 | `scheduler.cpp` | `sequence` 生命周期、连续批处理、持有 `engine::mtx` |
| Chat 模板 | `chat.cpp` / `chat_template.cpp` | minja Jinja 渲染 + 内置 ChatML 回退 |
| 响应拆分 | `response_parser.cpp` | `reasoning_content` / `content` / `tool_calls` 流式状态机 |
| 流式辅助 | `chat_util.h` | UTF-8 边界安全的 token piece 缓冲 |
| 采样 | `sampler.cpp` | 采样算法 |
| 引擎 | `engine.h` | prefill/decode/logits、前缀缓存 API、`generate_mm` |

两条请求路径：

* **纯文本**（无图片 part）：在 httplib 工作线程分词 → `scheduler::submit` → 连续批处理 → 输出队列。
* **多模态**（含图片 part）：解码/预处理图片 → 视觉 tower → 在专用生产者线程 `engine::generate_mm`，
  **绕过调度器与前缀缓存**。

---

## 2. HTTP 层

### 2.1 服务设置

`serve(engine&, server_config)`：

* `scheduler sched(e); sched.start();`
* `httplib::Server srv;`，读/写超时 3600 s，`set_payload_max_length(64 MiB)`（base64 图片需要）。
* `server_config`（`server.h:8-15`）的 `n_threads` 字段**从未被读取**；httplib 使用默认线程池
  （`max(8, hardware_concurrency()-1)`）。没有安装自定义 task queue。
* `model_id` 默认 `"qwen3.5-0.8b"`。

### 2.2 CORS

每个响应设置 `Access-Control-Allow-Origin: *`、`Allow-Headers: *`、
`Allow-Methods: GET, POST, OPTIONS`；`srv.Options(".*", ...)` 返回 204。
`GET /health` 与 `/v1/models` **不**设置 CORS。

### 2.3 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | `{"status":"ok"}` |
| GET | `/v1/models` | OpenAI 列表；id 取 GGUF `general.name`（回退 `model_id`），附 `meta` |
| GET | `/v1/models/{id}` | 单个模型；未知 id 返回 404 |
| POST | `/v1/chat/completions` | 流式 + 非流式；reasoning_content、工具调用/结果、`n`、stop、`stream_options.include_usage`、图片输入 |
| POST | `/v1/completions` | 纯文本补全；string/string[]/token 数组 prompt、`n`、`echo`、`suffix`、流式 + 非流式 |
| OPTIONS | `.*` | 204 + CORS |

无鉴权，无 `/v1/embeddings`、`/tokenize`。

### 2.4 请求解析

* JSON 解析失败 → 400 `{"error":{"message":"invalid json","type":"invalid_request_error"}}`。
* `parse_messages` 把 `role` 默认 `"user"`；`content` 可以是字符串、`null` 或 part 数组；
  part `type` 为 `"image_url"`/`"image"` 记图片，否则文本。assistant 消息还可带
  `reasoning_content`（或别名 `reasoning`）、`tool_calls`（OpenAI 形状，`function.arguments`
  为字符串或对象）或旧式 `function_call`；`tool` 消息读取 `tool_call_id`/`name`。若某消息无图片
  part，则清空 `parts` 走纯文本快速路径。
* `thinking`（`parse_thinking`）：`chat_template_kwargs.enable_thinking` 优先，其次平铺的
  `enable_thinking`/`thinking`，最后 `reasoning_effort`（非 `"none"` 即开启）。
* `tools_json`（`parse_tools_json`）：把请求 `tools` 数组回传给模板；`tool_choice == "none"`
  时清空，指定函数对象时只保留该函数。
* `gen_params` 来自 `parse_params`，`stops` 来自 `parse_stop`（字符串或字符串数组，最多 16 条），
  `n` 来自 `parse_n`（1..16）。

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

`/v1/completions` 的 `prompt` 支持字符串、字符串数组、token id 数组、token id 数组的数组；
`echo` 把 prompt 文本前置，`suffix` 追加到生成文本后；`best_of`（`>= n`，非流式）生成候选后按
logprob 打分取 top-`n`。`n` 来自 `parse_n`（1..16）。

### 2.5 长度拒绝

`reject_too_long` 返回 400 + `code:"context_length_exceeded"`，并带 `max_seq` 与
`prompt_tokens`。原因（注释）：否则 `scheduler::admit` 会静默退役超长 prompt，返回空的 200 流。
对 chat 文本、每个 completion prompt、以及**扩展后**的多模态 prompt 都调用。

其它 400：无 `--mmproj` 的图片请求；图片下载/解码/预处理失败；流式 completion 的
`prompt 数 * n` 超过 64。

### 2.6 非流式响应

* chat：`object:"chat.completion"`，每个 choice `{message:{role,content[,reasoning_content][,tool_calls]},finish_reason,logprobs}`，
  含 `usage`；`n` 个 choice 的 token 统计求和。仅有工具调用时 `content` 为 `null`；未请求 logprobs 时为
  `null`。
* completion：`object:"text_completion"`，choice `{text,finish_reason,logprobs}`；多个 prompt/`n` 展开为多个
  choice，`index` 顺序编号。
* 非流式路径同样先过 stop 过滤与 `response_parser`，因此与流式输出一致。
* `usage` 含 `prompt_tokens_details.cached_tokens`（前缀缓存命中的 prompt 数，无 `usage` 的旧客户端
  可忽略）与 `completion_tokens_details.reasoning_tokens`（推理 token 数），并额外给出
  `prompt_cache_hit_tokens`/`prompt_cache_miss_tokens` 两个 DeepSeek 风格字段。

### 2.7 流式 SSE

通用形态：

* 响应头 `text/event-stream`、`Cache-Control: no-cache`、`Connection: keep-alive`；
* 用 `set_chunked_content_provider`，内容 reader 从 `sse_queue` 取，队列结束调 `sink.done()`；资源释放
  回调 join 生产者线程（`sse_session`）。
* 每个事件是 `"data: " + dump_json(j) + "\n\n"`；`dump_json` 用 `error_handler_t::replace` 防止坏
  UTF-8 抛异常。
* 每个 choice 一个生产者线程（`n` 个）；最后一个完成的 choice 负责发 usage chunk（若
  `stream_options.include_usage`）与 `data: [DONE]`。
* 请求 logprobs 时，每个 content/text chunk 附带 `choices[].logprobs`（chat 为 `{content:[...]}`，
  completion 为 `{tokens,token_logprobs,top_logprobs,text_offset}`）。

chat chunk schema：

```json
{"id":"chatcmpl-N","object":"chat.completion.chunk","created":T,"model":"...",
 "choices":[{"index":0,"delta":{"reasoning_content":"..."},"finish_reason":null}]}
{"choices":[{"index":0,"delta":{"content":"..."},"finish_reason":null}]}
{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_...","type":"function",
  "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":null}]}
```

每个 choice 先发一个 `delta:{"role":"assistant"}`；随后是 reasoning/content/tool_calls 片段
（stop 处截断）；再发带 `finish_reason` 的最终 chunk（chat 为 `delta:{}`，completion 为 `text:""`）；
最后（可选 usage +）`data: [DONE]`。

### 2.8 回复拆分（`response_parser`）

`response_parser(thinking, parse_tools, emit)` 把生成文本拆成三类片段：

* **reasoning**：`thinking` 为真时，初始处于推理状态，`</think>` 之前（并去掉前导 `<think>`）的文本
  作为 `reasoning_content`；遇到 `</think>` 后转入 content。`thinking` 为假则不产生 reasoning。
* **content**：其余文本；`<tool_call>` 标记不作为 content 输出。
* **tool_call**：解析
  `<tool_call><function=NAME><parameter=K>V</parameter>…</function></tool_call>`，参数值按 JSON 反序列化
  （失败则作字符串），并回退支持 `{"name":…,"arguments":…}` JSON 形式；`arguments` 序列化为字符串，
  `id` 由 `call_…` 计数器生成。

`feed(piece)` 只发射完整片段：piece 末尾可能是 `</think>`/`<tool_call>` 的前缀时会被暂存，避免标记被
拆到两个 chunk；推理末尾的空白会被回退以便与模板的 rstrip 一致。`finish()` 冲刷剩余缓冲。parse_tools
为假（completion 端点）时 `<tool_call>` 直接当普通 content。

`finish_reason`：出现工具调用时为 `"tool_calls"`，否则 stop 命中为 `"stop"`，其余沿用调度器的
`stop`/`length`。

### 2.9 线程模型

* httplib 每个连接派一个工作线程执行 handler。
* 非流式 handler 阻塞在 `seq->pop`；流式 handler 为每个 choice 起一个生产者线程后立即返回，由
  chunked provider 驱动。
* `scheduler::submit` 多线程安全（锁 `scheduler::m`）；每个 `sequence` 的输出由自带 mutex+cv 保护。
* 所有引擎工作由 `engine::mtx` 串行化；多模态额外用文件级 `mm_req` 串行化（共享 `d_img_embd`），
  且 `n` 个 choice 顺序生成。

### 2.10 关闭与信号

`SIGINT`/`SIGTERM` 处理器只置原子 `g_term_requested`；watchdog 线程每 100 ms 轮询并调用 `srv.stop()`。
`srv.listen` 返回后 join watchdog，`serve` 返回；栈上 `scheduler` 析构调用 `shutdown()`，随后 `~engine`
flush 前缀缓存到磁盘。

---

## 3. 连续批处理调度器

### 3.1 `sequence`（`scheduler.h:15-65`）

* 身份/位置：`id`、`slot`（引擎状态槽 `[0,kMaxB)`）、`prompt`、`prompt_pos`、`blocks`、`recent`。
* 生成状态：`gp`、`ss`、`n_generated`、`finish_reason`（默认 `"stop"`）、`stops`。
* 生命周期：`admitted`、`finished`。
* 诊断（`PF_SRV_TIME`）：`t_submit`、`wait_ms`、`admit_ms`、`pf_ms`、`n_chunks`、`reused`。
* 输出：mutex `m` + condvar `cv` + `deque<string> text_out` + `prompt_tokens`；`push` 追加并通知，
  `pop` 阻塞直到有数据或 `finished`，排空时返回 false。

序列存在两个 vector：`waiting`（已提交未准入）与 `active`（已准入）。`m` 保护两者，`cv` 在
submit/shutdown 时唤醒循环。

### 3.2 线程生命周期

* `start()` 恰好起一个 `loop()` 线程。
* `shutdown()` 置 `stopping`、通知、join（析构调用）。
* `submit()`：锁 `m`，分配 `id`，拷贝 prompt/gp，设 `prompt_tokens`，
  `ss.seed(gp.seed ? gp.seed : steady_clock count)`，push 到 `waiting`，通知。
* 循环退出条件：`stopping && waiting.empty() && active.empty()`。

### 3.3 准入（`admit`）

在 `loop()` 内、**同时持有** `e.mtx` 与 `m` 时调用：

1. 超长 prompt → `retire(s,"length")` 并返回 true（从 `waiting` 移除）。
2. 找空闲 slot（标记 active 占用的槽，取第一个空闲 `[0,kMaxB)`）；无 → 返回 false（队头阻塞，本循环
   停止准入）。
3. 前缀缓存 `e.pc_admit(slot, prompt, blocks)`；`matched <= 0` 时 `e.zero_slot(slot)`；`s->reused = matched`。
4. 为首个 chunk 预留块：`first_end = min(matched+kMaxT, prompt.size())`，`need = ceil(max(first_end,1)/32)`，
   `e.alloc_block()` 补足。失败则 `e.pc_retire` 并返回 false。
5. 安装 `slot`、`prompt_pos = matched`、`blocks`、`e.set_table`、`admitted = true`。

`e.alloc_block()` 不是简单出池：它先驱逐前缀缓存节点，再在 cap 内提交更多池内存。

### 3.4 主循环

**阶段 1 — 准入 + 一个 prefill chunk**（先 `e.mtx` 后 `m`）：

* 从头到尾 drain `waiting` 直到 `admit` 失败，把准入的移入 `active`。
* 轮询 `active`：跳过 `finished`；第一个 `prompt_pos < prompt.size()` 的序列执行**恰好一个 chunk** 后
  break。这是公平性策略：active 的 prefill 严格 round-robin、每循环一个 chunk。
* chunk 大小：从 `e.batched_prefill_fit(rem)` 向下按 `kMaxT` 步尝试，取能容纳（按需分配块）的最大
  chunk-batched 尺寸；都不行则回退 `n = min(kMaxT, rem)`。
  * `batched_prefill_fit`：`PF_GEMM_DNNL` 时允许整个余量直到 `kMaxB*kMaxT`；DP4A 时只允许已录制的图尺寸。
* 饥饿守卫：`blocks.size()*32 < prompt_pos + n` 时无法写入 → `retire(s,"length")`。
* `last_chunk = (prompt_pos + n >= prompt.size())`。forward：模式 2 用 `prefill_batch`，否则
  `prefill_chunk`；只有最终 chunk 请求 LM head。
* 推进 `prompt_pos += n`，`pc_commit`。
* prompt 完成后取首个 token：`fetch_logits` → `recent = prompt` → `sample_token` → 追加，
  `n_generated++`。EOS 对客户端不可见（不推送）；`n_generated >= max_tokens` 以 `"length"` 退役，
  EOS 以 `"stop"`。

**阶段 2 — 批量 decode**（每个外层循环都执行，因此 prefill 与 decode 交错）：

* 收集最多 `kMaxB` 个“已 prefill 完成且 `recent` 非空”的 active 序列。
* `pos = recent.size()`（下一写入位置），按 `ceil(pos/32)` 扩块表。
* 有效性检查：decode 写入 `recent.back()` 于 `wpos = pos-1`；越界则 `retire(s,"length")`。
* `e.decode_batch`，逐行 `fetch_logits` → `sample_token` → 追加 → EOS/`max_tokens` 处理 → push piece。
* 已完成的从 `active` 擦除。

**空闲等待**：本轮无工作时 `cv.wait`（谓词 `stopping || !waiting.empty() || !active.empty()`）+
2 ms 超时兜底。

### 3.5 调度策略与边界

* **准入**：FIFO、贪心，遇到第一个不可准入者停止；容量受空闲槽（16）与空闲 KV 块限制。
* **prefill**：round-robin，每序列每循环一个 chunk，优先用最大可容纳的 chunk-batched 尺寸。
* **decode**：所有合格序列合并成一次 `decode_batch`。
* **无显式优先级、无 decode-first、不驱逐 active 序列**。块压力在 `alloc_block` 内解决；再失败则相关
  序列以 `"length"` 退役。
* **stop 字符串不在这里处理**，是服务器侧后处理。调度器只管 EOS/EOT、`max_tokens`、上下文/块耗尽、
  分配失败。
* **前缀缓存集成**：admit 时 `pc_admit` / `zero_slot`；每 chunk 后 `pc_commit`；退役时 `pc_retire`；
  准入失败平衡引用。多模态完全绕过。

---

## 4. Chat 渲染

### 4.1 `render_chat`

`render_chat(tmpl, msgs, add_generation_prompt, enable_thinking, tools_json)` 先试
`render_chat_template`，失败回退 `render_chat_builtin`。`chat_msg` 除 role/content/parts 外还带
`reasoning_content`、`tool_calls`（`{id,name,arguments(JSON 字符串)}`）、`tool_call_id`、`name`。

### 4.2 minja Jinja 封装

* `tmpl` 为空立即返回 false。
* `PF_CHAT_TMPL_DEBUG` 控制是否把异常写到 stderr。
* 每次调用**新建** `minja::chat_template`（模板无解析缓存）。
* 消息转 `nlohmann::ordered_json`：文本消息 `{role, content:<string>}`；含 part 的消息
  `{role, content:[{type:"image"}|{type:"text",text}]}`；assistant 带 `reasoning_content`，
  带 `tool_calls`（`arguments` 解析成对象交给模板），`tool` 带 `tool_call_id`/`name`。
* 输入：`messages`、`tools`（解析 `tools_json`）、`add_generation_prompt`、
  `extra_context = {"enable_thinking": bool}`。`bos_token`/`eos_token` 有意留空，因为渲染结果是完整
  prompt 字符串，由分词器按 `parse_special` 编码。
* `ct.apply` 抛出的任何异常都被捕获并返回 false，调用者回退。

### 4.3 内置 ChatML 回退

* 空消息 → 空串。
* `content_of`：无 parts 返回 `content`；否则拼接，图片 part 替换为
  `<|vision_start|><|image_pad|><|vision_end|>`。
* 有 `tools_json` 时在 system 消息里注入 Qwen 风格的 `# Tools` 说明块。
* user/assistant/tool 各按 ChatML 渲染；assistant 优先用 `reasoning_content`，否则识别 `</think>`
  拆出 reasoning，仅在“最新 user 之后的 assistant 轮”保留 `<think>...</think>`；`tool_calls` 渲染为
  `<tool_call><function=...><parameter=...>` 块。
* `add_generation_prompt` 时追加 `<|im_start|>assistant\n`；`enable_thinking` 追加 `<think>\n`，否则
  追加一个空的 ` <think>\n\n</think>\n\n`。

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
* `finish_reason = stopped ? "stop" : (有工具调用 ? "tool_calls" : seq->finish_reason)`。

因为不是采样期停止，stop 之后的 token（以及 mm 路径 EOS 之后）可能已在内部生成。

---

## 6. 多模态请求处理

* 启动时若有 `mmproj_path` 则 `mm.vm.load()`，从视觉超参导出 `image_preproc_cfg`，失败非致命（打日志，
  后续图片/视频请求 400）；若有 `audio_mmproj_path` 则 `mm.am.load()` 导出 `audio_preproc_cfg`，
  失败同样非致命（音频请求 400）。
* `parse_messages` 按顺序收集媒体的 `media_part`（`image_url`/`image`、`video_url`/`video`、
  `input_audio`（带 `data`+`format`）、`audio_url`），保留文本/媒体交错，`chat_part` 带 `kind` 标记。
* `load_media_bytes` 接受 `data:`（要求 `;base64`；`b64_decode` 容忍空白、遇到 `=` 停止）、内联
  `input_audio` base64 与 `http(s)://`（`fetch_http_image` 用 httplib 下载，跟随重定向，10 s 超时、
  10 MB 上限——超过即中断传输；HTTPS 需要构建时找到 OpenSSL，否则报错；`PF_MM_URL_FETCH=0` 可整体
  关闭远程抓取）。其它 scheme 一律 400。`parse_http_url` 解析 `scheme://host[:port]/path`，缺 host 或
  端口非法即拒绝。
* 图片 `mm_image_decode_mem` → `mm_image_preprocess`；视频 `mm_video_decode_mem`（`max_frames`/`max_side`
  由 `server_config` 控制）；音频 `mm_audio_decode_bytes`（WAV 原生，其它容器落临时文件走 ffmpeg CLI），
  失败均 400。
* `render_chat(..., add_generation_prompt=true, thinking, tools_json)` → `mm_build_prompt_mixed_device`
  （按 `order` 把 image/video/audio 分别经视觉/音频塔编码进 `e.d_img_embd`）→ 用**扩展后** token 数检查
  长度 → `run_mm_choice`（非流式）或 `stream_chat_mm_choices`（流式，逐个 choice 顺序生成）。
* 全程持有 `mm_req`；`generate_mm` 内部持有 `engine::mtx`。
* 注意：mm 路径 `finish_reason` 仅在 stop 命中时为 `"stop"`，EOS 终止报为 `"length"`（有工具调用时为
  `"tool_calls"`）；`n>1` 顺序生成；绕过前缀缓存与连续批处理。

---

## 7. 诊断环境变量

| 变量 | 作用 |
|---|---|
| `PF_SRV_TIME` | tokenize 计时与每序列 `prefill/wait/admit/chunks/reused` 日志 |
| `SCHED_DEBUG` | 准入/prefill/decode 决策日志 |
| `PF_CHAT_TMPL_DEBUG` | 记录导致回退 ChatML 的模板异常 |
| `PF_MM_URL_FETCH` | `0` 禁止下载远程 `http(s)://` 图片/视频/音频（base64 `data:` 不受影响） |
| `PF_AV_FFMPEG` | 音频/视频解码的 ffmpeg 可执行路径（默认 `ffmpeg`） |
| `PF_GEMM_DNNL` | 通过 `batched_prefill_fit` 改变 prefill chunk 尺寸 |

---

## 8. OpenAI 字段支持矩阵

图例：✅ 已实现；⚠️ 部分实现/仅近似；❌ 未实现（静默忽略或返回 400）。本表以当前 `server.cpp`
的解析与响应代码为准。

### 8.1 `POST /v1/chat/completions` 请求

| 字段 | 状态 | 说明 |
|---|---|---|
| `model` | ✅ | 仅回显/用于选择（单模型）；未知 id 不报错 |
| `messages` | ✅ | 见 8.4 |
| `stream` | ✅ | SSE；每 choice 一个生产者线程 |
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
| `seed` | ⚠️ | 尽力而为；`n>1` 时按 `seed+c` 派生 |
| `ignore_eos` | ✅ | llama.cpp 扩展 |
| `logprobs` / `top_logprobs` | ✅ | 生成 token 的 logprob（惩罚/偏置、温度缩放后的完整词表 log-softmax）；chat 返回 `logprobs.content[]`，图片(multimodal)路径不支持 |
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
| 图片 part `{"type":"image_url","image_url":{"url":...}}` | ✅ | `data:` base64 或 `http(s)://`（`PF_MM_URL_FETCH=0` 关闭远程） |
| 视频 part `{"type":"video_url","video_url":{"url":...}}` | ✅ | 同 `image_url`，需 `--mmproj`，由 `max_frames`/`max_side` 控制采样 |
| 音频 part `{"type":"input_audio","input_audio":{"data":...,"format":"wav"}}` | ✅ | `data` 为 base64 编码的内联音频（WAV/MP3/OGG…）；需 `--audio-mmproj` |
| 音频 part `{"type":"audio_url","audio_url":{"url":...}}` | ✅ | `data:` 或 `http(s)://`，同上 |

### 8.2 `POST /v1/chat/completions` 响应

| 字段 | 状态 | 说明 |
|---|---|---|
| `id` / `object` / `created` / `model` | ✅ | `object:"chat.completion"`（流式为 `.chunk`） |
| `choices[].index` | ✅ | |
| `choices[].message.role` | ✅ | `"assistant"` |
| `choices[].message.content` | ✅ | 仅有工具调用且无文本时为 `null` |
| `choices[].message.reasoning_content` | ✅ | 开启 thinking 且解析到推理时 |
| `choices[].message.tool_calls[]` | ✅ | `{id,type,function:{name,arguments}}`，`arguments` 为 JSON 字符串 |
| `choices[].finish_reason` | ✅ | `stop` / `length` / `tool_calls` |
| `choices[].logprobs` | ✅ | chat 为 `{content:[{token,logprob,bytes,top_logprobs}]}`；未请求时为 `null`；图片路径不产出 |
| `usage.{prompt,completion,total}_tokens` | ✅ | 多 choice 求和 |
| `usage.completion_tokens_details.reasoning_tokens` | ✅ | reasoning_content 覆盖的生成 token 数（按解析出的推理片段计） |
| `usage.completion_tokens_details.{audio,accepted_prediction,rejected_prediction}_tokens` | ✅ | 恒为 0（占位） |
| `usage.prompt_tokens_details.cached_tokens` | ✅ | 前缀缓存命中的 prompt token 数（`sequence::reused`） |
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
| `best_of` | ✅ | 非流式；生成 `best_of` 条，按 chosen logprob 求和取 top-`n`；`best_of < n` 或流式且 `best_of>1` 返回 400 |
| `logprobs` / `logit_bias` | ✅ | completion 用 legacy 数组 `{tokens,token_logprobs,top_logprobs,text_offset}`；prompt 位置（`echo`）不产出 logprob |
| `temperature` / `top_p` / `top_k` / `min_p` / 惩罚 / `seed` / `ignore_eos` / `stop` | ✅ | 同 chat |
| 响应 `choices[].{index,text,finish_reason}`、`usage` | ✅ | `object:"text_completion"` |
| 并发上限 | ⚠️ | `prompt数 × max(n,best_of) > 64` 返回 400 |

### 8.4 `messages` 元素

| 字段 | 状态 | 说明 |
|---|---|---|
| `role` | ✅ | `system` / `user` / `assistant` / `tool` |
| `role:"function"`（旧式） | ❌ | 未处理（模板会报错并回退内置渲染） |
| `content` | ✅ | 字符串、`null`、或 part 数组（`text` / `image_url` / `image` / `video_url` / `video` / `input_audio` / `audio_url`） |
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
| `GET /v1/models/{id}` | ✅ | 未知 id 返回 404 |

### 8.6 未提供的端点

`/v1/embeddings`、`/v1/tokenize`、`/v1/detokenize`、`/v1/audio/*`、`/v1/images/*`、
`/v1/moderations`、`/v1/files`、`/v1/batches` 均未实现。
