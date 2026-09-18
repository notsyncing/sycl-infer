# 设计 09：HTTP 服务与连续批处理

覆盖 `src/server/server.{h,cpp}`、`scheduler.{h,cpp}`、`chat.{h,cpp}`、`chat_template.{h,cpp}`、
`chat_util.h`，以及 `engine` 的采样接口。

---

## 1. 组件与数据流

| 层 | 文件 | 职责 |
|---|---|---|
| HTTP | `server.cpp` | httplib 服务、路由、JSON、SSE、CORS、图像处理、视觉 tower 调度 |
| 批处理 | `scheduler.cpp` | `sequence` 生命周期、连续批处理、持有 `engine::mtx` |
| Chat 模板 | `chat.cpp` / `chat_template.cpp` | minja Jinja 渲染 + 内置 ChatML 回退 |
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
| GET | `/v1/models` | OpenAI 列表，`created` 硬编码，`owned_by:"local"` |
| POST | `/v1/chat/completions` | 流式 + 非流式，支持 stop、`ignore_eos`、`stream_options.include_usage` |
| POST | `/v1/completions` | 纯文本补全，流式 + 非流式 |
| OPTIONS | `.*` | 204 + CORS |

无鉴权，无 `/v1/embeddings`、`/tokenize`。

### 2.4 请求解析

* JSON 解析失败 → 400 `{"error":{"message":"invalid json"}}`。
* `parse_messages`（`:199-243`）把 `role` 默认 `"user"`；`content` 可以是字符串或 part 数组；
  part `type` 为 `"image_url"`/`"image"` 记图片，否则文本。若某消息无图片 part，则清空 `parts` 走纯文本
  快速路径。
* `thinking` 来自 `chat_template_kwargs.enable_thinking`（默认 false）。
* `gen_params` 来自 `parse_params`，`stops` 来自 `parse_stop`（字符串或字符串数组）。

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

`/v1/completions` 的 `prompt` 数组**只用第一个元素**；无 `n`/`echo`/`logprobs`/`suffix`/`best_of`。

### 2.5 长度拒绝

`reject_too_long`（`:297-314`）返回 400 + `code:"context_length_exceeded"`，并带 `max_seq` 与
`prompt_tokens`。原因（注释）：否则 `scheduler::admit` 会静默退役超长 prompt，返回空的 200 流。
对 chat 文本、completion、以及**扩展后**的多模态 prompt 都调用。

其它 400：无 `--mmproj` 的图片请求；图片下载/解码/预处理失败。

### 2.6 非流式响应

* `make_chat_response`：`object:"chat.completion"`，单 choice `{message:{role,content},finish_reason}`，
  含 `usage`。
* `make_completion_response`：`object:"text_completion"`，choice `{text,finish_reason}`。
* chat 非流式：`sched.submit` 后循环 `seq->pop` 阻塞累积文本，每次追加都扫描 stop 字符串（`rfind`），
  命中则截断并记 `stopped`。`finish_reason = stopped ? "stop" : seq->finish_reason`。

### 2.7 流式 SSE

通用形态（`run_stream` 与多模态）：

* 响应头 `text/event-stream`、`Cache-Control: no-cache`、`Connection: keep-alive`；
* 用 `set_chunked_content_provider`，内容 reader 从 `sse_queue` 取，队列结束调 `sink.done()`；资源释放
  回调 join 生产者线程。
* 每个事件是 `"data: " + dump_json(j) + "\n\n"`；`dump_json` 用 `error_handler_t::replace` 防止坏
  UTF-8 抛异常。

chat chunk schema：

```json
{"id":"chatcmpl-N","object":"chat.completion.chunk","created":T,"model":"...",
 "choices":[{"index":0,"delta":{"content":"..."},"finish_reason":null}]}
```

帧顺序：若干 content chunk（在 stop 处截断）→ 带 `finish_reason` 的最终 chunk（chat 为 `delta:{}`，
completion 为 `text:""`）→ 若 `include_usage` 则一个 `choices:[]` 的 usage chunk →
`data: [DONE]`。

### 2.8 线程模型

* httplib 每个连接派一个工作线程执行 handler。
* 非流式 handler 阻塞在 `seq->pop`；流式 handler 起一个生产者线程后立即返回，由 chunked provider 驱动。
* `scheduler::submit` 多线程安全（锁 `scheduler::m`）；每个 `sequence` 的输出由自带 mutex+cv 保护。
* 所有引擎工作由 `engine::mtx` 串行化；多模态额外用文件级 `mm_req` 串行化（共享 `d_img_embd`）。

### 2.9 关闭与信号

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

`render_chat(tmpl, msgs, add_generation_prompt, enable_thinking)` 先试
`render_chat_template`，失败回退 `render_chat_builtin`。

### 4.2 minja Jinja 封装

* `tmpl` 为空立即返回 false。
* `PF_CHAT_TMPL_DEBUG` 控制是否把异常写到 stderr。
* 每次调用**新建** `minja::chat_template`（模板无解析缓存）。
* 消息转 `nlohmann::ordered_json`：文本消息 `{role, content:<string>}`；含 part 的消息
  `{role, content:[{type:"image"}|{type:"text",text}]}`。
* 输入：`messages`、空 `tools`、`add_generation_prompt`、
  `extra_context = {"enable_thinking": bool}`。`bos_token`/`eos_token` 有意留空，因为渲染结果是完整
  prompt 字符串，由分词器按 `parse_special` 编码。
* `ct.apply` 抛出的任何异常都被捕获并返回 false，调用者回退。

### 4.3 内置 ChatML 回退

* 空消息 → 空串。
* `content_of`：无 parts 返回 `content`；否则拼接，图片 part 替换为
  `<|vision_start|><|image_pad|><|vision_end|>`。
* 系统消息渲染为 `<|im_start|>system\n...\n<|im_end|>\n`。
* user/assistant/tool 各按 ChatML 渲染；assistant 会识别 `</think>` 拆出 reasoning，仅在“最新 user
  之后的 assistant 轮”保留 `<think>...</think>`。
* `add_generation_prompt` 时追加 `<|im_start|>assistant\n`；`enable_thinking` 追加 `<think>\n`，否则
  追加一个空的 ` <think>\n\n</think>\n\n`。

### 4.4 `utf8_stream_buffer`（`chat_util.h`）

`push(piece)` 按 UTF-8 首字节推算序列长度，只返回“完整且合法续字节”的安全前缀，把尾部不完整/非法
序列留在 `pending`；`flush()` 返回剩余部分（生成结束时避免丢尾部）。被调度器（每行一个）、多模态
生成、CLI 流式输出使用。

---

## 5. 停止字符串

stop 字符串由**服务器侧**在生成文本上匹配：

* 非流式：每次追加后用 `rfind` 扫描全部 stop，命中即截断并记 `stopped`。
* 流式：在完整累积缓冲上扫描，命中则发出截至截断点的前缀并 break。
* `finish_reason = stopped ? "stop" : seq->finish_reason`。

因为不是采样期停止，stop 之后的 token（以及 mm 路径 EOS 之后）可能已在内部生成。

---

## 6. 多模态请求处理

* 启动时若有 `mmproj_path` 则 `mm.vm.load()`，从视觉超参导出 `image_preproc_cfg`，失败非致命（打日志，
  后续图片请求 400）。
* `parse_messages` 按顺序收集图片 URL，保留文本/图片交错。
* `decode_image_url` **只接受 `data:` URL**（注释：有意排除远程抓取以避免 SSRF 代理），要求
  `;base64`；`b64_decode` 容忍空白、遇到 `=` 停止。
* 每张图 `mm_image_decode_mem` → `mm_image_preprocess`，失败 400。
* `render_chat(..., add_generation_prompt=true, thinking)` → `mm_build_prompt_device`（把合并嵌入写进
  `e.d_img_embd`）→ 用**扩展后** token 数检查长度 → `run_mm_generate`（非流式）或生产者线程（流式）。
* 全程持有 `mm_req`；`generate_mm` 内部持有 `engine::mtx`。
* 注意：mm 路径 `finish_reason` 仅在 stop 命中时为 `"stop"`，EOS 终止报为 `"length"`；不支持 `n>1`；
  绕过前缀缓存与连续批处理。

---

## 7. 诊断环境变量

| 变量 | 作用 |
|---|---|
| `PF_SRV_TIME` | tokenize 计时与每序列 `prefill/wait/admit/chunks/reused` 日志 |
| `SCHED_DEBUG` | 准入/prefill/decode 决策日志 |
| `PF_CHAT_TMPL_DEBUG` | 记录导致回退 ChatML 的模板异常 |
| `PF_GEMM_DNNL` | 通过 `batched_prefill_fit` 改变 prefill chunk 尺寸 |
