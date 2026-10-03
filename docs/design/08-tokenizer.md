# 设计 08：分词器

覆盖 `src/model/tokenizer.{h,cpp}`；pre-tokenize 的正则分词器、byte-encoding 表与 Unicode 类别表是
vendored 的 llama.cpp Unicode 实现（`third_party/unicode.{h,cpp}`，MIT）。Chat 模板渲染见
[09-server.md](09-server.md) §4，多模态占位符见 [10-multimodal.md](10-multimodal.md)。

---

## 1. 范围

从模型 GGUF 的 `tokenizer.ggml.*` 元数据构建一个 **GPT-2 byte-level BPE** 分词器，提供
`encode` / `decode` / `token_piece`。它是纯主机代码，不涉及 GPU。

---

## 2. 数据结构（`tokenizer.h:17-37`）

| 成员 | 内容 |
|---|---|
| `token_text` | 原始 token 字符串（与 GGUF 完全一致） |
| `token_type` | gguf token type id |
| `token_to_id` | `unordered_map<string,int>` |
| `bpe_rank` | `(left,right) → rank`，自定义 `pair_hash`（`tokenizer.h:11-15`） |
| `special_ids` | 特殊 token，按文本长度降序（最长匹配优先） |
| `n_vocab` | 词表大小 |
| 特殊 id | `eos_id`、`eot_id`、`pad_id`、`im_start_id`、`im_end_id`、`think_id`、`endthink_id` |

---

## 3. 词表加载（`tokenizer.cpp:33-91`）

* 必须有 `tokenizer.ggml.tokens` 且它是数组，否则抛 `gguf: missing tokenizer tokens`（`:37-39`）。
* 填充 `token_text`/`token_to_id`；`token_type` 默认 1（NORMAL），若存在
  `tokenizer.ggml.token_type` 则逐元素覆盖，长度不足时只覆盖前缀（`:48-52`）。
* `tokenizer.ggml.merges` 每项是 `"A B"`；分割点用 `word.find(' ', 1)`（保留前导空格），
  `bpe_rank[{A,B}] = i`。**为什么从 1 开始找**：byte-encoding 之后左侧符号本身就可能以空格（`Ġ`）
  开头，从 0 找会把符号内部切开。`emplace` 意味着重复 pair 保留先出现（rank 更小）的那个（`:54-64`）。
* `special_ids` 收集类型 2（UNKNOWN）、3（CONTROL）、4（USER_DEFINED），按文本长度降序排序
  —— 长度降序是为了 §4 的最长匹配（`:66-72`）。
* 具名特殊 token 按文本解析（`:74-84`）：`<|im_end|>` → `eos_id`、`<|endoftext|>` → `eot_id`、
  `<|vision_pad|>` → `pad_id`、`<|im_start|>`、`<|im_end|>`、`<think>`、`</think>`。
  回退：没有 `<|im_end|>` 时 `eos_id = tokenizer.ggml.eos_token_id`（默认 0）；没有 `<|endoftext|>`
  时 `eot_id = eos_id`（`:85-90`）。成员声明处的注释（`tokenizer.h:24-25`）标的就是这套默认名。

---

## 4. 编码（`tokenizer.cpp:93-196`）

1. `bpe_one`（`:95-125`）：把 pre-token 拆成 UTF-8 **字符**，然后反复合并 rank 最低的相邻 pair 直到
   无可合并。每轮全扫一遍取最小 rank（不是 llama.cpp 的 linked-list 版本），所以是 O(n²) 但预分词
   后的片段很短。
2. `encode_plain`（`:127-151`）：用 `unicode_regex_split` + **Qwen3.5 风格正则**做 pre-tokenize
   （`byte_encode=true`，把每个字节映射到 GPT-2 byte-encoding 字母表：`third_party/unicode.cpp:1403-1405`
   → `:196-212`，映射表在 `:148-170`），逐词 `bpe_one` 后映射到 id。
   * 正则写死在 `tokenizer.cpp:129-130`，在 vendored splitter 里精确匹配到
     `unicode_regex_split_custom_qwen35`（`third_party/unicode.cpp:1063-1064`，实现 `:610-736`）——
     是手写码点扫描器，不走 `std::regex`。与 Qwen2 的唯一差别是字母 run 也吞组合记号
     `[\p{L}\p{M}]+`（`:608-609`）。数字是**逐个** `\p{N}`（不是 GPT-2 的 `\p{N}{1,3}`），所以
     `"123"` 会切成 3 个 pre-token。
   * `bpe_one` 的初始符号已经是 byte-encoded 的单字符，所以 `token_to_id` 通常直接命中。
   * 查不到的符号按 UTF-8 字符逐个再查一次，仍查不到就**静默丢弃**（`:140-147`）。
3. `encode(text, parse_special=true)`（`:155-196`）：
   * `parse_special=false` 或 `special_ids` 为空时只做 plain encode（`:157-160`）。
   * 否则从左到右扫描：对 `special_ids`（已按长度降序）逐个 `text.compare`，空文本的 special token
     被跳过（`:174-184`）；命中就把 id 原样输出，未命中的字符进 `pending` 缓冲，直到下次命中或结尾
     才 `encode_plain`（`:165-194`）。逐字符扫 + 缓冲是为了让跨片段的普通文本仍按完整的 pre-tokenize
     规则切分。
   * **不自动添加 BOS**（`tokenizer.h:31` 的 “BOS handling” 注释已过时）；chat 模板负责插入
     `<|im_start|>`，见 §6。
   * CLI 与服务器都用 `parse_special=true`（`src/main.cpp:523`、`:526`，`src/server/server.cpp:1707`）。

---

## 5. 解码（`tokenizer.cpp:198-215`）

* `token_piece(id)`：越界返回 `""`；类型 2/3/4 原样返回；否则 `byte_decode(token_text[id])`。
* `byte_decode`（`:12-29`）：按 `unicode_len_utf8` 取一个码点，用 `unicode_utf8_to_byte` 反转 GPT-2
  字节映射（`third_party/unicode.cpp:1167-1170`）；长度越过字符串末尾时按原始字节原样拷贝（`:18-22`）。
  反查用 `map.at()`，查不到会抛 `std::out_of_range` —— 它覆盖的正是 `unicode_byte_to_utf8` 能吐出的
  全部码点（两张表由同一段范围生成，`third_party/unicode.cpp:148-194`），所以只有「没走 byte-encoding
  的 NORMAL token」才会踩到；类型 2/3/4 走原样返回正是为了避开它。
* `decode(ids)` 拼接 `token_piece`。

`token_piece` 是流式输出的唯一来源。单个 token piece 可能截断多字节 UTF-8，服务器/CLI 用
`utf8_stream_buffer`（`src/server/chat_util.h:8-57`）缓冲到边界再发出，见
[09-server.md](09-server.md) §4.4。

---

## 6. chat 模板

* `model::chat_template` 是 GGUF `tokenizer.chat_template` 的原始 Jinja 字符串
  （`src/model/model.cpp:66-68`），没有该键时是空串。
* 优先级只有一层：`render_chat`（`src/server/chat.cpp:37-44`）先试 `render_chat_template`
  （`src/server/chat_template.cpp:21-107`），失败就调 `render_chat_builtin`（`chat.cpp:85-180`）。
  失败的判定是 **`tmpl` 非空 且 minja 的 `ct.apply` 不抛异常**（`chat_template.cpp:23-25`、`:99-106`）。
* 传给模板的输入（`chat_template.cpp:30-97`）：`messages`（转 `nlohmann::ordered_json`；带 part 的消息
  把 content 变成数组，IMAGE/VIDEO/AUDIO 分别写成 `{"type":"image"|"video"|"input_audio"}`，见 `:39-45`）、
  `tools`（解析 `tools_json`）、`add_generation_prompt`、`extra_context = {"enable_thinking": bool}`。
  `bos_token`/`eos_token` 有意留空（`:28`），因为渲染结果是完整 prompt 文本，交给
  `encode(parse_special=true)` 自己切 —— 这也是 §4 不加 BOS 的原因。
* `ct.apply(inputs)` 用默认 `chat_template_options`（`third_party/minja/chat-template.hpp:55-61`、
  `:329-331`）：polyfill 全开、`use_bos_token`/`use_eos_token` 也开，但因为上面两个 token 是空串而没有
  副作用。**每次调用都新建 `minja::chat_template`，模板解析没有缓存。**
* 失败诊断只有 `PF_CHAT_TMPL_DEBUG`：它打印 `ex.what()`（`chat_template.cpp:26`、`:102-104`）以及
  tools JSON 解析失败（`:89-93`）。这是分辨「模板坏了」和「forward 坏了」的最快手段，配合
  `PF_DUMP_PROMPT`（`src/main.cpp:531-538`）打印真正送进模型的 prompt。

### 必须保留的坑：minja 的 `is undefined`

`third_party/minja/minja.hpp:1350-1351` 有一处本地补丁，给 `is` 运算符补上 Jinja 的 `undefined` 测试
（`if (name == "undefined") return l.is_null();`，与 `:1349` 的 `defined` 互为镜像）；上游 HEAD 同样
缺这个测试。它是 AGENTS.md 里记录的 `third_party/` 不改动的**唯一**例外。

为什么它是承重的：Qwen3.8-27B 的模板以 `{%- if enable_thinking is undefined or ... %}` 开头。没有这个
测试时 `is` 分支会在 `minja.hpp:1354` 抛 `Unknown type for 'is' operator: undefined`，
`render_chat_template` 捕获后返回 false，内建回退就产出一个**模型没被训练过的 prompt**，chat 回答退化
成单个 token —— 而且不报错，`PF_CHAT_TMPL_DEBUG` 是唯一的线索。

注意本仓库的 0.8B 参考模型用的是 `is defined`（`{%- if enable_thinking is defined and enable_thinking
is true %}`），所以它**测不出**这个补丁被回退，`test_chat_template` 也不会因此失败：换模型时才暴露。

---

## 7. 测试

* `tests/model/test_tokenizer.cpp`（纯 CPU）：加载参考模型 GGUF，打印 vocab 与 6 个特殊 id
  （`eos`/`eot`/`im_start`/`im_end`/`think`/`endthink`；`pad_id` 不打印），对 8 类输入做往返（ASCII、
  含标点与换行的代码片段、中文、ChatML 形状的 `user\n你好\nassistant\n`、空白/制表/CRLF、emoji、
  字面 `Ġ` 风格 token），打印 `roundtrip: OK|MISMATCH`。它**恒返回 0**（`:38`），是一个打印式诊断而
  不是门禁 —— 分词本身目前没有会失败的测试。
* `tests/model/test_chat_template.cpp`（纯 CPU）：拿 GGUF 自带模板过 minja，与参考 Jinja2
  （`python3 -m jinja2`）产出的期望串对拍 6 个 case（system_user / user_thinking / multi_turn /
  assistant_reasoning / tool_grouping / content_trimmed），外加内建回退、tools + tool_calls +
  tool_response、`reasoning_content` 三个子检查；失败返回非 0（`:46-153`）。这是 chat 渲染的唯一门禁。

---

## 8. 特殊 token 与停止条件（服务器侧）

分词器只管「文本 ↔ id」，下面几件事都在 `src/server/`：

* **占位符的产生**：内建回退的 `content_of` 把 IMAGE/VIDEO/AUDIO part 替换成
  `<|vision_start|><|image_pad|><|vision_end|>` / `...<|video_pad|>...` /
  `<|audio_start|><|audio_pad|><|audio_end|>`（`chat.cpp:95-109`）；minja 路径不自己写占位符，而是把
  part 变成 `{"type":"image"|"video"|"input_audio"}` 交给模型模板（`chat_template.cpp:39-45`）。
* **占位符的切分与抑制**：它们是特殊 token —— 在 GGUF 里被标为类型 2/3/4 时，`encode` 原样切出、
  `token_piece` 原样返回，不经过 byte-level 编码。它们也**永不进 `tok_embd`**：`embed` kernel 按
  `step_info::img_row[t]` 从视觉塔的输出逐元素拷贝（`src/backend/gpu/kernels/embed.cpp:30-36`），
  见 [10-multimodal.md](10-multimodal.md)。
* **停止条件**：只有**字符串** stop。`parse_stop` 接受 string 或 string[]，数组最多 `kMaxStops = 16`
  条（`server.cpp:98`、`:197-213`）；`stop_token_ids` **未实现**。过滤由 `stop_filter` 在生成文本上做，
  见 [09-server.md](09-server.md) §5。回复里 `<think>` / `</think>` 与 tool-call 标记的切分是
  `response_parser` 的事，见 [09-server.md](09-server.md) §2.8。