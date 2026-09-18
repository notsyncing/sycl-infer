# 设计 08：分词器

覆盖 `src/model/tokenizer.{h,cpp}`。Chat 模板渲染见 [09-server.md](09-server.md)。

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
| `bpe_rank` | `(left,right) → rank`，自定义 `pair_hash` |
| `special_ids` | 特殊 token，按文本长度降序（最长匹配优先） |
| `n_vocab` | 词表大小 |
| 特殊 id | `eos_id`、`eot_id`、`pad_id`、`im_start_id`、`im_end_id`、`think_id`、`endthink_id` |

---

## 3. 词表加载（`tokenizer.cpp:33-91`）

* 必须有 `tokenizer.ggml.tokens` 数组，否则抛 `gguf: missing tokenizer tokens`。
* 填充 `token_text`/`token_to_id`；`token_type` 默认 1（NORMAL），若存在
  `tokenizer.ggml.token_type` 则逐元素覆盖。
* `tokenizer.ggml.merges` 每项是 `"A B"`；分割点用 `word.find(' ', 1)`（保留前导空格），
  `bpe_rank[{A,B}] = i`。
* `special_ids` 收集类型 2（UNKNOWN）、3（CONTROL）、4（USER_DEFINED），按文本长度降序排序。
* 具名特殊 token 按文本解析：`<|im_end|>` → `eos_id`、`<|endoftext|>` → `eot_id`、
  `<|vision_pad|>` → `pad_id`、`<|im_start|>`、`<|im_end|>`、`<think>`、`</think>`。
  回退：没有 `<|im_end|>` 时 `eos_id = tokenizer.ggml.eos_token_id`（默认 0）；没有 `<|endoftext|>`
  时 `eot_id = eos_id`。

---

## 4. 编码（`tokenizer.cpp:93-196`）

1. `bpe_one`（`:95-125`）：把 pre-token 拆成 UTF-8 chunk，然后反复合并 rank 最低的相邻 pair，直到无可
   合并。
2. `encode_plain`（`:127-151`）：用 `unicode_regex_split` + **Qwen 风格正则**做 pre-tokenize
   （`byte_encode=true`，把输入字节映射到 GPT-2 byte-encoding 字母表），逐词 `bpe_one` 后映射到 id。
   未知符号回退为逐 UTF-8 字符的 token id；未知字符被丢弃。
3. `encode(text, parse_special=true)`（`:155-196`）：
   * `parse_special=false` 或无特殊 token 时只做 plain encode；
   * 否则从左到右对 `special_ids` 最长匹配，普通文本片段走 `encode_plain`，匹配到的特殊 token 原样
     输出。
   * **不自动添加 BOS**（头注释的 “BOS handling” 已过时）；chat 模板负责插入 `<|im_start|>`。

---

## 5. 解码（`tokenizer.cpp:198-215`）

* `token_piece(id)`：越界返回 `""`；类型 2/3/4 原样返回；否则 `byte_decode(token_text[id])`。
* `byte_decode`（`:12-29`）：用 vendored 的 `unicode_len_utf8` / `unicode_utf8_to_byte` 反转 GPT-2
  字节映射，把每个 byte-encoded 的 UTF-8 码点还原成原始字节。
* `decode(ids)` 拼接 `token_piece`。

流式输出时，单个 token piece 可能被截断多字节 UTF-8；服务器/CLI 用 `utf8_stream_buffer`
（`src/server/chat_util.h`）缓冲到边界再发出，见 [09-server.md](09-server.md)。

---

## 6. chat 模板

`model::chat_template` 是 GGUF `tokenizer.chat_template` 的原始 Jinja 字符串。渲染由
`render_chat`（`src/server/chat.cpp`）完成：优先用 minja 渲染该模板，失败则回退内置 ChatML。详见
[09-server.md](09-server.md)。

---

## 7. 测试

`tests/model/test_tokenizer.cpp`（纯 CPU）：加载参考模型 GGUF，打印 vocab/特殊 id，对 8 类输入做
往返（ASCII、CJK、空白/制表/CRLF、含特殊 token 的 chat 模板串、emoji、字面 `Ġ` 风格 token），
打印 `roundtrip: OK|MISMATCH`。它不设置失败退出码；严格校验由 `test_chat_template` 承担。
