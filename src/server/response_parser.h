#pragma once
#include <functional>
#include <string>
#include <vector>

namespace si {

// One function call decoded from a model reply, in the OpenAI wire shape.
struct response_tool_call {
    std::string id;        // "call_..." (assigned by the parser)
    std::string name;      // function name
    std::string arguments; // JSON object serialized as a string
};

enum class response_piece_kind { reasoning, content, tool_call };

struct response_piece {
    response_piece_kind kind = response_piece_kind::content;
    std::string text;   // reasoning/content fragment (empty for tool_call)
    response_tool_call call; // valid when kind == tool_call
};

// Splits a model reply into the OpenAI `reasoning_content` / `content` /
// `tool_calls` streams.  The Qwen3.5 template streams reasoning as plain text
// after a `<think>` prompt and closes it with `</think>`, then may emit
// `<tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>`
// blocks (see docs/design/09-server.md).  `feed()` accepts raw token pieces and
// emits only complete, unambiguous fragments: partial markers at a piece
// boundary are held back until they resolve, so a chunk never contains half a
// `</think>` or `<tool_call>`.
//
// When `thinking` is false the parser never produces reasoning.  When
// `parse_tools` is false the tool markers are passed through as content (used
// by the plain /v1/completions endpoint).
class response_parser {
  public:
    using emit_fn = std::function<void(const response_piece &)>;

    response_parser(bool thinking, bool parse_tools, emit_fn emit);

    void feed(const std::string & piece);
    // flush any held-back text and finish the parse
    void finish();

    const std::string & reasoning() const {
        return reasoning_;
    }
    const std::string & content() const {
        return content_;
    }
    const std::vector<response_tool_call> & tool_calls() const {
        return tools_;
    }
    bool has_reasoning() const {
        return thinking_;
    }

  private:
    enum class state { reasoning, content, tool };

    void process(bool final);
    void push_reasoning(std::string text);
    void push_content(std::string text, bool before_tool);

    bool thinking_ = false;
    bool parse_tools_ = false;
    emit_fn emit_;
    state st_ = state::content;
    std::string buf_;
    bool strip_content_leading_ = false;
    bool reason_start_ = true;
    std::string reasoning_;
    std::string content_;
    std::vector<response_tool_call> tools_;
};

// Parse a single `<function=...>...</function>` (or JSON) tool block body.
// Returns false when the block is not a function call.
bool parse_tool_block(const std::string & block, response_tool_call & out);

} // namespace si
