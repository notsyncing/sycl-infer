#pragma once
#include <string>
#include <utility>
#include <vector>

namespace si {

// Content-part modality for multimodal messages.  `kind` selects the
// placeholder substituted into the rendered template; `data`/`format`/`url`
// carry the raw bytes for image/video/audio parts.  Only `text` is mandatory
// (a text part is `kind == TEXT` with an empty payload).
enum class chat_part_kind { TEXT, IMAGE, VIDEO, AUDIO };

struct chat_part {
    chat_part_kind kind = chat_part_kind::TEXT;
    std::string text;
    // multimodal payload: decoded (or to-be-decoded) media, with the transport
    std::string data;       // raw decoded bytes (image/video/audio)
    std::string format;     // image: png/jpeg; audio: wav/mp3/ogg...
    std::string url;        // source data:/http(s) URL when set
};

// One assistant function call as received in a request (OpenAI `tool_calls`).
struct chat_tool_call {
    std::string id;
    std::string name;
    std::string arguments; // JSON object, kept as a string on the wire
};

struct chat_msg {
    std::string role;
    std::string content;
    // optional structured content: when non-empty it takes precedence over
    // `content` and lets a message interleave text and image parts
    std::vector<chat_part> parts;
    // assistant-only reasoning (OpenAI `reasoning_content`); also accepted
    // inline inside `content` as `<think>...</think>` by the template
    std::string reasoning_content;
    // assistant-only function calls, rendered by the template as `<tool_call>`
    std::vector<chat_tool_call> tool_calls;
    // role == "tool": which call this is the result of
    std::string tool_call_id;
    std::string name;

    chat_msg() = default;
    chat_msg(std::string r, std::string c, std::vector<chat_part> p = {})
        : role(std::move(r)), content(std::move(c)), parts(std::move(p)) {
    }
};

// Render the conversation with the model's Jinja chat template `tmpl` (the
// GGUF `tokenizer.chat_template` string).  Falls back to the built-in Qwen3.5
// ChatML renderer when the template is empty or uses unsupported syntax.
// `tools_json` is the request's OpenAI `tools` array serialized as JSON (empty
// when no tools are offered); it is passed to the template as `tools`.
std::string render_chat(const std::string & tmpl, const std::vector<chat_msg> & msgs, bool add_generation_prompt,
                        bool enable_thinking, const std::string & tools_json = "");

// Built-in ChatML fallback (see render_chat).
std::string render_chat_builtin(const std::vector<chat_msg> & msgs, bool add_generation_prompt, bool enable_thinking,
                                const std::string & tools_json = "");

} // namespace si
