#pragma once
#include <string>
#include <utility>
#include <vector>

namespace si {

struct chat_part {
    bool is_image = false;
    std::string text;
};

struct chat_msg {
    std::string role;
    std::string content;
    // optional structured content: when non-empty it takes precedence over
    // `content` and lets a message interleave text and image parts
    std::vector<chat_part> parts;

    chat_msg() = default;
    chat_msg(std::string r, std::string c, std::vector<chat_part> p = {})
        : role(std::move(r)), content(std::move(c)), parts(std::move(p)) {}
};

// Render the conversation with the model's Jinja chat template `tmpl` (the
// GGUF `tokenizer.chat_template` string).  Falls back to the built-in Qwen3.5
// ChatML renderer when the template is empty or uses unsupported syntax.
std::string render_chat(const std::string & tmpl, const std::vector<chat_msg> & msgs,
                        bool add_generation_prompt, bool enable_thinking);

// Built-in ChatML fallback (see render_chat).
std::string render_chat_builtin(const std::vector<chat_msg> & msgs, bool add_generation_prompt,
                                bool enable_thinking);

} // namespace si
