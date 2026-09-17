#pragma once
#include <string>
#include <vector>

namespace si {

struct chat_msg {
    std::string role;
    std::string content;
};

// Render the Qwen3.5 ChatML chat template (text-only path).
std::string render_chat(const std::vector<chat_msg> & msgs, bool add_generation_prompt,
                        bool enable_thinking);

} // namespace si
