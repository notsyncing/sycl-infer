// GGUF chat template rendering, delegated to the vendored minja library
// (third_party/minja, MIT): a full Jinja engine, the same one llama.cpp uses.
// We feed it the tokenizer.chat_template source, the conversation as JSON and
// the generation flags; on any parse/eval error the caller falls back to the
// built-in renderer (see chat.cpp).
#include "chat_template.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "chat.h"
#include "json.hpp"
#include "minja/chat-template.hpp"

namespace si {

bool render_chat_template(const std::string & tmpl, const std::vector<chat_msg> & msgs, bool add_generation_prompt,
                          bool enable_thinking, std::string & out) {
    if (tmpl.empty()) {
        return false;
    }
    static const bool dbg = getenv("PF_CHAT_TMPL_DEBUG") != nullptr;
    try {
        minja::chat_template ct(tmpl, /*bos_token=*/"", /*eos_token=*/"");

        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        for (const chat_msg & m : msgs) {
            if (m.parts.empty()) {
                messages.push_back({{"role", m.role}, {"content", m.content}});
                continue;
            }
            auto content = nlohmann::ordered_json::array();
            for (const chat_part & p : m.parts) {
                if (p.is_image) {
                    content.push_back({{"type", "image"}});
                } else {
                    content.push_back({{"type", "text"}, {"text", p.text}});
                }
            }
            messages.push_back({{"role", m.role}, {"content", content}});
        }

        minja::chat_template_inputs inputs;
        inputs.messages = std::move(messages);
        inputs.tools = nlohmann::ordered_json::array();
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.extra_context = nlohmann::ordered_json::object();
        inputs.extra_context["enable_thinking"] = enable_thinking;

        out = ct.apply(inputs);
        return true;
    } catch (const std::exception & ex) {
        if (dbg) {
            fprintf(stderr, "[chat_template] %s\n", ex.what());
        }
        return false;
    }
}

} // namespace si
