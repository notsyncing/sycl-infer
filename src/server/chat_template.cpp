// GGUF chat template rendering, delegated to the vendored minja library
// (third_party/minja, MIT): a full Jinja engine, the same one llama.cpp uses.
// We feed it the tokenizer.chat_template source, the conversation as JSON, the
// offered tools and the generation flags; on any parse/eval error the caller
// falls back to the built-in renderer (see chat.cpp).
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
                          bool enable_thinking, std::string & out, const std::string & tools_json) {
    if (tmpl.empty()) {
        return false;
    }
    static const bool dbg = getenv("PF_CHAT_TMPL_DEBUG") != nullptr;
    try {
        minja::chat_template ct(tmpl, /*bos_token=*/"", /*eos_token=*/"");

        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        for (const chat_msg & m : msgs) {
            nlohmann::ordered_json jm = nlohmann::ordered_json::object();
            jm["role"] = m.role;
            if (m.parts.empty()) {
                jm["content"] = m.content;
            } else {
                auto content = nlohmann::ordered_json::array();
                for (const chat_part & p : m.parts) {
                    switch (p.kind) {
                    case chat_part_kind::IMAGE: content.push_back({{"type", "image"}}); break;
                    case chat_part_kind::VIDEO: content.push_back({{"type", "video"}}); break;
                    case chat_part_kind::AUDIO: content.push_back({{"type", "input_audio"}}); break;
                    default: content.push_back({{"type", "text"}, {"text", p.text}}); break;
                    }
                }
                jm["content"] = std::move(content);
            }
            if (!m.reasoning_content.empty()) {
                jm["reasoning_content"] = m.reasoning_content;
            }
            if (!m.tool_calls.empty()) {
                auto calls = nlohmann::ordered_json::array();
                for (const chat_tool_call & tc : m.tool_calls) {
                    nlohmann::ordered_json args = nlohmann::ordered_json::object();
                    if (!tc.arguments.empty()) {
                        try {
                            args = nlohmann::ordered_json::parse(tc.arguments);
                        } catch (...) {
                            args = tc.arguments; // template tolerates a raw string
                        }
                    }
                    nlohmann::ordered_json call = {
                        {"id", tc.id},
                        {"type", "function"},
                        {"function", {{"name", tc.name}, {"arguments", std::move(args)}}},
                    };
                    calls.push_back(std::move(call));
                }
                jm["tool_calls"] = std::move(calls);
            }
            if (!m.tool_call_id.empty()) {
                jm["tool_call_id"] = m.tool_call_id;
            }
            if (!m.name.empty()) {
                jm["name"] = m.name;
            }
            messages.push_back(std::move(jm));
        }

        minja::chat_template_inputs inputs;
        inputs.messages = std::move(messages);
        inputs.tools = nlohmann::ordered_json::array();
        if (!tools_json.empty()) {
            try {
                nlohmann::ordered_json parsed = nlohmann::ordered_json::parse(tools_json);
                if (parsed.is_array()) {
                    inputs.tools = std::move(parsed);
                }
            } catch (const std::exception & ex) {
                if (dbg) {
                    fprintf(stderr, "[chat_template] bad tools json: %s\n", ex.what());
                }
            }
        }
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
