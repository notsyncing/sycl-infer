#include "chat.h"

#include "chat_template.h"

namespace si {

static std::string rstrip_nl(const std::string & s) {
    size_t e = s.size();
    while (e > 0 && s[e - 1] == '\n') e--;
    return s.substr(0, e);
}
static std::string lstrip_nl(const std::string & s) {
    size_t b = 0;
    while (b < s.size() && s[b] == '\n') b++;
    return s.substr(b);
}
static std::string trim(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\n' || s[b] == '\t' || s[b] == '\r')) b++;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\n' || s[e - 1] == '\t' || s[e - 1] == '\r')) e--;
    return s.substr(b, e - b);
}

std::string render_chat(const std::string & tmpl, const std::vector<chat_msg> & msgs,
                        bool add_generation_prompt, bool enable_thinking) {
    std::string out;
    if (render_chat_template(tmpl, msgs, add_generation_prompt, enable_thinking, out)) return out;
    return render_chat_builtin(msgs, add_generation_prompt, enable_thinking);
}

std::string render_chat_builtin(const std::vector<chat_msg> & msgs, bool add_generation_prompt,
                                bool enable_thinking) {
    std::string out;
    if (msgs.empty()) return out;

    // structured content: interleave text with the vision placeholder
    auto content_of = [](const chat_msg & m) {
        if (m.parts.empty()) return m.content;
        std::string s;
        for (const chat_part & p : m.parts) {
            if (p.is_image) s += "<|vision_start|><|image_pad|><|vision_end|>";
            else s += p.text;
        }
        return s;
    };

    // find last query index (last user message)
    int last_query = -1;
    for (int i = (int) msgs.size() - 1; i >= 0; i--) {
        if (msgs[i].role == "user") { last_query = i; break; }
    }

    size_t start = 0;
    if (msgs[0].role == "system") {
        const std::string c = trim(content_of(msgs[0]));
        out += "<|im_start|>system\n" + c + "<|im_end|>\n";
        start = 1;
    }

    for (size_t i = start; i < msgs.size(); i++) {
        const chat_msg & msg = msgs[i];
        const std::string mcontent = content_of(msg);
        if (msg.role == "user") {
            out += "<|im_start|>user\n" + mcontent + "<|im_end|>\n";
        } else if (msg.role == "assistant") {
            std::string content = mcontent;
            std::string reasoning;
            bool has_reasoning = false;
            const size_t close_pos = content.rfind("</think>");
            if (close_pos != std::string::npos) {
                std::string before = content.substr(0, close_pos);
                const size_t open_pos = before.rfind("<think>");
                if (open_pos != std::string::npos) {
                    reasoning = lstrip_nl(rstrip_nl(before.substr(open_pos + 7)));
                    content = lstrip_nl(content.substr(close_pos + 8));
                    has_reasoning = true;
                }
            }
            out += "<|im_start|>assistant\n";
            if (has_reasoning && (int) i > last_query) {
                out += "<think>\n" + trim(reasoning) + "\n</think>\n\n" + content;
            } else {
                out += content;
            }
            out += "<|im_end|>\n";
        } else if (msg.role == "tool") {
            out += "<|im_start|>user\n<tool_response>\n" + msg.content + "\n</tool_response><|im_end|>\n";
        }
    }

    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (enable_thinking) out += "<think>\n";
        else out += "<think>\n\n</think>\n\n";
    }
    return out;
}

} // namespace si
