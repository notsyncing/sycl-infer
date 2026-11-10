// DeepSeek-V4.1's structural chat renderer (the GGUF template is beyond minja).
// Hermetic: no model, no GGUF, no HTTP.
//
// The load-bearing case is the chat-mode transition: the template appends
// `<|Assistant|>` + `</think>` (an explicitly closed think block), NOT a bare
// `<|Assistant|>`.  The bare form made image prompts answer EOS immediately;
// thinking mode appends `<|Assistant|><think>` instead.
#include <cstdio>
#include <string>
#include <vector>

#include "chat.h"

using namespace si;

static int g_fail = 0;

static void check_eq(const char * what, const std::string & got, const std::string & want) {
    if (got == want) {
        printf("  [%s] OK\n", what);
        return;
    }
    g_fail++;
    printf("  [%s] FAILED\n    got:  %s\n    want: %s\n", what, got.c_str(), want.c_str());
}

static void check_has(const char * what, const std::string & hay, const std::string & needle) {
    if (hay.find(needle) != std::string::npos) {
        printf("  [%s] OK\n", what);
        return;
    }
    g_fail++;
    printf("  [%s] FAILED (missing '%s' in '%s')\n", what, needle.c_str(), hay.c_str());
}

static chat_msg txt(const char * role, const char * content) {
    return chat_msg(role, content);
}

int main() {
    const std::string effort =
        "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n";

    {
        std::vector<chat_msg> m = {txt("user", "The capital of France is")};
        const std::string got = render_chat_deepseek(m, true, false);
        check_eq("plain chat closes the think block",
                 got, "<｜User｜>The capital of France is<｜Assistant｜></think>");
        check_has("chat ends with assistant+close", got, "<｜Assistant｜></think>");
    }
    {
        std::vector<chat_msg> m = {txt("user", "The capital of France is")};
        const std::string got = render_chat_deepseek(m, true, true);
        check_eq("thinking adds system effort + open think",
                 got, "<｜System｜>" + effort
                          + "<｜User｜>The capital of France is<｜Assistant｜><think>");
    }
    {
        std::vector<chat_msg> m = {txt("system", "You are helpful."), txt("user", "Hi")};
        const std::string got = render_chat_deepseek(m, true, false);
        check_eq("system+user chat",
                 got, "<｜System｜>You are helpful.<｜User｜>Hi<｜Assistant｜></think>");
    }
    {
        chat_msg u("user", "");
        u.parts.push_back({chat_part_kind::IMAGE, "", "", "", ""});
        chat_part p;
        p.kind = chat_part_kind::TEXT;
        p.text = "What does it say?";
        u.parts.push_back(p);
        std::vector<chat_msg> m = {u};
        const std::string got = render_chat_deepseek(m, true, false);
        check_eq("image part + chat",
                 got, "<｜User｜><｜deepseek_image｜>\n\nWhat does it say?<｜Assistant｜></think>");
    }
    {
        std::vector<chat_msg> m = {txt("user", "u1"), txt("assistant", "a1"), txt("user", "u2")};
        const std::string got = render_chat_deepseek(m, true, false);
        check_eq("assistant history ends with EOS",
                 got, "<｜User｜>u1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>"
                      "<｜User｜>u2<｜Assistant｜></think>");
    }
    {
        std::vector<chat_msg> m = {txt("user", "u"), txt("tool", "r")};
        const std::string got = render_chat_deepseek(m, true, false);
        check_eq("tool result folds into a user turn",
                 got, "<｜User｜>u<｜User｜><tool_result>r</tool_result>"
                      "<｜Assistant｜></think>");
    }
    {
        std::vector<chat_msg> m = {txt("user", "u1"), txt("assistant", "a1"), txt("user", "u2")};
        const std::string got = render_chat_deepseek(m, true, true);
        check_eq("thinking history: only the last user opens think",
                 got, "<｜System｜>" + effort
                          + "<｜User｜>u1<｜Assistant｜></think>a1"
                            "<｜end▁of▁sentence｜>"
                            "<｜User｜>u2<｜Assistant｜><think>");
    }

    if (g_fail) {
        printf("chat deepseek test: %d FAILED\n", g_fail);
        return 1;
    }
    printf("chat deepseek test OK\n");
    return 0;
}
