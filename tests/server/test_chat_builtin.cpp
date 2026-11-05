// The built-in ChatML renderer (the fallback used when a GGUF has no chat
// template, or when the vendored Jinja engine cannot parse it).  Hermetic: no
// model, no GGUF, no HTTP.
//
// A fallback that renders subtly the wrong prompt is worse than no fallback:
// the model still answers, just not to the prompt it was trained on.  These
// cases pin the roles, the tool block, the thinking marker and the multimodal
// placeholders that the server's prompt expansion depends on.
#include <cstdio>
#include <string>

#include "chat.h"

using namespace si;

static int g_fail = 0;

static void check(bool ok, const char * what) {
    if (ok) {
        printf("  [%s] OK\n", what);
        return;
    }
    g_fail++;
    printf("  [%s] FAILED\n", what);
}

static void check_eq(const char * what, const std::string & got, const std::string & want) {
    if (got == want) {
        printf("  [%s] OK\n", what);
        return;
    }
    g_fail++;
    printf("  [%s] FAILED\n    got:  %s\n    want: %s\n", what, got.c_str(), want.c_str());
}

static void check_has(const char * what, const std::string & hay, const std::string & needle) {
    const bool ok = hay.find(needle) != std::string::npos;
    if (!ok) {
        printf("  [%s] FAILED (missing '%s')\n", what, needle.c_str());
        g_fail++;
        return;
    }
    printf("  [%s] OK\n", what);
}

static bool starts_with(const std::string & s, const std::string & p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

int main() {
    printf("empty conversation\n");
    check_eq("empty", render_chat_builtin({}, true, false), "");

    printf("basic roles\n");
    {
        std::vector<chat_msg> m = {chat_msg("system", "sys"), chat_msg("user", "hi")};
        const std::string s = render_chat_builtin(m, true, false);
        check_eq("system+user+generation",
                 s, "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
                    "<think>\n\n</think>\n\n");
        check_has("thinking marker when disabled", s, "<think>\n\n</think>\n\n");
    }
    {
        std::vector<chat_msg> m = {chat_msg("user", " hi ")};
        const std::string s = render_chat_builtin(m, false, true);
        check_eq("no generation prompt", s, "<|im_start|>user\n hi <|im_end|>\n");
    }
    {
        std::vector<chat_msg> m = {chat_msg("system", "sys"), chat_msg("user", "q"),
                                   chat_msg("assistant", "a"), chat_msg("user", "q2")};
        const std::string s = render_chat_builtin(m, true, true);
        check_has("assistant turn", s, "<|im_start|>assistant\na<|im_end|>\n");
        check_has("generation prompt with thinking on", s, "<|im_start|>assistant\n<think>\n");
    }

    printf("reasoning split\n");
    {
        std::vector<chat_msg> m = {chat_msg("user", "q"),
                                   chat_msg("assistant", "<think>why</think>answer")};
        const std::string s = render_chat_builtin(m, false, false);
        check_has("inline think is reopened", s, "<|im_start|>assistant\n<think>\nwhy\n</think>\n\nanswer<|im_end|>\n");
    }
    {
        std::vector<chat_msg> m = {chat_msg("user", "q")};
        m.push_back(chat_msg("assistant", ""));
        m.back().reasoning_content = "r";
        const std::string s = render_chat_builtin(m, false, false);
        check_has("structured reasoning", s, "<|im_start|>assistant\n<think>\nr\n</think>\n\n<|im_end|>\n");
    }

    printf("tool roles and calls\n");
    {
        std::vector<chat_msg> m = {chat_msg("user", "q")};
        m.push_back(chat_msg("tool", "42"));
        m.back().tool_call_id = "call_1";
        const std::string s = render_chat_builtin(m, false, false);
        check_has("tool result is a user tool_response",
                  s, "<|im_start|>user\n<tool_response>\n42\n</tool_response><|im_end|>\n");
    }
    {
        std::vector<chat_msg> m = {chat_msg("user", "q")};
        m.push_back(chat_msg("assistant", ""));
        chat_tool_call tc;
        tc.id = "call_1";
        tc.name = "lookup";
        tc.arguments = R"({"city":"Paris","n":2})";
        m.back().tool_calls.push_back(tc);
        const std::string s = render_chat_builtin(m, false, false, R"([{"type":"function"}])");
        check_has("tool call xml", s, "<tool_call>\n<function=lookup>\n<parameter=city>\nParis\n</parameter>\n");
        check_has("numeric argument keeps json spelling", s, "<parameter=n>\n2\n</parameter>");
        check_has("tools system block", s, "# Tools");
    }

    printf("multimodal placeholders\n");
    {
        std::vector<chat_msg> m;
        chat_msg u("user", "");
        chat_part img;
        img.kind = chat_part_kind::IMAGE;
        chat_part vid;
        vid.kind = chat_part_kind::VIDEO;
        chat_part aud;
        aud.kind = chat_part_kind::AUDIO;
        chat_part txt;
        txt.kind = chat_part_kind::TEXT;
        txt.text = "describe ";
        u.parts = {txt, img, vid, aud};
        m.push_back(u);
        const std::string s = render_chat_builtin(m, false, false);
        check_has("text part", s, "describe ");
        check_has("image placeholder", s, "<|vision_start|><|image_pad|><|vision_end|>");
        check_has("video placeholder", s, "<|vision_start|><|video_pad|><|vision_end|>");
        check_has("audio placeholder", s, "<|audio_start|><|audio_pad|><|audio_end|>");
    }

    printf("unknown role is ignored, not emitted\n");
    {
        std::vector<chat_msg> m = {chat_msg("user", "q"), chat_msg("observer", "x")};
        const std::string s = render_chat_builtin(m, false, false);
        check_has("known user turn", s, "<|im_start|>user\nq<|im_end|>\n");
        check(s.find("observer") == std::string::npos, "unknown role omitted");
    }
    {
        // These are guards against a future refactor, not behaviour end users see:
        // the renderer must always open with ChatML, never with an assistant turn.
        std::vector<chat_msg> m = {chat_msg("user", "q")};
        check(starts_with(render_chat_builtin(m, true, false), "<|im_start|>user\n"), "starts with the first role");
    }

    printf(g_fail ? "chat builtin test FAILED (%d)\n" : "chat builtin test OK\n", g_fail);
    return g_fail ? 1 : 0;
}
