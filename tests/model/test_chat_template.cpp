// Renders the model's real GGUF chat template through minja and compares
// against expected strings produced by the reference Jinja2 engine (regenerate
// with python3 -m jinja2 if the model changes).  CPU only, no GPU work.
#include <cstdio>
#include <string>
#include <vector>

#include "chat.h"
#include "chat_template.h"
#include "gguf.h"

using namespace si;

static int g_fail = 0;

static void check(const char * name, const std::string & got, const std::string & want) {
    if (got == want) {
        printf("  [%s] OK\n", name);
        return;
    }
    g_fail++;
    printf("  [%s] MISMATCH\n    got:  %s\n    want: %s\n", name, got.c_str(), want.c_str());
}

int main(int argc, char ** argv) {
    const char * model = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    gguf_file f;
    f.load(model);
    const std::string * tmpl = f.get_str("tokenizer.chat_template");
    if (!tmpl) {
        printf("no tokenizer.chat_template in %s\n", model);
        return 1;
    }
    if (tmpl->empty()) {
        printf("empty tokenizer.chat_template\n");
        return 1;
    }

    struct case_t {
        const char * name;
        std::vector<chat_msg> msgs;
        bool gen;
        bool think;
        std::string want;
    };
    std::vector<case_t> cases = {
        {"system_user",
         {{"system", "You are helpful."}, {"user", "Hi"}},
         true,
         false,
         "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
         "<|im_start|>assistant\n<think>\n\n</think>\n\n"},
        {"user_thinking",
         {{"user", "Hi"}},
         true,
         true,
         "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"},
        {"multi_turn",
         {{"system", "sys "}, {"user", "q1"}, {"assistant", "a1"}, {"user", "q2"}},
         true,
         false,
         "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nq1<|im_end|>\n"
         "<|im_start|>assistant\na1<|im_end|>\n<|im_start|>user\nq2<|im_end|>\n"
         "<|im_start|>assistant\n<think>\n\n</think>\n\n"},
        {"assistant_reasoning",
         {{"user", "q"}, {"assistant", "<think>\nmy reasoning\n</think>\n\nthe answer"}},
         false,
         false,
         "<|im_start|>user\nq<|im_end|>\n"
         "<|im_start|>assistant\n<think>\nmy reasoning\n</think>\n\nthe answer<|im_end|>\n"},
        {"tool_grouping",
         {{"user", "call it"}, {"tool", "{\"x\":1}"}, {"tool", "{\"y\":2}"}},
         true,
         false,
         "<|im_start|>user\ncall it<|im_end|>\n<|im_start|>user\n<tool_response>\n{\"x\":1}\n"
         "</tool_response>\n<tool_response>\n{\"y\":2}\n</tool_response><|im_end|>\n"
         "<|im_start|>assistant\n<think>\n\n</think>\n\n"},
        {"content_trimmed",
         {{"user", "  padded  "}},
         true,
         false,
         "<|im_start|>user\npadded<|im_end|>\n"
         "<|im_start|>assistant\n<think>\n\n</think>\n\n"},
    };

    for (const case_t & c : cases) {
        std::string out;
        const bool ok = render_chat_template(*tmpl, c.msgs, c.gen, c.think, out);
        if (!ok) {
            g_fail++;
            printf("  [%s] EVAL FAILED\n", c.name);
            continue;
        }
        check(c.name, out, c.want);
    }

    // render_chat() must fall back to the built-in renderer on an empty template
    {
        std::string out = render_chat("", {{"user", "Hi"}}, true, false);
        check("builtin_fallback", out,
              "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }

    // tools + assistant tool_calls + tool response through the real template
    {
        const std::string tools =
            "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
            "\"required\":[\"city\"]}}}]";
        chat_msg user("user", "weather in Paris?");
        chat_msg asst("assistant", "");
        asst.tool_calls.push_back({"call_1", "get_weather", "{\"city\":\"Paris\"}"});
        chat_msg tool("tool", "sunny, 20C");
        tool.tool_call_id = "call_1";
        chat_msg follow("user", "thanks");
        std::string out;
        const bool ok = render_chat_template(*tmpl, {user, asst, tool, follow}, true, false, out, tools);
        if (!ok) {
            g_fail++;
            printf("  [tools] EVAL FAILED\n");
        } else {
            const bool good = out.find("get_weather") != std::string::npos
                              && out.find("<function=get_weather>") != std::string::npos
                              && out.find("<parameter=city>") != std::string::npos
                              && out.find("Paris") != std::string::npos
                              && out.find("<tool_response>") != std::string::npos
                              && out.find("sunny, 20C") != std::string::npos;
            if (!good) {
                g_fail++;
                printf("  [tools] MISSING MARKUP\n%s\n", out.c_str());
            } else {
                printf("  [tools] OK\n");
            }
        }
    }

    // assistant reasoning_content is re-rendered as <think> for the latest turn
    {
        chat_msg user("user", "q");
        chat_msg asst("assistant", "the answer");
        asst.reasoning_content = "my thought";
        std::string out;
        const bool ok = render_chat_template(*tmpl, {user, asst}, false, false, out);
        if (!ok) {
            g_fail++;
            printf("  [reasoning_content] EVAL FAILED\n");
        } else if (out.find("<think>\nmy thought\n</think>\n\nthe answer") == std::string::npos) {
            g_fail++;
            printf("  [reasoning_content] MISSING\n%s\n", out.c_str());
        } else {
            printf("  [reasoning_content] OK\n");
        }
    }

    printf(g_fail ? "chat template test FAILED (%d)\n" : "chat template test OK\n", g_fail);
    return g_fail ? 1 : 0;
}
