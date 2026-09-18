// Unit tests for the OpenAI response splitter (reasoning_content + tool calls).
// CPU only, no model/GPU needed.
#include <cstdio>
#include <string>
#include <vector>

#include "response_parser.h"

using namespace si;

static int g_fail = 0;

struct collected {
    std::string reasoning, content;
    std::vector<response_tool_call> tools;
    // concatenation of the emitted fragments, to check they match the fields
    std::string reasoning_stream, content_stream;
};

static collected run(const std::string & text, bool thinking, bool tools, size_t chunk = 0) {
    collected c;
    response_parser p(thinking, tools, [&](const response_piece & piece) {
        switch (piece.kind) {
        case response_piece_kind::reasoning:
            c.reasoning_stream += piece.text;
            break;
        case response_piece_kind::content:
            c.content_stream += piece.text;
            break;
        case response_piece_kind::tool_call:
            c.tools.push_back(piece.call);
            break;
        }
    });
    if (chunk == 0) {
        p.feed(text);
    } else {
        for (size_t i = 0; i < text.size(); i += chunk) {
            p.feed(text.substr(i, chunk));
        }
    }
    p.finish();
    c.reasoning = p.reasoning();
    c.content = p.content();
    c.tools = p.tool_calls();
    return c;
}

static void check(const char * name, const std::string & got, const std::string & want) {
    if (got == want) {
        printf("  [%s] OK\n", name);
        return;
    }
    g_fail++;
    printf("  [%s] MISMATCH\n    got:  '%s'\n    want: '%s'\n", name, got.c_str(), want.c_str());
}

static void check_bool(const char * name, bool got) {
    if (got) {
        printf("  [%s] OK\n", name);
        return;
    }
    g_fail++;
    printf("  [%s] FAILED\n", name);
}

int main() {
    printf("reasoning:\n");
    {
        const collected c = run("I should check</think>Hello world", true, true, 1);
        check("reasoning text", c.reasoning, "I should check");
        check("content after think", c.content, "Hello world");
        check("reasoning stream", c.reasoning_stream, c.reasoning);
        check("content stream", c.content_stream, c.content);
    }
    {
        const collected c = run("<think>\nlet me see\n</think>\n\nthe answer", true, true, 0);
        check("open tag stripped", c.reasoning, "let me see");
        check("content leading newlines stripped", c.content, "the answer");
    }
    {
        const collected c = run("only reasoning, never closed", true, true, 3);
        check("unclosed reasoning", c.reasoning, "only reasoning, never closed");
        check("unclosed reasoning content empty", c.content, "");
    }
    {
        const collected c = run("<think>x</think>y", false, true, 4);
        check("thinking disabled keeps markers", c.content, "<think>x</think>y");
    }

    printf("tool calls:\n");
    const std::string tool =
        "\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\n";
    {
        const collected c = run("Let me look.\n</think>" + tool, true, true, 0);
        check("tool reasoning", c.reasoning, "Let me look.");
        check("tool content empty", c.content, "");
        check_bool("one tool", c.tools.size() == 1);
        if (c.tools.size() == 1) {
            check("tool name", c.tools[0].name, "get_weather");
            check("tool args", c.tools[0].arguments, "{\"city\":\"Paris\",\"days\":3}");
            check_bool("tool id", !c.tools[0].id.empty());
        }
    }
    {
        // char-by-char streaming must agree with the whole-text parse
        const collected c = run("Let me look.\n</think>" + tool, true, true, 1);
        check("stream tool reasoning", c.reasoning, "Let me look.");
        check("stream tool content", c.content, "");
        check_bool("stream one tool", c.tools.size() == 1);
        if (c.tools.size() == 1) {
            check("stream tool args", c.tools[0].arguments, "{\"city\":\"Paris\",\"days\":3}");
        }
    }
    {
        const collected c = run("Here you go.<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n"
                                "</tool_call>done",
                                false, true, 0);
        check("prose before tool", c.content, "Here you go.done");
        check_bool("prose tool parsed", c.tools.size() == 1 && c.tools[0].name == "f");
    }
    {
        const std::string two = "<tool_call>\n<function=a>\n</function>\n</tool_call>"
                                "<tool_call>\n<function=b>\n<parameter=x>\nhello world\n</parameter>\n</function>\n</tool_call>";
        const collected c = run(two, false, true, 2);
        check_bool("two tools", c.tools.size() == 2);
        if (c.tools.size() == 2) {
            check("tool a args empty", c.tools[0].arguments, "{}");
            check("tool b name", c.tools[1].name, "b");
            check("tool b string arg", c.tools[1].arguments, "{\"x\":\"hello world\"}");
        }
        check("two tools content empty", c.content, "");
    }
    {
        const collected c = run(tool, false, false, 0);
        check_bool("tools off: no tool parsed", c.tools.empty());
        check("tools off: markup is content", c.content, tool);
    }
    {
        const std::string svc = "<tool_call>\n<function=cfg>\n<parameter=opts>\n{\"a\":true,\"b\":[1,2]}\n"
                                "</parameter>\n</function>\n</tool_call>";
        const collected c = run(svc, false, true, 5);
        check_bool("object arg", c.tools.size() == 1);
        if (c.tools.size() == 1) {
            check("object arg json", c.tools[0].arguments, "{\"opts\":{\"a\":true,\"b\":[1,2]}}");
        }
    }
    {
        const std::string j = "<tool_call>{\"name\":\"jfun\",\"arguments\":{\"q\":\"x\"}}</tool_call>";
        const collected c = run(j, false, true, 0);
        check_bool("json fallback", c.tools.size() == 1);
        if (c.tools.size() == 1) {
            check("json fallback name", c.tools[0].name, "jfun");
            check("json fallback args", c.tools[0].arguments, "{\"q\":\"x\"}");
        }
    }

    printf(g_fail ? "response parser test FAILED (%d)\n" : "response parser test OK\n", g_fail);
    return g_fail ? 1 : 0;
}
