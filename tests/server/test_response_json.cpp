// The OpenAI response bodies and SSE frames, without a socket.
//
// These builders were file-local in server.cpp and reachable only by driving a
// live generation, so the shapes clients actually depend on - the reasoning vs
// content split, the tool_calls index, the logprob attachment, the usage
// counters - had no oracle of their own.  They are pure now, which is what makes
// this file possible.
//
// No model and no GPU.  The two logprob builders need a tokenizer and are pinned
// in test_tokenizer.
#include "response.h"

#include "json_util.h"
#include "response_parser.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using si::json;

static int g_fail = 0;

static void check(bool ok, const std::string & what) {
    if (!ok) {
        printf("  FAIL %s\n", what.c_str());
        g_fail++;
    }
}

static void test_usage() {
    printf("usage_json\n");
    const json u = si::usage_json(100, 20, 7, 40);
    check(u["prompt_tokens"] == 100, "prompt_tokens");
    check(u["completion_tokens"] == 20, "completion_tokens");
    check(u["total_tokens"] == 120, "total_tokens is prompt + completion");
    check(u["prompt_tokens_details"]["cached_tokens"] == 40, "cached_tokens in the OpenAI form");
    check(u["prompt_cache_hit_tokens"] == 40, "DeepSeek-style hit count matches");
    check(u["prompt_cache_miss_tokens"] == 60, "DeepSeek-style miss count is prompt - cached");
    check(u["completion_tokens_details"]["reasoning_tokens"] == 7, "reasoning_tokens");
    // the prediction counters exist so a client parsing them does not see a
    // missing field; they are honest zeros while speculative decoding is off
    check(u["completion_tokens_details"]["accepted_prediction_tokens"] == 0, "accepted_prediction_tokens present");
    check(u["completion_tokens_details"]["rejected_prediction_tokens"] == 0, "rejected_prediction_tokens present");
    // a cache hit cannot exceed the prompt
    const json u2 = si::usage_json(10, 1, 0, 99);
    check(u2["prompt_cache_miss_tokens"] == -89, "over-large cached count is passed through, not clamped silently");
}

static void test_message() {
    printf("chat_message_json\n");
    {
        si::choice_out out;
        out.content = "hello";
        out.reasoning = "because";
        const json m = si::chat_message_json(out, true);
        check(m["role"] == "assistant", "role is assistant");
        check(m["content"] == "hello", "content carried through");
        check(m["reasoning_content"] == "because", "reasoning_content when thinking");
        check(!m.contains("tool_calls"), "no tool_calls key when there are none");
    }
    check(!si::chat_message_json(si::choice_out{}, true).contains("reasoning_content"),
          "reasoning_content omitted when thinking is off");
    {
        // with thinking on but nothing reasoned about, the key must stay absent:
        // emitting an empty reasoning_content changes how some clients render
        si::choice_out out;
        out.content = "x";
        check(!si::chat_message_json(out, true).contains("reasoning_content"), "no empty reasoning_content");
    }
    {
        si::choice_out out;
        out.content = "";
        si::response_tool_call t;
        t.id = "c1";
        t.name = "f";
        t.arguments = "{}";
        out.tools.push_back(t);
        const json m = si::chat_message_json(out, false);
        check(m["content"].is_null(), "a tool call with no content sends content:null, not \"\"");
        check(m["tool_calls"].size() == 1, "one tool call");
        check(m["tool_calls"][0]["type"] == "function", "tool call type");
        check(m["tool_calls"][0]["function"]["name"] == "f", "tool call name");
    }
}

static void test_delta() {
    printf("chat_delta_json\n");
    int tool_index = 0;
    long long reasoning_tokens = 0;

    {
        si::response_piece p;
        p.kind = si::response_piece_kind::reasoning;
        p.text = "think";
        const json d = si::chat_delta_json(p, tool_index, reasoning_tokens);
        check(d.contains("reasoning_content") && d["reasoning_content"] == "think", "reasoning delta");
        check(reasoning_tokens == 1, "reasoning delta counts the fragment");
        check(tool_index == 0, "reasoning does not touch the tool index");
    }
    {
        si::response_piece p;
        p.kind = si::response_piece_kind::content;
        p.text = "out";
        const json d = si::chat_delta_json(p, tool_index, reasoning_tokens);
        check(d.contains("content") && d["content"] == "out", "content delta");
        check(!d.contains("reasoning_content"), "content delta carries no reasoning key");
        check(reasoning_tokens == 1, "content does not count as reasoning");
    }
    {
        // two calls must get two indices, or a client's tool_calls array has
        // two entries at index 0 and the arguments overwrite each other
        si::response_piece p;
        p.kind = si::response_piece_kind::tool_call;
        p.call.id = "c0";
        p.call.name = "a";
        p.call.arguments = "{}";
        const json d0 = si::chat_delta_json(p, tool_index, reasoning_tokens);
        p.call.id = "c1";
        p.call.name = "b";
        const json d1 = si::chat_delta_json(p, tool_index, reasoning_tokens);
        check(d0["tool_calls"][0]["index"] == 0 && d1["tool_calls"][0]["index"] == 1, "tool_call index increments");
        check(d1["tool_calls"][0]["id"] == "c1" && d1["tool_calls"][0]["function"]["name"] == "b",
              "second tool call carries its own id and name");
        check(tool_index == 2, "tool index ends at 2");
    }
}

static void test_lp_tracker() {
    printf("lp_tracker\n");
    si::lp_tracker t;
    si::lp_fragment f;
    f.text = "a";
    t.attach(f); // nothing sampled yet: must not attach
    check(f.logprob == 0.f, "attach with no token leaves the fragment alone");

    t.on_token(7, -0.5f, {{7, -0.5f}, {8, -1.25f}});
    si::lp_fragment a;
    a.text = "b";
    t.attach(a);
    check(std::fabs(a.logprob + 0.5f) < 1e-6, "attach folds in the sampled logprob");
    check(a.top.size() == 2, "attach folds in the top alternatives");

    // attaching twice must not re-stamp the same token onto a later fragment
    si::lp_fragment b;
    b.text = "c";
    t.attach(b);
    check(b.logprob == 0.f, "a token is attached to exactly one fragment");
}

static void test_chunks() {
    printf("chat_chunk / text_chunk\n");
    const json cc = si::chat_chunk("id1", "m", 1700000000ULL, 2, {{"content", "x"}}, nullptr);
    check(cc["object"] == "chat.completion.chunk", "chat chunk object name");
    check(cc["id"] == "id1" && cc["model"] == "m", "chat chunk id and model");
    check(cc["created"] == 1700000000ULL, "chat chunk created");
    check(cc["choices"].size() == 1 && cc["choices"][0]["index"] == 2, "chat chunk index");
    check(cc["choices"][0]["delta"]["content"] == "x", "chat chunk delta");
    check(cc["choices"][0]["finish_reason"].is_null(), "null finish_reason stays null mid-stream");
    check(!cc["choices"][0].contains("logprobs"), "logprobs key omitted when not requested");

    const json with_lp = si::chat_chunk("i", "m", 1, 0, json::object(), "stop", json{{"content", json::array()}});
    check(with_lp["choices"][0].contains("logprobs"), "logprobs key present when supplied");
    check(with_lp["choices"][0]["finish_reason"] == "stop", "finish_reason carried");

    const json tc = si::text_chunk("id2", "m", 5, 1, "txt", nullptr);
    check(tc["object"] == "text_completion", "completion chunk object name");
    check(tc["choices"][0]["text"] == "txt", "completion chunk text");
    check(!tc["choices"][0].contains("delta"), "completion chunk has no delta key");
}

static void test_misc() {
    printf("bytes_json / gen_id\n");
    const json b = si::bytes_json("A\xc3\xa9");
    check(b.size() == 3, "one array entry per byte");
    check(b[0] == 65 && b[1] == 0xc3 && b[2] == 0xa9, "raw byte values, not characters");

    const std::string i1 = si::gen_id();
    const std::string i2 = si::gen_id();
    check(i1 != i2, "ids are unique");
    check(i1.rfind("chatcmpl-", 0) == 0, "id prefix");
    // strictly increasing, so a client can order two responses by id
    const long a = std::atol(i1.c_str() + 9);
    const long b2 = std::atol(i2.c_str() + 9);
    check(b2 == a + 1, "ids increase by one");
}

int main() {
    test_usage();
    test_message();
    test_delta();
    test_lp_tracker();
    test_chunks();
    test_misc();

    if (g_fail) {
        printf("\n%d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("\nall response-building checks OK\n");
    return 0;
}