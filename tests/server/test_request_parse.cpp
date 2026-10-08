// The request surface, without a socket.
//
// These parsers used to be file-local inside server.cpp, reachable only through
// a live HTTP request, so every alias and precedence rule in them was untested.
// They are now pure functions (JSON in, plain values out), which is what makes
// this file possible - and the rules below are the ones a client actually
// depends on, so they are worth pinning rather than re-deriving.
//
// No model and no GPU.  parse_completion_prompts is the one exception (it needs
// a real tokenizer) and is covered in test_tokenizer.
#include "chat.h"
#include "common/env.h"
#include "json_util.h"
#include "media_fetch.h"
#include "request.h"
#include "sampler.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using si::json;

static int g_fail = 0;

static void check(bool ok, const std::string & what) {
    if (!ok) {
        printf("  FAIL %s\n", what.c_str());
        g_fail++;
    }
}

static json J(const char * s) {
    return json::parse(s);
}

static void test_params() {
    printf("parse_params\n");
    const si::gen_params d = si::parse_params(J("{}"));
    check(d.max_tokens == 256, "default max_tokens 256");
    check(d.temperature == 1.0f, "default temperature 1.0");
    check(d.top_p == 0.95f, "default top_p 0.95");
    check(d.top_k == 40, "default top_k 40");
    check(d.min_p == 0.0f, "default min_p 0");
    check(d.repeat_penalty == 1.0f, "default repetition penalty 1.0");
    check(d.repeat_last_n == 64, "default repeat_last_n 64");
    check(d.ignore_eos == false, "default ignore_eos off");
    check(!d.logprobs, "default logprobs off");

    // both max_tokens spellings, and the non-positive fallback
    check(si::parse_params(J(R"({"max_tokens": 7})")).max_tokens == 7, "max_tokens honoured");
    check(si::parse_params(J(R"({"max_completion_tokens": 9})")).max_tokens == 9, "max_completion_tokens alias");
    check(si::parse_params(J(R"({"max_tokens": 0})")).max_tokens == 256, "max_tokens 0 falls back to 256");
    check(si::parse_params(J(R"({"max_tokens": -5})")).max_tokens == 256, "negative max_tokens falls back to 256");

    // both repetition-penalty spellings (OpenAI's, then llama.cpp's)
    check(si::parse_params(J(R"({"repetition_penalty": 1.1})")).repeat_penalty == 1.1f, "repetition_penalty");
    check(si::parse_params(J(R"({"repeat_penalty": 1.2})")).repeat_penalty == 1.2f, "repeat_penalty alias");

    // logit_bias: clamped, and junk keys/ids dropped rather than fatal
    {
        const si::gen_params g = si::parse_params(J(R"({"logit_bias": {"5": 2.0, "7": -900.0, "x": 1.0, "-1": 3.0}})"));
        check(g.logit_bias.size() == 2, "logit_bias drops non-integer and negative keys");
        check(g.logit_bias.count(5) && std::abs(g.logit_bias.at(5) - 2.0f) < 1e-6, "logit_bias keeps a valid entry");
        check(g.logit_bias.count(7) && std::abs(g.logit_bias.at(7) + 100.0f) < 1e-6, "logit_bias clamps to -100");
    }
    // logprobs: chat sends a bool, completions a count
    check(si::parse_params(J(R"({"logprobs": true})")).logprobs, "logprobs true");
    {
        const si::gen_params g = si::parse_params(J(R"({"logprobs": 3})"));
        check(g.logprobs && g.top_logprobs == 3, "logprobs 3 -> logprobs on, top_logprobs 3");
        check(si::parse_params(J(R"({"logprobs": 0})")).top_logprobs == 0, "logprobs 0 -> top_logprobs 0");
        check(si::parse_params(J(R"({"logprobs": 99})")).top_logprobs == 20, "top_logprobs clamps to 20");
        check(si::parse_params(J(R"({"top_logprobs": -1})")).top_logprobs == 0, "top_logprobs clamps at 0");
    }
    check(si::parse_params(J(R"({"seed": 42})")).seed == 42u, "seed honoured");
    check(si::parse_params(J(R"({"ignore_eos": true})")).ignore_eos, "ignore_eos honoured");
}

static void test_counts() {
    printf("parse_n / parse_best_of / parse_include_usage\n");
    check(si::parse_n(J("{}")) == 1, "n defaults to 1");
    check(si::parse_n(J(R"({"n": 4})")) == 4, "n honoured");
    check(si::parse_n(J(R"({"n": 0})")) == 1, "n 0 -> 1");
    check(si::parse_n(J(R"({"n": -3})")) == 1, "negative n -> 1");
    check(si::parse_n(J(R"({"n": 999})")) == si::kMaxN, "n clamps to kMaxN");
    check(si::parse_n(J(R"({"n": "two"})")) == 1, "non-numeric n -> 1");

    check(si::parse_best_of(J("{}"), 3) == 3, "best_of defaults to n");
    check(si::parse_best_of(J(R"({"best_of": 8})"), 3) == 8, "best_of honoured");
    check(si::parse_best_of(J(R"({"best_of": 0})"), 3) == 1, "best_of 0 -> 1");
    check(si::parse_best_of(J(R"({"best_of": 999})"), 3) == si::kMaxN, "best_of clamps to kMaxN");

    check(!si::parse_include_usage(J("{}")), "include_usage defaults off");
    check(si::parse_include_usage(J(R"({"stream_options": {"include_usage": true}})")), "include_usage honoured");
    check(!si::parse_include_usage(J(R"({"stream_options": {}})")), "empty stream_options -> off");
}

static void test_stop_and_thinking() {
    printf("parse_stop / parse_thinking\n");
    check(si::parse_stop(J("{}")).empty(), "no stop -> none");
    check(si::parse_stop(J(R"({"stop": "END"})")).size() == 1, "stop string");
    check(si::parse_stop(J(R"({"stop": ["a", "b", 3, "c"]})")).size() == 3, "stop array skips non-strings");
    {
        // the cap is a real ceiling, so build a longer array than it
        json arr = json::array();
        for (int i = 0; i < (int)si::kMaxStops + 8; i++) {
            arr.push_back("s" + std::to_string(i));
        }
        json b = json::object();
        b["stop"] = arr;
        check(si::parse_stop(b).size() == si::kMaxStops, "stop array capped at kMaxStops");
    }

    // precedence: chat_template_kwargs, then enable_thinking, then thinking,
    // then reasoning_effort
    check(!si::parse_thinking(J("{}")), "thinking defaults off");
    check(si::parse_thinking(J(R"({"chat_template_kwargs": {"enable_thinking": true}})")), "kwargs enable_thinking");
    check(!si::parse_thinking(J(R"({"chat_template_kwargs": {"enable_thinking": false}, "enable_thinking": true})")),
          "chat_template_kwargs wins over enable_thinking");
    check(si::parse_thinking(J(R"({"enable_thinking": true})")), "enable_thinking");
    check(!si::parse_thinking(J(R"({"enable_thinking": false, "thinking": true})")),
          "enable_thinking wins over thinking");
    check(si::parse_thinking(J(R"({"thinking": true})")), "thinking");
    check(si::parse_thinking(J(R"({"reasoning_effort": "high"})")), "reasoning_effort high -> on");
    check(!si::parse_thinking(J(R"({"reasoning_effort": "none"})")), "reasoning_effort none -> off");
}

static void test_tools() {
    printf("parse_tools_json\n");
    const char * two = R"({"tools": [{"type":"function","function":{"name":"a"}},
                                     {"type":"function","function":{"name":"b"}}]})";
    check(si::parse_tools_json(J("{}")).empty(), "no tools -> empty");
    check(si::parse_tools_json(J(R"({"tools": []})")).empty(), "empty tools array -> empty");

    const std::string all = si::parse_tools_json(J(two));
    check(json::parse(all).size() == 2, "tool_choice auto keeps every tool");

    {
        json b = J(two);
        b["tool_choice"] = "none";
        check(si::parse_tools_json(b).empty(), "tool_choice none -> empty");
    }

    {
        json b = J(two);
        b["tool_choice"] = J(R"({"type":"function","function":{"name":"b"}})");
        const json picked = json::parse(si::parse_tools_json(b));
        check(picked.size() == 1, "tool_choice by nested name keeps one tool");
        check(picked[0]["function"]["name"] == "b", "tool_choice picked the named tool");
    }
    {
        json b = J(two);
        b["tool_choice"] = J(R"({"name": "a"})");
        const json picked = json::parse(si::parse_tools_json(b));
        check(picked.size() == 1 && picked[0]["function"]["name"] == "a", "tool_choice by flat name keeps one tool");
    }
    {
        // a tool_choice naming a function that is not offered must not pass a
        // malformed (unfiltered) array through to the template
        json b = J(two);
        b["tool_choice"] = J(R"({"function":{"name":"missing"}})");
        check(si::parse_tools_json(b).empty(), "tool_choice naming an absent tool -> empty");
    }
}

static void test_messages() {
    printf("parse_messages\n");
    std::vector<si::media_part> media;

    {
        media.clear();
        const auto m = si::parse_messages(J(R"({"messages": [{"role": "user", "content": "hi"}]})"), media);
        check(m.size() == 1, "one message parsed");
        check(m[0].role == "user" && m[0].content == "hi", "string content");
        check(m[0].parts.empty(), "text-only message keeps the plain path (parts cleared)");
        check(media.empty(), "no media parts");
    }
    check(si::parse_messages(J("{}"), media).empty(), "no messages -> empty");
    check(si::parse_messages(J(R"({"messages": "not an array"})"), media).empty(), "messages not an array -> empty");

    {
        // every media kind, in placeholder order, and the mixed text+media shape
        media.clear();
        json b = J(R"({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "look:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            {"type": "video_url", "video_url": "http://x/v.mp4"},
            {"type": "input_audio", "input_audio": {"data": "BB", "format": "wav"}},
            {"type": "audio_url", "audio_url": {"url": "http://x/a.wav", "format": "mp3"}}]}]})");
        const auto m = si::parse_messages(b, media);
        check(m.size() == 1, "media message parsed");
        check(m[0].content == "look:", "text parts are concatenated into content");
        check(!m[0].parts.empty(), "a message with media keeps its parts");
        check(media.size() == 4, "four media parts, in order");
        if (media.size() == 4) {
            check(media[0].kind == si::chat_part_kind::IMAGE && media[0].url == "data:image/png;base64,AA",
                  "image_url part");
            check(media[1].kind == si::chat_part_kind::VIDEO && media[1].url == "http://x/v.mp4", "video_url part");
            check(media[2].kind == si::chat_part_kind::AUDIO && media[2].data == "BB" && media[2].format == "wav",
                  "input_audio inline data");
            check(media[3].kind == si::chat_part_kind::AUDIO && media[3].url == "http://x/a.wav"
                      && media[3].format == "mp3",
                  "audio_url part");
        }
    }
    {
        // the `image`/`video` shorthand spellings, with a bare string payload
        media.clear();
        json b = J(R"({"messages": [{"content": [{"type": "image", "image": "data:image/png;base64,AA"}]}]})");
        si::parse_messages(b, media);
        check(media.size() == 1 && media[0].url == "data:image/png;base64,AA", "image shorthand");
    }

    {
        // assistant-side fields: reasoning aliases, tool_calls, tool_call_id, name
        media.clear();
        const auto m = si::parse_messages(
            J(R"({"messages": [{"role": "assistant",
                 "reasoning_content": "because",
                 "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": {"x": 1}}}],
                 "tool_call_id": "c0", "name": "bot"}]})"),
            media);
        check(m.size() == 1 && m[0].reasoning_content == "because", "reasoning_content");
        check(m[0].tool_calls.size() == 1, "one tool call");
        if (m[0].tool_calls.size() == 1) {
            check(m[0].tool_calls[0].id == "c1" && m[0].tool_calls[0].name == "f", "tool call id and name");
            check(m[0].tool_calls[0].arguments == R"({"x":1})", "object arguments re-serialized as JSON");
        }
        check(m[0].tool_call_id == "c0" && m[0].name == "bot", "tool_call_id and name");

        const auto r = si::parse_messages(J(R"({"messages": [{"reasoning": "alt"}]})"), media);
        check(r.size() == 1 && r[0].reasoning_content == "alt", "`reasoning` is an alias");
        const auto fc = si::parse_messages(J(R"({"messages": [{"function_call": {"name": "g", "arguments": "{}"}}]})"),
                                           media);
        check(fc.size() == 1 && fc[0].tool_calls.size() == 1 && fc[0].tool_calls[0].name == "g",
              "legacy function_call");
        const auto none = si::parse_messages(J(R"({"messages": [{"tool_calls": [{"function": {"name": ""}}]}]})"), media);
        check(none.size() == 1 && none[0].tool_calls.empty(), "a nameless tool call is dropped");
    }
}

static void test_prompt_rejection() {
    printf("check_prompt\n");
    check(si::check_prompt(2048, 10).ok, "a prompt that fits is accepted");
    check(si::check_prompt(2048, 10).body.empty(), "an accepted prompt has no body");
    check(si::check_prompt(2048, 2048).ok, "exactly max_seq is accepted");

    const si::prompt_verdict ev = si::check_prompt(2048, 0);
    check(!ev.ok, "an empty prompt is rejected");
    // the two refusals must be distinguishable, or the caller's log line claims
    // a context-length problem for a prompt that is merely empty
    check(!ev.too_long, "an empty prompt is not a context-length refusal");
    const json empty = json::parse(ev.body);
    check(!empty.empty(), "an empty prompt carries a body");
    check(empty["error"]["message"].get<std::string>().find("at least one token") != std::string::npos,
          "empty prompt names the reason");
    check(empty["error"]["type"] == "invalid_request_error", "empty prompt uses invalid_request_error");

    const si::prompt_verdict ov = si::check_prompt(2048, 4096);
    check(!ov.ok && ov.too_long, "an over-long prompt is rejected as too_long");
    const json over = json::parse(ov.body);
    check(!over.empty(), "an over-long prompt carries a body");
    check(over["error"]["code"] == "context_length_exceeded", "over-long prompt uses context_length_exceeded");
    check(over["max_seq"] == 2048, "the body echoes max_seq so a client can adapt");
    check(over["prompt_tokens"] == 4096, "the body echoes the offending length");
}

static void test_media_bytes() {
    printf("b64_decode / load_media_bytes\n");
    std::vector<uint8_t> bytes;
    std::string err;

    check(si::b64_decode("aGVsbG8=", bytes) && bytes.size() == 5, "base64 decodes");
    check(std::string(bytes.begin(), bytes.end()) == "hello", "base64 payload is right");
    check(si::b64_decode("aGVs\nbG8=", bytes) && bytes.size() == 5, "base64 skips whitespace");
    check(!si::b64_decode("aGVsbG8*", bytes), "an out-of-alphabet byte is an error");
    check(si::b64_decode("", bytes) && bytes.empty(), "empty base64 -> empty");

    si::media_part p;
    p.url = "data:image/png;base64,aGVsbG8=";
    check(si::load_media_bytes(p, bytes, err) && std::string(bytes.begin(), bytes.end()) == "hello",
          "data: URL decoded locally");

    p.url = "data:image/png,raw";
    check(!si::load_media_bytes(p, bytes, err) && err.find("base64") != std::string::npos,
          "a non-base64 data: URL is refused");

    p.url = "data:image/png;base64";
    check(!si::load_media_bytes(p, bytes, err) && err.find("malformed") != std::string::npos,
          "a data: URL without a comma is malformed");

    p.url = "ftp://host/f.png";
    check(!si::load_media_bytes(p, bytes, err) && err.find("unsupported media URL") != std::string::npos,
          "an unknown scheme is refused by name");

    p = si::media_part{};
    p.data = "aGVsbG8=";
    check(si::load_media_bytes(p, bytes, err), "inline base64 (input_audio) decoded");
    p.data = "not base64!";
    check(!si::load_media_bytes(p, bytes, err), "invalid inline base64 refused");

    p = si::media_part{};
    check(!si::load_media_bytes(p, bytes, err) && err.find("neither url nor data") != std::string::npos,
          "a part with neither url nor data is refused");

    // Remote media must not become readable just because a media part named a
    // URL: with the default policy the refusal comes from url_fetch, not from
    // the scheme check, which is what distinguishes "policy off" from
    // "unsupported scheme".  (test_url_fetch covers the enabled policies.)
    if (!si::env::flag("PF_MM_URL_FETCH")) {
        p = si::media_part{};
        p.url = "http://127.0.0.1:9/x.png";
        check(!si::load_media_bytes(p, bytes, err), "remote media is refused by default");
        check(err.find("PF_MM_URL_FETCH") != std::string::npos,
              "the refusal names the switch that would allow it");
    }
}

int main() {
    test_params();
    test_counts();
    test_stop_and_thinking();
    test_tools();
    test_messages();
    test_prompt_rejection();
    test_media_bytes();

    if (g_fail) {
        printf("\n%d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("\nall request-parsing checks OK\n");
    return 0;
}