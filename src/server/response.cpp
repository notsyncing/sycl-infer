#include "response.h"

// Named directly rather than through response.h: this file uses these symbols,
// and Diagnostics.UnusedIncludes: Strict wants the providing header.
#include "json_util.h"
#include "response_parser.h"
#include "sse.h"
#include "tokenizer.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>

namespace si {

void lp_tracker::on_token(int id, float lp, const std::vector<std::pair<int, float>> & top) {
    cur_id = id;
    cur_lp = lp;
    cur_top = top;
    cur_attached = false;
}

void lp_tracker::attach(lp_fragment & f) {
    if (!cur_attached && cur_id >= 0) {
        f.logprob = cur_lp;
        f.top = cur_top;
        cur_attached = true;
    }
}

json chat_delta_json(const response_piece & p, int & tool_index, long long & reasoning_tokens) {
    if (p.kind == response_piece_kind::reasoning) {
        reasoning_tokens++;
        return {{"reasoning_content", p.text}};
    }
    if (p.kind == response_piece_kind::content) {
        return {{"content", p.text}};
    }
    json call = {{"index", tool_index},
                 {"id", p.call.id},
                 {"type", "function"},
                 {"function", {{"name", p.call.name}, {"arguments", p.call.arguments}}}};
    tool_index++;
    return {{"tool_calls", json::array({call})}};
}

json chat_chunk(const std::string & id, const std::string & model, uint64_t created, int index, const json & delta,
                const json & finish_reason, const json & logprobs) {
    json ch = json::array();
    json choice = {{"index", index}, {"delta", delta}, {"finish_reason", finish_reason}};
    if (!logprobs.is_null()) {
        choice["logprobs"] = logprobs;
    }
    ch.push_back(std::move(choice));
    return {{"id", id},           {"object", "chat.completion.chunk"},
            {"created", created}, {"model", model},
            {"choices", ch}};
}

json text_chunk(const std::string & id, const std::string & model, uint64_t created, int index, const std::string & text,
                const json & finish_reason, const json & logprobs) {
    json ch = json::array();
    json choice = {{"index", index}, {"text", text}, {"finish_reason", finish_reason}};
    if (!logprobs.is_null()) {
        choice["logprobs"] = logprobs;
    }
    ch.push_back(std::move(choice));
    return {{"id", id}, {"object", "text_completion"}, {"created", created}, {"model", model}, {"choices", ch}};
}

std::string sse_usage_frame(const sse_session & sc) {
    json uj = {{"id", sc.id},
               {"object", sc.chat ? "chat.completion.chunk" : "text_completion"},
               {"created", sc.created},
               {"model", sc.model},
               {"choices", json::array()},
               {"usage", usage_json(sc.prompt_tokens, sc.completion_tokens, sc.reasoning_tokens, sc.cached_tokens)}};
    return "data: " + dump_json(uj) + "\n\n";
}

void sse_error(const std::shared_ptr<sse_session> & sc, int index, const char * msg) {
    json d = {{"error", {{"message", msg}, {"type", "server_error"}}}};
    if (sc->chat) {
        sc->q->push("data: " + dump_json(chat_chunk(sc->id, sc->model, sc->created, index, d, "stop")) + "\n\n");
    }
}

json tools_to_json(const std::vector<response_tool_call> & tools) {
    json arr = json::array();
    for (const auto & t : tools) {
        arr.push_back({{"id", t.id},
                       {"type", "function"},
                       {"function", {{"name", t.name}, {"arguments", t.arguments}}}});
    }
    return arr;
}

json bytes_json(const std::string & s) {
    json a = json::array();
    for (unsigned char c : s) {
        a.push_back((int)c);
    }
    return a;
}

json chat_logprobs_json(const std::vector<lp_fragment> & es, const tokenizer & tk) {
    json content = json::array();
    for (const lp_fragment & e : es) {
        json tops = json::array();
        for (const auto & p : e.top) {
            const std::string ts = tk.token_piece(p.first);
            tops.push_back({{"token", ts}, {"logprob", p.second}, {"bytes", bytes_json(ts)}});
        }
        content.push_back(
            {{"token", e.text}, {"logprob", e.logprob}, {"bytes", bytes_json(e.text)}, {"top_logprobs", tops}});
    }
    return {{"content", content}, {"refusal", nullptr}};
}

json completion_logprobs_json(const std::vector<sequence::token_out> & toks, const tokenizer & tk) {
    json tokens = json::array();
    json token_logprobs = json::array();
    json top_logprobs = json::array();
    json text_offset = json::array();
    size_t off = 0;
    for (const sequence::token_out & t : toks) {
        if (t.id < 0) {
            continue;
        }
        const std::string ts = tk.token_piece(t.id);
        tokens.push_back(ts);
        token_logprobs.push_back(t.logprob);
        json m = json::object();
        for (const auto & p : t.top) {
            m[tk.token_piece(p.first)] = p.second;
        }
        top_logprobs.push_back(std::move(m));
        text_offset.push_back((int)off);
        off += ts.size();
    }
    return {{"tokens", tokens},
            {"token_logprobs", token_logprobs},
            {"top_logprobs", top_logprobs},
            {"text_offset", text_offset}};
}

json chat_message_json(const choice_out & out, bool thinking) {
    json msg = {{"role", "assistant"}};
    if (!out.tools.empty()) {
        msg["content"] = out.content.empty() ? json(nullptr) : json(out.content);
        msg["tool_calls"] = tools_to_json(out.tools);
    } else {
        msg["content"] = out.content;
    }
    if (thinking && !out.reasoning.empty()) {
        msg["reasoning_content"] = out.reasoning;
    }
    return msg;
}

json usage_json(long long prompt, long long completion, long long reasoning, long long cached) {
    return {
        {"prompt_tokens", (int)prompt},
        {"completion_tokens", (int)completion},
        {"total_tokens", (int)(prompt + completion)},
        // OpenAI prompt caching detail ...
        {"prompt_tokens_details", {{"cached_tokens", (int)cached}, {"audio_tokens", 0}}},
        // ... and the DeepSeek-style top-level pair (same numbers)
        {"prompt_cache_hit_tokens", (int)cached},
        {"prompt_cache_miss_tokens", (int)(prompt - cached)},
        {"completion_tokens_details",
         {{"reasoning_tokens", (int)reasoning},
          {"audio_tokens", 0},
          {"accepted_prediction_tokens", 0},
          {"rejected_prediction_tokens", 0}}},
    };
}

std::string gen_id() {
    static std::atomic<uint64_t> counter{0};
    char buf[64];
    snprintf(buf, sizeof(buf), "chatcmpl-%llu", (unsigned long long)counter.fetch_add(1) + 1);
    return buf;
}

} // namespace si
