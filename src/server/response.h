#pragma once
// Building the OpenAI response bodies and SSE frames.
//
// The mirror of request.h: given what was generated, produce the JSON the client
// receives.  Pure and httplib-free, so the wire format is assertable on its own -
// the two shapes that are easiest to get subtly wrong (logprobs, and the
// reasoning/tool_calls split) are exactly the ones that were previously reachable
// only by driving a live generation.
//
// Two accumulator types travel with it because they are response state, not
// generation state: `lp_tracker` pairs a sampled token's logprob with the parser
// fragment it will end up inside, and `choice_out` is everything one choice
// produced.
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "json_util.h"
#include "response_parser.h" // response_piece, response_tool_call
#include "scheduler.h"      // sequence::token_out
#include "sse.h"            // sse_session
#include "tokenizer.h"

namespace si {

// One text fragment of a chat response, with the logprob of the token that
// produced it and that token's top alternatives.
struct lp_fragment {
    std::string text;
    float logprob = 0.f;
    std::vector<std::pair<int, float>> top;
};

// The two places that pair a sampled token's logprob with its parser fragment:
// `on_token` stashes them, `attach` folds them into the next text piece the
// parser emits and only attaches once.
struct lp_tracker {
    int cur_id = -1;
    float cur_lp = 0.f;
    std::vector<std::pair<int, float>> cur_top;
    bool cur_attached = true;

    void on_token(int id, float lp, const std::vector<std::pair<int, float>> & top);
    void attach(lp_fragment & f);
};

// Everything one choice produced, for both the streaming and the batched paths.
struct choice_out {
    std::string content;
    std::string reasoning;
    std::vector<response_tool_call> tools;
    std::string finish = "stop";
    int n_gen = 0;
    int prompt_tokens = 0;
    int reasoning_tokens = 0; // generated tokens that made up reasoning_content
    int cached_tokens = 0;    // prompt tokens served from the prefix cache
    // logprobs / best_of scoring; empty unless the request asked for them
    std::vector<sequence::token_out> tokens;
    std::vector<lp_fragment> chat_lp;
    double score = 0.0;
};

// One parsed response fragment -> the json delta for an SSE chunk.
// tool_index is the running tool_calls counter (function calls carry it).
json chat_delta_json(const response_piece & p, int & tool_index, long long & reasoning_tokens);

// chat `logprobs`: one entry per content fragment (OpenAI `logprobs.content`).
json chat_logprobs_json(const std::vector<lp_fragment> & es, const tokenizer & tk);

// completions `logprobs`: legacy parallel arrays over the generated tokens.
json completion_logprobs_json(const std::vector<sequence::token_out> & toks, const tokenizer & tk);

// `usage`, including the prefix-cache hit/miss counts in both the OpenAI
// `prompt_tokens_details` form and the DeepSeek-style top-level pair.
json usage_json(long long prompt, long long completion, long long reasoning, long long cached);

json tools_to_json(const std::vector<response_tool_call> & tools);
json bytes_json(const std::string & s);
json chat_message_json(const choice_out & out, bool thinking);

// A monotonically increasing `chatcmpl-N` id.
std::string gen_id();

// ---- SSE frames -------------------------------------------------------------
// sse.h stays free of the JSON chunk helpers, so the frames live here; the
// session's counters are read to fill the usage frame.

json chat_chunk(const std::string & id, const std::string & model, uint64_t created, int index, const json & delta,
                const json & finish_reason, const json & logprobs = json());
json text_chunk(const std::string & id, const std::string & model, uint64_t created, int index, const std::string & text,
                const json & finish_reason, const json & logprobs = json());

// The usage chunk sse_session::choice_done emits before [DONE].  A disconnected
// client never sees it: q->push drops everything after a cancel.
std::string sse_usage_frame(const sse_session & sc);
void sse_error(const std::shared_ptr<sse_session> & sc, int index, const char * msg);

} // namespace si
