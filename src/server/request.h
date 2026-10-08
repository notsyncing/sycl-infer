#pragma once
// OpenAI request bodies -> engine inputs.
//
// Everything here is pure: JSON in, plain values out.  No httplib, no engine
// state, no globals - so the whole request surface (params, messages, tools,
// completion prompts, the length rejection) can be exercised without a socket,
// which is why the two functions that used to reach into the engine take the one
// field each actually needs (`tokenizer &` and `int max_seq`) instead.
//
// What this file deliberately does NOT do is decide whether a media part may be
// read: parse_messages only records what a part was, and media_fetch decides.
#include <cstddef>
#include <string>
#include <vector>

#include "chat.h"
#include "json_util.h"
#include "media_fetch.h" // media_part, produced here and consumed there
#include "sampler.h"     // gen_params
#include "tokenizer.h"

namespace si {

// per-request ceilings
constexpr int kMaxN = 16;       // choices
constexpr int kMaxJobs = 64;    // completions: prompt count * max(n, best_of)
constexpr size_t kMaxStops = 16;

// Sampling / decoding parameters, with the API's aliases and defaults applied
// (`max_completion_tokens`, both repetition_penalty spellings, chat's boolean
// `logprobs` vs completions' count, logit_bias clamped to +-100).
gen_params parse_params(const json & body);

// number of choices, clamped to [1, kMaxN]
int parse_n(const json & body);

// completions `best_of`: generate this many candidates and return the `n`
// highest-scoring ones (`best_of < n` is invalid, as in OpenAI).
int parse_best_of(const json & body, int n);

// chat `stream_options.include_usage`
bool parse_include_usage(const json & body);

// `stop` as a string or an array of strings, capped at kMaxStops.
std::vector<std::string> parse_stop(const json & body);

// `enable_thinking` for the chat template.  `chat_template_kwargs` wins (that is
// the llama.cpp convention), then the flat `enable_thinking`/`thinking` flags,
// then `reasoning_effort` (anything but "none" turns reasoning on).
bool parse_thinking(const json & body);

// The OpenAI `tools` array serialized back to JSON, filtered by `tool_choice`.
// Returns "" when no tool is offered (so the template omits the tools block).
std::string parse_tools_json(const json & body);

// `messages` -> chat messages, plus every media part in placeholder order.
// A text-only message keeps the plain path (its `parts` list is cleared), which
// is what the renderers branch on.
std::vector<chat_msg> parse_messages(const json & body, std::vector<media_part> & media);

// The completions `prompt` in any of its four shapes.
struct completion_prompts {
    std::vector<std::vector<int>> tokens;
    std::vector<std::string> texts;
};

// OpenAI `prompt` may be a string, an array of strings, an array of token ids,
// or an array of token-id arrays.  Each entry becomes one completion prompt;
// anything unusable yields a single empty prompt rather than none, so the caller
// has one rejection path instead of two.
completion_prompts parse_completion_prompts(tokenizer & tk, const json & body);

// A prompt that does not fit max_seq cannot be prefilled, and an empty prompt
// cannot be decoded either (the scheduler would leave its sequence active
// forever).  Both are refused before admission instead of hanging or returning
// an empty HTTP 200.
//
// The verdict names *which* refusal it is, because the two want different log
// lines: an earlier version logged every refusal as `context_length_exceeded`,
// so an empty prompt was reported as "0 tokens > max_seq=N", which reads as a
// context problem when it is not one.  The body carries the distinction too
// (`invalid_request_error` vs `code: context_length_exceeded`).
struct prompt_verdict {
    bool ok = true;
    bool too_long = false; // meaningful only when !ok
    std::string body;      // the OpenAI error body; empty when ok
};
prompt_verdict check_prompt(int max_seq, size_t prompt_tokens);

} // namespace si
