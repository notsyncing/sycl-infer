#pragma once
#include <cstdint>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace si {

struct gen_params {
    int max_tokens = 256;
    float temperature = 1.0f;
    float top_p = 0.95f;
    int top_k = 40;
    float min_p = 0.0f;
    float repeat_penalty = 1.0f;
    int repeat_last_n = 64;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    uint64_t seed = 0;
    bool ignore_eos = false;
    // OpenAI logit_bias: token id -> additive logit bias (range [-100, 100]).
    // Empty by default, so sample_token does no extra work.
    std::unordered_map<int, float> logit_bias;
    // Report logprobs for the sampled token (chat `logprobs` or the completion
    // `logprobs` integer); `top_logprobs` bounds the alternatives (0..20).
    bool logprobs = false;
    int top_logprobs = 0;
    // internal: compute the chosen token's logprob for best_of scoring even
    // when the client did not ask for logprobs.
    bool need_score = false;

    bool wants_logprobs() const {
        return logprobs || need_score;
    }
};

// Per-token sampling detail.  Filled only when sample_token is given a
// non-null `out`; `top` is sized by gen_params::top_logprobs.
struct sample_logprobs {
    float logprob = 0.f;                    // logprob of the sampled token
    std::vector<std::pair<int, float>> top; // up to top_logprobs alternatives, highest first
};

struct sampler_state {
    std::mt19937_64 rng;
    void seed(uint64_t s) {
        rng.seed(s);
    }
};

int sample_token(const float * logits, int n_vocab, const gen_params & gp, const std::vector<int> & recent,
                 sampler_state & ss, sample_logprobs * out = nullptr);

} // namespace si
