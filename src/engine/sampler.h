#pragma once
#include <cstdint>
#include <random>
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
};

struct sampler_state {
    std::mt19937_64 rng;
    void seed(uint64_t s) {
        rng.seed(s);
    }
};

int sample_token(const float * logits, int n_vocab, const gen_params & gp, const std::vector<int> & recent,
                 sampler_state & ss);

} // namespace si
