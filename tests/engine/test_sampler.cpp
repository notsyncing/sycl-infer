// Unit tests for logit_bias and logprob reporting in the sampler.  CPU only.
#include <cmath>
#include <cstdio>
#include <vector>

#include "sampler.h"

using namespace si;

static int g_fail = 0;

static void check(bool ok, const char * name) {
    if (ok) {
        printf("  [%s] OK\n", name);
        return;
    }
    g_fail++;
    printf("  [%s] FAILED\n", name);
}

int main() {
    {
        std::vector<float> lg = {5.f, 1.f, 1.f, 1.f};
        gen_params gp;
        gp.temperature = 0.f;
        gp.logit_bias[2] = 100.f;
        sampler_state ss;
        ss.seed(1);
        check(sample_token(lg.data(), 4, gp, {}, ss) == 2, "logit_bias forces token");
    }
    {
        std::vector<float> lg = {5.f, 1.f, 1.f, 1.f};
        gen_params gp;
        gp.temperature = 0.f;
        gp.logit_bias[0] = -100.f;
        sampler_state ss;
        ss.seed(1);
        check(sample_token(lg.data(), 4, gp, {}, ss) == 1, "logit_bias bans token");
    }
    {
        std::vector<float> lg = {0.f, 0.f, 0.f, 0.f};
        gen_params gp;
        gp.temperature = 1.f;
        gp.logprobs = true;
        gp.top_logprobs = 2;
        sampler_state ss;
        ss.seed(2);
        sample_logprobs lp;
        sample_token(lg.data(), 4, gp, {}, ss, &lp);
        const double want = -std::log(4.0);
        check(std::fabs(lp.logprob - want) < 1e-5, "chosen logprob uniform");
        check(lp.top.size() == 2, "top size honored");
        check(!lp.top.empty() && std::fabs(lp.top[0].second - want) < 1e-5, "top logprob uniform");
    }
    {
        std::vector<float> lg = {0.5f, -1.f, 2.f, 0.f, 0.3f};
        gen_params gp;
        gp.temperature = 1.f;
        gp.logprobs = true;
        gp.top_logprobs = 5;
        sampler_state ss;
        ss.seed(3);
        sample_logprobs lp;
        sample_token(lg.data(), 5, gp, {}, ss, &lp);
        double sum = 0.0;
        for (const auto & p : lp.top) {
            sum += std::exp((double)p.second);
        }
        check(std::fabs(sum - 1.0) < 1e-4, "top logprobs normalized");
    }
    {
        std::vector<float> lg = {0.f, 0.f};
        gen_params gp;
        gp.temperature = 0.f;
        gp.need_score = true;
        sampler_state ss;
        ss.seed(4);
        sample_logprobs lp;
        sample_token(lg.data(), 2, gp, {}, ss, &lp);
        check(std::fabs(lp.logprob - (-std::log(2.0))) < 1e-5, "need_score computes chosen logprob");
        check(lp.top.empty(), "need_score skips top alternatives");
    }

    {
        // A temperature-0 tie goes to the lowest index (strict `>`), which is
        // also the device argmax's rule on the qwen4exp path.
        std::vector<float> lg = {2.f, 2.f, 1.f};
        gen_params gp;
        gp.temperature = 0.f;
        sampler_state ss;
        ss.seed(5);
        check(sample_token(lg.data(), 3, gp, {}, ss) == 0, "greedy tie -> lowest index");
    }
    {
        std::vector<float> lg = {1.f, 5.f, 2.f};
        gen_params gp;
        gp.temperature = 1.f;
        gp.top_k = 1; // top_k == 1 is the greedy branch even above temperature 0
        sampler_state ss;
        ss.seed(6);
        check(sample_token(lg.data(), 3, gp, {}, ss) == 1, "top_k=1 is greedy argmax");
    }
    {
        // min_p is relative to the *maximum* probability, not to 1: p =
        // {0.667, 0.333}, threshold 0.9*0.667 = 0.6, so the second is dropped.
        std::vector<float> lg = {0.f, std::log(0.5f)};
        gen_params gp;
        gp.temperature = 1.f;
        gp.top_k = 0;
        gp.top_p = 1.f;
        gp.min_p = 0.9f;
        sampler_state ss;
        ss.seed(7);
        bool all_first = true;
        for (int i = 0; i < 64; i++) {
            all_first = all_first && sample_token(lg.data(), 2, gp, {}, ss) == 0;
            ss.seed(7u + (uint64_t)i);
        }
        check(all_first, "min_p drops the low-probability tail");
    }
    {
        // p = {0.6, 0.3, 0.1}: the cumulative mass is already past 0.5 after
        // the first token, so nucleus sampling keeps exactly it.
        std::vector<float> lg = {std::log(0.6f), std::log(0.3f), std::log(0.1f)};
        gen_params gp;
        gp.temperature = 1.f;
        gp.top_k = 0;
        gp.top_p = 0.5f;
        sampler_state ss;
        ss.seed(8);
        bool all_first = true;
        for (int i = 0; i < 64; i++) {
            all_first = all_first && sample_token(lg.data(), 3, gp, {}, ss) == 0;
            ss.seed(8u + (uint64_t)i);
        }
        check(all_first, "top_p keeps the nucleus");
    }
    {
        // repeat_penalty > 1 divides a positive logit, so a recently seen token
        // loses the argmax.
        std::vector<float> lg = {4.f, 4.f};
        gen_params gp;
        gp.temperature = 0.f;
        gp.repeat_penalty = 4.f;
        sampler_state ss;
        ss.seed(9);
        check(sample_token(lg.data(), 2, gp, {1}, ss) == 0, "repeat_penalty demotes a recent token");
    }
    {
        // A positive presence penalty must demote, not promote (OpenAI).
        std::vector<float> lg = {0.f, 0.f};
        gen_params gp;
        gp.temperature = 0.f;
        gp.presence_penalty = 5.f;
        sampler_state ss;
        ss.seed(10);
        check(sample_token(lg.data(), 2, gp, {1}, ss) == 0, "presence_penalty demotes a seen token");
    }
    {
        // The frequency term multiplies the occurrence count.
        std::vector<float> lg = {0.f, 0.f};
        gen_params gp;
        gp.temperature = 0.f;
        gp.frequency_penalty = 1.f;
        sampler_state ss;
        ss.seed(11);
        check(sample_token(lg.data(), 2, gp, {1, 1}, ss) == 0, "frequency_penalty counts repeats");
    }
    {
        // Same seed, same distribution -> same token (the scheduler relies on
        // this for a deterministic replay with a pinned seed).
        std::vector<float> lg = {1.f, 1.f, 1.f, 1.f};
        gen_params gp;
        gp.temperature = 1.f;
        gp.top_k = 0;
        gp.top_p = 1.f;
        sampler_state a, b;
        a.seed(1234);
        b.seed(1234);
        bool same = true;
        for (int i = 0; i < 32; i++) {
            same = same && sample_token(lg.data(), 4, gp, {}, a) == sample_token(lg.data(), 4, gp, {}, b);
        }
        check(same, "same seed -> same stream");
    }

    {
        gen_params gp;
        gp.temperature = 0.f;
        check(gp.speculative_greedy(), "plain greedy eligible for speculation");
        gp.logit_bias[1] = 10.f;
        check(!gp.speculative_greedy(), "greedy logit_bias bypasses speculation");
        gp.logit_bias.clear();
        gp.repeat_penalty = 1.1f;
        check(!gp.speculative_greedy(), "greedy repeat penalty bypasses speculation");
        gp.repeat_penalty = 1.f;
        gp.presence_penalty = 0.1f;
        check(!gp.speculative_greedy(), "greedy presence penalty bypasses speculation");
        gp.presence_penalty = 0.f;
        gp.frequency_penalty = 0.1f;
        check(!gp.speculative_greedy(), "greedy frequency penalty bypasses speculation");
        gp.frequency_penalty = 0.f;
        gp.temperature = 1.f;
        gp.top_k = 1;
        check(gp.speculative_greedy(), "top_k=1 remains greedy eligible");
        gp.top_k = 40;
        check(!gp.speculative_greedy(), "sampled decode bypasses speculation");
    }

    printf(g_fail ? "sampler test FAILED (%d)\n" : "sampler test OK\n", g_fail);
    return g_fail ? 1 : 0;
}
