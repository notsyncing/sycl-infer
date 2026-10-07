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
