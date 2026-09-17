#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace si {

int sample_token(const float * logits, int n_vocab, const gen_params & gp, const std::vector<int> & recent,
                 sampler_state & ss) {
    // repetition / presence / frequency penalties
    std::vector<float> scr;
    const float * lg = logits;
    const bool do_pen = gp.repeat_penalty != 1.0f || gp.presence_penalty != 0.f || gp.frequency_penalty != 0.f;
    if (do_pen && !recent.empty()) {
        scr.assign(logits, logits + n_vocab);
        const int n = (int)recent.size();
        const int from = std::max(0, n - gp.repeat_last_n);
        std::unordered_map<int, int> counts;
        for (int i = from; i < n; i++) {
            counts[recent[i]]++;
        }
        for (auto & kv : counts) {
            float & v = scr[kv.first];
            if (gp.repeat_penalty != 1.0f) {
                v = v > 0 ? v / gp.repeat_penalty : v * gp.repeat_penalty;
            }
            v += gp.presence_penalty + gp.frequency_penalty * kv.second;
        }
        lg = scr.data();
    }

    if (gp.temperature <= 0.f || gp.top_k == 1) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++) {
            if (lg[i] > lg[best]) {
                best = i;
            }
        }
        return best;
    }

    const float inv_t = 1.0f / gp.temperature;
    float mx = -INFINITY;
    for (int i = 0; i < n_vocab; i++) {
        float v = lg[i] * inv_t;
        if (v > mx) {
            mx = v;
        }
    }
    // candidate set: everything within 20 nats of the max
    static thread_local std::vector<std::pair<float, int>> cand;
    cand.clear();
    const float thr = mx - 20.0f;
    for (int i = 0; i < n_vocab; i++) {
        const float v = lg[i] * inv_t;
        if (v >= thr) {
            cand.emplace_back(v, i);
        }
    }
    std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.first > b.first; });

    // softmax over candidates
    double sum = 0.0;
    for (auto & c : cand) {
        c.first = std::exp(c.first - mx);
        sum += c.first;
    }
    const float maxp = (float)(cand[0].first / sum);
    const float minp_thr = maxp * gp.min_p;

    int nmax = (int)cand.size();
    if (gp.top_k > 0) {
        nmax = std::min(nmax, gp.top_k);
    }
    double cum = 0.0;
    int keep = 0;
    for (int i = 0; i < nmax; i++) {
        const float p = (float)(cand[i].first / sum);
        if (i > 0 && gp.min_p > 0.f && p < minp_thr) {
            break;
        }
        cum += p;
        keep = i + 1;
        if (cum >= gp.top_p) {
            break;
        }
    }
    if (keep <= 0) {
        keep = 1;
    }
    // renormalize over kept
    double ksum = 0.0;
    for (int i = 0; i < keep; i++) {
        ksum += cand[i].first;
    }
    std::uniform_real_distribution<double> dist(0.0, ksum);
    double r = dist(ss.rng);
    double acc = 0.0;
    for (int i = 0; i < keep; i++) {
        acc += cand[i].first;
        if (r <= acc) {
            return cand[i].second;
        }
    }
    return cand[keep - 1].second;
}

} // namespace si
