// cpu_gdn with an *asymmetric* head count: q/k carry n_k_heads heads, v/state
// carry n_heads, and value head h must be paired with q/k head (h % n_k_heads).
// The reference implementation expands the q/k head axis with ggml_repeat_4d,
// whose tiling is modulo/interleaved - not blocked (h * n_k / n_heads).  The two
// agree only when n_k_heads == n_heads, which is exactly the 0.8B reference
// model, so this case has to be tested explicitly (Qwen3.8-27B: 16 vs 48).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cpu_types.h"

using namespace si;

static float sigmoid_(float v) {
    return 1.0f / (1.0f + std::exp(-v));
}

// deterministic values in [-1, 1)
static float rnd(int i) {
    const unsigned h = (unsigned)(i * 2654435761u + 1013904223u);
    return (float)((h >> 8) & 0xffff) / 32768.0f - 1.0f;
}

int main() {
    const int D = 128; // head_dim: the kernel tiles it as 4 x 32 lanes
    const int Hk = 2;  // key/q heads
    const int H = 6;   // value heads
    const int T = 3;   // tokens
    const int tpb = 32;
    const int slot = 1;
    const int n_slots = 2;
    const int cd = 2 * Hk * D + H * D;
    const float scale = 1.0f / std::sqrt((float)D);

    std::vector<float> conv((size_t)T * cd), alpha((size_t)T * H), beta((size_t)T * H), dts(H), sa(H);
    for (size_t i = 0; i < conv.size(); i++) {
        conv[i] = rnd((int)i);
    }
    for (size_t i = 0; i < alpha.size(); i++) {
        alpha[i] = rnd(1000 + (int)i) * 0.5f;
        beta[i] = rnd(2000 + (int)i);
    }
    for (int h = 0; h < H; h++) {
        dts[h] = rnd(3000 + h) * 0.2f;
        sa[h] = -0.5f - std::fabs(rnd(4000 + h)); // ssm_a is negative in the model
    }

    std::vector<float> st_init((size_t)n_slots * H * D * D);
    for (size_t i = 0; i < st_init.size(); i++) {
        st_init[i] = rnd(5000 + (int)i) * 0.05f;
    }

    cpu_step_info info;
    std::memset(&info, 0, sizeof(info));
    info.n_rows = 1;
    info.n_real = T;
    info.tpb = tpb;
    info.pos[0] = 0;
    info.slot[0] = slot;
    info.active[0] = 1;

    std::vector<float> out((size_t)tpb * H * D);
    cpu_pc_snap snap{};
    std::vector<float> stk = st_init, stref = st_init;
    cpu_gdn(conv.data(), alpha.data(), dts.data(), sa.data(), beta.data(), stk.data(), out.data(), &info, D, Hk, H, cd,
            scale, n_slots, 1, 0, -1, T, snap);

    // scalar reference with the modulo (interleaved) head pairing
    double maxd = 0, maxv = 0;
    std::vector<float> ref((size_t)tpb * H * D);
    for (int h = 0; h < H; h++) {
        const int kk = h % Hk;
        float * S = &stref[(size_t)slot * H * D * D + (size_t)h * D * D];
        for (int t = 0; t < T; t++) {
            const float * qt = &conv[(size_t)t * cd + 0 + kk * D];
            const float * kt = &conv[(size_t)t * cd + Hk * D + kk * D];
            const float * vt = &conv[(size_t)t * cd + 2 * Hk * D + h * D];
            const float bt = sigmoid_(beta[(size_t)t * H + h]);
            const float sp = std::log(1.0f + std::exp(alpha[(size_t)t * H + h] + dts[h]));
            const float g = std::exp(sa[h] * sp);
            for (int col = 0; col < D; col++) {
                float kvv = 0;
                for (int i = 0; i < D; i++) {
                    kvv += S[(size_t)col * D + i] * kt[i];
                }
                const float del = (vt[col] - g * kvv) * bt;
                float atn = 0;
                for (int i = 0; i < D; i++) {
                    S[(size_t)col * D + i] = g * S[(size_t)col * D + i] + kt[i] * del;
                    atn += S[(size_t)col * D + i] * qt[i];
                }
                ref[(size_t)t * H * D + h * D + col] = atn * scale;
            }
        }
    }
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = std::fabs((double)out[i] - (double)ref[i]);
        maxd = std::max(maxd, d);
        maxv = std::max(maxv, std::fabs((double)ref[i]));
    }
    // the state the kernel wrote back must match the reference's too
    double smax = 0, smaxv = 0;
    for (size_t i = 0; i < stk.size(); i++) {
        const double d = std::fabs((double)stk[i] - (double)stref[i]);
        smax = std::max(smax, d);
        smaxv = std::max(smaxv, std::fabs((double)stref[i]));
    }
    const bool ok = maxd < 1e-4 * std::max(1.0, maxv) && smax < 1e-4 * std::max(1.0, smaxv);
    printf("cpu_gdn heads(Hk=%d H=%d D=%d T=%d): out max|diff|=%.3e (max|ref|=%.3f)  state max|diff|=%.3e %s\n", Hk,
           H, D, T, maxd, maxv, smax, ok ? "OK" : "MISMATCH");
    return ok ? 0 : 1;
}
