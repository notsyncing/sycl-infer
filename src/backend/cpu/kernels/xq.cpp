// CPU activation quantization for the SIn/DP4A path (mirror of
// src/backend/gpu/kernels/xq.cpp): symmetric int8 with one fp32 scale per
// 32-values and the per-16-value sums used by the weight min correction.
// Layout is group-major: x8[(g*TB + t)*32 + i], xmeta fp32 pairs {scale,0},
// xsumq two int32 per group.
#include "common.h"

namespace si {

void cpu_xq(const float * x, const float * up, int x_stride, int up_stride, int8_t * x8, float * xmeta, int32_t * xsumq,
            const cpu_step_info * info, int TB, int K) {
    (void)info;
    const int groups = K / 32;
    par(TB * groups, [&](int gid) {
        const int t = gid / groups;
        const int g = gid % groups;
        float v[32];
        float mx = 0.0f;
        for (int i = 0; i < 32; i++) {
            float a = x[(size_t)t * x_stride + g * 32 + i];
            if (up) {
                a = cpu_silu(a) * up[(size_t)t * up_stride + g * 32 + i];
            }
            v[i] = a;
            mx = std::max(mx, std::fabs(a));
        }
        const float scale = mx > 0.0f ? mx / 127.0f : 1.0f;
        int32_t s0 = 0, s1 = 0;
        int8_t * dst = x8 + ((size_t)g * TB + t) * 32;
        for (int i = 0; i < 32; i++) {
            const int32_t q = (int32_t)std::max(-127.0f, std::min(127.0f, std::round(v[i] / scale)));
            dst[i] = (int8_t)q;
            if (i < 16) {
                s0 += q;
            } else {
                s1 += q;
            }
        }
        const size_t gi = (size_t)g * TB + t;
        xmeta[gi * 2] = scale; // sycl::float2 {scale, 0}
        xmeta[gi * 2 + 1] = 0.0f;
        xsumq[gi * 2 + 0] = s0;
        xsumq[gi * 2 + 1] = s1;
    });
}

} // namespace si
