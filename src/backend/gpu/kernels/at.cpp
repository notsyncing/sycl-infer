// ---------------------------------------------------------------------------
// Audio encoder kernels (Qwen3.5 "AuT"-style tower).
//
// Only the two pieces that differ from the vision tower live here: the 1D conv
// stem and the 1D RoPE.  Everything else (GEMM, LayerNorm, GELU, bias add,
// bidirectional attention) reuses the shared vision kernels in vit.cpp.
// ---------------------------------------------------------------------------
#include "kernels.h"
#include "common/rope.h"
#include "kernel_utils.h"

#include <cmath>

namespace si {

using namespace sycl;

void at_conv1d_launch(queue & q, const float * x, int x_frames, int x_in, int taps, int stride, int pad,
                      const float * w, const float * b, float * out, int y_frames, int w_out) {
    if (y_frames <= 0 || w_out <= 0) {
        return;
    }
    q.parallel_for(sycl::range<2>((size_t)y_frames, (size_t)w_out), [=](id<2> i) {
        const int t = (int)i[0];
        const int o = (int)i[1];
        float acc = b ? b[o] : 0.f;
        const float * wr = w + (size_t)o * taps * x_in;
        for (int tap = 0; tap < taps; tap++) {
            const int row = t * stride + tap - pad;
            if (row < 0 || row >= x_frames) {
                continue;
            }
            const float * xr = x + (size_t)row * x_in;
            const float * ww = wr + tap * x_in;
            for (int i2 = 0; i2 < x_in; i2++) {
                acc = sycl::fma(ww[i2], xr[i2], acc);
            }
        }
        out[(size_t)t * w_out + o] = acc;
    });
}

void at_rope1d_launch(queue & q, float * qkv, int qkv_stride, int n_tok, int n_head, int head_dim, float rope_base) {
    if (n_tok <= 0 || n_head <= 0) {
        return;
    }
    const int half = head_dim / 2;
    const int total = n_tok * n_head * half;
    const float log2b = sycl::log2(rope_base);
    q.parallel_for(sycl::range<1>((size_t)total), [=](id<1> it) {
        const int pair = it[0] % half;
        const int h = (it[0] / half) % n_head;
        const int t = (int)(it[0] / (half * n_head));
        const float theta = rope_theta((float)t, pair, head_dim, log2b);
        const float c = sycl::cos(theta), s = sycl::sin(theta);
        // Q and K live in the first two thirds of the fused qkv row
        float * qrow = qkv + (size_t)t * qkv_stride + h * head_dim;
        float * krow = qrow + n_head * head_dim;
        const float q0 = qrow[pair], q1 = qrow[pair + half];
        qrow[pair] = q0 * c - q1 * s;
        qrow[pair + half] = q0 * s + q1 * c;
        const float k0 = krow[pair], k1 = krow[pair + half];
        krow[pair] = k0 * c - k1 * s;
        krow[pair + half] = k0 * s + k1 * c;
    });
}

} // namespace si