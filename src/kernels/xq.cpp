#include "kernels.h"
#include "kernel_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

using namespace sycl;
using namespace si::kd;

// ---------------------------------------------------------------------------
// Activation quantization for the DP4A path.
//
// Symmetric int8 with one scale per 32 values (fp32: activations can be ~1e-5
// where an fp16 scale would underflow), plus the sum of each 16-value half so
// the integer dot can apply the weight min/zero-point:
//   sum_k w_k x_k = sx * (sw * sum_k qw_k qx_k - mw * sum_k qx_k)
// layout: x8[(g32*TB + t)*32 + i], xmeta[(g32*TB + t)], xsumq[(g16*TB + t)]
// ---------------------------------------------------------------------------
void xq_launch(queue & q, const float * x, const float * up, int x_stride, int up_stride, int8_t * x8,
               sycl::float2 * xmeta, int32_t * xsumq, const step_info * info, int TB, int K) {
    constexpr int G = 32;
    const int groups = K / G;
    // NOTE: n_real must be read *inside* the kernel: a host-side read would be
    // captured by value at graph-capture time, when step_info is still zeroed.
    // TB is the number of tokens this call quantizes (kMaxT for a prefill
    // chunk, the batch size for decode, the whole prompt for chunk-batched
    // prefill); n_real is tokens-per-row and is only used by the stateful
    // kernels, so it must not gate the quantization here (mode 2 would keep
    // every chunk but the first unquantized).
    const int ntok = TB;
    q.parallel_for(nd_range<1>((size_t)TB * groups * 32, 32), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int gid = it.get_group(0);
        const int t = gid / groups;
        const int g = gid % groups;
        const int lane = it.get_local_id(0);
        const sub_group sg = it.get_sub_group();
        // group-major layout: (t, g, i) -> x8[(g*TB + t)*G + i]
        const size_t o = ((size_t)g * TB + t) * G;
        float v = 0.f;
        if (t < ntok) {
            v = x[(size_t)t * x_stride + g * G + lane];
            // ffn_down: activations are silu(gate) * up (same order as the fp32 kernel)
            if (up) {
                v = silu_f(v) * up[(size_t)t * up_stride + g * G + lane];
            }
        }
        const float mx = sg_max(sycl::fabs(v), sg);
        const float scale = mx > 0.f ? mx / 127.0f : 1.0f;
        int32_t qv = 0;
        if (t < ntok) {
            qv = (int32_t)sycl::round(v / scale);
            qv = sycl::max(-127, sycl::min(127, qv));
        }
        x8[o + lane] = (int8_t)qv; // rows beyond n_real stay zeroed
        // sum per 16-value half (masks < 16 stay inside the lane's half)
        int32_t s = qv;
#pragma unroll
        for (int m = 1; m < 16; m <<= 1) {
            s += sycl::permute_group_by_xor(sg, s, m);
        }
        // one int2 {half0 sum, half1 sum} per 32-value group.  The permute is a
        // sub-group collective: it must be executed by all lanes (a divergent
        // call returns garbage).
        const int32_t s1 = sycl::permute_group_by_xor(sg, s, 16);
        if (lane == 0) {
            reinterpret_cast<sycl::int2 *>(xsumq)[(size_t)g * TB + t] = sycl::int2(s, s1);
        }
        if (lane == 0) {
            // activation scales must be fp32: activations can be ~1e-5, where
            // an fp16 scale would drop into subnormals (or zero) and destroy
            // the reconstruction.  Weight scales stay fp16 (small weights only
            // contribute small absolute error).
            xmeta[(size_t)g * TB + t] = sycl::float2(scale, 0.f);
        }
    });
}

} // namespace si
