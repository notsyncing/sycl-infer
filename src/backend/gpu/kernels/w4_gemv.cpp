// 4-bit (u4) decode GEMV: the M==1 counterpart of dnnl_gemm::gemm_w4.
//
// Weights are the native-width packing from common/w4.h: u4 values [N][K] with
// K inner and the low nibble first, plus per-(g,n) f16 step/offset planes where
// g = k/32.  A group's 32 nibbles are exactly 16 contiguous bytes, so one
// 16-byte load per lane per group covers the whole super-block.
//
//   y[n] = sum_g asa[g] * ( step[g][n]*QDOT_g[n] + off[g][n]*XS[g] )
//
// with QDOT the integer dot of the group's weights against the group's
// activations and XS its activation sum (from dnnl_gemm's w4_xs_launch).
// The nibble packing pairs k with k+1 in one byte, so the activations are
// pre-split into even/odd k planes (w4_split_act_launch): then the dp4a operands
// are plain 4-byte chunks of those planes.
//
// The step/off planes are (g,n) ordered (what oneDNN's grouped scales require),
// which is strided for a per-row GEMV - so each workgroup stages its RB rows'
// metadata into local memory first, where the load is contiguous in n.
#include "kernels.h"
#include "kernel_utils.h"

#include "dp4a.h"

namespace si {

using namespace sycl;
using namespace si::kd;

// f16 bits -> f32 (device side)
static inline float w4_h2f(uint16_t h) {
    sycl::half x;
    __builtin_memcpy((void *)&x, &h, sizeof(x));
    return (float)x;
}
void w4_split_act_launch(queue & q, const int8_t * axg, int8_t * axe, int8_t * axo, int M, int K) {
    const int kh = K / 2;
    q.parallel_for(range<1>((size_t)M * kh), [=](id<1> i) {
        const int m = (int)(i / (size_t)kh);
        const int j = (int)(i % (size_t)kh);
        const int8_t * s = axg + (size_t)m * K + 2 * (size_t)j;
        axe[(size_t)m * kh + j] = s[0];
        axo[(size_t)m * kh + j] = s[1];
    });
}

void w4_gemv_launch(queue & q, const uint8_t * vals, const uint16_t * scale, const uint16_t * off,
                    const int8_t * axe, const int8_t * axo, const uint16_t * asa, const float * xs, float * out,
                    int out_stride, const float * residual, float alpha, int K, int N) {
    const int ng = K / 32;
    constexpr int RB = 8; // rows per workgroup
    constexpr int TX = RB * 32;
    const int nwg = (N + RB - 1) / RB;
    q.submit([&](handler & h) {
        // step + off for this tile's RB rows, widened to f32
        local_accessor<float, 1> meta((size_t)2 * RB * ng, h);
        h.parallel_for(nd_range<1>((size_t)nwg * TX, TX), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int lid = (int)it.get_local_id(0);
            const int sg = lid / 32;
            const int lane = lid % 32;
            const int n0 = (int)it.get_group(0) * RB;
            for (int i = lid; i < RB * ng; i += TX) {
                const int r = i / ng;
                const int g = i % ng;
                const int n = n0 + r;
                const float s = (n < N) ? w4_h2f(scale[(size_t)g * N + n]) : 0.f;
                const float o = (n < N) ? w4_h2f(off[(size_t)g * N + n]) : 0.f;
                meta[i] = s;
                meta[(size_t)RB * ng + i] = o;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const int n = n0 + sg;
            if (n >= N) {
                return;
            }
            const uint8_t * wrow = vals + (size_t)n * (size_t)(K / 2);
            const int8_t * xrow_e = axe;
            const int8_t * xrow_o = axo;
            float acc = 0.f;
            for (int g = lane; g < ng; g += 32) {
                const uint4 wv = *reinterpret_cast<const uint4 *>(wrow + (size_t)g * 16);
                const uint32_t l0 = wv.x() & 0x0F0F0F0Fu, l1 = wv.y() & 0x0F0F0F0Fu;
                const uint32_t l2 = wv.z() & 0x0F0F0F0Fu, l3 = wv.w() & 0x0F0F0F0Fu;
                const uint32_t h0 = (wv.x() >> 4) & 0x0F0F0F0Fu, h1 = (wv.y() >> 4) & 0x0F0F0F0Fu;
                const uint32_t h2 = (wv.z() >> 4) & 0x0F0F0F0Fu, h3 = (wv.w() >> 4) & 0x0F0F0F0Fu;
                const uint32_t * xe = reinterpret_cast<const uint32_t *>(xrow_e + (size_t)g * 16);
                const uint32_t * xo = reinterpret_cast<const uint32_t *>(xrow_o + (size_t)g * 16);
                // dp4a_s8u8(a, b, c) = c + sum (int8)a * (uint8)b:
                // activations are the signed operand, nibbles the unsigned one
                int32_t qd = 0;
                qd = dp4a_s8u8(xe[0], l0, qd);
                qd = dp4a_s8u8(xe[1], l1, qd);
                qd = dp4a_s8u8(xe[2], l2, qd);
                qd = dp4a_s8u8(xe[3], l3, qd);
                qd = dp4a_s8u8(xo[0], h0, qd);
                qd = dp4a_s8u8(xo[1], h1, qd);
                qd = dp4a_s8u8(xo[2], h2, qd);
                qd = dp4a_s8u8(xo[3], h3, qd);
                const float sa = w4_h2f(asa[g]);
                acc += sa * (meta[(size_t)sg * ng + g] * (float)qd + meta[(size_t)RB * ng + (size_t)sg * ng + g] * xs[g]);
            }
            const sub_group sgg = it.get_sub_group();
            const float tot = reduce_over_group(sgg, acc, plus<float>());
            if (lane == 0) {
                // M == 1: `n` is the output *column* of the single output row,
                // so both out and residual are indexed by n directly (the row
                // stride only matters for M > 1, which uses the matmul).
                (void)out_stride;
                float v = alpha * tot;
                if (residual) {
                    v += residual[n];
                }
                out[n] = v;
            }
        });
    });
}

// int8 decode GEMV with per-32-group scales (M == 1), the int8 counterpart of
// w4_gemv_launch:
//   y[n] = sum_g asa[g] * step[g][n] * QDOT_g[n]
// with QDOT the signed dot of the group's int8 weights and activations.  The
// weights are signed, and dp4a_s8u8 takes (signed, unsigned), so the weight word
// is XORed with 0x80 (which adds 128 to each byte) and the resulting
// 128*sum(x) bias is removed with the group's activation sum.
void i8_grp_gemv_launch(queue & q, const int8_t * w8, const uint16_t * wsc, const int8_t * xq, const uint16_t * asa,
                        const float * xs, float * out, const float * residual, float alpha, int K, int N) {
    const int ng = K / 32;
    constexpr int RB = 8;
    constexpr int TX = RB * 32;
    const int nwg = (N + RB - 1) / RB;
    q.submit([&](handler & h) {
        local_accessor<float, 1> meta((size_t)RB * ng, h);
        h.parallel_for(nd_range<1>((size_t)nwg * TX, TX), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int lid = (int)it.get_local_id(0);
            const int sg = lid / 32;
            const int lane = lid % 32;
            const int n0 = (int)it.get_group(0) * RB;
            for (int i = lid; i < RB * ng; i += TX) {
                const int r = i / ng;
                const int g = i % ng;
                const int n = n0 + r;
                meta[i] = (n < N) ? w4_h2f(wsc[(size_t)g * N + n]) : 0.f;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const int n = n0 + sg;
            if (n >= N) {
                return;
            }
            const int8_t * wrow = w8 + (size_t)n * K;
            float acc = 0.f;
            for (int g = lane; g < ng; g += 32) {
                const uint32_t * wv = reinterpret_cast<const uint32_t *>(wrow + (size_t)g * 32);
                const uint32_t * xv = reinterpret_cast<const uint32_t *>(xq + (size_t)g * 32);
                int32_t qd = 0;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    qd = dp4a_s8u8(xv[j], wv[j] ^ 0x80808080u, qd);
                }
                qd -= 128 * (int32_t)xs[g]; // undo the XOR bias
                acc += w4_h2f(asa[g]) * meta[(size_t)sg * ng + g] * (float)qd;
            }
            const sub_group sgg = it.get_sub_group();
            const float tot = reduce_over_group(sgg, acc, plus<float>());
            if (lane == 0) {
                float v = alpha * tot;
                if (residual) {
                    v += residual[n];
                }
                out[n] = v;
            }
        });
    });
}

} // namespace si
