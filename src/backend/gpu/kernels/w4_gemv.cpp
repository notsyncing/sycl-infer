// 4-bit (u4) and grouped int8 decode GEMV kernels: the M==1 counterparts of
// dnnl_gemm::gemm_w4 / gemm.
//
// Weights are the native-width packing from common/w4.h: u4 values [N][K] with
// K inner and the low nibble first, plus per-(g,n) f16 step/offset planes where
// g = k/32.  A group's 32 nibbles are exactly 16 contiguous bytes, so one
// 32-byte (two uint4) load per lane per group covers the whole super-block.
//
//   y[n] = sum_g asa[g] * ( step[g][n]*QDOT_g[n] + off[g][n]*XS[g] )
//
// with QDOT the integer dot of the group's weights against the group's
// activations and XS its activation sum (produced by dnnl_gemm's
// activation quantizer, one sub-group reduction over the group).
// The nibble packing pairs k with k+1 in one byte, so the activations are
// pre-split into even/odd k planes (w4_split_act_launch): then the dp4a operands
// are plain 4-byte chunks of those planes.
//
// The step/off planes are (g,n) ordered (what oneDNN's grouped scales require),
// which is strided for a per-row GEMV, so each workgroup stages a tile of them
// into local memory.  The staging index is *g-major / row-inner* so consecutive
// lanes of a staging pass read consecutive n of the same g: that turns a
// cache-line-per-scale read into one contiguous line per (g, tile) and is worth
// 10-25% of the kernel's bandwidth (measured on the A770; see
// reports/gemv_staging.md).  The tile is staged as f16 (the planes are f16) so
// RB can be 16 without exceeding the SLM budget.
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

namespace {
// rows per workgroup for the u4 kernel; two f16 planes must fit in SLM.
// PF_W4_RB overrides it for A/B (the SLM tile size trades staging coalescing
// against occupancy: 2*RB*ng*2 bytes must fit the 48 KB budget).
constexpr int kW4RB = 16;
} // namespace

static int w4_rb_override() {
    static const int rb = [] {
        const char * e = getenv("PF_W4_RB");
        return e ? atoi(e) : 0;
    }();
    return rb;
}

template <int RB>
static void w4_gemv_impl(queue & q, const uint8_t * vals, const uint16_t * scale, const uint16_t * off,
                         const int8_t * axe, const int8_t * axo, const uint16_t * asa, const float * xs, float * out,
                         int out_stride, const float * residual, float alpha, int K, int N) {
    const int ng = K / 32;
    constexpr int TX = RB * 32;
    const int nwg = (N + RB - 1) / RB;
    q.submit([&](handler & h) {
        // step + off for this tile's RB rows, f16 (halves the SLM vs f32)
        local_accessor<uint16_t, 1> meta((size_t)2 * RB * ng, h);
        h.parallel_for(nd_range<1>((size_t)nwg * TX, TX), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int lid = (int)it.get_local_id(0);
            const int sg = lid / 32;
            const int lane = lid % 32;
            const int n0 = (int)it.get_group(0) * RB;
            // g-major / row-inner: consecutive lanes read consecutive n (one
            // contiguous run per group) instead of striding by N
            for (int i = lid; i < RB * ng; i += TX) {
                const int g = i / RB;
                const int r = i % RB;
                const int n = n0 + r;
                const bool ok = n < N;
                const size_t src = (size_t)g * N + n;
                meta[i] = ok ? scale[src] : (uint16_t)0;
                meta[(size_t)RB * ng + i] = ok ? off[src] : (uint16_t)0;
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
                // one group = 32 nibbles = 16 contiguous bytes
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
                const size_t mo = (size_t)g * RB + sg;
                acc += sa * (w4_h2f(meta[mo]) * (float)qd + w4_h2f(meta[(size_t)RB * ng + mo]) * xs[g]);
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

void w4_gemv_launch(queue & q, const uint8_t * vals, const uint16_t * scale, const uint16_t * off,
                    const int8_t * axe, const int8_t * axo, const uint16_t * asa, const float * xs, float * out,
                    int out_stride, const float * residual, float alpha, int K, int N) {
    const int ng = K / 32;
    const int ov = w4_rb_override();
    const auto fits = [&](int rb) { return (size_t)2 * (size_t)rb * (size_t)ng * sizeof(uint16_t) <= 48 * 1024; };
    if (ov == 8 && fits(8)) {
        w4_gemv_impl<8>(q, vals, scale, off, axe, axo, asa, xs, out, out_stride, residual, alpha, K, N);
    } else if (ov == 32 && fits(32)) {
        w4_gemv_impl<32>(q, vals, scale, off, axe, axo, asa, xs, out, out_stride, residual, alpha, K, N);
    } else if (ov == 16 && fits(16)) {
        w4_gemv_impl<16>(q, vals, scale, off, axe, axo, asa, xs, out, out_stride, residual, alpha, K, N);
    } else if (fits(kW4RB)) {
        w4_gemv_impl<kW4RB>(q, vals, scale, off, axe, axo, asa, xs, out, out_stride, residual, alpha, K, N);
    } else {
        w4_gemv_impl<8>(q, vals, scale, off, axe, axo, asa, xs, out, out_stride, residual, alpha, K, N);
    }
}

// int8 decode GEMV with per-32-group scales (M == 1), the int8 counterpart of
// w4_gemv_launch:
//   y[n] = sum_g asa[g] * step[g][n] * QDOT_g[n]
// with QDOT the signed dot of the group's int8 weights and activations.  The
// weights are signed, and dp4a_s8u8 takes (signed, unsigned), so the weight word
// is XORed with 0x80 (which adds 128 to each byte) and the resulting
// 128*sum(x) bias is removed with the group's activation sum.
template <int RB>
static void i8_grp_gemv_impl(queue & q, const int8_t * w8, const uint16_t * wsc, const int8_t * xq,
                             const uint16_t * asa, const float * xs, float * out, const float * residual,
                             float alpha, int K, int N) {
    const int ng = K / 32;
    constexpr int TX = RB * 32;
    const int nwg = (N + RB - 1) / RB;
    q.submit([&](handler & h) {
        local_accessor<uint16_t, 1> meta((size_t)RB * ng, h);
        h.parallel_for(nd_range<1>((size_t)nwg * TX, TX), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int lid = (int)it.get_local_id(0);
            const int sg = lid / 32;
            const int lane = lid % 32;
            const int n0 = (int)it.get_group(0) * RB;
            // g-major / row-inner staging (see the file header)
            for (int i = lid; i < RB * ng; i += TX) {
                const int g = i / RB;
                const int r = i % RB;
                const int n = n0 + r;
                meta[i] = (n < N) ? wsc[(size_t)g * N + n] : (uint16_t)0;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const int n = n0 + sg;
            if (n >= N) {
                return;
            }
            const int8_t * wrow = w8 + (size_t)n * K;
            float acc = 0.f;
            for (int g = lane; g < ng; g += 32) {
                const uint4 w0 = *reinterpret_cast<const uint4 *>(wrow + (size_t)g * 32);
                const uint4 w1 = *reinterpret_cast<const uint4 *>(wrow + (size_t)g * 32 + 16);
                const uint4 x0 = *reinterpret_cast<const uint4 *>(xq + (size_t)g * 32);
                const uint4 x1 = *reinterpret_cast<const uint4 *>(xq + (size_t)g * 32 + 16);
                int32_t qd = 0;
                qd = dp4a_s8u8((int32_t)x0.x(), w0.x() ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x0.y(), w0.y() ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x0.z(), w0.z() ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x0.w(), w0.w() ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x1.x(), w1.x() ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x1.y(), w1.y() ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x1.z(), w1.z() ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x1.w(), w1.w() ^ 0x80808080u, qd);
                qd -= 128 * (int32_t)xs[g]; // undo the XOR bias
                acc += w4_h2f(asa[g]) * w4_h2f(meta[(size_t)g * RB + sg]) * (float)qd;
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

void i8_grp_gemv_launch(queue & q, const int8_t * w8, const uint16_t * wsc, const int8_t * xq,
                        const uint16_t * asa, const float * xs, float * out, const float * residual, float alpha,
                        int K, int N) {
    const int ng = K / 32;
    const int ov = w4_rb_override();
    const auto fits = [&](int rb) { return (size_t)rb * (size_t)ng * sizeof(uint16_t) <= 48 * 1024; };
    if (ov == 8 && fits(8)) {
        i8_grp_gemv_impl<8>(q, w8, wsc, xq, asa, xs, out, residual, alpha, K, N);
    } else if (ov == 32 && fits(32)) {
        i8_grp_gemv_impl<32>(q, w8, wsc, xq, asa, xs, out, residual, alpha, K, N);
    } else if (ov == 16 && fits(16)) {
        i8_grp_gemv_impl<16>(q, w8, wsc, xq, asa, xs, out, residual, alpha, K, N);
    } else if (fits(kW4RB)) {
        i8_grp_gemv_impl<kW4RB>(q, w8, wsc, xq, asa, xs, out, residual, alpha, K, N);
    } else {
        i8_grp_gemv_impl<8>(q, w8, wsc, xq, asa, xs, out, residual, alpha, K, N);
    }
}


// ---- native-width 5-bit (Q5_K) ---------------------------------------------
// The weights are the u4 nibble plane (so the even/odd activation split above is
// reused unchanged) plus a 1-bit fifth-bit plane, recombined as lo4 | (bit << 4)
// - an OR, exact since lo4 < 16.  That is the whole extra cost over the u4
// kernel: 8 SLM LUT loads + 16 ALU per 32 values, against reading 0.75 B/weight
// instead of the int8 conversion's 1.125 and keeping the native values exactly.
// Measured at the card's ~405 GB/s read ceiling (the int8 GEMV reaches 92% of
// it), so the unpack is entirely hidden - see dev/bench_native56.cpp.
//
// The `hi` plane is 4 bytes per 32-group, laid out in the *split-plane* element
// order so that one nibble of an expanded mask covers exactly the four elements
// of one dp4a operand: bits 0-15 are the even elements (2i -> bit i), bits 16-31
// the odd ones (2i+1 -> bit i).  The 16-entry SLM table maps such a nibble to
// four bytes of 0/1 in one load.
static constexpr int kK5RB = 16;

template <int RB>
static void k5_gemv_impl(queue & q, const uint8_t * vals, const uint8_t * hi, const uint16_t * scale,
                         const uint16_t * off, const int8_t * axe, const int8_t * axo, const uint16_t * asa,
                         const float * xs, float * out, const float * residual, float alpha, int K, int N) {
    const int ng = K / 32;
    constexpr int TX = RB * 32;
    const int nwg = (N + RB - 1) / RB;
    q.submit([&](handler & h) {
        local_accessor<uint16_t, 1> meta((size_t)2 * RB * ng, h);
        local_accessor<uint32_t, 1> lt(16, h);
        h.parallel_for(nd_range<1>((size_t)nwg * TX, TX), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int lid = (int)it.get_local_id(0);
            const int sg = lid / 32;
            const int lane = lid % 32;
            const int n0 = (int)it.get_group(0) * RB;
            if (lid < 16) {
                lt[lid] = (uint32_t)((lid >> 0) & 1) | ((uint32_t)((lid >> 1) & 1) << 8)
                          | ((uint32_t)((lid >> 2) & 1) << 16) | ((uint32_t)((lid >> 3) & 1) << 24);
            }
            // g-major / row-inner scale staging (see the file header)
            for (int i = lid; i < RB * ng; i += TX) {
                const int g = i / RB;
                const int r = i % RB;
                const int n = n0 + r;
                const bool ok = n < N;
                const size_t src = (size_t)g * N + n;
                meta[i] = ok ? scale[src] : (uint16_t)0;
                meta[(size_t)RB * ng + i] = ok ? off[src] : (uint16_t)0;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const int n = n0 + sg;
            if (n >= N) {
                return;
            }
            const uint8_t * lrow = vals + (size_t)n * (size_t)(K / 2);
            const uint8_t * hrow = hi + (size_t)n * (size_t)(K / 8);
            float acc = 0.f;
            for (int g = lane; g < ng; g += 32) {
                const uint4 wv = *reinterpret_cast<const uint4 *>(lrow + (size_t)g * 16);
                uint32_t l0 = wv.x() & 0x0F0F0F0Fu, l1 = wv.y() & 0x0F0F0F0Fu;
                uint32_t l2 = wv.z() & 0x0F0F0F0Fu, l3 = wv.w() & 0x0F0F0F0Fu;
                uint32_t h0 = (wv.x() >> 4) & 0x0F0F0F0Fu, h1 = (wv.y() >> 4) & 0x0F0F0F0Fu;
                uint32_t h2 = (wv.z() >> 4) & 0x0F0F0F0Fu, h3 = (wv.w() >> 4) & 0x0F0F0F0Fu;
                const uint32_t hb = *reinterpret_cast<const uint32_t *>(hrow + (size_t)g * 4);
                l0 |= lt[(hb >> 0) & 0xFu] << 4;
                l1 |= lt[(hb >> 4) & 0xFu] << 4;
                l2 |= lt[(hb >> 8) & 0xFu] << 4;
                l3 |= lt[(hb >> 12) & 0xFu] << 4;
                h0 |= lt[(hb >> 16) & 0xFu] << 4;
                h1 |= lt[(hb >> 20) & 0xFu] << 4;
                h2 |= lt[(hb >> 24) & 0xFu] << 4;
                h3 |= lt[(hb >> 28) & 0xFu] << 4;
                const uint32_t * xe = reinterpret_cast<const uint32_t *>(axe + (size_t)g * 16);
                const uint32_t * xo = reinterpret_cast<const uint32_t *>(axo + (size_t)g * 16);
                // four independent accumulator chains: the serial one costs ~3%
                int32_t q0 = dp4a_s8u8(xe[0], l0, 0);
                q0 = dp4a_s8u8(xe[1], l1, q0);
                int32_t q1 = dp4a_s8u8(xe[2], l2, 0);
                q1 = dp4a_s8u8(xe[3], l3, q1);
                int32_t q2 = dp4a_s8u8(xo[0], h0, 0);
                q2 = dp4a_s8u8(xo[1], h1, q2);
                int32_t q3 = dp4a_s8u8(xo[2], h2, 0);
                q3 = dp4a_s8u8(xo[3], h3, q3);
                const int32_t qd = (q0 + q1) + (q2 + q3);
                const float sa = w4_h2f(asa[g]);
                const size_t mo = (size_t)g * RB + sg;
                acc += sa * (w4_h2f(meta[mo]) * (float)qd + w4_h2f(meta[(size_t)RB * ng + mo]) * xs[g]);
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

void k5_gemv_launch(queue & q, const uint8_t * vals, const uint8_t * hi, const uint16_t * scale, const uint16_t * off,
                    const int8_t * axe, const int8_t * axo, const uint16_t * asa, const float * xs, float * out,
                    const float * residual, float alpha, int K, int N) {
    const int ng = K / 32;
    const auto fits = [&](int rb) { return (size_t)2 * (size_t)rb * (size_t)ng * sizeof(uint16_t) + 64 <= 48 * 1024; };
    if (fits(kK5RB)) {
        k5_gemv_impl<kK5RB>(q, vals, hi, scale, off, axe, axo, asa, xs, out, residual, alpha, K, N);
    } else {
        k5_gemv_impl<8>(q, vals, hi, scale, off, axe, axo, asa, xs, out, residual, alpha, K, N);
    }
}

// Expand the (nibble, fifth-bit) planes into plain int8 q5 values in [0,31] for
// the prefill matmul:  out[n*K + k] = lo4 | (bit << 4).  Element order matters
// here (oneDNN's s8 weights are [N][K] row-major), while both source planes are
// stored in the split order, so the two halves are re-interleaved with a byte
// spread.  `bit_lut` is the 16-entry table the decode kernel builds in SLM
// (4 bits -> 4 bytes of 0/1), shared here through device memory.
void k5_expand_launch(queue & q, const uint8_t * vals, const uint8_t * hi, const uint32_t * bit_lut, int8_t * out, int K,
                      int N) {
    const int ng = K / 32;
    if (ng <= 0 || N <= 0) {
        return;
    }
    q.parallel_for(range<1>((size_t)N * ng), [=](id<1> i) {
        const int n = (int)(i / (size_t)ng);
        const int g = (int)(i % (size_t)ng);
        const uint8_t * vrow = vals + (size_t)n * (K / 2) + (size_t)g * 16;
        const uint32_t hb = *reinterpret_cast<const uint32_t *>(hi + (size_t)n * (K / 8) + (size_t)g * 4);
        uint32_t w[8];
        // Per 4 input bytes: the even and odd halves are rebuilt as 4-byte words,
        // then interleaved into element order.  A 64-bit byte-spread here costs
        // ~40 instructions per 8 values and capped this kernel at 200 GB/s (49%
        // of the read ceiling, instruction-bound); the 32-bit form below is ~12.
#pragma unroll
        for (int q4 = 0; q4 < 4; q4++) {
            const uint32_t v = *reinterpret_cast<const uint32_t *>(vrow + 4 * q4);
            const uint32_t ev = (v & 0x0F0F0F0Fu) | (bit_lut[(hb >> (4 * q4)) & 0xFu] << 4);
            const uint32_t od = ((v >> 4) & 0x0F0F0F0Fu) | (bit_lut[(hb >> (16 + 4 * q4)) & 0xFu] << 4);
            const uint32_t p = (ev & 0x00FF00FFu) | ((od & 0x00FF00FFu) << 8);
            const uint32_t t = ((ev >> 8) & 0x00FF00FFu) | (((od >> 8) & 0x00FF00FFu) << 8);
            w[2 * q4] = (p & 0xFFFFu) | (t << 16);
            w[2 * q4 + 1] = (p >> 16) | (t & 0xFFFF0000u);
        }
        uint32_t * dw = reinterpret_cast<uint32_t *>(out + (size_t)n * K + (size_t)g * 32);
        *reinterpret_cast<uint4 *>(dw) = uint4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<uint4 *>(dw + 4) = uint4(w[4], w[5], w[6], w[7]);
    });
}

// ---- codebook 4-bit (IQ4_XS / IQ4_NL) --------------------------------------
// The weights are 4-bit indices into the 16-entry int8 table `lut` plus a per-32
// f16 scale per row.  Expanding the indices through the table inside the kernel
// keeps the *native* values (nothing is re-quantized) while reading half the
// bytes the int8 conversion needed.
//
// The index plane is interleaved (byte k = elements 2k / 2k+1), so a *pair* of
// index bytes covers 4 consecutive elements = one dp4a operand, and a 256-entry
// uint16 table `lut16[b] = value(b & 0xF) | value(b >> 4) << 8` turns the pair
// into that operand with one shift and one OR.  That is the whole difference
// from the previous form, which needed four byte lookups plus three shifts and
// three ORs per operand (32 LUT loads + 24 ALU per 32 values, measured 49-67%
// of the card's read ceiling - i.e. instruction-bound).  `lut16` is built in
// SLM for the decode kernel and from a device copy for the prefill expansion.
//   y[n] = sum_g asa[g] * scale[g][n] * QDOT_g[n]
void cb4_expand_launch(queue & q, const uint8_t * idx, const uint16_t * lut16_dev, int8_t * out, int K, int N) {
    const int ng = K / 32;
    if (ng <= 0 || N <= 0) {
        return;
    }
    // one work-item per 32-value group: a 16-byte index load in, two 16-byte
    // int8 stores out (the scalar per-byte form made prefill ~1.5x slower, and
    // the 8 x 32-bit store form of the k5 expansion capped at half the ceiling)
    q.parallel_for(range<1>((size_t)N * ng), [=](id<1> i) {
        const int n = (int)(i / (size_t)ng);
        const int g = (int)(i % (size_t)ng);
        const uint8_t * src = idx + (size_t)n * (size_t)(K / 2) + (size_t)g * 16;
        const uint4 iv = *reinterpret_cast<const uint4 *>(src);
        const uint32_t raw[4] = {iv.x(), iv.y(), iv.z(), iv.w()};
        uint32_t w[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t v = raw[j];
            w[2 * j] = (uint32_t)lut16_dev[(v >> 0) & 0xFF] | ((uint32_t)lut16_dev[(v >> 8) & 0xFF] << 16);
            w[2 * j + 1] = (uint32_t)lut16_dev[(v >> 16) & 0xFF] | ((uint32_t)lut16_dev[(v >> 24) & 0xFF] << 16);
        }
        int8_t * dst = out + (size_t)n * K + (size_t)g * 32;
        *reinterpret_cast<uint4 *>(dst) = uint4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<uint4 *>(dst + 16) = uint4(w[4], w[5], w[6], w[7]);
    });
}

template <int RB>
static void cb4_gemv_impl(queue & q, const uint8_t * idx, const uint16_t * lut16, const uint16_t * scale,
                          const int8_t * xq, const uint16_t * asa, const float * xs, float * out, const float * residual,
                          float alpha, int K, int N) {
    const int ng = K / 32;
    constexpr int TX = RB * 32;
    const int nwg = (N + RB - 1) / RB;
    q.submit([&](handler & h) {
        local_accessor<uint16_t, 1> meta((size_t)RB * ng, h);
        local_accessor<uint16_t, 1> lt(256, h);
        h.parallel_for(nd_range<1>((size_t)nwg * TX, TX), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int lid = (int)it.get_local_id(0);
            const int sg = lid / 32;
            const int lane = lid % 32;
            const int n0 = (int)it.get_group(0) * RB;
            // expand the 16-entry codebook into the 256-entry byte-pair table
            for (int i = lid; i < 256; i += TX) {
                lt[i] = lut16[i];
            }
            // g-major / row-inner scale staging (see the file header)
            for (int i = lid; i < RB * ng; i += TX) {
                const int g = i / RB;
                const int r = i % RB;
                const int n = n0 + r;
                meta[i] = (n < N) ? scale[(size_t)g * N + n] : (uint16_t)0;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const int n = n0 + sg;
            if (n >= N) {
                return;
            }
            const uint8_t * irow = idx + (size_t)n * (size_t)(K / 2);
            float acc = 0.f;
            for (int g = lane; g < ng; g += 32) {
                const uint4 nv = *reinterpret_cast<const uint4 *>(irow + (size_t)g * 16);
                const uint32_t raw[4] = {nv.x(), nv.y(), nv.z(), nv.w()};
                uint32_t w[8];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t v = raw[j];
                    // 2 index bytes -> 4 codebook bytes (elements in order)
                    w[2 * j] = (uint32_t)lt[(v >> 0) & 0xFF] | ((uint32_t)lt[(v >> 8) & 0xFF] << 16);
                    w[2 * j + 1] = (uint32_t)lt[(v >> 16) & 0xFF] | ((uint32_t)lt[(v >> 24) & 0xFF] << 16);
                }
                const uint4 x0 = *reinterpret_cast<const uint4 *>(xq + (size_t)g * 32);
                const uint4 x1 = *reinterpret_cast<const uint4 *>(xq + (size_t)g * 32 + 16);
                int32_t qd = 0;
                qd = dp4a_s8u8((int32_t)x0.x(), w[0] ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x0.y(), w[1] ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x0.z(), w[2] ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x0.w(), w[3] ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x1.x(), w[4] ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x1.y(), w[5] ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x1.z(), w[6] ^ 0x80808080u, qd);
                qd = dp4a_s8u8((int32_t)x1.w(), w[7] ^ 0x80808080u, qd);
                qd -= 128 * (int32_t)xs[g]; // undo the XOR bias
                acc += w4_h2f(asa[g]) * w4_h2f(meta[(size_t)g * RB + sg]) * (float)qd;
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

void cb4_gemv_launch(queue & q, const uint8_t * idx, const uint16_t * lut16, const uint16_t * scale, const int8_t * xq,
                     const uint16_t * asa, const float * xs, float * out, const float * residual, float alpha, int K,
                     int N) {
    const int ng = K / 32;
    const auto fits = [&](int rb) { return (size_t)rb * (size_t)ng * sizeof(uint16_t) + 256 * 2 <= 48 * 1024; };
    if (fits(kW4RB)) {
        cb4_gemv_impl<kW4RB>(q, idx, lut16, scale, xq, asa, xs, out, residual, alpha, K, N);
    } else {
        cb4_gemv_impl<8>(q, idx, lut16, scale, xq, asa, xs, out, residual, alpha, K, N);
    }
}


// ---------------------------------------------------------------------------
// Batched (M > 1) native-u4 GEMM: the M-row counterpart of w4_gemv_impl.
//
// Same lane mapping as the M=1 GEMV (one sub-group per output column, lane = one
// K-partition stepping g += 32, so the 32 lanes read 512 B of consecutive weight
// words) so the arithmetic is bit-identical to it and the weight stream stays
// coalesced.  Measured lessons that shaped this version:
//   * a runtime row index (float acc[MM] + for m<M) demoted the accumulators to
//     local memory: 20 GB/s;
//   * a compile-time M with array/vec accumulators still spilled via the row
//     loop: 38.8 GB/s at M=2 (a hand-written straight-line M=2 body: 119.6);
//   * so every row is expanded with `if constexpr` and accumulated into a
//     constant-index slot, which keeps them in registers.
// Formula, identical to w4_gemv_impl (verified bit-equal by test_w4_gemm):
//   out[m*out_stride + n] = alpha * sum_g sa[m][g] * ( scale[g][n]*qd[m][g]
//                                                      + off[g][n]*xs[m][g] )
#define W4_GEMM_ROW(m)                                                                                  \
    do {                                                                                                \
        const uint4 ev = *reinterpret_cast<const uint4 *>(axe + (size_t)(m) * kh + (size_t)g * 16);     \
        const uint4 ov = *reinterpret_cast<const uint4 *>(axo + (size_t)(m) * kh + (size_t)g * 16);     \
        int32_t qd = 0;                                                                                 \
        qd = dp4a_s8u8(ev.x(), l0, qd);                                                                 \
        qd = dp4a_s8u8(ev.y(), l1, qd);                                                                 \
        qd = dp4a_s8u8(ev.z(), l2, qd);                                                                 \
        qd = dp4a_s8u8(ev.w(), l3, qd);                                                                 \
        qd = dp4a_s8u8(ov.x(), h0, qd);                                                                 \
        qd = dp4a_s8u8(ov.y(), h1, qd);                                                                 \
        qd = dp4a_s8u8(ov.z(), h2, qd);                                                                 \
        qd = dp4a_s8u8(ov.w(), h3, qd);                                                                 \
        acc[u][(m)] += w4_h2f(asa[(size_t)(m) * ng + g]) *                                             \
                       (sc_w * (float)qd + of_w * xs[(size_t)(m) * ng + g]);                            \
    } while (0)

template <int M, int U = 2, int RB = 16>
static void w4_gemm_batched(queue & q, const uint8_t * vals, const uint16_t * scale, const uint16_t * off,
                            const int8_t * axe, const int8_t * axo, const uint16_t * asa, const float * xs,
                            float * out, int out_stride, const float * residual, float alpha, int K, int N) {
    const int ng = K / 32;
    const int kh = K / 2;
    constexpr int TX = RB * 32;
    const int nwg = (N + RB - 1) / RB;
    q.submit([&](handler & h) {
        local_accessor<uint16_t, 1> meta((size_t)2 * RB * ng, h);
        h.parallel_for(nd_range<1>((size_t)nwg * TX, TX), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int lid = (int)it.get_local_id(0);
            const int sg = lid / 32;
            const int lane = lid % 32;
            const int n0 = (int)it.get_group(0) * RB;
            for (int i = lid; i < RB * ng; i += TX) {
                const int g = i / RB;
                const int r = i % RB;
                const int n = n0 + r;
                const bool ok = n < N;
                const size_t src = (size_t)g * N + n;
                meta[i] = ok ? scale[src] : (uint16_t)0;
                meta[(size_t)RB * ng + i] = ok ? off[src] : (uint16_t)0;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const int n = n0 + sg;
            if (n >= N) {
                return;
            }
            const uint8_t * wrow = vals + (size_t)n * (size_t)kh;
            float acc[U][M];
#pragma unroll
            for (int u = 0; u < U; u++) {
#pragma unroll
                for (int m = 0; m < M; m++) {
                    acc[u][m] = 0.f;
                }
            }
            for (int base = lane; base < ng; base += 32 * U) {
#pragma unroll
                for (int u = 0; u < U; u++) {
                    const int g = base + 32 * u;
                    if (g >= ng) {
                        continue;
                    }
                    const uint4 wv = *reinterpret_cast<const uint4 *>(wrow + (size_t)g * 16);
                    const uint32_t l0 = wv.x() & 0x0F0F0F0Fu, l1 = wv.y() & 0x0F0F0F0Fu;
                    const uint32_t l2 = wv.z() & 0x0F0F0F0Fu, l3 = wv.w() & 0x0F0F0F0Fu;
                    const uint32_t h0 = (wv.x() >> 4) & 0x0F0F0F0Fu, h1 = (wv.y() >> 4) & 0x0F0F0F0Fu;
                    const uint32_t h2 = (wv.z() >> 4) & 0x0F0F0F0Fu, h3 = (wv.w() >> 4) & 0x0F0F0F0Fu;
                    const float sc_w = w4_h2f(meta[(size_t)g * RB + sg]);
                    const float of_w = w4_h2f(meta[(size_t)RB * ng + (size_t)g * RB + sg]);
                    if constexpr (M >= 1) {
                        W4_GEMM_ROW(0);
                    }
                    if constexpr (M >= 2) {
                        W4_GEMM_ROW(1);
                    }
                    if constexpr (M >= 3) {
                        W4_GEMM_ROW(2);
                    }
                    if constexpr (M >= 4) {
                        W4_GEMM_ROW(3);
                    }
                    if constexpr (M >= 5) {
                        W4_GEMM_ROW(4);
                    }
                    if constexpr (M >= 6) {
                        W4_GEMM_ROW(5);
                    }
                    if constexpr (M >= 7) {
                        W4_GEMM_ROW(6);
                    }
                    if constexpr (M >= 8) {
                        W4_GEMM_ROW(7);
                    }
                }
            }
            const sub_group sgg = it.get_sub_group();
#pragma unroll
            for (int m = 0; m < M; m++) {
                float s = 0.f;
#pragma unroll
                for (int u = 0; u < U; u++) {
                    s += acc[u][m];
                }
                const float tot = reduce_over_group(sgg, s, plus<float>());
                if (lane == 0) {
                    float v = alpha * tot;
                    if (residual) {
                        v += residual[(size_t)m * out_stride + n];
                    }
                    out[(size_t)m * out_stride + n] = v;
                }
            }
        });
    });
}
#undef W4_GEMM_ROW

// Column-tiled variant: each sub-group covers TN output columns, so the per-row
// activations and per-(row,group) scales/sums (which dominate the traffic: the
// row-expanded kernel falls off as 1/M because they are re-read once per output
// column) are amortised over TN columns.  Everything is expanded with
// `if constexpr` so every accumulator index is a literal and stays in registers
// - a runtime index here cost 3x before.
#define W4_TN_ACC(c, m, u)                                                                              \
    do {                                                                                                \
        const uint4 ev = *reinterpret_cast<const uint4 *>(axe + (size_t)(m) * kh + (size_t)g * 16);     \
        const uint4 ov = *reinterpret_cast<const uint4 *>(axo + (size_t)(m) * kh + (size_t)g * 16);     \
        int32_t qd = 0;                                                                                 \
        qd = dp4a_s8u8(ev.x(), l0, qd);                                                                 \
        qd = dp4a_s8u8(ev.y(), l1, qd);                                                                 \
        qd = dp4a_s8u8(ev.z(), l2, qd);                                                                 \
        qd = dp4a_s8u8(ev.w(), l3, qd);                                                                 \
        qd = dp4a_s8u8(ov.x(), h0, qd);                                                                 \
        qd = dp4a_s8u8(ov.y(), h1, qd);                                                                 \
        qd = dp4a_s8u8(ov.z(), h2, qd);                                                                 \
        qd = dp4a_s8u8(ov.w(), h3, qd);                                                                 \
        acc[(u)][(m)][(c)] += w4_h2f(asa[(size_t)(m) * ng + g]) *                                       \
                              (sc_w[(c)] * (float)qd + of_w[(c)] * xs[(size_t)(m) * ng + g]);           \
    } while (0)

#define W4_TN_COL(c)                                                                                    \
    do {                                                                                                \
        const uint4 wv = wvv[(c)];                                                                      \
        const uint32_t l0 = wv.x() & 0x0F0F0F0Fu, l1 = wv.y() & 0x0F0F0F0Fu;                            \
        const uint32_t l2 = wv.z() & 0x0F0F0F0Fu, l3 = wv.w() & 0x0F0F0F0Fu;                            \
        const uint32_t h0 = (wv.x() >> 4) & 0x0F0F0F0Fu, h1 = (wv.y() >> 4) & 0x0F0F0F0Fu;              \
        const uint32_t h2 = (wv.z() >> 4) & 0x0F0F0F0Fu, h3 = (wv.w() >> 4) & 0x0F0F0F0Fu;              \
        if constexpr (M >= 1) {                                                                         \
            W4_TN_ACC(c, 0, u);                                                                         \
        }                                                                                               \
        if constexpr (M >= 2) {                                                                         \
            W4_TN_ACC(c, 1, u);                                                                         \
        }                                                                                               \
        if constexpr (M >= 3) {                                                                         \
            W4_TN_ACC(c, 2, u);                                                                         \
        }                                                                                               \
        if constexpr (M >= 4) {                                                                         \
            W4_TN_ACC(c, 3, u);                                                                         \
        }                                                                                               \
        if constexpr (M >= 5) {                                                                         \
            W4_TN_ACC(c, 4, u);                                                                         \
        }                                                                                               \
        if constexpr (M >= 6) {                                                                         \
            W4_TN_ACC(c, 5, u);                                                                         \
        }                                                                                               \
        if constexpr (M >= 7) {                                                                         \
            W4_TN_ACC(c, 6, u);                                                                         \
        }                                                                                               \
        if constexpr (M >= 8) {                                                                         \
            W4_TN_ACC(c, 7, u);                                                                         \
        }                                                                                               \
    } while (0)

template <int M, int TN, int RB = 16>
static void w4_gemm_tn(queue & q, const uint8_t * vals, const uint16_t * scale, const uint16_t * off,
                       const int8_t * axe, const int8_t * axo, const uint16_t * asa, const float * xs, float * out,
                       int out_stride, const float * residual, float alpha, int K, int N) {
    static_assert(RB % TN == 0, "RB must be a multiple of TN");
    constexpr int NG_SG = RB / TN;
    constexpr int TX = NG_SG * 32;
    const int ng = K / 32;
    const int kh = K / 2;
    const int nwg = (N + RB - 1) / RB;
    q.submit([&](handler & h) {
        local_accessor<uint16_t, 1> meta((size_t)2 * RB * ng, h);
        h.parallel_for(nd_range<1>((size_t)nwg * TX, TX), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int lid = (int)it.get_local_id(0);
            const int sg = lid / 32;
            const int lane = lid % 32;
            const int nb = (int)it.get_group(0) * RB;
            for (int i = lid; i < RB * ng; i += TX) {
                const int g = i / RB;
                const int r = i % RB;
                const int n = nb + r;
                const bool ok = n < N;
                const size_t src = (size_t)g * N + n;
                meta[i] = ok ? scale[src] : (uint16_t)0;
                meta[(size_t)RB * ng + i] = ok ? off[src] : (uint16_t)0;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const int n0 = nb + sg * TN;
            float acc[1][M][TN];
#pragma unroll
            for (int m = 0; m < M; m++) {
#pragma unroll
                for (int c = 0; c < TN; c++) {
                    acc[0][m][c] = 0.f;
                }
            }
            for (int g = lane; g < ng; g += 32) {
                constexpr int u = 0;
                uint4 wvv[TN];
#pragma unroll
                for (int c = 0; c < TN; c++) {
                    const int n = n0 + c;
                    wvv[c] = (n < N) ? *reinterpret_cast<const uint4 *>(vals + (size_t)n * (size_t)kh
                                                                       + (size_t)g * 16)
                                     : uint4(0, 0, 0, 0);
                }
                float sc_w[TN], of_w[TN];
#pragma unroll
                for (int c = 0; c < TN; c++) {
                    sc_w[c] = w4_h2f(meta[(size_t)g * RB + (sg * TN + c)]);
                    of_w[c] = w4_h2f(meta[(size_t)RB * ng + (size_t)g * RB + (sg * TN + c)]);
                }
                if constexpr (TN >= 1) {
                    W4_TN_COL(0);
                }
                if constexpr (TN >= 2) {
                    W4_TN_COL(1);
                }
                if constexpr (TN >= 4) {
                    W4_TN_COL(2);
                    W4_TN_COL(3);
                }
            }
            const sub_group sgg = it.get_sub_group();
#pragma unroll
            for (int m = 0; m < M; m++) {
#pragma unroll
                for (int c = 0; c < TN; c++) {
                    const int n = n0 + c;
                    if (n >= N) {
                        continue;
                    }
                    const float tot = reduce_over_group(sgg, acc[0][m][c], plus<float>());
                    if (lane == 0) {
                        float v = alpha * tot;
                        if (residual) {
                            v += residual[(size_t)m * out_stride + n];
                        }
                        out[(size_t)m * out_stride + n] = v;
                    }
                }
            }
        });
    });
}
#undef W4_TN_COL
#undef W4_TN_ACC

// Two columns per sub-group, but *without* shrinking the grid: the work-group
// still holds 16 sub-groups (TX = 512, RB = 32 columns), each sub-group owning
// two adjacent columns.  The earlier tiled attempt kept RB = 16 and halved the
// sub-group count, which cost 10x.  Here the per-row activations and the
// per-(row,group) scales/sums are loaded once per (lane, group) and reused for
// both columns (that traffic is what made the row-expanded kernel fall off as
// 1/M); the masks are hoisted per column and all accumulators are literal-index.
#define W4C2_R(m, c0, c1)                                                                               \
    do {                                                                                                \
        const uint4 ev = *reinterpret_cast<const uint4 *>(axe + (size_t)(m) * kh + (size_t)g * 16);      \
        const uint4 ov = *reinterpret_cast<const uint4 *>(axo + (size_t)(m) * kh + (size_t)g * 16);      \
        const float sam = w4_h2f(asa[(size_t)(m) * ng + g]);                                             \
        const float xsm = xs[(size_t)(m) * ng + g];                                                      \
        {                                                                                               \
            int32_t qd = 0;                                                                             \
            qd = dp4a_s8u8(ev.x(), l0a, qd);                                                            \
            qd = dp4a_s8u8(ev.y(), l1a, qd);                                                            \
            qd = dp4a_s8u8(ev.z(), l2a, qd);                                                            \
            qd = dp4a_s8u8(ev.w(), l3a, qd);                                                            \
            qd = dp4a_s8u8(ov.x(), h0a, qd);                                                            \
            qd = dp4a_s8u8(ov.y(), h1a, qd);                                                            \
            qd = dp4a_s8u8(ov.z(), h2a, qd);                                                            \
            qd = dp4a_s8u8(ov.w(), h3a, qd);                                                            \
            acc[(m)][(c0)] += sam * (sca * (float)qd + ofa * xsm);                                       \
        }                                                                                               \
        {                                                                                               \
            int32_t qd = 0;                                                                             \
            qd = dp4a_s8u8(ev.x(), l0b, qd);                                                            \
            qd = dp4a_s8u8(ev.y(), l1b, qd);                                                            \
            qd = dp4a_s8u8(ev.z(), l2b, qd);                                                            \
            qd = dp4a_s8u8(ev.w(), l3b, qd);                                                            \
            qd = dp4a_s8u8(ov.x(), h0b, qd);                                                            \
            qd = dp4a_s8u8(ov.y(), h1b, qd);                                                            \
            qd = dp4a_s8u8(ov.z(), h2b, qd);                                                            \
            qd = dp4a_s8u8(ov.w(), h3b, qd);                                                            \
            acc[(m)][(c1)] += sam * (scb * (float)qd + ofb * xsm);                                       \
        }                                                                                               \
    } while (0)

template <int M, int RB = 32>
static void w4_gemm_c2(queue & q, const uint8_t * vals, const uint16_t * scale, const uint16_t * off,
                       const int8_t * axe, const int8_t * axo, const uint16_t * asa, const float * xs, float * out,
                       int out_stride, const float * residual, float alpha, int K, int N) {
    constexpr int TX = 16 * 32;
    const int ng = K / 32;
    const int kh = K / 2;
    const int nwg = (N + RB - 1) / RB;
    q.submit([&](handler & h) {
        local_accessor<uint16_t, 1> meta((size_t)2 * RB * ng, h);
        h.parallel_for(nd_range<1>((size_t)nwg * TX, TX), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int lid = (int)it.get_local_id(0);
            const int sg = lid / 32;
            const int lane = lid % 32;
            const int nb = (int)it.get_group(0) * RB;
            for (int i = lid; i < RB * ng; i += TX) {
                const int g = i / RB;
                const int r = i % RB;
                const int n = nb + r;
                const bool ok = n < N;
                const size_t src = (size_t)g * N + n;
                meta[i] = ok ? scale[src] : (uint16_t)0;
                meta[(size_t)RB * ng + i] = ok ? off[src] : (uint16_t)0;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const int na = nb + sg * 2;
            const int nc = na + 1;
            float acc[M][2];
#pragma unroll
            for (int m = 0; m < M; m++) {
                acc[m][0] = 0.f;
                acc[m][1] = 0.f;
            }
            const uint8_t * wa = vals + (size_t)na * (size_t)kh;
            const uint8_t * wb = vals + (size_t)nc * (size_t)kh;
            for (int g = lane; g < ng; g += 32) {
                const uint4 va = *reinterpret_cast<const uint4 *>(wa + (size_t)g * 16);
                const uint4 vb = *reinterpret_cast<const uint4 *>(wb + (size_t)g * 16);
                const uint32_t l0a = va.x() & 0x0F0F0F0Fu, l1a = va.y() & 0x0F0F0F0Fu;
                const uint32_t l2a = va.z() & 0x0F0F0F0Fu, l3a = va.w() & 0x0F0F0F0Fu;
                const uint32_t h0a = (va.x() >> 4) & 0x0F0F0F0Fu, h1a = (va.y() >> 4) & 0x0F0F0F0Fu;
                const uint32_t h2a = (va.z() >> 4) & 0x0F0F0F0Fu, h3a = (va.w() >> 4) & 0x0F0F0F0Fu;
                const uint32_t l0b = vb.x() & 0x0F0F0F0Fu, l1b = vb.y() & 0x0F0F0F0Fu;
                const uint32_t l2b = vb.z() & 0x0F0F0F0Fu, l3b = vb.w() & 0x0F0F0F0Fu;
                const uint32_t h0b = (vb.x() >> 4) & 0x0F0F0F0Fu, h1b = (vb.y() >> 4) & 0x0F0F0F0Fu;
                const uint32_t h2b = (vb.z() >> 4) & 0x0F0F0F0Fu, h3b = (vb.w() >> 4) & 0x0F0F0F0Fu;
                const float sca = w4_h2f(meta[(size_t)g * RB + sg * 2]);
                const float ofa = w4_h2f(meta[(size_t)RB * ng + (size_t)g * RB + sg * 2]);
                const float scb = w4_h2f(meta[(size_t)g * RB + sg * 2 + 1]);
                const float ofb = w4_h2f(meta[(size_t)RB * ng + (size_t)g * RB + sg * 2 + 1]);
                if constexpr (M >= 1) {
                    W4C2_R(0, 0, 1);
                }
                if constexpr (M >= 2) {
                    W4C2_R(1, 0, 1);
                }
                if constexpr (M >= 3) {
                    W4C2_R(2, 0, 1);
                }
                if constexpr (M >= 4) {
                    W4C2_R(3, 0, 1);
                }
                if constexpr (M >= 5) {
                    W4C2_R(4, 0, 1);
                }
                if constexpr (M >= 6) {
                    W4C2_R(5, 0, 1);
                }
                if constexpr (M >= 7) {
                    W4C2_R(6, 0, 1);
                }
                if constexpr (M >= 8) {
                    W4C2_R(7, 0, 1);
                }
            }
            const sub_group sgg = it.get_sub_group();
#pragma unroll
            for (int m = 0; m < M; m++) {
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    const int n = na + c;
                    if (n >= N) {
                        continue;
                    }
                    const float tot = reduce_over_group(sgg, acc[m][c], plus<float>());
                    if (lane == 0) {
                        float v = alpha * tot;
                        if (residual) {
                            v += residual[(size_t)m * out_stride + n];
                        }
                        out[(size_t)m * out_stride + n] = v;
                    }
                }
            }
        });
    });
}
#undef W4C2_R

void w4_gemm_launch(queue & q, const uint8_t * vals, const uint16_t * scale, const uint16_t * off,
                    const int8_t * axe, const int8_t * axo, const uint16_t * asa, const float * xs, float * out,
                    int out_stride, const float * residual, float alpha, int M, int K, int N) {
    // PF_W4_GEMM_U: independent accumulator sets (latency hiding).
    static const int u = [] {
        const char * e = getenv("PF_W4_GEMM_U");
        const int v = e ? atoi(e) : 1;
        return (v == 1 || v == 2 || v == 4) ? v : 1;
    }();
#define W4_GEMM_CASE(mm)                                                                                \
    case mm:                                                                                            \
        if (u == 2) {                                                                                   \
            w4_gemm_batched<mm, 2>(q, vals, scale, off, axe, axo, asa, xs, out, out_stride, residual,   \
                                   alpha, K, N);                                                        \
        } else if (u == 4) {                                                                            \
            w4_gemm_batched<mm, 4>(q, vals, scale, off, axe, axo, asa, xs, out, out_stride, residual,   \
                                   alpha, K, N);                                                        \
        } else {                                                                                        \
            w4_gemm_batched<mm, 1>(q, vals, scale, off, axe, axo, asa, xs, out, out_stride, residual,   \
                                   alpha, K, N);                                                        \
        }                                                                                               \
        return;
    // Default 0: use the row-expanded kernel (the column-tiled variants measured
    // 4-10x slower - the sub-group-covering-TN-columns mapping is a poor fit
    // here; PF_W4_GEMM_TN=2|4 selects them for further experiments).
    static const int tn_env = [] {
        const char * e = getenv("PF_W4_GEMM_TN");
        const int v = e ? atoi(e) : 0;
        return (v == 0 || v == 2 || v == 3 || v == 4) ? v : 0;
    }();
#define W4_GEMM_TN_CASE(mm)                                                                             \
    case mm:                                                                                            \
        if (tn_env == 4) {                                                                              \
            w4_gemm_tn<mm, 4>(q, vals, scale, off, axe, axo, asa, xs, out, out_stride, residual, alpha, \
                              K, N);                                                                    \
        } else if (tn_env == 2) {                                                                       \
            w4_gemm_tn<mm, 2>(q, vals, scale, off, axe, axo, asa, xs, out, out_stride, residual, alpha, \
                              K, N);                                                                    \
        } else {                                                                                        \
            w4_gemm_batched<mm, 1>(q, vals, scale, off, axe, axo, asa, xs, out, out_stride, residual,   \
                                   alpha, K, N);                                                        \
        }                                                                                               \
        return;
#define W4_GEMM_C2_CASE(mm)                                                                             \
    case mm:                                                                                            \
        w4_gemm_c2<mm>(q, vals, scale, off, axe, axo, asa, xs, out, out_stride, residual, alpha, K, N);  \
        return;
    static const int c2_env = [] {
        const char * e = getenv("PF_W4_GEMM_TN");
        return e ? atoi(e) : 0;
    }();
    if (c2_env == 3) {
        switch (M) {
            W4_GEMM_C2_CASE(2)
            W4_GEMM_C2_CASE(3)
            W4_GEMM_C2_CASE(4)
            W4_GEMM_C2_CASE(5)
            W4_GEMM_C2_CASE(6)
            W4_GEMM_C2_CASE(7)
            W4_GEMM_C2_CASE(8)
        default:
            break;
        }
    }
#undef W4_GEMM_C2_CASE
    if (tn_env != 0) {
        switch (M) {
            W4_GEMM_TN_CASE(2)
            W4_GEMM_TN_CASE(3)
            W4_GEMM_TN_CASE(4)
            W4_GEMM_TN_CASE(5)
            W4_GEMM_TN_CASE(6)
            W4_GEMM_TN_CASE(7)
            W4_GEMM_TN_CASE(8)
        default:
            break;
        }
    }
#undef W4_GEMM_TN_CASE
    switch (M) {
        W4_GEMM_CASE(2)
        W4_GEMM_CASE(3)
        W4_GEMM_CASE(4)
        W4_GEMM_CASE(5)
        W4_GEMM_CASE(6)
        W4_GEMM_CASE(7)
        W4_GEMM_CASE(8)
    default:
        break;
    }
#undef W4_GEMM_CASE
    w4_gemm_batched<5, 1>(q, vals, scale, off, axe, axo, asa, xs, out, out_stride, residual, alpha, K, N);
}

} // namespace si
