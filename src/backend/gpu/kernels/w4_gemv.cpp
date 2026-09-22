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


// ---- codebook 4-bit (IQ4_XS / IQ4_NL) --------------------------------------
// The weights are 4-bit indices into the 16-entry int8 table `lut` plus a per-32
// f16 scale per row.  Expanding the indices through the table inside the kernel
// keeps the *native* values (nothing is re-quantized) while reading half the
// bytes the int8 conversion needed.  The index plane keeps the native element
// order: element e of a 32-group is the low nibble of byte e for e < 16 and the
// high nibble of byte e - 16 after that, so `raw[j]`'s low nibbles are elements
// 4j..4j+3 and its high nibbles are elements 16+4j..16+4j+3.
void cb4_expand_launch(queue & q, const uint8_t * idx, const int8_t * lut, int8_t * out, int K, int N) {
    const int ng = K / 32;
    if (ng <= 0 || N <= 0) {
        return;
    }
    // one work-item per 32-value group, widened: a 16-byte nibble load in, two
    // 16-byte int8 stores out (the scalar per-byte form made prefill ~1.5x slower)
    q.parallel_for(range<1>((size_t)N * ng), [=](id<1> i) {
        const int n = (int)(i / (size_t)ng);
        const int g = (int)(i % (size_t)ng);
        const uint8_t * src = idx + (size_t)n * (size_t)(K / 2) + (size_t)g * 16;
        const uint4 nv = *reinterpret_cast<const uint4 *>(src);
        const uint32_t raw[4] = {nv.x(), nv.y(), nv.z(), nv.w()};
        uint32_t w[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t v = raw[j];
            // low nibble of byte b is (v >> 8b); w[0..3] are elements 0..15,
            // w[4..7] are elements 16..31 (the native order)
            w[j] = (uint32_t)(uint8_t)lut[(v >> 0) & 0x0F] | ((uint32_t)(uint8_t)lut[(v >> 8) & 0x0F] << 8)
                   | ((uint32_t)(uint8_t)lut[(v >> 16) & 0x0F] << 16)
                   | ((uint32_t)(uint8_t)lut[(v >> 24) & 0x0F] << 24);
            w[4 + j] = (uint32_t)(uint8_t)lut[(v >> 4) & 0x0F] | ((uint32_t)(uint8_t)lut[(v >> 12) & 0x0F] << 8)
                       | ((uint32_t)(uint8_t)lut[(v >> 20) & 0x0F] << 16)
                       | ((uint32_t)(uint8_t)lut[(v >> 28) & 0x0F] << 24);
        }
        int8_t * dst = out + (size_t)n * K + (size_t)g * 32;
        *reinterpret_cast<uint4 *>(dst) = uint4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<uint4 *>(dst + 16) = uint4(w[4], w[5], w[6], w[7]);
    });
}

template <int RB>
static void cb4_gemv_impl(queue & q, const uint8_t * idx, const int8_t * lut, const uint16_t * scale,
                          const int8_t * xq, const uint16_t * asa, const float * xs, float * out, const float * residual,
                          float alpha, int K, int N) {
    const int ng = K / 32;
    constexpr int TX = RB * 32;
    const int nwg = (N + RB - 1) / RB;
    q.submit([&](handler & h) {
        local_accessor<uint16_t, 1> meta((size_t)RB * ng, h);
        local_accessor<int8_t, 1> lt(16, h);
        h.parallel_for(nd_range<1>((size_t)nwg * TX, TX), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int lid = (int)it.get_local_id(0);
            const int sg = lid / 32;
            const int lane = lid % 32;
            const int n0 = (int)it.get_group(0) * RB;
            if (lid < 16) {
                lt[lid] = lut[lid];
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
                    // the low nibble of byte b is (v >> 8b)
                    w[j] = (uint32_t)(uint8_t)lt[(v >> 0) & 0x0F]
                           | ((uint32_t)(uint8_t)lt[(v >> 8) & 0x0F] << 8)
                           | ((uint32_t)(uint8_t)lt[(v >> 16) & 0x0F] << 16)
                           | ((uint32_t)(uint8_t)lt[(v >> 24) & 0x0F] << 24);
                    // the high nibble of byte b is (v >> (8b + 4))
                    w[4 + j] = (uint32_t)(uint8_t)lt[(v >> 4) & 0x0F]
                               | ((uint32_t)(uint8_t)lt[(v >> 12) & 0x0F] << 8)
                               | ((uint32_t)(uint8_t)lt[(v >> 20) & 0x0F] << 16)
                               | ((uint32_t)(uint8_t)lt[(v >> 28) & 0x0F] << 24);
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

void cb4_gemv_launch(queue & q, const uint8_t * idx, const int8_t * lut, const uint16_t * scale, const int8_t * xq,
                     const uint16_t * asa, const float * xs, float * out, const float * residual, float alpha, int K,
                     int N) {
    const int ng = K / 32;
    const auto fits = [&](int rb) { return (size_t)rb * (size_t)ng * sizeof(uint16_t) + 16 <= 48 * 1024; };
    if (fits(kW4RB)) {
        cb4_gemv_impl<kW4RB>(q, idx, lut, scale, xq, asa, xs, out, residual, alpha, K, N);
    } else {
        cb4_gemv_impl<8>(q, idx, lut, scale, xq, asa, xs, out, residual, alpha, K, N);
    }
}

} // namespace si
