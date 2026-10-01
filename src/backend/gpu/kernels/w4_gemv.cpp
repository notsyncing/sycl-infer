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

#include <cstdlib>
#include <stdexcept>

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

// i8_grp_gemv_rows_multi_impl: one launch for a whole call group's int8
// (per-32-group f16 scale) segments, which share the quantized activation and the
// K.  The per-row arithmetic is i8_grp_gemv_impl's, unchanged (same lane->group
// mapping, same sub-group tree reduction), so every row stays bit-identical to
// the M=1 decode GEMV.  See i8_grp_seg in kernels.h for why.
template <int M, int NS>
static void i8_grp_gemv_rows_multi_impl(queue & q, const i8_grp_seg * segs, const int8_t * xq,
                                        const uint16_t * asa, const float * xs, int K, int total_rows) {
    const int ng = K / 32;
    // The descriptors must reach the kernel *by value*: the caller keeps them in
    // host memory (a stack array), and a device dereference of that reads zeros -
    // which looks like a correct-looking, silently empty GEMV.  Copying them into
    // the lambda's closure puts them in the kernel argument blob.
    i8_grp_seg seg_v[NS];
#pragma unroll
    for (int i = 0; i < NS; i++) {
        seg_v[i] = segs[i];
    }
    // One work-group per output row, 32 threads = one sub-group, so the row ->
    // (segment, row) map is a scan over NS (<= 4) captured values and there is no
    // cross-work-group reduction: the sub-group tree below is the only one.
    // Measured no worse than 4 rows per work-group on the narrow shapes.
    constexpr int TX = 32;
    q.submit([&](handler & h) {
        local_accessor<uint16_t, 1> meta((size_t)ng, h);
        h.parallel_for(nd_range<1>((size_t)total_rows * TX, TX), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int lane = (int)it.get_local_id(0);
            const int fr = (int)it.get_group(0);
            int rem = fr;
            int si = 0;
#pragma unroll
            for (int t = 0; t < NS; t++) {
                if (t + 1 < NS && rem >= seg_v[t].n_rows) {
                    rem -= seg_v[t].n_rows;
                    si = t + 1;
                }
            }
            const i8_grp_seg & S = seg_v[si];
            const int n = rem;
            // g-major scale staging for this row (the plane is [K/32][n_rows] of
            // its own segment), then the same group loop as the M=1 GEMV.
            for (int g = lane; g < ng; g += 32) {
                meta[g] = S.wsc[(size_t)g * S.n_rows + n];
            }
            it.barrier(sycl::access::fence_space::local_space);
            const int8_t * wrow = S.w8 + (size_t)n * K;
            float acc[M];
#pragma unroll
            for (int m = 0; m < M; m++) {
                acc[m] = 0.f;
            }
            for (int g = lane; g < ng; g += 32) {
                const uint4 w0 = *reinterpret_cast<const uint4 *>(wrow + (size_t)g * 32);
                const uint4 w1 = *reinterpret_cast<const uint4 *>(wrow + (size_t)g * 32 + 16);
                const float scw = w4_h2f(meta[g]);
#pragma unroll
                for (int m = 0; m < M; m++) {
                    const uint4 x0 = *reinterpret_cast<const uint4 *>(xq + (size_t)m * K + (size_t)g * 32);
                    const uint4 x1 = *reinterpret_cast<const uint4 *>(xq + (size_t)m * K + (size_t)g * 32 + 16);
                    int32_t qd = 0;
                    qd = dp4a_s8u8((int32_t)x0.x(), w0.x() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x0.y(), w0.y() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x0.z(), w0.z() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x0.w(), w0.w() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x1.x(), w1.x() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x1.y(), w1.y() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x1.z(), w1.z() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x1.w(), w1.w() ^ 0x80808080u, qd);
                    qd -= 128 * (int32_t)xs[(size_t)m * ng + g]; // undo the XOR bias
                    acc[m] += w4_h2f(asa[(size_t)m * ng + g]) * scw * (float)qd;
                }
            }
            const sub_group sgg = it.get_sub_group();
#pragma unroll
            for (int m = 0; m < M; m++) {
                const float tot = reduce_over_group(sgg, acc[m], plus<float>());
                if (lane == 0) {
                    float v = S.alpha * tot;
                    if (S.residual) {
                        v += S.residual[(size_t)m * S.out_stride + n];
                    }
                    S.out[(size_t)m * S.out_stride + n] = v;
                }
            }
        });
    });
}

#define I8GM(mm, ns) i8_grp_gemv_rows_multi_impl<mm, ns>(q, segs, xq, asa, xs, K, total_rows)

template <int M>
static void i8_grp_gemv_rows_multi_pick(queue & q, const i8_grp_seg * segs, int n_segs, const int8_t * xq,
                                         const uint16_t * asa, const float * xs, int K, int total_rows) {
    switch (n_segs) {
    case 1: I8GM(M, 1); return;
    case 2: I8GM(M, 2); return;
    case 3: I8GM(M, 3); return;
    case 4: I8GM(M, 4); return;
    default: break;
    }
    throw std::runtime_error("i8_grp_gemv_rows_multi_launch: too many segments");
}

#undef I8GM

void i8_grp_gemv_rows_multi_launch(queue & q, const i8_grp_seg * segs, int n_segs, int total_rows,
                                    const int8_t * xq, const uint16_t * asa, const float * xs, int M, int K) {
#define I8GR(mm) i8_grp_gemv_rows_multi_pick<mm>(q, segs, n_segs, xq, asa, xs, K, total_rows)
    switch (M) {
    case 1: I8GR(1); return;
    case 2: I8GR(2); return;
    case 3: I8GR(3); return;
    case 4: I8GR(4); return;
    case 5: I8GR(5); return;
    case 6: I8GR(6); return;
    case 7: I8GR(7); return;
    case 8: I8GR(8); return;
    case 9: I8GR(9); return;
    case 10: I8GR(10); return;
    case 11: I8GR(11); return;
    case 12: I8GR(12); return;
    case 13: I8GR(13); return;
    default: break;
    }
#undef I8GR
    throw std::runtime_error("i8_grp_gemv_rows_multi_launch: unsupported M");
}

void i8_grp_gemv_rows_launch(queue & q, const int8_t * w8, const uint16_t * wsc, const int8_t * xq,
                             const uint16_t * asa, const float * xs, float * out, int out_stride,
                             const float * residual, float alpha, int M, int K, int N) {
    const i8_grp_seg s = {w8, wsc, N, out, out_stride, alpha, residual};
    i8_grp_gemv_rows_multi_launch(q, &s, 1, N, xq, asa, xs, M, K);
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

// ---------------------------------------------------------------------------
// Batched native-width GEMM for the MTP speculative verify (M = 2..13).
//
// The M=1 GEMVs above give one sub-group per output column, so each lane reads
// the whole activation row once per column: the activation traffic is ~N*K*M
// bytes, N/RB times the weight stream, which is why the row-expanded
// w4_gemm_batched falls off as 1/M (43 GB/s at M=5).  This kernel keeps the same
// lane = group / sub-group = column mapping but stages the K-tile's activations
// in SLM once per workgroup, so the weight stream is read exactly once for all
// M rows and the activation re-reads stay inside SLM.
//
// Measured on the 27B shapes (K=5120, N=17408, M=7, A770): the oneDNN
// grouped-scale matmul this replaces runs the int8 weight path at ~120 GB/s
// end to end (the 27B verify cost 196 ms); this kernel reaches ~106 GB/s (u4),
// 131 (k5), 115 (cb4) and 241 (int8) per tensor.  It is *instruction-issue*
// bound, not bandwidth bound: dp4a consumes 4 B of each operand per
// instruction, so the floor is ~0.5 instructions/MAC and the int8 variant (2x
// the weight bytes for the same MAC count) is the only one that reaches the
// card's ~300 GB/s read path.  The win over oneDNN is therefore mostly the
// fused scale/offset/residual epilogue and the removal of oneDNN's per-call
// execute + scratch traffic, not raw bandwidth.
//
// Details that matter:
//   * SG=16: a SIMD32 float costs 4 GRFs and the M*C accumulators must all stay
//     live; SIMD32 spilled the whole accumulator set (18-23 KB/thread) and ran
//     4-5x slower.  SG=16 halves the per-variable GRF cost.
//   * C=1: one column per sub-group.  C>1 reuses one activation load across
//     columns, but the extra accumulators cost more than the saved loads.
//   * the activation is read as two uint4 (16-byte) loads rather than eight
//     4-byte ones; the slot is padded to APAD=48 bytes so both are 16-byte
//     aligned (the 48-byte stride gives a 2-way SLM bank conflict, which is
//     cheaper than 8 extra issue slots).
//   * FMT: 0 = u4 (Q4_K), 1 = k5 (Q5_K), 2 = cb4 (IQ4_XS/NL), 3 = grouped int8.
//     u4/k5 read the even/odd activation split (axe/axo), cb4/int8 the grouped
//     activations (axg); all consume asa (per-group act scale) and xs (signed
//     group sum), which act_quant_grp_launch already produces for any M.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Batched native-width GEMM for the MTP verify (M = 2..13).  See the header
// comment above for the contract.  Template knobs for tuning:
//   C     output columns owned by one sub-group (activation reuse across them)
//   KT    K-tile (weight groups) staged per barrier
//   SG    sub-group size (16 keeps the accumulator set in registers)
//   ORDER 0 = c-outer/m-inner (one activation reload per column),
//         1 = m-outer/c-inner (activation loaded once per row, C columns reuse)
//   TX    work-group size
// All orders accumulate each (m,n) over groups in the same ascending order and
// reduce over the sub-group identically, so results are bit-identical.
// ---------------------------------------------------------------------------
// Split a packed u4 dword into its low/high nibble planes.  OPQ=1 repacks
// bit 3 of each nibble into bit 4 (an OR of shifted masks) instead of a plain
// mask; the value is unchanged, but the shape mirrors the k5 path that IGC
// recognises as a hardware dp4a.
template <int OPQ>
static inline void u4_split(uint32_t v, uint32_t & lo, uint32_t & hi) {
    if constexpr (OPQ == 1) {
        lo = (v & 0x07070707u) | ((v << 1) & 0x10101010u);
        hi = ((v >> 4) & 0x07070707u) | ((v >> 3) & 0x10101010u);
    } else {
        lo = v & 0x0F0F0F0Fu;
        hi = (v >> 4) & 0x0F0F0F0Fu;
    }
}

template <int FMT, int M, int C, int KT, int SG, int ORDER, int TX, int OPQ = 0, int TREE = 0, int ABL = 0,
          int VECST = 0, int VECA = 0, int NOX = 0, int DIRECT = 0>
static void nat_gemm_impl(queue & q, const uint8_t * w0, const uint8_t * w1, const int8_t * w8,
                          const uint16_t * scale, const uint16_t * off, const uint16_t * lut16,
                          const uint32_t * bit_lut, const int8_t * axe, const int8_t * axo, const int8_t * axg,
                          const uint16_t * asa, const float * xs, float * out, int out_stride, const float * residual,
                          float alpha, int K, int N) {
    constexpr int NSG = TX / SG;
    constexpr int RB = NSG * C;
    // APAD: 16-byte-aligned slot per staged activation group, so a group's 32
    // bytes read back as two uint4.  32 (not 48) is the better trade here: it
    // shrinks act_s (the largest SLM array), and measured at M=7 on the A770
    // that wins ~1% for u4 and ~4-7% for k5/cb4 (occupancy, not fewer bank
    // conflicts).  Exception: at M>=12 the u4 kernel's larger accumulator set
    // wants the 48-byte stride back (u4 M=12: 0.572 -> 0.549 ms); u4 is the
    // only format that regresses there.
    constexpr int APAD = (FMT == 0 && M >= 12) ? 48 : 32;
    const int ng = K / 32;
    const int kh = K / 2;
    const int nwg = (N + RB - 1) / RB;
    q.submit([&](handler & h) {
        local_accessor<int8_t, 1> act_s((size_t)M * KT * APAD, h);
        local_accessor<uint16_t, 1> asa_s((size_t)M * KT, h);
        local_accessor<float, 1> xs_s((size_t)M * KT, h);
        local_accessor<uint16_t, 1> sc_s((size_t)KT * RB, h);
        local_accessor<uint16_t, 1> of_s((size_t)KT * RB, h);
        local_accessor<uint16_t, 1> lt16_s(256, h);
        local_accessor<uint32_t, 1> lt_s(16, h);
        h.parallel_for(nd_range<1>((size_t)nwg * TX, TX), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int lid = (int)it.get_local_id(0);
            const int sg = lid / SG;
            const int lane = lid % SG;
            const int n0 = (int)it.get_group(0) * RB;
            int8_t * act_p = act_s.get_multi_ptr<sycl::access::decorated::no>().get();
            uint16_t * asa_p = asa_s.get_multi_ptr<sycl::access::decorated::no>().get();
            float * xs_p = xs_s.get_multi_ptr<sycl::access::decorated::no>().get();
            uint16_t * sc_p = sc_s.get_multi_ptr<sycl::access::decorated::no>().get();
            uint16_t * of_p = of_s.get_multi_ptr<sycl::access::decorated::no>().get();
            uint16_t * lt16_p = lt16_s.get_multi_ptr<sycl::access::decorated::no>().get();
            uint32_t * lt_p = lt_s.get_multi_ptr<sycl::access::decorated::no>().get();
            if constexpr (FMT == 1) {
                for (int i = lid; i < 16; i += TX) {
                    lt_p[i] = bit_lut[i];
                }
            } else if constexpr (FMT == 2) {
                for (int i = lid; i < 256; i += TX) {
                    lt16_p[i] = lut16[i];
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            float acc[M * C];
#pragma unroll
            for (int i = 0; i < M * C; i++) {
                acc[i] = 0.f;
            }
            for (int g0 = 0; g0 < ng; g0 += KT) {
                const int kt = (ng - g0 < KT) ? (ng - g0) : KT;
                for (int i = lid; i < kt * M; i += TX) {
                    const int m = i / kt;
                    const int gt = i - m * kt;
                    int8_t * dst = act_p + ((size_t)m * KT + gt) * APAD;
                    if constexpr (DIRECT == 1) {
                        asa_p[i] = asa[(size_t)m * ng + (g0 + gt)];
                        xs_p[i] = xs[(size_t)m * ng + (g0 + gt)];
                        continue;
                    }
                    if constexpr (FMT <= 1) {
                        const int8_t * e = axe + (size_t)m * kh + (size_t)(g0 + gt) * 16;
                        const int8_t * o = axo + (size_t)m * kh + (size_t)(g0 + gt) * 16;
                        if constexpr (VECA == 1) {
                            *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(e);
                            *reinterpret_cast<uint4 *>(dst + 16) = *reinterpret_cast<const uint4 *>(o);
                        } else {
                            for (int j = 0; j < 16; j++) {
                                dst[j] = e[j];
                                dst[16 + j] = o[j];
                            }
                        }
                    } else {
                        const int8_t * xg = axg + (size_t)m * K + (size_t)(g0 + gt) * 32;
                        if constexpr (VECA == 1) {
                            *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(xg);
                            *reinterpret_cast<uint4 *>(dst + 16) = *reinterpret_cast<const uint4 *>(xg + 16);
                        } else {
                            for (int j = 0; j < 32; j++) {
                                dst[j] = xg[j];
                            }
                        }
                    }
                    asa_p[i] = asa[(size_t)m * ng + (g0 + gt)];
                    xs_p[i] = xs[(size_t)m * ng + (g0 + gt)];
                }
                if constexpr (VECST == 1) {
                    for (int i = lid * 8; i < kt * RB; i += TX * 8) {
                        const int gt = i / RB;
                        const int r = i - gt * RB;
                        const int n = n0 + r;
                        const size_t s = (size_t)(g0 + gt) * N + n;
                        if (n + 8 <= N) {
                            *reinterpret_cast<uint4 *>(&sc_p[i]) =
                                *reinterpret_cast<const uint4 *>(&scale[s]);
                            if constexpr (FMT <= 1) {
                                *reinterpret_cast<uint4 *>(&of_p[i]) =
                                    *reinterpret_cast<const uint4 *>(&off[s]);
                            }
                        } else {
                            for (int j = 0; j < 8 && i + j < kt * RB; j++) {
                                const int nn = n + j;
                                sc_p[i + j] = nn < N ? scale[s + j] : (uint16_t)0;
                                if constexpr (FMT <= 1) {
                                    of_p[i + j] = nn < N ? off[s + j] : (uint16_t)0;
                                }
                            }
                        }
                    }
                } else {
                    for (int i = lid; i < kt * RB; i += TX) {
                        const int gt = i / RB;
                        const int r = i - gt * RB;
                        const int n = n0 + r;
                        const bool ok = n < N;
                        const size_t s = (size_t)(g0 + gt) * N + n;
                        sc_p[i] = ok ? scale[s] : (uint16_t)0;
                        if constexpr (FMT <= 1) {
                            of_p[i] = ok ? off[s] : (uint16_t)0;
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if constexpr (ORDER == 1) {
                    // Group-loop body; the per-format unroll is chosen below.
                    auto nat_gt = [&](int gt) {
                        const int g = g0 + gt;
                        const int8_t * ag = act_p + (size_t)gt * APAD;
                        uint32_t wl[C][4];
                        uint32_t wh[C][4];
                        float scw[C];
                        float ofw[C];
#pragma unroll
                        for (int c = 0; c < C; c++) {
                            const int n = n0 + sg * C + c;
                            const bool ok = n < N;
                            const int nn = ok ? n : 0;
                            scw[c] = w4_h2f(sc_p[gt * RB + sg * C + c]);
                            ofw[c] = 0.f;
                            if constexpr (FMT <= 1) {
                                ofw[c] = w4_h2f(of_p[gt * RB + sg * C + c]);
                            }
                            if constexpr (FMT <= 1) {
                                const uint4 wv = ok ? *reinterpret_cast<const uint4 *>(w0 + (size_t)nn * kh
                                                                                       + (size_t)g * 16)
                                                    : uint4(0, 0, 0, 0);
                                u4_split<FMT == 0 ? OPQ : 0>(wv.x(), wl[c][0], wh[c][0]);
                                u4_split<FMT == 0 ? OPQ : 0>(wv.y(), wl[c][1], wh[c][1]);
                                u4_split<FMT == 0 ? OPQ : 0>(wv.z(), wl[c][2], wh[c][2]);
                                u4_split<FMT == 0 ? OPQ : 0>(wv.w(), wl[c][3], wh[c][3]);
                                if constexpr (FMT == 1) {
                                    const uint32_t hb =
                                        ok ? *reinterpret_cast<const uint32_t *>(w1 + (size_t)nn * (K / 8)
                                                                                  + (size_t)g * 4)
                                           : 0u;
                                    wl[c][0] |= lt_p[(hb >> 0) & 0xF] << 4;
                                    wl[c][1] |= lt_p[(hb >> 4) & 0xF] << 4;
                                    wl[c][2] |= lt_p[(hb >> 8) & 0xF] << 4;
                                    wl[c][3] |= lt_p[(hb >> 12) & 0xF] << 4;
                                    wh[c][0] |= lt_p[(hb >> 16) & 0xF] << 4;
                                    wh[c][1] |= lt_p[(hb >> 20) & 0xF] << 4;
                                    wh[c][2] |= lt_p[(hb >> 24) & 0xF] << 4;
                                    wh[c][3] |= lt_p[(hb >> 28) & 0xF] << 4;
                                }
                            } else if constexpr (FMT == 2) {
                                const uint4 nv = ok ? *reinterpret_cast<const uint4 *>(w0 + (size_t)nn * kh
                                                                                       + (size_t)g * 16)
                                                    : uint4(0, 0, 0, 0);
                                const uint32_t rw[4] = {nv.x(), nv.y(), nv.z(), nv.w()};
                                uint32_t tmp[8];
#pragma unroll
                                for (int j = 0; j < 4; j++) {
                                    const uint32_t v = rw[j];
                                    tmp[2 * j] = (uint32_t)lt16_p[(v >> 0) & 0xFF]
                                                 | ((uint32_t)lt16_p[(v >> 8) & 0xFF] << 16);
                                    tmp[2 * j + 1] = (uint32_t)lt16_p[(v >> 16) & 0xFF]
                                                     | ((uint32_t)lt16_p[(v >> 24) & 0xFF] << 16);
                                }
                                wl[c][0] = tmp[0];
                                wl[c][1] = tmp[1];
                                wl[c][2] = tmp[2];
                                wl[c][3] = tmp[3];
                                wh[c][0] = tmp[4];
                                wh[c][1] = tmp[5];
                                wh[c][2] = tmp[6];
                                wh[c][3] = tmp[7];
                            } else {
                                const uint4 a0 = ok ? *reinterpret_cast<const uint4 *>(w8 + (size_t)nn * K
                                                                                       + (size_t)g * 32)
                                                    : uint4(0, 0, 0, 0);
                                const uint4 a1 = ok ? *reinterpret_cast<const uint4 *>(w8 + (size_t)nn * K
                                                                                       + (size_t)g * 32 + 16)
                                                    : uint4(0, 0, 0, 0);
                                wl[c][0] = a0.x();
                                wl[c][1] = a0.y();
                                wl[c][2] = a0.z();
                                wl[c][3] = a0.w();
                                wh[c][0] = a1.x();
                                wh[c][1] = a1.y();
                                wh[c][2] = a1.z();
                                wh[c][3] = a1.w();
                            }
#pragma unroll
                            for (int j = 0; j < 4; j++) {
                                if constexpr (FMT >= 2 || NOX == 0) {
                                    wl[c][j] ^= 0x80808080u;
                                    wh[c][j] ^= 0x80808080u;
                                }
                            }
                        }
#pragma unroll
                        for (int m = 0; m < M; m++) {
                            const int8_t * amp = ag + (size_t)m * KT * APAD;
                            uint4 xe, xo;
                            if constexpr (ABL == 2) {
                                xe = uint4(1, 2, 3, 4);
                                xo = uint4(5, 6, 7, 8);
                            } else if constexpr (DIRECT == 1) {
                                if constexpr (FMT <= 1) {
                                    const int8_t * e0 = axe + (size_t)m * kh + (size_t)g * 16;
                                    const int8_t * o0 = axo + (size_t)m * kh + (size_t)g * 16;
                                    xe = *reinterpret_cast<const uint4 *>(e0);
                                    xo = *reinterpret_cast<const uint4 *>(o0);
                                } else {
                                    const int8_t * xg0 = axg + (size_t)m * K + (size_t)g * 32;
                                    xe = *reinterpret_cast<const uint4 *>(xg0);
                                    xo = *reinterpret_cast<const uint4 *>(xg0 + 16);
                                }
                            } else {
                                xe = *reinterpret_cast<const uint4 *>(amp);
                                xo = *reinterpret_cast<const uint4 *>(amp + 16);
                            }
                            const float sa = w4_h2f(asa_p[m * KT + gt]);
                            const float xsm = xs_p[m * KT + gt];
                            const int32_t xcorr = -128 * (int32_t)xsm;
#pragma unroll
                            for (int c = 0; c < C; c++) {
                                uint32_t a0 = wl[c][0], a1 = wl[c][1], a2 = wl[c][2], a3 = wl[c][3];
                                uint32_t b0 = wh[c][0], b1 = wh[c][1], b2 = wh[c][2], b3 = wh[c][3];
                                if constexpr (ABL == 3) {
                                    a0 = a1 = a2 = a3 = b0 = b1 = b2 = b3 = 0x80808080u;
                                }
                                int32_t qd;
                                if constexpr (TREE == 1) {
                                    int32_t q = 0;
                                    q = dp4a_s8u8((int32_t)xe.x(), a0, q);
                                    q = dp4a_s8u8((int32_t)xe.y(), a1, q);
                                    q = dp4a_s8u8((int32_t)xe.z(), a2, q);
                                    q = dp4a_s8u8((int32_t)xe.w(), a3, q);
                                    q = dp4a_s8u8((int32_t)xo.x(), b0, q);
                                    q = dp4a_s8u8((int32_t)xo.y(), b1, q);
                                    q = dp4a_s8u8((int32_t)xo.z(), b2, q);
                                    q = dp4a_s8u8((int32_t)xo.w(), b3, q);
                                    qd = (FMT >= 2 || NOX == 0) ? q + xcorr : q;
                                } else {
                                    int32_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;
                                    q0 = dp4a_s8u8((int32_t)xe.x(), a0, q0);
                                    q0 = dp4a_s8u8((int32_t)xe.y(), a1, q0);
                                    q1 = dp4a_s8u8((int32_t)xe.z(), a2, q1);
                                    q1 = dp4a_s8u8((int32_t)xe.w(), a3, q1);
                                    q2 = dp4a_s8u8((int32_t)xo.x(), b0, q2);
                                    q2 = dp4a_s8u8((int32_t)xo.y(), b1, q2);
                                    q3 = dp4a_s8u8((int32_t)xo.z(), b2, q3);
                                    q3 = dp4a_s8u8((int32_t)xo.w(), b3, q3);
                                    qd = (FMT >= 2 || NOX == 0) ? ((q0 + q1) + (q2 + q3)) + xcorr
                                                                 : ((q0 + q1) + (q2 + q3));
                                }
                                if constexpr (ABL == 1) {
                                    acc[m * C + c] += (float)qd;
                                } else {
                                    acc[m * C + c] += sa * (scw[c] * (float)qd + ofw[c] * xsm);
                                }
                            }
                        }
                    };
                    // Per-format group-loop unroll.  u4 keeps two iterations in
                    // flight: the second group's weight loads issue while the
                    // first group's dp4a chain runs, covering the load latency
                    // (u4's decode is only masks, so it has registers to spare).
                    // k5/cb4/i8 keep more decoded words live, and unrolling
                    // them once measured better (M=7: k5 0.248 -> 0.237, cb4
                    // 0.276 -> 0.259 ms) because the lower register pressure
                    // raises occupancy.
                    if constexpr (FMT == 0) {
#pragma unroll 2
                        for (int gt = lane; gt < kt; gt += SG) {
                            nat_gt(gt);
                        }
                    } else {
#pragma unroll 1
                        for (int gt = lane; gt < kt; gt += SG) {
                            nat_gt(gt);
                        }
                    }
                } else {
                    for (int gt = lane; gt < kt; gt += SG) {
                        const int g = g0 + gt;
                        const int8_t * ag = act_p + (size_t)gt * APAD;
#pragma unroll
                        for (int c = 0; c < C; c++) {
                            const int n = n0 + sg * C + c;
                            const bool ok = n < N;
                            const int nn = ok ? n : 0;
                            const float scw = w4_h2f(sc_p[gt * RB + sg * C + c]);
                            float ofw = 0.f;
                            if constexpr (FMT <= 1) {
                                ofw = w4_h2f(of_p[gt * RB + sg * C + c]);
                            }
                            uint32_t l0, l1, l2, l3, h0, h1, h2, h3;
                            if constexpr (FMT <= 1) {
                                const uint4 wv = ok ? *reinterpret_cast<const uint4 *>(w0 + (size_t)nn * kh
                                                                                       + (size_t)g * 16)
                                                    : uint4(0, 0, 0, 0);
                                u4_split<FMT == 0 ? OPQ : 0>(wv.x(), l0, h0);
                                u4_split<FMT == 0 ? OPQ : 0>(wv.y(), l1, h1);
                                u4_split<FMT == 0 ? OPQ : 0>(wv.z(), l2, h2);
                                u4_split<FMT == 0 ? OPQ : 0>(wv.w(), l3, h3);
                                if constexpr (FMT == 1) {
                                    const uint32_t hb =
                                        ok ? *reinterpret_cast<const uint32_t *>(w1 + (size_t)nn * (K / 8)
                                                                                  + (size_t)g * 4)
                                           : 0u;
                                    l0 |= lt_p[(hb >> 0) & 0xF] << 4;
                                    l1 |= lt_p[(hb >> 4) & 0xF] << 4;
                                    l2 |= lt_p[(hb >> 8) & 0xF] << 4;
                                    l3 |= lt_p[(hb >> 12) & 0xF] << 4;
                                    h0 |= lt_p[(hb >> 16) & 0xF] << 4;
                                    h1 |= lt_p[(hb >> 20) & 0xF] << 4;
                                    h2 |= lt_p[(hb >> 24) & 0xF] << 4;
                                    h3 |= lt_p[(hb >> 28) & 0xF] << 4;
                                }
                            } else if constexpr (FMT == 2) {
                                const uint4 nv = ok ? *reinterpret_cast<const uint4 *>(w0 + (size_t)nn * kh
                                                                                       + (size_t)g * 16)
                                                    : uint4(0, 0, 0, 0);
                                const uint32_t rw[4] = {nv.x(), nv.y(), nv.z(), nv.w()};
                                uint32_t tmp[8];
#pragma unroll
                                for (int j = 0; j < 4; j++) {
                                    const uint32_t v = rw[j];
                                    tmp[2 * j] = (uint32_t)lt16_p[(v >> 0) & 0xFF]
                                                 | ((uint32_t)lt16_p[(v >> 8) & 0xFF] << 16);
                                    tmp[2 * j + 1] = (uint32_t)lt16_p[(v >> 16) & 0xFF]
                                                     | ((uint32_t)lt16_p[(v >> 24) & 0xFF] << 16);
                                }
                                l0 = tmp[0];
                                l1 = tmp[1];
                                l2 = tmp[2];
                                l3 = tmp[3];
                                h0 = tmp[4];
                                h1 = tmp[5];
                                h2 = tmp[6];
                                h3 = tmp[7];
                            } else {
                                const uint4 a0 = ok ? *reinterpret_cast<const uint4 *>(w8 + (size_t)nn * K
                                                                                       + (size_t)g * 32)
                                                    : uint4(0, 0, 0, 0);
                                const uint4 a1 = ok ? *reinterpret_cast<const uint4 *>(w8 + (size_t)nn * K
                                                                                       + (size_t)g * 32 + 16)
                                                    : uint4(0, 0, 0, 0);
                                l0 = a0.x();
                                l1 = a0.y();
                                l2 = a0.z();
                                l3 = a0.w();
                                h0 = a1.x();
                                h1 = a1.y();
                                h2 = a1.z();
                                h3 = a1.w();
                            }
                            l0 ^= 0x80808080u; l1 ^= 0x80808080u;
                            l2 ^= 0x80808080u; l3 ^= 0x80808080u;
                            h0 ^= 0x80808080u; h1 ^= 0x80808080u;
                            h2 ^= 0x80808080u; h3 ^= 0x80808080u;
                            const int8_t * ap = ag;
                            const uint16_t * sap = asa_p + gt;
                            const float * xsp = xs_p + gt;
#pragma unroll
                            for (int m = 0; m < M; m++) {
                                const uint4 xe = *reinterpret_cast<const uint4 *>(ap);
                                const uint4 xo = *reinterpret_cast<const uint4 *>(ap + 16);
                                int32_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;
                                q0 = dp4a_s8u8((int32_t)xe.x(), l0, q0);
                                q0 = dp4a_s8u8((int32_t)xe.y(), l1, q0);
                                q1 = dp4a_s8u8((int32_t)xe.z(), l2, q1);
                                q1 = dp4a_s8u8((int32_t)xe.w(), l3, q1);
                                q2 = dp4a_s8u8((int32_t)xo.x(), h0, q2);
                                q2 = dp4a_s8u8((int32_t)xo.y(), h1, q2);
                                q3 = dp4a_s8u8((int32_t)xo.z(), h2, q3);
                                q3 = dp4a_s8u8((int32_t)xo.w(), h3, q3);
                                const int32_t qd = ((q0 + q1) + (q2 + q3)) - 128 * (int32_t)(*xsp);
                                acc[m * C + c] += w4_h2f(*sap) * (scw * (float)qd + ofw * (*xsp));
                                ap += KT * APAD;
                                sap += KT;
                                xsp += KT;
                            }
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            const sub_group sgg = it.get_sub_group();
#pragma unroll
            for (int m = 0; m < M; m++) {
#pragma unroll
                for (int c = 0; c < C; c++) {
                    const int n = n0 + sg * C + c;
                    if (n >= N) {
                        continue;
                    }
                    const float tot = reduce_over_group(sgg, acc[m * C + c], plus<float>());
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

template <int FMT>
static void nat_gemm_pick(queue & q, const uint8_t * w0, const uint8_t * w1, const int8_t * w8,
                          const uint16_t * scale, const uint16_t * off, const uint16_t * lut16,
                          const uint32_t * bit_lut, const int8_t * axe, const int8_t * axo, const int8_t * axg,
                          const uint16_t * asa, const float * xs, float * out, int out_stride, const float * residual,
                          float alpha, int M, int K, int N) {
    // Tuned on the 27B MTP verify shape (M=7, K=5120, N=17408, A770).
    //   C=2      two output columns per sub-group, so an activation load is
    //            amortised over two columns
    //   ORDER=1  m-outer / c-inner: the activation group for (row m, group g)
    //            is loaded once and reused across the C columns.  This is the
    //            change that matters: it also lets IGC form the hardware dp4a
    //            for the u4 nibble decode (the c-outer form scalarises it).
    //   SG=8     SIMD8 sub-groups (a SIMD16 accumulator set costs 2 GRF/var)
    //   TX=128   NSG=16, RB=32
    //   TREE=1   one dp4a accumulator chain per column (no 4-way combine tree)
    //   VECST=1  scale/offset staged with 16-byte loads
    //   VECA=1   activation staged with 16-byte loads (K%32==0 guarantees the
    //            16-byte alignment of every row/group offset)
    //   NOX=0    keep the original u4/k5 arithmetic (no sign-flip, no in-loop
    //            correction); results are bit-identical to the old kernel
    // PF_NAT=0 (in nat_gemm_launch) still disables the whole path.
#ifndef NAT_TX
#define NAT_TX 128
#endif
#ifndef NAT_C
#define NAT_C 2
#endif
#ifndef NAT_SG
#define NAT_SG 8
#endif
#define NAT_M(mm)                                                                                          \
    nat_gemm_impl<FMT, mm, NAT_C, 32, NAT_SG, 1, NAT_TX, 0, 1, 0, 1, 1, 0>(q, w0, w1, w8, scale, off,    \
                                                                        lut16, bit_lut, axe, axo, axg,    \
                                                                        asa, xs, out, out_stride,         \
                                                                        residual, alpha, K, N)
    switch (M) {
    case 2: NAT_M(2); return;
    case 3: NAT_M(3); return;
    case 4: NAT_M(4); return;
    case 5: NAT_M(5); return;
    case 6: NAT_M(6); return;
    case 7: NAT_M(7); return;
    case 8: NAT_M(8); return;
    case 9: NAT_M(9); return;
    case 10: NAT_M(10); return;
    case 11: NAT_M(11); return;
    case 12: NAT_M(12); return;
    case 13: NAT_M(13); return;
    default: return;
    }
#undef NAT_M
}

bool nat_gemm_launch(queue & q, int fmt, const void * w0, const void * w1, const int8_t * w8,
                     const uint16_t * scale, const uint16_t * off, const uint16_t * lut16, const uint32_t * bit_lut,
                     const int8_t * axe, const int8_t * axo, const int8_t * axg, const uint16_t * asa,
                     const float * xs, float * out, int out_stride, const float * residual, float alpha, int M, int K,
                     int N) {
    if (M < 2 || M > 13 || (K % 32) != 0) {
        return false;
    }
    // PF_NAT=0 disables the batched native GEMM everywhere (fall back to the
    // oneDNN grouped-scale matmul + epilogue), for A/B and as an escape hatch.
    static const bool nat_on = [] {
        const char * e = getenv("PF_NAT");
        return e ? atoi(e) != 0 : true;
    }();
    if (!nat_on) {
        return false;
    }
    const uint8_t * b0 = (const uint8_t *)w0;
    const uint8_t * b1 = (const uint8_t *)w1;
    switch (fmt) {
    case 0:
        nat_gemm_pick<0>(q, b0, b1, w8, scale, off, lut16, bit_lut, axe, axo, axg, asa, xs, out, out_stride,
                         residual, alpha, M, K, N);
        return true;
    case 1:
        nat_gemm_pick<1>(q, b0, b1, w8, scale, off, lut16, bit_lut, axe, axo, axg, asa, xs, out, out_stride,
                         residual, alpha, M, K, N);
        return true;
    case 2:
        nat_gemm_pick<2>(q, b0, b1, w8, scale, off, lut16, bit_lut, axe, axo, axg, asa, xs, out, out_stride,
                         residual, alpha, M, K, N);
        return true;
    case 3:
        nat_gemm_pick<3>(q, b0, b1, w8, scale, off, lut16, bit_lut, axe, axo, axg, asa, xs, out, out_stride,
                         residual, alpha, M, K, N);
        return true;
    default:
        return false;
    }
}

} // namespace si
