// CPU integer (int8) GEMV/GEMM from the original GGUF K-quant blocks.
//
// The SIn/DP4A path (dp4a.cpp) packs the weights first and expands the packed
// bits per call; this path instead extracts the 4/5/6-bit values straight from
// the GGUF block and does the dot in the integer domain against the quantized
// activation produced by cpu_xq:
//
//   sum_k w_k * x_k = sx * ( d*sc * sum q*qx  -  dmin*m * sum qx )   (Q4/Q5)
//                   = sx * d*sc * ( sum q*qx - 32 * sum qx )          (Q6)
//
// so there is no int->float conversion and one maddubs handles 32 values.
// Q8_0 and unknown formats fall back to the fused fp32 kernel, and so does the
// scalar ISA build (PF_CPU_ISA=scalar).
#include <cstdint>

#include <immintrin.h>

#include "common.h"

namespace si {

namespace {

__attribute__((target("avx2,fma"))) inline int32_t hsum_i32_256(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}
__attribute__((target("avx2,fma"))) inline int32_t hsum_i32_128(__m128i v) {
    v = _mm_hadd_epi32(v, v);
    v = _mm_hadd_epi32(v, v);
    return _mm_cvtsi128_si32(v);
}

// Q4_K: 8 groups of 32 unsigned 4-bit values; y = d*sc*q - dmin*m
__attribute__((target("avx2,fma"))) float q4k_i8_block(const block_q4_K * blk, const int8_t * x8, const float * xmeta,
                                                       const int32_t * xsumq, int TB, int t, int gbase) {
    const float d = ggml_half_to_float(blk->d);
    const float dm = ggml_half_to_float(blk->dmin);
    const uint8_t * qs = blk->qs;
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    float acc = 0.0f;
    for (int g = 0; g < 8; g++) {
        uint8_t sc, mm;
        get_scale_min_k4(g, blk->scales, &sc, &mm);
        const size_t ai = (size_t)(gbase + g) * TB + t;
        const __m256i bytes = _mm256_loadu_si256((const __m256i *)(qs + (g / 2) * 32));
        const __m256i q = (g & 1) ? _mm256_and_si256(_mm256_srli_epi16(bytes, 4), m0f)
                                  : _mm256_and_si256(bytes, m0f);
        const __m256i prod = _mm256_maddubs_epi16(q, _mm256_loadu_si256((const __m256i *)(x8 + ai * 32)));
        const int32_t wd = hsum_i32_256(_mm256_madd_epi16(prod, ones));
        const int32_t xsum = xsumq[ai * 2] + xsumq[ai * 2 + 1];
        acc += xmeta[ai * 2] * (d * sc * (float)wd - dm * mm * (float)xsum);
    }
    return acc;
}

// Q5_K: like Q4_K plus the 5th bit; bit g of qh[l] belongs to group g
__attribute__((target("avx2,fma"))) float q5k_i8_block(const block_q5_K * blk, const int8_t * x8, const float * xmeta,
                                                       const int32_t * xsumq, int TB, int t, int gbase) {
    const float d = ggml_half_to_float(blk->d);
    const float dm = ggml_half_to_float(blk->dmin);
    const uint8_t * qs = blk->qs;
    const __m256i qhv = _mm256_loadu_si256((const __m256i *)blk->qh);
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i z = _mm256_setzero_si256();
    float acc = 0.0f;
    for (int g = 0; g < 8; g++) {
        uint8_t sc, mm;
        get_scale_min_k4(g, blk->scales, &sc, &mm);
        const size_t ai = (size_t)(gbase + g) * TB + t;
        const __m256i bytes = _mm256_loadu_si256((const __m256i *)(qs + (g / 2) * 32));
        __m256i q = (g & 1) ? _mm256_and_si256(_mm256_srli_epi16(bytes, 4), m0f) : _mm256_and_si256(bytes, m0f);
        const __m256i bit = _mm256_and_si256(qhv, _mm256_set1_epi8((char)(1u << g)));
        const __m256i notset = _mm256_cmpeq_epi8(bit, z);
        q = _mm256_or_si256(q, _mm256_andnot_si256(notset, _mm256_set1_epi8(16)));
        const __m256i prod = _mm256_maddubs_epi16(q, _mm256_loadu_si256((const __m256i *)(x8 + ai * 32)));
        const int32_t wd = hsum_i32_256(_mm256_madd_epi16(prod, ones));
        const int32_t xsum = xsumq[ai * 2] + xsumq[ai * 2 + 1];
        acc += xmeta[ai * 2] * (d * sc * (float)wd - dm * mm * (float)xsum);
    }
    return acc;
}

// Q6_K: 16 groups of 16 values, y = d*sc*(q-32); the 16-value groups map to
// the two halves of the 32-value activation groups.
__attribute__((target("avx2,fma"))) float q6k_i8_block(const block_q6_K * blk, const int8_t * x8, const float * xmeta,
                                                       const int32_t * xsumq, int TB, int t, int gbase) {
    const float d = ggml_half_to_float(blk->d);
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i m3 = _mm_set1_epi8(0x03);
    const __m128i ones = _mm_set1_epi16(1);
    float acc = 0.0f;
    for (int s = 0; s < 16; s++) {
        const int s8 = s & 7, blk2 = s >> 3;
        const uint8_t * ql = blk->ql + blk2 * 64;
        const uint8_t * qh = blk->qh + blk2 * 32;
        const int plane = s8 >> 1, half = s8 & 1;
        const int shift = (plane == 0) ? 0 : ((plane == 1) ? 2 : ((plane == 2) ? 4 : 6));
        const uint8_t * qlp = ql + ((plane == 1 || plane == 3) ? 32 : 0) + half * 16;
        const uint8_t * qhp = qh + half * 16;
        const __m128i qlv = _mm_loadu_si128((const __m128i *)qlp);
        const __m128i qhv = _mm_loadu_si128((const __m128i *)qhp);
        const __m128i lo = (plane >= 2) ? _mm_and_si128(_mm_srli_epi16(qlv, 4), m0f) : _mm_and_si128(qlv, m0f);
        const __m128i two = _mm_and_si128(_mm_srli_epi16(qhv, shift), m3);
        const __m128i q = _mm_or_si128(lo, _mm_slli_epi16(two, 4)); // 0..63
        const int g32 = gbase + (s >> 1);
        const size_t ai = (size_t)g32 * TB + t;
        const __m128i xv = _mm_loadu_si128((const __m128i *)(x8 + ai * 32 + half * 16));
        const __m128i prod = _mm_maddubs_epi16(q, xv);
        const int32_t wd = hsum_i32_128(_mm_madd_epi16(prod, ones));
        const int32_t xsum = xsumq[ai * 2 + half];
        const float ds = d * (float)blk->scales[s];
        acc += xmeta[ai * 2] * ds * (float)(wd - 32 * xsum);
    }
    return acc;
}

float i8_row(uint32_t type, const void * wrow, const int8_t * x8, const float * xmeta, const int32_t * xsumq, int TB,
             int t, int K) {
    switch (type) {
    case 12: {
        float acc = 0.0f;
        const int nb = K / 256;
        const block_q4_K * w = (const block_q4_K *)wrow;
        for (int b = 0; b < nb; b++) {
            acc += q4k_i8_block(w + b, x8, xmeta, xsumq, TB, t, b * 8);
        }
        return acc;
    }
    case 13: {
        float acc = 0.0f;
        const int nb = K / 256;
        const block_q5_K * w = (const block_q5_K *)wrow;
        for (int b = 0; b < nb; b++) {
            acc += q5k_i8_block(w + b, x8, xmeta, xsumq, TB, t, b * 8);
        }
        return acc;
    }
    case 14: {
        float acc = 0.0f;
        const int nb = K / 256;
        const block_q6_K * w = (const block_q6_K *)wrow;
        for (int b = 0; b < nb; b++) {
            acc += q6k_i8_block(w + b, x8, xmeta, xsumq, TB, t, b * 8);
        }
        return acc;
    }
    default:
        return 0.0f;
    }
}

bool i8_supported(uint32_t type) {
    return (type == 12 || type == 13 || type == 14) && isa().qgemv_sb != nullptr;
}

// ---------------------------------------------------------------------------
// Tiled int8 GEMM for prefill batches (TB >= 2 work items).
//
// The per-(token,row) kernel above streams every weight row once per token,
// which at batch sizes (TB=64..384) turns the prefill into a weight-bandwidth
// problem (~121 ms per CPU layer for a 128-token batch, measured).  This path
// is weight-stationary over a token tile instead: a row block's quantized
// groups are unpacked once per (row-tile, k-block) and dotted against kI8Tt
// tokens at a time, so the weights are read once per row and the activations
// once per row-tile.  The per-element math is identical to i8_row (same group
// order), so results are bit-identical to the old loop.
// ---------------------------------------------------------------------------
constexpr int kI8Rt = 16;  // output-row tile
constexpr int kI8Tt = 32;  // token tile == one activation group

// Per-(k-block, activation-group) x side hoisted across the row tile: the x8
// vectors, per-token scales and activation sums are the same for every output
// row of the tile, so they are gathered once per (block, group) and reused by
// the row contributions below (the weight unpack stays row-specific).
struct qx_t {
    __m256i xv[kI8Tt];
    float xm[kI8Tt];
    int32_t xs0[kI8Tt], xs1[kI8Tt]; // the two per-16-value activation sums
};

__attribute__((target("avx2,fma"))) void q4k_g_contrib(const block_q4_K * blk, int g, const qx_t & qx, int TT,
                                                      float * acc) {
    const float d = ggml_half_to_float(blk->d);
    const float dm = ggml_half_to_float(blk->dmin);
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    uint8_t sc, mm;
    get_scale_min_k4(g, blk->scales, &sc, &mm);
    const __m256i bytes = _mm256_loadu_si256((const __m256i *)(blk->qs + (g / 2) * 32));
    const __m256i q = (g & 1) ? _mm256_and_si256(_mm256_srli_epi16(bytes, 4), m0f) : _mm256_and_si256(bytes, m0f);
    const float dsc = d * sc;
    const float dmm = dm * mm;
    for (int v = 0; v < TT; v++) {
        const __m256i prod = _mm256_maddubs_epi16(q, qx.xv[v]);
        const int32_t wd = hsum_i32_256(_mm256_madd_epi16(prod, ones));
        acc[v] += qx.xm[v] * (dsc * (float)wd - dmm * (float)(qx.xs0[v] + qx.xs1[v]));
    }
}

__attribute__((target("avx2,fma"))) void q5k_g_contrib(const block_q5_K * blk, int g, const qx_t & qx, int TT,
                                                      float * acc) {
    const float d = ggml_half_to_float(blk->d);
    const float dm = ggml_half_to_float(blk->dmin);
    const __m256i qhv = _mm256_loadu_si256((const __m256i *)blk->qh);
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i z = _mm256_setzero_si256();
    uint8_t sc, mm;
    get_scale_min_k4(g, blk->scales, &sc, &mm);
    const __m256i bytes = _mm256_loadu_si256((const __m256i *)(blk->qs + (g / 2) * 32));
    __m256i q = (g & 1) ? _mm256_and_si256(_mm256_srli_epi16(bytes, 4), m0f) : _mm256_and_si256(bytes, m0f);
    const __m256i bit = _mm256_and_si256(qhv, _mm256_set1_epi8((char)(1u << g)));
    const __m256i notset = _mm256_cmpeq_epi8(bit, z);
    q = _mm256_or_si256(q, _mm256_andnot_si256(notset, _mm256_set1_epi8(16)));
    const float dsc = d * sc;
    const float dmm = dm * mm;
    for (int v = 0; v < TT; v++) {
        const __m256i prod = _mm256_maddubs_epi16(q, qx.xv[v]);
        const int32_t wd = hsum_i32_256(_mm256_madd_epi16(prod, ones));
        acc[v] += qx.xm[v] * (dsc * (float)wd - dmm * (float)(qx.xs0[v] + qx.xs1[v]));
    }
}

__attribute__((target("avx2,fma"))) void q6k_g_contrib(const block_q6_K * blk, int g, const qx_t & qx, int TT,
                                                      float * acc) {
    const float d = ggml_half_to_float(blk->d);
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i m3 = _mm_set1_epi8(0x03);
    const __m128i ones = _mm_set1_epi16(1);
    // Q6 has 16 scales over 8 activation groups: scale s uses group g = s >> 1.
    for (int i = 0; i < 2; i++) {
        const int s = 2 * g + i, s8 = s & 7, blk2 = s >> 3, half = i;
        const uint8_t * ql = blk->ql + blk2 * 64;
        const uint8_t * qh = blk->qh + blk2 * 32;
        const int plane = s8 >> 1;
        const int shift = (plane == 0) ? 0 : ((plane == 1) ? 2 : ((plane == 2) ? 4 : 6));
        const uint8_t * qlp = ql + ((plane == 1 || plane == 3) ? 32 : 0) + half * 16;
        const uint8_t * qhp = qh + half * 16;
        const __m128i qlv = _mm_loadu_si128((const __m128i *)qlp);
        const __m128i qhv = _mm_loadu_si128((const __m128i *)qhp);
        const __m128i lo = (plane >= 2) ? _mm_and_si128(_mm_srli_epi16(qlv, 4), m0f) : _mm_and_si128(qlv, m0f);
        const __m128i two = _mm_and_si128(_mm_srli_epi16(qhv, shift), m3);
        const __m128i q = _mm_or_si128(lo, _mm_slli_epi16(two, 4)); // 0..63
        const float ds = d * (float)blk->scales[s];
        for (int v = 0; v < TT; v++) {
            const __m128i xv = half ? _mm256_extracti128_si256(qx.xv[v], 1) : _mm256_castsi256_si128(qx.xv[v]);
            const __m128i prod = _mm_maddubs_epi16(q, xv);
            const int32_t wd = hsum_i32_128(_mm_madd_epi16(prod, ones));
            const int32_t xsum = half ? qx.xs1[v] : qx.xs0[v];
            acc[v] += qx.xm[v] * ds * (float)(wd - 32 * xsum);
        }
    }
}

__attribute__((target("avx2,fma"))) void i8_gemm_tile_part(uint32_t type, const void * w, const int8_t * x8,
                                                           const float * xmeta, const int32_t * xsumq, int TB, int t0,
                                                           int TT, int r0, int rn, int nblocks, size_t row_bytes,
                                                           float * acc) {
    for (int b = 0; b < nblocks; b++) {
        const int gbase = b * 8;
        for (int g = 0; g < 8; g++) {
            qx_t qx;
            const int gg = gbase + g;
            for (int v = 0; v < TT; v++) {
                const size_t ai = (size_t)gg * TB + (t0 + v);
                qx.xv[v] = _mm256_loadu_si256((const __m256i *)(x8 + ai * 32));
                qx.xm[v] = xmeta[ai * 2];
                qx.xs0[v] = xsumq[ai * 2];
                qx.xs1[v] = xsumq[ai * 2 + 1];
            }
            for (int rr = 0; rr < rn; rr++) {
                const char * wb = (const char *)w + (size_t)(r0 + rr) * nblocks * row_bytes + (size_t)b * row_bytes;
                float * ar = acc + rr * kI8Tt;
                switch (type) {
                case 12: q4k_g_contrib((const block_q4_K *)wb, g, qx, TT, ar); break;
                case 13: q5k_g_contrib((const block_q5_K *)wb, g, qx, TT, ar); break;
                default: q6k_g_contrib((const block_q6_K *)wb, g, qx, TT, ar); break;
                }
            }
        }
    }
}

void cpu_i8_gemm_tiled(uint32_t type, const void * w, const int8_t * x8, const float * xmeta, const int32_t * xsumq,
                       int K, int rows, int TB, float * out, int out_stride, const float * residual, float alpha) {
    const int nblocks = K / 256;
    const size_t row_bytes = (type == 12) ? sizeof(block_q4_K) : (type == 13) ? sizeof(block_q5_K) : sizeof(block_q6_K);
    const int TT = std::min(kI8Tt, TB); // tokens per tile (TB is always a multiple of 32)
    const int nr = (rows + kI8Rt - 1) / kI8Rt;
    par(nr, [&](int rti) {
        float acc[kI8Rt * kI8Tt];
        const int r0 = rti * kI8Rt;
        const int rn = std::min(kI8Rt, rows - r0);
        for (int t0 = 0; t0 < TB; t0 += TT) {
            std::memset(acc, 0, sizeof(acc));
            i8_gemm_tile_part(type, w, x8, xmeta, xsumq, TB, t0, TT, r0, rn, nblocks, row_bytes, acc);
            for (int rr = 0; rr < rn; rr++) {
                const float * a = acc + rr * kI8Tt;
                for (int v = 0; v < TT; v++) {
                    const size_t o = (size_t)(t0 + v) * out_stride + (r0 + rr);
                    float val = alpha * a[v];
                    if (residual) {
                        val += residual[o];
                    }
                    out[o] = val;
                }
            }
        }
    });
}

} // namespace

void cpu_i8_gemv(const cpu_gemv_seg & seg, int n_threads) {
    (void)n_threads;
    if (!seg.w || !i8_supported(seg.type)) {
        cpu_gemv_group(seg.type, &seg, 1, seg.n_rows, 1, seg.K / 256, 0, 0);
        return;
    }
    const int K = seg.K;
    const uint32_t type = seg.type;
    const size_t row_bytes = (type == 12) ? sizeof(block_q4_K) : (type == 13) ? sizeof(block_q5_K) : sizeof(block_q6_K);
    const int nrows = (K / 256);
    par(seg.n_rows, [&](int row) {
        const void * wrow = (const char *)seg.w + (size_t)row * nrows * row_bytes;
        float v = seg.alpha * i8_row(type, wrow, seg.x8, seg.xmeta, seg.xsumq, 1, 0, K);
        if (seg.residual) {
            v += seg.residual[row];
        }
        seg.out[row] = v;
    });
}

void cpu_i8_gemm(const cpu_gemv_seg & seg, int TB, int n_threads) {
    (void)n_threads;
    if (!seg.w || !i8_supported(seg.type)) {
        cpu_gemv_group(seg.type, &seg, 1, seg.n_rows, TB, seg.K / 256, 0, 0);
        return;
    }
    if (TB >= 2) {
        // tiled weight-stationary GEMM: weights read once per row, activations
        // once per row-tile (see cpu_i8_gemm_tiled above)
        cpu_i8_gemm_tiled(seg.type, seg.w, seg.x8, seg.xmeta, seg.xsumq, seg.K, seg.n_rows, TB, seg.out, seg.out_stride,
                          seg.residual, seg.alpha);
        return;
    }
    const int K = seg.K;
    const uint32_t type = seg.type;
    const size_t row_bytes = (type == 12) ? sizeof(block_q4_K) : (type == 13) ? sizeof(block_q5_K) : sizeof(block_q6_K);
    const int nrows = (K / 256);
    const int rows = seg.n_rows;
    par(TB * rows, [&](int j) {
        const int t = j / rows;
        const int row = j % rows;
        const void * wrow = (const char *)seg.w + (size_t)row * nrows * row_bytes;
        float v = seg.alpha * i8_row(type, wrow, seg.x8, seg.xmeta, seg.xsumq, TB, t, K);
        if (seg.residual) {
            v += seg.residual[(size_t)t * seg.out_stride + row];
        }
        seg.out[(size_t)t * seg.out_stride + row] = v;
    });
}

} // namespace si
