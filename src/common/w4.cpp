#include "w4.h"

#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cstdlib>

namespace si {

bool w4_all_enabled() {
    static const bool on = [] {
        const char * e = getenv("PF_W4_ALL");
        return e && atoi(e) != 0;
    }();
    return on;
}

bool w4_k5_only() {
    static const bool on = [] {
        const char * e = getenv("PF_W4_K5");
        return e && atoi(e) != 0;
    }();
    return on;
}

bool w4_supported(uint32_t ggml_type) {
    // Q4_K: q in [0,15] on a per-32-group grid (d*sc_j, dmin*m_j).
    if (ggml_type == 12) {
        return true;
    }
    // PF_W4_K5=1: Q5_K only.  The k5 store is 0.875 B/weight and Q5_K is the
    // largest single byte consumer of the 27B weight pass (6.15 GB of 18.8,
    // 35 %), against the u4 grid's 0.625 - so this is the one lossy
    // re-quantization with a real payoff.  Separate from PF_W4_ALL (which also
    // takes the cb4/int8 types) because the accuracy cost is a per-type
    // decision, not a global one.
    if (w4_k5_only() && ggml_type == 13) {
        return true;
    }
    // PF_W4_ALL: every other quant type is re-quantized onto the same 4-bit
    // grid (see w4_all_enabled).  F32/F16 have no row_bytes entry.
    return w4_all_enabled() && ggml_type != 0 && ggml_type != 1;
}

// Q4_K: a 256-element super-block holds 8 groups of 32; the scale/min pair of
// each group is 6-bit packed (get_scale_min_k4), and the 4-bit values sit in
// qs[] as low nibble = first 32 elements, high nibble = next 32.
static bool pack_q4_K(const void * src, int K, int N, w4t & out) {
    if (K % QK_K != 0) {
        return false;
    }
    const int ng = K / kW4Group;
    out.vals.assign((size_t)N * K / 2, 0);
    out.scale.assign((size_t)ng * N, 0);
    out.off.assign((size_t)ng * N, 0);
    const size_t row_bytes = quant_row_bytes(12, K);
    for (int n = 0; n < N; n++) {
        const uint8_t * row = (const uint8_t *)src + (size_t)n * row_bytes;
        for (int b = 0; b < K / QK_K; b++) {
            const block_q4_K * x = (const block_q4_K *)row + b;
            const float d = ggml_half_to_float(x->d);
            const float mn = ggml_half_to_float(x->dmin);
            const uint8_t * qs = x->qs;
            int is = 0;
            int k = b * QK_K;
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc0, m0, sc1, m1;
                get_scale_min_k4(is + 0, x->scales, &sc0, &m0);
                get_scale_min_k4(is + 1, x->scales, &sc1, &m1);
                out.scale[(size_t)(k / kW4Group) * N + n] = ggml_float_to_half(d * sc0);
                out.off[(size_t)(k / kW4Group) * N + n] = (uint16_t)ggml_float_to_half(-mn * m0);
                out.scale[(size_t)(k / kW4Group + 1) * N + n] = ggml_float_to_half(d * sc1);
                out.off[(size_t)(k / kW4Group + 1) * N + n] = (uint16_t)ggml_float_to_half(-mn * m1);
                const size_t base = (size_t)n * K;
                for (int l = 0; l < 32; l++) {
                    const int qa = qs[l] & 0xF;
                    const int qb = (qs[l] >> 4) & 0xF;
                    size_t ia = base + k + l;
                    size_t ib = base + k + 32 + l;
                    out.vals[ia >> 1] |= (uint8_t)(qa << ((ia & 1) * 4));
                    out.vals[ib >> 1] |= (uint8_t)(qb << ((ib & 1) * 4));
                }
                qs += 32;
                is += 2;
                k += 64;
            }
        }
    }
    out.K = K;
    out.N = N;
    out.bits = 4;
    return true;
}

// Generic re-quantization onto the u4 grid (PF_W4_ALL): dequantize the row with
// the type's own reference dequantizer, then fit one affine (step, offset) per
// 32-group by min/max and round to a nibble.  This is the "cheap bytes" path:
// 0.625 B/weight instead of int8's 1.0625 for every non-Q4_K type, at the cost
// of the type's native resolution.  `off` is the *additive* constant (w ~=
// step*q + off) so it slots straight into the u4 GEMV's `step*qdot + off*xs`.
// Mirrors w8.cpp's PF_SI4 generic requant.
static bool pack_generic(uint32_t ggml_type, const void * src, int K, int N, w4t & out, bool allow) {
    if (!(allow || w4_all_enabled()) || (K % kW4Group) != 0) {
        return false;
    }
    const size_t row_bytes = quant_row_bytes(ggml_type, K);
    if (row_bytes == 0) {
        return false;
    }
    const int ng = K / kW4Group;
    out.vals.assign((size_t)N * K / 2, 0);
    out.scale.assign((size_t)ng * N, 0);
    out.off.assign((size_t)ng * N, 0);
    std::vector<float> row((size_t)K);
    double num = 0.0, den = 0.0;
    for (int n = 0; n < N; n++) {
        dequantize_row(ggml_type, (const char *)src + (size_t)n * row_bytes, row.data(), K);
        for (int g = 0; g < ng; g++) {
            const float * w = row.data() + (size_t)g * kW4Group;
            float mn = w[0], mx = w[0];
            for (int j = 1; j < kW4Group; j++) {
                mn = std::min(mn, w[j]);
                mx = std::max(mx, w[j]);
            }
            float step = (mx - mn) / 15.0f;
            if (!(step > 0.f)) {
                step = 1.f;
            }
            const float inv = 1.0f / step;
            out.scale[(size_t)g * N + n] = ggml_float_to_half(step);
            out.off[(size_t)g * N + n] = ggml_float_to_half(mn);
            const size_t base = (size_t)n * K + (size_t)g * kW4Group;
            for (int j = 0; j < kW4Group; j++) {
                int q = (int)std::lround((w[j] - mn) * inv);
                q = std::max(0, std::min(15, q));
                const size_t ia = base + j;
                out.vals[ia >> 1] |= (uint8_t)(q << ((ia & 1) * 4));
            }
        }
    }
    out.K = K;
    out.N = N;
    out.bits = 4;
    return true;
}


// Both IQ4_XS and IQ4_NL store a 32-value group's 4-bit codebook indices as
// "element j = low nibble of byte j, element 16+j = high nibble of byte j" (see
// the reference dequantizers).  The GEMV wants the *interleaved* order the u4
// and k5 planes use - byte k = elements 2k (low) / 2k+1 (high) - because then
// one 256-entry uint16 lookup turns an index byte into the two codebook values
// of one dp4a half-word, halving the LUT traffic per 4 weights (see the kernel
// comment).  This is a pure permutation: no value is touched.
static void interleave_group(const uint8_t * src, uint8_t * dst) {
    for (int k = 0; k < 16; k++) {
        uint8_t lo, hi;
        if (k < 8) {
            lo = (uint8_t)(src[2 * k] & 0xF);
            hi = (uint8_t)(src[2 * k + 1] & 0xF);
        } else {
            lo = (uint8_t)((src[2 * k - 16] >> 4) & 0xF);
            hi = (uint8_t)((src[2 * k - 15] >> 4) & 0xF);
        }
        dst[k] = (uint8_t)(lo | (hi << 4));
    }
}

bool cb4_supported(uint32_t ggml_type) {
    return ggml_type == 23 || ggml_type == 20; // IQ4_XS, IQ4_NL
}

bool cb4_pack(uint32_t ggml_type, const void * src, int K, int N, cb4t & out) {
    if (!src || !cb4_supported(ggml_type) || K <= 0 || N <= 0 || (K % 32) != 0) {
        return false;
    }
    const int ng = K / 32;
    const size_t row_bytes = quant_row_bytes(ggml_type, K);
    if (row_bytes == 0) {
        return false;
    }
    out.idx.assign((size_t)N * (size_t)(K / 2), 0);
    out.scale.assign((size_t)ng * N, 0);
    for (int n = 0; n < N; n++) {
        const char * row = (const char *)src + (size_t)n * row_bytes;
        uint8_t * irow = out.idx.data() + (size_t)n * (K / 2);
        if (ggml_type == 23) {
            // 136-byte super-block of 256: f16 d, u16 scales_h, u8 scales_l[4],
            // u8 qs[128]; per 32 values ls = scales_l/h[ib] and dl = d*(ls-32)
            for (int sb = 0; sb < K / QK_K; sb++) {
                const block_iq4_xs * blk = (const block_iq4_xs *)(row + (size_t)sb * sizeof(block_iq4_xs));
                const float d = ggml_half_to_float(blk->d);
                for (int ib = 0; ib < QK_K / 32; ib++) {
                    const int ls = ((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                                   | (((blk->scales_h >> (2 * ib)) & 3) << 4);
                    const int g = sb * (QK_K / 32) + ib;
                    out.scale[(size_t)g * N + n] = ggml_float_to_half(d * (float)(ls - 32));
                    interleave_group(blk->qs + (size_t)ib * 16, irow + (size_t)g * 16);
                }
            }
        } else {
            // 18-byte block of 32: f16 d, u8 qs[16]; a single scale per group
            for (int g = 0; g < ng; g++) {
                const block_iq4_nl * blk = (const block_iq4_nl *)(row + (size_t)g * sizeof(block_iq4_nl));
                out.scale[(size_t)g * N + n] = blk->d;
                interleave_group(blk->qs, irow + (size_t)g * 16);
            }
        }
    }
    out.K = K;
    out.N = N;
    return true;
}

// Q5_K: 256-element super-block, 8 groups of 32; per group q5 = (qs nibble) |
// (qh bit << 4) and w = d*sc*q5 - dmin*m, with the 6-bit (sc, m) pair read by
// get_scale_min_k4.  The native qs layout is split-half (the low nibble of
// qs[l] is element j+l, the high nibble element j+32+l), while the decode GEMV
// wants the interleaved u4 layout, so the nibbles are repacked here; the fifth
// bits (one per element, strided across qh) become the separate `hi` plane in
// the split-plane element order the kernel reads.
static bool pack_q5_K(const void * src, int K, int N, k5t & out) {
    if (K % QK_K != 0) {
        return false;
    }
    const int ng = K / kW4Group;
    out.vals.assign((size_t)N * K / 2, 0);
    out.hi.assign((size_t)N * (K / 8), 0);
    out.scale.assign((size_t)ng * N, 0);
    out.off.assign((size_t)ng * N, 0);
    const size_t row_bytes = quant_row_bytes(13, K);
    for (int n = 0; n < N; n++) {
        const uint8_t * row = (const uint8_t *)src + (size_t)n * row_bytes;
        uint8_t * vrow = out.vals.data() + (size_t)n * (K / 2);
        uint8_t * hrow = out.hi.data() + (size_t)n * (K / 8);
        for (int b = 0; b < K / QK_K; b++) {
            const block_q5_K * x = (const block_q5_K *)row + b;
            const float d = ggml_half_to_float(x->d);
            const float mn = ggml_half_to_float(x->dmin);
            const uint8_t * qs = x->qs;
            const uint8_t * qh = x->qh;
            int is = 0;
            int p = 0; // bit position of this 64-block's pair inside qh[]
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc0, m0, sc1, m1;
                get_scale_min_k4(is + 0, x->scales, &sc0, &m0);
                get_scale_min_k4(is + 1, x->scales, &sc1, &m1);
                const int g = (b * QK_K + j) / kW4Group; // +0 = low nibbles, +1 = high
                out.scale[(size_t)g * N + n] = ggml_float_to_half(d * sc0);
                out.off[(size_t)g * N + n] = (uint16_t)ggml_float_to_half(-mn * m0);
                out.scale[(size_t)(g + 1) * N + n] = ggml_float_to_half(d * sc1);
                out.off[(size_t)(g + 1) * N + n] = (uint16_t)ggml_float_to_half(-mn * m1);
                uint16_t ev0 = 0, od0 = 0, ev1 = 0, od1 = 0;
                for (int l = 0; l < 32; l++) {
                    const size_t e0 = (size_t)b * QK_K + j + l; // group g, element l
                    const size_t e1 = e0 + 32;                   // group g+1
                    vrow[e0 >> 1] |= (uint8_t)((qs[l] & 0xF) << ((e0 & 1) * 4));
                    vrow[e1 >> 1] |= (uint8_t)(((qs[l] >> 4) & 0xF) << ((e1 & 1) * 4));
                    const int b0 = (qh[l] >> p) & 1;
                    const int b1 = (qh[l] >> (p + 1)) & 1;
                    if (l & 1) {
                        od0 |= (uint16_t)(b0 << (l >> 1));
                        od1 |= (uint16_t)(b1 << (l >> 1));
                    } else {
                        ev0 |= (uint16_t)(b0 << (l >> 1));
                        ev1 |= (uint16_t)(b1 << (l >> 1));
                    }
                }
                // 4 bytes per group: even-element bits then odd-element bits
                uint8_t * h0 = hrow + (size_t)g * 4;
                uint8_t * h1 = h0 + 4;
                h0[0] = (uint8_t)(ev0 & 0xFF);
                h0[1] = (uint8_t)(ev0 >> 8);
                h0[2] = (uint8_t)(od0 & 0xFF);
                h0[3] = (uint8_t)(od0 >> 8);
                h1[0] = (uint8_t)(ev1 & 0xFF);
                h1[1] = (uint8_t)(ev1 >> 8);
                h1[2] = (uint8_t)(od1 & 0xFF);
                h1[3] = (uint8_t)(od1 >> 8);
                qs += 32;
                is += 2;
                p += 2;
            }
        }
    }
    out.K = K;
    out.N = N;
    return true;
}

bool k5_supported(uint32_t ggml_type) {
    return ggml_type == 13; // Q5_K
}

bool k5_pack(uint32_t ggml_type, const void * src, int K, int N, k5t & out) {
    if (!src || !k5_supported(ggml_type) || K <= 0 || N <= 0 || (K % QK_K) != 0) {
        return false;
    }
    return pack_q5_K(src, K, N, out);
}

bool w4_pack(uint32_t ggml_type, const void * src, int K, int N, w4t & out) {
    if (!src || K <= 0 || N <= 0) {
        return false;
    }
    switch (ggml_type) {
    case 12:
        return pack_q4_K(src, K, N, out);
    case 13:
        // PF_W4_K5 (see w4_supported): Q5_K onto the u4 grid, nothing else.
        return pack_generic(ggml_type, src, K, N, out, w4_k5_only());
    default:
        return pack_generic(ggml_type, src, K, N, out, false);
    }
}

// The lossy generic pack for one explicitly chosen tensor, skipping the
// PF_W4_ALL gate: the MTP draft's LM head (PF_MTP_HEAD_W4) is read once per
// drafted token, so 0.625 B/weight instead of int8's 1.0625 is the draft's
// dominant cost - and the draft only needs its argmax.
bool w4_pack_any(uint32_t ggml_type, const void * src, int K, int N, w4t & out) {
    if (!src || K <= 0 || N <= 0) {
        return false;
    }
    return pack_generic(ggml_type, src, K, N, out, true);
}

// The 2-bit draft store (see w2t).  Same per-32 group (min, step) fit as
// pack_generic with 4 levels instead of 16, and the nibble plane replaced by a
// 2-bit plane: 8 data bytes + 2 f16 per 32 weights = 0.375 B/weight.
bool w2_pack_any(uint32_t ggml_type, const void * src, int K, int N, w2t & out) {
    if (!src || K <= 0 || N <= 0 || (K % kW4Group) != 0) {
        return false;
    }
    const size_t row_bytes = quant_row_bytes(ggml_type, K);
    if (row_bytes == 0) {
        return false;
    }
    const int ng = K / kW4Group;
    out.vals.assign((size_t)N * K / 4, 0);
    out.scale.assign((size_t)ng * N, 0);
    out.off.assign((size_t)ng * N, 0);
    std::vector<float> row((size_t)K);
    double num = 0.0, den = 0.0;
    for (int n = 0; n < N; n++) {
        dequantize_row(ggml_type, (const char *)src + (size_t)n * row_bytes, row.data(), K);
        for (int g = 0; g < ng; g++) {
            const float * w = row.data() + (size_t)g * kW4Group;
            float mn = w[0], mx = w[0];
            for (int j = 1; j < kW4Group; j++) {
                mn = std::min(mn, w[j]);
                mx = std::max(mx, w[j]);
            }
            float step = (mx - mn) / 3.0f;
            if (!(step > 0.f)) {
                step = 1.f;
            }
            const float inv = 1.0f / step;
            out.scale[(size_t)g * N + n] = ggml_float_to_half(step);
            out.off[(size_t)g * N + n] = ggml_float_to_half(mn);
            const size_t base = (size_t)n * (K / 4) + (size_t)g * 8;
            for (int j = 0; j < kW4Group; j++) {
                int q = (int)std::lround((w[j] - mn) * inv);
                q = std::max(0, std::min(3, q));
                const size_t ia = base + (size_t)(j >> 2);
                out.vals[ia] |= (uint8_t)(q << ((j & 3) * 2));
                const double d = (double)(mn + step * q) - (double)w[j];
                num += d * d;
                den += (double)w[j] * w[j];
            }
        }
    }
    out.rel_l2 = den > 0.0 ? std::sqrt(num / den) : 0.0;
    out.K = K;
    out.N = N;
    return true;
}

} // namespace si
