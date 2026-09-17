#include "w8.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "quant.h"

namespace si {

// see w8.h for the format description
static inline uint32_t pack_half2(float a, float b) {
    uint16_t ha = ggml_float_to_half(a);
    uint16_t hb = ggml_float_to_half(b);
    return (uint32_t) ha | ((uint32_t) hb << 16);
}

// ---------------------------------------------------------------------------
// GGUF source extraction.  Each function returns the group's integer values
// (unsigned, exactly as stored in the file) plus the block scale/min.
// ---------------------------------------------------------------------------

// Q4_K: 32-value sub-blocks (8 per 256-value super-block), one nibble per byte.
static inline void src_group32_q4(const char * row, int g, uint8_t * out, float & scale, float & mn) {
    const int k0 = g * 32;
    const block_q4_K * blk = (const block_q4_K *) (row + (size_t) (k0 / QK_K) * sizeof(block_q4_K));
    const int e = k0 % QK_K;
    const int chunk = e / 64, part = (e % 64) / 32;
    uint8_t sc, m;
    get_scale_min_k4(e / 32, blk->scales, &sc, &m);
    scale = ggml_half_to_float(blk->d) * sc;
    mn = ggml_half_to_float(blk->dmin) * m;
    const uint8_t * qs = blk->qs + 32 * chunk;
    for (int i = 0; i < 32; i++) out[i] = part ? (uint8_t) (qs[i] >> 4) : (uint8_t) (qs[i] & 0xF);
}

// Q5_K: as Q4_K plus a 5th high bit per value.
static inline void src_group32_q5(const char * row, int g, uint8_t * out, float & scale, float & mn) {
    const int k0 = g * 32;
    const block_q5_K * blk = (const block_q5_K *) (row + (size_t) (k0 / QK_K) * sizeof(block_q5_K));
    const int e = k0 % QK_K;
    const int chunk = e / 64, part = (e % 64) / 32;
    uint8_t sc, m;
    get_scale_min_k4(e / 32, blk->scales, &sc, &m);
    scale = ggml_half_to_float(blk->d) * sc;
    mn = ggml_half_to_float(blk->dmin) * m;
    const uint8_t * qs = blk->qs + 32 * chunk;
    const int hbit = 2 * chunk + part;
    for (int i = 0; i < 32; i++) {
        const int nib = part ? (qs[i] >> 4) : (qs[i] & 0xF);
        out[i] = (uint8_t) (nib | (((blk->qh[i] >> hbit) & 1) << 4));
    }
}

// Q6_K: 16-value sub-blocks, 6-bit values, no explicit min (w = scale*(q-32)).
static inline void src_group16_q6(const char * row, int g, uint8_t * out, float & scale, float & mn) {
    const int k0 = g * 16;
    const block_q6_K * blk = (const block_q6_K *) (row + (size_t) (k0 / QK_K) * sizeof(block_q6_K));
    const int e = k0 % QK_K;
    const int h = e / 128;
    const int r = e % 128;
    const int u = r / 32;          // which 32-value region (0..3)
    const int l0 = r % 32;         // 0 or 16
    const uint8_t * ql = blk->ql + h * 64;
    const uint8_t * qh = blk->qh + h * 32;
    const float d = ggml_half_to_float(blk->d);
    scale = d * (float) blk->scales[h * 8 + u * 2 + l0 / 16];
    mn = scale * 32.0f;            // w = scale*(q-32)
    for (int i = 0; i < 16; i++) {
        const int l = l0 + i;
        uint8_t q6;
        switch (u) {
            case 0: q6 = (uint8_t) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)); break;
            case 1: q6 = (uint8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)); break;
            case 2: q6 = (uint8_t) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)); break;
            default: q6 = (uint8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)); break;
        }
        out[i] = q6;
    }
}

// ---------------------------------------------------------------------------
// packing of the integer values into the SIn layout
// ---------------------------------------------------------------------------

// one 16-value half: 8 bytes of interleaved nibbles (byte b holds value b in
// the low nibble and value b+8 in the high nibble)
static inline void pack_nib16(const uint8_t * q, uint8_t * out) {
    for (int b = 0; b < 8; b++) out[b] = (uint8_t) ((q[b] & 0xF) | ((q[b + 8] & 0xF) << 4));
}

// k=4, 32 values -> 16 bytes (two nibble halves)
static inline void pack4(const uint8_t * q, uint8_t * out) {
    pack_nib16(q, out);
    pack_nib16(q + 16, out + 8);
}

// 1-bit plane for a 16-value half (bit i = bit 4 of value i)
static inline uint16_t plane1(const uint8_t * q) {
    uint16_t pl = 0;
    for (int i = 0; i < 16; i++) pl |= (uint16_t) (((q[i] >> 4) & 1) << i);
    return pl;
}

// k=5, 32 values -> 24 bytes: [nibbles 0..15][nibbles 16..31][1-bit planes]
// [4 bytes of padding].  Groups are 8-byte aligned and so are both nibble
// halves, which lets the kernels use 8-byte loads.
static inline void pack5(const uint8_t * q, uint8_t * out) {
    pack_nib16(q, out);
    pack_nib16(q + 16, out + 8);
    const uint16_t p0 = plane1(q), p1 = plane1(q + 16);
    out[16] = (uint8_t) p0; out[17] = (uint8_t) (p0 >> 8);
    out[18] = (uint8_t) p1; out[19] = (uint8_t) (p1 >> 8);
    out[20] = out[21] = out[22] = out[23] = 0;
}

// k=6, 16 values -> 12 bytes: [nibbles][4-byte plane], 2 bits per value at
// plane bits 2i, 2i+1
static inline void pack6(const uint8_t * q, uint8_t * out) {
    pack_nib16(q, out);
    uint32_t pl = 0;
    for (int i = 0; i < 16; i++) pl |= (uint32_t) ((q[i] >> 4) & 3) << (2 * i);
    for (int j = 0; j < 4; j++) out[8 + j] = (uint8_t) (pl >> (8 * j));
}

// ---------------------------------------------------------------------------

static bool w8_force4() {
    static const bool f = [] {
        const char * e = getenv("PF_SI4");
        return e && atoi(e) != 0;
    }();
    return f;
}

uint32_t w8_effective_type(uint32_t type) { return w8_force4() ? 12u : type; }

size_t w8_vals_bytes(uint32_t type, int K, int N) {
    const int G = w8_force4() ? 32 : w8_group_size(type);
    const int gb = w8_force4() ? 16 : w8_group_bytes(type);
    return (size_t) N * (K / G) * gb;
}

size_t w8_meta_count(uint32_t type, int K, int N) {
    const int G = w8_force4() ? 32 : w8_group_size(type);
    return (size_t) N * (K / G);
}

size_t w8_meta_bytes(uint32_t type, int K, int N, bool scale_only) {
    const int e = (scale_only && !w8_force4() && type == 14) ? 2 : 4;
    return w8_meta_count(type, K, N) * (size_t) e;
}

// Q6_K groups hold w = scale*(q-32); the min stored in the 4-byte meta is
// always fp16(32*scale), so it can be derived from the scale alone -- unless
// fp16(scale) loses information relative to fp16(32*scale), i.e. the scale is
// subnormal-ish and the conversion flushes/rounds it while the min is normal.
// Those (rare) groups get the full 4-byte meta per tensor.
bool w8_q6_scale_only_ok(uint32_t ggml_type, const void * src, int K, int N) {
    if (ggml_type != 14 || w8_force4()) return false;
    const block_q6_K * p = (const block_q6_K *) src;
    const int nb = K / QK_K;
    const int rows = N;
    for (int r = 0; r < rows * nb; r++) {
        const block_q6_K * blk = p + r;
        const float d = ggml_half_to_float(blk->d);
        for (int g = 0; g < QK_K / 16; g++) {
            const float sc = d * (float) blk->scales[g];
            const uint16_t hs = ggml_float_to_half(sc);
            const uint16_t hm = ggml_float_to_half(32.0f * sc);
            if (ggml_half_to_float(hm) != 32.0f * ggml_half_to_float(hs)) return false;
        }
    }
    return true;
}

bool w8_repack(uint32_t type, const void * src, int K, int N, uint8_t * vals_out,
               uint32_t * meta_out, bool scale_only) {
    if (K % 32 != 0) return false;
    if (type != 12 && type != 13 && type != 14) return false;
    // PF_SI4=1 re-quantizes every tensor to 4-bit asymmetric groups of 32
    // (halves the bytes again at some quality cost); default keeps the GGUF
    // bit width and values exactly.
    const bool force4 = w8_force4();
    const int G = force4 ? 32 : w8_group_size(type);
    const int gb = force4 ? 16 : w8_group_bytes(type);
    const int MG = K / G;

    const size_t row_bytes = quant_row_bytes(type, K);
    const char * base = (const char *) src;
    std::vector<float> frow;
    if (force4) frow.resize(K);

    for (int r = 0; r < N; r++) {
        const int rb = r / kRB, ri = r % kRB;
        const char * row = base + (size_t) r * row_bytes;
        if (force4) dequantize_row(type, row, frow.data(), K);
        for (int g = 0; g < MG; g++) {
            uint8_t q[32];
            float s, m;
            if (force4) {
                // asymmetric 4-bit: w = s*q - m over the 32-value group
                float lo = frow[(size_t) g * 32], hi = lo;
                for (int i = 1; i < 32; i++) {
                    const float v = frow[(size_t) g * 32 + i];
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
                const float sc = (hi - lo) / 15.0f;
                const float inv = sc > 0.f ? 1.0f / sc : 0.f;
                for (int i = 0; i < 32; i++) {
                    int vq = (int) std::lround((frow[(size_t) g * 32 + i] - lo) * inv);
                    q[i] = (uint8_t) (vq < 0 ? 0 : (vq > 15 ? 15 : vq));
                }
                s = sc;
                m = -lo;  // kernel reconstructs w = scale*q - min
            } else {
                switch (type) {
                    case 12: src_group32_q4(row, g, q, s, m); break;
                    case 13: src_group32_q5(row, g, q, s, m); break;
                    default: src_group16_q6(row, g, q, s, m); break;
                }
            }
            const size_t o = (size_t) ((rb * MG + g) * kRB + ri);
            uint8_t * dst = vals_out + o * gb;
            if (gb == 16) pack4(q, dst);
            else if (gb == 24) pack5(q, dst);
            else if (gb == 12) pack6(q, dst);
            else std::memcpy(dst, q, (size_t) gb);
            // Q6_K with all-exact scales: min == 32*scale is derived by the
            // kernels, so only the fp16 scale is stored (2 bytes per group)
            if (scale_only && !force4 && type == 14)
                ((uint16_t *) meta_out)[o] = ggml_float_to_half(s);
            else
                meta_out[o] = pack_half2(s, m);
        }
    }
    return true;
}

} // namespace si
