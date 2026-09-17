#pragma once
// Quantized block layouts matching ggml (Q4_K, Q5_K, Q6_K, Q8_0) and
// reference dequantization routines usable on host and device.

#include <cstdint>
#include <cstring>
#include <cmath>

#define QK_K 256
#define QK8_0 32
#define K_SCALE_SIZE 12

struct block_q4_K {
    uint16_t d;      // super-block scale (f16)
    uint16_t dmin;   // super-block min scale (f16)
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144, "q4_K size");

struct block_q5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qh[QK_K / 8];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(block_q5_K) == 176, "q5_K size");

struct block_q6_K {
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    uint16_t d;
};
static_assert(sizeof(block_q6_K) == 210, "q6_K size");

struct block_q8_0 {
    uint16_t d;
    int8_t   qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 size");

// ---------------------------------------------------------------- f16 -> f32
static inline float ggml_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(h) & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

static inline uint16_t ggml_float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t) sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t) exp << 10) | (mant >> 13));
}

// packed 6-bit scales of Q4_K/Q5_K: j in [0,8)
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// ------------------------------------------------- reference (host) dequant
// y must have room for 256 floats
static inline void dequantize_block_q4_K(const block_q4_K * x, float * y) {
    const float d   = ggml_half_to_float(x->d);
    const float mn  = ggml_half_to_float(x->dmin);
    const uint8_t * q = x->qs;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(is + 0, x->scales, &sc0, &m0);
        get_scale_min_k4(is + 1, x->scales, &sc1, &m1);
        const float d1 = d * sc0, m1f = mn * m0;
        const float d2 = d * sc1, m2f = mn * m1;
        for (int l = 0; l < 32; l++) y[j + l]      = d1 * (q[l] & 0xF) - m1f;
        for (int l = 0; l < 32; l++) y[j + 32 + l] = d2 * (q[l] >> 4)  - m2f;
        q += 32;
        is += 2;
    }
}

static inline void dequantize_block_q5_K(const block_q5_K * x, float * y) {
    const float d  = ggml_half_to_float(x->d);
    const float mn = ggml_half_to_float(x->dmin);
    const uint8_t * ql = x->qs;
    const uint8_t * qh = x->qh;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(is + 0, x->scales, &sc0, &m0);
        get_scale_min_k4(is + 1, x->scales, &sc1, &m1);
        const float d1 = d * sc0, m1f = mn * m0;
        const float d2 = d * sc1, m2f = mn * m1;
        for (int l = 0; l < 32; l++) y[j + l]      = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1f;
        for (int l = 0; l < 32; l++) y[j + 32 + l] = d2 * ((ql[l] >> 4)  + (qh[l] & u2 ? 16 : 0)) - m2f;
        ql += 32;
        is += 2;
        u1 = (uint8_t)(u1 << 2);
        u2 = (uint8_t)(u2 << 2);
    }
}

static inline void dequantize_block_q6_K(const block_q6_K * x, float * y) {
    const float d = ggml_half_to_float(x->d);
    const uint8_t * ql = x->ql;
    const uint8_t * qh = x->qh;
    const int8_t  * sc = x->scales;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0]   = d * sc[is + 0] * q1;
            y[l + 32]  = d * sc[is + 2] * q2;
            y[l + 64]  = d * sc[is + 4] * q3;
            y[l + 96]  = d * sc[is + 6] * q4;
        }
        y  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

static inline void dequantize_block_q8_0(const block_q8_0 * x, float * y) {
    const float d = ggml_half_to_float(x->d);
    for (int i = 0; i < QK8_0; i++) y[i] = d * x->qs[i];
}

// generic: dequantize a full row of n elements
static inline void dequantize_row(uint32_t type, const void * data, float * y, int64_t n) {
    switch (type) {
        case 12: { // Q4_K
            auto * p = (const block_q4_K *) data;
            for (int64_t i = 0; i < n / QK_K; i++) dequantize_block_q4_K(p + i, y + i * QK_K);
        } break;
        case 13: { // Q5_K
            auto * p = (const block_q5_K *) data;
            for (int64_t i = 0; i < n / QK_K; i++) dequantize_block_q5_K(p + i, y + i * QK_K);
        } break;
        case 14: { // Q6_K
            auto * p = (const block_q6_K *) data;
            for (int64_t i = 0; i < n / QK_K; i++) dequantize_block_q6_K(p + i, y + i * QK_K);
        } break;
        case 8: { // Q8_0
            auto * p = (const block_q8_0 *) data;
            for (int64_t i = 0; i < n / QK8_0; i++) dequantize_block_q8_0(p + i, y + i * QK8_0);
        } break;
        case 0: std::memcpy(y, data, n * 4); break;
        default: break;
    }
}

static inline size_t quant_row_bytes(uint32_t type, int64_t n) {
    switch (type) {
        case 12: return (size_t)(n / QK_K) * sizeof(block_q4_K);
        case 13: return (size_t)(n / QK_K) * sizeof(block_q5_K);
        case 14: return (size_t)(n / QK_K) * sizeof(block_q6_K);
        case 8:  return (size_t)(n / QK8_0) * sizeof(block_q8_0);
        case 0:  return (size_t) n * 4;
        default: return 0;
    }
}
