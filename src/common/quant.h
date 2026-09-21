#pragma once
// Quantized block layouts matching ggml (Q4_K, Q5_K, Q6_K, Q8_0) and
// reference dequantization routines usable on host and device.

#include <cstdint>
#include <cstring>

#define QK_K 256
#define QK8_0 32
#define K_SCALE_SIZE 12

struct block_q4_K {
    uint16_t d;    // super-block scale (f16)
    uint16_t dmin; // super-block min scale (f16)
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144, "q4_K size");

struct block_q5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q5_K) == 176, "q5_K size");

struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};
static_assert(sizeof(block_q6_K) == 210, "q6_K size");

struct block_q8_0 {
    uint16_t d;
    int8_t qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 size");

// ---------------------------------------------------------------- f16 -> f32
static inline float ggml_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(h) & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
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
    int32_t exp = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) {
        return (uint16_t)sign;
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
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
    const float d = ggml_half_to_float(x->d);
    const float mn = ggml_half_to_float(x->dmin);
    const uint8_t * q = x->qs;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(is + 0, x->scales, &sc0, &m0);
        get_scale_min_k4(is + 1, x->scales, &sc1, &m1);
        const float d1 = d * sc0, m1f = mn * m0;
        const float d2 = d * sc1, m2f = mn * m1;
        for (int l = 0; l < 32; l++) {
            y[j + l] = d1 * (q[l] & 0xF) - m1f;
        }
        for (int l = 0; l < 32; l++) {
            y[j + 32 + l] = d2 * (q[l] >> 4) - m2f;
        }
        q += 32;
        is += 2;
    }
}

static inline void dequantize_block_q5_K(const block_q5_K * x, float * y) {
    const float d = ggml_half_to_float(x->d);
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
        for (int l = 0; l < 32; l++) {
            y[j + l] = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1f;
        }
        for (int l = 0; l < 32; l++) {
            y[j + 32 + l] = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2f;
        }
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
    const int8_t * sc = x->scales;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

static inline void dequantize_block_q8_0(const block_q8_0 * x, float * y) {
    const float d = ggml_half_to_float(x->d);
    for (int i = 0; i < QK8_0; i++) {
        y[i] = d * x->qs[i];
    }
}

// ------------------------------------------------- Q3_K / IQ codebooks
// Q3_K: 256-element super-block, 3-bit values (2-bit low plane + 1 high bit)
// with 6-bit packed scales, same packing as Q4_K's scales array.
struct block_q3_K {
    uint8_t hmask[QK_K / 8]; // 32 high bits
    uint8_t qs[QK_K / 4];    // 64 low 2-bit values
    uint8_t scales[12];      // 16 x 6-bit scales
    uint16_t d;
};
static_assert(sizeof(block_q3_K) == 110, "q3_K size");

// IQ3_S: 3.4375 bpw, 256-element super-block with an 8-entry sign byte per
// 32 values and 4-bit block scales.
struct block_iq3_s {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t signs[QK_K / 8];
    uint8_t scales[QK_K / 64];
};
static_assert(sizeof(block_iq3_s) == 110, "iq3_s size");

// IQ4_NL: non-linear 4-bit, 32-element blocks.
struct block_iq4_nl {
    uint16_t d;
    uint8_t qs[QK8_0 / 2];
};
static_assert(sizeof(block_iq4_nl) == 18, "iq4_nl size");

// IQ4_XS: 4.25 bpw, 256-element super-block: 8 sub-blocks of 32, 6-bit scales.
struct block_iq4_xs {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[QK_K / 64];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs size");

// codebooks from ggml (MIT): see third_party/ and THIRD_PARTY_NOTICES.md
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static const uint32_t iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

static inline void dequantize_block_q3_K(const block_q3_K * x, float * y) {
    const float d_all = ggml_half_to_float(x->d);
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    const int8_t * scales = (const int8_t *)aux;
    std::memcpy(aux, x->scales, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const uint8_t * q = x->qs;
    const uint8_t * hm = x->hmask;
    uint8_t m = 1;
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; l++) {
                *y++ = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            }
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; l++) {
                *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

static inline void dequantize_block_iq3_s(const block_iq3_s * x, float * y) {
    const float d = ggml_half_to_float(x->d);
    const uint8_t * qs = x->qs;
    const uint8_t * qh = x->qh;
    const uint8_t * signs = x->signs;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
        const float db1 = d * (1 + 2 * (x->scales[ib32 / 2] & 0xf));
        const float db2 = d * (1 + 2 * (x->scales[ib32 / 2] >> 4));
        for (int half = 0; half < 2; half++) {
            const float db = half ? db2 : db1;
            for (int l = 0; l < 4; l++) {
                const uint8_t * g1 = (const uint8_t *)(iq3s_grid + (qs[2 * l + 0] | ((qh[half] << (8 - 2 * l)) & 256)));
                const uint8_t * g2 = (const uint8_t *)(iq3s_grid + (qs[2 * l + 1] | ((qh[half] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; j++) {
                    y[j] = db * g1[j] * ((signs[l] & kmask_iq2xs[j]) ? -1.f : 1.f);
                    y[j + 4] = db * g2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}

static inline void dequantize_block_iq4_nl(const block_iq4_nl * x, float * y) {
    const float d = ggml_half_to_float(x->d);
    for (int j = 0; j < QK8_0 / 2; j++) {
        y[j] = d * kvalues_iq4nl[x->qs[j] & 0xf];
        y[j + QK8_0 / 2] = d * kvalues_iq4nl[x->qs[j] >> 4];
    }
}

static inline void dequantize_block_iq4_xs(const block_iq4_xs * x, float * y) {
    const float d = ggml_half_to_float(x->d);
    const uint8_t * qs = x->qs;
    for (int ib = 0; ib < QK_K / 32; ib++) {
        const int ls = ((x->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((x->scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        for (int j = 0; j < 16; j++) {
            y[j] = dl * kvalues_iq4nl[qs[j] & 0xf];
            y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
        }
        y += 32;
        qs += 16;
    }
}

// generic: dequantize a full row of n elements
static inline void dequantize_row(uint32_t type, const void * data, float * y, int64_t n) {
    switch (type) {
    case 12: { // Q4_K
        auto * p = (const block_q4_K *)data;
        for (int64_t i = 0; i < n / QK_K; i++) {
            dequantize_block_q4_K(p + i, y + i * QK_K);
        }
    } break;
    case 13: { // Q5_K
        auto * p = (const block_q5_K *)data;
        for (int64_t i = 0; i < n / QK_K; i++) {
            dequantize_block_q5_K(p + i, y + i * QK_K);
        }
    } break;
    case 14: { // Q6_K
        auto * p = (const block_q6_K *)data;
        for (int64_t i = 0; i < n / QK_K; i++) {
            dequantize_block_q6_K(p + i, y + i * QK_K);
        }
    } break;
    case 8: { // Q8_0
        auto * p = (const block_q8_0 *)data;
        for (int64_t i = 0; i < n / QK8_0; i++) {
            dequantize_block_q8_0(p + i, y + i * QK8_0);
        }
    } break;
    case 11: { // Q3_K
        auto * p = (const block_q3_K *)data;
        for (int64_t i = 0; i < n / QK_K; i++) {
            dequantize_block_q3_K(p + i, y + i * QK_K);
        }
    } break;
    case 20: { // IQ4_NL
        auto * p = (const block_iq4_nl *)data;
        for (int64_t i = 0; i < n / QK8_0; i++) {
            dequantize_block_iq4_nl(p + i, y + i * QK8_0);
        }
    } break;
    case 21: { // IQ3_S
        auto * p = (const block_iq3_s *)data;
        for (int64_t i = 0; i < n / QK_K; i++) {
            dequantize_block_iq3_s(p + i, y + i * QK_K);
        }
    } break;
    case 23: { // IQ4_XS
        auto * p = (const block_iq4_xs *)data;
        for (int64_t i = 0; i < n / QK_K; i++) {
            dequantize_block_iq4_xs(p + i, y + i * QK_K);
        }
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
    case 8: return (size_t)(n / QK8_0) * sizeof(block_q8_0);
    case 11: return (size_t)(n / QK_K) * sizeof(block_q3_K);
    case 20: return (size_t)(n / QK8_0) * sizeof(block_iq4_nl);
    case 21: return (size_t)(n / QK_K) * sizeof(block_iq3_s);
    case 23: return (size_t)(n / QK_K) * sizeof(block_iq4_xs);
    case 0: return (size_t)n * 4;
    default: return 0;
    }
}
