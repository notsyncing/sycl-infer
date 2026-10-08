// ggml_float_to_half: the subnormal range used to collapse to zero, and that is
// a correctness defect rather than a rounding nit.
//
// Every weight-store scale plane is written with this function (w4.cpp's u4 /
// k5 / cb4 / 2-bit packs, w8.cpp's int8 per-group scales, the KV scales).  f16
// subnormals represent [2^-24, 2^-14) with ten mantissa bits, and the old code
// returned plain zero for all of it - a thousand-fold range thrown away.
//
// Measured consequence on the 27B reference: blk.16.attn_qkv.weight has a
// super-block whose `d` is itself subnormal (4.47e-6, f16 0x004b), so `d*sc =
// 3.13e-5` fell below 2^-14 and `pack_q4_K` wrote a step plane of *identically
// zero*.  The u4 reconstruction of that column then collapsed onto the constant
// `-dmin*m`: 21 of 10240 columns (0.205%) at up to 240% relative L2 against the
// exact GGUF dequant, while every neighbouring column measured the expected
// 0.08% (the f16 rounding of the two constants).  After the fix: 0 of 10240, worst
// column 0.0010.
//
// No GPU, no model: this is the pure bit-level contract.
#include "quant.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>

// quant.h's conversion helpers are at global scope (it declares no namespace).
using ::ggml_float_to_half;
using ::ggml_half_to_float;

static int g_fail = 0;

static void expect_bits(const char * what, float v, uint16_t want) {
    const uint16_t got = ggml_float_to_half(v);
    if (got != want) {
        printf("  FAIL %-46s %.9g -> %04x, expected %04x (%.9g)\n", what, v, got, want,
               ggml_half_to_float(want));
        g_fail++;
    }
}

static void expect_zero(const char * what, float v) {
    const uint16_t got = ggml_float_to_half(v);
    if ((got & 0x7fff) != 0) {
        printf("  FAIL %-46s %.9g -> %04x (%.9g), expected +/-0\n", what, v, got, ggml_half_to_float(got));
        g_fail++;
    }
}

int main() {
    printf("f32 -> f16: subnormal range\n");

    // --- the boundary between normal and subnormal -------------------------
    // Expectations derived, not recalled: a subnormal f16 is h * 2^-24 with
    // h in [0, 1024), and a normal one is (1 << 10 | m) * 2^(exp - 15) * 2^-10.
    // (My first draft of these constants was wrong in four places - 1e-4 is
    // 1.6384 * 2^-14, i.e. (1<<10)|653 = 0x68d, not 0x1a8c - so the derivation
    // is spelled out rather than left as magic numbers.)
    expect_bits("smallest normal 2^-14", 6.103515625e-05f, 0x0400);
    expect_bits("just below 2^-14 (largest subnormal, h=1023)", 6.1035e-05f, 0x03ff);
    expect_bits("1.5 * 2^-15 (h=768)", 4.57763671875e-05f, 0x0300);
    expect_bits("2^-15 (h=512)", 3.0517578125e-05f, 0x0200);
    expect_bits("2^-16 (h=256)", 1.52587890625e-05f, 0x0100);
    expect_bits("smallest subnormal 2^-24 (h=1)", 5.9604644775390625e-08f, 0x0001);
    // truncating, like the normal path: half of the smallest subnormal has no
    // bits left and must become zero
    expect_zero("just under 2^-24 (truncates)", 2.9802322387695312e-08f);
    expect_zero("2^-25", 2.9802322387695312e-08f);
    expect_zero("1e-30", 1e-30f);
    expect_zero("f32 subnormal", std::numeric_limits<float>::denorm_min());
    expect_zero("+0", 0.0f);
    expect_zero("-0", -0.0f);
    // sign must survive into the subnormal range
    expect_bits("-1.5 * 2^-15 (sign | h=768)", -4.57763671875e-05f, 0x8300);

    // --- the normal range must be unchanged (truncation, no rounding) ------
    expect_bits("1.0", 1.0f, 0x3c00);
    expect_bits("-2.0", -2.0f, 0xc000);
    expect_bits("65504 (max f16)", 65504.0f, 0x7bff);
    expect_bits("1/3 (mantissa truncated, not rounded)", 1.0f / 3.0f, 0x3555);
    expect_bits("1e-4 (1.6384 * 2^-14 -> (1<<10)|653)", 1e-4f, 0x068d);
    expect_bits("overflow -> inf", 1e30f, 0x7c00);
    expect_bits("negative overflow -> -inf", -1e30f, 0xfc00);
    expect_bits("+inf", std::numeric_limits<float>::infinity(), 0x7c00);

    // --- the actual regression: nothing in [2^-24, 2^-14) may become zero --
    // This is the property the defect violated, asserted over the whole range
    // rather than at sampled points, because a sampled test would miss a
    // boundary that moved.
    int nonzero = 0, total = 0;
    for (int e = -24; e <= -15; e++) {          // 2^-24 .. just under 2^-14
        for (int m = 1; m < 8; m++) {
            const float v = (float)(m * std::pow(2.0, e));
            const uint16_t h = ggml_float_to_half(v);
            total++;
            if ((h & 0x7fff) != 0) {
                nonzero++;
            }
            // round-trip within the truncation error of the subnormal grid
            const float back = ggml_half_to_float(h);
            const float err = std::fabs(back - v) / v;
            if (err > 0.07) {
                printf("  FAIL subnormal round-trip %.9g -> %04x -> %.9g (rel %.4f)\n", v, h, back, err);
                g_fail++;
            }
        }
    }
    if (nonzero != total) {
        printf("  FAIL subnormal range: %d of %d values encoded as zero\n", total - nonzero, total);
        g_fail++;
    }
    printf("  subnormal samples: %d values, %d non-zero encodings\n", total, nonzero);

    // --- the exact value from the real tensor, for the record --------------
    // blk.16.attn_qkv.weight: d = 4.470348e-06 (f16 0x004b), sc = 12.
    {
        const float d = ggml_half_to_float(0x004b);
        const float step = d * 12.0f;
        const uint16_t h = ggml_float_to_half(step);
        const float back = ggml_half_to_float(h);
        if ((h & 0x7fff) == 0 || std::fabs(back - step) / step > 0.01) {
            printf("  FAIL real case d*sc = %.9g -> %04x -> %.9g\n", step, h, back);
            g_fail++;
        } else {
            printf("  real case d*sc = %.9g -> %04x -> %.9g (relative %.2e)\n", step, h, back,
                   std::fabs(back - step) / step);
        }
    }

    if (g_fail) {
        printf("%d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("all f16 conversion checks OK\n");
    return 0;
}