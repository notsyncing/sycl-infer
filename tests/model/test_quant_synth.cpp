// Synthetic block dequantization for the formats whose layout is easy to build
// by hand (Q8_0, Q5_0, Q5_1, F16, BF16), plus the row-size helpers and the SIn
// packer's rejection rules.  Hermetic: no GGUF, no GPU.
//
// Every other quant test in this tree needs a real model (and most need the
// 27B), so a bug in a block layout or in the f16 converter would first surface
// as a wrong answer on one machine.  These cases pin the bit patterns directly.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "quant.h"
#include "w8.h"

using namespace si;

static int g_fail = 0;

static void check(bool ok, const char * name) {
    if (ok) {
        printf("  [%s] OK\n", name);
        return;
    }
    g_fail++;
    printf("  [%s] FAILED\n", name);
}

static bool close(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps * std::fmax(1.f, std::fmax(std::fabs(a), std::fabs(b)));
}

// f16 bit patterns used to build a block with d = 1 (0x3C00), 0.5 (0x3800),
// and a subnormal (2^-24 = 0x0001).
static constexpr uint16_t kOne = 0x3C00;

int main() {
    printf("Q8_0\n");
    {
        block_q8_0 b{};
        b.d = kOne;
        for (int i = 0; i < QK8_0; i++) {
            b.qs[i] = (int8_t)(i - 16);
        }
        float y[QK8_0];
        dequantize_row(8, &b, y, QK8_0);
        bool ok = true;
        for (int i = 0; i < QK8_0; i++) {
            ok = ok && y[i] == (float)(i - 16);
        }
        check(ok, "Q8_0 values are d*q");
        check(quant_row_bytes(8, 32) == sizeof(block_q8_0), "Q8_0 row bytes");
    }

    printf("Q5_0\n");
    {
        // All-low-nibble 0x0F is q=15, signed -1.  qh bit 0 selects element 0
        // and bit 16 selects element 16 (the second half shifts by 12 and
        // masks 0x10, i.e. original bit 16), so this pins both shift paths.
        block_q5_0 b{};
        b.d = kOne;
        std::memset(b.qs, 0xFF, sizeof(b.qs));
        uint32_t qh = 1u | (1u << 16);
        std::memcpy(b.qh, &qh, sizeof(qh));
        float y[QK8_0];
        dequantize_row(6, &b, y, QK8_0);
        check(y[0] == 15.f, "Q5_0 high bit 0");
        check(y[16] == 15.f, "Q5_0 high bit 16");
        check(y[1] == -1.f && y[15] == -1.f && y[17] == -1.f, "Q5_0 low nibble signed");
        check(quant_row_bytes(6, 32) == sizeof(block_q5_0), "Q5_0 row bytes");
    }

    printf("Q5_1\n");
    {
        // The interleaved-halves layout: elements 0..15 are the low nibbles of
        // qs[0..15], elements 16..31 the high nibbles.  The old consecutive-pair
        // reading preserved the statistics but produced cos 0.7 on a real expert.
        block_q5_1 b{};
        b.d = kOne;
        b.m = 0x3800; // 0.5
        b.qs[0] = 0x01; // low nibble -> y[0], high nibble -> y[16]
        b.qh[0] = 0x10; // bit 4 -> y[4]
        float y[QK8_0];
        dequantize_row(7, &b, y, QK8_0);
        check(close(y[0], 1.5f), "Q5_1 low half + min");
        check(close(y[16], 0.5f), "Q5_1 high half + min");
        check(close(y[4], 16.5f), "Q5_1 high bit + min");
        check(quant_row_bytes(7, 32) == sizeof(block_q5_1), "Q5_1 row bytes");
    }

    printf("F16 / BF16\n");
    {
        const uint16_t h[3] = {kOne, 0x3800, 0x0001};
        float y[3];
        dequantize_row(1, h, y, 3);
        check(y[0] == 1.f && y[1] == 0.5f, "F16 normal values");
        check(close(y[2], std::ldexp(1.f, -24)), "F16 subnormal is not flushed to zero");

        const uint16_t bf[2] = {0x3F80, 0xBF80}; // 1.0, -1.0
        float z[2];
        dequantize_row(30, bf, z, 2);
        check(z[0] == 1.f && z[1] == -1.f, "BF16 is the top half of f32");
    }

    printf("row bytes\n");
    check(quant_row_bytes(12, 256) == sizeof(block_q4_K), "Q4_K row bytes");
    check(quant_row_bytes(13, 256) == sizeof(block_q5_K), "Q5_K row bytes");
    check(quant_row_bytes(14, 256) == sizeof(block_q6_K), "Q6_K row bytes");
    // KK-quants are per 256 values; a shorter row is the caller's error, but the
    // helper must not silently invent a larger block count.
    check(quant_row_bytes(12, 32) == 0, "Q4_K partial row reports zero bytes");

    printf("SIn packer guards\n");
    {
        uint8_t vals[64] = {};
        uint32_t meta[4] = {};
        // No type is a 32-value format that the packer accepts.
        check(!w8_repack(0, vals, 32, 1, vals, meta, false), "unsupported type rejected");
        // K must be a multiple of the 32-value group.
        block_q4_K q{};
        check(!w8_repack(12, &q, 33, 1, vals, meta, false), "K%32 rejected");
        // Byte-plane sizes agree with 128-row blocking and the group width.
        check(w8_vals_bytes(12, 256, 1) == (size_t)(256 / 32) * 16, "Q4_K vals bytes, N=1");
        check(w8_vals_bytes(12, 256, 129) == (size_t)129 * 8 * 16, "Q4_K vals bytes scale with N");
        check(w8_meta_count(12, 256, 129) == (size_t)129 * 8, "Q4_K meta entries scale with N");
        check(w8_meta_bytes(14, 256, 1, /*scale_only=*/true) == (size_t)(256 / 16) * 2,
              "Q6_K meta is scale-only (2 B/group)");
    }

    printf(g_fail ? "quant synth test FAILED (%d)\n" : "quant synth test OK\n", g_fail);
    return g_fail ? 1 : 0;
}
