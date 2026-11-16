// Hermetic test of the SIn weight layout and its slab copy.
//
// w8 is blocked over kRB (128) rows: group g of row r lives at
// ((rb*MG + g)*kRB + ri).  A tensor whose row count is not a multiple of 128
// therefore pads the last block, and the allocation must be padded too; the
// per-row stride used by the slab copy is NOT the padded allocation size of a
// one-row tensor.  Mixing the two either writes past the allocation or copies a
// transposed prefix, and both look like "the model's logits moved".
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "quant.h"
#include "w8.h"

using namespace si;

static int g_fail = 0;
static void bad(const char * what) {
    std::printf("test_w8_layout: FAIL (%s)\n", what);
    g_fail++;
}

int main() {
    const int K = 512, N = 200; // not a multiple of kRB
    const int G = 32, MG = K / G;
    const int gb = w8_group_bytes(12);
    const size_t row_bytes = quant_row_bytes(12, K);

    std::vector<block_q4_K> src((size_t)N * (K / QK_K));
    for (size_t n = 0; n < src.size(); n++) {
        src[n].d = ggml_float_to_half(0.004f * (float)((n % 5) + 1));
        src[n].dmin = ggml_float_to_half(0.001f * (float)((n % 3) + 1));
        for (int i = 0; i < K_SCALE_SIZE; i++) src[n].scales[i] = (uint8_t)(0x11 + 7 * i + n);
        for (int i = 0; i < QK_K / 2; i++) src[n].qs[i] = (uint8_t)((n * 37 + i * 11) & 0xFF);
    }

    // The layout invariant the allocation is built on.
    if (w8_vals_bytes(12, K, kRB) != w8_row_vals_bytes(12, K) * (size_t)kRB) {
        bad("w8_vals_bytes(128) != 128 * row stride");
    }
    if (w8_vals_bytes(12, K, N) != w8_row_vals_bytes(12, K) * (size_t)((N + kRB - 1) / kRB * kRB)) {
        bad("padded allocation size");
    }

    std::vector<uint8_t> vals(w8_vals_bytes(12, K, N));
    std::vector<uint32_t> meta((w8_meta_bytes(12, K, N) + 3) / 4);
    const size_t row_vals = w8_row_vals_bytes(12, K);
    const size_t row_meta = w8_row_meta_bytes(12, K);
    // The staging buffer is sized to the padded slab (production uses a
    // 4096-row slab; here the whole tensor is one slab).
    const int cp = (N + kRB - 1) / kRB * kRB;
    std::vector<uint8_t> hv(row_vals * (size_t)cp, 0);
    std::vector<uint8_t> hm(row_meta * (size_t)cp, 0);
    if (!w8_repack(12, src.data(), K, N, hv.data(), reinterpret_cast<uint32_t *>(hm.data()))) {
        bad("w8_repack");
        return 1;
    }
    // The slab copy build_w8_tensor performs for a last partial slab.
    if ((size_t)cp * row_vals > vals.size()) {
        bad("slab copy exceeds the padded allocation");
    } else {
        std::memcpy(vals.data(), hv.data(), (size_t)cp * row_vals);
        std::memcpy((uint8_t *)meta.data(), hm.data(), (size_t)cp * row_meta);
    }

    // Read every row back through the blocked layout and compare with the
    // source dequantization (the only error is f16 metadata rounding).
    double max_abs = 0, max_w = 0;
    std::vector<float> w(K);
    for (int r = 0; r < N; r++) {
        dequantize_row(12, (const char *)src.data() + (size_t)r * row_bytes, w.data(), K);
        const int rb = r / kRB, ri = r % kRB;
        for (int g = 0; g < MG; g++) {
            const uint8_t * q = vals.data() + ((size_t)(rb * MG + g) * kRB + ri) * gb;
            const uint32_t m = meta[(size_t)(rb * MG + g) * kRB + ri];
            const float step = ggml_half_to_float((uint16_t)(m & 0xFFFF));
            const float off = ggml_half_to_float((uint16_t)(m >> 16));
            for (int i = 0; i < 32; i++) {
                const int byte = (i & 7) + (i / 16) * 8;
                const int qv = (i & 8) ? ((q[byte] >> 4) & 0xF) : (q[byte] & 0xF);
                const float rec = step * (float)qv - off;
                const float ref = w[(size_t)g * 32 + i];
                max_abs = std::max(max_abs, (double)std::fabs(rec - ref));
                max_w = std::max(max_w, (double)std::fabs(ref));
            }
        }
    }
    const bool ok = max_abs <= 1e-3 * std::max(1.0, max_w);
    std::printf("w8 layout N=%d K=%d: max|exp|=%.4f max|diff|=%.3e %s\n", N, K, max_w, max_abs,
                ok ? "OK" : "FAIL");
    if (!ok) g_fail++;
    if (g_fail) {
        return 1;
    }
    std::printf("test_w8_layout: all checks OK\n");
    return 0;
}
