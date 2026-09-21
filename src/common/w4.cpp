#include "w4.h"

#include "quant.h"

#include <cstring>

namespace si {

bool w4_supported(uint32_t ggml_type) {
    // Q4_K: q in [0,15] on a per-32-group grid (d*sc_j, dmin*m_j).
    return ggml_type == 12;
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

bool w4_pack(uint32_t ggml_type, const void * src, int K, int N, w4t & out) {
    if (!src || K <= 0 || N <= 0) {
        return false;
    }
    switch (ggml_type) {
    case 12:
        return pack_q4_K(src, K, N, out);
    default:
        return false;
    }
}

} // namespace si
