// Raw IQ2 GEMV for DeepSeek-V4.1-Flash routed experts.  See ds41_iq2.h for the
// layout, the row mapping and the launcher contract.
//
// Design notes (measured trade-offs are in the probe report):
//   * one 32-lane sub-group per output row, lane = one 8-value codebook group;
//   * the 8x8 IQ2 block is used directly, no expansion pass and no SLM tile,
//     because every dequantized value is consumed exactly once by the dot;
//   * the codebook is a 14.5 KB global-memory blob (not a constexpr array):
//     the lanes index it independently and divergent constant-cache accesses
//     would serialize;
//   * an M-row batch reuses each weight load across all M rows; the weights
//     are the scarce stream (~0.29 B/value) while x is shared/L1-resident.
#include "ds41_iq2.h"

#include <stdexcept>

namespace si {

using namespace sycl;

namespace {

// IEEE binary16 -> binary32, including the subnormal range.  Same value as
// src/common/quant.h's ggml_half_to_float; kept local so this TU does not pull
// quant.h's host helpers into the device image.
inline float iq2_h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t e = 127 - 15 + 1;
            while (!(mant & 0x400u)) {
                mant <<= 1;
                e--;
            }
            mant &= 0x3FFu;
            bits = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

// MMAX is the compile-time batch bound (1, 2, 4 or 8); M is the live count.
// Variant 0 (default): lane = 8-value codebook group.  Its activation loads
// are strided by 32 bytes, but the 8 values of a group share one codebook
// lookup and one sub-block scale, which measured faster than the coalesced
// value-per-lane mapping below on all three IQ2 types.
template <uint32_t QT, int MMAX>
void iq2_gemv_group_impl(queue & q, const char * wbase, int K, int N, int M, int x_stride, int y_stride,
                         const float * dx, float * dy, const ds41_iq2_tables * tab) {
    constexpr int BB = (QT == 16) ? 66 : ((QT == 17) ? 74 : 82);
    constexpr int ROWS = kDs41Iq2RowsPerWg;
    constexpr int WG = ROWS * 32;
    const int nblocks = K / kDs41Iq2QK;
    const int row_bytes = nblocks * BB;
    const int n_wg = (N + ROWS - 1) / ROWS;

    q.submit([&](handler & h) {
        h.parallel_for(nd_range<1>((size_t)n_wg * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int tid = (int)it.get_local_id(0);
            const int lane = tid & 31;
            const int sg = tid >> 5;
            const int n = (int)it.get_group(0) * ROWS + sg;
            if (n >= N) {
                return;
            }
            const char * wrow = wbase + (size_t)n * row_bytes;
            // Each 256-value super-block is 32 groups of 8 values, so lane l
            // owns the 8 values [8l, 8l+8) of every super-block.
            const int ib = lane >> 2; // 32-value sub-block (0..7)
            const int l = lane & 3;   // 8-value group inside the sub-block

            float acc[MMAX];
#pragma unroll
            for (int m = 0; m < MMAX; ++m) {
                acc[m] = 0.f;
            }
            const float * xp[MMAX];
#pragma unroll
            for (int m = 0; m < MMAX; ++m) {
                xp[m] = dx + (size_t)m * x_stride;
            }

            for (int sb = 0; sb < nblocks; ++sb) {
                const char * bp = wrow + (size_t)sb * BB;
                const float d = iq2_h2f(*reinterpret_cast<const uint16_t *>(bp));
                const uint8_t * qs = reinterpret_cast<const uint8_t *>(bp) + 2;

                int gi;
                uint8_t signs;
                float db;
                if constexpr (QT == 16) { // IQ2_XXS
                    // 8 bytes per sub-block: 4 grid bytes, then 4 bytes packed
                    // as 4 x 7-bit signs + a 4-bit sub-block scale.  Load the
                    // two words whole: byte-at-a-time assembly emitted five
                    // loads per lane and dominated this type's time.
                    uint32_t gw;
                    uint32_t sw;
                    // A 66-byte block makes these two words 2-byte aligned on
                    // even super-blocks and 4-byte aligned on odd ones.  sb is
                    // warp-uniform, so this branch is not divergent.  The
                    // unaligned single 32-bit loads measured 76 us on the
                    // 2x2304 x 5120 cell; two aligned halves per word measure
                    // 45.7 and this parity split 40.3.
                    if (sb & 1) {
                        __builtin_memcpy(&gw, qs + 8 * ib, 4);
                        __builtin_memcpy(&sw, qs + 8 * ib + 4, 4);
                    } else {
                        uint16_t g0, g1, s0, s1;
                        __builtin_memcpy(&g0, qs + 8 * ib + 0, 2);
                        __builtin_memcpy(&g1, qs + 8 * ib + 2, 2);
                        __builtin_memcpy(&s0, qs + 8 * ib + 4, 2);
                        __builtin_memcpy(&s1, qs + 8 * ib + 6, 2);
                        gw = (uint32_t)g0 | ((uint32_t)g1 << 16);
                        sw = (uint32_t)s0 | ((uint32_t)s1 << 16);
                    }
                    gi = (int)((gw >> (8 * l)) & 0xffu);
                    signs = tab->signs[(sw >> (7 * l)) & 127u];
                    db = d * (0.5f + (float)(sw >> 28)) * 0.25f;
                } else if constexpr (QT == 17) { // IQ2_XS
                    // One u16 per group: 9-bit grid index, 7-bit sign index.
                    // 8 sub-block scale bytes follow the 64 code bytes.
                    uint16_t code;
                    __builtin_memcpy(&code, qs + 2 * (4 * ib + l), 2);
                    gi = code & 511;
                    signs = tab->signs[code >> 9];
                    const uint8_t sc = qs[64 + ib];
                    const int nib = (l >> 1) ? (sc >> 4) : (sc & 0xf);
                    db = d * (0.5f + (float)nib) * 0.25f;
                } else { // QT == 22, IQ2_S
                    // 4 low grid bytes + 2 high bits per group, then 4 sign
                    // bytes per sub-block, qh[8], scales[8].
                    const uint8_t qhb = qs[64 + ib];
                    gi = (int)qs[4 * ib + l] | (((int)qhb << (8 - 2 * l)) & 0x300);
                    signs = qs[32 + 4 * ib + l];
                    const uint8_t sc = qs[64 + 8 + ib];
                    const int nib = (l >> 1) ? (sc >> 4) : (sc & 0xf);
                    db = d * (0.5f + (float)nib) * 0.25f;
                }

                const uint64_t g =
                    (QT == 16) ? tab->xxs[gi] : ((QT == 17) ? tab->xs[gi] : tab->s[gi]);
                const int xoff = sb * kDs41Iq2QK + 8 * lane;

                // The lane's 8 grid values and signs do not depend on the
                // batch row, so decode them once and reuse them across M.
                float wv[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float v = (float)((g >> (8 * j)) & 0xFFu);
                    wv[j] = (signs & (1u << j)) ? -v : v;
                }
                // Two 16-byte activation loads instead of eight scalar loads.
                // At a fixed lane the 8 values are contiguous and 32-byte
                // aligned (xoff is a multiple of 8 floats, x_stride % 4 == 0
                // is enforced by the launcher).
#pragma unroll
                for (int m = 0; m < MMAX; ++m) {
                    if (m < M) {
                        const float * xr = xp[m] + xoff;
                        const float4 xa = *reinterpret_cast<const float4 *>(xr);
                        const float4 xb = *reinterpret_cast<const float4 *>(xr + 4);
                        float a = wv[0] * xa.x();
                        a = sycl::fma(wv[1], xa.y(), a);
                        a = sycl::fma(wv[2], xa.z(), a);
                        a = sycl::fma(wv[3], xa.w(), a);
                        a = sycl::fma(wv[4], xb.x(), a);
                        a = sycl::fma(wv[5], xb.y(), a);
                        a = sycl::fma(wv[6], xb.z(), a);
                        a = sycl::fma(wv[7], xb.w(), a);
                        // One db multiply per group instead of one per value:
                        // db is constant for a lane inside a super-block.
                        acc[m] = sycl::fma(db, a, acc[m]);
                    }
                }
            }

            const sub_group sgv = it.get_sub_group();
#pragma unroll
            for (int m = 0; m < MMAX; ++m) {
                if (m < M) {
                    float v = acc[m];
#pragma unroll
                    for (int mask = 16; mask > 0; mask >>= 1) {
                        v += sycl::permute_group_by_xor(sgv, v, mask);
                    }
                    if (lane == 0) {
                        dy[(size_t)m * y_stride + n] = v;
                    }
                }
            }
        });
    });
}

// Variant 1 (A/B only): lane = one value inside each 32-value sub-block.
//
//   lane l -> group g = l>>3, element e = l&7, at value 32*b + l of sub-block b.
//
// Every activation load is a fully coalesced 128-byte warp read and the
// per-sub-block code/sign/scale words are a lane-uniform broadcast, but the
// mapping needs one codebook lookup per lane per 32-value sub-block (8x more
// than variant 0) and an extra multiply per value.  Measured 1.6-3x SLOWER on
// all three types (see the probe report), so it is kept only for the A/B.
template <uint32_t QT, int MMAX>
void iq2_gemv_value_impl(queue & q, const char * wbase, int K, int N, int M, int x_stride, int y_stride,
                         const float * dx, float * dy, const ds41_iq2_tables * tab) {
    constexpr int BB = (QT == 16) ? 66 : ((QT == 17) ? 74 : 82);
    constexpr int ROWS = kDs41Iq2RowsPerWg;
    constexpr int WG = ROWS * 32;
    const int nblocks = K / kDs41Iq2QK;
    const int row_bytes = nblocks * BB;
    const int n_wg = (N + ROWS - 1) / ROWS;

    q.submit([&](handler & h) {
        h.parallel_for(nd_range<1>((size_t)n_wg * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int tid = (int)it.get_local_id(0);
            const int lane = tid & 31;
            const int sg = tid >> 5;
            const int n = (int)it.get_group(0) * ROWS + sg;
            if (n >= N) {
                return;
            }
            const char * wrow = wbase + (size_t)n * row_bytes;
            const int g = lane >> 3; // group 0..3 inside a 32-value sub-block
            const int e = lane & 7;  // element 0..7 inside the group

            float acc[MMAX];
#pragma unroll
            for (int m = 0; m < MMAX; ++m) {
                acc[m] = 0.f;
            }
            const float * xp[MMAX];
#pragma unroll
            for (int m = 0; m < MMAX; ++m) {
                xp[m] = dx + (size_t)m * x_stride;
            }

            for (int sb = 0; sb < nblocks; ++sb) {
                const char * bp = wrow + (size_t)sb * BB;
                const float d = iq2_h2f(*reinterpret_cast<const uint16_t *>(bp));
                const uint8_t * qs = reinterpret_cast<const uint8_t *>(bp) + 2;
#pragma unroll
                for (int b = 0; b < 8; ++b) {
                    int gi;
                    uint8_t signs;
                    float db;
                    if constexpr (QT == 16) { // IQ2_XXS
                        // These two words are lane-uniform: every lane reads
                        // the same sub-block's code/sign/scale.
                        uint32_t gw;
                        uint32_t sw;
                        __builtin_memcpy(&gw, qs + 8 * b, 4);
                        __builtin_memcpy(&sw, qs + 8 * b + 4, 4);
                        gi = (int)((gw >> (8 * g)) & 0xffu);
                        signs = tab->signs[(sw >> (7 * g)) & 127u];
                        db = d * (0.5f + (float)(sw >> 28)) * 0.25f;
                    } else if constexpr (QT == 17) { // IQ2_XS
                        uint16_t code;
                        __builtin_memcpy(&code, qs + 2 * (4 * b + g), 2);
                        gi = code & 511;
                        signs = tab->signs[code >> 9];
                        const uint8_t sc = qs[64 + b]; // lane-uniform
                        const int nib = (g >> 1) ? (sc >> 4) : (sc & 0xf);
                        db = d * (0.5f + (float)nib) * 0.25f;
                    } else { // QT == 22, IQ2_S
                        const uint8_t qhb = qs[64 + b]; // lane-uniform
                        gi = (int)qs[4 * b + g] | (((int)qhb << (8 - 2 * g)) & 0x300);
                        signs = qs[32 + 4 * b + g];
                        const uint8_t sc = qs[64 + 8 + b]; // lane-uniform
                        const int nib = (g >> 1) ? (sc >> 4) : (sc & 0xf);
                        db = d * (0.5f + (float)nib) * 0.25f;
                    }

                    const uint64_t gv =
                        (QT == 16) ? tab->xxs[gi] : ((QT == 17) ? tab->xs[gi] : tab->s[gi]);
                    float iv = (float)((gv >> (8 * e)) & 0xFFu);
                    if (signs & (1u << e)) {
                        iv = -iv;
                    }
                    const float wv = db * iv;
                    const int xoff = sb * kDs41Iq2QK + 32 * b + lane;
#pragma unroll
                    for (int m = 0; m < MMAX; ++m) {
                        if (m < M) {
                            acc[m] = sycl::fma(wv, xp[m][xoff], acc[m]);
                        }
                    }
                }
            }

            const sub_group sgv = it.get_sub_group();
#pragma unroll
            for (int m = 0; m < MMAX; ++m) {
                if (m < M) {
                    float v = acc[m];
#pragma unroll
                    for (int mask = 16; mask > 0; mask >>= 1) {
                        v += sycl::permute_group_by_xor(sgv, v, mask);
                    }
                    if (lane == 0) {
                        dy[(size_t)m * y_stride + n] = v;
                    }
                }
            }
        });
    });
}

template <uint32_t QT, int MMAX>
void iq2_dispatch_one(queue & q, const char * wbase, int K, int N, int M, int x_stride, int y_stride,
                      const float * dx, float * dy, const ds41_iq2_tables * tab, int variant) {
    if (variant == 1) {
        iq2_gemv_value_impl<QT, MMAX>(q, wbase, K, N, M, x_stride, y_stride, dx, dy, tab);
    } else {
        iq2_gemv_group_impl<QT, MMAX>(q, wbase, K, N, M, x_stride, y_stride, dx, dy, tab);
    }
}

template <uint32_t QT>
void iq2_dispatch_m(queue & q, const char * wbase, int K, int N, int M, int x_stride, int y_stride,
                    const float * dx, float * dy, const ds41_iq2_tables * tab, int variant) {
    if (M <= 1) {
        iq2_dispatch_one<QT, 1>(q, wbase, K, N, M, x_stride, y_stride, dx, dy, tab, variant);
    } else if (M <= 2) {
        iq2_dispatch_one<QT, 2>(q, wbase, K, N, M, x_stride, y_stride, dx, dy, tab, variant);
    } else if (M <= 4) {
        iq2_dispatch_one<QT, 4>(q, wbase, K, N, M, x_stride, y_stride, dx, dy, tab, variant);
    } else {
        iq2_dispatch_one<QT, 8>(q, wbase, K, N, M, x_stride, y_stride, dx, dy, tab, variant);
    }
}

} // namespace

void ds41_iq2_gemv_launch(queue & q, const void * w_row_base, int K, int N_out, uint32_t type, int expert,
                          size_t plane_bytes, const float * d_x, float * d_y, int M, int x_stride, int y_stride,
                          const ds41_iq2_tables * d_tables, int variant) {
    if (K <= 0 || N_out <= 0) {
        return;
    }
    if (K % kDs41Iq2QK != 0) {
        throw std::invalid_argument("ds41_iq2: K must be a multiple of 256");
    }
    const size_t bb = ds41_iq2_block_bytes(type);
    if (bb == 0) {
        throw std::invalid_argument("ds41_iq2: type is not IQ2_XXS/IQ2_XS/IQ2_S");
    }
    if (d_tables == nullptr) {
        throw std::invalid_argument("ds41_iq2: a device ds41_iq2_tables blob is required");
    }
    if (M < 1) {
        M = 1;
    }
    if (M > 8) {
        throw std::invalid_argument("ds41_iq2: M must be <= 8");
    }
    if (x_stride == 0) {
        x_stride = K;
    }
    if (y_stride == 0) {
        y_stride = N_out;
    }
    if (x_stride % 4 != 0) {
        // The activation path loads two float4 per lane.  x_stride defaults to
        // K, which is a multiple of 256.
        throw std::invalid_argument("ds41_iq2: x_stride must be a multiple of 4");
    }
    const size_t row_bytes = (size_t)(K / kDs41Iq2QK) * bb;
    if (plane_bytes == 0) {
        plane_bytes = row_bytes * (size_t)N_out;
    }
    const char * w = (const char *)w_row_base + (size_t)expert * plane_bytes;

    switch (type) {
    case 16: iq2_dispatch_m<16>(q, w, K, N_out, M, x_stride, y_stride, d_x, d_y, d_tables, variant); break;
    case 17: iq2_dispatch_m<17>(q, w, K, N_out, M, x_stride, y_stride, d_x, d_y, d_tables, variant); break;
    case 22: iq2_dispatch_m<22>(q, w, K, N_out, M, x_stride, y_stride, d_x, d_y, d_tables, variant); break;
    default: break;
    }
}

} // namespace si
