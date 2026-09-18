// CPU SIn int8 GEMV/GEMM (mirror of src/backend/gpu/kernels/dp4a_gemv.cpp and
// dp4a_gemm.cpp).  The packed group is expanded to activation-ordered unsigned
// bytes (the same w8_xword permutation the device kernels use) and dotted with
// the quantized activation by the ISA-dispatched integer dot, so AVX-VNNI /
// AVX-512-VNNI run a single dpbusd per 32 lanes.
#include "common.h"

#include "w8.h"

namespace si {

namespace {

uint32_t spread1(uint32_t x) {
    return (x & 1u) | ((x & 2u) << 7) | ((x & 4u) << 14) | ((x & 8u) << 21);
}
uint32_t spread2(uint32_t x) {
    return (x & 3u) | ((x & 12u) << 6) | ((x & 48u) << 12) | ((x & 192u) << 18);
}
void nib_half(const uint8_t * p, uint32_t * nw) {
    const uint32_t lo = *reinterpret_cast<const uint32_t *>(p);
    const uint32_t hi = *reinterpret_cast<const uint32_t *>(p + 4);
    nw[0] = lo & 0x0F0F0F0Fu;
    nw[1] = (lo >> 4) & 0x0F0F0F0Fu;
    nw[2] = hi & 0x0F0F0F0Fu;
    nw[3] = (hi >> 4) & 0x0F0F0F0Fu;
}
void expand_half(uint32_t qt, const uint8_t * p, int h, uint32_t * nw) {
    nib_half(p + h * 8, nw);
    if (qt == 13) {
        const uint32_t pl = *reinterpret_cast<const uint32_t *>(p + 16) >> (16 * h);
        nw[0] |= spread1(pl & 0xFu) << 4;
        nw[1] |= spread1((pl >> 8) & 0xFu) << 4;
        nw[2] |= spread1((pl >> 4) & 0xFu) << 4;
        nw[3] |= spread1((pl >> 12) & 0xFu) << 4;
    } else if (qt == 14) {
        const uint32_t pl = *reinterpret_cast<const uint32_t *>(p + 8);
        nw[0] |= spread2(pl & 0xFFu) << 4;
        nw[1] |= spread2((pl >> 16) & 0xFFu) << 4;
        nw[2] |= spread2((pl >> 8) & 0xFFu) << 4;
        nw[3] |= spread2((pl >> 24) & 0xFFu) << 4;
    }
}
// Expand a packed group into unsigned bytes in *activation order* (0..15 /
// 0..31 / 0..63).  The word expansion produces value sets {0..3, 8..11, 4..7,
// 12..15} per 16-value half; w8_xword() is the word->activation permutation
// the device kernels use, so the integer dot can run straight down the bytes.
int expand_group(uint32_t type, const uint8_t * p, uint8_t * out) {
    const int gs = w8_group_size(type);
    uint32_t w[8];
    expand_half(type, p, 0, w);
    if (type != 14) {
        expand_half(type, p, 1, w + 4);
    }
    const int words = gs / 4;
    for (int j = 0; j < words; j++) {
        const int jj = j % 4, half = j / 4;
        const int xw = (jj == 1) ? 2 : ((jj == 2) ? 1 : jj);
        const int base = (half * 4 + xw) * 4;
        for (int b = 0; b < 4; b++) {
            out[base + b] = (uint8_t)((w[j] >> (8 * b)) & 0xFF);
        }
    }
    return gs;
}
inline float h2f(uint16_t h) {
    return ggml_half_to_float(h);
}
void w8_meta(const w8t & w, size_t idx, float & sw, float & mw) {
    if (w.type == 14 && w.meta_elem == 2) {
        sw = h2f(((const uint16_t *)w.meta)[idx]);
        mw = 32.0f * sw;
        return;
    }
    const uint32_t m = w.meta[idx];
    sw = h2f((uint16_t)(m & 0xFFFF));
    mw = h2f((uint16_t)(m >> 16));
}
// index of a group's packed bytes for a given row (SIn blocked layout)
inline const uint8_t * w8_row_group(const w8t & w, int row, int g) {
    const int rb = row / kRB, ri = row % kRB;
    const int MG = w.K / w8_group_size(w.type);
    return w.vals + ((size_t)(rb * MG + g) * kRB + ri) * w8_group_bytes(w.type);
}
// integer dot of one weight group against the matching activation bytes
// (AVX-VNNI / AVX-512 VNNI when available, AVX2 otherwise)
int32_t w8_group_dot(uint32_t type, const uint8_t * wbytes, const int8_t * xg) {
    uint8_t wb[32];
    const int gs = expand_group(type, wbytes, wb);
    return isa().dot_i8(wb, xg, gs);
}
inline void w8_meta_at(const w8t & w, int row, int g, float & sw, float & mw) {
    const int MG = w.K / w8_group_size(w.type);
    const size_t idx = (size_t)((row / kRB) * MG + g) * kRB + (row % kRB);
    w8_meta(w, idx, sw, mw);
}
// sum over groups of x*w for one (token, row); x was quantized by cpu_xq
float dp4a_row(const cpu_gemv_seg & seg, int TB, int t, int row) {
    const w8t & w = seg.w8;
    const int gs = w8_group_size(w.type);
    const int MG = w.K / gs;
    float total = 0.0f;
    for (int g = 0; g < MG; g++) {
        // Q6 weight groups are 16 wide; activation groups are always 32
        const int g32 = (gs == 32) ? g : (g >> 1);
        const int half = (gs == 32) ? 0 : (g & 1);
        const size_t aidx = (size_t)g32 * TB + t;
        const int8_t * xg = seg.x8 + aidx * 32 + (half ? 16 : 0);
        const float sx = seg.xmeta[aidx * 2];
        float sw, mw;
        w8_meta_at(w, row, g, sw, mw);
        const int32_t wd = w8_group_dot(w.type, w8_row_group(w, row, g), xg);
        const int32_t xsum = (gs == 32) ? (seg.xsumq[aidx * 2] + seg.xsumq[aidx * 2 + 1]) : seg.xsumq[aidx * 2 + half];
        total += sx * (sw * (float)wd - mw * (float)xsum);
    }
    return total;
}

} // namespace

void cpu_dp4a_gemv(const cpu_gemv_seg & seg, int n_threads) {
    (void)n_threads;
    const w8t & w = seg.w8;
    if (!w.vals) {
        return;
    }
    par(w.N, [&](int row) {
        float v = seg.alpha * dp4a_row(seg, 1, 0, row);
        if (seg.residual) {
            v += seg.residual[row];
        }
        seg.out[row] = v;
    });
}

void cpu_dp4a_gemm(const cpu_gemv_seg & seg, int TB, int n_threads) {
    (void)n_threads;
    const w8t & w = seg.w8;
    if (!w.vals) {
        return;
    }
    const int rows = w.N;
    par(TB * rows, [&](int j) {
        const int t = j / rows;
        const int row = j % rows;
        float v = seg.alpha * dp4a_row(seg, TB, t, row);
        if (seg.residual) {
            v += seg.residual[(size_t)t * seg.out_stride + row];
        }
        seg.out[(size_t)t * seg.out_stride + row] = v;
    });
}

} // namespace si
