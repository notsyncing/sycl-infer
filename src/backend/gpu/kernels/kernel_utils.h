#pragma once
// ---------------------------------------------------------------------------
// Shared device-side helpers for the SYCL kernels in src/kernels.
//
// These are the pieces every kernel needs: vector load/store, KV cache element
// access for every storage dtype, GGUF/SIn weight dequantization, sub-group
// reductions and the DP4A (SIn) weight expansion.  They were the
// anonymous-namespace prologue of the old monolithic kernels.cpp; each kernel
// translation unit now includes this header and pulls the names in with
// `using namespace si::kd`.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp> // IWYU pragma: keep

#include "dp4a.h" // IWYU pragma: keep (re-exported for the dp4a GEMV/GEMM TUs)
#include "kernels.h"
#include "quant.h"
#include "w8.h"

namespace si {
namespace kd {

using namespace sycl;

inline float silu_f(float x) {
    return x / (1.0f + sycl::exp(-x));
}
inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + sycl::exp(-x));
}

// vector load/store through a raw pointer with a known alignment.  Without the
// hint the compiler must assume the pointer is only element-aligned and splits
// (e.g.) a uint4 SLM access into four 4-byte messages.
template <typename V, typename T> inline V vload(const T * p) {
    return *reinterpret_cast<const V *>(__builtin_assume_aligned(static_cast<const void *>(p), alignof(V)));
}
template <typename V, typename T> inline void vstore(T * p, const V & v) {
    *reinterpret_cast<V *>(__builtin_assume_aligned(static_cast<void *>(p), alignof(V))) = v;
}
// ---- KV storage element helpers (see kv_dtype_t in kernels.h) -------------
// The kernels are compiled once per storage type and dispatched at launch
// time; the math inside is always fp32.
template <typename KV> inline float kv_ld(const KV * p) {
    return (float)*p;
}
template <> inline float kv_ld<float>(const float * p) {
    return *p;
}
template <typename KV> inline void kv_st(KV * p, float v) {
    *p = (KV)v;
}

// load 4 contiguous KV elements as float4 (used by the vectorized attention
// path); the pool layout keeps these 4-element groups 8/16-byte aligned
template <typename KV> inline sycl::float4 kv_ld4(const KV * p) {
    if constexpr (std::is_same_v<KV, float>) {
        return vload<sycl::float4, float>(p);
    } else {
        const sycl::vec<KV, 4> v = vload<sycl::vec<KV, 4>, KV>(p);
        return sycl::float4((float)v[0], (float)v[1], (float)v[2], (float)v[3]);
    }
}

// ---- int8 KV geometry (see kv_dtype_t in kernels.h) ----------------------
// Per (KV block, kv head) the pool holds [kBlockSize rows of head_dim int8]
// followed by [kBlockSize rows of head_dim/32 fp16 scales], so one unit is
// self-contained and the addressing does not depend on the pool size.
inline size_t kv_i8_scale_n(int head_dim) {
    return (size_t)head_dim / kI8Q;
}
inline sycl::float4 i8_ld4(const int8_t * p) {
    const sycl::vec<int8_t, 4> v = *reinterpret_cast<const sycl::vec<int8_t, 4> *>(p);
    return sycl::float4((float)v[0], (float)v[1], (float)v[2], (float)v[3]);
}
// quantize one fp32 value to symmetric int8 with a block scale (round-half-away)
inline int8_t i8_quant(float v, float scale) {
    return (int8_t)sycl::clamp(sycl::round(v / scale), -127.f, 127.f);
}
// row data pointer for (KV block, kv head, token-in-block): the layout is the
// same for every dtype, only the element size changes
template <typename KV> inline const KV * kv_row_data(const KV * base, size_t unit, int ko, int head_dim) {
    return base + (unit * kBlockSize + ko) * head_dim;
}
// i8 only: the fp16 block scales of that row, in the separate scale plane
// (same [block][kv head][token] indexing, head_dim/kI8Q halves per row)
inline const sycl::half * kv_row_scales(const sycl::half * ksc_base, size_t unit, int ko, int head_dim) {
    return ksc_base + (unit * kBlockSize + ko) * kv_i8_scale_n(head_dim);
}

// ---- int4 KV geometry (see kv_dtype_t in kernels.h) -----------------------
// Two signed nibbles per byte (low nibble = even head dim), one fp16 scale per
// 32 head dimensions (the same plane as i8).  A row is head_dim/2 bytes.
inline float i4_cast(uint8_t byte, int hi) {
    const int v = hi ? (byte >> 4) : (byte & 0xF);
    return (float)((v ^ 8) - 8);
}
// unpack one already-masked nibble (low 4 bits) to signed fp32
inline float i4_nib(uint32_t x) {
    const int v = (int)(x & 0xFu);
    return (float)((v ^ 8) - 8);
}
inline float i4_ld(const uint8_t * row, int d) {
    return i4_cast(row[d >> 1], d & 1);
}
// load 4 contiguous i4 dims starting at `d` (d a multiple of 4) as float4.  The
// byte offset d/2 is even, so one aligned 16-bit load gets all four nibbles
// (two scalar byte loads measured slower: the decode kernel is instruction-,
// not byte-bound at long context).
inline sycl::float4 i4_ld4(const uint8_t * row, int d) {
    const uint32_t w = *reinterpret_cast<const uint16_t *>(
        __builtin_assume_aligned(static_cast<const void *>(row + (d >> 1)), 2));
    return sycl::float4(i4_nib(w), i4_nib(w >> 4), i4_nib(w >> 8), i4_nib(w >> 12));
}
// quantize one fp32 value to symmetric int4 with a block scale (round-half-away)
inline int8_t i4_quant(float v, float scale) {
    return (int8_t)sycl::clamp(sycl::round(v / scale), -7.f, 7.f);
}
template <> inline const uint8_t * kv_row_data<uint8_t>(const uint8_t * base, size_t unit, int ko, int head_dim) {
    return base + (unit * kBlockSize + ko) * (size_t)(head_dim / 2);
}

// Real token count of row `r`: mode 2 may have a partial last row; 0 in
// n_real_row means "use the uniform n_real" (mode 0/1, graph replay).
inline int row_nr(const step_info * info, int r) {
    const int nr = info->n_real_row[r];
    return nr > 0 ? nr : info->n_real;
}

inline float fast_h2f(uint16_t h) {
    const uint32_t exp = (h >> 10) & 0x1Fu;
    if (exp == 0 || exp == 31) {
        return ggml_half_to_float(h);
    }
    uint32_t bits = ((uint32_t)(h & 0x8000u) << 16) | ((exp + 112u) << 23) | ((uint32_t)(h & 0x3FFu) << 13);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}
// ---- SIn weight meta (see w8.h) -------------------------------------------
// Q6_K (type 14) groups satisfy min == 32*scale exactly; tensors whose scales
// all survive fp16 store only the fp16 scale (2 bytes per entry, `melem == 2`)
// and the kernels derive the min.  Every other case keeps the 4-byte {fp16
// scale, fp16 min} entry.  `melem` is a per-tensor runtime value so the Q6_K
// dispatch/templates stay unchanged; the branch is loop-invariant.
// unpack one group's meta entry at (meta + idx) into (scale, min)
template <uint32_t QT> inline void w8_sw_mw(const char * meta, size_t idx, int melem, float & sw, float & mw) {
    if constexpr (QT == 14) {
        if (melem == 2) {
            sw = fast_h2f(((const uint16_t *)meta)[idx]);
            mw = 32.0f * sw;
            return;
        }
    }
    const uint32_t m = ((const uint32_t *)meta)[idx];
    sw = fast_h2f((uint16_t)(m & 0xFFFF));
    mw = fast_h2f((uint16_t)(m >> 16));
}

// Q3_K scales are 16 packed 6-bit values; unpack into 16 int8s (ggml layout).
inline void unpack_q3k_scales(const block_q3_K * x, int8_t * scales) {
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    __builtin_memcpy(aux, x->scales, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    __builtin_memcpy(scales, aux, 16);
}

// one element (0..255) of an IQ/Q3_K 256-element super-block; element index
// e = 32*b + lane matches the SIn lane mapping so the lane kernels below can
// share the same indexing.
inline float dequant_elem_iq(uint32_t type, const char * sb, int e) {
    const int b = e / 32, lane = e % 32;
    switch (type) {
    case 11: { // Q3_K
        const auto * x = (const block_q3_K *)sb;
        int8_t scales[16];
        unpack_q3k_scales(x, scales);
        const int half = b / 4, j = b % 4;
        const uint8_t q = x->qs[half * 32 + lane];
        const int val = (int)((q >> (2 * j)) & 3) - ((x->hmask[lane] & (1u << (half * 4 + j))) ? 0 : 4);
        const int is = (half * 4 + j) * 2 + (lane >= 16 ? 1 : 0);
        return fast_h2f(x->d) * (float)(scales[is] - 32) * (float)val;
    }
    case 20: { // IQ4_NL
        const auto * blk = (const block_iq4_nl *)(sb + b * (int)sizeof(block_iq4_nl));
        const uint8_t byte = blk->qs[lane & 15];
        const int nib = (lane < 16) ? (byte & 0xf) : (byte >> 4);
        return fast_h2f(blk->d) * (float)kvalues_iq4nl[nib];
    }
    case 21: { // IQ3_S
        const auto * x = (const block_iq3_s *)sb;
        const int pi = b / 2, part = b % 2;
        const float db = fast_h2f(x->d) * (1.0f + 2.0f * (float)(part ? (x->scales[pi] >> 4) : (x->scales[pi] & 0xf)));
        const int l = lane / 8, k = lane % 8;
        const uint8_t * qs = x->qs + pi * 16 + part * 8;
        const uint8_t * sg = x->signs + pi * 8 + part * 4;
        const int qhb = x->qh[pi * 2 + part];
        int gidx, gi;
        if (k < 4) {
            gidx = qs[2 * l + 0] | ((qhb << (8 - 2 * l)) & 256);
            gi = k;
        } else {
            gidx = qs[2 * l + 1] | ((qhb << (7 - 2 * l)) & 256);
            gi = k - 4;
        }
        const float s = (sg[l] & kmask_iq2xs[k]) ? -1.f : 1.f;
        return db * (float)((const uint8_t *)(iq3s_grid + gidx))[gi] * s;
    }
    case 23: { // IQ4_XS
        const auto * x = (const block_iq4_xs *)sb;
        const int ls = ((x->scales_l[b / 2] >> (4 * (b % 2))) & 0xf) | (((x->scales_h >> (2 * b)) & 3) << 4);
        const uint8_t byte = x->qs[16 * b + (lane & 15)];
        const int nib = (lane < 16) ? (byte & 0xf) : (byte >> 4);
        return fast_h2f(x->d) * (float)(ls - 32) * (float)kvalues_iq4nl[nib];
    }
    default: return 0.f;
    }
}

// typed variant (compile-time dispatch)
template <uint32_t TYPE> inline void dequant_sb_lane_typed(const char * sb, int lane, float w[8]) {
    switch (TYPE) {
    case 11: { // Q3_K
        const auto * x = (const block_q3_K *)sb;
        int8_t scales[16];
        unpack_q3k_scales(x, scales);
        const float d = fast_h2f(x->d);
        const uint8_t hm = x->hmask[lane];
#pragma unroll
        for (int b = 0; b < 8; b++) {
            const int half = b / 4, j = b % 4;
            const uint8_t q = x->qs[half * 32 + lane];
            const int val = (int)((q >> (2 * j)) & 3) - ((hm & (1u << (half * 4 + j))) ? 0 : 4);
            const int is = (half * 4 + j) * 2 + (lane >= 16 ? 1 : 0);
            w[b] = d * (float)(scales[is] - 32) * (float)val;
        }
    } break;
    case 20: { // IQ4_NL: 8 independent 32-element blocks
#pragma unroll
        for (int b = 0; b < 8; b++) {
            const auto * blk = (const block_iq4_nl *)(sb + b * (int)sizeof(block_iq4_nl));
            const uint8_t byte = blk->qs[lane & 15];
            const int nib = (lane < 16) ? (byte & 0xf) : (byte >> 4);
            w[b] = fast_h2f(blk->d) * (float)kvalues_iq4nl[nib];
        }
    } break;
    case 21: { // IQ3_S
        const auto * x = (const block_iq3_s *)sb;
        const float d = fast_h2f(x->d);
        const int l = lane / 8, k = lane % 8;
#pragma unroll
        for (int b = 0; b < 8; b++) {
            const int pi = b / 2, part = b % 2;
            const float db = d * (1.0f + 2.0f * (float)(part ? (x->scales[pi] >> 4) : (x->scales[pi] & 0xf)));
            const uint8_t * qs = x->qs + pi * 16 + part * 8;
            const uint8_t * sg = x->signs + pi * 8 + part * 4;
            const int qhb = x->qh[pi * 2 + part];
            int gidx, gi;
            if (k < 4) {
                gidx = qs[2 * l + 0] | ((qhb << (8 - 2 * l)) & 256);
                gi = k;
            } else {
                gidx = qs[2 * l + 1] | ((qhb << (7 - 2 * l)) & 256);
                gi = k - 4;
            }
            const float s = (sg[l] & kmask_iq2xs[k]) ? -1.f : 1.f;
            w[b] = db * (float)((const uint8_t *)(iq3s_grid + gidx))[gi] * s;
        }
    } break;
    case 23: { // IQ4_XS
        const auto * x = (const block_iq4_xs *)sb;
        const float d = fast_h2f(x->d);
#pragma unroll
        for (int b = 0; b < 8; b++) {
            const int ls = ((x->scales_l[b / 2] >> (4 * (b % 2))) & 0xf) | (((x->scales_h >> (2 * b)) & 3) << 4);
            const uint8_t byte = x->qs[16 * b + (lane & 15)];
            const int nib = (lane < 16) ? (byte & 0xf) : (byte >> 4);
            w[b] = d * (float)(ls - 32) * (float)kvalues_iq4nl[nib];
        }
    } break;
    case 12: { // Q4_K
        auto * blk = (const block_q4_K *)sb;
        const float d = fast_h2f(blk->d);
        const float dmin = fast_h2f(blk->dmin);
        uint8_t sc[8], mm[8];
#pragma unroll
        for (int b = 0; b < 8; b++) {
            if (b < 4) {
                sc[b] = blk->scales[b] & 63;
                mm[b] = blk->scales[b + 4] & 63;
            } else {
                sc[b] = (uint8_t)((blk->scales[b + 4] & 0xF) | ((blk->scales[b - 4] >> 6) << 4));
                mm[b] = (uint8_t)((blk->scales[b + 4] >> 4) | ((blk->scales[b] >> 6) << 4));
            }
        }
#pragma unroll
        for (int b = 0; b < 8; b++) {
            const uint8_t byte = blk->qs[32 * (b >> 1) + lane];
            const int nib = (b & 1) ? (byte >> 4) : (byte & 0xF);
            w[b] = d * sc[b] * nib - dmin * mm[b];
        }
    } break;
    case 13: { // Q5_K
        auto * blk = (const block_q5_K *)sb;
        const float d = fast_h2f(blk->d);
        const float dmin = fast_h2f(blk->dmin);
        uint8_t sc[8], mm[8];
        const uint8_t qhb = blk->qh[lane];
#pragma unroll
        for (int b = 0; b < 8; b++) {
            if (b < 4) {
                sc[b] = blk->scales[b] & 63;
                mm[b] = blk->scales[b + 4] & 63;
            } else {
                sc[b] = (uint8_t)((blk->scales[b + 4] & 0xF) | ((blk->scales[b - 4] >> 6) << 4));
                mm[b] = (uint8_t)((blk->scales[b + 4] >> 4) | ((blk->scales[b] >> 6) << 4));
            }
        }
#pragma unroll
        for (int b = 0; b < 8; b++) {
            const uint8_t byte = blk->qs[32 * (b >> 1) + lane];
            int nib = (b & 1) ? (byte >> 4) : (byte & 0xF);
            nib += ((qhb >> b) & 1) << 4;
            w[b] = d * sc[b] * nib - dmin * mm[b];
        }
    } break;
    case 14: { // Q6_K
        auto * blk = (const block_q6_K *)sb;
        const float d = fast_h2f(blk->d);
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint8_t * ql = blk->ql + h * 64;
            const uint8_t * qh = blk->qh + h * 32;
            const int8_t * sc = blk->scales + h * 8;
            const uint8_t ql0 = ql[lane], ql1 = ql[lane + 32], qh0 = qh[lane];
            const int is = lane / 16;
            w[h * 4 + 0] = d * sc[is + 0] * (int)(((ql0 & 0xF) | (((qh0 >> 0) & 3) << 4)) - 32);
            w[h * 4 + 1] = d * sc[is + 2] * (int)(((ql1 & 0xF) | (((qh0 >> 2) & 3) << 4)) - 32);
            w[h * 4 + 2] = d * sc[is + 4] * (int)(((ql0 >> 4) | (((qh0 >> 4) & 3) << 4)) - 32);
            w[h * 4 + 3] = d * sc[is + 6] * (int)(((ql1 >> 4) | (((qh0 >> 6) & 3) << 4)) - 32);
        }
    } break;
    case 8: { // Q8_0
#pragma unroll
        for (int b = 0; b < 8; b++) {
            auto * blk = (const block_q8_0 *)(sb + b * (int)sizeof(block_q8_0));
            w[b] = fast_h2f(blk->d) * blk->qs[lane];
        }
    } break;
    default:
#pragma unroll
        for (int b = 0; b < 8; b++) {
            w[b] = ((const float *)sb)[32 * b + lane];
        }
        break;
    }
}

// weights of one 256-element superblock for lane `lane` (lane <-> element 32*b + lane)
inline void dequant_sb_lane(uint32_t type, const char * sb, int lane, float w[8]) {
    switch (type) {
    case 11: dequant_sb_lane_typed<11>(sb, lane, w); break;
    case 12: dequant_sb_lane_typed<12>(sb, lane, w); break;
    case 13: dequant_sb_lane_typed<13>(sb, lane, w); break;
    case 14: dequant_sb_lane_typed<14>(sb, lane, w); break;
    case 20: dequant_sb_lane_typed<20>(sb, lane, w); break;
    case 21: dequant_sb_lane_typed<21>(sb, lane, w); break;
    case 23: dequant_sb_lane_typed<23>(sb, lane, w); break;
    case 8: dequant_sb_lane_typed<8>(sb, lane, w); break;
    default: dequant_sb_lane_typed<0>(sb, lane, w); break;
    }
}

// element `e` (0..255) of a 256-element superblock
inline float dequant_elem_sb(uint32_t type, const char * sb, int e) {
    switch (type) {
    case 11:
    case 20:
    case 21:
    case 23: return dequant_elem_iq(type, sb, e);
    case 12: {
        auto * b = (const block_q4_K *)sb;
        const int chunk = e / 64, l = e % 64; // 4 chunks x 64 elems
        const int part = l / 32, li = l % 32; // low/high nibble
        uint8_t scv, mv;
        get_scale_min_k4(2 * chunk + part, b->scales, &scv, &mv);
        const uint8_t byte = b->qs[32 * chunk + li];
        const int nib = part ? (byte >> 4) : (byte & 0xF);
        return ggml_half_to_float(b->d) * scv * nib - ggml_half_to_float(b->dmin) * mv;
    }
    case 13: {
        auto * b = (const block_q5_K *)sb;
        const int chunk = e / 64, l = e % 64;
        const int part = l / 32, li = l % 32;
        uint8_t scv, mv;
        get_scale_min_k4(2 * chunk + part, b->scales, &scv, &mv);
        const uint8_t byte = b->qs[32 * chunk + li];
        int nib = part ? (byte >> 4) : (byte & 0xF);
        nib += ((b->qh[li] >> (2 * chunk + part)) & 1) << 4;
        return ggml_half_to_float(b->d) * scv * nib - ggml_half_to_float(b->dmin) * mv;
    }
    case 14: {
        auto * b = (const block_q6_K *)sb;
        const int half = e / 128, r = e % 128;
        const int bb = r / 32, l = r % 32;
        const uint8_t * ql = b->ql + half * 64;
        const uint8_t * qh = b->qh + half * 32;
        const int8_t * sc = b->scales + half * 8;
        int q, sci;
        if (bb == 0) {
            q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            sci = l / 16 + 0;
        } else if (bb == 1) {
            q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            sci = l / 16 + 2;
        } else if (bb == 2) {
            q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            sci = l / 16 + 4;
        } else {
            q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            sci = l / 16 + 6;
        }
        return ggml_half_to_float(b->d) * sc[sci] * (q - 32);
    }
    case 8: {
        const int blk = e / 32, l = e % 32;
        auto * b = (const block_q8_0 *)(sb + blk * (int)sizeof(block_q8_0));
        return ggml_half_to_float(b->d) * b->qs[l];
    }
    case 0: return ((const float *)sb)[e];
    default: return 0.f;
    }
}

// weight for lane `lane` of sub-block `b` within a 256-elem superblock
inline float dequant_lane_w(uint32_t type, const char * sb, int b, int lane) {
    switch (type) {
    case 11:
    case 20:
    case 21:
    case 23: return dequant_elem_iq(type, sb, 32 * b + lane);
    case 12: {
        auto * blk = (const block_q4_K *)sb;
        const int chunk = b / 2, part = b % 2;
        uint8_t scv, mv;
        get_scale_min_k4(b, blk->scales, &scv, &mv);
        const uint8_t byte = blk->qs[32 * chunk + lane];
        const int nib = part ? (byte >> 4) : (byte & 0xF);
        return ggml_half_to_float(blk->d) * scv * nib - ggml_half_to_float(blk->dmin) * mv;
    }
    case 13: {
        auto * blk = (const block_q5_K *)sb;
        const int chunk = b / 2, part = b % 2;
        uint8_t scv, mv;
        get_scale_min_k4(b, blk->scales, &scv, &mv);
        const uint8_t byte = blk->qs[32 * chunk + lane];
        int nib = part ? (byte >> 4) : (byte & 0xF);
        nib += ((blk->qh[lane] >> b) & 1) << 4;
        return ggml_half_to_float(blk->d) * scv * nib - ggml_half_to_float(blk->dmin) * mv;
    }
    case 14: {
        auto * blk = (const block_q6_K *)sb;
        const int half = b / 4, bb = b % 4;
        const uint8_t * ql = blk->ql + half * 64;
        const uint8_t * qh = blk->qh + half * 32;
        const int8_t * sc = blk->scales + half * 8;
        int q, sci;
        if (bb == 0) {
            q = (ql[lane] & 0xF) | (((qh[lane] >> 0) & 3) << 4);
            sci = lane / 16 + 0;
        } else if (bb == 1) {
            q = (ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4);
            sci = lane / 16 + 2;
        } else if (bb == 2) {
            q = (ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4);
            sci = lane / 16 + 4;
        } else {
            q = (ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4);
            sci = lane / 16 + 6;
        }
        return ggml_half_to_float(blk->d) * sc[sci] * (q - 32);
    }
    case 8: {
        auto * blk = (const block_q8_0 *)(sb + b * (int)sizeof(block_q8_0));
        return ggml_half_to_float(blk->d) * blk->qs[lane];
    }
    case 0: return ((const float *)sb)[32 * b + lane];
    default: return 0.f;
    }
}

inline int superblock_bytes(uint32_t type) {
    switch (type) {
    case 11: return 110;
    case 12: return 144;
    case 13: return 176;
    case 14: return 210;
    case 20: return 8 * 18;
    case 21: return 110;
    case 23: return 136;
    case 8: return 8 * 34;
    case 0: return 256 * 4;
    default: return 0;
    }
}
// Sub-group reductions.  N is a compile-time width (always 32 in this code
// base): querying sg.get_local_range() at runtime makes the shuffle loop
// dynamic, which measured ~2x slower in the GDN recurrence (the loop bound and
// the query ended up dominating the per-token work).
template <int N = 32> inline float sg_sum(float v, const sub_group & sg) {
    for (int mask = N / 2; mask > 0; mask >>= 1) {
        v += sycl::permute_group_by_xor(sg, v, mask);
    }
    return v;
}

template <int N = 32> inline int32_t sg_sum_i(int32_t v, const sub_group & sg) {
    for (int mask = N / 2; mask > 0; mask >>= 1) {
        v += sycl::permute_group_by_xor(sg, v, mask);
    }
    return v;
}

inline float dot4(const sycl::float4 & a, const sycl::float4 & b) {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
}
inline sycl::float4 mul4(const sycl::float4 & a, float b) {
    return sycl::float4(a.x() * b, a.y() * b, a.z() * b, a.w() * b);
}
inline sycl::float4 fma4(const sycl::float4 & a, float b, const sycl::float4 & c) {
    return sycl::float4(a.x() * b + c.x(), a.y() * b + c.y(), a.z() * b + c.z(), a.w() * b + c.w());
}

// sub-group reductions.  N is a compile-time width (always 32 in this code
// base): querying sg.get_local_range() at runtime makes the shuffle loop
// dynamic, which measured ~2x slower in the GDN recurrence (the loop bound and
// the query ended up dominating the per-token work).
template <int N = 32> inline float sg_max(float v, const sub_group & sg) {
    for (int mask = N / 2; mask > 0; mask >>= 1) {
        v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, mask));
    }
    return v;
}

inline uint32_t pack_half2(float a, float b) {
    return (uint32_t)ggml_float_to_half(a) | ((uint32_t)ggml_float_to_half(b) << 16);
}
inline uint32_t spread1(uint32_t x) {
    return (x & 1u) | ((x & 2u) << 7) | ((x & 4u) << 14) | ((x & 8u) << 21);
}
inline uint32_t spread2(uint32_t x) {
    return (x & 3u) | ((x & 12u) << 6) | ((x & 48u) << 12) | ((x & 192u) << 18);
}
// nibble half: 8 bytes -> 4 words with the value sets {0..3, 8..11, 4..7, 12..15}.
// GB = the group stride: when it is a multiple of 8 the halves are 8-byte
// aligned and one vector load replaces two scalar ones.
template <int GB> inline void w8_nib_half(const uint8_t * p, uint32_t * nw) {
    uint32_t lo, hi;
    if constexpr (GB % 8 == 0) {
        const uint2 n = *reinterpret_cast<const uint2 *>(p);
        lo = n.x();
        hi = n.y();
    } else {
        lo = *reinterpret_cast<const uint32_t *>(p);
        hi = *reinterpret_cast<const uint32_t *>(p + 4);
    }
    nw[0] = lo & 0x0F0F0F0Fu;        // values 0..3
    nw[1] = (lo >> 4) & 0x0F0F0F0Fu; // values 8..11
    nw[2] = hi & 0x0F0F0F0Fu;        // values 4..7
    nw[3] = (hi >> 4) & 0x0F0F0F0Fu; // values 12..15
}
// expand one 16-value half of a group into 4 dp4a words (used by the GEMM,
// which walks half groups so only 4 weight + 4 x words are live at a time)
template <uint32_t QT> inline void w8_expand_half(const uint8_t * p, int h, uint32_t * nw) {
    w8_nib_half<w8_group_bytes(QT)>(p + h * 8, nw);
    if constexpr (QT == 13) {
        const uint32_t pl = *reinterpret_cast<const uint32_t *>(p + 16) >> (16 * h);
        nw[0] |= spread1(pl & 0xFu) << 4;
        nw[1] |= spread1((pl >> 8) & 0xFu) << 4;
        nw[2] |= spread1((pl >> 4) & 0xFu) << 4;
        nw[3] |= spread1((pl >> 12) & 0xFu) << 4;
    } else if constexpr (QT == 14) {
        const uint32_t pl = *reinterpret_cast<const uint32_t *>(p + 8);
        nw[0] |= spread2(pl & 0xFFu) << 4;
        nw[1] |= spread2((pl >> 16) & 0xFFu) << 4;
        nw[2] |= spread2((pl >> 8) & 0xFFu) << 4;
        nw[3] |= spread2((pl >> 24) & 0xFFu) << 4;
    }
}

// expand a whole group (used by the GEMV, which holds one row's words at a time)
template <uint32_t QT> inline void w8_group_expand(const uint8_t * p, uint32_t * w) {
    w8_expand_half<QT>(p, 0, w);
    if constexpr (QT != 14) {
        w8_expand_half<QT>(p, 1, w + 4);
    }
}
// index of the x word that pairs with weight word slot j inside a half
template <uint32_t QT> inline int w8_xword(int j) {
    return (j == 1) ? 2 : (j == 2 ? 1 : j);
}

// K-split workspace shared by the DP4A GEMV/GEMM paths (defined in dp4a_common.cpp).
// One queue/device per process, so the buffer is a process-wide singleton.
float * gemm_ws(queue & q, size_t need);

} // namespace kd
} // namespace si
