#include "kernels.h"
#include "kernel_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

using namespace sycl;
using namespace si::kd;

// ---------------------------------------------------------------------------
// int8 GEMM via hardware dp4a (prefill): each lane computes TP tokens x RP rows
// of a 32-token tile; a 128-thread workgroup covers (WG/32)*rows_per_sg rows.
// ---------------------------------------------------------------------------
template <uint32_t QT, int TP, int RP>
static void dp4a_gemm_impl(queue & q, const w8t & w, const int8_t * x8, const sycl::float2 * xmeta,
                           const int32_t * xsumq, float * out, int out_stride, const float * residual, float alpha,
                           int TB) {
    const int K = w.K;
    const int N = w.N;
    constexpr int G = w8_group_size(QT);
    constexpr int GB = w8_group_bytes(QT);
    constexpr int NH = G / 16;
    constexpr int WG = 128;
    const int MG = K / G;
    const int TG = 32 / TP; // token groups
    const int RG = 32 / TG; // row groups per sub-group
    const int warp_rows = RG * RP;
    const int rows_per_wg = (WG / 32) * warp_rows;
    const int n_wg = (N + rows_per_wg - 1) / rows_per_wg;
    const int n_tok = TB < 32 ? TB : 32;
    q.parallel_for(nd_range<1>((size_t)n_wg * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int wid = it.get_local_id(0) / 32;
        const int lane = it.get_local_id(0) % 32;
        const int tg = lane % TG; // token group: TP consecutive tokens
        const int rg = lane / TG; // row group: RP consecutive rows
        const int row0 = it.get_group(0) * rows_per_wg + wid * warp_rows + rg * RP;
        const int t0 = tg * TP;
        const bool row_ok = (row0 + RP - 1 < N);
        const int rowc = sycl::min(row0, N - RP); // clamped base for uniform loads

        float acc[TP][RP];
#pragma unroll
        for (int tt = 0; tt < TP; tt++) {
#pragma unroll
            for (int rr = 0; rr < RP; rr++) {
                acc[tt][rr] = 0.f;
            }
        }

        const int8_t * xp[TP];
        const sycl::float2 * xmp[TP];
        const int32_t * xsp[TP];
#pragma unroll
        for (int tt = 0; tt < TP; tt++) {
            xp[tt] = x8 + (size_t)(t0 + tt) * 32;
            xmp[tt] = xmeta + t0 + tt;
            xsp[tt] = xsumq + t0 + tt;
        }
        const uint8_t * wp[RP];
        const char * wmp[RP];
        const int melem = w.meta_elem;
#pragma unroll
        for (int rr = 0; rr < RP; rr++) {
            const int row = rowc + rr;
            const int rb = row / kRB, ri = row % kRB;
            wp[rr] = w.vals + (size_t)rb * MG * kRB * GB + (size_t)ri * GB;
            wmp[rr] = (const char *)w.meta + ((size_t)rb * MG * kRB + ri) * melem;
        }
        for (int g = 0; g < MG; g++) {
            // walk the group in 16-value halves so only four weight words and
            // four x words are live at a time (the 4x4 form spilled registers)
            const int gx = (G == 32) ? g : (g >> 1);
            const int xo0 = (G == 32) ? 0 : (g & 1);
            int32_t dot[TP][RP];
#pragma unroll
            for (int tt = 0; tt < TP; tt++) {
#pragma unroll
                for (int rr = 0; rr < RP; rr++) {
                    dot[tt][rr] = 0;
                }
            }
#pragma unroll
            for (int h = 0; h < NH; h++) {
                uint32_t gw[RP][4];
#pragma unroll
                for (int rr = 0; rr < RP; rr++) {
                    w8_expand_half<QT>(wp[rr] + (size_t)g * kRB * GB, h, gw[rr]);
                }
                uint32_t xw[TP][4];
#pragma unroll
                for (int tt = 0; tt < TP; tt++) {
                    const uint4 v = *reinterpret_cast<const uint4 *>(xp[tt] + (size_t)gx * TB * 32 + (xo0 + h) * 16);
                    xw[tt][0] = v.x();
                    xw[tt][1] = v.y();
                    xw[tt][2] = v.z();
                    xw[tt][3] = v.w();
                }
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int jx = w8_xword<QT>(j);
#pragma unroll
                    for (int tt = 0; tt < TP; tt++) {
#pragma unroll
                        for (int rr = 0; rr < RP; rr++) {
                            dot[tt][rr] = dp4a_s8u8((int32_t)xw[tt][jx], gw[rr][j], dot[tt][rr]);
                        }
                    }
                }
            }
            float sw[RP], mw[RP];
#pragma unroll
            for (int rr = 0; rr < RP; rr++) {
                w8_sw_mw<QT>(wmp[rr], (size_t)g * kRB, melem, sw[rr], mw[rr]);
            }
#pragma unroll
            for (int tt = 0; tt < TP; tt++) {
                const float sx = xmp[tt][(size_t)gx * TB].x();
                const sycl::int2 sq = reinterpret_cast<const sycl::int2 *>(xsp[tt])[(size_t)gx * TB];
                const float c = (G == 32) ? (float)(sq.x() + sq.y()) : (float)(xo0 ? sq.y() : sq.x());
#pragma unroll
                for (int rr = 0; rr < RP; rr++) {
                    acc[tt][rr] += sx * (sw[rr] * (float)dot[tt][rr] - mw[rr] * c);
                }
            }
        }

        if (!row_ok) {
            return;
        }
#pragma unroll
        for (int tt = 0; tt < TP; tt++) {
            if (t0 + tt >= n_tok) {
                continue;
            }
#pragma unroll
            for (int rr = 0; rr < RP; rr++) {
                const size_t o = (size_t)(t0 + tt) * out_stride + row0 + rr;
                float v = alpha * acc[tt][rr];
                if (residual) {
                    v += residual[o];
                }
                out[o] = v;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Row-major DP4A GEMM (prefill): the decode GEMV's lane mapping extended to all
// 32 tokens.  Each lane owns one output row and keeps 32 float accumulators, so
// the weight loads are fully coalesced (32 consecutive rows = 512 B per
// sub-group per group) and every x/meta read is a broadcast.  The register-tile
// kernel above reads weights as 16-byte broadcasts (a few rows per sub-group)
// and x as 16-byte chunks at a stride, which the measurements show is 3x slower
// on the same data (the GEMV streams 12 GB/s with this mapping, the tiled GEMM
// only ~4 GB/s).
//
// TB is a template parameter so accumulators are statically indexed.
// ---------------------------------------------------------------------------
// MT (M-tiled): TB_T-token tiles along the token axis, one grid dimension, so a
// whole prompt (up to kMaxB*kMaxT tokens) runs in one dispatch and the token
// tiles of a row block are resident together (their weight reads merge in L2).
// tstride is the token stride of the quantized activation layout (the xq
// group stride), which equals the total token count in that case.
template <uint32_t QT, int WG, int TB_T, bool SPLIT, int SGS = 16, bool MT = false>
static void dp4a_row_gemm_impl(queue & q, const w8t & w, const int8_t * x8, const sycl::float2 * xmeta,
                               const int32_t * xsumq, float * out, int out_stride, const float * residual, float alpha,
                               int TB, int n_split = 1, float * ws = nullptr, int tstride = 0) {
    const int K = w.K;
    const int N = w.N;
    constexpr int G = w8_group_size(QT);
    constexpr int GB = w8_group_bytes(QT);
    constexpr int WGN = w8_words_per_group(QT);
    constexpr int NH = G / 16;
    const int MG = K / G;
    // the dispatcher only picks this kernel when TB == TB_T (or MT with
    // TB == tstride), so the token loops are straight-line code (no guards)
    const int n_wg = (N + WG - 1) / WG;
    const int n_tt = MT ? (tstride / TB_T) : 1;
    const int grid = n_wg * n_split * n_tt;
    // activation-token stride and this tile's first token: compile-time in the
    // non-MT case so the chunked path's codegen is unchanged
    const int GST = MT ? tstride : TB_T;
    const int T0 = 0;
    (void)TB;
#ifdef PF_MT_DEBUG
    if (MT) {
        static bool once = true;
        if (once) {
            once = false;
            fprintf(stderr, "[mtk] K=%d N=%d TB=%d tstride=%d n_tt=%d grid=%d\n", K, N, TB, tstride, n_tt, grid);
        }
    }
#endif
    q.parallel_for(nd_range<1>((size_t)grid * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SGS)]] {
        const int blk = it.get_group(0);
        const int tt = MT ? (blk % n_tt) : 0;
        const int rest = blk / n_tt;
        const int s = rest % n_split; // K-split index
        const int wg = rest / n_split;
        const int tbase = (MT ? tt * TB_T : 0) + T0;
        const int row = wg * WG + it.get_local_id(0);
        const bool row_ok = row < N;
        const int rowc = sycl::min(row, N - 1);
        const int rb = rowc / kRB, ri = rowc % kRB;
        const uint8_t * wp = w.vals + ((size_t)rb * MG * kRB + ri) * GB;
        const int melem = w.meta_elem;
        const char * wmp = (const char *)w.meta + ((size_t)rb * MG * kRB + ri) * melem;
        float acc[TB_T];
#pragma unroll
        for (int t = 0; t < TB_T; t++) {
            acc[t] = 0.f;
        }
        const int8_t * xp = x8;
        const sycl::float2 * xm = xmeta;
        const int32_t * xs = xsumq;
        const int g0 = (int)((long)s * MG / n_split);
        const int g1 = (int)((long)(s + 1) * MG / n_split);
        // optional: prefetch the next group's packed weight bytes while the
        // current group's 32-token body runs (the load has ~2400 cycles to
        // arrive); PF_ROW_PF=0 disables the experiment
        for (int g = g0; g < g1; g++) {
            uint32_t gw[WGN];
            w8_group_expand<QT>(wp + (size_t)g * kRB * GB, gw);
            float sw, mw;
            w8_sw_mw<QT>(wmp, (size_t)g * kRB, melem, sw, mw);
            const int gx = (G == 32) ? g : (g >> 1);
            const int xo0 = (G == 32) ? 0 : (g & 1);
#pragma unroll
            for (int t = 0; t < TB_T; t++) {
                uint32_t xw[WGN];
#pragma unroll
                for (int h = 0; h < NH; h++) {
                    const uint4 v =
                        *reinterpret_cast<const uint4 *>(xp + ((size_t)(gx * GST + tbase + t)) * 32 + (xo0 + h) * 16);
                    xw[h * 4 + 0] = v.x();
                    xw[h * 4 + 1] = v.y();
                    xw[h * 4 + 2] = v.z();
                    xw[h * 4 + 3] = v.w();
                }
                int32_t d = 0;
#pragma unroll
                for (int h = 0; h < NH; h++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        d = dp4a_s8u8((int32_t)xw[h * 4 + w8_xword<QT>(j)], gw[h * 4 + j], d);
                    }
                }
                const sycl::float2 f = xm[(size_t)gx * GST + tbase + t];
                const sycl::int2 sq = reinterpret_cast<const sycl::int2 *>(xs)[(size_t)gx * GST + tbase + t];
                const float c = (G == 32) ? (float)(sq.x() + sq.y()) : (float)(xo0 ? sq.y() : sq.x());
                acc[t] += f.x() * (sw * (float)d - mw * c);
            }
        }
        if (!row_ok) {
            return;
        }
        if constexpr (SPLIT) {
            // partials are packed with the tensor's own N (out_stride can be
            // larger, e.g. ffn_gate/up into a 2*n_ff buffer)
#pragma unroll
            for (int t = 0; t < TB_T; t++) {
                ws[((size_t)s * TB_T + t) * N + row] = acc[t];
            }
        } else {
#pragma unroll
            for (int t = 0; t < TB_T; t++) {
                const size_t o = (size_t)(tbase + t) * out_stride + row;
                float v = alpha * acc[t];
                if (residual) {
                    v += residual[o];
                }
                out[o] = v;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// M-tiled variant, 2 rows per lane, 16 tokens per tile (32 accumulators - the
// same register budget as 1 row x 32 tokens).  The x and meta loads are shared
// by the two rows, so the per-cell instruction count drops by ~25%.
// ---------------------------------------------------------------------------
template <uint32_t QT, int R = 2, int WG = 128, int TB_T = 16, int SGS = 16, bool SLM = false, bool PF = false>
static void dp4a_row2_mt_gemm_impl(queue & q, const w8t & w, const int8_t * x8, const sycl::float2 * xmeta,
                                   const int32_t * xsumq, float * out, int out_stride, const float * residual,
                                   float alpha, int tstride) {
    const int K = w.K;
    const int N = w.N;
    constexpr int G = w8_group_size(QT);
    constexpr int GB = w8_group_bytes(QT);
    constexpr int WGN = w8_words_per_group(QT);
    constexpr int NH = G / 16;
    const int MG = K / G;
    const int n_wg = (N + WG * R - 1) / (WG * R);
    const int n_tt = tstride / TB_T;
    const int grid = n_wg * n_tt;
    constexpr bool mt_pf = PF;
    q.submit([&](handler & h) {
        // SLM staging for the token tile's activations: each (group, token) cell
        // is 32 bytes that every lane of the workgroup reads (broadcast), so a
        // staged copy removes the per-cell global loads from the inner loop.
        sycl::local_accessor<int8_t, 1> xslm(SLM ? (size_t)(w.K / 32) * TB_T * 32 : 1, h);
        h.parallel_for(nd_range<1>((size_t)grid * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SGS)]] {
            const int blk = it.get_group(0);
            const int tt = blk % n_tt;
            const int wgrp = blk / n_tt;
            const int tbase = tt * TB_T;
            const int row0 = wgrp * WG * R + it.get_local_id(0) * R;
            if constexpr (SLM) {
                // one 32-byte cell per (group-pair, token), mirroring the global
                // group-major activation layout
                const int NG = K / 32;
                const int8_t * src = x8 + (size_t)tbase * 32;
                int8_t * dst = xslm.get_multi_ptr<sycl::access::decorated::yes>().get();
                const int tid = it.get_local_id(0);
                for (int c = tid; c < NG * TB_T; c += WG) {
                    const int gxc = c / TB_T, tcell = c % TB_T;
                    *reinterpret_cast<sycl::int4 *>(dst + (size_t)c * 32) =
                        *reinterpret_cast<const sycl::int4 *>(src + ((size_t)gxc * tstride + tcell) * 32);
                    *reinterpret_cast<sycl::int4 *>(dst + (size_t)c * 32 + 16) =
                        *reinterpret_cast<const sycl::int4 *>(src + ((size_t)gxc * tstride + tcell) * 32 + 16);
                }
                it.barrier();
            }
            const bool row_ok = row0 < N;
            const int rb = (row0 / kRB), ri = (row0 % kRB);
            const uint8_t * wp = w.vals + ((size_t)rb * MG * kRB + ri) * GB;
            const int melem = w.meta_elem;
            const char * wmp = (const char *)w.meta + ((size_t)rb * MG * kRB + ri) * melem;
            float acc[R][TB_T];
#pragma unroll
            for (int r = 0; r < R; r++) {
#pragma unroll
                for (int t = 0; t < TB_T; t++) {
                    acc[r][t] = 0.f;
                }
            }
            const int8_t * xp = x8;
            const sycl::float2 * xm = xmeta;
            const int32_t * xs = xsumq;
            for (int g = 0; g < MG; g++) {
                uint32_t gw[R][WGN];
                float sw[R], mw[R];
#pragma unroll
                for (int r = 0; r < R; r++) {
                    if (row0 + r < N) {
                        w8_group_expand<QT>(wp + (size_t)g * kRB * GB + (size_t)r * GB, gw[r]);
                        w8_sw_mw<QT>(wmp, (size_t)g * kRB + r, melem, sw[r], mw[r]);
                    } else {
                        sw[r] = 0.f;
                        mw[r] = 0.f;
                    }
                }
                const int gx = (G == 32) ? g : (g >> 1);
                const int xo0 = (G == 32) ? 0 : (g & 1);
                // PF_MT_PF=1: load the next token's activations and metadata while
                // this token's dp4a chain runs (the loads have the L1/L2 latency of
                // a full group stride to cover)
                uint4 xw2[2][NH];
                sycl::float2 f2[2];
                sycl::int2 sq2[2];
                auto loadxt = [&](int t, int slot) {
#pragma unroll
                    for (int h = 0; h < NH; h++) {
                        xw2[slot][h] = *reinterpret_cast<const uint4 *>(xp + ((size_t)(gx * tstride + tbase + t)) * 32
                                                                        + (xo0 + h) * 16);
                    }
                    f2[slot] = xm[(size_t)gx * tstride + tbase + t];
                    sq2[slot] = reinterpret_cast<const sycl::int2 *>(xs)[(size_t)gx * tstride + tbase + t];
                };
                if (mt_pf) {
                    loadxt(0, 0);
                }
#pragma unroll
                for (int t = 0; t < TB_T; t++) {
                    uint32_t xw[WGN];
                    sycl::float2 f;
                    sycl::int2 sq;
                    const int cur = mt_pf ? (t & 1) : 0;
                    if (mt_pf) {
                        if (t + 1 < TB_T) {
                            loadxt(t + 1, cur ^ 1);
                        }
#pragma unroll
                        for (int h = 0; h < NH; h++) {
                            xw[h * 4 + 0] = xw2[cur][h].x();
                            xw[h * 4 + 1] = xw2[cur][h].y();
                            xw[h * 4 + 2] = xw2[cur][h].z();
                            xw[h * 4 + 3] = xw2[cur][h].w();
                        }
                        f = f2[cur];
                        sq = sq2[cur];
                    } else {
#pragma unroll
                        for (int h = 0; h < NH; h++) {
                            const uint4 v = SLM ? *reinterpret_cast<const uint4 *>(
                                                      xslm.get_multi_ptr<sycl::access::decorated::yes>().get()
                                                      + ((size_t)(gx * TB_T + t)) * 32 + (xo0 + h) * 16)
                                                : *reinterpret_cast<const uint4 *>(
                                                      xp + ((size_t)(gx * tstride + tbase + t)) * 32 + (xo0 + h) * 16);
                            xw[h * 4 + 0] = v.x();
                            xw[h * 4 + 1] = v.y();
                            xw[h * 4 + 2] = v.z();
                            xw[h * 4 + 3] = v.w();
                        }
                        f = xm[(size_t)gx * tstride + tbase + t];
                        sq = reinterpret_cast<const sycl::int2 *>(xs)[(size_t)gx * tstride + tbase + t];
                    }
                    const float c = (G == 32) ? (float)(sq.x() + sq.y()) : (float)(xo0 ? sq.y() : sq.x());
                    const float fc = f.x() * c;
#pragma unroll
                    for (int r = 0; r < R; r++) {
                        int32_t d = 0;
#pragma unroll
                        for (int h = 0; h < NH; h++) {
#pragma unroll
                            for (int j = 0; j < 4; j++) {
                                d = dp4a_s8u8((int32_t)xw[h * 4 + w8_xword<QT>(j)], gw[r][h * 4 + j], d);
                            }
                        }
                        acc[r][t] += f.x() * sw[r] * (float)d - mw[r] * fc;
                    }
                }
            }
            if (!row_ok) {
                return;
            }
#pragma unroll
            for (int t = 0; t < TB_T; t++) {
                const size_t o = (size_t)(tbase + t) * out_stride + row0;
#pragma unroll
                for (int r = 0; r < R; r++) {
                    if (row0 + r >= N) {
                        break;
                    }
                    float v = alpha * acc[r][t];
                    if (residual) {
                        v += residual[o + r];
                    }
                    out[o + r] = v;
                }
            }
        });
    });
}

// ---------------------------------------------------------------------------
// ARCH=2 ("gemmstone-style" mapping): a 32x4 lane grid per workgroup, each
// lane owns a 4-row x 4-token register tile (16 accumulators).  The four rows'
// weights stay in registers for the whole token loop, and the four tokens share
// them, so the weight loads and the x/meta loads both amortize better than in
// the R2 kernel (~3.1 vs ~2.5 MACs per instruction).
// ---------------------------------------------------------------------------
template <uint32_t QT, int TB_T = 16, int SGS = 16>
static void dp4a_gs_mt_gemm_impl(queue & q, const w8t & w, const int8_t * x8, const sycl::float2 * xmeta,
                                 const int32_t * xsumq, float * out, int out_stride, const float * residual,
                                 float alpha, int tstride) {
    const int K = w.K;
    const int N = w.N;
    constexpr int G = w8_group_size(QT);
    constexpr int GB = w8_group_bytes(QT);
    constexpr int WGN = w8_words_per_group(QT);
    constexpr int NH = G / 16;
    const int MG = K / G;
    const int n_wg = (N + 127) / 128;
    const int n_tt = tstride / TB_T;
    const int grid = n_wg * n_tt;
    q.parallel_for(nd_range<1>((size_t)grid * 128, 128), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SGS)]] {
        const int blk = it.get_group(0);
        const int tt = blk % n_tt;
        const int wgrp = blk / n_tt;
        const int tbase = tt * TB_T;
        const int lid = it.get_local_id(0);
        const int rg = lid / 4; // 32 row groups -> 4 rows each
        const int tg = lid % 4; // 4 token groups -> 4 tokens each
        const int row0 = wgrp * 128 + rg * 4;
        const int t0 = tbase + tg * 4;
        const int rb = row0 / kRB, ri = row0 % kRB;
        const uint8_t * wp = w.vals + ((size_t)rb * MG * kRB + ri) * GB;
        const int melem = w.meta_elem;
        const char * wmp = (const char *)w.meta + ((size_t)rb * MG * kRB + ri) * melem;
        float acc[4][4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
#pragma unroll
            for (int t = 0; t < 4; t++) {
                acc[r][t] = 0.f;
            }
        }
        for (int g = 0; g < MG; g++) {
            uint32_t gw[4][WGN];
            float sw[4], mw[4];
#pragma unroll
            for (int r = 0; r < 4; r++) {
                w8_group_expand<QT>(wp + (size_t)g * kRB * GB + (size_t)r * GB, gw[r]);
                w8_sw_mw<QT>(wmp, (size_t)g * kRB + r, melem, sw[r], mw[r]);
            }
            const int gx = (G == 32) ? g : (g >> 1);
            const int xo0 = (G == 32) ? 0 : (g & 1);
#pragma unroll
            for (int tt4 = 0; tt4 < 4; tt4++) {
                const int t = t0 + tt4;
                uint32_t xw[WGN];
#pragma unroll
                for (int h = 0; h < NH; h++) {
                    const uint4 v =
                        *reinterpret_cast<const uint4 *>(x8 + ((size_t)(gx * tstride + t)) * 32 + (xo0 + h) * 16);
                    xw[h * 4 + 0] = v.x();
                    xw[h * 4 + 1] = v.y();
                    xw[h * 4 + 2] = v.z();
                    xw[h * 4 + 3] = v.w();
                }
                const sycl::float2 f = xmeta[(size_t)gx * tstride + t];
                const sycl::int2 sq = reinterpret_cast<const sycl::int2 *>(xsumq)[(size_t)gx * tstride + t];
                const float c = (G == 32) ? (float)(sq.x() + sq.y()) : (float)(xo0 ? sq.y() : sq.x());
                const float fc = f.x() * c;
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    int32_t d = 0;
#pragma unroll
                    for (int h = 0; h < NH; h++) {
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            d = dp4a_s8u8((int32_t)xw[h * 4 + w8_xword<QT>(j)], gw[r][h * 4 + j], d);
                        }
                    }
                    acc[r][tt4] += f.x() * sw[r] * (float)d - mw[r] * fc;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < 4; t++) {
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const int row = row0 + r;
                if (row >= N) {
                    continue;
                }
                const size_t o = (size_t)(t0 + t) * out_stride + row;
                float v = alpha * acc[r][t];
                if (residual) {
                    v += residual[o];
                }
                out[o] = v;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Row-major DP4A GEMM, 2 rows x 16 tokens per lane.
//
// The 1-row x 32-token mapping spends one x/meta load pair per (token, group)
// on every lane; hoisting those loads measured 45% faster, so this variant
// halves them: a sub-group still covers 32 rows x 32 tokens, but a lane owns
// two adjacent rows and one 16-token half, keeping the same 32 accumulators.
// The two rows' weight groups are contiguous, so they widen the weight load
// instead of doubling it.
// ---------------------------------------------------------------------------
template <uint32_t QT, int WG, int TB_T, bool SPLIT>
static void dp4a_row2_gemm_impl(queue & q, const w8t & w, const int8_t * x8, const sycl::float2 * xmeta,
                                const int32_t * xsumq, float * out, int out_stride, const float * residual, float alpha,
                                int TB, int n_split, float * ws) {
    const int K = w.K;
    const int N = w.N;
    constexpr int G = w8_group_size(QT);
    constexpr int GB = w8_group_bytes(QT);
    constexpr int WGN = w8_words_per_group(QT);
    constexpr int NH = G / 16;
    constexpr int TP = 16; // tokens per lane
    const int MG = K / G;
    const int n_wg = (N + WG - 1) / WG;
    const int grid = n_wg * n_split;
    (void)TB;
    q.parallel_for(nd_range<1>((size_t)grid * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int tid = it.get_local_id(0);
        const int sgl = tid / 32;
        const int lane = tid % 32;
        const int rp = lane % 16; // row pair within the sub-group
        const int th = lane / 16; // token half
        const int blk = it.get_group(0);
        const int s = blk % n_split;
        const int wg = blk / n_split;
        const int row0 = wg * WG + sgl * 32 + rp * 2; // rows row0, row0+1
        const bool row_ok = row0 + 1 < N;
        const int rowc = sycl::min(row0, N - 2);
        const int rb = rowc / kRB, ri = rowc % kRB;
        const bool cross = (ri + 1) >= kRB; // row pair spans two row-blocks?
        const uint8_t * wp = w.vals + ((size_t)rb * MG * kRB + ri) * GB;
        const int melem = w.meta_elem;
        const char * wmp = (const char *)w.meta + ((size_t)rb * MG * kRB + ri) * melem;
        const uint8_t * wp2;
        const char * wmp2;
        if (cross) {
            wp2 = w.vals + (size_t)(rb + 1) * MG * kRB * GB;
            wmp2 = (const char *)w.meta + (size_t)(rb + 1) * MG * kRB * melem;
        } else {
            wp2 = wp + GB;
            wmp2 = wmp + melem;
        }

        float acc0[TP], acc1[TP];
#pragma unroll
        for (int j = 0; j < TP; j++) {
            acc0[j] = 0.f;
            acc1[j] = 0.f;
        }
        const int8_t * xp = x8;
        const sycl::float2 * xm = xmeta;
        const int32_t * xs = xsumq;
        const int g0 = (int)((long)s * MG / n_split);
        const int g1 = (int)((long)(s + 1) * MG / n_split);
        for (int g = g0; g < g1; g++) {
            uint32_t gw0[WGN], gw1[WGN];
            w8_group_expand<QT>(wp + (size_t)g * kRB * GB, gw0);
            w8_group_expand<QT>(wp2 + (size_t)g * kRB * GB, gw1);
            float sw0, mw0, sw1, mw1;
            w8_sw_mw<QT>(wmp, (size_t)g * kRB, melem, sw0, mw0);
            w8_sw_mw<QT>(wmp2, (size_t)g * kRB, melem, sw1, mw1);
            const int gx = (G == 32) ? g : (g >> 1);
            const int xo0 = (G == 32) ? 0 : (g & 1);
#pragma unroll
            for (int j = 0; j < TP; j++) {
                const int t = th * TP + j;
                uint32_t xw[WGN];
#pragma unroll
                for (int h = 0; h < NH; h++) {
                    const uint4 v =
                        *reinterpret_cast<const uint4 *>(xp + ((size_t)(gx * TB + t)) * 32 + (xo0 + h) * 16);
                    xw[h * 4 + 0] = v.x();
                    xw[h * 4 + 1] = v.y();
                    xw[h * 4 + 2] = v.z();
                    xw[h * 4 + 3] = v.w();
                }
                int32_t d0 = 0, d1 = 0;
#pragma unroll
                for (int h = 0; h < NH; h++) {
#pragma unroll
                    for (int q2 = 0; q2 < 4; q2++) {
                        const int jx = h * 4 + w8_xword<QT>(q2);
                        d0 = dp4a_s8u8((int32_t)xw[jx], gw0[h * 4 + q2], d0);
                        d1 = dp4a_s8u8((int32_t)xw[jx], gw1[h * 4 + q2], d1);
                    }
                }
                const sycl::float2 f = xm[(size_t)gx * TB + t];
                const sycl::int2 sq = reinterpret_cast<const sycl::int2 *>(xs)[(size_t)gx * TB + t];
                const float c = (G == 32) ? (float)(sq.x() + sq.y()) : (float)(xo0 ? sq.y() : sq.x());
                acc0[j] += f.x() * (sw0 * (float)d0 - mw0 * c);
                acc1[j] += f.x() * (sw1 * (float)d1 - mw1 * c);
            }
        }
        if (!row_ok) {
            return;
        }
        if constexpr (SPLIT) {
#pragma unroll
            for (int j = 0; j < TP; j++) {
                const int t = th * TP + j;
                ws[((size_t)s * TB_T + t) * out_stride + row0] = acc0[j];
                ws[((size_t)s * TB_T + t) * out_stride + row0 + 1] = acc1[j];
            }
        } else {
#pragma unroll
            for (int j = 0; j < TP; j++) {
                const int t = th * TP + j;
                const size_t o = (size_t)t * out_stride + row0;
                float v0 = alpha * acc0[j];
                float v1 = alpha * acc1[j];
                if (residual) {
                    v0 += residual[o];
                    v1 += residual[o + 1];
                }
                out[o] = v0;
                out[o + 1] = v1;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Row-major DP4A GEMM with the activation (and its scale/min meta) staged in
// SLM per workgroup: the diagnostic showed that the per-token x loads from
// global L1 cost ~30% of the kernel, so they are replaced by cooperative
// staging (a straight copy of contiguous 1 KB per k-group) plus LDS reads.
// One workgroup handles one K-slice of WG rows, like the plain row kernel.
// ---------------------------------------------------------------------------
template <uint32_t QT, int WG, int TB_T, bool SPLIT>
static void dp4a_row_gemm_slm_impl(queue & q, const w8t & w, const int8_t * x8, const sycl::float2 * xmeta,
                                   const int32_t * xsumq, float * out, int out_stride, const float * residual,
                                   float alpha, int TB, int n_split, float * ws) {
    const int K = w.K;
    const int N = w.N;
    constexpr int G = w8_group_size(QT);
    constexpr int GB = w8_group_bytes(QT);
    constexpr int WGN = w8_words_per_group(QT);
    constexpr int NH = G / 16;
    const int MG = K / G;
    const int n_wg = (N + WG - 1) / WG;
    const int grid = n_wg * n_split;
    const int ng_max = (MG + n_split - 1) / n_split; // groups per split
    const int xwords = ng_max * TB_T * 8;            // 32 B per (group, token)
    const int xmwords = ng_max * TB_T * 4;           // float2 + int2 per (group, token)
    (void)TB;
    q.submit([&](handler & h) {
        local_accessor<uint32_t, 1> sx(xwords, h);
        local_accessor<uint32_t, 1> smm(xmwords, h);
        h.parallel_for(nd_range<1>((size_t)grid * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int tid = it.get_local_id(0);
            const int blk = it.get_group(0);
            const int s = blk % n_split;
            const int wg = blk / n_split;
            const int row = wg * WG + tid;
            const bool row_ok = row < N;
            const int rowc = sycl::min(row, N - 1);
            const int rb = rowc / kRB, ri = rowc % kRB;
            const uint8_t * wp = w.vals + ((size_t)rb * MG * kRB + ri) * GB;
            const int melem = w.meta_elem;
            const char * wmp = (const char *)w.meta + ((size_t)rb * MG * kRB + ri) * melem;
            const int g0 = (int)((long)s * MG / n_split);
            const int g1 = (int)((long)(s + 1) * MG / n_split);
            const int ng = g1 - g0;
            uint32_t * sxm = sx.template get_multi_ptr<sycl::access::decorated::no>().get();
            uint32_t * smem2 = smm.template get_multi_ptr<sycl::access::decorated::no>().get();

            // cooperative copy of this slice's activation data (contiguous per group)
            {
                const int total = ng * TB_T * 8; // uint32 words of x
                for (int i = tid; i < total; i += WG) {
                    sxm[i] = reinterpret_cast<const uint32_t *>(x8)[(size_t)g0 * TB_T * 8 + i];
                }
                // per (group, token): 4 words = float2 scale + int2 half sums
                const int mtotal = ng * TB_T * 4;
                for (int i = tid; i < mtotal; i += WG) {
                    const int gl = i / (TB_T * 4);
                    const int rem = i % (TB_T * 4);
                    const int t = rem / 4;
                    const int j = rem % 4;
                    const size_t src = (size_t)(g0 + gl) * TB + t;
                    smem2[i] = (j < 2) ? reinterpret_cast<const uint32_t *>(xmeta)[src * 2 + j]
                                       : reinterpret_cast<const uint32_t *>(xsumq)[src * 2 + (j - 2)];
                }
            }
            it.barrier();

            float acc[TB_T];
#pragma unroll
            for (int t = 0; t < TB_T; t++) {
                acc[t] = 0.f;
            }
            for (int g = g0; g < g1; g++) {
                const int gl = g - g0;
                uint32_t gw[WGN];
                w8_group_expand<QT>(wp + (size_t)g * kRB * GB, gw);
                float sw, mw;
                w8_sw_mw<QT>(wmp, (size_t)g * kRB, melem, sw, mw);
                const int xo0 = (G == 32) ? 0 : (g & 1);
                const int xli = (G == 32) ? gl : (g - (g & 1) - g0); // shared 32-value group
#pragma unroll
                for (int t = 0; t < TB_T; t++) {
                    uint32_t xw[WGN];
#pragma unroll
                    for (int h = 0; h < NH; h++) {
                        const int wi = (xli * TB_T + t) * 8 + (xo0 + h) * 4;
                        xw[h * 4 + 0] = sxm[wi + 0];
                        xw[h * 4 + 1] = sxm[wi + 1];
                        xw[h * 4 + 2] = sxm[wi + 2];
                        xw[h * 4 + 3] = sxm[wi + 3];
                    }
                    int32_t d = 0;
#pragma unroll
                    for (int h = 0; h < NH; h++) {
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            d = dp4a_s8u8((int32_t)xw[h * 4 + w8_xword<QT>(j)], gw[h * 4 + j], d);
                        }
                    }
                    const float sxv = reinterpret_cast<const float *>(smem2)[(xli * TB_T + t) * 4];
                    const sycl::int2 sq = *reinterpret_cast<const sycl::int2 *>(smem2 + (xli * TB_T + t) * 4 + 2);
                    const float c = (G == 32) ? (float)(sq.x() + sq.y()) : (float)((xo0) ? sq.y() : sq.x());
                    acc[t] += sxv * (sw * (float)d - mw * c);
                }
            }
            if (!row_ok) {
                return;
            }
            if constexpr (SPLIT) {
#pragma unroll
                for (int t = 0; t < TB_T; t++) {
                    ws[((size_t)s * TB_T + t) * out_stride + row] = acc[t];
                }
            } else {
#pragma unroll
                for (int t = 0; t < TB_T; t++) {
                    const size_t o = (size_t)t * out_stride + row;
                    float v = alpha * acc[t];
                    if (residual) {
                        v += residual[o];
                    }
                    out[o] = v;
                }
            }
        });
    });
}

// reduce the K-split partials and apply alpha/residual
static void dp4a_gemm_reduce_impl(queue & q, const float * ws, int n_split, float * out, int out_stride, int n_rows,
                                  const float * residual, float alpha, int TB) {
    // n_rows is the tensor's row count; out_stride can be larger (ffn_gate/up
    // share a 2*n_ff buffer), so both are needed and only [0, n_rows) is written
    q.parallel_for((size_t)TB * n_rows, [=](id<1> i) {
        const int t = (int)(i / n_rows);
        const int row = (int)(i % n_rows);
        float v = 0.f;
        for (int s = 0; s < n_split; s++) {
            v += ws[((size_t)s * TB + t) * n_rows + row];
        }
        const size_t o = (size_t)t * out_stride + row;
        v = alpha * v;
        if (residual) {
            v += residual[o];
        }
        out[o] = v;
    });
}

void dp4a_gemm_launch(queue & q, const w8t & w, const int8_t * x8, const sycl::float2 * xmeta, const int32_t * xsumq,
                      float * out, int out_stride, const float * residual, float alpha, int TB) {
    // default tile per weight type (measured, see README): Q4_K likes the
    // TP=1 x-coalescing, Q5_K/Q6_K prefer the wider TP=4 tile
    const char * row_env = getenv("PF_GEMM_ROW");
    const bool use_row = !row_env || atoi(row_env) != 0;
    // M-tiled path: more than one 32-token tile in one dispatch (chunk-batched
    // prefill).  The token tiles of a row block are co-resident, so their weight
    // reads merge in L2, and the extra grid dimension supplies the workgroups
    // the chunked path had to get from K-splitting.
    if (use_row && TB > 32 && (TB % 32) == 0 && TB <= kMaxB * kMaxT && w.N > 0) {
        static const bool dbg_mt = getenv("PF_DBG_MT") != nullptr;
        if (dbg_mt) {
            fprintf(stderr, "[mt] K=%d N=%d TB=%d type=%u\n", w.K, w.N, TB, w.type);
        }
        // 2 rows per lane (halves the per-cell x/meta loads)
        static const int arch = [] {
            const char * e = getenv("PF_GEMM_ARCH");
            return e ? atoi(e) : 1;
        }();
        if (arch == 2) {
            switch (w.type) {
            case 12: dp4a_gs_mt_gemm_impl<12>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
            case 13: dp4a_gs_mt_gemm_impl<13>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
            default: dp4a_gs_mt_gemm_impl<14>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
            }
        }
        static const int mt_r = [] {
            const char * e = getenv("PF_MT_R");
            if (e) {
                return atoi(e);
            }
            const char * e2 = getenv("PF_MT_R2");
            return (e2 && atoi(e2) == 0) ? 1 : 2;
        }();
        // R=1 with a 16-token tile: 16 accumulators (half the register pressure
        // of the R=2 kernel) - the occupancy experiment for the dp4a GEMM
        static const int mt_tb1 = [] {
            const char * e = getenv("PF_MT_TB");
            return e ? atoi(e) : 16;
        }();
        if (mt_r == 1) {
            if (mt_tb1 == 8) {
                switch (w.type) {
                case 12:
                    dp4a_row_gemm_impl<12, 128, 8, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                    alpha, TB, 1, nullptr, TB);
                    return;
                case 13:
                    dp4a_row_gemm_impl<13, 128, 8, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                    alpha, TB, 1, nullptr, TB);
                    return;
                default:
                    dp4a_row_gemm_impl<14, 128, 8, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                    alpha, TB, 1, nullptr, TB);
                    return;
                }
            }
            if (mt_tb1 == 16) {
                switch (w.type) {
                case 12:
                    dp4a_row_gemm_impl<12, 128, 16, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                     alpha, TB, 1, nullptr, TB);
                    return;
                case 13:
                    dp4a_row_gemm_impl<13, 128, 16, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                     alpha, TB, 1, nullptr, TB);
                    return;
                default:
                    dp4a_row_gemm_impl<14, 128, 16, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                     alpha, TB, 1, nullptr, TB);
                    return;
                }
            }
            switch (w.type) {
            case 12:
                dp4a_row_gemm_impl<12, 128, 32, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                 alpha, TB, 1, nullptr, TB);
                return;
            case 13:
                dp4a_row_gemm_impl<13, 128, 32, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                 alpha, TB, 1, nullptr, TB);
                return;
            default:
                dp4a_row_gemm_impl<14, 128, 32, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                 alpha, TB, 1, nullptr, TB);
                return;
            }
        }
        if (mt_r >= 2 && (TB % 16) == 0) {
            if (mt_r == 4) {
                switch (w.type) {
                case 12:
                    dp4a_row2_mt_gemm_impl<12, 4, 128, 8>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                    return;
                case 13:
                    dp4a_row2_mt_gemm_impl<13, 4, 128, 8>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                    return;
                default:
                    dp4a_row2_mt_gemm_impl<14, 4, 128, 8>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                    return;
                }
            }
            static const bool mt_pf = [] {
                const char * e = getenv("PF_MT_PF");
                return e && atoi(e) != 0;
            }();
            if (mt_pf) {
                switch (w.type) {
                case 12:
                    dp4a_row2_mt_gemm_impl<12, 2, 128, 16, 16, false, true>(q, w, x8, xmeta, xsumq, out, out_stride,
                                                                            residual, alpha, TB);
                    return;
                case 13:
                    dp4a_row2_mt_gemm_impl<13, 2, 128, 16, 16, false, true>(q, w, x8, xmeta, xsumq, out, out_stride,
                                                                            residual, alpha, TB);
                    return;
                default:
                    dp4a_row2_mt_gemm_impl<14, 2, 128, 16, 16, false, true>(q, w, x8, xmeta, xsumq, out, out_stride,
                                                                            residual, alpha, TB);
                    return;
                }
            }
            static const bool mt_slm = [] {
                const char * e = getenv("PF_MT_SLM");
                return e && atoi(e) != 0;
            }();
            if (mt_slm) {
                switch (w.type) {
                case 12:
                    dp4a_row2_mt_gemm_impl<12, 2, 128, 16, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                     alpha, TB);
                    return;
                case 13:
                    dp4a_row2_mt_gemm_impl<13, 2, 128, 16, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                     alpha, TB);
                    return;
                default:
                    dp4a_row2_mt_gemm_impl<14, 2, 128, 16, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                     alpha, TB);
                    return;
                }
            }
            static const int mt_tb = [] {
                const char * e = getenv("PF_MT_TB");
                return e ? atoi(e) : 16;
            }();
            static const int mt_wg2 = [] {
                const char * e = getenv("PF_MT_WG2");
                return e ? atoi(e) : 128;
            }();
            if (mt_tb == 8) {
                switch (w.type) {
                case 12:
                    dp4a_row2_mt_gemm_impl<12, 2, 128, 8>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                    return;
                case 13:
                    dp4a_row2_mt_gemm_impl<13, 2, 128, 8>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                    return;
                default:
                    dp4a_row2_mt_gemm_impl<14, 2, 128, 8>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                    return;
                }
            }
            if (mt_tb == 32) {
                switch (w.type) {
                case 12:
                    dp4a_row2_mt_gemm_impl<12, 2, 128, 32>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                           TB);
                    return;
                case 13:
                    dp4a_row2_mt_gemm_impl<13, 2, 128, 32>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                           TB);
                    return;
                default:
                    dp4a_row2_mt_gemm_impl<14, 2, 128, 32>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                           TB);
                    return;
                }
            }
            if (mt_wg2 == 64) {
                switch (w.type) {
                case 12:
                    dp4a_row2_mt_gemm_impl<12, 2, 64, 16>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                    return;
                case 13:
                    dp4a_row2_mt_gemm_impl<13, 2, 64, 16>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                    return;
                default:
                    dp4a_row2_mt_gemm_impl<14, 2, 64, 16>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                    return;
                }
            }
            switch (w.type) {
            case 12:
                dp4a_row2_mt_gemm_impl<12, 2, 128, 16>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                return;
            case 13:
                dp4a_row2_mt_gemm_impl<13, 2, 128, 16>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                return;
            default:
                dp4a_row2_mt_gemm_impl<14, 2, 128, 16>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                return;
            }
        }
        // PF_MT_WG=64: narrower workgroups (occupancy experiment)
        static const int mt_wg = [] {
            const char * e = getenv("PF_MT_WG");
            return e ? atoi(e) : 128;
        }();
        if (mt_wg == 64) {
            switch (w.type) {
            case 12:
                dp4a_row_gemm_impl<12, 64, 32, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                alpha, TB, 1, nullptr, TB);
                return;
            case 13:
                dp4a_row_gemm_impl<13, 64, 32, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                alpha, TB, 1, nullptr, TB);
                return;
            default:
                dp4a_row_gemm_impl<14, 64, 32, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual,
                                                                alpha, TB, 1, nullptr, TB);
                return;
            }
        }
        switch (w.type) {
        case 12:
            dp4a_row_gemm_impl<12, 128, 32, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                             TB, 1, nullptr, TB);
            return;
        case 13:
            dp4a_row_gemm_impl<13, 128, 32, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                             TB, 1, nullptr, TB);
            return;
        default:
            dp4a_row_gemm_impl<14, 128, 32, false, 16, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                             TB, 1, nullptr, TB);
            return;
        }
    }
    // row kernel runs SIMD16 by default (measured faster and required for block
    // reads); PF_GEMM_SG=32 restores the old SIMD32 codegen
    const char * sg_env = getenv("PF_GEMM_SG");
    const bool sg32 = sg_env && atoi(sg_env) == 32;
    // K-split row path: the row kernel wants ~16k output rows to fill the
    // machine; for smaller tensors split K (partial sums in a workspace, then a
    // flat reduce) so the same kernel gets enough workgroups per k-slice.
    if (use_row && TB == 32 && w.N <= 8192) {
        const char * sp_env = getenv("PF_GEMM_SPLIT");
        int S = sp_env ? atoi(sp_env) : 0;
        if (S == 0) {
            S = (int)((16384 + w.N - 1) / w.N);
        }
        if (S > 8) {
            S = 8;
        }
        if (S > 1) {
            // smaller workgroups raise the resident warp count (a 128-thread WG
            // occupies one subslice but only supplies 0.5 warps/EU)
            const char * wg_env = getenv("PF_GEMM_WG");
            const int wg = wg_env ? atoi(wg_env) : 128;
            // the partials are written with the segment's out_stride (which can
            // exceed w.N, e.g. ffn_gate/up into a 2*n_ff buffer), so size the
            // workspace with out_stride or it overflows into other allocations
            float * ws = gemm_ws(q, (size_t)S * TB * w.N);
            // SLM staging of x measured slower (staging + occupancy cost > the
            // saved L1 latency): opt-in only
            static const bool xslm = [] {
                const char * e = getenv("PF_GEMM_XSLM");
                return e && atoi(e) != 0;
            }();
            if (ws && xslm && w.type != 14 && w.N >= 2048) {
                switch (w.type) {
                case 12:
                    dp4a_row_gemm_slm_impl<12, 128, 32, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                              TB, S, ws);
                    break;
                default:
                    dp4a_row_gemm_slm_impl<13, 128, 32, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                              TB, S, ws);
                    break;
                }
                dp4a_gemm_reduce_impl(q, ws, S, out, out_stride, w.N, residual, alpha, TB);
                return;
            }
            if (ws && wg == 128 && sg32) {
                switch (w.type) {
                case 12:
                    dp4a_row_gemm_impl<12, 128, 32, true, 32>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                              TB, S, ws);
                    break;
                case 13:
                    dp4a_row_gemm_impl<13, 128, 32, true, 32>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                              TB, S, ws);
                    break;
                default:
                    dp4a_row_gemm_impl<14, 128, 32, true, 32>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                              TB, S, ws);
                    break;
                }
                dp4a_gemm_reduce_impl(q, ws, S, out, out_stride, w.N, residual, alpha, TB);
                return;
            }
            if (ws && wg == 128) {
                switch (w.type) {
                case 12:
                    dp4a_row_gemm_impl<12, 128, 32, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB,
                                                          S, ws);
                    break;
                case 13:
                    dp4a_row_gemm_impl<13, 128, 32, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB,
                                                          S, ws);
                    break;
                default:
                    dp4a_row_gemm_impl<14, 128, 32, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB,
                                                          S, ws);
                    break;
                }
                dp4a_gemm_reduce_impl(q, ws, S, out, out_stride, w.N, residual, alpha, TB);
                return;
            }
            if (ws && wg == 64) {
                switch (w.type) {
                case 12:
                    dp4a_row_gemm_impl<12, 64, 32, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB,
                                                         S, ws);
                    break;
                case 13:
                    dp4a_row_gemm_impl<13, 64, 32, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB,
                                                         S, ws);
                    break;
                default:
                    dp4a_row_gemm_impl<14, 64, 32, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB,
                                                         S, ws);
                    break;
                }
                dp4a_gemm_reduce_impl(q, ws, S, out, out_stride, w.N, residual, alpha, TB);
                return;
            }
            if (ws) {
                switch (w.type) {
                case 12:
                    dp4a_row_gemm_impl<12, 32, 32, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB,
                                                         S, ws);
                    break;
                case 13:
                    dp4a_row_gemm_impl<13, 32, 32, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB,
                                                         S, ws);
                    break;
                default:
                    dp4a_row_gemm_impl<14, 32, 32, true>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB,
                                                         S, ws);
                    break;
                }
                dp4a_gemm_reduce_impl(q, ws, S, out, out_stride, w.N, residual, alpha, TB);
                return;
            }
        }
    }
    // Enough output rows: the row kernel without splitting.
    if (use_row && w.N >= 2048 && (TB == 32 || TB == 16 || TB == 8)) {
        const bool wide = w.N >= 2048;
        switch (TB) {
        case 32:
            if (wide && sg32) {
                switch (w.type) {
                case 12:
                    dp4a_row_gemm_impl<12, 128, 32, false, 32>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                               TB, 1, nullptr);
                    return;
                case 13:
                    dp4a_row_gemm_impl<13, 128, 32, false, 32>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                               TB, 1, nullptr);
                    return;
                default:
                    dp4a_row_gemm_impl<14, 128, 32, false, 32>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha,
                                                               TB, 1, nullptr);
                    return;
                }
            }
            if (wide) {
                switch (w.type) {
                case 12:
                    dp4a_row_gemm_impl<12, 128, 32, false>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB,
                                                           1, nullptr);
                    return;
                case 13:
                    dp4a_row_gemm_impl<13, 128, 32, false>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB,
                                                           1, nullptr);
                    return;
                default:
                    dp4a_row_gemm_impl<14, 128, 32, false>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB,
                                                           1, nullptr);
                    return;
                }
            }
            switch (w.type) {
            case 12:
                dp4a_row_gemm_impl<12, 64, 32, false>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                return;
            case 13:
                dp4a_row_gemm_impl<13, 64, 32, false>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                return;
            default:
                dp4a_row_gemm_impl<14, 64, 32, false>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                return;
            }
        case 16:
            switch (w.type) {
            case 12:
                dp4a_row_gemm_impl<12, 64, 16, false>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                return;
            case 13:
                dp4a_row_gemm_impl<13, 64, 16, false>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                return;
            default:
                dp4a_row_gemm_impl<14, 64, 16, false>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                return;
            }
        default:
            switch (w.type) {
            case 12:
                dp4a_row_gemm_impl<12, 64, 8, false>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                return;
            case 13:
                dp4a_row_gemm_impl<13, 64, 8, false>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                return;
            default:
                dp4a_row_gemm_impl<14, 64, 8, false>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
                return;
            }
        }
    }
    const char * tile_env = getenv("PF_GEMM_TILE");
    const int tile = tile_env ? atoi(tile_env) : (w.type == 12 ? 12 : 42);
    switch (tile) {
    case 12:
        switch (w.type) {
        case 12: dp4a_gemm_impl<12, 1, 2>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        case 13: dp4a_gemm_impl<13, 1, 2>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        default: dp4a_gemm_impl<14, 1, 2>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        }
    case 14:
        switch (w.type) {
        case 12: dp4a_gemm_impl<12, 1, 4>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        case 13: dp4a_gemm_impl<13, 1, 4>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        default: dp4a_gemm_impl<14, 1, 4>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        }
    case 24:
        switch (w.type) {
        case 12: dp4a_gemm_impl<12, 2, 4>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        case 13: dp4a_gemm_impl<13, 2, 4>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        default: dp4a_gemm_impl<14, 2, 4>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        }
    case 42:
        switch (w.type) {
        case 12: dp4a_gemm_impl<12, 4, 2>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        case 13: dp4a_gemm_impl<13, 4, 2>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        default: dp4a_gemm_impl<14, 4, 2>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        }
    case 44:
        switch (w.type) {
        case 12: dp4a_gemm_impl<12, 4, 4>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        case 13: dp4a_gemm_impl<13, 4, 4>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        default: dp4a_gemm_impl<14, 4, 4>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        }
    default:
        switch (w.type) {
        case 12: dp4a_gemm_impl<12, 2, 2>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        case 13: dp4a_gemm_impl<13, 2, 2>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        default: dp4a_gemm_impl<14, 2, 2>(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB); return;
        }
    }
}

} // namespace si
