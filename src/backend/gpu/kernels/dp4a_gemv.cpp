#include "kernels.h"
#include "device/device_profile.h"
#include "kernel_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "common/env.h"

namespace si {

using namespace sycl;
using namespace si::kd;

// ---------------------------------------------------------------------------
// int8 GEMV via hardware dp4a (decode): one lane per output row, fully
// coalesced 16-byte group loads (rows within a 128-row block are contiguous).
// ---------------------------------------------------------------------------
template <uint32_t QT, bool SPLIT>
static void dp4a_gemv_impl(queue & q, const w8t & w, const int8_t * x8, const sycl::float2 * xmeta,
                           const int32_t * xsumq, float * out, const float * residual, float alpha, int n_split = 1,
                           float * ws = nullptr) {
    const int K = w.K;
    const int N = w.N;
    constexpr int G = w8_group_size(QT);
    constexpr int GB = w8_group_bytes(QT);
    constexpr int WGN = w8_words_per_group(QT);
    constexpr int WG = 128; // 4 sub-groups x 32 rows
    const int MG = K / G;
    const int n_wg = (N + WG - 1) / WG;
    const int grid = n_wg * n_split;
    q.parallel_for(nd_range<1>((size_t)grid * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int blk = it.get_group(0);
        const int sp = blk % n_split;
        const int row = (blk / n_split) * WG + it.get_local_id(0);
        if (row >= N) {
            return;
        }
        const int rb = row / kRB, ri = row % kRB;
        const uint8_t * wp = w.vals + ((size_t)rb * MG * kRB + ri) * GB;
        const int melem = w.meta_elem;
        const char * wmp = (const char *)w.meta + ((size_t)rb * MG * kRB + ri) * melem;
        const int g0 = (int)((long)sp * MG / n_split);
        const int g1 = (int)((long)(sp + 1) * MG / n_split);
        float acc = 0.f;
        for (int g = g0; g < g1; g++) {
            uint32_t gw[WGN];
            w8_group_expand<QT>(wp + (size_t)g * kRB * GB, gw);
            float sw, mw;
            w8_sw_mw<QT>(wmp, (size_t)g * kRB, melem, sw, mw);
            const int gx = (G == 32) ? g : (g >> 1);
            const float sx = xmeta[gx].x();
            const sycl::int2 sq = reinterpret_cast<const sycl::int2 *>(xsumq)[gx];
            const float c = (G == 32) ? (float)(sq.x() + sq.y()) : (float)((g & 1) ? sq.y() : sq.x());
            int32_t d = 0;
#pragma unroll
            for (int h = 0; h < G / 16; h++) {
                const int xo = (G == 32) ? h : (g & 1);
                const uint4 xw = *reinterpret_cast<const uint4 *>(x8 + (size_t)gx * 32 + xo * 16);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    d = dp4a_s8u8((int32_t)xw[w8_xword<QT>(j)], gw[h * 4 + j], d);
                }
            }
            acc += sx * (sw * (float)d - mw * c);
        }
        if constexpr (SPLIT) {
            ws[(size_t)sp * N + row] = acc;
        } else {
            float v = alpha * acc;
            if (residual) {
                v += residual[row];
            }
            out[row] = v;
        }
    });
}

// reduce the K-split GEMV partials (TB = 1)
static void dp4a_gemv_reduce_impl(queue & q, const float * ws, int n_split, float * out, int N, const float * residual,
                                  float alpha) {
    q.parallel_for((size_t)N, [=](id<1> i) {
        float v = 0.f;
        for (int s = 0; s < n_split; s++) {
            v += ws[(size_t)s * N + i];
        }
        v = alpha * v;
        if (residual) {
            v += residual[i];
        }
        out[i] = v;
    });
}
void dp4a_gemv_launch(queue & q, const w8t & w, const int8_t * x8, const sycl::float2 * xmeta, const int32_t * xsumq,
                      float * out, const float * residual, float alpha, int n_rows_out) {
    (void)n_rows_out;
    // decode is also latency-bound for small output sizes (4-8 workgroups), so
    // split K the same way as the prefill GEMM
    static const bool split_on = [] {
        const char * e = si::env::str("PF_GEMV_SPLIT");
        return !e || atoi(e) != 0;
    }();
    // how many output rows it takes to fill the machine: a device-profile value,
    // since it scales with the EU count
    const si::dev::profile & dp = si::dev::for_queue(q);
    const int fill_rows = dp.split.gemv_rows;
    const int split_cap = dp.split.max;
    int S = 1;
    if (split_on && w.N <= fill_rows * 2) {
        S = (int)((fill_rows + w.N - 1) / w.N);
        if (S > split_cap) {
            S = split_cap;
        }
    }
    if (S > 1) {
        float * ws = gemm_ws(q, (size_t)S * w.N);
        if (ws) {
            switch (w.type) {
            case 12: dp4a_gemv_impl<12, true>(q, w, x8, xmeta, xsumq, out, residual, alpha, S, ws); break;
            case 13: dp4a_gemv_impl<13, true>(q, w, x8, xmeta, xsumq, out, residual, alpha, S, ws); break;
            default: dp4a_gemv_impl<14, true>(q, w, x8, xmeta, xsumq, out, residual, alpha, S, ws); break;
            }
            dp4a_gemv_reduce_impl(q, ws, S, out, w.N, residual, alpha);
            return;
        }
    }
    switch (w.type) {
    case 12: dp4a_gemv_impl<12, false>(q, w, x8, xmeta, xsumq, out, residual, alpha); break;
    case 13: dp4a_gemv_impl<13, false>(q, w, x8, xmeta, xsumq, out, residual, alpha); break;
    default: dp4a_gemv_impl<14, false>(q, w, x8, xmeta, xsumq, out, residual, alpha); break;
    }
}

// ---------------------------------------------------------------------------
// Single-token decode GEMV over a row-major int8 weight [N][K] + per-row scale
// (the oneDNN conversion's layout).  The oneDNN primitive itself is dominated
// by per-call overhead at M=1, so the decode path reads its weight buffer
// directly.  One workgroup per output row, 256 lanes reducing over K.
// ---------------------------------------------------------------------------
void i8_row_gemv_launch(sycl::queue & q, const int8_t * w, const float * sw, const int8_t * xq, const float * sx_dev,
                        float * out, int K, int N, const float * residual, float alpha) {
    // one 256-thread workgroup per output row: every lane reduces over K
    // (coalesced int8 loads), then a workgroup reduction.  The single
    // activation row is shared by all workgroups (L1/L2 resident).
    constexpr int WG = 256;
    q.parallel_for(nd_range<1>((size_t)N * WG, WG), [=](nd_item<1> it) {
        const int n = (int)it.get_group(0);
        const int lane = (int)it.get_local_id(0);
        const float sx = sx_dev[0]; // device-side read (USM)
        const int8_t * wr = w + (size_t)n * K;
        int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        int k = lane;
        for (; k + 3 * WG < K; k += 4 * WG) {
            a0 += (int)wr[k] * (int)xq[k];
            a1 += (int)wr[k + WG] * (int)xq[k + WG];
            a2 += (int)wr[k + 2 * WG] * (int)xq[k + 2 * WG];
            a3 += (int)wr[k + 3 * WG] * (int)xq[k + 3 * WG];
        }
        for (; k < K; k += WG) {
            a0 += (int)wr[k] * (int)xq[k];
        }
        int acc = a0 + a1 + a2 + a3;
        acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<int>());
        if (lane == 0) {
            float v = alpha * sx * sw[n] * (float)acc;
            if (residual) {
                v += residual[n];
            }
            out[n] = v;
        }
    });
}

// R rows per sub-group: the 32 lanes split into R groups of 32/R lanes, each
// group streaming one weight row.  More rows per sub-group => fewer lanes per
// row => a deeper per-lane K loop (10 -> 40 iterations at R=4), which is what
// hides DRAM latency for this shallow GEMV.  Weights are XOR-biased to unsigned
// for dp4a and the 128*bias is removed once per row via the activation sum.
template <int R>
static void i8_row_gemv_multi_impl(queue & q, const gemv_seg * segs, int n_segs, int total_rows, const int8_t * xq,
                                   const float * sx_dev, const int32_t * xsum_dev) {
    constexpr int WG = 256;
    constexpr int SG = 32;
    constexpr int LPR = SG / R; // lanes per row
    constexpr int ROWS_PER_WG = (WG / SG) * R;
    const int n_wg = (total_rows + ROWS_PER_WG - 1) / ROWS_PER_WG;
    q.parallel_for(nd_range<1>((size_t)n_wg * WG, WG), [=](nd_item<1> it) {
        const int lane = (int)it.get_local_id(0) % SG;
        const int sgl = (int)it.get_local_id(0) / SG;
        const int r = lane / LPR;      // row within the sub-group
        const int ll = lane % LPR;     // lane within the row
        const int g = (int)it.get_group(0) * ROWS_PER_WG + sgl * R + r;
        if (g >= total_rows) {
            return;
        }
        const float sx = sx_dev[0];
        const int corr = xsum_dev ? xsum_dev[0] * 128 : 0;
        int s = 0, b = 0, nr = segs[0].n_rows;
        while (g >= b + nr && s + 1 < n_segs) {
            b += nr;
            s++;
            nr = segs[s].n_rows;
        }
        const gemv_seg & sgx = segs[s];
        const int row = g - b;
        const int K = sgx.K;
        const int8_t * wr = sgx.wi8 + (size_t)row * K;
        int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        // software pipelining: load the next iteration's 16-byte chunks while
        // the current four dp4a chains run, so the shallow K loop keeps enough
        // loads in flight to hide DRAM latency (no extra memory / layout).
        constexpr int stride = LPR * 16;
        const uint4 * xrow = reinterpret_cast<const uint4 *>(xq);
        const uint4 * wrow = reinterpret_cast<const uint4 *>(wr);
        constexpr int cs = LPR;   // chunk stride (16-byte units)
        const int nchunk = (K + 15) / 16;
        int ci = (ll < nchunk) ? ll : 0; // guard: short rows must not over-read
        uint4 xv = xrow[ci];
        uint4 wv = wrow[ci];
        for (int k = ll * 16; k < K; k += stride) {
            uint4 xn, wn;
            const bool more = (k + stride) < K;
            if (more) {
                xn = xrow[ci + cs];
                wn = wrow[ci + cs];
            }
            a0 = dp4a_s8u8((int32_t)xv.x(), wv.x() ^ 0x80808080u, a0);
            a1 = dp4a_s8u8((int32_t)xv.y(), wv.y() ^ 0x80808080u, a1);
            a2 = dp4a_s8u8((int32_t)xv.z(), wv.z() ^ 0x80808080u, a2);
            a3 = dp4a_s8u8((int32_t)xv.w(), wv.w() ^ 0x80808080u, a3);
            if (more) {
                xv = xn;
                wv = wn;
            }
            ci += cs;
        }
        int acc = a0 + a1 + a2 + a3;
        acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<int>());
        if (lane == 0) {
            float v = sgx.alpha * sx * sgx.wsc[row] * (float)(acc - corr);
            if (sgx.residual) {
                v += sgx.residual[row];
            }
            sgx.out[row] = v;
        }
    });
}

void i8_row_gemv_multi_launch(sycl::queue & q, const gemv_seg * segs, int n_segs, int total_rows, const int8_t * xq,
                              const float * sx_dev, const int32_t * xsum_dev) {
    static const int rows = [] {
        const char * e = si::env::str("PF_DEC_R");
        const int v = e ? atoi(e) : 1;
        return v == 1 ? 1 : (v == 2 ? 2 : (v == 4 ? 4 : 8));
    }();
    if (rows == 1) {
        i8_row_gemv_multi_impl<1>(q, segs, n_segs, total_rows, xq, sx_dev, xsum_dev);
    } else if (rows == 2) {
        i8_row_gemv_multi_impl<2>(q, segs, n_segs, total_rows, xq, sx_dev, xsum_dev);
    } else if (rows == 4) {
        i8_row_gemv_multi_impl<4>(q, segs, n_segs, total_rows, xq, sx_dev, xsum_dev);
    } else {
        i8_row_gemv_multi_impl<8>(q, segs, n_segs, total_rows, xq, sx_dev, xsum_dev);
    }
}

} // namespace si
