#include "kernels.h"
#include "kernel_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

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
        const char * e = getenv("PF_GEMV_SPLIT");
        return !e || atoi(e) != 0;
    }();
    int S = 1;
    if (split_on && w.N <= 4096) {
        S = (int)((2048 + w.N - 1) / w.N);
        if (S > 8) {
            S = 8;
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

} // namespace si
