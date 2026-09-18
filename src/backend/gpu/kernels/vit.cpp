// ---------------------------------------------------------------------------
// Vision encoder kernels (Qwen3.5 "clip" tower).
//
// All activations are fp32; weights are BF16 or F32 (row-major [N][K]).  The
// token count is bounded by kMaxImgPatches, so the attention kernel keeps the
// running max/sum per query row (online softmax) instead of materializing the
// n_tok x n_tok score matrix.
// ---------------------------------------------------------------------------
#include "kernels.h"
#include "kernel_utils.h"

#include <cmath>

#include "gguf.h"

namespace si {

using namespace sycl;
using namespace si::kd;

namespace {

inline float bf16_to_f32(uint16_t h) {
    return sycl::bit_cast<float>((uint32_t)h << 16);
}

template <uint32_t WT> inline float load_w(const void * w, size_t idx) {
    if constexpr (WT == GGML_TYPE_BF16) {
        return bf16_to_f32(((const uint16_t *)w)[idx]);
    } else {
        return ((const float *)w)[idx];
    }
}

// Load 8 consecutive weights for one row (requires K % 8 == 0 and an 8-element
// aligned row base, which holds for all vision GEMMs).
template <uint32_t WT> inline void load_w8(const void * w, size_t base, float * dst) {
    if constexpr (WT == GGML_TYPE_BF16) {
        const uint4 v = *reinterpret_cast<const uint4 *>((const uint16_t *)w + base);
        const uint32_t u[4] = {v.x(), v.y(), v.z(), v.w()};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            dst[2 * j] = sycl::bit_cast<float>((u[j] & 0xFFFFu) << 16);
            dst[2 * j + 1] = sycl::bit_cast<float>(u[j] & 0xFFFF0000u);
        }
    } else {
        const float4 a = *reinterpret_cast<const float4 *>((const float *)w + base);
        const float4 b = *reinterpret_cast<const float4 *>((const float *)w + base + 4);
        dst[0] = a.x();
        dst[1] = a.y();
        dst[2] = a.z();
        dst[3] = a.w();
        dst[4] = b.x();
        dst[5] = b.y();
        dst[6] = b.z();
        dst[7] = b.w();
    }
}

// ---------------------------------------------------------------------------
// GEMM: out[t][n] = alpha * sum_k W[n][k] * x[t][k] (+ residual[t][n]).
//
// One 128-thread workgroup covers a 64-row x 32-token tile; each lane owns
// TP=4 tokens x RP=4 rows and streams K in 64-wide chunks staged in SLM.  (A
// wider TP=8/KU=4 tiling was measured against this one on every real shape:
// it is faster only with f32 weights, and the vision linears are bf16, so the
// narrow tiling is kept for both.)
// ---------------------------------------------------------------------------
template <uint32_t WT, int TP, int RP, int KU>
static void vit_gemm_impl(queue & q, const void * wv, int N, int K, const float * x, int x_stride, float * out,
                          int out_stride, int T, float alpha, const float * residual) {
    if (N <= 0 || K <= 0 || T <= 0) {
        return;
    }
    constexpr int TG = 32 / TP; // token groups per sub-group
    constexpr int RG = 32 / TG; // row groups per sub-group
    constexpr int TT = TG * TP; // tokens per workgroup (32)
    constexpr int WG = 128;
    constexpr int warp_rows = RG * RP;
    constexpr int rows_per_wg = (WG / 32) * warp_rows;
    constexpr int KS = 64;      // K chunk staged in SLM
    constexpr int KVT = KU / 4; // float4 loads per k-group
    const int n_rt = (N + rows_per_wg - 1) / rows_per_wg;
    const int n_tt = (T + TT - 1) / TT;

    q.submit([&](handler & h) {
        // [token][k] so a lane's 8 consecutive k values are contiguous
        sycl::local_accessor<float, 1> xs(sycl::range<1>((size_t)TT * KS), h);
        h.parallel_for(nd_range<1>((size_t)n_rt * n_tt * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int gid = it.get_group(0);
            const int rt = gid % n_rt;
            const int tt = gid / n_rt;
            const int wid = it.get_local_id(0) / 32;
            const int lane = it.get_local_id(0) % 32;
            const int tg = lane % TG;
            const int rg = lane / TG;
            const int tok0 = tt * TT;
            const int t0 = tok0 + tg * TP;
            const int row_base = rt * rows_per_wg;
            const int row0 = row_base + wid * warp_rows + rg * RP;
            const int rowc = sycl::min(row0, sycl::max(0, N - RP));
            // weight row bases hoisted out of the k loop (the inner loop then only
            // adds the k offset, no 64-bit row*K multiply per step)
            const void * wrow_ptr[RP];
#pragma unroll
            for (int r = 0; r < RP; r++) {
                const int row = sycl::min(rowc + r, N - 1);
                wrow_ptr[r] = (const char *)wv + (size_t)row * K * (WT == GGML_TYPE_BF16 ? 2 : 4);
            }

            float acc[TP][RP];
#pragma unroll
            for (int t = 0; t < TP; t++) {
#pragma unroll
                for (int r = 0; r < RP; r++) {
                    acc[t][r] = 0.f;
                }
            }

            for (int k0 = 0; k0 < K; k0 += KS) {
                const int kc = sycl::min(KS, K - k0);
                for (int i = it.get_local_id(0); i < TT * KS; i += WG) {
                    const int t = i / KS, k = i % KS;
                    const int tk = tok0 + t;
                    xs[t * KS + k] = (k < kc && tk < T) ? x[(size_t)tk * x_stride + k0 + k] : 0.f;
                }
                it.barrier();
                const int kc8 = kc - (kc % KU);
                for (int k = 0; k < kc8; k += KU) {
                    float xv[KU][TP];
#pragma unroll
                    for (int t = 0; t < TP; t++) {
                        const float * xr = &xs[(tg * TP + t) * KS + k];
#pragma unroll
                        for (int v = 0; v < KVT; v++) {
                            const float4 a = *reinterpret_cast<const float4 *>(xr + 4 * v);
                            xv[4 * v + 0][t] = a.x();
                            xv[4 * v + 1][t] = a.y();
                            xv[4 * v + 2][t] = a.z();
                            xv[4 * v + 3][t] = a.w();
                        }
                    }
                    float wrow[RP][KU];
#pragma unroll
                    for (int r = 0; r < RP; r++) {
                        load_w8<WT>(wrow_ptr[r], k0 + k, wrow[r]);
                    }
#pragma unroll
                    for (int j = 0; j < KU; j++) {
#pragma unroll
                        for (int t = 0; t < TP; t++) {
#pragma unroll
                            for (int r = 0; r < RP; r++) {
                                acc[t][r] = sycl::fma(xv[j][t], wrow[r][j], acc[t][r]);
                            }
                        }
                    }
                }
                for (int k = kc8; k < kc; k++) {
#pragma unroll
                    for (int t = 0; t < TP; t++) {
                        const float xv = xs[(tg * TP + t) * KS + k];
#pragma unroll
                        for (int r = 0; r < RP; r++) {
                            acc[t][r] = sycl::fma(xv, load_w<WT>(wrow_ptr[r], k0 + k), acc[t][r]);
                        }
                    }
                }
                it.barrier();
            }

#pragma unroll
            for (int t = 0; t < TP; t++) {
                const int tok = t0 + t;
                if (tok >= T) {
                    break;
                }
                float * orow = out + (size_t)tok * out_stride;
                const float * rrow = residual ? residual + (size_t)tok * out_stride : nullptr;
#pragma unroll
                for (int r = 0; r < RP; r++) {
                    const int row = row0 + r;
                    if (row >= N) {
                        break;
                    }
                    float v = alpha * acc[t][r];
                    if (rrow) {
                        v += rrow[row];
                    }
                    orow[row] = v;
                }
            }
        });
    });
}

} // namespace

void vit_gemm_launch(queue & q, const void * w, uint32_t wtype, int N, int K, const float * x, int x_stride,
                     float * out, int out_stride, int T, float alpha, const float * residual) {
    if (wtype == GGML_TYPE_BF16) {
        vit_gemm_impl<GGML_TYPE_BF16, 4, 4, 8>(q, w, N, K, x, x_stride, out, out_stride, T, alpha, residual);
    } else {
        vit_gemm_impl<GGML_TYPE_F32, 4, 4, 8>(q, w, N, K, x, x_stride, out, out_stride, T, alpha, residual);
    }
}

// ---------------------------------------------------------------------------
// LayerNorm: one sub-group per row, lane-strided reduction then a shuffle sum.
// ---------------------------------------------------------------------------
void vit_layernorm_launch(queue & q, const float * x, int x_stride, const float * w, const float * b, float * out,
                          int out_stride, int rows, int n, float eps) {
    if (rows <= 0) {
        return;
    }
    constexpr int SG = 8; // sub-groups per workgroup
    const int n_wg = (rows + SG - 1) / SG;
    q.parallel_for(nd_range<1>((size_t)n_wg * 256, 256), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int sg = it.get_local_id(0) / 32;
        const int lane = it.get_local_id(0) % 32;
        const int row = it.get_group(0) * SG + sg;
        if (row >= rows) {
            return;
        }
        const float * xr = x + (size_t)row * x_stride;
        float s = 0.f, ss = 0.f;
        for (int i = lane; i < n; i += 32) {
            const float v = xr[i];
            s += v;
            ss = sycl::fma(v, v, ss);
        }
        const sub_group sgg = it.get_sub_group();
        s = sg_sum(s, sgg);
        ss = sg_sum(ss, sgg);
        const float mean = s / n;
        const float var = ss / n - mean * mean;
        const float inv = sycl::rsqrt(sycl::fmax(var, 0.f) + eps);
        float * orow = out + (size_t)row * out_stride;
        for (int i = lane; i < n; i += 32) {
            orow[i] = (xr[i] - mean) * inv * w[i] + (b ? b[i] : 0.f);
        }
    });
}

void vit_gelu_launch(queue & q, float * x, int n) {
    if (n <= 0) {
        return;
    }
    q.parallel_for(sycl::range<1>((size_t)n), [=](id<1> i) {
        const float v = x[i];
        const float c = 0.7978845608028654f; // sqrt(2/pi)
        x[i] = 0.5f * v * (1.0f + sycl::tanh(c * (v + 0.044715f * v * v * v)));
    });
}

void vit_add_bias_launch(queue & q, float * y, int y_stride, const float * bias, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    q.parallel_for(sycl::range<2>((size_t)rows, (size_t)cols),
                   [=](id<2> i) { y[(size_t)i[0] * y_stride + i[1]] += bias[i[1]]; });
}

void vit_add_launch(queue & q, float * y, int y_stride, const float * x, int x_stride, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    q.parallel_for(sycl::range<2>((size_t)rows, (size_t)cols),
                   [=](id<2> i) { y[(size_t)i[0] * y_stride + i[1]] += x[(size_t)i[0] * x_stride + i[1]]; });
}

// ---------------------------------------------------------------------------
// 2D vision RoPE on Q and K.  head_dim is split into four sections of
// head_dim/4 pairs; the first section uses the patch row, the second the patch
// column, and the frequency exponent restarts per section.
// ---------------------------------------------------------------------------
void vit_rope_launch(queue & q, float * qkv, int qkv_stride, int n_tok, int n_head, int head_dim, int out_w, int merge,
                     float rope_base) {
    if (n_tok <= 0 || n_head <= 0) {
        return;
    }
    const int half = head_dim / 2;
    const int n_pair = half; // pairs per token per head
    const int total = n_tok * n_head * n_pair;
    const float log2b = sycl::log2(rope_base);
    const int m2 = merge * merge;
    q.parallel_for(sycl::range<1>((size_t)total), [=](id<1> it) {
        const int pair = it[0] % n_pair;
        const int h = (it[0] / n_pair) % n_head;
        const int t = it[0] / (n_pair * n_head);
        // merged token -> pre-merge patch coordinate
        const int m = t / m2;
        const int sub = t % m2;
        const int my = out_w > 0 ? m / out_w : 0;
        const int mx = out_w > 0 ? m % out_w : 0;
        const int px = mx * merge + sub % merge;
        const int py = my * merge + sub / merge;
        const int sec = pair / (head_dim / 4); // 0 = row, 1 = col (2/3 unused here)
        const int p = pair % (head_dim / 4);
        const int pos = sec == 0 ? py : px;
        const float theta = (float)pos * sycl::exp2(-2.0f * p / (float)half * log2b);
        const float c = sycl::cos(theta), s = sycl::sin(theta);
        // Q and K live in the first two thirds of the fused qkv row
        float * qrow = qkv + (size_t)t * qkv_stride + h * head_dim;
        float * krow = qrow + n_head * head_dim;
        const float q0 = qrow[pair], q1 = qrow[pair + half];
        qrow[pair] = q0 * c - q1 * s;
        qrow[pair + half] = q0 * s + q1 * c;
        const float k0 = krow[pair], k1 = krow[pair + half];
        krow[pair] = k0 * c - k1 * s;
        krow[pair + half] = k0 * s + k1 * c;
    });
}

// ---------------------------------------------------------------------------
// Bidirectional attention.  One workgroup per (head, 16-query block); key/value
// tiles stream through SLM and the softmax is online (running max/sum), so the
// score matrix never leaves the workgroup.
// ---------------------------------------------------------------------------
void vit_attn_launch(queue & q, const float * qkv, int qkv_stride, float * out, int out_stride, int n_tok, int n_head,
                     int head_dim, float scale) {
    if (n_tok <= 0 || n_head <= 0) {
        return;
    }
    constexpr int BQ = 32;
    constexpr int BK = 64;
    constexpr int HD = 64; // vision head dim is fixed at 64
    if (head_dim != HD) {
        return; // callers guarantee this
    }
    constexpr int HDP = HD + 4;
    constexpr int SDP = BK + 4;
    constexpr int WG = 128;
    const int n_qb = (n_tok + BQ - 1) / BQ;
    const int embd = n_head * head_dim;
    const int total_qb = n_head * n_qb;

    q.submit([&](handler & h) {
        sycl::local_accessor<float, 1> s_q(sycl::range<1>((size_t)BQ * HDP), h);
        sycl::local_accessor<float, 1> s_k(sycl::range<1>((size_t)BK * HDP), h);
        sycl::local_accessor<float, 1> s_v(sycl::range<1>((size_t)BK * HDP), h);
        sycl::local_accessor<float, 1> s_s(sycl::range<1>((size_t)BQ * SDP), h);
        sycl::local_accessor<float, 1> s_acc(sycl::range<1>((size_t)BQ * HDP), h);
        sycl::local_accessor<float, 1> s_m(sycl::range<1>((size_t)BQ), h);
        sycl::local_accessor<float, 1> s_l(sycl::range<1>((size_t)BQ), h);
        sycl::local_accessor<float, 1> s_red(sycl::range<1>((size_t)BQ * 4), h);
        sycl::local_accessor<float, 1> s_rc(sycl::range<1>((size_t)BQ), h);

        h.parallel_for(nd_range<1>((size_t)total_qb * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int qb = it.get_group(0) % n_qb;
            const int hd = it.get_group(0) / n_qb;
            const int tid = it.get_local_id(0);
            const int q0 = qb * BQ;
            const int n_q = sycl::min(BQ, n_tok - q0);

            for (int i = tid; i < BQ; i += WG) {
                s_m[i] = -INFINITY;
                s_l[i] = 0.f;
            }
            for (int i = tid; i < BQ * HDP; i += WG) {
                s_acc[i] = 0.f;
            }
            for (int i = tid; i < BQ * HD; i += WG) {
                const int r = i / HD, d = i % HD;
                s_q[r * HDP + d] = (r < n_q) ? qkv[(size_t)(q0 + r) * qkv_stride + hd * HD + d] : 0.f;
            }
            it.barrier();

            for (int k0 = 0; k0 < n_tok; k0 += BK) {
                const int n_k = sycl::min(BK, n_tok - k0);
                for (int i = tid; i < BK * HD; i += WG) {
                    const int r = i / HD, d = i % HD;
                    if (r < n_k) {
                        const float * row = qkv + (size_t)(k0 + r) * qkv_stride;
                        s_k[r * HDP + d] = row[embd + hd * HD + d];
                        s_v[r * HDP + d] = row[2 * embd + hd * HD + d];
                    }
                }
                it.barrier();

                // scores: S[q][k] = dot(Q[q], K[k]) * scale
                for (int i = tid; i < BQ * BK; i += WG) {
                    const int r = i / BK, k = i % BK;
                    float acc = 0.f;
                    if (r < n_q && k < n_k) {
                        const float * qr = &s_q[r * HDP];
                        const float * kr = &s_k[k * HDP];
#pragma unroll
                        for (int d = 0; d < HD; d += 4) {
                            const float4 a = *reinterpret_cast<const float4 *>(qr + d);
                            const float4 b = *reinterpret_cast<const float4 *>(kr + d);
                            acc = sycl::fma(a.x(), b.x(), acc);
                            acc = sycl::fma(a.y(), b.y(), acc);
                            acc = sycl::fma(a.z(), b.z(), acc);
                            acc = sycl::fma(a.w(), b.w(), acc);
                        }
                        acc *= scale;
                    }
                    s_s[r * SDP + k] = acc;
                }
                it.barrier();

                // online softmax with 4 threads per query row (the row max, the
                // rescale, the exp and the partial sum are all spread over 128
                // threads instead of 32)
                {
                    const int r = tid / 4, j = tid % 4;
                    if (r < n_q) {
                        float pm = -INFINITY;
                        for (int k = j; k < n_k; k += 4) {
                            pm = sycl::fmax(pm, s_s[r * SDP + k]);
                        }
                        s_red[r * 4 + j] = pm;
                    }
                }
                it.barrier();
                if (tid < BQ && tid < n_q) {
                    const int r = tid;
                    float mnew = s_m[r];
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        mnew = sycl::fmax(mnew, s_red[r * 4 + j]);
                    }
                    const float corr = sycl::exp(s_m[r] - mnew);
                    s_rc[r] = corr;
                    s_l[r] *= corr;
                    s_m[r] = mnew;
                }
                it.barrier();
                {
                    const int r = tid / 4, j = tid % 4;
                    if (r < n_q) {
                        const float corr = s_rc[r];
                        if (corr != 1.f) {
                            float * ar = &s_acc[r * HDP + j * 16];
#pragma unroll
                            for (int d = 0; d < 16; d++) {
                                ar[d] *= corr;
                            }
                        }
                        const float mn = s_m[r];
                        float ps = 0.f;
                        for (int k = j; k < n_k; k += 4) {
                            const float p = sycl::exp(s_s[r * SDP + k] - mn);
                            s_s[r * SDP + k] = p;
                            ps += p;
                        }
                        s_red[r * 4 + j] = ps;
                    }
                }
                it.barrier();
                if (tid < BQ && tid < n_q) {
                    const int r = tid;
                    float s = s_l[r];
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        s += s_red[r * 4 + j];
                    }
                    s_l[r] = s;
                }
                it.barrier();

                // acc[q][d] += sum_k P[q][k] * V[k][d] (4 output dims per thread)
                {
                    constexpr int DB = HD / 4;
                    for (int i = tid; i < BQ * DB; i += WG) {
                        const int r = i / DB, db = i % DB;
                        if (r >= n_q) {
                            continue;
                        }
                        const float * pr = &s_s[r * SDP];
                        float4 acc = {0.f, 0.f, 0.f, 0.f};
                        for (int k = 0; k < n_k; k++) {
                            const float p = pr[k];
                            const float4 v = *reinterpret_cast<const float4 *>(&s_v[k * HDP + db * 4]);
                            acc.x() = sycl::fma(p, v.x(), acc.x());
                            acc.y() = sycl::fma(p, v.y(), acc.y());
                            acc.z() = sycl::fma(p, v.z(), acc.z());
                            acc.w() = sycl::fma(p, v.w(), acc.w());
                        }
                        float4 * ap = reinterpret_cast<float4 *>(&s_acc[r * HDP + db * 4]);
                        const float4 cur = *ap;
                        *ap = {cur.x() + acc.x(), cur.y() + acc.y(), cur.z() + acc.z(), cur.w() + acc.w()};
                    }
                }
                it.barrier();
            }

            for (int i = tid; i < BQ * HD; i += WG) {
                const int r = i / HD, d = i % HD;
                if (r >= n_q) {
                    continue;
                }
                out[(size_t)(q0 + r) * out_stride + hd * HD + d] = s_acc[r * HDP + d] / s_l[r];
            }
        });
    });
}

} // namespace si
