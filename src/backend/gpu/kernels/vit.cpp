// ---------------------------------------------------------------------------
// Vision encoder kernels (Qwen3.5 "clip" tower).
//
// All activations are fp32; weights are BF16 or F32 (row-major [N][K]).  The
// token count is bounded by kMaxImgPatches, so the attention kernel keeps the
// running max/sum per query row (online softmax) instead of materializing the
// n_tok x n_tok score matrix.
// ---------------------------------------------------------------------------
#include "kernels.h"
#include "device/device_profile.h"
#include "common/rope.h"
#include "common/env.h"
#include "kernel_utils.h"
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <unordered_map>

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
    } else if constexpr (WT == GGML_TYPE_F16) {
        return ggml_half_to_float(((const uint16_t *)w)[idx]);
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
    } else if constexpr (WT == GGML_TYPE_F16) {
        const uint4 v = *reinterpret_cast<const uint4 *>((const uint16_t *)w + base);
        const uint32_t u[4] = {v.x(), v.y(), v.z(), v.w()};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            dst[2 * j] = ggml_half_to_float((uint16_t)(u[j] & 0xFFFFu));
            dst[2 * j + 1] = ggml_half_to_float((uint16_t)(u[j] >> 16));
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
                wrow_ptr[r] = (const char *)wv + (size_t)row * K * ((WT == GGML_TYPE_BF16 || WT == GGML_TYPE_F16) ? 2 : 4);
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
    } else if (wtype == GGML_TYPE_F16) {
        vit_gemm_impl<GGML_TYPE_F16, 4, 4, 8>(q, w, N, K, x, x_stride, out, out_stride, T, alpha, residual);
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

// Qwen2-VL's vision blocks use quick_gelu: x * sigmoid(1.702*x).
void vit_quick_gelu_launch(queue & q, float * x, int n) {
    if (n <= 0) {
        return;
    }
    q.parallel_for(sycl::range<1>((size_t)n), [=](id<1> i) {
        const float v = x[i];
        x[i] = v / (1.0f + sycl::exp(-1.702f * v));
    });
}

// exact (erf) GELU, used by the patch merger (torch nn.GELU)
void vit_gelu_errf_launch(queue & q, float * x, int n) {
    if (n <= 0) {
        return;
    }
    q.parallel_for(sycl::range<1>((size_t)n), [=](id<1> i) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + sycl::erf(v * 0.7071067811865476f));
    });
}

// f32 GEMM path: the ffn_down input is silu(gate) * up.
void vit_silu_mul_launch(queue & q, const float * gate, const float * up, int stride, float * out, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    q.parallel_for(sycl::nd_range<1>((size_t)rows * 256, 256), [=](sycl::nd_item<1> it) {
        const int r = (int)it.get_group(0);
        const float * g = gate + (size_t)r * stride;
        const float * u = up + (size_t)r * stride;
        float * o = out + (size_t)r * cols;
        for (int i = (int)it.get_local_id(0); i < cols; i += 256) {
            const float v = g[i];
            o[i] = (v / (1.0f + sycl::exp(-v))) * u[i];
        }
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

void vit_copy_launch(queue & q, const float * x, int x_stride, float * out, int out_stride, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    q.parallel_for(sycl::range<2>((size_t)rows, (size_t)cols),
                   [=](id<2> i) { out[(size_t)i[0] * out_stride + i[1]] = x[(size_t)i[0] * x_stride + i[1]]; });
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
        const float theta = rope_theta((float)pos, p, half, log2b);
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
// BQ and BK are both part of the instantiation because the work-group's local
// memory is: s_q + s_acc = 2*BQ*HDP floats, s_k + s_v = 2*BK*HDP,
// s_s = BQ*(BK+4), with HDP = HD + 4.  At HD=72 with BQ=BK=64 that is 65.5 KB,
// over the 64 KB per-work-group limit every Xe-LP part has - and Level Zero
// reports a refused local allocation as UR_RESULT_ERROR_OUT_OF_RESOURCES,
// which reads like a full card when it is a 1.5 KB overshoot on a work-group.
// The query tile is the one to halve: BK is woven through the online softmax
// (the four-lanes-per-row reduction, the P/V inner loop), while BQ only
// selects how many query rows share the staged K/V.
template <int HD, int BQ, int BK, int WG = 128, int SG = 32>
static void vit_attn_impl(queue & q, const float * qkv, int qkv_stride, float * out, int out_stride, int n_tok,
                          int n_head, float scale) {
    if (n_tok <= 0 || n_head <= 0) {
        return;
    }
    constexpr int HDP = HD + 4;
    constexpr int SDP = BK + 4;
    const int n_qb = (n_tok + BQ - 1) / BQ;
    const int embd = n_head * HD;
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

        h.parallel_for(nd_range<1>((size_t)total_qb * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
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
                // 16-byte staged copies: the row base, the head offset and the
                // SLM stride are all multiples of 4 floats, so a float4 load
                // from the f32 qkv row lands aligned.  The scalar form this
                // replaces issued one 4-byte load per element with a /HD and
                // %HD per element (measured as part of the staging cost).
                constexpr int DV = HD / 4;
                for (int i = tid; i < BK * DV; i += WG) {
                    const int r = i / DV, d4 = (i % DV) * 4;
                    if (r < n_k) {
                        const float * row = qkv + (size_t)(k0 + r) * qkv_stride;
                        const float4 kk = *reinterpret_cast<const float4 *>(row + embd + hd * HD + d4);
                        const float4 vv = *reinterpret_cast<const float4 *>(row + 2 * embd + hd * HD + d4);
                        *reinterpret_cast<float4 *>(&s_k[r * HDP + d4]) = kk;
                        *reinterpret_cast<float4 *>(&s_v[r * HDP + d4]) = vv;
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
                        // Four independent accumulators: the single-accumulator
                        // form is an 80-deep dependent FMA chain per pair, and
                        // this loop is the kernel's hottest path (measured 95%
                        // of the vision forward once the linears moved to
                        // oneDNN).  The summation order changes, so the result
                        // differs at the last ulp - the md5 check is the oracle.
                        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
                        for (int d = 0; d < HD; d += 4) {
                            const float4 a = *reinterpret_cast<const float4 *>(qr + d);
                            const float4 b = *reinterpret_cast<const float4 *>(kr + d);
                            a0 = sycl::fma(a.x(), b.x(), a0);
                            a1 = sycl::fma(a.y(), b.y(), a1);
                            a2 = sycl::fma(a.z(), b.z(), a2);
                            a3 = sycl::fma(a.w(), b.w(), a3);
                        }
                        acc = ((a0 + a1) + (a2 + a3)) * scale;
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
                            // four lanes per query row, so each owns HD/4 of the
                            // accumulator.  This was `j * 16` with a fixed trip
                            // count of 16, which only covers 64 dims: at HD=72
                            // the last 8 dims were never rescaled when the
                            // running max rose, so they kept the previous
                            // tile's magnitude and the attention output came out
                            // with visibly more energy than the reference.
                            constexpr int DQ = HD / 4;
                            float * ar = &s_acc[r * HDP + j * DQ];
#pragma unroll
                            for (int d = 0; d < DQ; d++) {
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

// ---------------------------------------------------------------------------
// ILP variant of the tiled kernel: same arithmetic order, more independent
// accumulator chains per thread.
//
// The page output is bit-sensitive to the attention's fp summation ORDER (a
// f32 oneDNN matmul, which is numerically *better* than this kernel, flips the
// page md5), so the only shippable changes are ones that leave every per-score
// dot and every per-dim k-accumulation in the same order.  This kernel does:
// each score still runs a0..a3 over d%4 and then (a0+a1)+(a2+a3), and each PV
// dimension still accumulates k=0..n_k-1 in order.  It only changes how many
// of them one thread owns.
//
// Score: one thread owns TWO query rows for one key (8 accumulators, 3 float4
// loads per 8 FMAs instead of 2 per 4).  PV: one thread owns TWO rows for one
// 4-dim group, so the V float4 is loaded once for both rows.
template <int HD, int BQ, int BK, int WG = 1024, int SG = 16, int RQ = 2, int RQP = 2>
static void vit_attn_ilp(queue & q, const float * qkv, int qkv_stride, float * out, int out_stride, int n_tok,
                         int n_head, float scale) {
    if (n_tok <= 0 || n_head <= 0) {
        return;
    }
    constexpr int HDP = HD + 4;
    constexpr int SDP = BK + 4;
    static_assert(BQ % RQ == 0 && BQ % RQP == 0, "row block must divide BQ");
    const int n_qb = (n_tok + BQ - 1) / BQ;
    const int embd = n_head * HD;
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

        h.parallel_for(nd_range<1>((size_t)total_qb * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
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
                constexpr int DV = HD / 4;
                for (int i = tid; i < BK * DV; i += WG) {
                    const int r = i / DV, d4 = (i % DV) * 4;
                    if (r < n_k) {
                        const float * row = qkv + (size_t)(k0 + r) * qkv_stride;
                        const float4 kk = *reinterpret_cast<const float4 *>(row + embd + hd * HD + d4);
                        const float4 vv = *reinterpret_cast<const float4 *>(row + 2 * embd + hd * HD + d4);
                        *reinterpret_cast<float4 *>(&s_k[r * HDP + d4]) = kk;
                        *reinterpret_cast<float4 *>(&s_v[r * HDP + d4]) = vv;
                    }
                }
                it.barrier();

                // RQ query rows per thread for one key.  Every score keeps the
                // a0..a3/d%4 order and the (a0+a1)+(a2+a3) reduction.
                {
                    constexpr int NT = (BQ / RQ) * BK;
                    for (int i = tid; i < NT; i += WG) {
                        const int qg = i / BK, k = i % BK;
                        const int r0 = qg * RQ;
                        float a0[RQ], a1[RQ], a2[RQ], a3[RQ];
#pragma unroll
                        for (int t = 0; t < RQ; t++) {
                            a0[t] = a1[t] = a2[t] = a3[t] = 0.f;
                        }
                        if (k < n_k) {
                            const float * kr = &s_k[k * HDP];
#pragma unroll
                            for (int d = 0; d < HD; d += 4) {
                                const float4 kv = *reinterpret_cast<const float4 *>(kr + d);
#pragma unroll
                                for (int t = 0; t < RQ; t++) {
                                    const float4 qv = *reinterpret_cast<const float4 *>(&s_q[(r0 + t) * HDP + d]);
                                    a0[t] = sycl::fma(qv.x(), kv.x(), a0[t]);
                                    a1[t] = sycl::fma(qv.y(), kv.y(), a1[t]);
                                    a2[t] = sycl::fma(qv.z(), kv.z(), a2[t]);
                                    a3[t] = sycl::fma(qv.w(), kv.w(), a3[t]);
                                }
                            }
                        }
#pragma unroll
                        for (int t = 0; t < RQ; t++) {
                            if (r0 + t < n_q) {
                                s_s[(r0 + t) * SDP + k] =
                                    (k < n_k) ? ((a0[t] + a1[t]) + (a2[t] + a3[t])) * scale : 0.f;
                            }
                        }
                    }
                }
                it.barrier();

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
                            constexpr int DQ = HD / 4;
                            float * ar = &s_acc[r * HDP + j * DQ];
#pragma unroll
                            for (int d = 0; d < DQ; d++) {
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
                    float ss = s_l[r];
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        ss += s_red[r * 4 + j];
                    }
                    s_l[r] = ss;
                }
                it.barrier();

                // RQP rows per thread for one 4-dim group: V loaded once.
                {
                    constexpr int DB = HD / 4;
                    constexpr int NT = (BQ / RQP) * DB;
                    for (int i = tid; i < NT; i += WG) {
                        const int rg = i / DB, db = i % DB;
                        const int r0 = rg * RQP;
                        if (r0 >= n_q) {
                            continue;
                        }
                        float4 acc[RQP];
#pragma unroll
                        for (int t = 0; t < RQP; t++) {
                            acc[t] = {0.f, 0.f, 0.f, 0.f};
                        }
                        for (int k = 0; k < n_k; k++) {
                            const float4 v = *reinterpret_cast<const float4 *>(&s_v[k * HDP + db * 4]);
#pragma unroll
                            for (int t = 0; t < RQP; t++) {
                                if (r0 + t < n_q) {
                                    const float p = s_s[(r0 + t) * SDP + k];
                                    acc[t].x() = sycl::fma(p, v.x(), acc[t].x());
                                    acc[t].y() = sycl::fma(p, v.y(), acc[t].y());
                                    acc[t].z() = sycl::fma(p, v.z(), acc[t].z());
                                    acc[t].w() = sycl::fma(p, v.w(), acc[t].w());
                                }
                            }
                        }
#pragma unroll
                        for (int t = 0; t < RQP; t++) {
                            if (r0 + t < n_q) {
                                float4 * ap = reinterpret_cast<float4 *>(&s_acc[(r0 + t) * HDP + db * 4]);
                                const float4 cur = *ap;
                                *ap = {cur.x() + acc[t].x(), cur.y() + acc[t].y(), cur.z() + acc[t].z(),
                                       cur.w() + acc[t].w()};
                            }
                        }
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

// ---------------------------------------------------------------------------
// f32 QK/PV on oneDNN.
//
// The two GEMMs are plain f32 matmuls: no activation quantisation and PV uses
// the exact softmax weights.  Opt-in (`PF_VIT_ATTN=51`) because, although it is
// numerically *better* than the hand kernel (self-check cos 0.999142 against
// 0.999104), its different fp summation order flips the page md5 - the page is
// bit-sensitive to attention rounding, so a different order cannot ship as the
// default.  It is deterministic (an int8 variant, evaluated separately, was
// not: oneDNN's int8 matmul is nondeterministic at these shapes).
//
// Measured on the A770 (probe): QK [5476,80]x[80,2048] 0.32 ms and PV
// [5476,2048]x[2048,80] 0.24-0.46 ms, ~2.8 T-MAC/s, versus 0.86 T-MAC/s for the
// hand kernel.  The primitives are cached by (M,K,N) and the query/key extents
// are bucketed (multiples of 1024 / 512-1024-2048) so one page's 21 crops share
// a couple of primitives instead of creating ~80 of them.
//
// Bucketing pads the query rows (zeroed Q -> harmless) and the key columns.
// The key pad is never read by the softmax (it only loops k < nb) and its P
// column is explicitly zeroed, so a stale/garbage V pad contributes 0.
namespace {

constexpr int kOdNBmax = 2048;
constexpr int kOdTile = 1024;

struct vit_od_bufs {
    int cap_m = 0;
    float * qc = nullptr;  // [M][80]
    float * kc = nullptr;  // [NBmax][80]
    float * vc = nullptr;  // [NBmax][80]
    float * sc = nullptr;  // [M][NBmax]
    float * p = nullptr;   // [M][NBmax]
    float * pv = nullptr;  // [M][80]
    float * acc = nullptr; // [M][80]
    float * mrow = nullptr;
    float * lrow = nullptr;
    float * corr = nullptr;
};

static vit_od_bufs g_od;

static dnnl::engine & od_engine(sycl::queue & q) {
    static dnnl::engine e = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    return e;
}
static dnnl::stream & od_stream(sycl::queue & q) {
    static dnnl::stream s = dnnl::sycl_interop::make_stream(od_engine(q), q);
    return s;
}

struct od_prim {
    dnnl::matmul mm;
    dnnl::memory::desc x, w, d;
};

static const od_prim * od_get(std::unordered_map<uint64_t, od_prim> & cache, sycl::queue & q, int M, int K, int N,
                              bool qk) {
    const uint64_t key = ((uint64_t)M << 42) | ((uint64_t)K << 21) | (uint64_t)N | ((uint64_t)qk << 63);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return &it->second;
    }
    using dt = dnnl::memory::data_type;
    using tg = dnnl::memory::format_tag;
    od_prim e;
    e.x = dnnl::memory::desc({M, K}, dt::f32, tg::ab);
    e.w = qk ? dnnl::memory::desc({K, N}, dt::f32, tg::ba) : dnnl::memory::desc({K, N}, dt::f32, tg::ab);
    e.d = dnnl::memory::desc({M, N}, dt::f32, tg::ab);
    e.mm = dnnl::matmul(dnnl::matmul::primitive_desc(od_engine(q), e.x, e.w, e.d));
    return &cache.emplace(key, std::move(e)).first->second;
}

static bool vit_od_ready(sycl::queue & q, int M) {
    vit_od_bufs & b = g_od;
    if (M <= b.cap_m) {
        return true;
    }
    auto fre = [&](void * p) {
        if (p) {
            sycl::free(p, q);
        }
    };
    fre(b.qc);
    fre(b.kc);
    fre(b.vc);
    fre(b.sc);
    fre(b.p);
    fre(b.pv);
    fre(b.acc);
    fre(b.mrow);
    fre(b.lrow);
    fre(b.corr);
    b.cap_m = M;
    b.qc = sycl::malloc_device<float>((size_t)M * 80, q);
    b.kc = sycl::malloc_device<float>((size_t)kOdNBmax * 80, q);
    b.vc = sycl::malloc_device<float>((size_t)kOdNBmax * 80, q);
    b.sc = sycl::malloc_device<float>((size_t)M * kOdNBmax, q);
    b.p = sycl::malloc_device<float>((size_t)M * kOdNBmax, q);
    b.pv = sycl::malloc_device<float>((size_t)M * 80, q);
    b.acc = sycl::malloc_device<float>((size_t)M * 80, q);
    b.mrow = sycl::malloc_device<float>((size_t)M, q);
    b.lrow = sycl::malloc_device<float>((size_t)M, q);
    b.corr = sycl::malloc_device<float>((size_t)M, q);
    if (!b.qc || !b.kc || !b.vc || !b.sc || !b.p || !b.pv || !b.acc || !b.mrow || !b.lrow || !b.corr) {
        return false;
    }
    // padded query rows stay zero (only rows < np are ever written)
    q.memset(b.qc, 0, (size_t)M * 80 * sizeof(float));
    // K/V pad rows may hold stale finite values; P is zeroed there, so they
    // contribute 0.  Zero once so nothing is uninitialised.
    q.memset(b.kc, 0, (size_t)kOdNBmax * 80 * sizeof(float));
    q.memset(b.vc, 0, (size_t)kOdNBmax * 80 * sizeof(float));
    return true;
}

static void vit_od_copy_q(queue & q, const float * qkv, int qkv_stride, int np, int hd, float scale, float * qc) {
    q.parallel_for(sycl::range<2>((size_t)np, 80), [=](id<2> i) {
        const int r = (int)i[0], d = (int)i[1];
        qc[(size_t)r * 80 + d] = qkv[(size_t)r * qkv_stride + hd * 80 + d] * scale;
    });
}

static void vit_od_copy_kv(queue & q, const float * qkv, int qkv_stride, int k0, int nb, int E, int hd, float * kc,
                           float * vc) {
    q.parallel_for(sycl::range<2>((size_t)nb, 80), [=](id<2> i) {
        const int r = (int)i[0], d = (int)i[1];
        const float * row = qkv + (size_t)(k0 + r) * qkv_stride;
        kc[(size_t)r * 80 + d] = row[E + hd * 80 + d];
        vc[(size_t)r * 80 + d] = row[2 * E + hd * 80 + d];
    });
}

static void vit_od_softmax(queue & q, const float * sc, int M, int NB, int nb, float * p, float * mrow, float * lrow,
                           float * corr) {
    q.submit([&](handler & h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(8), h);
        h.parallel_for(nd_range<1>((size_t)M * 256, 256), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int row = (int)it.get_group(0);
            const int tid = it.get_local_id(0);
            const int sg = tid / 32, lane = tid % 32;
            const float * sr = sc + (size_t)row * NB;
            float bm = -INFINITY;
            for (int k = tid; k < nb; k += 256) {
                bm = sycl::fmax(bm, sr[k]);
            }
            bm = sycl::reduce_over_group(it.get_sub_group(), bm, sycl::maximum<float>());
            if (lane == 0) {
                red[sg] = bm;
            }
            it.barrier();
            if (tid == 0) {
                float mn = mrow[row];
                for (int j = 0; j < 8; j++) {
                    mn = sycl::fmax(mn, red[j]);
                }
                red[0] = mn;
            }
            it.barrier();
            const float mnew = red[0];
            const float cf = sycl::exp(mrow[row] - mnew);
            float * pr = p + (size_t)row * NB;
            float ps = 0.f;
            for (int k = tid; k < nb; k += 256) {
                const float pv = sycl::exp(sr[k] - mnew);
                pr[k] = pv;
                ps += pv;
            }
            for (int k = nb + tid; k < NB; k += 256) {
                pr[k] = 0.f;
            }
            ps = sycl::reduce_over_group(it.get_sub_group(), ps, sycl::plus<float>());
            if (lane == 0) {
                red[sg] = ps;
            }
            it.barrier();
            if (tid == 0) {
                float ss = 0.f;
                for (int j = 0; j < 8; j++) {
                    ss += red[j];
                }
                lrow[row] = lrow[row] * cf + ss;
                mrow[row] = mnew;
                corr[row] = cf;
            }
        });
    });
}

static void vit_od_acc(queue & q, const float * pv, const float * corr, float * acc, int M) {
    q.parallel_for(sycl::range<2>((size_t)M, 80), [=](id<2> i) {
        const int r = (int)i[0], d = (int)i[1];
        acc[(size_t)r * 80 + d] = acc[(size_t)r * 80 + d] * corr[r] + pv[(size_t)r * 80 + d];
    });
}

static bool vit_attn_od_launch(queue & q, const float * qkv, int qkv_stride, float * out, int out_stride, int n_tok,
                               int n_head, int head_dim, float scale) {
    if (head_dim != 80) {
        return false;
    }
    const int np = n_tok;
    const int M = std::max(kOdTile, ((np + kOdTile - 1) / kOdTile) * kOdTile);
    const int NB = np <= 512 ? 512 : (np <= 1024 ? 1024 : kOdNBmax);
    if (!vit_od_ready(q, M)) {
        return false;
    }
    vit_od_bufs & b = g_od;
    const int E = n_head * head_dim;
    static std::unordered_map<uint64_t, od_prim> qk_cache, pv_cache;
    const od_prim * qk = od_get(qk_cache, q, M, 80, NB, true);
    const od_prim * pvp = od_get(pv_cache, q, M, NB, 80, false);
    if (!qk || !pvp) {
        return false;
    }
    dnnl::engine & eng = od_engine(q);
    dnnl::stream & st = od_stream(q);
    auto mk = [&](const dnnl::memory::desc & md, void * ptr) {
        return dnnl::sycl_interop::make_memory(md, eng, dnnl::sycl_interop::memory_kind::usm, ptr);
    };
    for (int hd = 0; hd < n_head; hd++) {
        // capture pointers, not the `vit_od_bufs &` (a captured reference makes
        // the device read the host address of the struct; see the int8 path)
        float * mrp = b.mrow;
        float * lrp = b.lrow;
        float * crp = b.corr;
        q.parallel_for(sycl::range<1>((size_t)M), [=](id<1> i) {
            mrp[i[0]] = -INFINITY;
            lrp[i[0]] = 0.f;
            crp[i[0]] = 0.f;
        });
        q.memset(b.acc, 0, (size_t)M * 80 * sizeof(float));
        vit_od_copy_q(q, qkv, qkv_stride, np, hd, scale, b.qc);
        for (int k0 = 0; k0 < np; k0 += NB) {
            const int nb = std::min(NB, np - k0);
            vit_od_copy_kv(q, qkv, qkv_stride, k0, nb, E, hd, b.kc, b.vc);
            {
                auto xm = mk(qk->x, b.qc);
                auto wm = mk(qk->w, b.kc);
                auto dm = mk(qk->d, b.sc);
                qk->mm.execute(st, {{DNNL_ARG_SRC, xm}, {DNNL_ARG_WEIGHTS, wm}, {DNNL_ARG_DST, dm}});
            }
            vit_od_softmax(q, b.sc, M, NB, nb, b.p, b.mrow, b.lrow, b.corr);
            {
                auto xm = mk(pvp->x, b.p);
                auto wm = mk(pvp->w, b.vc);
                auto dm = mk(pvp->d, b.pv);
                pvp->mm.execute(st, {{DNNL_ARG_SRC, xm}, {DNNL_ARG_WEIGHTS, wm}, {DNNL_ARG_DST, dm}});
            }
            vit_od_acc(q, b.pv, b.corr, b.acc, M);
        }
        float * accp = b.acc;
        float * lp = b.lrow;
        q.parallel_for(sycl::range<2>((size_t)np, 80), [=](id<2> i) {
            const int r = (int)i[0], d = (int)i[1];
            out[(size_t)r * out_stride + hd * head_dim + d] = accp[(size_t)r * 80 + d] / lp[r];
        });
    }
    return true;
}

} // namespace

// The vision head dim is a property of the projector, not of the kernel: 64 for
// the Qwen-VL towers this was written against, 72 for the Qwen3.6/3.5 mmproj
// files.  The workgroup layout is indexed by (query row, key) and never by
// head_dim, and the inner products are float4 over an HDP = HD + 4 SLM stride,
// so any multiple of 4 works unchanged - the only requirement is that it be
// instantiated here, which keeps the loop trip counts compile-time.
void vit_attn_launch(queue & q, const float * qkv, int qkv_stride, float * out, int out_stride, int n_tok, int n_head,
                     int head_dim, float scale) {
    switch (head_dim) {
    case 64: vit_attn_impl<64, 32, 64>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
    case 72: vit_attn_impl<72, 16, 64>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
    // Qwen2-VL: 16 heads x 80.  The page output is bit-sensitive to the
    // attention's fp summation order, so the shipped kernels keep the original
    // per-score dot order (a0..a3 over d%4, then (a0+a1)+(a2+a3)), the BK=32
    // online-softmax grouping, and the PV k order.  Every change below is
    // order-preserving; the page md5 stays 10eed240936226fd4696d943aaabc44b.
    //
    // Measured on the layout page (np=5476, 32 layers, one A770, AOT, attention
    // ms).  Baseline 16/64/128 = 10473.  With vectorized K/V staging:
    //   16/32/512 4658 | 32/32/256 4563 | 64/16/512 3335 | 32/32/1024 3263
    //   BQ=48/BK=32/WG=1024 2865 (more query rows per staged key block)
    //   ILP (2 query rows/thread) + SG=16 2076 (default)
    // The ILP kernel owns 2 query rows per score dot and 2 rows per PV 4-dim
    // group, so a thread has 8 independent accumulators instead of 4 and the
    // V float4 is loaded once for both rows.  SG=16 puts four 16-lane
    // sub-groups per EU (SG=32 gives two) for latency hiding; SG=8 loses.
    // PF_VIT_ATTN picks an alternative tile for the sweep.
    case 80: {
        int cfg = SI_ENV_I32("PF_VIT_ATTN", 70);
        // The 1024-thread variants fill a 512-EU A770 core; an Iris Xe caps a
        // work-group at 512, so fall back to the 32/32/512 tile there (the
        // launcher's own device query is cached by si::dev).
        if (si::dev::wg_clamped_for_queue(q, 1024) < 1024
            && (cfg == 2 || cfg == 5 || cfg == 6 || cfg == 60 || cfg == 70 || cfg == 71 || cfg == 72 || cfg == 73
                || cfg == 76)) {
            cfg = 0;
        }
        if (cfg == 51) {
            if (vit_attn_od_launch(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, head_dim, scale)) {
                break;
            }
            cfg = 70;
        }
        switch (cfg) {
        case 1: vit_attn_impl<80, 64, 16, 512>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        case 2: vit_attn_impl<80, 32, 32, 1024>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        case 3: vit_attn_impl<80, 16, 32, 512>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        case 4: vit_attn_impl<80, 32, 32, 256>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        case 5: vit_attn_impl<80, 64, 16, 1024>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        case 6: vit_attn_impl<80, 48, 32, 1024>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        case 7: vit_attn_impl<80, 16, 64, 1024>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        case 60: vit_attn_impl<80, 48, 32, 1024, 16>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        case 70: vit_attn_ilp<80, 48, 32, 1024, 16, 2, 2>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        case 71: vit_attn_ilp<80, 48, 32, 1024, 32, 2, 2>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        case 72: vit_attn_ilp<80, 48, 32, 1024, 16, 4, 2>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        case 73: vit_attn_ilp<80, 48, 32, 1024, 16, 2, 4>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        case 76: vit_attn_ilp<80, 52, 32, 1024, 16, 2, 2>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        case 0:
        default: vit_attn_impl<80, 32, 32, 512>(q, qkv, qkv_stride, out, out_stride, n_tok, n_head, scale); break;
        }
        break;
    }
    default: break; // vision_model::runnable() rejects anything else up front
    }
}


} // namespace si
