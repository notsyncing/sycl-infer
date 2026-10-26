// qwen4exp additions: hyper-connection mixing and the token-side helpers.
// These are the only new per-layer kernels: GDN, attention and the GEMVs all
// use the existing launchers.
#include "kernels.h"
#include "kernel_utils.h"
#include "common/env.h"
#include "device/device_profile.h"

namespace si {

using namespace sycl;
using namespace si::kd;

namespace {

// One workgroup per (token, stream).  n_embd is 2560 on the reference model.
template <int WG>
static sycl::event hc_norm_impl(queue & q, const float * x, const float * w, float * out, int n_tok, int n_embd, int hc,
                                float eps) {
    constexpr int NSG = WG / 32;
    return q.submit([&](handler & h) {
        local_accessor<float, 1> red(NSG, h);
        h.parallel_for(nd_range<1>((size_t)n_tok * hc * WG, WG), [=](nd_item<1> it) {
            const int j = it.get_group(0);
            const int i = j / hc;
            const int s = j % hc;
            const int tid = it.get_local_id(0);
            const sub_group sg = it.get_sub_group();
            const float * xr = x + ((size_t)i * hc + s) * n_embd;
            const float * g = w + (size_t)s * n_embd;
            float * orow = out + ((size_t)i * hc + s) * n_embd;
            float ss = 0.f;
            for (int k = tid; k < n_embd; k += WG) {
                ss += xr[k] * xr[k];
            }
            ss = reduce_over_group(sg, ss, plus<float>());
            const int sgl = (int)sg.get_local_linear_id();
            const int sgi = (int)sg.get_group_linear_id();
            if (sgl == 0) {
                red[sgi] = ss;
            }
            it.barrier();
            if (sgi == 0) {
                float v = (sgl < NSG && sgl < 32) ? red[sgl] : 0.f;
                v = reduce_over_group(sg, v, plus<float>());
                if (sgl == 0) {
                    red[0] = v;
                }
            }
            it.barrier();
            const float scale = 1.0f / sycl::sqrt(red[0] / n_embd + eps);
            for (int k = tid; k < n_embd; k += WG) {
                orow[k] = xr[k] * scale * g[k];
            }
        });
    });
}

} // namespace

sycl::event q4_hc_norm_launch(queue & q, const float * x, const float * w, float * out, int n_tok, int n_embd,
                              int hc, float eps) {
    const int want = si::dev::wg_clamped_for_queue(q, 256);
    if (want >= 512) {
        return hc_norm_impl<512>(q, x, w, out, n_tok, n_embd, hc, eps);
    }
    return hc_norm_impl<256>(q, x, w, out, n_tok, n_embd, hc, eps);
}

sycl::event q4_hc_mix_launch(queue & q, const float * xn, const float * gate, float * mixed, int n_tok, int n_embd,
                             int hc) {
    return q.parallel_for(range<2>((size_t)n_tok, (size_t)n_embd), [=](id<2> id) {
        const int i = (int)id[0];
        const int c = (int)id[1];
        float acc = 0.f;
        for (int s = 0; s < hc; s++) {
            const size_t o = ((size_t)i * hc + s) * n_embd + c;
            acc += xn[o] * sigmoid_f(gate[o]);
        }
        mixed[(size_t)i * n_embd + c] = acc / (float)hc;
    });
}

sycl::event q4_hc_combine_launch(queue & q, float * residual, const float * block_out, const float * inject, int n_tok,
                                 int n_embd, int hc) {
    return q.parallel_for(range<2>((size_t)n_tok * hc, (size_t)n_embd), [=](id<2> id) {
        const int j = (int)id[0];
        const int c = (int)id[1];
        const int i = j / hc;
        const int s = j % hc;
        const float wgt = 2.0f * sigmoid_f(inject[(size_t)i * hc + s] / (float)hc);
        residual[(size_t)j * n_embd + c] += block_out[(size_t)i * n_embd + c] * wgt;
    });
}

sycl::event q4_silu_scale_launch(queue & q, float * x, int n, float scale) {
    return q.parallel_for(range<1>((size_t)n), [=](id<1> i) { x[i[0]] = silu_f(x[i[0]] * scale); });
}

void q4_up_gemv_launch(queue & q, const void * w, uint32_t type, const float * x, int xs, float * out, int os,
                       int n_tok, int K, int N) {
    const int nb = K / 32;
    q.parallel_for(range<2>((size_t)n_tok, (size_t)N), [=](id<2> id) {
        const int t = (int)id[0];
        const int n = (int)id[1];
        const float * xr = x + (size_t)t * xs;
        float acc = 0.f;
        if (type == 8) {
            const char * rp = (const char *)w + (size_t)n * nb * 34;
            for (int b = 0; b < nb; b++) {
                const block_q8_0 * q8 = (const block_q8_0 *)(rp + (size_t)b * 34);
                const float d = fast_h2f(q8->d);
                for (int i = 0; i < 32; i++) {
                    acc += d * (float)q8->qs[i] * xr[b * 32 + i];
                }
            }
        } else if (type == 30) {
            const uint16_t * rp = (const uint16_t *)w + (size_t)n * K;
            for (int k = 0; k < K; k++) {
                acc += sycl::bit_cast<float>((uint32_t)rp[k] << 16) * xr[k];
            }
        } else if (type == 7) {
            const char * rp = (const char *)w + (size_t)n * nb * 24;
            for (int b = 0; b < nb; b++) {
                const block_q5_1 * q5 = (const block_q5_1 *)(rp + (size_t)b * 24);
                const float d = fast_h2f(q5->d), m = fast_h2f(q5->m);
                for (int i = 0; i < 32; i++) {
                    const int hi = (q5->qh[i / 8] >> (i % 8)) & 1;
                    const int lo = (q5->qs[i % 16] >> (4 * (i / 16))) & 0xF;
                    acc += (d * (float)(lo | (hi << 4)) + m) * xr[b * 32 + i];
                }
            }
        } else {
            // F32: the HC inject projection (K = hc_dim = 10240, N = hc = 4) is
            // the only caller.  One work-item per output row leaves four lanes
            // to stream 80 KB between them, so the plain loop is DRAM-latency
            // bound: the compiler emits each load immediately before the FMA
            // that consumes it and the warp stalls on every one.  Two 16-value
            // batches are loaded into registers before they are accumulated,
            // which puts 32 requests in flight; the per-element accumulation
            // order is unchanged, so the sum is bit-identical to the plain
            // loop (measured on the 104 GiB qwen4exp: 75 -> 29 ms/token over
            // the 96 calls, 0.78 -> 0.31 ms each).
            const float * rp = (const float *)w + (size_t)n * K;
            int k = 0;
            for (; k + 16 <= K; k += 16) {
                float wv[16], xv[16];
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    wv[j] = rp[k + j];
                }
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    xv[j] = xr[k + j];
                }
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    acc += wv[j] * xv[j];
                }
            }
            for (; k < K; k++) {
                acc += rp[k] * xr[k];
            }
        }
        out[(size_t)t * os + n] = acc;
    });
}

// ---------------------------------------------------------------------------
// F32 HC inject (K = 10240, N = hc = 4, called twice per layer = 96 a token).
//
// The old path gave each (token, stream) ONE work-item, so four lanes streamed
// 160 KB of weights between them and every load was consumed by the FMA that
// followed it: ~0.31 ms per call, 30.8 ms/token over the 96 calls, the largest
// dense phase left after the Q8_0 GEMV work.  One 256-thread workgroup per
// (token, stream) gives the dot 256 lanes: each lane owns a strided float4
// chunk (a warp reads 512 contiguous bytes per step) with four independent
// accumulators, and the sum reduces through a warp shuffle plus one SLM slot
// per sub-group.  Four float4 pairs are loaded before the first FMA of each
// step - the same latency-covering trick the old F32 branch used with two
// 16-value batches, one level up.
//
// This changes the summation order (per-lane chunks and a tree instead of one
// serial prefix), so it is not bit-identical to the old path; it was validated
// against the token stream and the 27B decode-vs-prefill oracle, not assumed.
// PF_Q4_INJECT=0 restores the old one-lane-per-row kernel for the A/B.
template <int WG>
static void inject_f32_impl(queue & q, const float * w, const float * x, int xs, float * out, int os, int n_tok,
                            int K, int N, bool v4) {
    constexpr int NSG = WG / 32;
    const int n4 = K >> 2;
    q.submit([&](handler & h) {
        local_accessor<float, 1> red(NSG, h);
        h.parallel_for(nd_range<1>((size_t)n_tok * (size_t)N * WG, WG),
                       [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int g = (int)it.get_group(0);
                           const int row = g % N;
                           const int tok = g / N;
                           const int tid = (int)it.get_local_id(0);
                           const sub_group sg = it.get_sub_group();
                           const float * wp = w + (size_t)row * (size_t)K;
                           const float * xp = x + (size_t)tok * (size_t)xs;
                           float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                           if (v4) {
                               const float4 * w4 = (const float4 *)wp;
                               const float4 * x4 = (const float4 *)xp;
                               int i = tid;
                               for (; i + 3 * WG < n4; i += 4 * WG) {
                                   const float4 w0 = w4[i], x0 = x4[i];
                                   const float4 w1 = w4[i + WG], x1 = x4[i + WG];
                                   const float4 w2 = w4[i + 2 * WG], x2 = x4[i + 2 * WG];
                                   const float4 w3 = w4[i + 3 * WG], x3 = x4[i + 3 * WG];
                                   a0 += w0.x() * x0.x();
                                   a1 += w0.y() * x0.y();
                                   a2 += w0.z() * x0.z();
                                   a3 += w0.w() * x0.w();
                                   a0 += w1.x() * x1.x();
                                   a1 += w1.y() * x1.y();
                                   a2 += w1.z() * x1.z();
                                   a3 += w1.w() * x1.w();
                                   a0 += w2.x() * x2.x();
                                   a1 += w2.y() * x2.y();
                                   a2 += w2.z() * x2.z();
                                   a3 += w2.w() * x2.w();
                                   a0 += w3.x() * x3.x();
                                   a1 += w3.y() * x3.y();
                                   a2 += w3.z() * x3.z();
                                   a3 += w3.w() * x3.w();
                               }
                               for (; i < n4; i += WG) {
                                   const float4 wv = w4[i], xv = x4[i];
                                   a0 += wv.x() * xv.x();
                                   a1 += wv.y() * xv.y();
                                   a2 += wv.z() * xv.z();
                                   a3 += wv.w() * xv.w();
                               }
                           } else {
                               // K or the row stride is not 4-float aligned: the
                               // same workgroup/reduction shape on scalars.
                               for (int k = tid; k < K; k += WG) {
                                   a0 += wp[k] * xp[k];
                               }
                           }
                           float v = (a0 + a1) + (a2 + a3);
                           v = reduce_over_group(sg, v, plus<float>());
                           const int sgl = (int)sg.get_local_linear_id();
                           const int sgi = (int)sg.get_group_linear_id();
                           if (sgl == 0) {
                               red[sgi] = v;
                           }
                           it.barrier();
                           if (tid == 0) {
                               float s = red[0];
                               for (int i = 1; i < NSG; i++) {
                                   s += red[i];
                               }
                               out[(size_t)tok * (size_t)os + row] = s;
                           }
                       });
    });
}

// The HC scatter projection has N = hc = 4, which the generic GEMV's tiling
// does not accept; the F32 case (the only user of this launcher) is the
// parallel workgroup dot above, and the block formats keep the row-dot kernel.
void q4_inject_launch(queue & q, const void * w, uint32_t type, const float * x, int xs, float * out, int os,
                      int n_tok, int K, int N) {
    const bool f32 = type != 8 && type != 30 && type != 7; // q4_up_gemv's F32 catch-all
    static const bool par = si::env::i32("PF_Q4_INJECT", 1) != 0;
    if (f32 && par && n_tok > 0 && N > 0 && K > 0) {
        const bool v4 =
            ((K & 3) == 0) && ((xs & 3) == 0) && (((uintptr_t)w & 15) == 0) && (((uintptr_t)x & 15) == 0);
        inject_f32_impl<256>(q, (const float *)w, x, xs, out, os, n_tok, K, N, v4);
        return;
    }
    q4_up_gemv_launch(q, w, type, x, xs, out, os, n_tok, K, N);
}

void q4_repeat_launch(queue & q, const float * emb, float * residual, int n_tok, int n_embd, int hc) {
    q.parallel_for(range<2>((size_t)n_tok * hc, (size_t)n_embd), [=](id<2> id) {
        const int j = (int)id[0];
        const int c = (int)id[1];
        residual[(size_t)j * n_embd + c] = emb[(size_t)(j / hc) * n_embd + c];
    });
}

} // namespace si
