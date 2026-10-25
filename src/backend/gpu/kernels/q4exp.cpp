// qwen4exp additions: hyper-connection mixing and the token-side helpers.
// These are the only new per-layer kernels: GDN, attention and the GEMVs all
// use the existing launchers.
#include "kernels.h"
#include "kernel_utils.h"
#include "device/device_profile.h"

namespace si {

using namespace sycl;
using namespace si::kd;

namespace {

// One workgroup per (token, stream).  n_embd is 2560 on the reference model.
template <int WG>
static void hc_norm_impl(queue & q, const float * x, const float * w, float * out, int n_tok, int n_embd, int hc,
                         float eps) {
    constexpr int NSG = WG / 32;
    q.submit([&](handler & h) {
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

void q4_hc_norm_launch(queue & q, const float * x, const float * w, float * out, int n_tok, int n_embd, int hc,
                       float eps) {
    const int want = si::dev::wg_clamped_for_queue(q, 256);
    if (want >= 512) {
        hc_norm_impl<512>(q, x, w, out, n_tok, n_embd, hc, eps);
    } else {
        hc_norm_impl<256>(q, x, w, out, n_tok, n_embd, hc, eps);
    }
}

void q4_hc_mix_launch(queue & q, const float * xn, const float * gate, float * mixed, int n_tok, int n_embd, int hc) {
    q.parallel_for(range<2>((size_t)n_tok, (size_t)n_embd), [=](id<2> id) {
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

void q4_hc_combine_launch(queue & q, float * residual, const float * block_out, const float * inject, int n_tok,
                          int n_embd, int hc) {
    q.parallel_for(range<2>((size_t)n_tok * hc, (size_t)n_embd), [=](id<2> id) {
        const int j = (int)id[0];
        const int c = (int)id[1];
        const int i = j / hc;
        const int s = j % hc;
        const float wgt = 2.0f * sigmoid_f(inject[(size_t)i * hc + s] / (float)hc);
        residual[(size_t)j * n_embd + c] += block_out[(size_t)i * n_embd + c] * wgt;
    });
}

void q4_silu_scale_launch(queue & q, float * x, int n, float scale) {
    q.parallel_for(range<1>((size_t)n), [=](id<1> i) { x[i[0]] = silu_f(x[i[0]] * scale); });
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
            const float * rp = (const float *)w + (size_t)n * K;
            for (int k = 0; k < K; k++) {
                acc += rp[k] * xr[k];
            }
        }
        out[(size_t)t * os + n] = acc;
    });
}

// The HC scatter projection has N = hc = 4, which the generic GEMV's tiling
// does not accept; a direct per-(token, stream) row dot is both exact and tiny.
void q4_inject_launch(queue & q, const void * w, uint32_t type, const float * x, int xs, float * out, int os,
                      int n_tok, int K, int N) {
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
