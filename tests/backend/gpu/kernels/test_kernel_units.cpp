// Model-free unit tests for GPU kernels that have no stage-test snapshot.
//
// Every case builds synthetic inputs, runs one launcher, and compares against a
// scalar reference written here.  No GGUF, no engine: a regression is named by
// kernel, not by "the model's logits changed".
//
// GPU-only, but no model: registered under the gpu;kernel label.  The queue is
// explicit (gpu_selector_v, in order) because a default queue is
// implementation-defined and on a multi-GPU host is not the device this test
// reads back from.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels.h"
#include "quant.h"

using namespace si;
using namespace sycl;

static int g_fail = 0;

static void fail(const char * name, const char * what) {
    std::printf("%-32s FAIL (%s)\n", name, what);
    g_fail++;
}

static void check_f32(const char * name, const std::vector<float> & got, const std::vector<float> & exp, double tol,
                      double floor = 1.0) {
    if (got.size() != exp.size()) {
        fail(name, "size");
        return;
    }
    double maxd = 0, maxv = 0;
    for (size_t i = 0; i < got.size(); i++) {
        maxd = std::max(maxd, (double)std::fabs(got[i] - exp[i]));
        maxv = std::max(maxv, (double)std::fabs(exp[i]));
    }
    const bool ok = maxd <= tol * std::max(floor, maxv);
    std::printf("%-32s n=%-6zu max|exp|=%.5f max|diff|=%.3e %s\n", name, got.size(), maxv, maxd, ok ? "OK" : "FAIL");
    if (!ok) g_fail++;
}
static void check_i32(const char * name, const std::vector<int32_t> & got, const std::vector<int32_t> & exp) {
    bool ok = got.size() == exp.size();
    for (size_t i = 0; ok && i < got.size(); i++) ok = got[i] == exp[i];
    std::printf("%-32s n=%-6zu %s\n", name, got.size(), ok ? "OK" : "FAIL");
    if (!ok) g_fail++;
}
static void check_i8(const char * name, const std::vector<int8_t> & got, const std::vector<int8_t> & exp) {
    bool ok = got.size() == exp.size();
    for (size_t i = 0; ok && i < got.size(); i++) ok = got[i] == exp[i];
    std::printf("%-32s n=%-6zu %s\n", name, got.size(), ok ? "OK" : "FAIL");
    if (!ok) g_fail++;
}

static float rnd(int i, float scale = 2.0f) {
    const unsigned h = (unsigned)(i * 2654435761u + 1013904223u);
    return ((float)((h >> 8) & 0xffff) / 32768.0f - 1.0f) * scale;
}
static float silu_(float v) {
    return v / (1.0f + std::exp(-v));
}
static float sigmoid_(float v) {
    return 1.0f / (1.0f + std::exp(-v));
}

template <typename T> static T * dev(queue & q, size_t n) {
    return malloc_device<T>(n, q);
}
template <typename T> static void up(queue & q, T * d, const std::vector<T> & h) {
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
}
template <typename T> static void down(queue & q, std::vector<T> & h, const T * d) {
    q.memcpy(h.data(), d, h.size() * sizeof(T)).wait();
}

// ---------------------------------------------------------------------------
static void test_copy_row(queue & q) {
    const int n = 37, tpb = 32;
    std::vector<float> src((size_t)2 * tpb * n);
    for (size_t i = 0; i < src.size(); i++) src[i] = rnd((int)i);
    float * d_src = dev<float>(q, src.size());
    float * d_dst = dev<float>(q, n);
    step_info * d_info = dev<step_info>(q, 1);
    step_info info{};
    info.n_rows = 2;
    info.n_real = tpb;
    info.tpb = tpb;
    info.n_real_row[1] = 5;
    info.active[0] = info.active[1] = 1;
    up(q, d_src, src);
    up(q, d_info, {info});

    std::vector<float> got(n), exp(n);
    copy_row_launch(q, d_src, d_dst, d_info, n, -1);
    down(q, got, d_dst);
    std::memcpy(exp.data(), src.data() + ((size_t)1 * tpb + 4) * n, n * 4);
    check_f32("copy_row_launch last", got, exp, 0.0);

    copy_row_launch(q, d_src, d_dst, d_info, n, 3);
    down(q, got, d_dst);
    std::memcpy(exp.data(), src.data() + (size_t)3 * n, n * 4);
    check_f32("copy_row_launch row=3", got, exp, 0.0);
    free(d_src, q);
    free(d_dst, q);
    free(d_info, q);
}

// ---------------------------------------------------------------------------
static void test_gated_norm(queue & q) {
    const int n_heads = 3, head_dim = 64, n_real = 2, tpb = 32, per = n_heads * head_dim;
    const float eps = 1e-5f;
    std::vector<float> a((size_t)tpb * per), z((size_t)tpb * per), w(head_dim);
    for (size_t i = 0; i < a.size(); i++) a[i] = rnd((int)i);
    for (size_t i = 0; i < z.size(); i++) z[i] = rnd((int)i + 700);
    for (int i = 0; i < head_dim; i++) w[i] = 0.5f + 0.1f * i;
    float * d_a = dev<float>(q, a.size()), *d_z = dev<float>(q, z.size()), *d_w = dev<float>(q, w.size());
    float * d_o = dev<float>(q, a.size());
    step_info * d_info = dev<step_info>(q, 1);
    step_info info{};
    info.n_rows = 1;
    info.n_real = n_real;
    info.tpb = tpb;
    info.active[0] = 1;
    up(q, d_a, a);
    up(q, d_z, z);
    up(q, d_w, w);
    up(q, d_info, {info});
    std::vector<float> got(a.size(), -9.f), exp(a.size(), -9.f);
    up(q, d_o, got); // untouched slots stay at the sentinel on both sides
    gated_norm_launch(q, d_a, d_z, d_w, d_o, d_info, n_heads, head_dim, eps, 1, n_real, 0, false);
    down(q, got, d_o);
    for (int t = 0; t < n_real; t++) {
        for (int h = 0; h < n_heads; h++) {
            const size_t base = (size_t)t * per + (size_t)h * head_dim;
            double ss = 0;
            for (int d = 0; d < head_dim; d++) ss += (double)a[base + d] * a[base + d];
            const float inv = 1.0f / std::sqrt((float)(ss / head_dim) + eps);
            for (int d = 0; d < head_dim; d++) exp[base + d] = a[base + d] * inv * w[d] * silu_(z[base + d]);
        }
    }
    check_f32("gated_norm_launch", got, exp, 1e-6, 1e-3);
    free(d_a, q); free(d_z, q); free(d_w, q); free(d_o, q); free(d_info, q);
}

// ---------------------------------------------------------------------------
static void test_q4_hc(queue & q) {
    const int n_tok = 2, hc = 3, n_embd = 17;
    const float eps = 1e-6f;
    std::vector<float> x((size_t)n_tok * hc * n_embd), w((size_t)hc * n_embd);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i);
    for (size_t i = 0; i < w.size(); i++) w[i] = 0.4f + 0.1f * (float)(i % 5);
    float * d_x = dev<float>(q, x.size()), *d_w = dev<float>(q, w.size()), *d_o = dev<float>(q, x.size());
    up(q, d_x, x);
    up(q, d_w, w);
    q4_hc_norm_launch(q, d_x, d_w, d_o, n_tok, n_embd, hc, eps).wait();
    std::vector<float> got(x.size()), exp(x.size());
    down(q, got, d_o);
    for (int t = 0; t < n_tok; t++) {
        for (int s = 0; s < hc; s++) {
            const size_t base = ((size_t)t * hc + s) * n_embd;
            double ss = 0;
            for (int c = 0; c < n_embd; c++) ss += (double)x[base + c] * x[base + c];
            const float inv = 1.0f / std::sqrt((float)(ss / n_embd) + eps);
            for (int c = 0; c < n_embd; c++) exp[base + c] = x[base + c] * inv * w[(size_t)s * n_embd + c];
        }
    }
    check_f32("q4_hc_norm_launch", got, exp, 1e-5, 1e-3);

    std::vector<float> gate(x.size());
    for (size_t i = 0; i < gate.size(); i++) gate[i] = rnd((int)i + 400);
    float * d_g = dev<float>(q, gate.size()), *d_m = dev<float>(q, (size_t)n_tok * n_embd);
    up(q, d_g, gate);
    q4_hc_mix_launch(q, d_x, d_g, d_m, n_tok, n_embd, hc).wait();
    std::vector<float> mix((size_t)n_tok * n_embd), emix(mix.size());
    down(q, mix, d_m);
    for (int t = 0; t < n_tok; t++) {
        for (int c = 0; c < n_embd; c++) {
            float acc = 0;
            for (int s = 0; s < hc; s++) {
                const size_t o = ((size_t)t * hc + s) * n_embd + c;
                acc += x[o] * sigmoid_(gate[o]);
            }
            emix[(size_t)t * n_embd + c] = acc / (float)hc;
        }
    }
    check_f32("q4_hc_mix_launch", mix, emix, 1e-6, 1e-3);

    std::vector<float> block((size_t)n_tok * n_embd), inject((size_t)n_tok * hc), res = x, eres = x;
    for (size_t i = 0; i < block.size(); i++) block[i] = rnd((int)i + 900);
    for (size_t i = 0; i < inject.size(); i++) inject[i] = rnd((int)i + 1200);
    float * d_b = dev<float>(q, block.size()), *d_i = dev<float>(q, inject.size()), *d_r = dev<float>(q, res.size());
    up(q, d_b, block);
    up(q, d_i, inject);
    up(q, d_r, res);
    q4_hc_combine_launch(q, d_r, d_b, d_i, n_tok, n_embd, hc).wait();
    down(q, res, d_r);
    for (int t = 0; t < n_tok; t++) {
        for (int s = 0; s < hc; s++) {
            const float wgt = 2.0f * sigmoid_(inject[(size_t)t * hc + s] / (float)hc);
            for (int c = 0; c < n_embd; c++) {
                eres[((size_t)t * hc + s) * n_embd + c] += block[(size_t)t * n_embd + c] * wgt;
            }
        }
    }
    check_f32("q4_hc_combine_launch", res, eres, 1e-6, 1e-3);

    std::vector<float> ss = x;
    float * d_s = dev<float>(q, ss.size());
    up(q, d_s, ss);
    q4_silu_scale_launch(q, d_s, (int)ss.size(), 0.37f).wait();
    down(q, ss, d_s);
    for (size_t i = 0; i < ss.size(); i++) exp[i] = silu_(x[i] * 0.37f);
    check_f32("q4_silu_scale_launch", ss, exp, 1e-6, 1e-3);

    // repeat
    std::vector<float> emb((size_t)n_tok * n_embd, 0.f);
    for (size_t i = 0; i < emb.size(); i++) emb[i] = rnd((int)i + 1500);
    std::vector<float> rep((size_t)n_tok * hc * n_embd, -1.f), erep(rep.size(), -1.f);
    float * d_e = dev<float>(q, emb.size()), *d_rep = dev<float>(q, rep.size());
    up(q, d_e, emb);
    up(q, d_rep, rep);
    q4_repeat_launch(q, d_e, d_rep, n_tok, n_embd, hc);
    down(q, rep, d_rep);
    for (int j = 0; j < n_tok * hc; j++) {
        for (int c = 0; c < n_embd; c++) erep[(size_t)j * n_embd + c] = emb[(size_t)(j / hc) * n_embd + c];
    }
    check_f32("q4_repeat_launch", rep, erep, 0.0);

    free(d_x, q); free(d_w, q); free(d_o, q); free(d_g, q); free(d_m, q); free(d_b, q); free(d_i, q);
    free(d_r, q); free(d_s, q); free(d_e, q); free(d_rep, q);
}

// ---------------------------------------------------------------------------
static void test_q4_gemv(queue & q) {
    const int K = 40, N = 3, n_tok = 2;
    std::vector<float> x((size_t)n_tok * K), w((size_t)N * K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i);
    for (size_t i = 0; i < w.size(); i++) w[i] = rnd((int)i + 300) * 0.1f;
    float * d_x = dev<float>(q, x.size()), *d_w = dev<float>(q, w.size()), *d_o = dev<float>(q, (size_t)n_tok * N);
    up(q, d_x, x);
    up(q, d_w, w);
    q4_up_gemv_launch(q, d_w, 0, d_x, K, d_o, N, n_tok, K, N);
    std::vector<float> got((size_t)n_tok * N), exp(got.size());
    down(q, got, d_o);
    for (int t = 0; t < n_tok; t++) {
        for (int n = 0; n < N; n++) {
            double a = 0;
            for (int k = 0; k < K; k++) a += (double)w[(size_t)n * K + k] * x[(size_t)t * K + k];
            exp[(size_t)t * N + n] = (float)a;
        }
    }
    check_f32("q4_up_gemv_launch", got, exp, 1e-5, 1e-3);

    q4_inject_launch(q, d_w, 0, d_x, K, d_o, N, n_tok, K, N);
    down(q, got, d_o);
    check_f32("q4_inject_launch", got, exp, 1e-5, 1e-3);
    free(d_x, q); free(d_w, q); free(d_o, q);
}

// ---------------------------------------------------------------------------
static void test_q4_hc_fused(queue & q) {
    const int n_tok = 1, hc = 2, n_embd = 256, iK = 128, iN = 4;
    std::vector<float> ix((size_t)n_tok * iK), iw((size_t)iN * iK);
    std::vector<float> xn((size_t)n_tok * hc * n_embd), gate(xn.size());
    for (size_t i = 0; i < ix.size(); i++) ix[i] = rnd((int)i);
    for (size_t i = 0; i < iw.size(); i++) iw[i] = rnd((int)i + 300) * 0.1f;
    for (size_t i = 0; i < xn.size(); i++) xn[i] = rnd((int)i + 600);
    for (size_t i = 0; i < gate.size(); i++) gate[i] = rnd((int)i + 900);
    float * d_ix = dev<float>(q, ix.size()), *d_iw = dev<float>(q, iw.size());
    float * d_xn = dev<float>(q, xn.size()), *d_gate = dev<float>(q, gate.size());
    float * d_io = dev<float>(q, (size_t)n_tok * iN), *d_mix = dev<float>(q, (size_t)n_tok * n_embd);
    up(q, d_ix, ix); up(q, d_iw, iw); up(q, d_xn, xn); up(q, d_gate, gate);
    q4_hc_inject_mix_launch(q, d_iw, 0, d_ix, iK, d_io, iN, iK, iN, d_xn, d_gate, d_mix, n_tok, n_embd, hc).wait();
    std::vector<float> io((size_t)n_tok * iN), mix((size_t)n_tok * n_embd);
    down(q, io, d_io);
    down(q, mix, d_mix);
    std::vector<float> eio(io.size(), 0.f), emix(mix.size(), 0.f);
    for (int t = 0; t < n_tok; t++) {
        for (int n = 0; n < iN; n++) {
            double a = 0;
            for (int k = 0; k < iK; k++) a += (double)iw[(size_t)n * iK + k] * ix[(size_t)t * iK + k];
            eio[(size_t)t * iN + n] = (float)a;
        }
        for (int c = 0; c < n_embd; c++) {
            float a = 0;
            for (int s = 0; s < hc; s++) {
                const size_t o = ((size_t)t * hc + s) * n_embd + c;
                a += xn[o] * sigmoid_(gate[o]);
            }
            emix[(size_t)t * n_embd + c] = a / (float)hc;
        }
    }
    check_f32("q4_hc_inject_mix inject", io, eio, 1e-5, 1e-3);
    check_f32("q4_hc_inject_mix mix", mix, emix, 1e-6, 1e-3);
    free(d_ix, q); free(d_iw, q); free(d_xn, q); free(d_gate, q); free(d_io, q); free(d_mix, q);
}

// ---------------------------------------------------------------------------
static void test_mtp_capture_argmax(queue & q) {
    const int n_rows = 3, n = 41;
    std::vector<float> src((size_t)n_rows * n);
    for (size_t i = 0; i < src.size(); i++) src[i] = rnd((int)i);
    float * d_src = dev<float>(q, src.size()), *d_dst = dev<float>(q, src.size());
    up(q, d_src, src);
    mtp_capture_launch(q, d_src, d_dst, n_rows, n);
    std::vector<float> got(src.size());
    down(q, got, d_dst);
    check_f32("mtp_capture_launch", got, src, 0.0);

    // argmax: an exact tie goes to the lower index; a later value wins.
    const int M = 2, nv = 257;
    std::vector<float> logits((size_t)M * nv, -5.f);
    logits[(size_t)0 * nv + 200] = 4.0f;
    logits[(size_t)1 * nv + 17] = 3.0f;
    logits[(size_t)1 * nv + 200] = 3.0f; // tie: lower index wins
    float * d_l = dev<float>(q, logits.size());
    int32_t * d_i = dev<int32_t>(q, M);
    float * d_v = dev<float>(q, M);
    up(q, d_l, logits);
    mtp_argmax_launch(q, d_l, nv, d_i, d_v, M);
    std::vector<int32_t> ids(M);
    std::vector<float> vals(M);
    down(q, ids, d_i);
    down(q, vals, d_v);
    check_i32("mtp_argmax_launch ids", ids, {200, 17});
    check_f32("mtp_argmax_launch vals", vals, {4.0f, 3.0f}, 0.0);
    free(d_src, q); free(d_dst, q); free(d_l, q); free(d_i, q); free(d_v, q);
}

// ---------------------------------------------------------------------------
static void test_w4_split(queue & q) {
    const int M = 2, K = 34; // K/2 = 17, an odd tail per row
    std::vector<int8_t> axg((size_t)M * K);
    for (size_t i = 0; i < axg.size(); i++) axg[i] = (int8_t)((int)(i * 7) % 200 - 100);
    int8_t * d_g = dev<int8_t>(q, axg.size()), *d_e = dev<int8_t>(q, (size_t)M * (K / 2)),
           *d_o = dev<int8_t>(q, (size_t)M * (K / 2));
    up(q, d_g, axg);
    w4_split_act_launch(q, d_g, d_e, d_o, M, K);
    std::vector<int8_t> e((size_t)M * (K / 2)), o((size_t)M * (K / 2));
    down(q, e, d_e);
    down(q, o, d_o);
    std::vector<int8_t> ee(e.size()), eo(o.size());
    for (int m = 0; m < M; m++) {
        for (int j = 0; j < K / 2; j++) {
            ee[(size_t)m * (K / 2) + j] = axg[(size_t)m * K + 2 * j];
            eo[(size_t)m * (K / 2) + j] = axg[(size_t)m * K + 2 * j + 1];
        }
    }
    check_i8("w4_split_act even", e, ee);
    check_i8("w4_split_act odd", o, eo);
    free(d_g, q); free(d_e, q); free(d_o, q);
}

// ---------------------------------------------------------------------------
static void test_vit_elementwise(queue & q) {
    const int rows = 3, cols = 11;
    std::vector<float> g((size_t)rows * cols), u((size_t)rows * cols), y((size_t)rows * cols), xv((size_t)rows * cols),
        bias(cols);
    for (size_t i = 0; i < g.size(); i++) g[i] = rnd((int)i);
    for (size_t i = 0; i < u.size(); i++) u[i] = rnd((int)i + 300);
    for (size_t i = 0; i < y.size(); i++) y[i] = rnd((int)i + 600);
    for (size_t i = 0; i < xv.size(); i++) xv[i] = rnd((int)i + 900);
    for (int i = 0; i < cols; i++) bias[i] = rnd(i + 1200) * 0.1f;

    // gelu / quick_gelu / erf gelu: compare all three against their formulas.
    for (int which = 0; which < 3; which++) {
        std::vector<float> v((size_t)rows * cols);
        for (size_t i = 0; i < v.size(); i++) v[i] = rnd((int)i + which * 1000);
        std::vector<float> ev = v;
        float * d_v = dev<float>(q, v.size());
        up(q, d_v, v);
        if (which == 0) {
            vit_gelu_launch(q, d_v, (int)v.size());
            for (auto & a : ev) {
                const float x3 = a * a * a;
                a = 0.5f * a * (1.0f + std::tanh(0.7978845608028654f * (a + 0.044715f * x3)));
            }
        } else if (which == 1) {
            vit_quick_gelu_launch(q, d_v, (int)v.size());
            for (auto & a : ev) a = a / (1.0f + std::exp(-1.702f * a));
        } else {
            vit_gelu_errf_launch(q, d_v, (int)v.size());
            for (auto & a : ev) a = 0.5f * a * (1.0f + std::erf(a / std::sqrt(2.0f)));
        }
        std::vector<float> gv(v.size());
        down(q, gv, d_v);
        check_f32(which == 0 ? "vit_gelu_launch" : (which == 1 ? "vit_quick_gelu_launch" : "vit_gelu_errf_launch"),
                  gv, ev, 1e-6, 1e-3);
        free(d_v, q);
    }

    // silu_mul
    float * d_g = dev<float>(q, g.size()), *d_u = dev<float>(q, u.size()), *d_o = dev<float>(q, g.size());
    up(q, d_g, g); up(q, d_u, u);
    vit_silu_mul_launch(q, d_g, d_u, cols, d_o, rows, cols);
    std::vector<float> got(g.size());
    down(q, got, d_o);
    std::vector<float> exp(g.size());
    for (size_t i = 0; i < g.size(); i++) exp[i] = silu_(g[i]) * u[i];
    check_f32("vit_silu_mul_launch", got, exp, 1e-6, 1e-3);

    // add_bias / add / copy
    float * d_y = dev<float>(q, y.size()), *d_b = dev<float>(q, bias.size()), *d_x = dev<float>(q, xv.size());
    up(q, d_y, y); up(q, d_b, bias); up(q, d_x, xv);
    vit_add_bias_launch(q, d_y, cols, d_b, rows, cols);
    down(q, got, d_y);
    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) exp[(size_t)r * cols + c] = y[(size_t)r * cols + c] + bias[c];
    check_f32("vit_add_bias_launch", got, exp, 0.0);
    vit_add_launch(q, d_y, cols, d_x, cols, rows, cols);
    down(q, got, d_y);
    for (size_t i = 0; i < y.size(); i++) exp[i] += xv[i];
    check_f32("vit_add_launch", got, exp, 0.0);
    vit_copy_launch(q, d_x, cols, d_o, cols, rows, cols);
    down(q, got, d_o);
    check_f32("vit_copy_launch", got, xv, 0.0);

    // layernorm
    float * d_lnw = dev<float>(q, cols), *d_lnb = dev<float>(q, cols);
    up(q, d_lnw, bias);
    std::vector<float> b2(cols);
    for (int i = 0; i < cols; i++) b2[i] = rnd(i + 1500) * 0.2f;
    float * d_lnb2 = dev<float>(q, cols);
    up(q, d_lnb2, b2);
    vit_layernorm_launch(q, d_x, cols, d_lnw, d_lnb2, d_o, cols, rows, cols, 1e-5f);
    down(q, got, d_o);
    for (int r = 0; r < rows; r++) {
        double mean = 0, var = 0;
        for (int c = 0; c < cols; c++) mean += xv[(size_t)r * cols + c];
        mean /= cols;
        for (int c = 0; c < cols; c++) {
            const double d0 = xv[(size_t)r * cols + c] - mean;
            var += d0 * d0;
        }
        var /= cols;
        const float inv = 1.0f / std::sqrt((float)var + 1e-5f);
        for (int c = 0; c < cols; c++) {
            exp[(size_t)r * cols + c] = (float)((xv[(size_t)r * cols + c] - mean) * inv) * bias[c] + b2[c];
        }
    }
    check_f32("vit_layernorm_launch", got, exp, 1e-5, 1e-3);

    free(d_g, q); free(d_u, q); free(d_o, q); free(d_y, q); free(d_b, q); free(d_x, q);
    free(d_lnw, q); free(d_lnb, q); free(d_lnb2, q);
}

// ---------------------------------------------------------------------------
static void test_at(queue & q) {
    // conv1d: x_frames 6, x_in 3, taps 3, stride 1, pad 1 -> y_frames 6
    const int xf = 6, xi = 3, taps = 3, stride = 1, pad = 1, yf = 6, wo = 4;
    std::vector<float> x((size_t)xf * xi), w((size_t)wo * taps * xi), b(wo);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i);
    for (size_t i = 0; i < w.size(); i++) w[i] = rnd((int)i + 300) * 0.1f;
    for (int i = 0; i < wo; i++) b[i] = rnd(i + 700) * 0.1f;
    float * d_x = dev<float>(q, x.size()), *d_w = dev<float>(q, w.size()), *d_b = dev<float>(q, b.size());
    float * d_o = dev<float>(q, (size_t)yf * wo);
    up(q, d_x, x); up(q, d_w, w); up(q, d_b, b);
    at_conv1d_launch(q, d_x, xf, xi, taps, stride, pad, d_w, d_b, d_o, yf, wo);
    std::vector<float> got((size_t)yf * wo), exp(got.size());
    down(q, got, d_o);
    for (int t = 0; t < yf; t++) {
        for (int o = 0; o < wo; o++) {
            double a = b[o];
            for (int tp = 0; tp < taps; tp++) {
                const int fr = t * stride + tp - pad;
                if (fr < 0 || fr >= xf) continue;
                for (int i = 0; i < xi; i++) a += (double)w[(size_t)o * (taps * xi) + (size_t)tp * xi + i] * x[(size_t)fr * xi + i];
            }
            exp[(size_t)t * wo + o] = (float)a;
        }
    }
    check_f32("at_conv1d_launch", got, exp, 1e-6, 1e-3);

    // rope1d on a fused qkv buffer
    const int n_tok = 3, n_head = 2, hd = 4;
    const int stride_qkv = 3 * n_head * hd;
    std::vector<float> qkv((size_t)n_tok * stride_qkv);
    for (size_t i = 0; i < qkv.size(); i++) qkv[i] = rnd((int)i + 900);
    std::vector<float> eq = qkv;
    float * d_qkv = dev<float>(q, qkv.size());
    up(q, d_qkv, qkv);
    at_rope1d_launch(q, d_qkv, stride_qkv, n_tok, n_head, hd, 10000.f);
    down(q, qkv, d_qkv);
    for (int t = 0; t < n_tok; t++) {
        for (int pass = 0; pass < 2; pass++) { // q and k
            for (int h = 0; h < n_head; h++) {
                float * v = &eq[(size_t)t * stride_qkv + (size_t)pass * n_head * hd + (size_t)h * hd];
                const int half = hd / 2;
                for (int i = 0; i < half; i++) {
                    const float ang = (float)t * std::exp2(-2.0f * i / hd * std::log2(10000.f));
                    const float c = std::cos(ang), s = std::sin(ang);
                    const float x0 = v[i], x1 = v[i + half];
                    v[i] = x0 * c - x1 * s;
                    v[i + half] = x0 * s + x1 * c;
                }
            }
        }
    }
    check_f32("at_rope1d_launch", qkv, eq, 1e-5, 1e-3);

    free(d_x, q); free(d_w, q); free(d_b, q); free(d_o, q); free(d_qkv, q);
}

// ---------------------------------------------------------------------------
static void test_df_simple(queue & q) {
    const int rows = 3, n = 17;
    std::vector<float> src((size_t)rows * n), a((size_t)rows * n), b((size_t)rows * n);
    for (size_t i = 0; i < src.size(); i++) src[i] = rnd((int)i);
    for (size_t i = 0; i < a.size(); i++) a[i] = rnd((int)i + 300);
    for (size_t i = 0; i < b.size(); i++) b[i] = rnd((int)i + 600);
    float * d_src = dev<float>(q, src.size()), *d_dst = dev<float>(q, src.size());
    float * d_a = dev<float>(q, a.size()), *d_b = dev<float>(q, b.size()), *d_o = dev<float>(q, a.size());
    up(q, d_src, src); up(q, d_a, a); up(q, d_b, b);
    df_capture_launch(q, d_src, d_dst, rows, n, n);
    std::vector<float> got(src.size());
    down(q, got, d_dst);
    check_f32("df_capture_launch", got, src, 0.0);
    df_add_launch(q, d_a, d_b, d_o, rows, n);
    down(q, got, d_o);
    std::vector<float> exp(a.size());
    for (size_t i = 0; i < a.size(); i++) exp[i] = a[i] + b[i];
    check_f32("df_add_launch", got, exp, 0.0);
    free(d_src, q); free(d_dst, q); free(d_a, q); free(d_b, q); free(d_o, q);
}

// ---------------------------------------------------------------------------
static void test_moe_small(queue & q) {
    const int n_expert = 5, n_pairs = 7;
    std::vector<int32_t> ids{0, 3, 1, 0, 4, 3, 2};
    int32_t * d_ids = dev<int32_t>(q, n_pairs), *d_off = dev<int32_t>(q, n_expert + 1),
            *d_cur = dev<int32_t>(q, n_expert), *d_act = dev<int32_t>(q, n_expert + 1),
            *d_nact = dev<int32_t>(q, 1), *d_pair = dev<int32_t>(q, n_pairs), *d_pos = dev<int32_t>(q, n_pairs);
    up(q, d_ids, ids);
    moe_csr_clear_launch(q, d_off, d_cur, d_act, d_nact, n_expert);
    moe_csr_count_launch(q, d_ids, n_pairs, d_off);
    moe_csr_prefix_launch(q, d_off, d_act, d_nact, n_expert);
    moe_csr_scatter_launch(q, d_ids, d_off, d_cur, d_pair, n_pairs);
    moe_csr_pos_launch(q, d_pair, d_pos, n_pairs);

    std::vector<int32_t> off(n_expert + 1), act(n_expert + 1), nact(1), pair(n_pairs), pos(n_pairs);
    down(q, off, d_off);
    down(q, act, d_act);
    down(q, nact, d_nact);
    down(q, pair, d_pair);
    down(q, pos, d_pos);
    // exclusive prefix over counts [2,1,1,2,1] -> off [0,2,3,4,6,7]
    check_i32("moe_csr_prefix", off, {0, 2, 3, 4, 6, 7});
    check_i32("moe_csr_active", std::vector<int32_t>(act.begin(), act.begin() + 5), {0, 1, 2, 3, 4});
    const bool na = nact[0] == 5;
    std::printf("%-32s %s\n", "moe_csr_n_active", na ? "OK" : "FAIL");
    if (!na) g_fail++;
    // pair list is CSR-ordered: expert e's pairs are its original positions
    std::vector<int32_t> epair{0, 3, 2, 6, 1, 5, 4};
    check_i32("moe_csr_scatter", pair, epair);
    // pos is the inverse
    std::vector<int32_t> epos(n_pairs);
    for (int i = 0; i < n_pairs; i++) epos[pair[i]] = i;
    check_i32("moe_csr_pos", pos, epos);

    // reduce_csr: out[t] += sum_j wts[t*top_k+j] * acc[pos[t*top_k+j]]
    const int top_k = 2, n_tok = 3, n_embd = 4;
    std::vector<float> acc((size_t)n_pairs * n_embd), wts(n_pairs), out((size_t)n_tok * n_embd, 1.f);
    for (size_t i = 0; i < acc.size(); i++) acc[i] = rnd((int)i + 200);
    for (size_t i = 0; i < wts.size(); i++) wts[i] = 0.1f * (float)(i + 1);
    float * d_acc = dev<float>(q, acc.size()), *d_wts = dev<float>(q, wts.size()), *d_out = dev<float>(q, out.size());
    up(q, d_acc, acc); up(q, d_wts, wts); up(q, d_out, out); up(q, d_pos, pos);
    moe_reduce_csr_launch(q, d_acc, d_wts, d_pos, top_k, n_tok, n_embd, n_embd, d_out, n_embd);
    std::vector<float> got(out.size());
    down(q, got, d_out);
    std::vector<float> exp(out.size());
    for (int t = 0; t < n_tok; t++) {
        for (int e = 0; e < n_embd; e++) {
            double a = 1.0;
            for (int j = 0; j < top_k; j++) {
                const int pr = t * top_k + j;
                a += (double)wts[pr] * acc[(size_t)pos[pr] * n_embd + e];
            }
            exp[(size_t)t * n_embd + e] = (float)a;
        }
    }
    check_f32("moe_reduce_csr_launch", got, exp, 1e-6, 1e-3);

    // scatter + silu helpers
    std::vector<float> s((size_t)n_pairs * n_embd);
    for (size_t i = 0; i < s.size(); i++) s[i] = rnd((int)i + 500);
    std::vector<float> dst((size_t)n_pairs * n_embd, -1.f), edst(dst.size(), -1.f);
    float * d_s = dev<float>(q, s.size()), *d_dst = dev<float>(q, dst.size());
    up(q, d_s, s); up(q, d_dst, dst);
    moe_scatter_launch(q, d_s, d_pair, 0, n_pairs, n_embd, d_dst, n_embd);
    down(q, dst, d_dst);
    for (int p = 0; p < n_pairs; p++) {
        for (int c = 0; c < n_embd; c++) edst[(size_t)pair[p] * n_embd + c] = s[(size_t)p * n_embd + c];
    }
    check_f32("moe_scatter_launch", dst, edst, 0.0);

    std::vector<float> g = s, u = s, eh = s;
    for (size_t i = 0; i < u.size(); i++) u[i] = rnd((int)i + 800);
    float * d_ug = dev<float>(q, g.size()), *d_uu = dev<float>(q, u.size()), *d_h = dev<float>(q, g.size());
    up(q, d_ug, g); up(q, d_uu, u);
    moe_u4_silu_mul_launch(q, d_ug, d_uu, (int)g.size());
    down(q, g, d_ug);
    for (size_t i = 0; i < g.size(); i++) eh[i] = silu_(s[i]) * u[i];
    check_f32("moe_u4_silu_mul_launch", g, eh, 1e-6, 1e-3);

    // gate add: out += sigmoid(gate.x) * shared
    const int nt = 2, ne = 5;
    std::vector<float> xx((size_t)nt * ne), gg(ne), sh((size_t)nt * ne), oo((size_t)nt * ne, 0.5f);
    for (size_t i = 0; i < xx.size(); i++) xx[i] = rnd((int)i + 1000);
    for (size_t i = 0; i < gg.size(); i++) gg[i] = rnd((int)i + 1300) * 0.1f;
    for (size_t i = 0; i < sh.size(); i++) sh[i] = rnd((int)i + 1600);
    float * d_xx = dev<float>(q, xx.size()), *d_gg = dev<float>(q, gg.size()), *d_sh = dev<float>(q, sh.size()),
           *d_oo = dev<float>(q, oo.size());
    up(q, d_xx, xx); up(q, d_gg, gg); up(q, d_sh, sh); up(q, d_oo, oo);
    moe_gate_add_launch(q, d_gg, 0 /* F32 */, d_xx, ne, nt, ne, d_sh, ne, d_oo, ne);
    std::vector<float> og(oo.size());
    down(q, og, d_oo);
    for (int t = 0; t < nt; t++) {
        double a = 0;
        for (int k = 0; k < ne; k++) a += (double)gg[k] * xx[(size_t)t * ne + k];
        const float f = sigmoid_((float)a);
        for (int e = 0; e < ne; e++) oo[(size_t)t * ne + e] += f * sh[(size_t)t * ne + e];
    }
    check_f32("moe_gate_add_launch", og, oo, 1e-6, 1e-3);

    free(d_ids, q); free(d_off, q); free(d_cur, q); free(d_act, q); free(d_nact, q); free(d_pair, q); free(d_pos, q);
    free(d_acc, q); free(d_wts, q); free(d_out, q); free(d_s, q); free(d_dst, q);
    free(d_ug, q); free(d_uu, q); free(d_h, q); free(d_xx, q); free(d_gg, q); free(d_sh, q); free(d_oo, q);
}

int main() {
    try {
        queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
        std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        test_copy_row(q);
        test_gated_norm(q);
        test_q4_hc(q);
        test_q4_gemv(q);
        test_q4_hc_fused(q);
        test_mtp_capture_argmax(q);
        test_w4_split(q);
        test_vit_elementwise(q);
        test_at(q);
        test_df_simple(q);
        test_moe_small(q);
        const int t_gate = moe_grp_tile_rows(q, false, false);
        const int t_down = moe_grp_tile_rows(q, true, false);
        const int t_dec = moe_grp_tile_rows(q, false, true);
        const bool tok = t_gate >= 1 && t_down >= 1 && t_dec >= 1;
        std::printf("%-32s %d/%d/%d %s\n", "moe_grp_tile_rows", t_gate, t_down, t_dec, tok ? "OK" : "FAIL");
        if (!tok) g_fail++;
    } catch (const sycl::exception & e) {
        std::printf("SYCL exception: %s\n", e.what());
        return 1;
    }
    if (g_fail) {
        std::printf("test_kernel_units: FAILURES: %d\n", g_fail);
        return 1;
    }
    std::printf("test_kernel_units: all units OK\n");
    return 0;
}
