// Device-vs-host validation for the DeepSeek-V4.1-Flash misc device ops in
// src/backend/gpu/kernels/ds41_ops.cpp.  The host side reproduces the exact
// oracle code from src/engine/arch/deepseek41/forward_host.cpp:
//
//   fp8_round_row (window KV / attention output round),
//   Fwd::moe's swiglu and sqrt-softplus router,
//   compute_candidates_one, index_select_one's top-k,
//   attention_one's gather / window ring write, apply_engram_rows.
//
// Build (JIT, standalone):
//   icpx -fsycl -std=c++17 -O2 \
//     tests/backend/gpu/kernels/test_ds41_ops.cpp \
//     src/backend/gpu/kernels/ds41_ops.cpp -o dev/test_ds41_ops \
//     -Isrc -Isrc/common -Isrc/backend -Isrc/backend/cpu \
//     -Isrc/backend/gpu/kernels -Isrc/model -Isrc/mm -Isrc/engine \
//     -Isrc/server -Ithird_party \
//     -I/opt/intel/oneapi/dnnl/2026.0/include \
//     -Lbuild-ds41-jit -lsycl_infer_core -ldnnl
//
// Serialise GPU runs with: flock /tmp/ds41_gpu.lock -c '<command>'
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>

#include "ds41_ops.h"

using namespace si;

static int g_checks = 0;
static int g_failures = 0;

static void ok(bool cond, const char * what) {
    g_checks++;
    if (!cond) {
        g_failures++;
        printf("  FAIL %s\n", what);
    }
}

// ---------------------------------------------------------------- device utils
template <class T>
static T * dev_alloc(sycl::queue & q, size_t n) {
    T * p = sycl::malloc_device<T>(n, q);
    if (!p) {
        fprintf(stderr, "malloc_device failed (%zu elements)\n", n);
        std::exit(2);
    }
    return p;
}

template <class T>
static T * dev_upload(sycl::queue & q, const std::vector<T> & h) {
    T * d = dev_alloc<T>(q, h.size());
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
    return d;
}

template <class T>
static std::vector<T> dev_download(sycl::queue & q, const T * d, size_t n) {
    std::vector<T> h(n);
    q.memcpy(h.data(), d, n * sizeof(T)).wait();
    return h;
}

struct Metrics {
    double max_abs = 0.0;
    double rel_peak = 0.0;
    double peak = 0.0;
};

static Metrics compare(const std::vector<float> & ref, const std::vector<float> & got) {
    Metrics m;
    for (size_t i = 0; i < ref.size(); i++) {
        const double a = std::fabs((double)ref[i] - (double)got[i]);
        m.max_abs = std::max(m.max_abs, a);
        m.peak = std::max(m.peak, std::fabs((double)ref[i]));
    }
    m.rel_peak = (m.peak > 0.0) ? m.max_abs / m.peak : m.max_abs;
    return m;
}

static void report(const char * tag, const Metrics & m) {
    printf("  %-22s max_abs=%.3e rel_peak=%.3e peak=%.3e\n", tag, m.max_abs, m.rel_peak, m.peak);
}

template <class F>
static double time_launch(sycl::queue & q, F && f, int reps) {
    for (int i = 0; i < 3; i++) {
        f();
    }
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) {
        f();
    }
    q.wait();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / (double)reps;
}

static std::mt19937 g_rng(20261106);

static std::vector<float> rnd(size_t n, float sigma = 1.f) {
    std::normal_distribution<float> d(0.f, sigma);
    std::vector<float> v(n);
    for (auto & x : v) {
        x = d(g_rng);
    }
    return v;
}

static uint16_t f32_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

// ---------------------------------------------------------------- host oracles
static float e4m3_host(float v) {
    const bool neg = std::signbit(v);
    float a = std::fabs(v);
    if (!(a > 0.f)) {
        return neg ? -0.0f : 0.0f;
    }
    if (a >= 448.f) {
        return neg ? -448.f : 448.f;
    }
    int e;
    std::frexp(a, &e);
    const int E = e - 1;
    const float step = (E < -6) ? std::ldexp(1.f, -9) : std::ldexp(1.f, E - 3);
    const float q = a / step;
    const float r = std::nearbyint(q);
    float out = r * step;
    if (out > 448.f) {
        out = 448.f;
    }
    return neg ? -out : out;
}

// fp8_round_row (arch/deepseek41/forward_host.cpp): one power-of-two E4M3 scale per `block`.
static void fp8_round_ref(float * v, int n, int block) {
    for (int b = 0; b < n; b += block) {
        const int m = std::min(block, n - b);
        float amax = 1e-4f;
        for (int i = 0; i < m; i++) {
            amax = std::max(amax, std::fabs(v[b + i]));
        }
        const float s = std::exp2(std::ceil(std::log2(amax / 448.f)));
        for (int i = 0; i < m; i++) {
            float x = v[b + i] / s;
            x = std::max(-448.f, std::min(448.f, x));
            v[b + i] = e4m3_host(x) * s;
        }
    }
}

static float silu_host(float x) { return x / (1.f + std::exp(-x)); }

// The host AVX2 dot's structure: two 8-lane accumulators (residues 0..7 and
// 8..15) and its horizontal tree.  A scalar std::fma replica is used so the
// device's 16-residue kernel can be compared bit for bit.
static float dot_ref16(const float * a, const float * b, int n) {
    float acc[16];
    for (int m = 0; m < 16; m++) {
        acc[m] = 0.f;
    }
    const int nvec = n & ~15;
    for (int i = 0; i < nvec; i += 16) {
        for (int m = 0; m < 16; m++) {
            acc[m] = std::fma(a[i + m], b[i + m], acc[m]);
        }
    }
    const float s0 = acc[0] + acc[8];
    const float s1 = acc[1] + acc[9];
    const float s2 = acc[2] + acc[10];
    const float s3 = acc[3] + acc[11];
    const float s4 = acc[4] + acc[12];
    const float s5 = acc[5] + acc[13];
    const float s6 = acc[6] + acc[14];
    const float s7 = acc[7] + acc[15];
    float r = ((s0 + s4) + (s1 + s5)) + ((s2 + s6) + (s3 + s7));
    for (int i = nvec; i < n; i++) {
        r = std::fma(a[i], b[i], r);
    }
    return r;
}

static float bf16_to_f32_host(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static void router_ref(const void * gate, uint32_t type, const float * x, int n_tok, int K,
                       const float * bias, const float * bias_vl, int use_vl, int E, int top_k,
                       float scale, std::vector<float> & logits, std::vector<int32_t> & ids,
                       std::vector<float> & wts) {
    logits.assign((size_t)n_tok * E, 0.f);
    ids.assign((size_t)n_tok * top_k, 0);
    wts.assign((size_t)n_tok * top_k, 0.f);
    const float * bsel = (use_vl && bias_vl) ? bias_vl : bias;
    for (int t = 0; t < n_tok; t++) {
        const float * xr = x + (size_t)t * K;
        std::vector<float> sc((size_t)E);
        for (int e = 0; e < E; e++) {
            float l;
            if (type == 30) {
                const uint16_t * W = (const uint16_t *)gate;
                std::vector<float> w((size_t)K);
                for (int i = 0; i < K; i++) {
                    w[i] = bf16_to_f32_host(W[(size_t)e * K + i]);
                }
                l = dot_ref16(xr, w.data(), K);
            } else if (type == 1) {
                const sycl::half * W = (const sycl::half *)gate;
                std::vector<float> w((size_t)K);
                for (int i = 0; i < K; i++) {
                    w[i] = (float)W[(size_t)e * K + i];
                }
                l = dot_ref16(xr, w.data(), K);
            } else {
                const float * W = (const float *)gate;
                l = dot_ref16(xr, W + (size_t)e * K, K);
            }
            logits[(size_t)t * E + e] = l;
            const float sp = (l > 20.f) ? l : std::log1p(std::exp(l));
            sc[e] = std::sqrt(sp);
        }
        std::vector<uint8_t> used((size_t)E, 0);
        const int kk = std::min(top_k, E);
        float sum = 1e-20f;
        for (int j = 0; j < kk; j++) {
            int best = -1;
            float bestv = 0.f;
            for (int e = 0; e < E; e++) {
                if (used[e]) {
                    continue;
                }
                const float b = bsel ? bsel[e] : 0.f;
                const float v = sc[e] + b;
                if (best < 0 || v > bestv) {
                    best = e;
                    bestv = v;
                }
            }
            used[best] = 1;
            ids[(size_t)t * top_k + j] = best;
            sum += sc[best];
        }
        for (int j = 0; j < kk; j++) {
            const int e = ids[(size_t)t * top_k + j];
            wts[(size_t)t * top_k + j] = sc[e] / sum * scale;
        }
    }
}

static std::vector<int32_t> select_ref(const std::vector<float> & sc, int n, int k) {
    std::vector<int32_t> ord((size_t)n);
    for (int i = 0; i < n; i++) {
        ord[i] = i;
    }
    std::partial_sort(ord.begin(), ord.begin() + k, ord.end(), [&](int a, int b) {
        const float sa = sc[a], sb = sc[b];
        if (sa != sb) {
            return sa > sb;
        }
        return a < b;
    });
    ord.resize((size_t)k);
    std::sort(ord.begin(), ord.end());
    return ord;
}

// compute_candidates_one.
static void cand_ref(const std::vector<float> & score, int n_vis, int block, int topk_blocks,
                     std::vector<uint8_t> & keep, int & nb) {
    nb = (n_vis + block - 1) / block;
    keep.assign((size_t)std::max(nb, 0), 0);
    if (nb <= 0) {
        return;
    }
    std::vector<float> bmax((size_t)nb, -INFINITY);
    for (int b = 0; b < nb; b++) {
        const int hi = std::min(n_vis, (b + 1) * block);
        for (int c = b * block; c < hi; c++) {
            bmax[b] = std::max(bmax[b], score[c]);
        }
    }
    const int last = (n_vis - 1) / block;
    if (last >= 0 && last < nb) {
        bmax[last] = INFINITY;
    }
    std::vector<int> ord((size_t)nb);
    for (int b = 0; b < nb; b++) {
        ord[b] = b;
    }
    const int take = std::min(topk_blocks, nb);
    std::partial_sort(ord.begin(), ord.begin() + take, ord.end(), [&](int a, int b) {
        if (bmax[a] != bmax[b]) {
            return bmax[a] > bmax[b];
        }
        return a < b;
    });
    for (int i = 0; i < take; i++) {
        const int b = ord[i];
        if (bmax[b] > -INFINITY) {
            keep[b] = 1;
        }
    }
}

// One token's Engram apply (apply_engram_rows).
static void engram_ref(std::vector<float> & h, const std::vector<float> & kv, const std::vector<float> & gate,
                       int hc, int dim) {
    const float * value = kv.data() + (size_t)hc * dim;
    for (int k = 0; k < hc; k++) {
        const float * hk = h.data() + (size_t)k * dim;
        const float * kk = kv.data() + (size_t)k * dim;
        const float * wg = gate.data() + (size_t)k * dim;
        double ssh = 0, ssk = 0;
        for (int i = 0; i < dim; i++) {
            ssh += (double)hk[i] * (double)hk[i];
            ssk += (double)kk[i] * (double)kk[i];
        }
        const float rstd = (1.f / std::sqrt((float)(ssh / dim) + 1e-20f)) *
                           (1.f / std::sqrt((float)(ssk / dim) + 1e-20f));
        float sd = 0;
        for (int i = 0; i < dim; i++) {
            sd = std::fma(hk[i] * wg[i], kk[i], sd);
        }
        const float dot = sd * rstd / std::sqrt((float)dim);
        const float g = 1.f / (1.f + std::exp(-std::copysign(std::sqrt(std::max(std::fabs(dot), 1e-6f)), dot)));
        for (int i = 0; i < dim; i++) {
            h[(size_t)k * dim + i] += g * value[i];
        }
    }
}

// ---------------------------------------------------------------- tests
static void test_fp8(sycl::queue & q) {
    printf("[fp8_round] n=512/1000/31, blocks 32 (window KV + attention out)\n");
    struct Case {
        int n;
        int block;
        std::vector<float> v;
    };
    std::vector<Case> cases;
    cases.push_back({512, 32, rnd(512, 1.f)});
    {
        std::vector<float> e = rnd(512, 1.f);
        for (int i = 0; i < 512; i += 3) {
            e[i] = 0.f;
        }
        e[0] = 448.f;    // exactly the E4M3 ceiling
        e[33] = -448.f;
        e[64] = 448.f * 2.f;   // forces the next power-of-two scale
        e[96] = 1e-5f;
        cases.push_back({512, 32, e});
    }
    cases.push_back({1000, 32, rnd(1000, 0.3f)}); // partial final block
    cases.push_back({31, 32, rnd(31, 2.f)});

    for (auto & c : cases) {
        std::vector<float> ref = c.v;
        fp8_round_ref(ref.data(), c.n, c.block);
        float * d = dev_upload(q, c.v);
        ds41_fp8_round_launch(q, d, c.n, c.block);
        std::vector<float> got = dev_download(q, d, c.v.size());
        sycl::free(d, q);
        ok(std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0,
           "fp8 bit-exact vs fp8_round_row");
        int bad = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            if (std::memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
                bad++;
            }
        }
        printf("  n=%4d block=%2d -> %s (%d differing floats)\n", c.n, c.block,
               bad ? "MISMATCH" : "BIT-EXACT", bad);
    }
    // timing, the window-KV / attention-output depth
    std::vector<float> t = rnd(512, 1.f);
    float * d = dev_upload(q, t);
    const double ms = time_launch(q, [&] { ds41_fp8_round_launch(q, d, 512, 32); }, 200);
    printf("  time n=512: %.4f ms\n", ms);
    sycl::free(d, q);
}

static void test_swiglu(sycl::queue & q) {
    printf("[swiglu_clamp] n=2304, lim sweep\n");
    const int n = 2304;
    for (float lim : {7.f, 1.5f, 0.25f}) {
        std::vector<float> g = rnd(n, 2.f), u = rnd(n, 2.f);
        g[0] = lim + 5.f;
        g[1] = lim;
        u[2] = lim + 5.f;
        u[3] = -lim - 5.f;
        std::vector<float> ref((size_t)n);
        for (int i = 0; i < n; i++) {
            const float gg = std::min(g[i], lim);
            const float uu = std::max(-lim, std::min(lim, u[i]));
            ref[i] = silu_host(gg) * uu;
        }
        float * dg = dev_upload(q, g);
        float * du = dev_upload(q, u);
        float * dout = dev_alloc<float>(q, n);
        ds41_swiglu_clamp_launch(q, dg, du, dout, n, lim);
        Metrics m = compare(ref, dev_download(q, dout, n));
        sycl::free(dg, q);
        sycl::free(du, q);
        sycl::free(dout, q);
        char tag[64];
        snprintf(tag, sizeof(tag), "lim=%g", lim);
        report(tag, m);
        ok(m.max_abs < 2e-5 || m.rel_peak < 2e-6, "swiglu within tolerance");
    }
    std::vector<float> g = rnd(n, 1.f), u = rnd(n, 1.f);
    float * dg = dev_upload(q, g);
    float * du = dev_upload(q, u);
    float * dout = dev_alloc<float>(q, n);
    const double ms = time_launch(q, [&] { ds41_swiglu_clamp_launch(q, dg, du, dout, n, 7.f); }, 200);
    printf("  time n=2304: %.4f ms\n", ms);
    sycl::free(dg, q);
    sycl::free(du, q);
    sycl::free(dout, q);
}

static void test_router(sycl::queue & q) {
    printf("[router] n_expert=384 n_embd=5120 top_k=8, BF16 and F32 gate rows\n");
    const int K = 5120, E = 384, topk = 8, T = 2;
    const float scale = 2.5f;
    // BF16 weights: the model's gate_inp type.
    std::vector<float> wf = rnd((size_t)E * K, 0.02f);
    std::vector<uint16_t> wb((size_t)E * K);
    for (size_t i = 0; i < wb.size(); i++) {
        wb[i] = f32_to_bf16(wf[i]);
    }
    std::vector<float> x = rnd((size_t)T * K, 0.05f);
    std::vector<float> bias = rnd(E, 0.1f);
    std::vector<float> bias_vl = rnd(E, 0.1f);

    for (int use_vl = 0; use_vl <= 1; use_vl++) {
        std::vector<float> ref_logits, ref_wts;
        std::vector<int32_t> ref_ids;
        router_ref(wb.data(), 30, x.data(), T, K, bias.data(), bias_vl.data(), use_vl, E, topk, scale,
                   ref_logits, ref_ids, ref_wts);
        uint16_t * dg = dev_upload(q, wb);
        float * dx = dev_upload(q, x);
        float * db = dev_upload(q, bias);
        float * dbv = dev_upload(q, bias_vl);
        float * dl = dev_alloc<float>(q, (size_t)T * E);
        int32_t * di = dev_alloc<int32_t>(q, (size_t)T * topk);
        float * dw = dev_alloc<float>(q, (size_t)T * topk);
        ds41_router_launch(q, dg, 30, dx, T, K, db, dbv, use_vl, E, topk, scale, dl, di, dw);
        const auto got_l = dev_download(q, dl, (size_t)T * E);
        const auto got_i = dev_download(q, di, (size_t)T * topk);
        const auto got_w = dev_download(q, dw, (size_t)T * topk);
        const Metrics ml = compare(ref_logits, got_l);
        const Metrics mw = compare(ref_wts, got_w);
        int idbad = 0;
        for (size_t i = 0; i < ref_ids.size(); i++) {
            if (ref_ids[i] != got_i[i]) {
                idbad++;
            }
        }
        char tag[64];
        snprintf(tag, sizeof(tag), "use_vl=%d logits", use_vl);
        report(tag, ml);
        snprintf(tag, sizeof(tag), "use_vl=%d weights", use_vl);
        report(tag, mw);
        printf("  use_vl=%d ids: %s (%d/%zu differ)\n", use_vl, idbad ? "MISMATCH" : "EXACT", idbad,
               ref_ids.size());
        ok(idbad == 0, "router ids exact");
        ok(mw.max_abs <= 1e-6, "router weights <= 1e-6");
        ok(ml.max_abs < 5e-6, "router logits match the AVX2-order reference");
        sycl::free(dg, q);
        sycl::free(dx, q);
        sycl::free(db, q);
        sycl::free(dbv, q);
        sycl::free(dl, q);
        sycl::free(di, q);
        sycl::free(dw, q);
    }

    // Exact-tie / bias-only edge: x = 0 makes every logit exactly 0, so the
    // selection must fall back to the bias and then to the lowest index.
    {
        std::vector<float> xz((size_t)T * K, 0.f);
        std::vector<float> zb(E, 0.f);
        std::vector<float> ref_logits, ref_wts;
        std::vector<int32_t> ref_ids;
        router_ref(wb.data(), 30, xz.data(), 1, K, zb.data(), nullptr, 0, E, topk, scale, ref_logits,
                   ref_ids, ref_wts);
        uint16_t * dg = dev_upload(q, wb);
        float * dx = dev_upload(q, xz);
        float * db = dev_upload(q, zb);
        float * dl = dev_alloc<float>(q, E);
        int32_t * di = dev_alloc<int32_t>(q, topk);
        float * dw = dev_alloc<float>(q, topk);
        ds41_router_launch(q, dg, 30, dx, 1, K, db, nullptr, 0, E, topk, scale, dl, di, dw);
        const auto gi = dev_download(q, di, topk);
        int bad = 0;
        for (int j = 0; j < topk; j++) {
            if (gi[j] != j) { // ties -> 0..topk-1
                bad++;
            }
        }
        printf("  tie rule (x=0): ids = %d %d %d ... %s\n", gi[0], gi[1], gi[2],
               bad ? "MISMATCH" : "OK (0..7)");
        ok(bad == 0, "router ties to lowest index");
        sycl::free(dg, q);
        sycl::free(dx, q);
        sycl::free(db, q);
        sycl::free(dl, q);
        sycl::free(di, q);
        sycl::free(dw, q);
    }

    // F32 gate rows.
    {
        std::vector<float> ref_logits, ref_wts;
        std::vector<int32_t> ref_ids;
        router_ref(wf.data(), 0, x.data(), T, K, bias.data(), bias_vl.data(), 0, E, topk, scale,
                   ref_logits, ref_ids, ref_wts);
        float * dg = dev_upload(q, wf);
        float * dx = dev_upload(q, x);
        float * db = dev_upload(q, bias);
        float * dl = dev_alloc<float>(q, (size_t)T * E);
        int32_t * di = dev_alloc<int32_t>(q, (size_t)T * topk);
        float * dw = dev_alloc<float>(q, (size_t)T * topk);
        ds41_router_launch(q, dg, 0, dx, T, K, db, nullptr, 0, E, topk, scale, dl, di, dw);
        const auto got_i = dev_download(q, di, (size_t)T * topk);
        const Metrics mw = compare(ref_wts, dev_download(q, dw, (size_t)T * topk));
        int bad = 0;
        for (size_t i = 0; i < ref_ids.size(); i++) {
            if (ref_ids[i] != got_i[i]) {
                bad++;
            }
        }
        report("f32 gate weights", mw);
        printf("  f32 gate ids: %s (%d differ)\n", bad ? "MISMATCH" : "EXACT", bad);
        ok(bad == 0 && mw.max_abs <= 1e-6, "f32 gate router exact");
        sycl::free(dg, q);
        sycl::free(dx, q);
        sycl::free(db, q);
        sycl::free(dl, q);
        sycl::free(di, q);
        sycl::free(dw, q);
    }

    // top_k larger than the model's 8 exercises the full selection loop.
    {
        const int big_k = 32;
        std::vector<float> ref_logits, ref_wts;
        std::vector<int32_t> ref_ids;
        router_ref(wb.data(), 30, x.data(), T, K, bias.data(), bias_vl.data(), 0, E, big_k, scale,
                   ref_logits, ref_ids, ref_wts);
        uint16_t * dg = dev_upload(q, wb);
        float * dx = dev_upload(q, x);
        float * db = dev_upload(q, bias);
        float * dl = dev_alloc<float>(q, (size_t)T * E);
        int32_t * di = dev_alloc<int32_t>(q, (size_t)T * big_k);
        float * dw = dev_alloc<float>(q, (size_t)T * big_k);
        ds41_router_launch(q, dg, 30, dx, T, K, db, nullptr, 0, E, big_k, scale, dl, di, dw);
        const auto got_i = dev_download(q, di, (size_t)T * big_k);
        const Metrics mw = compare(ref_wts, dev_download(q, dw, (size_t)T * big_k));
        int bad = 0;
        for (size_t i = 0; i < ref_ids.size(); i++) {
            if (ref_ids[i] != got_i[i]) {
                bad++;
            }
        }
        report("top_k=32 weights", mw);
        printf("  top_k=32 ids: %s (%d differ)\n", bad ? "MISMATCH" : "EXACT", bad);
        ok(bad == 0 && mw.max_abs <= 1e-6, "router top_k=32 exact");
        sycl::free(dg, q);
        sycl::free(dx, q);
        sycl::free(db, q);
        sycl::free(dl, q);
        sycl::free(di, q);
        sycl::free(dw, q);
    }

    // timing, one token / BF16
    {
        uint16_t * dg = dev_upload(q, wb);
        float * dx = dev_upload(q, x);
        float * db = dev_upload(q, bias);
        float * dl = dev_alloc<float>(q, (size_t)E);
        int32_t * di = dev_alloc<int32_t>(q, topk);
        float * dw = dev_alloc<float>(q, topk);
        const double ms = time_launch(
            q, [&] { ds41_router_launch(q, dg, 30, dx, 1, K, db, nullptr, 0, E, topk, scale, dl, di, dw); },
            100);
        printf("  time 1 tok x 384 experts x 5120 (bf16): %.4f ms\n", ms);
        sycl::free(dg, q);
        sycl::free(dx, q);
        sycl::free(db, q);
        sycl::free(dl, q);
        sycl::free(di, q);
        sycl::free(dw, q);
    }
}

static void test_select(sycl::queue & q) {
    printf("[select] top-512 of 2048, ties and -inf\n");
    struct Case {
        const char * name;
        std::vector<float> s;
        int k;
    };
    std::vector<Case> cases;
    cases.push_back({"random", rnd(2048, 1.f), 512});
    {
        std::vector<float> s = rnd(2048, 1.f);
        // exact duplicates -> must break to the lower index
        for (int i = 0; i < 64; i++) {
            s[i] = 0.25f;
        }
        for (int i = 300; i < 900; i++) {
            s[i] = -INFINITY;
        }
        cases.push_back({"ties+-inf", s, 512});
    }
    cases.push_back({"k=1", rnd(2048, 1.f), 1});
    cases.push_back({"k=777", rnd(2048, 1.f), 777});

    for (auto & c : cases) {
        const std::vector<int32_t> ref = select_ref(c.s, 2048, c.k);
        float * ds = dev_upload(q, c.s);
        int32_t * di = dev_alloc<int32_t>(q, (size_t)c.k);
        ds41_select_launch(q, ds, 2048, c.k, di);
        const auto got = dev_download(q, di, (size_t)c.k);
        int bad = 0;
        for (int i = 0; i < c.k; i++) {
            if (ref[(size_t)i] != got[(size_t)i]) {
                bad++;
            }
        }
        printf("  %-10s k=%-4d -> %s (%d differ)\n", c.name, c.k, bad ? "MISMATCH" : "EXACT", bad);
        ok(bad == 0, "select matches the partial_sort oracle");
        sycl::free(ds, q);
        sycl::free(di, q);
    }
    {
        std::vector<float> s = rnd(2048, 1.f);
        float * ds = dev_upload(q, s);
        int32_t * di = dev_alloc<int32_t>(q, 512);
        const double ms = time_launch(q, [&] { ds41_select_launch(q, ds, 2048, 512, di); }, 200);
        printf("  time 512 of 2048: %.4f ms\n", ms);
        sycl::free(ds, q);
        sycl::free(di, q);
    }
}

static void test_cand(sycl::queue & q) {
    printf("[cand_blocks / cand_mask] n_vis = 1000/512/7, block 8/4/4\n");
    struct Case {
        int n_vis;
        int block;
        int topk;
        std::vector<float> score;
    };
    std::vector<Case> cases;
    cases.push_back({1000, 8, 16, rnd(1000, 1.f)});
    cases.push_back({512, 4, 5, rnd(512, 1.f)});
    cases.push_back({7, 4, 8, rnd(7, 1.f)});
    {
        std::vector<float> s = rnd(512, 1.f);
        for (int i = 0; i < 20; i++) {
            s[i] = -INFINITY;
        }
        cases.push_back({512, 4, 3, s});
    }
    for (auto & c : cases) {
        std::vector<uint8_t> ref_keep;
        int ref_nb = 0;
        cand_ref(c.score, c.n_vis, c.block, c.topk, ref_keep, ref_nb);
        float * ds = dev_upload(q, c.score);
        const int nbmax = (c.n_vis + c.block - 1) / c.block;
        uint8_t * dk = dev_alloc<uint8_t>(q, (size_t)nbmax + 1);
        int * dnb = dev_alloc<int>(q, 1);
        q.memset(dk, 0, (size_t)nbmax + 1).wait();
        ds41_cand_blocks_launch(q, ds, c.n_vis, c.block, c.topk, dk, dnb);
        const auto got_keep = dev_download(q, dk, (size_t)ref_nb);
        const int got_nb = dev_download(q, dnb, 1)[0];
        bool same = (got_nb == ref_nb) && got_keep.size() == ref_keep.size() &&
                    std::memcmp(got_keep.data(), ref_keep.data(), ref_keep.size()) == 0;
        // mask
        std::vector<float> ref_score = c.score;
        for (int i = 0; i < c.n_vis; i++) {
            const int b = i / c.block;
            if (b >= ref_nb || !ref_keep[(size_t)b]) {
                ref_score[(size_t)i] = -INFINITY;
            }
        }
        ds41_cand_mask_launch(q, ds, dk, c.n_vis, c.block);
        const auto got_score = dev_download(q, ds, (size_t)c.n_vis);
        bool mask_ok = std::memcmp(got_score.data(), ref_score.data(), ref_score.size() * sizeof(float)) == 0;
        printf("  n_vis=%4d block=%d topk=%d nb=%d -> keep %s, mask %s\n", c.n_vis, c.block, c.topk,
               got_nb, same ? "EXACT" : "MISMATCH", mask_ok ? "EXACT" : "MISMATCH");
        ok(same, "cand blocks match compute_candidates_one");
        ok(mask_ok, "cand mask bit-exact");
        sycl::free(ds, q);
        sycl::free(dk, q);
        sycl::free(dnb, q);
    }
}

static void test_gather_window(sycl::queue & q) {
    printf("[gather_sel] 32 of 64 rows x 512\n");
    {
        const int nrow = 64, d = 512, nsel = 32;
        std::vector<float> ckv = rnd((size_t)nrow * d, 1.f);
        std::vector<int32_t> sel((size_t)nsel);
        for (int i = 0; i < nsel; i++) {
            sel[(size_t)i] = ((i * 37) % nrow);
        }
        sel[0] = -1; // "not visible" sentinel zeroes the row
        std::vector<float> ref((size_t)nsel * d);
        for (int i = 0; i < nsel; i++) {
            if (sel[(size_t)i] >= 0) {
                std::memcpy(ref.data() + (size_t)i * d, ckv.data() + (size_t)sel[(size_t)i] * d,
                            sizeof(float) * d);
            }
        }
        float * dc = dev_upload(q, ckv);
        int32_t * ds = dev_upload(q, sel);
        float * dout = dev_alloc<float>(q, (size_t)nsel * d);
        ds41_gather_sel_launch(q, dc, ds, nsel, d, dout);
        const auto got = dev_download(q, dout, (size_t)nsel * d);
        const bool same = std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0;
        printf("  gather -> %s\n", same ? "MEMCMP-EXACT" : "MISMATCH");
        ok(same, "gather_sel memcmp");
        sycl::free(dc, q);
        sycl::free(ds, q);
        sycl::free(dout, q);
    }
    printf("[window_write] cap=128 d=512, 200 positions\n");
    {
        const int cap = 128, d = 512, npos = 200;
        std::vector<float> row((size_t)npos * d);
        std::mt19937 rr(7);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : row) {
            v = nd(rr);
        }
        std::vector<float> ref_win((size_t)cap * d, 0.f);
        std::vector<int> ref_pos((size_t)cap, -1);
        float * drow = dev_upload(q, row);
        float * dw = dev_alloc<float>(q, (size_t)cap * d);
        q.memset(dw, 0, (size_t)cap * d * sizeof(float)).wait();
        int * dp = dev_alloc<int>(q, cap);
        q.memset(dp, 0xff, (size_t)cap * sizeof(int)).wait();
        for (int p = 0; p < npos; p++) {
            std::memcpy(ref_win.data() + (size_t)(p % cap) * d, row.data() + (size_t)p * d,
                        sizeof(float) * d);
            ref_pos[(size_t)(p % cap)] = p;
            ds41_window_write_launch(q, drow + (size_t)p * d, dw, dp, p, cap, d);
        }
        const auto got_win = dev_download(q, dw, (size_t)cap * d);
        const auto got_pos = dev_download(q, dp, cap);
        const bool w_ok = std::memcmp(ref_win.data(), got_win.data(), ref_win.size() * sizeof(float)) == 0;
        const bool p_ok = std::memcmp(ref_pos.data(), got_pos.data(), ref_pos.size() * sizeof(int)) == 0;
        printf("  ring %s, positions %s (last slot %d)\n", w_ok ? "MEMCMP-EXACT" : "MISMATCH",
               p_ok ? "MEMCMP-EXACT" : "MISMATCH", got_pos[(size_t)((npos - 1) % cap)]);
        ok(w_ok && p_ok, "window write memcmp");
        sycl::free(drow, q);
        sycl::free(dw, q);
        sycl::free(dp, q);
    }
}

static void test_engram(sycl::queue & q) {
    printf("[engram_apply] hc=4 d=5120, random / zero-h / negative-dot\n");
    const int hc = 4, d = 5120;
    struct Case {
        const char * name;
        std::vector<float> h;
        std::vector<float> kv;
        std::vector<float> gate;
    };
    std::vector<Case> cases;
    cases.push_back({"random", rnd((size_t)hc * d, 0.5f), rnd((size_t)(hc + 1) * d, 0.5f),
                     rnd((size_t)hc * d, 0.5f)});
    cases.push_back({"zero-h", std::vector<float>((size_t)hc * d, 0.f), rnd((size_t)(hc + 1) * d, 0.5f),
                     rnd((size_t)hc * d, 0.5f)});
    {
        std::vector<float> h = rnd((size_t)hc * d, 0.5f);
        std::vector<float> kv = rnd((size_t)(hc + 1) * d, 0.5f);
        std::vector<float> gate = rnd((size_t)hc * d, 0.5f);
        for (int k = 0; k < hc; k++) {
            for (int i = 0; i < d; i++) {
                gate[(size_t)k * d + i] = -std::fabs(gate[(size_t)k * d + i]);
            }
        }
        cases.push_back({"negative-dot", h, kv, gate});
    }
    for (auto & c : cases) {
        std::vector<float> ref = c.h;
        engram_ref(ref, c.kv, c.gate, hc, d);
        float * dh = dev_upload(q, c.h);
        float * dkv = dev_upload(q, c.kv);
        float * dg = dev_upload(q, c.gate);
        ds41_engram_apply_launch(q, dh, dkv, dg, hc, d);
        const auto got = dev_download(q, dh, (size_t)hc * d);
        const Metrics m = compare(ref, got);
        report(c.name, m);
        // An all-zero h makes dot exactly 0 and `copysign(sqrt(1e-6), 0)`
        // exposes the SIGN of that zero; +0 and -0 therefore differ by
        // g(+1e-3)-g(-1e-3) ~ 5e-4 times the value.  Both are the same gate
        // magnitude, so this one case is judged in that band.
        const double tol = std::strcmp(c.name, "zero-h") == 0 ? 2e-3 : 2e-4;
        if (std::strcmp(c.name, "zero-h") == 0) {
            printf("    (zero-h: exact-zero dot, sign-of-zero band)\n");
        }
        ok(m.max_abs < tol || m.rel_peak < 2e-5, "engram within tolerance");
        sycl::free(dh, q);
        sycl::free(dkv, q);
        sycl::free(dg, q);
    }
    {
        std::vector<float> h = rnd((size_t)hc * d, 0.5f), kv = rnd((size_t)(hc + 1) * d, 0.5f),
                            gate = rnd((size_t)hc * d, 0.5f);
        float * dh = dev_upload(q, h);
        float * dkv = dev_upload(q, kv);
        float * dg = dev_upload(q, gate);
        const double ms = time_launch(q, [&] { ds41_engram_apply_launch(q, dh, dkv, dg, hc, d); }, 200);
        printf("  time hc=4 d=5120: %.4f ms\n", ms);
        sycl::free(dh, q);
        sycl::free(dkv, q);
        sycl::free(dg, q);
    }
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        test_fp8(q);
        test_swiglu(q);
        test_router(q);
        test_select(q);
        test_cand(q);
        test_gather_window(q);
        test_engram(q);
        printf("\nds41 ops: %d checks, %d failures\n", g_checks, g_failures);
        return g_failures ? 1 : 0;
    } catch (const std::exception & ex) {
        fprintf(stderr, "fatal: %s\n", ex.what());
        return 2;
    }
}
