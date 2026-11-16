// Hermetic unit tests for the CPU elementwise / copy / quantize kernels.
//
// Every case builds its own deterministic inputs and compares one launcher
// against a scalar reference written here.  No GGUF and no GPU: this is the
// reference half of the GPU twins' differential tests, and it is the only
// coverage that can fail independently of both implementations.
//
// Run with PF_CPU_ISA=scalar|avx2|avx512 to pin an ISA variant; the arithmetic
// is identical by construction, so a variant that fails is a kernel bug.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "cpu_types.h"

using namespace si;

static int g_fail = 0;

static void fail(const char * name, const char * what) {
    std::printf("%-28s FAIL (%s)\n", name, what);
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
    std::printf("%-28s n=%-6zu max|exp|=%.5f max|diff|=%.3e %s\n", name, got.size(), maxv, maxd, ok ? "OK" : "FAIL");
    if (!ok) {
        g_fail++;
    }
}

static void check_i32(const char * name, const std::vector<int32_t> & got, const std::vector<int32_t> & exp) {
    bool ok = got.size() == exp.size();
    for (size_t i = 0; ok && i < got.size(); i++) {
        ok = got[i] == exp[i];
    }
    std::printf("%-28s n=%-6zu %s\n", name, got.size(), ok ? "OK" : "FAIL");
    if (!ok) {
        g_fail++;
    }
}

static void check_i8(const char * name, const std::vector<int8_t> & got, const std::vector<int8_t> & exp) {
    bool ok = got.size() == exp.size();
    for (size_t i = 0; ok && i < got.size(); i++) {
        ok = got[i] == exp[i];
    }
    std::printf("%-28s n=%-6zu %s\n", name, got.size(), ok ? "OK" : "FAIL");
    if (!ok) {
        g_fail++;
    }
}

static float sigmoid_(float v) {
    return 1.0f / (1.0f + std::exp(-v));
}
static float silu_(float v) {
    return v / (1.0f + std::exp(-v));
}

// deterministic values, never a power-of-two pattern that hides tail bugs
static float rnd(int i, float scale = 2.0f) {
    const unsigned h = (unsigned)(i * 2654435761u + 1013904223u);
    return ((float)((h >> 8) & 0xffff) / 32768.0f - 1.0f) * scale;
}

// ---------------------------------------------------------------------------
// copy_row: the last real row (mode-2 partial row) and a named row.
static void test_copy_row() {
    cpu_step_info info{};
    info.n_rows = 2;
    info.n_real = 32;
    info.tpb = 32;
    info.n_real_row[1] = 5;
    info.active[0] = info.active[1] = 1;
    const int n = 7;
    std::vector<float> src((size_t)2 * 32 * n);
    for (size_t i = 0; i < src.size(); i++) src[i] = rnd((int)i);

    std::vector<float> dst(n), exp(n);
    cpu_copy_row(src.data(), dst.data(), &info, n, -1);
    // row 1, token 4
    std::memcpy(exp.data(), src.data() + ((size_t)1 * 32 + 4) * n, n * 4);
    check_f32("cpu_copy_row last", dst, exp, 0.0);

    std::vector<float> dst2(n), exp2(n);
    cpu_copy_row(src.data(), dst2.data(), &info, n, 0);
    std::memcpy(exp2.data(), src.data(), n * 4);
    check_f32("cpu_copy_row row=0", dst2, exp2, 0.0);
}

// ---------------------------------------------------------------------------
// embed: a dequantized F32 row, a Q8_0 row (tail width 96), and an image row
// that bypasses the token table.
static void test_embed() {
    const int n_embd = 96; // 3 Q8_0 blocks, not a power of two
    cpu_step_info info{};
    info.n_rows = 1;
    info.n_real = 3;
    info.tpb = 32;
    info.active[0] = 1;
    info.tokens[0] = 2;
    info.tokens[1] = 0;
    info.tokens[2] = 1;
    info.img_row[0] = info.img_row[1] = info.img_row[2] = -1;

    // F32 table
    std::vector<float> tab((size_t)4 * n_embd);
    for (size_t i = 0; i < tab.size(); i++) tab[i] = rnd((int)i);
    std::vector<float> out((size_t)3 * n_embd), exp((size_t)3 * n_embd);
    cpu_embed(tab.data(), GGML_TYPE_F32, &info, out.data(), n_embd, (size_t)n_embd * 4);
    for (int t = 0; t < 3; t++) {
        std::memcpy(exp.data() + (size_t)t * n_embd, tab.data() + (size_t)info.tokens[t] * n_embd, n_embd * 4);
    }
    check_f32("cpu_embed F32", out, exp, 0.0);

    // Q8_0 table
    const int nb = n_embd / 32;
    std::vector<block_q8_0> qtab((size_t)4 * nb);
    for (size_t b = 0; b < qtab.size(); b++) {
        qtab[b].d = ggml_float_to_half(0.01f * (float)((int)(b % 17) + 1));
        for (int i = 0; i < 32; i++) qtab[b].qs[i] = (int8_t)((int)((b * 13 + i * 7) % 255) - 127);
    }
    std::vector<float> outq((size_t)3 * n_embd), expq((size_t)3 * n_embd);
    cpu_embed(qtab.data(), GGML_TYPE_Q8_0, &info, outq.data(), n_embd, (size_t)nb * sizeof(block_q8_0));
    for (int t = 0; t < 3; t++) {
        dequantize_row(GGML_TYPE_Q8_0, &qtab[(size_t)info.tokens[t] * nb], expq.data() + (size_t)t * n_embd, n_embd);
    }
    check_f32("cpu_embed Q8_0", outq, expq, 0.0);

    // image row overrides token 1
    std::vector<float> img((size_t)2 * n_embd);
    for (size_t i = 0; i < img.size(); i++) img[i] = rnd((int)i + 100);
    info.img_embd = img.data();
    info.img_row[1] = 1;
    std::vector<float> outm((size_t)3 * n_embd), expm((size_t)3 * n_embd);
    cpu_embed(tab.data(), GGML_TYPE_F32, &info, outm.data(), n_embd, (size_t)n_embd * 4);
    std::memcpy(expm.data(), tab.data() + (size_t)info.tokens[0] * n_embd, n_embd * 4);
    std::memcpy(expm.data() + n_embd, img.data() + n_embd, n_embd * 4);
    std::memcpy(expm.data() + 2 * n_embd, tab.data() + (size_t)info.tokens[2] * n_embd, n_embd * 4);
    check_f32("cpu_embed image row", outm, expm, 0.0);
}

// ---------------------------------------------------------------------------
// gated_norm: per-head RMSNorm with a silu (and sigmoid) gate.
static void test_gated_norm() {
    const int n_heads = 3, head_dim = 5, n_real = 2, tpb = 32, n_rows = 1;
    const float eps = 1e-5f;
    const int per = n_heads * head_dim;
    cpu_step_info info{};
    info.n_rows = n_rows;
    info.n_real = n_real;
    info.tpb = tpb;
    info.active[0] = 1;
    std::vector<float> attn((size_t)tpb * per), z((size_t)tpb * per), w(head_dim);
    for (size_t i = 0; i < attn.size(); i++) attn[i] = rnd((int)i);
    for (size_t i = 0; i < z.size(); i++) z[i] = rnd((int)i + 5000);
    for (int i = 0; i < head_dim; i++) w[i] = 0.5f + 0.1f * i;
    std::vector<float> out((size_t)tpb * per, -9.f), exp((size_t)tpb * per, -9.f);
    cpu_gated_norm(attn.data(), z.data(), w.data(), out.data(), &info, n_heads, head_dim, eps, n_rows, n_real, 0, false);
    for (int t = 0; t < n_real; t++) {
        for (int h = 0; h < n_heads; h++) {
            const size_t base = (size_t)t * per + (size_t)h * head_dim;
            double ss = 0;
            for (int d = 0; d < head_dim; d++) ss += (double)attn[base + d] * attn[base + d];
            const float inv = 1.0f / std::sqrt((float)(ss / head_dim) + eps);
            for (int d = 0; d < head_dim; d++) {
                exp[base + d] = attn[base + d] * inv * w[d] * silu_(z[base + d]);
            }
        }
    }
    check_f32("cpu_gated_norm silu", out, exp, 1e-6, 1e-3);

    std::vector<float> outs((size_t)tpb * per, -9.f), exps((size_t)tpb * per, -9.f);
    cpu_gated_norm(attn.data(), z.data(), w.data(), outs.data(), &info, n_heads, head_dim, eps, n_rows, n_real, 0, true);
    for (int t = 0; t < n_real; t++) {
        for (int h = 0; h < n_heads; h++) {
            const size_t base = (size_t)t * per + (size_t)h * head_dim;
            double ss = 0;
            for (int d = 0; d < head_dim; d++) ss += (double)attn[base + d] * attn[base + d];
            const float inv = 1.0f / std::sqrt((float)(ss / head_dim) + eps);
            for (int d = 0; d < head_dim; d++) {
                exps[base + d] = attn[base + d] * inv * w[d] * sigmoid_(z[base + d]);
            }
        }
    }
    check_f32("cpu_gated_norm sigmoid", outs, exps, 1e-6, 1e-3);
}

// ---------------------------------------------------------------------------
// xq: symmetric int8 with one f16-rounded scale per 32 values, group-major
// layout, and the per-16 sums used by the weight min correction.
static void test_xq() {
    const int TB = 3, K = 64, G = K / 32;
    std::vector<float> x((size_t)TB * K), up((size_t)TB * K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i);
    for (size_t i = 0; i < up.size(); i++) up[i] = rnd((int)i + 9000);
    cpu_step_info info{};
    info.n_rows = 1;
    info.n_real = TB;
    info.tpb = 32;

    for (int use_up = 0; use_up < 2; use_up++) {
        std::vector<int8_t> x8((size_t)TB * K);
        std::vector<float> xmeta((size_t)TB * G * 2);
        std::vector<int32_t> xsum((size_t)TB * G * 2);
        cpu_xq(x.data(), use_up ? up.data() : nullptr, K, K, x8.data(), xmeta.data(), xsum.data(), &info, TB, K);

        std::vector<int8_t> e8((size_t)TB * K);
        std::vector<float> emeta((size_t)TB * G * 2);
        std::vector<int32_t> esum((size_t)TB * G * 2);
        for (int t = 0; t < TB; t++) {
            for (int g = 0; g < G; g++) {
                float v[32], mx = 0;
                for (int i = 0; i < 32; i++) {
                    const size_t at = (size_t)t * K + g * 32 + i;
                    v[i] = use_up ? silu_(x[at]) * up[at] : x[at];
                    mx = std::max(mx, std::fabs(v[i]));
                }
                const float scale = mx > 0 ? mx / 127.0f : 1.0f;
                const float inv = mx > 0 ? 127.0f / mx : 0.0f;
                int32_t s0 = 0, s1 = 0;
                for (int i = 0; i < 32; i++) {
                    const int q = (int)std::max(-127.0f, std::min(127.0f, std::round(v[i] * inv)));
                    e8[(size_t)g * TB * 32 + (size_t)t * 32 + i] = (int8_t)q;
                    if (i < 16) s0 += q; else s1 += q;
                }
                const size_t gi = (size_t)g * TB + t;
                emeta[gi * 2] = ggml_half_to_float(ggml_float_to_half(mx > 0 ? scale : 0.0f));
                emeta[gi * 2 + 1] = 0.0f;
                esum[gi * 2] = s0;
                esum[gi * 2 + 1] = s1;
            }
        }
        check_i8(use_up ? "cpu_xq up int8" : "cpu_xq int8", x8, e8);
        check_f32(use_up ? "cpu_xq up scale" : "cpu_xq scale", xmeta, emeta, 0.0);
        check_i32(use_up ? "cpu_xq up sums" : "cpu_xq sums", xsum, esum);
    }

    // A zero group must encode as scale 0 (and stay all-zero), not as a
    // division by zero.
    std::vector<float> zx((size_t)K, 0.f);
    std::vector<int8_t> x8((size_t)K, 7);
    std::vector<float> xmeta(2 * G, -1.f);
    std::vector<int32_t> xsum(2 * G, -1);
    cpu_xq(zx.data(), nullptr, K, K, x8.data(), xmeta.data(), xsum.data(), &info, 1, K);
    bool zero_ok = true;
    for (int i = 0; i < K; i++) zero_ok = zero_ok && x8[i] == 0;
    for (int g = 0; g < G; g++) zero_ok = zero_ok && xmeta[(size_t)g * 2] == 0.f && xsum[(size_t)g * 2] == 0;
    std::printf("%-28s %s\n", "cpu_xq zero group", zero_ok ? "OK" : "FAIL");
    if (!zero_ok) g_fail++;
}

// ---------------------------------------------------------------------------
// qwen4exp hyper-connections.
static void test_q4_hc() {
    const int n_tok = 3, hc = 3, n_embd = 5;
    const float eps = 1e-6f;
    std::vector<int32_t> rows{n_tok - 1, 0, 1}; // decode-style remap
    std::vector<float> x((size_t)n_tok * hc * n_embd), w((size_t)hc * n_embd);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i);
    for (size_t i = 0; i < w.size(); i++) w[i] = 0.5f + 0.2f * (float)(i % 7);

    std::vector<float> out((size_t)n_tok * hc * n_embd), exp((size_t)n_tok * hc * n_embd);
    cpu_q4_hc_norm(x.data(), w.data(), out.data(), rows.data(), n_tok, n_embd, hc, eps);
    for (int t = 0; t < n_tok; t++) {
        for (int s = 0; s < hc; s++) {
            const size_t base = (size_t)t * hc * n_embd + (size_t)s * n_embd;
            double ss = 0;
            for (int c = 0; c < n_embd; c++) ss += (double)x[base + c] * x[base + c];
            const float inv = 1.0f / std::sqrt((float)(ss / n_embd) + eps);
            for (int c = 0; c < n_embd; c++) exp[base + c] = x[base + c] * inv * w[(size_t)s * n_embd + c];
        }
    }
    check_f32("cpu_q4_hc_norm", out, exp, 1e-6, 1e-3);

    std::vector<float> gate((size_t)n_tok * hc * n_embd);
    for (size_t i = 0; i < gate.size(); i++) gate[i] = rnd((int)i + 400);
    std::vector<float> mix((size_t)n_tok * n_embd, -7.f), emix((size_t)n_tok * n_embd, -7.f);
    cpu_q4_hc_mix(out.data(), gate.data(), mix.data(), rows.data(), n_tok, n_embd, hc);
    for (int t = 0; t < n_tok; t++) {
        for (int c = 0; c < n_embd; c++) {
            float acc = 0;
            for (int s = 0; s < hc; s++) {
                const size_t o = (size_t)t * hc * n_embd + (size_t)s * n_embd + c;
                acc += out[o] * sigmoid_(gate[o]);
            }
            emix[(size_t)t * n_embd + c] = acc / (float)hc;
        }
    }
    check_f32("cpu_q4_hc_mix", mix, emix, 1e-6, 1e-3);

    std::vector<float> block((size_t)n_tok * n_embd), inject((size_t)n_tok * hc);
    for (size_t i = 0; i < block.size(); i++) block[i] = rnd((int)i + 800);
    for (size_t i = 0; i < inject.size(); i++) inject[i] = rnd((int)i + 1200);
    std::vector<float> res = x, eres = x;
    cpu_q4_hc_combine(res.data(), block.data(), inject.data(), rows.data(), n_tok, n_embd, hc);
    for (int t = 0; t < n_tok; t++) {
        for (int s = 0; s < hc; s++) {
            const float wgt = 2.0f * sigmoid_(inject[(size_t)t * hc + s] / (float)hc);
            for (int c = 0; c < n_embd; c++) {
                eres[(size_t)t * hc * n_embd + (size_t)s * n_embd + c] +=
                    block[(size_t)t * n_embd + c] * wgt;
            }
        }
    }
    check_f32("cpu_q4_hc_combine", res, eres, 1e-6, 1e-3);

    std::vector<float> ss = x, ess = x;
    const float sc = 0.37f;
    cpu_q4_silu_scaled(ss.data(), (int)ss.size(), sc);
    for (size_t i = 0; i < ess.size(); i++) ess[i] = silu_(x[i] * sc);
    check_f32("cpu_q4_silu_scaled", ss, ess, 1e-6, 1e-3);
}

// ---------------------------------------------------------------------------
// q4_up_gemv / q4_arb_gemv: the arbitrary-K projections (the HC up has K=320,
// the shared-expert down K=640).  F32, Q8_0 and BF16 paths.
static void test_q4_gemv() {
    const int K = 40, N = 3, n_tok = 2;
    std::vector<float> x((size_t)n_tok * K), up((size_t)n_tok * K), wf((size_t)N * K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i);
    for (size_t i = 0; i < up.size(); i++) up[i] = rnd((int)i + 300);
    for (size_t i = 0; i < wf.size(); i++) wf[i] = rnd((int)i + 600) * 0.1f;

    std::vector<float> out((size_t)n_tok * N), exp((size_t)n_tok * N);
    cpu_q4_up_gemv(wf.data(), GGML_TYPE_F32, x.data(), K, out.data(), N, n_tok, K, N);
    for (int t = 0; t < n_tok; t++) {
        for (int n = 0; n < N; n++) {
            double a = 0;
            for (int k = 0; k < K; k++) a += (double)wf[(size_t)n * K + k] * x[(size_t)t * K + k];
            exp[(size_t)t * N + n] = (float)a;
        }
    }
    check_f32("cpu_q4_up_gemv F32", out, exp, 1e-5, 1e-3);

    cpu_q4_arb_gemv(wf.data(), GGML_TYPE_F32, x.data(), K, up.data(), out.data(), N, n_tok, K, N);
    for (int t = 0; t < n_tok; t++) {
        for (int n = 0; n < N; n++) {
            double a = 0;
            for (int k = 0; k < K; k++) {
                a += (double)wf[(size_t)n * K + k] * silu_(x[(size_t)t * K + k]) * up[(size_t)t * K + k];
            }
            exp[(size_t)t * N + n] = (float)a;
        }
    }
    check_f32("cpu_q4_arb_gemv F32+up", out, exp, 1e-4, 1e-3);

    // Q8_0: K must be a whole number of 32-value blocks.
    const int K8 = 64, nb = K8 / 32;
    std::vector<block_q8_0> wq((size_t)N * nb);
    for (size_t b = 0; b < wq.size(); b++) {
        wq[b].d = ggml_float_to_half(0.008f * (float)((int)(b % 5) + 1));
        for (int i = 0; i < 32; i++) wq[b].qs[i] = (int8_t)((int)((b * 17 + i * 5) % 255) - 127);
    }
    std::vector<float> x8((size_t)n_tok * K8), up8((size_t)n_tok * K8), o8((size_t)n_tok * N), e8v((size_t)n_tok * N);
    for (size_t i = 0; i < x8.size(); i++) x8[i] = rnd((int)i + 1500);
    for (size_t i = 0; i < up8.size(); i++) up8[i] = rnd((int)i + 2500);
    cpu_q4_arb_gemv(wq.data(), GGML_TYPE_Q8_0, x8.data(), K8, up8.data(), o8.data(), N, n_tok, K8, N);
    for (int t = 0; t < n_tok; t++) {
        for (int n = 0; n < N; n++) {
            double a = 0;
            for (int b = 0; b < nb; b++) {
                const block_q8_0 * q = &wq[(size_t)n * nb + b];
                const float d = ggml_half_to_float(q->d);
                double part = 0;
                for (int i = 0; i < 32; i++) {
                    const size_t at = (size_t)t * K8 + b * 32 + i;
                    part += (double)q->qs[i] * silu_(x8[at]) * up8[at];
                }
                a += d * part;
            }
            e8v[(size_t)t * N + n] = (float)a;
        }
    }
    check_f32("cpu_q4_arb_gemv Q8_0+up", o8, e8v, 1e-4, 1e-3);

    // BF16
    std::vector<uint16_t> wb((size_t)N * K);
    for (size_t i = 0; i < wb.size(); i++) {
        const float v = rnd((int)i + 3500) * 0.1f;
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        wb[i] = (uint16_t)(bits >> 16);
    }
    cpu_q4_up_gemv(wb.data(), GGML_TYPE_BF16, x.data(), K, out.data(), N, n_tok, K, N);
    for (int t = 0; t < n_tok; t++) {
        for (int n = 0; n < N; n++) {
            double a = 0;
            for (int k = 0; k < K; k++) {
                a += (double)ggml_bf16_to_float(wb[(size_t)n * K + k]) * x[(size_t)t * K + k];
            }
            exp[(size_t)t * N + n] = (float)a;
        }
    }
    check_f32("cpu_q4_up_gemv BF16", out, exp, 1e-5, 1e-3);
}

// ---------------------------------------------------------------------------
// Thread-count override is part of the backend contract: 0 restores auto and
// a positive value is what cpu_thread_count reports.
static void test_threads() {
    const int old = cpu_thread_count();
    cpu_set_thread_count(3);
    const int set = cpu_thread_count();
    cpu_set_thread_count(0);
    const int restored = cpu_thread_count();
    const bool ok = set == 3 && restored > 0;
    std::printf("%-28s old=%d set=%d restored=%d %s\n", "cpu_thread_count", old, set, restored, ok ? "OK" : "FAIL");
    if (!ok) g_fail++;
}

int main() {
    test_threads();
    test_copy_row();
    test_embed();
    test_gated_norm();
    test_xq();
    test_q4_hc();
    test_q4_gemv();
    if (g_fail) {
        std::printf("test_cpu_elementwise_units: FAILURES: %d\n", g_fail);
        return 1;
    }
    std::printf("test_cpu_elementwise_units: all units OK\n");
    return 0;
}
