// Hermetic unit tests for the host MoE mirror and the QSA host kernels.
//
// The MoE path is the CPU half of a hybrid --layer-map run, so a drift here is
// a layer-number-dependent error rather than a crash; the reference below is
// the same scalar arithmetic the GPU kernels document (double accumulation,
// rank-counted top-k, renormalisation).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cpu_types.h"
#include "quant.h"

using namespace si;

static int g_fail = 0;

static void fail(const char * name, const char * what) {
    std::printf("%-30s FAIL (%s)\n", name, what);
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
    std::printf("%-30s n=%-6zu max|exp|=%.5f max|diff|=%.3e %s\n", name, got.size(), maxv, maxd, ok ? "OK" : "FAIL");
    if (!ok) g_fail++;
}
static void check_i32(const char * name, const std::vector<int32_t> & got, const std::vector<int32_t> & exp) {
    bool ok = got.size() == exp.size();
    for (size_t i = 0; ok && i < got.size(); i++) ok = got[i] == exp[i];
    std::printf("%-30s n=%-6zu %s\n", name, got.size(), ok ? "OK" : "FAIL");
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

// ---------------------------------------------------------------------------
static void test_moe_route() {
    const int n_tok = 3, n_embd = 7, n_expert = 5, top_k = 2;
    std::vector<float> x((size_t)n_tok * n_embd), gate((size_t)n_expert * n_embd);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i);
    for (size_t i = 0; i < gate.size(); i++) gate[i] = rnd((int)i + 300) * 0.1f;
    std::vector<float> logits((size_t)n_tok * n_expert);
    std::vector<int32_t> ids((size_t)n_tok * top_k);
    std::vector<float> wts((size_t)n_tok * top_k);
    cpu_moe_route(x.data(), n_embd, gate.data(), GGML_TYPE_F32, n_tok, n_embd, n_expert, top_k, logits.data(),
                  ids.data(), wts.data(), false);

    // scalar reference, matching the rank-counting tie rule (lower index wins)
    std::vector<int32_t> eids((size_t)n_tok * top_k);
    std::vector<float> ewts((size_t)n_tok * top_k);
    for (int t = 0; t < n_tok; t++) {
        std::vector<double> p(n_expert);
        double mx = -1e300;
        for (int e = 0; e < n_expert; e++) {
            double a = 0;
            for (int k = 0; k < n_embd; k++) a += (double)gate[(size_t)e * n_embd + k] * x[(size_t)t * n_embd + k];
            p[e] = a;
            mx = std::max(mx, a);
        }
        double sum = 0;
        for (int e = 0; e < n_expert; e++) {
            p[e] = std::exp(p[e] - mx);
            sum += p[e];
        }
        for (int e = 0; e < n_expert; e++) p[e] /= sum;
        double wsum = 0;
        for (int e = 0; e < n_expert; e++) {
            int rank = 0;
            for (int f = 0; f < n_expert; f++) {
                if (p[f] > p[e] || (p[f] == p[e] && f < e)) rank++;
            }
            if (rank < top_k) {
                eids[(size_t)t * top_k + rank] = e;
                ewts[(size_t)t * top_k + rank] = (float)p[e];
                wsum += p[e];
            }
        }
        const float den = (float)std::fmax(wsum, 6.103515625e-5);
        for (int j = 0; j < top_k; j++) ewts[(size_t)t * top_k + j] /= den;
    }
    check_i32("cpu_moe_route ids", ids, eids);
    check_f32("cpu_moe_route weights", wts, ewts, 1e-6, 1e-6);
}

// ---------------------------------------------------------------------------
static void test_moe_experts_f32() {
    const int n_tok = 2, top_k = 2, n_embd = 7, n_ff = 10;
    std::vector<float> x((size_t)n_tok * n_embd);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 700);
    std::vector<int32_t> ids = {0, 2, 1, 0};

    cpu_moe M{};
    M.gate_exps.type = M.up_exps.type = M.down_exps.type = GGML_TYPE_F32;
    M.gate_exps.K = M.up_exps.K = n_embd;
    M.gate_exps.N = M.up_exps.N = n_ff;
    M.down_exps.K = n_ff;
    M.down_exps.N = n_embd;
    M.gate_exps.plane_bytes = M.up_exps.plane_bytes = (size_t)n_ff * n_embd * 4;
    M.down_exps.plane_bytes = (size_t)n_embd * n_ff * 4;
    M.n_expert = 3;
    M.top_k = top_k;
    M.n_ff_exp = n_ff;
    M.n_embd = n_embd;

    std::vector<float> g((size_t)3 * n_ff * n_embd), u((size_t)3 * n_ff * n_embd), d((size_t)3 * n_embd * n_ff);
    for (size_t i = 0; i < g.size(); i++) g[i] = rnd((int)i + 1000);
    for (size_t i = 0; i < u.size(); i++) u[i] = rnd((int)i + 3000);
    for (size_t i = 0; i < d.size(); i++) d[i] = rnd((int)i + 5000);
    M.gate_exps.data = (const char *)g.data();
    M.up_exps.data = (const char *)u.data();
    M.down_exps.data = (const char *)d.data();

    const int np = n_tok * top_k;
    std::vector<float> acc((size_t)np * n_embd, 0.f);
    cpu_moe_experts(M, x.data(), n_embd, n_tok, top_k, ids.data(), nullptr, acc.data(), n_embd, false);

    std::vector<float> exp((size_t)np * n_embd, 0.f);
    for (int pr = 0; pr < np; pr++) {
        const int t = pr / top_k, e = ids[pr];
        std::vector<double> h(n_ff);
        for (int i = 0; i < n_ff; i++) {
            double ag = 0, au = 0;
            for (int k = 0; k < n_embd; k++) {
                ag += (double)g[((size_t)e * n_ff + i) * n_embd + k] * x[(size_t)t * n_embd + k];
                au += (double)u[((size_t)e * n_ff + i) * n_embd + k] * x[(size_t)t * n_embd + k];
            }
            h[i] = (double)((float)silu_((float)ag) * (float)au);
        }
        for (int n = 0; n < n_embd; n++) {
            double a = 0;
            for (int i = 0; i < n_ff; i++) a += (double)d[((size_t)e * n_embd + n) * n_ff + i] * h[i];
            exp[(size_t)pr * n_embd + n] = (float)a;
        }
    }
    check_f32("cpu_moe_experts F32", acc, exp, 1e-5, 1e-3);

    // weighted reduction
    std::vector<float> wts((size_t)np);
    for (int i = 0; i < np; i++) wts[i] = 0.2f + 0.1f * i;
    std::vector<float> out((size_t)n_tok * n_embd, 1.5f), eout((size_t)n_tok * n_embd, 1.5f);
    cpu_moe_reduce(acc.data(), wts.data(), top_k, n_tok, n_embd, n_embd, out.data(), n_embd);
    for (int t = 0; t < n_tok; t++) {
        for (int n = 0; n < n_embd; n++) {
            double a = 0;
            for (int j = 0; j < top_k; j++) {
                const size_t pr = (size_t)t * top_k + j;
                a += (double)wts[pr] * acc[pr * n_embd + n];
            }
            eout[(size_t)t * n_embd + n] += (float)a;
        }
    }
    check_f32("cpu_moe_reduce", out, eout, 1e-6, 1e-3);

    // shared-expert epilogue
    std::vector<float> shared((size_t)n_tok * n_embd), egate(n_embd);
    for (size_t i = 0; i < shared.size(); i++) shared[i] = rnd((int)i + 9000);
    for (int i = 0; i < n_embd; i++) egate[i] = rnd(i + 12000) * 0.1f;
    std::vector<float> eout2 = out, eexp = out;
    cpu_moe_epilogue(x.data(), n_embd, egate.data(), GGML_TYPE_F32, n_tok, n_embd, shared.data(), n_embd,
                     eout2.data(), n_embd);
    (void)eexp;
    for (int t = 0; t < n_tok; t++) {
        double a = 0;
        for (int k = 0; k < n_embd; k++) a += (double)egate[k] * x[(size_t)t * n_embd + k];
        const float gfac = sigmoid_((float)a);
        for (int n = 0; n < n_embd; n++) eexp[(size_t)t * n_embd + n] += gfac * shared[(size_t)t * n_embd + n];
    }
    check_f32("cpu_moe_epilogue", eout2, eexp, 1e-6, 1e-3);
}

// ---------------------------------------------------------------------------
static void test_qsa() {
    const int K = 24, N = 5, n_tok = 2;
    std::vector<uint16_t> w((size_t)N * K);
    std::vector<float> x((size_t)n_tok * K);
    for (size_t i = 0; i < w.size(); i++) {
        const float v = rnd((int)i + 400) * 0.5f;
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        w[i] = (uint16_t)(bits >> 16);
    }
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 800);
    std::vector<float> out((size_t)n_tok * N), exp((size_t)n_tok * N);
    cpu_qsa_bf16_gemv(w.data(), x.data(), K, out.data(), N, n_tok, K, N);
    for (int t = 0; t < n_tok; t++) {
        for (int n = 0; n < N; n++) {
            double a = 0;
            for (int k = 0; k < K; k++) {
                a += (double)ggml_bf16_to_float(w[(size_t)n * K + k]) * x[(size_t)t * K + k];
            }
            exp[(size_t)t * N + n] = (float)a;
        }
    }
    check_f32("cpu_qsa_bf16_gemv", out, exp, 1e-6, 1e-3);

    // pool scores: sum over heads of max(dot,0), times inv
    const int id = 8, inh = 3, n_pool = 4;
    std::vector<float> q((size_t)inh * id), pool((size_t)n_pool * id), scores(n_pool), escores(n_pool);
    for (size_t i = 0; i < q.size(); i++) q[i] = rnd((int)i + 1200);
    for (size_t i = 0; i < pool.size(); i++) pool[i] = rnd((int)i + 1600);
    const float inv = 0.37f;
    cpu_qsa_score_pools(q.data(), pool.data(), n_pool, id, inh, inv, scores.data());
    for (int p = 0; p < n_pool; p++) {
        double s = 0;
        for (int h = 0; h < inh; h++) {
            double dd = 0;
            for (int k = 0; k < id; k++) dd += (double)q[(size_t)h * id + k] * pool[(size_t)p * id + k];
            s += dd > 0 ? dd : 0;
        }
        escores[p] = (float)s * inv;
    }
    check_f32("cpu_qsa_score_pools", scores, escores, 1e-6, 1e-3);

    // sparse attention over selected rows, f32 and f16 bases.
    for (int hd : {16, 256}) {
        const int nsel = 5, kv_stride = hd;
        std::vector<float> qq((size_t)2 * hd), kb((size_t)8 * kv_stride), vb((size_t)8 * kv_stride);
        std::vector<int> sel{0, 3, 1, 7, 4};
        for (size_t i = 0; i < qq.size(); i++) qq[i] = rnd((int)i + 2000);
        for (size_t i = 0; i < kb.size(); i++) kb[i] = rnd((int)i + 2400);
        for (size_t i = 0; i < vb.size(); i++) vb[i] = rnd((int)i + 2800);
        std::vector<float> o(hd), e(hd);
        cpu_qsa_attn_head(qq.data(), kb.data(), vb.data(), kv_stride, sel.data(), nsel, hd, 0.1f, o.data());
        // scalar reference (same order)
        std::vector<float> sc(nsel);
        float mx = -3.402823466e38f;
        for (int j = 0; j < nsel; j++) {
            float d = 0;
            for (int k = 0; k < hd; k++) d += qq[k] * kb[(size_t)sel[j] * kv_stride + k];
            sc[j] = d * 0.1f;
            mx = std::max(mx, sc[j]);
        }
        float sum = 0;
        for (int j = 0; j < nsel; j++) {
            sc[j] = std::exp(sc[j] - mx);
            sum += sc[j];
        }
        for (int k = 0; k < hd; k++) e[k] = 0;
        for (int j = 0; j < nsel; j++) {
            const float wgt = sc[j] / sum;
            for (int k = 0; k < hd; k++) e[k] += wgt * vb[(size_t)sel[j] * kv_stride + k];
        }
        for (int k = 0; k < hd; k++) e[k] *= sigmoid_(qq[hd + k]);
        char name[64];
        std::snprintf(name, sizeof(name), "cpu_qsa_attn_head f32 hd=%d", hd);
        check_f32(name, o, e, 1e-5, 1e-3);

        // f16 bases: convert the same rows (f16 round-trip is exact for the
        // reference's reads, so compare against the f16 values)
        std::vector<uint16_t> k16(kb.size()), v16(vb.size());
        for (size_t i = 0; i < kb.size(); i++) k16[i] = ggml_float_to_half(kb[i]);
        for (size_t i = 0; i < vb.size(); i++) v16[i] = ggml_float_to_half(vb[i]);
        std::vector<float> o16(hd), e16(hd);
        cpu_qsa_attn_head_h(qq.data(), k16.data(), v16.data(), kv_stride, sel.data(), nsel, hd, 0.1f, o16.data());
        std::vector<float> sc16(nsel);
        mx = -3.402823466e38f;
        for (int j = 0; j < nsel; j++) {
            float d = 0;
            for (int k = 0; k < hd; k++) d += qq[k] * ggml_half_to_float(k16[(size_t)sel[j] * kv_stride + k]);
            sc16[j] = d * 0.1f;
            mx = std::max(mx, sc16[j]);
        }
        sum = 0;
        for (int j = 0; j < nsel; j++) {
            sc16[j] = std::exp(sc16[j] - mx);
            sum += sc16[j];
        }
        for (int k = 0; k < hd; k++) e16[k] = 0;
        for (int j = 0; j < nsel; j++) {
            const float wgt = sc16[j] / sum;
            for (int k = 0; k < hd; k++) e16[k] += wgt * ggml_half_to_float(v16[(size_t)sel[j] * kv_stride + k]);
        }
        for (int k = 0; k < hd; k++) e16[k] *= sigmoid_(qq[hd + k]);
        std::snprintf(name, sizeof(name), "cpu_qsa_attn_head f16 hd=%d", hd);
        check_f32(name, o16, e16, 1e-5, 1e-3);
    }
    std::printf("%-30s %s\n", "cpu_qsa_f16_ok", cpu_qsa_f16_ok() ? "true (OK)" : "false (scalar)");
}

int main() {
    test_moe_route();
    test_moe_experts_f32();
    test_qsa();
    if (g_fail) {
        std::printf("test_cpu_moe_qsa_units: FAILURES: %d\n", g_fail);
        return 1;
    }
    std::printf("test_cpu_moe_qsa_units: all units OK\n");
    return 0;
}
