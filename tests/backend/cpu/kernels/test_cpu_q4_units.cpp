// Hermetic unit tests for the qwen4exp host kernels: the native Q5_1/Q8_0
// routed-down projections, the Q5_1 panel repack, the batched int8 expert
// GEMMs, and the PLE gate/apply pair.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cpu_types.h"
#include "kernels/common.h"

using namespace si;

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

struct act {
    std::vector<int8_t> x8;
    std::vector<float> meta;
    std::vector<int32_t> sum;
};

static act make_act(const std::vector<float> & x, int TB, int K) {
    act a;
    a.x8.resize((size_t)TB * K);
    a.meta.resize((size_t)TB * (K / 32) * 2);
    a.sum.resize((size_t)TB * (K / 32) * 2);
    cpu_step_info info{};
    info.n_rows = 1;
    info.n_real = TB;
    info.tpb = 32;
    cpu_xq(x.data(), nullptr, K, K, a.x8.data(), a.meta.data(), a.sum.data(), &info, TB, K);
    return a;
}

static std::vector<block_q5_1> make_q5_1(int N, int G) {
    std::vector<block_q5_1> b((size_t)N * G);
    for (size_t j = 0; j < b.size(); j++) {
        b[j].d = ggml_float_to_half(0.002f * (float)((j % 7) + 1));
        b[j].m = ggml_float_to_half(-0.01f * (float)((j % 5) + 1));
        uint32_t qh = (uint32_t)(j * 2654435761u);
        std::memcpy(b[j].qh, &qh, 4);
        for (int i = 0; i < 16; i++) b[j].qs[i] = (uint8_t)((j * 37 + i * 19) & 0xFF);
    }
    return b;
}

static void test_native_down() {
    const int K = 128, G = K / 32, N = 3, P = 2;
    const size_t row_bytes = (size_t)G * sizeof(block_q5_1);
    auto q5 = make_q5_1(N, G);
    std::vector<float> x((size_t)P * K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 300);
    act a = make_act(x, P, K);
    std::vector<int> pairs{1, 0};
    std::vector<float> acc((size_t)P * N, -1.f), exp((size_t)P * N, -1.f);
    cpu_q5_1_down_grouped(q5.data(), row_bytes, a.x8.data(), a.meta.data(), a.sum.data(), P, P, N, K, acc.data(), N,
                          pairs.data());

    for (int p = 0; p < P; p++) {
        const int pr = pairs[p];
        for (int n = 0; n < N; n++) {
            double total = 0;
            for (int g = 0; g < G; g++) {
                const block_q5_1 & b = q5[(size_t)n * G + g];
                const float d = ggml_half_to_float(b.d), m = ggml_half_to_float(b.m);
                uint32_t qh;
                std::memcpy(&qh, b.qh, 4);
                int32_t qd = 0, xs = 0;
                for (int i = 0; i < 32; i++) {
                    const int lo = (b.qs[i % 16] >> (4 * (i / 16))) & 0xF;
                    const int q = lo | (int)(((qh >> i) & 1u) << 4);
                    const size_t gi = (size_t)g * P + pr;
                    qd += q * a.x8[gi * 32 + i];
                    xs += a.x8[gi * 32 + i];
                }
                total += (double)a.meta[(size_t)g * P * 2 + pr * 2] * (d * (float)qd + m * (float)xs);
            }
            exp[(size_t)pr * N + n] = (float)total;
        }
    }
    check_f32("cpu_q5_1_down_grouped", acc, exp, 1e-6, 1e-3);

    std::vector<float> accs((size_t)P * N, -1.f);
    cpu_q5_1_down_grouped_serial(q5.data(), row_bytes, a.x8.data(), a.meta.data(), a.sum.data(), P, P, N, K, accs.data(),
                                 N, pairs.data());
    check_f32("cpu_q5_1_down_serial", accs, exp, 1e-6, 1e-3);

    // Q8_0: signed weights, no min term.
    const size_t q8rb = (size_t)G * sizeof(block_q8_0);
    std::vector<block_q8_0> q8((size_t)N * G);
    for (size_t j = 0; j < q8.size(); j++) {
        q8[j].d = ggml_float_to_half(0.004f * (float)((j % 9) + 1));
        for (int i = 0; i < 32; i++) q8[j].qs[i] = (int8_t)((int)((j * 29 + i * 7) % 255) - 127);
    }
    std::vector<float> oc((size_t)P * N, -1.f), ec((size_t)P * N, -1.f);
    cpu_q8_0_down_grouped(q8.data(), q8rb, a.x8.data(), a.meta.data(), a.sum.data(), P, P, N, K, oc.data(), N,
                          pairs.data());
    for (int p = 0; p < P; p++) {
        const int pr = pairs[p];
        for (int n = 0; n < N; n++) {
            double total = 0;
            for (int g = 0; g < G; g++) {
                const block_q8_0 & b = q8[(size_t)n * G + g];
                const float d = ggml_half_to_float(b.d);
                int32_t qd = 0;
                for (int i = 0; i < 32; i++) {
                    const size_t gi = (size_t)g * P + pr;
                    qd += (int32_t)b.qs[i] * a.x8[gi * 32 + i];
                }
                total += (double)a.meta[(size_t)g * P * 2 + pr * 2] * d * (float)qd;
            }
            ec[(size_t)pr * N + n] = (float)total;
        }
    }
    check_f32("cpu_q8_0_down_grouped", oc, ec, 1e-6, 1e-3);

    // Panel repack: decode each panel back to its four Q5_1 blocks.
    std::vector<uint8_t> packed((size_t)N * row_bytes);
    q5_1_sb_pack(q5.data(), K, N, 1, packed.data());
    bool pack_ok = true;
    for (int n = 0; n < N; n++) {
        for (int ng = 0; ng < G; ng += 4) {
            const uint8_t * p = packed.data() + (size_t)n * row_bytes + (size_t)(ng / 4) * 96;
            for (int sub = 0; sub < 4; sub++) {
                const block_q5_1 & b = q5[(size_t)n * G + ng + sub];
                uint32_t qh0;
                std::memcpy(&qh0, b.qh, 4);
                pack_ok = pack_ok && std::memcmp(p + 80 + 4 * sub, &b.d, 2) == 0;
                pack_ok = pack_ok && std::memcmp(p + 82 + 4 * sub, &b.m, 2) == 0;
                for (int i = 0; i < 32; i++) {
                    const uint8_t dqs = p[32 * (sub / 2) + i];
                    const int lo = (sub & 1) ? (dqs >> 4) : (dqs & 0xF);
                    const uint32_t qh16 = (uint32_t)p[64 + (i >> 2) * 2] | ((uint32_t)p[65 + (i >> 2) * 2] << 8);
                    const int hi = (qh16 >> (4 * sub + (i & 3))) & 1;
                    const int qp = lo | (hi << 4);
                    const int qo = (b.qs[i % 16] >> (4 * (i / 16))) & 0xF;
                    const int oh = (qh0 >> i) & 1;
                    pack_ok = pack_ok && qp == (qo | (oh << 4));
                }
            }
        }
    }
    std::printf("%-32s %s\n", "q5_1_sb_pack round-trip", pack_ok ? "OK" : "FAIL");
    if (!pack_ok) g_fail++;
}

// ---------------------------------------------------------------------------
static void test_i8_batch() {
    const int N = 3, K = 256, TB = 3;
    std::vector<block_q4_K> blocks(N);
    for (int n = 0; n < N; n++) {
        blocks[n].d = ggml_float_to_half(0.003f * (n + 1));
        blocks[n].dmin = ggml_float_to_half(0.001f * (n + 1));
        for (int i = 0; i < K_SCALE_SIZE; i++) blocks[n].scales[i] = (uint8_t)(0x21 + i);
        for (int i = 0; i < QK_K / 2; i++) blocks[n].qs[i] = (uint8_t)((n * 41 + i * 13) & 0xFF);
    }
    std::vector<float> x((size_t)TB * K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 4000);
    act a = make_act(x, TB, K);

    std::vector<float> ref((size_t)TB * N);
    for (int t = 0; t < TB; t++) {
        for (int n = 0; n < N; n++) {
            std::vector<float> w(K);
            dequantize_row(GGML_TYPE_Q4_K, &blocks[n], w.data(), K);
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)w[k] * x[(size_t)t * K + k];
            ref[(size_t)t * N + n] = (float)s;
        }
    }
    std::vector<float> ob((size_t)TB * N), os((size_t)TB * N), oe((size_t)TB * N);
    cpu_i8_gemm_batch(GGML_TYPE_Q4_K, blocks.data(), a.x8.data(), a.meta.data(), a.sum.data(), K, N, TB, ob.data(), N);
    cpu_i8_gemm_batch_serial(GGML_TYPE_Q4_K, blocks.data(), a.x8.data(), a.meta.data(), a.sum.data(), K, N, TB, os.data(),
                             N);
    cpu_i8_expert_gemm(GGML_TYPE_Q4_K, blocks.data(), a.x8.data(), a.meta.data(), a.sum.data(), K, N, TB, oe.data(), N);
    check_f32("cpu_i8_gemm_batch", ob, ref, 3e-2, 1e-2);
    check_f32("cpu_i8_gemm_batch_serial", os, ref, 3e-2, 1e-2);
    check_f32("cpu_i8_expert_gemm", oe, ref, 3e-2, 1e-2);
}

// ---------------------------------------------------------------------------
static void test_ple() {
    const int n_tok = 2, hc = 2, n_embd = 5, hc_dim = hc * n_embd;
    const int ngram = 1, kern = 2, hist_rows = (kern - 1) * ngram;
    const float eps = 1e-6f;
    std::vector<int32_t> rows{0, 1}, slots{0, 0}, poss{0, 1};

    // gate
    std::vector<float> key((size_t)n_tok * hc * n_embd), hidden((size_t)n_tok * hc * n_embd);
    std::vector<float> wk((size_t)hc * n_embd), wq((size_t)hc * n_embd);
    for (size_t i = 0; i < key.size(); i++) key[i] = rnd((int)i);
    for (size_t i = 0; i < hidden.size(); i++) hidden[i] = rnd((int)i + 500);
    for (size_t i = 0; i < wk.size(); i++) wk[i] = 0.4f + 0.1f * (float)(i % 3);
    for (size_t i = 0; i < wq.size(); i++) wq[i] = 0.3f + 0.2f * (float)(i % 4);
    std::vector<float> gate((size_t)n_tok * hc, -1.f), egate((size_t)n_tok * hc, -1.f);
    cpu_q4_ple_gate(key.data(), hidden.data(), wk.data(), wq.data(), gate.data(), rows.data(), n_tok, n_embd, hc, eps);
    for (int t = 0; t < n_tok; t++) {
        for (int s = 0; s < hc; s++) {
            const size_t base = (size_t)t * hc * n_embd + (size_t)s * n_embd;
            double sk = 0, sq = 0, acc = 0;
            for (int i = 0; i < n_embd; i++) {
                sk += (double)key[base + i] * key[base + i];
                sq += (double)hidden[base + i] * hidden[base + i];
            }
            const float ik = 1.0f / std::sqrt((float)(sk / n_embd) + eps);
            const float iq = 1.0f / std::sqrt((float)(sq / n_embd) + eps);
            for (int i = 0; i < n_embd; i++) {
                acc += (double)key[base + i] * wk[(size_t)s * n_embd + i] * ik * hidden[base + i] *
                       wq[(size_t)s * n_embd + i] * iq;
            }
            const float sv = (float)acc / std::sqrt((float)n_embd);
            const float mag = std::sqrt(std::max(1e-6f, std::fabs(sv)));
            egate[(size_t)t * hc + s] = sigmoid_(sv < 0 ? -mag : mag);
        }
    }
    check_f32("cpu_q4_ple_gate", gate, egate, 1e-6, 1e-3);

    // apply: two tokens in one slot, history empty, kernel 2, ngram 1.
    std::vector<float> value((size_t)n_tok * n_embd), gated((size_t)n_tok * hc_dim), normed((size_t)n_tok * hc_dim);
    std::vector<float> wconv((size_t)hc * n_embd), convw((size_t)kern * hc_dim);
    for (size_t i = 0; i < value.size(); i++) value[i] = rnd((int)i + 900);
    for (size_t i = 0; i < wconv.size(); i++) wconv[i] = 0.5f + 0.2f * (float)(i % 3);
    for (size_t i = 0; i < convw.size(); i++) convw[i] = 0.25f * (float)((int)i % 5) - 0.3f;
    std::vector<float> residual((size_t)n_tok * hc_dim), eres((size_t)n_tok * hc_dim);
    for (size_t i = 0; i < residual.size(); i++) residual[i] = eres[i] = rnd((int)i + 1300);
    std::vector<float> hist((size_t)hist_rows * hc_dim, 0.f);
    std::vector<int32_t> nhist{0};
    cpu_q4_ple_apply(value.data(), gate.data(), wconv.data(), convw.data(), residual.data(), gated.data(),
                     normed.data(), hist.data(), nhist.data(), rows.data(), slots.data(), poss.data(), n_tok, n_embd,
                     hc, ngram, kern, eps);

    // reference: gated = value*gate, normed = rmsnorm(gated, wconv)
    std::vector<float> egated((size_t)n_tok * hc_dim), enormed((size_t)n_tok * hc_dim);
    for (int t = 0; t < n_tok; t++) {
        for (int s = 0; s < hc; s++) {
            const size_t base = (size_t)t * hc_dim + (size_t)s * n_embd;
            for (int c = 0; c < n_embd; c++) egated[base + c] = value[(size_t)t * n_embd + c] * gate[(size_t)t * hc + s];
            double ss = 0;
            for (int c = 0; c < n_embd; c++) ss += (double)egated[base + c] * egated[base + c];
            const float inv = 1.0f / std::sqrt((float)(ss / n_embd) + eps);
            for (int c = 0; c < n_embd; c++) enormed[base + c] = egated[base + c] * inv * wconv[(size_t)s * n_embd + c];
        }
    }
    // conv taps: token 0 has no history (k=0 -> 0), token 1 uses token 0's row
    for (int t = 0; t < n_tok; t++) {
        for (int c = 0; c < hc_dim; c++) {
            double acc = 0;
            for (int k = 0; k < kern; k++) {
                const int d = (kern - 1 - k) * ngram;
                const int tp = poss[t] - d;
                float xv = 0;
                if (tp >= 0) {
                    const int gi = t - (poss[t] - tp);
                    xv = enormed[(size_t)gi * hc_dim + c];
                }
                acc += (double)convw[(size_t)k + (size_t)kern * c] * xv;
            }
            eres[(size_t)t * hc_dim + c] += egated[(size_t)t * hc_dim + c] + silu_((float)acc);
        }
    }
    check_f32("cpu_q4_ple_apply gate/norm", gated, egated, 1e-6, 1e-3);
    check_f32("cpu_q4_ple_apply normed", normed, enormed, 1e-6, 1e-3);
    check_f32("cpu_q4_ple_apply residual", residual, eres, 1e-5, 1e-3);
    // the newest row is appended to the history
    check_f32("cpu_q4_ple_apply history", std::vector<float>(hist.end() - hc_dim, hist.end()),
              std::vector<float>(enormed.end() - hc_dim, enormed.end()), 1e-6, 1e-3);
    const bool nh = nhist[0] == 1;
    std::printf("%-32s %s\n", "cpu_q4_ple_apply n_hist", nh ? "OK" : "FAIL");
    if (!nh) g_fail++;
}

// ---------------------------------------------------------------------------
// The composition used for a host-resident expert: one quantized activation,
// gate/up through cpu_i8_gemv1_chunked, a quantized h, and the int8 down store.
static void test_moe_experts_i8() {
    const int K = 256, n_ff = 256, G = K / 32, Gd = n_ff / 32;
    auto mk = [](int n_rows) {
        std::vector<uint8_t> p((size_t)n_rows * quant_row_bytes(GGML_TYPE_Q4_K, 256));
        auto * b = (block_q4_K *)p.data();
        for (int r = 0; r < n_rows; r++) {
            b[r].d = ggml_float_to_half(0.004f * (float)(r % 5 + 1));
            b[r].dmin = ggml_float_to_half(0.001f * (float)(r % 3 + 1));
            for (int i = 0; i < K_SCALE_SIZE; i++) b[r].scales[i] = (uint8_t)(0x11 + 7 * i + r);
            for (int i = 0; i < QK_K / 2; i++) b[r].qs[i] = (uint8_t)((r * 37 + i * 11) & 0xFF);
        }
        return p;
    };
    auto gate = mk(n_ff), up = mk(n_ff), down = mk(K);
    std::vector<uint8_t> d8((size_t)K * n_ff);
    std::vector<float> dsc((size_t)K * Gd);
    cpu_q4_convert_down(down.data(), GGML_TYPE_Q4_K, n_ff, K, 1,
                        (size_t)K * quant_row_bytes(GGML_TYPE_Q4_K, n_ff), d8.data(), dsc.data());
    cpu_moe M{};
    M.gate_exps.data = (const char *)gate.data();
    M.gate_exps.type = M.up_exps.type = GGML_TYPE_Q4_K;
    M.gate_exps.K = M.up_exps.K = K;
    M.gate_exps.N = M.up_exps.N = n_ff;
    M.gate_exps.plane_bytes = M.up_exps.plane_bytes = (size_t)n_ff * quant_row_bytes(GGML_TYPE_Q4_K, K);
    M.up_exps.data = (const char *)up.data();
    M.down_i8 = d8.data();
    M.down_sc = dsc.data();
    M.down_N = K;
    M.down_K = n_ff;
    M.n_expert = 1;
    M.n_ff_exp = n_ff;
    M.n_embd = K;
    std::vector<float> x((size_t)K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 3000);
    std::vector<float> acc((size_t)K, 0.f);
    int32_t ids[1] = {0};
    cpu_moe_experts_i8(M, x.data(), K, 1, ids, acc.data(), K);

    cpu_step_info info{};
    info.n_rows = 1;
    info.n_real = 1;
    info.tpb = 32;
    std::vector<int8_t> x8((size_t)K);
    std::vector<float> xm((size_t)G * 2);
    std::vector<int32_t> xs((size_t)G * 2);
    cpu_xq(x.data(), nullptr, K, K, x8.data(), xm.data(), xs.data(), &info, 1, K);
    std::vector<float> ubuf(n_ff), wrow(K);
    for (int n = 0; n < n_ff; n++) {
        double g = 0, u = 0;
        for (int grp = 0; grp < G; grp++) {
            dequantize_row(GGML_TYPE_Q4_K, gate.data() + (size_t)n * quant_row_bytes(GGML_TYPE_Q4_K, K), wrow.data(), K);
            for (int i = 0; i < 32; i++) g += (double)wrow[grp * 32 + i] * x8[grp * 32 + i] * xm[grp * 2];
            dequantize_row(GGML_TYPE_Q4_K, up.data() + (size_t)n * quant_row_bytes(GGML_TYPE_Q4_K, K), wrow.data(), K);
            for (int i = 0; i < 32; i++) u += (double)wrow[grp * 32 + i] * x8[grp * 32 + i] * xm[grp * 2];
        }
        ubuf[n] = silu_((float)g) * (float)u;
    }
    std::vector<int8_t> h8((size_t)n_ff);
    std::vector<float> hm((size_t)Gd * 2);
    std::vector<int32_t> hs((size_t)Gd * 2);
    cpu_xq(ubuf.data(), nullptr, n_ff, n_ff, h8.data(), hm.data(), hs.data(), &info, 1, n_ff);
    std::vector<float> exp((size_t)K, 0.f);
    for (int n = 0; n < K; n++) {
        double a = 0;
        for (int grp = 0; grp < Gd; grp++) {
            int32_t qd = 0;
            for (int i = 0; i < 32; i++) {
                qd += (int32_t)((int)d8[(size_t)n * n_ff + grp * 32 + i] - 128) * h8[grp * 32 + i];
            }
            a += (double)hm[grp * 2] * dsc[(size_t)n * Gd + grp] * qd;
        }
        exp[n] = (float)a;
    }
    check_f32("cpu_moe_experts_i8", acc, exp, 1e-5, 1e-3);
}

int main() {
    test_native_down();
    test_i8_batch();
    test_ple();
    test_moe_experts_i8();
    if (g_fail) {
        std::printf("test_cpu_q4_units: FAILURES: %d\n", g_fail);
        return 1;
    }
    std::printf("test_cpu_q4_units: all units OK\n");
    return 0;
}
