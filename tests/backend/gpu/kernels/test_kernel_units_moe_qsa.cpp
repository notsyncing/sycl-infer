// Model-free unit tests for the MoE housekeeping/scatter kernels, the MTP
// candidate readout, the QSA scratch/gather/selector kernels, and the fused
// qwen4exp hyper-connection dispatch.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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
    if (got.size() != exp.size()) return fail(name, "size");
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
static float silu_(float v) { return v / (1.0f + std::exp(-v)); }
template <typename T> static T * dev(queue & q, size_t n) { return malloc_device<T>(n, q); }
template <typename T> static void up(queue & q, T * d, const std::vector<T> & h) {
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
}
template <typename T> static void dn(queue & q, std::vector<T> & h, const T * d) {
    q.memcpy(h.data(), d, h.size() * sizeof(T)).wait();
}

// ---------------------------------------------------------------------------
static void test_moe_scatter(queue & q) {
    const int n_pairs = 4, N = 5, K = 64, ng = K / 32;
    std::vector<int32_t> pair{3, 0, 2, 1};
    int32_t * d_pair = dev<int32_t>(q, pair.size());
    up(q, d_pair, pair);
    std::vector<float> x((size_t)4 * K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i);

    // u4_gather: tile row j = x[pair[j]/top_k]; top_k = 1 -> pair[j]
    float * d_x = dev<float>(q, x.size());
    float * d_tile = dev<float>(q, (size_t)6 * K);
    up(q, d_x, x);
    moe_u4_gather_launch(q, d_x, d_pair, 0, n_pairs, 6, 1, K, d_tile);
    std::vector<float> tile((size_t)6 * K);
    q.memcpy(tile.data(), d_tile, tile.size() * 4).wait();
    std::vector<float> etile(tile.size(), 0.f);
    for (int j = 0; j < 6; j++) {
        if (j < n_pairs) {
            for (int k = 0; k < K; k++) etile[(size_t)j * K + k] = x[(size_t)pair[j] * K + k];
        }
    }
    check_f32("moe_u4_gather_launch", tile, etile, 0.0);

    // quantized tile gather (txe/txo/tcorr null; tq/tasa/txs populated)
    std::vector<int8_t> xq((size_t)4 * K);
    std::vector<uint16_t> asa((size_t)4 * ng);
    std::vector<float> xs((size_t)4 * ng);
    for (size_t i = 0; i < xq.size(); i++) xq[i] = (int8_t)((int)(i * 11) % 200 - 100);
    for (size_t i = 0; i < asa.size(); i++) asa[i] = ggml_float_to_half(0.01f * (float)(i % 5 + 1));
    for (size_t i = 0; i < xs.size(); i++) xs[i] = (float)((int)i % 17) - 8.f;
    int8_t * d_xq = dev<int8_t>(q, xq.size());
    uint16_t * d_asa = dev<uint16_t>(q, asa.size());
    float * d_xs = dev<float>(q, xs.size());
    int8_t * d_tq = dev<int8_t>(q, (size_t)6 * K);
    uint16_t * d_tasa = dev<uint16_t>(q, (size_t)6 * ng);
    float * d_txs = dev<float>(q, (size_t)6 * ng);
    uint16_t * d_tcorr = dev<uint16_t>(q, (size_t)6 * ng);
    up(q, d_xq, xq); up(q, d_asa, asa); up(q, d_xs, xs);
    moe_u4_gather_i8_launch(q, d_xq, d_asa, d_xs, d_pair, 0, n_pairs, 6, 1, K, d_tq, nullptr, nullptr, d_tasa, d_txs,
                            d_tcorr);
    std::vector<int8_t> tq((size_t)6 * K);
    std::vector<uint16_t> tasa((size_t)6 * ng), tcorr((size_t)6 * ng);
    std::vector<float> txs((size_t)6 * ng);
    q.memcpy(tq.data(), d_tq, tq.size()).wait();
    q.memcpy(tasa.data(), d_tasa, tasa.size() * 2).wait();
    q.memcpy(txs.data(), d_txs, txs.size() * 4).wait();
    q.memcpy(tcorr.data(), d_tcorr, tcorr.size() * 2).wait();
    bool gok = true;
    for (int j = 0; j < 6; j++) {
        for (int k = 0; k < K; k++) {
            const int8_t v = j < n_pairs ? xq[(size_t)pair[j] * K + k] : 0;
            gok = gok && tq[(size_t)j * K + k] == v;
        }
        for (int g = 0; g < ng; g++) {
            if (j < n_pairs) {
                gok = gok && tasa[(size_t)j * ng + g] == asa[(size_t)pair[j] * ng + g];
                gok = gok && txs[(size_t)j * ng + g] == xs[(size_t)pair[j] * ng + g];
                sycl::half h(xs[(size_t)pair[j] * ng + g] * ggml_half_to_float(asa[(size_t)pair[j] * ng + g]));
                uint16_t hb;
                std::memcpy(&hb, &h, 2);
                gok = gok && tcorr[(size_t)j * ng + g] == hb;
            }
        }
    }
    std::printf("%-32s %s\n", "moe_u4_gather_i8_launch", gok ? "OK" : "FAIL");
    if (!gok) g_fail++;

    // scatter+silu over pair rows
    std::vector<float> g((size_t)n_pairs * N), u((size_t)n_pairs * N), h((size_t)5 * N, -1.f), eh(h.size(), -1.f);
    for (size_t i = 0; i < g.size(); i++) g[i] = rnd((int)i + 400);
    for (size_t i = 0; i < u.size(); i++) u[i] = rnd((int)i + 700);
    float * d_g = dev<float>(q, g.size()), *d_u = dev<float>(q, u.size()), *d_h = dev<float>(q, h.size());
    up(q, d_g, g); up(q, d_u, u); up(q, d_h, h);
    moe_u4_scatter_silu_launch(q, d_g, d_u, d_pair, 0, n_pairs, N, d_h, N);
    q.memcpy(h.data(), d_h, h.size() * 4).wait();
    for (int j = 0; j < n_pairs; j++) {
        for (int c = 0; c < N; c++) eh[(size_t)pair[j] * N + c] = silu_(g[(size_t)j * N + c]) * u[(size_t)j * N + c];
    }
    check_f32("moe_u4_scatter_silu", h, eh, 1e-6, 1e-3);

    // fused gate+up silu, linear and scattered
    const int rows = 3;
    std::vector<float> gu((size_t)rows * 2 * N), hf((size_t)rows * N, -1.f), ehf(hf.size(), -1.f);
    for (size_t i = 0; i < gu.size(); i++) gu[i] = rnd((int)i + 1000);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < N; c++) ehf[(size_t)r * N + c] = silu_(gu[(size_t)r * 2 * N + c]) * gu[(size_t)r * 2 * N + N + c];
    }
    float * d_gu = dev<float>(q, gu.size()), *d_hf = dev<float>(q, hf.size());
    up(q, d_gu, gu); up(q, d_hf, hf);
    moe_u4_silu_fused_launch(q, d_gu, rows, N, d_hf, N);
    q.memcpy(hf.data(), d_hf, hf.size() * 4).wait();
    check_f32("moe_u4_silu_fused", hf, ehf, 1e-6, 1e-3);

    std::vector<float> hs((size_t)5 * N, -1.f), ehs(hs.size(), -1.f);
    int32_t * d_rows = dev<int32_t>(q, rows);
    up(q, d_rows, std::vector<int32_t>{0, 2, 4});
    float * d_hs = dev<float>(q, hs.size());
    up(q, d_hs, hs);
    moe_u4_silu_fused_scatter_launch(q, d_gu, d_rows, rows, N, d_hs, N);
    q.memcpy(hs.data(), d_hs, hs.size() * 4).wait();
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < N; c++) ehs[(size_t)(r * 2) * N + c] = ehf[(size_t)r * N + c];
    }
    check_f32("moe_u4_silu_fused_scatter", hs, ehs, 1e-6, 1e-3);

    std::vector<float> hp((size_t)n_pairs * N), ehp(hp.size());
    float * d_hp = dev<float>(q, hp.size());
    moe_u4_silu_pair_launch(q, d_g, d_u, n_pairs, N, d_hp);
    q.memcpy(hp.data(), d_hp, hp.size() * 4).wait();
    for (size_t i = 0; i < hp.size(); i++) ehp[i] = silu_(g[i]) * u[i];
    check_f32("moe_u4_silu_pair", hp, ehp, 1e-6, 1e-3);

    // csr gather/corr: the pair->row map is the only difference.
    int8_t * d_cq = dev<int8_t>(q, (size_t)n_pairs * K);
    uint16_t * d_ca = dev<uint16_t>(q, (size_t)n_pairs * ng);
    float * d_cx = dev<float>(q, (size_t)n_pairs * ng);
    uint16_t * d_cc = dev<uint16_t>(q, (size_t)n_pairs * ng);
    moe_csr_gather_i8_launch(q, d_xq, d_asa, d_xs, d_pair, n_pairs, 1, K, d_cq, d_ca, d_cx, d_cc);
    std::vector<int8_t> cq((size_t)n_pairs * K);
    q.memcpy(cq.data(), d_cq, cq.size()).wait();
    bool cok = true;
    for (int p = 0; p < n_pairs; p++) {
        for (int k = 0; k < K; k++) cok = cok && cq[(size_t)p * K + k] == xq[(size_t)pair[p] * K + k];
    }
    std::printf("%-32s %s\n", "moe_csr_gather_i8", cok ? "OK" : "FAIL");
    if (!cok) g_fail++;

    std::vector<uint16_t> corr((size_t)4 * ng), ecorr((size_t)4 * ng);
    uint16_t * d_co = dev<uint16_t>(q, corr.size());
    moe_csr_corr_launch(q, d_asa, d_xs, d_co, 4, ng);
    q.memcpy(corr.data(), d_co, corr.size() * 2).wait();
    for (size_t i = 0; i < corr.size(); i++) {
        sycl::half h(xs[i] * ggml_half_to_float(asa[i]));
        std::memcpy(&ecorr[i], &h, 2);
    }
    bool rok = corr == ecorr;
    std::printf("%-32s %s\n", "moe_csr_corr", rok ? "OK" : "FAIL");
    if (!rok) g_fail++;

    free(d_pair, q); free(d_x, q); free(d_tile, q); free(d_xq, q); free(d_asa, q); free(d_xs, q);
    free(d_tq, q); free(d_tasa, q); free(d_txs, q); free(d_tcorr, q); free(d_g, q); free(d_u, q);
    free(d_h, q); free(d_gu, q); free(d_hf, q); free(d_rows, q); free(d_hs, q); free(d_hp, q);
    free(d_cq, q); free(d_ca, q); free(d_cx, q); free(d_cc, q); free(d_co, q);
}

// ---------------------------------------------------------------------------
static void test_moe_down_corr(queue & q) {
    const int K = 64, N = 5, ng = K / 32, E = 3;
    const size_t vb = (size_t)N * K, sb = (size_t)ng * N * 2, plane = vb + 2 * sb;
    std::vector<uint8_t> bank((size_t)E * plane);
    std::vector<int32_t> off{0, 2, 3, 4}, active{0, 1, 2};
    std::vector<uint16_t> asa((size_t)4 * ng);
    std::vector<float> xs((size_t)4 * ng);
    for (int e = 0; e < E; e++) {
        std::vector<uint16_t> op((size_t)ng * N);
        for (size_t i = 0; i < op.size(); i++) op[i] = ggml_float_to_half(0.01f * (float)((i + e) % 7 + 1));
        std::memcpy(bank.data() + (size_t)e * plane + vb + sb, op.data(), op.size() * 2);
    }
    for (size_t i = 0; i < asa.size(); i++) asa[i] = ggml_float_to_half(0.02f * (float)(i % 4 + 1));
    for (size_t i = 0; i < xs.size(); i++) xs[i] = (float)((int)i % 11) - 5.f;
    uint8_t * d_bank = dev<uint8_t>(q, bank.size());
    int32_t * d_off = dev<int32_t>(q, off.size()), *d_act = dev<int32_t>(q, active.size()),
            *d_nact = dev<int32_t>(q, 1);
    uint16_t * d_asa = dev<uint16_t>(q, asa.size());
    float * d_xs = dev<float>(q, xs.size()), *d_acc = dev<float>(q, (size_t)4 * N);
    up(q, d_bank, bank); up(q, d_off, off); up(q, d_act, active);
    up(q, d_nact, std::vector<int32_t>{3}); up(q, d_asa, asa); up(q, d_xs, xs);
    std::vector<float> acc((size_t)4 * N, 0.f);
    up(q, d_acc, acc);
    moe_down_corr_add_launch(q, d_bank, plane, K, N, d_off, d_act, d_nact, 3, d_asa, d_xs, d_acc, N);
    q.memcpy(acc.data(), d_acc, acc.size() * 4).wait();
    std::vector<float> exp(acc.size(), 0.f);
    for (int p = 0; p < 4; p++) {
        const int e = (p < 2) ? 0 : (p == 2 ? 1 : 2);
        for (int n = 0; n < N; n++) {
            float s = 0;
            for (int g = 0; g < ng; g++) {
                const uint16_t hb = *(const uint16_t *)(bank.data() + (size_t)e * plane + vb + sb +
                                                        ((size_t)g * N + n) * 2);
                const float offv = ggml_half_to_float(hb);
                const float cg = (float)(sycl::half)(ggml_half_to_float(asa[(size_t)p * ng + g]) * xs[(size_t)p * ng + g]);
                s = std::fma(offv, cg, s);
            }
            exp[(size_t)p * N + n] += s;
        }
    }
    check_f32("moe_down_corr_add", acc, exp, 1e-6, 1e-3);
    free(d_bank, q); free(d_off, q); free(d_act, q); free(d_nact, q); free(d_asa, q); free(d_xs, q); free(d_acc, q);
}

// ---------------------------------------------------------------------------
static void test_mtp(queue & q) {
    const int n_embd = 256, n_rows = 2, tpb = 32, n_real = 2;
    const float eps = 1e-6f;
    const int row_bytes = n_embd * 4;
    std::vector<float> table((size_t)8 * n_embd), enorm(n_embd), hnorm(n_embd), h((size_t)2 * tpb * n_embd),
        hprev((size_t)n_rows * n_embd);
    for (size_t i = 0; i < table.size(); i++) table[i] = rnd((int)i);
    for (int i = 0; i < n_embd; i++) {
        enorm[i] = 0.5f + 0.1f * (i % 3);
        hnorm[i] = 0.7f + 0.05f * (i % 5);
    }
    for (size_t i = 0; i < h.size(); i++) h[i] = rnd((int)i + 500);
    for (size_t i = 0; i < hprev.size(); i++) hprev[i] = rnd((int)i + 900);
    float * d_tab = dev<float>(q, table.size()), *d_en = dev<float>(q, n_embd), *d_hn = dev<float>(q, n_embd);
    float * d_h = dev<float>(q, h.size()), *d_hp = dev<float>(q, hprev.size());
    float * d_out = dev<float>(q, (size_t)n_rows * tpb * 2 * n_embd);
    up(q, d_tab, table); up(q, d_en, enorm); up(q, d_hn, hnorm); up(q, d_h, h); up(q, d_hp, hprev);
    step_info info{};
    info.n_rows = n_rows;
    info.tpb = tpb;
    info.n_real = n_real;
    for (int r = 0; r < n_rows; r++) {
        info.active[r] = 1;
        info.n_real_row[r] = n_real;
        info.pos[r] = 0;
        for (int t = 0; t < tpb; t++) info.tokens[r * tpb + t] = (r * 3 + t) % 8;
    }
    // step_info is host USM in the engine: mtp_concat reads info->tpb on the
    // host to size its grid, so a device-only allocation faults in the launcher.
    step_info * d_info = malloc_host<step_info>(1, q);
    *d_info = info;
    mtp_concat_launch(q, d_tab, 0 /* F32 */, (size_t)row_bytes, d_en, d_hn, d_h, d_hp, d_info, d_out, n_embd, eps,
                      nullptr);
    std::vector<float> out((size_t)n_rows * tpb * 2 * n_embd);
    q.memcpy(out.data(), d_out, out.size() * 4).wait();
    std::vector<float> eout(out.size(), 0.f);
    for (int r = 0; r < n_rows; r++) {
        for (int t = 0; t < n_real; t++) {
            const int slot = r * tpb + t;
            const int tok = info.tokens[slot];
            const float * ev = table.data() + (size_t)tok * n_embd;
            const float * hv = (slot == 0) ? hprev.data() + (size_t)r * n_embd : h.data() + (size_t)(slot - 1) * n_embd;
            double se = 0, sh = 0;
            for (int i = 0; i < n_embd; i++) {
                se += (double)ev[i] * ev[i];
                sh += (double)hv[i] * hv[i];
            }
            const float re = 1.0f / std::sqrt((float)(se / n_embd) + eps);
            const float rh = 1.0f / std::sqrt((float)(sh / n_embd) + eps);
            for (int i = 0; i < n_embd; i++) {
                eout[(size_t)slot * 2 * n_embd + i] = ev[i] * re * enorm[i];
                eout[(size_t)slot * 2 * n_embd + n_embd + i] = hv[i] * rh * hnorm[i];
            }
        }
    }
    check_f32("mtp_concat_launch", out, eout, 1e-5, 1e-3);

    // candidate set: the argmax plus every logit above (argmax - margin).
    const int n = 20, cap = 8;
    std::vector<float> logits(n);
    for (int i = 0; i < n; i++) logits[i] = -2.f - 0.1f * i;
    logits[7] = 3.0f;
    logits[11] = 2.4f;
    logits[3] = 2.2f;
    float * d_l = dev<float>(q, n);
    int32_t * d_am_i = dev<int32_t>(q, 1);
    float * d_am_v = dev<float>(q, 1);
    int32_t * d_ids = dev<int32_t>(q, cap);
    up(q, d_l, logits); up(q, d_am_i, std::vector<int32_t>{7}); up(q, d_am_v, std::vector<float>{3.0f});
    mtp_cand_launch(q, d_l, n, d_am_i, d_am_v, 0.5f, d_ids, cap);
    std::vector<int32_t> ids(cap);
    q.memcpy(ids.data(), d_ids, cap * 4).wait();
    bool cok = ids[0] == 7;
    // every emitted candidate must be above threshold (except the -1 pad)
    for (int i = 1; i < cap; i++) {
        if (ids[i] >= 0) cok = cok && logits[ids[i]] > 2.5f;
    }
    std::printf("%-32s %s\n", "mtp_cand_launch", cok ? "OK" : "FAIL");
    if (!cok) g_fail++;

    // gather: the same int8 dot for gathered rows, then its argmax.
    const int K = 256, N = 16, ng = K / 32;
    std::vector<int8_t> w8((size_t)N * K);
    std::vector<uint16_t> wsc((size_t)ng * N);
    std::vector<float> xx((size_t)K), candv(cap, 0.f);
    for (size_t i = 0; i < w8.size(); i++) w8[i] = (int8_t)((int)(i * 7) % 255 - 127);
    for (size_t i = 0; i < wsc.size(); i++) wsc[i] = ggml_float_to_half(0.002f * (float)(i % 5 + 1));
    for (size_t i = 0; i < xx.size(); i++) xx[i] = rnd((int)i + 1500);
    std::vector<int8_t> xq((size_t)K);
    std::vector<uint16_t> asa((size_t)ng);
    std::vector<float> xs((size_t)ng);
    for (int g = 0; g < ng; g++) {
        float mx = 0;
        for (int i = 0; i < 32; i++) mx = std::max(mx, std::fabs(xx[g * 32 + i]));
        const float sc = mx > 0 ? mx / 127.f : 1.f;
        asa[g] = ggml_float_to_half(sc);
        float sum = 0;
        for (int i = 0; i < 32; i++) {
            const int qv = (int)std::max(-127.f, std::min(127.f, std::round(xx[g * 32 + i] / sc)));
            xq[g * 32 + i] = (int8_t)qv;
            sum += qv;
        }
        xs[g] = sum;
    }
    std::vector<int32_t> gids{0, 5, 9, -1, 15, 3, 11, 2};
    int8_t * d_w8 = dev<int8_t>(q, w8.size());
    uint16_t * d_wsc = dev<uint16_t>(q, wsc.size());
    int8_t * d_xq = dev<int8_t>(q, K);
    uint16_t * d_asa = dev<uint16_t>(q, ng);
    float * d_xsg = dev<float>(q, ng);
    int32_t * d_gid = dev<int32_t>(q, cap);
    float * d_val = dev<float>(q, cap);
    up(q, d_w8, w8); up(q, d_wsc, wsc); up(q, d_xq, xq); up(q, d_asa, asa); up(q, d_xsg, xs); up(q, d_gid, gids);
    mtp_gather_launch(q, d_gid, cap, d_w8, d_wsc, d_xq, d_asa, d_xsg, K, N, d_val);
    std::vector<float> vals(cap);
    q.memcpy(vals.data(), d_val, cap * 4).wait();
    std::vector<float> ev(cap, 0.f);
    for (int c = 0; c < cap; c++) {
        if (gids[c] < 0) {
            ev[c] = -INFINITY;
            continue;
        }
        double s = 0;
        for (int g = 0; g < ng; g++) {
            int32_t qd = 0;
            for (int i = 0; i < 32; i++) qd += (int32_t)w8[(size_t)gids[c] * K + g * 32 + i] * xq[g * 32 + i];
            s += (double)ggml_half_to_float(asa[g]) * ggml_half_to_float(wsc[(size_t)g * N + gids[c]]) * qd;
        }
        ev[c] = (float)s;
    }
    // -inf compares equal; check the finite entries and the -1 entry
    bool vok = vals[3] < -1e30f;
    for (int c = 0; c < cap; c++) {
        if (gids[c] >= 0) vok = vok && std::fabs(vals[c] - ev[c]) < 1e-4f * std::max(1.f, std::fabs(ev[c]));
    }
    std::printf("%-32s %s\n", "mtp_gather_launch", vok ? "OK" : "FAIL");
    if (!vok) g_fail++;

    int32_t * d_aid = dev<int32_t>(q, 1);
    mtp_gather_argmax_launch(q, d_val, d_gid, cap, d_aid, nullptr);
    int32_t aidx = -1;
    q.memcpy(&aidx, d_aid, 4).wait();
    int best = -1;
    float bv = -INFINITY;
    for (int c = 0; c < cap; c++) {
        if (ev[c] > bv || (ev[c] == bv && gids[c] < best)) {
            bv = ev[c];
            best = gids[c];
        }
    }
    const bool aok = aidx == best;
    std::printf("%-32s got=%d exp=%d %s\n", "mtp_gather_argmax", aidx, best, aok ? "OK" : "FAIL");
    if (!aok) g_fail++;

    free(d_tab, q); free(d_en, q); free(d_hn, q); free(d_h, q); free(d_hp, q); free(d_out, q); free(d_info, q);
    free(d_l, q); free(d_am_i, q); free(d_am_v, q); free(d_ids, q); free(d_w8, q); free(d_wsc, q);
    free(d_xq, q); free(d_asa, q); free(d_xsg, q); free(d_gid, q); free(d_val, q); free(d_aid, q);
}

// ---------------------------------------------------------------------------
static void test_qsa_indexer(queue & q) {
    const int K = 64, id = 32, inh = 2, n_rot = 16;
    const int qrows = inh * id;
    std::vector<uint16_t> qw((size_t)qrows * K), kw((size_t)id * K);
    std::vector<float> x((size_t)K), qn(id), qout(qrows), kout(id);
    for (size_t i = 0; i < qw.size(); i++) {
        const float v = rnd((int)i + 3000) * 0.1f;
        uint32_t b;
        std::memcpy(&b, &v, 4);
        qw[i] = (uint16_t)(b >> 16);
    }
    for (size_t i = 0; i < kw.size(); i++) {
        const float v = rnd((int)i + 5000) * 0.1f;
        uint32_t b;
        std::memcpy(&b, &v, 4);
        kw[i] = (uint16_t)(b >> 16);
    }
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 7000);
    for (int i = 0; i < id; i++) qn[i] = 0.5f + 0.1f * (i % 7);
    uint16_t * d_qw = dev<uint16_t>(q, qw.size()), *d_kw = dev<uint16_t>(q, kw.size());
    float * d_x = dev<float>(q, K), *d_qn = dev<float>(q, id);
    float * d_qo = dev<float>(q, qrows), *d_ko = dev<float>(q, id);
    up(q, d_qw, qw); up(q, d_kw, kw); up(q, d_x, x); up(q, d_qn, qn);
    const float base = 10000.f, eps = 1e-6f;
    qsa_indexer_launch(q, d_x, d_qw, d_kw, K, inh, id, n_rot, base, 3, d_qn, d_qo, d_ko, eps);
    dn(q, qout, d_qo);
    dn(q, kout, d_ko);

    auto dot = [&](const uint16_t * w, int row) {
        double s = 0;
        for (int k = 0; k < K; k++) s += (double)ggml_bf16_to_float(w[(size_t)row * K + k]) * x[k];
        return (float)s;
    };
    std::vector<float> ek(id), eq(qrows);
    for (int r = 0; r < id; r++) ek[r] = dot(kw.data(), r);
    for (int h = 0; h < inh; h++) {
        float * p = &eq[(size_t)h * id];
        for (int i = 0; i < id; i++) p[i] = dot(qw.data(), h * id + i);
        double ss = 0;
        for (int i = 0; i < id; i++) ss += (double)p[i] * p[i];
        const float inv = 1.0f / std::sqrt((float)(ss / id) + eps);
        for (int i = 0; i < id; i++) p[i] *= inv * qn[i];
        const int half = n_rot / 2;
        for (int i = 0; i < half; i++) {
            const float ang = 3.f * std::exp2(-2.f * i / (float)n_rot * std::log2(base));
            const float c = std::cos(ang), sn = std::sin(ang);
            const float x0 = p[i], x1 = p[i + half];
            p[i] = x0 * c - x1 * sn;
            p[i + half] = x0 * sn + x1 * c;
        }
    }
    check_f32("qsa_indexer q", qout, eq, 1e-5, 1e-3);
    check_f32("qsa_indexer k", kout, ek, 1e-5, 1e-3);
    free(d_qw, q); free(d_kw, q); free(d_x, q); free(d_qn, q); free(d_qo, q); free(d_ko, q);
}

static void test_qsa_idx_update(queue & q) {
    const int id = 32, kp = 2, n_rot = 16;
    const float base = 10000.f, eps = 1e-6f;
    std::vector<float> k0(id), k1(id), kn(id);
    for (int i = 0; i < id; i++) {
        k0[i] = rnd(i + 100);
        k1[i] = rnd(i + 300);
        kn[i] = 0.6f + 0.1f * (i % 5);
    }
    float * d_ring = dev<float>(q, (size_t)kp * id), *d_pool = dev<float>(q, id);
    float * d_k0 = dev<float>(q, id), *d_k1 = dev<float>(q, id), *d_kn = dev<float>(q, id);
    up(q, d_k0, k0); up(q, d_k1, k1); up(q, d_kn, kn);
    q.memset(d_ring, 0, (size_t)kp * id * 4);
    q.memset(d_pool, 0, (size_t)id * 4);
    qsa_idx_update_launch(q, d_k0, 0, d_ring, d_pool, d_kn, id, kp, n_rot, base, eps);
    qsa_idx_update_launch(q, d_k1, 1, d_ring, d_pool, d_kn, id, kp, n_rot, base, eps);
    std::vector<float> ring((size_t)kp * id), pool(id);
    dn(q, ring, d_ring);
    dn(q, pool, d_pool);
    std::vector<float> ering((size_t)kp * id), epool(id);
    for (int t = 0; t < kp; t++) {
        const std::vector<float> & src = t == 0 ? k0 : k1;
        for (int i = 0; i < id; i++) ering[(size_t)t * id + i] = ggml_half_to_float(ggml_float_to_half(src[i]));
    }
    for (int i = 0; i < id; i++) epool[i] = 0.5f * (ering[i] + ering[id + i]);
    double ss = 0;
    for (int i = 0; i < id; i++) ss += (double)epool[i] * epool[i];
    const float inv = 1.0f / std::sqrt((float)(ss / id) + eps);
    for (int i = 0; i < id; i++) epool[i] *= inv * kn[i];
    const int half = n_rot / 2;
    for (int i = 0; i < half; i++) {
        const float ang = 0.f; // pos - (kp-1) = 0
        const float c = std::cos(ang), sn = std::sin(ang);
        const float x0 = epool[i], x1 = epool[i + half];
        epool[i] = x0 * c - x1 * sn;
        epool[i + half] = x0 * sn + x1 * c;
    }
    for (int i = 0; i < id; i++) epool[i] = ggml_half_to_float(ggml_float_to_half(epool[i]));
    check_f32("qsa_idx_update ring", ring, ering, 0.0);
    check_f32("qsa_idx_update pool", pool, epool, 1e-6, 1e-3);
    free(d_ring, q); free(d_pool, q); free(d_k0, q); free(d_k1, q); free(d_kn, q);
}

// ---------------------------------------------------------------------------
static void test_qsa(queue & q) {
    // scratch step_info
    step_info * d_info = dev<step_info>(q, 1);
    step_info zi{};
    up(q, d_info, {zi});
    qsa_sinfo_launch(q, d_info, 9);
    step_info oi{};
    q.memcpy(&oi, d_info, sizeof(oi)).wait();
    bool sok = oi.n_rows == 1 && oi.n_real == 1 && oi.tpb == 1 && oi.pos[0] == 8 && oi.slot[0] == 0;
    std::printf("%-32s %s\n", "qsa_sinfo_launch", sok ? "OK" : "FAIL");
    if (!sok) g_fail++;
    std::vector<int32_t> counts{3, 5};
    int32_t * d_counts = dev<int32_t>(q, counts.size());
    up(q, d_counts, counts);
    up(q, d_info, {zi});
    qsa_sinfo_multi_launch(q, d_info, d_counts, 2);
    q.memcpy(&oi, d_info, sizeof(oi)).wait();
    bool mok = oi.n_rows == 2 && oi.pos[0] == 2 && oi.pos[1] == 4 && oi.slot[0] == 0 && oi.slot[1] == 1;
    std::printf("%-32s %s\n", "qsa_sinfo_multi_launch", mok ? "OK" : "FAIL");
    if (!mok) g_fail++;

    // gather one row of nsel keys from a paged f16 pool.
    const int n_kv = 2, hd = 8, bs = 32, nsel = 5;
    std::vector<uint16_t> kpool((size_t)2 * n_kv * bs * hd), vpool(kpool.size());
    for (size_t i = 0; i < kpool.size(); i++) {
        kpool[i] = ggml_float_to_half(rnd((int)i) * 0.1f);
        vpool[i] = ggml_float_to_half(rnd((int)i + 500) * 0.1f);
    }
    int32_t tables[2] = {0, 1};
    std::vector<int32_t> sel{0, 3, 31, 32, 40};
    uint16_t * d_kp = dev<uint16_t>(q, kpool.size()), *d_vp = dev<uint16_t>(q, vpool.size());
    int32_t * d_tab = dev<int32_t>(q, 2), *d_gather_sel = dev<int32_t>(q, nsel);
    // the compact destination is a paged pool: one block of `bs` keys per kv
    // head, not nsel keys packed tightly.
    const int nblk_g = (nsel + bs - 1) / bs;
    const size_t csize = (size_t)nblk_g * n_kv * bs * hd;
    uint16_t * d_ck = dev<uint16_t>(q, csize), *d_cv = dev<uint16_t>(q, csize);
    up(q, d_kp, kpool); up(q, d_vp, vpool); up(q, d_tab, std::vector<int32_t>{0, 1}); up(q, d_gather_sel, sel);
    qsa_gather_launch(q, d_kp, d_vp, d_tab, d_gather_sel, nsel, n_kv, hd, bs, d_ck, d_cv);
    std::vector<uint16_t> ck(csize), cv(ck.size());
    q.memcpy(ck.data(), d_ck, ck.size() * 2).wait();
    q.memcpy(cv.data(), d_cv, cv.size() * 2).wait();
    bool gok = true;
    for (int j = 0; j < nsel; j++) {
        const int sp = sel[j];
        const int sb = tables[sp / bs], so = sp % bs;
        for (int kh = 0; kh < n_kv; kh++) {
            for (int d = 0; d < hd; d++) {
                const size_t src = ((size_t)sb * n_kv + kh) * bs * hd + (size_t)so * hd + d;
                const size_t dst = ((size_t)(j / bs) * n_kv + kh) * bs * hd + (size_t)(j % bs) * hd + d;
                gok = gok && ck[dst] == kpool[src] && cv[dst] == vpool[src];
            }
        }
    }
    std::printf("%-32s %s\n", "qsa_gather_launch", gok ? "OK" : "FAIL");
    if (!gok) g_fail++;

    // multi-row gather (this path asserts head_dim 256 and block 32).
    const int n_rows = 2, cap_blocks = 1, hd_r = 256, n_kv_r = 1;
    std::vector<uint16_t> kpool_r((size_t)2 * n_kv_r * bs * hd_r), vpool_r(kpool_r.size());
    for (size_t i = 0; i < kpool_r.size(); i++) {
        kpool_r[i] = ggml_float_to_half(rnd((int)i + 3000) * 0.1f);
        vpool_r[i] = ggml_float_to_half(rnd((int)i + 3500) * 0.1f);
    }
    std::vector<int32_t> gsel((size_t)n_rows * n_kv_r * bs), gcounts{3, 2};
    for (int r = 0; r < n_rows; r++) {
        for (int j = 0; j < n_kv_r * bs; j++) gsel[(size_t)r * n_kv_r * bs + j] = r * 3 + j;
    }
    uint16_t * d_kp_r = dev<uint16_t>(q, kpool_r.size()), *d_vp_r = dev<uint16_t>(q, vpool_r.size());
    int32_t * d_gsel = dev<int32_t>(q, gsel.size()), *d_gc = dev<int32_t>(q, 2);
    uint16_t * d_gck = dev<uint16_t>(q, (size_t)n_rows * cap_blocks * n_kv_r * bs * hd_r);
    uint16_t * d_gcv = dev<uint16_t>(q, (size_t)n_rows * cap_blocks * n_kv_r * bs * hd_r);
    up(q, d_kp_r, kpool_r); up(q, d_vp_r, vpool_r); up(q, d_gsel, gsel); up(q, d_gc, gcounts);
    qsa_gather_rows_launch(q, d_kp_r, d_vp_r, d_tab, d_gsel, d_gc, n_kv_r * bs, cap_blocks, n_rows, n_kv_r, hd_r, bs,
                           d_gck, d_gcv);
    std::vector<uint16_t> gck((size_t)n_rows * cap_blocks * n_kv_r * bs * hd_r);
    q.memcpy(gck.data(), d_gck, gck.size() * 2).wait();
    bool rok = true;
    for (int r = 0; r < n_rows; r++) {
        for (int j = 0; j < gcounts[r]; j++) {
            const int sp = gsel[(size_t)r * n_kv_r * bs + j];
            const int sb = tables[sp / bs], so = sp % bs;
            for (int kh = 0; kh < n_kv_r; kh++) {
                for (int d = 0; d < hd_r; d++) {
                    const size_t src = ((size_t)sb * n_kv_r + kh) * bs * hd_r + (size_t)so * hd_r + d;
                    const size_t dst = (((size_t)r * cap_blocks + j / bs) * n_kv_r + kh) * bs * hd_r +
                                       (size_t)(j % bs) * hd_r + d;
                    rok = rok && gck[dst] == kpool_r[src];
                }
            }
        }
    }
    std::printf("%-32s %s\n", "qsa_gather_rows_launch", rok ? "OK" : "FAIL");
    if (!rok) g_fail++;

    // selector scores: the same sum of positive head dots as the host kernel.
    const int id = 6, inh = 3, n_pool = 7;
    std::vector<float> qa((size_t)inh * id), pool((size_t)n_pool * id), sc(n_pool), esc(n_pool);
    for (size_t i = 0; i < qa.size(); i++) qa[i] = rnd((int)i + 2000);
    for (size_t i = 0; i < pool.size(); i++) pool[i] = rnd((int)i + 2300);
    float * d_qa = dev<float>(q, qa.size()), *d_pool = dev<float>(q, pool.size()), *d_sc = dev<float>(q, n_pool);
    up(q, d_qa, qa); up(q, d_pool, pool);
    qsa_score_launch(q, d_qa, d_pool, n_pool, id, inh, 0.7f, d_sc);
    q.memcpy(sc.data(), d_sc, n_pool * 4).wait();
    for (int p = 0; p < n_pool; p++) {
        double s = 0;
        for (int h = 0; h < inh; h++) {
            double d = 0;
            for (int k = 0; k < id; k++) d += (double)qa[(size_t)h * id + k] * pool[(size_t)p * id + k];
            s += d > 0 ? d : 0;
        }
        esc[p] = (float)s * 0.7f;
    }
    check_f32("qsa_score_launch", sc, esc, 1e-6, 1e-3);

    // top-k selection and the sel expansion.
    const int np = 12, ntop = 4, kp = 2, nvis = 17;
    std::vector<float> ss(np);
    for (int i = 0; i < np; i++) ss[i] = (i == 3 || i == 8 || i == 1 || i == 6) ? 5.f - 0.1f * i : (float)i * 0.1f;
    float * d_scores = dev<float>(q, np);
    int32_t * d_pools = dev<int32_t>(q, 512), *d_sel = dev<int32_t>(q, 512);
    up(q, d_scores, ss);
    qsa_topk_launch(q, d_scores, np, ntop, kp, nvis, d_pools, d_sel);
    std::vector<int32_t> pools(ntop), selout((size_t)ntop * kp + (nvis - (nvis / kp) * kp));
    q.memcpy(pools.data(), d_pools, ntop * 4).wait();
    q.memcpy(selout.data(), d_sel, selout.size() * 4).wait();
    std::vector<int> order(np);
    for (int i = 0; i < np; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (ss[a] != ss[b]) return ss[a] > ss[b];
        return a < b;
    });
    std::vector<int32_t> epools(order.begin(), order.begin() + ntop);
    std::sort(epools.begin(), epools.end());
    if (pools == epools) {
        bool eok = true;
        const int npool = nvis / kp;
        for (size_t j = 0; j < selout.size(); j++) {
            const int32_t e = j < (size_t)ntop * kp ? epools[j / kp] * kp + (int)(j % kp)
                                                     : npool * kp + (int)(j - (size_t)ntop * kp);
            eok = eok && selout[j] == e;
        }
        std::printf("%-32s %s\n", "qsa_topk_launch", eok ? "OK" : "FAIL");
        if (!eok) g_fail++;
    } else {
        fail("qsa_topk_launch", "pool set");
    }

    free(d_info, q); free(d_counts, q); free(d_kp, q); free(d_vp, q); free(d_tab, q); free(d_gather_sel, q);
    free(d_ck, q); free(d_cv, q); free(d_kp_r, q); free(d_vp_r, q); free(d_gsel, q); free(d_gc, q);
    free(d_gck, q); free(d_gcv, q);
    free(d_qa, q); free(d_pool, q); free(d_sc, q); free(d_scores, q); free(d_pools, q);
}

int main() {
    try {
        queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
        std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        test_moe_scatter(q);
        test_moe_down_corr(q);
        test_mtp(q);
        test_qsa(q);
        test_qsa_indexer(q);
        test_qsa_idx_update(q);
    } catch (const sycl::exception & e) {
        std::printf("SYCL exception: %s\n", e.what());
        return 1;
    }
    if (g_fail) {
        std::printf("test_kernel_units_moe_qsa: FAILURES: %d\n", g_fail);
        return 1;
    }
    std::printf("test_kernel_units_moe_qsa: all units OK\n");
    return 0;
}
