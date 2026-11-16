// Model-free unit tests for the routed-expert bank GEMVs, the grouped expert
// GEMMs, the fused u4 quantize+GEMV, the mixed Q8_0/F32 dispatch, and the KV
// checksum accessors.
//
// The banks are synthetic and the references decode the same store definition
// in scalar C++; no GGUF and no engine are involved.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kernels.h"
#include "quant.h"
#include "w4.h"
#include "cpu_types.h"

using namespace si;
using namespace sycl;

static int g_fail = 0;
static void fail(const char * name, const char * what) {
    std::printf("%-34s FAIL (%s)\n", name, what);
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
    std::printf("%-34s n=%-6zu max|exp|=%.5f max|diff|=%.3e %s\n", name, got.size(), maxv, maxd, ok ? "OK" : "FAIL");
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

struct act {
    std::vector<int8_t> xq;
    std::vector<uint16_t> asa;
    std::vector<float> xs;
};
static act make_act(const std::vector<float> & x, int K) {
    const int ng = K / 32;
    act a;
    a.xq.resize((size_t)K);
    a.asa.resize((size_t)ng);
    a.xs.assign((size_t)ng, 0.f);
    for (int g = 0; g < ng; g++) {
        float mx = 0;
        for (int i = 0; i < 32; i++) mx = std::max(mx, std::fabs(x[(size_t)g * 32 + i]));
        const float sc = mx > 0 ? mx / 127.0f : 1.0f;
        a.asa[g] = ggml_float_to_half(sc);
        float sum = 0;
        for (int i = 0; i < 32; i++) {
            const int q = (int)std::max(-127.f, std::min(127.f, std::round(x[(size_t)g * 32 + i] / sc)));
            a.xq[(size_t)g * 32 + i] = (int8_t)q;
            sum += q;
        }
        a.xs[g] = sum;
    }
    return a;
}

static std::vector<block_q4_K> make_q4k(int N, int K) {
    std::vector<block_q4_K> b((size_t)N * (K / 256));
    for (size_t n = 0; n < b.size(); n++) {
        b[n].d = ggml_float_to_half(0.004f * (float)((n % 5) + 1));
        b[n].dmin = ggml_float_to_half(0.001f * (float)((n % 3) + 1));
        for (int i = 0; i < K_SCALE_SIZE; i++) b[n].scales[i] = (uint8_t)(0x11 + 7 * i + n);
        for (int i = 0; i < QK_K / 2; i++) b[n].qs[i] = (uint8_t)((n * 37 + i * 11) & 0xFF);
    }
    return b;
}

// ---------------------------------------------------------------------------
static void test_i8_down(queue & q) {
    const int K = 256, N = 8, ng = K / 32;
    std::vector<int8_t> w((size_t)N * K);
    std::vector<uint16_t> step((size_t)ng * N), off((size_t)ng * N);
    for (size_t i = 0; i < w.size(); i++) w[i] = (int8_t)((int)(i * 13) % 255 - 127);
    for (int g = 0; g < ng; g++) {
        for (int n = 0; n < N; n++) {
            step[(size_t)g * N + n] = ggml_float_to_half(0.002f * (float)((g + n) % 5 + 1));
            off[(size_t)g * N + n] = ggml_float_to_half(-0.0005f * (float)((g + 2 * n) % 3));
        }
    }
    std::vector<uint8_t> bank((size_t)N * K + 2 * ng * N * 2);
    std::memcpy(bank.data(), w.data(), w.size());
    std::memcpy(bank.data() + (size_t)N * K, step.data(), step.size() * 2);
    std::memcpy(bank.data() + (size_t)N * K + (size_t)ng * N * 2, off.data(), off.size() * 2);
    std::vector<float> x((size_t)K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 100);
    act a = make_act(x, K);
    uint8_t * d_bank = dev<uint8_t>(q, bank.size());
    int32_t * d_ids = dev<int32_t>(q, 1);
    int8_t * d_xq = dev<int8_t>(q, K);
    uint16_t * d_asa = dev<uint16_t>(q, ng);
    float * d_xs = dev<float>(q, ng), *d_out = dev<float>(q, (size_t)N);
    up(q, d_bank, bank); up(q, d_ids, std::vector<int32_t>{0}); up(q, d_xq, a.xq); up(q, d_asa, a.asa); up(q, d_xs, a.xs);
    moe_i8_gemv_down_launch(q, d_bank, bank.size(), d_ids, 1, d_xq, d_asa, d_xs, K, N, d_out, N);
    std::vector<float> got(N), exp(N, 0.f);
    dn(q, got, d_out);
    for (int n = 0; n < N; n++) {
        double s = 0;
        for (int g = 0; g < ng; g++) {
            int32_t qd = 0;
            for (int i = 0; i < 32; i++) qd += (int32_t)w[(size_t)n * K + g * 32 + i] * a.xq[g * 32 + i];
            s += (double)ggml_half_to_float(a.asa[g]) *
                 (ggml_half_to_float(step[(size_t)g * N + n]) * qd +
                  ggml_half_to_float(off[(size_t)g * N + n]) * a.xs[g]);
        }
        exp[n] = (float)s;
    }
    check_f32("moe_i8_gemv_down_launch", got, exp, 1e-5, 1e-3);
    free(d_bank, q); free(d_ids, q); free(d_xq, q); free(d_asa, q); free(d_xs, q); free(d_out, q);
}

// ---------------------------------------------------------------------------
static void test_u4_gemv_and_grouped(queue & q) {
    const int K = 256, N = 128, ng = K / 32;
    auto blocks = make_q4k(N, K);
    w4t w;
    if (!w4_pack(12, blocks.data(), K, N, w)) return fail("w4_pack", "false");
    const size_t vb = (size_t)N * (K / 2);
    const size_t plane = vb + 2 * (size_t)ng * N * 2;
    std::vector<uint8_t> bank(plane);
    std::memcpy(bank.data(), w.vals.data(), w.vals.size());
    std::memcpy(bank.data() + vb, w.scale.data(), w.scale.size() * 2);
    std::memcpy(bank.data() + vb + (size_t)ng * N * 2, w.off.data(), w.off.size() * 2);
    std::vector<float> x((size_t)K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 400);
    act a = make_act(x, K);
    uint8_t * d_bank = dev<uint8_t>(q, bank.size());
    int32_t * d_ids = dev<int32_t>(q, 1), *d_slot = dev<int32_t>(q, 1);
    int8_t * d_xq = dev<int8_t>(q, K);
    uint16_t * d_asa = dev<uint16_t>(q, ng);
    float * d_xs = dev<float>(q, ng), *d_out = dev<float>(q, (size_t)N), *d_outg = dev<float>(q, (size_t)N);
    up(q, d_bank, bank); up(q, d_ids, std::vector<int32_t>{0}); up(q, d_slot, std::vector<int32_t>{0});
    up(q, d_xq, a.xq); up(q, d_asa, a.asa); up(q, d_xs, a.xs);
    moe_u4_gemv_launch(q, d_bank, plane, d_slot, d_ids, 1, d_xq, d_asa, d_xs, K, N, d_out, N, nullptr, 0);
    std::vector<float> got(N), exp(N, 0.f);
    dn(q, got, d_out);
    for (int n = 0; n < N; n++) {
        double s = 0;
        for (int g = 0; g < ng; g++) {
            int32_t qd = 0;
            for (int i = 0; i < 32; i++) {
                const int k = g * 32 + i;
                const uint8_t by = w.vals[(size_t)n * (K / 2) + k / 2];
                const int q = (k & 1) ? (by >> 4) : (by & 0xF);
                qd += q * a.xq[k];
            }
            s += (double)ggml_half_to_float(a.asa[g]) *
                 (ggml_half_to_float(w.scale[(size_t)g * N + n]) * qd +
                  ggml_half_to_float(w.off[(size_t)g * N + n]) * a.xs[g]);
        }
        exp[n] = (float)s;
    }
    check_f32("moe_u4_gemv_launch", got, exp, 1e-5, 1e-3);

    // grouped u4: the same bank, one expert, one pair, compared with the GEMV.
    int32_t * d_off = dev<int32_t>(q, 2), *d_pair = dev<int32_t>(q, 1), *d_active = dev<int32_t>(q, 1),
            *d_nact = dev<int32_t>(q, 1);
    up(q, d_off, std::vector<int32_t>{0, 1}); up(q, d_pair, std::vector<int32_t>{0});
    up(q, d_active, std::vector<int32_t>{0}); up(q, d_nact, std::vector<int32_t>{1});
    const bool ok = moe_grp_u4_launch(q, d_bank, plane, d_off, d_pair, d_active, d_nact, 1, d_xq, d_asa, d_xs, d_outg,
                                      N, K, N, 1, nullptr);
    if (!ok) {
        fail("moe_grp_u4_launch", "declined");
    } else {
        std::vector<float> gotg(N);
        dn(q, gotg, d_outg);
        check_f32("moe_grp_u4_launch", gotg, exp, 1e-5, 1e-3);
    }
    free(d_bank, q); free(d_ids, q); free(d_slot, q); free(d_xq, q); free(d_asa, q); free(d_xs, q);
    free(d_out, q); free(d_outg, q); free(d_off, q); free(d_pair, q); free(d_active, q); free(d_nact, q);
}

// ---------------------------------------------------------------------------
static void test_native_down(queue & q) {
    const int K = 128, N = 5, ng = K / 32; // K whole 128-value panel for the sb form
    std::vector<block_q5_1> q5((size_t)N * ng);
    std::vector<block_q8_0> q8((size_t)N * ng);
    for (size_t j = 0; j < q5.size(); j++) {
        q5[j].d = ggml_float_to_half(0.002f * (float)((j % 7) + 1));
        q5[j].m = ggml_float_to_half(-0.01f * (float)((j % 5) + 1));
        uint32_t qh = (uint32_t)(j * 2654435761u);
        std::memcpy(q5[j].qh, &qh, 4);
        for (int i = 0; i < 16; i++) q5[j].qs[i] = (uint8_t)((j * 37 + i * 19) & 0xFF);
        q8[j].d = ggml_float_to_half(0.004f * (float)((j % 9) + 1));
        for (int i = 0; i < 32; i++) q8[j].qs[i] = (int8_t)((int)((j * 29 + i * 7) % 255) - 127);
    }
    std::vector<float> x((size_t)K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 900);
    act a = make_act(x, K);
    int32_t * d_off = dev<int32_t>(q, 2), *d_pair = dev<int32_t>(q, 1), *d_act = dev<int32_t>(q, 1),
            *d_nact = dev<int32_t>(q, 1), *d_smap = dev<int32_t>(q, 1);
    up(q, d_off, std::vector<int32_t>{0, 1}); up(q, d_pair, std::vector<int32_t>{0});
    up(q, d_act, std::vector<int32_t>{0}); up(q, d_nact, std::vector<int32_t>{1}); up(q, d_smap, std::vector<int32_t>{0});
    int8_t * d_xq = dev<int8_t>(q, K);
    uint16_t * d_asa = dev<uint16_t>(q, ng);
    float * d_xs = dev<float>(q, ng), *d_acc = dev<float>(q, (size_t)N);
    up(q, d_xq, a.xq); up(q, d_asa, a.asa); up(q, d_xs, a.xs);

    // raw Q5_1
    std::vector<uint8_t> bank5((size_t)N * ng * sizeof(block_q5_1));
    std::memcpy(bank5.data(), q5.data(), bank5.size());
    uint8_t * d_b5 = dev<uint8_t>(q, bank5.size());
    up(q, d_b5, bank5);
    moe_down_q5_1_launch(q, d_b5, d_smap, bank5.size(), d_off, d_pair, d_act, d_nact, 1, 1, 1, d_xq, d_asa, d_xs,
                         d_acc, N, N, K);
    std::vector<float> g5(N), e5(N, 0.f);
    dn(q, g5, d_acc);
    for (int n = 0; n < N; n++) {
        double s = 0;
        for (int g = 0; g < ng; g++) {
            const block_q5_1 & b = q5[(size_t)n * ng + g];
            int32_t qd = 0, xs = 0;
            for (int i = 0; i < 32; i++) {
                const int lo = (b.qs[i % 16] >> (4 * (i / 16))) & 0xF;
                const int hi = (b.qh[i / 8] >> (i % 8)) & 1;
                qd += (lo | (hi << 4)) * a.xq[g * 32 + i];
                xs += a.xq[g * 32 + i];
            }
            s += (double)ggml_half_to_float(a.asa[g]) *
                 (ggml_half_to_float(b.d) * qd + ggml_half_to_float(b.m) * xs);
        }
        e5[n] = (float)s;
    }
    check_f32("moe_down_q5_1_launch", g5, e5, 1e-5, 1e-3);

    // panel repack: same numbers by contract
    std::vector<uint8_t> packed((size_t)N * (ng / 4) * 96);
    q5_1_sb_pack(q5.data(), K, N, 1, packed.data());
    uint8_t * d_bp = dev<uint8_t>(q, packed.size());
    up(q, d_bp, packed);
    moe_down_q5_1_sb_launch(q, d_bp, d_smap, packed.size(), d_off, d_pair, d_act, d_nact, 1, 1, 1, d_xq, d_asa, d_xs,
                            d_acc, N, N, K);
    std::vector<float> gp(N);
    dn(q, gp, d_acc);
    check_f32("moe_down_q5_1_sb_launch", gp, e5, 1e-5, 1e-3);

    // Q8_0
    std::vector<uint8_t> bank8((size_t)N * ng * sizeof(block_q8_0));
    std::memcpy(bank8.data(), q8.data(), bank8.size());
    uint8_t * d_b8 = dev<uint8_t>(q, bank8.size());
    up(q, d_b8, bank8);
    moe_down_q8_0_launch(q, d_b8, d_smap, bank8.size(), d_off, d_pair, d_act, d_nact, 1, 1, 1, d_xq, d_asa, d_xs,
                         d_acc, N, N, K);
    std::vector<float> g8(N), e8(N, 0.f);
    dn(q, g8, d_acc);
    for (int n = 0; n < N; n++) {
        double s = 0;
        for (int g = 0; g < ng; g++) {
            const block_q8_0 & b = q8[(size_t)n * ng + g];
            int32_t qd = 0;
            for (int i = 0; i < 32; i++) qd += (int32_t)b.qs[i] * a.xq[g * 32 + i];
            s += (double)ggml_half_to_float(a.asa[g]) * ggml_half_to_float(b.d) * qd;
        }
        e8[n] = (float)s;
    }
    check_f32("moe_down_q8_0_launch", g8, e8, 1e-5, 1e-3);

    free(d_off, q); free(d_pair, q); free(d_act, q); free(d_nact, q); free(d_smap, q);
    free(d_xq, q); free(d_asa, q); free(d_xs, q); free(d_acc, q); free(d_b5, q); free(d_bp, q); free(d_b8, q);
}

// ---------------------------------------------------------------------------
static void test_grouped_gate_up(queue & q) {
    const int K = 256, N = 64; // one superblock per row, N a whole tile
    const size_t row_bytes = quant_row_bytes(12, K);
    auto g = make_q4k(N, K), u = make_q4k(N, K);
    for (size_t i = 0; i < u.size(); i++) u[i].d = ggml_float_to_half(0.006f * (float)((i % 4) + 1));
    std::vector<float> x((size_t)K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 1200);
    act a = make_act(x, K);
    uint8_t * d_g = dev<uint8_t>(q, g.size() * sizeof(block_q4_K));
    uint8_t * d_u = dev<uint8_t>(q, u.size() * sizeof(block_q4_K));
    q.memcpy(d_g, g.data(), g.size() * sizeof(block_q4_K)).wait();
    q.memcpy(d_u, u.data(), u.size() * sizeof(block_q4_K)).wait();
    int32_t * d_off = dev<int32_t>(q, 2), *d_pair = dev<int32_t>(q, 1), *d_act = dev<int32_t>(q, 1),
            *d_nact = dev<int32_t>(q, 1), *d_smap = dev<int32_t>(q, 1);
    up(q, d_off, std::vector<int32_t>{0, 1}); up(q, d_pair, std::vector<int32_t>{0});
    up(q, d_act, std::vector<int32_t>{0}); up(q, d_nact, std::vector<int32_t>{1}); up(q, d_smap, std::vector<int32_t>{0});
    int8_t * d_xq = dev<int8_t>(q, K);
    uint16_t * d_asa = dev<uint16_t>(q, K / 32);
    float * d_xs = dev<float>(q, K / 32);
    up(q, d_xq, a.xq); up(q, d_asa, a.asa); up(q, d_xs, a.xs);
    const size_t plane = (size_t)N * row_bytes;
    float * d_h = dev<float>(q, (size_t)N);
    const bool ok = moe_grp_gate_up_launch(q, d_g, d_u, d_smap, plane, plane, d_off, d_pair, d_act, d_nact, 1, 1, 1,
                                           d_xq, d_asa, d_xs, d_h, N, K, N, 1, 12, 12, true);
    if (!ok) {
        fail("moe_grp_gate_up_launch", "declined (needs a profile/type it accepts)");
    } else {
        std::vector<float> gh(N), eh(N);
        dn(q, gh, d_h);
        std::vector<float> wr(K);
        for (int n = 0; n < N; n++) {
            dequantize_row(12, (const char *)g.data() + (size_t)n * row_bytes, wr.data(), K);
            double gd = 0;
            for (int k = 0; k < K; k++) gd += (double)wr[k] * ggml_half_to_float(a.asa[k / 32]) * a.xq[k];
            dequantize_row(12, (const char *)u.data() + (size_t)n * row_bytes, wr.data(), K);
            double ud = 0;
            for (int k = 0; k < K; k++) ud += (double)wr[k] * ggml_half_to_float(a.asa[k / 32]) * a.xq[k];
            eh[n] = silu_((float)gd) * (float)ud;
        }
        check_f32("moe_grp_gate_up_launch", gh, eh, 1e-4, 1e-3);
    }
    free(d_g, q); free(d_u, q); free(d_off, q); free(d_pair, q); free(d_act, q); free(d_nact, q); free(d_smap, q);
    free(d_xq, q); free(d_asa, q); free(d_xs, q); free(d_h, q);
}

// ---------------------------------------------------------------------------
static void test_row_gemv(queue & q) {
    const int K = 256, N = 8;
    const size_t row_bytes = quant_row_bytes(12, K);
    auto dwn = make_q4k(N, K);
    std::vector<float> x((size_t)K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 1500);
    act a = make_act(x, K);
    uint8_t * d_b = dev<uint8_t>(q, dwn.size() * sizeof(block_q4_K));
    q.memcpy(d_b, dwn.data(), dwn.size() * sizeof(block_q4_K)).wait();
    int32_t * d_ids = dev<int32_t>(q, 1), *d_smap = dev<int32_t>(q, 1);
    up(q, d_ids, std::vector<int32_t>{0}); up(q, d_smap, std::vector<int32_t>{0});
    int8_t * d_xq = dev<int8_t>(q, K);
    uint16_t * d_asa = dev<uint16_t>(q, K / 32);
    float * d_out = dev<float>(q, (size_t)N);
    up(q, d_xq, a.xq); up(q, d_asa, a.asa);
    const bool ok = moe_row_gemv_down_launch(q, d_b, d_smap, (size_t)N * row_bytes, d_ids, 1, d_xq, d_asa, d_out, N,
                                             N, K, 12);
    if (!ok) {
        fail("moe_row_gemv_down_launch", "declined (needs a profile/type it accepts)");
    } else {
        std::vector<float> got(N), exp(N);
        dn(q, got, d_out);
        std::vector<float> wr(K);
        for (int n = 0; n < N; n++) {
            dequantize_row(12, (const char *)dwn.data() + (size_t)n * row_bytes, wr.data(), K);
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)wr[k] * ggml_half_to_float(a.asa[k / 32]) * a.xq[k];
            exp[n] = (float)s;
        }
        check_f32("moe_row_gemv_down_launch", got, exp, 1e-4, 1e-3);
    }
    free(d_b, q); free(d_ids, q); free(d_smap, q); free(d_xq, q); free(d_asa, q); free(d_out, q);
}

// ---------------------------------------------------------------------------
static void test_row_gate_up(queue & q) {
    const int K = 256, N = 8;
    const size_t row_bytes = quant_row_bytes(12, K);
    auto g = make_q4k(N, K), u = make_q4k(N, K);
    for (size_t i = 0; i < u.size(); i++) u[i].d = ggml_float_to_half(0.006f * (float)((i % 4) + 1));
    std::vector<float> x((size_t)K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 1750);
    act a = make_act(x, K);
    uint8_t * d_g = dev<uint8_t>(q, g.size() * sizeof(block_q4_K));
    uint8_t * d_u = dev<uint8_t>(q, u.size() * sizeof(block_q4_K));
    q.memcpy(d_g, g.data(), g.size() * sizeof(block_q4_K)).wait();
    q.memcpy(d_u, u.data(), u.size() * sizeof(block_q4_K)).wait();
    int32_t * d_ids = dev<int32_t>(q, 1), *d_smap = dev<int32_t>(q, 1);
    up(q, d_ids, std::vector<int32_t>{0}); up(q, d_smap, std::vector<int32_t>{0});
    int8_t * d_xq = dev<int8_t>(q, K);
    uint16_t * d_asa = dev<uint16_t>(q, K / 32);
    float * d_h = dev<float>(q, (size_t)N);
    up(q, d_xq, a.xq); up(q, d_asa, a.asa);
    const bool ok = moe_row_gemv_gate_up_launch(q, d_g, d_u, d_smap, (size_t)N * row_bytes, (size_t)N * row_bytes,
                                                d_ids, 1, d_xq, d_asa, d_h, N, K, N, 12, 12);
    if (!ok) {
        fail("moe_row_gemv_gate_up_launch", "declined");
    } else {
        std::vector<float> got(N), exp(N);
        dn(q, got, d_h);
        std::vector<float> wr(K);
        for (int n = 0; n < N; n++) {
            dequantize_row(12, (const char *)g.data() + (size_t)n * row_bytes, wr.data(), K);
            double gd = 0;
            for (int k = 0; k < K; k++) gd += (double)wr[k] * ggml_half_to_float(a.asa[k / 32]) * a.xq[k];
            dequantize_row(12, (const char *)u.data() + (size_t)n * row_bytes, wr.data(), K);
            double ud = 0;
            for (int k = 0; k < K; k++) ud += (double)wr[k] * ggml_half_to_float(a.asa[k / 32]) * a.xq[k];
            exp[n] = silu_((float)gd) * (float)ud;
        }
        check_f32("moe_row_gemv_gate_up_launch", got, exp, 1e-4, 1e-3);
    }
    free(d_g, q); free(d_u, q); free(d_ids, q); free(d_smap, q); free(d_xq, q); free(d_asa, q); free(d_h, q);
}

// ---------------------------------------------------------------------------
static void test_gemv_mix_and_kv(queue & q) {
    // mixed Q8_0 + F32 in one dispatch: each output must be its segment's dot.
    const int K = 256, N8 = 3, N0 = 2;
    std::vector<block_q8_0> wq((size_t)N8 * (K / 32));
    std::vector<float> w0((size_t)N0 * K), x(K);
    for (size_t i = 0; i < wq.size(); i++) {
        wq[i].d = ggml_float_to_half(0.005f * (float)((i % 5) + 1));
        for (int j = 0; j < 32; j++) wq[i].qs[j] = (int8_t)((int)((i * 17 + j * 3) % 255) - 127);
    }
    for (size_t i = 0; i < w0.size(); i++) w0[i] = rnd((int)i + 2000) * 0.1f;
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 2500);
    block_q8_0 * d_wq = dev<block_q8_0>(q, wq.size());
    float * d_w0 = dev<float>(q, w0.size()), *d_x = dev<float>(q, K);
    float * d_o8 = dev<float>(q, N8), *d_o0 = dev<float>(q, N0);
    up(q, d_wq, wq); up(q, d_w0, w0); up(q, d_x, x);

    gemv_seg hs[2] = {};
    hs[0].w = d_wq; hs[0].type = 8; hs[0].K = K; hs[0].n_rows = N8; hs[0].x = d_x; hs[0].x_stride = K;
    hs[0].out = d_o8; hs[0].out_stride = N8; hs[0].alpha = 1.f;
    hs[1].w = d_w0; hs[1].type = 0; hs[1].K = K; hs[1].n_rows = N0; hs[1].x = d_x; hs[1].x_stride = K;
    hs[1].out = d_o0; hs[1].out_stride = N0; hs[1].alpha = 1.f;
    int n8 = 0, n0 = 0, wgw = 0, wpr0 = 0;
    bool fs = false;
    const bool planned = gemv_mix_q8_f32_plan(q, hs, 2, 1, n8, n0, wgw, wpr0, fs);
    if (!planned) {
        fail("gemv_mix_q8_f32_plan", "declined an eligible mixed group");
    } else {
        gemv_seg * d_segs = dev<gemv_seg>(q, 2);
        q.memcpy(d_segs, hs, 2 * sizeof(gemv_seg)).wait();
        gemv_mix_q8_f32_launch(q, d_segs, 2, n8, n0, wgw, wpr0, fs);
        std::vector<float> g8(N8), g0(N0), e8(N8), e0(N0);
        dn(q, g8, d_o8); dn(q, g0, d_o0);
        std::vector<float> wrow(K);
        for (int n = 0; n < N8; n++) {
            dequantize_row(8, (const char *)wq.data() + (size_t)n * (K / 32) * sizeof(block_q8_0), wrow.data(), K);
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)wrow[k] * x[k];
            e8[n] = (float)s;
        }
        for (int n = 0; n < N0; n++) {
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)w0[(size_t)n * K + k] * x[k];
            e0[n] = (float)s;
        }
        check_f32("gemv_mix_q8_f32 Q8_0", g8, e8, 1e-5, 1e-3);
        check_f32("gemv_mix_q8_f32 F32", g0, e0, 1e-5, 1e-3);
        free(d_segs, q);
    }

    // KV checksum accessors: only live when PF_KVSTORE_SUM is set.
    if (std::getenv("PF_KVSTORE_SUM")) {
        const int s0 = kv_store_sum_slot(q, 1, 1);
        const int s1 = kv_store_sum_slot(q, 1, 1);
        int64_t * p0 = kv_store_sum_ptr(q); // allocates and consumes a slot too
        const bool ok = s0 == 0 && s1 == 1 && p0 != nullptr;
        std::printf("%-34s %s\n", "kv_store_sum accessors", ok ? "OK" : "FAIL");
        if (!ok) g_fail++;
    }
    free(d_wq, q); free(d_w0, q); free(d_x, q); free(d_o8, q); free(d_o0, q);
}

int main() {
    setenv("PF_ATTN_XMX_MIN", "1", 1);
    setenv("PF_KVSTORE_SUM", "1", 1);
    // make the mixed Q8_0/F32 group eligible on any card (the planner's real
    // gates are per-card row thresholds)
    setenv("PF_Q4_GEMV8_SPLIT_ROWS", "1", 1);
    setenv("PF_Q4_F32_SPLIT", "1", 1);
    try {
        queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
        std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        test_i8_down(q);
        test_u4_gemv_and_grouped(q);
        test_native_down(q);
        test_grouped_gate_up(q);
        test_row_gemv(q);
        test_row_gate_up(q);
        test_gemv_mix_and_kv(q);
    } catch (const sycl::exception & e) {
        std::printf("SYCL exception: %s\n", e.what());
        return 1;
    }
    if (g_fail) {
        std::printf("test_kernel_units_moe2: FAILURES: %d\n", g_fail);
        return 1;
    }
    std::printf("test_kernel_units_moe2: all units OK\n");
    return 0;
}
