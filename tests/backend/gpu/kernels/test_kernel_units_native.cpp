// Model-free unit tests for the native-width weight stores (u4, 5-bit,
// codebook, 2-bit) and the narrow grouped-int8 GEMVs.
//
// The weights are synthetic GGUF blocks packed by the production packers, and
// the references decode the same store definition in scalar C++.  This is the
// numerical half of the u4/k5/cb4/int8 kernels: a packing or lane-mapping
// regression cannot hide behind "the model's logits moved".
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "kernels.h"
#include "quant.h"
#include "w4.h"

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
template <typename T> static T * dev(queue & q, size_t n) { return malloc_device<T>(n, q); }
template <typename T> static void up(queue & q, T * d, const std::vector<T> & h) {
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
}

// Per-32-group symmetric int8 activation, in the layouts the kernels consume.
struct act {
    std::vector<int8_t> xq;      // [K]
    std::vector<int8_t> axe, axo; // [K/2] even/odd split
    std::vector<uint16_t> asa;   // [ng] f16 scale
    std::vector<float> xs;       // [ng] group sum of the int8 values
};
static act make_act(const std::vector<float> & x, int K) {
    const int ng = K / 32;
    act a;
    a.xq.resize((size_t)K);
    a.axe.resize((size_t)K / 2);
    a.axo.resize((size_t)K / 2);
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
    for (int j = 0; j < K / 2; j++) {
        a.axe[j] = a.xq[2 * j];
        a.axo[j] = a.xq[2 * j + 1];
    }
    return a;
}

// A Q4_K tensor, N rows of one 256-value superblock each.
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

// dequantized reference weight row
static void ref_row(const void * blocks, uint32_t type, int K, int row, std::vector<float> & w) {
    w.resize((size_t)K);
    dequantize_row(type, (const char *)blocks + (size_t)row * quant_row_bytes(type, K), w.data(), K);
}

static void test_w4(queue & q) {
    // 128 rows keeps the Q4_K superblocks and exercises the row-block tile;
    // N=3 output columns keeps the reference small.  K is a multiple of 256.
    const int K = 512, N = 3, M = 2;
    auto blocks = make_q4k(N, K);
    w4t w;
    if (!w4_pack(12, blocks.data(), K, N, w)) return fail("w4_pack", "false");
    std::vector<float> x((size_t)M * K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 100);
    std::vector<act> a(M);
    for (int m = 0; m < M; m++) {
        std::vector<float> xr(x.begin() + (size_t)m * K, x.begin() + (size_t)(m + 1) * K);
        a[m] = make_act(xr, K);
    }

    uint8_t * d_vals = dev<uint8_t>(q, w.vals.size());
    uint16_t * d_scale = dev<uint16_t>(q, w.scale.size()), *d_off = dev<uint16_t>(q, w.off.size());
    up(q, d_vals, w.vals);
    up(q, d_scale, w.scale);
    up(q, d_off, w.off);

    // M = 1
    int8_t * d_axe = dev<int8_t>(q, a[0].axe.size()), *d_axo = dev<int8_t>(q, a[0].axo.size());
    uint16_t * d_asa = dev<uint16_t>(q, a[0].asa.size());
    float * d_xs = dev<float>(q, a[0].xs.size());
    float * d_out = dev<float>(q, (size_t)N);
    up(q, d_axe, a[0].axe); up(q, d_axo, a[0].axo); up(q, d_asa, a[0].asa); up(q, d_xs, a[0].xs);
    w4_gemv_launch(q, d_vals, d_scale, d_off, d_axe, d_axo, d_asa, d_xs, d_out, N, nullptr, 1.f, K, N);
    std::vector<float> got(N), exp(N);
    q.memcpy(got.data(), d_out, N * 4).wait();
    std::vector<float> wrow;
    for (int n = 0; n < N; n++) {
        ref_row(blocks.data(), 12, K, n, wrow);
        double s = 0;
        for (int k = 0; k < K; k++) s += (double)wrow[k] * ggml_half_to_float(a[0].asa[k / 32]) * a[0].xq[k];
        exp[n] = (float)s;
    }
    check_f32("w4_gemv_launch", got, exp, 2e-3, 1e-3);

    // fused quantizer + u4 GEMV: the same row, with the activation quantized
    // inside the launcher (sycl::half scale, one f16 ulp from the CPU form).
    float * d_xf = dev<float>(q, K);
    float * d_of = dev<float>(q, (size_t)N);
    up(q, d_xf, std::vector<float>(x.begin(), x.begin() + K));
    w4_grp_seg qseg{};
    qseg.vals = d_vals; qseg.scale = d_scale; qseg.off = d_off; qseg.n_rows = N;
    qseg.out = d_of; qseg.alpha = 1.f; qseg.residual = nullptr;
    const bool qok = w4_grp_quant_gemv_launch(q, &qseg, 1, N, d_xf, nullptr, K);
    if (!qok) {
        fail("w4_grp_quant_gemv", "declined");
    } else {
        std::vector<float> gq(N);
        q.memcpy(gq.data(), d_of, N * 4).wait();
        check_f32("w4_grp_quant_gemv", gq, exp, 1e-3, 1e-3);
    }

    // M = 2
    std::vector<int8_t> axe2((size_t)M * (K / 2)), axo2((size_t)M * (K / 2));
    std::vector<uint16_t> asa2;
    std::vector<float> xs2;
    for (int m = 0; m < M; m++) {
        std::memcpy(axe2.data() + (size_t)m * (K / 2), a[m].axe.data(), (size_t)K / 2);
        std::memcpy(axo2.data() + (size_t)m * (K / 2), a[m].axo.data(), (size_t)K / 2);
        asa2.insert(asa2.end(), a[m].asa.begin(), a[m].asa.end());
        xs2.insert(xs2.end(), a[m].xs.begin(), a[m].xs.end());
    }
    int8_t * d_axe2 = dev<int8_t>(q, axe2.size()), *d_axo2 = dev<int8_t>(q, axo2.size());
    uint16_t * d_asa2 = dev<uint16_t>(q, asa2.size());
    float * d_xs2 = dev<float>(q, xs2.size());
    float * d_out2 = dev<float>(q, (size_t)M * N), *d_outn = dev<float>(q, (size_t)M * N);
    up(q, d_axe2, axe2); up(q, d_axo2, axo2); up(q, d_asa2, asa2); up(q, d_xs2, xs2);
    w4_gemm_launch(q, d_vals, d_scale, d_off, d_axe2, d_axo2, d_asa2, d_xs2, d_out2, N, nullptr, 1.f, M, K, N);
    std::vector<float> got2((size_t)M * N), exp2((size_t)M * N);
    q.memcpy(got2.data(), d_out2, got2.size() * 4).wait();
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            ref_row(blocks.data(), 12, K, n, wrow);
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)wrow[k] * ggml_half_to_float(a[m].asa[k / 32]) * a[m].xq[k];
            exp2[(size_t)m * N + n] = (float)s;
        }
    }
    check_f32("w4_gemm_launch", got2, exp2, 2e-3, 1e-3);

    const bool nat = nat_gemm_launch(q, 0, d_vals, nullptr, nullptr, d_scale, d_off, nullptr, nullptr, d_axe2,
                                     d_axo2, nullptr, d_asa2, d_xs2, d_outn, N, nullptr, 1.f, M, K, N);
    if (!nat) {
        fail("nat_gemm_launch", "declined fmt 0 M=2");
    } else {
        std::vector<float> gotn((size_t)M * N);
        q.memcpy(gotn.data(), d_outn, gotn.size() * 4).wait();
        check_f32("nat_gemm_launch fmt0", gotn, exp2, 2e-3, 1e-3);
    }

    free(d_vals, q); free(d_scale, q); free(d_off, q); free(d_axe, q); free(d_axo, q); free(d_asa, q);
    free(d_xs, q); free(d_out, q); free(d_axe2, q); free(d_axo2, q); free(d_asa2, q); free(d_xs2, q);
    free(d_out2, q); free(d_outn, q); free(d_xf, q); free(d_of, q);
}

static void test_k5_expand(queue & q) {
    const int K = 256, N = 3;
    std::vector<block_q5_K> src((size_t)N);
    for (int n = 0; n < N; n++) {
        src[n].d = ggml_float_to_half(0.003f * (n + 1));
        src[n].dmin = ggml_float_to_half(0.001f * (n + 1));
        for (int i = 0; i < K_SCALE_SIZE; i++) src[n].scales[i] = (uint8_t)(0x12 + 5 * i + n);
        for (int i = 0; i < QK_K / 8; i++) src[n].qh[i] = (uint8_t)((n * 17 + i * 31) & 0xFF);
        for (int i = 0; i < QK_K / 2; i++) src[n].qs[i] = (uint8_t)((n * 23 + i * 9) & 0xFF);
    }
    k5t w;
    if (!k5_pack(13, src.data(), K, N, w)) return fail("k5_pack", "false");
    std::vector<uint32_t> bl(16);
    for (int i = 0; i < 16; i++) {
        bl[i] = (uint32_t)(i & 1) | ((uint32_t)((i >> 1) & 1) << 8) | ((uint32_t)((i >> 2) & 1) << 16) |
                ((uint32_t)((i >> 3) & 1) << 24);
    }
    uint8_t * d_v = dev<uint8_t>(q, w.vals.size());
    uint8_t * d_h = dev<uint8_t>(q, w.hi.size());
    uint32_t * d_bl = dev<uint32_t>(q, bl.size());
    int8_t * d_o = dev<int8_t>(q, (size_t)N * K);
    up(q, d_v, w.vals); up(q, d_h, w.hi); up(q, d_bl, bl);
    k5_expand_launch(q, d_v, d_h, d_bl, d_o, K, N);
    std::vector<int8_t> got((size_t)N * K), exp((size_t)N * K);
    q.memcpy(got.data(), d_o, got.size()).wait();
    for (int n = 0; n < N; n++) {
        const block_q5_K & b = src[n];
        for (int i = 0; i < K; i++) {
            const int j64 = i / 64;
            const int l = i % 64;
            const int l32 = l & 31;
            const int lo = (l < 32) ? (b.qs[j64 * 32 + l32] & 0xF) : ((b.qs[j64 * 32 + l32] >> 4) & 0xF);
            const int bit = (l < 32) ? (2 * j64) : (2 * j64 + 1);
            const int hi = (b.qh[l32] >> bit) & 1;
            exp[(size_t)n * K + i] = (int8_t)(lo | (hi << 4));
        }
    }
    check_i8("k5_expand_launch", got, exp);
    free(d_v, q); free(d_h, q); free(d_bl, q); free(d_o, q);
}

static void test_w2(queue & q) {
    const int K = 256, N = 3;
    auto blocks = make_q4k(N, K);
    w2t w;
    if (!w2_pack_any(12, blocks.data(), K, N, w)) return fail("w2_pack_any", "false");
    std::vector<float> x((size_t)K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 900);
    act a = make_act(x, K);
    uint8_t * d_vals = dev<uint8_t>(q, w.vals.size());
    uint16_t * d_scale = dev<uint16_t>(q, w.scale.size()), *d_off = dev<uint16_t>(q, w.off.size());
    int8_t * d_xq = dev<int8_t>(q, a.xq.size());
    uint16_t * d_asa = dev<uint16_t>(q, a.asa.size());
    float * d_xs = dev<float>(q, a.xs.size());
    float * d_out = dev<float>(q, (size_t)N);
    up(q, d_vals, w.vals); up(q, d_scale, w.scale); up(q, d_off, w.off);
    up(q, d_xq, a.xq); up(q, d_asa, a.asa); up(q, d_xs, a.xs);
    w2_gemv_launch(q, d_vals, d_scale, d_off, d_xq, d_asa, d_xs, d_out, N, nullptr, 1.f, K, N);
    std::vector<float> got(N), exp(N, 0.f);
    q.memcpy(got.data(), d_out, N * 4).wait();
    const int ng = K / 32;
    for (int n = 0; n < N; n++) {
        double s = 0;
        for (int g = 0; g < ng; g++) {
            int32_t qd = 0;
            for (int i = 0; i < 32; i++) {
                const int k = g * 32 + i;
                const int q = (w.vals[(size_t)n * (K / 4) + k / 4] >> (2 * (k & 3))) & 3;
                qd += q * a.xq[k];
            }
            const float step = ggml_half_to_float(w.scale[(size_t)g * N + n]);
            const float off = ggml_half_to_float(w.off[(size_t)g * N + n]);
            s += (double)ggml_half_to_float(a.asa[g]) * (step * (float)qd + off * a.xs[g]);
        }
        exp[n] = (float)s;
    }
    check_f32("w2_gemv_launch", got, exp, 1e-5, 1e-3);
    free(d_vals, q); free(d_scale, q); free(d_off, q); free(d_xq, q); free(d_asa, q); free(d_xs, q);
    free(d_out, q);
}

static void test_cb4(queue & q) {
    // One IQ4_NL block per 32 values: f16 d + 16 index bytes.
    const int K = 64, N = 3, ng = K / 32;
    std::vector<block_iq4_nl> src((size_t)N * ng);
    for (size_t j = 0; j < src.size(); j++) {
        src[j].d = ggml_float_to_half(0.003f * (float)((j % 5) + 1));
        for (int i = 0; i < 16; i++) src[j].qs[i] = (uint8_t)((j * 29 + i * 7) & 0xFF);
    }
    cb4t w;
    if (!cb4_pack(20, src.data(), K, N, w)) return fail("cb4_pack", "false");
    std::vector<float> x((size_t)K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 1500);
    act a = make_act(x, K);
    uint8_t * d_idx = dev<uint8_t>(q, w.idx.size());
    uint16_t * d_scale = dev<uint16_t>(q, w.scale.size());
    int8_t * d_xq = dev<int8_t>(q, a.xq.size());
    uint16_t * d_asa = dev<uint16_t>(q, a.asa.size());
    float * d_xs = dev<float>(q, a.xs.size());
    float * d_out = dev<float>(q, (size_t)N);
    std::vector<uint16_t> lut16(256);
    for (int b = 0; b < 256; b++) {
        lut16[b] = (uint16_t)((uint8_t)kvalues_iq4nl[b & 0xF] | ((uint16_t)(uint8_t)kvalues_iq4nl[b >> 4] << 8));
    }
    uint16_t * d_lut = dev<uint16_t>(q, lut16.size());
    up(q, d_idx, w.idx); up(q, d_scale, w.scale); up(q, d_lut, lut16);
    up(q, d_xq, a.xq); up(q, d_asa, a.asa); up(q, d_xs, a.xs);
    cb4_gemv_launch(q, d_idx, d_lut, d_scale, d_xq, d_asa, d_xs, d_out, nullptr, 1.f, K, N);
    std::vector<float> got(N), exp(N, 0.f);
    q.memcpy(got.data(), d_out, N * 4).wait();
    for (int n = 0; n < N; n++) {
        double s = 0;
        for (int g = 0; g < ng; g++) {
            int32_t qd = 0;
            for (int i = 0; i < 32; i++) {
                const uint8_t b = w.idx[(size_t)n * (K / 2) + g * 16 + i / 2];
                const int idx = (i & 1) ? (b >> 4) : (b & 0xF);
                qd += (int32_t)kvalues_iq4nl[idx] * a.xq[g * 32 + i];
            }
            s += (double)ggml_half_to_float(a.asa[g]) * ggml_half_to_float(w.scale[(size_t)g * N + n]) * (float)qd;
        }
        exp[n] = (float)s;
    }
    check_f32("cb4_gemv_launch", got, exp, 1e-5, 1e-3);

    int8_t * d_e = dev<int8_t>(q, (size_t)N * K);
    cb4_expand_launch(q, d_idx, d_lut, d_e, K, N);
    std::vector<int8_t> ex((size_t)N * K);
    q.memcpy(ex.data(), d_e, ex.size()).wait();
    std::vector<int8_t> eexp(ex.size());
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            const uint8_t b = w.idx[(size_t)n * (K / 2) + (k / 32) * 16 + (k % 32) / 2];
            const int idx = (k & 1) ? (b >> 4) : (b & 0xF);
            eexp[(size_t)n * K + k] = kvalues_iq4nl[idx];
        }
    }
    check_i8("cb4_expand_launch", ex, eexp);
    free(d_idx, q); free(d_scale, q); free(d_lut, q); free(d_xq, q); free(d_asa, q); free(d_xs, q);
    free(d_out, q); free(d_e, q);
}

static void test_i8grp(queue & q) {
    const int K = 512, N = 5, M = 2, ng = K / 32;
    std::vector<int8_t> w((size_t)N * K);
    std::vector<uint16_t> wsc((size_t)ng * N);
    for (size_t i = 0; i < w.size(); i++) w[i] = (int8_t)((int)(i * 13) % 255 - 127);
    for (int g = 0; g < ng; g++) {
        for (int n = 0; n < N; n++) wsc[(size_t)g * N + n] = ggml_float_to_half(0.002f * (float)((g + n) % 5 + 1));
    }
    std::vector<float> x((size_t)K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 2100);
    act a = make_act(x, K);
    int8_t * d_w = dev<int8_t>(q, w.size());
    uint16_t * d_wsc = dev<uint16_t>(q, wsc.size());
    int8_t * d_xq = dev<int8_t>(q, a.xq.size());
    uint16_t * d_asa = dev<uint16_t>(q, a.asa.size());
    float * d_xs = dev<float>(q, a.xs.size());
    float * d_out = dev<float>(q, (size_t)N);
    up(q, d_w, w); up(q, d_wsc, wsc); up(q, d_xq, a.xq); up(q, d_asa, a.asa); up(q, d_xs, a.xs);

    auto ref = [&](int n) {
        double s = 0;
        for (int g = 0; g < ng; g++) {
            int32_t qd = 0;
            for (int i = 0; i < 32; i++) qd += (int32_t)w[(size_t)n * K + g * 32 + i] * a.xq[g * 32 + i];
            s += (double)ggml_half_to_float(a.asa[g]) * ggml_half_to_float(wsc[(size_t)g * N + n]) * qd;
        }
        return (float)s;
    };
    std::vector<float> exp(N);
    for (int n = 0; n < N; n++) exp[n] = ref(n);

    i8_grp_gemv_launch(q, d_w, d_wsc, d_xq, d_asa, d_xs, d_out, nullptr, 1.f, K, N);
    std::vector<float> got(N);
    q.memcpy(got.data(), d_out, N * 4).wait();
    check_f32("i8_grp_gemv_launch", got, exp, 1e-5, 1e-3);

    // M rows of activation, one segment.
    std::vector<int8_t> xq2((size_t)M * K);
    std::vector<uint16_t> asa2;
    std::vector<float> xs2;
    for (int m = 0; m < M; m++) {
        std::vector<float> xm((size_t)K);
        for (int i = 0; i < K; i++) xm[i] = x[i] * (1.0f + 0.1f * m);
        act am = make_act(xm, K);
        std::memcpy(xq2.data() + (size_t)m * K, am.xq.data(), K);
        asa2.insert(asa2.end(), am.asa.begin(), am.asa.end());
        xs2.insert(xs2.end(), am.xs.begin(), am.xs.end());
    }
    int8_t * d_xq2 = dev<int8_t>(q, xq2.size());
    uint16_t * d_asa2 = dev<uint16_t>(q, asa2.size());
    float * d_xs2 = dev<float>(q, xs2.size());
    float * d_out2 = dev<float>(q, (size_t)M * N);
    up(q, d_xq2, xq2); up(q, d_asa2, asa2); up(q, d_xs2, xs2);
    i8_grp_gemv_rows_launch(q, d_w, d_wsc, d_xq2, d_asa2, d_xs2, d_out2, N, nullptr, 1.f, M, K, N);
    std::vector<float> gotr((size_t)M * N), expr((size_t)M * N);
    q.memcpy(gotr.data(), d_out2, gotr.size() * 4).wait();
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int g = 0; g < ng; g++) {
                int32_t qd = 0;
                for (int i = 0; i < 32; i++) qd += (int32_t)w[(size_t)n * K + g * 32 + i] * xq2[(size_t)m * K + g * 32 + i];
                s += (double)ggml_half_to_float(asa2[(size_t)m * ng + g]) * ggml_half_to_float(wsc[(size_t)g * N + n]) * qd;
            }
            expr[(size_t)m * N + n] = (float)s;
        }
    }
    check_f32("i8_grp_gemv_rows_launch", gotr, expr, 1e-5, 1e-3);

    // multi: one segment, the same rows as the single-segment form (the NS=1
    // instantiation of the multi path).
    i8_grp_seg segs[1] = {{d_w, d_wsc, N, d_out2, N, 1.f, nullptr}};
    std::vector<float> zero((size_t)M * N, 0.f);
    q.memcpy(d_out2, zero.data(), zero.size() * 4).wait();
    i8_grp_gemv_rows_multi_launch(q, segs, 1, N, d_xq2, d_asa2, d_xs2, M, K);
    std::vector<float> gotm((size_t)M * N);
    q.memcpy(gotm.data(), d_out2, gotm.size() * 4).wait();
    check_f32("i8_grp_gemv_rows_multi", gotm, expr, 1e-5, 1e-3);

    // fused quantizer + GEMV: use the contiguous float row.
    i8_grp_seg qs{d_w, d_wsc, N, d_out, N, 1.f, nullptr};
    float * d_xf = dev<float>(q, x.size());
    up(q, d_xf, x);
    i8_grp_quant_gemv_launch(q, &qs, 1, N, d_xf, nullptr, K);
    std::vector<float> gotq(N);
    q.memcpy(gotq.data(), d_out, N * 4).wait();
    // the fused quantizer stores the scale as sycl::half (round-to-nearest) while
    // this reference uses ggml_float_to_half (truncating): one f16 ulp per group.
    check_f32("i8_grp_quant_gemv", gotq, exp, 1e-3, 1e-3);

    free(d_w, q); free(d_wsc, q); free(d_xq, q); free(d_asa, q); free(d_xs, q); free(d_out, q);
    free(d_xq2, q); free(d_asa2, q); free(d_xs2, q); free(d_out2, q); free(d_xf, q);
}

int main() {
    try {
        queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
        std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        test_w4(q);
        test_k5_expand(q);
        test_w2(q);
        test_cb4(q);
        test_i8grp(q);
    } catch (const sycl::exception & e) {
        std::printf("SYCL exception: %s\n", e.what());
        return 1;
    }
    if (g_fail) {
        std::printf("test_kernel_units_native: FAILURES: %d\n", g_fail);
        return 1;
    }
    std::printf("test_kernel_units_native: all units OK\n");
    return 0;
}
