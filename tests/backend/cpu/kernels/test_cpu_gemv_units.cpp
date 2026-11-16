// Hermetic unit tests for the CPU integer GEMV/GEMM paths and the qwen4exp
// down-projection stores.  The weights are synthetic GGUF blocks so the
// reference is the engine's own host dequantizer, not a second quantizer.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "kernels/common.h"
#include "w8.h"

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

// A valid Q4_K tensor: N rows of exactly one 256-value superblock each.
static std::vector<block_q4_K> make_q4k(int N) {
    std::vector<block_q4_K> b(N);
    for (int n = 0; n < N; n++) {
        b[n].d = ggml_float_to_half(0.004f * (float)(n + 1));
        b[n].dmin = ggml_float_to_half(0.001f * (float)(n + 1));
        for (int i = 0; i < K_SCALE_SIZE; i++) b[n].scales[i] = (uint8_t)(0x11 + 7 * i);
        for (int i = 0; i < QK_K / 2; i++) b[n].qs[i] = (uint8_t)((n * 37 + i * 11) & 0xFF);
    }
    return b;
}

// x8/xmeta/xsumq for TB rows of K values, via the verified quantizer.
struct act {
    std::vector<int8_t> x8;
    std::vector<float> meta;
    std::vector<int32_t> sum;
    std::vector<float> x;
};

static act make_act(const std::vector<float> & x, int TB, int K, int salt) {
    act a;
    a.x = x;
    a.x8.resize((size_t)TB * K);
    a.meta.resize((size_t)TB * (K / 32) * 2);
    a.sum.resize((size_t)TB * (K / 32) * 2);
    cpu_step_info info{};
    info.n_rows = 1;
    info.n_real = TB;
    info.tpb = 32;
    cpu_xq(x.data(), nullptr, K, K, a.x8.data(), a.meta.data(), a.sum.data(), &info, TB, K);
    (void)salt;
    return a;
}

// The exact fp32 reference for one row of the source blocks.
static void ref_row(const void * blocks, uint32_t type, int K, const float * x, float * out, int n) {
    std::vector<float> w(K);
    for (int r = 0; r < n; r++) {
        dequantize_row(type, (const char *)blocks + (size_t)r * quant_row_bytes(type, K), w.data(), K);
        double a = 0;
        for (int k = 0; k < K; k++) a += (double)w[k] * x[k];
        out[r] = (float)a;
    }
}

static void test_i8_gemv() {
    const int N = 3, K = 256;
    auto blocks = make_q4k(N);
    std::vector<float> x((size_t)K);
    for (int i = 0; i < K; i++) x[i] = rnd(i + 100);

    cpu_gemv_seg s{};
    s.w = blocks.data();
    s.type = GGML_TYPE_Q4_K;
    s.K = K;
    s.n_rows = N;
    s.x = x.data();
    s.x_stride = K;
    s.out = nullptr;
    s.out_stride = N;
    s.alpha = 1.0f;
    s.i8 = true;
    act a = make_act(x, 1, K, 1);
    std::vector<float> out(N), exp(N);
    s.x8 = a.x8.data();
    s.xmeta = a.meta.data();
    s.xsumq = a.sum.data();
    std::vector<float> o(N);
    s.out = o.data();
    cpu_i8_gemv(s, 4);
    ref_row(blocks.data(), GGML_TYPE_Q4_K, K, x.data(), exp.data(), N);
    check_f32("cpu_i8_gemv Q4_K", o, exp, 3e-2, 1e-2);

    // GEMM over three token rows.
    const int TB = 3;
    std::vector<float> xm((size_t)TB * K);
    for (size_t i = 0; i < xm.size(); i++) xm[i] = rnd((int)i + 700);
    act am = make_act(xm, TB, K, 2);
    s.x = xm.data();
    s.x_stride = K;
    s.x8 = am.x8.data();
    s.xmeta = am.meta.data();
    s.xsumq = am.sum.data();
    std::vector<float> og((size_t)TB * N);
    s.out = og.data();
    s.out_stride = N;
    cpu_i8_gemm(s, TB, 4);
    std::vector<float> eg((size_t)TB * N);
    for (int t = 0; t < TB; t++) ref_row(blocks.data(), GGML_TYPE_Q4_K, K, xm.data() + (size_t)t * K, eg.data() + (size_t)t * N, N);
    check_f32("cpu_i8_gemm Q4_K", og, eg, 3e-2, 1e-2);

    // The one-token chunked form must equal the one-token GEMV.
    std::vector<float> x1(xm.begin(), xm.begin() + K);
    act a1 = make_act(x1, 1, K, 5);
    std::vector<float> oc(N, 0.f), e1(N);
    s.x = x1.data();
    s.x8 = a1.x8.data();
    s.xmeta = a1.meta.data();
    s.xsumq = a1.sum.data();
    s.out = oc.data();
    s.out_stride = N;
    cpu_i8_gemv1_chunked(s, 2);
    ref_row(blocks.data(), GGML_TYPE_Q4_K, K, x1.data(), e1.data(), N);
    check_f32("cpu_i8_gemv1_chunked", oc, e1, 3e-2, 1e-2);
}

static void test_dp4a() {
    // kRB (128) rows: the SIn layout is blocked over 128 rows, so a smaller N
    // packs into a padded block.  N = 128 also makes the reference cover a
    // full row block.
    const int N = 128, K = 256;
    auto blocks = make_q4k(N);
    std::vector<uint8_t> vals(w8_vals_bytes(GGML_TYPE_Q4_K, K, N));
    std::vector<uint32_t> meta((w8_meta_bytes(GGML_TYPE_Q4_K, K, N) + 3) / 4);
    const bool packed = w8_repack(GGML_TYPE_Q4_K, blocks.data(), K, N, vals.data(), meta.data());
    if (!packed) {
        fail("w8_repack", "returned false");
        return;
    }
    w8t w;
    w.vals = vals.data();
    w.meta = meta.data();
    w.K = K;
    w.N = N;
    w.type = GGML_TYPE_Q4_K;
    w.bits = 4;
    w.meta_elem = 4;

    const int TB = 3;
    std::vector<float> x((size_t)TB * K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 1100);
    act a = make_act(x, TB, K, 3);

    cpu_gemv_seg s{};
    s.type = GGML_TYPE_Q4_K;
    s.K = K;
    s.n_rows = N;
    s.x = x.data();
    s.x_stride = K;
    s.x8 = a.x8.data();
    s.xmeta = a.meta.data();
    s.xsumq = a.sum.data();
    s.out_stride = N;
    s.alpha = 1.0f;
    s.w8 = w;

    std::vector<float> x1(x.begin(), x.begin() + K);
    act a1 = make_act(x1, 1, K, 6);
    s.x = x1.data();
    s.x8 = a1.x8.data();
    s.xmeta = a1.meta.data();
    s.xsumq = a1.sum.data();
    std::vector<float> o(N);
    s.out = o.data();
    cpu_dp4a_gemv(s, 4);
    std::vector<float> exp(N);
    ref_row(blocks.data(), GGML_TYPE_Q4_K, K, x1.data(), exp.data(), N);
    check_f32("cpu_dp4a_gemv Q4_K", o, exp, 3e-2, 1e-2);

    s.x = x.data();
    s.x8 = a.x8.data();
    s.xmeta = a.meta.data();
    s.xsumq = a.sum.data();
    std::vector<float> og((size_t)TB * N);
    s.out = og.data();
    cpu_dp4a_gemm(s, TB, 4);
    std::vector<float> eg((size_t)TB * N);
    for (int t = 0; t < TB; t++) {
        ref_row(blocks.data(), GGML_TYPE_Q4_K, K, x.data() + (size_t)t * K, eg.data() + (size_t)t * N, N);
    }
    check_f32("cpu_dp4a_gemm Q4_K", og, eg, 3e-2, 1e-2);
}

// ---------------------------------------------------------------------------
// qwen4exp's routed-down stores (int8 re-quantization of the GGUF blocks).
static void test_q4_down() {
    const int E = 2, N = 3, K = 64, G = K / 32;
    const size_t row_bytes = quant_row_bytes(GGML_TYPE_Q8_0, K);
    std::vector<uint8_t> src((size_t)E * N * row_bytes);
    for (int e = 0; e < E; e++) {
        for (int n = 0; n < N; n++) {
            block_q8_0 * b = (block_q8_0 *)(src.data() + ((size_t)e * N + n) * row_bytes);
            for (int g = 0; g < G; g++) {
                b[g].d = ggml_float_to_half(0.003f * (float)(e + n + g + 1));
                for (int i = 0; i < 32; i++) b[g].qs[i] = (int8_t)((int)((e * 31 + n * 7 + g * 13 + i * 3) % 255) - 127);
            }
        }
    }
    std::vector<uint8_t> dst((size_t)E * N * K);
    std::vector<float> sc((size_t)E * N * G);
    cpu_q4_convert_down(src.data(), GGML_TYPE_Q8_0, K, N, E, (size_t)N * row_bytes, dst.data(), sc.data());

    // The store is a per-32 symmetric int8 quantization of each dequantized row.
    double max_err = 0, max_val = 0;
    std::vector<float> v(K);
    for (int e = 0; e < E; e++) {
        for (int n = 0; n < N; n++) {
            dequantize_row(GGML_TYPE_Q8_0, src.data() + ((size_t)e * N + n) * row_bytes, v.data(), K);
            for (int g = 0; g < G; g++) {
                float mx = 0;
                for (int i = 0; i < 32; i++) mx = std::max(mx, std::fabs(v[g * 32 + i]));
                const float s = mx > 0 ? mx / 127.0f : 1.0f;
                const float * srow = &sc[((size_t)e * N + n) * G];
                if (std::fabs(srow[g] - s) > 1e-7f * std::max(1.f, s)) fail("cpu_q4_convert_down scale", "mismatch");
                for (int i = 0; i < 32; i++) {
                    const int q = (int)std::max(-127, std::min(127, (int)std::lround(v[g * 32 + i] / s)));
                    const int got = (int)dst[((size_t)e * N + n) * K + g * 32 + i] - 128;
                    const float rec = (float)got * s;
                    max_err = std::max(max_err, (double)std::fabs(rec - v[g * 32 + i]));
                    max_val = std::max(max_val, (double)std::fabs(v[g * 32 + i]));
                    if (got != q) fail("cpu_q4_convert_down quant", "mismatch");
                }
            }
        }
    }
    std::printf("%-32s max|exp|=%.5f max|err|=%.3e %s\n", "cpu_q4_convert_down", max_val, max_err,
                max_err <= max_val / 127.0 * 1.5 ? "OK" : "FAIL");
    if (max_err > max_val / 127.0 * 1.5) g_fail++;

    // Grouped down with P pairs; xTB == P so pair p is activation row p.
    const int P = 5;
    std::vector<float> x((size_t)P * K);
    for (size_t i = 0; i < x.size(); i++) x[i] = rnd((int)i + 2000);
    act a = make_act(x, P, K, 4);
    std::vector<int> pairs{0, 2, 4, 1, 3};
    std::vector<float> acc((size_t)P * N, 0.f);
    cpu_q4_down_grouped(dst.data(), sc.data(), a.x8.data(), a.meta.data(), a.sum.data(), P, P, N, K, acc.data(), N,
                        pairs.data());
    std::vector<float> exp((size_t)P * N, 0.f);
    for (int p = 0; p < P; p++) {
        const int pr = pairs[p];
        for (int n = 0; n < N; n++) {
            double total = 0;
            for (int g = 0; g < G; g++) {
                const size_t gi = (size_t)g * P + pr;
                double d = 0;
                for (int i = 0; i < 32; i++) {
                    d += (double)((int)dst[((size_t)n) * K + g * 32 + i] - 128) * a.x8[gi * 32 + i];
                }
                total += (double)a.meta[gi * 2] * sc[(size_t)n * G + g] * d;
            }
            exp[(size_t)pr * N + n] = (float)total;
        }
    }
    check_f32("cpu_q4_down_grouped", acc, exp, 1e-6, 1e-3);

    // The serial twin must be the same numbers.
    std::vector<float> accs((size_t)P * N, 0.f);
    cpu_q4_down_grouped_serial(dst.data(), sc.data(), a.x8.data(), a.meta.data(), a.sum.data(), P, P, N, K, accs.data(),
                               N, pairs.data());
    check_f32("cpu_q4_down_grouped_serial", accs, exp, 1e-6, 1e-3);
}

int main(int argc, char ** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    if (only.empty() || only == "i8") test_i8_gemv();
    if (only.empty() || only == "dp4a") test_dp4a();
    if (only.empty() || only == "down") test_q4_down();
    if (g_fail) {
        std::printf("test_cpu_gemv_units: FAILURES: %d\n", g_fail);
        return 1;
    }
    std::printf("test_cpu_gemv_units: all units OK\n");
    return 0;
}
