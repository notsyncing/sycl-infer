// CPU kernel unit test: the host dequantized GEMV and RMSNorm against the
// quant.h host reference.  Mirrors tests/backend/gpu/kernels/test_gemv.cpp for
// the CPU backend; no GPU is needed (the weights stay in the mmap).
//
// Run with PF_CPU_ISA=scalar|avx2|avx512 to validate a specific ISA variant.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "model.h"
#include "quant.h"

using namespace si;

static int failures = 0;

static void check_gemv(model & m, const std::string & name, int TB) {
    const gguf_tensor_info * ti = m.gguf.find(name);
    if (!ti) {
        printf("SKIP %s (missing)\n", name.c_str());
        return;
    }
    const int K = (int)ti->dims[0];
    const int N = (int)ti->n_rows();
    printf("test %-28s type=%-5s K=%-5d N=%-6d TB=%d ... ", name.c_str(), ggml_type_name(ti->type), K, N, TB);
    fflush(stdout);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t)TB * K);
    for (auto & v : hx) {
        v = nd(rng);
    }
    std::vector<float> hy((size_t)TB * N);

    cpu_gemv_seg s{};
    s.w = ti->data;
    s.type = ti->type;
    s.K = K;
    s.n_rows = N;
    s.x = hx.data();
    s.x_stride = K;
    s.out = hy.data();
    s.out_stride = N;
    s.alpha = 1.0f;

    // emulate the prefill slicing (8-token slices) for TB == 8
    std::vector<cpu_gemv_seg> segs;
    if (TB == 8) {
        for (int sl = 0; sl * 8 < TB; sl++) {
            cpu_gemv_seg c = s;
            c.x = s.x + (size_t)sl * 8 * s.x_stride;
            c.out = s.out + (size_t)sl * 8 * s.out_stride;
            segs.push_back(c);
        }
    } else {
        segs.push_back(s);
    }
    cpu_gemv_group(s.type, segs.data(), (int)segs.size(), N * (int)segs.size(), TB, K / 256, 0, 0);

    std::vector<float> wrow(K);
    double max_err = 0, ref_max = 0;
    const int rstep = N > 4096 ? 97 : 1;
    for (int t = 0; t < TB; t++) {
        for (int r = 0; r < N; r += rstep) {
            dequantize_row(ti->type, (const char *)ti->data + (size_t)r * quant_row_bytes(ti->type, K), wrow.data(), K);
            double ref = 0;
            for (int k = 0; k < K; k++) {
                ref += (double)wrow[k] * hx[(size_t)t * K + k];
            }
            const double err = std::fabs(ref - hy[(size_t)t * N + r]);
            max_err = std::max(max_err, err / std::max(1.0, std::fabs(ref)));
            ref_max = std::max(ref_max, std::fabs(ref));
        }
    }
    const bool ok = max_err < 2e-3;
    printf("rel err max=%.3e (|ref|max=%.1f) %s\n", max_err, ref_max, ok ? "OK" : "FAIL");
    if (!ok) {
        failures++;
    }
}

static void check_i8(model & m, const std::string & name, int TB) {
    const gguf_tensor_info * ti = m.gguf.find(name);
    if (!ti) {
        printf("SKIP %s (missing)\n", name.c_str());
        return;
    }
    const int K = (int)ti->dims[0];
    const int N = (int)ti->n_rows();
    printf("i8   %-28s type=%-5s K=%-5d N=%-6d TB=%d ... ", name.c_str(), ggml_type_name(ti->type), K, N, TB);
    fflush(stdout);
    const int groups = K / 32;

    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t)TB * K);
    for (auto & v : hx) {
        v = nd(rng);
    }
    std::vector<float> hy((size_t)TB * N);
    std::vector<int8_t> x8((size_t)TB * K);
    std::vector<float> xmeta((size_t)TB * groups * 2);
    std::vector<int32_t> xsumq((size_t)TB * groups * 2);
    cpu_step_info info{};
    cpu_xq(hx.data(), nullptr, K, K, x8.data(), xmeta.data(), xsumq.data(), &info, TB, K);

    cpu_gemv_seg s{};
    s.w = ti->data;
    s.type = ti->type;
    s.K = K;
    s.n_rows = N;
    s.x = hx.data();
    s.x_stride = K;
    s.out = hy.data();
    s.out_stride = N;
    s.alpha = 1.0f;
    s.i8 = true;
    s.x8 = x8.data();
    s.xmeta = xmeta.data();
    s.xsumq = xsumq.data();
    if (TB == 1) {
        cpu_i8_gemv(s, 0);
    } else {
        cpu_i8_gemm(s, TB, 0);
    }

    std::vector<float> wrow(K);
    double max_err = 0, ref_max = 0;
    const int rstep = N > 4096 ? 97 : 1;
    for (int t = 0; t < TB; t++) {
        for (int r = 0; r < N; r += rstep) {
            dequantize_row(ti->type, (const char *)ti->data + (size_t)r * quant_row_bytes(ti->type, K), wrow.data(), K);
            double ref = 0;
            for (int k = 0; k < K; k++) {
                ref += (double)wrow[k] * hx[(size_t)t * K + k];
            }
            const double err = std::fabs(ref - hy[(size_t)t * N + r]);
            max_err = std::max(max_err, err / std::max(1.0, std::fabs(ref)));
            ref_max = std::max(ref_max, std::fabs(ref));
        }
    }
    // int8 activations add quantization error; allow a few percent
    const bool ok = max_err < 5e-2;
    printf("rel err max=%.3e (|ref|max=%.1f) %s\n", max_err, ref_max, ok ? "OK" : "FAIL");
    if (!ok) {
        failures++;
    }
}

static void check_rmsnorm() {
    const int n = 1024;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n), w(n), out(n);
    double ss = 0;
    for (int i = 0; i < n; i++) {
        x[i] = nd(rng);
        w[i] = nd(rng) * 0.5f + 1.0f;
        ss += (double)x[i] * x[i];
    }
    const float eps = 1e-6f;
    const double inv = 1.0 / std::sqrt(ss / n + eps);
    cpu_rmsnorm(x.data(), w.data(), out.data(), 1, n, eps);
    double max_err = 0;
    for (int i = 0; i < n; i++) {
        const double ref = (double)x[i] * inv * w[i];
        max_err = std::max(max_err, std::fabs(ref - out[i]) / std::max(1e-6, std::fabs(ref)));
    }
    const bool ok = max_err < 1e-5;
    printf("test %-28s rel err max=%.3e %s\n", "cpu_rmsnorm", max_err, ok ? "OK" : "FAIL");
    if (!ok) {
        failures++;
    }
}

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    try {
        model m;
        m.load(model_path);
        printf("model loaded: %d layers, embd=%d\n", m.hp.n_layer, m.hp.n_embd);
        check_rmsnorm();
        check_gemv(m, "token_embd.weight", 1);
        check_gemv(m, "blk.0.ffn_gate.weight", 1);
        check_gemv(m, "blk.0.ffn_down.weight", 1);
        check_gemv(m, "blk.0.attn_qkv.weight", 1);
        check_gemv(m, "blk.3.attn_q.weight", 1);
        check_gemv(m, "blk.3.ffn_down.weight", 1);
        check_gemv(m, "blk.0.ffn_gate.weight", 8);
        check_gemv(m, "blk.3.attn_q.weight", 8);
        check_gemv(m, "token_embd.weight", 16);
        check_gemv(m, "blk.0.ffn_down.weight", 32);
        check_i8(m, "blk.0.ffn_gate.weight", 1);  // Q4_K
        check_i8(m, "blk.0.attn_qkv.weight", 1);  // Q5_K
        check_i8(m, "blk.0.ffn_down.weight", 1);  // Q6_K
        check_i8(m, "blk.3.attn_q.weight", 1);    // Q4_K
        check_i8(m, "blk.0.ffn_gate.weight", 8);
        check_i8(m, "blk.0.ffn_down.weight", 8);
        check_i8(m, "blk.0.ffn_gate.weight", 32);
        check_i8(m, "blk.0.attn_qkv.weight", 32);
        check_i8(m, "blk.0.ffn_down.weight", 32);
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    printf(failures ? "FAILURES: %d\n" : "all ok\n", failures);
    return failures ? 1 : 0;
}
