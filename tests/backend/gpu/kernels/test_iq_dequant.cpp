// Validates the GPU dequant/GEMV path for the IQ/Q3_K formats against the host
// reference in quant.h.  Run with the Qwen3.8-27B GGUF (the only mixed-quant
// test model); tensors absent from the model are skipped.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels.h"
#include "model.h"
#include "quant.h"

using namespace si;

static int failures = 0;

static void check_tensor(sycl::queue & q, model & m, const char * name, int TB, int rows_req) {
    const gguf_tensor_info * ti = m.gguf.find(name);
    if (!ti) {
        printf("SKIP %s (missing)\n", name);
        return;
    }
    const int K = (int)ti->dims[0];
    const int N = (int)ti->n_rows();
    const int rows = std::min(rows_req, N);
    if (K % 256 != 0) {
        printf("SKIP %s (K%%256)\n", name);
        return;
    }
    const size_t row_bytes = quant_row_bytes(ti->type, K);
    const size_t raw_bytes = row_bytes * (size_t)rows;
    printf("test %-28s type=%-8s K=%-5d rows=%-3d TB=%d ... ", name, ggml_type_name(ti->type), K, rows, TB);
    fflush(stdout);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t)TB * K);
    for (auto & v : hx) {
        v = nd(rng);
    }
    std::vector<float> hy((size_t)TB * rows);

    uint8_t * dw = sycl::malloc_device<uint8_t>(raw_bytes, q);
    float * dx = sycl::malloc_device<float>((size_t)TB * K, q);
    float * dy = sycl::malloc_device<float>((size_t)TB * rows, q);
    q.memcpy(dw, ti->data, raw_bytes).wait();
    q.memcpy(dx, hx.data(), hx.size() * 4).wait();

    gemv_seg s{};
    s.w = dw;
    s.type = ti->type;
    s.K = K;
    s.n_rows = rows;
    s.x = dx;
    s.x_stride = K;
    s.act_up = nullptr;
    s.out = dy;
    s.out_stride = rows;
    s.residual = nullptr;
    s.alpha = 1.0f;
    gemv_seg * dsegs = sycl::malloc_device<gemv_seg>(1, q);
    q.memcpy(dsegs, &s, sizeof(s)).wait();
    gemv_group_launch(q, ti->type, dsegs, 1, rows, TB, K / 256, 0);
    q.memcpy(hy.data(), dy, hy.size() * 4).wait();

    std::vector<float> wrow(K);
    double max_err = 0;
    for (int t = 0; t < TB; t++) {
        for (int r = 0; r < rows; r++) {
            dequantize_row(ti->type, (const char *)ti->data + (size_t)r * row_bytes, wrow.data(), K);
            double ref = 0;
            for (int k = 0; k < K; k++) {
                ref += (double)wrow[k] * hx[(size_t)t * K + k];
            }
            const double err = std::fabs(ref - hy[(size_t)t * rows + r]);
            max_err = std::max(max_err, err / std::max(1.0, std::fabs(ref)));
        }
    }
    const bool ok = max_err < 2e-3;
    printf("rel err max=%.3e %s\n", max_err, ok ? "OK" : "FAIL");
    if (!ok) {
        failures++;
    }
    sycl::free(dw, q);
    sycl::free(dx, q);
    sycl::free(dy, q);
    sycl::free(dsegs, q);
}

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf";
    try {
        model m;
        m.load(model_path);
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
        printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        check_tensor(q, m, "blk.0.ffn_up.weight", 1, 64);    // Q3_K
        check_tensor(q, m, "blk.1.ffn_down.weight", 1, 64);  // IQ4_NL
        check_tensor(q, m, "blk.14.ffn_down.weight", 1, 64); // IQ3_S
        check_tensor(q, m, "blk.0.ffn_gate.weight", 1, 64);  // IQ4_XS
        check_tensor(q, m, "blk.0.ffn_gate.weight", 32, 64);
        check_tensor(q, m, "blk.1.ffn_down.weight", 32, 64);
        check_tensor(q, m, "blk.0.ffn_up.weight", 32, 64);
        // the K-quants and Q8_0 are used by the GDN layer's linears, whose fp32
        // path diverges from int8 at layer 0
        check_tensor(q, m, "blk.0.attn_qkv.weight", 1, 64);  // Q5_K, K=5120 N=10240
        check_tensor(q, m, "blk.0.attn_qkv.weight", 32, 64);
        check_tensor(q, m, "blk.0.ssm_out.weight", 1, 64);   // Q5_K, K=6144
        check_tensor(q, m, "blk.0.ssm_alpha.weight", 1, 48); // Q8_0, N=48
        check_tensor(q, m, "blk.3.ffn_down.weight", 1, 64);  // Q6_K
        // IQ4_XS at large K: this is the exact (type,K) that the engine's fp32
        // path gets wrong at layer 0 (blk.0.ffn_down, K=17408 N=5120)
        check_tensor(q, m, "blk.0.ffn_down.weight", 1, 64);  // IQ4_XS K=17408
        check_tensor(q, m, "blk.0.ffn_down.weight", 32, 64);
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    printf(failures ? "FAILURES: %d\n" : "all ok\n", failures);
    return failures ? 1 : 0;
}
