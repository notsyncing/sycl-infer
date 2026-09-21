// Per-type quantization audit: for one tensor of each ggml type used by the
// model, compare the int8 (oneDNN) GEMM against a host fp32 reference built
// from the *true* dequantized weights and the same quantized activations.  This
// isolates how lossy the int8 conversion is for each type - e.g. types whose
// native granularity is finer than the conversion's per-row scale (Q8_0, Q6_K)
// should come out nearly exact while a per-row scale on Q4_K loses ~0.5%.
#include "dnnl_gemm.h"
#include "gguf.h"
#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>

static const char * kWant[] = {
    "blk.0.attn_qkv.weight",  // Q5_K
    "blk.0.ssm_alpha.weight", // Q8_0
    "blk.0.ffn_gate.weight",  // IQ4_XS
    "blk.0.ffn_up.weight",    // Q3_K
    "blk.0.ffn_down.weight",  // IQ4_XS K=17408
    "blk.1.ffn_down.weight",  // IQ4_NL
    "blk.1.attn_qkv.weight",  // Q4_K
    "blk.3.ffn_down.weight",  // IQ4_NL / Q6_K
    "blk.5.attn_qkv.weight",  // whatever type
    "blk.10.ffn_down.weight",
};

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf";
    gguf_file f;
    f.load(path);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    si::dnnl_gemm D(q);
    const int M = 32;
    std::mt19937 rng(99);

    for (const char * name : kWant) {
        const gguf_tensor_info * ti = f.find(name);
        if (!ti) {
            printf("  %-26s SKIP (missing)\n", name);
            continue;
        }
        const int K = (int)ti->dims[0];
        int N = (int)ti->n_rows();
        if (K % 256 != 0) {
            printf("  %-26s SKIP K%%256\n", name);
            continue;
        }
        const char * base = (const char *)f.map_base + f.data_offset + ti->offset;
        const void * key = base;
        if (!D.add_weight(key, base, ti->type, K, N)) {
            printf("  %-26s SKIP (no int8)\n", name);
            continue;
        }
        std::vector<float> x((size_t)M * K);
        for (auto & v : x) {
            v = (float)((int)(rng() % 2001) - 1000) / 1000.f;
        }
        float * xd = sycl::malloc_device<float>((size_t)M * K, q);
        float * od = sycl::malloc_device<float>((size_t)M * N, q);
        q.memcpy(xd, x.data(), (size_t)M * K * 4).wait();
        D.quantize(xd, nullptr, K, 0, M, K);
        if (!D.gemm(key, nullptr, 1.f, M, K, od, N)) {
            printf("  %-26s SKIP (gemm)\n", name);
            sycl::free(xd, q);
            sycl::free(od, q);
            continue;
        }
        std::vector<float> out((size_t)M * N);
        q.memcpy(out.data(), od, (size_t)M * N * 4).wait();
        // the int8 path now consumes the per-32-group quantization, so the
        // reference must use it too
        std::vector<int8_t> hxq((size_t)M * K);
        std::vector<uint16_t> hsa((size_t)M * (K / 32));
        q.memcpy(hxq.data(), D.act_grp_data(), (size_t)M * K).wait();
        q.memcpy(hsa.data(), D.act_grp_scales(), (size_t)M * (K / 32) * 2).wait();
        auto h2f = [](uint16_t h) {
            sycl::half x;
            std::memcpy(&x, &h, 2);
            return (float)x;
        };

        // reference over all columns of row 0 (N can be large; M=32 is enough)
        const size_t rb = ggml_row_bytes(ti->type, K);
        std::vector<float> wr((size_t)K);
        std::vector<double> ref((size_t)N, 0.0);
        for (int n = 0; n < N; n++) {
            dequantize_row(ti->type, base + (size_t)n * rb, wr.data(), K);
            double acc = 0;
            const int8_t * xr = hxq.data();
            for (int k = 0; k < K; k++) {
                acc += (double)wr[k] * (double)xr[k] * (double)h2f(hsa[(size_t)(k / 32)]);
            }
            ref[(size_t)n] = acc;
        }
        double sum = 0, worst = 0, l1 = 0;
        for (int n = 0; n < N; n++) {
            const double got = (double)out[(size_t)n]; // row 0
            const double d = std::fabs(got - ref[(size_t)n]);
            sum += d;
            l1 += std::fabs(ref[(size_t)n]);
            worst = std::fmax(worst, d / std::fmax(1.0, std::fabs(ref[(size_t)n])));
        }
        printf("  %-26s type=%-7s K=%-5d N=%-6d  mean|d|/mean|ref|=%.4f  max rel=%.4f\n", name,
               ggml_type_name(ti->type), K, N, sum / std::fmax(1.0, l1), worst);
        sycl::free(xd, q);
        sycl::free(od, q);
    }
    return 0;
}
