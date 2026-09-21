// GPU check of the 4-bit (u4) prefill GEMM: compares dnnl_gemm::gemm_w4 against
// an fp32 host reference built from the *true* dequantized Q4_K weights and the
// same quantized activations, so the residual is purely the weight
// representation error (expected ~0.08%, vs ~1% for the int8 conversion).
#include "dnnl_gemm.h"
#include "gguf.h"
#include "quant.h"
#include "w4.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf";
    gguf_file f;
    f.load(path);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    si::dnnl_gemm D(q);

    const int N_CMP = 64; // compare this many output columns (host reference cost)
    std::mt19937 rng(77);
    int checked = 0;
    long samples = 0;
    double sum_rel = 0, worst = 0;

    for (auto & t : f.tensors) {
        if (t.type != 12 || t.dims.empty() || t.dims[0] % QK_K != 0) {
            continue;
        }
        const int K = (int)t.dims[0];
        int N = 1;
        for (size_t i = 1; i < t.dims.size(); i++) {
            N *= (int)t.dims[i];
        }
        if (N < 4096) {
            continue; // want the real prefill shapes
        }
        const char * base = (const char *)f.map_base + f.data_offset + t.offset;
        const void * key = base;
        // register the int8 conversion too, so both paths can be measured
        // against the same fp32 reference
        if (!D.add_weight(key, base, t.type, K, N)) {
            printf("  %-30s int8 add failed\n", t.name.c_str());
            continue;
        }
        if (!D.add_weight_w4(key, base, t.type, K, N)) {
            // legitimately not u4-eligible (N over the f32 accumulator budget,
            // e.g. token_embd) - it keeps the int8 conversion in the engine
            printf("  %-30s K=%5d N=%5d  SKIP (not u4-eligible)\n", t.name.c_str(), K, N);
            continue;
        }
        for (int M : {1, 32, 512}) {
        // random activations in [-1,1], then the same quantization the engine uses
        std::vector<float> x((size_t)M * K);
        for (auto & v : x) {
            v = (float)((int)(rng() % 2001) - 1000) / 1000.f;
        }
        // the kernels take device USM (the engine passes its own device buffers)
        float * xd = sycl::malloc_device<float>((size_t)M * K, q);
        float * od = sycl::malloc_device<float>((size_t)M * N, q);
        q.memcpy(xd, x.data(), (size_t)M * K * 4).wait();
        if (!D.quantize(xd, nullptr, K, 0, M, K)) {
            printf("quantize failed\n");
            return 1;
        }
        float * od8 = sycl::malloc_device<float>((size_t)M * N, q);
        const bool have_i8 = D.gemm(key, nullptr, 1.f, M, K, od8, N);
        std::vector<float> out8;
        if (have_i8) {
            out8.resize((size_t)M * N);
            q.memcpy(out8.data(), od8, (size_t)M * N * 4).wait();
        }
        sycl::free(od8, q);
        if (!D.gemm_w4(key, nullptr, 1.f, M, K, od, N)) {
            // this (M,N) has no primitive (M*N over the f32 accumulator budget)
            printf("  %-30s K=%5d N=%5d M=%3d  SKIP (no primitive)\n", t.name.c_str(), K, N, M);
            sycl::free(xd, q);
            sycl::free(od, q);
            continue;
        }
        std::vector<float> out((size_t)M * N);
        q.memcpy(out.data(), od, (size_t)M * N * 4).wait();
        sycl::free(xd, q);
        sycl::free(od, q);

        // pull the *grouped* quantized activations back: the u4 GEMM consumes
        // the per-32-group form now, so the reference must use it too
        std::vector<int8_t> hxq((size_t)M * K);
        std::vector<uint16_t> hsa((size_t)M * (K / 32));
        q.memcpy(hxq.data(), D.act_grp_data(), (size_t)M * K).wait();
        q.memcpy(hsa.data(), D.act_grp_scales(), (size_t)M * (K / 32) * 2).wait();
        auto h2f = [](uint16_t h) {
            sycl::half x;
            std::memcpy(&x, &h, 2);
            return (float)x;
        };

        // reference: sx[m] * sum_k w_true[k][n] * xq[m][k]
        const size_t rb = ggml_row_bytes(t.type, K);
        std::vector<float> wr((size_t)K);
        double tsum = 0, tworst = 0, ref_last = 0, got_last = 0;
        double tsum8 = 0, tworst8 = 0, tsum_ns = 0;
        std::vector<double> ref_no_sa((size_t)N, 0.0);
        for (int n = 0; n < N_CMP; n++) {
            dequantize_row(t.type, base + (size_t)n * rb, wr.data(), K);
            for (int m = 0; m < M; m++) {
                const int8_t * xr = hxq.data() + (size_t)m * K;
                double acc = 0, acc_no_sa = 0;
                for (int k = 0; k < K; k++) {
                    const double wv = (double)wr[k] * (double)xr[k];
                    acc += wv * (double)h2f(hsa[(size_t)m * (K / 32) + k / 32]);
                    acc_no_sa += wv;
                }
                const double ref = acc;
                if (m == 0 && n < 4) {
                    ref_no_sa[(size_t)n] = acc_no_sa;
                }
                const double got = (double)out[(size_t)m * N + n];
                ref_last = ref;
                got_last = got;
                const double e = std::fabs(got - ref) / std::fmax(1.0, std::fabs(ref));
                if (m == 0) {
                    tsum_ns += std::fabs(got - ref_no_sa[(size_t)n]) / std::fmax(1.0, std::fabs(ref_no_sa[(size_t)n]));
                }
                tsum += e;
                samples++;
                tworst = std::fmax(tworst, e);
                if (have_i8) {
                    const double g8 = (double)out8[(size_t)m * N + n];
                    const double e8 = std::fabs(g8 - ref) / std::fmax(1.0, std::fabs(ref));
                    tsum8 += e8;
                    tworst8 = std::fmax(tworst8, e8);
                }
            }
        }
        sum_rel += tsum;
        worst = std::fmax(worst, tworst);
        printf("  %-26s K=%5d N=%5d M=%3d  u4: mean=%.6f worst=%.6f | u4-no-sa-ref=%.6f | int8: %.6f\n",
               t.name.c_str(), K, N, M, tsum / (double)(N_CMP * M), tworst, tsum_ns / (double)N_CMP,
               tsum8 / (double)(N_CMP * M));
        }
        if (++checked >= 3) {
            break;
        }
    }
    if (!checked) {
        printf("no Q4_K tensor found\n");
        return 1;
    }
    const double mean = sum_rel / (double)std::max(1L, samples);
    printf("\noverall mean rel=%.6f  worst=%.6f\n", mean, worst);
    const bool ok = worst < 0.01;
    printf("%s\n", ok ? "w4 gemm OK" : "w4 gemm FAILED");
    return ok ? 0 : 1;
}
