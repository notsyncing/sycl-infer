// Localises the u4 numerical discrepancy: for every Q4_K tensor, run both the
// int8 GEMM and the u4 GEMM on the *same* quantized activations, and compare.
// The two weight representations differ by ~1% (0.98% int8 vs 0.04% u4), so a
// per-tensor disagreement far above that means the u4 packing or its
// step/offset planes are wrong for that tensor.
#include "dnnl_gemm.h"
#include "gguf.h"
#include "quant.h"
#include "w4.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf";
    const int M = (argc > 2) ? atoi(argv[2]) : 32;
    gguf_file f;
    f.load(path);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    printf("device: %s  M=%d\n", q.get_device().get_info<sycl::info::device::name>().c_str(), M);
    si::dnnl_gemm D(q);

    std::mt19937 rng(31);
    double g_worst = 0;
    int n_tensors = 0, n_bad = 0;
    for (auto & t : f.tensors) {
        if (t.type != 12 || t.dims.empty() || t.dims[0] % QK_K != 0) {
            continue;
        }
        const int K = (int)t.dims[0];
        int N = 1;
        for (size_t i = 1; i < t.dims.size(); i++) {
            N *= (int)t.dims[i];
        }
        const char * base = (const char *)f.map_base + f.data_offset + t.offset;
        const void * key = base;
        if (!D.add_weight(key, base, t.type, K, N)) {
            printf("  %-34s int8 add failed\n", t.name.c_str());
            continue;
        }
        if (!D.add_weight_w4(key, base, t.type, K, N)) {
            printf("  %-34s u4 SKIP (not eligible)\n", t.name.c_str());
            continue;
        }
        n_tensors++;
        std::vector<float> x((size_t)M * K);
        for (auto & v : x) {
            v = (float)((int)(rng() % 2001) - 1000) / 1000.f;
        }
        float * xd = sycl::malloc_device<float>((size_t)M * K, q);
        float * o1 = sycl::malloc_device<float>((size_t)M * N, q);
        float * o2 = sycl::malloc_device<float>((size_t)M * N, q);
        q.memcpy(xd, x.data(), (size_t)M * K * 4).wait();
        const si::act_view av = D.quantize(xd, nullptr, K, 0, M, K);
        const bool ok1 = D.gemm(av, key, nullptr, 1.f, M, K, o1, N);
        const bool ok2 = D.gemm_w4(av, key, nullptr, 1.f, M, K, o2, N);
        if (!ok1 || !ok2) {
            printf("  %-34s gemm int8=%d u4=%d\n", t.name.c_str(), (int)ok1, (int)ok2);
            sycl::free(xd, q);
            sycl::free(o1, q);
            sycl::free(o2, q);
            continue;
        }
        std::vector<float> h1((size_t)M * N), h2((size_t)M * N);
        q.memcpy(h1.data(), o1, (size_t)M * N * 4).wait();
        q.memcpy(h2.data(), o2, (size_t)M * N * 4).wait();
        // compare against the magnitude of the int8 result
        double worst = 0, mean = 0, amax = 0;
        int bad = 0;
        for (size_t i = 0; i < h1.size(); i++) {
            const double a = std::fabs((double)h1[i]);
            amax = std::fmax(amax, a);
            const double d = std::fabs((double)h2[i] - (double)h1[i]);
            mean += d;
            if (d > worst) {
                worst = d;
            }
            if (d > 0.05 * std::fmax(1.0, a)) {
                bad++;
            }
        }
        mean /= (double)h1.size();
        const double rel = amax > 0 ? worst / amax : 0;
        if (rel > 0.05 || bad) {
            n_bad++;
            printf("  %-34s K=%5d N=%5d  BAD worst/|max|=%.4f  bad=%d/%zu  |max|=%.3f\n", t.name.c_str(), K, N, rel,
                   bad, h1.size(), amax);
        }
        g_worst = std::fmax(g_worst, rel);
        sycl::free(xd, q);
        sycl::free(o1, q);
        sycl::free(o2, q);
    }
    printf("\n%d tensors compared, %d flagged, worst worst/|max| = %.4f\n", n_tensors, n_bad, g_worst);
    printf("%s\n", n_bad == 0 ? "u4 vs int8 consistent" : "u4 vs int8 DISAGREE (see flagged tensors)");
    return n_bad ? 1 : 0;
}
