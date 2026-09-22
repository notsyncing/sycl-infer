// GPU check of the native-width 5-bit (Q5_K) decode GEMV: compares
// k5_gemv_launch against an exact host reference built from the *native* Q5_K
// weights (the dequantizer's own q5 extraction) and the same integer
// activations, so the only residual is the f16 rounding of the two per-(g,n)
// metadata planes - the same convention the u4 path uses.
#include "gguf.h"
#include "kernels.h"
#include "quant.h"
#include "w4.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>

static const int kMaxRows = 96; // host reference cost is O(rows*K)

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf";
    gguf_file f;
    f.load(path);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(4242);
    int checked = 0, failed = 0;
    double worst = 0, worst_abs = 0;

    for (auto & t : f.tensors) {
        if (t.type != 13 || t.dims.empty() || t.dims[0] % QK_K != 0) {
            continue;
        }
        const int K = (int)t.dims[0];
        int N = 1;
        for (size_t i = 1; i < t.dims.size(); i++) {
            N *= (int)t.dims[i];
        }
        if (N < 1024) {
            continue;
        }
        const char * base = (const char *)f.map_base + f.data_offset + t.offset;
        si::k5t k5;
        if (!si::k5_pack(13, base, K, N, k5)) {
            printf("  %-26s K=%5d N=%5d pack FAILED\n", t.name.c_str(), K, N);
            failed++;
            continue;
        }
        // activations: small deterministic int8 values, so the integer dot is
        // exact and the host reference is cheap to write
        std::vector<int8_t> axg((size_t)K);
        for (int k = 0; k < K; k++) {
            axg[(size_t)k] = (int8_t)((int)((k * 7919) % 21) - 10);
        }
        std::vector<int8_t> axe((size_t)K / 2), axo((size_t)K / 2);
        for (int g = 0; g < K / 32; g++) {
            for (int j = 0; j < 16; j++) {
                axe[(size_t)g * 16 + j] = axg[(size_t)g * 32 + 2 * j];
                axo[(size_t)g * 16 + j] = axg[(size_t)g * 32 + 2 * j + 1];
            }
        }
        std::vector<uint16_t> asa((size_t)K / 32, 0x3C00); // 1.0
        std::vector<float> xs((size_t)K / 32, 0.f);
        for (int g = 0; g < K / 32; g++) {
            int s = 0;
            for (int j = 0; j < 32; j++) {
                s += axg[(size_t)g * 32 + j];
            }
            xs[(size_t)g] = (float)s;
        }
        uint8_t * d_lo = sycl::malloc_device<uint8_t>(k5.vals.size(), q);
        uint8_t * d_hi = sycl::malloc_device<uint8_t>(k5.hi.size(), q);
        uint16_t * d_sc = sycl::malloc_device<uint16_t>(k5.scale.size(), q);
        uint16_t * d_off = sycl::malloc_device<uint16_t>(k5.off.size(), q);
        int8_t * d_axe = sycl::malloc_device<int8_t>((size_t)K / 2, q);
        int8_t * d_axo = sycl::malloc_device<int8_t>((size_t)K / 2, q);
        uint16_t * d_asa = sycl::malloc_device<uint16_t>((size_t)K / 32, q);
        float * d_xs = sycl::malloc_device<float>((size_t)K / 32, q);
        float * d_out = sycl::malloc_device<float>((size_t)N, q);
        q.memcpy(d_lo, k5.vals.data(), k5.vals.size());
        q.memcpy(d_hi, k5.hi.data(), k5.hi.size());
        q.memcpy(d_sc, k5.scale.data(), k5.scale.size() * 2);
        q.memcpy(d_off, k5.off.data(), k5.off.size() * 2);
        q.memcpy(d_axe, axe.data(), axe.size());
        q.memcpy(d_axo, axo.data(), axo.size());
        q.memcpy(d_asa, asa.data(), asa.size() * 2);
        q.memcpy(d_xs, xs.data(), xs.size() * 4);
        q.memset(d_out, 0, (size_t)N * 4).wait();
        si::k5_gemv_launch(q, d_lo, d_hi, d_sc, d_off, d_axe, d_axo, d_asa, d_xs, d_out, nullptr, 1.f, K, N);
        q.wait();
        std::vector<float> out((size_t)N);
        q.memcpy(out.data(), d_out, (size_t)N * 4).wait();

        // host reference: for a row sample, walk the *native* blocks and use the
        // dequantizer's own q5 rule plus the packed f16 step/off
        const size_t row_bytes = quant_row_bytes(13, K);
        const int nblocks = K / QK_K;
        double tmax = 0, tmax_abs = 0;
        std::vector<int> rows;
        for (int i = 0; i < kMaxRows && i < N; i++) {
            rows.push_back(i);
        }
        rows.push_back(N - 1);
        for (int n : rows) {
            const uint8_t * row = (const uint8_t *)base + (size_t)n * row_bytes;
            double ref = 0;
            for (int b = 0; b < nblocks; b++) {
                const block_q5_K * x = (const block_q5_K *)row + b;
                const uint8_t * qs = x->qs;
                const uint8_t * qh = x->qh;
                int is = 0, p = 0;
                for (int j = 0; j < QK_K; j += 64) {
                    uint8_t sc0, m0, sc1, m1;
                    get_scale_min_k4(is + 0, x->scales, &sc0, &m0);
                    get_scale_min_k4(is + 1, x->scales, &sc1, &m1);
                    const int g = (b * QK_K + j) / 32;
                    const float step0 = ggml_half_to_float(k5.scale[(size_t)g * N + n]);
                    const float off0 = ggml_half_to_float(k5.off[(size_t)g * N + n]);
                    const float step1 = ggml_half_to_float(k5.scale[(size_t)(g + 1) * N + n]);
                    const float off1 = ggml_half_to_float(k5.off[(size_t)(g + 1) * N + n]);
                    double dot0 = 0, dot1 = 0;
                    for (int l = 0; l < 32; l++) {
                        const int q0 = (qs[l] & 0xF) + (((qh[l] >> p) & 1) << 4);
                        const int q1 = ((qs[l] >> 4) & 0xF) + (((qh[l] >> (p + 1)) & 1) << 4);
                        dot0 += (double)q0 * (double)axg[(size_t)(b * QK_K + j + l)];
                        dot1 += (double)q1 * (double)axg[(size_t)(b * QK_K + j + 32 + l)];
                    }
                    ref += (double)step0 * dot0 + (double)off0 * (double)xs[(size_t)g];
                    ref += (double)step1 * dot1 + (double)off1 * (double)xs[(size_t)(g + 1)];
                    qs += 32;
                    is += 2;
                    p += 2;
                }
            }
            const double got = (double)out[(size_t)n];
            const double e = std::fabs(got - ref) / std::fmax(1.0, std::fabs(ref));
            tmax = std::fmax(tmax, e);
            tmax_abs = std::fmax(tmax_abs, std::fabs(got - ref));
        }
        const bool ok = tmax < 1e-4;
        printf("  %-26s K=%5d N=%5d  max rel=%.3g  max abs=%.3g  %s\n", t.name.c_str(), K, N, tmax, tmax_abs,
               ok ? "OK" : "FAIL");
        if (!ok) {
            failed++;
        }
        worst = std::fmax(worst, tmax);
        worst_abs = std::fmax(worst_abs, tmax_abs);
        checked++;
        sycl::free(d_lo, q);
        sycl::free(d_hi, q);
        sycl::free(d_sc, q);
        sycl::free(d_off, q);
        sycl::free(d_axe, q);
        sycl::free(d_axo, q);
        sycl::free(d_asa, q);
        sycl::free(d_xs, q);
        sycl::free(d_out, q);
        if (checked >= 4) {
            break;
        }
    }
    if (!checked) {
        printf("no Q5_K tensor found\n");
        return 1;
    }
    printf("\nchecked %d tensors, worst rel %.3g (max abs %.3g)\n", checked, worst, worst_abs);
    const bool ok = failed == 0;
    printf("%s\n", ok ? "k5 gemv OK" : "k5 gemv FAILED");
    return ok ? 0 : 1;
}
