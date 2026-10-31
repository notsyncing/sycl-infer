// The formula recogniser's split-K attention PV against a naive f32
// reference.  Hermetic: no model, no graph, one GPU queue.
//
// The kernel is `attn_pv_launch` (src/backend/onnx/kernels.cpp).  The
// reference is the same arithmetic the executor's `matmul_bcast_launch`
// performs for C = P[..., 1, S] x V[..., S, D], accumulated in f32 in key
// order; a second reference accumulates in double, so the tolerance does not
// depend on the summation order.
//
// S = 1: the split count is 1 and the accumulation order is identical, so
// the result must be bit-exact.  Every other shape: rel_rms < 1e-5 against
// the double reference.

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "backend/onnx/kernels.h"

namespace {

void naive_pv(sycl::queue & q, const float * p, const float * v, float * c, int64_t BH,
              int64_t S, int64_t D) {
    q.parallel_for(sycl::range<1>(BH * D), [=](sycl::item<1> it) {
        const int64_t i = it.get_id(0);
        const int64_t bh = i / D;
        const int64_t d = i % D;
        float acc = 0.0f;
        for (int64_t s = 0; s < S; ++s) acc += p[bh * S + s] * v[(bh * S + s) * D + d];
        c[i] = acc;
    });
}

} // namespace

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    std::printf("[pv] %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const int64_t depths[] = {1, 64, 144, 1024, 2500};
    const int64_t dims[] = {32, 64};
    const int64_t heads[] = {8, 16};
    int failures = 0;
    for (int64_t D : dims) {
        for (int64_t BH : heads) {
            for (int64_t S : depths) {
                const size_t np = (size_t)(BH * S);
                const size_t nv = (size_t)(BH * S * D);
                float * dp = sycl::malloc_device<float>(np, q);
                float * dv = sycl::malloc_device<float>(nv, q);
                float * dc = sycl::malloc_device<float>((size_t)(BH * D), q);
                float * dn = sycl::malloc_device<float>((size_t)(BH * D), q);
                const int64_t splits = si::onnx::attn_pv_splits(S);
                float * part = sycl::malloc_device<float>((size_t)(BH * splits * D), q);
                if (!dp || !dv || !dc || !dn || !part) {
                    std::printf("[pv] FAIL D=%lld BH=%lld S=%lld: allocation\n", (long long)D,
                                (long long)BH, (long long)S);
                    ++failures;
                    return 1;
                }
                std::vector<float> hp(np), hv(nv);
                std::mt19937 rng(1234 + (unsigned)S * 17 + (unsigned)D * 7 + (unsigned)BH);
                std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
                for (float & x : hp) x = dist(rng);
                for (float & x : hv) x = dist(rng);
                q.memcpy(dp, hp.data(), np * sizeof(float));
                q.memcpy(dv, hv.data(), nv * sizeof(float));

                naive_pv(q, dp, dv, dn, BH, S, D);
                si::onnx::attn_pv_launch(q, dp, dv, dc, part, BH, S, D);
                std::vector<float> hc((size_t)(BH * D)), hn((size_t)(BH * D));
                q.memcpy(hc.data(), dc, hc.size() * sizeof(float));
                q.memcpy(hn.data(), dn, hn.size() * sizeof(float));
                q.wait();

                double ref_energy = 0.0, err_energy = 0.0, err_naive = 0.0;
                bool exact = true;
                for (int64_t bh = 0; bh < BH; ++bh) {
                    for (int64_t d = 0; d < D; ++d) {
                        double ref = 0.0;
                        for (int64_t s = 0; s < S; ++s) {
                            ref += (double)hp[(size_t)(bh * S + s)] *
                                   (double)hv[(size_t)((bh * S + s) * D + d)];
                        }
                        const double got = (double)hc[(size_t)(bh * D + d)];
                        const double nv = (double)hn[(size_t)(bh * D + d)];
                        const double e = got - ref;
                        err_energy += e * e;
                        ref_energy += ref * ref;
                        const double en = nv - ref;
                        err_naive += en * en;
                        if (hc[(size_t)(bh * D + d)] != hn[(size_t)(bh * D + d)]) exact = false;
                    }
                }
                const double rel = std::sqrt(err_energy / (ref_energy > 0 ? ref_energy : 1.0));
                const double rel_naive = std::sqrt(err_naive / (ref_energy > 0 ? ref_energy : 1.0));
                const bool ok = (S == 1 ? exact : rel < 1e-5);
                std::printf("[pv] D=%lld BH=%lld S=%lld splits=%lld rel_rms=%.3e "
                            "naive_rel=%.3e exact=%d %s\n",
                            (long long)D, (long long)BH, (long long)S, (long long)splits, rel,
                            rel_naive, (int)exact, ok ? "OK" : "FAIL");
                if (!ok) ++failures;
                sycl::free(dp, q);
                sycl::free(dv, q);
                sycl::free(dc, q);
                sycl::free(dn, q);
                sycl::free(part, q);
            }
        }
    }
    std::printf("[pv] %s (%d failure(s))\n", failures ? "FAILED" : "all OK", failures);
    return failures ? 1 : 0;
}
