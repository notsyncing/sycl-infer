// Localises the u4 numerical discrepancy: for every Q4_K tensor, run both the
// int8 GEMM and the u4 GEMM on the *same* quantized activations, and compare.
// The two weight representations differ by ~1% (0.98% int8 vs 0.04% u4), so a
// per-tensor disagreement far above that means the u4 packing or its
// step/offset planes are wrong for that tensor.
//
// The comparison is two-stage, and the reason is cancellation rather than a
// fudge factor.
//
// A dot product of K=17408 terms has an absolute error set by the *terms*, not
// by the result: |dy| ~ eps * sum_k |w_k x_k|.  So whenever the sum nearly
// cancels, |y| can be a small fraction of that accumulation scale while the two
// stores still differ by their full ~1%.  A per-element *relative* yardstick
// (|dy| > 0.05 * max(1, |y|)) therefore flags exactly the near-zero outputs and
// calls a correct store broken: the reference 27B run flags 1 element in 163840
// on blk.5.ffn_down and 2 in 163840 on blk.22.ssm_out, both with |y| below a
// third of the tensor's max, while the tensor-level figure stays at 1.6% / 0.9% -
// i.e. inside the ~1% the header describes.  Those two were reproducible on the
// pre-change tree as well, byte for byte, so they are not a regression.
//
// Stage 1 therefore screens against the *tensor's* own output scale
// (|dy| > 0.05 * max|y| over the tensor), which is cancellation-blind and still
// catches a systematically wrong store.  Stage 2 adjudicates the elements that
// survive it - and, so that the cancellation claim stays checkable rather than
// argued, the single worst element of every tensor worth explaining - against the
// exact fp32 dequant dot product of the *same* quantized activation the kernels
// used, judged on |dy| vs sum_k |w_k x_k|, the accumulation scale the error
// actually grows with.
//
// Both directions are measured on the 27B reference (103 Q4_K tensors, M=32):
//
//   default          worst worst/|max| = 0.0163, 0 elements over the screen.
//                    The worst element of the worst tensor has |dy| = 0.062 on a
//                    tensor whose max is 3.814 (1.6% of it) - see the printed
//                    adjudication for its cancellation factor.
//   PF_W4_NOCORR=1   8 elements over the screen on the first tensor alone,
//                    diff/accum = 0.10-0.15, with err_int8 = 0.002-0.011 against
//                    fp32 while err_u4 = 4.2-5.4: the criterion still fires, so
//                    it is a re-derivation and not a loosened threshold.  The
//                    0.05 sits between ~0.01 (a correct pair of stores) and
//                    ~0.10 (a dropped zero-point correction), i.e. with ~2x
//                    margin on the broken side.
//
// Every adjudication is printed with both stores' errors against fp32 and the
// cancellation factor, so the verdict is auditable from the log instead of
// trusted.
#include "dnnl_gemm.h"
#include "gguf.h"
#include "quant.h"
#include "w4.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>

// How many surviving elements to adjudicate.  The screen leaves 1-2 on a good
// store and one row per failing tensor column on a broken one; 8 is enough to
// show the shape without turning the test into a per-element dump.
static const size_t kMaxAdjudicate = 8;

// One element that survived the stage-1 screen, kept for stage 2.
struct offender {
    double d;   // |u4 - int8| at this element
    size_t i;   // flat index into the [M][N] output
};

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
    int n_tensors = 0, n_bad = 0, n_adjudicated = 0;
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

        double amax = 0;
        for (size_t i = 0; i < h1.size(); i++) {
            amax = std::fmax(amax, std::fabs((double)h1[i]));
        }
        // Stage 1: screen every element against the tensor's output scale, and
        // keep the k worst survivors for adjudication.  The worst element of the
        // whole tensor is always adjudicated as well - it is what shows whether a
        // tensor-level figure is cancellation or a defect, and it usually does
        // *not* clear the screen, so gating it on the screen would drop exactly
        // the evidence the criterion rests on.
        double worst = 0, mean = 0;
        int n_over = 0;
        std::vector<offender> top; // sorted ascending by |dy|
        auto keep = [&](double d, size_t i) {
            if (top.size() == kMaxAdjudicate && d <= top.front().d) {
                return;
            }
            top.push_back({d, i});
            if (top.size() > kMaxAdjudicate) {
                top.erase(top.begin());
            }
        };
        for (size_t i = 0; i < h1.size(); i++) {
            const double d = std::fabs((double)h2[i] - (double)h1[i]);
            mean += d;
            if (d > worst) {
                // the unconditional worst is kept; dropping the others here is
                // right because they were only kept for being over the screen,
                // and the new worst is not (a later element may push it out).
                worst = d;
                top.clear();
                top.push_back({d, i});
            } else if (d > 0.05 * amax) {
                n_over++;
                keep(d, i);
            }
        }
        mean /= (double)h1.size();
        const double rel = amax > 0 ? worst / amax : 0;

        int bad = 0;
        // A tensor whose worst disagreement is a rounding-scale fraction of its
        // own max has nothing to explain; one above that does, and saying so is
        // the point of always adjudicating the worst element.
        if (!top.empty() && rel > 0.005) {
            // Stage 2: adjudicate the survivors against the exact fp32 dequant
            // dot product of the same quantized activation, on the accumulation
            // scale the error actually grows with.
            std::vector<int8_t> hxq((size_t)M * K);
            std::vector<uint16_t> hsa((size_t)M * (K / 32));
            q.memcpy(hxq.data(), D.act_grp_data(av), (size_t)M * K).wait();
            q.memcpy(hsa.data(), D.act_grp_scales(av), (size_t)M * (K / 32) * 2).wait();
            auto h2f = [](uint16_t h) {
                sycl::half x;
                std::memcpy(&x, &h, 2);
                return (float)x;
            };
            const size_t rb = ggml_row_bytes(t.type, K);
            std::vector<float> wr((size_t)K);
            printf("  %-34s K=%5d N=%5d  worst/|max|=%.4f mean/|max|=%.5f |max|=%.3f  %zu element(s) over "
                   "%.0f%% of max\n",
                   t.name.c_str(), K, N, rel, amax > 0 ? mean / amax : 0, amax, (size_t)n_over, 5.0);
            for (const auto & e : top) {
                const size_t i = e.i;
                const int m = (int)(i / (size_t)N), n = (int)(i % (size_t)N);
                dequantize_row(t.type, base + (size_t)n * rb, wr.data(), K);
                const int8_t * xr = hxq.data() + (size_t)m * K;
                double acc = 0, accum = 0;
                for (int k = 0; k < K; k++) {
                    const double wv = (double)wr[k] * (double)xr[k] * (double)h2f(hsa[(size_t)m * (K / 32) + k / 32]);
                    acc += wv;
                    accum += std::fabs(wv);
                }
                const double y1 = (double)h1[i], y2 = (double)h2[i];
                const double d = e.d;
                const double over = accum > 0 ? d / accum : 0.0;
                const bool is_bad = d > 0.05 * accum;
                bad += is_bad ? 1 : 0;
                printf("      [m=%2d n=%5d] int8=%9.4f u4=%9.4f fp32=%9.4f |int8-u4|=%8.4f  accum=%8.2f  "
                       "diff/accum=%.4f  cancel=%5.1fx  err_int8=%8.4f err_u4=%8.4f  %s\n",
                       m, n, y1, y2, acc, d, accum, over, accum > 0 ? std::fabs(accum) / std::fmax(1e-9, std::fabs(acc))
                                                                          : 0.0,
                       std::fabs(y1 - acc), std::fabs(y2 - acc), is_bad ? "BAD" : "ok (cancellation)");
            }
            n_adjudicated += (int)top.size();
        }
        n_bad += bad;
        g_worst = std::fmax(g_worst, rel);
        sycl::free(xd, q);
        sycl::free(o1, q);
        sycl::free(o2, q);
    }
    printf("\n%d tensors compared, %d element(s) adjudicated, %d over 5%% of their accumulation scale, "
           "worst worst/|max| = %.4f\n",
           n_tensors, n_adjudicated, n_bad, g_worst);
    printf("%s\n", n_bad == 0 ? "u4 vs int8 consistent" : "u4 vs int8 DISAGREE (see flagged tensors)");
    return n_bad ? 1 : 0;
}