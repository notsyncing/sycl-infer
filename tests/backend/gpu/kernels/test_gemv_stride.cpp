// Probes the fp32 gemv_group kernel with the exact parameter combinations the
// engine uses for a SwiGLU down-projection:
//   * x_stride == K  vs  x_stride == 2*K (the gate/up interleaved layout)
//   * residual == nullptr vs residual == out (the in-place residual add)
// at TB=1 and TB=32.  Isolates which combination the kernel mishandles.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels.h"
#include "model.h"
#include "quant.h"

using namespace si;

static int failures = 0;

static void run(sycl::queue & q, model & m, const char * name, uint32_t type, int TB, int rows, int xmult,
                bool inplace) {
    const gguf_tensor_info * ti = m.gguf.find(name);
    if (!ti) {
        printf("SKIP %s\n", name);
        return;
    }
    const int K = (int)ti->dims[0];
    const size_t row_bytes = quant_row_bytes(ti->type, K);
    const size_t raw_bytes = row_bytes * (size_t)rows;
    const int XS = K * xmult;

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t)TB * XS, 0.f);
    for (int t = 0; t < TB; t++) {
        for (int k = 0; k < K; k++) {
            hx[(size_t)t * XS + k] = nd(rng);
        }
    }
    const float resid0 = 0.5f;

    uint8_t * dw = sycl::malloc_device<uint8_t>(raw_bytes, q);
    float * dx = sycl::malloc_device<float>((size_t)TB * XS, q);
    float * dy = sycl::malloc_device<float>((size_t)TB * rows, q);
    q.memcpy(dw, ti->data, raw_bytes).wait();
    q.memcpy(dx, hx.data(), hx.size() * 4).wait();
    std::vector<float> hy0((size_t)TB * rows, resid0);
    q.memcpy(dy, hy0.data(), hy0.size() * 4).wait();

    gemv_seg s{};
    s.w = dw;
    s.type = ti->type;
    s.K = K;
    s.n_rows = rows;
    s.x = dx;
    s.x_stride = XS;
    s.act_up = nullptr;
    s.out = dy;
    s.out_stride = rows;
    s.residual = inplace ? dy : nullptr;
    s.alpha = 1.0f;
    gemv_seg * dsegs = sycl::malloc_device<gemv_seg>(1, q);
    q.memcpy(dsegs, &s, sizeof(s)).wait();
    gemv_group_launch(q, ti->type, dsegs, 1, rows, TB, K / 256, 0);
    std::vector<float> hy((size_t)TB * rows);
    q.memcpy(hy.data(), dy, hy.size() * 4).wait();

    std::vector<float> wrow(K);
    double max_err = 0;
    for (int t = 0; t < TB; t++) {
        for (int r = 0; r < rows; r++) {
            dequantize_row(ti->type, (const char *)ti->data + (size_t)r * row_bytes, wrow.data(), K);
            double ref = 0;
            for (int k = 0; k < K; k++) {
                ref += (double)wrow[k] * hx[(size_t)t * XS + k];
            }
            if (inplace) {
                ref += resid0;
            }
            const double err = std::fabs(ref - hy[(size_t)t * rows + r]);
            max_err = std::max(max_err, err / std::max(1.0, std::fabs(ref)));
        }
    }
    const bool ok = max_err < 2e-3;
    printf("  %-22s type=%-7s K=%-5d rows=%-5d TB=%-2d xmult=%d inplace=%d ... rel err=%.3e %s\n", name,
           ggml_type_name(type), K, rows, TB, xmult, (int)inplace, max_err, ok ? "OK" : "FAIL");
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
        // the engine's ffn_down call: IQ4_XS K=17408, N=5120, x_stride=2K,
        // residual == out
        for (int TB : {1, 32}) {
            for (int xm : {1, 2}) {
                for (bool ip : {false, true}) {
                    run(q, m, "blk.0.ffn_down.weight", 23, TB, 5120, xm, ip);
                }
            }
        }
        // control: a type/K whose engine call has x_stride == K
        for (int TB : {1, 32}) {
            run(q, m, "blk.0.attn_qkv.weight", 13, TB, 10240, 1, false);
        }
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    printf(failures ? "FAILURES: %d\n" : "all ok\n", failures);
    return failures ? 1 : 0;
}
