// Loads the model once and evaluates prompts of increasing length to locate a
// failure threshold / hang in the prefill path (used for the 27B bring-up).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine.h"
#include "common/env.h"

using namespace si;

int main(int argc, char ** argv) {
    if (si::env::flag("TEST_FP32")) {
        setenv("PF_DP4A", "0", 1); // strict fp32 comparison path
    }
    const char * model_path = argc > 1 ? argv[1] : "/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf";
    const char * lm = si::env::str("TEST_LAYER_MAP");
    const char * dv = si::env::str("TEST_DEVICE");
    const int dev = (dv && strcmp(dv, "cpu") == 0) ? 1 : -1;
    try {
        engine_config ec;
        ec.model_path = model_path;
        ec.max_seq = 2048;
        ec.n_blocks = 512;
        ec.device = dev;
        ec.layer_map = lm ? lm : "";
        engine e(ec);
        std::vector<int> lens;
        for (int i = 2; i < argc; i++) {
            lens.push_back(atoi(argv[i]));
        }
        if (lens.empty()) {
            lens = {8, 64, 480, 512, 544};
        }
        for (int n : lens) {
            std::vector<int> toks((size_t)n, 9419);
            printf("eval n=%d ... ", n);
            fflush(stdout);
            auto t0 = std::chrono::high_resolution_clock::now();
            auto lg = e.eval(toks);
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();
            int best = 0;
            for (size_t i = 1; i < lg.size(); i++) {
                if (lg[i] > lg[best]) best = (int)i;
            }
            printf("done top=%d (%.4f)  %.3fs = %.1f tok/s\n", best, lg[best], dt, n / dt);
            fflush(stdout);
        }
        // decode throughput: small prompt, then 128 greedy steps
        if (!si::env::flag("TEST_SKIP_DECODE")) {
            gen_params gp;
            gp.max_tokens = 128;
            gp.temperature = 0.f;
            std::vector<int> prompt(8, 9419);
            auto t0 = std::chrono::high_resolution_clock::now();
            e.generate(prompt, gp, [](int) { return true; });
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();
            printf("decode 128 tok: %.3fs = %.2f tok/s (incl. 8-token prefill)\n", dt, 128 / dt);
            fflush(stdout);
        }
        // isolated dp4a_gemm benchmark on one device (TB=32, kMaxT chunk)
        if (si::env::flag("TEST_KERNEL_BENCH")) {
            sycl::queue & q = e.q;
            const int TB = 32;
            const char * names[] = {"ffn_up", "ffn_down", "wqkv", "wgate", "ssm_out"};
            const wt * ts[] = {&e.m.layers[0].ffn_up, &e.m.layers[0].ffn_down, &e.m.layers[0].wqkv,
                               &e.m.layers[0].wgate, &e.m.layers[0].ssm_out};
            (void)names;
            // raw DRAM read ceiling (the decode GEMV's upper bound)
            {
                const size_t nb = (size_t)1024 * 1024 * 1024; // 1 GiB
                uint8_t * buf = sycl::malloc_device<uint8_t>(nb, q);
                q.memset(buf, 1, nb).wait();
                const int wgsz = 256, nwg = 4096;
                auto rd = [&]() {
                    q.parallel_for(sycl::nd_range<1>((size_t)nwg * wgsz, wgsz), [=](sycl::nd_item<1> it) {
                        const sycl::uint4 * p = reinterpret_cast<const sycl::uint4 *>(buf);
                        const size_t n4 = nb / 16;
                        uint32_t acc = 0;
                        for (size_t i = it.get_global_id(0); i < n4; i += (size_t)nwg * wgsz) {
                            sycl::uint4 v = p[i];
                            acc += v.x() + v.y() + v.z() + v.w();
                        }
                        if (acc == 0xdeadbeefu) {
                            buf[0] = (uint8_t)acc;
                        }
                    });
                };
                rd();
                q.wait();
                auto b0 = std::chrono::high_resolution_clock::now();
                for (int rep = 0; rep < 5; rep++) {
                    rd();
                }
                q.wait();
                auto b1 = std::chrono::high_resolution_clock::now();
                const double bms = std::chrono::duration<double, std::milli>(b1 - b0).count() / 5;
                printf("  dram_read 1GiB: %.3f ms  %.1f GB/s\n", bms, (double)nb / (bms * 1e-3) / 1e9);
                fflush(stdout);
                sycl::free(buf, q);
            }
            for (const wt * t : ts) {
                if (!t->data) {
                    continue;
                }
                w8t w;
                e.m.build_w8_one(q, *t, w);
                if (!w.vals) {
                    continue;
                }
                const int K = w.K, N = w.N;
                int8_t * x8 = sycl::malloc_device<int8_t>((size_t)TB * K, q);
                sycl::float2 * xm = sycl::malloc_device<sycl::float2>((size_t)TB * (K / 32), q);
                int32_t * xs = sycl::malloc_device<int32_t>((size_t)TB * (K / 16), q);
                float * out = sycl::malloc_device<float>((size_t)TB * N, q);
                q.memset(x8, 0, (size_t)TB * K).wait();
                q.memset(xm, 0, (size_t)TB * (K / 32) * sizeof(sycl::float2)).wait();
                q.memset(xs, 0, (size_t)TB * (K / 16) * 4).wait();
                const int iters = 20;
                for (int i = 0; i < 3; i++) {
                    dp4a_gemm_launch(q, w, x8, xm, xs, out, N, nullptr, 1.f, TB);
                }
                q.wait();
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iters; i++) {
                    dp4a_gemm_launch(q, w, x8, xm, xs, out, N, nullptr, 1.f, TB);
                }
                q.wait();
                auto t1 = std::chrono::high_resolution_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
                const double bytes = (double)K * N * (w.type == 12 ? 0.5 : (w.type == 13 ? 0.625 : 0.75));
                printf("  dp4a_gemm K=%5d N=%5d type=%u: %.3f ms/call  %.1f GB/s  %.2f TFLOP/s\n", K, N, w.type, ms,
                       bytes / (ms * 1e-3) / 1e9, 2.0 * K * N * TB / (ms * 1e-3) / 1e12);
                fflush(stdout);
                // the decode path on the same tensor: SI8 single-token GEMV.
                // GF: what the int8 decode GEMV (oneDNN buffer) costs for the
                // same tensor, so the two representations can be compared.
                const double wbytes =
                    (double)w8_vals_bytes(w.type, K, N) + (double)w8_meta_count(w.type, K, N) * (double)w.meta_elem;
                for (int i = 0; i < 3; i++) {
                    dp4a_gemv_launch(q, w, x8, xm, xs, out, nullptr, 1.f, N);
                }
                q.wait();
                auto g0 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iters; i++) {
                    dp4a_gemv_launch(q, w, x8, xm, xs, out, nullptr, 1.f, N);
                }
                q.wait();
                auto g1 = std::chrono::high_resolution_clock::now();
                const double gms = std::chrono::duration<double, std::milli>(g1 - g0).count() / iters;
                printf("  dp4a_gemv K=%5d N=%5d type=%u meta=%d: %.3f ms/call  %.1f GB/s  (%.1f MB)\n", K, N, w.type,
                       w.meta_elem, gms, wbytes / (gms * 1e-3) / 1e9, wbytes / 1e6);
                fflush(stdout);
                // the production decode GEMV: plain int8 rows from the oneDNN
                // conversion, so the two decode representations are comparable.
                if (dnnl_gemm * D = e.dnnl_for(0)) {
                    const int8_t * wq = D->weight_data((const void *)t->data);
                    const float * wsc = D->weight_scales((const void *)t->data);
                    if (wq && wsc) {
                        float * sx = sycl::malloc_device<float>(1, q);
                        q.memset(sx, 0, 4).wait();
                        for (int i = 0; i < 3; i++) {
                            i8_row_gemv_launch(q, wq, wsc, x8, sx, out, K, N, nullptr, 1.f);
                        }
                        q.wait();
                        auto i0 = std::chrono::high_resolution_clock::now();
                        for (int i = 0; i < iters; i++) {
                            i8_row_gemv_launch(q, wq, wsc, x8, sx, out, K, N, nullptr, 1.f);
                        }
                        q.wait();
                        auto i1 = std::chrono::high_resolution_clock::now();
                        const double ims = std::chrono::duration<double, std::milli>(i1 - i0).count() / iters;
                        const double ib = (double)K * (double)N;
                        printf("  i8_gemv   K=%5d N=%5d            : %.3f ms/call  %.1f GB/s  (%.1f MB)\n", K, N, ims,
                               ib / (ims * 1e-3) / 1e9, ib / 1e6);
                        fflush(stdout);
                        // the production decode kernel (sub-group per row, dp4a)
                        {
                            gemv_seg sg{};
                            sg.wi8 = wq;
                            sg.wsc = wsc;
                            sg.K = K;
                            sg.n_rows = N;
                            sg.out = out;
                            sg.out_stride = N;
                            sg.alpha = 1.f;
                            for (int i = 0; i < 3; i++) {
                                i8_row_gemv_multi_launch(q, &sg, 1, N, x8, sx, nullptr);
                            }
                            q.wait();
                            auto m0 = std::chrono::high_resolution_clock::now();
                            for (int i = 0; i < iters; i++) {
                                i8_row_gemv_multi_launch(q, &sg, 1, N, x8, sx, nullptr);
                            }
                            q.wait();
                            auto m1 = std::chrono::high_resolution_clock::now();
                            const double mms = std::chrono::duration<double, std::milli>(m1 - m0).count() / iters;
                            printf("  i8_multi  K=%5d N=%5d            : %.3f ms/call  %.1f GB/s\n", K, N, mms,
                                   ib / (mms * 1e-3) / 1e9);
                            fflush(stdout);
                        }
                        sycl::free(sx, q);
                    }
                }
                sycl::free(x8, q);
                sycl::free(xm, q);
                sycl::free(xs, q);
                sycl::free(out, q);
            }
        }
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    printf("all lengths ok\n");
    return 0;
}
