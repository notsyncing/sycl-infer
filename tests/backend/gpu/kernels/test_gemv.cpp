#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "engine.h"
#include "kernels.h"
#include "model.h"
#include "quant.h"

using namespace si;

static int failures = 0;

static void check_gemv(engine & e, const std::string & name, int TB) {
    const gguf_tensor_info * ti = e.m.gguf.find(name);
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

    float * dx = sycl::malloc_device<float>((size_t)TB * K, e.q);
    float * dy = sycl::malloc_device<float>((size_t)TB * N, e.q);
    e.q.memcpy(dx, hx.data(), hx.size() * 4).wait();

    gemv_seg s{};
    s.w = e.m.dev_ptr(ti->data);
    s.type = ti->type;
    s.K = K;
    s.n_rows = N;
    s.x = dx;
    s.x_stride = K;
    s.act_up = nullptr;
    s.out = dy;
    s.out_stride = N;
    s.residual = nullptr;
    s.alpha = 1.0f;

    // build segments: when TB>1 emulate the prefill slicing (multiple segments per call)
    std::vector<gemv_seg> segs;
    if (TB == 1) {
        segs.push_back(s);
    } else if (TB == 8) {
        const int slice = 8; // engine uses 8-token slices for the prefill path
        for (int sl = 0; sl * slice < TB; sl++) {
            gemv_seg c = s;
            c.x = s.x + (size_t)sl * slice * s.x_stride;
            c.out = s.out + (size_t)sl * slice * s.out_stride;
            segs.push_back(c);
        }
    } else {
        segs.push_back(s); // TB=16/32: single segment (decode-style rows)
    }
    gemv_seg * dsegs = sycl::malloc_device<gemv_seg>(segs.size(), e.q);
    e.q.memcpy(dsegs, segs.data(), segs.size() * sizeof(gemv_seg)).wait();
    int tot = 0;
    for (auto & g : segs) {
        tot += g.n_rows;
    }
    gemv_group_launch(e.q, s.type, dsegs, (int)segs.size(), tot, TB, K / 256);
    e.q.memcpy(hy.data(), dy, hy.size() * 4).wait();

    // cpu reference (sample rows for large N)
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
            if (t == 0 && r < 4) {
                printf("\n   r%d: ref=%.6f got=%.6f", r, ref, hy[(size_t)t * N + r]);
            }
        }
    }
    const bool ok = max_err < 2e-3;
    printf("\n   rel err max=%.3e (|ref|max=%.1f) %s\n", max_err, ref_max, ok ? "OK" : "FAIL");
    if (!ok) {
        failures++;
    }
    sycl::free(dx, e.q);
    sycl::free(dy, e.q);
    sycl::free(dsegs, e.q);
}

int main(int argc, char ** argv) {
    setenv("PF_DP4A", "0", 1); // strict tests validate the fp32 path

    const char * model_path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    try {
        engine_config ec;
        ec.model_path = model_path;
        ec.max_seq = 512;
        engine e(ec);
        printf("model loaded: %d layers, embd=%d\n", e.m.hp.n_layer, e.m.hp.n_embd);

        check_gemv(e, "token_embd.weight", 1);
        check_gemv(e, "blk.0.ffn_gate.weight", 1);
        check_gemv(e, "blk.0.ffn_down.weight", 1);
        check_gemv(e, "blk.0.attn_qkv.weight", 1);
        check_gemv(e, "blk.0.attn_gate.weight", 1);
        check_gemv(e, "blk.0.ssm_beta.weight", 1);
        check_gemv(e, "blk.0.ssm_out.weight", 1);
        check_gemv(e, "blk.3.attn_q.weight", 1);
        check_gemv(e, "blk.3.attn_v.weight", 1);
        check_gemv(e, "blk.3.ffn_down.weight", 1);
        check_gemv(e, "blk.0.ffn_gate.weight", 8);
        check_gemv(e, "blk.0.ffn_down.weight", 8);
        check_gemv(e, "blk.3.attn_q.weight", 8);
        check_gemv(e, "blk.0.attn_qkv.weight", 8);
        check_gemv(e, "blk.0.ssm_beta.weight", 8);
        check_gemv(e, "token_embd.weight", 32);
        check_gemv(e, "blk.0.ffn_gate.weight", 16);
        check_gemv(e, "blk.0.attn_qkv.weight", 16);
        check_gemv(e, "blk.0.ssm_out.weight", 16);
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    printf(failures ? "FAILURES: %d\n" : "all ok\n", failures);
    return failures ? 1 : 0;
}
