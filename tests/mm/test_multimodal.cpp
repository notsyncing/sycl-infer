// Multimodal module tests: image preprocessing geometry, the host vision
// encoder (shape/finiteness) and the prompt/position layout produced by
// mm_build_prompt.  The vision encoder is validated against the llama.cpp
// reference out of band; this test guards the integration logic.
//
// usage: test_multimodal [text.gguf] [mmproj.gguf]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "image.h"
#include "kernels.h"
#include "multimodal.h"
#include "tokenizer.h"
#include "vision.h"

using namespace si;

static int fails = 0;

static void check(bool ok, const char * what) {
    printf("  %-46s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) fails++;
}

static void test_target_size() {
    printf("image target size\n");
    image_preproc_cfg cfg;
    cfg.patch_size = 16;
    cfg.merge = 2;
    cfg.min_pixels = 8192;
    cfg.max_pixels = 4194304;
    int w = 0, h = 0;
    // 768x768 is already aligned and inside the pixel budget
    mm_image_target_size(768, 768, cfg, w, h);
    check(w == 768 && h == 768, "768x768 stays 768x768");
    // both sides snap to a multiple of patch*merge = 32
    mm_image_target_size(1069, 893, cfg, w, h);
    check(w % 32 == 0 && h % 32 == 0, "1069x893 aligns to 32");
    check(w == 1056 && h == 896, "1069x893 -> 1056x896");
    // a tiny image is upscaled above min_pixels
    mm_image_target_size(16, 16, cfg, w, h);
    check(w * h >= cfg.min_pixels && w % 32 == 0 && h % 32 == 0, "16x16 upscaled above min_pixels");
}

static void test_vision(const std::string & mmproj_path) {
    printf("vision encoder\n");
    vision_model vm;
    try {
        vm.load(mmproj_path);
    } catch (const std::exception & ex) {
        printf("  skip (cannot load %s: %s)\n", mmproj_path.c_str(), ex.what());
        return;
    }
    image_preproc_cfg cfg;
    cfg.patch_size = vm.hp.patch_size;
    cfg.merge = vm.hp.merge;
    const int patch_area = cfg.patch_size * cfg.patch_size * cfg.merge * cfg.merge;
    cfg.min_pixels = 8 * patch_area;
    cfg.max_pixels = kMaxImgTokens * patch_area;
    for (int c = 0; c < 3; c++) {
        cfg.mean[c] = vm.hp.mean[c];
        cfg.std[c] = vm.hp.std[c];
    }

    // deterministic synthetic image
    const int W = 96, H = 96;
    std::vector<uint8_t> rgb((size_t) W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            uint8_t * p = rgb.data() + ((size_t) y * W + x) * 3;
            p[0] = (uint8_t) ((x * 255) / (W - 1));
            p[1] = (uint8_t) ((y * 255) / (H - 1));
            p[2] = (uint8_t) (((x + y) * 255) / (W + H - 2));
        }
    mm_image img = mm_image_preprocess(rgb.data(), W, H, cfg);
    check(img.width == 96 && img.height == 96, "preprocessed size");
    vision_input vin = vision_model::make_input(vm, img);
    check(vin.n_patches == 36, "patch count = 6x6");
    check(vin.n_out == 9, "merged tokens = 3x3");

    std::vector<float> embd;
    vm.encode_host(vin, embd);
    check((int) embd.size() == vin.n_out * vm.hp.proj_dim, "embedding shape n_out*proj_dim");
    bool finite = true;
    double norm = 0;
    for (float v : embd) {
        if (!std::isfinite(v)) finite = false;
        norm += (double) v * v;
    }
    check(finite, "embeddings finite");
    check(norm > 0.0, "embeddings non-zero");
}

static void test_prompt(const std::string & text_path, const std::string & mmproj_path) {
    printf("prompt layout\n");
    gguf_file gf;
    try {
        gf.load(text_path);
    } catch (const std::exception & ex) {
        printf("  skip (cannot load %s: %s)\n", text_path.c_str(), ex.what());
        return;
    }
    tokenizer tk;
    tk.load(gf);
    if (tk.token_to_id.find("<|image_pad|>") == tk.token_to_id.end()) {
        printf("  skip (tokenizer has no <|image_pad|>)\n");
        return;
    }
    vision_model vm;
    try {
        vm.load(mmproj_path);
    } catch (const std::exception & ex) {
        printf("  skip (cannot load mmproj: %s)\n", ex.what());
        return;
    }
    image_preproc_cfg cfg;
    cfg.patch_size = vm.hp.patch_size;
    cfg.merge = vm.hp.merge;
    const int patch_area = cfg.patch_size * cfg.patch_size * cfg.merge * cfg.merge;
    cfg.min_pixels = 8 * patch_area;
    cfg.max_pixels = kMaxImgTokens * patch_area;
    for (int c = 0; c < 3; c++) {
        cfg.mean[c] = vm.hp.mean[c];
        cfg.std[c] = vm.hp.std[c];
    }
    const int W = 96, H = 96;
    std::vector<uint8_t> rgb((size_t) W * H * 3, 128);
    mm_image img = mm_image_preprocess(rgb.data(), W, H, cfg);
    vision_input vin = vision_model::make_input(vm, img);

    const std::string rendered = "<|vision_start|><|image_pad|><|vision_end|>hello";
    mm_prompt mp = mm_build_prompt(tk, rendered, {img}, vm, vm.hp.proj_dim);

    check(mp.n_img == 1, "one image");
    check((int) mp.img_row.size() == (int) mp.tokens.size(), "img_row covers every token");
    check(mp.mrope.size() == 4 * mp.tokens.size(), "mrope is 4 positions per token");
    check(mp.embd.size() == (size_t) vin.n_out * vm.hp.proj_dim, "cloned image embeddings");
    check(mp.pos_after == 1 + std::max(vin.out_w, vin.out_h) + 2,
          "image consumes max(nx,ny) positions");
    // the single image region starts after <|vision_start|>
    const int n = (int) mp.tokens.size();
    check(mp.img_row[1] == 0 && mp.img_row[vin.n_out + 1] == -1, "image rows mapped then cleared");
    bool pos_ok = true;
    for (int t = 0; t < vin.n_out; t++) {
        if (mp.mrope[t + 1] != 1) pos_ok = false;                        // temporal = base
        if (mp.mrope[n + t + 1] != 1 + t / vin.out_w) pos_ok = false;    // row
        if (mp.mrope[2 * n + t + 1] != 1 + t % vin.out_w) pos_ok = false; // col
    }
    check(pos_ok, "image M-RoPE (t,row,col) layout");
}

static void test_device(const std::string & mmproj_path) {
    printf("device vision encoder\n");
    vision_model vm;
    try {
        vm.load(mmproj_path);
    } catch (const std::exception & ex) {
        printf("  skip (cannot load mmproj: %s)\n", ex.what());
        return;
    }
    image_preproc_cfg cfg;
    cfg.patch_size = vm.hp.patch_size;
    cfg.merge = vm.hp.merge;
    const int patch_area = cfg.patch_size * cfg.patch_size * cfg.merge * cfg.merge;
    cfg.min_pixels = 8 * patch_area;
    cfg.max_pixels = kMaxImgTokens * patch_area;
    for (int c = 0; c < 3; c++) {
        cfg.mean[c] = vm.hp.mean[c];
        cfg.std[c] = vm.hp.std[c];
    }
    // a larger, deterministic image so the comparison covers many patch tokens
    const int W = 256, H = 256;
    std::vector<uint8_t> rgb((size_t) W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            uint8_t * p = rgb.data() + ((size_t) y * W + x) * 3;
            p[0] = (uint8_t) ((x * 255) / (W - 1));
            p[1] = (uint8_t) ((y * 255) / (H - 1));
            p[2] = (uint8_t) (((x * 3 + y * 5) * 255) / (W * 3 + H * 5));
        }
    mm_image img = mm_image_preprocess(rgb.data(), W, H, cfg);
    vision_input vin = vision_model::make_input(vm, img);

    std::vector<float> host;
    const auto th0 = std::chrono::steady_clock::now();
    vm.encode_host(vin, host);
    const double host_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - th0).count();

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    float * d = sycl::malloc_device<float>(host.size(), q);
    check(d != nullptr, "device allocation");
    if (!d) return;
    const auto tc0 = std::chrono::steady_clock::now();
    vm.encode_device(q, vin, d); // warm up (kernel JIT)
    const double cold_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - tc0).count();
    double best = 1e9;
    for (int rep = 0; rep < 5; rep++) {
        const auto t0 = std::chrono::steady_clock::now();
        vm.encode_device(q, vin, d);
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::vector<float> dev(host.size());
    q.memcpy(dev.data(), d, host.size() * sizeof(float)).wait();

    double maxd = 0, maxv = 0;
    bool finite = true;
    for (size_t i = 0; i < host.size(); i++) {
        if (!std::isfinite(dev[i])) finite = false;
        maxd = std::max(maxd, (double) std::fabs(dev[i] - host[i]));
        maxv = std::max(maxv, (double) std::fabs(host[i]));
    }
    const bool ok = finite && maxd <= 5e-3 * std::max(1.0, maxv);
    printf("  %-46s max|host-dev|=%.5f (max|host|=%.4f) %s\n", "device matches host reference", maxd,
           maxv, ok ? "OK" : "FAIL");
    if (!ok) fails++;
    printf("  encode %dx%d (%d patches, %d merged): host %.1f ms, device cold %.1f ms / warm %.2f ms (%.1fx)\n",
           img.width, img.height, vin.n_patches, vin.n_out, host_ms, cold_ms, best,
           host_ms / std::max(1e-3, best));
    sycl::free(d, q);
}

// Kernel-level checks against direct host references, so a failure points at
// one kernel instead of the whole tower.
static void kern_cmp(const char * what, const std::vector<float> & got,
                     const std::vector<float> & exp, double tol) {
    double maxd = 0, maxv = 0;
    for (size_t i = 0; i < got.size(); i++) {
        maxd = std::max(maxd, (double) std::fabs(got[i] - exp[i]));
        maxv = std::max(maxv, (double) std::fabs(exp[i]));
    }
    const bool ok = maxd <= tol * std::max(1.0, maxv);
    printf("  %-46s max|diff|=%.6f (max=%.4f) %s\n", what, maxd, maxv, ok ? "OK" : "FAIL");
    if (!ok) fails++;
}

static void test_kernels() {
    printf("vision kernels\n");
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    auto rnd = [](int i) {
        unsigned v = (unsigned) i * 2654435761u;
        v ^= v >> 13;
        v *= 2246822519u;
        return (float) (v % 2001) / 1000.f - 1.f;
    };

    { // GEMM
        const int N = 24, K = 200, T = 5;
        std::vector<float> W((size_t) N * K), x((size_t) T * K), exp((size_t) T * N), got((size_t) T * N);
        for (int i = 0; i < N * K; i++) W[i] = rnd(i + 1);
        for (int i = 0; i < T * K; i++) x[i] = rnd(i + 7);
        for (int t = 0; t < T; t++)
            for (int n = 0; n < N; n++) {
                float a = 0;
                for (int k = 0; k < K; k++) a += W[(size_t) n * K + k] * x[(size_t) t * K + k];
                exp[(size_t) t * N + n] = a;
            }
        float *dW = sycl::malloc_device<float>(W.size(), q), *dx = sycl::malloc_device<float>(x.size(), q);
        float *dout = sycl::malloc_device<float>(got.size(), q);
        q.memcpy(dW, W.data(), W.size() * 4);
        q.memcpy(dx, x.data(), x.size() * 4);
        vit_gemm_launch(q, dW, GGML_TYPE_F32, N, K, dx, K, dout, N, T, 1.f, nullptr);
        q.memcpy(got.data(), dout, got.size() * 4).wait();
        kern_cmp("vit_gemm", got, exp, 1e-5);
        sycl::free(dW, q); sycl::free(dx, q); sycl::free(dout, q);
    }
    { // LayerNorm
        const int rows = 3, n = 96;
        std::vector<float> x((size_t) rows * n), w(n), b(n), exp((size_t) rows * n), got((size_t) rows * n);
        for (int i = 0; i < rows * n; i++) x[i] = rnd(i + 3);
        for (int i = 0; i < n; i++) { w[i] = rnd(i + 11) * 0.5f + 1.f; b[i] = rnd(i + 13) * 0.2f; }
        for (int r = 0; r < rows; r++) {
            const float * xr = x.data() + (size_t) r * n;
            double s = 0; for (int i = 0; i < n; i++) s += xr[i];
            double mean = s / n, var = 0;
            for (int i = 0; i < n; i++) { double d = xr[i] - mean; var += d * d; }
            var /= n;
            const float inv = 1.f / std::sqrt((float) var + 1e-6f);
            for (int i = 0; i < n; i++)
                exp[(size_t) r * n + i] = (xr[i] - (float) mean) * inv * w[i] + b[i];
        }
        float *dx = sycl::malloc_device<float>(x.size(), q), *dw = sycl::malloc_device<float>(n, q);
        float *db = sycl::malloc_device<float>(n, q), *dout = sycl::malloc_device<float>(got.size(), q);
        q.memcpy(dx, x.data(), x.size() * 4);
        q.memcpy(dw, w.data(), n * 4);
        q.memcpy(db, b.data(), n * 4);
        vit_layernorm_launch(q, dx, n, dw, db, dout, n, rows, n, 1e-6f);
        q.memcpy(got.data(), dout, got.size() * 4).wait();
        kern_cmp("vit_layernorm", got, exp, 1e-4);
        sycl::free(dx, q); sycl::free(dw, q); sycl::free(db, q); sycl::free(dout, q);
    }
    { // vision RoPE on a fused qkv buffer
        const int n_tok = 12, n_head = 2, HD = 64, embd = n_head * HD;
        const int out_w = 2, merge = 2;
        std::vector<float> qkv((size_t) n_tok * 3 * embd), exp;
        for (size_t i = 0; i < qkv.size(); i++) qkv[i] = rnd((int) i + 5);
        exp = qkv;
        const float log2b = std::log2(10000.f);
        for (int t = 0; t < n_tok; t++) {
            const int m = t / (merge * merge), sub = t % (merge * merge);
            const int py = (m / out_w) * merge + sub / merge;
            const int px = (m % out_w) * merge + sub % merge;
            for (int h = 0; h < n_head; h++) {
                for (int pair = 0; pair < HD / 2; pair++) {
                    const int sec = pair / (HD / 4), p = pair % (HD / 4);
                    const float th = (float) (sec == 0 ? py : px) *
                                     std::exp2(-2.f * p / (float) (HD / 2) * log2b);
                    const float c = std::cos(th), s = std::sin(th);
                    for (int half = 0; half < 2; half++) {
                        float * row = exp.data() + (size_t) t * 3 * embd +
                                      (half ? embd : 0) + h * HD;
                        const float a = row[pair], bb = row[pair + HD / 2];
                        row[pair] = a * c - bb * s;
                        row[pair + HD / 2] = a * s + bb * c;
                    }
                }
            }
        }
        float * d = sycl::malloc_device<float>(qkv.size(), q);
        q.memcpy(d, qkv.data(), qkv.size() * 4);
        vit_rope_launch(q, d, 3 * embd, n_tok, n_head, HD, out_w, merge, 10000.f);
        std::vector<float> got(qkv.size());
        q.memcpy(got.data(), d, got.size() * 4).wait();
        kern_cmp("vit_rope", got, exp, 1e-5);
        sycl::free(d, q);
    }
    { // bidirectional attention
        const int n_tok = 200, n_head = 2, HD = 64, embd = n_head * HD;
        std::vector<float> qkv((size_t) n_tok * 3 * embd), got((size_t) n_tok * embd);
        for (size_t i = 0; i < qkv.size(); i++) qkv[i] = rnd((int) i + 17) * 0.3f;
        std::vector<float> exp((size_t) n_tok * embd, 0.f);
        const float scale = 1.f / std::sqrt((float) HD);
        for (int h = 0; h < n_head; h++) {
            for (int i = 0; i < n_tok; i++) {
                std::vector<float> sc(n_tok);
                float mx = -1e30f;
                for (int j = 0; j < n_tok; j++) {
                    float a = 0;
                    for (int d = 0; d < HD; d++)
                        a += qkv[(size_t) i * 3 * embd + h * HD + d] *
                             qkv[(size_t) j * 3 * embd + embd + h * HD + d];
                    sc[j] = a * scale;
                    mx = std::max(mx, sc[j]);
                }
                float sum = 0;
                for (int j = 0; j < n_tok; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                for (int d = 0; d < HD; d++) {
                    float a = 0;
                    for (int j = 0; j < n_tok; j++)
                        a += sc[j] * qkv[(size_t) j * 3 * embd + 2 * embd + h * HD + d];
                    exp[(size_t) i * embd + h * HD + d] = a / sum;
                }
            }
        }
        float * d = sycl::malloc_device<float>(qkv.size(), q), *dout = sycl::malloc_device<float>(got.size(), q);
        q.memcpy(d, qkv.data(), qkv.size() * 4);
        vit_attn_launch(q, d, 3 * embd, dout, embd, n_tok, n_head, HD, scale);
        q.memcpy(got.data(), dout, got.size() * 4).wait();
        kern_cmp("vit_attn", got, exp, 1e-4);
        sycl::free(d, q); sycl::free(dout, q);
    }
}

static void bench_kernels() {
    printf("kernel timings (T=256 tokens)\n");
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    const int T = 256, E = 768, ff = 3072, NH = 12, HD = 64, embd = NH * HD;
    auto timeit = [&](const char * what, auto && fn) {
        fn(); // warm
        double best = 1e9;
        for (int r = 0; r < 5; r++) {
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            q.wait();
            best = std::min(best, std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - t0).count());
        }
        printf("  %-24s %.3f ms\n", what, best);
    };
    struct B { float *p; };
    std::vector<B> keep;
    auto alloc = [&](size_t n) { float * p = sycl::malloc_device<float>(n, q); keep.push_back({p}); return p; };
    float * x = alloc((size_t) T * E), * qt = alloc((size_t) T * 3 * E), * attn = alloc((size_t) T * E);
    float * up = alloc((size_t) ff * E), * down = alloc((size_t) E * ff), * ffn = alloc((size_t) T * ff);
    float * qk = alloc((size_t) T * 3 * embd);
    std::vector<float> hz((size_t) ff * E, 0.01f);
    q.memcpy(up, hz.data(), hz.size() * 4); q.memcpy(down, hz.data(), hz.size() * 4);

    timeit("gemm qkv 2304x768", [&] {
        vit_gemm_launch(q, down, GGML_TYPE_F32, 3 * E, E, x, E, qt, 3 * E, T, 1.f, nullptr);
    });
    timeit("gemm ffn_up 3072x768", [&] {
        vit_gemm_launch(q, up, GGML_TYPE_F32, ff, E, x, E, ffn, ff, T, 1.f, nullptr);
    });
    timeit("gemm ffn_down 768x3072", [&] {
        vit_gemm_launch(q, down, GGML_TYPE_F32, E, ff, ffn, ff, x, E, T, 1.f, nullptr);
    });
    timeit("rope", [&] { vit_rope_launch(q, qk, 3 * embd, T, NH, HD, 8, 2, 10000.f); });
    timeit("attn n_tok=256", [&] { vit_attn_launch(q, qk, 3 * embd, attn, E, T, NH, HD, 0.125f); });
    timeit("layernorm", [&] { vit_layernorm_launch(q, x, E, up, up, attn, E, T, E, 1e-6f); });
    timeit("gelu", [&] { vit_gelu_launch(q, ffn, T * ff); });
    timeit("add_bias", [&] { vit_add_bias_launch(q, qt, 3 * E, x, T, 3 * E); });
    for (auto & b : keep) sycl::free(b.p, q);

}

int main(int argc, char ** argv) {
    const std::string text_path =
        argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    const std::string mmproj_path =
        argc > 2 ? argv[2] : "/home/sfc/临时/Qwen3.5-0.8B-mmproj-BF16.gguf";
    test_target_size();
    test_kernels();
    bench_kernels();
    test_vision(mmproj_path);
    test_device(mmproj_path);
    test_prompt(text_path, mmproj_path);
    if (fails) {
        printf("test_multimodal: FAILURES: %d\n", fails);
        return 1;
    }
    printf("test_multimodal: all OK\n");
    return 0;
}
