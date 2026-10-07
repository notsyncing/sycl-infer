// Multimodal module tests: image preprocessing geometry, the host vision
// encoder (shape/finiteness) and the prompt/position layout produced by
// mm_build_prompt.  The vision encoder is validated against the llama.cpp
// reference out of band; this test guards the integration logic.
//
// usage: test_multimodal [text.gguf] [mmproj.gguf]
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio.h"
#include "audio_model.h"
#include "image.h"
#include "kernels.h"
#include "multimodal.h"
#include "tokenizer.h"
#include "video.h"
#include "vision.h"

using namespace si;

static int fails = 0;

static void check(bool ok, const char * what) {
    printf("  %-46s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) {
        fails++;
    }
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

static void test_vision_width_guard() {
    vision_model vm;
    vm.hp.proj_dim = 7;
    tokenizer tk;
    bool rejected = false;
    try {
        mm_build_prompt(tk, "", {}, vm, /*text n_embd=*/8);
    } catch (const std::runtime_error & ex) {
        rejected = std::string(ex.what()).find("output width") != std::string::npos;
    }
    check(rejected, "vision width mismatch rejected before copying embeddings");
}

// ---------------------------------------------------------------------------
// Video decode (hermetic: synthetic AVI built in memory, no model files)
// ---------------------------------------------------------------------------
static void avi_put32(std::vector<uint8_t> & b, uint32_t v) {
    for (int i = 0; i < 4; i++) {
        b.push_back((uint8_t)(v >> (8 * i)));
    }
}

static void avi_put16(std::vector<uint8_t> & b, uint32_t v) {
    b.push_back((uint8_t)v);
    b.push_back((uint8_t)(v >> 8));
}

// fourcc, little-endian size, data, pad to even
static void avi_chunk(std::vector<uint8_t> & b, const char * id, const std::vector<uint8_t> & data) {
    b.insert(b.end(), id, id + 4);
    avi_put32(b, (uint32_t)data.size());
    b.insert(b.end(), data.begin(), data.end());
    if (data.size() & 1) {
        b.push_back(0);
    }
}

static void avi_list(std::vector<uint8_t> & b, const char * type, const std::vector<uint8_t> & data) {
    std::vector<uint8_t> body(type, type + 4);
    body.insert(body.end(), data.begin(), data.end());
    avi_chunk(b, "LIST", body);
}

// One video stream of uncompressed BI_RGB frames, preceded by an audio stream
// (so the video is stream 1: "01db" chunks, and "00wb" audio chunks to skip).
static std::vector<uint8_t> make_avi(int w, int h, int bpp, const std::vector<std::vector<uint8_t>> & frames) {
    std::vector<uint8_t> avih;
    avi_put32(avih, 100000); // 10 fps
    for (int i = 0; i < 13; i++) {
        avi_put32(avih, 0);
    }
    std::vector<uint8_t> strl_a, strh_a(56, 0), strf_a(16, 0);
    std::memcpy(strh_a.data(), "auds", 4);
    avi_chunk(strl_a, "strh", strh_a);
    avi_chunk(strl_a, "strf", strf_a);
    std::vector<uint8_t> strl_v, strh_v(56, 0), strf_v;
    std::memcpy(strh_v.data(), "vids", 4);
    avi_put32(strf_v, 40);
    avi_put32(strf_v, (uint32_t)w);
    avi_put32(strf_v, (uint32_t)h);
    avi_put16(strf_v, 1);
    avi_put16(strf_v, (uint32_t)bpp);
    avi_put32(strf_v, 0); // BI_RGB
    for (int i = 0; i < 5; i++) {
        avi_put32(strf_v, 0);
    }
    avi_chunk(strl_v, "strh", strh_v);
    avi_chunk(strl_v, "strf", strf_v);
    std::vector<uint8_t> hdrl;
    avi_chunk(hdrl, "avih", avih);
    avi_list(hdrl, "strl", strl_a);
    avi_list(hdrl, "strl", strl_v);
    std::vector<uint8_t> movi;
    avi_chunk(movi, "JUNK", std::vector<uint8_t>(5, 0xEE)); // odd size: padded
    for (const auto & f : frames) {
        avi_chunk(movi, "00wb", std::vector<uint8_t>(7, 0x55)); // audio, odd size
        avi_chunk(movi, "01db", f);
    }
    std::vector<uint8_t> body{'A', 'V', 'I', ' '};
    avi_list(body, "hdrl", hdrl);
    avi_list(body, "movi", movi);
    avi_chunk(body, "idx1", std::vector<uint8_t>(16, 0));
    std::vector<uint8_t> file{'R', 'I', 'F', 'F'};
    avi_put32(file, (uint32_t)body.size());
    file.insert(file.end(), body.begin(), body.end());
    return file;
}

static void test_video_avi() {
    printf("video: synthetic AVI\n");
    // 3x2 bottom-up 24-bit: each DIB row is 9 bytes padded to 12
    const int W = 3, H = 2;
    std::vector<std::vector<uint8_t>> frames;
    for (int t = 0; t < 3; t++) {
        std::vector<uint8_t> f(12 * H, 0xAA); // padding stays 0xAA
        for (int y = 0; y < H; y++) {      // stored row y = image row H-1-y
            for (int x = 0; x < W; x++) {
                const int iy = H - 1 - y;
                // BGR order in the file
                f[(size_t)y * 12 + x * 3 + 0] = (uint8_t)(30 * t + 3);      // B
                f[(size_t)y * 12 + x * 3 + 1] = (uint8_t)(10 * iy + x);     // G
                f[(size_t)y * 12 + x * 3 + 2] = (uint8_t)(100 + t);         // R
            }
        }
        frames.push_back(f);
    }
    const std::vector<uint8_t> avi = make_avi(W, H, 24, frames);
    mm_video_fmt fmt;
    fmt.max_frames = 16;
    mm_video v;
    std::string err;
    bool ok = mm_video_decode_mem(avi.data(), avi.size(), fmt, v, &err);
    check(ok && v.frames.size() == 3, "3 frames decoded");
    if (ok && v.frames.size() == 3) {
        check(v.width() == W && v.height() == H, "geometry 3x2 (odd width)");
        bool px = true;
        for (int t = 0; t < 3; t++) {
            const auto & f = v.frames[t];
            px = px && f.rgb.size() == (size_t)W * H * 3;
            for (int y = 0; y < H && px; y++) {
                for (int x = 0; x < W; x++) {
                    const uint8_t * p = &f.rgb[((size_t)y * W + x) * 3];
                    if (p[0] != 100 + t || p[1] != 10 * y + x || p[2] != 30 * t + 3) {
                        px = false;
                    }
                }
            }
        }
        check(px, "pixels: BGR->RGB, row padding, bottom-up");
        check(std::fabs(v.frames[2].pts - 0.2) < 1e-6, "pts from avih frame rate");
    }
    // uniform sampling to max_frames
    fmt.max_frames = 2;
    mm_video v2;
    ok = mm_video_decode_mem(avi.data(), avi.size(), fmt, v2, &err);
    check(ok && v2.frames.size() == 2 && v2.frames[1].rgb[0] == 102, "max_frames=2 samples frames 0 and 2");
    // max_frames <= 0 is rejected before any division
    for (int mf : {0, -4}) {
        fmt.max_frames = mf;
        mm_video v3;
        err.clear();
        ok = mm_video_decode_mem(avi.data(), avi.size(), fmt, v3, &err);
        check(!ok && v3.frames.empty() && !err.empty(), "max_frames <= 0 fails cleanly");
    }
    fmt.max_frames = 16;
    // a file cut in the middle of the last frame keeps the complete ones
    {
        std::vector<uint8_t> cut(avi.begin(), avi.end());
        // find the third "01db" chunk and truncate inside its data
        size_t pos = 0;
        int seen = 0;
        for (size_t i = 12; i + 4 <= cut.size(); i++) {
            if (std::memcmp(cut.data() + i, "01db", 4) == 0 && ++seen == 3) {
                pos = i;
                break;
            }
        }
        cut.resize(pos + 8 + 5);
        mm_video v4;
        ok = mm_video_decode_mem(cut.data(), cut.size(), fmt, v4, &err);
        check(ok && v4.frames.size() == 2, "truncated last frame is dropped");
    }
    // 16-bit BI_RGB is 5-5-5, rows padded to 4 bytes (3 px = 6 -> 8 bytes)
    {
        std::vector<uint8_t> f(8 * 2, 0);
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 3; x++) {
                const uint16_t val = (uint16_t)(31 << 10); // pure red in 5-5-5
                f[(size_t)y * 8 + x * 2] = (uint8_t)val;
                f[(size_t)y * 8 + x * 2 + 1] = (uint8_t)(val >> 8);
            }
        }
        const std::vector<uint8_t> a16 = make_avi(3, 2, 16, {f});
        mm_video v5;
        ok = mm_video_decode_mem(a16.data(), a16.size(), fmt, v5, &err);
        check(ok && v5.frames.size() == 1 && v5.frames[0].rgb[0] == 255 && v5.frames[0].rgb[1] == 0
                  && v5.frames[0].rgb[2] == 0,
              "16-bit BI_RGB decodes as 5-5-5");
    }
    // chunk sizes that run far past the buffer must neither crash nor hang
    {
        std::vector<uint8_t> bad = avi;
        // the first chunk inside the RIFF body is "LIST hdrl": corrupt its size
        const uint32_t huge = 0xFFFFFFF0u;
        std::memcpy(bad.data() + 16, &huge, 4);
        mm_video v6;
        ok = mm_video_decode_mem(bad.data(), bad.size(), fmt, v6, &err);
        check(!ok || v6.frames.size() <= 3, "oversized hdrl chunk is bounded");
        std::vector<uint8_t> bad2 = avi;
        std::memcpy(bad2.data() + 4, &huge, 4); // bogus RIFF size is ignored
        mm_video v7;
        ok = mm_video_decode_mem(bad2.data(), bad2.size(), fmt, v7, &err);
        check(ok && v7.frames.size() == 3, "bogus RIFF size does not matter");
    }
}

// ffmpeg forces even sides; the raw frame read must use the filtered size.
static void test_video_ffmpeg_odd() {
    printf("video: ffmpeg odd-size clip\n");
    const char * ff = std::getenv("PF_AV_FFMPEG");
    const std::string ffmpeg = (ff && ff[0]) ? ff : "ffmpeg";
    const std::string tmp = (std::filesystem::temp_directory_path()
                             / ("test_mm_odd_" + std::to_string((long)getpid()) + ".nut")).string();
    const std::string gen = ffmpeg + " -v error -y -f lavfi -i \"color=c=red:s=33x21:r=5:d=1\" -c:v rawvideo "
                            "-pix_fmt rgb24 \"" + tmp + "\" 2>/dev/null";
    const int rc = std::system(gen.c_str());
    if (rc != 0) {
        printf("  skip (ffmpeg not available or cannot create the clip)\n");
        std::remove(tmp.c_str());
        return;
    }
    mm_video_fmt fmt;
    fmt.max_frames = 4;
    mm_video v;
    std::string err;
    const bool ok = mm_video_decode_file(tmp, fmt, v, &err);
    std::remove(tmp.c_str());
    check(ok && !v.frames.empty(), "33x21 clip decodes");
    if (ok && !v.frames.empty()) {
        const auto & f = v.frames[0];
        check(f.width == 32 && f.height == 20 && f.rgb.size() == (size_t)32 * 20 * 3, "frame geometry is the even 32x20");
        check(f.rgb.size() >= 3 && f.rgb[0] > 200 && f.rgb[1] < 60 && f.rgb[2] < 60, "pixels are not shifted/garbled");
        check(v.frames.size() <= 4, "at most max_frames");
    }
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
    std::vector<uint8_t> rgb((size_t)W * H * 3);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t * p = rgb.data() + ((size_t)y * W + x) * 3;
            p[0] = (uint8_t)((x * 255) / (W - 1));
            p[1] = (uint8_t)((y * 255) / (H - 1));
            p[2] = (uint8_t)(((x + y) * 255) / (W + H - 2));
        }
    }
    mm_image img = mm_image_preprocess(rgb.data(), W, H, cfg);
    check(img.width == 96 && img.height == 96, "preprocessed size");
    vision_input vin = vision_model::make_input(vm, img);
    check(vin.n_patches == 36, "patch count = 6x6");
    check(vin.n_out == 9, "merged tokens = 3x3");

    std::vector<float> embd;
    vm.encode_host(vin, embd);
    check((int)embd.size() == vin.n_out * vm.hp.proj_dim, "embedding shape n_out*proj_dim");
    bool finite = true;
    double norm = 0;
    for (float v : embd) {
        if (!std::isfinite(v)) {
            finite = false;
        }
        norm += (double)v * v;
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
    std::vector<uint8_t> rgb((size_t)W * H * 3, 128);
    mm_image img = mm_image_preprocess(rgb.data(), W, H, cfg);
    vision_input vin = vision_model::make_input(vm, img);

    const std::string rendered = "<|vision_start|><|image_pad|><|vision_end|>hello";
    mm_prompt mp = mm_build_prompt(tk, rendered, {img}, vm, vm.hp.proj_dim);

    check(mp.n_img == 1, "one image");
    check((int)mp.img_row.size() == (int)mp.tokens.size(), "img_row covers every token");
    check(mp.mrope.size() == 4 * mp.tokens.size(), "mrope is 4 positions per token");
    check(mp.embd.size() == (size_t)vin.n_out * vm.hp.proj_dim, "cloned image embeddings");
    check(mp.pos_after == 1 + std::max(vin.out_w, vin.out_h) + 2, "image consumes max(nx,ny) positions");
    // the single image region starts after <|vision_start|>
    const int n = (int)mp.tokens.size();
    check(mp.img_row[1] == 0 && mp.img_row[vin.n_out + 1] == -1, "image rows mapped then cleared");
    bool pos_ok = true;
    for (int t = 0; t < vin.n_out; t++) {
        if (mp.mrope[t + 1] != 1) {
            pos_ok = false; // temporal = base
        }
        if (mp.mrope[n + t + 1] != 1 + t / vin.out_w) {
            pos_ok = false; // row
        }
        if (mp.mrope[2 * n + t + 1] != 1 + t % vin.out_w) {
            pos_ok = false; // col
        }
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
    std::vector<uint8_t> rgb((size_t)W * H * 3);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t * p = rgb.data() + ((size_t)y * W + x) * 3;
            p[0] = (uint8_t)((x * 255) / (W - 1));
            p[1] = (uint8_t)((y * 255) / (H - 1));
            p[2] = (uint8_t)(((x * 3 + y * 5) * 255) / (W * 3 + H * 5));
        }
    }
    mm_image img = mm_image_preprocess(rgb.data(), W, H, cfg);
    vision_input vin = vision_model::make_input(vm, img);

    std::vector<float> host;
    const auto th0 = std::chrono::steady_clock::now();
    vm.encode_host(vin, host);
    const double host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - th0).count();

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    float * d = sycl::malloc_device<float>(host.size(), q);
    check(d != nullptr, "device allocation");
    if (!d) {
        return;
    }
    const auto tc0 = std::chrono::steady_clock::now();
    vm.encode_device(q, vin, d); // warm up (kernel JIT)
    const double cold_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count();
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
        if (!std::isfinite(dev[i])) {
            finite = false;
        }
        maxd = std::max(maxd, (double)std::fabs(dev[i] - host[i]));
        maxv = std::max(maxv, (double)std::fabs(host[i]));
    }
    const bool ok = finite && maxd <= 5e-3 * std::max(1.0, maxv);
    printf("  %-46s max|host-dev|=%.5f (max|host|=%.4f) %s\n", "device matches host reference", maxd, maxv,
           ok ? "OK" : "FAIL");
    if (!ok) {
        fails++;
    }
    printf("  encode %dx%d (%d patches, %d merged): host %.1f ms, device cold %.1f ms / warm %.2f ms (%.1fx)\n",
           img.width, img.height, vin.n_patches, vin.n_out, host_ms, cold_ms, best, host_ms / std::max(1e-3, best));
    sycl::free(d, q);
}

// Kernel-level checks against direct host references, so a failure points at
// one kernel instead of the whole tower.
static void kern_cmp(const char * what, const std::vector<float> & got, const std::vector<float> & exp, double tol) {
    double maxd = 0, maxv = 0;
    for (size_t i = 0; i < got.size(); i++) {
        maxd = std::max(maxd, (double)std::fabs(got[i] - exp[i]));
        maxv = std::max(maxv, (double)std::fabs(exp[i]));
    }
    const bool ok = maxd <= tol * std::max(1.0, maxv);
    printf("  %-46s max|diff|=%.6f (max=%.4f) %s\n", what, maxd, maxv, ok ? "OK" : "FAIL");
    if (!ok) {
        fails++;
    }
}

static void test_kernels() {
    printf("vision kernels\n");
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    auto rnd = [](int i) {
        unsigned v = (unsigned)i * 2654435761u;
        v ^= v >> 13;
        v *= 2246822519u;
        return (float)(v % 2001) / 1000.f - 1.f;
    };

    { // GEMM
        const int N = 24, K = 200, T = 5;
        std::vector<float> W((size_t)N * K), x((size_t)T * K), exp((size_t)T * N), got((size_t)T * N);
        for (int i = 0; i < N * K; i++) {
            W[i] = rnd(i + 1);
        }
        for (int i = 0; i < T * K; i++) {
            x[i] = rnd(i + 7);
        }
        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                float a = 0;
                for (int k = 0; k < K; k++) {
                    a += W[(size_t)n * K + k] * x[(size_t)t * K + k];
                }
                exp[(size_t)t * N + n] = a;
            }
        }
        float *dW = sycl::malloc_device<float>(W.size(), q), *dx = sycl::malloc_device<float>(x.size(), q);
        float * dout = sycl::malloc_device<float>(got.size(), q);
        q.memcpy(dW, W.data(), W.size() * 4);
        q.memcpy(dx, x.data(), x.size() * 4);
        vit_gemm_launch(q, dW, GGML_TYPE_F32, N, K, dx, K, dout, N, T, 1.f, nullptr);
        q.memcpy(got.data(), dout, got.size() * 4).wait();
        kern_cmp("vit_gemm", got, exp, 1e-5);
        sycl::free(dW, q);
        sycl::free(dx, q);
        sycl::free(dout, q);
    }
    { // LayerNorm
        const int rows = 3, n = 96;
        std::vector<float> x((size_t)rows * n), w(n), b(n), exp((size_t)rows * n), got((size_t)rows * n);
        for (int i = 0; i < rows * n; i++) {
            x[i] = rnd(i + 3);
        }
        for (int i = 0; i < n; i++) {
            w[i] = rnd(i + 11) * 0.5f + 1.f;
            b[i] = rnd(i + 13) * 0.2f;
        }
        for (int r = 0; r < rows; r++) {
            const float * xr = x.data() + (size_t)r * n;
            double s = 0;
            for (int i = 0; i < n; i++) {
                s += xr[i];
            }
            double mean = s / n, var = 0;
            for (int i = 0; i < n; i++) {
                double d = xr[i] - mean;
                var += d * d;
            }
            var /= n;
            const float inv = 1.f / std::sqrt((float)var + 1e-6f);
            for (int i = 0; i < n; i++) {
                exp[(size_t)r * n + i] = (xr[i] - (float)mean) * inv * w[i] + b[i];
            }
        }
        float *dx = sycl::malloc_device<float>(x.size(), q), *dw = sycl::malloc_device<float>(n, q);
        float *db = sycl::malloc_device<float>(n, q), *dout = sycl::malloc_device<float>(got.size(), q);
        q.memcpy(dx, x.data(), x.size() * 4);
        q.memcpy(dw, w.data(), n * 4);
        q.memcpy(db, b.data(), n * 4);
        vit_layernorm_launch(q, dx, n, dw, db, dout, n, rows, n, 1e-6f);
        q.memcpy(got.data(), dout, got.size() * 4).wait();
        kern_cmp("vit_layernorm", got, exp, 1e-4);
        sycl::free(dx, q);
        sycl::free(dw, q);
        sycl::free(db, q);
        sycl::free(dout, q);
    }
    { // vision RoPE on a fused qkv buffer
        const int n_tok = 12, n_head = 2, HD = 64, embd = n_head * HD;
        const int out_w = 2, merge = 2;
        std::vector<float> qkv((size_t)n_tok * 3 * embd), exp;
        for (size_t i = 0; i < qkv.size(); i++) {
            qkv[i] = rnd((int)i + 5);
        }
        exp = qkv;
        const float log2b = std::log2(10000.f);
        for (int t = 0; t < n_tok; t++) {
            const int m = t / (merge * merge), sub = t % (merge * merge);
            const int py = (m / out_w) * merge + sub / merge;
            const int px = (m % out_w) * merge + sub % merge;
            for (int h = 0; h < n_head; h++) {
                for (int pair = 0; pair < HD / 2; pair++) {
                    const int sec = pair / (HD / 4), p = pair % (HD / 4);
                    const float th = (float)(sec == 0 ? py : px) * std::exp2(-2.f * p / (float)(HD / 2) * log2b);
                    const float c = std::cos(th), s = std::sin(th);
                    for (int half = 0; half < 2; half++) {
                        float * row = exp.data() + (size_t)t * 3 * embd + (half ? embd : 0) + h * HD;
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
        std::vector<float> qkv((size_t)n_tok * 3 * embd), got((size_t)n_tok * embd);
        for (size_t i = 0; i < qkv.size(); i++) {
            qkv[i] = rnd((int)i + 17) * 0.3f;
        }
        std::vector<float> exp((size_t)n_tok * embd, 0.f);
        const float scale = 1.f / std::sqrt((float)HD);
        for (int h = 0; h < n_head; h++) {
            for (int i = 0; i < n_tok; i++) {
                std::vector<float> sc(n_tok);
                float mx = -1e30f;
                for (int j = 0; j < n_tok; j++) {
                    float a = 0;
                    for (int d = 0; d < HD; d++) {
                        a += qkv[(size_t)i * 3 * embd + h * HD + d] * qkv[(size_t)j * 3 * embd + embd + h * HD + d];
                    }
                    sc[j] = a * scale;
                    mx = std::max(mx, sc[j]);
                }
                float sum = 0;
                for (int j = 0; j < n_tok; j++) {
                    sc[j] = std::exp(sc[j] - mx);
                    sum += sc[j];
                }
                for (int d = 0; d < HD; d++) {
                    float a = 0;
                    for (int j = 0; j < n_tok; j++) {
                        a += sc[j] * qkv[(size_t)j * 3 * embd + 2 * embd + h * HD + d];
                    }
                    exp[(size_t)i * embd + h * HD + d] = a / sum;
                }
            }
        }
        float *d = sycl::malloc_device<float>(qkv.size(), q), *dout = sycl::malloc_device<float>(got.size(), q);
        q.memcpy(d, qkv.data(), qkv.size() * 4);
        vit_attn_launch(q, d, 3 * embd, dout, embd, n_tok, n_head, HD, scale);
        q.memcpy(got.data(), dout, got.size() * 4).wait();
        kern_cmp("vit_attn", got, exp, 1e-4);
        sycl::free(d, q);
        sycl::free(dout, q);
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
            best = std::min(best,
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        printf("  %-24s %.3f ms\n", what, best);
    };
    struct B {
        float * p;
    };
    std::vector<B> keep;
    auto alloc = [&](size_t n) {
        float * p = sycl::malloc_device<float>(n, q);
        keep.push_back({p});
        return p;
    };
    float *x = alloc((size_t)T * E), *qt = alloc((size_t)T * 3 * E), *attn = alloc((size_t)T * E);
    float *up = alloc((size_t)ff * E), *down = alloc((size_t)E * ff), *ffn = alloc((size_t)T * ff);
    float * qk = alloc((size_t)T * 3 * embd);
    std::vector<float> hz((size_t)ff * E, 0.01f);
    q.memcpy(up, hz.data(), hz.size() * 4);
    q.memcpy(down, hz.data(), hz.size() * 4);

    timeit("gemm qkv 2304x768",
           [&] { vit_gemm_launch(q, down, GGML_TYPE_F32, 3 * E, E, x, E, qt, 3 * E, T, 1.f, nullptr); });
    timeit("gemm ffn_up 3072x768",
           [&] { vit_gemm_launch(q, up, GGML_TYPE_F32, ff, E, x, E, ffn, ff, T, 1.f, nullptr); });
    timeit("gemm ffn_down 768x3072",
           [&] { vit_gemm_launch(q, down, GGML_TYPE_F32, E, ff, ffn, ff, x, E, T, 1.f, nullptr); });
    timeit("rope", [&] { vit_rope_launch(q, qk, 3 * embd, T, NH, HD, 8, 2, 10000.f); });
    timeit("attn n_tok=256", [&] { vit_attn_launch(q, qk, 3 * embd, attn, E, T, NH, HD, 0.125f); });
    timeit("layernorm", [&] { vit_layernorm_launch(q, x, E, up, up, attn, E, T, E, 1e-6f); });
    timeit("gelu", [&] { vit_gelu_launch(q, ffn, T * ff); });
    timeit("add_bias", [&] { vit_add_bias_launch(q, qt, 3 * E, x, T, 3 * E); });
    for (auto & b : keep) {
        sycl::free(b.p, q);
    }
}

static void test_prompt_video(const std::string & text_path, const std::string & mmproj_path) {
    printf("video prompt layout\n");
    gguf_file gf, mgf;
    try {
        gf.load(text_path);
        mgf.load(mmproj_path);
    } catch (const std::exception & ex) {
        printf("  skip (cannot load model: %s)\n", ex.what());
        return;
    }
    tokenizer tk;
    tk.load(gf);
    if (tk.token_to_id.find("<|video_pad|>") == tk.token_to_id.end()) {
        printf("  skip (tokenizer has no <|video_pad|>)\n");
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
    // 4 synthetic frames, distinct content per frame
    const int W = 96, H = 96;
    std::vector<mm_video_frame> frames;
    for (int f = 0; f < 4; f++) {
        mm_video_frame fr;
        fr.width = W;
        fr.height = H;
        fr.pts = f;
        fr.rgb.resize((size_t)W * H * 3);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                uint8_t * p = fr.rgb.data() + ((size_t)y * W + x) * 3;
                p[0] = (uint8_t)(((x + f) * 255) / (W - 1));
                p[1] = (uint8_t)(y * 255 / (H - 1));
                p[2] = (uint8_t)(f * 60 + (x + y) * 30 % 200);
            }
        }
        frames.push_back(std::move(fr));
    }
    mm_video vid;
    vid.frames = frames;

    // grid derived from the preprocessed frame (96x96 → 3x3 merged, like images)
    mm_image img0 = mm_image_preprocess(frames[0].rgb.data(), W, H, cfg);
    vision_input vi0 = vision_model::make_input(vm, img0);
    const int G = vi0.out_w * vi0.out_h;

    const int T = 4;
    const std::string rendered = "<|vision_start|><|video_pad|><|vision_end|>watch this";
    mm_prompt mp = mm_build_prompt_video(tk, rendered, {vid}, vm, vm.hp.proj_dim, T);

    check(mp.n_img == 1, "one video");
    check((int)mp.img_row.size() == (int)mp.tokens.size(), "img_row covers every token");
    check(mp.mrope.size() == 4 * mp.tokens.size(), "mrope is 4 positions per token");
    check(mp.embd.size() == (size_t)T * G * vm.hp.proj_dim, "video embedding rows = T*grid");
    // the block adds T*G tokens but consumes only max(nx,ny) positions:
    // pos = sum_block(n_pos) + non-block tokens = total - T*G + n_pos
    check(mp.pos_after == (int)mp.tokens.size() - T * G + std::max(vi0.out_w, vi0.out_h),
          "video consumes max(nx,ny) positions");
    // token indexing: [0]=<|vision_start|>, then T*G image tokens, then text / <|vision_end|>
    check(mp.img_row[1] == 0 && mp.img_row[1 + (size_t)T * G] == -1, "video rows mapped then cleared");
    bool pos_ok = true;
    const int n = (int)mp.tokens.size();
    for (int i = 0; i < T * G; i++) {
        const int frame = i / G, row = (i % G) / vi0.out_w, col = (i % G) % vi0.out_w;
        if (mp.mrope[1 + i] != 1 + frame) {
            pos_ok = false; // temporal = base + frame
        }
        if (mp.mrope[n + 1 + i] != 1 + row) {
            pos_ok = false; // row = base + row
        }
        if (mp.mrope[2 * n + 1 + i] != 1 + col) {
            pos_ok = false; // col = base + col
        }
    }
    check(pos_ok, "video M-RoPE (t,row,col) layout");
}

// a tiny log-mel input + audio geometry; the front end has no GGUF dependency
static void test_audio_geom() {
    printf("audio input geometry\n");
    const int n_frames = 20;
    audio_input in;
    in.n_frames = n_frames;
    in.n_out = (n_frames + 1) / 2;
    check(in.n_out == 10, "n_out = ceil(n_frames/2)");
}

namespace {
// reference conv1d matching at_conv1d_launch
void ref_conv1d(const float * x, int xf, int xi, int taps, int stride, int pad, const float * w, const float * b,
                float * out, int yf, int wo) {
    for (int t = 0; t < yf; t++) {
        for (int o = 0; o < wo; o++) {
            float acc = b ? b[o] : 0.f;
            for (int tap = 0; tap < taps; tap++) {
                const int row = t * stride + tap - pad;
                if (row < 0 || row >= xf) {
                    continue;
                }
                const float * wr = w + (size_t)o * taps * xi;
                for (int i = 0; i < xi; i++) {
                    acc += wr[tap * xi + i] * x[(size_t)row * xi + i];
                }
            }
            out[(size_t)t * wo + o] = acc;
        }
    }
}
} // namespace

// host-side audio preprocess + decode round trip (WAV → samples → log-mel)
static void test_audio_prep() {
    printf("audio preprocessor\n");
    // build a 1 kHz sine in 16-bit PCM as bytes
    const int rate = 16000;
    const int n = rate / 4;
    audio_preproc_cfg cfg;
    cfg.sample_rate = rate;
    // craft a minimal WAV
    const uint32_t data_bytes = (uint32_t)n * 2;
    std::vector<uint8_t> wav(44 + data_bytes);
    std::memcpy(wav.data(), "RIFF", 4);
    const uint32_t riffsz = 36 + data_bytes;
    std::memcpy(wav.data() + 4, &riffsz, 4);
    std::memcpy(wav.data() + 8, "WAVE", 4);
    std::memcpy(wav.data() + 12, "fmt ", 4);
    const uint32_t fmt = 16;
    std::memcpy(wav.data() + 16, &fmt, 4);
    const uint16_t pcm = 1, ch = 1;
    std::memcpy(wav.data() + 20, &pcm, 2);
    std::memcpy(wav.data() + 22, &ch, 2);
    std::memcpy(wav.data() + 24, &rate, 4);
    const uint32_t br = (uint32_t)rate * 2;
    std::memcpy(wav.data() + 28, &br, 4);
    const uint16_t ba = 2;
    std::memcpy(wav.data() + 32, &ba, 2);
    const uint16_t bits = 16;
    std::memcpy(wav.data() + 34, &bits, 2);
    std::memcpy(wav.data() + 36, "data", 4);
    std::memcpy(wav.data() + 40, &data_bytes, 4);
    for (int i = 0; i < n; i++) {
        const int16_t v = (int16_t)(30000.f * std::sin(2.f * 3.14159265f * 1000.f * i / rate));
        std::memcpy(wav.data() + 44 + (size_t)i * 2, &v, 2);
    }

    mm_audio a;
    std::string err;
    check(mm_audio_decode_wav(wav.data(), wav.size(), a, &err), "WAV decode");
    check(a.sample_rate == rate && (int)a.samples.size() == n, "sample count/rate");
    int nf = 0;
    std::vector<float> mel;
    mm_audio_preprocess(a, cfg, mel, nf);
    check(nf >= 5, "mel frames produced");
    float p = 0;
    for (size_t i = 0; i < mel.size(); i++) {
        if (!std::isfinite(mel[i])) {
            p = -1;
            break;
        }
        p += mel[i] * mel[i];
    }
    check(p > 0, "mel finite + non-zero");
    // a bogus len buffer must fail cleanly
    check(!mm_audio_decode_wav(wav.data(), 10, a, &err), "short WAV rejected");
}

static void test_audio_kernels() {
    printf("audio kernels\n");
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    auto rnd = [](int i) {
        unsigned v = (unsigned)i * 2654435761u;
        v ^= v >> 13;
        v *= 2246822519u;
        return (float)(v % 2001) / 1000.f - 1.f;
    };
    { // at_conv1d vs the reference
        const int xf = 37, xi = 20, taps = 3, stride = 2, pad = 1, wo = 13, yf = (xf + 1) / 2;
        std::vector<float> x((size_t)xf * xi), w((size_t)wo * taps * xi);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = rnd((int)i + 1);
        }
        for (size_t i = 0; i < w.size(); i++) {
            w[i] = rnd((int)i + 3) * 0.05f;
        }
        std::vector<float> exp((size_t)yf * wo), got((size_t)yf * wo);
        ref_conv1d(x.data(), xf, xi, taps, stride, pad, w.data(), nullptr, exp.data(), yf, wo);
        float *dx = sycl::malloc_device<float>(x.size(), q), *dw = sycl::malloc_device<float>(w.size(), q);
        float *dout = sycl::malloc_device<float>(got.size(), q);
        q.memcpy(dx, x.data(), x.size() * 4);
        q.memcpy(dw, w.data(), w.size() * 4);
        at_conv1d_launch(q, dx, xf, xi, taps, stride, pad, dw, nullptr, dout, yf, wo);
        q.memcpy(got.data(), dout, got.size() * 4).wait();
        kern_cmp("at_conv1d", got, exp, 1e-5);
        sycl::free(dx, q);
        sycl::free(dw, q);
        sycl::free(dout, q);
    }
    { // at_rope1d vs the same inverse-frequency math (position = token index)
        const int n_tok = 1000, n_head = 8, HD = 64, embd = n_head * HD;
        std::vector<float> qkv((size_t)n_tok * 3 * embd), exp;
        for (size_t i = 0; i < qkv.size(); i++) {
            qkv[i] = rnd((int)i + 7) * 0.4f;
        }
        exp = qkv;
        const float base = 10000.f, l2b = std::log2(base);
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < n_head; h++) {
                for (int p = 0; p < HD / 2; p++) {
                    const float th = (float)t * std::exp2(-2.f * p / (float)HD * l2b);
                    const float c = std::cos(th), s = std::sin(th);
                    for (int half = 0; half < 2; half++) {
                        float * row = exp.data() + (size_t)t * 3 * embd + (half ? embd : 0) + h * HD;
                        const float a = row[p], b = row[p + HD / 2];
                        row[p] = a * c - b * s;
                        row[p + HD / 2] = a * s + b * c;
                    }
                }
            }
        }
        float *d = sycl::malloc_device<float>(qkv.size(), q);
        q.memcpy(d, qkv.data(), qkv.size() * 4);
        at_rope1d_launch(q, d, 3 * embd, n_tok, n_head, HD, base);
        std::vector<float> got(qkv.size());
        q.memcpy(got.data(), d, got.size() * 4).wait();
        kern_cmp("at_rope1d", got, exp, 1e-5);
        sycl::free(d, q);
    }
    { // at_conv1d matches conv1d_all (the audio_model host helper) exactly
        const int xf = 22, xi = 10, taps = 3, stride = 1, pad = 1, wo = 7, yf = xf;
        std::vector<float> x((size_t)xf * xi), w((size_t)wo * taps * xi);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = rnd((int)i + 11);
        }
        for (size_t i = 0; i < w.size(); i++) {
            w[i] = rnd((int)i + 13) * 0.1f;
        }
        std::vector<float> b(wo);
        for (int i = 0; i < wo; i++) {
            b[i] = rnd(i + 17) * 0.1f;
        }
        // reference via the audio_model host conv
        std::vector<float> exp((size_t)yf * wo);
        // inline the same conv as conv1d_all (kept local to avoid leaking internals)
        for (int t = 0; t < yf; t++) {
            for (int o = 0; o < wo; o++) {
                float acc = b[o];
                const float * wr = w.data() + (size_t)o * taps * xi;
                for (int tap = 0; tap < taps; tap++) {
                    const int row = t * stride + tap - pad;
                    if (row < 0 || row >= xf) {
                        continue;
                    }
                    for (int i = 0; i < xi; i++) {
                        acc += wr[tap * xi + i] * x[(size_t)row * xi + i];
                    }
                }
                exp[(size_t)t * wo + o] = acc;
            }
        }
        float *dx = sycl::malloc_device<float>(x.size(), q), *dw = sycl::malloc_device<float>(w.size(), q);
        float *db = sycl::malloc_device<float>(wo, q), *dout = sycl::malloc_device<float>(exp.size(), q);
        q.memcpy(dx, x.data(), x.size() * 4);
        q.memcpy(dw, w.data(), w.size() * 4);
        q.memcpy(db, b.data(), wo * 4);
        at_conv1d_launch(q, dx, xf, xi, taps, stride, pad, dw, db, dout, yf, wo);
        std::vector<float> got(exp.size());
        q.memcpy(got.data(), dout, got.size() * 4).wait();
        kern_cmp("at_conv1d (bias)", got, exp, 1e-5);
        sycl::free(dx, q);
        sycl::free(dw, q);
        sycl::free(db, q);
        sycl::free(dout, q);
    }
}

// audio tower host vs device on synthetic weights, no GGUF needed beyond what
// the encoder requires (skip if no audio gguf is given)
static void test_audio_encoder(const std::string & audio_path) {
    printf("audio encoder\n");
    if (audio_path.empty()) {
        printf("  skip (no audio mmproj path)\n");
        return;
    }
    audio_model am;
    try {
        am.load(audio_path);
    } catch (const std::exception & ex) {
        printf("  skip (cannot load %s: %s)\n", audio_path.c_str(), ex.what());
        return;
    }
    const int n_mel = am.hp.n_mel, n_frames = 40;
    std::vector<float> mel((size_t)n_frames * n_mel, 0.01f);
    for (size_t i = 0; i < mel.size(); i++) {
        mel[i] = 0.01f * (float)(i % 7) + 0.001f * (float)((i * 3) % 11);
    }
    audio_input in = audio_model::make_input(am, mel.data(), n_frames);
    std::vector<float> host, dev;
    am.encode_host(in, host);
    check((int)host.size() == in.n_out * audio_out_width(am), "host embedding shape");
    check((int)in.n_out == (n_frames + 1) / 2, "n_out geometry");
    double hn = 0;
    for (float v : host) {
        hn += (double)v * v;
    }
    check(std::isfinite(hn) && hn > 0, "host embeddings finite + non-zero");

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    float * d = sycl::malloc_device<float>(host.size(), q);
    check(d != nullptr, "device allocation");
    if (!d) {
        return;
    }
    am.encode_device(q, in, d);
    std::vector<float> dev_(host.size());
    q.memcpy(dev_.data(), d, host.size() * sizeof(float)).wait();
    double maxd = 0, maxv = 0;
    bool finite = true;
    for (size_t i = 0; i < host.size(); i++) {
        if (!std::isfinite(dev_[i])) {
            finite = false;
        }
        maxd = std::max(maxd, (double)std::fabs(dev_[i] - host[i]));
        maxv = std::max(maxv, (double)std::fabs(host[i]));
    }
    const bool ok = finite && maxd <= 5e-3 * std::max(1.0, maxv);
    printf("  %-46s max|host-dev|=%.5f (max|host|=%.4f) %s\n", "device matches host reference", maxd, maxv,
           ok ? "OK" : "FAIL");
    if (!ok) {
        fails++;
    }
    sycl::free(d, q);
}

int main(int argc, char ** argv) {
    const std::string text_path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    const std::string mmproj_path = argc > 2 ? argv[2] : "/home/sfc/临时/Qwen3.5-0.8B-mmproj-BF16.gguf";
    const std::string audio_path = argc > 3 ? argv[3] : "";
    test_target_size();
    test_vision_width_guard();
    test_video_avi();
    test_video_ffmpeg_odd();
    if (argc > 1 && std::strcmp(argv[1], "--video-only") == 0) {
        if (fails) {
            printf("test_multimodal: FAILURES: %d\n", fails);
            return 1;
        }
        printf("test_multimodal: video OK\n");
        return 0;
    }
    test_kernels();
    bench_kernels();
    test_vision(mmproj_path);
    test_device(mmproj_path);
    test_prompt(text_path, mmproj_path);
    test_prompt_video(text_path, mmproj_path);
    test_audio_geom();
    test_audio_prep();
    test_audio_kernels();
    test_audio_encoder(audio_path);
    if (fails) {
        printf("test_multimodal: FAILURES: %d\n", fails);
        return 1;
    }
    printf("test_multimodal: all OK\n");
    return 0;
}
