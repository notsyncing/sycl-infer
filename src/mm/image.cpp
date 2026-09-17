#include "image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_NO_STDIO
#include "stb/stb_image.h"

namespace si {

namespace {

// Pillow's bicubic kernel (a = -0.5).  ggml/PyTorch use a = -0.75, so this only
// matches the reference preprocessing because llama.cpp also uses Pillow's.
double bicubic_w(double x) {
    if (x < 0.0) x = -x;
    constexpr double a = -0.5;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

double filter_w(double x) { return bicubic_w(x); }

uint8_t clip8(double v) {
    if (v <= 0.0) return 0;
    if (v >= 255.0) return 255;
    return (uint8_t) std::lround(v);
}

// Two-pass separable resample of an interleaved RGB8 buffer.
void resize_bicubic(const uint8_t * src, int sw, int sh, uint8_t * dst, int dw, int dh) {
    // horizontal pass: sh x sw -> sh x dw
    std::vector<float> tmp((size_t) sh * dw * 3);
    for (int y = 0; y < sh; y++) {
        for (int x = 0; x < dw; x++) {
            // rebuild per-x taps (kernel offsets differ per output pixel)
            const double scale = (double) sw / dw;
            const double filterscale = scale < 1.0 ? 1.0 : scale;
            const double support = 2.0 * filterscale;
            const double center = (x + 0.5) * scale;
            int xmin = (int) (center - support + 0.5);
            if (xmin < 0) xmin = 0;
            int xmax = (int) (center + support + 0.5);
            if (xmax > sw) xmax = sw;
            double acc[3] = {0, 0, 0}, ww = 0;
            for (int i = xmin; i < xmax; i++) {
                const double w = filter_w((i - center + 0.5) / filterscale);
                const uint8_t * p = src + ((size_t) y * sw + i) * 3;
                acc[0] += w * p[0];
                acc[1] += w * p[1];
                acc[2] += w * p[2];
                ww += w;
            }
            float * o = tmp.data() + ((size_t) y * dw + x) * 3;
            o[0] = (float) (acc[0] / ww);
            o[1] = (float) (acc[1] / ww);
            o[2] = (float) (acc[2] / ww);
        }
    }
    // vertical pass: sh x dw -> dh x dw
    const double scale = (double) sh / dh;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = 2.0 * filterscale;
    for (int y = 0; y < dh; y++) {
        const double center = (y + 0.5) * scale;
        int ymin = (int) (center - support + 0.5);
        if (ymin < 0) ymin = 0;
        int ymax = (int) (center + support + 0.5);
        if (ymax > sh) ymax = sh;
        double wsum = 0;
        std::vector<double> ws(ymax - ymin);
        for (int i = ymin; i < ymax; i++) {
            const double w = filter_w((i - center + 0.5) / filterscale);
            ws[i - ymin] = w;
            wsum += w;
        }
        for (int x = 0; x < dw; x++) {
            double acc[3] = {0, 0, 0};
            for (int i = ymin; i < ymax; i++) {
                const float * p = tmp.data() + ((size_t) i * dw + x) * 3;
                const double w = ws[i - ymin];
                acc[0] += w * p[0];
                acc[1] += w * p[1];
                acc[2] += w * p[2];
            }
            uint8_t * o = dst + ((size_t) y * dw + x) * 3;
            o[0] = clip8(acc[0] / wsum);
            o[1] = clip8(acc[1] / wsum);
            o[2] = clip8(acc[2] / wsum);
        }
    }
}

} // namespace

void mm_image_target_size(int w, int h, const image_preproc_cfg & cfg, int & out_w, int & out_h) {
    out_w = 0;
    out_h = 0;
    if (w <= 0 || h <= 0) return;
    const int align = cfg.patch_size * cfg.merge;
    auto round_by = [align](float x) { return (int) std::lround(x / (float) align) * align; };
    auto ceil_by = [align](float x) { return (int) std::ceil(x / (float) align) * align; };
    auto floor_by = [align](float x) { return (int) std::floor(x / (float) align) * align; };

    int w_bar = std::max(align, round_by((float) w));
    int h_bar = std::max(align, round_by((float) h));
    if (cfg.max_pixels > 0 && h_bar * w_bar > cfg.max_pixels) {
        const float beta = std::sqrt((float) h * w / cfg.max_pixels);
        h_bar = std::max(align, floor_by(h / beta));
        w_bar = std::max(align, floor_by(w / beta));
    } else if (cfg.min_pixels > 0 && h_bar * w_bar < cfg.min_pixels) {
        const float beta = std::sqrt((float) cfg.min_pixels / ((float) h * w));
        h_bar = ceil_by(h * beta);
        w_bar = ceil_by(w * beta);
    }
    out_w = w_bar;
    out_h = h_bar;
}

mm_image mm_image_preprocess(const uint8_t * rgb, int w, int h, const image_preproc_cfg & cfg) {
    mm_image out;
    if (!rgb || w <= 0 || h <= 0) return out;
    int tw = 0, th = 0;
    mm_image_target_size(w, h, cfg, tw, th);

    const uint8_t * src = rgb;
    std::vector<uint8_t> resized;
    if (tw != w || th != h) {
        resized.resize((size_t) tw * th * 3);
        resize_bicubic(rgb, w, h, resized.data(), tw, th);
        src = resized.data();
    }

    out.width = tw;
    out.height = th;
    const size_t n = (size_t) tw * th;
    out.chw.resize(n * 3);
    for (size_t p = 0; p < n; p++) {
        for (int c = 0; c < 3; c++) {
            const float v = (float) src[p * 3 + c] / 255.0f;
            out.chw[(size_t) c * n + p] = (v - cfg.mean[c]) / cfg.std[c];
        }
    }
    return out;
}

bool mm_image_decode_mem(const uint8_t * data, size_t len, std::vector<uint8_t> & rgb, int & w, int & h,
                         std::string * err) {
    int comp = 0;
    stbi_uc * px = stbi_load_from_memory(data, (int) len, &w, &h, &comp, 3);
    if (!px) {
        if (err) *err = stbi_failure_reason() ? stbi_failure_reason() : "decode failed";
        return false;
    }
    rgb.assign(px, px + (size_t) w * h * 3);
    stbi_image_free(px);
    return true;
}

bool mm_image_decode_file(const std::string & path, std::vector<uint8_t> & rgb, int & w, int & h,
                          std::string * err) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        std::fclose(f);
        if (err) *err = "empty file " + path;
        return false;
    }
    std::vector<uint8_t> buf((size_t) sz);
    const size_t rd = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (rd != buf.size()) {
        if (err) *err = "short read " + path;
        return false;
    }
    return mm_image_decode_mem(buf.data(), buf.size(), rgb, w, h, err);
}

} // namespace si
