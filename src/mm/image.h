#pragma once
// ---------------------------------------------------------------------------
// Host-side image decoding and preprocessing for vision models.
//
// `mm_image_preprocess` mirrors the Qwen-VL (qwen2vl / qwen25vl / qwen3vl)
// preprocessing used by the reference implementation: "smart resize" that keeps
// the aspect ratio and aligns both sides to patch_size*merge, a Pillow-style
// bicubic resample, then per-channel normalization and a plane-major (CHW)
// float32 layout for the vision encoder.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <string>
#include <vector>

namespace si {

// Decoded + preprocessed image.  `chw` holds three ny*nx planes (R, G, B) of
// float32, already normalized with (x - mean) / std.
struct mm_image {
    int width = 0;
    int height = 0;
    std::vector<float> chw;
};

struct image_preproc_cfg {
    int patch_size = 16;
    int merge = 2;
    int min_pixels = 0;   // 0 = disabled
    int max_pixels = 0;   // 0 = disabled
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std[3] = {0.5f, 0.5f, 0.5f};
};

// Decode a PNG/JPEG/BMP image file into interleaved RGB8.  Returns false and
// fills `err` on failure.
bool mm_image_decode_file(const std::string & path, std::vector<uint8_t> & rgb, int & w, int & h,
                          std::string * err = nullptr);
// Decode an in-memory encoded image (used for base64 data URLs).
bool mm_image_decode_mem(const uint8_t * data, size_t len, std::vector<uint8_t> & rgb, int & w, int & h,
                         std::string * err = nullptr);

// Resize (aspect preserving, aligned) + normalize + convert to plane-major f32.
mm_image mm_image_preprocess(const uint8_t * rgb, int w, int h, const image_preproc_cfg & cfg);

// The aligned target size chosen by the smart-resize step (exposed for tests).
void mm_image_target_size(int w, int h, const image_preproc_cfg & cfg, int & out_w, int & out_h);

} // namespace si
