#pragma once
// ---------------------------------------------------------------------------
// Multimodal prompt assembly: tokenize a chat-template rendering that contains
// image placeholders, expand each `<|image_pad|>` into the merged vision tokens
// and derive the M-RoPE positions the text model needs.
//
// The position layout follows the reference: an image of nx*ny merged tokens
// consumes max(nx, ny) positions.  Every image token shares the temporal
// position `base`, and uses `base + row` / `base + col` for the row/col
// sections.  Text tokens continue one position per token.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "image.h"

// `tokenizer` lives in the global namespace (see src/model/tokenizer.h)
struct tokenizer;

namespace si {

struct vision_model;

struct mm_prompt {
    std::vector<int> tokens;
    std::vector<int32_t> mrope;   // 4 * n, section-major ([s * n + i])
    std::vector<int32_t> img_row; // n, -1 or a row in `embd`
    std::vector<float> embd;      // n_img * text n_embd (host path)
    const float * d_embd = nullptr; // device path: n_img * n_embd, owned by caller
    int n_img = 0;
    int pos_after = 0;            // next decode position
    bool has_images() const { return n_img > 0; }
};

// `rendered` must contain exactly one `<|image_pad|>` per entry of `images`.
// `n_embd` is the text model width (the vision projection width).
mm_prompt mm_build_prompt(const tokenizer & tk, const std::string & rendered,
                          const std::vector<mm_image> & images, const vision_model & vm,
                          int n_embd);

// Device variant: runs the vision tower on the GPU and writes the merged
// embeddings into `d_out` (n_img * n_embd floats).  `vm` must outlive the call.
mm_prompt mm_build_prompt_device(vision_model & vm, sycl::queue & q, const tokenizer & tk,
                                 const std::string & rendered, const std::vector<mm_image> & images,
                                 int n_embd, float * d_out);

} // namespace si
