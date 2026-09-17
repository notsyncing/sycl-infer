#pragma once
// ---------------------------------------------------------------------------
// Qwen3.5 vision encoder ("clip" mmproj, projector qwen3vl_merger).
//
// The mmproj GGUF is a 12-layer ViT over 16x16 patches (fused QKV + GELU MLP,
// LayerNorm, learned absolute position embeddings) followed by a 2x2 patch
// merger.  `vision_model::encode_host` implements the reference forward on the
// host; the SYCL path in src/kernels/vit.cpp mirrors it on the device.
//
// The token order after the "spatial merge" reorder is the one the text model
// sees: 2x2 patch groups (dy,dx) are laid out contiguously, so the merger can
// concatenate 4 consecutive row-vectors into a 4*n_embd feature vector.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "gguf.h"
#include "image.h"

namespace si {

// view of one vision weight: N rows of K elements, row-major
struct vt {
    const void * data = nullptr;
    uint32_t type = 0;
    int K = 0;
    int N = 0;
};

struct vision_hparams {
    int image_size = 0;
    int patch_size = 0;
    int n_embd = 0;
    int n_ff = 0;
    int n_layer = 0;
    int n_head = 0;
    int head_dim = 0;
    int proj_dim = 0;
    int merge = 2;
    int n_pos_side = 0; // sqrt(position_embd rows)
    float eps = 1e-6f;
    float rope_base = 10000.0f;
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std[3] = {0.5f, 0.5f, 0.5f};
};

struct vision_layer {
    vt qkv, out;
    const float * qkv_b = nullptr;
    const float * out_b = nullptr;
    vt up, down;
    const float * up_b = nullptr;
    const float * down_b = nullptr;
    const float * ln1 = nullptr;
    const float * ln1_b = nullptr;
    const float * ln2 = nullptr;
    const float * ln2_b = nullptr;
};

// A preprocessed image plus the geometry the vision encoder needs.
struct vision_input {
    const float * chw = nullptr; // plane-major R,G,B, width*height each
    int width = 0;
    int height = 0;
    int pw = 0;                  // patches per row (width / patch_size)
    int ph = 0;                  // patches per column
    int n_patches = 0;           // pw * ph (tokens entering the ViT)
    int out_w = 0;               // merged grid width (pw / merge)
    int out_h = 0;               // merged grid height (ph / merge)
    int n_out = 0;               // merged tokens (out_w * out_h)
};

struct vision_model {
    gguf_file gguf;
    vision_hparams hp;
    // patch embedding: combined (weight0 + weight1) conv, already dequantized
    std::vector<float> patch_w; // [n_embd][3*patch*patch]
    const float * patch_b = nullptr;
    const float * pos_embd = nullptr; // [n_pos_side*n_pos_side][n_embd] (dequantized)
    std::vector<float> pos_embd_host;
    const float * post_ln = nullptr;
    const float * post_ln_b = nullptr;
    std::vector<vision_layer> layers;
    vt mm0, mm2;
    const float * mm0_b = nullptr;
    const float * mm2_b = nullptr;

    void load(const std::string & path);

    // geometry helper used by preprocessing and the host/device forward
    static vision_input make_input(const vision_model & vm, const mm_image & img);

    // patch embedding + patch bias + learned position embedding, laid out in
    // the merged token order the text model expects: [np][n_embd] (host path)
    void build_patch_input(const vision_input & in, std::vector<float> & out) const;
    // raw patch vectors [np][3*patch_size^2] in the same merged token order
    // (device path: the patch projection runs as a GEMM)
    void build_raw_patches(const vision_input & in, std::vector<float> & out) const;
    // sampled learned position embedding [np][n_embd] in merged token order
    void build_pos_emb(const vision_input & in, std::vector<float> & out) const;

    // device copy of the weights (one blob)
    void * dev_weights = nullptr;
    size_t dev_weights_size = 0;
    void upload(sycl::queue & q);
    ~vision_model();
    const void * dev_ptr(const void * host_ptr) const {
        return (const char *) dev_weights + ((const char *) host_ptr - (const char *) gguf.map_base);
    }

    // Reference forward on the host: writes n_out * proj_dim embeddings.
    void encode_host(const vision_input & in, std::vector<float> & out) const;

    // Vision forward on the device.  `d_out` must hold n_out * proj_dim floats;
    // the weights are uploaded on first use and the scratch buffers are grown to
    // the largest image seen (bounded by kMaxImgPatches).
    void encode_device(sycl::queue & q, const vision_input & in, float * d_out);

    // device scratch (owned by this model)
    void * d_patch_w = nullptr;
    float *d_patch_in = nullptr, *d_x = nullptr, *d_ln = nullptr, *d_qkv = nullptr;
    float *d_attn = nullptr, *d_ffn = nullptr, *d_mm0 = nullptr, *d_pos = nullptr;
    int scratch_patches = 0;
};

// dequantize helpers shared with the device path
float vt_get(const vt & t, int row, int k);

} // namespace si
