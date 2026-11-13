#pragma once
// ---------------------------------------------------------------------------
// The engine's one vision encoder ("clip" mmproj): Qwen3.5 (`qwen3vl_merger`)
// and Qwen2-VL (`qwen2vl_merger`).
//
// The per-architecture differences are selected by `vision_model::kind`, which
// comes from the text model's profile (`model_profile::vision`) and is
// cross-checked against the mmproj's `clip.projector_type` at load:
//
//   qwen3vl: patch 16, fused `v.blk.N.attn_qkv`, learned `v.position_embd`,
//            gelu_tanh FFN, gelu_tanh merger, head_dim 64/72.
//   qwen2vl: patch 14 + temporal 2 (the two patch convs are summed), separate
//            `attn_q/k/v` weights and biases (combined into one [3E][E] f32
//            view at load), no position embeddings, quick_gelu FFN, exact-erf
//            GELU merger, `v.post_ln` per patch before the 2x2 concat, head_dim
//            80.
//   deepseek41v: patch 14, fused `attn_qkv`, no position embeddings, RMSNorm
//            (eps 1e-6) instead of LayerNorm, SwiGLU FFN (ffn_gate * ffn_up),
//            2D vision RoPE over the 32 half-split pairs (pairs 0-15 row,
//            16-31 column), bidirectional attention, head_dim 64.  The
//            projector is a 3x3 patch-merge (unfold on a grid zero-padded to
//            multiples of 3) -> mm.1 (9216->5120, erf GELU) -> mm.2
//            (5120->5120).  The token block is plain reading order with the
//            learned sentinel rows: [img_start] then each row's cells followed
//            by [image_newline], then [img_end].
//
// The qwen variants use a 2x2 patch merger; deepseek41v a 3x3 one.
// `vision_model::encode_host` implements the reference forward on the host;
// the SYCL path in src/backend/gpu/kernels/vit.cpp (plus the ds41-specific
// RMSNorm/unfold/assemble kernels in src/backend/gpu/kernels/ds41_vision.cpp)
// mirrors it on the device.
//
// The token order after the "spatial merge" reorder is the one the text model
// sees: 2x2 patch groups (dy,dx) are laid out contiguously, so the merger can
// concatenate 4 consecutive row-vectors into a 4*n_embd feature vector.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <string>
#include <vector>

#include <sycl/sycl.hpp> // IWYU pragma: keep

#include "gguf.h"
#include "image.h"
#include "model_profile.h"
#include "weight_store.h"

namespace si {

struct dnnl_gemm;

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
    // Preprocessing pixel floor.  Qwen3-VL's official processor uses
    // shortest_edge = 65536 (64 merged tokens); the engine previously used
    // 8 * patch_area for every tower, which let a small image through at a
    // token count far below the model's reliable floor.
    int min_pixels = 0;
    // deepseek41v: the cap is on the whole token block (newlines + sentinels),
    // not on a pixel count.  0 for the other towers.
    int max_n_token = 0;
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
    // deepseek41v's SwiGLU gate projection (`v.blk.N.ffn_gate`); null for the
    // qwen towers, whose FFN is a single up projection + GELU.
    vt gate;
    const float * gate_b = nullptr;
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
    int pw = 0;        // patches per row (width / patch_size)
    int ph = 0;        // patches per column
    int n_patches = 0; // pw * ph (tokens entering the ViT)
    int out_w = 0;     // merged grid width (pw / merge)
    int out_h = 0;     // merged grid height (ph / merge)
    int n_out = 0;     // merged tokens (out_w * out_h)
    // deepseek41v: the emitted token block is not out_w*out_h.  `n_llm_w` /
    // `n_llm_h` are the 3x3-merged grid, `n_grid` its cell count, and `n_out`
    // is the *block* length (n_llm_h*(n_llm_w+1)+2, sentinels included).
    // `n_pos` is how many text positions the block consumes (plain 1-D, so =
    // n_out for deepseek41v; for the qwen towers it is max(out_w,out_h)).
    int n_llm_w = 0;
    int n_llm_h = 0;
    int n_grid = 0;
    int n_pos = 0;
};

struct vision_model {
    gguf_file gguf;
    vision_hparams hp;
    // Which tower this file is (selected by the text model's profile and
    // cross-checked against clip.projector_type).
    model_profile::vision_kind kind = model_profile::vision_kind::qwen3vl;
    // patch embedding: combined (weight0 + weight1) conv, already dequantized,
    // rows padded to `patch_K` (a multiple of 8; the device GEMM's 16-byte
    // weight loads require it)
    std::vector<float> patch_w; // [n_embd][patch_K]
    int patch_K = 0;
    const float * patch_b = nullptr;
    const float * pos_embd = nullptr; // [n_pos_side*n_pos_side][n_embd] (dequantized)
    std::vector<float> pos_embd_host;
    const float * post_ln = nullptr;
    const float * post_ln_b = nullptr;
    std::vector<vision_layer> layers;
    vt mm0, mm2;
    const float * mm0_b = nullptr;
    const float * mm2_b = nullptr;
    // deepseek41v's learned sentinel embeddings (5120-wide f32 rows):
    // appended after the projector grid in the order img_start, img_end,
    // image_newline and then gathered by the reading-order layout.
    const float * img_start = nullptr;
    const float * img_end = nullptr;
    const float * img_newline = nullptr;

    // `kind` is the text model's declared vision tower; `none` is refused.  The
    // mmproj's `clip.projector_type` is cross-checked and a disagreement throws.
    void load(const std::string & path, model_profile::vision_kind kind = model_profile::vision_kind::qwen3vl);

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

    // device copy of the weights: the whole file 1:1, or a compact span set
    // (src/backend/weight_store.h) when every linear is on the oneDNN path.
    weight_store weights;
    // PF_VISION_DNNL (default on for qwen2vl): every vision linear runs through
    // the text model's oneDNN int8 matmul instead of vit_gemm.  `gemm` is a
    // private instance created on upload; each call falls back to vit_gemm for a
    // key the instance does not hold (so a projector with an unsupported shape
    // still runs, just at the old speed).
    dnnl_gemm * gemm = nullptr;
    // One vision linear: out[0..T)[0..N) = alpha*W*x + residual, chunked to the
    // oneDNN activation scratch's M cap.  Returns false when this weight is not
    // on the dnnl path (caller uses vit_gemm).
    bool linear_dnnl(sycl::queue & q, const void * key, const float * x, int x_stride, int T, int K, float * out,
                     int out_stride, const float * residual, float alpha, int N);
    // Extra host-side f32 copies: the combined qwen2vl q/k/v, and - under
    // PF_VISION_F32 - every linear.  `d_w32` is the device mirror and
    // `dev_ptr` resolves pointers into either blob, so kernels do not care
    // which representation a tensor uses.
    //
    // The qwen2vl q|k|v matrix is 3*n_embd^2 floats (629 MB on MinerU's tower)
    // and is *only* read by the host reference forward and by the vit_gemm
    // fallback: the oneDNN int8 converter reads the separate q/k/v tensors
    // directly (add_weight_blocks).  Building it eagerly cost 0.67 s and
    // 148 ms of H2D on every page, so it is built on demand (`ensure_w32`) and
    // the device mirror exists only when a kernel can actually read it.
    mutable std::vector<float> w32;
    float * d_w32 = nullptr;
    // bytes the f32 side blob would occupy (its offsets are fixed at load, so
    // the views stay valid when the values are filled later)
    size_t w32_bytes = 0;
    mutable bool w32_built = false;
    // qwen2vl's combined q|k|v bias ([n_layer][3*E] f32) is small enough to
    // build and upload eagerly; it is read on the device by vit_add_bias.
    std::vector<float> qkv_b32;
    float * d_qkv_b32 = nullptr;
    struct qkv_src {
        vt q, k, v;
        size_t off = 0;
    };
    std::vector<qkv_src> host_qkv;
    void ensure_w32() const;
    // Queue every device allocation below was made on (set on first upload):
    // sycl::free needs it, and freeing on another queue is UB.  All callers
    // pass the engine queue, so one queue per model holds in practice.
    sycl::queue * dev_q = nullptr;
    void upload(sycl::queue & q);
    // releases the weight blob + patch weights + scratch on dev_q (no-op when
    // nothing was ever uploaded).  Models are never copied (a copy would
    // double-free), only referenced - and the deleted copy ctor suppresses
    // the implicit default ctor, so it is defaulted explicitly.
    vision_model() = default;
    ~vision_model();
    vision_model(const vision_model &) = delete;
    vision_model & operator=(const vision_model &) = delete;
    const void * dev_ptr(const void * host_ptr) const {
        const char * h = (const char *)host_ptr;
        if (d_w32 && w32_bytes && h >= (const char *)w32.data() && h < (const char *)w32.data() + w32_bytes) {
            return (const char *)d_w32 + (h - (const char *)w32.data());
        }
        if (d_qkv_b32 && !qkv_b32.empty() && h >= (const char *)qkv_b32.data()
            && h < (const char *)qkv_b32.data() + qkv_b32.size() * sizeof(float)) {
            return (const char *)d_qkv_b32 + (h - (const char *)qkv_b32.data());
        }
        // no blob -> identity; contiguous -> offset from map_base; compact ->
        // span search (null for a tensor the blob deliberately omitted)
        return weights.ptr(host_ptr);
    }

    // Reference forward on the host: writes n_out * proj_dim embeddings.
    void encode_host(const vision_input & in, std::vector<float> & out) const;

    // Whether this projector can actually be *run*, and why not if it cannot.
    //
    // The encode path used to be the only place that knew the constraint, so the
    // check could only be discovered by running an encode - which threw from
    // inside whatever the caller was doing, and a caller with more work after
    // the vision section lost all of it.  `load` accepts a well-formed GGUF
    // whose head_dim the encoder does not implement, so "it loaded" has always
    // meant "the file is readable", never "this projector runs".
    //
    // encode_host/encode_device call this and throw the same message, so the
    // query and the throw cannot drift apart.
    bool runnable(std::string * why = nullptr) const;

    // Vision forward on the device.  `d_out` must hold n_out * proj_dim floats;
    // the weights are uploaded on first use and the scratch buffers are grown to
    // the largest image seen (bounded by kMaxImgPatches).
    void encode_device(sycl::queue & q, const vision_input & in, float * d_out);

    // device scratch (owned by this model)
    void * d_patch_w = nullptr;
    float *d_patch_in = nullptr, *d_x = nullptr, *d_ln = nullptr, *d_qkv = nullptr;
    float *d_attn = nullptr, *d_ffn = nullptr, *d_mm0 = nullptr, *d_pos = nullptr;
    // deepseek41v only: SwiGLU gate, and the projector's intermediate grid
    float *d_gate = nullptr, *d_grid = nullptr;
    int scratch_patches = 0;
};

// dequantize helpers shared with the device path
float vt_get(const vt & t, int row, int k);

// DeepSeek-V4.1-Flash ("deepseek41v") preprocessing.  This is deliberately a
// separate entry point rather than an `mm_image_preprocess` option: its plan is
// a min-pixel upscale + patch-14 align + a single token-block solver, its
// resize is a letterbox on RGB 127 (not a stretch), and every normalized value
// is rounded to bf16 before the tower sees it.  The caller decodes the RGB8
// buffer and passes `vm.hp`; the result is planar RGB f32 in [0,1]-normalized
// form, already bf16-rounded.
mm_image mm_image_preprocess_deepseek41v(const uint8_t * rgb, int w, int h, const vision_hparams & hp);
// Dispatch the architecture's preprocessing: deepseek41v has its own plan
// (min-pixel upscale, pad-to-42, bf16 rounding); the qwen towers use the shared
// smart-resize.  Callers that already have the loaded vision_model use this.
mm_image mm_image_preprocess_vision(const vision_model & vm, const uint8_t * rgb, int w, int h,
                                    const image_preproc_cfg & cfg);

} // namespace si
