#pragma once
#include <sycl/sycl.hpp>                // IWYU pragma: keep
#include <sycl/ext/oneapi/bfloat16.hpp> // IWYU pragma: keep
#include <cstdint>

#include "kv_type.h" // IWYU pragma: keep (re-exported for the kernel TUs)
#include "w8.h"

namespace si {

constexpr int kMaxT = 32;      // max tokens per row (prefill chunk)
constexpr int kI8Q = 32;       // int8 KV: head dimensions per quant scale
constexpr int kMaxB = 16;      // max concurrent sequences (continuous batching)
constexpr int kMaxRows = 32;   // max rows in the per-token buffers
constexpr int kBlockSize = 32; // paged KV cache block size (tokens)

constexpr int kMaxSplits = 64; // partial-buffer capacity (prefill K-split)
// decode K-split capacity: the grouped decode kernel fans out over kv heads
// rather than query heads, so it needs (n_head/n_head_kv) x more splits for
// the same warp count (PF_DEC_SPLIT, default kMaxSplits)
constexpr int kMaxDecSplits = 256;
// prefix-cache snapshot map length: boundaries are indexed by complete blocks,
// so this covers max_seq <= kPcMapLen*32 tokens (32768 with 1024)
constexpr int kPcMapLen = 1024;
// max merged vision tokens one image may contribute to a prompt (bounds the
// engine's image-embedding buffer and the vision scratch buffers)
constexpr int kMaxImgTokens = 1024;
// max ViT patch tokens one image may produce (4 patches per merged token); the
// device vision buffers are sized from this
constexpr int kMaxImgPatches = 4 * kMaxImgTokens;

// one GEMV output segment (for batched multi-segment launches)
struct gemv_seg {
    const void * w;  // quantized weight rows [n_rows][K]
    uint32_t type;   // ggml type of w
    int32_t K;       // input length
    int32_t n_rows;  // number of output rows
    const float * x; // input activations, row stride x_stride
    int32_t x_stride;
    const float * act_up; // if non-null: x is activated as silu(x)*act_up
    float * out;          // output, row stride out_stride
    int32_t out_stride;
    const float * residual; // optional residual to add
    float alpha;            // output multiplier
    // optional pre-extracted per-32-value (scale, min) pairs in fp32, row-major
    // [n_rows][K/32]: lets the decode GEMV skip the packed 6-bit scale decode
    const sycl::float2 * meta32 = nullptr;
    // DP4A path: when w8.vals != nullptr this segment runs on the int8 GEMM
    // with activations quantized by xq_launch into the shared x8 scratch.
    w8t w8;
    const int8_t * x8 = nullptr;
    const sycl::float2 * xmeta = nullptr; // activation scales (fp32!)
    const int32_t * xsumq = nullptr;
};

// scalar state shared between kernels of one step (device memory)
struct step_info {
    int32_t n_rows;                // rows (sequences) in this step: 1 or B
    int32_t n_real;                // tokens per row (prefill chunk length or 1)
    int32_t tpb;                   // token slots per row buffer (row = r*tpb + t)
    int32_t pos[kMaxB];            // first position of each row
    int32_t slot[kMaxB];           // state slot / KV block-table row per row
    int32_t active[kMaxB];         // 1 if the row participates
    int32_t tokens[kMaxB * kMaxT]; // token ids (row * kMaxT + t)
    // ---- prefix-cache state snapshots (PF_PREFIX_CACHE) -------------------
    // A tracked prefill writes the conv/GDN state at every completed 32-token
    // block boundary into the checkpoint pool: `pc_active` is 1 only during
    // the forward of that prefill, `pc_row_slot[b]` names the checkpoint slot
    // for boundary b (= b complete blocks, i.e. after token b*32) or -1.
    // Both are read inside the kernels, so they survive graph replay.
    int32_t pc_active;
    int32_t pc_stride;              // floats per checkpoint slot
    float * pc_base;                // checkpoint pool (device USM)
    int32_t pc_row_slot[kPcMapLen]; // boundary -> checkpoint slot or -1

    // ---- multimodal input (images) ----------------------------------------
    // When `mrope_on` is set the attention RoPE reads a full 4-section position
    // per token from `mrope` (section-major: mrope[s*kMaxB*kMaxT + r*kMaxT + t])
    // instead of the implicit `pos[r] + t`.  `img_row[t]` >= 0 replaces the
    // token's embedding with row `img_row[t]` of `img_embd` (n_embd floats).
    int32_t mrope_on;
    int32_t mrope_sections[4]; // text-model rope sections (pairs)
    int32_t mrope[4 * kMaxB * kMaxT];
    const float * img_embd;
    int32_t img_row[kMaxB * kMaxT];
};

// per-layer snapshot target handed to the conv/GDN kernels: the state slice of
// layer `gi` inside a checkpoint slot is
//   base + slot*stride + layer_off + [0, gdn_per)          (GDN state)
//   base + slot*stride + layer_off + gdn_per + [0, conv_per) (conv state)
struct pc_snap {
    float * base = nullptr;
    int64_t stride = 0;
    int64_t layer_off = 0;
    int32_t gdn_per = 0;
    int32_t conv_per = 0;
};

void rmsnorm_launch(sycl::queue & q, const float * x, const float * w, float * out, int n_rows, int n, float eps);
void copy_row_launch(sycl::queue & q, const float * src, float * dst, const step_info * info, int n, int row = -1);
void embed_launch(sycl::queue & q, const void * table, uint32_t type, const step_info * info, float * out, int n_embd,
                  size_t row_bytes);
void gemv_group_launch(sycl::queue & q, uint32_t type, const gemv_seg * segs, int n_segs, int total_rows, int TB,
                       int nsb, int n_tok_blocks = 0);
// kscales/vscales: the int8 per-32 fp16 scale planes (nullptr for the other
// storage types); one plane entry per (block, kv head, token, 32 dims)
void qk_norm_rope_launch(sycl::queue & q, float * qbuf, float * kbuf, float * vbuf, const float * q_norm,
                         const float * k_norm, void * kpool, void * vpool, const int32_t * tables,
                         const step_info * info, int n_head, int n_head_kv, int head_dim, int n_rot, float rope_base,
                         float eps, int max_blocks, int n_rows, int n_real, const void * kscales = nullptr,
                         const void * vscales = nullptr);
void attn_launch(sycl::queue & q, const float * qbuf, const float * gate, const void * kpool, const void * vpool,
                 float * partials, const int32_t * tables, int n_head, int n_head_kv, int head_dim, int n_splits,
                 const step_info * info, float scale, int max_blocks, int n_rows, int n_real, float * out = nullptr,
                 int group = -1, const void * kscales = nullptr, const void * vscales = nullptr);
// group: -1 = auto (PF_DEC_GROUP, default on), 0 = classic one-warp-per-head,
// 1 = grouped (one warp per kv head, 4 query heads sharing each K/V load).
// Only honoured for n_real == 1, head_dim == 256, n_head == 4*n_head_kv and
// out == nullptr (a fused n_splits == 1 call keeps the classic kernel).
void attn_combine_launch(sycl::queue & q, const float * partials, const float * gate, float * out,
                         const step_info * info, int n_head, int head_dim, int n_splits, int n_rows, int n_real);
void conv_l2_launch(sycl::queue & q, const float * qkv_raw, float * conv_state, const float * conv_w, float * conv_out,
                    const step_info * info, int conv_dim, int kernel_size, int head_k_dim, int n_k_heads, float eps,
                    int n_rows, int n_real, int row0 = 0, int tpb_arg = -1, bool cross_row = false);
void conv_state_update_launch(sycl::queue & q, const float * qkv_raw, float * conv_state, const step_info * info,
                              int conv_dim, int kernel_size, int n_rows, int row0 = 0, int tpb_arg = -1,
                              int nreal_arg = -1, bool last_row_only = false, pc_snap snap = {});
void gdn_launch(sycl::queue & q, const float * conv_out, const float * alpha, const float * dt_bias,
                const float * ssm_a, const float * beta, float * state, float * attn_out, const step_info * info,
                int head_dim, int n_heads, int conv_dim, float scale, int n_slots, int n_rows, int row0 = 0,
                int tpb_arg = -1, int nreal_arg = -1, pc_snap snap = {});
void gated_norm_launch(sycl::queue & q, const float * attn, const float * z, const float * weight, float * out,
                       const step_info * info, int n_heads, int head_dim, float eps, int n_rows, int n_real,
                       int row0 = 0);

// ---------------------------------------------------------------------------
// DP4A (int8) path.  Weights live in the SI8 format (see w8.h); activations
// are quantized on the fly by xq_launch into per-16-element symmetric int8
// groups (scale + sum(q) kept alongside for the weight min/zero-point).
//
// xq_launch: x8[t][k] = clamp(round(act/sx)), where act = x (or silu(x)*up).
//   xmeta[t][g] = fp16(sx) | fp16(0)<<16      (g = k/16)
//   xsumq[t][g] = sum of the 16 int8 values    (for the min correction)
void xq_launch(sycl::queue & q, const float * x, const float * up, int x_stride, int up_stride, int8_t * x8,
               sycl::float2 * xmeta, int32_t * xsumq, const step_info * info, int TB, int K);

// dp4a_gemv_launch: single-token decode GEMV on SI8 weights.
// One sub-group covers 32 rows (lane = row), so the group-major weight layout
// gives fully coalesced 16-byte loads for the whole sub-group.  The quantized
// activation group (16 bytes) and its scale/sum are broadcast.
void dp4a_gemv_launch(sycl::queue & q, const w8t & w, const int8_t * x8, const sycl::float2 * xmeta,
                      const int32_t * xsumq, float * out, const float * residual, float alpha, int n_rows_out);

// dp4a_gemm_launch: out[t][row] = alpha * sum_k w[row][k] * x[t][k] (+ residual)
// with w in SI8 and x quantized as above.  One workgroup covers 256 rows, each
// sub-group a 32-row x 32-token tile (hardware dp4a).
void dp4a_gemm_launch(sycl::queue & q, const w8t & w, const int8_t * x8, const sycl::float2 * xmeta,
                      const int32_t * xsumq, float * out, int out_stride, const float * residual, float alpha, int TB);

// ---------------------------------------------------------------------------
// Vision encoder kernels (src/kernels/vit.cpp).  All activations are fp32 and
// the weights are BF16 (type 30) or F32 (type 0), row-major [N][K].
// ---------------------------------------------------------------------------

// out[t][n] = alpha * sum_k W[n][k] * x[t][k] + (residual ? residual[t][n] : 0)
void vit_gemm_launch(sycl::queue & q, const void * w, uint32_t wtype, int N, int K, const float * x, int x_stride,
                     float * out, int out_stride, int T, float alpha, const float * residual);

// LayerNorm over n elements per row (bias optional), with an optional input
// stride and output stride.
void vit_layernorm_launch(sycl::queue & q, const float * x, int x_stride, const float * w, const float * b, float * out,
                          int out_stride, int rows, int n, float eps);

// in-place GELU (tanh approximation), matching ggml_gelu
void vit_gelu_launch(sycl::queue & q, float * x, int n);

// y[row][i] += bias[i] for rows rows of cols elements
void vit_add_bias_launch(sycl::queue & q, float * y, int y_stride, const float * bias, int rows, int cols);

// y[row][i] += x[row][i] for rows rows of cols elements
void vit_add_launch(sycl::queue & q, float * y, int y_stride, const float * x, int x_stride, int rows, int cols);

// 2D "vision" RoPE on the Q and K halves of the fused qkv buffer
// ([n_tok][3*n_embd]); pairs 0..head_dim/4-1 use the patch row, the rest the
// patch column, with the frequency exponent reset per section.
void vit_rope_launch(sycl::queue & q, float * qkv, int qkv_stride, int n_tok, int n_head, int head_dim, int out_w,
                     int merge, float rope_base);

// bidirectional (non-causal) attention over all n_tok tokens, per head
void vit_attn_launch(sycl::queue & q, const float * qkv, int qkv_stride, float * out, int out_stride, int n_tok,
                     int n_head, int head_dim, float scale);

} // namespace si
