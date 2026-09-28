#pragma once
#include <sycl/sycl.hpp>                // IWYU pragma: keep
#include <sycl/ext/oneapi/bfloat16.hpp> // IWYU pragma: keep
#include <cstdint>

#include "kv_type.h" // IWYU pragma: keep (re-exported for the kernel TUs)
#include "w8.h"

namespace si {

constexpr int kMaxT = 32;      // max tokens per row (prefill chunk)
constexpr int kI8Q = 32;       // int8/int4 KV: head dimensions per quant scale
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
    int32_t dev = 0; // backend device index (multi-device grouping)
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
    // CPU-only: run the integer GEMV straight from the GGUF blocks in `w`
    // (type/K/n_rows are the real tensor geometry), no SIn packing needed.
    bool i8 = false;
    // XMX/decode path: int8 oneDNN weight, row-major [n_rows][K] with one fp32
    // scale per row in `wsc` (`wi8` points into the oneDNN weight buffer).  The
    // decode GEMV reads this instead of the oneDNN primitive (M=1 is dominated
    // by per-call overhead there).
    const int8_t * wi8 = nullptr;
    const float * wsc = nullptr;
};

// scalar state shared between kernels of one step (device memory)
struct step_info {
    int32_t n_rows;                // rows (sequences) in this step: 1 or B
    int32_t n_real;                // tokens per row (prefill chunk length or 1)
    int32_t tpb;                   // token slots per row buffer (row = r*tpb + t)
    int32_t pos[kMaxB];            // first position of each row
    // Per-row real token count for chunk-batched prefill (mode 2): the last
    // row of a batch may be partial when the prompt is not a multiple of
    // kMaxT.  0 means "use n_real" (mode 0/1 and graph-replay callers).
    int32_t n_real_row[kMaxB];
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
    // When `mtp_dt` is set a *tracked* forward snapshots the recurrent state
    // after EVERY token instead of at 32-token boundaries: the GDN kernel writes
    // state(t) into pc_base + pc_row_slot[t]*pc_stride.  The MTP spec-decode
    // verify uses this to roll the state back to the accepted draft length.
    int32_t mtp_dt;
    // MTP spec verify: compute the forward but do NOT store the recurrent state
    // (GDN state / conv window) back.  The verify's batch must leave the live
    // state untouched so the commit can advance it by exactly the accepted
    // tokens; a dry verify also disables the per-token snapshots.
    int32_t mtp_dry;

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
// MTP (NextN) draft head preparation: for every live token of `info` compute
//   out[i] = [ rmsnorm(enorm, emb(tok_i)) ; rmsnorm(hnorm, h_{i-1}) ]
// (2*n_embd wide, embedding half first - the order blk.N.nextn.eh_proj expects).
// `h` holds the main model's hidden states [token][n_embd] for the same batch;
// token i takes h[i-1] and the first token of each row takes `h_prev[r]` (the
// main hidden at the position before the row's first token).
// MTP draft head: copy the main model's post-output-norm hidden rows into a
// dedicated buffer right after the forward that produced them, so the draft
// head never reads the shared activation scratch (which later kernels reuse).
void mtp_capture_launch(sycl::queue & q, const float * src, float * dst, int n_rows, int n);
void mtp_concat_launch(sycl::queue & q, const void * table, uint32_t type, size_t row_bytes, const float * enorm,
                       const float * hnorm, const float * h, const float * h_prev, const step_info * info,
                       float * out, int n_embd, float eps);
void copy_row_launch(sycl::queue & q, const float * src, float * dst, const step_info * info, int n, int row = -1);
void embed_launch(sycl::queue & q, const void * table, uint32_t type, const step_info * info, float * out, int n_embd,
                  size_t row_bytes);
// 4-bit (u4) decode GEMV (M == 1) and the even/odd activation split it needs.
// See src/backend/gpu/kernels/w4_gemv.cpp and common/w4.h.
void w4_split_act_launch(sycl::queue & q, const int8_t * axg, int8_t * axe, int8_t * axo, int M, int K);
void i8_grp_gemv_launch(sycl::queue & q, const int8_t * w8, const uint16_t * wsc, const int8_t * xq,
                        const uint16_t * asa, const float * xs, float * out, const float * residual, float alpha, int K,
                        int N);
void w4_gemm_launch(sycl::queue & q, const uint8_t * vals, const uint16_t * scale, const uint16_t * off,
                    const int8_t * axe, const int8_t * axo, const uint16_t * asa, const float * xs, float * out,
                    int out_stride, const float * residual, float alpha, int M, int K, int N);
void w4_gemv_launch(sycl::queue & q, const uint8_t * vals, const uint16_t * scale, const uint16_t * off,
                    const int8_t * axe, const int8_t * axo, const uint16_t * asa, const float * xs, float * out,
                    int out_stride, const float * residual, float alpha, int K, int N);
// Batched (M = 2..13) native-width GEMM for the MTP speculative verify: the
// activation tile is staged in SLM once per workgroup and reused across 16*C
// output columns, so the weight stream is read once for all M rows.  `fmt` is
// 0 = u4, 1 = k5, 2 = cb4, 3 = grouped int8; w0/w1 are the format's value
// planes (u4 vals / k5 vals+hi / cb4 idx / int8 w8), lut16/bit_lut the cb4/k5
// lookups, and scale/off the per-(g,n) f16 planes (off unused for cb4/int8).
// Returns false (caller must fall back) for M outside 2..13 or K % 32 != 0.
bool nat_gemm_launch(sycl::queue & q, int fmt, const void * w0, const void * w1, const int8_t * w8,
                     const uint16_t * scale, const uint16_t * off, const uint16_t * lut16, const uint32_t * bit_lut,
                     const int8_t * axe, const int8_t * axo, const int8_t * axg, const uint16_t * asa,
                     const float * xs, float * out, int out_stride, const float * residual, float alpha, int M, int K,
                     int N);

// Native-width 5-bit (Q5_K) decode GEMV (M == 1): the nibble plane is the u4
// one, so the even/odd activation split is reused; the fifth bit comes from the
// `hi` plane and is recombined with one OR per dp4a operand.  See common/w4.h.
//   y[n] = sum_g asa[g] * ( step[g][n]*QDOT_g[n] + off[g][n]*XS[g] )
void k5_gemv_launch(sycl::queue & q, const uint8_t * vals, const uint8_t * hi, const uint16_t * scale,
                    const uint16_t * off, const int8_t * axe, const int8_t * axo, const uint16_t * asa,
                    const float * xs, float * out, const float * residual, float alpha, int K, int N);
// expand the two 5-bit planes into int8 q5 in element order, for the prefill
// matmul: out[n*K + k] = lo4 | (bit << 4), in [0,31]
void k5_expand_launch(sycl::queue & q, const uint8_t * vals, const uint8_t * hi, const uint32_t * bit_lut, int8_t * out,
                      int K, int N);

// Codebook 4-bit (IQ4_XS / IQ4_NL) decode GEMV (M == 1) and the index->int8
// expansion the prefill matmul needs.  See common/w4.h and w4_gemv.cpp.
//   y[n] = sum_g asa[g] * scale[g][n] * QDOT_g[n],  QDOT from the 16-entry int8
//   table `lut` indexed by the packed nibbles (so the native values are exact).
// `lut16` is the 256-entry byte-pair table lut16[b] = value(b & 0xF) |
// value(b >> 4) << 8 over the 16 codebook values (see cb4_pack).  The index
// plane is interleaved (byte k = elements 2k/2k+1), so one index byte gives the
// two codebook values of a dp4a half-word.
void cb4_gemv_launch(sycl::queue & q, const uint8_t * idx, const uint16_t * lut16, const uint16_t * scale,
                     const int8_t * xq, const uint16_t * asa, const float * xs, float * out, const float * residual,
                     float alpha, int K, int N);
// expand the index plane into int8 weights (for the oneDNN prefill matmul):
// out[n*K + k] = the codebook value of index k, in element order
void cb4_expand_launch(sycl::queue & q, const uint8_t * idx, const uint16_t * lut16, int8_t * out, int K, int N);

void gemv_group_launch(sycl::queue & q, uint32_t type, const gemv_seg * segs, int n_segs, int total_rows, int TB,
                       int nsb, int n_tok_blocks = 0);
// kscales/vscales: the int8/int4 per-32 fp16 scale planes (nullptr for the
// other storage types); one plane entry per (block, kv head, token, 32 dims)
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
// XMX (oneDNN int8 matmul) prefill attention - default on, PF_ATTN_XMX=0
// disables, see attn_xmx.cpp.
// Fills the split-partials layout; returns false if the call is out of scope
// (i8 KV, head_dim 256, oneDNN available), in which case the caller runs the
// classic kernel.
bool attn_xmx_launch(sycl::queue & q, const float * qbuf, const float * gate, const void * kpool, const void * vpool,
                     float * partials, const int32_t * tables, int n_head, int n_head_kv, int head_dim, int n_splits,
                     const step_info * info, float scale, int max_blocks, int n_rows, int n_real, float * out,
                     const void * kscales, const void * vscales);
void conv_l2_launch(sycl::queue & q, const float * qkv_raw, float * conv_state, const float * conv_w, float * conv_out,
                    const step_info * info, int conv_dim, int kernel_size, int head_k_dim, int n_k_heads, float eps,
                    int n_rows, int n_real, int row0 = 0, int tpb_arg = -1, bool cross_row = false);
void conv_state_update_launch(sycl::queue & q, const float * qkv_raw, float * conv_state, const step_info * info,
                              int conv_dim, int kernel_size, int n_rows, int row0 = 0, int tpb_arg = -1,
                              int nreal_arg = -1, bool last_row_only = false, pc_snap snap = {});
void gdn_launch(sycl::queue & q, const float * conv_out, const float * alpha, const float * dt_bias,
                const float * ssm_a, const float * beta, float * state, float * attn_out, const step_info * info,
                int head_dim, int n_k_heads, int n_heads, int conv_dim, float scale, int n_slots, int n_rows,
                int row0 = 0, int tpb_arg = -1, int nreal_arg = -1, pc_snap snap = {});
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

// i8_row_gemv_launch: single-token decode GEMV over a row-major [N][K] int8
// weight with one fp32 scale per row (the oneDNN weight buffer). `xq` is the
// row-quantized activation (K int8 values) and `sx_dev` a device pointer to its
// scale (read inside the kernel).
void i8_row_gemv_launch(sycl::queue & q, const int8_t * w, const float * sw, const int8_t * xq, const float * sx_dev,
                        float * out, int K, int N, const float * residual, float alpha);

// One launch for all segments of a decode call: `segs` (device array, n_segs
// entries) share the activation row `xq`/`sx_dev`.  total_rows = sum(n_rows).
void i8_row_gemv_multi_launch(sycl::queue & q, const gemv_seg * segs, int n_segs, int total_rows, const int8_t * xq,
                              const float * sx_dev, const int32_t * xsum_dev);

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

// copy rows of `cols` elements: out[row][i] = x[row][i]
void vit_copy_launch(sycl::queue & q, const float * x, int x_stride, float * out, int out_stride, int rows, int cols);

// ---------------------------------------------------------------------------
// Audio encoder kernels (src/kernels/at.cpp).
// ---------------------------------------------------------------------------

// 1D conv (reference conv1d_all in audio_model.cpp):
//   out[t][o] = b[o] + sum_tap sum_i w[o][tap*x_in+i] * x[t*stride+tap-pad][i]
// with zero padding outside the input; `w` is f32 [w_out][taps*x_in].
void at_conv1d_launch(sycl::queue & q, const float * x, int x_frames, int x_in, int taps, int stride, int pad,
                      const float * w, const float * b, float * out, int y_frames, int w_out);

// 1D RoPE on the Q/K halves of a fused qkv buffer ([n_tok][3*n_embd]):
// every pair of token t uses position t with the standard inverse frequencies.
void at_rope1d_launch(sycl::queue & q, float * qkv, int qkv_stride, int n_tok, int n_head, int head_dim,
                      float rope_base);

} // namespace si
