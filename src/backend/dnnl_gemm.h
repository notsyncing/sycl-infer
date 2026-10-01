#pragma once
// ---------------------------------------------------------------------------
// oneDNN GPU int8 matmul for the M-tiled (chunk-batched, mode 2) prefill
// GEMMs, on by default; PF_GEMM_DNNL=0 disables it (the dp4a path runs and no
// allocation happens).
//
// Weight conversion: each GGUF K-quant tensor (Q4_K/Q5_K/Q6_K) is dequantized
// row by row and re-quantized to row-major int8 [N][K] with one symmetric
// scale per output row (the oneDNN "ba" weight layout, which the GPU int8
// matmul reads at full speed).  Activations are quantized per row too, once
// per call and shared by all tensors of that call (the ffn_down input is
// silu(gate)*up, applied during quantization).
//
// The matmul itself is a plain dnnl::matmul (src [M,K] s8, weights [K,N] s8
// in "ba" order, dst [M,N] s32) with no scales attribute: the per-row scales,
// the residual add and alpha are applied by a small elementwise epilogue
// kernel, so the integer accumulation stays exact.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <memory>
#include <sycl/sycl.hpp> // IWYU pragma: keep

namespace si {

// true unless PF_GEMM_DNNL=0 (read once per process)
bool dnnl_gemm_enabled();

struct dnnl_gemm {
    explicit dnnl_gemm(sycl::queue & q);
    ~dnnl_gemm();
    dnnl_gemm(const dnnl_gemm &) = delete;
    dnnl_gemm & operator=(const dnnl_gemm &) = delete;

    // Convert one tensor.  `key` is the w8t.vals pointer of the matching SIn
    // copy (the DNNL path keys its cache by it); `host_data` is the mapped GGUF
    // tensor.  Returns false for unsupported types/shapes.
    bool add_weight(const void * key, const void * host_data, uint32_t ggml_type, int K, int N);
    bool has_weight(const void * key) const;

    // ---- 4-bit (u4) weights (see common/w4.h) -----------------------------
    // Native-width packing of a Q4_K tensor: u4 values [N][K] plus per-32-group
    // f16 step/offset planes.  Lossless within the f16 metadata (0.077% error
    // vs the int8 conversion's 0.98%) at 0.625 bytes/weight instead of 1.0.
    // The matmul keeps oneDNN's grouped f16 weight scales (= step) and moves
    // the offset (zero-point) into a separate correction term, because
    // oneDNN's grouped zero-point descriptors do not validate.
    // any_type: pack a type with no native 4-bit grid (the draft LM head).
    // gemv_only: skip the prefill primitive ladder - the consumer is the M=1
    // u4 GEMV only (the MTP draft layer, which never prefills).
    bool add_weight_w4(const void * key, const void * host_data, uint32_t ggml_type, int K, int N,
                       bool any_type = false, bool gemv_only = false);
    bool has_weight_w4(const void * key) const;

    // ---- native-width 5-bit weights (Q5_K, see common/w4.h) ---------------
    // The 4-bit nibble plane plus a 1-bit fifth-bit plane and the same two
    // per-(g,n) f16 planes the u4 path uses: 0.75 B/weight against the int8
    // conversion's 1.125, with the native values kept exactly (no tensor is
    // re-quantized).  Decode recombines the planes in-kernel (k5_gemv_launch,
    // measured at the card's read ceiling); prefill expands q5 to int8 into the
    // shared scratch and runs the grouped-scale int8 primitive with the u4
    // offset-correction epilogue.
    bool add_weight_k5(const void * key, const void * host_data, uint32_t ggml_type, int K, int N);
    bool has_weight_k5(const void * key) const;

    // ---- codebook 4-bit weights (IQ4_XS / IQ4_NL, see common/w4.h) ---------
    // The native nibble indices plus one f16 scale per (32-value group, row):
    // 0.5625 B/weight instead of the int8 conversion's 1.0625, with the native
    // values kept exactly.  Decode expands the 16-entry table in-kernel;
    // prefill expands the indices to int8 into a reused scratch (the biggest
    // tensor, ~95 MB) and runs the ordinary int8 primitive over it.
    bool add_weight_cb4(const void * key, const void * host_data, uint32_t ggml_type, int K, int N);
    bool has_weight_cb4(const void * key) const;
    // diagnostics: total device bytes held by this instance's weight tables
    // (int8 + u4 + codebook + the codebook prefill scratch)
    size_t weight_bytes() const;
    // Same contract as gemm(); out = alpha*sx[m]*(acc4 + correction) + residual.
    bool gemm_w4(const void * key, const float * residual, float alpha, int M, int K, float * out, int out_stride);

    // Quantize one call's activations: x is token-major [M][x_stride]; if up is
    // non-null the value is silu(x)*up (ffn_down).  The result stays valid
    // until the next quantize() call.
    // do_split also produces the even/odd k deinterleave of the grouped
    // activation (the u4 decode GEMV's operands); prefill only needs xq + asa +
    // xs, so it passes false and skips the extra stores.
    bool quantize(const float * x, const float * up, int x_stride, int up_stride, int M, int K,
                  bool do_split = true);
    // out[m][row] = alpha * sx[m]*sw[row]*acc[m][row] + residual[m][row]
    // (acc from the oneDNN matmul of the currently quantized activations).
    // Returns false and writes nothing when the tensor/shape is unsupported.
    bool gemm(const void * key, const float * residual, float alpha, int M, int K, float * out, int out_stride);

    // Execute every cached primitive once on dummy data so the first real
    // request does not pay the one-time kernel load.  Call after all weights
    // were added.  Returns the number of primitives executed.
    int warmup();

    // ---- diagnostics (dev checks) ----
    const int8_t * weight_data(const void * key) const;
    const float * weight_scales(const void * key) const;
    // The grouped int8 weight's per-32-group f16 step plane ([K/32][N], the
    // layout i8_grp_gemv_launch and the MTP candidate head read); null for the
    // u4/k5/cb4 entries, which carry their own scale planes.
    const uint16_t * weight_group_scales(const void * key) const;
    // One launch for a whole call group of *narrow* int8 (grouped-scale)
    // segments: they share the quantized activation and the K, and each of them
    // is small enough that a per-segment launch's latency dominates its 0.26 MB
    // (the GDN's ssm_alpha / ssm_beta, 48 rows each - see kernels.h).  `keys`
    // are the segments' oneDNN keys, `n_rows` their output rows.  Returns false
    // without launching anything when the group is not eligible (a non-int8
    // segment, mixed K, more than four segments, or more rows than the narrow
    // cutoff), so the caller keeps its per-segment path.
    bool gemm_i8_group(const void * const * keys, const int * n_rows, const float * const * outs,
                       int out_stride, const float * const * residuals, const float * alphas, int n_segs, int M,
                       int K);
    const int8_t * act_data() const;
    // u4 path: the per-32-group quantized activations and their f16 scales
    // (device pointers), produced alongside the per-row form when a u4 weight
    // has been registered.
    const int8_t * act_grp_data() const;
    const uint16_t * act_grp_scales() const;
    const float * act_scales() const;
    // Per-32-group f32 sum of the quantized activations ([M][K/32]): the XOR-bias
    // correction the u4/k5/int8 grouped GEMVs apply, and the MTP candidate head's.
    const float * act_group_sums() const;
    // per-row sum of the quantized activations (int32, device memory) used by
    // the dp4a decode GEMV's weight bias correction
    const int32_t * act_sum() const;

    // ---- XMX (int8) attention GEMMs (PF_ATTN_XMX, see attn.cpp) -----------
    // Plain int8 matmuls with an s32 accumulator for the two attention GEMMs:
    //   QK: [M, K=256] s8  x  [K, N=blk] s8 -> [M, N] s32
    //   PV: [M, K=blk] u8  x  [K, N=256] s8 -> [M, N] s32
    // The K operand of QK is stored [N, K] (per-key row of head_dim), so it is
    // read with a "ba" (transposed) weight descriptor.  No scales attribute:
    // the attention path rescales K/V/i8 to a single per-row scale at gather
    // time and applies the remaining scales in its own epilogue.  Shapes are
    // cached; the caller owns the operand/dst USM buffers.  Returns false when
    // the shape is unsupported (the caller falls back to the classic kernel).
    bool attn_qk(int M, int blk, const int8_t * q_kmajor, const int8_t * k_keymajor, int32_t * dst);
    bool attn_pv(int M, int blk, const uint8_t * p_major, const int8_t * v_keymajor, int32_t * dst);

private:
    struct impl;
    std::unique_ptr<impl> p;
};

// Queue -> dnnl_gemm registry: the attention kernels (src/backend/gpu/kernels/
// attn_xmx.cpp) are launched through the backend abstraction, which does not
// carry the engine's per-device dnnl_gemm table, so the engine registers each
// instance here at construction and the XMX attention looks it up by queue.
void dnnl_register_queue(sycl::queue & q, dnnl_gemm * g);
dnnl_gemm * dnnl_for_queue(sycl::queue & q);

} // namespace si
