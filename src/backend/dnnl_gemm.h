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
    bool add_weight_w4(const void * key, const void * host_data, uint32_t ggml_type, int K, int N);
    bool has_weight_w4(const void * key) const;

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
    const int8_t * act_data() const;
    // u4 path: the per-32-group quantized activations and their f16 scales
    // (device pointers), produced alongside the per-row form when a u4 weight
    // has been registered.
    const int8_t * act_grp_data() const;
    const uint16_t * act_grp_scales() const;
    const float * act_scales() const;
    // per-row sum of the quantized activations (int32, device memory) used by
    // the dp4a decode GEMV's weight bias correction
    const int32_t * act_sum() const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};

} // namespace si
