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

    // Quantize one call's activations: x is token-major [M][x_stride]; if up is
    // non-null the value is silu(x)*up (ffn_down).  The result stays valid
    // until the next quantize() call.
    bool quantize(const float * x, const float * up, int x_stride, int up_stride, int M, int K);
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
    const float * act_scales() const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};

} // namespace si
