#pragma once
// ---------------------------------------------------------------------------
// Plain native-width weight packing (w4) for the 4-bit prefill path.
//
// The GGUF K-quant types already store every weight on a per-32-group uniform
// grid,  w = step_g * q - offset_g,  with q at the type's native bit width and
// (step_g, offset_g) = (d*sc_j, dmin*m_j) read straight from the super-block.
// Measured against the model's own dequantized values (relative L2, per row):
//
//     int8 per-row (the previous conversion)    0.98 %
//     u4 with per-group min/max recomputed      4.39 %
//     u4 with the native (step, offset)         0.077 %   (f16 metadata)
//     u4 with the native (step, offset)         0         (f32 metadata)
//
// So keeping (q, step_g, offset_g) is both *lossless* and 1.6x smaller than
// int8: it is the weight representation for the 4-bit path.  The only error is
// the f16 rounding of the metadata, and it replaces ~1% of re-quantization
// error that the int8 conversion *adds*, so 4-bit is strictly more accurate.
//
// Layout (consumed by oneDNN as weights {K,N} format_tag::ba + grouped scales):
//     vals [((size_t)n * K + k) >> 1]  nibble (k & 1)     u4, K inner, lo first
//     scale[g * N + n]                 f16 step           (oneDNN grouped scales)
//     off  [g * N + n]                 f16 offset         (correction coeff)
// with g = k / 32.  Two separate f16 planes, because oneDNN's grouped-scale
// memory must be a contiguous f16 tensor in exactly that order (validated),
// and the correction reads the offset plane as a {NG,N} f16 matrix.
// Total K*N/2 + (K/32)*N*4 bytes, versus K*N for int8.
//
// The prefill matmul is
//     out = oneDNN(u4 weights, grouped f16 scales = step)  +  correction
//     correction[m][n] = sum_g offset[n][g] * XS[m][g]
// where XS[m][g] = sum_{k in g} x[m][k] is the activation group sum: the
// zero-point moves into a small M x (K/32) x N term instead of needing
// oneDNN's grouped zero-points (whose layout did not validate).  Since
// w = offset + step*q, the correction coefficient is exactly offset, so it
// needs no metadata beyond the same f16 offset the packing already stores.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <vector>

namespace si {

constexpr int kW4Group = 32;

struct w4t {
    std::vector<uint8_t> vals;   // [N][K] u4, K inner, low nibble first
    std::vector<uint16_t> scale; // [K/32][N] f16 step
    std::vector<uint16_t> off;   // [K/32][N] f16 offset
    int K = 0;
    int N = 0;
    int bits = 4;
    bool ok() const {
        return !vals.empty() && !scale.empty() && !off.empty();
    }
};

// True when this ggml type has a native-width linear grid here.
bool w4_supported(uint32_t ggml_type);

// Pack one GGUF tensor at its native width.  Returns false (and leaves `out`
// untouched) for unsupported types, which keep the int8 conversion.
bool w4_pack(uint32_t ggml_type, const void * src, int K, int N, w4t & out);

} // namespace si
