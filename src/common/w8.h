#pragma once
// ---------------------------------------------------------------------------
// SIn weight format ("sycl-infer, n-bit"), group-major layout.
//
// Motivation: the GGUF K-quant layouts (Q4_K/Q5_K/Q6_K) are designed for
// scalar/byte-wise dequantization.  On this GPU (Iris Xe-LP) that path is
// limited by instructions and load sectors, while hardware DP4A gives 4x the
// raw throughput of FP32 FMA.  SIn keeps the *exact* integer values stored in
// the GGUF file (0..15 / 0..31 / 0..63) packed at their native bit width and
// only changes the layout and the scale/min representation:
//
//   w = scale * q - min          (q = original integer)
//   sum_k w*x = sx * (scale * sum q*qx - min * sum qx),  x ~= sx*qx
//
// Layout (the important part): values are k-group-major and blocked over rows
// so a warp's loads are contiguous:
//
//   row block RB = 128 rows, group G values (G = 32 for Q4_K/Q5_K, 16 for
//   Q6_K, matching the GGUF scale granularity):
//     vals[((rb*MG + g)*RB + ri) * GB]      packed k-bit unsigned values
//     meta[((rb*MG + g)*RB + ri)]           uint32, {fp16 scale, fp16 min}
//     MG = K/G groups per row, GB = G*k/8 bytes per group
//
// Q6_K groups (type 14) hold w = scale*(q-32) with no free min: the min is
// exactly 32*scale, so the meta plane stores only the fp16 scale (2 bytes per
// group instead of 4) and the read path derives the min.  This removes the
// whole min half of the Q6_K meta stream (the LM head is the largest tensor in
// the model and Q6_K: 60 MB of the 242 MB it reads per token are meta).
//
// Packing per group (little-endian byte order):
//   k=4 (G=32, 16 B): bytes 0..7  -> values 0..15, byte b = v[b] | v[b+8]<<4
//                     bytes 8..15 -> values 16..31, same interleave
//   k=5 (G=32, 24 B): like k=4 plus a 1-bit plane, 4 bytes of padding
//   k=6 (G=16, 12 B): two 8-value sub-groups of 6 bytes, value i at bit 6*i
//
// Activations use the same idea with symmetric int8 groups of 32 (TB tokens):
//     x8[(g*TB + t)*32 + i], xmeta[(g*TB + t)] = fp32 scale,
//     xsumq per 16 values (K/16 entries) so 16-value weight groups (Q6_K) can
//     use their own min correction.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>

namespace si {

constexpr int kRB = 128; // weight row-block for the SIn layout

struct w8t {
    uint8_t * vals = nullptr;  // packed k-bit values, see layout above
    uint32_t * meta = nullptr; // {fp16 scale, fp16 min} (4 B/group), or the
                               // fp16 scale alone when meta_elem == 2 (Q6_K
                               // tensors whose groups all satisfy
                               // min == 32*scale exactly)
    int32_t K = 0;
    int32_t N = 0;
    uint32_t type = 0;     // source ggml type (12/13/14)
    int32_t bits = 0;      // packed value width (4/5/6/8)
    int32_t meta_elem = 4; // bytes per meta entry of THIS tensor (2 or 4)
    bool ok() const {
        return vals != nullptr;
    }
};

// host-side repack of one GGUF K-quant tensor into the SIn layout.
// `meta_out` must have room for w8_meta_bytes(type, K, N, scale_only) bytes
// (it is uint32-typed for the callers' convenience, but scale_only writes
// 16-bit entries).
// `scale_only`: write just the fp16 scale (the kernel derives min == 32*scale).
// returns false if the type is unsupported.
bool w8_repack(uint32_t ggml_type, const void * src, int K, int N, uint8_t * vals_out, uint32_t * meta_out,
               bool scale_only = false);

// Q6_K scale_only is valid only if every group's fp16 scale is exact enough
// that 32*fp16(scale) == fp16(32*scale): a flushed-to-zero fp16 scale with a
// non-zero min would change the reconstructed weights.  Returns false for
// other types, under PF_SI4, or when any group fails the test.
bool w8_q6_scale_only_ok(uint32_t ggml_type, const void * src, int K, int N);

// device buffer sizes for one tensor (accounts for the PF_SI4 override).
// w8_meta_count is the number of meta *entries* (groups); w8_meta_bytes is the
// size of the plane (2 bytes/entry when scale_only).
size_t w8_vals_bytes(uint32_t type, int K, int N);
size_t w8_meta_count(uint32_t type, int K, int N);
size_t w8_meta_bytes(uint32_t type, int K, int N, bool scale_only = false);
// the layout type the kernels must use (12 = 4-bit G=32 when PF_SI4 is set)
uint32_t w8_effective_type(uint32_t type);

// values per group (the meta granularity) and bytes of packed values per group
static inline constexpr int w8_group_size(uint32_t type) {
    return type == 14 ? 16 : 32;
}
static inline constexpr int w8_group_bytes(uint32_t type) {
    // Q5_K groups carry 4 bytes of padding so rows stay 8-byte aligned
    if (type == 13) {
        return 24;
    }
    const int k = type == 12 ? 4 : 6;
    return w8_group_size(type) * k / 8;
}
static inline constexpr int w8_meta_per_row(uint32_t type, int K) {
    return K / w8_group_size(type);
}
// dp4a words produced per packed group (4 values per word)
static inline constexpr int w8_words_per_group(uint32_t type) {
    return w8_group_size(type) / 4;
}

} // namespace si
