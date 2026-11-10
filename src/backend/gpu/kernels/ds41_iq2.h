#pragma once
// ---------------------------------------------------------------------------
// Raw (no expansion pass) fp32 GEMV over the ggml IQ2 family used by
// DeepSeek-V4.1-Flash's routed experts:
//
//   IQ2_XXS (ggml type 16, 66 B / 256 values)
//   IQ2_XS  (ggml type 17, 74 B / 256 values)
//   IQ2_S   (ggml type 22, 82 B / 256 values)
//
// The weight row is read directly in its GGUF block layout.  One 32-lane
// sub-group owns one output row.  The default mapping (`variant = 0`) puts
// lane `l` on the 8-value codebook group `l` (`[8l, 8l+8)`): the group's
// eight values share one codebook lookup, sign pattern and sub-block scale,
// which measured fastest.  `variant = 1` is the coalesced value-per-lane A/B
// (lane `l` on value `32*b + l`): its activation reads are perfectly coalesced
// but it needs one codebook lookup per lane per 32-value sub-block and is
// 1.6-3x slower on the real tensors.  Both accumulate fp32 and reduce the 32
// lane partials with the usual XOR tree.
//
// `ds41_iq2_gemv_launch` computes, for n in [0, N_out) and m in [0, M):
//
//   y[m][n] = dot(dequant(w[expert][n][0..K]), x[m][0..K])
//
// with the weight row read from
//
//   w_row_base + expert * plane_bytes + n * (K/256) * block_bytes(type).
//
// `w_row_base` is the tensor base (expert 0's plane).  Pass `expert = 0` and
// `plane_bytes = 0` when the pointer already names one expert's plane.
//
// The codebook/scale tables are device USM memory (see ds41_iq2_tables.h and
// dev/ds41_iq2_probe.cpp): a 14.5 KB `ds41_iq2_tables` blob allocated once and
// passed here.  A global-memory blob, rather than a constexpr array, keeps the
// 8 KB IQ2_S codebook out of the constant address space: the 32 lanes of a
// warp index it independently, and divergent constant-cache accesses would
// serialize.
// ---------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp> // IWYU pragma: keep

#include "ds41_iq2_tables.h"

namespace si {

constexpr int kDs41Iq2QK = 256; // values per IQ2 super-block

// Bytes per 256-value IQ2 super-block, or 0 for a non-IQ2 type.
inline constexpr size_t ds41_iq2_block_bytes(uint32_t type) {
    switch (type) {
    case 16: return 2 + 32 * 2;              // IQ2_XXS: f16 d + 8 x (4 grid + 4 sign/scale) bytes
    case 17: return 2 + 32 * 2 + 8;          // IQ2_XS: + 8 sub-block scales
    case 22: return 2 + 64 + 8 + 8;          // IQ2_S: f16 d + 64 qs + qh[8] + scales[8]
    default: return 0;
    }
}

// Number of output rows one sub-group (warp) computes per grid group.
inline constexpr int kDs41Iq2RowsPerWg = 8; // 8 sub-groups of 32 lanes = 256 threads

// Launch the raw IQ2 GEMV.
//
//   q            in-order or out-of-order SYCL queue
//   w_row_base   device pointer to expert 0's plane of the [K][N][E] tensor
//   K            input length; must be a multiple of 256
//   N_out        output rows (dims[1] of the expert tensor)
//   type         ggml type: 16 (IQ2_XXS), 17 (IQ2_XS), 22 (IQ2_S)
//   expert       expert index; row base = w_row_base + expert * plane_bytes
//   plane_bytes  bytes per expert; 0 = derive as N_out * (K/256) * block_bytes
//   d_x          device input activations, M rows of K floats
//   d_y          device output, M rows of N_out floats
//   M            rows to batch (1..8); the weights are read once for all M rows
//   x_stride     floats between x rows (0 = K)
//   y_stride     floats between y rows (0 = N_out)
//   d_tables     device copy of the IQ2 codebook blob (must be non-null)
//   variant      0 = group-per-lane (default), 1 = coalesced value-per-lane A/B
void ds41_iq2_gemv_launch(sycl::queue & q, const void * w_row_base, int K, int N_out, uint32_t type,
                          int expert, size_t plane_bytes, const float * d_x, float * d_y, int M = 1,
                          int x_stride = 0, int y_stride = 0, const ds41_iq2_tables * d_tables = nullptr,
                          int variant = 0);

} // namespace si
