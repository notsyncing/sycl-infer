#pragma once
// Grouped, CSR-driven tiled expert GEMM for the routed-expert prefill.
//
// This is dev/bf16_lab.cpp's measured tiled GEMM (BK=32, TM=TN=4, fp32 weight
// tile in SLM, software prefetch over the raw Q4_K/Q5_K superblocks) turned
// into the engine's grouped form: one workgroup per (active expert,
// BN-column block), looping that expert's pairs out of the CSR in BM-row tiles
// (BM = 8 / 32 / 64).  BN is 256 by default; an opt-in Xe2 build
// (SI_MOE_TILED_WIDE) also compiles a BN=512 gate/up tile.  The grid is
// n_act_max * (N/BN) workgroups.
//
// BM is chosen from the batch's average pair density (pairs / experts): the
// 64-row tile for a dense gate/up batch, 32 for down and for a normal density,
// 8 for a sparse tail.  With SI_MOE_TILED_WIDE a dense gate/up batch takes the
// 73.5 KB BM=64 x BN=512 tile instead; everything else stays at BN=256.
//
// Weight bank: `bank + slot_map[active[ia]] * slot_stride` is the expert's
// [N][K] matrix in native GGUF blocks, row n at `+ n * RB` with
// RB = (K/256)*144 for Q4_K (12) and (K/256)*176 for Q5_K (13).  The blocks are
// read where they lie; nothing is reshaped or copied.
//
// Activation: int8 `xq` with one f16 scale per 32-value group in `asa`.
//   top_k > 0: activation row = pair[p] / top_k  (gate/up; xq has n_tok rows)
//   top_k == 0: activation row = pair[p]         (down; xq has n_pairs rows
//                                                 indexed by pair id, because
//                                                 moe_actq quantizes d_moe_h,
//                                                 whose rows are pair ids -
//                                                 same contract as moe.cpp's
//                                                 `trowv[m] = pr` for FUSE==1)
//
// Output: `out[pair[p] * out_stride + cb*256 + n]`.  With `gate` non-null the
// value is multiplied by `silu(gate[pair[p] * out_stride + cb*256 + n])`, the
// up projection's epilogue over the gate projection's f32 output.

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

namespace si {

// One workgroup per (active expert, BN-column block): 512 threads at BN=256,
// 1024 at the opt-in BN=512.  The workgroup loops the expert's pairs from the
// CSR in BM-row tiles (BM = 8, 32 or 64, picked from the average pair density),
// so the number of workgroups depends only on n_act_max, N and BN.
// `top_k` > 0: activation row = pair[p] / top_k (gate/up, xq has n_tok rows);
// `top_k` == 0: activation row = pair[p] (down, xq has n_pairs rows).
// `gate` != null: out = silu(gate[row*out_stride+n]) * acc (the up epilogue).
bool moe_tiled_gemm_launch(sycl::queue & q, const void * bank, size_t slot_stride,
                           const int32_t * slot_map, const int32_t * off, const int32_t * pair,
                           const int32_t * active, const int32_t * n_active, int n_act_max,
                           const int8_t * xq, const uint16_t * asa, float * out, int out_stride,
                           int K, int N, int top_k, uint32_t type, const float * gate,
                           // The layer's total pair count (the engine knows it).
                           // Negative = read it from the device (one 4-byte D2H);
                           // passing it removes that drain from the prefill.
                           int n_pairs = -1);

// Declines (returns false) unless it can serve this shape/type: type must be
// Q4_K (12) for gate/up and Q4_K (12) or Q5_K (13) for down; N % 256 == 0
// (the BN=512 tile additionally needs N % 512 == 0); K % 256 == 0.
bool moe_tiled_supported(int K, int N, uint32_t type);

} // namespace si
