// ---------------------------------------------------------------------------
// DeepSeek-V4.1-Flash MLA attention + compressor / indexer kernels.
//
// The model's attention is not GQA: there is ONE 512-wide K == V "head" shared
// by 64 query heads, and every query attends to
//   * the 128-token sliding-window ring (rows in ascending position order),
//   * up to 512 selected compressed rows (already gathered in the reference's
//     ascending selection order),
// with the per-head sink as an extra value-0 key.  Output = sum p_i * k_i
// (V == K), then the inverse RoPE over the LAST `n_rot` dims at the query
// position.
//
// All launchers take device USM pointers and are ordered by the caller's queue.
// ---------------------------------------------------------------------------
#pragma once

#include <sycl/sycl.hpp>

namespace si {

// One query row, one layer.
//   q       [n_head][head_dim]         q_b output, q-RoPE already applied
//   win     [win_cap][head_dim]        the layer's window ring
//   win_pos [win_cap]                  position stored in each ring slot, or -1
//   ckv     [n_sel][head_dim]          selected compressed rows, reference order
//   sink    [n_head]                   attn_sinks.weight
//   freq    [n_rot/2]                  RoPE frequency table for this layer
//   out     [n_head][head_dim]         attention output, de-roped in place
// `n_win` is the number of visible window rows; row j is position
// `win_start + j`, slot `(win_start + j) % win_cap`, and is skipped when the
// ring's stored position disagrees.  `scale` is 1/sqrt(head_dim).
void ds41_mla_attn_launch(sycl::queue & q, const float * qv, const float * win, const int * win_pos,
                          int win_cap, const float * ckv, int n_sel, const float * sink, int n_head,
                          int head_dim, int pos, int n_win, int win_start, float scale,
                          const float * freq, int rot_off, int n_rot, float * out);

// ratio-way softmax-gated sum per column, then RMSNorm with `norm`.
//   kvc   [n_groups*ratio][d]      raw compressor.wkv rows
//   gate  [n_groups*ratio][d]      raw compressor.wgate rows (ignored if ratio==1)
//   norm  [d]
//   out   [n_groups][d]            latent rows (pre-RoPE, pre-fp4)
void ds41_compressor_launch(sycl::queue & q, const float * kvc, const float * gate, const float * norm,
                            float * out, int n_groups, int ratio, int d, float eps);

// In-place RoPE on the LAST `rd` dims of each row, starting at `off`.
// Row r uses position `pos0 + r*pos_stride`.  `inverse` rotates by -angle.
void ds41_rope_launch(sycl::queue & q, float * v, int n_rows, int row_stride, int off, int rd, int pos0,
                      int pos_stride, const float * freq, bool inverse);

// In-place E2M1/E4M3 "fp4" block rounding (reference `fp4_round_row`).
// `block` values per group; `e4m3_scale` selects the E4M3 (vs power-of-two)
// per-group scale.
void ds41_fp4_round_launch(sycl::queue & q, float * v, int n, int block, bool e4m3_scale);

// DSA lightning-indexer score for one query:
//   score[r] = sum_h relu(dot(qi_h, ik_r)) * w[h]
//   qi    [n_head][hdim]
//   ik    [n_rows][hdim]
//   w     [n_head]
//   score [n_rows]
void ds41_indexer_score_launch(sycl::queue & q, const float * qi, const float * ik, const float * w,
                               float * score, int n_rows, int n_head, int hdim);

} // namespace si
