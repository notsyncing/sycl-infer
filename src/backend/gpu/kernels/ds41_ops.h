// ---------------------------------------------------------------------------
// DeepSeek-V4.1-Flash device ops that are not covered by ds41_iq2.h,
// ds41_attn.h or ds41_hc.h.  All pointers are device USM and the caller's
// queue orders the work.  The CPU oracles are the matching functions in
// src/engine/engine_ds41.cpp (cited per op in reports/ds41_gpu_design.md §5.2).
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include <sycl/sycl.hpp> // IWYU pragma: keep

namespace si {

// In-place E4M3 rounding with one power-of-two scale per `block` values
// (window KV, attention output).  Matches `fp8_round_row`, bit-exact.
void ds41_fp8_round_launch(sycl::queue & q, float * v, int n, int block = 32);

// out = silu(min(g, lim)) * clamp(u, -lim, lim).  One-sided gate clamp.
void ds41_swiglu_clamp_launch(sycl::queue & q, const float * g, const float * u, float * out, int n, float lim);

// MoE router for `n_tok` rows: logits = x·gate_inp^T (raw BF16/F32), then
// score = sqrt(softplus(logit)); the bias (bias_vl when use_vl) is added only
// to the selection score; top-k ties to the lowest index; weights are
// score/(sum+1e-20)*scale.  `ids`/`wts` are [n_tok][top_k].
void ds41_router_launch(sycl::queue & q, const void * gate_inp, uint32_t type, const float * x, int n_tok,
                        int n_embd, const float * bias, const float * bias_vl, int use_vl, int n_expert,
                        int top_k, float scale, float * logits, int32_t * ids, float * wts);

// Top-k over `scores[0..n)`: descending score, ascending index on ties, output
// written in ascending index order (the attention gather order).
void ds41_select_launch(sycl::queue & q, const float * scores, int n, int k, int32_t * ids_ascending);

// Candidate blocks: per-block max over `block` compressed rows, +inf on the
// newest visible block, then top-`min(topk_blocks, nb)` blocks into `keep`
// ([nb] bytes, 0/1).  `nb` is written as the block count.
void ds41_cand_blocks_launch(sycl::queue & q, const float * score, int n_vis, int block, int topk_blocks,
                             uint8_t * keep, int * nb);

// Mask scores outside the kept blocks with -inf (in place).
void ds41_cand_mask_launch(sycl::queue & q, float * score, const uint8_t * keep, int n_vis, int block);

// Gather selected compressed rows: out[i][0..d) = ckv[sel[i]][0..d).
void ds41_gather_sel_launch(sycl::queue & q, const float * ckv, const int32_t * sel, int n_sel, int d, float * out);

// Write `row[0..d)` into the window ring at `pos % cap` and store `pos`.
void ds41_window_write_launch(sycl::queue & q, const float * row, float * win, int * win_pos, int pos, int cap,
                              int d);

// Engram apply: dot_s = sum_e rms(h)[e]*rms(key)[e]*gate[e] / sqrt(d); the
// result is gated by sign(dot)*sqrt(max(|dot|,1e-6)) and added to h.
void ds41_engram_apply_launch(sycl::queue & q, float * h, const float * kv, const float * gate, int hc, int d);

} // namespace si
