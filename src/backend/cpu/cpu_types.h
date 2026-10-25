#pragma once
// ---------------------------------------------------------------------------
// CPU backend shared types and kernel launch declarations.
//
// This header is deliberately SYCL-free: the CPU kernels in
// src/backend/cpu/kernels/ are compiled with `-fno-sycl` (see CMakeLists) so
// they are plain host translation units that can carry AVX2 / AVX-VNNI /
// AVX-512 intrinsics without dragging the SYCL device pass along.  The
// SYCL-aware backend (src/backend/cpu/cpu_backend.cpp) converts `step_info` /
// `gemv_seg` into the mirror structs below at the call boundary.
//
// The launch signatures mirror src/backend/gpu/kernels/kernels.h one-to-one so
// the backend can forward them.  Every hot loop is dispatched on the host ISA
// by src/common/cpu_isa.h; correctness is independent of the chosen variant.
// ---------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>

#include "quant.h"
#include "w8.h"
#include "model/gguf.h" // the ggml_type enum (SYCL-free)

namespace si {

// mirror of kv_dtype_t (src/backend/gpu/kernels/kv_type.h) without the SYCL include
enum class cpu_kv_dtype : int { f32 = 0, bf16 = 1, f16 = 2, i8 = 3, i4 = 4 };

inline int cpu_kv_elem_bytes(cpu_kv_dtype t) {
    return t == cpu_kv_dtype::f32 ? 4 : ((t == cpu_kv_dtype::i8 || t == cpu_kv_dtype::i4) ? 1 : 2);
}

constexpr int kCpuI8Q = 32;

// mirror of step_info; only the fields the CPU kernels read are carried over
struct cpu_step_info {
    int32_t n_rows;
    int32_t n_real;
    int32_t tpb;
    // per-row real token count for chunk-batched prefill (mirror of step_info)
    int32_t n_real_row[16];
    int32_t pos[16];
    int32_t slot[16];
    int32_t active[16];
    int32_t tokens[16 * 32];
    int32_t mtp_dt;
    int32_t mtp_dry;
    int32_t pc_active;
    int32_t pc_stride;
    float * pc_base;
    int32_t pc_row_slot[1024];
    int32_t mrope_on;
    int32_t mrope_sections[4];
    int32_t mrope[4 * 16 * 32];
    const float * img_embd;
    int32_t img_row[16 * 32];
};

// mirror of gemv_seg.  `meta32` is the fp32 (scale, min) side array reinterpreted
// as flat pairs [n_rows][K/32][2]; `w8`/`x8`/`xmeta`/`xsumq` carry the SIn int8
// path and are null on the fp32 path.
struct cpu_gemv_seg {
    const void * w = nullptr;
    uint32_t type = 0;
    int32_t K = 0;
    int32_t n_rows = 0;
    const float * x = nullptr;
    int32_t x_stride = 0;
    const float * act_up = nullptr;
    float * out = nullptr;
    int32_t out_stride = 0;
    const float * residual = nullptr;
    float alpha = 1.0f;
    const float * meta32 = nullptr;
    w8t w8;
    const int8_t * x8 = nullptr;
    const float * xmeta = nullptr; // pair (scale, 0) per 32-group of activations
    const int32_t * xsumq = nullptr;
    bool i8 = false; // integer GEMV from the GGUF blocks in `w` (CPU path)
};

// mirror of pc_snap
struct cpu_pc_snap {
    float * base = nullptr;
    int64_t stride = 0;
    int64_t layer_off = 0;
    int32_t gdn_per = 0;
    int32_t conv_per = 0;
    // Total size of the checkpoint pool in floats.  The kernel computes its
    // write offset from a slot index and a layer offset that it cannot otherwise
    // bound, so the bound has to travel with the descriptor; 0 disables the
    // check.  See the pcguard reports in cpu_gdn / cpu_conv_state_update.
    int64_t cap_floats = 0;
};

// One flag for both snapshot sites, read once: the check is a single comparison
// against a bound the kernel does not otherwise have.
extern const bool cpu_pc_guard;

void cpu_rmsnorm(const float * x, const float * w, float * out, int n_rows, int n, float eps);
void cpu_copy_row(const float * src, float * dst, const cpu_step_info * info, int n, int row);
void cpu_embed(const void * table, uint32_t type, const cpu_step_info * info, float * out, int n_embd,
               size_t row_bytes);
void cpu_gemv_group(uint32_t type, const cpu_gemv_seg * segs, int n_segs, int total_rows, int TB, int nsb,
                    int n_tok_blocks, int n_threads);
void cpu_qk_norm_rope(float * qbuf, float * kbuf, float * vbuf, const float * q_norm, const float * k_norm, void * kpool,
                      void * vpool, const int32_t * tables, const cpu_step_info * info, int n_head, int n_head_kv,
                      int head_dim, int n_rot, float rope_base, const float * rope_freqs, float rope_mscale, float eps, int max_blocks,
                      int n_rows, int n_real, cpu_kv_dtype kkv, cpu_kv_dtype vkv, const void * kscales,
                      const void * vscales);
void cpu_attn(const float * qbuf, const float * gate, const void * kpool, const void * vpool, float * partials,
              const int32_t * tables, int n_head, int n_head_kv, int head_dim, int n_splits,
              const cpu_step_info * info, float scale, int max_blocks, int n_rows, int n_real, float * out,
              cpu_kv_dtype kkv, cpu_kv_dtype vkv, const void * kscales, const void * vscales);
void cpu_attn_combine(const float * partials, const float * gate, float * out, const cpu_step_info * info, int n_head,
                      int head_dim, int n_splits, int n_rows, int n_real);
void cpu_conv_l2(const float * qkv_raw, float * conv_state, const float * conv_w, float * conv_out,
                 const cpu_step_info * info, int conv_dim, int kernel_size, int head_k_dim, int n_k_heads, float eps,
                 int n_rows, int n_real, int row0, int tpb_arg, bool cross_row);
void cpu_conv_state_update(const float * qkv_raw, float * conv_state, const cpu_step_info * info, int conv_dim,
                           int kernel_size, int n_rows, int row0, int tpb_arg, int nreal_arg, bool last_row_only,
                           cpu_pc_snap snap);
void cpu_gdn(const float * conv_out, const float * alpha, const float * dt_bias, const float * ssm_a,
             const float * beta, float * state, float * attn_out, const cpu_step_info * info, int head_dim,
             int n_k_heads, int n_heads, int conv_dim, float scale, int n_slots, int n_rows, int row0, int tpb_arg,
             int nreal_arg, cpu_pc_snap snap);
void cpu_gated_norm(const float * attn, const float * z, const float * weight, float * out, const cpu_step_info * info,
                    int n_heads, int head_dim, float eps, int n_rows, int n_real, int row0, bool sigmoid_gate);

// activation quantization + SIn int8 GEMV/GEMM (DP4A path, AVX-VNNI when present)
void cpu_xq(const float * x, const float * up, int x_stride, int up_stride, int8_t * x8, float * xmeta, int32_t * xsumq,
            const cpu_step_info * info, int TB, int K);
void cpu_dp4a_gemv(const cpu_gemv_seg & seg, int n_threads);
void cpu_dp4a_gemm(const cpu_gemv_seg & seg, int TB, int n_threads);
// integer GEMV/GEMM from the original GGUF K-quant blocks (no SIn packing)
void cpu_i8_gemv(const cpu_gemv_seg & seg, int n_threads);
void cpu_i8_gemm(const cpu_gemv_seg & seg, int TB, int n_threads);

// number of worker threads the CPU backend uses (PF_CPU_THREADS overrides)
int cpu_thread_count();
// override the CPU worker thread count; 0 restores auto (PF_CPU_THREADS, else
// the hardware concurrency).  Must be called before the first CPU kernel runs.
void cpu_set_thread_count(int n);

// ---- routed-MoE mirror (see src/backend/cpu/kernels/moe.cpp) -------------------
// Raw host pointers only: this header is SYCL-free, so it cannot carry `moe_t`.
// engine_graph's MoE branch fills one of these per layer and dispatches to the
// CPU kernels when the layer's partition is a cpu device.
struct cpu_moe_wt {
    const char * data = nullptr;
    uint32_t type = 0;
    int K = 0;         // input width of one row
    int N = 0;         // rows
    size_t plane_bytes = 0;
};

struct cpu_moe {
    cpu_moe_wt gate_exps, up_exps, down_exps;
    const void * gate_inp = nullptr;            // router
    uint32_t gate_inp_type = GGML_TYPE_F32;
    const void * gate_inp_shexp = nullptr;      // shared expert's gate
    uint32_t gate_inp_shexp_type = GGML_TYPE_F32;
    int n_expert = 0, top_k = 0, n_ff_exp = 0, n_embd = 0;
    // optional qwen4exp down store (see model.h): uint8 q+128, one float scale
    // per 32-value group, [E][N][K] and [E][N][K/32]
    const uint8_t * down_i8 = nullptr;
    const float * down_sc = nullptr;
    int down_K = 0, down_N = 0, down_E = 0;
    // native down: read the raw GGUF blocks instead of the int8 store when the
    // file's down projection is Q5_1 (0.75 vs 1.125 B/weight).  plane_bytes is
    // per expert; row_bytes is one output row.
    const void * down_raw = nullptr;
    size_t down_row_bytes = 0;
    uint32_t down_type = 0;
};

void cpu_moe_route(const float * x, int x_stride, const void * gate_inp, uint32_t gp_type, int n_tok, int n_embd,
                   int n_expert, int top_k, float * logits, int32_t * ids, float * wts, bool fast_f32 = false);
void cpu_moe_experts(const cpu_moe & M, const float * x, int x_stride, int n_tok, int top_k, const int32_t * ids,
                     const int * slots, float * acc, int acc_stride, bool fast_f32 = false);
void cpu_moe_reduce(const float * acc, const float * wts, int top_k, int n_tok, int n_embd, int acc_stride, float * out,
                    int out_stride);
void cpu_moe_epilogue(const float * x, int x_stride, const void * gate, uint32_t gtype, int n_tok, int n_embd,
                      const float * shared, int sh_stride, float * out, int out_stride);

// ---- qwen4exp (hyper-connections + PLE) host kernels -----------------------
void cpu_q4_hc_norm(const float * x, const float * w, float * out, const int32_t * rows, int n_tok, int n_embd,
                    int hc, float eps);
void cpu_q4_hc_mix(const float * xn, const float * gate, float * mixed, const int32_t * rows, int n_tok, int n_embd,
                   int hc);
void cpu_q4_hc_combine(float * residual, const float * block_out, const float * inject, const int32_t * rows,
                       int n_tok, int n_embd, int hc);
void cpu_q4_silu_scaled(float * x, int n, float scale);
// Convert one routed-expert down tensor to the symmetric int8 store above.
void cpu_q4_convert_down(const void * src, uint32_t type, int K, int N, int E, size_t plane_bytes, uint8_t * dst,
                         float * sc);
// Grouped down projection: P pairs of one expert, weights read once per
// 64-pair tile.  `pairs` maps a local pair to the caller's acc row.
void cpu_q4_down_grouped(const uint8_t * w, const float * sc, const int8_t * x8, const float * xmeta,
                         const int32_t * xsumq, int xTB, int P, int N, int K, float * acc, int acc_stride,
                         const int * pairs);
void cpu_q4_down_grouped_serial(const uint8_t * w, const float * sc, const int8_t * x8, const float * xmeta,
                                const int32_t * xsumq, int xTB, int P, int N, int K, float * acc, int acc_stride,
                                const int * pairs);
void cpu_q5_1_down_grouped_serial(const void * w, size_t row_bytes, const int8_t * x8, const float * xmeta,
                                  const int32_t * xsumq, int xTB, int P, int N, int K, float * acc, int acc_stride,
                                  const int * pairs);
// Batch int8 GEMM straight from GGUF K-quant blocks (gate/up of one expert).
void cpu_i8_gemm_batch(uint32_t type, const void * w, const int8_t * x8, const float * xmeta, const int32_t * xsumq,
                       int K, int rows, int TB, float * out, int out_stride);
void cpu_i8_gemm_batch_serial(uint32_t type, const void * w, const int8_t * x8, const float * xmeta,
                              const int32_t * xsumq, int K, int rows, int TB, float * out, int out_stride);
void cpu_i8_expert_gemm(uint32_t type, const void * w, const int8_t * x8, const float * xmeta, const int32_t * xsumq,
                        int K, int rows, int TB, float * out, int out_stride);
// y[t][n] = sum_k W[n][k] * x[t][k] for a K that is not a multiple of 256 (the
// HC up-projection has K = hc_low_rank = 320); the ordinary GEMV works in
// 256-value superblocks and truncates such a row.
void cpu_q4_up_gemv(const void * w, uint32_t type, const float * x, int xs, float * out, int os, int n_tok, int K,
                    int N);
// arbitrary-K GEMV with an optional act_up (silu(x)*up) operand - the shared
// expert's down projection has K = 640, also not a 256 multiple
void cpu_q4_arb_gemv(const void * w, uint32_t type, const float * x, int xs, const float * up, float * out, int os,
                     int n_tok, int K, int N);
void cpu_q4_ple_gate(const float * key, const float * hidden, const float * w_key, const float * w_query, float * gate,
                     const int32_t * rows, int n_tok, int n_embd, int hc, float eps);
void cpu_q4_ple_apply(const float * value, const float * gate, const float * w_conv, const float * conv_w,
                      float * residual, float * gated, float * normed, float * hist, int32_t * n_hist,
                      const int32_t * rows, const int32_t * slots, const int32_t * poss, int n_tok, int n_embd, int hc,
                      int ngram, int kern, float eps);

} // namespace si
