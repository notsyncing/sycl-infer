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

#include "w8.h"

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
};

void cpu_rmsnorm(const float * x, const float * w, float * out, int n_rows, int n, float eps);
void cpu_copy_row(const float * src, float * dst, const cpu_step_info * info, int n, int row);
void cpu_embed(const void * table, uint32_t type, const cpu_step_info * info, float * out, int n_embd,
               size_t row_bytes);
void cpu_gemv_group(uint32_t type, const cpu_gemv_seg * segs, int n_segs, int total_rows, int TB, int nsb,
                    int n_tok_blocks, int n_threads);
void cpu_qk_norm_rope(float * qbuf, float * kbuf, float * vbuf, const float * q_norm, const float * k_norm, void * kpool,
                      void * vpool, const int32_t * tables, const cpu_step_info * info, int n_head, int n_head_kv,
                      int head_dim, int n_rot, float rope_base, float eps, int max_blocks, int n_rows, int n_real,
                      cpu_kv_dtype kkv, cpu_kv_dtype vkv, const void * kscales, const void * vscales);
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
                    int n_heads, int head_dim, float eps, int n_rows, int n_real, int row0);

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

} // namespace si
