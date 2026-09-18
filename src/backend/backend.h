#pragma once
// ---------------------------------------------------------------------------
// Compute backend abstraction (shared by the two backends).
//
// The engine's forward pass is written against this interface; the GPU backend
// (src/backend/gpu/) forwards to the SYCL kernel library
// (src/backend/gpu/kernels/) and the host CPU backend (src/backend/cpu/) calls
// the intrinsic kernel library (src/backend/cpu/kernels/).  The launch
// signatures mirror kernels.h so the GPU backend is a thin forwarding layer and
// the CPU backend only has to convert the shared PODs (`step_info`,
// `gemv_seg`) into their SYCL-free mirrors.
//
// Backends are single-device: one backend owns one queue/device (or the host
// CPU).  Multi-device execution composes several backends, one per layer
// partition (see engine::layer_dev_).
// ---------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <sycl/sycl.hpp> // IWYU pragma: keep

#include "kernels.h"
#include "kv_type.h"

namespace si {

enum class device_kind : int { gpu = 0, cpu = 1 };

struct compute_backend {
    virtual ~compute_backend() = default;
    virtual const char * name() const = 0;
    virtual bool is_cpu() const = 0;
    virtual device_kind kind() const = 0;

    // ---- kernels (mirror src/backend/gpu/kernels/kernels.h) ----
    virtual void rmsnorm(const float * x, const float * w, float * out, int n_rows, int n, float eps) = 0;
    virtual void copy_row(const float * src, float * dst, const step_info * info, int n, int row) = 0;
    virtual void embed(const void * table, uint32_t type, const step_info * info, float * out, int n_embd,
                       size_t row_bytes) = 0;
    virtual void gemv_group(uint32_t type, const gemv_seg * segs, int n_segs, int total_rows, int TB, int nsb,
                            int n_tok_blocks) = 0;
    virtual void qk_norm_rope(float * qbuf, float * kbuf, float * vbuf, const float * q_norm, const float * k_norm,
                              void * kpool, void * vpool, const int32_t * tables, const step_info * info, int n_head,
                              int n_head_kv, int head_dim, int n_rot, float rope_base, float eps, int max_blocks,
                              int n_rows, int n_real, const void * kscales, const void * vscales) = 0;
    virtual void attn(const float * qbuf, const float * gate, const void * kpool, const void * vpool, float * partials,
                      const int32_t * tables, int n_head, int n_head_kv, int head_dim, int n_splits,
                      const step_info * info, float scale, int max_blocks, int n_rows, int n_real, float * out,
                      int group, const void * kscales, const void * vscales) = 0;
    virtual void attn_combine(const float * partials, const float * gate, float * out, const step_info * info,
                              int n_head, int head_dim, int n_splits, int n_rows, int n_real) = 0;
    virtual void conv_l2(const float * qkv_raw, float * conv_state, const float * conv_w, float * conv_out,
                         const step_info * info, int conv_dim, int kernel_size, int head_k_dim, int n_k_heads,
                         float eps, int n_rows, int n_real, int row0, int tpb_arg, bool cross_row) = 0;
    virtual void conv_state_update(const float * qkv_raw, float * conv_state, const step_info * info, int conv_dim,
                                   int kernel_size, int n_rows, int row0, int tpb_arg, int nreal_arg,
                                   bool last_row_only, pc_snap snap) = 0;
    virtual void gdn(const float * conv_out, const float * alpha, const float * dt_bias, const float * ssm_a,
                     const float * beta, float * state, float * attn_out, const step_info * info, int head_dim,
                     int n_heads, int conv_dim, float scale, int n_slots, int n_rows, int row0, int tpb_arg,
                     int nreal_arg, pc_snap snap) = 0;
    virtual void gated_norm(const float * attn, const float * z, const float * weight, float * out,
                            const step_info * info, int n_heads, int head_dim, float eps, int n_rows, int n_real,
                            int row0) = 0;

    // ---- SIn int8 (DP4A) path ----
    virtual void xq(const float * x, const float * up, int x_stride, int up_stride, int8_t * x8, sycl::float2 * xmeta,
                    int32_t * xsumq, const step_info * info, int TB, int K) = 0;
    virtual void dp4a_gemv(const w8t & w, const int8_t * x8, const sycl::float2 * xmeta, const int32_t * xsumq,
                           float * out, const float * residual, float alpha) = 0;
    virtual void dp4a_gemm(const w8t & w, const int8_t * x8, const sycl::float2 * xmeta, const int32_t * xsumq,
                           float * out, int out_stride, const float * residual, float alpha, int TB) = 0;
    // CPU integer path: the integer dot is computed straight from the GGUF
    // blocks referenced by `s.w` (activated into x8 by xq); the GPU does not
    // set s.i8 and may fall back to the fp32 segments.
    virtual void i8_gemv(const gemv_seg & s) = 0;
    virtual void i8_gemm(const gemv_seg & s, int TB) = 0;

    virtual void synchronize() = 0;
};

std::unique_ptr<compute_backend> make_gpu_backend(sycl::queue & q);
std::unique_ptr<compute_backend> make_cpu_backend();

inline const char * device_kind_name(device_kind k) {
    return k == device_kind::cpu ? "cpu" : "gpu";
}

} // namespace si
