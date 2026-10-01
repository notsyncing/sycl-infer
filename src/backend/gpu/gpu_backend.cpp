// GPU compute backend: forwards every compute_backend call to the SYCL kernel
// library in src/backend/gpu/kernels/ (the same calls the engine used before
// the backend abstraction existed).  All allocations and command-graph handling
// stay in the engine; this layer is intentionally stateless beyond the queue.
#include "backend.h"

#include <memory>

namespace si {

namespace {

struct gpu_backend : compute_backend {
    sycl::queue & q;
    explicit gpu_backend(sycl::queue & qq) : q(qq) {}
    const char * name() const override {
        return "gpu";
    }
    bool is_cpu() const override {
        return false;
    }
    device_kind kind() const override {
        return device_kind::gpu;
    }

    void rmsnorm(const float * x, const float * w, float * out, int n_rows, int n, float eps) override {
        rmsnorm_launch(q, x, w, out, n_rows, n, eps);
    }
    void copy_row(const float * src, float * dst, const step_info * info, int n, int row) override {
        copy_row_launch(q, src, dst, info, n, row);
    }
    void embed(const void * table, uint32_t type, const step_info * info, float * out, int n_embd,
               size_t row_bytes) override {
        embed_launch(q, table, type, info, out, n_embd, row_bytes);
    }
    void mtp_capture(const float * src, float * dst, int n_rows, int n) override {
        mtp_capture_launch(q, src, dst, n_rows, n);
    }
    void mtp_argmax(const float * logits, int n, int32_t * out_idx, float * out_val, int M) override {
        mtp_argmax_launch(q, logits, n, out_idx, out_val, M);
    }
    void mtp_concat(const void * table, uint32_t type, size_t row_bytes, const float * enorm, const float * hnorm,
                    const float * h, const float * h_prev, const step_info * info, float * out, int n_embd, float eps,
                    const int32_t * tok_dev) override {
        mtp_concat_launch(q, table, type, row_bytes, enorm, hnorm, h, h_prev, info, out, n_embd, eps, tok_dev);
    }
    void mtp_cand(const float * logits, int n, const int32_t * am_idx, const float * am_val, float margin, int32_t * ids,
                  int cap) override {
        mtp_cand_launch(q, logits, n, am_idx, am_val, margin, ids, cap);
    }
    void mtp_gather(const int32_t * ids, int cap, const int8_t * w8, const uint16_t * wsc, const int8_t * xq,
                    const uint16_t * asa, const float * xs, int K, int N, float * vals) override {
        mtp_gather_launch(q, ids, cap, w8, wsc, xq, asa, xs, K, N, vals);
    }
    void mtp_gather_argmax(const float * vals, const int32_t * ids, int cap, int32_t * out_id, float * out_val) override {
        mtp_gather_argmax_launch(q, vals, ids, cap, out_id, out_val);
    }
    void gemv_group(uint32_t type, const gemv_seg * segs, int n_segs, int total_rows, int TB, int nsb,
                    int n_tok_blocks) override {
        gemv_group_launch(q, type, segs, n_segs, total_rows, TB, nsb, n_tok_blocks);
    }
    void qk_norm_rope(float * qbuf, float * kbuf, float * vbuf, const float * q_norm, const float * k_norm,
                      void * kpool, void * vpool, const int32_t * tables, const step_info * info, int n_head,
                      int n_head_kv, int head_dim, int n_rot, float rope_base, float eps, int max_blocks, int n_rows,
                      int n_real, const void * kscales, const void * vscales) override {
        qk_norm_rope_launch(q, qbuf, kbuf, vbuf, q_norm, k_norm, kpool, vpool, tables, info, n_head, n_head_kv,
                            head_dim, n_rot, rope_base, eps, max_blocks, n_rows, n_real, kscales, vscales);
    }
    void attn(const float * qbuf, const float * gate, const void * kpool, const void * vpool, float * partials,
              const int32_t * tables, int n_head, int n_head_kv, int head_dim, int n_splits, const step_info * info,
              float scale, int max_blocks, int n_rows, int n_real, float * out, int group, const void * kscales,
              const void * vscales) override {
        attn_launch(q, qbuf, gate, kpool, vpool, partials, tables, n_head, n_head_kv, head_dim, n_splits, info, scale,
                    max_blocks, n_rows, n_real, out, group, kscales, vscales);
    }
    void attn_combine(const float * partials, const float * gate, float * out, const step_info * info, int n_head,
                      int head_dim, int n_splits, int n_rows, int n_real) override {
        attn_combine_launch(q, partials, gate, out, info, n_head, head_dim, n_splits, n_rows, n_real);
    }
    void conv_l2(const float * qkv_raw, float * conv_state, const float * conv_w, float * conv_out,
                 const step_info * info, int conv_dim, int kernel_size, int head_k_dim, int n_k_heads, float eps,
                 int n_rows, int n_real, int row0, int tpb_arg, bool cross_row) override {
        conv_l2_launch(q, qkv_raw, conv_state, conv_w, conv_out, info, conv_dim, kernel_size, head_k_dim, n_k_heads,
                       eps, n_rows, n_real, row0, tpb_arg, cross_row);
    }
    void conv_state_update(const float * qkv_raw, float * conv_state, const step_info * info, int conv_dim,
                           int kernel_size, int n_rows, int row0, int tpb_arg, int nreal_arg, bool last_row_only,
                           pc_snap snap) override {
        conv_state_update_launch(q, qkv_raw, conv_state, info, conv_dim, kernel_size, n_rows, row0, tpb_arg, nreal_arg,
                                 last_row_only, snap);
    }
    void gdn(const float * conv_out, const float * alpha, const float * dt_bias, const float * ssm_a,
             const float * beta, float * state, float * attn_out, const step_info * info, int head_dim,
             int n_k_heads, int n_heads, int conv_dim, float scale, int n_slots, int n_rows, int row0, int tpb_arg,
             int nreal_arg, pc_snap snap) override {
        gdn_launch(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                   conv_dim, scale, n_slots, n_rows, row0, tpb_arg, nreal_arg, snap);
    }
    void gated_norm(const float * attn, const float * z, const float * weight, float * out, const step_info * info,
                    int n_heads, int head_dim, float eps, int n_rows, int n_real, int row0) override {
        gated_norm_launch(q, attn, z, weight, out, info, n_heads, head_dim, eps, n_rows, n_real, row0);
    }
    void xq(const float * x, const float * up, int x_stride, int up_stride, int8_t * x8, sycl::float2 * xmeta,
            int32_t * xsumq, const step_info * info, int TB, int K) override {
        xq_launch(q, x, up, x_stride, up_stride, x8, xmeta, xsumq, info, TB, K);
    }
    void dp4a_gemv(const w8t & w, const int8_t * x8, const sycl::float2 * xmeta, const int32_t * xsumq, float * out,
                   const float * residual, float alpha) override {
        dp4a_gemv_launch(q, w, x8, xmeta, xsumq, out, residual, alpha, w.N);
    }
    void dp4a_gemm(const w8t & w, const int8_t * x8, const sycl::float2 * xmeta, const int32_t * xsumq, float * out,
                   int out_stride, const float * residual, float alpha, int TB) override {
        dp4a_gemm_launch(q, w, x8, xmeta, xsumq, out, out_stride, residual, alpha, TB);
    }
    // The GPU records the SIn path through w8; s.i8 is CPU-only.  Keep a correct
    // fp32 fallback so a mis-set flag cannot crash.
    void i8_gemv(const gemv_seg & s) override {
        gemv_group_launch(q, s.type, &s, 1, s.n_rows, 1, s.K / 256, 0);
    }
    void i8_gemm(const gemv_seg & s, int TB) override {
        gemv_group_launch(q, s.type, &s, 1, s.n_rows, TB, s.K / 256, 0);
    }
    void i8_row_gemv(const gemv_seg & s, const int8_t * xq, const float * sx) override {
        i8_row_gemv_launch(q, s.wi8, s.wsc, xq, sx, s.out, s.K, s.n_rows, s.residual, s.alpha);
    }
    void i8_row_gemv_multi(const gemv_seg * segs, int n_segs, int total_rows, const int8_t * xq, const float * sx,
                           const int32_t * xsum) override {
        i8_row_gemv_multi_launch(q, segs, n_segs, total_rows, xq, sx, xsum);
    }
    void synchronize() override {
        q.wait();
    }
};

} // namespace

std::unique_ptr<compute_backend> make_gpu_backend(sycl::queue & q) {
    return std::make_unique<gpu_backend>(q);
}

} // namespace si
