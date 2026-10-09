// CPU compute backend: converts the shared engine PODs (`step_info`,
// `gemv_seg`, `pc_snap`) into the SYCL-free mirrors declared in
// src/backend/cpu/cpu_types.h and calls the host kernel library in
// src/backend/cpu/kernels/.  The CPU kernels run synchronously, so the
// conversion buffers only need to live for the duration of the call.
#include "backend.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "cpu_types.h"

namespace si {

namespace {

cpu_step_info to_cpu_info(const step_info * in) {
    cpu_step_info c{};
    c.n_rows = in->n_rows;
    c.n_real = in->n_real;
    c.tpb = in->tpb;
    std::memcpy(c.n_real_row, in->n_real_row, sizeof(c.n_real_row));
    std::memcpy(c.pos, in->pos, sizeof(c.pos));
    std::memcpy(c.slot, in->slot, sizeof(c.slot));
    std::memcpy(c.active, in->active, sizeof(c.active));
    std::memcpy(c.tokens, in->tokens, sizeof(c.tokens));
    c.mtp_dt = in->mtp_dt;
    c.mtp_dry = in->mtp_dry;
    c.pc_active = in->pc_active;
    c.pc_stride = in->pc_stride;
    c.pc_base = in->pc_base;
    std::memcpy(c.pc_row_slot, in->pc_row_slot, sizeof(c.pc_row_slot));
    c.mrope_on = in->mrope_on;
    std::memcpy(c.mrope_sections, in->mrope_sections, sizeof(c.mrope_sections));
    std::memcpy(c.mrope, in->mrope, sizeof(c.mrope));
    c.img_embd = in->img_embd;
    std::memcpy(c.img_row, in->img_row, sizeof(c.img_row));
    return c;
}

cpu_gemv_seg to_cpu_seg(const gemv_seg & s) {
    cpu_gemv_seg c;
    c.w = s.w;
    c.type = s.type;
    c.K = s.K;
    c.n_rows = s.n_rows;
    c.x = s.x;
    c.x_stride = s.x_stride;
    c.act_up = s.act_up;
    c.out = s.out;
    c.out_stride = s.out_stride;
    c.residual = s.residual;
    c.alpha = s.alpha;
    c.meta32 = reinterpret_cast<const float *>(s.meta32);
    c.w8 = s.w8;
    c.x8 = s.x8;
    c.xmeta = reinterpret_cast<const float *>(s.xmeta);
    c.xsumq = s.xsumq;
    c.i8 = s.i8;
    return c;
}

cpu_pc_snap to_cpu_snap(const pc_snap & s) {
    cpu_pc_snap c;
    c.base = s.base;
    c.stride = s.stride;
    c.layer_off = s.layer_off;
    c.gdn_per = s.gdn_per;
    c.conv_per = s.conv_per;
    // The pool's real size has to reach the kernel: it computes its snapshot
    // offset from a slot index and a layer offset that it cannot bound on its own,
    // and an out-of-range one writes into whatever follows the pool.
    c.cap_floats = s.cap_floats;
    return c;
}

cpu_kv_dtype to_cpu_kv() {
    return static_cast<cpu_kv_dtype>((int)kv_k_dtype());
}
cpu_kv_dtype to_cpu_vkv() {
    return static_cast<cpu_kv_dtype>((int)kv_v_dtype());
}

struct cpu_backend : compute_backend {
    // Supplied by the engine: this TU is compiled with -fno-sycl so it cannot hold a
    // sycl::queue, but it still has to be able to honour synchronize().
    std::function<void()> sync_;
    const char * name() const override {
        return "cpu";
    }
    bool is_cpu() const override {
        return true;
    }
    device_kind kind() const override {
        return device_kind::cpu;
    }

    void rmsnorm(const float * x, const float * w, float * out, int n_rows, int n, float eps) override {
        cpu_rmsnorm(x, w, out, n_rows, n, eps);
    }
    void copy_row(const float * src, float * dst, const step_info * info, int n, int row) override {
        const cpu_step_info ci = to_cpu_info(info);
        cpu_copy_row(src, dst, &ci, n, row);
    }
    void embed(const void * table, uint32_t type, const step_info * info, float * out, int n_embd,
               size_t row_bytes) override {
        const cpu_step_info ci = to_cpu_info(info);
        cpu_embed(table, type, &ci, out, n_embd, row_bytes);
    }
    void mtp_capture(const float * src, float * dst, int n_rows, int n) override {
        std::memcpy(dst, src, (size_t)n_rows * (size_t)n * sizeof(float));
    }
    void mtp_argmax(const float * logits, int n, int32_t * out_idx, float * out_val, int M) override {
        for (int r = 0; r < M; r++) {
            const float * row = logits + (size_t)r * n;
            int best = 0;
            for (int i = 1; i < n; i++) {
                if (row[i] > row[best]) {
                    best = i;
                }
            }
            out_idx[r] = best;
            if (out_val) {
                out_val[r] = row[best];
            }
        }
    }
    void mtp_concat(const void * table, uint32_t, size_t, const float *, const float *, const float *, const float *,
                    const step_info *, float *, int, float, const int32_t *) override {
        (void)table;
        throw std::runtime_error("mtp_concat: the MTP draft head is GPU-only");
    }
    void mtp_cand(const float *, int, const int32_t *, const float *, float, int32_t *, int) override {
        throw std::runtime_error("mtp_cand: the MTP draft head is GPU-only");
    }
    void mtp_gather(const int32_t *, int, const int8_t *, const uint16_t *, const int8_t *, const uint16_t *,
                    const float *, int, int, float *) override {
        throw std::runtime_error("mtp_gather: the MTP draft head is GPU-only");
    }
    void mtp_gather_argmax(const float *, const int32_t *, int, int32_t *, float *) override {
        throw std::runtime_error("mtp_gather_argmax: the MTP draft head is GPU-only");
    }
    void gemv_group(uint32_t type, const gemv_seg * segs, int n_segs, int total_rows, int TB, int nsb,
                    int n_tok_blocks) override {
        std::vector<cpu_gemv_seg> cs((size_t)std::max(n_segs, 1));
        for (int i = 0; i < n_segs; i++) {
            cs[i] = to_cpu_seg(segs[i]);
        }
        cpu_gemv_group(type, cs.data(), n_segs, total_rows, TB, nsb, n_tok_blocks, 0);
    }
    void qk_norm_rope(float * qbuf, float * kbuf, float * vbuf, const float * q_norm, const float * k_norm,
                      void * kpool, void * vpool, const int32_t * tables, const step_info * info, int n_head,
                      int n_head_kv, int head_dim, int n_rot, float rope_base, const float * rope_freqs, float rope_mscale,
                      float eps, int max_blocks, int n_rows, int n_real, const void * kscales,
                      const void * vscales) override {
        const cpu_step_info ci = to_cpu_info(info);
        cpu_qk_norm_rope(qbuf, kbuf, vbuf, q_norm, k_norm, kpool, vpool, tables, &ci, n_head, n_head_kv, head_dim,
                         n_rot, rope_base, rope_freqs, rope_mscale, eps, max_blocks, n_rows, n_real, to_cpu_kv(),
                         to_cpu_vkv(), kscales, vscales);
    }
    void attn(const float * qbuf, const float * gate, const void * kpool, const void * vpool, float * partials,
              const int32_t * tables, int n_head, int n_head_kv, int head_dim, int n_splits, const step_info * info,
              float scale, int max_blocks, int n_rows, int n_real, float * out, int group, const void * kscales,
              const void * vscales) override {
        (void)group;
        const cpu_step_info ci = to_cpu_info(info);
        cpu_attn(qbuf, gate, kpool, vpool, partials, tables, n_head, n_head_kv, head_dim, n_splits, &ci, scale,
                 max_blocks, n_rows, n_real, out, to_cpu_kv(), to_cpu_vkv(), kscales, vscales);
    }
    void attn_combine(const float * partials, const float * gate, float * out, const step_info * info, int n_head,
                      int head_dim, int n_splits, int n_rows, int n_real) override {
        const cpu_step_info ci = to_cpu_info(info);
        cpu_attn_combine(partials, gate, out, &ci, n_head, head_dim, n_splits, n_rows, n_real);
    }
    void conv_l2(const float * qkv_raw, float * conv_state, const float * conv_w, float * conv_out,
                 const step_info * info, int conv_dim, int kernel_size, int head_k_dim, int n_k_heads, float eps,
                 int n_rows, int n_real, int row0, int tpb_arg, bool cross_row) override {
        const cpu_step_info ci = to_cpu_info(info);
        cpu_conv_l2(qkv_raw, conv_state, conv_w, conv_out, &ci, conv_dim, kernel_size, head_k_dim, n_k_heads, eps,
                    n_rows, n_real, row0, tpb_arg, cross_row);
    }
    void conv_state_update(const float * qkv_raw, float * conv_state, const step_info * info, int conv_dim,
                           int kernel_size, int n_rows, int row0, int tpb_arg, int nreal_arg, bool last_row_only,
                           pc_snap snap) override {
        const cpu_step_info ci = to_cpu_info(info);
        cpu_conv_state_update(qkv_raw, conv_state, &ci, conv_dim, kernel_size, n_rows, row0, tpb_arg, nreal_arg,
                              last_row_only, to_cpu_snap(snap));
    }
    void gdn(const float * conv_out, const float * alpha, const float * dt_bias, const float * ssm_a,
             const float * beta, float * state, float * attn_out, const step_info * info, int head_dim,
             int n_k_heads, int n_heads, int conv_dim, float scale, int n_slots, int n_rows, int row0, int tpb_arg,
             int nreal_arg, pc_snap snap) override {
        const cpu_step_info ci = to_cpu_info(info);
        cpu_gdn(conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, &ci, head_dim, n_k_heads, n_heads, conv_dim,
                scale, n_slots, n_rows, row0, tpb_arg, nreal_arg, to_cpu_snap(snap));
    }
    void gated_norm(const float * attn, const float * z, const float * weight, float * out, const step_info * info,
                    int n_heads, int head_dim, float eps, int n_rows, int n_real, int row0) override {
        const cpu_step_info ci = to_cpu_info(info);
        cpu_gated_norm(attn, z, weight, out, &ci, n_heads, head_dim, eps, n_rows, n_real, row0);
    }
    void xq(const float * x, const float * up, int x_stride, int up_stride, int8_t * x8, sycl::float2 * xmeta,
            int32_t * xsumq, const step_info * info, int TB, int K) override {
        const cpu_step_info ci = to_cpu_info(info);
        cpu_xq(x, up, x_stride, up_stride, x8, reinterpret_cast<float *>(xmeta), xsumq, &ci, TB, K);
    }
    void dp4a_gemv(const w8t & w, const int8_t * x8, const sycl::float2 * xmeta, const int32_t * xsumq, float * out,
                   const float * residual, float alpha) override {
        cpu_gemv_seg s;
        s.w8 = w;
        s.x8 = x8;
        s.xmeta = reinterpret_cast<const float *>(xmeta);
        s.xsumq = xsumq;
        s.out = out;
        s.residual = residual;
        s.alpha = alpha;
        cpu_dp4a_gemv(s, 0);
    }
    void dp4a_gemm(const w8t & w, const int8_t * x8, const sycl::float2 * xmeta, const int32_t * xsumq, float * out,
                   int out_stride, const float * residual, float alpha, int TB) override {
        cpu_gemv_seg s;
        s.w8 = w;
        s.x8 = x8;
        s.xmeta = reinterpret_cast<const float *>(xmeta);
        s.xsumq = xsumq;
        s.out = out;
        s.out_stride = out_stride;
        s.residual = residual;
        s.alpha = alpha;
        cpu_dp4a_gemm(s, TB, 0);
    }
    void i8_gemv(const gemv_seg & s) override {
        const cpu_gemv_seg cs = to_cpu_seg(s);
        cpu_i8_gemv(cs, 0);
    }
    void i8_gemm(const gemv_seg & s, int TB) override {
        const cpu_gemv_seg cs = to_cpu_seg(s);
        cpu_i8_gemm(cs, TB, 0);
    }
    void i8_row_gemv(const gemv_seg & s, const int8_t * xq, const float * sx) override {
        // CPU partitions never receive oneDNN weights (the XMX path is GPU
        // only); they run the i8 path instead, so this is unreachable.
        (void)s;
        (void)xq;
        (void)sx;
    }
    void i8_row_gemv_multi(const gemv_seg * segs, int n_segs, int total_rows, const int8_t * xq, const float * sx,
                           const int32_t * xsum) override {
        (void)segs;
        (void)n_segs;
        (void)total_rows;
        (void)xq;
        (void)sx;
        (void)xsum;
    }
    // Not an empty override: engine::sync_all() funnels every "wait until the
    // submitted work is done" through here, and two of its callers - the prefix
    // cache's block serialize/deserialize - use it for correctness, not timing,
    // because they hand a temporary host std::vector to an asynchronous
    // queue::memcpy.  Empty here meant the vector could die with the copy still
    // in flight.
    void synchronize() override {
        if (sync_) {
            sync_();
        }
    }
};

} // namespace

std::unique_ptr<compute_backend> make_cpu_backend(std::function<void()> sync) {
    auto b = std::make_unique<cpu_backend>();
    b->sync_ = std::move(sync);
    return b;
}

} // namespace si
