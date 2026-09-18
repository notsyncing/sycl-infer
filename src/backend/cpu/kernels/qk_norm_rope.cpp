// CPU q/k RMSNorm + RoPE + paged KV write (mirror of
// src/backend/gpu/kernels/qk_norm_rope.cpp).  Handles every KV storage dtype
// (f32/bf16/f16/i8); for i8 it computes the per-32-dim fp16 scale planes.
// The M-RoPE branch uses the same pair->section mapping as the device kernel.
#include "common.h"

namespace si {

void cpu_qk_norm_rope(float * qbuf, float * kbuf, float * vbuf, const float * q_norm, const float * k_norm, void * kpool,
                      void * vpool, const int32_t * tables, const cpu_step_info * info, int n_head, int n_head_kv,
                      int head_dim, int n_rot, float rope_base, float eps, int max_blocks, int n_rows, int n_real,
                      cpu_kv_dtype kv, const void * kscales, const void * vscales) {
    (void)max_blocks;
    const int qstride = n_head * 2 * head_dim;
    const int kvstride = n_head_kv * head_dim;
    const int total = n_rows * n_real;
    par(total, [&](int idx) {
        const int r = idx / n_real;
        const int t = idx % n_real;
        if (r >= info->n_rows || !info->active[r]) {
            return;
        }
        const int pos = info->pos[r] + t;
        const int row = r * info->tpb + t;
        const int32_t * table = tables + (size_t)info->slot[r] * max_blocks;
        for (int h = 0; h < n_head; h++) {
            float * qh = qbuf + (size_t)row * qstride + (size_t)h * 2 * head_dim;
            rmsnorm_inplace(qh, q_norm, head_dim, eps);
            if (info->mrope_on) {
                const int ridx = r * kCpuT + t;
                const int n = kCpuB * kCpuT;
                if (n_rot >= 2) {
                    const int half = n_rot / 2;
                    for (int i = 0; i < half; i++) {
                        const int sec = mrope_section(info, i);
                        const int rp = info->mrope[sec * n + ridx];
                        const float ang = (float)rp * std::exp2(-2.0f * i / n_rot * std::log2(rope_base));
                        const float c = std::cos(ang), s = std::sin(ang);
                        const float x0 = qh[i], x1 = qh[i + half];
                        qh[i] = x0 * c - x1 * s;
                        qh[i + half] = x0 * s + x1 * c;
                    }
                }
            } else {
                rope_apply(qh, n_rot, rope_base, (float)pos);
            }
        }
        for (int h = 0; h < n_head_kv; h++) {
            float * khp = kbuf + (size_t)row * kvstride + (size_t)h * head_dim;
            rmsnorm_inplace(khp, k_norm, head_dim, eps);
            if (info->mrope_on) {
                const int ridx = r * kCpuT + t;
                const int n = kCpuB * kCpuT;
                if (n_rot >= 2) {
                    const int half = n_rot / 2;
                    for (int i = 0; i < half; i++) {
                        const int sec = mrope_section(info, i);
                        const int rp = info->mrope[sec * n + ridx];
                        const float ang = (float)rp * std::exp2(-2.0f * i / n_rot * std::log2(rope_base));
                        const float c = std::cos(ang), s = std::sin(ang);
                        const float x0 = khp[i], x1 = khp[i + half];
                        khp[i] = x0 * c - x1 * s;
                        khp[i + half] = x0 * s + x1 * c;
                    }
                }
            } else {
                rope_apply(khp, n_rot, rope_base, (float)pos);
            }
            const int kb = table[pos / kCpuBlk];
            const int ko = pos % kCpuBlk;
            const size_t unit = (size_t)kb * n_head_kv + h;
            if (kv == cpu_kv_dtype::i8) {
                int8_t * krow = (int8_t *)kpool + kv_row_off(unit, ko, head_dim);
                uint16_t * ksc = (uint16_t *)kscales + kv_scale_off(unit, ko, head_dim);
                const int nq = head_dim / kCpuI8Q;
                for (int i = 0; i < nq; i++) {
                    float mx = 0.0f;
                    for (int d = 0; d < kCpuI8Q; d++) {
                        mx = std::max(mx, std::fabs(khp[i * kCpuI8Q + d]));
                    }
                    const float sc = mx > 0.0f ? mx / 127.0f : 1.0f;
                    for (int d = 0; d < kCpuI8Q; d++) {
                        krow[i * kCpuI8Q + d] =
                            (int8_t)std::max(-127.0f, std::min(127.0f, std::round(khp[i * kCpuI8Q + d] / sc)));
                    }
                    ksc[i] = ggml_float_to_half(sc);
                }
            } else {
                for (int d = 0; d < head_dim; d++) {
                    kv_store(kv, kpool, kv_row_off(unit, ko, head_dim) + d, khp[d]);
                }
            }
        }
        for (int h = 0; h < n_head_kv; h++) {
            const float * vhp = vbuf + (size_t)row * kvstride + (size_t)h * head_dim;
            const int vb = table[pos / kCpuBlk];
            const int ko = pos % kCpuBlk;
            const size_t unit = (size_t)vb * n_head_kv + h;
            if (kv == cpu_kv_dtype::i8) {
                int8_t * vrow = (int8_t *)vpool + kv_row_off(unit, ko, head_dim);
                uint16_t * vsc = (uint16_t *)vscales + kv_scale_off(unit, ko, head_dim);
                const int nq = head_dim / kCpuI8Q;
                for (int i = 0; i < nq; i++) {
                    float mx = 0.0f;
                    for (int d = 0; d < kCpuI8Q; d++) {
                        mx = std::max(mx, std::fabs(vhp[i * kCpuI8Q + d]));
                    }
                    const float sc = mx > 0.0f ? mx / 127.0f : 1.0f;
                    for (int d = 0; d < kCpuI8Q; d++) {
                        vrow[i * kCpuI8Q + d] =
                            (int8_t)std::max(-127.0f, std::min(127.0f, std::round(vhp[i * kCpuI8Q + d] / sc)));
                    }
                    vsc[i] = ggml_float_to_half(sc);
                }
            } else {
                for (int d = 0; d < head_dim; d++) {
                    kv_store(kv, vpool, kv_row_off(unit, ko, head_dim) + d, vhp[d]);
                }
            }
        }
    });
}

} // namespace si
