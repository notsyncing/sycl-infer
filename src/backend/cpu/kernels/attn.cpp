// CPU paged attention (mirror of src/backend/gpu/kernels/attn.cpp): online
// softmax over the block table, every KV storage dtype, int8 scale planes, and
// an optional split-partial path.  With the CPU plan n_splits == 1, so the
// fused branch (out != nullptr) writes the gated head output directly and the
// combine kernel is skipped.
#include "common.h"

namespace si {

void cpu_attn(const float * qbuf, const float * gate, const void * kpool, const void * vpool, float * partials,
              const int32_t * tables, int n_head, int n_head_kv, int head_dim, int n_splits,
              const cpu_step_info * info, float scale, int max_blocks, int n_rows, int n_real, float * out,
              cpu_kv_dtype kv, const void * kscales, const void * vscales) {
    (void)gate;
    const int qstride = n_head * 2 * head_dim;
    const int pstride = 2 + head_dim;
    const int total = n_rows * n_real * n_head;
    par(total, [&](int gid) {
        const int r = gid / (n_real * n_head);
        const int t = (gid / n_head) % n_real;
        const int h = gid % n_head;
        if (r >= info->n_rows || t >= info->n_real || !info->active[r]) {
            return;
        }
        const int pos = info->pos[r] + t;
        const int row = r * info->tpb + t;
        const int n_kv = pos + 1;
        const int kvh = (h * n_head_kv) / n_head;
        const int32_t * table = tables + (size_t)info->slot[r] * max_blocks;
        const float * q = qbuf + (size_t)row * qstride + (size_t)h * 2 * head_dim;
        const float * gb = qbuf + (size_t)row * qstride + (size_t)h * 2 * head_dim + head_dim;
        for (int s = 0; s < n_splits; s++) {
            const int chunk = (n_kv + n_splits - 1) / n_splits;
            const int t0 = s * chunk;
            const int t1 = std::min(t0 + chunk, n_kv);
            float m = -INFINITY, l = 0.0f;
            float acc[512];
            const int hd = head_dim;
            for (int d = 0; d < hd; d++) {
                acc[d] = 0.0f;
            }
            for (int kk = t0; kk < t1; kk++) {
                const int kb = table[kk / kCpuBlk];
                const int ko = kk % kCpuBlk;
                const size_t unit = (size_t)kb * n_head_kv + kvh;
                const size_t off = kv_row_off(unit, ko, head_dim);
                float dot = 0.0f;
                if (kv == cpu_kv_dtype::i8) {
                    const uint16_t * ks = (const uint16_t *)kscales + kv_scale_off(unit, ko, head_dim);
                    for (int i = 0; i < hd / kCpuI8Q; i++) {
                        const float sc = ggml_half_to_float(ks[i]);
                        float sub = 0.0f;
                        for (int d = 0; d < kCpuI8Q; d++) {
                            sub += q[i * kCpuI8Q + d] * (float)((const int8_t *)kpool)[off + i * kCpuI8Q + d];
                        }
                        dot += sub * sc;
                    }
                } else if (kv == cpu_kv_dtype::i4) {
                    const uint16_t * ks = (const uint16_t *)kscales + kv_scale_off(unit, ko, head_dim);
                    for (int i = 0; i < hd / kCpuI8Q; i++) {
                        const float sc = ggml_half_to_float(ks[i]);
                        for (int d = 0; d < kCpuI8Q; d++) {
                            dot += q[i * kCpuI8Q + d] * kv_load(kv, kpool, off + i * kCpuI8Q + d) * sc;
                        }
                    }
                } else {
                    for (int d = 0; d < hd; d++) {
                        dot += q[d] * kv_load(kv, kpool, off + d);
                    }
                }
                dot *= scale;
                const float mnew = std::max(m, dot);
                const float e = std::exp(dot - mnew);
                const float corr = std::exp(m - mnew);
                l = l * corr + e;
                if (kv == cpu_kv_dtype::i8) {
                    const uint16_t * vs = (const uint16_t *)vscales + kv_scale_off(unit, ko, head_dim);
                    for (int i = 0; i < hd / kCpuI8Q; i++) {
                        const float sc = ggml_half_to_float(vs[i]);
                        for (int d = 0; d < kCpuI8Q; d++) {
                            acc[i * kCpuI8Q + d] = acc[i * kCpuI8Q + d] * corr
                                                   + e * (float)((const int8_t *)vpool)[off + i * kCpuI8Q + d] * sc;
                        }
                    }
                } else if (kv == cpu_kv_dtype::i4) {
                    const uint16_t * vs = (const uint16_t *)vscales + kv_scale_off(unit, ko, head_dim);
                    for (int i = 0; i < hd / kCpuI8Q; i++) {
                        const float sc = ggml_half_to_float(vs[i]);
                        for (int d = 0; d < kCpuI8Q; d++) {
                            acc[i * kCpuI8Q + d] =
                                acc[i * kCpuI8Q + d] * corr + e * kv_load(kv, vpool, off + i * kCpuI8Q + d) * sc;
                        }
                    }
                } else {
                    for (int d = 0; d < hd; d++) {
                        acc[d] = acc[d] * corr + e * kv_load(kv, vpool, off + d);
                    }
                }
                m = mnew;
            }
            if (out) {
                float * ob = out + (size_t)row * n_head * head_dim + h * head_dim;
                for (int d = 0; d < hd; d++) {
                    ob[d] = (l > 0.0f ? acc[d] / l : 0.0f) * cpu_sigmoid(gb[d]);
                }
                return; // n_splits == 1 fused contract
            }
            float * part = partials + (((size_t)row * n_head + h) * n_splits + s) * pstride;
            part[0] = m;
            part[1] = l;
            for (int d = 0; d < hd; d++) {
                part[2 + d] = acc[d];
            }
        }
    });
}

void cpu_attn_combine(const float * partials, const float * gate, float * out, const cpu_step_info * info, int n_head,
                      int head_dim, int n_splits, int n_rows, int n_real) {
    const int pstride = 2 + head_dim;
    const int qstride = n_head * 2 * head_dim;
    const int total = n_rows * n_real * n_head;
    par(total, [&](int gid) {
        const int r = gid / (n_real * n_head);
        const int t = (gid / n_head) % n_real;
        const int h = gid % n_head;
        if (r >= info->n_rows || t >= info->n_real || !info->active[r]) {
            return;
        }
        const int row = r * info->tpb + t;
        const float * part = partials + (((size_t)row * n_head + h) * n_splits) * pstride;
        float M = -INFINITY;
        for (int s = 0; s < n_splits; s++) {
            M = std::max(M, part[s * pstride]);
        }
        float sum = 0.0f;
        float * ob = out + (size_t)row * n_head * head_dim + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            ob[d] = 0.0f;
        }
        for (int s = 0; s < n_splits; s++) {
            const float w = std::exp(part[s * pstride] - M);
            sum += part[s * pstride + 1] * w;
            for (int d = 0; d < head_dim; d++) {
                ob[d] += part[s * pstride + 2 + d] * w;
            }
        }
        const float * gb = gate + (size_t)row * qstride + h * 2 * head_dim + head_dim;
        for (int d = 0; d < head_dim; d++) {
            ob[d] = (sum > 0.0f ? ob[d] / sum : 0.0f) * cpu_sigmoid(gb[d]);
        }
    });
}

} // namespace si
