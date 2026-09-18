// CPU gated RMSNorm (mirror of src/backend/gpu/kernels/gated_norm.cpp): per
// (row, token, head) normalize the head vector and apply the silu gate.
#include "common.h"

namespace si {

void cpu_gated_norm(const float * attn, const float * z, const float * weight, float * out, const cpu_step_info * info,
                    int n_heads, int head_dim, float eps, int n_rows, int n_real, int row0) {
    const int total = n_rows * n_real * n_heads;
    par(total, [&](int gid) {
        const int r = gid / (n_real * n_heads);
        const int t = (gid / n_heads) % n_real;
        const int head = gid % n_heads;
        const int rr = row0 + r;
        if (rr >= info->n_rows || t >= info->n_real || !info->active[rr]) {
            return;
        }
        const int row = rr * info->tpb + t;
        const float * a = attn + (size_t)row * n_heads * head_dim + (size_t)head * head_dim;
        const float * zv = z + (size_t)row * n_heads * head_dim + (size_t)head * head_dim;
        float * o = out + (size_t)row * n_heads * head_dim + (size_t)head * head_dim;
        const float ss = sumsq_f32(a, head_dim);
        const float inv = 1.0f / std::sqrt(ss / head_dim + eps);
        for (int d = 0; d < head_dim; d++) {
            o[d] = a[d] * inv * weight[d] * cpu_silu(zv[d]);
        }
    });
}

} // namespace si
