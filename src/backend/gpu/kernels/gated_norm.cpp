#include "kernels.h"
#include "kernel_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

using namespace sycl;
using namespace si::kd;

// ---------------------------------------------------------------------------
void gated_norm_launch(queue & q, const float * attn, const float * z, const float * weight, float * out,
                       const step_info * info, int n_heads, int head_dim, float eps, int n_rows, int n_real, int row0) {
    q.parallel_for(
        nd_range<1>((size_t)n_rows * n_real * n_heads * 32, 32), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int gid = it.get_group(0);
            const int r = gid / (n_real * n_heads);
            const int t = (gid / n_heads) % n_real;
            const int head = gid % n_heads;
            const int lane = it.get_local_id(0);
            const int rr = row0 + r;
            if (rr >= info->n_rows || t >= row_nr(info, rr) || !info->active[rr]) {
                return;
            }
            const int row = rr * info->tpb + t;
            const sub_group sgg = it.get_sub_group();
            const float * a = attn + (size_t)row * n_heads * head_dim + head * head_dim;
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < head_dim / 32; i++) {
                float v = a[lane + 32 * i];
                ss += v * v;
            }
            const float inv = 1.0f / sycl::sqrt(sg_sum(ss, sgg) / head_dim + eps);
#pragma unroll
            for (int i = 0; i < head_dim / 32; i++) {
                const int idx = lane + 32 * i;
                const float zv = z[(size_t)row * n_heads * head_dim + head * head_dim + idx];
                out[(size_t)row * n_heads * head_dim + head * head_dim + idx] = a[idx] * inv * weight[idx] * silu_f(zv);
            }
        });
}

} // namespace si
