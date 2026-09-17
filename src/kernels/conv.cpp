#include "kernels.h"
#include "kernel_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

using namespace sycl;
using namespace si::kd;

// ---------------------------------------------------------------------------
// cross_row: taps before the first token of a row read the *previous row's*
// qkv (the whole batch is materialized), not the slot state - so one call can
// cover a whole chunk-batched prefill instead of one call per chunk row.
void conv_l2_launch(queue & q, const float * qkv_raw, float * conv_state,
                    const float * conv_w, float * conv_out, const step_info * info,
                    int conv_dim, int kernel_size, int head_k_dim, int n_k_heads, float eps,
                    int n_rows, int n_real, int row0, int tpb_arg, bool cross_row) {
    const int group_dim = head_k_dim;
    const int n_groups = conv_dim / group_dim;
    const int n_norm_groups = 2 * n_k_heads; // q and k parts
    const int tpb = tpb_arg > 0 ? tpb_arg : info->tpb;
    q.submit([&](handler & h) {
        local_accessor<float, 1> red(128, h);
        h.parallel_for(nd_range<1>((size_t) n_rows * n_real * n_groups * 128, 128), [=](nd_item<1> it) {
            const int g = it.get_group(0);
            const int r = g / (n_real * n_groups);
            const int t = (g / n_groups) % n_real;
            const int grp = g % n_groups;
            const int lane = it.get_local_id(0);
            const int rr = row0 + r;
            if (rr >= info->n_rows || t >= n_real || !info->active[rr]) return;
            const int row = rr * tpb + t;
            float * cstate = conv_state + (size_t) info->slot[rr] * 3 * conv_dim;
            const int cb = grp * group_dim;
            const int ch = cb + lane;
            float v = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float wv = conv_w[(size_t) i + (size_t) kernel_size * ch];
                // cross_row: taps are absolute token positions in the batch (the
                // previous rows' qkv is materialized); otherwise the row is
                // self-contained and earlier tokens come from the slot state
                const int p = t + i - (kernel_size - 1);
                const int gp = cross_row ? (rr * tpb + p) : p;
                const int qi = cross_row ? gp : (rr * tpb + p);
                if (gp >= 0) v += wv * qkv_raw[(size_t) qi * conv_dim + ch];
                else         v += wv * cstate[(size_t) (gp + kernel_size - 1) * conv_dim + ch];
            }
            v = silu_f(v);
            if (grp < n_norm_groups) {
                red[lane] = v * v;
                it.barrier();
                for (int s = 64; s > 0; s >>= 1) {
                    if (lane < s) red[lane] += red[lane + s];
                    it.barrier();
                }
                v *= 1.0f / sycl::fmax(sycl::sqrt(red[0]), eps);
            }
            conv_out[(size_t) row * conv_dim + cb + lane] = v;
        });
    });
}

// last_row_only: only the final row of the batch updates the persistent slot
// state (used when one call covers all chunk rows).  Block boundaries marked in
// info->pc_row_slot are always processed and written to the checkpoint pool.
void conv_state_update_launch(queue & q, const float * qkv_raw, float * conv_state,
                              const step_info * info, int conv_dim, int kernel_size,
                              int n_rows, int row0, int tpb_arg, int nreal_arg,
                              bool last_row_only, pc_snap snap) {
    (void) kernel_size;
    const int per = conv_dim / 256;
    const int tpb = tpb_arg > 0 ? tpb_arg : info->tpb;
    q.parallel_for(nd_range<1>((size_t) n_rows * per * 256, 256), [=](nd_item<1> it) {
        const int r = it.get_group(0) / per;
        const int i = (it.get_group(0) % per) * 256 + it.get_local_id(0);
        const int rr = row0 + r;
        if (rr >= info->n_rows || !info->active[rr]) return;
        const int n = nreal_arg > 0 ? nreal_arg : info->n_real;
        // prefix cache: this row ends a 32-token block -> checkpoint it
        const int end_tok = info->pos[rr] + n;
        const int cap = (info->pc_active && end_tok % kBlockSize == 0)
                            ? info->pc_row_slot[end_tok / kBlockSize]
                            : -1;
        const bool is_last = (rr == info->n_rows - 1);
        if (last_row_only && !is_last && cap < 0) return;
        float * cstate = conv_state + (size_t) info->slot[rr] * 3 * conv_dim;
        const float * raw = qkv_raw + (size_t) rr * tpb * conv_dim;
        const float old1 = cstate[(size_t) 1 * conv_dim + i];
        const float old2 = cstate[(size_t) 2 * conv_dim + i];
        const float v2 = raw[(size_t) (n - 1) * conv_dim + i];
        const float v1 = (n >= 2) ? raw[(size_t) (n - 2) * conv_dim + i] : old2;
        const float v0 = (n >= 3) ? raw[(size_t) (n - 3) * conv_dim + i]
                                  : ((n == 2) ? old2 : old1);
        if (!last_row_only || is_last) {
            cstate[(size_t) 0 * conv_dim + i] = v0;
            cstate[(size_t) 1 * conv_dim + i] = v1;
            cstate[(size_t) 2 * conv_dim + i] = v2;
        }
        if (cap >= 0) {
            float * dst = snap.base + (size_t) cap * snap.stride + snap.layer_off + snap.gdn_per;
            dst[(size_t) 0 * conv_dim + i] = v0;
            dst[(size_t) 1 * conv_dim + i] = v1;
            dst[(size_t) 2 * conv_dim + i] = v2;
        }
    });
}

} // namespace si
