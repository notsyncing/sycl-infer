// CPU conv (+ L2 norm) and conv state update (mirror of
// src/backend/gpu/kernels/conv.cpp).  `cross_row` reads taps from the
// materialized previous chunk row; the state update also writes the
// prefix-cache snapshot at a 32-token block boundary.
#include "common.h"

namespace si {

void cpu_conv_l2(const float * qkv_raw, float * conv_state, const float * conv_w, float * conv_out,
                 const cpu_step_info * info, int conv_dim, int kernel_size, int head_k_dim, int n_k_heads, float eps,
                 int n_rows, int n_real, int row0, int tpb_arg, bool cross_row) {
    const int group_dim = head_k_dim;
    const int n_groups = conv_dim / group_dim;
    const int n_norm_groups = 2 * n_k_heads;
    const int tpb = tpb_arg > 0 ? tpb_arg : info->tpb;
    // The per-(row, token) taps are independent in both call shapes used by the
    // engine: batched prefill runs n_rows=1 with tpb=the whole stream, decode
    // runs n_real=1 over the batch.  Flattening (row, token) lets one dispatch
    // cover either, keeping the per-channel norm sequential inside a token.
    par(n_rows * n_real, [&](int id) {
        const int r = id / n_real;
        const int t = id - r * n_real;
        const int rr = row0 + r;
        if (rr >= info->n_rows || !info->active[rr]) {
            return;
        }
        float * cstate = conv_state + (size_t)info->slot[rr] * 3 * conv_dim;
        const int row = rr * tpb + t;
        for (int grp = 0; grp < n_groups; grp++) {
            const int cb = grp * group_dim;
            for (int lane = 0; lane < group_dim; lane++) {
                const int ch = cb + lane;
                float v = 0.0f;
                for (int i = 0; i < 4; i++) {
                    const float wv = conv_w[(size_t)i + (size_t)kernel_size * ch];
                    const int p = t + i - (kernel_size - 1);
                    const int gp = cross_row ? (rr * tpb + p) : p;
                    const int qi = cross_row ? gp : (rr * tpb + p);
                    if (gp >= 0) {
                        v += wv * qkv_raw[(size_t)qi * conv_dim + ch];
                    } else {
                        v += wv * cstate[(size_t)(gp + kernel_size - 1) * conv_dim + ch];
                    }
                }
                v = cpu_silu(v);
                conv_out[(size_t)row * conv_dim + ch] = v;
            }
            if (grp < n_norm_groups) {
                float ss = 0.0f;
                for (int lane = 0; lane < group_dim; lane++) {
                    const float v = conv_out[(size_t)row * conv_dim + cb + lane];
                    ss += v * v;
                }
                const float inv = 1.0f / std::fmax(std::sqrt(ss), eps);
                for (int lane = 0; lane < group_dim; lane++) {
                    conv_out[(size_t)row * conv_dim + cb + lane] *= inv;
                }
            }
        }
    });
}

void cpu_conv_state_update(const float * qkv_raw, float * conv_state, const cpu_step_info * info, int conv_dim,
                           int kernel_size, int n_rows, int row0, int tpb_arg, int nreal_arg, bool last_row_only,
                           cpu_pc_snap snap) {
    (void)kernel_size;
    const int tpb = tpb_arg > 0 ? tpb_arg : info->tpb;
    for (int r = 0; r < n_rows; r++) {
        const int rr = row0 + r;
        if (rr >= info->n_rows || !info->active[rr]) {
            continue;
        }
        const int n = nreal_arg > 0 ? nreal_arg : info->n_real;
        const int end_tok = info->pos[rr] + n;
        const int cap = (info->pc_active && end_tok % kCpuBlk == 0) ? info->pc_row_slot[end_tok / kCpuBlk] : -1;
        const bool is_last = (rr == info->n_rows - 1);
        if (last_row_only && !is_last && cap < 0) {
            continue;
        }
        float * cstate = conv_state + (size_t)info->slot[rr] * 3 * conv_dim;
        const float * raw = qkv_raw + (size_t)rr * tpb * conv_dim;
        for (int i = 0; i < conv_dim; i++) {
            const float old1 = cstate[(size_t)1 * conv_dim + i];
            const float old2 = cstate[(size_t)2 * conv_dim + i];
            const float v2 = raw[(size_t)(n - 1) * conv_dim + i];
            const float v1 = (n >= 2) ? raw[(size_t)(n - 2) * conv_dim + i] : old2;
            const float v0 = (n >= 3) ? raw[(size_t)(n - 3) * conv_dim + i] : ((n == 2) ? old2 : old1);
            if ((!last_row_only || is_last) && !info->mtp_dry) {
                cstate[(size_t)0 * conv_dim + i] = v0;
                cstate[(size_t)1 * conv_dim + i] = v1;
                cstate[(size_t)2 * conv_dim + i] = v2;
            }
            if (cap >= 0) {
                float * dst = snap.base + (size_t)cap * snap.stride + snap.layer_off + snap.gdn_per;
                dst[(size_t)0 * conv_dim + i] = v0;
                dst[(size_t)1 * conv_dim + i] = v1;
                dst[(size_t)2 * conv_dim + i] = v2;
            }
        }
    }
}

} // namespace si
