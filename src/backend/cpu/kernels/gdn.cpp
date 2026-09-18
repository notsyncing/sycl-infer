// CPU GDN recurrence (mirror of src/backend/gpu/kernels/gdn.cpp, C=1 mapping):
// per (row, head, state column) sequential scan over the chunk's tokens, with
// the per-32-token prefix-cache state snapshot.
#include "common.h"

namespace si {

void cpu_gdn(const float * conv_out, const float * alpha, const float * dt_bias, const float * ssm_a, const float * beta,
             float * state, float * attn_out, const cpu_step_info * info, int head_dim, int n_heads, int conv_dim,
             float scale, int n_slots, int n_rows, int row0, int tpb_arg, int nreal_arg, cpu_pc_snap snap) {
    (void)n_slots;
    const int q_off = 0;
    const int k_off = n_heads * head_dim;
    const int v_off = 2 * n_heads * head_dim;
    const int tpb = tpb_arg > 0 ? tpb_arg : info->tpb;
    const int n_real = nreal_arg > 0 ? nreal_arg : info->n_real;
    const int pc_on = info->pc_active;
    for (int r = 0; r < n_rows; r++) {
        const int rr = row0 + r;
        if (rr >= info->n_rows || !info->active[rr]) {
            continue;
        }
        const int pbase = info->pos[rr];
        for (int head = 0; head < n_heads; head++) {
            const float A = ssm_a[head];
            float * st =
                state + (size_t)info->slot[rr] * n_heads * head_dim * head_dim + (size_t)head * head_dim * head_dim;
            const float * cor = conv_out + (size_t)rr * tpb * conv_dim;
            const float * alr = alpha + (size_t)rr * tpb * n_heads;
            const float * ber = beta + (size_t)rr * tpb * n_heads;
            for (int t = 0; t < n_real; t++) {
                const float * qv = cor + (size_t)t * conv_dim + q_off + (size_t)head * head_dim;
                const float * kv = cor + (size_t)t * conv_dim + k_off + (size_t)head * head_dim;
                const float bt = cpu_sigmoid(ber[t * n_heads + head]);
                const float sp = std::log(1.0f + std::exp(alr[t * n_heads + head] + dt_bias[head]));
                const float g = std::exp(A * sp);
                for (int col = 0; col < head_dim; col++) {
                    float * s = st + (size_t)col * head_dim;
                    const float vval = cor[(size_t)t * conv_dim + v_off + (size_t)head * head_dim + col];
                    float kvv = 0.0f;
                    for (int i = 0; i < head_dim; i++) {
                        kvv += s[i] * kv[i];
                    }
                    const float del = (vval - g * kvv) * bt;
                    float atn = 0.0f;
                    for (int i = 0; i < head_dim; i++) {
                        s[i] = g * s[i] + kv[i] * del;
                        atn += s[i] * qv[i];
                    }
                    attn_out[((size_t)rr * tpb + t) * n_heads * head_dim + (size_t)head * head_dim + col] = atn * scale;
                }
                // prefix cache: this token completes a 32-token block -> snapshot
                if (pc_on && (pbase + t + 1) % kCpuBlk == 0) {
                    const int stt = info->pc_row_slot[(pbase + t + 1) / kCpuBlk];
                    if (stt >= 0) {
                        float * dst =
                            snap.base + (size_t)stt * snap.stride + snap.layer_off + (size_t)head * head_dim * head_dim;
                        std::memcpy(dst, st, (size_t)head_dim * head_dim * 4);
                    }
                }
            }
        }
    }
}

} // namespace si
