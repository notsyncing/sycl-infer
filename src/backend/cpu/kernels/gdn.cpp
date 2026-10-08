// CPU GDN recurrence (mirror of src/backend/gpu/kernels/gdn.cpp, C=1 mapping):
// per (row, head, state column) sequential scan over the chunk's tokens, with
// the per-32-token prefix-cache state snapshot.
#include "common.h"

#include <cstdio>

namespace si {

void cpu_gdn(const float * conv_out, const float * alpha, const float * dt_bias, const float * ssm_a, const float * beta,
             float * state, float * attn_out, const cpu_step_info * info, int head_dim, int n_k_heads, int n_heads,
             int conv_dim, float scale, int n_slots, int n_rows, int row0, int tpb_arg, int nreal_arg, cpu_pc_snap snap) {
    (void)n_slots;
    const int q_off = 0;
    const int k_off = n_k_heads * head_dim;
    const int v_off = 2 * n_k_heads * head_dim;
    const int tpb = tpb_arg > 0 ? tpb_arg : info->tpb;
    const int n_real = nreal_arg > 0 ? nreal_arg : info->n_real;
    const int pc_on = info->pc_active;
    // Every (row, head) recurrence chain is independent: it owns its slice of
    // `state` (slot[rr], head), writes disjoint attn_out rows/tokens, and its
    // prefix-cache snapshot (same slot/head) — so the head axis is dispatched
    // across the worker pool.  The per-32-token block snapshot lands once we
    // have finished all columns of the head (no cross-task aliasing).
    par(n_rows * n_heads, [&](int id) {
        const int r = id / n_heads;
        const int head = id - r * n_heads;
        const int rr = row0 + r;
        if (rr >= info->n_rows || !info->active[rr]) {
            return;
        }
        const int pbase = info->pos[rr];
        const float A = ssm_a[head];
        // v head -> q/k head is modulo (interleaved): the reference expands the
        // q/k head axis with ggml_repeat_4d, which tiles by modulo.  Blocked
        // (head * n_k_heads / n_heads) coincides only when the model has as
        // many key heads as value heads (0.8B), not for the 27B (16 vs 48).
        const int qk_head = head % n_k_heads;
        float * st = state + (size_t)info->slot[rr] * n_heads * head_dim * head_dim +
                     (size_t)head * head_dim * head_dim;
        // MTP spec verify: run the recurrence on a copy so the batch leaves the
        // live state untouched (the commit replays only the accepted tokens).
        std::vector<float> st_local;
        if (info->mtp_dry) {
            st_local.assign((size_t)head_dim * head_dim, 0.0f);
            std::memcpy(st_local.data(), st, (size_t)head_dim * head_dim * 4);
            st = st_local.data();
        }
        const float * cor = conv_out + (size_t)rr * tpb * conv_dim;
        const float * alr = alpha + (size_t)rr * tpb * n_heads;
        const float * ber = beta + (size_t)rr * tpb * n_heads;
        for (int t = 0; t < n_real; t++) {
            const float * qv = cor + (size_t)t * conv_dim + q_off + (size_t)qk_head * head_dim;
            const float * kv = cor + (size_t)t * conv_dim + k_off + (size_t)qk_head * head_dim;
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
            if (pc_on && (info->mtp_dt || (pbase + t + 1) % kCpuBlk == 0)) {
                const int stt = info->pc_row_slot[info->mtp_dt ? t : (pbase + t + 1) / kCpuBlk];
                if (stt >= 0) {
                    const int64_t at = (int64_t)stt * snap.stride + snap.layer_off
                                       + (int64_t)head * head_dim * head_dim;
                    // The checkpoint pool is sized from pc_state_floats, which is
                    // per *checkpoint slot*; a slot index that runs past the pool,
                    // or a layer offset that runs past the slot, writes into
                    // whatever follows the pool with no diagnostic at all.  Both
                    // are one comparison against a bound the caller has to carry,
                    // because the kernel cannot see the pool's size.
                    if (cpu_pc_guard && snap.cap_floats > 0
                        && at + (int64_t)head_dim * head_dim > snap.cap_floats) {
                        fprintf(stderr,
                                "[pcguard] GDN snapshot past pool: stt=%d head=%d at=%lld +%lld > cap=%lld "
                                "(stride=%lld layer_off=%lld, gdn_per=%d)\n",
                                stt, head, (long long)at, (long long)(head_dim * head_dim), (long long)snap.cap_floats,
                                (long long)snap.stride, (long long)snap.layer_off, snap.gdn_per);
                    }
                    float * dst = snap.base + at;
                    std::memcpy(dst, st, (size_t)head_dim * head_dim * 4);
                }
            }
        }
    });
}

} // namespace si
