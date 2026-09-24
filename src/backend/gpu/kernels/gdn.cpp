#include "kernels.h"
#include "kernel_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

using namespace sycl;
using namespace si::kd;

// ---------------------------------------------------------------------------
// GDN recurrence.  One warp owns C adjacent state rows (cols) of one head: the
// q/k vectors are loaded once per token per warp and shared by all C rows, and
// the C independent reduction chains hide the shuffle latency.  C=1 is the
// original mapping (one state row per warp, q/k re-read per row).
template <int C, int WPW = 8>
static void gdn_kernel(queue & q, const float * conv_out, const float * alpha, const float * dt_bias,
                       const float * ssm_a, const float * beta, float * state, float * attn_out, const step_info * info,
                       int head_dim, int n_k_heads, int n_heads, int conv_dim, float scale, int n_rows, int row0,
                       int tpb_arg, int nreal_arg, pc_snap snap) {
    const int col_groups = head_dim / C;
    const int total_warps = n_heads * col_groups;
    const int n_wg = (total_warps * n_rows + WPW - 1) / WPW;
    const int q_off = 0;
    const int k_off = n_k_heads * head_dim;
    const int v_off = 2 * n_k_heads * head_dim;
    q.parallel_for(
        nd_range<1>((size_t)n_wg * WPW * 32, WPW * 32), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int gid = it.get_group(0) * WPW + it.get_local_id(0) / 32;
            const int r = gid / total_warps;
            const int rr = row0 + r;
            const int wid = gid % total_warps;
            const int lane = it.get_local_id(0) % 32;
            if (rr >= info->n_rows || !info->active[rr]) {
                return;
            }
            const int head = wid / col_groups;
            const int col0 = (wid % col_groups) * C;
            // q/k carry n_k_heads heads, v carries n_heads; every v head is
            // paired with q/k head (head % n_k_heads).  The reference expands
            // the q/k head axis with ggml_repeat_4d, whose tiling is modulo
            // (interleaved), not blocked - the two agree only when
            // n_k_heads == n_heads (the 0.8B reference); Qwen3.8-27B has
            // 16 key heads against 48 value heads.
            const int qk_head = head % n_k_heads;
            const sub_group sgg = it.get_sub_group();
            // hoist the step_info fields: reloading them from host USM inside the
            // token loop costs more than the arithmetic (they alias our writes)
            const int tpb = tpb_arg > 0 ? tpb_arg : info->tpb;
            // nreal_arg overrides the row's token count for the fully-fused
            // chunk-batched prefill (PF_GDN_FUSE=2): one row of `nreal_arg`
            // tokens walks the whole batch and carries the recurrence state,
            // exactly like cpu_gdn.  Without the override the kernel processed
            // only row 0 (kMaxT tokens) and left every later chunk row with a
            // stale state - the mode-2 prefill bug.
            const int n_real = nreal_arg > 0 ? nreal_arg : row_nr(info, rr);
            const int pc_on = info->pc_active;
            const int pbase = info->pos[rr];

            float * st = state + (size_t)info->slot[rr] * n_heads * head_dim * head_dim
                         + ((size_t)head * head_dim + col0) * head_dim;
            float s[C][4];
#pragma unroll
            for (int c = 0; c < C; c++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    s[c][j] = st[(size_t)c * head_dim + lane + 32 * j];
                }
            }

            const float A = ssm_a[head];
            const float * cor = conv_out + (size_t)rr * tpb * conv_dim;
            const float * alr = alpha + (size_t)rr * tpb * n_heads;
            const float * ber = beta + (size_t)rr * tpb * n_heads;

            for (int t = 0; t < n_real; t++) {
                const float * qv = cor + (size_t)t * conv_dim + q_off + qk_head * head_dim;
                const float * kv = cor + (size_t)t * conv_dim + k_off + qk_head * head_dim;
                const float bt = sigmoid_f(ber[t * n_heads + head]);
                const float sp = sycl::log(1.0f + sycl::exp(alr[t * n_heads + head] + dt_bias[head]));
                const float g = sycl::exp(A * sp);

                float vval[C], dots[C];
#pragma unroll
                for (int c = 0; c < C; c++) {
                    vval[c] = cor[(size_t)t * conv_dim + v_off + head * head_dim + col0 + c];
                    float a = 0.f;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        a += s[c][j] * kv[lane + 32 * j];
                    }
                    dots[c] = a;
                }
                float del[C];
#pragma unroll
                for (int c = 0; c < C; c++) {
                    del[c] = (vval[c] - g * sg_sum(dots[c], sgg)) * bt;
                }
                float at[C];
#pragma unroll
                for (int c = 0; c < C; c++) {
                    float a = 0.f;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        s[c][j] = g * s[c][j] + kv[lane + 32 * j] * del[c];
                        a += s[c][j] * qv[lane + 32 * j];
                    }
                    at[c] = sg_sum(a, sgg) * scale;
                }
                if (lane == 0) {
#pragma unroll
                    for (int c = 0; c < C; c++) {
                        attn_out[((size_t)rr * tpb + t) * n_heads * head_dim + head * head_dim + col0 + c] = at[c];
                    }
                }
                // prefix cache: this token completes a 32-token block -> snapshot
                if (pc_on && (pbase + t + 1) % kBlockSize == 0) {
                    const int stt = info->pc_row_slot[(pbase + t + 1) / kBlockSize];
                    if (stt >= 0) {
                        float * dst = snap.base + (size_t)stt * snap.stride + snap.layer_off
                                      + ((size_t)head * head_dim + col0) * head_dim;
#pragma unroll
                        for (int c = 0; c < C; c++) {
#pragma unroll
                            for (int j = 0; j < 4; j++) {
                                dst[(size_t)c * head_dim + lane + 32 * j] = s[c][j];
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < C; c++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    st[(size_t)c * head_dim + lane + 32 * j] = s[c][j];
                }
            }
        });
}

// float4 variant: a lane owns 4 *contiguous* state columns, so q/k come in one
// 16-byte load per lane (4x fewer load instructions) and the state lives in one
// float4 register.
template <int C, int WPW = 8>
static void gdn_f4_kernel(queue & q, const float * conv_out, const float * alpha, const float * dt_bias,
                          const float * ssm_a, const float * beta, float * state, float * attn_out,
                          const step_info * info, int head_dim, int n_k_heads, int n_heads, int conv_dim, float scale,
                          int n_rows, int row0, int tpb_arg, int nreal_arg, pc_snap snap) {
    const int col_groups = head_dim / C;
    const int total_warps = n_heads * col_groups;
    const int n_wg = (total_warps * n_rows + WPW - 1) / WPW;
    const int q_off = 0;
    const int k_off = n_k_heads * head_dim;
    const int v_off = 2 * n_k_heads * head_dim;
    q.parallel_for(
        nd_range<1>((size_t)n_wg * WPW * 32, WPW * 32), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int gid = it.get_group(0) * WPW + it.get_local_id(0) / 32;
            const int r = gid / total_warps;
            const int rr = row0 + r;
            const int wid = gid % total_warps;
            const int lane = it.get_local_id(0) % 32;
            if (rr >= info->n_rows || !info->active[rr]) {
                return;
            }
            const int head = wid / col_groups;
            const int col0 = (wid % col_groups) * C;
            // modulo (interleaved) q/k head pairing, as in the scalar kernel
            const int qk_head = head % n_k_heads;
            const sub_group sgg = it.get_sub_group();
            const int j0 = 4 * lane;
            const int tpb = tpb_arg > 0 ? tpb_arg : info->tpb;
            // see the scalar kernel: nreal_arg is the fused mode-2 batch length
            const int n_real = nreal_arg > 0 ? nreal_arg : row_nr(info, rr);
            const int pc_on = info->pc_active;
            const int pbase = info->pos[rr];

            float * st = state + (size_t)info->slot[rr] * n_heads * head_dim * head_dim
                         + ((size_t)head * head_dim + col0) * head_dim;
            float4 s[C];
#pragma unroll
            for (int c = 0; c < C; c++) {
                s[c] = *(const float4 *)(st + (size_t)c * head_dim + j0);
            }

            const float A = ssm_a[head];
            const float * cor = conv_out + (size_t)rr * tpb * conv_dim;
            const float * alr = alpha + (size_t)rr * tpb * n_heads;
            const float * ber = beta + (size_t)rr * tpb * n_heads;

            for (int t = 0; t < n_real; t++) {
                const float * qv = cor + (size_t)t * conv_dim + q_off + qk_head * head_dim;
                const float * kv = cor + (size_t)t * conv_dim + k_off + qk_head * head_dim;
                const float4 k4 = *(const float4 *)(kv + j0);
                const float4 q4 = *(const float4 *)(qv + j0);
                const float bt = sigmoid_f(ber[t * n_heads + head]);
                const float sp = sycl::log(1.0f + sycl::exp(alr[t * n_heads + head] + dt_bias[head]));
                const float g = sycl::exp(A * sp);
                float del[C];
#pragma unroll
                for (int c = 0; c < C; c++) {
                    const float vval = cor[(size_t)t * conv_dim + v_off + head * head_dim + col0 + c];
                    const float kvv = sg_sum(dot4(s[c], k4), sgg);
                    del[c] = (vval - g * kvv) * bt;
                }
                float at[C];
#pragma unroll
                for (int c = 0; c < C; c++) {
                    s[c] = fma4(k4, del[c], mul4(s[c], g));
                    at[c] = sg_sum(dot4(s[c], q4), sgg) * scale;
                }
                if (lane == 0) {
#pragma unroll
                    for (int c = 0; c < C; c++) {
                        attn_out[((size_t)rr * tpb + t) * n_heads * head_dim + head * head_dim + col0 + c] = at[c];
                    }
                }
                // prefix cache: this token completes a 32-token block -> snapshot
                if (pc_on && (pbase + t + 1) % kBlockSize == 0) {
                    const int stt = info->pc_row_slot[(pbase + t + 1) / kBlockSize];
                    if (stt >= 0) {
                        float * dst = snap.base + (size_t)stt * snap.stride + snap.layer_off
                                      + ((size_t)head * head_dim + col0) * head_dim;
#pragma unroll
                        for (int c = 0; c < C; c++) {
                            *(float4 *)(dst + (size_t)c * head_dim + j0) = s[c];
                        }
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < C; c++) {
                *(float4 *)(st + (size_t)c * head_dim + j0) = s[c];
            }
        });
}

void gdn_launch(queue & q, const float * conv_out, const float * alpha, const float * dt_bias, const float * ssm_a,
                const float * beta, float * state, float * attn_out, const step_info * info, int head_dim,
                int n_k_heads, int n_heads, int conv_dim, float scale, int n_slots, int n_rows, int row0,
                int tpb_arg, int nreal_arg, pc_snap snap) {
    (void)n_slots;
    // PF_GDN_COLS=2/4/8: state rows per warp (default 4; 1 = original mapping)
    static const int cols = [] {
        const char * e = getenv("PF_GDN_COLS");
        const int c = e ? atoi(e) : 2;
        return c == 1 || c == 2 || c == 4 || c == 8 ? c : 2;
    }();
    // PF_GDN_WG=4/8: warps per workgroup (occupancy experiment)
    static const int wpw = [] {
        const char * e = getenv("PF_GDN_WG");
        const int c = e ? atoi(e) : 8;
        return c == 1 ? 1 : (c == 2 ? 2 : (c == 4 ? 4 : 8));
    }();
    // decode has one token per row: halving the warp count costs more than the
    // shared q/k loads save, so only batch columns when the row has real work
    const int n_real = nreal_arg > 0 ? nreal_arg : info->n_real;
    const int c = (n_real >= 8 && head_dim % cols == 0) ? cols : 1;
    // PF_GDN_PD=0: classic form (at from the updated state)

    // float4 (contiguous 4 columns per lane) variant: one 16-byte load per
    // lane for q/k instead of four 4-byte loads (default; PF_GDN_VEC=0 reverts)
    static const bool f4 = [] {
        const char * e = getenv("PF_GDN_VEC");
        return !(e && atoi(e) == 0);
    }();
    if (f4 && head_dim % 4 == 0) {
        if (wpw == 2) {
            switch (c) {
            case 1:
                gdn_f4_kernel<1, 2>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                    conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
                return;
            default:
                gdn_f4_kernel<2, 2>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                    conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
                return;
            }
        }
        switch (c) {
        case 1:
            gdn_f4_kernel<1, 4>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        case 2:
            gdn_f4_kernel<2, 4>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        default:
            gdn_f4_kernel<4, 4>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        }
    }
    if (wpw == 1) {
        switch (c) {
        case 1:
            gdn_kernel<1, 1>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                             conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            break;
        default:
            gdn_kernel<2, 1>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                             conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            break;
        }
        return;
    }
    if (wpw == 2) {
        switch (c) {
        case 1:
            gdn_kernel<1, 2>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                             conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            break;
        default:
            gdn_kernel<2, 2>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                             conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            break;
        }
        return;
    }
    if (wpw == 4) {
        switch (c) {
        case 2:
            gdn_kernel<2, 4>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                             conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            break;
        case 4:
            gdn_kernel<4, 4>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                             conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            break;
        default:
            gdn_kernel<1, 4>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                             conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            break;
        }
        return;
    }
    switch (c) {
    case 1:
        gdn_kernel<1>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads, conv_dim,
                      scale, n_rows, row0, tpb_arg, nreal_arg, snap);
        break;
    case 4:
        gdn_kernel<4>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads, conv_dim,
                      scale, n_rows, row0, tpb_arg, nreal_arg, snap);
        break;
    case 8:
        gdn_kernel<8>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads, conv_dim,
                      scale, n_rows, row0, tpb_arg, nreal_arg, snap);
        break;
    default:
        gdn_kernel<2>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads, conv_dim,
                      scale, n_rows, row0, tpb_arg, nreal_arg, snap);
        break;
    }
}

} // namespace si
