#include "kernels.h"
#include "device/device_profile.h"
#include "kernel_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "common/env.h"

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
                // MTP spec verify (info->mtp_dt): snapshot after EVERY token so
                // the recurrence can be rolled back to the accepted draft length.
                if (pc_on && (info->mtp_dt || (pbase + t + 1) % kBlockSize == 0)) {
                    const int stt = info->pc_row_slot[info->mtp_dt ? t : (pbase + t + 1) / kBlockSize];
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
            if (!info->mtp_dry) {
#pragma unroll
                for (int c = 0; c < C; c++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        st[(size_t)c * head_dim + lane + 32 * j] = s[c][j];
                    }
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
                // (see the scalar kernel: info->mtp_dt snapshots every token)
                if (pc_on && (info->mtp_dt || (pbase + t + 1) % kBlockSize == 0)) {
                    const int stt = info->pc_row_slot[info->mtp_dt ? t : (pbase + t + 1) / kBlockSize];
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
            if (!info->mtp_dry) {
#pragma unroll
                for (int c = 0; c < C; c++) {
                    *(float4 *)(st + (size_t)c * head_dim + j0) = s[c];
                }
            }
        });
}

// ---------------------------------------------------------------------------
// Work-group-per-head GDN.  Every work-group owns one (row, head): all
// head_dim/C column groups of that head are its warps, so the 1 KiB of q/k a
// token needs is staged in shared local memory once per token instead of being
// re-read by each column group (the per-warp mapping pays that re-read
// head_dim/C times, which is what makes the chunk-batched prefill's serial
// token loop expensive).  The work-group shape also lets the gated norm and
// the conv-window update ride in the same launch: the norm needs all 128
// attn values of the head (one work-group barrier), and the conv state is only
// read by the conv kernel - which has already run - so updating it here has no
// reader race.  Same dot / same reduction order as the per-warp kernels, so
// the emitted stream is unchanged.
//
// F4 mirrors the float4 decode mapping (lanes own four contiguous j); the
// scalar mapping mirrors gdn_kernel's (lanes own lane+32*j).  Which one runs
// is the same n_real rule gdn_launch uses, so the arithmetic per token is
// bit-identical to the kernel it replaces.
template <int C, bool F4>
static void gdn_head_kernel(queue & q, const float * qkv_raw, float * conv_state, int conv_k,
                            const float * conv_out, const float * alpha, const float * dt_bias, const float * ssm_a,
                            const float * beta, float * state, float * attn_out, const float * z,
                            const float * norm_w, float * merged, const step_info * info, int head_dim,
                            int n_k_heads, int n_heads, int conv_dim, float scale, int n_rows, int row0,
                            int tpb_arg, int nreal_arg, pc_snap snap, bool fuse_state, bool fuse_gated,
                            float eps, bool last_row_only) {
    const int col_groups = head_dim / C;
    const int wg = col_groups * 32;
    q.submit([&](handler & h) {
        local_accessor<float, 1> sh(2 * head_dim + 2, h);
        h.parallel_for(nd_range<1>((size_t)n_rows * n_heads * wg, wg),
                       [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int g = it.get_group(0);
            const int r = g / n_heads;
            const int head = g % n_heads;
            const int rr = row0 + r;
            if (rr >= info->n_rows || !info->active[rr]) {
                return;
            }
            const int lid = it.get_local_id(0);
            const int lane = lid % 32;
            const int wid = lid / 32; // column group
            const int col0 = wid * C;
            const int qk_head = head % n_k_heads;
            const sub_group sgg = it.get_sub_group();
            const int tpb = tpb_arg > 0 ? tpb_arg : info->tpb;
            const int n_real = nreal_arg > 0 ? nreal_arg : row_nr(info, rr);
            const int q_off = 0;
            const int k_off = n_k_heads * head_dim;
            const int v_off = 2 * n_k_heads * head_dim;
            const int j0 = 4 * lane;
            float * st = state + (size_t)info->slot[rr] * n_heads * head_dim * head_dim
                         + ((size_t)head * head_dim + col0) * head_dim;
            // The state element a lane owns follows the *same* mapping as the
            // arithmetic: float4 decode lanes own four contiguous j, the
            // scalar (chunk) lanes own lane + 32*j.
            const int jb = F4 ? j0 : lane;
            const int js = F4 ? 1 : 32;
            float s[C][4];
#pragma unroll
            for (int c = 0; c < C; c++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    s[c][j] = st[(size_t)c * head_dim + jb + js * j];
                }
            }
            const float A = ssm_a[head];
            const float dtb = dt_bias[head];
            const float * cor = conv_out + (size_t)rr * tpb * conv_dim;
            const int pc_on = info->pc_active;
            const int pbase = info->pos[rr];
            for (int t = 0; t < n_real; t++) {
                // stage this token's q/k (and alpha/beta) once per work-group
                it.barrier();
#pragma unroll 1
                for (int i = lid; i < 2 * head_dim; i += wg) {
                    const int src = (size_t)(rr * tpb + t) * conv_dim + qk_head * head_dim
                                    + (i < head_dim ? i : (k_off + i - head_dim));
                    sh[i] = conv_out[src];
                }
                if (lid == 2 * head_dim) {
                    sh[2 * head_dim] = alpha[(size_t)(rr * tpb + t) * n_heads + head];
                    sh[2 * head_dim + 1] = beta[(size_t)(rr * tpb + t) * n_heads + head];
                }
                it.barrier();
                const float bt = sigmoid_f(sh[2 * head_dim + 1]);
                const float sp = sycl::log(1.0f + sycl::exp(sh[2 * head_dim] + dtb));
                const float gmul = sycl::exp(A * sp);
                float del[C], at[C];
#pragma unroll
                for (int c = 0; c < C; c++) {
                    float dots = 0.f;
                    if (F4) {
                        float4 k4{sh[head_dim + j0], sh[head_dim + j0 + 1], sh[head_dim + j0 + 2],
                                  sh[head_dim + j0 + 3]};
                        float4 s4{s[c][0], s[c][1], s[c][2], s[c][3]};
                        dots = dot4(s4, k4);
                    } else {
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            dots += s[c][j] * sh[head_dim + lane + 32 * j];
                        }
                    }
                    const float vval = cor[(size_t)t * conv_dim + v_off + head * head_dim + col0 + c];
                    del[c] = (vval - gmul * sg_sum(dots, sgg)) * bt;
                }
#pragma unroll
                for (int c = 0; c < C; c++) {
                    float a = 0.f;
                    if (F4) {
                        float4 k4{sh[head_dim + j0], sh[head_dim + j0 + 1], sh[head_dim + j0 + 2],
                                  sh[head_dim + j0 + 3]};
                        float4 q4{sh[j0], sh[j0 + 1], sh[j0 + 2], sh[j0 + 3]};
                        float4 s4{s[c][0], s[c][1], s[c][2], s[c][3]};
                        s4 = fma4(k4, del[c], mul4(s4, gmul));
                        s[c][0] = s4.x();
                        s[c][1] = s4.y();
                        s[c][2] = s4.z();
                        s[c][3] = s4.w();
                        a = dot4(s4, q4);
                    } else {
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            s[c][j] = gmul * s[c][j] + sh[head_dim + lane + 32 * j] * del[c];
                            a += s[c][j] * sh[lane + 32 * j];
                        }
                    }
                    at[c] = sg_sum(a, sgg) * scale;
                }
                if (lane == 0) {
#pragma unroll
                    for (int c = 0; c < C; c++) {
                        attn_out[((size_t)rr * tpb + t) * n_heads * head_dim + head * head_dim + col0 + c] = at[c];
                    }
                }
                if (pc_on && (info->mtp_dt || (pbase + t + 1) % kBlockSize == 0)) {
                    const int stt = info->pc_row_slot[info->mtp_dt ? t : (pbase + t + 1) / kBlockSize];
                    if (stt >= 0) {
                        float * dst = snap.base + (size_t)stt * snap.stride + snap.layer_off
                                      + ((size_t)head * head_dim + col0) * head_dim;
#pragma unroll
                        for (int c = 0; c < C; c++) {
#pragma unroll
                            for (int j = 0; j < 4; j++) {
                                dst[(size_t)c * head_dim + jb + js * j] = s[c][j];
                            }
                        }
                    }
                }
            }
            // gated norm: one warp per token of this head, exactly the
            // gated_norm_launch body (32 lanes x head_dim/32 values).
            if (fuse_gated) {
                it.barrier();
                for (int t = wid; t < n_real; t += col_groups) {
                    const int row = rr * tpb + t;
                    const float * a = attn_out + (size_t)row * n_heads * head_dim + head * head_dim;
                    float ss = 0.f;
#pragma unroll
                    for (int i = 0; i < head_dim / 32; i++) {
                        const float v = a[lane + 32 * i];
                        ss += v * v;
                    }
                    const float inv = 1.0f / sycl::sqrt(sg_sum(ss, sgg) / head_dim + eps);
#pragma unroll
                    for (int i = 0; i < head_dim / 32; i++) {
                        const int idx = lane + 32 * i;
                        merged[(size_t)row * n_heads * head_dim + head * head_dim + idx] =
                            a[idx] * inv * norm_w[idx] * silu_f(z[(size_t)row * n_heads * head_dim + head * head_dim + idx]);
                    }
                }
            }
            // conv-window update: every channel belongs to exactly one head or
            // q/k group, so the q/k plane is written by the head that owns the
            // qk group (head < n_k_heads) and v by every head.  The conv kernel
            // has already read the old window (it is a previous launch), so the
            // in-place shift has no reader race.
            if (fuse_state) {
                // A whole-batch call (nreal_arg > 0, mode 2) passes n_rows=1
                // over the linear token buffer; it *is* the last row.
                const int n = nreal_arg > 0 ? nreal_arg : row_nr(info, rr);
                const int base = rr * tpb;
                const int a = base + n - 1;
                const int is_last = nreal_arg > 0 ? 1 : (rr == info->n_rows - 1);
                float * cstate = conv_state + (size_t)info->slot[rr] * 3 * conv_dim;
                const int end_tok = info->pos[rr] + n;
                const int cap = (info->pc_active && end_tok % kBlockSize == 0) ? info->pc_row_slot[end_tok / kBlockSize] : -1;
                const bool wr = (!last_row_only || is_last) && !info->mtp_dry;
                float * snapdst = cap >= 0 ? snap.base + (size_t)cap * snap.stride + snap.layer_off + snap.gdn_per
                                           : nullptr;
#pragma unroll 1
                for (int i = lid; i < 3 * head_dim; i += wg) {
                    const int plane = i / head_dim;
                    const int ch_i = i % head_dim;
                    int ch;
                    if (plane == 2) {
                        ch = v_off + head * head_dim + ch_i;
                    } else if (head < n_k_heads) {
                        ch = (plane == 0 ? q_off : k_off) + qk_head * head_dim + ch_i;
                    } else {
                        continue;
                    }
                    const float old1 = cstate[(size_t)1 * conv_dim + ch];
                    const float old2 = cstate[(size_t)2 * conv_dim + ch];
                    const float v2 = qkv_raw[(size_t)a * conv_dim + ch];
                    const int a1 = a - 1;
                    const float v1 = (a1 >= base) ? qkv_raw[(size_t)a1 * conv_dim + ch]
                                                  : ((last_row_only && a1 >= 0) ? qkv_raw[(size_t)a1 * conv_dim + ch] : old2);
                    const int a0 = a - 2;
                    const float v0 = (a0 >= base) ? qkv_raw[(size_t)a0 * conv_dim + ch]
                                                  : ((last_row_only && a0 >= 0) ? qkv_raw[(size_t)a0 * conv_dim + ch] : old1);
                    if (wr) {
                        cstate[(size_t)0 * conv_dim + ch] = v0;
                        cstate[(size_t)1 * conv_dim + ch] = v1;
                        cstate[(size_t)2 * conv_dim + ch] = v2;
                    }
                    if (snapdst) {
                        snapdst[(size_t)0 * conv_dim + ch] = v0;
                        snapdst[(size_t)1 * conv_dim + ch] = v1;
                        snapdst[(size_t)2 * conv_dim + ch] = v2;
                    }
                }
            }
            if (!info->mtp_dry) {
#pragma unroll
                for (int c = 0; c < C; c++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        st[(size_t)c * head_dim + jb + js * j] = s[c][j];
                    }
                }
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Warp-per-group conv.  conv_l2_launch runs one 128-thread work-group per
// (row, token, channel group) and reduces the q/k L2 norm with a 7-round SLM
// barrier tree.  The same tree is exactly a warp reduction when each lane owns
// the four channels (l, l+32, l+64, l+96): the first two tree steps are
// (v[l]^2+v[l+64]^2) + (v[l+32]^2+v[l+96]^2), and the remaining strides
// 16/8/4/2/1 are the sub-group butterfly.  So one warp does a group's 128
// channels with no local memory and no barriers, bit-identically.
//
// update_state folds the conv-window shift into the same lane for a
// single-token row: that lane is the only reader and writer of its channels,
// so the in-place shift needs no cross-work-item ordering (a separate
// conv_state_update launch copies the same three floats).  Chunked rows keep
// the separate launch because their first tokens read the old window while
// the last token writes the new one.
void gdn_conv_launch(queue & q, const float * qkv_raw, float * conv_state, const float * conv_w, float * conv_out,
                     const step_info * info, int conv_dim, int kernel_size, int head_k_dim, int n_k_heads,
                     float eps, int n_rows, int n_real, int row0, int tpb_arg, bool cross_row,
                     bool update_state, pc_snap snap) {
    const int n_groups = conv_dim / head_k_dim;
    const int n_norm_groups = 2 * n_k_heads;
    const int tpb = tpb_arg > 0 ? tpb_arg : info->tpb;
    q.parallel_for(
        nd_range<1>((size_t)n_rows * n_real * n_groups * 32, 32),
        [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int g = it.get_group(0);
            const int r = g / (n_real * n_groups);
            const int t = (g / n_groups) % n_real;
            const int grp = g % n_groups;
            const int lane = it.get_local_id(0);
            const int rr = row0 + r;
            if (rr >= info->n_rows || t >= row_nr(info, rr) || !info->active[rr]) {
                return;
            }
            const int row = rr * tpb + t;
            const sub_group sgg = it.get_sub_group();
            const int cb = grp * head_k_dim;
            float v[4];
            float ss = 0.f;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                // tree layout: lane owns channels lane + 32*k
                const int ch = cb + lane + 32 * k;
                float acc = 0.f;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const float wv = conv_w[(size_t)i + (size_t)kernel_size * ch];
                    const int p = t + i - (kernel_size - 1);
                    const int gp = cross_row ? (rr * tpb + p) : p;
                    const int qi = cross_row ? gp : (rr * tpb + p);
                    if (gp >= 0) {
                        acc += wv * qkv_raw[(size_t)qi * conv_dim + ch];
                    } else {
                        acc += wv * conv_state[(size_t)info->slot[rr] * 3 * conv_dim
                                               + (size_t)(gp + kernel_size - 1) * conv_dim + ch];
                    }
                }
                v[k] = silu_f(acc);
            }
            if (grp < n_norm_groups) {
                // exact 128-tree association: ((v0^2+v2^2) + (v1^2+v3^2))
                const float a = v[0] * v[0] + v[2] * v[2];
                const float b = v[1] * v[1] + v[3] * v[3];
                ss = sg_sum(a + b, sgg);
                const float inv = 1.0f / sycl::fmax(sycl::sqrt(ss), eps);
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    v[k] *= inv;
                }
            }
#pragma unroll
            for (int k = 0; k < 4; k++) {
                conv_out[(size_t)row * conv_dim + cb + lane + 32 * k] = v[k];
            }
            if (update_state && !cross_row && n_real == 1) {
                // the only token of this row: shift the window in place
                const int end_tok = info->pos[rr] + 1;
                const int cap = (info->pc_active && end_tok % kBlockSize == 0) ? info->pc_row_slot[end_tok / kBlockSize] : -1;
                float * cstate = conv_state + (size_t)info->slot[rr] * 3 * conv_dim;
                float * snapdst = cap >= 0 ? snap.base + (size_t)cap * snap.stride + snap.layer_off + snap.gdn_per
                                           : nullptr;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int ch = cb + lane + 32 * k;
                    const float old1 = cstate[(size_t)1 * conv_dim + ch];
                    const float old2 = cstate[(size_t)2 * conv_dim + ch];
                    const float x = qkv_raw[(size_t)row * conv_dim + ch];
                    if (!info->mtp_dry) {
                        cstate[(size_t)0 * conv_dim + ch] = old1;
                        cstate[(size_t)1 * conv_dim + ch] = old2;
                        cstate[(size_t)2 * conv_dim + ch] = x;
                    }
                    if (snapdst) {
                        snapdst[(size_t)0 * conv_dim + ch] = old1;
                        snapdst[(size_t)1 * conv_dim + ch] = old2;
                        snapdst[(size_t)2 * conv_dim + ch] = x;
                    }
                }
            }
        });
}

void gdn_head_launch(queue & q, const float * qkv_raw, float * conv_state, int conv_k, const float * conv_out,
                     const float * alpha, const float * dt_bias, const float * ssm_a, const float * beta,
                     float * state, float * attn_out, const float * z, const float * norm_w, float * merged,
                     const step_info * info, int head_dim, int n_k_heads, int n_heads, int conv_dim, float scale,
                     int n_rows, int row0, int tpb_arg, int nreal_arg, pc_snap snap, bool fuse_state,
                     bool fuse_gated, float eps, bool last_row_only, int cols) {
    const int n_real = nreal_arg > 0 ? nreal_arg : info->n_real;
    const si::dev::profile & dp = si::dev::for_queue(q);
    static const int vec_mode = [] {
        const char * e = si::env::str("PF_GDN_VEC");
        return e ? atoi(e) : -1;
    }();
    // Same float4 rule gdn_launch uses, so the per-token arithmetic is the one
    // the kernel being replaced would have produced.
    const bool f4 = (vec_mode < 0) ? (n_real < dp.shape.gdn_vec_max_rows) : (vec_mode != 0);
    const int cc = cols == 4 ? 4 : 8;
    if (cc == 4) {
        if (f4) {
            gdn_head_kernel<4, true>(q, qkv_raw, conv_state, conv_k, conv_out, alpha, dt_bias, ssm_a, beta, state,
                                     attn_out, z, norm_w, merged, info, head_dim, n_k_heads, n_heads, conv_dim, scale,
                                     n_rows, row0, tpb_arg, nreal_arg, snap, fuse_state, fuse_gated, eps,
                                     last_row_only);
        } else {
            gdn_head_kernel<4, false>(q, qkv_raw, conv_state, conv_k, conv_out, alpha, dt_bias, ssm_a, beta, state,
                                      attn_out, z, norm_w, merged, info, head_dim, n_k_heads, n_heads, conv_dim, scale,
                                      n_rows, row0, tpb_arg, nreal_arg, snap, fuse_state, fuse_gated, eps,
                                      last_row_only);
        }
    } else {
        if (f4) {
            gdn_head_kernel<8, true>(q, qkv_raw, conv_state, conv_k, conv_out, alpha, dt_bias, ssm_a, beta, state,
                                     attn_out, z, norm_w, merged, info, head_dim, n_k_heads, n_heads, conv_dim, scale,
                                     n_rows, row0, tpb_arg, nreal_arg, snap, fuse_state, fuse_gated, eps,
                                     last_row_only);
        } else {
            gdn_head_kernel<8, false>(q, qkv_raw, conv_state, conv_k, conv_out, alpha, dt_bias, ssm_a, beta, state,
                                      attn_out, z, norm_w, merged, info, head_dim, n_k_heads, n_heads, conv_dim, scale,
                                      n_rows, row0, tpb_arg, nreal_arg, snap, fuse_state, fuse_gated, eps,
                                      last_row_only);
        }
    }
}

void gdn_launch(queue & q, const float * conv_out, const float * alpha, const float * dt_bias, const float * ssm_a,
                const float * beta, float * state, float * attn_out, const step_info * info, int head_dim,
                int n_k_heads, int n_heads, int conv_dim, float scale, int n_slots, int n_rows, int row0,
                int tpb_arg, int nreal_arg, pc_snap snap) {
    (void)n_slots;
    const si::dev::profile & dp = si::dev::for_queue(q);
    // PF_GDN_COLS=1/2/4/8: state rows per warp (1 = the original mapping), and
    // PF_GDN_WG: warps per workgroup.  The *env* is the same on every card, so it
    // is read once; the profile default is not - it comes from this queue's own
    // device via `dp`, so a split across cards with different gdn tuning gets the
    // right shape on each.  Baking either into a function-local static would pin
    // the whole process to whichever card resolved first, which is the bug this
    // whole per-device path exists to fix.
    static const int cols_env = [] {
        const char * e = si::env::str("PF_GDN_COLS");
        return e ? atoi(e) : 0;
    }();
    const int cols_src = cols_env ? cols_env : dp.shape.gdn_cols;
    const int cols = (cols_src == 1 || cols_src == 2 || cols_src == 4 || cols_src == 8) ? cols_src : 2;
    static const int wpw_env = [] {
        const char * e = si::env::str("PF_GDN_WG");
        return e ? atoi(e) : 0;
    }();
    const int wpw_src = wpw_env ? wpw_env : dp.shape.gdn_warps_per_wg;
    const int wpw = wpw_src == 1 ? 1 : (wpw_src == 2 ? 2 : (wpw_src == 4 ? 4 : (wpw_src == 16 ? 16 : 8)));
    // decode has one token per row: halving the warp count costs more than the
    // shared q/k loads save, so only batch columns when the row has real work
    const int n_real = nreal_arg > 0 ? nreal_arg : info->n_real;
    // PF_GDN_COLS_MIN: the n_real at which batching columns starts paying.  The
    // default 8 is the prefill's row width; the MTP verify runs n_real = k+1 = 5
    // and was therefore excluded, so the knob exists to price the verify's
    // recurrence against the multi-column path (measured: every PF_GDN_COLS>1
    // variant is 1-1.7 ms slower there, so the gate is right).
    static const int cols_min = [] {
        const char * e = si::env::str("PF_GDN_COLS_MIN");
        return e ? atoi(e) : 8;
    }();
    const int c = (n_real >= cols_min && head_dim % cols == 0) ? cols : 1;

    // float4 (contiguous 4 columns per lane) variant: one 16-byte load per
    // lane for q/k instead of four 4-byte loads; the state lives in one float4
    // register.  It only pays for a *single* token per row: the float4 state
    // slice is 4x the registers, so as soon as a row carries more the occupancy
    // collapses.  Measured on the 27B / 2x A770, the gdn kernel's ms per pass:
    //
    //   n_real     1     2     4     8    12    16    32
    //   float4  4.64  7.66  7.68 10.24 12.38 14.63 22.03
    //   scalar  4.89  5.23  5.23  5.92  6.91  8.26 12.17
    //
    // i.e. float4 wins by 0.25 ms at n_real=1 (the plain decode) and loses by
    // 1.46x from n_real=2, widening to 1.81x at 32 - its marginal is 0.51 ms per
    // extra token against the scalar path's 0.25.  The MTP verify sits at
    // n_real = k+1 = 5, where the scalar kernel is 2.5 ms of a ~95 ms pass
    // faster.  PF_GDN_VEC: unset/-1 = this rule, 0 = always scalar, 1 = always
    // float4 (the old default, kept for A/B).
    static const int vec_mode = [] {
        const char * e = si::env::str("PF_GDN_VEC");
        return e ? atoi(e) : -1;
    }();
    const bool f4 = (vec_mode < 0) ? (n_real < dp.shape.gdn_vec_max_rows) : (vec_mode != 0);
    if (f4 && head_dim % 4 == 0) {
        // Column batching only has C = 1/2/4 instantiations (a wider request
        // runs the 4-column form: the kernel derives its grid from its own C,
        // so this is a valid mapping, just not a wider one).
        const int cc = (c == 1) ? 1 : (c == 2 ? 2 : 4);
        // WPW (warps per work-group) changes only the launch shape, never the
        // arithmetic, so it is a pure occupancy knob.  Measured on the Xe2
        // part: 4 and 8 are equal within noise, 16 costs ~2% (fewer groups to
        // schedule); the profile default is 8.
        switch (wpw * 10 + cc) {
        case 21:
            gdn_f4_kernel<1, 2>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        case 22:
            gdn_f4_kernel<2, 2>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        case 24:
            gdn_f4_kernel<4, 2>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        case 41:
            gdn_f4_kernel<1, 4>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        case 42:
            gdn_f4_kernel<2, 4>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        case 44:
            gdn_f4_kernel<4, 4>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        case 81:
            gdn_f4_kernel<1, 8>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        case 82:
            gdn_f4_kernel<2, 8>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        case 84:
            gdn_f4_kernel<4, 8>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        case 161:
            gdn_f4_kernel<1, 16>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                 conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        case 162:
            gdn_f4_kernel<2, 16>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                 conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        case 164:
            gdn_f4_kernel<4, 16>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                                 conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            return;
        default:
            gdn_f4_kernel<1, 8>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
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
    if (wpw == 16) {
        switch (c) {
        case 1:
            gdn_kernel<1, 16>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                              conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            break;
        case 4:
            gdn_kernel<4, 16>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                              conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            break;
        case 8:
            gdn_kernel<8, 16>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
                              conv_dim, scale, n_rows, row0, tpb_arg, nreal_arg, snap);
            break;
        default:
            gdn_kernel<2, 16>(q, conv_out, alpha, dt_bias, ssm_a, beta, state, attn_out, info, head_dim, n_k_heads, n_heads,
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
