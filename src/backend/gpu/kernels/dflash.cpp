// DFlash / DFlash2 draft-model kernels.
//
// Four shapes the main model does not have, all of them small and all of them on
// the draft's own buffers (see engine_dflash.cpp for the forward that drives
// them):
//
//   df_conv_launch          the DFlash2 grouped dynamic depthwise convolution
//                           out[i,c] = sum_t (base[t,c,side] + d[i,t,g(c)]) x[i-t,c]
//   df_qknorm_rope_store_   Q/K rms-norm + RoPE, and the K/V write into the
//     launch               draft's own ring cache (used by both the block
//                           forward and the target-feature injection)
//   df_attn_launch /        non-causal-in-block attention over [committed ring
//     df_attn_combine      window, whole block], head_dim 128, f16 or f32 ring
//   df_topk_launch          per-row top-k over a [M][n_vocab] logit matrix
//   df_sel_launch           the DFlash2 candidate selector: edge(p->c) =
//                           <A[p] * gate(h_i), B[c]> + unary_i[c] for every
//                           (candidate at i-1, candidate at i) pair
#include "kernels.h"
#include "common/rope.h"
#include "kernel_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include "common/env.h"

namespace si {

using namespace sycl;

// Largest top-K the per-lane SLM lists fit, DERIVED rather than guessed.  Each
// lane keeps a [K] list plus the scratch slot at index WG*K, so the two list
// accessors are (WG*K + 1) elements each, and the two reduction scratch arrays
// are WG elements each:
//
//     8*(WG*K + 1) + 8*WG  <=  64 KB
//
// At WG=256 that is K <= 30 (K=31 needs 65544 bytes, 8 over).  The original guard
// said K <= 32 with WG*K+2 lists and no scratch in the budget: K=32 needs 65552
// bytes and K=31 65544, so both failed to LAUNCH with
// UR_RESULT_ERROR_OUT_OF_RESOURCES - a config error surfacing as a resource
// error, which is exactly the shape of bug that only a test finds.  DFlash2's
// sel_top_k is 16, so the model cannot reach this; it is a guard, not a limit.
constexpr int kTopkWG = 256;
constexpr int kTopkSlmBudget = 64 * 1024;
constexpr int kTopkMaxK = (kTopkSlmBudget - 8 * kTopkWG - 8) / (8 * kTopkWG);
static_assert(kTopkMaxK >= 16, "a model must be able to ask for at least sel_top_k=16");

using namespace si::kd;

static inline float mb_h2f(uint16_t h) {
    sycl::half x;
    __builtin_memcpy((void *)&x, &h, sizeof(x));
    return (float)x;
}

// ---------------------------------------------------------------------------
// Strided row capture: dst[t * dst_row_stride + i] = src[t * n + i].
//
// The DFlash drafter wants the target's hidden states from several layers in one
// interleaved [token][n_tgt_layer * n_embd] buffer, so each layer's capture is a
// copy with a row stride and a column offset (dst = base + slot * n_embd).
// Copy one target layer's hidden state into its slot of the interleaved feature
// buffer.  `dst` already points at the slot's first row and `dst_row_stride` is the
// slot pitch (n_feat), so row t lands at dst + t*dst_row_stride.
void df_capture_launch(queue & q, const float * src, float * dst, int n_rows, int n, int dst_row_stride) {
    if (n_rows <= 0 || n <= 0) {
        return;
    }
    q.parallel_for(nd_range<1>((size_t)n_rows * n, 256), [=](nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        const int t = (int)(i / n);
        dst[(size_t)t * dst_row_stride + (i % n)] = src[i];
    });
}

// ---------------------------------------------------------------------------
// DFlash2 grouped dynamic depthwise convolution.
//
// `base` is the static kernel [n_embd][conv_k][2] (side 0 = the sublayer input,
// side 1 = its output) and `dyn` the per-position delta [n_rows][2*conv_k*n_groups]
// projected from the sublayer's own input, laid out [group][tap][side].  Both the
// static and the dynamic part are per *group* g(c) = c / conv_group, so one
// projection serves conv_group channels.  Taps that fall before the block start
// read the left zero padding and contribute nothing (llama.cpp's n_taps clamp).
void df_conv_launch(queue & q, const float * x, const float * dyn, const float * base, float * out,
                    const float * residual, int n_rows, int width, int base_width, int conv_k, int conv_group,
                    int dyn_proj, int side) {
    // PF_DFLASH_NOCONV=1 turns every conv into a no-op.  WRONG RESULTS, and it is
    // only there to price the convs in isolation: PF_DFLASH_OPTIME attributes 7.6 ms
    // of the draft block forward's 20.5 ms to them (20 convs, ~2.5 MB total) while
    // each one is only 6 rows x 5120 channels x 2 taps, so 0.38 ms apiece is not
    // explained by the traffic.
    if (si::env::flag("PF_DFLASH_NOCONV")) {
        return;
    }
    if (n_rows <= 0 || conv_k <= 0 || width <= 0) {
        return;
    }
    // `width` is the conv's own channel count, which is NOT the draft's n_embd on
    // the attention side: that conv runs on the n_head * head_dim-wide attention
    // output (llama.cpp's build_dflash2_conv reads hidden_size = cur->ne[0], and
    // views only a prefix of the [n_embd][conv_k][2] base tensor).  So:
    //   * the groups - and therefore which part of the projection the conv reads -
    //     follow `width`, while the projection's *row stride* stays dyn_proj
    //   * llama.cpp views the base as a *prefix* of `width * conv_k` elements of
    //     the side slice and reshapes it to [group, width/group, conv_k], so the
    //     base's tap stride is `width` too - not the tensor's own n_embd
    (void)base_width;
    const int n_groups = width / conv_group;
    // One work-item per (row, channel), indexed from the *global* id.  A plain
    // range, not nd_range: the kernel has no barriers and no local memory, so a
    // work-group size buys nothing - and a fixed 256 local size is a LATENT
    // BUG, because SYCL rejects a non-uniform group count and n_rows*width is
    // only a multiple of 256 by luck (the 27B shapes are: 6*5120=30720=120*256
    // and 6*4096=24576=96*256, which is why it never fired in the model).  Any
    // other width died with "Non-uniform work-groups are not supported".
    q.parallel_for(range<1>((size_t)n_rows * width), [=](id<1> it) {
        const size_t gid = it[0];
        const int i = (int)(gid / width);
        const int c = (int)(gid % width);
        const int g = c / conv_group;
        const float * dbase = base + (size_t)width * conv_k * side;
        const float * ddy = dyn + (size_t)i * dyn_proj;
        float acc = 0.f;
        for (int t = 0; t < conv_k && t <= i; t++) {
            // llama.cpp builds the weight as
            //   base_side = view(base, [group, n_groups, conv_k]) at side*nb[2]
            //              -> base_side[c, g, t] = base[side][c*conv_k + t]
            //   coeffs    = reshape(dynamic, [n_groups, conv_k, 2, n_tokens])
            //              -> coeffs[g, t, side, tok] = dyn[(g*conv_k + t)*2 + side]
            // and repeats the coeffs out to the group.  So the CHANNEL is the outer
            // axis of the base tap and `side` is the *inner* axis of the dynamic
            // coefficients - transposing either one mixes up (channel, tap).
            // llama.cpp's build_dflash2_conv composes two tensors whose axes are
            // mirrored, which is the whole trap here:
            //
            //   base:  view_1d(base, W*K, side*nb[2]) reshaped to
            //          [group_size, n_groups, kernel_size]  =>  channel is innermost,
            //          so channel c and tap t sit at  (c%g) + g*((c/g) + n_groups*t)
            //                                              == c + width*t
            //
            //   dynamic: reshape_4d(dyn, n_groups, kernel, 2, tokens) gives that
            //          tensor nb[1] = n_groups and nb[2] = n_groups*kernel, and the
            //          side view is taken at `side*nb[2]`, so (g, t, side) sits at
            //              g + n_groups*t + side*n_groups*kernel
            //          - group innermost, and the side offset is a whole tap-plane.
            //
            // Getting one right and the other transposed costs a whole layer of
            // accuracy and nothing else catches it: both sides still self-consistently
            // reproduce a host recompute written with the same wrong index.
            const float w = dbase[(size_t)c + (size_t)width * t] + ddy[(size_t)g + (size_t)n_groups * t + (size_t)side * n_groups * conv_k];
            acc += w * x[(size_t)(i - t) * width + c];
        }
        // the residual add (llama.cpp's ggml_add after the conv) lives here so
        // the conv's destination can also be the residual's buffer: every work
        // item reads and writes exactly one element
        out[(size_t)i * width + c] = residual ? residual[(size_t)i * width + c] + acc : acc;
    });
}

// out[row][c] = a[row][c] + b[row][c] (the residual add of a sublayer that has
// no convolution of its own - DFlash1 - or a stage test's convenience).
void df_add_launch(queue & q, const float * a, const float * b, float * out, int n_rows, int n) {
    if (n_rows <= 0 || n <= 0) {
        return;
    }
    q.parallel_for(nd_range<1>((size_t)n_rows * n, 256), [=](nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        out[i] = a[i] + b[i];
    });
}

// ---------------------------------------------------------------------------
// Q/K rms-norm + RoPE and the K/V ring write.
//
// The draft's attention is NOT the main model's: head_dim 128, no attention
// gate, and a *ring* cache whose cell for absolute position p is p % ring (the
// draft's sliding window bounds how much of it is ever live).  Both the block
// forward and the feature injection end here - they differ only in which
// activation the K/V GEMM read.
//
// `pos0_dev` holds the absolute position of every row on the device, so the
// caller can record the draft step as a command graph.
namespace {
template <typename KV>
static void df_rope_store_impl(queue & q, float * qbuf, float * kbuf, float * vbuf, const float * q_norm,
                               const float * k_norm, KV * kpool, KV * vpool, const int32_t * pos_dev, int n_rows,
                               int n_head, int n_head_kv, int head_dim, int n_rot, float rope_base, float eps,
                               int ring, bool do_q, int row_stride, bool do_norm) {
    const int NSG = n_head + 2 * n_head_kv; // Q + K + V sub-groups per row
    // One sub-group per work-group, not NSG of them: the draft has 32 query heads,
    // so NSG*32 = 1536 threads per row would exceed the 1024 work-group limit, and
    // every lane of a sub-group needs the whole head anyway (the rms-norm reduction
    // is over the sub-group).
    q.parallel_for(nd_range<1>((size_t)n_rows * NSG * 32, 32), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int gid = (int)it.get_group(0);
        const int r = gid / NSG;
        const int sg = gid % NSG;
        const int tid = (int)it.get_local_id(0);
        const int lane = tid;
        const sub_group sgg = it.get_sub_group();
        const int pos = pos_dev[r];
        auto rope = [&](float * h) {
            if (lane < n_rot / 2) {
                const float ang = rope_theta((float)pos, lane, n_rot, sycl::log2(rope_base));
                const float c = sycl::cos(ang), s = sycl::sin(ang);
                const float x0 = h[lane], x1 = h[lane + n_rot / 2];
                h[lane] = x0 * c - x1 * s;
                h[lane + n_rot / 2] = x0 * s + x1 * c;
            }
        };
        if (sg < n_head) {
            if (!do_q) {
                return;
            }
            float * qh = qbuf + (size_t)r * row_stride + (size_t)sg * head_dim;
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < head_dim / 32; i++) {
                const float v = qh[lane + 32 * i];
                ss += v * v;
            }
            if (do_norm) {
                const float inv = 1.0f / sycl::sqrt(sg_sum(ss, sgg) / head_dim + eps);
#pragma unroll
                for (int i = 0; i < head_dim / 32; i++) {
                    qh[lane + 32 * i] *= inv * q_norm[lane + 32 * i];
                }
            }
            rope(qh);
        } else if (sg < n_head + n_head_kv) {
            const int kh = sg - n_head;
            float * khp = kbuf + (size_t)r * row_stride + (size_t)kh * head_dim;
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < head_dim / 32; i++) {
                const float v = khp[lane + 32 * i];
                ss += v * v;
            }
            if (do_norm) {
                const float inv = 1.0f / sycl::sqrt(sg_sum(ss, sgg) / head_dim + eps);
#pragma unroll
                for (int i = 0; i < head_dim / 32; i++) {
                    khp[lane + 32 * i] *= inv * k_norm[lane + 32 * i];
                }
            }
            rope(khp);
            KV * dst = kpool + ((size_t)(pos % ring) * n_head_kv + kh) * head_dim;
#pragma unroll
            for (int i = 0; i < head_dim / 32; i++) {
                kv_st(dst + lane + 32 * i, khp[lane + 32 * i]);
            }
        } else {
            const int vh = sg - n_head - n_head_kv;
            const float * vhp = vbuf + (size_t)r * row_stride + (size_t)vh * head_dim;
            KV * dst = vpool + ((size_t)(pos % ring) * n_head_kv + vh) * head_dim;
#pragma unroll
            for (int i = 0; i < head_dim / 32; i++) {
                kv_st(dst + lane + 32 * i, vhp[lane + 32 * i]);
            }
        }
    });
}
} // namespace

void df_qknorm_rope_store_launch(queue & q, float * qbuf, float * kbuf, float * vbuf, const float * q_norm,
                                 const float * k_norm, void * kpool, void * vpool, const int32_t * pos_dev, int n_rows,
                                 int n_head, int n_head_kv, int head_dim, int n_rot, float rope_base, float eps,
                                 int ring, int kv_bytes, bool do_q, int row_stride, bool do_norm) {
    if (n_rows <= 0) {
        return;
    }
    if (kv_bytes == 4) {
        df_rope_store_impl<float>(q, qbuf, kbuf, vbuf, q_norm, k_norm, (float *)kpool, (float *)vpool, pos_dev, n_rows,
                                  n_head, n_head_kv, head_dim, n_rot, rope_base, eps, ring, do_q, row_stride,
                                  do_norm);
    } else {
        df_rope_store_impl<sycl::half>(q, qbuf, kbuf, vbuf, q_norm, k_norm, (sycl::half *)kpool, (sycl::half *)vpool,
                                      pos_dev, n_rows, n_head, n_head_kv, head_dim, n_rot, rope_base, eps, ring, do_q,
                                      row_stride, do_norm);
    }
}

// ---------------------------------------------------------------------------
// Draft attention: one warp per (row, head, key split), 4 dims per lane.
//
// The key set is the union of
//   * the committed positions inside the sliding window  [max(0,p-swa+1), pos0-1]
//   * the whole draft block                                 [pos0, pos0+n_blk-1]
// i.e. non-causal *inside* the block (that is the block-diffusion trick: every
// mask position may read every other position of its own block) and sliding over
// the committed past.  Partials use the main model's [row][head][split][2+hd]
// layout so the combine reads like attn_combine_launch.
void df_attn_launch(queue & q, const float * qbuf, const void * kpool, const void * vpool, float * partials,
                    const int32_t * pos_dev, int n_rows, int n_blk, int n_head, int n_head_kv, int head_dim,
                    int ring, int swa, int n_splits, float scale, int kv_bytes, int row_stride) {
    if (n_rows <= 0 || n_splits <= 0) {
        return;
    }
    const int pstride = 2 + head_dim;
    const int ngroup = n_rows * n_head * n_splits;
    // PF_DFLASH_NOCOMMIT=1: drop the committed (injected) keys so every row sees
    // only the block.  Diagnostic: if the mask rows' attention recovers, the
    // injected K/V are what differs from llama.cpp, not the block's own.
    const int nocmt = si::env::str("PF_DFLASH_NOCOMMIT") ? atoi(si::env::str("PF_DFLASH_NOCOMMIT")) : 0;
    const int causblk = si::env::str("PF_DFLASH_CAUSAL") ? atoi(si::env::str("PF_DFLASH_CAUSAL")) : 0;
    q.parallel_for(nd_range<1>((size_t)ngroup * 32, 32), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int g = (int)it.get_group(0);
        const int r = g / (n_head * n_splits);
        const int h = (g / n_splits) % n_head;
        const int s = g % n_splits;
        const int lane = (int)it.get_local_id(0);
        const int pos0 = pos_dev[0];
        const int p = pos0 + r; // this row's absolute position
        // keys = committed positions in the sliding window + the whole block
        // (non-causal inside it).  The committed range starts at 0: the ring holds
        // every position the features were injected for, and only the window
        // (swa) bounds it.
        // The block's own keys stop at this row when PF_DFLASH_CAUSAL=1.  Row 0 is
        // unaffected either way (it sees only itself), which is exactly why an anchor
        // row can match a reference while the mask rows do not.
        const int hi = causblk ? p : pos0 + n_blk - 1;
        // PF_DFLASH_NOCOMMIT=1: drop the committed (injected) keys and let every
        // row see only the block.  Diagnostic for the mask rows' attention being
        // ~1.6x short against llama.cpp: if the mask rows recover, the injected
        // K/V are what differs; if not, it is the block's own attention.
        const int nocomit = nocmt;
        int lo = (swa > 0) ? std::max(0, p - swa + 1) : 0;
        if (nocomit && lo < pos0) {
            lo = pos0;
        }
        const int nk = hi - lo + 1;
        const int per = (nk + n_splits - 1) / n_splits;
        const int k0 = lo + s * per;
        const int k1 = std::min(hi + 1, k0 + per);
        const int kh = h / (n_head / n_head_kv);
        // Q lives in the fused q|k|v buffer, so a row is qkv_stride floats apart -
        // NOT n_head*head_dim.  With the segment width the anchor row (r == 0, whose
        // offset is 0 either way) matched a host reference exactly while every mask
        // row read the previous row's K region as its query.
        const float * qh = qbuf + (size_t)r * row_stride + (size_t)h * head_dim;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        float mx = -std::numeric_limits<float>::infinity();
        float ls = 0.f;
        for (int kk = k0; kk < k1; kk++) {
            const size_t cell = (size_t)(kk % ring) * n_head_kv + kh;
            // one warp per key, 4 dims per lane: only this lane's 4 K and 4 V
            // values are live at a time (a per-lane head_dim array would be 128
            // registers and spill)
            float k4[4], v4[4];
            if (kv_bytes == 4) {
                const float * kr = (const float *)kpool + cell * head_dim + lane;
                const float * vr = (const float *)vpool + cell * head_dim + lane;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    k4[i] = kr[32 * i];
                    v4[i] = vr[32 * i];
                }
            } else {
                const uint16_t * kr = (const uint16_t *)kpool + cell * head_dim + lane;
                const uint16_t * vr = (const uint16_t *)vpool + cell * head_dim + lane;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    k4[i] = fast_h2f(kr[32 * i]);
                    v4[i] = fast_h2f(vr[32 * i]);
                }
            }
            float qd = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                qd += qh[lane + 32 * i] * k4[i];
            }
            qd = sg_sum(qd, it.get_sub_group()) * scale;
            const float nm = mx > qd ? mx : qd;
            const float corr = sycl::exp(mx - nm);
            const float pw = sycl::exp(qd - nm);
            ls = ls * corr + pw;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                acc[i] = acc[i] * corr + pw * v4[i];
            }
            mx = nm;
        }
        float * out = partials + (size_t)g * pstride;
        out[0] = (ls > 0.f) ? mx : -std::numeric_limits<float>::infinity();
        out[1] = ls;
#pragma unroll
        for (int i = 0; i < head_dim / 32; i++) {
            out[2 + lane + 32 * i] = acc[i];
        }
    });
}

void df_attn_combine_launch(queue & q, const float * partials, float * out, int n_rows, int n_head, int head_dim,
                            int n_splits, const float * sinks) {
    if (n_rows <= 0 || n_splits <= 0) {
        return;
    }
    const int pstride = 2 + head_dim;
    const int ngroup = n_rows * n_head;
    q.parallel_for(nd_range<1>((size_t)ngroup * head_dim, head_dim), [=](nd_item<1> it) {
        const int g = (int)it.get_group(0);
        const int d = (int)it.get_local_id(0);
        const float * base = partials + (size_t)g * n_splits * pstride;
        float mx = -std::numeric_limits<float>::infinity();
        for (int s = 0; s < n_splits; s++) {
            mx = std::max(mx, base[(size_t)s * pstride]);
        }
        float ls = 0.f, acc = 0.f;
        for (int s = 0; s < n_splits; s++) {
            const float * pp = base + (size_t)s * pstride;
            const float w = (pp[1] > 0.f) ? sycl::exp(pp[0] - mx) : 0.f;
            ls += pp[1] * w;
            acc += pp[2 + d] * w;
        }
        float o = (ls > 0.f) ? (acc / ls) : 0.f;
        if (sinks && ls > 0.f) {
            // Attention sink (llama.cpp's ggml_soft_max_add_sinks): one extra key per
            // query head with score sinks[h] and value 0.  It leaves the numerator
            // alone and only adds exp(sink - max) to the denominator, so the whole
            // effect is a factor z / (z + exp(sink - max)).  It has to be applied HERE
            // rather than in the per-split kernel because both mx and ls are only
            // global after the splits are combined.
            const float h0 = sinks[g / n_rows];
            o *= 1.f / (1.f + sycl::exp(h0 - mx - sycl::log(ls)));
        }
        out[(size_t)g * head_dim + d] = o;
    });
}

// ---------------------------------------------------------------------------
// Per-row top-k over an [M][n] logit matrix (DFlash2's candidate sets).
//
// One work-group per row: every lane keeps a private descending top-K in SLM
// behind a threshold (one compare per element in the common case), then K-1
// pairwise merges of two sorted-K lists halve the candidate count per level
// (8 levels for 256 lanes).  Exact, and unlike K rounds of argmax-and-exclude it
// reads the row once.  Ties resolve to the lower token id.
// One workgroup reduces one contiguous slice of one row to its top-K, written to
// pids[(r*S + sl)*K .. ] in descending order.  Splitting the row this way puts
// M*S workgroups on the device instead of M: with M=6 a one-workgroup-per-row
// grid occupies 6 of 512 EUs and the whole kernel measured 2.1 ms for 5.96 MB
// (~2.8 GB/s), i.e. ~100x off the bandwidth floor and 8% of a 28 ms cycle.
static void df_topk_slice(queue & q, const float * logits, int n, int32_t * pids, float * pvals, int M, int K,
                          int S) {
    if (M <= 0 || n <= 0 || K <= 0 || K > kTopkMaxK || S <= 0) {
        return;
    }
    const int K_ = K;
    constexpr int WG = 256;
    const float kNinf = -std::numeric_limits<float>::infinity();
    q.submit([&](handler & h) {
        local_accessor<float, 1> lv((size_t)WG * K_ + 2, h);
        local_accessor<int32_t, 1> li((size_t)WG * K_ + 1, h);
        local_accessor<float, 1> red(WG, h);
        local_accessor<int32_t, 1> rid(WG, h);
        h.parallel_for(nd_range<1>((size_t)M * S * WG, WG), [=](nd_item<1> it) {
            const int g = (int)it.get_group(0);
            const int r = g / S;
            const int sl = g - r * S;
            const int tid = (int)it.get_local_id(0);
            const float * row = logits + (size_t)r * n;
            const int lo = (int)(((long long)n * sl) / S);
            const int hi = (int)(((long long)n * (sl + 1)) / S);
            int cnt = 0;
            for (int i = lo + tid; i < hi; i += WG) {
                const float v = row[i];
                int p = (cnt < K_) ? cnt : K_ - 1;
                if (cnt == K_ && v <= lv[(K_ - 1) * WG + tid]) {
                    continue;
                }
                while (p > 0) {
                    const float pv = lv[(p - 1) * WG + tid];
                    const int32_t pi = li[(p - 1) * WG + tid];
                    if (pv > v || (pv == v && pi < i)) {
                        break;
                    }
                    lv[p * WG + tid] = pv;
                    li[p * WG + tid] = pi;
                    p--;
                }
                lv[p * WG + tid] = v;
                li[p * WG + tid] = i;
                if (cnt < K_) {
                    cnt++;
                }
            }
            it.barrier(access::fence_space::local_space);
            constexpr int32_t kNone = std::numeric_limits<int32_t>::max();
            const int rbase = r * S;
            for (int rk = 0; rk < K_; rk++) {
                float v = kNinf;
                for (int k = 0; k < cnt; k++) {
                    v = sycl::fmax(v, lv[k * WG + tid]);
                }
                red[tid] = v;
                it.barrier(access::fence_space::local_space);
                for (int st = WG / 2; st > 0; st >>= 1) {
                    if (tid < st) {
                        red[tid] = sycl::fmax(red[tid], red[tid + st]);
                    }
                    it.barrier(access::fence_space::local_space);
                }
                const float gv = red[0];
                int32_t gi = kNone;
                for (int k = 0; k < cnt; k++) {
                    if (lv[k * WG + tid] == gv && li[k * WG + tid] < gi) {
                        gi = li[k * WG + tid];
                    }
                }
                rid[tid] = gi;
                it.barrier(access::fence_space::local_space);
                for (int st = WG / 2; st > 0; st >>= 1) {
                    if (tid < st) {
                        rid[tid] = std::min(rid[tid], rid[tid + st]);
                    }
                    it.barrier(access::fence_space::local_space);
                }
                gi = rid[0];
                if (tid == 0) {
                    pids[(size_t)(rbase + sl) * K_ + rk] = gi;
                    pvals[(size_t)(rbase + sl) * K_ + rk] = gv;
                    lv[WG * K_] = gv;
                    li[WG * K_] = gi;
                }
                it.barrier(access::fence_space::local_space);
                for (int k = 0; k < cnt; k++) {
                    if (li[k * WG + tid] == li[WG * K_]) {
                        lv[k * WG + tid] = kNinf;
                        li[k * WG + tid] = kNone;
                    }
                }
                it.barrier(access::fence_space::local_space);
            }
        });
    });
    (void)M;
}

void df_topk_launch(queue & q, const float * logits, int n, int32_t * ids, float * vals, int M, int K) {
    if (M <= 0 || n <= 0 || K <= 0 || K > kTopkMaxK) {
        return;
    }
    const int K_ = K;
    constexpr int WG = 256;
    const float kNinf = -std::numeric_limits<float>::infinity();
    q.submit([&](handler & h) {
        local_accessor<float, 1> lv((size_t)WG * K_ + 1, h);
        local_accessor<int32_t, 1> li((size_t)WG * K_ + 2, h);
        local_accessor<float, 1> red(WG, h);
        local_accessor<int32_t, 1> rid(WG, h);
        h.parallel_for(nd_range<1>((size_t)M * WG, WG), [=](nd_item<1> it) {
            const int r = (int)it.get_group(0);
            const int tid = (int)it.get_local_id(0);
            const float * row = logits + (size_t)r * n;
            // 1. each lane keeps its own descending top-K, behind a threshold so
            //    the common case is one compare per element
            int cnt = 0;
            for (int i = tid; i < n; i += WG) {
                const float v = row[i];
                int p = (cnt < K_) ? cnt : K_ - 1;
                if (cnt == K_ && v <= lv[(K_ - 1) * WG + tid]) {
                    continue;
                }
                while (p > 0) {
                    const float pv = lv[(p - 1) * WG + tid];
                    const int32_t pi = li[(p - 1) * WG + tid];
                    if (pv > v || (pv == v && pi < i)) {
                        break;
                    }
                    lv[p * WG + tid] = pv;
                    li[p * WG + tid] = pi;
                    p--;
                }
                lv[p * WG + tid] = v;
                li[p * WG + tid] = i;
                if (cnt < K_) {
                    cnt++;
                }
            }
            it.barrier(access::fence_space::local_space);
            // 2. K rounds of "take the global max, emit it, remove it".  A merge
            //    tree would be marginally faster, but 16 rounds over the SLM lists
            //    costs microseconds and is obviously correct.  Ties resolve to the
            //    lower token id, matching the host's `if (v[i] > best)` scan.
            constexpr int32_t kNone = std::numeric_limits<int32_t>::max();
            for (int rk = 0; rk < K_; rk++) {
                float v = kNinf;
                for (int k = 0; k < cnt; k++) {
                    v = sycl::fmax(v, lv[k * WG + tid]);
                }
                red[tid] = v;
                it.barrier(access::fence_space::local_space);
                // SLM tree over the 256 lanes (no sub-group permutes: this kernel's
                // sub-group range is a runtime value and the tree keeps the code
                // independent of it)
                for (int st = WG / 2; st > 0; st >>= 1) {
                    if (tid < st) {
                        red[tid] = sycl::fmax(red[tid], red[tid + st]);
                    }
                    it.barrier(access::fence_space::local_space);
                }
                // the winner is the lowest-index entry equal to the winning value
                int32_t ix = kNone;
                for (int k = 0; k < cnt; k++) {
                    if (lv[k * WG + tid] == v && li[k * WG + tid] < ix) {
                        ix = li[k * WG + tid];
                    }
                }
                rid[tid] = ix;
                it.barrier(access::fence_space::local_space);
                // every lane now holds the same global value in red[0]; pick the
                // lowest-index lane that holds it
                const float gv = red[0];
                int32_t gi = kNone;
                for (int k = 0; k < cnt; k++) {
                    if (lv[k * WG + tid] == gv && li[k * WG + tid] < gi) {
                        gi = li[k * WG + tid];
                    }
                }
                rid[tid] = gi;
                it.barrier(access::fence_space::local_space);
                for (int st = WG / 2; st > 0; st >>= 1) {
                    if (tid < st) {
                        rid[tid] = std::min(rid[tid], rid[tid + st]);
                    }
                    it.barrier(access::fence_space::local_space);
                }
                gi = rid[0];
                if (tid == 0) {
                    ids[(size_t)r * K_ + rk] = gi;
                    vals[(size_t)r * K_ + rk] = gv;
                    lv[WG * K_] = gv;
                    li[WG * K_] = gi;
                }
                it.barrier(access::fence_space::local_space);
                // remove the winner: every index appears in exactly one lane
                for (int k = 0; k < cnt; k++) {
                    if (li[k * WG + tid] == li[WG * K_]) {
                        lv[k * WG + tid] = kNinf;
                        li[k * WG + tid] = kNone;
                    }
                }
                it.barrier(access::fence_space::local_space);
            }
            // ggml's top_k swaps the first two entries after the sort ("emphasize
            // that the order is not important").  DFlash2's selector is NOT
            // order-insensitive: the lattice walk treats candidate index k as the
            // successor of predecessor index k, so the reference's (1,0) order is
            // part of the model's definition.  Reproduce it exactly, or every
            // transition edge is paired with the wrong predecessor.
            if (K_ > 1 && tid == 0) {
                const int32_t ti = ids[(size_t)r * K_ + 0];
                const float tv = vals[(size_t)r * K_ + 0];
                ids[(size_t)r * K_ + 0] = ids[(size_t)r * K_ + 1];
                vals[(size_t)r * K_ + 0] = vals[(size_t)r * K_ + 1];
                ids[(size_t)r * K_ + 1] = ti;
                vals[(size_t)r * K_ + 1] = tv;
            }
        });
    });
}

// Merge pass: one workgroup per row reduces the S*K slice partials to the row's
// top-K.  Same "K rounds of take the global max and remove it" structure as the
// slice pass, so the tie-break (lowest token id wins) is identical at both stages
// and the merged order matches what a single-pass reduction would produce.
static void df_topk_merge(queue & q, int32_t * pids, float * pvals, int32_t * ids, float * vals, int M, int K, int S) {
    constexpr int WG = 256;
    const int K_ = K;
    const float kNinf = -std::numeric_limits<float>::infinity();
    q.submit([&](handler & h) {
        local_accessor<float, 1> lv((size_t)WG * K_ + 1, h);
        local_accessor<int32_t, 1> li((size_t)WG * K_ + 1, h);
        local_accessor<float, 1> red(WG, h);
        local_accessor<int32_t, 1> rid(WG, h);
        h.parallel_for(nd_range<1>((size_t)M * WG, WG), [=](nd_item<1> it) {
            const int r = (int)it.get_group(0);
            const int tid = (int)it.get_local_id(0);
            const int ncand = S * K_;
            int cnt = 0;
            // lane t owns candidates t, t+WG, ... and keeps the descending top-K
            for (int c = tid; c < ncand; c += WG) {
                const float v = pvals[(size_t)r * ncand + c];
                const int32_t id = pids[(size_t)r * ncand + c];
                if (!(v > kNinf)) {
                    continue;  // slice ran short of K candidates
                }
                int p = (cnt < K_) ? cnt : K_ - 1;
                if (cnt == K_ && v <= lv[(K_ - 1) * WG + tid]) {
                    continue;
                }
                while (p > 0) {
                    const float pv = lv[(p - 1) * WG + tid];
                    const int32_t pi = li[(p - 1) * WG + tid];
                    if (pv > v || (pv == v && pi < id)) {
                        break;
                    }
                    lv[p * WG + tid] = pv;
                    li[p * WG + tid] = pi;
                    p--;
                }
                lv[p * WG + tid] = v;
                li[p * WG + tid] = id;
                if (cnt < K_) {
                    cnt++;
                }
            }
            it.barrier(access::fence_space::local_space);
            constexpr int32_t kNone = std::numeric_limits<int32_t>::max();
            for (int rk = 0; rk < K_; rk++) {
                float v = kNinf;
                for (int k = 0; k < cnt; k++) {
                    v = sycl::fmax(v, lv[k * WG + tid]);
                }
                red[tid] = v;
                it.barrier(access::fence_space::local_space);
                for (int st = WG / 2; st > 0; st >>= 1) {
                    if (tid < st) {
                        red[tid] = sycl::fmax(red[tid], red[tid + st]);
                    }
                    it.barrier(access::fence_space::local_space);
                }
                const float gv = red[0];
                int32_t gi = kNone;
                for (int k = 0; k < cnt; k++) {
                    if (lv[k * WG + tid] == gv && li[k * WG + tid] < gi) {
                        gi = li[k * WG + tid];
                    }
                }
                rid[tid] = gi;
                it.barrier(access::fence_space::local_space);
                for (int st = WG / 2; st > 0; st >>= 1) {
                    if (tid < st) {
                        rid[tid] = std::min(rid[tid], rid[tid + st]);
                    }
                    it.barrier(access::fence_space::local_space);
                }
                gi = rid[0];
                if (tid == 0) {
                    ids[(size_t)r * K_ + rk] = gi;
                    vals[(size_t)r * K_ + rk] = gv;
                    lv[WG * K_] = gv;
                    li[WG * K_] = gi;
                }
                it.barrier(access::fence_space::local_space);
                for (int k = 0; k < cnt; k++) {
                    if (li[k * WG + tid] == li[WG * K_]) {
                        lv[k * WG + tid] = kNinf;
                        li[k * WG + tid] = kNone;
                    }
                }
                it.barrier(access::fence_space::local_space);
            }
            // ggml's top_k swaps the first two entries after the sort.  DFlash2's
            // selector is NOT order-insensitive - the lattice walk treats
            // candidate k as the successor of candidate k - so reproduce it.
            if (K_ > 1 && tid == 0) {
                const int32_t ti = ids[(size_t)r * K_ + 0];
                const float tv = vals[(size_t)r * K_ + 0];
                ids[(size_t)r * K_ + 0] = ids[(size_t)r * K_ + 1];
                vals[(size_t)r * K_ + 0] = vals[(size_t)r * K_ + 1];
                ids[(size_t)r * K_ + 1] = ti;
                vals[(size_t)r * K_ + 1] = tv;
            }
        });
    });
}

// Sliced top-K: `S` workgroups per row, then one merge pass.  Callers that cannot
// provide partial scratch get the original single-pass behaviour.
void df_topk_launch(queue & q, const float * logits, int n, int32_t * ids, float * vals, int M, int K, int32_t * pids,
                    float * pvals, int S) {
    if (M <= 0 || n <= 0 || K <= 0 || K > kTopkMaxK) {
        return;
    }
    if (!pids || !pvals || S <= 1) {
        df_topk_launch(q, logits, n, ids, vals, M, K);
        return;
    }
    // Keep at least a couple of elements per lane in every slice, so a lane's
    // top-K list is not trivially short and the merge has real candidates.
    int s_use = S;
    while (s_use > 1 && (long long)n / s_use < (long long)2 * kTopkWG) {
        s_use >>= 1;
    }
    df_topk_slice(q, logits, n, pids, pvals, M, K, s_use);
    df_topk_merge(q, pids, pvals, ids, vals, M, K, s_use);
}

// ---------------------------------------------------------------------------
// DFlash2 candidate selector.
//
// For every block position i (1..n_blk-1) and every (p, c) pair of candidates
// (p from position i-1's set, c from position i's set):
//     edge(p->c) = < A[p] * gate(h_i), B[c] > + unary_i[c]
// with A/B the per-token rank-space codebooks (row = one Q4_K block, 256 floats),
// gate = selector_hidden.weight @ h_i and unary the candidate's own logit.  The
// host then walks one path through the lattice (see engine_dflash.cpp).
//
// One work-group per position: the 16 conditional rows (A rows premultiplied by
// gate) and the 16 successor rows are staged in SLM, then the 256 threads each
// evaluate one (p, c) pair.
void df_sel_launch(queue & q, const int32_t * ids, const float * vals, const float * gate,
                   const uint8_t * prev_vals, const uint16_t * prev_scales, const uint16_t * prev_offs,
                   const uint8_t * next_vals, const uint16_t * next_scales, const uint16_t * next_offs, float * lattice,
                   int32_t anchor, int n_blk, int n_vocab, int rank, int K) {
    if (n_blk <= 1 || K <= 0 || rank <= 0) {
        return;
    }
    const int K_ = K;
    const int NPOS = n_blk - 1;
    const int row = K_ + K_ * K_; // ids + score matrix, per position
    constexpr int WG = 256;
    const int nchunk = (K_ * rank) / (WG * 4); // 4-element chunks per thread
    q.submit([&](handler & h) {
        local_accessor<float, 1> cond((size_t)K_ * rank, h);
        local_accessor<float, 1> succ((size_t)K_ * rank, h);
        h.parallel_for(nd_range<1>((size_t)NPOS * WG, WG), [=](nd_item<1> it) {
            const int i = (int)it.get_group(0) + 1; // block position
            const int tid = (int)it.get_local_id(0);
            // stage the K*rank codebook elements: K*rank/4 chunks of 4 dims,
            // `WG` threads at a time (16 rows x 256 dims = 4 chunks per thread)
            for (int ch = 0; ch < nchunk; ch++) {
                const int e = (ch * WG + tid) * 4; // element index into [K][rank]
                const int p = e / rank;
                const int d = e % rank;
                // position 1's predecessor is the anchor token, replicated over
                // the 16 rows exactly as llama.cpp's n_pred == 1 expansion does
                const int32_t id_a = (i == 1) ? anchor : ids[(size_t)(i - 1) * K_ + p];
                // the gate is per block position: llama.cpp's gate_blk is
                // [rank, n_pos, n_blocks], so a position's rows use ITS gate
                const float gt = gate[(size_t)i * rank + d];
                if (id_a >= 0 && id_a < n_vocab) {
                    const uint8_t * nb = prev_vals + (size_t)id_a * (rank / 2);
                    const float st = mb_h2f(prev_scales[(size_t)(d / 32) * n_vocab + id_a]);
                    const float of = mb_h2f(prev_offs[(size_t)(d / 32) * n_vocab + id_a]);
                    for (int s = 0; s < 4; s++) {
                        const int ee = d + s;
                        const uint8_t by = nb[ee >> 1];
                        const uint32_t qv = (ee & 1) ? (uint32_t)(by >> 4) : (uint32_t)(by & 0xF);
                        cond[(size_t)p * rank + ee] = (st * (float)qv + of) * gt;
                    }
                } else {
                    for (int s = 0; s < 4; s++) {
                        cond[(size_t)p * rank + d + s] = 0.f;
                    }
                }
            }
            for (int ch = 0; ch < nchunk; ch++) {
                const int e = (ch * WG + tid) * 4; // element index into [K][rank]
                const int c = e / rank;
                const int d = e % rank;
                const int32_t id_b = ids[(size_t)i * K_ + c];
                if (id_b >= 0 && id_b < n_vocab) {
                    const uint8_t * nb = next_vals + (size_t)id_b * (rank / 2);
                    const float st = mb_h2f(next_scales[(size_t)(d / 32) * n_vocab + id_b]);
                    const float of = mb_h2f(next_offs[(size_t)(d / 32) * n_vocab + id_b]);
                    for (int s = 0; s < 4; s++) {
                        const int ee = d + s;
                        const uint8_t by = nb[ee >> 1];
                        const uint32_t qv = (ee & 1) ? (uint32_t)(by >> 4) : (uint32_t)(by & 0xF);
                        succ[(size_t)c * rank + ee] = st * (float)qv + of;
                    }
                } else {
                    for (int s = 0; s < 4; s++) {
                        succ[(size_t)c * rank + d + s] = 0.f;
                    }
                }
            }
            it.barrier(access::fence_space::local_space);
            if (tid < K_ * K_) {
                const int pp = tid / K_;
                const int cc = tid % K_;
                const float * a = &cond[(size_t)pp * rank];
                const float * b = &succ[(size_t)cc * rank];
                float s = 0.f;
                for (int e = 0; e < rank; e++) {
                    s += a[e] * b[e];
                }
                lattice[(size_t)i * row + K_ + tid] = s + vals[(size_t)i * K_ + cc];
            }
        });
    });
    // the candidate ids are copied next to the scores so the host reads one row
    q.parallel_for(nd_range<1>((size_t)NPOS * K_, K_), [=](nd_item<1> it) {
        const int i = (int)it.get_group(0) + 1;
        const int c = (int)it.get_local_id(0);
        lattice[(size_t)i * row + c] = (float)ids[(size_t)i * K_ + c];
    });
}

} // namespace si
