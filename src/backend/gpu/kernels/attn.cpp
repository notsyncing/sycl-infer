#include "kernels.h"
#include "device/device_profile.h"
#include "kernel_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "common/env.h"

namespace si {

using namespace sycl;
using namespace si::kd;

// ---------------------------------------------------------------------------
// PF_ATTN_VEC=0: classic per-lane strided mapping
static inline bool attn_vec_env() {
    static const bool v = [] {
        const char * e = si::env::str("PF_ATTN_VEC");
        const int dflt = si::dev::active().attn.vec;
        return (e ? atoi(e) != 0 : dflt != 0);
    }();
    return v;
}

// ---------------------------------------------------------------------------
// Grouped attention (PF_DEC_GROUP=1; opt-in, decode and partial-split
// prefill).  n_head/n_head_kv query heads share one kv head (4 for this
// model), so the classic kernel - one warp per (row, token, head, split) -
// reads each kv head's K and V once per query head, i.e. 4x per key, which is
// what saturates L2 at long context.  Here one warp owns all HPG heads of a
// kv head: K/V are loaded once per key and reused from registers for every
// head, and the per-head arithmetic (dot product, online softmax, V
// accumulation) keeps the exact op sequence of the classic vectorized kernel,
// so the partials are bit-identical and argmax/top-1 cannot change by
// construction.
template <int HPG, typename KV>
static void attn_group_kernel(queue & q, const float * qbuf, const KV * kp_base, const KV * vp_base, float * partials,
                              const int32_t * tables, int n_head, int n_head_kv, int head_dim, int n_splits,
                              const step_info * info, float scale, int max_blocks, int n_rows, int n_real) {
    constexpr int HD = 256;
    const int qstride = n_head * 2 * head_dim;
    const int pstride = 2 + head_dim;
    const int n_wg = n_rows * n_real * n_head_kv * n_splits;
    q.parallel_for(nd_range<1>((size_t)n_wg * 32, 32), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int gid = it.get_group(0);
        const int r = gid / (n_real * n_head_kv * n_splits);
        const int rem0 = gid % (n_real * n_head_kv * n_splits);
        const int t = rem0 / (n_head_kv * n_splits);
        const int rem = rem0 % (n_head_kv * n_splits);
        const int kvh = rem / n_splits;
        const int s = rem % n_splits;
        const int h0 = kvh * HPG;
        const int lane = it.get_local_id(0);
        const sub_group sgg = it.get_sub_group();
        float * part0 =
            partials + (((size_t)r * info->tpb + t) * n_head + h0) * n_splits * pstride + (size_t)s * pstride;

        if (r >= info->n_rows || t >= row_nr(info, r) || !info->active[r]) {
#pragma unroll
            for (int j = 0; j < HPG; j++) {
                float * part = part0 + (size_t)j * n_splits * pstride;
                if (lane == 0) {
                    part[0] = -INFINITY;
                    part[1] = 0.f;
                }
#pragma unroll
                for (int i = 0; i < HD / 32; i++) {
                    part[2 + lane + 32 * i] = 0.f;
                }
            }
            return;
        }
        const int pos = info->pos[r] + t;
        const int row = r * info->tpb + t;
        const int n_kv = pos + 1;
        const int chunk = (n_kv + n_splits - 1) / n_splits;
        const int t0 = s * chunk;
        const int t1 = sycl::min(t0 + chunk, n_kv);
        const int32_t * table = tables + (size_t)info->slot[r] * max_blocks;

        // same lane->dim mapping as the classic vectorized path: two coalesced
        // float4 chunks per lane, [4*lane, 4*lane+4) and + HD/2
        const int d0 = lane * 4;
        const int d1 = d0 + HD / 2;
        sycl::float4 qa[HPG], qb[HPG], aa[HPG], ab[HPG];
        float m[HPG], l[HPG];
#pragma unroll
        for (int j = 0; j < HPG; j++) {
            const float * qh = qbuf + (size_t)row * qstride + (size_t)(h0 + j) * 2 * head_dim;
            qa[j] = *reinterpret_cast<const sycl::float4 *>(qh + d0);
            qb[j] = *reinterpret_cast<const sycl::float4 *>(qh + d1);
            aa[j] = sycl::float4(0.f, 0.f, 0.f, 0.f);
            ab[j] = sycl::float4(0.f, 0.f, 0.f, 0.f);
            m[j] = -INFINITY;
            l[j] = 0.f;
        }
        for (int kk = t0; kk < t1; kk++) {
            const int kb = table[kk / kBlockSize];
            const int ko = kk % kBlockSize;
            const KV * kp = kp_base + (((size_t)kb * n_head_kv + kvh) * kBlockSize + ko) * head_dim;
            const KV * vp = vp_base + (((size_t)kb * n_head_kv + kvh) * kBlockSize + ko) * head_dim;
            // one K load and one V load per key, shared by all HPG heads
            const sycl::float4 ka = kv_ld4(kp + d0);
            const sycl::float4 kb4 = kv_ld4(kp + d1);
            const sycl::float4 va = kv_ld4(vp + d0);
            const sycl::float4 vb = kv_ld4(vp + d1);
#pragma unroll
            for (int j = 0; j < HPG; j++) {
                float dot = dot4(qa[j], ka) + dot4(qb[j], kb4);
                dot = sg_sum(dot, sgg) * scale;
                const float mnew = sycl::max(m[j], dot);
                const float e = sycl::exp(dot - mnew);
                const float corr = sycl::exp(m[j] - mnew);
                l[j] = l[j] * corr + e;
                aa[j] = fma4(va, e, mul4(aa[j], corr));
                ab[j] = fma4(vb, e, mul4(ab[j], corr));
                m[j] = mnew;
            }
        }
#pragma unroll
        for (int j = 0; j < HPG; j++) {
            float * part = part0 + (size_t)j * n_splits * pstride;
            if (lane == 0) {
                part[0] = m[j];
                part[1] = l[j];
            }
            *reinterpret_cast<sycl::float4 *>(part + 2 + d0) = aa[j];
            *reinterpret_cast<sycl::float4 *>(part + 2 + d1) = ab[j];
        }
    });
}

// ---------------------------------------------------------------------------
// Flash-style tiled prefill attention (i8/i4 KV, opt-in via PF_ATTN_FLASH=1).
//
// The classic kernel gives one warp one (row, token, query head, split), so
// every K/V row is read once per query head (HPG-fold redundancy, 6x on the
// 27B) and once per query token; at 16k that is ~1.66 TB of K/V reads per
// 512-token chunk at ~613 GB/s, i.e. bandwidth-bound.  This kernel keeps the
// per-key online softmax but stages a BK-key K/V block in SLM once and reuses
// it across a query tile of BQ tokens and all HPG query heads of the kv head
// (BQ*HPG-fold traffic cut).  One work-group = (row r, query tile qt, kv head,
// split s); one warp owns one (query, head) pair.  The partials layout is
// unchanged, so attn_combine_launch still merges it.
template <int HPG, typename KV>
static void attn_flash_kernel(queue & q, const float * qbuf, const KV * kp_base, const KV * vp_base,
                              const sycl::half * ksc_base, const sycl::half * vsc_base, float * partials,
                              const int32_t * tables, int n_head, int n_head_kv, int head_dim, int n_splits,
                              const step_info * info, float scale, int max_blocks, int n_rows, int n_real) {
    constexpr int HD = 256;
    constexpr int BQ = 4;  // query tokens per work-group
    constexpr int BK = 64; // keys staged per SLM block
    constexpr bool I4 = std::is_same_v<KV, uint8_t>;
    constexpr int ROW = I4 ? HD / 2 : HD; // pool elements per KV row
    constexpr int NSC = HD / 32;          // fp16 block scales per row
    const int qstride = n_head * 2 * head_dim;
    const int pstride = 2 + head_dim;
    const int n_qt = (kMaxT + BQ - 1) / BQ;
    const int n_wg = n_rows * n_qt * n_head_kv * n_splits;
    const int wg = BQ * HPG * 32;
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<KV, 1> s_k((size_t)BK * ROW, h);
        sycl::local_accessor<KV, 1> s_v((size_t)BK * ROW, h);
        sycl::local_accessor<sycl::half, 1> s_ks((size_t)BK * NSC, h);
        sycl::local_accessor<sycl::half, 1> s_vs((size_t)BK * NSC, h);
        h.parallel_for(nd_range<1>((size_t)n_wg * wg, wg), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int gid = it.get_group(0);
            const int r = gid / (n_qt * n_head_kv * n_splits);
            const int rem0 = gid % (n_qt * n_head_kv * n_splits);
            const int qt = rem0 / (n_head_kv * n_splits);
            const int rem = rem0 % (n_head_kv * n_splits);
            const int kvh = rem / n_splits;
            const int s = rem % n_splits;
            const int h0 = kvh * HPG;
            const int tid = it.get_local_linear_id();
            const int warp = tid / 32;
            const int lane = tid % 32;
            const int ql = warp / HPG; // query token inside the tile
            const int j = warp % HPG;  // query head inside the kv group
            const int t = qt * BQ + ql;
            const sub_group sgg = it.get_sub_group();
            const bool valid = (t < row_nr(info, r)) && info->active[r];
            const int pos = valid ? info->pos[r] + t : -1;
            // Uniform key range for the whole work-group (barriers must be hit
            // the same number of times by every warp): the split covers
            // [0, max_nkv) of the tile's largest position, and each warp masks
            // the keys beyond its own causal bound below.
            const int tmax = info->active[r] ? sycl::min(qt * BQ + BQ, row_nr(info, r)) : 0;
            const int max_nkv = tmax > 0 ? info->pos[r] + tmax : 0;
            const int chunk = (max_nkv + n_splits - 1) / n_splits;
            const int t0 = s * chunk;
            const int t1 = sycl::min(t0 + chunk, max_nkv);
            const int32_t * table = tables + (size_t)info->slot[r] * max_blocks;
            const int d0 = lane * 4;
            const int d1 = d0 + HD / 2;
            const int row = r * info->tpb + t;
            float * part = partials + (((size_t)r * info->tpb + t) * n_head + h0 + j) * n_splits * pstride
                           + (size_t)s * pstride;
            sycl::float4 qa(0.f, 0.f, 0.f, 0.f), qb(0.f, 0.f, 0.f, 0.f);
            sycl::float4 aa(0.f, 0.f, 0.f, 0.f), ab(0.f, 0.f, 0.f, 0.f);
            float m = -INFINITY, l = 0.f;
            if (valid) {
                const float * qh = qbuf + (size_t)row * qstride + (size_t)(h0 + j) * 2 * head_dim;
                qa = *reinterpret_cast<const sycl::float4 *>(qh + d0);
                qb = *reinterpret_cast<const sycl::float4 *>(qh + d1);
            }
            for (int k0 = t0; k0 < t1; k0 += BK) {
                const int nk = sycl::min(BK, t1 - k0);
                for (int i = tid; i < nk * ROW; i += wg) {
                    const int k = i / ROW, jj = i % ROW;
                    const int kk = k0 + k;
                    const int kb = table[kk / kBlockSize];
                    const int ko = kk % kBlockSize;
                    const size_t unit = (size_t)kb * n_head_kv + kvh;
                    s_k[i] = kv_row_data(kp_base, unit, ko, head_dim)[jj];
                    s_v[i] = kv_row_data(vp_base, unit, ko, head_dim)[jj];
                }
                for (int i = tid; i < nk * NSC; i += wg) {
                    const int k = i / NSC, jj = i % NSC;
                    const int kk = k0 + k;
                    const int kb = table[kk / kBlockSize];
                    const int ko = kk % kBlockSize;
                    const size_t unit = (size_t)kb * n_head_kv + kvh;
                    s_ks[i] = kv_row_scales(ksc_base, unit, ko, head_dim)[jj];
                    s_vs[i] = kv_row_scales(vsc_base, unit, ko, head_dim)[jj];
                }
                it.barrier();
                if (valid) {
                    for (int k = 0; k < nk; k++) {
                        if (k0 + k > pos) {
                            break; // causal
                        }
                        sycl::float4 ka, kb4, va, vb;
                        if constexpr (I4) {
                            ka = i4_ld4(&s_k[k * ROW], d0);
                            kb4 = i4_ld4(&s_k[k * ROW], d1);
                            va = i4_ld4(&s_v[k * ROW], d0);
                            vb = i4_ld4(&s_v[k * ROW], d1);
                        } else {
                            ka = kv_ld4(&s_k[k * ROW] + d0);
                            kb4 = kv_ld4(&s_k[k * ROW] + d1);
                            va = kv_ld4(&s_v[k * ROW] + d0);
                            vb = kv_ld4(&s_v[k * ROW] + d1);
                        }
                        const float sA = (float)s_ks[k * NSC + lane / 8];
                        const float sB = (float)s_ks[k * NSC + 4 + lane / 8];
                        const float vsA = (float)s_vs[k * NSC + lane / 8];
                        const float vsB = (float)s_vs[k * NSC + 4 + lane / 8];
                        float dot = dot4(qa, ka) * sA + dot4(qb, kb4) * sB;
                        dot = sg_sum(dot, sgg) * scale;
                        const float mnew = sycl::max(m, dot);
                        const float e = sycl::exp(dot - mnew);
                        const float corr = sycl::exp(m - mnew);
                        l = l * corr + e;
                        aa = fma4(va, e * vsA, mul4(aa, corr));
                        ab = fma4(vb, e * vsB, mul4(ab, corr));
                        m = mnew;
                    }
                }
                it.barrier();
            }
            if (valid) {
                if (lane == 0) {
                    part[0] = m;
                    part[1] = l;
                }
                *reinterpret_cast<sycl::float4 *>(part + 2 + d0) = aa;
                *reinterpret_cast<sycl::float4 *>(part + 2 + d1) = ab;
            }
        });
    });
}

// PF_ATTN_FLASH=1 enables the flash tiled prefill attention.  It is opt-in
// (default off): the prototype stages K/V in SLM and does FA-2 sub-blocks, but
// measured on 2x A770 it is *not* faster than the classic kernel (96 vs 140
// t/s pp512 at 16k), because the classic is bound by the per-score arithmetic
// (the 32-lane dot + sub-group reduction + exp chain), not by the redundant
// K/V traffic that the tiling removes.  Beating it needs the dot in int8/dp4a
// and a key-per-lane mapping, not just tiling.
static inline bool attn_flash_env() {
    static const bool v = [] {
        const char * e = si::env::str("PF_ATTN_FLASH");
        return e && atoi(e) != 0;
    }();
    return v;
}

// PF_DEC_GROUP=1 enables the grouped kernel; it is opt-in because on the Iris
// Xe the classic kernel is measurably faster (it has 4x the warps: the K/V
// redundancy is served by L2 and the extra parallelism wins).  The grouped path
// needs a non-fused call (out == nullptr) and n_head = 4 * n_head_kv with the
// 256-wide head.
static inline int dec_group_env() {
    static const int v = [] {
        const char * e = si::env::str("PF_DEC_GROUP");
        // the grouped decode kernel has n_head_kv instead of n_head workgroups; it
        // is off on BOTH parts but for opposite measured reasons (A770: 5x slower
        // at 64k; Iris Xe: measurably faster for the classic kernel, which has 4x
        // the warps).  The default still belongs in the profile because it is a
        // per-device measurement, and the reasoning differs per device.
        const int dflt = si::dev::active().attn.dec_group;
        return (e ? atoi(e) : dflt) != 0 ? 1 : 0;
    }();
    return v;
}

void attn_launch(queue & q, const float * qbuf, const float * gate, const void * kpool, const void * vpool,
                 float * partials, const int32_t * tables, int n_head, int n_head_kv, int head_dim, int n_splits,
                 const step_info * info, float scale, int max_blocks, int n_rows, int n_real, float * out, int group,
                 const void * kscales, const void * vscales) {
    constexpr int HD = 256;
    const bool avec = attn_vec_env();
    const int qstride = n_head * 2 * head_dim;
    const int pstride = 2 + head_dim;
    const bool fuse = out != nullptr && n_splits == 1;
    const int n_wg = n_rows * n_real * n_head * n_splits;
    // XMX (oneDNN int8) attention: the fused mode-2 shape and the non-fused
    // (partials) mode-1/tail shape.  Opt-in: PF_ATTN_XMX=1.
    if (n_real > 1 && head_dim == HD) {
        if (attn_xmx_launch(q, qbuf, gate, kpool, vpool, partials, tables, n_head, n_head_kv, head_dim, n_splits,
                            info, scale, max_blocks, n_rows, n_real, out, kscales, vscales)) {
            return;
        }
    }
    // flash tiled prefill attention for i8/i4 KV (see attn_flash_kernel).
    // n_real > 1 restricts it to the prefill path: the single-token decode
    // keeps the dedicated classic/grouped kernels.
    if (!fuse && n_real > 1 && attn_flash_env() && head_dim == HD && n_head_kv > 0 && n_head % n_head_kv == 0
        && (n_head / n_head_kv == 4 || n_head / n_head_kv == 6) && kv_k_dtype() == kv_v_dtype()
        && kv_dtype_has_scales(kv_k_dtype())) {
        const int hpg = n_head / n_head_kv;
        static const bool dbg = si::env::flag("PF_ATTN_DBG");
        if (dbg) {
            fprintf(stderr, "[attn] flash kernel: hpg=%d n_splits=%d n_rows=%d n_real=%d\n", hpg, n_splits, n_rows,
                    n_real);
        }
        const sycl::half * ksc = (const sycl::half *)kscales;
        const sycl::half * vsc = (const sycl::half *)vscales;
        auto fk = [&](const auto * kp, const auto * vp) {
            if (hpg == 4) {
                attn_flash_kernel<4>(q, qbuf, kp, vp, ksc, vsc, partials, tables, n_head, n_head_kv, head_dim, n_splits,
                                     info, scale, max_blocks, n_rows, n_real);
            } else {
                attn_flash_kernel<6>(q, qbuf, kp, vp, ksc, vsc, partials, tables, n_head, n_head_kv, head_dim, n_splits,
                                     info, scale, max_blocks, n_rows, n_real);
            }
        };
        if (kv_k_dtype() == kv_dtype_t::i8) {
            fk((const int8_t *)kpool, (const int8_t *)vpool);
        } else {
            fk((const uint8_t *)kpool, (const uint8_t *)vpool);
        }
        return;
    }
    const int grp = group >= 0 ? group : dec_group_env();
    // the grouped kernel has no int8/int4 path (it reuses kv_ld4 on a full row)
    if (grp && !kv_dtype_has_scales(kv_k_dtype()) && !kv_dtype_has_scales(kv_v_dtype()) && !fuse && head_dim == HD
        && n_head_kv > 0 && n_head % n_head_kv == 0 && n_head / n_head_kv == 4) {
        switch (kv_k_dtype()) {
        case kv_dtype_t::f32:
            attn_group_kernel<4>(q, qbuf, (const float *)kpool, (const float *)vpool, partials, tables, n_head,
                                 n_head_kv, head_dim, n_splits, info, scale, max_blocks, n_rows, n_real);
            return;
        case kv_dtype_t::bf16:
            attn_group_kernel<4>(q, qbuf, (const sycl::ext::oneapi::bfloat16 *)kpool,
                                 (const sycl::ext::oneapi::bfloat16 *)vpool, partials, tables, n_head, n_head_kv,
                                 head_dim, n_splits, info, scale, max_blocks, n_rows, n_real);
            return;
        case kv_dtype_t::f16:
            attn_group_kernel<4>(q, qbuf, (const sycl::half *)kpool, (const sycl::half *)vpool, partials, tables,
                                 n_head, n_head_kv, head_dim, n_splits, info, scale, max_blocks, n_rows, n_real);
            return;
        case kv_dtype_t::i8: // quantized: no grouped path (excluded by the guard above)
        case kv_dtype_t::i4: break;
        }
    }
    auto launch = [&](const auto * kp_base, const auto * vp_base, const sycl::half * ksc_base,
                      const sycl::half * vsc_base) {
        using KV = std::remove_cv_t<std::remove_pointer_t<decltype(kp_base)>>;
        using VV = std::remove_cv_t<std::remove_pointer_t<decltype(vp_base)>>;
        constexpr bool I8 = std::is_same_v<KV, int8_t>;
        constexpr bool I4 = std::is_same_v<KV, uint8_t>;
        constexpr bool V8 = std::is_same_v<VV, int8_t>;
        constexpr bool V4 = std::is_same_v<VV, uint8_t>;
        q.parallel_for(nd_range<1>((size_t)n_wg * 32, 32), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int gid = it.get_group(0);
            const int r = gid / (n_real * n_head * n_splits);
            const int rem0 = gid % (n_real * n_head * n_splits);
            const int t = rem0 / (n_head * n_splits);
            const int rem = rem0 % (n_head * n_splits);
            const int h = rem / n_splits;
            const int s = rem % n_splits;
            const int lane = it.get_local_id(0);
            const sub_group sgg = it.get_sub_group();
            float * part =
                partials + (((size_t)r * info->tpb + t) * n_head + h) * n_splits * pstride + (size_t)s * pstride;

            if (r >= info->n_rows || t >= row_nr(info, r) || !info->active[r]) {
                if (lane == 0) {
                    part[0] = -INFINITY;
                    part[1] = 0.f;
                }
#pragma unroll
                for (int i = 0; i < HD / 32; i++) {
                    part[2 + lane + 32 * i] = 0.f;
                }
                return;
            }
            const int pos = info->pos[r] + t;
            const int row = r * info->tpb + t;
            const int n_kv = pos + 1;
            const int chunk = (n_kv + n_splits - 1) / n_splits;
            const int t0 = s * chunk;
            const int t1 = sycl::min(t0 + chunk, n_kv);

            const float * q = qbuf + (size_t)row * qstride + (size_t)h * 2 * head_dim;
            const int kvh = (h * n_head_kv) / n_head;
            const int32_t * table = tables + (size_t)info->slot[r] * max_blocks;

            // each lane owns 8 *contiguous* head dimensions (4x fewer load
            // instructions; the partials layout is unchanged)
            if (avec) {
                // two 16-byte chunks per lane, both warp-coalesced:
                // chunk A = floats [4*lane, 4*lane+4), chunk B = + HD/2
                const int d0 = lane * 4;
                const int d1 = d0 + HD / 2;
                const sycl::float4 qa = *reinterpret_cast<const sycl::float4 *>(q + d0);
                const sycl::float4 qb = *reinterpret_cast<const sycl::float4 *>(q + d1);
                sycl::float4 aa(0.f, 0.f, 0.f, 0.f), ab(0.f, 0.f, 0.f, 0.f);
                float m = -INFINITY, l = 0.f;
                for (int kk = t0; kk < t1; kk++) {
                    const int kb = table[kk / kBlockSize];
                    const int ko = kk % kBlockSize;
                    const size_t unit = (size_t)kb * n_head_kv + kvh;
                    const auto * kp = kv_row_data(kp_base, unit, ko, head_dim);
                    const auto * vp = kv_row_data(vp_base, unit, ko, head_dim);
                    sycl::float4 ka, kb4, va, vb;
                    if constexpr (I4) {
                        ka = i4_ld4(kp, d0);
                        kb4 = i4_ld4(kp, d1);
                    } else {
                        ka = kv_ld4(kp + d0);
                        kb4 = kv_ld4(kp + d1);
                    }
                    if constexpr (V4) {
                        va = i4_ld4(vp, d0);
                        vb = i4_ld4(vp, d1);
                    } else {
                        va = kv_ld4(vp + d0);
                        vb = kv_ld4(vp + d1);
                    }
                    float dot;
                    if constexpr (I8 || I4) {
                        // the two 4-dim chunks of a lane sit in blocks lane/8 and
                        // 4 + lane/8; the block scale multiplies the chunk's dot4
                        const auto * ks = kv_row_scales(ksc_base, unit, ko, head_dim);
                        const float sA = (float)ks[lane / 8];
                        const float sB = (float)ks[4 + lane / 8];
                        dot = dot4(qa, ka) * sA + dot4(qb, kb4) * sB;
                    } else {
                        dot = dot4(qa, ka) + dot4(qb, kb4);
                    }
                    dot = sg_sum(dot, sgg) * scale;
                    const float mnew = sycl::max(m, dot);
                    const float e = sycl::exp(dot - mnew);
                    const float corr = sycl::exp(m - mnew);
                    l = l * corr + e;
                    if constexpr (V8 || V4) {
                        const auto * vs = kv_row_scales(vsc_base, unit, ko, head_dim);
                        aa = fma4(va, e * (float)vs[lane / 8], mul4(aa, corr));
                        ab = fma4(vb, e * (float)vs[4 + lane / 8], mul4(ab, corr));
                    } else {
                        aa = fma4(va, e, mul4(aa, corr));
                        ab = fma4(vb, e, mul4(ab, corr));
                    }
                    m = mnew;
                }
                if (fuse) {
                    // n_splits == 1: no partial buffer round-trip and no combine
                    // kernel - every lane already holds the full softmax sum
                    const float * gb = gate + (size_t)row * qstride + h * 2 * head_dim + head_dim;
                    float * ob = out + (size_t)row * n_head * head_dim + h * head_dim;
                    sycl::float4 oa, obb;
                    for (int j = 0; j < 4; j++) {
                        oa[j] = (l > 0.f ? aa[j] / l : 0.f) * sigmoid_f(gb[d0 + j]);
                        obb[j] = (l > 0.f ? ab[j] / l : 0.f) * sigmoid_f(gb[d1 + j]);
                    }
                    *reinterpret_cast<sycl::float4 *>(ob + d0) = oa;
                    *reinterpret_cast<sycl::float4 *>(ob + d1) = obb;
                    return;
                }
                if (lane == 0) {
                    part[0] = m;
                    part[1] = l;
                }
                *reinterpret_cast<sycl::float4 *>(part + 2 + d0) = aa;
                *reinterpret_cast<sycl::float4 *>(part + 2 + d1) = ab;
                return;
            }

            float acc[HD / 32];
#pragma unroll
            for (int i = 0; i < HD / 32; i++) {
                acc[i] = 0.f;
            }
            float m = -INFINITY, l = 0.f;

            for (int kk = t0; kk < t1; kk++) {
                const int kb = table[kk / kBlockSize];
                const int ko = kk % kBlockSize;
                const size_t unit = (size_t)kb * n_head_kv + kvh;
                const auto * kp = kv_row_data(kp_base, unit, ko, head_dim);
                const auto * vp = kv_row_data(vp_base, unit, ko, head_dim);
                float dot = 0.f;
                if constexpr (I8 || I4) {
                    // dims lane+32*i are exactly block i: one scale per iteration
                    const auto * ks = kv_row_scales(ksc_base, unit, ko, head_dim);
#pragma unroll
                    for (int i = 0; i < HD / 32; i++) {
                        float kv;
                        if constexpr (I4) {
                            kv = i4_ld(kp, lane + 32 * i);
                        } else {
                            kv = kv_ld(kp + lane + 32 * i);
                        }
                        dot += q[lane + 32 * i] * kv * (float)ks[i];
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < HD / 32; i++) {
                        dot += q[lane + 32 * i] * kv_ld(kp + lane + 32 * i);
                    }
                }
                dot = sg_sum(dot, sgg) * scale;
                const float mnew = sycl::max(m, dot);
                const float e = sycl::exp(dot - mnew);
                const float corr = sycl::exp(m - mnew);
                l = l * corr + e;
                if constexpr (V8 || V4) {
                    const auto * vs = kv_row_scales(vsc_base, unit, ko, head_dim);
#pragma unroll
                    for (int i = 0; i < HD / 32; i++) {
                        float vv;
                        if constexpr (V4) {
                            vv = i4_ld(vp, lane + 32 * i);
                        } else {
                            vv = kv_ld(vp + lane + 32 * i);
                        }
                        acc[i] = acc[i] * corr + e * vv * (float)vs[i];
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < HD / 32; i++) {
                        acc[i] = acc[i] * corr + e * kv_ld(vp + lane + 32 * i);
                    }
                }
                m = mnew;
            }

            if (fuse) {
                const float * gb = gate + (size_t)row * qstride + h * 2 * head_dim + head_dim;
                float * ob = out + (size_t)row * n_head * head_dim + h * head_dim;
#pragma unroll
                for (int i = 0; i < HD / 32; i++) {
                    ob[lane + 32 * i] = (l > 0.f ? acc[i] / l : 0.f) * sigmoid_f(gb[lane + 32 * i]);
                }
                return;
            }
            if (lane == 0) {
                part[0] = m;
                part[1] = l;
            }
#pragma unroll
            for (int i = 0; i < HD / 32; i++) {
                part[2 + lane + 32 * i] = acc[i];
            }
        });
    };
    const sycl::half * ksc = (const sycl::half *)kscales;
    const sycl::half * vsc = (const sycl::half *)vscales;
    using bf16 = sycl::ext::oneapi::bfloat16;
    // K picks the launch instantiation, V is dispatched inside it, so
    // --kv-type K:V can mix the two scale-carrying types (i4/i8).
    auto with_k = [&](const auto * kp, const sycl::half * ksc_p) {
        switch (kv_v_dtype()) {
        case kv_dtype_t::f32: launch(kp, (const float *)vpool, ksc_p, nullptr); break;
        case kv_dtype_t::bf16: launch(kp, (const bf16 *)vpool, ksc_p, nullptr); break;
        case kv_dtype_t::f16: launch(kp, (const sycl::half *)vpool, ksc_p, nullptr); break;
        case kv_dtype_t::i8: launch(kp, (const int8_t *)vpool, ksc_p, vsc); break;
        case kv_dtype_t::i4: launch(kp, (const uint8_t *)vpool, ksc_p, vsc); break;
        }
    };
    switch (kv_k_dtype()) {
    case kv_dtype_t::f32: with_k((const float *)kpool, nullptr); break;
    case kv_dtype_t::bf16: with_k((const bf16 *)kpool, nullptr); break;
    case kv_dtype_t::f16: with_k((const sycl::half *)kpool, nullptr); break;
    case kv_dtype_t::i8: with_k((const int8_t *)kpool, ksc); break;
    case kv_dtype_t::i4: with_k((const uint8_t *)kpool, ksc); break;
    }
}

void attn_combine_launch(queue & q, const float * partials, const float * gate, float * out, const step_info * info,
                         int n_head, int head_dim, int n_splits, int n_rows, int n_real) {
    const int pstride = 2 + head_dim;
    const int qstride = n_head * 2 * head_dim;
    q.parallel_for(nd_range<1>((size_t)n_rows * n_real * n_head * head_dim, head_dim), [=](nd_item<1> it) {
        const int gid = it.get_group(0);
        const int r = gid / (n_real * n_head);
        const int t = (gid / n_head) % n_real;
        const int h = gid % n_head;
        const int d = it.get_local_id(0);
        if (r >= info->n_rows || t >= row_nr(info, r) || !info->active[r]) {
            return;
        }
        const int row = r * info->tpb + t;
        const float * part = partials + (((size_t)r * info->tpb + t) * n_head + h) * n_splits * pstride;
        float M = -INFINITY;
        for (int s = 0; s < n_splits; s++) {
            M = sycl::max(M, part[s * pstride]);
        }
        float sum = 0.f, acc = 0.f;
        for (int s = 0; s < n_splits; s++) {
            const float w = sycl::exp(part[s * pstride] - M);
            sum += part[s * pstride + 1] * w;
            acc += part[s * pstride + 2 + d] * w;
        }
        const float gate_v = sigmoid_f(gate[(size_t)row * qstride + h * 2 * head_dim + head_dim + d]);
        out[(size_t)row * n_head * head_dim + h * head_dim + d] = (sum > 0.f ? acc / sum : 0.f) * gate_v;
    });
}

} // namespace si
