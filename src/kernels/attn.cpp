#include "kernels.h"
#include "kernel_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

using namespace sycl;
using namespace si::kd;

// ---------------------------------------------------------------------------
// PF_ATTN_VEC=0: classic per-lane strided mapping
static inline bool attn_vec_env() {
    static const bool v = [] {
        const char * e = getenv("PF_ATTN_VEC");
        return !(e && atoi(e) == 0);
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
static void attn_group_kernel(queue & q, const float * qbuf, const KV * kp_base,
                              const KV * vp_base, float * partials, const int32_t * tables,
                              int n_head, int n_head_kv, int head_dim, int n_splits,
                              const step_info * info, float scale, int max_blocks, int n_rows,
                              int n_real) {
    constexpr int HD = 256;
    const int qstride = n_head * 2 * head_dim;
    const int pstride = 2 + head_dim;
    const int n_wg = n_rows * n_real * n_head_kv * n_splits;
    q.parallel_for(nd_range<1>((size_t) n_wg * 32, 32), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
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
        float * part0 = partials +
                        (((size_t) r * info->tpb + t) * n_head + h0) * n_splits * pstride +
                        (size_t) s * pstride;

        if (r >= info->n_rows || t >= info->n_real || !info->active[r]) {
#pragma unroll
            for (int j = 0; j < HPG; j++) {
                float * part = part0 + (size_t) j * n_splits * pstride;
                if (lane == 0) { part[0] = -INFINITY; part[1] = 0.f; }
#pragma unroll
                for (int i = 0; i < HD / 32; i++) part[2 + lane + 32 * i] = 0.f;
            }
            return;
        }
        const int pos = info->pos[r] + t;
        const int row = r * info->tpb + t;
        const int n_kv = pos + 1;
        const int chunk = (n_kv + n_splits - 1) / n_splits;
        const int t0 = s * chunk;
        const int t1 = sycl::min(t0 + chunk, n_kv);
        const int32_t * table = tables + (size_t) info->slot[r] * max_blocks;

        // same lane->dim mapping as the classic vectorized path: two coalesced
        // float4 chunks per lane, [4*lane, 4*lane+4) and + HD/2
        const int d0 = lane * 4;
        const int d1 = d0 + HD / 2;
        sycl::float4 qa[HPG], qb[HPG], aa[HPG], ab[HPG];
        float m[HPG], l[HPG];
#pragma unroll
        for (int j = 0; j < HPG; j++) {
            const float * qh = qbuf + (size_t) row * qstride + (size_t) (h0 + j) * 2 * head_dim;
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
            const KV * kp = kp_base + (((size_t) kb * n_head_kv + kvh) * kBlockSize + ko) * head_dim;
            const KV * vp = vp_base + (((size_t) kb * n_head_kv + kvh) * kBlockSize + ko) * head_dim;
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
            float * part = part0 + (size_t) j * n_splits * pstride;
            if (lane == 0) { part[0] = m[j]; part[1] = l[j]; }
            *reinterpret_cast<sycl::float4 *>(part + 2 + d0) = aa[j];
            *reinterpret_cast<sycl::float4 *>(part + 2 + d1) = ab[j];
        }
    });
}

// PF_DEC_GROUP=1 enables the grouped kernel; it is opt-in because on the Iris
// Xe the classic kernel is measurably faster (it has 4x the warps: the K/V
// redundancy is served by L2 and the extra parallelism wins).  The grouped path
// needs a non-fused call (out == nullptr) and n_head = 4 * n_head_kv with the
// 256-wide head.
static inline int dec_group_env() {
    static const int v = [] {
        const char * e = getenv("PF_DEC_GROUP");
        return (e && atoi(e) != 0) ? 1 : 0;
    }();
    return v;
}

void attn_launch(queue & q, const float * qbuf, const float * gate,
                 const void * kpool, const void * vpool, float * partials,
                 const int32_t * tables,
                 int n_head, int n_head_kv, int head_dim,
                 int n_splits, const step_info * info, float scale, int max_blocks, int n_rows,
                 int n_real, float * out, int group, const void * kscales, const void * vscales) {
    constexpr int HD = 256;
    const bool avec = attn_vec_env();
    const int qstride = n_head * 2 * head_dim;
    const int pstride = 2 + head_dim;
    const bool fuse = out != nullptr && n_splits == 1;
    const int n_wg = n_rows * n_real * n_head * n_splits;
    const int grp = group >= 0 ? group : dec_group_env();
    // the grouped kernel has no int8 path (it reuses kv_ld4 on a full row)
    if (grp && kv_dtype() != kv_dtype_t::i8 && !fuse && head_dim == HD && n_head_kv > 0 &&
        n_head % n_head_kv == 0 && n_head / n_head_kv == 4) {
        switch (kv_dtype()) {
            case kv_dtype_t::f32:
                attn_group_kernel<4>(q, qbuf, (const float *) kpool, (const float *) vpool,
                                     partials, tables, n_head, n_head_kv, head_dim, n_splits,
                                     info, scale, max_blocks, n_rows, n_real);
                return;
            case kv_dtype_t::bf16:
                attn_group_kernel<4>(q, qbuf, (const sycl::ext::oneapi::bfloat16 *) kpool,
                                     (const sycl::ext::oneapi::bfloat16 *) vpool, partials,
                                     tables, n_head, n_head_kv, head_dim, n_splits, info,
                                     scale, max_blocks, n_rows, n_real);
                return;
            case kv_dtype_t::f16:
                attn_group_kernel<4>(q, qbuf, (const sycl::half *) kpool,
                                     (const sycl::half *) vpool, partials, tables, n_head,
                                     n_head_kv, head_dim, n_splits, info, scale, max_blocks,
                                     n_rows, n_real);
                return;
            case kv_dtype_t::i8:
                break; // int8 has no grouped path (excluded by the guard above)
        }
    }
    auto launch = [&](const auto * kp_base, const auto * vp_base, const sycl::half * ksc_base,
                      const sycl::half * vsc_base) {
        using KV = std::remove_cv_t<std::remove_pointer_t<decltype(kp_base)>>;
        constexpr bool I8 = std::is_same_v<KV, int8_t>;
        q.parallel_for(nd_range<1>((size_t) n_wg * 32, 32), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int gid = it.get_group(0);
        const int r = gid / (n_real * n_head * n_splits);
        const int rem0 = gid % (n_real * n_head * n_splits);
        const int t = rem0 / (n_head * n_splits);
        const int rem = rem0 % (n_head * n_splits);
        const int h = rem / n_splits;
        const int s = rem % n_splits;
        const int lane = it.get_local_id(0);
        const sub_group sgg = it.get_sub_group();
        float * part = partials + (((size_t) r * info->tpb + t) * n_head + h) * n_splits * pstride + (size_t) s * pstride;

        if (r >= info->n_rows || t >= info->n_real || !info->active[r]) {
            if (lane == 0) { part[0] = -INFINITY; part[1] = 0.f; }
#pragma unroll
            for (int i = 0; i < HD / 32; i++) part[2 + lane + 32 * i] = 0.f;
            return;
        }
        const int pos = info->pos[r] + t;
        const int row = r * info->tpb + t;
        const int n_kv = pos + 1;
        const int chunk = (n_kv + n_splits - 1) / n_splits;
        const int t0 = s * chunk;
        const int t1 = sycl::min(t0 + chunk, n_kv);

        const float * q = qbuf + (size_t) row * qstride + (size_t) h * 2 * head_dim;
        const int kvh = (h * n_head_kv) / n_head;
        const int32_t * table = tables + (size_t) info->slot[r] * max_blocks;

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
                const size_t unit = (size_t) kb * n_head_kv + kvh;
                const auto * kp = kv_row_data(kp_base, unit, ko, head_dim);
                const auto * vp = kv_row_data(vp_base, unit, ko, head_dim);
                const sycl::float4 ka = kv_ld4(kp + d0);
                const sycl::float4 kb4 = kv_ld4(kp + d1);
                float dot;
                if constexpr (I8) {
                    // the two 4-dim chunks of a lane sit in blocks lane/8 and
                    // 4 + lane/8; the block scale multiplies the chunk's dot4
                    const auto * ks = kv_row_scales(ksc_base, unit, ko, head_dim);
                    const float sA = (float) ks[lane / 8];
                    const float sB = (float) ks[4 + lane / 8];
                    dot = dot4(qa, ka) * sA + dot4(qb, kb4) * sB;
                } else {
                    dot = dot4(qa, ka) + dot4(qb, kb4);
                }
                dot = sg_sum(dot, sgg) * scale;
                const float mnew = sycl::max(m, dot);
                const float e = sycl::exp(dot - mnew);
                const float corr = sycl::exp(m - mnew);
                l = l * corr + e;
                const sycl::float4 va = kv_ld4(vp + d0);
                const sycl::float4 vb = kv_ld4(vp + d1);
                if constexpr (I8) {
                    const auto * vs = kv_row_scales(vsc_base, unit, ko, head_dim);
                    aa = fma4(va, e * (float) vs[lane / 8], mul4(aa, corr));
                    ab = fma4(vb, e * (float) vs[4 + lane / 8], mul4(ab, corr));
                } else {
                    aa = fma4(va, e, mul4(aa, corr));
                    ab = fma4(vb, e, mul4(ab, corr));
                }
                m = mnew;
            }
            if (fuse) {
                // n_splits == 1: no partial buffer round-trip and no combine
                // kernel - every lane already holds the full softmax sum
                const float * gb = gate + (size_t) row * qstride + h * 2 * head_dim + head_dim;
                float * ob = out + (size_t) row * n_head * head_dim + h * head_dim;
                sycl::float4 oa, obb;
                for (int j = 0; j < 4; j++) {
                    oa[j] = (l > 0.f ? aa[j] / l : 0.f) * sigmoid_f(gb[d0 + j]);
                    obb[j] = (l > 0.f ? ab[j] / l : 0.f) * sigmoid_f(gb[d1 + j]);
                }
                *reinterpret_cast<sycl::float4 *>(ob + d0) = oa;
                *reinterpret_cast<sycl::float4 *>(ob + d1) = obb;
                return;
            }
            if (lane == 0) { part[0] = m; part[1] = l; }
            *reinterpret_cast<sycl::float4 *>(part + 2 + d0) = aa;
            *reinterpret_cast<sycl::float4 *>(part + 2 + d1) = ab;
            return;
        }

        float acc[HD / 32];
#pragma unroll
        for (int i = 0; i < HD / 32; i++) acc[i] = 0.f;
        float m = -INFINITY, l = 0.f;

        for (int kk = t0; kk < t1; kk++) {
            const int kb = table[kk / kBlockSize];
            const int ko = kk % kBlockSize;
            const size_t unit = (size_t) kb * n_head_kv + kvh;
            const auto * kp = kv_row_data(kp_base, unit, ko, head_dim);
            const auto * vp = kv_row_data(vp_base, unit, ko, head_dim);
            float dot = 0.f;
            if constexpr (I8) {
                // dims lane+32*i are exactly block i: one scale per iteration
                const auto * ks = kv_row_scales(ksc_base, unit, ko, head_dim);
#pragma unroll
                for (int i = 0; i < HD / 32; i++)
                    dot += q[lane + 32 * i] * kv_ld(kp + lane + 32 * i) * (float) ks[i];
            } else {
#pragma unroll
                for (int i = 0; i < HD / 32; i++) dot += q[lane + 32 * i] * kv_ld(kp + lane + 32 * i);
            }
            dot = sg_sum(dot, sgg) * scale;
            const float mnew = sycl::max(m, dot);
            const float e = sycl::exp(dot - mnew);
            const float corr = sycl::exp(m - mnew);
            l = l * corr + e;
            if constexpr (I8) {
                const auto * vs = kv_row_scales(vsc_base, unit, ko, head_dim);
#pragma unroll
                for (int i = 0; i < HD / 32; i++)
                    acc[i] = acc[i] * corr + e * kv_ld(vp + lane + 32 * i) * (float) vs[i];
            } else {
#pragma unroll
                for (int i = 0; i < HD / 32; i++)
                    acc[i] = acc[i] * corr + e * kv_ld(vp + lane + 32 * i);
            }
            m = mnew;
        }

        if (fuse) {
            const float * gb = gate + (size_t) row * qstride + h * 2 * head_dim + head_dim;
            float * ob = out + (size_t) row * n_head * head_dim + h * head_dim;
#pragma unroll
            for (int i = 0; i < HD / 32; i++)
                ob[lane + 32 * i] =
                    (l > 0.f ? acc[i] / l : 0.f) * sigmoid_f(gb[lane + 32 * i]);
            return;
        }
        if (lane == 0) { part[0] = m; part[1] = l; }
#pragma unroll
        for (int i = 0; i < HD / 32; i++) part[2 + lane + 32 * i] = acc[i];
        });
    };
    const sycl::half * ksc = (const sycl::half *) kscales;
    const sycl::half * vsc = (const sycl::half *) vscales;
    switch (kv_dtype()) {
        case kv_dtype_t::f32:  launch((const float *) kpool, (const float *) vpool, nullptr, nullptr); break;
        case kv_dtype_t::bf16: launch((const sycl::ext::oneapi::bfloat16 *) kpool,
                                      (const sycl::ext::oneapi::bfloat16 *) vpool, nullptr, nullptr); break;
        case kv_dtype_t::f16:  launch((const sycl::half *) kpool, (const sycl::half *) vpool, nullptr, nullptr); break;
        case kv_dtype_t::i8:   launch((const int8_t *) kpool, (const int8_t *) vpool, ksc, vsc); break;
    }
}

void attn_combine_launch(queue & q, const float * partials, const float * gate,
                         float * out, const step_info * info, int n_head, int head_dim,
                         int n_splits, int n_rows, int n_real) {
    const int pstride = 2 + head_dim;
    const int qstride = n_head * 2 * head_dim;
    q.parallel_for(nd_range<1>((size_t) n_rows * n_real * n_head * head_dim, head_dim), [=](nd_item<1> it) {
        const int gid = it.get_group(0);
        const int r = gid / (n_real * n_head);
        const int t = (gid / n_head) % n_real;
        const int h = gid % n_head;
        const int d = it.get_local_id(0);
        if (r >= info->n_rows || t >= info->n_real || !info->active[r]) return;
        const int row = r * info->tpb + t;
        const float * part = partials + (((size_t) r * info->tpb + t) * n_head + h) * n_splits * pstride;
        float M = -INFINITY;
        for (int s = 0; s < n_splits; s++) M = sycl::max(M, part[s * pstride]);
        float sum = 0.f, acc = 0.f;
        for (int s = 0; s < n_splits; s++) {
            const float w = sycl::exp(part[s * pstride] - M);
            sum += part[s * pstride + 1] * w;
            acc += part[s * pstride + 2 + d] * w;
        }
        const float gate_v = sigmoid_f(gate[(size_t) row * qstride + h * 2 * head_dim + head_dim + d]);
        out[(size_t) row * n_head * head_dim + h * head_dim + d] = (sum > 0.f ? acc / sum : 0.f) * gate_v;
    });
}

} // namespace si
