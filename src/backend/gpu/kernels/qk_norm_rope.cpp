#include "kernels.h"
#include "env.h"
#include "common/rope.h"
#include "kernel_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

using namespace sycl;
using namespace si::kd;

// ---------------------------------------------------------------------------
// PF_KVSTORE_SUM: per-call device-side checksum of the K/V this call is ABOUT TO
// store - i.e. of `kbuf`/`vbuf` after the in-place norm and RoPE, which are the
// exact values the store quantizes.
//
// Same rule and the same reason as PF_ACTQ_SUM / PF_XMX_SUM: a host read-back inside
// the forward is what made the per-layer fingerprint oracle useless, so this
// accumulates on the device and is read once at process exit.  It is folded into the
// EXISTING kernel rather than launched separately, so it adds no launch and no
// ordering edge - both of which this investigation has shown can move the answer.
//
// It exists because the store itself was read and found sound (sg_max over the 32
// contiguous dims each lane holds, every lane gets the same scale, lane 0 writes it,
// reqd_sub_group_size(32) present), and because PF_XMX_SUM showed the pool's bytes
// differ while the block-max reduction over them does not.  So either the store's
// input already differs - which this measures - or the store is not deterministic,
// which it should not be.
struct kv_sum_holder {
    static constexpr int kSlots = 8192;
    sycl::queue q;
    int64_t * sum = nullptr;
    int n = 0;
    // PF_KVSTORE_EXTENTS: what each slot's call was, so the slot index is decodable
    // without guessing.  An earlier reading of these results assumed slots 0-19 were
    // the startup decode-graph recording; that was never checked, and this is how to
    // check it.
    std::vector<int> ex_nrows, ex_nreal, ex_pos;
    ~kv_sum_holder() {
        if (!sum || n == 0) {
            return;
        }
        try {
            std::vector<int64_t> h((size_t)n * 3);
            q.memcpy(h.data(), sum, (size_t)n * 24).wait();
            uint64_t hh = 1469598103934665603ULL;
            for (int i = 0; i < n * 3; i++) {
                hh = (hh ^ (uint64_t)h[i]) * 1099511628211ULL;
            }
            fprintf(stderr, "[kvsum] calls=%d checksum=%016llx\n", n, (unsigned long long)hh);
            for (int i = 0; i < n; i++) {
                fprintf(stderr, "[kvsumslot] %d %lld %lld nrows=%d nreal=%d pos=%d\n", i, (long long)h[3 * i],
                        (long long)h[3 * i + 1], (int)ex_nrows[i], (int)ex_nreal[i], (int)h[3 * i + 2]);
            }
        } catch (sycl::exception & e) {
            fprintf(stderr, "[kvsum] readback failed: %s\n", e.what());
        }
    }
    int slot(queue & qq) {
        if (!si::env::flag("PF_KVSTORE_SUM")) {
            return -1;
        }
        if (!sum) {
            q = qq;
            sum = sycl::malloc_device<int64_t>(kSlots * 3, q);
            if (!sum) {
                return -1;
            }
            // NO .wait() here: the multi-device decode command graph is RECORDED at
            // startup and REPLAYED afterwards, and a host wait on an event
            // belonging to a recorded graph throws ("wait method cannot be used
            // for an event associated with a command graph").  On an in-order queue
            // the memset is ordered before every kernel submitted after it, which
            // is all this needs.
            q.memset(sum, 0, (size_t)kSlots * 24);
        }
        return n < kSlots ? n++ : -1;
    }
    int slot_ex(queue & qq, int n_rows, int n_real) {
        const int i = slot(qq);
        if (i < 0) {
            return -1;
        }
        ex_nrows.push_back(n_rows);
        ex_nreal.push_back(n_real);
        return i;
    }
};
static kv_sum_holder g_kv_sum;

void qk_norm_rope_launch(queue & q, float * qbuf, float * kbuf, float * vbuf, const float * q_norm,
                         const float * k_norm, void * kpool, void * vpool, const int32_t * tables,
                         const step_info * info, int n_head, int n_head_kv, int head_dim, int n_rot, float rope_base,
                         const float * rope_freqs, float rope_mscale, float eps, int max_blocks, int n_rows,
                         int n_real, const void * kscales, const void * vscales, int sum_slot, int64_t * sum_kv) {
    const int qstride = n_head * 2 * head_dim;
    const int n_sg = n_head + 2 * n_head_kv; // Q + K + V groups
    auto launch = [&](auto * kdst_base, auto * vdst_base, const sycl::half * ksc_base, const sycl::half * vsc_base) {
        using KVT = std::remove_cv_t<std::remove_pointer_t<decltype(kdst_base)>>;
        using VVT = std::remove_cv_t<std::remove_pointer_t<decltype(vdst_base)>>;
        constexpr bool I8 = std::is_same_v<KVT, int8_t>;
        constexpr bool I4 = std::is_same_v<KVT, uint8_t>;
        constexpr bool V8 = std::is_same_v<VVT, int8_t>;
        constexpr bool V4 = std::is_same_v<VVT, uint8_t>;
        q.parallel_for(
            nd_range<1>((size_t)n_rows * n_real * n_sg * 32, n_sg * 32),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int gid = it.get_group(0);
                const int r = gid / n_real;
                const int t = gid % n_real;
                if (r >= info->n_rows || t >= row_nr(info, r) || !info->active[r]) {
                    return;
                }
                const int tid = it.get_local_id(0);
                const int sg = tid / 32;
                const int lane = tid % 32;
                const int pos = info->pos[r] + t;
                const int row = r * info->tpb + t;
                const int32_t * table = tables + (size_t)info->slot[r] * max_blocks;
                const sub_group sgg = it.get_sub_group();
                // PF_KVSTORE_SUM, pre side: hash the K/V as they arrive, i.e. the
                // qkv GEMM's fp32 output, BEFORE this kernel's in-place norm and RoPE
                // touch them.  Taken here rather than in a launch before the kernel so
                // that no extra submission and no new ordering edge is introduced -
                // this investigation has shown both can move the answer.
                //
                // pre and post live in the same slot pair, so one measurement splits
                // "the qkv GEMM's epilogue" from "the norm + RoPE": if `pre` already
                // differs the epilogue is at fault, if only `post` does then the
                // in-place norm + RoPE is.
                // The slot's chunk, recorded by the DEVICE: `info->pos` lives in USM
                // so the host cannot read it (and a host read would be a wait inside a
                // graph recording).  gid == 0 is row 0 / token 0, whose position is the
                // chunk's first position, and n_real tells the chunk width, so
                // `pos / (n_real * 32)` is the chunk index.
                if (sum_kv && gid == 0 && tid == 0) {
                    sum_kv[3 * sum_slot + 2] = (int64_t)info->pos[0];
                }
                if (sum_kv && sg >= n_head) {
                    const float * src = (sg < n_head + n_head_kv)
                                             ? (kbuf + (size_t)row * (n_head_kv * head_dim) +
                                                (size_t)(sg - n_head) * head_dim)
                                             : (vbuf + (size_t)row * (n_head_kv * head_dim) +
                                                (size_t)(sg - n_head - n_head_kv) * head_dim);
                    int64_t pre = 0;
#pragma unroll
                    for (int i = 0; i < head_dim / 32; i++) {
                        pre = pre * 1000003 + (int64_t)*(const uint32_t *)&src[lane + 32 * i];
                    }
                    const int64_t psub = sycl::reduce_over_group(sgg, pre, sycl::plus<int64_t>());
                    if (lane == 0) {
                        sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                          sycl::access::address_space::global_space>
                            apre(sum_kv[3 * sum_slot]);
                        apre.fetch_add(psub);
                    }
                }
                // M-RoPE: with an image in the prompt the rotation position is no longer
                // `pos` for every dimension pair.  Qwen3.5 uses interleaved M-RoPE: the
                // pair index selects the temporal/row/col position by an interleaved
                // 0,1,2 pattern (the frequency exponent stays the global pair index).
                float rpos = (float)pos;
                if (info->mrope_on) {
                    const int ridx = r * kMaxT + t;
                    const int n = kMaxB * kMaxT;
                    const int s0 = info->mrope_sections[0];
                    const int s1 = info->mrope_sections[1];
                    const int s2 = info->mrope_sections[2];
                    const int sect_dims = s0 + s1 + s2 + info->mrope_sections[3];
                    const int sector = sect_dims > 0 ? lane % sect_dims : lane;
                    int sec = 3;
                    if (sector % 3 == 1 && sector < 3 * s1) {
                        sec = 1;
                    } else if (sector % 3 == 2 && sector < 3 * s2) {
                        sec = 2;
                    } else if (sector % 3 == 0 && sector < 3 * s0) {
                        sec = 0;
                    }
                    rpos = (float)info->mrope[sec * n + ridx];
                }

                if (sg < n_head) {
                    float * qh = qbuf + (size_t)row * qstride + (size_t)sg * 2 * head_dim;
                    float ss = 0.f;
#pragma unroll
                    for (int i = 0; i < head_dim / 32; i++) {
                        float v = qh[lane + 32 * i];
                        ss += v * v;
                    }
                    const float inv = 1.0f / sycl::sqrt(sg_sum(ss, sgg) / head_dim + eps);
#pragma unroll
                    for (int i = 0; i < head_dim / 32; i++) {
                        qh[lane + 32 * i] *= inv * q_norm[lane + 32 * i];
                    }
                    if (lane < n_rot / 2) {
                        // YaRN: a non-null table carries the per-pair frequency
                        // already stretched by 1/factor on the low-frequency pairs
                        // (src/common/yarn.h).  A null table keeps the original
                        // expression bit-for-bit, which is every default-off run.
                        const float ang = rope_freqs ? rpos * rope_freqs[lane]
                                                     : rope_theta(rpos, lane, n_rot, sycl::log2(rope_base));
                        const float c = sycl::cos(ang) * rope_mscale, s = sycl::sin(ang) * rope_mscale;
                        const float x0 = qh[lane], x1 = qh[lane + n_rot / 2];
                        qh[lane] = x0 * c - x1 * s;
                        qh[lane + n_rot / 2] = x0 * s + x1 * c;
                    }
                } else if (sg < n_head + n_head_kv) {
                    const int kh = sg - n_head;
                    float * khp = kbuf + (size_t)row * (n_head_kv * head_dim) + (size_t)kh * head_dim;
                    float ss = 0.f;
#pragma unroll
                    for (int i = 0; i < head_dim / 32; i++) {
                        float v = khp[lane + 32 * i];
                        ss += v * v;
                    }
                    const float inv = 1.0f / sycl::sqrt(sg_sum(ss, sgg) / head_dim + eps);
#pragma unroll
                    for (int i = 0; i < head_dim / 32; i++) {
                        khp[lane + 32 * i] *= inv * k_norm[lane + 32 * i];
                    }
                    if (lane < n_rot / 2) {
                        const float ang = rope_freqs ? rpos * rope_freqs[lane]
                                                     : rope_theta(rpos, lane, n_rot, sycl::log2(rope_base));
                        const float c = sycl::cos(ang) * rope_mscale, s = sycl::sin(ang) * rope_mscale;
                        const float x0 = khp[lane], x1 = khp[lane + n_rot / 2];
                        khp[lane] = x0 * c - x1 * s;
                        khp[lane + n_rot / 2] = x0 * s + x1 * c;
                    }
                    const int kb = table[pos / kBlockSize];
                    if constexpr (I8) {
                        // quantize block i = the 32 dims held by the warp for lane l:
                        // dims l+32*i; one fp16 scale per block, written by lane 0
                        const int ko = pos % kBlockSize;
                        const size_t unit = (size_t)kb * n_head_kv + kh;
                        auto * krow = kdst_base + (unit * kBlockSize + ko) * head_dim;
                        auto * ksc = (sycl::half *)kv_row_scales(ksc_base, unit, ko, head_dim);
#pragma unroll
                        for (int i = 0; i < head_dim / 32; i++) {
                            const float v = khp[lane + 32 * i];
                            const float m = sg_max(sycl::fabs(v), sgg);
                            const float sc = m > 0.f ? m / 127.f : 1.f;
                            krow[lane + 32 * i] = i8_quant(v, sc);
                            if (lane == 0) {
                                ksc[i] = (sycl::half)sc;
                            }
                        }
                    } else if constexpr (I4) {
                        // two nibbles per byte: even lane l packs dims l (low) and
                        // l+1 (high) into byte (i*32+l)/2
                        const int ko = pos % kBlockSize;
                        const size_t unit = (size_t)kb * n_head_kv + kh;
                        auto * krow = kdst_base + (unit * kBlockSize + ko) * (size_t)(head_dim / 2);
                        auto * ksc = (sycl::half *)kv_row_scales(ksc_base, unit, ko, head_dim);
#pragma unroll
                        for (int i = 0; i < head_dim / 32; i++) {
                            const float v = khp[lane + 32 * i];
                            const float m = sg_max(sycl::fabs(v), sgg);
                            const float sc = m > 0.f ? m / 7.f : 1.f;
                            const uint32_t nib = (uint32_t)((uint8_t)i4_quant(v, sc) & 0xF);
                            const uint32_t nb = sycl::permute_group_by_xor(sgg, nib, 1);
                            if ((lane & 1) == 0) {
                                krow[(i * 32 + lane) / 2] = (uint8_t)(nib | (nb << 4));
                            }
                            if (lane == 0) {
                                ksc[i] = (sycl::half)sc;
                            }
                        }
                    } else {
                        auto * kdst =
                            kdst_base + (((size_t)kb * n_head_kv + kh) * kBlockSize + (pos % kBlockSize)) * head_dim;
#pragma unroll
                        for (int i = 0; i < head_dim / 32; i++) {
                            kv_st(kdst + lane + 32 * i, khp[lane + 32 * i]);
                        }
                    }
                } else if (sg < n_head + 2 * n_head_kv) {
                    const int vh = sg - n_head - n_head_kv;
                    const float * vhp = vbuf + (size_t)row * (n_head_kv * head_dim) + (size_t)vh * head_dim;
                    const int vb = table[pos / kBlockSize];
                    if constexpr (V8) {
                        const int ko = pos % kBlockSize;
                        const size_t unit = (size_t)vb * n_head_kv + vh;
                        auto * vrow = vdst_base + (unit * kBlockSize + ko) * head_dim;
                        auto * vsc = (sycl::half *)kv_row_scales(vsc_base, unit, ko, head_dim);
#pragma unroll
                        for (int i = 0; i < head_dim / 32; i++) {
                            const float v = vhp[lane + 32 * i];
                            const float m = sg_max(sycl::fabs(v), sgg);
                            const float sc = m > 0.f ? m / 127.f : 1.f;
                            vrow[lane + 32 * i] = i8_quant(v, sc);
                            if (lane == 0) {
                                vsc[i] = (sycl::half)sc;
                            }
                        }
                    } else if constexpr (V4) {
                        const int ko = pos % kBlockSize;
                        const size_t unit = (size_t)vb * n_head_kv + vh;
                        auto * vrow = vdst_base + (unit * kBlockSize + ko) * (size_t)(head_dim / 2);
                        auto * vsc = (sycl::half *)kv_row_scales(vsc_base, unit, ko, head_dim);
#pragma unroll
                        for (int i = 0; i < head_dim / 32; i++) {
                            const float v = vhp[lane + 32 * i];
                            const float m = sg_max(sycl::fabs(v), sgg);
                            const float sc = m > 0.f ? m / 7.f : 1.f;
                            const uint32_t nib = (uint32_t)((uint8_t)i4_quant(v, sc) & 0xF);
                            const uint32_t nb = sycl::permute_group_by_xor(sgg, nib, 1);
                            if ((lane & 1) == 0) {
                                vrow[(i * 32 + lane) / 2] = (uint8_t)(nib | (nb << 4));
                            }
                            if (lane == 0) {
                                vsc[i] = (sycl::half)sc;
                            }
                        }
                    } else {
                        auto * vdst =
                            vdst_base + (((size_t)vb * n_head_kv + vh) * kBlockSize + (pos % kBlockSize)) * head_dim;
#pragma unroll
                        for (int i = 0; i < head_dim / 32; i++) {
                            kv_st(vdst + lane + 32 * i, vhp[lane + 32 * i]);
                        }
                    }
                }
                // PF_KVSTORE_SUM: fold the post-norm, post-RoPE K and V values this
                // work-group holds.  One sub-group reduction and one atomic per
                // sub-group, all inside the existing kernel - no extra launch, so no
                // new ordering edge to confuse the thing being measured.
                if (sum_kv) {
                    int64_t acc = 0;
                    if (sg >= n_head) {
                        const float * src = (sg < n_head + n_head_kv)
                                                 ? (kbuf + (size_t)row * (n_head_kv * head_dim) +
                                                    (size_t)(sg - n_head) * head_dim)
                                                 : (vbuf + (size_t)row * (n_head_kv * head_dim) +
                                                    (size_t)(sg - n_head - n_head_kv) * head_dim);
#pragma unroll
                        for (int i = 0; i < head_dim / 32; i++) {
                            acc = acc * 1000003 + (int64_t)*(const uint32_t *)&src[lane + 32 * i];
                        }
                    }
                    const int64_t sub =
                        sycl::reduce_over_group(sgg, acc, sycl::plus<int64_t>());
                    if (lane == 0) {
                        sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                          sycl::access::address_space::global_space>
                            a(sum_kv[3 * sum_slot + 1]);
                        a.fetch_add(sub);
                    }
                }
            });
    };
    const sycl::half * ksc = (const sycl::half *)kscales;
    const sycl::half * vsc = (const sycl::half *)vscales;
    using bf16 = sycl::ext::oneapi::bfloat16;
    auto with_k = [&](auto * kp, const sycl::half * ksc_p) {
        switch (kv_v_dtype()) {
        case kv_dtype_t::f32: launch(kp, (float *)vpool, ksc_p, nullptr); break;
        case kv_dtype_t::bf16: launch(kp, (bf16 *)vpool, ksc_p, nullptr); break;
        case kv_dtype_t::f16: launch(kp, (sycl::half *)vpool, ksc_p, nullptr); break;
        case kv_dtype_t::i8: launch(kp, (int8_t *)vpool, ksc_p, vsc); break;
        case kv_dtype_t::i4: launch(kp, (uint8_t *)vpool, ksc_p, vsc); break;
        }
    };
    switch (kv_k_dtype()) {
    case kv_dtype_t::f32: with_k((float *)kpool, nullptr); break;
    case kv_dtype_t::bf16: with_k((bf16 *)kpool, nullptr); break;
    case kv_dtype_t::f16: with_k((sycl::half *)kpool, nullptr); break;
    case kv_dtype_t::i8: with_k((int8_t *)kpool, ksc); break;
    case kv_dtype_t::i4: with_k((uint8_t *)kpool, ksc); break;
    }
}

// PF_KVSTORE_SUM: the slot/buffer pair the backend hands in, kept out of the
// backend so the holder stays the single owner of the allocation.
int kv_store_sum_slot(sycl::queue & q, int n_rows, int n_real) {
    return g_kv_sum.slot_ex(q, n_rows, n_real);
}
int64_t * kv_store_sum_ptr(sycl::queue & q) {
    g_kv_sum.slot(q);
    return g_kv_sum.sum;
}

} // namespace si
