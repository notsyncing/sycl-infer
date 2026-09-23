#include "kernels.h"
#include "kernel_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

using namespace sycl;
using namespace si::kd;

// ---------------------------------------------------------------------------
void qk_norm_rope_launch(queue & q, float * qbuf, float * kbuf, float * vbuf, const float * q_norm,
                         const float * k_norm, void * kpool, void * vpool, const int32_t * tables,
                         const step_info * info, int n_head, int n_head_kv, int head_dim, int n_rot, float rope_base,
                         float eps, int max_blocks, int n_rows, int n_real, const void * kscales,
                         const void * vscales) {
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
                        const float ang = rpos * sycl::exp2(-2.0f * lane / n_rot * sycl::log2(rope_base));
                        const float c = sycl::cos(ang), s = sycl::sin(ang);
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
                        const float ang = rpos * sycl::exp2(-2.0f * lane / n_rot * sycl::log2(rope_base));
                        const float c = sycl::cos(ang), s = sycl::sin(ang);
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

} // namespace si
