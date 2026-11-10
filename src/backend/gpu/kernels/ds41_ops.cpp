// DeepSeek-V4.1-Flash device ops for the window KV round, the shared-expert /
// routed-expert clamp, the sqrt-softplus router, the DSA selector, the
// candidate-block stage, the compressed-row gather, the window ring write and
// the Engram apply.  The host oracles are the matching functions in
// src/engine/engine_ds41.cpp: fp8_round_row, Fwd::moe, compute_candidates_one,
// index_select_one's top-k, attention_one's gather/window write and
// apply_engram_rows.  See ds41_ops.h for the contracts.
#include "ds41_ops.h"

#include <climits>
#include <cmath>
#include <cstdint>

namespace si {

using namespace sycl;

namespace {

// ---------------------------------------------------------------- fp8 helpers
// 2^e from the float bits.  sycl::exp2 is not exact enough on this backend for
// a quantization scale (measured ~3.7% off at 2^-6 in the ds41_attn probe), so
// the scale is assembled exactly as the host's std::exp2 would be.
inline float pow2f(int e) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = (uint32_t)(e + 127) << 23;
    return v.f;
}

// ceil(log2(q)) for a normal positive float, exactly.  The host uses
// ceil(log2(amax / 448)); every power-of-two boundary must stay on the same
// side or the fp8 block scale differs by 2x.
inline int ceil_log2f(float q) {
    union {
        float f;
        uint32_t u;
    } b;
    b.f = q;
    const int E = (int)((b.u >> 23) & 0xffu) - 127;
    const bool exact_pow2 = (b.u & 0x007fffffu) == 0;
    return exact_pow2 ? E : E + 1;
}

// E4M3FN nearest representable value, round to nearest even.  Same code as
// ds41_attn.cpp / the host e4m3_val.
inline float e4m3_val(float v) {
    const bool neg = sycl::signbit(v);
    float a = sycl::fabs(v);
    if (!(a > 0.f)) {
        return neg ? -0.0f : 0.0f;
    }
    if (a >= 448.f) {
        return neg ? -448.f : 448.f;
    }
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = a;
    const int E = (int)((bits.u >> 23) & 0xffu) - 127;
    const float step = (E < -6) ? pow2f(-9) : pow2f(E - 3);
    const float q = a / step;
    float out = sycl::rint(q) * step;
    if (out > 448.f) {
        out = 448.f;
    }
    return neg ? -out : out;
}

// ---------------------------------------------------------------- misc helpers
inline float silu_f(float x) { return x / (1.f + sycl::exp(-x)); }

inline float bf16_to_f32(uint16_t v) {
    union {
        uint32_t u;
        float f;
    } b;
    b.u = (uint32_t)v << 16;
    return b.f;
}

// Monotonic float -> uint32 order key (signed floats, -0/+0 canonicalised by
// the caller).  Ties in the selector break to the lower index, which the rank
// key encodes in the low word.
inline uint32_t order_key(float v) {
    union {
        float f;
        uint32_t u;
    } b;
    b.f = v;
    return (b.u & 0x80000000u) ? ~b.u : (b.u | 0x80000000u);
}

inline uint64_t rank_key(float v, int idx) {
    return ((uint64_t)order_key(v) << 32) | (uint32_t)(0xffffffffu - (uint32_t)idx);
}

// Work-group size of the two radix-select stages (ds41_select_launch and the
// candidate-block cut).  The radix loop is written inline at both sites: a
// SYCL local_accessor cannot cross a normal function boundary.
constexpr int kSelWG = 256;

inline int next_pow2(int n) {
    int p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

} // namespace

// ---------------------------------------------------------------- fp8 round
// One work-item per `block`-wide group (the reference walks the row in blocks
// of 32).  The block scale is identical to the host's power-of-two E4M3 scale,
// so the rounded values are bit-exact.
void ds41_fp8_round_launch(queue & q, float * v, int n, int block) {
    if (!v || n <= 0 || block <= 0) {
        return;
    }
    const int nblk = (n + block - 1) / block;
    q.parallel_for(range<1>((size_t)nblk), [=](id<1> ib) {
        const int i0 = (int)ib[0] * block;
        const int m = (block < n - i0) ? block : (n - i0);
        float amax = 1e-4f;
        for (int i = 0; i < m; i++) {
            amax = sycl::fmax(amax, sycl::fabs(v[i0 + i]));
        }
        const float s = pow2f(ceil_log2f(amax / 448.f));
        for (int i = 0; i < m; i++) {
            float x = v[i0 + i] / s;
            x = sycl::fmax(-448.f, sycl::fmin(448.f, x));
            v[i0 + i] = e4m3_val(x) * s;
        }
    });
}

// ---------------------------------------------------------------- swiglu clamp
// out[i] = silu(min(g, lim)) * clamp(u, -lim, lim), one thread per element.
void ds41_swiglu_clamp_launch(queue & q, const float * g, const float * u, float * out, int n, float lim) {
    if (!g || !u || !out || n <= 0) {
        return;
    }
    q.parallel_for(range<1>((size_t)n), [=](id<1> i) {
        const float gg = sycl::fmin(g[i], lim); // std::min(g, lim)
        const float uu = sycl::fmax(-lim, sycl::fmin(lim, u[i]));
        out[i] = silu_f(gg) * uu;
    });
}

// ---------------------------------------------------------------- router
// Phase 1: logits = x * gate_inp^T on the raw BF16/F32 rows.
//
// Geometry: 4 warps per expert, 8 warps (256 threads) per work-group => 2
// experts per work-group and 32 lanes per warp over a contiguous K slice
// (split-K = 4).  A decode (n_expert 384) is 192 work-groups = 1536 warps, so
// the weight read is coalesced (one 64-byte sector per warp load) and the
// dependent FMA chain per lane is K/128 = 40 long.  Each warp reduces with the
// XOR tree; the four split partials are summed in ascending split order.
void ds41_router_launch(queue & q, const void * gate_inp, uint32_t type, const float * x, int n_tok,
                        int n_embd, const float * bias, const float * bias_vl, int use_vl, int n_expert,
                        int top_k, float scale, float * logits, int32_t * ids, float * wts) {
    if (!gate_inp || !x || !logits || !ids || !wts) {
        return;
    }
    if (n_tok <= 0 || n_embd <= 0 || n_expert <= 0 || top_k <= 0) {
        return;
    }
    const int K = n_embd;
    const int E = n_expert;
    const int kk = top_k < E ? top_k : E;

    constexpr int WG = 256;
    constexpr int kSplit = 4;              // warps per expert
    constexpr int kWarps = WG / 32;        // 8
    constexpr int kExprPerWg = kWarps / kSplit; // 2
    const int egroups = (E + kExprPerWg - 1) / kExprPerWg;

    // ---- phase 1: raw logits
    q.submit([&](handler & h) {
        local_accessor<float, 1> part((size_t)kWarps, h);
        h.parallel_for(nd_range<1>((size_t)n_tok * (size_t)egroups * WG, WG),
                       [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int tid = (int)it.get_local_id(0);
                           const int gid = (int)it.get_group(0);
                           const int tok = gid / egroups;
                           const int eg = gid % egroups;
                           const int warp = tid / 32;
                           const int lane = tid % 32;
                           const int ex = warp / kSplit; // expert within group
                           const int sp = warp % kSplit; // K split
                           const int e = eg * kExprPerWg + ex;
                           const int kb = (int)((long long)sp * K / kSplit);
                           const int ke = (int)((long long)(sp + 1) * K / kSplit);
                           const float * xr = x + (size_t)tok * (size_t)K;

                           float acc = 0.f;
                           if (e < E) {
                               const size_t base = (size_t)e * (size_t)K;
                               if (type == 30) {
                                   const uint16_t * W = (const uint16_t *)gate_inp;
                                   for (int i = kb + lane; i < ke; i += 32) {
                                       acc = sycl::fma(xr[i], bf16_to_f32(W[base + (size_t)i]), acc);
                                   }
                               } else if (type == 1) {
                                   const sycl::half * W = (const sycl::half *)gate_inp;
                                   for (int i = kb + lane; i < ke; i += 32) {
                                       acc = sycl::fma(xr[i], (float)W[base + (size_t)i], acc);
                                   }
                               } else {
                                   const float * W = (const float *)gate_inp;
                                   for (int i = kb + lane; i < ke; i += 32) {
                                       acc = sycl::fma(xr[i], W[base + (size_t)i], acc);
                                   }
                               }
                           }

                           const sub_group sg = it.get_sub_group();
                           for (int off = 16; off > 0; off >>= 1) {
                               acc += sycl::permute_group_by_xor(sg, acc, off);
                           }
                           if (lane == 0) {
                               part[warp] = acc;
                           }
                           it.barrier(sycl::access::fence_space::local_space);
                           if (lane == 0 && sp == 0 && e < E) {
                               float v = 0.f;
                               for (int s = 0; s < kSplit; s++) {
                                   v += part[ex * kSplit + s];
                               }
                               logits[(size_t)tok * (size_t)E + (size_t)e] = v;
                           }
                       });
    });

    // ---- phase 2: sqrt(softplus(logit)), then the host's bias-only selection
    // and `score / (sum + 1e-20) * scale` weights, in selection order.
    //
    // ONE warp per token and no work-group barrier: lane `l` owns the experts
    // `l, l+32, ...`, keeps their selection scores (and their used flag) in
    // local memory that no other lane touches, and each of the k rounds is a
    // warp XOR all-reduce of (value, lowest index).  A 256-thread work-group
    // with a radix scan here measured ~90 us on the A770 - the work-group
    // barriers dominate - while this warp version is several times cheaper.
    // The winner's score is recomputed
    // from `logits` for the weight sum, which is bit-identical to the value
    // selection used (same expression, same input).
    const float * bsel = (use_vl && bias_vl) ? bias_vl : bias;
    q.submit([&](handler & h) {
        local_accessor<float, 1> sv((size_t)E, h);
        local_accessor<uint8_t, 1> ls((size_t)E, h);
        h.parallel_for(nd_range<1>((size_t)n_tok * 32, 32),
                       [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int tok = (int)it.get_group(0);
                           const int lane = (int)it.get_local_id(0);
                           const sub_group sg = it.get_sub_group();
                           float * lp = logits + (size_t)tok * (size_t)E;
                           for (int e = lane; e < E; e += 32) {
                               const float l = lp[e];
                               const float sp = (l > 20.f) ? l : sycl::log1p(sycl::exp(l));
                               sv[e] = sycl::sqrt(sp) + (bsel ? bsel[e] : 0.f);
                               ls[e] = 0;
                           }
                           int32_t * idr = ids + (size_t)tok * (size_t)top_k;
                           for (int j = 0; j < kk; j++) {
                               float bv = -INFINITY;
                               int32_t bi = INT_MAX;
                               for (int e = lane; e < E; e += 32) {
                                   if (ls[e]) {
                                       continue;
                                   }
                                   const float v = sv[e];
                                   if (v > bv || (v == bv && e < bi)) {
                                       bv = v;
                                       bi = e;
                                   }
                               }
                               for (int off = 16; off > 0; off >>= 1) {
                                   const float ov = sycl::permute_group_by_xor(sg, bv, off);
                                   const int32_t oi = sycl::permute_group_by_xor(sg, bi, off);
                                   if (ov > bv || (ov == bv && oi < bi)) {
                                       bv = ov;
                                       bi = oi;
                                   }
                               }
                               // Every lane now holds the same winner.
                               if (bi >= lane && ((bi - lane) % 32) == 0) {
                                   ls[bi] = 1;
                               }
                               if (lane == 0) {
                                   idr[j] = bi;
                               }
                           }
                           if (lane == 0) {
                               float sum = 1e-20f;
                               for (int j = 0; j < kk; j++) {
                                   const float l = lp[idr[j]];
                                   const float sp = (l > 20.f) ? l : sycl::log1p(sycl::exp(l));
                                   sum += sycl::sqrt(sp);
                               }
                               float * wr = wts + (size_t)tok * (size_t)top_k;
                               for (int j = 0; j < kk; j++) {
                                   const float l = lp[idr[j]];
                                   const float sp = (l > 20.f) ? l : sycl::log1p(sycl::exp(l));
                                   wr[j] = sycl::sqrt(sp) / sum * scale;
                               }
                           }
                       });
    });
}

// ---------------------------------------------------------------- selector
// Top-k over scores[0..n): descending score, ties to the lowest index, and the
// result is emitted in ascending index order (the attention gather order).
// One 256-thread work-group: an 8-pass byte-radix selection over the 64-bit
// rank key (same scheme as qsa_topk_launch), then a bitonic sort of the k
// selected ids so the output is ascending.  The radix key is unique, so the
// cutoff is exact.
void ds41_select_launch(queue & q, const float * scores, int n, int k, int32_t * ids_ascending) {
    if (!scores || !ids_ascending || n <= 0 || k <= 0) {
        return;
    }
    if (k >= n) {
        q.parallel_for(range<1>((size_t)n), [=](id<1> i) { ids_ascending[i] = (int32_t)i; });
        return;
    }
    if (k > 8192) {
        return; // the local sort buffer below would exceed the SLM budget
    }
    const int cap = next_pow2(k);
    constexpr int WG = kSelWG;
    q.submit([&](handler & h) {
        local_accessor<uint32_t, 1> hist(256, h);
        local_accessor<uint64_t, 1> lprefix(1, h);
        local_accessor<int, 1> lrem(1, h);
        local_accessor<int, 1> lpos(1, h);
        local_accessor<int32_t, 1> sk((size_t)cap, h);
        h.parallel_for(nd_range<1>((size_t)WG, WG), [=](nd_item<1> it) {
            const int lane = (int)it.get_local_id(0);
            if (lane == 0) {
                lprefix[0] = 0;
                lrem[0] = k;
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int shift = 56; shift >= 0; shift -= 8) {
                for (int i = lane; i < 256; i += WG) {
                    hist[i] = 0;
                }
                it.barrier(sycl::access::fence_space::local_space);
                const uint64_t mask = shift == 56 ? 0ULL : (~0ULL << (shift + 8));
                const uint64_t pref = lprefix[0];
                for (int c = lane; c < n; c += WG) {
                    const float v = (scores[c] == 0.f) ? 0.f : scores[c];
                    const uint64_t key = rank_key(v, c);
                    if ((key & mask) != pref) {
                        continue;
                    }
                    const uint32_t b = (uint32_t)((key >> shift) & 0xff);
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        a(hist[b]);
                    a.fetch_add(1);
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (lane == 0) {
                    const int rem = lrem[0];
                    int acc = 0, b = -1;
                    for (int xb = 255; xb >= 0; --xb) {
                        acc += (int)hist[xb];
                        if (acc >= rem) {
                            b = xb;
                            break;
                        }
                    }
                    if (b >= 0) {
                        lprefix[0] |= (uint64_t)(uint32_t)b << shift;
                        lrem[0] = rem - (acc - (int)hist[b]);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            const uint64_t cutoff = lprefix[0];
            if (lane == 0) {
                lpos[0] = 0;
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int c = lane; c < n; c += WG) {
                const float v = (scores[c] == 0.f) ? 0.f : scores[c];
                const uint64_t key = rank_key(v, c);
                if (key < cutoff) {
                    continue;
                }
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>
                    a(lpos[0]);
                const int slot = a.fetch_add(1);
                if (slot < k) {
                    sk[slot] = (int32_t)c;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int i = lane; i < cap; i += WG) {
                if (i >= k) {
                    sk[i] = INT_MAX;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            // One compare-exchange pass over the whole cap; each lane owns a
            // strided subset of the disjoint partner pairs.  cap may exceed
            // the work-group size.
            for (int kc = 2; kc <= cap; kc <<= 1) {
                for (int j = kc >> 1; j > 0; j >>= 1) {
                    for (int i = lane; i < cap; i += WG) {
                        const int partner = i ^ j;
                        if (partner > i) {
                            const bool asc = ((i & kc) == 0);
                            const int32_t a = sk[i], b = sk[partner];
                            if (asc ? (a > b) : (a < b)) {
                                sk[i] = b;
                                sk[partner] = a;
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
            }
            for (int i = lane; i < k; i += WG) {
                ids_ascending[i] = sk[i];
            }
        });
    });
}

// ---------------------------------------------------------------- candidate blocks
// Per-block max over the compressed rows, +inf on the newest visible block,
// then the top `min(topk_blocks, nb)` blocks marked in `keep` (0/1 bytes) and
// `*nb` written.  One work-group: the block maxima live in SLM, the selection
// is the same byte-radix cut as the selector (only membership matters here, so
// the selected blocks do not need to be ordered).
void ds41_cand_blocks_launch(queue & q, const float * score, int n_vis, int block, int topk_blocks,
                             uint8_t * keep, int * nb) {
    const int nblk = (n_vis > 0 && block > 0) ? (n_vis + block - 1) / block : 0;
    if (nb) {
        // written below in the kernel when nblk > 0; for 0 there is no launch
        // work to do, so set it from a trivial kernel to keep it ordered.
        q.single_task([=]() { *nb = nblk; });
    }
    if (!score || !keep || n_vis <= 0 || block <= 0 || nblk <= 0) {
        return;
    }
    if (nblk > 16384) {
        return; // SLM budget for the per-block maxima
    }
    constexpr int WG = kSelWG;
    const int take = topk_blocks < nblk ? topk_blocks : nblk;
    q.submit([&](handler & h) {
        local_accessor<float, 1> bmax((size_t)nblk, h);
        local_accessor<uint8_t, 1> lk((size_t)nblk, h);
        local_accessor<uint32_t, 1> hist(256, h);
        local_accessor<uint64_t, 1> lprefix(1, h);
        local_accessor<int, 1> lrem(1, h);
        h.parallel_for(nd_range<1>((size_t)WG, WG), [=](nd_item<1> it) {
            const int tid = (int)it.get_local_id(0);
            for (int b = tid; b < nblk; b += WG) {
                const int hi = n_vis < (b + 1) * block ? n_vis : (b + 1) * block;
                float m = -INFINITY;
                for (int c = b * block; c < hi; c++) {
                    m = sycl::fmax(m, score[c]);
                }
                bmax[b] = m;
                lk[b] = 0;
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (tid == 0) {
                const int last = (n_vis - 1) / block;
                if (last >= 0 && last < nblk) {
                    bmax[last] = INFINITY;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);

            if (take <= 0 || take >= nblk) {
                for (int b = tid; b < nblk; b += WG) {
                    lk[b] = (bmax[b] > -INFINITY) ? 1 : 0;
                }
            } else {
                if (tid == 0) {
                    lprefix[0] = 0;
                    lrem[0] = take;
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int shift = 56; shift >= 0; shift -= 8) {
                    for (int i = tid; i < 256; i += WG) {
                        hist[i] = 0;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    const uint64_t mask = shift == 56 ? 0ULL : (~0ULL << (shift + 8));
                    const uint64_t pref = lprefix[0];
                    for (int b = tid; b < nblk; b += WG) {
                        const uint64_t key = rank_key(bmax[b], b);
                        if ((key & mask) != pref) {
                            continue;
                        }
                        const uint32_t bb = (uint32_t)((key >> shift) & 0xff);
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            a(hist[bb]);
                        a.fetch_add(1);
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if (tid == 0) {
                        const int rem = lrem[0];
                        int acc = 0, b = -1;
                        for (int xb = 255; xb >= 0; --xb) {
                            acc += (int)hist[xb];
                            if (acc >= rem) {
                                b = xb;
                                break;
                            }
                        }
                        if (b >= 0) {
                            lprefix[0] |= (uint64_t)(uint32_t)b << shift;
                            lrem[0] = rem - (acc - (int)hist[b]);
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                const uint64_t cutoff = lprefix[0];
                for (int b = tid; b < nblk; b += WG) {
                    const uint64_t key = rank_key(bmax[b], b);
                    lk[b] = (key >= cutoff && bmax[b] > -INFINITY) ? 1 : 0;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int b = tid; b < nblk; b += WG) {
                keep[b] = lk[b];
            }
        });
    });
}

// ---------------------------------------------------------------- candidate mask
// In place: anything outside a kept block becomes -inf.
void ds41_cand_mask_launch(queue & q, float * score, const uint8_t * keep, int n_vis, int block) {
    if (!score || !keep || n_vis <= 0 || block <= 0) {
        return;
    }
    q.parallel_for(range<1>((size_t)n_vis), [=](id<1> i) {
        const int b = (int)i[0] / block;
        if (!keep[b]) {
            score[i[0]] = -INFINITY;
        }
    });
}

// ---------------------------------------------------------------- gather
// out[i][0..d) = ckv[sel[i]][0..d).  A negative selector (the ref's "not
// visible" sentinel) zeroes the row so a gather never reads before ckv.
void ds41_gather_sel_launch(queue & q, const float * ckv, const int32_t * sel, int n_sel, int d, float * out) {
    if (!ckv || !sel || !out || n_sel <= 0 || d <= 0) {
        return;
    }
    q.parallel_for(range<2>((size_t)n_sel, (size_t)d), [=](id<2> idx) {
        const int i = (int)idx[0];
        const int j = (int)idx[1];
        const int32_t s = sel[i];
        out[(size_t)i * (size_t)d + (size_t)j] =
            (s >= 0) ? ckv[(size_t)s * (size_t)d + (size_t)j] : 0.f;
    });
}

// ---------------------------------------------------------------- window write
// Ring write: row -> win[pos % cap], and win_pos[pos % cap] = pos.  One thread
// per element; thread 0 also publishes the position.
void ds41_window_write_launch(queue & q, const float * row, float * win, int * win_pos, int pos, int cap,
                              int d) {
    if (!row || !win || !win_pos || cap <= 0 || d <= 0) {
        return;
    }
    int slot = pos % cap;
    if (slot < 0) {
        slot += cap;
    }
    q.parallel_for(range<1>((size_t)d), [=](id<1> i) {
        win[(size_t)slot * (size_t)d + (size_t)i[0]] = row[i[0]];
        if (i[0] == 0) {
            win_pos[slot] = pos;
        }
    });
}

// ---------------------------------------------------------------- Engram apply
// One work-group per hyper-connection stream.  Three reductions (|h|^2, |k|^2
// and sum h*gate*k) feed the host's `rstd`, `dot` and signed-sqrt gate, then
// every lane adds g*value to h.  Geometry: 256 threads, warp XOR reduction into
// 8 sub-group partials, then the sub-group totals summed in ascending order.
void ds41_engram_apply_launch(queue & q, float * h, const float * kv, const float * gate, int hc, int d) {
    if (!h || !kv || !gate || hc <= 0 || d <= 0) {
        return;
    }
    constexpr int WG = 256;
    q.submit([&](handler & hd) {
        local_accessor<float, 1> ph((size_t)(WG / 32), hd);
        local_accessor<float, 1> pk((size_t)(WG / 32), hd);
        local_accessor<float, 1> pd((size_t)(WG / 32), hd);
        local_accessor<float, 1> lg(1, hd);
        hd.parallel_for(nd_range<1>((size_t)hc * WG, WG),
                        [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                            const int k = (int)it.get_group(0);
                            const int tid = (int)it.get_local_id(0);
                            const int sub = tid / 32;
                            const int lane = tid % 32;
                            float * hk = h + (size_t)k * (size_t)d;
                            const float * kk = kv + (size_t)k * (size_t)d;
                            const float * wg = gate + (size_t)k * (size_t)d;
                            const float * value = kv + (size_t)hc * (size_t)d;

                            float ssh = 0.f, ssk = 0.f, sd = 0.f;
                            for (int i = tid; i < d; i += WG) {
                                const float a = hk[i];
                                const float b = kk[i];
                                ssh = sycl::fma(a, a, ssh);
                                ssk = sycl::fma(b, b, ssk);
                                sd = sycl::fma(a * wg[i], b, sd);
                            }
                            const sub_group sg = it.get_sub_group();
                            for (int off = 16; off > 0; off >>= 1) {
                                ssh += sycl::permute_group_by_xor(sg, ssh, off);
                                ssk += sycl::permute_group_by_xor(sg, ssk, off);
                                sd += sycl::permute_group_by_xor(sg, sd, off);
                            }
                            if (lane == 0) {
                                ph[sub] = ssh;
                                pk[sub] = ssk;
                                pd[sub] = sd;
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                            if (tid == 0) {
                                float th = 0.f, tk = 0.f, td = 0.f;
                                for (int s = 0; s < WG / 32; s++) {
                                    th += ph[s];
                                    tk += pk[s];
                                    td += pd[s];
                                }
                                const float rstd = (1.f / sycl::sqrt(th / (float)d + 1e-20f)) *
                                                   (1.f / sycl::sqrt(tk / (float)d + 1e-20f));
                                float dot = td * rstd / sycl::sqrt((float)d);
                                // `copysign(sqrt(max(|dot|,1e-6)), dot)` makes
                                // the sign of an exact zero observable, and
                                // whether a zero-initialised accumulator ends
                                // up +0 or -0 is a compiler choice.  Canonicalise
                                // it so an all-zero row is deterministic.
                                union {
                                    float f;
                                    uint32_t u;
                                } db;
                                db.f = dot;
                                if ((db.u << 1) == 0u) {
                                    db.u = 0u;
                                    dot = db.f;
                                }
                                const float g = 1.f / (1.f + sycl::exp(-sycl::copysign(
                                                                     sycl::sqrt(sycl::fmax(sycl::fabs(dot), 1e-6f)),
                                                                     dot)));
                                lg[0] = g;
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                            const float g = lg[0];
                            for (int i = tid; i < d; i += WG) {
                                hk[i] += g * value[i];
                            }
                        });
    });
}

} // namespace si
