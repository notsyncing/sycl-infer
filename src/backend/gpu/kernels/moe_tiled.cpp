// Grouped, CSR-driven tiled expert GEMM for the routed-expert prefill.
//
// One workgroup per (active expert, BN-column block, 256 by default); the
// workgroup walks that expert's CSR pairs in BM-row tiles, so the weight pass
// is amortised over every token that selected the expert.
//
// Geometry: BN columns (256 for every default instantiation; the opt-in Xe2
// gate/up tile adds BN=512), BK = 32 = exactly one Q4_K/Q5_K sub-block (one
// 32-value group), TM = BM/8 rows x TN = 4 columns per lane (two lanes per
// column, 16 values each - so NT = 2*BN threads for every BM), fp32 weight tile
// in SLM, and the k-loop unrolled two k at a time.
//
// BM is a template parameter and the launcher picks it from the batch's average
// pair density (`pairs / experts`, i.e. how many routed pairs an expert has
// before the tile pads it):
//
//   * BM=64 covers the whole mean-32 skew of a 1024-token batch in one weight
//     pass instead of a 32-row tile plus a mostly-empty second one;
//   * BM=32 is the shipped shape (and the down projection's choice - the
//     N=2048 shape needs the extra column blocks more than the taller tile);
//   * BM=8 stops a 141-token tail (mean ~4 pairs/expert) from padding every
//     expert to 32 rows.
//
// Measured on the 35B A3B shape (gate [512][2048] Q4_K, 256 experts, 1024-token
// batch, mean 32 pairs/expert): BM=32 BN=256 18.02 ms (0.477 T-MAC/s), BM=64
// BN=256 11.70 ms (0.734).  Whole 3213-token routed half (gate+up+Q5 down over
// the engine's 1024/1024/1024/141 split): 6904 ms shipped, 5401 ms with BM=64
// on the full batches and BM=8 on the tail.  The 64-row tile needs 41.5 KB of
// SLM, so it still lowers into a 64 KB AOT device image (adl-p/acm-g10).
//
// Every BM variant is bit-identical to the shipped BM=32 output: BM changes
// only how many pairs one tile covers, never the per-pair dot or the store
// order (measured max|d| = 0 for BM = 8, 16, 32, 64).
//
// Three things here are measured, and each replaced something that was slower:
//
//  * KU = 2.  The compute loop with four loads in flight and 32 dependent FMAs
//    reaches 1.61 T-MAC/s register-resident-SLM (dev/fma_ceil.cpp: 1.96 pure
//    FMA, 0.98 at one k per iteration), i.e. it is issue-bound at 89% of the
//    FMA peak.  At KU = 1 the load-use latency of every k is exposed.
//
//  * Superblock-resident metadata.  The 16-byte block header, the 32-byte qh
//    plane (Q5_K) and the 32-byte qs chunk are loaded ONCE per 256-value
//    superblock, not once per 32-value group.  A group pair shares its qs
//    chunk (even group = low nibbles, odd = high nibbles), so the staging
//    DRAM traffic per row per superblock goes from 8*(16+32) = 384 B to
//    16+128 = 144 B for Q4_K and from 8*(16+32+32) = 640 B to 176 B for
//    Q5_K - the unique bytes.  That is worth 0.94 -> 1.00 T-MAC/s on gate/up
//    and 0.79 -> 0.86 on Q5_K down.
//
//  * All threads stage.  With NT == 2*BN each column's 32 values are staged by
//    two lanes, 16 values each, so there is no idle half of the workgroup
//    behind the barrier (the old `if (tid < BN)` left 512 of 1024 threads
//    waiting).
//
// The partially-filled last pair tile predicate-skips the FMA loop for every
// lane whose TM rows are all padding.  On the real routing (mean ~32 pairs per
// expert, so ~half the experts need a second, mostly-empty tile) that is worth
// ~10% and it is exact: those lanes write no output either way.
//
// BN is 256 for every default projection: N % 256 == 0 is already required, so
// gate/up (N = 512) is two column blocks and down (N = 2048) is eight, and the
// 34-42 KB of SLM fits every card the kernel is enabled on.  The earlier BN=512
// tile (0.996 gate/up against 0.964 at BN=256) was dropped as the default
// because its 75,264 B image does not fit the 64 KB AOT target.
//
// SI_MOE_TILED_WIDE (opt-in, compiled only when defined) adds the measured
// better Xe2 gate/up shape, BM=64 x BN=512: 0.911 T-MAC/s against 0.734 at
// BN=256 on the 512-EU part, for 75,264 B of SLM.  It is selected only for
// gate/up, a dense batch (avg >= 16, i.e. the 64-row tile) and a card that can
// take 1024 threads and 73.5 KB of SLM; everything else stays at BN=256, and
// down always does.  A build without the macro never instantiates it, so a
// 64 KB AOT image (adl-p/acm-g10) never sees the kernel; build it with
// -DSI_MOE_TILED_WIDE=1 and an Xe2 AOT device (e.g. SYCL_INFER_AOT_DEVICE=lnl).

#include "moe_tiled.h"

#include <cstdio>

#include "common/env.h"
#include "device/device_profile.h"
#include "kernel_utils.h"

namespace si {

using namespace sycl;
using namespace si::kd;

namespace {

constexpr int kBK = 32;   // one 32-value group
constexpr int kTN = 4;    // columns per lane

// BM/BN-spelled tile metadata.  TM = BM/8 makes (BM/TM) * (BN/TN) == 2*BN, i.e.
// the "two lanes per column, 16 values each" staging mapping is identical for
// every instantiation, which is what keeps the outputs bit-identical.
template <int BM, int BN>
struct tile_shape {
    static constexpr int TM = BM / 8;
    static constexpr int NT = 2 * BN;
    // vec<TM> loads need kASP*4 % (TM*4) == 0: +4 for TM <= 4, +8 for TM = 8.
    static constexpr int ASP = (TM > 4) ? BM + 8 : BM + 4;
    static_assert(BM % 8 == 0, "TM = BM/8");
    static_assert((BM / TM) * (BN / kTN) == NT, "every thread stages");
    static constexpr size_t slm_bytes =
        (size_t)kBK * BN * sizeof(float) +           // fp32 weight tile
        (size_t)kBK * ASP * sizeof(float) +          // fp32 activation tile
        2 * (size_t)BM * sizeof(int32_t);            // row / pair table
};

// The 6-bit scale/min pair of sub-block `sb`, out of the three 32-bit words the
// superblock header holds (identical packing to quant.h's get_scale_min_k4).
inline void q4k_scale_min(uint32_t s0, uint32_t s1, uint32_t s2, int sb, int & sc8, int & mi8) {
    const uint64_t lo = (uint64_t)s0 | ((uint64_t)s1 << 32);
    const uint32_t qsb = (uint32_t)((lo >> (8 * sb)) & 0xFFu);
    const uint32_t qs4 = (sb < 4) ? (uint32_t)((lo >> (8 * (sb + 4))) & 0xFFu)
                                  : (uint32_t)((s2 >> (8 * (sb - 4))) & 0xFFu);
    if (sb < 4) {
        sc8 = (int)(qsb & 63u);
        mi8 = (int)(qs4 & 63u);
    } else {
        const uint32_t qm4 = (uint32_t)((lo >> (8 * (sb - 4))) & 0xFFu);
        sc8 = (int)((qs4 & 0xFu) | ((qm4 >> 6) << 4));
        mi8 = (int)((qs4 >> 4) | ((qsb >> 6) << 4));
    }
}

// QT is the GGML type id: 12 = Q4_K (144 B / 256 values), 13 = Q5_K (176 B).
template <uint32_t QT, int BM, int BN>
void moe_tiled_impl(queue & q, const char * bank, size_t slot_stride, const int32_t * slot_map,
                    const int32_t * off, const int32_t * pair, const int32_t * active,
                    const int32_t * n_active, int n_act_max, const int8_t * xq,
                    const uint16_t * asa, float * out, int out_stride, int K, int N, int top_k,
                    const float * gate) {
    constexpr int kRB = (QT == 12) ? 144 : 176;
    constexpr int kQOff = (QT == 12) ? 16 : 48; // qs offset inside a superblock
    constexpr int kNT = tile_shape<BM, BN>::NT;
    constexpr int kTM = tile_shape<BM, BN>::TM;
    constexpr int kASP = tile_shape<BM, BN>::ASP;
    constexpr int kNG = BN / kTN;
    // Two lanes per column, 16 values each: NT == 2*BN in every instantiation.
    constexpr int kKH = 2;
    constexpr int kKV = kBK / kKH;
    const int nblk = N / BN;
    const int nkt = K / kBK;
    const int gstride = K / 32;                        // f16 scales per activation row
    const size_t rb = (size_t)(K / 256) * kRB;         // bytes per output row

    q.submit([&](handler & h) {
        local_accessor<float, 1> as(kBK * kASP, h);
        local_accessor<float, 1> af(kBK * BN, h);
        local_accessor<int32_t, 1> rows(2 * BM, h);

        h.parallel_for(nd_range<1>((size_t)n_act_max * (size_t)nblk * kNT, kNT),
                       [=](nd_item<1> it) {
                           const int gid = (int)it.get_group(0);
                           const int ia = gid / nblk;
                           // Empty grid slots of n_act_max (and a cached expert
                           // that is not resident on this device) retire whole
                           // workgroups before any barrier.
                           if (ia >= n_active[0]) {
                               return;
                           }
                           const int cb = gid - ia * nblk;
                           const int e = active[ia];
                           const int slot = slot_map[e];
                           if (slot < 0) {
                               return;
                           }
                           const int p_first = off[e];
                           const int np = off[e + 1] - p_first;
                           if (np <= 0) {
                               return;
                           }

                           const char * wbase = bank + (size_t)slot * slot_stride;
                           const int tid = (int)it.get_local_id(0);
                           const int m0 = (tid / kNG) * kTM;
                           const int c0 = (tid % kNG) * kTN;
                           const size_t col0 = (size_t)cb * BN;
                           const int scol = tid % BN;          // column this lane stages
                           const int khalf = tid / BN;         // which 16 values of the group
                           const char * wrow = wbase + (col0 + (size_t)scol) * rb;

                           for (int pt = 0; pt * BM < np; pt++) {
                               const int p0 = pt * BM;
                               if (tid < BM) {
                                   const int p = p0 + tid;
                                   if (p < np) {
                                       const int pr = pair[p_first + p];
                                       rows[tid] = (top_k > 0) ? (pr / top_k) : pr;
                                       rows[BM + tid] = pr;
                                   } else {
                                       rows[tid] = -1;
                                       rows[BM + tid] = -1;
                                   }
                               }
                               it.barrier(access::fence_space::local_space);

                               // A lane whose TM rows are all padding has no
                               // output to produce; skip its whole FMA loop
                               // (the barriers below are still reached by
                               // every thread).  ~10% on the real routing.
                               bool lane_used = false;
#pragma unroll
                               for (int i = 0; i < kTM; i++)
                                   lane_used = lane_used || rows[BM + m0 + i] >= 0;

                               float acc[kTM][kTN];
#pragma unroll
                               for (int i = 0; i < kTM; i++)
#pragma unroll
                                   for (int j = 0; j < kTN; j++) acc[i][j] = 0.f;

                               // Superblock-resident metadata (registers).
                               uint32_t hdr[4], qw[8], hw[8];
                               for (int kt = 0; kt < nkt; kt++) {
                                   const int k0 = kt * kBK;
                                   const int sb = (k0 & 255) >> 5;

                                   // ---- weight staging: unpack this lane's
                                   // half of one 32-value group into the fp32
                                   // SLM tile.
                                   {
                                       const size_t b = (size_t)(k0 / 256) * kRB;
                                       if (sb == 0) {
                                           const uint4 h4 = vload<uint4>(wrow + b);
                                           hdr[0] = h4.x();
                                           hdr[1] = h4.y();
                                           hdr[2] = h4.z();
                                           hdr[3] = h4.w();
                                           if constexpr (QT == 13) {
                                               const uint4 c0v = vload<uint4>(wrow + b + 16 + 16 * khalf);
                                               hw[0] = c0v.x();
                                               hw[1] = c0v.y();
                                               hw[2] = c0v.z();
                                               hw[3] = c0v.w();
                                           }
                                       }
                                       if (!(sb & 1)) {
                                           const int qo = kQOff + (sb >> 1) * 32 + 16 * khalf;
                                           const uint4 a0 = vload<uint4>(wrow + b + qo);
                                           qw[0] = a0.x();
                                           qw[1] = a0.y();
                                           qw[2] = a0.z();
                                           qw[3] = a0.w();
                                       }
                                       int sc8, mi8;
                                       q4k_scale_min(hdr[1], hdr[2], hdr[3], sb, sc8, mi8);
                                       const float s =
                                           fast_h2f((uint16_t)(hdr[0] & 0xFFFFu)) * (float)sc8;
                                       const float mf =
                                           fast_h2f((uint16_t)(hdr[0] >> 16)) * (float)mi8;
                                       const int sh = (sb & 1) ? 4 : 0;
                                       float * dst = &af[(khalf * kKV) * BN + scol];
#pragma unroll
                                       for (int k = 0; k < kKV; k++) {
                                           uint32_t v = (qw[k >> 2] >> (8 * (k & 3) + sh)) & 0x0Fu;
                                           if constexpr (QT == 13) {
                                               v |= ((hw[k >> 2] >> (8 * (k & 3) + sb)) & 1u) << 4;
                                           }
                                           dst[k * BN] = s * (float)v - mf;
                                       }
                                   }

                                   // ---- activation tile: int8 x its
                                   // per-32-group f16 scale, fp32 in SLM; zero
                                   // past the tile's valid pair count.
                                   for (int t = tid; t < BM * kBK; t += kNT) {
                                       const int k = t & 31;
                                       const int m = t >> 5;
                                       const int prow = rows[m];
                                       float v = 0.f;
                                       if (prow >= 0) {
                                           v = (float)xq[(size_t)prow * K + k0 + k] *
                                               fast_h2f(asa[(size_t)prow * gstride + (k0 >> 5)]);
                                       }
                                       as[k * kASP + m] = v;
                                   }
                                   it.barrier(access::fence_space::local_space);

                                   if (lane_used) {
                                       // Two k per iteration: the four loads
                                       // issue back to back and the 32 FMAs
                                       // that follow hide their latency.
#pragma unroll
                                       for (int k = 0; k < kBK; k += 2) {
                                           const vec<float, kTM> a0 =
                                               vload<vec<float, kTM>>(&as[(k + 0) * kASP + m0]);
                                           const vec<float, kTM> a1 =
                                               vload<vec<float, kTM>>(&as[(k + 1) * kASP + m0]);
                                           const vec<float, kTN> w0 =
                                               vload<vec<float, kTN>>(&af[(k + 0) * BN + c0]);
                                           const vec<float, kTN> w1 =
                                               vload<vec<float, kTN>>(&af[(k + 1) * BN + c0]);
#pragma unroll
                                           for (int i = 0; i < kTM; i++)
#pragma unroll
                                               for (int j = 0; j < kTN; j++) {
                                                   acc[i][j] = sycl::fma(a0[i], w0[j], acc[i][j]);
                                                   acc[i][j] = sycl::fma(a1[i], w1[j], acc[i][j]);
                                               }
                                       }
                                   }
                                   it.barrier(access::fence_space::local_space);
                               }

#pragma unroll
                               for (int i = 0; i < kTM; i++) {
                                   const int pid = rows[BM + m0 + i];
                                   if (pid < 0) {
                                       continue;
                                   }
                                   const size_t o = (size_t)pid * (size_t)out_stride + col0 + c0;
                                   float * op = out + o;
                                   if (gate != nullptr) {
                                       const float * gp = gate + o;
#pragma unroll
                                       for (int j = 0; j < kTN; j++) op[j] = silu_f(gp[j]) * acc[i][j];
                                   } else {
#pragma unroll
                                       for (int j = 0; j < kTN; j++) op[j] = acc[i][j];
                                   }
                               }
                               // The epilogue reads the SLM row table and the
                               // next pair tile overwrites it.
                               if ((pt + 1) * BM < np) {
                                   it.barrier(access::fence_space::local_space);
                               }
                           }
                       });
    });
}

// Gate/up buys from the taller tile (one weight pass over the mean-32 skew);
// down stays at 32 (its N=2048 shape wants the column blocks) and only drops to
// 8 when the batch is too sparse to fill a 32-row tile.
inline int tiled_bm(int avg, int top_k) {
    if (top_k > 0) {
        return (avg >= 16) ? 64 : (avg < 8 ? 8 : 32);
    }
    return (avg < 8) ? 8 : 32;
}

template <uint32_t QT>
inline void tiled_dispatch_bm(queue & q, int bm, const char * bank, size_t slot_stride,
                              const int32_t * slot_map, const int32_t * off, const int32_t * pair,
                              const int32_t * active, const int32_t * n_active, int n_act_max,
                              const int8_t * xq, const uint16_t * asa, float * out, int out_stride,
                              int K, int N, int top_k, const float * gate) {
    switch (bm) {
    case 64:
        moe_tiled_impl<QT, 64, 256>(q, bank, slot_stride, slot_map, off, pair, active, n_active,
                                    n_act_max, xq, asa, out, out_stride, K, N, top_k, gate);
        break;
    case 8:
        moe_tiled_impl<QT, 8, 256>(q, bank, slot_stride, slot_map, off, pair, active, n_active,
                                   n_act_max, xq, asa, out, out_stride, K, N, top_k, gate);
        break;
    default:
        moe_tiled_impl<QT, 32, 256>(q, bank, slot_stride, slot_map, off, pair, active, n_active,
                                    n_act_max, xq, asa, out, out_stride, K, N, top_k, gate);
        break;
    }
}

#ifdef SI_MOE_TILED_WIDE
// Xe2-only gate/up tile: BN=512 x BM=64 measured 0.911 T-MAC/s against 0.734 at
// BN=256 on the 512-EU part (gate [512][2048] Q4_K, 256 experts, 1024 tokens).
// It needs 75,264 B of SLM (73.5 KiB), so it is compiled only when the build
// opts in: -DSI_MOE_TILED_WIDE=1, and an AOT device that accepts it (an Xe2
// target such as SYCL_INFER_AOT_DEVICE=lnl).  A 64 KB AOT image must not see it.
constexpr size_t kTiledWideSlm = tile_shape<64, 512>::slm_bytes;
constexpr int kTiledWideNT = tile_shape<64, 512>::NT; // 1024 threads
static_assert(kTiledWideSlm == 75264, "wide tile SLM changed; update the CMake guard");
#endif

} // namespace

bool moe_tiled_supported(int K, int N, uint32_t type) {
    if (type != 12 && type != 13) {
        return false;
    }
    if (N <= 0 || N % 256 != 0) {
        return false;
    }
    if (K <= 0 || K % 256 != 0) {
        return false;
    }
    return true;
}

bool moe_tiled_gemm_launch(queue & q, const void * bank, size_t slot_stride,
                           const int32_t * slot_map, const int32_t * off, const int32_t * pair,
                           const int32_t * active, const int32_t * n_active, int n_act_max,
                           const int8_t * xq, const uint16_t * asa, float * out, int out_stride,
                           int K, int N, int top_k, uint32_t type, const float * gate, int n_pairs_in) {
    // Q5_K is a down weight for this engine (its qh plane makes every staged
    // value 22% more bytes), so gate/up stays Q4_K.
    if (!moe_tiled_supported(K, N, type)) {
        return false;
    }
    if (top_k > 0 && type != 12) {
        return false;
    }
    if (n_act_max <= 0 || out_stride < N) {
        return false;
    }
    if (bank == nullptr || slot_map == nullptr || off == nullptr || pair == nullptr ||
        active == nullptr || n_active == nullptr || xq == nullptr || asa == nullptr ||
        out == nullptr) {
        return false;
    }
    // ---- tile selection: the batch's average pair density, pairs / experts.
    //
    // The caller's n_act_max is the grid bound, and the engine computes it as
    // min(n_expert, n_pairs), so `off[n_act_max]` is the CSR's last prefix entry
    // - the exact total pair count whenever pairs >= n_expert (the regime the
    // tiled path runs in: its caller gate is n_pairs >= 256, and the 35B A3B has
    // 256 experts).  Below that n_act_max == n_pairs, off[n_act_max] is a lower
    // bound on the total, and the result is still the conservative choice (a
    // handful of pairs per expert => the 8-row tile).
    //
    // This is one 4-byte D2H read.  The launcher cannot see n_pairs otherwise
    // (off/active are device pointers and the recorded-graph paths never reach
    // this kernel - it is mode-2 prefill only, which is replayed directly), so
    // the alternative was a new argument and an engine call-site change.  The
    // read is skipped when PF_MOE_TILED_BM pins a tile, which is the A/B and
    // the way to take the sync out of a multi-device prefill.
    const int bm_env = SI_ENV_I32("PF_MOE_TILED_BM", 0);
    int bm = bm_env;
    if (bm == 0) {
        int n_pairs = n_pairs_in;
        if (n_pairs < 0) {
            n_pairs = 0;
            q.memcpy(&n_pairs, off + n_act_max, sizeof(int32_t)).wait();
        }
        const int avg = n_pairs / (n_act_max > 0 ? n_act_max : 1);
        bm = tiled_bm(avg, top_k);
        if (si::env::flag("PF_MOE_TILED_DBG")) {
            std::fprintf(stderr, "[tiled] top_k=%d n_act_max=%d n_pairs=%d avg=%d -> BM=%d\n",
                         top_k, n_act_max, n_pairs, avg, bm);
        }
    }

    const size_t slm = (size_t)si::dev::for_queue(q).hw.local_mem_size;
    // Per-BM SLM at BN=256: 34,368 B (BM=8), 37,632 (BM=32), 42,496 (BM=64);
    // all three fit a 64 KB AOT device image (adl-p/acm-g10), so they lower
    // everywhere even though only Xe2 runs this kernel.  The opt-in wide tile is
    // 75,264 B and is only considered when this build compiled it
    // (SI_MOE_TILED_WIDE).
    // Cached per device (si::dev::for_queue); a live get_info in a launcher is
    // the 3.2 ms-per-launch trap the device-profile refactor removed.
    int wg = tile_shape<32, 256>::NT; // 512
    size_t slm_need = bm == 64   ? tile_shape<64, 256>::slm_bytes
                      : bm == 8  ? tile_shape<8, 256>::slm_bytes
                                 : tile_shape<32, 256>::slm_bytes;
#ifdef SI_MOE_TILED_WIDE
    // Gate/up only: down measured better at BN=256, and N must be a whole number
    // of 512-column blocks (n_ff_exp = 512 on the 35B A3B).  The 73.5 KB tile is
    // also skipped on a card that cannot take it, falling back to BN=256.
    const bool wide = top_k > 0 && bm == 64 && (N % 512 == 0) && slm >= kTiledWideSlm &&
                      si::dev::wg_clamped_for_queue(q, kTiledWideNT) >= kTiledWideNT;
    if (wide) {
        wg = kTiledWideNT;
        slm_need = kTiledWideSlm;
        if (si::env::flag("PF_MOE_TILED_DBG")) {
            std::fprintf(stderr, "[tiled] wide gate/up: BM=64 BN=512 (%zu B SLM, %d threads)\n",
                         slm_need, wg);
        }
    }
#endif
    if (si::dev::wg_clamped_for_queue(q, wg) < wg || slm < slm_need) {
        return false;
    }
    if (type == 12) {
#ifdef SI_MOE_TILED_WIDE
        if (wide) {
            moe_tiled_impl<12, 64, 512>(q, (const char *)bank, slot_stride, slot_map, off, pair,
                                        active, n_active, n_act_max, xq, asa, out, out_stride, K, N,
                                        top_k, gate);
            return true;
        }
#endif
        tiled_dispatch_bm<12>(q, bm, (const char *)bank, slot_stride, slot_map, off, pair, active,
                              n_active, n_act_max, xq, asa, out, out_stride, K, N, top_k, gate);
    } else {
        tiled_dispatch_bm<13>(q, bm, (const char *)bank, slot_stride, slot_map, off, pair, active,
                              n_active, n_act_max, xq, asa, out, out_stride, K, N, top_k, gate);
    }
    return true;
}

} // namespace si
