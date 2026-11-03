// Grouped u4 gate/up GEMM over the resident per-expert u4 bank.
//
// The routed-expert prefill's gate/up used to be one oneDNN u4 matmul per
// (expert, 64-pair tile): at the 35B A3B's density (~40 pairs/expert) that is
// ~190 dispatches per layer-chunk plus a gather, an activation quantize and a
// scatter per tile.  None of that work depends on the expert - the activation
// tile is the same token rows for every expert - so it is hoisted out:
//
//   * one activation quantizer call per layer (act_quant_grp_run), and
//   * one kernel launch over every (active expert, column block) of the layer.
//
// Each workgroup stages one Q4_K row block's u4 plane for its column range in
// K-chunks of 128 values (4 groups), expands the nibbles to int8 in SLM, and
// runs the expert's pairs in BM-row tiles.  The arithmetic is the bank's own
// formula, exactly as the decode GEMV computes it:
//
//     out[m][n] = sum_g sa[m][g] * (step[g][n] * dot(q[g][n], xq[m][g])
//                                   + off[g][n] * sum(xq[m][g]))
//
// with dot() a dp4a over the group's 32 int8 values (q is 0..15, unsigned) and
// sum() the same quantizer's integer group sum (xs).  That is the same
// expression the oneDNN path computes (oneDNN's grouped-scale u4 matmul plus
// the shift(xs) correction), so this is a different accumulation *order*, not
// a different formula.
//
// The bank plane is [N][K/2] u4 (see common/w4.h), so a column's 64 bytes for
// a 128-value K chunk are one cache line and the staging map is
// (column, quarter) so a warp reads eight whole lines.  Staging scales/offs is
// [K/32][N] f16 and therefore already coalesced along n.
//
// One projection per launch: the caller runs gate first and passes its output
// as `gate` to the up launch, which applies silu(gate)*up in place.  Fusing the
// two would double the accumulator count (spills) and the SLM tile for no
// weight-traffic saving.
#include "kernels.h"

#include <cstdio>

#include "common/env.h"
#include "device/device_profile.h"
#include "w4.h" // w4_group_shift (PF_W4_GROUP)
#include "kernel_utils.h"

namespace si {

using namespace sycl;
using namespace si::kd;

namespace {

constexpr int kKC = 128; // K values per staged chunk (4 groups of 32)
constexpr int kKG = kKC / 32;

// First four of the eight u4 values packed in one uint32 (elements 0..3).
inline uint32_t u4_first4(uint32_t v) {
    return (v & 0x0000000Fu) | ((v & 0x000000F0u) << 4) | ((v & 0x00000F00u) << 8) | ((v & 0x0000F000u) << 12);
}
inline uint32_t u4_next4(uint32_t v) {
    return u4_first4(v >> 16); // elements 4..7
}

// Expand one 16-byte group (32 u4 values) to 8 int8 words, in element order.
inline void u4_expand16(const uint8_t * p, uint32_t * w) {
    const uint4 v = *reinterpret_cast<const uint4 *>(p);
    w[0] = u4_first4(v.x());
    w[1] = u4_next4(v.x());
    w[2] = u4_first4(v.y());
    w[3] = u4_next4(v.y());
    w[4] = u4_first4(v.z());
    w[5] = u4_next4(v.z());
    w[6] = u4_first4(v.w());
    w[7] = u4_next4(v.w());
}

// Hardware f16->f32.  kernel_utils' fast_h2f has a denormal/NaN branch that
// costs ~8 instructions; called 32x per output tile (the step/off factors were
// inside the row*column loop) it dominated the dp4a issue rate in the first
// version: hoisting the factors and using the one-instruction conversion took
// the kernel from 0.32 to (see below) T-MAC/s.
inline float h2f(uint16_t h) {
    sycl::half x;
    __builtin_memcpy((void *)&x, &h, sizeof(x));
    return (float)x;
}

// The expanded-int8 weight tile is stored chunk-major: one uint4 (16 int8) at
// ((gt*8 + col_local*2 + half)*32 + lane).  Lanes are 16 B apart inside a
// chunk, which is the widest conflict-free stride a 16-byte SLM load can have
// (8 lanes fill the 32 banks per cycle).  Storing it column-major instead - the
// obvious layout - puts all 32 lanes on the same four banks (a column's 32 B
// stride is exactly the bank-wrap period) and measured 5.4x slower end to end.
template <int TC>
inline size_t wchunk(int gt, int c, int half, int lane) {
    return (((size_t)(gt * (TC * 2) + c * 2 + half)) * 32 + (size_t)lane) * 16;
}

// BM pair rows per weight pass.  TM = BM / 8 keeps the thread row map
// (8 row-groups x 32 column-groups = 256 threads) true for every BM.
template <int BM, int BN, int TC>
struct shape {
    static constexpr int TM = BM / 8;
    static constexpr size_t slm_bytes = (size_t)kKG * BN * 32     // int8 weight tile
                                        + (size_t)BM * kKC         // int8 activation tile
                                        + 2 * (size_t)kKG * BN * 2 // step + off
                                        + (size_t)BM * kKG * 2      // activation scales
                                        + (size_t)BM * kKG * 4      // activation group sums
                                        + (size_t)BM * 4;           // row table
    static_assert(BM % 8 == 0, "TM = BM/8");
    static_assert(BN == TC * 32, "thread column map must cover BN");
    static constexpr int TX = 256;
};

template <int BM, int BN, int TC>
void grp_u4_impl(queue & q, const uint8_t * bank, size_t plane_bytes, const int32_t * off,
                 const int32_t * pair, const int32_t * active, const int32_t * n_active, int n_act_max,
                 const int8_t * xq, const uint16_t * asa, const float * xs, float * out, int out_stride, int K,
                 int N, int top_k, const float * gate, int abl) {
    constexpr int TM = shape<BM, BN, TC>::TM;
    constexpr int kTX = shape<BM, BN, TC>::TX;
    const int nblk = N / BN;
    const int ng = K / 32;
    const int gs = si::w4_group_shift(); // host value; device code cannot read it
    const int ngw = ng >> gs;
    const size_t rb = (size_t)(K / 2);
    const int nkt = K / kKC;
    const size_t vb = (size_t)N * rb;

    q.submit([&](handler & h) {
        // uint32 element types, not int8_t: a byte accessor makes IGC lower every
        // 4-byte SLM read as four byte inserts (measured ~590 `mov .b` and 297
        // `shr` per group against 256 dp4a, i.e. 0.51 T-MAC/s).  The buffers hold
        // the same bytes; only the accessor's element type changes.
        local_accessor<uint32_t, 1> ws((size_t)kKG * BN * 32 / 4, h);
        local_accessor<uint32_t, 1> as((size_t)BM * kKC / 4, h);
        local_accessor<uint16_t, 1> scs((size_t)kKG * BN, h);
        local_accessor<uint16_t, 1> ofs((size_t)kKG * BN, h);
        local_accessor<uint16_t, 1> asas((size_t)BM * kKG, h);
        local_accessor<float, 1> xss((size_t)BM * kKG, h);
        local_accessor<int32_t, 1> rows(BM, h); // activation row = pair / top_k
        local_accessor<int32_t, 1> pids(BM, h); // output row = the pair id

        h.parallel_for(nd_range<1>((size_t)n_act_max * (size_t)nblk * kTX, kTX), [=](nd_item<1> it) {
            const int gid = (int)it.get_group(0);
            const int ia = gid / nblk;
            const int cb = gid - ia * nblk;
            if (ia >= n_active[0]) {
                return;
            }
            const int e = active[ia];
            const int p_first = off[e];
            const int np = off[e + 1] - p_first;
            if (np <= 0) {
                return;
            }
            const int tid = (int)it.get_local_id(0);
            const int mg = tid >> 5; // 8 row groups
            const int cg = tid & 31; // 32 column groups
            const int m0 = mg * TM;
            const int c0 = cg * TC;
            const size_t col0 = (size_t)cb * BN;
            const uint8_t * plane = bank + (size_t)e * plane_bytes;
            const uint16_t * scl = reinterpret_cast<const uint16_t *>(plane + vb);
            const uint16_t * ofl = scl + (size_t)ngw * (size_t)N;

            for (int pt = 0; pt * BM < np; pt++) {
                const int pbase = pt * BM;
                const int nv = (np - pbase < BM) ? (np - pbase) : BM;
                if (tid < BM) {
                    const int pr = (tid < nv) ? pair[p_first + pbase + tid] : -1;
                    pids[tid] = pr;
                    rows[tid] = (pr >= 0) ? (pr / top_k) : -1;
                }
                float acc[TM][TC];
#pragma unroll
                for (int i = 0; i < TM; i++)
#pragma unroll
                    for (int j = 0; j < TC; j++) acc[i][j] = 0.f;

                for (int kc0 = 0; kc0 < K; kc0 += kKC) {
                    // ---- weight staging: (column, quarter) -> one 64 B line
                    const bool do_stage = (abl != 2);
                    for (int i = tid; do_stage && i < BN * 4; i += kTX) {
                        const int col = i >> 2;
                        const int qq = i & 3;
                        const uint8_t * p = plane + (col0 + (size_t)col) * rb + (size_t)(kc0 / 2) + (size_t)qq * 16;
                        uint32_t tmp[8];
                        u4_expand16(p, tmp);
                        const int lane_l = col / TC; // which thread owns this column
                        const int cl = col - lane_l * TC;
                        *reinterpret_cast<uint4 *>(&ws[wchunk<TC>(qq, cl, 0, lane_l) / 4]) =
                            uint4(tmp[0], tmp[1], tmp[2], tmp[3]);
                        *reinterpret_cast<uint4 *>(&ws[wchunk<TC>(qq, cl, 1, lane_l) / 4]) =
                            uint4(tmp[4], tmp[5], tmp[6], tmp[7]);
                    }
                    // ---- step / off for this chunk
                    for (int i = tid; do_stage && i < kKG * BN; i += kTX) {
                        const int g = i / BN;
                        const int n = i - g * BN;
                        const size_t s = (size_t)((kc0 / 32 + g) >> gs) * (size_t)N + col0 + n;
                        scs[i] = scl[s];
                        ofs[i] = ofl[s];
                    }
                    it.barrier(access::fence_space::local_space);
                    // rows[] is visible after this barrier; the activation
                    // staging below reads it.
                    for (int i = tid; do_stage && i < BM * (kKC / 16); i += kTX) {
                        const int m = i / (kKC / 16);
                        const int kk = (i % (kKC / 16)) * 16;
                        uint4 * d = reinterpret_cast<uint4 *>(&as[((size_t)m * kKC + kk) / 4]);
                        const int arow = rows[m];
                        if (arow >= 0) {
                            *d = *reinterpret_cast<const uint4 *>(xq + (size_t)arow * K + kc0 + kk);
                        } else {
                            *d = uint4(0, 0, 0, 0);
                        }
                    }
                    for (int i = tid; do_stage && i < BM * kKG; i += kTX) {
                        const int m = i / kKG;
                        const int g = i - m * kKG;
                        const int arow = rows[m];
                        if (arow >= 0) {
                            asas[i] = asa[(size_t)arow * ng + kc0 / 32 + g];
                            xss[i] = xs[(size_t)arow * ng + kc0 / 32 + g];
                        } else {
                            asas[i] = 0;
                            xss[i] = 0.f;
                        }
                    }
                    it.barrier(access::fence_space::local_space);

                    // ---- compute: TM rows x TC columns per thread
                    for (int gt = 0; gt < kKG && abl != 1; gt++) {
                        uint32_t wt[TC][8];
#pragma unroll
                        for (int c = 0; c < TC; c++) {
                            const uint4 lo = *reinterpret_cast<const uint4 *>(&ws[wchunk<TC>(gt, c, 0, cg) / 4]);
                            const uint4 hi = *reinterpret_cast<const uint4 *>(&ws[wchunk<TC>(gt, c, 1, cg) / 4]);
                            wt[c][0] = lo.x();
                            wt[c][1] = lo.y();
                            wt[c][2] = lo.z();
                            wt[c][3] = lo.w();
                            wt[c][4] = hi.x();
                            wt[c][5] = hi.y();
                            wt[c][6] = hi.z();
                            wt[c][7] = hi.w();
                        }
                        // The step/off factors depend only on (group, column),
                        // so convert them once per group, not once per output.
                        float stv[TC], ofv[TC];
#pragma unroll
                        for (int c = 0; c < TC; c++) {
                            stv[c] = h2f(scs[gt * BN + c0 + c]);
                            ofv[c] = h2f(ofs[gt * BN + c0 + c]);
                        }
                        // Word-major inner loop: for one weight word j the TM
                        // rows x TC columns are all independent, so the compiler
                        // has TM*TC dp4a chains in flight instead of the two a
                        // (row, column) pair offers.  A padded row is clamped to
                        // row 0 and its result is simply not stored - a `continue`
                        // made `acc` dynamically indexed and spilled it to scratch
                        // (0.23 T-MAC/s against a 3.7 dp4a issue peak).
                        float sa[TM], xsr[TM];
#pragma unroll
                        for (int m = 0; m < TM; m++) {
                            const int rs = (m0 + m) < nv ? (m0 + m) : 0;
                            sa[m] = h2f(asas[rs * kKG + gt]);
                            xsr[m] = xss[rs * kKG + gt];
                        }
                        int32_t q[TM][TC];
#pragma unroll
                        for (int m = 0; m < TM; m++)
#pragma unroll
                            for (int c = 0; c < TC; c++) q[m][c] = 0;
#pragma unroll
                        for (int j = 0; j < 8; j++) {
                            uint32_t av[TM];
#pragma unroll
                            for (int m = 0; m < TM; m++) {
                                const int rs = (m0 + m) < nv ? (m0 + m) : 0;
                                av[m] = as[((size_t)rs * kKC + gt * 32) / 4 + j];
                            }
#pragma unroll
                            for (int m = 0; m < TM; m++)
#pragma unroll
                                for (int c = 0; c < TC; c++)
                                    q[m][c] = dp4a_s8u8((int32_t)av[m], wt[c][j], q[m][c]);
                        }
#pragma unroll
                        for (int m = 0; m < TM; m++)
#pragma unroll
                            for (int c = 0; c < TC; c++)
                                acc[m][c] = sycl::fma(sa[m], stv[c] * (float)q[m][c] + ofv[c] * xsr[m], acc[m][c]);
                    }
                    it.barrier(access::fence_space::local_space);
                }

                // ---- epilogue: pair rows, optional silu(gate)*acc
#pragma unroll
                for (int m = 0; m < TM; m++) {
                    const int rr = m0 + m;
                    if (rr >= nv) {
                        continue;
                    }
                    const int pid = pids[rr];
                    if (pid < 0) {
                        continue;
                    }
                    const size_t o = (size_t)pid * (size_t)out_stride + col0 + c0;
#pragma unroll
                    for (int c = 0; c < TC; c++) {
                        out[o + c] = gate ? silu_f(gate[o + c]) * acc[m][c] : acc[m][c];
                    }
                }
                it.barrier(access::fence_space::local_space);
            }
        });
    });
}

} // namespace

bool moe_grp_u4_launch(queue & q, const uint8_t * bank, size_t plane_bytes, const int32_t * off,
                       const int32_t * pair, const int32_t * active, const int32_t * n_active, int n_act_max,
                       const int8_t * xq, const uint16_t * asa, const float * xs, float * out, int out_stride, int K,
                       int N, int top_k, const float * gate) {
    if (bank == nullptr || off == nullptr || pair == nullptr || active == nullptr || n_active == nullptr ||
        xq == nullptr || asa == nullptr || xs == nullptr || out == nullptr) {
        return false;
    }
    // gate/up only: the activation row of a pair is pair / top_k.
    if (top_k <= 0 || K <= 0 || N <= 0 || K % kKC != 0 || N % 128 != 0 || out_stride < N) {
        return false;
    }
    const size_t slm = (size_t)si::dev::for_queue(q).hw.local_mem_size;
    if (slm < shape<32, 128, 4>::slm_bytes || si::dev::wg_clamped_for_queue(q, 256) < 256) {
        return false;
    }
    // Tile sweep (all bit-identical): BM pair rows x BN columns, TC columns per
    // thread.  BM=32/TC=4 is the measured default; see the report.
    const int bm = si::env::i32("PF_MOE_GRP_BM", 32);
    const int bn = si::env::i32("PF_MOE_GRP_BN", 128);
    const int abl = si::env::i32("PF_MOE_GRP_ABL", 0); // 1 = staging only, 2 = compute only (wrong results)
    if (bn == 64) {
        if (bm == 64) {
            grp_u4_impl<64, 64, 2>(q, bank, plane_bytes, off, pair, active, n_active, n_act_max, xq, asa, xs, out,
                                   out_stride, K, N, top_k, gate, abl);
        } else {
            grp_u4_impl<32, 64, 2>(q, bank, plane_bytes, off, pair, active, n_active, n_act_max, xq, asa, xs, out,
                                   out_stride, K, N, top_k, gate, abl);
        }
    } else if (bm == 64) {
        grp_u4_impl<64, 128, 4>(q, bank, plane_bytes, off, pair, active, n_active, n_act_max, xq, asa, xs, out,
                                out_stride, K, N, top_k, gate, abl);
    } else if (bm == 16) {
        grp_u4_impl<16, 128, 4>(q, bank, plane_bytes, off, pair, active, n_active, n_act_max, xq, asa, xs, out,
                                out_stride, K, N, top_k, gate, abl);
    } else {
        grp_u4_impl<32, 128, 4>(q, bank, plane_bytes, off, pair, active, n_active, n_act_max, xq, asa, xs, out,
                                out_stride, K, N, top_k, gate, abl);
    }
    return true;
}

} // namespace si
