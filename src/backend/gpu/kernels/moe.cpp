// Mixture-of-Experts kernels: router, per-expert gate/up and down, and the
// weighted reduction.
//
// The one thing that makes MoE different from every other GEMV in this engine
// is that *which* weight an activation is multiplied by is a runtime value: the
// router picks k of n_expert experts per token, and nothing about that choice
// is known when a command graph is recorded.  So no launcher here takes a
// `(base pointer, K, N)` weight the way the dense paths do.  Instead the expert
// id is turned into a weight address through a *slot table*:
//
//     slot = slot_map[expert]          (-1 => not resident on this device)
//     w    = bank + (size_t)slot * slot_stride
//
// `slot_map` is the identity (slot == expert) when every expert is resident, so
// the resident and the cached case run the same kernel with one extra
// indirection per work-group.  That indirection is also what makes the
// host-resident (ncmoe) case possible without a second kernel: the host fills
// slot_map per step and any expert it did not make resident is skipped here and
// computed on the host instead.
//
// The expert tensors are read in their native GGUF packing (Q4_K / Q5_K / Q6_K /
// Q8_0 blocks) with the same `dequant_sb_lane_typed` lane split the dense fp32
// GEMV uses, so an expert plane is byte-for-byte the weight a dense GEMV would
// have been handed.
//
// Work decomposition: one work-group per (active expert, row block) with one
// warp per output row, looping over the tokens that selected the expert.  The
// token -> expert map is built once per layer into a CSR (`off`/`pair`) so an
// expert's weights are streamed once for all of its tokens instead of once per
// token - that reuse is the point of a grouped GEMM, and a plain per-pair order
// would read a prefill's expert weights once per token that selected them.
#include "kernels.h"
#include "w4.h" // PF_W4_GROUP (w4_group_shift)
#include "device/device_profile.h"
#include "kernel_utils.h"
#include "gguf.h" // GGML_TYPE_* (the kernel headers only pull the raw type ids)

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "common/env.h"

namespace si {

using namespace sycl;
using namespace si::kd;

static constexpr int kMoERG = 8;
// Pairs processed per tile in the expert kernels.  The dequantized superblock is
// reused across this many pairs, so this is the arithmetic-intensity knob: with
// ~14 pairs per expert on a 512-token prefill it makes the row read ~once
// instead of ~14 times.  Small enough that the per-pair accumulators stay in
// registers next to the warps' other live state.
static constexpr int kMoEPB = 8;
// Host-side only, and forwarded into the kernels below: device code may not call a
// host function or read a non-constexpr static (both are hard compile errors).
// PF_MOE_DQ2 halves the dequant ARITHMETIC and changes no load; see the note on
// dequant_sb_lane_typed in kernel_utils.h for the experiment it belongs to.
static inline bool moe_dq2() {
    static const bool v = si::env::flag("PF_MOE_DQ2");
    return v;
}
// PF_MOE_NOACT is the other half of the decomposition: it keeps every weight load
// and every dequant op and replaces the ACTIVATION operand of the FMA with a
// value derived from the loop indices, so the activation load disappears and
// nothing else does.  Together with PF_MOE_DQ2 (which drops the weight
// arithmetic and keeps both loads) the four combinations account for the decode
// expert kernel's time as load / weight-alu / activation-load / fma, one variable
// at a time.  Both are wrong-answering.
static inline bool moe_noact() {
    static const bool v = si::env::flag("PF_MOE_NOACT");
    return v;
}
// PF_MOE_PERSIST: a work-stealing form of the grouped expert GEMM.
//
// WHY, and it is not bandwidth and not dp4a issue rate.  The measured routing on
// the 35B A3B is 71 active experts with 1..256 pairs each, mean 54.6 - so a
// work-group that owns a 256-pair expert walks 32 row tiles while one that owns a
// 1-pair expert walks 1.  That is a 32x imbalance on a STATIC grid, and it is why
// this kernel sustains only 50-68 GB/s of unique weight against the card's
// 376 GB/s (AGENTS.md: "Bandwidth is not the scarce resource here; occupancy
// is").  PF_MOE_FLAT moves the row tile into the grid and buys 2% on gate, -11%
// on down, because it is still a static assignment with a host-sized grid: the
// exact tile count lives on the device, so the grid is sized by the upper bound
// `n_pairs/M + n_expert` and about a third of the work-groups scan off the end.
//
// This form makes the grid the MACHINE instead of the problem and lets the
// device decide who does what: a persistent work-group pulls (expert, row tile,
// column block) triples from an atomic counter until they run out.  The
// 32-tile expert then becomes 32 independent work items that 32 different
// work-groups start immediately, instead of 32 tiles serialized inside one of
// them.  Column block is the INNER index of a work item, so consecutive items are
// consecutive column blocks of the same (expert, tile) and the weight stream
// stays sequential across work items.
//
// The counter and the work-item count are per-queue scratch, keyed on the queue
// address exactly like attn_xmx's xmx_get - a shared static would be a
// cross-device bug under --layer-map.
struct moe_persist_scratch {
    int32_t * counter = nullptr; // atomic work-item cursor
    int32_t * nitems = nullptr;  // exact work-item count, summed on device
};
static moe_persist_scratch & moe_persist_get(queue & q) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lk(mtx);
    static std::vector<std::pair<queue *, moe_persist_scratch>> reg;
    for (auto & e : reg) {
        if (e.first == &q) {
            return e.second;
        }
    }
    reg.emplace_back(&q, moe_persist_scratch{});
    moe_persist_scratch & s = reg.back().second;
    if (!s.counter) {
        s.counter = (int32_t *) malloc_device(2 * sizeof(int32_t), q);
    }
    return s;
}

// PF_MOE_NPB: consecutive OUTPUT COLUMNS one warp walks, reusing one accumulator
// set.  The decode expert kernel runs n_act x n_ff_exp warps (4096 on the 35B A3B)
// and each reads a different output row, i.e. a superblock 1152 bytes further
// along, so the memory system sees 4096 concurrent read streams on a 1152-byte
// stride.  That is the shape DRAM row buffers thrash on, and it is the one
// explanation left for the expert path's ~24 GB/s against the dense int8 GEMV's
// 132 GB/s on the SAME device memory.  (Partly right - see the table below.)
// Walking NPB adjacent columns turns each
// warp's 1-stream-per-row into one sequential NPB*1152-byte run and divides the
// stream count by NPB.
//
// The columns are walked SEQUENTIALLY, not accumulated together: holding NPB rows
// of ag[]/au[] at once is NPB*kMoEPB*2 floats and blows the register budget.  With
// the loop outside, the accumulators are reused and nothing else changes.
//
// MEASURED, tg128 ms/token, all five cells in ONE run because the box drifts ~13%
// between runs.  The hypothesis is only PARTLY right: dividing the stream count
// buys 6-7% and then the occupancy loss takes over, hard.
//
//   NPB        1      2      4      8     16
//   depth 512 43.2   40.3   40.5   43.3   54.5
//   depth 16k 46.1    -     43.5   46.4   57.6
//   depth 64k 54.9    -     52.2   55.1   66.4
//
// NPB=4 is best or tied-best at all three depths, so it is the default.
// dev/grp_min is 0/8192 at EVERY NPB, so the row-boundary handling is right.
static inline int moe_npb() {
    static const int v = si::env::i32("PF_MOE_NPB", 4);
    return v < 1 ? 1 : (v > 16 ? 16 : v);
}
static constexpr int kMoEWG = kMoERG * 32;

// Round a *thread* count up to a whole number of 256-thread work-groups.
// Level Zero (and therefore every nd_range on an A770) rejects a global size
// that is not a multiple of the local one with "Non-uniform work-groups are not
// supported", so every grid derived from a runtime extent is rounded up here
// and the tail threads exit on their own bound.  Callers pass THREADS, not
// work-groups: the expert kernels below are indexed by warp, so a grid sized in
// work-groups runs 1/8 of the rows.
inline size_t moe_grid(size_t threads) {
    return ((threads + kMoEWG - 1) / kMoEWG) * kMoEWG;
}

static constexpr int kMoENW = kMoERG; // warps per work-group

namespace {

inline float moe_h2f(uint16_t h) {
    sycl::half x;
    __builtin_memcpy((void *)&x, &h, sizeof(x));
    return (float)x;
}

// bf16 -> f32 is a 16-bit left shift into the f32 exponent field.
inline float moe_bf2f(uint16_t h) {
    const uint32_t b = (uint32_t)h << 16;
    float f;
    __builtin_memcpy(&f, &b, sizeof(f));
    return f;
}


// Read element `k` of a gate_inp row.  The router's type is the GGUF's: the
// trunk layers ship f32 and the NextN block bf16, so it travels with the
// launcher rather than being assumed.
template <uint32_t QT>
inline float moe_rd(const char * p, int k) {
    if constexpr (QT == GGML_TYPE_F32) {
        return ((const float *)p)[k];
    } else if constexpr (QT == GGML_TYPE_F16) {
        return moe_h2f(((const uint16_t *)p)[k]);
    } else {
        return moe_bf2f(((const uint16_t *)p)[k]);
    }
}

// One quantized row dot.  `wrow` points at row `n`'s first superblock and `xr`
// at the activation row; the 32-lane x 8 split is the dense fp32 GEMV's.
template <uint32_t QT>
inline float moe_dot(const char * wrow, const float * xr, int n_sb, int sb_bytes, int lane) {
    float a = 0.f;
    for (int sb = 0; sb < n_sb; sb++) {
        float w[8];
        dequant_sb_lane_typed<QT>(wrow + (size_t)sb * sb_bytes, lane, w);
        const float * xs = xr + (size_t)sb * 256;
#pragma unroll
        for (int b = 0; b < 8; b++) {
            a += w[b] * xs[32 * b + lane];
        }
    }
    return a;
}


} // namespace

// ---------------------------------------------------------------------------
// Router: logits = gate_inp . x, softmax over all n_expert, top k, then
// weights /= clamp(sum, 6.103515625e-5) - the reference's `norm_w`.
//
// The top-k breaks ties towards the *lowest* expert index, which is what the
// reference's argsort does, so the selected set is deterministic.
//
// The router is three plain kernels over global memory: no local_accessor, no
// barrier, no sub-group collective.
//
// The one-work-group-per-token version (logits in SLM, softmax + top-k after a
// barrier) is the obvious formulation and it is *wrong on the A770's Level Zero
// backend*: the kernel body never took effect, so ids/wts came back as their
// initial contents - every token "selected" expert 0 with weight 0, which reads
// as a router bug and is not one.  The same source passes under OpenCL and on
// the Iris Xe, so it is that backend's codegen for this shape, not the
// algorithm.  None of the three kernels below needs shared memory: the logits
// are n_expert*n_tok floats (8 KB for the 35B), so spilling them to USM costs
// nothing next to a barrier the scheduler cannot honour.
template <uint32_t RT>
static sycl::event moe_router_logits(queue & q, const void * gate_inp, int esize, const float * x, int x_stride,
                                     int n_tok, int n_embd, int n_expert, float * logits) {
    // ONE SUB-GROUP PER (token, expert), lanes tiling K in the same 32-lane x 8
    // split the expert dot and the dense fp32 GEMV use.
    //
    // This was `range<2>(n_tok, n_expert)` - one *work-item* per logit, each
    // walking n_embd elements serially.  For a decode that is 256 threads on a
    // 512-EU part, each streaming a different 8 KB row, so a warp instruction
    // touches 32 cache lines to use 128 bytes and there are not enough warps in
    // flight to hide the latency.  Measured on the 35B A3B it was 4.4 s of a
    // 4.6 s decode - 95% of the whole forward, for what is 0.5 M-MAC of actual
    // work.  The expert GEMMs that used to be blamed for the prefill's 94% are
    // 1.1 s of the same prefill once this is tiled.
    const int n_sb = n_embd / 256;
    const size_t n_work = (size_t) n_tok * (size_t) n_expert;
    return q.parallel_for(nd_range<1>((n_work * 32 + 31) & ~(size_t) 31, 32), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int lane = (int) it.get_local_id(0);
        const sub_group sgg = it.get_sub_group();
        const size_t w = (size_t) it.get_group(0);
        if (w >= n_work) {
            return; // group-uniform, so the collective below still runs in step
        }
        const int t = (int) (w / (size_t) n_expert);
        const int e = (int) (w % (size_t) n_expert);
        // gate_inp is [n_expert][n_embd] row-major, so expert e's row starts at
        // e*n_embd ELEMENTS.  Indexing that in bytes (e*n_embd) reads four
        // experts per stride and yields a plausible, entirely wrong logit
        // vector - which is exactly what it did.
        const char * wr = (const char *) gate_inp + (size_t) e * (size_t) n_embd * esize;
        const float * xr = x + (size_t) t * x_stride;
        float a = 0.f;
        for (int sb = 0; sb < n_sb; sb++) {
            const char * p = wr + (size_t) sb * 256 * esize;
            const float * xs = xr + (size_t) sb * 256;
#pragma unroll
            for (int b = 0; b < 8; b++) {
                a += moe_rd<RT>(p, 32 * b + lane) * xs[32 * b + lane];
            }
        }
        const float r = sg_sum(a, sgg);
        if (lane == 0) {
            logits[(size_t) t * n_expert + e] = r;
        }
    });
}

// ---------------------------------------------------------------------------
// Tiled router logits: one sub-group per (one token x EB experts) tile.
//
// WHY.  The GEMV above is one sub-group per (token, expert), so each of a
// token's n_expert sub-groups re-reads the token's whole activation row
// (n_expert x the activation traffic) and every expert row is re-read once per
// token.  At the 35B A3B's prefill shape (803 tokens, K=2048, 256 experts) that
// is 3.4 GB of loads per layer for 0.42 GMAC and measures ~2.6 ms per layer
// (403 ms over the 40 layers of a 3213-token prefill, 5% of it) at
// 0.16 T-MAC/s.  It is not a bandwidth limit: the redundancy is the limit.
//
// A tile amortises the loads - the one activation value and the EB weight values
// a lane loads per 32-lane k-chunk feed EB FMAs, so the token row is read once
// instead of n_expert times - but the tile is bounded from above by the register
// file, and on this part the cliff is sharp: IGC keeps this kernel at
// 128 GRF/thread (4 hardware threads per EU, so 16 accs is already most of the
// latency-hiding budget), and `.spill size` goes 0 at
// 16 accumulators to 4224 B at 32 (8x4: 1.65 -> 3.63 ms) and 14208 B at 64
// (8x8).  Measured per layer at 803 tokens (one process per cell):
//
//   1x16 (expert-major) 1.13 ms   2x8 1.26   4x4 1.65   8x4 3.63   8x8 4.79
//   1x8 1.47            1x32 2.04  2x16 1.45  16x4 12.5
//
// so 1x16 is what ships for a prefill batch: it holds 16 experts' partial sums
// in registers for one token (17 loads per 16 FMAs, and the 17 loads in flight
// per step are what hides the L1/L2 latency at this occupancy), and the
// expert-major tile order makes those 16 rows (128 KB) L1-resident across the
// consecutive token tiles that reuse them.  A wider tile can only come from
// more accumulators, and 32 already spills (see above).
//
// Two structures were built and measured as *slower* on this part and are not
// here: an SLM-staged work-group tile (16x16, 512 threads, 2.15-3.7 ms across
// BK 64..512 - the staging loop plus the per-chunk barriers cost more than the
// L2 traffic they save, and the larger work-groups spill), and a
// bandwidth-optimal 8x8 register tile (traffic 4x lower, 0.77 ms of loads in
// isolation, but 14 KB of spill makes it 4.79 ms).
//
// THE SUMMATION ORDER IS THE CONTRACT.  Lanes still tile K in the GEMV's exact
// split (lane l owns k = lane + 32*r, r ascending) and each output is reduced
// by the same 32-lane butterfly from lane 0, so every logit is BIT-IDENTICAL to
// moe_router_logits - not "within an ulp".  The MoE's discrete top-8 turns a
// one-ulp logit difference into a different expert set, so PF_MOE_RT_CHK runs
// both kernels over the same input and compares the raw logits.
template <uint32_t RT, int EB>
static sycl::event moe_router_logits_tiled(queue & q, const void * gate_inp, int esize, const float * x,
                                           int x_stride, int n_tok, int n_embd, int n_expert, float * logits) {
    // BT is 1: a tile is one token x EB experts, so every k of the token's row
    // is read once and reused by EB FMAs.  Requires n_expert % EB == 0, which
    // the launcher guarantees - with the tile always full there is no per-load
    // guard and no pointer array.  That matters: an array of EB weight pointers
    // (or a clamped row index per expert) spills this kernel by 2304 B and
    // measures 4.7x slower (5.4 ms against 1.13 at 803 tokens).
    const int n_et = n_expert / EB;
    const int n_tiles = n_tok * n_et;
    // Sub-groups per work-group.  Tiles are laid out expert-tile-fastest, so a
    // work-group's four sub-groups are four expert tiles of the SAME token -
    // the token row they share is one L1/L2 hit, not four - while consecutive
    // work-groups walk the token axis for a fixed expert block, which keeps
    // those EB weight rows (EB*8 KB = 128 KB) L1-resident.
    constexpr int GW = 4;
    const size_t n_wg = ((size_t)n_tiles + GW - 1) / GW;
    const size_t rowb = (size_t)n_embd * (size_t)esize;
    return q.parallel_for(nd_range<1>(n_wg * (GW * 32), GW * 32),
                          [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int lane = (int)(it.get_local_id(0) & 31);
        const int sgi = (int)(it.get_local_id(0) >> 5);
        const sub_group sgg = it.get_sub_group();
        const size_t tile = it.get_group(0) * GW + (size_t)sgi;
        if (tile >= (size_t)n_tiles) {
            return; // sub-group-uniform: every lane of this sub-group exits
        }
        const int t = (int)(tile % (size_t)n_tok); // expert-major
        const int e0 = (int)(tile / (size_t)n_tok) * EB;
        const float * xb = x + (size_t)t * x_stride;
        const char * wb = (const char *)gate_inp + (size_t)e0 * rowb;
        float acc[EB];
#pragma unroll
        for (int j = 0; j < EB; j++) {
            acc[j] = 0.f;
        }
        // r ascending over the K axis is the GEMV's per-lane term order:
        // lane l owns k = lane + 32*r, exactly as in moe_router_logits.
        const int nr = n_embd / 32;
        for (int r = 0; r < nr; r++) {
            const int k = lane + 32 * r;
            const float xv = xb[k];
            float wv[EB];
#pragma unroll
            for (int j = 0; j < EB; j++) {
                wv[j] = moe_rd<RT>(wb + rowb * (size_t)j, k);
            }
#pragma unroll
            for (int j = 0; j < EB; j++) {
                acc[j] += xv * wv[j];
            }
        }
        // The same butterfly as the GEMV, and lane 0's value is the reference
        // one (a butterfly's lanes each add in a different order), so the store
        // is lane 0 only - exactly as above.
#pragma unroll
        for (int j = 0; j < EB; j++) {
            const float v = sg_sum(acc[j], sgg);
            if (lane == 0) {
                logits[(size_t)t * n_expert + (e0 + j)] = v;
            }
        }
    });
}

// Tile-shape dispatch.  Only shapes that were measured are instantiated; the
// row count picks the one that won at that row count (the crossover between
// them is flat, so 64 rows is the switch):
//
//   n_tok >= 64  1x16   (0.373 T-MAC/s at 803; 0.365 at 1024)
//   n_tok >= 16  1x8    (0.277 at 128 - the 16-wide tile has fewer tiles than
//                        the part has warp slots until ~64 rows)
//   n_tok < 16   the GEMV, which is faster there (256 warps: 0.063 ms at n=8
//                        against 0.101 for the tile)
//
// EB must divide n_expert (the kernel has no partial-tile path); the launcher
// picks a divisor and falls back to the GEMV when there is none.
template <uint32_t RT>
static sycl::event moe_router_logits_tiled_dispatch(queue & q, const void * gate_inp, int esize, const float * x,
                                                    int x_stride, int n_tok, int n_embd, int n_expert, int eb,
                                                    float * logits) {
    if (eb >= 16) {
        return moe_router_logits_tiled<RT, 16>(q, gate_inp, esize, x, x_stride, n_tok, n_embd, n_expert, logits);
    }
    if (eb >= 8) {
        return moe_router_logits_tiled<RT, 8>(q, gate_inp, esize, x, x_stride, n_tok, n_embd, n_expert, logits);
    }
    return moe_router_logits_tiled<RT, 4>(q, gate_inp, esize, x, x_stride, n_tok, n_embd, n_expert, logits);
}

// One entry point for both logits forms, so the exactness check can run the one
// that was NOT selected over the same input.
template <uint32_t RT>
static sycl::event moe_router_logits_any(queue & q, bool gemv, int eb, const void * gate_inp, int esize,
                                         const float * x, int x_stride, int n_tok, int n_embd, int n_expert,
                                         float * logits) {
    if (gemv) {
        return moe_router_logits<RT>(q, gate_inp, esize, x, x_stride, n_tok, n_embd, n_expert, logits);
    }
    return moe_router_logits_tiled_dispatch<RT>(q, gate_inp, esize, x, x_stride, n_tok, n_embd, n_expert, eb, logits);
}

// The register-blocked tiling of the above (one 32-lane sub-group per
// `MT tokens x EB experts` tile, lanes tiling K) was DELETED, not left behind a
// switch.  It was 2.1x faster in isolation - 45.5 -> 20.5 ms of a ~1914 ms prefill -
// and it was also WRONG, in two separate ways that a benchmark cannot see:
//
//   * its `sg_sum` cross-lane step had no `reqd_sub_group_size(32)`, so on a
//     device whose sub-group width is not 32 a 32-thread work-group is TWO
//     sub-groups and several lane-0s raced on the same `logits[t][e]`.  That made
//     the whole engine's output vary between processes: 13 of 13 runs of the same
//     binary produced a different prefill argmax, with the top1-top2 margin
//     ranging 0.04 to 1.28.  Adding the attribute makes it deterministic (4/4
//     identical) but does NOT make it right.
//   * with the attribute it still computes a different logit vector than the GEMV
//     form above: pp512's argmax is 11171 against the established 4821, and it is
//     SLOWER - 751.0 against 772.7 t/s prefill, and 49.1 against 38.1 ms/token on
//     a depth-0 decode (1.29x).  The speedup was real; it was buying a wrong
//     answer, and the MoE's discrete top-8 selection turned a one-ulp difference
//     into a different logit vector rather than a smaller error.
//
// The lesson is the one this file keeps repeating: a bisection built on an
// instrument that cannot see the bug finds nothing, and a speedup measured on a
// kernel whose output was never checked against a reference is not a speedup.
// `dev/grp_rout` HAD reported that the tiled and GEMV forms disagreed on 512 of
// 512 tokens' expert sets; that was filed as "probably a pre-existing defect in
// the reference" and not acted on, and it was this.

// softmax over the n_expert logits of one token, in place
static sycl::event moe_router_softmax(queue & q, float * logits, int n_tok, int n_expert) {
    return q.parallel_for(range<1>((size_t)n_tok), [=](id<1> i) {
        float * p = logits + (size_t) i[0] * n_expert;
        float mx = -3.402823466e38f;
        for (int e = 0; e < n_expert; e++) {
            mx = sycl::fmax(mx, p[e]);
        }
        float sum = 0.f;
        for (int e = 0; e < n_expert; e++) {
            p[e] = sycl::exp(p[e] - mx);
            sum += p[e];
        }
        const float inv = 1.f / sum;
        for (int e = 0; e < n_expert; e++) {
            p[e] *= inv;
        }
    });
}

// Top-k by rank, not by k rounds of argmax: an expert's rank is how many
// experts beat it (higher probability, or equal probability and a lower index -
// the reference argsort's tie rule) and it claims slot `rank` if that is below
// top_k.  Ranks are a permutation of 0..n_expert-1 because ties break on the
// index, so exactly one work-item writes each slot and no coordination is needed.
//
// The rank loop is O(n_expert^2), and it used to run one *work-item* per token -
// so a decode's 256 experts were ranked by a single thread doing 65536
// iterations, twice over (ids and weights).  Measured on the 35B A3B, the router
// - which is 0.5 M-MAC of real work - was 4.4 s of a 4.6 s decode.  It is now
// one work-item per (token, expert), i.e. n_expert-way parallel, and the weight
// pass reads the ids back instead of re-deriving them.
template <bool WANT_IDS>
static sycl::event moe_router_topk(queue & q, const float * probs, int n_tok, int n_expert, int top_k, void * out,
                                   const int32_t * ids_in) {
    if (WANT_IDS) {
        return q.parallel_for(range<2>((size_t)n_tok, (size_t)n_expert), [=](id<2> i) {
            const int t = (int) i[0], e = (int) i[1];
            const float * p = probs + (size_t) t * n_expert;
            const float pe = p[e];
            int rank = 0;
            for (int f = 0; f < n_expert; f++) {
                const float pf = p[f];
                if (pf > pe || (pf == pe && f < e)) {
                    rank++;
                }
            }
            if (rank < top_k) {
                ((int32_t *) out)[(size_t) t * top_k + rank] = e;
            }
        });
    } else {
        // the ids are already known, so this is O(top_k) per token: read the
        // selected probabilities, normalise by their clamped sum, and write.
        return q.parallel_for(range<1>((size_t)n_tok), [=](id<1> i) {
            const size_t t = (size_t) i[0];
            const float * p = probs + t * n_expert;
            float * wt = (float *) out + t * top_k;
            const int32_t * id = ids_in + t * top_k;
            float sum = 0.f;
            for (int j = 0; j < top_k; j++) {
                const float v = p[id[j]];
                wt[j] = v;
                sum += v;
            }
            const float den = sycl::fmax(sum, 6.103515625e-5f);
            for (int j = 0; j < top_k; j++) {
                wt[j] /= den;
            }
        });
    }
}

// Fused softmax + top-k + router weights: ONE workgroup per token, ONE launch
// instead of the three kernels above.
//
// WHY.  On a decode (n_tok=1) the sequential softmax kernel - a single
// work-item running 512 exp() in a dependent loop - measures 0.186 ms of a
// 0.269 ms router, i.e. 69% of it, against 0.019 ms for the 5.24 MB dot
// (PF_MOE_RT_TIME; the dot reads at 276 GB/s, the softmax at one lane).  Two
// more launches cost two more command submissions.  The token->expert choice
// is on every layer's critical path, so a 43-layer decode pays 43 x 0.25 ms.
//
// WHAT IS PRESERVED BIT-FOR-BIT (the router feeds the LRU and a one-ulp
// difference in a logit has already been shown to flip a near-tie):
//   * the max is fmax, which is exact in any order;
//   * the exp is the same sycl::exp on the same (logit - max);
//   * the SUM IS SEQUENTIAL IN INDEX ORDER by lane 0 (512 adds are ~1 us),
//     so `1/sum` is the old kernel's value to the last bit;
//   * the normalized p[e] = exp*inv is the same product;
//   * the top-k rank scan compares the normalized values in the same order
//     and breaks ties to the lower index;
//   * the weights sum in the same j order and divide by the same clamped den.
// Only the *parallel* part (the exp, the rank scan) moved to 256 lanes.
//
// The SLM is 2 KB of logits/probs plus the reductions; the barriers are for
// local-memory visibility only.  The id/weight buffers are the caller's (a
// pinned host-USM buffer on the decode path), so this does not repeat the
// one-workgroup-per-token router that never took effect on this backend - that
// one kept the ids in SLM and had no sub-group width; this one writes through
// the same global pointers the three kernels did.
//
// Measured (qwen4exp, n_expert 512, top_k 10, one A770): the three sequential
// kernels are 0.269 ms of a 0.292 ms router on device (logits 0.019, softmax
// 0.186, topk 0.055, wts 0.009) and the fused kernel is 0.040 ms, so the
// router's device time falls ~4.5x.  Its remaining cost is a fixed ~40 us for
// the single work-group (a 32/128/256/512-thread group all measure the same),
// i.e. the work itself, not its organisation.  PF_MOE_RT_CHK proves the ids,
// weights and normalized logits are bit-identical to the three-kernel form.
template <int WG>
static sycl::event moe_router_softmax_topk(queue & q, float * logits, int n_tok, int n_expert, int top_k,
                                           int32_t * ids, float * wts) {
    constexpr int NSG = WG / 32;
    // PF_MOE_RT_RANK=1 keeps the O(n_expert^2) rank scan (a lane per expert,
    // each counting how many beat it) for the A/B.  Default: k rounds of
    // (max, ties to the lower index) over the whole row - the same ordering by
    // construction, but O(k * n_expert / WG + k * log) instead of n_expert
    // sequential steps on one EU.  On this model (n_expert 512, top_k 10) the
    // rank scan is 512 steps of 16 warps on a single EU and measures 0.079 ms;
    // it is 3.4 ms/token across the 43 banked layers.
    const bool rank_scan = si::env::flag("PF_MOE_RT_RANK");
    // Phase ablation for the device-time probe (wrong results by design; the
    // outputs stay in range so the engine keeps running).  This is how the
    // fused kernel's ~40 us serial chain is attributed: 1 = skip exp,
    // 2 = skip the p[] writeback, 3 = skip the top-k selection,
    // 5 = skip the sequential normaliser sum, 6 = skip the 1/sum multiply.
    // Function-scope statics (read once) copied into locals: a `static` cannot
    // be referenced from inside the SYCL kernel lambda, but the env read must
    // not happen per layer either.
    static const int abl_c = si::env::i32("PF_MOE_RT_ABL", 0);
    const int abl = abl_c;
    // PF_MOE_RT_FAST (default ON) is the shipped switch: batch the fused
    // kernel's dependent shared-memory loads into registers (the top-k scan,
    // the cross-warp max fold and the weight-sum tail).  Every comparison runs
    // in the original order with the original tie rule, so ids/wts/logits are
    // bit-identical; only the SLM load latency overlaps.  Measured on qwen4exp
    // (512 experts, top_k 10, one A770, alternating A/B 3 runs each): fused
    // kernel 0.0422 -> 0.0377 ms/layer (-10.7%), router device time
    // 0.0627 -> 0.0582 ms/layer, generated text byte-identical.
    static const bool fast_c = si::env::i32("PF_MOE_RT_FAST", 1) != 0;
    const bool fast = fast_c;
    return q.submit([&](handler & h) {
        local_accessor<float, 1> pl((size_t)n_expert, h);
        local_accessor<float, 1> red((size_t)NSG + 1, h);
        local_accessor<float, 1> wl((size_t)top_k, h);
        h.parallel_for(nd_range<1>((size_t)n_tok * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int t = (int)it.get_group(0);
            const int tid = (int)it.get_local_id(0);
            const int lane = tid & 31;
            const int sg = tid >> 5;
            const sub_group sgg = it.get_sub_group();
            float * __restrict__ p = logits + (size_t)t * n_expert;
            // max over the row: fmax is exact, so the order is free
            float mx = -3.402823466e38f;
            for (int e = tid; e < n_expert; e += WG) {
                mx = sycl::fmax(mx, p[e]);
            }
            mx = sg_max(mx, sgg);
            if (lane == 0) {
                red[sg] = mx;
            }
            it.barrier();
            mx = red[0];
            if (fast) {
                // same fmax sequence (fmax is exact and commutative), loads
                // hoisted 8 at a time so the cross-warp slots overlap
                int s = 1;
                for (; s + 7 < NSG; s += 8) {
                    const float a0 = red[s], a1 = red[s + 1], a2 = red[s + 2], a3 = red[s + 3];
                    const float a4 = red[s + 4], a5 = red[s + 5], a6 = red[s + 6], a7 = red[s + 7];
                    mx = sycl::fmax(mx, a0);
                    mx = sycl::fmax(mx, a1);
                    mx = sycl::fmax(mx, a2);
                    mx = sycl::fmax(mx, a3);
                    mx = sycl::fmax(mx, a4);
                    mx = sycl::fmax(mx, a5);
                    mx = sycl::fmax(mx, a6);
                    mx = sycl::fmax(mx, a7);
                }
                for (; s < NSG; s++) {
                    mx = sycl::fmax(mx, red[s]);
                }
            } else {
                for (int s = 1; s < NSG; s++) {
                    mx = sycl::fmax(mx, red[s]);
                }
            }
            // exp in parallel, kept in SLM (and written back for compatibility)
            for (int e = tid; e < n_expert; e += WG) {
                pl[e] = (abl == 1) ? p[e] : sycl::exp(p[e] - mx);
            }
            it.barrier();
            // the reference's sum: sequential, ascending index, on one lane.
            // Batched 8 loads then 8 dependent adds: the adds keep the
            // reference's order exactly (so 1/sum is bit-identical) while the
            // loads overlap instead of stalling the chain every iteration.
            float inv;
            {
                if (abl != 5 && tid == 0) {
                    float sum = 0.f;
                    int e = 0;
                    for (; e + 8 <= n_expert; e += 8) {
                        const float v0 = pl[e + 0], v1 = pl[e + 1], v2 = pl[e + 2], v3 = pl[e + 3];
                        const float v4 = pl[e + 4], v5 = pl[e + 5], v6 = pl[e + 6], v7 = pl[e + 7];
                        sum += v0;
                        sum += v1;
                        sum += v2;
                        sum += v3;
                        sum += v4;
                        sum += v5;
                        sum += v6;
                        sum += v7;
                    }
                    for (; e < n_expert; e++) {
                        sum += pl[e];
                    }
                    red[NSG] = sum;
                }
                it.barrier();
                inv = (abl == 5) ? 1.f : 1.f / red[NSG];
            }
            for (int e = tid; e < n_expert; e += WG) {
                if (abl != 6) {
                    pl[e] *= inv;
                }
                if (abl != 2) {
                    p[e] = pl[e];
                }
            }
            it.barrier();
            if (abl == 3) {
                // valid ids/weights, no selection work (timing only)
                if (tid < top_k) {
                    ids[(size_t)t * top_k + tid] = tid;
                    wl[tid] = pl[tid];
                }
            } else if (rank_scan) {
                // top-k by rank, exactly as moe_router_topk<true>
                for (int e = tid; e < n_expert; e += WG) {
                    const float pe = pl[e];
                    int rank = 0;
                    for (int f = 0; f < n_expert; f++) {
                        const float pf = pl[f];
                        if (pf > pe || (pf == pe && f < e)) {
                            rank++;
                        }
                    }
                    if (rank < top_k) {
                        ids[(size_t)t * top_k + rank] = e;
                    }
                }
                it.barrier();
                // weights, exactly as moe_router_topk<false>
                if (tid < top_k) {
                    wl[tid] = pl[ids[(size_t)t * top_k + tid]];
                }
            } else {
                // k rounds of (max, ties to the lower index).  The k-th round
                // writes rank k, so the order is the rank scan's by
                // construction.  A selected entry is retired with -FLT_MAX,
                // which no normalized probability can equal.
                //
                // The whole selection runs on ONE warp.  The WG-wide form
                // needs two `it.barrier()`s per round (eight warps of a
                // 256-thread group, 20 barriers for top_k=10) and measured
                // 0.057-0.058 ms; a warp's own shuffle reduce needs only a
                // sub-group fence for the retire and measures 0.040-0.041 ms.
                if (tid < 32) {
                    for (int r = 0; r < top_k; r++) {
                        float best = -3.402823466e38f;
                        int bi = n_expert;
                        int e = lane;
                        // PF_MOE_RT_FAST: hoist the round's independent SLM
                        // loads into registers, 8 at a time, before the
                        // dependent `best`/`bi` chain consumes them.  The
                        // compares run in exactly the original order (lane's
                        // elements, ascending e, same tie rule), so the ids
                        // and weights are BIT-IDENTICAL to the serial form -
                        // only the load latency is overlapped.
                        if (fast) {
                            for (; e + 32 * 7 < n_expert; e += 32 * 8) {
                                const float v0 = pl[e], v1 = pl[e + 32], v2 = pl[e + 64], v3 = pl[e + 96];
                                const float v4 = pl[e + 128], v5 = pl[e + 160], v6 = pl[e + 192], v7 = pl[e + 224];
                                if (v0 > best || (v0 == best && e < bi)) { best = v0; bi = e; }
                                if (v1 > best || (v1 == best && e + 32 < bi)) { best = v1; bi = e + 32; }
                                if (v2 > best || (v2 == best && e + 64 < bi)) { best = v2; bi = e + 64; }
                                if (v3 > best || (v3 == best && e + 96 < bi)) { best = v3; bi = e + 96; }
                                if (v4 > best || (v4 == best && e + 128 < bi)) { best = v4; bi = e + 128; }
                                if (v5 > best || (v5 == best && e + 160 < bi)) { best = v5; bi = e + 160; }
                                if (v6 > best || (v6 == best && e + 192 < bi)) { best = v6; bi = e + 192; }
                                if (v7 > best || (v7 == best && e + 224 < bi)) { best = v7; bi = e + 224; }
                            }
                        }
                        for (; e < n_expert; e += 32) {
                            const float v = pl[e];
                            if (v > best || (v == best && e < bi)) {
                                best = v;
                                bi = e;
                            }
                        }
                        if (abl == 8) {
                            // timing only: scan only, no cross-lane merge/retire
                            if (lane == 0) {
                                ids[(size_t)t * top_k + r] = bi;
                                wl[r] = best;
                            }
                        } else {
                            for (int mask = 16; mask > 0; mask >>= 1) {
                                const float ov = sycl::permute_group_by_xor(sgg, best, mask);
                                const int oi = sycl::permute_group_by_xor(sgg, bi, mask);
                                if (ov > best || (ov == best && oi < bi)) {
                                    best = ov;
                                    bi = oi;
                                }
                            }
                            if (abl == 9) {
                                // timing only: scan + tree, no retire/barrier
                                if (lane == 0) {
                                    ids[(size_t)t * top_k + r] = bi;
                                    wl[r] = best;
                                }
                            } else {
                                if (lane == 0) {
                                    ids[(size_t)t * top_k + r] = bi;
                                    wl[r] = best;
                                    pl[bi] = -3.402823466e38f;
                                }
                                sycl::group_barrier(sgg);
                            }
                        }
                    }
                }
            }
            it.barrier();
            if (tid == 0) {
                float sum = 0.f;
                int j = 0;
                if (fast) {
                    for (; j + 7 < top_k; j += 8) {
                        const float a0 = wl[j], a1 = wl[j + 1], a2 = wl[j + 2], a3 = wl[j + 3];
                        const float a4 = wl[j + 4], a5 = wl[j + 5], a6 = wl[j + 6], a7 = wl[j + 7];
                        sum += a0;
                        sum += a1;
                        sum += a2;
                        sum += a3;
                        sum += a4;
                        sum += a5;
                        sum += a6;
                        sum += a7;
                    }
                }
                for (; j < top_k; j++) {
                    sum += wl[j];
                }
                red[NSG] = sycl::fmax(sum, 6.103515625e-5f);
            }
            it.barrier();
            if (tid < top_k) {
                wts[(size_t)t * top_k + tid] = wl[tid] / red[NSG];
            }
        });
    });
}

void moe_router_launch(queue & q, const void * gate_inp, uint32_t gp_type, const float * x, int x_stride, int n_tok,
                       int n_embd, int n_expert, int top_k, float * logits, int32_t * ids, float * wts) {
    // PF_MOE_RT_TIME: the kernels' own device time from event profiling.
    // Diagnostic only (it adds one wait at the end); the per-kernel spacings
    // are what tell a logits-bound router from a softmax/top-k or a
    // launch-overhead one.  It reports the three-kernel form under
    // PF_MOE_RT_OLD and the logits + fused pair otherwise.
    const bool rt_time = si::env::flag("PF_MOE_RT_TIME");
    // The logits kernel: the tiled form is the default, PF_MOE_ROUTER=0 forces
    // the original one-sub-group-per-(token, expert) GEMV for the A/B.
    // PF_MOE_ROUTER_EB overrides the expert tile the row count picks.
    //
    // The tile is 1 x EB (see the measured table above the kernel): a prefill
    // batch uses 16 experts, a shorter one 8 (a tile has one sub-group per
    // expert block, so 16 tokens x 16 tiles is fewer warps than a 64-EU part can
    // run until ~64 rows), and < 16 rows keep the GEMV - its 256 warps beat the
    // tile's 4-16 at launch-overhead scale (M=8: 0.063 against 0.101 ms; M=16
    // is where the tile takes over: 0.065 against 0.084).
    // EB is capped by the expert count so a model with few experts does not run
    // a mostly empty tile.
    static const int rt_mode = si::env::i32("PF_MOE_ROUTER", 1);
    static const int rt_eb = si::env::i32("PF_MOE_ROUTER_EB", 0);
    int eb = rt_eb > 0 ? rt_eb : (n_tok >= 64 ? 16 : 8);
    if (eb > n_expert) {
        eb = n_expert;
    }
    // EB must divide n_expert so a tile is always full; walk down to the
    // largest divisor of the {16,8,4} set (0 = none, use the GEMV).
    while (eb > 1 && (n_expert % eb) != 0) {
        eb >>= 1;
    }
    if (eb < 4 || (n_expert % eb) != 0) {
        eb = 0;
    }
    const bool rt_gemv = rt_mode == 0 || n_tok < 16 || eb == 0 || (n_embd % 32) != 0;
    sycl::event e0;
#define SI_MOE_R(GP, ESZ)                                                                                              \
    e0 = moe_router_logits_any<GP>(q, rt_gemv, eb, gate_inp, ESZ, x, x_stride, n_tok, n_embd, n_expert, logits)
    switch (gp_type) {
    case GGML_TYPE_F32: SI_MOE_R(GGML_TYPE_F32, 4); break;
    case GGML_TYPE_F16: SI_MOE_R(GGML_TYPE_F16, 2); break;
    case GGML_TYPE_BF16: SI_MOE_R(GGML_TYPE_BF16, 2); break;
    default:
        throw std::runtime_error("moe: ffn_gate_inp must be f32/f16/bf16 (got " + std::to_string(gp_type) + ")");
    }
#undef SI_MOE_R
    // PF_MOE_RT_OLD=1 restores the three sequential kernels for the A/B;
    // PF_MOE_RT_CHK=1 runs both on the caller's real input once and compares
    // ids/wts/logits bit-for-bit (the router's discrete choice is load-bearing,
    // so "looks coherent" is not the bar).
    static const bool rt_old = si::env::flag("PF_MOE_RT_OLD");
    static const bool rt_chk = si::env::flag("PF_MOE_RT_CHK");
    // The check waits on the kernels' events, which is illegal while a command
    // graph is being recorded (the generic path records a decode graph at
    // startup with n_tok=1).  So it runs on the prefill shapes by default; a
    // direct-replay engine (qwen4exp) takes PF_MOE_RT_CHK_DEC=1 for n_tok=1.
    static const bool chk_dec = si::env::flag("PF_MOE_RT_CHK_DEC");
    static int chk_last_n = -1;
    const bool do_chk = rt_chk && chk_last_n != n_tok && (n_tok > 1 || chk_dec);
    // PF_MOE_RT_WG forces the fused kernel's work-group width (A/B knob; the
    // default splits the row into at most 512 lanes).
    static const int rt_wg = si::env::i32("PF_MOE_RT_WG", 0);
    // Resolve the width ONCE: the launch and the [moert] report must not be able
    // to disagree about the work-group the kernel actually ran.
    const int fwg = (rt_wg == 32) ? 32
                    : (rt_wg == 128) ? 128
                    : (rt_wg == 256 || (rt_wg != 512 && n_expert <= 256)) ? 256
                                                                          : 512;
    auto fused = [&]() -> sycl::event {
        if (fwg == 32) {
            return moe_router_softmax_topk<32>(q, logits, n_tok, n_expert, top_k, ids, wts);
        }
        if (fwg == 128) {
            return moe_router_softmax_topk<128>(q, logits, n_tok, n_expert, top_k, ids, wts);
        }
        if (fwg == 256) {
            return moe_router_softmax_topk<256>(q, logits, n_tok, n_expert, top_k, ids, wts);
        }
        return moe_router_softmax_topk<512>(q, logits, n_tok, n_expert, top_k, ids, wts);
    };
    sycl::event e1, e2, e3, ef;
    if (!rt_old && !do_chk) {
        ef = fused();
    } else if (rt_old) {
        e1 = moe_router_softmax(q, logits, n_tok, n_expert);
        e2 = moe_router_topk<true>(q, logits, n_tok, n_expert, top_k, ids, nullptr);
        e3 = moe_router_topk<false>(q, logits, n_tok, n_expert, top_k, wts, ids);
    }
    if (do_chk) {
        chk_last_n = n_tok;
        std::vector<float> h_log_raw((size_t)n_tok * n_expert);
        q.memcpy(h_log_raw.data(), logits, h_log_raw.size() * sizeof(float)).wait();
        // The OTHER logits form over the same input.  The tiled kernel must
        // reproduce the GEMV's logits BIT-FOR-BIT (it keeps the per-lane term
        // order and the 32-lane butterfly), not merely the same top-k: the
        // MoE's discrete selection is where a one-ulp difference shows up.
        {
            std::vector<float> h_log_alt(h_log_raw.size());
            sycl::event ealt;
#define SI_MOE_RALT(GP, ESZ)                                                                                           \
    ealt = moe_router_logits_any<GP>(q, !rt_gemv, eb, gate_inp, ESZ, x, x_stride, n_tok, n_embd, n_expert, logits)
            switch (gp_type) {
            case GGML_TYPE_F32: SI_MOE_RALT(GGML_TYPE_F32, 4); break;
            case GGML_TYPE_F16: SI_MOE_RALT(GGML_TYPE_F16, 2); break;
            case GGML_TYPE_BF16: SI_MOE_RALT(GGML_TYPE_BF16, 2); break;
            default: break;
            }
#undef SI_MOE_RALT
            ealt.wait();
            q.memcpy(h_log_alt.data(), logits, h_log_alt.size() * sizeof(float)).wait();
            int alt_bad = 0;
            float alt_max = 0.f;
            for (size_t i = 0; i < h_log_alt.size(); i++) {
                const float d = std::fabs(h_log_raw[i] - h_log_alt[i]);
                alt_max = std::fmax(alt_max, d);
                alt_bad += !(h_log_raw[i] == h_log_alt[i]);
            }
            fprintf(stderr, "[moertchk] logits %s vs %s: bad=%d/%zu maxdiff=%.3g\n", rt_gemv ? "gemv" : "tiled",
                    rt_gemv ? "tiled" : "gemv", alt_bad, h_log_alt.size(), (double)alt_max);
            // the selected path's raw logits are what the comparisons below run on
            q.memcpy(logits, h_log_raw.data(), h_log_raw.size() * sizeof(float)).wait();
        }
        e1 = moe_router_softmax(q, logits, n_tok, n_expert);
        e2 = moe_router_topk<true>(q, logits, n_tok, n_expert, top_k, ids, nullptr);
        e3 = moe_router_topk<false>(q, logits, n_tok, n_expert, top_k, wts, ids);
        e3.wait();
        std::vector<int32_t> h_ids_old((size_t)n_tok * top_k);
        std::vector<float> h_wts_old((size_t)n_tok * top_k);
        std::vector<float> h_log_old((size_t)n_tok * n_expert);
        q.memcpy(h_ids_old.data(), ids, h_ids_old.size() * sizeof(int32_t)).wait();
        q.memcpy(h_wts_old.data(), wts, h_wts_old.size() * sizeof(float)).wait();
        q.memcpy(h_log_old.data(), logits, h_log_old.size() * sizeof(float)).wait();
        // restore the raw logits, then rerun the fused path into the same
        // buffers and compare
        q.memcpy(logits, h_log_raw.data(), h_log_raw.size() * sizeof(float)).wait();
        fused().wait();
        std::vector<int32_t> h_ids_new((size_t)n_tok * top_k);
        std::vector<float> h_wts_new((size_t)n_tok * top_k);
        std::vector<float> h_log_new((size_t)n_tok * n_expert);
        q.memcpy(h_ids_new.data(), ids, h_ids_new.size() * sizeof(int32_t)).wait();
        q.memcpy(h_wts_new.data(), wts, h_wts_new.size() * sizeof(float)).wait();
        q.memcpy(h_log_new.data(), logits, h_log_new.size() * sizeof(float)).wait();
        int id_bad = 0, w_bad = 0, l_bad = 0;
        float lmax = 0.f;
        for (size_t i = 0; i < h_ids_old.size(); i++) {
            id_bad += h_ids_old[i] != h_ids_new[i];
        }
        for (size_t i = 0; i < h_wts_old.size(); i++) {
            w_bad += !(h_wts_old[i] == h_wts_new[i]);
        }
        for (size_t i = 0; i < h_log_old.size(); i++) {
            const float d = std::fabs(h_log_old[i] - h_log_new[i]);
            lmax = std::fmax(lmax, d);
            l_bad += !(h_log_old[i] == h_log_new[i]);
        }
        fprintf(stderr, "[moertchk] n=%d ne=%d ids_bad=%d/%zu wts_bad=%d/%zu logits_bad=%d maxdiff=%.3g\n", n_tok,
                n_expert, id_bad, h_ids_old.size(), w_bad, h_wts_old.size(), l_bad, (double)lmax);
    }
    if (rt_time && !rt_old && !do_chk) {
        ef.wait();
        const double ts = (double)ef.get_profiling_info<sycl::info::event_profiling::command_start>() / 1.0e6;
        const double te = (double)ef.get_profiling_info<sycl::info::event_profiling::command_end>() / 1.0e6;
        const double t0 = (double)e0.get_profiling_info<sycl::info::event_profiling::command_end>() / 1.0e6;
        // Grid report (measurement aid): the logits GEMV launches one 32-lane
        // sub-group per (token, expert), so a decode is `n_expert` work-groups
        // of one warp; the fused softmax/top-k launches one work-group per
        // token.  Print both with the card's compute-unit count so the wave
        // occupancy is arithmetically visible in the same line as the device
        // times.
        static const int cu = [&q] {
            try {
                return (int)q.get_device().get_info<sycl::info::device::max_compute_units>();
            } catch (...) {
                return 0;
            }
        }();
        const size_t lgroups = (size_t)n_tok * (size_t)n_expert;
        fprintf(stderr,
                "[moert] n=%d ne=%d wg=%d cu=%d lgroups=%zu lwarps=%zu fgroups=%d fwg=%d fwarps=%d "
                "logits=%.4f fused=%.4f gap=%.4f ms\n",
                n_tok, n_expert, fwg, cu, lgroups, lgroups, n_tok, fwg,
                (int)((size_t)n_tok * (size_t)(fwg / 32)),
                t0 - (double)e0.get_profiling_info<sycl::info::event_profiling::command_start>() / 1.0e6, te - ts,
                ts - t0);
    }
    if (rt_time && (rt_old || do_chk)) {
        e3.wait();
        auto ms = [](const sycl::event & e) {
            return (double)(e.get_profiling_info<sycl::info::event_profiling::command_end>()
                            - e.get_profiling_info<sycl::info::event_profiling::command_start>())
                   / 1.0e6;
        };
        auto st = [](const sycl::event & e) {
            return (double)e.get_profiling_info<sycl::info::event_profiling::command_start>() / 1.0e6;
        };
        fprintf(stderr, "[moert] n=%d ne=%d logits=%.3f softmax=%.3f topk=%.3f wts=%.3f gap01=%.3f gap12=%.3f gap23=%.3f ms\n",
                n_tok, n_expert, ms(e0), ms(e1), ms(e2), ms(e3), st(e1) - (st(e0) + ms(e0)),
                st(e2) - (st(e1) + ms(e1)), st(e3) - (st(e2) + ms(e2)));
    }
}

// ---------------------------------------------------------------------------
// token -> expert CSR: clear, count, prefix+active, scatter.  Four tiny device
// launches, so the whole selection stays on the device and survives command
// graph capture (the router's choice is made on the device every step).
void moe_csr_clear_launch(queue & q, int32_t * off, int32_t * cur, int32_t * active, int32_t * n_active, int n_expert) {
    q.parallel_for(range<1>((size_t)n_expert + 1), [=](id<1> i) { off[i[0]] = 0; });
    q.parallel_for(range<1>((size_t)n_expert), [=](id<1> i) { cur[i[0]] = 0; });
    q.parallel_for(range<1>(1), [=](id<1>) {
        active[0] = 0;
        n_active[0] = 0;
    });
}

void moe_csr_count_launch(queue & q, const int32_t * ids, int n_pairs, int32_t * off) {
    q.parallel_for(range<1>((size_t)n_pairs), [=](id<1> i) {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space> a(off[ids[i[0]] + 1]);
        a.fetch_add(1);
    });
}

// Exclusive prefix sum over n_expert plus the list of experts with at least one
// token.  One work-item: n_expert is a few hundred and this is one pass, not per
// pair.  `off[n_expert]` is written as the total, because the expert kernels
// read `off[e+1]` for the last expert.
void moe_csr_prefix_launch(queue & q, int32_t * off, int32_t * active, int32_t * n_active, int n_expert) {
    q.parallel_for(range<1>(1), [=](id<1>) {
        int run = 0, na = 0;
        for (int e = 0; e < n_expert; e++) {
            const int c = off[e + 1];
            off[e] = run;
            if (c > 0) {
                active[na++] = e;
            }
            run += c;
        }
        off[n_expert] = run;
        active[na] = -1;
        n_active[0] = na;
    });
}

void moe_csr_scatter_launch(queue & q, const int32_t * ids, int32_t * off, int32_t * cur, int32_t * pair, int n_pairs) {
    q.parallel_for(range<1>((size_t)n_pairs), [=](id<1> i) {
        const int p = (int)i[0];
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space> a(cur[ids[p]]);
        const int s = a.fetch_add(1);
        pair[off[ids[p]] + s] = p;
    });
}

void moe_csr_pos_launch(queue & q, const int32_t * pair, int32_t * pos, int n_pairs) {
    q.parallel_for(range<1>((size_t)n_pairs), [=](id<1> i) {
        const int p = (int)i[0];
        pos[pair[p]] = p;
    });
}

// ===========================================================================
// Grouped dp4a expert GEMM.
//
// WHY THIS EXISTS.  The fp32 expert kernels below are the only GEMMs in the
// engine that still dequantize a 4-bit weight to float and multiply in fp32,
// and on a 512-token prefill of the 35B A3B they are 90% of the whole forward
// (PF_MOE_TIME: gate_up 1302 ms + down 580 ms of 2155 ms) while reading 18 GiB
// of weights - which at the card's measured 376 GB/s is 26 ms.  They are
// neither bandwidth bound nor arithmetic bound: they are bound by
// re-dequantizing the same weight row once per routed pair, a dozen integer
// ops per value against one FMA of useful work.
//
// WHAT IT DOES INSTEAD.  Quantize the layer's activation once to int8 with a
// scale and a group sum per 32 values (the form the dense oneDNN path already
// consumes) and run dp4a against it, reading the expert weights in their native
// GGUF packing.  The nibbles are already what dp4a wants: a Q4_K group is 32
// values in 32 bytes, one nibble each, so a lane takes 4 bytes = one dp4a
// operand with no unpacking at all.  Q5_K adds the one high bit from `qh` with
// two shifts and an OR; Q6_K adds its two-bit plane the same way.
//
// THE LANE SPLIT IS THE WHOLE DESIGN.  Each 8-lane sub-group takes one 32-value
// scale group: lane l owns values 4l..4l+3, i.e. exactly one dp4a operand, and
// 8 lanes x 4 values = the group exactly.  So no lane repeats another's work -
// which is what makes this bandwidth bound rather than dp4a bound - and the
// only cross-lane step is one `sg_sum` per output, once, at the end of the
// tile.  (`nat_gemm` divides the same work 8 ways across its lanes and is
// therefore dp4a bound; that shape suits a dense M<=13 GEMM, not M~31 rows
// spread over 256 experts.)
//
// THE ASYMMETRIC ZERO POINT costs one extra dp4a and nothing else.  A GGUF
// weight is w = dsc*u8 - dmn, so the exact product needs the activation's group
// sum next to its group dot.  Taking that sum over the lane's OWN four values
// (`dp4a_s8u8(x, 0x01010101, 0)` - a multiply by one is a dp4a) rather than over
// the group makes the correction per-lane: no broadcast, no divide by the lane
// count, and correct for Q6_K's 16-value scale group as well as Q4_K's and
// Q5_K's 32-value one.
//
// WORK DECOMPOSITION.  One work-group per (active expert, column block); inside
// it the expert's routed pairs in tiles of M rows, then the K groups.  A weight
// byte is read once per row tile, so M has to cover the routed-pair
// distribution's mass: PF_MOE_HIST on the 35B at 512 tokens is 133 active
// experts, mean 31, max 124, and M=16 leaves 65 of the 133 in one tile (M=32:
// 95, M=64: 121) - i.e. the weight stream is read 2.26 / 1.45 / 1.15 times.
// ===========================================================================

// d and dmin of one superblock.  Hoisted to the superblock because they are
// constant across its eight groups, and returned as SCALARS on purpose: an
// array of the eight group scales indexed by a loop variable lands in private
// (scratch) memory, and private memory on this part is ~350x slower than a
// register.  See the "runtime-indexed arrays" note on moe_grp_impl.
template <uint32_t QT>
static inline void moe_sb_d(const char * sb, float & d, float & dmin) {
    if constexpr (QT == 12) {
        d = moe_h2f(((const block_q4_K *) sb)->d);
        dmin = moe_h2f(((const block_q4_K *) sb)->dmin);
    } else if constexpr (QT == 13) {
        d = moe_h2f(((const block_q5_K *) sb)->d);
        dmin = moe_h2f(((const block_q5_K *) sb)->dmin);
    } else {
        d = moe_h2f(((const block_q6_K *) sb)->d);
        dmin = 0.f; // q6_K's zero point is 32*d*sc and is folded per group below
    }
}

// One 32-value group's scale pair, straight out of the block: two byte loads
// and a handful of ops.  Recomputing this per group costs ~2 extra L1 hits and
// buys not touching scratch at all, which is the difference between 51 ms and
// the number below.
template <uint32_t QT>
static inline void moe_grp_sc(const char * sb, int gg, int lane, float d, float dmin, float & dsc, float & dmn) {
    if constexpr (QT == 12 || QT == 13) {
        const uint8_t * s = (QT == 12) ? ((const block_q4_K *) sb)->scales : ((const block_q5_K *) sb)->scales;
        uint8_t sc, mn;
        get_scale_min_k4(gg, s, &sc, &mn);
        dsc = d * (float)sc;
        dmn = dmin * (float)mn;
    } else {
        // q6_K: a signed byte per SIXTEEN values, and a lane's four values all
        // fall in one of them, so the scale is per-lane here.
        const float e = d * (float)((const block_q6_K *) sb)->scales[2 * (gg & 3) + (lane >> 4)];
        dsc = e;
        dmn = 32.0f * e;
    }
}

// One lane's four weights of 32-value group `gb` of a superblock, as a dp4a
// uint8 operand: w = dsc*u8 - dmn with dsc/dmn from moe_sb_scale above.
//
// `gb` is the group index WITHIN the superblock (0..7).  Passing the row-global
// index instead indexes qs[] up to 7x past its end, so the first eighth of the
// dot product comes out right and the rest is whatever the next rows hold - a 3%
// error, not a fault.
template <uint32_t QT>
static inline uint32_t moe_dp4a_w(const char * sb, int gb, int lane) {
    if constexpr (QT == 12) { // Q4_K: 32 values in 32 bytes, one nibble each
        const auto * x = (const block_q4_K *) sb;
        const uint32_t v = *reinterpret_cast<const uint32_t *>(x->qs + (((gb >> 1) << 5) + (lane << 2)));
        return (gb & 1) ? ((v >> 4) & 0x0F0F0F0Fu) : (v & 0x0F0F0F0Fu);
    } else if constexpr (QT == 13) { // Q5_K: the same plus the one high bit
        const auto * x = (const block_q5_K *) sb;
        const uint32_t v = *reinterpret_cast<const uint32_t *>(x->qs + (((gb >> 1) << 5) + (lane << 2)));
        const uint32_t h = *reinterpret_cast<const uint32_t *>(x->qh + (lane << 2));
        return ((gb & 1) ? ((v >> 4) & 0x0F0F0F0Fu) : (v & 0x0F0F0F0Fu)) | (((h >> gb) & 0x01010101u) << 4);
    } else { // Q6_K: 6 bits = a nibble plus a two-bit plane, scale per 16 values
        const auto * x = (const block_q6_K *) sb;
        const int g0 = gb & 3, l = lane << 2;
        const uint32_t ql = *reinterpret_cast<const uint32_t *>(x->ql + ((g0 & 1) ? 32 + l : l));
        const uint32_t qh = *reinterpret_cast<const uint32_t *>(x->qh + l);
        const uint32_t n = (g0 >> 1) ? ((ql >> 4) & 0x0F0F0F0Fu) : (ql & 0x0F0F0F0Fu);
        const uint32_t t = (qh >> (2 * g0)) & 0x03030303u;
        return n | ((t << 4) & 0x30303030u);
    }
}

// FUSE = 2 reads the gate and up banks in one pass and stores silu(g)*u (the
// activation row is fetched once for both); FUSE = 1 reads one bank and stores
// the plain dot product.  It also decides the row index: gate/up quantize the
// token rows, so a pair's row is pair/top_k, while down quantizes the pair rows
// themselves.
template <uint32_t QT, int FUSE, int M, int C, int NSG, int HOIST = 1, int KC = 0>
static void moe_grp_impl(queue & q, const char * w0, const char * w1, size_t plane0, size_t plane1,
                         const int32_t * slot_map, const int32_t * off, const int32_t * pair,
                         const int32_t * active, const int32_t * n_active, int n_act_max, int n_expert,
                         int n_pairs, bool flat, const int8_t * xq, const uint16_t * asa, const float * xs, float * out,
                         int out_stride, int K, int N, int top_k, bool persist) {
    constexpr int SG = 8; // 8 lanes x 4 values = one 32-value scale group
    constexpr int TX = NSG * SG;
    constexpr int RB = NSG * C; // columns one work-group owns
    const int ng = K / 32;
    const int nsb = K / 256; // superblocks per row
    const int sb = superblock_bytes(QT);
    const size_t row_bytes = (size_t)(K / 256) * (size_t)sb;
    const int nwg = N / RB;
    // FLAT TILE GRID.  The grid used to be one work-group per (expert, column
    // block) with the expert's ceil(p/M) row tiles looped INSIDE it, so a
    // work-group's cost was proportional to that expert's pair count.  On the real
    // routing (PF_MOE_HIST: 71 active experts, mean 54.6 pairs, max 256) that is
    // 1..32 tiles, and no ordering fixes it: a host simulation of greedy
    // list-scheduling over 512 slots gives an in-order makespan of 34 tile-units,
    // an LPT makespan of 34, and a perfect balance of 17 - the single 32-tile
    // work-group IS the critical path and it is the same work-group in every
    // order.  Lifting the row tile into the grid makes every work-group exactly
    // one tile, which is the only thing that moves that bound.
    //
    // The grid is sized by the UPPER bound sum_e ceil(p_e/M) <= n_pairs/M +
    // n_expert (ceil(p/M) <= p/M + 1, and at most n_expert experts have a pair),
    // because the exact count exists only on the device and the grid has to be a
    // host-side constant.  Surplus work-groups scan off the end of the CSR and
    // return; the scan is over n_expert int32 (1 KB, L1-resident) and the surplus
    // is about a third of the grid, which is noise against 18 GiB of weight.
    const int nwt = (n_pairs + M - 1) / M + n_expert;
    // Level Zero rejects a global size that is not a multiple of 256 threads
    // ("Non-uniform work-groups are not supported"), and n_act_max * nwg is a
    // product of two runtime extents - with a 2-row batch it is 16 work-groups
    // and the launch dies.  Round it up to an even count and let the surplus
    // work-groups exit on the `ia >= n_act_max` test.
    const int ngrid = ((nwt * nwg + 1) / 2) * 2;      // flat:   (tile, column block)
    const int ngrid_loop = ((n_act_max * nwg + 1) / 2) * 2; // looped: (expert, column block)
    // NO reqd_sub_group_size here.  Asking this backend for 8-lane sub-groups
    // compiles, launches, and then hands every work-item `get_group(0) == 0`:
    // the whole grid collapses onto one work-group and 127/128 of the experts are
    // never computed.  (The existing expert kernels ask for 32 and are fine.)
    // The 8-lane unit is an *aligned slice* of the natural 32-lane sub-group, so
    // the reduction is the xor-1/2/4 ladder, which cannot leave the slice.
    // nd_range<1> takes a TOTAL work-item count, not a work-group count: with
    // `(ngrid, TX)` the runtime divides again and the whole grid collapses to
    // `ngrid / TX` work-groups - which is 1 here, and 127/128 of the experts are
    // then silently never computed.  Both halves of that mistake look like a
    // correct-but-oddly-slow kernel.
    // KC > 0 stages each K-chunk of the activation tile in SLM.  All NSG
    // sub-groups of this work-group read the SAME M x KC activation bytes, so
    // without it the launch issues NSG times more load requests than the tile has
    // bytes.  KC is bounded by this backend's per-work-group SLM limit, which
    // rejects 34816 B (UR_RESULT_ERROR_OUT_OF_RESOURCES) and accepts 16384;
    // M*KC + M*(KC/32)*2 at M=16, KC=512 is 8704 B.
    // dev/slm_probe measures the stage+barrier at 0.012 ms of a 0.779 ms
    // pipeline (1.5%), so this is close to free - see AGENTS.md, which records a
    // first attempt that was rejected on the strength of a kernel that never
    // passed exactness.
    // Device-visible cursors for the persistent form, at function scope so the
    // kernel lambda can capture the POINTERS.  Null when !persist, and the kernel
    // never dereferences them in that case.
    int32_t * cursor = nullptr;
    int32_t * total_dev = nullptr;
    moe_persist_scratch * ps = nullptr;
    if (persist) {
        ps = &moe_persist_get(q);
        // The work-item count is `sum_e ceil(p_e / M) * ncb` and p_e only exists on
        // the device, so the host cannot size the grid from it - which is the whole
        // reason this form exists.  Sum it here and zero the cursor in the same
        // single-work-item launch, so the main kernel can compare against a real
        // total instead of an upper bound and there is no surplus work-group to
        // pay for.
        // Never capture `ps`: that is a host std::vector-backed struct.
        cursor = ps->counter;
        total_dev = ps->counter + 1;
        q.parallel_for(range<1>(1), [=](id<1>) {
            int nt = 0;
            for (int e = 0; e < n_expert; e++) {
                nt += (off[e + 1] - off[e] + M - 1) / M;
            }
            total_dev[0] = nt * (int)nwg;
            atomic_ref<int32_t, memory_order::relaxed, memory_scope::device,
                        access::address_space::global_space>(cursor[0])
                .store(0);
        });
    }
    // The persistent grid is sized by the MACHINE, not by the problem.  One
    // work-group per (expert, column block) would be n_act_max * nwg, and the
    // problem there is not the count but that the count is a host upper bound
    // over a device-side tile list.  Here it is simply "enough warps to hide
    // latency", and the device decides the assignment.
    // Read ONCE per process and cached: this is inside a launcher that runs 80
    // times per chunk, and an uncached device lookup here would be the exact
    // regression this file records ("Never put a device query in a kernel
    // launcher" - 3.2 ms per launch, 210 ms in one verify pass).  si::dev::
    // profile_int already caches the env half; the EU count comes from the
    // already-resolved per-queue profile.
    static const int persist_wg = si::env::i32("PF_MOE_PERSIST_WG", 0);
    const int npersist = persist_wg > 0 ? persist_wg : 2 * si::dev::for_queue(q).hw.compute_units;
    const size_t slm_act = (KC > 0) ? (size_t)M * (size_t)KC : (size_t)4;
    q.submit([&](handler & h) {
        local_accessor<int8_t, 1> l_act(slm_act, h);
        local_accessor<uint16_t, 1> l_asa((size_t)M * (size_t)((KC > 0) ? KC : 32) / 32 * 2, h);
        local_accessor<int32_t, 1> l_item(1, h);
        h.parallel_for(nd_range<1>((size_t)(persist ? (npersist > 0 ? npersist : 1024) : (flat ? ngrid : ngrid_loop)) * TX, TX),
                     [=](nd_item<1> it) {
        const int lid = (int)it.get_local_id(0);
        const int lane = lid % SG;
        const sub_group sgg = it.get_sub_group();
        const int gw = (int)it.get_group(0);
        // (tile, column block) -> (expert, row offset).  A linear scan of the
        // CSR rather than a prebuilt tile list: n_expert is a few hundred, the
        // array is 1 KB and L1-resident, and this way there is no scratch to
        // allocate, no second kernel to launch, and nothing for dev/grp_min to
        // have to set up differently from the engine.
    // THE WORK LOOP.  For the static forms this runs exactly once - the work item
    // came from the grid - and for the persistent form it runs until the atomic
    // cursor is exhausted.
    //
    // This loop is CORRECTNESS, not an optimisation.  The first version drew ONCE
    // per work-group and returned, and it measured 2.85x on the prefill with every
    // argmax oracle unchanged - which was the tell, not the win.  A real prefill has
    // sum_e ceil(p_e/M) * ncb work items (~9000 at 512 tokens) against a grid of
    // 2*compute_units (~1024), so that version silently DROPPED ~90% of the work and
    // still produced the right argmax.  The argmax is a weak oracle for exactly this
    // class of change - this file records the prefill argmax sitting at 4821 while
    // the router was selecting a completely wrong expert set - and the standing rule
    // is that a stub which beats the real kernel means the stub is wrong.  It was.
    //
    // `persist || pass == 0` keeps the static forms at exactly one iteration.  Every
    // exit inside the loop is work-group-uniform (the item is broadcast from lane 0,
    // and e/slot depend only on it), so the barriers stay collective.
    for (int pass = 0; persist || pass == 0; pass++) {
            int e = n_expert, cb = 0, rem = 0;
            if (persist) {
                // Pull work items until the device-side total is exhausted.  The cursor
                // and the total are both DEVICE memory written by the prepare launch
                // above, so nothing here is a host value: `total` is the exact
                // sum_e ceil(p_e/M) * ncb, which the host cannot know, and that is the
                // entire reason this form exists.  Column block is the INNER index, so
                // consecutive items walk consecutive column blocks of one weight row and
                // the stream stays sequential.
                const int total = total_dev[0];
                // ONE WORK ITEM PER WORK-GROUP, and that is the whole subtlety here.
                // The first version of this had every lane call fetch_add, so the 128
                // lanes of a work-group drew 128 DIFFERENT items: sg_sum then summed
                // eight unrelated (expert, tile, column) computations, and every lane
                // stored to a different output column.  Exactness caught it at
                // 8191/8192 mismatches and - the part worth keeping - raising
                // PF_MOE_PERSIST_WG from 64 to 4096 changed nothing, which is what a
                // per-lane draw looks like and what a starved grid does not.
                //
                // So lane 0 draws once and publishes through SLM.  EVERY lane must
                // reach the barrier below, including the ones whose work-group has
                // finished, so the early-out is placed AFTER it - this file already
                // records the shape of this mistake ("sg_sum is a sub-group collective,
                // so it has to be called by ALL 32 lanes"; moving it under a lane test
                // compiled, produced plausible text, and was wrong).
                if (lid == 0) {
                    atomic_ref<int32_t, memory_order::relaxed, memory_scope::device,
                                access::address_space::global_space> cur(cursor[0]);
                    l_item[0] = cur.fetch_add(1);
                }
                it.barrier(sycl::access::fence_space::local_space);
                const int item = l_item[0];
                if (item >= total) {
                    return; // this work-group's share is done, and now uniformly so
                }
                int t = item / (int)nwg;
                cb = item - t * (int)nwg;
                e = n_expert;
                for (int ee = 0; ee < n_expert; ee++) {
                    const int nt = (off[ee + 1] - off[ee] + M - 1) / M;
                    if (t < nt) {
                        e = ee;
                        rem = t;
                        break;
                    }
                    t -= nt;
                }
                if (e >= n_expert) {
                    return; // should not happen; the prepare pass summed exactly this
                }
            } else if (flat) {
                const int t = gw / nwg;
                cb = gw - t * nwg;
                rem = t;
                e = 0;
                for (; e < n_expert; e++) {
                    const int nt = (off[e + 1] - off[e] + M - 1) / M;
                    if (rem < nt) {
                        break;
                    }
                    rem -= nt;
                }
                if (e >= n_expert) {
                    return; // surplus grid slot
                }
            } else {
                const int ia = gw / nwg;
                cb = gw - ia * nwg;
                if (ia >= n_act_max) {
                    return;
                }
                if (ia >= n_active[0]) {
                    return; // fewer experts selected than the grid was sized for
                }
                e = active[ia];
            }
            const int slot = slot_map[e];
            if (slot < 0) {
                return; // host-resident expert: the host computes it
            }
            const char * b0 = w0 + (size_t)slot * plane0;
            const char * b1 = (FUSE == 2) ? (w1 + (size_t)slot * plane1) : b0;
            const int e1 = off[e + 1];
            const int i0base = (flat || persist) ? (off[e] + rem * M) : off[e];
            const int ntile = (flat || persist) ? 1 : ((e1 - i0base + M - 1) / M);
            const int n0 = cb * RB + (lid / SG) * C;
            float acc[M][FUSE][C];
            // NOT named `it`: that is the nd_item<1> this kernel's every `it.barrier`
            // refers to, and shadowing it turns `it.barrier(...)` into `int.barrier`.
            for (int itk = 0; itk < ntile; itk++) {
            const int i0 = i0base + itk * M;
            const int np = sycl::min(M, e1 - i0);
    #pragma unroll
                for (int i = 0; i < M; i++) {
    #pragma unroll
                    for (int f = 0; f < FUSE; f++) {
    #pragma unroll
                        for (int c = 0; c < C; c++) {
                            acc[i][f][c] = 0.f;
                        }
                    }
                }
                // HOISTED out of the K loop.  `pair[i0+m]`, the top_k division and
                // the row address are per (tile, row), not per (tile, row, group):
                // leaving them inside cost 2 of the 3 memory ops in the inner body
                // (one `pair` load and one integer division per 32-value group per
                // row), and the inner body only has FUSE*C dp4a to hide them behind.
                // Measured 8.94 -> 6.03 ms for the `down` projection.
                //
                // But it costs M *integer registers* on top of the M*FUSE*C float
                // accumulators, and M is what cannot grow (M=24/32 are the only
                // instantiations IGC reports as spilling, and they measure 1.4-1.6x
                // slower *despite* moving 1.56x less weight).  HOIST=0 recomputes
                // the row inside the group loop instead, trading M registers for
                // M*nsb*8 extra loads - which only pays if that is enough to stop
                // the spill.  PF_MOE_CFG cases 16+ are that A/B.
                int trowv[HOIST ? M : 1];
    #pragma unroll
                for (int m = 0; m < (HOIST ? M : 1); m++) {
                    const int pr = (m < np) ? pair[i0 + m] : 0;
                    trowv[m] = (FUSE == 2) ? (pr / top_k) : pr;
                }
                // The K loop.  With KC == 0 this is one iteration over the whole
                // reduction and the activation is read from global; with KC > 0 it is
                // K/KC chunks, each staged once into SLM.
                for (int kc = 0; kc < K; kc += ((KC > 0) ? KC : K)) {
                    if constexpr (KC > 0) {
                        int8_t * ap = l_act.get_multi_ptr<access::decorated::no>().get();
                        uint16_t * sp = l_asa.get_multi_ptr<access::decorated::no>().get();
                        const int ngc = KC / 32;
    #pragma unroll
                        for (int m = 0; m < M; m++) {
                            if (m >= np) {
                                continue;
                            }
                            const int8_t * src = xq + (size_t)trowv[m] * (size_t)K + (size_t)kc;
                            int8_t * dst = ap + (size_t)m * (size_t)KC;
                            for (int i = lid; i < KC / 4; i += TX) {
                                *reinterpret_cast<uint4 *>(dst + i * 4) = *reinterpret_cast<const uint4 *>(src + i * 4);
                            }
                            for (int i = lid; i < ngc; i += TX) {
                                sp[(size_t)m * (size_t)ngc + i] = asa[(size_t)trowv[m] * (size_t)ng + kc / 32 + i];
                            }
                        }
                        it.barrier(access::fence_space::local_space);
                    }
                    const int koff = kc;
                    for (int sbi = 0; sbi < ((KC > 0) ? (KC / 256) : nsb); sbi++) {
                    // d/dmin and the block base, once per (bank, column,
                    // superblock): scalars only, and `sbp` is loop-invariant in gg.
                    float dv[FUSE][C], dmv[FUSE][C];
                    const char * sbbase[FUSE][C];
    #pragma unroll
                    for (int f = 0; f < FUSE; f++) {
    #pragma unroll
                        for (int c = 0; c < C; c++) {
                            const int n = n0 + c;
                            const char * bb = (FUSE == 2 && f == 1) ? b1 : b0;
                            sbbase[f][c] = bb + (size_t)n * row_bytes + (size_t)(koff / 256 + sbi) * (size_t)sb;
                            if (n < N) {
                                moe_sb_d<QT>(sbbase[f][c], dv[f][c], dmv[f][c]);
                            } else {
                                dv[f][c] = 0.f;
                                dmv[f][c] = 0.f;
                            }
                        }
                    }
                    // FULLY unrolled, and that is the point rather than a style
                    // choice.  `gg` is a compile-time constant inside, so the scale
                    // decode is constant-indexed and all eight groups' weight and
                    // scale loads for a superblock issue before the first dp4a
                    // waits on any of them.  With a rolled loop each group pays the
                    // full global latency with nothing to overlap it, and the whole
                    // superblock loop is 99% of the kernel: measured 51 ms rolled vs
                    // 0.18 ms with the loop removed, and the loads - not the dp4a -
                    // are what is being waited on.
    #pragma unroll
                    for (int gg = 0; gg < 8; gg++) {
                        const int g = koff / 32 + sbi * 8 + gg;
                        uint32_t wq[FUSE][C];
                        float dsc[FUSE][C], dmn[FUSE][C];
    #pragma unroll
                        for (int f = 0; f < FUSE; f++) {
    #pragma unroll
                            for (int c = 0; c < C; c++) {
                                const int n = n0 + c;
                                const char * sbp = sbbase[f][c];
                                wq[f][c] = (n < N) ? moe_dp4a_w<QT>(sbp, gg, lane) : 0u;
                                if (n < N) {
                                    moe_grp_sc<QT>(sbp, gg, lane, dv[f][c], dmv[f][c], dsc[f][c], dmn[f][c]);
                                } else {
                                    dsc[f][c] = 0.f;
                                    dmn[f][c] = 0.f;
                                }
                            }
                        }
                        // The bound is M, not np, ON PURPOSE.  With `m < np` the
                        // index into acc[] is a runtime value, so `acc` is placed in
                        // private (scratch) memory and every dp4a becomes a scratch
                        // load + store - which measured 51 ms against 0.18 ms for
                        // the identical loop nest.  With the bound at the template M
                        // the loop unrolls, `m` becomes a constant, and acc lives in
                        // registers; the price is dp4a for rows the expert does not
                        // have, which the PF_MOE_HIST distribution puts at ~14%.
    #pragma unroll
                        for (int m = 0; m < M; m++) {
                            // predicated, NOT `break`: a runtime `break` in a
                            // compile-time-bound loop stops the unroller, and then
                            // every index into acc[] is a runtime value, so acc lands
                            // in private scratch memory.  Measured 51 ms that way
                            // against 0.18 ms with the same loop nest removed.
                            if (m >= np) {
                                continue;
                            }
                            const int pr2 = (m < np) ? pair[i0 + m] : 0;
                            const int trow = HOIST ? trowv[m] : ((FUSE == 2) ? (pr2 / top_k) : pr2);
                            uint32_t xa;
                            float A;
                            if constexpr (KC > 0) {
                                const int8_t * ap = l_act.get_multi_ptr<access::decorated::no>().get();
                                const uint16_t * sp = l_asa.get_multi_ptr<access::decorated::no>().get();
                                const int gl = g - koff / 32;
                                xa = *reinterpret_cast<const uint32_t *>(
                                    ap + (size_t)m * (size_t)KC + (size_t)gl * 32 + (size_t)lane * 4);
                                A = moe_h2f(sp[(size_t)m * (size_t)(KC / 32) + gl]);
                            } else {
                                xa = *reinterpret_cast<const uint32_t *>(
                                    xq + (size_t)trow * (size_t)K + (size_t)g * 32 + (size_t)lane * 4);
                                A = moe_h2f(asa[(size_t)trow * (size_t)ng + g]);
                            }
                            const float asl = -A * (float)dp4a_s8u8(xa, 0x01010101u, 0);
    #pragma unroll
                            for (int f = 0; f < FUSE; f++) {
    #pragma unroll
                                for (int c = 0; c < C; c++) {
                                    const int32_t q = dp4a_s8u8(xa, wq[f][c], 0);
                                    acc[m][f][c] = sycl::fma(A * dsc[f][c], (float)q, acc[m][f][c]);
                                    acc[m][f][c] = sycl::fma(asl, dmn[f][c], acc[m][f][c]);
                                }
                            }
                        }
                    } // gg
                    } // sbi
                    if constexpr (KC > 0) {
                        // every thread must arrive before the next chunk restages
                        // over the same SLM
                        it.barrier(access::fence_space::local_space);
                    }
                } // kc
                // one cross-lane step for the whole tile, then the store.  `pair[]` is built
                // by an atomic scatter, so the rows of one expert are in NO
                // particular order - the store has to index through it, not assume
                // pair[i0+m] == pair[i0]+m (which silently misplaces every row of
                // every expert that selected more than one pair).
    #pragma unroll
                for (int m = 0; m < M; m++) {
                    if (m >= np) {
                        continue;
                    }
    #pragma unroll
                    for (int c = 0; c < C; c++) {
                        if (n0 + c >= N) {
                            continue;
                        }
                        float v[FUSE];
    #pragma unroll
                        for (int f = 0; f < FUSE; f++) {
                            // every lane must enter the collective
                            v[f] = sg_sum<SG>(acc[m][f][c], sgg);
                        }
                        if (lane == 0) {
                            out[(size_t)pair[i0 + m] * (size_t)out_stride + n0 + c] =
                                (FUSE == 2) ? silu_f(v[0]) * v[1] : v[0];
                        }
                    }
                }
    }
        }
        });
    });
}

// Quantize a MoE activation tile to the form the dense int8 path consumes:
// per-32-group int8, an f16 scale per group and the f32 sum of the group.
// One 32-lane group per (row, group) - the same shape as
// `act_quant_grp_launch`, so the two agree on what an activation is.
void moe_actq_launch(queue & q, const float * x, int x_stride, int n_rows, int K, int8_t * xq, uint16_t * asa,
                     float * xs) {
    const int ng = K / 32;
    q.parallel_for(nd_range<1>((size_t)n_rows * (size_t)ng * 32, 32), [=](nd_item<1> it) {
        const int gid = (int)it.get_group(0);
        const int m = gid / ng, g = gid - m * ng;
        const int lane = (int)it.get_local_id(0);
        const float * xr = x + (size_t)m * (size_t)x_stride + (size_t)g * 32;
        const float v = xr[lane];
        const float mx = sycl::reduce_over_group(it.get_sub_group(), sycl::fabs(v), sycl::maximum<float>());
        // An all-zero group stores scale 0, not 1.  With scale 1 a zero
        // activation group would contribute -127 * dmin instead of 0, i.e. the
        // zero point applied at 127x; the same guard the KV path needs.
        const float sc = mx > 0.f ? mx / 127.0f : 0.f;
        const float inv = (sc > 0.f) ? (127.0f / mx) : 0.f;
        const int qi = (int)sycl::round(v * inv);
        const int8_t q8 = (int8_t)sycl::max(-127, sycl::min(127, qi));
        xq[(size_t)m * (size_t)K + (size_t)g * 32 + lane] = q8;
        if (lane == 0) {
            sycl::half h(sc);
            uint16_t bits;
            __builtin_memcpy((void *)&bits, &h, sizeof(h));
            asa[(size_t)m * (size_t)ng + g] = bits;
        }
        const int s = sycl::reduce_over_group(it.get_sub_group(), (int)q8, sycl::plus<int>());
        if (lane == 0) {
            xs[(size_t)m * (size_t)ng + g] = (float)s;
        }
    });
}

// ---------------------------------------------------------------------------
// Xe2 decode expert GEMV: 16-byte lanes, one coalesced 128-byte weight load per
// superblock, and one 16-byte header load for d/dmin/scales.
//
// The previous form read 4 bytes per lane (one 32-value group split over eight
// lanes) and paid ~5 scalar loads per superblock for d, dmin and the 6-bit
// scales; it measured 61 GB/s on the 35B decode shape, issue-bound, not
// bandwidth-bound.  This form gives a lane a 16-byte unit of `qs` (16 low
// nibbles of group 2c + 16 high nibbles of group 2c+1), so eight consecutive
// lanes cover the superblock's 128 bytes as ONE contiguous run and a lane does
// 8 dp4a per weight load.  Bytes 0..15 of a superblock are exactly
// (d, dmin, scales[12]), so the whole metadata is one uint4 and the two groups'
// (scale,min) pairs are extracted with ALU only.  Measured on this part:
// gate/up 61 -> 101 GB/s, down 56 -> 100 GB/s (DRAM-sized banks), i.e. 96-98%
// of a 16-byte-per-lane read probe on the same access shape.
//
// RPW output rows share a warp (16 lanes per row at RPW=2); a workgroup is WGW
// warps so a group of consecutive warps walks consecutive weight rows, which is
// the dense u4 GEMV's winning shape.  The reduction is a butterfly over the
// row's lane group only.
// ---------------------------------------------------------------------------
// One 4-byte weight word: four low-nibble values and four high-nibble values,
// with Q5_K's fifth bit folded in.  The activation sums the asymmetric
// zero-point term needs are one extra dp4a each (measured free at this rate).
template <int QT>
static inline void moe_dp_word(uint32_t qv, uint32_t hw, int cbits, uint32_t xlow, uint32_t xhigh, int & qdl,
                               int & qdh, int & sxl, int & sxh) {
    uint32_t lo = qv & 0x0F0F0F0Fu;
    uint32_t hi = (qv >> 4) & 0x0F0F0F0Fu;
    if constexpr (QT == 13) {
        lo |= ((hw >> cbits) & 0x01010101u) << 4;
        hi |= ((hw >> (cbits + 1)) & 0x01010101u) << 4;
    }
    qdl = dp4a_s8u8((int32_t)xlow, lo, qdl);
    qdh = dp4a_s8u8((int32_t)xhigh, hi, qdh);
    sxl = dp4a_s8u8((int32_t)xlow, 0x01010101u, sxl);
    sxh = dp4a_s8u8((int32_t)xhigh, 0x01010101u, sxh);
}

template <uint32_t QT, int FUSE, int UNITSZ, int RPW, int UNITSX, int WGW>
static void moe_row_gemv_impl(queue & q, const char * w0, const char * w1, size_t plane0, size_t plane1,
                              const int32_t * slot_map, const int32_t * ids, int n_pairs, int K, int N,
                              const int8_t * xq, const uint16_t * asa, float * out, int out_stride,
                              const float * gate = nullptr, bool act_pair = true) {
    constexpr int SB = (QT == 12) ? 144 : 176; // block_q4_K / block_q5_K
    constexpr int QOFF = (QT == 12) ? 16 : 48;
    constexpr int NW = UNITSZ / 16; // uint4s per unit
    constexpr int LPR = 32 / RPW;   // lanes per output row
    constexpr int USP = 128 / UNITSZ;
    constexpr int TX = WGW * 32;
    static_assert((LPR * UNITSX) % USP == 0, "the unit walk must tile whole superblocks");
    const int nsb = K / 256;
    const int ng = K / 32;
    const size_t row_bytes = (size_t)nsb * (size_t)SB;
    const int sbpi = (LPR * UNITSX) / USP; // superblocks consumed per iteration
    const int nrb = (N + RPW - 1) / RPW;
    const int njobs = n_pairs * nrb;
    const int ngrp = (njobs + WGW - 1) / WGW;
    q.parallel_for(nd_range<1>((size_t)ngrp * TX, TX),
                   [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                       const int lid = (int)it.get_local_id(0);
                       const int warp = lid >> 5, lane = lid & 31;
                       const int job = (int)it.get_group(0) * WGW + warp;
                       if (job >= njobs) {
                           return;
                       }
                       const int ia = job / nrb, rb = job - ia * nrb;
                       const int li = lane % LPR;
                       const int nrow = rb * RPW + (lane / LPR);
                       const int n = sycl::min(nrow, N - 1);
                       // A single-token decode's router output IS the CSR: pair
                       // ia has expert ids[ia]; gate/up's activation is the one
                       // token (row 0), down's is the pair itself.
                       const int e = ids[ia];
                       const int slot = slot_map[e];
                       if (slot < 0) {
                           return;
                       }
                       const int trow = act_pair ? ia : 0;
                       const char * b0 = w0 + (size_t)slot * plane0 + (size_t)n * row_bytes;
                       const char * b1 = (FUSE == 2) ? (w1 + (size_t)slot * plane1 + (size_t)n * row_bytes)
                                                     : nullptr;
                       const int8_t * xr = xq + (size_t)trow * (size_t)K;
                       const uint16_t * ar = asa + (size_t)trow * (size_t)ng;
                       const int sbl0 = (li * UNITSX) / USP;
                       const int ui0 = (li * UNITSX) % USP;
                       float acc = 0.f, acc1 = 0.f;
                       for (int sb = sbl0; sb < nsb; sb += sbpi) {
                           const char * p0 = b0 + (size_t)sb * SB;
                           const char * p1 = b1 ? (b1 + (size_t)sb * SB) : nullptr;
                           // d, dmin, scales[12] in one 16-byte load.
                           const uint4 h = *reinterpret_cast<const uint4 *>(p0);
                           uint32_t hdr1 = 0, uw0 = 0, uw1 = 0, uw2 = 0;
                           const uint4 u = (FUSE == 2) ? *reinterpret_cast<const uint4 *>(p1) : uint4(0, 0, 0, 0);
                           if constexpr (FUSE == 2) {
                               hdr1 = u.x();
                               uw0 = u.y();
                               uw1 = u.z();
                               uw2 = u.w();
                           }
#pragma unroll
                           for (int k = 0; k < UNITSX; k++) {
                               const int off = (ui0 + k) * UNITSZ;
                               const int cb = off >> 5;
                               const int cbits = 2 * cb;
                               uint4 xl[NW], xh[NW];
#pragma unroll
                               for (int w = 0; w < NW; w++) {
                                   const int cc = (off + 16 * w) >> 5;
                                   const int hh = ((off + 16 * w) >> 4) & 1;
                                   const int xo = sb * 256 + 64 * cc + 16 * hh;
                                   xl[w] = *reinterpret_cast<const uint4 *>(xr + xo);
                                   xh[w] = *reinterpret_cast<const uint4 *>(xr + xo + 32);
                               }
                               const float alo = moe_h2f(ar[sb * 8 + 2 * cb]);
                               const float ahi = moe_h2f(ar[sb * 8 + 2 * cb + 1]);
                               // The six 6-bit scale/min fields of a chunk are
                               // three 16-bit pairs of the header's three scale
                               // words, selected by cb>>1 and shifted by
                               // 16*(cb&1) - no byte loads, no branches.
                               auto do_bank = [&](const char * p, uint32_t hdr, uint32_t sw0, uint32_t sw1,
                                                  uint32_t sw2, float & accp) {
                                   const int hi = 16 * (cb & 1);
                                   const uint32_t wa = (cb < 2) ? sw0 : sw1;
                                   const uint32_t wb = (cb < 2) ? sw1 : sw2;
                                   const uint32_t pa = (wa >> hi) & 0xFFFFu; // s[j], s[j+1]
                                   const uint32_t pb = (wb >> hi) & 0xFFFFu; // s[j+4], s[j+5]
                                   uint32_t sd0, sm0, sd1, sm1;
                                   if (cb < 2) {
                                       sd0 = pa & 63u;
                                       sm0 = pb & 63u;
                                       sd1 = (pa >> 8) & 63u;
                                       sm1 = (pb >> 8) & 63u;
                                   } else {
                                       const uint32_t pc = (sw0 >> hi) & 0xFFFFu;
                                       sd0 = (pb & 0x0Fu) | ((pc & 0xC0u) >> 2);
                                       sm0 = ((pb & 0xF0u) >> 4) | ((pa & 0xC0u) >> 2);
                                       sd1 = ((pb >> 8) & 0x0Fu) | (((pc >> 8) & 0xC0u) >> 2);
                                       sm1 = (((pb >> 8) & 0xF0u) >> 4) | ((((pa >> 8) & 0xC0u)) >> 2);
                                   }
                                   uint4 qw[NW], hq[NW];
#pragma unroll
                                   for (int w = 0; w < NW; w++) {
                                       qw[w] = *reinterpret_cast<const uint4 *>(p + QOFF + off + 16 * w);
                                       if constexpr (QT == 13) {
                                           const int hh = ((off + 16 * w) >> 4) & 1;
                                           hq[w] = *reinterpret_cast<const uint4 *>(p + 16 + 16 * hh);
                                       }
                                   }
                                   int qdl = 0, qdh = 0, sxl = 0, sxh = 0;
#pragma unroll
                                   for (int w = 0; w < NW; w++) {
                                       const uint4 v = qw[w];
                                       uint32_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;
                                       if constexpr (QT == 13) {
                                           q0 = hq[w].x();
                                           q1 = hq[w].y();
                                           q2 = hq[w].z();
                                           q3 = hq[w].w();
                                       }
                                       moe_dp_word<QT>(v.x(), q0, cbits, xl[w].x(), xh[w].x(), qdl, qdh, sxl, sxh);
                                       moe_dp_word<QT>(v.y(), q1, cbits, xl[w].y(), xh[w].y(), qdl, qdh, sxl, sxh);
                                       moe_dp_word<QT>(v.z(), q2, cbits, xl[w].z(), xh[w].z(), qdl, qdh, sxl, sxh);
                                       moe_dp_word<QT>(v.w(), q3, cbits, xl[w].w(), xh[w].w(), qdl, qdh, sxl, sxh);
                                   }
                                   const float dsb = moe_h2f((uint16_t)hdr);
                                   const float mnsb = moe_h2f((uint16_t)(hdr >> 16));
                                   const float st0 = dsb * (float)sd0, of0 = mnsb * (float)sm0;
                                   const float st1 = dsb * (float)sd1, of1 = mnsb * (float)sm1;
                                   accp += alo * (st0 * (float)qdl - of0 * (float)sxl);
                                   accp += ahi * (st1 * (float)qdh - of1 * (float)sxh);
                               };
                               do_bank(p0, h.x(), h.y(), h.z(), h.w(), acc);
                               if constexpr (FUSE == 2) {
                                   do_bank(p1, hdr1, uw0, uw1, uw2, acc1);
                               }
                           }
                       }
                       const sub_group sg = it.get_sub_group();
#pragma unroll
                       for (int m = 1; m < LPR; m <<= 1) {
                           acc += permute_group_by_xor(sg, acc, m);
                           if constexpr (FUSE == 2) {
                               acc1 += permute_group_by_xor(sg, acc1, m);
                           }
                       }
                       if (li == 0 && nrow < N) {
                           const size_t o = (size_t)ia * (size_t)out_stride + n;
                           if constexpr (FUSE == 2) {
                               out[o] = silu_f(acc) * acc1;
                           } else if (gate) {
                               // The up pass: one launch per bank (more warps,
                               // one weight stream per warp) with the gate read
                               // back from the same element this thread owns.
                               out[o] = silu_f(gate[o]) * acc;
                           } else {
                               out[o] = acc;
                           }
                       }
                   });
}

bool moe_row_gemv_gate_up_launch(queue & q, const void * gbank, const void * ubank, const int32_t * slot_map,
                                 size_t slot_gb, size_t slot_ub, const int32_t * ids, int n_pairs,
                                 const int8_t * xq, const uint16_t * asa, float * h, int h_stride, int n_embd,
                                 int n_ff_exp, uint32_t gtype, uint32_t utype) {
    if (gtype != utype) {
        return false;
    }
    const char * g = (const char *)gbank;
    const char * u = (const char *)ubank;
    // Two one-bank passes by default: the fused form read two weight streams per
    // warp and measured 60-73 GB/s of an 8-expert read; splitting doubles the
    // warps and leaves one stream per warp.  The up pass reads back the gate
    // element it owns (in-order queue, one writer per element).
    static const bool fuse2 = si::env::flag("PF_MOE_ROW_FUSE2");
#define SI_ROW_GU2(QT)                                                                                             \
    do {                                                                                                           \
        if (fuse2) {                                                                                               \
            moe_row_gemv_impl<QT, 2, 16, 2, 2, 8>(q, g, u, slot_gb, slot_ub, slot_map, ids, n_pairs, n_embd,       \
                                                  n_ff_exp, xq, asa, h, h_stride, nullptr, false);                 \
        } else {                                                                                                   \
            moe_row_gemv_impl<QT, 1, 16, 2, 2, 8>(q, g, nullptr, slot_gb, 0, slot_map, ids, n_pairs, n_embd,       \
                                                  n_ff_exp, xq, asa, h, h_stride, nullptr, false);                 \
            moe_row_gemv_impl<QT, 1, 16, 2, 2, 8>(q, u, nullptr, slot_ub, 0, slot_map, ids, n_pairs, n_embd,       \
                                                  n_ff_exp, xq, asa, h, h_stride, h, false);                       \
        }                                                                                                          \
    } while (0)
    switch (gtype) {
    case 12: SI_ROW_GU2(12); return true;
    case 13: SI_ROW_GU2(13); return true;
    default: return false;
    }
#undef SI_ROW_GU2
}

bool moe_row_gemv_down_launch(queue & q, const void * dbank, const int32_t * slot_map, size_t slot_db,
                              const int32_t * ids, int n_pairs, const int8_t * xq, const uint16_t * asa,
                              float * acc, int acc_stride, int n_embd, int n_ff_exp, uint32_t dtype) {
    const char * d = (const char *)dbank;
    // K = n_ff_exp = 512 -> two superblocks per row: a 32-byte unit is a whole
    // chunk, so four lanes cover a row and one iteration covers it end to end.
#define SI_ROW_D(QT)                                                                                               \
    moe_row_gemv_impl<QT, 1, 32, 4, 1, 16>(q, d, nullptr, slot_db, 0, slot_map, ids, n_pairs, n_ff_exp, n_embd,    \
                                          xq, asa, acc, acc_stride, nullptr, true)
    switch (dtype) {
    case 12: SI_ROW_D(12); return true;
    case 13: SI_ROW_D(13); return true;
    default: return false;
    }
#undef SI_ROW_D
}

namespace {

// Tile-shape dispatch.  The shape is a bandwidth/re-read/latency trade and all
// three terms are measured (see AGENTS.md):
//   NSG  columns per work-group is NSG*C, so a larger NSG means fewer
//        work-groups and fewer re-reads of the activation tile
//   C    columns per lane, i.e. dp4a issued per activation load
//   M    rows per tile, i.e. weight re-read factor ceil(p_e/M) (PF_MOE_HIST)
// The accumulator is M*FUSE*C floats and it has to stay in REGISTERS, so the
// product is the binding constraint - an M*C*FUSE that spills costs 350x.
// PF_MOE_CFG picks it; the default is the measured best of the sweep and lives
// in the card's device profile (the sweep was measured per card).
//
// Resolve the case: env override, else this queue's card profile.  cfg_dec is
// special because 0 is a *measured value* there ("auto"), and profile_int()
// folds an env 0 into "unset" - so that one reads the raw env with a -1
// sentinel, or an explicit PF_MOE_CFG_DEC=0 could never restore auto on a card
// whose profile pins a decode tile.
static int moe_cfg_for(const queue & q, bool down, bool dec) {
    const si::dev::profile & dp = si::dev::for_queue(q);
    if (dec) {
        static const int e = si::env::i32("PF_MOE_CFG_DEC", -1);
        return e >= 0 ? e : dp.moe.cfg_dec;
    }
    return down ? si::dev::profile_int(q, "PF_MOE_CFG_DOWN", dp.moe.cfg_down)
                : si::dev::profile_int(q, "PF_MOE_CFG", dp.moe.cfg_gate);
}

// The row tile M a case selects.  It is the first argument of every
// SI_MOE_TILE* below and it is kept next to the dispatch on purpose: the
// PF_MOE_HIST schedule model needs the tile before the kernel runs, and it used
// to guess "cfg 43 => 8, otherwise 16", which is wrong for every other case.
// `default:` is the 128-thread M=16 tile.
static int moe_cfg_tile_m(int cfg) {
    switch (cfg) {
    case 1: return 32;
    case 2: return 16;
    case 3: return 8;
    case 4: return 32;
    case 7: return 64;
    case 8: return 64;
    case 9: return 32;
    case 10: return 64;
    case 11: return 32;
    case 12: return 64;
    case 13: return 24;
    case 14: return 48;
    case 15: return 32;
    case 16: return 32;
    case 17: return 24;
    case 18: return 48;
    case 19: return 64;
    case 20: return 32;
    case 21: return 16;
    case 30: return 32;
    case 32: return 32;
    case 40:
    case 41: return 8;
    case 42: return 4;
    case 43: return 8;
    case 44: return 16;
    case 45: return 4;
    case 46: return 8;
    case 47: return 4;
    case 48: return 8;
    case 49: return 4;
    case 50: return 32;
    case 51: return 16;
    case 52: return 12;
    case 53: return 8;
    case 54: return 8;
    case 55: return 16;
    case 56: return 24;
    case 57: return 8;
    case 58: return 1;
    case 59: return 2;
    default: return 16;
    }
}

template <int FUSE, uint32_t QT>
static bool moe_grp_go(queue & q, const char * w0, const char * w1, size_t plane0, size_t plane1, const int32_t * slot_map,
                       const int32_t * off, const int32_t * pair, const int32_t * active, const int32_t * n_active,
                       int n_act_max, int n_expert, int n_pairs, bool flat, const int8_t * xq, const uint16_t * asa,
                       const float * xs, float * out, int out_stride, int K, int N, int top_k, bool persist, bool dec) {
    // One-shot launch geometry: this kernel's cost is set by the grid
    // (experts x N/RB) and by the row tiles, neither of which is visible from
    // outside, and the quiet way to be wrong is to launch a grid that does not
    // match the CSR.
    static const bool dbg = si::env::flag("PF_MOE_GRPDBG");
    // gate/up and down have DIFFERENT optima, measured (dev/bench_grp, the
    // 35B geometry): gate/up wants C=2 (C=4 spills 128 accumulators and costs
    // 4.5x), down wants C=1 (4.57 vs 6.33 ms) because its accumulator count is
    // half and occupancy wins over dp4a-per-activation-load.  So the config is
    // per projection; PF_MOE_CFG / PF_MOE_CFG_DOWN override either.
    // M=8 / C=2 / NSG=16, the measured optimum for BOTH projections
    // (dev/bench_grp, 35B geometry, best of 2): gate+up 6.01 ms and down 3.38 ms
    // against 7.05/4.12 for the M=16 C=2 tile that shipped before, and
    // 7.29/3.76 for M=16 C=1.  One config, not a per-projection pair.
    //
    // The reason M went DOWN and not up is the shape of this kernel: the weight
    // re-read factor is ceil(p_e/M), so M=8 reads the weight ~1.87x more than
    // M=16 and is still 17% faster, because the kernel sustains only 50-68 GB/s
    // of unique weight against a 376 GB/s card - **bandwidth is not the scarce
    // resource here, occupancy is.**  What binds is the register budget
    // P = M*FUSE*C: every P=32 config lands at 6.0-6.1 ms, P=64 jumps to
    // 7.0-7.3 and P=128 to 20-51.  (An earlier note here claimed the binding
    // constraint was the dp4a-per-load ratio M*FUSE*C/(M+FUSE*C); the sweep
    // refutes it - the optimum has ratio 2.67, BELOW the M=16 tile's 3.20.)
    // The DECODE tile.  0 = auto: ONE row per expert for a single-token decode
    // (every selected expert gets exactly one pair, so a wider tile is pure
    // waste) and TWO rows for a batched one (some experts get two or three
    // pairs, and a one-row tile would re-read their weights once per extra
    // pair).  A non-zero value forces that tile for both projections.
    //
    // Measured on the 35B A3B / 2x A770 (dev/bench_conc, the decode step, the
    // grouped dp4a path, the fp32 path at 38.98 / 87.62 / 135.84 / 191.96 ms
    // for N=1 / 8 / 16 / 32):
    //
    //   N=1    M=1 37.69   M=2 41.50
    //   N=8    M=1 79.62   M=2 78.62
    //   N=16   M=1 136.01  M=2 130.50
    //   N=32   M=1 212.77  M=2 191.64
    //
    // (cases 58 and 59 are those two tiles)
    const int cfg_resolved = moe_cfg_for(q, FUSE == 1, dec);
    const int cfg = dec ? (cfg_resolved ? cfg_resolved : ((n_pairs <= top_k) ? 58 : 59)) : cfg_resolved;
#define SI_MOE_TILE5(MV, NSGV, CV, HV)                                                                                       \
    do {                                                                                                                 \
        constexpr int NSG = NSGV, C = CV, RB = NSG * C;                                                                  \
        if ((K % 256) != 0 || (N % RB) != 0 || n_pairs <= 0 || n_expert <= 0) {                                                         \
            return false;                                                                                                 \
        }                                                                                                                \
        if (dbg) {                                                                                                       \
            fprintf(stderr, "[grp] fuse=%d qt=%u M=%d C=%d NSG=%d HOIST=%d KC=0 RB=%d K=%d N=%d grid=%zu n_expert=%d n_pairs=%d flat=%d persist=%d\n", FUSE,  \
                    (unsigned)QT, MV, C, NSG, HV, RB, K, N,                                                       \
                    (size_t)((n_pairs + MV - 1) / MV + n_expert) * (size_t)(N / RB), n_expert, n_pairs, flat ? 1 : 0, persist ? 1 : 0);  \
        }                                                                                                                \
        moe_grp_impl<QT, FUSE, MV, C, NSG, HV>(q, w0, w1, plane0, plane1, slot_map, off, pair, active, n_active,          \
                                                n_act_max, n_expert, n_pairs, flat, xq, asa, xs, out, out_stride, K, N,    \
                                                top_k, persist);                    \
        return true;                                                                                                     \
    } while (0)
#define SI_MOE_TILE3(MV, NSGV, CV) SI_MOE_TILE5(MV, NSGV, CV, 1)
#define SI_MOE_TILE6(MV, NSGV, CV, HV, KCV)                                                                               \
    do {                                                                                                                 \
        constexpr int NSG = NSGV, C = CV, RB = NSG * C;                                                                  \
        if ((K % KCV) != 0 || (N % RB) != 0 || n_pairs <= 0 || n_expert <= 0) {                                                         \
            return false;                                                                                                 \
        }                                                                                                                \
        if (dbg) {                                                                                                       \
            fprintf(stderr, "[grp] fuse=%d qt=%u M=%d C=%d NSG=%d HOIST=%d KC=%d RB=%d K=%d N=%d grid=%zu n_expert=%d n_pairs=%d flat=%d persist=%d\n",   \
                    FUSE, (unsigned)QT, MV, C, NSG, HV, KCV, RB, K, N,                                                    \
                    (size_t)((n_pairs + MV - 1) / MV + n_expert) * (size_t)(N / RB), n_expert, n_pairs, flat ? 1 : 0, persist ? 1 : 0);                               \
        }                                                                                                                \
        moe_grp_impl<QT, FUSE, MV, C, NSG, HV, KCV>(q, w0, w1, plane0, plane1, slot_map, off, pair, active, n_active,       \
                                                  n_act_max, n_expert, n_pairs, flat, xq, asa, xs, out, out_stride, K, \
                                                  N, top_k, persist);                  \
        return true;                                                                                                     \
    } while (0)
#define SI_MOE_TILE(NSGV, CV) SI_MOE_TILE3(16, NSGV, CV)
    switch (cfg) {
    case 1: SI_MOE_TILE(32, 2);   // 256 thr, 64 col
    case 2: SI_MOE_TILE(16, 1);   // 128 thr, 16 col
    case 3: SI_MOE_TILE(8, 2);    // 64 thr, 16 col
    case 4: SI_MOE_TILE(32, 1);   // 256 thr, 32 col
    case 7: SI_MOE_TILE(64, 1);   // 512 thr, 64 col
    case 8: SI_MOE_TILE(64, 2);   // 512 thr, 128 col
    // M sweep at C=1 / NSG=32: M is the weight-re-read knob (PF_MOE_HIST puts
    // the factor at 2.26 for M=16, 1.45 for M=32, 1.15 for M=64) and the
    // accumulator is M*FUSE*C floats, so C=1 is what lets M grow at all.
    case 9:  SI_MOE_TILE3(32, 32, 1);
    case 10: SI_MOE_TILE3(64, 32, 1);
    case 11: SI_MOE_TILE3(32, 16, 1);
    case 12: SI_MOE_TILE3(64, 16, 1);
    case 13: SI_MOE_TILE3(24, 32, 1);
    case 14: SI_MOE_TILE3(48, 32, 1);
    case 15: SI_MOE_TILE3(32, 64, 1);
    // M > 16 spills because acc is M*FUSE*C and the hoisted row is M more
    // registers.  These drop the hoist (M*nsb*8 extra `pair` loads and
    // divisions) to test whether that alone is enough to stop the spill -
    // the only route left to the smaller weight-re-read factor.
    case 16: SI_MOE_TILE5(32, 32, 1, 0);
    case 17: SI_MOE_TILE5(24, 32, 1, 0);
    case 18: SI_MOE_TILE5(48, 32, 1, 0);
    case 19: SI_MOE_TILE5(64, 32, 1, 0);
    case 20: SI_MOE_TILE5(32, 16, 1, 0);
    case 21: SI_MOE_TILE5(16, 32, 1, 0);
    // SLM activation staging (KC > 0), as an A/B only.  Correctness first: the
    // M=16 staged forms are wrong in exactly one row (512/8192 = one row of 512
    // columns) and M=8 in two, so only the M=32 forms survive `dev/grp_min`, and
    // those are then 1.3-1.4x SLOWER than reading the activation from global
    // (gate 9.0-9.3 vs 7.06, down 4.9-5.4 vs 4.12) - M=32 costs a spill and the
    // activation re-read was never the bottleneck, which is what the earlier
    // PF_MOE_ABL=8 measurement said (ABL=8 was within 1% of the full kernel).
    // KC=128 (cases 31/34) measures 6x FASTER and 8192/8192 wrong: it is the
    // same artifact that produced the first "4.9x" reading, and those cases are
    // deleted rather than left as a switch that trades correctness for speed.
    case 30: SI_MOE_TILE6(32, 32, 1, 1, 512);
    case 32: SI_MOE_TILE6(32, 16, 1, 1, 256);
    // The register-budget sweep.  The kernel is NOT weight-bandwidth-bound (it
    // sustains 50-68 GB/s of unique weight against a 376 GB/s card), so weight
    // traffic is the cheap currency and `M*FUSE*C/(M + FUSE*C)` - dp4a issued
    // per memory op - is the expensive one.  That ratio is maximised, at a FIXED
    // register budget P = M*FUSE*C, when M == FUSE*C; the shipped tile has M=16
    // against FUSE*C=4 (gate+up) and FUSE*C=1 (`down`, ratio 0.94, the worst
    // point on the curve).  So these cases move budget OUT of M and INTO C,
    // holding P at or below 64.
    case 40: SI_MOE_TILE3(8, 16, 4);   // gate P=64 ratio 4.0
    case 41: SI_MOE_TILE3(8, 32, 4);   // gate P=64 ratio 4.0, RB=128
    case 42: SI_MOE_TILE3(4, 16, 8);   // gate P=64 ratio 3.2, RB=128
    case 43: SI_MOE_TILE3(8, 16, 2);   // gate P=32 ratio 2.7
    case 44: SI_MOE_TILE3(16, 16, 1);  // gate P=32 ratio 2.4
    case 45: SI_MOE_TILE3(4, 16, 4);   // gate P=32 ratio 2.7
    case 46: SI_MOE_TILE3(8, 32, 2);   // down P=16 ratio 1.6
    case 47: SI_MOE_TILE3(4, 32, 4);   // down P=16 ratio 2.0
    case 48: SI_MOE_TILE3(8, 32, 8);   // down P=64 ratio 4.4
    case 49: SI_MOE_TILE3(4, 32, 8);   // down P=32 ratio 2.7
    case 50: SI_MOE_TILE3(32, 32, 2);  // down P=64 ratio 3.0
    case 51: SI_MOE_TILE3(16, 16, 4);  // down P=64 ratio 3.6
    // Closing in on M=8 C=2, which beat both projections.  What the sweep says
    // binds is the register budget P = M*FUSE*C and not the dp4a-per-load
    // ratio: at equal P, M=8/C=2 (6.0 ms) beats M=4/C=4 (10.0) because M=4
    // doubles the weight re-read and this kernel is nowhere near bandwidth-bound.
    case 52: SI_MOE_TILE3(12, 16, 2);
    case 53: SI_MOE_TILE3(8, 8, 2);   // 64 thr, RB=16
    case 54: SI_MOE_TILE3(8, 64, 2);  // 512 thr, RB=128
    case 55: SI_MOE_TILE3(16, 8, 2);  // 128 thr, RB=16
    case 56: SI_MOE_TILE3(24, 16, 2);
    case 57: SI_MOE_TILE3(8, 16, 2);  // == the winner, kept as a named case
    // Decode-shaped tiles: a decode step routes ONE token per expert, so the
    // M=8 tile is seven eighths empty - the same weights are read but eight
    // times the dp4a work is done.  M=1 is the shape a decode step actually
    // has.  (PF_MOE_DP4A_DEC=1 runs the grouped dp4a path on a decode step;
    // these are the tiles for it.)
    case 58: SI_MOE_TILE3(1, 16, 2);  // decode: one row per expert
    case 59: SI_MOE_TILE3(2, 16, 2);  // two rows per expert
    default: SI_MOE_TILE(16, 2);  // 128 thr, 32 col
    }
#undef SI_MOE_TILE
#undef SI_MOE_TILE3
#undef SI_MOE_TILE5
#undef SI_MOE_TILE6
}

} // namespace

int moe_grp_tile_rows(queue & q, bool down, bool dec) {
    const int cfg = moe_cfg_for(q, down, dec);
    if (cfg <= 0) {
        // A decode's "auto" is one row per expert; the non-decode dispatch has
        // no 0 case and lands on its M=16 default.
        return dec ? 1 : 16;
    }
    return moe_cfg_tile_m(cfg);
}

bool moe_grp_gate_up_launch(queue & q, const void * gbank, const void * ubank, const int32_t * slot_map,
                            size_t slot_gb, size_t slot_ub, const int32_t * off, const int32_t * pair,
                            const int32_t * active, const int32_t * n_active, int n_act_max, int n_expert,
                            int n_pairs, const int8_t * xq, const uint16_t * asa, const float * xs, float * h, int h_stride,
                            int n_embd, int n_ff_exp, int top_k, uint32_t gtype, uint32_t utype, bool dec) {
    // one work-group reads both banks, so they must agree on K, N and type
    if (gtype != utype) {
        return false;
    }
    // PF_MOE_FLAT puts the row tile in the grid (one work-group = one tile)
    // instead of looping it inside an (expert, column block) work-group.  The
    // trade is not uniform across the two projections - see the note on `flat`
    // in moe_grp_impl and AGENTS.md.
    //
    // A DECODE step uses the looped grid, and that is a measured decision, not
    // a default.  The flat grid has to be sized from a host-side UPPER bound
    // (sum_e ceil(p_e/M) <= n_pairs/M + n_expert, because p_e lives on the
    // device), so on a decode - where n_pairs == top_k and every selected
    // expert has exactly one tile - it launches n_expert surplus work-groups
    // that each scan the whole CSR and exit.  qwen4exp's bank has 10 active
    // experts of 512 per layer, so the flat gate/up grid is 10440 work-groups
    // against the looped form's 200, and the measured decode is 426.3 ms/token
    // against 395.4 (interleaved A/B, two pairs per setting, 37 banked layers).
    // The prefill keeps the flat grid: its tile count is genuinely large, so
    // the surplus is ~1/3 and lifting the tile out of the work-group is what
    // makes every work-group exactly one tile.
    // PF_MOE_FLAT_DEC=1 restores the flat grid on a decode for the A/B.
    const si::dev::profile & dp = si::dev::for_queue(q);
    const bool flat_dec = si::dev::profile_flag(q, "PF_MOE_FLAT_DEC", dp.moe.flat_dec != 0);
    const bool flat = dec ? flat_dec : si::dev::profile_flag(q, "PF_MOE_FLAT", dp.moe.flat != 0);
    static const int persist_env = si::env::i32("PF_MOE_PERSIST", 0);
    const char * g = (const char *)gbank, * u = (const char *)ubank;
#define SI_MOE_GRP_GU(QT)                                                                                                   \
    moe_grp_go<2, QT>(q, g, u, slot_gb, slot_ub, slot_map, off, pair, active, n_active, n_act_max, n_expert, n_pairs,  \
                      flat, xq, asa, xs, h, h_stride, n_embd, n_ff_exp, top_k, persist_env != 0, dec)
    switch (gtype) {
    case 12: return SI_MOE_GRP_GU(12);
    case 13: return SI_MOE_GRP_GU(13);
    case 14: return SI_MOE_GRP_GU(14);
    default: return false;
    }
#undef SI_MOE_GRP_GU
}

bool moe_grp_down_launch(queue & q, const void * dbank, const int32_t * slot_map, size_t slot_db, const int32_t * off,
                         const int32_t * pair, const int32_t * active, const int32_t * n_active, int n_act_max,
                         int n_expert, int n_pairs, const int8_t * xq, const uint16_t * asa, const float * xs, float * acc,
                         int acc_stride, int n_embd, int n_ff_exp, uint32_t dtype, bool dec) {
    const char * d = (const char *)dbank;
    const bool flat = si::env::i32("PF_MOE_FLAT_DOWN", 0) != 0;
    // PF_MOE_PERSIST: work-stealing grid.  Read here, ONCE per process, and passed
    // as a kernel argument - reading an env var inside a kernel body is the trap
    // this file already records ("reading env inside a kernel").
    static const int persist_env = si::env::i32("PF_MOE_PERSIST", 0);
#define SI_MOE_GRP_D(QT)                                                                                                   \
    moe_grp_go<1, QT>(q, d, nullptr, slot_db, 0, slot_map, off, pair, active, n_active, n_act_max, n_expert, n_pairs, \
                      flat, xq, asa, xs, acc, acc_stride, n_ff_exp, n_embd, 1, persist_env != 0, dec)
    switch (dtype) {
    case 12: return SI_MOE_GRP_D(12);
    case 13: return SI_MOE_GRP_D(13);
    case 14: return SI_MOE_GRP_D(14);
    default: return false;
    }
#undef SI_MOE_GRP_D
}

// ---------------------------------------------------------------------------
// Native Q5_1 down, for the qwen4exp expert bank.
//
// WHY.  The grouped dp4a down above reads K-quant superblocks (256 values,
// eight 32-value groups with a shared d/dmin), and Q5_1 is neither: its block
// IS one 32-value group with its own d and m, and qwen4exp's expert FFN down
// has K = n_ff_exp = 640 = 20 groups, not a multiple of 256.  So
// `moe_grp_down_launch` declines the shape and the fp32-lane dequant kernel
// (`moe_down_impl<7>`) took over - a full 5-bit reconstruction and an fp32 FMA
// per weight, which is why the per-expert LRU bank prefilled at 50 s instead
// of the ~32 s the gate/up path suggests.
//
// The format is dp4a-friendly once the nibble split is respected.  A Q5_1
// block is 24 B per 32 values:
//
//     d (f16 at 0)  m (f16 at 2)  qh (u32 at 4)  qs[16] at 8
//     q_i = ((qs[i%16] >> 4*(i/16)) & 0xF) | (((qh >> i) & 1) << 4)
//
// and a lane's four values i = 4*(lane%8) .. +3 all fall in ONE 16-value half,
// so the nibble shift is a single uniform `>> 4` and the four high bits are one
// shift-and-spread out of qh - exactly one dp4a operand per lane per group,
// with w = d*q + m so the group sum (`xs`) carries the +m term:
//
//     out = sum_g asa_g * (d_g * dot(q, xq) + m_g * sum(xq))
//
// WORK DECOMPOSITION.  One warp per output row and NPB rows walked per warp.
// The 32 lanes split the row's groups four ways (`lane/8` is the group slice,
// `lane%8` the four-value chunk), so for K=640 each lane owns five groups; the
// row's blocks are decoded ONCE into registers and the expert's routed pairs
// are looped INSIDE, so a weight byte is read once for all of the expert's
// pairs rather than once per pair (the fp32 kernel's superblock-outer,
// pair-inner order is the same idea, but per 8-pair tile).  One warp reduction
// per pair folds the four group slices.
//
// The whole-row register cache is why the group count per lane is a template
// constant: a runtime-indexed array would land in private scratch, and a
// scratch-resident accumulator is ~350x slower on this part (see the note on
// moe_grp_impl).  `ng > 4*NGL` walks the row in passes and adds the partial
// sums; K=640 is exactly one pass.
// ---------------------------------------------------------------------------
template <int NPB>
static void moe_down_q5_1_impl(queue & q, const char * dbank, const int32_t * slot_map, size_t slot_db,
                               const int32_t * off, const int32_t * pair, const int32_t * active,
                               const int32_t * n_active, int n_act_max, const int8_t * xq, const uint16_t * asa,
                               const float * xs, float * acc, int acc_stride, int n_embd, int n_ff_exp) {
    constexpr int SG = 8;   // lanes per 32-value group: 8 x 4 values
    constexpr int GSL = 4;  // group slices per warp (SG * GSL == 32 lanes)
    constexpr int NGL = 5;  // groups per lane per pass: 20 = K/32 for this model
    const int ng = n_ff_exp / 32;
    const size_t row_bytes = (size_t)ng * sizeof(block_q5_1);
    const int nrb = (n_embd + NPB - 1) / NPB;
    const int nw = n_act_max * nrb;
    if (nw <= 0) {
        return;
    }
    q.parallel_for(nd_range<1>(moe_grid((size_t)nw * 32), kMoEWG),
                   [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int lid = (int)it.get_local_id(0);
        const int sg = lid / 32;
        const int lane = lid % 32;
        const sub_group sgg = it.get_sub_group();
        const int w = (int)it.get_group(0) * kMoENW + sg; // global warp id
        if (w >= nw) {
            return;
        }
        const int ia = w / nrb;
        const int nb = w % nrb;
        if (ia >= n_active[0]) {
            return;
        }
        const int e = active[ia];
        const int slot = slot_map[e];
        if (slot < 0) {
            return; // host-resident expert: the host computes it
        }
        const int b0 = off[e], b1 = off[e + 1];
        const int gs = lane / SG; // which group slice this lane owns
        const int ch = lane % SG; // the lane's four-value chunk inside a group
        const char * base = dbank + (size_t)slot * slot_db;
        for (int rr = 0; rr < NPB; rr++) {
            const int n = nb * NPB + rr;
            if (n >= n_embd) {
                break;
            }
            const char * dr = base + (size_t)n * row_bytes;
            for (int g0 = 0; g0 < ng; g0 += GSL * NGL) {
                // Decode this pass's groups once, into registers.  `gr` is
                // compile-time-offset inside the unrolled loop, which is what
                // keeps these arrays in registers instead of scratch.
                uint32_t wq[NGL];
                float wd[NGL], wm[NGL];
#pragma unroll
                for (int j = 0; j < NGL; j++) {
                    const int gr = g0 + gs + GSL * j;
                    wq[j] = 0u;
                    wd[j] = 0.f;
                    wm[j] = 0.f;
                    if (gr < ng) {
                        const char * b = dr + (size_t)gr * sizeof(block_q5_1);
                        const auto * blk = (const block_q5_1 *)b;
                        const uint32_t qh = *reinterpret_cast<const uint32_t *>(blk->qh);
                        const uint32_t v = *reinterpret_cast<const uint32_t *>(blk->qs + 4 * (ch & 3));
                        // i = 4*ch .. +3 is entirely inside one 16-value half,
                        // so every one of the lane's four values takes the same
                        // nibble shift; the high bit plane is per value.
                        const uint32_t nib = (ch < 4) ? (v & 0x0F0F0F0Fu) : ((v >> 4) & 0x0F0F0F0Fu);
                        const uint32_t hb = (qh >> (4 * ch)) & 0xFu;
                        const uint32_t hi = ((hb & 1u) << 4) | ((hb & 2u) << 11) | ((hb & 4u) << 18) |
                                            ((hb & 8u) << 25);
                        wq[j] = nib | hi;
                        wd[j] = fast_h2f(blk->d);
                        wm[j] = fast_h2f(blk->m);
                    }
                }
                for (int i0 = b0; i0 < b1; i0++) {
                    const int pr = pair[i0];
                    const int8_t * xr = xq + (size_t)pr * (size_t)n_ff_exp;
                    const uint16_t * ar = asa + (size_t)pr * (size_t)ng;
                    const float * sr = xs + (size_t)pr * (size_t)ng;
                    float a = 0.f;
#pragma unroll
                    for (int j = 0; j < NGL; j++) {
                        const int gr = g0 + gs + GSL * j;
                        if (gr < ng) {
                            const uint32_t xa =
                                *reinterpret_cast<const uint32_t *>(xr + (size_t)gr * 32 + (size_t)ch * 4);
                            const int32_t qd = dp4a_s8u8(xa, wq[j], 0);
                            const float A = fast_h2f(ar[gr]);
                            a = sycl::fma(A * wd[j], (float)qd, a);
                            // The +m zero point is `m * (sum of the WHOLE
                            // 32-value group)`, and a group is split over eight
                            // lanes here.  Adding it on every lane multiplies it
                            // by eight - which is exactly how this kernel first
                            // measured cos 0.08 with an output ~16x too large
                            // (m is negative and comparable to d*q).  One lane
                            // per group carries it, so the warp reduction sums
                            // it once.
                            if (ch == 0) {
                                a = sycl::fma(A * wm[j], sr[gr], a);
                            }
                        }
                    }
                    // every lane must enter the collective
                    const float v = sg_sum(a, sgg);
                    if (lane == 0) {
                        // one pass for K=640; the later passes only exist so a
                        // wider Q5_1 tensor is still correct.
                        if (g0 == 0) {
                            acc[(size_t)pr * (size_t)acc_stride + n] = v;
                        } else {
                            acc[(size_t)pr * (size_t)acc_stride + n] += v;
                        }
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Q5_1 down on the 128-value panel repack (`q5_1_sb_pack`, PF_Q4_Q5SB).
//
// The same work decomposition, lane/group/pair map and FMA order as the SG=8
// Q5_1 kernel above; only the block decode changes.  A panel covers four
// consecutive 32-value groups in 96 B (64 qs + 16 qh + 16 meta), byte-identical
// to four Q5_1 blocks, and a lane's uint32 at qs + 32p + 4*ch carries both
// groups of a pair (low nibble = even group, high = odd), so the warp reads the
// plane as 32 B contiguous per instruction instead of 16 B per block.  The
// panel is `gr>>2`, the pair `(gr>>1)&1`, the group-in-pair `gr&3`; the f16
// d/m bits are copied raw by the host pack, so the arithmetic is bit-identical
// to `moe_down_q5_1_impl` (verified by PF_DUMP_DEC_LOGITS).
// ---------------------------------------------------------------------------
template <int NPB>
static void moe_down_q5_1_sb_impl(queue & q, const char * dbank, const int32_t * slot_map, size_t slot_db,
                                  const int32_t * off, const int32_t * pair, const int32_t * active,
                                  const int32_t * n_active, int n_act_max, const int8_t * xq, const uint16_t * asa,
                                  const float * xs, float * acc, int acc_stride, int n_embd, int n_ff_exp) {
    constexpr int SG = 8;   // lanes per 32-value group: 8 x 4 values
    constexpr int GSL = 4;  // group slices per warp (SG * GSL == 32 lanes)
    constexpr int NGL = 5;  // groups per lane per pass: 20 = K/32 for this model
    const int ng = n_ff_exp / 32;
    if (ng <= 0 || (ng % 4) != 0) {
        return; // the repack only exists for whole 128-value panels
    }
    const size_t row_bytes = (size_t)(ng / 4) * 96;
    const int nrb = (n_embd + NPB - 1) / NPB;
    const int nw = n_act_max * nrb;
    if (nw <= 0) {
        return;
    }
    q.parallel_for(nd_range<1>(moe_grid((size_t)nw * 32), kMoEWG),
                   [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int lid = (int)it.get_local_id(0);
        const int sg = lid / 32;
        const int lane = lid % 32;
        const sub_group sgg = it.get_sub_group();
        const int w = (int)it.get_group(0) * kMoENW + sg; // global warp id
        if (w >= nw) {
            return;
        }
        const int ia = w / nrb;
        const int nb = w % nrb;
        if (ia >= n_active[0]) {
            return;
        }
        const int e = active[ia];
        const int slot = slot_map[e];
        if (slot < 0) {
            return; // host-resident expert: the host computes it
        }
        const int b0 = off[e], b1 = off[e + 1];
        const int gs = lane / SG; // which group slice this lane owns
        const int ch = lane % SG; // the lane's four-value chunk inside a group
        const char * base = dbank + (size_t)slot * slot_db;
        // Panel/pair/sub for group gr = g0 + gs + GSL*j: g0 is a multiple of 4
        // and gs < GSL == 4, so the low two bits of gr are `gs`.  The offsets
        // are therefore per-lane invariants the compiler can hoist out of the
        // unrolled j loop:
        //   panel = (gr>>2) = g0/4 + j, pair = (gr>>1)&1 = gs>>1,
        //   sub   = gr&3   = gs.
        const int qs_lane = 32 * (gs >> 1) + 4 * ch; // uint32 in the pair's qs
        const int qh_lane = 64 + 2 * ch;             // uint16 of high bits
        const int me_lane = 80 + 4 * gs;             // uint32 {f16 d, f16 m}
        const uint32_t qh_sh = 4u * (uint32_t)gs;
        const uint32_t nib_sh = (uint32_t)(gs & 1) * 4u; // odd group = high nibble
        for (int rr = 0; rr < NPB; rr++) {
            const int n = nb * NPB + rr;
            if (n >= n_embd) {
                break;
            }
            const char * dr = base + (size_t)n * row_bytes;
            for (int g0 = 0; g0 < ng; g0 += GSL * NGL) {
                const char * dr0 = dr + (size_t)(g0 / GSL) * 96;
                // Decode this pass's groups once, into registers.  A panel is
                // compile-time-offset inside the unrolled loop, which is what
                // keeps these arrays in registers instead of scratch.
                uint32_t wq[NGL];
                float wd[NGL], wm[NGL];
#pragma unroll
                for (int j = 0; j < NGL; j++) {
                    const int gr = g0 + gs + GSL * j;
                    wq[j] = 0u;
                    wd[j] = 0.f;
                    wm[j] = 0.f;
                    if (gr < ng) {
                        const char * pb = dr0 + (size_t)j * 96;
                        const uint32_t v = *reinterpret_cast<const uint32_t *>(pb + qs_lane);
                        // The panel stores the pair's even group in the low
                        // nibble and the odd group in the high one.
                        const uint32_t nib = (v >> nib_sh) & 0x0F0F0F0Fu;
                        const uint16_t hw = *reinterpret_cast<const uint16_t *>(pb + qh_lane);
                        const uint32_t hb = (uint32_t)(hw >> qh_sh) & 0xFu;
                        const uint32_t hi = ((hb & 1u) << 4) | ((hb & 2u) << 11) | ((hb & 4u) << 18) |
                                            ((hb & 8u) << 25);
                        wq[j] = nib | hi;
                        const uint32_t meta = *reinterpret_cast<const uint32_t *>(pb + me_lane);
                        wd[j] = fast_h2f((uint16_t)(meta & 0xFFFFu));
                        wm[j] = fast_h2f((uint16_t)(meta >> 16));
                    }
                }
                for (int i0 = b0; i0 < b1; i0++) {
                    const int pr = pair[i0];
                    const int8_t * xr = xq + (size_t)pr * (size_t)n_ff_exp;
                    const uint16_t * ar = asa + (size_t)pr * (size_t)ng;
                    const float * sr = xs + (size_t)pr * (size_t)ng;
                    float a = 0.f;
#pragma unroll
                    for (int j = 0; j < NGL; j++) {
                        const int gr = g0 + gs + GSL * j;
                        if (gr < ng) {
                            const uint32_t xa =
                                *reinterpret_cast<const uint32_t *>(xr + (size_t)gr * 32 + (size_t)ch * 4);
                            const int32_t qd = dp4a_s8u8(xa, wq[j], 0);
                            const float A = fast_h2f(ar[gr]);
                            a = sycl::fma(A * wd[j], (float)qd, a);
                            // One lane per group carries the +m zero point, so
                            // the warp reduction sums it exactly once.
                            if (ch == 0) {
                                a = sycl::fma(A * wm[j], sr[gr], a);
                            }
                        }
                    }
                    // every lane must enter the collective
                    const float v = sg_sum(a, sgg);
                    if (lane == 0) {
                        if (g0 == 0) {
                            acc[(size_t)pr * (size_t)acc_stride + n] = v;
                        } else {
                            acc[(size_t)pr * (size_t)acc_stride + n] += v;
                        }
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// One lane per 32-value group (`PF_Q4_Q5_LANE=1`).
//
// WHY.  The SG=8 kernel above splits a block across eight lanes, so the warp
// issues four memory instructions per group slice (qh, qs word, d, m) and the
// 32 lanes together cover only four blocks per instruction group.  Cold-stream
// numbers (dev/bench_q5cold.cpp: a 256-slot / 300 MB bank, 25 disjoint 10-slot
// subsets so the reuse distance is ~300 MB, device time from barrier pairs;
// K=640 N=2560, 10 experts):
//
//     shipped (SG=8, NPB=8)          73.8 GB/s     (also 79.5 at 256 experts)
//     shipped (SG=8, NPB=1)          81-85 GB/s    (136.5 at 256 experts)
//     one lane per group, NPB=1      98-101 GB/s   (191.8 at 256 experts)
//     grouped Q4_K down, same bytes  178 GB/s      (281.5 at 256 experts)
//     pure coalesced uint4 read      363.5 GB/s    (ceiling)
//
// so the ~65 GB/s is not the DRAM stream, it is the instruction mix: SG=8 needs
// one load per 24 B of row, and the decode->dp4a->shuffle chain behind each is
// what the warp waits on.  Giving one lane the whole block instead loads its
// 24 B with qh (4 B), d+m (4 B) and two 8-byte qs loads and runs eight dp4a
// back to back; that is 2-4 loads for 20 blocks per warp-row instead of 20,
// and the whole weight row is read once per warp.  The old probe
// (dev/bench_q5down.cpp) replayed 12.29 MB against a 16 MB L2 and reported a
// 3.3-4.0x; that number was L2 and is withdrawn - cold, the gain is 1.3x.
//
// But with one block per lane only 20 of 32 lanes work, so the kernel needs
// concurrency, not rows per warp: NPB=1 (one row per warp, 25600 warps at a
// decode's ~10 experts) is the best shape (PF_Q4_Q5_LANE_NPB).  The extreme
// sensitivity to NPB is the idle-lane trade: at 20/32 lanes the warp is
// latency-bound and every wave matters.
//
// NOT BIT-IDENTICAL to the SG=8 kernel: each lane now sums all eight chunks of
// its group and there is one 32-lane reduction per row, so the fp32
// association of a 640-value dot changes.  The computed sum is the same
// (rel-rms 1.5e-7 against the SG=8 kernel and ~1e-4 against an f64-style host
// reference, i.e. inside the kernel's own existing error).  The weight decode
// is hoisted out of the pair loop, so a prefill's many pairs still read each
// row once.
//
// Two vector loads are deliberate: `qs` starts at byte 8 of a 24-byte block,
// so it is 8-byte aligned but only every other block is 16-byte aligned.  A
// uint4 load there is a misaligned 16-byte access (the AOT-widening trap the
// Q8_0 GEMV hit); two uint2 loads at +0/+8 are safe and measure the same.
// ---------------------------------------------------------------------------
template <int NPB>
static void moe_down_q5_1_lane_impl(queue & q, const char * dbank, const int32_t * slot_map, size_t slot_db,
                                    const int32_t * off, const int32_t * pair, const int32_t * active,
                                    const int32_t * n_active, int n_act_max, const int8_t * xq,
                                    const uint16_t * asa, const float * xs, float * acc, int acc_stride, int n_embd,
                                    int n_ff_exp) {
    const int ng = n_ff_exp / 32;
    const size_t row_bytes = (size_t)ng * sizeof(block_q5_1);
    const int nrb = (n_embd + NPB - 1) / NPB;
    const int nw = n_act_max * nrb;
    if (nw <= 0) {
        return;
    }
    q.parallel_for(nd_range<1>(moe_grid((size_t)nw * 32), kMoEWG),
                   [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int lid = (int)it.get_local_id(0);
        const int sg = lid / 32;
        const int lane = lid % 32;
        const sub_group sgg = it.get_sub_group();
        const int w = (int)it.get_group(0) * kMoENW + sg; // global warp id
        if (w >= nw) {
            return;
        }
        const int ia = w / nrb;
        const int nb = w % nrb;
        if (ia >= n_active[0]) {
            return;
        }
        const int e = active[ia];
        const int slot = slot_map[e];
        if (slot < 0) {
            return; // host-resident expert: the host computes it
        }
        const int b0 = off[e], b1 = off[e + 1];
        const char * base = dbank + (size_t)slot * slot_db;
        for (int rr = 0; rr < NPB; rr++) {
            const int n = nb * NPB + rr;
            if (n >= n_embd) {
                break;
            }
            const char * dr = base + (size_t)n * row_bytes;
            // ng == 20 on qwen4exp; the pass loop keeps a wider Q5_1 tensor
            // correct (32 groups per pass, one per lane).
            for (int g0 = 0; g0 < ng; g0 += 32) {
                const int gr = g0 + lane;
                // Decode this lane's group once, outside the pair loop.
                uint32_t wq[8];
                float wd = 0.f, wm = 0.f;
                if (gr < ng) {
                    const auto * blk =
                        (const block_q5_1 *)(dr + (size_t)gr * sizeof(block_q5_1));
                    const uint32_t qh = *reinterpret_cast<const uint32_t *>(blk->qh);
                    // d at 0, m at 2: one 4-byte value (little endian).
                    const uint32_t dm = ((uint32_t)blk->m << 16) | (uint32_t)blk->d;
                    const uint2 q0 = *reinterpret_cast<const uint2 *>(blk->qs);
                    const uint2 q1 = *reinterpret_cast<const uint2 *>(blk->qs + 8);
                    const uint32_t qw[4] = {q0.x(), q0.y(), q1.x(), q1.y()};
                    wd = fast_h2f((uint16_t)(dm & 0xFFFFu));
                    wm = fast_h2f((uint16_t)(dm >> 16));
#pragma unroll
                    for (int c = 0; c < 8; c++) {
                        const uint32_t v = qw[c & 3];
                        const uint32_t nib = (c < 4) ? (v & 0x0F0F0F0Fu) : ((v >> 4) & 0x0F0F0F0Fu);
                        const uint32_t hb = (qh >> (4 * c)) & 0xFu;
                        const uint32_t hi = ((hb & 1u) << 4) | ((hb & 2u) << 11) | ((hb & 4u) << 18) |
                                            ((hb & 8u) << 25);
                        wq[c] = nib | hi;
                    }
                }
                for (int i0 = b0; i0 < b1; i0++) {
                    const int pr = pair[i0];
                    const int8_t * xr = xq + (size_t)pr * (size_t)n_ff_exp;
                    const uint16_t * ar = asa + (size_t)pr * (size_t)ng;
                    const float * sr = xs + (size_t)pr * (size_t)ng;
                    float a = 0.f;
                    if (gr < ng) {
                        const float A = fast_h2f(ar[gr]);
#pragma unroll
                        for (int c = 0; c < 8; c++) {
                            const uint32_t xa = *reinterpret_cast<const uint32_t *>(
                                xr + (size_t)gr * 32 + (size_t)c * 4);
                            const int32_t qd = dp4a_s8u8(xa, wq[c], 0);
                            a = sycl::fma(A * wd, (float)qd, a);
                        }
                        // The +m term is the whole 32-value group's sum, and
                        // this lane owns the whole group, so it is added once.
                        a = sycl::fma(A * wm, sr[gr], a);
                    }
                    const float v = sg_sum(a, sgg);
                    if (lane == 0) {
                        if (g0 == 0) {
                            acc[(size_t)pr * (size_t)acc_stride + n] = v;
                        } else {
                            acc[(size_t)pr * (size_t)acc_stride + n] += v;
                        }
                    }
                }
            }
        }
    });
}

void moe_down_q5_1_launch(queue & q, const void * dbank, const int32_t * slot_map, size_t slot_db,
                          const int32_t * off, const int32_t * pair, const int32_t * active,
                          const int32_t * n_active, int n_act_max, int n_expert, int n_pairs, const int8_t * xq,
                          const uint16_t * asa, const float * xs, float * acc, int acc_stride, int n_embd,
                          int n_ff_exp) {
    (void)n_expert; // the grid is bounded by n_act_max, as the callers pass it
    if (n_pairs <= 0 || n_embd <= 0 || (n_ff_exp % 32) != 0) {
        return;
    }
    // PF_Q4_Q5_LANE=1 switches to the one-lane-per-group kernel above; the
    // default 0 keeps the shipped SG=8 kernel so the two can be A/B'd.
    // On that path NPB=1 is the measured optimum (see the note above): the
    // warp uses only 20 of its 32 lanes, so it needs waves, not rows.
    // Only NPB=1 is instantiated: it is the measured optimum for this kernel
    // (the warp uses 20 of its 32 lanes, so it needs waves, not rows), and each
    // extra instantiation is another AOT device image for a path that is
    // opt-in and not bit-identical.
    static const int lane_mode = si::env::i32("PF_Q4_Q5_LANE", 0);
    if (lane_mode != 0) {
        moe_down_q5_1_lane_impl<1>(q, (const char *)dbank, slot_map, slot_db, off, pair, active, n_active, n_act_max,
                                   xq, asa, xs, acc, acc_stride, n_embd, n_ff_exp);
        return;
    }
    // NPB rows per warp.  The kernel is latency-bound, not bandwidth-bound:
    // the grid is n_act x ceil(n_embd/NPB) warps, so at NPB=8 a decode with
    // ~10 active experts launches only ~3200 warps against 4096 resident slots
    // (0.8 of a wave) and each warp's dependent decode->dp4a chain has nothing
    // to hide behind.
    //
    // The default is 1, measured on a COLD stream (dev/bench_q5cold.cpp: a
    // 256-slot / 300 MB bank, 25 disjoint 10-slot subsets per sample, so the
    // reuse distance is ~300 MB).  NPB=8 / NPB=1 / one-lane-per-group are
    // 73.8 / 84.5 / 101.3 GB/s there, and the in-engine AOT kernel times are
    // 0.1935 / 0.1745 ms per layer - the old 8 was chosen from a probe whose
    // 12.29 MB working set was replayed against a 16 MB L2, i.e. from L2, not
    // from DRAM.  NPB=1 is BIT-IDENTICAL to NPB=8 and measured 90.71 ->
    // 89.41 ms/token on the 48-token greedy decode (three alternating runs,
    // same output md5), and the 576-token prefill is not worse.
    // Two instantiations only: the shipped NPB=1 and the old default NPB=8 the
    // A/B compares against.  2 and 4 measured no better than 1 and would each
    // cost another AOT device image.
    static const int npb = si::env::i32("PF_Q4_Q5NPB", 1);
    if (npb == 8) {
        moe_down_q5_1_impl<8>(q, (const char *)dbank, slot_map, slot_db, off, pair, active, n_active, n_act_max, xq,
                              asa, xs, acc, acc_stride, n_embd, n_ff_exp);
    } else {
        moe_down_q5_1_impl<1>(q, (const char *)dbank, slot_map, slot_db, off, pair, active, n_active, n_act_max, xq,
                              asa, xs, acc, acc_stride, n_embd, n_ff_exp);
    }
}

// The same contract on the 128-value panel repack (`q5_1_sb_pack`).  The bank
// plane the caller passes holds the repacked bytes, so only a layer the host
// actually repacked may take this entry point.
void moe_down_q5_1_sb_launch(queue & q, const void * dbank, const int32_t * slot_map, size_t slot_db,
                             const int32_t * off, const int32_t * pair, const int32_t * active,
                             const int32_t * n_active, int n_act_max, int n_expert, int n_pairs, const int8_t * xq,
                             const uint16_t * asa, const float * xs, float * acc, int acc_stride, int n_embd,
                             int n_ff_exp) {
    (void)n_expert;
    if (n_pairs <= 0 || n_embd <= 0 || (n_ff_exp % 128) != 0) {
        return;
    }
    // NPB is the same A/B knob as the raw Q5_1 kernel (PF_Q4_Q5NPB, default 1).
    // Both orientations are arithmetic-identical, so the shipped NPB=1 and the
    // old NPB=8 stay instantiated for the comparison.
    static const int npb = si::env::i32("PF_Q4_Q5NPB", 1);
    if (npb == 8) {
        moe_down_q5_1_sb_impl<8>(q, (const char *)dbank, slot_map, slot_db, off, pair, active, n_active, n_act_max,
                                 xq, asa, xs, acc, acc_stride, n_embd, n_ff_exp);
    } else {
        moe_down_q5_1_sb_impl<1>(q, (const char *)dbank, slot_map, slot_db, off, pair, active, n_active, n_act_max,
                                 xq, asa, xs, acc, acc_stride, n_embd, n_ff_exp);
    }
}

// Native Q8_0 down, for the qwen4exp expert bank.
//
// The same contract and work decomposition as the Q5_1 down above, but the
// block is `{f16 d; int8 qs[32]}` (34 B, one 32-value group) and there is no
// minimum.  A lane's four values are four signed int8 weights, so the dot uses
// the signed-by-unsigned dp4a with the weights biased into unsigned:
//
//     q_u = q_s + 128  =>  sum(q_s x) = sum(q_u x) - 128 * sum(x)
//
// and the last term is the activation's group sum `xs[g]`, carried on one lane
// per group (adding it on every lane would count it eight times - the Q5_1
// kernel's original +m bug).  `qs` starts at byte 2, so the four bytes are read
// as two uint16 loads; an unaligned uint32 load is the AOT-widening trap the
// Q8_0 GEMV already hit.
//
// This is what lets the four Q4_K/Q8_0-down layers stop running their routed
// experts on the host.
template <int NPB>
static void moe_down_q8_0_impl(queue & q, const char * dbank, const int32_t * slot_map, size_t slot_db,
                               const int32_t * off, const int32_t * pair, const int32_t * active,
                               const int32_t * n_active, int n_act_max, const int8_t * xq, const uint16_t * asa,
                               const float * xs, float * acc, int acc_stride, int n_embd, int n_ff_exp) {
    constexpr int SG = 8;
    constexpr int GSL = 4;
    constexpr int NGL = 5;
    const int ng = n_ff_exp / 32;
    const size_t row_bytes = (size_t)ng * sizeof(block_q8_0);
    const int nrb = (n_embd + NPB - 1) / NPB;
    const int nw = n_act_max * nrb;
    if (nw <= 0) {
        return;
    }
    q.parallel_for(nd_range<1>(moe_grid((size_t)nw * 32), kMoEWG),
                   [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int lid = (int)it.get_local_id(0);
        const int sg = lid / 32;
        const int lane = lid % 32;
        const sub_group sgg = it.get_sub_group();
        const int w = (int)it.get_group(0) * kMoENW + sg;
        if (w >= nw) {
            return;
        }
        const int ia = w / nrb;
        const int nb = w % nrb;
        if (ia >= n_active[0]) {
            return;
        }
        const int e = active[ia];
        const int slot = slot_map[e];
        if (slot < 0) {
            return;
        }
        const int b0 = off[e], b1 = off[e + 1];
        const int gs = lane / SG;
        const int ch = lane % SG;
        const char * base = dbank + (size_t)slot * slot_db;
        for (int rr = 0; rr < NPB; rr++) {
            const int n = nb * NPB + rr;
            if (n >= n_embd) {
                break;
            }
            const char * dr = base + (size_t)n * row_bytes;
            for (int g0 = 0; g0 < ng; g0 += GSL * NGL) {
                uint32_t wq[NGL];
                float wd[NGL];
#pragma unroll
                for (int j = 0; j < NGL; j++) {
                    const int gr = g0 + gs + GSL * j;
                    wq[j] = 0u;
                    wd[j] = 0.f;
                    if (gr < ng) {
                        const auto * blk = (const block_q8_0 *)(dr + (size_t)gr * sizeof(block_q8_0));
                        const char * qp = (const char *)blk->qs + (size_t)ch * 4;
                        const uint16_t lo = *reinterpret_cast<const uint16_t *>(qp);
                        const uint16_t hi = *reinterpret_cast<const uint16_t *>(qp + 2);
                        wq[j] = ((uint32_t)lo | ((uint32_t)hi << 16)) ^ 0x80808080u;
                        wd[j] = fast_h2f(blk->d);
                    }
                }
                for (int i0 = b0; i0 < b1; i0++) {
                    const int pr = pair[i0];
                    const int8_t * xr = xq + (size_t)pr * (size_t)n_ff_exp;
                    const uint16_t * ar = asa + (size_t)pr * (size_t)ng;
                    const float * sr = xs + (size_t)pr * (size_t)ng;
                    float a = 0.f;
#pragma unroll
                    for (int j = 0; j < NGL; j++) {
                        const int gr = g0 + gs + GSL * j;
                        if (gr < ng) {
                            const uint32_t xa =
                                *reinterpret_cast<const uint32_t *>(xr + (size_t)gr * 32 + (size_t)ch * 4);
                            const float A = fast_h2f(ar[gr]);
                            const int32_t qd = dp4a_s8u8(xa, wq[j], 0);
                            a = sycl::fma(A * wd[j], (float)qd, a);
                            // The -128 bias correction is a whole-group sum;
                            // one lane per group carries it, as in Q5_1.
                            if (ch == 0) {
                                a = sycl::fma(A * wd[j], -128.f * sr[gr], a);
                            }
                        }
                    }
                    const float v = sg_sum(a, sgg);
                    if (lane == 0) {
                        if (g0 == 0) {
                            acc[(size_t)pr * (size_t)acc_stride + n] = v;
                        } else {
                            acc[(size_t)pr * (size_t)acc_stride + n] += v;
                        }
                    }
                }
            }
        }
    });
}

void moe_down_q8_0_launch(queue & q, const void * dbank, const int32_t * slot_map, size_t slot_db,
                          const int32_t * off, const int32_t * pair, const int32_t * active,
                          const int32_t * n_active, int n_act_max, int n_expert, int n_pairs, const int8_t * xq,
                          const uint16_t * asa, const float * xs, float * acc, int acc_stride, int n_embd,
                          int n_ff_exp) {
    (void)n_expert;
    if (n_pairs <= 0 || n_embd <= 0 || (n_ff_exp % 32) != 0) {
        return;
    }
    moe_down_q8_0_impl<8>(q, (const char *)dbank, slot_map, slot_db, off, pair, active, n_active, n_act_max, xq, asa,
                          xs, acc, acc_stride, n_embd, n_ff_exp);
}

// ---------------------------------------------------------------------------
// gate + up for every (token, expert) pair in one pass per expert row block.
// Both tensors are read in the same loop so the activation row is fetched once
// for the pair, and what is stored is silu(gate) * up - all `down` needs.
//
// This is the fp32 fallback: `moe_gate_up_launch` above is what the engine
// falls back to when the grouped dp4a path declines a shape, and it is also the
// reference the dp4a path is checked against.  `moe_grp_gate_up_launch` computes
// the same product off the same GGUF blocks with an int8-quantized activation.
template <uint32_t GT, uint32_t UT>
static void moe_gate_up_impl(queue & q, const char * gbank, const char * ubank, const int32_t * slot_map,
                             size_t slot_gb, size_t slot_ub, const int32_t * off, const int32_t * pair,
                             const int32_t * active, const int32_t * n_active, int n_act_max, const float * x,
                             int x_stride, float * h, int h_stride, int n_embd, int n_ff_exp, int top_k, bool dq2, bool noact, int npb) {
    const int n_sb = n_embd / 256;
    const int sgb = superblock_bytes(GT), sub = superblock_bytes(UT);
    // ONE WARP PER OUTPUT ROW, so the rows-per-expert count is the row count
    // itself: nw warps cover n_act_max experts x n_ff_exp rows.  `npb` walks NPB
    // ADJACENT rows per warp instead of one, which divides the number of
    // concurrent weight-read streams by NPB; see moe_npb() for why that is the
    // thing worth spending warps on.
    const int nrb = (n_ff_exp + npb - 1) / npb;
    const int nw = n_act_max * nrb;
    q.parallel_for(nd_range<1>(moe_grid((size_t)nw * 32), kMoEWG),
                   [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int lid = (int)it.get_local_id(0);
        const int sg = lid / 32;
        const int lane = lid % 32;
        const sub_group sgg = it.get_sub_group();
        // ONE WARP per (expert, row-block), indexed by the *global warp id*.
        // Two wrong indices are natural here and both are silent: the
        // work-group index addresses 3 of the 560 items and leaves the rest
        // unrun, and the work-item index varies `n` across the lanes of one
        // warp - so only the rows that happen to land on lane 0 are ever
        // written.  (Both were written, and both produced "almost right".)
        const int g = (int)it.get_group(0) * kMoENW + sg;
        if (g >= nw) {
            return;
        }
        const int ia = g / nrb;
        const int nb = g % nrb;
        if (ia >= n_active[0]) {
            return;
        }
        const int e = active[ia];
        const int slot = slot_map[e];
        if (slot < 0) {
            return; // host-resident expert: the host computes it
        }
        for (int rr = 0; rr < npb; rr++) {
        const int n = nb * npb + rr;
        if (n >= n_ff_exp) {
            break;
        }
        const char * gr = gbank + (size_t)slot * slot_gb + (size_t)n * n_sb * sgb;
        const char * ur = ubank + (size_t)slot * slot_ub + (size_t)n * n_sb * sub;
        // REUSE THE DEQUANTIZED ROW ACROSS THE EXPERT'S PAIRS.
        //
        // This used to call moe_dot inside the pair loop, so this row's gate and
        // up halves were read and dequantized once per routed pair.  A 512-token
        // prefill routes 4096 pairs over 256 experts, so every row was
        // dequantized ~14 times over - and dequantizing Q4_K is a dozen integer
        // ops per value against one FMA of useful work.  That is why a 3B-active
        // model prefilled at 6% of the rate the 27B dense manages, and why
        // PF_ABL_NOMOE made the same prefill 17.5x faster: the dequant ALU, not
        // the arithmetic and not the bandwidth, was the whole cost.
        //
        // The loop order is now superblock-outer, pair-inner, and the pair
        // accumulators live in registers: 8 floats of dequantized weight per
        // lane per superblock, reused across kMoEPB pairs.  That keeps the grid
        // (eight warps per work-group) and therefore the occupancy exactly as
        // they were - staging the row in SLM instead bought the same reuse and
        // threw the occupancy away, and measured no faster.
        for (int i0 = off[e]; i0 < off[e + 1]; i0 += kMoEPB) {
            const int np = (off[e + 1] - i0 < kMoEPB) ? (off[e + 1] - i0) : kMoEPB;
            // Declared inside the tile loop so the row-block walk below (PF_MOE_NPB)
            // REUSES one accumulator set across NPB adjacent output columns: holding
            // NPB rows of ag[]/au[] at once is NPB*kMoEPB*2 floats and blows the
            // register budget.
            float ag[kMoEPB], au[kMoEPB];
#pragma unroll
            for (int j = 0; j < kMoEPB; j++) {
                ag[j] = 0.f;
                au[j] = 0.f;
            }
            for (int sb = 0; sb < n_sb; sb++) {
                float w[8], u[8];
                dequant_sb_lane_typed<GT>(gr + (size_t)sb * sgb, lane, w, dq2);
                dequant_sb_lane_typed<UT>(ur + (size_t)sb * sub, lane, u, dq2);
#pragma unroll
                for (int j = 0; j < kMoEPB; j++) {
                    if (j < np) {
                        const float * xr = x + (size_t)(pair[i0 + j] / top_k) * x_stride;
#pragma unroll
                        for (int b = 0; b < 8; b++) {
                            const float xv = noact ? (float)(sb * 8 + b + lane)
                                                    : xr[(size_t)sb * 256 + 32 * b + lane];
                            ag[j] += w[b] * xv;
                            au[j] += u[b] * xv;
                        }
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < kMoEPB; j++) {
                if (j < np) {
                    // every lane must enter the collective
                    const float gv = sg_sum(ag[j], sgg);
                    const float uv = sg_sum(au[j], sgg);
                    if (lane == 0) {
                        h[(size_t)pair[i0 + j] * h_stride + n] = silu_f(gv) * uv;
                    }
                }
            }
        } // i0
        } // rr: adjacent output columns, walked sequentially so ag/au are reused
    });
}

void moe_gate_up_launch(queue & q, const void * gbank, const void * ubank, const int32_t * slot_map, size_t slot_gb,
                        size_t slot_ub, const int32_t * off, const int32_t * pair, const int32_t * active,
                        const int32_t * n_active, int n_act_max, const float * x, int x_stride, float * h, int h_stride,
                        int n_embd, int n_ff_exp, int top_k, uint32_t gtype, uint32_t utype) {
#define SI_MOE_GU(GT, UT)                                                                                                  \
    moe_gate_up_impl<GT, UT>(q, (const char *)gbank, (const char *)ubank, slot_map, slot_gb, slot_ub, off, pair, active, \
                             n_active, n_act_max, x, x_stride, h, h_stride, n_embd, n_ff_exp, top_k, moe_dq2(), moe_noact(), moe_npb())
    if (gtype == 12 && utype == 12) {
        SI_MOE_GU(12, 12);
    } else if (gtype == 13 && utype == 13) {
        SI_MOE_GU(13, 13);
    } else if (gtype == 14 && utype == 14) {
        SI_MOE_GU(14, 14);
    } else if (gtype == 8 && utype == 8) {
        SI_MOE_GU(8, 8);
    } else if (gtype == 0 && utype == 0) {
        SI_MOE_GU(0, 0);
    } else if (gtype == 12 && utype == 0) {
        SI_MOE_GU(12, 0);
    } else if (gtype == 0 && utype == 12) {
        SI_MOE_GU(0, 12);
    } else if (gtype == 13 && utype == 0) {
        SI_MOE_GU(13, 0);
    } else if (gtype == 0 && utype == 13) {
        SI_MOE_GU(0, 13);
    } else if (gtype == 8 && utype == 12) {
        SI_MOE_GU(8, 12);
    } else if (gtype == 12 && utype == 8) {
        SI_MOE_GU(12, 8);
    } else {
        throw std::runtime_error("moe: unsupported gate/up expert types " + std::to_string(gtype) + "/" +
                                 std::to_string(utype));
    }
#undef SI_MOE_GU
}

// ---------------------------------------------------------------------------
// down: per (expert, row block) over the expert's tokens, each writing its own
// token's partial - no atomics, because one row block of one expert is owned by
// exactly one work-group.
template <uint32_t DT>
static void moe_down_impl(queue & q, const char * dbank, const int32_t * slot_map, size_t slot_db, const int32_t * off,
                          const int32_t * pair, const int32_t * active, const int32_t * n_active, int n_act_max,
                          const float * h, int h_stride, float * acc, int acc_stride, int n_embd, int n_ff_exp,   \
                          bool dq2, bool noact, int npb) {
    // n_ff_exp is NOT always a multiple of 256: qwen4exp's expert down has
    // K = 640 = 2.5 superblocks.  Integer-dividing here used 2, so every output
    // row after the first was read 384 bytes into the plane instead of 480 -
    // the whole down projection was the wrong weights - and the last partial
    // superblock read past the row.  Round up, use the real row stride, and
    // mask the blocks beyond n_ff_exp.
    const int n_sb = (n_ff_exp + 255) / 256;
    const int sdb = superblock_bytes(DT);
    const size_t row_bytes = quant_row_bytes(DT, n_ff_exp);
    const int nrb = (n_embd + npb - 1) / npb; // one warp per NPB adjacent output rows
    const int nw = n_act_max * nrb;
    q.parallel_for(nd_range<1>(moe_grid((size_t)nw * 32), kMoEWG),
                   [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int lid = (int)it.get_local_id(0);
        const int sg = lid / 32;
        const int lane = lid % 32;
        const sub_group sgg = it.get_sub_group();
        const int g = (int)it.get_group(0) * kMoENW + sg; // global warp id
        if (g >= nw) {
            return;
        }
        const int ia = g / nrb;
        const int nb = g % nrb;
        if (ia >= n_active[0]) {
            return;
        }
        const int e = active[ia];
        const int slot = slot_map[e];
        if (slot < 0) {
            return;
        }
        for (int rr = 0; rr < npb; rr++) {
        const int n = nb * npb + rr;
        if (n >= n_embd) {
            break;
        }
        const char * dr = dbank + (size_t)slot * slot_db + (size_t)n * row_bytes;
        // superblock-outer, pair-inner with per-pair accumulators, for the same
        // reason as moe_gate_up_impl: the row used to be re-read and
        // re-dequantized once per routed pair.
        float a[kMoEPB];
        for (int i0 = off[e]; i0 < off[e + 1]; i0 += kMoEPB) {
            const int np = (off[e + 1] - i0 < kMoEPB) ? (off[e + 1] - i0) : kMoEPB;
#pragma unroll
            for (int j = 0; j < kMoEPB; j++) {
                a[j] = 0.f;
            }
            for (int sb = 0; sb < n_sb; sb++) {
                float w[8];
                dequant_sb_lane_typed<DT>(dr + (size_t)sb * sdb, lane, w, dq2);
#pragma unroll
                for (int b = 0; b < 8; b++) {
                    if (sb * 256 + 32 * b >= n_ff_exp) {
                        w[b] = 0.f; // past a partial last superblock
                    }
                }
#pragma unroll
                for (int j = 0; j < kMoEPB; j++) {
                    if (j < np) {
                        const float * hr = h + (size_t)pair[i0 + j] * h_stride;
#pragma unroll
                        for (int b = 0; b < 8; b++) {
                            const int kk = sb * 256 + 32 * b + lane;
                            if (kk >= n_ff_exp) {
                                continue;
                            }
                            const float hv = noact ? (float)(sb * 8 + b + lane) : hr[(size_t)kk];
                            a[j] += w[b] * hv;
                        }
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < kMoEPB; j++) {
                if (j < np) {
                    const float d = sg_sum(a[j], sgg);
                    if (lane == 0) {
                        acc[(size_t)pair[i0 + j] * acc_stride + n] = d;
                    }
                }
            }
        } // i0
        } // rr
    });
}

void moe_down_launch(queue & q, const void * dbank, const int32_t * slot_map, size_t slot_db, const int32_t * off,
                     const int32_t * pair, const int32_t * active, const int32_t * n_active, int n_act_max,
                     const float * h, int h_stride, float * acc, int acc_stride, int n_embd, int n_ff_exp,
                     uint32_t dtype) {
#define SI_MOE_D(DT)                                                                                             \
    moe_down_impl<DT>(q, (const char *)dbank, slot_map, slot_db, off, pair, active, n_active, n_act_max, h, h_stride, \
                      acc, acc_stride, n_embd, n_ff_exp, moe_dq2(), moe_noact(), moe_npb())
    switch (dtype) {
    case 12: SI_MOE_D(12); break;
    case 13: SI_MOE_D(13); break;
    case 14: SI_MOE_D(14); break;
    case 8: SI_MOE_D(8); break;
    case 7: SI_MOE_D(7); break;
    case 0: SI_MOE_D(0); break;
    default: throw std::runtime_error("moe: unsupported down expert type " + std::to_string(dtype));
    }
#undef SI_MOE_D
}

// out[t] += sum_j wts[t*top_k + j] * acc[(t*top_k + j) * acc_stride]
//
// 1-D grid, not 2-D: a 2-D nd_range with a 256-wide local dim would need
// ceil(n_embd/256) >= 256 blocks per row, and SYCL rejects a local range larger
// than the global one ("Non-uniform work-groups are not supported").
void moe_reduce_launch(queue & q, const float * acc, const float * wts, int top_k, int n_tok, int n_embd,
                       int acc_stride, float * out, int out_stride) {
    const int n_eb = (n_embd + 255) / 256;
    q.parallel_for(nd_range<1>((size_t)n_tok * n_eb * 256, 256), [=](nd_item<1> it) {
        const int g = (int)it.get_group(0);
        const int t = g / n_eb;
        const int e = (g - t * n_eb) * 256 + (int)it.get_local_id(0);
        if (e >= n_embd) {
            return;
        }
        float a = 0.f;
        for (int j = 0; j < top_k; j++) {
            const size_t pr = (size_t)t * top_k + j;
            a += wts[pr] * acc[pr * (size_t)acc_stride + e];
        }
        out[(size_t)t * out_stride + e] += a;
    });
}

void moe_reduce_csr_launch(queue & q, const float * acc, const float * wts, const int32_t * pos, int top_k, int n_tok,
                           int n_embd, int acc_stride, float * out, int out_stride) {
    const int n_eb = (n_embd + 255) / 256;
    q.parallel_for(nd_range<1>((size_t)n_tok * n_eb * 256, 256), [=](nd_item<1> it) {
        const int g = (int)it.get_group(0);
        const int t = g / n_eb;
        const int e = (g - t * n_eb) * 256 + (int)it.get_local_id(0);
        if (e >= n_embd) {
            return;
        }
        float a = 0.f;
        for (int j = 0; j < top_k; j++) {
            const size_t pr = (size_t)t * top_k + j;
            a += wts[pr] * acc[(size_t)pos[pr] * (size_t)acc_stride + e];
        }
        out[(size_t)t * out_stride + e] += a;
    });
}

// MoE epilogue: out[t] += sigmoid(gate . x[t]) * shared[t].
//
// One work-group per token computes the gate as a full reduction over n_embd
// (the shared expert's gate is a single scalar per token, so a 256-lane dot is
// far cheaper than staging the shared row to do it twice).  A null `gate` skips
// the factor, which is what a model without a shared-expert gate needs.
template <uint32_t GT>
static void moe_gate_add_impl(queue & q, const char * gate, const float * x, int x_stride, int n_tok, int n_embd,
                              const float * shared, int sh_stride, float * out, int out_stride, bool have_gate) {
    // ONE SUB-GROUP PER TOKEN, and every lane strides by the sub-group width.
    // This used to be a 256-wide work-group whose gate reduction was
    // `reduce_over_group(sub_group)` - so each of the eight sub-groups reduced
    // only its own eighth of the dot product (the loop strided by 256, the
    // collective spanned 32) and every lane then added the shared expert scaled
    // by its *partial* gate.  A partial dot product of a large negative value
    // is much less negative, so sigmoid() of it is much larger: the shared
    // expert came out ~20x too strong, which is invisible in every folded
    // fingerprint of it and shows up only as the layer's residual being wrong.
    // The width now matches the collective, so no SLM or barrier is needed.
    q.parallel_for(nd_range<1>((size_t)n_tok * 32, 32), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int lane = (int)it.get_local_id(0);
        const int t = (int)it.get_group(0);
        const float * xr = x + (size_t)t * x_stride;
        float g = 1.f;
        if (have_gate) {
            float a = 0.f;
            for (int k = lane; k < n_embd; k += 32) {
                a += moe_rd<GT>(gate, k) * xr[k];
            }
            a = reduce_over_group(it.get_sub_group(), a, plus<float>());
            g = sigmoid_f(a);
        }
        for (int e = lane; e < n_embd; e += 32) {
            out[(size_t)t * out_stride + e] += g * shared[(size_t)t * sh_stride + e];
        }
    });
}

void moe_gate_add_launch(queue & q, const void * gate, uint32_t gtype, const float * x, int x_stride, int n_tok,
                         int n_embd, const float * shared, int sh_stride, float * out, int out_stride) {
    if (!gate) {
        moe_gate_add_impl<GGML_TYPE_F32>(q, nullptr, x, x_stride, n_tok, n_embd, shared, sh_stride, out, out_stride,
                                         false);
    } else if (gtype == GGML_TYPE_BF16) {
        moe_gate_add_impl<GGML_TYPE_BF16>(q, (const char *)gate, x, x_stride, n_tok, n_embd, shared, sh_stride, out,
                                          out_stride, true);
    } else {
        moe_gate_add_impl<GGML_TYPE_F32>(q, (const char *)gate, x, x_stride, n_tok, n_embd, shared, sh_stride, out,
                                         out_stride, true);
    }
}

// ---------------------------------------------------------------------------
// Routed-expert u4 bank (PF_MOE_U4BANK): the two small kernels the per-expert
// oneDNN gate/up GEMM needs around it.  The bank itself is packed at load; the
// GEMM consumes it, so these only move rows in and out of the GEMM's [M][K]
// activation tile and its [M][N] output tiles.
// ---------------------------------------------------------------------------

// tile[j][k] = x[(pair[p0+j] / top_k)][k] for j < n, zero for the padded rows.
// The padding is what makes the GEMM's M a fixed 32/64 bucket: a padded row must
// quantize to a benign scale (zero -> the quantizer keeps 1.0), not to whatever
// the scratch held before.
void moe_u4_gather_launch(queue & q, const float * x, const int32_t * pair, int p0, int n, int tile_m, int top_k,
                          int K, float * tile) {
    q.parallel_for(range<2>((size_t)tile_m, (size_t)K), [=](id<2> i) {
        const int j = (int)i[0];
        const int k = (int)i[1];
        float v = 0.f;
        if (j < n) {
            const int row = pair[p0 + j] / top_k;
            v = x[(size_t)row * K + k];
        }
        tile[(size_t)j * K + k] = v;
    });
}

// Gather an already-quantized activation tile: the same rows the per-tile
// quantizer would have produced, without the f32 gather or the max pass.  One
// 16-byte chunk per work-item, and the first K/32 chunks also copy the scale and
// group sum.  Padded rows are zero (scale 0 would be a *different* quantizer
// result, but the row is not read by the GEMM's real outputs; 1.0 matches the
// quantizer's zero-row convention).
void moe_u4_gather_i8_launch(queue & q, const int8_t * xq, const uint16_t * asa, const float * xs,
                             const int32_t * pair, int p0, int n, int tile_m, int top_k, int K, int8_t * tq,
                             int8_t * txe, int8_t * txo, uint16_t * tasa, float * txs, uint16_t * tcorr) {
    const int ng = K / 32;
    q.parallel_for(range<2>((size_t)tile_m, (size_t)(K / 16)), [=](id<2> i) {
        const int j = (int)i[0];
        const int c = (int)i[1];
        uint4 v(0, 0, 0, 0);
        int row = -1;
        if (j < n) {
            row = pair ? pair[p0 + j] / top_k : p0 + j;
            v = *reinterpret_cast<const uint4 *>(xq + (size_t)row * K + (size_t)c * 16);
        }
        *reinterpret_cast<uint4 *>(tq + (size_t)j * K + (size_t)c * 16) = v;
        if (txe) {
            // even/odd k planes for the native batched GEMM: a 16-byte chunk is
            // 16 consecutive values, so its 8 even and 8 odd values land at
            // c*8 in the two half-width planes.
            const int8_t * src = tq + (size_t)j * K + (size_t)c * 16;
            int8_t * e = txe + (size_t)j * (K / 2) + (size_t)c * 8;
            int8_t * o = txo + (size_t)j * (K / 2) + (size_t)c * 8;
            for (int x = 0; x < 8; x++) {
                e[x] = src[2 * x];
                o[x] = src[2 * x + 1];
            }
        }
        if (c < ng) {
            const uint16_t sa_b = row >= 0 ? asa[(size_t)row * ng + c] : (uint16_t)0x3C00;
            const float gs = row >= 0 ? xs[(size_t)row * ng + c] : 0.f;
            tasa[(size_t)j * ng + c] = sa_b;
            txs[(size_t)j * ng + c] = gs;
            // The u4 correction matmul's A operand (xs * sa, f16).  quantize()
            // fills this plane; a gather that skips it leaves the last call's
            // values there and corrupts every gate/up result.
            sycl::half h(gs * moe_h2f(sa_b));
            uint16_t hb;
            __builtin_memcpy((void *)&hb, &h, sizeof(h));
            tcorr[(size_t)j * ng + c] = hb;
        }
    });
}

// h[pair[p0+j]][i] = silu(g[j][i]) * u[j][i].  The output row is the pair id,
// not the tile row: the down kernel and the weighted reduction are indexed by
// pair, and an expert's pairs are contiguous in the CSR but not in pair-id order.
void moe_u4_scatter_silu_launch(queue & q, const float * g, const float * u, const int32_t * pair, int p0, int n,
                                int N, float * h, int h_stride) {
    static const bool abl_scg = [] {
        const char * e = si::env::str("PF_MOE_ABL_SCG");
        return e && atoi(e) != 0;
    }();
    if (abl_scg)
        return;

    q.parallel_for(range<2>((size_t)n, (size_t)N), [=](id<2> i) {
        const int j = (int)i[0];
        const int c = (int)i[1];
        const size_t src = (size_t)j * N + c;
        const size_t dst = (size_t)pair[p0 + j] * h_stride + c;
        h[dst] = silu_f(g[src]) * u[src];
    });
}

void moe_scatter_launch(queue & q, const float * src, const int32_t * pair, int p0, int n, int N, float * dst,
                        int dst_stride) {
    static const bool abl_scd = [] {
        const char * e = si::env::str("PF_MOE_ABL_SCD");
        return e && atoi(e) != 0;
    }();
    if (abl_scd)
        return;

    q.parallel_for(range<2>((size_t)n, (size_t)N), [=](id<2> i) {
        const int j = (int)i[0];
        const int c = (int)i[1];
        dst[(size_t)pair[p0 + j] * dst_stride + c] = src[(size_t)j * N + c];
    });
}

// h[i] = silu(g[i]) * u[i] over n elements: the decode form, where both rows
// already sit at their pair id (one M=1 GEMV per pair wrote them).
void moe_u4_silu_mul_launch(queue & q, float * g, const float * u, int n) {
    q.parallel_for(range<1>((size_t)n), [=](id<1> i) { g[i] = silu_f(g[i]) * u[i]; });
}

void moe_u4_silu_pair_launch(queue & q, const float * g, const float * u, int n_pairs, int N, float * h) {
    q.parallel_for(range<1>((size_t)n_pairs * (size_t)N), [=](id<1> i) {
        const size_t x = i[0];
        h[x] = silu_f(g[x]) * u[x];
    });
}

// ---------------------------------------------------------------------------
// Grouped u4 expert GEMV: the decode form of the bank, ONE launch per
// projection.
//
// The host loop this replaces walked `n_pairs` on the host (d_moe_ids read
// back, one w4_gemv_launch per (pair, projection), then a silu-mul kernel), so
// every layer needed a host->device wait and the decode command graph had to
// be skipped.  Here the router ids stay on the device: the expert of pair ia is
// ids[ia] and its plane is addressed inside the kernel, so the whole layer is
// kernels on the queue and records into the decode graph again.
//
// Each (pair, row block) is one work-group, shaped exactly like
// w4_gemv_launch's ONE tensor: RB sub-groups, the sub-group owning output row
// n, and the (g,n) scale/off planes staged into SLM g-major / row-inner so the
// strided reads coalesce.
//
// The activation is the contiguous per-32-group int8 tile `moe_actq_launch`
// writes (no even/odd split).  One 16-byte weight load at group g is the 32
// nibbles of k = 32g..32g+31 in order (byte j: low = 2j, high = 2j+1), so a
// dp4a operand needs the *four consecutive* nibbles of two bytes expanded into
// four bytes; the two helpers below do that in registers.  That is eight dp4a
// per group, the same count as the dense u4 GEMV.
// ---------------------------------------------------------------------------
namespace {

// The low/high nibbles of two weight bytes -> four consecutive uint8 lanes.
// w's bytes 0,1 carry elements 0..3 (low,high,low,high).
inline uint32_t u4_lo_pair(uint32_t w) {
    return (w & 0x0000000Fu) | ((w & 0x000000F0u) << 4) | ((w & 0x00000F00u) << 8) | ((w & 0x0000F000u) << 12);
}
// w's bytes 2,3 carry elements 4..7, starting at bit 16.
inline uint32_t u4_hi_pair(uint32_t w) {
    return ((w >> 16) & 0x0000000Fu) | ((w >> 12) & 0x00000F00u) | ((w >> 8) & 0x000F0000u) |
           ((w >> 4) & 0x0F000000u);
}
// dp4a of four consecutive int8 activations against an expanded nibble word.
inline int32_t u4_dp(const int8_t * x, uint32_t w, int32_t c) {
    uint32_t a;
    __builtin_memcpy((void *)&a, x, sizeof(a));
    return dp4a_s8u8((int32_t)a, w, c);
}

int moe_u4_rb_for(const queue & q) {
    static const int env_rb = si::env::i32("PF_W4_RB", 0);
    const int want = env_rb > 0 ? env_rb : si::dev::for_queue(q).shape.w4_rows_per_wg;
    return want <= 8 ? 8 : (want <= 16 ? 16 : 32);
}

} // namespace

template <int RB>
static void moe_u4_gemv_impl(queue & q, const uint8_t * bank, size_t plane_bytes, const int32_t * slot_map,
                             const int32_t * ids, int n_pairs, const int8_t * xq, const uint16_t * asa,
                             const float * xs, int K, int N, float * out, int out_stride, const float * gate,
                             int top_k) {
    const int ng = K / 32;
    // PF_W4_GROUP: the bank's scale/off planes are per group; a widened group
    // is F consecutive 32-value groups sharing one (step, off).
    const int gs = si::w4_group_shift();
    const int ngw = ng >> gs;
    const size_t vb = (size_t)N * (size_t)(K / 2);
    const int nwg = (N + RB - 1) / RB;
    constexpr int TX = RB * 32;
    q.submit([&](handler & h) {
        // step + off for this pair's RB rows, f16 (the dense u4 GEMV's tile)
        local_accessor<uint16_t, 1> meta((size_t)2 * RB * ng, h);
        h.parallel_for(nd_range<1>((size_t)n_pairs * (size_t)nwg * TX, TX), [=](nd_item<1> it)
                           [[sycl::reqd_sub_group_size(32)]] {
            const int lid = (int)it.get_local_id(0);
            const int sg = lid / 32;
            const int lane = lid % 32;
            const int gid = (int)it.get_group(0);
            const int p = gid / nwg;
            const int n0 = (gid - p * nwg) * RB;
            const int e = ids[p];
            const int slot = slot_map ? slot_map[e] : e;
            const uint8_t * plane = bank + (size_t)slot * plane_bytes;
            const uint16_t * scale = reinterpret_cast<const uint16_t *>(plane + vb);
            const uint16_t * off = scale + (size_t)ngw * (size_t)N;
            // g-major / row-inner: consecutive lanes read consecutive n
            for (int i = lid; i < RB * ng; i += TX) {
                const int g = i / RB;
                const int r = i - g * RB;
                const int n = n0 + r;
                const bool ok = n < N;
                const size_t src = (size_t)(g >> gs) * (size_t)N + n;
                meta[i] = ok ? scale[src] : (uint16_t)0;
                meta[(size_t)RB * ng + i] = ok ? off[src] : (uint16_t)0;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const int n = n0 + sg;
            if (n >= N) {
                return;
            }
            const uint8_t * wrow = plane + (size_t)n * (size_t)(K / 2);
            const int arow = top_k > 0 ? p / top_k : p;
            const int8_t * xrow = xq + (size_t)arow * (size_t)K;
            const uint16_t * asr = asa + (size_t)arow * (size_t)ng;
            const float * xsr = xs + (size_t)arow * (size_t)ng;
            float acc = 0.f;
            for (int g = lane; g < ng; g += 32) {
                // 16 bytes = the group's 32 nibbles, elements 32g..32g+31
                const uint4 wv = *reinterpret_cast<const uint4 *>(wrow + (size_t)g * 16);
                const int8_t * xg = xrow + (size_t)g * 32;
                int32_t qd = 0;
                qd = u4_dp(xg + 0, u4_lo_pair(wv.x()), qd);
                qd = u4_dp(xg + 4, u4_hi_pair(wv.x()), qd);
                qd = u4_dp(xg + 8, u4_lo_pair(wv.y()), qd);
                qd = u4_dp(xg + 12, u4_hi_pair(wv.y()), qd);
                qd = u4_dp(xg + 16, u4_lo_pair(wv.z()), qd);
                qd = u4_dp(xg + 20, u4_hi_pair(wv.z()), qd);
                qd = u4_dp(xg + 24, u4_lo_pair(wv.w()), qd);
                qd = u4_dp(xg + 28, u4_hi_pair(wv.w()), qd);
                const float sa = moe_h2f(asr[g]);
                const size_t mo = (size_t)g * RB + sg;
                acc += sa * (moe_h2f(meta[mo]) * (float)qd + moe_h2f(meta[(size_t)RB * ng + mo]) * xsr[g]);
            }
            const sub_group sgg = it.get_sub_group();
            const float tot = reduce_over_group(sgg, acc, plus<float>());
            if (lane == 0) {
                const size_t oi = (size_t)p * (size_t)out_stride + n;
                out[oi] = gate ? silu_f(gate[oi]) * tot : tot;
            }
        });
    });
}

void moe_u4_gemv_launch(queue & q, const uint8_t * bank, size_t plane_bytes, const int32_t * slot_map,
                        const int32_t * ids, int n_pairs, const int8_t * xq, const uint16_t * asa, const float * xs,
                        int K, int N, float * out, int out_stride, const float * gate, int top_k) {
    const int ng = K / 32;
    const int rb = moe_u4_rb_for(q);
    const auto fits = [&](int n) {
        return (size_t)2 * (size_t)n * (size_t)ng * sizeof(uint16_t) <=
               (size_t)si::dev::for_queue(q).slm.budget_bytes;
    };
    if (rb == 8 && fits(8)) {
        moe_u4_gemv_impl<8>(q, bank, plane_bytes, slot_map, ids, n_pairs, xq, asa, xs, K, N, out, out_stride, gate,
                            top_k);
    } else if (rb == 32 && fits(32)) {
        moe_u4_gemv_impl<32>(q, bank, plane_bytes, slot_map, ids, n_pairs, xq, asa, xs, K, N, out, out_stride, gate,
                             top_k);
    } else if (fits(16)) {
        moe_u4_gemv_impl<16>(q, bank, plane_bytes, slot_map, ids, n_pairs, xq, asa, xs, K, N, out, out_stride, gate,
                             top_k);
    } else {
        moe_u4_gemv_impl<8>(q, bank, plane_bytes, slot_map, ids, n_pairs, xq, asa, xs, K, N, out, out_stride, gate,
                            top_k);
    }
}

// Down int8 bank decode GEMV (see the declaration in kernels.h).
void moe_i8_gemv_down_launch(queue & q, const uint8_t * bank, size_t plane_bytes, const int32_t * ids, int n_pairs,
                             const int8_t * xq, const uint16_t * asa, const float * xs, int K, int N, float * out,
                             int out_stride) {
    const int ng = K / 32;
    const size_t vb = (size_t)N * (size_t)K;
    const int RB = 8; // warps per workgroup (8 output rows)
    q.submit([&](handler & h) {
        h.parallel_for(nd_range<1>((size_t)n_pairs * ((N + RB - 1) / RB) * (RB * 32), RB * 32),
                       [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int lid = (int)it.get_local_id(0);
                           const int sg = lid / 32;
                           const int lane = lid % 32;
                           const int gid = (int)it.get_group(0);
                           const int nblk = (N + RB - 1) / RB;
                           const int p = gid / nblk;
                           const int n = (gid - p * nblk) * RB + sg;
                           if (n >= N) {
                               return;
                           }
                           const uint8_t * plane = bank + (size_t)ids[p] * plane_bytes;
                           const int8_t * wrow = (const int8_t *)plane + (size_t)n * K;
                           const uint16_t * step = (const uint16_t *)(plane + vb) + (size_t)n;
                           const uint16_t * offp = step + (size_t)ng * N;
                           const int8_t * xrow = xq + (size_t)p * K;
                           const uint16_t * asr = asa + (size_t)p * ng;
                           float acc = 0.f;
                           for (int g = lane; g < ng; g += 32) {
                               const uint4 w0 = *reinterpret_cast<const uint4 *>(wrow + (size_t)g * 32);
                               const uint4 w1 = *reinterpret_cast<const uint4 *>(wrow + (size_t)g * 32 + 16);
                               const uint4 x0 = *reinterpret_cast<const uint4 *>(xrow + (size_t)g * 32);
                               const uint4 x1 = *reinterpret_cast<const uint4 *>(xrow + (size_t)g * 32 + 16);
                               int32_t qd = 0;
                               qd = dp4a_s8s8((int32_t)x0.x(), w0.x(), qd);
                               qd = dp4a_s8s8((int32_t)x0.y(), w0.y(), qd);
                               qd = dp4a_s8s8((int32_t)x0.z(), w0.z(), qd);
                               qd = dp4a_s8s8((int32_t)x0.w(), w0.w(), qd);
                               qd = dp4a_s8s8((int32_t)x1.x(), w1.x(), qd);
                               qd = dp4a_s8s8((int32_t)x1.y(), w1.y(), qd);
                               qd = dp4a_s8s8((int32_t)x1.z(), w1.z(), qd);
                               qd = dp4a_s8s8((int32_t)x1.w(), w1.w(), qd);
                               const float sa = moe_h2f(asr[g]);
                               const float st = moe_h2f(step[(size_t)g * N]);
                               const float ofv = moe_h2f(offp[(size_t)g * N]);
                               acc += sa * (st * (float)qd + ofv * xs[(size_t)p * ng + g]);
                           }
                           const float tot = reduce_over_group(it.get_sub_group(), acc, plus<float>());
                           if (lane == 0) {
                               out[(size_t)p * out_stride + n] = tot;
                           }
                       });
    });
}


void moe_csr_gather_i8_launch(queue & q, const int8_t * xq, const uint16_t * asa, const float * xs,
                              const int32_t * pair, int n_pairs, int top_k, int K, int8_t * tq, uint16_t * tasa,
                              float * txs, uint16_t * tcorr) {
    const int ng = K / 32;
    q.parallel_for(range<2>((size_t)n_pairs, (size_t)(K / 16)), [=](id<2> i) {
        const int p = (int)i[0];
        const int c = (int)i[1];
        const int row = top_k > 0 ? pair[p] / top_k : pair[p];
        const uint4 v = *reinterpret_cast<const uint4 *>(xq + (size_t)row * K + (size_t)c * 16);
        *reinterpret_cast<uint4 *>(tq + (size_t)p * K + (size_t)c * 16) = v;
        if (c < ng) {
            const uint16_t sa = asa[(size_t)row * ng + c];
            const float gs = xs[(size_t)row * ng + c];
            tasa[(size_t)p * ng + c] = sa;
            txs[(size_t)p * ng + c] = gs;
            sycl::half h(gs * moe_h2f(sa));
            uint16_t hb;
            __builtin_memcpy((void *)&hb, &h, sizeof(hb));
            tcorr[(size_t)p * ng + c] = hb;
        }
    });
}

void moe_csr_corr_launch(queue & q, const uint16_t * asa, const float * xs, uint16_t * corr, int n_rows, int ng) {
    q.parallel_for(range<1>((size_t)n_rows * (size_t)ng), [=](id<1> i) {
        sycl::half h(xs[i] * moe_h2f(asa[i]));
        uint16_t hb;
        __builtin_memcpy((void *)&hb, &h, sizeof(hb));
        corr[i] = hb;
    });
}


void moe_u4_silu_fused_launch(queue & q, const float * gu, int n_rows, int N, float * h, int h_stride) {
    const int n2 = 2 * N;
    q.parallel_for(range<1>((size_t)n_rows * (size_t)N), [=](id<1> i) {
        const size_t x = i[0];
        const int r = (int)(x / (size_t)N);
        const int c = (int)(x - (size_t)r * (size_t)N);
        h[(size_t)r * h_stride + c] = silu_f(gu[(size_t)r * n2 + c]) * gu[(size_t)r * n2 + N + c];
    });
}

void moe_u4_silu_fused_scatter_launch(queue & q, const float * gu, const int32_t * pair, int n_pairs, int N,
                                      float * h, int h_stride) {
    const int n2 = 2 * N;
    q.parallel_for(range<1>((size_t)n_pairs * (size_t)N), [=](id<1> i) {
        const size_t x = i[0];
        const int r = (int)(x / (size_t)N);
        const int c = (int)(x - (size_t)r * (size_t)N);
        const int pid = pair ? pair[r] : r;
        h[(size_t)pid * h_stride + c] = silu_f(gu[(size_t)r * n2 + c]) * gu[(size_t)r * n2 + N + c];
    });
}


void moe_down_corr_add_launch(queue & q, const uint8_t * bank, size_t plane_bytes, int K, int N, const int32_t * off,
                              const int32_t * active, const int32_t * n_active, int n_act_max, const uint16_t * asa,
                              const float * xs, float * acc, int acc_stride) {
    const int ng = K / 32;
    const size_t vb = (size_t)N * (size_t)K;      // int8 values
    const size_t sb = (size_t)ng * (size_t)N * 2; // f16 step plane
    const int nblk = (N + 255) / 256;
    q.parallel_for(nd_range<1>((size_t)n_act_max * (size_t)nblk * 256, 256), [=](nd_item<1> it) {
        const int gid = (int)it.get_group(0);
        const int ia = gid / nblk;
        if (ia >= n_active[0]) {
            return;
        }
        const int nb = gid - ia * nblk;
        const int e = active[ia];
        const int p0 = off[e];
        const int pe = off[e + 1] - p0;
        if (pe <= 0) {
            return;
        }
        const uint16_t * op = reinterpret_cast<const uint16_t *>(bank + (size_t)e * plane_bytes + vb + sb);
        const int tid = (int)it.get_local_id(0);
        const int n = nb * 256 + tid;
        if (n >= N) {
            return;
        }
        for (int p = p0; p < p0 + pe; p++) {
            float s = 0.f;
            const float * xr = xs + (size_t)p * (size_t)ng;
            const uint16_t * ar = asa + (size_t)p * (size_t)ng;
            for (int g = 0; g < ng; g++) {
                sycl::half h, ha;
                const uint16_t hb = op[(size_t)g * (size_t)N + n];
                __builtin_memcpy((void *)&h, &hb, sizeof(h));
                __builtin_memcpy((void *)&ha, &ar[g], sizeof(ha));
                // the correction's A operand is the quantizer's group sum times
                // the group scale (corrsrc = xs*asa), exactly as the oneDNN
                // correction matmul consumes it
                const float cg = (float)(sycl::half)((float)ha * xr[g]);
                s = sycl::fma((float)h, cg, s);
            }
            acc[(size_t)p * (size_t)acc_stride + n] += s;
        }
    });
}

} // namespace si
