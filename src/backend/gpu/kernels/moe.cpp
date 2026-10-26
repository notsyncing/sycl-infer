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
#include "device/device_profile.h"
#include "kernel_utils.h"
#include "gguf.h" // GGML_TYPE_* (the kernel headers only pull the raw type ids)

#include <cmath>
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
            for (int s = 1; s < NSG; s++) {
                mx = sycl::fmax(mx, red[s]);
            }
            // exp in parallel, kept in SLM (and written back for compatibility)
            for (int e = tid; e < n_expert; e += WG) {
                pl[e] = sycl::exp(p[e] - mx);
            }
            it.barrier();
            // the reference's sum: sequential, ascending index, on one lane.
            // Batched 8 loads then 8 dependent adds: the adds keep the
            // reference's order exactly (so 1/sum is bit-identical) while the
            // loads overlap instead of stalling the chain every iteration.
            if (tid == 0) {
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
            const float inv = 1.f / red[NSG];
            for (int e = tid; e < n_expert; e += WG) {
                pl[e] *= inv;
                p[e] = pl[e];
            }
            it.barrier();
            if (rank_scan) {
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
                        for (int e = lane; e < n_expert; e += 32) {
                            const float v = pl[e];
                            if (v > best || (v == best && e < bi)) {
                                best = v;
                                bi = e;
                            }
                        }
                        for (int mask = 16; mask > 0; mask >>= 1) {
                            const float ov = sycl::permute_group_by_xor(sgg, best, mask);
                            const int oi = sycl::permute_group_by_xor(sgg, bi, mask);
                            if (ov > best || (ov == best && oi < bi)) {
                                best = ov;
                                bi = oi;
                            }
                        }
                        if (lane == 0) {
                            ids[(size_t)t * top_k + r] = bi;
                            wl[r] = best;
                            pl[bi] = -3.402823466e38f;
                        }
                        sycl::group_barrier(sgg);
                    }
                }
            }
            it.barrier();
            if (tid == 0) {
                float sum = 0.f;
                for (int j = 0; j < top_k; j++) {
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
    sycl::event e0;
#define SI_MOE_R(GP, ESZ) e0 = moe_router_logits<GP>(q, gate_inp, ESZ, x, x_stride, n_tok, n_embd, n_expert, logits)
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
    auto fused = [&]() -> sycl::event {
        if (rt_wg == 32) {
            return moe_router_softmax_topk<32>(q, logits, n_tok, n_expert, top_k, ids, wts);
        }
        if (rt_wg == 128) {
            return moe_router_softmax_topk<128>(q, logits, n_tok, n_expert, top_k, ids, wts);
        }
        if (rt_wg == 256 || (rt_wg != 512 && n_expert <= 256)) {
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
        fprintf(stderr, "[moert] n=%d ne=%d wg=%d logits=%.3f fused=%.3f gap=%.3f ms\n", n_tok, n_expert,
                n_expert <= 256 ? 256 : 512, t0 - (double)e0.get_profiling_info<sycl::info::event_profiling::command_start>() / 1.0e6,
                te - ts, ts - t0);
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

namespace {

// Tile-shape dispatch.  The shape is a bandwidth/re-read/latency trade and all
// three terms are measured (see AGENTS.md):
//   NSG  columns per work-group is NSG*C, so a larger NSG means fewer
//        work-groups and fewer re-reads of the activation tile
//   C    columns per lane, i.e. dp4a issued per activation load
//   M    rows per tile, i.e. weight re-read factor ceil(p_e/M) (PF_MOE_HIST)
// The accumulator is M*FUSE*C floats and it has to stay in REGISTERS, so the
// product is the binding constraint - an M*C*FUSE that spills costs 350x.
// PF_MOE_CFG picks it; the default is the measured best of the sweep.
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
    static const int cfg_gate = si::env::i32("PF_MOE_CFG", 43);
    static const int cfg_down = si::env::i32("PF_MOE_CFG_DOWN", 43);
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
    static const int cfg_dec = si::env::i32("PF_MOE_CFG_DEC", 0);
    const int cfg = dec ? (cfg_dec ? cfg_dec : ((n_pairs <= top_k) ? 58 : 59))
                         : ((FUSE == 2) ? cfg_gate : cfg_down);
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
    static const int flat_dec = si::env::i32("PF_MOE_FLAT_DEC", 0);
    const bool flat = dec ? (flat_dec != 0) : (si::env::i32("PF_MOE_FLAT", 1) != 0);
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

void moe_down_q5_1_launch(queue & q, const void * dbank, const int32_t * slot_map, size_t slot_db,
                          const int32_t * off, const int32_t * pair, const int32_t * active,
                          const int32_t * n_active, int n_act_max, int n_expert, int n_pairs, const int8_t * xq,
                          const uint16_t * asa, const float * xs, float * acc, int acc_stride, int n_embd,
                          int n_ff_exp) {
    (void)n_expert; // the grid is bounded by n_act_max, as the callers pass it
    if (n_pairs <= 0 || n_embd <= 0 || (n_ff_exp % 32) != 0) {
        return;
    }
    moe_down_q5_1_impl<8>(q, (const char *)dbank, slot_map, slot_db, off, pair, active, n_active, n_act_max, xq, asa,
                          xs, acc, acc_stride, n_embd, n_ff_exp);
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

} // namespace si
