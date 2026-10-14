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

#include <cstdlib>
#include <cstring>
#include <stdexcept>
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
static void moe_router_logits(queue & q, const void * gate_inp, int esize, const float * x, int x_stride, int n_tok,
                              int n_embd, int n_expert, float * logits) {
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
    q.parallel_for(nd_range<1>((n_work * 32 + 31) & ~(size_t) 31, 32), [=](nd_item<1> it) {
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

// softmax over the n_expert logits of one token, in place
static void moe_router_softmax(queue & q, float * logits, int n_tok, int n_expert) {
    q.parallel_for(range<1>((size_t)n_tok), [=](id<1> i) {
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
static void moe_router_topk(queue & q, const float * probs, int n_tok, int n_expert, int top_k, void * out,
                            const int32_t * ids_in) {
    if (WANT_IDS) {
        q.parallel_for(range<2>((size_t)n_tok, (size_t)n_expert), [=](id<2> i) {
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
        q.parallel_for(range<1>((size_t)n_tok), [=](id<1> i) {
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

void moe_router_launch(queue & q, const void * gate_inp, uint32_t gp_type, const float * x, int x_stride, int n_tok,
                       int n_embd, int n_expert, int top_k, float * logits, int32_t * ids, float * wts) {
#define SI_MOE_R(GP, ESZ) moe_router_logits<GP>(q, gate_inp, ESZ, x, x_stride, n_tok, n_embd, n_expert, logits)
    switch (gp_type) {
    case GGML_TYPE_F32: SI_MOE_R(GGML_TYPE_F32, 4); break;
    case GGML_TYPE_F16: SI_MOE_R(GGML_TYPE_F16, 2); break;
    case GGML_TYPE_BF16: SI_MOE_R(GGML_TYPE_BF16, 2); break;
    default:
        throw std::runtime_error("moe: ffn_gate_inp must be f32/f16/bf16 (got " + std::to_string(gp_type) + ")");
    }
#undef SI_MOE_R
    moe_router_softmax(q, logits, n_tok, n_expert);
    moe_router_topk<true>(q, logits, n_tok, n_expert, top_k, ids, nullptr);
    moe_router_topk<false>(q, logits, n_tok, n_expert, top_k, wts, ids);
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
template <uint32_t QT, int FUSE, int M, int C, int NSG>
static void moe_grp_impl(queue & q, const char * w0, const char * w1, size_t plane0, size_t plane1,
                         const int32_t * slot_map, const int32_t * off, const int32_t * pair, const int32_t * active,
                         const int32_t * n_active, int n_act_max, const int8_t * xq, const uint16_t * asa, const float * xs,
                         float * out, int out_stride, int K, int N, int top_k) {
    constexpr int SG = 8; // 8 lanes x 4 values = one 32-value scale group
    constexpr int TX = NSG * SG;
    constexpr int RB = NSG * C; // columns one work-group owns
    const int ng = K / 32;
    const int nsb = K / 256; // superblocks per row
    const int sb = superblock_bytes(QT);
    const size_t row_bytes = (size_t)(K / 256) * (size_t)sb;
    const int nwg = N / RB;
    // Level Zero rejects a global size that is not a multiple of 256 threads
    // ("Non-uniform work-groups are not supported"), and n_act_max * nwg is a
    // product of two runtime extents - with a 2-row batch it is 16 work-groups
    // and the launch dies.  Round it up to an even count and let the surplus
    // work-groups exit on the `ia >= n_act_max` test.
    const int ngrid = ((n_act_max * nwg + 1) / 2) * 2;
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
    q.parallel_for(nd_range<1>((size_t)ngrid * TX, TX), [=](nd_item<1> it) {
        const int lid = (int)it.get_local_id(0);
        const int lane = lid % SG;
        const sub_group sgg = it.get_sub_group();
        const int gw = (int)it.get_group(0);
        const int ia = gw / nwg;
        if (ia >= n_act_max) {
            return;
        }
        if (ia >= n_active[0]) {
            return; // fewer experts selected than the grid was sized for
        }
        const int e = active[ia];
        const int slot = slot_map[e];
        if (slot < 0) {
            return; // host-resident expert: the host computes it
        }
        const char * b0 = w0 + (size_t)slot * plane0;
        const char * b1 = (FUSE == 2) ? (w1 + (size_t)slot * plane1) : b0;
        const int e1 = off[e + 1];
        const int n0 = (gw - ia * nwg) * RB + (lid / SG) * C;
        float acc[M][FUSE][C];
        for (int i0 = off[e]; i0 < e1; i0 += M) {
            const int np = (e1 - i0 < M) ? (e1 - i0) : M;
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
            int trowv[M];
#pragma unroll
            for (int m = 0; m < M; m++) {
                const int pr = (m < np) ? pair[i0 + m] : 0;
                trowv[m] = (FUSE == 2) ? (pr / top_k) : pr;
            }
            for (int sbi = 0; sbi < nsb; sbi++) {
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
                        sbbase[f][c] = bb + (size_t)n * row_bytes + (size_t)sbi * (size_t)sb;
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
                    const int g = sbi * 8 + gg;
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
                        const int trow = trowv[m];
                        const uint32_t xa = *reinterpret_cast<const uint32_t *>(
                            xq + (size_t)trow * (size_t)K + (size_t)g * 32 + (size_t)lane * 4);
                        const float A = moe_h2f(asa[(size_t)trow * (size_t)ng + g]);
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
                }
            }
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
                       int n_act_max, const int8_t * xq, const uint16_t * asa, const float * xs, float * out,
                       int out_stride, int K, int N, int top_k) {
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
    // The shipped default is M=16 / NSG=16 / C=2: 8.94 ms for the gate+up
    // projection and 6.03 ms for `down` at the 35B geometry (dev/bench_grp).
    // See AGENTS.md for the sweep - notably, down prefers C=1 (4.57 ms) and
    // M=32 costs less bandwidth but spills its accumulator.
    static const int cfg_gate = si::env::i32("PF_MOE_CFG", 0);
    static const int cfg_down = si::env::i32("PF_MOE_CFG_DOWN", 0);
    const int cfg = (FUSE == 2) ? cfg_gate : cfg_down;
#define SI_MOE_TILE3(MV, NSGV, CV)                                                                                            \
    do {                                                                                                                 \
        constexpr int NSG = NSGV, C = CV, RB = NSG * C;                                                                  \
        if ((K % 256) != 0 || (N % RB) != 0 || n_act_max <= 0) {                                                         \
            return false;                                                                                                 \
        }                                                                                                                \
        if (dbg) {                                                                                                       \
            fprintf(stderr, "[grp] fuse=%d qt=%u M=%d C=%d NSG=%d RB=%d K=%d N=%d grid=%zu n_act_max=%d\n", FUSE,        \
                    (unsigned)QT, MV, C, NSG, RB, K, N, (size_t)n_act_max * (size_t)(N / RB), n_act_max);                   \
        }                                                                                                                \
        moe_grp_impl<QT, FUSE, MV, C, NSG>(q, w0, w1, plane0, plane1, slot_map, off, pair, active, n_active, n_act_max,     \
                                           xq, asa, xs, out, out_stride, K, N, top_k);                                  \
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
    default: SI_MOE_TILE(16, 2);  // 128 thr, 32 col
    }
#undef SI_MOE_TILE
#undef SI_MOE_TILE3
}

} // namespace

bool moe_grp_gate_up_launch(queue & q, const void * gbank, const void * ubank, const int32_t * slot_map,
                            size_t slot_gb, size_t slot_ub, const int32_t * off, const int32_t * pair,
                            const int32_t * active, const int32_t * n_active, int n_act_max, const int8_t * xq,
                            const uint16_t * asa, const float * xs, float * h, int h_stride, int n_embd, int n_ff_exp,
                            int top_k, uint32_t gtype, uint32_t utype) {
    // one work-group reads both banks, so they must agree on K, N and type
    if (gtype != utype) {
        return false;
    }
    const char * g = (const char *)gbank, * u = (const char *)ubank;
#define SI_MOE_GRP_GU(QT)                                                                                                   \
    moe_grp_go<2, QT>(q, g, u, slot_gb, slot_ub, slot_map, off, pair, active, n_active, n_act_max, xq, asa, xs, h,     \
                      h_stride, n_embd, n_ff_exp, top_k)
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
                         const int8_t * xq, const uint16_t * asa, const float * xs, float * acc, int acc_stride,
                         int n_embd, int n_ff_exp, uint32_t dtype) {
    const char * d = (const char *)dbank;
#define SI_MOE_GRP_D(QT)                                                                                                   \
    moe_grp_go<1, QT>(q, d, nullptr, slot_db, 0, slot_map, off, pair, active, n_active, n_act_max, xq, asa, xs, acc,   \
                      acc_stride, n_ff_exp, n_embd, 1)
    switch (dtype) {
    case 12: return SI_MOE_GRP_D(12);
    case 13: return SI_MOE_GRP_D(13);
    case 14: return SI_MOE_GRP_D(14);
    default: return false;
    }
#undef SI_MOE_GRP_D
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
                             int x_stride, float * h, int h_stride, int n_embd, int n_ff_exp, int top_k) {
    const int n_sb = n_embd / 256;
    const int sgb = superblock_bytes(GT), sub = superblock_bytes(UT);
    // ONE WARP PER OUTPUT ROW, so the rows-per-expert count is the row count
    // itself: nw warps cover n_act_max experts x n_ff_exp rows.
    const int nrb = n_ff_exp;
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
        const int n = g % nrb;
        if (ia >= n_active[0]) {
            return;
        }
        const int e = active[ia];
        const int slot = slot_map[e];
        if (slot < 0) {
            return; // host-resident expert: the host computes it
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
        float ag[kMoEPB], au[kMoEPB];
        for (int i0 = off[e]; i0 < off[e + 1]; i0 += kMoEPB) {
            const int np = (off[e + 1] - i0 < kMoEPB) ? (off[e + 1] - i0) : kMoEPB;
#pragma unroll
            for (int j = 0; j < kMoEPB; j++) {
                ag[j] = 0.f;
                au[j] = 0.f;
            }
            for (int sb = 0; sb < n_sb; sb++) {
                float w[8], u[8];
                dequant_sb_lane_typed<GT>(gr + (size_t)sb * sgb, lane, w);
                dequant_sb_lane_typed<UT>(ur + (size_t)sb * sub, lane, u);
#pragma unroll
                for (int j = 0; j < kMoEPB; j++) {
                    if (j < np) {
                        const float * xr = x + (size_t)(pair[i0 + j] / top_k) * x_stride;
#pragma unroll
                        for (int b = 0; b < 8; b++) {
                            const float xv = xr[(size_t)sb * 256 + 32 * b + lane];
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
        }
    });
}

void moe_gate_up_launch(queue & q, const void * gbank, const void * ubank, const int32_t * slot_map, size_t slot_gb,
                        size_t slot_ub, const int32_t * off, const int32_t * pair, const int32_t * active,
                        const int32_t * n_active, int n_act_max, const float * x, int x_stride, float * h, int h_stride,
                        int n_embd, int n_ff_exp, int top_k, uint32_t gtype, uint32_t utype) {
#define SI_MOE_GU(GT, UT)                                                                                                  \
    moe_gate_up_impl<GT, UT>(q, (const char *)gbank, (const char *)ubank, slot_map, slot_gb, slot_ub, off, pair, active, \
                             n_active, n_act_max, x, x_stride, h, h_stride, n_embd, n_ff_exp, top_k)
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
                          const float * h, int h_stride, float * acc, int acc_stride, int n_embd, int n_ff_exp) {
    const int n_sb = n_ff_exp / 256;
    const int sdb = superblock_bytes(DT);
    const int nrb = n_embd; // one warp per output row
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
        const int n = g % nrb;
        if (ia >= n_active[0]) {
            return;
        }
        const int e = active[ia];
        const int slot = slot_map[e];
        if (slot < 0) {
            return;
        }
        const char * dr = dbank + (size_t)slot * slot_db + (size_t)n * n_sb * sdb;
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
                dequant_sb_lane_typed<DT>(dr + (size_t)sb * sdb, lane, w);
#pragma unroll
                for (int j = 0; j < kMoEPB; j++) {
                    if (j < np) {
                        const float * hr = h + (size_t)pair[i0 + j] * h_stride;
#pragma unroll
                        for (int b = 0; b < 8; b++) {
                            a[j] += w[b] * hr[(size_t)sb * 256 + 32 * b + lane];
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
        }
    });
}

void moe_down_launch(queue & q, const void * dbank, const int32_t * slot_map, size_t slot_db, const int32_t * off,
                     const int32_t * pair, const int32_t * active, const int32_t * n_active, int n_act_max,
                     const float * h, int h_stride, float * acc, int acc_stride, int n_embd, int n_ff_exp,
                     uint32_t dtype) {
#define SI_MOE_D(DT)                                                                                             \
    moe_down_impl<DT>(q, (const char *)dbank, slot_map, slot_db, off, pair, active, n_active, n_act_max, h, h_stride, \
                      acc, acc_stride, n_embd, n_ff_exp)
    switch (dtype) {
    case 12: SI_MOE_D(12); break;
    case 13: SI_MOE_D(13); break;
    case 14: SI_MOE_D(14); break;
    case 8: SI_MOE_D(8); break;
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
