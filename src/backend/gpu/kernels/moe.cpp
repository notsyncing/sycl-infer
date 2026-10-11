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
    q.parallel_for(range<2>((size_t)n_tok, (size_t)n_expert), [=](id<2> i) {
        const int t = (int) i[0], e = (int) i[1];
        // gate_inp is [n_expert][n_embd] row-major, so expert e's row starts at
        // e*n_embd ELEMENTS.  Indexing that in bytes (e*n_embd) reads four
        // experts per stride and yields a plausible, entirely wrong logit
        // vector - which is exactly what it did.
        const char * wr = (const char *) gate_inp + (size_t) e * (size_t) n_embd * esize;
        const float * xr = x + (size_t) t * x_stride;
        float a = 0.f;
        for (int k = 0; k < n_embd; k++) {
            a += moe_rd<RT>(wr, k) * xr[k];
        }
        logits[(size_t) t * n_expert + e] = a;
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
// top_k.  O(n_expert^2) comparisons in one work-item, but n_expert is a few
// hundred and this is n_tok work-items, so it is cheaper than any cross-lane
// scheme and it has no barrier to get wrong.
//
// The ids and the weights are two kernels, not one, and that is not a style
// choice: with both `int32_t * ids` and `float * wts` as arguments of a single
// kernel, this toolchain drops every store through `ids` - including an
// unconditional one outside any branch - while the stores through `wts` in the
// same basic block land and are bit-exact against the reference.  Split, both
// halves are correct.  The rank is recomputed rather than read back from ids so
// each kernel keeps a single output pointer; n_expert is 37-256 and n_tok is at
// most 32, so the duplicate pass is a few million comparisons.
template <bool WANT_IDS>
static void moe_router_topk(queue & q, const float * probs, int n_tok, int n_expert, int top_k, void * out) {
    q.parallel_for(range<1>((size_t)n_tok), [=](id<1> i) {
        const float * p = probs + (size_t) i[0] * n_expert;
        int32_t * id = WANT_IDS ? (int32_t *) out + (size_t) i[0] * top_k : nullptr;
        float * wt = WANT_IDS ? nullptr : (float *) out + (size_t) i[0] * top_k;
        float sum = 0.f;
        for (int e = 0; e < n_expert; e++) {
            const float pe = p[e];
            int rank = 0;
            for (int f = 0; f < n_expert; f++) {
                const float pf = p[f];
                if (pf > pe || (pf == pe && f < e)) {
                    rank++;
                }
            }
            if (rank < top_k) {
                if (WANT_IDS) {
                    id[rank] = e;
                } else {
                    wt[rank] = pe;
                    sum += pe;
                }
            }
        }
        if (!WANT_IDS) {
            const float den = sycl::fmax(sum, 6.103515625e-5f);
            for (int j = 0; j < top_k; j++) {
                wt[j] /= den;
            }
        }
    });
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
    moe_router_topk<true>(q, logits, n_tok, n_expert, top_k, ids);
    moe_router_topk<false>(q, logits, n_tok, n_expert, top_k, wts);
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

// ---------------------------------------------------------------------------
// gate + up for every (token, expert) pair in one pass per expert row block.
// Both tensors are read in the same loop so the activation row is fetched once
// for the pair, and what is stored is silu(gate) * up - all `down` needs.
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
        for (int i = off[e]; i < off[e + 1]; i++) {
            const int pr = pair[i];
            const float * xr = x + (size_t)(pr / top_k) * x_stride;
            const float gv = sg_sum(moe_dot<GT>(gr, xr, n_sb, sgb, lane), sgg);
            const float uv = sg_sum(moe_dot<UT>(ur, xr, n_sb, sub, lane), sgg);
            if (lane == 0) {
                h[(size_t)pr * h_stride + n] = silu_f(gv) * uv;
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
        for (int i = off[e]; i < off[e + 1]; i++) {
            const int pr = pair[i];
            const float a = sg_sum(moe_dot<DT>(dr, h + (size_t)pr * h_stride, n_sb, sdb, lane), sgg);
            if (lane == 0) {
                acc[(size_t)pr * acc_stride + n] = a;
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
