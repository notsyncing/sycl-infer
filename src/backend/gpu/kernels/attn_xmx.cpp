#include "kernels.h"
#include "kernel_utils.h"

#include "dnnl_gemm.h"
#include "device/device_profile.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>
#include "common/env.h"

namespace si {

using namespace sycl;
using namespace si::kd;

// ===========================================================================
// XMX (oneDNN int8) prefill attention - opt-in via PF_ATTN_XMX=1.
//
// The classic prefill kernel is bound by per-score scalar work: one warp owns
// (row, token, query head, split) and every key costs a 32-lane dot, a
// sub-group reduction and an exp, and each K/V row is re-read once per query
// head (HPG-fold redundancy).  This path expresses the two attention GEMMs as
// plain int8 oneDNN matmuls on the A770 XMX units:
//
//   QK^T : [M, 256] s8  x  [256, nk] s8 -> [M, nk] s32   (dnnl::matmul)
//   softmax + requantize P -> u8 (per-row scale), causal mask per query row
//   PV   : [M, nk] u8   x  [nk, 256] s8 -> [M, 256] s32   (dnnl::matmul)
//   online-softmax rescale + accumulate, then divide by l
//
// M stacks all HPG query heads that share one kv head, for every active
// (row, token) of the batch, so a single matmul covers HPG*kMaxQ query rows
// and the paged KV is gathered (and re-quantized) once per kv head per key
// block instead of once per query head.
//
// A plain int8 matmul sums over k, so a per-key scale cannot be applied after
// it: K and V must each share ONE block-wide scale.  The 32-group source i8
// scales are folded into the values first, then the block max picks the scale.
//
// Scope: i8 KV, head_dim 256, oneDNN available.  Handles both the fused
// (out != nullptr, n_splits == 1) and the split (partials) shapes.
// ===========================================================================

// Per queue, not process-wide: the default comes from that queue's device
// profile (the oneDNN int8 matmul path pays off on a part with a fast int8 GEMM
// and is untested where DP4A was worth 4x over the scalar path), and a split
// across two cards must be able to enable it on one and not the other.
// PF_ATTN_XMX=0 restores the classic kernel.
bool attn_xmx_enabled(const sycl::queue & q) {
    return si::dev::profile_flag(q, "PF_ATTN_XMX", si::dev::for_queue(q).attn.xmx);
}

// XMX only pays off once the classic kernel's per-score work dominates the
// oneDNN launch/scratch overhead.  After the index-array fix (see the report)
// the cross-over on 27B / 2x A770 is ~1.5-2k keys: classic chunks cost
// ~437 ms at pos 512 rising ~0.078 ms/token, while an XMX chunk is a flat
// ~480-560 ms.  A 16k cold prefill totals 16.7 s at 2048 vs 18.4 s at 6144,
// and 1024/0 are within noise (16.7-16.8 s) while being slower on short
// prompts.  Below this key count the classic kernel is used.
static int xmx_min_keys(const sycl::queue & q) {
    return si::dev::profile_int(q, "PF_ATTN_XMX_MIN", si::dev::for_queue(q).attn.xmx_min_keys);
}

static constexpr int kD = 256;    // head_dim (fixed for this model)
static constexpr int kBlk = 8192; // keys per matmul block
static constexpr int kMaxQ = kMaxT * kMaxB;

// Work-groups in the gather's first-stage max reduction.  The reduction used to
// be a single 256-thread work-group over the whole key block, which on a 512-EU
// device was the worst-scaled piece of the prefill; see xmx_gather.  Both counts
// come from the device profile (src/device/profile_*.cpp): it is a latency fix,
// so the right value depends on how many EUs there are to hide it with.
// PF_XMX_GRED=N overrides the work-group count for A/B.
// Read from the queue's own profile where they are used (see below) rather than
// frozen at namespace scope: the scratch buffer is sized per queue, so a
// process-wide count would both mis-size one card's buffer and pick the wrong
// work-group count for the other.
static int gather_red_max(const sycl::queue & q) {
    return si::dev::for_queue(q).attn.xmx_gather_red_max;
}
static int gather_red(const sycl::queue & q) {
    return si::dev::profile_int(q, "PF_XMX_GRED", si::dev::for_queue(q).attn.xmx_gather_red);
}
// Work-group width for the gather's requantize stage (`xmx_gather` pass 2).
// Held constant at 64 across every `PF_XMX_GQ` setting so the sweep varies only
// the lanes-per-key-row stripe and not the work-group width - see the note there.
static constexpr int QG_WG = 64;
// Threads per key row in the gather's requantize stage (`xmx_gather` pass 2).
// One is the old `range<1>` shape; kD/32 = 8 is the point where every lane owns
// exactly one f16 scale group.  Above 8 lanes idle, so the range stops there, and
// only powers of two are accepted so `QG_WG/qg` rows per work-group is exact.
static int gather_q(const sycl::queue & q) {
    const int want = si::dev::profile_int(q, "PF_XMX_GQ", 8);
    if (want < 1) {
        return 1;
    }
    int v = 1;
    while (v * 2 <= want && v * 2 <= kD / 32 && v * 2 <= QG_WG) {
        v *= 2;
    }
    return v;
}

// Width used for the *last* key block of a call.  The oneDNN matmuls always run
// at a fixed width (primitive creation is ~15 ms per new shape, so a width that
// followed the chunk size made prims never reusable -- a 12x regression), which
// means every block runs at kBlk columns even when only a few hundred keys
// remain, and the QK matmul, the softmax's pass over the score row, the PV matmul
// and the K/V gather all zero-fill and then compute those columns for nothing.
//
// The waste is largest at *shallow* depth and nearly gone at 64k, because the
// true key count is `pos0 + 512`, not `depth + 512`: a chunk ending at 4096
// needs 4608 keys = 1 block = 44 % padding, at 16384 31 %, and at 65536 only
// 0.8 % (65024 keys into 8 blocks of 8192).  Shrinking only the tail block costs
// at most log2(kBlk) extra primitive shapes per M, which are then cached.
//
// Measured: NEUTRAL end to end at both 16k (563.7/565.3/566.7 vs
// 563.3/565.9/567.3 ms per marginal chunk) and 64k (971.0/972.7/975.8 vs
// 971.3/972.8/978.3 ms) -- the padded columns turn out to be nearly free.  Kept
// because it cannot lose; `PF_XMX_TAILBLK=0` restores the old behaviour.
static inline int xmx_tail_blk(int rem) {
    static const bool off = [] {
        const char * e = si::env::str("PF_XMX_TAILBLK");
        return e && atoi(e) == 0;
    }();
    if (off) {
        return kBlk;
    }
    int b = kBlk;
    while (b > 256 && rem <= b / 2) {
        b >>= 1;
    }
    return b;
}

struct xmx_bufs {
    int8_t * k = nullptr;  // [kBlk * kD]
    int8_t * v = nullptr;  // [kBlk * kD]
    float * bs = nullptr;  // [2 + 2*kGatherRed] block scales + gather max partials
    int8_t * q8 = nullptr; // [Mcap * kD]
    float * qsc = nullptr; // [Mcap]
    int32_t * orow = nullptr; // [Mcap] qbuf row index (r * tpb + t)
    int32_t * oh = nullptr;   // [Mcap] query head
    int32_t * olim = nullptr; // [Mcap] causal limit (absolute key count)
    int32_t * qk = nullptr;   // [Mcap * kBlk]
    uint8_t * p = nullptr;    // [Mcap * kBlk]
    float * ps = nullptr;     // [Mcap]
    int32_t * pv = nullptr;   // [Mcap * kD]
    float * acc = nullptr;    // [Mcap * kD]
    float * m = nullptr;      // [Mcap]
    float * l = nullptr;      // [Mcap]
    int cap_m = 0;
    int cap_keys = 0;
};

// One scratch set per device.  A function-local static here was shared by
// every device in a --layer-map run, so device 1's matmuls read/wrote device
// 0's USM (cross-device page thrash, ~12x slower).  Key on the queue pointer,
// which is a stable engine member.
static std::vector<std::pair<queue *, xmx_bufs>> & xmx_reg() {
    static std::vector<std::pair<queue *, xmx_bufs>> v;
    return v;
}

static xmx_bufs & xmx_get(queue & q, int m_cap, int key_cap) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lk(mtx);
    auto & reg = xmx_reg();
    xmx_bufs * pb = nullptr;
    for (auto & e : reg) {
        if (e.first == &q) {
            pb = &e.second;
            break;
        }
    }
    if (!pb) {
        reg.emplace_back(&q, xmx_bufs{});
        pb = &reg.back().second;
    }
    xmx_bufs & b = *pb;
    // Only the query-tile capacity sizes a buffer; the key block is always
    // kBlk wide and the K/V scratch is fixed, so max_nkv growth must NOT
    // reallocate (a per-chunk realloc of the [M,kBlk] scratch cost seconds).
    (void)key_cap;
    const bool big = m_cap > b.cap_m;
    if (big) {
        auto fr = [&](auto * p) {
            if (p) {
                sycl::free(p, q);
            }
        };
        fr(b.k);
        fr(b.v);
        fr(b.bs);
        fr(b.q8);
        fr(b.qsc);
        fr(b.orow);
        fr(b.oh);
        fr(b.olim);
        fr(b.qk);
        fr(b.p);
        fr(b.ps);
        fr(b.pv);
        fr(b.acc);
        fr(b.m);
        fr(b.l);
    }
    if (big || !b.k) {
        b.cap_m = m_cap;
        b.cap_keys = key_cap;
        b.k = sycl::malloc_device<int8_t>((size_t)kBlk * kD, q);
        b.v = sycl::malloc_device<int8_t>((size_t)kBlk * kD, q);
        b.bs = sycl::malloc_device<float>(2 + 2 * gather_red_max(q), q);
        b.q8 = sycl::malloc_device<int8_t>((size_t)m_cap * kD, q);
        b.qsc = sycl::malloc_device<float>(m_cap, q);
        b.orow = sycl::malloc_device<int32_t>(m_cap, q);
        b.oh = sycl::malloc_device<int32_t>(m_cap, q);
        b.olim = sycl::malloc_device<int32_t>(m_cap, q);
        b.qk = sycl::malloc_device<int32_t>((size_t)m_cap * kBlk, q);
        b.p = sycl::malloc_device<uint8_t>((size_t)m_cap * kBlk, q);
        b.ps = sycl::malloc_device<float>(m_cap, q);
        b.pv = sycl::malloc_device<int32_t>((size_t)m_cap * kD, q);
        b.acc = sycl::malloc_device<float>((size_t)m_cap * kD, q);
        b.m = sycl::malloc_device<float>(m_cap, q);
        b.l = sycl::malloc_device<float>(m_cap, q);
        // PF_XMX_POISON=1: fill every freshly allocated scratch buffer with a
        // recognisable garbage pattern instead of leaving malloc_device's
        // uninitialised bytes in place.  This is the direct test of "a reused
        // scratch leaks the previous call's bytes or reads never-written bytes",
        // and it is the only hypothesis that fits the observed SHAPE: the whole
        // logit vector shifts by a small amount uniformly rather than a few
        // elements going wrong, and `bs[0]`/`bs[1]` - one scale per 8192-key
        // block - is the only place in this file where an entire block's data
        // collapses into a single number.  A different block max requantizes the
        // whole block slightly differently and every query row moves.
        //
        // 0x7F7F7F7F as float is ~1.7e38 and as int is a large positive int32, so
        // a read of poisoned memory is loud rather than quietly plausible.  If the
        // divergence is unaffected, uninitialised scratch is refuted outright.
        if (si::env::flag("PF_XMX_POISON")) {
            constexpr uint8_t kP = 0x7F;
            auto poison = [&](void * ptr, size_t bytes) {
                if (ptr) {
                    q.memset(ptr, kP, bytes).wait();
                }
            };
            poison(b.k, (size_t)kBlk * kD);
            poison(b.v, (size_t)kBlk * kD);
            poison(b.bs, (2 + 2 * gather_red_max(q)) * sizeof(float));
            poison(b.q8, (size_t)m_cap * kD);
            poison(b.qsc, (size_t)m_cap * sizeof(float));
            poison(b.orow, (size_t)m_cap * sizeof(int32_t));
            poison(b.oh, (size_t)m_cap * sizeof(int32_t));
            poison(b.olim, (size_t)m_cap * sizeof(int32_t));
            poison(b.qk, (size_t)m_cap * kBlk * sizeof(int32_t));
            poison(b.p, (size_t)m_cap * kBlk);
            poison(b.ps, (size_t)m_cap * sizeof(float));
            poison(b.pv, (size_t)m_cap * kD * sizeof(int32_t));
            poison(b.acc, (size_t)m_cap * kD * sizeof(float));
            poison(b.m, (size_t)m_cap * sizeof(float));
            poison(b.l, (size_t)m_cap * sizeof(float));
        }
    }
    return b;
}

// Stacked Q tile [M][256] -> s8 with one per-row scale.  Row i comes from
// qbuf row orow[i] at query head oh[i].
static void xmx_quant_q(queue & q, const float * qbuf, int qstride, const int32_t * orow, const int32_t * oh, int M,
                        int8_t * q8, float * qsc, float * m, float * l, float * acc) {
    // One launch does the Q quantization AND the (m, l, acc) reset: the reset
    // was two extra launches per kv head and the per-kernel submission overhead
    // dominates this path.
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>((size_t)M), [=](sycl::id<1> id) {
            const int i = id[0];
            const float * qh = qbuf + (size_t)orow[i] * qstride + (size_t)oh[i] * 2 * kD;
            float amax = 0;
            for (int c = 0; c < kD; c++) {
                amax = sycl::fmax(amax, sycl::fabs(qh[c]));
            }
            const float sr = amax > 0 ? amax / 127.f : 1.f;
            qsc[i] = sr;
            for (int c = 0; c < kD; c++) {
                const float v = qh[c] / sr;
                q8[(size_t)i * kD + c] = (int8_t)sycl::round(sycl::fmin(127.f, sycl::fmax(-127.f, v)));
                acc[(size_t)i * kD + c] = 0.f;
            }
            m[i] = -INFINITY;
            l[i] = 0.f;
        });
    });
}

// Gather the key block [k0, k0+nk) of kv head kvh into contiguous [nk][256]
// with ONE scale for the whole block.  bs[0] = K block scale, bs[1] = V block
// scale.
// PF_XMX_SUM: per-gather-call device-side integer checksums, read exactly ONCE at
// process exit.
//
// Same design rule and the same reason as PF_ACTQ_SUM: a host read-back inside the
// forward is precisely what made the per-layer fingerprint oracle useless, so these
// accumulate on the device and nothing is read back until the process is done.
//
// The point is the SPLIT.  `sumkv` hashes the gathered K/V bytes and `sumbs` the two
// block scales, separately.  If `sumkv` moves, the paged pool's contents are changing;
// if only `sumbs` moves, the gather read identical bytes and the two-stage fmax
// reduction is what differs.  That is the one measurement left after the arm
// bisection showed `attn_xmx` creates the difference and the size said "a ulp in the
// exponent of the score requantization", which leaves exactly these two candidates.
//
// The queue is held BY VALUE so the holder can read back safely at exit even if the
// engine has already gone; a SYCL queue keeps its context and device alive.
struct xmx_sum_holder {
    static constexpr int kSlots = 8192;
    sycl::queue q;
    int64_t * sumkv = nullptr;
    int64_t * sumbs = nullptr;
    int n = 0;
    ~xmx_sum_holder() {
        if (!sumkv || n == 0) {
            return;
        }
        try {
            std::vector<int64_t> hkv((size_t)n), hbs((size_t)n * 2);
            q.memcpy(hkv.data(), sumkv, (size_t)n * 8).wait();
            q.memcpy(hbs.data(), sumbs, (size_t)n * 16).wait();
            uint64_t hkv_h = 1469598103934665603ULL, hbs_h = 1469598103934665603ULL;
            for (int i = 0; i < n; i++) {
                hkv_h = (hkv_h ^ (uint64_t)hkv[i]) * 1099511628211ULL;
                for (int k = 0; k < 2; k++) {
                    hbs_h = (hbs_h ^ (uint64_t)hbs[2 * i + k]) * 1099511628211ULL;
                }
            }
            fprintf(stderr, "[xmxsum] calls=%d kv=%016llx bs=%016llx\n", n, (unsigned long long)hkv_h,
                    (unsigned long long)hbs_h);
            for (int i = 0; i < n; i++) {
                fprintf(stderr, "[xmxsumslot] %d %lld %lld %lld\n", i, (long long)hkv[i], (long long)hbs[2 * i],
                        (long long)hbs[2 * i + 1]);
            }
        } catch (sycl::exception & e) {
            fprintf(stderr, "[xmxsum] readback failed: %s\n", e.what());
        }
    }
    int slot(queue & qq) {
        if (!si::env::flag("PF_XMX_SUM")) {
            return -1;
        }
        if (!sumkv) {
            q = qq;
            sumkv = sycl::malloc_device<int64_t>(kSlots, q);
            sumbs = sycl::malloc_device<int64_t>(kSlots * 2, q);
            if (!sumkv || !sumbs) {
                sumkv = sumbs = nullptr;
                return -1;
            }
            // No .wait(): the multi-device decode command graph is recorded at
            // startup and replayed, and a host wait on an event belonging to a
            // recorded graph throws.  The in-order queue already orders the memset
            // before every kernel submitted after it.
            q.memset(sumkv, 0, (size_t)kSlots * 8);
            q.memset(sumbs, 0, (size_t)kSlots * 16);
        }
        return n < kSlots ? n++ : -1;
    }
};
static xmx_sum_holder g_xmx_sum;

static void xmx_gather(queue & q, const int8_t * kp_base, const int8_t * vp_base, const sycl::half * ksc_base,
                       const sycl::half * vsc_base, const int32_t * table, int n_head_kv, int kvh, int k0, int nk, int blk,
                       int8_t * gk, int8_t * gv, float * bs, int slot, int64_t * sumkv, int64_t * sumbs) {
    // The oneDNN matmul runs at the fixed block width `blk` (primitive creation
    // is ~15 ms per new shape, so the shape must not follow the chunk size).
    // Keys beyond nk are zero-filled; the softmax's causal limit (<= nk for the
    // last block) masks them and a zero V row contributes nothing to PV.
    //
    // Pass 1 reduces max|k| and max|v| over the block so the requantization has
    // ONE scale per K/V pair (a plain int8 matmul sums over k, so a per-key
    // scale cannot be applied after it).  This used to be a *single* 256-thread
    // work-group reducing all nk rows, which on a 512-EU device is the single
    // worst-scaled piece of the whole prefill: measured per attention layer at
    // 64k depth it was 18.1 ms of the 37.8 ms the layer costs (the QK matmul is
    // 6.7, the softmax 6.1, the PV matmul 4.3) -- one work-group cannot cover
    // enough of the memory system to get anywhere near the bandwidth ceiling.
    // It is now a two-stage reduction: `kGatherRed` work-groups each reduce a
    // slice into `part`, then one small group folds those.  fmax is exact and
    // associative, so the resulting block scale -- and therefore the whole
    // requantized block, bit for bit -- is unchanged.
    // per call, because it comes from this queue's device; by value because a
    // kernel lambda cannot capture a runtime-initialised global
    const int nslice = gather_red(q);
    float * part = bs + 2;         // [2 * gather_red_max(q)] partials, past bs[0..1]
    q.submit([&](sycl::handler & h) {
        h.parallel_for(nd_range<1>((size_t)nslice * 64, 64), [=](nd_item<1> it) {
            const int s = it.get_group(0);
            const int tid = it.get_local_linear_id();
            float km = 0, vm = 0;
            // each slice takes a contiguous run of rows, one row per lane
            for (int row = s * 64 + tid; row < nk; row += nslice * 64) {
                const int kk = k0 + row;
                const int kb = table[kk / kBlockSize];
                const int ko = kk % kBlockSize;
                const size_t unit = (size_t)kb * n_head_kv + kvh;
                const int8_t * krow = kv_row_data(kp_base, unit, ko, kD);
                const int8_t * vrow = kv_row_data(vp_base, unit, ko, kD);
                const sycl::half * ksc = kv_row_scales(ksc_base, unit, ko, kD);
                const sycl::half * vsc = kv_row_scales(vsc_base, unit, ko, kD);
// One fp16 scale per 32 head dims, so max|k*s| over a group is
                // |s| * max|k| -- exact, and it collapses 256 multiply+fabs+max
                // into 8.  Four groups are done per iteration into four
                // independent accumulators: the scalar form was one 256-long
                // fmax dependency chain per lane, which is what this stage's
                // remaining 9 ms is.  Data loads are 16-byte vectors.
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                float vacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int g0 = 0; g0 < kD / 32; g0 += 4) {
                    const sycl::vec<sycl::half, 4> shk =
                        vload<sycl::vec<sycl::half, 4>, sycl::half>(ksc + g0);
                    const sycl::vec<sycl::half, 4> shv =
                        vload<sycl::vec<sycl::half, 4>, sycl::half>(vsc + g0);
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        const int g = g0 + u;
                        const sycl::vec<int8_t, 16> a0v = vload<sycl::vec<int8_t, 16>, int8_t>(krow + g * 32);
                        const sycl::vec<int8_t, 16> a1v = vload<sycl::vec<int8_t, 16>, int8_t>(krow + g * 32 + 16);
                        const sycl::vec<int8_t, 16> b0v = vload<sycl::vec<int8_t, 16>, int8_t>(vrow + g * 32);
                        const sycl::vec<int8_t, 16> b1v = vload<sycl::vec<int8_t, 16>, int8_t>(vrow + g * 32 + 16);
                        float mk = 0, mv = 0;
#pragma unroll
                        for (int j = 0; j < 16; j++) {
                            mk = sycl::fmax(mk, sycl::fabs((float)a0v[j]));
                            mk = sycl::fmax(mk, sycl::fabs((float)a1v[j]));
                            mv = sycl::fmax(mv, sycl::fabs((float)b0v[j]));
                            mv = sycl::fmax(mv, sycl::fabs((float)b1v[j]));
                        }
                        acc[u] = sycl::fmax(acc[u], sycl::fabs((float)shk[u]) * mk);
                        vacc[u] = sycl::fmax(vacc[u], sycl::fabs((float)shv[u]) * mv);
                    }
                }
                km = sycl::fmax(sycl::fmax(acc[0], acc[1]), sycl::fmax(acc[2], acc[3]));
                vm = sycl::fmax(sycl::fmax(vacc[0], vacc[1]), sycl::fmax(vacc[2], vacc[3]));
            }
            part[s] = km;             // slice-private slot, no atomic needed
            part[nslice + s] = vm;
        });
    });
    // stage 2: fold the slices into the two block scales (nslice <= 64 floats)
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(2), [=](sycl::id<1> id) {
            float m = 0;
            const float * p = part + (id[0] == 0 ? 0 : nslice);
            for (int s = 0; s < nslice; s++) {
                m = sycl::fmax(m, p[s]);
            }
            bs[id[0]] = m > 0 ? m / 127.f : 1.f;
            if (sumbs) {
                sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                    ab(sumbs[2 * slot + (int)id[0]]);
                ab.fetch_add((int64_t)*(const uint32_t *)&bs[id[0]]);
            }
        });
    });
    // pass 2: quantize every key row with the block scales, zero-fill the pad.
    //
    // This is the fix pass 1 already got, applied to the stage that never had
    // it.  `range<1>(blk)` is ONE work-item per key: at the fixed matmul width
    // blk = kBlk = 8192 that is 8192 work-items = 256 warps on a 512-EU card -
    // half the machine with nothing to schedule - and each of those threads then
    // walked kD = 256 elements one byte at a time, behind a dependent
    // `table[kk / kBlockSize]` load.
    //
    // Per key the stage moves 256 (K) + 256 (V) + 32 (two f16 scale planes,
    // 8 groups each) in and 512 out, so 1056 B, and a full 8192-key iteration is
    // ~8.7 MB - ~23 us at the card's 376 GB/s.  It measured ~386 us per
    // iteration (3472 us per call at 64k, 9 iterations), i.e. **17x the
    // bandwidth floor**, while the softmax beside it sat at 48% of that floor.
    //
    // `qg` lanes per key row inside a QG_WG-thread work-group, so `QG_WG/qg` rows are
    // in flight per work-group and each lane owns whole 32-value scale groups.
    // The work-group width is held CONSTANT across the whole sweep on purpose:
    // letting it vary with `qg` would confound "how many lanes per row" with
    // "how many lanes a warp keeps busy", and on a SIMD32 part an 8-thread
    // work-group is the worst possible width for the second question.
    // `PF_XMX_GQ` sweeps `qg` alone; 1 is the old one-thread-per-row shape with
    // the 16-byte accesses kept, so the sweep separates the two effects.
    //
    // The scale stays inside the multiply: `(float)k[c] * (float)ksc[g] / ksr`
    // is the *original* association, and hoisting it to `(k[c] * (ksc[g]/ksr))`
    // would be a different fp32 expression and a different bit pattern.  Only
    // the loop-invariant `ksc[g]` is hoisted.
    const int qg = gather_q(q);
    const int rpw = QG_WG / qg; // rows per work-group
    q.submit([&](sycl::handler & h) {
        h.parallel_for(nd_range<1>((size_t)((blk + rpw - 1) / rpw) * QG_WG, QG_WG), [=](nd_item<1> it) {
            const int lr = (int)it.get_local_linear_id();
            const int row = (int)it.get_group(0) * rpw + lr / qg;
            const int gl = lr % qg;
            if (row >= blk) {
                return;
            }
            int8_t * gkr = gk + (size_t)row * kD;
            int8_t * gvr = gv + (size_t)row * kD;
            if (row >= nk) {
                // cold path (only the last chunk's tail).  Scalar, so it needs no
                // vec construction and cannot fail the 16-byte alignment the live
                // path relies on.  Strided by `gl`, NOT by `lr`: a work-group spans
                // QG_WG/qg different rows, so `lr` would give each row only the
                // bytes its own lane slot happens to land on and leave the rest of
                // the pad row holding whatever the previous chunk left there.
                for (int c = gl; c < kD; c += qg) {
                    gkr[c] = 0;
                    gvr[c] = 0;
                }
                return;
            }
            const int kk = k0 + row;
            const int kb = table[kk / kBlockSize];
            const int ko = kk % kBlockSize;
            const size_t unit = (size_t)kb * n_head_kv + kvh;
            const int8_t * krow = kv_row_data(kp_base, unit, ko, kD);
            const int8_t * vrow = kv_row_data(vp_base, unit, ko, kD);
            const sycl::half * ksc = kv_row_scales(ksc_base, unit, ko, kD);
            const sycl::half * vsc = kv_row_scales(vsc_base, unit, ko, kD);
            const float ksr = bs[0], vsr = bs[1];
            for (int g = gl; g < kD / 32; g += qg) { // one f16 scale per 32 dims
                const float ksg = (float)ksc[g];
                const float vsg = (float)vsc[g];
                for (int u = 0; u < 2; u++) { // two 16-byte vectors per group
                    const int c = g * 32 + u * 16;
                    const sycl::vec<int8_t, 16> a = vload<sycl::vec<int8_t, 16>, int8_t>(krow + c);
                    const sycl::vec<int8_t, 16> b = vload<sycl::vec<int8_t, 16>, int8_t>(vrow + c);
                    sycl::vec<int8_t, 16> oa, ob;
#pragma unroll
                    for (int j = 0; j < 16; j++) {
                        const float kv = (float)a[j] * ksg / ksr;
                        const float vv = (float)b[j] * vsg / vsr;
                        oa[j] = (int8_t)sycl::round(sycl::fmin(127.f, sycl::fmax(-127.f, kv)));
                        ob[j] = (int8_t)sycl::round(sycl::fmin(127.f, sycl::fmax(-127.f, vv)));
                    }
                    vstore<sycl::vec<int8_t, 16>, int8_t>(gkr + c, oa);
                    vstore<sycl::vec<int8_t, 16>, int8_t>(gvr + c, ob);
                }
            }
        });
    });
    // PF_XMX_SUM: hash what pass 2 just wrote.  A separate launch rather than an
    // extra accumulation inside pass 2, deliberately: pass 1's four fmax
    // accumulators plus two more 64-bit values would change register pressure and
    // therefore the timing of the stage this investigation is measuring.  One extra
    // launch on the same in-order queue is ordered after pass 2 by construction.
    if (sumkv) {
        constexpr int CW = 256;
        q.submit([&](sycl::handler & h) {
            local_accessor<int64_t, 1> red(CW, h);
            h.parallel_for(nd_range<1>((size_t)((blk + 7) / 8) * CW, CW), [=](nd_item<1> it) {
                const int row = it.get_group(0) * 8;
                const int gl = (int)it.get_local_id(0);
                int64_t v = 0;
                for (int r = row; r < row + 8 && r < blk; r++) {
                    for (int c = gl; c < kD; c += CW) {
                        v = v * 31 + (int64_t)gk[(size_t)r * kD + c];
                        v = v * 31 + (int64_t)gv[(size_t)r * kD + c];
                    }
                }
                red[gl] = v;
                it.barrier();
                if (gl == 0) {
                    int64_t t = 0;
                    for (int i = 0; i < CW; i++) {
                        t = t * 1000003 + red[i];
                    }
                    sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>
                        ak(sumkv[slot]);
                    ak.fetch_add(t);
                }
            });
        });
    }
}

// Online softmax over one key block with a per-row causal mask, requantize
// P -> u8, rescale acc.  One work-group per query row; the block's keys are
// striped over T threads so all SFUs are busy (the previous one-thread-per-row
// form left the GPU mostly idle and was the prefill bottleneck).
//
// With `blo` the block max and `mb = max(m[i], blo)` the running max,
//   pmax = exp(blo - mb)                     (<= 1)
//   e2   = exp(score - blo)  in (0, 1]       (one exp per key)
//   p8   = round(255 * e2),  ps = pmax / 255
//   sum  = pmax * sum(e2)
// so P is requantized against the block max directly and no second exp pass
// (or pmax scan) is needed.  One block-wide K scale (`bs[0]`).
static void xmx_softmax(queue & q, const int32_t * qk, const float * qsc, const float * bs, float attn_scale, int M,
                        int nk, int ld, int k0, const int32_t * olim, uint8_t * p8, float * ps, float * acc, float * m,
                        float * l) {
    const int T = 256;
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> red(T, h);
        h.parallel_for(nd_range<1>((size_t)M * T, T), [=](nd_item<1> it) {
            const int i = it.get_group(0);
            const int tid = it.get_local_linear_id();
            const float sc = qsc[i] * bs[0] * attn_scale;
            // Visible keys in this block, clamped to [0, nk].
            //
            // The clamp to 0 is load-bearing and was missing.  `olim[i] - k0` goes
            // NEGATIVE for any query row that sits before this block's first key,
            // which happens as soon as there is more than one key block: chunk
            // N's rows all carry positions < the start of chunk N+1's blocks, so
            // every later block re-masks them.  `lim` then reaches the masked
            // loop below as a negative start, `k < nk` is true, and `prow[k] = 0`
            // indexes before the row - row 0 writes in front of the `p8`
            // allocation, and row i writes into row i-1's live P values, which the
            // row i-1 work-group is filling at the same time.  So it is a real
            // out-of-bounds write AND a cross-row race whose winner is not
            // scheduled, which is why it only shows up as history-dependent
            // divergence rather than as a crash.
            //
            // Semantically 0 is also the right value, not just the safe one: a row
            // that sees no key in this block has no visible key in it, so the
            // whole block is masked and pass 1/pass 2 are skipped.
            const int lim = sycl::max(0, sycl::min(nk, olim[i] - k0));
            const int vlim = lim - (lim % 4);            // vector part of the visible range
            const int32_t * qrow = qk + (size_t)i * ld;
            uint8_t * prow = p8 + (size_t)i * ld;
            // pass 1: running-block max, 4 keys per step (int4 load, float4 fmax)
            float lm = -INFINITY;
            for (int k = tid * 4; k + 4 <= lim; k += T * 4) {
                const int4 qv = *reinterpret_cast<const int4 *>(qrow + k);
                const float4 f = sycl::float4((float)qv.x(), (float)qv.y(), (float)qv.z(), (float)qv.w()) * sc;
                lm = sycl::fmax(lm, sycl::fmax(sycl::fmax(f.x(), f.y()), sycl::fmax(f.z(), f.w())));
            }
            if (tid < lim - vlim) { // <= 3 leftover keys
                lm = sycl::fmax(lm, (float)qrow[vlim + tid] * sc);
            }
            red[tid] = lm;
            it.barrier();
            for (int st = T / 2; st > 0; st >>= 1) {
                if (tid < st) {
                    red[tid] = sycl::fmax(red[tid], red[tid + st]);
                }
                it.barrier();
            }
            const float blo = red[0];
            const float mb = sycl::fmax(m[i], blo);
            const float corr = m[i] == -INFINITY ? 0.f : sycl::native::exp(m[i] - mb);
            const float pmax = blo == -INFINITY ? 0.f : sycl::native::exp(blo - mb);
            float s2 = 0.f;
            if (blo != -INFINITY) {
                // pass 2: exp + requantize + sum, 4 keys per step
                for (int k = tid * 4; k + 4 <= lim; k += T * 4) {
                    const int4 qv = *reinterpret_cast<const int4 *>(qrow + k);
                    const float4 f = sycl::float4((float)qv.x(), (float)qv.y(), (float)qv.z(), (float)qv.w()) * sc - blo;
                    const float4 e = sycl::float4(sycl::native::exp(f.x()), sycl::native::exp(f.y()),
                                                  sycl::native::exp(f.z()), sycl::native::exp(f.w()));
                    const uchar4 o = sycl::uchar4((uint8_t)sycl::round(sycl::fmin(255.f, e.x() * 255.f)),
                                                  (uint8_t)sycl::round(sycl::fmin(255.f, e.y() * 255.f)),
                                                  (uint8_t)sycl::round(sycl::fmin(255.f, e.z() * 255.f)),
                                                  (uint8_t)sycl::round(sycl::fmin(255.f, e.w() * 255.f)));
                    *reinterpret_cast<uchar4 *>(prow + k) = o;
                    s2 += e.x() + e.y() + e.z() + e.w();
                }
                if (tid < lim - vlim) {
                    const int k = vlim + tid;
                    const float e2 = sycl::native::exp((float)qrow[k] * sc - blo);
                    prow[k] = (uint8_t)sycl::round(sycl::fmin(255.f, e2 * 255.f));
                    s2 += e2;
                }
            }
            for (int k = lim + tid; k < nk; k += T) { // masked keys -> P = 0
                prow[k] = 0;
            }
            red[tid] = s2;
            it.barrier();
            for (int st = T / 2; st > 0; st >>= 1) {
                if (tid < st) {
                    red[tid] += red[tid + st];
                }
                it.barrier();
            }
            for (int c = tid; c < kD; c += T) {
                acc[(size_t)i * kD + c] *= corr;
            }
            if (tid == 0) {
                ps[i] = (pmax > 0.f ? pmax : 1.f) / 255.f;
                l[i] = l[i] * corr + pmax * red[0];
                m[i] = mb;
            }
        });
    });
}

// acc += ps * bs[1] * pv.  V shares ONE block scale, so the matmul result
// needs a single factor (a per-key V scale is already summed over k).
static void xmx_pv_acc(queue & q, const int32_t * pv, const float * ps, const float * bs, int M, float * acc) {
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>((size_t)M * kD), [=](sycl::id<1> id) {
            const int i = id[0] / kD, c = id[0] % kD;
            acc[(size_t)i * kD + c] += ps[i] * bs[1] * (float)pv[(size_t)i * kD + c];
        });
    });
}

// Write the result: fused -> gate * (acc / l); otherwise split-0 partials plus
// (-inf, 0, 0) for every other split so attn_combine merges them away.
static void xmx_output(queue & q, const float * acc, const float * m, const float * l, const int32_t * orow,
                       const int32_t * oh, const float * gate, float * out, float * partials, int qstride,
                       int n_head, int n_splits, int pstride, int M, bool fuse) {
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>((size_t)M), [=](sycl::id<1> id) {
            const int i = id[0];
            const int row = orow[i], hh = oh[i];
            const float inv = l[i] > 0.f ? 1.f / l[i] : 0.f;
            if (fuse) {
                const float * gb = gate + (size_t)row * qstride + (size_t)hh * 2 * kD + kD;
                float * ob = out + (size_t)row * n_head * kD + (size_t)hh * kD;
                for (int c = 0; c < kD; c++) {
                    ob[c] = acc[(size_t)i * kD + c] * inv * sigmoid_f(gb[c]);
                }
                return;
            }
            for (int s = 0; s < n_splits; s++) {
                float * p = partials + (((size_t)row * n_head + hh) * n_splits + s) * pstride;
                if (s == 0) {
                    p[0] = m[i];
                    p[1] = l[i];
                    for (int c = 0; c < kD; c++) {
                        p[2 + c] = acc[(size_t)i * kD + c];
                    }
                } else {
                    p[0] = -INFINITY;
                    p[1] = 0.f;
                    for (int c = 0; c < kD; c++) {
                        p[2 + c] = 0.f;
                    }
                }
            }
        });
    });
}

// Returns false when the call is out of scope (the caller then uses the
// classic kernel).
bool attn_xmx_launch(queue & q, const float * qbuf, const float * gate, const void * kpool, const void * vpool,
                     float * partials, const int32_t * tables, int n_head, int n_head_kv, int head_dim, int n_splits,
                     const step_info * info, float scale, int max_blocks, int n_rows, int n_real, float * out,
                     const void * kscales, const void * vscales) {
    if (!attn_xmx_enabled(q) || head_dim != kD || kpool == nullptr || kscales == nullptr || vscales == nullptr) {
        return false;
    }
    dnnl_gemm * D = dnnl_for_queue(q);
    if (!D || kv_k_dtype() != kv_dtype_t::i8 || kv_v_dtype() != kv_dtype_t::i8 || n_head_kv <= 0
        || n_head % n_head_kv != 0) {
        return false;
    }
    const int HPG = n_head / n_head_kv;
    const bool fuse = out != nullptr && n_splits == 1;
    const int qstride = n_head * 2 * head_dim;
    const int pstride = 2 + head_dim;
    const sycl::half * ksc = (const sycl::half *)kscales;
    const sycl::half * vsc = (const sycl::half *)vscales;
    const auto * kp_base = (const int8_t *)kpool;
    const auto * vp_base = (const int8_t *)vpool;

    // the whole key range needed by any active token; the K/V gather is shared
    // by every query row, masked per row in the softmax
    int max_nkv = 0;
    for (int r = 0; r < n_rows; r++) {
        if (info->active[r]) {
            max_nkv = std::max(max_nkv, info->pos[r] + row_nr(info, r));
        }
    }
    if (max_nkv <= 0) {
        return true; // nothing to do
    }
    if (max_nkv < xmx_min_keys(q)) {
        return false; // classic is faster at short context
    }
    // every stacked query row is matched against the SAME paged K/V, so the
    // active rows must share one page table (mode 1 / mode 2 both use a single
    // slot for the whole chunk; decode's per-row slots never reach here)
    for (int r = 0; r < n_rows; r++) {
        if (info->active[r] && info->slot[r] != info->slot[0]) {
            return false;
        }
    }
    const int32_t * table = tables + (size_t)info->slot[0] * max_blocks;
    xmx_bufs & b = xmx_get(q, HPG * kMaxQ, max_nkv);

    // total stacked query rows (every kv head enumerates the same tokens)
    int Mtot = 0;
    for (int r = 0; r < n_rows; r++) {
        if (info->active[r]) {
            Mtot += row_nr(info, r) * HPG;
        }
    }
    if (Mtot == 0) {
        return true;
    }

    // PF_XMX_TIME=1: wall-clock the whole call (the queue is in-order, so one
    // wait at the end covers everything submitted here).  This is the only way
    // to attribute the *mode-2* 512-token prefill attention, because the
    // engine's PF_PROF profiler requires PF_NOGRAPH, which forces the mode-1
    // 32-token chunk path instead.
    const bool tm = si::env::flag("PF_XMX_TIME");
    // PF_XMX_BREAKDOWN=1: per-stage wall clock inside the block loop.  Every
    // stage is measured with a q.wait() on the in-order queue, so this
    // serialises and inflates the absolute numbers -- it is for *attribution*
    // (which stage owns the time), not for timing.
    const bool brk = si::env::flag("PF_XMX_BREAKDOWN");
    double b_gather = 0, b_qk = 0, b_sm = 0, b_pv = 0;
    long b_cols = 0, b_blocks = 0;
    std::chrono::steady_clock::time_point t_gather, t_qk, t_sm, t_pv;
    std::chrono::steady_clock::time_point t0;
    if (tm || brk) {
        q.wait();
        t0 = std::chrono::steady_clock::now();
    }
    for (int kvh = 0; kvh < n_head_kv; kvh++) {
        const int M = Mtot;
        if (si::env::flag("PF_XMX_DBG") && kvh == 0) {
            // HPG / n_rows / sum(row_nr) are printed because they decide whether
            // `M` can exceed the scratch capacity: m_cap = HPG * kMaxT * kMaxB and
            // Mtot = HPG * sum(row_nr), so the ONLY way M overflows is
            // sum(row_nr) > kMaxT*kMaxB, which is either a batch bigger than the
            // documented bound or n_rows past kMaxB (and then step_info's own
            // [kMaxB] row arrays overflow too).  Cheap to make visible.
            {
                int srow = 0;
                for (int r = 0; r < n_rows; r++) {
                    if (info->active[r]) {
                        srow += row_nr(info, r);
                    }
                }
                fprintf(stderr, "[xmx] HPG=%d n_rows=%d sum_row_nr=%d cap=%d M=%d ", HPG, n_rows, srow,
                        HPG * kMaxQ, M);
            }
            fprintf(stderr, "max_nkv=%d nblk@%d=", max_nkv, kBlk);
            for (int k0 = 0; k0 < max_nkv;) {
                const int blk = (max_nkv - k0 > kBlk) ? kBlk : xmx_tail_blk(max_nkv - k0);
                fprintf(stderr, "%d", blk);
                if (blk < kBlk) fprintf(stderr, "(%d)", std::min(blk, max_nkv - k0));
                k0 += blk;
            }
            fprintf(stderr, " total_cols=%d\n", max_nkv);
        }
        // Build the per-row (qbuf row, query head, causal limit) on the device
        // from `info`.  Three host->device q.memcpy per kv head used to do this;
        // a host-pointer copy on the in-order queue serialised the host against
        // the GPU (and raced with the reused host vector).
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>((size_t)M), [=](sycl::id<1> id) {
                int idx = (int)id[0];
                int rr = -1, tt = 0, jj = 0;
                for (int r = 0; r < n_rows; r++) {
                    if (!info->active[r]) {
                        continue;
                    }
                    const int cnt = row_nr(info, r) * HPG;
                    if (idx < cnt) {
                        tt = idx / HPG;
                        jj = idx % HPG;
                        rr = r;
                        break;
                    }
                    idx -= cnt;
                }
                if (rr < 0) {
                    b.orow[id[0]] = 0;
                    b.oh[id[0]] = 0;
                    b.olim[id[0]] = 0;
                    return;
                }
                b.orow[id[0]] = rr * info->tpb + tt;
                b.oh[id[0]] = kvh * HPG + jj;
                b.olim[id[0]] = info->pos[rr] + tt + 1;
            });
        });
        xmx_quant_q(q, qbuf, qstride, b.orow, b.oh, M, b.q8, b.qsc, b.m, b.l, b.acc);
        for (int k0 = 0; k0 < max_nkv;) {
            // Full blocks run at kBlk; the last one shrinks to the smallest
            // power-of-two width that still covers the remainder, so the tail
            // does not compute (and does not pay DRAM traffic for) thousands of
            // zero columns.  The width is always a compile-time constant, so the
            // oneDNN primitives are a small cached set.
            const int blk = (max_nkv - k0 > kBlk) ? kBlk : xmx_tail_blk(max_nkv - k0);
            const int nk = std::min(blk, max_nkv - k0);
            const bool bd = brk;
            if (bd) {
                q.wait();
                t_gather = std::chrono::steady_clock::now();
            }
            const int xsum_slot = g_xmx_sum.slot(q);
            xmx_gather(q, kp_base, vp_base, ksc, vsc, table, n_head_kv, kvh, k0, nk, blk, b.k, b.v, b.bs, xsum_slot,
                       xsum_slot >= 0 ? g_xmx_sum.sumkv : nullptr, xsum_slot >= 0 ? g_xmx_sum.sumbs : nullptr);
            if (bd) {
                q.wait();
                t_qk = std::chrono::steady_clock::now();
            }
            if (!D->attn_qk(M, blk, b.q8, b.k, b.qk)) {
                return false;
            }
            if (bd) {
                q.wait();
                t_sm = std::chrono::steady_clock::now();
            }
            xmx_softmax(q, b.qk, b.qsc, b.bs, scale, M, nk, blk, k0, b.olim, b.p, b.ps, b.acc, b.m, b.l);
            if (bd) {
                q.wait();
                t_pv = std::chrono::steady_clock::now();
            }
            if (!D->attn_pv(M, blk, b.p, b.v, b.pv)) {
                return false;
            }
            xmx_pv_acc(q, b.pv, b.ps, b.bs, M, b.acc);
            if (bd) {
                q.wait();
                const auto t_end = std::chrono::steady_clock::now();
                auto ms = [&](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point c) {
                    return std::chrono::duration<double, std::milli>(c - a).count();
                };
                b_gather += ms(t_gather, t_qk);
                b_qk += ms(t_qk, t_sm);
                b_sm += ms(t_sm, t_pv);
                b_pv += ms(t_pv, t_end);
                b_cols += blk;
                b_blocks++;
            }
            k0 += blk;
        }
        xmx_output(q, b.acc, b.m, b.l, b.orow, b.oh, gate, out, partials, qstride, n_head, n_splits, pstride, M,
                   fuse);
    }
    if (tm || brk) {
        q.wait();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        static double acc_ms = 0;
        static long acc_n = 0;
        static double g_qk = 0, g_sm = 0, g_pv = 0, g_ga = 0;
        static long g_cols = 0, g_blk = 0, g_calls = 0;
        acc_ms += ms;
        acc_n++;
        g_qk += b_qk;
        g_sm += b_sm;
        g_pv += b_pv;
        g_ga += b_gather;
        g_cols += b_cols;
        g_blk += b_blocks;
        g_calls++;
        if ((acc_n % 64) == 0) {
            const double d = (double)g_calls;
            // `blks/call` is the fixed-width kBlk chunk count, which is what makes
            // the matmul cost a step function of the key length rather than linear
            // (primitive creation is ~15 ms per new shape, so the width is fixed
            // and padded instead) - the number that explains a surprising total.
            fprintf(stderr,
                    "[xmx] %ld calls %.2f ms total %.3f/call | gather %.3f  qk %.3f  softmax %.3f  pv %.3f "
                    "(ms/call) cols/call %.0f  blks/call %.1f\n",
                    acc_n, acc_ms, acc_ms / acc_n, g_ga / d, g_qk / d, g_sm / d, g_pv / d, (double)g_cols / d,
                    (double)g_blk / d);
        }
    }
    return true;
}

} // namespace si
