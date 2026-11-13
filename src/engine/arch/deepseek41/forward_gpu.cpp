// DeepSeek-V4.1-Flash GPU forward (`deepseek41`), the device counterpart of the
// validated host forward in arch/deepseek41/forward_host.cpp.  See docs/design/23-deepseek41.md.
//
// What runs where:
//   * device: the token embedding expand, every dense GEMV (q LoRA, kv, the 8
//     wo_a groups + wo_b, the shared expert, the router's logits, the head),
//     RMSNorm, the mHC mixer, MLA attention, the compressor, the indexer score,
//     the fp8/fp4 activations, and the routed expert GEMVs (IQ2 raw or any
//     GGUF type the generic fp32 GEMV reads).
//   * host: the Engram n-gram hash and its 24-row IQ4_NL gather (the table is
//     ~100 GiB and is never uploaded), the 384-wide router top-6 (a D2H of 384
//     floats), the DSA top-k / candidate blocks (a D2H of <= n_comp floats), the
//     expert LRU, the sampler, and the expert H2D copies.
//
// The layer split is the engine's `--layer-map`; every layer's window ring and
// each KV source's compressed cache live on the device that owns the layer.
// With `0-19:gpu.0,20-39:gpu.1` the compressed-cache dependency falls exactly
// on the partition boundary (layers 21..39 read layer 20's cache on gpu.1), so
// only the 4-stream hidden state and the lagged `pre` mix cross the boundary.
//
// Environment:
//   PF_DS41_GPU=0      force the host forward
//   PF_DS41_BANK=0     disable the routed-expert bank (one staging slot/layer)
//   PF_DS41_TIME=1     per-phase decode timing (adds a wait per phase)
//   PF_DS41_STATS=1    routing / upload counts
//   PF_DS41_NOQ=1      disable the fp8/fp4 activation roundings (debug)
//   PF_DS41_ENG_BF16=1 round the Engram gather to bf16 (torch path A/B)
//   PF_DS41_CHECK=1    compare the first token's logits against the host forward
//   PF_DS41_SPLIT=1    split each layer's routed experts across BOTH cards so
//                      the two PCIe links upload concurrently (default off;
//                      A/B against the unsplit path).  PF_DS41_SPLIT_RATIO is
//                      the share of expert ids on card 0 (default 0.34 = the
//                      measured 5.18:9.98 GB/s link ratio);
//                      PF_DS41_BANK0_MB / PF_DS41_BANK1_MB override the two
//                      cards' bank budgets (defaults split --moe-vram-mb by the
//                      same ratio).  PF_DS41_ABL=nomoe|noattn|nohead is a
//                      wrong-output timing ablation.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <sycl/sycl.hpp>

#include "ds41_attn.h"
#include "ds41_hc.h"
#include "ds41_iq2.h"
#include "ds41_iq2_tables.h"
#include "dnnl_gemm.h"
#include "engine.h"
#include "arch/deepseek41/forward.h"
#include "arch/deepseek41/accessors.h"
#include "arch/deepseek41/hparams.h"
#include "quant.h"
#include "sampler.h"
#include "common/env.h"

namespace si {

using namespace sycl;

namespace {

// ===========================================================================
// Scalar helpers shared with the host reference (arch/deepseek41/forward_host.cpp).
// ===========================================================================
inline float sigmoidg(float x) { return 1.f / (1.f + sycl::exp(-x)); }
inline float silug(float x) { return x / (1.f + sycl::exp(-x)); }
inline float softplusg(float x) { return x > 20.f ? x : sycl::log1p(sycl::exp(x)); }

inline float pow2g(int e) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = (uint32_t)(e + 127) << 23;
    return v.f;
}

// E4M3FN nearest representable (round to nearest even), same bits trick as the
// host reference and ds41_attn.cpp.
inline float e4m3g(float v) {
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
    const uint32_t u = bits.u;
    const int E = (int)((u >> 23) & 0xffu) - 127;
    const float step = (E < -6) ? pow2g(-9) : pow2g(E - 3);
    float out = sycl::rint(a / step) * step;
    if (out > 448.f) {
        out = 448.f;
    }
    return neg ? -out : out;
}

// fp8_round_row: one power-of-two scale per 32 values.  The exponent is taken
// from the float bits (exact) rather than via log2/exp2, which measured ~3.7%
// off on this device in the ds41_attn probe.
void fp8_round_launch(queue & q, float * v, int n) {
    if (!v || n <= 0) {
        return;
    }
    q.parallel_for(range<1>((size_t)((n + 31) / 32)), [=](id<1> ib) {
        const int b = (int)ib[0] * 32;
        const int m = sycl::min(32, n - b);
        float amax = 1e-4f;
        for (int i = 0; i < m; i++) {
            amax = sycl::fmax(amax, sycl::fabs(v[b + i]));
        }
        const float quot = amax / 448.f;
        union {
            float f;
            uint32_t u;
        } bits;
        bits.f = quot;
        const uint32_t u = bits.u;
        const int E = (int)((u >> 23) & 0xffu) - 127;
        const bool exact = (u & 0x007fffffu) == 0;
        const float s = pow2g(exact ? E : E + 1); // 2^ceil(log2 quot)
        for (int i = 0; i < m; i++) {
            float xv = v[b + i] / s;
            xv = sycl::fmax(-448.f, sycl::fmin(448.f, xv));
            v[b + i] = e4m3g(xv) * s;
        }
    });
}

void swiglu_clamp_launch(queue & q, const float * g, const float * u, float * out, int n, float lim) {
    q.parallel_for(range<1>((size_t)n), [=](id<1> i) {
        const float gv = sycl::fmin(g[i[0]], lim);
        const float uv = sycl::fmax(-lim, sycl::fmin(lim, u[i[0]]));
        out[i[0]] = silug(gv) * uv;
    });
}

void yadd_launch(queue & q, float * y, const float * acc, float w, int n) {
    q.parallel_for(range<1>((size_t)n), [=](id<1> i) { y[i[0]] += w * acc[i[0]]; });
}

void window_write_launch(queue & q, const float * row, float * win, int * win_pos, int pos, int cap, int d) {
    const int slot = ((pos % cap) + cap) % cap;
    q.parallel_for(range<1>((size_t)(d + 1)), [=](id<1> i) {
        const int e = (int)i[0];
        if (e < d) {
            win[(size_t)slot * (size_t)d + (size_t)e] = row[e];
        } else {
            win_pos[slot] = pos;
        }
    });
}

void gather_rows_launch(queue & q, const float * ckv, const int32_t * sel, int n_sel, int d, float * out) {
    if (n_sel <= 0) {
        return;
    }
    q.parallel_for(range<2>((size_t)n_sel, (size_t)d), [=](id<2> i) {
        out[(size_t)i[0] * (size_t)d + (size_t)i[1]] =
            ckv[(size_t)sel[i[0]] * (size_t)d + (size_t)i[1]];
    });
}

// Engram apply: dot_s over one stream's d values, then h += gate*value.
void engram_dot_launch(queue & q, const float * h, const float * kv, const float * gate, int hc, int d,
                       float * gbuf) {
    constexpr int WG = 256;
    q.submit([&](handler & hd) {
        local_accessor<float, 1> red(3 * WG, hd);
        hd.parallel_for(nd_range<1>((size_t)hc * WG, WG), [=](nd_item<1> it) {
            const int k = (int)it.get_group(0);
            const int tid = (int)it.get_local_id(0);
            const float * hk = h + (size_t)k * (size_t)d;
            const float * kk = kv + (size_t)k * (size_t)d;
            const float * wg = gate + (size_t)k * (size_t)d;
            float sh = 0.f, sk = 0.f, sd = 0.f;
            for (int i = tid; i < d; i += WG) {
                const float hv = hk[i];
                const float kvv = kk[i];
                sh = sycl::fma(hv, hv, sh);
                sk = sycl::fma(kvv, kvv, sk);
                sd = sycl::fma(hv * wg[i], kvv, sd);
            }
            red[tid] = sh;
            red[WG + tid] = sk;
            red[2 * WG + tid] = sd;
            sycl::group_barrier(it.get_group());
            for (int s = WG / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    red[tid] += red[tid + s];
                    red[WG + tid] += red[WG + tid + s];
                    red[2 * WG + tid] += red[2 * WG + tid + s];
                }
                sycl::group_barrier(it.get_group());
            }
            if (tid == 0) {
                const float rh = sycl::rsqrt(red[0] / (float)d + 1e-20f);
                const float rk = sycl::rsqrt(red[WG] / (float)d + 1e-20f);
                const float dot = red[2 * WG] * rh * rk / sycl::sqrt((float)d);
                gbuf[k] = sigmoidg(sycl::copysign(sycl::sqrt(sycl::fmax(sycl::fabs(dot), 1e-6f)), dot));
            }
        });
    });
}

void engram_add_launch(queue & q, float * h, const float * kv, const float * gbuf, int hc, int d) {
    const float * value = kv + (size_t)hc * (size_t)d;
    q.parallel_for(range<2>((size_t)hc, (size_t)d), [=](id<2> i) {
        const int k = (int)i[0];
        h[(size_t)k * (size_t)d + (size_t)i[1]] += gbuf[k] * value[i[1]];
    });
}

// bf16 round (round-to-nearest-even on the 16-bit boundary), same as the host.
inline float bf16g(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    u &= 0xFFFF0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

// ---------------------------------------------------------------------------
// YaRN/window frequency tables, exactly the host's build_freqs.
// ---------------------------------------------------------------------------
void build_freqs(float * fr, int rd, float base, int orig, float factor, float bf, float bs) {
    for (int j = 0; j < rd / 2; j++) {
        fr[j] = 1.f / std::pow(base, 2.f * (float)j / (float)rd);
    }
    if (orig > 0) {
        auto corr = [&](float rot) {
            return (float)rd * std::log((float)orig / (rot * 2.f * (float)M_PI)) /
                   (2.f * std::log(base));
        };
        const int low = std::max((int)std::floor(corr(bf)), 0);
        const int high = std::min((int)std::ceil(corr(bs)), rd - 1);
        for (int j = 0; j < rd / 2; j++) {
            float ramp = ((float)j - (float)low) / std::max((float)(high - low), 1e-3f);
            ramp = std::min(1.f, std::max(0.f, ramp));
            const float smooth = 1.f - ramp;
            fr[j] = fr[j] / factor * (1.f - smooth) + fr[j] * smooth;
        }
    }
}

} // namespace

// ===========================================================================
// Expert staging copy crew
// ===========================================================================
// A bank miss copies the expert's three planes from the (page-cache-backed)
// mmap into pinned staging before the H2D.  That memcpy is one host thread and
// ~0.9 ms per miss; once the split makes both PCIe links concurrent the copies
// are the token's critical path (measured: 72 ms/token unsplit, 119 ms split).
// Four workers splitting each plane in half take the 9.2 MB copy to ~0.3 ms.
// A fixed crew with an atomic job cursor, one generation at a time.
struct ds41_copy_crew {
    struct job {
        const uint8_t * s;
        uint8_t * d;
        size_t n;
    };
    // Three helpers plus the calling thread.  The generation handshake is
    // spin-waited: a condvar wake for a 0.3 ms copy cost ~0.5 ms per miss, which
    // was most of the staging time it was meant to hide.
    static constexpr int kW = 3;
    job jobs[16];
    std::atomic<int> njobs{0}, next{0}, done{0}, gen{0};
    std::atomic<bool> quit{false};
    std::thread th[kW];

    ds41_copy_crew() {
        for (int i = 0; i < kW; i++) {
            th[i] = std::thread([this] { worker(); });
        }
    }
    ~ds41_copy_crew() {
        quit.store(true, std::memory_order_release);
        for (auto & t : th) {
            t.join();
        }
    }
    void copy_range(int k) {
        std::memcpy(jobs[k].d, jobs[k].s, jobs[k].n);
    }
    void worker() {
        int seen = 0;
        for (;;) {
            while (gen.load(std::memory_order_acquire) == seen) {
                if (quit.load(std::memory_order_acquire)) {
                    return;
                }
                std::this_thread::yield();
            }
            seen = gen.load(std::memory_order_acquire);
            const int n = njobs.load(std::memory_order_relaxed);
            for (;;) {
                const int k = next.fetch_add(1, std::memory_order_relaxed);
                if (k >= n) {
                    break;
                }
                copy_range(k);
            }
            done.fetch_add(1, std::memory_order_release);
        }
    }
    void run(const job * js, int n) {
        for (int i = 0; i < n; i++) {
            jobs[i] = js[i];
        }
        njobs.store(n, std::memory_order_relaxed);
        next.store(0, std::memory_order_relaxed);
        done.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_release);
        // the caller copies too
        for (;;) {
            const int k = next.fetch_add(1, std::memory_order_relaxed);
            if (k >= n) {
                break;
            }
            copy_range(k);
        }
        while (done.load(std::memory_order_acquire) < kW) {
            std::this_thread::yield();
        }
    }
};

// ===========================================================================
// State
// ===========================================================================
struct ds41_gpu_state {
    engine & e;
    model & m;
    const hparams & hp;
    const ds41_hparams & d;
    int ndev = 0;
    int n_layer = 0, dim = 0, hc = 0, kv_dim = 0, W = 0, idx_dim = 0;
    int q_lora = 0, H = 0, D = 0, n_rot = 0, rot_off = 0, E = 0, topk = 0, ff = 0, ffsh = 0;
    int idx_nh = 0, idx_hd = 0, idx_k = 0, out_g = 0, out_rank = 0;
    int max_seq = 0;
    int hc_mix = 0;
    float freq_win[32] = {0}, freq_comp[32] = {0};
    bool noq = false, eng_bf16 = false, stats = false;
    int timing = 0;
    bool moe_vl = false; // the current token is an image row: VL router bias
    bool dbg = false;
    // Wrong-output-only timing ablations (PF_DS41_ABL=nomoe|noattn|nohead):
    // they attribute the per-token cost by removing a phase, exactly as the
    // measured-results notes prescribe; never use them for correctness.
    bool abl_nomoe = false, abl_noattn = false, abl_nohead = false, abl_nostage = false;
    int bank_budget_mb = 0;

    // ---- per-device scratch ------------------------------------------------
    struct devbuf {
        int idx = 0;
        sycl::queue * q = nullptr;
        std::unique_ptr<sycl::queue> cq; // separate in-order copy queue (H2D overlap)
        float *h = nullptr, *res = nullptr, *mixx = nullptr, *x = nullptr, *y = nullptr, *acc = nullptr;
        float *qr = nullptr, *qb = nullptr, *ob = nullptr, *kvrow = nullptr, *latent = nullptr;
        float *og = nullptr, *xn = nullptr, *g = nullptr, *u = nullptr, *hh = nullptr, *sh = nullptr;
        float *score = nullptr, *qi = nullptr, *iw = nullptr, *ikproj = nullptr, *rowo = nullptr;
        float *compact = nullptr;
        int32_t *sel = nullptr;
        float *logits = nullptr;
        float *a_pre = nullptr, *a_post = nullptr, *a_comb = nullptr;
        float *f_pre = nullptr, *f_post = nullptr, *f_comb = nullptr;
        float *pre_prev = nullptr, *mix_mixes = nullptr, *mix_partials = nullptr;
        float *fwin = nullptr, *fcomp = nullptr;
        int32_t *idummy = nullptr;
        // Engram stage
        float *eng_ev = nullptr, *eng_kv = nullptr, *eng_g = nullptr;
        // expert split: this card's MoE input, its per-j expert outputs, and
        // the peer's copied-in outputs (indexed [j*dim]).
        float *px = nullptr, *pacc = nullptr, *premote = nullptr;
        // gemv_seg ring (host staging + device mirror)
        gemv_seg *hsegs = nullptr, *dsegs = nullptr;
        int seg_cap = 0, seg_used = 0;
        ds41_iq2_tables *tab = nullptr;
        size_t maxclen = 0;
        // pinned expert-upload staging ring (8 planes, non-blocking H2D): one
        // layer selects up to `topk` (6) experts, so 4 slots could stall the
        // ring on the last two.
        uint8_t * estage = nullptr;
        size_t estage_cap = 0;
        int estage_next = 0;
        sycl::event estage_ev[8];
        bool estage_pending[8] = {false, false, false, false, false, false, false, false};
    };
    std::vector<devbuf> dv;

    // ---- per-layer window rings (allocated on the owning device) ----------
    std::vector<float *> win;
    std::vector<int32_t *> winpos;

    // ---- group-shared compressed caches ------------------------------------
    struct src {
        int dev = -1;
        int ratio = 0;
        size_t maxclen = 0;
        float *ckv = nullptr, *ik = nullptr, *rkv = nullptr, *rsc = nullptr;
        int clen = 0, ring_n = 0, ik_rows = 0;
    };
    std::vector<src> sources;
    std::vector<int> kv_slot_of; // [n_layer] -> source index or -1

    // ---- routed-expert bank (per layer, on the owning device) -------------
    struct bank {
        int dev = -1;
        uint32_t gt = 0, ut = 0, dt = 0;
        size_t pg = 0, pu = 0, pd = 0;
        int R = 0;
        void *g = nullptr, *u = nullptr, *d = nullptr;
        std::vector<int32_t> e_slot; // [n_expert] -> slot or -1
        std::vector<int32_t> slot_e; // [R] -> expert or -1
        std::vector<uint32_t> age;   // [R]
        uint32_t tick = 0;
        bool iq2_g = false, iq2_u = false, iq2_d = false;
        size_t uploads = 0; // planes
        uint64_t hits = 0, misses = 0; // bank statistics
        size_t miss_bytes = 0;
    };
    // One bank per (layer, card): banks[li*ndev + c].  Without the split only
    // the layer's owning card has one allocated; with PF_DS41_SPLIT every card
    // holds the experts whose id it owns (ids < split_c0 on card 0), so both
    // links upload concurrently.
    std::vector<bank> banks;
    bool split_exp = false;
    int split_c0 = 0;
    int expert_card(int e) const { return split_exp && e >= split_c0 ? 1 : 0; }
    bank & bank_of(int li, int c) { return banks[(size_t)li * (size_t)ndev + (size_t)c]; }
    // routing frequency from the prompt, used to calibrate the resident set
    std::vector<std::vector<uint32_t>> route_freq;
    bool calib = false;
    void calibrate();

    // ---- per-sequence state (one sequence; the profile caps rows at 1) -----
    int pos = 0;
    int64_t hist[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int hist_n = 0;
    int pad = 0;
    std::vector<uint8_t> cand;
    int cand_nb = 0;
    bool has_cand = false;
    int sel_k = 0;
    int sel_dev = -1;

    // small host-side copies of tensors the host reads
    std::vector<float> router_bias, router_bias_vl;
    const float * router_bias_p = nullptr, *router_bias_vl_p = nullptr;

    // host staging
    float *hstage = nullptr, *pstage = nullptr, *sstage = nullptr;
    float *xp_stage = nullptr, *pacc_host = nullptr;
    std::vector<float> logits_host;
    std::vector<float> h0;

    // timing
    struct times {
        double dense = 0, attn = 0, moe = 0, head = 0, emb = 0, total = 0;
        uint64_t tok = 0;
    } tm;
    times tm_prof; // PF_DS41_TIME=2 same-stream phase split (adds per-phase waits)

    std::unique_ptr<ds41_copy_crew> crew;
    std::vector<std::pair<void *, int>> owned;
    std::vector<void *> host_owned;

    ds41_gpu_state(engine & en);
    ~ds41_gpu_state();

    // ---- helpers ----------------------------------------------------------
    sycl::queue & Q(int dev) const { return *dv[(size_t)dev].q; }
    devbuf & DV(int dev) { return dv[(size_t)dev]; }
    const devbuf & DV(int dev) const { return dv[(size_t)dev]; }
    const void * wp(int dev, const void * host) const { return e.wptr(dev, host); }
    // A dense GEMV's key: the uploaded device pointer, or the host pointer when
    // the tensor is registered as a oneDNN store (the upload then skips its raw
    // copy).  Both are what gemv1 resolves the store by, and the raw fallback
    // needs the device pointer - so this is wptr() with wkey()'s exception.
    const void * wk(int dev, const void * host) const { return e.wkey(dev, host); }
    int dev_of_layer(int li) const { return e.layer_dev_[(size_t)li]; }

    template <class T>
    T * alloc_dev(int dev, size_t n, const char * what) {
        if (n == 0) {
            return nullptr; // a model with no Engram layers allocates no row stage
        }
        T * p = sycl::malloc_device<T>(n, Q(dev));
        if (!p) {
            throw std::runtime_error(std::string("ds41 GPU: device allocation failed (") + what +
                                     ", " + std::to_string((unsigned long long)(n * sizeof(T))) + " bytes)");
        }
        owned.emplace_back((void *)p, dev);
        return p;
    }
    void * alloc_host(size_t n, const char * what) {
        void * p = sycl::malloc_host(n, Q(0));
        if (!p) {
            throw std::runtime_error(std::string("ds41 GPU: host allocation failed (") + what + ")");
        }
        host_owned.push_back(p);
        return p;
    }

    void reset_state();

    // dense GEMV (raw GGUF type) and group form
    void gemv_calls(int dev, uint32_t type, std::vector<gemv_seg> & segs);
    void gemv1(int dev, const void * w, uint32_t type, int K, int N, const float * x, float * out,
               const float * residual = nullptr, float alpha = 1.f, int out_stride = 0);
    void rms(int dev, const float * x, const void * w, float * out, int n);
    void rms_rows(int dev, const float * x, const void * w, float * out, int rows, int n);

    // forward pieces
    void forward_token(int tok, bool want_logits, const mm_prompt * mm);
    void attention(int li, int dev, const float * x, float * o);
    void push_kvsrc(int li, int dev, const float * x);
    void append_src(int li, int sidx, const float * latent, int block);
    void index_select(int li, int dev, const float * x, const float * qr);
    void moe(int li, int dev, const float * x, float * y);
    int bank_slot(int li, int iexp, int card = -1);
    void engram_apply(int li, int dev);

    void handoff(int from, int to);
    void head(int dev);

    void route_one(int li, const float * x, int dev, int32_t * ids, float * wts);
};

ds41_gpu_state::ds41_gpu_state(engine & en)
    : e(en), m(en.m), hp(en.m.hp), d(ds41_hp(en.m.hp)) {
    noq = si::env::flag("PF_DS41_NOQ");
    eng_bf16 = si::env::flag("PF_DS41_ENG_BF16");
    stats = si::env::flag("PF_DS41_STATS");
    timing = si::env::i32("PF_DS41_TIME", 0);
    dbg = si::env::flag("PF_DS41_DEBUG");
    if (const char * ab = si::env::str("PF_DS41_ABL")) {
        abl_nomoe = std::strstr(ab, "nomoe") != nullptr;
        abl_noattn = std::strstr(ab, "noattn") != nullptr;
        abl_nohead = std::strstr(ab, "nohead") != nullptr;
        abl_nostage = std::strstr(ab, "nostage") != nullptr;
    }
    n_layer = hp.n_layer;
    dim = hp.n_embd;
    hc = d.hc_mult;
    kv_dim = d.kv_dim;
    W = d.sliding_window;
    idx_dim = d.idx_head_dim;
    q_lora = d.q_lora_rank;
    H = hp.n_head;
    D = d.kv_dim;
    n_rot = hp.n_rot;
    rot_off = d.rot_off;
    E = hp.n_expert;
    topk = hp.n_expert_used;
    ff = hp.n_ff_exp > 0 ? hp.n_ff_exp : hp.n_ff;
    ffsh = hp.n_ff_shexp > 0 ? hp.n_ff_shexp : hp.n_ff;
    idx_nh = d.idx_n_head;
    idx_hd = d.idx_head_dim;
    idx_k = d.idx_top_k;
    out_g = d.out_group_count;
    out_rank = d.out_lora_rank;
    max_seq = std::max(1, en.max_seq);
    hc_mix = (2 + hc) * hc;
    ndev = (int)en.dev_kind_.size();
    split_exp = ndev >= 2 && si::env::i32("PF_DS41_SPLIT", 1) != 0;
    {
        float ratio = 0.34f;
        if (const char * sr = si::env::str("PF_DS41_SPLIT_RATIO")) {
            ratio = (float)atof(sr);
        }
        ratio = std::min(0.9f, std::max(0.1f, ratio));
        split_c0 = split_exp ? std::max(1, std::min(E - 1, (int)std::lround((double)ratio * (double)E))) : 0;
    }

    // Routed-expert bank budget: --moe-vram-mb (the profile is runtime_bank, so
    // the generic cache is inactive and this config field is free for ds41).
    bank_budget_mb = en.moe_cfg_.vram_mb;
    if (const char * eb = si::env::str("PF_DS41_BANK_MB")) {
        bank_budget_mb = atoi(eb);
    }
    const char * eb0 = si::env::str("PF_DS41_BANK");
    if (eb0 && atoi(eb0) == 0) {
        bank_budget_mb = 0;
    }

    build_freqs(freq_win, n_rot, hp.rope_base, 0, 1.f, 0.f, 0.f);
    const float factor = d.yarn_factor > 0.f ? d.yarn_factor : 16.f;
    const int orig = d.yarn_orig > 0 ? d.yarn_orig : 65536;
    const float bf = d.yarn_beta_fast > 0.f ? d.yarn_beta_fast : 32.f;
    const float bs = d.yarn_beta_slow > 0.f ? d.yarn_beta_slow : 1.f;
    build_freqs(freq_comp, n_rot, d.compress_rope_base > 0.f ? d.compress_rope_base : 160000.f, orig,
                factor, bf, bs);

    pad = ds41_model(m).engram_token_map.empty() ? 0 : ds41_model(m).engram_token_map[(size_t)d.engram_pad];

    // ---- per-device scratch ----------------------------------------------
    dv.resize((size_t)ndev);
    for (int i = 0; i < ndev; i++) {
        devbuf & t = dv[(size_t)i];
        t.idx = i;
        t.q = &e.dev_queue(i);
        try {
            t.cq = std::make_unique<sycl::queue>(t.q->get_context(), t.q->get_device(),
                                                 sycl::property::queue::in_order());
        } catch (...) {
            t.cq.reset();
        }
        const size_t hc_d = (size_t)hc * (size_t)dim;
        t.h = alloc_dev<float>(i, hc_d, "h");
        t.res = alloc_dev<float>(i, hc_d, "res");
        t.mixx = alloc_dev<float>(i, (size_t)dim, "mixx");
        t.x = alloc_dev<float>(i, (size_t)dim, "x");
        t.y = alloc_dev<float>(i, (size_t)dim, "y");
        t.acc = alloc_dev<float>(i, (size_t)dim, "acc");
        t.qr = alloc_dev<float>(i, (size_t)q_lora, "qr");
        t.qb = alloc_dev<float>(i, (size_t)H * (size_t)D, "q");
        t.ob = alloc_dev<float>(i, (size_t)H * (size_t)D, "o");
        t.kvrow = alloc_dev<float>(i, (size_t)D, "kvrow");
        t.latent = alloc_dev<float>(i, (size_t)D, "latent");
        t.og = alloc_dev<float>(i, (size_t)out_g * (size_t)out_rank, "og");
        t.xn = alloc_dev<float>(i, (size_t)dim, "xn");
        t.g = alloc_dev<float>(i, (size_t)ff, "gate");
        t.u = alloc_dev<float>(i, (size_t)ff, "up");
        t.hh = alloc_dev<float>(i, (size_t)ff, "hh");
        t.sh = alloc_dev<float>(i, (size_t)ffsh, "shared");
        t.qi = alloc_dev<float>(i, (size_t)idx_nh * (size_t)idx_hd, "index q");
        t.iw = alloc_dev<float>(i, (size_t)idx_nh, "index w");
        t.ikproj = alloc_dev<float>(i, (size_t)idx_hd, "index k");
        t.rowo = alloc_dev<float>(i, (size_t)D, "row");
        t.compact = alloc_dev<float>(i, (size_t)idx_k * (size_t)D, "compact");
        t.sel = alloc_dev<int32_t>(i, (size_t)idx_k, "sel");
        t.logits = alloc_dev<float>(i, (size_t)hp.n_vocab, "logits");
        t.a_pre = alloc_dev<float>(i, (size_t)hc, "a_pre");
        t.a_post = alloc_dev<float>(i, (size_t)hc, "a_post");
        t.a_comb = alloc_dev<float>(i, (size_t)hc * (size_t)hc, "a_comb");
        t.f_pre = alloc_dev<float>(i, (size_t)hc, "f_pre");
        t.f_post = alloc_dev<float>(i, (size_t)hc, "f_post");
        t.f_comb = alloc_dev<float>(i, (size_t)hc * (size_t)hc, "f_comb");
        t.pre_prev = alloc_dev<float>(i, (size_t)hc, "pre_prev");
        t.mix_mixes = alloc_dev<float>(i, (size_t)hc_mix, "mixes");
        t.mix_partials = alloc_dev<float>(i, (size_t)ds41_hc_mix_partials(hc), "mix partials");
        t.fwin = alloc_dev<float>(i, 32, "freq win");
        t.fcomp = alloc_dev<float>(i, 32, "freq comp");
        t.eng_ev = alloc_dev<float>(i, (size_t)d.ple_n_heads * (size_t)d.ple_head_dim, "engram rows");
        t.eng_kv = alloc_dev<float>(i, (size_t)(hc + 1) * (size_t)dim, "engram kv");
        t.eng_g = alloc_dev<float>(i, (size_t)hc, "engram gate");
        if (split_exp) {
            t.px = alloc_dev<float>(i, (size_t)dim, "peer moe x");
            t.pacc = alloc_dev<float>(i, (size_t)topk * (size_t)dim, "local expert acc");
            t.premote = alloc_dev<float>(i, (size_t)topk * (size_t)dim, "peer expert acc");
        }
        t.idummy = alloc_dev<int32_t>(i, 8, "scratch");
        t.seg_cap = 512;
        t.hsegs = (gemv_seg *)alloc_host((size_t)t.seg_cap * sizeof(gemv_seg), "seg staging");
        t.dsegs = alloc_dev<gemv_seg>(i, (size_t)t.seg_cap, "seg ring");
        t.maxclen = (size_t)max_seq + 2;
        t.tab = alloc_dev<ds41_iq2_tables>(i, 1, "iq2 tables");

        t.q->memcpy(t.fwin, freq_win, 32 * sizeof(float));
        t.q->memcpy(t.fcomp, freq_comp, 32 * sizeof(float));
        ds41_iq2_tables htab;
        std::memcpy(htab.xxs, ds41_iq2xxs_grid, sizeof(htab.xxs));
        std::memcpy(htab.xs, ds41_iq2xs_grid, sizeof(htab.xs));
        std::memcpy(htab.s, ds41_iq2s_grid, sizeof(htab.s));
        std::memcpy(htab.signs, ds41_ksigns_iq2xs, sizeof(htab.signs));
        t.q->memcpy(t.tab, &htab, sizeof(htab));
    }
    // per-device score buffer (max compressed rows for the largest ratio-2 cache)
    for (int i = 0; i < ndev; i++) {
        dv[(size_t)i].maxclen = (size_t)((max_seq + 1) / 2 + 2);
        dv[(size_t)i].score = alloc_dev<float>(i, dv[(size_t)i].maxclen, "score");
    }
    // host staging (shared USM context across all GPU partitions)
    hstage = (float *)alloc_host((size_t)hc * (size_t)dim * sizeof(float), "h handoff");
    pstage = (float *)alloc_host((size_t)hc * sizeof(float), "pre handoff");
    const size_t sstage_n = (size_t)std::max({(size_t)hp.n_vocab, (size_t)(max_seq + 2), (size_t)q_lora,
                                              (size_t)out_rank, (size_t)H * (size_t)D,
                                              (size_t)(hc + 1) * (size_t)dim, (size_t)idx_nh * (size_t)idx_hd});
    sstage = (float *)alloc_host(sstage_n * sizeof(float), "d2h staging");
    if (split_exp) {
        xp_stage = (float *)alloc_host((size_t)dim * sizeof(float), "peer x staging");
        pacc_host = (float *)alloc_host((size_t)topk * (size_t)dim * sizeof(float), "peer acc staging");
    }
    logits_host.resize((size_t)hp.n_vocab);
    h0.resize((size_t)dim);

    // host copies of the router biases (read every token, and the GGUF pages of
    // these 384-value tensors would otherwise be released after the upload).
    if (ds41_layer_of(m.layers[0]).exp_probs_b) {
        router_bias.assign(ds41_layer_of(m.layers[0]).exp_probs_b, ds41_layer_of(m.layers[0]).exp_probs_b + E);
        router_bias_p = router_bias.data();
    }
    if (ds41_layer_of(m.layers[0]).exp_probs_b_vl) {
        router_bias_vl.assign(ds41_layer_of(m.layers[0]).exp_probs_b_vl, ds41_layer_of(m.layers[0]).exp_probs_b_vl + E);
        router_bias_vl_p = router_bias_vl.data();
    }

    // ---- per-layer window rings ------------------------------------------
    win.assign((size_t)n_layer, nullptr);
    winpos.assign((size_t)n_layer, nullptr);
    for (int li = 0; li < n_layer; li++) {
        const int dev = dev_of_layer(li);
        if (dev < 0 || dev >= ndev || e.dev_kind_[(size_t)dev] != 0) {
            throw std::runtime_error("ds41 GPU: every layer must be on a GPU partition");
        }
        win[(size_t)li] = alloc_dev<float>(dev, (size_t)W * (size_t)D, "window ring");
        winpos[(size_t)li] = alloc_dev<int32_t>(dev, (size_t)W, "window positions");
        Q(dev).memset(win[(size_t)li], 0, (size_t)W * (size_t)D * sizeof(float));
        Q(dev).memset(winpos[(size_t)li], 0xff, (size_t)W * sizeof(int32_t));
    }

    // ---- group-shared compressed caches ----------------------------------
    sources.resize((size_t)d.n_kv_src);
    kv_slot_of.assign((size_t)n_layer, -1);
    for (int s = 0; s < d.n_kv_src; s++) {
        const int src_layer = d.kv_src[s];
        const int dev = dev_of_layer(src_layer);
        src & S = sources[(size_t)s];
        S.dev = dev;
        S.ratio = d.compress_ratio[src_layer];
        if (S.ratio <= 0) {
            throw std::runtime_error("ds41 GPU: a KV source with ratio 0");
        }
        S.maxclen = (size_t)((max_seq + S.ratio - 1) / S.ratio + 2);
        S.ckv = alloc_dev<float>(dev, S.maxclen * (size_t)D, "compressed kv");
        S.ik = alloc_dev<float>(dev, S.maxclen * (size_t)idx_hd, "compressed index k");
        S.rkv = alloc_dev<float>(dev, (size_t)S.ratio * (size_t)D, "compressor kv ring");
        S.rsc = alloc_dev<float>(dev, (size_t)S.ratio * (size_t)D, "compressor gate ring");
    }
    for (int li = 0; li < n_layer; li++) {
        const int src = d.kv_src_of[li];
        if (src < 0) {
            continue;
        }
        for (int s = 0; s < d.n_kv_src; s++) {
            if (d.kv_src[s] == src) {
                kv_slot_of[(size_t)li] = s;
                break;
            }
        }
    }

    // ---- routed-expert bank ----------------------------------------------
    // banks[li*ndev + c]: without the split only the owning card is allocated;
    // with PF_DS41_SPLIT each card holds the experts whose id it owns, so the
    // two PCIe links upload concurrently (the layers stay on their owner).
    banks.assign((size_t)n_layer * (size_t)std::max(1, ndev), bank{});
    route_freq.assign((size_t)n_layer, std::vector<uint32_t>((size_t)E, 0));
    const size_t budget_bytes = (size_t)std::max(0, bank_budget_mb) * 1024 * 1024;
    size_t total_bank_bytes = 0;
    size_t card_budget[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // Reserve what the sequence state costs on each card before the bank is
    // sized: the compressed caches and their index keys, the window rings, and
    // 256 MiB of activation/scratch/logits margin.  Without this the default
    // split fits at ctx 32k by luck and fails at 64k (measured DEVICE_LOST).
    size_t card_reserve[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (const src & S : sources) {
        if (S.dev < 0 || S.dev >= ndev) {
            continue;
        }
        card_reserve[S.dev] += S.maxclen * ((size_t)D + (size_t)idx_hd) * sizeof(float) +
                               (size_t)S.ratio * (size_t)D * 2 * sizeof(float);
    }
    for (int li = 0; li < n_layer; li++) {
        const int dev = dev_of_layer(li);
        if (dev >= 0 && dev < ndev) {
            card_reserve[dev] += (size_t)W * (size_t)D * sizeof(float) + (size_t)W * sizeof(int32_t);
        }
    }
    for (int c = 0; c < ndev; c++) {
        card_reserve[c] += 512ULL * 1024 * 1024;
    }
    if (split_exp) {
        // The id ratio (split_c0, by link speed) is NOT the budget ratio: card 0
        // carries the dense MLA weights, the LM head and (at depth) three of the
        // four compressed caches, so it gets slightly less of the bank budget.
        // 0.47/0.53 at the default 16 GB is the measured fit on this box;
        // PF_DS41_BANK0_MB / PF_DS41_BANK1_MB override either side.
        double br = 0.47;
        if (const char * brs = si::env::str("PF_DS41_SPLIT_BUDGET_RATIO")) {
            br = atof(brs);
        }
        br = std::min(0.9, std::max(0.1, br));
        card_budget[0] = (size_t)((double)budget_bytes * br);
        for (int c = 1; c < ndev; c++) {
            card_budget[c] = (size_t)((double)budget_bytes * (1.0 - br) / (double)(ndev - 1));
        }
        for (int c = 0; c < ndev; c++) {
            card_budget[c] = card_budget[c] > card_reserve[c] ? card_budget[c] - card_reserve[c] : 0;
        }
        if (const char * b0 = si::env::str("PF_DS41_BANK0_MB")) {
            card_budget[0] = (size_t)std::max(0, atoi(b0)) * 1024 * 1024;
        }
        if (const char * b1 = si::env::str("PF_DS41_BANK1_MB")) {
            card_budget[1] = (size_t)std::max(0, atoi(b1)) * 1024 * 1024;
        }
    } else {
        // Every card budgets the same total; the per-layer share divides by
        // n_layer, so each layer's bank is sized identically on its owner.
        for (int c = 0; c < ndev; c++) {
            card_budget[c] = budget_bytes > card_reserve[c] ? budget_bytes - card_reserve[c] : 0;
        }
    }
    for (int li = 0; li < n_layer; li++) {
        const moe_t & M = m.layers[(size_t)li].moe;
        for (int c = 0; c < ndev; c++) {
            if (!split_exp && c != dev_of_layer(li)) {
                continue;
            }
            bank & B = bank_of(li, c);
            B.dev = c;
            B.gt = M.gate_exps.type;
            B.ut = M.up_exps.type;
            B.dt = M.down_exps.type;
            B.pg = M.gate_exps.plane_bytes;
            B.pu = M.up_exps.plane_bytes;
            B.pd = M.down_exps.plane_bytes;
            const size_t per = B.pg + B.pu + B.pd;
            int cap_ids = E;
            if (split_exp) {
                const int first = c == 0 ? 0 : split_c0;
                const int last = c == 0 ? split_c0 : E;
                cap_ids = std::max(1, last - first);
            }
            int R = 1;
            if (card_budget[c] > 0 && per > 0) {
                // --moe-vram-mb is the TOTAL bank budget across all devices; the
                // per-layer share divides by the total layer count.  With the
                // split each card has its own budget (ratio split, overridable).
                const size_t share = card_budget[c] / (size_t)std::max(1, n_layer);
                R = (int)(share / per);
                R = std::max(1, std::min(R, cap_ids));
            }
            B.R = R;
            try {
                B.g = alloc_dev<uint8_t>(B.dev, (size_t)B.R * B.pg, "expert gate bank");
                B.u = alloc_dev<uint8_t>(B.dev, (size_t)B.R * B.pu, "expert up bank");
                B.d = alloc_dev<uint8_t>(B.dev, (size_t)B.R * B.pd, "expert down bank");
            } catch (const std::exception & ex) {
                throw std::runtime_error(std::string("ds41 GPU expert bank: layer ") + std::to_string(li) +
                                         " card " + std::to_string(c) + " R=" + std::to_string(B.R) +
                                         " per-expert " + std::to_string((double)per / 1048576.0) + " MiB: " +
                                         ex.what());
            }
            total_bank_bytes += (size_t)B.R * per;
            B.e_slot.assign((size_t)E, -1);
            B.slot_e.assign((size_t)B.R, -1);
            B.age.assign((size_t)B.R, 0);
            B.tick = 0;
            // IQ2 uses the raw IQ2 GEMV; every other type goes through the
            // generic fp32-dequant GEMV (Q4_K/Q5_K/Q6_K/Q8_0/BF16/F32 all
            // supported).
            B.iq2_g = ds41_iq2_block_bytes(B.gt) != 0;
            B.iq2_u = ds41_iq2_block_bytes(B.ut) != 0;
            B.iq2_d = ds41_iq2_block_bytes(B.dt) != 0;
            if ((size_t)B.dev < e.dev_used_bytes_.size()) {
                e.dev_used_bytes_[(size_t)B.dev] += (uint64_t)((size_t)B.R * per);
            }
        }
    }
    // Pinned staging per device: the 3 planes of one expert are memcpy'd
    // from the mmap into pinned host memory, then copied asynchronously, so a
    // bank miss does not block the host/queue on pageable USM.
    {
        std::vector<size_t> max_plane((size_t)ndev, 0);
        for (int li = 0; li < n_layer; li++) {
            const moe_t & M = m.layers[(size_t)li].moe;
            const size_t per = M.gate_exps.plane_bytes + M.up_exps.plane_bytes + M.down_exps.plane_bytes;
            const int dv_i = dev_of_layer(li);
            if (split_exp) {
                // every card can serve any layer's experts
                for (int d2 = 0; d2 < ndev; d2++) {
                    max_plane[(size_t)d2] = std::max(max_plane[(size_t)d2], per);
                }
            } else if (dv_i >= 0 && dv_i < ndev) {
                max_plane[(size_t)dv_i] = std::max(max_plane[(size_t)dv_i], per);
            }
        }
        for (int d = 0; d < ndev; d++) {
            if (max_plane[(size_t)d] == 0) {
                continue;
            }
            dv[(size_t)d].estage_cap = max_plane[(size_t)d];
            dv[(size_t)d].estage = (uint8_t *)alloc_host(8 * max_plane[(size_t)d], "expert staging ring");
            for (int k = 0; k < 8; k++) {
                dv[(size_t)d].estage_ev[k] = sycl::event{};
            }
        }
    }

    if (!dv.empty() && dv[0].estage) {
        crew = std::make_unique<ds41_copy_crew>();
    }
    fprintf(stderr, "[ds41] head: type=%u K=%d N=%d rowbytes=%zu | tok_embd type=%u K=%d\n",
            (unsigned)m.output.type, m.output.K, m.output.N,
            (size_t)ggml_row_bytes(m.output.type, (uint64_t)m.output.K), (unsigned)m.tok_embd.type, m.tok_embd.K);
    fprintf(stderr,
            "[ds41] expert bank: budget=%d MB%s, %.1f MiB resident (card0 R=%d, card1 R=%d), "
            "E=%d top%d, expert type %u/%u/%u%s\n",
            bank_budget_mb, split_exp ? " (split across cards)" : "", (double)total_bank_bytes / 1048576.0,
            n_layer > 0 ? bank_of(0, 0).R : 0, (n_layer > 0 && ndev > 1) ? bank_of(0, 1).R : 0, E, topk,
            (unsigned)m.layers[0].moe.gate_exps.type, (unsigned)m.layers[0].moe.up_exps.type,
            (unsigned)m.layers[0].moe.down_exps.type,
            n_layer == 0 ? "" : (bank_of(0, dev_of_layer(0)).iq2_g ? " (IQ2 GEMV)" : " (raw GEMV)"));
}

void ds41_gpu_state::calibrate() {
    // Keep the prompt's most-frequently-routed experts in the bank.  The LRU
    // during the prefill ends up holding the last token's experts; the hot set
    // over the whole prompt is a better prior for the decode.
    for (int li = 0; li < n_layer; li++) {
        for (int c = 0; c < ndev; c++) {
            if (!split_exp && c != dev_of_layer(li)) {
                continue;
            }
            bank & B = bank_of(li, c);
            std::vector<std::pair<uint32_t, int>> v;
            v.reserve((size_t)E);
            const int first = split_exp && c != 0 ? split_c0 : 0;
            const int last = split_exp && c == 0 ? split_c0 : E;
            for (int e = first; e < last; e++) {
                const uint32_t f = route_freq[(size_t)li][(size_t)e];
                if (f) {
                    v.emplace_back(f, e);
                }
            }
            const int keep = std::min<int>(B.R, (int)v.size());
            std::partial_sort(v.begin(), v.begin() + keep, v.end(),
                              [](const std::pair<uint32_t, int> & a, const std::pair<uint32_t, int> & b) {
                                  return a.first != b.first ? a.first > b.first : a.second < b.second;
                              });
            for (int i = 0; i < keep; i++) {
                bank_slot(li, v[(size_t)i].second, c);
            }
        }
    }
    calib = false;
}

ds41_gpu_state::~ds41_gpu_state() {
    for (auto & t : dv) {
        if (t.cq) {
            t.cq->wait();
        }
        if (t.q) {
            t.q->wait();
        }
    }
    for (auto & pr : owned) {
        if (pr.first && pr.second >= 0 && pr.second < (int)dv.size() && dv[(size_t)pr.second].q) {
            sycl::free(pr.first, Q(pr.second));
        }
    }
    owned.clear();
    for (void * p : host_owned) {
        if (p) {
            sycl::free(p, Q(0));
        }
    }
    host_owned.clear();
}

// ===========================================================================
// Dense GEMV helpers
// ===========================================================================
void ds41_gpu_state::gemv_calls(int dev, uint32_t type, std::vector<gemv_seg> & segs) {
    if (segs.empty()) {
        return;
    }
    devbuf & t = DV(dev);
    int total_rows = 0;
    for (const gemv_seg & s : segs) {
        total_rows += s.n_rows;
    }
    int done = 0;
    while (done < (int)segs.size()) {
        if (t.seg_used + (int)segs.size() - done > t.seg_cap) {
            t.q->wait();
            t.seg_used = 0;
        }
        const int base = t.seg_used;
        const int n = (int)segs.size() - done;
        for (int i = 0; i < n; i++) {
            t.hsegs[base + i] = segs[(size_t)(done + i)];
        }
        t.q->memcpy(t.dsegs + base, t.hsegs + base, (size_t)n * sizeof(gemv_seg));
        const int sb = segs[(size_t)done].K / 256;
        gemv_group_launch(*t.q, type, t.dsegs + base, n, total_rows, 1, sb, 0);
        t.seg_used += n;
        done += n;
    }
}

void ds41_gpu_state::gemv1(int dev, const void * w, uint32_t type, int K, int N, const float * x, float * out,
                           const float * residual, float alpha, int out_stride) {
    // A oneDNN-managed dense store (setup_ds41_dense_store): the k5 M=1 GEMV
    // reads the native 5-bit values at 249 GB/s instead of the raw
    // fp32-dequant path's 110 (PF_DS41_DENSE_Q8 additionally routes Q8_0
    // through the int8 GEMV; it is off by default).
    // The activation is quantized per 32-value group exactly as every other
    // oneDNN decode path does; `quantize` + `gemm` are one launch each on this
    // device's in-order queue.  The plain path below stays for the expert bank,
    // the BF16 compressor/indexer tensors and PF_DS41_DENSE_STORE=0.
    if ((out_stride == 0 || out_stride == N) && K > 0 && (K % 32) == 0) {
        if (dnnl_gemm * D = e.dnnl_for(dev)) {
            if (D->has_weight_k5(w) || D->has_weight(w)) {
                auto v = D->quantize(x, nullptr, K, 0, 1, K, /*do_split=*/true);
                if (v.valid() && D->gemm(v, w, residual, alpha, 1, K, out, N)) {
                    return;
                }
            }
        }
    }
    gemv_seg s{};
    s.w = w;
    s.type = type;
    s.K = K;
    s.n_rows = N;
    s.dev = dev;
    s.x = x;
    s.x_stride = K;
    s.out = out;
    s.out_stride = out_stride ? out_stride : N;
    s.residual = residual;
    s.alpha = alpha;
    std::vector<gemv_seg> one{s};
    gemv_calls(dev, type, one);
}

void ds41_gpu_state::rms(int dev, const float * x, const void * w, float * out, int n) {
    rmsnorm_launch(Q(dev), x, (const float *)wp(dev, w), out, 1, n, hp.rms_eps);
}

void ds41_gpu_state::rms_rows(int dev, const float * x, const void * w, float * out, int rows, int n) {
    rmsnorm_launch(Q(dev), x, (const float *)wp(dev, w), out, rows, n, hp.rms_eps);
}

// ===========================================================================
// Sequence state
// ===========================================================================
void ds41_gpu_state::reset_state() {
    pos = 0;
    hist[0] = hist[1] = hist[2] = hist[3] = hist[4] = hist[5] = hist[6] = hist[7] = 0;
    hist_n = 0;
    cand.clear();
    cand_nb = 0;
    has_cand = false;
    sel_k = 0;
    sel_dev = -1;
    for (int i = 0; i < ndev; i++) {
        for (int li = 0; li < n_layer; li++) {
            if (dev_of_layer(li) != i) {
                continue;
            }
            Q(i).memset(win[(size_t)li], 0, (size_t)W * (size_t)D * sizeof(float));
            Q(i).memset(winpos[(size_t)li], 0xff, (size_t)W * sizeof(int32_t));
        }
    }
    for (src & S : sources) {
        S.clen = S.ring_n = S.ik_rows = 0;
    }
    for (bank & B : banks) {
        std::fill(B.e_slot.begin(), B.e_slot.end(), -1);
        std::fill(B.slot_e.begin(), B.slot_e.end(), -1);
        std::fill(B.age.begin(), B.age.end(), 0);
        B.tick = 0;
    }
}

// ===========================================================================
// Engram (host hash + gather, device projection/apply)
// ===========================================================================
void ds41_gpu_state::engram_apply(int li, int dev) {
    const layer_t & L = m.layers[(size_t)li];
    const int ncol = d.ple_n_heads;
    const int hd = d.ple_head_dim;
    const int eidx = d.engram_index_of[li];

    // n-gram hash for the current token (hist is most-recent-first)
    const int ng = std::min(d.ple_ngram, 8);
    int64_t tok[8];
    for (int s = 0; s < ng; s++) {
        tok[s] = (s < hist_n) ? hist[s] : pad;
    }
    std::vector<int> rows((size_t)ncol, 0);
    int64_t rolling = (int64_t)((uint64_t)tok[0] * (uint64_t)d.ple_mult[(size_t)eidx * 4 + 0]);
    for (int o = 1; o < ng; o++) {
        const int64_t prod = (int64_t)((uint64_t)tok[o] * (uint64_t)d.ple_mult[(size_t)eidx * 4 + o]);
        rolling = (int64_t)((uint64_t)rolling ^ (uint64_t)prod);
        for (int hj = 0; hj < d.ple_heads_per_ngram; hj++) {
            const int col = (o - 1) * d.ple_heads_per_ngram + hj;
            const int64_t voc = (int64_t)d.ple_vocab[(size_t)eidx * ncol + col];
            int64_t r = rolling % voc;
            if (r < 0) {
                r += voc;
            }
            rows[(size_t)col] = (int)(r + (int64_t)d.ple_off[(size_t)eidx * ncol + col]);
        }
    }

    // gather + dequantize the 24 IQ4_NL rows on the host
    const wt & tab = ds41_model(m).ple_table;
    const size_t rb = ggml_row_bytes(tab.type, (uint64_t)tab.K);
    const char * base = (const char *)tab.data;
    std::vector<float> ev((size_t)ncol * (size_t)hd);
    for (int col = 0; col < ncol; col++) {
        dequantize_row(tab.type, base + (size_t)rows[(size_t)col] * rb, ev.data() + (size_t)col * hd, hd);
    }
    if (eng_bf16) {
        for (float & v : ev) {
            v = bf16g(v);
        }
    }
    devbuf & t = DV(dev);
    t.q->memcpy(t.eng_ev, ev.data(), ev.size() * sizeof(float)).wait();
    gemv1(dev, wk(dev, ds41_layer_of(L).engram_kv.data), ds41_layer_of(L).engram_kv.type, ds41_layer_of(L).engram_kv.K, ds41_layer_of(L).engram_kv.N,
          t.eng_ev, t.eng_kv);
    engram_dot_launch(*t.q, t.h, t.eng_kv, (const float *)wp(dev, ds41_layer_of(L).engram_gate), hc, dim, t.eng_g);
    engram_add_launch(*t.q, t.h, t.eng_kv, t.eng_g, hc, dim);
}

// ===========================================================================
// KV source (compressor) and indexer
// ===========================================================================
void ds41_gpu_state::append_src(int li, int sidx, const float * latent, int block) {
    const layer_t & L = m.layers[(size_t)li];
    src & S = sources[(size_t)sidx];
    const int dev = S.dev;
    devbuf & t = DV(dev);
    sycl::queue & q = *t.q;
    const int row = S.clen;
    if (row >= (int)S.maxclen) {
        throw std::runtime_error("ds41 GPU: compressed cache overflow");
    }
    // compressed K row: rope at the block start, then fp4(block=16, e4m3)
    q.memcpy(t.rowo, latent, (size_t)D * sizeof(float));
    ds41_rope_launch(q, t.rowo, 1, D, rot_off, n_rot, block * S.ratio, 0, t.fcomp, false);
    if (!noq) {
        ds41_fp4_round_launch(q, t.rowo, D, 16, true);
    }
    q.memcpy(S.ckv + (size_t)row * (size_t)D, t.rowo, (size_t)D * sizeof(float));

    // index key from the *pre-rope* latent
    gemv1(dev, wk(dev, ds41_layer_of(L).idx_attn_k.data), ds41_layer_of(L).idx_attn_k.type, ds41_layer_of(L).idx_attn_k.K,
          ds41_layer_of(L).idx_attn_k.N, latent, t.ikproj);
    rms(dev, t.ikproj, ds41_layer_of(L).idx_k_norm, t.ikproj, idx_hd);
    ds41_rope_launch(q, t.ikproj, 1, idx_hd, idx_hd - 64, 64, block * S.ratio, 0, t.fcomp, false);
    if (!noq) {
        ds41_fp4_round_launch(q, t.ikproj, idx_hd, 32, false);
    }
    q.memcpy(S.ik + (size_t)row * (size_t)idx_hd, t.ikproj, (size_t)idx_hd * sizeof(float));
    S.clen = row + 1;
    S.ik_rows = S.clen;
}

void ds41_gpu_state::push_kvsrc(int li, int dev, const float * x) {
    const layer_t & L = m.layers[(size_t)li];
    const int sidx = kv_slot_of[(size_t)li];
    if (sidx < 0) {
        return;
    }
    src & S = sources[(size_t)sidx];
    devbuf & t = DV(dev);
    if (S.ratio == 1) {
        gemv1(dev, wk(dev, ds41_layer_of(L).comp_wkv.data), ds41_layer_of(L).comp_wkv.type, ds41_layer_of(L).comp_wkv.K, ds41_layer_of(L).comp_wkv.N, x,
              t.latent);
        rms(dev, t.latent, ds41_layer_of(L).comp_norm, t.latent, D);
        append_src(li, sidx, t.latent, S.clen);
        return;
    }
    float * kdst = S.rkv + (size_t)S.ring_n * (size_t)D;
    gemv1(dev, wk(dev, ds41_layer_of(L).comp_wkv.data), ds41_layer_of(L).comp_wkv.type, ds41_layer_of(L).comp_wkv.K, ds41_layer_of(L).comp_wkv.N, x, kdst);
    if (ds41_layer_of(L).comp_wgate.data) {
        float * gdst = S.rsc + (size_t)S.ring_n * (size_t)D;
        gemv1(dev, wk(dev, ds41_layer_of(L).comp_wgate.data), ds41_layer_of(L).comp_wgate.type, ds41_layer_of(L).comp_wgate.K,
              ds41_layer_of(L).comp_wgate.N, x, gdst);
    }
    S.ring_n++;
    if (S.ring_n < S.ratio) {
        return;
    }
    ds41_compressor_launch(*t.q, S.rkv, S.rsc, (const float *)wp(dev, ds41_layer_of(L).comp_norm), t.latent, 1,
                           S.ratio, D, hp.rms_eps);
    append_src(li, sidx, t.latent, S.clen);
    S.ring_n = 0;
}

void ds41_gpu_state::index_select(int li, int dev, const float * x, const float * qr) {
    const layer_t & L = m.layers[(size_t)li];
    const int ratio = d.compress_ratio[li];
    const int sidx = kv_slot_of[(size_t)li];
    if (sidx < 0) {
        sel_k = 0;
        return;
    }
    src & S = sources[(size_t)sidx];
    const int clen = std::min((pos + 1) / ratio, S.clen);
    if (clen <= 0) {
        sel_k = 0;
        return;
    }
    devbuf & t = DV(dev);
    sycl::queue & q = *t.q;

    gemv1(dev, wk(dev, ds41_layer_of(L).idx_attn_q_b.data), ds41_layer_of(L).idx_attn_q_b.type, ds41_layer_of(L).idx_attn_q_b.K,
          ds41_layer_of(L).idx_attn_q_b.N, qr, t.qi);
    ds41_rope_launch(q, t.qi, idx_nh, idx_hd, idx_hd - 64, 64, pos, 0, t.fcomp, false);
    if (!noq) {
        ds41_fp4_round_launch(q, t.qi, idx_nh * idx_hd, 32, false);
    }
    float wscale = 1.f / std::sqrt((float)idx_hd * (float)idx_nh);
    float wdev[64];
    {
        std::vector<float> wh((size_t)idx_nh);
        // idx_proj → host, scale, upload
        gemv1(dev, wk(dev, ds41_layer_of(L).idx_proj.data), ds41_layer_of(L).idx_proj.type, ds41_layer_of(L).idx_proj.K, ds41_layer_of(L).idx_proj.N, x,
              t.iw);
        q.memcpy(wdev, t.iw, (size_t)idx_nh * sizeof(float)).wait();
        for (int h = 0; h < idx_nh; h++) {
            wdev[h] *= wscale;
        }
        q.memcpy(t.iw, wdev, (size_t)idx_nh * sizeof(float)).wait();
        (void)wh;
    }

    ds41_indexer_score_launch(q, t.qi, S.ik, t.iw, t.score, clen, idx_nh, idx_hd);
    q.memcpy(sstage, t.score, (size_t)clen * sizeof(float)).wait();

    // candidate blocks at the candidate source layer
    if (li == d.cand_src) {
        const int block = d.cand_block;
        const int nb = (clen + block - 1) / block;
        cand.assign((size_t)nb, 0);
        cand_nb = nb;
        std::vector<float> bmax((size_t)nb);
        for (int b = 0; b < nb; b++) {
            float mx = -std::numeric_limits<float>::infinity();
            const int hi = std::min(clen, (b + 1) * block);
            for (int cc = b * block; cc < hi; cc++) {
                mx = std::max(mx, sstage[cc]);
            }
            bmax[(size_t)b] = mx;
        }
        const int last = (clen - 1) / block;
        if (last >= 0 && last < nb) {
            bmax[(size_t)last] = std::numeric_limits<float>::infinity();
        }
        std::vector<int> ord((size_t)nb);
        for (int b = 0; b < nb; b++) {
            ord[(size_t)b] = b;
        }
        const int take = std::min(d.cand_topk_blocks, nb);
        std::partial_sort(ord.begin(), ord.begin() + take, ord.end(), [&](int a, int b) {
            if (bmax[(size_t)a] != bmax[(size_t)b]) {
                return bmax[(size_t)a] > bmax[(size_t)b];
            }
            return a < b;
        });
        for (int i = 0; i < take; i++) {
            const int b = ord[(size_t)i];
            if (bmax[(size_t)b] > -std::numeric_limits<float>::infinity()) {
                cand[(size_t)b] = 1;
            }
        }
        has_cand = true;
    } else if (d.cand_src >= 0 && li > d.cand_src && has_cand) {
        for (int cc = 0; cc < clen; cc++) {
            const int b = cc / d.cand_block;
            if (b >= cand_nb || !cand[(size_t)b]) {
                sstage[cc] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    const int k = std::min(idx_k, clen);
    std::vector<int> ord((size_t)clen);
    for (int cc = 0; cc < clen; cc++) {
        ord[(size_t)cc] = cc;
    }
    std::partial_sort(ord.begin(), ord.begin() + k, ord.end(), [&](int a, int b) {
        const float sa = sstage[a], sb = sstage[b];
        if (sa != sb) {
            return sa > sb;
        }
        return a < b;
    });
    ord.resize((size_t)k);
    std::sort(ord.begin(), ord.end());
    q.memcpy(t.sel, ord.data(), (size_t)k * sizeof(int32_t)).wait();
    gather_rows_launch(q, S.ckv, t.sel, k, D, t.compact);
    sel_k = k;
    sel_dev = dev;
}

// ===========================================================================
// Attention (one token)
// ===========================================================================
void ds41_gpu_state::attention(int li, int dev, const float * x, float * oo) {
    const layer_t & L = m.layers[(size_t)li];
    const int ratio = d.compress_ratio[li];
    const bool comp = ratio > 0;
    const float * fc = comp ? DV(dev).fcomp : DV(dev).fwin;
    devbuf & t = DV(dev);
    sycl::queue & q = *t.q;

    // qr = rms(wq_a x, q_a_norm)
    gemv1(dev, wk(dev, ds41_layer_of(L).wq_a.data), ds41_layer_of(L).wq_a.type, ds41_layer_of(L).wq_a.K, ds41_layer_of(L).wq_a.N, x, t.qr);
    if (dbg) { q.memcpy(sstage, t.qr, (size_t)q_lora*sizeof(float)).wait(); double ss=0; for(int i=0;i<q_lora;i++) ss+=(double)sstage[i]*sstage[i]; fprintf(stderr, "[ds41g]   qr rms %.6f\n", std::sqrt(ss/(double)q_lora)); }
    rms(dev, t.qr, ds41_layer_of(L).q_a_norm, t.qr, q_lora);
    gemv1(dev, wk(dev, ds41_layer_of(L).wq_b.data), ds41_layer_of(L).wq_b.type, ds41_layer_of(L).wq_b.K, ds41_layer_of(L).wq_b.N, t.qr, t.qb);
    if (dbg) { q.memcpy(sstage, t.qb, 64*sizeof(float)).wait(); double ss=0; for(int i=0;i<64;i++) ss+=(double)sstage[i]*sstage[i]; fprintf(stderr, "[ds41g]   q[0:64] rms %.6f\n", std::sqrt(ss/64.0)); }
    ds41_rope_launch(q, t.qb, H, D, rot_off, n_rot, pos, 0, fc, false);

    // window K row
    gemv1(dev, wk(dev, ds41_layer_of(L).wkv.data), ds41_layer_of(L).wkv.type, ds41_layer_of(L).wkv.K, ds41_layer_of(L).wkv.N, x, t.kvrow);
    rms(dev, t.kvrow, ds41_layer_of(L).kv_norm, t.kvrow, D);
    ds41_rope_launch(q, t.kvrow, 1, D, rot_off, n_rot, pos, 0, fc, false);
    if (!noq) {
        fp8_round_launch(q, t.kvrow, D);
    }
    window_write_launch(q, t.kvrow, win[(size_t)li], winpos[(size_t)li], pos, W, D);

    if (comp) {
        const int src = d.kv_src_of[li];
        const bool is_src = (src == li);
        const bool is_idx = (li == d.topk_src_of[li]);
        if (is_src) {
            push_kvsrc(li, dev, x);
        }
        if (is_idx) {
            index_select(li, dev, x, t.qr);
        }
    }

    const int n_win = std::min(pos + 1, W);
    const int win_start = std::max(0, pos - W + 1);
    int K = 0;
    if (comp && kv_slot_of[(size_t)li] >= 0) {
        K = std::min(sel_k, (int)sources[(size_t)kv_slot_of[(size_t)li]].clen);
    }
    // The compact buffer is only valid on the device that produced the
    // selection; with the 19/20 split every consumer is on that device.
    if (sel_dev != dev) {
        K = 0;
    }
    ds41_mla_attn_launch(q, t.qb, win[(size_t)li], winpos[(size_t)li], W,
                         (K > 0) ? t.compact : nullptr, K,
                         (const float *)wp(dev, ds41_layer_of(L).attn_sinks), H, D, pos, n_win, win_start,
                         1.f / std::sqrt((float)D), fc, rot_off, n_rot, t.ob);
    if (!noq) {
        fp8_round_launch(q, t.ob, H * D);
    }

    // 8 groups through wo_a, then wo_b
    const int gsz = H / out_g * D;
    const size_t row_b = ggml_row_bytes(ds41_layer_of(L).wo_a.type, (uint64_t)ds41_layer_of(L).wo_a.K);
    const char * wab = (const char *)wk(dev, ds41_layer_of(L).wo_a.data);
    for (int g = 0; g < out_g; g++) {
        gemv1(dev, wab + (size_t)g * (size_t)out_rank * row_b, ds41_layer_of(L).wo_a.type, gsz, out_rank,
              t.ob + (size_t)g * (size_t)gsz, t.og + (size_t)g * (size_t)out_rank);
    }
    if (dbg) { q.memcpy(sstage, t.og, (size_t)out_rank*sizeof(float)).wait(); double ss=0; for(int i=0;i<out_rank;i++) ss+=(double)sstage[i]*sstage[i]; fprintf(stderr, "[ds41g]   og rms %.6f\n", std::sqrt(ss/(double)out_rank));
        q.memcpy(sstage, t.ob, (size_t)std::min(64,H*D)*sizeof(float)).wait(); ss=0; double mx=0; for(int i=0;i<std::min(64,H*D);i++){ss+=(double)sstage[i]*sstage[i]; mx=std::max(mx,(double)std::fabs(sstage[i]));} fprintf(stderr, "[ds41g]   attn pre-wo rms %.6f max %.6f\n", std::sqrt(ss/64.0), mx); }
    gemv1(dev, wk(dev, ds41_layer_of(L).wo_b.data), ds41_layer_of(L).wo_b.type, ds41_layer_of(L).wo_b.K, ds41_layer_of(L).wo_b.N, t.og, oo);
}

// ===========================================================================
// MoE
// ===========================================================================
int ds41_gpu_state::bank_slot(int li, int iexp, int card) {
    const int c = card >= 0 ? card : dev_of_layer(li);
    bank & B = bank_of(li, c);
    const moe_t & M = m.layers[(size_t)li].moe;
    if (B.e_slot[(size_t)iexp] >= 0) {
        const int s = B.e_slot[(size_t)iexp];
        B.age[(size_t)s] = ++B.tick;
        B.hits++;
        return s;
    }
    B.misses++;
    B.miss_bytes += B.pg + B.pu + B.pd;
    int v = -1;
    for (int r = 0; r < B.R; r++) {
        if (B.slot_e[(size_t)r] < 0) {
            v = r;
            break;
        }
    }
    if (v < 0) {
        uint32_t best = std::numeric_limits<uint32_t>::max();
        for (int r = 0; r < B.R; r++) {
            if (B.age[(size_t)r] < best) {
                best = B.age[(size_t)r];
                v = r;
            }
        }
    }
    if (B.slot_e[(size_t)v] >= 0) {
        B.e_slot[(size_t)B.slot_e[(size_t)v]] = -1;
    }
    B.slot_e[(size_t)v] = iexp;
    B.e_slot[(size_t)iexp] = v;
    B.age[(size_t)v] = ++B.tick;
    sycl::queue & q = Q(B.dev);
    devbuf & t = DV(B.dev);
    sycl::queue & cq = t.cq ? *t.cq : q;
    if (t.estage) {
        // Stage all three planes in pinned memory, then enqueue the H2D.  The
        // in-order queue orders the copy before the GEMV, but the copy itself
        // is non-blocking, so the host can stage the next expert while this
        // one transfers.  A 4-slot ring is enough to cover a token's 6 experts.
        const int si = t.estage_next++ & 7;
        if (t.estage_pending[si]) {
            t.estage_ev[si].wait();
            t.estage_pending[si] = false;
        }
        uint8_t * st = t.estage + (size_t)si * t.estage_cap;
        if (!abl_nostage) {
            const uint8_t * sg = (const uint8_t *)M.gate_exps.plane(iexp);
            const uint8_t * su = (const uint8_t *)M.up_exps.plane(iexp);
            const uint8_t * sd = (const uint8_t *)M.down_exps.plane(iexp);
            auto half = [](ds41_copy_crew::job * js, int & k, const uint8_t * src, uint8_t * dst, size_t n) {
                const size_t h = n / 2;
                js[k++] = {src, dst, h};
                js[k++] = {src + h, dst + h, n - h};
            };
            ds41_copy_crew::job js[6];
            int k = 0;
            half(js, k, sg, st, B.pg);
            half(js, k, su, st + B.pg, B.pu);
            half(js, k, sd, st + B.pg + B.pu, B.pd);
            if (crew) {
                crew->run(js, k);
            } else {
                for (int j = 0; j < k; j++) {
                    std::memcpy(js[j].d, js[j].s, js[j].n);
                }
            }
        }
        cq.memcpy((uint8_t *)B.g + (size_t)v * B.pg, st, B.pg);
        cq.memcpy((uint8_t *)B.u + (size_t)v * B.pu, st + B.pg, B.pu);
        t.estage_ev[si] = cq.memcpy((uint8_t *)B.d + (size_t)v * B.pd, st + B.pg + B.pu, B.pd);
        t.estage_pending[si] = true;
        if (t.cq) {
            // Make the compute queue wait for the copy, so the next GEMV reads
            // the slot; the copy queue is free to start the next expert's H2D.
            q.ext_oneapi_submit_barrier({t.estage_ev[si]});
        }
    } else {
        q.memcpy((uint8_t *)B.g + (size_t)v * B.pg, M.gate_exps.plane(iexp), B.pg);
        q.memcpy((uint8_t *)B.u + (size_t)v * B.pu, M.up_exps.plane(iexp), B.pu);
        q.memcpy((uint8_t *)B.d + (size_t)v * B.pd, M.down_exps.plane(iexp), B.pd);
    }
    B.uploads += 3;
    return v;
}

void ds41_gpu_state::route_one(int li, const float * x, int dev, int32_t * ids, float * wts) {
    const layer_t & L = m.layers[(size_t)li];
    devbuf & t = DV(dev);
    sycl::queue & q = *t.q;
    gemv1(dev, wk(dev, L.moe.gate_inp), L.moe.gate_inp_type, dim, E, x, t.logits);
    q.memcpy(sstage, t.logits, (size_t)E * sizeof(float)).wait();

    const float * bias = (moe_vl && router_bias_vl_p) ? router_bias_vl_p : router_bias_p;
    std::vector<float> score((size_t)E);
    for (int i = 0; i < E; i++) {
        score[(size_t)i] = std::sqrt(softplusg(sstage[i]));
    }
    std::vector<int> ord((size_t)E);
    for (int i = 0; i < E; i++) {
        ord[(size_t)i] = i;
    }
    std::partial_sort(ord.begin(), ord.begin() + topk, ord.end(), [&](int a, int b) {
        const float va = score[(size_t)a] + (bias ? bias[a] : 0.f);
        const float vb = score[(size_t)b] + (bias ? bias[b] : 0.f);
        if (va != vb) {
            return va > vb;
        }
        return a < b;
    });
    float sum = 1e-20f;
    for (int j = 0; j < topk; j++) {
        ids[j] = ord[(size_t)j];
        sum += score[(size_t)ord[(size_t)j]];
    }
    for (int j = 0; j < topk; j++) {
        wts[j] = score[(size_t)ord[(size_t)j]] / sum * d.expert_scale;
    }

    if (calib) {
        for (int j = 0; j < topk; j++) {
            route_freq[(size_t)li][(size_t)ids[j]]++;
        }
    }
}

void ds41_gpu_state::moe(int li, int dev, const float * x, float * y) {
    const layer_t & L = m.layers[(size_t)li];
    devbuf & t = DV(dev);
    sycl::queue & q = *t.q;
    const float lim = d.swiglu_clamp[li];

    q.memset(y, 0, (size_t)dim * sizeof(float));

    if (split_exp) {
        // Publish x to the peer before route_one's blocking logits wait: that
        // wait drains the owner's in-order queue, so the copy is complete by
        // the time the peer's H2D is enqueued, with no extra host wait.
        q.memcpy(xp_stage, x, (size_t)dim * sizeof(float));
    }

    int32_t ids[16];
    float wts[16];
    route_one(li, x, dev, ids, wts);

    // One expert's three GEMVs + clamp, on `card`'s queue, into `out`.  The
    // types come from that card's bank for this layer (both cards hold the same
    // per-layer type facts; only the resident ids differ).
    auto run_expert = [&](int card, int ie, const float * xin, float * out) {
        devbuf & dv = DV(card);
        sycl::queue & dq = *dv.q;
        bank & B = bank_of(li, card);
        const int slot = bank_slot(li, ie, card);
        const void * gw = (const uint8_t *)B.g + (size_t)slot * B.pg;
        const void * uw = (const uint8_t *)B.u + (size_t)slot * B.pu;
        const void * dw = (const uint8_t *)B.d + (size_t)slot * B.pd;
        if (B.iq2_g) {
            ds41_iq2_gemv_launch(dq, gw, dim, ff, B.gt, 0, 0, xin, dv.g, 1, 0, 0, dv.tab);
        } else {
            gemv1(card, gw, B.gt, dim, ff, xin, dv.g);
        }
        if (B.iq2_u) {
            ds41_iq2_gemv_launch(dq, uw, dim, ff, B.ut, 0, 0, xin, dv.u, 1, 0, 0, dv.tab);
        } else {
            gemv1(card, uw, B.ut, dim, ff, xin, dv.u);
        }
        swiglu_clamp_launch(dq, dv.g, dv.u, dv.hh, ff, lim);
        if (B.iq2_d) {
            ds41_iq2_gemv_launch(dq, dw, ff, dim, B.dt, 0, 0, dv.hh, out, 1, 0, 0, dv.tab);
        } else {
            gemv1(card, dw, B.dt, ff, dim, dv.hh, out);
        }
    };

    if (!split_exp) {
        for (int j = 0; j < topk; j++) {
            run_expert(dev, ids[j], x, t.acc);
            yadd_launch(q, y, t.acc, wts[j], dim);
        }
    } else {
        const int peer = (dev + 1) % ndev;
        int own_j[16];
        int peer_j[16];
        int n_own = 0, n_peer = 0;
        for (int j = 0; j < topk; j++) {
            if (expert_card(ids[j]) == dev) {
                own_j[n_own++] = j;
            } else {
                peer_j[n_peer++] = j;
            }
        }
        sycl::event peer_ev;
        if (n_peer > 0) {
            Q(peer).memcpy(DV(peer).px, xp_stage, (size_t)dim * sizeof(float));
            for (int k = 0; k < n_peer; k++) {
                const int j = peer_j[k];
                run_expert(peer, ids[j], DV(peer).px, DV(peer).pacc + (size_t)j * (size_t)dim);
            }
            // The whole top-k block comes back so the owner can replay the adds
            // in the original order; 120 KB per layer is noise next to the
            // expert planes.
            peer_ev = Q(peer).memcpy(pacc_host, DV(peer).pacc,
                                     (size_t)topk * (size_t)dim * sizeof(float));
        }
        for (int k = 0; k < n_own; k++) {
            const int j = own_j[k];
            run_expert(dev, ids[j], x, t.pacc + (size_t)j * (size_t)dim);
        }
        if (n_peer > 0) {
            // The owner's own GEMVs are already running; this waits only for
            // the peer's remainder, not for the owner's queue.
            peer_ev.wait();
            q.memcpy(t.premote, pacc_host, (size_t)topk * (size_t)dim * sizeof(float));
        }
        // Same expert dots and the same add sequence as the unsplit path, so
        // this is bit-identical to it.
        for (int j = 0; j < topk; j++) {
            const float * acc = expert_card(ids[j]) == dev ? t.pacc + (size_t)j * (size_t)dim
                                                           : t.premote + (size_t)j * (size_t)dim;
            yadd_launch(q, y, acc, wts[j], dim);
        }
    }

    // shared expert (ungated add, same one-sided clamp family)
    const float slim = d.swiglu_clamp_shexp[li];
    gemv1(dev, wk(dev, L.ffn_gate.data), L.ffn_gate.type, L.ffn_gate.K, L.ffn_gate.N, x, t.g);
    gemv1(dev, wk(dev, L.ffn_up.data), L.ffn_up.type, L.ffn_up.K, L.ffn_up.N, x, t.u);
    swiglu_clamp_launch(q, t.g, t.u, t.sh, ffsh, slim);
    gemv1(dev, wk(dev, L.ffn_down.data), L.ffn_down.type, L.ffn_down.K, L.ffn_down.N, t.sh, t.acc);
    yadd_launch(q, y, t.acc, 1.f, dim);
}

// ===========================================================================
// Cross-partition handoff + head
// ===========================================================================
void ds41_gpu_state::handoff(int from, int to) {
    if (from == to) {
        return;
    }
    const size_t hb = (size_t)hc * (size_t)dim * sizeof(float);
    const size_t pb = (size_t)hc * sizeof(float);
    Q(from).memcpy(hstage, DV(from).h, hb).wait();
    Q(from).memcpy(pstage, DV(from).pre_prev, pb).wait();
    Q(to).memcpy(DV(to).h, hstage, hb).wait();
    Q(to).memcpy(DV(to).pre_prev, pstage, pb).wait();
}

void ds41_gpu_state::head(int dev) {
    devbuf & t = DV(dev);
    sycl::queue & q = *t.q;
    ds41_hc_pre_launch(q, t.h, t.pre_prev, t.mixx, hc, dim);
    rms(dev, t.mixx, m.output_norm, t.xn, dim);
    gemv1(dev, wk(dev, m.output.data), m.output.type, m.output.K, m.output.N, t.xn, t.logits);
}

// ===========================================================================
// One token
// ===========================================================================
void ds41_gpu_state::forward_token(int tok, bool want_logits, const mm_prompt * mm) {
    const auto t_start = std::chrono::steady_clock::now();
    const auto t_start0 = t_start; // phase split reference (PF_DS41_TIME=2)
    const bool is_img = mm && (size_t)pos < mm->img_row.size() && mm->img_row[(size_t)pos] >= 0;

    // mapped-token history (most recent first)
    int64_t mapped = tok;
    if (!ds41_model(m).engram_token_map.empty() && tok >= 0 && (size_t)tok < ds41_model(m).engram_token_map.size()) {
        mapped = ds41_model(m).engram_token_map[(size_t)tok];
    }
    for (int i = 7; i > 0; i--) {
        hist[i] = hist[i - 1];
    }
    hist[0] = mapped;
    if (hist_n < 8) {
        hist_n++;
    }

    // embedding on the host (one row) then expand to hc streams on dev 0
    const wt & te = m.tok_embd;
    if (is_img) {
        const int rr = mm->img_row[(size_t)pos];
        std::memcpy(h0.data(), mm->embd.data() + (size_t)rr * (size_t)dim, (size_t)dim * sizeof(float));
    } else {
        const size_t rb = ggml_row_bytes(te.type, (uint64_t)te.K);
        const char * base = (const char *)wp(0, te.data);
        dequantize_row(te.type, base + (size_t)tok * rb, h0.data(), dim);
    }
    int cur = 0;
    {
        devbuf & t = DV(0);
        t.q->memcpy(t.h, h0.data(), (size_t)dim * sizeof(float));
        q4_repeat_launch(*t.q, t.h, t.h, 1, dim, hc);
        const float init_pre[8] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        t.q->memcpy(t.pre_prev, init_pre, (size_t)hc * sizeof(float)).wait();
    }

    for (int li = 0; li < n_layer; li++) {
        const int dev = dev_of_layer(li);
        if (dev != cur) {
            handoff(cur, dev);
            cur = dev;
        }
        const layer_t & L = m.layers[(size_t)li];
        devbuf & t = DV(dev);
        sycl::queue & q = *t.q;

        if (d.is_ple[li] && !is_img) {
            engram_apply(li, dev);
        }
        // res = h
        q.memcpy(t.res, t.h, (size_t)hc * (size_t)dim * sizeof(float));

        ds41_hc_mixes_launch(q, t.h, (const float *)wp(dev, ds41_layer_of(L).hc_attn_fn.data),
                             (const float *)wp(dev, ds41_layer_of(L).hc_attn_scale),
                             (const float *)wp(dev, ds41_layer_of(L).hc_attn_base), t.mix_mixes, t.mix_partials,
                             t.a_pre, t.a_post, t.a_comb, hc, dim, d.hc_sinkhorn, d.hc_eps);
        ds41_hc_pre_launch(q, t.h, t.pre_prev, t.mixx, hc, dim);
        rms(dev, t.mixx, L.attn_norm, t.x, dim);
        if (dbg) {
            q.memcpy(sstage, t.x, (size_t)dim * sizeof(float)).wait();
            double ss = 0; for (int i=0;i<dim;i++) ss += (double)sstage[i]*sstage[i];
            fprintf(stderr, "[ds41g]  L%d x rms %.6f\n", li, std::sqrt(ss/(double)dim));
        }
        if (!abl_noattn) {
            double a0 = 0;
            if (timing >= 2) {
                Q(dev).wait();
                a0 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start0).count();
            }
            attention(li, dev, t.x, t.ob);
            if (timing >= 2) {
                Q(dev).wait();
                tm_prof.attn += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start0).count() - a0;
            }
        }
        if (dbg) {
            q.memcpy(sstage, t.ob, (size_t)dim * sizeof(float)).wait();
            double ss = 0; for (int i=0;i<dim;i++) ss += (double)sstage[i]*sstage[i];
            fprintf(stderr, "[ds41g]  L%d attn out rms %.6f\n", li, std::sqrt(ss/(double)dim));
        }
        ds41_hc_post_launch(q, t.ob, t.res, t.a_post, t.a_comb, t.h, hc, dim);

        // res = h (post-attention)
        q.memcpy(t.res, t.h, (size_t)hc * (size_t)dim * sizeof(float));
        ds41_hc_mixes_launch(q, t.h, (const float *)wp(dev, ds41_layer_of(L).hc_ffn_fn.data),
                             (const float *)wp(dev, ds41_layer_of(L).hc_ffn_scale),
                             (const float *)wp(dev, ds41_layer_of(L).hc_ffn_base), t.mix_mixes, t.mix_partials,
                             t.f_pre, t.f_post, t.f_comb, hc, dim, d.hc_sinkhorn, d.hc_eps);
        ds41_hc_pre_launch(q, t.h, t.a_pre, t.mixx, hc, dim);
        rms(dev, t.mixx, L.post_attn_norm, t.x, dim);
        moe_vl = is_img;
        if (!abl_nomoe) {
            double m0 = 0;
            if (timing >= 2) {
                Q(dev).wait();
                m0 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start0).count();
            }
            moe(li, dev, t.x, t.y);
            if (timing >= 2) {
                Q(dev).wait();
                tm_prof.moe += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start0).count() - m0;
            }
        }
        moe_vl = false;
        if (dbg) {
            q.memcpy(sstage, t.y, (size_t)dim * sizeof(float)).wait();
            double ss = 0; for (int i=0;i<dim;i++) ss += (double)sstage[i]*sstage[i];
            fprintf(stderr, "[ds41g]  L%d moe out rms %.6f\n", li, std::sqrt(ss/(double)dim));
        }
        ds41_hc_post_launch(q, t.y, t.res, t.f_post, t.f_comb, t.h, hc, dim);
        q.memcpy(t.pre_prev, t.f_pre, (size_t)hc * sizeof(float));
        if (dbg) {
            q.memcpy(sstage, t.h, (size_t)dim * sizeof(float)).wait();
            double ss = 0;
            for (int i = 0; i < dim; i++) ss += (double)sstage[i] * sstage[i];
            fprintf(stderr, "[ds41g] layer %2d rms(h[s0]) %.6f\n", li, std::sqrt(ss / (double)dim));
        }
    }
    pos++;

    if (!want_logits) {
        if (timing) {
            Q(cur).wait();
            // Count prefill tokens too: before this the average only covered
            // the logits steps and a prompt-only run printed a bogus rate.
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
            tm.total += ms;
            tm.tok++;
            if (timing >= 2) {
                tm.attn = tm_prof.attn;
                tm.moe = tm_prof.moe;
            }
        }
        return;
    }
    if (cur != 0) {
        handoff(cur, 0);
    }
    if (!abl_nohead) {
        double h0ms = 0;
        if (timing >= 2) {
            Q(0).wait();
            h0ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
        }
        head(0);
        if (timing >= 2) {
            Q(0).wait();
            tm.head += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count() -
                       h0ms;
        }
    }
    devbuf & t = DV(0);
    Q(0).memcpy(logits_host.data(), t.logits, (size_t)hp.n_vocab * sizeof(float)).wait();
    if (timing) {
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
        tm.total += ms;
        tm.tok++;
        if (timing >= 2) {
            tm.attn = tm_prof.attn;
            tm.moe = tm_prof.moe;
        }
    }
}

} // namespace si

// ===========================================================================
// Public entry points
// ===========================================================================
namespace si {

ds41_gpu_state * ds41_gpu_create(engine & e) {
    try {
        return new ds41_gpu_state(e);
    } catch (const std::exception & ex) {
        fprintf(stderr, "[ds41] GPU forward unavailable: %s (falling back to the host forward)\n", ex.what());
        return nullptr;
    }
}

void ds41_gpu_destroy(ds41_gpu_state * st) { delete st; }

std::vector<int> ds41_gpu_generate(model & m, tokenizer & tk, const std::vector<int> & prompt, const mm_prompt * mm,
                                   const gen_params & gp, const std::function<bool(int)> & cb,
                                   std::vector<float> * first_logits, size_t * cached_tokens,
                                   ds41_gpu_state & st) {
    if (cached_tokens) {
        *cached_tokens = 0;
    }
    st.reset_state();
    std::vector<int> ids = prompt;
    std::vector<int> gen;
    sampler_state ss;
    ss.seed(gp.seed ? gp.seed : std::random_device{}());
    st.logits_host.assign((size_t)m.hp.n_vocab, 0.f);

    st.calib = true;
    for (size_t i = 0; i < ids.size(); i++) {
        st.forward_token(ids[i], i + 1 == ids.size(), mm);
    }
    if (si::env::i32("PF_DS41_CAL", 0) != 0) {
        st.calibrate();
    }
    st.calib = false;
    if (first_logits) {
        *first_logits = st.logits_host;
    }
    // PF_DUMP_DEC_LOGITS=<prefix>: the same per-step logit dump the generic
    // decode loop writes (prefix.<step>.bin, step 0 = the prompt).  It is what
    // separates "the store moved a near-tie" from "the store is wrong".
    const char * dbg_lg = si::env::str("PF_DUMP_DEC_LOGITS");
    auto dump_logits = [&](int step) {
        if (!dbg_lg) {
            return;
        }
        char path[1024];
        snprintf(path, sizeof(path), "%s.%d.bin", dbg_lg, step);
        FILE * f = fopen(path, "wb");
        if (f) {
            fwrite(st.logits_host.data(), sizeof(float), st.logits_host.size(), f);
            fclose(f);
        }
    };
    const float * logits = st.logits_host.data();
    dump_logits(0);
    for (int step = 0; step < gp.max_tokens; step++) {
        const int tok = sample_token(logits, (int)st.logits_host.size(), gp, ids, ss);
        if (!gp.ignore_eos && (tok == tk.eos_id || tok == 1)) {
            break;
        }
        gen.push_back(tok);
        if (cb && !cb(tok)) {
            break;
        }
        ids.push_back(tok);
        st.forward_token(tok, true, mm);
        logits = st.logits_host.data();
        dump_logits(step + 1);
    }
    if (st.timing) {
        fprintf(stderr, "[ds41g] %llu forwards, %.2f ms/token total", (unsigned long long)st.tm.tok,
                st.tm.tok ? st.tm.total / (double)st.tm.tok : 0.0);
        if (st.timing >= 2 && st.tm.tok) {
            fprintf(stderr, ", attn %.2f + moe %.2f + head %.2f ms/token",
                    st.tm.attn / (double)st.tm.tok, st.tm.moe / (double)st.tm.tok,
                    st.tm.head / (double)st.tm.tok);
        }
        fprintf(stderr, "\n");
    }
    if (st.stats) {
        size_t up = 0;
        for (const auto & b : st.banks) {
            up += b.uploads;
        }
        uint64_t hits = 0, misses = 0;
        size_t mb = 0;
        for (const auto & b : st.banks) {
            hits += b.hits;
            misses += b.misses;
            mb += b.miss_bytes;
        }
        const double total = (double)(hits + misses);
        fprintf(stderr, "[ds41g] expert plane uploads=%zu bank hit=%.1f%% (%llu/%llu) miss=%.1f MiB\n", up,
                total > 0 ? 100.0 * (double)hits / total : 0.0,
                (unsigned long long)hits, (unsigned long long)total, (double)mb / 1048576.0);
    }
    return gen;
}

} // namespace si
