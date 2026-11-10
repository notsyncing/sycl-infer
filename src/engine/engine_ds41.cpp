// dev/ds41_gen.cpp — standalone CPU reference forward + greedy generation for
// DeepSeek-V4.1-Flash (`deepseek41`).
//
// It loads the GGUF through the project loader, dequantizes the dense weights
// to f32 once into RAM (the routed experts stay in the mmap and are streamed
// one row at a time), and runs the *whole-sequence* forward the authoritative
// torch oracle (Strata tools/dsv41/ref.py) describes:
//
//   mHC lagged pre-mix, 128-token sliding window + group-shared compressed KV,
//   the two-level DSA indexer with candidate blocks, Engram, sqrt-softplus MoE.
//
// Correctness first: every step is f32 and follows ref.py's order as closely as
// a scalar CPU can.  The fp8/fp4 activation roundings are implemented
// (PF_DS41_NOQ=1 disables them for debugging).
//
// The default path is an *incremental* decoder: the prompt is consumed once
// into a persistent per-sequence state (window K rings, the group-shared
// compressed caches and their DSA index keys, the Engram token history) and
// each generated token costs one token's forward instead of the whole prefix.
// `--naive` keeps the original whole-sequence forward as the oracle and
// `--check` runs both and reports the per-step logits delta.
//
// usage: ds41_gen <gguf> [--prompt TEXT] [--ids a,b,c] [--max-tokens N]
//                 [--naive] [--check] [--ignore-eos]
// env: PF_DS41_NOQ, PF_DS41_DEBUG, PF_DS41_BF16EMB, PF_DS41_ENG_BF16,
//      PF_DS41_THREADS, PF_DS41_IQ3XXS (local IQ3_XXS dequant)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "model.h"
#include "model_arch.h"
#include "quant.h"
#include "tokenizer.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "engine.h"
#include "ds41_prefix_cache.h"
#include "sampler.h"
#include "common/env.h"

namespace si {
namespace {

using namespace si;

// ---------------------------------------------------------------- flags
static bool g_noq = false;      // disable the fp8/fp4 activation roundings
static bool g_dbg = false;      // per-layer diagnostics
static bool g_iq3xxs = false;   // use the local IQ3_XXS dequant (oracle lacks it)
static bool g_bf16_emb = false; // round the token embedding to bf16 (torch path)
static bool g_eng_bf16 = false; // round the Engram gather to bf16 (torch path)
static int g_nt = 24;           // worker threads

// Multimodal text-side view for the current request (set by ds41_generate).
// Prompt positions equal token indices (deepseek41 uses plain 1-D positions), so
// `img_row[pos]` selects the vision embedding row for that token.
struct MmView {
    const float * embd = nullptr;
    const int32_t * img_row = nullptr;
    int n_embd = 0;
    int n_tok = 0;
};
static MmView g_mm;
static bool g_moe_vl = false; // the current token is an image row: VL router bias



// ---------------------------------------------------------------- scalar helpers
static inline float sigmoidf_(float x) { return 1.f / (1.f + std::exp(-x)); }
static inline float silu_(float x) { return x / (1.f + std::exp(-x)); }
static inline float softplus_(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }

// 4/8-lane dot; the reduction order differs from the reference, which is
// unavoidable on a CPU and irrelevant to the argmax at this margin.
static inline float dot_f32(const float * a, const float * b, int n) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
    }
    __m256 acc = _mm256_add_ps(a0, a1);
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    __m128 sm = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, sm);
    sm = _mm_add_ss(sm, sh);
    float s = _mm_cvtss_f32(sm);
    for (; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
#else
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; i++) {
        s0 += a[i] * b[i];
    }
    return (s0 + s1) + (s2 + s3);
#endif
}

static inline float sumsq_f32(const float * a, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) {
        s += (double)a[i] * (double)a[i];
    }
    return (float)s;
}

// `y = x * rsqrt(mean(x^2) + eps) * w` over the last axis (n values).
static void rms_w(const float * x, const float * w, int n, float eps, float * y) {
    float r = 1.f / std::sqrt(sumsq_f32(x, n) / (float)n + eps);
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * r * w[i];
    }
}

static void rms_plain(const float * x, int n, float eps, float * y) {
    float r = 1.f / std::sqrt(sumsq_f32(x, n) / (float)n + eps);
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * r;
    }
}

static inline float bf16_round(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb; // round to nearest even on the 16-bit boundary
    u &= 0xFFFF0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

// ---------------------------------------------------------------- fp8 / fp4 (ref.py:38-79)
// E4M3FN value of the nearest representable number (round-to-nearest-even).
static inline float e4m3_val(float v) {
    bool neg = std::signbit(v);
    float a = std::fabs(v);
    if (!(a > 0.f)) {
        return neg ? -0.0f : 0.0f;
    }
    if (a >= 448.f) {
        return neg ? -448.f : 448.f;
    }
    int e;
    std::frexp(a, &e); // a = m * 2^e, m in [0.5, 1)
    int E = e - 1;     // floor(log2 a)
    float step = (E < -6) ? std::ldexp(1.f, -9) : std::ldexp(1.f, E - 3);
    float q = a / step;
    float r = std::nearbyint(q); // current rounding mode = nearest-even
    float out = r * step;
    if (out > 448.f) {
        out = 448.f;
    }
    return neg ? -out : out;
}

static const float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static inline float e2m1_val(float v) {
    float a = std::fabs(v);
    if (a > 6.f) {
        a = 6.f;
    }
    int best = 0;
    float bd = 1e30f;
    for (int i = 0; i < 8; i++) {
        float d = std::fabs(a - kE2M1[i]);
        if (d < bd) {
            bd = d;
            best = i;
        }
    }
    // ties go to the even mantissa: if the next grid point is equally far and
    // the lower index is odd, take the upper one (ref.py:56-63).
    if (best < 7 && std::fabs(a - kE2M1[best + 1]) == bd && (best & 1)) {
        best++;
    }
    return std::copysign(kE2M1[best], v);
}

// in-place `n` values, `block` per group.
static void fp8_round_row(float * v, int n) {
    for (int b = 0; b < n; b += 32) {
        int m = std::min(32, n - b);
        float amax = 1e-4f;
        for (int i = 0; i < m; i++) {
            amax = std::max(amax, std::fabs(v[b + i]));
        }
        float s = std::exp2(std::ceil(std::log2(amax / 448.f)));
        for (int i = 0; i < m; i++) {
            float x = v[b + i] / s;
            x = std::max(-448.f, std::min(448.f, x));
            v[b + i] = e4m3_val(x) * s;
        }
    }
}

static void fp4_round_row(float * v, int n, int block, bool e4m3_scale) {
    for (int b = 0; b < n; b += block) {
        int m = std::min(block, n - b);
        float amax = 0.f;
        for (int i = 0; i < m; i++) {
            amax = std::max(amax, std::fabs(v[b + i]));
        }
        float s;
        if (e4m3_scale) {
            amax = std::max(amax, 6.f * std::ldexp(1.f, -9));
            s = e4m3_val(amax / 6.f);
        } else {
            amax = std::max(amax, 6.f * std::ldexp(1.f, -126));
            float q = amax / 6.f;
            int e;
            float mant = std::frexp(q, &e); // q = mant*2^e, mant in [0.5,1)
            s = std::ldexp(1.f, mant == 0.5f ? e - 1 : e);
        }
        for (int i = 0; i < m; i++) {
            float x = std::max(-6.f, std::min(6.f, v[b + i] / s));
            v[b + i] = e2m1_val(x) * s;
        }
    }
}


// ---------------------------------------------------------------- IQ3_XXS
// `src/common/quant.h`'s dequantize_row() has no case for ggml type 18
// (IQ3_XXS); a whole-sequence forward would silently reuse the caller's
// buffer for that type.  One routed-down projection of this model is stored
// that way, so the dev tool carries llama.cpp's table and block walk locally
// (src/ is off-limits for this change).  Every other type is handed straight
// to the project's dequantizer, so nothing else changes.
static const uint32_t kIq3xxsGrid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

static inline void dequant_block_iq3_xxs_local(const void * data, float * y) {
    const uint8_t * p = (const uint8_t *)data;
    uint16_t dh;
    std::memcpy(&dh, p, sizeof(dh));
    const float d = ggml_half_to_float(dh);
    const uint8_t * qs = p + 2;
    const uint8_t * scales_and_signs = qs + 256 / 4;
    for (int ib32 = 0; ib32 < 256 / 32; ++ib32) {
        uint32_t aux32;
        std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(aux32));
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const uint8_t * grid1 = (const uint8_t *)(kIq3xxsGrid + qs[2 * l + 0]);
            const uint8_t * grid2 = (const uint8_t *)(kIq3xxsGrid + qs[2 * l + 1]);
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                y[j + 4] = db * grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
            }
            y += 8;
        }
        qs += 8;
    }
}

static void dequant_row_local(uint32_t type, const void * data, float * y, int64_t n) {
    if (type == 18) { // IQ3_XXS: 2 + 3*(QK_K/8) = 98 bytes per 256 values
        for (int64_t i = 0; i < n / 256; i++) {
            dequant_block_iq3_xxs_local((const uint8_t *)data + (size_t)i * 98, y + i * 256);
        }
        return;
    }
    dequantize_row(type, data, y, n);
}

// ---------------------------------------------------------------- worker pool
struct Pool {
    int nw;
    std::vector<std::thread> ths;
    std::mutex mu;
    std::condition_variable cv_start, cv_done;
    std::function<void(int)> fn;
    uint64_t gen = 0;
    int pending = 0;
    bool stop = false;

    explicit Pool(int n) : nw(std::max(1, n)) {
        for (int i = 0; i < nw; i++) {
            ths.emplace_back([this, i] { loop(i); });
        }
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
            gen++;
        }
        cv_start.notify_all();
        for (auto & t : ths) {
            t.join();
        }
    }
    void loop(int id) {
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(mu);
            cv_start.wait(lk, [&] { return stop || gen != seen; });
            if (stop) {
                return;
            }
            seen = gen;
            auto f = fn;
            lk.unlock();
            f(id);
            lk.lock();
            if (--pending == 0) {
                cv_done.notify_all();
            }
        }
    }
    void run(const std::function<void(int)> & f) {
        std::unique_lock<std::mutex> lk(mu);
        fn = f;
        pending = nw;
        gen++;
        cv_start.notify_all();
        cv_done.wait(lk, [&] { return pending == 0; });
    }
};

template <class F>
static void par_for(Pool & p, int n, F && f) {
    if (n <= 0) {
        return;
    }
    int nw = p.nw;
    p.run([&f, n, nw](int id) {
        for (int i = id; i < n; i += nw) {
            f(i);
        }
    });
}

// ---------------------------------------------------------------- dense weight cache
struct Dense {
    std::unordered_map<const void *, std::vector<float>> map;

    const float * get(const wt & w) {
        if (!w.data || w.N <= 0 || w.K <= 0) {
            return nullptr;
        }
        if (w.type == 0) {
            return (const float *)w.data; // already f32 in the mmap
        }
        auto it = map.find(w.data);
        if (it != map.end()) {
            return it->second.data();
        }
        const size_t n = (size_t)w.K * (size_t)w.N;
        std::vector<float> buf(n);
        const size_t rb = ggml_row_bytes(w.type, (uint64_t)w.K);
        const char * base = (const char *)w.data;
        int nt = std::min<int>(g_nt, std::max(1, w.N));
        std::vector<std::thread> th;
        for (int i = 0; i < nt; i++) {
            th.emplace_back([&, i] {
                for (int r = i; r < w.N; r += nt) {
                    dequant_row_local(w.type, base + (size_t)r * rb, buf.data() + (size_t)r * (size_t)w.K,
                                   w.K);
                }
            });
        }
        for (auto & t : th) {
            t.join();
        }
        auto ins = map.emplace(w.data, std::move(buf));
        return ins.first->second.data();
    }
};

// ---------------------------------------------------------------- model context
struct Ctx {
    model * m = nullptr;
    const hparams * hp = nullptr;
    const ds41_hparams * d = nullptr;
    Dense dense;
    Pool * pool = nullptr;
    float freq_win[32];
    float freq_comp[32];
    int rd = 64;

    const float * W(const wt & w) { return dense.get(w); }
};

static float meta_f32(const gguf_file & f, const char * k, float def) {
    std::string key = std::string("deepseek41.") + k;
    const gguf_kv * v = f.meta(key);
    if (!v) {
        return def;
    }
    float out = (float)v->f64;
    return out != 0.f ? out : def;
}

static void build_freqs(float * fr, int rd, float base, int orig, float factor, float bf, float bs) {
    for (int j = 0; j < rd / 2; j++) {
        fr[j] = 1.f / std::pow(base, 2.f * (float)j / (float)rd);
    }
    if (orig > 0) {
        auto corr = [&](float rot) {
            return (float)rd * std::log((float)orig / (rot * 2.f * (float)M_PI)) /
                   (2.f * std::log(base));
        };
        int low = std::max((int)std::floor(corr(bf)), 0);
        int high = std::min((int)std::ceil(corr(bs)), rd - 1);
        for (int j = 0; j < rd / 2; j++) {
            float ramp = ((float)j - (float)low) / std::max((float)(high - low), 1e-3f);
            ramp = std::min(1.f, std::max(0.f, ramp));
            float smooth = 1.f - ramp;
            fr[j] = fr[j] / factor * (1.f - smooth) + fr[j] * smooth;
        }
    }
}

static inline void rope_one(float * v, int off, int rd, int pos, const float * fr, bool inverse) {
    for (int j = 0; j < rd / 2; j++) {
        float ang = (float)pos * fr[j];
        float c = std::cos(ang), s = std::sin(ang);
        if (inverse) {
            s = -s;
        }
        float x0 = v[off + 2 * j], x1 = v[off + 2 * j + 1];
        v[off + 2 * j] = x0 * c - x1 * s;
        v[off + 2 * j + 1] = x0 * s + x1 * c;
    }
}

// y[t*N+n] = dot(X[t*K..], W[n*K..]); X row stride = K, W row stride = K.
static void linear(Ctx & c, const float * W, int N, int K, const float * X, int T, int xstride,
                   float * Y, int ystride) {
    par_for(*c.pool, N, [&](int n) {
        const float * wr = W + (size_t)n * (size_t)K;
        for (int t = 0; t < T; t++) {
            Y[(size_t)t * (size_t)ystride + (size_t)n] = dot_f32(wr, X + (size_t)t * (size_t)xstride, K);
        }
    });
}

static void linear_wt(Ctx & c, const wt & w, const float * X, int T, float * Y) {
    const float * W = c.W(w);
    linear(c, W, w.N, w.K, X, T, w.K, Y, w.N);
}

// ---------------------------------------------------------------- hyper-connections
// h: [T][hc][d]; fn: [24][hc*d]; scale[3]; base[24]
static void hc_mixes(Ctx & c, const float * h, int T, int hc, int d, const float * fn,
                     const float * scale, const float * base, float eps, int iters,
                     std::vector<float> & pre, std::vector<float> & post,
                     std::vector<float> & comb) {
    const int fd = hc * d;
    const int nm = (2 + hc) * hc;
    pre.assign((size_t)T * hc, 0.f);
    post.assign((size_t)T * hc, 0.f);
    comb.assign((size_t)T * hc * hc, 0.f);

    std::vector<float> mixes((size_t)T * nm);
    par_for(*c.pool, T, [&](int t) {
        const float * flat = h + (size_t)t * fd;
        float r = 1.f / std::sqrt(sumsq_f32(flat, fd) / (float)fd + 1e-20f);
        for (int m = 0; m < nm; m++) {
            mixes[(size_t)t * nm + m] = dot_f32(flat, fn + (size_t)m * fd, fd) * r;
        }
    });

    for (int t = 0; t < T; t++) {
        const float * mx = mixes.data() + (size_t)t * nm;
        for (int s = 0; s < hc; s++) {
            pre[(size_t)t * hc + s] = sigmoidf_(mx[s] * scale[0] + base[s]) + eps;
            post[(size_t)t * hc + s] = 2.f * sigmoidf_(mx[hc + s] * scale[1] + base[hc + s]);
        }
        float * C = comb.data() + (size_t)t * hc * hc; // [j][k]
        for (int j = 0; j < hc; j++) {
            float * row = C + (size_t)j * hc;
            float m = -1e30f;
            for (int k = 0; k < hc; k++) {
                row[k] = mx[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
                m = std::max(m, row[k]);
            }
            float sum = 0;
            for (int k = 0; k < hc; k++) {
                row[k] = std::exp(row[k] - m);
                sum += row[k];
            }
            for (int k = 0; k < hc; k++) {
                row[k] = row[k] / sum + eps;
            }
        }
        for (int k = 0; k < hc; k++) { // column (destination) normalisation
            float s = eps;
            for (int j = 0; j < hc; j++) {
                s += C[(size_t)j * hc + k];
            }
            for (int j = 0; j < hc; j++) {
                C[(size_t)j * hc + k] /= s;
            }
        }
        for (int it = 0; it < iters - 1; it++) {
            for (int j = 0; j < hc; j++) { // row
                float s = eps;
                for (int k = 0; k < hc; k++) {
                    s += C[(size_t)j * hc + k];
                }
                for (int k = 0; k < hc; k++) {
                    C[(size_t)j * hc + k] /= s;
                }
            }
            for (int k = 0; k < hc; k++) { // column
                float s = eps;
                for (int j = 0; j < hc; j++) {
                    s += C[(size_t)j * hc + k];
                }
                for (int j = 0; j < hc; j++) {
                    C[(size_t)j * hc + k] /= s;
                }
            }
        }
    }
}

static void hc_pre(const float * h, const float * mix, int T, int hc, int d, float * y) {
    for (int t = 0; t < T; t++) {
        float * dst = y + (size_t)t * d;
        for (int e = 0; e < d; e++) {
            dst[e] = 0;
        }
        for (int s = 0; s < hc; s++) {
            float w = mix[(size_t)t * hc + s];
            const float * src = h + ((size_t)t * hc + s) * d;
            for (int e = 0; e < d; e++) {
                dst[e] += w * src[e];
            }
        }
    }
}

static void hc_post(const float * y, const float * res, const float * post, const float * comb, int T,
                    int hc, int d, float * out) {
    for (int t = 0; t < T; t++) {
        const float * yy = y + (size_t)t * d;
        const float * rr = res + (size_t)t * hc * d;
        const float * C = comb + (size_t)t * hc * hc;
        for (int k = 0; k < hc; k++) {
            float * dst = out + ((size_t)t * hc + k) * d;
            float pk = post[(size_t)t * hc + k];
            for (int e = 0; e < d; e++) {
                dst[e] = pk * yy[e];
            }
            for (int j = 0; j < hc; j++) {
                float cj = C[(size_t)j * hc + k];
                const float * rj = rr + (size_t)j * d;
                for (int e = 0; e < d; e++) {
                    dst[e] += cj * rj[e];
                }
            }
        }
    }
}

// ---------------------------------------------------------------- Engram
static void engram_hash(Ctx & c, const std::vector<int> & ids, int T, int e, std::vector<int> & out) {
    const ds41_hparams & d = *c.d;
    const int ncol = d.ple_n_heads; // 24
    const int * tm = c.m->engram_token_map.data();
    int pad = tm[d.engram_pad];
    out.assign((size_t)T * ncol, 0);
    for (int p = 0; p < T; p++) {
        int64_t tok[4];
        for (int s = 0; s < d.ple_ngram; s++) {
            int q = p - s;
            tok[s] = (q >= 0) ? (int64_t)tm[ids[(size_t)q]] : (int64_t)pad;
        }
        int64_t rolling = (int64_t)((uint64_t)tok[0] * (uint64_t)d.ple_mult[(size_t)e * 4 + 0]);
        for (int o = 1; o < d.ple_ngram; o++) {
            int64_t prod = (int64_t)((uint64_t)tok[o] * (uint64_t)d.ple_mult[(size_t)e * 4 + o]);
            rolling = (int64_t)((uint64_t)rolling ^ (uint64_t)prod);
            for (int hj = 0; hj < d.ple_heads_per_ngram; hj++) {
                int col = (o - 1) * d.ple_heads_per_ngram + hj;
                int64_t voc = (int64_t)d.ple_vocab[(size_t)e * ncol + col];
                int64_t r = rolling % voc;
                if (r < 0) {
                    r += voc;
                }
                out[(size_t)p * ncol + col] = (int)(r + (int64_t)d.ple_off[(size_t)e * ncol + col]);
            }
        }
    }
}

// The hash for a single new token: `hist[0..hist_n)` are the *mapped* token ids
// of the current token and its predecessors (most recent first).  Positions
// before the first token are replaced by the mapped pad id, exactly as
// engram_hash does for q < 0.
static void engram_hash_one(Ctx & c, const int64_t * hist, int hist_n, int e,
                            std::vector<int> & out) {
    const ds41_hparams & d = *c.d;
    const int ncol = d.ple_n_heads;
    const int ng = std::min(d.ple_ngram, 8);
    const int64_t pad =
        c.m->engram_token_map.empty() ? 0 : c.m->engram_token_map[(size_t)d.engram_pad];
    int64_t tok[8];
    for (int s = 0; s < ng; s++) {
        tok[s] = (s < hist_n) ? hist[s] : pad;
    }
    out.assign((size_t)ncol, 0);
    int64_t rolling = (int64_t)((uint64_t)tok[0] * (uint64_t)d.ple_mult[(size_t)e * 4 + 0]);
    for (int o = 1; o < ng; o++) {
        int64_t prod = (int64_t)((uint64_t)tok[o] * (uint64_t)d.ple_mult[(size_t)e * 4 + o]);
        rolling = (int64_t)((uint64_t)rolling ^ (uint64_t)prod);
        for (int hj = 0; hj < d.ple_heads_per_ngram; hj++) {
            int col = (o - 1) * d.ple_heads_per_ngram + hj;
            int64_t voc = (int64_t)d.ple_vocab[(size_t)e * ncol + col];
            int64_t r = rolling % voc;
            if (r < 0) {
                r += voc;
            }
            out[(size_t)col] = (int)(r + (int64_t)d.ple_off[(size_t)e * ncol + col]);
        }
    }
}

static void apply_engram_rows(Ctx & c, int li, std::vector<float> & h, int T,
                              const std::vector<int> & rows) {
    const ds41_hparams & d = *c.d;
    const hparams & hp = *c.hp;
    const int hc = d.hc_mult, dim = hp.n_embd;
    const layer_t & L = c.m->layers[(size_t)li];
    const int ncol = d.ple_n_heads;
    const int hd = d.ple_head_dim; // 256

    // Gather the 24 rows a token needs.
    const wt & tab = c.m->ple_table;
    const size_t rb = ggml_row_bytes(tab.type, (uint64_t)tab.K);
    const char * base = (const char *)tab.data;
    std::vector<float> ev((size_t)T * ncol * hd, 0.f);
    for (int t = 0; t < T; t++) {
        for (int col = 0; col < ncol; col++) {
            int row = rows[(size_t)t * ncol + col];
            float * dst = ev.data() + ((size_t)t * ncol + col) * hd;
            dequant_row_local(tab.type, base + (size_t)row * rb, dst, hd);
        }
    }
    if (g_eng_bf16) {
        for (auto & v : ev) {
            v = bf16_round(v);
        }
    }

    // kv = e @ engram_kv^T  -> [T][(hc+1)*dim]
    const int Kj = ncol * hd;          // 6144
    const int Nj = (hc + 1) * dim;     // 25600
    std::vector<float> kv((size_t)T * Nj, 0.f);
    const wt & ekv = L.ds.engram_kv;
    const float * Wk = c.W(ekv);
    linear(c, Wk, ekv.N, ekv.K, ev.data(), T, Kj, kv.data(), Nj);

    const float * gate_w = L.ds.engram_gate; // [hc][dim]
    for (int t = 0; t < T; t++) {
        const float * kvt = kv.data() + (size_t)t * Nj;
        const float * key = kvt;
        const float * value = kvt + (size_t)hc * dim;
        for (int k = 0; k < hc; k++) {
            const float * hk = h.data() + ((size_t)t * hc + k) * dim;
            const float * kk = key + (size_t)k * dim;
            const float * wg = gate_w + (size_t)k * dim;
            float rstd = 1.f / std::sqrt(sumsq_f32(hk, dim) / (float)dim + 1e-20f) *
                         1.f / std::sqrt(sumsq_f32(kk, dim) / (float)dim + 1e-20f);
            float sd = 0;
            for (int x = 0; x < dim; x++) {
                sd += hk[x] * wg[x] * kk[x];
            }
            float dot = sd * rstd / std::sqrt((float)dim);
            float g = sigmoidf_(std::copysign(std::sqrt(std::max(std::fabs(dot), 1e-6f)), dot));
            float * dst = h.data() + ((size_t)t * hc + k) * dim;
            for (int x = 0; x < dim; x++) {
                dst[x] += g * value[x];
            }
        }
    }
}

static void apply_engram(Ctx & c, int li, std::vector<float> & h, const std::vector<int> & ids, int T) {
    const int e = c.d->engram_index_of[li];
    std::vector<int> rows;
    engram_hash(c, ids, T, e, rows);
    apply_engram_rows(c, li, h, T, rows);
}

// ---------------------------------------------------------------- sparse attention
// q: [T][H][D]; kv_all: [Nk][D]; widx: [T][W]; sel: [T][K]; sink: [H]
static void sparse_attn(const float * q, int T, int H, int D, const float * kv_all, int nk,
                        const int * widx, int W, const int * sel, int K, const float * sink,
                        float scale, float * out) {
    for (int t = 0; t < T; t++) {
        const int * si = sel + (size_t)t * K;
        const int * wi = widx + (size_t)t * W;
        for (int h = 0; h < H; h++) {
            const float * qh = q + ((size_t)t * H + h) * D;
            float m = sink[h];
            for (int j = 0; j < W + K; j++) {
                int idx = (j < W) ? wi[j] : si[j - W];
                if (idx < 0 || idx >= nk) {
                    continue;
                }
                float s = dot_f32(qh, kv_all + (size_t)idx * D, D) * scale;
                m = std::max(m, s);
            }
            m = std::max(m, -1e30f);
            float den = std::exp(sink[h] - m);
            float acc[512];
            for (int d = 0; d < D; d++) {
                acc[d] = 0.f;
            }
            for (int j = 0; j < W + K; j++) {
                int idx = (j < W) ? wi[j] : si[j - W];
                if (idx < 0 || idx >= nk) {
                    continue;
                }
                const float * v = kv_all + (size_t)idx * D;
                float p = std::exp(dot_f32(qh, v, D) * scale - m);
                den += p;
                for (int d = 0; d < D; d++) {
                    acc[d] += p * v[d];
                }
            }
            float inv = den > 0.f ? 1.f / den : 0.f;
            float * od = out + ((size_t)t * H + h) * D;
            for (int d = 0; d < D; d++) {
                od[d] = acc[d] * inv;
            }
        }
    }
}

// ---------------------------------------------------------------- forward
struct SState {
    std::vector<float> ckv; // [rows][512]
    int ckv_rows = 0;
    std::vector<float> ik; // [rows][128]
    int ik_rows = 0;
    std::vector<int> sel; // [T][k]
    int sel_k = 0;
    std::vector<uint8_t> cand; // [T][nb] block keep mask
    int cand_clen = 0;
    int cand_nb = 0;
    bool has_cand = false;
};

struct Fwd {
    Ctx * c;
    const hparams * hp;
    const ds41_hparams * d;
    int T;
    std::vector<int> ids;

    // --- attention
    void attention(int li, const float * x, SState & S, float * out_o);
    void indexer(int li, const float * x, const float * qr, const float * latent, int clen, int ratio,
                 SState & S);
    void compressor(int li, const float * x, int ratio, std::vector<float> & latent, int & clen);
    void moe(int li, const float * x, float * y);
};

static void compute_candidates(const float * score, int T, int clen, int ratio, int block,
                               int topk_blocks, std::vector<uint8_t> & keep, int & nb_out) {
    int nb = (clen + block - 1) / block;
    nb_out = nb;
    keep.assign((size_t)T * nb, 0);
    std::vector<float> bmax((size_t)nb);
    std::vector<int> ord;
    for (int t = 0; t < T; t++) {
        int lens = (t + 1) / ratio;
        for (int b = 0; b < nb; b++) {
            float m = -INFINITY;
            int hi = std::min(clen, (b + 1) * block);
            for (int c = b * block; c < hi; c++) {
                m = std::max(m, score[(size_t)t * clen + c]);
            }
            bmax[(size_t)b] = m;
        }
        int last = lens > 0 ? (lens - 1) / block : -1; // floor((lens-1)/block), torch floor_divide
        if (last >= 0 && last < nb) {
            bmax[(size_t)last] = INFINITY;
        }
        ord.resize((size_t)nb);
        for (int b = 0; b < nb; b++) {
            ord[(size_t)b] = b;
        }
        int take = std::min(topk_blocks, nb);
        std::partial_sort(ord.begin(), ord.begin() + take, ord.end(), [&](int a, int b) {
            if (bmax[(size_t)a] != bmax[(size_t)b]) {
                return bmax[(size_t)a] > bmax[(size_t)b];
            }
            return a < b;
        });
        for (int i = 0; i < take; i++) {
            int b = ord[(size_t)i];
            if (bmax[(size_t)b] > -INFINITY) {
                keep[(size_t)t * nb + b] = 1;
            }
        }
    }
}

void Fwd::compressor(int li, const float * x, int ratio, std::vector<float> & latent, int & clen) {
    const int D = d->kv_dim;
    const layer_t & L = c->m->layers[(size_t)li];
    clen = T / ratio;
    latent.assign((size_t)std::max(clen, 1) * D, 0.f);
    if (clen == 0) {
        return;
    }
    if (ratio == 1) {
        std::vector<float> tmp((size_t)T * D);
        linear_wt(*c, L.ds.comp_wkv, x, T, tmp.data());
        for (int t = 0; t < T; t++) {
            rms_w(tmp.data() + (size_t)t * D, L.ds.comp_norm, D, 1e-20f,
                  latent.data() + (size_t)t * D);
        }
        return;
    }
    const int cut = T - T % ratio;
    std::vector<float> kvc((size_t)cut * D), sc((size_t)cut * D);
    linear_wt(*c, L.ds.comp_wkv, x, cut, kvc.data());
    linear_wt(*c, L.ds.comp_wgate, x, cut, sc.data());
    for (int g = 0; g < clen; g++) {
        float * dst = latent.data() + (size_t)g * D;
        for (int e = 0; e < D; e++) {
            float m = -INFINITY;
            for (int j = 0; j < ratio; j++) {
                m = std::max(m, sc[(size_t)(g * ratio + j) * D + e]);
            }
            float sum = 0;
            for (int j = 0; j < ratio; j++) {
                sum += std::exp(sc[(size_t)(g * ratio + j) * D + e] - m);
            }
            float acc = 0;
            for (int j = 0; j < ratio; j++) {
                float p = std::exp(sc[(size_t)(g * ratio + j) * D + e] - m) / sum;
                acc += p * kvc[(size_t)(g * ratio + j) * D + e];
            }
            dst[e] = acc;
        }
        rms_w(dst, L.ds.comp_norm, D, 1e-20f, dst);
    }
}

void Fwd::indexer(int li, const float * x, const float * qr, const float * latent, int clen,
                  int ratio, SState & S) {
    const layer_t & L = c->m->layers[(size_t)li];
    const int nh = d->idx_n_head, hdim = d->idx_head_dim; // 32,128
    const int rd = 64;
    const int off = hdim - rd; // 64
    if (clen == 0) {
        return;
    }
    // own index keys (KV sources only)
    if (d->kv_src_of[li] == li && latent) {
        S.ik.assign((size_t)clen * hdim, 0.f);
        std::vector<float> tmp((size_t)clen * hdim);
        linear_wt(*c, L.ds.idx_attn_k, latent, clen, tmp.data());
        for (int g = 0; g < clen; g++) {
            float * row = tmp.data() + (size_t)g * hdim;
            rms_w(row, L.ds.idx_k_norm, hdim, 1e-20f, row);
            rope_one(row, off, rd, g * ratio, c->freq_comp, false);
            if (!g_noq) {
                fp4_round_row(row, hdim, 32, false);
            }
            std::memcpy(S.ik.data() + (size_t)g * hdim, row, sizeof(float) * hdim);
        }
        S.ik_rows = clen;
    }
    // index query from qr
    std::vector<float> qi((size_t)T * nh * hdim);
    linear_wt(*c, L.ds.idx_attn_q_b, qr, T, qi.data());
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < nh; h++) {
            float * row = qi.data() + ((size_t)t * nh + h) * hdim;
            rope_one(row, off, rd, t, c->freq_comp, false);
            if (!g_noq) {
                fp4_round_row(row, hdim, 32, false);
            }
        }
    }
    // per-head weight
    std::vector<float> w((size_t)T * nh);
    linear_wt(*c, L.ds.idx_proj, x, T, w.data());
    const float wscale = 1.f / std::sqrt((float)hdim * (float)nh);
    for (auto & v : w) {
        v *= wscale;
    }

    std::vector<float> score((size_t)T * clen);
    const int ik_rows = S.ik_rows;
    for (int t = 0; t < T; t++) {
        int lens = (t + 1) / ratio;
        for (int cc = 0; cc < clen; cc++) {
            float acc = 0;
            if (cc < lens && cc < ik_rows) {
                const float * ik = S.ik.data() + (size_t)cc * hdim;
                for (int h = 0; h < nh; h++) {
                    float s = dot_f32(qi.data() + ((size_t)t * nh + h) * hdim, ik, hdim);
                    if (s > 0) {
                        acc += s * w[(size_t)t * nh + h];
                    }
                }
                score[(size_t)t * clen + cc] = acc;
            } else {
                score[(size_t)t * clen + cc] = -INFINITY;
            }
        }
    }

    if (li == d->cand_src) {
        compute_candidates(score.data(), T, clen, ratio, d->cand_block, d->cand_topk_blocks, S.cand,
                           S.cand_nb);
        S.cand_clen = clen;
        S.has_cand = true;
    } else if (d->cand_src >= 0 && li > d->cand_src && S.has_cand) {
        for (int t = 0; t < T; t++) {
            for (int cc = 0; cc < clen; cc++) {
                int b = cc / d->cand_block;
                if (b >= S.cand_nb || !S.cand[(size_t)t * S.cand_nb + b]) {
                    score[(size_t)t * clen + cc] = -INFINITY;
                }
            }
        }
    }

    const int k = std::min(d->idx_top_k, clen);
    S.sel.assign((size_t)T * k, -1);
    S.sel_k = k;
    std::vector<int> ord;
    for (int t = 0; t < T; t++) {
        ord.resize((size_t)clen);
        for (int cc = 0; cc < clen; cc++) {
            ord[(size_t)cc] = cc;
        }
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(), [&](int a, int b) {
            float sa = score[(size_t)t * clen + a], sb = score[(size_t)t * clen + b];
            if (sa != sb) {
                return sa > sb;
            }
            return a < b;
        });
        ord.resize((size_t)k);
        std::sort(ord.begin(), ord.end());
        int lens = (t + 1) / ratio;
        for (int j = 0; j < k; j++) {
            int cc = ord[(size_t)j];
            S.sel[(size_t)t * k + j] = (cc < lens) ? cc + T : -1;
        }
    }
}

void Fwd::attention(int li, const float * x, SState & S, float * o_out) {
    const int dim = hp->n_embd;
    const int H = hp->n_head, D = d->kv_dim, rd = hp->n_rot;
    const int off = d->rot_off;
    const int ratio = d->compress_ratio[li];
    const bool comp = ratio > 0;
    const layer_t & L = c->m->layers[(size_t)li];
    const float * fc = comp ? c->freq_comp : c->freq_win;

    // qr = rms(x @ wq_a^T, q_a_norm)
    std::vector<float> qr((size_t)T * d->q_lora_rank);
    {
        std::vector<float> tmp((size_t)T * d->q_lora_rank);
        linear_wt(*c, L.ds.wq_a, x, T, tmp.data());
        for (int t = 0; t < T; t++) {
            rms_w(tmp.data() + (size_t)t * d->q_lora_rank, L.ds.q_a_norm, d->q_lora_rank, 1e-20f,
                  qr.data() + (size_t)t * d->q_lora_rank);
        }
    }
    // q
    std::vector<float> q((size_t)T * H * D);
    linear_wt(*c, L.ds.wq_b, qr.data(), T, q.data());
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++) {
            rope_one(q.data() + ((size_t)t * H + h) * D, off, rd, t, fc, false);
        }
    }
    // kv (window), K == V
    std::vector<float> kv((size_t)T * D);
    {
        std::vector<float> tmp((size_t)T * D);
        linear_wt(*c, L.ds.wkv, x, T, tmp.data());
        for (int t = 0; t < T; t++) {
            float * row = tmp.data() + (size_t)t * D;
            rms_w(row, L.ds.kv_norm, D, 1e-20f, row);
            rope_one(row, off, rd, t, fc, false);
            if (!g_noq) {
                fp8_round_row(row, D);
            }
            std::memcpy(kv.data() + (size_t)t * D, row, sizeof(float) * D);
        }
    }
    // compressed KV (owner layers only)
    std::vector<float> latent, ckv, lat;
    int clen = 0;
    if (comp) {
        const bool is_src = (d->kv_src_of[li] == li);
        if (is_src) {
            compressor(li, x, ratio, latent, clen);
            if (li < hp->n_layer && li == d->topk_src_of[li]) {
                indexer(li, x, qr.data(), latent.data(), clen, ratio, S);
            }
            if (clen > 0) {
                lat.assign((size_t)clen * D, 0.f);
                for (int g = 0; g < clen; g++) {
                    float * row = lat.data() + (size_t)g * D;
                    std::memcpy(row, latent.data() + (size_t)g * D, sizeof(float) * D);
                    rope_one(row, off, rd, g * ratio, fc, false);
                    if (!g_noq) {
                        fp4_round_row(row, D, 16, true);
                    }
                }
                S.ckv = lat;
                S.ckv_rows = clen;
            }
        } else if (li == d->topk_src_of[li]) {
            // index source that is not a KV source (24/28/32/36): reuse ik
            clen = T / ratio;
            indexer(li, x, qr.data(), nullptr, clen, ratio, S);
        } else {
            clen = T / ratio;
        }
        ckv = S.ckv;
        if ((int)ckv.size() < clen * D) {
            clen = (int)ckv.size() / D;
        }
    }

    // build the union of window rows and selected compressed rows
    const int W = std::min(T, d->sliding_window);
    std::vector<int> widx((size_t)T * W, -1);
    for (int t = 0; t < T; t++) {
        int start = std::max(0, t - d->sliding_window + 1);
        for (int j = 0; j < W; j++) {
            int idx = start + j;
            widx[(size_t)t * W + j] = (idx > t) ? -1 : idx;
        }
    }
    const int K = S.sel_k;
    // kv_all = [kv ; ckv]
    const int nk = T + clen;
    std::vector<float> kv_all((size_t)nk * D);
    std::memcpy(kv_all.data(), kv.data(), sizeof(float) * T * D);
    if (clen > 0) {
        std::memcpy(kv_all.data() + (size_t)T * D, ckv.data(), sizeof(float) * clen * D);
    }
    std::vector<float> o((size_t)T * H * D, 0.f);
    sparse_attn(q.data(), T, H, D, kv_all.data(), nk, widx.data(), W, S.sel.data(),
                comp ? K : 0, L.ds.attn_sinks, 1.f / std::sqrt((float)D), o.data());

    // inverse rope, then fp8 over [T][G][4096]
    const int G = d->out_group_count;
    const int gsz = H / G * D; // 4096
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++) {
            rope_one(o.data() + ((size_t)t * H + h) * D, off, rd, t, fc, true);
        }
        if (!g_noq) {
            fp8_round_row(o.data() + (size_t)t * H * D, H * D);
        }
    }
    // 8 groups through wo_a, concatenated, then wo_b
    const wt & wa = L.ds.wo_a;
    const float * Wa = c->W(wa);
    std::vector<float> og((size_t)T * G * d->out_lora_rank, 0.f);
    for (int g = 0; g < G; g++) {
        // rows [g*1024, (g+1)*1024) of wo_a (K = gsz)
        const int N = d->out_lora_rank;
        const float * Wg = Wa + (size_t)g * N * (size_t)wa.K;
        par_for(*c->pool, N, [&](int n) {
            const float * wr = Wg + (size_t)n * (size_t)wa.K;
            for (int t = 0; t < T; t++) {
                og[(size_t)t * G * N + (size_t)g * N + n] =
                    dot_f32(wr, o.data() + (size_t)t * H * D + (size_t)g * (size_t)gsz, gsz);
            }
        });
    }
    // wo_b: [K=8192][N=5120]
    std::vector<float> y((size_t)T * dim, 0.f);
    linear_wt(*c, L.ds.wo_b, og.data(), T, y.data());
    std::memcpy(o_out, y.data(), sizeof(float) * (size_t)T * dim);
}

// ---------------------------------------------------------------- MoE
void Fwd::moe(int li, const float * x, float * y) {
    const hparams & hp = *c->hp;
    const ds41_hparams & d = *c->d;
    const int n_embd = hp.n_embd;
    const int E = hp.n_expert, K = hp.n_expert_used;
    const int ff = hp.n_ff_exp;
    const layer_t & L = c->m->layers[(size_t)li];

    // router logits
    wt gw;
    gw.data = L.moe.gate_inp;
    gw.type = L.moe.gate_inp_type;
    gw.K = n_embd;
    gw.N = E;
    std::vector<float> logits((size_t)T * E);
    linear_wt(*c, gw, x, T, logits.data());

    std::vector<float> scores((size_t)T * E);
    for (size_t i = 0; i < scores.size(); i++) {
        scores[i] = std::sqrt(softplus_(logits[i]));
    }
    // top-k per token (ties -> lowest index)
    std::vector<int> sel((size_t)T * K);
    std::vector<float> wts((size_t)T * K);
    for (int t = 0; t < T; t++) {
        const float * sc = scores.data() + (size_t)t * E;
        std::vector<int> ord((size_t)E);
        for (int e = 0; e < E; e++) {
            ord[(size_t)e] = e;
        }
        std::partial_sort(ord.begin(), ord.begin() + K, ord.end(), [&](int a, int b) {
            float va = sc[a] + (g_moe_vl && L.ds.exp_probs_b_vl ? L.ds.exp_probs_b_vl[a] : L.ds.exp_probs_b[a]);
            float vb = sc[b] + (g_moe_vl && L.ds.exp_probs_b_vl ? L.ds.exp_probs_b_vl[b] : L.ds.exp_probs_b[b]);
            if (va != vb) {
                return va > vb;
            }
            return a < b;
        });
        float sum = 1e-20f;
        for (int j = 0; j < K; j++) {
            sel[(size_t)t * K + j] = ord[(size_t)j];
            sum += sc[ord[(size_t)j]];
        }
        for (int j = 0; j < K; j++) {
            wts[(size_t)t * K + j] = sc[ord[(size_t)j]] / sum * d.expert_scale;
        }
    }

    for (int t = 0; t < T; t++) {
        float * yt = y + (size_t)t * n_embd;
        for (int i = 0; i < n_embd; i++) {
            yt[i] = 0.f;
        }
    }

    // group tokens by expert; experts run in ascending id (torch.unique order)
    std::vector<std::vector<std::pair<int, int>>> by_e((size_t)E);
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < K; j++) {
            int e = sel[(size_t)t * K + j];
            by_e[(size_t)e].emplace_back(t, j);
        }
    }

    const float lim = d.swiglu_clamp[li];
    const moe_wt & gw_e = L.moe.gate_exps;
    const moe_wt & uw_e = L.moe.up_exps;
    const moe_wt & dw_e = L.moe.down_exps;
    const size_t grb = ggml_row_bytes(gw_e.type, (uint64_t)gw_e.K);
    const size_t urb = ggml_row_bytes(uw_e.type, (uint64_t)uw_e.K);
    const size_t drb = ggml_row_bytes(dw_e.type, (uint64_t)dw_e.K);
    std::vector<float> xbuf((size_t)T * n_embd), h((size_t)T * ff);

    for (int e = 0; e < E; e++) {
        auto & lst = by_e[(size_t)e];
        if (lst.empty()) {
            continue;
        }
        const int p = (int)lst.size();
        // gathered activations for this expert
        for (int r = 0; r < p; r++) {
            std::memcpy(xbuf.data() + (size_t)r * n_embd, x + (size_t)lst[(size_t)r].first * n_embd,
                        sizeof(float) * n_embd);
        }
        const char * gbase = (const char *)gw_e.plane(e);
        const char * ubase = (const char *)uw_e.plane(e);
        const char * dbase = (const char *)dw_e.plane(e);

        par_for(*c->pool, gw_e.N, [&](int n) {
            static thread_local std::vector<float> rg, ru;
            rg.resize((size_t)gw_e.K);
            ru.resize((size_t)uw_e.K);
            dequant_row_local(gw_e.type, gbase + (size_t)n * grb, rg.data(), gw_e.K);
            dequant_row_local(uw_e.type, ubase + (size_t)n * urb, ru.data(), uw_e.K);
            for (int r = 0; r < p; r++) {
                const float * xr = xbuf.data() + (size_t)r * n_embd;
                float g = dot_f32(rg.data(), xr, gw_e.K);
                float u = dot_f32(ru.data(), xr, uw_e.K);
                g = std::min(g, lim);
                u = std::max(-lim, std::min(lim, u));
                h[(size_t)r * ff + (size_t)n] = silu_(g) * u;
            }
        });
        // weight the activation rows before the down projection (reference order)
        for (int r = 0; r < p; r++) {
            float w = wts[(size_t)lst[(size_t)r].first * K + lst[(size_t)r].second];
            float * hr = h.data() + (size_t)r * ff;
            for (int i = 0; i < ff; i++) {
                hr[i] *= w;
            }
        }
        par_for(*c->pool, dw_e.N, [&](int n) {
            static thread_local std::vector<float> rd;
            rd.resize((size_t)dw_e.K);
            dequant_row_local(dw_e.type, dbase + (size_t)n * drb, rd.data(), dw_e.K);
            for (int r = 0; r < p; r++) {
                int t = lst[(size_t)r].first;
                float v = dot_f32(rd.data(), h.data() + (size_t)r * ff, dw_e.K);
                y[(size_t)t * n_embd + (size_t)n] += v;
            }
        });
    }

    // shared expert (ungated add, same clamp family)
    {
        const float slim = d.swiglu_clamp_shexp[li];
        std::vector<float> sg((size_t)T * hp.n_ff_shexp), su((size_t)T * hp.n_ff_shexp);
        linear_wt(*c, L.ffn_gate, x, T, sg.data());
        linear_wt(*c, L.ffn_up, x, T, su.data());
        std::vector<float> sh((size_t)T * hp.n_ff_shexp);
        for (size_t i = 0; i < sh.size(); i++) {
            float g = std::min(sg[i], slim);
            float u = std::max(-slim, std::min(slim, su[i]));
            sh[i] = silu_(g) * u;
        }
        std::vector<float> sy((size_t)T * n_embd);
        linear_wt(*c, L.ffn_down, sh.data(), T, sy.data());
        for (size_t i = 0; i < (size_t)T * n_embd; i++) {
            y[i] += sy[i];
        }
    }
}

// ---------------------------------------------------------------- the whole forward
// returns the logits of the LAST token only (greedy decode needs nothing else)
static void forward(Ctx & c, const std::vector<int> & ids, std::vector<float> & logits) {
    const hparams & hp = *c.hp;
    const ds41_hparams & d = *c.d;
    const int T = (int)ids.size();
    const int dim = hp.n_embd, hc = d.hc_mult;

    // embedding: h[t][s][e] = dequant(tok_embd[ids[t]])
    std::vector<float> h((size_t)T * hc * dim);
    {
        const wt & te = c.m->tok_embd;
        const size_t rb = ggml_row_bytes(te.type, (uint64_t)te.K);
        const char * base = (const char *)te.data;
        for (int t = 0; t < T; t++) {
            float * row = h.data() + ((size_t)t * hc + 0) * dim;
            const void * src = base + (size_t)ids[(size_t)t] * rb;
            dequant_row_local(te.type, src, row, dim);
            if (g_bf16_emb) {
                for (int i = 0; i < dim; i++) {
                    row[i] = bf16_round(row[i]);
                }
            }
            for (int s = 1; s < hc; s++) {
                std::memcpy(h.data() + ((size_t)t * hc + s) * dim, row, sizeof(float) * dim);
            }
        }
    }
    std::vector<float> pre((size_t)T * hc, 0.f);
    for (int t = 0; t < T; t++) {
        pre[(size_t)t * hc + 0] = 1.f;
    }

    Fwd fw{&c, &hp, &d, T, ids};
    SState S;

    std::vector<float> res, mix_x, x, o, y;
    std::vector<float> a_pre, a_post, a_comb, f_pre, f_post, f_comb;

    for (int li = 0; li < hp.n_layer; li++) {
        const layer_t & L = c.m->layers[(size_t)li];
        if (d.is_ple[li]) {
            apply_engram(c, li, h, ids, T);
        }
        res = h;
        hc_mixes(c, h.data(), T, hc, dim, c.W(L.ds.hc_attn_fn), L.ds.hc_attn_scale,
                 L.ds.hc_attn_base, d.hc_eps, d.hc_sinkhorn, a_pre, a_post, a_comb);
        mix_x.assign((size_t)T * dim, 0.f);
        hc_pre(h.data(), pre.data(), T, hc, dim, mix_x.data());
        x.assign((size_t)T * dim, 0.f);
        for (int t = 0; t < T; t++) {
            rms_w(mix_x.data() + (size_t)t * dim, L.attn_norm, dim, 1e-20f,
                  x.data() + (size_t)t * dim);
        }
        o.assign((size_t)T * dim, 0.f);
        fw.attention(li, x.data(), S, o.data());
        hc_post(o.data(), res.data(), a_post.data(), a_comb.data(), T, hc, dim, h.data());

        res = h;
        hc_mixes(c, h.data(), T, hc, dim, c.W(L.ds.hc_ffn_fn), L.ds.hc_ffn_scale,
                 L.ds.hc_ffn_base, d.hc_eps, d.hc_sinkhorn, f_pre, f_post, f_comb);
        hc_pre(h.data(), a_pre.data(), T, hc, dim, mix_x.data());
        for (int t = 0; t < T; t++) {
            rms_w(mix_x.data() + (size_t)t * dim, L.post_attn_norm, dim, 1e-20f,
                  x.data() + (size_t)t * dim);
        }
        y.assign((size_t)T * dim, 0.f);
        fw.moe(li, x.data(), y.data());
        hc_post(y.data(), res.data(), f_post.data(), f_comb.data(), T, hc, dim, h.data());
        pre = f_pre;

        if (g_dbg) {
            double ss = 0;
            for (int i = 0; i < T * dim; i++) {
                ss += (double)h[(size_t)i] * h[(size_t)i];
            }
            fprintf(stderr, "[ds41] layer %2d rms(h[last]) %.6f\n", li, std::sqrt(ss / (T * dim)));
        }
    }

    // final norm and head for the last token
    mix_x.assign((size_t)dim, 0.f);
    hc_pre(h.data() + (size_t)(T - 1) * hc * dim, pre.data() + (size_t)(T - 1) * hc, 1, hc, dim,
            mix_x.data());
    std::vector<float> xn((size_t)dim);
    rms_w(mix_x.data(), c.m->output_norm, dim, 1e-20f, xn.data());

    const wt & head = c.m->output;
    const float * Wh = c.W(head);
    logits.assign((size_t)head.N, 0.f);
    linear(c, Wh, head.N, head.K, xn.data(), 1, dim, logits.data(), head.N);
}

// ---------------------------------------------------------------- incremental decode
// Persistent per-sequence state.  Every value kept here is exactly the f32 the
// whole-sequence forward recomputes from the same prefix, so a decode step is
// O(1) in the sequence length for the trunk (the routed experts are still read
// per token).  `--check` compares it against the `forward` oracle step by step.
//
// Two caveats found while validating it against that oracle, neither of which
// changes the greedy ids on the reference prompt:
//   * src/common/quant.h's dequantize_row() has no case for ggml type 18
//     (IQ3_XXS), and one routed-down tensor of this model is stored that way.
//     The oracle therefore leaves the caller's `rd` buffer untouched at that
//     layer and reads whatever the previous dequant left there; the value is a
//     function of the whole forward's expert order, not of a token prefix.
//     PF_DS41_IQ3XXS=1 enables the local correct dequant (both paths) for an
//     apples-to-apples check.
//   * Even with the correct dequant the model amplifies a residual ulp-level
//     accumulation-order difference once the prefix is a few tokens long (the
//     check is bit-exact at T=1/T=2 and the argmax/generated ids agree later);
//     this is the same prefill-vs-decode sensitivity docs/design/04-engine.md
//     records for the GPU paths.
//
//   * per layer: the 128-token window K ring (the window K row is the same
//     fp8-rounded, RoPE'd row the whole-sequence forward builds for that token);
//   * per KV source layer: the compressed cache rows (the RoPE'd/fp4-rounded
//     512-wide K row and its 128-wide DSA index key) plus the `ratio`-deep
//     ring of compressor projections that forms the partial block;
//   * the Engram mapped-token history, so the n-gram hash for one new token is
//     the same hash the whole-sequence forward computes.
struct DecState {
    struct KvCache {
        int ratio = 0;
        int clen = 0;            // completed compressed rows
        int ik_rows = 0;         // DSA index-key rows present
        std::vector<float> ckv;  // [clen][kv_dim]
        std::vector<float> ik;   // [ik_rows][idx_head_dim]
        std::vector<float> rkv;  // [ratio][kv_dim] compressor ring
        std::vector<float> rsc;  // [ratio][kv_dim] compressor gate ring
        int ring_n = 0;
    };

    int pos = 0;
    int n_layer = 0, dim = 0, kv_dim = 0, W = 0, idx_dim = 0;
    std::vector<float> win;   // [n_layer][W][kv_dim]
    std::vector<int> win_at;  // [n_layer][W] position stored, -1 empty
    std::vector<KvCache> kvc; // one per KV source
    std::vector<int> kv_slot; // [n_layer] -> kvc index, -1 for window-only

    // per-token transients (the SState equivalent, rebuilt for every token)
    std::vector<int> sel;
    int sel_k = 0;
    std::vector<uint8_t> cand;
    int cand_nb = 0, cand_clen = 0;
    bool has_cand = false;

    // Engram history of *mapped* token ids, most recent first
    int64_t hist[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int hist_n = 0;
    int pad = 0;
};

static void dec_init(Ctx & c, DecState & st) {
    const hparams & hp = *c.hp;
    const ds41_hparams & d = *c.d;
    st.n_layer = hp.n_layer;
    st.dim = hp.n_embd;
    st.kv_dim = d.kv_dim;
    st.W = d.sliding_window;
    st.idx_dim = d.idx_head_dim;
    st.pos = 0;
    st.win.assign((size_t)hp.n_layer * (size_t)st.W * (size_t)st.kv_dim, 0.f);
    st.win_at.assign((size_t)hp.n_layer * (size_t)st.W, -1);
    st.kv_slot.assign((size_t)hp.n_layer, -1);
    st.kvc.clear();
    st.kvc.resize((size_t)d.n_kv_src);
    for (int s = 0; s < d.n_kv_src; s++) {
        DecState::KvCache & K = st.kvc[(size_t)s];
        const int src = d.kv_src[s];
        K.ratio = d.compress_ratio[src];
        K.rkv.assign((size_t)std::max(K.ratio, 1) * (size_t)st.kv_dim, 0.f);
        K.rsc.assign((size_t)std::max(K.ratio, 1) * (size_t)st.kv_dim, 0.f);
    }
    for (int li = 0; li < hp.n_layer; li++) {
        const int src = d.kv_src_of[li];
        if (src < 0) {
            continue;
        }
        for (int s = 0; s < d.n_kv_src; s++) {
            if (d.kv_src[s] == src) {
                st.kv_slot[(size_t)li] = s;
                break;
            }
        }
    }
    st.pad = c.m->engram_token_map.empty() ? 0 : c.m->engram_token_map[(size_t)d.engram_pad];
    st.hist_n = 0;
    st.sel_k = 0;
    st.has_cand = false;
    st.cand_nb = 0;
}

static void dec_push_token(Ctx & c, DecState & st, int tok) {
    int64_t m = tok;
    if (!c.m->engram_token_map.empty() && tok >= 0 &&
        (size_t)tok < c.m->engram_token_map.size()) {
        m = c.m->engram_token_map[(size_t)tok];
    }
    for (int i = 7; i > 0; i--) {
        st.hist[i] = st.hist[i - 1];
    }
    st.hist[0] = m;
    if (st.hist_n < 8) {
        st.hist_n++;
    }
}

// Append one completed compressed block: the RoPE'd + fp4-rounded 512-wide K row
// and (when this source owns the indexer) its 128-wide index key.
static void kvsrc_append(Ctx & c, const layer_t & L, DecState::KvCache & K, const float * latent,
                         int block, bool with_ik) {
    const ds41_hparams & d = *c.d;
    const int D = d.kv_dim;
    const int hdim = d.idx_head_dim;
    const size_t row = (size_t)K.clen;
    K.ckv.resize((row + 1) * (size_t)D);
    float * dst = K.ckv.data() + row * (size_t)D;
    std::memcpy(dst, latent, sizeof(float) * (size_t)D);
    rope_one(dst, d.rot_off, c.hp->n_rot, block * K.ratio, c.freq_comp, false);
    if (!g_noq) {
        fp4_round_row(dst, D, 16, true);
    }
    if (with_ik) {
        K.ik.resize((row + 1) * (size_t)hdim);
        float * ir = K.ik.data() + row * (size_t)hdim;
        linear_wt(c, L.ds.idx_attn_k, latent, 1, ir);
        rms_w(ir, L.ds.idx_k_norm, hdim, 1e-20f, ir);
        rope_one(ir, hdim - 64, 64, block * K.ratio, c.freq_comp, false);
        if (!g_noq) {
            fp4_round_row(ir, hdim, 32, false);
        }
        K.ik_rows = (int)row + 1;
    }
    K.clen = (int)row + 1;
}

// Advance one KV source by one token.  `ratio == 1` completes a block on every
// token; otherwise the per-token compressor projections accumulate in the ring
// and the block is formed when `ratio` tokens are present, which is exactly the
// block boundary the whole-sequence compressor uses.
static void kvsrc_push_token(Ctx & c, int li, const float * x, DecState::KvCache & K,
                             bool with_ik) {
    const ds41_hparams & d = *c.d;
    const layer_t & L = c.m->layers[(size_t)li];
    const int D = d.kv_dim;
    const int ratio = K.ratio;
    if (ratio <= 0) {
        return;
    }
    if (ratio == 1) {
        std::vector<float> latent((size_t)D);
        linear_wt(c, L.ds.comp_wkv, x, 1, latent.data());
        rms_w(latent.data(), L.ds.comp_norm, D, 1e-20f, latent.data());
        kvsrc_append(c, L, K, latent.data(), K.clen, with_ik);
        return;
    }
    linear_wt(c, L.ds.comp_wkv, x, 1, K.rkv.data() + (size_t)K.ring_n * (size_t)D);
    linear_wt(c, L.ds.comp_wgate, x, 1, K.rsc.data() + (size_t)K.ring_n * (size_t)D);
    K.ring_n++;
    if (K.ring_n < ratio) {
        return;
    }
    std::vector<float> latent((size_t)D);
    for (int e = 0; e < D; e++) {
        float m = -INFINITY;
        for (int j = 0; j < ratio; j++) {
            m = std::max(m, K.rsc[(size_t)j * (size_t)D + (size_t)e]);
        }
        float sum = 0;
        for (int j = 0; j < ratio; j++) {
            sum += std::exp(K.rsc[(size_t)j * (size_t)D + (size_t)e] - m);
        }
        float acc = 0;
        for (int j = 0; j < ratio; j++) {
            float p = std::exp(K.rsc[(size_t)j * (size_t)D + (size_t)e] - m) / sum;
            acc += p * K.rkv[(size_t)j * (size_t)D + (size_t)e];
        }
        latent[(size_t)e] = acc;
    }
    rms_w(latent.data(), L.ds.comp_norm, D, 1e-20f, latent.data());
    kvsrc_append(c, L, K, latent.data(), K.clen, with_ik);
    K.ring_n = 0;
}

// The candidate-block stage for one query (compute_candidates with T == 1 and
// t == the last position, so lens == clen).
static void compute_candidates_one(const float * score, int clen, int block, int topk_blocks,
                                   std::vector<uint8_t> & keep, int & nb_out) {
    int nb = (clen + block - 1) / block;
    nb_out = nb;
    keep.assign((size_t)std::max(nb, 0), 0);
    if (nb <= 0) {
        return;
    }
    std::vector<float> bmax((size_t)nb);
    for (int b = 0; b < nb; b++) {
        float m = -INFINITY;
        int hi = std::min(clen, (b + 1) * block);
        for (int cc = b * block; cc < hi; cc++) {
            m = std::max(m, score[(size_t)cc]);
        }
        bmax[(size_t)b] = m;
    }
    const int lens = clen;
    int last = lens > 0 ? (lens - 1) / block : -1;
    if (last >= 0 && last < nb) {
        bmax[(size_t)last] = INFINITY;
    }
    std::vector<int> ord((size_t)nb);
    for (int b = 0; b < nb; b++) {
        ord[(size_t)b] = b;
    }
    int take = std::min(topk_blocks, nb);
    std::partial_sort(ord.begin(), ord.begin() + take, ord.end(), [&](int a, int b) {
        if (bmax[(size_t)a] != bmax[(size_t)b]) {
            return bmax[(size_t)a] > bmax[(size_t)b];
        }
        return a < b;
    });
    for (int i = 0; i < take; i++) {
        int b = ord[(size_t)i];
        if (bmax[(size_t)b] > -INFINITY) {
            keep[(size_t)b] = 1;
        }
    }
}

// The DSA indexer / selector for one query.  The score and the top-k are
// recomputed per token from the persistent compressed rows, exactly as the
// whole-sequence indexer does for its last row.
static void index_select_one(Ctx & c, int li, const float * x, const float * qr, DecState & st) {
    const ds41_hparams & d = *c.d;
    const layer_t & L = c.m->layers[(size_t)li];
    const int nh = d.idx_n_head, hdim = d.idx_head_dim;
    const int ratio = d.compress_ratio[li];
    const int p = st.pos;
    st.sel.clear();
    st.sel_k = 0;
    const int src = d.kv_src_of[li];
    if (src < 0 || st.kv_slot[(size_t)src] < 0) {
        return;
    }
    DecState::KvCache & Kc = st.kvc[(size_t)st.kv_slot[(size_t)src]];
    int clen = (p + 1) / ratio;
    if (clen > Kc.clen) {
        clen = Kc.clen;
    }
    if (clen <= 0) {
        return;
    }
    // index query from qr (one token)
    std::vector<float> qi((size_t)nh * (size_t)hdim);
    linear_wt(c, L.ds.idx_attn_q_b, qr, 1, qi.data());
    for (int h = 0; h < nh; h++) {
        float * row = qi.data() + (size_t)h * (size_t)hdim;
        rope_one(row, hdim - 64, 64, p, c.freq_comp, false);
        if (!g_noq) {
            fp4_round_row(row, hdim, 32, false);
        }
    }
    // per-head weight
    std::vector<float> w((size_t)nh);
    linear_wt(c, L.ds.idx_proj, x, 1, w.data());
    const float wscale = 1.f / std::sqrt((float)hdim * (float)nh);
    for (auto & v : w) {
        v *= wscale;
    }
    const int ik_rows = Kc.ik_rows;
    std::vector<float> score((size_t)clen);
    for (int cc = 0; cc < clen; cc++) {
        if (cc < ik_rows) {
            const float * ik = Kc.ik.data() + (size_t)cc * (size_t)hdim;
            float acc = 0;
            for (int h = 0; h < nh; h++) {
                float s = dot_f32(qi.data() + (size_t)h * (size_t)hdim, ik, hdim);
                if (s > 0) {
                    acc += s * w[(size_t)h];
                }
            }
            score[(size_t)cc] = acc;
        } else {
            score[(size_t)cc] = -INFINITY;
        }
    }
    if (li == d.cand_src) {
        compute_candidates_one(score.data(), clen, d.cand_block, d.cand_topk_blocks, st.cand,
                               st.cand_nb);
        st.cand_clen = clen;
        st.has_cand = true;
    } else if (d.cand_src >= 0 && li > d.cand_src && st.has_cand) {
        for (int cc = 0; cc < clen; cc++) {
            int b = cc / d.cand_block;
            if (b >= st.cand_nb || !st.cand[(size_t)b]) {
                score[(size_t)cc] = -INFINITY;
            }
        }
    }
    const int k = std::min(d.idx_top_k, clen);
    st.sel.assign((size_t)k, -1);
    st.sel_k = k;
    std::vector<int> ord((size_t)clen);
    for (int cc = 0; cc < clen; cc++) {
        ord[(size_t)cc] = cc;
    }
    std::partial_sort(ord.begin(), ord.begin() + k, ord.end(), [&](int a, int b) {
        float sa = score[(size_t)a], sb = score[(size_t)b];
        if (sa != sb) {
            return sa > sb;
        }
        return a < b;
    });
    ord.resize((size_t)k);
    std::sort(ord.begin(), ord.end());
    const int lens = (p + 1) / ratio;
    for (int j = 0; j < k; j++) {
        int cc = ord[(size_t)j];
        st.sel[(size_t)j] = (cc < lens) ? cc : -1;
    }
}

static void attention_one(Ctx & c, int li, const float * x, DecState & st, float * o_out) {
    const hparams & hp = *c.hp;
    const ds41_hparams & d = *c.d;
    const int dim = hp.n_embd;
    const int H = hp.n_head, D = d.kv_dim, rd = hp.n_rot;
    const int off = d.rot_off;
    const int ratio = d.compress_ratio[li];
    const bool comp = ratio > 0;
    const layer_t & L = c.m->layers[(size_t)li];
    const float * fc = comp ? c.freq_comp : c.freq_win;
    const int p = st.pos;
    const int Wcap = d.sliding_window;

    // qr = rms(x @ wq_a^T, q_a_norm)
    std::vector<float> qr((size_t)d.q_lora_rank);
    {
        std::vector<float> tmp((size_t)d.q_lora_rank);
        linear_wt(c, L.ds.wq_a, x, 1, tmp.data());
        rms_w(tmp.data(), L.ds.q_a_norm, d.q_lora_rank, 1e-20f, qr.data());
    }
    // q
    std::vector<float> q((size_t)H * (size_t)D);
    linear_wt(c, L.ds.wq_b, qr.data(), 1, q.data());
    for (int h = 0; h < H; h++) {
        rope_one(q.data() + (size_t)h * (size_t)D, off, rd, p, fc, false);
    }

    // window KV row for this position (K == V)
    {
        float * wrow =
            st.win.data() + ((size_t)li * (size_t)Wcap + (size_t)(p % Wcap)) * (size_t)D;
        std::vector<float> tmp((size_t)D);
        linear_wt(c, L.ds.wkv, x, 1, tmp.data());
        rms_w(tmp.data(), L.ds.kv_norm, D, 1e-20f, tmp.data());
        rope_one(tmp.data(), off, rd, p, fc, false);
        if (!g_noq) {
            fp8_round_row(tmp.data(), D);
        }
        std::memcpy(wrow, tmp.data(), sizeof(float) * (size_t)D);
        st.win_at[(size_t)li * (size_t)Wcap + (size_t)(p % Wcap)] = p;
    }

    if (comp) {
        const int src = d.kv_src_of[li];
        const bool is_src = (src == li);
        const bool is_idx = (li == d.topk_src_of[li]);
        if (is_src) {
            DecState::KvCache & Kc = st.kvc[(size_t)st.kv_slot[(size_t)li]];
            kvsrc_push_token(c, li, x, Kc, is_idx);
        }
        if (is_idx) {
            index_select_one(c, li, x, qr.data(), st);
        }
    }

    // attention over the visible window rows (ascending positions) then the
    // selected compressed rows (ascending block index).  This is the same order
    // sparse_attn walks for the last token of the whole-sequence forward.
    const int Wvis = std::min(p + 1, d.sliding_window);
    const int start = std::max(0, p - d.sliding_window + 1);
    int K = 0;
    DecState::KvCache * Kc = nullptr;
    if (comp) {
        K = st.sel_k;
        const int src = d.kv_src_of[li];
        if (src >= 0 && st.kv_slot[(size_t)src] >= 0) {
            Kc = &st.kvc[(size_t)st.kv_slot[(size_t)src]];
            K = std::min(K, Kc->clen);
            K = std::min(K, (int)st.sel.size());
        } else {
            K = 0;
        }
    }
    const int nk = Wvis + K;
    std::vector<float> kv_all((size_t)nk * (size_t)D);
    for (int j = 0; j < Wvis; j++) {
        const int pos = start + j;
        const int slot = pos % d.sliding_window;
        const float * r =
            st.win.data() + ((size_t)li * (size_t)Wcap + (size_t)slot) * (size_t)D;
        std::memcpy(kv_all.data() + (size_t)j * (size_t)D, r, sizeof(float) * (size_t)D);
    }
    for (int j = 0; j < K; j++) {
        const int cc = st.sel[(size_t)j];
        if (cc < 0) {
            continue;
        }
        std::memcpy(kv_all.data() + (size_t)(Wvis + j) * (size_t)D,
                    Kc->ckv.data() + (size_t)cc * (size_t)D, sizeof(float) * (size_t)D);
    }
    std::vector<int> widx((size_t)Wvis);
    for (int j = 0; j < Wvis; j++) {
        widx[(size_t)j] = j;
    }
    std::vector<int> selidx((size_t)std::max(K, 1), -1);
    for (int j = 0; j < K; j++) {
        selidx[(size_t)j] = (st.sel[(size_t)j] >= 0) ? Wvis + j : -1;
    }
    std::vector<float> o((size_t)H * (size_t)D, 0.f);
    sparse_attn(q.data(), 1, H, D, kv_all.data(), nk, widx.data(), Wvis, selidx.data(), K,
                L.ds.attn_sinks, 1.f / std::sqrt((float)D), o.data());

    // inverse rope, then fp8 over [H*D]
    for (int h = 0; h < H; h++) {
        rope_one(o.data() + (size_t)h * (size_t)D, off, rd, p, fc, true);
    }
    if (!g_noq) {
        fp8_round_row(o.data(), H * D);
    }
    // 8 groups through wo_a, concatenated, then wo_b
    const int G = d.out_group_count;
    const int gsz = H / G * D;
    const wt & wa = L.ds.wo_a;
    const float * Wa = c.W(wa);
    std::vector<float> og((size_t)G * (size_t)d.out_lora_rank, 0.f);
    for (int g = 0; g < G; g++) {
        const int N = d.out_lora_rank;
        const float * Wg = Wa + (size_t)g * (size_t)N * (size_t)wa.K;
        par_for(*c.pool, N, [&](int n) {
            const float * wr = Wg + (size_t)n * (size_t)wa.K;
            og[(size_t)g * (size_t)N + (size_t)n] =
                dot_f32(wr, o.data() + (size_t)g * (size_t)gsz, gsz);
        });
    }
    std::vector<float> y((size_t)dim, 0.f);
    linear_wt(c, L.ds.wo_b, og.data(), 1, y.data());
    std::memcpy(o_out, y.data(), sizeof(float) * (size_t)dim);
}

// One token through the whole trunk.  `want_logits` is false for all but the
// last prompt token, which skips the (129280-wide) head on the prompt pass.
static void forward_one(Ctx & c, int tok, DecState & st, std::vector<float> * logits,
                        bool want_logits) {
    const hparams & hp = *c.hp;
    const ds41_hparams & d = *c.d;
    const int dim = hp.n_embd, hc = d.hc_mult;
    const int T = 1;
    const bool is_img = g_mm.img_row && st.pos < g_mm.n_tok && g_mm.img_row[st.pos] >= 0;

    dec_push_token(c, st, tok);

    // embedding: h[s][e] = dequant(tok_embd[tok])
    std::vector<float> h((size_t)hc * (size_t)dim);
    {
        const wt & te = c.m->tok_embd;
        const size_t rb = ggml_row_bytes(te.type, (uint64_t)te.K);
        const char * base = (const char *)te.data;
        float * row = h.data();
        if (is_img) {
            const int rr = g_mm.img_row[st.pos];
            std::memcpy(row, g_mm.embd + (size_t)rr * (size_t)dim, sizeof(float) * (size_t)dim);
        } else {
            dequant_row_local(te.type, base + (size_t)tok * rb, row, dim);
        }
        if (g_bf16_emb) {
            for (int i = 0; i < dim; i++) {
                row[i] = bf16_round(row[i]);
            }
        }
        for (int s = 1; s < hc; s++) {
            std::memcpy(h.data() + (size_t)s * (size_t)dim, row, sizeof(float) * (size_t)dim);
        }
    }
    std::vector<float> pre((size_t)hc, 0.f);
    pre[0] = 1.f;

    Fwd fw{&c, &hp, &d, 1, {}};

    std::vector<float> res, mix_x, x, o, y;
    std::vector<float> a_pre, a_post, a_comb, f_pre, f_post, f_comb;

    for (int li = 0; li < hp.n_layer; li++) {
        const layer_t & L = c.m->layers[(size_t)li];
        if (d.is_ple[li] && !is_img) {
            std::vector<int> rows;
            engram_hash_one(c, st.hist, st.hist_n, d.engram_index_of[li], rows);
            apply_engram_rows(c, li, h, T, rows);
        }
        res = h;
        hc_mixes(c, h.data(), T, hc, dim, c.W(L.ds.hc_attn_fn), L.ds.hc_attn_scale,
                 L.ds.hc_attn_base, d.hc_eps, d.hc_sinkhorn, a_pre, a_post, a_comb);
        mix_x.assign((size_t)dim, 0.f);
        hc_pre(h.data(), pre.data(), T, hc, dim, mix_x.data());
        x.assign((size_t)dim, 0.f);
        rms_w(mix_x.data(), L.attn_norm, dim, 1e-20f, x.data());
        o.assign((size_t)dim, 0.f);
        attention_one(c, li, x.data(), st, o.data());
        hc_post(o.data(), res.data(), a_post.data(), a_comb.data(), T, hc, dim, h.data());

        res = h;
        hc_mixes(c, h.data(), T, hc, dim, c.W(L.ds.hc_ffn_fn), L.ds.hc_ffn_scale,
                 L.ds.hc_ffn_base, d.hc_eps, d.hc_sinkhorn, f_pre, f_post, f_comb);
        mix_x.assign((size_t)dim, 0.f);
        hc_pre(h.data(), a_pre.data(), T, hc, dim, mix_x.data());
        x.assign((size_t)dim, 0.f);
        rms_w(mix_x.data(), L.post_attn_norm, dim, 1e-20f, x.data());
        y.assign((size_t)dim, 0.f);
        g_moe_vl = is_img;
        fw.moe(li, x.data(), y.data());
        g_moe_vl = false;
        hc_post(y.data(), res.data(), f_post.data(), f_comb.data(), T, hc, dim, h.data());
        pre = f_pre;

        if (g_dbg) {
            double ss = 0;
            for (int i = 0; i < dim; i++) {
                ss += (double)h[(size_t)i] * (double)h[(size_t)i];
            }
            fprintf(stderr, "[ds41] layer %2d rms(h[last]) %.6f\n", li, std::sqrt(ss / (double)dim));
        }
    }
    st.pos++;

    if (!want_logits) {
        return;
    }
    // final norm and head
    std::vector<float> mix((size_t)dim, 0.f), xn((size_t)dim);
    hc_pre(h.data(), pre.data(), 1, hc, dim, mix.data());
    rms_w(mix.data(), c.m->output_norm, dim, 1e-20f, xn.data());
    const wt & head = c.m->output;
    const float * Wh = c.W(head);
    logits->assign((size_t)head.N, 0.f);
    linear(c, Wh, head.N, head.K, xn.data(), 1, dim, logits->data(), head.N);
}

// A ds41-specific, in-process prefix cache.  The engine's tiered prefix cache is
// keyed on the paged KV pool, which this forward does not use; the reusable
// state here is the window ring + compressed caches + DSA candidate mask +
// Engram history.  Snapshots are keyed by a chained per-token hash
// (ds41_prefix_cache.h), so a request that shares a prefix with any snapshot
// resumes from the longest match and forwards only the suffix.  The usual
// server pattern (one system prompt followed by different user turns) therefore
// hits even when the prompt as a whole is new.
//
// A snapshot is taken at the prompt end and every `PF_DS41_PC_EVERY` tokens
// (default 16) during it.  Total bytes are bounded (default 512 MiB,
// `PF_DS41_PC_MB`), evicting least-recently-used snapshots; a state larger than
// a quarter of the cap is skipped, so a long-context slice cannot pin the
// process.  `PF_DS41_PC=0` disables lookups and stores (the cold-run A/B).
struct PcPayload {
    DecState st;
    std::vector<float> logits;
};

static size_t pc_cap_bytes() {
    const char * mb = si::env::str("PF_DS41_PC_MB");
    const long m = mb ? atol(mb) : 512;
    return (size_t)(m > 0 ? m : 0) << 20;
}
static ds41_prefix_cache<PcPayload> & pc_cache() {
    static ds41_prefix_cache<PcPayload> c(pc_cap_bytes());
    return c;
}
static bool pc_enabled() {
    const char * v = si::env::str("PF_DS41_PC");
    return v == nullptr || (v[0] != '\0' && !(v[0] == '0' && v[1] == '\0'));
}
static int pc_every() {
    const int n = si::env::i32("PF_DS41_PC_EVERY", 16);
    return n > 0 ? n : 16;
}
static size_t pc_state_bytes(const DecState & st) {
    size_t b = st.win.size() * sizeof(float) + st.win_at.size() * sizeof(int) + st.sel.size() * sizeof(int)
               + st.cand.size() * sizeof(uint8_t);
    for (const auto & K : st.kvc) {
        b += K.ckv.size() * sizeof(float) + K.ik.size() * sizeof(float) + K.rkv.size() * sizeof(float)
             + K.rsc.size() * sizeof(float);
    }
    return b + sizeof(DecState);
}
static bool pc_state_ok(const DecState & st, const hparams & hp, const ds41_hparams & d) {
    return st.n_layer == hp.n_layer && st.dim == hp.n_embd && st.kv_dim == d.kv_dim && st.W == d.sliding_window
           && st.pos >= 0 && st.kvc.size() == (size_t)d.n_kv_src
           && st.win.size() == (size_t)hp.n_layer * (size_t)d.sliding_window * (size_t)d.kv_dim
           && st.win_at.size() == (size_t)hp.n_layer * (size_t)d.sliding_window;
}
static void pc_store(uint64_t h, int consumed, const DecState & st, const std::vector<float> & logits) {
    const bool dbg = si::env::flag("PF_DS41_PC_DEBUG");
    const size_t b = pc_state_bytes(st) + logits.size() * sizeof(float);
    if (b > pc_cache().max_entry_bytes()) {
        if (dbg) {
            fprintf(stderr, "[ds41pc] skip hash=%016llx consumed=%d bytes=%zu (cap/4=%zu)\n", (unsigned long long)h,
                    consumed, b, pc_cache().max_entry_bytes());
        }
        return;
    }
    // A mid-prompt snapshot can share its hash with an earlier prompt-end
    // snapshot of the same length.  The states are identical, but only the end
    // snapshot carries logits, and an exact-prompt hit must not lose them.
    if (logits.empty()) {
        const auto * old = pc_cache().peek(h);
        if (old && old->consumed == consumed && !old->payload.logits.empty()) {
            pc_cache().touch(h);
            return;
        }
    }
    PcPayload p;
    p.st = st;
    p.logits = logits;
    const bool ok = pc_cache().insert(h, consumed, std::move(p), b);
    if (dbg) {
        fprintf(stderr, "[ds41pc] %s hash=%016llx consumed=%d bytes=%zu entries=%zu total=%zu\n",
                ok ? "store" : "reject", (unsigned long long)h, consumed, b, pc_cache().size(), pc_cache().bytes());
    }
}

// Forward ids[begin..] on a state whose chain hash for [0..begin) is `chain`
// (ds41_pc_seed() when begin == 0).  Takes the prompt-end snapshot with the
// logits, plus one without logits at every `pc_every` boundary.
static void forward_prompt_range(Ctx & c, const std::vector<int> & ids, size_t begin, uint64_t chain, DecState & st,
                                 std::vector<float> & logits, bool pc_on) {
    logits.clear();
    const size_t every = (size_t)pc_every();
    const std::vector<float> no_logits;
    for (size_t i = begin; i < ids.size(); i++) {
        chain = ds41_pc_step(chain, ids[i]);
        const bool last = (i + 1 == ids.size());
        forward_one(c, ids[i], st, &logits, last);
        if (pc_on && !last && (i + 1) % every == 0) {
            pc_store(chain, (int)(i + 1), st, no_logits);
        }
    }
    if (pc_on && begin < ids.size()) {
        pc_store(chain, (int)ids.size(), st, logits);
    }
}

static void decode_one(Ctx & c, int tok, DecState & st, std::vector<float> & logits) {
    forward_one(c, tok, st, &logits, true);
}

// ---------------------------------------------------------------- main

} // anonymous namespace
// ---------------------------------------------------------------------------
// Engine entry point.  The reference math above is shared with dev/ds41_gen.cpp
// (the validated CPU oracle).  This wrapper builds the per-request context from
// the engine's already-loaded model, runs the prompt once into a persistent
// DecState, and then decodes one token at a time (O(1) trunk state per step).
// ---------------------------------------------------------------------------
std::vector<int> ds41_generate(model & m, tokenizer & tk, const std::vector<int> & prompt, const mm_prompt * mm,
                               const gen_params & gp, const std::function<bool(int)> & cb,
                               std::vector<float> * first_logits, size_t * cached_tokens) {
    if (cached_tokens) {
        *cached_tokens = 0;
    }
    g_noq = si::env::flag("PF_DS41_NOQ");
    g_dbg = si::env::flag("PF_DS41_DEBUG");
    g_bf16_emb = si::env::flag("PF_DS41_BF16EMB");
    g_eng_bf16 = si::env::flag("PF_DS41_ENG_BF16");
    if (const char * nt = si::env::str("PF_DS41_THREADS")) {
        g_nt = atoi(nt);
    } else if (const char * nt = si::env::str("PF_CPU_THREADS")) {
        g_nt = atoi(nt);
    }
    const hparams & hp = m.hp;
    const ds41_hparams & d = hp.ds;
    Pool pool(g_nt);
    Ctx c;
    c.m = &m;
    c.hp = &hp;
    c.d = &d;
    c.pool = &pool;
    c.rd = hp.n_rot;
    build_freqs(c.freq_win, hp.n_rot, hp.rope_base, 0, 1.f, 0.f, 0.f);
    const float factor = d.yarn_factor > 0.f ? d.yarn_factor : 16.f;
    const int orig = d.yarn_orig > 0 ? d.yarn_orig : 65536;
    const float bf = d.yarn_beta_fast > 0.f ? d.yarn_beta_fast : 32.f;
    const float bs = d.yarn_beta_slow > 0.f ? d.yarn_beta_slow : 1.f;
    build_freqs(c.freq_comp, hp.n_rot, d.compress_rope_base > 0.f ? d.compress_rope_base : 160000.f, orig, factor,
                bf, bs);

    g_mm = MmView{};
    if (mm && !mm->embd.empty() && !mm->img_row.empty()) {
        g_mm.embd = mm->embd.data();
        g_mm.img_row = mm->img_row.data();
        g_mm.n_embd = hp.n_embd;
        g_mm.n_tok = (int)mm->img_row.size();
    }
    if (si::env::flag("PF_DS41_MMDEBUG")) {
        int n_img_rows = 0;
        for (int r : mm ? mm->img_row : std::vector<int>{}) {
            n_img_rows += r >= 0 ? 1 : 0;
        }
        fprintf(stderr, "[ds41mm] prompt=%zu img_row=%zu image_rows=%d embd=%zu d_embd=%p n_img=%d\n",
                prompt.size(), mm ? mm->img_row.size() : 0, n_img_rows, mm ? mm->embd.size() : 0,
                mm ? (const void *)mm->d_embd : nullptr, mm ? mm->n_img : 0);
    }
    sampler_state ss;
    ss.seed(gp.seed ? gp.seed : std::random_device{}());
    DecState st;
    std::vector<float> logits;
    std::vector<int> ids = prompt;
    std::vector<int> gen;
    // The cache is only valid for text prompts: an image row changes the state
    // for the same token ids, so a media request neither looks up nor stores.
    const bool pc_on = pc_enabled() && mm == nullptr;
    bool reused = false;
    size_t resumed = 0;
    uint64_t chain = ds41_pc_seed();
    if (pc_on) {
        const auto hit = pc_cache().longest(ids);
        if (hit.found) {
            auto * e = pc_cache().find(hit.hash);
            if (e && e->consumed == hit.consumed && pc_state_ok(e->payload.st, hp, d)
                && e->payload.st.pos == hit.consumed
                && (hit.consumed < (int)ids.size() || !e->payload.logits.empty())) {
                st = e->payload.st;
                logits = e->payload.logits;
                chain = hit.hash;
                resumed = (size_t)hit.consumed;
                reused = true;
            }
        }
        if (si::env::flag("PF_DS41_PC_DEBUG")) {
            fprintf(stderr, "[ds41pc] %s prompt=%zu resumed=%zu%s\n", reused ? "hit" : "miss", ids.size(), resumed,
                    reused && resumed == ids.size() ? " (exact)" : "");
        }
    }
    if (!reused) {
        dec_init(c, st);
        resumed = 0;
        chain = ds41_pc_seed();
        logits.clear();
    }
    if (resumed < ids.size()) {
        forward_prompt_range(c, ids, resumed, chain, st, logits, pc_on);
    }
    if (first_logits) {
        *first_logits = logits;
    }
    if (cached_tokens) {
        *cached_tokens = resumed;
    }
    if (mm && si::env::flag("PF_DS41_MMDEBUG")) {
        std::vector<int> ord(logits.size());
        for (size_t i = 0; i < ord.size(); i++) {
            ord[i] = (int)i;
        }
        const int n = std::min<int>(8, (int)ord.size());
        std::partial_sort(ord.begin(), ord.begin() + n, ord.end(),
                          [&](int a, int b) { return logits[(size_t)a] > logits[(size_t)b]; });
        fprintf(stderr, "[ds41mm] first top8:");
        for (int i = 0; i < n; i++) fprintf(stderr, " %d:%.3f", ord[(size_t)i], logits[(size_t)ord[(size_t)i]]);
        fprintf(stderr, " | tokens[0..3]=%d,%d,%d,%d img_row=%d,%d,%d\n",
                ids[0], ids.size() > 1 ? ids[1] : -1, ids.size() > 2 ? ids[2] : -1, ids.size() > 3 ? ids[3] : -1,
                mm->img_row.size() > 1 ? mm->img_row[1] : -1, mm->img_row.size() > 2 ? mm->img_row[2] : -1,
                mm->img_row.size() > 3 ? mm->img_row[3] : -1);
    }
    for (int step = 0; step < gp.max_tokens; step++) {
        const int tok = sample_token(logits.data(), (int)logits.size(), gp, ids, ss);
        if (!gp.ignore_eos && (tok == tk.eos_id || tok == 1)) {
            break;
        }
        gen.push_back(tok);
        if (cb && !cb(tok)) {
            break;
        }
        ids.push_back(tok);
        decode_one(c, tok, st, logits);
    }
    g_mm = MmView{};
    return gen;
}

} // namespace si
