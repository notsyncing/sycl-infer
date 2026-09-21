// ---------------------------------------------------------------------------
// Shared CPU kernel infrastructure: the persistent worker pool, the runtime
// ISA dispatch table (AVX2 / AVX-VNNI / AVX-512, selected from cpu_isa.h), the
// fused dequantized GEMV row, and the RMSNorm helpers used by more than one
// kernel file.
//
// Compiled with -fno-sycl (see CMakeLists), so <immintrin.h> and the target
// attributes are host-only.
// ---------------------------------------------------------------------------
#include "common.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include <immintrin.h>

#include "cpu_isa.h"

namespace si {

namespace {

// CLI override (--cpu-threads); 0 = auto.  Read when the pool is first used.
int g_cpu_threads_override = 0;

// Number of physical cores, topology-agnostic (works on hybrid P/E parts and
// on uniform CPUs alike): count the distinct (physical id, core id) pairs in
// /proc/cpuinfo.  Returns 0 when the platform does not expose them (e.g. some
// ARM kernels), in which case the caller falls back to hardware_concurrency.
int detect_physical_cores() {
    FILE * f = fopen("/proc/cpuinfo", "r");
    if (!f) {
        return 0;
    }
    std::set<std::pair<int, int>> cores;
    char line[256];
    int phys = -1, core = -1;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\0') {
            if (phys >= 0 && core >= 0) {
                cores.insert({phys, core});
            }
            phys = core = -1;
            continue;
        }
        const char * colon;
        if (strncmp(line, "physical id", 11) == 0 && (colon = strchr(line, ':')) != nullptr) {
            phys = atoi(colon + 1);
        } else if (strncmp(line, "core id", 7) == 0 && (colon = strchr(line, ':')) != nullptr) {
            core = atoi(colon + 1);
        }
    }
    if (phys >= 0 && core >= 0) {
        cores.insert({phys, core});
    }
    fclose(f);
    return (int)cores.size();
}

int detect_threads() {
    if (g_cpu_threads_override > 0) {
        return g_cpu_threads_override;
    }
    if (const char * e = getenv("PF_CPU_THREADS")) {
        const int n = atoi(e);
        if (n >= 1) {
            return n;
        }
    }
    const unsigned hw = std::thread::hardware_concurrency();
    // physical cores are the better default for the AVX2/FMA-bound CPU kernels
    // (HT siblings and slow E-cores add little and can even hurt)
    const int phys = detect_physical_cores();
    if (phys >= 1 && (int)hw >= 1 && phys <= (int)hw) {
        return phys;
    }
    return hw ? (int)hw : 1;
}

// One pool per process.  The forward pass issues many small parallel regions
// (one per GEMV call group), so a condition-variable wake dominates once the
// pool has more than a couple of workers.  Workers spin briefly on the
// generation counter and only sleep on the CV when idle; the publisher only
// notifies when a worker is known to be asleep.  Every worker is still counted
// (`remaining_`), so main acks each run and a generation is never skipped; the
// one-time `ready_` handshake makes the first generation race-free.
class thread_pool {
public:
    static thread_pool & get() {
        static thread_pool p;
        return p;
    }

    void run(int n, const std::function<void(int)> & fn) {
        if (n <= 0) {
            return;
        }
        if (n == 1) {
            // frequent in decode (rmsnorm/embed rows==1): do not wake the pool
            fn(0);
            return;
        }
        if (nthreads_ <= 1) {
            for (int i = 0; i < n; i++) {
                fn(i);
            }
            return;
        }
        {
            std::unique_lock<std::mutex> lk(m_);
            // first call: every worker must have recorded its baseline
            // generation, else a late-starting worker could read the new gen as
            // its baseline and skip the run (main would wait forever)
            if (!ready_all_) {
                ready_cv_.wait(lk, [&] { return ready_ == (int)workers_.size(); });
                ready_all_ = true;
            }
            job_ = &fn;
            n_ = n;
            next_.store(0, std::memory_order_relaxed);
            remaining_ = (int)workers_.size();
            gen_.fetch_add(1, std::memory_order_release);
            if (sleepers_.load(std::memory_order_relaxed) > 0) {
                cv_.notify_all();
            }
        }
        int i;
        while ((i = next_.fetch_add(1, std::memory_order_relaxed)) < n) {
            fn(i);
        }
        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [&] { return remaining_ == 0; });
        job_ = nullptr;
    }

private:
    static constexpr int kSpin = 128;

    thread_pool() : nthreads_(detect_threads()) {
        for (int i = 0; i + 1 < nthreads_; i++) {
            workers_.emplace_back([this] { worker(); });
        }
    }
    ~thread_pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_.store(true, std::memory_order_relaxed);
            gen_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
        for (auto & t : workers_) {
            t.join();
        }
    }
    thread_pool(const thread_pool &) = delete;
    thread_pool & operator=(const thread_pool &) = delete;

    void worker() {
        uint64_t seen;
        {
            std::lock_guard<std::mutex> lk(m_);
            seen = gen_.load(std::memory_order_relaxed);
            ready_++;
            ready_cv_.notify_all();
        }
        for (;;) {
            int spins = 0;
            while (gen_.load(std::memory_order_acquire) == seen) {
                if (stop_.load(std::memory_order_relaxed)) {
                    return;
                }
                if (++spins < kSpin) {
                    _mm_pause();
                } else {
                    std::unique_lock<std::mutex> lk(m_);
                    sleepers_.fetch_add(1, std::memory_order_relaxed);
                    cv_.wait(lk, [&] {
                        return stop_.load(std::memory_order_relaxed) || gen_.load(std::memory_order_acquire) != seen;
                    });
                    sleepers_.fetch_sub(1, std::memory_order_relaxed);
                }
            }
            if (stop_.load(std::memory_order_relaxed)) {
                return;
            }
            seen = gen_.load(std::memory_order_acquire);
            const std::function<void(int)> * job = job_;
            const int n = n_;
            int i;
            while ((i = next_.fetch_add(1, std::memory_order_relaxed)) < n) {
                (*job)(i);
            }
            {
                std::lock_guard<std::mutex> lk(m_);
                if (--remaining_ == 0) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    int nthreads_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_, ready_cv_;
    const std::function<void(int)> * job_ = nullptr;
    int n_ = 0, remaining_ = 0;
    std::atomic<int> next_{0};
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> sleepers_{0};
    std::atomic<bool> stop_{false};
    int ready_ = 0;
    bool ready_all_ = false;
};

// ------------------------------------------------------------- SIMD kernels
__attribute__((target("avx2,fma"))) float dot_f32_avx2(const float * __restrict a, const float * __restrict b,
                                                       int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) float dot_f32_avx512(const float * __restrict a,
                                                                         const float * __restrict b, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

__attribute__((target("avx2,fma"))) float sumsq_f32_avx2(const float * __restrict a, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(a + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        s += a[i] * a[i];
    }
    return s;
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) float sumsq_f32_avx512(const float * __restrict a, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 v = _mm512_loadu_ps(a + i);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) {
        s += a[i] * a[i];
    }
    return s;
}

// int8 dot product: unsigned weight bytes (0..63) x signed activation bytes.
// AVX2 uses maddubs+madd; AVX-VNNI / AVX-512-VNNI use a single dpbusd.
__attribute__((target("avx2,fma"))) int32_t dot_i8_avx2(const uint8_t * __restrict w, const int8_t * __restrict x,
                                                        int n) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i wv = _mm256_loadu_si256((const __m256i *)(w + i));
        const __m256i xv = _mm256_loadu_si256((const __m256i *)(x + i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(wv, xv), ones));
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    int32_t s = _mm_cvtsi128_si32(lo);
    for (; i < n; i++) {
        s += (int)w[i] * (int)x[i];
    }
    return s;
}

__attribute__((target("avx2,avxvnni"))) int32_t dot_i8_vnni(const uint8_t * __restrict w, const int8_t * __restrict x,
                                                            int n) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i wv = _mm256_loadu_si256((const __m256i *)(w + i));
        const __m256i xv = _mm256_loadu_si256((const __m256i *)(x + i));
        acc = _mm256_dpbusd_epi32(acc, wv, xv);
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    int32_t s = _mm_cvtsi128_si32(lo);
    for (; i < n; i++) {
        s += (int)w[i] * (int)x[i];
    }
    return s;
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni"))) int32_t dot_i8_avx512(const uint8_t * __restrict w,
                                                                                     const int8_t * __restrict x,
                                                                                     int n) {
    __m512i acc = _mm512_setzero_si512();
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        const __m512i wv = _mm512_loadu_si512((const void *)(w + i));
        const __m512i xv = _mm512_loadu_si512((const void *)(x + i));
        acc = _mm512_dpbusd_epi32(acc, wv, xv);
    }
    int32_t s = _mm512_reduce_add_epi32(acc);
    for (; i < n; i++) {
        s += (int)w[i] * (int)x[i];
    }
    return s;
}

// ------------------------------------------------- fused quantized GEMV dot
// These replace the scalar reference dequant (quant.h) + separate fp32 dot:
// each 32-value sub-block is expanded with SIMD and immediately accumulated
// into the fp32 dot, which removes the reference dequant that profiling showed
// at ~60% of decode.  All four row formats use the same layout as quant.h.
__attribute__((target("avx2,fma"))) float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// one Q4_K superblock (256 values): y = d*sc*q - dmin*m, q in 0..15
__attribute__((target("avx2,fma"))) float q4k_dot_sb(const block_q4_K * blk, const float * x) {
    const float d = ggml_half_to_float(blk->d);
    const float mn = ggml_half_to_float(blk->dmin);
    const uint8_t * q = blk->qs;
    __m256 acc = _mm256_setzero_ps();
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(is + 0, blk->scales, &sc0, &m0);
        get_scale_min_k4(is + 1, blk->scales, &sc1, &m1);
        const __m256 d1 = _mm256_set1_ps(d * sc0), f1 = _mm256_set1_ps(mn * m0);
        const __m256 d2 = _mm256_set1_ps(d * sc1), f2 = _mm256_set1_ps(mn * m1);
        for (int l = 0; l < 32; l += 8) {
            const __m128i bytes = _mm_loadl_epi64((const __m128i *)(q + l));
            const __m256i vi = _mm256_cvtepu8_epi32(bytes);
            const __m256 lo = _mm256_cvtepi32_ps(_mm256_and_si256(vi, _mm256_set1_epi32(0xF)));
            const __m256 hi = _mm256_cvtepi32_ps(_mm256_srli_epi32(vi, 4));
            acc = _mm256_fmadd_ps(_mm256_fmsub_ps(lo, d1, f1), _mm256_loadu_ps(x + j + l), acc);
            acc = _mm256_fmadd_ps(_mm256_fmsub_ps(hi, d2, f2), _mm256_loadu_ps(x + j + 32 + l), acc);
        }
        q += 32;
        is += 2;
    }
    return hsum256(acc);
}

// one Q5_K superblock: Q4_K plus the 5th bit from qh (per 64-value group)
__attribute__((target("avx2,fma"))) float q5k_dot_sb(const block_q5_K * blk, const float * x) {
    const float d = ggml_half_to_float(blk->d);
    const float mn = ggml_half_to_float(blk->dmin);
    const uint8_t * q = blk->qs;
    const uint8_t * qh = blk->qh;
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i hi4 = _mm256_set1_epi32(0xF);
    __m256 acc = _mm256_setzero_ps();
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(is + 0, blk->scales, &sc0, &m0);
        get_scale_min_k4(is + 1, blk->scales, &sc1, &m1);
        const __m256 d1 = _mm256_set1_ps(d * sc0), f1 = _mm256_set1_ps(mn * m0);
        const __m256 d2 = _mm256_set1_ps(d * sc1), f2 = _mm256_set1_ps(mn * m1);
        const __m256i v1 = _mm256_set1_epi32(u1), v2 = _mm256_set1_epi32(u2);
        for (int l = 0; l < 32; l += 8) {
            const __m256i vi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(q + l)));
            const __m256i hb = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qh + l)));
            const __m256i lob = _mm256_slli_epi32(_mm256_min_epu32(_mm256_and_si256(hb, v1), one), 4);
            const __m256i hib = _mm256_slli_epi32(_mm256_min_epu32(_mm256_and_si256(hb, v2), one), 4);
            const __m256 lo = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_and_si256(vi, hi4), lob));
            const __m256 hi = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_srli_epi32(vi, 4), hib));
            acc = _mm256_fmadd_ps(_mm256_fmsub_ps(lo, d1, f1), _mm256_loadu_ps(x + j + l), acc);
            acc = _mm256_fmadd_ps(_mm256_fmsub_ps(hi, d2, f2), _mm256_loadu_ps(x + j + 32 + l), acc);
        }
        q += 32;
        is += 2;
        u1 = (uint8_t)(u1 << 2);
        u2 = (uint8_t)(u2 << 2);
    }
    return hsum256(acc);
}

// one Q6_K superblock: 6-bit values, per-16 scales, no min term
__attribute__((target("avx2,fma"))) float q6k_dot_sb(const block_q6_K * blk, const float * x) {
    const float d = ggml_half_to_float(blk->d);
    const uint8_t * ql = blk->ql;
    const uint8_t * qh = blk->qh;
    const int8_t * sc = blk->scales;
    const __m256i m3 = _mm256_set1_epi32(3);
    const __m256i c32 = _mm256_set1_epi32(32);
    __m256 acc = _mm256_setzero_ps();
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l += 8) {
            const int is = l / 16;
            const __m256 s0 = _mm256_set1_ps(d * sc[is + 0]);
            const __m256 s1 = _mm256_set1_ps(d * sc[is + 2]);
            const __m256 s2 = _mm256_set1_ps(d * sc[is + 4]);
            const __m256 s3 = _mm256_set1_ps(d * sc[is + 6]);
            const __m256i a = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(ql + l)));
            const __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(ql + l + 32)));
            const __m256i h = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qh + l)));
            __m256i q1 = _mm256_add_epi32(_mm256_and_si256(a, _mm256_set1_epi32(0xF)),
                                          _mm256_slli_epi32(_mm256_and_si256(h, m3), 4));
            __m256i q2 = _mm256_add_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0xF)),
                                          _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 2), m3), 4));
            __m256i q3 = _mm256_add_epi32(_mm256_srli_epi32(a, 4),
                                          _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 4), m3), 4));
            __m256i q4 = _mm256_add_epi32(_mm256_srli_epi32(b, 4),
                                          _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 6), m3), 4));
            q1 = _mm256_sub_epi32(q1, c32);
            q2 = _mm256_sub_epi32(q2, c32);
            q3 = _mm256_sub_epi32(q3, c32);
            q4 = _mm256_sub_epi32(q4, c32);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q1), s0), _mm256_loadu_ps(x + n + l), acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q2), s1), _mm256_loadu_ps(x + n + l + 32), acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q3), s2), _mm256_loadu_ps(x + n + l + 64), acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q4), s3), _mm256_loadu_ps(x + n + l + 96), acc);
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return hsum256(acc);
}

// 256 values = 8 Q8_0 blocks
__attribute__((target("avx2,fma"))) float q8_0_dot_sb(const block_q8_0 * w, const float * x) {
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < 8; b++) {
        const block_q8_0 * blk = w + b;
        const __m256 d = _mm256_set1_ps(ggml_half_to_float(blk->d));
        for (int l = 0; l < 32; l += 8) {
            const __m256i vi = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(blk->qs + l)));
            const __m256 vf = _mm256_mul_ps(_mm256_cvtepi32_ps(vi), d);
            acc = _mm256_fmadd_ps(vf, _mm256_loadu_ps(x + b * 32 + l), acc);
        }
    }
    return hsum256(acc);
}

__attribute__((target("avx2,fma"))) float qgemv_sb_avx2(uint32_t type, const char * w, const float * x) {
    switch (type) {
    case 12: return q4k_dot_sb((const block_q4_K *)w, x);
    case 13: return q5k_dot_sb((const block_q5_K *)w, x);
    case 14: return q6k_dot_sb((const block_q6_K *)w, x);
    case 8: return q8_0_dot_sb((const block_q8_0 *)w, x);
    default: return 0.0f;
    }
}

// runtime dispatch table (resolved once)
struct isa_table : cpu_isa_dispatch {
    isa_table() {
        switch (cpu_best_isa()) {
        case cpu_isa::avx512:
            dot_f32 = dot_f32_avx512;
            sumsq = sumsq_f32_avx512;
            break;
        case cpu_isa::avx2:
            dot_f32 = dot_f32_avx2;
            sumsq = sumsq_f32_avx2;
            break;
        default:
            dot_f32 = nullptr;
            sumsq = nullptr;
            break;
        }
        if (!dot_f32) {
            dot_f32 = [](const float * a, const float * b, int n) {
                float s = 0;
                for (int i = 0; i < n; i++) {
                    s += a[i] * b[i];
                }
                return s;
            };
            sumsq = [](const float * a, int n) {
                float s = 0;
                for (int i = 0; i < n; i++) {
                    s += a[i] * a[i];
                }
                return s;
            };
        }
        if (cpu_best_isa() == cpu_isa::avx512 && cpu_features_of().avx512vnni) {
            dot_i8 = dot_i8_avx512;
        } else if (cpu_has_avxvnni()) {
            dot_i8 = dot_i8_vnni;
        } else if (cpu_best_isa() != cpu_isa::scalar) {
            dot_i8 = dot_i8_avx2;
        } else {
            dot_i8 = [](const uint8_t * w, const int8_t * x, int n) {
                int32_t s = 0;
                for (int i = 0; i < n; i++) {
                    s += (int)w[i] * (int)x[i];
                }
                return s;
            };
        }
        qgemv_sb = (cpu_best_isa() != cpu_isa::scalar) ? qgemv_sb_avx2 : nullptr;
    }
};

} // namespace

const cpu_isa_dispatch & isa() {
    static const isa_table t;
    return t;
}

int cpu_thread_count() {
    return detect_threads();
}

void cpu_set_thread_count(int n) {
    g_cpu_threads_override = n > 0 ? n : 0;
}

void cpu_par(int n, const std::function<void(int)> & fn) {
    thread_pool::get().run(n, fn);
}

// ------------------------------------------------------------- weight dequant
int sb_bytes(uint32_t type) {
    switch (type) {
    case 11: return 110;
    case 12: return 144;
    case 13: return 176;
    case 14: return 210;
    case 20: return 8 * 18; // 8 IQ4_NL blocks per 256-value superblock
    case 21: return 110;
    case 23: return 136;
    case 8: return 8 * 34; // 8 Q8_0 blocks per 256-value superblock
    case 0: return 1024;
    default: return 0;
    }
}

int sb_step(uint32_t type) {
    (void)type;
    return 256; // every supported format dequantizes 256 values per call
}

int dequant_sb(uint32_t type, const char * p, float * dst) {
    switch (type) {
    case 12:
        dequantize_block_q4_K((const block_q4_K *)p, dst);
        return 256;
    case 13:
        dequantize_block_q5_K((const block_q5_K *)p, dst);
        return 256;
    case 14:
        dequantize_block_q6_K((const block_q6_K *)p, dst);
        return 256;
    case 8:
        for (int b = 0; b < 8; b++) {
            dequantize_block_q8_0((const block_q8_0 *)(p + b * 34), dst + b * 32);
        }
        return 256;
    case 11:
        dequantize_block_q3_K((const block_q3_K *)p, dst);
        return 256;
    case 20:
        for (int b = 0; b < 8; b++) {
            dequantize_block_iq4_nl((const block_iq4_nl *)(p + b * 18), dst + b * 32);
        }
        return 256;
    case 21:
        dequantize_block_iq3_s((const block_iq3_s *)p, dst);
        return 256;
    case 23:
        dequantize_block_iq4_xs((const block_iq4_xs *)p, dst);
        return 256;
    case 0:
        std::memcpy(dst, p, 256 * 4);
        return 256;
    default:
        return 0;
    }
}

void gemv_row(uint32_t type, const char * wrow, const float * x, const float * up, int K, float alpha,
              const float * residual, float * out) {
    const int step = sb_step(type);
    const int bytes = sb_bytes(type);
    if (type == 0) {
        const float s = dot_f32(x, (const float *)wrow, K);
        *out = alpha * s + (residual ? *residual : 0.0f);
        return;
    }
    float acc = 0.0f;
    float tmp[256];
    float xa[256];
    const bool need_act = (up != nullptr);
    // the fused single-pass dot only knows the K-quant/native types; the
    // IQ/Q3_K formats fall back to dequantize + fp32 dot
    const bool fused = (isa().qgemv_sb != nullptr) && (type == 12 || type == 13 || type == 14 || type == 8);
    for (int off = 0; off < K; off += step) {
        const char * wp = wrow + (size_t)(off / step) * bytes;
        const float * xr;
        if (need_act) {
            for (int i = 0; i < step; i++) {
                xa[i] = cpu_silu(x[off + i]) * up[off + i];
            }
            xr = xa;
        } else {
            xr = x + off;
        }
        if (fused) {
            acc += isa().qgemv_sb(type, wp, xr);
        } else {
            const int n = dequant_sb(type, wp, tmp);
            acc += dot_f32(tmp, xr, n);
        }
    }
    *out = alpha * acc + (residual ? *residual : 0.0f);
}

// ------------------------------------------------------------------ RMSNorm
void rmsnorm_row(const float * x, const float * w, float * out, int n, float eps) {
    const float ss = sumsq_f32(x, n);
    const float scale = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * scale * w[i];
    }
}

void rmsnorm_inplace(float * x, const float * w, int n, float eps) {
    const float ss = sumsq_f32(x, n);
    const float scale = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) {
        x[i] *= scale * w[i];
    }
}

} // namespace si
