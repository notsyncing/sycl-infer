// Host QSA kernels for qwen4exp: the indexer's BF16 projections and the sparse
// attention over the selected key rows.  Compiled with -fno-sycl (see
// CMakeLists) so the AVX2 / AVX-512 variants can carry intrinsics.
//
// The sparse attention is the only QSA loop with real arithmetic: 24 query
// heads x ~2048 selected keys x (q.k + softmax + v) per attention layer.  On
// the 2x A770 box (i9-10940X, AVX-512) it measured 33 ms/token at ctx ~2k and
// was the largest host item; the scalar V accumulation, the scalar exp and the
// per-key indirect `dot_f32` call all disappear here.  The decay of the
// accumulation order is accepted: the q4 stream is the oracle, and
// PF_CPU_ISA=scalar keeps the exact previous form for the score/mix A/B.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common.h"
#include "common/cpu_isa.h"
#include "quant.h"
#include "common/env.h"

#if !defined(__SYCL_DEVICE_ONLY__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define SI_QSA_X86 1
#endif

namespace si {

namespace {

#ifdef SI_QSA_X86

// A 16-value f16 load is exactly the f32 value the f32 mirror held: ggml's
// f16 round-trip is exact, so the arithmetic below is unchanged.  Templating
// the attention on the element type keeps one instruction sequence for both
// PF_QSA_F16MIR settings.
template <typename T> struct qsa_ld512;
template <> struct qsa_ld512<float> {
    __attribute__((target("avx512f,avx512bw,avx512vl"))) static __m512 load(const float * p) {
        return _mm512_loadu_ps(p);
    }
};
template <> struct qsa_ld512<uint16_t> {
    __attribute__((target("avx512f,avx512bw,avx512vl"))) static __m512 load(const uint16_t * p) {
        return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(const void *)p));
    }
};

// ---------------------------------------------------------------- AVX-512
__attribute__((target("avx512f,avx512bw,avx512vl"))) inline __m512 bf16x16(const uint16_t * p) {
    const __m256i h = _mm256_loadu_si256((const __m256i *)(const void *)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) inline __m512 exp512(__m512 x) {
    // exp(x) for x <= 0: range reduction by ln2 and a degree-6 Taylor series
    // (|r| <= 0.347, relative error ~1e-7, i.e. the same order as expf).
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f)),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.6931471805599453f), x);
    __m512 p = _mm512_set1_ps(1.0f / 720.0f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 120.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) float bf16_dot_avx512(const uint16_t * w,
                                                                            const float * __restrict x, int K) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(),
           a3 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 64 <= K; i += 64) {
        a0 = _mm512_fmadd_ps(bf16x16(w + i), _mm512_loadu_ps(x + i), a0);
        a1 = _mm512_fmadd_ps(bf16x16(w + i + 16), _mm512_loadu_ps(x + i + 16), a1);
        a2 = _mm512_fmadd_ps(bf16x16(w + i + 32), _mm512_loadu_ps(x + i + 32), a2);
        a3 = _mm512_fmadd_ps(bf16x16(w + i + 48), _mm512_loadu_ps(x + i + 48), a3);
    }
    for (; i + 16 <= K; i += 16) {
        a0 = _mm512_fmadd_ps(bf16x16(w + i), _mm512_loadu_ps(x + i), a0);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
    for (; i < K; i++) {
        s += ggml_bf16_to_float(w[i]) * x[i];
    }
    return s;
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) float dot_avx512(const float * __restrict a,
                                                                     const float * __restrict b, int n) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(),
           a3 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), a3);
    }
    for (; i + 16 <= n; i += 16) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), a0);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
    for (; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

template <typename T>
__attribute__((target("avx512f,avx512bw,avx512vl"))) void attn_head_avx512_t(
    const float * q, const T * kbase, const T * vbase, int kv_stride, const int * sel, int nsel, int hd,
    float scale, float * out) {
    if (hd != 256) { // the model's geometry; the generic scalar form otherwise
        return;
    }
    thread_local std::vector<float> sc;
    sc.resize((size_t)nsel);
    const __m512 q0 = _mm512_loadu_ps(q + 0), q1 = _mm512_loadu_ps(q + 16), q2 = _mm512_loadu_ps(q + 32),
                q3 = _mm512_loadu_ps(q + 48), q4 = _mm512_loadu_ps(q + 64), q5 = _mm512_loadu_ps(q + 80),
                q6 = _mm512_loadu_ps(q + 96), q7 = _mm512_loadu_ps(q + 112), q8 = _mm512_loadu_ps(q + 128),
                q9 = _mm512_loadu_ps(q + 144), q10 = _mm512_loadu_ps(q + 160), q11 = _mm512_loadu_ps(q + 176),
                q12 = _mm512_loadu_ps(q + 192), q13 = _mm512_loadu_ps(q + 208), q14 = _mm512_loadu_ps(q + 224),
                q15 = _mm512_loadu_ps(q + 240);
    __m512 vmax = _mm512_set1_ps(-3.402823466e38f);
    for (int j = 0; j < nsel; j++) {
        const T * kp = kbase + (size_t)sel[j] * kv_stride;
        __m512 a0 = _mm512_mul_ps(q0, qsa_ld512<T>::load(kp + 0));
        __m512 a1 = _mm512_mul_ps(q1, qsa_ld512<T>::load(kp + 16));
        __m512 a2 = _mm512_mul_ps(q2, qsa_ld512<T>::load(kp + 32));
        __m512 a3 = _mm512_mul_ps(q3, qsa_ld512<T>::load(kp + 48));
        a0 = _mm512_fmadd_ps(q4, qsa_ld512<T>::load(kp + 64), a0);
        a1 = _mm512_fmadd_ps(q5, qsa_ld512<T>::load(kp + 80), a1);
        a2 = _mm512_fmadd_ps(q6, qsa_ld512<T>::load(kp + 96), a2);
        a3 = _mm512_fmadd_ps(q7, qsa_ld512<T>::load(kp + 112), a3);
        a0 = _mm512_fmadd_ps(q8, qsa_ld512<T>::load(kp + 128), a0);
        a1 = _mm512_fmadd_ps(q9, qsa_ld512<T>::load(kp + 144), a1);
        a2 = _mm512_fmadd_ps(q10, qsa_ld512<T>::load(kp + 160), a2);
        a3 = _mm512_fmadd_ps(q11, qsa_ld512<T>::load(kp + 176), a3);
        a0 = _mm512_fmadd_ps(q12, qsa_ld512<T>::load(kp + 192), a0);
        a1 = _mm512_fmadd_ps(q13, qsa_ld512<T>::load(kp + 208), a1);
        a2 = _mm512_fmadd_ps(q14, qsa_ld512<T>::load(kp + 224), a2);
        a3 = _mm512_fmadd_ps(q15, qsa_ld512<T>::load(kp + 240), a3);
        const float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3))) * scale;
        sc[(size_t)j] = s;
        vmax = _mm512_max_ps(vmax, _mm512_set1_ps(s));
    }
    const __m512 mx = _mm512_set1_ps(_mm512_reduce_max_ps(vmax));
    __m512 vsum = _mm512_setzero_ps();
    int jt = 0;
    for (; jt + 16 <= nsel; jt += 16) {
        const __m512 e = exp512(_mm512_sub_ps(_mm512_loadu_ps(sc.data() + jt), mx));
        _mm512_storeu_ps(sc.data() + jt, e);
        vsum = _mm512_add_ps(vsum, e);
    }
    float tail = 0.f;
    for (; jt < nsel; jt++) {
        sc[(size_t)jt] = std::exp(sc[(size_t)jt] - _mm_cvtss_f32(_mm512_castps512_ps128(mx)));
        tail += sc[(size_t)jt];
    }
    const float inv = 1.0f / (_mm512_reduce_add_ps(vsum) + tail);
    __m512 o0 = _mm512_setzero_ps(), o1 = _mm512_setzero_ps(), o2 = _mm512_setzero_ps(),
           o3 = _mm512_setzero_ps(), o4 = _mm512_setzero_ps(), o5 = _mm512_setzero_ps(),
           o6 = _mm512_setzero_ps(), o7 = _mm512_setzero_ps(), o8 = _mm512_setzero_ps(),
           o9 = _mm512_setzero_ps(), o10 = _mm512_setzero_ps(), o11 = _mm512_setzero_ps(),
           o12 = _mm512_setzero_ps(), o13 = _mm512_setzero_ps(), o14 = _mm512_setzero_ps(),
           o15 = _mm512_setzero_ps();
    for (int j = 0; j < nsel; j++) {
        const __m512 w = _mm512_set1_ps(sc[(size_t)j] * inv);
        const T * vp = vbase + (size_t)sel[j] * kv_stride;
        o0 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 0), o0);
        o1 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 16), o1);
        o2 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 32), o2);
        o3 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 48), o3);
        o4 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 64), o4);
        o5 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 80), o5);
        o6 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 96), o6);
        o7 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 112), o7);
        o8 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 128), o8);
        o9 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 144), o9);
        o10 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 160), o10);
        o11 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 176), o11);
        o12 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 192), o12);
        o13 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 208), o13);
        o14 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 224), o14);
        o15 = _mm512_fmadd_ps(w, qsa_ld512<T>::load(vp + 240), o15);
    }
    const float * gate = q + 256;
#define SI_GATE(B, O)                                                                                                  \
    do {                                                                                                               \
        const __m512 g = _mm512_loadu_ps(gate + (B) * 16);                                                              \
        const __m512 sg = _mm512_div_ps(_mm512_set1_ps(1.0f),                                                           \
                                        _mm512_add_ps(_mm512_set1_ps(1.0f), exp512(_mm512_sub_ps(_mm512_setzero_ps(), g)))); \
        _mm512_storeu_ps(out + (B) * 16, _mm512_mul_ps(O, sg));                                                         \
    } while (0)
    SI_GATE(0, o0);
    SI_GATE(1, o1);
    SI_GATE(2, o2);
    SI_GATE(3, o3);
    SI_GATE(4, o4);
    SI_GATE(5, o5);
    SI_GATE(6, o6);
    SI_GATE(7, o7);
    SI_GATE(8, o8);
    SI_GATE(9, o9);
    SI_GATE(10, o10);
    SI_GATE(11, o11);
    SI_GATE(12, o12);
    SI_GATE(13, o13);
    SI_GATE(14, o14);
    SI_GATE(15, o15);
#undef SI_GATE
}

void attn_head_avx512(const float * q, const float * kbase, const float * vbase, int kv_stride, const int * sel,
                      int nsel, int hd, float scale, float * out) {
    attn_head_avx512_t<float>(q, kbase, vbase, kv_stride, sel, nsel, hd, scale, out);
}

void attn_head_avx512_h(const float * q, const uint16_t * kbase, const uint16_t * vbase, int kv_stride,
                        const int * sel, int nsel, int hd, float scale, float * out) {
    attn_head_avx512_t<uint16_t>(q, kbase, vbase, kv_stride, sel, nsel, hd, scale, out);
}

// Serial on purpose: 512 pools x 4 heads x 128 values is ~2.7 us for a layer
// with AVX-512, below the cost of waking the thread pool once per layer.
__attribute__((target("avx512f,avx512bw,avx512vl"))) void score_pools_avx512(
    const float * q, const float * pool, int n_pool, int id, int inh, float inv, float * scores) {
    for (int p = 0; p < n_pool; p++) {
        float s = 0.f;
        for (int h = 0; h < inh; h++) {
            const float d = dot_avx512(q + (size_t)h * id, pool + (size_t)p * id, id);
            s += d > 0.f ? d : 0.f;
        }
        scores[(size_t)p] = s * inv;
    }
}

// ------------------------------------------------------------------ AVX2
template <typename T> struct qsa_ld256;
template <> struct qsa_ld256<float> {
    __attribute__((target("avx2,fma"))) static __m256 load(const float * p) {
        return _mm256_loadu_ps(p);
    }
};
template <> struct qsa_ld256<uint16_t> {
    __attribute__((target("avx2,fma,f16c"))) static __m256 load(const uint16_t * p) {
        return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)p));
    }
};

__attribute__((target("avx2,fma"))) inline __m256 bf16x8(const uint16_t * p) {
    const __m128i h = _mm_loadu_si128((const __m128i *)(const void *)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

__attribute__((target("avx2,fma"))) inline __m256 exp256(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.6931471805599453f), x);
    __m256 p = _mm256_set1_ps(1.0f / 720.0f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

__attribute__((target("avx2,fma"))) float bf16_dot_avx2(const uint16_t * w, const float * __restrict x, int K) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(),
           a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= K; i += 32) {
        a0 = _mm256_fmadd_ps(bf16x8(w + i), _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(bf16x8(w + i + 8), _mm256_loadu_ps(x + i + 8), a1);
        a2 = _mm256_fmadd_ps(bf16x8(w + i + 16), _mm256_loadu_ps(x + i + 16), a2);
        a3 = _mm256_fmadd_ps(bf16x8(w + i + 24), _mm256_loadu_ps(x + i + 24), a3);
    }
    for (; i + 8 <= K; i += 8) {
        a0 = _mm256_fmadd_ps(bf16x8(w + i), _mm256_loadu_ps(x + i), a0);
    }
    __m256 a = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(a);
    __m128 hi = _mm256_extractf128_ps(a, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < K; i++) {
        s += ggml_bf16_to_float(w[i]) * x[i];
    }
    return s;
}

__attribute__((target("avx2,fma"))) float dot_avx2(const float * __restrict a, const float * __restrict b, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(),
           a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
    }
    for (; i + 8 <= n; i += 8) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    }
    const __m256 av = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(av);
    __m128 hi = _mm256_extractf128_ps(av, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

template <typename T>
__attribute__((target("avx2,fma,f16c"))) void attn_head_avx2_t(const float * q, const T * kbase, const T * vbase,
                                                              int kv_stride, const int * sel, int nsel, int hd,
                                                              float scale, float * out) {
    if (hd != 256) {
        return;
    }
    thread_local std::vector<float> sc;
    sc.resize((size_t)nsel);
    __m256 qv[32];
    for (int b = 0; b < 32; b++) {
        qv[b] = _mm256_loadu_ps(q + b * 8);
    }
    __m256 vmax = _mm256_set1_ps(-3.402823466e38f);
    for (int j = 0; j < nsel; j++) {
        const T * kp = kbase + (size_t)sel[j] * kv_stride;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(),
               a3 = _mm256_setzero_ps();
        for (int b = 0; b < 32; b += 4) {
            a0 = _mm256_fmadd_ps(qv[b], qsa_ld256<T>::load(kp + b * 8), a0);
            a1 = _mm256_fmadd_ps(qv[b + 1], qsa_ld256<T>::load(kp + (b + 1) * 8), a1);
            a2 = _mm256_fmadd_ps(qv[b + 2], qsa_ld256<T>::load(kp + (b + 2) * 8), a2);
            a3 = _mm256_fmadd_ps(qv[b + 3], qsa_ld256<T>::load(kp + (b + 3) * 8), a3);
        }
        __m256 a = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        __m128 lo = _mm256_castps256_ps128(a);
        lo = _mm_add_ps(lo, _mm256_extractf128_ps(a, 1));
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        const float s = _mm_cvtss_f32(lo) * scale;
        sc[(size_t)j] = s;
        vmax = _mm256_max_ps(vmax, _mm256_set1_ps(s));
    }
    __m128 hmax = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
    hmax = _mm_max_ps(hmax, _mm_movehl_ps(hmax, hmax));
    hmax = _mm_max_ss(hmax, _mm_shuffle_ps(hmax, hmax, 1));
    const __m256 mx = _mm256_broadcastss_ps(hmax);
    __m256 vsum = _mm256_setzero_ps();
    int jt = 0;
    for (; jt + 8 <= nsel; jt += 8) {
        const __m256 e = exp256(_mm256_sub_ps(_mm256_loadu_ps(sc.data() + jt), mx));
        _mm256_storeu_ps(sc.data() + jt, e);
        vsum = _mm256_add_ps(vsum, e);
    }
    float tail = 0.f;
    for (; jt < nsel; jt++) {
        sc[(size_t)jt] = std::exp(sc[(size_t)jt] - _mm_cvtss_f32(_mm256_castps256_ps128(mx)));
        tail += sc[(size_t)jt];
    }
    __m128 sl = _mm_add_ps(_mm256_castps256_ps128(vsum), _mm256_extractf128_ps(vsum, 1));
    sl = _mm_hadd_ps(sl, sl);
    sl = _mm_hadd_ps(sl, sl);
    const float inv = 1.0f / (_mm_cvtss_f32(sl) + tail);
    __m256 o[32];
    for (int b = 0; b < 32; b++) {
        o[b] = _mm256_setzero_ps();
    }
    for (int j = 0; j < nsel; j++) {
        const __m256 w = _mm256_set1_ps(sc[(size_t)j] * inv);
        const T * vp = vbase + (size_t)sel[j] * kv_stride;
        for (int b = 0; b < 32; b++) {
            o[b] = _mm256_fmadd_ps(w, qsa_ld256<T>::load(vp + b * 8), o[b]);
        }
    }
    const float * gate = q + 256;
    for (int b = 0; b < 32; b++) {
        const __m256 g = _mm256_loadu_ps(gate + b * 8);
        const __m256 sg = _mm256_div_ps(_mm256_set1_ps(1.0f),
                                        _mm256_add_ps(_mm256_set1_ps(1.0f), exp256(_mm256_sub_ps(_mm256_setzero_ps(), g))));
        _mm256_storeu_ps(out + b * 8, _mm256_mul_ps(o[b], sg));
    }
}

void attn_head_avx2(const float * q, const float * kbase, const float * vbase, int kv_stride, const int * sel,
                    int nsel, int hd, float scale, float * out) {
    attn_head_avx2_t<float>(q, kbase, vbase, kv_stride, sel, nsel, hd, scale, out);
}

void attn_head_avx2_h(const float * q, const uint16_t * kbase, const uint16_t * vbase, int kv_stride,
                      const int * sel, int nsel, int hd, float scale, float * out) {
    attn_head_avx2_t<uint16_t>(q, kbase, vbase, kv_stride, sel, nsel, hd, scale, out);
}

__attribute__((target("avx2,fma"))) void score_pools_avx2(const float * q, const float * pool, int n_pool, int id,
                                                          int inh, float inv, float * scores) {
    for (int p = 0; p < n_pool; p++) {
        float s = 0.f;
        for (int h = 0; h < inh; h++) {
            const float d = dot_avx2(q + (size_t)h * id, pool + (size_t)p * id, id);
            s += d > 0.f ? d : 0.f;
        }
        scores[(size_t)p] = s * inv;
    }
}

#endif // SI_QSA_X86

// ---------------------------------------------------------------- scalar
float bf16_dot_scalar(const uint16_t * w, const float * __restrict x, int K) {
    float acc = 0.f;
    for (int k = 0; k < K; k++) {
        acc += ggml_bf16_to_float(w[k]) * x[k];
    }
    return acc;
}

template <typename T> inline float qsa_load_scalar(T v);
template <> inline float qsa_load_scalar<float>(float v) {
    return v;
}
template <> inline float qsa_load_scalar<uint16_t>(uint16_t v) {
    return ggml_half_to_float(v);
}

template <typename T>
void attn_head_scalar_t(const float * q, const T * kbase, const T * vbase, int kv_stride, const int * sel,
                        int nsel, int hd, float scale, float * out) {
    thread_local std::vector<float> sc;
    sc.resize((size_t)nsel);
    float mx = -3.402823466e38f;
    for (int j = 0; j < nsel; j++) {
        const T * kp = kbase + (size_t)sel[j] * kv_stride;
        float d = 0.f;
        for (int b = 0; b < hd; b++) {
            d += q[b] * qsa_load_scalar<T>(kp[b]);
        }
        const float s = d * scale;
        sc[(size_t)j] = s;
        mx = std::max(mx, s);
    }
    float sum = 0.f;
    for (int j = 0; j < nsel; j++) {
        sc[(size_t)j] = std::exp(sc[(size_t)j] - mx);
        sum += sc[(size_t)j];
    }
    const float inv = 1.f / sum;
    for (int d = 0; d < hd; d++) {
        out[d] = 0.f;
    }
    for (int j = 0; j < nsel; j++) {
        const float w = sc[(size_t)j] * inv;
        const T * vp = vbase + (size_t)sel[j] * kv_stride;
        for (int d = 0; d < hd; d++) {
            out[d] += w * qsa_load_scalar<T>(vp[d]);
        }
    }
    const float * gate = q + hd;
    for (int d = 0; d < hd; d++) {
        out[d] *= 1.f / (1.f + std::exp(-gate[d]));
    }
}

void attn_head_scalar(const float * q, const float * kbase, const float * vbase, int kv_stride, const int * sel,
                      int nsel, int hd, float scale, float * out) {
    attn_head_scalar_t<float>(q, kbase, vbase, kv_stride, sel, nsel, hd, scale, out);
}

void attn_head_scalar_h(const float * q, const uint16_t * kbase, const uint16_t * vbase, int kv_stride,
                        const int * sel, int nsel, int hd, float scale, float * out) {
    attn_head_scalar_t<uint16_t>(q, kbase, vbase, kv_stride, sel, nsel, hd, scale, out);
}

void score_pools_scalar(const float * q, const float * pool, int n_pool, int id, int inh, float inv, float * scores) {
    for (int p = 0; p < n_pool; p++) {
        float s = 0.f;
        for (int h = 0; h < inh; h++) {
            float d = 0.f;
            for (int k = 0; k < id; k++) {
                d += q[(size_t)h * id + k] * pool[(size_t)p * id + k];
            }
            s += d > 0.f ? d : 0.f;
        }
        scores[(size_t)p] = s * inv;
    }
}

struct qsa_dispatch {
    float (*bf16_dot)(const uint16_t *, const float *, int);
    void (*attn_head)(const float *, const float *, const float *, int, const int *, int, int, float, float *);
    void (*attn_head_h)(const float *, const uint16_t *, const uint16_t *, int, const int *, int, int, float, float *);
    void (*score_pools)(const float *, const float *, int, int, int, float, float *);
};

const qsa_dispatch & qsa_isa() {
    static const qsa_dispatch d = [] {
#ifdef SI_QSA_X86
        // PF_QSA_ISA isolates the QSA kernels from the rest of the CPU backend:
        // PF_CPU_ISA=scalar also scalarises the host MoE/HC/PLE, which is not
        // an A/B of this path (one env value, fixed for the process).
        const char * f = si::env::str("PF_QSA_ISA");
        const bool force_scalar = f && (std::strcmp(f, "scalar") == 0 || std::strcmp(f, "0") == 0);
        const bool force_avx2 = f && std::strcmp(f, "avx2") == 0;
        const bool force_avx512 = f && std::strcmp(f, "avx512") == 0;
        const cpu_features & cf = cpu_features_of();
        const bool can512 = !force_scalar && cf.avx512f && cf.avx512bw;
        const bool can256 = !force_scalar && cf.avx2 && cf.fma;
        const cpu_isa best = cpu_best_isa();
        // A forced variant still has to be runnable: PF_QSA_ISA=avx512 on an
        // AVX2-only host falls back rather than executing a SIGILL.
        if ((force_avx512 && can512) || (!f && best == cpu_isa::avx512)) {
            return qsa_dispatch{bf16_dot_avx512, attn_head_avx512, attn_head_avx512_h, score_pools_avx512};
        }
        if ((force_avx2 && can256) || (force_avx512 && can256) || (!f && best == cpu_isa::avx2)) {
            return qsa_dispatch{bf16_dot_avx2, attn_head_avx2, attn_head_avx2_h, score_pools_avx2};
        }
#endif
        return qsa_dispatch{bf16_dot_scalar, attn_head_scalar, attn_head_scalar_h, score_pools_scalar};
    }();
    return d;
}

} // namespace

void cpu_qsa_bf16_gemv(const void * w, const float * x, int xs, float * out, int os, int n_tok, int K, int N) {
    // Rows per parallel job.  The dot itself is only ~160 AVX-512 FMAs, so a
    // job per output row is dispatch-bound: 640 rows/layer x 12 layers is
    // ~7 700 jobs/token and measured ~3 ms even with AVX-512.  Eight rows per
    // job cuts the job count 8x and keeps x hot; prefill (1024 tokens) is the
    // same picture at 8x the rows.
    constexpr int kRowsPerJob = 8;
    const uint16_t * wp = (const uint16_t *)w;
    const int jobs_per_tok = (N + kRowsPerJob - 1) / kRowsPerJob;
    par(n_tok * jobs_per_tok, [&](int j) {
        const int t = j / jobs_per_tok;
        const int n0 = (j % jobs_per_tok) * kRowsPerJob;
        const int ne = std::min(N, n0 + kRowsPerJob);
        const float * xr = x + (size_t)t * xs;
        for (int n = n0; n < ne; n++) {
            out[(size_t)t * os + n] = qsa_isa().bf16_dot(wp + (size_t)n * K, xr, K);
        }
    });
}

void cpu_qsa_score_pools(const float * q, const float * pool, int n_pool, int id, int inh, float inv, float * scores) {
    qsa_isa().score_pools(q, pool, n_pool, id, inh, inv, scores);
}

bool cpu_qsa_f16_ok() {
    // The f16 read is an F16C cvtph in the AVX2/AVX-512 forms.  The scalar form
    // has to call ggml_half_to_float per element, which stops the compiler
    // auto-vectorising the loops: measured 4.6x slower on the 128-token sparse
    // prefill (12 477 vs 2 721 ms), so PF_QSA_F16MIR defaults off there and an
    // explicit =1 still forces it.
    return qsa_isa().attn_head != attn_head_scalar;
}

void cpu_qsa_attn_head(const float * q, const float * kbase, const float * vbase, int kv_stride, const int * sel,
                       int nsel, int hd, float scale, float * out) {
    if (hd != 256) { // the vector forms are specialised for the model's head
        attn_head_scalar(q, kbase, vbase, kv_stride, sel, nsel, hd, scale, out);
        return;
    }
    qsa_isa().attn_head(q, kbase, vbase, kv_stride, sel, nsel, hd, scale, out);
}

void cpu_qsa_attn_head_h(const float * q, const uint16_t * kbase, const uint16_t * vbase, int kv_stride,
                         const int * sel, int nsel, int hd, float scale, float * out) {
    if (hd != 256) {
        attn_head_scalar_h(q, kbase, vbase, kv_stride, sel, nsel, hd, scale, out);
        return;
    }
    qsa_isa().attn_head_h(q, kbase, vbase, kv_stride, sel, nsel, hd, scale, out);
}

} // namespace si
