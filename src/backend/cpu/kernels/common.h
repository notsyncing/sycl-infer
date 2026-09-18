#pragma once
// ---------------------------------------------------------------------------
// Shared host-side helpers for the CPU kernels in this directory.
//
// This header is SYCL-free (the whole directory is compiled with -fno-sycl):
// mirrors of the engine PODs live in cpu_types.h, the ISA dispatch table and
// the parallel-for pool in common.cpp.  Everything here mirrors the SYCL
// kernels in src/backend/gpu/kernels/ so the CPU backend is numerically
// equivalent to the GPU one.
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>

#include "cpu_types.h" // mirrors + launcher declarations (no SYCL)
#include "quant.h"     // ggml block formats + host dequant

namespace si {

constexpr int kCpuT = 32;   // kMaxT
constexpr int kCpuB = 16;   // kMaxB
constexpr int kCpuBlk = 32; // kBlockSize
// max segments flattened into one parallel region by cpu_gemv_group
constexpr int kCpuGemvMaxSegs = 64;

inline float cpu_silu(float x) {
    return x / (1.0f + std::exp(-x));
}
inline float cpu_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ---- parallel for (persistent worker pool, see common.cpp) ----------------
void cpu_par(int n, const std::function<void(int)> & fn);
template <typename F> inline void par(int n, F && f) {
    cpu_par(n, std::function<void(int)>(std::forward<F>(f)));
}

// ---- ISA-dispatched primitives (see common.cpp) ---------------------------
struct cpu_isa_dispatch {
    float (*dot_f32)(const float *, const float *, int);
    float (*sumsq)(const float *, int);
    int32_t (*dot_i8)(const uint8_t *, const int8_t *, int);
    // fused dequant + dot for one 256-value superblock (null = scalar fallback)
    float (*qgemv_sb)(uint32_t type, const char * w, const float * x);
};
const cpu_isa_dispatch & isa();
inline float dot_f32(const float * a, const float * b, int n) {
    return isa().dot_f32(a, b, n);
}
inline float sumsq_f32(const float * a, int n) {
    return isa().sumsq(a, n);
}

// ---- fused dequantized GEMV (see common.cpp) ------------------------------
int sb_bytes(uint32_t type);
int sb_step(uint32_t type);
int dequant_sb(uint32_t type, const char * p, float * dst);
// out = alpha * sum_k W[row][k] * act(x[k]) (+ residual); act = silu(x)*up
void gemv_row(uint32_t type, const char * wrow, const float * x, const float * up, int K, float alpha,
              const float * residual, float * out);

// ---- RMSNorm helpers (shared with qk_norm_rope) ---------------------------
void rmsnorm_row(const float * x, const float * w, float * out, int n, float eps);
void rmsnorm_inplace(float * x, const float * w, int n, float eps);

// ---- KV element access (all storage dtypes, see kv_type.h) -----------------
inline float kv_load(cpu_kv_dtype t, const void * base, size_t i) {
    switch (t) {
    case cpu_kv_dtype::f32: return ((const float *)base)[i];
    case cpu_kv_dtype::bf16: {
        uint32_t bits = ((const uint16_t *)base)[i];
        bits <<= 16;
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }
    case cpu_kv_dtype::f16: return ggml_half_to_float(((const uint16_t *)base)[i]);
    default: return (float)((const int8_t *)base)[i];
    }
}
inline void kv_store(cpu_kv_dtype t, void * base, size_t i, float v) {
    switch (t) {
    case cpu_kv_dtype::f32: ((float *)base)[i] = v; break;
    case cpu_kv_dtype::bf16: {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        ((uint16_t *)base)[i] = (uint16_t)(bits >> 16);
    } break;
    case cpu_kv_dtype::f16: ((uint16_t *)base)[i] = ggml_float_to_half(v); break;
    default: ((int8_t *)base)[i] = (int8_t)std::max(-127.0f, std::min(127.0f, std::round(v))); break;
    }
}
inline size_t kv_row_off(size_t unit, int ko, int head_dim) {
    return (unit * kCpuBlk + ko) * (size_t)head_dim;
}
inline size_t kv_scale_off(size_t unit, int ko, int head_dim) {
    return (unit * kCpuBlk + ko) * (size_t)(head_dim / kCpuI8Q);
}

// ---- RoPE (plain + the 4-section M-RoPE used by the text model) ------------
inline int mrope_section(const cpu_step_info * info, int pair) {
    const int s0 = info->mrope_sections[0];
    const int s1 = info->mrope_sections[1];
    const int s2 = info->mrope_sections[2];
    const int s3 = info->mrope_sections[3];
    const int sect = s0 + s1 + s2 + s3;
    const int sector = sect > 0 ? (pair % sect) : pair;
    if (sector % 3 == 1 && sector < 3 * s1) {
        return 1;
    }
    if (sector % 3 == 2 && sector < 3 * s2) {
        return 2;
    }
    if (sector % 3 == 0 && sector < 3 * s0) {
        return 0;
    }
    return 3;
}
inline void rope_apply(float * v, int n_rot, float base, float rpos) {
    const int half = n_rot / 2;
    for (int i = 0; i < half; i++) {
        const float ang = rpos * std::exp2(-2.0f * i / n_rot * std::log2(base));
        const float c = std::cos(ang), s = std::sin(ang);
        const float x0 = v[i], x1 = v[i + half];
        v[i] = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
    }
}

} // namespace si
