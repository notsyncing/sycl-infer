#pragma once
// ---------------------------------------------------------------------------
// KV cache storage type.  The paged K/V pools hold the values as written by
// qk_norm_rope (K is post-RMSNorm+RoPE, V is the raw projection) and are read
// by the flash attention kernel; all arithmetic stays fp32, only the storage
// is narrowed.  PF_KV_TYPE=i8 (default) stores int8 with one fp16 scale per 32
// head dimensions (3 KB/token/layer per K+V vs 6 with bf16, 12 with f32);
// PF_KV_TYPE=bf16 is the previous default, f16 the same size with a smaller
// exponent, f32 restores the historical pools.  PF_KV_F32=1 is an alias for
// PF_KV_TYPE=f32.
//   i8 = int8 with one fp16 scale per 32 head dimensions.  A row (one token of
//   one kv head) is head_dim bytes; the per-(block, kv head) unit appends a
//   scale plane: kBlockSize rows x (head_dim/32) fp16.  The values are
//   symmetric int8 in [-127, 127] and are dequantized to fp32 in the kernel.
// ---------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>                // IWYU pragma: keep
#include <sycl/ext/oneapi/bfloat16.hpp> // IWYU pragma: keep

namespace si {

enum class kv_dtype_t : int { f32 = 0, bf16 = 1, f16 = 2, i8 = 3 };

// selected once per process from PF_KV_TYPE / PF_KV_F32 / PF_KV_BF16
kv_dtype_t kv_dtype();
const char * kv_dtype_name(kv_dtype_t t);

inline int kv_dtype_bytes(kv_dtype_t t) {
    return t == kv_dtype_t::f32 ? 4 : (t == kv_dtype_t::i8 ? 1 : 2);
}

// host-side element access (used by the pool reader in engine and the tests).
// For i8 this returns the raw int8 value; the scale handling lives in
// engine::kv_read_vec, which knows the row geometry.
inline float kv_ld_host(kv_dtype_t t, const void * base, size_t i) {
    if (t == kv_dtype_t::f32) {
        return ((const float *)base)[i];
    }
    if (t == kv_dtype_t::bf16) {
        return (float)((const sycl::ext::oneapi::bfloat16 *)base)[i];
    }
    if (t == kv_dtype_t::i8) {
        return (float)((const int8_t *)base)[i];
    }
    return (float)((const sycl::half *)base)[i];
}

} // namespace si
