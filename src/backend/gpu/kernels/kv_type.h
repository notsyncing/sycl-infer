#pragma once
// ---------------------------------------------------------------------------
// KV cache storage type.  The paged K/V pools hold the values as written by
// qk_norm_rope (K is post-RMSNorm+RoPE, V is the raw projection) and are read
// by the flash attention kernel; all arithmetic stays fp32, only the storage
// is narrowed.  PF_KV_TYPE=i8 (default) stores int8 with one fp16 scale per 32
// head dimensions (3 KB/token/layer per K+V vs 6 with bf16, 12 with f32);
// PF_KV_TYPE=i4 packs two signed 4-bit values per byte with the same per-32
// fp16 scales (1.5 KB/token/layer, half of i8); PF_KV_TYPE=bf16 is the
// previous default, f16 the same size with a smaller exponent, f32 restores the
// historical pools.  PF_KV_F32=1 is an alias for PF_KV_TYPE=f32.  The CLI
// `--kv-type` overrides the environment (see kv_dtype_set).
//   i8 = int8 with one fp16 scale per 32 head dimensions.  A row (one token of
//   one kv head) is head_dim bytes; the per-(block, kv head) unit appends a
//   scale plane: kBlockSize rows x (head_dim/32) fp16.  The values are
//   symmetric int8 in [-127, 127] and are dequantized to fp32 in the kernel.
//   i4 = two signed 4-bit values per byte (low nibble = even head dim), one
//   fp16 scale per 32 head dimensions.  A row is head_dim/2 bytes; the scale
//   plane is identical to i8.  Values are symmetric in [-7, 7], stored as
//   two's-complement nibbles and dequantized to fp32 in the kernel.
// ---------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>                // IWYU pragma: keep
#include <sycl/ext/oneapi/bfloat16.hpp> // IWYU pragma: keep

namespace si {

enum class kv_dtype_t : int { f32 = 0, bf16 = 1, f16 = 2, i8 = 3, i4 = 4 };

// selected from --kv-type, else once per process from PF_KV_TYPE / PF_KV_F32 /
// PF_KV_BF16.  `--kv-type K:V` (or PF_KV_TYPE=K:V) selects the K and V storage
// independently; a single name sets both.  Mixed types are supported for the
// two scale-carrying types (i4/i8) and for the different-width fp types.
kv_dtype_t kv_dtype();    // the K type (legacy single-type callers)
kv_dtype_t kv_k_dtype();
kv_dtype_t kv_v_dtype();
const char * kv_dtype_name(kv_dtype_t t);
// Return false and print nothing when `spec` is not a known KV type name.
bool kv_dtype_parse(const char * spec, kv_dtype_t & out);
// `K:V` or a single name -> k and v.  Returns false for an unknown name or an
// unsupported mix.
bool kv_dtype_parse_pair(const char * spec, kv_dtype_t & k, kv_dtype_t & v);
// Override the process-wide KV type (CLI --kv-type).  Must be called before the
// engine (and thus kv_setup) is constructed.
void kv_dtype_set(kv_dtype_t t);
void kv_dtype_set_kv(kv_dtype_t k, kv_dtype_t v);
// True when K and V may use `k`/`v`: the scale-carrying i4/i8 pair, or equal.
bool kv_dtype_mix_ok(kv_dtype_t k, kv_dtype_t v);

inline int kv_dtype_bytes(kv_dtype_t t) {
    return t == kv_dtype_t::f32 ? 4 : (t == kv_dtype_t::i8 ? 1 : (t == kv_dtype_t::i4 ? 1 : 2));
}

// bits per stored element (i4 packs two elements per byte)
inline int kv_dtype_bits(kv_dtype_t t) {
    switch (t) {
    case kv_dtype_t::f32: return 32;
    case kv_dtype_t::i8: return 8;
    case kv_dtype_t::i4: return 4;
    case kv_dtype_t::f16:
    case kv_dtype_t::bf16: return 16;
    }
    return 16;
}

// bytes of one row (head_dim elements of one token/one kv head)
inline size_t kv_dtype_row_bytes(kv_dtype_t t, int head_dim) {
    return (size_t)head_dim * (size_t)kv_dtype_bits(t) / 8;
}

// i8 and i4 keep a separate per-32 fp16 scale plane; the other types do not
inline bool kv_dtype_has_scales(kv_dtype_t t) {
    return t == kv_dtype_t::i8 || t == kv_dtype_t::i4;
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
    if (t == kv_dtype_t::i4) {
        const uint8_t byte = ((const uint8_t *)base)[i >> 1];
        const int v = (i & 1) ? (byte >> 4) : (byte & 0xF);
        return (float)((v ^ 8) - 8);
    }
    return (float)((const sycl::half *)base)[i];
}

} // namespace si
