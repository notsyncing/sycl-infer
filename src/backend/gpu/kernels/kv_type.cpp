#include "kv_type.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

// KV cache storage type selection (env, read once).
//   PF_KV_TYPE = i8 | bf16 | f16 | f32   (default i8; PF_KV_F32=1 forces f32)
//   PF_KV_BF16=0 is an alias for PF_KV_TYPE=f32
// i8 halves the KV bytes again (6 KB/token vs 12 KB/token with bf16) and is
// the default because long-context decode is DRAM-byte bound on this GPU;
// PF_KV_TYPE=bf16 restores the previous default, PF_KV_TYPE=f32 the historical
// fp32 pools.
kv_dtype_t kv_dtype() {
    static const kv_dtype_t t = [] {
        const char * e = getenv("PF_KV_TYPE");
        const char * f = getenv("PF_KV_F32");
        const char * b = getenv("PF_KV_BF16");
        if ((f && atoi(f) != 0) || (b && atoi(b) == 0)) {
            return kv_dtype_t::f32;
        }
        if (!e || !*e) {
            return kv_dtype_t::i8;
        }
        if (!strcmp(e, "f32") || !strcmp(e, "fp32") || !strcmp(e, "0")) {
            return kv_dtype_t::f32;
        }
        if (!strcmp(e, "f16") || !strcmp(e, "fp16")) {
            return kv_dtype_t::f16;
        }
        if (!strcmp(e, "bf16")) {
            return kv_dtype_t::bf16;
        }
        if (!strcmp(e, "i8") || !strcmp(e, "int8") || !strcmp(e, "q8")) {
            return kv_dtype_t::i8;
        }
        fprintf(stderr, "[kv] unknown PF_KV_TYPE='%s' (i8|bf16|f16|f32), using i8\n", e);
        return kv_dtype_t::i8;
    }();
    return t;
}

const char * kv_dtype_name(kv_dtype_t t) {
    switch (t) {
    case kv_dtype_t::f32: return "f32";
    case kv_dtype_t::f16: return "f16";
    case kv_dtype_t::i8: return "i8";
    default: return "bf16";
    }
}

} // namespace si
