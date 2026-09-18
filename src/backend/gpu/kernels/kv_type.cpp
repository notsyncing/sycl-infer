#include "kv_type.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

// KV cache storage type selection.
//   --kv-type i4|i8|bf16|f16|f32 overrides everything
//   PF_KV_TYPE = i4|int4|q4 | i8|int8|q8 | bf16 | f16|fp16 | f32|fp32|0
//     (default i8; PF_KV_F32=1 forces f32, PF_KV_BF16=0 is an alias for f32)
// i8 halves the KV bytes vs bf16 and is the default because long-context decode
// is DRAM-byte bound on this GPU; i4 halves them again at a larger quantization
// error; PF_KV_TYPE=bf16 restores the previous default, PF_KV_TYPE=f32 the
// historical fp32 pools.
static bool parse_name(const char * e, kv_dtype_t & out) {
    if (!e || !*e) {
        return false;
    }
    if (!strcmp(e, "f32") || !strcmp(e, "fp32") || !strcmp(e, "0")) {
        out = kv_dtype_t::f32;
    } else if (!strcmp(e, "f16") || !strcmp(e, "fp16")) {
        out = kv_dtype_t::f16;
    } else if (!strcmp(e, "bf16")) {
        out = kv_dtype_t::bf16;
    } else if (!strcmp(e, "i8") || !strcmp(e, "int8") || !strcmp(e, "q8")) {
        out = kv_dtype_t::i8;
    } else if (!strcmp(e, "i4") || !strcmp(e, "int4") || !strcmp(e, "q4")) {
        out = kv_dtype_t::i4;
    } else {
        return false;
    }
    return true;
}

bool kv_dtype_parse(const char * spec, kv_dtype_t & out) {
    return parse_name(spec, out);
}

namespace {
// -1 = unresolved: read the environment on first use.  A CLI override stores
// the resolved value directly before the engine (and its worker threads) exist.
std::atomic<int> g_kv_dtype{-1};

kv_dtype_t from_env() {
    const char * e = getenv("PF_KV_TYPE");
    const char * f = getenv("PF_KV_F32");
    const char * b = getenv("PF_KV_BF16");
    if ((f && atoi(f) != 0) || (b && atoi(b) == 0)) {
        return kv_dtype_t::f32;
    }
    if (!e || !*e) {
        return kv_dtype_t::i8;
    }
    kv_dtype_t t;
    if (!parse_name(e, t)) {
        fprintf(stderr, "[kv] unknown PF_KV_TYPE='%s' (i4|i8|bf16|f16|f32), using i8\n", e);
        return kv_dtype_t::i8;
    }
    return t;
}
} // namespace

kv_dtype_t kv_dtype() {
    int v = g_kv_dtype.load(std::memory_order_acquire);
    if (v < 0) {
        v = (int)from_env();
        g_kv_dtype.store(v, std::memory_order_release);
    }
    return (kv_dtype_t)v;
}

void kv_dtype_set(kv_dtype_t t) {
    g_kv_dtype.store((int)t, std::memory_order_release);
}

const char * kv_dtype_name(kv_dtype_t t) {
    switch (t) {
    case kv_dtype_t::f32: return "f32";
    case kv_dtype_t::f16: return "f16";
    case kv_dtype_t::i8: return "i8";
    case kv_dtype_t::i4: return "i4";
    default: return "bf16";
    }
}

} // namespace si
