#include "kv_type.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "common/env.h"

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

bool kv_dtype_mix_ok(kv_dtype_t k, kv_dtype_t v) {
    if (k == v) {
        return true;
    }
    // the only mixed geometry the kernels implement: the two scale-carrying
    // types, whose rows differ (1 byte/elem vs 1/2) but share the per-32 plane
    const bool ki = k == kv_dtype_t::i4 || k == kv_dtype_t::i8;
    const bool vi = v == kv_dtype_t::i4 || v == kv_dtype_t::i8;
    return ki && vi;
}

bool kv_dtype_parse_pair(const char * spec, kv_dtype_t & k, kv_dtype_t & v) {
    if (!spec || !*spec) {
        return false;
    }
    const char * colon = strchr(spec, ':');
    if (!colon) {
        kv_dtype_t t;
        if (!parse_name(spec, t)) {
            return false;
        }
        k = v = t;
        return true;
    }
    std::string ks(spec, colon - spec);
    kv_dtype_t kt, vt;
    if (!parse_name(ks.c_str(), kt) || !parse_name(colon + 1, vt)) {
        return false;
    }
    if (!kv_dtype_mix_ok(kt, vt)) {
        return false;
    }
    k = kt;
    v = vt;
    return true;
}

namespace {
// -1 = unresolved: read the environment on first use.  A CLI override stores
// the resolved value directly before the engine (and its worker threads) exist.
std::atomic<int> g_kv_k{-1};
std::atomic<int> g_kv_v{-1};

void from_env(kv_dtype_t & k, kv_dtype_t & v) {
    const char * e = si::env::str("PF_KV_TYPE");
    const char * f = si::env::str("PF_KV_F32");
    const char * b = si::env::str("PF_KV_BF16");
    if ((f && atoi(f) != 0) || (b && atoi(b) == 0)) {
        k = v = kv_dtype_t::f32;
        return;
    }
    if (!e || !*e) {
        k = v = kv_dtype_t::i8;
        return;
    }
    if (!kv_dtype_parse_pair(e, k, v)) {
        fprintf(stderr, "[kv] unknown PF_KV_TYPE='%s' (i4|i8|bf16|f16|f32, or K:V), using i8\n", e);
        k = v = kv_dtype_t::i8;
    }
}
} // namespace

kv_dtype_t kv_k_dtype() {
    int v = g_kv_k.load(std::memory_order_acquire);
    if (v < 0) {
        kv_dtype_t k, vv;
        from_env(k, vv);
        g_kv_k.store((int)k, std::memory_order_release);
        g_kv_v.store((int)vv, std::memory_order_release);
        v = (int)k;
    }
    return (kv_dtype_t)v;
}

kv_dtype_t kv_v_dtype() {
    int v = g_kv_v.load(std::memory_order_acquire);
    if (v < 0) {
        kv_dtype_t k, vv;
        from_env(k, vv);
        g_kv_k.store((int)k, std::memory_order_release);
        g_kv_v.store((int)vv, std::memory_order_release);
        v = (int)vv;
    }
    return (kv_dtype_t)v;
}

kv_dtype_t kv_dtype() {
    return kv_k_dtype();
}

void kv_dtype_set(kv_dtype_t t) {
    kv_dtype_set_kv(t, t);
}

void kv_dtype_set_kv(kv_dtype_t k, kv_dtype_t v) {
    g_kv_k.store((int)k, std::memory_order_release);
    g_kv_v.store((int)v, std::memory_order_release);
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
