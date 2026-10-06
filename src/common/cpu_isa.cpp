#include "cpu_isa.h"
#include "common/env.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !defined(__SYCL_DEVICE_ONLY__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#define SI_CPU_X86 1
#endif

namespace si {

namespace {

#ifdef SI_CPU_X86
unsigned cpuid_leaf7_ecx_bit(int bit) {
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return 0;
    }
    return (ecx >> bit) & 1u;
}
#endif

cpu_features detect() {
    cpu_features f;
#if !defined(__SYCL_DEVICE_ONLY__)
#ifdef SI_CPU_X86
    __builtin_cpu_init();
    f.avx2 = __builtin_cpu_supports("avx2") != 0;
    f.fma = __builtin_cpu_supports("fma") != 0;
    f.f16c = __builtin_cpu_supports("f16c") != 0;
    f.avx512f = __builtin_cpu_supports("avx512f") != 0;
    f.avx512bw = __builtin_cpu_supports("avx512bw") != 0;
    // AVX512-VNNI is CPUID leaf 7 subleaf 0, ECX bit 11
    f.avx512vnni = __builtin_cpu_supports("avx512vnni") != 0 && cpuid_leaf7_ecx_bit(11) != 0;
    // AVX-VNNI (256-bit) is CPUID leaf 7 subleaf 0, ECX bit 4; it needs the
    // AVX2/OSXSAVE state that the avx2 builtin already validated.
    f.avxvnni = f.avx2 && cpuid_leaf7_ecx_bit(4) != 0;
#endif
#endif
    return f;
}

cpu_isa forced_isa(const cpu_features & f) {
    const char * e = si::env::str("PF_CPU_ISA");
    if (!e || !*e || std::strcmp(e, "auto") == 0) {
        return cpu_isa::scalar; // sentinel: not forced
    }
    if (std::strcmp(e, "scalar") == 0 || std::strcmp(e, "0") == 0) {
        return cpu_isa::scalar;
    }
    if (std::strcmp(e, "avx2") == 0) {
        return f.avx2 ? cpu_isa::avx2 : cpu_isa::scalar;
    }
    if (std::strcmp(e, "avx512") == 0) {
        return (f.avx512f && f.avx512bw) ? cpu_isa::avx512 : (f.avx2 ? cpu_isa::avx2 : cpu_isa::scalar);
    }
    if (std::strcmp(e, "avxvnni") == 0 || std::strcmp(e, "vnni") == 0) {
        return f.avx2 ? cpu_isa::avx2 : cpu_isa::scalar; // VNNI is a sub-variant of the AVX2 tier
    }
    return cpu_isa::scalar;
}

} // namespace

const cpu_features & cpu_features_of() {
    static const cpu_features f = detect();
    return f;
}

cpu_isa cpu_best_isa() {
    const cpu_features & f = cpu_features_of();
    const cpu_isa forced = forced_isa(f);
    // Any PF_CPU_ISA setting wins over the detected level; an unknown value
    // means scalar (conservative, used by the tests to force the fallback).
    if (const char * e = si::env::str("PF_CPU_ISA")) {
        if (*e && std::strcmp(e, "auto") != 0) {
            return forced;
        }
    }
    if (f.avx512f && f.avx512bw) {
        return cpu_isa::avx512;
    }
    if (f.avx2 && f.fma) {
        return cpu_isa::avx2;
    }
    return cpu_isa::scalar;
}

bool cpu_has_avxvnni() {
    const cpu_features & f = cpu_features_of();
    if (const char * e = si::env::str("PF_CPU_ISA")) {
        if (*e && std::strcmp(e, "scalar") == 0) {
            return false;
        }
        if (std::strcmp(e, "avx2") == 0) {
            return false;
        }
    }
    return f.avxvnni || f.avx512vnni;
}

const char * cpu_isa_name(cpu_isa isa) {
    switch (isa) {
    case cpu_isa::avx512:
        return "avx512";
    case cpu_isa::avx2:
        return "avx2";
    default:
        return "scalar";
    }
}

const char * cpu_isa_spec() {
    static char buf[64];
    const cpu_isa isa = cpu_best_isa();
    if (isa == cpu_isa::avx2 && cpu_has_avxvnni()) {
        std::snprintf(buf, sizeof(buf), "avx2+avxvnni");
    } else {
        std::snprintf(buf, sizeof(buf), "%s", cpu_isa_name(isa));
    }
    return buf;
}

} // namespace si
