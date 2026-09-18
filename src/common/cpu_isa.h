#pragma once
// ---------------------------------------------------------------------------
// Host CPU instruction-set detection and dispatch.
//
// The CPU compute backend (src/backend/cpu) ships the same operator under several ISA
// variants; this header is the single place that decides which one runs.  The
// variants are plain host functions with GCC/Clang `target` attributes, so a
// single translation unit can carry AVX2, AVX-VNNI and AVX-512 code and pick
// at run time.  `PF_CPU_ISA` forces a variant (scalar|avx2|avxvnni|avx512) so
// the tests can compare them against each other.
//
// The device pass of `-fsycl` also parses these TUs; every intrinsic use is
// guarded by `#if !defined(__SYCL_DEVICE_ONLY__)` and the CPU kernels are only
// ever called through the CPU backend, which never runs on a device.
// ---------------------------------------------------------------------------
namespace si {

// Highest vector width the host can run.
enum class cpu_isa : int {
    scalar = 0, // portable C, no vector builtins
    avx2 = 1,   // AVX2 + FMA (+ F16C when present)
    avx512 = 2, // AVX-512 F/BW (+ VNNI when present)
};

struct cpu_features {
    bool avx2 = false;
    bool fma = false;
    bool f16c = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512vnni = false;
    bool avxvnni = false; // AVX-VNNI (256-bit, no AVX-512 required)
};

const cpu_features & cpu_features_of();
cpu_isa cpu_best_isa();
// AVX-VNNI usable for the 256-bit int8 dot product (either AVX512-VNNI or
// AVX-VNNI).  `cpu_best_isa()` stays AVX2 when only AVX-VNNI is present.
bool cpu_has_avxvnni();
const char * cpu_isa_name(cpu_isa isa);
// The runtime-selected variant name ("scalar"/"avx2"/"avxvnni"/"avx512");
// "avxvnni" is reported when AVX2 + AVX-VNNI is the best combination.
const char * cpu_isa_spec();

} // namespace si
