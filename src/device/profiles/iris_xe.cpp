// Device profile: Intel Iris Xe Graphics (Xe-LP, integrated).
//
// This is the part the engine was originally written on (see src/common/w8.h:
// "On this GPU (Iris Xe-LP) that path is limited by instructions and load
// sectors, while hardware DP4A gives 4x the raw throughput"), and it is the
// *other* half of the SIn int8 design.  The A770 is the newer profile and carries
// far more measurement behind it, so most fields here are INHERITED and say so.
//
// Being explicit about which numbers are unmeasured matters: the integrated part
// shares system memory with the CPU, so every bandwidth-sensitive constant here is
// suspect until it is re-measured on an idle machine.
//
// This file is the card's COMPLETE implementation: its values, its key, and the
// matcher that recognises it.  Nothing outside here knows what an Iris Xe is.
#include "device/device_profile.h"

#include <string>

namespace si::dev {

static const profile iris_xe = {
    .key = "iris_xe",
    .name = "Intel Iris Xe (Xe-LP)",
    .provenance =
        "Hardware facts are reported by SYCL on this machine (96 EUs, 64 KB SLM, "
        "512-thread max work-group).  THE REST IS INHERITED from the A770, not "
        "re-measured: both parts are Xe-LP so the 8-warp/EU sub-group lattice is the "
        "same, but the optimum was not re-swept.  The reason: this GPU shares system "
        "memory with the host, and dev/bench_attn27 --dec-only on an otherwise loaded "
        "machine returned 2.1-4.4 ms for one configuration across three repeats, i.e. "
        "the noise exceeded the effect.  Re-run the sweeps on an idle box before "
        "trusting these for bandwidth-sensitive work.",

    // measured (sycl): 96 EUs is 1/5.3 of an A770, and the max work-group is HALF
    // it -- 512, not 1024, which is load-bearing (see shape.rmsnorm_wg).
    .hw = {
        .compute_units = 96,
        .max_work_group_size = 512,
        .local_mem_size = 65536,
    },

    // INHERITED: same Xe-LP lattice, so 7.5 is the expected shape.  With 96 EUs
    // this gives dec_splits = 7.5*96/n_head against the A770's 480, which is the
    // point of deriving it from the EU count.
    .occ = {.warps_per_eu_x2 = 15},

    // INHERITED: the SLM size matches the A770 exactly, so the reserve transfers.
    // This one is a hardware fact rather than a tuning.
    .slm = {.budget_bytes = 48 * 1024},

    // INHERITED, and the most suspect group here: these size the machine to fill,
    // so they should scale with 96 EUs rather than 512.  Left at the A770 values
    // pending a sweep -- a deliberate "not yet measured", not a claim that they are
    // right.
    .split = {
        .gemv_rows = 2048,
        .gemm_rows = 16384,
        .max = 8,
    },

    // rmsnorm_wg 512 is MEASURED and load-bearing: it is the widest group this
    // part accepts, so the A770's 1024-thread kernel does not launch here at all.
    // The rest is INHERITED -- the GDN float4/scalar crossover is a register-file
    // measurement and the register file is the same Xe-LP one, so it plausibly
    // transfers, but the surrounding ms table was measured on the A770.
    .shape = {
        .gemv_rows_per_wg = 8,
        .rmsnorm_wg = 512,
        .gdn_cols = 2,
        .gdn_warps_per_wg = 8,
        .gdn_vec_max_rows = 2,
    },

    // gather_red INHERITED: a one-work-group version would be worse still at 96
    // EUs, so keeping 64 is the conservative direction.
    // xmx is INHERITED but UNTESTED: this is the part where DP4A was worth 4x over
    // the scalar path, so whether oneDNN's int8 matmul pays off is unknown -- do
    // not read "on" as "measured good".
    // dec_group is MEASURED here, for the OPPOSITE reason to the A770: the grouped
    // kernel has n_head_kv rather than n_head workgroups, and the classic kernel
    // is measurably faster on this part because it has 4x the warps.  Same value,
    // opposite evidence -- the case the split exists for.
    .attn = {
        .xmx_gather_red = 64,
        .xmx_gather_red_max = 512,
        .vec = 1,
        .xmx = 1,
        .xmx_min_keys = 2048,
        .split_keys = 512,
        .dec_group = 0,
    },

    .wt = {
        .w4 = 1,
        .k5 = 1,
        .cb4 = 1,
        .dp4a = 1,
        .gemm_dnnl = 1,
        .dpas_in_gemm = 0, // same Xe-LP DPAS, same 37x-slower measurement
    },
};

static bool iris_xe_matches(const std::string & device_name) {
    // "Iris" only, deliberately NOT "Xe": "Xe" is a substring of "Iris Xe" and
    // also catches future Xe parts whose tunings differ.  A new Xe-LP/Xe-HP card
    // should get its own file and row rather than inherit these.
    return device_name.find("Iris") != std::string::npos;
}

// extern: the registry in device_registry.cpp names this symbol
extern const device_entry iris_xe_device = {
    .key = "iris_xe",
    .prof = &iris_xe,
    .matches = iris_xe_matches,
};

} // namespace si::dev