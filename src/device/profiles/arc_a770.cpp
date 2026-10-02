// Device profile: Intel Arc A770 (DG2) -- the reference card for this engine.
//
// Two A770s, 16 GB each, in the 2-device split the reports measure
// (--layer-map 0-31:gpu.0,32-63:gpu.1).  Every number here was measured on one of
// them unless the field says otherwise; the citations are to the reports that
// carry the full tables.
//
// This file is the card's COMPLETE implementation: its values, its key, and the
// matcher that recognises it.  Nothing outside here knows what an A770 is.
#include "device/device_profile.h"

#include <string>

namespace si::dev {

static const profile arc_a770 = {
    .key = "arc_a770",
    .name = "Intel Arc A770 (DG2)",
    .provenance =
        "measured on 1-2x A770 16 GB (DG2, 512 EUs/card). Decode split curve and the "
        "8-warps/EU cliff: reports/d64k_pp_tg_evaluation.md 3.1. Prefill attention stage "
        "breakdown and the gather reduction: same report 3.2. GEMV/GEMM/GDN tunings: "
        "reports/tg128_20tps_evaluation.md and reports/mtp_ceiling.md.",

    // reported by sycl, used as the fallback when the query is unavailable
    .hw = {
        .compute_units = 512,
        .max_work_group_size = 1024, // rmsnorm uses the whole group
        .local_mem_size = 65536,     // 64 KB SLM per work-group
    },

    // MEASURED: 7.5 of the 8 warps/EU a Xe-LP EU holds.  The cliff itself is
    // architectural (4 sub-schedulers x 2 warp slots).
    .occ = {.warps_per_eu_x2 = 15},

    // A deliberate reserve under the 64 KB hardware limit: the w4/k5/codebook
    // GEMVs' fits(rb) checks use it, and nat_gemm's M*KT*APAD tile is sized on it.
    .slm = {.budget_bytes = 48 * 1024},

    // the decode int8 GEMV splits K when the output is too small to fill the
    // machine; the row GEMM wants the same thing at a much wider threshold.
    .split = {
        .gemv_rows = 2048,
        .gemm_rows = 16384,
        .max = 8,
    },

    // MEASURED: 1024 threads over n_embd=5120 cut the serial load iterations from
    // 20 to 5 and the SLM reduction from 8 barriers to 2; 256 threads measured
    // 32 us/call = 4.1 ms of an 85 ms multi-device decode.
    // GDN cols 2: the float4 state slice costs 4x the registers, so it only wins
    // at one real row per pass (the n_real x {float4,scalar} ms table in gdn.cpp).
    .shape = {
        .gemv_rows_per_wg = 8,
        .rmsnorm_wg = 1024,
        .gdn_cols = 2,
        .gdn_warps_per_wg = 8,
        .gdn_vec_max_rows = 2,
    },

    // MEASURED: a single work-group over the whole key block cost 18.1 ms of the
    // 37.8 ms one attention layer takes at 64k depth; 64 gets it to 9.1 and
    // 256/512 are no better.
    // xmx on: oneDNN's int8 matmul beats the dp4a kernels by ~11x at M=512 here.
    .attn = {
        .xmx_gather_red = 64,
        .xmx_gather_red_max = 512,
        .vec = 1,
        .xmx = 1,
        .xmx_min_keys = 2048, // below this the classic kernel wins
        .split_keys = 512,
        .dec_group = 0,       // measured 5x slower than classic at 64k (report 4.1)
    },

    // the native 4/5-bit stores save real decode bandwidth here (0.625/0.75 vs
    // 1.0/1.125 B/weight), which is why all of them are on.
    .wt = {
        .w4 = 1,
        .k5 = 1,
        .cb4 = 1,
        .dp4a = 1,
        .gemm_dnnl = 1,
        .dpas_in_gemm = 0, // measured 37x slower than dp4a in a real GEMM
    },
};

static bool arc_a770_matches(const std::string & device_name) {
    // "Intel Arc A770 (DG2)" reports as-is on Level Zero.  Matching the family
    // name as well means a future A750/A580 lands here rather than falling
    // through -- which is deliberate, because it is the same DG2 part, and the
    // alternative is silently inheriting someone else's tuning.
    return device_name.find("Arc") != std::string::npos ||
           device_name.find("A770") != std::string::npos;
}

// extern: the registry in device_registry.cpp names this symbol
extern const device_entry arc_a770_device = {
    .key = "arc_a770",
    .prof = &arc_a770,
    .matches = arc_a770_matches,
};

} // namespace si::dev