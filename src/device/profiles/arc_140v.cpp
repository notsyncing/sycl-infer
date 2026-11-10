// Device profile: Intel Arc 140V Graphics (Xe2, Lunar Lake, integrated).
//
// The driver reports this part as "Intel(R) Arc(TM) Graphics" with no SKU, so
// the matcher keys on that exact signature rather than on a model name.  It is
// a 64-EU Xe2 part with XMX (DPAS), 128 KB of SLM per work-group and a 1024
// thread max work-group, sharing LPDDR5X with the CPU: dev/bench_bw2 measures
// ~107 GB/s of peak read bandwidth (against 376 GB/s on an A770), which is the
// single number every decode decision here has to be read against.
//
// PROVENANCE.  The hardware facts are queried by SYCL and repeated here only as
// the fallback.  The tuning groups are mostly INHERITED from the A770 and are
// marked as such: this part is bandwidth-starved where the A770 is not, so the
// reads that matter (split thresholds, w4 rows) should be re-swept.  The one
// value that IS measured here is moe.row_gemv: the grouped expert kernel's
// eight-lane-per-group mapping issues scattered 32-byte requests and measures
// ~22 GB/s of unique weight on this part, while the warp-per-row form (one
// coalesced 128-byte load per superblock) is what a 107 GB/s part needs.
#include "device/device_profile.h"

#include <string>

namespace si::dev {

static const profile arc_140v = {
    .key = "arc_140v",
    .name = "Intel Arc 140V (Xe2, Lunar Lake)",
    .provenance =
        "Hardware facts MEASURED (SYCL): 64 EUs, 128 KB SLM, 1024-thread max "
        "work-group, XMX/DPAS advertised.  Peak read bandwidth measured at "
        "107 GB/s (dev/bench_bw2, 1 GB USM buffer).  moe.row_gemv is MEASURED "
        "here: the single-token decode's expert GEMV as one warp per output row "
        "(a 128-byte coalesced weight load per superblock) instead of the "
        "grouped kernel's scattered 32-byte group loads.  Every other tuning "
        "group is INHERITED from arc_a770 and NOT re-swept on this part; in "
        "particular split.*, shape.w4_rows_per_wg / q8_rows / f32_rows and "
        "attn.* are 512-EU values and should be treated as a starting point.  "
        "The banked prefill expert GEMM tiles ARE swept here: gu_tile 32 "
        "(5.46 s against the cost model's 5.65 s on a 3.3k-token 35B prefill) "
        "and dn_tile 96 (5.36 s against 128's 5.46 s; 192/256 lose).",

    .hw = {
        .compute_units = 64,
        .max_work_group_size = 1024,
        .local_mem_size = 131072,
        // Xe2: XMX is present and the driver is expected to advertise
        // cl_intel_subgroup_matrix_multiply_accumulate.  QUERIED at run time;
        // this is only the fallback.
        .has_dpas = 1,

        // Xe2 (Lunar Lake) reports sub-group sizes {16, 32} only - measured
        // with info::device::sub_group_sizes on this part.  A kernel declared
        // reqd_sub_group_size(8) does not launch here, so nat_gemm's batched
        // GEMM runs its SG=16 instantiation (NSG=8, RB=16); the SG=8 form is
        // still compiled for the Xe-LP/Xe-HPG cards.
        .sub_group8 = 0,
    },

    // INHERITED: the 8-warp/EU Xe2 lattice is not measured here.  With 64 EUs
    // the decode split count derives as warps_per_eu_x2 * 64 / (2 * n_head).
    .occ = {.warps_per_eu_x2 = 15},

    // INHERITED budget of the measured 128 KB SLM (the kernels were sized
    // against 48 KB on both Arc parts (Xe-HPG A770 / Xe-LP Iris) and that reserve still fits).
    .slm = {.budget_bytes = 48 * 1024},

    // INHERITED, and the group most likely to be wrong: these thresholds size a
    // K-split to fill the machine, so a 64-EU part fills at a quarter of the row
    // count.  Left at the A770 values pending a sweep - an explicit "not yet
    // measured", not a claim that they are right.
    .split = {
        .gemv_rows = 2048,
        .gemm_rows = 16384,
        .max = 8,
    },

    // rmsnorm_wg 1024 is safe here because the part accepts it; the clamp in
    // si::dev::wg_clamped_for_queue still guards a smaller card.  The rest is
    // INHERITED: the GDN scalar/float4 crossover is a register-file property of
    // the same 8-lane lattice, but its ms table was measured on the A770.
    .shape = {
        .gemv_rows_per_wg = 8,
        .rmsnorm_wg = 1024,
        .gdn_cols = 8, // MEASURED: 290 -> 212 ms per 804-token state scan; 16 is a cliff
        .gdn_warps_per_wg = 8,
        .gdn_vec_max_rows = 2,
        .w4_rows_per_wg = 16, // INHERITED (SLM budget is the same)
        .q8_rows = 512,       // INHERITED: scales with EU count
        .f32_rows = 128,      // INHERITED
    },

    // INHERITED.  attn.xmx is oneDNN's int8 prefill attention and needs
    // head_dim 256, so it does not apply to the 35B A3B (head_dim 128) either
    // way.  dec_group stays off (both Arc parts measured it off, for opposite
    // reasons); it has not been re-measured on Xe2.
    .attn = {
        .xmx_gather_red = 64,
        .xmx_gather_red_max = 512,
        .vec = 1,
        .xmx = 1,
        // MEASURED on this card: the 804-key chunk is 153.7 ms on the classic
        // kernel and 48 ms on xmx; 2048 was the A770 crossover, inherited.
        .xmx_min_keys = 512,
        .split_keys = 512,
        .dec_group = 0,
        .vit_tile = 70,       // INHERITED; 1024-thread tiles launch on this part
        .xmx_gather_q = 8,    // INHERITED (unmeasured here)
        .dflash_slices = 8,   // INHERITED (unmeasured here)
    },

    // INHERITED: the native stores save bytes, which matters MORE on a
    // bandwidth-starved part, so all four are on.  dpas_in_gemm stays off: the
    // DPAS GEMM was measured 37x slower on the A770 and has not been rebuilt for
    // Xe2; this profile does not claim it would lose here.
    .wt = {
        .w4 = 1,
        .k5 = 1,
        .cb4 = 1,
        .dp4a = 1,
        .gemm_dnnl = 1,
        .dpas_in_gemm = 0,
        // MEASURED on this part (35B A3B, PF_W4_ALL, ctx 8192, i8 KV, 128-token
        // greedy requests, one process per cell): the batched native GEMM is
        // SLOWER than oneDNN's fallback here, so it stays off.  With the SG=16
        // instantiation enabled (PF_NAT=1) the aggregate decode was 26.2 /
        // 33.4 tok/s at N=2 / N=4 against 27.4 / 36.1 for the oneDNN int8
        // matmul (-3.5% / -7%), N=1 unchanged (29.0, and its stream is
        // byte-identical).  Reason: only two tensors reach it - the LM head
        // (K=2048, N=248320) and one N=8192 - and on Xe2 oneDNN runs its int8
        // matmul with DPAS, which is the shape XMX is best at; on DG2 the same
        // comparison went the other way because that library has no XMX kernel
        // at all.  PF_NAT=1 forces it on for a re-measure on another workload.
        .nat_gemm = 0,
    },

    // The grouped tile cases and grids are INHERITED.  row_gemv is the Xe2
    // value: a single-token decode is a set of independent per-(expert,row)
    // GEMVs, and one warp per row reads each weight row as 128-byte coalesced
    // loads.  It applies only to Q4_K/Q5_K and only when every active expert has
    // exactly one pair; other shapes fall back to the grouped path.
    .moe = {
        .cfg_gate = 43,
        .cfg_down = 43,
        .cfg_dec = 0,
        .flat = 1,
        .flat_dec = 0,
        .dp4a = 1,
        .dp4a_dec = 1,
        .upload_ring = 32,
        .pf_wmin = 24,
        .row_gemv = 1,
        .tiled = 1,
        .gu_tile = 32,
        .dn_tile = 96,
        .w4_qfuse = 1,
    },
};

static bool arc_140v_matches(const std::string & device_name) {
    // Lunar Lake reports NO SKU: "Intel(R) Arc(TM) Graphics".  Match that
    // signature (Arc immediately followed by the trademark and "Graphics"), which
    // is never true of "Intel(R) Arc(TM) A770 Graphics" or any other SKU-bearing
    // DG2/BMG name.  A future part that reports the same generic name gets this
    // profile, which is the intended behaviour until it is measured itself.
    return device_name.find("Arc(TM) Graphics") != std::string::npos
           || device_name.find("Arc(R) Graphics") != std::string::npos
           || device_name.find("Arc Graphics") != std::string::npos;
}

// extern: the registry in device_registry.cpp names this symbol
extern const device_entry arc_140v_device = {
    .key = "arc_140v",
    .prof = &arc_140v,
    .matches = arc_140v_matches,
};

} // namespace si::dev
