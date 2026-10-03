// Device registry: the list of cards, and the selection that reads it.
//
// This file knows nothing about any particular GPU.  Each card owns a
// src/device/profiles/<card>.cpp that defines one `device_entry` -- its key, its
// values, and its own matcher -- and all this file does is hold those entries in
// match order and pick the first whose matcher accepts the name SYCL reports.
// Adding a card is one new file plus one row below.
#include "device/device_profile.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace si::dev {

// the registered cards -- one extern per src/device/profiles/<card>.cpp, and one
// row here.  Order is match order, first match wins.
extern const device_entry arc_a770_device; // src/device/profiles/arc_a770.cpp
extern const device_entry iris_xe_device;  // src/device/profiles/iris_xe.cpp

const device_entry *const kDevices[] = {
    &arc_a770_device,
    &iris_xe_device,
};
const int kNumDevices = (int)(sizeof(kDevices) / sizeof(kDevices[0]));

// A card nobody claims.  Built from the arc_a770 entry at compile time so it
// cannot drift from the row it copies -- which is the whole point of copying it.
const profile & unknown_profile() {
    static const profile unknown = [] {
        profile u = *kDevices[0]->prof;
        u.key = "unknown";
        u.name = "unrecognised GPU";
        u.provenance = "NO MEASUREMENTS.  Every tuned value is a copy of "
                       "arc_a770's, which is a guess: add a profile for this card "
                       "under src/device/profiles/ before trusting any of them.";
        u.hw.compute_units = 0;
        u.hw.max_work_group_size = 0;
        u.hw.local_mem_size = 0;
        return u;
    }();
    return unknown;
}

const profile & for_name(const std::string & device_name) {
    for (int i = 0; i < kNumDevices; i++) {
        if (kDevices[i]->matches(device_name)) {
            return *kDevices[i]->prof;
        }
    }
    return unknown_profile();
}

void report(const profile & p, const char * how) {
    if (getenv("PF_DEVICE_INFO")) {
        fprintf(stderr,
                "[dev] profile=%s  (%s)\n"
                "      provenance: %s\n"
                "      hw: eus=%d max_wg=%d slm=%d B\n"
                "      occupancy: warps_per_eu=%.1f (x2=%d)\n"
                "      slm_budget=%d B\n"
                "      split rows: gemv=%d gemm=%d max=%d\n"
                "      shapes: gemv_rows_per_wg=%d rmsnorm_wg=%d gdn(cols=%d warps=%d vec_max=%d)\n"
                "      attn: gather_red=%d (max %d) vec=%d xmx=%d xmx_min_keys=%d "
                "split_keys=%d dec_group=%d\n"
                "      weights: w4=%d k5=%d cb4=%d dp4a=%d gemm_dnnl=%d dpas=%d\n",
                p.key, how, p.provenance, p.hw.compute_units, p.hw.max_work_group_size,
                p.hw.local_mem_size, p.occ.warps_per_eu_x2 / 2.0, p.occ.warps_per_eu_x2,
                p.slm.budget_bytes, p.split.gemv_rows, p.split.gemm_rows, p.split.max,
                p.shape.gemv_rows_per_wg, p.shape.rmsnorm_wg, p.shape.gdn_cols,
                p.shape.gdn_warps_per_wg, p.shape.gdn_vec_max_rows, p.attn.xmx_gather_red,
                p.attn.xmx_gather_red_max, p.attn.vec, p.attn.xmx, p.attn.xmx_min_keys,
                p.attn.split_keys, p.attn.dec_group, p.wt.w4, p.wt.k5, p.wt.cb4, p.wt.dp4a,
                p.wt.gemm_dnnl, p.wt.dpas_in_gemm);
    }
    if (std::strcmp(p.key, "unknown") == 0) {
        fprintf(stderr,
                "[dev] WARNING: no device profile matches this GPU; tuned values are guesses "
                "copied from arc_a770.  Add a file under src/device/profiles/ and a row in "
                "kDevices -- see AGENTS.md.\n");
    }
}

const profile & active() {
    static const profile * resolved = [] () -> const profile * {
        // SI_FORCE_DEVICE_PROFILE is baked in by CMake
        // (-DSYCL_INFER_AOT_PROFILE=...) so an AOT binary built for one card
        // still selects that card's profile when the SYCL device name would
        // mislead.  PF_DEVICE_PROFILE is the runtime equivalent and wins, so a
        // baked binary can still be re-pointed for an A/B.
        const char * baked = SI_FORCE_DEVICE_PROFILE;
        if (const char * forced = getenv("PF_DEVICE_PROFILE")) {
            baked = forced;
        }
        if (baked && *baked) {
            for (int i = 0; i < kNumDevices; i++) {
                if (std::strcmp(baked, kDevices[i]->key) == 0) {
                    report(*kDevices[i]->prof, "forced by PF_DEVICE_PROFILE / build");
                    return kDevices[i]->prof;
                }
            }
            if (std::strcmp(baked, "unknown") == 0) {
                const profile & p = unknown_profile();
                report(p, "forced by PF_DEVICE_PROFILE / build");
                return &p;
            }
            fprintf(stderr, "[dev] device profile '%s' is not a registered card; "
                            "falling back to auto-detect\n",
                    baked);
        }
        // Ask SYCL for the device name once, if a GPU is present at all.  This
        // must not throw: it runs during init paths where the engine has not
        // necessarily selected a device yet.  The string is held by value --
        // get_info<name>() returns a temporary, and taking .c_str() on it leaves
        // `name` dangling, which is exactly the kind of bug that makes a card we
        // do have a profile for resolve as "unknown".
        std::string name;
        try {
            auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
            if (!devs.empty()) {
                name = devs[0].get_info<sycl::info::device::name>();
            }
        } catch (...) {
            // no GPU visible (build-time tooling, a CPU-only run) -> unknown
        }
        const profile & p = for_name(name);
        report(p, "resolved from device name");
        return &p;
    }();
    return *resolved;
}

int wg_clamped(int want) {
    // The device limit is a HARDWARE fact and cannot change over a run, but the
    // query is not free: `device::get_devices()` enumerates every GPU through
    // the driver and builds SYCL device objects, and `rmsnorm_launch` calls this
    // on EVERY launch.  Measured 3.2 ms per call, i.e. 210 ms of pure host
    // overhead in one MTP verify pass (65 rmsnorm calls) and nothing at all in
    // the graphed decode - the graph records the launch once and replays it, so
    // the bug was invisible in the plain-decode numbers and dominated the
    // speculative cycle.  Query once, like active() does.
    static const int limit = [] {
        int lim = 0;
        try {
            auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
            if (!devs.empty()) {
                lim = (int)devs[0].get_info<sycl::info::device::max_work_group_size>();
            }
        } catch (...) {
        }
        if (!lim) {
            lim = active().hw.max_work_group_size;
        }
        if (lim < 32) {
            lim = 32; // a sub-group is the floor for anything below this
        }
        return lim;
    }();
    // round down to a whole sub-group: the two-level SLM reductions all assume
    // the group is a multiple of 32
    return want > limit ? (limit / 32) * 32 : want;
}

} // namespace si::dev