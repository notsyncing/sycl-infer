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
#include <mutex>
#include <string>
#include <vector>
#include "common/env.h"

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

profile_split check_profiles_homogeneous(const std::vector<std::string> & device_names) {
    profile_split r;
    if (device_names.empty()) {
        return r;
    }
    const profile & first = for_name(device_names[0]);
    r.first_name = device_names[0];
    r.first_key = first.key;
    for (size_t i = 1; i < device_names.size(); i++) {
        const profile & p = for_name(device_names[i]);
        if (std::strcmp(p.key, first.key) != 0) {
            r.homogeneous = false;
            r.other_name = device_names[i];
            r.other_key = p.key;
            return r;
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// Per-device resolution, so a heterogeneous split is actually supported.
//
// Everything above is process-wide: one profile and one max_work_group_size,
// both from the first GPU.  Launchers, though, are handed a `queue`, so they
// can ask which device they are about to launch on.  Two rules make that cheap
// enough for a launch path:
//
//   * the lookup keys on sycl::device equality, which compares the underlying
//     handle - a couple of pointer compares, NOT the device enumeration that
//     cost 3.2 ms per call (210 ms of host time in one MTP verify pass);
//   * the name and the work-group limit are queried exactly once per device, on
//     first sight, and cached.
//
// Both facts are cached together because both come from the same device and the
// work-group limit is a driver query in its own right.
namespace {

struct dev_record {
    sycl::device dev;
    const profile * prof = nullptr;
    int max_wg = 0;
};

std::mutex & dev_cache_mutex() {
    static std::mutex m;
    return m;
}
std::vector<dev_record> & dev_cache() {
    static std::vector<dev_record> v;
    return v;
}

// Resolve and cache one device.  Called with the lock held by the slow path.
const dev_record & fill(sycl::device d) {
    dev_record rec;
    rec.dev = d;
    const char * forced = si::env::str("PF_DEVICE_PROFILE");
    const char * baked = SI_FORCE_DEVICE_PROFILE;
    if (forced && *forced) {
        baked = forced;
    }
    if (baked && *baked) {
        // A forced profile applies to every queue; the hardware facts still come
        // from the device, which is the whole split between tuning and fact.
        for (int i = 0; i < kNumDevices; i++) {
            if (std::strcmp(baked, kDevices[i]->key) == 0) {
                rec.prof = kDevices[i]->prof;
                break;
            }
        }
        if (!rec.prof && std::strcmp(baked, "unknown") == 0) {
            rec.prof = &unknown_profile();
        }
        if (rec.prof) {
            report(*rec.prof, "forced by PF_DEVICE_PROFILE / build (per queue)");
        }
    }
    if (!rec.prof) {
        std::string name;
        try {
            name = d.get_info<sycl::info::device::name>();
        } catch (...) {
        }
        rec.prof = &for_name(name);
        report(*rec.prof, ("resolved from this queue's device: " + name).c_str());
    }
    try {
        rec.max_wg = (int)d.get_info<sycl::info::device::max_work_group_size>();
    } catch (...) {
    }
    if (!rec.max_wg) {
        rec.max_wg = rec.prof->hw.max_work_group_size;
    }
    if (rec.max_wg < 32) {
        rec.max_wg = 32; // a sub-group is the floor for anything below this
    }
    dev_cache().push_back(rec);
    return dev_cache().back();
}

const dev_record & record_for(const sycl::device & d) {
    {
        std::lock_guard<std::mutex> lk(dev_cache_mutex());
        for (const dev_record & r : dev_cache()) {
            if (r.dev == d) {
                return r;
            }
        }
    }
    // resolve outside the lock: a driver query under a mutex is the one thing
    // that could serialise two engines' first launches
    std::lock_guard<std::mutex> lk(dev_cache_mutex());
    for (const dev_record & r : dev_cache()) {
        if (r.dev == d) {
            return r;
        }
    }
    return fill(d);
}

} // namespace

const profile & for_queue(const sycl::queue & q) {
    return *record_for(q.get_device()).prof;
}

int wg_clamped_impl(int want, const sycl::device * dev) {
    int limit;
    if (dev) {
        limit = record_for(*dev).max_wg;
    } else {
        static const int limit0 = [] {
            auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
            if (devs.empty()) {
                return 32;
            }
            const sycl::device first = devs[0];
            return record_for(first).max_wg;
        }();
        limit = limit0;
    }
    // round down to a whole sub-group: the two-level SLM reductions all assume
    // the group is a multiple of 32
    return want > limit ? (limit / 32) * 32 : want;
}

int wg_clamped_for_queue(const sycl::queue & q, int want) {
    const sycl::device d = q.get_device();
    return wg_clamped_impl(want, &d);
}

// Env values are the same on every card, so they are read once and cached.  The
// cache is a small fixed table keyed by the *name*, not a function-local static:
// a static inside profile_int would be per function, so two different env names
// passed to the same call site would share one slot - and templating on the
// literal is not an option either, since `const char (&)[N]` instantiates by
// length and two same-length literals collapse into one instantiation.
int cached_env(const char * name, int unset_value) {
    struct slot {
        const char * name;
        int val;
    };
    static std::mutex m;
    static slot table[32];
    static int used = 0;
    std::lock_guard<std::mutex> lk(m);
    for (int i = 0; i < used; i++) {
        if (std::strcmp(table[i].name, name) == 0) {
            return table[i].val;
        }
    }
    const char * e = si::env::str(name);
    const int v = e ? std::atoi(e) : unset_value;
    if (used < (int)(sizeof(table) / sizeof(table[0]))) {
        table[used].name = name;
        table[used].val = v;
        used++;
    }
    return v;
}

int profile_int(const sycl::queue & q, const char * env_name, int profile_default) {
    const int v = cached_env(env_name, 0);
    return v ? v : profile_default;
}

bool profile_flag(const sycl::queue & q, const char * env_name, bool profile_default) {
    const int v = cached_env(env_name, -1);
    return v >= 0 ? (v != 0) : profile_default;
}

void report(const profile & p, const char * how) {
    if (si::env::flag("PF_DEVICE_INFO")) {
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
    static const profile * resolved = []() -> const profile * {
        // SI_FORCE_DEVICE_PROFILE is baked in by CMake
        // (-DSYCL_INFER_AOT_PROFILE=...) so an AOT binary built for one card
        // still selects that card's profile when the SYCL device name would
        // mislead.  PF_DEVICE_PROFILE is the runtime equivalent and wins, so a
        // baked binary can still be re-pointed for an A/B.
        const char * baked = SI_FORCE_DEVICE_PROFILE;
        if (const char * forced = si::env::str("PF_DEVICE_PROFILE")) {
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
    return wg_clamped_impl(want, /*dev=*/nullptr);
}

} // namespace si::dev