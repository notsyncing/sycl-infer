// Device profiles: every tuned constant that depends on which GPU we run on,
// and the registry that decides which card we are on.
//
// WHY THIS EXISTS
// ---------------
// The kernel library and the engine carry a lot of numbers that were picked by
// measuring on one specific card (historically the reference Iris Xe-LP, then
// the 2x Arc A770).  Most are *not* universal: the occupancy wave, the sub-group
// lattice, the usable SLM budget, the read bandwidth and whether the oneDNN int8
// matmul path pays off at all differ between a discrete DG2 part and an
// integrated Xe-LP part.  Leaving those as literals spread through the sources
// means a new card silently gets the last card's tuning.
//
// The rule (see AGENTS.md "Device profiles"): a value measured on a particular
// GPU belongs in that GPU's profile, not as a literal at its use site.  The use
// site asks for it by name: si::dev::active().<group>.<field>.
//
// LAYOUT -- ONE FILE PER CARD, ONE REGISTRY
// -----------------------------------------
//   src/device/profiles/<card>.cpp   one card's complete implementation: its
//                                    key, its values, and its OWN matcher
//   src/device/device_registry.cpp   the list of those entries, and the
//                                    selection / reporting that reads them
//   src/device/device_profile.h      this file: the struct and the API
//
// A card is one self-contained file, so the registry holds no knowledge of any
// particular card and a card file holds no knowledge of any other.  Adding a card
// is one new file plus one row in kDevices.
//
// Every value is written with a DESIGNATED INITIALIZER -- `.key = ...`,
// `.shape = {.rmsnorm_wg = ...}` -- so the fields are explicit and a reordered or
// inserted field cannot silently shift every value after it.  (That is not
// hypothetical: an earlier draft of this file had rmsnorm_wg under attn_vec and
// every later field printed one slot off.)  The build uses -Wall -Wextra, so an
// omitted field warns rather than defaulting silently.
//
// Provenance is part of the data, not a comment: every card states how its
// numbers were obtained, and a value carried across from another card because
// the two share an architecture is marked INHERITED rather than implied.
#pragma once

#include <string>
#include <vector>
#include <sycl/sycl.hpp>

// Baked in by CMake (-DSYCL_INFER_AOT_PROFILE=<key>); empty means auto-detect.
#ifndef SI_FORCE_DEVICE_PROFILE
#define SI_FORCE_DEVICE_PROFILE ""
#endif

namespace si::dev {

struct profile {
    const char * key;        // "arc_a770" / "iris_xe", what PF_DEVICE_PROFILE matches
    const char * name;       // human name, for the [dev] banner
    const char * provenance; // how these numbers were obtained

    // Hardware facts SYCL already reports; the profile value is the fallback for
    // when the query is unavailable.
    struct hw_t {
        int compute_units;
        int max_work_group_size;
        int local_mem_size; // bytes of SLM per work-group
    } hw;

    // The decode attention launches n_head * dec_splits warps per layer and a
    // Xe-LP EU holds 8 of them; past that the excess is not co-resident and the
    // time steps up.  dec_splits = (wpe/2) * EUs / n_head.  Measured on one A770
    // (512 EUs): nsp 128/144/160/168/176 -> 2.177/1.748/1.699/1.744/2.678 ms per
    // layer at 64k depth, where 176 = 8.25 warps/EU falls off the cliff.
    //
    // Stored DOUBLED (15 = 7.5): the optimum is fractional and an int "7" makes
    // dec_splits 7/4 of what it should be -- 72 instead of 160 on the A770 --
    // which that curve prices at ~1.9x on the attention.
    struct occ_t {
        int warps_per_eu_x2;
    } occ;

    // The u4/k5/codebook decode GEMVs stage two f16 planes per row in SLM and
    // refuse a rows-per-workgroup value that does not fit; nat_gemm's
    // M*KT*APAD tile is sized against the same budget.  48 KB of the 64 KB limit
    // is a deliberate reserve, not the hardware limit.
    struct slm_t {
        int budget_bytes;
    } slm;

    // How much output a kernel needs before splitting is worth it.  These scale
    // with the EU count, so they are the most card-specific of the tuning.
    struct split_t {
        int gemv_rows; // decode int8 GEMV: split K below this many output rows
        int gemm_rows; // prefill row GEMM: same idea, a much wider threshold
        int max;       // cap on the resulting split count
    } split;

    // Per-kernel launch geometry.  rmsnorm_wg is the one field that differs as
    // HARDWARE rather than tuning: the A770 takes a 1024-thread work-group and
    // the Iris Xe only 512, so an unconditional 1024-thread kernel does not
    // launch on the integrated part at all.  It must be a template parameter (a
    // SYCL kernel cannot capture a runtime-initialised global), which also lets
    // the strided load loops unroll.
    struct shape_t {
        int gemv_rows_per_wg; // rows per work-group, vectorized decode GEMV
        int rmsnorm_wg;       // one work-group per row; the widest accepted
        int gdn_cols;         // GDN state rows per warp
        int gdn_warps_per_wg;
        int gdn_vec_max_rows; // float4 state slice only wins below this many
                              // real rows per pass (it costs 4x the registers)
    } shape;

    // The gather's stage-1 max reduction is a latency fix for running ONE
    // work-group over a whole key block: that cost 18.1 ms of the 37.8 ms one
    // attention layer takes at 64k depth, and 64 work-groups gets it to 9.1
    // (256/512 no better).
    struct attn_t {
        int xmx_gather_red;
        int xmx_gather_red_max;
        int vec;          // PF_ATTN_VEC
        int xmx;          // PF_ATTN_XMX (oneDNN int8 matmul prefill attention)
        int xmx_min_keys; // below this the classic kernel wins
        int split_keys;   // PF_ATTN_SPLIT_KEYS
        int dec_group;    // PF_DEC_GROUP (grouped decode attention)
    } attn;

    // Feature defaults.  The native 4/5-bit stores cut bytes per weight, so
    // whether they pay depends on the card's read bandwidth relative to its
    // compute -- a per-device measurement, hence a profile default rather than a
    // constant.  The env var overrides each one for A/B.
    struct wt_t {
        int w4;           // PF_W4 (native u4 store)
        int k5;           // PF_K5 (native 5-bit store)
        int cb4;          // PF_CB4 (codebook 4-bit store)
        int dp4a;         // PF_DP4A
        int gemm_dnnl;    // PF_GEMM_DNNL (oneDNN prefill GEMM)
        int dpas_in_gemm; // esimd::dpas: measured 37x slower than dp4a in a real
                          // GEMM on both cards (7.44 ms against nat_gemm's 0.200
                          // at M=5), so nothing depends on it; kept for a part
                          // where it wins
    } wt;
};

// ---------------------------------------------------------------------------
// The registry.  A card contributes exactly ONE `device_entry`, defined in its
// own translation unit under src/device/profiles/, holding its key, its values
// and its own matcher -- so "how is this GPU recognised" lives with the card it
// recognises, not in a central switch.
// ---------------------------------------------------------------------------
struct device_entry {
    const char * key;                                          // explicit
    const profile * prof;                                      // that card's values
    bool (*matches)(const std::string & device_name);          // that card's matcher
};

// The registered cards, in match order: the first entry whose matcher accepts the
// device name wins, so a broad matcher belongs after any specific one.  Adding a
// card = one file under src/device/profiles/ + one row here.
extern const device_entry *const kDevices[];
extern const int kNumDevices;

// The loud fallback for a card nobody claims: key "unknown", hardware facts at 0
// (so a derived quantity that divides by the EU count is obviously wrong rather
// than plausibly right), and every tuning a copy of arc_a770's.
const profile & unknown_profile();

// The profile for the GPU we are running on, chosen once from the SYCL device
// name and cached for the process.  `PF_DEVICE_PROFILE=<key>` forces one;
// `PF_DEVICE_INFO=1` prints the resolved profile and its provenance.
//
// Prefer for_queue() in anything that is handed a queue: active() is
// process-wide, so it silently applies the first GPU's tuning to a second one.
const profile & active();

// The profile of the device behind `q`, cached per device identity (SYCL's
// device operator== compares handles, so the lookup is a pointer compare and
// not the get_devices() enumeration that cost 3.2 ms per launch).  A kernel
// launcher must use this, not active(), or a heterogeneous --layer-map runs
// every card with the first card's tuning.
const profile & for_queue(const sycl::queue & q);

// Work-group width clamp against one specific device.  The device limit is a
// hardware fact and cannot change over a run, but querying it is not free
// (`device::get_devices()` measured 3.2 ms per call: 210 ms of host time in one
// MTP verify pass, invisible in the graphed decode because the graph records the
// launch once), so it is cached per device alongside the profile.
int wg_clamped_for_queue(const sycl::queue & q, int want);

// The "env overrides a profile default" pattern, per device.
//
// Nearly every tuning is `env ? atoi(env) : active().shape.x`, and the old form
// of that - a function-local `static const` initialised from a profile value -
// silently pins the whole process to whichever card resolved first.  These two
// helpers make the correct shape the only one that is convenient: the env is the
// same on every card so it is read once, the default is read per call from the
// queue's own device (a pointer compare through for_queue, not a driver query).
// Returns `profile_default` when the env is unset or parses to 0.
int profile_int(const sycl::queue & q, const char * env_name, int profile_default);
// `env=0` wins over a true default, anything else lets the default stand.
bool profile_flag(const sycl::queue & q, const char * env_name, bool profile_default);

// Resolve from a SYCL device name without touching any device state, by asking
// each registered card's own matcher.
const profile & for_name(const std::string & device_name);

// Why a set of GPUs cannot share one profile, if it cannot.
//
// active() resolves once per process from the *first* GPU and wg_clamped() caches
// that card's max_work_group_size, so every launch - on every card - reads the
// first card's tuning.  A heterogeneous --layer-map therefore does not merely
// run the second card with the wrong occupancy/SLM budget/split rows: it can
// launch a work-group the second card cannot accept (the A770 takes 1024
// threads, the Iris Xe only 512), which is a hard launch failure rather than a
// slowdown.  So the split is refused rather than silently mis-tuned.
//
// Pure over device names, so a test can drive it without two cards in the box.
struct profile_split {
    bool homogeneous = true;
    std::string first_name, first_key;
    std::string other_name, other_key; // the first device that disagrees
};
profile_split check_profiles_homogeneous(const std::vector<std::string> & device_names);

// The banner.  A shared helper so the auto-detect path and a forced
// PF_DEVICE_PROFILE both report themselves; returning early from the forced
// branch used to skip this, which made a forced profile look like it had not
// been applied.
void report(const profile & p, const char * how);

// Clamp a work-group width to what the device actually accepts.  This is what
// makes a profile safe on a card nobody claims: the profile supplies the *wanted*
// width (a tuning) and this subtracts the hardware limit, so an unprofiled card
// gets a smaller kernel rather than a launch failure.
int wg_clamped(int want);

} // namespace si::dev