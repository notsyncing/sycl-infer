// Device-profile homogeneity (check_profiles_homogeneous).  Hermetic: no GPU.
//
// Why this check exists: active() resolves one profile per process from the
// *first* GPU, and wg_clamped() caches that card's max_work_group_size, so every
// launch - on every card - reads the first card's tuning.  A heterogeneous
// --layer-map therefore mis-tunes the second card, and can launch a work-group
// it cannot accept (the A770 allows 1024 threads, the Iris Xe only 512), which
// is a launch failure rather than a slowdown.  The pure helper is driven here
// with synthetic device names so the logic is covered without two cards.
#include <cstdio>
#include <string>
#include <vector>

#include "device/device_profile.h"

using namespace si;

static int g_fail = 0;

static void homogeneous(const char * what, const std::vector<std::string> & names, const char * expect_key) {
    const dev::profile_split s = dev::check_profiles_homogeneous(names);
    if (!s.homogeneous) {
        fprintf(stderr, "FAIL %s: expected homogeneous, got '%s' vs '%s'\n", what, s.first_key.c_str(),
                s.other_key.c_str());
        g_fail++;
        return;
    }
    if (std::string(s.first_key) != expect_key) {
        fprintf(stderr, "FAIL %s: expected profile '%s', got '%s'\n", what, expect_key, s.first_key.c_str());
        g_fail++;
        return;
    }
    printf("  homogeneous %-38s -> %s\n", what, s.first_key.c_str());
}

static void heterogeneous(const char * what, const std::vector<std::string> & names, const char * expect_first,
                          const char * expect_other) {
    const dev::profile_split s = dev::check_profiles_homogeneous(names);
    if (s.homogeneous) {
        fprintf(stderr, "FAIL %s: expected a split, got homogeneous '%s'\n", what, s.first_key.c_str());
        g_fail++;
        return;
    }
    if (s.first_key != expect_first || s.other_key != expect_other) {
        fprintf(stderr, "FAIL %s: expected '%s' vs '%s', got '%s' vs '%s'\n", what, expect_first, expect_other,
                s.first_key.c_str(), s.other_key.c_str());
        g_fail++;
        return;
    }
    // the message has to name both cards, or an operator cannot act on it
    if (s.first_name.empty() || s.other_name.empty()) {
        fprintf(stderr, "FAIL %s: the split does not name the devices\n", what);
        g_fail++;
        return;
    }
    printf("  SPLIT      %-38s -> '%s' vs '%s'\n", what, s.first_key.c_str(), s.other_key.c_str());
}

static void expect_int(const char * what, int got, int want) {
    if (got != want) {
        fprintf(stderr, "FAIL %s: expected %d, got %d\n", what, want, got);
        g_fail++;
        return;
    }
    printf("  value      %-38s -> %d\n", what, got);
}

// The narrowed device-profile fields (moe.* and the three GEMV shape gates).
// They are pinned here for the same reason the split/shape groups are pinned in
// the card files: a field that silently keeps a literal at its use site reads
// as "the profile decided", and a card that omits one gets the compiler's zero.
static void profile_values(const char * who, const dev::profile & p) {
    expect_int((std::string(who) + " moe.cfg_gate").c_str(), p.moe.cfg_gate, 43);
    expect_int((std::string(who) + " moe.cfg_down").c_str(), p.moe.cfg_down, 43);
    // 0 is the measured value: "auto" picks one row per expert for a
    // single-token decode and two for a batched one.
    expect_int((std::string(who) + " moe.cfg_dec").c_str(), p.moe.cfg_dec, 0);
    expect_int((std::string(who) + " moe.flat").c_str(), p.moe.flat, 1);
    expect_int((std::string(who) + " moe.flat_dec").c_str(), p.moe.flat_dec, 0);
    expect_int((std::string(who) + " moe.dp4a").c_str(), p.moe.dp4a, 1);
    expect_int((std::string(who) + " moe.dp4a_dec").c_str(), p.moe.dp4a_dec, 1);
    expect_int((std::string(who) + " moe.upload_ring").c_str(), p.moe.upload_ring, 32);
    expect_int((std::string(who) + " moe.pf_wmin").c_str(), p.moe.pf_wmin, 24);
    expect_int((std::string(who) + " shape.w4_rows_per_wg").c_str(), p.shape.w4_rows_per_wg, 16);
    expect_int((std::string(who) + " shape.q8_rows").c_str(), p.shape.q8_rows, 512);
    expect_int((std::string(who) + " shape.f32_rows").c_str(), p.shape.f32_rows, 128);
    // Invariants a card file cannot express: only 8/16/32 are instantiated for
    // the native-weight GEMV, and a ring/workspace of 0 would disable a path
    // rather than tune it.
    if (p.shape.w4_rows_per_wg != 8 && p.shape.w4_rows_per_wg != 16 && p.shape.w4_rows_per_wg != 32) {
        fprintf(stderr, "FAIL %s shape.w4_rows_per_wg=%d is not an instantiated tile\n", who,
                p.shape.w4_rows_per_wg);
        g_fail++;
    }
    if (p.moe.upload_ring <= 0 || p.moe.pf_wmin < 1) {
        fprintf(stderr, "FAIL %s moe ring/workspace must be positive (ring=%d wmin=%d)\n", who,
                p.moe.upload_ring, p.moe.pf_wmin);
        g_fail++;
    }
}

int main() {
    printf("check_profiles_homogeneous\n");
    // The names the two shipped profiles match.
    const std::string a770 = "Intel(R) Arc(TM) A770 Graphics";
    const std::string iris = "Intel(R) Iris(R) Xe Graphics";

    homogeneous("empty list", {}, "");
    homogeneous("one A770", {a770}, "arc_a770");
    homogeneous("two identical A770s", {a770, a770}, "arc_a770");
    homogeneous("three identical A770s", {a770, a770, a770}, "arc_a770");
    homogeneous("two Iris Xe", {iris, iris}, "iris_xe");

    // The case that matters: the two shipped cards have different work-group
    // limits (1024 vs 512), so a split across them cannot be launched safely.
    heterogeneous("A770 + Iris Xe", {a770, iris}, "arc_a770", "iris_xe");
    heterogeneous("Iris Xe + A770 (order)", {iris, a770}, "iris_xe", "arc_a770");
    // The disagreement may be anywhere in the list, not just position 1.
    heterogeneous("A770, A770, Iris Xe", {a770, a770, iris}, "arc_a770", "iris_xe");

    // An unprofiled card resolves to "unknown"; two of those are homogeneous
    // (same wrong-but-consistent tuning), and they differ from a real card.
    heterogeneous("A770 + unknown card", {a770, "Some Future GPU"}, "arc_a770", "unknown");
    homogeneous("two unknown cards", {"Future GPU A", "Future GPU B"}, "unknown");

    printf("profile values (moe.* and the GEMV shape gates)\n");
    profile_values("arc_a770", dev::for_name(a770));
    // The Iris profile inherits these from the A770 and SAYS SO in its
    // provenance; the values still have to be present and identical, because a
    // missing field is a zero tile, not "unmeasured".
    profile_values("iris_xe", dev::for_name(iris));
    // The unclaimed card is a copy of the A770 row, new fields included.
    profile_values("unknown", dev::unknown_profile());
    // ...with its hardware facts zeroed, so a derived quantity cannot look
    // plausibly right on a card nobody profiled.
    if (dev::unknown_profile().hw.compute_units != 0 || dev::unknown_profile().hw.has_dpas != -1) {
        fprintf(stderr, "FAIL unknown profile must zero its hardware facts and leave DPAS unasked\n");
        g_fail++;
    }

    if (g_fail) {
        fprintf(stderr, "test_dev_profile: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_dev_profile: all checks OK\n");
    return 0;
}