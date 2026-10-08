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

    if (g_fail) {
        fprintf(stderr, "test_dev_profile: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_dev_profile: all checks OK\n");
    return 0;
}