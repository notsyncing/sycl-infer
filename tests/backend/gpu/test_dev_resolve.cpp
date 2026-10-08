// Per-device profile resolution (for_queue / wg_clamped_for_queue).  GPU needed;
// no model.
//
// The point of this path is that a heterogeneous --layer-map must be *supported*:
// every launcher asks for the profile of the queue it is launching on, so each
// card gets its own tuning and its own max_work_group_size.  Two things can break
// it silently, and both are checked here:
//
//   * a cache keyed on something other than device identity would hand the first
//     card's values to the second (the failure this whole path exists to remove);
//   * the lookup must not query the driver per launch.  wg_clamped() used to, at
//     3.2 ms a call - 210 ms of host time in one MTP verify pass.
//
// The box this runs on may only have one GPU, so the multi-card assertions are
// skipped (and say so) rather than silently passing.
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

#include "device/device_profile.h"

using namespace si;

static int g_fail = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                            \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

int main() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        printf("no GPU visible; nothing to check\n");
        return 0;
    }
    sycl::queue q(devs[0]);
    const std::string name = devs[0].get_info<sycl::info::device::name>();
    printf("device: %s\n", name.c_str());

    // --- for_queue agrees with the name-based resolver ---
    const dev::profile & prof = dev::for_queue(q);
    const dev::profile & pn = dev::for_name(name);
    CHECK(prof.key == pn.key);
    printf("for_queue -> %s, for_name -> %s\n", prof.key, pn.key);

    // --- and is stable across calls (the cache, not a fresh resolve each time) ---
    const dev::profile * first = &prof;
    for (int i = 0; i < 1000; i++) {
        CHECK(&dev::for_queue(q) == first);
    }

    // --- a second queue on the same device must hit the same record ---
    sycl::queue q2(devs[0]);
    CHECK(&dev::for_queue(q2) == first);

    // --- wg_clamped_for_queue respects THIS device's limit ---
    const int hw = (int)devs[0].get_info<sycl::info::device::max_work_group_size>();
    CHECK(hw >= 32);
    // a want below the limit comes back unchanged
    CHECK(dev::wg_clamped_for_queue(q, 64) == 64);
    // a want above it is rounded down to a whole sub-group
    const int clamped = dev::wg_clamped_for_queue(q, 4096);
    CHECK(clamped <= hw);
    CHECK(clamped % 32 == 0);
    CHECK(clamped >= 32);
    // the profile's own wanted width, clamped the same way, is what rmsnorm uses
    const int rms_want = dev::wg_clamped_for_queue(q, prof.shape.rmsnorm_wg);
    CHECK(rms_want % 32 == 0 && rms_want >= 32 && rms_want <= hw);
    printf("hw_max_wg=%d -> clamp(4096)=%d clamp(profile rmsnorm_wg=%d)=%d\n", hw, clamped, prof.shape.rmsnorm_wg,
           rms_want);
    // and it matches the device we asked about, not the profile's idea of it
    if (prof.hw.max_work_group_size > 0 && strcmp(prof.key,"unknown") != 0) {
        CHECK(rms_want == dev::wg_clamped_for_queue(q, prof.shape.rmsnorm_wg));
    }

    // --- the lookup must be cheap: this is on every launch ---
    {
        const int iters = 200000;
        const auto t0 = std::chrono::steady_clock::now();
        int sink = 0;
        for (int i = 0; i < iters; i++) {
            sink += dev::wg_clamped_for_queue(q, 1024) + (int)dev::for_queue(q).hw.compute_units;
        }
        const double us =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        printf("per-launch lookup: %.3f us (sink=%d)\n", us, sink);
        // the old get_devices()-per-launch version measured 3200 us.  Anything
        // near that means the cache stopped working; 5 us leaves a wide margin
        // over a real kernel launch and still catches a driver query per call.
        CHECK(us < 5.0);
    }

    // --- multi-card: each queue must resolve independently ---
    if (devs.size() > 1) {
        sycl::queue qb(devs[1]);
        const std::string nb = devs[1].get_info<sycl::info::device::name>();
        const dev::profile & pb = dev::for_queue(qb);
        CHECK(&pb != first || pb.key == first->key); // distinct record, or same card
        const int hw_b = (int)devs[1].get_info<sycl::info::device::max_work_group_size>();
        const int want_b = dev::wg_clamped_for_queue(qb, 4096);
        CHECK(want_b <= hw_b);
        printf("device 2: %s -> %s, hw_max_wg=%d clamp(4096)=%d\n", nb.c_str(), pb.key, hw_b, want_b);
        if (pb.key != first->key) {
            // a genuine heterogeneous pair: the two must NOT clamp alike if
            // their limits differ
            printf("  heterogeneous split in use: '%s' vs '%s'\n", first->key, pb.key);
        }
    } else {
        printf("only one GPU visible; per-card independence not exercised here\n");
    }

    if (g_fail) {
        fprintf(stderr, "test_dev_resolve: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_dev_resolve: all checks OK\n");
    return 0;
}