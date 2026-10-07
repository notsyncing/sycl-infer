// Ownership-helper tests for the vision/audio encoder towers (no model files).
// pin_dev_queue / free_dev_ptrs live in enc_common.h; the model destructors
// are thin wrappers over free_dev_ptrs, so this pins down the helper contract
// they rely on: pin-once, throw on a second queue (freeing on another queue
// is UB), release-a-whole-list, and null-safety.  Needs a GPU (device USM)
// but no GGUF: the full upload/scratch/destructor path is exercised by
// test_multimodal's test_device whenever a head_dim==64 mmproj is present.
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include "enc_common.h"

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
    sycl::queue q(sycl::gpu_selector_v);
    printf("dev: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // pin-once + same-queue idempotent
    {
        sycl::queue * slot = nullptr;
        sycl::queue & r = pin_dev_queue(slot, q, "t");
        CHECK(slot == &q);
        CHECK(&r == &q);
        sycl::queue & r2 = pin_dev_queue(slot, q, "t");
        CHECK(&r2 == &q);
    }
    // a second queue object is rejected loudly (fail fast, not silent UB)
    {
        sycl::queue q2(q.get_device());
        sycl::queue * slot = nullptr;
        pin_dev_queue(slot, q, "t");
        bool threw = false;
        try {
            pin_dev_queue(slot, q2, "t");
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
    }
    // free_dev_ptrs: null queue is a no-op (never uploaded), nulls are skipped
    {
        free_dev_ptrs(nullptr, {nullptr, nullptr});
        float * a = sycl::malloc_device<float>(16, q);
        float * b = sycl::malloc_device<float>(32, q);
        CHECK(a && b);
        q.memset(a, 0, 16 * 4).wait();
        free_dev_ptrs(&q, {a, nullptr, b});
        // balanced: allocate + free the same count again must still succeed
        float * c = sycl::malloc_device<float>(48, q);
        CHECK(c);
        free_dev_ptrs(&q, {c});
    }

    if (g_fail) {
        fprintf(stderr, "test_enc_own: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_enc_own: all checks OK\n");
    return 0;
}
