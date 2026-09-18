#include "kernel_utils.h"

namespace si {
namespace kd {

// Workspace for K-split partial sums (one queue/device per process).
// In chunk-batched prefill several rows' GEMMs are in flight at once, so each
// row gets its own slot (the command graph does not order USM accesses, only
// recorded dependencies).
float * gemm_ws(sycl::queue & q, size_t need) {
    static float * buf = nullptr;
    static size_t cap = 0;
    if (need > cap) {
        if (buf) {
            sycl::free(buf, q);
        }
        buf = sycl::malloc_device<float>(need, q);
        cap = need;
        if (!buf) {
            cap = 0;
        }
    }
    return buf;
}

} // namespace kd
} // namespace si
