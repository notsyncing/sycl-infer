// CPU RMSNorm (mirror of src/backend/gpu/kernels/rmsnorm.cpp).  The reduction
// and the scaling loop are dispatched on the host ISA in common.cpp.
#include "common.h"

#include <cstdio>
#include <cstdlib>

namespace si {

void cpu_rmsnorm(const float * x, const float * w, float * out, int n_rows, int n, float eps) {
    if (getenv("PF_CPU_DBG")) {
        fprintf(stderr, "[cpu_rmsnorm] x=%p w=%p out=%p rows=%d n=%d\n", (const void *)x, (const void *)w,
                (void *)out, n_rows, n);
    }
    if (!x || !w || !out || n <= 0 || n_rows <= 0) {
        return;
    }
    par(n_rows, [&](int r) { rmsnorm_row(x + (size_t)r * n, w, out + (size_t)r * n, n, eps); });
}

} // namespace si
