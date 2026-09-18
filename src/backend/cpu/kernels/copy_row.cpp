// CPU copy_row (mirror of src/backend/gpu/kernels/copy_row.cpp): copy the last
// (or a named) row of the per-token buffer into the single hidden state.
#include "common.h"

namespace si {

void cpu_copy_row(const float * src, float * dst, const cpu_step_info * info, int n, int row) {
    const int rr = (row >= 0) ? row : (info->n_rows * info->n_real - 1);
    std::memcpy(dst, src + (size_t)rr * n, (size_t)n * 4);
}

} // namespace si
