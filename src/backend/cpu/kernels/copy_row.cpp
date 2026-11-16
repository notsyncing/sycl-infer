// CPU copy_row (mirror of src/backend/gpu/kernels/copy_row.cpp): copy the last
// (or a named) row of the per-token buffer into the single hidden state.
#include "common.h"

namespace si {

void cpu_copy_row(const float * src, float * dst, const cpu_step_info * info, int n, int row) {
    int rr = row;
    if (rr < 0) {
        // Last real token of the last row.  A chunk-batched prefill (mode 2)
        // may end with a partial row, so its real count is n_real_row, not
        // n_real - taking n_real reads one of the row's dead token slots.
        const int lr = info->n_rows - 1;
        const int nr = info->n_real_row[lr] > 0 ? info->n_real_row[lr] : info->n_real;
        rr = lr * info->tpb + nr - 1;
    }
    std::memcpy(dst, src + (size_t)rr * n, (size_t)n * 4);
}

} // namespace si
