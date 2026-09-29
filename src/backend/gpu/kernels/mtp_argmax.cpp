// Device-side row argmax: replaces the host scan of the MTP verify's logits,
// which needs a host copy of M*n_vocab floats per speculative cycle (~7 MB at
// M=7, n_vocab=248320 - comparable to a whole draft pass) and then a serial
// 7*248320-element comparison on one core.
//
// One work-item per row, scanning its row with a fixed stride-1 order.  The
// verify is only 2..13 rows (the draft length plus one), so the parallelism that
// matters is *within* a row and a whole row is small next to the GEMM that
// produced it; a simple per-row scan is easier to keep bit-identical to the
// host's `v[i] > v[best]` than a tree reduction, and the rows are independent
// so they still overlap each other on the device.
#include "kernels.h"
#include "kernel_utils.h"

namespace si {
using namespace sycl;

void mtp_argmax_launch(queue & q, const float * logits, int n, int32_t * out_idx, float * out_val, int M) {
    if (M <= 0 || n <= 0) {
        return;
    }
    q.submit([&](handler & h) {
        h.parallel_for(range<1>((size_t)M), [=](id<1> it) {
            const int r = (int)it.get(0);
            const float * row = logits + (size_t)r * n;
            // same order and same strict > as the host scan, so the result is
            // bit-identical (the lowest index wins every tie)
            float best = row[0];
            int bidx = 0;
            for (int i = 1; i < n; i++) {
                if (row[i] > best) {
                    best = row[i];
                    bidx = i;
                }
            }
            out_idx[r] = bidx;
            if (out_val) {
                out_val[r] = best;
            }
        });
    });
}

} // namespace si

