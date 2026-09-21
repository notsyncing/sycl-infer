#include "kernels.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

using namespace sycl;

void copy_row_launch(queue & q, const float * src, float * dst, const step_info * info, int n, int row) {
    q.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) {
        const int i = it.get_local_id(0);
        int rr = row;
        if (rr < 0) {
            // last real token (mode 2 may have a partial last row)
            const int lr = info->n_rows - 1;
            const int nr = info->n_real_row[lr] > 0 ? info->n_real_row[lr] : info->n_real;
            rr = lr * info->tpb + nr - 1;
        }
        const float * s = src + (size_t)rr * n;
        for (int j = i; j < n; j += 256) {
            dst[j] = s[j];
        }
    });
}

} // namespace si
