#include "kernels.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

using namespace sycl;

// ---------------------------------------------------------------------------
void rmsnorm_launch(queue & q, const float * x, const float * w, float * out, int n_rows, int n, float eps) {
    q.submit([&](handler & h) {
        local_accessor<float, 1> red(256, h);
        h.parallel_for(nd_range<1>((size_t)n_rows * 256, 256), [=](nd_item<1> it) {
            const int row = it.get_group(0);
            const int tid = it.get_local_id(0);
            const float * xr = x + (size_t)row * n;
            float ss = 0.f;
            for (int i = tid; i < n; i += 256) {
                ss += xr[i] * xr[i];
            }
            red[tid] = ss;
            it.barrier();
            for (int s = 128; s > 0; s >>= 1) {
                if (tid < s) {
                    red[tid] += red[tid + s];
                }
                it.barrier();
            }
            const float scale = 1.0f / sycl::sqrt(red[0] / n + eps);
            float * orow = out + (size_t)row * n;
            for (int i = tid; i < n; i += 256) {
                orow[i] = xr[i] * scale * w[i];
            }
        });
    });
}

} // namespace si
