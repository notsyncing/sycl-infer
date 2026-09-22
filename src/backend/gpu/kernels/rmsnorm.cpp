#include "kernels.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

using namespace sycl;

// ---------------------------------------------------------------------------
// One workgroup of kWG threads per row.  Decode runs a single row, so the
// workgroup *is* the whole parallelism: 256 threads over n_embd=5120 meant 20
// serial load iterations plus an 8-step SLM tree (measured 32 us/call, 4.1 ms of
// a 85 ms 27B multi-device decode).  1024 threads cut that to 5 iterations, and
// a sub-group + SLM two-level reduction replaces the 8 barriers with 2.
namespace {
constexpr int kWG = 1024;
constexpr int kNSG = kWG / 32;
} // namespace

void rmsnorm_launch(queue & q, const float * x, const float * w, float * out, int n_rows, int n, float eps) {
    q.submit([&](handler & h) {
        local_accessor<float, 1> red(kNSG, h);
        h.parallel_for(nd_range<1>((size_t)n_rows * kWG, kWG), [=](nd_item<1> it) {
            const int row = it.get_group(0);
            const int tid = it.get_local_id(0);
            const sub_group sg = it.get_sub_group();
            const float * xr = x + (size_t)row * n;
            float ss = 0.f;
            for (int i = tid; i < n; i += kWG) {
                ss += xr[i] * xr[i];
            }
            ss = reduce_over_group(sg, ss, plus<float>());
            const int sgl = (int)sg.get_local_linear_id();
            const int sgi = (int)sg.get_group_linear_id();
            if (sgl == 0) {
                red[sgi] = ss;
            }
            it.barrier();
            if (sgi == 0) {
                float v = (sgl < kNSG) ? red[sgl] : 0.f;
                v = reduce_over_group(sg, v, plus<float>());
                if (sgl == 0) {
                    red[0] = v;
                }
            }
            it.barrier();
            const float scale = 1.0f / sycl::sqrt(red[0] / n + eps);
            float * orow = out + (size_t)row * n;
            for (int i = tid; i < n; i += kWG) {
                orow[i] = xr[i] * scale * w[i];
            }
        });
    });
}

} // namespace si
