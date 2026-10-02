#include "kernels.h"
#include "device/device_profile.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

using namespace sycl;

// ---------------------------------------------------------------------------
// One workgroup of WG threads per row.  Decode runs a single row, so the
// workgroup *is* the whole parallelism: 256 threads over n_embd=5120 meant 20
// serial load iterations plus an 8-step SLM tree (measured 32 us/call, 4.1 ms of
// a 85 ms 27B multi-device decode).  1024 threads cut that to 5 iterations, and
// a sub-group + SLM two-level reduction replaces the 8 barriers with 2.
//
// WG is a TEMPLATE parameter, not a runtime variable, and that is a device
// difference rather than a stylistic one: the width comes from the device profile
// clamped to the device's real max_work_group_size, because the A770 accepts
// 1024 threads and the Iris Xe only 512 -- a 1024-thread group does not launch on
// the latter at all.  A runtime value would also stop the compiler from unrolling
// the strided load loops.
namespace {
template <int WG>
static void rmsnorm_impl(queue & q, const float * x, const float * w, float * out, int n_rows, int n, float eps) {
    constexpr int NSG = WG / 32; // one SLM slot per sub-group
    q.submit([&](handler & h) {
        local_accessor<float, 1> red(NSG, h);
        h.parallel_for(nd_range<1>((size_t)n_rows * WG, WG), [=](nd_item<1> it) {
            const int row = it.get_group(0);
            const int tid = it.get_local_id(0);
            const sub_group sg = it.get_sub_group();
            const float * xr = x + (size_t)row * n;
            float ss = 0.f;
            for (int i = tid; i < n; i += WG) {
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
                // NSG slots were written; read exactly those (the group is only
                // NSG sub-groups wide, so there is nothing else to mask off, but
                // keep the guard so a sub-group count above 32 would still work)
                float v = (sgl < NSG && sgl < 32) ? red[sgl] : 0.f;
                v = reduce_over_group(sg, v, plus<float>());
                if (sgl == 0) {
                    red[0] = v;
                }
            }
            it.barrier();
            const float scale = 1.0f / sycl::sqrt(red[0] / n + eps);
            float * orow = out + (size_t)row * n;
            for (int i = tid; i < n; i += WG) {
                orow[i] = xr[i] * scale * w[i];
            }
        });
    });
}
} // namespace

void rmsnorm_launch(queue & q, const float * x, const float * w, float * out, int n_rows, int n, float eps) {
    // the widest width the device accepts, from the profile (see the header
    // comment); anything narrower than 256 is not worth an instantiation
    const int want = si::dev::wg_clamped(si::dev::active().shape.rmsnorm_wg);
    if (want >= 1024) {
        rmsnorm_impl<1024>(q, x, w, out, n_rows, n, eps);
    } else if (want >= 512) {
        rmsnorm_impl<512>(q, x, w, out, n_rows, n, eps);
    } else if (want >= 256) {
        rmsnorm_impl<256>(q, x, w, out, n_rows, n, eps);
    } else {
        rmsnorm_impl<128>(q, x, w, out, n_rows, n, eps);
    }
}

} // namespace si