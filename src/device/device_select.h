#pragma once
// GPU device selection.
//
// `sycl::gpu_selector_v` is not a stable choice on a machine that exposes the
// same card through two backends: this box enumerates the two Arc A770s under
// OpenCL *and* Level Zero, and `gpu_selector_v` picks the OpenCL one.  The
// engine's decode path records a SYCL command graph, which the OpenCL backend
// does not implement, so the run then exits with
//
//     opencl backend is not supported by SYCL Graph extension
//
// even though a perfectly usable Level Zero device is right there.  The
// selector below prefers Level Zero when it is present and falls back to the
// ordinary `gpu_selector_v` when it is not (an OpenCL-only or CPU-only host
// keeps its old behaviour).  `ONEAPI_DEVICE_SELECTOR` is still honoured by
// `get_devices()` first, so an explicit selector continues to win.
//
// The returned vector is the GPU list in SYCL's enumeration order, filtered to
// one backend, so `gpu.N` in `--layer-map` / `--device` addresses the same
// devices the queues use.  It is *not* cached: the engine calls it once per
// process at startup, and `test_dev_resolve` guards the per-queue profile
// lookup separately.
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

namespace si::dev {

inline bool is_level_zero_device(const sycl::device & d) {
    try {
        return d.get_backend() == sycl::backend::ext_oneapi_level_zero;
    } catch (...) {
        return false;
    }
}

// All GPUs the process can see.  If any of them is a Level Zero device, only
// those are returned; otherwise the full SYCL list is returned unchanged.
inline std::vector<sycl::device> gpu_devices() {
    std::vector<sycl::device> all;
    try {
        all = sycl::device::get_devices(sycl::info::device_type::gpu);
    } catch (...) {
        return {};
    }
    std::vector<sycl::device> l0;
    l0.reserve(all.size());
    for (const sycl::device & d : all) {
        if (is_level_zero_device(d)) {
            l0.push_back(d);
        }
    }
    return l0.empty() ? all : l0;
}

// The preferred GPU.  Throws like `gpu_selector_v` when there is no GPU at all;
// callers that must survive that (the ONNX AUTO path) use `gpu_devices()`.
inline sycl::device gpu_device() {
    std::vector<sycl::device> v = gpu_devices();
    if (!v.empty()) {
        return v.front();
    }
    return sycl::device(sycl::gpu_selector_v);
}

} // namespace si::dev
