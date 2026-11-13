#pragma once

namespace si {

// DeepSeek-V4.1-Flash's GPU forward state, kept behind an opaque pointer: the
// definition lives in arch/deepseek41/forward_gpu.cpp and is never needed by a
// translation unit that only composes the engine.
struct ds41_gpu_state;

struct deepseek41_state {
    // The GPU implementation of the model-owned forward
    // (arch/deepseek41/forward_gpu.cpp); null on a CPU run, for a non-ds41
    // model, or when PF_DS41_GPU=0.
    ds41_gpu_state * gpu = nullptr;
    // True when setup_ds41_dense_store() registered the per-device oneDNN
    // stores the forward resolves through dnnl_for(dev) - k5 for Q5_K (exact
    // native values, 0.75 B/w) and, behind PF_DS41_DENSE_Q8, the ordinary int8
    // conversion for Q8_0.  A converted tensor then skips its raw device
    // upload (see upload_device_weights) and wkey() returns the host pointer,
    // which is the store's key - exactly what the forward passes to gemv1.
    bool dense_store = false;
};

} // namespace si
