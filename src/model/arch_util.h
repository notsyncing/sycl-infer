#pragma once
// Shared loader helpers for the per-architecture loaders.  Header-only and
// `inline`: the binding rules and error messages are identical for every
// architecture, and duplicating them per loader is how the copies drift.
#include <cstdint>
#include <stdexcept>
#include <string>

#include "gguf.h"
#include "model_types.h"

namespace si {
namespace arch {

// Ordinary tensor binding (defined in model.cpp).  A missing tensor, or a
// tensor whose type differs from `expect_type` (0xFFFFFFFF = any), is a load
// error.
wt bind_tensor(const gguf_file & f, const std::string & name, uint32_t expect_type = 0xFFFFFFFF);
const float * bind_f32(const gguf_file & f, const std::string & name);

// A 3-D routed-expert tensor [K][N][E].  `wt` cannot express it: its N is the
// product of dims[1:], which for an expert tensor is N*E, and the stride
// between experts (nb[2]) is not derivable from the element count once the
// blocks are packed.
inline moe_wt bind_moe(const gguf_file & f, const std::string & name) {
    const gguf_tensor_info * ti = f.find(name);
    if (!ti) {
        throw std::runtime_error("missing tensor: " + name);
    }
    if (ti->dims.size() != 3) {
        throw std::runtime_error("expected a 3-D expert tensor: " + name);
    }
    moe_wt t;
    t.data = ti->data;
    t.type = ti->type;
    t.K = (int32_t)ti->dims[0];
    t.N = (int32_t)ti->dims[1];
    t.E = (int32_t)ti->dims[2];
    // ggml stores the tensors row-contiguous (nb[0] = type size, nb[1] =
    // row_size(ne[0]), nb[2] = row_size(ne[0])*ne[1]), so the plane stride is
    // the size of one [K][N] plane.
    t.plane_bytes = ggml_row_bytes(ti->type, ti->dims[0] * ti->dims[1]);
    return t;
}

// A required metadata value; absent is a load error naming the exact key.
inline const gguf_kv * require_meta(const gguf_file & f, const std::string & key) {
    const gguf_kv * v = f.meta(key);
    if (!v) {
        throw std::runtime_error("missing metadata: " + key);
    }
    return v;
}

} // namespace arch
} // namespace si
