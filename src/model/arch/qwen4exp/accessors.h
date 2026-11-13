#pragma once
// Accessors for qwen4exp's pimpl extras (`layer_t::extras` /
// `model::extras`).  This header includes model.h and the tensor structs, so
// q4 code includes it instead of reaching into the generic slots.
#include <cassert>

#include "model.h"
#include "tensors.h"

namespace si {

inline qwen4exp_layer &q4_layer(layer_t &L) {
    assert(L.extras && L.extras->k == arch_layer::kind::qwen4exp);
    return static_cast<qwen4exp_layer &>(*L.extras);
}

inline const qwen4exp_layer &q4_layer(const layer_t &L) {
    assert(L.extras && L.extras->k == arch_layer::kind::qwen4exp);
    return static_cast<const qwen4exp_layer &>(*L.extras);
}

inline qwen4exp_model_extras &q4_model(model &m) {
    assert(m.extras && m.extras->k == arch_model_extras::kind::qwen4exp);
    return static_cast<qwen4exp_model_extras &>(*m.extras);
}

inline const qwen4exp_model_extras &q4_model(const model &m) {
    assert(m.extras && m.extras->k == arch_model_extras::kind::qwen4exp);
    return static_cast<const qwen4exp_model_extras &>(*m.extras);
}

} // namespace si
