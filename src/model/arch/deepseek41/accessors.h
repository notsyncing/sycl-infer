#pragma once
// Accessors for deepseek41's pimpl extras (`layer_t::extras` /
// `model::extras`).  This header includes model.h and the tensor structs, so
// ds41 code includes it instead of reaching into the generic slots.
#include <cassert>

#include "model.h"
#include "tensors.h"

namespace si {

inline ds41_layer &ds41_layer_of(layer_t &L) {
    assert(L.extras && L.extras->k == arch_layer::kind::deepseek41);
    return static_cast<ds41_layer &>(*L.extras);
}

inline const ds41_layer &ds41_layer_of(const layer_t &L) {
    assert(L.extras && L.extras->k == arch_layer::kind::deepseek41);
    return static_cast<const ds41_layer &>(*L.extras);
}

inline ds41_model_extras &ds41_model(model &m) {
    assert(m.extras && m.extras->k == arch_model_extras::kind::deepseek41);
    return static_cast<ds41_model_extras &>(*m.extras);
}

inline const ds41_model_extras &ds41_model(const model &m) {
    assert(m.extras && m.extras->k == arch_model_extras::kind::deepseek41);
    return static_cast<const ds41_model_extras &>(*m.extras);
}

} // namespace si
