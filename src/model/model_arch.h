#pragma once
// ---------------------------------------------------------------------------
// Model architecture dispatch.
//
// model::load() reads `general.architecture` from the GGUF and hands the parsed
// file to the matching loader from the registry below.  Each supported
// architecture lives in its own translation unit and fills hparams + the layer
// bindings (see qwen35.cpp).
// ---------------------------------------------------------------------------
#include <cstddef>
#include <string>

#include "model.h"
#include "model_profile.h"

namespace si {
namespace arch {

struct loader {
    const char * name;              // general.architecture value
    void (*load)(model & m);        // fill hparams + bind this architecture's tensors
    const model_profile * prof;     // declarative profile; key must equal `name`
};

// registry lookup (defined in model.cpp); nullptr when the architecture is unknown
const loader * find(const std::string & name);

// the whole registry (n rows).  PF_MODEL_INFO and test_model_profile read it so
// a new architecture cannot exist without a profile, and there is no second
// table to drift.
const loader * registry(size_t & n);

// per-architecture loaders
void load_qwen35(model & m);
void load_qwen35moe(model & m);
void load_qwen4exp(model & m);

// shared tensor-binding helpers (defined in model.cpp)
wt bind_tensor(const gguf_file & f, const std::string & name, uint32_t expect_type = 0xFFFFFFFF);
const float * bind_f32(const gguf_file & f, const std::string & name);

} // namespace arch
} // namespace si
