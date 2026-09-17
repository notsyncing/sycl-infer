#pragma once
// ---------------------------------------------------------------------------
// Model architecture dispatch.
//
// model::load() reads `general.architecture` from the GGUF and hands the parsed
// file to the matching loader from the registry below.  Each supported
// architecture lives in its own translation unit and fills hparams + the layer
// bindings (see qwen35.cpp).
// ---------------------------------------------------------------------------
#include <string>

#include "model.h"

namespace si {
namespace arch {

struct loader {
    const char * name;       // general.architecture value
    void (*load)(model & m); // fill hparams + bind this architecture's tensors
};

// registry lookup (defined in model.cpp); nullptr when the architecture is unknown
const loader * find(const std::string & name);

// per-architecture loaders
void load_qwen35(model & m);

// shared tensor-binding helpers (defined in model.cpp)
wt bind_tensor(const gguf_file & f, const std::string & name, uint32_t expect_type = 0xFFFFFFFF);
const float * bind_f32(const gguf_file & f, const std::string & name);

} // namespace arch
} // namespace si
