#pragma once
// ---------------------------------------------------------------------------
// Model architecture dispatch: the public registry seam.
//
// model::load() reads `general.architecture` from the GGUF and hands the parsed
// file to the matching loader from this registry.  Every architecture lives in
// its own `arch/<arch>/` directory (loader + declarative profile); the table
// itself is in the internal hub `arch/registry.cpp`, which is the only model-side
// file that names an architecture.  This header names none.
// ---------------------------------------------------------------------------
#include <cstddef>
#include <string>

#include "model_profile.h"

namespace si {
struct model;

namespace arch {

struct loader {
    const char * name;              // general.architecture value
    void (*load)(model & m);        // fill hparams + bind this architecture's tensors
    const model_profile * prof;     // declarative profile; key must equal `name`
};

// registry lookup (defined in arch/registry.cpp); nullptr when the architecture
// is unknown.
const loader * find(const std::string & name);

// the whole registry (n rows).  PF_MODEL_INFO and test_model_profile read it so
// a new architecture cannot exist without a profile, and there is no second
// table to drift.
const loader * registry(size_t & n);

} // namespace arch
} // namespace si
