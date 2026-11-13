// The model architecture registry hub.  This is the ONE model-side translation
// unit where architecture names are allowed: it owns the explicit loader table
// and includes each architecture's `load.h`.  The table is explicit rather than
// self-registered because the core is a static library, so a registration from
// an object no other symbol references would be dropped at link time.
#include "arch/registry.h"

#include <cstddef>
#include <string>

#include "arch/deepseek41/load.h"
#include "arch/qwen2vl/load.h"
#include "arch/qwen35/load.h"
#include "arch/qwen35moe/load.h"
#include "arch/qwen4exp/load.h"

namespace si {
namespace arch {

// Supported architectures.  Add one loader per model type; the loaders live in
// their own translation units (arch/<arch>/load.cpp).  Every row carries its
// declarative profile, and model::load asserts the profile's key is the row's
// exact general.architecture value: a row without a matching profile is a load
// error, not a silent fallback to another architecture's policy.
static const loader kLoaders[] = {
    {"qwen35", &load_qwen35, &qwen35_profile()},
    {"qwen35moe", &load_qwen35moe, &qwen35moe_profile()},
    {"qwen4exp", &load_qwen4exp, &qwen4exp_profile()},
    {"qwen2vl", &load_qwen2vl, &qwen2vl_profile()},
    {"deepseek41", &load_deepseek41, &deepseek41_profile()},
};

const loader * find(const std::string & name) {
    for (const loader & l : kLoaders) {
        if (name == l.name) {
            return &l;
        }
    }
    return nullptr;
}

const loader * registry(size_t & n) {
    n = sizeof(kLoaders) / sizeof(kLoaders[0]);
    return kLoaders;
}

} // namespace arch
} // namespace si
