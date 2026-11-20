// Internal header for the FFN activation capture used by the research C API.
// This is NOT part of the public C ABI; it only declares the engine-side
// helper that src/capi/capi_ffn.cpp calls.  Kept under research/ so the
// capture logic stays isolated from the inference engine's public surface.
#pragma once

#include <cstddef>
#include <vector>

namespace si {

struct engine;

namespace research {

// For each tokenized prompt, prefill one chunk through the engine and copy the
// FFN [gate|up] activation of token `slot` for the requested layer into `out`.
// `layer` < 0 means all layers (rows stacked by layer).  Out layout is
// [n_prompts][n_layers][2*n_ff] floats.  Returns 0 on success, -1 when a prompt
// exceeds one prefill chunk or layer/slot is out of range.  The caller must
// not submit generation requests on the engine while this runs.
int capture_ffn_activations(engine & e, const std::vector<std::vector<int> > & prompts, int layer, int slot,
                            std::vector<float> & out);

} // namespace research
} // namespace si
