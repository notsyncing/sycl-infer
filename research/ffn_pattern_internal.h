// Internal representation of an FFN pattern: top-K neuron ids per text/layer,
// plus (optionally) their z values.  Kept opaque to C callers.
#pragma once

#include <cstddef>
#include <vector>

struct syi_ffn_pattern {
    size_t n_texts = 0;
    size_t n_layers = 0;
    size_t top_k = 0;
    size_t dim = 0; // width of the vector the top-K was taken from (n_ff or 2*n_ff)
    // [text][layer][top_k] flat
    std::vector<int32_t> ids;
    std::vector<float> z;
    bool has_z = false;
};
