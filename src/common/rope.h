#pragma once

#include <cmath>

namespace si {

// The rotation angle for pair `i` of a `dim`-wide head at position `pos`
// (`theta = pos * base^(-2i/dim)`), shared by every caller:
//   - the text model's RoPE (dim = head_dim, or n_rot for a partial rotation),
//   - the audio tower (dim = head_dim),
//   - the 4-section vision RoPE, which restarts the pair count at `dim`=half
//     for every row/col section.
inline float rope_theta(float pos, int i, int dim, float log2_base) {
    return pos * std::exp2(-2.0f * i / (float)dim * log2_base);
}

} // namespace si
