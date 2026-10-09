#pragma once

// YaRN context extension (Peng et al. 2023, "YaRN: Efficient Context Window
// Extension of Large Language Models"), NTK-by-parts variant, as implemented in
// llama.cpp's `llama_rope_yarn` and in HF's `YaRNScaledRotaryEmbedding`.
//
// The base RoPE angle for pair `i` of a `dim`-wide rotation is
//     theta_i(pos) = pos * base^(-2i/dim)
// (see rope.h).  YaRN replaces the per-pair *frequency* base^(-2i/dim) with a
// per-pair table that is
//   - untouched for the high-frequency pairs (they never see a position long
//     enough for the period to matter), and
//   - divided by `factor` for the low-frequency pairs, whose period is longer
//     than the trained context and which therefore alias first,
// with a smooth ramp between the two regions.  Interpolating only the tail is
// what distinguishes YaRN from plain position interpolation (which stretches
// every pair and costs perplexity even at short context) and from dynamic NTK
// (which stretches every pair and shifts the high-frequency content).
//
// The interpolation also *compresses* the attention logits (the distances the
// attention sees shrink), so cos and sin are multiplied by `mscale` >= 1 to
// compensate - the attention-temperature term of the paper.
//
// Nothing here needs the device: the table is n_rot/2 floats, built once on the
// host and uploaded, and the kernels read it instead of recomputing exp2.  That
// is also why the default (YaRN off) passes a null pointer and keeps the
// existing expression - see qk_norm_rope.cpp.
//
// Note on quality: YaRN is a *scaling* method.  The paper's gains come from a
// small amount of long-context fine-tuning on top; used zero-shot it extends the
// context but degrades quality relative to the trained window, so this flag is
// opt-in and the engine prints the effective factor when it is on.

#include <algorithm>
#include <cmath>
#include <vector>

namespace si {

struct yarn_params {
    bool on = false;
    int orig_ctx = 0;         // trained context length (from the GGUF)
    float factor = 1.0f;      // target/orig; 1.0 means "no extension"
    float beta_fast = 32.0f;  // high-frequency cutoff, in rotations
    float beta_slow = 1.0f;   // low-frequency cutoff, in rotations
    float attn_factor = 0.0f; // 0 = derive 1 + 0.1*log10(factor)
    float mscale = 1.0f;      // cos/sin multiplier, derived by yarn_build()
};

// The pair index at which a rotation of `num_rotations` turns occurs, i.e. the
// pair index whose wavelength is 2*pi*num_rotations tokens long.  Identical to
// HF's find_correction_dim().
inline float yarn_correction_dim(float num_rotations, float dim, float base, float max_pos) {
    return (dim * std::log(max_pos / (num_rotations * 2.0f * 3.14159265358979f))) / (2.0f * std::log(base));
}

// HF's linear_ramp_mask(): 0 below `lo`, 1 above `hi`, linear in between.
inline float yarn_ramp(float lo, float hi, float v) {
    if (hi <= lo) {
        hi = lo + 1.0e-3f;
    }
    return std::min(1.0f, std::max(0.0f, (v - lo) / (hi - lo)));
}

// Fills `inv` with n_rot/2 per-pair inverse frequencies and returns the
// attention scale to apply to cos/sin.  `p.mscale` is written; `inv` is left as
// the exact base^(-2i/n_rot) sequence when factor == 1 (or the table is off), so
// the caller can use the result unconditionally once the table exists.
//
// Two details are load-bearing for bit-exactness with the plain path, and both
// are about which of two algebraically equal blends is used:
//   - the seed uses n_rot itself, not 2*n_pairs, so it is the *same* expression
//     rope.h evaluates (`exp2(-2i/n_rot * log2_base)`);
//   - the two blends are written so that each anchor is exact.  The ramp is
//     `extrap` outside [lo, hi] and `interp` beyond hi, so a result of
//     `extrap + (blended - extrap) * m` is exact in the head (blended *is*
//     extrap, so the difference is a hard zero) but off by an ulp of *extrap* in
//     the tail - which is 1/factor of the value there, i.e. a factor-32 stretch
//     would carry ~2e-6 of relative error instead of 0.  Writing it as
//     `blended + (extrap - blended) * (1 - m)` fixes the tail and keeps the head,
//     because there `blended - extrap` is again exactly zero.  Only the ~16 pairs
//     inside the ramp carry rounding at all.
inline std::vector<float> yarn_build(const yarn_params & p, int n_rot, float base, float * mscale_out) {
    const int n_pairs = n_rot / 2;
    std::vector<float> inv((size_t)(n_pairs > 0 ? n_pairs : 0));
    if (n_pairs <= 0) {
        if (mscale_out) {
            *mscale_out = 1.0f;
        }
        return inv;
    }
    const float dim = (float)n_rot;
    const float log2_base = std::log2(base);
    for (int i = 0; i < n_pairs; i++) {
        inv[(size_t)i] = std::exp2(-2.0f * i / dim * log2_base);
    }

    float mscale = 1.0f;
    if (p.on && p.factor > 1.0f && p.orig_ctx > 0) {
        float lo = yarn_correction_dim(p.beta_fast, dim, base, (float)p.orig_ctx);
        float hi = yarn_correction_dim(p.beta_slow, dim, base, (float)p.orig_ctx);
        if (lo > hi) {
            std::swap(lo, hi);
        }
        // spread the two cutoffs apart by one pair, as HF does, so the ramp is
        // not degenerate when they collide
        const float delta = (hi - lo) / dim;
        lo += delta;
        hi += delta;
        for (int i = 0; i < n_pairs; i++) {
            const float theta_extrap = inv[(size_t)i];
            const float theta_interp = theta_extrap / p.factor;
            const float mask_low = 1.0f - yarn_ramp(lo, hi, (float)i);
            const float mask_high = yarn_ramp(0.0f, hi, (float)i);
            const float blended = theta_interp * (1.0f - mask_low) + theta_extrap * mask_low;
            inv[(size_t)i] = blended + (theta_extrap - blended) * (1.0f - mask_high);
        }
        mscale = p.attn_factor > 0.0f ? p.attn_factor : 1.0f + 0.1f * std::log10(p.factor);
        mscale = std::max(1.0f, mscale);
    }
    if (mscale_out) {
        *mscale_out = mscale;
    }
    return inv;
}

} // namespace si