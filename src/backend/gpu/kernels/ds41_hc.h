// ---------------------------------------------------------------------------
// DeepSeek-V4.1-Flash hyper-connection (mHC) kernels.
//
// One token's full mixer, matching `engine_ds41.cpp`'s `hc_mixes` / `hc_pre` /
// `hc_post` exactly (same order, same eps placement):
//
//   flat   = h.view(hc*d)                       row-major (stream, element)
//   r      = rsqrt(mean(flat^2) + 1e-20)
//   mixes  = (flat @ fn^T) * r                  fn [nm][hc*d], nm=(2+hc)*hc
//   pre[s] = sigmoid(mixes[s]     * scale[0] + base[s])     + eps
//   post[s]= 2*sigmoid(mixes[hc+s] * scale[1] + base[hc+s])
//   C[j][k]= softmax_k(mixes[2hc+j*hc+k] * scale[2] + base[2hc+j*hc+k]) + eps
//   C      = 1 column normalisation, then (iters-1) x (row, column), every
//            denominator + eps
//   hc_pre : y[e]    = sum_s mix[s] * h[s*d+e]
//   hc_post: out[k][e] = post[k]*y[e] + sum_j C[j][k]*res[j][e]
//
// The dot products are split-K: kDs41HcMixSplit workgroups per mix row write
// partial (sumsq, dot) pairs, and a second kernel folds them and runs the
// Sinkhorn pass.  `mixes` is the [nm] output; `partials` is caller-provided
// device scratch of `ds41_hc_mix_partials(hc)` floats.
// ---------------------------------------------------------------------------
#pragma once

#include <sycl/sycl.hpp>

namespace si {

constexpr int kDs41HcMixSplit = 8;

// Required `partials` length for a given hc: nm * 2 * kDs41HcMixSplit.
inline int ds41_hc_mix_partials(int hc) { return (2 + hc) * hc * 2 * kDs41HcMixSplit; }

// Full per-token mixer.  All pointers are device USM.
//   flat     [hc*d]      (device)
//   fn       [nm][hc*d]  (device)
//   scale    [3]         (device)
//   base     [nm]        (device)
//   mixes    [nm]        (device output)
//   partials [nm*2*kDs41HcMixSplit] (device scratch)
//   pre      [hc], post [hc], comb [hc*hc] (device outputs, comb[j*hc+k])
void ds41_hc_mixes_launch(sycl::queue & q, const float * flat, const float * fn, const float * scale,
                          const float * base, float * mixes, float * partials, float * pre, float * post,
                          float * comb, int hc, int d, int iters, float eps);

// y[e] = sum_s mix[s] * h[s*d+e]          (one token)
void ds41_hc_pre_launch(sycl::queue & q, const float * h, const float * mix, float * y, int hc, int d);

// out[k][e] = post[k]*y[e] + sum_j comb[j*hc+k]*res[j][e]   (one token)
void ds41_hc_post_launch(sycl::queue & q, const float * y, const float * res, const float * post,
                         const float * comb, float * out, int hc, int d);

} // namespace si
