#pragma once

#include <sycl/sycl.hpp>

namespace si {
namespace onnx {

// A oneDNN f32 convolution, for the calls the SYCL kernel is slow on.
//
// The weights are the runtime's own layout: (C, kh, kw, M) with group 1.
// They are reordered once into the layout oneDNN's primitive picked and the
// reordered buffer is cached, so the stream that reaches the card is the
// primitive's own (a strided reorder at every execute would cost more than
// the convolution).
//
// Returns false when the call is not one oneDNN should take (a grouped or
// depthwise convolution, a diagnostic probe, a primitive that failed to
// build); the caller then runs the SYCL kernel, which stays the fallback for
// every shape.
// A per-executor switch: the primitive cache is keyed by the input shape,
// and a model whose input size changes with every request (the OCR detector,
// the table UNET) would otherwise cache a primitive and its JIT code per
// shape - several GB over a document.  Fixed-shape models (the layout
// detector) keep oneDNN; the variable ones run the SYCL kernel.
void conv_dnnl_set_enabled(bool on);
bool conv_dnnl_enabled();
// The convolution primitive's policy, from `PF_ONNX_CONV_DNNL`:
//   0 - never: every convolution runs the SYCL kernel.
//   1 - the default: build a primitive from the *second* sighting of a
//       (weight, N, geometry) key, and only for N >= 8.  The fixed-shape
//       models (the layout detector's 800x800 batches) amortise it; a
//       variable-shape model keeps the SYCL kernel.
//   2 - build on the first sighting, at every batch size.
// One function is the only reader of the variable, so the call-site gate and
// the launcher cannot disagree about it (`ops_exec.cpp` used to default the
// gate on while the launcher defaulted the policy to 0, which silently kept
// every non-1x1 convolution on the SYCL kernel whatever `"dnnl": true` said).
int conv_dnnl_policy();
// The device memory oneDNN holds for the weight reorders and the shared
// scratchpad (it is allocated outside the executor's pool).
void conv_dnnl_mem(size_t * weights, size_t * scratch);

bool conv_dnnl_launch(sycl::queue & q, const float * x, const float * w, float * y,
                      int N, int C, int H, int W, int M, int Ho, int Wo, int kh, int kw,
                      int sh, int sw, int ph, int pw, int dh, int dw);

// A batch-1 f32 matmul (M, K) x (K, N), both operands in their plain layout.
// The layout model's decoder is 119 MatMuls and 36 ms of the forward; the
// SYCL kernel is one work-item per output element with a scalar K-loop.
bool matmul_dnnl_launch(sycl::queue & q, const float * a, const float * b, float * y, int64_t M,
                        int64_t K, int64_t N);

// A 1x1, stride-1, group-1 convolution as a batched matmul.
//
// The recogniser, the detector, the layout detector and the table models are
// all full of 1x1 convolutions (61 of the detector's 83), and the direct
// convolution evaluates them one output pixel at a time: 183 of the
// detector's 431 ms.  A 1x1 convolution is `W^T (M,C) x X (C,HW)` per image,
// so the weights are transposed once (cached by pointer) and each image is
// an (M,K)x(K,N) matmul on oneDNN.
//
// `w` is the runtime's (C,1,1,M) weight; `x` and `y` are NCHW.
bool conv1x1_matmul_launch(sycl::queue & q, const float * w, const float * x, float * y, int N,
                           int M, int C, int HW);

} // namespace onnx
} // namespace si
