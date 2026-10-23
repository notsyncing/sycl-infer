#pragma once
// The hard sigmoid's and the hard swish's parameters, in the linear form the
// convolution/elementwise epilogue computes: clamp(a * x + b, 0, 1).
//
// These are ONNX semantics rather than a card's tuning, so they live in
// common/ and are shared by the graph transforms (which fold the activation)
// and the executor (which applies it).  The definition did not change with the
// opset: the hard sigmoid is max(0, min(1, alpha*x + beta)), and ONNXRuntime's
// is the same at opset 11, 12 and 14 (measured with a one-node model).  The
// hard swish has no attributes at all - ONNXRuntime rejects an alpha one - so
// its linear part is fixed to x/6 + 0.5.

#include <cstdint>

namespace si {
namespace onnx {

inline void hardsigmoid_params(int64_t /*opset*/, float alpha, float beta, float * out) {
    out[0] = alpha;
    out[1] = beta;
}

inline void hardswish_params(float * out) {
    out[0] = 1.0f / 6.0f;
    out[1] = 0.5f;
}

} // namespace onnx
} // namespace si
