#pragma once
// Shared, host-only helpers for the graph transforms.

#include "graph_build.h"

#include <string>
#include <vector>

namespace si {
namespace onnx {
namespace transforms {

// The activations that can ride along in a convolution/elementwise epilogue.
extern const char * const kActivations[8];

// Parse an activation node into the linear/clip parameters the epilogue
// wants.  Returns false when the activation is not one the epilogue can
// represent (and when SI_ONNX_NOFUSE_ACT is set).
bool parse_activation(const graph_build & g, const onnxpb::node_proto & n,
                      std::vector<float> & params);

// Whether a value is a broadcast scalar with a known value.  This is what
// proves the mask head's "+ 0" and its 1e8 sentinel without materialising
// the 48 MB tensors.
bool splat(const graph_build & g, const std::string & name, double & value, int depth = 0);

bool reads(const onnxpb::node_proto & n, const std::string & value);

} // namespace transforms
} // namespace onnx
} // namespace si
