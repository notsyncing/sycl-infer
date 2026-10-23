#pragma once
// The ONNX runtime graph: the parsed model, optimised into an executable
// form.
//
// The pipeline from a file to a runnable graph is:
//
//   parse (proto.h)  ->  constant folding  ->  operator fusion  ->  shape
//   inference  ->  the execution plan
//
// The two middle steps are where the performance comes from, and they are
// the reason a hand-written runtime can be competitive:
//
//   * Constant folding evaluates the sub-graphs whose inputs are all
//     constants.  A torch-exported model is full of them: the table models
//     carry 484 Constant nodes, and the layout model's shape arithmetic
//     (78 Shape, 165 Reshape) collapses into constants.
//   * Operator fusion removes the memory traffic between the operators.  The
//     text detector is 83 convolutions followed by 88 Mul/Add pairs (the
//     batch norm that was folded into the weights) and 20 activations; fused,
//     that is 83 kernels instead of 200, and each convolution's epilogue
//     does the scale, the bias and the activation in registers.
//
// The fusion rules are deliberately narrow and explicit: each one is a
// pattern with a name, and each one is checked against the operator's
// semantics (a BatchNorm folds into a Conv only when the Conv has no
// padding that the norm's spatial extent would disagree with, and so on).

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "proto.h"

namespace si {
namespace onnx {

// A tensor in the runtime: a shape, a type and (at run time) a device
// pointer.
// The runtime's element size.
//
// A boolean is a *float* 0/1 in this runtime: the device holds float32, and
// a one-byte boolean on the device is read as three bytes of whatever is
// next.  The one-byte boolean is the *wire's* format, and the boundary
// converts it (see executor::run).
//
// The integer values (the dimensions, the axes, the indices) live on the
// host, in their own width.
inline size_t runtime_elem_size(int32_t dtype) {
    if (dtype == onnxpb::INT64) return 8;
    if (dtype == onnxpb::INT8 || dtype == onnxpb::UINT8) return 1;
    return 4;  // a float, a boolean, an int32
}

struct tensor {
    std::string name;
    std::vector<int64_t> shape;
    int32_t dtype = onnxpb::FLOAT;
    // The number of elements.
    size_t count = 0;
    // The device allocation (USM).  Null until the plan allocates it.
    void * dev = nullptr;
    // For a host tensor (an initializer before it is uploaded, or a value
    // that the host computes).
    std::vector<uint8_t> host;
    bool on_host = false;

    size_t bytes() const { return count * runtime_elem_size(dtype); }
    int64_t dim(size_t i) const { return i < shape.size() ? shape[i] : 1; }
    size_t rank() const { return shape.size(); }
    bool empty() const { return count == 0; }
};

// The fused kinds the executor knows.  A kind is either an ONNX operator
// (executed as it is) or a fused pattern (the optimizer rewrote it).
enum class op_kind {
    // The ONNX operators, executed one by one.
    generic,
    // A convolution with the batch norm folded in and an optional
    // activation in the epilogue.
    conv,
    // An elementwise binary/unary operator.  It may run in place: the
    // executor reuses an input buffer that dies at this operator instead of
    // allocating a new one (the mask head's Add chains are ten 48 MB
    // tensors, and a chain of adds needs one buffer, not ten).
    elementwise,
    // A value-preserving metadata change (Reshape/Flatten/Squeeze/
    // Unsqueeze/Identity).  The executor borrows the input's buffer when it
    // is the input's last use, instead of copying the whole tensor.  The
    // output keeps its own shape, so the tensor metadata is still correct.
    alias,
    // A copy with a permutation of the axes (Transpose).
    permute,
};

// One executable operation.
struct exec_op {
    op_kind kind = op_kind::generic;
    // The ONNX node it came from (the attributes are read from it).
    const onnxpb::node_proto * node = nullptr;
    // The tensor slots: the inputs and the outputs.
    std::vector<int> in;
    std::vector<int> out;
    // The per-channel scale and bias folded out of a BatchNorm (and the
    // Mul/Add pair that expresses one).
    std::vector<float> scale;
    std::vector<float> bias;
    // The activation in the convolution's epilogue: "none", "relu",
    // "sigmoid", "hardsigmoid", "hardswish", "gelu", "silu", "tanh".
    std::string activation;
    // The weight is already transposed (the builder did it).
    bool weight_transposed = false;
    // The activation's parameters (the hardsigmoid/hardswish bounds).
    std::vector<float> act_params;
    // The operator's own parameters, copied out of the attributes so the
    // executor does not have to walk the attribute list.
    std::vector<int64_t> ints;
    std::vector<float> floats;
    std::string str;
    // An If operator: its two branches are inlined into the op list (the
    // exporter's sub-graphs are not self-contained), and only the selected
    // branch runs.  These are the op ranges of the two branches.
    int if_then_begin = -1, if_then_end = -1;
    int if_else_begin = -1, if_else_end = -1;
    // The operator is inside an If's branch: only one of the two branches
    // runs, so a value it produces is not a constant (the other branch would
    // produce a different one) and it must not be folded.
    bool in_if_branch = false;
};

// The executable graph.
struct exec_graph {
    // The nodes the ops refer to.  The graph owns them: the operators
    // point at their node's attributes, and a node that outlived its owner
    // would be a dangling pointer.
    std::vector<onnxpb::node_proto> nodes;
    std::vector<exec_op> ops;
    // The tensors, indexed by the slots the ops refer to.
    std::vector<tensor> tensors;
    // The names of the graph's inputs and outputs, and the slot of each.
    std::vector<int> input_slots;
    std::vector<int> output_slots;
    // For each output: does a node produce it?  A Loop's body may declare an
    // output that no node produces, and ONNXRuntime returns the body's
    // *input* of the same name unchanged - so the executor has to know which
    // outputs are pass-throughs.
    std::vector<bool> output_produced;
    // The metadata (the "character" list).
    std::map<std::string, std::string> metadata;
    int64_t opset = 0;
    // The values this graph reads from an *enclosing* graph: the local slot
    // and the enclosing graph's slot.
    //
    // ONNX lets a subgraph read the values of the graph that holds it (the
    // "outer scope"), and the formula recogniser's Loop body does: it reads
    // a transpose that the enclosing graph computes.  A subgraph that is
    // built on its own cannot resolve such a name, so the builder records
    // the reference and the executor copies the value in.
    std::vector<std::pair<int, int>> outer_refs;
};

// Build the executable graph from a model file.  Throws std::runtime_error
// with a message that says which operator it could not handle.
exec_graph build_graph(const std::string & path, bool verbose = false);
// The same, from a graph: a Loop's body is a graph, and it is built the same
// way as the model.
//
// `scope` is the enclosing graph, for the values the subgraph reads from it
// (ONNX's outer scope).  A null scope means the subgraph is self-contained.
exec_graph build_graph_from_graph(const onnxpb::graph_proto & graph, int64_t opset,
                                  const std::string & path, bool verbose = false,
                                  const exec_graph * scope = nullptr);

// The convolution weights, transposed so that the output channels of one
// (c, kh, kw) are contiguous: the kernel reads them as a run.
//
// The transposition happens once per model, but *when* depends on where the
// weight comes from: a weight that is an initializer is transposed while the
// graph is built, and a weight that a Constant node produces is transposed
// after the constant sub-graphs have been evaluated.
tensor transpose_conv_weight(const tensor & w, int64_t group);
tensor transpose_conv_transpose_weight(const tensor & w, int64_t group);

} // namespace onnx
} // namespace si
