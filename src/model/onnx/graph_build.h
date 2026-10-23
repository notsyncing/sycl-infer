#pragma once
// The graph in its build/lowering phase.
//
// The parser fills in the nodes, the named values and the constant analysis;
// then the transform pipeline (transforms/*.cpp) rewrites the nodes.  The
// executor's op list is lowered from this state *after* the transforms, so a
// pass never has to know about command graphs, queues or device pointers.
//
// Keeping this state in a header is deliberate: each transform is a small,
// independently testable function over this graph, not another branch in the
// builder.

#include "graph.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace si {
namespace onnx {

struct build_value {
    tensor t;
    bool is_init = false;
};

struct graph_build {
    std::vector<onnxpb::node_proto> nodes;
    std::vector<build_value> values;
    std::map<std::string, int> slot;
    std::map<std::string, bool> is_const;
    std::vector<bool> node_const;

    // The consumers: the first one (for the old conv chain) and all of them
    // (the passes that need to prove a value has exactly one consumer).
    std::map<std::string, int> consumer;
    std::map<std::string, std::vector<int>> users;
    std::map<std::string, int> producer_of;
    std::set<std::string> graph_outputs;

    // A dead node is skipped when the executor is lowered.  `alias` renames a
    // value to the producing node that survives the fusion.
    std::vector<bool> dead;
    std::map<std::string, std::string> alias;

    // The epilogues the lowering attaches to a convolution or an
    // elementwise operator.
    std::map<size_t, std::vector<float>> conv_scale;
    std::map<size_t, std::vector<float>> conv_bias;
    std::map<size_t, std::string> conv_act;
    std::map<size_t, std::vector<float>> conv_act_params;
    std::map<size_t, std::string> ew_act;
    std::map<size_t, std::vector<float>> ew_act_params;

    // Idempotence for the fixed-point loop: a convolution is examined once,
    // even if another pass changes its consumers later.
    std::set<size_t> fused_convs;

    int64_t opset = 0;
    bool fuse = true;
    bool verbose = false;
};


} // namespace onnx
} // namespace si
