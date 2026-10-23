#include "transform.h"
#include "transform_util.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace si {
namespace onnx {
namespace transforms {

bool elementwise_activation(graph_build & g) {
    bool changed = false;
    for (size_t i = 0; i < g.nodes.size() && g.fuse; ++i) {
        if (g.dead[i]) continue;
        const onnxpb::node_proto & n = g.nodes[i];
        if (n.op_type != "Add" && n.op_type != "Sub" && n.op_type != "Mul" &&
            n.op_type != "Div") {
            continue;
        }
        if (n.output.empty() || g.graph_outputs.count(n.output[0])) continue;
        auto uit = g.users.find(n.output[0]);
        if (uit == g.users.end() || uit->second.size() != 1) continue;
        const int ai = uit->second[0];
        if (g.dead[ai]) continue;
        const onnxpb::node_proto & an = g.nodes[ai];
        if (an.output.empty() || g.graph_outputs.count(an.output[0])) continue;
        bool is_act = false;
        for (const char * a : kActivations) {
            if (an.op_type == a) is_act = true;
        }
        if (!is_act) continue;
        std::vector<float> params;
        if (!parse_activation(g, an, params)) continue;
        g.ew_act[i] = an.op_type;
        g.ew_act_params[i] = params;
        changed = true;
        g.dead[ai] = true;
        g.alias[an.output[0]] = n.output[0];
        if (getenv("SI_ONNX_TRACE")) {
            fprintf(stderr, "[onnx] the fusion: %s <- %s (elementwise epilogue %s)\n",
                    n.name.c_str(), an.name.c_str(), an.op_type.c_str());
        }
    }


    return changed;
}

} // namespace transforms
} // namespace onnx
} // namespace si
