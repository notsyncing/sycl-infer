#include "transform.h"
#include "transform_util.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace si {
namespace onnx {
namespace transforms {

bool matmul_bias(graph_build & g) {
    bool changed = false;
    for (size_t i = 0; i < g.nodes.size() && g.fuse; ++i) {
        if (g.dead[i] || g.nodes[i].op_type != "MatMul") continue;
        const onnxpb::node_proto & n = g.nodes[i];
        if (n.output.empty() || n.input.size() < 2 || g.graph_outputs.count(n.output[0])) continue;
        auto uit = g.users.find(n.output[0]);
        if (uit == g.users.end() || uit->second.size() != 1) continue;
        const int ai = uit->second[0];
        if (g.dead[ai]) continue;
        const onnxpb::node_proto & an = g.nodes[ai];
        if (an.op_type != "Add" || an.output.empty() || g.graph_outputs.count(an.output[0])) continue;
        // The bias has to be a constant 1-D vector of the MatMul's N.  The
        // weight is a constant for every such call in the layout model, and
        // its last dimension is the output width.
        auto bit = g.slot.find(n.input[1]);
        if (bit == g.slot.end() || !g.values[bit->second].is_init) continue;
        const tensor & w = g.values[bit->second].t;
        if (w.shape.size() < 2) continue;
        const int64_t N = w.shape.back();
        std::string bias_name;
        int nconst = 0;
        for (const std::string & in : an.input) {
            if (in.empty()) continue;
            auto it = g.slot.find(in);
            if (it == g.slot.end() || !g.values[it->second].is_init) continue;
            const tensor & b = g.values[it->second].t;
            if (b.dtype != onnxpb::FLOAT || b.count != (size_t)N) continue;
            bias_name = in;
            ++nconst;
        }
        if (nconst != 1) continue;
        // If the Add already carries a fused activation from the pass above,
        // move it onto the MatMul; if not, look for a single activation
        // g.consumer here.  The result is one epilogue pass (bias + act) after
        // the matmul, not three operators.
        std::string act;
        std::vector<float> aps;
        auto eait = g.ew_act.find((size_t)ai);
        if (eait != g.ew_act.end()) {
            act = eait->second;
            auto apit = g.ew_act_params.find((size_t)ai);
            if (apit != g.ew_act_params.end()) aps = apit->second;
            g.ew_act.erase(eait);
            g.ew_act_params.erase((size_t)ai);
        } else if (!an.output.empty() && !g.graph_outputs.count(an.output[0])) {
            auto aut = g.users.find(an.output[0]);
            if (aut != g.users.end() && aut->second.size() == 1) {
                const int acti = aut->second[0];
                const onnxpb::node_proto & actn = g.nodes[(size_t)acti];
                bool is_act = false;
                for (const char * a : kActivations) {
                    if (actn.op_type == a) is_act = true;
                }
                if (is_act && !actn.output.empty() && !g.graph_outputs.count(actn.output[0])) {
                    std::vector<float> p2;
                    if (parse_activation(g, actn, p2)) {
                        act = actn.op_type;
                        aps = p2;
                        g.dead[(size_t)acti] = true;
                        g.alias[actn.output[0]] = n.output[0];
                    }
                }
            }
        }
        if (!act.empty()) {
            g.ew_act[i] = act;
            g.ew_act_params[i] = aps;
        }
        g.dead[ai] = true;
        g.alias[an.output[0]] = n.output[0];
        g.nodes[i].input.push_back(bias_name);
        changed = true;
        if (getenv("SI_ONNX_TRACE")) {
            fprintf(stderr, "[onnx] the fusion: %s <- %s (bias %s)\n", n.name.c_str(),
                    an.name.c_str(), bias_name.c_str());
        }
    }


    return changed;
}

} // namespace transforms
} // namespace onnx
} // namespace si
