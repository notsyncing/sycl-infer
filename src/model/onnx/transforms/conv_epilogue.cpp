#include "transform.h"
#include "transform_util.h"

#include "onnx_activation.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace si {
namespace onnx {
namespace transforms {

bool conv_epilogue(graph_build & g) {
    bool changed = false;
    for (size_t i = 0; i < g.nodes.size() && g.fuse; ++i) {
        if (g.dead[i]) continue;
        const onnxpb::node_proto & n = g.nodes[i];
        if (n.op_type != "Conv" && n.op_type != "ConvTranspose") continue;
        if (n.output.empty()) continue;
        if (!g.fused_convs.insert(i).second) continue;
        std::string cur = n.output[0];
        int guard = 0;
        while (guard++ < 64) {
            auto cit = g.consumer.find(cur);
            if (cit == g.consumer.end()) break;
            const int ci = cit->second;
            const onnxpb::node_proto & cn = g.nodes[ci];
            const std::string & op = cn.op_type;
            bool is_act = false;
            for (const char * a : kActivations) {
                if (op == a) is_act = true;
            }
            if (getenv("SI_ONNX_TRACE") && n.op_type == "Conv") {
                fprintf(stderr, "[onnx] the fusion: %s <- %s (%s)\n", n.name.c_str(),
                        cn.name.c_str(), op.c_str());
            }
            // The fold replaces the folded value's *name* with the
            // convolution's, so a g.consumer that reads both of them would
            // read the same value twice.  (The layout model has
            // y = x * sigmoid(x): the fold turned it into y = x * x, and the
            // decoder's boxes came out wrong.)
            if (reads(cn, cur)) break;
            if (op == "BatchNormalization") {
                // The four constants: the scale, the bias, the mean and the
                // variance.  (The first input is the data, which is not
                // constant - that is the whole point of the folding.)
                std::vector<const build_value *> bn;
                for (size_t k = 1; k < cn.input.size(); ++k) {
                    if (cn.input[k].empty()) continue;
                    auto it = g.slot.find(cn.input[k]);
                    if (it == g.slot.end() || !g.values[it->second].is_init) continue;
                    bn.push_back(&g.values[it->second]);
                }
                if (bn.size() < 4) break;
                const int64_t n_ch = bn[0]->t.count;
                float eps = 1e-5f;
                for (const onnxpb::attribute & a : cn.attribute) {
                    if (a.name == "epsilon") eps = (float)a.f;
                }
                std::vector<float> scale((size_t)n_ch), bias((size_t)n_ch);
                const float * gamma = (const float *)bn[0]->t.host.data();
                const float * beta = (const float *)bn[1]->t.host.data();
                const float * mean = (const float *)bn[2]->t.host.data();
                const float * var = (const float *)bn[3]->t.host.data();
                for (int64_t k = 0; k < n_ch; ++k) {
                    const float inv = 1.0f / std::sqrt(var[k] + eps);
                    scale[(size_t)k] = gamma[k] * inv;
                    bias[(size_t)k] = beta[k] - mean[k] * gamma[k] * inv;
                }
                g.conv_scale[i] = scale;
                g.conv_bias[i] = bias;
                changed = true;
                g.dead[ci] = true;
                if (!cn.output.empty()) g.alias[cn.output[0]] = cur;
                cur = cn.output[0];
                continue;
            }
            if ((op == "Mul" || op == "Add") && g.node_const[ci]) {
                int which = -1;
                for (size_t k = 0; k < cn.input.size(); ++k) {
                    if (cn.input[k].empty()) continue;
                    auto it = g.is_const.find(cn.input[k]);
                    if (it != g.is_const.end() && it->second) {
                        which = (int)k;
                        break;
                    }
                }
                if (which < 0) break;
                const int vs = g.slot[cn.input[which]];
                const build_value & v = g.values[vs];
                if (v.t.host.size() < 4) break;
                const int64_t n_ch = v.t.count;
                std::vector<float> vals((size_t)n_ch);
                std::memcpy(vals.data(), v.t.host.data(), (size_t)n_ch * 4);
                if (op == "Mul") {
                    g.conv_scale[i] = vals;
                } else {
                    g.conv_bias[i] = vals;
                }
                changed = true;
                g.dead[ci] = true;
                if (!cn.output.empty()) g.alias[cn.output[0]] = cur;
                cur = cn.output[0];
                continue;
            }
            if (is_act) {
                // The activation's parameters, in the linear form the
                // epilogue computes: p0 * v + p1.
                //
                // The hard sigmoid is max(0, min(1, alpha*v + beta)) with
                // ONNX's defaults alpha=0.2 and beta=0.5, and the hard
                // swish is x * max(0, min(1, (x + 3) / 6)) - a fixed form,
                // with no attributes at all.
                std::vector<float> params;
                bool foldable = true;
                if (getenv("SI_ONNX_NOFUSE_ACT")) foldable = false;
                if (op == "HardSigmoid") {
                    float alpha = 0.2f, beta = 0.5f;
                    for (const onnxpb::attribute & a : cn.attribute) {
                        if (a.name == "alpha") alpha = a.f;
                        if (a.name == "beta") beta = a.f;
                    }
                    float p[2] = {0.0f, 0.0f};
                    hardsigmoid_params(g.opset, alpha, beta, p);
                    params = {p[0], p[1]};
                } else if (op == "HardSwish") {
                    // The form is fixed: x * max(0, min(1, (x + 3) / 6)).
                    float p[2] = {0.0f, 0.0f};
                    hardswish_params(p);
                    params = {p[0], p[1]};
                } else if (op == "Clip") {
                    // The bounds are inputs, and they have to be constants.
                    float lo = -3.0e38f, hi = 3.0e38f;
                    for (size_t k = 1; k < cn.input.size(); ++k) {
                        if (cn.input[k].empty()) continue;
                        auto it = g.slot.find(cn.input[k]);
                        if (it == g.slot.end() || !g.values[it->second].is_init ||
                            g.values[it->second].t.host.size() < 4) {
                            foldable = false;
                            break;
                        }
                        const float v = *(const float *)g.values[it->second].t.host.data();
                        if (k == 1) lo = v;
                        else hi = v;
                    }
                    params = {lo, hi};
                } else if (op == "Gelu") {
                    // Only the tanh approximation is in the epilogue.
                    std::string approx;
                    for (const onnxpb::attribute & a : cn.attribute) {
                        if (a.name == "approximate") approx = a.s;
                    }
                    if (approx != "tanh") foldable = false;
                }
                if (!foldable) break;
                g.conv_act[i] = op;
                g.conv_act_params[i] = params;
                changed = true;
                g.dead[ci] = true;
                if (!cn.output.empty()) g.alias[cn.output[0]] = cur;
                cur = cn.output[0];
                continue;
            }
            break;
        }
    }


    return changed;
}

} // namespace transforms
} // namespace onnx
} // namespace si
