#include "transform_util.h"

#include "onnx_activation.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>

namespace si {
namespace onnx {
namespace transforms {

const char * const kActivations[8] = {"Relu",  "Sigmoid", "Tanh", "HardSigmoid",
                                     "HardSwish", "Gelu", "Silu", "Clip"};

bool reads(const onnxpb::node_proto & n, const std::string & value) {
    for (const std::string & in : n.input) {
        if (in == value) return true;
    }
    return false;
}

bool parse_activation(const graph_build & g, const onnxpb::node_proto & n,
                      std::vector<float> & params) {
    if (getenv("SI_ONNX_NOFUSE_ACT")) return false;
    params.clear();
    const std::string & op = n.op_type;
    if (op == "HardSigmoid") {
        float alpha = 0.2f, beta = 0.5f;
        for (const onnxpb::attribute & a : n.attribute) {
            if (a.name == "alpha") alpha = a.f;
            if (a.name == "beta") beta = a.f;
        }
        float p[2] = {0.0f, 0.0f};
        hardsigmoid_params(g.opset, alpha, beta, p);
        params = {p[0], p[1]};
        return true;
    }
    if (op == "HardSwish") {
        float p[2] = {0.0f, 0.0f};
        hardswish_params(p);
        params = {p[0], p[1]};
        return true;
    }
    if (op == "Clip") {
        // The bounds are inputs, and they have to be constants.
        float lo = -3.0e38f, hi = 3.0e38f;
        for (size_t k = 1; k < n.input.size(); ++k) {
            if (n.input[k].empty()) continue;
            auto it = g.slot.find(n.input[k]);
            if (it == g.slot.end() || !g.values[(size_t)it->second].is_init ||
                g.values[(size_t)it->second].t.host.size() < 4) {
                return false;
            }
            const float v = *(const float *)g.values[(size_t)it->second].t.host.data();
            if (k == 1) lo = v;
            else hi = v;
        }
        params = {lo, hi};
        return true;
    }
    if (op == "Gelu") {
        std::string approx;
        for (const onnxpb::attribute & a : n.attribute) {
            if (a.name == "approximate") approx = a.s;
        }
        return approx == "tanh";
    }
    return op == "Relu" || op == "Sigmoid" || op == "Tanh" || op == "Silu";
}

bool splat(const graph_build & g, const std::string & name, double & value, int depth) {
    if (depth > 32) return false;
    auto sit = g.slot.find(name);
    if (sit != g.slot.end() && g.values[(size_t)sit->second].is_init) {
        const tensor & t = g.values[(size_t)sit->second].t;
        if (t.dtype != onnxpb::FLOAT || t.host.size() < 4) return false;
        const size_t n = std::max<size_t>(1, t.host.size() / 4);
        float v0 = 0.0f;
        std::memcpy(&v0, t.host.data(), 4);
        for (size_t k = 1; k < n; ++k) {
            float v = 0.0f;
            std::memcpy(&v, t.host.data() + k * 4, 4);
            if (v != v0) return false;
        }
        value = v0;
        return true;
    }
    auto pit = g.producer_of.find(name);
    if (pit == g.producer_of.end()) return false;
    const onnxpb::node_proto & p = g.nodes[(size_t)pit->second];
    if (p.op_type == "Constant") {
        for (const onnxpb::attribute & a : p.attribute) {
            if (a.name != "value" || a.type_ != onnxpb::attribute::type::t) continue;
            if (a.t.data_type != onnxpb::FLOAT || a.t.raw.size() < 4) return false;
            float v0 = 0.0f;
            std::memcpy(&v0, a.t.raw.data(), 4);
            for (size_t k = 1; k * 4 < a.t.raw.size(); ++k) {
                float v = 0.0f;
                std::memcpy(&v, a.t.raw.data() + k * 4, 4);
                if (v != v0) return false;
            }
            value = v0;
            return true;
        }
        return false;
    }
    if (p.input.empty()) return false;
    if (p.op_type == "Expand" || p.op_type == "Cast" || p.op_type == "Unsqueeze" ||
        p.op_type == "Squeeze" || p.op_type == "Reshape" || p.op_type == "Flatten" ||
        p.op_type == "Identity") {
        return splat(g, p.input[0], value, depth + 1);
    }
    if (p.op_type == "Add" || p.op_type == "Sub" || p.op_type == "Mul" ||
        p.op_type == "Div") {
        if (p.input.size() < 2 || p.input[0].empty() || p.input[1].empty()) return false;
        double a = 0.0, b = 0.0;
        if (!splat(g, p.input[0], a, depth + 1) || !splat(g, p.input[1], b, depth + 1)) {
            return false;
        }
        if (p.op_type == "Add") value = a + b;
        else if (p.op_type == "Sub") value = a - b;
        else if (p.op_type == "Mul") value = a * b;
        else value = b == 0.0 ? 0.0 : a / b;
        return true;
    }
    return false;
}

} // namespace transforms
} // namespace onnx
} // namespace si
