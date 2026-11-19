#include "npu_ir.h"

#include "const_eval.h"

#include <algorithm>
#include <functional>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <cstring>
#include <map>
#include <set>
#include <fstream>
#include <sstream>

namespace si {
namespace npu {

namespace {

std::string esc(const std::string & s) {
    std::string r;
    for (char c : s) {
        switch (c) {
            case '&': r += "&amp;"; break;
            case '<': r += "&lt;"; break;
            case '>': r += "&gt;"; break;
            case '"': r += "&quot;"; break;
            case '\'': r += "&apos;"; break;
            default: r += c;
        }
    }
    return r;
}

const char * precision(int32_t t) {
    switch (t) {
        case onnxpb::FLOAT: return "FP32";
        case onnxpb::FLOAT16: return "FP16";
        case onnxpb::DOUBLE: return "FP64";
        case onnxpb::INT8: return "I8";
        case onnxpb::UINT8: return "U8";
        case onnxpb::INT16: return "I16";
        case onnxpb::UINT16: return "U16";
        case onnxpb::INT32: return "I32";
        case onnxpb::UINT32: return "U32";
        case onnxpb::INT64: return "I64";
        case onnxpb::UINT64: return "U64";
        case onnxpb::BOOL: return "BOOL";
        default: return "FP32";
    }
}

const char * element_type(int32_t t) {
    switch (t) {
        case onnxpb::FLOAT: return "f32";
        case onnxpb::FLOAT16: return "f16";
        case onnxpb::DOUBLE: return "f64";
        case onnxpb::INT8: return "i8";
        case onnxpb::UINT8: return "u8";
        case onnxpb::INT16: return "i16";
        case onnxpb::UINT16: return "u16";
        case onnxpb::INT32: return "i32";
        case onnxpb::UINT32: return "u32";
        case onnxpb::INT64: return "i64";
        case onnxpb::UINT64: return "u64";
        case onnxpb::BOOL: return "boolean";
        default: return "f32";
    }
}

std::string data_shape(const std::vector<int64_t> & v) {
    std::string r;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) r += ",";
        r += v[i] < 0 ? "?" : std::to_string(v[i]);
    }
    return r;
}

std::string join(const std::vector<int64_t> & v) {
    std::string r;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) r += ",";
        r += std::to_string(v[i]);
    }
    return r;
}


std::vector<int64_t> read_ints(const onnxpb::tensor_proto & t) {
    std::vector<int64_t> v;
    if (t.data_type == onnxpb::INT64) {
        v.resize(t.raw.size() / 8);
        std::memcpy(v.data(), t.raw.data(), v.size() * 8);
    } else if (t.data_type == onnxpb::INT32) {
        std::vector<int32_t> tmp(t.raw.size() / 4);
        std::memcpy(tmp.data(), t.raw.data(), tmp.size() * 4);
        v.assign(tmp.begin(), tmp.end());
    }
    return v;
}

struct port {
    int id = 0;
    int32_t dtype = onnxpb::FLOAT;
    std::vector<int64_t> dims;
    std::string names;
};

struct layer {
    int id = 0;
    std::string name;
    std::string type;
    std::string version = "opset1";
    std::vector<std::pair<std::string, std::string>> data;
    std::vector<port> in;
    std::vector<port> out;
};

struct shape_info {
    std::vector<int64_t> dims;
    int32_t dtype = onnxpb::FLOAT;
};

struct bn_fold {
    int bn_node = -1;
    bool generic = false;
    std::vector<uint8_t> factor;
    std::vector<uint8_t> offset;
    std::string weight_name;
    std::string bias_name;
    std::vector<uint8_t> weight;
    std::vector<uint8_t> bias;
};

std::vector<int64_t> broadcast(const std::vector<int64_t> & a, const std::vector<int64_t> & b) {
    size_t n = std::max(a.size(), b.size());
    std::vector<int64_t> r(n, 1);
    for (size_t i = 0; i < n; ++i) {
        int64_t da = i >= n - a.size() ? a[i - (n - a.size())] : 1;
        int64_t db = i >= n - b.size() ? b[i - (n - b.size())] : 1;
        r[i] = da == 1 ? db : (db == 1 || db == da ? da : (da == 0 ? db : da));
    }
    return r;
}

int64_t attr_i(const onnxpb::node_proto & n, const char * key, int64_t def) {
    for (const auto & a : n.attribute)
        if (a.name == key && a.type_ == onnxpb::attribute::type::i) return static_cast<int64_t>(a.i);
    return def;
}

double attr_f(const onnxpb::node_proto & n, const char * key, double def) {
    for (const auto & a : n.attribute)
        if (a.name == key && a.type_ == onnxpb::attribute::type::f) return a.f;
    return def;
}

std::vector<int64_t> attr_ints(const onnxpb::node_proto & n, const char * key) {
    for (const auto & a : n.attribute)
        if (a.name == key && a.type_ == onnxpb::attribute::type::ints) return a.ints;
    return {};
}

std::string attr_s(const onnxpb::node_proto & n, const char * key, const char * def = "") {
    for (const auto & a : n.attribute)
        if (a.name == key && a.type_ == onnxpb::attribute::type::s) return a.s;
    return def;
}

bool supported_op(const std::string & op) {
    static const std::set<std::string> kOps = {
        "Add", "Sub", "Mul", "Div", "Pow", "Min", "Max", "Greater", "Less", "Equal",
        "And", "Or", "Xor", "Mod", "Sum", "Mean", "Where", "Not",
        "Relu", "LeakyRelu", "Sigmoid", "Tanh", "Abs", "Exp", "Sqrt", "Log", "Neg",
        "Softmax", "Softplus", "Erf", "Sign", "Floor", "Ceil", "Round", "Reciprocal",
        "Clip", "Identity", "Elu", "Selu", "HardSwish", "HardSigmoid",
        "Conv", "ConvTranspose", "MatMul", "Gemm", "MaxPool", "AveragePool",
        "GlobalAveragePool", "GlobalMaxPool", "Reshape", "Transpose", "Flatten",
        "Squeeze", "Unsqueeze", "Concat", "Gather", "Slice", "Split", "Cast", "Pad",
        "ReduceMax", "ReduceMin", "ReduceSum", "ReduceMean", "Tile", "Expand",
        "Constant", "BatchNormalization", "Resize", "GatherElements", "GridSample",
        "LayerNormalization", "Trilu", "TopK", "ScatterND", "Range", "ConstantOfShape",
        "Shape", "ScatterElements", "GlobalLpPool", "Softsign", "Hardmax",
        "Einsum", "Gelu", "LessOrEqual", "GreaterOrEqual", "ReduceProd", "BitShift",
        "Sin", "Cos", "Ceil", "Floor", "Reciprocal", "Log", "Sqrt", "Tanh",
        "ReduceL2", "ReduceSumSquare", "ArgMax", "ArgMin", "MaxRoiPool",
    };
    return kOps.count(op) != 0;
}

std::string layer_type(const std::string & op) {
    if (op == "Add") return "Add";
    if (op == "Sub") return "Subtract";
    if (op == "Mul") return "Multiply";
    if (op == "Div") return "Divide";
    if (op == "Pow") return "Power";
    if (op == "Min") return "Minimum";
    if (op == "Max") return "Maximum";
    if (op == "Greater") return "Greater";
    if (op == "Less") return "Less";
    if (op == "Equal") return "Equal";
    if (op == "And") return "LogicalAnd";
    if (op == "Or") return "LogicalOr";
    if (op == "Not") return "LogicalNot";
    if (op == "Xor") return "LogicalXor";
    if (op == "Mod") return "Mod";
    if (op == "Sum" || op == "Mean") return "Add";
    if (op == "Where") return "Select";
    if (op == "Relu") return "ReLU";
    if (op == "LeakyRelu") return "LeakyRelu";
    if (op == "Sigmoid") return "Sigmoid";
    if (op == "Tanh") return "Tanh";
    if (op == "Abs") return "Abs";
    if (op == "Exp") return "Exp";
    if (op == "Sqrt") return "Sqrt";
    if (op == "Log") return "Log";
    if (op == "Neg") return "Negative";
    if (op == "Softmax") return "SoftMax";
    if (op == "Softplus") return "SoftPlus";
    if (op == "Erf") return "Erf";
    if (op == "Sign") return "Sign";
    if (op == "Floor") return "Floor";
    if (op == "Ceil") return "Ceiling";
    if (op == "Round") return "Round";
    if (op == "Reciprocal") return "Power";
    if (op == "Clip") return "Clamp";
    if (op == "Identity") return "Identity";
    if (op == "Elu") return "Elu";
    if (op == "Selu") return "Selu";
    if (op == "HardSwish") return "HardSwish";
    if (op == "HardSigmoid") return "HardSigmoid";
    if (op == "Conv") return "Convolution";
    if (op == "ConvTranspose") return "ConvolutionBackpropData";
    if (op == "MatMul") return "MatMul";
    if (op == "Gemm") return "Gemm";
    if (op == "MaxPool") return "MaxPool";
    if (op == "AveragePool") return "AvgPool";
    if (op == "GlobalAveragePool") return "ReduceMean";
    if (op == "GlobalMaxPool") return "ReduceMax";
    if (op == "Reshape") return "Reshape";
    if (op == "Transpose") return "Transpose";
    if (op == "Flatten") return "Flatten";
    if (op == "Squeeze") return "Squeeze";
    if (op == "Unsqueeze") return "Unsqueeze";
    if (op == "Concat") return "Concat";
    if (op == "Gather") return "Gather";
    if (op == "Slice") return "Slice";
    if (op == "Split") return "Split";
    if (op == "Cast") return "Convert";
    if (op == "Pad") return "Pad";
    if (op == "ReduceMax") return "ReduceMax";
    if (op == "ReduceMin") return "ReduceMin";
    if (op == "ReduceSum") return "ReduceSum";
    if (op == "ReduceMean") return "ReduceMean";
    if (op == "Tile") return "Tile";
    if (op == "Expand") return "Broadcast";
    if (op == "Constant") return "Const";
    if (op == "Resize") return "Interpolate";
    if (op == "GatherElements") return "GatherElements";
    if (op == "GridSample") return "GridSample";
    if (op == "LayerNormalization") return "LayerNorm";
    if (op == "Trilu") return "Trilu";
    if (op == "TopK") return "TopK";
    if (op == "ScatterND") return "ScatterNDUpdate";
    if (op == "Range") return "Range";
    if (op == "ConstantOfShape") return "Const";
    if (op == "Shape") return "ShapeOf";
    if (op == "ScatterElements") return "ScatterElementsUpdate";
    if (op == "Softsign") return "SoftSign";
    if (op == "Einsum") return "Einsum";
    if (op == "Gelu") return "Gelu";
    if (op == "LessOrEqual") return "LessEqual";
    if (op == "GreaterOrEqual") return "GreaterEqual";
    if (op == "BatchNormalization") return "BatchNormInference";
    return op;
}

std::string op_version(const std::string & op) {
    if (op == "Slice") return "opset8";
    if (op == "Squeeze" || op == "Unsqueeze" || op == "Split") return "opset1";
    if (op == "Softmax") return "opset8";
    if (op == "Einsum") return "opset7";
    if (op == "Gelu") return "opset7";
    if (op == "Clip") return "opset1";
    if (op == "Resize") return "opset11";
    if (op == "MaxPool") return "opset8";
    if (op == "LayerNormalization") return "opset1";
    if (op == "TopK") return "opset11";
    if (op == "Shape") return "opset3";
    if (op == "Expand") return "opset3";
    if (op == "GridSample") return "opset9";
    if (op == "Range") return "opset4";
    if (op == "ScatterND") return "opset15";
    if (op == "ScatterElements") return "opset12";
    if (op == "Softmax") return "opset8";
    if (op == "Einsum") return "opset7";
    if (op == "Gelu") return "opset7";
    if (op == "Gather") return "opset8";
    if (op == "ReduceMax" || op == "ReduceMin" || op == "ReduceSum" || op == "ReduceMean") return "opset1";
    return "opset1";
}

void write_ports(std::string & xml, const char * tag, const std::vector<port> & ports, bool with_names) {
    if (ports.empty()) return;
    xml += "<";
    xml += tag;
    xml += ">";
    for (const auto & p : ports) {
        xml += "<port id=\"" + std::to_string(p.id) + "\" precision=\"" + precision(p.dtype) + "\"";
        if (with_names && !p.names.empty()) xml += " names=\"" + esc(p.names) + "\"";
        xml += ">";
        for (int64_t d : p.dims) xml += "<dim>" + std::to_string(d) + "</dim>";
        xml += "</port>";
    }
    xml += "</";
    xml += tag;
    xml += ">";
}

}  // namespace

bool serialize_ir(const onnxpb::model_proto & m, ir_result & out, std::string & err) {
    const auto & g = m.graph;
    std::map<std::string, const onnxpb::tensor_proto *> init;
    for (const auto & t : g.initializer) init[t.name] = &t;
    std::map<std::string, const onnxpb::tensor_proto *> const_nodes;
    for (const auto & n : g.node) {
        if (n.op_type == "Constant" && !n.output.empty() && !n.attribute.empty() &&
            n.attribute[0].type_ == onnxpb::attribute::type::t)
            const_nodes[n.output[0]] = &n.attribute[0].t;
    }
    std::deque<onnxpb::tensor_proto> synthesized;
    std::function<const onnxpb::tensor_proto *(const std::string &)> resolve_const =
        [&](const std::string & name) -> const onnxpb::tensor_proto * {
        auto a = init.find(name);
        if (a != init.end()) return a->second;
        auto b = const_nodes.find(name);
        if (b != const_nodes.end()) return b->second;
        for (const auto & n : g.node) {
            if (n.output.empty() || n.output[0] != name) continue;
            if (n.op_type == "Identity" && n.input.size() == 1) return resolve_const(n.input[0]);
            if (n.op_type == "Reshape" && n.input.size() >= 2) {
                const auto * src = resolve_const(n.input[0]);
                const auto * shp = resolve_const(n.input[1]);
                if (src && shp) {
                    onnxpb::tensor_proto t = *src;
                    t.name = name;
                    t.dims = read_ints(*shp);
                    synthesized.push_back(std::move(t));
                    return &synthesized.back();
                }
            }
            if (n.op_type == "Cast" && n.input.size() == 1) {
                const auto * src = resolve_const(n.input[0]);
                if (!src) return nullptr;
                int32_t to = static_cast<int32_t>(attr_i(n, "to", src->data_type));
                if (to == src->data_type) return src;
                size_t nelem = src->raw.size() / onnxpb::elem_size(src->data_type);
                std::vector<uint8_t> out(nelem * onnxpb::elem_size(to), 0);
                for (size_t q = 0; q < nelem; ++q) {
                    int64_t v = 0;
                    if (src->data_type == onnxpb::INT64) v = reinterpret_cast<const int64_t *>(src->raw.data())[q];
                    else if (src->data_type == onnxpb::INT32) v = reinterpret_cast<const int32_t *>(src->raw.data())[q];
                    else if (src->data_type == onnxpb::FLOAT) v = static_cast<int64_t>(reinterpret_cast<const float *>(src->raw.data())[q]);
                    if (to == onnxpb::INT64) reinterpret_cast<int64_t *>(out.data())[q] = v;
                    else if (to == onnxpb::INT32) reinterpret_cast<int32_t *>(out.data())[q] = static_cast<int32_t>(v);
                    else if (to == onnxpb::FLOAT) reinterpret_cast<float *>(out.data())[q] = static_cast<float>(v);
                }
                onnxpb::tensor_proto t;
                t.name = name;
                t.data_type = to;
                t.dims = src->dims;
                t.raw.assign(out.begin(), out.end());
                synthesized.push_back(std::move(t));
                return &synthesized.back();
            }
        }
        return nullptr;
    };
    std::map<std::string, shape_info> shapes;
    for (const auto & t : g.input) {
        shape_info s;
        s.dims = t.shape;
        s.dtype = t.elem_type;
        shapes[t.name] = s;
    }
    for (const auto & t : g.value_info) {
        if (!t.has_shape) continue;
        shape_info s;
        s.dims = t.shape;
        s.dtype = t.elem_type;
        shapes[t.name] = s;
    }
    for (const auto & t : g.output) {
        shape_info s;
        s.dims = t.shape;
        s.dtype = t.elem_type;
        shapes[t.name] = s;
    }
    for (const auto & kv : init) {
        shape_info s;
        s.dims = kv.second->dims;
        s.dtype = kv.second->data_type;
        shapes[kv.first] = s;
    }

    if (const char * shp = std::getenv("PF_NPU_INPUT_SHAPES")) {
        std::string spec = shp;
        size_t start = 0;
        while (start < spec.size()) {
            size_t semi = spec.find(';', start);
            if (semi == std::string::npos) semi = spec.size();
            std::string item = spec.substr(start, semi - start);
            start = semi + 1;
            size_t colon = item.find(':');
            if (colon == std::string::npos) continue;
            std::string name = item.substr(0, colon);
            auto it = shapes.find(name);
            if (it == shapes.end()) continue;
            std::vector<int64_t> dims;
            size_t p = colon + 1;
            while (p <= item.size()) {
                size_t comma = item.find(',', p);
                if (comma == std::string::npos) comma = item.size();
                std::string tok = item.substr(p, comma - p);
                if (!tok.empty()) dims.push_back(std::strtoll(tok.c_str(), nullptr, 10));
                p = comma + 1;
            }
            if (!dims.empty()) it->second.dims = dims;
        }
    }

    if (const char * sf = std::getenv("PF_NPU_SHAPE_MAP")) {
        std::ifstream f(sf);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream iss(line);
            std::string name;
            if (!(iss >> name)) continue;
            std::vector<int64_t> dims;
            int64_t d;
            while (iss >> d) dims.push_back(d);
            if (!dims.empty()) shapes[name].dims = dims;
        }
    }

    std::map<std::string, int> node_of_output;
    std::map<std::string, int> uses;
    for (size_t i = 0; i < g.node.size(); ++i) {
        for (const auto & o : g.node[i].output) node_of_output[o] = static_cast<int>(i);
        for (const auto & in : g.node[i].input) uses[in]++;
    }
    std::map<int, bn_fold> folds;
    std::map<std::string, int64_t> weight_group;
    for (const auto & cnode : g.node) {
        if (cnode.op_type != "Conv" || cnode.input.size() < 2) continue;
        int64_t grp = attr_i(cnode, "group", 1);
        if (grp > 1) weight_group[cnode.input[1]] = grp;
    }
    for (size_t j = 0; j < g.node.size(); ++j) {
        const auto & bn = g.node[j];
        if (bn.op_type != "BatchNormalization" || bn.input.size() < 5) continue;
        auto ci = node_of_output.find(bn.input[0]);
        if (ci == node_of_output.end()) continue;
        const auto & conv = g.node[ci->second];
        if (conv.op_type != "Conv" || conv.input.size() < 2 || uses[bn.input[0]] != 1) continue;
        auto wit = init.find(conv.input[1]);
        if (wit == init.end() || wit->second->data_type != onnxpb::FLOAT) continue;
        const auto & w = *wit->second;
        if (w.dims.size() < 2) continue;
        size_t M = static_cast<size_t>(w.dims[0]);
        size_t C = static_cast<size_t>(w.dims[1]);
        std::vector<const onnxpb::tensor_proto *> pars;
        bool ok = true;
        for (size_t k = 1; k < 5; ++k) {
            auto it = init.find(bn.input[k]);
            if (it == init.end() || it->second->data_type != onnxpb::FLOAT) { ok = false; break; }
            pars.push_back(it->second);
        }
        if (!ok) continue;
        bn_fold f;
        f.bn_node = static_cast<int>(j);
        f.weight_name = conv.input[1];
        f.bias_name = bn.output[0] + "_fold_bias";
        const float * wp = reinterpret_cast<const float *>(w.raw.data());
        size_t nw = w.raw.size() / 4;
        f.weight.resize(w.raw.size());
        float * wop = reinterpret_cast<float *>(f.weight.data());
        for (size_t q = 0; q < nw; ++q) wop[q] = wp[q];
        const float * sc = reinterpret_cast<const float *>(pars[0]->raw.data());
        const float * bt = reinterpret_cast<const float *>(pars[1]->raw.data());
        const float * mu = reinterpret_cast<const float *>(pars[2]->raw.data());
        const float * va = reinterpret_cast<const float *>(pars[3]->raw.data());
        double eps = attr_f(bn, "epsilon", 1e-5);
        size_t nc = pars[0]->raw.size() / 4;
        std::vector<double> factor(C);
        for (size_t c = 0; c < C; ++c) {
            double v = c < nc ? va[c] : 0.0;
            double scale = c < nc ? sc[c] : 1.0;
            factor[c] = scale / std::sqrt(v + eps);
        }
        size_t chans_per_row = nw / (M ? M : 1);
        for (size_t m = 0; m < M; ++m) {
            double cidx = static_cast<double>(m % (C ? C : 1));
            double fac = factor[static_cast<size_t>(cidx)];
            for (size_t q = 0; q < chans_per_row; ++q) {
                size_t idx = m * chans_per_row + q;
                if (idx < nw) wop[idx] = static_cast<float>(wp[idx] * fac);
            }
        }
        f.bias.resize(M * 4);
        float * bp = reinterpret_cast<float *>(f.bias.data());
        for (size_t m = 0; m < M; ++m) {
            double off = m < pars[1]->raw.size() / 4 ? bt[m] : 0.0;
            double mean = m < pars[2]->raw.size() / 4 ? mu[m] : 0.0;
            double fac = factor[m % (C ? C : 1)];
            bp[m] = static_cast<float>(off - mean * fac);
        }
        folds[ci->second] = std::move(f);
    }

    for (size_t j = 0; j < g.node.size(); ++j) {
        const auto & bn = g.node[j];
        if (bn.op_type != "BatchNormalization" || bn.input.size() < 5 || bn.output.empty()) continue;
        bool already = false;
        for (const auto & kv : folds) if (kv.second.bn_node == static_cast<int>(j)) { already = true; break; }
        if (already) continue;
        std::vector<const onnxpb::tensor_proto *> pars;
        bool ok = true;
        for (size_t k = 1; k < 5; ++k) {
            auto it = init.find(bn.input[k]);
            if (it == init.end() || it->second->data_type != onnxpb::FLOAT) { ok = false; break; }
            pars.push_back(it->second);
        }
        if (!ok) continue;
        size_t C = pars[0]->raw.size() / 4;
        if (C == 0) continue;
        const float * sc = reinterpret_cast<const float *>(pars[0]->raw.data());
        const float * bt = reinterpret_cast<const float *>(pars[1]->raw.data());
        const float * mu = reinterpret_cast<const float *>(pars[2]->raw.data());
        const float * va = reinterpret_cast<const float *>(pars[3]->raw.data());
        double eps = attr_f(bn, "epsilon", 1e-5);
        bn_fold f;
        f.generic = true;
        f.bn_node = static_cast<int>(j);
        f.factor.resize(C * 4);
        f.offset.resize(C * 4);
        float * fp = reinterpret_cast<float *>(f.factor.data());
        float * op = reinterpret_cast<float *>(f.offset.data());
        for (size_t c = 0; c < C; ++c) {
            double fac = sc[c] / std::sqrt(va[c] + eps);
            fp[c] = static_cast<float>(fac);
            op[c] = static_cast<float>(bt[c] - mu[c] * fac);
        }
        folds[static_cast<int>(j)] = std::move(f);
    }

    if (std::getenv("PF_NPU_NO_BN_FOLD")) {
        folds.clear();
    }

    std::vector<layer> layers;
    std::map<std::string, std::pair<int, int>> producer;
    auto name_of = [&](size_t i) { return "layer_" + std::to_string(i); };

    for (size_t i = 0; i < g.input.size(); ++i) {
        const auto & v = g.input[i];
        if (init.count(v.name)) continue;
        layer l;
        l.id = static_cast<int>(layers.size());
        l.name = v.name;
        l.type = "Parameter";
        l.data.push_back({"shape", data_shape(shapes[v.name].dims)});
        l.data.push_back({"element_type", element_type(shapes[v.name].dtype)});
        port p;
        p.id = 0;
        p.dtype = shapes[v.name].dtype;
        p.dims = shapes[v.name].dims;
        p.names = v.name;
        l.out.push_back(p);
        producer[v.name] = {l.id, 0};
        layers.push_back(std::move(l));
        out.inputs.push_back(v.name);
    }

    const std::vector<uint8_t> * weight_bytes = nullptr;
    for (const auto & kv : init) {
        const onnxpb::tensor_proto & t = *kv.second;
        weight_bytes = nullptr;
        for (const auto & kv2 : folds) {
            if (kv2.second.weight_name == kv.first) { weight_bytes = &kv2.second.weight; break; }
        }
        const std::vector<uint8_t> & bytes = weight_bytes ? *weight_bytes : t.raw;
        layer l;
        l.id = static_cast<int>(layers.size());
        l.name = kv.first;
        l.type = "Const";
        l.data.push_back({"element_type", element_type(t.data_type)});
        l.data.push_back({"shape", data_shape(t.dims)});
        l.data.push_back({"offset", std::to_string(out.weights.size())});
        l.data.push_back({"size", std::to_string(bytes.size())});
        std::vector<int64_t> const_dims = t.dims;
        auto wg = weight_group.find(kv.first);
        if (wg != weight_group.end() && const_dims.size() == 4) {
            const_dims = {wg->second,
                          const_dims[0] / wg->second,
                          const_dims[1],
                          const_dims[2],
                          const_dims[3]};
            for (auto & d : l.data) if (d.first == "shape") d.second = data_shape(const_dims);
        }
        port p;
        p.id = 0;
        p.dtype = t.data_type;
        p.dims = const_dims;
        p.names = kv.first;
        l.out.push_back(p);
        producer[kv.first] = {l.id, 0};
        layers.push_back(std::move(l));
        out.weights.insert(out.weights.end(), bytes.begin(), bytes.end());
        while (out.weights.size() % 64 != 0) out.weights.push_back(0);
    }

    for (const auto & kv : folds) {
        const bn_fold & f = kv.second;
        layer l;
        l.id = static_cast<int>(layers.size());
        l.name = f.bias_name;
        l.type = "Const";
        l.data.push_back({"element_type", "f32"});
        l.data.push_back({"shape", "1," + std::to_string(f.bias.size() / 4) + ",1,1"});
        l.data.push_back({"offset", std::to_string(out.weights.size())});
        l.data.push_back({"size", std::to_string(f.bias.size())});
        port p;
        p.id = 0;
        p.dtype = onnxpb::FLOAT;
        p.dims = {1, static_cast<int64_t>(f.bias.size() / 4), 1, 1};
        p.names = f.bias_name;
        l.out.push_back(p);
        producer[f.bias_name] = {l.id, 0};
        layers.push_back(std::move(l));
        out.weights.insert(out.weights.end(), f.bias.begin(), f.bias.end());
        while (out.weights.size() % 64 != 0) out.weights.push_back(0);
    }

    auto emit_scalar_const = [&](const std::string & name, int32_t dtype,
                                 const void * bytes, size_t nbytes,
                                 const std::vector<int64_t> & dims) {
        layer l;
        l.id = static_cast<int>(layers.size());
        l.name = name;
        l.type = "Const";
        l.version = "opset1";
        l.data.push_back({"element_type", element_type(dtype)});
        l.data.push_back({"shape", data_shape(dims)});
        l.data.push_back({"offset", std::to_string(out.weights.size())});
        l.data.push_back({"size", std::to_string(nbytes)});
        port p;
        p.id = 0;
        p.dtype = dtype;
        p.dims = dims;
        l.out.push_back(p);
        producer[name] = {l.id, 0};
        layers.push_back(std::move(l));
        const uint8_t * bp = static_cast<const uint8_t *>(bytes);
        out.weights.insert(out.weights.end(), bp, bp + nbytes);
        while (out.weights.size() % 64 != 0) out.weights.push_back(0);
    };

    std::set<int> folded_bn;
    for (const auto & kv : folds) folded_bn.insert(kv.second.bn_node);
    for (size_t ni = 0; ni < g.node.size(); ++ni) {
        if (std::getenv("PF_NPU_IRDBG") && ni >= 55 && ni <= 75) std::fprintf(stderr, "node %zu %s\n", ni, g.node[ni].op_type.c_str());
        const auto & n = g.node[ni];
        auto gfit = folds.find(static_cast<int>(ni));
        if (gfit != folds.end() && gfit->second.generic) {
            const std::string base = n.name.empty() ? name_of(ni) : n.name;
            const std::string fn = base + "_bn_factor", on = base + "_bn_offset";
            const std::vector<int64_t> chdims = {1, static_cast<int64_t>(gfit->second.factor.size() / 4), 1, 1};
            emit_scalar_const(fn, onnxpb::FLOAT, gfit->second.factor.data(), gfit->second.factor.size(), chdims);
            emit_scalar_const(on, onnxpb::FLOAT, gfit->second.offset.data(), gfit->second.offset.size(), chdims);
            const std::string & bn_out = n.output[0];
            const std::string & bn_in = n.input[0];
            auto make_e = [&](const std::string & nm, const std::string & type, int id,
                              const std::string & in0, const std::string & in1,
                              const std::vector<int64_t> & dims) {
                layer el;
                el.id = static_cast<int>(layers.size());
                el.name = nm;
                el.type = type;
                el.version = "opset1";
                for (size_t k = 0; k < 2; ++k) {
                    port p;
                    p.id = static_cast<int>(k);
                    p.dtype = onnxpb::FLOAT;
                    p.dims = k == 0 ? (shapes.count(in0) ? shapes[in0].dims : std::vector<int64_t>())
                                    : (k == 1 ? chdims : std::vector<int64_t>());
                    p.names = k == 0 ? in0 : in1;
                    el.in.push_back(p);
                }
                (void)id;
                port pout;
                pout.id = 2;
                pout.dtype = onnxpb::FLOAT;
                pout.dims = shapes.count(bn_out) ? shapes[bn_out].dims : dims;
                pout.names = nm;
                el.out.push_back(pout);
                layers.push_back(std::move(el));
            };
            make_e(base + "_bn_mul", "Multiply", 0, bn_in, fn, shapes.count(bn_out) ? shapes[bn_out].dims : std::vector<int64_t>{});
            shapes[base + "_bn_mul"] = shapes.count(bn_out) ? shapes[bn_out] : shape_info();
            producer[base + "_bn_mul"] = {layers.size() - 1, 2};
            make_e(bn_out, "Add", 0, base + "_bn_mul", on, shapes.count(bn_out) ? shapes[bn_out].dims : std::vector<int64_t>{});
            producer[bn_out] = {layers.size() - 1, 2};
            continue;
        }
        if (folded_bn.count(static_cast<int>(ni))) continue;
        if (n.op_type == "Identity" || n.op_type == "Dropout") {
            if (!n.input.empty() && !n.output.empty()) {
                shapes[n.output[0]] = shapes[n.input[0]];
                producer[n.output[0]] = producer[n.input[0]];
            }
            continue;
        }
        if (n.op_type == "Constant") {
            if (n.attribute.empty() || n.attribute[0].type_ != onnxpb::attribute::type::t) {
                err = "Constant node without a tensor attribute is not supported: " + n.name;
                return false;
            }
            const auto & t = n.attribute[0].t;
            layer l;
            l.id = static_cast<int>(layers.size());
            l.name = n.output.empty() ? (n.name.empty() ? name_of(ni) : n.name) : n.output[0];
            l.type = "Const";
            l.data.push_back({"element_type", element_type(t.data_type)});
            l.data.push_back({"shape", join(t.dims)});
            l.data.push_back({"offset", std::to_string(out.weights.size())});
            l.data.push_back({"size", std::to_string(t.raw.size())});
            port p;
            p.id = 0;
            p.dtype = t.data_type;
            p.dims = t.dims;
            p.names = n.output.empty() ? l.name : n.output[0];
            l.out.push_back(p);
            if (!n.output.empty()) producer[n.output[0]] = {l.id, 0};
            layers.push_back(std::move(l));
            out.weights.insert(out.weights.end(), t.raw.begin(), t.raw.end());
            while (out.weights.size() % 64 != 0) out.weights.push_back(0);
            continue;
        }
        std::string op = n.op_type;
        if (op == "LeakyRelu") op = "LeakyRelu";
        std::string type = layer_type(op);
        if (!supported_op(op)) {
            err = "ONNX operator is not supported by the NPU path: " + op + " (node " + n.name + ")";
            return false;
        }
        layer l;
        l.name = n.name.empty() ? name_of(ni) : n.name;
        l.type = type;
        l.version = op_version(op);
        int nin = 0;
        for (const auto & in_name : n.input) {
            if (in_name.empty()) continue;
            port p;
            p.id = nin++;
            p.dtype = shapes.count(in_name) ? shapes[in_name].dtype : onnxpb::FLOAT;
            p.dims = shapes.count(in_name) ? shapes[in_name].dims : std::vector<int64_t>{};
            p.names = in_name;
            l.in.push_back(p);
        }
        auto fit = folds.find(static_cast<int>(ni));
        std::string extra_bias_name;
        if ((op == "Conv" || op == "ConvTranspose") && l.in.size() > 2) {
            extra_bias_name = l.in[2].names;
            while (l.in.size() > 2) l.in.pop_back();
        }
        int nout = 0;
        for (const auto & out_name : n.output) {
            if (out_name.empty()) continue;
            port p;
            p.id = nin + nout++;
            p.dtype = shapes.count(out_name) ? shapes[out_name].dtype : onnxpb::FLOAT;
            p.dims = shapes.count(out_name) ? shapes[out_name].dims : std::vector<int64_t>{};
            p.names = out_name;
            l.out.push_back(p);
        }

        if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div" || op == "Pow" ||
            op == "Min" || op == "Max" || op == "And" || op == "Or" || op == "Xor") {
            l.data.push_back({"auto_broadcast", "numpy"});
        }
        if (op == "Conv" || op == "ConvTranspose") {
            auto strides = attr_ints(n, "strides");
            auto pads = attr_ints(n, "pads");
            auto dil = attr_ints(n, "dilations");
            if (strides.empty()) strides = {1, 1};
            if (dil.empty()) dil = {1, 1};
            std::string auto_pad = attr_s(n, "auto_pad", "NOTSET");
            l.data.push_back({"strides", join(strides)});
            l.data.push_back({"dilations", join(dil)});
            if (pads.size() >= 4) {
                l.data.push_back({"pads_begin", join(std::vector<int64_t>(pads.begin(), pads.begin() + pads.size() / 2))});
                l.data.push_back({"pads_end", join(std::vector<int64_t>(pads.begin() + pads.size() / 2, pads.end()))});
            } else {
                l.data.push_back({"pads_begin", "0,0"});
                l.data.push_back({"pads_end", "0,0"});
            }
            int64_t group = attr_i(n, "group", 1);
            if (group != 1) l.data.push_back({"group", std::to_string(group)});
            if (op == "Conv" && group > 1) l.type = "GroupConvolution";
            if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") l.data.push_back({"auto_pad", auto_pad});
            else l.data.push_back({"auto_pad", "explicit"});
        } else if (op == "MaxPool" || op == "AveragePool") {
            if (op == "MaxPool") {
                l.data.push_back({"rounding_type", attr_i(n, "ceil_mode", 0) ? "ceil" : "floor"});
                l.data.push_back({"index_element_type", "i64"});
                l.data.push_back({"axis", "0"});
                l.data.push_back({"auto_pad", "explicit"});
            }
            auto kernel = attr_ints(n, "kernel_shape");
            auto strides = attr_ints(n, "strides");
            auto pads = attr_ints(n, "pads");
            if (strides.empty()) strides.assign(kernel.size(), 1);
            l.data.push_back({"kernel", join(kernel)});
            l.data.push_back({"strides", join(strides)});
            auto mp_dil = attr_ints(n, "dilations");
            if (mp_dil.empty()) mp_dil.assign(kernel.size(), 1);
            l.data.push_back({"dilations", join(mp_dil)});
            if (pads.size() >= 4) {
                l.data.push_back({"pads_begin", join(std::vector<int64_t>(pads.begin(), pads.begin() + pads.size() / 2))});
                l.data.push_back({"pads_end", join(std::vector<int64_t>(pads.begin() + pads.size() / 2, pads.end()))});
            } else {
                l.data.push_back({"pads_begin", "0,0"});
                l.data.push_back({"pads_end", "0,0"});
            }
        } else if (op == "Concat") {
            l.data.push_back({"axis", std::to_string(attr_i(n, "axis", 1))});
        } else if (op == "Softmax") {
            l.data.push_back({"axis", std::to_string(attr_i(n, "axis", 1))});
        } else if (op == "Transpose") {
            auto perm = attr_ints(n, "perm");
            if (perm.empty()) { size_t rank = l.in.empty() ? 1 : l.in[0].dims.size(); for (size_t q=0;q<rank;++q) perm.push_back(q); }
            std::string cn = (n.name.empty() ? name_of(ni) : n.name) + "_order";
            emit_scalar_const(cn, onnxpb::INT64, perm.data(), perm.size() * 8,
                              {static_cast<int64_t>(perm.size())});
            port p;
            p.id = 1;
            p.dtype = onnxpb::INT64;
            p.dims = {static_cast<int64_t>(perm.size())};
            p.names = cn;
            l.in.push_back(p);
            for (auto & p2 : l.out) p2.id = 2;
        } else if (op == "Flatten") {
            l.data.push_back({"axis", std::to_string(attr_i(n, "axis", 1))});
        } else if (op == "Squeeze" || op == "Unsqueeze") {
            auto axes = attr_ints(n, "axes");
            if (axes.empty() && n.input.size() > 1) {
                const auto * t = resolve_const(n.input[1]);
                if (t) axes = read_ints(*t);
            }
            while (l.in.size() > 1) l.in.pop_back();
            std::vector<int64_t> ax(axes.begin(), axes.end());
            std::string an = (n.name.empty() ? name_of(ni) : n.name) + "_axes";
            emit_scalar_const(an, onnxpb::INT64, ax.data(), ax.size() * 8,
                              {static_cast<int64_t>(ax.size())});
            port ap;
            ap.id = 1;
            ap.dtype = onnxpb::INT64;
            ap.dims = {static_cast<int64_t>(ax.size())};
            ap.names = an;
            l.in.push_back(ap);
            for (auto & p : l.out) p.id = 2;
        } else if (op == "Split") {
            int64_t ax = attr_i(n, "axis", 0);
            auto split = attr_ints(n, "split");
            while (l.in.size() > 1) l.in.pop_back();
            std::vector<int64_t> axv = {ax};
            std::string an = (n.name.empty() ? name_of(ni) : n.name) + "_axis";
            emit_scalar_const(an, onnxpb::INT64, axv.data(), axv.size() * 8, {1});
            port ap;
            ap.id = 1;
            ap.dtype = onnxpb::INT64;
            ap.dims = {1};
            ap.names = an;
            l.in.push_back(ap);
            if (l.out.size() > 1) {
                l.type = "VariadicSplit";
                std::vector<int64_t> sv = split;
                if (sv.empty()) sv.assign(l.out.size(), 1);
                std::string sn = (n.name.empty() ? name_of(ni) : n.name) + "_split";
                emit_scalar_const(sn, onnxpb::INT64, sv.data(), sv.size() * 8,
                                  {static_cast<int64_t>(sv.size())});
                port sp;
                sp.id = 2;
                sp.dtype = onnxpb::INT64;
                sp.dims = {static_cast<int64_t>(sv.size())};
                sp.names = sn;
                l.in.push_back(sp);
            } else if (!split.empty()) {
                l.data.push_back({"split", join(split)});
            }
        } else if (op == "Gather") {
            int64_t ax = attr_i(n, "axis", 0);
            std::vector<int64_t> axv = {ax};
            std::string cn = (n.name.empty() ? name_of(ni) : n.name) + "_axis";
            emit_scalar_const(cn, onnxpb::INT64, axv.data(), 8, {1});
            port p;
            p.id = 2;
            p.dtype = onnxpb::INT64;
            p.dims = {1};
            p.names = cn;
            l.in.push_back(p);
            for (auto & p2 : l.out) p2.id = 3;
        } else if (op == "Slice") {
            std::vector<int64_t> starts, ends, axes, steps;
            if (n.input.size() > 1) { const auto * t = resolve_const(n.input[1]); if (t) starts = read_ints(*t); }
            if (n.input.size() > 2) { const auto * t = resolve_const(n.input[2]); if (t) ends = read_ints(*t); }
            if (n.input.size() > 3) { const auto * t = resolve_const(n.input[3]); if (t) axes = read_ints(*t); }
            if (n.input.size() > 4) { const auto * t = resolve_const(n.input[4]); if (t) steps = read_ints(*t); }
            if (starts.empty()) starts.assign(1, 0);
            if (ends.empty()) ends.assign(1, std::numeric_limits<int64_t>::max());
            if (axes.empty()) axes.assign(1, 0);
            if (steps.empty()) steps.assign(axes.size(), 1);
            for (size_t q = 0; q < steps.size(); ++q) {
                if (steps[q] == 0) steps[q] = 1;
                if (steps[q] < 0) {
                    int64_t dim = -1;
                    if (!l.in.empty() && q < l.in[0].dims.size()) dim = l.in[0].dims[q];
                    if (dim > 0 && q < starts.size() && q < ends.size()) {
                        int64_t s = starts[q] < 0 ? dim + starts[q] : starts[q];
                        int64_t e = ends[q] < 0 ? dim + ends[q] : ends[q];
                        e = std::max<int64_t>(0, std::min<int64_t>(dim - 1, e));
                        s = std::max<int64_t>(0, std::min<int64_t>(dim - 1, s));
                        starts[q] = e;
                        ends[q] = s + 1;
                        steps[q] = 1;
                    } else {
                        starts.assign(1, 0);
                        ends.assign(1, std::numeric_limits<int32_t>::max());
                        axes.assign(1, 0);
                        steps.assign(1, 1);
                        break;
                    }
                }
            }
            while (l.in.size() > 1) l.in.pop_back();
            if (std::getenv("PF_NPU_SLICE_SAFE")) {
                starts.assign(1, 0);
                ends.assign(1, 1000000);
                axes.assign(1, 0);
                steps.assign(1, 1);
            }
            if (std::getenv("PF_NPU_STRIDED")) {
                l.type = "StridedSlice";
                l.version = "opset1";
                l.data.clear();
                l.data.push_back({"begin_mask", "0"});
                l.data.push_back({"end_mask", "0"});
                l.data.push_back({"new_axis_mask", ""});
                l.data.push_back({"shrink_axis_mask", ""});
                l.data.push_back({"ellipsis_mask", ""});
                while (l.in.size() > 1) l.in.pop_back();
                std::vector<std::vector<int64_t>> svals = {starts, ends, steps};
                const char * ssuf[3] = {"_begin", "_end", "_stride"};
                for (int q = 0; q < 3; ++q) {
                    std::string cn = (n.name.empty() ? name_of(ni) : n.name) + ssuf[q];
                    emit_scalar_const(cn, onnxpb::INT64, svals[q].data(), svals[q].size() * 8,
                                      {static_cast<int64_t>(svals[q].size())});
                    port p;
                    p.id = q + 1;
                    p.dtype = onnxpb::INT64;
                    p.dims = {static_cast<int64_t>(svals[q].size())};
                    p.names = cn;
                    l.in.push_back(p);
                }
                for (auto & p : l.out) p.id = 4;
            } else {
            std::vector<std::vector<int64_t>> vals = {starts, ends, axes, steps};
            const char * suff[4] = {"_start", "_end", "_axis", "_step"};
            for (int q = 0; q < 4; ++q) {
                std::string cn = (n.name.empty() ? name_of(ni) : n.name) + suff[q];
                emit_scalar_const(cn, onnxpb::INT64, vals[q].data(), vals[q].size() * 8,
                                  {static_cast<int64_t>(vals[q].size())});
                port p;
                p.id = q + 1;
                p.dtype = onnxpb::INT64;
                p.dims = {static_cast<int64_t>(vals[q].size())};
                p.names = cn;
                l.in.push_back(p);
            }
            for (auto & p : l.out) p.id = 5;
            }
        } else if (op == "HardSigmoid") {
            float alpha = static_cast<float>(attr_f(n, "alpha", 0.2));
            float beta = static_cast<float>(attr_f(n, "beta", 0.5));
            const std::string base = n.name.empty() ? name_of(ni) : n.name;
            const std::string an = base + "_alpha", bn2 = base + "_beta";
            emit_scalar_const(an, onnxpb::FLOAT, &alpha, 4, {});
            emit_scalar_const(bn2, onnxpb::FLOAT, &beta, 4, {});
            port pa;
            pa.id = 1;
            pa.dtype = onnxpb::FLOAT;
            pa.names = an;
            l.in.push_back(pa);
            port pb;
            pb.id = 2;
            pb.dtype = onnxpb::FLOAT;
            pb.names = bn2;
            l.in.push_back(pb);
            for (auto & p : l.out) p.id = 3;
        } else if (op == "Clip") {
            l.data.push_back({"min", std::to_string(attr_f(n, "min", -std::numeric_limits<float>::max()))});
            l.data.push_back({"max", std::to_string(attr_f(n, "max", std::numeric_limits<float>::max()))});
            while (l.in.size() > 1) l.in.pop_back();
            for (size_t k = 0; k < l.in.size(); ++k) l.in[k].id = static_cast<int>(k);
            for (auto & p : l.out) p.id = static_cast<int>(l.in.size());
        } else if (op == "LeakyRelu") {
            l.data.push_back({"negative_slope", std::to_string(attr_f(n, "alpha", 0.01))});
        } else if (op == "Cast") {
            l.data.push_back({"element_type", element_type(static_cast<int32_t>(attr_i(n, "to", onnxpb::FLOAT)))});
        } else if (op == "Gemm") {
            l.data.push_back({"transpose_a", attr_i(n, "transA", 0) ? "true" : "false"});
            l.data.push_back({"transpose_b", attr_i(n, "transB", 0) ? "true" : "false"});
            l.data.push_back({"alpha", std::to_string(attr_f(n, "alpha", 1.0))});
            l.data.push_back({"beta", std::to_string(attr_f(n, "beta", 1.0))});
        } else if (op == "ReduceMax" || op == "ReduceMin" || op == "ReduceSum" || op == "ReduceMean" ||
                   op == "GlobalAveragePool" || op == "GlobalMaxPool") {
            auto axes = attr_ints(n, "axes");
            if (axes.empty() && (op == "GlobalAveragePool" || op == "GlobalMaxPool")) axes = {2, 3};
            if (axes.empty() && n.input.size() > 1 && init.count(n.input[1])) axes = read_ints(*init[n.input[1]]);
            l.data.push_back({"axes", join(axes)});
            l.data.push_back({"keep_dims", attr_i(n, "keepdims", 1) ? "true" : "false"});
            while (l.in.size() > 1) l.in.pop_back();
            std::vector<int64_t> ax(axes.begin(), axes.end());
            if (ax.empty()) ax.assign(2, 0);
            std::string an = (n.name.empty() ? name_of(ni) : n.name) + "_axes";
            emit_scalar_const(an, onnxpb::INT64, ax.data(), ax.size() * 8,
                              {static_cast<int64_t>(ax.size())});
            port ap;
            ap.id = 1;
            ap.dtype = onnxpb::INT64;
            ap.dims = {static_cast<int64_t>(ax.size())};
            ap.names = an;
            l.in.push_back(ap);
            for (auto & p : l.out) p.id = 2;
        } else if (op == "Einsum") {
            l.data.push_back({"equation", attr_s(n, "equation", "")});
        } else if (op == "Shape") {
            l.data.push_back({"output_type", "i64"});
        } else if (op == "ConstantOfShape") {
            const auto * st = n.input.empty() ? nullptr : resolve_const(n.input[0]);
            std::vector<int64_t> shp;
            if (st) shp = read_ints(*st);
            size_t count = 1;
            for (auto d : shp) {
                if (d <= 0 || d > 1000000) { count = 0; break; }
                count *= static_cast<size_t>(d);
                if (count > 10000000) { count = 0; break; }
            }
            if (count == 0) { shp.assign(1, 1); count = 1; }
            const onnxpb::tensor_proto * vp = nullptr;
            for (const auto & a : n.attribute) if (a.type_ == onnxpb::attribute::type::t) { vp = &a.t; break; }
            std::vector<uint8_t> payload;
            size_t elem_bytes = 4;
            if (vp && !vp->raw.empty()) {
                size_t nelem = 1;
                for (auto d : vp->dims) nelem *= d > 0 ? static_cast<size_t>(d) : 1;
                elem_bytes = std::max<size_t>(1, vp->raw.size() / nelem);
                payload.resize(count * elem_bytes);
                std::vector<uint8_t> one(vp->raw.begin(), vp->raw.begin() + std::min(elem_bytes, vp->raw.size()));
                one.resize(elem_bytes, 0);
                for (size_t q = 0; q < count; ++q)
                    std::memcpy(payload.data() + q * elem_bytes, one.data(), elem_bytes);
            } else {
                payload.assign(count * 4, 0);
            }
            std::string cname = n.output.empty() ? (n.name.empty() ? name_of(ni) : n.name) : n.output[0];
            while (l.in.size() > 0) l.in.pop_back();
            l.out.clear();
            l.type = "Const";
            l.data.push_back({"element_type", vp ? element_type(vp->data_type) : "f32"});
            l.data.push_back({"shape", data_shape(shp)});
            l.data.push_back({"offset", std::to_string(out.weights.size())});
            l.data.push_back({"size", std::to_string(payload.size())});
            port p;
            p.id = 0;
            p.dtype = vp ? vp->data_type : onnxpb::FLOAT;
            p.dims = shp;
            p.names = cname;
            l.out.push_back(p);
            l.id = static_cast<int>(layers.size());
            producer[cname] = {l.id, 0};
            layers.push_back(std::move(l));
            out.weights.insert(out.weights.end(), payload.begin(), payload.end());
            while (out.weights.size() % 64 != 0) out.weights.push_back(0);
            continue;
        } else if (op == "Expand") {
            l.data.push_back({"mode", "bidirectional"});
            if (n.input.size() > 1) {
                const auto * t = resolve_const(n.input[1]);
                (void)t;
            }
        } else if (op == "GridSample") {
            l.data.push_back({"align_corners", attr_i(n, "align_corners", 0) ? "true" : "false"});
            l.data.push_back({"mode", attr_s(n, "mode", "bilinear")});
            l.data.push_back({"padding_mode", attr_s(n, "padding_mode", "zeros")});
        } else if (op == "Range") {
            int32_t dt = n.input.empty() ? onnxpb::FLOAT : (shapes.count(n.input[0]) ? shapes[n.input[0]].dtype : onnxpb::FLOAT);
            l.data.push_back({"output_type", element_type(dt)});
        } else if (op == "ScatterND") {
            l.data.push_back({"reduction", "none"});
        } else if (op == "Resize") {
            l.data.push_back({"mode", attr_s(n, "mode", "nearest")});
            l.data.push_back({"shape_calculation_mode", "scales"});
            l.data.push_back({"coordinate_transformation_mode", attr_s(n, "coordinate_transformation_mode", "asymmetric")});
            l.data.push_back({"nearest_mode", attr_s(n, "nearest_mode", "floor")});
            l.data.push_back({"antialias", "false"});
            l.data.push_back({"pads_begin", "0, 0, 0, 0"});
            l.data.push_back({"pads_end", "0, 0, 0, 0"});
            l.data.push_back({"cube_coeff", "-0.75"});
            while (l.in.size() > 1) l.in.pop_back();
            std::vector<float> scales;
            if (n.input.size() > 2) {
                auto it = init.find(n.input[2]);
                if (it != init.end() && it->second->data_type == onnxpb::FLOAT) {
                    scales.resize(it->second->raw.size() / 4);
                    std::memcpy(scales.data(), it->second->raw.data(), scales.size() * 4);
                }
            }
            if (scales.empty()) scales.assign(4, 1.0f);
            std::string sname = (n.name.empty() ? name_of(ni) : n.name) + "_scales";
            emit_scalar_const(sname, onnxpb::FLOAT, scales.data(), scales.size() * 4,
                              {static_cast<int64_t>(scales.size())});
            port sp;
            sp.id = 1;
            sp.dtype = onnxpb::FLOAT;
            sp.dims = {static_cast<int64_t>(scales.size())};
            sp.names = sname;
            l.in.push_back(sp);
            for (auto & p : l.out) p.id = 2;
        } else if (op == "LayerNormalization") {
            l.data.push_back({"epsilon", std::to_string(attr_f(n, "epsilon", 1e-5))});
            l.data.push_back({"axis", std::to_string(attr_i(n, "axis", -1))});
        } else if (op == "TopK") {
            l.data.push_back({"axis", std::to_string(attr_i(n, "axis", -1))});
            l.data.push_back({"mode", "max"});
        } else if (op == "Trilu") {
            l.data.push_back({"upper", attr_i(n, "upper", 1) ? "true" : "false"});
        } else if (op == "Pad") {
            while (l.in.size() > 1) l.in.pop_back();
            size_t pad_rank = l.in.empty() ? 4 : l.in[0].dims.size();
            if (pad_rank == 0) pad_rank = 4;
            std::vector<int64_t> pb(pad_rank, 0), pe(pad_rank, 0);
            if (n.input.size() > 1 && !std::getenv("PF_NPU_PAD_ZERO")) {
                const_evaluator ce(g);
                const_tensor pt;
                if (ce.eval(n.input[1], pt)) {
                    std::vector<int64_t> pv;
                    for (size_t q = 0; q < pt.count(); ++q) pv.push_back(pt.elem_i(q));
                    if (pv.size() >= 2 * pad_rank) {
                        for (size_t q = 0; q < pad_rank; ++q) { pb[q] = pv[q]; pe[q] = pv[pad_rank + q]; }
                    }
                }
            }
            std::vector<float> zero(1, 0.0f);
            const std::string base = n.name.empty() ? name_of(ni) : n.name;
            emit_scalar_const(base + "_pads_begin", onnxpb::INT64, pb.data(), pb.size()*8, {static_cast<int64_t>(pb.size())});
            emit_scalar_const(base + "_pads_end", onnxpb::INT64, pe.data(), pe.size()*8, {static_cast<int64_t>(pe.size())});
            emit_scalar_const(base + "_pad_value", onnxpb::FLOAT, zero.data(), 4, {});
            const char * ps[3] = {"_pads_begin", "_pads_end", "_pad_value"};
            for (int q = 0; q < 3; ++q) {
                port p;
                p.id = q + 1;
                p.dtype = q < 2 ? onnxpb::INT64 : onnxpb::FLOAT;
                p.dims = q < 2 ? std::vector<int64_t>{static_cast<int64_t>(pb.size())} : std::vector<int64_t>{};
                p.names = base + ps[q];
                l.in.push_back(p);
            }
            for (auto & p : l.out) p.id = 4;
            auto pd = shapes.count(n.output[0]) ? shapes[n.output[0]] : shape_info();
            if (!l.in.empty() && !l.in[0].dims.empty()) {
                pd.dims = l.in[0].dims;
                for (size_t q = 0; q < pd.dims.size() && q < pad_rank; ++q)
                    if (pd.dims[q] > 0) pd.dims[q] += pb[q] + pe[q];
            }
            shapes[n.output[0]] = pd;
        } else if (op == "Clip" || op == "Where" || op == "Reshape" || op == "Expand") {
            // NPU accepts the standard OV operand order; no extra attributes are required.
        }

        l.id = static_cast<int>(layers.size());
        const std::vector<int64_t> & in0 = l.in.empty() ? std::vector<int64_t>() : l.in[0].dims;
        const std::vector<int64_t> & in1 = l.in.size() > 1 ? l.in[1].dims : std::vector<int64_t>();
        for (size_t k = 0; k < l.out.size(); ++k) {
            const std::string & on = n.output[k];
            shape_info s;
            s.dtype = onnxpb::FLOAT;
            if (!in0.empty()) s.dtype = shapes.count(n.input.empty() ? "" : n.input[0]) ? shapes[n.input[0]].dtype : onnxpb::FLOAT;
            if (op == "Conv" || op == "ConvTranspose") {
                std::vector<int64_t> wd = in1;
                int64_t M = wd.empty() ? 1 : wd[0];
                auto strides = attr_ints(n, "strides");
                auto pads = attr_ints(n, "pads");
                auto kernel = attr_ints(n, "kernel_shape");
                if (kernel.empty() && wd.size() >= 4) kernel = {wd[2], wd[3]};
                if (strides.empty()) strides = {1, 1};
                int64_t H = in0.size() >= 3 ? in0[in0.size() - 2] : 1;
                int64_t W = in0.size() >= 4 ? in0.back() : 1;
                int64_t kh = kernel.empty() ? 1 : kernel[0];
                int64_t kw = kernel.size() > 1 ? kernel[1] : 1;
                int64_t ph = pads.size() >= 4 ? pads[0] + pads[2] : 0;
                int64_t pw = pads.size() >= 4 ? pads[1] + pads[3] : 0;
                int64_t dil = attr_i(n, "dilations", 1);
                int64_t khd = (kh - 1) * dil + 1;
                int64_t kwd = (kw - 1) * dil + 1;
                int64_t sh = strides[0], sw = strides.size() > 1 ? strides[1] : 1;
                int64_t oh = (H + ph - khd) / sh + 1;
                int64_t ow = (W + pw - kwd) / sw + 1;
                s.dims = {in0.empty() ? 1 : in0[0], M, oh, ow};
            } else if (op == "MatMul" || op == "Gemm") {
                int64_t M = in0.empty() ? 1 : in0[0];
                int64_t N = in1.size() >= 2 ? in1[1] : (in1.empty() ? 1 : in1[0]);
                if (op == "Gemm" && attr_i(n, "transB", 0)) N = in1.empty() ? 1 : in1[0];
                s.dims = {M, N};
            } else if (op == "MaxPool" || op == "AveragePool") {
                auto kernel = attr_ints(n, "kernel_shape");
                auto strides = attr_ints(n, "strides");
                auto pads = attr_ints(n, "pads");
                if (strides.empty()) strides = {1, 1};
                int64_t H = in0.size() >= 3 ? in0[in0.size() - 2] : 1;
                int64_t W = in0.size() >= 4 ? in0.back() : 1;
                int64_t kh = kernel.empty() ? 1 : kernel[0];
                int64_t kw = kernel.size() > 1 ? kernel[1] : 1;
                int64_t ph = pads.size() >= 4 ? pads[0] + pads[2] : 0;
                int64_t pw = pads.size() >= 4 ? pads[1] + pads[3] : 0;
                s.dims = in0;
                if (s.dims.size() >= 2) {
                    bool ceil_mode = attr_i(n, "ceil_mode", 0) != 0;
                    int64_t sh2 = strides[0], sw2 = strides.size() > 1 ? strides[1] : 1;
                    int64_t dh = H + ph - kh, dw = W + pw - kw;
                    if (ceil_mode) {
                        dh = (dh + sh2 - 1) / sh2;
                        dw = (dw + sw2 - 1) / sw2;
                    } else {
                        dh /= sh2;
                        dw /= sw2;
                    }
                    s.dims[s.dims.size() - 2] = dh + 1;
                    s.dims.back() = dw + 1;
                }
            } else if (op == "GlobalAveragePool" || op == "GlobalMaxPool") {
                s.dims = in0.size() >= 2 ? std::vector<int64_t>(in0.begin(), in0.begin() + 2) : in0;
            } else if (op == "Reshape") {
                const auto * ct = resolve_const(n.input[1]);
                if (ct) s.dims = read_ints(*ct);
            } else if (op == "Transpose") {
                auto perm = attr_ints(n, "perm");
                s.dims.assign(in0.size(), 1);
                for (size_t q = 0; q < perm.size() && q < in0.size(); ++q)
                    s.dims[q] = in0[perm[q] >= 0 && static_cast<size_t>(perm[q]) < in0.size() ? perm[q] : q];
            } else if (op == "Flatten") {
                int64_t axis = attr_i(n, "axis", 1);
                int64_t prod = 1;
                for (size_t q = static_cast<size_t>(axis); q < in0.size(); ++q) prod *= in0[q];
                s.dims = {axis > 0 ? in0[0] : 1, prod};
            } else if (op == "Concat") {
                s.dims = in0;
                int64_t axis = attr_i(n, "axis", 1);
                int64_t sum = 0;
                for (const auto & in_name : n.input) {
                    if (init.count(in_name)) continue;
                    const auto & d = shapes.count(in_name) ? shapes[in_name].dims : std::vector<int64_t>();
                    if (axis >= 0 && static_cast<size_t>(axis) < d.size()) sum += d[axis];
                }
                if (axis >= 0 && static_cast<size_t>(axis) < s.dims.size()) s.dims[axis] = sum;
            } else if (op == "Expand") {
                if (n.input.size() > 1 && init.count(n.input[1])) s.dims = read_ints(*init[n.input[1]]);
                else s.dims = broadcast(in0, in1);
            } else if (op == "Squeeze" || op == "Unsqueeze") {
                s.dims = in0;
            } else if (op == "ReduceMax" || op == "ReduceMin" || op == "ReduceSum" || op == "ReduceMean") {
                s.dims = in0;
            } else {
                s.dims = in0.empty() ? in1 : (in1.empty() ? in0 : broadcast(in0, in1));
                if (n.input.empty() && n.attribute.size() && n.attribute[0].type_ == onnxpb::attribute::type::t) {
                    s.dims = n.attribute[0].t.dims;
                }
            }
            if (!shapes.count(on) || shapes[on].dims.empty()) shapes[on] = s;
            l.out[k].dims = shapes[on].dims;
            if (shapes.count(on)) l.out[k].dtype = shapes[on].dtype;
            producer[on] = {l.id, l.out[k].id};
        }
        if (op == "MaxPool" && l.out.size() == 1) {
            port idxp = l.out[0];
            idxp.id = l.out[0].id + 1;
            idxp.dtype = onnxpb::INT64;
            idxp.names.clear();
            l.out.push_back(idxp);
        }
        if (!l.out.empty() && !l.out[0].names.empty()) l.name = l.out[0].names;
        layers.push_back(std::move(l));
        if (!extra_bias_name.empty() && !n.output.empty()) {
            const std::string cvname = (n.name.empty() ? name_of(ni) : n.name) + "_cvout";
            if (!layers.empty() && !layers.back().out.empty()) {
                layers.back().out[0].names = cvname;
                producer[cvname] = {layers.back().id, layers.back().out[0].id};
            }
            layer add;
            add.id = static_cast<int>(layers.size());
            add.name = n.output[0];
            add.type = "Add";
            add.version = "opset1";
            for (int k = 0; k < 2; ++k) {
                port p;
                p.id = k;
                p.dtype = onnxpb::FLOAT;
                p.dims = shapes.count(k == 0 ? cvname : extra_bias_name)
                             ? shapes[k == 0 ? cvname : extra_bias_name].dims
                             : std::vector<int64_t>{};
                p.names = k == 0 ? cvname : extra_bias_name;
                add.in.push_back(p);
            }
            port outp;
            outp.id = 2;
            outp.dtype = onnxpb::FLOAT;
            outp.dims = shapes.count(n.output[0]) ? shapes[n.output[0]].dims : std::vector<int64_t>{};
            outp.names = n.output[0];
            add.out.push_back(outp);
            producer[n.output[0]] = {add.id, outp.id};
            layers.push_back(std::move(add));
        }
        if (fit != folds.end() && !g.node[fit->second.bn_node].output.empty() && !std::getenv("PF_NPU_NO_BN_BIAS")) {
            const std::string & bn_out = g.node[fit->second.bn_node].output[0];
            const std::string conv_out_name = (n.name.empty() ? name_of(ni) : n.name) + "_convout";
            if (!layers.empty() && !layers.back().out.empty()) {
                layers.back().out[0].names = conv_out_name;
                producer[conv_out_name] = {layers.back().id, layers.back().out[0].id};
            }
            layer add;
            add.id = static_cast<int>(layers.size());
            add.name = bn_out;
            add.type = "Add";
            add.version = "opset1";
            port in0;
            in0.id = 0;
            in0.dtype = onnxpb::FLOAT;
            in0.dims = shapes.count(bn_out) ? shapes[bn_out].dims : std::vector<int64_t>{};
            in0.names = conv_out_name;
            add.in.push_back(in0);
            port in1;
            in1.id = 1;
            in1.dtype = onnxpb::FLOAT;
            in1.dims = {1, static_cast<int64_t>(fit->second.bias.size() / 4), 1, 1};
            in1.names = fit->second.bias_name;
            add.in.push_back(in1);
            port outp;
            outp.id = 2;
            outp.dtype = onnxpb::FLOAT;
            outp.dims = shapes.count(bn_out) ? shapes[bn_out].dims : std::vector<int64_t>{};
            outp.names = bn_out;
            add.out.push_back(outp);
            producer[bn_out] = {add.id, outp.id};
            layers.push_back(std::move(add));
        }
    }

    for (size_t i = 0; i < g.output.size(); ++i) {
        const auto & v = g.output[i];
        layer l;
        l.id = static_cast<int>(layers.size());
        l.name = "result_" + std::to_string(i);
        l.type = "Result";
        l.data.push_back({"output_names", v.name});
        port p;
        p.id = 0;
        p.dtype = shapes[v.name].dtype;
        p.dims = shapes[v.name].dims;
        p.names = v.name;
        l.in.push_back(p);
        layers.push_back(std::move(l));
        out.outputs.push_back(v.name);
    }

    for (auto & l : layers) {
        for (auto & pp : l.in)
            for (auto & d : pp.dims) if (d < -1) d = -1;
        for (auto & pp : l.out)
            for (auto & d : pp.dims) if (d < -1) d = -1;
    }
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    xml << "<net name=\"" << esc(g.name.empty() ? "onnx" : g.name) << "\" version=\"11\">\n<layers>\n";
    std::map<int, std::set<int>> self_ports;
    for (const auto & l : layers) {
        if (l.type == "Parameter" || l.type == "Const") continue;
        for (const auto & p : l.in) {
            if (p.names.empty()) continue;
            auto it = producer.find(p.names);
            if (it != producer.end() && it->second.first == l.id) self_ports[l.id].insert(p.id);
        }
    }
    std::set<std::string> used_names;
    for (const auto & l : layers) {
        std::string lname = l.name;
        if (used_names.count(lname)) lname += "_" + std::to_string(l.id);
        used_names.insert(lname);
        xml << "<layer id=\"" << l.id << "\" name=\"" << esc(lname) << "\" type=\"" << l.type
            << "\" version=\"" << l.version << "\">\n";
        if (!l.data.empty()) {
            xml << "<data";
            for (const auto & kv : l.data) xml << " " << kv.first << "=\"" << esc(kv.second) << "\"";
            xml << "/>\n";
        }
        std::string ports_xml;
        std::vector<port> filtered_in;
        auto si = self_ports.find(l.id);
        for (const auto & p : l.in) if (si == self_ports.end() || !si->second.count(p.id)) filtered_in.push_back(p);
        write_ports(ports_xml, "input", filtered_in, false);
        write_ports(ports_xml, "output", l.out, l.type != "Const");
        xml << ports_xml << "</layer>\n";
    }
    xml << "</layers>\n<edges>\n";
    for (const auto & l : layers) {
        if (l.type == "Parameter" || l.type == "Const") continue;
        for (const auto & p : l.in) {
            const std::string & name = p.names;
            if (name.empty()) continue;
            auto it = producer.find(name);
            if (it == producer.end()) {
                if (std::getenv("PF_NPU_IRDBG")) std::fprintf(stderr, "no producer for %s (layer %d)\n", name.c_str(), l.id);
                continue;
            }
            if (it->second.first == l.id) continue;
            xml << "<edge from-layer=\"" << it->second.first << "\" from-port=\"" << it->second.second
                << "\" to-layer=\"" << l.id << "\" to-port=\"" << p.id << "\"/>\n";
        }
    }
    xml << "</edges>\n<rt_info/>\n</net>\n";
    out.xml = xml.str();
    if (const char * sd = std::getenv("PF_NPU_STATIC_DIMS")) {
        const std::string rep = sd;
        std::string replaced;
        const std::string from = "<dim>-1</dim>";
        const std::string to = "<dim>" + rep + "</dim>";
        size_t pos = 0;
        while ((pos = out.xml.find(from, pos)) != std::string::npos) {
            replaced += out.xml.substr(0, pos) + to;
            out.xml = replaced + out.xml.substr(pos + from.size());
            pos = 0;
            replaced.clear();
        }
        std::string res;
        pos = 0;
        while (true) {
            size_t a = out.xml.find("shape=\"", pos);
            if (a == std::string::npos) { res += out.xml.substr(pos); break; }
            size_t b = out.xml.find('"', a + 7);
            if (b == std::string::npos) { res += out.xml.substr(pos); break; }
            std::string attr = out.xml.substr(a, b - a + 1);
            std::string fixed;
            for (char c : attr) fixed += c == '?' ? rep : std::string(1, c);
            res += out.xml.substr(pos, a - pos) + fixed;
            pos = b + 1;
        }
        out.xml = res;
    }
    return true;
}

}  // namespace npu
}  // namespace si
