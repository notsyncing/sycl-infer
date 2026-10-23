#include "graph.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>

#include "ops.h"

namespace si {
namespace onnx {

namespace {

// A tensor value in the build phase.
struct build_value {
    tensor t;
    bool is_init = false;
};

} // namespace

// The weight of a convolution, transposed to (C/g, kh, kw, M/g) so that the
// output channels of one (c, kh, kw) are contiguous.
tensor transpose_conv_weight(const tensor & w, int64_t group) {
    tensor out;
    out.dtype = w.dtype;
    // The weight is (M, C/group, kh, kw): its second dimension is *already*
    // the per-group channel count.  (Dividing it by the group again is what
    // breaks a depthwise convolution, whose weight is (M, 1, kh, kw) with
    // group = M.)
    const int64_t M = w.shape[0];
    const int64_t cg = w.shape[1];
    const int64_t kh = w.shape[2];
    const int64_t kw = w.shape[3];
    const int64_t mg = M / group;
    // The layout is (group * C/g, kh, kw, M/g): the groups keep their own
    // weights.  A layout of (C/g, kh, kw, M/g) drops the group, and a
    // depthwise convolution (group = C, C/g = 1, M/g = 1) then has one set
    // of nine weights for all its channels.
    const int64_t C = group * cg;
    out.shape = {C, kh, kw, mg};
    out.count = (size_t)(C * kh * kw * mg);
    out.on_host = true;
    out.host.resize(out.count * 4);
    const float * src = (const float *)w.host.data();
    float * dst = (float *)out.host.data();
    for (int64_t g = 0; g < group; ++g) {
        for (int64_t c = 0; c < cg; ++c) {
            for (int64_t r = 0; r < kh; ++r) {
                for (int64_t s = 0; s < kw; ++s) {
                    for (int64_t m = 0; m < mg; ++m) {
                        // The source is (M, C/g, kh, kw): the output
                        // channels come first, then the group's input
                        // channels.
                        const int64_t si = ((((g * mg + m) * cg + c) * kh + r) * kw) + s;
                        const int64_t di = (((((g * cg + c) * kh + r) * kw + s) * mg) + m);
                        dst[di] = src[si];
                    }
                }
            }
        }
    }
    return out;
}

// The weight of a transposed convolution, transposed to (kh, kw, C, M/g).
tensor transpose_conv_transpose_weight(const tensor & w, int64_t group) {
    tensor out;
    out.dtype = w.dtype;
    // The weight is (C, M/group, kh, kw).
    const int64_t C = w.shape[0];
    const int64_t Mg = w.shape[1];
    const int64_t kh = w.shape[2];
    const int64_t kw = w.shape[3];
    const int64_t mg = Mg / group;
    const int64_t cg = C / group;
    out.shape = {kh, kw, C, mg};
    out.count = (size_t)(kh * kw * C * mg);
    out.on_host = true;
    out.host.resize(out.count * 4);
    const float * src = (const float *)w.host.data();
    float * dst = (float *)out.host.data();
    for (int64_t g = 0; g < group; ++g) {
        for (int64_t c = 0; c < cg; ++c) {
            for (int64_t r = 0; r < kh; ++r) {
                for (int64_t s = 0; s < kw; ++s) {
                    for (int64_t m = 0; m < mg; ++m) {
                        const int64_t si = ((((g * cg + c) * Mg + (g * mg + m)) * kh + r) * kw + s);
                        const int64_t di = ((((r * kw + s) * C) + (g * cg + c)) * mg + m);
                        dst[di] = src[si];
                    }
                }
            }
        }
    }
    return out;
}

namespace {

const char * const kActivations[] = {"Relu", "Sigmoid", "Tanh",      "HardSigmoid",
                                     "HardSwish", "Gelu", "Silu", "Clip"};

} // namespace

exec_graph build_graph(const std::string & path, bool verbose) {
    const onnxpb::model_proto model = onnxpb::parse_model(path);
    exec_graph g = build_graph_from_graph(model.graph, model.opset, path, verbose);
    for (const auto & kv : model.metadata) g.metadata[kv.first] = kv.second;
    return g;
}

// The builder works on a graph, so that a Loop's body - which is a graph -
// is built the same way as the model.
exec_graph build_graph_from_graph(const onnxpb::graph_proto & graph, int64_t opset,
                                  const std::string & path, bool verbose,
                                  const exec_graph * scope) {
    exec_graph g;
    g.opset = opset;
    if (getenv("SI_ONNX_LOOPDBG")) {
        fprintf(stderr, "[onnx] the builder: %s, %zu nodes, %zu initializers, %zu inputs\n",
                path.c_str(), graph.node.size(), graph.initializer.size(), graph.input.size());
    }

    // The named values.
    std::map<std::string, int> slot;
    std::vector<std::string> names;
    std::vector<build_value> values;
    auto add_value = [&](const std::string & name) -> int {
        auto it = slot.find(name);
        if (it != slot.end()) return it->second;
        const int s = (int)values.size();
        slot[name] = s;
        names.push_back(name);
        values.push_back(build_value());
        return s;
    };
    // The enclosing graph's names: ONNX lets a subgraph read the values of
    // the graph that holds it, and the formula recogniser's Loop body does.
    std::map<std::string, int> scope_slot;
    if (scope) {
        for (size_t i = 0; i < scope->tensors.size(); ++i) {
            scope_slot.emplace(scope->tensors[i].name, (int)i);
        }
    }
    // The input of an operator: the value it reads is the subgraph's own if
    // the subgraph has it, and the *enclosing* graph's otherwise.
    auto resolve_input = [&](const std::string & name) -> int {
        auto it = slot.find(name);
        if (it != slot.end()) return it->second;
        auto sit = scope_slot.find(name);
        if (sit != scope_slot.end()) {
            const int s = add_value(name);
            g.outer_refs.push_back({s, sit->second});
            // The shape and the type come from the enclosing graph, so the
            // shape functions of the subgraph see them.  The *value* does
            // not: it is copied in when the subgraph runs, and it changes
            // between the calls.
            const tensor & st = scope->tensors[sit->second];
            values[(size_t)s].t.name = name;
            values[(size_t)s].t.dtype = st.dtype;
            values[(size_t)s].t.shape = st.shape;
            values[(size_t)s].t.count = st.count;
            return s;
        }
        return add_value(name);
    };

    // An initializer: a constant value of the graph.
    auto add_initializer = [&](const onnxpb::tensor_proto & t) {
        const int s = add_value(t.name);
        build_value & v = values[s];
        v.t.name = t.name;
        v.t.dtype = t.data_type;
        v.t.shape = t.dims;
        v.t.count = 1;
        for (int64_t d : t.dims) v.t.count *= (size_t)d;
        v.t.on_host = true;
        v.t.host = t.raw;
        if (!t.external_data.empty() && v.t.host.empty()) {
            const size_t slash = path.find_last_of('/');
            const std::string dir = slash == std::string::npos ? std::string(".") : path.substr(0, slash);
            const std::string loc = dir + "/" + t.external_data;
            FILE * f = fopen(loc.c_str(), "rb");
            if (!f) throw std::runtime_error("cannot open the external data " + loc);
            fseek(f, 0, SEEK_END);
            const long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            v.t.host.resize((size_t)sz);
            const size_t rd = fread(v.t.host.data(), 1, (size_t)sz, f);
            (void)rd;
            fclose(f);
        }
        // A boolean is a float 0/1 in this runtime, so its bytes are
        // converted from the model's one-byte-per-element form.
        if (t.data_type == onnxpb::BOOL) {
            v.t.host = bool_to_runtime_bytes(v.t.host);
        }
        const size_t want = v.t.count * runtime_elem_size(t.data_type);
        if (v.t.host.size() < want) v.t.host.resize(want, 0);
        v.is_init = true;
    };
    for (const onnxpb::tensor_proto & t : graph.initializer) add_initializer(t);
    for (const onnxpb::value_info & vi : graph.input) {
        const int s = add_value(vi.name);
        build_value & v = values[s];
        v.t.name = vi.name;
        v.t.dtype = vi.elem_type;
        v.t.shape = vi.shape;
    }

    const std::vector<onnxpb::node_proto> & nodes = graph.node;

    // The constant analysis.
    std::map<std::string, bool> is_const;
    for (const auto & kv : slot) {
        is_const[kv.first] = values[kv.second].is_init;
    }
    std::vector<bool> node_const(nodes.size(), false);
    for (size_t i = 0; i < nodes.size(); ++i) {
        const onnxpb::node_proto & n = nodes[i];
        if (n.op_type == "Constant") {
            node_const[i] = true;
            continue;
        }
        bool all = true;
        for (const std::string & in : n.input) {
            if (in.empty()) continue;
            auto it = is_const.find(in);
            if (it == is_const.end() || !it->second) {
                all = false;
                break;
            }
        }
        node_const[i] = all;
    }

    // The consumer of a value (the first one).
    std::map<std::string, int> consumer;
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (const std::string & in : nodes[i].input) {
            if (!in.empty() && consumer.find(in) == consumer.end()) consumer[in] = (int)i;
        }
    }

    // The fusion: a convolution followed by the multiplication and the
    // addition of a folded batch norm, and by an activation.
    std::vector<bool> dead(nodes.size(), false);
    std::map<size_t, std::vector<float>> conv_scale, conv_bias;
    std::map<size_t, std::string> conv_act;
    std::map<size_t, std::vector<float>> conv_act_params;
    // The values that a fused operator produced: they are the convolution's
    // output under another name, and the consumers are rewritten to use it.
    std::map<std::string, std::string> alias;
    const bool fuse = getenv("SI_ONNX_NOFUSE") == nullptr;
    // Whether a node reads a value.
    auto reads = [&](const onnxpb::node_proto & n, const std::string & v) {
        for (const std::string & in : n.input) {
            if (in == v) return true;
        }
        return false;
    };
    for (size_t i = 0; i < nodes.size() && fuse; ++i) {
        const onnxpb::node_proto & n = nodes[i];
        if (n.op_type != "Conv" && n.op_type != "ConvTranspose") continue;
        std::string cur = n.output[0];
        int guard = 0;
        while (guard++ < 64) {
            auto cit = consumer.find(cur);
            if (cit == consumer.end()) break;
            const int ci = cit->second;
            const onnxpb::node_proto & cn = nodes[ci];
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
            // convolution's, so a consumer that reads both of them would
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
                    auto it = slot.find(cn.input[k]);
                    if (it == slot.end() || !values[it->second].is_init) continue;
                    bn.push_back(&values[it->second]);
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
                conv_scale[i] = scale;
                conv_bias[i] = bias;
                dead[ci] = true;
                if (!cn.output.empty()) alias[cn.output[0]] = cur;
                cur = cn.output[0];
                continue;
            }
            if ((op == "Mul" || op == "Add") && node_const[ci]) {
                int which = -1;
                for (size_t k = 0; k < cn.input.size(); ++k) {
                    if (cn.input[k].empty()) continue;
                    auto it = is_const.find(cn.input[k]);
                    if (it != is_const.end() && it->second) {
                        which = (int)k;
                        break;
                    }
                }
                if (which < 0) break;
                const int vs = slot[cn.input[which]];
                const build_value & v = values[vs];
                if (v.t.host.size() < 4) break;
                const int64_t n_ch = v.t.count;
                std::vector<float> vals((size_t)n_ch);
                std::memcpy(vals.data(), v.t.host.data(), (size_t)n_ch * 4);
                if (op == "Mul") {
                    conv_scale[i] = vals;
                } else {
                    conv_bias[i] = vals;
                }
                dead[ci] = true;
                if (!cn.output.empty()) alias[cn.output[0]] = cur;
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
                    hardsigmoid_params(opset, alpha, beta, p);
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
                        auto it = slot.find(cn.input[k]);
                        if (it == slot.end() || !values[it->second].is_init ||
                            values[it->second].t.host.size() < 4) {
                            foldable = false;
                            break;
                        }
                        const float v = *(const float *)values[it->second].t.host.data();
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
                conv_act[i] = op;
                conv_act_params[i] = params;
                dead[ci] = true;
                if (!cn.output.empty()) alias[cn.output[0]] = cur;
                cur = cn.output[0];
                continue;
            }
            break;
        }
    }

    // The executable ops.  The nodes are copied into the graph, because the
    // operators point at their attributes.
    g.nodes = nodes;
    const size_t outer_nodes = g.nodes.size();
    // The If operators' branches are appended to the node list, and the
    // operators point *into* it: the list must not move, or every pointer
    // taken so far dangles.  So the room is reserved before anything is
    // built.
    {
        size_t extra = 0;
        for (size_t i = 0; i < outer_nodes; ++i) {
            if (g.nodes[i].op_type != "If") continue;
            for (const onnxpb::attribute & a : g.nodes[i].attribute) {
                if (a.g) extra += a.g->node.size();
            }
        }
        // + the pass-through copies an If's branch may need (an empty
        // branch declares a constant as its output, and the copy that writes
        // the If's slot is a node too).  A reallocation here would leave
        // every operator's `node` pointer dangling.
        g.nodes.reserve(outer_nodes + extra + 64);
    }
    // The values that exist: the initializers, the graph's inputs and the
    // outputs of the operators that are not dead.  An operator that reads
    // anything else reads a value nothing produces.
    std::set<std::string> produced, is_init, is_graph_input;
    for (const onnxpb::tensor_proto & t : graph.initializer) is_init.insert(t.name);
    for (const onnxpb::value_info & vi : graph.input) is_graph_input.insert(vi.name);
    for (size_t i = 0; i < outer_nodes; ++i) {
        if (dead[i]) continue;
        for (const std::string & o : g.nodes[i].output) {
            if (!o.empty()) produced.insert(o);
        }
    }
    // The operator of one node.  The If operators call it for their
    // branches' nodes too, which is why it is a function.
    auto make_op = [&](size_t idx) {
        const onnxpb::node_proto & n = g.nodes[idx];
        exec_op op;
        op.node = &g.nodes[idx];
        op.kind = op_kind::generic;
        for (const std::string & in : n.input) {
            // An omitted optional input is an empty name: it keeps its
            // place, so the operator's input positions stay right.
            if (in.empty()) {
                op.in.push_back(-1);
                continue;
            }
            // A value that a fused operator produced is the convolution's
            // output under another name.
            std::string name = in;
            // The chain can be long (a convolution, then a batch norm, then
            // a scale, then an activation, ...), and stopping early leaves
            // the consumer reading a value that the fusion removed.
            for (int guard = 0; guard < 64 && alias.count(name); ++guard) {
                name = alias[name];
            }
            if (getenv("SI_ONNX_TRACE") && !produced.count(name) && !is_init.count(name) &&
                !is_graph_input.count(name)) {
                fprintf(stderr, "[onnx] the graph: '%s' reads '%s', which nothing produces\n",
                        n.name.c_str(), name.c_str());
            }
            op.in.push_back(resolve_input(name));
        }
        for (const std::string & out : n.output) {
            if (out.empty()) continue;
            op.out.push_back(add_value(out));
        }
        if (n.op_type == "Conv" || n.op_type == "ConvTranspose") {
            op.kind = op_kind::conv;
            int64_t grp = 1;
            for (const onnxpb::attribute & a : n.attribute) {
                if (a.name == "group") grp = a.i;
            }
            const int wi = op.in.size() > 1 ? op.in[1] : -1;
            if (wi < 0) throw std::runtime_error("a convolution without a weight");
            const tensor w = values[wi].t;
            if (verbose) {
                fprintf(stderr, "[onnx] conv %s: the weight slot %d holds %zu bytes, dims",
                        n.name.c_str(), wi, w.host.size());
                for (size_t k = 0; k < w.shape.size(); ++k) {
                    fprintf(stderr, " %lld", (long long)w.shape[k]);
                }
                fprintf(stderr, " group %lld\n", (long long)grp);
            }
            if (!w.host.empty()) {
                values[wi].t = n.op_type == "Conv" ? transpose_conv_weight(w, grp)
                                                  : transpose_conv_transpose_weight(w, grp);
                values[wi].t.name = w.name;
                op.weight_transposed = true;
            }
            // The fused batch norm and the activation ride along as extra
            // inputs: in[3] is the scale, in[4] the bias.
            auto add_const = [&](const std::vector<float> & v, const char * tag) -> int {
                const std::string nm = std::string(n.name.empty() ? n.op_type : n.name) + "." + tag;
                const int s = add_value(nm);
                build_value & bv = values[s];
                bv.t.name = nm;
                bv.t.dtype = onnxpb::FLOAT;
                bv.t.shape = {(int64_t)v.size()};
                bv.t.count = v.size();
                bv.t.on_host = true;
                bv.t.host.resize(v.size() * 4);
                std::memcpy(bv.t.host.data(), v.data(), v.size() * 4);
                bv.is_init = true;
                return s;
            };
            auto sit = conv_scale.find(idx);
            if (sit != conv_scale.end()) op.in.push_back(add_const(sit->second, "scale"));
            auto bit = conv_bias.find(idx);
            if (bit != conv_bias.end()) op.in.push_back(add_const(bit->second, "bias"));
            auto ait = conv_act.find(idx);
            if (ait != conv_act.end()) op.activation = ait->second;
            auto apit = conv_act_params.find(idx);
            if (apit != conv_act_params.end()) op.act_params = apit->second;
            if (getenv("SI_ONNX_TRACE") && !op.activation.empty()) {
                fprintf(stderr, "[onnx] the fused %s: the activation %s", n.name.c_str(),
                        op.activation.c_str());
                for (size_t k = 0; k < op.act_params.size(); ++k) {
                    fprintf(stderr, " %g", op.act_params[k]);
                }
                fprintf(stderr, "\n");
            }
        }
        return op;
    };
    for (size_t i = 0; i < outer_nodes; ++i) {
        if (dead[i]) continue;
        const onnxpb::node_proto & n = g.nodes[i];
        if (n.op_type != "If") {
            g.ops.push_back(make_op(i));
            continue;
        }
        // The If: its two branches are inlined into the op list, and only the
        // selected one runs.
        //
        // The exporter's sub-graphs are not self-contained: a branch reads the
        // values of the graph that holds it (the wireless table structurer's
        // decoder reads a carried value), so a branch cannot be a graph of its
        // own - it would read values nothing produces.  Inlined, the branch
        // shares the slots and the values flow.
        //
        // The branch's outputs are the If's outputs, by position: the
        // exporter gives them different names ('_generated_var_1' against
        // '_generated_var_3'), so the producing node's output is renamed, and
        // the old name stays as an alias for a branch that reads its own
        // output.
        exec_op op = make_op(i);
        // The If itself has no output: the branch's producing node writes it.
        op.out.clear();
        const size_t if_op = g.ops.size();
        g.ops.push_back(op);
        for (const onnxpb::attribute & a : n.attribute) {
            if (a.name != "then_branch" && a.name != "else_branch") continue;
            if (!a.g) continue;
            const onnxpb::graph_proto & bg = *a.g;
            // The branch's own initializers: the weights it uses are its
            // own, not the enclosing graph's, and the branch is inlined so
            // they have to come with it.
            for (const onnxpb::tensor_proto & t : bg.initializer) add_initializer(t);
            // Where this branch's nodes begin: before the pass-through
            // copies below, so they are part of the branch too (they must run
            // only when this branch is selected).
            const size_t node_begin = g.nodes.size();
            // The branch's outputs: the If's, by position.
            //
            // A branch output that no node of the branch produces is a value
            // of the *enclosing* scope - the Paddle export's empty branch
            // declares a constant as its output.  ONNX says the If's output
            // is the selected branch's output, so the value has to be copied
            // into the If's slot; without it nothing writes the slot and it
            // keeps whatever the other branch left there (a false condition's
            // zero, for ever).
            for (size_t k = 0; k < bg.output.size() && k < n.output.size(); ++k) {
                const std::string & old_name = bg.output[k].name;
                const std::string & new_name = n.output[k];
                if (old_name == new_name) continue;
                bool produced = false;
                for (const onnxpb::node_proto & bn : bg.node) {
                    for (const std::string & o : bn.output) {
                        if (o == old_name) produced = true;
                    }
                }
                const int s = add_value(new_name);
                if (!produced) {
                    onnxpb::node_proto id;
                    id.op_type = "Identity";
                    id.input.push_back(old_name);
                    id.output.push_back(new_name);
                    g.nodes.push_back(id);
                    // **No alias.**  The alias maps the branch's output name
                    // to the If's slot, for a branch that reads its own
                    // output; here the name is the *enclosing scope's* value,
                    // and aliasing it would make the copy read the If's own
                    // (stale) output.
                    continue;
                }
                slot[old_name] = s;   // the alias, for an internal read
            }
            // The branch's nodes.
            for (const onnxpb::node_proto & bn : bg.node) {
                onnxpb::node_proto copy = bn;
                for (std::string & o : copy.output) {
                    for (size_t k = 0; k < bg.output.size() && k < n.output.size(); ++k) {
                        if (bg.output[k].name == o && o != n.output[k]) o = n.output[k];
                    }
                }
                g.nodes.push_back(copy);
            }
            const size_t op_begin = g.ops.size();
            // The branch's operators: the nodes that were just appended.
            for (size_t k = node_begin; k < g.nodes.size(); ++k) {
                exec_op bop = make_op(k);
                // The branch's operators are not folded: only one of the two
                // branches runs, so a value they produce is not a constant
                // (the other branch would produce a different one).
                bop.in_if_branch = true;
                g.ops.push_back(bop);
            }
            const int ob = (int)op_begin;
            const int oe = (int)g.ops.size();
            if (a.name == "then_branch") {
                op.if_then_begin = ob;
                op.if_then_end = oe;
            } else {
                op.if_else_begin = ob;
                op.if_else_end = oe;
            }
            if (getenv("SI_ONNX_LOOPDBG")) {
                fprintf(stderr, "[onnx] the builder: the If at the node %zu: %s ops [%d, %d)\n", i,
                        a.name.c_str(), ob, oe);
            }
        }
        g.ops[if_op] = op;
    }

    for (const onnxpb::value_info & vi : graph.output) {
        std::string name = vi.name;
        for (int guard = 0; guard < 64 && alias.count(name); ++guard) {
            name = alias[name];
        }
        g.output_slots.push_back(add_value(name));
        // Does a node produce it?  A Loop's body may declare an output that
        // no node produces: ONNXRuntime returns the body's input of the same
        // name unchanged (a pass-through).
        g.output_produced.push_back(produced.count(name) > 0);
    }
    for (const onnxpb::value_info & vi : graph.input) {
        auto it = slot.find(vi.name);
        if (it == slot.end()) continue;
        if (values[it->second].is_init) continue;
        g.input_slots.push_back(it->second);
    }

    g.tensors.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        g.tensors[i] = values[i].t;
        g.tensors[i].name = names[i];
    }
    if (getenv("SI_ONNX_LOOPDBG")) {
        fprintf(stderr, "[onnx] the builder: %s: %zu ops, %zu tensors\n", path.c_str(), g.ops.size(),
                g.tensors.size());
    }
    if (verbose) {
        fprintf(stderr, "[onnx] %s: %zu ops, %zu tensors, opset %lld\n", path.c_str(), g.ops.size(),
                g.tensors.size(), (long long)g.opset);
    }
    return g;
}

} // namespace onnx
} // namespace si
