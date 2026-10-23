#include "transform.h"
#include "transform_util.h"

#include <cstdlib>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace si {
namespace onnx {
namespace transforms {

bool mask_box(graph_build & g) {
    bool changed = false;
    if (g.fuse && !getenv("SI_ONNX_NOMASKFUSE")) {
        auto uof = [&](const std::string & name) -> const std::vector<int> & {
            static const std::vector<int> empty;
            auto it = g.users.find(name);
            return it == g.users.end() ? empty : it->second;
        };
        auto to_int = [&](const onnxpb::node_proto & n) -> int64_t {
            for (const onnxpb::attribute & a : n.attribute) {
                if (a.name == "to") return a.i;
            }
            return -1;
        };
        auto keepdims = [&](const onnxpb::node_proto & n) -> int64_t {
            for (const onnxpb::attribute & a : n.attribute) {
                if (a.name == "keepdims") return a.i;
            }
            return 1;
        };
        auto axes_are = [&](const onnxpb::node_proto & n,
                           const std::vector<int64_t> & want) -> bool {
            for (const onnxpb::attribute & a : n.attribute) {
                if (a.name == "axes") return a.ints == want;
            }
            return false;
        };
        auto sole_after_reshape = [&](const std::string & v, const char * want) -> int {
            for (int ci : uof(v)) {
                const onnxpb::node_proto & cn = g.nodes[(size_t)ci];
                if (cn.op_type == want) return ci;
                if (cn.op_type == "Reshape" && cn.output.size() == 1) {
                    for (int ri : uof(cn.output[0])) {
                        if (g.nodes[(size_t)ri].op_type == want) return ri;
                    }
                }
            }
            return -1;
        };
        struct mask_path {
            int mul = -1;
            int rmax = -1;
            int rmin = -1;
            std::string grid;
            std::string max_name;
            std::string min_name;
        };
        bool found = false;
        for (size_t mi = 0; mi < g.nodes.size() && !found; ++mi) {
            if (g.dead[mi] || g.nodes[mi].op_type != "MatMul") continue;
            const onnxpb::node_proto & mm = g.nodes[mi];
            if (mm.output.empty()) continue;
            const std::string & mm_out = mm.output[0];
            if (uof(mm_out).size() != 1) continue;
            const int r1 = uof(mm_out)[0];
            if (g.nodes[(size_t)r1].op_type != "Reshape" || g.nodes[(size_t)r1].output.empty()) continue;
            if (uof(g.nodes[(size_t)r1].output[0]).size() != 1) continue;
            const int gt = uof(g.nodes[(size_t)r1].output[0])[0];
            const onnxpb::node_proto & gn = g.nodes[(size_t)gt];
            if (gn.op_type != "Greater" || gn.input.size() < 2 || gn.output.empty()) continue;
            double threshold = 0.0;
            if (!splat(g, gn.input[1], threshold, 0)) {
                continue;
            }
            const std::string & gt_out = gn.output[0];
            std::string cf, ci_name;
            int ci = -1, any_rmax = -1, any_cast = -1;
            for (int ui : uof(gt_out)) {
                const onnxpb::node_proto & un = g.nodes[(size_t)ui];
                if (un.op_type != "Cast") continue;
                const int64_t to = to_int(un);
                if (to == onnxpb::FLOAT) cf = un.output[0];
                if (to == onnxpb::INT32 || to == onnxpb::INT64) {
                    ci = ui;
                    ci_name = un.output[0];
                }
            }
            if (cf.empty() || ci < 0) {
                continue;
            }
            any_rmax = sole_after_reshape(ci_name, "ReduceMax");
            if (any_rmax < 0) {
                continue;
            }
            const onnxpb::node_proto & arn = g.nodes[(size_t)any_rmax];
            if (arn.output.empty() || keepdims(arn) != 0 || !axes_are(arn, {2, 3})) {
                continue;
            }
            for (int ui : uof(arn.output[0])) {
                const onnxpb::node_proto & un = g.nodes[(size_t)ui];
                if (un.op_type == "Cast" && to_int(un) == onnxpb::BOOL) any_cast = ui;
            }
            if (any_cast < 0) {
                continue;
            }
            std::vector<mask_path> paths;
            double sentinel = 0.0;
            bool ok = true;
            for (int ui : uof(cf)) {
                const onnxpb::node_proto & mul = g.nodes[(size_t)ui];
                if (mul.op_type != "Mul" || mul.input.size() < 2 || mul.output.empty()) continue;
                std::string grid;
                for (const std::string & in : mul.input) {
                    if (in == cf) continue;
                    auto it = g.slot.find(in);
                    if (it == g.slot.end() || !g.values[it->second].is_init) continue;
                    const tensor & grid_t = g.values[it->second].t;
                    if (grid_t.dtype == onnxpb::FLOAT && grid_t.shape.size() == 2 &&
                        grid_t.count > 1) {
                        grid = in;
                    }
                }
                if (grid.empty()) continue;
                mask_path p;
                p.mul = ui;
                p.grid = grid;
                p.rmax = sole_after_reshape(mul.output[0], "ReduceMax");
                if (p.rmax < 0 || g.nodes[(size_t)p.rmax].output.empty() ||
                    keepdims(g.nodes[(size_t)p.rmax]) != 0 ||
                    !axes_are(g.nodes[(size_t)p.rmax], {-1})) {
                    continue;
                }
                p.max_name = g.nodes[(size_t)p.rmax].output[0];
                // The min path: Mul -> Add(+0) -> Where -> Reshape -> ReduceMin.
                int add1 = -1;
                for (int ai : uof(mul.output[0])) {
                    const onnxpb::node_proto & an = g.nodes[(size_t)ai];
                    if (an.op_type != "Add" || an.input.size() < 2) continue;
                    bool other_zero = false;
                    for (const std::string & in : an.input) {
                        if (in == mul.output[0]) continue;
                        double z = 0.0;
                        if (splat(g, in, z, 0) && z == 0.0) other_zero = true;
                    }
                    if (other_zero) {
                        add1 = ai;
                        break;
                    }
                }
                if (add1 < 0) {
                    ok = false;
                    break;
                }
                int wh = -1;
                for (int wi : uof(g.nodes[(size_t)add1].output[0])) {
                    if (g.nodes[(size_t)wi].op_type == "Where") wh = wi;
                }
                if (wh < 0) {
                    ok = false;
                    break;
                }
                const onnxpb::node_proto & wn = g.nodes[(size_t)wh];
                if (wn.input.size() < 3) {
                    ok = false;
                    break;
                }
                // The condition is Cast(bool, Add(mask_float, 0)).
                auto cond_it = g.producer_of.find(wn.input[0]);
                if (cond_it == g.producer_of.end()) {
                    ok = false;
                    break;
                }
                const onnxpb::node_proto & cond = g.nodes[(size_t)cond_it->second];
                if (cond.op_type != "Cast" || to_int(cond) != onnxpb::BOOL || cond.input.empty()) {
                    ok = false;
                    break;
                }
                auto cond_add_it = g.producer_of.find(cond.input[0]);
                if (cond_add_it == g.producer_of.end()) {
                    ok = false;
                    break;
                }
                const onnxpb::node_proto & cond_add = g.nodes[(size_t)cond_add_it->second];
                if (cond_add.op_type != "Add") {
                    ok = false;
                    break;
                }
                bool has_cf = false, has_zero = false;
                for (const std::string & in : cond_add.input) {
                    if (in == cf) has_cf = true;
                    double z = 0.0;
                    if (splat(g, in, z, 0) && z == 0.0) has_zero = true;
                }
                if (!has_cf || !has_zero) {
                    ok = false;
                    break;
                }
                // The false branch is Add(1e8, 0).
                auto false_it = g.producer_of.find(wn.input[2]);
                if (false_it == g.producer_of.end()) {
                    ok = false;
                    break;
                }
                const onnxpb::node_proto & fn = g.nodes[(size_t)false_it->second];
                if (fn.op_type != "Add") {
                    ok = false;
                    break;
                }
                bool has_sentinel = false, has_false_zero = false;
                double this_sentinel = 0.0;
                for (const std::string & in : fn.input) {
                    double v = 0.0;
                    if (!splat(g, in, v, 0)) continue;
                    if (v == 0.0) has_false_zero = true;
                    else {
                        has_sentinel = true;
                        this_sentinel = v;
                    }
                }
                if (!has_sentinel || !has_false_zero) {
                    ok = false;
                    break;
                }
                if (sentinel == 0.0) sentinel = this_sentinel;
                if (this_sentinel != sentinel) {
                    ok = false;
                    break;
                }
                p.rmin = sole_after_reshape(wn.output[0], "ReduceMin");
                if (p.rmin < 0 || g.nodes[(size_t)p.rmin].output.empty() ||
                    keepdims(g.nodes[(size_t)p.rmin]) != 0 ||
                    !axes_are(g.nodes[(size_t)p.rmin], {-1})) {
                    ok = false;
                    break;
                }
                p.min_name = g.nodes[(size_t)p.rmin].output[0];
                paths.push_back(p);
            }
            if (!ok || paths.size() != 2) {
                continue;
            }
            if (sentinel == 0.0) {
                continue;
            }
            const std::string any_name = g.nodes[(size_t)any_cast].output[0];
            const std::string & max_a = paths[0].max_name;
            const std::string & max_b = paths[1].max_name;
            const std::string & min_a = paths[0].min_name;
            const std::string & min_b = paths[1].min_name;
            const std::string & grid_a = paths[0].grid;
            const std::string & grid_b = paths[1].grid;

            // Everything in the cone below the five outputs is g.dead.  The
            // cone stops at the MatMul (the bmm is still computed) and at
            // this Greater (it becomes the fused operator).
            std::set<int> cone;
            std::function<void(const std::string &)> mark = [&](const std::string & v) {
                auto pit = g.producer_of.find(v);
                if (pit == g.producer_of.end()) return;
                const int p = pit->second;
                if (p == (int)mi || p == gt) return;
                if (!cone.insert(p).second) return;
                for (const std::string & in : g.nodes[(size_t)p].input) mark(in);
            };
            mark(any_name);
            mark(max_a);
            mark(max_b);
            mark(min_a);
            mark(min_b);
            // The reshape in front of the Greater only reinterprets the
            // bmm's [1,300,40000] as [1,300,200,200].  It dies with the
            // fused scan, but its *target* is built from Shape(add.40) -
            // and following that target would walk into the whole decoder
            // and mark it g.dead.  The target chain is small and stays; only
            // the reshape itself is removed.
            cone.insert(r1);
            std::set<std::string> fused_out = {any_name, max_a, max_b, min_a, min_b};
            bool safe = true;
            for (int p : cone) {
                for (const std::string & out : g.nodes[(size_t)p].output) {
                    if (out.empty() || fused_out.count(out)) continue;
                    if (g.graph_outputs.count(out)) {
                        safe = false;
                        break;
                    }
                    for (int u : uof(out)) {
                        if (u == (int)mi || u == gt || cone.count(u) || g.dead[(size_t)u]) continue;
                        safe = false;
                        break;
                    }
                    if (!safe) break;
                }
                if (!safe) break;
            }
            if (!safe) {
                continue;
            }
            for (int p : cone) g.dead[(size_t)p] = true;
            // The Greater becomes the fused scan.  Its inputs are the bmm
            // and the two coordinate grids; the threshold and the empty-mask
            // sentinel ride as attributes.
            onnxpb::node_proto fused;
            fused.name = gn.name.empty() ? "MaskBox" : gn.name;
            fused.op_type = "MaskBox";
            fused.input = {mm_out, grid_a, grid_b};
            fused.output = {any_name, max_a, max_b, min_a, min_b};
            onnxpb::attribute at;
            at.name = "threshold";
            at.type_ = onnxpb::attribute::type::f;
            at.f = (float)threshold;
            fused.attribute.push_back(at);
            onnxpb::attribute as;
            as.name = "sentinel";
            as.type_ = onnxpb::attribute::type::f;
            as.f = (float)sentinel;
            fused.attribute.push_back(as);
            g.nodes[(size_t)gt] = fused;
            if (getenv("SI_ONNX_TRACE")) {
                fprintf(stderr,
                        "[onnx] the fusion: %s (mask head: rows from '%s', grids '%s'/'%s', "
                        "sentinel %g) -> one scan\n",
                        fused.name.c_str(), mm_out.c_str(), grid_a.c_str(), grid_b.c_str(), sentinel);
            }
            changed = true;
        }
    }


    return changed;
}

} // namespace transforms
} // namespace onnx
} // namespace si
