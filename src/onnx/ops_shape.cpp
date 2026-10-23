#include <cstdio>

#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>

#include "kernels.h"

namespace si {
namespace onnx {

// ------------------------------------------------------------- the shapes

bool shape_elementwise(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    std::vector<std::vector<int64_t>> in;
    for (const tensor * t : c.in) in.push_back(t->shape);
    out.assign(c.out.size(), broadcast_shapes(in));
    return true;
}

// The Constant: the value is the attribute's tensor.
bool shape_constant(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    std::vector<int64_t> shape;
    int32_t dt = onnxpb::FLOAT;
    for (const onnxpb::attribute & a : c.node.attribute) {
        if (a.name != "value" || a.type_ != onnxpb::attribute::type::t) continue;
        shape = a.t.dims;
        dt = a.t.data_type;
    }
    for (tensor * t : c.out) {
        t->shape = shape;
        t->dtype = dt;
    }
    out.assign(c.out.size(), shape);
    return true;
}

bool shape_softmax(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    out.assign(c.out.size(), c.in[0]->shape);
    return true;
}

// The CumSum: the input's shape with the axis dim kept (the default of the
// `keepdims` attribute is 1).
bool shape_cumsum(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    // The CumSum does **not** reduce: the output has the input's shape and
    // the values are the cumulative sums along the axis.  It is not a Reduce
    // - it has no `keepdims` attribute, and the axis stays where it is.
    //
    // (Treating it as a reduction is what made the formula recogniser's loop
    // never end: its mask of the classes seen so far was summed along the
    // wrong axis, so the count of each class collapsed to one element and
    // the "every class has been seen" test could not become true.)
    out.assign(c.out.size(), c.in[0]->shape);
    return true;
}

bool shape_layer_norm(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    out.assign(c.out.size(), c.in[0]->shape);
    return true;
}

bool shape_conv(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    if (getenv("SI_ONNX_TRACE")) {
        fprintf(stderr, "[onnx] the conv's shape: the input '%s' [", x->name.c_str());
        for (int64_t d : x->shape) fprintf(stderr, " %lld", (long long)d);
        fprintf(stderr, "]\n");
    }
    const tensor * w = c.in[1];
    const int64_t N = x->shape[0];
    const int64_t C = x->shape[1];
    const int64_t H = x->shape[2];
    const int64_t W = x->shape[3];
    // The weight is transposed to (C/g, kh, kw, M/g).
    const int64_t group = c.attr_i("group", 1);
    const int64_t M = w->shape[3] * group;
    const int64_t kg = w->shape[1];
    const int64_t kg2 = w->shape[2];
    std::vector<int64_t> strides = c.attr_ints("strides");
    std::vector<int64_t> pads = c.attr_ints("pads");
    std::vector<int64_t> dil = c.attr_ints("dilations");
    const int64_t sh = strides.size() == 2 ? strides[0] : 1;
    const int64_t sw = strides.size() == 2 ? strides[1] : 1;
    // The pads are (top, left, bottom, right) or (top, left).
    const int64_t ph = pads.size() >= 2 ? pads[0] : 0;
    const int64_t pw = pads.size() >= 2 ? pads[1] : 0;
    const int64_t dh = dil.size() == 2 ? dil[0] : 1;
    const int64_t dw = dil.size() == 2 ? dil[1] : 1;
    const spatial_geom g = spatial_geometry(auto_pad_value(c), H, W, kg, kg2, sh, sw, ph, pw,
                                             dh, dw);
    out.assign(c.out.size(), {N, M, g.out_h, g.out_w});
    (void)C;
    return true;
}

bool shape_conv_transpose(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    const tensor * w = c.in[1];
    const int64_t N = x->shape[0];
    const int64_t H = x->shape[2];
    const int64_t W = x->shape[3];
    // The weight is transposed to (kh, kw, C, M/g).
    const int64_t group = c.attr_i("group", 1);
    const int64_t M = w->shape[3] * group;
    const int64_t kg = w->shape[0];
    const int64_t kg2 = w->shape[1];
    std::vector<int64_t> pads = c.attr_ints("pads");
    const int64_t ph = pads.size() >= 2 ? pads[0] : 0;
    const int64_t pw = pads.size() >= 2 ? pads[1] : 0;
    // The transposed convolution grows the output: the geometry is the
    // mirror of the forward one, plus the output padding.
    spatial_geom g;
    conv_transpose_geom(c, H, W, kg, kg2, &g.out_h, &g.out_w);
    g.pad_h = ph;
    g.pad_w = pw;
    out.assign(c.out.size(), {N, M, g.out_h, g.out_w});
    return true;
}

bool shape_pool(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    const int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    std::vector<int64_t> kernel = c.attr_ints("kernel_shape");
    std::vector<int64_t> strides = c.attr_ints("strides");
    std::vector<int64_t> pads = c.attr_ints("pads");
    const int64_t kh = kernel.size() == 2 ? kernel[0] : 1;
    const int64_t kw = kernel.size() == 2 ? kernel[1] : 1;
    const int64_t sh = strides.size() == 2 ? strides[0] : 1;
    const int64_t sw = strides.size() == 2 ? strides[1] : 1;
    const int64_t ph = pads.size() >= 2 ? pads[0] : 0;
    const int64_t pw = pads.size() >= 2 ? pads[1] : 0;
    const spatial_geom g =
        spatial_geometry(auto_pad_value(c), H, W, kh, kw, sh, sw, ph, pw, 1, 1);
    out.assign(c.out.size(), {N, C, g.out_h, g.out_w});
    return true;
}

bool shape_global_avg_pool(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    out.assign(c.out.size(), {x->shape[0], x->shape[1], 1, 1});
    return true;
}

bool shape_resize(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    // The sizes are an attribute, or a constant input.
    const std::vector<int64_t> attr_sizes = c.attr_ints("sizes");
    if (attr_sizes.size() == 4) {
        out.assign(c.out.size(), attr_sizes);
        return true;
    }
    // The Resize's inputs are (X, roi, scales, sizes): the scales and the
    // sizes are told apart by their *type*, not by their position (the roi
    // comes first and is usually empty).
    const tensor * scales = nullptr;
    const tensor * sizes = nullptr;
    for (size_t i = 1; i < c.in.size(); ++i) {
        const tensor * t = c.in[i];
        if (!t->on_host || t->count == 0) continue;
        if (t->dtype == onnxpb::INT64 && !sizes) sizes = t;
        else if (t->dtype == onnxpb::FLOAT && !scales) scales = t;
    }
    if (sizes) {
        const std::vector<int64_t> s = read_i64(sizes);
        if (s.size() != x->shape.size()) return false;
        out.assign(c.out.size(), s);
        return true;
    }
    if (!scales) return false;
    // The scales: the output is the input scaled by them.
    const std::vector<float> sc = read_f32(scales);
    if (sc.size() != x->shape.size()) return false;
    std::vector<int64_t> shape = x->shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        shape[i] = (int64_t)std::floor((double)shape[i] * (double)sc[i]);
    }
    out.assign(c.out.size(), shape);
    return true;
}

bool shape_reduce(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    std::vector<int64_t> axes = c.attr_ints("axes");
    std::vector<int64_t> shape = x->shape;
    int64_t keep = c.attr_i("keepdims", 1);
    if (axes.empty() && c.in.size() > 1 && c.in[1]->on_host) {
        // The axes are a constant input.
        const int64_t n = c.in[1]->count;
        for (int64_t i = 0; i < n; ++i) {
            int64_t a = 0;
            std::memcpy(&a, c.in[1]->host.data() + i * 8, 8);
            axes.push_back(a);
        }
    }
    if (axes.empty()) return false;
    const int64_t rank = (int64_t)shape.size();
    for (int64_t a : axes) {
        if (a < 0) a += rank;
        if (a < 0 || a >= rank) continue;
        if (keep) {
            shape[a] = 1;
        } else {
            shape.erase(shape.begin() + a);
        }
    }
    out.assign(c.out.size(), shape);
    return true;
}

// The GridSample: the output has the grid's spatial size.
bool shape_grid_sample(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    const tensor * grid = c.in[1];
    if (grid->shape.size() < 3) return false;
    // The grid is (N, Ho, Wo, 2).
    const int64_t N = x->shape[0], C = x->shape[1];
    const int64_t Ho = grid->shape[grid->shape.size() - 3];
    const int64_t Wo = grid->shape[grid->shape.size() - 2];
    out.assign(c.out.size(), {N, C, Ho, Wo});
    return true;
}

// The GatherND: the output's shape is the indices' shape without its last
// dim, followed by the data's shape from the last index on.
bool shape_gather_nd(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    const tensor * idx = c.in[1];
    if (idx->shape.empty()) return false;
    const int64_t k = idx->shape.back();
    if (k > (int64_t)x->shape.size()) return false;
    std::vector<int64_t> shape(idx->shape.begin(), idx->shape.end() - 1);
    for (int64_t i = k; i < (int64_t)x->shape.size(); ++i) shape.push_back(x->shape[i]);
    out.assign(c.out.size(), shape);
    return true;
}

// The ScatterND: the output has the data's shape.
bool shape_scatter_nd(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    out.assign(c.out.size(), c.in[0]->shape);
    return true;
}

// The Einsum: the output's shape follows the equation.
bool shape_einsum(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const std::string eq = c.attr_s("equation");
    const size_t arrow = eq.find("->");
    if (arrow == std::string::npos) return false;
    const std::string out_term = eq.substr(arrow + 2);
    // The dimension of every letter: the inputs must agree.
    std::map<char, int64_t> dims;
    size_t p = 0;
    int k = 0;
    while (p < arrow) {
        const size_t comma = eq.find(',', p);
        const size_t end = comma == std::string::npos || comma > arrow ? arrow : comma;
        const std::string term = eq.substr(p, end - p);
        const tensor * x = c.in[k++];
        if (term.size() != x->shape.size()) return false;
        for (size_t i = 0; i < term.size(); ++i) {
            const char l = term[i];
            auto it = dims.find(l);
            if (it == dims.end()) {
                dims[l] = x->shape[i];
            } else if (it->second != x->shape[i]) {
                return false;
            }
        }
        if (comma == std::string::npos || comma > arrow) break;
        p = comma + 1;
    }
    std::vector<int64_t> shape;
    for (char l : out_term) {
        auto it = dims.find(l);
        if (it == dims.end()) return false;
        shape.push_back(it->second);
    }
    out.assign(c.out.size(), shape);
    return true;
}

bool shape_concat(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    std::vector<const tensor *> ins;
    for (const tensor * t : c.in) ins.push_back(t);
    // The output's rank, and each input's shape after the promotion: a
    // rank-0 value concatenated along axis 0 becomes a rank-1 one (numpy's
    // rule, and Paddle's export relies on it).
    const concat_geom g = concat_geometry(ins, c.attr_i("axis", 0));
    if (!g.ok) return false;
    // The output is the inputs concatenated along the axis; the other axes
    // are broadcast (an input may have 1 there).
    std::vector<int64_t> shape = g.padded.empty() ? std::vector<int64_t>() : g.padded[0];
    int64_t sum = 0;
    for (const std::vector<int64_t> & s : g.padded) sum += s[g.axis];
    for (size_t i = 1; i < g.padded.size(); ++i) {
        const std::vector<int64_t> & s = g.padded[i];
        for (size_t d = 0; d < shape.size(); ++d) {
            if ((int64_t)d == g.axis) continue;
            if (s[d] != 1 && shape[d] != 1 && s[d] != shape[d]) return false;
            if (shape[d] == 1) shape[d] = s[d];
        }
    }
    shape[g.axis] = sum;
    out.assign(c.out.size(), shape);
    return true;
}

bool shape_matmul(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * a = c.in[0];
    const tensor * b = c.in[1];
    const int64_t ra = (int64_t)a->shape.size();
    const int64_t rb = (int64_t)b->shape.size();
    std::vector<int64_t> sa = a->shape, sb = b->shape;
    bool ta = false, tb = false;
    if (ra == 1) { ta = true; sa.insert(sa.begin(), 1); }
    if (rb == 1) { tb = true; sb.push_back(1); }
    if (ra == 2 && rb == 2) {
        out.assign(c.out.size(), {sa[0], sb[1]});
        return true;
    }
    // The batched case: the broadcast of the leading dims.
    std::vector<int64_t> lead(sa.begin(), sa.end() - 2);
    std::vector<int64_t> blead(sb.begin(), sb.end() - 2);
    const size_t l = std::max(lead.size(), blead.size());
    std::vector<int64_t> bl(l, 1);
    for (size_t i = 0; i < lead.size(); ++i) bl[l - lead.size() + i] = lead[i];
    for (size_t i = 0; i < blead.size(); ++i) {
        const size_t p = l - blead.size() + i;
        if (bl[p] != blead[i] && bl[p] != 1 && blead[i] != 1) return false;
        if (bl[p] == 1) bl[p] = blead[i];
    }
    if (getenv("SI_ONNX_TRACE")) {
        fprintf(stderr, "[onnx] the MatMul: a");
        for (int64_t d : a->shape) fprintf(stderr, " %lld", (long long)d);
        fprintf(stderr, "  b");
        for (int64_t d : b->shape) fprintf(stderr, " %lld", (long long)d);
        fprintf(stderr, "\n");
    }
    std::vector<int64_t> res = bl;
    res.push_back(sa[sa.size() - 2]);
    res.push_back(sb[sb.size() - 1]);
    if (ta) res.erase(res.end() - 2);
    if (tb) res.erase(res.end() - 1);
    out.assign(c.out.size(), res);
    return true;
}

bool shape_gemm(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * a = c.in[0];
    const tensor * b = c.in[1];
    const bool trans_a = c.attr_i("transA", 0) != 0;
    const bool trans_b = c.attr_i("transB", 0) != 0;
    const int64_t M = trans_a ? a->shape[1] : a->shape[0];
    const int64_t N = trans_b ? b->shape[0] : b->shape[1];
    out.assign(c.out.size(), {M, N});
    return true;
}

bool shape_same(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    out.assign(c.out.size(), c.in[0]->shape);
    return true;
}

// The ConstantOfShape: the output's shape is the *values* of the input
// (the dims), not the input's shape.
//
// The slanet model's loop-carried tensors are built this way, and the dims
// are computed at run time (a concat of a shape and a constant), so a
// shape function that reads the input's shape gives [3] where the value
// is [2, 49, 8] - and every consumer of it is then the wrong shape.
bool shape_constant_of_shape(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const std::vector<int64_t> dims = read_i64(c.in[0]);
    out.assign(c.out.size(), dims);
    // The type is the `value` attribute's, not the shape input's.  The shape
    // input is an int64 (it is a shape), and the value it produces is
    // whatever the attribute says - float32 by default, per the spec.
    //
    // This is not a detail: the wireless table structurer's GRU reads its
    // initial hidden state from a ConstantOfShape whose value is a float, and
    // with the int64 type the carried value is an integer - so the GRU's
    // MatMul takes the executor's default output type (the first input's)
    // and writes its float result into an int64 buffer.  The GRU's output
    // then reads back as garbage and its softmax comes out uniform.
    int32_t dt = onnxpb::FLOAT;
    for (const onnxpb::attribute & a : c.node.attribute) {
        if (a.name == "value" && a.type_ == onnxpb::attribute::type::t) dt = a.t.data_type;
    }
    for (tensor * t : c.out) t->dtype = dt;
    return true;
}

// The Cast: the shape is the input's, and the type is the "to" attribute.
bool shape_cast(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    out.assign(c.out.size(), c.in[0]->shape);
    int32_t to = onnxpb::FLOAT;
    for (const onnxpb::attribute & a : c.node.attribute) {
        if (a.name == "to") to = (int32_t)a.i;
    }
    // A BOOL is a float 0/1 in this runtime.
    //
    // The device holds float32, and a one-byte value would make every
    // consumer read a quarter of the bytes it should - the layout model's
    // decoder casts a float to BOOL to use it as a Where's condition, and
    // with a one-byte output the *next* operator reads a buffer four times
    // too small.  The value is the same (0 or 1) and the consumer only asks
    // "is it non-zero".
    if (to == onnxpb::BOOL) to = onnxpb::FLOAT;
    for (tensor * t : c.out) t->dtype = to;
    return true;
}

// The comparison operators: their output is a BOOL, which is a float 0/1
// here (see shape_cast).  The *value* inputs of a Where are the ones that
// decide the output's type, not the condition: the condition is a BOOL, and
// an output that takes the condition's type is a one-byte buffer that the
// next operator writes floats into.
bool shape_compare(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    std::vector<std::vector<int64_t>> in;
    for (const tensor * t : c.in) in.push_back(t->shape);
    out.assign(c.out.size(), broadcast_shapes(in));
    for (tensor * t : c.out) t->dtype = onnxpb::FLOAT;
    return true;
}

// The Shape operator: the input's dimensions, as an int64 vector.  (It is
// not the input's shape: the output is a one-dimensional vector of the
// rank's elements, and getting that wrong makes every consumer of it read
// the wrong number of bytes.)
bool shape_shape(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const int64_t rank = (int64_t)c.in[0]->shape.size();
    int64_t start = 0, end = rank;
    if (c.in.size() > 1 && c.in[1]->on_host && c.in[1]->count >= 2) {
        std::memcpy(&start, c.in[1]->host.data(), 8);
        std::memcpy(&end, c.in[1]->host.data() + 8, 8);
        if (start < 0) start += rank;
        if (end < 0) end += rank;
        start = std::max<int64_t>(0, std::min<int64_t>(start, rank));
        end = std::max<int64_t>(start, std::min<int64_t>(end, rank));
    }
    out.assign(c.out.size(), {end - start});
    for (tensor * t : c.out) t->dtype = onnxpb::INT64;
    return true;
}

bool shape_reshape(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    // The target shape is a constant input.
    if (c.in.size() < 2 || !c.in[1]->on_host) {
        if (getenv("SI_ONNX_TRACE")) {
            fprintf(stderr,
                    "[onnx] the reshape has no target: the inputs %zu, the target's on_host %d, the "
                    "target's dtype %d, the input's dtype %d, the host %zu\n",
                    c.in.size(), c.in.size() > 1 ? (int)c.in[1]->on_host : -1,
                    c.in.size() > 1 ? (int)c.in[1]->dtype : -1, c.in[0]->dtype,
                    c.in.size() > 1 ? c.in[1]->host.size() : 0);
            if (c.in.size() > 1) {
                fprintf(stderr, "    the target is '%s'\n", c.in[1]->name.c_str());
            }
        }
        return false;
    }
    std::vector<int64_t> shape = read_i64(c.in[1]);
    if (getenv("SI_ONNX_TRACE")) {
        fprintf(stderr, "[onnx] the reshape's target:");
        for (int64_t d : shape) fprintf(stderr, " %lld", (long long)d);
        fprintf(stderr, "  (the input's count %zu", c.in[0]->count);
        for (int64_t d : c.in[0]->shape) fprintf(stderr, " %lld", (long long)d);
        fprintf(stderr, ", the target's host %zu)\n", c.in[1]->host.size());
    }
    const bool allowzero = c.attr_i("allowzero", 0) != 0;
    // An empty target: the output is a scalar (a rank-0 value with one
    // element).  ONNX says so explicitly, and Paddle's export uses it.
    if (shape.empty()) {
        out.assign(c.out.size(), {});
        return true;
    }
    int64_t total = 1;
    for (int64_t d : c.in[0]->shape) total *= d;
    int64_t known = 1;
    int64_t neg = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0 && !allowzero) {
            shape[i] = c.in[0]->shape[i];
        } else if (shape[i] == -1) {
            neg = (int64_t)i;
        }
        if (shape[i] > 0) known *= shape[i];
    }
    if (neg >= 0) shape[neg] = total / (known ? known : 1);
    out.assign(c.out.size(), shape);
    return true;
}

bool shape_squeeze(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    std::vector<int64_t> shape = c.in[0]->shape;
    std::vector<int64_t> axes = c.attr_ints("axes");
    if (axes.empty() && c.in.size() > 1 && c.in[1]->on_host) {
        axes = read_i64(c.in[1]);
    }
    if (axes.empty()) {
        shape.erase(std::remove(shape.begin(), shape.end(), (int64_t)1), shape.end());
    } else {
        const int64_t rank = (int64_t)shape.size();
        std::vector<bool> drop(shape.size(), false);
        for (int64_t a : axes) {
            if (a < 0) a += rank;
            if (a >= 0 && a < rank) drop[a] = true;
        }
        std::vector<int64_t> ns;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (!drop[i]) ns.push_back(shape[i]);
        }
        shape = ns;
    }
    out.assign(c.out.size(), shape);
    return true;
}

bool shape_unsqueeze(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    std::vector<int64_t> shape = c.in[0]->shape;
    std::vector<int64_t> axes = c.attr_ints("axes");
    if (axes.empty() && c.in.size() > 1 && c.in[1]->on_host) {
        axes = read_i64(c.in[1]);
    }
    const int64_t rank = (int64_t)shape.size() + (int64_t)axes.size();
    for (int64_t a : axes) {
        if (a < 0) a += rank;
        shape.insert(shape.begin() + a, 1);
    }
    out.assign(c.out.size(), shape);
    return true;
}

bool shape_flatten(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    const int64_t rank = (int64_t)x->shape.size();
    int64_t axis = c.attr_i("axis", 1);
    if (axis < 0) axis += rank;
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= x->shape[i];
    for (int64_t i = axis; i < rank; ++i) inner *= x->shape[i];
    out.assign(c.out.size(), {outer, inner});
    return true;
}

bool shape_transpose(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    std::vector<int64_t> perm = c.attr_ints("perm");
    const int64_t rank = (int64_t)x->shape.size();
    if (perm.empty()) {
        for (int64_t i = 0; i < rank; ++i) perm.push_back(rank - 1 - i);
    }
    std::vector<int64_t> shape;
    for (int64_t p : perm) shape.push_back(x->shape[p]);
    out.assign(c.out.size(), shape);
    return true;
}

bool shape_gather(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    const tensor * idx = c.in[1];
    int64_t axis = c.attr_i("axis", 0);
    const int64_t rank = (int64_t)x->shape.size();
    if (axis < 0) axis += rank;
    std::vector<int64_t> shape;
    for (int64_t i = 0; i < axis; ++i) shape.push_back(x->shape[i]);
    for (int64_t d : idx->shape) shape.push_back(d);
    for (int64_t i = axis + 1; i < rank; ++i) shape.push_back(x->shape[i]);
    out.assign(c.out.size(), shape);
    return true;
}

bool shape_slice(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    // The starts, the ends and the axes are constant inputs.
    if (getenv("SI_ONNX_SHAPE")) {
        // The slice's inputs, one line: a slice of a rank-0 value is the
        // value itself, and the shape function has to say so.
        fprintf(stderr, "[onnx] the slice's shape: %zu inputs", c.in.size());
        for (size_t i = 0; i < c.in.size(); ++i) {
            const tensor * t = c.in[i];
            if (!t) {
                fprintf(stderr, " | in[%zu]: NULL", i);
                continue;
            }
            fprintf(stderr, " | in[%zu]: '%s' dtype=%d count=%zu host=%zu on_host=%d shape=[", i,
                    t->name.c_str(), (int)t->dtype, t->count, t->host.size(), (int)t->on_host);
            for (int64_t d : t->shape) fprintf(stderr, " %lld", (long long)d);
            fprintf(stderr, "]");
        }
        fprintf(stderr, "\n");
    }
    const tensor * x = c.in[0];
    if (c.in.size() < 3) return false;
    std::vector<int64_t> starts, ends, axes, steps;
    // The reader follows the tensor's type: the models mix int32 and int64
    // for the same kind of value.
    auto read = [&](const tensor * t, std::vector<int64_t> & v) {
        const bool is32 = t->dtype == onnxpb::INT32;
        const size_t esz = is32 ? 4 : 8;
        const int64_t n = std::min<int64_t>(t->count, (int64_t)(t->host.size() / esz));
        for (int64_t i = 0; i < n; ++i) {
            int64_t a = 0;
            if (is32) {
                int32_t b = 0;
                std::memcpy(&b, t->host.data() + i * 4, 4);
                a = b;
            } else {
                std::memcpy(&a, t->host.data() + i * 8, 8);
            }
            v.push_back(a);
        }
    };
    if (!c.in[1]->on_host || !c.in[2]->on_host) return false;
    read(c.in[1], starts);
    read(c.in[2], ends);
    if (c.in.size() > 3 && c.in[3]->on_host) read(c.in[3], axes);
    if (c.in.size() > 4 && c.in[4]->on_host) read(c.in[4], steps);
    std::vector<int64_t> shape = x->shape;
    const int64_t rank = (int64_t)shape.size();
    // A rank-0 value has no axis to slice: the output is the value itself.
    // (And shape[0] on an empty vector is a wild read - the crash that a
    // carried value with an empty shape produced.)
    if (rank == 0) {
        out.assign(c.out.size(), shape);
        return true;
    }
    for (size_t i = 0; i < starts.size(); ++i) {
        int64_t a = axes.size() > i ? axes[i] : (int64_t)i;
        if (a < 0) a += rank;
        if (a < 0 || a >= rank) a = 0;  // an out-of-range axis would read past the shape
        int64_t st = starts[i], en = ends[i];
        const int64_t step = steps.size() > i ? steps[i] : 1;
        if (st < 0) st += shape[a];
        if (en < 0) en += shape[a];
        if (step > 0) {
            st = std::max<int64_t>(0, std::min<int64_t>(st, shape[a]));
            en = std::max<int64_t>(0, std::min<int64_t>(en, shape[a]));
            shape[a] = (en - st + step - 1) / step;
        } else {
            st = std::max<int64_t>(0, std::min<int64_t>(st, shape[a] - 1));
            en = std::max<int64_t>(-1, std::min<int64_t>(en, shape[a] - 1));
            shape[a] = (st - en + (-step) - 1) / (-step);
        }
    }
    out.assign(c.out.size(), shape);
    return true;
}

// The ScatterElements: the output has the data's shape.  The updates are
// scattered into it along the axis.
bool shape_scatter_elements(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    out.assign(c.out.size(), c.in[0]->shape);
    return true;
}

// The OneHot: the indices' shape with the depth appended.
bool shape_one_hot(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    std::vector<int64_t> shape = c.in[0]->shape;
    int64_t depth = 0;
    if (c.in.size() > 1) {
        const std::vector<int64_t> v = read_i64(c.in[1]);
        if (!v.empty()) depth = v[0];
    }
    shape.push_back(depth);
    out.assign(c.out.size(), shape);
    // The type is the *values'* (the third input), not the indices': the
    // indices are int64 and the one-hot vector is the values' type.  (The
    // wireless table structurer's one-hot is concatenated with a float
    // score, and a one-hot written as int64 makes the concatenation
    // garbage - the decoder's argmax then picks a random column and the
    // loop never terminates.)
    int32_t dt = onnxpb::FLOAT;
    if (c.in.size() > 2) dt = c.in[2]->dtype;
    for (tensor * t : c.out) t->dtype = dt;
    return true;
}

bool shape_topk(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    // The k: an *input* since opset 10, an attribute before that.  Reading
    // only the attribute gives a k of 1, and a TopK of 1 is a model that
    // finds one box instead of a thousand.
    int64_t k = c.attr_i("k", 1);
    if (c.in.size() > 1 && c.in[1]->on_host) {
        const std::vector<int64_t> v = read_i64(c.in[1]);
        if (!v.empty()) k = v[0];
    }
    std::vector<int64_t> shape = x->shape;
    shape.back() = k;
    out.clear();
    out.push_back(shape);
    out.push_back(shape);
    // The second output is the *indices*: int64.  Without it the output
    // takes the values' type (float), and the operator that reads the
    // indices copies a float buffer as if it were int64 - half the bytes,
    // and the value is garbage.
    if (c.out.size() > 1) c.out[1]->dtype = onnxpb::INT64;
    return true;
}

bool shape_argmax(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    for (tensor * t : c.out) t->dtype = onnxpb::INT64;
    const tensor * x = c.in[0];
    int64_t axis = c.attr_i("axis", 0);
    const int64_t rank = (int64_t)x->shape.size();
    if (axis < 0) axis += rank;
    std::vector<int64_t> shape = x->shape;
    const bool keep = c.attr_i("keepdims", 1) != 0;
    if (keep) {
        shape[axis] = 1;
    } else {
        shape.erase(shape.begin() + axis);
    }
    out.assign(c.out.size(), shape);
    return true;
}

bool shape_split(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    int64_t axis = c.attr_i("axis", 0);
    const int64_t rank = (int64_t)x->shape.size();
    if (axis < 0) axis += rank;
    // The chunk sizes: an *input* since opset 13, an attribute before that.
    //
    // Reading only the attribute is not a rounding error: the layout model's
    // decoder splits the 13 125 encoder positions into the three feature
    // levels (10 000, 2 500 and 625), and dividing them equally gives
    // 4 375 each - a shape that does not match the anchor grid the same
    // branch builds, and a Reshape downstream that does not preserve the
    // element count.
    std::vector<int64_t> split = c.attr_ints("split");
    if (c.in.size() > 1 && c.in[1]->on_host) {
        const std::vector<int64_t> v = read_i64(c.in[1]);
        if (!v.empty()) split = v;
    }
    if (split.empty()) {
        const int64_t n = c.out.size();
        const int64_t each = x->shape[axis] / (n ? n : 1);
        for (int64_t i = 0; i < n; ++i) split.push_back(each);
    }
    for (size_t i = 0; i < c.out.size(); ++i) {
        std::vector<int64_t> shape = x->shape;
        shape[axis] = i < split.size() ? split[i] : shape[axis];
        out.push_back(shape);
    }
    return true;
}

bool shape_where(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    // The shape: the broadcast of the *value* inputs (the condition has the
    // shape of the others).
    std::vector<std::vector<int64_t>> in;
    for (size_t i = 1; i < c.in.size(); ++i) in.push_back(c.in[i]->shape);
    if (in.empty()) in.push_back(c.in[0]->shape);
    out.assign(c.out.size(), broadcast_shapes(in));
    // The type: the values', not the condition's.  The condition is a BOOL
    // (a float 0/1 here), and an output that took its type would be a
    // one-byte buffer that the next operator writes floats into.
    if (c.in.size() > 1) {
        for (tensor * t : c.out) t->dtype = c.in[1]->dtype;
    }
    return true;
}

bool shape_expand(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    if (c.in.size() < 2 || !c.in[1]->on_host) return false;
    const std::vector<int64_t> shape = read_i64(c.in[1]);
    out.assign(c.out.size(), shape);
    return true;
}

bool shape_tile(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    if (c.in.size() < 2 || !c.in[1]->on_host) return false;
    // The repeats: the models mix int32 and int64 for them, so the width
    // follows the tensor's type.
    const std::vector<int64_t> rep = read_i64(c.in[1]);
    std::vector<int64_t> shape = c.in[0]->shape;
    // The repeats apply from the `axis` on (the default is 0).
    int64_t axis = c.attr_i("axis", 0);
    const int64_t rank = (int64_t)shape.size();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis > rank) axis = 0;
    for (size_t i = 0; i < rep.size(); ++i) {
        const size_t d = (size_t)axis + i;
        if (d < shape.size()) shape[d] *= rep[i];
    }
    out.assign(c.out.size(), shape);
    return true;
}

bool shape_range(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    if (c.in.empty() || !c.in[0]->on_host) return false;
    // The start, the limit and the delta.  They are int64 in the models
    // that compute them (the Range's limit is the number of the boxes, and
    // it comes from a Shape), so they are read as int64.
    auto val = [&](size_t i, double def) -> double {
        if (i >= c.in.size() || !c.in[i]->on_host || c.in[i]->count == 0) return def;
        if (c.in[i]->dtype == onnxpb::INT64) {
            const std::vector<int64_t> v = read_i64(c.in[i]);
            return v.empty() ? def : (double)v[0];
        }
        float v = 0;
        std::memcpy(&v, c.in[i]->host.data(), 4);
        return v;
    };
    const double start = val(0, 0.0);
    const double limit = val(1, 0.0);
    const double delta = val(2, 1.0);
    const int64_t n = delta != 0 ? (int64_t)std::ceil((limit - start) / delta) : 0;
    out.assign(c.out.size(), {std::max<int64_t>(n, 0)});
    // The type is int64: a Range's output is always an integer vector,
    // whatever the type of its bounds (the models give it int64 bounds and
    // read an int64 range).
    for (tensor * t : c.out) t->dtype = onnxpb::INT64;
    return true;
}

} // namespace onnx
} // namespace si
