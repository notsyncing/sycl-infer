#include "const_eval.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>

namespace si {
namespace npu {

namespace {

size_t esz(int32_t t) { return onnxpb::elem_size(t); }

std::vector<int64_t> attr_ints(const onnxpb::node_proto & n, const char * key) {
    for (const auto & a : n.attribute)
        if (a.name == key && a.type_ == onnxpb::attribute::type::ints) return a.ints;
    return {};
}

int64_t attr_i(const onnxpb::node_proto & n, const char * key, int64_t def) {
    for (const auto & a : n.attribute)
        if (a.name == key && a.type_ == onnxpb::attribute::type::i) return static_cast<int64_t>(a.i);
    return def;
}

std::string attr_s(const onnxpb::node_proto & n, const char * key) {
    for (const auto & a : n.attribute)
        if (a.name == key && a.type_ == onnxpb::attribute::type::s) return a.s;
    return "";
}

const onnxpb::tensor_proto * attr_t(const onnxpb::node_proto & n) {
    for (const auto & a : n.attribute)
        if (a.type_ == onnxpb::attribute::type::t) return &a.t;
    return nullptr;
}

size_t dim_product(const std::vector<int64_t> & dims) {
    size_t n = 1;
    for (auto d : dims) n *= d > 0 ? static_cast<size_t>(d) : 1;
    return n;
}

std::vector<int64_t> shape_of(const const_tensor & t) { return t.dims; }

std::vector<int64_t> strides_of(const std::vector<int64_t> & dims) {
    std::vector<int64_t> st(dims.size());
    int64_t acc = 1;
    for (size_t i = dims.size(); i > 0; --i) {
        st[i - 1] = acc;
        acc *= dims[i - 1] > 0 ? dims[i - 1] : 1;
    }
    return st;
}

bool broadcast_index(size_t out_flat, const std::vector<int64_t> & out_dims,
                     const std::vector<int64_t> & in_dims, size_t & in_flat) {
    auto os = strides_of(out_dims), is = strides_of(in_dims);
    size_t offset = out_dims.size() - in_dims.size();
    in_flat = 0;
    for (size_t d = 0; d < in_dims.size(); ++d) {
        size_t idx = (out_flat / os[d + offset]) % static_cast<size_t>(out_dims[d + offset] > 0 ? out_dims[d + offset] : 1);
        if (in_dims[d] == 1) idx = 0;
        in_flat += idx * static_cast<size_t>(is[d]);
    }
    return true;
}

std::vector<int64_t> broadcast_dims(const std::vector<int64_t> & a, const std::vector<int64_t> & b) {
    size_t n = std::max(a.size(), b.size());
    std::vector<int64_t> r(n, 1);
    for (size_t i = 0; i < n; ++i) {
        int64_t da = i >= n - a.size() ? a[i - (n - a.size())] : 1;
        int64_t db = i >= n - b.size() ? b[i - (n - b.size())] : 1;
        r[i] = da == 1 ? db : (db == 1 || db == da ? da : (da == 0 ? db : da));
    }
    return r;
}

}  // namespace

size_t const_tensor::count() const { return dim_product(dims); }

int64_t const_tensor::elem_i(size_t i) const {
    size_t w = esz(dtype);
    if (w == 8) return reinterpret_cast<const int64_t *>(raw.data())[i];
    if (w == 4) {
        if (dtype == onnxpb::FLOAT) return static_cast<int64_t>(reinterpret_cast<const float *>(raw.data())[i]);
        return reinterpret_cast<const int32_t *>(raw.data())[i];
    }
    if (w == 2) return reinterpret_cast<const int16_t *>(raw.data())[i];
    if (w == 1) return static_cast<int8_t>(raw[i]);
    return 0;
}

float const_tensor::elem_f(size_t i) const {
    if (dtype == onnxpb::FLOAT) return reinterpret_cast<const float *>(raw.data())[i];
    return static_cast<float>(elem_i(i));
}

const_evaluator::const_evaluator(const onnxpb::graph_proto & g) : g_(g) {
    for (const auto & t : g.initializer) {
        names_.push_back(t.name);
        cache_.push_back({t.data_type, t.dims, t.raw});
        states_.push_back(1);
    }
}

int const_evaluator::find(const std::string & name) const {
    for (size_t i = 0; i < names_.size(); ++i)
        if (names_[i] == name) return static_cast<int>(i);
    return -1;
}

bool const_evaluator::eval(const std::string & name, const_tensor & out) {
    int idx = find(name);
    if (idx >= 0 && states_[idx] == 1) {
        out = cache_[idx];
        return true;
    }
    if (idx >= 0 && states_[idx] == 2) return false;
    const onnxpb::node_proto * producer = nullptr;
    for (const auto & n : g_.node) {
        for (const auto & o : n.output) {
            if (o == name) { producer = &n; break; }
        }
        if (producer) break;
    }
    if (!producer) return false;
    if (idx < 0) {
        names_.push_back(name);
        cache_.emplace_back();
        states_.push_back(0);
        idx = static_cast<int>(names_.size()) - 1;
    }
    states_[idx] = 2;
    const_tensor result;
    if (!eval_node(*producer, result, 0)) return false;
    cache_[idx] = result;
    states_[idx] = 1;
    out = result;
    return true;
}

bool const_evaluator::eval_node(const onnxpb::node_proto & n, const_tensor & out, int depth) {
    if (depth > 64) return false;
    if (n.op_type == "Constant") {
        const auto * t = attr_t(n);
        if (!t) return false;
        out = {t->data_type, t->dims, t->raw};
        return true;
    }
    if (n.op_type == "Identity") {
        if (n.input.empty()) return false;
        return eval(n.input[0], out);
    }
    std::vector<const_tensor> in(n.input.size());
    for (size_t i = 0; i < n.input.size(); ++i) {
        if (n.input[i].empty()) continue;
        if (!eval(n.input[i], in[i])) return false;
    }
    const auto & a = in.empty() ? in[0] : in[0];
    (void)a;

    if (n.op_type == "Cast") {
        if (in.empty()) return false;
        out = in[0];
        out.dtype = static_cast<int32_t>(attr_i(n, "to", in[0].dtype));
        size_t w = esz(out.dtype);
        size_t nelem = in[0].count();
        std::vector<uint8_t> raw(nelem * w, 0);
        for (size_t q = 0; q < nelem; ++q) {
            int64_t v = in[0].elem_i(q);
            if (w == 8) reinterpret_cast<int64_t *>(raw.data())[q] = v;
            else if (w == 4) {
                if (out.dtype == onnxpb::FLOAT) reinterpret_cast<float *>(raw.data())[q] = in[0].dtype == onnxpb::FLOAT
                    ? reinterpret_cast<const float *>(in[0].raw.data())[q] : static_cast<float>(v);
                else reinterpret_cast<int32_t *>(raw.data())[q] = static_cast<int32_t>(v);
            } else if (w == 2) reinterpret_cast<int16_t *>(raw.data())[q] = static_cast<int16_t>(v);
            else raw[q] = static_cast<uint8_t>(v);
        }
        out.raw = std::move(raw);
        return true;
    }
    if (n.op_type == "Shape") {
        if (in.empty()) return false;
        out.dtype = static_cast<int32_t>(attr_i(n, "to", onnxpb::INT64));
        out.dims = {static_cast<int64_t>(in[0].dims.size())};
        out.raw.assign(out.dims[0] * 8, 0);
        for (size_t q = 0; q < in[0].dims.size(); ++q)
            reinterpret_cast<int64_t *>(out.raw.data())[q] = in[0].dims[q];
        return true;
    }
    if (n.op_type == "Reshape") {
        if (in.size() < 2) return false;
        out = in[0];
        std::vector<int64_t> dims;
        for (size_t q = 0; q < in[1].count(); ++q) dims.push_back(in[1].elem_i(q));
        if (std::find(dims.begin(), dims.end(), 0) != dims.end()) {
            size_t known = 1;
            for (auto d : dims) if (d > 0) known *= static_cast<size_t>(d);
            size_t total = in[0].count();
            for (auto & d : dims) if (d == 0) d = static_cast<int64_t>(total / (known ? known : 1));
        }
        out.dims = dims;
        return true;
    }
    if (n.op_type == "Transpose") {
        if (in.empty()) return false;
        out = in[0];
        auto perm = attr_ints(n, "perm");
        if (perm.empty()) for (size_t q = 0; q < in[0].dims.size(); ++q) perm.push_back(static_cast<int64_t>(in[0].dims.size() - 1 - q));
        out.dims.resize(perm.size());
        for (size_t q = 0; q < perm.size(); ++q)
            out.dims[q] = in[0].dims[perm[q] >= 0 && static_cast<size_t>(perm[q]) < in[0].dims.size() ? perm[q] : q];
        auto src = in[0];
        auto ss = strides_of(src.dims), ds = strides_of(out.dims);
        size_t nelem = src.count();
        out.raw.assign(nelem * esz(out.dtype), 0);
        for (size_t q = 0; q < nelem; ++q) {
            size_t sflat = 0, rem = q;
            for (size_t d = 0; d < src.dims.size(); ++d) {
                size_t coord = rem / static_cast<size_t>(ds[d]);
                rem %= static_cast<size_t>(ds[d]);
                sflat += coord * static_cast<size_t>(ss[perm[d]]);
            }
            size_t w = esz(out.dtype);
            std::memcpy(out.raw.data() + q * w, src.raw.data() + sflat * w, w);
        }
        return true;
    }
    if (n.op_type == "Concat") {
        if (in.empty()) return false;
        int64_t axis = attr_i(n, "axis", 0);
        if (axis < 0) axis += static_cast<int64_t>(in[0].dims.size());
        size_t ax = static_cast<size_t>(axis);
        out = in[0];
        out.dims[ax] = 0;
        for (const auto & t : in) if (t.dims.size() > ax) out.dims[ax] += t.dims[ax];
        size_t nelem = dim_product(out.dims);
        out.raw.assign(nelem * esz(out.dtype), 0);
        size_t w = esz(out.dtype);
        auto os = strides_of(out.dims);
        size_t written = 0;
        for (const auto & t : in) {
            auto ts = strides_of(t.dims);
            for (size_t q = 0; q < t.count(); ++q) {
                size_t rem = q, oflat = 0;
                for (size_t d = 0; d < t.dims.size(); ++d) {
                    size_t coord = rem / static_cast<size_t>(ts[d]);
                    rem %= static_cast<size_t>(ts[d]);
                    oflat += (coord + (d == ax ? written : 0)) * static_cast<size_t>(os[d]);
                }
                std::memcpy(out.raw.data() + oflat * w, t.raw.data() + q * w, w);
            }
            written += static_cast<size_t>(t.dims[ax]);
        }
        return true;
    }
    if (n.op_type == "ConstantOfShape") {
        if (in.empty()) return false;
        out.dims.clear();
        for (size_t q = 0; q < in[0].count(); ++q) out.dims.push_back(in[0].elem_i(q));
        const auto * t = attr_t(n);
        out.dtype = t ? t->data_type : onnxpb::FLOAT;
        size_t nelem = dim_product(out.dims);
        size_t w = esz(out.dtype);
        out.raw.assign(nelem * w, 0);
        if (t && !t->raw.empty()) {
            size_t ew = esz(t->data_type);
            for (size_t q = 0; q < nelem; ++q)
                std::memcpy(out.raw.data() + q * w, t->raw.data() + (q % (t->raw.size() / ew)) * ew, w);
        }
        return true;
    }
    if (n.op_type == "Range") {
        if (in.size() < 3) return false;
        double start = in[0].elem_f(0), limit = in[1].elem_f(0), delta = in[2].elem_f(0);
        if (delta == 0) return false;
        size_t count = static_cast<size_t>((limit - start) / delta);
        out.dtype = in[0].dtype;
        out.dims = {static_cast<int64_t>(count)};
        out.raw.assign(count * esz(out.dtype), 0);
        for (size_t q = 0; q < count; ++q) {
            double v = start + delta * static_cast<double>(q);
            if (out.dtype == onnxpb::FLOAT) reinterpret_cast<float *>(out.raw.data())[q] = static_cast<float>(v);
            else if (out.dtype == onnxpb::INT64) reinterpret_cast<int64_t *>(out.raw.data())[q] = static_cast<int64_t>(v);
            else if (out.dtype == onnxpb::INT32) reinterpret_cast<int32_t *>(out.raw.data())[q] = static_cast<int32_t>(v);
        }
        return true;
    }
    if (n.op_type == "Slice") {
        if (in.empty()) return false;
        out = in[0];
        std::vector<int64_t> starts, ends, axes, steps;
        if (in.size() > 1) for (size_t q = 0; q < in[1].count(); ++q) starts.push_back(in[1].elem_i(q));
        if (in.size() > 2) for (size_t q = 0; q < in[2].count(); ++q) ends.push_back(in[2].elem_i(q));
        if (in.size() > 3) for (size_t q = 0; q < in[3].count(); ++q) axes.push_back(in[3].elem_i(q));
        if (in.size() > 4) for (size_t q = 0; q < in[4].count(); ++q) steps.push_back(in[4].elem_i(q));
        if (axes.empty()) for (size_t q = 0; q < starts.size(); ++q) axes.push_back(static_cast<int64_t>(q));
        if (steps.empty()) steps.assign(axes.size(), 1);
        for (size_t q = 0; q < axes.size(); ++q) {
            int64_t ax = axes[q];
            if (ax < 0) ax += static_cast<int64_t>(out.dims.size());
            if (ax < 0 || static_cast<size_t>(ax) >= out.dims.size()) continue;
            int64_t d = out.dims[ax];
            int64_t s = starts[q] < 0 ? d + starts[q] : starts[q];
            int64_t e = ends[q] < 0 ? d + ends[q] : ends[q];
            s = std::max<int64_t>(0, std::min<int64_t>(d, s));
            e = std::max<int64_t>(0, std::min<int64_t>(d, e));
            out.dims[ax] = steps[q] > 0 ? (e > s ? (e - s + steps[q] - 1) / steps[q] : 0) : 0;
        }
        return true;
    }
    if (n.op_type == "Gather") {
        if (in.size() < 2) return false;
        int64_t axis = attr_i(n, "axis", 0);
        if (axis < 0) axis += static_cast<int64_t>(in[0].dims.size());
        size_t ax = static_cast<size_t>(axis);
        out.dtype = in[0].dtype;
        out.dims = in[0].dims;
        out.dims.erase(out.dims.begin() + ax);
        out.dims.insert(out.dims.begin() + ax, in[1].dims.begin(), in[1].dims.end());
        size_t nelem = dim_product(out.dims);
        out.raw.assign(nelem * esz(out.dtype), 0);
        auto ss = strides_of(in[0].dims), is = strides_of(in[1].dims), ds = strides_of(out.dims);
        size_t w = esz(out.dtype);
        for (size_t q = 0; q < nelem; ++q) {
            size_t rem = q;
            size_t srcflat = 0;
            for (size_t d = 0; d < out.dims.size(); ++d) {
                size_t coord = rem / static_cast<size_t>(ds[d]);
                rem %= static_cast<size_t>(ds[d]);
                if (d == ax) {
                    size_t idx = 0;
                    for (size_t k = 0; k < in[1].dims.size(); ++k) {
                        size_t ic = (coord / static_cast<size_t>(is[k])) % static_cast<size_t>(in[1].dims[k] > 0 ? in[1].dims[k] : 1);
                        idx += ic * static_cast<size_t>(in[0].dims[ax]);
                    }
                    srcflat += idx * static_cast<size_t>(ss[ax]);
                } else {
                    size_t sd = d < ax ? d : d - in[1].dims.size() + 1;
                    srcflat += coord * static_cast<size_t>(ss[sd]);
                }
            }
            std::memcpy(out.raw.data() + q * w, in[0].raw.data() + srcflat * w, w);
        }
        return true;
    }
    if (n.op_type == "Unsqueeze" || n.op_type == "Squeeze") {
        if (in.empty()) return false;
        out = in[0];
        std::vector<int64_t> axes = attr_ints(n, "axes");
        if (axes.empty() && in.size() > 1) for (size_t q = 0; q < in[1].count(); ++q) axes.push_back(in[1].elem_i(q));
        if (n.op_type == "Unsqueeze") {
            for (auto ax0 : axes) {
                int64_t ax = ax0 < 0 ? ax0 + static_cast<int64_t>(out.dims.size() + 1) : ax0;
                out.dims.insert(out.dims.begin() + ax, 1);
            }
        } else {
            std::sort(axes.rbegin(), axes.rend());
            for (auto ax0 : axes) {
                int64_t ax = ax0 < 0 ? ax0 + static_cast<int64_t>(out.dims.size()) : ax0;
                if (ax >= 0 && static_cast<size_t>(ax) < out.dims.size() && out.dims[ax] == 1)
                    out.dims.erase(out.dims.begin() + ax);
            }
        }
        return true;
    }
    return false;
}

}  // namespace npu
}  // namespace si
