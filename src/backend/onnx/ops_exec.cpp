#include <cstdio>

#include <cmath>

#include "ops.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

#include "kernels.h"
#include "conv_dnnl.h"

namespace si {
namespace onnx {

namespace {
// The diagnostics and the on/off switches are read once.  `getenv` walks the
// whole environment, and every operator executor called it: the table
// structurer's Loop body runs 157 operators per iteration, and one
// environment scan per operator is a measurable part of an iteration.
bool trace_on() {
    static const bool on = getenv("SI_ONNX_TRACE") != nullptr;
    return on;
}
bool probe_on() {
    static const bool on = getenv("SI_ONNX_PROBE") != nullptr;
    return on;
}
bool echo_on() {
    static const bool on = getenv("SI_ONNX_ECHO") != nullptr;
    return on;
}
// PF_ONNX_DNNL / PF_ONNX_MATMUL1X1: on unless set to 0.
bool conv_dnnl_on() {
    static const bool on = [] {
        const char * e = getenv("PF_ONNX_DNNL");
        return !e || atoi(e) != 0;
    }();
    return on;
}
// The convolution primitive's own A/B, separate from the matmul path:
// PF_ONNX_CONV_DNNL=0 keeps the SYCL kernel for the non-1x1 convolutions.
bool conv_dnnl_conv_on() {
    static const bool on = [] {
        const char * e = getenv("PF_ONNX_CONV_DNNL");
        return !e || atoi(e) != 0;
    }();
    return on;
}
bool conv_1x1_matmul_on() {
    static const bool on = [] {
        const char * e = getenv("PF_ONNX_MATMUL1X1");
        return !e || atoi(e) != 0;
    }();
    return on;
}
} // namespace

// The shape functions live in ops_shape.cpp; they are declared here so the
// registry can refer to them.
bool shape_elementwise(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_cumsum(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_softmax(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_layer_norm(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_conv(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_conv_transpose(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_pool(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_global_avg_pool(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_resize(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_reduce(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_concat(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_grid_sample(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_gather_nd(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_scatter_nd(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_scatter_elements(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_constant_of_shape(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_constant(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_one_hot(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_einsum(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_matmul(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_gemm(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_same(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_cast(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_shape(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_reshape(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_squeeze(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_unsqueeze(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_flatten(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_transpose(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_gather(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_slice(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_topk(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_argmax(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_split(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_where(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_compare(const op_ctx &, std::vector<std::vector<int64_t>> &);
void exec_expand(op_ctx & c);
void exec_scatter_elements(op_ctx & c);
void exec_one_hot(op_ctx & c);
bool shape_expand(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_tile(const op_ctx &, std::vector<std::vector<int64_t>> &);
bool shape_range(const op_ctx &, std::vector<std::vector<int64_t>> &);

namespace {

// The output's values, in the output's type.  (Defined below; the
// element-wise operators use it, and it is after them.)
void write_host_values(op_ctx & c, const std::vector<double> & v);

// The scratch buffer, split into the small int64 arrays the operators need.
// They are on the *device*: a host array read from a kernel reads freed
// stack (and faults on a discrete GPU).
//
//   0: the broadcast's first strides   8: the second
//  16: the strided copy's source strides  24: the destination  32: the shape
//  40: the gather's and the top-k's indices
int64_t * op_workspace(op_ctx & c, int which) {
    if (!c.workspace || c.workspace_bytes < 4096) return nullptr;
    return (int64_t *)c.workspace + which * 8;
}

// The strides of a shape, for the broadcasting walk: an axis of size 1 gets
// a stride of 0 (the same element is read again).
void broadcast_strides(const std::vector<int64_t> & in, const std::vector<int64_t> & out,
                       std::vector<int64_t> & strides) {
    strides.assign(out.size(), 0);
    int64_t s = 1;
    for (size_t i = in.size(); i-- > 0;) {
        const size_t p = out.size() - in.size() + i;
        strides[p] = in[i] == 1 ? 0 : s;
        s *= in[i];
    }
}

// The input's linear index for an output's linear index, with the
// broadcasting strides.
int64_t broadcast_index(int64_t out_index, const std::vector<int64_t> & out_shape,
                        const std::vector<int64_t> & strides) {
    int64_t idx = 0;
    int64_t rem = out_index;
    for (size_t d = out_shape.size(); d-- > 0;) {
        const int64_t dim = out_shape[d] > 0 ? out_shape[d] : 1;
        const int64_t co = rem % dim;
        rem /= dim;
        idx += co * strides[d];
    }
    return idx;
}

ew_op ew_from_name(const std::string & n) {
    if (n == "Add") return ew_op::add;
    if (n == "Sub") return ew_op::sub;
    if (n == "Mul") return ew_op::mul;
    if (n == "Div") return ew_op::div;
    if (n == "Pow") return ew_op::pow;
    if (n == "Neg") return ew_op::neg;
    if (n == "Exp") return ew_op::exp;
    if (n == "Log") return ew_op::log;
    if (n == "Sqrt") return ew_op::sqrt;
    if (n == "Erf") return ew_op::erf;
    if (n == "Floor") return ew_op::floor;
    if (n == "Ceil") return ew_op::ceil;
    if (n == "Round") return ew_op::round;
    if (n == "Relu") return ew_op::relu;
    if (n == "Sigmoid") return ew_op::sigmoid;
    if (n == "Tanh") return ew_op::tanh;
    if (n == "HardSigmoid") return ew_op::hardsigmoid;
    if (n == "HardSwish") return ew_op::hardswish;
    if (n == "Gelu") return ew_op::gelu;
    if (n == "Silu") return ew_op::silu;
    if (n == "Clip") return ew_op::clip;
    if (n == "Softplus") return ew_op::softplus;
    if (n == "Abs") return ew_op::abs;
    if (n == "Greater") return ew_op::greater;
    if (n == "Less") return ew_op::less;
    if (n == "Equal") return ew_op::equal;
    if (n == "And") return ew_op::logical_and;
    if (n == "Or") return ew_op::logical_or;
    if (n == "Not") return ew_op::logical_not;
    if (n == "GreaterOrEqual") return ew_op::greater_equal;
    if (n == "LessOrEqual") return ew_op::less_equal;
    if (n == "Sign") return ew_op::sign;
    if (n == "Max") return ew_op::max_;
    if (n == "Min") return ew_op::min_;
    if (n == "Mod") return ew_op::mod_;
    if (n == "none" || n.empty()) return ew_op::none;
    throw std::runtime_error("unknown elementwise operator " + n);
}

// The activation parameters of the activations that have them.
std::vector<float> act_params(const op_ctx & c, const std::string & name) {
    if (name == "HardSigmoid") {
        // alpha and beta, or ONNX's defaults of 0.2 and 0.5, in the linear
        // form the epilogue computes.
        std::vector<float> a = c.attr_floats("alpha");
        std::vector<float> b = c.attr_floats("beta");
        const float alpha = a.empty() ? 0.2f : a[0];
        const float beta = b.empty() ? 0.5f : b[0];
        float p[2] = {0.0f, 0.0f};
        hardsigmoid_params(c.opset, alpha, beta, p);
        if (trace_on()) {
            fprintf(stderr, "[onnx] the %s: opset %lld alpha %g beta %g -> %g %g\n", name.c_str(),
                    (long long)c.opset, alpha, beta, p[0], p[1]);
        }
        return {p[0], p[1]};
    }
    if (name == "HardSwish") {
        // The form is fixed: x * max(0, min(1, (x + 3) / 6)).
        float p[2] = {0.0f, 0.0f};
        hardswish_params(p);
        if (trace_on()) {
            fprintf(stderr, "[onnx] the %s: opset %lld -> %g %g\n", name.c_str(), (long long)c.opset,
                    p[0], p[1]);
        }
        return {p[0], p[1]};
    }
    if (name == "Clip") {
        const std::vector<float> lo = c.attr_floats("min");
        const std::vector<float> hi = c.attr_floats("max");
        // The clip is applied as a scaled hard-sigmoid when both bounds are
        // there, and as a plain clamp otherwise - the executor handles the
        // clamp case separately.
        if (!lo.empty() && !hi.empty()) return {lo[0], hi[0]};
    }
    return {};
}

// ------------------------------------------------------------- the kernels

// A straight copy of an int64 value: it stays on the host, because the
// shape functions read it there (they run before the operator does).
bool copy_i64(op_ctx & c) {
    if (c.out.empty() || !is_int(c.out[0])) return false;
    // The device's own copy: a metadata operator (reshape / squeeze /
    // unsqueeze / flatten / identity) on a device-resident integer is a
    // copy, not a value conversion, and the TopK index chain must not come
    // back to the host for it.
    if (c.in[0]->dev && c.in[0]->host.empty() && c.out[0]->dev && c.out[0]->count > 0) {
        const size_t bytes = c.out[0]->count * (is_int(c.in[0]) ? int_elem_size(c.in[0]) : 4);
        q_copy_dev(c.q, c.in[0]->dev, c.out[0]->dev, bytes);
        return true;
    }
    const std::vector<int64_t> v = host_i64(c.in[0]);
    const size_t esz = int_elem_size(c.out[0]);
    c.out[0]->host.assign(v.size() * esz, 0);
    for (size_t i = 0; i < v.size(); ++i) {
        if (esz == 4) {
            const int32_t x = (int32_t)v[i];
            std::memcpy(c.out[0]->host.data() + i * 4, &x, 4);
        } else {
            std::memcpy(c.out[0]->host.data() + i * 8, &v[i], 8);
        }
    }
    c.out[0]->on_host = true;
    return true;
}

// A metadata operator (Reshape / Squeeze / Unsqueeze / Flatten / Identity)
// does not change the bytes, only the shape.  A value that is already on the
// host therefore stays there: the device copy would send it across the bus
// only for the next operator to pull it back.  The formula recogniser's
// shape chain is exactly that (`Reshape -> Cast -> ...` on a scalar), and the
// round trip was a queue drain per link.
bool copy_host_small(op_ctx & c) {
    if (c.out.empty() || c.in.empty()) return false;
    const tensor * x = c.in[0];
    if (!x->on_host || x->host.empty()) return false;
    if (x->count > (size_t)65536) return false;
    c.out[0]->host = x->host;
    c.out[0]->on_host = true;
    return true;
}

// The elementwise operator, applied on the host.
//
// It is the same arithmetic as the device kernel (see ew_functor in
// kernels.cpp), for the values that live on the host: an integer value (the
// dimensions, the axes) has no device copy, so the device path would read
// uninitialised memory.  The layout model multiplies a Shape's value by a
// constant to build a reshape's target, and with the device path the target
// came out as zeros - which is a reshape to a scalar, and a decoder that
// finds no boxes.
float ew_host(ew_op op, float a, float b, const std::vector<float> & params) {
    switch (op) {
    case ew_op::add: return a + b;
    case ew_op::sub: return a - b;
    case ew_op::mul: return a * b;
    case ew_op::div: return b == 0.0f ? 0.0f : a / b;
    case ew_op::pow: return std::pow(a, b);
    case ew_op::neg: return -a;
    case ew_op::abs: return std::fabs(a);
    case ew_op::floor: return std::floor(a);
    case ew_op::ceil: return std::ceil(a);
    case ew_op::sign: return a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : 0.0f);
    case ew_op::greater: return a > b ? 1.0f : 0.0f;
    case ew_op::less: return a < b ? 1.0f : 0.0f;
    case ew_op::equal: return a == b ? 1.0f : 0.0f;
    case ew_op::greater_equal: return a >= b ? 1.0f : 0.0f;
    case ew_op::less_equal: return a <= b ? 1.0f : 0.0f;
    case ew_op::logical_and: return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f;
    case ew_op::logical_or: return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f;
    case ew_op::logical_not: return a == 0.0f ? 1.0f : 0.0f;
    case ew_op::max_: return a > b ? a : b;
    case ew_op::min_: return a < b ? a : b;
    case ew_op::mod_: {
        if (b == 0.0f) return 0.0f;
        return a - b * std::floor(a / b);
    }
    default: break;
    }
    (void)params;
    return 0.0f;
}

// Whether ew_host implements the operator.  It returns 0 for the ones it does
// not, so a host path must not take those: a sigmoid of a host value would
// come out as a tensor of zeros.
bool ew_host_ok(ew_op op) {
    switch (op) {
    case ew_op::add:
    case ew_op::sub:
    case ew_op::mul:
    case ew_op::div:
    case ew_op::pow:
    case ew_op::neg:
    case ew_op::abs:
    case ew_op::floor:
    case ew_op::ceil:
    case ew_op::sign:
    case ew_op::greater:
    case ew_op::less:
    case ew_op::equal:
    case ew_op::greater_equal:
    case ew_op::less_equal:
    case ew_op::logical_and:
    case ew_op::logical_or:
    case ew_op::logical_not:
    case ew_op::max_:
    case ew_op::min_:
    case ew_op::mod_:
        return true;
    default:
        return false;
    }
}

void exec_elementwise(op_ctx & c) {
    const std::string & name = c.node.op_type;
    const ew_op op = ew_from_name(name);
    // The activation the graph fused into this operator's epilogue (the
    // layout model's 41 "Add -> Relu/Clip/Sigmoid" pairs).  The same
    // apply_act the convolution uses, so the result cannot drift from the
    // standalone activation.
    const ew_op act = ew_from_name(c.activation.empty() ? "none" : c.activation);
    act_params_t ap;
    for (size_t k = 0; k < c.act_params.size() && k < 4; ++k) ap.p[k] = c.act_params[k];
    // The inputs of a comparison or a logical operator may be integers or
    // booleans - a loop's condition compares an int64 counter with an int64
    // limit, and a model's mask is a float - and the device kernel reads
    // floats.  An integer input therefore has to be converted before it
    // reaches the device, and the conversion follows the tensor's type.
    //
    // The values are few (a condition is a scalar), so the conversion is
    // done on the host.
    const bool a_int = !c.in.empty() && c.in[0]->dtype != onnxpb::FLOAT;
    const bool b_int = c.in.size() > 1 && c.in[1]->dtype != onnxpb::FLOAT;
    const bool int_in = a_int || b_int;
    // An int64 arithmetic operator on a device-resident index tensor runs on
    // the device.  The layout model's decoder takes a TopK index modulo the
    // grid size (int64 % int64 scalar) before it builds the box coordinates;
    // the host path below reads `host_i64`, which is empty for a value the
    // device owns, and every index would come out as 0.
    if (c.out[0]->dtype == onnxpb::INT64 &&
        (op == ew_op::mod_ || op == ew_op::max_ || op == ew_op::min_ || op == ew_op::add ||
         op == ew_op::sub || op == ew_op::mul)) {
        const tensor * A = c.in[0];
        const tensor * B = c.in.size() > 1 ? c.in[1] : nullptr;
        const bool a_dev = A->dev && A->host.empty();
        const bool b_dev = B && B->dev && B->host.empty();
        const bool shape_ok = !B || A->count == B->count || A->count == 1 || B->count == 1;
        if ((a_dev || b_dev) && shape_ok && (A->dtype == onnxpb::INT64) &&
            (!B || B->dtype == onnxpb::INT64)) {
            int scalar = -1;
            if (B && B->count == 1 && A->count != 1) scalar = 1;
            else if (B && A->count == 1 && B->count != 1) scalar = 0;
            ew_i64_launch(c.q, (const int64_t *)A->dev, B ? (const int64_t *)B->dev : nullptr,
                          (int64_t *)c.out[0]->dev, c.out[0]->count, op, scalar);
            return;
        }
    }
    // A comparison on device-resident int64 operands, writing the runtime's
    // boolean (a float 0/1; `shape_compare` gives a comparison FLOAT even
    // where ONNX says BOOL).
    //
    // Without this the output fell through to the integer host path, whose
    // `ensure_host` downloads every operand - and the download drains the
    // in-order queue, so the formula recogniser's Loop paid two of them per
    // iteration (the `Equal`s in its stop check).  Those two downloads were
    // 6.0 of the loop's 11.4 host ms per step.
    const bool is_cmp = op == ew_op::equal || op == ew_op::greater || op == ew_op::less ||
                        op == ew_op::greater_equal || op == ew_op::less_equal ||
                        op == ew_op::logical_and || op == ew_op::logical_or ||
                        op == ew_op::logical_not;
    if ((c.out[0]->dtype == onnxpb::BOOL || c.out[0]->dtype == onnxpb::FLOAT) && is_cmp &&
        !c.in.empty()) {
        const tensor * A = c.in[0];
        const tensor * B = c.in.size() > 1 ? c.in[1] : nullptr;
        const bool a_dev = A->dev && A->host.empty();
        const bool b_dev = B && B->dev && B->host.empty();
        const bool shape_ok = !B || A->count == B->count || A->count == 1 || B->count == 1;
        if (c.out[0]->dev && (a_dev || b_dev) && shape_ok && A->dtype == onnxpb::INT64 &&
            (!B || B->dtype == onnxpb::INT64) && (!B || B->dev)) {
            int scalar = -1;
            if (B && B->count == 1 && A->count != 1) scalar = 1;
            else if (B && A->count == 1 && B->count != 1) scalar = 0;
            ew_i64_cmp_launch(c.q, (const int64_t *)A->dev, B ? (const int64_t *)B->dev : nullptr,
                              (float *)c.out[0]->dev, c.out[0]->count, op, scalar);
            return;
        }
    }
    // A *small* elementwise whose inputs are all already on the host stays
    // there, whatever the output's type.
    //
    // The control flow of a decoded table is built out of scalar and
    // few-hundred-element comparisons, boolean ops and integer adds.  Their
    // inputs are the host-resident integer values, but the *comparisons*
    // produce a float (a BOOL is a float 0/1 here), so the generic rule sent
    // the result to the device - and the integer cast that follows it came
    // back with a blocking download.  That chain was several of the twenty
    // queue drains one table-decoder iteration paid.  A few thousand
    // elements is far below the point where the device is the faster side
    // (the comment on the branch below is about twelve million).
    {
        const size_t n = c.out[0]->count;
        bool all_host = n > 0 && n <= (size_t)8192 && act == ew_op::none && ew_host_ok(op);
        if (all_host) {
            for (const tensor * t : c.in) {
                if (!t || !t->on_host || t->host.empty()) {
                    all_host = false;
                    break;
                }
            }
        }
        if (all_host) {
            std::vector<double> av, bv;
            if (!c.in.empty()) av = host_values_any(c.q, c.in[0]);
            if (c.in.size() > 1) bv = host_values_any(c.q, c.in[1]);
            const std::vector<int64_t> & sa = c.in[0]->shape;
            const std::vector<int64_t> & sb = c.in.size() > 1 ? c.in[1]->shape
                                                              : std::vector<int64_t>();
            const std::vector<int64_t> & so = c.out[0]->shape;
            std::vector<int64_t> as, bs;
            broadcast_strides(sa, so, as);
            broadcast_strides(sb, so, bs);
            const std::vector<float> params = act_params(c, name);
            std::vector<double> res(n, 0.0);
            for (size_t i = 0; i < n; ++i) {
                const int64_t ai = broadcast_index(i, so, as);
                const int64_t bi = broadcast_index(i, so, bs);
                const float x = (ai >= 0 && ai < (int64_t)av.size()) ? (float)av[(size_t)ai] : 0.0f;
                const float y = !bv.empty() && bi >= 0 && bi < (int64_t)bv.size()
                                    ? (float)bv[(size_t)bi] : 0.0f;
                res[i] = (double)ew_host(op, x, y, params);
            }
            if (is_int(c.out[0])) {
                write_host_values(c, res);
            } else {
                // A float result of the control flow stays on the host too:
                // the next operator is another comparison, a cast to an
                // integer, or the branch condition, and all of them read it
                // there.  A device consumer uploads it (a few bytes).
                c.out[0]->host.assign(n * 4, 0);
                for (size_t i = 0; i < n; ++i) {
                    const float f = (float)res[i];
                    std::memcpy(c.out[0]->host.data() + i * 4, &f, 4);
                }
                c.out[0]->on_host = true;
            }
            return;
        }
    }
    // A float output with one *integer* operand: the operand is converted on
    // the host, uploaded into the small scratch, and the arithmetic stays on
    // the device.  (The host would take three seconds for the layout model's
    // twelve million elements.)
    if (!is_int(c.out[0]) && int_in && !c.in.empty() && !(a_int && b_int) && c.scratch) {
        const tensor * it = a_int ? c.in[0] : c.in[1];
        const size_t n = it->count;
        if (n * 4 <= c.scratch_bytes) {
            if (it->dtype != onnxpb::FLOAT) ensure_host(c.q, const_cast<tensor *>(it));
            const std::vector<double> v = host_values_any(c.q, it);
            // The conversion to float32 is exact only up to 2^24: a bigger
            // integer would come back changed, so the host does that one.
            bool exact = true;
            for (double x : v) {
                if (x > 16777216.0 || x < -16777216.0) {
                    exact = false;
                    break;
                }
            }
            if (exact) {
                std::vector<float> f(n, 0.0f);
                for (size_t i = 0; i < n; ++i) f[i] = i < v.size() ? (float)v[i] : 0.0f;
                q_upload(c.q, f.data(), c.scratch, n * 4);
                const float * conv = (const float *)c.scratch;
                const std::vector<int64_t> & sa = c.in[0]->shape;
                const std::vector<int64_t> & sb = c.in[1]->shape;
                const std::vector<int64_t> & so = c.out[0]->shape;
                float * yy = (float *)c.out[0]->dev;
                if (a_int) {
                    // a is the integer, b is the float.
                    const float * bb = (const float *)c.in[1]->dev;
                    if (c.in[1]->count == 1) {
                        ew_launch(c.q, conv, bb, yy, c.out[0]->count, op, act_params(c, name), false, 0, act, ap);
                    } else if (c.in[1]->count == c.out[0]->count) {
                        ew_launch(c.q, conv, bb, yy, c.out[0]->count, op, act_params(c, name), false, -1, act, ap);
                    } else {
                        std::vector<int64_t> as, bs;
                        broadcast_strides(sa, so, as);
                        broadcast_strides(sb, so, bs);
                        ew_bcast_launch(c.q, conv, as.data(), bb, bs.data(), yy, so.data(),
                                        (int)so.size(), c.out[0]->count, op, act_params(c, name), act, ap);
                    }
                } else {
                    // b is the integer, a is the float.
                    const float * aa = (const float *)c.in[0]->dev;
                    if (c.in[0]->count == 1) {
                        ew_launch(c.q, aa, conv, yy, c.out[0]->count, op, act_params(c, name), false, 1, act, ap);
                    } else if (c.in[0]->count == c.out[0]->count) {
                        ew_launch(c.q, aa, conv, yy, c.out[0]->count, op, act_params(c, name), false, -1, act, ap);
                    } else {
                        std::vector<int64_t> as, bs;
                        broadcast_strides(sa, so, as);
                        broadcast_strides(sb, so, bs);
                        ew_bcast_launch(c.q, aa, as.data(), conv, bs.data(), yy, so.data(),
                                        (int)so.size(), c.out[0]->count, op, act_params(c, name), act, ap);
                    }
                }
                return;
            }
        }
    }
    if (is_int(c.out[0]) || int_in) {
        // The host path: an integer value lives on the host, so the
        // arithmetic is done there.  (The inputs may be of different shapes,
        // so the broadcast is the same one the device path does.)  A value
        // the device owns is read back first.
        for (const tensor * t : c.in) {
            if (t && t->dtype != onnxpb::FLOAT && t->host.empty())
                ensure_host(c.q, const_cast<tensor *>(t));
        }
        const std::vector<double> av = host_values_any(c.q, c.in[0]);
        const std::vector<double> bv =
            c.in.size() > 1 ? host_values_any(c.q, c.in[1]) : std::vector<double>();
        const std::vector<int64_t> & sa = c.in[0]->shape;
        const std::vector<int64_t> & sb = c.in.size() > 1 ? c.in[1]->shape
                                                          : std::vector<int64_t>();
        const std::vector<int64_t> & so = c.out[0]->shape;
        std::vector<int64_t> as, bs;
        broadcast_strides(sa, so, as);
        broadcast_strides(sb, so, bs);
        const size_t n = c.out[0]->count;
        const std::vector<float> params = act_params(c, name);
        std::vector<double> res(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            // The output's coordinates, and the inputs' elements.
            const int64_t ai = broadcast_index(i, so, as);
            const int64_t bi = broadcast_index(i, so, bs);
            const float x = (ai >= 0 && ai < (int64_t)av.size()) ? (float)av[(size_t)ai] : 0.0f;
            const float y = !bv.empty() && bi >= 0 && bi < (int64_t)bv.size() ? (float)bv[(size_t)bi]
                                                                            : 0.0f;
            res[i] = (double)ew_host(op, x, y, params);
        }
        write_host_values(c, res);
        return;
    }
    if (c.in.size() == 1) {
        ew_launch(c.q, (const float *)c.in[0]->dev, nullptr, (float *)c.out[0]->dev, c.out[0]->count,
                  op, act_params(c, name), echo_on(), -1, act, ap);
        return;
    }
    // Two inputs: they may need broadcasting.
    const std::vector<int64_t> & sa = c.in[0]->shape;
    const std::vector<int64_t> & sb = c.in[1]->shape;
    const std::vector<int64_t> & so = c.out[0]->shape;
    if (sa == sb) {
        ew_launch(c.q, (const float *)c.in[0]->dev, (const float *)c.in[1]->dev,
                  (float *)c.out[0]->dev, c.out[0]->count, op, act_params(c, name), false, -1, act, ap);
        return;
    }
    // One operand is a single value: the broadcast kernel would work out
    // every element's coordinates (an integer division per axis) to read one
    // number.  The layout model's decoder divides twelve million elements by
    // a scalar, and that was 230 milliseconds.
    if (c.in[0]->count == 1 && c.in[1]->count != 1) {
        // The scalar is the *first* operand: the operator is not commutative
        // (a subtraction and a division are not), so the flag has to say
        // which one it is.
        ew_launch(c.q, (const float *)c.in[0]->dev, (const float *)c.in[1]->dev,
                  (float *)c.out[0]->dev, c.out[0]->count, op, act_params(c, name), false, 0, act, ap);
        return;
    }
    if (c.in[1]->count == 1 && c.in[0]->count != 1) {
        ew_launch(c.q, (const float *)c.in[0]->dev, (const float *)c.in[1]->dev,
                  (float *)c.out[0]->dev, c.out[0]->count, op, act_params(c, name), false, 1, act, ap);
        return;
    }
    std::vector<int64_t> as, bs;
    broadcast_strides(sa, so, as);
    broadcast_strides(sb, so, bs);
    const int rank = (int)so.size();
    // The kernel copies the strides into its own arrays before the launch,
    // so they stay on the host.
    ew_bcast_launch(c.q, (const float *)c.in[0]->dev, as.data(), (const float *)c.in[1]->dev,
                    bs.data(), (float *)c.out[0]->dev, so.data(), rank, c.out[0]->count, op,
                    act_params(c, name), act, ap);
}

void exec_conv(op_ctx & c) {
    const tensor * x = c.in[0];
    const tensor * w = c.in[1];
    // An omitted optional input is -1, not a null pointer.
    const tensor * bias = c.in.size() > 2 && c.in[2] >= 0 ? c.in[2] : nullptr;
    std::vector<int64_t> strides = c.attr_ints("strides");
    std::vector<int64_t> pads = c.attr_ints("pads");
    std::vector<int64_t> dil = c.attr_ints("dilations");
    const int64_t group = c.attr_i("group", 1);
    const int sh = strides.size() == 2 ? (int)strides[0] : 1;
    const int sw = strides.size() == 2 ? (int)strides[1] : 1;
    const int ph = pads.size() >= 2 ? (int)pads[0] : 0;
    const int pw = pads.size() >= 2 ? (int)pads[1] : 0;
    const int dh = dil.size() == 2 ? (int)dil[0] : 1;
    const int dw = dil.size() == 2 ? (int)dil[1] : 1;
    const int N = (int)x->shape[0], C = (int)x->shape[1], H = (int)x->shape[2], W = (int)x->shape[3];
    // The weight is pre-transposed to (C/g, kh, kw, M/g) by the builder.
    const int mg = (int)w->shape[3];
    const int M = mg * (int)group;
    const int kh = (int)w->shape[1], kw = (int)w->shape[2];
    // The effective padding: the auto_pad attribute decides it, and the
    // shape function computes it the same way.
    const spatial_geom g = spatial_geometry(auto_pad_value(c), H, W, kh, kw, sh, sw, ph, pw, dh,
                                             dw);
    const int eph = (int)g.pad_h, epw = (int)g.pad_w;
    // The output's spatial size comes from the shape function, so that the
    // operator and the shape cannot disagree: the auto_pad rule gives a
    // different size than the pads formula (SAME_UPPER gives ceil(in/stride)
    // with the padding computed from it).
    int Ho = (int)g.out_h, Wo = (int)g.out_w;
    if (c.out[0]->shape.size() == 4) {
        Ho = (int)c.out[0]->shape[2];
        Wo = (int)c.out[0]->shape[3];
    }
    // The fused batch norm: the scale is in[3] and the bias is in[4] (the
    // builder appends them).  They are the per-output-channel scale and
    // bias of the norm that was folded into the convolution.
    const tensor * scale = c.in.size() > 3 && c.in[3] >= 0 ? c.in[3] : nullptr;
    const tensor * add = c.in.size() > 4 && c.in[4] >= 0 ? c.in[4] : nullptr;
    // The fused activation.  (A "none" is the identity, and the epilogue's
    // default is the identity too.)
    const ew_op act = ew_from_name(c.activation.empty() ? "none" : c.activation);
    // The activation's parameters travel by value into the kernel: a host
    // pointer read from device code is garbage.
    act_params_t ap;
    for (size_t k = 0; k < c.act_params.size() && k < 4; ++k) ap.p[k] = c.act_params[k];
    // The output channels per work-item: a divisor of the group's channels.
    int tm = 8;
    while (tm > 1 && (mg % tm != 0)) tm /= 2;
    if (trace_on()) {
        fprintf(stderr,
                "[onnx] the conv: N %d C %d H %d W %d M %d Ho %d Wo %d kh %d kw %d sh %d sw %d ph %d pw %d "
                "dh %d dw %d group %d tm %d act %d params",
                N, C, H, W, M, Ho, Wo, kh, kw, sh, sw, eph, epw, dh, dw, (int)group, tm, (int)act);
        for (size_t k = 0; k < 4; ++k) fprintf(stderr, " %g", ap.p[k]);
        fprintf(stderr, "\n");
    }
    // oneDNN's f32 convolution for the group-1 calls: the primitive picks
    // its own blocked layout and vectorised loop, which measured 1.3-4.5x
    // faster than the SYCL kernel on the deep layers of the layout model.
    // A grouped/depthwise convolution, a probe run or a primitive oneDNN
    // refuses falls back to the SYCL kernel (which stays the only path for
    // those shapes).
    const bool try_dnnl = conv_dnnl_on() && conv_dnnl_conv_on();
    // A 1x1 stride-1 group-1 convolution is a batch of matmuls; it is 61 of
    // the detector's 83 convolutions and the biggest single block of its
    // time.  PF_ONNX_MATMUL1X1=0 restores the direct kernel.
    if (group == 1 && kh == 1 && kw == 1 && sh == 1 && sw == 1 && dh == 1 && dw == 1 &&
        eph == 0 && epw == 0 && !probe_on()) {
        if (conv_1x1_matmul_on()) {
            if (conv1x1_matmul_launch(c.q, (const float *)w->dev, (const float *)x->dev,
                                      (float *)c.out[0]->dev, N, M, C, Ho * Wo)) {
                if (bias || scale || add || act != ew_op::none) {
                    conv_epilogue_launch(c.q, (float *)c.out[0]->dev, N, M, Ho, Wo,
                                         bias ? (const float *)bias->dev : nullptr,
                                         scale ? (const float *)scale->dev : nullptr,
                                         add ? (const float *)add->dev : nullptr, act, ap);
                }
                return;
            }
        }
    }
    if (try_dnnl && group == 1 && !probe_on()) {
        if (conv_dnnl_launch(c.q, (const float *)x->dev, (const float *)w->dev,
                             (float *)c.out[0]->dev, N, C, H, W, M, Ho, Wo, kh, kw, sh, sw,
                             eph, epw, dh, dw)) {
            // The epilogue is a separate pass so the semantics are the SYCL
            // conv's exactly (bias, then scale, then add, then the
            // activation).  It returns early for a convolution with no
            // epilogue at all.
            if (bias || scale || add || act != ew_op::none) {
                conv_epilogue_launch(c.q, (float *)c.out[0]->dev, N, M, Ho, Wo,
                                     bias ? (const float *)bias->dev : nullptr,
                                     scale ? (const float *)scale->dev : nullptr,
                                     add ? (const float *)add->dev : nullptr, act, ap);
            }
            return;
        }
    }
    conv2d_launch(c.q, (const float *)x->dev, (const float *)w->dev,
                  bias ? (const float *)bias->dev : nullptr, scale ? (const float *)scale->dev : nullptr,
                  add ? (const float *)add->dev : nullptr, (float *)c.out[0]->dev, N, C, H, W, M, Ho,
                  Wo, kh, kw, sh, sw, eph, epw, dh, dw, (int)group, tm, act, ap,
                  probe_on());
}

void exec_conv_transpose(op_ctx & c) {
    const tensor * x = c.in[0];
    const tensor * w = c.in[1];
    const tensor * bias = c.in.size() > 2 && c.in[2] >= 0 ? c.in[2] : nullptr;
    std::vector<int64_t> strides = c.attr_ints("strides");
    std::vector<int64_t> pads = c.attr_ints("pads");
    std::vector<int64_t> dil = c.attr_ints("dilations");
    const int64_t group = c.attr_i("group", 1);
    const int sh = strides.size() == 2 ? (int)strides[0] : 1;
    const int sw = strides.size() == 2 ? (int)strides[1] : 1;
    const int ph = pads.size() == 4 ? (int)pads[0] : 0;
    const int pw = pads.size() == 4 ? (int)pads[1] : 0;
    const int dh = dil.size() == 2 ? (int)dil[0] : 1;
    const int dw = dil.size() == 2 ? (int)dil[1] : 1;
    const int N = (int)x->shape[0], C = (int)x->shape[1], H = (int)x->shape[2], W = (int)x->shape[3];
    // The weight is pre-transposed to (kh, kw, C, M/g).
    const int kh = (int)w->shape[0], kw = (int)w->shape[1];
    const int mg = (int)w->shape[3];
    const int M = mg * (int)group;
    const tensor * scale = c.in.size() > 3 && c.in[3] >= 0 ? c.in[3] : nullptr;
    const tensor * add = c.in.size() > 4 && c.in[4] >= 0 ? c.in[4] : nullptr;
    const ew_op act = ew_from_name(c.activation.empty() ? "none" : c.activation);
    // The activation's parameters travel by value into the kernel: a host
    // pointer read from device code is garbage.
    act_params_t ap;
    for (size_t k = 0; k < c.act_params.size() && k < 4; ++k) ap.p[k] = c.act_params[k];
    // The output's geometry: the executor and the kernel have to agree on
    // it (the output padding, and the explicit output shape).
    int64_t Ho = 0, Wo = 0;
    conv_transpose_geom(c, H, W, kh, kw, &Ho, &Wo);
    conv_transpose2d_launch(c.q, (const float *)x->dev, (const float *)w->dev,
                            bias ? (const float *)bias->dev : nullptr,
                            scale ? (const float *)scale->dev : nullptr,
                            add ? (const float *)add->dev : nullptr, (float *)c.out[0]->dev, N, C, H, W,
                            M, kh, kw, sh, sw, ph, pw, dh, dw, (int)group, (int)Ho, (int)Wo, act, ap);
}

void exec_pool(op_ctx & c) {
    const tensor * x = c.in[0];
    std::vector<int64_t> kernel = c.attr_ints("kernel_shape");
    std::vector<int64_t> strides = c.attr_ints("strides");
    std::vector<int64_t> pads = c.attr_ints("pads");
    const int kh = kernel.size() == 2 ? (int)kernel[0] : 1;
    const int kw = kernel.size() == 2 ? (int)kernel[1] : 1;
    const int sh = strides.size() == 2 ? (int)strides[0] : 1;
    const int sw = strides.size() == 2 ? (int)strides[1] : 1;
    const int ph = pads.size() >= 2 ? (int)pads[0] : 0;
    const int pw = pads.size() >= 2 ? (int)pads[1] : 0;
    const bool average = c.node.op_type == "AveragePool";
    const bool cip = c.attr_i("count_include_pad", 0) != 0;
    const spatial_geom g = spatial_geometry(auto_pad_value(c), x->shape[2], x->shape[3], kh,
                                             kw, sh, sw, ph, pw, 1, 1);
    int Ho = (int)g.out_h, Wo = (int)g.out_w;
    if (c.out[0]->shape.size() == 4) {
        Ho = (int)c.out[0]->shape[2];
        Wo = (int)c.out[0]->shape[3];
    }
    pool2d_launch(c.q, (const float *)x->dev, (float *)c.out[0]->dev, (int)x->shape[0],
                  (int)x->shape[1], (int)x->shape[2], (int)x->shape[3], Ho, Wo, kh, kw, sh, sw,
                  (int)g.pad_h, (int)g.pad_w, average, cip);
}

void exec_global_avg_pool(op_ctx & c) {
    const tensor * x = c.in[0];
    global_avg_pool_launch(c.q, (const float *)x->dev, (float *)c.out[0]->dev, (int)x->shape[0],
                           (int)x->shape[1], (int)x->shape[2], (int)x->shape[3]);
}

// The Resize's output's size, computed at run time: the sizes (or the
// scales) are usually *computed* values, so the shape function cannot
// answer.
bool size_resize(op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    const tensor * x = c.in[0];
    // The inputs are (X, roi, scales, sizes).
    const tensor * scales = c.in.size() > 2 && c.in[2] >= 0 ? c.in[2] : nullptr;
    const tensor * sizes = c.in.size() > 3 && c.in[3] >= 0 ? c.in[3] : nullptr;
    std::vector<int64_t> s;
    std::vector<float> sc;
    if (trace_on()) {
        fprintf(stderr, "[onnx] the resize's size: the input's rank %zu, the sizes %zu (host %zu, dtype %d), the scales %zu (host %zu)\n",
                x->shape.size(), sizes ? sizes->count : 0, sizes ? sizes->host.size() : 0,
                sizes ? (int)sizes->dtype : -1, scales ? scales->count : 0,
                scales ? scales->host.size() : 0);
        if (sizes) {
            const std::vector<int64_t> v = read_i64(sizes);
            fprintf(stderr, "    the sizes:");
            for (int64_t d : v) fprintf(stderr, " %lld", (long long)d);
            fprintf(stderr, "\n");
        }
    }
    if (sizes && sizes->count > 0) {
        s = read_i64(sizes);
    } else if (scales && scales->count > 0) {
        // The scales may only be on the device: the values are needed here.
        sc.resize((size_t)scales->count);
        q_download(c.q, (const float *)scales->dev, sc.data(), sc.size() * 4);
    }
    std::vector<int64_t> shape = x->shape;
    if (!s.empty()) {
        if (s.size() != shape.size()) return false;
        shape = s;
    } else if (!sc.empty()) {
        if (sc.size() != shape.size()) return false;
        for (size_t i = 0; i < shape.size(); ++i) {
            shape[i] = (int64_t)std::floor((double)shape[i] * (double)sc[i]);
        }
    } else {
        return false;
    }
    out.assign(c.out.size(), shape);
    return true;
}

void exec_resize(op_ctx & c) {
    const tensor * x = c.in[0];
    if (trace_on()) {
        fprintf(stderr, "[onnx] the resize: the input");
        for (int64_t d : x->shape) fprintf(stderr, " %lld", (long long)d);
        fprintf(stderr, "  the output");
        for (int64_t d : c.out[0]->shape) fprintf(stderr, " %lld", (long long)d);
        fprintf(stderr, "  the inputs %zu", c.in.size());
        for (size_t i = 1; i < c.in.size(); ++i) {
            fprintf(stderr, " [%s dtype %d count %zu host %zu]", c.in[i]->name.c_str(),
                    (int)c.in[i]->dtype, c.in[i]->count, c.in[i]->host.size());
        }
        fprintf(stderr, "\n");
    }
    std::string mode = c.attr_s("mode");
    if (mode.empty()) mode = "nearest";
    const bool nearest = mode != "linear";
    const int N = (int)x->shape[0], C = (int)x->shape[1], H = (int)x->shape[2], W = (int)x->shape[3];
    const int Ho = (int)c.out[0]->shape[2], Wo = (int)c.out[0]->shape[3];
    resize_launch(c.q, (const float *)x->dev, (float *)c.out[0]->dev, N, C, H, W, Ho, Wo, nearest);
}

void exec_reduce(op_ctx & c) {
    const tensor * x = c.in[0];
    const std::string & name = c.node.op_type;
    // The axes: the attribute, or a constant input.
    std::vector<int64_t> axes = c.attr_ints("axes");
    if (axes.empty() && c.in.size() > 1 && c.in[1]->on_host) {
        axes = read_i64(c.in[1]);
    }
    const int64_t rank = (int64_t)x->shape.size();
    ew_op op = ew_op::add;
    if (name == "ReduceMax") op = ew_op::max_;
    else if (name == "ReduceMin") op = ew_op::min_;
    else if (name == "ReduceSum") op = ew_op::add;
    else op = ew_op::add;  // the mean: the sum divided by the count
    const bool mean = name == "ReduceMean";
    if (is_int(x) || is_int(c.out[0])) {
        // An integer value.  The device has an integer reduction - the same
        // walk with a different element type - so it stays there: the layout
        // model's decoder reduces a mask of twelve million int32 elements, and
        // the host takes 300 milliseconds for it.
        //
        // The axes have to be a contiguous run for the device kernel (the
        // same rule as the float path); anything else falls back to the host.
        int64_t first = rank, last = -1;
        for (int64_t a : axes) {
            if (a < 0) a += rank;
            if (a < 0 || a >= rank) continue;
            first = std::min(first, a);
            last = std::max(last, a);
        }
        if (last < 0) {
            first = 0;
            last = rank - 1;
        }
        bool contiguous = true;
        for (int64_t a = first; a <= last; ++a) {
            bool in_axis = false;
            for (int64_t b : axes) {
                if (b < 0) b += rank;
                if (b == a) in_axis = true;
            }
            if (!in_axis) {
                contiguous = false;
                break;
            }
        }
        if (axes.empty()) contiguous = true;
        // A few hundred elements on the host are cheaper there than a kernel
        // plus the integer output's blocking download.
        if (contiguous && x->count <= (size_t)8192 && x->on_host && !x->host.empty()) {
            contiguous = false;
        }
        if (contiguous) {
            int64_t axis = 1, inner = 1, outer = 1;
            for (int64_t d = first; d <= last; ++d) axis *= x->shape[d];
            for (int64_t d = last + 1; d < rank; ++d) inner *= x->shape[d];
            for (int64_t d = 0; d < first; ++d) outer *= x->shape[d];
            const int in_kind = x->dtype == onnxpb::INT64 ? 2 : (x->dtype == onnxpb::INT32 ? 1 : 0);
            const int out_kind =
                c.out[0]->dtype == onnxpb::INT64 ? 2 : (c.out[0]->dtype == onnxpb::INT32 ? 1 : 0);
            reduce_launch_typed(c.q, x->dev, c.out[0]->dev, outer, axis, inner, op, mean, in_kind,
                                out_kind);
            return;
        }
        // The axes are not contiguous: the host does it, over the axes the
        // operator names.  (The values are few - a mask of a few hundred
        // elements - so the host is fine.)
        ensure_host(c.q, const_cast<tensor *>(x));
        const std::vector<int64_t> v = host_i64(x);
        const std::vector<int64_t> & is = x->shape;
        const std::vector<int64_t> & os = c.out[0]->shape;
        std::vector<bool> reduced((size_t)rank, false);
        for (int64_t a : axes) {
            if (a < 0) a += rank;
            if (a < 0 || a >= rank) continue;
            reduced[(size_t)a] = true;
        }
        if (axes.empty()) {
            for (int64_t i = 0; i < rank; ++i) reduced[(size_t)i] = true;
        }
        std::vector<int64_t> stride((size_t)rank, 1);
        for (int64_t i = rank - 2; i >= 0; --i) stride[(size_t)i] = stride[(size_t)i + 1] * is[(size_t)i + 1];
        std::vector<int64_t> out_axis((size_t)rank, 0);
        {
            int64_t k = 0;
            for (int64_t i = 0; i < rank; ++i) {
                if (reduced[(size_t)i]) continue;
                out_axis[(size_t)i] = k++;
            }
        }
        int64_t total = 1;
        for (int64_t i = 0; i < rank; ++i) {
            if (reduced[(size_t)i]) total *= is[(size_t)i];
        }
        const int64_t n_out = c.out[0]->count;
        std::vector<int64_t> res((size_t)n_out, 0);
        std::vector<int64_t> coord((size_t)rank, 0), rc((size_t)rank, 0);
        for (int64_t o = 0; o < n_out; ++o) {
            int64_t rem = o;
            for (int64_t i = 0; i < rank; ++i) {
                if (reduced[(size_t)i]) {
                    coord[(size_t)i] = 0;
                    continue;
                }
                const int64_t d = os[(size_t)out_axis[(size_t)i]];
                coord[(size_t)i] = d > 0 ? rem % d : 0;
                if (d > 0) rem /= d;
            }
            int64_t acc = 0;
            bool first = true;
            for (int64_t t = 0; t < total; ++t) {
                int64_t r = t;
                for (int64_t i = 0; i < rank; ++i) {
                    if (!reduced[(size_t)i]) continue;
                    rc[(size_t)i] = r % is[(size_t)i];
                    r /= is[(size_t)i];
                }
                int64_t e = 0;
                for (int64_t i = 0; i < rank; ++i) {
                    e += (reduced[(size_t)i] ? rc[(size_t)i] : coord[(size_t)i]) * stride[(size_t)i];
                }
                const int64_t xval = (e >= 0 && e < (int64_t)v.size()) ? v[(size_t)e] : 0;
                if (first) {
                    acc = xval;
                    first = false;
                } else if (op == ew_op::max_) {
                    if (xval > acc) acc = xval;
                } else if (op == ew_op::min_) {
                    if (xval < acc) acc = xval;
                } else {
                    acc += xval;
                }
            }
            if (mean && total) acc = (int64_t)((double)acc / (double)total);
            res[(size_t)o] = acc;
        }
        // The output, in the output's type.
        const size_t esz = int_elem_size(c.out[0]);
        std::vector<uint8_t> raw((size_t)n_out * esz, 0);
        for (int64_t i = 0; i < n_out; ++i) {
            if (esz == 4) {
                const int32_t w = (int32_t)res[(size_t)i];
                std::memcpy(raw.data() + i * 4, &w, 4);
            } else {
                std::memcpy(raw.data() + i * 8, &res[(size_t)i], 8);
            }
        }
        c.out[0]->host = raw;
        c.out[0]->on_host = true;
        return;
    }
    if (axes.empty()) {
        // The whole tensor.
        const int64_t n = x->count;
        float * y = (float *)c.out[0]->dev;
        const float * xp = (const float *)x->dev;
        // The host does this one: it is a single value and it happens once
        // per call.
        std::vector<float> tmp(n);
        q_download(c.q, xp, tmp.data(), n * 4);
        float acc = tmp[0];
        for (int64_t i = 1; i < n; ++i) {
            if (op == ew_op::max_) acc = tmp[i] > acc ? tmp[i] : acc;
            else if (op == ew_op::min_) acc = tmp[i] < acc ? tmp[i] : acc;
            else acc += tmp[i];
        }
        if (mean) acc /= (float)n;
        q_upload(c.q, &acc, y, 4);
        return;
    }
    // The reduction is over a contiguous run of axes: the elements before it
    // are the "outer" ones, the ones after it the "inner" ones.  The kernel
    // walks exactly that layout, so the axes have to be contiguous.
    int64_t first = rank, last = -1;
    for (int64_t a : axes) {
        if (a < 0) a += rank;
        if (a < 0 || a >= rank) continue;
        first = std::min(first, a);
        last = std::max(last, a);
    }
    if (last < 0) {
        // The whole tensor.
        first = 0;
        last = rank - 1;
    }
    for (int64_t a = first; a <= last; ++a) {
        bool in_axis = false;
        for (int64_t b : axes) {
            if (b < 0) b += rank;
            if (b == a) in_axis = true;
        }
        if (!in_axis) {
            // The axes are not contiguous: fall back to the host.
            first = 0;
            last = rank - 1;
            break;
        }
    }
    int64_t axis = 1, inner = 1, outer = 1;
    for (int64_t d = first; d <= last; ++d) axis *= x->shape[d];
    for (int64_t d = last + 1; d < rank; ++d) inner *= x->shape[d];
    for (int64_t d = 0; d < first; ++d) outer *= x->shape[d];
    // The mean divides by the count of the reduced elements, on the device.
    reduce_launch(c.q, (const float *)x->dev, (float *)c.out[0]->dev, outer, axis, inner, op, mean);
}

void exec_grid_sample(op_ctx & c) {
    const tensor * x = c.in[0];
    const tensor * grid = c.in[1];
    const int N = (int)x->shape[0], C = (int)x->shape[1];
    const int H = (int)x->shape[2], W = (int)x->shape[3];
    const int64_t r = (int64_t)grid->shape.size();
    const int Ho = (int)grid->shape[r - 3], Wo = (int)grid->shape[r - 2];
    // The mode and the padding.
    const std::string mode = c.attr_s("mode");
    const std::string padding = c.attr_s("padding_mode");
    if (mode != "bilinear" && mode != "linear") {
        throw std::runtime_error("the GridSample only does bilinear");
    }
    int pad = 0;
    if (padding == "border") pad = 1;
    else if (padding == "reflection") pad = 2;
    else if (!padding.empty() && padding != "zeros") {
        throw std::runtime_error("the GridSample's padding mode " + padding + " is not implemented");
    }
    grid_sample_launch(c.q, (const float *)x->dev, (const float *)grid->dev, (float *)c.out[0]->dev, N,
                       C, H, W, Ho, Wo, 0, pad);
}

// The values of a tensor, on the host, as doubles.
//
// The operators that work on the host (the GatherND, the ScatterND) need
// them either way, and the *integer* values are on the host already: the
// device holds float32, and an integer value has no device copy at all.  A
// download of a null device pointer is what the operator used to do, and it
// is what the error said.
std::vector<double> read_host_values(sycl::queue & q, const tensor * t, bool ints) {
    std::vector<double> v;
    if (ints) {
        const std::vector<int64_t> i = host_i64(t);
        v.reserve(i.size());
        for (int64_t x : i) v.push_back((double)x);
        return v;
    }
    std::vector<float> f((size_t)t->count);
    if (t->count) q_download(q, (const float *)t->dev, f.data(), f.size() * 4);
    v.reserve(f.size());
    for (float x : f) v.push_back((double)x);
    return v;
}

// The output's values, in the output's type.
//
// An *integer* value lives on the host: the shape functions read it there,
// and they run before the operator does.  A *float* value lives on the
// device: the next operator is a kernel, and a kernel that reads a host-only
// buffer reads nothing (the gather's output went into the layer norm as
// zeros).
void write_host_values(op_ctx & c, const std::vector<double> & v) {
    const size_t n = c.out[0]->count;
    if (is_int(c.out[0])) {
        const size_t esz = int_elem_size(c.out[0]);
        c.out[0]->host.assign(n * esz, 0);
        for (size_t i = 0; i < n && i < v.size(); ++i) {
            const int64_t x = (int64_t)v[i];
            if (esz == 4) {
                const int32_t w = (int32_t)x;
                std::memcpy(c.out[0]->host.data() + i * 4, &w, 4);
            } else {
                std::memcpy(c.out[0]->host.data() + i * 8, &x, 8);
            }
        }
        c.out[0]->on_host = true;
        return;
    }
    if (n <= (size_t)8192) {
        // A small float result of a host computation stays on the host too.
        // The comparisons and the boolean ops of a decoded table produce
        // floats (a BOOL is a float 0/1 here) and feed each other and the
        // integer casts; uploading each one just made the next operator read
        // it back off the device, and the integer cast after that paid a
        // blocking download.  A device consumer uploads it (a few bytes).
        c.out[0]->host.assign(n * 4, 0);
        for (size_t i = 0; i < n; ++i) {
            const float f = i < v.size() ? (float)v[i] : 0.0f;
            std::memcpy(c.out[0]->host.data() + i * 4, &f, 4);
        }
        c.out[0]->on_host = true;
        return;
    }
    std::vector<float> f(n, 0.0f);
    for (size_t i = 0; i < n && i < v.size(); ++i) f[i] = (float)v[i];
    if (n) q_upload(c.q, f.data(), (float *)c.out[0]->dev, n * 4);
}

void exec_gather_nd(op_ctx & c) {
    const tensor * x = c.in[0];
    const tensor * idx = c.in[1];
    // The indices are int64 and on the host; the gathered values are few
    // enough that the host does the walk.
    const std::vector<int64_t> ids = host_i64(idx);
    const int64_t k = idx->shape.empty() ? 0 : idx->shape.back();
    const int64_t rank = (int64_t)x->shape.size();
    // The data's strides.
    std::vector<int64_t> stride(rank, 1);
    for (int64_t i = rank - 2; i >= 0; --i) stride[i] = stride[i + 1] * x->shape[i + 1];
    // The output's shape: the indices' shape without its last dim, then
    // the data's shape from the last index on.
    std::vector<int64_t> out_shape(idx->shape.begin(), idx->shape.end() - 1);
    for (int64_t i = k; i < rank; ++i) out_shape.push_back(x->shape[i]);
    int64_t inner = 1;
    for (int64_t i = k; i < rank; ++i) inner *= x->shape[i];
    const int64_t n_idx = (int64_t)ids.size() / (k > 0 ? k : 1);
    const bool ints = is_int(c.out[0]);
    // The device path: the gather is a strided copy, and the host's walk
    // over the layout model's twelve million mask elements is 166
    // milliseconds.  An integer data tensor (the box-order chain gathers one
    // int64 index vector with another) is gathered in its own width.
    if (x->dev && idx->dev && inner > 0 && k > 0) {
        if (ints) {
            gather_nd_typed_launch(c.q, x->dev, idx->dev, c.out[0]->dev,
                                   (int64_t)c.out[0]->count, inner, k,
                                   idx->dtype == onnxpb::INT32, stride.data(), x->shape.data(),
                                   (int)rank, (int)int_elem_size(c.out[0]));
        } else {
            gather_nd_launch(c.q, (const float *)x->dev, idx->dev, (float *)c.out[0]->dev,
                             (int64_t)c.out[0]->count, inner, k, idx->dtype == onnxpb::INT32,
                             stride.data(), x->shape.data(), (int)rank);
        }
        return;
    }
    if (ints) {
        ensure_host(c.q, const_cast<tensor *>(x));
        ensure_host(c.q, const_cast<tensor *>(idx));
    }
    const std::vector<double> data = read_host_values(c.q, x, ints);
    std::vector<double> res((size_t)c.out[0]->count, 0.0);
    for (int64_t n = 0; n < n_idx; ++n) {
        int64_t off = 0;
        for (int64_t d = 0; d < k; ++d) {
            int64_t id = ids[(size_t)(n * k + d)];
            if (id < 0) id += x->shape[d];
            if (id < 0 || id >= x->shape[d]) {
                off = -1;
                break;
            }
            off += id * stride[d];
        }
        if (off < 0) continue;
        const int64_t dst = n * inner;
        for (int64_t i = 0; i < inner; ++i) {
            if (dst + i < (int64_t)res.size() && off + i < (int64_t)data.size()) {
                res[(size_t)(dst + i)] = data[(size_t)(off + i)];
            }
        }
    }
    write_host_values(c, res);
}

void exec_scatter_nd(op_ctx & c) {
    const tensor * data = c.in[0];
    const tensor * idx = c.in[1];
    const tensor * upd = c.in[2];
    // The output is a copy of the data with the updates scattered in.
    const bool ints = is_int(c.out[0]);
    const int64_t k = idx->shape.empty() ? 0 : idx->shape.back();
    const int64_t rank = (int64_t)data->shape.size();
    std::vector<int64_t> stride(rank, 1);
    for (int64_t i = rank - 2; i >= 0; --i) stride[i] = stride[i + 1] * data->shape[i + 1];
    int64_t inner = 1;
    for (int64_t i = k; i < rank; ++i) inner *= data->shape[i];
    // The device path: an integer scatter over device tensors (the layout
    // model's `index_put` of the box order).  The copy of the data is the
    // other half of the operator, and the queue is in order, so it is issued
    // just before the scatter kernel.
    if (ints && data->dev && idx->dev && upd->dev && inner > 0 && k > 0 && upd->count > 0) {
        const int esz = (int)int_elem_size(c.out[0]);
        q_copy_dev(c.q, data->dev, c.out[0]->dev, data->count * (size_t)esz);
        scatter_nd_typed_launch(c.q, data->dev, idx->dev, upd->dev, c.out[0]->dev,
                                (int64_t)data->count, (int64_t)upd->count, inner, k,
                                idx->dtype == onnxpb::INT32, stride.data(), data->shape.data(),
                                (int)rank, esz);
        return;
    }
    if (ints) {
        ensure_host(c.q, const_cast<tensor *>(data));
        ensure_host(c.q, const_cast<tensor *>(idx));
        ensure_host(c.q, const_cast<tensor *>(upd));
    }
    std::vector<double> res = read_host_values(c.q, data, ints);
    const std::vector<int64_t> ids = host_i64(idx);
    const std::vector<double> u = read_host_values(c.q, upd, ints);
    const int64_t n_idx = k > 0 ? (int64_t)ids.size() / k : 0;
    if (trace_on()) {
        fprintf(stderr, "[onnx] the scatter: the data '%s' [%lld] %zu elements, the indices '%s' "
                        "%zu elements (k %lld), the updates '%s' %zu elements, the output %zu\n",
                data->name.c_str(), (long long)rank, data->count, idx->name.c_str(), ids.size(),
                (long long)k, upd->name.c_str(), upd->count, c.out[0]->count);
        for (size_t i = 0; i < ids.size() && i < 8; ++i) fprintf(stderr, " %lld", (long long)ids[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "    the hosts: the data %zu, the indices %zu, the updates %zu (the dtypes %d %d %d)\n",
                data->host.size(), idx->host.size(), upd->host.size(),
                (int)data->dtype, (int)idx->dtype, (int)upd->dtype);
        fprintf(stderr, "    the values: the data");
        for (size_t i = 0; i < res.size() && i < 4; ++i) fprintf(stderr, " %g", res[i]);
        fprintf(stderr, " the updates");
        for (size_t i = 0; i < u.size() && i < 4; ++i) fprintf(stderr, " %g", u[i]);
        fprintf(stderr, " the output's type %d, the count %zu\n", (int)c.out[0]->dtype,
                c.out[0]->count);
    }
    for (int64_t n = 0; n < n_idx; ++n) {
        int64_t off = 0;
        for (int64_t d = 0; d < k; ++d) {
            int64_t id = ids[(size_t)(n * k + d)];
            if (id < 0) id += data->shape[d];
            if (id < 0 || id >= data->shape[d]) {
                off = -1;
                break;
            }
            off += id * stride[d];
        }
        if (off < 0) continue;
        for (int64_t i = 0; i < inner; ++i) {
            const int64_t src = n * inner + i;
            if (src < (int64_t)u.size() && off + i < (int64_t)res.size()) {
                res[(size_t)(off + i)] = u[(size_t)src];
            }
        }
    }
    write_host_values(c, res);
}

void exec_einsum(op_ctx & c) {
    // The equation: the input's subscripts, then "->", then the output's.
    const std::string eq = c.attr_s("equation");
    const size_t arrow = eq.find("->");
    if (arrow == std::string::npos) {
        throw std::runtime_error("the Einsum's equation has no output");
    }
    const std::string out_term = eq.substr(arrow + 2);
    // The terms and the dimension of every letter.
    std::vector<std::string> terms;
    std::map<char, int64_t> dims;
    size_t p = 0;
    int k = 0;
    while (p < arrow) {
        const size_t comma = eq.find(',', p);
        const size_t end = (comma == std::string::npos || comma > arrow) ? arrow : comma;
        terms.push_back(eq.substr(p, end - p));
        const tensor * x = c.in[k++];
        for (size_t i = 0; i < terms.back().size() && i < x->shape.size(); ++i) {
            dims[terms.back()[i]] = x->shape[i];
        }
        if (comma == std::string::npos || comma > arrow) break;
        p = comma + 1;
    }
    // The batched matmul case: two operands whose terms are a common
    // prefix (the batch), one free letter and the contracted letter, and the
    // contracted letter last in both.
    //
    //   'bmd,bnd->bmn'
    //
    // That is a batched matrix multiply with the second operand
    // transposed - and the device has a kernel for it, while the host's
    // walk over the tensors is fifty times slower (the layout model's
    // decoder spends 108 milliseconds there).
    if (k == 2 && terms.size() == 2) {
        std::string batch;
        for (char l : terms[0]) {
            if (terms[1].find(l) == std::string::npos) break;
            if (out_term.find(l) == std::string::npos) break;
            batch.push_back(l);
        }
        const std::string a_tail = terms[0].substr(batch.size());
        const std::string b_tail = terms[1].substr(batch.size());
        const bool ok = a_tail.size() == 2 && b_tail.size() == 2 && a_tail[1] == b_tail[1] &&
                        out_term.find(a_tail[0]) != std::string::npos &&
                        out_term.find(b_tail[0]) != std::string::npos &&
                        out_term.find(a_tail[1]) == std::string::npos;
        if (ok) {
            int64_t B = 1;
            for (char l : batch) B *= dims[l];
            const int64_t M = dims[a_tail[0]];
            const int64_t N = dims[b_tail[0]];
            const int64_t K = dims[a_tail[1]];
            // The output's layout is the batch, then M, then N - which is
            // what the kernel writes.
            if ((int64_t)c.out[0]->count == B * M * N) {
                matmul_launch(c.q, (const float *)c.in[0]->dev, (const float *)c.in[1]->dev,
                              (float *)c.out[0]->dev, B, M, K, N, true);
                return;
            }
        }
    }
    // The inputs, on the host.
    std::vector<std::vector<float>> vals(k);
    for (int i = 0; i < k; ++i) {
        vals[i].resize((size_t)c.in[i]->count);
        q_download(c.q, (const float *)c.in[i]->dev, vals[i].data(), vals[i].size() * 4);
    }
    // The contracted letters: the ones that are not in the output.
    std::vector<char> contract;
    for (const auto & kv : dims) {
        bool in_out = false;
        for (char l : out_term) {
            if (l == kv.first) in_out = true;
        }
        if (!in_out) contract.push_back(kv.first);
    }
    // The output's shape and the strides of each input.
    std::vector<int64_t> out_shape;
    for (char l : out_term) out_shape.push_back(dims[l]);
    std::vector<std::vector<int64_t>> in_stride(k);
    for (int i = 0; i < k; ++i) {
        const int64_t r = (int64_t)c.in[i]->shape.size();
        in_stride[i].assign((size_t)r, 1);
        for (int64_t d = r - 2; d >= 0; --d) in_stride[i][(size_t)d] = in_stride[i][(size_t)d + 1] * c.in[i]->shape[(size_t)d + 1];
    }
    // The contraction: the number of the tuples is the product of the
    // contracted dims.
    int64_t n_contract = 1;
    for (char l : contract) n_contract *= dims[l];
    const int64_t out_count = (int64_t)c.out[0]->count;
    std::vector<float> res((size_t)out_count, 0.0f);
    // The output's strides.
    std::vector<int64_t> out_stride(out_shape.size(), 1);
    for (int64_t d = (int64_t)out_shape.size() - 2; d >= 0; --d) {
        out_stride[(size_t)d] = out_stride[(size_t)d + 1] * out_shape[(size_t)d + 1];
    }
    std::vector<int64_t> coord(26, 0);
    for (int64_t o = 0; o < out_count; ++o) {
        // The output's coordinates.
        int64_t rest = o;
        for (int64_t d = (int64_t)out_shape.size() - 1; d >= 0; --d) {
            coord[(size_t)(out_term[d] - 'a')] = rest % out_shape[(size_t)d];
            rest /= out_shape[(size_t)d];
        }
        float acc = 0.0f;
        for (int64_t ci = 0; ci < n_contract; ++ci) {
            // The contracted coordinates.
            int64_t r2 = ci;
            for (int64_t d = (int64_t)contract.size() - 1; d >= 0; --d) {
                coord[(size_t)(contract[(size_t)d] - 'a')] = r2 % dims[contract[(size_t)d]];
                r2 /= dims[contract[(size_t)d]];
            }
            float prod = 1.0f;
            for (int i = 0; i < k; ++i) {
                int64_t off = 0;
                for (size_t d = 0; d < terms[i].size(); ++d) {
                    off += coord[(size_t)(terms[i][d] - 'a')] * in_stride[i][d];
                }
                if (off < 0 || off >= (int64_t)vals[i].size()) {
                    prod = 0.0f;
                    break;
                }
                prod *= vals[i][(size_t)off];
            }
            acc += prod;
        }
        res[(size_t)o] = acc;
    }
    // The output is a float value: it lives on the device (the next
    // operator is a kernel).
    std::vector<double> d(res.size());
    for (size_t i = 0; i < res.size(); ++i) d[i] = res[i];
    write_host_values(c, d);
}

void exec_concat(op_ctx & c) {
    std::vector<const float *> ptrs;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<const tensor *> ins;
    for (const tensor * t : c.in) ins.push_back(t);
    // The promotion: a rank-0 value concatenated along axis 0 becomes a
    // rank-1 one.  Without it the input contributes nothing and the output
    // comes out short.
    const concat_geom g = concat_geometry(ins, c.attr_i("axis", 0));
    if (!g.ok) throw std::runtime_error("the Concat's axis is out of range");
    const int64_t axis = g.axis;
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= c.in[0]->shape[i];
    if (is_int(c.out[0])) {
        // A concatenation that consumes a device-resident integer (a TopK's
        // indices, the index arithmetic built from them) runs on the device
        // in the element's own width.  A concatenation of *host* values - the
        // shape arithmetic, which is most of the model's 136 concats - keeps
        // its host path: its inputs are small and never had a device copy.
        bool dev_in = false;
        bool all_dev = true;
        for (const tensor * t : c.in) {
            if (t->dtype != onnxpb::FLOAT && t->dev && t->host.empty()) dev_in = true;
            if (!t->dev) all_dev = false;
        }
        if (dev_in && all_dev && c.out[0]->count > 0) {
            std::vector<const void *> ptrs;
            std::vector<std::vector<int64_t>> shp;
            for (size_t k = 0; k < c.in.size(); ++k) {
                ptrs.push_back(c.in[k]->dev);
                shp.push_back(g.padded[k]);
            }
            concat_typed_launch(c.q, ptrs, shp, c.out[0]->dev, (int)axis, outer,
                                (int)int_elem_size(c.out[0]));
            return;
        }
        // The output holds int64 (the dimensions, most of the time): it
        // stays on the host.  An input the device owns is read back first.
        for (const tensor * t : c.in) ensure_host(c.q, const_cast<tensor *>(t));
        const int64_t out_axis =
            c.out[0]->shape.empty() || axis >= (int64_t)c.out[0]->shape.size() ? 0 : c.out[0]->shape[axis];
        int64_t inner = 1;
        for (int64_t i = axis + 1; i < (int64_t)c.out[0]->shape.size(); ++i) {
            inner *= c.out[0]->shape[i];
        }
        std::vector<int64_t> res((size_t)c.out[0]->count, 0);
        int64_t off = 0;
        for (size_t k = 0; k < c.in.size(); ++k) {
            const std::vector<int64_t> v = host_i64(c.in[k]);
            const std::vector<int64_t> & s = g.padded[k];
            // The input's own strides: the promoted shape's outer and inner.
            int64_t in_inner = 1;
            for (int64_t i = axis + 1; i < (int64_t)s.size(); ++i) in_inner *= s[i];
            const int64_t n_axis = s[axis];
            for (int64_t o = 0; o < outer; ++o) {
                for (int64_t a = 0; a < n_axis; ++a) {
                    for (int64_t i = 0; i < inner; ++i) {
                        const int64_t src = (o * n_axis + a) * in_inner + i;
                        const int64_t dst = ((o * out_axis + (off + a)) * inner) + i;
                        if (src < (int64_t)v.size() && dst < (int64_t)res.size()) {
                            res[(size_t)dst] = v[(size_t)src];
                        }
                    }
                }
            }
            off += n_axis;
        }
        const size_t esz = int_elem_size(c.out[0]);
        c.out[0]->host.assign(res.size() * esz, 0);
        for (size_t i = 0; i < res.size(); ++i) {
            if (esz == 4) {
                const int32_t x = (int32_t)res[i];
                std::memcpy(c.out[0]->host.data() + i * 4, &x, 4);
            } else {
                std::memcpy(c.out[0]->host.data() + i * 8, &res[i], 8);
            }
        }
        c.out[0]->on_host = true;
        return;
    }
    for (size_t k = 0; k < c.in.size(); ++k) {
        ptrs.push_back((const float *)c.in[k]->dev);
        shapes.push_back(g.padded[k]);
    }
    concat_launch(c.q, ptrs, shapes, (float *)c.out[0]->dev, (int)axis, outer);
}

void exec_softmax(op_ctx & c) {
    const tensor * x = c.in[0];
    int64_t axis = c.attr_i("axis", -1);
    const int64_t rank = (int64_t)x->shape.size();
    if (axis < 0) axis += rank;
    // The softmax is over the last axis; the other axes are folded into the
    // row count, and the axis is moved to the end by the executor's copy
    // when it is not.
    //
    // A rank-0 value has no axis: the softmax of a scalar is 1 (and the
    // loop over the axes would read shape[-1]).
    if (rank == 0) {
        if (c.out[0]->count) {
            const float one = 1.0f;
            q_upload(c.q, &one, (float *)c.out[0]->dev, 4);
        }
        return;
    }
    int64_t cols = 1;
    for (int64_t i = axis; i < rank; ++i) cols *= x->shape[i];
    // A zero column count is a division by zero, and a division by zero is a
    // crash: the process dies and the client sees a reset connection.  An
    // error that names the operator and the shape is worth much more.
    if (cols <= 0) {
        std::string sh;
        for (int64_t d : x->shape) sh += " " + std::to_string(d);
        throw std::runtime_error("the Softmax's input has an empty axis: the shape is [" + sh + "]");
    }
    const size_t rows = (size_t)(x->count / cols);
    softmax_launch(c.q, (const float *)x->dev, (float *)c.out[0]->dev, rows, cols);
}

void exec_layer_norm(op_ctx & c) {
    const tensor * x = c.in[0];
    const tensor * gamma = c.in.size() > 1 ? c.in[1] : nullptr;
    const tensor * beta = c.in.size() > 2 ? c.in[2] : nullptr;
    // The epsilon is a float attribute; the default is ONNX's 1e-5.
    float eps = 1e-5f;
    if (const onnxpb::attribute * a = c.attr("epsilon")) {
        if (a->type_ == onnxpb::attribute::type::f) eps = a->f;
    }
    const int64_t cols = x->shape.size() ? x->shape.back() : 0;
    if (cols <= 0) {
        std::string sh;
        for (int64_t d : x->shape) sh += " " + std::to_string(d);
        throw std::runtime_error("the LayerNormalization's input has an empty axis: the shape is [" + sh + "]");
    }
    const size_t rows = (size_t)(x->count / cols);
    layer_norm_launch(c.q, (const float *)x->dev, gamma ? (const float *)gamma->dev : nullptr,
                      beta ? (const float *)beta->dev : nullptr, (float *)c.out[0]->dev, rows, cols,
                      eps);
}

void exec_matmul(op_ctx & c) {
    const tensor * a = c.in[0];
    const tensor * b = c.in[1];
    // The bias the graph fused into this MatMul (an "Add(const)" after it).
    // The third input is not part of ONNX's MatMul: only the builder puts it
    // there, and it checks the width.
    const float * bias = c.in.size() > 2 && c.in[2] >= 0 ? (const float *)c.in[2]->dev : nullptr;
    if (bias && (c.in[2]->count != (size_t)b->shape.back() || c.in[2]->dtype != onnxpb::FLOAT)) {
        throw std::runtime_error("the fused MatMul bias does not match the output width");
    }
    const ew_op act = ew_from_name(c.activation.empty() ? "none" : c.activation);
    act_params_t ap;
    for (size_t k = 0; k < c.act_params.size() && k < 4; ++k) ap.p[k] = c.act_params[k];
    // The bias and the activation are one pass over the output.  oneDNN's
    // *bias* matmul was 4.5x slower than the same primitive without a bias
    // (36 ms against 8 ms for the layout's 123 matmuls), so the bias is a
    // separate epilogue here, as the convolution's already is.
    auto epilogue = [&](int64_t width) {
        if (bias || act != ew_op::none) {
            bias_act_launch(c.q, (float *)c.out[0]->dev, bias, c.out[0]->count, width, act, ap);
        }
    };
    // The 2-D case: (M, K) x (K, N).
    if (a->shape.size() == 2 && b->shape.size() == 2) {
        if (!matmul_dnnl_launch(c.q, (const float *)a->dev, (const float *)b->dev,
                                (float *)c.out[0]->dev, a->shape[0], a->shape[1], b->shape[1])) {
            matmul_launch(c.q, (const float *)a->dev, (const float *)b->dev,
                          (float *)c.out[0]->dev, 1, a->shape[0], a->shape[1], b->shape[1],
                          false);
        }
        epilogue(b->shape[1]);
        return;
    }
    // The batched case: the leading dims are broadcast.
    //
    // The leading dims of the two inputs are aligned from the right, and
    // each input's stride per leading dim is 0 where its dim is 1 (the
    // broadcast reads the same matrix again).
    std::vector<int64_t> sa = a->shape, sb = b->shape;
    const bool ta = sa.size() == 1, tb = sb.size() == 1;
    if (ta) sa.insert(sa.begin(), 1);
    if (tb) sb.push_back(1);
    const int64_t M = sa[sa.size() - 2], K = sa[sa.size() - 1], N = sb[sb.size() - 1];
    std::vector<int64_t> lead(sa.begin(), sa.end() - 2), blead(sb.begin(), sb.end() - 2);
    const size_t l = std::max(lead.size(), blead.size());
    // The strides over each input's own leading dims: 0 for a dim of 1
    // (the broadcast reads the same matrix again).
    std::vector<int64_t> a_str(lead.size(), 0), b_str(blead.size(), 0);
    {
        int64_t s = 1;
        for (size_t i = lead.size(); i-- > 0;) {
            a_str[i] = lead[i] == 1 ? 0 : s;
            s *= lead[i];
        }
        s = 1;
        for (size_t i = blead.size(); i-- > 0;) {
            b_str[i] = blead[i] == 1 ? 0 : s;
            s *= blead[i];
        }
    }
    // The leading dims are aligned from the right.
    std::vector<int64_t> out_batch(l, 1), as(l, 0), bs(l, 0);
    for (size_t i = 0; i < l; ++i) {
        const size_t pa = l - lead.size() + i;
        const size_t pb = l - blead.size() + i;
        const int64_t av = pa < lead.size() ? lead[pa] : 1;
        const int64_t bv = pb < blead.size() ? blead[pb] : 1;
        if (av != bv && av != 1 && bv != 1) {
            throw std::runtime_error("the batched MatMul's leading dims do not broadcast");
        }
        out_batch[i] = std::max(av, bv);
        as[i] = pa < lead.size() ? a_str[pa] : 0;
        bs[i] = pb < blead.size() ? b_str[pb] : 0;
    }
    if (trace_on()) {
        fprintf(stderr, "[onnx] the batched MatMul: a");
        for (int64_t d : a->shape) fprintf(stderr, " %lld", (long long)d);
        fprintf(stderr, "  b");
        for (int64_t d : b->shape) fprintf(stderr, " %lld", (long long)d);
        fprintf(stderr, "  out");
        for (int64_t d : c.out[0]->shape) fprintf(stderr, " %lld", (long long)d);
        fprintf(stderr, "  M %lld K %lld N %lld\n", (long long)M, (long long)K, (long long)N);
    }
    // The formula recogniser's self-attention PV: P[..., 1, S] x V[..., S, D],
    // with the head axis as the batch.  It is the one matmul whose K grows
    // with the decode depth, and the naive kernel is one work-item per
    // output (H*D = 512 at S=2500), so it was 1.09 ms per call, 12 calls per
    // step.  The split-K form (one work-group per head and key slice, D
    // lanes over the output columns) measures 0.054 ms.
    //
    // The routing is deliberately narrow - it is an attention shape, not a
    // general batched matmul: M = 1, the head dim 32 or 64, at least 128
    // keys, and the leading dims are exactly [H] or [1, H] with unit
    // strides (i.e. contiguous [H][1][K] x [H][K][D], or the same with a
    // leading 1).  Anything else keeps the naive/oneDNN path.  The
    // workspace capacity is checked here: the buffer is the executor's own
    // PV scratch, not the shared op workspace, so it cannot alias the
    // top-k's or the gather's scratch.
    if (M == 1 && (N == 32 || N == 64) && K >= 128 && l <= 2) {
        static const bool pv_on = [] {
            const char * e = getenv("PF_ONNX_MM_PV");
            return !(e && *e && std::atoi(e) == 0);
        }();
        int64_t H = 0;
        bool h_ok = false;
        if (l == 1 && out_batch[0] > 1 && as[0] == 1 && bs[0] == 1) {
            H = out_batch[0];
            h_ok = true;
        } else if (l == 2 && out_batch[0] == 1 && out_batch[1] > 1 && as[0] == 0 && bs[0] == 0 &&
                   as[1] == 1 && bs[1] == 1) {
            H = out_batch[1];
            h_ok = true;
        }
        if (pv_on && h_ok) {
            const int64_t splits = attn_pv_splits(K);
            const size_t need = (size_t)H * (size_t)splits * (size_t)N * sizeof(float);
            if (c.pv_workspace && c.pv_workspace_bytes >= need) {
                attn_pv_launch(c.q, (const float *)a->dev, (const float *)b->dev,
                               (float *)c.out[0]->dev, (float *)c.pv_workspace, H, K, N);
                epilogue(N);
                return;
            }
        }
    }
    // A batch of one (the layout's decoder bmm is [1,300,32] x [1,32,40000])
    // is a plain matmul; the leading dims of >1 keep the SYCL kernel.
    bool one = true;
    for (int64_t d : out_batch) {
        if (d != 1) {
            one = false;
            break;
        }
    }
    if (!one || !matmul_dnnl_launch(c.q, (const float *)a->dev, (const float *)b->dev,
                                    (float *)c.out[0]->dev, M, K, N)) {
        matmul_bcast_launch(c.q, (const float *)a->dev, (const float *)b->dev,
                            (float *)c.out[0]->dev, out_batch.data(), (int)l, M, K, N,
                            as.data(), bs.data(), 0, 0);
    }
    epilogue(N);
}

void exec_batch_norm(op_ctx & c) {
    const tensor * x = c.in[0];
    // The scale, the bias, the mean and the variance.
    const tensor * scale = c.in.size() > 1 && c.in[1] >= 0 ? c.in[1] : nullptr;
    const tensor * bias = c.in.size() > 2 && c.in[2] >= 0 ? c.in[2] : nullptr;
    const tensor * mean = c.in.size() > 3 && c.in[3] >= 0 ? c.in[3] : nullptr;
    const tensor * var = c.in.size() > 4 && c.in[4] >= 0 ? c.in[4] : nullptr;
    float eps = 1e-5f;
    for (const onnxpb::attribute & a : c.node.attribute) {
        if (a.name == "epsilon") eps = a.f;
    }
    const int64_t rank = (int64_t)x->shape.size();
    if (rank < 2) {
        throw std::runtime_error("the BatchNormalization needs a channel axis");
    }
    const int C = (int)x->shape[1];
    int64_t HW = 1;
    for (int64_t i = 2; i < rank; ++i) HW *= x->shape[i];
    batchnorm_launch(c.q, (const float *)x->dev, scale ? (const float *)scale->dev : nullptr,
                     bias ? (const float *)bias->dev : nullptr, mean ? (const float *)mean->dev : nullptr,
                     var ? (const float *)var->dev : nullptr, (float *)c.out[0]->dev,
                     (int)x->shape[0], C, HW, eps);
}

void exec_gemm(op_ctx & c) {
    const tensor * a = c.in[0];
    const tensor * b = c.in[1];
    const bool trans_a = c.attr_i("transA", 0) != 0;
    const bool trans_b = c.attr_i("transB", 0) != 0;
    const int64_t M = trans_a ? a->shape[1] : a->shape[0];
    const int64_t K = trans_a ? a->shape[0] : a->shape[1];
    const int64_t N = trans_b ? b->shape[0] : b->shape[1];
    // The bias is the third input, and it is per output column.  Without it
    // the Gemm is off by a whole vector: the table classifiers' scores are
    // the logit *minus* a learned bias, and the class that wins flips.
    const tensor * bias = c.in.size() > 2 && c.in[2] >= 0 ? c.in[2] : nullptr;
    matmul_launch(c.q, (const float *)a->dev, (const float *)b->dev, (float *)c.out[0]->dev, 1, M, K, N,
                  trans_b, bias ? (const float *)bias->dev : nullptr);
}

void exec_identity(op_ctx & c) {
    if (copy_i64(c)) return;
    if (copy_host_small(c)) return;
    q_copy_dev(c.q, (const float *)c.in[0]->dev, (float *)c.out[0]->dev, c.in[0]->count * 4);
}

void exec_shape(op_ctx & c) {
    // The shape of the input, as int64.
    const tensor * x = c.in[0];
    std::vector<int64_t> s(x->shape.begin(), x->shape.end());
    if (c.node.op_type == "Shape") {
        // The start and the end select a range of the dims.
        int64_t start = 0, end = (int64_t)s.size();
        if (c.in.size() > 1 && c.in[1]->on_host && c.in[1]->count >= 2) {
            std::memcpy(&start, c.in[1]->host.data(), 8);
            std::memcpy(&end, c.in[1]->host.data() + 8, 8);
        }
        if (start < 0) start += (int64_t)s.size();
        if (end < 0) end += (int64_t)s.size();
        start = std::max<int64_t>(0, std::min<int64_t>(start, (int64_t)s.size()));
        end = std::max<int64_t>(start, std::min<int64_t>(end, (int64_t)s.size()));
        s = std::vector<int64_t>(s.begin() + start, s.begin() + end);
    }
    c.out[0]->host.assign(s.size() * 8, 0);
    std::memcpy(c.out[0]->host.data(), s.data(), s.size() * 8);
    c.out[0]->on_host = true;
    c.out[0]->count = s.size();
}

void exec_reshape(op_ctx & c) {
    // The target shape is a constant; the copy is a straight one.
    if (copy_i64(c)) return;
    if (copy_host_small(c)) return;
    q_copy_dev(c.q, (const float *)c.in[0]->dev, (float *)c.out[0]->dev, c.in[0]->count * 4);
}

void exec_transpose(op_ctx & c) {
    const tensor * x = c.in[0];
    if (is_int(c.out[0])) {
        // An integer that lives on the device is permuted there, in its own
        // width.  (The layout model transposes the `index_put` result, an
        // int64 [B, N] tensor in the middle of the box-order chain.)
        if (x->dev && x->host.empty() && c.out[0]->count > 0) {
            std::vector<int64_t> p = c.attr_ints("perm");
            const int64_t rk = (int64_t)x->shape.size();
            if (p.empty()) {
                for (int64_t i = 0; i < rk; ++i) p.push_back(rk - 1 - i);
            }
            std::vector<int64_t> src((size_t)rk), dst((size_t)rk), os((size_t)rk);
            int64_t s = 1;
            for (int64_t i = rk - 1; i >= 0; --i) {
                src[(size_t)i] = s;
                s *= x->shape[(size_t)i];
            }
            for (int64_t i = 0; i < rk; ++i) os[(size_t)i] = x->shape[(size_t)p[(size_t)i]];
            s = 1;
            for (int64_t i = rk - 1; i >= 0; --i) {
                dst[(size_t)i] = s;
                s *= os[(size_t)i];
            }
            std::vector<int64_t> ss((size_t)rk), ds((size_t)rk);
            for (int64_t i = 0; i < rk; ++i) {
                ss[(size_t)i] = src[(size_t)p[(size_t)i]];
                ds[(size_t)i] = dst[(size_t)i];
            }
            strided_copy_typed_launch(c.q, x->dev, c.out[0]->dev, ss.data(), ds.data(), os.data(),
                                      (int)rk, (int64_t)c.out[0]->count,
                                      (int)int_elem_size(c.out[0]));
            return;
        }
        // The values stay on the host: the permutation of the axes.
        ensure_host(c.q, const_cast<tensor *>(x));
        const std::vector<int64_t> p = c.attr_ints("perm");
        const int64_t rank = (int64_t)x->shape.size();
        std::vector<int64_t> perm = p;
        if (perm.empty()) {
            for (int64_t i = 0; i < rank; ++i) perm.push_back(rank - 1 - i);
        }
        const std::vector<int64_t> v = host_i64(x);
        std::vector<int64_t> res((size_t)c.out[0]->count, 0);
        // The strides of the source and of the destination.
        std::vector<int64_t> src(rank, 1), dst(rank, 1), os(rank);
        for (int64_t i = 0; i < rank; ++i) os[i] = c.out[0]->shape[i];
        int64_t s = 1;
        for (int64_t i = rank - 1; i >= 0; --i) {
            src[i] = s;
            s *= x->shape[i];
        }
        s = 1;
        for (int64_t i = rank - 1; i >= 0; --i) {
            dst[i] = s;
            s *= os[i];
        }
        // Row-major: the *last* output axis is the fastest-varying, so the
        // coordinates have to be peeled from the back.  Peeling from the
        // front (d = 0..rank-1) is column-major and is an identity only when
        // every leading extent is 1 - i.e. exactly batch=1, which is how
        // this read as correct for the layout model's [B,300] `order`
        // transpose at B=1 and mixed the pages at B>1.
        for (int64_t o = 0; o < (int64_t)c.out[0]->count; ++o) {
            int64_t rem = o, si = 0;
            for (int64_t d = rank - 1; d >= 0; --d) {
                const int64_t co = rem % os[d];
                rem /= os[d];
                si += co * src[perm[d]];
            }
            if (si < (int64_t)v.size()) res[(size_t)o] = v[(size_t)si];
        }
        const size_t esz = int_elem_size(c.out[0]);
        c.out[0]->host.assign(res.size() * esz, 0);
        for (size_t i = 0; i < res.size(); ++i) {
            if (esz == 4) {
                const int32_t x = (int32_t)res[i];
                std::memcpy(c.out[0]->host.data() + i * 4, &x, 4);
            } else {
                std::memcpy(c.out[0]->host.data() + i * 8, &res[i], 8);
            }
        }
        c.out[0]->on_host = true;
        return;
    }
    std::vector<int64_t> perm = c.attr_ints("perm");
    const int64_t rank = (int64_t)x->shape.size();
    if (perm.empty()) {
        for (int64_t i = 0; i < rank; ++i) perm.push_back(rank - 1 - i);
    }
    // The strides of the source and of the destination.
    std::vector<int64_t> src(rank), dst(rank), out(rank);
    int64_t s = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
        src[i] = s;
        s *= x->shape[i];
    }
    for (int64_t i = 0; i < rank; ++i) out[i] = x->shape[perm[i]];
    s = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
        dst[i] = s;
        s *= out[i];
    }
    // The kernel walks the *output's* axes, so the strides have to be in
    // that order too: the source's stride for the output's axis i is the
    // source's stride of the axis perm[i].
    std::vector<int64_t> ss(rank), ds(rank), os(rank);
    for (int64_t i = 0; i < rank; ++i) {
        os[i] = out[i];
        ss[i] = src[perm[i]];
        ds[i] = dst[i];
    }
    strided_copy_launch(c.q, (const float *)x->dev, (float *)c.out[0]->dev, ss.data(), ds.data(),
                        os.data(), (int)rank, x->count);
}

void exec_flatten(op_ctx & c) {
    if (copy_i64(c)) return;
    if (copy_host_small(c)) return;
    q_copy_dev(c.q, (const float *)c.in[0]->dev, (float *)c.out[0]->dev, c.in[0]->count * 4);
}

void exec_gather(op_ctx & c) {
    const tensor * x = c.in[0];
    const tensor * idx = c.in[1];
    int64_t axis = c.attr_i("axis", 0);
    const int64_t rank = (int64_t)x->shape.size();
    if (axis < 0) axis += rank;
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= x->shape[i];
    for (int64_t i = axis + 1; i < rank; ++i) inner *= x->shape[i];
    // The indices are a constant (the models gather with a constant).
    const std::vector<int64_t> ids = read_i64(idx);
    if (is_int(c.out[0])) {
        // The output holds int64: it stays on the host.
        const std::vector<int64_t> data = host_i64(x);
        std::vector<int64_t> res((size_t)c.out[0]->count, 0);
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t g = 0; g < (int64_t)ids.size(); ++g) {
                int64_t id = ids[(size_t)g];
                if (id < 0) id += x->shape[axis];
                for (int64_t i = 0; i < inner; ++i) {
                    const int64_t src = (o * x->shape[axis] + id) * inner + i;
                    const int64_t dst = (o * (int64_t)ids.size() + g) * inner + i;
                    if (src < (int64_t)data.size() && dst < (int64_t)res.size()) {
                        res[(size_t)dst] = data[(size_t)src];
                    }
                }
            }
        }
        const size_t esz = int_elem_size(c.out[0]);
        c.out[0]->host.assign(res.size() * esz, 0);
        for (size_t i = 0; i < res.size(); ++i) {
            if (esz == 4) {
                const int32_t x = (int32_t)res[i];
                std::memcpy(c.out[0]->host.data() + i * 4, &x, 4);
            } else {
                std::memcpy(c.out[0]->host.data() + i * 8, &res[i], 8);
            }
        }
        c.out[0]->on_host = true;
        return;
    }
    {
        int64_t * ws = op_workspace(c, 5);
        if (!ws || ids.size() > 1024) {
            throw std::runtime_error("the gather needs a workspace");
        }
        // The indices go to the device *through the queue*.  The workspace is
        // device memory, and writing to it from the host is a fault on a
        // discrete GPU (the kernel reads it, so it has to be there).
        //
        // (The upload used to be a device-to-device copy of the workspace
        // onto itself, and the values were written into the device buffer
        // from the host: the formula recogniser's gather is the one that
        // reached it, and it died with a segfault.)
        q_upload(c.q, ids.data(), ws, ids.size() * sizeof(int64_t));
        gather_launch(c.q, (const float *)x->dev, ws, (float *)c.out[0]->dev, outer, x->shape[axis],
                      inner, (int64_t)ids.size());
    }
}

void exec_slice(op_ctx & c) {
    const tensor * x = c.in[0];
    // The starts, the ends, the axes and the steps are constant inputs.
    const std::vector<int64_t> starts = read_i64(c.in[1]);
    const std::vector<int64_t> ends = read_i64(c.in[2]);
    const std::vector<int64_t> axes = c.in.size() > 3 ? read_i64(c.in[3]) : std::vector<int64_t>();
    const std::vector<int64_t> steps = c.in.size() > 4 ? read_i64(c.in[4]) : std::vector<int64_t>();
    if (trace_on()) {
        fprintf(stderr, "[onnx] the slice '%s': the input [", x->name.c_str());
        for (int64_t d : x->shape) fprintf(stderr, " %lld", (long long)d);
        fprintf(stderr, "] the starts");
        for (int64_t v : starts) fprintf(stderr, " %lld", (long long)v);
        fprintf(stderr, " the ends");
        for (int64_t v : ends) fprintf(stderr, " %lld", (long long)v);
        fprintf(stderr, " the axes");
        for (int64_t v : axes) fprintf(stderr, " %lld", (long long)v);
        fprintf(stderr, " the steps");
        for (int64_t v : steps) fprintf(stderr, " %lld", (long long)v);
        fprintf(stderr, " (the axes' dtype %d, the steps' dtype %d)\n",
                c.in.size() > 3 ? (int)c.in[3]->dtype : -1,
                c.in.size() > 4 ? (int)c.in[4]->dtype : -1);
        // The raw bytes of the bounds' host copies: a value that is right
        // on the host and wrong on the device means the device copy was
        // overwritten.
        for (size_t k = 1; k < c.in.size() && k < 5; ++k) {
            const std::vector<uint8_t> & h = c.in[k]->host;
            fprintf(stderr, "    the in[%zu] '%s': the host %zu bytes", k, c.in[k]->name.c_str(),
                    h.size());
            for (size_t b = 0; b < h.size() && b < 16; ++b) fprintf(stderr, " %02x", h[b]);
            fprintf(stderr, "\n");
        }
    }
    const int64_t rank = (int64_t)x->shape.size();
    // A rank-0 value has no axis: the slice is the value itself.  (And
    // x->shape[0] on an empty shape is a wild read - the crash that a
    // carried value with an empty shape produced.)
    if (rank == 0) {
        const size_t esz = is_int(c.out[0]) ? int_elem_size(c.out[0]) : 4;
        c.out[0]->host.assign(x->host.size() ? x->host.size() : esz, 0);
        c.out[0]->on_host = true;
        return;
    }
    // The axes that are sliced, and the slice of each.
    std::vector<bool> sliced((size_t)rank, false);
    std::vector<int64_t> st_of((size_t)rank, 0), en_of((size_t)rank, 0), step_of((size_t)rank, 1);
    for (size_t i = 0; i < starts.size(); ++i) {
        int64_t a = axes.size() > i ? axes[i] : (int64_t)i;
        if (a < 0) a += rank;
        if (a < 0 || a >= rank) a = 0;
        int64_t st = starts[i], en = ends[i];
        const int64_t step = steps.size() > i ? steps[i] : 1;
        if (st < 0) st += x->shape[(size_t)a];
        if (en < 0) en += x->shape[(size_t)a];
        if (step > 0) {
            st = std::max<int64_t>(0, std::min<int64_t>(st, x->shape[(size_t)a]));
            en = std::max<int64_t>(0, std::min<int64_t>(en, x->shape[(size_t)a]));
        } else {
            st = std::max<int64_t>(0, std::min<int64_t>(st, x->shape[(size_t)a] - 1));
            en = std::max<int64_t>(-1, std::min<int64_t>(en, x->shape[(size_t)a] - 1));
        }
        sliced[(size_t)a] = true;
        st_of[(size_t)a] = st;
        en_of[(size_t)a] = en;
        step_of[(size_t)a] = step;
    }
    // The input's strides.
    std::vector<int64_t> in_stride((size_t)rank, 1);
    for (int64_t d = rank - 2; d >= 0; --d) {
        in_stride[(size_t)d] = in_stride[(size_t)d + 1] * x->shape[(size_t)d + 1];
    }
    // The output's shape: the input's, with the sliced axes *shortened*.
    //
    // The rank is kept (opset 10 and later): the sliced axes stay where they
    // are, they are only smaller.  The legacy form (opset 1) removed them,
    // and the two must not be mixed: the values are gathered with the
    // output's own rank, so a rank-reducing shape with a rank-keeping gather
    // reads the wrong elements - and reads the *first* one over and over,
    // because the output's coordinate space is empty.  That is not a
    // rounding error: the layout model reshapes with the dims it sliced out
    // of a Shape, so the whole decoder moves by a factor of the channel
    // count.
    std::vector<int64_t> out_shape = x->shape;
    for (int64_t d = 0; d < rank; ++d) {
        if (!sliced[(size_t)d]) continue;
        const int64_t step = step_of[(size_t)d];
        out_shape[(size_t)d] = step > 0 ? (en_of[(size_t)d] - st_of[(size_t)d] + step - 1) / step
                                        : (st_of[(size_t)d] - en_of[(size_t)d] + (-step) - 1) / (-step);
        if (out_shape[(size_t)d] < 0) out_shape[(size_t)d] = 0;
    }
    if (is_int(c.out[0])) {
        // A slice of a device-resident integer is a strided copy in its own
        // width.  The host walk below reads `host_i64(x)`, which is *empty*
        // for a device-authoritative int (the TopK chain never came back to
        // the host) - and an empty gather silently writes zeros, which is an
        // index of 0 for every box.
        bool negative_step = false;
        for (int64_t d = 0; d < rank; ++d) {
            if (sliced[(size_t)d] && step_of[(size_t)d] < 0) negative_step = true;
        }
        if (!negative_step && x->dev && x->host.empty() && c.out[0]->count > 0) {
            std::vector<int64_t> src(rank), dst(rank), out(rank);
            int64_t base = 0;
            for (int64_t d = 0; d < rank; ++d) {
                src[(size_t)d] = in_stride[(size_t)d];
                out[(size_t)d] = x->shape[(size_t)d];
                if (!sliced[(size_t)d]) continue;
                out[(size_t)d] = out_shape[(size_t)d];
                src[(size_t)d] = in_stride[(size_t)d] * step_of[(size_t)d];
                base += st_of[(size_t)d] * in_stride[(size_t)d];
            }
            int64_t s = 1;
            for (int64_t i = rank - 1; i >= 0; --i) {
                dst[(size_t)i] = s;
                s *= out[(size_t)i];
            }
            strided_copy_typed_launch(c.q, (const uint8_t *)x->dev + base * (int64_t)int_elem_size(c.out[0]),
                                      c.out[0]->dev, src.data(), dst.data(), out.data(), (int)rank,
                                      c.out[0]->count, (int)int_elem_size(c.out[0]));
            return;
        }
        // The values are walked on the host: the int64 ones have to be
        // there (the shape functions read them), and the gather is a
        // strided read of a handful of elements.
        ensure_host(c.q, const_cast<tensor *>(x));
        const std::vector<int64_t> iv = host_i64(x);
        const int64_t n_out = c.out[0]->count;
        const size_t esz = int_elem_size(c.out[0]);
        std::vector<uint8_t> raw((size_t)n_out * esz, 0);
        for (int64_t o = 0; o < n_out; ++o) {
            // The output's coordinates, and the input's element.
            int64_t rem = o, si = 0;
            for (int64_t d = 0; d < rank; ++d) {
                const int64_t dim = out_shape[(size_t)d];
                const int64_t co = dim > 0 ? rem % dim : 0;
                rem /= dim > 0 ? dim : 1;
                const int64_t v = sliced[(size_t)d] ? st_of[(size_t)d] + co * step_of[(size_t)d] : co;
                si += v * in_stride[(size_t)d];
            }
            const int64_t v = (si >= 0 && si < (int64_t)iv.size()) ? iv[(size_t)si] : 0;
            if (esz == 4) {
                const int32_t w = (int32_t)v;
                std::memcpy(raw.data() + o * 4, &w, 4);
            } else {
                std::memcpy(raw.data() + o * 8, &v, 8);
            }
        }
        c.out[0]->host = raw;
        c.out[0]->on_host = true;
        if (trace_on()) {
            fprintf(stderr, "    the slice's int output '%s': %zu bytes, the dtype %d\n",
                    c.out[0]->name.c_str(), c.out[0]->host.size(), (int)c.out[0]->dtype);
        }
        return;
    }
    // The same rank: a strided copy on the device.
    //
    // The source's index for the output's coordinate `a` along a sliced axis
    // is `start + a * step`, so the copy is a *strided* read with the
    // start as its base.  Without the base the second slice of a tensor
    // reads the first slice's elements again: the layout model's decoder
    // slices the anchors in two halves, and the second half came out as a
    // copy of the first.
    //
    // A negative step breaks the formulation (the source's index then
    // decreases as the coordinate increases), so that case goes through the
    // host's gather.
    bool negative_step = false;
    for (int64_t d = 0; d < rank; ++d) {
        if (sliced[(size_t)d] && step_of[(size_t)d] < 0) negative_step = true;
    }
    if (negative_step) {
        // The host's gather: the values are walked there.
        const bool ints = is_int(c.out[0]);
        const std::vector<int64_t> iv = ints ? std::vector<int64_t>() : host_i64(x);
        std::vector<float> fv;
        if (!ints) {
            fv.resize((size_t)x->count);
            q_download(c.q, (const float *)x->dev, fv.data(), fv.size() * 4);
        }
        const int64_t n_out = c.out[0]->count;
        const size_t esz = int_elem_size(c.out[0]);
        std::vector<uint8_t> raw((size_t)n_out * (ints ? esz : 4), 0);
        for (int64_t o = 0; o < n_out; ++o) {
            int64_t rem = o, si = 0;
            for (int64_t d = 0; d < rank; ++d) {
                const int64_t dim = out_shape[(size_t)d] > 0 ? out_shape[(size_t)d] : 1;
                const int64_t co = rem % dim;
                rem /= dim;
                const int64_t v = sliced[(size_t)d] ? st_of[(size_t)d] + co * step_of[(size_t)d] : co;
                si += v * in_stride[(size_t)d];
            }
            if (ints) {
                const int64_t v = (si >= 0 && si < (int64_t)iv.size()) ? iv[(size_t)si] : 0;
                if (esz == 4) {
                    const int32_t w = (int32_t)v;
                    std::memcpy(raw.data() + o * 4, &w, 4);
                } else {
                    std::memcpy(raw.data() + o * 8, &v, 8);
                }
            } else {
                const float v = (si >= 0 && si < (int64_t)fv.size()) ? fv[(size_t)si] : 0.0f;
                std::memcpy(raw.data() + o * 4, &v, 4);
            }
        }
        if (ints) {
            c.out[0]->host = raw;
            c.out[0]->on_host = true;
        } else if (n_out) {
            q_upload(c.q, raw.data(), (float *)c.out[0]->dev, (size_t)n_out * 4);
        }
        return;
    }
    std::vector<int64_t> src(rank), dst(rank), out(rank);
    for (int64_t d = 0; d < rank; ++d) out[(size_t)d] = x->shape[(size_t)d];
    // The source's strides: the sliced axes take their stride times the step,
    // the others keep theirs.  (An axis with a stride of 0 would read the
    // same element over and over, and the output came out as a repetition of
    // its first row.)
    int64_t base = 0;
    for (int64_t d = 0; d < rank; ++d) {
        src[(size_t)d] = in_stride[(size_t)d];
        if (!sliced[(size_t)d]) continue;
        out[(size_t)d] = out_shape[(size_t)d];
        src[(size_t)d] = in_stride[(size_t)d] * step_of[(size_t)d];
        base += st_of[(size_t)d] * in_stride[(size_t)d];
    }
    if (trace_on()) {
        fprintf(stderr, "[onnx] the slice's copy: the rank %lld the out", (long long)rank);
        for (int64_t d = 0; d < rank; ++d) fprintf(stderr, " %lld", (long long)out[(size_t)d]);
        fprintf(stderr, " the src");
        for (int64_t d = 0; d < rank; ++d) fprintf(stderr, " %lld", (long long)src[(size_t)d]);
        fprintf(stderr, " the base %lld\n", (long long)base);
    }
    int64_t s = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
        dst[(size_t)i] = s;
        s *= out[(size_t)i];
    }
    strided_copy_launch(c.q, (const float *)x->dev + base, (float *)c.out[0]->dev, src.data(),
                        dst.data(), out.data(), (int)rank, c.out[0]->count);
}

void exec_topk(op_ctx & c) {
    const tensor * x = c.in[0];
    // The k: an *input* since opset 10, an attribute before that.
    int64_t k = c.attr_i("k", 1);
    if (c.in.size() > 1 && c.in[1]->on_host) {
        const std::vector<int64_t> v = read_i64(c.in[1]);
        if (!v.empty()) k = v[0];
    }
    const int64_t cols = x->shape.empty() ? 0 : x->shape.back();
    if (cols <= 0) throw std::runtime_error("the TopK's input has no columns");
    const size_t rows = (size_t)(x->count / cols);
    // The values and the indices: the indices are int64, and they are
    // written by the kernel *on the device* (a host array would be freed
    // before the kernel runs).
    const size_t n = (size_t)(rows * k);
    // The indices go straight into the output's device buffer.  The old form
    // wrote them into a workspace and downloaded the workspace here, and the
    // download waits for the whole (in-order) queue - so a 38 KB index array
    // blocked the host behind every convolution that had been submitted
    // before it.  On the layout model that drain was 1.08 s per batch-16
    // call, the largest single item in the call, and the host had nothing to
    // do with the bytes.  The downstream index chain is device-side now, so
    // the value never needs a host copy here; a consumer that does need one
    // (a shape parameter, or an operator with no device path) materialises
    // it at its own point of use.
    if (!c.out[1]->dev) throw std::runtime_error("the top-k has no device buffer for its indices");
    // The `largest` attribute: 0 picks the smallest k (the layout model's
    // anchor selection), and picking the largest instead reorders every
    // anchor - the boxes come out in the wrong places.
    const bool largest = c.attr_i("largest", 1) != 0;
    topk_launch(c.q, (const float *)x->dev, (float *)c.out[0]->dev, (int64_t *)c.out[1]->dev,
                rows, cols, (int)k, largest);
    c.out[1]->count = n;
    c.out[1]->on_host = false;
}

// The host's copy of a float value, read back from the device when the
// value lives there.
std::vector<float> read_host_floats(sycl::queue & q, const tensor * t) {
    std::vector<float> v((size_t)t->count, 0.0f);
    if (!t->on_host && t->dev && v.size()) {
        q_download(q, (const float *)t->dev, v.data(), v.size() * 4);
        return v;
    }
    const size_t have = t->host.size() / 4;
    for (size_t i = 0; i < v.size() && i < have; ++i) {
        std::memcpy(&v[i], t->host.data() + i * 4, 4);
    }
    return v;
}

void exec_argmax(op_ctx & c) {
    const tensor * x = c.in[0];
    int64_t axis = c.attr_i("axis", 0);
    const int64_t rank = (int64_t)x->shape.size();
    if (axis < 0) axis += rank;
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) inner *= x->shape[i];
    const int64_t n = x->shape[axis];
    const int64_t outer = x->count / (n * inner);
    // A device-resident float input: one work-item per output position scans
    // the `n` values.  The host loop below was 14.6 ms for the UNET's
    // 3-class 1024x288 map (a download and a serial scan), and the write-back
    // downloads the int64 result, which is what the consumers need.
    if (x->dev && !x->on_host && c.out[0]->dtype == onnxpb::INT64 && c.out[0]->dev) {
        argmax_launch(c.q, (const float *)x->dev, (int64_t *)c.out[0]->dev, outer, n, inner);
        return;
    }
    // The host does it: the rows are few.
    //
    const std::vector<float> tmp = read_host_floats(c.q, x);
    std::vector<int64_t> res((size_t)(outer * inner));
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            float best = -3.0e38f;
            int64_t bi = 0;
            for (int64_t k = 0; k < n; ++k) {
                const float v = tmp[(size_t)((o * n + k) * inner + i)];
                if (k == 0 || v > best) {
                    best = v;
                    bi = k;
                }
            }
            res[(size_t)(o * inner + i)] = bi;
        }
    }
    c.out[0]->host.assign(res.size() * 8, 0);
    std::memcpy(c.out[0]->host.data(), res.data(), res.size() * 8);
    c.out[0]->on_host = true;
    c.out[0]->count = res.size();
}

void exec_not(op_ctx & c) {
    // A boolean input is one byte per element, and the device kernel reads
    // floats: the conversion is done on the host.  (The values are few - a
    // loop's condition is a scalar - and reading a one-byte buffer as floats
    // reads three bytes of whatever is next in memory.)
    if (c.in[0]->dtype != onnxpb::FLOAT) {
        const std::vector<double> av = host_values_any(c.q, c.in[0]);
        std::vector<double> res(av.size(), 0.0);
        for (size_t i = 0; i < av.size(); ++i) res[i] = av[i] == 0.0 ? 1.0 : 0.0;
        write_host_values(c, res);
        return;
    }
    ew_launch(c.q, (const float *)c.in[0]->dev, nullptr, (float *)c.out[0]->dev, c.out[0]->count,
              ew_op::logical_not, {});
}

void exec_where(op_ctx & c) {
    // The condition selects between the two values.  The device does it: the
    // host loop over the layout decoder's twelve million elements was 150
    // milliseconds a call, and the Where is a gather either way.
    const tensor * cond = c.in[0];
    const tensor * a = c.in[1];
    const tensor * b = c.in[2];
    const int64_t n = c.out[0]->count;
    if (is_int(c.out[0])) {
        // An integer value lives on the host.
        const std::vector<int64_t> cv = host_i64(cond);
        const std::vector<int64_t> av = host_i64(a);
        const std::vector<int64_t> bv = host_i64(b);
        std::vector<int64_t> res((size_t)n, 0);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t ci = i < (int64_t)cv.size() ? i : (int64_t)cv.size() - 1;
            const int64_t ai = i < (int64_t)av.size() ? i : (int64_t)av.size() - 1;
            const int64_t bi = i < (int64_t)bv.size() ? i : (int64_t)bv.size() - 1;
            res[(size_t)i] = cv[(size_t)ci] != 0 ? av[(size_t)ai] : bv[(size_t)bi];
        }
        const size_t esz = int_elem_size(c.out[0]);
        c.out[0]->host.assign((size_t)n * esz, 0);
        for (size_t i = 0; i < (size_t)n; ++i) {
            if (esz == 4) {
                const int32_t x = (int32_t)res[i];
                std::memcpy(c.out[0]->host.data() + i * 4, &x, 4);
            } else {
                std::memcpy(c.out[0]->host.data() + i * 8, &res[i], 8);
            }
        }
        c.out[0]->on_host = true;
        return;
    }
    where_launch(c.q, (const float *)cond->dev, (const float *)a->dev, (const float *)b->dev,
                 (float *)c.out[0]->dev, (size_t)n);
}


// The shape operators: the data is copied, the shape changes.
void exec_copy(op_ctx & c) {
    if (copy_i64(c)) return;
    if (copy_host_small(c)) return;
    q_copy_dev(c.q, (const float *)c.in[0]->dev, (float *)c.out[0]->dev, c.in[0]->count * 4);
}

// The ScatterElements: the updates are scattered into the data along the
// axis.  The indices select the position *along the axis* (not a point), so
// the update at [i, ...] goes to [indices[i], ...] along it.
void exec_scatter_elements(op_ctx & c) {
    const tensor * data = c.in[0];
    const tensor * idx = c.in[1];
    const tensor * upd = c.in[2];
    const bool ints = is_int(c.out[0]);
    const int64_t rank = (int64_t)data->shape.size();
    if (rank == 0) {
        const size_t esz = ints ? int_elem_size(c.out[0]) : 4;
        c.out[0]->host.assign(c.out[0]->count * esz, 0);
        c.out[0]->on_host = true;
        return;
    }
    // The classic form: the indices have the data's shape with the axis
    // dim replaced by 1, and the updates have *either* the data's shape (a
    // slice of the axis dim) *or* the indices' shape (a single value per
    // index).
    //
    //   output[b, index[b, 0, ...], c, ...] = updates[b, c, ...]      (slice)
    //   output[b, index[b, 0, ...], 0, ...] = updates[b, 0, ...]      (value)
    //
    // The wireless table structurer's decoder uses the second form: its
    // updates have the *indices'* shape, so each iteration writes one value
    // per column, and the "slice" reading (which looks for a whole axis of
    // updates) finds nothing and writes nothing at all.
    //
    // (The GatherND is a different operator and a different convention: its
    // indices' *last* dim is the number of the dims to index, and the
    // trailing block is contiguous.  Mixing the two is how the layout's
    // mask came out wrong.)
    int64_t axis = c.attr_i("axis", 0);
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) axis = 0;
    int64_t L = 1, T = 1, dim = data->shape[axis];
    for (int64_t d = 0; d < axis; ++d) L *= data->shape[d];
    for (int64_t d = axis + 1; d < rank; ++d) T *= data->shape[d];
    std::vector<int64_t> stride((size_t)rank, 1);
    for (int64_t d = rank - 2; d >= 0; --d) stride[(size_t)d] = stride[(size_t)d + 1] * data->shape[(size_t)d + 1];
    // The device path.  The value form and the slice form are both a copy of
    // the data plus one kernel; the host path below stays for the types and
    // the layouts a kernel cannot serve (a zero-length axis, a tensor with
    // no device buffer).
    const int okind = data->dtype == onnxpb::FLOAT     ? kScatterF32
                      : data->dtype == onnxpb::INT32   ? kScatterI32
                      : data->dtype == onnxpb::INT64   ? kScatterI64
                                                       : -1;
    const int ikind = idx->dtype == onnxpb::INT32   ? kScatterIdxI32
                      : idx->dtype == onnxpb::INT64 ? kScatterIdxI64
                                                    : -1;
    const bool slice_form = upd->count >= data->count && data->count > 0;
    // A small integer scatter whose inputs are host values stays on the
    // host: its output is an integer, so the device path would download it
    // with a blocking wait, and the inputs are already here.
    const bool small_int_host = ints && data->count <= (size_t)8192 && data->on_host &&
                                idx->on_host && upd->on_host;
    if (!small_int_host && okind >= 0 && ikind >= 0 && L > 0 && T > 0 && dim > 0 && data->dev &&
        idx->dev && upd->dev && c.out[0]->dev) {
        q_copy_dev(c.q, data->dev, c.out[0]->dev, data->count * runtime_elem_size(data->dtype));
        scatter_elements_launch(c.q, data->dev, idx->dev, ikind, (int64_t)idx->count, upd->dev,
                                (int64_t)upd->count, c.out[0]->dev, L, T, dim, slice_form, okind);
        // The device buffer is the value.  An integer output is still
        // downloaded by the executor's write-back (and only once, at the end
        // of the run), so this is not a host/device disagreement.
        c.out[0]->on_host = false;
        return;
    }
    const std::vector<int64_t> ids = host_i64(idx);
    const std::vector<double> u = read_host_values(c.q, upd, ints);
    std::vector<double> res = read_host_values(c.q, data, ints);
    if (trace_on()) {
        fprintf(stderr,
                "[onnx] the scatter elements: the data '%s' [%lld] %zu elements (the dtype %d), the "
                "indices '%s' %zu elements (the dtype %d), the updates '%s' %zu elements (the dtype %d), "
                "the output %zu (the dtype %d), the axis %lld\n",
                data->name.c_str(), (long long)rank, data->count, (int)data->dtype, idx->name.c_str(),
                ids.size(), (int)idx->dtype, upd->name.c_str(), upd->count, (int)upd->dtype,
                c.out[0]->count, (int)c.out[0]->dtype, (long long)axis);
        fprintf(stderr, "    the indices:");
        for (size_t i = 0; i < ids.size() && i < 8; ++i) fprintf(stderr, " %lld", (long long)ids[i]);
        fprintf(stderr, "\n    the updates:");
        for (size_t i = 0; i < u.size() && i < 8; ++i) fprintf(stderr, " %g", u[i]);
        fprintf(stderr, "\n    the data:");
        for (size_t i = 0; i < res.size() && i < 8; ++i) fprintf(stderr, " %g", res[i]);
        fprintf(stderr, "\n");
    }
    // The updates carry a slice of the axis dim, or one value per index.
    const bool host_slice_form = u.size() >= (size_t)(data->count) && data->count > 0;
    for (int64_t l = 0; l < L; ++l) {
        // The leading coordinates, in the data's dims.
        int64_t base = 0, rem = l;
        for (int64_t d = axis - 1; d >= 0; --d) {
            const int64_t q = rem % data->shape[(size_t)d];
            rem /= data->shape[(size_t)d];
            base += q * stride[(size_t)d];
        }
        for (int64_t t = 0; t < T; ++t) {
            int64_t id = (l * T + t) < (int64_t)ids.size() ? ids[(size_t)(l * T + t)] : 0;
            if (id < 0) id += dim;
            if (id < 0 || id >= dim) continue;
            if (host_slice_form) {
                for (int64_t cc = 0; cc < dim; ++cc) {
                    const int64_t dst = base + (id + cc) * stride[(size_t)axis] + t;
                    const int64_t src = ((l * dim) + cc) * T + t;
                    if (dst < (int64_t)res.size() && src < (int64_t)u.size()) {
                        res[(size_t)dst] = u[(size_t)src];
                    }
                }
            } else {
                const int64_t dst = base + id * stride[(size_t)axis] + t;
                const int64_t src = l * T + t;
                if (dst < (int64_t)res.size() && src < (int64_t)u.size()) {
                    res[(size_t)dst] = u[(size_t)src];
                }
            }
        }
    }
    write_host_values(c, res);
}

// The bitwise operators: BitwiseNot, BitwiseAnd and BitwiseOr.
//
// They work on integers, and the values are few (a formula recogniser's
// masks), so they are done on the host - the device holds float32, and a
// bitwise operation on the float bits would be nonsense.
void exec_bitwise(op_ctx & c) {
    const std::string & name = c.node.op_type;
    const tensor * x = c.in[0];
    const size_t n = c.out[0]->count;
    // The host path must see the bytes: a value the device owns (or one whose
    // producer wrote it on the device) has an empty host copy, and reading it
    // as an empty vector silently turns every operand into zero.
    ensure_host(c.q, const_cast<tensor *>(x));
    std::vector<int64_t> a = host_i64(x);
    std::vector<int64_t> b;
    if (c.in.size() > 1) {
        ensure_host(c.q, const_cast<tensor *>(c.in[1]));
        b = host_i64(c.in[1]);
    }
    const size_t esz = int_elem_size(c.out[0]);
    std::vector<uint8_t> raw((size_t)n * esz, 0);
    for (size_t i = 0; i < n; ++i) {
        int64_t v = 0;
        if (name == "BitwiseNot") {
            v = i < a.size() ? ~a[i] : 0;
        } else {
            const int64_t y = i < b.size() ? b[i] : 0;
            const int64_t x0 = i < a.size() ? a[i] : 0;
            if (name == "BitwiseAnd") v = x0 & y;
            else v = x0 | y;
        }
        if (esz == 4) {
            const int32_t w = (int32_t)v;
            std::memcpy(raw.data() + i * 4, &w, 4);
        } else {
            std::memcpy(raw.data() + i * 8, &v, 8);
        }
    }
    c.out[0]->host = raw;
    c.out[0]->on_host = true;
}

// The CumSum: the cumulative sum along an axis.
//
// The axis is an input (a constant), and the values are few, so the sum is
// done on the host.
void exec_cumsum(op_ctx & c) {
    const tensor * x = c.in[0];
    const std::vector<int64_t> & is = x->shape;
    const int64_t rank = (int64_t)is.size();
    // The axis: the second input, a constant.
    int64_t axis = 0;
    if (c.in.size() > 1) {
        const std::vector<int64_t> a = read_i64(c.in[1]);
        if (!a.empty()) axis = a[0];
    }
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) axis = 0;
    // The strides, and the number of the elements before and after the axis.
    std::vector<int64_t> stride((size_t)rank, 1);
    for (int64_t i = rank - 2; i >= 0; --i) stride[(size_t)i] = stride[(size_t)i + 1] * is[(size_t)i + 1];
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < axis; ++d) outer *= is[(size_t)d];
    for (int64_t d = axis + 1; d < rank; ++d) inner *= is[(size_t)d];
    // The `exclusive` attribute (opset 11): the sum *without* the element
    // itself.  It is 0 by default, and the formula recogniser does not use
    // it.
    const bool exclusive = c.attr_i("exclusive", 0) != 0;
    const size_t n = c.out[0]->count;
    // A device-resident integer cumsum stays on the device.  The formula
    // recogniser's stop check runs this on a 2 560-element mask every step,
    // and the host path downloaded it: the queue drain at the end of the
    // body cost 2.8 ms of an 8.1 ms step.
    if (x->dev && x->host.empty() && c.out[0]->dev && x->dtype == c.out[0]->dtype &&
        (x->dtype == onnxpb::INT64 || x->dtype == onnxpb::INT32)) {
        const int64_t axis_len = is[(size_t)axis];
        cumsum_i_launch(c.q, x->dev, c.out[0]->dev, x->dtype == onnxpb::INT64 ? 8 : 4, outer,
                        axis_len, inner, exclusive);
        c.out[0]->on_host = false;
        return;
    }
    ensure_host(c.q, const_cast<tensor *>(x));
    const std::vector<int64_t> v = host_i64(x);
    std::vector<int64_t> res((size_t)n, 0);
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            int64_t acc = 0;
            for (int64_t a = 0; a < is[(size_t)axis]; ++a) {
                const int64_t e = o * is[(size_t)axis] * inner + a * inner + i;
                const int64_t x = (e >= 0 && e < (int64_t)v.size()) ? v[(size_t)e] : 0;
                if (!exclusive) acc += x;
                if (e >= 0 && e < (int64_t)res.size()) res[(size_t)e] = acc;
                if (exclusive) acc += x;
            }
        }
    }
    // The output, in the output's type.
    const size_t esz = int_elem_size(c.out[0]);
    std::vector<uint8_t> raw((size_t)n * esz, 0);
    for (size_t i = 0; i < n; ++i) {
        if (esz == 4) {
            const int32_t w = (int32_t)res[i];
            std::memcpy(raw.data() + i * 4, &w, 4);
        } else {
            std::memcpy(raw.data() + i * 8, &res[i], 8);
        }
    }
    c.out[0]->host = raw;
    c.out[0]->on_host = true;
}

// The Constant: the value is the attribute's tensor.
//
// A constant that the executor folded does not run; this one is the
// *branch* of an If, where only one of the two branches runs and the value
// has to be produced by the selected one.
void exec_constant(op_ctx & c) {
    for (const onnxpb::attribute & a : c.node.attribute) {
        if (a.name != "value" || a.type_ != onnxpb::attribute::type::t) continue;
        for (size_t k = 0; k < c.out.size(); ++k) {
            // A boolean is a float 0/1 in this runtime.
            c.out[k]->host = a.t.data_type == onnxpb::BOOL ? bool_to_runtime_bytes(a.t.raw)
                                                            : a.t.raw;
            c.out[k]->on_host = true;
        }
    }
}

// The OneHot: a vector of zeros with a one at the index of each value.
void exec_one_hot(op_ctx & c) {
    const tensor * idx = c.in[0];
    const std::vector<int64_t> ids = host_i64(idx);
    int64_t depth = 0;
    if (c.in.size() > 1) {
        const std::vector<int64_t> v = read_i64(c.in[1]);
        if (!v.empty()) depth = v[0];
    }
    // The values: [off, on].
    double off = 0.0, on = 1.0;
    if (c.in.size() > 2) {
        const std::vector<double> v = read_host_values(c.q, c.in[2], is_int(c.in[2]));
        if (v.size() >= 2) {
            off = v[0];
            on = v[1];
        }
    }
    const bool ints = is_int(c.out[0]);
    std::vector<double> res((size_t)c.out[0]->count, off);
    for (size_t i = 0; i < ids.size(); ++i) {
        int64_t id = ids[i];
        if (id < 0) id += depth;
        if (id < 0 || id >= depth) continue;
        const size_t dst = i * (size_t)depth + (size_t)id;
        if (dst < res.size()) res[dst] = on;
    }
    const size_t esz = ints ? 8 : 4;
    c.out[0]->host.assign(res.size() * esz, 0);
    for (size_t i = 0; i < res.size(); ++i) {
        if (ints) {
            const int64_t x = (int64_t)res[i];
            std::memcpy(c.out[0]->host.data() + i * 8, &x, 8);
        } else {
            const float x = (float)res[i];
            std::memcpy(c.out[0]->host.data() + i * 4, &x, 4);
        }
    }
    c.out[0]->on_host = true;
}

void exec_expand(op_ctx & c) {
    // The value is broadcast to the output's shape.  A copy of the input
    // would fill only the input's own elements: the layout model's decoder
    // expands a scalar into the whole anchor grid, and the copy left the
    // rest of the buffer uninitialised.
    const tensor * x = c.in[0];
    const std::vector<int64_t> & is = x->shape;
    const std::vector<int64_t> & os = c.out[0]->shape;
    const size_t n = c.out[0]->count;
    if (n == 0) return;
    // The input's strides in the output's index space: an axis of size 1 is
    // broadcast (a stride of 0), the others keep theirs.  They are copied
    // out: the kernel may not capture a host pointer.
    int64_t str[8] = {0}, osz[8] = {0};
    const int rank = (int)std::min<size_t>(os.size(), 8);
    {
        int64_t s = 1;
        for (size_t i = is.size(); i-- > 0;) {
            const size_t p = os.size() - is.size() + i;
            if (p < 8) str[p] = is[i] == 1 ? 0 : s;
            s *= is[i];
        }
        for (int d = 0; d < rank; ++d) osz[d] = os[d] > 0 ? os[d] : 1;
    }
    if (is_int(c.out[0])) {
        // A device-resident integer expands on the device, in its own width.
        if (x->dev && x->host.empty() && n > 0) {
            expand_typed_launch(c.q, x->dev, c.out[0]->dev, str, osz, rank, (int64_t)n,
                                (int)int_elem_size(c.out[0]));
            return;
        }
        // An *integer* value lives on the host: the Expand of an integer (the
        // layout model's decoder expands the arange into the box indices)
        // reads a host-only buffer through the device pointer, and the values
        // come out as garbage - which the scatter then scatters, and the
        // boxes come out wrong.
        ensure_host(c.q, const_cast<tensor *>(x));
        const std::vector<int64_t> v = host_i64(x);
        const size_t esz = int_elem_size(c.out[0]);
        std::vector<uint8_t> raw(n * esz, 0);
        for (size_t i = 0; i < n; ++i) {
            int64_t rem = (int64_t)i, idx = 0;
            for (int d = rank - 1; d >= 0; --d) {
                const int64_t co = rem % osz[d];
                rem /= osz[d];
                idx += co * str[d];
            }
            const int64_t val = (idx >= 0 && idx < (int64_t)v.size()) ? v[(size_t)idx] : 0;
            if (esz == 4) {
                const int32_t w = (int32_t)val;
                std::memcpy(raw.data() + i * 4, &w, 4);
            } else {
                std::memcpy(raw.data() + i * 8, &val, 8);
            }
        }
        c.out[0]->host = raw;
        c.out[0]->on_host = true;
        return;
    }
    const float * xdev = (const float *)x->dev;
    float * y = (float *)c.out[0]->dev;
    c.q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) {
            const size_t i = it.get_id(0);
            int64_t rem = (int64_t)i, idx = 0;
            for (int d = rank - 1; d >= 0; --d) {
                const int64_t co = rem % osz[d];
                rem /= osz[d];
                idx += co * str[d];
            }
            y[i] = xdev[idx];
        });
    });
}

void exec_cast(op_ctx & c) {
    // The target type.
    int32_t to = onnxpb::FLOAT;
    for (const onnxpb::attribute & a : c.node.attribute) {
        if (a.name == "to") to = (int32_t)a.i;
    }
    // A cast *to* a boolean is a test: the result is 0 or 1 whatever the
    // value was.  (The models use it to turn a counter into a flag: the
    // loop's condition is a cast of a comparison to a boolean, and a cast
    // that passed the value through would leave the condition at 49 for
    // ever.)
    const bool to_bool = to == onnxpb::BOOL;
    // A BOOL is a float 0/1 in this runtime (see shape_cast).
    if (to_bool) to = onnxpb::FLOAT;
    const int32_t from = c.in[0]->dtype;
    const size_t n = c.in[0]->count;
    // The conversion is by *value*, not by copying the bytes: a float32 to
    // int32 cast that copies the bits gives the float's bit pattern as an
    // integer, and the layout model's mask (a float that says "inside the
    // polygon" or not) comes out as a number of the order of 1e9.
    //
    // The device does it, in both directions: a host loop over three million
    // elements is 70 milliseconds a call, and the layout model casts a dozen
    // times.  The input's device copy is the raw bytes the executor uploaded
    // (prepare_inputs), so the device reads it in its own width.
    //
    // The *output's* width is the target's, and it is what the kernel has to
    // write: an int64 to int64 cast that writes floats leaves the consumer
    // reading the float's bits as an integer.
    // A BOOL is a float 0/1 in this runtime (see shape_cast), but a cast
    // *to* a boolean is a test: the result is 0 or 1 whatever the value was.
    const int in_kind = from == onnxpb::INT64  ? kCastI64
                       : from == onnxpb::INT32  ? kCastI32
                       : from == onnxpb::BOOL   ? kCastBool
                                                : kCastF32;
    int out_kind = to == onnxpb::INT64 ? kCastI64 : to == onnxpb::INT32 ? kCastI32 : kCastF32;
    if (to_bool) out_kind = kCastToBool;
    // An integer target whose input is already on the host is converted
    // *there*.  An integer output has to reach the host whatever produced it
    // (the shape arithmetic and the Slice bounds read it), and the executor's
    // write-back used to download it with a blocking queue wait.  A device
    // cast of a scalar or a few hundred elements is then a kernel launch plus
    // a full drain; the table structurer's Loop body has fourteen such casts
    // per iteration, and they were thirty per cent of its blocking downloads.
    // The conversion is by value and in the target's width, exactly as the
    // kernel's.
    const bool int_target = to == onnxpb::INT64 || to == onnxpb::INT32;
    if (n > 0 && c.in[0]->on_host && !c.in[0]->host.empty() &&
        (int_target ? n <= (size_t)65536 : n <= (size_t)8192)) {
        const std::vector<double> v = host_values_any(c.q, c.in[0]);
        if (int_target) {
            const size_t esz = to == onnxpb::INT64 ? 8 : 4;
            c.out[0]->host.assign(n * esz, 0);
            for (size_t i = 0; i < n; ++i) {
                const int64_t x = (int64_t)(i < v.size() ? v[i] : 0.0);
                if (esz == 8) {
                    std::memcpy(c.out[0]->host.data() + i * 8, &x, 8);
                } else {
                    const int32_t y = (int32_t)x;
                    std::memcpy(c.out[0]->host.data() + i * 4, &y, 4);
                }
            }
        } else {
            // A small float target stays on the host too: the next operator
            // is usually another cast, a comparison or a boolean op, and all
            // of them read it there.  (The int-to-float casts of the
            // decoder's counters used to upload and come back.)  A cast to a
            // BOOL is a test, so the value is 0 or 1.
            c.out[0]->host.assign(n * 4, 0);
            for (size_t i = 0; i < n; ++i) {
                const double d = i < v.size() ? v[i] : 0.0;
                const float f = to_bool ? (d != 0.0 ? 1.0f : 0.0f) : (float)d;
                std::memcpy(c.out[0]->host.data() + i * 4, &f, 4);
            }
        }
        c.out[0]->on_host = true;
        return;
    }
    if (n) cast_launch(c.q, c.in[0]->dev, c.out[0]->dev, n, in_kind, out_kind);
}

void exec_clip(op_ctx & c) {
    float lo = -3.0e38f, hi = 3.0e38f;
    if (c.in.size() > 1) {
        const std::vector<float> v = read_f32(c.in[1]);
        if (!v.empty()) lo = v[0];
    }
    if (c.in.size() > 2) {
        const std::vector<float> v = read_f32(c.in[2]);
        if (!v.empty()) hi = v[0];
    }
    const size_t n = c.out[0]->count;
    const float * x = (const float *)c.in[0]->dev;
    float * y = (float *)c.out[0]->dev;
    c.q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) {
            const size_t i = it.get_id(0);
            const float v = x[i];
            y[i] = v < lo ? lo : (v > hi ? hi : v);
        });
    });
}

void exec_constant_of_shape(op_ctx & c) {
    // The value is an attribute; the default is zero.
    float v = 0.0f;
    for (const onnxpb::attribute & a : c.node.attribute) {
        if (a.name != "value" || a.type_ != onnxpb::attribute::type::t) continue;
        if (a.t.raw.size() >= 4) std::memcpy(&v, a.t.raw.data(), 4);
    }
    const size_t n = c.out[0]->count;
    float * y = (float *)c.out[0]->dev;
    c.q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) { y[it.get_id(0)] = v; });
    });
}

void exec_range(op_ctx & c) {
    // The start, the limit and the delta.  They are int64 in the models
    // that compute them (the Range's limit is the number of the boxes, and
    // it comes from a Shape).
    //
    // Two things the device does not have: fp64 (the A770's OpenCL device
    // does not list it, and a kernel that uses it fails to launch with
    // "Required aspect fp64 is not supported on the device"), and a way to
    // write an int64 range into an int64 buffer (the naive kernel writes
    // floats, and the values come out as garbage).
    //
    // The values are computed on the host: an integer value belongs there
    // anyway (the shape functions read it there, and they run before the
    // operator does), and a Range is short.
    const bool ints = is_int(c.out[0]);
    auto val = [&](size_t i, double def) -> double {
        if (i >= c.in.size() || c.in[i]->host.empty()) return def;
        if (c.in[i]->dtype == onnxpb::INT64) {
            const std::vector<int64_t> v = read_i64(c.in[i]);
            return v.empty() ? def : (double)v[0];
        }
        if (c.in[i]->dtype == onnxpb::INT32) {
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
    const size_t n = c.out[0]->count;
    if (ints) {
        const int64_t s = (int64_t)start;
        const int64_t d = (int64_t)delta;
        const size_t esz = int_elem_size(c.out[0]);
        std::vector<uint8_t> raw(n * esz, 0);
        for (size_t i = 0; i < n; ++i) {
            const int64_t v = s + d * (int64_t)i;
            if (esz == 4) {
                const int32_t w = (int32_t)v;
                std::memcpy(raw.data() + i * 4, &w, 4);
            } else {
                std::memcpy(raw.data() + i * 8, &v, 8);
            }
        }
        c.out[0]->host = raw;
        c.out[0]->on_host = true;
        return;
    }
    const float s = (float)start;
    const float d = (float)delta;
    float * y = (float *)c.out[0]->dev;
    c.q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) {
            const size_t i = it.get_id(0);
            y[i] = s + d * (float)i;
        });
    });
    (void)limit;
}

void exec_split(op_ctx & c) {
    // The data is copied into the outputs, one slice each.
    const tensor * x = c.in[0];
    int64_t axis = c.attr_i("axis", 0);
    const int64_t rank = (int64_t)x->shape.size();
    if (axis < 0) axis += rank;
    std::vector<int64_t> split = c.attr_ints("split");
    if (c.in.size() > 1 && c.in[1]->on_host) {
        const std::vector<int64_t> v = read_i64(c.in[1]);
        if (!v.empty()) split = v;
    }
    if (split.empty()) {
        const int64_t n = (int64_t)c.out.size();
        const int64_t each = x->shape[axis] / (n ? n : 1);
        for (int64_t i = 0; i < n; ++i) split.push_back(each);
    }
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= x->shape[i];
    for (int64_t i = axis + 1; i < rank; ++i) inner *= x->shape[i];
    const int64_t axis_size = x->shape[axis];
    int64_t off = 0;
    for (size_t k = 0; k < c.out.size(); ++k) {
        const int64_t n = k < split.size() ? split[k] : 0;
        const int64_t count = outer * n * inner;
        // The source is *strided*: the chunk is the k-th slice along the
        // axis, and the axis is not the outermost one.  Reading it as one
        // contiguous block takes the first element of every outer block
        // instead - which is what the layout model's decoder got: the
        // encoder's three feature levels came out interleaved.
        const float * src = (const float *)x->dev;
        float * dst = (float *)c.out[k]->dev;
        const int64_t chunk_off = off;
        c.q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(count), [=](sycl::item<1> it) {
                const int64_t idx = it.get_id(0);
                const int64_t i = idx % inner;
                const int64_t a = (idx / inner) % n;
                const int64_t o = idx / (inner * n);
                dst[idx] = src[((int64_t)o * axis_size + chunk_off + a) * inner + i];
            });
        });
        off += n;
    }
}

void exec_tile(op_ctx & c) {
    // The data is repeated: the copy walks the output's index and takes
    // the source's index modulo the input's extent.
    const tensor * x = c.in[0];
    const std::vector<int64_t> rep = read_i64(c.in.size() > 1 ? c.in[1] : nullptr);
    const int64_t rank = (int64_t)x->shape.size();
    std::vector<int64_t> out_shape = x->shape;
    // The repeats apply from the `axis` on (the default is 0).
    int64_t axis = c.attr_i("axis", 0);
    if (axis < 0) axis += rank;
    if (axis < 0 || axis > rank) axis = 0;
    for (size_t i = 0; i < rep.size(); ++i) {
        const size_t d = (size_t)axis + i;
        if (d < out_shape.size()) out_shape[d] *= rep[i];
    }
    if (is_int(c.out[0])) {
        // A tile of a device-resident integer stays on the device.  The
        // float kernel below writes four bytes per element, so an int64
        // output would come back half written; the typed kernel writes the
        // element's own width.
        if (x->dev && x->host.empty() && c.out[0]->count > 0) {
            std::vector<int64_t> ss((size_t)rank, 1), ds((size_t)rank, 1);
            int64_t s = 1;
            for (int64_t d = rank - 1; d >= 0; --d) {
                ss[(size_t)d] = s;
                s *= x->shape[(size_t)d];
            }
            s = 1;
            for (int64_t d = rank - 1; d >= 0; --d) {
                ds[(size_t)d] = s;
                s *= out_shape[(size_t)d];
            }
            tile_typed_launch(c.q, x->dev, c.out[0]->dev, ss.data(), ds.data(), out_shape.data(),
                              x->shape.data(), (int)rank, (int64_t)c.out[0]->count,
                              (int)int_elem_size(c.out[0]));
            return;
        }
        // An integer value lives on the host, and the tile of one is a
        // strided copy of a handful of elements.
        //
        // (The device kernel writes float32, so an integer output would
        // come back half written - and the layout model's anchor indices
        // are exactly that: an int64 tile of an arange.)
        ensure_host(c.q, const_cast<tensor *>(x));
        const std::vector<int64_t> src = host_i64(x);
        std::vector<int64_t> ss((size_t)rank, 1), ds((size_t)rank, 1);
        int64_t s = 1;
        for (int64_t d = rank - 1; d >= 0; --d) {
            ss[(size_t)d] = s;
            s *= x->shape[(size_t)d];
        }
        s = 1;
        for (int64_t d = rank - 1; d >= 0; --d) {
            ds[(size_t)d] = s;
            s *= out_shape[(size_t)d];
        }
        const size_t total = (size_t)c.out[0]->count;
        std::vector<int64_t> res(total, 0);
        for (size_t idx = 0; idx < total; ++idx) {
            int64_t rest = (int64_t)idx, sp = 0, dp = 0;
            for (int64_t d = rank - 1; d >= 0; --d) {
                const int64_t dim = rest % out_shape[(size_t)d];
                rest /= out_shape[(size_t)d];
                sp += (dim % x->shape[(size_t)d]) * ss[(size_t)d];
                dp += dim * ds[(size_t)d];
            }
            if (sp < (int64_t)src.size()) res[(size_t)dp] = src[(size_t)sp];
        }
        const size_t esz = int_elem_size(c.out[0]);
        c.out[0]->host.assign(res.size() * esz, 0);
        for (size_t i = 0; i < res.size(); ++i) {
            if (esz == 4) {
                const int32_t y = (int32_t)res[i];
                std::memcpy(c.out[0]->host.data() + i * 4, &y, 4);
            } else {
                std::memcpy(c.out[0]->host.data() + i * 8, &res[i], 8);
            }
        }
        c.out[0]->on_host = true;
        return;
    }
    std::vector<int64_t> src_stride(rank, 1), dst_stride(rank, 1);
    int64_t s = 1;
    for (int64_t d = rank - 1; d >= 0; --d) {
        src_stride[d] = s;
        s *= x->shape[d];
    }
    s = 1;
    for (int64_t d = rank - 1; d >= 0; --d) {
        dst_stride[d] = s;
        s *= out_shape[d];
    }
    const size_t total = c.out[0]->count;
    const float * src = (const float *)x->dev;
    float * dst = (float *)c.out[0]->dev;
    // The shapes and the strides are copied out: a std::vector is not
    // device copyable.
    int64_t os[8] = {0}, ss[8] = {0}, ds[8] = {0}, xs[8] = {0};
    const int n = rank > 8 ? 8 : (int)rank;
    for (int d = 0; d < n; ++d) {
        os[d] = out_shape[d];
        ss[d] = src_stride[d];
        ds[d] = dst_stride[d];
        xs[d] = x->shape[d];
    }
    c.q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            int64_t rest = idx;
            int64_t sp = 0, dp = 0;
            for (int d = n - 1; d >= 0; --d) {
                const int64_t dim = rest % os[d];
                rest /= os[d];
                sp += (dim % xs[d]) * ss[d];
                dp += dim * ds[d];
            }
            dst[dp] = src[sp];
        });
    });
}

// ------------------------------------------------------------- mask head

// The fused mask-to-box scan (see graph.cpp).  The bmm's output is
// [rows][H*W], the grids are [H][W], and the five results are [1][rows]:
// bool(any mask), max(grid_a), max(grid_b), min(grid_a), min(grid_b).
bool shape_mask_box(const op_ctx & c, std::vector<std::vector<int64_t>> & out) {
    if (c.in.size() < 3 || !c.in[0] || !c.in[1] || !c.in[2]) return false;
    const tensor * logits = c.in[0];
    const tensor * ga = c.in[1];
    const tensor * gb = c.in[2];
    if (ga->shape.size() != 2 || gb->shape.size() != 2) return false;
    if (ga->count == 0 || gb->count != ga->count) return false;
    if (logits->count == 0 || logits->shape.empty() || logits->count % ga->count != 0) return false;
    // The original reductions drop the two spatial axes and keep every
    // leading axis: [B,300,200,200] -> [B,300], not [1,B*300].  The layout
    // batch is 1 normally, but a batch of 2 is a valid call and the
    // consumers' Add/MatMul shapes depend on the batch axis staying there.
    if (logits->shape.back() != (int64_t)ga->count) return false;
    std::vector<int64_t> oshape(logits->shape.begin(), logits->shape.end() - 1);
    if (oshape.empty()) oshape.push_back(1);
    out.assign(c.out.size(), oshape);
    if (!c.out.empty()) c.out[0]->dtype = onnxpb::BOOL;
    for (size_t k = 1; k < c.out.size(); ++k) c.out[k]->dtype = onnxpb::FLOAT;
    return true;
}

void exec_mask_box(op_ctx & c) {
    const float * logits = (const float *)c.in[0]->dev;
    const float * ga = (const float *)c.in[1]->dev;
    const float * gb = (const float *)c.in[2]->dev;
    const int64_t area = c.in[1]->count;
    const int64_t height = c.in[1]->shape[c.in[1]->shape.size() - 2];
    const int64_t width = c.in[1]->shape.back();
    const int64_t rows = area > 0 ? (int64_t)(c.in[0]->count / area) : 0;
    float threshold = 0.0f, sentinel = 1.0e8f;
    const std::vector<float> th = c.attr_floats("threshold");
    const std::vector<float> se = c.attr_floats("sentinel");
    if (!th.empty()) threshold = th[0];
    if (!se.empty()) sentinel = se[0];
    mask_box_launch(c.q, logits, ga, gb, (float *)c.out[0]->dev, (float *)c.out[1]->dev,
                    (float *)c.out[2]->dev, (float *)c.out[3]->dev, (float *)c.out[4]->dev, rows,
                    height, width, threshold, sentinel);
}

} // namespace

// ------------------------------------------------------------- the registry

namespace {

const op_entry kOps[] = {
    // The elementwise operators.
    {"Add", shape_elementwise, exec_elementwise},
    {"Sub", shape_elementwise, exec_elementwise},
    {"Mul", shape_elementwise, exec_elementwise},
    {"Div", shape_elementwise, exec_elementwise},
    {"Pow", shape_elementwise, exec_elementwise},
    {"Neg", shape_elementwise, exec_elementwise},
    {"Exp", shape_elementwise, exec_elementwise},
    {"Log", shape_elementwise, exec_elementwise},
    {"Sqrt", shape_elementwise, exec_elementwise},
    {"Erf", shape_elementwise, exec_elementwise},
    {"Floor", shape_elementwise, exec_elementwise},
    {"Ceil", shape_elementwise, exec_elementwise},
    {"Round", shape_elementwise, exec_elementwise},
    {"Relu", shape_elementwise, exec_elementwise},
    {"Sigmoid", shape_elementwise, exec_elementwise},
    {"Tanh", shape_elementwise, exec_elementwise},
    {"HardSigmoid", shape_elementwise, exec_elementwise},
    {"HardSwish", shape_elementwise, exec_elementwise},
    {"Gelu", shape_elementwise, exec_elementwise},
    {"Silu", shape_elementwise, exec_elementwise},
    {"Softplus", shape_elementwise, exec_elementwise},
    {"Abs", shape_elementwise, exec_elementwise},
    {"Greater", shape_compare, exec_elementwise},
    {"GreaterOrEqual", shape_compare, exec_elementwise},
    {"Less", shape_compare, exec_elementwise},
    {"LessOrEqual", shape_compare, exec_elementwise},
    {"Equal", shape_compare, exec_elementwise},
    {"And", shape_compare, exec_elementwise},
    {"Or", shape_compare, exec_elementwise},
    {"Not", shape_compare, exec_not},
    {"BitwiseNot", shape_elementwise, exec_bitwise},
    {"BitwiseAnd", shape_elementwise, exec_bitwise},
    {"BitwiseOr", shape_elementwise, exec_bitwise},
    {"CumSum", shape_cumsum, exec_cumsum},
    {"Sign", shape_elementwise, exec_elementwise},
    {"Max", shape_elementwise, exec_elementwise},
    {"Min", shape_elementwise, exec_elementwise},
    {"Mod", shape_elementwise, exec_elementwise},
    {"GridSample", shape_grid_sample, exec_grid_sample},
    {"GatherND", shape_gather_nd, exec_gather_nd},
    {"ScatterND", shape_scatter_nd, exec_scatter_nd},
    {"ScatterElements", shape_scatter_elements, exec_scatter_elements},
    {"OneHot", shape_one_hot, exec_one_hot},
    {"Einsum", shape_einsum, exec_einsum},
    // The neural network operators.
    {"Conv", shape_conv, exec_conv},
    {"ConvTranspose", shape_conv_transpose, exec_conv_transpose},
    {"MaxPool", shape_pool, exec_pool},
    {"AveragePool", shape_pool, exec_pool},
    {"GlobalAveragePool", shape_global_avg_pool, exec_global_avg_pool},
    {"Resize", shape_resize, exec_resize},
    {"ReduceMean", shape_reduce, exec_reduce},
    {"ReduceMax", shape_reduce, exec_reduce},
    {"ReduceMin", shape_reduce, exec_reduce},
    {"ReduceSum", shape_reduce, exec_reduce},
    {"Concat", shape_concat, exec_concat},
    {"Softmax", shape_softmax, exec_softmax},
    {"LayerNormalization", shape_layer_norm, exec_layer_norm},
    {"MatMul", shape_matmul, exec_matmul},
    {"MaskBox", shape_mask_box, exec_mask_box},
    {"BatchNormalization", shape_same, exec_batch_norm},
    {"Gemm", shape_gemm, exec_gemm},
    // The shape operators.
    {"Identity", shape_same, exec_identity},
    {"Shape", shape_shape, exec_shape},
    {"Reshape", shape_reshape, exec_reshape},
    {"Flatten", shape_flatten, exec_flatten},
    {"Transpose", shape_transpose, exec_transpose},
    {"Gather", shape_gather, exec_gather},
    {"Slice", shape_slice, exec_slice},
    {"TopK", shape_topk, exec_topk},
    {"ArgMax", shape_argmax, exec_argmax},
    {"Where", shape_where, exec_where},
    {"Squeeze", shape_squeeze, exec_copy},
    {"Unsqueeze", shape_unsqueeze, exec_copy},
    {"Cast", shape_cast, exec_cast},
    {"Clip", shape_same, exec_clip},
    {"ConstantOfShape", shape_constant_of_shape, exec_constant_of_shape},
    {"Constant", shape_constant, exec_constant},
    {"Range", shape_range, exec_range},
    {"Split", shape_split, exec_split},
    {"Tile", shape_tile, exec_tile},
    {"Expand", shape_expand, exec_expand},
};

} // namespace

// The operators whose output's shape depends on the data.
struct size_entry {
    const char * name;
    size_fn fn;
};

const size_entry kSizeOps[] = {
    {"Resize", size_resize},
};

const size_fn * find_size_fn(const std::string & op_type) {
    for (const size_entry & e : kSizeOps) {
        if (op_type == e.name) return &e.fn;
    }
    return nullptr;
}

const op_entry * find_op(const std::string & op_type) {
    for (const op_entry & e : kOps) {
        if (op_type == e.name) return &e;
    }
    return nullptr;
}

} // namespace onnx
} // namespace si
