#pragma once
// The operator library: the shapes and the kernels of the ONNX operators the
// document pipeline uses.
//
// Every operator is a pair: a *shape* function (what the outputs look like,
// given the inputs) and an *execute* function (the kernel).  The shape
// function is what lets the graph be planned before it runs - and it is also
// what the constant folder uses to evaluate the constant sub-graphs.
//
// The kernels are written in the project's own style (src/backend/gpu/kernels)
// and are compiled ahead of time with the rest of the engine: there is no
// runtime JIT anywhere in this path.  That is deliberate - the same reason
// the LLM path has an AOT build: a JIT in the middle of a document pipeline
// is a latency cliff the first request pays.

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "graph.h"
#include "onnx_activation.h"
#include "onnx_types.h"
#include "proto.h"

namespace si {
namespace onnx {

class sycl_queue; // the forward declaration is not needed: the queue is a
                  // reference in the context.

} // namespace onnx
} // namespace si

#include <sycl/sycl.hpp>

namespace si {
namespace onnx {

// The context of one operator invocation.
struct op_ctx {
    op_ctx(sycl::queue & qq, const onnxpb::node_proto & n) : q(qq), node(n) {}
    sycl::queue & q;
    const onnxpb::node_proto & node;
    // The input tensors.  A tensor is either on the device (dev != null) or
    // on the host (on_host).
    std::vector<tensor *> in;
    // The output tensors.  The shape function has already given them their
    // shape; the operator fills them.
    std::vector<tensor *> out;
    // The workspace: a scratch buffer the operator may use.
    void * workspace = nullptr;
    size_t workspace_bytes = 0;
    // A small device buffer, for the operators that have to put a converted
    // value on the device (the layout model divides a probability map by an
    // integer shape, and the device kernel reads floats).
    void * scratch = nullptr;
    size_t scratch_bytes = 0;
    // The activation fused into a convolution's epilogue, and its
    // parameters.
    std::string activation;
    std::vector<float> act_params;
    // The model's opset.  It matters: the hard sigmoid and the hard swish
    // changed their definition in opset 14 (the linear form replaced the
    // ratio form), so the same attributes mean different things.
    int64_t opset = 0;

    int64_t attr_i(const char * name, int64_t def) const;
    std::vector<int64_t> attr_ints(const char * name) const;
    std::vector<float> attr_floats(const char * name) const;
    std::string attr_s(const char * name) const;
    bool has_attr(const char * name) const;
    const onnxpb::attribute * attr(const char * name) const;
};

// The shape function: fills `shapes` with the output shapes.  False means
// "this operator's shape depends on its data" - the caller then has to ask
// the operator at run time.
using shape_fn = bool (*)(const op_ctx &, std::vector<std::vector<int64_t>> & shapes);
// The execute function.
using exec_fn = void (*)(op_ctx &);
// The *size* function: the output's shape, computed at run time.  It is
// needed by the operators whose output's size depends on the data (a
// Resize whose sizes are computed, a Range whose limit is the number of the
// boxes).  The shape function cannot answer those, because the data is not
// there yet.
using size_fn = bool (*)(op_ctx &, std::vector<std::vector<int64_t>> & shapes);

struct op_entry {
    const char * name;
    shape_fn shape;
    exec_fn exec;
};

// The registry.  A null entry means the operator is not implemented (the
// graph builder reports it by name).
const op_entry * find_op(const std::string & op_type);
// The size function of an operator whose output's shape depends on the
// data; null when there is none.
const size_fn * find_size_fn(const std::string & op_type);

// The number of elements of a shape (-1 for a dynamic dim).
inline int64_t shape_count(const std::vector<int64_t> & s) {
    int64_t n = 1;
    for (int64_t d : s) {
        if (d < 0) return -1;
        n *= d;
    }
    return n;
}

// The int64 constants an operator reads from the host.  The count is
// clamped to what the buffer actually holds: a truncated constant must not
// turn into a read past the end.
inline std::vector<int64_t> read_i64(const tensor * t) {
    std::vector<int64_t> v;
    if (!t || t->host.empty()) return v;
    // The models mix int32 and int64 for the same kind of value (the
    // axes, the starts), so the width follows the tensor's type.
    const bool is32 = t->dtype == onnxpb::INT32;
    const size_t esz = is32 ? 4 : 8;
    const size_t n = std::min<size_t>(t->count, t->host.size() / esz);
    v.resize(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = 0;
        if (is32) {
            int32_t b = 0;
            std::memcpy(&b, t->host.data() + i * 4, 4);
            a = b;
        } else {
            std::memcpy(&a, t->host.data() + i * 8, 8);
        }
        v[i] = a;
    }
    return v;
}

// The float constants an operator reads from the host.
inline std::vector<float> read_f32(const tensor * t) {
    std::vector<float> v;
    if (!t || t->host.empty()) return v;
    const size_t n = std::min<size_t>(t->count, t->host.size() / 4);
    v.resize(n);
    for (size_t i = 0; i < n; ++i) {
        float a = 0;
        std::memcpy(&a, t->host.data() + i * 4, 4);
        v[i] = a;
    }
    return v;
}

// The ONNX auto_pad values.
enum auto_pad : int64_t {
    kSameUpper = 0,
    kSameLower = 1,
    kValid = 2,
    kNotSet = 3,
};

// The auto_pad attribute, which the exporters write either as an int (the
// schema's form) or as a string ("SAME_UPPER"), so both are accepted.
struct op_ctx;

inline int64_t auto_pad_value(const op_ctx & c, const char * name = "auto_pad") {
    const onnxpb::attribute * a = c.attr(name);
    if (!a) return kNotSet;
    if (a->type_ == onnxpb::attribute::type::i) return a->i;
    if (a->type_ == onnxpb::attribute::type::s) {
        if (a->s == "SAME_UPPER") return kSameUpper;
        if (a->s == "SAME_LOWER") return kSameLower;
        if (a->s == "VALID") return kValid;
        return kNotSet;
    }
    return kNotSet;
}

// The geometry of a convolution or a pooling: the output's spatial size and
// the *effective* padding.
//
// The auto_pad attribute matters: a model that says SAME_UPPER gets an
// output of ceil(input/stride) with the padding computed from it, and a
// model that does not get the pads-based size.  Reading the pads alone
// gives the wrong size - and a size that does not match the operator's, so
// the operators disagree about the shape of their own output.
struct spatial_geom {
    int64_t out_h = 0;
    int64_t out_w = 0;
    int64_t pad_h = 0;  // the padding *before* the first element
    int64_t pad_w = 0;
};

inline spatial_geom spatial_geometry(int64_t auto_pad, int64_t H, int64_t W, int64_t kh,
                                     int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw,
                                     int64_t dh, int64_t dw) {
    spatial_geom g;
    if (auto_pad == kSameUpper || auto_pad == kSameLower) {
        g.out_h = (H + sh - 1) / sh;  // ceil
        g.out_w = (W + sw - 1) / sw;
        const int64_t need_h = (g.out_h - 1) * sh + dh * (kh - 1) + 1;
        const int64_t need_w = (g.out_w - 1) * sw + dw * (kw - 1) + 1;
        const int64_t total_h = need_h - H > 0 ? need_h - H : 0;
        const int64_t total_w = need_w - W > 0 ? need_w - W : 0;
        // SAME_UPPER puts the extra padding at the end.
        g.pad_h = (auto_pad == kSameUpper) ? total_h / 2 : total_h - total_h / 2;
        g.pad_w = (auto_pad == kSameUpper) ? total_w / 2 : total_w - total_w / 2;
        return g;
    }
    g.out_h = (H + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    g.out_w = (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    g.pad_h = ph;
    g.pad_w = pw;
    return g;
}

// The transposed convolution's output geometry.
//
// The output is bigger than the input by the stride, and the geometry is the
// mirror of the forward one:
//
//   out = (in - 1) * stride - 2 * pad + (kernel - 1) * dilation + 1
//
// plus the **output padding**: "additional elements added to the side of the
// output shape" (ONNX's ConvTranspose).  Without it the UNet's decoder comes
// out one pixel short in both directions - and a decoder that is one pixel
// short puts every box one pixel off.
//
// The `output_shape` attribute, when it is there, replaces the geometry
// altogether.
inline void conv_transpose_geom(const op_ctx & c, int64_t H, int64_t W, int64_t kh, int64_t kw,
                                int64_t * out_h, int64_t * out_w) {
    const std::vector<int64_t> strides = c.attr_ints("strides");
    const std::vector<int64_t> pads = c.attr_ints("pads");
    const std::vector<int64_t> dil = c.attr_ints("dilations");
    const std::vector<int64_t> opad = c.attr_ints("output_padding");
    const std::vector<int64_t> oshape = c.attr_ints("output_shape");
    const int64_t sh = strides.size() == 2 ? strides[0] : 1;
    const int64_t sw = strides.size() == 2 ? strides[1] : 1;
    const int64_t ph = pads.size() >= 2 ? pads[0] : 0;
    const int64_t pw = pads.size() >= 2 ? pads[1] : 0;
    const int64_t dh = dil.size() == 2 ? dil[0] : 1;
    const int64_t dw = dil.size() == 2 ? dil[1] : 1;
    const int64_t oph = opad.size() == 2 ? opad[0] : 0;
    const int64_t opw = opad.size() == 2 ? opad[1] : 0;
    int64_t h = (H - 1) * sh - 2 * ph + dh * (kh - 1) + 1 + oph;
    int64_t w = (W - 1) * sw - 2 * pw + dw * (kw - 1) + 1 + opw;
    // The explicit output shape: the spatial dims are the last two.
    if (oshape.size() >= 2) {
        h = oshape[oshape.size() - 2];
        w = oshape[oshape.size() - 1];
    }
    *out_h = h;
    *out_w = w;
}

// Whether a tensor holds integer values.
//
// The integer values (the dimensions, the axes, the indices) stay on the
// host: they are small, and they are read by the *shape* functions, which
// run before the operator does and cannot read the device.
inline bool is_int(const tensor * t) {
    return t && (t->dtype == onnxpb::INT64 || t->dtype == onnxpb::INT32);
}

// Whether a tensor holds int64 values.
inline bool is_i64(const tensor * t) { return t && t->dtype == onnxpb::INT64; }

// The size of one element of an integer tensor.
//
// A boolean is four bytes here, like a float: the runtime keeps a boolean
// as a float 0/1 (the device holds float32), so an operator that writes one
// writes four bytes per element - and the buffer it writes into is that
// size.
inline size_t int_elem_size(const tensor * t) {
    if (!t) return 8;
    if (t->dtype == onnxpb::INT32) return 4;
    if (t->dtype == onnxpb::BOOL) return 4;
    return 8;
}

// The host's values of an int64 tensor.
std::vector<int64_t> host_i64(const tensor * t);

// Fill an integer tensor's host copy from its device buffer when the value
// is device-authoritative (a TopK's indices, the index arithmetic built from
// them).  A host path that is reached with such an input would otherwise see
// an empty `host` vector and compute with zeros - which is an index of 0 for
// every element, not an error.
void ensure_host(sycl::queue & q, tensor * t);

// The host's values of a tensor, whatever its type.
//
// The comparison and the logical operators mix the types: a loop's
// condition compares an int64 counter with an int64 limit, and a model's
// mask is a float compared with a float.  The device kernel reads floats,
// so an integer input has to be converted before it reaches the device -
// and the conversion follows the *tensor's* type, not the width the value
// happens to have.
std::vector<double> host_values_any(sycl::queue & q, const tensor * t);

// A boolean value in the runtime's representation.
//
// The model's bytes are one byte per element (that is the wire's format, and
// the ONNX one); the runtime's is a float 0/1, four bytes per element, because
// the device holds float32 and the operators write floats.

// The ONNX broadcast rules for a set of shapes.
std::vector<int64_t> broadcast_shapes(const std::vector<std::vector<int64_t>> & in);

// The geometry of a Concat: the output's rank, the axis, and each input's
// shape after the promotion.
//
// numpy's rule: every input is left-padded with 1s to the output's rank, and
// the output's rank is the largest input's rank - but at least `axis + 1`, so
// that concatenating two scalars along axis 0 gives a rank-1 value.  The
// ONNX spec asks for the same rank everywhere, but Paddle's export builds
// vectors out of scalars (the shape arithmetic of the table models), and
// numpy's promotion is what every other runtime does with them.
//
// Without the promotion a rank-0 input contributes *nothing* to the output:
// the table models' shape vectors come out one element short, and the
// operator downstream (a Resize, whose sizes are the vector) then reads a
// truncated size.
struct concat_geom {
    int64_t rank = 0;
    int64_t axis = 0;
    std::vector<std::vector<int64_t>> padded;
    bool ok = false;
};

inline concat_geom concat_geometry(const std::vector<const tensor *> & in, int64_t axis) {
    concat_geom g;
    int64_t rank = 0;
    for (const tensor * t : in) rank = std::max<int64_t>(rank, (int64_t)t->shape.size());
    if (axis < 0) axis += rank > 0 ? rank : 1;
    if (axis < 0) return g;
    rank = std::max<int64_t>(rank, axis + 1);
    g.rank = rank;
    g.axis = axis;
    for (const tensor * t : in) {
        std::vector<int64_t> s = t->shape;
        while ((int64_t)s.size() < rank) s.insert(s.begin(), 1);
        g.padded.push_back(std::move(s));
    }
    g.ok = true;
    return g;
}

} // namespace onnx
} // namespace si
