#include "ops.h"

#include "kernels.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>

namespace si {
namespace onnx {

// ------------------------------------------------------------- attributes

const onnxpb::attribute * op_ctx::attr(const char * name) const {
    for (const auto & a : node.attribute) {
        if (a.name == name) return &a;
    }
    return nullptr;
}

bool op_ctx::has_attr(const char * name) const { return attr(name) != nullptr; }

int64_t op_ctx::attr_i(const char * name, int64_t def) const {
    const onnxpb::attribute * a = attr(name);
    if (!a) return def;
    switch (a->type_) {
    case onnxpb::attribute::type::i: return a->i;
    case onnxpb::attribute::type::f: return (int64_t)a->f;
    case onnxpb::attribute::type::s: {
        // A numeric attribute stored as a string: the schema allows it, and
        // a malformed value must not take the model down.
        try {
            return std::stoll(a->s);
        } catch (const std::exception &) {
            return def;
        }
    }
    default: return def;
    }
}

std::vector<int64_t> op_ctx::attr_ints(const char * name) const {
    const onnxpb::attribute * a = attr(name);
    if (!a) return {};
    if (a->type_ == onnxpb::attribute::type::ints) return a->ints;
    if (a->type_ == onnxpb::attribute::type::i) return {a->i};
    return {};
}

std::vector<float> op_ctx::attr_floats(const char * name) const {
    const onnxpb::attribute * a = attr(name);
    if (!a) return {};
    if (a->type_ == onnxpb::attribute::type::floats) return a->floats;
    if (a->type_ == onnxpb::attribute::type::f) return {a->f};
    return {};
}

std::string op_ctx::attr_s(const char * name) const {
    const onnxpb::attribute * a = attr(name);
    if (!a) return {};
    if (a->type_ == onnxpb::attribute::type::s) return a->s;
    return {};
}

// The host's values of an integer tensor.
//
// The width follows the tensor's type: the models mix int32 and int64 for
// the same kind of value (the axes, the starts, the indices), and reading an
// int32 buffer as int64 gives every value the next one's low half.
std::vector<int64_t> host_i64(const tensor * t) {
    std::vector<int64_t> v;
    if (!t || t->host.empty()) return v;
    if (t->dtype == onnxpb::BOOL) {
        // A boolean is a *float* 0/1 in this runtime, so it is four bytes
        // per element.  (It used to be read as one byte, which is the wire's
        // format: the low byte of 1.0f is 0x00, so a boolean that was true
        // read as false - and the formula recogniser's loop condition is
        // exactly such a boolean.)
        const size_t n = t->host.size() / 4;
        v.resize(n);
        for (size_t i = 0; i < n; ++i) {
            float x = 0.0f;
            std::memcpy(&x, t->host.data() + i * 4, 4);
            v[i] = x != 0.0f ? 1 : 0;
        }
        return v;
    }
    if (t->dtype == onnxpb::INT32) {
        const size_t n = t->host.size() / 4;
        v.resize(n);
        for (size_t i = 0; i < n; ++i) {
            int32_t x = 0;
            std::memcpy(&x, t->host.data() + i * 4, 4);
            v[i] = (int64_t)x;
        }
        return v;
    }
    const size_t n = t->host.size() / 8;
    v.resize(n);
    if (n) std::memcpy(v.data(), t->host.data(), n * 8);
    return v;
}

// The host's values of a tensor, whatever its type.
//
// A *float* value lives on the device (the executor keeps the host copy only
// for the integers and the graph's outputs), so it is read back here; an
// integer or a boolean lives on the host.
//
// The width follows the tensor's type: a boolean is one byte per element, an
// int32 four and an int64 eight.  Reading a boolean as int64 reads eight
// bytes of a one-byte buffer - and the loop's condition (a Not of a boolean
// constant) came out as garbage that way.
std::vector<double> host_values_any(sycl::queue & q, const tensor * t) {
    std::vector<double> v;
    if (!t) return v;
    if (t->dtype == onnxpb::FLOAT || t->dtype == onnxpb::BOOL) {
        // A float, and a boolean: a boolean is a float 0/1 in this runtime.
        if (t->on_host && !t->host.empty()) {
            const size_t n = t->host.size() / 4;
            v.resize(n);
            for (size_t i = 0; i < n; ++i) {
                float x = 0.0f;
                std::memcpy(&x, t->host.data() + i * 4, 4);
                v[i] = (double)x;
            }
            return v;
        }
        const size_t n = t->count;
        std::vector<float> f(n, 0.0f);
        if (n && t->dev) q_download(q, (const float *)t->dev, f.data(), n * 4);
        v.reserve(n);
        for (float x : f) v.push_back((double)x);
        return v;
    }
    const std::vector<int64_t> i = host_i64(t);
    v.reserve(i.size());
    for (int64_t x : i) v.push_back((double)x);
    return v;
}


// ------------------------------------------------------------- the shapes
std::vector<int64_t> broadcast_shapes(const std::vector<std::vector<int64_t>> & in) {
    size_t rank = 0;
    for (const auto & s : in) rank = std::max(rank, s.size());
    std::vector<int64_t> out(rank, 1);
    for (const auto & s : in) {
        const size_t off = rank - s.size();
        for (size_t i = 0; i < s.size(); ++i) {
            const int64_t a = s[i], b = out[off + i];
            if (a == b) continue;
            if (a == 1) continue;
            if (b == 1) {
                out[off + i] = a;
                continue;
            }
            // A dynamic dim: take the other one.
            if (a < 0) continue;
            if (b < 0) {
                out[off + i] = a;
                continue;
            }
            throw std::runtime_error("shapes do not broadcast");
        }
    }
    return out;
}

} // namespace onnx
} // namespace si
