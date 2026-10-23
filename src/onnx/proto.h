#pragma once
// A minimal protobuf reader for ONNX's ModelProto.
//
// ONNX stores a model as a ModelProto (a protobuf message), and the engine
// has to read it without a protobuf library: the whole point of the ONNX path
// is that the runtime is *ours*.  The subset of the schema the reader needs is
// small - the graph, the nodes, the attributes, the tensors and the value
// info - and protobuf's wire format is simple enough to read directly.
//
// The reader is deliberately strict: a malformed model must produce an error
// with a message, never a wrong number.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace si {
namespace onnxpb {

// The ONNX element types (TensorProto.DataType).
enum elem_type : int32_t {
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
    BFLOAT16 = 16,
};

// Bytes per element, 0 for the types the runtime does not carry.
size_t elem_size(int32_t t);

// The name of an element type, for the error messages.
const char * elem_name(int32_t t);

// A protobuf cursor over a byte range.
class reader {
public:
    reader() = default;
    reader(const uint8_t * p, size_t n) : p_(p), n_(n) {}

    bool done() const { return off_ >= n_; }
    size_t off() const { return off_; }
    bool ok() const { return ok_; }

    // Read the next field's tag.  Returns false at the end of the buffer or
    // on a malformed one.
    bool next(uint32_t & field, uint32_t & wire);
    uint64_t varint();
    uint32_t fixed32();
    uint64_t fixed64();
    std::string bytes();
    // A length-delimited field: its length and a cursor over its content.
    bool sub(uint32_t want, uint32_t wire, reader & out);
    // Skip a field of the given wire type.
    bool skip(uint32_t wire);

    // A repeated numeric field.  ONNX's schema is proto2, so the fields are
    // *unpacked*: every element is its own field (wire type 0 for an
    // integer, 5 for a float).  The packed form (wire type 2) is accepted
    // too, because some exporters emit it.
    template <class F> bool each_int(uint32_t want, uint32_t wire, F f) {
        if (wire == 0) {
            f(varint());
            return ok_;
        }
        return each(want, wire, [&](reader & r) { f(r.varint()); });
    }
    template <class F> bool each_float(uint32_t want, uint32_t wire, F f) {
        if (wire == 5) {
            f(fixed32());
            return ok_;
        }
        if (wire == 1) {
            f(fixed64());
            return ok_;
        }
        return each(want, wire, [&](reader & r) { f(r.fixed32()); });
    }

    // A repeated *string* (or bytes) field.  The field's content is the
    // string itself, not a message, so it is read directly - unlike a
    // repeated message, where the content is a sub-message.
    template <class F> bool each_bytes(uint32_t want, uint32_t wire, F f) {
        while (ok_ && !done()) {
            uint32_t f2, w2;
            if (!next(f2, w2)) return false;
            if (f2 == want && w2 == 2) {
                f(bytes());
            } else if (!skip(w2)) {
                return false;
            }
        }
        return true;
    }

    // A repeated field: call this for every occurrence.  The callback
    // reads the field's content; it does not have to return anything.
    template <class F> bool each(uint32_t want, uint32_t wire, F f) {
        if (wire != 2) return skip(wire);
        reader r;
        if (!sub(want, wire, r)) return false;
        while (!r.done()) {
            const size_t before = r.off();
            f(r);
            if (!r.ok()) return false;
            // The callback has to consume something, or this loops for ever.
            if (r.off() == before) return fail("a repeated field made no progress");
        }
        return true;
    }

    bool fail(const std::string & why) {
        if (ok_) err_ = why;
        ok_ = false;
        return false;
    }
    const std::string & error() const { return err_; }

private:
    const uint8_t * p_;
    size_t n_;
    size_t off_ = 0;
    bool ok_ = true;
    std::string err_;
};

// ---------------------------------------------------------------- messages

struct tensor_proto {
    std::string name;
    int32_t data_type = UNDEFINED;
    std::vector<int64_t> dims;
    // The payload, in the element type of `data_type`.
    std::vector<uint8_t> raw;
    // The external-data location, when the payload is not inline.
    std::string external_data;
    uint64_t external_offset = 0;
};

struct value_info {
    std::string name;
    int32_t elem_type = UNDEFINED;
    // -1 for a dynamic dimension, otherwise the size.
    std::vector<int64_t> shape;
    bool has_shape = false;
};

struct attribute {
    std::string name;
    enum class type { undefined, f, i, s, t, g, floats, ints, strings, tensors, graphs } type_ = type::undefined;
    float f = 0;
    int64_t i = 0;
    std::string s;
    tensor_proto t;
    std::vector<float> floats;
    std::vector<int64_t> ints;
    std::vector<std::string> strings;
    std::vector<tensor_proto> tensors;
    // A subgraph attribute (the Loop body).
    std::shared_ptr<struct graph_proto> g;
    std::vector<std::shared_ptr<struct graph_proto>> graphs;
};

struct node_proto {
    std::string name;
    std::string op_type;
    std::string domain;
    std::vector<std::string> input;
    std::vector<std::string> output;
    std::vector<attribute> attribute;
};

struct graph_proto {
    std::string name;
    std::vector<node_proto> node;
    std::vector<tensor_proto> initializer;
    std::vector<value_info> input;
    std::vector<value_info> output;
    std::vector<value_info> value_info;
};

struct model_proto {
    int64_t ir_version = 0;
    int64_t opset = 0;
    std::string producer_name;
    std::string domain;
    graph_proto graph;
    // The metadata (the "character" list the OCR and table decoders need).
    std::vector<std::pair<std::string, std::string>> metadata;
};

// Parse a model from a file.  Throws std::runtime_error on anything the
// reader cannot make sense of.
model_proto parse_model(const std::string & path);

} // namespace onnxpb
} // namespace si
