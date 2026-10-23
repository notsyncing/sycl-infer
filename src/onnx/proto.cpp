#include "proto.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace si {
namespace onnxpb {

size_t elem_size(int32_t t) {
    switch (t) {
    case FLOAT: return 4;
    case UINT8: return 1;
    case INT8: return 1;
    case UINT16: return 2;
    case INT16: return 2;
    case INT32: return 4;
    case INT64: return 8;
    case BOOL: return 1;
    case FLOAT16: return 2;
    case DOUBLE: return 8;
    case UINT32: return 4;
    case UINT64: return 8;
    case BFLOAT16: return 2;
    default: return 0;
    }
}

const char * elem_name(int32_t t) {
    switch (t) {
    case FLOAT: return "float32";
    case UINT8: return "uint8";
    case INT8: return "int8";
    case UINT16: return "uint16";
    case INT16: return "int16";
    case INT32: return "int32";
    case INT64: return "int64";
    case STRING: return "string";
    case BOOL: return "bool";
    case FLOAT16: return "float16";
    case DOUBLE: return "float64";
    case UINT32: return "uint32";
    case UINT64: return "uint64";
    case BFLOAT16: return "bfloat16";
    default: return "?";
    }
}

// -------------------------------------------------------------- the reader

bool reader::next(uint32_t & field, uint32_t & wire) {
    if (!ok_ || done()) return false;
    const uint64_t tag = varint();
    if (!ok_) return false;
    field = static_cast<uint32_t>(tag >> 3);
    wire = static_cast<uint32_t>(tag & 7);
    if (field == 0) return fail("field number 0 is not valid");
    return true;
}

uint64_t reader::varint() {
    uint64_t v = 0;
    int shift = 0;
    for (;;) {
        if (off_ >= n_) {
            fail("truncated varint");
            return 0;
        }
        const uint8_t b = p_[off_++];
        v |= (uint64_t)(b & 0x7f) << shift;
        if (!(b & 0x80)) return v;
        shift += 7;
        if (shift > 63) {
            fail("varint is too long");
            return 0;
        }
    }
}

uint32_t reader::fixed32() {
    if (off_ + 4 > n_) {
        fail("truncated 32-bit field");
        return 0;
    }
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t)p_[off_ + i] << (8 * i);
    off_ += 4;
    return v;
}

uint64_t reader::fixed64() {
    if (off_ + 8 > n_) {
        fail("truncated 64-bit field");
        return 0;
    }
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t)p_[off_ + i] << (8 * i);
    off_ += 8;
    return v;
}

std::string reader::bytes() {
    const uint64_t len = varint();
    if (!ok_) return {};
    if (len > n_ - off_) {
        fail("length-delimited field runs past the end");
        return {};
    }
    std::string s((const char *)p_ + off_, (size_t)len);
    off_ += (size_t)len;
    return s;
}

bool reader::sub(uint32_t want, uint32_t wire, reader & out) {
    (void)want;
    if (wire != 2) return fail("expected a length-delimited field");
    const uint64_t len = varint();
    if (!ok_) return false;
    if (len > n_ - off_) return fail("sub-message runs past the end");
    out = reader(p_ + off_, (size_t)len);
    off_ += (size_t)len;
    return true;
}

bool reader::skip(uint32_t wire) {
    switch (wire) {
    case 0: varint(); return ok_;
    case 1: fixed64(); return ok_;
    case 2: bytes(); return ok_;
    case 5: fixed32(); return ok_;
    default: return fail("unknown wire type " + std::to_string(wire));
    }
}

// ------------------------------------------------------------- the schema

namespace {

void read_graph(reader & r, graph_proto & g);
void read_tensor(reader & r, tensor_proto & t);
void read_string_entry(reader & r, std::string & key, uint64_t & offset);

// The attribute type codes (AttributeProto.AttributeType).
enum attr_type : int32_t {
    A_UNDEFINED = 0,
    A_FLOAT = 1,
    A_INT = 2,
    A_STRING = 3,
    A_TENSOR = 4,
    A_GRAPH = 5,
    A_SPARSE_TENSOR = 11,
    A_TYPE_PROTO = 13,
    A_FLOATS = 6,
    A_INTS = 7,
    A_STRINGS = 8,
    A_TENSORS = 9,
    A_GRAPHS = 10,
    A_SPARSE_TENSORS = 12,
    A_TYPE_PROTOS = 14,
};

void read_tensor(reader & r, tensor_proto & t) {
    uint32_t f, w;
    while (r.next(f, w)) {
        switch (f) {
        case 1: // dims
            if (w == 2) {
                reader dr;
                if (!r.sub(f, w, dr)) break;
                while (!dr.done()) t.dims.push_back((int64_t)dr.varint());
            } else {
                t.dims.push_back((int64_t)r.varint());
            }
            break;
        case 2: // data_type
            t.data_type = (int32_t)r.varint();
            break;
        case 4: // float_data
            r.each(f, w, [&](reader & d) {
                const uint32_t bits = d.fixed32();
                t.raw.resize(t.raw.size() + 4);
                std::memcpy(t.raw.data() + t.raw.size() - 4, &bits, 4);
            });
            break;
        case 5: // int32_data
            r.each(f, w, [&](reader & d) {
                const uint32_t v = (uint32_t)d.varint();
                t.raw.resize(t.raw.size() + 4);
                std::memcpy(t.raw.data() + t.raw.size() - 4, &v, 4);
            });
            break;
        case 7: // int64_data
            r.each(f, w, [&](reader & d) {
                const uint64_t v = d.varint();
                t.raw.resize(t.raw.size() + 8);
                std::memcpy(t.raw.data() + t.raw.size() - 8, &v, 8);
            });
            break;
        case 8: // name
            t.name = r.bytes();
            break;
        case 9: // raw_data
            {
                const std::string payload = r.bytes();
                t.raw.assign(payload.begin(), payload.end());
            }
            break;
        case 10: // double_data
            r.each(f, w, [&](reader & d) {
                const uint64_t bits = d.fixed64();
                t.raw.resize(t.raw.size() + 8);
                std::memcpy(t.raw.data() + t.raw.size() - 8, &bits, 8);
            });
            break;
        case 11: // uint64_data
            r.each(f, w, [&](reader & d) {
                const uint64_t v = d.varint();
                t.raw.resize(t.raw.size() + 8);
                std::memcpy(t.raw.data() + t.raw.size() - 8, &v, 8);
            });
            break;
        case 12: // doc_string
            r.bytes();
            break;
        case 13: // external_data
            r.each(f, w, [&](reader & d) { read_string_entry(d, t.external_data, t.external_offset); });
            break;
        case 14: // data_location
            r.skip(w);
            break;
        default:
            if (!r.skip(w)) return;
        }
    }
}

void read_string_entry(reader & r, std::string & key, uint64_t & offset) {
    uint32_t f, w;
    while (r.next(f, w)) {
        if (f == 1) { // key
            key = r.bytes();
        } else if (f == 2) { // value
            const std::string v = r.bytes();
            if (key == "offset") offset = strtoull(v.c_str(), nullptr, 10);
        } else if (!r.skip(w)) {
            return;
        }
    }
}

void read_value_info(reader & r, value_info & v) {
    uint32_t f, w;
    while (r.next(f, w)) {
        if (f == 1) { // name
            v.name = r.bytes();
        } else if (f == 2) { // type
            reader tr;
            if (!r.sub(f, w, tr)) break;
            uint32_t tf, tw;
            while (tr.next(tf, tw)) {
                if (tf == 1) { // tensor_type
                    reader tt;
                    if (!tr.sub(tf, tw, tt)) return;
                    uint32_t ef, ew;
                    while (tt.next(ef, ew)) {
                        if (ef == 1) { // elem_type
                            v.elem_type = (int32_t)tt.varint();
                        } else if (ef == 2) { // shape
                            reader sh;
                            if (!tt.sub(ef, ew, sh)) return;
                            v.has_shape = true;
                            uint32_t df, dw;
                            while (sh.next(df, dw)) {
                                if (df != 1) {
                                    if (!sh.skip(dw)) return;
                                    continue;
                                }
                                reader dim;
                                if (!sh.sub(df, dw, dim)) break;
                                int64_t size = -1;
                                uint32_t xf, xw;
                                while (dim.next(xf, xw)) {
                                    if (xf == 1) { // dim_value
                                        size = (int64_t)dim.varint();
                                    } else if (!dim.skip(xw)) {
                                        return;
                                    }
                                }
                                v.shape.push_back(size);
                            }
                        } else if (!tt.skip(ew)) {
                            return;
                        }
                    }
                } else if (!tr.skip(tw)) {
                    return;
                }
            }
        } else if (!r.skip(w)) {
            return;
        }
    }
}

void read_attribute(reader & r, attribute & a) {
    uint32_t f, w;
    while (r.next(f, w)) {
        switch (f) {
        case 1: a.name = r.bytes(); break;
        case 2: {
            // The float is a fixed32 whose *bits* are the value: assigning
            // the integer to the float would convert it numerically (an
            // alpha of 0.1667 became 1.04e9).
            const uint32_t bits = r.fixed32();
            std::memcpy(&a.f, &bits, 4);
            a.type_ = attribute::type::f;
            break;
        }
        case 3: a.i = (int64_t)r.varint(); a.type_ = attribute::type::i; break;
        case 4: a.s = r.bytes(); a.type_ = attribute::type::s; break;
        case 5: {
            reader tr;
            if (!r.sub(f, w, tr)) break;
            read_tensor(tr, a.t);
            a.type_ = attribute::type::t;
            break;
        }
        case 6: {
            reader tr;
            if (!r.sub(f, w, tr)) break;
            a.g = std::make_shared<graph_proto>();
            read_graph(tr, *a.g);
            a.type_ = attribute::type::g;
            break;
        }
        case 7:
            if (w == 2) {
                reader dr;
                if (!r.sub(f, w, dr)) break;
                while (!dr.done()) {
                    const uint32_t v = dr.fixed32();
                    float fv;
                    std::memcpy(&fv, &v, 4);
                    a.floats.push_back(fv);
                }
            } else {
                const uint32_t v = r.fixed32();
                float fv;
                std::memcpy(&fv, &v, 4);
                a.floats.push_back(fv);
            }
            a.type_ = attribute::type::floats;
            break;
        case 8:
            if (w == 2) {
                reader dr;
                if (!r.sub(f, w, dr)) break;
                while (!dr.done()) a.ints.push_back((int64_t)dr.varint());
            } else {
                a.ints.push_back((int64_t)r.varint());
            }
            a.type_ = attribute::type::ints;
            break;
        case 9: a.strings.push_back(r.bytes()); a.type_ = attribute::type::strings; break;
        case 10: {
            reader tr;
            if (!r.sub(f, w, tr)) break;
            tensor_proto t;
            read_tensor(tr, t);
            a.tensors.push_back(std::move(t));
            a.type_ = attribute::type::tensors;
            break;
        }
        case 11: {
            reader tr;
            if (!r.sub(f, w, tr)) break;
            auto g = std::make_shared<graph_proto>();
            read_graph(tr, *g);
            a.graphs.push_back(g);
            a.type_ = attribute::type::graphs;
            break;
        }
        case 20: // the explicit type
            {
                reader tr;
                if (!r.sub(f, w, tr)) break;
                tr.varint();
            }
            break;
        case 13: // doc_string
            r.bytes();
            break;
        case 21: // ref_attr_name
            r.bytes();
            break;
        default:
            if (!r.skip(w)) return;
        }
    }
}

void read_node(reader & r, node_proto & n) {
    uint32_t f, w;
    while (r.next(f, w)) {
        switch (f) {
        case 1: n.input.push_back(r.bytes()); break;
        case 2: n.output.push_back(r.bytes()); break;
        case 3: n.name = r.bytes(); break;
        case 4: n.op_type = r.bytes(); break;
        case 5: {
            reader tr;
            if (!r.sub(f, w, tr)) break;
            attribute a;
            read_attribute(tr, a);
            n.attribute.push_back(std::move(a));
            break;
        }
        case 6: r.bytes(); break;  // doc_string
        case 7: n.domain = r.bytes(); break;
        default:
            if (!r.skip(w)) return;
        }
    }
}

void read_graph(reader & r, graph_proto & g) {
    uint32_t f, w;
    while (r.next(f, w)) {
        switch (f) {
        case 1: r.each(f, w, [&](reader & d) {
            node_proto n;
            read_node(d, n);
            g.node.push_back(std::move(n));
        }); break;
        case 2: g.name = r.bytes(); break;
        case 5: r.each(f, w, [&](reader & d) {
            tensor_proto t;
            read_tensor(d, t);
            g.initializer.push_back(std::move(t));
        }); break;
        case 10: r.bytes(); break; // doc_string
        case 11: r.each(f, w, [&](reader & d) {
            value_info v;
            read_value_info(d, v);
            g.input.push_back(std::move(v));
        }); break;
        case 12: r.each(f, w, [&](reader & d) {
            value_info v;
            read_value_info(d, v);
            g.output.push_back(std::move(v));
        }); break;
        case 13: r.each(f, w, [&](reader & d) {
            value_info v;
            read_value_info(d, v);
            g.value_info.push_back(std::move(v));
        }); break;
        default:
            if (!r.skip(w)) return;
        }
    }
}

} // namespace

model_proto parse_model(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open the model file " + path);
    std::ostringstream ss;
    ss << file.rdbuf();
    const std::string buf = ss.str();
    if (buf.empty()) throw std::runtime_error("the model file is empty: " + path);

    model_proto m;
    reader r((const uint8_t *)buf.data(), buf.size());
    uint32_t f, w;
    while (r.next(f, w)) {
        switch (f) {
        case 1: m.ir_version = (int64_t)r.varint(); break;
        case 2: m.producer_name = r.bytes(); break;
        case 3: r.bytes(); break; // producer_version
        case 4: m.domain = r.bytes(); break;
        case 5: r.varint(); break; // model_version
        case 6: r.bytes(); break; // doc_string
        case 7: {
            reader gr;
            if (!r.sub(f, w, gr)) break;
            read_graph(gr, m.graph);
            break;
        }
        case 8: { // opset_import
            reader ir;
            if (!r.sub(f, w, ir)) break;
            int64_t version = 0;
            std::string domain;
            uint32_t xf, xw;
            while (ir.next(xf, xw)) {
                if (xf == 1) { // domain
                    domain = ir.bytes();
                } else if (xf == 2) { // version
                    version = (int64_t)ir.varint();
                } else if (!ir.skip(xw)) {
                    break;
                }
            }
            if (domain.empty() || domain == "ai.onnx") m.opset = version;
            break;
        }
        case 14: { // metadata_props
            reader mr;
            if (!r.sub(f, w, mr)) break;
            std::string key, value;
            uint64_t off = 0;
            uint32_t mf, mw;
            while (mr.next(mf, mw)) {
                if (mf == 1) key = mr.bytes();
                else if (mf == 2) value = mr.bytes();
                else if (!mr.skip(mw)) break;
            }
            (void)off;
            m.metadata.emplace_back(key, value);
            break;
        }
        default:
            if (!r.skip(w)) break;
        }
    }
    if (!r.ok()) throw std::runtime_error("the model file is malformed: " + r.error());
    if (m.graph.node.empty()) throw std::runtime_error("the model has no nodes: " + path);
    return m;
}

} // namespace onnxpb
} // namespace si
