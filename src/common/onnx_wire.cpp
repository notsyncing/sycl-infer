#include "onnx_wire.h"

#include <cstring>

namespace si {
namespace onnxwire {

namespace {

// A cursor over the request buffer.  Every read is bounds-checked, because
// the buffer comes off the network: a truncated or hostile payload must
// produce a decode error, never an out-of-bounds read.
struct reader {
    const uint8_t * p = nullptr;
    size_t n = 0;
    size_t off = 0;
    bool bad = false;

    bool need(size_t k) {
        if (bad || off + k > n) {
            bad = true;
            return false;
        }
        return true;
    }
    uint8_t u8() {
        if (!need(1)) return 0;
        return p[off++];
    }
    uint16_t u16() {
        if (!need(2)) return 0;
        uint16_t v = (uint16_t)p[off] | ((uint16_t)p[off + 1] << 8);
        off += 2;
        return v;
    }
    uint32_t u32() {
        if (!need(4)) return 0;
        uint32_t v = (uint32_t)p[off] | ((uint32_t)p[off + 1] << 8) | ((uint32_t)p[off + 2] << 16) |
                     ((uint32_t)p[off + 3] << 24);
        off += 4;
        return v;
    }
    uint64_t u64() {
        if (!need(8)) return 0;
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= (uint64_t)p[off + i] << (8 * i);
        off += 8;
        return v;
    }
    std::string str(uint16_t len) {
        if (!need(len)) return {};
        std::string s((const char *)p + off, len);
        off += len;
        return s;
    }
    bool bytes(uint64_t len, std::vector<uint8_t> & out) {
        if (len > (uint64_t)n || off + (size_t)len > n) {
            bad = true;
            return false;
        }
        out.assign(p + off, p + off + (size_t)len);
        off += (size_t)len;
        return true;
    }
};

struct writer {
    std::vector<uint8_t> buf;

    void u8(uint8_t v) { buf.push_back(v); }
    void u16(uint16_t v) {
        buf.push_back((uint8_t)(v & 0xff));
        buf.push_back((uint8_t)((v >> 8) & 0xff));
    }
    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) buf.push_back((uint8_t)((v >> (8 * i)) & 0xff));
    }
    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) buf.push_back((uint8_t)((v >> (8 * i)) & 0xff));
    }
    void str(const std::string & s) {
        u16((uint16_t)s.size());
        buf.insert(buf.end(), s.begin(), s.end());
    }
    void raw(const uint8_t * d, size_t n) { buf.insert(buf.end(), d, d + n); }
};

} // namespace

size_t elem_size(dtype t) {
    switch (t) {
    case dtype::f32: return 4;
    case dtype::f16: return 2;
    case dtype::i64: return 8;
    case dtype::i32: return 4;
    case dtype::i8: return 1;
    case dtype::u8: return 1;
    case dtype::boolean: return 1;
    }
    return 0;
}

const char * dtype_name(dtype t) {
    switch (t) {
    case dtype::f32: return "f32";
    case dtype::f16: return "f16";
    case dtype::i64: return "i64";
    case dtype::i32: return "i32";
    case dtype::i8: return "i8";
    case dtype::u8: return "u8";
    case dtype::boolean: return "bool";
    }
    return "?";
}

size_t tensor::elem_count() const {
    size_t n = 1;
    for (uint64_t d : shape) {
        // A shape that would overflow the element count is a malformed
        // tensor; saturate rather than wrap.
        if (d != 0 && n > (size_t)-1 / (size_t)d) return (size_t)-1;
        n *= (size_t)d;
    }
    return n;
}

bool tensor::consistent() const {
    const size_t es = elem_size(type);
    if (es == 0) return false;
    const size_t n = elem_count();
    if (n == (size_t)-1) return false;
    return n * es == data.size();
}

bool decode_request(const uint8_t * p, size_t n, std::string & model, std::vector<tensor> & inputs,
                    std::string & err) {
    reader r{p, n};
    if (n < 4 || std::memcmp(p, kReqMagic, 4) != 0) {
        err = "bad request magic";
        return false;
    }
    r.off = 4;
    const uint32_t ver = r.u32();
    if (ver != kVersion) {
        err = "unsupported protocol version " + std::to_string(ver);
        return false;
    }
    const uint16_t mlen = r.u16();
    model = r.str(mlen);
    const uint32_t n_in = r.u32();
    if (r.bad) {
        err = "truncated request header";
        return false;
    }
    // A request cannot need more inputs than it has bytes: the smallest
    // possible input costs 14 bytes of header, so this bounds the loop.
    if (n_in > n) {
        err = "implausible input count";
        return false;
    }
    inputs.clear();
    inputs.reserve(n_in);
    for (uint32_t i = 0; i < n_in; ++i) {
        tensor t;
        const uint16_t nlen = r.u16();
        t.name = r.str(nlen);
        t.type = (dtype)r.u8();
        const uint8_t nd = r.u8();
        if (r.bad) {
            err = "truncated input header";
            return false;
        }
        t.shape.resize(nd);
        for (uint8_t d = 0; d < nd; ++d) t.shape[d] = r.u32();
        const uint64_t nbytes = r.u64();
        if (r.bad) {
            err = "truncated input header";
            return false;
        }
        if (!r.bytes(nbytes, t.data)) {
            err = "truncated input payload";
            return false;
        }
        if (!t.consistent()) {
            err = "input '" + t.name + "' shape does not match its payload";
            return false;
        }
        inputs.push_back(std::move(t));
    }
    return true;
}

bool peek_request_model(const uint8_t * p, size_t n, std::string & model, std::string & err) {
    if (n < 10 || std::memcmp(p, kReqMagic, 4) != 0) {
        err = "not a request (magic/header)";
        return false;
    }
    reader r{p, n, 4, false};
    const uint32_t version = r.u32();
    if (version != kVersion) {
        err = "unsupported request version " + std::to_string(version);
        return false;
    }
    const uint16_t len = r.u16();
    if (r.bad || r.off + len > n) {
        err = "truncated model name";
        return false;
    }
    model.assign(reinterpret_cast<const char *>(p + r.off), len);
    return true;
}

bool decode_response(const uint8_t * p, size_t n, int32_t & status, std::string & msg,
                     std::vector<tensor> & outputs, std::string & err) {
    reader r{p, n};
    if (n < 4 || std::memcmp(p, kRespMagic, 4) != 0) {
        err = "bad response magic";
        return false;
    }
    r.off = 4;
    status = (int32_t)r.u32();
    const uint32_t mlen = r.u32();
    msg = r.str((uint16_t)mlen);
    const uint32_t n_out = r.u32();
    if (r.bad) {
        err = "truncated response header";
        return false;
    }
    if (n_out > n) {
        err = "implausible output count";
        return false;
    }
    outputs.clear();
    outputs.reserve(n_out);
    for (uint32_t i = 0; i < n_out; ++i) {
        tensor t;
        const uint16_t nlen = r.u16();
        t.name = r.str(nlen);
        t.type = (dtype)r.u8();
        const uint8_t nd = r.u8();
        if (r.bad) {
            err = "truncated output header";
            return false;
        }
        t.shape.resize(nd);
        for (uint8_t d = 0; d < nd; ++d) t.shape[d] = r.u32();
        const uint64_t nbytes = r.u64();
        if (r.bad) {
            err = "truncated output header";
            return false;
        }
        if (!r.bytes(nbytes, t.data)) {
            err = "truncated output payload";
            return false;
        }
        outputs.push_back(std::move(t));
    }
    return true;
}

std::vector<uint8_t> encode_request(const std::string & model, const std::vector<tensor> & inputs) {
    writer w;
    w.raw((const uint8_t *)kReqMagic, 4);
    w.u32(kVersion);
    w.str(model);
    w.u32((uint32_t)inputs.size());
    for (const tensor & t : inputs) {
        w.str(t.name);
        w.u8((uint8_t)t.type);
        w.u8((uint8_t)t.shape.size());
        for (uint64_t d : t.shape) w.u32((uint32_t)d);
        w.u64((uint64_t)t.data.size());
        if (!t.data.empty()) w.raw(t.data.data(), t.data.size());
    }
    return std::move(w.buf);
}

std::vector<uint8_t> encode_response(int32_t status, const std::string & msg,
                                     const std::vector<tensor> & outputs) {
    writer w;
    w.raw((const uint8_t *)kRespMagic, 4);
    w.u32((uint32_t)status);
    w.u32((uint32_t)msg.size());
    w.raw((const uint8_t *)msg.data(), msg.size());
    w.u32((uint32_t)outputs.size());
    for (const tensor & t : outputs) {
        w.str(t.name);
        w.u8((uint8_t)t.type);
        w.u8((uint8_t)t.shape.size());
        for (uint64_t d : t.shape) w.u32((uint32_t)d);
        w.u64((uint64_t)t.data.size());
        if (!t.data.empty()) w.raw(t.data.data(), t.data.size());
    }
    return std::move(w.buf);
}

} // namespace onnxwire
} // namespace si
