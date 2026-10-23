#pragma once
// The wire protocol between the ONNX service and its clients.
//
// A document pipeline moves *tensors*, not JSON: a single 800x800x3 float
// page is 7.7 MB, so a JSON/base64 encoding would spend more time in the
// decoder than the model does in the forward pass.  The protocol is therefore
// a flat little-endian binary framing with explicit lengths everywhere, and
// the tensor payloads are the raw buffers - no conversion, no copy beyond the
// one the HTTP layer already makes.
//
//   request   'S' 'I' 'O' 'N' | u32 version | u16 name_len | name | u32 n_in
//             | per input:  u16 name_len | name | u8 dtype | u8 n_dim
//                         | u32 dim[n_dim] | u64 n_bytes | payload
//   response  'S' 'I' 'O' 'R' | i32 status | u32 msg_len | msg | u32 n_out
//             | per output: u16 name_len | name | u8 dtype | u8 n_dim
//                         | u32 dim[n_dim] | u64 n_bytes | payload
//
// status 0 means success; anything else is an error and `msg` says why.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace si {
namespace onnxwire {

enum class dtype : uint8_t {
    f32 = 0,
    f16 = 1,
    i64 = 2,
    i32 = 3,
    i8 = 4,
    u8 = 5,
    boolean = 6,
};

// Bytes per element, or 0 for an unknown dtype.
size_t elem_size(dtype t);

// The name OpenVINO reports for a dtype, for the error messages.
const char * dtype_name(dtype t);

struct tensor {
    std::string name;
    dtype type = dtype::f32;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> data;

    size_t elem_count() const;
    // True when `shape` describes exactly `data` (i.e. the tensor is
    // self-consistent).  The decoder checks it, so a truncated payload is a
    // decode error rather than an out-of-bounds read later.
    bool consistent() const;
};

constexpr uint32_t kVersion = 1;
constexpr char kReqMagic[4] = {'S', 'I', 'O', 'N'};
constexpr char kRespMagic[4] = {'S', 'I', 'O', 'R'};

// Decoding.  `err` is filled on failure and the return value says whether
// the buffer parsed.  Every read is bounds-checked: the input is a network
// buffer and a malformed one must not be able to walk off the end.
bool decode_request(const uint8_t * p, size_t n, std::string & model, std::vector<tensor> & inputs,
                    std::string & err);
// Read only the model name from a request.  The router needs the routing key
// and must not copy the (potentially multi-hundred-MB) tensor payloads.
bool peek_request_model(const uint8_t * p, size_t n, std::string & model, std::string & err);
bool decode_response(const uint8_t * p, size_t n, int32_t & status, std::string & msg,
                     std::vector<tensor> & outputs, std::string & err);

// Encoding.
std::vector<uint8_t> encode_request(const std::string & model, const std::vector<tensor> & inputs);
std::vector<uint8_t> encode_response(int32_t status, const std::string & msg,
                                     const std::vector<tensor> & outputs);

} // namespace onnxwire
} // namespace si
