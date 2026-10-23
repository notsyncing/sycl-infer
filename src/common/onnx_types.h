#pragma once
// ONNX's boolean is one byte in the model file and a float 0/1 in the runtime
// (the device holds float32).  This is the boundary conversion, shared by the
// graph loader and the executor.

#include <cstdint>
#include <cstring>
#include <vector>

namespace si {
namespace onnx {

inline std::vector<uint8_t> bool_to_runtime_bytes(const std::vector<uint8_t> & raw) {
    std::vector<uint8_t> out(raw.size() * 4, 0);
    for (size_t i = 0; i < raw.size(); ++i) {
        const float v = raw[i] ? 1.0f : 0.0f;
        std::memcpy(out.data() + i * 4, &v, 4);
    }
    return out;
}

} // namespace onnx
} // namespace si
