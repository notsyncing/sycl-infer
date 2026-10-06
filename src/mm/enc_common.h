#pragma once

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <vector>

#include "vision.h"
#include "quant.h"

// Helpers shared between the vision and audio encoder bring-up code.  They
// used to be pasted twice, once per tower; error messages use the neutral
// "encoder: " prefix.
namespace si {

inline vt bind_vt(const gguf_file & f, const std::string & name) {
    const gguf_tensor_info * ti = f.find(name);
    if (!ti) {
        throw std::runtime_error("encoder: missing tensor: " + name);
    }
    vt t;
    t.data = ti->data;
    t.type = ti->type;
    t.K = (int)ti->dims[0];
    t.N = (int)ti->n_rows();
    return t;
}
inline const float * bind_f32_opt(const gguf_file & f, const std::string & name) {
    const gguf_tensor_info * ti = f.find(name);
    if (!ti) {
        return nullptr;
    }
    if (ti->type != GGML_TYPE_F32) {
        throw std::runtime_error("encoder: expected f32 for " + name);
    }
    return (const float *)ti->data;
}
inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
inline void layer_norm(const float * x, const float * w, const float * b, float * out, int n, float eps) {
    double mean = 0;
    for (int i = 0; i < n; i++) {
        mean += x[i];
    }
    mean /= n;
    double var = 0;
    for (int i = 0; i < n; i++) {
        const double d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    const float inv = 1.0f / std::sqrt((float)var + eps);
    for (int i = 0; i < n; i++) {
        const float v = (x[i] - (float)mean) * inv;
        out[i] = b ? v * w[i] + b[i] : v * w[i];
    }
}
inline void gelu_tanh(float * x, int n) {
    const float c = 0.7978845608028654f;
    for (int i = 0; i < n; i++) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
    }
}
inline void dequant_row(const vt & t, int row, float * dst) {
    const size_t off = (size_t)row * t.K;
    switch (t.type) {
    case GGML_TYPE_F32: std::memcpy(dst, (const float *)t.data + off, sizeof(float) * t.K); break;
    case GGML_TYPE_F16:
        for (int k = 0; k < t.K; k++) {
            dst[k] = ggml_half_to_float(((const uint16_t *)t.data)[off + k]);
        }
        break;
    case GGML_TYPE_BF16:
        for (int k = 0; k < t.K; k++) {
            dst[k] = bf16_to_f32(((const uint16_t *)t.data)[off + k]);
        }
        break;
    default: throw std::runtime_error("encoder: unsupported weight type " + std::to_string(t.type));
    }
}
inline void matmul_all(const vt & w, const float * bias, const float * x, float * out, int ntok) {
    std::vector<float> row(w.K);
    for (int o = 0; o < w.N; o++) {
        dequant_row(w, o, row.data());
        const float b = bias ? bias[o] : 0.f;
        for (int t = 0; t < ntok; t++) {
            const float * xt = x + (size_t)t * w.K;
            float acc = b;
            for (int k = 0; k < w.K; k++) {
                acc += row[k] * xt[k];
            }
            out[(size_t)t * w.N + o] = acc;
        }
    }
}


} // namespace si

