#pragma once
// Local DP4A (int8 dot product with int32 accumulate) helper.
//
// oneAPI ships `sycl/ext/oneapi/dot_product.hpp`, but in this release its
// functions are not `inline`, so including it from more than one translation
// unit breaks the link.  We keep our own static-inline copy; the compiler
// recognizes the pattern and emits the hardware dp4a instruction (verified:
// ~4.8 T-MAC/s on the test machine).
//
// Semantics: acc + sum_i (int8_t) x[i] * (uint8_t) w[i]  (x signed, w unsigned)

#include <cstdint>

namespace si {

inline int32_t dp4a_s8u8(int32_t a, uint32_t b, int32_t c) {
    return c + (int32_t)(int8_t)((a >> 0) & 0xFF) * (int32_t)(uint8_t)((b >> 0) & 0xFF)
           + (int32_t)(int8_t)((a >> 8) & 0xFF) * (int32_t)(uint8_t)((b >> 8) & 0xFF)
           + (int32_t)(int8_t)((a >> 16) & 0xFF) * (int32_t)(uint8_t)((b >> 16) & 0xFF)
           + (int32_t)(int8_t)((a >> 24) & 0xFF) * (int32_t)(uint8_t)((b >> 24) & 0xFF);
}

} // namespace si
