#include "kernels.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>

#include "device/device_profile.h"

namespace si {
namespace onnx {

namespace {

// The elementwise functor.  The op is a runtime value, but it is uniform
// across the work-group, so the branch is free.
struct ew_functor {
    ew_op op;
    // The parameters travel *by value*: a host pointer read from a device
    // kernel faults on a discrete GPU.
    float params[4];

    float operator()(float a, float b) const {
        switch (op) {
        case ew_op::add: return a + b;
        case ew_op::sub: return a - b;
        case ew_op::mul: return a * b;
        case ew_op::div: return b != 0.0f ? a / b : 0.0f;
        case ew_op::pow: return std::pow(a, b);
        case ew_op::neg: return -a;
        case ew_op::exp: return std::exp(a);
        case ew_op::log: return std::log(a);
        case ew_op::sqrt: return std::sqrt(a);
        case ew_op::erf: return std::erf(a);
        case ew_op::floor: return std::floor(a);
        case ew_op::ceil: return std::ceil(a);
        case ew_op::round: return std::round(a);
        case ew_op::relu: return a > 0.0f ? a : 0.0f;
        case ew_op::sigmoid: {
            // Numerically stable: the exponent is never positive, so it
            // cannot overflow.  (The device's exp is a fast approximation
            // and it is badly wrong for a large negative argument: the naive
            // form turned a logit of +18 into a probability of 6e-7.)
            if (a >= 0.0f) return 1.0f / (1.0f + std::exp(-a));
            const float e = std::exp(a);
            return e / (1.0f + e);
        }
        case ew_op::tanh: return std::tanh(a);
        case ew_op::hardsigmoid: {
            // The linear form: clamp(a * x + b, 0, 1).  (The opset decides
            // what a and b are; see hardsigmoid_params.)
            const float t = params[0] * a + params[1];
            return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        }
        case ew_op::hardswish: {
            const float t = params[0] * a + params[1];
            const float c = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
            return a * c;
        }
        case ew_op::gelu: {
            const float c = 0.7978845608028654f;
            return 0.5f * a * (1.0f + std::tanh(c * (a + 0.044715f * a * a * a)));
        }
        case ew_op::silu: return a / (1.0f + std::exp(-a));
        case ew_op::softplus: return std::log1p(std::exp(a));
        case ew_op::abs: return std::fabs(a);
        case ew_op::greater: return a > b ? 1.0f : 0.0f;
        case ew_op::less: return a < b ? 1.0f : 0.0f;
        case ew_op::equal: return a == b ? 1.0f : 0.0f;
        case ew_op::greater_equal: return a >= b ? 1.0f : 0.0f;
        case ew_op::less_equal: return a <= b ? 1.0f : 0.0f;
        case ew_op::logical_and: return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f;
        case ew_op::logical_or: return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f;
        case ew_op::logical_not: return a == 0.0f ? 1.0f : 0.0f;
        case ew_op::sign: return a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : 0.0f);
        case ew_op::max_: return a > b ? a : b;
        case ew_op::min_: return a < b ? a : b;
        case ew_op::mod_: {
            // The fmod: the remainder has the sign of the dividend.  (The
            // other form, the one without the attribute, follows the
            // divisor's sign.)
            if (b == 0.0f) return 0.0f;
            return a - b * std::floor(a / b);
        }
        case ew_op::clip: {
            // The clamp: the bounds are the first two parameters.
            const float lo = params[0], hi = params[1];
            return a < lo ? lo : (a > hi ? hi : a);
        }
        case ew_op::none: return a;
        }
        return 0.0f;
    }
};

// The activation in a convolution's epilogue.
float apply_act(float v, ew_op act, const act_params_t & params) {
    switch (act) {
    case ew_op::relu: return v > 0.0f ? v : 0.0f;
    case ew_op::sigmoid: {
        // Stable form, the same one ew_functor uses: the exponent is never
        // positive, so a large negative activation cannot turn exp into inf.
        if (v >= 0.0f) return 1.0f / (1.0f + std::exp(-v));
        const float e = std::exp(v);
        return e / (1.0f + e);
    }
    case ew_op::hardsigmoid: {
        // The linear form: clamp(a * x + b, 0, 1).  (The opset decides what
        // a and b are; see hardsigmoid_params.)
        const float t = params.p[0] * v + params.p[1];
        return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    }
    case ew_op::hardswish: {
        const float t = params.p[0] * v + params.p[1];
        const float c = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        return v * c;
    }
    case ew_op::gelu: {
        const float c = 0.7978845608028654f;
        return 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
    }
    case ew_op::silu: return v / (1.0f + std::exp(-v));
    case ew_op::clip: return v < params.p[0] ? params.p[0] : (v > params.p[1] ? params.p[1] : v);
    case ew_op::tanh: return std::tanh(v);
    default: return v;
    }
}

// The convolution, with the output channels of a work-item in registers.
// The template parameter is the number of channels per work-item: a compile
// time value, so the accumulator lives in registers.
template <int TM>
void conv2d_kernel(sycl::queue & q, const float * x, const float * w, const float * bias,
                    const float * scale, const float * add, float * y, int N, int C, int H, int W,
                    int M, int Ho, int Wo, int kh, int kw, int sh, int sw, int ph, int pw, int dh,
                    int dw, int group, ew_op act, act_params_t act_params, bool probe) {
    if (Ho <= 0 || Wo <= 0) return;
    const int cg = C / group;
    const int mg = M / group;
    // A work-item's TM output channels have to be inside ONE group: a
    // depthwise convolution has one output channel per group, so the tile
    // is over (group, the channels inside the group).
    const int per_group = mg / TM > 0 ? mg / TM : 1;
    const int nblocks = group * per_group;
    const int64_t total = (int64_t)N * Ho * Wo * nblocks;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            // The index order is picked for the shape:
            //
            //   * a depthwise convolution (one channel per group, TM = 1)
            //     wants the output *column* fastest: lanes then read adjacent
            //     input pixels, while the channel-fastest order made every
            //     lane of a warp read a plane H*W floats from its neighbour.
            //     The detector and the table models are mostly depthwise;
            //   * a group-1 convolution wants the channel block fastest:
            //     every lane of a warp then reads the *same* input pixel (one
            //     cache line for the whole warp, a 32x cut in input traffic)
            //     and its own 8-float weight block (contiguous across lanes).
            //     The recogniser and the layout detector are mostly these.
            //
            // One ordering for both measured 4.6x slower on the SLANet
            // decoder once the depthwise case was fixed.
            int wo, ho, blk, n;
            if (mg > 1) {
                blk = (int)(idx % nblocks);
                const int64_t b1 = idx / nblocks;
                wo = (int)(b1 % Wo);
                const int64_t b2 = b1 / Wo;
                ho = (int)(b2 % Ho);
                n = (int)(b2 / Ho);
            } else {
                wo = (int)(idx % Wo);
                const int64_t b1 = idx / Wo;
                ho = (int)(b1 % Ho);
                const int64_t b2 = b1 / Ho;
                blk = (int)(b2 % nblocks);
                n = (int)(b2 / nblocks);
            }
            const int g = blk / per_group;          // the group
            const int mloc = (blk % per_group) * TM;  // the channel inside it
            const int m0 = g * mg + mloc;             // the first output channel
            float acc[TM];
#pragma unroll
            for (int t = 0; t < TM; ++t) acc[t] = 0.0f;
            const int64_t xn = (int64_t)n * C * H * W;
            for (int c = 0; c < cg; ++c) {
                const float * xc = x + xn + ((int64_t)(g * cg + c) * H) * W;
                // The weight is (C, kh, kw, M/g): the group's channels are
                // the first C/g of them.
                const float * wc = w + (int64_t)(g * cg + c) * kh * kw * mg + mloc;
                for (int r = 0; r < kh; ++r) {
                    const int ih = ho * sh - ph + r * dh;
                    if (ih < 0 || ih >= H) continue;
                    for (int s = 0; s < kw; ++s) {
                        const int iw = wo * sw - pw + s * dw;
                        if (iw < 0 || iw >= W) continue;
                        const float v = xc[(int64_t)ih * W + iw];
                        const float * wp = wc + ((int64_t)r * kw + s) * mg;
#pragma unroll
                        for (int t = 0; t < TM; ++t) acc[t] += v * wp[t];
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < TM; ++t) {
                const int m = m0 + t;
                float v = acc[t];
                // The order matters: the convolution's own bias, then the
                // batch norm's scale, then its bias.  A batch norm is
                // (x - mean) * gamma / sqrt(var + eps) + beta, and the
                // convolution's bias is part of x.
                if (bias) v += bias[m];
                if (scale) v *= scale[m];
                if (add) v += add[m];
                // The probe: the value *before* the activation, so the
                // convolution's own arithmetic can be told apart from the
                // epilogue's.
                y[(((int64_t)n * M + m) * Ho + ho) * Wo + wo] = probe ? v : apply_act(v, act, act_params);
            }
        });
    });
}

} // namespace

// The host copies are *synchronous*.  A queued copy reads its source buffer
// later, and the host buffers here are temporaries (a vector in the
// operator's frame): an asynchronous copy would read them after they are
// destroyed, which is a use-after-free the heap notices much later.
// (A device-to-device copy has no host buffer - see `q_copy_dev` below.)
void q_upload(sycl::queue & q, const void * src, void * dst, size_t bytes) {
    if (bytes == 0) return;
    q.memcpy(dst, src, bytes).wait();
}

std::atomic<long> g_q_downloads{0};

void q_download(sycl::queue & q, const void * src, void * dst, size_t bytes) {
    if (bytes == 0) return;
    ++g_q_downloads;
    q.memcpy(dst, src, bytes).wait();
}

void q_copy_dev(sycl::queue & q, const void * src, void * dst, size_t bytes) {
    if (bytes == 0) return;
    // **Not waited.**  A device-to-device copy is a queued command like any
    // other, and the executor's queue is in order: the next operator's
    // kernels see this copy's writes without a host round trip.  The wait
    // was 40 queue drains per table-decoder iteration (one per
    // Squeeze/Unsqueeze/Reshape, which are all this copy), and a drain waits
    // for *every* kernel ahead of it - so the host never got ahead of the
    // device and each iteration cost a full submit-plus-drain cycle.
    q.memcpy(dst, src, bytes);
}

// The elementwise kernel.
//
// The device holds float32, so this reads floats.  An integer operand is
// converted before it gets here (the operator does it: an integer value
// lives on the host, and a comparison of two int64 masks has to be an
// integer comparison - reading an integer as a float reads its bits).
void ew_launch(sycl::queue & q, const float * a, const float * b, float * y, size_t n, ew_op op,
               const std::vector<float> & params, bool probe, int scalar_op, ew_op act,
               act_params_t ap) {
    if (n == 0) return;
    float p[4] = {-3.0f, 3.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < params.size() && i < 4; ++i) p[i] = params[i];
    const act_params_t actp = ap;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) {
            const size_t i = it.get_id(0);
            // The scalar operand is read once, and the other one is read per
            // element.
            const float x = scalar_op == 0 ? (a ? a[0] : 0.0f) : a[i];
            const float z = b ? (scalar_op == 1 ? b[0] : b[i]) : 0.0f;
            ew_functor f{op, {p[0], p[1], p[2], p[3]}};
            const float v = f(x, z);
            y[i] = probe ? x : apply_act(v, act, actp);  // the probe: echo the input
        });
    });
}

void cast_launch(sycl::queue & q, const void * x, void * y, size_t n, int in_kind, int out_kind) {
    if (n == 0) return;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) {
            const size_t i = it.get_id(0);
            // The input, in its own width.  A float, not a double: the
            // device has no fp64 (the A770 rejects a kernel that uses it),
            // and the runtime is f32 throughout.
            int64_t iv = 0;
            float dv = 0.0f;
            if (in_kind == kCastF32) {
                dv = ((const float *)x)[i];
            } else if (in_kind == kCastI32) {
                iv = (int64_t)((const int32_t *)x)[i];
            } else if (in_kind == kCastBool) {
                // A boolean is a float 0/1 in this runtime, so it is read as
                // a float (the model's one-byte form is converted at the
                // boundary).
                iv = ((const float *)x)[i] != 0.0f ? 1 : 0;
            } else {
                iv = ((const int64_t *)x)[i];
            }
            // The output, in its own width.  The float to an integer
            // conversion truncates toward zero, which is what ONNX's Cast
            // does.
            //
            // A cast *to* a boolean is not a conversion but a test: the
            // result is 0 or 1 whatever the value was.  (The value of a
            // boolean is its truth, and the models use the cast to turn a
            // counter into a flag: the loop's condition is a cast of a
            // comparison to a boolean.)
            if (out_kind == kCastU8) {
                const int64_t v = in_kind == kCastF32 ? (int64_t)dv : iv;
                ((uint8_t *)y)[i] = (uint8_t)v;
            } else if (out_kind == kCastToBool) {
                const int64_t v = in_kind == kCastF32 ? (dv != 0.0f ? 1 : 0) : (iv != 0 ? 1 : 0);
                ((float *)y)[i] = (float)v;
            } else if (out_kind == kCastF32) {
                ((float *)y)[i] = in_kind == kCastF32 ? dv : (float)iv;
            } else if (out_kind == kCastI32) {
                const int64_t v = in_kind == kCastF32 ? (int64_t)dv : iv;
                ((int32_t *)y)[i] = (int32_t)v;
            } else {
                const int64_t v = in_kind == kCastF32 ? (int64_t)dv : iv;
                ((int64_t *)y)[i] = v;
            }
        });
    });
}

void where_launch(sycl::queue & q, const float * cond, const float * a, const float * b, float * y,
                  size_t n) {
    if (n == 0) return;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) {
            const size_t i = it.get_id(0);
            // The condition is a float 0/1 (the runtime has no boolean on the
            // device), so "is it non-zero" is the test.
            y[i] = cond[i] != 0.0f ? a[i] : b[i];
        });
    });
}

void ew_bcast_launch(sycl::queue & q, const float * a, const int64_t * a_stride, const float * b,
                     const int64_t * b_stride, float * y, const int64_t * out_shape, int rank,
                     size_t count, ew_op op, const std::vector<float> & params, ew_op act,
                     act_params_t ap) {
    if (count == 0) return;
    float p[4] = {-3.0f, 3.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < params.size() && i < 4; ++i) p[i] = params[i];
    const act_params_t actp = ap;
    // The strides and the shape are copied out: the kernel may not capture
    // a host pointer (and a std::vector is not device copyable).
    int64_t as[8] = {0}, bs[8] = {0}, os[8] = {0};
    const int n = rank > 8 ? 8 : rank;
    for (int d = 0; d < n; ++d) {
        as[d] = a_stride[d];
        bs[d] = b_stride ? b_stride[d] : 0;
        os[d] = out_shape[d];
    }
    // The elementwise broadcast that matters: one operand is constant over a
    // long *contiguous* suffix of the output (the model's normalize divides
    // each NCHW channel by one scalar over H*W elements).  The generic
    // kernel decomposes every element's coordinates once per axis - four
    // 64-bit divisions for an 800x800 image - which measured 14 ms for 1.9M
    // elements.  A 2D range gives the run index and the position inside the
    // run with no division at all, and the run is coalesced.
    if (b) {
        int suffix = 0;
        int64_t inner = 1, a_expected = 1;
        for (int d = n - 1; d >= 0; --d) {
            if (bs[d] != 0) break;
            if (as[d] != a_expected) break;
            inner *= os[d];
            a_expected *= os[d];
            ++suffix;
        }
        if (suffix > 0 && inner >= 64) {
            const int64_t outer = (int64_t)count / inner;
            if (outer * inner == (int64_t)count) {
                q.submit([&](sycl::handler & h) {
                    h.parallel_for(sycl::range<2>((size_t)outer, (size_t)inner),
                                   [=](sycl::item<2> it) {
                                       const int64_t o = (int64_t)it.get_id(0);
                                       const int64_t r = (int64_t)it.get_id(1);
                                       int64_t rest = o, ao = 0, bo = 0;
                                       for (int d = n - suffix - 1; d >= 0; --d) {
                                           const int64_t dim = os[d] > 0 ? os[d] : 1;
                                           const int64_t c = rest % dim;
                                           rest /= dim;
                                           ao += c * as[d];
                                           bo += c * bs[d];
                                       }
                                       ew_functor f{op, {p[0], p[1], p[2], p[3]}};
                                       y[o * inner + r] =
                                           apply_act(f(a[ao + r], b[bo]), act, actp);
                                   });
                });
                return;
            }
        }
    }
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(count), [=](sycl::item<1> it) {
            const size_t idx = it.get_id(0);
            int64_t rest = (int64_t)idx;
            int64_t ao = 0, bo = 0;
            for (int d = n - 1; d >= 0; --d) {
                const int64_t dim = rest % os[d];
                rest /= os[d];
                ao += dim * as[d];
                bo += dim * bs[d];
            }
            ew_functor f{op, {p[0], p[1], p[2], p[3]}};
            y[idx] = apply_act(f(a[ao], b ? b[bo] : 0.0f), act, actp);
        });
    });
}

void bias_act_launch(sycl::queue & q, float * y, const float * bias, size_t n, int64_t width,
                     ew_op act, act_params_t ap) {
    if (!y || n == 0 || (bias && width <= 0)) return;
    const int64_t w = width;
    const act_params_t actp = ap;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) {
            const size_t i = it.get_id(0);
            float v = y[i];
            if (bias) v += bias[(int64_t)(i % (size_t)w)];
            y[i] = apply_act(v, act, actp);
        });
    });
}

void mask_box_launch(sycl::queue & q, const float * logits, const float * grid_a,
                     const float * grid_b, float * any, float * max_a, float * max_b,
                     float * min_a, float * min_b, int64_t rows, int64_t height, int64_t width,
                     float threshold, float sentinel) {
    if (!logits || !grid_a || !grid_b || rows <= 0 || height <= 0 || width <= 0) return;
    const int64_t area = height * width;
    constexpr int WG = 256;
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> part(5 * WG, h);
        h.parallel_for(sycl::nd_range<1>((size_t)rows * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t row = it.get_group(0);
            const int lane = (int)it.get_local_id(0);
            const float * p = logits + row * area;
            float vany = 0.0f;
            float vmax_a = -INFINITY, vmax_b = -INFINITY;
            float vmin_a = sentinel, vmin_b = sentinel;
            for (int64_t k = lane; k < area; k += WG) {
                if (p[k] > threshold) {
                    vany = 1.0f;
                    const float a = grid_a[k];
                    const float b = grid_b[k];
                    vmax_a = a > vmax_a ? a : vmax_a;
                    vmax_b = b > vmax_b ? b : vmax_b;
                    vmin_a = a < vmin_a ? a : vmin_a;
                    vmin_b = b < vmin_b ? b : vmin_b;
                }
            }
            part[0 * WG + lane] = vany;
            part[1 * WG + lane] = vmax_a;
            part[2 * WG + lane] = vmax_b;
            part[3 * WG + lane] = vmin_a;
            part[4 * WG + lane] = vmin_b;
            it.barrier(sycl::access::fence_space::local_space);
            for (int stride = WG / 2; stride > 0; stride /= 2) {
                if (lane < stride) {
                    const int a = lane, b = lane + stride;
                    part[0 * WG + a] = part[0 * WG + b] > part[0 * WG + a]
                                           ? part[0 * WG + b] : part[0 * WG + a];
                    part[1 * WG + a] = part[1 * WG + b] > part[1 * WG + a]
                                           ? part[1 * WG + b] : part[1 * WG + a];
                    part[2 * WG + a] = part[2 * WG + b] > part[2 * WG + a]
                                           ? part[2 * WG + b] : part[2 * WG + a];
                    part[3 * WG + a] = part[3 * WG + b] < part[3 * WG + a]
                                           ? part[3 * WG + b] : part[3 * WG + a];
                    part[4 * WG + a] = part[4 * WG + b] < part[4 * WG + a]
                                           ? part[4 * WG + b] : part[4 * WG + a];
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            if (lane == 0) {
                any[row] = part[0 * WG];
                max_a[row] = part[0 * WG] != 0.0f ? part[1 * WG] : 0.0f;
                max_b[row] = part[0 * WG] != 0.0f ? part[2 * WG] : 0.0f;
                min_a[row] = part[3 * WG];
                min_b[row] = part[4 * WG];
            }
        });
    });
}

void conv2d_launch(sycl::queue & q, const float * x, const float * w, const float * bias,
                   const float * scale, const float * add, float * y, int N, int C, int H, int W,
                   int M, int Ho, int Wo, int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw,
                   int group, int tm, ew_op act, act_params_t act_params,
                   bool probe) {
    const act_params_t p = act_params;
    // The number of channels per work-item: a divisor of the output
    // channels per group, and a compile-time value so the accumulator stays
    // in registers.
    // The channels per work-item: a divisor of the group's channels (a
    // depthwise convolution has one, so it gets one).
    const int mg = (M / group) > 0 ? M / group : 1;
    int t = 8;
    while (t > 1 && (t > tm || mg % t != 0)) t /= 2;
    if (t > tm) t = 1;
    switch (t) {
    case 8: conv2d_kernel<8>(q, x, w, bias, scale, add, y, N, C, H, W, M, Ho, Wo, kh, kw, sh, sw, ph, pw, dh, dw, group, act, p, probe); break;
    case 4: conv2d_kernel<4>(q, x, w, bias, scale, add, y, N, C, H, W, M, Ho, Wo, kh, kw, sh, sw, ph, pw, dh, dw, group, act, p, probe); break;
    case 2: conv2d_kernel<2>(q, x, w, bias, scale, add, y, N, C, H, W, M, Ho, Wo, kh, kw, sh, sw, ph, pw, dh, dw, group, act, p, probe); break;
    default: conv2d_kernel<1>(q, x, w, bias, scale, add, y, N, C, H, W, M, Ho, Wo, kh, kw, sh, sw, ph, pw, dh, dw, group, act, p, probe); break;
    }
}

void conv_epilogue_launch(sycl::queue & q, float * y, int N, int C, int H, int W,
                          const float * bias, const float * scale, const float * add, ew_op act,
                          act_params_t act_params) {
    const int64_t hw = (int64_t)H * W;
    const int64_t n = (int64_t)N * C * hw;
    if (n <= 0) return;
    const act_params_t p = act_params;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) {
            const int64_t i = it.get_id(0);
            const int c = (int)((i / hw) % C);
            float v = y[i];
            // The order matters: the convolution's own bias, then the batch
            // norm's scale, then its bias - the same sequence conv2d_kernel
            // applies.
            if (bias) v += bias[c];
            if (scale) v *= scale[c];
            if (add) v += add[c];
            y[i] = apply_act(v, act, p);
        });
    });
}

void conv_transpose2d_launch(sycl::queue & q, const float * x, const float * w, const float * bias,
                              const float * scale, const float * add, float * y, int N, int C, int H,
                              int W, int M, int kh, int kw, int sh, int sw, int ph, int pw, int dh,
                              int dw, int group, int Ho, int Wo, ew_op act, act_params_t act_params) {
    if (Ho <= 0 || Wo <= 0) return;
    const int cg = C / group;
    const int mg = M / group;
    const int64_t total = (int64_t)N * M * Ho * Wo;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            const int wo = (int)(idx % Wo);
            const int64_t t1 = idx / Wo;
            const int ho = (int)(t1 % Ho);
            const int64_t t2 = t1 / Ho;
            const int m = (int)(t2 % M);
            const int n = (int)(t2 / M);
            const int g = m / mg;
            const int ml = m - g * mg;
            float acc = 0.0f;
            // The input positions that land on this output pixel.
            for (int c = 0; c < cg; ++c) {
                for (int r = 0; r < kh; ++r) {
                    const int ih = (ho + ph - r * dh);
                    if (ih < 0 || ih % sh != 0) continue;
                    const int iy = ih / sh;
                    if (iy < 0 || iy >= H) continue;
                    for (int s = 0; s < kw; ++s) {
                        const int iw = (wo + pw - s * dw);
                        if (iw < 0 || iw % sw != 0) continue;
                        const int ix = iw / sw;
                        if (ix < 0 || ix >= W) continue;
                        const float v = x[(((int64_t)n * C + g * cg + c) * H + iy) * W + ix];
                        // The weight is (C, M/g, kh, kw), pre-transposed to
                        // (kh, kw, C, M/g).
                        // The weight is (kh, kw, C, M/g).
                        const float wv = w[(((int64_t)r * kw + s) * C + (g * cg + c)) * mg + ml];
                        acc += v * wv;
                    }
                }
            }
            if (bias) acc += bias[m];
            if (scale) acc *= scale[m];
            if (add) acc += add[m];
            acc = apply_act(acc, act, act_params);
            y[idx] = acc;
        });
    });
}

void pool2d_launch(sycl::queue & q, const float * x, float * y, int N, int C, int H, int W, int Ho,
                   int Wo, int kh, int kw, int sh, int sw, int ph, int pw, bool average,
                   bool count_include_pad) {
    if (Ho <= 0 || Wo <= 0) return;
    const int64_t total = (int64_t)N * C * Ho * Wo;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            const int wo = (int)(idx % Wo);
            const int64_t t1 = idx / Wo;
            const int ho = (int)(t1 % Ho);
            const int64_t t2 = t1 / Ho;
            const int c = (int)(t2 % C);
            const int n = (int)(t2 / C);
            const float * xp = x + (((int64_t)n * C + c) * H) * W;
            float acc = average ? 0.0f : -3.0e38f;
            int cnt = 0;
            for (int r = 0; r < kh; ++r) {
                const int ih = ho * sh - ph + r;
                if (ih < 0 || ih >= H) {
                    if (average && count_include_pad) ++cnt;
                    continue;
                }
                for (int s = 0; s < kw; ++s) {
                    const int iw = wo * sw - pw + s;
                    if (iw < 0 || iw >= W) {
                        if (average && count_include_pad) ++cnt;
                        continue;
                    }
                    const float v = xp[(int64_t)ih * W + iw];
                    if (average) {
                        acc += v;
                    } else if (v > acc) {
                        acc = v;
                    }
                    ++cnt;
                }
            }
            if (average) acc = cnt ? acc / (float)cnt : 0.0f;
            y[idx] = acc;
        });
    });
}

void resize_launch(sycl::queue & q, const float * x, float * y, int N, int C, int H, int W, int Ho,
                   int Wo, bool nearest) {
    const int64_t total = (int64_t)N * C * Ho * Wo;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            const int wo = (int)(idx % Wo);
            const int64_t t1 = idx / Wo;
            const int ho = (int)(t1 % Ho);
            const int64_t t2 = t1 / Ho;
            const int c = (int)(t2 % C);
            const int n = (int)(t2 / C);
            const float * xp = x + (((int64_t)n * C + c) * H) * W;
            float v = 0.0f;
            if (nearest) {
                // The source index is floor(wo * W / Wo) for
                // coordinate_transformation_mode=asymmetric with
                // nearest_mode=floor.  **Integer arithmetic, not float**:
                // `wo * W / Wo` in float is compiled to a reciprocal
                // multiply, and on this device the reciprocal is slightly
                // below 1/60 - so at an exact multiple (upsampling 30 -> 60,
                // wo = 2) the floor came out one short and the first source
                // pixel took three columns instead of two.  That shifted the
                // whole detector probability map by a pixel (and the table
                // text boxes with it).
                const int ih = (int)(((int64_t)ho * H) / Ho);
                const int iw = (int)(((int64_t)wo * W) / Wo);
                v = xp[(int64_t)ih * W + iw];
            } else {
                // Bilinear, align_corners = false: the source coordinate is
                // (i + 0.5) * scale - 0.5.
                const float fy = ((float)ho + 0.5f) * (float)H / (float)Ho - 0.5f;
                const float fx = ((float)wo + 0.5f) * (float)W / (float)Wo - 0.5f;
                const int y0 = (int)std::floor(fy);
                const int x0 = (int)std::floor(fx);
                const float dy = fy - (float)y0;
                const float dx = fx - (float)x0;
                auto at = [&](int yy, int xx) -> float {
                    const int cy = yy < 0 ? 0 : (yy >= H ? H - 1 : yy);
                    const int cx = xx < 0 ? 0 : (xx >= W ? W - 1 : xx);
                    return xp[(int64_t)cy * W + cx];
                };
                v = at(y0, x0) * (1 - dy) * (1 - dx) + at(y0, x0 + 1) * (1 - dy) * dx +
                    at(y0 + 1, x0) * dy * (1 - dx) + at(y0 + 1, x0 + 1) * dy * dx;
            }
            y[idx] = v;
        });
    });
}

void global_avg_pool_launch(sycl::queue & q, const float * x, float * y, int N, int C, int H,
                            int W) {
    const int64_t total = (int64_t)N * C;
    const int64_t hw = (int64_t)H * W;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t c = it.get_id(0);
            const float * xp = x + c * hw;
            float acc = 0.0f;
            for (int64_t i = 0; i < hw; ++i) acc += xp[i];
            y[c] = hw ? acc / (float)hw : 0.0f;
        });
    });
}

void argmax_launch(sycl::queue & q, const float * x, int64_t * y, int64_t outer, int64_t n,
                   int64_t inner) {
    if (outer <= 0 || inner <= 0 || n <= 0) return;
    const int64_t total = outer * inner;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            const int64_t o = idx / inner;
            const int64_t i = idx % inner;
            const float * p = x + o * n * inner + i;
            float best = p[0];
            int64_t bi = 0;
            // The host's tie-break: the first maximum wins (`v > best`).
            for (int64_t k = 1; k < n; ++k) {
                const float v = p[k * inner];
                if (v > best) {
                    best = v;
                    bi = k;
                }
            }
            y[idx] = bi;
        });
    });
}

// One work-group per row, each lane walking a strided slice: the recogniser's
// probability tensor is read with coalesced 32-bit accesses and the whole
// row is reduced in shared memory.  The combine takes the larger value and,
// on a tie, the *lower* index - which is the order numpy's argmax reports
// and what the host decoder assumed.
void argmax_max_launch(sycl::queue & q, const float * x, int64_t * y_idx, float * y_max,
                       int64_t outer, int64_t n) {
    if (outer <= 0 || n <= 0) return;
    constexpr int WG = 256;
    if (n < 2 * WG) {
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>((size_t)outer), [=](sycl::item<1> it) {
                const int64_t row = it.get_id(0);
                const float * p = x + row * n;
                float best = p[0];
                int64_t bi = 0;
                for (int64_t k = 1; k < n; ++k) {
                    const float v = p[k];
                    if (v > best) {
                        best = v;
                        bi = k;
                    }
                }
                y_idx[row] = bi;
                y_max[row] = best;
            });
        });
        return;
    }
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> sv(sycl::range<1>(WG), h);
        sycl::local_accessor<int64_t, 1> si(sycl::range<1>(WG), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)outer * WG),
                                         sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) {
                           const int64_t row = it.get_group(0);
                           const int64_t lid = (int64_t)it.get_local_id(0);
                           const float * p = x + row * n;
                           float best = -3.0e38f;
                           int64_t bi = 0;
                           for (int64_t k = lid; k < n; k += WG) {
                               const float v = p[k];
                               if (v > best) {
                                   best = v;
                                   bi = k;
                               }
                           }
                           sv[lid] = best;
                           si[lid] = bi;
                           it.barrier();
                           for (int s = WG / 2; s > 0; s >>= 1) {
                               if (lid < s) {
                                   const float ov = sv[lid + s];
                                   const int64_t oi = si[lid + s];
                                   if (ov > sv[lid] || (ov == sv[lid] && oi < si[lid])) {
                                       sv[lid] = ov;
                                       si[lid] = oi;
                                   }
                               }
                               it.barrier();
                           }
                           if (lid == 0) {
                               y_idx[row] = si[0];
                               y_max[row] = sv[0];
                           }
                       });
    });
}

// The reduction over an *integer* input.
//
// The layout model's decoder reduces a mask of twelve million int32
// elements, and the host does that in 282 milliseconds - the device does
// it in about a millisecond.  The kernel is the same walk with a
// different element type, and the *output* is an integer too.
// (The implementation is templated on the operator, below: the switch that
// used to sit in the inner loop was the reason 12M elements took 6 ms.)

// The reduction's combine, as a template so the operator is *not* a switch
// inside the inner loop.  The switch was the whole cost: 12M elements took
// 6 ms (8 GB/s) because every iteration was a branch on a uniform value the
// compiler could not hoist, which also kept the accumulator chain serial.
template <ew_op OP, class U>
static inline U red_bin(U a, U b) {
    if constexpr (OP == ew_op::add) return (U)(a + b);
    else if constexpr (OP == ew_op::mul) return (U)(a * b);
    else if constexpr (OP == ew_op::max_) return b > a ? b : a;
    else if constexpr (OP == ew_op::min_) return b < a ? b : a;
    else return (U)(a + b);
}

template <ew_op OP, class T, class U>
static inline U red_init() {
    if constexpr (OP == ew_op::add) return (U)0;
    else if constexpr (OP == ew_op::mul) return (U)1;
    else if constexpr (OP == ew_op::max_) return std::numeric_limits<U>::lowest();
    else if constexpr (OP == ew_op::min_) return std::numeric_limits<U>::max();
    else return (U)0;
}

template <ew_op OP, class T, class U>
static void reduce_launch_op(sycl::queue & q, const T * x, U * y, int64_t outer, int64_t axis,
                             int64_t inner, bool mean) {
    if (axis <= 0 || inner <= 0 || outer <= 0) return;
    const int64_t total = outer * inner;
    const U init = red_init<OP, T, U>();
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            const int64_t o = idx / inner;
            const int64_t i = idx % inner;
            const T * xp = x + o * axis * inner + i;
            // Four independent chains: the reduction is associative, and one
            // dependent load per iteration leaves the memory system idle.
            U a0 = init, a1 = init, a2 = init, a3 = init;
            int64_t k = 0;
            for (; k + 3 < axis; k += 4) {
                a0 = red_bin<OP>(a0, (U)xp[k * inner]);
                a1 = red_bin<OP>(a1, (U)xp[(k + 1) * inner]);
                a2 = red_bin<OP>(a2, (U)xp[(k + 2) * inner]);
                a3 = red_bin<OP>(a3, (U)xp[(k + 3) * inner]);
            }
            for (; k < axis; ++k) a0 = red_bin<OP>(a0, (U)xp[k * inner]);
            U acc = red_bin<OP>(red_bin<OP>(a0, a1), red_bin<OP>(a2, a3));
            if (mean) acc = (U)(acc / (U)axis);
            y[idx] = acc;
        });
    });
}

// A reduction over the *last* axis (inner == 1) with many rows and a long
// axis is the layout's mask head: 300 rows of 40 000 values.  The one
// work-item per row form gives 300 threads and leaves the card idle; a
// work-group per row gives 256 threads each and a tree reduction in SLM.
// (The layout's six 12M-element reductions were 6 ms each at ~8 GB/s.)
template <ew_op OP, class T, class U>
static void reduce_rows_launch(sycl::queue & q, const T * x, U * y, int64_t outer,
                               int64_t axis, bool mean) {
    constexpr int WG = 256;
    const U init = red_init<OP, T, U>();
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<U, 1> part(WG, h);
        h.parallel_for(sycl::nd_range<1>((size_t)outer * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t o = it.get_group(0);
            const int lane = (int)it.get_local_id(0);
            const T * p = x + o * axis;
            U acc = init;
            for (int64_t k = lane; k < axis; k += WG) acc = red_bin<OP>(acc, (U)p[k]);
            part[lane] = acc;
            it.barrier(sycl::access::fence_space::local_space);
            for (int stride = WG / 2; stride > 0; stride /= 2) {
                if (lane < stride) part[lane] = red_bin<OP>(part[lane], part[lane + stride]);
                it.barrier(sycl::access::fence_space::local_space);
            }
            if (lane == 0) y[o] = mean ? (U)(part[0] / (U)axis) : part[0];
        });
    });
}

template <class T, class U>
static void reduce_dispatch(sycl::queue & q, const T * x, U * y, int64_t outer, int64_t axis,
                            int64_t inner, ew_op op, bool mean) {
    // The long last-axis case: one work-group per row.  The layout's mask
    // max is an *int* cast of the mask (12M elements) with 300 rows, and the
    // one-work-item-per-row form left it at 300 threads (6 ms); the
    // work-group form is 256 threads per row.
    if (inner == 1 && axis >= 512 && outer >= 8) {
        switch (op) {
        case ew_op::add: reduce_rows_launch<ew_op::add>(q, x, y, outer, axis, mean); return;
        case ew_op::mul: reduce_rows_launch<ew_op::mul>(q, x, y, outer, axis, mean); return;
        case ew_op::max_: reduce_rows_launch<ew_op::max_>(q, x, y, outer, axis, mean); return;
        case ew_op::min_: reduce_rows_launch<ew_op::min_>(q, x, y, outer, axis, mean); return;
        default: reduce_rows_launch<ew_op::add>(q, x, y, outer, axis, mean); return;
        }
    }
    switch (op) {
    case ew_op::add: reduce_launch_op<ew_op::add>(q, x, y, outer, axis, inner, mean); break;
    case ew_op::mul: reduce_launch_op<ew_op::mul>(q, x, y, outer, axis, inner, mean); break;
    case ew_op::max_: reduce_launch_op<ew_op::max_>(q, x, y, outer, axis, inner, mean); break;
    case ew_op::min_: reduce_launch_op<ew_op::min_>(q, x, y, outer, axis, inner, mean); break;
    default: reduce_launch_op<ew_op::add>(q, x, y, outer, axis, inner, mean); break;
    }
}

// The reduction's dispatch: the element type of the input and of the output.
//
// The integer case is a separate instantiation, not a conversion: the
// layout model's mask is twelve million int32 elements, and converting
// them to float first would cost a second pass over all of them.
void reduce_launch_typed(sycl::queue & q, const void * x, void * y, int64_t outer, int64_t axis,
                        int64_t inner, ew_op op, bool mean, int in_kind, int out_kind) {
    if (in_kind == 2 && out_kind == 2) {
        reduce_dispatch(q, (const int64_t *)x, (int64_t *)y, outer, axis, inner, op, mean);
    } else if (in_kind == 1 && out_kind == 1) {
        reduce_dispatch(q, (const int32_t *)x, (int32_t *)y, outer, axis, inner, op, mean);
    } else if (in_kind == 1 && out_kind == 2) {
        reduce_dispatch(q, (const int32_t *)x, (int64_t *)y, outer, axis, inner, op, mean);
    } else if (in_kind == 2 && out_kind == 1) {
        reduce_dispatch(q, (const int64_t *)x, (int32_t *)y, outer, axis, inner, op, mean);
    } else {
        reduce_dispatch(q, (const float *)x, (float *)y, outer, axis, inner, op, mean);
    }
}

void reduce_launch(sycl::queue & q, const float * x, float * y, int64_t outer, int64_t axis,
                   int64_t inner, ew_op op, bool mean) {
    reduce_dispatch(q, x, y, outer, axis, inner, op, mean);
}

void softmax_launch(sycl::queue & q, const float * x, float * y, size_t rows, int64_t cols) {
    constexpr int WG = 256;
    // A wide row is reduced by a whole work-group: one work-item per row
    // leaves the device idle.  The recogniser's softmax is [1536, 18710] -
    // 1536 work-items is three per EU - and it measured 25.7 ms, three times
    // the conv time it follows.
    if (cols < 2 * WG || rows == 0) {
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(rows ? rows : 1), [=](sycl::item<1> it) {
                if (!rows) return;
                const size_t r = it.get_id(0);
                if (r >= rows) return;
                const float * xp = x + (int64_t)r * cols;
                float * yp = y + (int64_t)r * cols;
                float mx = xp[0];
                for (int64_t i = 1; i < cols; ++i) mx = xp[i] > mx ? xp[i] : mx;
                float sum = 0.0f;
                for (int64_t i = 0; i < cols; ++i) {
                    const float e = std::exp(xp[i] - mx);
                    yp[i] = e;
                    sum += e;
                }
                const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int64_t i = 0; i < cols; ++i) yp[i] *= inv;
            });
        });
        return;
    }
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(WG), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) {
                           const int64_t r = it.get_group(0);
                           const int64_t lane = it.get_local_id(0);
                           const float * xp = x + r * cols;
                           float * yp = y + r * cols;
                           // The maximum, in a private accumulator and then
                           // a shared tree.
                           float mx = -3.0e38f;
                           for (int64_t i = lane; i < cols; i += WG) {
                               const float v = xp[i];
                               if (v > mx) mx = v;
                           }
                           red[lane] = mx;
                           it.barrier();
                           for (int s = WG / 2; s > 0; s >>= 1) {
                               if (lane < s && red[lane + s] > red[lane]) red[lane] = red[lane + s];
                               it.barrier();
                           }
                           mx = red[0];
                           it.barrier();
                           // exp and the sum, each lane over its own slice.
                           float sum = 0.0f;
                           for (int64_t i = lane; i < cols; i += WG) {
                               const float e = sycl::exp(xp[i] - mx);
                               yp[i] = e;
                               sum += e;
                           }
                           red[lane] = sum;
                           it.barrier();
                           for (int s = WG / 2; s > 0; s >>= 1) {
                               if (lane < s) red[lane] += red[lane + s];
                               it.barrier();
                           }
                           const float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;
                           for (int64_t i = lane; i < cols; i += WG) yp[i] *= inv;
                       });
    });
}

void layer_norm_launch(sycl::queue & q, const float * x, const float * gamma, const float * beta,
                       float * y, size_t rows, int64_t cols, float eps) {
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(rows), [=](sycl::item<1> it) {
            const size_t r = it.get_id(0);
            const float * xp = x + (int64_t)r * cols;
            float * yp = y + (int64_t)r * cols;
            float sum = 0.0f;
            for (int64_t i = 0; i < cols; ++i) sum += xp[i];
            const float mean = sum / (float)cols;
            float var = 0.0f;
            for (int64_t i = 0; i < cols; ++i) {
                const float d = xp[i] - mean;
                var += d * d;
            }
            var /= (float)cols;
            const float inv = 1.0f / std::sqrt(var + eps);
            for (int64_t i = 0; i < cols; ++i) {
                float v = (xp[i] - mean) * inv;
                if (gamma) v *= gamma[i];
                if (beta) v += beta[i];
                yp[i] = v;
            }
        });
    });
}

void gather_nd_launch(sycl::queue & q, const float * data, const void * indices, float * out,
                      int64_t n_out, int64_t inner, int64_t r, bool indices_are_32,
                      const int64_t * data_stride, const int64_t * data_dim, int n_stride) {
    // The data's strides and dims travel *by value*: a host pointer read
    // from a device kernel reads freed stack.
    constexpr int kMax = 8;
    int64_t ds[kMax] = {0};
    int64_t dm[kMax] = {0};
    const int n = n_stride > kMax ? kMax : n_stride;
    for (int i = 0; i < n; ++i) {
        ds[i] = data_stride[i];
        dm[i] = data_dim[i];
    }
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n_out), [=](sycl::item<1> it) {
            const int64_t o = it.get_id(0);
            // The index tuple and the trailing offset.  The trailing block is
            // *contiguous* in the data, so it is one addition and not a
            // coordinate decomposition.
            const int64_t b = o / inner;
            const int64_t t = o - b * inner;
            int64_t off = t;
            bool bad = false;
            for (int64_t c = 0; c < r; ++c) {
                int64_t v = indices_are_32 ? (int64_t)((const int32_t *)indices)[b * r + c]
                                           : ((const int64_t *)indices)[b * r + c];
                if (v < 0) v += dm[c];
                if (v < 0 || v >= dm[c]) {
                    bad = true;
                    break;
                }
                off += v * ds[c];
            }
            out[o] = bad ? 0.0f : data[off];
        });
    });
}

void matmul_launch(sycl::queue & q, const float * a, const float * b, float * c, int64_t batch,
                   int64_t M, int64_t K, int64_t N, bool transpose_b, const float * bias) {
    const int64_t total = batch * M * N;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            const int64_t n = idx % N;
            const int64_t m = (idx / N) % M;
            const int64_t bi = idx / (N * M);
            const float * ap = a + bi * M * K;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += ap[m * K + k] * (transpose_b ? b[bi * K * N + n * K + k] : b[bi * K * N + k * N + n]);
            }
            // The bias is per output column.
            if (bias) acc += bias[n];
            c[idx] = acc;
        });
    });
}

void matmul_bcast_launch(sycl::queue & q, const float * a, const float * b, float * c,
                         const int64_t * out_batch, int rank, int64_t M, int64_t K, int64_t N,
                         const int64_t * a_stride, const int64_t * b_stride, int64_t a_batch,
                         int64_t b_batch) {
    // The leading dims are broadcast: an input's leading dim of 1 is read
    // again (a stride of 0).
    //
    // The arrays travel *by value*: a host pointer read from a device kernel
    // reads freed stack (the kernel runs after the launcher has returned).
    constexpr int kMaxRank = 8;
    int64_t ob[kMaxRank] = {0};
    int64_t as_[kMaxRank] = {0};
    int64_t bs_[kMaxRank] = {0};
    const int r = rank > kMaxRank ? kMaxRank : rank;
    int64_t batch = 1;
    for (int d = 0; d < r; ++d) {
        ob[d] = out_batch[d];
        as_[d] = a_stride[d];
        bs_[d] = b_stride[d];
        batch *= ob[d];
    }
    const int64_t total = batch * M * N;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            const int64_t n = idx % N;
            const int64_t m = (idx / N) % M;
            const int64_t bi = idx / (N * M);
            // The batch index, in the leading dims' coordinates.
            int64_t coord[kMaxRank] = {0};
            int64_t rest = bi;
            for (int d = r - 1; d >= 0; --d) {
                coord[d] = rest % ob[d];
                rest /= ob[d];
            }
            int64_t ao = 0, bo = 0;
            for (int d = 0; d < r; ++d) {
                ao += coord[d] * as_[d];
                bo += coord[d] * bs_[d];
            }
            const float * ap = a + ao * M * K;
            const float * bp = b + bo * K * N;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += ap[m * K + k] * bp[k * N + n];
            }
            c[idx] = acc;
        });
    });
}

void batchnorm_launch(sycl::queue & q, const float * x, const float * scale, const float * b,
                      const float * mean, const float * var, float * y, int N, int C, int64_t HW,
                      float eps) {
    const int64_t total = (int64_t)N * C * HW;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            const int64_t c = (idx / HW) % C;
            const float inv = 1.0f / std::sqrt(var[c] + eps);
            float v = (x[idx] - mean[c]) * inv;
            if (scale) v *= scale[c];
            if (b) v += b[c];
            y[idx] = v;
        });
    });
}

void concat_launch(sycl::queue & q, const std::vector<const float *> & in,
                   const std::vector<std::vector<int64_t>> & in_shapes, float * y, int axis,
                   int64_t outer) {
    // The output is the inputs concatenated along `axis`; the other axes are
    // broadcast (an input may have 1 there).
    //
    // The copy is a gather over the *output's* index space: for every output
    // element, the source is the input that owns the element's axis index,
    // at the input's own (broadcast) coordinates.
    //
    // Everything the kernel needs is in fixed-size arrays: a std::vector is
    // not device copyable.
    constexpr int kMaxRank = 8;
    constexpr int kMaxIn = 16;
    const int64_t rank = (int64_t)in_shapes[0].size();
    const int n = rank > kMaxRank ? kMaxRank : (int)rank;
    const int nin = (int)std::min<size_t>(in.size(), kMaxIn);

    int64_t out_shape[kMaxRank] = {0};
    for (int d = 0; d < n; ++d) out_shape[d] = in_shapes[0][d];
    int64_t out_axis = 0;
    for (int i = 0; i < nin; ++i) {
        out_axis += in_shapes[i][axis];
        for (int d = 0; d < n; ++d) {
            if (d == axis) continue;
            if (in_shapes[i][d] != 1 && out_shape[d] == 1) out_shape[d] = in_shapes[i][d];
        }
    }
    out_shape[axis] = out_axis;
    int64_t total = 1;
    for (int d = 0; d < n; ++d) total *= out_shape[d];
    if (total == 0) return;

    int64_t ostride[kMaxRank] = {0};
    ostride[n - 1] = 1;
    for (int d = n - 2; d >= 0; --d) ostride[d] = ostride[d + 1] * out_shape[d + 1];

    int64_t offs[kMaxIn] = {0};
    int64_t asz[kMaxIn] = {0};
    int64_t ishape[kMaxIn][kMaxRank] = {{0}};
    const float * ptrs[kMaxIn] = {nullptr};
    {
        int64_t acc = 0;
        for (int i = 0; i < nin; ++i) {
            offs[i] = acc;
            asz[i] = in_shapes[i][axis];
            acc += asz[i];
            ptrs[i] = in[i];
            for (int d = 0; d < n; ++d) ishape[i][d] = in_shapes[i][d];
        }
    }

    // The copy runs over each *input's* own elements, not over the output:
    // the input is contiguous, so its element index is its source offset,
    // and only the destination offset needs the coordinates.  The old version
    // launched one kernel per input over the whole output and made every
    // element work out which input owned it - nin passes over the output and
    // a full coordinate decomposition per element, which is what made 140
    // concats 42 ms of the layout model.
    for (int k = 0; k < nin; ++k) {
        const float * src_ptr = ptrs[k];
        const int64_t base = offs[k] * ostride[axis];
        int64_t count = 1;
        for (int d = 0; d < n; ++d) count *= ishape[k][d];
        if (count == 0) continue;
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(count), [=](sycl::item<1> it) {
                const int64_t idx = it.get_id(0);
                int64_t rest = idx;
                int64_t dp = base;
                // The input's own coordinates; a size-1 dim broadcasts (its
                // coordinate is 0, so it adds nothing).
                for (int d = n - 1; d >= 0; --d) {
                    const int64_t dim = ishape[k][d];
                    const int64_t c = dim > 0 ? rest % dim : 0;
                    rest /= dim > 0 ? dim : 1;
                    dp += c * ostride[d];
                }
                y[dp] = src_ptr[idx];
            });
        });
    }
}

void strided_copy_launch(sycl::queue & q, const float * x, float * y, const int64_t * src_stride,
                         const int64_t * dst_stride, const int64_t * out_shape, int rank,
                         int64_t total) {
    if (total == 0) return;
    // The strides and the shape are copied out: the kernel may not capture
    // a host pointer.
    int64_t ss[8] = {0}, ds[8] = {0}, os[8] = {0};
    const int n = rank > 8 ? 8 : rank;
    for (int d = 0; d < n; ++d) {
        ss[d] = src_stride[d];
        ds[d] = dst_stride[d];
        os[d] = out_shape[d];
    }
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            int64_t rest = idx;
            int64_t src = 0;
            int64_t dst = 0;
            for (int d = n - 1; d >= 0; --d) {
                const int64_t dim = rest % os[d];
                rest /= os[d];
                src += dim * ss[d];
                dst += dim * ds[d];
            }
            y[dst] = x[src];
        });
    });
}

void gather_launch(sycl::queue & q, const float * x, const int64_t * idx, float * y, int64_t outer,
                   int64_t axis_dim, int64_t inner, int64_t n_idx) {
    const int64_t total = outer * n_idx * inner;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx_pos = it.get_id(0);
            const int64_t i = idx_pos % inner;
            const int64_t k = (idx_pos / inner) % n_idx;
            const int64_t o = idx_pos / (inner * n_idx);
            const int64_t pick = idx[k];
            y[idx_pos] = x[o * axis_dim * inner + pick * inner + i];
        });
    });
}

// The order key: a larger key is a better candidate for both directions.
// The float bits are made monotonic (the sign flip), the direction is
// inverted for the smallest-k case, and the low 32 bits are the *inverted*
// column index so that an equal value prefers the lower index.  The index
// makes every key unique, which is what lets the radix select stop at an
// exact k-th key instead of dealing with ties separately.
static inline uint64_t topk_key(float v, int64_t idx, bool largest) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = v;
    uint32_t k = (bits.u & 0x80000000u) ? ~bits.u : (bits.u | 0x80000000u);
    if (!largest) k = ~k;
    return ((uint64_t)k << 32) | (uint32_t)(0xffffffffu - (uint32_t)idx);
}

// Sort k existing (value, index) pairs best-first.  The heap's root is the
// worst pair (smallest value for largest, largest for smallest; on a tie the
// higher index), and moving it to the shrinking end leaves the array
// best-first.
static inline void topk_sort_in_place(int k, bool largest, float * hv, int64_t * hi) {
    if (k <= 0) return;
    auto worse = [=](float av, int64_t ai, float bv, int64_t bi) {
        if (largest) return av < bv || (av == bv && ai > bi);
        return av > bv || (av == bv && ai > bi);
    };
    auto sift = [&](int start, int n) {
        int i = start;
        while (true) {
            const int l = 2 * i + 1;
            if (l >= n) break;
            const int rr = l + 1;
            int w = l;
            if (rr < n && worse(hv[rr], hi[rr], hv[l], hi[l])) w = rr;
            if (!worse(hv[w], hi[w], hv[i], hi[i])) break;
            const float tv = hv[i];
            const int64_t ti = hi[i];
            hv[i] = hv[w];
            hi[i] = hi[w];
            hv[w] = tv;
            hi[w] = ti;
            i = w;
        }
    };
    for (int i = k / 2 - 1; i >= 0; --i) sift(i, k);
    for (int n = k; n > 1;) {
        --n;
        const float tv = hv[0];
        const int64_t ti = hi[0];
        hv[0] = hv[n];
        hi[0] = hi[n];
        hv[n] = tv;
        hi[n] = ti;
        sift(0, n);
    }
}

// The fallback for k >= cols (and for very short rows): one heap per row.
// It writes exactly k entries, with the unused tail marked index -1, which
// is the ONNX TopK output that the old implementation produced.
static inline void topk_heap_row(const float * xp, int64_t begin, int64_t end, int k, bool largest,
                                 float * hv, int64_t * hi) {
    auto worse = [=](float av, int64_t ai, float bv, int64_t bi) {
        if (largest) return av < bv || (av == bv && ai > bi);
        return av > bv || (av == bv && ai > bi);
    };
    auto sift_down = [&](int n) {
        int i = 0;
        while (true) {
            const int l = 2 * i + 1;
            if (l >= n) break;
            const int rr = l + 1;
            int w = l;
            if (rr < n && worse(hv[rr], hi[rr], hv[l], hi[l])) w = rr;
            if (!worse(hv[w], hi[w], hv[i], hi[i])) break;
            const float tv = hv[i];
            const int64_t ti = hi[i];
            hv[i] = hv[w];
            hi[i] = hi[w];
            hv[w] = tv;
            hi[w] = ti;
            i = w;
        }
    };
    int m = 0;
    for (int64_t c = begin; c < end; ++c) {
        const float v = xp[c];
        if (m < k) {
            int i = m++;
            hv[i] = v;
            hi[i] = c;
            while (i > 0) {
                const int p = (i - 1) / 2;
                if (!worse(hv[i], hi[i], hv[p], hi[p])) break;
                const float tv = hv[i];
                const int64_t ti = hi[i];
                hv[i] = hv[p];
                hi[i] = hi[p];
                hv[p] = tv;
                hi[p] = ti;
                i = p;
            }
        } else if (worse(hv[0], hi[0], v, c)) {
            hv[0] = v;
            hi[0] = c;
            sift_down(m);
        }
    }
    topk_sort_in_place(m, largest, hv, hi);
    for (int j = m; j < k; ++j) {
        hv[j] = 0.0f;
        hi[j] = -1;
    }
}

void topk_launch(sycl::queue & q, const float * x, float * values, int64_t * indices, size_t rows,
                   int64_t cols, int k, bool largest) {
    if (k <= 0 || rows == 0 || cols <= 0) return;
    if (k >= cols || cols < 256) {
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(rows), [=](sycl::item<1> it) {
                const size_t r = it.get_id(0);
                topk_heap_row(x + (int64_t)r * cols, 0, cols, k, largest,
                              values + (int64_t)r * k, indices + (int64_t)r * k);
            });
        });
        return;
    }

    // One work-group per row, one radix pass per key byte.  The 64-bit key is
    // (monotonic float, inverted index), so a pass bins a byte and the
    // work-group keeps the bucket that contains the k-th key.  Eight passes
    // read the row eight times (104k values for 13 125 columns), but the
    // reads are coalesced across the 256 lanes and the work-group is the
    // unit of parallelism: the old implementation used one thread per row
    // and random global shifts.
    constexpr int WG = 256;
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<uint32_t, 1> hist(256, h);
        sycl::local_accessor<uint64_t, 1> lprefix(1, h);
        sycl::local_accessor<int, 1> lrem(1, h);
        sycl::local_accessor<int, 1> lpos(1, h);
        h.parallel_for(sycl::nd_range<1>((size_t)rows * WG, WG), [=](sycl::nd_item<1> it) {
            const size_t r = it.get_group(0);
            const int lane = (int)it.get_local_id(0);
            const float * xp = x + (int64_t)r * cols;
            if (lane == 0) {
                lprefix[0] = 0;
                lrem[0] = k;
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int shift = 56; shift >= 0; shift -= 8) {
                for (int i = lane; i < 256; i += WG) hist[i] = 0;
                it.barrier(sycl::access::fence_space::local_space);
                const uint64_t mask = shift == 56 ? 0ULL : (~0ULL << (shift + 8));
                const uint64_t pref = lprefix[0];
                for (int64_t c = lane; c < cols; c += WG) {
                    const uint64_t key = topk_key(xp[c], c, largest);
                    if ((key & mask) != pref) continue;
                    const uint32_t b = (uint32_t)((key >> shift) & 0xff);
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        a(hist[b]);
                    a.fetch_add(1);
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (lane == 0) {
                    const int rem = lrem[0];
                    int acc = 0, b = -1;
                    for (int xb = 255; xb >= 0; --xb) {
                        acc += (int)hist[xb];
                        if (acc >= rem) {
                            b = xb;
                            break;
                        }
                    }
                    if (b >= 0) {
                        lprefix[0] |= (uint64_t)(uint32_t)b << shift;
                        lrem[0] = rem - (acc - (int)hist[b]);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            const uint64_t cutoff = lprefix[0];
            if (lane == 0) lpos[0] = 0;
            it.barrier(sycl::access::fence_space::local_space);
            for (int64_t c = lane; c < cols; c += WG) {
                const float v = xp[c];
                const uint64_t key = topk_key(v, c, largest);
                if (key < cutoff) continue;
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>
                    a(lpos[0]);
                const int slot = a.fetch_add(1);
                if (slot < k) {
                    values[(int64_t)r * k + slot] = v;
                    indices[(int64_t)r * k + slot] = c;
                }
            }
        });
    });
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(rows), [=](sycl::item<1> it) {
            const size_t r = it.get_id(0);
            topk_sort_in_place(k, largest, values + (int64_t)r * k, indices + (int64_t)r * k);
        });
    });
}

void grid_sample_launch(sycl::queue & q, const float * x, const float * grid, float * y, int N,
                        int C, int H, int W, int Ho, int Wo, int mode, int padding) {
    const int64_t total = (int64_t)N * Ho * Wo;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            const int wo = (int)(idx % Wo);
            const int64_t t1 = idx / Wo;
            const int ho = (int)(t1 % Ho);
            const int n = (int)(t1 / Ho);
            const float * gp = grid + ((int64_t)n * Ho + ho) * Wo * 2 + wo * 2;
            const float gx = gp[0], gy = gp[1];
            const float * xp = x + (int64_t)n * C * H * W;
            for (int c = 0; c < C; ++c) {
                const float * xc = xp + (int64_t)c * H * W;
                // Bilinear, align_corners = false, "border" padding.
                const float fx = ((gx + 1.0f) * W - 1.0f) * 0.5f;
                const float fy = ((gy + 1.0f) * H - 1.0f) * 0.5f;
                const int x0 = (int)std::floor(fx);
                const int y0 = (int)std::floor(fy);
                const float dx = fx - (float)x0;
                const float dy = fy - (float)y0;
                // The padding: zeros (the model asks for it), border
                // (the edge is repeated) or reflection.
                auto at = [&](int yy, int xx) -> float {
                    int cy = yy, cx = xx;
                    if (padding == 0) {
                        if (cy < 0 || cy >= H || cx < 0 || cx >= W) return 0.0f;
                    } else if (padding == 1) {
                        cy = cy < 0 ? 0 : (cy >= H ? H - 1 : cy);
                        cx = cx < 0 ? 0 : (cx >= W ? W - 1 : cx);
                    } else {
                        while (cy < 0 || cy >= H) cy = cy < 0 ? -cy : 2 * H - 2 - cy;
                        while (cx < 0 || cx >= W) cx = cx < 0 ? -cx : 2 * W - 2 - cx;
                    }
                    return xc[(int64_t)cy * W + cx];
                };
                float v = at(y0, x0) * (1 - dy) * (1 - dx) + at(y0, x0 + 1) * (1 - dy) * dx +
                          at(y0 + 1, x0) * dy * (1 - dx) + at(y0 + 1, x0 + 1) * dy * dx;
                y[((int64_t)n * C + c) * Ho * Wo + (int64_t)ho * Wo + wo] = v;
            }
        });
    });
}

} // namespace onnx
} // namespace si

namespace si {
namespace onnx {

// The ScatterElements: the destination of (l, p, t) is (l*dim + p)*T + t,
// where p is the axis position.  The indices give p for every (l, t).
namespace {

template <typename VT>
void scatter_elements_typed(sycl::queue & q, const void * idx, int idx_kind, int64_t n_idx,
                            const void * upd, int64_t n_upd, void * out, int64_t L, int64_t T,
                            int64_t dim, bool slice_form) {
    if (L <= 0 || T <= 0 || dim <= 0) return;
    const int64_t total = L * dim * T;
    if (slice_form) {
        // One work-item per update element (the updates have the data's
        // shape).  (l, cc, t) writes at axis position id + cc, which is the
        // host loop's order; a destination past the end is dropped, exactly
        // as the host code's bounds check does.
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>((size_t)total), [=](sycl::item<1> it) {
                const int64_t i = (int64_t)it.get_id(0);
                const int64_t l = i / (dim * T);
                const int64_t rem = i - l * (dim * T);
                const int64_t cc = rem / T;
                const int64_t t = rem - cc * T;
                const int64_t k = l * T + t;
                if (k >= n_upd) return;
                const int64_t id0 = k >= n_idx ? 0
                                    : idx_kind == kScatterIdxI32
                                        ? (int64_t)((const int32_t *)idx)[k]
                                        : ((const int64_t *)idx)[k];
                const int64_t id = id0 < 0 ? id0 + dim : id0;
                const int64_t dst = (l * dim + (id + cc)) * T + t;
                if (dst >= 0 && dst < total) ((VT *)out)[dst] = ((const VT *)upd)[i];
            });
        });
        return;
    }
    // One work-item per index (one value per (l, t)).
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>((size_t)(L * T)), [=](sycl::item<1> it) {
            const int64_t i = (int64_t)it.get_id(0);
            const int64_t l = i / T;
            const int64_t t = i - l * T;
            // The host loop reads one index per (l, t) and one update per
            // index; a short list defaults the index to 0 and drops the
            // update, which are mirrored here.
            if (i >= n_upd) return;
            const int64_t id0 = i >= n_idx ? 0
                                : idx_kind == kScatterIdxI32
                                    ? (int64_t)((const int32_t *)idx)[i]
                                    : ((const int64_t *)idx)[i];
            const int64_t id = id0 < 0 ? id0 + dim : id0;
            if (id < 0 || id >= dim) return;
            ((VT *)out)[(l * dim + id) * T + t] = ((const VT *)upd)[i];
        });
    });
}

} // namespace

void scatter_elements_launch(sycl::queue & q, const void * data, const void * idx, int idx_kind,
                             int64_t n_idx, const void * upd, int64_t n_upd, void * out, int64_t L,
                             int64_t T, int64_t dim, bool slice_form, int kind) {
    // The copy of the data is the operator's other half; the caller issues it
    // (q_copy_dev) so this stays one kernel.
    (void)data;
    if (kind == kScatterI32) {
        scatter_elements_typed<int32_t>(q, idx, idx_kind, n_idx, upd, n_upd, out, L, T, dim,
                                        slice_form);
    } else if (kind == kScatterI64) {
        scatter_elements_typed<int64_t>(q, idx, idx_kind, n_idx, upd, n_upd, out, L, T, dim,
                                        slice_form);
    } else {
        scatter_elements_typed<float>(q, idx, idx_kind, n_idx, upd, n_upd, out, L, T, dim,
                                      slice_form);
    }
}

// ------------------------------------------------- the integer tensors' moves
//
// An integer value that stays on the device (the layout model's TopK indices
// and the index arithmetic after them) is moved by these, in its own width.
// They are the same walks as the float kernels above: a 4-byte arm covers
// int32 as well as float (the copy is a bit copy), and an 8-byte arm covers
// int64.

namespace {

template <typename T>
void concat_typed_impl(sycl::queue & q, const std::vector<const void *> & in,
                       const std::vector<std::vector<int64_t>> & in_shapes, void * y, int axis,
                       int64_t outer) {
    (void)outer;
    constexpr int kMaxRank = 8;
    constexpr int kMaxIn = 16;
    const int64_t rank = (int64_t)in_shapes[0].size();
    const int n = rank > kMaxRank ? kMaxRank : (int)rank;
    const int nin = (int)std::min<size_t>(in.size(), kMaxIn);
    int64_t out_shape[kMaxRank] = {0};
    for (int d = 0; d < n; ++d) out_shape[d] = in_shapes[0][d];
    int64_t out_axis = 0;
    for (int i = 0; i < nin; ++i) {
        out_axis += in_shapes[i][axis];
        for (int d = 0; d < n; ++d) {
            if (d == axis) continue;
            if (in_shapes[i][d] != 1 && out_shape[d] == 1) out_shape[d] = in_shapes[i][d];
        }
    }
    out_shape[axis] = out_axis;
    int64_t total = 1;
    for (int d = 0; d < n; ++d) total *= out_shape[d];
    if (total == 0) return;
    int64_t ostride[kMaxRank] = {0};
    ostride[n - 1] = 1;
    for (int d = n - 2; d >= 0; --d) ostride[d] = ostride[d + 1] * out_shape[d + 1];
    int64_t offs[kMaxIn] = {0};
    int64_t ishape[kMaxIn][kMaxRank] = {{0}};
    const T * ptrs[kMaxIn] = {nullptr};
    int64_t acc = 0;
    for (int i = 0; i < nin; ++i) {
        offs[i] = acc;
        acc += in_shapes[i][axis];
        ptrs[i] = (const T *)in[i];
        for (int d = 0; d < n; ++d) ishape[i][d] = in_shapes[i][d];
    }
    for (int k = 0; k < nin; ++k) {
        const T * src_ptr = ptrs[k];
        const int64_t base = offs[k] * ostride[axis];
        int64_t count = 1;
        for (int d = 0; d < n; ++d) count *= ishape[k][d];
        if (count == 0) continue;
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(count), [=](sycl::item<1> it) {
                const int64_t idx = it.get_id(0);
                int64_t rest = idx;
                int64_t dp = base;
                for (int d = n - 1; d >= 0; --d) {
                    const int64_t dim = ishape[k][d];
                    const int64_t c = dim > 0 ? rest % dim : 0;
                    rest /= dim > 0 ? dim : 1;
                    dp += c * ostride[d];
                }
                ((T *)y)[dp] = src_ptr[idx];
            });
        });
    }
}

template <typename T>
void strided_copy_typed_impl(sycl::queue & q, const void * xv, void * yv,
                             const int64_t * src_stride, const int64_t * dst_stride,
                             const int64_t * out_shape, int rank, int64_t total) {
    if (total == 0) return;
    int64_t ss[8] = {0}, ds[8] = {0}, os[8] = {0};
    const int n = rank > 8 ? 8 : rank;
    for (int d = 0; d < n; ++d) {
        ss[d] = src_stride[d];
        ds[d] = dst_stride[d];
        os[d] = out_shape[d];
    }
    const T * x = (const T *)xv;
    T * y = (T *)yv;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            int64_t rest = idx, src = 0, dst = 0;
            for (int d = n - 1; d >= 0; --d) {
                const int64_t dim = rest % os[d];
                rest /= os[d];
                src += dim * ss[d];
                dst += dim * ds[d];
            }
            y[dst] = x[src];
        });
    });
}

template <typename T>
void gather_nd_typed_impl(sycl::queue & q, const void * data, const void * indices, void * out,
                          int64_t n_out, int64_t inner, int64_t r, bool indices_are_32,
                          const int64_t * data_stride, const int64_t * data_dim, int n_stride) {
    constexpr int kMax = 8;
    int64_t ds[kMax] = {0};
    int64_t dm[kMax] = {0};
    const int n = n_stride > kMax ? kMax : n_stride;
    for (int i = 0; i < n; ++i) {
        ds[i] = data_stride[i];
        dm[i] = data_dim[i];
    }
    const T * dp = (const T *)data;
    T * y = (T *)out;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n_out), [=](sycl::item<1> it) {
            const int64_t o = it.get_id(0);
            const int64_t b = o / inner;
            const int64_t t = o - b * inner;
            int64_t off = t;
            bool bad = false;
            for (int64_t c = 0; c < r; ++c) {
                int64_t v = indices_are_32 ? (int64_t)((const int32_t *)indices)[b * r + c]
                                           : ((const int64_t *)indices)[b * r + c];
                if (v < 0) v += dm[c];
                if (v < 0 || v >= dm[c]) {
                    bad = true;
                    break;
                }
                off += v * ds[c];
            }
            y[o] = bad ? (T)0 : dp[off];
        });
    });
}

template <typename T>
void scatter_nd_typed_impl(sycl::queue & q, const void * data, const void * indices,
                           const void * upd, void * out, int64_t n_data, int64_t n_upd,
                           int64_t inner, int64_t r, bool indices_are_32,
                           const int64_t * data_stride, const int64_t * data_dim, int n_stride) {
    constexpr int kMax = 8;
    int64_t ds[kMax] = {0};
    int64_t dm[kMax] = {0};
    const int n = n_stride > kMax ? kMax : n_stride;
    for (int i = 0; i < n; ++i) {
        ds[i] = data_stride[i];
        dm[i] = data_dim[i];
    }
    const T * up = (const T *)upd;
    T * y = (T *)out;
    // The copy of the data is the operator's other half; the caller issues it
    // (q_copy_dev) so this stays one kernel, exactly as the ScatterElements.
    (void)data;
    (void)n_data;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(std::max<int64_t>(n_upd, 1)), [=](sycl::item<1> it) {
            const int64_t o = it.get_id(0);
            if (o >= n_upd) return;
            const int64_t b = o / inner;
            const int64_t t = o - b * inner;
            int64_t off = t;
            for (int64_t c = 0; c < r; ++c) {
                int64_t v = indices_are_32 ? (int64_t)((const int32_t *)indices)[b * r + c]
                                           : ((const int64_t *)indices)[b * r + c];
                if (v < 0) v += dm[c];
                if (v < 0 || v >= dm[c]) return;
                off += v * ds[c];
            }
            y[off] = up[o];
        });
    });
}

template <typename T>
void tile_typed_impl(sycl::queue & q, const void * xv, void * yv, const int64_t * src_stride,
                     const int64_t * dst_stride, const int64_t * out_shape,
                     const int64_t * in_shape, int rank, int64_t total) {
    int64_t ss[8] = {0}, ds[8] = {0}, os[8] = {0}, xs[8] = {0};
    const int n = rank > 8 ? 8 : rank;
    for (int d = 0; d < n; ++d) {
        ss[d] = src_stride[d];
        ds[d] = dst_stride[d];
        os[d] = out_shape[d];
        xs[d] = in_shape[d];
    }
    const T * x = (const T *)xv;
    T * y = (T *)yv;
    q.submit([&](sycl::handler & h) {
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
            y[dp] = x[sp];
        });
    });
}

template <typename T>
void expand_typed_impl(sycl::queue & q, const void * xv, void * yv, const int64_t * strides,
                       const int64_t * out_shape, int rank, int64_t total) {
    int64_t st[8] = {0}, os[8] = {0};
    const int n = rank > 8 ? 8 : rank;
    for (int d = 0; d < n; ++d) {
        st[d] = strides[d];
        os[d] = out_shape[d] > 0 ? out_shape[d] : 1;
    }
    const T * x = (const T *)xv;
    T * y = (T *)yv;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            int64_t rest = idx, sp = 0;
            for (int d = n - 1; d >= 0; --d) {
                const int64_t co = rest % os[d];
                rest /= os[d];
                sp += co * st[d];
            }
            y[idx] = x[sp];
        });
    });
}

} // namespace

void concat_typed_launch(sycl::queue & q, const std::vector<const void *> & in,
                         const std::vector<std::vector<int64_t>> & in_shapes, void * y, int axis,
                         int64_t outer, int elem_size) {
    if (in.empty() || in_shapes.empty()) return;
    if (elem_size == 8) {
        concat_typed_impl<int64_t>(q, in, in_shapes, y, axis, outer);
    } else if (elem_size == 4) {
        concat_typed_impl<int32_t>(q, in, in_shapes, y, axis, outer);
    } else {
        throw std::runtime_error("the concat's element size is not 4 or 8");
    }
}

void strided_copy_typed_launch(sycl::queue & q, const void * x, void * y,
                               const int64_t * src_stride, const int64_t * dst_stride,
                               const int64_t * out_shape, int rank, int64_t total, int elem_size) {
    if (elem_size == 8) {
        strided_copy_typed_impl<int64_t>(q, x, y, src_stride, dst_stride, out_shape, rank, total);
    } else if (elem_size == 4) {
        strided_copy_typed_impl<int32_t>(q, x, y, src_stride, dst_stride, out_shape, rank, total);
    } else {
        throw std::runtime_error("the strided copy's element size is not 4 or 8");
    }
}

void gather_nd_typed_launch(sycl::queue & q, const void * data, const void * indices, void * out,
                            int64_t n_out, int64_t inner, int64_t r, bool indices_are_32,
                            const int64_t * data_stride, const int64_t * data_dim, int n_stride,
                            int elem_size) {
    if (elem_size == 8) {
        gather_nd_typed_impl<int64_t>(q, data, indices, out, n_out, inner, r, indices_are_32,
                                      data_stride, data_dim, n_stride);
    } else if (elem_size == 4) {
        gather_nd_typed_impl<int32_t>(q, data, indices, out, n_out, inner, r, indices_are_32,
                                      data_stride, data_dim, n_stride);
    } else {
        throw std::runtime_error("the gather's element size is not 4 or 8");
    }
}

void scatter_nd_typed_launch(sycl::queue & q, const void * data, const void * indices,
                             const void * upd, void * out, int64_t n_data, int64_t n_upd,
                             int64_t inner, int64_t r, bool indices_are_32,
                             const int64_t * data_stride, const int64_t * data_dim, int n_stride,
                             int elem_size) {
    if (elem_size == 8) {
        scatter_nd_typed_impl<int64_t>(q, data, indices, upd, out, n_data, n_upd, inner, r,
                                       indices_are_32, data_stride, data_dim, n_stride);
    } else if (elem_size == 4) {
        scatter_nd_typed_impl<int32_t>(q, data, indices, upd, out, n_data, n_upd, inner, r,
                                       indices_are_32, data_stride, data_dim, n_stride);
    } else {
        throw std::runtime_error("the scatter's element size is not 4 or 8");
    }
}

void tile_typed_launch(sycl::queue & q, const void * x, void * y, const int64_t * src_stride,
                       const int64_t * dst_stride, const int64_t * out_shape,
                       const int64_t * in_shape, int rank, int64_t total, int elem_size) {
    if (elem_size == 8) {
        tile_typed_impl<int64_t>(q, x, y, src_stride, dst_stride, out_shape, in_shape, rank, total);
    } else if (elem_size == 4) {
        tile_typed_impl<int32_t>(q, x, y, src_stride, dst_stride, out_shape, in_shape, rank, total);
    } else {
        throw std::runtime_error("the tile's element size is not 4 or 8");
    }
}

void expand_typed_launch(sycl::queue & q, const void * x, void * y, const int64_t * strides,
                         const int64_t * out_shape, int rank, int64_t total, int elem_size) {
    if (elem_size == 8) {
        expand_typed_impl<int64_t>(q, x, y, strides, out_shape, rank, total);
    } else if (elem_size == 4) {
        expand_typed_impl<int32_t>(q, x, y, strides, out_shape, rank, total);
    } else {
        throw std::runtime_error("the expand's element size is not 4 or 8");
    }
}

void ew_i64_launch(sycl::queue & q, const int64_t * a, const int64_t * b, int64_t * y, size_t n,
                   ew_op op, int scalar_op) {
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) {
            const size_t i = it.get_id(0);
            const int64_t x = a[scalar_op == 0 ? 0 : i];
            const int64_t z = b ? b[scalar_op == 1 ? 0 : i] : 0;
            int64_t r = 0;
            switch (op) {
            case ew_op::mod_: r = z != 0 ? x % z : 0; break;
            case ew_op::max_: r = x > z ? x : z; break;
            case ew_op::min_: r = x < z ? x : z; break;
            case ew_op::add: r = x + z; break;
            case ew_op::sub: r = x - z; break;
            case ew_op::mul: r = x * z; break;
            default: r = x; break;
            }
            y[i] = r;
        });
    });
}

} // namespace onnx
} // namespace si
