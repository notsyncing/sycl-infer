#include "kernels.h"

#include <cmath>
#include <cstring>

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
    case ew_op::sigmoid: return 1.0f / (1.0f + std::exp(-v));
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
            const int blk = (int)(idx % nblocks);
            const int64_t t1 = idx / nblocks;
            const int wo = (int)(t1 % Wo);
            const int64_t t2 = t1 / Wo;
            const int ho = (int)(t2 % Ho);
            const int n = (int)(t2 / Ho);
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

// The copies are *synchronous*.  A queued copy reads its source buffer
// later, and the host buffers here are temporaries (a vector in the
// operator's frame): an asynchronous copy would read them after they are
// destroyed, which is a use-after-free the heap notices much later.
void q_upload(sycl::queue & q, const void * src, void * dst, size_t bytes) {
    if (bytes == 0) return;
    q.memcpy(dst, src, bytes).wait();
}

void q_download(sycl::queue & q, const void * src, void * dst, size_t bytes) {
    if (bytes == 0) return;
    q.memcpy(dst, src, bytes).wait();
}

void q_copy_dev(sycl::queue & q, const void * src, void * dst, size_t bytes) {
    if (bytes == 0) return;
    q.memcpy(dst, src, bytes).wait();
}

// The elementwise kernel.
//
// The device holds float32, so this reads floats.  An integer operand is
// converted before it gets here (the operator does it: an integer value
// lives on the host, and a comparison of two int64 masks has to be an
// integer comparison - reading an integer as a float reads its bits).
void ew_launch(sycl::queue & q, const float * a, const float * b, float * y, size_t n, ew_op op,
               const std::vector<float> & params, bool probe, int scalar_op) {
    if (n == 0) return;
    float p[4] = {-3.0f, 3.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < params.size() && i < 4; ++i) p[i] = params[i];
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) {
            const size_t i = it.get_id(0);
            // The scalar operand is read once, and the other one is read per
            // element.
            const float x = scalar_op == 0 ? (a ? a[0] : 0.0f) : a[i];
            const float z = b ? (scalar_op == 1 ? b[0] : b[i]) : 0.0f;
            ew_functor f{op, {p[0], p[1], p[2], p[3]}};
            y[i] = f(x, z);
            if (probe) y[i] = x;  // the probe: echo the input
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
            if (out_kind == kCastToBool) {
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
                     size_t count, ew_op op, const std::vector<float> & params) {
    if (count == 0) return;
    float p[4] = {-3.0f, 3.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < params.size() && i < 4; ++i) p[i] = params[i];
    // The strides and the shape are copied out: the kernel may not capture
    // a host pointer (and a std::vector is not device copyable).
    int64_t as[8] = {0}, bs[8] = {0}, os[8] = {0};
    const int n = rank > 8 ? 8 : rank;
    for (int d = 0; d < n; ++d) {
        as[d] = a_stride[d];
        bs[d] = b_stride ? b_stride[d] : 0;
        os[d] = out_shape[d];
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
            y[idx] = f(a[ao], b ? b[bo] : 0.0f);
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
                const int ih = (int)((float)ho * H / (float)Ho);
                const int iw = (int)((float)wo * W / (float)Wo);
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

// The reduction over an *integer* input.
//
// The layout model's decoder reduces a mask of twelve million int32
// elements, and the host does that in 282 milliseconds - the device does
// it in about a millisecond.  The kernel is the same walk with a
// different element type, and the *output* is an integer too.
template <typename T, typename U>
void reduce_launch_int(sycl::queue & q, const T * x, U * y, int64_t outer, int64_t axis,
                        int64_t inner, ew_op op, bool mean) {
    const int64_t total = outer * inner;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            const int64_t o = idx / inner;
            const int64_t i = idx % inner;
            const T * xp = x + o * axis * inner + i;
            U acc = (U)xp[0];
            for (int64_t k = 1; k < axis; ++k) {
                const U v = (U)xp[k * inner];
                switch (op) {
                case ew_op::add: acc += v; break;
                case ew_op::mul: acc *= v; break;
                case ew_op::max_: acc = v > acc ? v : acc; break;
                case ew_op::min_: acc = v < acc ? v : acc; break;
                default: acc += v; break;
                }
            }
            if (mean) acc = (U)(acc / (U)axis);
            y[idx] = acc;
        });
    });
}

void reduce_launch(sycl::queue & q, const float * x, float * y, int64_t outer, int64_t axis,
                   int64_t inner, ew_op op, bool mean) {
    // The reduction is over the `axis` elements that follow every `inner`
    // elements, for each of the `outer` rows.
    const int64_t total = outer * inner;
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
            const int64_t idx = it.get_id(0);
            const int64_t o = idx / inner;
            const int64_t i = idx % inner;
            const float * xp = x + o * axis * inner + i;
            float acc = xp[0];
            for (int64_t k = 1; k < axis; ++k) {
                const float v = xp[k * inner];
                switch (op) {
                case ew_op::add: acc += v; break;
                case ew_op::mul: acc *= v; break;
                case ew_op::max_: acc = v > acc ? v : acc; break;
                case ew_op::min_: acc = v < acc ? v : acc; break;
                default: acc += v; break;
                }
            }
            // The mean divides by the count of the reduced elements.  (The
            // kernel used to divide whenever the operator was a sum, which
            // made the *sum* wrong and the mean wrong twice over.)
            if (mean) acc /= (float)axis;
            y[idx] = acc;
        });
    });
}

// The reduction's dispatch: the element type of the input and of the output.
//
// The integer case is a separate instantiation, not a conversion: the
// layout model's mask is twelve million int32 elements, and converting
// them to float first would cost a second pass over all of them.
void reduce_launch_typed(sycl::queue & q, const void * x, void * y, int64_t outer, int64_t axis,
                        int64_t inner, ew_op op, bool mean, int in_kind, int out_kind) {
    if (in_kind == 2 && out_kind == 2) {
        reduce_launch_int<int64_t, int64_t>(q, (const int64_t *)x, (int64_t *)y, outer, axis, inner,
                                            op, mean);
    } else if (in_kind == 1 && out_kind == 1) {
        reduce_launch_int<int32_t, int32_t>(q, (const int32_t *)x, (int32_t *)y, outer, axis, inner,
                                            op, mean);
    } else if (in_kind == 1 && out_kind == 2) {
        reduce_launch_int<int32_t, int64_t>(q, (const int32_t *)x, (int64_t *)y, outer, axis, inner,
                                            op, mean);
    } else if (in_kind == 2 && out_kind == 1) {
        reduce_launch_int<int64_t, int32_t>(q, (const int64_t *)x, (int32_t *)y, outer, axis, inner,
                                            op, mean);
    } else {
        reduce_launch(q, (const float *)x, (float *)y, outer, axis, inner, op, mean);
    }
}

void softmax_launch(sycl::queue & q, const float * x, float * y, size_t rows, int64_t cols) {
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(rows), [=](sycl::item<1> it) {
            const size_t r = it.get_id(0);
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
    int64_t istride[kMaxIn][kMaxRank] = {{0}};
    const float * ptrs[kMaxIn] = {nullptr};
    {
        int64_t acc = 0;
        for (int i = 0; i < nin; ++i) {
            offs[i] = acc;
            asz[i] = in_shapes[i][axis];
            acc += asz[i];
            ptrs[i] = in[i];
            int64_t s = 1;
            for (int d = n - 1; d >= 0; --d) {
                if (d == axis) {
                    istride[i][d] = s;
                } else {
                    istride[i][d] = (in_shapes[i][d] == 1) ? 0 : s;
                }
                s *= in_shapes[i][d];
            }
        }
    }

    for (int k = 0; k < nin; ++k) {
        const float * src_ptr = ptrs[k];
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(total), [=](sycl::item<1> it) {
                const int64_t idx = it.get_id(0);
                // The element's coordinates in the output's index space.
                int64_t coord[kMaxRank] = {0};
                int64_t rest = idx;
                for (int d = n - 1; d >= 0; --d) {
                    coord[d] = rest % out_shape[d];
                    rest /= out_shape[d];
                }
                // The axis: the element belongs to the input that covers it.
                const int64_t a = coord[axis];
                int which = 0;
                for (int i = 0; i < nin; ++i) {
                    if (a < offs[i] + asz[i]) {
                        which = i;
                        break;
                    }
                }
                if (which != k) return;  // another input's element
                // The source uses the input's own axis index, the
                // destination the output's.
                int64_t sp = 0, dp = 0;
                for (int d = 0; d < n; ++d) {
                    const int64_t cd = (d == axis) ? (a - offs[which]) : coord[d];
                    sp += cd * istride[k][d];
                    dp += coord[d] * ostride[d];
                }
                y[dp] = src_ptr[sp];
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

void topk_launch(sycl::queue & q, const float * x, float * values, int64_t * indices, size_t rows,
                   int64_t cols, int k, bool largest, float * scratch_v, int64_t * scratch_i) {
    // One work-item per row, and the row's sorted list in the scratch.
    //
    // The obvious implementation - k passes over the columns, each picking
    // the best of the ones not yet picked - is O(k^2 * cols) because the
    // "is it taken" test is O(k).  The layout model's decoder asks for the
    // top 300 of 13 125 columns, and that is 1.2 billion comparisons per
    // row: 60 seconds for one page.
    //
    // The list is kept sorted instead, and a column is inserted into it: the
    // whole pass is O(cols * k), which is 4 million - a millisecond.
    //
    // The tie-break is the lower index, which is what the reference
    // implementation does: a new element with the same value does not
    // displace the one that is already there.
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(rows), [=](sycl::item<1> it) {
            const size_t r = it.get_id(0);
            const float * xp = x + (int64_t)r * cols;
            float * sv = scratch_v + (int64_t)r * k;
            int64_t * si = scratch_i + (int64_t)r * k;
            int m = 0;
            for (int64_t c = 0; c < cols; ++c) {
                const float v = xp[c];
                if (m == k) {
                    // The list is full: only a better value can get in.
                    const float worst = largest ? sv[k - 1] : sv[0];
                    if (largest ? (v <= worst) : (v >= worst)) continue;
                }
                // The position: the list is sorted, so a binary search finds
                // it in log2(k) comparisons instead of k.
                //
                // The position is the first element that is *worse* than the
                // new one, which is what puts an equal value after the one
                // that is already there (the lower index first).
                int lo = 0, hi = m;
                while (lo < hi) {
                    const int mid = (lo + hi) / 2;
                    const bool worse = largest ? (sv[mid] < v) : (sv[mid] > v);
                    if (worse) hi = mid;
                    else lo = mid + 1;
                }
                const int pos = lo;
                // The shift: the elements from the position on move up by
                // one.  It is a block copy and not a loop: the loop is a
                // chain of dependent global reads and writes, and the chain
                // *is* the cost (the layout model's top-k was 59
                // milliseconds of it).  The blocks are read into registers
                // first, so there is no dependency between the read and the
                // write.
                //
                // (A memmove here is five times *slower*: the device's libc
                // call is not inlined, and the top-k does one per column.)
                //
                // A full list drops its last element.
                const int n = (m < k ? m : m - 1) - pos;
                // The blocks, from the top: the copy moves the elements up,
                // so a block that is read after the one below it has been
                // written reads the *written* values.  (The copy from the
                // bottom up is 4 times faster and it is wrong: the boxes
                // come out in a different order.)
                int end = pos + n;
                while (end - pos >= 8) {
                    const int i = end - 8;
                    float tv[8];
                    int64_t ti[8];
                    for (int j = 0; j < 8; ++j) {
                        tv[j] = sv[i + j];
                        ti[j] = si[i + j];
                    }
                    for (int j = 0; j < 8; ++j) {
                        sv[i + j + 1] = tv[j];
                        si[i + j + 1] = ti[j];
                    }
                    end = i;
                }
                // The tail, from the top as well.
                for (int i = end; i > pos; --i) {
                    sv[i] = sv[i - 1];
                    si[i] = si[i - 1];
                }
                sv[pos] = v;
                si[pos] = c;
                if (m < k) ++m;
            }
            for (int j = 0; j < k; ++j) {
                values[(int64_t)r * k + j] = j < m ? sv[j] : 0.0f;
                indices[(int64_t)r * k + j] = j < m ? si[j] : -1;
            }
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
