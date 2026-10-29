#pragma once
// The SYCL kernels of the ONNX runtime.
//
// Every kernel takes its shapes as *runtime* arguments.  That is the whole
// point of writing them here instead of using a graph runtime: a kernel with
// a runtime shape handles every shape the document produces, so the device
// never has to compile a new kernel set (which is what costs seconds of JIT
// per shape in a graph runtime), and the binary stays ahead-of-time compiled
// like the rest of the engine.
//
// The conventions follow src/backend/gpu/kernels: a launcher per kernel, the
// queue passed in, the device profile consulted through si::dev::for_queue.

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

namespace si {
namespace onnx {

// The copies between the host and the device.
void q_upload(sycl::queue & q, const void * src, void * dst, size_t bytes);
void q_download(sycl::queue & q, const void * src, void * dst, size_t bytes);
// A diagnostic tally of the blocking device->host copies (SI_ONNX_OPSTAT).
extern std::atomic<long> g_q_downloads;
void q_copy_dev(sycl::queue & q, const void * src, void * dst, size_t bytes);

// The elementwise operators, as one kernel with a runtime switch.  The
// switch is on a value that is uniform across the work-group, so it costs
// nothing.
enum class ew_op : int {
    add = 0, sub, mul, div, pow, neg, exp, log, sqrt, erf, floor, ceil, round,
    relu, sigmoid, tanh, hardsigmoid, hardswish, gelu, silu, softplus, abs,
    greater, less, equal, greater_equal, less_equal, logical_and, logical_or, logical_not, sign,
    max_, min_, clip, mod_,
    // The identity: no activation at all.
    none,
};

// The activation's parameters, by value.
//
// They travel *by value* on purpose: a host pointer read from a device
// kernel is garbage (and it faults on a discrete GPU).  The elementwise
// functor has carried them by value from the start; the convolution's
// epilogue did not, and the hard sigmoid read whatever the host's stack
// held - which is why the text detector's mask came out as zeros.
struct act_params_t {
    float p[4] = {-3.0f, 3.0f, 0.0f, 0.0f};
};

// y = op(x, y) for the binary operators, y = op(x) for the unary ones.
// `b` may be null for a unary operator.
//
// `scalar_op` says which operand holds a single value that is broadcast over
// the whole output: -1 for none, 0 for `a` and 1 for `b`.  The broadcast
// kernel has to work out each element's coordinates (an integer division
// per axis), and a scalar operand - which is what the layout model's
// decoder divides by, and what its sentinel constant is - does not need any
// of that.
//
// **Which operand matters.**  The two are not interchangeable: `a - b` is
// not `b - a`, and a flag that only says "one of them is a scalar" reads
// the wrong one.
void ew_launch(sycl::queue & q, const float * a, const float * b, float * y, size_t n,
               ew_op op, const std::vector<float> & params, bool probe = false, int scalar_op = -1,
               ew_op act = ew_op::none, act_params_t act_params = {});

// The kinds of value a cast converts between.  The *output*'s kind matters
// as much as the input's: a cast from int64 to int64 has to write eight
// bytes per element, and a kernel that writes floats into it leaves the
// consumer reading the float's bits as an integer.
// The Cast's input and output widths.
//
//   kCastF32    a float32
//   kCastI32    an int32
//   kCastI64    an int64
//   kCastBool   a boolean: ONE BYTE per element
//   kCastToBool the output is a float32 that holds 0 or 1
//   kCastU8     the output is a uint8 (an output transfer's narrowing cast)
//
// The boolean is a byte and not a float because the models carry real
// booleans (a loop's condition, a mask), and reading a one-byte buffer as
// floats reads three bytes of whatever is next in memory.
enum cast_kind { kCastF32 = 0, kCastI32 = 1, kCastI64 = 2, kCastBool = 3, kCastToBool = 4,
                 kCastU8 = 5 };

// A cast.  The device does the conversion (a host loop over millions of
// elements is 40 milliseconds per call), and the value is written in the
// output's own width.
void cast_launch(sycl::queue & q, const void * x, void * y, size_t n, int in_kind, int out_kind);

// A where: the condition (a float 0/1) picks between the two values.
void where_launch(sycl::queue & q, const float * cond, const float * a, const float * b, float * y,
                  size_t n);

// An elementwise operator with broadcasting: the two inputs may have
// different shapes, and the strides (0 for a broadcast axis) say how to walk
// them.
void ew_bcast_launch(sycl::queue & q, const float * a, const int64_t * a_stride,
                     const float * b, const int64_t * b_stride, float * y, const int64_t * out_shape,
                     int rank, size_t n, ew_op op, const std::vector<float> & params,
                     ew_op act = ew_op::none, act_params_t act_params = {});

// A batched matrix multiply with broadcast leading dims: a (..., M, K) * b
// (..., K, N) -> c (..., M, N).  The leading dims of the two inputs are
// broadcast against each other, so each input gets its own stride per
// leading dim.
void matmul_bcast_launch(sycl::queue & q, const float * a, const float * b, float * c,
                         const int64_t * out_batch, int rank, int64_t M, int64_t K, int64_t N,
                         const int64_t * a_stride, const int64_t * b_stride, int64_t a_batch,
                         int64_t b_batch);

// y[i] = act(y[i] + bias[i % N]): the MatMul's bias and fused activation in
// one pass.  `bias` may be null (activation only).
void bias_act_launch(sycl::queue & q, float * y, const float * bias, size_t n, int64_t width,
                     ew_op act, act_params_t act_params);

// The layout detector's mask-to-box head in one scan.  `logits` is
// [rows][height*width] (the bmm's output), the grids are [height][width]
// coordinate maps, and the outputs are one value per row: whether the mask
// is non-empty, the max of each grid over the mask, and the min (or the
// sentinel when the mask is empty).  The original graph materialised ten
// 12M-element tensors to compute these five values.
void mask_box_launch(sycl::queue & q, const float * logits, const float * grid_a,
                     const float * grid_b, float * any, float * max_a, float * max_b,
                     float * min_a, float * min_b, int64_t rows, int64_t height, int64_t width,
                     float threshold, float sentinel);

// A convolution: x (N, C, H, W) * w (M, C/g, kh, kw) + bias + scale*add, with
// an activation in the epilogue.  The weights are pre-transposed to
// (C/g, kh, kw, M) so that the M weights of one (c, kh, kw) are contiguous.
// Each work-item computes one output pixel and `tm` output channels.
void conv2d_launch(sycl::queue & q, const float * x, const float * w, const float * bias,
                   const float * scale, const float * add, float * y, int N, int C, int H,
                   int W, int M, int Ho, int Wo, int kh, int kw, int sh, int sw, int ph, int pw, int dh,
                   int dw, int group, int tm, ew_op act, act_params_t act_params,
                   bool probe = false);

// The convolution's epilogue, as a separate pass: after oneDNN has written
// the convolution's output, `y = act(y*scale[c] + add[c] + bias[c])`.  The
// order is the SYCL conv's own (bias, then the batch norm's scale, then its
// add, then the activation).
void conv_epilogue_launch(sycl::queue & q, float * y, int N, int C, int H, int W,
                          const float * bias, const float * scale, const float * add, ew_op act,
                          act_params_t act_params);

// The transposed convolution: x (N, C, H, W) * w (C, M/g, kh, kw) -> y
// (N, M, Ho, Wo).  The weights are pre-transposed to (kh, kw, C, M).
// The transposed convolution.  `Ho` and `Wo` are the output's spatial dims:
// the caller computes them (the output padding and the explicit output
// shape are its business, and the executor's shape function has to agree).
void conv_transpose2d_launch(sycl::queue & q, const float * x, const float * w, const float * bias,
                             const float * scale, const float * add, float * y, int N, int C, int H,
                             int W, int M, int kh, int kw, int sh, int sw, int ph, int pw, int dh,
                             int dw, int group, int Ho, int Wo, ew_op act,
                             act_params_t act_params);

// A batch norm over the channel axis: y = (x - mean) / sqrt(var + eps) *
// scale + b.
void batchnorm_launch(sycl::queue & q, const float * x, const float * scale, const float * b,
                      const float * mean, const float * var, float * y, int N, int C, int64_t HW,
                      float eps);

// A max or average pool over a window.
void pool2d_launch(sycl::queue & q, const float * x, float * y, int N, int C, int H, int W, int Ho,
                   int Wo, int kh, int kw, int sh, int sw, int ph, int pw, bool average,
                   bool count_include_pad);

// A resize (nearest or linear) to a new size.
void resize_launch(sycl::queue & q, const float * x, float * y, int N, int C, int H, int W,
                   int Ho, int Wo, bool nearest);

// A global average pool over the spatial dims.
void global_avg_pool_launch(sycl::queue & q, const float * x, float * y, int N, int C, int H,
                            int W);

// The argmax over `n` values with stride `inner`: one work-item per
// (outer, inner) position.  The layout/UNET masks are tens of millions of
// elements, and the host loop that did this was 14.6 ms for the UNET's
// 3-class 1024x288 map.
void argmax_launch(sycl::queue & q, const float * x, int64_t * y, int64_t outer, int64_t n,
                   int64_t inner);

// The argmax over the last axis of a contiguous [outer][n] tensor, *and* the
// value at that index, in one pass.
//
// The recogniser's output is [B, T, 18710] of float32 - 115 MB for a batch of
// 32 - and the client needs the CTC argmax and its score.  Downloading the
// tensor and reducing it on the host costs four copies of those bytes per
// call; this kernel reads it once and the transfer is 16 bytes per row.
// Ties break to the lowest index, as `argmax_launch` and numpy do.
void argmax_max_launch(sycl::queue & q, const float * x, int64_t * y_idx, float * y_max,
                       int64_t outer, int64_t n);

// A reduction over the trailing axes (mean, max, min, sum).
// The reduction's element types: 0 = float32, 1 = int32, 2 = int64.
void reduce_launch_typed(sycl::queue & q, const void * x, void * y, int64_t outer, int64_t axis,
                        int64_t inner, ew_op op, bool mean, int in_kind, int out_kind);
void reduce_launch(sycl::queue & q, const float * x, float * y, int64_t outer, int64_t axis,
                   int64_t inner, ew_op op, bool mean);

// A softmax over the last axis.
void softmax_launch(sycl::queue & q, const float * x, float * y, size_t rows, int64_t cols);

// A layer normalisation over the last axis: y = (x - mean) / sqrt(var + eps) * gamma + beta.
void layer_norm_launch(sycl::queue & q, const float * x, const float * gamma, const float * beta,
                       float * y, size_t rows, int64_t cols, float eps);

// A batched matrix multiply: (M, K) x (K, N) -> (M, N), for a stack of them.
//
// The bias is optional and per output column (the Gemm's).
void matmul_launch(sycl::queue & q, const float * a, const float * b, float * c, int64_t batch,
                   int64_t M, int64_t K, int64_t N, bool transpose_b, const float * bias = nullptr);

// A GatherND: the output's element (b, t) comes from the data's element
// (indices[b][0..k-1], t), where the trailing block (the data's dims from
// the k-th on) is contiguous in both.
//
// `inner` is the trailing block's size, `k` the number of the indexed dims,
// and `data_stride` / `data_dim` the strides and the sizes of the data's
// first k dims.
// A GatherND: the output's element (b, t) is the data's element
// (indices[b][0], ..., indices[b][r-1], t) - the trailing block is
// contiguous in the data.
//
// `r` is the number of the index components (the indices' last dim) and
// `inner` the size of the trailing block (the data's dims from the r-th on).
void gather_nd_launch(sycl::queue & q, const float * data, const void * indices, float * out,
                      int64_t n_out, int64_t inner, int64_t r, bool indices_are_32,
                      const int64_t * data_stride, const int64_t * data_dim, int n_stride);

// A concatenation along an axis: the inputs are copied into the output at
// their offsets.
void concat_launch(sycl::queue & q, const std::vector<const float *> & in,
                   const std::vector<std::vector<int64_t>> & in_shapes, float * y, int axis,
                   int64_t outer);

// A strided copy: the source is read with a per-axis stride and written
// contiguously.  It is the workhorse of the shape operators (reshape,
// transpose, squeeze, slice, gather).
void strided_copy_launch(sycl::queue & q, const float * x, float * y, const int64_t * src_stride,
                         const int64_t * dst_stride, const int64_t * out_shape, int rank,
                         int64_t total);

// A gather along an axis with a list of indices.
void gather_launch(sycl::queue & q, const float * x, const int64_t * idx, float * y,
                   int64_t outer, int64_t axis_dim, int64_t inner, int64_t n_idx);

// The top-k values and indices along the last axis.
//
// `largest` is the operator's attribute: the layout model's decoder asks for
// the *smallest* k (largest=0), and picking the largest instead reorders
// every anchor - the boxes come out in the wrong places.
//
// The top-k of every row.  A work-group selects the k-th (value, index) key
// with a radix scan when k < cols; `indices` is the k indices per row that
// the caller reads back, `values` the k values.  The small-k fallback (k >=
// cols) uses one heap per row.
void topk_launch(sycl::queue & q, const float * x, float * values, int64_t * indices, size_t rows,
                  int64_t cols, int k, bool largest);

// A grid sample (bilinear, "bilinear" padding mode, align_corners = 0).
// A grid sample: the grid gives, for every output pixel, the (x, y) point
// of the input to sample.  The padding is 0 = zeros, 1 = border,
// 2 = reflection.
void grid_sample_launch(sycl::queue & q, const float * x, const float * grid, float * y, int N,
                        int C, int H, int W, int Ho, int Wo, int mode, int padding);

// The ScatterElements, on the device.
//
// The operator is a copy of `data` with the updates written along one axis.
// The host version downloads the whole data tensor, walks it and uploads it
// again - and the table structurer's decoder does that three times *per
// iteration* into a [B, 501, 50] map: at batch 31 that is 4.6 M elements of
// round trip per iteration and it was 37% of the model's time.  The value it
// writes is one row per token, which a kernel does without touching the
// host.
//
// The geometry: the axis is `dim` long, `L` is the product of the dims
// before it and `T` of the dims after it, so the destination of
// (l, axis position p, t) is `(l*dim + p)*T + t`.  The indices have the
// data's shape with the axis dim replaced by 1, i.e. one index per (l, t).
// `slice_form` selects the updates' form: a whole axis slice (the updates
// have the data's shape) or one value per index (the updates have the
// indices' shape).
enum scatter_kind { kScatterF32 = 0, kScatterI32 = 1, kScatterI64 = 2 };
enum scatter_idx_kind { kScatterIdxI32 = 0, kScatterIdxI64 = 1 };
void scatter_elements_launch(sycl::queue & q, const void * data, const void * idx, int idx_kind,
                             int64_t n_idx, const void * upd, int64_t n_upd, void * out, int64_t L,
                             int64_t T, int64_t dim, bool slice_form, int kind);

} // namespace onnx
} // namespace si
