// ---------------------------------------------------------------------------
// DeepSeek-V4.1-Flash vision tower: the three kernels the shared `vit_*`
// library does not already provide.
//
//   ds41_rmsnorm_launch   RMSNorm (weight only) for every block's ln1/ln2 and
//                         the final post_ln.  vit_layernorm_launch subtracts
//                         the mean, which is wrong for this tower.
//   ds41_unfold_launch    the aligner's 3x3 non-overlapping patch merge: zero-
//                         pad the patch grid to multiples of 3 (implicitly, by
//                         writing zero for out-of-grid taps) and lay each cell
//                         out in the reference's im2col order
//                         k = channel*9 + kh*3 + kw.
//   ds41_assemble_launch  the reading-order token block: [img_start], then each
//                         row's cells followed by [image_newline], then
//                         [img_end].  The sentinels are learned rows of the
//                         embedding width, appended after the projector grid.
//
// The linears themselves reuse vit_gemm_launch / vit_add_bias / vit_rope /
// vit_attn / vit_silu_mul from vit.cpp.
// ---------------------------------------------------------------------------
#include "kernels.h"

namespace si {

using namespace sycl;

namespace {

// Sub-group sum for the 32-lane reduction (kernels.h's helpers live in a
// header whose dequant templates emit pre-existing -Warray-bounds noise; this
// tower only needs the reduction).
inline float sg_sum32(float v, const sycl::sub_group & sg) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += sycl::permute_group_by_xor(sg, v, mask);
    }
    return v;
}

} // namespace

// One sub-group per row, lane-strided sum of squares then a shuffle reduction.
// Mirrors vit_layernorm_launch minus the mean subtraction.
void ds41_rmsnorm_launch(queue & q, const float * x, int x_stride, const float * w, float * out, int out_stride,
                         int rows, int n, float eps) {
    if (rows <= 0 || n <= 0) {
        return;
    }
    constexpr int SG = 8; // sub-groups per workgroup
    const int n_wg = (rows + SG - 1) / SG;
    q.parallel_for(nd_range<1>((size_t)n_wg * 256, 256), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int sg = it.get_local_id(0) / 32;
        const int lane = it.get_local_id(0) % 32;
        const int row = it.get_group(0) * SG + sg;
        if (row >= rows) {
            return;
        }
        const float * xr = x + (size_t)row * x_stride;
        float ss = 0.f;
        for (int i = lane; i < n; i += 32) {
            const float v = xr[i];
            ss = sycl::fma(v, v, ss);
        }
        const sub_group sgg = it.get_sub_group();
        ss = sg_sum32(ss, sgg);
        const float inv = sycl::rsqrt(ss / (float)n + eps);
        float * orow = out + (size_t)row * out_stride;
        for (int i = lane; i < n; i += 32) {
            orow[i] = xr[i] * inv * w[i];
        }
    });
}

// Patch-grid unfold.  `x` is [ph*pw][E] in raster order, `fe` is
// [nh*nw][9*E].  A tap outside the real patch grid contributes zero, which is
// the reference's explicit zero-pad of the grid to multiples of `r`.
void ds41_unfold_launch(queue & q, const float * x, int pw, int ph, int nw, int nh, int E, float * fe) {
    if (pw <= 0 || ph <= 0 || nw <= 0 || nh <= 0 || E <= 0) {
        return;
    }
    const int K9 = 9 * E;
    q.parallel_for(sycl::range<2>((size_t)nh * nw, (size_t)K9), [=](id<2> idx) {
        const int cell = (int)idx[0];
        const int k = (int)idx[1];
        const int cy = cell / nw;
        const int cx = cell % nw;
        const int c = k / 9;
        const int tap = k % 9;
        const int py = cy * 3 + tap / 3;
        const int px = cx * 3 + tap % 3;
        float v = 0.f;
        if (py < ph && px < pw) {
            v = x[((size_t)py * pw + px) * E + c];
        }
        fe[(size_t)cell * K9 + k] = v;
    });
}

// Reading-order token block.  `grid` is [nh*nw][P]; the sentinels are the three
// learned 5120-wide rows.
void ds41_assemble_launch(queue & q, const float * grid, int nw, int nh, int P, const float * img_start,
                          const float * img_end, const float * img_newline, float * out) {
    if (nw <= 0 || nh <= 0 || P <= 0) {
        return;
    }
    const int n_out = nh * (nw + 1) + 2;
    q.parallel_for(sycl::range<2>((size_t)n_out, (size_t)P), [=](id<2> idx) {
        const int r = (int)idx[0];
        const int c = (int)idx[1];
        const float * src = nullptr;
        if (r == 0) {
            src = img_start + c;
        } else if (r == n_out - 1) {
            src = img_end + c;
        } else {
            const int rr = r - 1;
            const int row = rr / (nw + 1);
            const int col = rr % (nw + 1);
            src = col < nw ? grid + ((size_t)row * nw + col) * P + c : img_newline + c;
        }
        out[(size_t)r * P + c] = *src;
    });
}

} // namespace si
