#include "kernels.h"
#include "device/device_profile.h"
#include "kernel_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "common/env.h"

namespace si {

using namespace sycl;
using namespace si::kd;

// ---------------------------------------------------------------------------
// n_tb > 1: one grid dimension over TB-token blocks, so chunk-batched prefill
// can cover many rows in a single dispatch (the per-token pointers are just
// offset by tblk*TB*stride; mode 2 lays tokens out flat).
template <uint32_t QT, int TB, int RPS, int NSB, int SGW = 8>
static void gemv_multi_kernel(queue & q, const gemv_seg * segs, int n_segs, int total_rows, int n_tb = 1) {
    const int rows_per_wg = RPS * SGW;
    const int n_wg = (total_rows + rows_per_wg - 1) / rows_per_wg;
    const int wg_threads = SGW * 32;
    q.submit([&](handler & h) {
        local_accessor<float, 1> xs(TB == 1 ? 1 : 256 * TB, h);
        h.parallel_for(nd_range<1>((size_t)n_wg * n_tb * wg_threads, wg_threads),
                       [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int tid = it.get_local_id(0);
                           const int sg = tid / 32;
                           const int lane = tid % 32;
                           const int tblk = TB == 1 ? 0 : (it.get_group(0) / n_wg);
                           const int row_base = (it.get_group(0) % n_wg) * rows_per_wg + sg * RPS;
                           (void)xs;

                           int seg_idx = -1;
                           int row_in_seg = 0;
                           {
                               int acc_rows = 0;
                               for (int s = 0; s < n_segs; s++) {
                                   if (row_base < acc_rows + segs[s].n_rows) {
                                       seg_idx = s;
                                       row_in_seg = row_base - acc_rows;
                                       break;
                                   }
                                   acc_rows += segs[s].n_rows;
                               }
                           }
                           if (seg_idx < 0) {
                               return;
                           }

                           // cache the segment descriptor in registers (avoid re-loading from device memory)
                           const gemv_seg sgd = segs[seg_idx];
                           const char * __restrict__ W = (const char *)sgd.w;
                           const int NROWS = sgd.n_rows;
                           const int XSTRIDE = sgd.x_stride;
                           const int OSTRIDE = sgd.out_stride;
                           const float * __restrict__ X = sgd.x + (size_t)tblk * TB * sgd.x_stride;
                           const float * __restrict__ UP =
                               sgd.act_up ? sgd.act_up + (size_t)tblk * TB * sgd.x_stride : nullptr;
                           const float * __restrict__ RES =
                               sgd.residual ? sgd.residual + (size_t)tblk * TB * sgd.out_stride : nullptr;
                           float * __restrict__ OUT = sgd.out + (size_t)tblk * TB * sgd.out_stride;
                           const float ALPHA = sgd.alpha;

                           float acc[RPS][TB];
                           for (int r = 0; r < RPS; r++) {
                               for (int t = 0; t < TB; t++) {
                                   acc[r][t] = 0.f;
                               }
                           }

                           const int n_sb = NSB ? NSB : (sgd.K / 256);
                           const int sb_bytes = superblock_bytes(QT);

                           {
                               for (int sb = 0; sb < n_sb; sb++) {
                                   if constexpr (TB > 1) {
                                       for (int i = tid; i < 256; i += wg_threads) {
                                           for (int t = 0; t < TB; t++) {
                                               float v = X[(size_t)t * XSTRIDE + sb * 256 + i];
                                               if (UP) {
                                                   const float g = v;
                                                   const float uv = UP[(size_t)t * XSTRIDE + sb * 256 + i];
                                                   v = silu_f(g) * uv;
                                               }
                                               xs[t * 256 + i] = v;
                                           }
                                       }
                                       it.barrier();
                                   }
                                   for (int r = 0; r < RPS; r++) {
                                       const int row = row_in_seg + r;
                                       if (row >= NROWS) {
                                           continue;
                                       }
                                       const char * wrow = W + ((size_t)row * n_sb + sb) * sb_bytes;
                                       float w[8];
                                       dequant_sb_lane_typed<QT>(wrow, lane, w);
                                       if constexpr (TB == 1) {
                                           const float * xrow = X + sb * 256;
                                           const float * urow = UP ? UP + sb * 256 : nullptr;
                                           float a = 0.f;
#pragma unroll
                                           for (int b = 0; b < 8; b++) {
                                               float xv = xrow[32 * b + lane];
                                               if (urow) {
                                                   xv = silu_f(xv) * urow[32 * b + lane];
                                               }
                                               a += w[b] * xv;
                                           }
                                           acc[r][0] += a;
                                       } else {
#pragma unroll
                                           for (int b = 0; b < 8; b++) {
#pragma unroll
                                               for (int t = 0; t < TB; t++) {
                                                   acc[r][t] += w[b] * xs[t * 256 + 32 * b + lane];
                                               }
                                           }
                                       }
                                   }
                                   if constexpr (TB > 1) {
                                       it.barrier();
                                   }
                               }
                           }

                           const sub_group sgg = it.get_sub_group();
                           for (int r = 0; r < RPS; r++) {
                               const int row = row_in_seg + r;
                               if (row >= NROWS) {
                                   continue;
                               }
                               for (int t = 0; t < TB; t++) {
                                   float v = sg_sum(acc[r][t], sgg);
                                   if (lane == 0) {
                                       const size_t oidx = (size_t)t * OSTRIDE + row;
                                       float o = v * ALPHA;
                                       if (RES) {
                                           o += RES[oidx];
                                       }
                                       OUT[oidx] = o;
                                   }
                               }
                           }
                       });
    });
}

#define GEMV_DISPATCH_TB_SGW(QT, TB_, SGW_)                                                                            \
    do {                                                                                                               \
        switch (nsb) {                                                                                                 \
        case 4: gemv_multi_kernel<QT, TB_, 1, 4, SGW_>(q, segs, n_segs, total_rows, n_tb); break;                      \
        case 8: gemv_multi_kernel<QT, TB_, 1, 8, SGW_>(q, segs, n_segs, total_rows, n_tb); break;                      \
        case 14: gemv_multi_kernel<QT, TB_, 1, 14, SGW_>(q, segs, n_segs, total_rows, n_tb); break;                    \
        default: gemv_multi_kernel<QT, TB_, 1, 0, SGW_>(q, segs, n_segs, total_rows, n_tb); break;                     \
        }                                                                                                              \
    } while (0)

#define GEMV_DISPATCH_TB(QT, TB_)                                                                                      \
    do {                                                                                                               \
        switch (nsb) {                                                                                                 \
        case 4: gemv_multi_kernel<QT, TB_, 1, 4, 8>(q, segs, n_segs, total_rows, n_tb); break;              \
        case 8: gemv_multi_kernel<QT, TB_, 1, 8, 8>(q, segs, n_segs, total_rows, n_tb); break;              \
        case 14: gemv_multi_kernel<QT, TB_, 1, 14, 8>(q, segs, n_segs, total_rows, n_tb); break;            \
        default: gemv_multi_kernel<QT, TB_, 1, 0, 8>(q, segs, n_segs, total_rows, n_tb); break;             \
        }                                                                                                              \
    } while (0)

// experimental: cfg = RPS*100 + SGW (SGW ignored unless listed)
static int gemv_env_cfg(const char * name) {
    const char * s = si::env::str(name);
    return s ? atoi(s) : 0;
}

#define GEMV_DISPATCH_CFG(QT, TB_, CFG)                                                                                \
    do {                                                                                                               \
        switch (CFG) {                                                                                                 \
        case 104: gemv_multi_kernel<QT, TB_, 1, 0, 4>(q, segs, n_segs, total_rows); break;                   \
        case 204: gemv_multi_kernel<QT, TB_, 2, 0, 4>(q, segs, n_segs, total_rows); break;                   \
        case 208: gemv_multi_kernel<QT, TB_, 2, 0, 8>(q, segs, n_segs, total_rows); break;                   \
        case 216: gemv_multi_kernel<QT, TB_, 2, 0, 16>(q, segs, n_segs, total_rows); break;                  \
        case 404: gemv_multi_kernel<QT, TB_, 4, 0, 4>(q, segs, n_segs, total_rows); break;                   \
        case 408: gemv_multi_kernel<QT, TB_, 4, 0, 8>(q, segs, n_segs, total_rows); break;                   \
        case 416: gemv_multi_kernel<QT, TB_, 4, 0, 16>(q, segs, n_segs, total_rows); break;                  \
        case 808: gemv_multi_kernel<QT, TB_, 8, 0, 8>(q, segs, n_segs, total_rows); break;                   \
        default: gemv_multi_kernel<QT, TB_, 1, 0, 8>(q, segs, n_segs, total_rows); break;                    \
        }                                                                                                              \
    } while (0)

// dispatch a type over the four token-block widths
#define GEMV_TB_SWITCH(QT)                                                                                             \
    do {                                                                                                               \
        if (TB == 1) {                                                                                                 \
            GEMV_DISPATCH_TB(QT, 1);                                                                                   \
        } else if (TB == 8) {                                                                                          \
            GEMV_DISPATCH_TB(QT, 8);                                                                                   \
        } else if (TB == 16) {                                                                                         \
            GEMV_DISPATCH_TB(QT, 16);                                                                                  \
        } else {                                                                                                       \
            GEMV_DISPATCH_TB(QT, 32);                                                                                  \
        }                                                                                                              \
    } while (0)

// ---------------------------------------------------------------------------
// Single-token GEMV for Q4_K / Q5_K with a vectorized dequant mapping.
//
// The generic kernel assigns lane = k-position, so each lane dequantizes 8
// values from 8 *different* 32-value sub-blocks: it must decode all 8
// (scale, min) pairs and does 4 byte gathers.  Here each lane instead covers 8
// *consecutive* values of ONE sub-block: the 8 bytes come from a single
// uint2 load and the lane needs exactly one (scale, min) pair, and the
// activations are two float4 loads.  Measured ~1.5x fewer instructions/value.
template <uint32_t QT>
static void gemv_dec_vec_kernel(queue & q, const gemv_seg * segs, int n_segs, int total_rows, int nsb) {
    // rows per work-group comes from the device profile (8 on both cards): it is
    // a workgroup size, so it scales with what the part wants resident
    const int rows_per_wg = si::dev::for_queue(q).shape.gemv_rows_per_wg;
    const int n_wg = (total_rows + rows_per_wg - 1) / rows_per_wg;
    const int sb_bytes = (QT == 13) ? 176 : 144;
    q.parallel_for(nd_range<1>((size_t)n_wg * 256, 256), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int sg = it.get_local_id(0) / 32;
        const int lane = it.get_local_id(0) % 32;
        const int row_base = it.get_group(0) * rows_per_wg + sg;
        int seg_idx = -1, row_in_seg = 0;
        {
            int acc = 0;
            for (int si = 0; si < n_segs; si++) {
                if (row_base < acc + segs[si].n_rows) {
                    seg_idx = si;
                    row_in_seg = row_base - acc;
                    break;
                }
                acc += segs[si].n_rows;
            }
        }
        if (seg_idx < 0) {
            return;
        }
        const gemv_seg sgd = segs[seg_idx];
        const int n_sb = nsb ? nsb : (sgd.K / 256);
        const char * wrow = (const char *)sgd.w + (size_t)row_in_seg * n_sb * sb_bytes;
        // lane covers elements [32*s + 8*m, +8): s = lane/4 (sub-block), m = lane%4
        const int s = lane / 4, m = lane % 4;
        const int c = s / 2, half = s % 2;
        float acc = 0.f;
        for (int sb = 0; sb < n_sb; sb++) {
            const char * blk = wrow + (size_t)sb * sb_bytes;
            float ds, dm;
            if (sgd.meta32) {
                // pre-extracted fp32 (scale, min) for this 32-value sub-block
                const sycl::float2 m = sgd.meta32[(size_t)row_in_seg * (sgd.K / 32) + sb * 8 + s];
                ds = m.x();
                dm = m.y();
            } else {
                const float d = fast_h2f(*(const uint16_t *)blk);
                const float dmin = fast_h2f(*(const uint16_t *)(blk + 2));
                uint8_t sc, mn;
                get_scale_min_k4(s, (const uint8_t *)(blk + 4), &sc, &mn);
                ds = d * sc;
                dm = dmin * mn;
            }
            // single-token decode: x has exactly one row
            const float * xb = sgd.x + sb * 256 + 32 * s + 8 * m;
            // ffn_down applies silu(x)*up (the up operand shares x's stride)
            const float * upb = sgd.act_up ? sgd.act_up + sb * 256 + 32 * s + 8 * m : nullptr;
            const uint8_t * qb;
            uint8_t hb[8];
            if constexpr (QT == 13) {
                qb = (const uint8_t *)(blk + 48) + 32 * c + 8 * m;
                const uint8_t * qh = (const uint8_t *)(blk + 16) + 8 * m;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    hb[j] = qh[j];
                }
            } else {
                qb = (const uint8_t *)(blk + 16) + 32 * c + 8 * m;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    hb[j] = 0;
                }
            }
            const uint2 q = *reinterpret_cast<const uint2 *>(qb);
            const float4 xa = *reinterpret_cast<const float4 *>(xb);
            const float4 xb4 = *reinterpret_cast<const float4 *>(xb + 4);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t byte = (q.x() >> (8 * j)) & 0xFF;
                const int nib = half ? (int)(byte >> 4) : (int)(byte & 0xF);
                const int hbit = (QT == 13) ? (((hb[j] >> (2 * c + half)) & 1) << 4) : 0;
                const float xv = upb ? silu_f(xa[j]) * upb[j] : xa[j];
                acc += (ds * (float)(nib + hbit) - dm) * xv;
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t byte = (q.y() >> (8 * j)) & 0xFF;
                const int nib = half ? (int)(byte >> 4) : (int)(byte & 0xF);
                const int hbit = (QT == 13) ? (((hb[j + 4] >> (2 * c + half)) & 1) << 4) : 0;
                const float xv = upb ? silu_f(xb4[j]) * upb[j + 4] : xb4[j];
                acc += (ds * (float)(nib + hbit) - dm) * xv;
            }
        }
        acc = sg_sum(acc, it.get_sub_group());
        if (lane == 0) {
            float v = sgd.alpha * acc;
            if (sgd.residual) {
                v += sgd.residual[row_in_seg];
            }
            sgd.out[row_in_seg] = v;
        }
    });
}

// ---------------------------------------------------------------------------
// Q8_0 single-token decode GEMV, 8 consecutive values per lane.
//
// Q8_0 stores 32 *contiguous* int8 values plus one f16 scale in a 34-byte
// block, so lane -> (s = block in the 256-value superblock, m = 8-value chunk)
// lets one lane cover 8 values of ONE block: one scale load, four uint16 weight
// loads and two float4 activation loads.  The generic lane = k-position mapping
// needs 8 of each for the same 8 values (8 different blocks' scales, 8 byte
// loads, 8 scalar activation loads) - that is the dense-trunk floor described
// in the Q8_0 note in AGENTS.md (55-60 GB/s for wide shapes).
//
// The qs payload starts at offset 2 of a 34-byte block, so a value window is
// only 2-byte aligned.  An 8-byte (uint2) load at that alignment is accepted by
// the hardware in a JIT build but the AOT (acm-g10) lowering of this TU widened
// it: the logits came out uncorrelated with the generic path (max|diff| 15.4 of
// a 18.6 max) while the same code was exact under JIT.  So the weights are read
// as four naturally aligned uint16 (two int8 each); the scale and the two
// float4 activation loads are still 1 and 2 per 8 values instead of 8 and 8.
//
// `WPR > 1` is the narrow-shape occupancy fix.  The HC down (K = 10240,
// N = 320) leaves a one-row-per-warp grid at 320 warps = 0.6 warps/EU;
// WPR warps split the superblocks of ONE row inside a work-group and reduce
// through SLM, without a second kernel or a global partial buffer.
//
// Both mappings change the order the K elements are summed in (ulp level), so
// the path is gated by PF_Q4_GEMV8: default on (0 restores the generic GEMV) and
// checked against the logits and the token stream - see the measurements in the
// Q8_0 note in AGENTS.md.
static inline float q8_0_dot_vec(const char * __restrict__ row, int sb0, int sb1, const float * __restrict__ x,
                                 int lane) {
    const int s = lane >> 2;          // 32-value block inside the superblock
    const int m = (lane & 3) << 3;    // 8-value chunk inside that block
    float acc = 0.f;
    for (int sb = sb0; sb < sb1; sb++) {
        const char * blk = row + (size_t)(sb * 8 + s) * 34;
        const float d = fast_h2f(*(const uint16_t *)blk);
        const uint16_t * q16 = reinterpret_cast<const uint16_t *>(blk + 2 + m);
        const uint16_t w0 = q16[0], w1 = q16[1], w2 = q16[2], w3 = q16[3];
        const float * xp = x + (size_t)sb * 256 + 32 * s + m;
        const float4 xa = *reinterpret_cast<const float4 *>(xp);
        const float4 xb = *reinterpret_cast<const float4 *>(xp + 4);
        acc += d * (float)(int8_t)(uint8_t)(w0 >> 0) * xa.x();
        acc += d * (float)(int8_t)(uint8_t)(w0 >> 8) * xa.y();
        acc += d * (float)(int8_t)(uint8_t)(w1 >> 0) * xa.z();
        acc += d * (float)(int8_t)(uint8_t)(w1 >> 8) * xa.w();
        acc += d * (float)(int8_t)(uint8_t)(w2 >> 0) * xb.x();
        acc += d * (float)(int8_t)(uint8_t)(w2 >> 8) * xb.y();
        acc += d * (float)(int8_t)(uint8_t)(w3 >> 0) * xb.z();
        acc += d * (float)(int8_t)(uint8_t)(w3 >> 8) * xb.w();
    }
    return acc;
}

// Generic lane = k-position Q8_0 dot, kept for a K that is not a multiple of
// 256 and for the silu(x)*up epilogue.  The terms and their per-lane order are
// the generic kernel's (k = lane, lane+32, ...); only the superblock-boundary
// association differs (one extra rounding per 256 values).
static inline float q8_0_dot_scalar(const char * __restrict__ row, int K, const float * __restrict__ x,
                                    const float * __restrict__ up, int lane) {
    float acc = 0.f;
    for (int k = lane; k < K; k += 32) {
        const char * blk = row + (size_t)(k >> 5) * 34;
        float xv = x[k];
        if (up) {
            xv = silu_f(xv) * up[k];
        }
        acc += fast_h2f(*(const uint16_t *)blk) * (float)(int8_t)blk[2 + (k & 31)] * xv;
    }
    return acc;
}

static inline void gemv_q8_find_seg(const gemv_seg * segs, int n_segs, int row_base, int & seg_idx,
                                    int & row_in_seg) {
    int acc_rows = 0;
    seg_idx = -1;
    row_in_seg = 0;
    for (int s = 0; s < n_segs; s++) {
        if (row_base < acc_rows + segs[s].n_rows) {
            seg_idx = s;
            row_in_seg = row_base - acc_rows;
            return;
        }
        acc_rows += segs[s].n_rows;
    }
}

// One warp per output row.
static void gemv_dec_q8_kernel(queue & q, const gemv_seg * segs, int n_segs, int total_rows, int rows_per_wg,
                               bool force_scalar) {
    const int n_wg = (total_rows + rows_per_wg - 1) / rows_per_wg;
    q.parallel_for(nd_range<1>((size_t)n_wg * rows_per_wg * 32, (size_t)rows_per_wg * 32),
                   [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                       const int sg = (int)it.get_local_id(0) / 32;
                       const int lane = (int)it.get_local_id(0) % 32;
                       const int row_base = (int)it.get_group(0) * rows_per_wg + sg;
                       if (row_base >= total_rows) {
                           return;
                       }
                       int seg_idx = -1, row_in_seg = 0;
                       gemv_q8_find_seg(segs, n_segs, row_base, seg_idx, row_in_seg);
                       if (seg_idx < 0) {
                           return;
                       }
                       const gemv_seg sgd = segs[seg_idx];
                       const int n_sb = sgd.K / 256;
                       const char * row = (const char *)sgd.w + (size_t)row_in_seg * (size_t)(sgd.K / 32) * 34;
                       const bool vec = !force_scalar && !sgd.act_up && n_sb * 256 == sgd.K;
                       float acc = vec ? q8_0_dot_vec(row, 0, n_sb, sgd.x, lane)
                                       : q8_0_dot_scalar(row, sgd.K, sgd.x, sgd.act_up, lane);
                       acc = sg_sum(acc, it.get_sub_group());
                       if (lane == 0) {
                           float v = acc * sgd.alpha;
                           if (sgd.residual) {
                               v += sgd.residual[row_in_seg];
                           }
                           sgd.out[row_in_seg] = v;
                       }
                   });
}

// WPR warps on ONE output row, each taking a contiguous superblock range, with
// an SLM reduction at the end.  Grid = one work-group per row.
template <int WPR>
static void gemv_dec_q8_split_kernel(queue & q, const gemv_seg * segs, int n_segs, int total_rows,
                                     bool force_scalar) {
    q.submit([&](handler & h) {
        local_accessor<float, 1> part(WPR, h);
        h.parallel_for(nd_range<1>((size_t)total_rows * WPR * 32, (size_t)WPR * 32),
                       [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int w = (int)it.get_local_id(0) / 32;
                           const int lane = (int)it.get_local_id(0) % 32;
                           const int row = (int)it.get_group(0);
                           int seg_idx = -1, row_in_seg = 0;
                           gemv_q8_find_seg(segs, n_segs, row, seg_idx, row_in_seg);
                           if (seg_idx < 0) {
                               return;
                           }
                           const gemv_seg sgd = segs[seg_idx];
                           const int n_sb = sgd.K / 256;
                           const char * rp = (const char *)sgd.w + (size_t)row_in_seg * (size_t)(sgd.K / 32) * 34;
                           const bool vec = !force_scalar && !sgd.act_up && n_sb * 256 == sgd.K;
                           float acc;
                           if (vec) {
                               // whole superblocks per warp, so the 8-value
                               // lane mapping is unchanged
                               const int sb0 = (int)((int64_t)n_sb * w / WPR);
                               const int sb1 = (int)((int64_t)n_sb * (w + 1) / WPR);
                               acc = q8_0_dot_vec(rp, sb0, sb1, sgd.x, lane);
                           } else {
                               // unsupported geometry: one warp does the whole
                               // row in the generic order, the others add zero
                               acc = w == 0 ? q8_0_dot_scalar(rp, sgd.K, sgd.x, sgd.act_up, lane) : 0.f;
                           }
                           acc = sg_sum(acc, it.get_sub_group());
                           if (lane == 0) {
                               part[w] = acc;
                           }
                           it.barrier();
                           if (it.get_local_id(0) == 0) {
                               float v = 0.f;
                               for (int i = 0; i < WPR; i++) {
                                   v += part[i];
                               }
                               v *= sgd.alpha;
                               if (sgd.residual) {
                                   v += sgd.residual[row_in_seg];
                               }
                               sgd.out[row_in_seg] = v;
                           }
                       });
    });
}

// F32 decode GEMV for a NARROW group: WPR warps split one row's superblocks,
// reduce through SLM.
//
// qwen4exp's ssm_beta/ssm_alpha are [48][2560] f32 each, so the 2-tensor group
// is 96 rows.  One warp per row leaves 96 warps (0.19/EU) and the kernel
// measures 0.104 ms for 0.98 MB (9.4 GB/s) - it is latency bound, not byte
// bound; the same rows spread over WPR warps per row reach the card's rate
// (the Q8_0 split path takes the 320-row HC down to 0.035 ms for 13.1 MB).
//
// It changes the summation order: the generic kernel accumulates all
// superblocks in one lane chain, this one reduces WPR partial sums.  The
// per-superblock `w[x] * x[k]` order is the generic kernel's, and `n_sb` is
// `K / 256` so a tail shorter than a superblock is dropped exactly as the
// generic path drops it.  Measured on qwen4exp: 0.104 -> 0.031 ms per call at
// WPR=8 (36 calls/token), a 160-token generation is byte-identical to the
// generic path, and the 64-token decode wall is 14.36 s against 14.55 s.
// PF_Q4_F32_SPLIT selects the warp count (0 = generic, 8 = default).
template <int WPR>
static void gemv_dec_f32_split_kernel(queue & q, const gemv_seg * segs, int n_segs, int total_rows) {
    q.submit([&](handler & h) {
        local_accessor<float, 1> part(WPR, h);
        h.parallel_for(nd_range<1>((size_t)total_rows * WPR * 32, (size_t)WPR * 32),
                       [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int w = (int)it.get_local_id(0) / 32;
                           const int lane = (int)it.get_local_id(0) % 32;
                           const int row = (int)it.get_group(0);
                           int seg_idx = -1, row_in_seg = 0;
                           gemv_q8_find_seg(segs, n_segs, row, seg_idx, row_in_seg);
                           if (seg_idx < 0) {
                               return;
                           }
                           const gemv_seg sgd = segs[seg_idx];
                           const int n_sb = sgd.K / 256;
                           // the generic kernel's row stride (n_sb superblocks),
                           // so a K that is not a multiple of 256 is covered
                           // exactly as the generic path covers it
                           const char * rp =
                               (const char *)sgd.w + (size_t)row_in_seg * (size_t)n_sb * 1024;
                           const int sb0 = (int)((int64_t)n_sb * w / WPR);
                           const int sb1 = (int)((int64_t)n_sb * (w + 1) / WPR);
                           float acc = 0.f;
                           for (int sb = sb0; sb < sb1; sb++) {
                               const char * p = rp + (size_t)sb * 1024;
                               const float * xrow = sgd.x + (size_t)sb * 256;
#pragma unroll
                               for (int b = 0; b < 8; b++) {
                                   acc += ((const float *)p)[32 * b + lane] * xrow[32 * b + lane];
                               }
                           }
                           acc = sg_sum(acc, it.get_sub_group());
                           if (lane == 0) {
                               part[w] = acc;
                           }
                           it.barrier();
                           if (it.get_local_id(0) == 0) {
                               float v = 0.f;
                               for (int i = 0; i < WPR; i++) {
                                   v += part[i];
                               }
                               v *= sgd.alpha;
                               if (sgd.residual) {
                                   v += sgd.residual[row_in_seg];
                               }
                               sgd.out[row_in_seg] = v;
                           }
                       });
    });
}

static void gemv_group_impl(queue & q, uint32_t type, const gemv_seg * segs, int n_segs, int total_rows, int TB,
                            int nsb, int n_tok_blocks) {
    const int n_tb = (TB > 1 && n_tok_blocks > 0) ? n_tok_blocks : 1;
    static const bool dbg_g = si::env::flag("PF_DBG_GEMV");
    if (dbg_g) {
        // the segments live on the device, so only log the host side here
        fprintf(stderr, "[gemv] type=%u n_segs=%d total_rows=%d TB=%d nsb=%d segs=%p\n", type, n_segs,
                total_rows, TB, nsb, (const void *)segs);
    }
    static const int cfg1 = gemv_env_cfg("GEMV_CFG1");
    // single-token decode: use the vectorized-dequant kernels for Q4_K/Q5_K
    static const bool dec_vec = [] {
        const char * e = si::env::str("GEMV_DEC_VEC");
        return !(e && atoi(e) == 0);
    }();
    static const bool vec12 = [] {
        const char * e = si::env::str("GEMV_VEC12");
        return !(e && atoi(e) == 0);
    }();
    static const bool vec13 = [] {
        const char * e = si::env::str("GEMV_VEC13");
        return !(e && atoi(e) == 0);
    }();
    const bool use_vec = dec_vec && ((type == 12 && vec12) || (type == 13 && vec13));
    if (TB == 1 && !cfg1 && use_vec && (type == 12 || type == 13)) {
        if (type == 12) {
            gemv_dec_vec_kernel<12>(q, segs, n_segs, total_rows, nsb);
        } else {
            gemv_dec_vec_kernel<13>(q, segs, n_segs, total_rows, nsb);
        }
        return;
    }
    // F32 narrow decode group (qwen4exp's ssm_beta/ssm_alpha): split the row
    // over WPR warps.  On by default; PF_Q4_F32_SPLIT=0 restores the generic
    // path (see the kernel comment for the measurement and the reassociation).
    // The row gate and the warp count are per-card tuning (profile.shape.
    // f32_rows / split.max).  The env is read with a -1 sentinel rather than
    // through profile_int: 0 is a real value here (off), and profile_int folds
    // an env 0 into "unset".
    const si::dev::profile & dp = si::dev::for_queue(q);
    static const int f32_split_env = si::env::i32("PF_Q4_F32_SPLIT", -1);
    const int f32_split = f32_split_env >= 0 ? f32_split_env : dp.split.max;
    if (TB == 1 && !cfg1 && type == 0 && total_rows <= dp.shape.f32_rows && f32_split > 0) {
        switch (f32_split) {
        case 2: gemv_dec_f32_split_kernel<2>(q, segs, n_segs, total_rows); return;
        case 4: gemv_dec_f32_split_kernel<4>(q, segs, n_segs, total_rows); return;
        case 16: gemv_dec_f32_split_kernel<16>(q, segs, n_segs, total_rows); return;
        default: gemv_dec_f32_split_kernel<8>(q, segs, n_segs, total_rows); return;
        }
    }
    // Q8_0 decode: vectorized 8-values-per-lane mapping, with a warp split for
    // the narrow shapes.  On by default; PF_Q4_GEMV8=0 restores the generic
    // GEMV (see the Q8_0 note above for the measurement).
    static const int g8_on = si::env::i32("PF_Q4_GEMV8", 1);
    // A/B bisection only: PF_Q4_GEMV8_MAX_ROWS leaves the wide shapes (the LM
    // head) on the generic path.
    static const int g8_max_rows = si::env::i32("PF_Q4_GEMV8_MAX_ROWS", 1 << 30);
    if (TB == 1 && g8_on && !cfg1 && type == 8 && total_rows <= g8_max_rows) {
        // Same rule as the F32 gate: the env is a -1-sentinel override so that
        // 0 (never split) and 1 (one warp, i.e. the unsplit kernel) stay
        // expressible; the default is this card's profile.
        static const int g8_wpr_env = si::env::i32("PF_Q4_GEMV8_WPR", -1);
        static const int g8_rows_env = si::env::i32("PF_Q4_GEMV8_SPLIT_ROWS", -1);
        const int g8_wpr = g8_wpr_env >= 0 ? g8_wpr_env : dp.split.max;
        const int g8_rows = g8_rows_env >= 0 ? g8_rows_env : dp.shape.q8_rows;
        static const int g8_rpw_env = si::env::i32("PF_Q4_GEMV8_RPW", 0);
        static const bool g8_scalar = si::env::flag("PF_Q4_GEMV8_SCALAR");
        if (g8_wpr > 1 && total_rows <= g8_rows) {
            switch (g8_wpr) {
            case 2: gemv_dec_q8_split_kernel<2>(q, segs, n_segs, total_rows, g8_scalar); return;
            case 4: gemv_dec_q8_split_kernel<4>(q, segs, n_segs, total_rows, g8_scalar); return;
            case 16: gemv_dec_q8_split_kernel<16>(q, segs, n_segs, total_rows, g8_scalar); return;
            default: gemv_dec_q8_split_kernel<8>(q, segs, n_segs, total_rows, g8_scalar); return;
            }
        }
        const int rpw = g8_rpw_env > 0 ? g8_rpw_env : si::dev::for_queue(q).shape.gemv_rows_per_wg;
        gemv_dec_q8_kernel(q, segs, n_segs, total_rows, rpw, g8_scalar);
        return;
    }
    static const int cfg8 = gemv_env_cfg("GEMV_CFG8");
    static const int cfg16 = gemv_env_cfg("GEMV_CFG16");
    static const int cfg32 = gemv_env_cfg("GEMV_CFG32");
    if ((TB == 1 && cfg1) || (TB == 8 && cfg8) || (TB == 16 && cfg16) || (TB == 32 && cfg32)) {
#define GEMV_CFG_ALL(CFG, TB_)                                                                                         \
    switch (type) {                                                                                                    \
    case 12: GEMV_DISPATCH_CFG(12, TB_, CFG); break;                                                                   \
    case 13: GEMV_DISPATCH_CFG(13, TB_, CFG); break;                                                                   \
    case 14: GEMV_DISPATCH_CFG(14, TB_, CFG); break;                                                                   \
    case 11: GEMV_DISPATCH_CFG(11, TB_, CFG); break;                                                                   \
    case 20: GEMV_DISPATCH_CFG(20, TB_, CFG); break;                                                                   \
    case 21: GEMV_DISPATCH_CFG(21, TB_, CFG); break;                                                                   \
    case 23: GEMV_DISPATCH_CFG(23, TB_, CFG); break;                                                                   \
    case 8: GEMV_DISPATCH_CFG(8, TB_, CFG); break;                                                                     \
    case 7: GEMV_DISPATCH_CFG(7, TB_, CFG); break;                                                                     \
    case 30: GEMV_DISPATCH_CFG(30, TB_, CFG); break;                                                                   \
    case 1: GEMV_DISPATCH_CFG(1, TB_, CFG); break;                                                                     \
    default: GEMV_DISPATCH_CFG(0, TB_, CFG); break;                                                                    \
    }
        if (TB == 1) {
            GEMV_CFG_ALL(cfg1, 1)
        } else if (TB == 8) {
            GEMV_CFG_ALL(cfg8, 8)
        } else if (TB == 16) {
            GEMV_CFG_ALL(cfg16, 16)
        } else {
            GEMV_CFG_ALL(cfg32, 32)
        }
        return;
    }
    switch (type) {
    case 12:
        if (TB == 1) {
            GEMV_DISPATCH_TB(12, 1);
        } else if (TB == 8) {
            GEMV_DISPATCH_TB(12, 8);
        } else if (TB == 16) {
            GEMV_DISPATCH_TB(12, 16);
        } else {
            GEMV_DISPATCH_TB(12, 32);
        }
        break;
    case 13:
        if (TB == 1) {
            GEMV_DISPATCH_TB(13, 1);
        } else if (TB == 8) {
            GEMV_DISPATCH_TB(13, 8);
        } else if (TB == 16) {
            GEMV_DISPATCH_TB(13, 16);
        } else {
            GEMV_DISPATCH_TB(13, 32);
        }
        break;
    case 14:
        if (TB == 1) {
            GEMV_DISPATCH_TB(14, 1);
        } else if (TB == 8) {
            GEMV_DISPATCH_TB(14, 8);
        } else if (TB == 16) {
            GEMV_DISPATCH_TB(14, 16);
        } else {
            GEMV_DISPATCH_TB(14, 32);
        }
        break;
    case 11:
        GEMV_TB_SWITCH(11);
        break;
    case 20:
        GEMV_TB_SWITCH(20);
        break;
    case 21:
        GEMV_TB_SWITCH(21);
        break;
    case 23:
        GEMV_TB_SWITCH(23);
        break;
    case 8:
        if (TB == 1) {
            GEMV_DISPATCH_TB(8, 1);
        } else if (TB == 8) {
            GEMV_DISPATCH_TB(8, 8);
        } else if (TB == 16) {
            GEMV_DISPATCH_TB(8, 16);
        } else {
            GEMV_DISPATCH_TB(8, 32);
        }
        break;
    case 7:
        GEMV_TB_SWITCH(7);
        break;
    case 30:
        GEMV_TB_SWITCH(30);
        break;
    case 1:
        GEMV_TB_SWITCH(1);
        break;
    default:
        if (TB == 1) {
            GEMV_DISPATCH_TB(0, 1);
        } else if (TB == 8) {
            GEMV_DISPATCH_TB(0, 8);
        } else if (TB == 16) {
            GEMV_DISPATCH_TB(0, 16);
        } else {
            GEMV_DISPATCH_TB(0, 32);
        }
        break;
    }
}

void gemv_group_launch(queue & q, uint32_t type, const gemv_seg * segs, int n_segs, int total_rows, int TB, int nsb,
                       int n_tok_blocks) {
    gemv_group_impl(q, type, segs, n_segs, total_rows, TB, nsb, n_tok_blocks);
}
#undef GEMV_DISPATCH_TB

} // namespace si
