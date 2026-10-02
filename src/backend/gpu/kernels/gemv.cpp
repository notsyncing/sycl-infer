#include "kernels.h"
#include "device/device_profile.h"
#include "kernel_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

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
        case 4: gemv_multi_kernel<QT, TB_, 1, 4, 8>(q, segs, n_segs, total_rows, n_tb); break;                         \
        case 8: gemv_multi_kernel<QT, TB_, 1, 8, 8>(q, segs, n_segs, total_rows, n_tb); break;                         \
        case 14: gemv_multi_kernel<QT, TB_, 1, 14, 8>(q, segs, n_segs, total_rows, n_tb); break;                       \
        default: gemv_multi_kernel<QT, TB_, 1, 0, 8>(q, segs, n_segs, total_rows, n_tb); break;                        \
        }                                                                                                              \
    } while (0)

// experimental: cfg = RPS*100 + SGW (SGW ignored unless listed)
static int gemv_env_cfg(const char * name) {
    const char * s = getenv(name);
    return s ? atoi(s) : 0;
}

#define GEMV_DISPATCH_CFG(QT, TB_, CFG)                                                                                \
    do {                                                                                                               \
        switch (CFG) {                                                                                                 \
        case 104: gemv_multi_kernel<QT, TB_, 1, 0, 4>(q, segs, n_segs, total_rows); break;                             \
        case 204: gemv_multi_kernel<QT, TB_, 2, 0, 4>(q, segs, n_segs, total_rows); break;                             \
        case 208: gemv_multi_kernel<QT, TB_, 2, 0, 8>(q, segs, n_segs, total_rows); break;                             \
        case 216: gemv_multi_kernel<QT, TB_, 2, 0, 16>(q, segs, n_segs, total_rows); break;                            \
        case 404: gemv_multi_kernel<QT, TB_, 4, 0, 4>(q, segs, n_segs, total_rows); break;                             \
        case 408: gemv_multi_kernel<QT, TB_, 4, 0, 8>(q, segs, n_segs, total_rows); break;                             \
        case 416: gemv_multi_kernel<QT, TB_, 4, 0, 16>(q, segs, n_segs, total_rows); break;                            \
        case 808: gemv_multi_kernel<QT, TB_, 8, 0, 8>(q, segs, n_segs, total_rows); break;                             \
        default: gemv_multi_kernel<QT, TB_, 1, 0, 8>(q, segs, n_segs, total_rows); break;                              \
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
    const int rows_per_wg = si::dev::active().shape.gemv_rows_per_wg;
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

void gemv_group_launch(queue & q, uint32_t type, const gemv_seg * segs, int n_segs, int total_rows, int TB, int nsb,
                       int n_tok_blocks) {
    const int n_tb = (TB > 1 && n_tok_blocks > 0) ? n_tok_blocks : 1;
    static const bool dbg_g = getenv("PF_DBG_GEMV") != nullptr;
    if (dbg_g) {
        // the segments live on the device, so only log the host side here
        fprintf(stderr, "[gemv] type=%u n_segs=%d total_rows=%d TB=%d nsb=%d segs=%p\n", type, n_segs, total_rows, TB,
                nsb, (const void *)segs);
    }
    static const int cfg1 = gemv_env_cfg("GEMV_CFG1");
    // single-token decode: use the vectorized-dequant kernels for Q4_K/Q5_K
    static const bool dec_vec = [] {
        const char * e = getenv("GEMV_DEC_VEC");
        return !(e && atoi(e) == 0);
    }();
    static const bool vec12 = [] {
        const char * e = getenv("GEMV_VEC12");
        return !(e && atoi(e) == 0);
    }();
    static const bool vec13 = [] {
        const char * e = getenv("GEMV_VEC13");
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
#undef GEMV_DISPATCH_TB

} // namespace si
