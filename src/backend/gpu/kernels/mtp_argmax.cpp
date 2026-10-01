// Device-side row argmax for the MTP verify's accept test.
//
// Replaces the host scan of the verify's logits, which needs a host copy of
// M*n_vocab floats per speculative cycle (~7 MB at M=5, n_vocab=248320) and then
// a serial comparison on one core - measured at 33.7 ms/cycle, i.e. 24% of the
// whole MTP cycle.
//
// One work-group per row: 256 lanes scan the row with a strided, coalesced walk
// (each lane keeps a private (value,index) pair in registers), then reduce over
// SLM.  Ties resolve to the LOWEST index - both in the per-lane scan and in the
// reduction - so the result is bit-identical to the host's `if (v[i] > best)`
// scan and the emitted token cannot change.
#include "kernels.h"
#include "dp4a.h"
#include "kernel_utils.h"

#include <cstdint>
#include <limits>

namespace si {
using namespace sycl;

static inline float mb_h2f(uint16_t h) {
    sycl::half x;
    __builtin_memcpy((void *)&x, &h, sizeof(x));
    return (float)x;
}

// ---------------------------------------------------------------------------
// Candidate set for the MTP draft's LM head (see engine_mtp.cpp).
//
// The draft needs one token per step and its head GEMV reads all 248320 rows
// (794 MB of u4, 2.4 ms of the ~3.3 ms a draft step costs) only to throw
// everything but the argmax away.  The draft chain's continuation lives in a few
// hundred tokens, so the head is instead evaluated on a *candidate set* gathered
// from a distribution the target itself just produced.
//
// ids[0] is always the exact argmax of the source row, so the restricted argmax
// of that same row is bit-identical to the full scan; the candidates are the
// tokens within `margin` logits of it, which is what makes the set a good proxy
// for the *next* position.  Each of the WG threads owns a disjoint slice of the
// output (no atomics, no counter - unused slots are written as -1 and the gather
// skips them), so the whole buffer is defined by every launch.
void mtp_cand_launch(queue & q, const float * logits, int n, const int32_t * am_idx, const float * am_val, float margin,
                     int32_t * ids, int cap) {
    if (cap <= 1 || n <= 0) {
        return;
    }
    constexpr int WG = 256;
    const int per = (cap - 1 + WG - 1) / WG;
    q.parallel_for(nd_range<1>(range<1>(WG), range<1>(WG)), [=](nd_item<1> it) {
        const int lid = (int)it.get_local_id(0);
        const float thr = am_val[0] - margin;
        int32_t * my = ids + 1 + (size_t)lid * per;
        int m = 0;
        for (int i = lid; i < n; i += WG) {
            if (m < per && logits[i] > thr) {
                my[m++] = i;
            }
        }
        for (; m < per; m++) {
            my[m] = -1;
        }
        if (lid == 0) {
            ids[0] = am_idx[0];
        }
    });
}

// One candidate row per work-group: the same int8 grouped-scale dot product the
// decode GEMV computes (i8_grp_gemv_launch), for a gathered row id.  The row's
// weights are contiguous, so the read is a plain stream; the per-32-group f16
// step plane is g-major ([K/32][N]) and therefore a scattered 2-byte load per
// group, which is negligible next to the 32 bytes of weights it scales.
void mtp_gather_launch(queue & q, const int32_t * ids, int cap, const int8_t * w8, const uint16_t * wsc,
                       const int8_t * xq, const uint16_t * asa, const float * xs, int K, int N, float * vals) {
    if (cap <= 0 || K <= 0 || N <= 0) {
        return;
    }
    constexpr int WG = 256;
    const int ng = K / 32;
    q.submit([&](handler & h) {
        local_accessor<float, 1> sv((size_t)WG, h);
        h.parallel_for(nd_range<1>(range<1>((size_t)cap * WG), range<1>(WG)), [=](nd_item<1> it) {
            const int c = (int)it.get_group(0);
            const int lid = (int)it.get_local_id(0);
            const int id = ids[c];
            float acc = 0.f;
            if (id >= 0 && id < N) {
                const int8_t * wrow = w8 + (size_t)id * K;
                for (int g = lid; g < ng; g += WG) {
                    const uint4 w0 = *reinterpret_cast<const uint4 *>(wrow + (size_t)g * 32);
                    const uint4 w1 = *reinterpret_cast<const uint4 *>(wrow + (size_t)g * 32 + 16);
                    const uint4 x0 = *reinterpret_cast<const uint4 *>(xq + (size_t)g * 32);
                    const uint4 x1 = *reinterpret_cast<const uint4 *>(xq + (size_t)g * 32 + 16);
                    int32_t qd = 0;
                    qd = dp4a_s8u8((int32_t)x0.x(), w0.x() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x0.y(), w0.y() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x0.z(), w0.z() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x0.w(), w0.w() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x1.x(), w1.x() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x1.y(), w1.y() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x1.z(), w1.z() ^ 0x80808080u, qd);
                    qd = dp4a_s8u8((int32_t)x1.w(), w1.w() ^ 0x80808080u, qd);
                    qd -= 128 * (int32_t)xs[g]; // undo the XOR bias
                    acc += mb_h2f(asa[g]) * mb_h2f(wsc[(size_t)g * N + id]) * (float)qd;
                }
            }
            sv[lid] = acc;
            it.barrier(access::fence_space::local_space);
            for (int s = WG / 2; s > 0; s >>= 1) {
                if (lid < s) {
                    sv[lid] += sv[lid + s];
                }
                it.barrier(access::fence_space::local_space);
            }
            if (lid == 0) {
                vals[c] = (id >= 0 && id < N) ? sv[0] : -std::numeric_limits<float>::infinity();
            }
        });
    });
}

// argmax over the gathered candidates, ties to the lowest token id (the same
// rule the full host scan and mtp_argmax_launch use).
void mtp_gather_argmax_launch(queue & q, const float * vals, const int32_t * ids, int cap, int32_t * out_id,
                              float * out_val) {
    if (cap <= 0) {
        return;
    }
    constexpr int WG = 256;
    q.submit([&](handler & h) {
        local_accessor<float, 1> sv((size_t)WG, h);
        local_accessor<int, 1> si((size_t)WG, h);
        h.parallel_for(nd_range<1>(range<1>(WG), range<1>(WG)), [=](nd_item<1> it) {
            const int lid = (int)it.get_local_id(0);
            float best = -std::numeric_limits<float>::infinity();
            int bid = -1;
            for (int c = lid; c < cap; c += WG) {
                const float v = vals[c];
                if (v > best || (v == best && ids[c] < bid)) {
                    best = v;
                    bid = ids[c];
                }
            }
            sv[lid] = best;
            si[lid] = bid;
            it.barrier(access::fence_space::local_space);
            for (int s = WG / 2; s > 0; s >>= 1) {
                if (lid < s) {
                    const float ov = sv[lid + s];
                    const int oi = si[lid + s];
                    if (ov > sv[lid] || (ov == sv[lid] && oi < si[lid])) {
                        sv[lid] = ov;
                        si[lid] = oi;
                    }
                }
                it.barrier(access::fence_space::local_space);
            }
            if (lid == 0) {
                out_id[0] = si[0];
                if (out_val) {
                    out_val[0] = sv[0];
                }
            }
        });
    });
}

void mtp_argmax_launch(queue & q, const float * logits, int n, int32_t * out_idx, float * out_val, int M) {
    if (M <= 0 || n <= 0) {
        return;
    }
    constexpr int WG = 256;
    q.submit([&](handler & h) {
        local_accessor<float, 1> sv((size_t)WG, h);
        local_accessor<int, 1> si((size_t)WG, h);
        h.parallel_for(nd_range<1>(range<1>((size_t)M * WG), range<1>(WG)), [=](nd_item<1> it) {
            const int r = (int)it.get_group(0);
            const int lid = (int)it.get_local_id(0);
            const float * row = logits + (size_t)r * n;
            float best = -std::numeric_limits<float>::infinity();
            int bidx = 0;
            // strided walk: consecutive lanes touch consecutive addresses
            for (int i = lid; i < n; i += WG) {
                const float v = row[i];
                // strict > keeps the first (lowest) index on a tie
                if (v > best) {
                    best = v;
                    bidx = i;
                }
            }
            sv[lid] = best;
            si[lid] = bidx;
            it.barrier(access::fence_space::local_space);
            for (int s = WG / 2; s > 0; s >>= 1) {
                if (lid < s) {
                    const float ov = sv[lid + s];
                    const int oi = si[lid + s];
                    // lower index wins a tie, so the result matches the host scan
                    if (ov > sv[lid] || (ov == sv[lid] && oi < si[lid])) {
                        sv[lid] = ov;
                        si[lid] = oi;
                    }
                }
                it.barrier(access::fence_space::local_space);
            }
            if (lid == 0) {
                out_idx[r] = si[0];
                if (out_val) {
                    out_val[r] = sv[0];
                }
            }
        });
    });
}

} // namespace si
