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
#include "kernel_utils.h"

namespace si {
using namespace sycl;

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
