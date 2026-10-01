// MTP (NextN) draft-head kernels.  The MTP layer itself reuses the ordinary
// full-attention kernels (rmsnorm / gemv / qk_norm_rope / attn); only the input
// preparation is new: it fuses the two RMSNorm gates and the concatenation of
// the token embedding and the previous main-model hidden state.
#include "kernels.h"
#include "kernel_utils.h"

#include <cstdio>

namespace si {

using namespace sycl;
using namespace si::kd;

void mtp_capture_launch(queue & q, const float * src, float * dst, int n_rows, int n) {
    if (n_rows <= 0 || n <= 0) {
        return;
    }
    const size_t total = (size_t)n_rows * (size_t)n;
    q.parallel_for(nd_range<1>(((total + 255) / 256) * 256, 256), [=](nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        if (i < total) {
            dst[i] = src[i];
        }
    });
}

// out[slot] = [ rmsnorm(enorm, emb(tok_slot)) ; rmsnorm(hnorm, h_prev_of(slot)) ]
// `tok_dev` (optional) supplies the token from device memory instead of
// step_info::tokens, which is what keeps the MTP draft chain on the device.
// The embedding half comes first, matching ggml_concat(e_norm, h_norm, dim=0)
// in the reference MTP graph.
void mtp_concat_launch(queue & q, const void * table, uint32_t type, size_t row_bytes, const float * enorm,
                       const float * hnorm, const float * h, const float * h_prev, const step_info * info,
                       float * out, int n_embd, float eps, const int32_t * tok_dev) {
    const int n_sb = n_embd / 256; // n_embd is a multiple of 256 for every qwen35 model
    const int tpb = info->tpb > 0 ? info->tpb : kMaxT;
    const int sb_bytes = superblock_bytes(type);
    // Fixed grid (kMaxB*kMaxT rows, one work-group per token): the live row count
    // is device-side state, so a host-computed extent would be frozen into a
    // recorded command graph.
    q.parallel_for(nd_range<1>((size_t)kMaxB * kMaxT * 256, 256), [=](nd_item<1> it) {
        const int slot = it.get_group(0);
        const int r = slot / tpb;
        const int t = slot - r * tpb;
        const int lid = it.get_local_id(0);
        if (r >= info->n_rows || !info->active[r] || t >= row_nr(info, r)) {
            return;
        }
        // tok_dev: the MTP draft chain's own token (written on the device by the
        // candidate-restricted head's argmax), so a draft step needs no host
        // round-trip; null during the prefill / the first step of a chain.
        const int tok = tok_dev ? tok_dev[slot] : info->tokens[slot];
        const char * erow = (const char *)table + (size_t)tok * row_bytes;
        const float * hrow = (slot == 0) ? h_prev + (size_t)r * n_embd : h + (size_t)(slot - 1) * n_embd;

        // Each thread owns one element of every 256-element superblock, so the
        // dequantized embedding and the hidden row both live in registers for
        // the second pass (no re-read, no SLM).
        float ev[32];
        float hv[32];
        float se = 0.f, sh = 0.f;
        for (int sb = 0; sb < n_sb; sb++) {
            ev[sb] = dequant_elem_sb(type, erow + (size_t)sb * sb_bytes, lid);
            hv[sb] = hrow[(size_t)sb * 256 + lid];
            se = sycl::fma(ev[sb], ev[sb], se);
            sh = sycl::fma(hv[sb], hv[sb], sh);
        }
        const sycl::group<1> g = it.get_group();
        se = sycl::reduce_over_group(g, se, sycl::plus<float>());
        sh = sycl::reduce_over_group(g, sh, sycl::plus<float>());
        const float re = sycl::rsqrt(se / (float)n_embd + eps);
        const float rh = sycl::rsqrt(sh / (float)n_embd + eps);
        float * orow = out + (size_t)slot * 2 * n_embd;
        for (int sb = 0; sb < n_sb; sb++) {
            const int i = sb * 256 + lid;
            orow[i] = ev[sb] * re * enorm[i];
            orow[n_embd + i] = hv[sb] * rh * hnorm[i];
        }
    });
}

} // namespace si
