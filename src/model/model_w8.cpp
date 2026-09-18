// SIn (DP4A) weight copies: build and free the int8 packed copies of the
// quantized tensors.  Split out of model.cpp so the architecture loaders stay
// focused on hparams/tensor bindings.
#include <algorithm>
#include <stdexcept>
#include <vector>

#include "model.h"
#include "quant.h"
#include "w8.h"

namespace si {

// ---------------------------------------------------------------------------
// SIn repack (DP4A path).  By default the integer weight values are copied
// verbatim from the GGUF K-quant blocks (packed at their native bit width);
// only the layout and scale/min representation change.  PF_SI4=1 re-quantizes
// every tensor to 4-bit asymmetric groups instead.  See w8.h for the format.
static void build_w8_tensor(sycl::queue & q, const wt & t, w8t & out, bool host) {
    out = w8t{};
    if (t.type != 12 && t.type != 13 && t.type != 14) {
        return;
    }
    const size_t vals_bytes = w8_vals_bytes(t.type, t.K, t.N);
    // Q6_K tensors whose scales all survive fp16 exactly store only the scale
    // (2 bytes/group): min == 32*scale is derived by the kernels (see w8.h)
    const bool scale_only = w8_q6_scale_only_ok(t.type, t.data, t.K, t.N);
    const size_t meta_bytes = w8_meta_bytes(t.type, t.K, t.N, scale_only);
    const uint32_t ltype = w8_effective_type(t.type);
    out.vals = host ? sycl::malloc_host<uint8_t>(vals_bytes, q) : sycl::malloc_device<uint8_t>(vals_bytes, q);
    out.meta = meta_bytes ? (host ? reinterpret_cast<uint32_t *>(sycl::malloc_host<uint8_t>(meta_bytes, q))
                                  : reinterpret_cast<uint32_t *>(sycl::malloc_device<uint8_t>(meta_bytes, q)))
                          : nullptr;
    out.K = t.K;
    out.N = t.N;
    out.type = ltype;
    out.bits = out.type == 12 ? 4 : (out.type == 13 ? 5 : 6);
    out.meta_elem = scale_only ? 2 : 4;
    if (!out.vals || (meta_bytes && !out.meta)) {
        throw std::runtime_error("w8 allocation failed");
    }
    // repack in slabs to keep host memory bounded (the LM head is ~250 MB)
    const int slab = 4096; // rows per slab
    const size_t row_vals = w8_vals_bytes(t.type, t.K, 1);
    const size_t row_meta_b = w8_meta_bytes(t.type, t.K, 1, scale_only);
    std::vector<uint8_t> hv((size_t)slab * row_vals);
    std::vector<uint8_t> hm((size_t)slab * row_meta_b);
    const size_t row_bytes = quant_row_bytes(t.type, t.K);
    for (int r0 = 0; r0 < t.N; r0 += slab) {
        const int nr = std::min(slab, t.N - r0);
        const char * src = (const char *)t.data + (size_t)r0 * row_bytes;
        if (!w8_repack(t.type, src, t.K, nr, hv.data(), reinterpret_cast<uint32_t *>(hm.data()), scale_only)) {
            throw std::runtime_error("w8_repack failed");
        }
        q.memcpy(out.vals + (size_t)r0 * row_vals, hv.data(), (size_t)nr * row_vals);
        q.memcpy((uint8_t *)out.meta + (size_t)r0 * row_meta_b, hm.data(), (size_t)nr * row_meta_b);
        // the host staging buffers are reused by the next slab: the async copies
        // must complete before we overwrite them (otherwise the transfer races)
        q.wait();
    }
}

static void free_w8_tensor(sycl::queue & q, w8t & t) {
    if (t.vals) {
        sycl::free(t.vals, q);
    }
    if (t.meta) {
        sycl::free(t.meta, q);
    }
    t = w8t{};
}

void model::free_w8(sycl::queue & q) {
    free_w8_tensor(q, tok_embd8);
    for (auto & L : layers) {
        free_w8_tensor(q, L.ffn_gate8);
        free_w8_tensor(q, L.ffn_up8);
        free_w8_tensor(q, L.ffn_down8);
        free_w8_tensor(q, L.wqkv8);
        free_w8_tensor(q, L.wgate8);
        free_w8_tensor(q, L.ssm_out8);
        free_w8_tensor(q, L.wq8);
        free_w8_tensor(q, L.wk8);
        free_w8_tensor(q, L.wv8);
        free_w8_tensor(q, L.wo8);
    }
}

void model::build_w8(sycl::queue & q, bool host) {
    build_w8_tensor(q, tok_embd, tok_embd8, host);
    for (auto & L : layers) {
        build_w8_tensor(q, L.ffn_gate, L.ffn_gate8, host);
        build_w8_tensor(q, L.ffn_up, L.ffn_up8, host);
        build_w8_tensor(q, L.ffn_down, L.ffn_down8, host);
        if (L.recurrent) {
            build_w8_tensor(q, L.wqkv, L.wqkv8, host);
            build_w8_tensor(q, L.wgate, L.wgate8, host);
            build_w8_tensor(q, L.ssm_out, L.ssm_out8, host);
        } else {
            build_w8_tensor(q, L.wq, L.wq8, host);
            build_w8_tensor(q, L.wk, L.wk8, host);
            build_w8_tensor(q, L.wv, L.wv8, host);
            build_w8_tensor(q, L.wo, L.wo8, host);
        }
    }
}

} // namespace si
