// The text model's device weight storage: the single-device whole-mmap blob
// and the SIn (DP4A) int8 copies.  Moved out of model.cpp/model_w8.cpp so
// `model` stays a description of the file (GGUF, hparams, tensor bindings)
// and never includes SYCL.  The w8t data slots themselves stay in
// model.output8 / layer_t::*8 because the plan's segments and m.output8 point
// at them; this store owns the allocations and the per-device maps.
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "weight_store.h"
#include "model.h"
#include "quant.h"
#include "w8.h"

namespace si {

struct weight_store::impl {
    void * blob = nullptr;
    size_t bytes = 0;
    const char * base = nullptr; // contiguous form: the host range the blob mirrors
    bool contiguous = false;
    std::vector<weight_span> spans; // compact form, sorted by host
    std::vector<std::unordered_map<const void *, w8t>> dev_maps;
};

weight_store::weight_store() : p_(std::make_unique<impl>()) {}
weight_store::~weight_store() = default;

// ---------------------------------------------------------------------------
// SIn repack (DP4A path).  By default the integer weight values are copied
// verbatim from the GGUF K-quant blocks (packed at their native bit width);
// only the layout and scale/min representation change.  PF_SI4=1 re-quantizes
// every tensor to 4-bit asymmetric groups instead.  See w8.h for the format.
static void build_w8_tensor(sycl::queue & q, const wt & t, w8t & out, bool host) {
    out = w8t{};
    if (t.type != 12 && t.type != 13 && t.type != 14 && !w8_requant_type(t.type)) {
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

void weight_store::upload_contiguous(const void * host_base, size_t bytes, sycl::queue & q, const char * what) {
    p_->contiguous = true;
    p_->base = (const char *)host_base;
    p_->bytes = bytes;
    p_->spans.clear();
    // malloc(0) is implementation-defined; keep the non-null sentinel the old
    // whole-blob path had so `uploaded()` means "the blob exists".
    p_->blob = sycl::malloc_device(bytes ? bytes : 1, q);
    if (!p_->blob) {
        throw std::runtime_error(std::string(what) + "device allocation failed");
    }
    if (bytes) {
        q.memcpy(p_->blob, host_base, bytes).wait();
    }
}

void weight_store::upload_spans(std::vector<weight_span> spans, size_t total_bytes, sycl::queue & q,
                                const char * what) {
    // ptr() binary-searches by host pointer; sort here so every caller is safe
    // (the off values are unchanged by the sort).
    std::sort(spans.begin(), spans.end(),
              [](const weight_span & a, const weight_span & b) { return a.host < b.host; });
    p_->contiguous = false;
    p_->base = nullptr;
    p_->bytes = total_bytes;
    p_->spans = std::move(spans);
    p_->blob = sycl::malloc_device(total_bytes ? total_bytes : 1, q);
    if (!p_->blob) {
        throw std::runtime_error(std::string(what) + "device allocation failed");
    }
    for (const weight_span & sp : p_->spans) {
        if (sp.bytes) {
            q.memcpy((char *)p_->blob + sp.off, sp.host, sp.bytes);
        }
    }
    q.wait();
}

void weight_store::clear() {
    p_->blob = nullptr;
    p_->bytes = 0;
    p_->base = nullptr;
    p_->contiguous = false;
    p_->spans.clear();
}

bool weight_store::uploaded() const {
    return p_->blob != nullptr;
}

void weight_store::upload(model & m, sycl::queue & q, bool host) {
    if (host) {
        clear();
        return;
    }
    // A split GGUF is mapped as several files inside one virtual reservation;
    // copying the whole reservation would upload its padding and needs the
    // entire model in device memory.  Large split models take the per-tensor
    // upload path instead (the multi-device/layer-map one).
    if (m.gguf.is_split()) {
        throw std::runtime_error("single-blob upload is not supported for a split GGUF: use --layer-map");
    }
    upload_contiguous(m.gguf.map_base, m.gguf.map_size, q, "");
}

void weight_store::release(model & m, sycl::queue & q) {
    free_w8_tensor(q, m.output8);
    for (auto & L : m.layers) {
        layer_t::wl wls[7];
        const int nw = L.lay_wts(wls);
        for (int i = 0; i < nw; i++) {
            free_w8_tensor(q, *wls[i].w8);
        }
    }
    release(q);
}

void weight_store::release(sycl::queue & q) {
    if (p_->blob) {
        sycl::free(p_->blob, q);
        p_->blob = nullptr;
    }
    p_->bytes = 0;
    p_->base = nullptr;
    p_->contiguous = false;
    p_->spans.clear();
}

size_t weight_store::device_bytes() const {
    return p_->bytes;
}

const void * weight_store::ptr(const void * host_ptr) const {
    if (!p_->blob) {
        return host_ptr; // CPU backend / nothing uploaded: weights stay in the host range
    }
    if (p_->contiguous) {
        return (const char *)p_->blob + ((const char *)host_ptr - p_->base);
    }
    // Compact blob: binary-search the uploaded spans (sorted by host pointer).
    const char * h = (const char *)host_ptr;
    size_t lo = 0, hi = p_->spans.size();
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if ((const char *)p_->spans[mid].host <= h) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo > 0) {
        const weight_span & s = p_->spans[lo - 1];
        const char * hs = (const char *)s.host;
        if (h >= hs && h < hs + s.bytes) {
            return (const char *)p_->blob + s.off + (h - hs);
        }
    }
    return nullptr; // a tensor the compact blob deliberately omitted
}

// Multi-device support: build the SIn copy of a single tensor on a given
// queue/device (the per-device decode path repacks only its own partition's
// layer weights).  Non-K-quant tensors (fp16/bf16) yield an empty w8t.
void weight_store::build_w8_one(sycl::queue & q, const wt & t, w8t & out) {
    build_w8_tensor(q, t, out, false);
}

void weight_store::build_w8(model & m, sycl::queue & q, bool host) {
    build_w8_tensor(q, m.output, m.output8, host);
    for (auto & L : m.layers) {
        layer_t::wl wls[7];
        const int nw = L.lay_wts(wls);
        for (int i = 0; i < nw; i++) {
            build_w8_tensor(q, *wls[i].w, *wls[i].w8, host);
        }
    }
}

void weight_store::begin_devices(int n) {
    p_->dev_maps.assign((size_t)n, {});
}

void weight_store::build_w8_device(int dev, sycl::queue & q, const wt & t) {
    if (!t.data || dev < 0) {
        return;
    }
    w8t w;
    build_w8_one(q, t, w);
    if (!w.vals) {
        return;
    }
    if ((size_t)dev >= p_->dev_maps.size()) {
        p_->dev_maps.resize((size_t)dev + 1);
    }
    p_->dev_maps[(size_t)dev][t.data] = w;
}

const w8t * weight_store::w8_for(int dev, const void * host) const {
    if (dev < 0 || (size_t)dev >= p_->dev_maps.size()) {
        return nullptr;
    }
    const auto & mp = p_->dev_maps[(size_t)dev];
    const auto it = mp.find(host);
    return it != mp.end() ? &it->second : nullptr;
}

bool weight_store::has_w8(int dev, const void * host) const {
    return w8_for(dev, host) != nullptr;
}

void weight_store::free_devices(const std::vector<sycl::queue *> & queues) {
    for (size_t d = 0; d < p_->dev_maps.size(); d++) {
        for (auto & kv : p_->dev_maps[d]) {
            if (kv.second.vals && d < queues.size() && queues[d]) {
                sycl::free(kv.second.vals, *queues[d]);
            }
            if (kv.second.meta && d < queues.size() && queues[d]) {
                sycl::free(kv.second.meta, *queues[d]);
            }
        }
    }
    p_->dev_maps.clear();
}

} // namespace si
