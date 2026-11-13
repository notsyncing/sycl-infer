#pragma once
#include <cstddef>
#include <memory>
#include <vector>
#include <sycl/sycl.hpp>          // the store is a device object; model.h must not include SYCL
#include "w8.h"

namespace si {
struct model;
struct wt;

// One host range inside a compact device blob: `bytes` at `host`, copied to
// the blob's byte offset `off`.  Spans are sorted by `host` so `ptr()` can
// binary-search them (a tensor the compact blob deliberately omitted is a
// miss and resolves to null).
struct weight_span {
    const void * host = nullptr;
    size_t bytes = 0;
    size_t off = 0; // device byte offset
};

// The device-side weight storage for every model in the engine: the text
// model's whole-mmap blob and SIn int8 copies, the vision/audio mmproj blobs
// (whole-file or compact spans), and the DFlash f32 copies.  `model` owns only
// what the file says (GGUF, hparams, tensor bindings) and the w8t data slots
// the plan reaches through; every allocation, upload, pointer translation and
// free lives here.
class weight_store {
public:
    weight_store();
    ~weight_store();
    weight_store(const weight_store &) = delete;
    weight_store & operator=(const weight_store &) = delete;

    // Whole host range mapped 1:1 on the device (text whole-file, audio, the
    // vision fallback).  Allocates `bytes` and copies `host_base`.
    void upload_contiguous(const void * host_base, size_t bytes, sycl::queue & q, const char * what);
    // Compact set of host spans (vision compact blob, DFlash selected tensors).
    // `total_bytes` is the blob size; each span is copied to its `off`.  The
    // spans are stored sorted by host pointer.  An empty span set still
    // allocates one byte so `uploaded()` is true.
    void upload_spans(std::vector<weight_span> spans, size_t total_bytes, sycl::queue & q, const char * what);
    // Host/CPU identity: drop the blob descriptor without freeing (only valid
    // when no blob is live; text calls it for a CPU or multi-device run).
    void clear();
    bool uploaded() const;

    // Single-device text path: copy the whole mmap to one device blob (or mark
    // host/CPU).  Split GGUFs are refused exactly as today (use --layer-map).
    void upload(model & m, sycl::queue & q, bool host);
    // Build/free the single-device SIn copies into model.output8 / layer_t::*8.
    void build_w8(model & m, sycl::queue & q, bool host = false);
    void build_w8_one(sycl::queue & q, const wt & t, w8t & out);
    // Free the single-device w8 copies AND the device blob.
    void release(model & m, sycl::queue & q);
    // Free the generic blob (contiguous or spans).  Safe with no blob.
    void release(sycl::queue & q);
    size_t device_bytes() const;
    // The device pointer for a host/mmap pointer: identity when there is no
    // blob, a contiguous offset for the whole-file form, and a span search for
    // the compact form (null on a tensor the compact blob omitted).
    const void * ptr(const void * host_ptr) const;

    // Multi-device: per-device SIn copies keyed by the host tensor pointer.
    void begin_devices(int n);
    void build_w8_device(int dev, sycl::queue & q, const wt & t);
    const w8t * w8_for(int dev, const void * host) const;
    bool has_w8(int dev, const void * host) const;
    void free_devices(const std::vector<sycl::queue *> & queues);

private:
    struct impl;
    std::unique_ptr<impl> p_;
};
} // namespace si
