#include "model.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "model_arch.h"

namespace si {

namespace arch {

wt bind_tensor(const gguf_file & f, const std::string & name, uint32_t expect_type) {
    const gguf_tensor_info * ti = f.find(name);
    if (!ti) {
        throw std::runtime_error("missing tensor: " + name);
    }
    if (expect_type != 0xFFFFFFFF && ti->type != expect_type) {
        throw std::runtime_error("unexpected type for " + name);
    }
    wt t;
    t.data = ti->data;
    t.type = ti->type;
    t.K = (int32_t)ti->dims[0];
    t.N = (int32_t)ti->n_rows();
    return t;
}

const float * bind_f32(const gguf_file & f, const std::string & name) {
    const gguf_tensor_info * ti = f.find(name);
    if (!ti) {
        throw std::runtime_error("missing tensor: " + name);
    }
    if (ti->type != GGML_TYPE_F32) {
        throw std::runtime_error("expected f32 for " + name);
    }
    return (const float *)ti->data;
}

// Supported architectures.  Add one loader per model type; the loaders live in
// their own translation units (e.g. qwen35.cpp).
static const loader kLoaders[] = {
    {"qwen35", &load_qwen35},
};

const loader * find(const std::string & name) {
    for (const loader & l : kLoaders) {
        if (name == l.name) {
            return &l;
        }
    }
    return nullptr;
}

} // namespace arch

// ---------------------------------------------------------------------------
// Generic model lifecycle.  Architecture-specific hparams and tensor bindings
// are delegated to the loader selected by `general.architecture`.
void model::load(const std::string & path) {
    gguf.load(path);
    if (const std::string * ct = gguf.get_str("tokenizer.chat_template")) {
        chat_template = *ct;
    }
    const std::string * arch_p = gguf.get_str("general.architecture");
    const std::string name = arch_p ? *arch_p : "";
    const arch::loader * a = arch::find(name);
    if (!a) {
        throw std::runtime_error("unsupported architecture: " + name);
    }
    a->load(*this);
}

int model_context_length(const std::string & path) {
    gguf_file f;
    f.load(path);
    const std::string * arch = f.get_str("general.architecture");
    if (!arch) {
        return 0;
    }
    return (int)f.get_u32(*arch + ".context_length", 0);
}

void model::upload(sycl::queue & q, bool host) {
    if (host) {
        dev_weights = nullptr;
        dev_weights_size = 0;
        return;
    }
    dev_weights_size = gguf.map_size;
    dev_weights = sycl::malloc_device(dev_weights_size, q);
    if (!dev_weights) {
        throw std::runtime_error("device allocation failed");
    }
    q.memcpy(dev_weights, gguf.map_base, dev_weights_size).wait();
}

// MADV_DONTNEED over the page-aligned range covering [p, p+len) clipped to the
// mapping.  Only *resident* pages drop out of the process RSS; the mapping and
// its file contents are untouched, so this is safe even if a host reader shows
// up later (it just faults the page back in).
void model::page_out_host(const void * p, size_t len) {
    if (!gguf.map_base || !p || len == 0) {
        return;
    }
    const uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE);
    const uintptr_t base = (uintptr_t)gguf.map_base;
    const uintptr_t end = base + gguf.map_size;
    uintptr_t a = (uintptr_t)p & ~(pg - 1);
    uintptr_t b = ((uintptr_t)p + len + pg - 1) & ~(pg - 1);
    if (a < base) {
        a = base;
    }
    if (b > end) {
        b = end;
    }
    if (b > a) {
        madvise((void *)a, (size_t)(b - a), MADV_DONTNEED);
        // now that the PTEs are gone, let the kernel drop the page cache too so
        // the memory is actually freed rather than just unmapped from the
        // process (a still-mapped CPU-partition range is skipped by the kernel)
        gguf.drop_cache((size_t)(a - base), (size_t)(b - a));
    }
}

} // namespace si
