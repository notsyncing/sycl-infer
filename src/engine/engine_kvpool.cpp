#include "engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace si {

// byte-sized device allocation (KV pools when they are not virtual USM)
static void * dalloc_bytes(sycl::queue & q, size_t bytes) {
    void * p = sycl::malloc_device(bytes, q);
    if (!p) {
        throw std::runtime_error("device alloc failed");
    }
    return p;
}
// ---------------------------------------------------------------------------
// Dynamic KV block pool.
//
// The pool address range is reserved once (virtual USM) so every pointer the
// recorded graphs hold stays valid; physical memory is committed in extents on
// demand up to `pool_cap` blocks.  Blocks are handed out lowest-id-first, so a
// completely free top extent can be unmapped again (shrink).  If the backend
// has no virtual USM support the pool falls back to a plain fixed allocation
// of the initial size.
size_t engine::kv_block_bytes() const {
    return (size_t)m.hp.n_head_kv * kBlockSize * m.hp.head_dim * kv_elem_bytes();
}

void engine::kv_read_vec(int which, size_t elem_off, float * dst, int n) {
    const kv_dtype_t dt = kv_dtype();
    const int esz = kv_dtype_bytes(dt);
    const void * base = which ? d_vpool : d_kpool;
    if (!base || !dst || n <= 0) {
        return;
    }
    if (dt == kv_dtype_t::i8) {
        // elem_off indexes the paged element grid
        // [n_blocks][n_head_kv][kBlockSize][head_dim]; each (unit = block x kv
        // head) is [kBlockSize rows of head_dim int8][scale plane of
        // kBlockSize x head_dim/32 fp16].  Fetch the data range and the scale
        // planes of the touched units, then dequantize on the host.
        const int hd = m.hp.head_dim;
        const int nq = hd / kI8Q;
        std::vector<uint8_t> tmp((size_t)n);
        q.memcpy(tmp.data(), (const int8_t *)base + elem_off, (size_t)n).wait();
        const size_t r0 = elem_off / hd, r1 = (elem_off + (size_t)n - 1) / hd;
        std::vector<sycl::half> sc((r1 - r0 + 1) * (size_t)nq);
        const void * sbase = which ? d_vscales : d_kscales;
        for (size_t r = r0; r <= r1; r++) {
            q.memcpy(&sc[(r - r0) * nq], (const sycl::half *)sbase + r * nq, (size_t)nq * 2).wait();
        }
        for (int i = 0; i < n; i++) {
            const size_t e = elem_off + (size_t)i;
            const size_t row = e / hd;
            dst[i] = (float)(int8_t)tmp[(size_t)i] * (float)sc[(row - r0) * nq + (e % hd) / kI8Q];
        }
        return;
    }
    if (esz == 4) {
        q.memcpy((char *)dst, (const char *)base + elem_off * 4, (size_t)n * 4).wait();
        return;
    }
    std::vector<uint8_t> tmp((size_t)n * esz);
    q.memcpy((char *)tmp.data(), (const char *)base + elem_off * esz, tmp.size()).wait();
    for (int i = 0; i < n; i++) {
        dst[i] = kv_ld_host(dt, tmp.data(), (size_t)i);
    }
}

int engine::attn_layers() const {
    int n = 0;
    for (int il = 0; il < m.hp.n_layer; il++) {
        n += !m.hp.is_recr(il);
    }
    return n;
}

void engine::kv_setup(int n_attn, int initial_blocks) {
    const size_t block_bytes = kv_block_bytes();
    const size_t mb2 = (size_t)2 << 20;
    // Level Zero requires mappings of >= 2 MB to be 2 MB aligned, so the whole
    // layout (layer stride and extent starts) is kept on 2 MB boundaries.
    // With fp32 KV a block is 64 KB (32 blocks = 2 MB); a 2-byte element type
    // halves that, so the granule is derived from the block size.
    const int align = (int)std::max<size_t>(1, (mb2 + block_bytes - 1) / block_bytes);
    kv_align_blocks = align;
    if (pool_cap < align) {
        pool_cap = align;
    }
    if (pool_cap % align) {
        pool_cap = (pool_cap + align - 1) / align * align;
    }
    if (initial_blocks > pool_cap) {
        initial_blocks = pool_cap;
    }
    if (initial_blocks % align) {
        initial_blocks = (initial_blocks + align - 1) / align * align;
    }
    if (initial_blocks > pool_cap) {
        initial_blocks = pool_cap;
    }

    bool ok = false;
    try {
        const size_t gran = sx::get_mem_granularity(q.get_device(), q.get_context());
        kv_vbase = sx::reserve_virtual_mem((((size_t)pool_cap * block_bytes * n_attn) + gran - 1) / gran * gran,
                                           q.get_context());
        kv_vbase_v = sx::reserve_virtual_mem((((size_t)pool_cap * block_bytes * n_attn) + gran - 1) / gran * gran,
                                             q.get_context());
        kv_reserve_bytes = (((size_t)pool_cap * block_bytes * n_attn) + gran - 1) / gran * gran;
        ok = kv_vbase != 0 && kv_vbase_v != 0;
        kv_map2 = ok && kv_vbase % mb2 == 0 && kv_vbase_v % mb2 == 0 && ((size_t)pool_cap * block_bytes) % mb2 == 0;
    } catch (const std::exception & ex) {
        fprintf(stderr, "[kv] virtual USM unavailable (%s)\n", ex.what());
    }
    kv_virtual = ok;
    if (kv_virtual && !kv_map2) {
        // sub-2 MB mappings only need 64 KB alignment (guaranteed by the
        // reservation), so growth is limited to extents below 2 MB
        kv_map2 = false;
        if (initial_blocks > 16) {
            initial_blocks = 16;
        }
        fprintf(stderr, "[kv] reservation is not 2 MB aligned: growth uses <2 MB extents\n");
    }
    if (!kv_virtual) {
        // fixed pool fallback: no growth, no shrink
        pool_cap = initial_blocks;
        n_blocks = initial_blocks; // kv_layer_stride / reservation shrink with it
        kv_layer_stride = (size_t)n_blocks * kv_block_bytes();
    } else {
        n_blocks = pool_cap;
        kv_layer_stride = (size_t)n_blocks * kv_block_bytes();
    }
    // int8: separate fp16 scale planes, indexed like the data pool (one plane
    // per layer for K and V).  They are tiny (head_dim/32 halves per row) and
    // do not participate in the virtual-memory growth.
    if (kv_dtype() == kv_dtype_t::i8) {
        kv_scale_stride = (size_t)n_blocks * m.hp.n_head_kv * kBlockSize * (m.hp.head_dim / kI8Q) * sizeof(sycl::half);
        d_kscales = dalloc_bytes(q, kv_scale_stride * n_attn);
        d_vscales = dalloc_bytes(q, kv_scale_stride * n_attn);
        if (!d_kscales || !d_vscales) {
            throw std::runtime_error("int8 KV scale planes failed");
        }
    }
    block_used_.assign(n_blocks, 0);
    block_extent_.assign(n_blocks, -1);
    pc_block_node_.assign(n_blocks, -1);
    pool_initial = initial_blocks;
    pool_blocks = 0;
    if (!kv_virtual) {
        d_kpool = dalloc_bytes(q, (size_t)initial_blocks * block_bytes * n_attn);
        d_vpool = dalloc_bytes(q, (size_t)initial_blocks * block_bytes * n_attn);
        kv_grow(initial_blocks);
        return;
    }
    d_kpool = (void *)kv_vbase;
    d_vpool = (void *)kv_vbase_v;
    kv_initializing = true;
    const bool ok0 = kv_grow(initial_blocks);
    kv_initializing = false;
    if (!ok0) {
        throw std::runtime_error("KV pool reservation failed");
    }
    pool_print("init");
}

// commit up to `want` more blocks (bounded by the cap)
bool engine::kv_grow(int want) {
    if (want < 1 || pool_blocks >= pool_cap) {
        return false;
    }
    if (kv_map2) {
        // keep extent starts on 2 MB boundaries (see kv_setup)
        const int a = std::max(1, kv_align_blocks);
        want = (want + a - 1) / a * a;
    } else if (want > 16) {
        want = 16;
    }
    const int count = std::min(want, pool_cap - pool_blocks);
    const int first = pool_blocks;
    const size_t bytes = (size_t)count * kv_block_bytes();
    size_t map_bytes = bytes;
    try {
        const size_t gran = sx::get_mem_granularity(q.get_device(), q.get_context());
        map_bytes = ((bytes + gran - 1) / gran) * gran;
    } catch (...) {
    }
    kv_extent ex;
    ex.first = first;
    ex.count = count;
    ex.map_bytes = map_bytes;
    try {
        if (kv_virtual) {
            const size_t layer_bytes = (size_t)n_blocks * kv_block_bytes();
            const int na = attn_layers();
            ex.phys_k.reserve(na);
            ex.phys_v.reserve(na);
            for (int il = 0; il < na; il++) {
                const uintptr_t off = (uintptr_t)il * layer_bytes + (uintptr_t)first * kv_block_bytes();
                auto pk = std::make_unique<sx::physical_mem>(q, map_bytes);
                pk->map(kv_vbase + off, map_bytes, sx::address_access_mode::read_write);
                ex.phys_k.push_back(std::move(pk));
                auto pv = std::make_unique<sx::physical_mem>(q, map_bytes);
                pv->map(kv_vbase_v + off, map_bytes, sx::address_access_mode::read_write);
                ex.phys_v.push_back(std::move(pv));
            }
        }
    } catch (const std::exception & ex2) {
        // leave the extent unmapped; the blocks are simply not available
        fprintf(stderr, "[kv] grow failed (%s): pool stays at %d blocks\n", ex2.what(), pool_blocks);
        return false;
    }
    for (int b = first; b < first + count; b++) {
        free_blocks_.push(b);
        block_extent_[b] = (int)kv_extents_.size();
    }
    kv_extents_.push_back(std::move(ex));
    extent_used_.push_back(0);
    pool_blocks += count;
    pool_grows++;
    if (kv_virtual && !kv_initializing) {
        pool_print("grow");
    }
    return true;
}

void engine::kv_shrink() {
    if (!kv_virtual) {
        return;
    }
    // only memory *above* the initial pool is given back: a burst that grew the
    // pool can release it, but the historical fixed pool never churns
    while (!kv_extents_.empty() && extent_used_.back() == 0 && kv_extents_.back().first >= pool_initial) {
        kv_extent ex = std::move(kv_extents_.back());
        // the extent's blocks are all free: drop them from the free list
        std::vector<int> keep;
        keep.reserve(free_blocks_.size());
        while (!free_blocks_.empty()) {
            int b = free_blocks_.top();
            free_blocks_.pop();
            if (b < ex.first || b >= ex.first + ex.count) {
                keep.push_back(b);
            }
        }
        for (int b : keep) {
            free_blocks_.push(b);
        }
        const size_t layer_bytes = (size_t)n_blocks * kv_block_bytes();
        const int na = attn_layers();
        for (int il = 0; il < na && il < (int)ex.phys_k.size(); il++) {
            const uintptr_t off = (uintptr_t)il * layer_bytes + (uintptr_t)ex.first * kv_block_bytes();
            sx::unmap((void *)(kv_vbase + off), ex.map_bytes, q.get_context());
            sx::unmap((void *)(kv_vbase_v + off), ex.map_bytes, q.get_context());
        }
        ex.phys_k.clear();
        ex.phys_v.clear();
        for (int b = ex.first; b < ex.first + ex.count; b++) {
            block_extent_[b] = -1;
        }
        pool_blocks -= ex.count;
        kv_extents_.pop_back();
        extent_used_.pop_back();
        pool_shrinks++;
        pool_print("shrink");
    }
}

void engine::kv_release_pool() {
    if (kv_virtual) {
        const size_t layer_bytes = (size_t)n_blocks * kv_block_bytes();
        const int na = attn_layers();
        for (auto & ex : kv_extents_) {
            for (int il = 0; il < na && il < (int)ex.phys_k.size(); il++) {
                const uintptr_t off = (uintptr_t)il * layer_bytes + (uintptr_t)ex.first * kv_block_bytes();
                sx::unmap((void *)(kv_vbase + off), ex.map_bytes, q.get_context());
                sx::unmap((void *)(kv_vbase_v + off), ex.map_bytes, q.get_context());
            }
            ex.phys_k.clear();
            ex.phys_v.clear();
        }
        kv_extents_.clear();
        if (kv_vbase) {
            sx::free_virtual_mem(kv_vbase, kv_reserve_bytes, q.get_context());
        }
        if (kv_vbase_v) {
            sx::free_virtual_mem(kv_vbase_v, kv_reserve_bytes, q.get_context());
        }
        kv_vbase = kv_vbase_v = 0;
    } else {
        if (d_kpool) {
            sycl::free(d_kpool, q);
        }
        if (d_vpool) {
            sycl::free(d_vpool, q);
        }
        if (d_kscales) {
            sycl::free(d_kscales, q);
        }
        if (d_vscales) {
            sycl::free(d_vscales, q);
        }
        d_kscales = d_vscales = nullptr;
    }
    d_kpool = d_vpool = nullptr;
}

void engine::pool_print(const char * tag) const {
    const double mb = 1024.0 * 1024.0;
    const double block_mb = (double)kv_block_bytes() * attn_layers() * 2.0 / mb;
    fprintf(stderr,
            "[kv] %s: pool=%d blocks (%.0f MB), cap=%d (%.0f MB), in_use=%d, peak=%llu, "
            "grows=%llu shrinks=%llu%s\n",
            tag, pool_blocks, pool_blocks * block_mb, pool_cap, pool_cap * block_mb, pool_used_blocks(),
            (unsigned long long)pool_peak, (unsigned long long)pool_grows, (unsigned long long)pool_shrinks,
            kv_virtual ? "" : " (fixed)");
}

int engine::alloc_block() {
    for (;;) {
        if (!free_blocks_.empty()) {
            const int b = free_blocks_.top();
            free_blocks_.pop();
            if (block_used_[b]) {
                continue; // defensive: never hand out a block twice
            }
            block_used_[b] = 1;
            if (block_extent_[b] >= 0) {
                extent_used_[block_extent_[b]]++;
            }
            if ((uint64_t)pool_used_blocks() > pool_peak) {
                pool_peak = pool_used_blocks();
            }
            return b;
        }
        // out of free blocks: first give back cache nodes, then commit more
        // of the reserved range
        if (pc_enabled && pc_evict_lru()) {
            continue;
        }
        if (pool_blocks < pool_cap && kv_grow(pool_chunk)) {
            continue;
        }
        return -1;
    }
}

void engine::free_block(int b) {
    if (b < 0 || b >= (int)block_used_.size()) {
        return;
    }
    if (!block_used_[b]) {
        return; // already free (double release)
    }
    block_used_[b] = 0;
    if (block_extent_[b] >= 0) {
        extent_used_[block_extent_[b]]--;
    }
    free_blocks_.push(b);
    // a fully free top extent can go back to the system
    if (kv_virtual && !kv_extents_.empty() && block_extent_[b] == (int)kv_extents_.size() - 1) {
        kv_shrink();
    }
}

void engine::set_table(int slot, const std::vector<int> & blocks) {
    if (slot < 0 || slot >= kMaxB) {
        return;
    }
    for (size_t i = 0; i < blocks.size() && i < (size_t)max_blocks; i++) {
        h_tables[(size_t)slot * max_blocks + i] = blocks[i];
    }
    q.memcpy(d_tables + (size_t)slot * max_blocks, h_tables.data() + (size_t)slot * max_blocks, (size_t)max_blocks * 4);
}

} // namespace si
