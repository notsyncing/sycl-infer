#include "moe_cache.h"

#include "engine.h"
#include "quant.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>


namespace si {

moe_cache_cfg moe_cache_cfg::from_env(int resident_per_layer) {
    moe_cache_cfg c;
    c.resident_per_layer = resident_per_layer;
    if (const char * v = getenv("PF_MOE_RESIDENT")) {
        c.resident_per_layer = atoi(v);
    }
    if (const char * v = getenv("PF_MOE_VRAM_MB")) {
        c.vram_mb = (size_t) atoi(v);
    }
    if (const char * v = getenv("PF_MOE_RAM_MB")) {
        c.ram_mb = (size_t) atoi(v);
        c.ram_tier = c.ram_mb > 0;
    }
    if (c.resident_per_layer > 0 && c.vram_mb == 0) {
        // no explicit budget: the caller sizes the bank from the count
        c.vram_mb = 0;
    }
    return c;
}

void moe_cache::init(const model & m, const moe_cache_cfg & cfg, int n_slots_needed) {
    cfg_ = cfg;
    n_layer_ = m.hp.n_layer;
    n_expert_ = m.hp.n_expert;
    const int rows = n_layer_ + (m.hp.n_mtp > 0 ? 1 : 0);
    freq_.assign((size_t) rows * (size_t) n_expert_, 0);
    slot_.assign(freq_.size(), -1);
    frozen_ = false;

    const int total = cfg_.host_only ? 0 : (cfg_.resident_per_layer > 0 ? cfg_.resident_per_layer : n_expert_);
    resident_ = std::min(total, n_expert_);
    if (cfg_.vram_mb > 0) {
        // a VRAM budget is authoritative: keep as many of the hottest experts as
        // one shared slot costs.  One slot is gate+up+down of one layer.
        size_t per_expert = 0;
        if (m.hp.n_layer > 0 && m.layers[0].moe.ok()) {
            const moe_t & M = m.layers[0].moe;
            per_expert = M.gate_exps.plane_bytes + M.up_exps.plane_bytes + M.down_exps.plane_bytes;
        }
        if (per_expert > 0) {
            const size_t cap = cfg_.vram_mb * 1024u * 1024u / per_expert;
            resident_ = (int) std::min<size_t>((size_t) resident_, cap);
        }
    }
    (void) n_slots_needed;
    if (active()) {
        fprintf(stderr, "[moe] expert cache: %d of %d experts resident per layer", resident_, n_expert_);
        if (cfg_.ram_tier) {
            fprintf(stderr, ", RAM tier %zu MB", cfg_.ram_mb);
        }
        fprintf(stderr, "\n");
    }
}

void moe_cache::note(int layer, int expert) {
    if (frozen_ || layer < 0 || layer >= (int) freq_.size() / n_expert_ || expert < 0 || expert >= n_expert_) {
        return;
    }
    freq_[(size_t) layer * n_expert_ + expert]++;
}

void moe_cache::slotmap(const std::vector<moe_t> & layers, std::vector<int32_t> & out) const {
    const size_t rows = freq_.size() / (size_t) n_expert_;
    out.assign(rows * (size_t) n_expert_, -1);
    if (resident_ >= n_expert_) {
        // everything resident: the identity map, which is what the resident mode
        // has always used
        for (size_t il = 0; il < rows; il++) {
            for (int e = 0; e < n_expert_; e++) {
                out[il * n_expert_ + e] = e;
            }
        }
        return;
    }
    // the hottest `resident_` experts of each layer keep a device slot; every
    // other expert is -1 and its pairs are computed on the host.  Ties break on
    // the lower index so the set is deterministic, which matters because the
    // slot map is captured into command graphs.
    for (size_t il = 0; il < rows; il++) {
        std::vector<int> order((size_t) n_expert_);
        std::iota(order.begin(), order.end(), 0);
        const uint32_t * f = &freq_[il * n_expert_];
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            if (f[a] != f[b]) {
                return f[a] > f[b];
            }
            return a < b;
        });
        for (int i = 0; i < resident_ && i < n_expert_; i++) {
            out[il * n_expert_ + order[(size_t) i]] = i;
        }
    }
    (void) layers;
}

// The ncmoe host half: for every (token, expert) pair the device skipped because
// its expert is not resident, evaluate the expert from the GGUF blocks here and
// fold it into the residual with the same weighted reduction the GPU uses.
//
// The device kernels already honour slot == -1, so this is the *only* place the
// variable part exists; the resident pairs are zeroed in the accumulator so the
// shared moe_acc_add_launch adds exactly the host contribution.  It reads the
// mmap'd blocks directly (page_out_host may have evicted them, so a miss is a
// disk fault - which is the point of the RAM tier in front of it).
void engine::moe_cpu_experts(size_t moe_il, const moe_t & M, int n_tok, int top_k, int n_pairs, int dev,
                                 const float * xbuf) {
    const hparams & hp = m.hp;
    if (moe_il >= moe_slotmap_host_.size() / (size_t) hp.n_expert) {
        return;
    }
    const int32_t * row = &moe_slotmap_host_[moe_il * (size_t) hp.n_expert];
    sycl::queue & qq = dev_queue(dev);

    std::vector<int32_t> ids((size_t) n_pairs);
    std::vector<float> wts((size_t) n_pairs);
    qq.memcpy(ids.data(), d_moe_ids, ids.size() * sizeof(int32_t)).wait();
    qq.memcpy(wts.data(), d_moe_wts, wts.size() * sizeof(float)).wait();

    static long long n_calls = 0;
    n_calls++;
    if (si::env::flag("PF_MOE_CPUDBG") && (n_calls % 40) == 0) {
        fprintf(stderr, "[moe] cpu expert pass: call %lld, il=%zu, n_tok=%d, n_pairs=%d\n", n_calls, moe_il, n_tok,
                n_pairs);
    }
    std::vector<int> cpu_pairs;
    for (int pr = 0; pr < n_pairs; pr++) {
        const int e = ids[(size_t) pr];
        if (e >= 0 && e < hp.n_expert && row[e] < 0) {
            cpu_pairs.push_back(pr);
        }
    }

    // Keep the rows the device wrote for the resident pairs: the reduction that
    // follows weights every row of this buffer, so the host half fills the gaps
    // rather than replacing it.
    std::vector<float> acc((size_t) n_pairs * (size_t) hp.n_embd);
    qq.memcpy(acc.data(), d_moe_acc, acc.size() * sizeof(float)).wait();
    if (cpu_pairs.empty()) {
        return;
    }

    if (si::env::flag("PF_MOE_CPUDBG")) {
        fprintf(stderr, "[moe] cpu expert pass: il=%zu n_tok=%d spilled %zu of %d pairs\n", moe_il, n_tok, cpu_pairs.size(),
                n_pairs);
    }
    // The activation rows the spilled pairs need.  xbuf is the MoE's *own*
    // input, which is not always d_xnorm: the speculative draft runs its layer
    // out of d_mtp_xnorm, and reading d_xnorm there silently evaluates the
    // experts against another layer's activation.
    if (!xbuf) {
        return;
    }
    std::vector<int> toks;
    for (int pr : cpu_pairs) {
        toks.push_back(pr / top_k);
    }
    std::sort(toks.begin(), toks.end());
    toks.erase(std::unique(toks.begin(), toks.end()), toks.end());
    std::vector<float> x((size_t) toks.size() * (size_t) hp.n_embd);
    for (size_t i = 0; i < toks.size(); i++) {
        qq.memcpy(x.data() + i * (size_t) hp.n_embd, xbuf + (size_t) toks[i] * hp.n_embd,
                  (size_t) hp.n_embd * sizeof(float))
            .wait();
    }

    const int F = hp.n_ff_exp;
    std::vector<float> rowbuf((size_t) hp.n_embd), g((size_t) F), u((size_t) F), h((size_t) F), y((size_t) hp.n_embd);
    const size_t rb_g = quant_row_bytes(M.gate_exps.type, M.gate_exps.K);
    const size_t rb_u = quant_row_bytes(M.up_exps.type, M.up_exps.K);
    const size_t rb_d = quant_row_bytes(M.down_exps.type, M.down_exps.K);
    for (int pr : cpu_pairs) {
        const int e = ids[(size_t) pr];
        const float * xr = x.data() + (size_t) (std::lower_bound(toks.begin(), toks.end(), pr / top_k) - toks.begin()) *
                                              (size_t) hp.n_embd;
        // Tier 1 first: if this expert's blocks are already in host RAM, the
        // host half does not have to re-fault them out of a page that
        // release_host_weight_pages() dropped.
        // Tier 1 first: if this expert's blocks are already in host RAM, the
        // host half does not have to re-fault them out of a page that
        // release_host_weight_pages() dropped.  One entry is this expert's three
        // planes packed end to end, so the base pointers differ from the mmap's
        // (which are three separate tensors at three separate file offsets).
        const size_t ng = (size_t) M.gate_exps.plane_bytes, nu = (size_t) M.up_exps.plane_bytes;
        const char * gp = nullptr, * up = nullptr, * dp = nullptr;
        if (const char * cached = moe_cache_.ram(m, M, (int) moe_il, e)) {
            gp = cached;
            up = cached + ng;
            dp = cached + ng + nu;
        } else {
            // tier 0: the GGUF mmap.  gate/up/down are three distinct tensors, so
            // each is addressed from its own base (the packed RAM entry is not).
            gp = (const char *) M.gate_exps.data + (size_t) e * ng;
            up = (const char *) M.up_exps.data + (size_t) e * nu;
            dp = (const char *) M.down_exps.data + (size_t) e * M.down_exps.plane_bytes;
            moe_cache_.ram_put(m, M, (int) moe_il, e);
        }
        for (int i = 0; i < F; i++) {
            double ag = 0, au = 0;
            dequantize_row(M.gate_exps.type, gp + (size_t) i * rb_g, rowbuf.data(), M.gate_exps.K);
            for (int k = 0; k < M.gate_exps.K; k++) {
                ag += (double) rowbuf[(size_t) k] * xr[k];
            }
            g[(size_t) i] = (float) ag;
            dequantize_row(M.up_exps.type, up + (size_t) i * rb_u, rowbuf.data(), M.up_exps.K);
            for (int k = 0; k < M.up_exps.K; k++) {
                au += (double) rowbuf[(size_t) k] * xr[k];
            }
            u[(size_t) i] = (float) au;
            h[(size_t) i] = g[(size_t) i] / (1.f + std::exp(-g[(size_t) i])) * u[(size_t) i];
        }
        for (int r = 0; r < hp.n_embd; r++) {
            double ad = 0;
            dequantize_row(M.down_exps.type, dp + (size_t) r * rb_d, rowbuf.data(), M.down_exps.K);
            for (int k = 0; k < M.down_exps.K; k++) {
                ad += (double) rowbuf[(size_t) k] * h[(size_t) k];
            }
            y[(size_t) r] = (float) ad;
        }
        float * dst = acc.data() + (size_t) pr * (size_t) hp.n_embd;
        for (int r = 0; r < hp.n_embd; r++) {
            dst[r] = y[(size_t) r];
        }
    }

    // hand the completed rows back; the caller's weighted reduction now sees a
    // full d_moe_acc (device rows + host rows) and sums it exactly once.
    qq.memcpy(d_moe_acc, acc.data(), acc.size() * sizeof(float)).wait();
}


// One calibration pass: replay the router's choices from a prefill into the
// frequency table, then freeze the resident set.  Before this runs every expert
// is resident, so the first request is always correct - it just does not save
// anything yet, exactly like a cold cache.
void engine::moe_calibrate(int dev) {
    if (!moe_cache_.active() || moe_cache_.all_resident() || moe_calib_done_) {
        return;
    }
    const hparams & hp = m.hp;
    // (per-layer ids come from moe_calib_ids_; the top-k count is already baked into them)
    for (size_t il = 0; il < moe_calib_ids_.size() && il < moe_all_.size(); il++) {
        for (int32_t e : moe_calib_ids_[il]) {
            moe_cache_.note((int) il, e);
        }
    }
    moe_calib_done_ = true;
    moe_cache_.freeze();
    // rebuild the slot table from the measured frequencies
    std::vector<int32_t> host;
    moe_cache_.slotmap(moe_all_, host);
    moe_slotmap_host_ = host;
    for (size_t d = 0; d < moe_as_.size(); d++) {
        dev_queue((int) d).memcpy(moe_as_[d].slotmap, host.data(), host.size() * sizeof(int32_t)).wait();
    }
    size_t resident = 0;
    for (int32_t v : host) {
        resident += (v >= 0) ? 1 : 0;
    }
    fprintf(stderr, "[moe] expert cache: calibration done, %zu of %zu (layer, expert) pairs resident on the device\n",
            resident, moe_all_.size() * (size_t) hp.n_expert);
}


// -- tier 1: the host RAM cache of expert blocks --------------------------------
// The ncmoe host half evaluates non-resident experts straight out of the GGUF,
// but release_host_weight_pages() madvise(DONTNEED)s those very pages once the
// weights are on their devices.  Reading them again is correct and slow (a page
// fault per row); caching the *blocks* here keeps the hot experts' bytes in RAM,
// which is what makes the CPU half viable rather than merely correct.  The
// content is the raw GGUF blocks, not dequantized floats: one expert is ~0.6 MB
// of blocks against ~25 MB of fp32 rows, and the host re-dequantizes per pair
// anyway.
static uint64_t moe_ram_key(int layer, int expert) {
    return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert;
}

const char * moe_cache::ram(const model & m, const moe_t & M, int layer, int expert) const {
    if (!cfg_.ram_tier) {
        return nullptr;
    }
    auto it = ram_idx_.find(moe_ram_key(layer, expert));
    if (it == ram_idx_.end()) {
        (ram_misses_)++;
        return nullptr;
    }
    (ram_hits_)++;
    // the entry's bytes are only valid if they still describe this model: the
    // arena is keyed by (layer, expert), so a different model with the same
    // shape would silently answer with the previous one's weights
    const ram_entry & e = *it->second;
    if (e.bytes.size() != (size_t) (M.gate_exps.plane_bytes + M.up_exps.plane_bytes + M.down_exps.plane_bytes)) {
        ram_misses_++;
        return nullptr;
    }
    (void) m;
    return (const char *) e.bytes.data();
}

void moe_cache::ram_put(const model & m, const moe_t & M, int layer, int expert) {
    if (!cfg_.ram_tier) {
        return;
    }
    const uint64_t key = moe_ram_key(layer, expert);
    auto it = ram_idx_.find(key);
    if (it != ram_idx_.end()) {
        ram_lru_.erase(it->second);
        ram_idx_.erase(it);
    }
    ram_entry e;
    e.key = key;
    const size_t ng = (size_t) M.gate_exps.plane_bytes, nu = (size_t) M.up_exps.plane_bytes,
                 nd = (size_t) M.down_exps.plane_bytes;
    e.bytes.resize(ng + nu + nd);
    const char * base = (const char *) m.gguf.map_base;
    auto copy = [&](const moe_wt & t, size_t off, size_t n, size_t plane) {
        if (!t.data || !n) {
            return;
        }
        const size_t file_off = (size_t) ((const char *) t.data - base) + plane;
        std::memcpy(e.bytes.data() + off, base + file_off, n);
    };
    // each plane is one *expert's* slice, not the tensor's base: the entry has to
    // be this expert's weights or the host half silently evaluates expert 0
    copy(M.gate_exps, 0, ng, (size_t) expert * ng);
    copy(M.up_exps, ng, nu, (size_t) expert * nu);
    copy(M.down_exps, ng + nu, nd, (size_t) expert * nd);
    ram_lru_.push_front(std::move(e));
    ram_idx_[key] = ram_lru_.begin();
    ram_bytes_ += ng + nu + nd;
    // evict from the cold end until the budget holds again
    const size_t budget = cfg_.ram_mb * 1024u * 1024u;
    while (ram_bytes_ > budget && ram_lru_.size() > 1) {
        auto last = std::prev(ram_lru_.end());
        ram_bytes_ -= last->bytes.size();
        ram_idx_.erase(last->key);
        ram_lru_.erase(last);
    }
}

} // namespace si