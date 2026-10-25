// Qwen3.8-Flash-Next forward path.
//
// The architecture replaces every layer norm with hyper-connections, adds an
// n-gram PLE module on one layer and a sparse indexer on the full-attention
// layers.  Those do not fit the generic seg_plan (one GEMV list per layer plus
// a fixed set of epilogues), so this file walks the layer stack itself.  It
// reuses the engine's buffers, the paged KV pool and every existing GDN /
// attention / GEMV launcher.
//
// Placement: the dense trunk lives on the GPUs named by --layer-map; the routed
// expert bank (70 GiB) and the PLE table (27 GiB) stay in host RAM.  The host
// MoE half therefore runs for every layer - the same split the ncmoe path uses,
// just with no resident set at all - and the PLE module runs on the host too.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <cstring>
#include <stdexcept>

#include "engine.h"
#include "common/quant.h"
#include "backend/cpu/cpu_types.h"

namespace si {

namespace {

int q4_gemv_tb(int mode, int n_tok) {
    if (mode == 2) {
        return kMaxT; // full rows only
    }
    if (n_tok <= 1) {
        return 1;
    }
    return n_tok <= 8 ? 8 : (n_tok <= 16 ? 16 : 32);
}

} // namespace

void engine::q4_alloc() {
    const hparams & hp = m.hp;
    // One-time conversion of the routed down bank to a symmetric int8 store
    // (per-32 scale).  Q5_1's block is not a K-quant superblock and K=640 is
    // not a 256 multiple, so the K-quant int8 GEMM cannot consume it; after
    // this the grouped prefill path reads each expert's down once.
    const char * em0 = si::env::str("PF_Q4_MIRROR");
    const bool mirror_mode = (em0 && em0[0] == '1') && moe_cfg_.vram_mb == 0;
    if (hp.n_expert > 0) {
        const double c0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        size_t bytes = 0;
        for (int il = 0; il < hp.n_layer; il++) {
            layer_t & L = m.layers[(size_t)il];
            const int d = multi_dev ? layer_dev_[(size_t)il] : 0;
            const bool host_layer = !multi_dev || d < 0 || dev_kind_[(size_t)d] == 1;
            if (mirror_mode && !host_layer) {
                continue; // GPU layers read the pinned mirror; no int8 down store
            }
            const int E = L.moe.down_exps.E, N = L.moe.down_exps.N, K = L.moe.down_exps.K;
            const size_t nb = (size_t)E * N * K;
            uint8_t * d8 = (uint8_t *)std::malloc(nb);
            float * ds = (float *)std::malloc((size_t)E * N * (K / 32) * sizeof(float));
            if (!d8 || !ds) {
                throw std::runtime_error("qwen4exp: down int8 conversion allocation failed");
            }
            cpu_q4_convert_down(L.moe.down_exps.data, L.moe.down_exps.type, K, N, E, L.moe.down_exps.plane_bytes, d8,
                                ds);
            L.moe.down_i8 = d8;
            L.moe.down_sc = ds;
            L.moe.down_i8_K = K;
            L.moe.down_i8_N = N;
            L.moe.down_i8_E = E;
            q4_owned_.push_back(d8);
            q4_owned_.push_back(ds);
            bytes += nb + (size_t)E * N * (K / 32) * sizeof(float);
        }
        const double c1 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        fprintf(stderr, "[q4] down int8 store: %.1f GiB in %.1f s\n", (double)bytes / (1024.0 * 1024.0 * 1024.0),
                c1 - c0);
    }
    const int R = kMaxB * kMaxT;
    const int hc_dim = hp.hc_mult * hp.n_embd;
    const size_t nb = multi_dev ? backends_.size() : 1;
    q4b_.assign(nb, {});
    for (size_t d = 0; d < nb; d++) {
        q4_bufs & b = q4b_[d];
        auto F = [&](size_t n) { return (float *)dev_alloc_on((int)d, n * sizeof(float)); };
        b.res = F((size_t)R * hc_dim);
        b.xn = F((size_t)R * hc_dim);
        b.gate = F((size_t)R * hc_dim);
        b.mixed = F((size_t)R * hp.n_embd);
        b.block = F((size_t)R * hp.n_embd);
        b.lo = F((size_t)R * hp.hc_low_rank);
        b.inj = F((size_t)R * hp.hc_mult);
        // One slot per GEMV call in a forward (48 layers x ~16 calls), so a
        // slot is never reused while its enqueued copy is in flight.
        b.segs_cap = 4096;
        b.segs = (gemv_seg *)dev_alloc_on((int)d, (size_t)b.segs_cap * sizeof(gemv_seg));
        b.host_segs = (gemv_seg *)sycl::malloc_host((size_t)b.segs_cap * sizeof(gemv_seg), q);
        b.router_logits = F((size_t)R * hp.n_expert);
        b.router_ids = (int32_t *)dev_alloc_on((int)d, (size_t)R * hp.n_expert_used * sizeof(int32_t));
        b.router_wts = F((size_t)R * hp.n_expert_used);
        if (!b.res || !b.xn || !b.gate || !b.segs) {
            throw std::runtime_error("qwen4exp: q4 activation allocation failed");
        }
    }
    q4_host_res_ = (float *)sycl::malloc_host((size_t)R * hc_dim * sizeof(float), q);
    q4_host_mix_ = (float *)sycl::malloc_host((size_t)R * hp.n_embd * sizeof(float), q);
    q4_host_out_ = (float *)sycl::malloc_host((size_t)R * hp.n_embd * sizeof(float), q);
    q4_moe_logits_ = (float *)sycl::malloc_host((size_t)R * hp.n_expert * sizeof(float), q);
    q4_moe_wts_ = (float *)sycl::malloc_host((size_t)R * hp.n_expert_used * sizeof(float), q);
    q4_moe_acc_ = (float *)sycl::malloc_host((size_t)R * hp.n_expert_used * hp.n_embd * sizeof(float), q);
    q4_moe_sh_ = (float *)sycl::malloc_host((size_t)R * 2 * hp.n_ff * sizeof(float), q);
    q4_moe_shout_ = (float *)sycl::malloc_host((size_t)R * hp.n_embd * sizeof(float), q);
    q4_moe_ids_ = (int32_t *)sycl::malloc_host((size_t)R * hp.n_expert_used * sizeof(int32_t), q);
    if (!q4_host_res_ || !q4_moe_logits_ || !q4_moe_acc_ || !q4_moe_ids_) {
        throw std::runtime_error("qwen4exp: host scratch allocation failed");
    }
    q4_alloc_banks();
    qsa_on_ = hp.idx_kpool > 1 && hp.idx_top_k > 0 && max_seq > hp.idx_top_k && hp.idx_n_head > 0;
    if (qsa_on_) {
        qsa_max_seq_ = max_seq;
        qsa_cache_.resize((size_t)attn_layers());
        for (auto & v : qsa_cache_) {
            v.resize((size_t)state_slots_);
        }
        fprintf(stderr, "[q4] QSA: host indexer, top-%d of %d-key pools (ctx %d > %d)\n", hp.idx_top_k, hp.idx_kpool,
                max_seq, hp.idx_top_k);
    }
    if (hp.ple_layer >= 0 && engram_ram_mb_ > 0 && hp.ple_head_dim > 0) {
        const size_t row_bytes = (size_t)hp.ple_head_dim * sizeof(float) + sizeof(int32_t);
        size_t cap = (engram_ram_mb_ * 1024ull * 1024ull) / row_bytes;
        if (cap < 1024) {
            cap = 1024;
        }
        size_t p2 = 1;
        while (p2 * 2 <= cap) {
            p2 *= 2;
        }
        engram_cap_ = p2;
        engram_tag_.assign(engram_cap_, -1);
        engram_row_.assign(engram_cap_ * (size_t)hp.ple_head_dim, 0.f);
        fprintf(stderr, "[q4] engram RAM tier: %zu rows (%.0f MiB) + mmap disk tier\n", engram_cap_,
                (double)(engram_cap_ * row_bytes) / (1024.0 * 1024.0));
    }
    if (hp.ple_layer >= 0) {
        const int hist_rows = (hp.ple_conv - 1) * hp.ple_ngram;
        q4_ple_hist_ = (float *)sycl::malloc_host((size_t)state_slots_ * hist_rows * hc_dim * sizeof(float), q);
        q4_ple_nhist_ = (int32_t *)sycl::malloc_host((size_t)state_slots_ * sizeof(int32_t), q);
        q4_ple_emb_ = (float *)sycl::malloc_host((size_t)R * hp.n_embd * sizeof(float), q);
        q4_ple_key_ = (float *)sycl::malloc_host((size_t)R * hc_dim * sizeof(float), q);
        q4_ple_val_ = (float *)sycl::malloc_host((size_t)R * hp.n_embd * sizeof(float), q);
        q4_ple_gate_ = (float *)sycl::malloc_host((size_t)R * hp.hc_mult * sizeof(float), q);
        q4_ple_a_ = (float *)sycl::malloc_host((size_t)R * hc_dim * sizeof(float), q);
        q4_ple_b_ = (float *)sycl::malloc_host((size_t)R * hc_dim * sizeof(float), q);
        q4_ple_prev_ = (int32_t *)sycl::malloc_host((size_t)state_slots_ * 8 * sizeof(int32_t), q);
        std::memset(q4_ple_hist_, 0, (size_t)state_slots_ * hist_rows * hc_dim * sizeof(float));
        std::memset(q4_ple_nhist_, 0, (size_t)state_slots_ * sizeof(int32_t));
        for (int i = 0; i < state_slots_ * 8; i++) {
            q4_ple_prev_[i] = hp.ple_eos;
        }
    }
}

// The routed bank, the routers and the shared expert are read from the host
// mmap on every token.  A cold 4 KB fault per page costs ~2 us, and at 1.4 GB
// of expert traffic per token that is hundreds of ms even when the file is in
// the page cache; page_out_host deliberately leaves these pages mapped but
// says nothing about *populating* them.  Touch one byte per page once, in
// parallel, after the page release has decided which tensors are host-side.
void engine::q4_prefault() {
    if (const char * e = si::env::str("PF_Q4_PREFAULT")) {
        if (e[0] == '0' && e[1] == '\0') {
            return; // default on; PF_Q4_PREFAULT=0 disables
        }
    }
    std::vector<std::pair<const char *, size_t>> ranges;
    auto add = [&](const void * p) {
        if (p) {
            ranges.push_back({(const char *)p, 0});
        }
    };
    for (int il = 0; il < m.hp.n_layer; il++) {
        const layer_t & L = m.layers[(size_t)il];
        // Read by the host half on every step, whatever the layer's device:
        // routed experts, their router, the shared expert, and the PLE module.
        add(L.moe.gate_exps.data);
        add(L.moe.up_exps.data);
        add(L.moe.down_exps.data);
        add(L.moe.gate_inp);
        add(L.ffn_gate.data);
        add(L.ffn_up.data);
        add(L.ffn_down.data);
        if (L.ple.ok()) {
            add(L.ple.key.data);
            add(L.ple.value.data);
            add(L.ple.conv);
        }
        if (L.idx.ok()) {
            // the host QSA indexer projects from these on every step
            add(L.idx.q_proj.data);
            add(L.idx.k_proj.data);
            add(L.idx.q_norm);
            add(L.idx.k_norm);
        }
        const bool host_layer = !multi_dev || layer_dev_[(size_t)il] < 0
                                || dev_kind_[(size_t)layer_dev_[(size_t)il]] == 1;
        if (!host_layer) {
            continue; // dense trunk is on a device and already uploaded
        }
        add(L.hc_attn.norm);
        add(L.hc_attn.down.data);
        add(L.hc_attn.up.data);
        add(L.hc_attn.inject.data);
        add(L.hc_ffn.norm);
        add(L.hc_ffn.down.data);
        add(L.hc_ffn.up.data);
        add(L.hc_ffn.inject.data);
        if (L.recurrent) {
            add(L.wqkv.data);
            add(L.wgate.data);
            add(L.ssm_beta.data);
            add(L.ssm_alpha.data);
            add(L.ssm_out.data);
            add(L.ssm_a);
            add(L.ssm_dt);
            add(L.ssm_norm);
            add(L.ssm_conv1d);
        } else {
            add(L.wq.data);
            add(L.wk.data);
            add(L.wv.data);
            add(L.wo.data);
            add(L.q_norm);
            add(L.k_norm);
        }
    }
    // The tensor sizes are not stored on layer_t, so derive each range from
    // the GGUF tensor record by matching the data pointer.
    std::unordered_map<const void *, size_t> sz;
    for (const auto & t : m.gguf.tensors) {
        sz[t.data] = t.nbytes();
    }
    size_t total = 0;
    for (auto & r : ranges) {
        auto it = sz.find(r.first);
        r.second = it == sz.end() ? 0 : it->second;
        total += r.second;
    }
    const double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::atomic<size_t> next{0};
    std::atomic<unsigned long long> acc{0};
    const int nt = std::max(1, std::min(16, (int)std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int w = 0; w < nt; w++) {
        th.emplace_back([&] {
            unsigned long long a = 0;
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= ranges.size()) {
                    break;
                }
                const char * p = ranges[i].first;
                const size_t n = ranges[i].second;
                for (size_t off = 0; off < n; off += 4096) {
                    a += (unsigned char)p[off];
                }
            }
            acc.fetch_add(a);
        });
    }
    for (auto & t : th) {
        t.join();
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
    fprintf(stderr, "[q4] prefaulted %.1f GiB of host-side weights in %.1f s (%llu)\n", (double)total / (1024.0 * 1024.0 * 1024.0),
            dt, acc.load());
}

// ---------------------------------------------------------------------------
// LRU expert upload through a pinned staging ring.
//
// WHY.  A q.memcpy whose source is not USM makes this backend block the calling
// thread until the whole in-order queue drains: measured on the A770, a 1 MB
// copy behind a kernel costs 1.7 ms of host time against 0.016 ms when the
// source is pinned.  The LRU's source is the GGUF mmap, so each of an expert's
// three planes paid that rendezvous on top of a ~2 GB/s copy - PF_Q4_HOT_TIME
// prices it at 1.43 ms per expert, i.e. 225 MB and 112 ms of a decode token at
// --moe-vram-mb 20000 (the largest single item in the FFN section, larger than
// gate/up and down together).
//
// The ring keeps the ordering exactly as it was: the device copy is still
// enqueued on the layer's own in-order queue, before the kernels that read the
// bank, and the slot map copy that follows it is ordered the same way.  What
// changes is only that a plain host memcpy into pinned memory happens instead
// of a pageable device copy, so the host can go on enqueuing (measured 39 ms
// of host time for a token's 223 MB against 112 ms before).  A ring slot is
// reused only after its copy has completed, which paces the host against the
// device instead of idling it.
//
// PF_Q4_STAGE=0 restores the direct pageable copies for the A/B.
// ---------------------------------------------------------------------------
void engine::q4_upload_init(int nd, size_t max_plane) {
    static const bool stage_off = si::env::i32("PF_Q4_STAGE", 1) == 0;
    if (stage_off || nd <= 0 || max_plane == 0) {
        return;
    }
    // 32 slots is ~5 decode layers of misses at the measured 6.3 copies/layer,
    // so the host can stay that far ahead of the device without ever stalling
    // on a copy it issued for a layer that is still queued.
    q4_up_slots_ = 32;
    q4_up_stride_ = ((max_plane + 4095) / 4096) * 4096;
    const size_t bytes = (size_t)q4_up_slots_ * q4_up_stride_;
    q4_up_stage_.assign((size_t)nd, nullptr);
    q4_up_next_.assign((size_t)nd, 0);
    q4_up_ev_.assign((size_t)nd * (size_t)q4_up_slots_, sycl::event());
    q4_up_live_.assign((size_t)nd * (size_t)q4_up_slots_, 0);
    for (int d = 0; d < nd; d++) {
        if (multi_dev && d < (int)dev_kind_.size() && dev_kind_[(size_t)d] == 1) {
            continue; // a cpu partition owns no bank
        }
        char * p = (char *)sycl::malloc_host(bytes, dev_queue(d));
        if (!p) {
            fprintf(stderr, "[q4] upload staging: %.1f MB pinned allocation failed; using direct copies\n",
                    (double)bytes / (1024.0 * 1024.0));
            for (int e = 0; e < d; e++) {
                if (q4_up_stage_[(size_t)e]) {
                    sycl::free(q4_up_stage_[(size_t)e], dev_queue(e));
                }
            }
            q4_up_stage_.clear();
            q4_up_ev_.clear();
            q4_up_live_.clear();
            q4_up_slots_ = 0;
            return;
        }
        q4_up_stage_[(size_t)d] = p;
    }
    fprintf(stderr, "[q4] upload staging: %d x %.2f MB pinned per device\n", q4_up_slots_,
            (double)q4_up_stride_ / (1024.0 * 1024.0));
}

void engine::q4_upload(int dev, void * dst, const void * src, size_t n) {
    sycl::queue & qd = dev_queue(dev);
    if (q4_up_slots_ <= 0 || dev < 0 || dev >= (int)q4_up_stage_.size() || !q4_up_stage_[(size_t)dev]
        || n > q4_up_stride_) {
        qd.memcpy(dst, src, n);
        return;
    }
    int & nx = q4_up_next_[(size_t)dev];
    const int slot = nx;
    nx = (nx + 1) % q4_up_slots_;
    const size_t idx = (size_t)dev * (size_t)q4_up_slots_ + (size_t)slot;
    if (q4_up_live_[idx]) {
        const auto t0 = std::chrono::steady_clock::now();
        q4_up_ev_[idx].wait();
        q4_up_wait_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        q4_up_live_[idx] = 0;
    }
    char * st = q4_up_stage_[(size_t)dev] + (size_t)slot * q4_up_stride_;
    std::memcpy(st, src, n);
    q4_up_ev_[idx] = qd.memcpy(dst, st, n);
    q4_up_live_[idx] = 1;
}

// Move whole layers' routed banks into VRAM, up to --moe-vram-mb per device.
// A banked layer runs its entire MoE on the card (all experts resident), so it
// needs no host expert traffic and no mixed/ids round trip; the layers that do
// not fit stay on the host.  Unlike the ncmoe slot map, this is a real bank:
// the kernels read `bank + slot*plane_bytes` from device memory.
void engine::q4_alloc_banks() {
    const hparams & hp = m.hp;
    q4_bank_layer_.assign((size_t)hp.n_layer, {});
    // Strata's --stream-experts, in its simplest form: every expert lives in a
    // pinned host mirror and the GPU expert kernels read it over PCIe.  A single
    // sycl::malloc_host fails above ~2 GiB on this driver, but one allocation per
    // expert tensor (each < 2 GiB) works; 48 layers x 3 tensors = 72 GiB total
    // (measured 72 GiB pinned and 4.28 GB/s device read on the A770).
    const char * em = si::env::str("PF_Q4_MIRROR");
    // Opt-in: on this box the GPU reads a pinned host mirror at 4.28 GB/s and
    // the two cards' layer pipelines are serial, so the CPU i8 expert path is
    // faster.  PF_Q4_MIRROR=1 enables the Strata-style stream for A/B.
    q4_mirror_ = (em && em[0] == '1') && moe_cfg_.vram_mb == 0;
    if (q4_mirror_) {
        const int nd = multi_dev ? (int)backends_.size() : 1;
        std::vector<int32_t *> sm((size_t)nd, nullptr);
        std::vector<size_t> used((size_t)nd, 0);
        int nbank = 0;
        size_t total = 0;
        const char * hl = si::env::str("PF_Q4_HOT_LAYERS");
        const int hmax = hl ? std::max(0, atoi(hl)) : 1 << 30;
        for (int il = 0; il < hp.n_layer && nbank < hmax; il++) {
            const int dev = multi_dev ? layer_dev_[(size_t)il] : 0;
            if (dev < 0 || dev >= nd || (multi_dev ? dev_kind_[(size_t)dev] : (cpu_mode ? 1 : 0)) == 1) {
                continue;
            }
            layer_t & L = m.layers[(size_t)il];
            const size_t bg = (size_t)L.moe.gate_exps.E * L.moe.gate_exps.plane_bytes;
            const size_t bu = (size_t)L.moe.up_exps.E * L.moe.up_exps.plane_bytes;
            const size_t bd = (size_t)L.moe.down_exps.E * L.moe.down_exps.plane_bytes;
            if (bg >= (2ull << 30) || bu >= (2ull << 30) || bd >= (2ull << 30)) {
                fprintf(stderr, "[q4] mirror: layer %d tensor >= 2 GiB, leaving it on the host\n", il);
                continue;
            }
            if (!sm[(size_t)dev]) {
                int32_t * mdev = (int32_t *)dev_alloc_on(dev, (size_t)L.moe.gate_exps.E * sizeof(int32_t));
                std::vector<int32_t> mh((size_t)L.moe.gate_exps.E);
                for (int e = 0; e < L.moe.gate_exps.E; e++) {
                    mh[(size_t)e] = e;
                }
                dev_queue(dev).memcpy(mdev, mh.data(), mh.size() * sizeof(int32_t));
                sm[(size_t)dev] = mdev;
            }
            q4_bank_t b;
            b.g = sycl::malloc_host(bg, q);
            b.u = sycl::malloc_host(bu, q);
            b.d = sycl::malloc_host(bd + 256, q);
            if (!b.g || !b.u || !b.d) {
                if (b.g) sycl::free(b.g, q);
                if (b.u) sycl::free(b.u, q);
                if (b.d) sycl::free(b.d, q);
                fprintf(stderr, "[q4] mirror: pinned allocation failed at layer %d\n", il);
                continue;
            }
            std::memcpy(b.g, L.moe.gate_exps.data, bg);
            std::memcpy(b.u, L.moe.up_exps.data, bu);
            std::memcpy(b.d, L.moe.down_exps.data, bd);
            b.sg = L.moe.gate_exps.plane_bytes;
            b.su = L.moe.up_exps.plane_bytes;
            b.sd = L.moe.down_exps.plane_bytes;
            b.sm = sm[(size_t)dev];
            q4_bank_layer_[(size_t)il] = b;
            q4_host_usm_.push_back(b.g);
            q4_host_usm_.push_back(b.u);
            q4_host_usm_.push_back(b.d);
            used[(size_t)dev] += bg + bu + bd;
            total += bg + bu + bd;
            nbank++;
        }
        fprintf(stderr, "[q4] pinned expert mirror: %d layers, %.1f GiB (GPU reads over PCIe)\n", nbank,
                (double)total / (1024.0 * 1024.0 * 1024.0));
        if (nbank) {
            return;
        }
    }
    const size_t budget = moe_cfg_.vram_mb * 1024ull * 1024ull;
    if (budget == 0) {
        return;
    }
    // The banked GPU MoE path is not output-correct for qwen4exp: with 7-10
    // layers resident the model emits garbage while the all-host path answers
    // correctly, and it reproduces with the host router, with either expert GEMM
    // replaced by its generic fallback, and with both.  Do not ship it enabled
    // until the banked path is diffed against the host one; PF_Q4_BANK=1 keeps
    // the old allocation for that bisection.
    const char * bk = si::env::str("PF_Q4_BANK");
    if (bk && bk[0] == '0') {
        fprintf(stderr, "[q4] VRAM expert bank disabled by PF_Q4_BANK=0\n");
        return;
    }
    const int nd = multi_dev ? (int)backends_.size() : 1;
    // The per-expert hot bank is experimental (PF_Q4_HOT=1): its host spill
    // changes the arithmetic, which changes the next prefill's routing, so the
    // calibrated set does not converge (see AGENTS.md).  The whole-layer bank is
    // the default.
    // The per-expert LRU is the default: with the dedicated Q5_1 dp4a down it
    // is correct (GPU-vs-host acc cosine 0.99998) and faster than the
    // whole-layer bank (39.6 s against 45.5 on the 512-token prefill).
    // PF_Q4_BANK_LAYER=1 selects the whole-layer bank for A/B.
    const bool layer_mode = si::env::flag("PF_Q4_BANK_LAYER");
    if (!layer_mode) {
        // Per-expert hot bank: R slots in every eligible GPU layer, chosen from
        // the first prefill's routing (q4_hot_calibrate).  The budget is split
        // evenly per card and R is one uniform count, which the histogram says
        // costs almost nothing against a global oracle set (per-layer R=155
        // covers 97.5% of the pairs, global top-6200 95.8%).
        const int E = hp.n_expert;
        int nd_gpu = 0;
        std::vector<size_t> sum_e((size_t)nd, 0);
        for (int il = 0; il < hp.n_layer; il++) {
            const int dev = multi_dev ? layer_dev_[(size_t)il] : 0;
            if (dev < 0 || dev >= nd || (multi_dev ? dev_kind_[(size_t)dev] : (cpu_mode ? 1 : 0)) == 1) {
                continue;
            }
            const layer_t & L = m.layers[(size_t)il];
            if (L.moe.gate_exps.type != 12 || L.moe.up_exps.type != 12 || L.moe.down_exps.type != 7) {
                continue;
            }
            sum_e[(size_t)dev] += L.moe.gate_exps.plane_bytes + L.moe.up_exps.plane_bytes +
                                  L.moe.down_exps.plane_bytes;
        }
        for (int d = 0; d < nd; d++) {
            if (sum_e[(size_t)d] > 0) {
                nd_gpu++;
            }
        }
        if (nd_gpu == 0) {
            return;
        }
        const size_t per_dev = budget / (size_t)nd_gpu;
        int R = 512;
        for (int d = 0; d < nd; d++) {
            if (sum_e[(size_t)d] > 0) {
                R = std::min(R, (int)(per_dev / sum_e[(size_t)d]));
            }
        }
        if (R < 8) {
            fprintf(stderr, "[q4] hot bank: budget too small for 8 slots/layer, all experts stay on the host\n");
            return;
        }
        int nbank = 0;
        size_t total = 0;
        const char * hln = si::env::str("PF_Q4_HOT_LAYERS");
        const int hn = hln ? std::max(0, atoi(hln)) : 1 << 30;
        const char * hon = si::env::str("PF_Q4_HOT_ONLY");
        const int honly = hon ? atoi(hon) : -1;
        for (int il = 0; il < hp.n_layer && nbank < hn; il++) {
            if (honly >= 0 && il != honly) {
                continue;
            }
            const int dev = multi_dev ? layer_dev_[(size_t)il] : 0;
            if (dev < 0 || dev >= nd || (multi_dev ? dev_kind_[(size_t)dev] : (cpu_mode ? 1 : 0)) == 1) {
                continue;
            }
            layer_t & L = m.layers[(size_t)il];
            if (L.moe.gate_exps.type != 12 || L.moe.up_exps.type != 12 || L.moe.down_exps.type != 7) {
                continue;
            }
            q4_bank_t b;
            const size_t gb = L.moe.gate_exps.plane_bytes, ub = L.moe.up_exps.plane_bytes,
                         db = L.moe.down_exps.plane_bytes;
            b.g = dev_alloc_on(dev, (size_t)R * gb);
            b.u = dev_alloc_on(dev, (size_t)R * ub);
            b.d = dev_alloc_on(dev, (size_t)R * db + 256);
            b.sm = (int32_t *)dev_alloc_on(dev, (size_t)E * sizeof(int32_t));
            if (!b.g || !b.u || !b.d || !b.sm) {
                throw std::runtime_error("qwen4exp: hot expert bank allocation failed (lower --moe-vram-mb)");
            }
            b.sg = gb;
            b.su = ub;
            b.sd = db;
            std::vector<int32_t> mh((size_t)E, -1);
            dev_queue(dev).memcpy(b.sm, mh.data(), mh.size() * sizeof(int32_t));
            q4_bank_layer_[(size_t)il] = b;
            total += (size_t)R * (gb + ub + db);
            nbank++;
        }
        q4_hot_ = true;
        q4_hot_ready_ = true;
        q4_hot_r_ = R;
        q4_sm_host_.assign((size_t)hp.n_layer * (size_t)E, -1);
        q4_lru_age_.assign((size_t)hp.n_layer * (size_t)R, 0);
        fprintf(stderr, "[q4] hot expert bank: %d slots/layer, %d layers, %.1f GiB (calibrating on first prefill)\n", R,
                nbank, (double)total / (1024.0 * 1024.0 * 1024.0));
        size_t max_plane = 0;
        for (int il = 0; il < hp.n_layer; il++) {
            const q4_bank_t & b = q4_bank_layer_[(size_t)il];
            if (!b.g) {
                continue;
            }
            max_plane = std::max(max_plane, std::max(b.sg, std::max(b.su, b.sd)));
        }
        q4_upload_init(nd, max_plane);
        const char * sf = si::env::str("PF_Q4_SM_LOAD");
        if (sf) {
            FILE * f = fopen(sf, "rb");
            if (f) {
                q4_sm_host_.resize((size_t)hp.n_layer * (size_t)E);
                const size_t n = fread(q4_sm_host_.data(), sizeof(int32_t), q4_sm_host_.size(), f);
                fclose(f);
                if (n == q4_sm_host_.size()) {
                    for (int il = 0; il < hp.n_layer; il++) {
                        q4_bank_t & b = q4_bank_layer_[(size_t)il];
                        if (!b.g) {
                            continue;
                        }
                        const layer_t & L = m.layers[(size_t)il];
                        const int32_t * row = &q4_sm_host_[(size_t)il * (size_t)E];
                        const int dev = multi_dev ? layer_dev_[(size_t)il] : 0;
                        sycl::queue & qd = dev_queue(dev);
                        qd.memcpy(b.sm, row, (size_t)E * sizeof(int32_t));
                        for (int e = 0; e < E; e++) {
                            const int i = row[(size_t)e];
                            if (i < 0 || i >= R) {
                                continue;
                            }
                            q4_upload(dev, (char *)b.g + (size_t)i * b.sg,
                                      (const char *)L.moe.gate_exps.data + (size_t)e * b.sg, b.sg);
                            q4_upload(dev, (char *)b.u + (size_t)i * b.su,
                                      (const char *)L.moe.up_exps.data + (size_t)e * b.su, b.su);
                            q4_upload(dev, (char *)b.d + (size_t)i * b.sd,
                                      (const char *)L.moe.down_exps.data + (size_t)e * b.sd, b.sd);
                        }
                    }
                    q4_hot_ready_ = true;
                    fprintf(stderr, "[q4] hot bank: slot map loaded from %s (static calibration)\n", sf);
                } else {
                    fprintf(stderr, "[q4] slot map %s is short (%zu/%zu); ignored\n", sf, n, q4_sm_host_.size());
                }
            } else {
                fprintf(stderr, "[q4] slot map %s not readable\n", sf);
            }
        }
        const char * cf = si::env::str("PF_Q4_CAL_LOAD");
        if (cf) {
            FILE * f = fopen(cf, "rb");
            if (!f) {
                fprintf(stderr, "[q4] calibration file %s not readable; will calibrate on the first prefill\n", cf);
            } else {
                q4_cal_.resize((size_t)hp.n_layer * (size_t)E);
                const size_t n = fread(q4_cal_.data(), sizeof(uint32_t), q4_cal_.size(), f);
                fclose(f);
                if (n == q4_cal_.size()) {
                    q4_cal_pairs_ = 1;
                    q4_hot_calibrate();
                } else {
                    fprintf(stderr, "[q4] calibration file %s is short (%zu/%zu); ignored\n", cf, n, q4_cal_.size());
                    q4_cal_.clear();
                }
            }
        }
        return;
    }
    if (si::env::flag("PF_Q4_BANK_TYPES")) {
        for (int il = 0; il < hp.n_layer; il++) {
            const int dev = multi_dev ? layer_dev_[(size_t)il] : 0;
            if (dev < 0 || dev >= nd || (multi_dev ? dev_kind_[(size_t)dev] : cpu_mode) == 1) {
                continue;
            }
            const layer_t & L = m.layers[(size_t)il];
            fprintf(stderr, "[q4bank] cand il=%d dev=%d g=%u u=%u d=%u\n", il, dev, L.moe.gate_exps.type,
                    L.moe.up_exps.type, L.moe.down_exps.type);
        }
        return;
    }
    int nd_gpu = 0;
    for (int d = 0; d < nd; d++) {
        if ((multi_dev ? dev_kind_[(size_t)d] : (cpu_mode ? 1 : 0)) == 0) {
            nd_gpu++;
        }
    }
    if (nd_gpu == 0) {
        return;
    }
    const size_t per_dev = budget / (size_t)nd_gpu;
    std::vector<size_t> used((size_t)nd, 0);
    std::vector<int32_t *> sm((size_t)nd, nullptr);
    int nbank = 0;
    size_t total = 0;
    const char * bm = si::env::str("PF_Q4_BANK_MAX");
    const int bmax = bm ? std::max(0, atoi(bm)) : 1 << 30;
    const char * bs = si::env::str("PF_Q4_BANK_SKIP");
    const int bskip = bs ? std::max(0, atoi(bs)) : 0;
    for (int il = 0; il < hp.n_layer && nbank < bmax; il++) {
        if (il < bskip) {
            continue;
        }
        const int dev = multi_dev ? layer_dev_[(size_t)il] : 0;
        if (dev < 0 || dev >= nd || (multi_dev ? dev_kind_[(size_t)dev] : cpu_mode) == 1) {
            continue;
        }
        layer_t & L = m.layers[(size_t)il];
        // The GPU Q5_K gate/up (layer 2) and Q8_0 down (layers 4, 30) are not
        // output-validated; leave those layers on the host path.
        if (L.moe.gate_exps.type != 12 || L.moe.up_exps.type != 12 || L.moe.down_exps.type != 7) {
            continue;
        }
        const size_t bg = (size_t)L.moe.gate_exps.E * L.moe.gate_exps.plane_bytes;
        const size_t bu = (size_t)L.moe.up_exps.E * L.moe.up_exps.plane_bytes;
        const size_t bd = (size_t)L.moe.down_exps.E * L.moe.down_exps.plane_bytes;
        const size_t need = bg + bu + bd;
        if (used[(size_t)dev] + need > per_dev) {
            continue;
        }
        if (!sm[(size_t)dev]) {
            int32_t * mdev = (int32_t *)dev_alloc_on(dev, (size_t)L.moe.gate_exps.E * sizeof(int32_t));
            std::vector<int32_t> mh((size_t)L.moe.gate_exps.E);
            for (int e = 0; e < L.moe.gate_exps.E; e++) {
                mh[(size_t)e] = e;
            }
            dev_queue(dev).memcpy(mdev, mh.data(), mh.size() * sizeof(int32_t));
            sm[(size_t)dev] = mdev;
        }
        q4_bank_t b;
        b.g = dev_alloc_on(dev, bg);
        b.u = dev_alloc_on(dev, bu);
        b.d = dev_alloc_on(dev, bd + 256);
        if (!b.g || !b.u || !b.d) {
            throw std::runtime_error("qwen4exp: VRAM expert bank allocation failed (lower --moe-vram-mb)");
        }
        b.sg = L.moe.gate_exps.plane_bytes;
        b.su = L.moe.up_exps.plane_bytes;
        b.sd = L.moe.down_exps.plane_bytes;
        b.sm = sm[(size_t)dev];
        sycl::queue & qd = dev_queue(dev);
        qd.memcpy(b.g, L.moe.gate_exps.data, bg);
        qd.memcpy(b.u, L.moe.up_exps.data, bu);
        qd.memcpy(b.d, L.moe.down_exps.data, bd);
        q4_bank_layer_[(size_t)il] = b;
        if (si::env::flag("PF_Q4_BANKCHK")) {
            auto chk = [&](const char * nm, void * dp, const void * sp, size_t bytes) {
                std::vector<uint8_t> got(256), want(256);
                const size_t off = std::min(bytes > 256 ? bytes - 256 : 0, (size_t)5 * 4096);
                qd.memcpy(got.data(), (const char *)dp + off, 256).wait();
                memcpy(want.data(), (const char *)sp + off, 256);
                int bad = 0;
                for (int i = 0; i < 256; i++) {
                    bad += got[(size_t)i] != want[(size_t)i];
                }
                fprintf(stderr, "[q4bank] il=%d %s dev=%p host=%p bad=%d\n", il, nm, dp, sp, bad);
            };
            fprintf(stderr, "[q4bank] il=%d g=%p u=%p d=%p types g=%u u=%u d=%u\n", il, b.g, b.u, b.d,
                    L.moe.gate_exps.type, L.moe.up_exps.type, L.moe.down_exps.type);
            chk("g", b.g, L.moe.gate_exps.data, bg);
            chk("u", b.u, L.moe.up_exps.data, bu);
            chk("d", b.d, L.moe.down_exps.data, bd);
        }
        used[(size_t)dev] += need;
        total += need;
        nbank++;
    }
    if (nbank) {
        fprintf(stderr, "[q4] VRAM expert bank: %d of %d layers (%.1f MiB total, %zu MiB/device budget)\n", nbank,
                hp.n_layer, (double)total / (1024.0 * 1024.0), per_dev / (1024ull * 1024ull));
    }
}

void engine::q4_zero_state(int slot) {
    const hparams & hp = m.hp;
    if (slot < 0 || slot >= state_slots_) {
        return;
    }
    if (q4_ple_hist_) {
        const int hist_rows = (hp.ple_conv - 1) * hp.ple_ngram;
        const int hc_dim = hp.hc_mult * hp.n_embd;
        std::memset(q4_ple_hist_ + (size_t)slot * hist_rows * hc_dim, 0, (size_t)hist_rows * hc_dim * sizeof(float));
        q4_ple_nhist_[slot] = 0;
        for (int j = 0; j < 8; j++) {
            q4_ple_prev_[(size_t)slot * 8 + j] = hp.ple_eos;
        }
    }
}

void engine::q4_prefill_text(const std::vector<int> & toks, int slot, int n, int start) {
    // The CPU kernels do not bound their loops by row_nr, so a partial last row
    // in the mode-2 layout would process padding tokens (and pollute the KV and
    // the GDN state).  Full kMaxT rows go through mode 2; the < kMaxT tail goes
    // through mode 1, which is a single row of exactly `rem` tokens.
    // The per-expert bank holds R slots in a layer; a 1024-token mode-2 batch
    // selects more distinct experts than that, so prefill walks in chunks whose
    // active set fits (the LRU uploads on miss and never falls back to the host).
    const char * pfc = si::env::str("PF_Q4_PF_CHUNK");
    // A 128-token chunk keeps a layer's active set under ~180 experts; that
    // needs the 20 GiB bank (179 slots).  With the 18 GiB bank (161) a 128
    // chunk overflows the slots, so the safe default follows the slot count.
    // Only the LRU needs small chunks (its slot set must hold the chunk's
    // active experts); the whole-layer bank and the host path keep the large
    // batch that made them 3-4 s faster.
    int CH = kMaxB * kMaxT;
    if (q4_hot_) {
        CH = pfc ? std::max(kMaxT, atoi(pfc)) : (q4_hot_r_ >= 176 ? 128 : 64);
    }
    CH = std::max(kMaxT, (CH / kMaxT) * kMaxT);
    const int R = std::min(kMaxB * kMaxT, CH);
    int done = 0;
    while (done < n) {
        const int c = std::min(R, n - done);
        const int full = (c / kMaxT) * kMaxT;
        if (full > 0) {
            const int NCH = full / kMaxT;
            step_info * inf = d_info;
            inf->n_rows = NCH;
            inf->tpb = kMaxT;
            inf->n_real = kMaxT;
            inf->pc_active = 0;
            inf->mtp_dt = 0;
            inf->mtp_dry = 0;
            inf->mrope_on = 0;
            inf->img_embd = nullptr;
            for (int r = 0; r < NCH; r++) {
                inf->n_real_row[r] = kMaxT;
                inf->pos[r] = start + done + r * kMaxT;
                inf->slot[r] = slot;
                inf->active[r] = 1;
                for (int t = 0; t < kMaxT; t++) {
                    inf->tokens[r * kMaxT + t] = toks[(size_t)start + done + r * kMaxT + t];
                }
            }
            for (int r = NCH; r < kMaxB; r++) {
                inf->n_real_row[r] = 0;
                inf->active[r] = 0;
                inf->pos[r] = 0;
                inf->slot[r] = slot;
            }
            const bool last = (done + full == n);
            q4_forward(2, full, last);
            done += full;
            // Do not fall through with the rest of the prompt as a "tail": the
            // tail path is a single mode-1 row of at most kMaxT tokens, and a
            // 1038-token remainder here wrote past step_info::tokens and then
            // crashed the Level Zero driver in the next USM copy.
            continue;
        }
        const int rem = n - done;
        if (rem > 0) {
            step_info * inf = d_info;
            inf->n_rows = 1;
            inf->tpb = kMaxT;
            inf->n_real = rem;
            inf->pc_active = 0;
            inf->mtp_dt = 0;
            inf->mtp_dry = 0;
            inf->mrope_on = 0;
            inf->img_embd = nullptr;
            inf->n_real_row[0] = rem;
            inf->pos[0] = start + done;
            inf->slot[0] = slot;
            inf->active[0] = 1;
            for (int t = 0; t < rem; t++) {
                inf->tokens[t] = toks[(size_t)start + done + t];
            }
            for (int t = rem; t < kMaxT; t++) {
                inf->tokens[t] = 0;
            }
            for (int r = 1; r < kMaxB; r++) {
                inf->n_real_row[r] = 0;
                inf->active[r] = 0;
                inf->pos[r] = 0;
                inf->slot[r] = slot;
            }
            q4_forward(1, rem, true);
            done += rem;
        }
    }
}

void engine::q4_hot_calibrate() {
    const int E = m.hp.n_expert;
    const int R = q4_hot_r_;
    if (R <= 0 || q4_cal_.empty()) {
        return;
    }
    size_t bytes = 0;
    int layers = 0;
    size_t copied = 0;
    std::vector<int> order((size_t)E);
    for (int il = 0; il < m.hp.n_layer; il++) {
        q4_bank_t & b = q4_bank_layer_[(size_t)il];
        if (!b.g) {
            continue;
        }
        const layer_t & L = m.layers[(size_t)il];
        const uint32_t * c = &q4_cal_[(size_t)il * (size_t)E];
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int bb) {
            if (c[a] != c[bb]) {
                return c[a] > c[bb];
            }
            return a < bb;
        });
        std::vector<int32_t> sm((size_t)E, -1);
        for (int i = 0; i < R; i++) {
            sm[(size_t)order[(size_t)i]] = i;
        }
        const int dev = multi_dev ? layer_dev_[(size_t)il] : 0;
        sycl::queue & qd = dev_queue(dev);
        qd.memcpy(b.sm, sm.data(), (size_t)E * sizeof(int32_t));
        int32_t * old = &q4_sm_host_[(size_t)il * (size_t)E];
        for (int i = 0; i < R; i++) {
            const int e = order[(size_t)i];
            if (old[(size_t)e] == i) {
                continue; // already in this slot from the previous calibration
            }
            qd.memcpy((char *)b.g + (size_t)i * b.sg, (const char *)L.moe.gate_exps.data + (size_t)e * b.sg, b.sg);
            qd.memcpy((char *)b.u + (size_t)i * b.su, (const char *)L.moe.up_exps.data + (size_t)e * b.su, b.su);
            qd.memcpy((char *)b.d + (size_t)i * b.sd, (const char *)L.moe.down_exps.data + (size_t)e * b.sd, b.sd);
            copied++;
        }
        std::memcpy(old, sm.data(), (size_t)E * sizeof(int32_t));
        bytes += (size_t)R * (b.sg + b.su + b.sd);
        layers++;
    }
    q4_hot_ready_ = true;
    const size_t pairs = q4_cal_pairs_;
    std::fill(q4_cal_.begin(), q4_cal_.end(), 0);
    q4_cal_pairs_ = 0;
    fprintf(stderr, "[q4] hot calibration: %d slots/layer on %d layers, %zu experts copied, %.1f GiB, %zu pairs\n", R,
            layers, copied, (double)bytes / (1024.0 * 1024.0 * 1024.0), pairs);
    const char * sf = si::env::str("PF_Q4_SM_SAVE");
    if (sf) {
        FILE * f = fopen(sf, "wb");
        if (f) {
            fwrite(q4_sm_host_.data(), sizeof(int32_t), q4_sm_host_.size(), f);
            fclose(f);
            fprintf(stderr, "[q4] hot calibration: slot map saved to %s\n", sf);
        }
    }
}

void engine::q4_forward(int mode, int n_tok, bool with_head) {
    const hparams & hp = m.hp;
    const step_info * inf = d_info;
    const int hc = hp.hc_mult;
    const int hc_dim = hc * hp.n_embd;
    const int n_embd = hp.n_embd;
    const int tb = q4_gemv_tb(mode, n_tok);
    const int nblocks = (mode == 2 && n_tok > kMaxT) ? (n_tok + kMaxT - 1) / kMaxT : 1;
    const float inv_hc = 1.0f / (float)hc;
    const bool multi = multi_dev;
    for (q4_bufs & qb : q4b_) {
        qb.seg_used = 0;
    }
    auto is_cpu = [&](int d) { return multi ? (dev_kind_[(size_t)d] == 1) : cpu_mode; };
    auto DQ = [&](int d) -> sycl::queue & { return dev_queue(d); };
    auto BE = [&](int d) -> compute_backend & { return multi ? *backends_[(size_t)d] : *be; };
    auto buf = [&](int d) -> q4_bufs & { return q4b_[(size_t)d]; };
    static const bool q4trace = std::getenv("PF_Q4_QSADBG") != nullptr;
    auto to_host = [&](int d, void * dst, const void * src, size_t bytes) {
        if (q4trace) {
            fprintf(stderr, "[q4] to_host d=%d bytes=%zu\n", d, bytes);
        }
        if (is_cpu(d)) {
            std::memcpy(dst, src, bytes);
        } else {
            DQ(d).memcpy(dst, src, bytes).wait();
        }
    };
    auto from_host = [&](int d, void * dst, const void * src, size_t bytes) {
        if (q4trace) {
            fprintf(stderr, "[q4] from_host d=%d bytes=%zu\n", d, bytes);
        }
        if (is_cpu(d)) {
            std::memcpy(dst, src, bytes);
        } else {
            DQ(d).memcpy(dst, src, bytes).wait();
        }
    };
    static const bool q4dbg = si::env::flag("PF_Q4_DUMP");
    int q4_il = -1;
    auto dbg_stats = [&](const char * tag, int d, const float * p, size_t n, int streams) {
        if (!q4dbg) {
            return;
        }
        std::vector<float> h(n);
        to_host(d, h.data(), p, n * 4);
        double ss = 0;
        float mx = 0;
        for (float v : h) {
            ss += (double)v * v;
            mx = std::max(mx, std::fabs(v));
        }
        if (const char * bp = si::env::str("PF_Q4_BIN")) {
            static bool q4_bin_done = false;
            if (!q4_bin_done && std::string(tag) == "moe_routed" && q4_il == 0) {
                q4_bin_done = true;
                if (FILE * f = fopen(bp, "wb")) {
                    fwrite(h.data(), sizeof(float), n, f);
                    fclose(f);
                }
            }
        }
        fprintf(stderr, "[q4] L%-2d %-22s rms=%.6g max=%.6g first=%.6g %.6g %.6g %.6g\n", q4_il, tag,
                std::sqrt(ss / (double)n), mx, h[0], h[1], h[2], h[3]);
        if (streams == 0 && n % (size_t)n_tok == 0 && n_tok > 1) {
            const size_t per = n / (size_t)n_tok;
            for (int t = 0; t < n_tok; t++) {
                double q = 0;
                for (size_t i = 0; i < per; i++) {
                    q += (double)h[(size_t)t * per + i] * h[(size_t)t * per + i];
                }
                fprintf(stderr, "[q4] L%-2d   tok%d rms=%.6g\n", q4_il, t, std::sqrt(q / (double)per));
            }
        }
        if (streams > 0 && h.size() % (size_t)streams == 0) {
            const size_t per = h.size() / (size_t)streams;
            for (int s = 0; s < streams; s++) {
                double q = 0;
                for (size_t i = 0; i < per; i++) {
                    q += (double)h[(size_t)s * per + i] * h[(size_t)s * per + i];
                }
                fprintf(stderr, "[q4] L%-2d   stream%d rms=%.6g\n", q4_il, s, std::sqrt(q / (double)per));
            }
        }
    };
    auto tok_of = [&](int i) -> int32_t {
        // step_info::tokens is flat for a prefill (row*kMaxT + t) and one entry
        // per sequence for a decode (the engine's own convention)
        return inf->tokens[i];
    };
    auto pos_of = [&](int i) -> int {
        if (mode == 2) {
            return inf->pos[i / kMaxT] + (i % kMaxT);
        }
        return inf->pos[i];
    };
    auto slot_of = [&](int i) -> int {
        return inf->slot[mode == 2 ? i / kMaxT : i];
    };

    auto gemv = [&](int d, const wt & w, const float * x, int xs, float * out, int os, int tb_use, int nb_use) {
        if (!w.data) {
            return;
        }
        // Prefill: oneDNN's int8 matmul reads the weights once per M-token
        // batch.  The GEMV below reads them once per 32-token block, which is
        // the prefill's dominant cost for the dense trunk.  The head's single
        // row (nb_use == 0) and the odd shapes (HC up K=320, inject N=4) stay
        // on their own paths.
        if (mode != 0 && nb_use > 0 && !is_cpu(d) && (w.K % 256) == 0 && w.N >= 8) {
            dnnl_gemm * D = dnnl_for(d);
            // q4exp keeps the raw device copy as well as the oneDNN conversion,
            // so wkey() resolves to the device pointer while the conversion was
            // keyed by the host pointer.  Address the oneDNN table by the host
            // pointer explicitly.
            const void * key = w.data;
            if (D && D->has_weight(key)) {
                act_view v = D->quantize(x, nullptr, xs, 0, n_tok, w.K, false);
                if (v.valid() && D->gemm(v, key, nullptr, 1.0f, n_tok, w.K, out, os)) {
                    return;
                }
            }
        }
        gemv_seg s{};
        s.dev = d;
        s.w = wptr(d, w.data);
        s.type = w.type;
        s.K = w.K;
        s.n_rows = w.N;
        s.x = x;
        s.x_stride = xs;
        s.out = out;
        s.out_stride = os;
        s.alpha = 1.0f;
        q4_bufs & b = buf(d);
        if (b.seg_used >= b.segs_cap) {
            DQ(d).wait();
            b.seg_used = 0;
        }
        const int si = b.seg_used++;
        gemv_seg * sp = b.segs + si;
        if (is_cpu(d)) {
            std::memcpy(sp, &s, sizeof(s));
        } else {
            // Blocking copy: the async host-ring version is cheaper but crosses
            // prefill-chunk boundaries with an in-flight copy source and kills
            // the Level Zero driver inside urEnqueueUSMMemcpy on a 2-chunk
            // prompt.  56 bytes per call is not where the time goes.
            if (q4trace) {
            fprintf(stderr, "[q4] seg copy d=%d K=%d N=%d\n", d, w.K, w.N);
        }
        DQ(d).memcpy(sp, &s, sizeof(s)).wait();
        }
        BE(d).gemv_group(w.type, sp, 1, w.N, tb_use, 0, nb_use);
    };
    // host GEMV helpers (the shared expert and PLE); TB is the token count
    auto hgemv1 = [&](const wt & w, const float * x, int xs, float * out, int os) {
        cpu_gemv_seg s{};
        s.w = w.data;
        s.type = w.type;
        s.K = w.K;
        s.n_rows = w.N;
        s.x = x;
        s.x_stride = xs;
        s.out = out;
        s.out_stride = os;
        s.alpha = 1.0f;
        cpu_gemv_group(w.type, &s, 1, w.N, n_tok, 0, 0, 0);
    };
    auto hgate_up = [&](const wt & g, const wt & u, const float * x, int xs, float * sh, int sh_stride) {
        for (int which = 0; which < 2; which++) {
            const wt & w = which ? u : g;
            cpu_gemv_seg s{};
            s.w = w.data;
            s.type = w.type;
            s.K = w.K;
            s.n_rows = w.N;
            s.x = x;
            s.x_stride = xs;
            s.out = sh + (which ? g.N : 0);
            s.out_stride = sh_stride;
            s.alpha = 1.0f;
            cpu_gemv_group(w.type, &s, 1, w.N, n_tok, 0, 0, 0);
        }
    };
    auto hdown = [&](const wt & w, float * sh, int sh_stride, float * out, int os, bool act) {
        if (w.K % 256 != 0) {
            // the generic host GEMV works in 256-value superblocks and would
            // truncate this row; the shared expert's down projection is K=640
            cpu_q4_arb_gemv(w.data, w.type, sh, sh_stride, act ? sh + hp.n_ff : nullptr, out, os, n_tok, w.K, w.N);
            return;
        }
        cpu_gemv_seg s{};
        s.w = w.data;
        s.type = w.type;
        s.K = w.K;
        s.n_rows = w.N;
        s.x = sh;
        s.x_stride = sh_stride;
        s.act_up = act ? sh + hp.n_ff : nullptr;
        s.out = out;
        s.out_stride = os;
        s.alpha = 1.0f;
        cpu_gemv_group(w.type, &s, 1, w.N, n_tok, 0, 0, 0);
    };

    auto hc_mix = [&](int d, const hc_wt & H, float *& inject) {
        q4_bufs & b = buf(d);
        if (is_cpu(d)) {
            cpu_q4_hc_norm(b.res, H.norm, b.xn, nullptr, n_tok, n_embd, hc, hp.rms_eps);
        } else {
            q4_hc_norm_launch(DQ(d), b.res, wf32(d, H.norm), b.xn, n_tok, n_embd, hc, hp.rms_eps);
        }
        dbg_stats("hc_norm", d, b.xn, (size_t)n_tok * hc_dim, hc);
        gemv(d, H.down, b.xn, hc_dim, b.lo, hp.hc_low_rank, tb, nblocks);
        dbg_stats("hc_lo_pre", d, b.lo, (size_t)n_tok * hp.hc_low_rank, 0);
        if (is_cpu(d)) {
            cpu_q4_silu_scaled(b.lo, n_tok * hp.hc_low_rank, inv_hc);
        } else {
            q4_silu_scale_launch(DQ(d), b.lo, n_tok * hp.hc_low_rank, inv_hc);
        }
        // K = hc_low_rank = 320 is not a 256 multiple: use the tail-aware GEMV
        if (is_cpu(d)) {
            cpu_q4_up_gemv(H.up.data, H.up.type, b.lo, hp.hc_low_rank, b.gate, hc_dim, n_tok, hp.hc_low_rank, hc_dim);
        } else {
            q4_up_gemv_launch(DQ(d), wptr(d, H.up.data), H.up.type, b.lo, hp.hc_low_rank, b.gate, hc_dim, n_tok,
                              hp.hc_low_rank, hc_dim);
        }
        dbg_stats("hc_gate", d, b.gate, (size_t)n_tok * hc_dim, hc);
        if (is_cpu(d)) {
            cpu_gemv_seg s{};
            s.w = H.inject.data;
            s.type = H.inject.type;
            s.K = H.inject.K;
            s.n_rows = H.inject.N;
            s.x = b.xn;
            s.x_stride = hc_dim;
            s.out = b.inj;
            s.out_stride = hc;
            s.alpha = 1.0f;
            cpu_gemv_group(H.inject.type, &s, 1, H.inject.N, n_tok, 0, 0, 0);
        } else {
            q4_inject_launch(DQ(d), wptr(d, H.inject.data), H.inject.type, b.xn, hc_dim, b.inj, hc, n_tok,
                             H.inject.K, H.inject.N);
        }
        dbg_stats("hc_inject", d, b.inj, (size_t)n_tok * hc, 0);
        if (is_cpu(d)) {
            cpu_q4_hc_mix(b.xn, b.gate, b.mixed, nullptr, n_tok, n_embd, hc);
        } else {
            q4_hc_mix_launch(DQ(d), b.xn, b.gate, b.mixed, n_tok, n_embd, hc);
        }
        inject = b.inj;
    };
    auto hc_combine = [&](int d, const float * inject) {
        q4_bufs & b = buf(d);
        if (is_cpu(d)) {
            cpu_q4_hc_combine(b.res, b.block, inject, nullptr, n_tok, n_embd, hc);
        } else {
            q4_hc_combine_launch(DQ(d), b.res, b.block, inject, n_tok, n_embd, hc);
        }
    };

    static const bool q4time = si::env::flag("PF_Q4_TIME");
    auto now_ms = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    const double t_start = q4time ? now_ms() : 0.0;
    double t_moe = 0.0, t_route = 0.0, t_exp = 0.0, t_shared = 0.0, t_cp = 0.0;
    // ---- embedding + hc copies on device 0 --------------------------------
    const int d0 = 0;
    {
        q4_bufs & b = buf(d0);
        BE(d0).embed(wptr(d0, m.tok_embd.data), m.tok_embd.type, inf, b.mixed, n_embd, m.tok_embd_row_bytes);
        dbg_stats("embed", d0, b.mixed, (size_t)n_tok * n_embd, 0);
        if (is_cpu(d0)) {
            for (int i = 0; i < n_tok; i++) {
                for (int s = 0; s < hc; s++) {
                    std::memcpy(b.res + ((size_t)i * hc + s) * n_embd, b.mixed + (size_t)i * n_embd,
                                (size_t)n_embd * sizeof(float));
                }
            }
        } else {
            q4_repeat_launch(DQ(d0), b.mixed, b.res, n_tok, n_embd, hc);
        }
    }

    const bool hist_on = si::env::flag("PF_Q4_HIST");
    // A short tail prefill (the scheduler's 12-token chat suffix) is not a
    // calibration sample: its routing is not representative and it would
    // overwrite the 512-token set.  Only real prefill chunks calibrate.
    const bool hot_cal = q4_hot_ && mode != 0 && n_tok >= 64;
    auto cal_add = [&](int il) {
        if (!hot_cal) {
            return;
        }
        if (q4_cal_.empty()) {
            q4_cal_.assign((size_t)hp.n_layer * (size_t)hp.n_expert, 0);
        }
        for (int t = 0; t < n_tok; t++) {
            for (int j = 0; j < hp.n_expert_used; j++) {
                const int e = q4_moe_ids_[t * hp.n_expert_used + j];
                q4_cal_pairs_++;
                if (e >= 0 && e < hp.n_expert) {
                    q4_cal_[(size_t)il * (size_t)hp.n_expert + (size_t)e]++;
                }
            }
        }
    };
    auto hist_add = [&](int il) {
        if (!hist_on) {
            return;
        }
        std::vector<uint64_t> & H = (mode == 0) ? q4_hist_dec_ : q4_hist_pre_;
        if (H.empty()) {
            H.assign((size_t)hp.n_layer * (size_t)hp.n_expert, 0);
        }
        size_t & pairs = (mode == 0) ? q4_hist_dec_pairs_ : q4_hist_pre_pairs_;
        for (int t = 0; t < n_tok; t++) {
            for (int j = 0; j < hp.n_expert_used; j++) {
                const int e = q4_moe_ids_[t * hp.n_expert_used + j];
                pairs++;
                if (e >= 0 && e < hp.n_expert) {
                    H[(size_t)il * (size_t)hp.n_expert + e]++;
                }
            }
        }
    };

    int cur_dev = d0;
    int attn_idx = 0;
    static bool warned_qsa = false;
    for (int il = 0; il < hp.n_layer; il++) {
        q4_il = il;
        const layer_t & L = m.layers[(size_t)il];
        const int dev = multi ? layer_dev_[(size_t)il] : 0;
        if (dev != cur_dev) {
            to_host(cur_dev, q4_host_res_, buf(cur_dev).res, (size_t)n_tok * hc_dim * sizeof(float));
            from_host(dev, buf(dev).res, q4_host_res_, (size_t)n_tok * hc_dim * sizeof(float));
            cur_dev = dev;
        }
        q4_bufs & b = buf(dev);

        // ---- PLE: host gather + projections + gate + conv -----------------
        if (il == hp.ple_layer && L.ple.ok() && !si::env::flag("PF_Q4_NOPLE")) {
            std::vector<int32_t> idx((size_t)hp.ple_n_heads * n_tok);
            for (int i = 0; i < n_tok; i++) {
                const int32_t tok = tok_of(i);
                const int slot = slot_of(i);
                int64_t ctx[8];
                ctx[0] = tok;
                bool cut = false;
                for (int s = 1; s < hp.ple_ngram; s++) {
                    int32_t prev;
                    if (mode != 0 && i - s >= 0) {
                        prev = inf->tokens[i - s];
                    } else {
                        prev = q4_ple_prev_[(size_t)slot * 8 + (hp.ple_ngram - 1 - s)];
                    }
                    cut = cut || prev == hp.ple_eos;
                    ctx[s] = cut ? hp.ple_eos : prev;
                }
                for (int ng = 2; ng <= hp.ple_ngram; ng++) {
                    uint64_t mixed = (uint64_t)ctx[0] * (uint64_t)hp.ple_mult[0];
                    for (int j = 1; j < ng; j++) {
                        mixed ^= (uint64_t)ctx[j] * (uint64_t)hp.ple_mult[j];
                    }
                    const int base = (ng - 2) * hp.ple_heads_per_ngram;
                    for (int g = 0; g < hp.ple_heads_per_ngram; g++) {
                        const int hi = base + g;
                        idx[(size_t)i * hp.ple_n_heads + hi] =
                            (int32_t)(mixed % hp.ple_vocab[hi] + hp.ple_off[hi]);
                    }
                }
            }
            for (int i = 0; i < n_tok; i++) {
                for (int h = 0; h < hp.ple_n_heads; h++) {
                    const int32_t row = idx[(size_t)i * hp.ple_n_heads + h];
                    const size_t rb = quant_row_bytes(m.ple_table.type, hp.ple_head_dim);
                    const char * src = (const char *)m.ple_table.data + (size_t)row * rb;
                    float * dst = q4_ple_emb_ + (size_t)i * n_embd + (size_t)h * hp.ple_head_dim;
                    const uint32_t slot = engram_cap_ ? ((uint32_t)row & (uint32_t)(engram_cap_ - 1)) : 0;
                    if (engram_cap_ && engram_tag_[slot] == row) {
                        std::memcpy(dst, engram_row_.data() + (size_t)slot * hp.ple_head_dim,
                                    (size_t)hp.ple_head_dim * sizeof(float));
                        engram_hits_++;
                    } else {
                        dequantize_row(m.ple_table.type, src, dst, hp.ple_head_dim);
                        if (engram_cap_) {
                            std::memcpy(engram_row_.data() + (size_t)slot * hp.ple_head_dim, dst,
                                        (size_t)hp.ple_head_dim * sizeof(float));
                            engram_tag_[slot] = row;
                        }
                        engram_misses_++;
                    }
                }
            }
            dbg_stats("ple_emb", dev, q4_ple_emb_, (size_t)n_tok * n_embd, 0);
            hgemv1(L.ple.key, q4_ple_emb_, n_embd, q4_ple_key_, hc_dim);
            hgemv1(L.ple.value, q4_ple_emb_, n_embd, q4_ple_val_, n_embd);
            to_host(dev, q4_host_res_, b.res, (size_t)n_tok * hc_dim * sizeof(float));
            cpu_q4_ple_gate(q4_ple_key_, q4_host_res_, L.ple.norm_key, L.ple.norm_query, q4_ple_gate_, nullptr, n_tok,
                            n_embd, hc, hp.rms_eps);
            dbg_stats("ple_gate", dev, q4_ple_gate_, (size_t)n_tok * hc, 0);
            std::vector<int32_t> slots((size_t)n_tok), poss((size_t)n_tok);
            for (int i = 0; i < n_tok; i++) {
                slots[(size_t)i] = slot_of(i);
                poss[(size_t)i] = pos_of(i);
            }
            cpu_q4_ple_apply(q4_ple_val_, q4_ple_gate_, L.ple.norm_conv, L.ple.conv, q4_host_res_, q4_ple_a_,
                             q4_ple_b_, q4_ple_hist_, q4_ple_nhist_, nullptr, slots.data(), poss.data(), n_tok, n_embd,
                             hc, hp.ple_ngram, hp.ple_conv, hp.rms_eps);
            dbg_stats("ple_gated", dev, q4_ple_a_, (size_t)n_tok * hc_dim, 0);
            from_host(dev, b.res, q4_host_res_, (size_t)n_tok * hc_dim * sizeof(float));
            // advance the per-slot token history
            for (int i = 0; i < n_tok; i++) {
                const int slot = slot_of(i);
                int32_t * p = q4_ple_prev_ + (size_t)slot * 8;
                for (int j = 0; j + 1 < hp.ple_ngram; j++) {
                    p[j] = p[j + 1];
                }
                p[hp.ple_ngram - 2] = tok_of(i);
            }
        }

        dbg_stats("res_in", dev, b.res, (size_t)n_tok * hc_dim, hc);
        // ---- token mixer ---------------------------------------------------
        float * inj1 = nullptr;
        hc_mix(dev, L.hc_attn, inj1);
        dbg_stats("hc_mixed_attn", dev, b.mixed, (size_t)n_tok * n_embd, 0);
        compute_backend & be = BE(dev);
        if (L.recurrent) {
            float * QKV = multi ? as_[(size_t)dev].qkv : d_qkv;
            float * Z = multi ? as_[(size_t)dev].z : d_z;
            float * BETA = multi ? as_[(size_t)dev].beta : d_beta;
            float * ALPHA = multi ? as_[(size_t)dev].alpha : d_alpha;
            float * CONVOUT = multi ? as_[(size_t)dev].conv_out : d_conv_out;
            float * ATTNPRE = multi ? as_[(size_t)dev].attn_pre : d_attn_pre;
            float * ATTMERGED = multi ? as_[(size_t)dev].attn_merged : d_attn_merged;
            gemv(dev, L.wqkv, b.mixed, n_embd, QKV, hp.qkv_dim(), tb, nblocks);
            gemv(dev, L.wgate, b.mixed, n_embd, Z, hp.d_inner, tb, nblocks);
            gemv(dev, L.ssm_beta, b.mixed, n_embd, BETA, hp.dt_rank, tb, nblocks);
            gemv(dev, L.ssm_alpha, b.mixed, n_embd, ALPHA, hp.dt_rank, tb, nblocks);
            dbg_stats("qkv", dev, QKV, (size_t)n_tok * hp.qkv_dim(), 0);
            dbg_stats("z", dev, Z, (size_t)n_tok * hp.d_inner, 0);
            const int gi = m.gdn_layer_index[(size_t)il];
            const int gl = multi ? layer_gdn_local_[(size_t)il] : gi;
            const size_t conv_per = (size_t)(hp.conv_k - 1) * hp.qkv_dim();
            const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state;
            float * cs = (multi ? as_[(size_t)dev].conv_state : d_conv_state) + (size_t)gl * state_slots_ * conv_per;
            float * gs = (multi ? as_[(size_t)dev].gdn_state : d_gdn_state) + (size_t)gl * state_slots_ * gdn_per;
            if (mode == 2) {
                const int NCH = (n_tok + kMaxT - 1) / kMaxT;
                be.conv_l2(QKV, cs, wf32(dev, L.ssm_conv1d), CONVOUT, inf, hp.qkv_dim(), hp.conv_k, hp.d_state,
                           hp.n_group, hp.rms_eps, NCH, kMaxT, 0, kMaxT, true);
                be.conv_state_update(QKV, cs, inf, hp.qkv_dim(), hp.conv_k, NCH, 0, kMaxT, kMaxT, true, {});
                be.gdn(CONVOUT, ALPHA, wf32(dev, L.ssm_dt), wf32(dev, L.ssm_a), BETA, gs, ATTNPRE, inf, hp.d_state,
                       hp.n_group, hp.dt_rank, hp.qkv_dim(), 1.0f / std::sqrt((float)hp.d_state), state_slots_, 1, 0,
                       n_tok, n_tok, {});
            } else {
                const int nrows = inf->n_rows;
                const int nreal_call = (mode == 1) ? n_tok : 1;
                be.conv_l2(QKV, cs, wf32(dev, L.ssm_conv1d), CONVOUT, inf, hp.qkv_dim(), hp.conv_k, hp.d_state,
                           hp.n_group, hp.rms_eps, nrows, nreal_call, 0, -1, false);
                be.conv_state_update(QKV, cs, inf, hp.qkv_dim(), hp.conv_k, nrows, 0, -1, nreal_call, false, {});
                be.gdn(CONVOUT, ALPHA, wf32(dev, L.ssm_dt), wf32(dev, L.ssm_a), BETA, gs, ATTNPRE, inf, hp.d_state,
                       hp.n_group, hp.dt_rank, hp.qkv_dim(), 1.0f / std::sqrt((float)hp.d_state), state_slots_, nrows,
                       0, -1, nreal_call, {});
            }
            dbg_stats("conv_out", dev, CONVOUT, (size_t)n_tok * hp.qkv_dim(), 0);
            dbg_stats("attn_pre_gdn", dev, ATTNPRE, (size_t)n_tok * hp.d_inner, 0);
            be.gated_norm(ATTNPRE, Z, wf32(dev, L.ssm_norm), ATTMERGED, inf, hp.dt_rank, hp.d_state, hp.rms_eps,
                          inf->n_rows, mode == 2 ? kMaxT : ((mode == 1) ? n_tok : 1), 0, true);
            dbg_stats("gdn_normed", dev, ATTMERGED, (size_t)n_tok * hp.d_inner, 0);
            gemv(dev, L.ssm_out, ATTMERGED, hp.d_inner, b.block, n_embd, tb, nblocks);
        } else {
            float * QBUF = multi ? as_[(size_t)dev].qbuf : d_qbuf;
            float * KBUF = multi ? as_[(size_t)dev].kbuf : d_kbuf;
            float * VBUF = multi ? as_[(size_t)dev].vbuf : d_vbuf;
            float * ATTN_OUT = multi ? as_[(size_t)dev].attn_out : d_attn_out;
            float * PART = multi ? as_[(size_t)dev].partials : d_partials;
            float * PARTD = multi ? as_[(size_t)dev].partials_dec : d_partials_dec;
            gemv(dev, L.wq, b.mixed, n_embd, QBUF, hp.n_head * 2 * hp.head_dim, tb, nblocks);
            gemv(dev, L.wk, b.mixed, n_embd, KBUF, hp.n_head_kv * hp.head_dim, tb, nblocks);
            gemv(dev, L.wv, b.mixed, n_embd, VBUF, hp.n_head_kv * hp.head_dim, tb, nblocks);
            dbg_stats("Qraw", dev, QBUF, (size_t)n_tok * hp.n_head * 2 * hp.head_dim, 0);
            dbg_stats("Kraw", dev, KBUF, (size_t)n_tok * hp.n_head_kv * hp.head_dim, 0);
            dbg_stats("Vraw", dev, VBUF, (size_t)n_tok * hp.n_head_kv * hp.head_dim, 0);
            void * kp = nullptr;
            void * vp = nullptr;
            const void * ksc = nullptr;
            const void * vsc = nullptr;
            kv_layer_ptrs(attn_idx, (const char *&)kp, (const char *&)vp, (const char *&)ksc, (const char *&)vsc);
            const int nreal_call = (mode == 2) ? kMaxT : ((mode == 1) ? n_tok : 1);
            be.qk_norm_rope(QBUF, KBUF, VBUF, wf32(dev, L.q_norm), wf32(dev, L.k_norm), kp, vp, d_tables, inf,
                            hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_rot, hp.rope_base, rope_freqs_for(dev),
                            d_rope_mscale_, hp.rms_eps, max_blocks, inf->n_rows, nreal_call, ksc, vsc);
            dbg_stats("Qrope", dev, QBUF, (size_t)n_tok * hp.n_head * 2 * hp.head_dim, 0);
            dbg_stats("Krope", dev, KBUF, (size_t)n_tok * hp.n_head_kv * hp.head_dim, 0);
            const bool qsa_done = qsa_on_ && qsa_attention(attn_idx, L, dev, n_tok, inf, b.mixed, QBUF, KBUF, VBUF,
                                                           ATTN_OUT);
            if (qsa_done) {
                gemv(dev, L.wo, ATTN_OUT, hp.n_head * hp.head_dim, b.block, n_embd, tb, nblocks);
            } else {
            int nsp = 1;
            float * part = PART;
            if (mode == 0) {
                nsp = std::max(1, dec_splits);
                part = PARTD;
            } else {
                int max_nkv = 0;
                for (int r = 0; r < inf->n_rows; r++) {
                    max_nkv = std::max(max_nkv, inf->pos[r] + nreal_call);
                }
                if (max_nkv > hp.idx_top_k && !warned_qsa && hp.idx_kpool > 1) {
                    warned_qsa = true;
                    fprintf(stderr,
                            "[q4] WARNING: context %d exceeds the QSA budget %d; dense attention is used and the "
                            "result will diverge from the reference at this depth.\n",
                            max_nkv, hp.idx_top_k);
                }
                nsp = (max_nkv + 511) / 512;
                nsp = std::min(std::max(nsp, 1), std::max(1, n_splits));
            }
            be.attn(QBUF, QBUF, kp, vp, part, d_tables, hp.n_head, hp.n_head_kv, hp.head_dim, nsp, inf, hp.attn_scale,
                    max_blocks, inf->n_rows, nreal_call, nullptr, -1, ksc, vsc);
            be.attn_combine(part, QBUF, ATTN_OUT, inf, hp.n_head, hp.head_dim, nsp, inf->n_rows, nreal_call);
            dbg_stats("attn_gated", dev, ATTN_OUT, (size_t)n_tok * hp.n_head * hp.head_dim, 0);
            gemv(dev, L.wo, ATTN_OUT, hp.n_head * hp.head_dim, b.block, n_embd, tb, nblocks);
            }
            attn_idx++;
        }
        dbg_stats("block_attn", dev, b.block, (size_t)n_tok * n_embd, 0);
        hc_combine(dev, inj1);
        dbg_stats("hc_combine1", dev, b.res, (size_t)n_tok * hc_dim, hc);

        // ---- FFN: routed bank + shared expert on the host ------------------
        const double tm0 = q4time ? now_ms() : 0.0;
        float * inj2 = nullptr;
        hc_mix(dev, L.hc_ffn, inj2);
        dbg_stats("hc_mixed_ffn", dev, b.mixed, (size_t)n_tok * n_embd, 0);
        const double tc0 = q4time ? now_ms() : 0.0;
        const int k = hp.n_expert_used;
        const q4_bank_t * bank = nullptr;
        if (!is_cpu(dev) && il < (int)q4_bank_layer_.size() && q4_bank_layer_[(size_t)il].g
            && (!q4_hot_ || q4_hot_ready_) && L.moe.gate_exps.type == 12 && L.moe.up_exps.type == 12
            && L.moe.down_exps.type == 7) {
            bank = &q4_bank_layer_[(size_t)il];
        }
        if (bank) {
            // Whole layer resident: router, CSR, both expert GEMMs and the
            // reduction all run on the card; the mixed/ids/wts round trip is gone.
            moe_act & A = moe_as_[(size_t)dev];
            sycl::queue & qd = DQ(dev);
            const int np = n_tok * k;
            const bool htt = si::env::flag("PF_Q4_HOT_TIME");
            double thl = htt ? now_ms() : 0.0;
            const double th0 = thl;
            const double q4_up_wait_snap = q4_up_wait_ms_;
            int upmiss = 0;
            size_t upbytes = 0;
            if (htt && mode == 0) {
                // Drain the attention/GDN work enqueued before this section so
                // the per-stage numbers below are this layer's FFN and not the
                // token mixer's queue behind it.  Diagnostic only.
                qd.wait();
                fprintf(stderr, "[q4hot] il=%d pre=%.1f", il, now_ms() - thl);
                thl = now_ms();
            }
            double ts0 = 0.0;
            if (si::env::flag("PF_Q4_HOSTROUTER")) {
                to_host(dev, q4_host_mix_, b.mixed, (size_t)n_tok * n_embd * sizeof(float));
                cpu_moe_route(q4_host_mix_, n_embd, L.moe.gate_inp, L.moe.gate_inp_type, n_tok, n_embd, hp.n_expert, k,
                              q4_moe_logits_, q4_moe_ids_, q4_moe_wts_, true);
                qd.memcpy(A.ids, q4_moe_ids_, (size_t)np * sizeof(int32_t));
                qd.memcpy(A.wts, q4_moe_wts_, (size_t)np * sizeof(float)).wait();
            } else {
                moe_router_launch(qd, wptr(dev, L.moe.gate_inp), L.moe.gate_inp_type, b.mixed, n_embd, n_tok, n_embd,
                                  hp.n_expert, k, A.logits, A.ids, A.wts);
            }
            if (hist_on || (q4_hot_ && mode != 0)) {
                if (!si::env::flag("PF_Q4_HOSTROUTER")) {
                    qd.memcpy(q4_moe_ids_, A.ids, (size_t)np * sizeof(int32_t)).wait();
                }
                if (hist_on) {
                    hist_add(il);
                }
                cal_add(il);
            }
            if (q4_hot_ && q4_hot_ready_) {
                // All-GPU LRU: every expert the step selects is uploaded into a
                // slot before the kernels run, so resident and newly uploaded
                // experts use the same arithmetic and the next step's routing
                // does not depend on where an expert happened to live.
                const double ti0 = htt ? now_ms() : 0.0;
                qd.memcpy(q4_moe_ids_, A.ids, (size_t)np * sizeof(int32_t)).wait();
                if (htt) {
                    const double t = now_ms();
                    fprintf(stderr, " ids=%.1f", t - ti0);
                    thl = t;
                }
                const int E = hp.n_expert;
                const int R = q4_hot_r_;
                int32_t * sm = &q4_sm_host_[(size_t)il * (size_t)E];
                uint32_t * age = &q4_lru_age_[(size_t)il * (size_t)R];
                std::vector<char> seen((size_t)E, 0);
                std::vector<int> act;
                for (int pr = 0; pr < np; pr++) {
                    const int e = q4_moe_ids_[pr];
                    if (e >= 0 && e < E && !seen[(size_t)e]) {
                        seen[(size_t)e] = 1;
                        act.push_back(e);
                    }
                }
                std::vector<int> slot_e((size_t)R, -1);
                std::vector<char> slot_act((size_t)R, 0);
                for (int e = 0; e < E; e++) {
                    const int sl = sm[e];
                    if (sl >= 0 && sl < R) {
                        slot_e[(size_t)sl] = e;
                    }
                }
                ++q4_lru_tick_;
                for (int e : act) {
                    const int sl = sm[e];
                    if (sl >= 0 && sl < R) {
                        slot_act[(size_t)sl] = 1;
                        age[(size_t)sl] = q4_lru_tick_;
                    }
                }
                if ((int)act.size() > R) {
                    fprintf(stderr,
                            "[q4] hot bank: layer %d selects %zu experts against %d slots; lower PF_Q4_PF_CHUNK\n", il,
                            act.size(), R);
                }
                bool changed = false;
                for (int e : act) {
                    if (sm[e] >= 0) {
                        continue;
                    }
                    int victim = -1;
                    for (int sl = 0; sl < R; sl++) {
                        if (slot_e[(size_t)sl] < 0 && !slot_act[(size_t)sl]) {
                            victim = sl;
                            break;
                        }
                    }
                    for (int sl = 0; sl < R && victim < 0; sl++) {
                        if (!slot_act[(size_t)sl] && (victim < 0 || age[(size_t)sl] < age[(size_t)victim])) {
                            victim = sl;
                        }
                    }
                    for (int sl = 0; sl < R && victim < 0; sl++) {
                        if (victim < 0 || age[(size_t)sl] < age[(size_t)victim]) {
                            victim = sl;
                        }
                    }
                    if (victim < 0) {
                        break;
                    }
                    if (slot_e[(size_t)victim] >= 0) {
                        sm[slot_e[(size_t)victim]] = -1;
                    }
                    sm[e] = victim;
                    slot_e[(size_t)victim] = e;
                    slot_act[(size_t)victim] = 1;
                    age[(size_t)victim] = q4_lru_tick_;
                    const size_t gb = bank->sg, ub = bank->su, db = bank->sd;
                    q4_upload(dev, (char *)bank->g + (size_t)victim * gb,
                              (const char *)L.moe.gate_exps.data + (size_t)e * gb, gb);
                    q4_upload(dev, (char *)bank->u + (size_t)victim * ub,
                              (const char *)L.moe.up_exps.data + (size_t)e * ub, ub);
                    q4_upload(dev, (char *)bank->d + (size_t)victim * db,
                              (const char *)L.moe.down_exps.data + (size_t)e * db, db);
                    upmiss++;
                    upbytes += gb + ub + db;
                    changed = true;
                }
                if (changed) {
                    q4_upload(dev, bank->sm, sm, (size_t)E * sizeof(int32_t));
                }
                if (!q4_hotchk_done_ && si::env::flag("PF_Q4_HOTCHK")) {
                    int bg = 0, bu = 0, bd = 0;
                    std::vector<uint8_t> got(256), want(256);
                    for (int e : act) {
                        const int sl = sm[e];
                        if (sl < 0) {
                            continue;
                        }
                        auto one = [&](int & bad, const char * dst, const char * src, size_t plane) {
                            const size_t off = plane > 256 ? (plane - 256) / 2 : 0;
                            qd.memcpy(got.data(), dst + (size_t)sl * plane + off, 256).wait();
                            std::memcpy(want.data(), src + (size_t)e * plane + off, 256);
                            for (int i = 0; i < 256; i++) {
                                bad += got[(size_t)i] != want[(size_t)i];
                            }
                        };
                        one(bg, (const char *)bank->g, (const char *)L.moe.gate_exps.data, bank->sg);
                        one(bu, (const char *)bank->u, (const char *)L.moe.up_exps.data, bank->su);
                        one(bd, (const char *)bank->d, (const char *)L.moe.down_exps.data, bank->sd);
                    }
                    fprintf(stderr, "[q4hotchk] il=%d act=%zu bad=%d/%d/%d\n", il, act.size(), bg, bu, bd);
                }
                if (htt) {
                    const double tl = now_ms();
                    qd.wait();
                    const double t = now_ms();
                    fprintf(stderr, " lru=%.1f up=%.1f lwait=%.1f miss=%d upmb=%.2f", tl - thl, t - tl,
                            q4_up_wait_ms_ - q4_up_wait_snap, upmiss, (double)upbytes / (1024.0 * 1024.0));
                    thl = t;
                }
            }
            moe_csr_clear_launch(qd, A.off, A.cur, A.active, A.nact, hp.n_expert);
            moe_csr_count_launch(qd, A.ids, np, A.off);
            moe_csr_prefix_launch(qd, A.off, A.active, A.nact, hp.n_expert);
            moe_csr_scatter_launch(qd, A.ids, A.off, A.cur, A.pair, np);
            moe_actq_launch(qd, b.mixed, n_embd, n_tok, n_embd, A.xq, A.asa, A.xs);
            if (htt) {
                qd.wait();
                const double t = now_ms();
                fprintf(stderr, " aq=%.1f", t - thl);
                thl = t;
            }
            // `active` can never hold more experts than there are pairs: for a
            // decode that is 10, not 512.  The grid is n_act_max * column
            // blocks, so launching n_expert made 502 of 512 workgroups return
            // immediately on every decode step.
            const int n_act = std::min(hp.n_expert, np);
            const bool dec = (mode == 0);
            bool ok = false;
            if (!si::env::flag("PF_Q4_NOGRP") && !si::env::flag("PF_Q4_NOGRP_UP")) {
                ok = moe_grp_gate_up_launch(qd, bank->g, bank->u, bank->sm, bank->sg, bank->su, A.off, A.pair,
                                            A.active, A.nact, n_act, hp.n_expert, np, A.xq, A.asa, A.xs, A.h,
                                            hp.n_ff_exp, n_embd, hp.n_ff_exp, k, L.moe.gate_exps.type,
                                            L.moe.up_exps.type, dec);
            }
            if (!ok) {
                moe_gate_up_launch(qd, bank->g, bank->u, bank->sm, bank->sg, bank->su, A.off, A.pair, A.active, A.nact,
                                   n_act, b.mixed, n_embd, A.h, hp.n_ff_exp, n_embd, hp.n_ff_exp, k,
                                   L.moe.gate_exps.type, L.moe.up_exps.type);
            }
            if (htt) {
                qd.wait();
                const double t = now_ms();
                fprintf(stderr, " gu=%.1f", t - thl);
                thl = t;
            }
            moe_actq_launch(qd, A.h, hp.n_ff_exp, np, hp.n_ff_exp, A.xq, A.asa, A.xs);
            if (q4_hot_) {
                // Every pair the layer selects is resident by now (the LRU
                // uploaded it above); zero so a slot skipped for an inactive
                // expert cannot carry the previous layer's value.
                qd.memset(A.acc, 0, (size_t)np * (size_t)n_embd * sizeof(float));
            }
            bool okd = false;
            const bool grp_off = si::env::flag("PF_Q4_NOGRP") || si::env::flag("PF_Q4_NOGRP_DOWN");
            if (L.moe.down_exps.type == 7 && !grp_off) {
                // Q5_1 is one 32-value group per block, not a 256-value
                // superblock, so `moe_grp_down_launch` declines the bank's
                // K = n_ff_exp = 640 and the down used to fall through to the
                // fp32-lane dequant kernel - the one projection of the bank
                // that was not on dp4a.  `PF_Q4_NOGRP_DOWN=1` is the A/B back
                // to that path.
                moe_down_q5_1_launch(qd, bank->d, bank->sm, bank->sd, A.off, A.pair, A.active, A.nact, n_act,
                                     hp.n_expert, np, A.xq, A.asa, A.xs, A.acc, n_embd, n_embd, hp.n_ff_exp);
                okd = true;
            } else if (!grp_off) {
                okd = moe_grp_down_launch(qd, bank->d, bank->sm, bank->sd, A.off, A.pair, A.active, A.nact, n_act,
                                          hp.n_expert, np, A.xq, A.asa, A.xs, A.acc, n_embd, n_embd, hp.n_ff_exp,
                                          L.moe.down_exps.type, dec);
            }
            if (!okd) {
                moe_down_launch(qd, bank->d, bank->sm, bank->sd, A.off, A.pair, A.active, A.nact, n_act, A.h,
                                hp.n_ff_exp, A.acc, n_embd, n_embd, hp.n_ff_exp, L.moe.down_exps.type);
            }
            if (htt) {
                qd.wait();
                const double t = now_ms();
                fprintf(stderr, " dn=%.1f", t - thl);
                thl = t;
            }
            // moe_reduce ACCUMULATES (`out +=`), and b.block still holds the
            // residual stream's routed slot from the previous layer: zero it
            // first or every banked layer adds onto the last one (cos 0.27
            // against the host for one layer, garbage for seven).
            qd.memset(b.block, 0, (size_t)n_tok * n_embd * sizeof(float));
            const char * ffs = si::env::str("PF_Q4_FFCMP");
            if (ffs && atoi(ffs) == il) {
                qd.memcpy(q4_moe_ids_, A.ids, (size_t)np * sizeof(int32_t)).wait();
                std::vector<float> gacc((size_t)np * (size_t)n_embd);
                qd.memcpy(gacc.data(), A.acc, gacc.size() * sizeof(float)).wait();
                to_host(dev, q4_host_mix_, b.mixed, (size_t)n_tok * n_embd * sizeof(float));
                std::vector<float> xs((size_t)np * (size_t)n_embd);
                for (int pr = 0; pr < np; pr++) {
                    std::memcpy(xs.data() + (size_t)pr * n_embd,
                                q4_host_mix_ + (size_t)(pr / k) * n_embd, (size_t)n_embd * sizeof(float));
                }
                std::vector<int32_t> ids(q4_moe_ids_, q4_moe_ids_ + np);
                cpu_moe CM;
                CM.gate_exps = {(const char *)L.moe.gate_exps.data, L.moe.gate_exps.type, L.moe.gate_exps.K,
                                L.moe.gate_exps.N, L.moe.gate_exps.plane_bytes};
                CM.up_exps = {(const char *)L.moe.up_exps.data, L.moe.up_exps.type, L.moe.up_exps.K, L.moe.up_exps.N,
                              L.moe.up_exps.plane_bytes};
                CM.down_exps = {(const char *)L.moe.down_exps.data, L.moe.down_exps.type, L.moe.down_exps.K,
                                L.moe.down_exps.N, L.moe.down_exps.plane_bytes};
                CM.n_expert = hp.n_expert;
                CM.top_k = 1;
                CM.n_ff_exp = hp.n_ff_exp;
                CM.n_embd = n_embd;
                CM.down_i8 = L.moe.down_i8;
                CM.down_sc = L.moe.down_sc;
                CM.down_K = L.moe.down_i8_K;
                CM.down_N = L.moe.down_i8_N;
                CM.down_E = L.moe.down_i8_E;
                if (si::env::flag("PF_Q4_FFCMP_NATIVE") && L.moe.down_exps.type == 7) {
                    CM.down_raw = L.moe.down_exps.data;
                    CM.down_row_bytes = quant_row_bytes(L.moe.down_exps.type, L.moe.down_exps.K);
                    CM.down_type = L.moe.down_exps.type;
                }
                std::vector<float> hacc((size_t)np * (size_t)n_embd);
                cpu_moe_experts(CM, xs.data(), n_embd, np, 1, ids.data(), nullptr, hacc.data(), n_embd, true);
                double dot = 0, na = 0, nb = 0, mx = 0;
                double worst = 1.0;
                int wpr = -1;
                for (int pr = 0; pr < np; pr++) {
                    double d = 0, a = 0, c = 0;
                    for (int r = 0; r < n_embd; r++) {
                        const double g = gacc[(size_t)pr * n_embd + r], h = hacc[(size_t)pr * n_embd + r];
                        d += g * h;
                        a += g * g;
                        c += h * h;
                        mx = std::max(mx, std::fabs(g - h));
                    }
                    dot += d;
                    na += a;
                    nb += c;
                    const double cs = d / (std::sqrt(a * c) + 1e-30);
                    if (cs < worst) {
                        worst = cs;
                        wpr = pr;
                    }
                }
                fprintf(stderr, "[q4ffcmp] il=%d np=%d cos=%.6f rms=%.4f max=%.4f worst=%.4f pr=%d\n", il, np,
                        dot / (std::sqrt(na * nb) + 1e-30), std::sqrt(na / (nb + 1e-30)), mx, worst, wpr);
                if (si::env::flag("PF_Q4_FFCMP_H")) {
                    std::vector<float> gh((size_t)np * (size_t)hp.n_ff_exp);
                    qd.memcpy(gh.data(), A.h, gh.size() * sizeof(float)).wait();
                    const int F = hp.n_ff_exp;
                    std::vector<float> rowbuf((size_t)L.moe.gate_exps.K);
                    double d2 = 0, a2 = 0, c2 = 0;
                    const int e0 = ids[0];
                    const char * gp = (const char *)L.moe.gate_exps.data + (size_t)e0 * L.moe.gate_exps.plane_bytes;
                    const char * up = (const char *)L.moe.up_exps.data + (size_t)e0 * L.moe.up_exps.plane_bytes;
                    const size_t rg = quant_row_bytes(L.moe.gate_exps.type, L.moe.gate_exps.K);
                    const size_t ru = quant_row_bytes(L.moe.up_exps.type, L.moe.up_exps.K);
                    for (int i = 0; i < F; i++) {
                        dequantize_row(L.moe.gate_exps.type, gp + (size_t)i * rg, rowbuf.data(), L.moe.gate_exps.K);
                        double ag = 0;
                        for (int kk = 0; kk < L.moe.gate_exps.K; kk++) {
                            ag += (double)rowbuf[(size_t)kk] * xs[(size_t)kk];
                        }
                        dequantize_row(L.moe.up_exps.type, up + (size_t)i * ru, rowbuf.data(), L.moe.up_exps.K);
                        double au = 0;
                        for (int kk = 0; kk < L.moe.up_exps.K; kk++) {
                            au += (double)rowbuf[(size_t)kk] * xs[(size_t)kk];
                        }
                        const double hh = (ag / (1.0 + std::exp(-ag))) * au;
                        const double gg = gh[(size_t)i];
                        d2 += gg * hh;
                        a2 += gg * gg;
                        c2 += hh * hh;
                    }
                    fprintf(stderr, "[q4ffcmp]   h pair0 e=%d cos=%.5f rms=%.4f\n", e0,
                            d2 / (std::sqrt(a2 * c2) + 1e-30), std::sqrt(a2 / (c2 + 1e-30)));
                }
                if (si::env::flag("PF_Q4_FFCMP_MAP")) {
                    for (int pr = 0; pr < std::min(np, 4); pr++) {
                        double best = -2;
                        int bi = -1;
                        for (int q = 0; q < np; q++) {
                            double d = 0, a = 0, c = 0;
                            for (int r = 0; r < n_embd; r++) {
                                const double g = gacc[(size_t)pr * n_embd + r], h = hacc[(size_t)q * n_embd + r];
                                d += g * h;
                                a += g * g;
                                c += h * h;
                            }
                            const double cs = d / (std::sqrt(a * c) + 1e-30);
                            if (cs > best) {
                                best = cs;
                                bi = q;
                            }
                        }
                        fprintf(stderr, "[q4ffcmp]   pr=%d best=%d cos=%.4f\n", pr, bi, best);
                    }
                }
            }
            moe_reduce_launch(qd, A.acc, A.wts, k, n_tok, n_embd, n_embd, b.block, n_embd);
            if (htt) {
                qd.wait();
                const double t = now_ms();
                fprintf(stderr, " red=%.1f", t - thl);
                thl = t;
            }
            const double tsh0 = htt ? now_ms() : 0.0;
            // Shared expert: oneDNN on the same card (it has its own converted
            // weights), otherwise the host path with two small copies.
            dnnl_gemm * D = dnnl_for(dev);
            bool sh_gpu = false;
            if (!si::env::flag("PF_Q4_NO_SHGPU") && D && D->has_weight(L.ffn_gate.data)
                && D->has_weight(L.ffn_up.data) && D->has_weight(L.ffn_down.data)) {
                act_view v = D->quantize(b.mixed, nullptr, n_embd, 0, n_tok, n_embd, false);
                if (v.valid() && D->gemm(v, L.ffn_gate.data, nullptr, 1.0f, n_tok, n_embd, A.sh, 2 * hp.n_ff)
                    && D->gemm(v, L.ffn_up.data, nullptr, 1.0f, n_tok, n_embd, A.sh + hp.n_ff, 2 * hp.n_ff)) {
                    act_view vd = D->quantize(A.sh, A.sh + hp.n_ff, 2 * hp.n_ff, 2 * hp.n_ff, n_tok, hp.n_ff, false);
                    if (vd.valid()
                        && D->gemm(vd, L.ffn_down.data, nullptr, 1.0f, n_tok, hp.n_ff, A.shout, n_embd)) {
                        moe_gate_add_launch(qd, wptr(dev, L.moe.gate_inp_shexp), L.moe.gate_inp_shexp_type, b.mixed,
                                            n_embd, n_tok, n_embd, A.shout, n_embd, b.block, n_embd);
                        sh_gpu = true;
                    }
                }
            }
            if (!sh_gpu) {
                to_host(dev, q4_host_mix_, b.mixed, (size_t)n_tok * n_embd * sizeof(float));
                to_host(dev, q4_host_out_, b.block, (size_t)n_tok * n_embd * sizeof(float));
                hgate_up(L.ffn_gate, L.ffn_up, q4_host_mix_, n_embd, q4_moe_sh_, 2 * hp.n_ff);
                hdown(L.ffn_down, q4_moe_sh_, 2 * hp.n_ff, q4_moe_shout_, n_embd, true);
                cpu_moe_epilogue(q4_host_mix_, n_embd, L.moe.gate_inp_shexp, L.moe.gate_inp_shexp_type, n_tok, n_embd,
                                 q4_moe_shout_, n_embd, q4_host_out_, n_embd);
                from_host(dev, b.block, q4_host_out_, (size_t)n_tok * n_embd * sizeof(float));
            }
            if (htt) {
                qd.wait();
                const double t = now_ms();
                fprintf(stderr, " sh=%.1f tot=%.1f\n", t - tsh0, t - th0);
                thl = t;
            }
            if (q4time) {
                t_moe += now_ms() - tm0;
            }
            hc_combine(dev, inj2);
            dbg_stats("l_last", dev, b.res, (size_t)n_tok * hc_dim, hc);
            continue;
        }
        if (is_cpu(dev)) {
            to_host(dev, q4_host_mix_, b.mixed, (size_t)n_tok * n_embd * sizeof(float));
        } else {
            // The router and its weights live on the device already; running it
            // there avoids streaming 5 MB of router per layer from host RAM.
            moe_router_launch(DQ(dev), wptr(dev, L.moe.gate_inp), L.moe.gate_inp_type, b.mixed, n_embd, n_tok, n_embd,
                              hp.n_expert, k, b.router_logits, b.router_ids, b.router_wts);
            sycl::queue & qd = DQ(dev);
            if (q4trace) {
                fprintf(stderr, "[q4] router copies d=%d n=%d\n", dev, n_tok);
            }
            qd.memcpy(q4_host_mix_, b.mixed, (size_t)n_tok * n_embd * sizeof(float));
            qd.memcpy(q4_moe_ids_, b.router_ids, (size_t)n_tok * k * sizeof(int32_t));
            qd.memcpy(q4_moe_wts_, b.router_wts, (size_t)n_tok * k * sizeof(float)).wait();
        }
        if (q4time) { t_cp += now_ms() - tc0; }
        cpu_moe CM;
        CM.gate_exps = {(const char *)L.moe.gate_exps.data, L.moe.gate_exps.type, L.moe.gate_exps.K, L.moe.gate_exps.N,
                        L.moe.gate_exps.plane_bytes};
        CM.up_exps = {(const char *)L.moe.up_exps.data, L.moe.up_exps.type, L.moe.up_exps.K, L.moe.up_exps.N,
                      L.moe.up_exps.plane_bytes};
        CM.down_exps = {(const char *)L.moe.down_exps.data, L.moe.down_exps.type, L.moe.down_exps.K,
                        L.moe.down_exps.N, L.moe.down_exps.plane_bytes};
        CM.gate_inp = L.moe.gate_inp;
        CM.gate_inp_type = L.moe.gate_inp_type;
        CM.gate_inp_shexp = L.moe.gate_inp_shexp;
        CM.gate_inp_shexp_type = L.moe.gate_inp_shexp_type;
        CM.n_expert = hp.n_expert;
        CM.top_k = k;
        CM.n_ff_exp = hp.n_ff_exp;
        CM.n_embd = n_embd;
        CM.down_i8 = L.moe.down_i8;
        CM.down_sc = L.moe.down_sc;
        CM.down_K = L.moe.down_i8_K;
        CM.down_N = L.moe.down_i8_N;
        CM.down_E = L.moe.down_i8_E;
        if (L.moe.down_exps.type == 7 && si::env::flag("PF_Q4_DOWN_NATIVE")) {
            CM.down_raw = L.moe.down_exps.data;
            CM.down_row_bytes = quant_row_bytes(L.moe.down_exps.type, L.moe.down_exps.K);
            CM.down_type = L.moe.down_exps.type;
        }
        const double tr0 = q4time ? now_ms() : 0.0;
        if (is_cpu(dev)) {
            cpu_moe_route(q4_host_mix_, n_embd, CM.gate_inp, CM.gate_inp_type, n_tok, n_embd, hp.n_expert, k,
                          q4_moe_logits_, q4_moe_ids_, q4_moe_wts_, true);
        }
        if (q4time) { t_route += now_ms() - tr0; }
        hist_add(il);
        cal_add(il);
        if (q4dbg && il == 0 && is_cpu(dev)) {
            for (int t = 0; t < n_tok; t++) {
                int bi[5] = {-1, -1, -1, -1, -1};
                for (int e = 0; e < hp.n_expert; e++) {
                    for (int j = 0; j < 5; j++) {
                        if (bi[j] < 0 || q4_moe_logits_[t * hp.n_expert + e] > q4_moe_logits_[t * hp.n_expert + bi[j]]) {
                            for (int z = 4; z > j; z--) {
                                bi[z] = bi[z - 1];
                            }
                            bi[j] = e;
                            break;
                        }
                    }
                }
                fprintf(stderr, "[q4] L0 moe_logits tok%d:", t);
                for (int j = 0; j < 5; j++) {
                    fprintf(stderr, " %d(%.4f)", bi[j], q4_moe_logits_[t * hp.n_expert + bi[j]]);
                }
                fprintf(stderr, "\n");
                fprintf(stderr, "[q4] L0 moe_ids tok%d:", t);
                for (int j = 0; j < k; j++) {
                    fprintf(stderr, " %d(%.3f)", q4_moe_ids_[t * k + j], q4_moe_wts_[t * k + j]);
                }
                fprintf(stderr, "\n");
            }
        }
        const double te0 = q4time ? now_ms() : 0.0;
        cpu_moe_experts(CM, q4_host_mix_, n_embd, n_tok, k, q4_moe_ids_, nullptr, q4_moe_acc_, n_embd, true);
        std::memset(q4_host_out_, 0, (size_t)n_tok * n_embd * sizeof(float));
        cpu_moe_reduce(q4_moe_acc_, q4_moe_wts_, k, n_tok, n_embd, n_embd, q4_host_out_, n_embd);
        if (q4time) { t_exp += now_ms() - te0; }
        dbg_stats("moe_routed", dev, q4_host_out_, (size_t)n_tok * n_embd, 0);
        const double ts0 = q4time ? now_ms() : 0.0;
        // The shared expert is dense (Q8_0) and oneDNN has its int8 copy on the
        // card.  Running it on the device is M-token amortised; the per-token
        // host GEMV read 3.4 MB/token/layer (5.8 s of a 512-token prefill).
        bool sh_gpu = false;
        if (!is_cpu(dev)) {
            dnnl_gemm * D = dnnl_for(dev);
            moe_act & A = moe_as_[(size_t)dev];
            if (D && D->has_weight(L.ffn_gate.data) && D->has_weight(L.ffn_up.data)
                && D->has_weight(L.ffn_down.data)) {
                act_view v = D->quantize(b.mixed, nullptr, n_embd, 0, n_tok, n_embd, false);
                if (v.valid() && D->gemm(v, L.ffn_gate.data, nullptr, 1.0f, n_tok, n_embd, A.sh, 2 * hp.n_ff)
                    && D->gemm(v, L.ffn_up.data, nullptr, 1.0f, n_tok, n_embd, A.sh + hp.n_ff, 2 * hp.n_ff)) {
                    act_view vd = D->quantize(A.sh, A.sh + hp.n_ff, 2 * hp.n_ff, 2 * hp.n_ff, n_tok, hp.n_ff, false);
                    if (vd.valid()
                        && D->gemm(vd, L.ffn_down.data, nullptr, 1.0f, n_tok, hp.n_ff, A.shout, n_embd)) {
                        from_host(dev, b.block, q4_host_out_, (size_t)n_tok * n_embd * sizeof(float));
                        moe_gate_add_launch(DQ(dev), wptr(dev, L.moe.gate_inp_shexp), L.moe.gate_inp_shexp_type,
                                            b.mixed, n_embd, n_tok, n_embd, A.shout, n_embd, b.block, n_embd);
                        sh_gpu = true;
                    }
                }
            }
        }
        if (!sh_gpu) {
            hgate_up(L.ffn_gate, L.ffn_up, q4_host_mix_, n_embd, q4_moe_sh_, 2 * hp.n_ff);
            dbg_stats("moe_sh_up", dev, q4_moe_sh_, (size_t)n_tok * 2 * hp.n_ff, 0);
            hdown(L.ffn_down, q4_moe_sh_, 2 * hp.n_ff, q4_moe_shout_, n_embd, true);
            dbg_stats("moe_shared", dev, q4_moe_shout_, (size_t)n_tok * n_embd, 0);
            cpu_moe_epilogue(q4_host_mix_, n_embd, L.moe.gate_inp_shexp, L.moe.gate_inp_shexp_type, n_tok, n_embd,
                             q4_moe_shout_, n_embd, q4_host_out_, n_embd);
            dbg_stats("moe_epi", dev, q4_host_out_, (size_t)n_tok * n_embd, 0);
            from_host(dev, b.block, q4_host_out_, (size_t)n_tok * n_embd * sizeof(float));
        }
        if (q4time) { t_shared += now_ms() - ts0; }
        const double tc1 = q4time ? now_ms() : 0.0;
        from_host(dev, b.block, q4_host_out_, (size_t)n_tok * n_embd * sizeof(float));
        if (q4time) { t_cp += now_ms() - tc1; }
        dbg_stats("ffn_out", dev, b.block, (size_t)n_tok * n_embd, 0);
        hc_combine(dev, inj2);
        dbg_stats("l_last", dev, b.res, (size_t)n_tok * hc_dim, hc);
        if (q4time) {
            t_moe += now_ms() - tm0;
        }
    }

    q4_hotchk_done_ = true;
    if (q4time) {
        const double total = now_ms() - t_start;
        fprintf(stderr,
                "[q4] fwd n=%d moe_host=%.1f rest=%.1f | route=%.1f experts=%.1f shared=%.1f copies=%.1f ms\n", n_tok,
                t_moe, total - t_moe, t_route, t_exp, t_shared, t_cp);
    }
    if (!with_head) {
        // A partial chunk returns while its kernels and seg-ring copies may
        // still be in flight; the next chunk reuses the host staging ring and
        // the same USM buffers.  Wait, or the driver sees an overwritten copy
        // source (and the second prefill chunk dies inside urEnqueueUSMMemcpy).
        if (multi) {
            sync_all();
        } else {
            be_of(0).synchronize();
        }
        return;
    }
    // ---- final hyper-connection mixer + LM head ---------------------------
    if (cur_dev != d0) {
        to_host(cur_dev, q4_host_res_, buf(cur_dev).res, (size_t)n_tok * hc_dim * sizeof(float));
        from_host(d0, buf(d0).res, q4_host_res_, (size_t)n_tok * hc_dim * sizeof(float));
    }
    q4_bufs & b0 = buf(d0);
    const hc_wt & H = m.hc_head;
    if (is_cpu(d0)) {
        cpu_q4_hc_norm(b0.res, H.norm, b0.xn, nullptr, n_tok, n_embd, hc, hp.rms_eps);
    } else {
        q4_hc_norm_launch(DQ(d0), b0.res, wf32(d0, H.norm), b0.xn, n_tok, n_embd, hc, hp.rms_eps);
    }
    gemv(d0, H.down, b0.xn, hc_dim, b0.lo, hp.hc_low_rank, tb, nblocks);
    if (is_cpu(d0)) {
        cpu_q4_silu_scaled(b0.lo, n_tok * hp.hc_low_rank, inv_hc);
    } else {
        q4_silu_scale_launch(DQ(d0), b0.lo, n_tok * hp.hc_low_rank, inv_hc);
    }
    if (is_cpu(d0)) {
        cpu_q4_up_gemv(H.up.data, H.up.type, b0.lo, hp.hc_low_rank, b0.gate, hc_dim, n_tok, hp.hc_low_rank, hc_dim);
    } else {
        q4_up_gemv_launch(DQ(d0), wptr(d0, H.up.data), H.up.type, b0.lo, hp.hc_low_rank, b0.gate, hc_dim, n_tok,
                          hp.hc_low_rank, hc_dim);
    }
    if (is_cpu(d0)) {
        cpu_q4_hc_mix(b0.xn, b0.gate, b0.mixed, nullptr, n_tok, n_embd, hc);
    } else {
        q4_hc_mix_launch(DQ(d0), b0.xn, b0.gate, b0.mixed, n_tok, n_embd, hc);
    }
    dbg_stats("result_norm", d0, b0.mixed, (size_t)n_tok * n_embd, 0);
    if (mode != 0) {
        // a prefill (mode 1 tail or mode 2 rows) only needs the last token's
        // head; a decode (mode 0) needs one row per sequence below
        if (is_cpu(d0)) {
            std::memcpy(d_last_hidden, b0.mixed + (size_t)(n_tok - 1) * n_embd, (size_t)n_embd * sizeof(float));
        } else {
            DQ(d0).memcpy(d_last_hidden, b0.mixed + (size_t)(n_tok - 1) * n_embd, (size_t)n_embd * sizeof(float)).wait();
        }
        gemv(d0, m.output, b0.mixed + (size_t)(n_tok - 1) * n_embd, n_embd, d_logits, hp.n_vocab, 1, 0);
    } else {
        if (is_cpu(d0)) {
            std::memcpy(d_last_hidden, b0.mixed, (size_t)n_embd * sizeof(float));
        } else {
            DQ(d0).memcpy(d_last_hidden, b0.mixed, (size_t)n_embd * sizeof(float)).wait();
        }
        gemv(d0, m.output, b0.mixed, n_embd, d_logits, hp.n_vocab, tb, 1);
        if (q4dbg) {
            std::vector<float> lg((size_t)hp.n_vocab);
            to_host(d0, lg.data(), d_logits, (size_t)hp.n_vocab * sizeof(float));
            int bi[5] = {-1, -1, -1, -1, -1};
            for (int e = 0; e < hp.n_vocab; e++) {
                for (int j = 0; j < 5; j++) {
                    if (bi[j] < 0 || lg[(size_t)e] > lg[(size_t)bi[j]]) {
                        for (int z = 4; z > j; z--) {
                            bi[z] = bi[z - 1];
                        }
                        bi[j] = e;
                        break;
                    }
                }
            }
            fprintf(stderr, "[q4] decode logits:");
            for (int j = 0; j < 5; j++) {
                fprintf(stderr, " %d(%.4f)", bi[j], lg[(size_t)bi[j]]);
            }
            fprintf(stderr, "\n");
        }
    }
    if (multi) {
        sync_all();
    } else {
        BE(0).synchronize();
    }
}


void engine::q4_hist_report() const {
    auto go = [&](const char * tag, const std::vector<uint64_t> & H, size_t pairs) {
        if (H.empty()) {
            return;
        }
        const int n_layer = m.hp.n_layer, n_expert = m.hp.n_expert;
        std::vector<std::pair<uint64_t, int>> v;
        uint64_t tot = 0;
        for (size_t i = 0; i < H.size(); i++) {
            if (H[i]) {
                v.push_back({H[i], (int)i});
                tot += H[i];
            }
        }
        if (tot == 0) {
            return;
        }
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        fprintf(stderr, "[q4hist] %s: %zu pairs, %zu/%d slots used, total=%llu\n", tag, pairs, v.size(),
                n_layer * n_expert, (unsigned long long)tot);
        uint64_t cum = 0;
        size_t vi = 0;
        const size_t budgets[] = {1000, 2000, 4000, 6200, 8000, 12000, 16000};
        for (size_t B : budgets) {
            while (vi < v.size() && vi < B) {
                cum += v[vi++].first;
            }
            fprintf(stderr, "[q4hist]   top-%-5zu %6.2f%%\n", B, 100.0 * (double)cum / (double)tot);
        }
        std::vector<uint64_t> per((size_t)n_layer, 0);
        for (size_t i = 0; i < H.size(); i++) {
            per[i / (size_t)n_expert] += H[i];
        }
        std::sort(per.begin(), per.end(), [](uint64_t a, uint64_t b) { return a > b; });
        auto share = [&](int n) {
            uint64_t c = 0;
            for (int i = 0; i < n && i < n_layer; i++) {
                c += per[(size_t)i];
            }
            return 100.0 * (double)c / (double)tot;
        };
        fprintf(stderr, "[q4hist]   whole layers: top-8 %.1f%%, top-10 %.1f%%, top-12 %.1f%%\n", share(8),
                share(10), share(12));
        // uniform per-layer slot allocation: R experts kept in every layer
        const int Rs[] = {64, 96, 129, 155, 192, 256};
        for (int R : Rs) {
            uint64_t c = 0;
            for (int il = 0; il < n_layer; il++) {
                std::vector<uint64_t> row(H.begin() + (size_t)il * n_expert,
                                          H.begin() + (size_t)(il + 1) * n_expert);
                std::sort(row.begin(), row.end(), [](uint64_t a, uint64_t b) { return a > b; });
                for (int i = 0; i < R && i < n_expert; i++) {
                    c += row[(size_t)i];
                }
            }
            fprintf(stderr, "[q4hist]   per-layer R=%-3d %6.2f%%\n", R, 100.0 * (double)c / (double)tot);
        }
    };
    go("prefill", q4_hist_pre_, q4_hist_pre_pairs_);
    go("decode", q4_hist_dec_, q4_hist_dec_pairs_);
    const char * cf = si::env::str("PF_Q4_CAL_SAVE");
    if (cf && !q4_hist_pre_.empty()) {
        FILE * f = fopen(cf, "wb");
        if (f) {
            std::vector<uint32_t> out(q4_hist_pre_.size());
            for (size_t i = 0; i < out.size(); i++) {
                out[i] = (uint32_t)std::min<uint64_t>(q4_hist_pre_[i], 0xFFFFFFFFu);
            }
            fwrite(out.data(), sizeof(uint32_t), out.size(), f);
            fclose(f);
            fprintf(stderr, "[q4hist] saved calibration to %s\n", cf);
        }
    }
}

} // namespace si
