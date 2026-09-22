#include "engine.h"
#include "quant.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace si {

// Resolve the requested device: -1 = auto from PF_DEVICE (default gpu),
// 0/1 = gpu/cpu.  Kept independent of the queue so a CPU build on a machine
// without a SYCL CPU device still reports cpu while using a GPU context purely
// for host-USM allocation.
static int resolve_device(int device) {
    if (device >= 0) {
        return device ? 1 : 0;
    }
    if (const char * e = getenv("PF_DEVICE")) {
        if (strcmp(e, "cpu") == 0 || strcmp(e, "host") == 0 || strcmp(e, "1") == 0) {
            return 1;
        }
    }
    return 0;
}

static sycl::queue make_queue(int device_req, sycl::context * ctx) {
    if (device_req == 1) {
        try {
            return sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
        } catch (...) {
            // no SYCL CPU device available: the queue is only used for
            // host-USM allocation / copies, compute runs on the CPU backend
        }
    }
    if (ctx) {
        // shared multi-GPU context: the primary queue is its first device
        return sycl::queue(*ctx, ctx->get_devices().front(), sycl::property::queue::in_order());
    }
    return sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order());
}

// A --layer-map that places layers on two or more distinct GPUs shares one
// context across all enumerated GPUs (USM host allocations are context-scoped).
static std::shared_ptr<sycl::context> make_md_context(const std::string & layer_map) {
    if (layer_map.empty()) {
        return nullptr;
    }
    std::vector<sycl::device> gpus;
    try {
        gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    } catch (...) {
        return nullptr;
    }
    if (gpus.size() < 2) {
        return nullptr;
    }
    // parse "gpu.N" / "gpu" targets and count distinct device indices
    std::vector<int> idx;
    for (size_t p = 0; (p = layer_map.find("gpu", p)) != std::string::npos;) {
        int id = 0;
        if (p + 3 < layer_map.size() && layer_map[p + 3] == '.') {
            id = atoi(layer_map.c_str() + p + 4);
        }
        if (std::find(idx.begin(), idx.end(), id) == idx.end()) {
            idx.push_back(id);
        }
        p += 3;
    }
    if (idx.size() < 2) {
        return nullptr;
    }
    try {
        return std::make_shared<sycl::context>(gpus);
    } catch (...) {
        return nullptr;
    }
}

engine::engine(const std::string & model_path, int max_seq_, int n_splits_, int n_blocks_, int kv_cap_mb,
               const std::string & pc_dir_arg, int pc_disk_mb, int pc_mem_mb, int pc_ram_mb, int pc_vram_mb,
               int device, const std::string & layer_map)
    : device_req(resolve_device(device)), md_ctx_(make_md_context(layer_map)), q(make_queue(device_req, md_ctx_.get())),
      max_seq(max_seq_), n_splits(n_splits_), n_blocks(n_blocks_) {
    dev_kind = device_req == 1 ? device_kind::cpu : device_kind::gpu;
    cpu_mode = dev_kind == device_kind::cpu;
    m.load(model_path);
    tk.load(m.gguf);
    if (!layer_map.empty()) {
        setup_multi_device(layer_map);
    }
    if (!multi_dev) {
        be = cpu_mode ? make_cpu_backend() : make_gpu_backend(q);
    }
    if (cpu_mode || multi_dev) {
        n_splits = 1;
        dec_splits = 1;
    }
    // upload weights after setup_multi_device so multi-device can skip the
    // full-blob device copy and instead upload only each partition's tensors
    m.upload(q, /*host=*/cpu_mode || multi_dev);
    {
        // fp32 scale side arrays: measured as a net loss on this GPU (the packed
        // scales share cache lines with the weights; a separate array adds a
        // memory stream) -> opt-in only (PF_META=1, ~260 MB)
        const char * envm = getenv("PF_META");
        use_meta32 = !cpu_mode && !multi_dev && envm && atoi(envm) != 0;
        if (use_meta32) {
            build_meta32(q);
        }
        // int8 GEMM is on by default on both backends.  The GPU packs the SIn
        // w8 copies; the CPU's integer kernel reads the GGUF blocks directly,
        // so it does not build (or pay for) the w8 copies.
        const char * env = getenv("PF_DP4A");
        const bool dp4a_env_off = env && atoi(env) == 0;
        pf8 = !multi_dev && !dp4a_env_off;
        // PF_DP4A_DEC=0 forces the fp32 decode (both backends)
        const char * envd = getenv("PF_DP4A_DEC");
        pf8_dec = pf8 && !(envd && atoi(envd) == 0);
        if (pf8 && !cpu_mode) {
            m.build_w8(q); // SI8 copies cost ~700 MB; only the GPU needs them
        }
    }
    // prefix cache: ON by default (a hash lookup per request plus bounded
    // PF_PC_STATES * ~19 MB of state checkpoints; a cold pass captures the
    // conv/GDN state at every block boundary it crosses).  PF_PREFIX_CACHE=0
    // disables it; PF_PC_STATES=N bounds the checkpoint pool.
    {
        // decode attention K-split: the single-token shape has only n_head
        // workgroups, so at long context a wider split is what keeps the GPU
        // busy (2x the fan-out of n_splits at ~0.3 ms combine cost)
        const char * e = getenv("PF_DEC_SPLIT");
        dec_splits = e ? atoi(e) : kMaxSplits;
        if (dec_splits < 1) {
            dec_splits = 1;
        }
        if (dec_splits > kMaxDecSplits) {
            dec_splits = kMaxDecSplits;
        }
        const char * ep = getenv("PF_PREFIX_CACHE");
        pc_enabled = !(ep && atoi(ep) == 0);
        // multi-device note: block ids and recurrent-state checkpoints are global
        // (host-USM shared buffers), so the three-tier cache works here too; only
        // the block serialize/deserialize must resolve each attention layer to its
        // device's pool - see kv_layer_ptrs()
        if (pc_enabled) {
            const hparams & hp = m.hp;
            int n_gdn = 0;
            for (int il = 0; il < hp.n_layer; il++) {
                n_gdn += hp.is_recr(il);
            }
            pc_state_floats = (size_t)n_gdn * (gdn_per_slot() + conv_per_slot());
        }
        // VRAM budget: cache nodes resident in device memory are bounded by the
        // checkpoint pool (a stateful node also holds one KV block, and the KV
        // pool bounds the state-less ones).  An explicit PF_PC_STATES wins;
        // otherwise --pc-vram-mb / PF_PC_VRAM_MB (or its --pc-mem-mb /
        // PF_PC_MEM_MB alias) is divided by the per-node bytes; default 8.
        const size_t per_node = (pc_state_floats ? pc_state_floats * 4 : 0) + pc_block_blob_bytes();
        int vram_mb = -1;
        if (pc_vram_mb >= 0) {
            vram_mb = pc_vram_mb;
        } else if (pc_mem_mb >= 0) {
            vram_mb = pc_mem_mb;
        }
        int states = -1;
        if (vram_mb < 0) {
            if (const char * en = getenv("PF_PC_STATES")) {
                states = atoi(en);
            } else if (const char * ev = getenv("PF_PC_VRAM_MB")) {
                vram_mb = atoi(ev);
            } else if (const char * em = getenv("PF_PC_MEM_MB")) {
                vram_mb = atoi(em);
            }
        }
        if (states < 0 && vram_mb >= 0) {
            if (vram_mb == 0) {
                states = 0;
            } else if (per_node > 0) {
                states = (int)std::max<int64_t>(1, (int64_t)vram_mb * 1024 * 1024 / (int64_t)per_node);
            }
        }
        if (states < 0) {
            states = 8; // historical default
        }
        const int ram_mb = pc_ram_mb >= 0 ? pc_ram_mb : (getenv("PF_PC_RAM_MB") ? atoi(getenv("PF_PC_RAM_MB")) : 512);
        const int disk_mb =
            pc_disk_mb >= 0 ? pc_disk_mb : (getenv("PF_PC_DISK_MB") ? atoi(getenv("PF_PC_DISK_MB")) : 1024);
        size_t ram_bytes = (size_t)std::max(ram_mb, 0) * 1024 * 1024;
        size_t disk_bytes = (size_t)std::max(disk_mb, 0) * 1024 * 1024;
        size_t vram_bytes = (size_t)std::max(states, 0) * per_node;
        // The three tiers are one KV budget: an explicit device-pool ceiling
        // (--kv-cap-mb / PF_KV_CAP_MB) bounds their sum, shrinking disk first,
        // then RAM, then the VRAM checkpoint count.  Without one the three
        // configured tiers define the budget themselves.
        bool tiers_clamped = false;
        if (pc_enabled && kv_cap_mb > 0) {
            const size_t cap = (size_t)kv_cap_mb * 1024 * 1024;
            const size_t total = vram_bytes + ram_bytes + disk_bytes;
            if (total > cap) {
                tiers_clamped = true;
                size_t over = total - cap;
                const size_t cut_disk = std::min(over, disk_bytes);
                disk_bytes -= cut_disk;
                over -= cut_disk;
                const size_t cut_ram = std::min(over, ram_bytes);
                ram_bytes -= cut_ram;
                over -= cut_ram;
                if (over > 0) {
                    vram_bytes = vram_bytes > over ? vram_bytes - over : 0;
                    states = per_node > 0 ? (int)(vram_bytes / per_node) : 0;
                    vram_bytes = (size_t)std::max(states, 0) * per_node;
                }
                fprintf(stderr,
                        "[pc] tiers clamped to --kv-cap-mb %d: vram=%d checkpoints, ram=%.0f MB, disk=%.0f MB\n",
                        kv_cap_mb, states, (double)ram_bytes / (1024.0 * 1024.0), (double)disk_bytes / (1024.0 * 1024.0));
            }
        }
        pc_max_states = states;
        pc_vram_bytes = vram_bytes;
        pc_ram_bytes = ram_bytes;
        pc_disk_bytes = disk_bytes;
        // no checkpoints -> nothing to resume from, so the whole cache is off
        if (pc_max_states == 0) {
            pc_enabled = false;
        }
        // RAM tier: on by default with the cache (a cheap staging tier between
        // the small VRAM pool and disk); --pc-ram-mb / PF_PC_RAM_MB, 0 = off.
        if (pc_enabled && pc_ram_bytes > 0) {
            pc_ram_init();
        }
        // disk tier: directory from the flag, else PF_PC_DIR.  A disk budget
        // clamped to zero by the KV cap disables the tier (instead of the usual
        // "0 = unbounded" meaning).
        if (!pc_dir_arg.empty()) {
            pc_dir = pc_dir_arg;
        } else if (const char * ed = getenv("PF_PC_DIR")) {
            pc_dir = ed;
        }
        if (tiers_clamped && pc_disk_bytes == 0) {
            pc_dir.clear();
        }
        if (pc_enabled && !pc_dir.empty()) {
            pc_disk_init();
        }
    }
    // multi-device: md_int8 (set in setup_multi_device) builds per-device SIn
    // copies and skips the raw upload; oneDNN is then not needed (the dp4a
    // GEMMs run on the w8 views) and would duplicate the int8 weights.
    // oneDNN int8 matmul for the prefill GEMMs (PF_GEMM_DNNL, default on).
    // The converted weights live in device USM and are built once here;
    // PF_GEMM_DNNL=0 keeps the dp4a/fp32 path bit-identical to before.
    use_dnnl = !cpu_mode && !multi_dev && pf8 && dnnl_gemm_enabled();
    if (use_dnnl) {
        dnnl = std::make_unique<dnnl_gemm>(q);
        auto add = [&](const wt & t, const w8t & w8) {
            if (w8.vals) {
                dnnl->add_weight(w8.vals, t.data, t.type, t.K, t.N);
            }
        };
        for (auto & L : m.layers) {
            add(L.ffn_gate, L.ffn_gate8);
            add(L.ffn_up, L.ffn_up8);
            add(L.ffn_down, L.ffn_down8);
            if (L.recurrent) {
                add(L.wqkv, L.wqkv8);
                add(L.wgate, L.wgate8);
                add(L.ssm_out, L.ssm_out8);
            } else {
                add(L.wq, L.wq8);
                add(L.wk, L.wk8);
                add(L.wv, L.wv8);
                add(L.wo, L.wo8);
            }
        }
        // the LM head runs on the single-token dp4a GEMV in mode 2 (no conversion)
        // pre-execute every cached primitive once so the first real request does
        // not pay the one-time kernel load (measured as first-vs-later ttfr)
        const char * envw = getenv("PF_DNNL_NOWARM");
        if (!(envw && atoi(envw) != 0)) {
            dnnl->warmup();
        }
    } else if (multi_dev && !md_int8 && !md_xmx && dnnl_gemm_enabled() &&
               [&] {
                   const char * en = getenv("PF_DP4A");
                   return !(en && atoi(en) == 0); // PF_DP4A=0 forces the fp32 path
               }()) {
        // one dnnl_gemm per GPU backend, holding only the layers that device
        // computes (the CPU partitions keep the fp32/i8 host path).  Every
        // prefill GEMM tensor of a GPU layer is converted - including the
        // small ssm_beta/ssm_alpha, which ride in the wqkv call.
        const char * envw = getenv("PF_DNNL_NOWARM");
        const bool nowarm = envw && atoi(envw) != 0;
        dnnl_dev_.resize(backends_.size());
        for (size_t d = 0; d < backends_.size(); d++) {
            if (dev_kind_[d] != 0) {
                continue; // CPU backend: no oneDNN
            }
            auto D = std::make_unique<dnnl_gemm>(*dev_queues_[d]);
            auto add = [&](const wt & t) {
                if (!t.data) {
                    return; // attention layers have no ssm_beta/ssm_alpha
                }
                // key the conversion on the per-device SIn copy when the decode
                // path builds one - the plan segments then carry the same
                // w8.vals, so oneDNN and dp4a agree on the key.  Tensors without
                // a w8 view (ssm_beta/ssm_alpha) key on the fp32 device pointer.
                const void * key = nullptr;
                if (w8_dev_.size()) {
                    auto it = w8_dev_[(size_t)d].find(t.data);
                    if (it != w8_dev_[(size_t)d].end()) {
                        key = it->second.vals;
                    }
                }
                if (!key) {
                    key = wptr((int)d, t.data); // == the segments' sj.w
                }
                if (key) {
                    D->add_weight(key, t.data, t.type, t.K, t.N);
                }
            };
            for (int il = 0; il < m.hp.n_layer; il++) {
                if (layer_dev_[(size_t)il] != (int)d) {
                    continue;
                }
                const layer_t & L = m.layers[il];
                add(L.ffn_gate);
                add(L.ffn_up);
                add(L.ffn_down);
                add(L.ssm_beta);
                add(L.ssm_alpha);
                if (L.recurrent) {
                    add(L.wqkv);
                    add(L.wgate);
                    add(L.ssm_out);
                } else {
                    add(L.wq);
                    add(L.wk);
                    add(L.wv);
                    add(L.wo);
                }
            }
            if (!nowarm) {
                D->warmup();
            }
            dnnl_dev_[d] = std::move(D);
        }
    }
    ffn_stride = 2 * m.hp.n_ff;
    max_blocks = (max_seq + kBlockSize - 1) / kBlockSize;
    // dynamic pool: `n_blocks_` is the initial committed size, the cap is the
    // virtual address reservation (--kv-cap-mb / PF_KV_CAP_MB, default: no
    // growth).  kv_layer_stride is built from the *reservation*, so the
    // per-layer base pointers recorded in the graphs stay valid when the pool
    // grows.
    {
        int n_attn = 0;
        for (int il = 0; il < m.hp.n_layer; il++) {
            n_attn += !m.hp.is_recr(il);
        }
        const size_t block_bytes = kv_block_bytes();
        // kv_cap_mb < 0 = auto: reserve exactly what max_seq needs (the pool
        // still commits memory lazily in extents, so the extra range is only
        // virtual address space)
        const int cap_blocks = kv_cap_mb > 0
                                   ? (int)(((int64_t)kv_cap_mb * 1024 * 1024) / ((int64_t)block_bytes * n_attn))
                                   : (kv_cap_mb < 0 ? max_blocks : 0);
        pool_cap = std::max(n_blocks_, cap_blocks);
        // the VRAM cache tier lives in this pool: the reservation must be able
        // to hold at least one KV block per cached checkpoint (the pool still
        // commits memory lazily, so this is address space, not committed RAM)
        if (pc_enabled) {
            pool_cap = std::max(pool_cap, pc_max_states);
        }
        pool_initial = n_blocks_;
    }
    {
        const char * eg = getenv("PF_KV_GROW");
        if (eg) {
            pool_chunk = atoi(eg);
        }
        if (pool_chunk < 1) {
            pool_chunk = 1;
        }
    }
    n_blocks = pool_cap;
    alloc_buffers();
    build_graphs();
    // all weights are on their devices now: release the host pages of the GGUF
    // so the process does not carry the whole file in its RSS
    release_host_weight_pages();
}

engine::~engine() {
    // graceful shutdown: persist the VRAM/RAM tiers to disk (runs while the
    // device pools and checkpoint slots are still alive)
    pc_flush_to_disk();
    // release device/USM allocations (tests construct engines in loops)
    auto f = [&](auto * p) {
        if (p) {
            sycl::free(p, q);
        }
    };
    // per-device activation sets own their allocations; clear the members first
    // so the single-device frees below do not double-free
    if (!as_.empty()) {
        for (size_t d = 0; d < as_.size(); d++) {
            sycl::queue & qd = (d < dev_queues_.size() && dev_queues_[d] && dev_kind_[d] == 0) ? *dev_queues_[d] : q;
            auto fa = [&](auto * p) {
                if (p) {
                    sycl::free(p, qd);
                }
            };
            const act_set & a = as_[d];
            fa(a.x);
            fa(a.xnorm);
            fa(a.qkv);
            fa(a.z);
            fa(a.beta);
            fa(a.alpha);
            fa(a.conv_out);
            fa(a.attn_pre);
            fa(a.attn_merged);
            fa(a.qbuf);
            fa(a.kbuf);
            fa(a.vbuf);
            fa(a.attn_out);
            fa(a.ffn);
            fa(a.partials);
            fa(a.partials_dec);
            sycl::free(a.x8, qd);
            sycl::free(a.xmeta, qd);
            sycl::free(a.xsumq, qd);
            fa(a.gdn_state);
            fa(a.conv_state);
        }
        as_.clear();
        d_x = d_xnorm = d_qkv = d_z = d_beta = d_alpha = d_conv_out = nullptr;
        d_attn_pre = d_attn_merged = d_qbuf = d_kbuf = d_vbuf = d_attn_out = nullptr;
        d_ffn = d_partials = d_partials_dec = nullptr;
        d_x8 = nullptr;
        d_xmeta = nullptr;
        d_xsumq = nullptr;
        d_gdn_state = nullptr;
        d_conv_state = nullptr;
    }
    if (h_handoff) {
        sycl::free(h_handoff, q);
        h_handoff = nullptr;
    }
    f(d_tables);
    f(d_info);
    f(d_segs_dec);
    f(d_segs_dec8);
    f(d_segs_pf);
    f(d_segs_pf8);
    f(d_segs_pf8_nh);
    for (int i = 0; i < kPfSlots; i++) {
        f(d_segs_pf_slot[i]);
        f(d_segs_pf_nh_slot[i]);
    }
    f(d_segs_aux);
    f(d_segs_pfb);
    for (auto & b : buckets_) {
        f(b.d_segs);
    }
    f(d_x8);
    f(d_xmeta);
    f(d_xsumq);
    f(d_x);
    f(d_xnorm);
    f(d_qkv);
    f(d_z);
    f(d_beta);
    f(d_alpha);
    f(d_conv_out);
    f(d_attn_pre);
    f(d_attn_merged);
    f(d_qbuf);
    f(d_kbuf);
    f(d_vbuf);
    f(d_attn_out);
    f(d_ffn);
    f(d_partials);
    f(d_partials_dec);
    f(d_logits);
    f(d_last_hidden);
    f(d_gdn_state);
    f(d_conv_state);
    f(d_pc_states);
    pool_print("exit");
    pc_print_stats("exit");
    if (pcd) {
        pcd->print_stats("exit");
    }
    if (pcr) {
        pcr->print_stats("exit");
    }
    kv_release_pool();
    m.free_w8(q);
    // multi-device per-device SIn copies: free on each device's own queue
    for (size_t d = 0; d < w8_dev_.size(); d++) {
        for (auto & kv : w8_dev_[d]) {
            if (kv.second.vals && d < dev_queues_.size() && dev_queues_[d]) {
                sycl::free(kv.second.vals, *dev_queues_[d]);
            }
            if (kv.second.meta && d < dev_queues_.size() && dev_queues_[d]) {
                sycl::free(kv.second.meta, *dev_queues_[d]);
            }
        }
    }
    for (size_t d = 0; d < weight_maps_.size(); d++) {
        // the partition was allocated on the owning device's queue
        sycl::queue & qd = (multi_dev && d < dev_queues_.size() && dev_queues_[d]) ? *dev_queues_[d] : q;
        for (auto & kv : weight_maps_[d]) {
            sycl::free(kv.second, qd);
        }
    }
    weight_maps_.clear();
    if (m.dev_weights) {
        sycl::free(m.dev_weights, q);
    }
    if (h_logits) {
        free(h_logits);
    }
}

// Pre-extract the per-32-value (scale, min) pairs of Q4_K/Q5_K tensors into an
// fp32 side array.  The generic GEMV decodes the packed 6-bit scales on every
// pass (a large part of its instruction count); the side array turns that into
// one float2 load.
void engine::build_meta32(sycl::queue & q) {
    auto add = [&](const wt & t) {
        if (t.type != 12 && t.type != 13) {
            return; // Q4_K / Q5_K only
        }
        const int K = t.K, N = t.N;
        const int nsb = K / 32;
        sycl::float2 * dev = sycl::malloc_device<sycl::float2>((size_t)N * nsb, q);
        const int slab = 4096;
        std::vector<sycl::float2> h((size_t)slab * nsb);
        const size_t row_bytes = quant_row_bytes(t.type, K);
        const size_t blk_bytes = (t.type == 12) ? 144 : 176;
        for (int r0 = 0; r0 < N; r0 += slab) {
            const int nr = std::min(slab, N - r0);
            for (int r = 0; r < nr; r++) {
                const char * row = (const char *)t.data + (size_t)(r0 + r) * row_bytes;
                for (int xb = 0; xb < K / 256; xb++) {
                    const auto * blk = (const block_q4_K *)(row + (size_t)xb * blk_bytes);
                    const float d = ggml_half_to_float(blk->d);
                    const float dmin = ggml_half_to_float(blk->dmin);
                    for (int sb = 0; sb < 8; sb++) {
                        uint8_t sc, mm;
                        get_scale_min_k4(sb, blk->scales, &sc, &mm);
                        h[(size_t)r * nsb + xb * 8 + sb] = sycl::float2(d * (float)sc, dmin * (float)mm);
                    }
                }
            }
            q.memcpy(dev + (size_t)r0 * nsb, h.data(), (size_t)nr * nsb * sizeof(sycl::float2)).wait();
        }
        meta32_[t.data] = dev;
    };
    add(m.tok_embd);
    for (auto & L : m.layers) {
        add(L.ffn_gate);
        add(L.ffn_up);
        add(L.ffn_down);
        add(L.wqkv);
        add(L.wgate);
        add(L.ssm_out);
        add(L.wq);
        add(L.wk);
        add(L.wv);
        add(L.wo);
    }
}

void * engine::alloc_bytes(size_t bytes) {
    const bool host = cpu_mode || host_act;
    void * p = host ? sycl::malloc_host(bytes, q) : sycl::malloc_device(bytes, q);
    if (!p) {
        throw std::runtime_error("device alloc failed");
    }
    return p;
}

void * engine::dev_alloc_on(int dev, size_t bytes) {
    if (!multi_dev) {
        return alloc_bytes(bytes);
    }
    void * p = dev_kind_[(size_t)dev] == 1 ? sycl::malloc_host(bytes, q)
                                           : sycl::malloc_device(bytes, *dev_queues_[(size_t)dev]);
    if (!p) {
        throw std::runtime_error("device alloc failed");
    }
    return p;
}

engine::act_set engine::alloc_act_set(int dev) {
    const hparams & hp = m.hp;
    const int R = kMaxB * kMaxT;
    auto A = [&](size_t n) { return (float *)dev_alloc_on(dev, n * sizeof(float)); };
    act_set a;
    a.x = A((size_t)R * hp.n_embd);
    a.xnorm = A((size_t)R * hp.n_embd);
    a.qkv = A((size_t)R * hp.qkv_dim());
    a.z = A((size_t)R * hp.d_inner);
    a.beta = A((size_t)R * hp.dt_rank);
    a.alpha = A((size_t)R * hp.dt_rank);
    a.conv_out = A((size_t)R * hp.qkv_dim());
    a.attn_pre = A((size_t)R * hp.d_inner);
    a.attn_merged = A((size_t)R * hp.d_inner);
    a.qbuf = A((size_t)R * hp.n_head * 2 * hp.head_dim);
    a.kbuf = A((size_t)R * hp.n_head_kv * hp.head_dim);
    a.vbuf = A((size_t)R * hp.n_head_kv * hp.head_dim);
    a.attn_out = A((size_t)R * hp.n_head * hp.head_dim);
    a.ffn = A((size_t)R * ffn_stride);
    a.partials = A((size_t)R * hp.n_head * n_splits * (2 + hp.head_dim));
    a.partials_dec = A((size_t)kMaxB * hp.n_head * dec_splits * (2 + hp.head_dim));
    const int maxK = std::max({hp.n_embd, hp.qkv_dim() / 16 * 16, hp.d_inner, hp.n_head * hp.head_dim, hp.n_ff});
    const size_t xrows = (size_t)kMaxB * kMaxT;
    a.x8 = (int8_t *)dev_alloc_on(dev, xrows * maxK);
    a.xmeta = (sycl::float2 *)dev_alloc_on(dev, xrows * (maxK / 32) * sizeof(sycl::float2));
    a.xsumq = (int32_t *)dev_alloc_on(dev, xrows * (maxK / 16) * sizeof(int32_t));
    // recurrent state for this partition's own GDN layers (local indexing)
    const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state;
    const size_t conv_per = (size_t)(hp.conv_k - 1) * hp.qkv_dim();
    const int ng = n_gdn_dev_[(size_t)dev];
    a.gdn_state = (float *)dev_alloc_on(dev, (size_t)kMaxB * ng * gdn_per * sizeof(float));
    a.conv_state = (float *)dev_alloc_on(dev, (size_t)kMaxB * ng * conv_per * sizeof(float));
    return a;
}

void engine::bind_acts(int dev) {
    if (as_.empty()) {
        return;
    }
    const act_set & a = as_[(size_t)dev];
    d_x = a.x;
    d_xnorm = a.xnorm;
    d_qkv = a.qkv;
    d_z = a.z;
    d_beta = a.beta;
    d_alpha = a.alpha;
    d_conv_out = a.conv_out;
    d_attn_pre = a.attn_pre;
    d_attn_merged = a.attn_merged;
    d_qbuf = a.qbuf;
    d_kbuf = a.kbuf;
    d_vbuf = a.vbuf;
    d_attn_out = a.attn_out;
    d_ffn = a.ffn;
    d_partials = a.partials;
    d_partials_dec = a.partials_dec;
    d_x8 = a.x8;
    d_xmeta = a.xmeta;
    d_xsumq = a.xsumq;
    d_gdn_state = a.gdn_state;
    d_conv_state = a.conv_state;
}

void engine::handoff_x(int from, int to, size_t rows) {
    if (from == to) {
        return;
    }
    const size_t cap = (size_t)kMaxB * kMaxT;
    if (rows > cap) {
        rows = cap;
    }
    if (rows == 0) {
        return;
    }
    const size_t bytes = rows * m.hp.n_embd * sizeof(float);
    if (!h_handoff) {
        h_handoff = sycl::malloc_host(cap * m.hp.n_embd * sizeof(float), q);
        if (!h_handoff) {
            throw std::runtime_error("handoff staging allocation failed");
        }
    }
    sycl::queue & qf = dev_queue(from);
    // wait for the producing device's enqueued layers, then stage through host
    qf.wait();
    qf.memcpy(h_handoff, as_[(size_t)from].x, bytes).wait();
    // the destination copies before its own kernels (in-order queue)
    dev_queue(to).memcpy(as_[(size_t)to].x, h_handoff, bytes);
}

void engine::sync_all() {
    if (multi_dev) {
        for (auto & b : backends_) {
            b->synchronize();
        }
        return;
    }
    backend().synchronize();
}

const void * engine::wptr(int dev, const void * host) const {
    if (!multi_dev) {
        return m.dev_ptr(host);
    }
    const auto & mp = weight_maps_[(size_t)dev];
    if (mp.empty()) {
        return host; // CPU partition: weights stay in the host mmap
    }
    const auto it = mp.find(host);
    if (it == mp.end()) {
        throw std::runtime_error("multi-device: weight tensor not uploaded to this device's partition");
    }
    return it->second;
}

// Upload only the tensors of the layers placed on `dev` (plus the global
// tok_embd / output_norm, which always run on the primary device) so the GPU
// holds just its partition of the weights instead of a whole-file copy.  Each
// tensor gets its own device allocation; wptr() resolves host -> device through
// this map, so the CPU-side layers' weights never occupy GPU memory.
void engine::upload_device_weights(int dev) {
    auto & mp = weight_maps_[(size_t)dev];
    std::unordered_set<const void *> need;
    auto addp = [&need](const void * p) {
        if (p) {
            need.insert(p);
        }
    };
    // global tensors (embedding, LM head, output norm) always run on the
    // primary device: only device 0 carries them, which matters because
    // tok_embd + output are ~10 GB for a large-vocab model
    if (dev == 0) {
        addp(m.tok_embd.data);
        addp(m.output.data);
        addp(m.output_norm);
    }
    int needed_by_layer = 0;
    for (int il = 0; il < m.hp.n_layer; il++) {
        if (layer_dev_[(size_t)il] != dev) {
            continue;
        }
        needed_by_layer++;
        const layer_t & L = m.layers[(size_t)il];
        addp(L.attn_norm);
        addp(L.post_attn_norm);
        addp(L.ffn_gate.data);
        addp(L.ffn_up.data);
        addp(L.ffn_down.data);
        if (L.recurrent) {
            addp(L.wqkv.data);
            addp(L.wgate.data);
            addp(L.ssm_beta.data);
            addp(L.ssm_alpha.data);
            addp(L.ssm_out.data);
            addp(L.ssm_a);
            addp(L.ssm_dt);
            addp(L.ssm_norm);
            addp(L.ssm_conv1d);
        } else {
            addp(L.wq.data);
            addp(L.wk.data);
            addp(L.wv.data);
            addp(L.wo.data);
            addp(L.q_norm);
            addp(L.k_norm);
        }
    }
    size_t bytes = 0;
    size_t n = 0;
    sycl::queue & qd = multi_dev ? *dev_queues_[(size_t)dev] : q;
    for (const auto & t : m.gguf.tensors) {
        if (!need.count(t.data)) {
            continue; // a CPU-side layer's weight: stays in the host mmap only
        }
        if (dev < (int)w8_dev_.size() && w8_dev_[(size_t)dev].count(t.data)) {
            // the int8 (SIn) view replaces the raw device copy
            continue;
        }
        if (dev < (int)dnnl_dev_.size() && dnnl_dev_[(size_t)dev]
            && (dnnl_dev_[(size_t)dev]->has_weight(t.data) || dnnl_dev_[(size_t)dev]->has_weight_w4(t.data)
                || dnnl_dev_[(size_t)dev]->has_weight_k5(t.data) || dnnl_dev_[(size_t)dev]->has_weight_cb4(t.data))) {
            // the oneDNN int8 (XMX) copy - or the u4 (4-bit) copy - replaces the
            // raw device copy.  This must cover the 4-bit path too: otherwise the
            // raw fp32 weight gets uploaded, wkey() starts returning the device
            // pointer, has_weight_w4(key) no longer matches and the plan silently
            // drops the call to the fp32 gemv_group path.
            continue;
        }
        // The embedding table is only ever read as a few rows per step (one row
        // per token, 20 KB for this model), so it lives in *host* USM instead of
        // device memory: host USM is device-accessible through the shared
        // multi-device context, the PCIe traffic is ~3.5 us/token, and it frees
        // ~0.7 GB of device memory per card for the KV cache.
        void * g = nullptr;
        if (dev == 0 && t.data == m.tok_embd.data) {
            g = sycl::malloc_host(t.nbytes(), qd);
        } else {
            g = sycl::malloc_device(t.nbytes(), qd);
        }
        if (!g) {
            throw std::runtime_error("multi-device weight upload failed (out of device memory)");
        }
        qd.memcpy(g, t.data, t.nbytes()).wait();
        mp[t.data] = g;
        bytes += t.nbytes();
        n++;
        // the raw device copy is complete and this is a GPU partition's tensor,
        // so the host pages are dead: drop them now to keep the load peak low
        // (the final release_host_weight_pages() covers anything left over)
        m.page_out_host(t.data, t.nbytes());

    }
    fprintf(stderr, "[dev] device %d: uploaded %zu tensors (%d layers), %.1f MB\n", dev, n, needed_by_layer,
            (double)bytes / (1024.0 * 1024.0));
}

const float * engine::wf32(int dev, const float * host) const {
    return (const float *)wptr(dev, host);
}

// Once every weight tensor has been copied (or converted) onto its device, the
// host mmap is dead weight for a GPU-only run: the 27B keeps 15.7 GB of file
// pages resident in the process RSS forever even though no kernel reads them
// from the host.  Drop those pages with MADV_DONTNEED.  The mapping stays valid
// (plain device-pointer arithmetic still uses gguf.map_base), so a stray host
// read just re-faults from the file.  CPU partitions are the exception: their
// host kernels read the mmap on every token, so their tensors' pages stay.
void engine::release_host_weight_pages() {
    if (!m.gguf.map_base || cpu_mode) {
        return;
    }
    std::unordered_set<const void *> keep;
    if (multi_dev) {
        for (int il = 0; il < m.hp.n_layer; il++) {
            const int d = layer_dev_[(size_t)il];
            if (d < 0 || d >= (int)dev_kind_.size() || dev_kind_[(size_t)d] != 1) {
                continue; // GPU (or unassigned) layer: device-only
            }
            const layer_t & L = m.layers[(size_t)il];
            auto kp = [&](const void * p) {
                if (p) {
                    keep.insert(p);
                }
            };
            kp(L.attn_norm);
            kp(L.post_attn_norm);
            kp(L.ffn_gate.data);
            kp(L.ffn_up.data);
            kp(L.ffn_down.data);
            if (L.recurrent) {
                kp(L.wqkv.data);
                kp(L.wgate.data);
                kp(L.ssm_beta.data);
                kp(L.ssm_alpha.data);
                kp(L.ssm_out.data);
                kp(L.ssm_a);
                kp(L.ssm_dt);
                kp(L.ssm_norm);
                kp(L.ssm_conv1d);
            } else {
                kp(L.wq.data);
                kp(L.wk.data);
                kp(L.wv.data);
                kp(L.wo.data);
                kp(L.q_norm);
                kp(L.k_norm);
            }
        }
    }
    size_t dropped = 0;
    for (const auto & t : m.gguf.tensors) {
        if (keep.count(t.data)) {
            continue;
        }
        m.page_out_host(t.data, t.nbytes());
        dropped += t.nbytes();
    }
    if (keep.empty()) {
        // no host consumer at all: the header (already parsed into the kv map /
        // token strings) is dead too
        m.page_out_host(m.gguf.map_base, m.gguf.data_offset);
    }
    fprintf(stderr, "[mem] released %.1f MB of host-resident GGUF pages (RSS)\n", (double)dropped / (1024.0 * 1024.0));
}

// ---------------------------------------------------------------------------
// Multi-device layer placement (pipeline parallel).  `layer_map` is a
// comma-separated list of `begin-end:device` ranges covering [0, n_layer);
// `device` is `gpu[.N]` (N = SYCL GPU index, default 0) or `cpu`.  The ranges
// execute in order, each device computing its contiguous layer range and handing
// the hidden state to the next at the partition boundary; each device gets its
// own weights and its own paged KV pool (same global block ids), so every
// attention layer's KV is stored on the device that computes it.
// Build the per-device oneDNN int8 weights for the XMX prefill/decode path.
// Keyed by the host weight pointer: upload_device_weights then skips the raw
// device copy of a converted tensor, so the partition's linears live once as
// int8.  Returns false if no device can run the int8 matmul (caller falls back
// to the SIn/dp4a path).
bool engine::setup_md_dnnl() {
    if (!dnnl_gemm_enabled()) {
        return false;
    }
    if (const char * en = getenv("PF_DP4A"); en && atoi(en) == 0) {
        return false; // PF_DP4A=0 forces the fp32 path
    }
    dnnl_dev_.clear();
    dnnl_dev_.resize(backends_.size());
    bool any = false;
    for (size_t d = 0; d < backends_.size(); d++) {
        if (dev_kind_[d] != 0 || !dev_queues_[d]) {
            continue; // CPU partitions stay on the host
        }
        auto D = std::make_unique<dnnl_gemm>(*dev_queues_[d]);
        bool dev_ok = false;
        // PF_W4: represent the types that have a native-width 4-bit packing
        // (Q4_K) as u4 + per-group f16 step/offset instead of int8.  Measured
        // lossless within the f16 metadata (0.039% GEMM error vs int8's 0.98%)
        // and 0.625 B/weight instead of 1.0.  Unsupported types keep int8.
        // Default on: the u4 path is verified against an fp32 reference (0.037%
        // vs the int8 conversion's ~0.5-1%, test_w4_gemm) and is exact in the
        // sense that it keeps Q4_K's native per-32 grid instead of re-quantizing
        // onto a per-row one.  It saves 1.4 GB/card of weight memory for ~5%
        // decode and ~3% prefill throughput.  PF_W4=0 restores pure int8.
        static const bool w4_on = [] {
            const char * e = getenv("PF_W4");
            return !e || atoi(e) != 0;
        }();
        int n_w4 = 0, n_i8 = 0, n_cb = 0, n_k5 = 0;
        // PF_K5=0 keeps Q5_K on the int8 conversion (A/B knob)
        static const bool add_k5 = [] {
            const char * e = getenv("PF_K5");
            return !e || atoi(e) != 0;
        }();
        // PF_CB4=0 keeps IQ4_XS/IQ4_NL on the int8 conversion (A/B knob)
        static const bool add_cb = [] {
            const char * e = getenv("PF_CB4");
            return !e || atoi(e) != 0;
        }();
        auto add = [&](const wt & t) {
            if (!t.data) {
                return;
            }
            bool converted = false;
            if (w4_on && D->add_weight_w4(t.data, t.data, t.type, t.K, t.N)) {
                dev_ok = true;
                n_w4++;
                converted = true;
            }
            // PF_K5=0 keeps Q5_K on the int8 conversion (A/B knob)
            else if (add_k5 && D->add_weight_k5(t.data, t.data, t.type, t.K, t.N)) {
                dev_ok = true;
                n_k5++;
                converted = true;
            } else if (add_cb && D->add_weight_cb4(t.data, t.data, t.type, t.K, t.N)) {
                dev_ok = true;
                n_cb++;
                converted = true;
            } else if (D->add_weight(t.data, t.data, t.type, t.K, t.N)) {
                dev_ok = true;
                n_i8++;
                converted = true;
            }
            if (converted) {
                // the host read is done and the tensor lives on the device now,
                // so drop its file pages immediately: this keeps the load-time
                // peak at one tensor's worth instead of the whole GGUF (the
                // upload pass below skips converted tensors, so nothing re-reads
                // them from the host)
                m.page_out_tensor(t);
            }
        };
        if (d == 0) {
            // The LM head is a global pinned to backend 0.  Convert it here -
            // *before* upload_device_weights - keyed by its host pointer, which
            // makes the upload skip its raw 1.0 GB device copy exactly like the
            // converted layer tensors.  gemv_at then needs its oneDNN branch for
            // the head in *both* phases (build_plan keys the segment through
            // wkey(), which returns this host key); see the single-token
            // relaxation in record_forward.
            add(m.output);
        }
        for (int il = 0; il < m.hp.n_layer; il++) {
            if (layer_dev_[(size_t)il] != (int)d) {
                continue;
            }
            const layer_t & L = m.layers[(size_t)il];
            add(L.ffn_gate);
            add(L.ffn_up);
            add(L.ffn_down);
            add(L.ssm_beta);
            add(L.ssm_alpha);
            if (L.recurrent) {
                add(L.wqkv);
                add(L.wgate);
                add(L.ssm_out);
            } else {
                add(L.wq);
                add(L.wk);
                add(L.wv);
                add(L.wo);
            }
        }
        if (dev_ok) {
            fprintf(stderr, "[dev] device %zu weights: %d u4, %d k5, %d codebook, %d int8, %.1f MiB on device\n", d,
                    n_w4, n_k5, n_cb, n_i8, (double)D->weight_bytes() / (1024.0 * 1024.0));
            const char * envw = getenv("PF_DNNL_NOWARM");
            if (!(envw && atoi(envw) != 0)) {
                D->warmup();
            }
            dnnl_dev_[d] = std::move(D);
            any = true;
        } else {
            fprintf(stderr, "[dev] oneDNN: device %zu has no convertible layer weights\n", d);
        }
    }
    fprintf(stderr, "[dev] oneDNN multi-device: %s\n", any ? "enabled" : "unavailable (dp4a fallback)");
    if (!any) {
        dnnl_dev_.clear();
    }
    return any;
}

void engine::setup_multi_device(const std::string & layer_map) {
    struct ent {
        int l0, l1, kind, dev_idx;
    };
    std::vector<ent> ents;
    size_t pos = 0;
    while (pos < layer_map.size()) {
        size_t comma = layer_map.find(',', pos);
        std::string tok = layer_map.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? layer_map.size() : comma + 1;
        if (tok.empty()) {
            continue;
        }
        size_t colon = tok.find(':');
        if (colon == std::string::npos) {
            throw std::runtime_error("bad --layer-map entry (expect begin-end:device): " + tok);
        }
        const std::string rng = tok.substr(0, colon);
        const std::string dev = tok.substr(colon + 1);
        size_t dash = rng.find('-');
        if (dash == std::string::npos) {
            throw std::runtime_error("bad --layer-map range: " + rng);
        }
        ent e;
        e.l0 = atoi(rng.substr(0, dash).c_str());
        // `begin-end` is an inclusive layer range
        e.l1 = atoi(rng.substr(dash + 1).c_str()) + 1;
        if (dev == "cpu" || dev == "host" || dev == "1") {
            e.kind = 1;
            e.dev_idx = -1;
        } else if (dev.rfind("gpu.", 0) == 0) {
            e.kind = 0;
            e.dev_idx = atoi(dev.substr(4).c_str());
        } else if (dev == "gpu") {
            e.kind = 0;
            e.dev_idx = 0;
        } else {
            throw std::runtime_error("bad --layer-map device (expect gpu[.N] or cpu): " + dev);
        }
        if (e.l0 < 0 || e.l1 <= e.l0) {
            throw std::runtime_error("bad --layer-map range: " + rng);
        }
        ents.push_back(e);
    }
    if (ents.empty()) {
        throw std::runtime_error("empty --layer-map");
    }
    // require a contiguous cover of [0, n_layer) in order
    int expect = 0;
    for (const ent & e : ents) {
        if (e.l0 != expect) {
            throw std::runtime_error("--layer-map must cover every layer without gaps");
        }
        expect = e.l1;
    }
    if (expect != m.hp.n_layer) {
        throw std::runtime_error("--layer-map must end at the model's layer count");
    }

    // collect unique GPU device indices used, in sorted order
    std::vector<int> gpu_indices;
    for (const ent & e : ents) {
        if (e.kind == 0 && e.dev_idx >= 0) {
            if (std::find(gpu_indices.begin(), gpu_indices.end(), e.dev_idx) == gpu_indices.end()) {
                gpu_indices.push_back(e.dev_idx);
            }
        }
    }
    std::sort(gpu_indices.begin(), gpu_indices.end());
    const bool has_gpu = !gpu_indices.empty();
    const bool has_cpu = std::any_of(ents.begin(), ents.end(), [](const ent & e) { return e.kind == 1; });

    // map from physical gpu index -> backend index
    std::unordered_map<int, int> gpu_backend_idx;
    for (size_t i = 0; i < gpu_indices.size(); i++) {
        gpu_backend_idx[gpu_indices[(size_t)i]] = (int)i;
    }
    const int cpu_backend_idx = has_cpu ? (int)gpu_indices.size() : -1;
    const int ndev = (int)gpu_indices.size() + (has_cpu ? 1 : 0);
    backends_.resize((size_t)ndev);
    dev_kind_.assign((size_t)ndev, 0);
    weight_maps_.assign((size_t)ndev, {});
    dev_queues_.assign((size_t)ndev, nullptr);
    owned_queues_.clear();
    owned_queues_.reserve((size_t)gpu_indices.size());

    {
        std::vector<sycl::device> all_gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
        for (size_t i = 0; i < gpu_indices.size(); i++) {
            const int phys = gpu_indices[(size_t)i];
            const int bidx = (int)i;
            dev_kind_[(size_t)bidx] = 0;
            sycl::device target;
            if ((size_t)phys < all_gpus.size()) {
                target = all_gpus[(size_t)phys];
            } else {
                target = sycl::device(sycl::gpu_selector_v);
            }
            if (md_ctx_) {
                // all GPU partitions share the primary context
                owned_queues_.push_back(
                    std::make_unique<sycl::queue>(*md_ctx_, target, sycl::property::queue::in_order()));
            } else {
                owned_queues_.push_back(std::make_unique<sycl::queue>(target, sycl::property::queue::in_order()));
            }
            dev_queues_[(size_t)bidx] = owned_queues_.back().get();
            backends_[(size_t)bidx] = make_gpu_backend(*dev_queues_[(size_t)bidx]);
            fprintf(stderr, "[dev] backend %d -> gpu.%d (%s)\n", bidx, phys,
                    target.get_info<sycl::info::device::name>().c_str());
        }
    }
    if (cpu_backend_idx >= 0) {
        dev_kind_[(size_t)cpu_backend_idx] = 1;
        backends_[(size_t)cpu_backend_idx] = make_cpu_backend();
    }

    layer_dev_.assign((size_t)m.hp.n_layer, has_gpu ? 0 : 0);
    layer_attn_local_.assign((size_t)m.hp.n_layer, -1);
    layer_gdn_local_.assign((size_t)m.hp.n_layer, -1);
    n_gdn_dev_.assign((size_t)ndev, 0);
    std::vector<int> attn_local((size_t)ndev, 0);
    for (const ent & e : ents) {
        const int dev = e.kind == 1 ? cpu_backend_idx : gpu_backend_idx[e.dev_idx];
        for (int il = e.l0; il < e.l1; il++) {
            layer_dev_[(size_t)il] = dev;
            if (m.hp.is_recr(il)) {
                layer_gdn_local_[(size_t)il] = n_gdn_dev_[(size_t)dev]++;
            } else {
                layer_attn_local_[(size_t)il] = attn_local[(size_t)dev]++;
            }
        }
    }
    // Multi-device state must be set before the upload: upload_device_weights
    // chooses the owning device's queue from `multi_dev`, and wptr()/the plan
    // builders consult the flags.  (Previously this was set after the upload,
    // so every partition was allocated on the primary GPU.)
    multi_dev = true;
    host_act = true;
    pf8 = false;
    use_dnnl = false;
    cpu_mode = false;

    // Per-device weight representation.  Preference order, gated by device
    // support: oneDNN int8 (XMX, both prefill and decode) -> SIn/w8 dp4a (when
    // oneDNN is unavailable or PF_GEMM_DNNL=0) -> raw fp32 (PF_DP4A=0).
    {
        const char * env = getenv("PF_DP4A");
        const bool dp4a_env_off = env && atoi(env) == 0;
        md_xmx = has_gpu && !dp4a_env_off && setup_md_dnnl();
        md_int8 = has_gpu && !dp4a_env_off && !md_xmx;
        fprintf(stderr, "[dev] multi-device weight path: %s\n",
                md_xmx ? "oneDNN int8 (XMX)" : (md_int8 ? "SIn int8 (dp4a)" : "fp32"));
    }
    // Per-device oneDNN int8 weights (XMX) were built above before the upload;
    // converted tensors skip their raw device copy.  The SIn/w8 fallback built
    // its copies just above.  Comment retained for the fallback path:
    if (md_int8) {
        w8_dev_.assign((size_t)ndev, {});
        for (int il = 0; il < m.hp.n_layer; il++) {
            const int d = layer_dev_[(size_t)il];
            if (d < 0 || dev_kind_[(size_t)d] != 0 || !dev_queues_[(size_t)d]) {
                continue;
            }
            const layer_t & L = m.layers[il];
            auto add = [&](const wt & t) {
                if (!t.data) {
                    return;
                }
                w8t w;
                m.build_w8_one(*dev_queues_[(size_t)d], t, w);
                if (w.vals) {
                    w8_dev_[(size_t)d][t.data] = w;
                }
            };
            add(L.ffn_gate);
            add(L.ffn_up);
            add(L.ffn_down);
            if (L.recurrent) {
                add(L.wqkv);
                add(L.wgate);
                add(L.ssm_out);
            } else {
                add(L.wq);
                add(L.wk);
                add(L.wv);
                add(L.wo);
            }
        }
        // the LM head's int8 copy lives on the primary device (decode head)
        m.build_w8_one(*dev_queues_[0], m.output, m.output8);
    }
    for (size_t i = 0; i < gpu_indices.size(); i++) {
        upload_device_weights((int)i);
    }
    // CPU partitions keep weight_maps_ empty: wptr() then returns the host mmap
    // pointer directly (their weights are never copied to a device)
    fprintf(stderr, "[dev] multi-device layer map: %s (%d backends)\n", layer_map.c_str(), ndev);
}

void engine::alloc_buffers() {
    const hparams & hp = m.hp;
    // per-token buffers must cover every (row, token) cell: chunk-batched
    // prefill (mode 2) lays tokens out flat, one row per chunk, so it needs
    // kMaxB*kMaxT rows rather than kMaxRows (the old chunk-local value)
    const int R = kMaxB * kMaxT;
    if (!multi_dev) {
    d_x = alloc_elems<float>((size_t)R * hp.n_embd);
    d_xnorm = alloc_elems<float>((size_t)R * hp.n_embd);
    d_qkv = alloc_elems<float>((size_t)R * hp.qkv_dim());
    d_z = alloc_elems<float>((size_t)R * hp.d_inner);
    d_beta = alloc_elems<float>((size_t)R * hp.dt_rank);
    d_alpha = alloc_elems<float>((size_t)R * hp.dt_rank);
    d_conv_out = alloc_elems<float>((size_t)R * hp.qkv_dim());
    d_attn_pre = alloc_elems<float>((size_t)R * hp.d_inner);
    d_attn_merged = alloc_elems<float>((size_t)R * hp.d_inner);
    d_qbuf = alloc_elems<float>((size_t)R * hp.n_head * 2 * hp.head_dim);
    d_kbuf = alloc_elems<float>((size_t)R * hp.n_head_kv * hp.head_dim);
    d_vbuf = alloc_elems<float>((size_t)R * hp.n_head_kv * hp.head_dim);
    d_attn_out = alloc_elems<float>((size_t)R * hp.n_head * hp.head_dim);
    d_ffn = alloc_elems<float>((size_t)R * ffn_stride);
    d_partials = alloc_elems<float>((size_t)R * hp.n_head * n_splits * (2 + hp.head_dim));
    d_partials_dec = alloc_elems<float>((size_t)kMaxB * hp.n_head * dec_splits * (2 + hp.head_dim));
    }
    d_logits = alloc_elems<float>((size_t)kMaxB * hp.n_vocab);
    d_last_hidden = alloc_elems<float>((size_t)hp.n_embd);
    // merged vision-token embeddings of the current multimodal prompt
    d_img_embd = alloc_elems<float>((size_t)kMaxImgTokens * hp.n_embd);

    int n_gdn = 0, n_attn = 0;
    for (int il = 0; il < hp.n_layer; il++) {
        if (hp.is_recr(il)) {
            n_gdn++;
        } else {
            n_attn++;
        }
    }
    // byte stride of one attention layer in the pool (the int8 layout adds a
    // per-unit scale plane, so this is not an element count any more)
    kv_layer_stride = (size_t)n_blocks * kv_block_bytes();
    kv_setup(n_attn, pool_initial);
    if (!multi_dev) {
        // single-device: one global recurrent-state array.  Multi-device keeps
        // each partition's slice in act_set (device USM, local indexing).
        d_gdn_state = alloc_elems<float>((size_t)kMaxB * n_gdn * hp.dt_rank * hp.d_state * hp.d_state);
        d_conv_state = alloc_elems<float>((size_t)kMaxB * n_gdn * (hp.conv_k - 1) * hp.qkv_dim());
    }
    h_tables.assign((size_t)kMaxB * max_blocks, 0);
    d_tables = alloc_elems<int32_t>((size_t)kMaxB * max_blocks);
    q.memcpy(d_tables, h_tables.data(), h_tables.size() * 4).wait();

    h_logits = (float *)malloc((size_t)kMaxB * hp.n_vocab * 4);
    d_info = sycl::malloc_host<step_info>(1, q);
    std::memset(d_info, 0, sizeof(step_info));
    d_segs_dec = alloc_elems<gemv_seg>(1024);
    d_segs_pf = alloc_elems<gemv_seg>(4096);
    d_segs_pf8 = alloc_elems<gemv_seg>(4096);
    {
        // SI8 activation scratch: max K is ffn_down's (n_ff) unless a bigger
        // projection shows up; use the max over the plan-relevant dims
        const int maxK = std::max({hp.n_embd, hp.qkv_dim() / 16 * 16, hp.d_inner, hp.n_head * hp.head_dim, hp.n_ff});
        // chunk-batched prefill quantizes one chunk row per tpb slot, so the
        // activation buffers must cover all rows (kMaxB * kMaxT)
        if (multi_dev) {
            // one activation set per partition, resident in its own device USM
            as_.resize(backends_.size());
            for (size_t d = 0; d < backends_.size(); d++) {
                as_[d] = alloc_act_set((int)d);
            }
            bind_acts(0);
        } else {
            const size_t xrows = (size_t)kMaxB * kMaxT;
            d_x8 = alloc_elems<int8_t>(xrows * maxK);
            d_xmeta = alloc_elems<sycl::float2>(xrows * (maxK / 32));
            d_xsumq = alloc_elems<int32_t>(xrows * (maxK / 16));
        }
    }
    d_segs_aux = alloc_elems<gemv_seg>(8);
    for (int tb : {1, 2, 4, 8, 16}) {
        dec_bucket b;
        b.tb = tb;
        b.d_segs = alloc_elems<gemv_seg>(1024);
        buckets_.push_back(std::move(b));
    }
    if (pc_enabled && pc_max_states > 0) {
        // bounded checkpoint store: one full conv+GDN state per slot
        d_pc_states = alloc_elems<float>((size_t)pc_max_states * pc_state_floats);
        pc_state_owner.assign(pc_max_states, -1);
        pc_state_stamp.assign(pc_max_states, 0);
        pc_state_free.resize(pc_max_states);
        for (int i = 0; i < pc_max_states; i++) {
            pc_state_free[i] = pc_max_states - 1 - i;
        }
    }
    reset_state();
}

void engine::zero_slot(int slot) {
    const hparams & hp = m.hp;
    const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state;
    const size_t conv_per = (size_t)(hp.conv_k - 1) * hp.qkv_dim();
    for (int il = 0; il < hp.n_layer; il++) {
        if (!hp.is_recr(il)) {
            continue;
        }
        if (multi_dev) {
            // device-resident per-partition recurrent state: zero via its queue
            const int dev = layer_dev_[(size_t)il];
            const int gl = layer_gdn_local_[(size_t)il];
            sycl::queue & qd = dev_queue(dev);
            qd.memset(as_[(size_t)dev].gdn_state + ((size_t)gl * kMaxB + slot) * gdn_per, 0, gdn_per * 4);
            qd.memset(as_[(size_t)dev].conv_state + ((size_t)gl * kMaxB + slot) * conv_per, 0, conv_per * 4);
            continue;
        }
        const int gi = m.gdn_layer_index[il];
        q.memset(d_gdn_state + ((size_t)gi * kMaxB + slot) * gdn_per, 0, gdn_per * 4);
        q.memset(d_conv_state + ((size_t)gi * kMaxB + slot) * conv_per, 0, conv_per * 4);
    }
    if (multi_dev) {
        sync_all();
    }
}
void engine::reset_state() {
    const hparams & hp = m.hp;
    const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state;
    const size_t conv_per = (size_t)(hp.conv_k - 1) * hp.qkv_dim();
    if (multi_dev) {
        for (size_t d = 0; d < as_.size(); d++) {
            sycl::queue & qd = dev_queue((int)d);
            if (n_gdn_dev_[d] > 0) {
                qd.memset(as_[d].gdn_state, 0, (size_t)kMaxB * n_gdn_dev_[d] * gdn_per * 4);
                qd.memset(as_[d].conv_state, 0, (size_t)kMaxB * n_gdn_dev_[d] * conv_per * 4);
            }
        }
        sync_all();
    } else {
        int n_gdn = 0;
        for (int il = 0; il < hp.n_layer; il++) {
            n_gdn += hp.is_recr(il);
        }
        q.memset(d_gdn_state, 0, (size_t)kMaxB * n_gdn * gdn_per * 4);
        q.memset(d_conv_state, 0, (size_t)kMaxB * n_gdn * conv_per * 4);
        q.wait();
    }
    std::memset(d_info, 0, sizeof(step_info));
    // rope sections are model constants; the memset above clears them
    std::memcpy(d_info->mrope_sections, hp.rope_sections, sizeof(hp.rope_sections));
    // the single-sequence entry points (eval/generate) do not run the prefix
    // cache: never let a stale tracking flag capture snapshots for them
    for (auto & s : pc_slot_) {
        s = {};
    }
    for (auto & p : pc_pending_) {
        pc_state_release(p.second);
    }
    pc_pending_.clear();
}

void engine::reset_single() {
    reset_state();
    d_info->n_rows = 1;
    d_info->tpb = kMaxT;
    d_info->n_real = 1;
    for (int r = 0; r < kMaxB; r++) {
        d_info->n_real_row[r] = 1;
    }
    d_info->pos[0] = 0;
    d_info->slot[0] = 0;
    d_info->active[0] = 1;
    d_info->tokens[0] = 0;
}
// ---------------------------------------------------------------------------
void engine::prefill_chunk(const std::vector<int> & toks, int start, int n, int slot, bool with_head) {
    pc_capture_begin(slot, toks, start, start, n);
    d_info->n_rows = 1;
    d_info->tpb = kMaxT;
    d_info->n_real = n;
    for (int r = 0; r < kMaxB; r++) {
        d_info->n_real_row[r] = n;
    }
    d_info->pos[0] = start;
    d_info->slot[0] = slot;
    d_info->active[0] = 1;
    for (int i = 0; i < n; i++) {
        d_info->tokens[i] = toks[start + i];
    }
    // PF_NOGRAPH=1: run the recorded sequence directly (diagnostics/PF_PROF)
    static const bool nog = getenv("PF_NOGRAPH") != nullptr;
    if (cpu_mode || multi_dev) {
        if (pf8 || multi_dev) {
            // full-chunk path: SI8 on single-device CPU, per-device oneDNN/w8
            // on the multi-device GPU layers (the CPU partitions run i8)
            const seg_plan & pl = with_head ? plan_pf8_ : plan_pf8_nh_;
            gemv_seg * ds = with_head ? d_segs_pf8 : d_segs_pf8_nh;
            record_forward(1, pl, ds, kMaxT);
        } else {
            // pick the smallest plan that still covers the n active tokens
            int si = (n + kPfSlice - 1) / kPfSlice - 1;
            if (si < 0) {
                si = 0;
            }
            if (si >= kPfSlots) {
                si = kPfSlots - 1;
            }
            const seg_plan & pl = with_head ? plan_pf_slot[si] : plan_pf_nh_slot[si];
            gemv_seg * ds = with_head ? d_segs_pf_slot[si] : d_segs_pf_nh_slot[si];
            record_forward(1, pl, ds, (si + 1) * kPfSlice);
        }
        sync_all();
        d_info->pc_active = 0;
        return;
    }
    if (nog && pf8) {
        record_forward(1, plan_pf8_, d_segs_pf8, kMaxT);
        q.wait();
        d_info->pc_active = 0;
        return;
    }
    q.ext_oneapi_graph(pf8 ? (with_head ? *e_pf8 : *e_pf8_nh) : *e_pf);
    q.wait();
    d_info->pc_active = 0;
}

// Chunk-batched prefill: `n` tokens (multiple of kMaxT) in one forward pass with
// one row per kMaxT-token chunk.  The batch size must be one of the recorded
// variants (see build_graphs); callers pick it with batched_prefill_fit().
// mode 2 (chunk-batched) for every full kMaxT multiple it supports, mode 1
// (chunked) only for the tail: mode 2 reads each weight once per forward, so it
// is ~2x the chunked throughput for the same tokens
void engine::prefill_text(const std::vector<int> & toks, int slot, int n) {
    int pos = 0;
    while (pos < n) {
        const int rem = n - pos;
        const int fit = cpu_mode ? 0 : batched_prefill_fit(rem);
        if (fit >= 1 && fit <= rem) {
            prefill_batch(toks, pos, fit, slot, pos);
            pos += fit;
        } else {
            const int c = std::min<int>(kMaxT, rem);
            prefill_chunk(toks, pos, c, slot);
            pos += c;
        }
    }
}

void engine::prefill_batch(const std::vector<int> & toks, int start, int n, int slot, int pos0) {
    // ceil: the last row may be partial (any n up to kMaxB*kMaxT), encoded in
    // d_info->n_real_row so the kernels process only its real tokens
    const int NCH = (n + kMaxT - 1) / kMaxT;
    if (cpu_mode) {
        throw std::runtime_error("prefill_batch: not supported on the CPU backend");
    }
    static const bool dbg_pfb = getenv("PF_DBG_PFB") != nullptr;
    pc_capture_begin(slot, toks, start, pos0, n);
    if (dbg_pfb) {
        fprintf(stderr, "[pfb] n=%d NCH=%d segs=%zu x8=%p\n", n, NCH, plan_pfb_.segs.size(), (void *)d_x8);
    }
    d_info->n_rows = NCH;
    d_info->tpb = kMaxT;
    d_info->n_real = kMaxT; // grid extent; the per-row count refines it
    for (int r = 0; r < NCH; r++) {
        const int rem = std::min(kMaxT, n - r * kMaxT);
        d_info->n_real_row[r] = rem;
        d_info->pos[r] = pos0 + r * kMaxT;
        d_info->slot[r] = slot;
        d_info->active[r] = 1;
        for (int t = 0; t < rem; t++) {
            d_info->tokens[r * kMaxT + t] = toks[start + r * kMaxT + t];
        }
        for (int t = rem; t < kMaxT; t++) {
            d_info->tokens[r * kMaxT + t] = 0; // padding, never processed
        }
    }
    for (int r = NCH; r < kMaxB; r++) {
        d_info->n_real_row[r] = 0;
        d_info->pos[r] = 0;
        d_info->slot[r] = slot;
        d_info->active[r] = 0;
    }
    static const bool nog = getenv("PF_NOGRAPH") != nullptr;
    // oneDNN primitives cannot be recorded into a SYCL command graph, so the
    // PF_GEMM_DNNL path always uses the direct (non-graph) replay of mode 2.
    // Multi-device has no recorded graphs at all, so it also always goes direct:
    // the GPU partition runs oneDNN int8 at the batch M (add_weight pre-builds
    // every M in 32..512 and acc_cap covers the widest N), the CPU partition
    // i8_gemm, and the small non-convertible tensors one fp32 grid dispatch.
    if ((nog || use_dnnl || multi_dev) && !plan_pfb_.segs.empty()) {
        if (dbg_pfb) {
            fprintf(stderr, "[pfb] direct record_forward(2)\n");
        }
        record_forward(2, plan_pfb_, d_segs_pfb, n, d_segs_pfb);
        if (multi_dev) {
            sync_all();
        } else {
            q.wait();
        }
        d_info->pc_active = 0;
        return;
    }
    const pfb_variant * v = nullptr;
    for (const auto & pv : pfb_vars_) {
        if (pv.ntok == n) {
            v = &pv;
            break;
        }
    }
    if (!v) {
        throw std::runtime_error("prefill_batch: no recorded variant for this size");
    }
    if (dbg_pfb) {
        fprintf(stderr, "[pfb] graph replay ntok=%d\n", v->ntok);
    }
    q.ext_oneapi_graph(*v->e);
    q.wait();
    d_info->pc_active = 0;
}

void engine::decode_batch(const int32_t * tokens, const int32_t * poss, const int32_t * slots, int n_rows) {
    d_info->pc_active = 0; // decode never captures checkpoints
    d_info->n_rows = n_rows;
    d_info->tpb = 1;
    d_info->n_real = 1;
    for (int r = 0; r < kMaxB; r++) {
        d_info->pos[r] = (r < n_rows) ? poss[r] : 0;
        d_info->slot[r] = (r < n_rows) ? slots[r] : (r % kMaxB);
        d_info->active[r] = (r < n_rows) ? 1 : 0;
        d_info->tokens[r] = (r < n_rows) ? tokens[r] : 0;
        d_info->n_real_row[r] = 1;
    }
    if (pf8_dec && n_rows == 1 && e_dec8) {
        q.ext_oneapi_graph(*e_dec8);
        q.wait();
        return;
    }
    if (cpu_mode && pf8_dec && n_rows == 1 && d_segs_dec8) {
        record_forward(0, plan_dec8_, d_segs_dec8, 1);
        sync_all();
        return;
    }
    if (multi_dev && (md_int8 || md_xmx) && n_rows == 1 && d_segs_dec8) {
        // single-token decode on the per-device weight copies: the GPU layers
        // run dp4a_gemv (SIn) or the oneDNN int8 GEMM (XMX).  With the
        // per-partition command graphs each partition's ~350 kernels go out in
        // one submission instead of one per kernel.
        if (md_dec_ok) {
            replay_md_dec_graphs();
        } else {
            record_forward(0, plan_dec8_, d_segs_dec8, 1);
        }
        sync_all();
        return;
    }
    // pick the smallest captured graph that fits the batch
    dec_bucket * b = nullptr;
    for (auto & bk : buckets_) {
        if (bk.tb >= n_rows) {
            b = &bk;
            break;
        }
    }
    if (!b) {
        b = &buckets_.back();
    }
    if (cpu_mode || multi_dev) {
        record_forward(0, b->plan, b->d_segs, b->tb);
        sync_all();
        return;
    }
    q.ext_oneapi_graph(*b->e);
    q.wait();
}

void engine::fetch_logits(int row, float * out) {
    const hparams & hp = m.hp;
    q.memcpy(h_logits, d_logits + (size_t)row * hp.n_vocab, (size_t)hp.n_vocab * 4).wait();
    std::memcpy(out, h_logits, (size_t)hp.n_vocab * 4);
}

void engine::forward_plain_pf() {
    record_forward(1, plan_pf_, d_segs_pf, kMaxT);
}

void engine::forward_plain_pf8() {
    record_forward(1, plan_pf8_, d_segs_pf8, kMaxT);
}

void engine::forward_plain_dec(int rows) {
    for (auto & b : buckets_) {
        if (b.tb == rows) {
            record_forward(0, b.plan, b.d_segs, rows);
            return;
        }
    }
}
std::vector<float> engine::run_head() {
    // d_last_hidden already holds the post-output_norm hidden state of the last
    // token (the prefill graph writes it via copy_row).
    const hparams & hp = m.hp;
    // Prefer the primary device's int8 head.  The raw Q6_K device copy is *not*
    // uploaded when the head was converted (upload_device_weights consults
    // has_weight(host)), so wptr() would throw here; also, the fp32 gemv_group
    // this used to run is the same slow path the decode head had.  Fall back to
    // the fp32 group only when there is no oneDNN head at all.
    bool head_done = false;
    if (dnnl_gemm * D = dnnl_for(0)) {
        // wkey: the host (oneDNN) key when the head was converted, the uploaded
        // device pointer otherwise
        const void * hk = wkey(0, m.output.data);
        if (hk && D->quantize(d_last_hidden, nullptr, hp.n_embd, hp.n_embd, 1, hp.n_embd)) {
            head_done = D->gemm_w4(hk, nullptr, 1.0f, 1, hp.n_embd, d_logits, hp.n_vocab)
                        || D->gemm(hk, nullptr, 1.0f, 1, hp.n_embd, d_logits, hp.n_vocab);
        }
    }
    if (head_done) {
        q.memcpy(h_logits, d_logits, (size_t)hp.n_vocab * 4).wait();
        return std::vector<float>(h_logits, h_logits + hp.n_vocab);
    }
    gemv_seg s{};
    s.w = wptr(0, m.output.data);
    s.type = m.output.type;
    s.K = hp.n_embd;
    s.n_rows = hp.n_vocab;
    s.x = d_last_hidden;
    s.x_stride = hp.n_embd;
    s.act_up = nullptr;
    s.out = d_logits;
    s.out_stride = hp.n_vocab;
    s.residual = nullptr;
    s.alpha = 1.0f;
    q.memcpy(d_segs_aux, &s, sizeof(s)).wait();
    backend().gemv_group(s.type, d_segs_aux, 1, hp.n_vocab, 1, hp.n_embd / 256, 0);
    backend().synchronize();
    q.memcpy(h_logits, d_logits, (size_t)hp.n_vocab * 4).wait();
    return std::vector<float>(h_logits, h_logits + hp.n_vocab);
}

std::vector<float> engine::eval(const std::vector<int> & tokens) {
    std::lock_guard<std::mutex> lk(mtx);
    reset_single();
    if (tokens.empty()) {
        return {};
    }
    // assign blocks for the whole prompt
    std::vector<int> blocks;
    const int need = ((int)tokens.size() + kBlockSize - 1) / kBlockSize;
    for (int i = 0; i < need; i++) {
        int b = alloc_block();
        if (b < 0) {
            throw std::runtime_error("out of KV blocks");
        }
        blocks.push_back(b);
    }
    set_table(0, blocks);
    prefill_text(tokens, 0, (int)tokens.size());
    auto out = run_head();
    for (int b : blocks) {
        free_block(b);
    }
    return out;
}

std::vector<int> engine::generate(const std::vector<int> & prompt, const gen_params & gp,
                                  const std::function<bool(int)> & cb, std::vector<float> * first_logits) {
    std::lock_guard<std::mutex> lk(mtx);
    return generate_impl(prompt, nullptr, gp, cb, first_logits);
}

std::vector<int> engine::generate_mm(const mm_prompt & p, const gen_params & gp, const std::function<bool(int)> & cb,
                                     std::vector<float> * first_logits) {
    std::lock_guard<std::mutex> lk(mtx);
    return generate_impl(p.tokens, (p.has_images() ? &p : nullptr), gp, cb, first_logits);
}

std::vector<int> engine::generate_impl(const std::vector<int> & prompt, const mm_prompt * mm, const gen_params & gp,
                                       const std::function<bool(int)> & cb, std::vector<float> * first_logits) {
    reset_single();
    const hparams & hp = m.hp;
    sampler_state ss;
    ss.seed(gp.seed ? gp.seed : std::random_device{}());

    std::vector<int> out;
    std::vector<float> logits;
    int pos = 0;
    const int nprompt = (int)prompt.size();
    if (nprompt == 0) {
        return out;
    }

    if (mm) {
        const size_t rows = (size_t)mm->embd.size() / std::max(1, hp.n_embd);
        if (rows > (size_t)kMaxImgTokens) {
            throw std::runtime_error("multimodal: too many image tokens");
        }
        // the device path already produced the embeddings in the caller's buffer
        if (!mm->d_embd && !mm->embd.empty()) {
            q.memcpy(d_img_embd, mm->embd.data(), mm->embd.size() * sizeof(float)).wait();
        }
    }

    std::vector<int> blocks;
    const int need = (std::max(nprompt, 1) + kBlockSize - 1) / kBlockSize;
    for (int i = 0; i < need; i++) {
        int b = alloc_block();
        if (b < 0) {
            throw std::runtime_error("out of KV blocks");
        }
        blocks.push_back(b);
    }
    set_table(0, blocks);

    if (mm) {
        while (pos < nprompt) {
            const int n = std::min<int>(kMaxT, nprompt - pos);
            // per-chunk M-RoPE positions (section-major) and image row mapping
            d_info->mrope_on = 1;
            d_info->img_embd = mm->d_embd ? mm->d_embd : (mm->embd.empty() ? nullptr : d_img_embd);
            for (int s = 0; s < 4; s++) {
                for (int i = 0; i < n; i++) {
                    d_info->mrope[s * (kMaxB * kMaxT) + i] = mm->mrope[(size_t)s * nprompt + pos + i];
                }
            }
            for (int i = 0; i < n; i++) {
                d_info->img_row[i] = mm->img_row[pos + i];
            }
            prefill_chunk(prompt, pos, n, 0);
            pos += n;
        }
    } else {
        // text-only: use the fastest path (mode 2 for the bulk, mode 1 tail)
        prefill_text(prompt, 0, nprompt);
        pos = nprompt;
    }
    logits = run_head();
    if (first_logits) {
        *first_logits = logits;
    }

    // decode positions: an image shifts the position counter away from the
    // token count, so continue from the prompt's final position.  The KV slot
    // index stays the *token* count (`pos`); the RoPE position is carried
    // separately through the M-RoPE array.
    int next_pos = mm ? mm->pos_after : nprompt;
    d_info->img_embd = nullptr;

    int32_t tok_buf[kMaxB] = {0}, pos_buf[kMaxB] = {0}, slot_buf[kMaxB] = {0};
    // PF_DUMP_GEN: log the decode loop's decisions (the sampled id, whether it
    // is an EOS, why the loop stopped) - a token that is never emitted (because
    // the EOS check runs before `cb`) is otherwise invisible from the CLI.
    static const bool dbg_gen = getenv("PF_DUMP_GEN") != nullptr;
    // PF_DUMP_DEC_LOGITS=<prefix>: write the logits the sampler sees at every
    // step (prefix.0.bin = the prompt prefill, prefix.1.bin = the first decode)
    // so a decode step can be diffed against a prefill of the same sequence.
    const char * dbg_lg = getenv("PF_DUMP_DEC_LOGITS");
    for (int step = 0; step < gp.max_tokens; step++) {
        if (dbg_lg) {
            char path[512];
            snprintf(path, sizeof(path), "%s.%d.bin", dbg_lg, step);
            if (FILE * fp = fopen(path, "wb")) {
                fwrite(logits.data(), sizeof(float), logits.size(), fp);
                fclose(fp);
            }
        }
        const int tok = sample_token(logits.data(), hp.n_vocab, gp, out, ss);
        out.push_back(tok);
        if (dbg_gen) {
            fprintf(stderr, "[gen] step=%d pos=%d id=%d eos=%d\n", step, pos, tok, (int)is_eos(tok));
        }
        if (!gp.ignore_eos && is_eos(tok)) {
            break;
        }
        if (!cb(tok)) {
            break;
        }
        if (step + 1 >= gp.max_tokens) {
            break;
        }
        if (pos >= max_seq - 1) {
            break;
        }
        // grow the block table if needed (KV slots advance one per token)
        if (pos % kBlockSize == 0) {
            int b = alloc_block();
            if (b < 0) {
                break;
            }
            blocks.push_back(b);
            set_table(0, blocks);
        }
        tok_buf[0] = tok;
        pos_buf[0] = pos;
        slot_buf[0] = 0;
        if (mm) {
            // single decode row: every section uses the running position
            d_info->mrope_on = 1;
            for (int s = 0; s < 4; s++) {
                d_info->mrope[s * (kMaxB * kMaxT)] = next_pos;
            }
        }
        decode_batch(tok_buf, pos_buf, slot_buf, 1);
        pos++;
        next_pos++;
        fetch_logits(0, logits.data());
    }
    for (int b : blocks) {
        free_block(b);
    }
    return out;
}

} // namespace si
