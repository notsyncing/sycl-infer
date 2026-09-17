#include "engine.h"
#include "quant.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace si {

static float * dalloc(sycl::queue & q, size_t n) {
    float * p = sycl::malloc_device<float>(n, q);
    if (!p) {
        throw std::runtime_error("device alloc failed");
    }
    return p;
}
engine::engine(const std::string & model_path, int max_seq_, int n_splits_, int n_blocks_, int kv_cap_mb)
    : q(sycl::gpu_selector_v, sycl::property::queue::in_order()), max_seq(max_seq_), n_splits(n_splits_),
      n_blocks(n_blocks_) {
    m.load(model_path);
    tk.load(m.gguf);
    m.upload(q);
    {
        // fp32 scale side arrays: measured as a net loss on this GPU (the packed
        // scales share cache lines with the weights; a separate array adds a
        // memory stream) -> opt-in only (PF_META=1, ~260 MB)
        const char * envm = getenv("PF_META");
        use_meta32 = envm && atoi(envm) != 0;
        if (use_meta32) {
            build_meta32(q);
        }
        // DP4A is on by default (3x prefill); PF_DP4A=0 forces the fp32 path
        const char * env = getenv("PF_DP4A");
        pf8 = !(env && atoi(env) == 0);
        // the packed SI8 decode GEMV now beats the fp32 kernel (~30%);
        // PF_DP4A_DEC=0 forces the fp32 decode
        const char * envd = getenv("PF_DP4A_DEC");
        pf8_dec = pf8 && !(envd && atoi(envd) == 0);
        if (pf8) {
            m.build_w8(q); // SI8 copies cost ~700 MB; only for the DP4A path
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
        const char * en = getenv("PF_PC_STATES");
        pc_max_states = en ? atoi(en) : 8;
        if (pc_max_states < 0) {
            pc_max_states = 0;
        }
        // no checkpoints -> nothing to resume from, so the whole cache is off
        if (pc_max_states == 0) {
            pc_enabled = false;
        }
        if (pc_enabled) {
            const hparams & hp = m.hp;
            int n_gdn = 0;
            for (int il = 0; il < hp.n_layer; il++) {
                n_gdn += hp.is_recr(il);
            }
            pc_state_floats = (size_t)n_gdn * (gdn_per_slot() + conv_per_slot());
        }
    }
    // oneDNN int8 matmul for the mode-2 prefill GEMMs (PF_GEMM_DNNL, default
    // on).  The converted weights live in device USM and are built once here;
    // PF_GEMM_DNNL=0 keeps the dp4a chunk-batched path bit-identical to before.
    use_dnnl = pf8 && dnnl_gemm_enabled();
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
}

engine::~engine() {
    // release device/USM allocations (tests construct engines in loops)
    auto f = [&](auto * p) {
        if (p) {
            sycl::free(p, q);
        }
    };
    f(d_tables);
    f(d_info);
    f(d_segs_dec);
    f(d_segs_pf);
    f(d_segs_pf8);
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
    kv_release_pool();
    m.free_w8(q);
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

void engine::alloc_buffers() {
    const hparams & hp = m.hp;
    // per-token buffers must cover every (row, token) cell: chunk-batched
    // prefill (mode 2) lays tokens out flat, one row per chunk, so it needs
    // kMaxB*kMaxT rows rather than kMaxRows (the old chunk-local value)
    const int R = kMaxB * kMaxT;
    d_x = dalloc(q, (size_t)R * hp.n_embd);
    d_xnorm = dalloc(q, (size_t)R * hp.n_embd);
    d_qkv = dalloc(q, (size_t)R * 3 * hp.d_inner);
    d_z = dalloc(q, (size_t)R * hp.d_inner);
    d_beta = dalloc(q, (size_t)R * hp.dt_rank);
    d_alpha = dalloc(q, (size_t)R * hp.dt_rank);
    d_conv_out = dalloc(q, (size_t)R * 3 * hp.d_inner);
    d_attn_pre = dalloc(q, (size_t)R * hp.d_inner);
    d_attn_merged = dalloc(q, (size_t)R * hp.d_inner);
    d_qbuf = dalloc(q, (size_t)R * hp.n_head * 2 * hp.head_dim);
    d_kbuf = dalloc(q, (size_t)R * hp.n_head_kv * hp.head_dim);
    d_vbuf = dalloc(q, (size_t)R * hp.n_head_kv * hp.head_dim);
    d_attn_out = dalloc(q, (size_t)R * hp.n_head * hp.head_dim);
    d_ffn = dalloc(q, (size_t)R * ffn_stride);
    d_partials = dalloc(q, (size_t)R * hp.n_head * n_splits * (2 + hp.head_dim));
    d_partials_dec = dalloc(q, (size_t)kMaxB * hp.n_head * dec_splits * (2 + hp.head_dim));
    d_logits = dalloc(q, (size_t)kMaxB * hp.n_vocab);
    d_last_hidden = dalloc(q, (size_t)hp.n_embd);
    // merged vision-token embeddings of the current multimodal prompt
    d_img_embd = dalloc(q, (size_t)kMaxImgTokens * hp.n_embd);

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
    d_gdn_state = dalloc(q, (size_t)kMaxB * n_gdn * hp.dt_rank * hp.d_state * hp.d_state);
    d_conv_state = dalloc(q, (size_t)kMaxB * n_gdn * (hp.conv_k - 1) * 3 * hp.d_inner);
    h_tables.assign((size_t)kMaxB * max_blocks, 0);
    d_tables = sycl::malloc_device<int32_t>((size_t)kMaxB * max_blocks, q);
    q.memcpy(d_tables, h_tables.data(), h_tables.size() * 4).wait();

    h_logits = (float *)malloc((size_t)kMaxB * hp.n_vocab * 4);
    d_info = sycl::malloc_host<step_info>(1, q);
    std::memset(d_info, 0, sizeof(step_info));
    d_segs_dec = sycl::malloc_device<gemv_seg>(1024, q);
    d_segs_pf = sycl::malloc_device<gemv_seg>(4096, q);
    d_segs_pf8 = sycl::malloc_device<gemv_seg>(4096, q);
    {
        // SI8 activation scratch: max K is ffn_down's (n_ff) unless a bigger
        // projection shows up; use the max over the plan-relevant dims
        const int maxK = std::max({hp.n_embd, 3 * hp.d_inner / 16 * 16, hp.d_inner, hp.n_head * hp.head_dim, hp.n_ff});
        // chunk-batched prefill quantizes one chunk row per tpb slot, so the
        // activation buffers must cover all rows (kMaxB * kMaxT)
        const size_t xrows = (size_t)kMaxB * kMaxT;
        d_x8 = sycl::malloc_device<int8_t>(xrows * maxK, q);
        d_xmeta = sycl::malloc_device<sycl::float2>(xrows * (maxK / 32), q);
        d_xsumq = sycl::malloc_device<int32_t>(xrows * (maxK / 16), q);
    }
    d_segs_aux = sycl::malloc_device<gemv_seg>(8, q);
    for (int tb : {1, 2, 4, 8, 16}) {
        dec_bucket b;
        b.tb = tb;
        b.d_segs = sycl::malloc_device<gemv_seg>(1024, q);
        buckets_.push_back(std::move(b));
    }
    if (pc_enabled && pc_max_states > 0) {
        // bounded checkpoint store: one full conv+GDN state per slot
        d_pc_states = dalloc(q, (size_t)pc_max_states * pc_state_floats);
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
    const size_t conv_per = (size_t)(hp.conv_k - 1) * 3 * hp.d_inner;
    int gi = 0;
    for (int il = 0; il < hp.n_layer; il++) {
        if (!hp.is_recr(il)) {
            continue;
        }
        q.memset(d_gdn_state + ((size_t)gi * kMaxB + slot) * gdn_per, 0, gdn_per * 4);
        q.memset(d_conv_state + ((size_t)gi * kMaxB + slot) * conv_per, 0, conv_per * 4);
        gi++;
    }
}
void engine::reset_state() {
    const hparams & hp = m.hp;
    int n_gdn = 0;
    for (int il = 0; il < hp.n_layer; il++) {
        n_gdn += hp.is_recr(il);
    }
    q.memset(d_gdn_state, 0, (size_t)kMaxB * n_gdn * hp.dt_rank * hp.d_state * hp.d_state * 4);
    q.memset(d_conv_state, 0, (size_t)kMaxB * n_gdn * (hp.conv_k - 1) * 3 * hp.d_inner * 4);
    q.wait();
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
    d_info->pos[0] = start;
    d_info->slot[0] = slot;
    d_info->active[0] = 1;
    for (int i = 0; i < n; i++) {
        d_info->tokens[i] = toks[start + i];
    }
    // PF_NOGRAPH=1: run the recorded sequence directly (diagnostics/PF_PROF)
    static const bool nog = getenv("PF_NOGRAPH") != nullptr;
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
void engine::prefill_batch(const std::vector<int> & toks, int start, int n, int slot, int pos0) {
    const int NCH = n / kMaxT;
    static const bool dbg_pfb = getenv("PF_DBG_PFB") != nullptr;
    pc_capture_begin(slot, toks, start, pos0, n);
    if (dbg_pfb) {
        fprintf(stderr, "[pfb] n=%d NCH=%d segs=%zu x8=%p\n", n, NCH, plan_pfb_.segs.size(), (void *)d_x8);
    }
    d_info->n_rows = NCH;
    d_info->tpb = kMaxT;
    d_info->n_real = kMaxT;
    for (int r = 0; r < NCH; r++) {
        d_info->pos[r] = pos0 + r * kMaxT;
        d_info->slot[r] = slot;
        d_info->active[r] = 1;
        for (int t = 0; t < kMaxT; t++) {
            d_info->tokens[r * kMaxT + t] = toks[start + r * kMaxT + t];
        }
    }
    for (int r = NCH; r < kMaxB; r++) {
        d_info->pos[r] = 0;
        d_info->slot[r] = slot;
        d_info->active[r] = 0;
    }
    static const bool nog = getenv("PF_NOGRAPH") != nullptr;
    // oneDNN primitives cannot be recorded into a SYCL command graph, so the
    // PF_GEMM_DNNL path always uses the direct (non-graph) replay of mode 2
    if ((nog || use_dnnl) && !plan_pfb_.segs.empty()) {
        if (dbg_pfb) {
            fprintf(stderr, "[pfb] direct record_forward(2)\n");
        }
        record_forward(2, plan_pfb_, d_segs_pfb, n, d_segs_pfb);
        q.wait();
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
    }
    if (pf8_dec && n_rows == 1 && e_dec8) {
        q.ext_oneapi_graph(*e_dec8);
        q.wait();
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
    gemv_seg s{};
    s.w = m.dev_ptr(m.tok_embd.data);
    s.type = m.tok_embd.type;
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
    gemv_group_launch(q, s.type, d_segs_aux, 1, hp.n_vocab, 1, hp.n_embd / 256);
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
    int pos = 0;
    while (pos < (int)tokens.size()) {
        const int n = std::min<int>(kMaxT, (int)tokens.size() - pos);
        prefill_chunk(tokens, pos, n, 0);
        pos += n;
    }
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

    while (pos < nprompt) {
        const int n = std::min<int>(kMaxT, nprompt - pos);
        if (mm) {
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
        }
        prefill_chunk(prompt, pos, n, 0);
        pos += n;
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
    for (int step = 0; step < gp.max_tokens; step++) {
        const int tok = sample_token(logits.data(), hp.n_vocab, gp, out, ss);
        out.push_back(tok);
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
