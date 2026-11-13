// QSA (query-side sparse attention) for qwen4exp.
//
// Up to `indexer_top_k` keys the reference's selector covers every visible
// key, so dense attention is exactly equivalent; beyond it the model attends
// to the top `top_k` keys pooled in blocks of `compress_ratio` plus the
// incomplete tail.  This implementation keeps the selector and the attention
// on the host:
//   * the indexer weights are small and already in the host mmap;
//   * there is no host K/V mirror: the selected rows are gathered from the
//     paged pool and dequantized (`qsa_gather_positions`), so the host and
//     device paths see the same stored values and a prefix-cache resume works
//     for any KV type;
//   * the result is written back to the layer's ATTN_OUT buffer and the
//     existing wo GEMV runs on the device.
//
// The host QSA cost is not the dots: measured at ctx 4096 the four blocking
// D2H copies of a single layer were 23.7 of 26.8 ms/token, because each one
// drains the in-order queue and the host MoE otherwise overlaps the device.
// They are now enqueued together into per-layer pinned staging, and while the
// selector is dense-equivalent (pos < top_k, the common short-context decode)
// the whole update is deferred to the end of the forward and waited once -
// after the LM head has been enqueued, so the indexer projection and the mirror
// update run beside it.  A dense prefill chunk larger than kMaxB defers too:
// mixed rides the main queue (its source is reused by the next layer) and K/V
// ride a per-device copy queue, guarded before the next attention layer's qkv
// GEMV reuses the buffers.  The sparse path (> top_k keys) still runs the
// selector and the attention inline, before the layer's `wo`.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "engine.h"
#include "arch/qwen4exp/accessors.h"
#include "arch/qwen4exp/hparams.h"
#include "common/env.h"
#include "backend/cpu/kernels/common.h"
#include "backend/cpu/cpu_types.h"
#include "backend/gpu/kernels/kernels.h"
#include "backend/gpu/kernels/kv_type.h"

namespace si {

namespace {

double qsa_now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void head_norm(float * x, const float * w, int n, float eps) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    const float s = 1.f / std::sqrt(ss / (float)n + eps);
    for (int i = 0; i < n; i++) {
        x[i] = x[i] * s * w[i];
    }
}

void head_rope(float * x, int n_rot, float base, int pos) {
    rope_apply(x, n_rot, base, (float)pos);
}

// The reference keeps the indexer and main KV in the model's cache type (f16
// here), so every value it scores/attends to has been rounded to f16.  The host
// mirrors have to round too, or the f32 values drift by ~1e-3 and the top-k ties
// (rectified scores of exactly 0) pick different pools.
inline float f16r(float x) {
    return ggml_half_to_float(ggml_float_to_half(x));
}

// One-layer Stage 2 probe: PF_QSA_ATTN_LAYER selects the attention index and
// PF_QSA_ATTN_DUMP is the path prefix.  The .qsa file is the gated attention
// row; forward.cpp writes the matching .wo file after the projection.
int qsa_dump_layer() {
    static const int l = si::env::i32("PF_QSA_ATTN_LAYER", 0);
    return l;
}

// The QSA pool width: the reference pools `qsa.compress_ratio`
// (`q4_hp(hp).idx_kpool`) raw keys per selector entry, so a model that pools 8 must
// score one entry per 8-token block.  This file used to hardcode 4, which is
// the shipped model's ratio and therefore invisible until another model
// arrives: the wrong width still runs, it just selects the wrong key set.
//
// GREP GUARD: the pool geometry in this file is `kp` and nothing else - there
// must be no `pos / 4`, `pos % 4`, `p * 4` or `4 * id` pool arithmetic left.
// q4x_.qsa.on already requires idx_kpool > 1; the 4 is the fallback for a
// configuration that has no QSA metadata, where the selector is never reached.
int qsa_pool_div(const hparams & hp) {
    const int kp = q4_hp(hp).idx_kpool > 1 ? q4_hp(hp).idx_kpool : 4;
    // A divisor below 2 pools nothing and makes `pos % kp` meaningless.  The
    // fallback is 4, so this can only fire on a hand-built hparams - which is
    // exactly the case an assert is for, and it keeps a future edit from
    // quietly reintroducing a literal that disagrees with the metadata.
    assert(kp >= 2 && "QSA pool divisor must come from q4_hp(hp).idx_kpool");
    return kp;
}

} // namespace

// Per-layer pinned staging, grown to the largest chunk seen.  A pageable
// std::vector makes the copy synchronous on this backend, so the four D2H
// inputs and the sparse H2D output live in USM host memory.
void engine::qsa_stage_ensure(int a, int n_tok, bool need_q, bool need_out, int dev) {
    const hparams & hp = m.hp;
    if ((int)q4x_.qsa.stage.size() <= a) {
        q4x_.qsa.stage.resize((size_t)a + 1);
    }
    qsa_state::stage_t & st = q4x_.qsa.stage[(size_t)a];
    if (st.dev < 0) {
        st.dev = dev; // a layer never moves; the staging stays in its context
    }
    sycl::queue & qd = dev_queue(dev);
    auto realloc_to = [&](float *& p, size_t elems) {
        float * np = (float *)sycl::malloc_host(elems * sizeof(float), qd);
        if (!np) {
            throw std::runtime_error("qwen4exp: QSA staging allocation failed");
        }
        if (p) {
            sycl::free(p, qd);
        }
        p = np;
    };
    const size_t n_embd = (size_t)hp.n_embd;
    if (st.cap < (size_t)n_tok) {
        size_t nc = st.cap ? st.cap : 1;
        while (nc < (size_t)n_tok) {
            nc <<= 1;
        }
        realloc_to(st.mixed, nc * n_embd);
        st.cap = nc;
    }
    if (need_q || need_out) {
        if ((int)q4x_.qsa.host_stage.size() <= dev) {
            q4x_.qsa.host_stage.resize((size_t)dev + 1);
        }
        qsa_state::host_stage_t & hs = q4x_.qsa.host_stage[(size_t)dev];
        hs.dev = dev;
        auto realloc_hs = [&](float *& p, size_t & cap, size_t elems) {
            if (cap >= elems) {
                return;
            }
            size_t nc = cap ? cap : 1;
            while (nc < elems) {
                nc <<= 1;
            }
            float * np = (float *)sycl::malloc_host(nc * sizeof(float), qd);
            if (!np) {
                throw std::runtime_error("qwen4exp: QSA staging allocation failed");
            }
            if (p) {
                sycl::free(p, qd);
            }
            p = np;
            cap = nc;
        };
        if (need_q) {
            realloc_hs(hs.q, hs.cap_q, (size_t)n_tok * (size_t)hp.n_head * 2 * (size_t)hp.head_dim);
        }
        if (need_out) {
            realloc_hs(hs.out, hs.cap_out, (size_t)n_tok * (size_t)hp.n_head * (size_t)hp.head_dim);
        }
    }
}

// PF_QSA_EARLY: enqueue the per-layer `mixed` D2H before the QKV GEMVs.  The
// copy writes the same pinned staging that qsa_attention used to fill after
// qk_norm_rope, and it is submitted on the main in-order queue after hc_mix
// (its producer) and before the next layer's hc_mix (which rewrites the
// source), so the queue ordering provides both dependencies.  The event is
// consumed by qsa_attention, which then waits for this 10 KB copy alone instead
// of draining the whole QKV/rope chain.
void engine::qsa_early_mixed(int a, const layer_t & L, int dev, int mode, int n_tok, const step_info * inf,
                             const float * mixed) {
    if (!q4x_.qsa.early || !q4x_.qsa.on || !q4x_.qsa.devattn || mode != 0 || n_tok != 1 || a < 0
        || a >= (int)q4x_.qsa.cache.size() || a >= (int)q4x_.qsa.early_ev.size()) {
        return;
    }
    const bool is_cpu_dev = multi_dev ? (dev_kind_[(size_t)dev] == 1) : cpu_mode;
    if (is_cpu_dev || !q4_layer(L).idx.ok()) {
        return;
    }
    const hparams & hp = m.hp;
    if (hp.head_dim != 256 || kv_k_dtype() != kv_dtype_t::f16 || kv_v_dtype() != kv_dtype_t::f16) {
        return;
    }
    const int slot = inf->slot[0];
    const int pos = inf->pos[0];
    if (slot < 0 || slot >= (int)q4x_.qsa.cache[(size_t)a].size()) {
        return;
    }
    // Dense-equivalent tokens are deferred by qsa_attention, not waited for.
    if (pos + 1 <= q4_hp(hp).idx_top_k) {
        return;
    }
    // Stage first so qsa_attention's own qsa_stage_ensure(a, 1, ...) is a
    // no-op and the pointer the copy wrote is the one the host update reads.
    qsa_stage_ensure(a, n_tok, false, false, dev);
    qsa_state::stage_t & st = q4x_.qsa.stage[(size_t)a];
    qsa_state::early_copy & ec = q4x_.qsa.early_ev[(size_t)a];
    ec.ev = dev_queue(dev).memcpy(st.mixed, mixed, (size_t)n_tok * (size_t)hp.n_embd * sizeof(float));
    ec.dev = dev;
    ec.kind = 0;
    ec.valid = true;
}

// PF_QSA_IDX_DEV pinned staging: the two small D2H results.  The selector only
// reads the normed+RoPEd q (inh*id) and the pool update the raw k (id), so this
// is 2.5 KB/token/layer, not the 10 KB `mixed` copy it replaces.
void engine::qsa_idx_stage_ensure(int a, int n_tok, int dev) {
    const hparams & hp = m.hp;
    if ((int)q4x_.qsa.stage.size() <= a) {
        q4x_.qsa.stage.resize((size_t)a + 1);
    }
    qsa_state::stage_t & st = q4x_.qsa.stage[(size_t)a];
    if (st.dev < 0) {
        st.dev = dev;
    }
    sycl::queue & qd = dev_queue(dev);
    auto realloc_to = [&](float *& p, size_t elems) {
        float * np = (float *)sycl::malloc_host(elems * sizeof(float), qd);
        if (!np) {
            throw std::runtime_error("qwen4exp: QSA indexer staging allocation failed");
        }
        if (p) {
            sycl::free(p, qd);
        }
        p = np;
    };
    if (st.cap_i < (size_t)n_tok) {
        size_t nc = st.cap_i ? st.cap_i : 1;
        while (nc < (size_t)n_tok) {
            nc <<= 1;
        }
        realloc_to(st.iq, nc * (size_t)q4_hp(hp).idx_n_head * (size_t)q4_hp(hp).idx_head_dim);
        realloc_to(st.ik, nc * (size_t)q4_hp(hp).idx_head_dim);
        st.cap_i = nc;
    }
}

// PF_QSA_IDX_DEV device output buffers, per device and grown to the largest
// batch served (decode: one row).  A single buffer per device is safe even
// though every layer writes it: the D2H copy is enqueued on the same in-order
// queue right after the kernel and before the next layer's kernel, so the copy
// has consumed it before it can be overwritten.
void engine::qsa_idx_dev_ensure(int dev, int n_tok) {
    const hparams & hp = m.hp;
    if ((int)q4x_.qsa.idx_dev_buf.size() <= dev) {
        q4x_.qsa.idx_dev_buf.resize((size_t)dev + 1);
    }
    qsa_state::idx_dev_buf_t & b = q4x_.qsa.idx_dev_buf[(size_t)dev];
    if (b.dev == dev && b.cap >= n_tok) {
        return;
    }
    const size_t qn = (size_t)n_tok * (size_t)q4_hp(hp).idx_n_head * (size_t)q4_hp(hp).idx_head_dim;
    const size_t kn = (size_t)n_tok * (size_t)q4_hp(hp).idx_head_dim;
    if (b.q) {
        sycl::free(b.q, dev_queue(dev));
    }
    if (b.k) {
        sycl::free(b.k, dev_queue(dev));
    }
    b.q = (float *)dev_alloc_on(dev, qn * sizeof(float));
    b.k = (float *)dev_alloc_on(dev, kn * sizeof(float));
    if (!b.q || !b.k) {
        throw std::runtime_error("qwen4exp: QSA indexer device buffer allocation failed");
    }
    b.cap = n_tok;
    b.dev = dev;
}

// PF_QSA_EARLY (indexer form): run the indexer projection on the device right
// after `mixed` is final and enqueue only the 2.5 KB the host needs.  The host
// wait in qsa_attention is then on this tiny chain, not on the layer's
// QKV/rope cells.  Every shape this does not serve falls back to the old
// `mixed` copy + host projection.
void engine::qsa_early_indexer(int a, const layer_t & L, int dev, int mode, int n_tok, const step_info * inf,
                               const float * mixed) {
    const bool usable = q4x_.qsa.idx_dev && q4x_.qsa.early && q4x_.qsa.on && q4x_.qsa.devattn && mode == 0 && n_tok == 1 && a >= 0
                        && a < (int)q4x_.qsa.cache.size() && a < (int)q4x_.qsa.early_ev.size() && q4_layer(L).idx.ok()
                        && q4_layer(L).idx.q_proj.type == 30 && q4_layer(L).idx.k_proj.type == 30
                        && std::getenv("PF_QSA_MIX") == nullptr;
    const bool is_cpu_dev = multi_dev ? (dev_kind_[(size_t)dev] == 1) : cpu_mode;
    const hparams & hp = m.hp;
    const int slot = inf->slot[0];
    const int pos = inf->pos[0];
    if (!usable || is_cpu_dev || hp.head_dim != 256 || kv_k_dtype() != kv_dtype_t::f16
        || kv_v_dtype() != kv_dtype_t::f16 || slot < 0 || slot >= (int)q4x_.qsa.cache[(size_t)a].size()
        || pos + 1 <= q4_hp(hp).idx_top_k) {
        qsa_early_mixed(a, L, dev, mode, n_tok, inf, mixed);
        return;
    }
    // PF_QSA_SEL_DEV: the ring/pool and the selector live on the card.  The
    // host cache must already hold the prefill's pools; it does for every
    // sparse token (a prompt that crossed top_k built it, and a decode that
    // crossed it flushed the deferred dense update at the end of that step).
    const bool have_host = slot < (int)q4x_.qsa.cache[(size_t)a].size() && q4x_.qsa.cache[(size_t)a][(size_t)slot] != nullptr;
    if (q4x_.qsa.sel_dev && have_host) {
        qsa_idx_dev_ensure(dev, n_tok);
        qsa_state::idx_dev_buf_t & b = q4x_.qsa.idx_dev_buf[(size_t)dev];
        sycl::queue & q = dev_queue(dev);
        qsa_indexer_launch(q, mixed, wptr(dev, q4_layer(L).idx.q_proj.data), wptr(dev, q4_layer(L).idx.k_proj.data), hp.n_embd,
                           q4_hp(hp).idx_n_head, q4_hp(hp).idx_head_dim, hp.n_rot, hp.rope_base, pos, wf32(dev, q4_layer(L).idx.q_norm), b.q,
                           b.k, hp.rms_eps);
        qsa_idx_device_chain(a, L, dev, slot, pos);
        qsa_state::early_copy & ec = q4x_.qsa.early_ev[(size_t)a];
        ec.dev = dev;
        ec.kind = 2;
        ec.valid = true;
        return;
    }
    qsa_idx_stage_ensure(a, n_tok, dev);
    qsa_idx_dev_ensure(dev, n_tok);
    qsa_state::stage_t & st = q4x_.qsa.stage[(size_t)a];
    qsa_state::idx_dev_buf_t & b = q4x_.qsa.idx_dev_buf[(size_t)dev];
    sycl::queue & q = dev_queue(dev);
    qsa_indexer_launch(q, mixed, wptr(dev, q4_layer(L).idx.q_proj.data), wptr(dev, q4_layer(L).idx.k_proj.data), hp.n_embd, q4_hp(hp).idx_n_head,
                       q4_hp(hp).idx_head_dim, hp.n_rot, hp.rope_base, pos, wf32(dev, q4_layer(L).idx.q_norm), b.q, b.k, hp.rms_eps);
    qsa_state::early_copy & ec = q4x_.qsa.early_ev[(size_t)a];
    // The second copy's event covers the first on the in-order queue.
    (void)q.memcpy(st.iq, b.q, (size_t)n_tok * (size_t)q4_hp(hp).idx_n_head * (size_t)q4_hp(hp).idx_head_dim * sizeof(float));
    ec.ev = q.memcpy(st.ik, b.k, (size_t)n_tok * (size_t)q4_hp(hp).idx_head_dim * sizeof(float));
    ec.dev = dev;
    ec.kind = 1;
    ec.valid = true;
}

// PF_QSA_SEL_DEV device ring/pool, per (attention layer, slot) and allocated
// lazily on the first sparse decode token.  The pool holds the host's
// f16-rounded f32 values, so the device and host selectors see identical
// numbers; `ready` means the host state up to this token has been handed off.
void engine::qsa_idx_slot_ensure(int a, int slot, int dev) {
    if ((int)q4x_.qsa.idx_slot.size() <= a) {
        q4x_.qsa.idx_slot.resize((size_t)a + 1);
    }
    auto & v = q4x_.qsa.idx_slot[(size_t)a];
    if ((int)v.size() <= slot) {
        v.resize((size_t)slot + 1);
    }
    auto & p = v[(size_t)slot];
    if (!p) {
        p = std::make_unique<qsa_state::idx_slot_dev>();
    }
    if (p->pool && p->dev == dev) {
        return;
    }
    if (p->pool) {
        dev_queue(p->dev).wait();
        sycl::free(p->pool, dev_queue(p->dev));
        sycl::free(p->ring, dev_queue(p->dev));
    }
    const hparams & hp = m.hp;
    const int id = q4_hp(hp).idx_head_dim;
    const int kp = qsa_pool_div(hp);
    const size_t npool = (size_t)q4x_.qsa.max_seq / (size_t)kp + 2;
    sycl::queue & q = dev_queue(dev);
    p->pool = (float *)dev_alloc_on(dev, npool * (size_t)id * sizeof(float));
    p->ring = (float *)dev_alloc_on(dev, (size_t)kp * (size_t)id * sizeof(float));
    if (!p->pool || !p->ring) {
        throw std::runtime_error("qwen4exp: QSA device ring/pool allocation failed");
    }
    q.memset(p->pool, 0, npool * (size_t)id * sizeof(float));
    q.memset(p->ring, 0, (size_t)kp * (size_t)id * sizeof(float));
    p->dev = dev;
    p->ready = false;
}

void engine::qsa_idx_slot_release(int a, int slot) {
    if (a < 0 || a >= (int)q4x_.qsa.idx_slot.size()) {
        return;
    }
    auto & v = q4x_.qsa.idx_slot[(size_t)a];
    if (slot < 0) {
        for (auto & p : v) {
            if (p && p->pool) {
                dev_queue(p->dev).wait();
                sycl::free(p->pool, dev_queue(p->dev));
                sycl::free(p->ring, dev_queue(p->dev));
            }
            p.reset();
        }
        return;
    }
    if (slot >= (int)v.size() || !v[(size_t)slot]) {
        return;
    }
    auto & p = v[(size_t)slot];
    if (p->pool) {
        dev_queue(p->dev).wait();
        sycl::free(p->pool, dev_queue(p->dev));
        sycl::free(p->ring, dev_queue(p->dev));
    }
    p.reset();
}

void engine::qsa_idx_slot_release_all() {
    for (int a = 0; a < (int)q4x_.qsa.idx_slot.size(); a++) {
        qsa_idx_slot_release(a, -1);
    }
    q4x_.qsa.idx_slot.clear();
}

void engine::qsa_handoff_ensure(int dev, size_t n) {
    if ((int)q4x_.qsa.handoff.size() <= dev) {
        q4x_.qsa.handoff.resize((size_t)dev + 1);
    }
    qsa_state::handoff_t & h = q4x_.qsa.handoff[(size_t)dev];
    if (h.stage && h.cap >= n) {
        return;
    }
    if (h.stage) {
        if (h.ev_valid) {
            h.ev.wait();
            h.ev_valid = false;
        }
        sycl::free(h.stage, dev_queue(dev));
    }
    h.stage = (float *)sycl::malloc_host(n * sizeof(float), dev_queue(dev));
    if (!h.stage) {
        throw std::runtime_error("qwen4exp: QSA handoff staging allocation failed");
    }
    h.cap = n;
    h.dev = dev;
    h.ev_valid = false;
}

// One sparse decode token's full device QSA chain, enqueued right after the
// indexer projection and before the layer's QKV GEMVs: hand the host pool/ring
// over once, update the ring/pool, score the pools and select.  The gather and
// attention stay in qsa_attention (they need this layer's paged K/V, which
// qk_norm_rope produces later); the in-order queue orders them.
void engine::qsa_idx_device_chain(int a, const layer_t & L, int dev, int slot, int pos) {
    const hparams & hp = m.hp;
    const int id = q4_hp(hp).idx_head_dim;
    const int kp = qsa_pool_div(hp);
    qsa_idx_slot_ensure(a, slot, dev);
    qsa_state::idx_dev_buf_t & b = q4x_.qsa.idx_dev_buf[(size_t)dev];
    qsa_state::idx_slot_dev & sd = *q4x_.qsa.idx_slot[(size_t)a][(size_t)slot];
    qsa_devattn_ensure(dev);
    qsa_state::dev_buf & d = q4x_.qsa.dev[(size_t)dev];
    sycl::queue & q = dev_queue(dev);
    if (!sd.ready) {
        const int npool = pos / kp;
        const size_t npo = (size_t)npool * (size_t)id;
        const size_t nr = (size_t)kp * (size_t)id;
        qsa_handoff_ensure(dev, npo + nr);
        qsa_state::handoff_t & h = q4x_.qsa.handoff[(size_t)dev];
        if (h.ev_valid) {
            h.ev.wait();
            h.ev_valid = false;
        }
        const qsa_state::slot_state * hs = nullptr;
        if (a < (int)q4x_.qsa.cache.size() && slot < (int)q4x_.qsa.cache[(size_t)a].size()) {
            hs = q4x_.qsa.cache[(size_t)a][(size_t)slot].get();
        }
        if (hs) {
            if (npo) {
                std::memcpy(h.stage, hs->pool.data(), npo * sizeof(float));
            }
            std::memcpy(h.stage + npo, hs->ring.data(), nr * sizeof(float));
        } else {
            std::memset(h.stage, 0, (npo + nr) * sizeof(float));
        }
        (void)q.memcpy(sd.pool, h.stage, npo * sizeof(float));
        h.ev = q.memcpy(sd.ring, h.stage + npo, nr * sizeof(float));
        h.ev_valid = true;
        sd.ready = true;
    }
    qsa_idx_update_launch(q, b.k, pos, sd.ring, sd.pool, wf32(dev, q4_layer(L).idx.k_norm), id, kp, hp.n_rot, hp.rope_base,
                          hp.rms_eps);
    const int n_vis = pos + 1;
    const int n_pool_vis = n_vis / kp;
    const int n_pool_max = q4_hp(hp).idx_top_k / kp;
    const int n_top = std::min(n_pool_vis, n_pool_max);
    if (n_top > 0) {
        qsa_score_launch(q, b.q, sd.pool, n_pool_vis, id, q4_hp(hp).idx_n_head, 1.0f / std::sqrt((float)id), d.scores);
        qsa_topk_launch(q, d.scores, n_pool_vis, n_top, kp, n_vis, d.dpools, d.dsel);
    }
}

// Stage 3 consumption with the selection already on the device: same gather +
// attention as qsa_devattn_run, without the host `sel` vector or its H2D.
void engine::qsa_devattn_run_dev(int a, int dev, int slot, int pos, int nsel, const float * qbuf, float * attn_out) {
    (void)pos;
    const hparams & hp = m.hp;
    const int nh = hp.n_head, nkv = hp.n_head_kv, hd = hp.head_dim;
    if (nsel <= 0) {
        return;
    }
    qsa_devattn_aux(a, dev);
    qsa_state::dev_buf & d = q4x_.qsa.dev[(size_t)dev];
    qsa_state::dev_aux & x = q4x_.qsa.aux[(size_t)a];
    if (nsel > d.cap_keys) {
        throw std::runtime_error("qwen4exp: QSA Stage 2 sel exceeds the compact-pool capacity");
    }
    const int nblk = (nsel + kBlockSize - 1) / kBlockSize;
    sycl::queue & q = dev_queue(dev);
    const char * kp = nullptr;
    const char * vp = nullptr;
    const char * ksc = nullptr;
    const char * vsc = nullptr;
    kv_layer_ptrs(a, kp, vp, ksc, vsc);
    qsa_sinfo_launch(q, x.sinfo, nsel);
    qsa_gather_launch(q, kp, vp, d_tables + (size_t)slot * max_blocks, d.dsel, nsel, nkv, hd, kBlockSize, d.ck,
                      d.cv);
    float * part = multi_dev ? as_[(size_t)dev].partials_dec : d_partials_dec;
    const int nsp = std::max(1, dec_splits);
    compute_backend & cbe = multi_dev ? *backends_[(size_t)dev] : *be;
    cbe.attn(qbuf, qbuf, d.ck, d.cv, part, d.tab, nh, nkv, hd, nsp, x.sinfo, hp.attn_scale, nblk, 1, 1, nullptr, 0,
             nullptr, nullptr);
    cbe.attn_combine(part, qbuf, attn_out, x.sinfo, nh, hd, nsp, 1, 1);
}

// The per-token QSA state update: the two indexer projections, their norm+RoPE,
// the raw-key ring/pool and the main K/V mirror.  It is split out of
// qsa_attention so a decode that is still dense-equivalent can run it at the
// end of the forward from host-staged copies (see qsa_flush).
void engine::qsa_apply_update(int a, const layer_t & L, int n_tok, const int * posv, const int * slotv,
                              const float * mh, float * qpos, bool with_q, const float * iq_pre,
                              const float * ik_pre, const step_info * pcinf) {
    const hparams & hp = m.hp;
    const int id = q4_hp(hp).idx_head_dim, inh = q4_hp(hp).idx_n_head;
    const int n_embd = hp.n_embd;
    const int kp = qsa_pool_div(hp);
    static const bool qtime = si::env::flag("PF_QSA_TIME");
    double q_t0 = qtime ? qsa_now_ms() : 0.0;

    // Indexer projections (BF16 weights, host GEMV) for the whole batch.  The
    // BF16 dot is 39 MB/token of weights at a 2-byte element, so the ISA
    // variant is worth real time; a non-BF16 indexer falls back to the generic
    // arbitrary-K GEMV.  The q projection is 80% of it (512 of 640 rows) and is
    // read only by the selector, so a dense-equivalent token skips it: this
    // token's q is never scored, and the next token computes its own.
    // PF_QSA_IDX_DEV instead supplies iq (already normed+RoPEd) and the raw ik
    // from the device; the ring/pool/mirror update below is unchanged.
    std::vector<float> iq((with_q && !iq_pre) ? (size_t)n_tok * inh * id : 0);
    std::vector<float> ik(ik_pre ? 0 : (size_t)n_tok * id);
    if (!iq_pre || !ik_pre) {
        if (q4_layer(L).idx.k_proj.type == 30) {
            if (with_q && !iq_pre) {
                cpu_qsa_bf16_gemv(q4_layer(L).idx.q_proj.data, mh, n_embd, iq.data(), inh * id, n_tok, n_embd, inh * id);
            }
            if (!ik_pre) {
                cpu_qsa_bf16_gemv(q4_layer(L).idx.k_proj.data, mh, n_embd, ik.data(), id, n_tok, n_embd, id);
            }
        } else {
            if (with_q && !iq_pre) {
                cpu_q4_up_gemv(q4_layer(L).idx.q_proj.data, q4_layer(L).idx.q_proj.type, mh, n_embd, iq.data(), inh * id, n_tok, n_embd,
                               inh * id);
            }
            if (!ik_pre) {
                cpu_q4_up_gemv(q4_layer(L).idx.k_proj.data, q4_layer(L).idx.k_proj.type, mh, n_embd, ik.data(), id, n_tok, n_embd, id);
            }
        }
    }
    const float * iqs = iq_pre ? iq_pre : iq.data();
    const float * iks = ik_pre ? ik_pre : ik.data();
    if (qtime) {
        q4x_.qsa.gemv_ms += qsa_now_ms() - q_t0;
        q_t0 = qsa_now_ms();
    }

    for (int i = 0; i < n_tok; i++) {
        const int pos = posv[i];
        const int slot = slotv[i];
        float * qrw = (with_q && !iq_pre) ? iq.data() + (size_t)i * inh * id : nullptr;
        const float * qr = with_q ? iqs + (size_t)i * inh * id : nullptr;
        if (qrw) {
            for (int h = 0; h < inh; h++) {
                head_norm(qrw + h * id, q4_layer(L).idx.q_norm, id, hp.rms_eps);
                head_rope(qrw + h * id, hp.n_rot, hp.rope_base, pos);
            }
        }
        if (with_q && qpos) {
            std::memcpy(qpos + (size_t)i * inh * id, qr, (size_t)inh * id * sizeof(float));
        }
        // The cache stores the *raw* indexer key: the reference pools the raw
        // keys of a block and only then applies k_norm + RoPE to the mean.
        // Normalising/rotating each member first and pooling those gives a
        // different (and wrong) selector score.
        const float * kr = iks + (size_t)i * id;

        if (slot < 0 || slot >= (int)q4x_.qsa.cache[(size_t)a].size()) {
            continue;
        }
        auto & sp = q4x_.qsa.cache[(size_t)a][(size_t)slot];
        if (!sp) {
            sp = std::make_unique<qsa_state::slot_state>();
            sp->pool.assign(((size_t)q4x_.qsa.max_seq / (size_t)kp + 1) * id, 0.f);
            sp->ring.assign((size_t)kp * id, 0.f);
            sp->n_pool = 0;
        }
        sp->n_pool = std::max(sp->n_pool, pos / kp); // complete groups before pos
        float * ring = sp->ring.data() + (size_t)(pos % kp) * id;
        for (int d = 0; d < id; d++) {
            ring[d] = f16r(kr[d]);
        }
        if (pos % kp == kp - 1) {
            float * pl = sp->pool.data() + (size_t)(pos / kp) * id;
            // kp = 4 is the shipped model: the loop below accumulates in the
            // same order and 1/4 is exact, so this is bit-identical to the
            // old four-term sum.
            const float inv_kp = 1.0f / (float)kp;
            for (int d = 0; d < id; d++) {
                float acc = sp->ring[(size_t)0 * id + d];
                for (int j = 1; j < kp; j++) {
                    acc += sp->ring[(size_t)j * id + d];
                }
                pl[d] = inv_kp * acc;
            }
            head_norm(pl, q4_layer(L).idx.k_norm, id, hp.rms_eps);
            head_rope(pl, hp.n_rot, hp.rope_base, pos - (kp - 1));
            for (int d = 0; d < id; d++) {
                pl[d] = f16r(pl[d]);
            }
            sp->n_pool = std::max(sp->n_pool, pos / kp + 1);
        }
        // Prefix cache: the raw indexer pool/ring is not in the paged KV and
        // cannot be recomputed from a restored prefix, so checkpoint it at each
        // completed 32-token block (the boundary's checkpoint slot is in the
        // forward's step_info).
        if (pcinf && pcinf->pc_active && (pos + 1) % kBlockSize == 0) {
            const int st = pcinf->pc_row_slot[((pos + 1) / kBlockSize) % kPcMapLen];
            if (st >= 0) {
                q4_pc_qsa_store(st, a, sp->pool.data(), sp->n_pool, sp->ring.data());
            }
        }
    }
    if (qtime) {
        q4x_.qsa.upd_ms += qsa_now_ms() - q_t0;
    }
}

// A slot is about to be handed to a new sequence (or the engine is being
// reset): the per-sequence host state must not survive it.  A fresh sequence
// starting at position 0 would overwrite every row/pool it reads before the
// selector runs, but that is an implicit argument and it does not hold for a
// resume that starts above 0 - and a deferred update must never land on the
// next sequence's slot.  qsa_flush() first keeps an in-flight staging copy from
// being reused, and applies any update for other slots that a multi-row batch
// might still own.
void engine::qsa_reset_slot(int slot) {
    if (!q4x_.qsa.on) {
        return;
    }
    static const bool qdbg = std::getenv("PF_Q4_QSADBG") != nullptr;
    qsa_flush();
    int had = 0;
    for (auto & layers : q4x_.qsa.cache) {
        for (size_t i = 0; i < layers.size(); i++) {
            if ((slot < 0 || (int)i == slot) && layers[i]) {
                had++;
            }
        }
    }
    for (auto & p : q4x_.qsa.pending) {
        bool match = slot < 0;
        for (int s : p.slot) {
            if (slot < 0 || s == slot) {
                match = true;
                break;
            }
        }
        if (!match) {
            continue;
        }
        p.active = false;
        p.has_ev = false;
        p.mixed = nullptr;
        p.pos.clear();
        p.slot.clear();
    }
    for (auto & layers : q4x_.qsa.cache) {
        for (size_t i = 0; i < layers.size(); i++) {
            if (slot < 0 || (int)i == slot) {
                layers[i].reset();
            }
        }
    }
    // The device ring/pool mirrors the host state; it must not survive a slot
    // handoff either (a deferred update or a device chain could still read it).
    if (q4x_.qsa.sel_dev) {
        if (slot < 0) {
            qsa_idx_slot_release_all();
            q4x_.qsa.idx_slot.resize((size_t)attn_layers());
        } else {
            for (int a = 0; a < (int)q4x_.qsa.idx_slot.size(); a++) {
                qsa_idx_slot_release(a, slot);
            }
        }
    }
    if (qdbg) {
        fprintf(stderr, "[qsa] reset slot=%d: cleared %d layer state(s)\n", slot, had);
    }
}

// Wait for the copies deferred by qsa_attention and apply the update of every
// pending layer.  Called at the end of a dense forward/chunk, after the LM
// head's kernels (or the chunk's remaining layers) have been enqueued: the
// event waits return once the copies are done, so the host work runs while the
// device is still busy.
double engine::qsa_flush() {
    if (!q4x_.qsa.on || q4x_.qsa.pending.empty()) {
        return 0.0;
    }
    static const bool qtime = si::env::flag("PF_QSA_TIME");
    const double t0 = qtime ? qsa_now_ms() : 0.0;
    bool any = false;
    for (auto & p : q4x_.qsa.pending) {
        if (p.active) {
            any = true;
            if (p.has_ev) {
                p.ev.wait();
                p.has_ev = false;
            }
        }
    }
    if (!any) {
        return 0.0;
    }
    if (qtime) {
        q4x_.qsa.copy_ms += qsa_now_ms() - t0;
    }
    for (auto & p : q4x_.qsa.pending) {
        if (!p.active) {
            continue;
        }
        qsa_apply_update(p.a, *p.L, p.n_tok, p.pos.data(), p.slot.data(), p.mixed, nullptr, false, nullptr, nullptr,
                         p.inf);
        p.active = false;
    }
    return qtime ? qsa_now_ms() - t0 : 0.0;
}

// PF_QSA_DEVATTN scratch, allocated on first use.  The compact pool uses the
// paged layout so attn's kv_row_data() indexes it unchanged; the sequential
// table presents the selected rows as positions 0..nsel-1.
void engine::qsa_devattn_ensure(int dev) {
    const hparams & hp = m.hp;
    if ((int)q4x_.qsa.dev.size() <= dev) {
        q4x_.qsa.dev.resize((size_t)dev + 1);
    }
    qsa_state::dev_buf & d = q4x_.qsa.dev[(size_t)dev];
    if (d.dev == dev && d.cap_keys > 0 && d.cap_pool > 0) {
        return;
    }
    const int per = qsa_pool_div(hp);
    const int need_keys = q4_hp(hp).idx_top_k + per + 1; // selected pools' `per` keys + the incomplete tail
    const int cap = ((need_keys + kBlockSize - 1) / kBlockSize) * kBlockSize;
    const int blocks = cap / kBlockSize;
    const int kvd = hp.n_head_kv * hp.head_dim;
    const int ntok_max = kMaxB * kMaxT;
    // The compact pool is `cap_wave` rows: each row's selected keys occupy
    // blocks [r*cap_blocks, (r+1)*cap_blocks), and the identity table maps them
    // with a cap_blocks stride.  Decode uses row 0 only, so its layout and its
    // values are unchanged.
    const int wave = std::max(1, std::min(kMaxB, q4x_.qsa.pf_wave));
    const size_t pool_blocks = (size_t)wave * (size_t)blocks;
    sycl::queue & q = dev_queue(dev);
    d.ck = (sycl::half *)dev_alloc_on(dev, pool_blocks * (size_t)kvd * kBlockSize * sizeof(sycl::half));
    d.cv = (sycl::half *)dev_alloc_on(dev, pool_blocks * (size_t)kvd * kBlockSize * sizeof(sycl::half));
    d.dsel = (int32_t *)dev_alloc_on(dev, (size_t)ntok_max * (size_t)cap * sizeof(int32_t));
    d.dcounts = (int32_t *)dev_alloc_on(dev, (size_t)ntok_max * sizeof(int32_t));
    d.tab = (int32_t *)dev_alloc_on(dev, pool_blocks * sizeof(int32_t));
    d.htab = (int32_t *)sycl::malloc_host(pool_blocks * sizeof(int32_t), q);
    const int npool_cap = q4x_.qsa.max_seq / per + 2;
    d.scores = (float *)dev_alloc_on(dev, (size_t)npool_cap * sizeof(float));
    d.dpools = (int32_t *)dev_alloc_on(dev, (size_t)npool_cap * sizeof(int32_t));
    if (!d.ck || !d.cv || !d.dsel || !d.dcounts || !d.tab || !d.htab || !d.scores || !d.dpools) {
        throw std::runtime_error("qwen4exp: QSA Stage 2 scratch allocation failed");
    }
    for (size_t i = 0; i < pool_blocks; i++) {
        d.htab[i] = (int32_t)i;
    }
    (void)q.memcpy(d.tab, d.htab, pool_blocks * sizeof(int32_t));
    d.cap_keys = cap;
    d.cap_blocks = blocks;
    d.cap_wave = wave;
    d.cap_pool = npool_cap;
    d.dev = dev;
    if (si::env::flag("PF_Q4_QSADBG")) {
        fprintf(stderr, "[qsa] devattn scratch dev=%d keys=%d blocks=%d wave=%d\n", dev, cap, blocks, wave);
    }
}

void engine::qsa_devattn_aux(int a, int dev) {
    qsa_devattn_ensure(dev);
    if ((int)q4x_.qsa.aux.size() <= a) {
        q4x_.qsa.aux.resize((size_t)a + 1);
    }
    qsa_state::dev_aux & x = q4x_.qsa.aux[(size_t)a];
    if (x.hsel) {
        return;
    }
    qsa_state::dev_buf & d = q4x_.qsa.dev[(size_t)dev];
    sycl::queue & q = dev_queue(dev);
    const size_t ntok_max = (size_t)kMaxB * kMaxT;
    x.hsel = (int32_t *)sycl::malloc_host(ntok_max * (size_t)d.cap_keys * sizeof(int32_t), q);
    x.hcnt = (int32_t *)sycl::malloc_host(ntok_max * sizeof(int32_t), q);
    // The decode `sel` staging is one row, so two buffers of cap_keys are tiny
    // next to the prefill staging above.
    x.dsel_stage[0] = (int32_t *)sycl::malloc_host((size_t)d.cap_keys * sizeof(int32_t), q);
    x.dsel_stage[1] = (int32_t *)sycl::malloc_host((size_t)d.cap_keys * sizeof(int32_t), q);
    x.sinfo = (step_info *)dev_alloc_on(dev, sizeof(step_info));
    if (!x.hsel || !x.hcnt || !x.sinfo || !x.dsel_stage[0] || !x.dsel_stage[1]) {
        throw std::runtime_error("qwen4exp: QSA Stage 2 aux allocation failed");
    }
    x.dev = dev;
}

void engine::qsa_devattn_run(int a, int dev, int slot, int pos, const std::vector<int> & sel, const float * qbuf,
                             float * attn_out) {
    (void)pos;
    const hparams & hp = m.hp;
    const int nh = hp.n_head, nkv = hp.n_head_kv, hd = hp.head_dim;
    const int nsel = (int)sel.size();
    if (nsel <= 0) {
        return;
    }
    qsa_devattn_aux(a, dev);
    qsa_state::dev_buf & d = q4x_.qsa.dev[(size_t)dev];
    qsa_state::dev_aux & x = q4x_.qsa.aux[(size_t)a];
    if (nsel > d.cap_keys) {
        throw std::runtime_error("qwen4exp: QSA Stage 2 sel exceeds the compact-pool capacity");
    }
    const int nblk = (nsel + kBlockSize - 1) / kBlockSize;
    sycl::queue & q = dev_queue(dev);
    // PF_QSA_EARLY: issue the `sel` H2D without a host wait so it does not
    // drain the layer's QKV/rope chain.  Two pinned buffers are round-robined;
    // before a buffer is rewritten, the copy that last read it (two calls
    // earlier, i.e. the previous token's use of this parity) is waited for.
    // That event is ordinarily long complete, and an event wait only covers
    // its own copy, not the queue tail.  The gather below is submitted after
    // this H2D on the same in-order queue, so the device still orders it.
    if (q4x_.qsa.early) {
        const int par = x.dsel_par;
        x.dsel_par ^= 1;
        if (x.dsel_ev_valid[par]) {
            x.dsel_ev[par].wait();
            x.dsel_ev_valid[par] = false;
        }
        std::memcpy(x.dsel_stage[par], sel.data(), (size_t)nsel * sizeof(int32_t));
        x.dsel_ev[par] = q.memcpy(d.dsel, x.dsel_stage[par], (size_t)nsel * sizeof(int32_t));
        x.dsel_ev_valid[par] = true;
    } else {
        // The pinned staging is per layer, but it is reused by the next token;
        // wait for the (small) copy so the host write cannot race it.
        std::memcpy(x.hsel, sel.data(), (size_t)nsel * sizeof(int32_t));
        sycl::event e = q.memcpy(d.dsel, x.hsel, (size_t)nsel * sizeof(int32_t));
        e.wait();
    }
    const char * kp = nullptr;
    const char * vp = nullptr;
    const char * ksc = nullptr;
    const char * vsc = nullptr;
    kv_layer_ptrs(a, kp, vp, ksc, vsc);
    qsa_sinfo_launch(q, x.sinfo, nsel);
    qsa_gather_launch(q, kp, vp, d_tables + (size_t)slot * max_blocks, d.dsel, nsel, nkv, hd, kBlockSize, d.ck,
                      d.cv);
    float * part = multi_dev ? as_[(size_t)dev].partials_dec : d_partials_dec;
    const int nsp = std::max(1, dec_splits);
    compute_backend & cbe = multi_dev ? *backends_[(size_t)dev] : *be;
    // The fused form (out != null, n_splits == 1) would gate and write in one
    // kernel; the split form is much wider and attn_combine applies the same
    // QBUF gate once.  group=0 keeps the classic split kernel (the grouped one
    // assumes the paged geometry and is not needed here).
    cbe.attn(qbuf, qbuf, d.ck, d.cv, part, d.tab, nh, nkv, hd, nsp, x.sinfo, hp.attn_scale, nblk, 1, 1, nullptr, 0,
             nullptr, nullptr);
    cbe.attn_combine(part, qbuf, attn_out, x.sinfo, nh, hd, nsp, 1, 1);
    if (si::env::flag("PF_Q4_QSADBG")) {
        fprintf(stderr, "[qsa] devattn a=%d slot=%d nsel=%d nblk=%d nsp=%d\n", a, slot, nsel, nblk, nsp);
    }
}

// Sparse prefill on the device.  The host has already run the selector for
// every token into `hsel`/`hcnt` (row stride is the compact-pool capacity), so
// the selected sets are exactly the Stage 1 ones.  Each wave compacts `T` rows
// into its own slice of the compact pool, then the ordinary paged-attention
// kernels run over it.  There is deliberately no per-wave host wait: the queue
// is in-order, and the only host-side copy is the one H2D of the ids/counts
// per layer (the pinned staging is per layer, so a later layer cannot overwrite
// it before the copy has been consumed).
void engine::qsa_devattn_prefill_run(int a, int dev, int n_tok, int slot, const int32_t * hsel, const int32_t * hcnt,
                                    const float * qbuf, float * attn_out) {
    if (n_tok <= 0) {
        return;
    }
    const hparams & hp = m.hp;
    const int nh = hp.n_head, nkv = hp.n_head_kv, hd = hp.head_dim;
    if (n_tok > kMaxB * kMaxT) {
        throw std::runtime_error("qwen4exp: QSA prefill chunk exceeds the device-path row capacity");
    }
    qsa_devattn_aux(a, dev);
    qsa_state::dev_buf & d = q4x_.qsa.dev[(size_t)dev];
    qsa_state::dev_aux & x = q4x_.qsa.aux[(size_t)a];
    if (slot < 0) {
        throw std::runtime_error("qwen4exp: QSA prefill device path needs a valid slot");
    }
    sycl::queue & q = dev_queue(dev);
    const size_t nsel_bytes = (size_t)n_tok * (size_t)d.cap_keys * sizeof(int32_t);
    std::memcpy(x.hsel, hsel, nsel_bytes);
    std::memcpy(x.hcnt, hcnt, (size_t)n_tok * sizeof(int32_t));
    // Both copies are queued behind the layer's own kernels on the in-order
    // compute queue; gather reads them immediately after.
    (void)q.memcpy(d.dsel, x.hsel, nsel_bytes);
    (void)q.memcpy(d.dcounts, x.hcnt, (size_t)n_tok * sizeof(int32_t));

    const char * kp = nullptr;
    const char * vp = nullptr;
    const char * ksc = nullptr;
    const char * vsc = nullptr;
    kv_layer_ptrs(a, kp, vp, ksc, vsc);
    const int32_t * src = d_tables + (size_t)slot * max_blocks;
    const int qstride = nh * 2 * hd;
    float * part = multi_dev ? as_[(size_t)dev].partials_dec : d_partials_dec;
    // The decode split count is what makes the decode path fill the card; the
    // same grid over a cap_wave-row wave is the shape this path was measured
    // with.  PF_QSA_PF_NSP overrides it for the A/B (0 = decode splits).
    static const int nsp_env = si::env::i32("PF_QSA_PF_NSP", 0);
    const int nsp_max = std::max(1, dec_splits);
    // partials_dec is sized for dec_splits splits; an override may only lower it.
    const int nsp = (nsp_env > 0 && nsp_env < nsp_max) ? nsp_env : nsp_max;
    compute_backend & cbe = multi_dev ? *backends_[(size_t)dev] : *be;
    const int W = std::min(d.cap_wave, n_tok);
    for (int w = 0; w < n_tok; w += W) {
        const int T = std::min(W, n_tok - w);
        qsa_sinfo_multi_launch(q, x.sinfo, d.dcounts + w, T);
        qsa_gather_rows_launch(q, kp, vp, src, d.dsel + (size_t)w * d.cap_keys, d.dcounts + w, d.cap_keys,
                               d.cap_blocks, T, nkv, hd, kBlockSize, d.ck, d.cv);
        // group = 0 keeps the classic split kernel: the grouped one assumes the
        // paged geometry and is not needed here (same call as the decode path).
        cbe.attn(qbuf + (size_t)w * qstride, qbuf + (size_t)w * qstride, d.ck, d.cv, part, d.tab, nh, nkv, hd, nsp,
                 x.sinfo, hp.attn_scale, d.cap_blocks, T, 1, nullptr, 0, nullptr, nullptr);
        cbe.attn_combine(part, qbuf + (size_t)w * qstride, attn_out + (size_t)w * nh * hd, x.sinfo, nh, hd, nsp, T,
                         1);
    }
    if (si::env::flag("PF_Q4_QSADBG")) {
        fprintf(stderr, "[qsa] devattn prefill a=%d slot=%d n_tok=%d wave=%d nsp=%d\n", a, slot, n_tok, W, nsp);
    }
}

// Gather `npos` (ascending) token positions of one (layer, slot) from the
// paged K/V into compact f32 rows [npos][n_head_kv*head_dim].  The paged pool is
// the only K/V copy the host path reads: there is no per-slot mirror, so the
// same values the device attention would see are what the host attention sees,
// and a prefix-cache resume with a quantized KV type is exact.  One D2H copy
// per touched block (and its scale plane) makes this O(npos/32) copies instead
// of O(npos); a prefill chunk passes the union of its rows' selections.
void engine::qsa_gather_positions(int a, int slot, const int32_t * positions, int npos) {
    if (npos <= 0 || a < 0 || a >= (int)q4x_.qsa.stage.size() || slot < 0 || slot >= state_slots_) {
        return;
    }
    qsa_state::stage_t & st = q4x_.qsa.stage[(size_t)a];
    const hparams & hp = m.hp;
    const int hd = hp.head_dim, nkv = hp.n_head_kv;
    const int kvd = nkv * hd, kbs = kBlockSize;
    const int nq = hd / kI8Q;
    const kv_dtype_t kdt = kv_k_dtype(), vdt = kv_v_dtype();
    const bool ksc = kv_dtype_has_scales(kdt), vsc = kv_dtype_has_scales(vdt);
    const size_t krow = kv_dtype_row_bytes(kdt, hd), vrow = kv_dtype_row_bytes(vdt, hd);
    const size_t kblk = kv_block_bytes(), vblk = kv_v_block_bytes();
    const size_t kscblk = (size_t)nkv * (size_t)kbs * (size_t)nq * sizeof(sycl::half);
    const size_t vscblk = (size_t)nkv * (size_t)kbs * (size_t)nq * sizeof(sycl::half);
    const char * kp = nullptr, * vp = nullptr, * ks = nullptr, * vs = nullptr;
    kv_layer_ptrs(a, kp, vp, ks, vs);
    if (!kp || !vp) {
        return;
    }
    const int dev = attn_dev(a);
    const bool cpu = multi_dev ? (dev_kind_[(size_t)dev] == 1) : cpu_mode;
    sycl::queue & qd = dev_queue(dev);
    static const bool qdbg = si::env::flag("PF_Q4_QSADBG");
    if (qdbg) {
        fprintf(stderr,
                "[qsa] gather a=%d slot=%d npos=%d kdt=%d vdt=%d kblk=%zu vblk=%zu krow=%zu vrow=%zu "
                "kp=%p vp=%p ks=%p vs=%p cpu=%d\n",
                a, slot, npos, (int)kdt, (int)vdt, kblk, vblk, krow, vrow, (const void *)kp, (const void *)vp,
                (const void *)ks, (const void *)vs, (int)cpu);
    }

    // Group the (ascending) positions by physical block.  A selected pool never
    // straddles a block boundary because kBlockSize is a multiple of kp.
    st.gblk.clear();
    st.grange.clear();
    for (int j = 0; j < npos;) {
        const int b = h_tables[(size_t)slot * max_blocks + (size_t)(positions[j] / kbs)];
        const int j0 = j;
        while (j < npos && h_tables[(size_t)slot * max_blocks + (size_t)(positions[j] / kbs)] == b) {
            j++;
        }
        st.gblk.push_back(b);
        st.grange.push_back(j0);
        st.grange.push_back(j);
    }
    const int nb = (int)st.gblk.size();
    if (nb <= 0) {
        return;
    }
    // Coalesce consecutive physical block ids into runs.  A sequence's block
    // table is normally one contiguous run (blocks are handed out lowest-first),
    // so this turns ~nblocks D2H calls per layer into one and is what keeps the
    // host gather from costing ~20 ms/layer in copy-call overhead.
    st.gstage.assign((size_t)nb, 0);
    st.grun_start.clear();
    st.grun_len.clear();
    int stage_blocks = 0;
    for (int i = 0; i < nb;) {
        int j = i + 1;
        while (j < nb && st.gblk[(size_t)j] == st.gblk[(size_t)(j - 1)] + 1) {
            j++;
        }
        const int b0 = st.gblk[(size_t)i];
        for (int k = i; k < j; k++) {
            st.gstage[(size_t)k] = stage_blocks + (st.gblk[(size_t)k] - b0);
        }
        st.grun_start.push_back(i);
        st.grun_len.push_back(j - i);
        stage_blocks += j - i;
        i = j;
    }
    auto grow = [&](uint8_t *& p, size_t & cap, size_t need) {
        if (p && cap >= need) {
            return;
        }
        if (p) {
            sycl::free(p, qd);
        }
        p = (uint8_t *)sycl::malloc_host(need ? need : 1, qd);
        if (!p) {
            throw std::runtime_error("qwen4exp: QSA gather staging allocation failed");
        }
        cap = need;
    };
    grow(st.sbk, st.cap_sbk, (size_t)stage_blocks * kblk);
    grow(st.sbv, st.cap_sbv, (size_t)stage_blocks * vblk);
    if (ksc) {
        grow(st.sbsk, st.cap_sbsk, (size_t)stage_blocks * kscblk);
    }
    if (vsc) {
        grow(st.sbsv, st.cap_sbsv, (size_t)stage_blocks * vscblk);
    }
    for (size_t r = 0; r < st.grun_start.size(); r++) {
        const int i = st.grun_start[r];
        const int b = st.gblk[(size_t)i];
        const size_t bytes_k = (size_t)st.grun_len[r] * kblk;
        const size_t bytes_v = (size_t)st.grun_len[r] * vblk;
        uint8_t * dk = st.sbk + (size_t)st.gstage[(size_t)i] * kblk;
        uint8_t * dv = st.sbv + (size_t)st.gstage[(size_t)i] * vblk;
        if (cpu) {
            std::memcpy(dk, kp + (size_t)b * kblk, bytes_k);
            std::memcpy(dv, vp + (size_t)b * vblk, bytes_v);
            if (ksc) {
                std::memcpy(st.sbsk + (size_t)st.gstage[(size_t)i] * kscblk, ks + (size_t)b * kscblk,
                            (size_t)st.grun_len[r] * kscblk);
            }
            if (vsc) {
                std::memcpy(st.sbsv + (size_t)st.gstage[(size_t)i] * vscblk, vs + (size_t)b * vscblk,
                            (size_t)st.grun_len[r] * vscblk);
            }
        } else {
            qd.memcpy(dk, kp + (size_t)b * kblk, bytes_k);
            qd.memcpy(dv, vp + (size_t)b * vblk, bytes_v);
            if (ksc) {
                qd.memcpy(st.sbsk + (size_t)st.gstage[(size_t)i] * kscblk, ks + (size_t)b * kscblk,
                          (size_t)st.grun_len[r] * kscblk);
            }
            if (vsc) {
                qd.memcpy(st.sbsv + (size_t)st.gstage[(size_t)i] * vscblk, vs + (size_t)b * vscblk,
                          (size_t)st.grun_len[r] * vscblk);
            }
        }
    }
    if (!cpu) {
        qd.wait();
    }
    if (qdbg) {
        fprintf(stderr, "[qsa] gather copied nb=%d runs=%d first_blk=%d last_blk=%d\n", nb,
                (int)st.grun_start.size(), st.gblk.front(), st.gblk.back());
    }
    st.gk.resize((size_t)npos * (size_t)kvd);
    st.gv.resize((size_t)npos * (size_t)kvd);
    auto put_row = [&](const uint8_t * data, const uint8_t * scales, size_t rowb, kv_dtype_t dt, int so, int h,
                       float * dst) {
        const uint8_t * row = data + ((size_t)h * (size_t)kbs + (size_t)so) * rowb;
        const sycl::half * sc = scales ? (const sycl::half *)(scales + ((size_t)h * (size_t)kbs + (size_t)so)
                                                                           * (size_t)nq * sizeof(sycl::half))
                                       : nullptr;
        switch (dt) {
        case kv_dtype_t::f32:
            std::memcpy(dst, row, (size_t)hd * sizeof(float));
            break;
        case kv_dtype_t::f16:
            for (int d = 0; d < hd; d++) {
                uint16_t u;
                std::memcpy(&u, row + (size_t)d * 2, sizeof(u));
                dst[d] = ggml_half_to_float(u);
            }
            break;
        case kv_dtype_t::bf16:
            for (int d = 0; d < hd; d++) {
                uint16_t u;
                std::memcpy(&u, row + (size_t)d * 2, sizeof(u));
                dst[d] = ggml_bf16_to_float(u);
            }
            break;
        case kv_dtype_t::i8:
            // One scale per 32 values: keep the scale in a register across the
            // group so the loop auto-vectorises to a cvt + multiply.
            for (int d = 0; d < hd; d += kI8Q) {
                const float s = (float)sc[d / kI8Q];
                for (int k = 0; k < kI8Q; k++) {
                    dst[d + k] = (float)((const int8_t *)row)[d + k] * s;
                }
            }
            break;
        case kv_dtype_t::i4:
            for (int d = 0; d < hd; d += kI8Q) {
                const float s = (float)sc[d / kI8Q];
                for (int k = 0; k < kI8Q; k++) {
                    const int dd = d + k;
                    const uint8_t byte = row[dd >> 1];
                    const int v = (dd & 1) ? (byte >> 4) : (byte & 0xF);
                    dst[dd] = (float)((v ^ 8) - 8) * s;
                }
            }
            break;
        }
    };
    for (int i = 0; i < nb; i++) {
        const int j0 = st.grange[(size_t)i * 2], j1 = st.grange[(size_t)i * 2 + 1];
        const uint8_t * bkd = st.sbk + (size_t)st.gstage[(size_t)i] * kblk;
        const uint8_t * bvd = st.sbv + (size_t)st.gstage[(size_t)i] * vblk;
        const uint8_t * bks = ksc ? st.sbsk + (size_t)st.gstage[(size_t)i] * kscblk : nullptr;
        const uint8_t * bvs = vsc ? st.sbsv + (size_t)st.gstage[(size_t)i] * vscblk : nullptr;
        for (int j = j0; j < j1; j++) {
            const int so = positions[j] % kbs;
            for (int h = 0; h < nkv; h++) {
                put_row(bkd, bks, krow, kdt, so, h, st.gk.data() + (size_t)j * (size_t)kvd + (size_t)h * (size_t)hd);
                put_row(bvd, bvs, vrow, vdt, so, h, st.gv.data() + (size_t)j * (size_t)kvd + (size_t)h * (size_t)hd);
            }
        }
    }
}

bool engine::qsa_attention(int a, const layer_t & L, int dev, int n_tok, const step_info * inf, const float * mixed,
                           float * qbuf, const float * kbuf, const float * vbuf, float * attn_out) {
    if (!q4x_.qsa.on || !q4_layer(L).idx.ok() || a < 0 || a >= (int)q4x_.qsa.cache.size()) {
        return false;
    }
    const hparams & hp = m.hp;
    const int hd = hp.head_dim, nh = hp.n_head, nkv = hp.n_head_kv;
    const int id = q4_hp(hp).idx_head_dim, inh = q4_hp(hp).idx_n_head;
    const int kvd = nkv * hd;
    const int gqa = nh / nkv;
    const int kp = qsa_pool_div(hp);
    const int n_pool_max = q4_hp(hp).idx_top_k / kp;
    const int stride = hd * 2;

    auto is_cpu = [&](int d) { return multi_dev ? (dev_kind_[(size_t)d] == 1) : cpu_mode; };
    // The D2H copies are enqueued together and waited once: a copy on its own
    // `wait()` drains the in-order queue, and with the host MoE running beside
    // the device those drains were the bulk of the QSA cost.
    auto to_host_async = [&](void * dst, const void * src, size_t bytes) -> sycl::event {
        if (is_cpu(dev)) {
            std::memcpy(dst, src, bytes);
            return {};
        }
        return dev_queue(dev).memcpy(dst, src, bytes);
    };
    // The sparse attention's H2D rides the main in-order queue without a wait:
    // `wo` is enqueued after it, and the destination ATTN_OUT is not reused
    // until the next attention layer.  The source is this layer's pinned
    // staging, which outlives the copy (the engine syncs before the next
    // forward).
    auto from_host_async = [&](void * dst, const void * src, size_t bytes) -> sycl::event {
        if (is_cpu(dev)) {
            std::memcpy(dst, src, bytes);
            return {};
        }
        return dev_queue(dev).memcpy(dst, src, bytes);
    };

    static const bool qdbg = std::getenv("PF_Q4_QSADBG") != nullptr;
    static const bool qtime = si::env::flag("PF_QSA_TIME");
    double q_t0 = qtime ? qsa_now_ms() : 0.0;
    if (qdbg) {
        fprintf(stderr, "[qsa] enter a=%d n_tok=%d pos0=%d tpb=%d rows=%d\n", a, n_tok, inf->pos[0], inf->tpb,
                inf->n_rows);
    }
    const int n_embd = hp.n_embd;
    // `need` depends only on the positions, so it is known before any copy.
    // The indexer pool/ring must be warm at every token (so the selector is
    // ready when the context crosses the budget), but Q is read only by the
    // sparse attention, so its copy is skipped while the selector is
    // dense-equivalent.
    bool need = false;
    std::vector<int> posv((size_t)n_tok), slotv((size_t)n_tok);
    for (int i = 0; i < n_tok; i++) {
        const int row = inf->tpb == 1 ? i : i / inf->tpb;
        const int t = inf->tpb == 1 ? 0 : i % inf->tpb;
        posv[(size_t)i] = inf->pos[row] + t;
        slotv[(size_t)i] = inf->slot[row];
        const int slot = slotv[(size_t)i];
        if (slot >= 0 && slot < (int)q4x_.qsa.cache[(size_t)a].size() && posv[(size_t)i] + 1 > q4_hp(hp).idx_top_k) {
            need = true;
        }
    }

    // PF_QSA_SEL_DEV: the early chain already produced the selected key list
    // on the device (and the host pool was handed over before the current
    // token's update).  No host wait is needed: the gather/attention are
    // enqueued here, after qk_norm_rope, on the same in-order queue.
    bool dev_sel = false;
    if (q4x_.qsa.sel_dev && need && n_tok == 1 && a < (int)q4x_.qsa.early_ev.size()) {
        qsa_state::early_copy & ec = q4x_.qsa.early_ev[(size_t)a];
        if (ec.valid && ec.kind == 2 && ec.dev == dev) {
            dev_sel = true;
            ec.valid = false;
        }
    }
    if (dev_sel) {
        const int pos0 = posv[0];
        const int slot0 = slotv[0];
        const int n_vis = pos0 + 1;
        const int n_pool_vis = n_vis / kp;
        const int n_top = std::min(n_pool_vis, n_pool_max);
        const int nsel = n_top * kp + (n_vis - n_pool_vis * kp);
        const double q_att0 = qtime ? qsa_now_ms() : 0.0;
        qsa_devattn_run_dev(a, dev, slot0, pos0, nsel, qbuf, attn_out);
        if (qtime) {
            q4x_.qsa.attn_ms += qsa_now_ms() - q_att0;
            q4x_.qsa.rows++;
            q4x_.qsa.nsel += nsel;
        }
        if (a == 0) {
            const char * sp_path = std::getenv("PF_QSA_SCORE");
            const char * pp = std::getenv("PF_QSA_POOLS");
            const char * qp = std::getenv("PF_QSA_Q");
            if (sp_path || pp || qp) {
                sycl::queue & q = dev_queue(dev);
                qsa_state::dev_buf & d = q4x_.qsa.dev[(size_t)dev];
                if (sp_path) {
                    std::vector<float> h((size_t)n_pool_vis);
                    q.memcpy(h.data(), d.scores, h.size() * sizeof(float));
                    q.wait();
                    if (FILE * fp = fopen(sp_path, "wb")) {
                        fwrite(h.data(), sizeof(float), h.size(), fp);
                        fclose(fp);
                    }
                }
                if (pp) {
                    std::vector<int32_t> h((size_t)n_top);
                    q.memcpy(h.data(), d.dpools, h.size() * sizeof(int32_t));
                    q.wait();
                    if (FILE * fp = fopen(pp, "wb")) {
                        fwrite(h.data(), sizeof(int32_t), h.size(), fp);
                        fclose(fp);
                    }
                }
                if (qp) {
                    std::vector<float> h((size_t)inh * id);
                    q.memcpy(h.data(), q4x_.qsa.idx_dev_buf[(size_t)dev].q, h.size() * sizeof(float));
                    q.wait();
                    if (FILE * fp = fopen(qp, "wb")) {
                        fwrite(h.data(), sizeof(float), h.size(), fp);
                        fclose(fp);
                    }
                }
            }
        }
        return true;
    }

    // Dense-equivalent forwards (decode and a prefill tail/chunk) defer the
    // indexer update to the end of the forward.  Only `mixed` (10 KB/token) is
    // copied now: the main K/V is not mirrored, so there is no K/V D2H to
    // overlap.  The copy rides the main in-order queue, which orders it after
    // this layer's hc_mix (the source) and before the next layer's hc_mix.
    static const bool defer_on = SI_ENV_I32("PF_QSA_DEFER", 1) != 0;
    const bool defer = defer_on && !need && a < (int)q4x_.qsa.pending.size();
    if (defer) {
        qsa_stage_ensure(a, n_tok, false, false, dev);
        qsa_state::stage_t & st = q4x_.qsa.stage[(size_t)a];
        qsa_state::pending_t & p = q4x_.qsa.pending[(size_t)a];
        p.L = &L;
        p.a = a;
        p.n_tok = n_tok;
        p.dev = dev;
        p.pos.assign(posv.begin(), posv.end());
        p.slot.assign(slotv.begin(), slotv.end());
        p.inf = inf;
        p.mixed = st.mixed;
        const size_t mb = (size_t)n_tok * (size_t)n_embd * sizeof(float);
        if (is_cpu(dev)) {
            std::memcpy(p.mixed, mixed, mb);
            p.has_ev = false;
        } else {
            p.ev = dev_queue(dev).memcpy(p.mixed, mixed, mb);
            p.has_ev = true;
        }
        p.active = true;
        if (qtime) {
            q4x_.qsa.copy_ms += qsa_now_ms() - q_t0;
        }
        if (qdbg) {
            fprintf(stderr, "[qsa] deferred a=%d n_tok=%d\n", a, n_tok);
        }
        return false;
    }

    // Stage 2: the sparse GPU f16 path attends to the selected rows of the
    // paged pool on the device.  The selector still runs here and is unchanged;
    // only the main Q/K/V D2H and the host attention go away.
    //  * one token (decode): row 0 of the compact pool;
    //  * a prefill chunk (n_tok > 1): `cap_wave` rows per wave, and every row
    //    must name the same slot because the compact table is per sequence.
    // Dense forwards, CPU, non-f16 KV and a mixed-slot chunk keep the Stage 1
    // host path, including the mirror update.
    const bool f16kv = kv_k_dtype() == kv_dtype_t::f16 && kv_v_dtype() == kv_dtype_t::f16;
    const bool dev_path = q4x_.qsa.devattn && need && n_tok == 1 && !is_cpu(dev) && f16kv && hd == 256;
    bool slot_const = need && n_tok > 1;
    int chunk_slot = -1;
    if (slot_const) {
        for (int i = 0; i < n_tok; i++) {
            const int sl = slotv[(size_t)i];
            if (sl < 0 || sl >= (int)q4x_.qsa.cache[(size_t)a].size() || !q4x_.qsa.cache[(size_t)a][(size_t)sl]) {
                slot_const = false;
                break;
            }
            if (chunk_slot < 0) {
                chunk_slot = sl;
            } else if (chunk_slot != sl) {
                slot_const = false;
                break;
            }
        }
    }
    const bool dev_pf = q4x_.qsa.pf_dev && q4x_.qsa.devattn && slot_const && !is_cpu(dev) && f16kv && hd == 256
                        && n_tok <= kMaxB * kMaxT;
    const bool dev_attn = dev_path || dev_pf;

    // The sparse path runs inline and reads the mirror: apply any update a
    // preceding dense chunk deferred before the selector sees the pool.
    qsa_flush();
    qsa_stage_ensure(a, n_tok, need && !dev_attn, need && !dev_attn, dev);
    qsa_state::stage_t & st = q4x_.qsa.stage[(size_t)a];
    qsa_state::host_stage_t * hs = (!dev_attn && dev >= 0 && dev < (int)q4x_.qsa.host_stage.size())
                              ? &q4x_.qsa.host_stage[(size_t)dev]
                              : nullptr;

    const size_t mb = (size_t)n_tok * (size_t)n_embd * sizeof(float);
    const size_t qb = (size_t)n_tok * (size_t)nh * (size_t)stride * sizeof(float);
    (void)kbuf;
    (void)vbuf;
    // PF_QSA_EARLY: the dev-attn sparse path's `mixed` D2H was enqueued by
    // qsa_early_mixed() before this layer's QKV GEMVs.  Consume that event and
    // wait for the 10 KB copy alone.  The dev-attn path reads nothing else
    // from the device (update_mirror is off), so the old full queue drain was
    // only waiting for the QKV/rope chain the host does not need yet.
    bool early = false;
    bool idx_early = false;
    sycl::event early_ev;
    if (q4x_.qsa.early && dev_attn && !is_cpu(dev) && a < (int)q4x_.qsa.early_ev.size()) {
        qsa_state::early_copy & ec = q4x_.qsa.early_ev[(size_t)a];
        if (ec.valid && ec.dev == dev) {
            early = true;
            idx_early = ec.kind == 1;
            early_ev = ec.ev;
            ec.valid = false;
        }
    }
    if (!early && !idx_early) {
        (void)to_host_async(st.mixed, mixed, mb);
    }
    if (need && !dev_attn) {
        (void)to_host_async(hs->q, qbuf, qb);
    }
    if (!is_cpu(dev)) {
        if (early) {
            early_ev.wait();
        } else {
            dev_queue(dev).wait();
        }
    }
    if (qtime) {
        q4x_.qsa.copy_ms += qsa_now_ms() - q_t0;
        q_t0 = qsa_now_ms();
    }

    std::vector<float> qpos((size_t)n_tok * inh * id);
    qsa_apply_update(a, L, n_tok, posv.data(), slotv.data(), st.mixed, qpos.data(), need,
                     idx_early ? st.iq : nullptr, idx_early ? st.ik : nullptr, inf);

    if (qdbg) {
        fprintf(stderr, "[qsa] caches updated need=%d\n", (int)need);
    }
    if (a == 0) {
        if (const char * qp = std::getenv("PF_QSA_Q")) {
            if (FILE * fp = fopen(qp, "wb")) {
                fwrite(qpos.data() + (size_t)(n_tok - 1) * inh * id, sizeof(float), (size_t)inh * id, fp);
                fclose(fp);
            }
        }
        if (const char * mp = std::getenv("PF_QSA_MIX")) {
            if (FILE * fp = fopen(mp, "wb")) {
                fwrite(st.mixed + (size_t)(n_tok - 1) * n_embd, sizeof(float), (size_t)n_embd, fp);
                fclose(fp);
            }
        }
    }
    if (!need) {
        return false; // dense attention is exactly equivalent here
    }

    if (const char * kp = std::getenv("PF_QSA_POOLK")) {
        if (a < (int)q4x_.qsa.cache.size() && !q4x_.qsa.cache[(size_t)a].empty() && q4x_.qsa.cache[(size_t)a][0]) {
            if (FILE * fp = fopen(kp, "wb")) {
                fwrite(q4x_.qsa.cache[(size_t)a][0]->pool.data(), sizeof(float), (size_t)id, fp);
                fclose(fp);
            }
        }
    }
    // The selector is identical for both paths: ggml_top_k's CPU form is
    // std::partial_sort over the score with `a > b`, and reproducing it with
    // the same algorithm/comparator is what keeps equal (rectified-to-zero)
    // scores selecting the same pools.  `select_row` returns false only for an
    // invalid slot, which the host loop treats as a zero row.
    auto select_row = [&](int i, std::vector<int> & sel) -> bool {
        const int pos = posv[(size_t)i];
        const int slot = slotv[(size_t)i];
        if (slot < 0 || slot >= (int)q4x_.qsa.cache[(size_t)a].size() || !q4x_.qsa.cache[(size_t)a][(size_t)slot]) {
            return false;
        }
        const qsa_state::slot_state & sp = *q4x_.qsa.cache[(size_t)a][(size_t)slot];
        const int n_vis = pos + 1;
        const int n_pool_vis = n_vis / kp;
        sel.clear();
        if (n_pool_vis <= n_pool_max) {
            sel.resize((size_t)n_vis);
            for (int p = 0; p < n_vis; p++) {
                sel[(size_t)p] = p;
            }
        } else {
            const float * qr = qpos.data() + (size_t)i * inh * id;
            const float inv = 1.0f / std::sqrt((float)id);
            std::vector<float> scores((size_t)n_pool_vis);
            cpu_qsa_score_pools(qr, sp.pool.data(), n_pool_vis, id, inh, inv, scores.data());
            if (a == 0 && i == n_tok - 1) {
                if (const char * sp_path = std::getenv("PF_QSA_SCORE")) {
                    if (FILE * fp = fopen(sp_path, "wb")) {
                        fwrite(scores.data(), sizeof(float), scores.size(), fp);
                        fclose(fp);
                    }
                }
            }
            std::vector<int> idx((size_t)n_pool_vis);
            for (int p = 0; p < n_pool_vis; p++) {
                idx[(size_t)p] = p;
            }
            const int n_top = std::min(n_pool_vis, n_pool_max);
            // Canonical order: score descending, pool index ascending on a tie.
            // The device selector uses the same rank key (score, inverted
            // index), so the two select the same set by construction; only
            // fp-noise boundary scores can move a pool.
            std::partial_sort(idx.begin(), idx.begin() + n_top, idx.end(), [&](int a, int b) {
                const float sa = scores[(size_t)a], sb = scores[(size_t)b];
                return sa > sb || (sa == sb && a < b);
            });
            std::vector<int> pools(idx.begin(), idx.begin() + n_top);
            if (a == 0 && i == n_tok - 1) {
                if (const char * pp = std::getenv("PF_QSA_POOLS")) {
                    if (FILE * fp = fopen(pp, "wb")) {
                        fwrite(pools.data(), sizeof(int), pools.size(), fp);
                        fclose(fp);
                    }
                }
            }
            std::sort(pools.begin(), pools.end());
            sel.reserve(pools.size() * (size_t)kp + (size_t)(kp - 1));
            for (int p : pools) {
                for (int j = 0; j < kp; j++) {
                    sel.push_back(p * kp + j);
                }
            }
            for (int p = n_pool_vis * kp; p <= pos; p++) {
                sel.push_back(p);
            }
        }
        return true;
    };
    bool used_dev = false;
    if (!dev_attn) {
        if (hs->out_pending) {
            hs->out_ev.wait();
            hs->out_pending = false;
        }
        std::memset(hs->out, 0, (size_t)n_tok * (size_t)nh * (size_t)hd * sizeof(float));
    }
    if (dev_pf) {
        qsa_devattn_aux(a, dev);
        qsa_state::dev_buf & d = q4x_.qsa.dev[(size_t)dev];
        qsa_state::dev_aux & x = q4x_.qsa.aux[(size_t)a];
        const double q_sel0 = qtime ? qsa_now_ms() : 0.0;
        // The selection is per token and independent: the shared read-only
        // pool makes it parallel, and each row writes its own slice.
        par(n_tok, [&](int i) {
            std::vector<int> sel;
            const bool ok = select_row(i, sel);
            const int nsel = ok ? (int)sel.size() : 0;
            int32_t * dst = x.hsel + (size_t)i * (size_t)d.cap_keys;
            for (int j = 0; j < nsel; j++) {
                dst[j] = (int32_t)sel[(size_t)j];
            }
            x.hcnt[i] = nsel;
            if (qdbg) {
                fprintf(stderr, "[qsa] row=%d pos=%d slot=%d nsel=%d\n", i, posv[(size_t)i], slotv[(size_t)i], nsel);
            }
        });
        if (qtime) {
            q4x_.qsa.sel_ms += qsa_now_ms() - q_sel0;
            q4x_.qsa.rows += n_tok;
            for (int i = 0; i < n_tok; i++) {
                q4x_.qsa.nsel += x.hcnt[i];
            }
        }
        const double q_att0 = qtime ? qsa_now_ms() : 0.0;
        qsa_devattn_prefill_run(a, dev, n_tok, chunk_slot, x.hsel, x.hcnt, qbuf, attn_out);
        if (qtime) {
            q4x_.qsa.attn_ms += qsa_now_ms() - q_att0;
        }
        used_dev = true;
    }
    if (dev_path) {
        for (int i = 0; i < n_tok; i++) {
            const int pos = posv[(size_t)i];
            const int slot = slotv[(size_t)i];
            const double q_row0 = qtime ? qsa_now_ms() : 0.0;
            std::vector<int> sel;
            if (!select_row(i, sel)) {
                continue;
            }
            const int nsel = (int)sel.size();
            if (qtime) {
                q4x_.qsa.sel_ms += qsa_now_ms() - q_row0;
                q4x_.qsa.rows++;
                q4x_.qsa.nsel += nsel;
            }
            if (qdbg) {
                fprintf(stderr, "[qsa] row=%d pos=%d slot=%d nsel=%d\n", i, pos, slot, nsel);
            }
            const double q_att0 = qtime ? qsa_now_ms() : 0.0;
            qsa_devattn_run(a, dev, slot, pos, sel, qbuf, attn_out);
            used_dev = true;
            if (qtime) {
                q4x_.qsa.attn_ms += qsa_now_ms() - q_att0;
            }
        }
    } else if (!dev_pf) {
        // Host sparse attention.  There is no host K/V mirror: the selected
        // rows are gathered from the paged pool and dequantized into a compact
        // f16 buffer.  A prefill chunk gathers the UNION of its rows'
        // selections once per layer (the alternative, one gather per row, is
        // O(n_tok * top_k) D2H traffic), then every row maps its own selection
        // into that compact buffer.
        int n_vis_max = 0;
        for (int i = 0; i < n_tok; i++) {
            n_vis_max = std::max(n_vis_max, posv[(size_t)i] + 1);
        }
        bool one_slot = n_tok > 0;
        for (int i = 1; i < n_tok; i++) {
            if (slotv[(size_t)i] != slotv[0]) {
                one_slot = false;
            }
        }
        const double q_sel0 = qtime ? qsa_now_ms() : 0.0;
        if (one_slot && n_tok > 0) {
            st.gused.assign((size_t)n_vis_max, 0);
            st.goff.assign((size_t)n_tok + 1, 0);
            st.gcnt.assign((size_t)n_tok, 0);
            st.gsel_all.clear();
            std::vector<int> sel;
            for (int i = 0; i < n_tok; i++) {
                sel.clear();
                if (!select_row(i, sel)) {
                    st.goff[(size_t)i + 1] = st.goff[(size_t)i];
                    continue;
                }
                st.gcnt[(size_t)i] = (int)sel.size();
                st.goff[(size_t)i + 1] = st.goff[(size_t)i] + (int)sel.size();
                st.gsel_all.insert(st.gsel_all.end(), sel.begin(), sel.end());
                for (int p : sel) {
                    if (p >= 0 && p < n_vis_max) {
                        st.gused[(size_t)p] = 1;
                    }
                }
            }
            st.gpos.clear();
            for (int p = 0; p < n_vis_max; p++) {
                if (st.gused[(size_t)p]) {
                    st.gpos.push_back(p);
                }
            }
            if (!st.gpos.empty()) {
                qsa_gather_positions(a, slotv[0], st.gpos.data(), (int)st.gpos.size());
                st.gmap.assign((size_t)n_vis_max, -1);
                for (int j = 0; j < (int)st.gpos.size(); j++) {
                    st.gmap[(size_t)st.gpos[(size_t)j]] = j;
                }
            }
        }
        if (qtime) {
            q4x_.qsa.sel_ms += qsa_now_ms() - q_sel0;
            q4x_.qsa.rows += n_tok;
        }
        const double q_att0 = qtime ? qsa_now_ms() : 0.0;
        if (one_slot && n_tok > 0) {
            for (int i = 0; i < n_tok; i++) {
                const int nsel = st.gcnt[(size_t)i];
                if (nsel <= 0) {
                    continue;
                }
                if (qdbg) {
                    fprintf(stderr, "[qsa] row=%d pos=%d slot=%d nsel=%d (union=%d)\n", i, posv[(size_t)i],
                            slotv[(size_t)i], nsel, (int)st.gpos.size());
                }
                st.gsel.resize((size_t)nsel);
                for (int j = 0; j < nsel; j++) {
                    st.gsel[(size_t)j] = st.gmap[(size_t)st.gsel_all[(size_t)st.goff[(size_t)i] + (size_t)j]];
                }
                if (qtime) {
                    q4x_.qsa.nsel += nsel;
                }
                par(nh, [&](int h) {
                    const int kvh = h / gqa;
                    const float * q = hs->q + (size_t)i * nh * stride + (size_t)h * stride;
                    float * o = hs->out + (size_t)i * nh * hd + (size_t)h * hd;
                    cpu_qsa_attn_head(q, st.gk.data() + (size_t)kvh * hd, st.gv.data() + (size_t)kvh * hd, kvd,
                                      st.gsel.data(), nsel, hd, hp.attn_scale, o);
                });
            }
        } else {
            // Batched rows on different slots (not used by qwen4exp's one-row
            // decode, but the host path is generic): gather each row alone.
            for (int i = 0; i < n_tok; i++) {
                std::vector<int> sel;
                if (!select_row(i, sel)) {
                    continue;
                }
                const int nsel = (int)sel.size();
                if (nsel <= 0) {
                    continue;
                }
                const int nv = posv[(size_t)i] + 1;
                st.gused.assign((size_t)nv, 0);
                for (int p : sel) {
                    st.gused[(size_t)p] = 1;
                }
                st.gpos.clear();
                for (int p = 0; p < nv; p++) {
                    if (st.gused[(size_t)p]) {
                        st.gpos.push_back(p);
                    }
                }
                qsa_gather_positions(a, slotv[(size_t)i], st.gpos.data(), (int)st.gpos.size());
                st.gmap.assign((size_t)nv, -1);
                for (int j = 0; j < (int)st.gpos.size(); j++) {
                    st.gmap[(size_t)st.gpos[(size_t)j]] = j;
                }
                st.gsel.resize((size_t)nsel);
                for (int j = 0; j < nsel; j++) {
                    st.gsel[(size_t)j] = st.gmap[(size_t)sel[(size_t)j]];
                }
                if (qtime) {
                    q4x_.qsa.rows++;
                    q4x_.qsa.nsel += nsel;
                }
                par(nh, [&](int h) {
                    const int kvh = h / gqa;
                    const float * q = hs->q + (size_t)i * nh * stride + (size_t)h * stride;
                    float * o = hs->out + (size_t)i * nh * hd + (size_t)h * hd;
                    cpu_qsa_attn_head(q, st.gk.data() + (size_t)kvh * hd, st.gv.data() + (size_t)kvh * hd, kvd,
                                      st.gsel.data(), nsel, hd, hp.attn_scale, o);
                });
            }
        }
        if (qtime) {
            q4x_.qsa.attn_ms += qsa_now_ms() - q_att0;
        }
    }
    if (qdbg) {
        fprintf(stderr, "[qsa] attention done, writing out\n");
    }
    if (const char * dp = si::env::str("PF_QSA_ATTN_DUMP")) {
        if (a == qsa_dump_layer()) {
            // PF_QSA_ATTN_DUMP_ALL writes every row of the chunk (the per-row
            // oracle for a device/host attention A/B); the default writes the
            // last row, which is what the Stage 2 decode check uses.
            const bool dump_all = si::env::flag("PF_QSA_ATTN_DUMP_ALL");
            const int i = n_tok - 1;
            const int dpos = posv[(size_t)i];
            const size_t nrow = (size_t)nh * (size_t)hd;
            const size_t nout = dump_all ? (size_t)n_tok * nrow : nrow;
            const size_t off = dump_all ? 0 : (size_t)i * nrow;
            char path[1024];
            snprintf(path, sizeof(path), dump_all ? "%s.qsa.%d.%d.all.bin" : "%s.qsa.%d.%d.bin", dp, a, dpos);
            if (FILE * fp = fopen(path, "wb")) {
                if (used_dev) {
                    std::vector<float> hout(nout);
                    sycl::queue & q = dev_queue(dev);
                    q.memcpy(hout.data(), attn_out + off, nout * sizeof(float));
                    q.wait();
                    fwrite(hout.data(), sizeof(float), nout, fp);
                } else {
                    fwrite(hs->out + off, sizeof(float), nout, fp);
                }
                fclose(fp);
            }
        }
    }
    if (!used_dev) {
        // The host path filled st.out; a dev-path row with an invalid slot has
        // no host buffer written (the slot guard above skipped it), and the
        // Stage 1 host path would have left that row zeroed.
        if (dev_path) {
            std::memset(hs->out, 0, (size_t)n_tok * (size_t)nh * (size_t)hd * sizeof(float));
        }
        hs->out_ev = from_host_async(attn_out, hs->out, (size_t)n_tok * (size_t)nh * (size_t)hd * sizeof(float));
        hs->out_pending = !is_cpu(dev);
    }
    if (qdbg) {
        fprintf(stderr, "[qsa] out written\n");
    }
    return true;
}

} // namespace si
