// QSA (query-side sparse attention) for qwen4exp.
//
// Up to `indexer_top_k` keys the reference's selector covers every visible
// key, so dense attention is exactly equivalent; beyond it the model attends
// to the top `top_k` keys pooled in blocks of `compress_ratio` plus the
// incomplete tail.  This implementation keeps the selector and the attention
// on the host:
//   * the indexer weights are small and already in the host mmap;
//   * full-attention layers keep a host mirror of their own K/V (allocated
//     lazily per layer and slot), so the selected keys need no device gather;
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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "engine.h"
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
// row; engine_q4exp writes the matching .wo file after the projection.
int qsa_dump_layer() {
    static const int l = si::env::i32("PF_QSA_ATTN_LAYER", 0);
    return l;
}

} // namespace

// Same context as the compute queue: the pinned staging and the device K/V the
// copies read belong to it, and a queue in another context cannot use them.
sycl::queue & engine::qsa_copy_queue(int dev) {
    if ((int)qsa_copy_q_.size() <= dev) {
        qsa_copy_q_.resize((size_t)dev + 1);
    }
    auto & p = qsa_copy_q_[(size_t)dev];
    if (!p) {
        sycl::queue & qd = dev_queue(dev);
        p = std::make_unique<sycl::queue>(qd.get_context(), qd.get_device(),
                                          sycl::property_list{sycl::property::queue::in_order()});
    }
    return *p;
}

// Per-layer pinned staging, grown to the largest chunk seen.  A pageable
// std::vector makes the copy synchronous on this backend, so the four D2H
// inputs and the sparse H2D output live in USM host memory.
void engine::qsa_stage_ensure(int a, int n_tok, bool need_q, bool need_out, int dev) {
    const hparams & hp = m.hp;
    if ((int)qsa_stage_.size() <= a) {
        qsa_stage_.resize((size_t)a + 1);
    }
    qsa_stage & st = qsa_stage_[(size_t)a];
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
    const size_t kvd = (size_t)hp.n_head_kv * hp.head_dim;
    const size_t n_embd = (size_t)hp.n_embd;
    if (st.cap < (size_t)n_tok) {
        size_t nc = st.cap ? st.cap : 1;
        while (nc < (size_t)n_tok) {
            nc <<= 1;
        }
        realloc_to(st.mixed, nc * n_embd);
        realloc_to(st.k, nc * kvd);
        realloc_to(st.v, nc * kvd);
        st.cap = nc;
    }
    if (need_q && st.cap_q < (size_t)n_tok) {
        size_t nc = st.cap_q ? st.cap_q : 1;
        while (nc < (size_t)n_tok) {
            nc <<= 1;
        }
        realloc_to(st.q, nc * (size_t)hp.n_head * 2 * (size_t)hp.head_dim);
        st.cap_q = nc;
    }
    if (need_out && st.cap_out < (size_t)n_tok) {
        size_t nc = st.cap_out ? st.cap_out : 1;
        while (nc < (size_t)n_tok) {
            nc <<= 1;
        }
        realloc_to(st.out, nc * (size_t)hp.n_head * (size_t)hp.head_dim);
        st.cap_out = nc;
    }
}

// Put the main queue's dependency on the deferred K/V copy back before the
// next attention layer's qkv GEMV overwrites KBUF/VBUF.  It is a device-side
// barrier (not a host wait), so the three GDN layers and FFNs in between still
// execute and the copy overlaps them.
void engine::qsa_copy_guard(int a, int dev) {
    if (!qsa_on_ || !qsa_copyq_ || qsa_pending_.empty()) {
        return;
    }
    const qsa_pending * last = nullptr;
    for (auto & p : qsa_pending_) {
        if (p.active && p.copyq && p.has_ev && p.dev == dev && p.a < a) {
            last = &p; // the copy queue is in-order: the latest event covers all
        }
    }
    if (!last) {
        return;
    }
    const sycl::event ev = last->ev;
    dev_queue(dev).submit([&](sycl::handler & h) {
        h.depends_on(ev);
        h.ext_oneapi_barrier();
    });
}

// The per-token QSA state update: the two indexer projections, their norm+RoPE,
// the raw-key ring/pool and the main K/V mirror.  It is split out of
// qsa_attention so a decode that is still dense-equivalent can run it at the
// end of the forward from host-staged copies (see qsa_flush).
void engine::qsa_apply_update(int a, const layer_t & L, int n_tok, const int * posv, const int * slotv,
                              const float * mh, const float * kh, const float * vh, float * qpos, bool with_q,
                              bool update_mirror) {
    const hparams & hp = m.hp;
    const int id = hp.idx_head_dim, inh = hp.idx_n_head;
    const int kvd = hp.n_head_kv * hp.head_dim;
    const int n_embd = hp.n_embd;
    static const bool qtime = si::env::flag("PF_QSA_TIME");
    double q_t0 = qtime ? qsa_now_ms() : 0.0;

    // Indexer projections (BF16 weights, host GEMV) for the whole batch.  The
    // BF16 dot is 39 MB/token of weights at a 2-byte element, so the ISA
    // variant is worth real time; a non-BF16 indexer falls back to the generic
    // arbitrary-K GEMV.  The q projection is 80% of it (512 of 640 rows) and is
    // read only by the selector, so a dense-equivalent token skips it: this
    // token's q is never scored, and the next token computes its own.
    std::vector<float> iq(with_q ? (size_t)n_tok * inh * id : 0);
    std::vector<float> ik((size_t)n_tok * id);
    if (L.idx.k_proj.type == 30) {
        if (with_q) {
            cpu_qsa_bf16_gemv(L.idx.q_proj.data, mh, n_embd, iq.data(), inh * id, n_tok, n_embd, inh * id);
        }
        cpu_qsa_bf16_gemv(L.idx.k_proj.data, mh, n_embd, ik.data(), id, n_tok, n_embd, id);
    } else {
        if (with_q) {
            cpu_q4_up_gemv(L.idx.q_proj.data, L.idx.q_proj.type, mh, n_embd, iq.data(), inh * id, n_tok, n_embd,
                           inh * id);
        }
        cpu_q4_up_gemv(L.idx.k_proj.data, L.idx.k_proj.type, mh, n_embd, ik.data(), id, n_tok, n_embd, id);
    }
    if (qtime) {
        qsa_gemv_ms_ += qsa_now_ms() - q_t0;
        q_t0 = qsa_now_ms();
    }

    for (int i = 0; i < n_tok; i++) {
        const int pos = posv[i];
        const int slot = slotv[i];
        float * qr = with_q ? iq.data() + (size_t)i * inh * id : nullptr;
        if (with_q) {
            for (int h = 0; h < inh; h++) {
                head_norm(qr + h * id, L.idx.q_norm, id, hp.rms_eps);
                head_rope(qr + h * id, hp.n_rot, hp.rope_base, pos);
            }
            if (qpos) {
                std::memcpy(qpos + (size_t)i * inh * id, qr, (size_t)inh * id * sizeof(float));
            }
        }
        // The cache stores the *raw* indexer key: the reference pools the raw
        // keys of a block and only then applies k_norm + RoPE to the mean.
        // Normalising/rotating each member first and pooling those gives a
        // different (and wrong) selector score.
        float * kr = ik.data() + (size_t)i * id;

        if (slot < 0 || slot >= (int)qsa_cache_[(size_t)a].size()) {
            continue;
        }
        auto & sp = qsa_cache_[(size_t)a][(size_t)slot];
        if (!sp) {
            sp = std::make_unique<qsa_slot_state>();
            if (qsa_f16mir_) {
                sp->k16.assign((size_t)qsa_max_seq_ * kvd, 0);
                sp->v16.assign((size_t)qsa_max_seq_ * kvd, 0);
            } else {
                sp->k.assign((size_t)qsa_max_seq_ * kvd, 0.f);
                sp->v.assign((size_t)qsa_max_seq_ * kvd, 0.f);
            }
            sp->pool.assign(((size_t)qsa_max_seq_ / 4 + 1) * id, 0.f);
            sp->ring.assign((size_t)4 * id, 0.f);
            sp->n_pool = 0;
        }
        sp->n_pool = std::max(sp->n_pool, pos / 4); // complete groups before pos
        float * ring = sp->ring.data() + (size_t)(pos % 4) * id;
        for (int d = 0; d < id; d++) {
            ring[d] = f16r(kr[d]);
        }
        if (pos % 4 == 3) {
            float * pl = sp->pool.data() + (size_t)(pos / 4) * id;
            for (int d = 0; d < id; d++) {
                pl[d] = 0.25f * (sp->ring[(size_t)0 * id + d] + sp->ring[(size_t)1 * id + d]
                                 + sp->ring[(size_t)2 * id + d] + sp->ring[(size_t)3 * id + d]);
            }
            head_norm(pl, L.idx.k_norm, id, hp.rms_eps);
            head_rope(pl, hp.n_rot, hp.rope_base, pos - 3);
            for (int d = 0; d < id; d++) {
                pl[d] = f16r(pl[d]);
            }
            sp->n_pool = std::max(sp->n_pool, pos / 4 + 1);
        }
        if (update_mirror) {
            const float * ks = kh + (size_t)i * kvd;
            const float * vs = vh + (size_t)i * kvd;
            // f16r() is exactly one rounding: the f16 store is the same value
            // the f32 mirror held, and the read converts it back bit-for-bit.
            if (qsa_f16mir_) {
                uint16_t * kd = sp->k16.data() + (size_t)pos * kvd;
                uint16_t * vd = sp->v16.data() + (size_t)pos * kvd;
                for (int d = 0; d < kvd; d++) {
                    kd[d] = ggml_float_to_half(ks[d]);
                    vd[d] = ggml_float_to_half(vs[d]);
                }
            } else {
                float * kd = sp->k.data() + (size_t)pos * kvd;
                float * vd = sp->v.data() + (size_t)pos * kvd;
                for (int d = 0; d < kvd; d++) {
                    kd[d] = f16r(ks[d]);
                    vd[d] = f16r(vs[d]);
                }
            }
        }
    }
    if (qtime) {
        qsa_upd_ms_ += qsa_now_ms() - q_t0;
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
    if (!qsa_on_) {
        return;
    }
    static const bool qdbg = std::getenv("PF_Q4_QSADBG") != nullptr;
    qsa_flush();
    int had = 0;
    for (auto & layers : qsa_cache_) {
        for (size_t i = 0; i < layers.size(); i++) {
            if ((slot < 0 || (int)i == slot) && layers[i]) {
                had++;
            }
        }
    }
    for (auto & p : qsa_pending_) {
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
        p.mixed = p.k = p.v = nullptr;
        p.pos.clear();
        p.slot.clear();
    }
    for (auto & layers : qsa_cache_) {
        for (size_t i = 0; i < layers.size(); i++) {
            if (slot < 0 || (int)i == slot) {
                layers[i].reset();
            }
        }
    }
    if (qdbg) {
        fprintf(stderr, "[qsa] reset slot=%d: cleared %d layer mirror(s)\n", slot, had);
    }
}

// Wait for the copies deferred by qsa_attention and apply the update of every
// pending layer.  Called at the end of a dense forward/chunk, after the LM
// head's kernels (or the chunk's remaining layers) have been enqueued: the
// event waits return once the copies are done, so the host work runs while the
// device is still busy.
double engine::qsa_flush() {
    if (!qsa_on_ || qsa_pending_.empty()) {
        return 0.0;
    }
    static const bool qtime = si::env::flag("PF_QSA_TIME");
    const double t0 = qtime ? qsa_now_ms() : 0.0;
    bool any = false;
    for (auto & p : qsa_pending_) {
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
        qsa_copy_ms_ += qsa_now_ms() - t0;
    }
    for (auto & p : qsa_pending_) {
        if (!p.active) {
            continue;
        }
        qsa_apply_update(p.a, *p.L, p.n_tok, p.pos.data(), p.slot.data(), p.mixed, p.k, p.v, nullptr, false, true);
        p.active = false;
    }
    return qtime ? qsa_now_ms() - t0 : 0.0;
}

// PF_QSA_DEVATTN scratch, allocated on first use.  The compact pool uses the
// paged layout so attn's kv_row_data() indexes it unchanged; the sequential
// table presents the selected rows as positions 0..nsel-1.
void engine::qsa_devattn_ensure(int dev) {
    const hparams & hp = m.hp;
    if ((int)qsa_dev_.size() <= dev) {
        qsa_dev_.resize((size_t)dev + 1);
    }
    qsa_dev_buf & d = qsa_dev_[(size_t)dev];
    if (d.dev == dev && d.cap_keys > 0) {
        return;
    }
    const int per = hp.idx_kpool > 1 ? hp.idx_kpool : 4;
    const int need_keys = hp.idx_top_k + per + 1; // selected pools' 4 keys + the incomplete tail
    const int cap = ((need_keys + kBlockSize - 1) / kBlockSize) * kBlockSize;
    const int blocks = cap / kBlockSize;
    const int kvd = hp.n_head_kv * hp.head_dim;
    sycl::queue & q = dev_queue(dev);
    d.ck = (sycl::half *)dev_alloc_on(dev, (size_t)cap * (size_t)kvd * sizeof(sycl::half));
    d.cv = (sycl::half *)dev_alloc_on(dev, (size_t)cap * (size_t)kvd * sizeof(sycl::half));
    d.dsel = (int32_t *)dev_alloc_on(dev, (size_t)cap * sizeof(int32_t));
    d.tab = (int32_t *)dev_alloc_on(dev, (size_t)blocks * sizeof(int32_t));
    d.htab = (int32_t *)sycl::malloc_host((size_t)blocks * sizeof(int32_t), q);
    if (!d.ck || !d.cv || !d.dsel || !d.tab || !d.htab) {
        throw std::runtime_error("qwen4exp: QSA Stage 2 scratch allocation failed");
    }
    for (int i = 0; i < blocks; i++) {
        d.htab[i] = i;
    }
    (void)q.memcpy(d.tab, d.htab, (size_t)blocks * sizeof(int32_t));
    d.cap_keys = cap;
    d.cap_blocks = blocks;
    d.dev = dev;
    if (si::env::flag("PF_Q4_QSADBG")) {
        fprintf(stderr, "[qsa] devattn scratch dev=%d keys=%d blocks=%d\n", dev, cap, blocks);
    }
}

void engine::qsa_devattn_aux(int a, int dev) {
    qsa_devattn_ensure(dev);
    if ((int)qsa_aux_.size() <= a) {
        qsa_aux_.resize((size_t)a + 1);
    }
    qsa_dev_aux & x = qsa_aux_[(size_t)a];
    if (x.hsel) {
        return;
    }
    qsa_dev_buf & d = qsa_dev_[(size_t)dev];
    sycl::queue & q = dev_queue(dev);
    x.hsel = (int32_t *)sycl::malloc_host((size_t)d.cap_keys * sizeof(int32_t), q);
    x.sinfo = (step_info *)dev_alloc_on(dev, sizeof(step_info));
    if (!x.hsel || !x.sinfo) {
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
    qsa_dev_buf & d = qsa_dev_[(size_t)dev];
    qsa_dev_aux & x = qsa_aux_[(size_t)a];
    if (nsel > d.cap_keys) {
        throw std::runtime_error("qwen4exp: QSA Stage 2 sel exceeds the compact-pool capacity");
    }
    const int nblk = (nsel + kBlockSize - 1) / kBlockSize;
    sycl::queue & q = dev_queue(dev);
    // The pinned staging is per layer, but it is reused by the next token; wait
    // for the (small) copy so the host write cannot race it.  The queue is
    // already drained by the mixed-copy wait, so this is the copy alone.
    std::memcpy(x.hsel, sel.data(), (size_t)nsel * sizeof(int32_t));
    sycl::event e = q.memcpy(d.dsel, x.hsel, (size_t)nsel * sizeof(int32_t));
    e.wait();
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

bool engine::qsa_attention(int a, const layer_t & L, int dev, int n_tok, const step_info * inf, const float * mixed,
                           float * qbuf, const float * kbuf, const float * vbuf, float * attn_out) {
    if (!qsa_on_ || !L.idx.ok() || a < 0 || a >= (int)qsa_cache_.size()) {
        return false;
    }
    const hparams & hp = m.hp;
    const int hd = hp.head_dim, nh = hp.n_head, nkv = hp.n_head_kv;
    const int id = hp.idx_head_dim, inh = hp.idx_n_head;
    const int kvd = nkv * hd;
    const int gqa = nh / nkv;
    const int n_pool_max = hp.idx_top_k / (hp.idx_kpool > 0 ? hp.idx_kpool : 4);
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
    auto from_host_async = [&](void * dst, const void * src, size_t bytes) {
        if (is_cpu(dev)) {
            std::memcpy(dst, src, bytes);
        } else {
            dev_queue(dev).memcpy(dst, src, bytes);
        }
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
    // The main K/V mirror is needed at every token (the indexer cache must be
    // warm when the context crosses the budget), but Q is read only by the
    // sparse attention, so its 49 KB/token/layer copy is skipped while the
    // selector is dense-equivalent.
    bool need = false;
    std::vector<int> posv((size_t)n_tok), slotv((size_t)n_tok);
    for (int i = 0; i < n_tok; i++) {
        const int row = inf->tpb == 1 ? i : i / inf->tpb;
        const int t = inf->tpb == 1 ? 0 : i % inf->tpb;
        posv[(size_t)i] = inf->pos[row] + t;
        slotv[(size_t)i] = inf->slot[row];
        const int slot = slotv[(size_t)i];
        if (slot >= 0 && slot < (int)qsa_cache_[(size_t)a].size() && posv[(size_t)i] + 1 > hp.idx_top_k) {
            need = true;
        }
    }

    // Dense-equivalent forwards (decode, and a prefill tail of up to kMaxB
    // tokens) enqueue the copies and let the end of the forward apply the
    // update.  A dense prefill chunk larger than kMaxB defers the same way, but
    // K/V go on the per-device copy queue so they overlap the following layers;
    // only the per-token update moves - the selector still has to see this
    // token, so the sparse path keeps the inline form.
    static const bool defer_on = SI_ENV_I32("PF_QSA_DEFER", 1) != 0;
    const bool defer_small = defer_on && !need && n_tok <= kMaxB;
    const bool defer_big = defer_on && qsa_copyq_ && !need && n_tok > kMaxB;
    const bool defer = (defer_small || defer_big) && a < (int)qsa_pending_.size();
    if (defer) {
        qsa_stage_ensure(a, n_tok, false, false, dev);
        qsa_stage & st = qsa_stage_[(size_t)a];
        qsa_pending & p = qsa_pending_[(size_t)a];
        p.L = &L;
        p.a = a;
        p.n_tok = n_tok;
        p.dev = dev;
        p.pos.assign(posv.begin(), posv.end());
        p.slot.assign(slotv.begin(), slotv.end());
        p.mixed = st.mixed;
        p.k = st.k;
        p.v = st.v;
        const size_t mb = (size_t)n_tok * (size_t)n_embd * sizeof(float);
        const size_t kb = (size_t)n_tok * (size_t)kvd * sizeof(float);
        if (is_cpu(dev)) {
            std::memcpy(p.mixed, mixed, mb);
            std::memcpy(p.k, kbuf, kb);
            std::memcpy(p.v, vbuf, kb);
            p.copyq = false;
            p.has_ev = false;
        } else if (defer_big) {
            // mixed stays on the main queue: its source b.mixed is rewritten by
            // the next layer's hc_mix, and the in-order queue orders that after
            // the copy.  The barrier marks the point K/V may start from, and
            // (because it is submitted after the mixed copy) its event also
            // covers that copy for qsa_flush.
            (void)to_host_async(p.mixed, mixed, mb);
            sycl::event mark = dev_queue(dev).submit([&](sycl::handler & h) { h.ext_oneapi_barrier(); });
            sycl::queue & cq = qsa_copy_queue(dev);
            // One explicit memory operation per command group.
            sycl::event ekm = cq.submit([&](sycl::handler & h) {
                h.depends_on(mark);
                h.memcpy(p.k, kbuf, kb);
            });
            p.ev = cq.submit([&](sycl::handler & h) {
                h.depends_on(ekm);
                h.memcpy(p.v, vbuf, kb);
            });
            p.copyq = true;
            p.has_ev = true;
        } else {
            (void)to_host_async(p.mixed, mixed, mb);
            (void)to_host_async(p.k, kbuf, kb);
            p.ev = to_host_async(p.v, vbuf, kb);
            p.copyq = false;
            p.has_ev = true;
        }
        p.active = true;
        if (qtime) {
            qsa_copy_ms_ += qsa_now_ms() - q_t0;
        }
        if (qdbg) {
            fprintf(stderr, "[qsa] deferred a=%d n_tok=%d copyq=%d\n", a, n_tok, (int)p.copyq);
        }
        return false;
    }

    // Stage 2: the sparse, single-token GPU f16 path attends to the selected
    // rows of the paged pool on the device.  The selector still runs here and
    // is unchanged; only the main Q/K/V D2H and the host attention go away.
    // Dense forwards, prefill chunks (n_tok > 1), CPU and non-f16 KV keep the
    // Stage 1 host path, including the mirror update.
    const bool f16kv = kv_k_dtype() == kv_dtype_t::f16 && kv_v_dtype() == kv_dtype_t::f16;
    const bool dev_path = qsa_devattn_ && need && n_tok == 1 && !is_cpu(dev) && f16kv && hd == 256;

    // The sparse path runs inline and reads the mirror: apply any update a
    // preceding dense chunk deferred before the selector sees the pool.
    qsa_flush();
    qsa_stage_ensure(a, n_tok, need && !dev_path, need, dev);
    qsa_stage & st = qsa_stage_[(size_t)a];

    const size_t mb = (size_t)n_tok * (size_t)n_embd * sizeof(float);
    const size_t qb = (size_t)n_tok * (size_t)nh * (size_t)stride * sizeof(float);
    const size_t kb = (size_t)n_tok * (size_t)kvd * sizeof(float);
    (void)to_host_async(st.mixed, mixed, mb);
    if (need && !dev_path) {
        (void)to_host_async(st.q, qbuf, qb);
    }
    if (!dev_path) {
        (void)to_host_async(st.k, kbuf, kb);
        (void)to_host_async(st.v, vbuf, kb);
    }
    if (!is_cpu(dev)) {
        dev_queue(dev).wait();
    }
    if (qtime) {
        qsa_copy_ms_ += qsa_now_ms() - q_t0;
        q_t0 = qsa_now_ms();
    }

    std::vector<float> qpos((size_t)n_tok * inh * id);
    qsa_apply_update(a, L, n_tok, posv.data(), slotv.data(), st.mixed, st.k, st.v, qpos.data(), need, !dev_path);

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
        if (a < (int)qsa_cache_.size() && !qsa_cache_[(size_t)a].empty() && qsa_cache_[(size_t)a][0]) {
            if (FILE * fp = fopen(kp, "wb")) {
                fwrite(qsa_cache_[(size_t)a][0]->pool.data(), sizeof(float), (size_t)id, fp);
                fclose(fp);
            }
        }
    }
    bool used_dev = false;
    if (!dev_path) {
        std::memset(st.out, 0, (size_t)n_tok * (size_t)nh * (size_t)hd * sizeof(float));
    }
    for (int i = 0; i < n_tok; i++) {
        const int pos = posv[(size_t)i];
        const int slot = slotv[(size_t)i];
        if (slot < 0 || slot >= (int)qsa_cache_[(size_t)a].size() || !qsa_cache_[(size_t)a][(size_t)slot]) {
            continue;
        }
        const double q_row0 = qtime ? qsa_now_ms() : 0.0;
        const qsa_slot_state & sp = *qsa_cache_[(size_t)a][(size_t)slot];
        const int n_vis = pos + 1;
        const int n_pool_vis = n_vis / 4;
        std::vector<int> sel;
        if (n_pool_vis <= n_pool_max) {
            sel.resize((size_t)n_vis);
            for (int p = 0; p < n_vis; p++) {
                sel[(size_t)p] = p;
            }
        } else {
            // The reference runs ggml_top_k, whose CPU implementation is
            // std::partial_sort over the score with `a > b`.  Reproduce it with
            // the same algorithm and comparator so equal (rectified-to-zero)
            // scores select the same pools; a priority queue with a lower-index
            // tie-break picks a different subset.
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
            std::partial_sort(idx.begin(), idx.begin() + n_top, idx.end(),
                              [&](int a, int b) { return scores[(size_t)a] > scores[(size_t)b]; });
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
            sel.reserve(pools.size() * 4 + 3);
            for (int p : pools) {
                for (int j = 0; j < 4; j++) {
                    sel.push_back(p * 4 + j);
                }
            }
            for (int p = n_pool_vis * 4; p <= pos; p++) {
                sel.push_back(p);
            }
        }
        const int nsel = (int)sel.size();
        if (qtime) {
            qsa_sel_ms_ += qsa_now_ms() - q_row0;
            qsa_rows_++;
            qsa_nsel_ += nsel;
        }
        if (qdbg) {
            fprintf(stderr, "[qsa] row=%d pos=%d slot=%d nsel=%d\n", i, pos, slot, nsel);
        }
        const int kvh_stride = kvd;
        const double q_att0 = qtime ? qsa_now_ms() : 0.0;
        if (dev_path) {
            qsa_devattn_run(a, dev, slot, pos, sel, qbuf, attn_out);
            used_dev = true;
        } else {
            // one job per query head, ~nsel*2*head_dim MAC each; the kernel
            // picks the scalar / AVX2 / AVX-512 form on the host ISA.
            par(nh, [&](int h) {
                const int kvh = h / gqa;
                const float * q = st.q + (size_t)i * nh * stride + (size_t)h * stride;
                float * o = st.out + (size_t)i * nh * hd + (size_t)h * hd;
                if (qsa_f16mir_) {
                    cpu_qsa_attn_head_h(q, sp.k16.data() + (size_t)kvh * hd, sp.v16.data() + (size_t)kvh * hd,
                                        kvh_stride, sel.data(), nsel, hd, hp.attn_scale, o);
                } else {
                    cpu_qsa_attn_head(q, sp.k.data() + (size_t)kvh * hd, sp.v.data() + (size_t)kvh * hd, kvh_stride,
                                      sel.data(), nsel, hd, hp.attn_scale, o);
                }
            });
        }
        if (qtime) {
            qsa_attn_ms_ += qsa_now_ms() - q_att0;
        }
    }
    if (qdbg) {
        fprintf(stderr, "[qsa] attention done, writing out\n");
    }
    if (const char * dp = si::env::str("PF_QSA_ATTN_DUMP")) {
        if (a == qsa_dump_layer()) {
            const int i = n_tok - 1;
            const int dpos = posv[(size_t)i];
            char path[1024];
            snprintf(path, sizeof(path), "%s.qsa.%d.%d.bin", dp, a, dpos);
            if (FILE * fp = fopen(path, "wb")) {
                if (used_dev) {
                    std::vector<float> hout((size_t)nh * (size_t)hd);
                    sycl::queue & q = dev_queue(dev);
                    q.memcpy(hout.data(), attn_out, hout.size() * sizeof(float));
                    q.wait();
                    fwrite(hout.data(), sizeof(float), hout.size(), fp);
                } else {
                    fwrite(st.out + (size_t)i * nh * hd, sizeof(float), (size_t)nh * hd, fp);
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
            std::memset(st.out, 0, (size_t)n_tok * (size_t)nh * (size_t)hd * sizeof(float));
        }
        from_host_async(attn_out, st.out, (size_t)n_tok * (size_t)nh * (size_t)hd * sizeof(float));
    }
    if (qdbg) {
        fprintf(stderr, "[qsa] out written\n");
    }
    return true;
}

} // namespace si
