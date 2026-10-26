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
// They are now enqueued together, and while the selector is dense-equivalent
// (pos < top_k, the common short-context decode) the whole update is deferred
// to the end of the forward and waited once - after the LM head has been
// enqueued, so the indexer projection and the mirror update run beside it.
// Prefill pays the update once per chunk; the sparse path (> top_k keys) still
// runs the selector and the attention inline, before the layer's `wo`.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "engine.h"
#include "common/env.h"
#include "backend/cpu/kernels/common.h"
#include "backend/cpu/cpu_types.h"

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

} // namespace

// The per-token QSA state update: the two indexer projections, their norm+RoPE,
// the raw-key ring/pool and the main K/V mirror.  It is split out of
// qsa_attention so a decode that is still dense-equivalent can run it at the
// end of the forward from host-staged copies (see qsa_flush).
void engine::qsa_apply_update(int a, const layer_t & L, int n_tok, const int * posv, const int * slotv,
                              const float * mh, const float * kh, const float * vh, float * qpos, bool with_q) {
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
            sp->k.assign((size_t)qsa_max_seq_ * kvd, 0.f);
            sp->v.assign((size_t)qsa_max_seq_ * kvd, 0.f);
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
        {
            const float * ks = kh + (size_t)i * kvd;
            const float * vs = vh + (size_t)i * kvd;
            float * kd = sp->k.data() + (size_t)pos * kvd;
            float * vd = sp->v.data() + (size_t)pos * kvd;
            for (int d = 0; d < kvd; d++) {
                kd[d] = f16r(ks[d]);
                vd[d] = f16r(vs[d]);
            }
        }
    }
    if (qtime) {
        qsa_upd_ms_ += qsa_now_ms() - q_t0;
    }
}

// Wait for the copies deferred by qsa_attention and apply the update of every
// pending layer.  Called at the end of a decode forward, after the LM head's
// kernels have been enqueued: the event waits return once the copies (which
// sit ahead of the head in the in-order queue) are done, so the host work runs
// while the head is still on the device.
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
        qsa_apply_update(p.a, *p.L, p.n_tok, p.pos.data(), p.slot.data(), p.mixed.data(), p.k.data(), p.v.data(),
                         nullptr, false);
        p.active = false;
    }
    return qtime ? qsa_now_ms() - t0 : 0.0;
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
    auto from_host = [&](void * dst, const void * src, size_t bytes) {
        if (is_cpu(dev)) {
            std::memcpy(dst, src, bytes);
        } else {
            dev_queue(dev).memcpy(dst, src, bytes).wait();
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
    // update.  Only the per-token update moves: the selector still has to see
    // this token, so the sparse path and prefill chunks larger than the
    // staging window keep the inline form.
    static const bool defer_on = SI_ENV_I32("PF_QSA_DEFER", 1) != 0;
    const bool defer = defer_on && !need && n_tok <= kMaxB && a < (int)qsa_pending_.size();
    if (defer) {
        qsa_pending & p = qsa_pending_[(size_t)a];
        p.L = &L;
        p.a = a;
        p.n_tok = n_tok;
        p.pos.assign(posv.begin(), posv.end());
        p.slot.assign(slotv.begin(), slotv.end());
        p.mixed.resize((size_t)n_tok * n_embd);
        p.k.resize((size_t)n_tok * kvd);
        p.v.resize((size_t)n_tok * kvd);
        sycl::event e = to_host_async(p.mixed.data(), mixed, p.mixed.size() * sizeof(float));
        (void)e;
        to_host_async(p.k.data(), kbuf, p.k.size() * sizeof(float));
        p.ev = to_host_async(p.v.data(), vbuf, p.v.size() * sizeof(float));
        p.has_ev = !is_cpu(dev);
        p.active = true;
        if (qtime) {
            qsa_copy_ms_ += qsa_now_ms() - q_t0;
        }
        if (qdbg) {
            fprintf(stderr, "[qsa] deferred a=%d n_tok=%d\n", a, n_tok);
        }
        return false;
    }

    std::vector<float> mh((size_t)n_tok * n_embd);
    std::vector<float> qh(need ? (size_t)n_tok * nh * stride : 0);
    std::vector<float> kh((size_t)n_tok * kvd);
    std::vector<float> vh((size_t)n_tok * kvd);
    to_host_async(mh.data(), mixed, mh.size() * sizeof(float));
    if (need) {
        to_host_async(qh.data(), qbuf, qh.size() * sizeof(float));
    }
    to_host_async(kh.data(), kbuf, kh.size() * sizeof(float));
    to_host_async(vh.data(), vbuf, vh.size() * sizeof(float));
    if (!is_cpu(dev)) {
        dev_queue(dev).wait();
    }
    if (qtime) {
        qsa_copy_ms_ += qsa_now_ms() - q_t0;
        q_t0 = qsa_now_ms();
    }

    std::vector<float> qpos((size_t)n_tok * inh * id);
    qsa_apply_update(a, L, n_tok, posv.data(), slotv.data(), mh.data(), kh.data(), vh.data(), qpos.data(), need);

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
                fwrite(mh.data() + (size_t)(n_tok - 1) * n_embd, sizeof(float), (size_t)n_embd, fp);
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
    std::vector<float> out((size_t)n_tok * nh * hd, 0.f);
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
        // one job per query head, ~nsel*2*head_dim MAC each; the kernel picks
        // the scalar / AVX2 / AVX-512 form on the host ISA.
        par(nh, [&](int h) {
            const int kvh = h / gqa;
            const float * q = qh.data() + (size_t)i * nh * stride + (size_t)h * stride;
            cpu_qsa_attn_head(q, sp.k.data() + (size_t)kvh * hd, sp.v.data() + (size_t)kvh * hd, kvh_stride,
                              sel.data(), nsel, hd, hp.attn_scale, out.data() + (size_t)i * nh * hd + (size_t)h * hd);
        });
        if (qtime) {
            qsa_attn_ms_ += qsa_now_ms() - q_att0;
        }
    }
    if (qdbg) {
        fprintf(stderr, "[qsa] attention done, writing out\n");
    }
    from_host(attn_out, out.data(), out.size() * sizeof(float));
    if (qdbg) {
        fprintf(stderr, "[qsa] out written\n");
    }
    return true;
}

} // namespace si
