// qwen4exp's arch_ops hooks (see arch/arch_ops.h).  The forward itself stays in
// arch/qwen4exp/forward.cpp and qsa.cpp; these are the entry points the generic
// engine calls from its constructor/destructor and from the forward and
// prefix-cache dispatch sites.
#include "engine.h"
#include "arch/qwen4exp/hparams.h"
#include "arch/registry_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

void arch_ops_registry::q4_setup(engine & e) {
    e.q4_alloc();
}

void arch_ops_registry::q4_prefault(engine & e) {
    e.q4_prefault();
}

void arch_ops_registry::q4_slot_zero(engine & e, int slot) {
    e.q4_zero_state(slot);
}

void arch_ops_registry::q4_hist_report(engine & e) {
    if (!e.q4x_.q4_hist_pre_.empty() || !e.q4x_.q4_hist_dec_.empty()) {
        e.q4_hist_report();
    }
}

// ---- constructor config / teardown / per-slot reset ------------------------

void arch_ops_registry::q4_configure(engine & e, const engine_config & cfg) {
    // The PLE engram budget is a model-profile default (qwen4exp 4096 MB); an
    // explicit --engram-ram-mb / engine_config value still wins.  Resolved after
    // the load, because the profile is selected by the GGUF's architecture.
    e.q4x_.engram_ram_mb_ =
        cfg.engram_ram_mb == engine_config::kEngramRamAuto ? e.mp_->policy.engram_ram_mb_default : cfg.engram_ram_mb;
    // Tail-skip precedence: an explicit --q4-tail-skip / engine_config value
    // wins, else PF_Q4_TAIL_SKIP, else 0 (the exact expert set).  Resolved once
    // here so the forward has no second default at its use site.
    e.q4x_.q4_tail_skip_ = cfg.q4_tail_skip >= 0 ? cfg.q4_tail_skip : si::env::i32("PF_Q4_TAIL_SKIP", 0);
}

void arch_ops_registry::q4_destroy(engine & e) {
    // The routing diagnostic used to run before this block (the destructor's
    // generic hist_report call); keep that order here so the print sequence is
    // unchanged.  The row sets hist_report to null so it is not printed twice.
    q4_hist_report(e);
    if (e.q4x_.engram_hits_ + e.q4x_.engram_misses_ > 0) {
        fprintf(stderr, "[q4] engram cache: %zu hits / %zu misses (%.1f%% hit rate)\n", e.q4x_.engram_hits_,
                e.q4x_.engram_misses_,
                100.0 * (double)e.q4x_.engram_hits_ / (double)(e.q4x_.engram_hits_ + e.q4x_.engram_misses_));
    }
    for (void * p : e.q4x_.q4_owned_) {
        std::free(p);
    }
    e.q4x_.q4_owned_.clear();
    for (void * p : e.q4x_.q4_host_usm_) {
        sycl::free(p, e.q);
    }
    e.q4x_.q4_host_usm_.clear();
    for (auto & pr : e.q4x_.q4_pin_src_) {
        sycl::free(pr.first, e.dev_queue(pr.second));
    }
    e.q4x_.q4_pin_src_.clear();
    for (auto & st : e.q4x_.qsa.stage) {
        sycl::queue & qd = e.dev_queue(st.dev < 0 ? 0 : st.dev);
        auto fq = [&](float * p) {
            if (p) {
                sycl::free(p, qd);
            }
        };
        fq(st.mixed);
        fq(st.iq);
        fq(st.ik);
        for (uint8_t * p : {st.sbk, st.sbv, st.sbsk, st.sbsv}) {
            if (p) {
                sycl::free(p, qd);
            }
        }
    }
    e.q4x_.qsa.stage.clear();
    for (auto & hs : e.q4x_.qsa.host_stage) {
        sycl::queue & qd = e.dev_queue(hs.dev < 0 ? 0 : hs.dev);
        if (hs.out_pending) {
            hs.out_ev.wait();
            hs.out_pending = false;
        }
        if (hs.q) {
            sycl::free(hs.q, qd);
        }
        if (hs.out) {
            sycl::free(hs.out, qd);
        }
    }
    e.q4x_.qsa.host_stage.clear();
    for (auto & b : e.q4x_.qsa.idx_dev_buf) {
        if (b.q) {
            sycl::free(b.q, e.dev_queue(b.dev < 0 ? 0 : b.dev));
        }
        if (b.k) {
            sycl::free(b.k, e.dev_queue(b.dev < 0 ? 0 : b.dev));
        }
    }
    e.q4x_.qsa.idx_dev_buf.clear();
    e.qsa_idx_slot_release_all();
    for (auto & h : e.q4x_.qsa.handoff) {
        if (h.stage) {
            sycl::free(h.stage, e.dev_queue(h.dev < 0 ? 0 : h.dev));
        }
    }
    e.q4x_.qsa.handoff.clear();
    for (auto & d : e.q4x_.qsa.dev) {
        sycl::queue & qd = e.dev_queue(d.dev < 0 ? 0 : d.dev);
        if (d.ck) {
            sycl::free(d.ck, qd);
        }
        if (d.cv) {
            sycl::free(d.cv, qd);
        }
        if (d.dsel) {
            sycl::free(d.dsel, qd);
        }
        if (d.dcounts) {
            sycl::free(d.dcounts, qd);
        }
        if (d.tab) {
            sycl::free(d.tab, qd);
        }
        if (d.htab) {
            sycl::free(d.htab, qd);
        }
        if (d.scores) {
            sycl::free(d.scores, qd);
        }
        if (d.dpools) {
            sycl::free(d.dpools, qd);
        }
    }
    e.q4x_.qsa.dev.clear();
    for (auto & x : e.q4x_.qsa.aux) {
        sycl::queue & qd = e.dev_queue(x.dev < 0 ? 0 : x.dev);
        if (x.hsel) {
            sycl::free(x.hsel, qd);
        }
        if (x.hcnt) {
            sycl::free(x.hcnt, qd);
        }
        if (x.sinfo) {
            sycl::free(x.sinfo, qd);
        }
    }
    e.q4x_.qsa.aux.clear();
}

void arch_ops_registry::q4_reset_extra(engine & e) {
    if (e.q4x_.q4_ple_hist_) {
        const int hist_rows = (q4_hp(e.m.hp).ple_conv - 1) * q4_hp(e.m.hp).ple_ngram;
        const int hc_dim = q4_hp(e.m.hp).hc_mult * e.m.hp.n_embd;
        std::memset(e.q4x_.q4_ple_hist_, 0, (size_t)e.state_slots_ * hist_rows * hc_dim * sizeof(float));
        std::memset(e.q4x_.q4_ple_nhist_, 0, (size_t)e.state_slots_ * sizeof(int32_t));
        for (int i = 0; i < e.state_slots_ * 8; i++) {
            e.q4x_.q4_ple_prev_[i] = q4_hp(e.m.hp).ple_eos;
        }
    }
}

// ---- forward dispatch ------------------------------------------------------

void arch_ops_registry::q4_prefill_text(engine & e, const std::vector<int> & toks, int slot, int n, int start) {
    e.q4_prefill_text(toks, slot, n, start);
}

void arch_ops_registry::q4_prefill_mm(engine & e, const mm_prompt & p, int slot) {
    e.q4_prefill_mm(p, slot);
}

void arch_ops_registry::q4_decode_batch(engine & e, const int32_t * tokens, const int32_t * poss,
                                        const int32_t * slots, int n_rows) {
    e.prefill_flush();
    e.pf_info_ = nullptr;
    e.d_info->pc_active = 0;
    e.d_info->n_rows = n_rows;
    e.d_info->tpb = 1;
    e.d_info->n_real = 1;
    // generate_mm writes the single decode row's M-RoPE position just before
    // this call (an image's RoPE position is not the token index).  Keep it
    // for that loop; every text-only caller resets step_info or prefills
    // through q4_prefill_text, which clears mrope_on.
    e.d_info->mrope_on = e.mm_decode_rope_ ? 1 : 0;
    e.d_info->img_embd = nullptr;
    for (int r = 0; r < kMaxB; r++) {
        e.d_info->pos[r] = (r < n_rows) ? poss[r] : 0;
        e.d_info->slot[r] = (r < n_rows) ? slots[r] : (r % kMaxB);
        e.d_info->active[r] = (r < n_rows) ? 1 : 0;
        e.d_info->tokens[r] = (r < n_rows) ? tokens[r] : 0;
        e.d_info->n_real_row[r] = 1;
    }
    e.q4_forward(0, n_rows, true);
}

std::vector<float> arch_ops_registry::q4_run_head(engine & e) {
    e.q.memcpy(e.h_logits, e.d_logits, (size_t)e.m.hp.n_vocab * 4).wait();
    if (const char * dp = si::env::str("PF_DUMP_Q4LOG")) {
        if (FILE * fp = fopen(dp, "wb")) {
            fwrite(e.h_logits, sizeof(float), (size_t)e.m.hp.n_vocab, fp);
            fclose(fp);
        }
        int top[5] = {0};
        for (int i = 0; i < e.m.hp.n_vocab; i++) {
            for (int j = 0; j < 5; j++) {
                if (e.h_logits[i] > e.h_logits[top[j]] || (j == 0 && i == 0)) {
                    for (int k = 4; k > j; k--) {
                        top[k] = top[k - 1];
                    }
                    top[j] = i;
                    break;
                }
            }
        }
        fprintf(stderr, "[q4] top5:");
        for (int j = 0; j < 5; j++) {
            fprintf(stderr, " %d(%.4f)", top[j], e.h_logits[top[j]]);
        }
        fprintf(stderr, "\n");
    }
    return std::vector<float>(e.h_logits, e.h_logits + e.m.hp.n_vocab);
}

// ---- prefix-cache checkpoint extras ---------------------------------------

size_t arch_ops_registry::q4_pc_extra_floats(engine & e, int depth_blocks) {
    return e.q4_pc_extra_floats(depth_blocks);
}

void arch_ops_registry::q4_pc_serialize(engine & e, float * out, int st, int depth) {
    const hparams & hp = e.m.hp;
    const qwen4exp_hparams & qh = q4_hp(hp);
    float * w = out;
    if (e.q4x_.q4_ple_hist_ && st >= 0 && (size_t)st < e.q4x_.q4_pc_ple_.size()) {
        const int hist_rows = (qh.ple_conv - 1) * qh.ple_ngram;
        const int hc_dim = qh.hc_mult * hp.n_embd;
        const qwen4exp_state::q4_pc_ple_state & p = e.q4x_.q4_pc_ple_[(size_t)st];
        if (p.valid && p.hist.size() == (size_t)hist_rows * (size_t)hc_dim) {
            std::memcpy(w, p.hist.data(), p.hist.size() * sizeof(float));
        } else {
            std::memset(w, 0, (size_t)hist_rows * (size_t)hc_dim * sizeof(float));
        }
        w += (size_t)hist_rows * (size_t)hc_dim;
        for (int j = 0; j < 8; j++) {
            w[j] = (size_t)j < p.prev.size() ? (float)p.prev[(size_t)j] : (float)qh.ple_eos;
        }
        w += 8;
        *w++ = (float)(p.valid ? p.nhist : 0);
    }
    if (e.q4x_.qsa.on && depth > 0) {
        const int id = qh.idx_head_dim;
        const int kp = qh.idx_kpool > 1 ? qh.idx_kpool : 4;
        const size_t np = (size_t)(depth * kBlockSize) / (size_t)kp;
        for (int a = 0; a < e.attn_layers(); a++) {
            const qwen4exp_state::q4_pc_qsa_state * q =
                st >= 0 && (size_t)st < e.q4x_.q4_pc_qsa_.size() ? &e.q4x_.q4_pc_qsa_[(size_t)st] : nullptr;
            if (q && a < (int)q->pool.size() && q->pool[(size_t)a].size() >= np * (size_t)id) {
                std::memcpy(w, q->pool[(size_t)a].data(), np * (size_t)id * sizeof(float));
            } else {
                std::memset(w, 0, np * (size_t)id * sizeof(float));
            }
            w += np * (size_t)id;
            if (q && a < (int)q->ring.size() && q->ring[(size_t)a].size() >= (size_t)kp * (size_t)id) {
                std::memcpy(w, q->ring[(size_t)a].data(), (size_t)kp * (size_t)id * sizeof(float));
            } else {
                std::memset(w, 0, (size_t)kp * (size_t)id * sizeof(float));
            }
            w += (size_t)kp * (size_t)id;
        }
    }
}

void arch_ops_registry::q4_pc_deserialize(engine & e, const float * in, int st, int depth_blocks) {
    if ((size_t)st >= e.q4x_.q4_pc_ple_.size()) {
        return;
    }
    const hparams & hp = e.m.hp;
    const qwen4exp_hparams & qh = q4_hp(hp);
    const float * r = in;
    if (e.q4x_.q4_ple_hist_) {
        const int hist_rows = (qh.ple_conv - 1) * qh.ple_ngram;
        const int hc_dim = qh.hc_mult * hp.n_embd;
        qwen4exp_state::q4_pc_ple_state & p = e.q4x_.q4_pc_ple_[(size_t)st];
        p.hist.assign(r, r + (size_t)hist_rows * (size_t)hc_dim);
        r += (size_t)hist_rows * (size_t)hc_dim;
        p.prev.resize(8);
        for (int j = 0; j < 8; j++) {
            p.prev[(size_t)j] = (int32_t)r[j];
        }
        r += 8;
        p.nhist = (int32_t)(*r++);
        p.valid = true;
    }
    if (e.q4x_.qsa.on && depth_blocks > 0) {
        const int id = qh.idx_head_dim;
        const int kp = qh.idx_kpool > 1 ? qh.idx_kpool : 4;
        const size_t np = (size_t)(depth_blocks * kBlockSize) / (size_t)kp;
        qwen4exp_state::q4_pc_qsa_state & q = e.q4x_.q4_pc_qsa_[(size_t)st];
        q.pool.resize((size_t)e.attn_layers());
        q.ring.resize((size_t)e.attn_layers());
        for (int a = 0; a < e.attn_layers(); a++) {
            q.pool[(size_t)a].assign(r, r + np * (size_t)id);
            r += np * (size_t)id;
            q.ring[(size_t)a].assign(r, r + (size_t)kp * (size_t)id);
            r += (size_t)kp * (size_t)id;
        }
        q.valid = true;
    }
}

void arch_ops_registry::q4_pc_restore(engine & e, int slot, int st, int depth_blocks) {
    e.q4_pc_restore(slot, st, depth_blocks);
}

void arch_ops_registry::q4_pc_clear(engine & e, int st) {
    e.q4_pc_clear(st);
}

void arch_ops_registry::q4_pc_resume_prefixes(engine & e, std::vector<int> & prefixes, int L) {
    prefixes.push_back(0);
    // qwen4exp chunks its own prefill at a fixed size (kMaxB*kMaxT,
    // or PF_Q4_PF_CHUNK when the expert bank is active).  Resuming at
    // one of those boundaries replays the exact suffix batching the
    // cold run used; resuming at an arbitrary 32-token boundary would
    // prefill the remainder in a differently sized batch and change M,
    // which moves the activation quantization and can flip a near tie.
    int ch = kMaxB * kMaxT;
    if (e.q4x_.q4_hot_) {
        if (const char * ev = si::env::str("PF_Q4_PF_CHUNK")) {
            ch = std::max(kMaxT, atoi(ev));
        }
    }
    ch = std::max(kMaxT, (ch / kMaxT) * kMaxT);
    // q4_prefill_common caps its chunk at kMaxB*kMaxT (R), so an
    // oversized PF_Q4_PF_CHUNK still runs in 1024-token chunks.
    ch = std::min<int>(ch, kMaxB * kMaxT);
    int ppos = ch;
    while (ppos < L) {
        prefixes.push_back(ppos);
        ppos += ch;
    }
    prefixes.push_back(L);
}

bool arch_ops_registry::q4_pc_block_aligned_capture(engine & e) {
    (void)e;
    // qwen4exp's own chunker only resumes exactly at a cold chunk
    // boundary, so it keeps the old "the batch must end on a block
    // boundary" rule; its final partial chunk captures nothing.
    return true;
}

} // namespace si
