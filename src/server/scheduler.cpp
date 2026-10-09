#include "scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <ratio>
#include <string>
#include <utility>
#include <vector>

#include "chat_util.h"
#include "kernels.h"
#include "sampler.h"
#include "common/env.h"

namespace si {

static bool srv_time() {
    static const bool t = si::env::flag("PF_SRV_TIME");
    return t;
}

static double dms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void scheduler::start() {
    th = std::thread([this] { loop(); });
}

void scheduler::shutdown() {
    {
        std::lock_guard<std::mutex> lk(m);
        stopping = true;
    }
    cv.notify_all();
    if (th.joinable()) {
        th.join();
    }
}

std::shared_ptr<sequence> scheduler::submit(std::vector<int> prompt, const gen_params & gp,
                                            std::vector<std::string> stops) {
    auto s = std::make_shared<sequence>();
    {
        std::lock_guard<std::mutex> lk(m);
        s->id = next_id++;
        s->prompt = std::move(prompt);
        s->gp = gp;
        s->stops = std::move(stops);
        s->prompt_tokens = (int)s->prompt.size();
        s->ss.seed(gp.seed ? gp.seed : (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count());
        if (srv_time()) {
            s->t_submit = std::chrono::steady_clock::now();
        }
        waiting.push_back(s);
    }
    cv.notify_all();
    return s;
}

bool scheduler::admit(std::shared_ptr<sequence> & s) {
    // the client disconnected before this sequence was admitted: retire it
    // instead of giving it a slot (and a prefix-cache admit) it will not use
    if (s->cancelled) {
        retire(s, "stop");
        return true;
    }
    // A direct scheduler caller may bypass HTTP validation.  Without this an
    // empty sequence is admitted but neither the prefill nor decode loop can
    // make progress, leaving its worker blocked in pop_token indefinitely.
    if (s->prompt.empty()) {
        retire(s, "stop");
        return true;
    }
    // a prompt that does not fit the configured context cannot be prefilled
    // (its block table row is only max_seq/kBlockSize entries long): retire it
    // cleanly instead of writing KV through table entries that do not exist
    if ((int)s->prompt.size() > e.max_seq) {
        retire(s, "length");
        return true;
    }
    // find a free state slot
    // max_slots bounds the engine's recurrent state, so it bounds the slots we may
    // hand out (engine_config::max_slots; kMaxB by default).
    const int n_slots = e.max_slots;
    bool used[kMaxB] = {false};
    for (auto & a : active) {
        used[a->slot] = true;
    }
    int slot = -1;
    for (int i = 0; i < n_slots; i++) {
        if (!used[i]) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        return false;
    }
    // prefix cache: attach cached KV blocks and restore the recurrent state of
    // the matched prefix; `matched` token positions are then already done
    std::vector<int> blocks;
    const int matched = e.pc_admit(slot, s->prompt, blocks);
    if (matched <= 0) {
        e.zero_slot(slot);
    }
    s->reused = matched;
    // allocate blocks for the first prefill chunk
    const int first_end = std::min<int>(matched + kMaxT, (int)s->prompt.size());
    const int need = (std::max(first_end, 1) + kBlockSize - 1) / kBlockSize;
    for (int i = (int)blocks.size(); i < need; i++) {
        int b = e.alloc_block();
        if (b < 0) {
            e.pc_retire(slot, blocks);
            return false;
        }
        blocks.push_back(b);
    }
    s->slot = slot;
    s->prompt_pos = matched;
    s->blocks = std::move(blocks);
    e.set_table(slot, s->blocks);
    s->admitted = true;
    return true;
}

void scheduler::retire(std::shared_ptr<sequence> & s, const char * reason) {
    s->finish_reason = reason;
    // a pending pipelined prefill chunk (device 1 not yet run) must complete
    // before its blocks are freed
    e.prefill_flush();
    e.pc_retire(s->slot, s->blocks);
    s->blocks.clear();
    {
        std::lock_guard<std::mutex> lk(s->m);
        s->finished = true;
    }
    s->cv.notify_all();
}

static bool dbg() {
    static bool d = si::env::flag("SCHED_DEBUG");
    return d;
}

// Sample one token from `logits`, advance the sequence's state, and hand the
// caller the UTF-8 piece (empty for EOS, or while a token's bytes are still
// incomplete).  Shared by the two places a token is produced - the first one out
// of the head prefill, and each row of a decode batch - which previously carried
// two copies of the same sequence, including the two easy-to-diverge details:
// the control token is never streamed, and want_logprobs decides whether the
// piece is sent even when the UTF-8 buffer has nothing yet.
//
// `done` and `eos` are separate on purpose: `eos` says the sampled token was a
// control token, `done` says the sequence must now be retired.  Collapsing them
// is wrong, because the decode path's reason string tests `&& !eos` - a run that
// ends by reaching max_tokens must report "length", and reusing `done` for `eos`
// turns that into "stop".
struct sample_outcome {
    bool eos = false;  // the sampled token was EOS/EOT (never streamed)
    bool done = false; // eos, or max_tokens reached
};

static sample_outcome sample_and_emit(engine & e, std::shared_ptr<sequence> & s, utf8_stream_buffer & ub,
                                      const float * logits, int n_vocab) {
    const bool want_lp = s->gp.wants_logprobs();
    sample_logprobs lp;
    const int tok = sample_token(logits, n_vocab, s->gp, s->recent, s->ss, want_lp ? &lp : nullptr);
    if (dbg()) {
        fprintf(stderr, "[sched]   sampled seq=%d tok=%d\n", s->id, tok);
    }
    s->recent.push_back(tok);
    s->n_generated++;
    // don't stream the EOS token itself: it is a control token, not content
    const bool eos = !s->gp.ignore_eos && (tok == e.tk.eos_id || tok == e.tk.eot_id);
    if (!eos) {
        std::string piece = ub.push(e.tk.token_piece(tok));
        if (want_lp) {
            sequence::token_out t;
            t.text = std::move(piece);
            t.id = tok;
            t.logprob = lp.logprob;
            t.top = std::move(lp.top);
            s->push_token(std::move(t));
        } else if (!piece.empty()) {
            s->push(piece);
        }
    }
    sample_outcome o;
    o.eos = eos;
    o.done = eos || s->n_generated >= s->gp.max_tokens;
    return o;
}

void scheduler::loop() {
    std::vector<int32_t> toks(kMaxB, 0), poss(kMaxB, 0), slots(kMaxB, 0);
    std::vector<float> logits(e.m.hp.n_vocab);
    std::vector<utf8_stream_buffer> ubs(kMaxB);

    while (true) {
        {
            std::lock_guard<std::mutex> lk(m);
            if (stopping && waiting.empty() && active.empty()) {
                break;
            }
        }
        bool did_work = false;

        // ---- admit waiting sequences ----
        {
            std::lock_guard<std::mutex> lk(e.mtx);
            // The sequence mutex is RELEASED across the engine calls below (a
            // ~300 ms prefill / ~65 ms decode).  Holding it through them and
            // immediately re-locking it in this loop starves
            // scheduler::submit(): requests were only admitted once the
            // previous generation had finished, so nothing ever batched.
            // e.mtx still serialises the engine itself.
            std::unique_lock<std::mutex> lk2(m);
            while (!waiting.empty()) {
                auto s = waiting.front();
                const auto t_a0 = std::chrono::steady_clock::now();
                if (!admit(s)) {
                    break;
                }
                if (srv_time()) {
                    s->admit_ms = dms(t_a0, std::chrono::steady_clock::now());
                }
                waiting.erase(waiting.begin());
                active.push_back(s);
            }
            // ---- one chunked prefill step (round robin) ----
            for (auto & s : active) {
                if (s->finished) {
                    continue;
                }
                // client gone: hand the slot and blocks back now rather than
                // after the rest of max_tokens (retire is the only path that
                // runs pc_retire)
                if (s->cancelled) {
                    retire(s, "stop");
                    continue;
                }
                if (s->prompt_pos < (int)s->prompt.size()) {
                    const int rem = (int)s->prompt.size() - s->prompt_pos;
                    // chunk-batched prefill: one forward covers the whole batch
                    // (weights read ~once instead of once per 32-token chunk);
                    // pick the largest size that fits the remaining prompt and
                    // the block pool.  The tail used to fall back to mode 1
                    // because mode-2 was only trusted for >= 2*kMaxT; now that
                    // the fused-GDN mode-2 bug is fixed any M is correct, and
                    // mode-2 handles the tail too (a mode-1 chunk costs ~0.3 ms
                    // of handoff/sync per 32 tokens, so a 41-token tail was
                    // ~0.7 s of a 553-token prefill).
                    int n = 0;
                    bool batch_pf = false;
                    for (int cand = e.batched_prefill_fit(rem); cand >= kMaxT; cand -= kMaxT) {
                        const int need = (s->prompt_pos + cand + kBlockSize - 1) / kBlockSize;
                        bool ok = true;
                        while ((int)s->blocks.size() < need) {
                            int b = e.alloc_block();
                            if (b < 0) {
                                ok = false;
                                break;
                            }
                            s->blocks.push_back(b);
                            e.set_table(s->slot, s->blocks);
                        }
                        if (ok) {
                            n = cand;
                            batch_pf = true;
                            break;
                        }
                    }
                    if (!batch_pf) {
                        n = std::min<int>(kMaxT, rem);
                        // ensure blocks for this chunk
                        const int end = s->prompt_pos + n;
                        const int need = (end + kBlockSize - 1) / kBlockSize;
                        while ((int)s->blocks.size() < need) {
                            int b = e.alloc_block();
                            if (b < 0) {
                                break;
                            }
                            s->blocks.push_back(b);
                            e.set_table(s->slot, s->blocks);
                        }
                    }
                    // the pool (or another sequence) may have starved this chunk:
                    // prefill must not write through unallocated block-table slots
                    if ((int)s->blocks.size() * kBlockSize < s->prompt_pos + n) {
                        retire(s, "length");
                        break;
                    }
                    // only the final prompt chunk needs the LM head
                    const bool last_chunk = (s->prompt_pos + n >= (int)s->prompt.size());
                    static const bool tdbg = srv_time();
                    const auto t_pf0 = std::chrono::steady_clock::now();
                    if (tdbg && s->n_chunks == 0) {
                        s->wait_ms = dms(s->t_submit, t_pf0);
                    }
                    if (batch_pf) {
                        lk2.unlock();
                        e.prefill_batch(s->prompt, s->prompt_pos, n, s->slot, s->prompt_pos);
                        lk2.lock();
                    } else {
                        lk2.unlock();
                        e.prefill_chunk(s->prompt, s->prompt_pos, n, s->slot, last_chunk);
                        lk2.lock();
                    }
                    const auto t_pf1 = std::chrono::steady_clock::now();
                    if (tdbg) {
                        s->pf_ms += dms(t_pf0, t_pf1);
                        // Per-chunk prefill time: a single long prompt then yields
                        // the marginal prefill throughput at every depth (the
                        // chunk at pos P is pp<n>@P).  Host-only diagnostic.
                        fprintf(stderr, "[srv] chunk pos=%d n=%d ms=%.2f\n", s->prompt_pos, n,
                                dms(t_pf0, t_pf1));
                    }
                    s->n_chunks++;
                    if (dbg()) {
                        fprintf(stderr, "[sched] prefill seq=%d slot=%d chunk pos=%d n=%d (prompt %zu)\n", s->id,
                                s->slot, s->prompt_pos, n, s->prompt.size());
                    }
                    s->prompt_pos += n;
                    // prefix cache: turn the blocks just prefilled into cache
                    // nodes (and capture a state checkpoint on block boundaries)
                    e.pc_commit(s->slot, s->prompt, s->blocks, s->prompt_pos);
                    did_work = true;
                    // if this completes the prompt, fetch the first token's logits
                    if (s->prompt_pos >= (int)s->prompt.size()) {
                        // the prefill graph produced logits into d_logits row 0;
                        // the pipelined multi-device prefill still owes the last
                        // chunk's device-1 phase (and the head)
                        e.prefill_flush();
                        const auto t_fl0 = std::chrono::steady_clock::now();
                        e.fetch_logits(0, logits.data());
                        s->recent = s->prompt;
                        const auto t_sm0 = std::chrono::steady_clock::now();
                        if (tdbg) {
                            const auto t_now = std::chrono::steady_clock::now();
                            auto ms = [](auto a, auto b) {
                                return std::chrono::duration<double, std::milli>(b - a).count();
                            };
                            // `prefill` is the last chunk's forward; `prefill_total`
                            // sums all chunk forwards (the first token cannot be
                            // sampled before every prompt token has been prefilled)
                            fprintf(stderr,
                                    "[srv] prefill=%.1f fetch=%.2f sample=%.2f to_first_tok=%.1f ms "
                                    "(wait=%.2f admit=%.2f prefill_total=%.1f chunks=%d reused=%d total=%.1f)\n",
                                    ms(t_pf0, t_pf1), ms(t_fl0, t_sm0), ms(t_sm0, t_now), ms(t_pf0, t_now), s->wait_ms,
                                    s->admit_ms, s->pf_ms, s->n_chunks, s->reused, ms(s->t_submit, t_now));
                        }
                        // NB: this path's reason string differs from the decode one
                        // below on purpose (no `&& !eos`), as it did before both used
                        // their own copy of the rule.
                        if (sample_and_emit(e, s, ubs[0], logits.data(), e.m.hp.n_vocab).done) {
                            retire(s, s->n_generated >= s->gp.max_tokens ? "length" : "stop");
                        }
                    }
                    break; // one prefill chunk per iteration
                }
            }
        }

        // ---- batched decode ----
        {
            std::lock_guard<std::mutex> lk(e.mtx);
            std::unique_lock<std::mutex> lk2(m);
            int nb = 0;
            std::vector<std::shared_ptr<sequence>> batch;
            for (auto & s : active) {
                if (s->finished) {
                    continue;
                }
                // drop it from the batch: the KV write below would otherwise
                // run for a client that is not listening
                if (s->cancelled) {
                    retire(s, "stop");
                    continue;
                }
                // Backpressure, not a stall: a consumer that cannot keep up
                // pauses this sequence (it keeps its slot and KV, and resumes
                // once drained) instead of growing out_q without bound or
                // blocking the loop for every other sequence.  The 2 ms wait
                // fallback below is what re-checks it; that is negligible
                // against a ~65 ms decode step.
                if (s->out_queue_full()) {
                    if (dbg()) {
                        fprintf(stderr, "[sched] seq=%d paused: out_q >= %zu\n", s->id,
                                sequence::kMaxOutQueue);
                    }
                    continue;
                }
                if (s->prompt_pos < (int)s->prompt.size()) {
                    continue; // still prefilling
                }
                if (s->recent.empty()) {
                    continue;
                }
                // PF_SCHED_MAX_DECODE caps the rows per decode forward.  It exists so a test can
                // pin batching off and isolate what differs between two concurrent
                // requests: with one row the decode path is bit-for-bit the
                // single-request path, so any divergence is admission or prefill.
                // Default 0 = no cap (kMaxB), the shipped behaviour.
                static const int dbg_cap = si::env::i32("PF_SCHED_MAX_DECODE", 0);
                if (nb >= (dbg_cap > 0 ? dbg_cap : kMaxB)) {
                    break;
                }
                // grow the block table if needed
                const int pos = (int)s->recent.size(); // next position to write
                const int need = (pos + kBlockSize - 1) / kBlockSize;
                while ((int)s->blocks.size() < need) {
                    int b = e.alloc_block();
                    if (b < 0) {
                        break;
                    }
                    s->blocks.push_back(b);
                    e.set_table(s->slot, s->blocks);
                }
                // the decode writes the KV of recent.back() at position pos-1;
                // it must be below max_seq and backed by an allocated block
                const int wpos = pos - 1;
                if (wpos >= e.max_seq || wpos >= (int)s->blocks.size() * kBlockSize) {
                    retire(s, "length");
                    continue;
                }
                toks[nb] = s->recent.back();
                poss[nb] = pos - 1;
                slots[nb] = s->slot;
                batch.push_back(s);
                nb++;
            }
            if (nb > 0) {
                lk2.unlock();
                e.decode_batch(toks.data(), poss.data(), slots.data(), nb);
                lk2.lock();
                if (dbg()) {
                    fprintf(stderr, "[sched] decode nb=%d:", nb);
                    for (int r = 0; r < nb; r++) {
                        fprintf(stderr, " seq=%d slot=%d pos=%d tok=%d", batch[r]->id, slots[r], poss[r], toks[r]);
                    }
                    fprintf(stderr, "\n");
                }
                did_work = true;
                for (int r = 0; r < nb; r++) {
                    auto & s = batch[r];
                    e.fetch_logits(r, logits.data());
                    // the same helper the prefill's first token uses
                    const sample_outcome o = sample_and_emit(e, s, ubs[r], logits.data(), e.m.hp.n_vocab);
                    if (o.done) {
                        retire(s, s->n_generated >= s->gp.max_tokens && !o.eos ? "length" : "stop");
                    }
                }
            }
            // remove finished
            active.erase(std::remove_if(active.begin(), active.end(),
                                        [](const std::shared_ptr<sequence> & s) { return s->finished; }),
                         active.end());
        }

        if (!did_work) {
            // Wake on the next submit/shutdown instead of a fixed 2 ms poll: the
            // old sleep added 0-2 ms to every first token.  The timeout is only
            // a fallback.
            std::unique_lock<std::mutex> lk(m);
            cv.wait_for(lk, std::chrono::milliseconds(2),
                        [&] { return stopping || !waiting.empty() || !active.empty(); });
        }
    }
}

} // namespace si
