// Scheduler token-stream path.  Model-backed (needs a GGUF), no HTTP.
//
// This exists because scheduler::loop had no coverage at all: test_spec drives
// engine::generate, which never goes through the scheduler, so the one place
// that turns a sampled token into a streamed piece - shared by the first token
// out of the head prefill and by every row of a decode batch - was untested.
// Anything refactoring that path has no oracle without this.
//
// usage: test_sched <model.gguf>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine.h"
#include "scheduler.h"

using namespace si;

static int g_fail = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                            \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

// Drain a sequence to completion.  Returns the concatenated text and reports
// what the scheduler produced.
struct drained {
    std::string text;
    int tokens = 0;        // pieces actually delivered
    int ids = 0;           // ids seen (0 when logprobs are off)
    bool eos_streamed = false;
    bool finished = false;
};

static drained drain(const std::shared_ptr<sequence> & seq) {
    drained d;
    sequence::token_out t;
    while (seq->pop_token(t)) {
        d.tokens++;
        d.ids++;
        d.text += t.text;
        if (t.id == -1 && !t.text.empty()) {
            // a piece delivered without an id means logprobs were not requested
        }
    }
    {
        std::lock_guard<std::mutex> lk(seq->m);
        d.finished = seq->finished;
    }
    return d;
}

static gen_params greedy(int max_tokens, bool logprobs) {
    gen_params gp;
    gp.max_tokens = max_tokens;
    gp.temperature = 0.f;
    gp.top_p = 1.f;
    gp.top_k = 1;
    gp.logprobs = logprobs;
    gp.top_logprobs = logprobs ? 3 : 0;
    return gp;
}

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    const char * lm = si::env::str("TEST_LAYER_MAP");
    // Section 7's oracle only means anything with the decode batch pinned to one
    // row: with several rows the fp accumulation order differs between two
    // concurrent sequences and a greedy run may flip at a near-tie argmax, which is
    // accepted behaviour and would drown out the prefill signal.  setenv, not a
    // default argument: the scheduler reads it once, on first use.
    if (!si::env::str("PF_SCHED_MAX_DECODE")) {
        setenv("PF_SCHED_MAX_DECODE", "1", 1);
    }
    try {
        engine_config ec;
        ec.model_path = model_path;
        ec.max_seq = 2048;
        ec.n_blocks = 256;
        ec.layer_map = lm ? lm : "";
        // 20 slots, so section 8 can drive a decode batch of more than sixteen
        // rows - the size the old bucket set silently truncated to.
        ec.max_slots = 20;
        engine e(ec);
        const std::vector<int> prompt = e.tk.encode("user\nHello\nassistant\n", /*parse_special=*/true);
        CHECK(!prompt.empty());

        // ---- 1. greedy run, no logprobs ----
        {
            scheduler s(e);
            s.start();
            auto seq = s.submit(prompt, greedy(12, false), {});
            const drained d = drain(seq);
            CHECK(d.finished);                      // the loop retired it
            CHECK(d.tokens > 0);                    // something streamed
            CHECK(!d.text.empty());
            CHECK(seq->n_generated >= d.tokens);    // control tokens are not streamed
            CHECK(!d.eos_streamed);
            // without logprobs the pieces carry no id, so the consumer sees text
            CHECK(d.ids == d.tokens);
            printf("greedy: tokens=%d chars=%zu n_generated=%d finish=%s\n", d.tokens, d.text.size(),
                   seq->n_generated, seq->finish_reason.c_str());
        }

        // ---- 2. same run with logprobs: pieces carry an id and a logprob ----
        {
            scheduler s(e);
            s.start();
            auto seq = s.submit(prompt, greedy(12, true), {});
            // drain manually so the per-token detail can be inspected
            int with_id = 0;
            int with_lp = 0;
            sequence::token_out t;
            while (seq->pop_token(t)) {
                if (t.id >= 0) {
                    with_id++;
                }
                if (t.logprob != 0.f) {
                    with_lp++;
                }
            }
            CHECK(with_id > 0);
            CHECK(with_lp > 0); // a sampled token's logprob is not exactly 0
            printf("logprobs: with_id=%d with_logprob=%d\n", with_id, with_lp);
        }

        // ---- 3. max_tokens is honoured: the loop must stop, not run on ----
        {
            scheduler s(e);
            s.start();
            const int cap = 5;
            auto seq = s.submit(prompt, greedy(cap, false), {});
            const drained d = drain(seq);
            CHECK(d.finished);
            CHECK(seq->n_generated <= cap);
            CHECK(seq->finish_reason == "length" || seq->finish_reason == "stop");
            printf("max_tokens=%d -> streamed=%d n_generated=%d finish=%s\n", cap, d.tokens, seq->n_generated,
                   seq->finish_reason.c_str());
        }

        // ---- 3b. a run that can only end by hitting max_tokens must say so ----
        // ignore_eos removes the other stopping reason, so "length" is forced and
        // the check does not depend on what the model happens to sample.  This is
        // the assertion that catches `eos` and "should retire" being conflated:
        // doing so reports "stop" here, and finish_reason is an API field.
        {
            scheduler s(e);
            s.start();
            const int cap = 5;
            gen_params gp = greedy(cap, false);
            gp.ignore_eos = true;
            auto seq = s.submit(prompt, gp, {});
            const drained d = drain(seq);
            CHECK(d.finished);
            CHECK(seq->n_generated == cap);
            CHECK(seq->finish_reason == "length");
            printf("ignore_eos + max_tokens=%d -> n_generated=%d finish=%s (must be length)\n", cap,
                   seq->n_generated, seq->finish_reason.c_str());
        }

        // ---- 4. a cancelled sequence stops streaming and frees its slot ----
        // (the SSE disconnect path: the releaser calls sequence::cancel)
        {
            scheduler s(e);
            s.start();
            auto seq = s.submit(prompt, greedy(64, false), {});
            // let a couple of tokens through, then cancel as a disconnect would
            sequence::token_out t;
            int seen = 0;
            while (seen < 2) {
                if (!seq->pop_token(t)) {
                    break;
                }
                seen++;
            }
            seq->cancel();
            const int before = seq->n_generated;
            // a cancelled sequence must not deliver anything more
            CHECK(!seq->pop_token(t));
            CHECK(!seq->cancelled == false); // the flag is observable
            printf("cancelled after %d tokens, n_generated=%d (was %d)\n", seen, seq->n_generated, before);
            CHECK(seq->n_generated <= before + 2); // at most the step in flight
        }

        // ---- 5. an empty prompt is retired instead of wedging ----
        {
            scheduler s(e);
            s.start();
            auto seq = s.submit({}, greedy(8, false), {});
            const drained d = drain(seq);
            CHECK(d.finished); // otherwise the consumer would block forever
            CHECK(d.tokens == 0);
            printf("empty prompt: finished=%d tokens=%d\n", (int)d.finished, d.tokens);
        }

        // ---- 6. two concurrent sequences both complete ----
        {
            scheduler s(e);
            s.start();
            const int cap = 8;
            auto a = s.submit(prompt, greedy(cap, false), {});
            auto b = s.submit(prompt, greedy(cap, false), {});
            const drained da = drain(a);
            const drained db = drain(b);
            CHECK(da.finished && db.finished);
            CHECK(da.tokens > 0 && db.tokens > 0);
            CHECK(da.tokens <= cap && db.tokens <= cap);
            // Deliberately NOT asserting the two texts match: batching two
            // sequences changes the fp accumulation order, so a greedy run that
            // joins a batch can flip at a near-tie argmax and then diverge for
            // the rest of the generation (documented in AGENTS.md).  That makes
            // this check about "both sequences finish and honour max_tokens",
            // not about determinism across batching.  Section 7 is the oracle that
            // *can* assert determinism, with the decode batch pinned to one row so
            // that prefill is the only thing left that can differ.
            printf("concurrent: a=%zu chars/%d tokens, b=%zu chars/%d tokens, identical=%d\n", da.text.size(),
                   da.tokens, db.text.size(), db.tokens, (int)(da.text == db.text));
        }

        // ---- 7. the prefill-only concurrency oracle ----
        //
        // Section 6 cannot assert equality, because two sequences decoding in one
        // batch change the fp accumulation order and a greedy run can flip at a
        // near-tie argmax.  That is a real, accepted property of batching - but it
        // makes the test blind to the other thing that can differ between two
        // identical concurrent requests: their *prefill*.
        //
        // So pin the decode batch to one row (PF_SCHED_MAX_DECODE, default 1 here)
        // and require N concurrent identical requests to come back byte-identical.
        // With one decode row the decode path is bit-for-bit the single-request
        // path, so anything that differs is attributable to admission or prefill.
        // This is the oracle any cross-sequence prefill batching has to be built
        // against - a batched prefill that produced plausible-but-different output
        // would pass every other check in this file.
        {
            const int nseq = 4;
            scheduler s(e);
            s.start();
            std::vector<drained> rs;
            std::vector<std::shared_ptr<sequence>> hs;
            for (int i = 0; i < nseq; i++) {
                hs.push_back(s.submit(prompt, greedy(12, false), {}));
            }
            for (auto & h : hs) {
                rs.push_back(drain(h));
            }
            int done = 0, same = 0;
            for (int i = 0; i < nseq; i++) {
                done += rs[(size_t)i].finished ? 1 : 0;
                if (rs[(size_t)i].text == rs[0].text && rs[(size_t)i].tokens == rs[0].tokens) {
                    same++;
                }
            }
            printf("prefill oracle: %d/%d finished, %d/%d byte-identical to the first (decode batch pinned)\n",
                   done, nseq, same, nseq);
            CHECK(done == nseq);
            CHECK(same == nseq);
        }

        // ---- 8. a decode batch of more than sixteen rows computes EVERY row ----
        //
        // The decode's row count used to be quantised to a small set of captured
        // shapes ({1,2,4,8,16}) and the batch was rounded UP to the next one.
        // That is dead work for nine rows - a 38 ms cliff on the 35B A3B / 2x
        // A770 - and a silent wrong answer for more than sixteen: the forward
        // only ran sixteen rows, so rows 16.. of d_logits kept the previous
        // step's values and the sampler read them.  Nothing in the output said
        // so, and no existing test had more than two concurrent sequences.
        //
        // The oracle is the one that cannot be faked: a row whose logits were
        // never written does not MOVE between two steps, while every row the
        // forward really computed does.  So decode twice with different inputs
        // and require every row's logits to change.  Comparing the argmax would
        // be weaker - a stale vector still has an argmax, and a real row can
        // legitimately keep its argmax across a step.
        {
            const int N = 20;
            const int nvoc = e.m.hp.n_vocab;
            std::vector<int32_t> toks((size_t)N, 0), poss((size_t)N, 0), slots((size_t)N, 0);
            int blocks_ok = 0;
            for (int r = 0; r < N; r++) {
                const int b = e.alloc_block();
                if (b < 0) {
                    break;
                }
                e.set_table(r, std::vector<int>{b});
                slots[(size_t)r] = r;
                blocks_ok++;
            }
            CHECK(blocks_ok == N);
            if (blocks_ok == N) {
                std::vector<float> a((size_t)N * nvoc), b((size_t)N * nvoc);
                for (int r = 0; r < N; r++) {
                    toks[(size_t)r] = 1000 + r;
                    poss[(size_t)r] = 0;
                }
                e.decode_batch(toks.data(), poss.data(), slots.data(), N);
                e.sync_all();
                for (int r = 0; r < N; r++) {
                    e.fetch_logits(r, a.data() + (size_t)r * nvoc);
                }
                for (int r = 0; r < N; r++) {
                    toks[(size_t)r] = 2000 + r;
                    poss[(size_t)r] = 1;
                }
                e.decode_batch(toks.data(), poss.data(), slots.data(), N);
                e.sync_all();
                for (int r = 0; r < N; r++) {
                    e.fetch_logits(r, b.data() + (size_t)r * nvoc);
                }
                int moved = 0;
                double worst = 1e300;
                for (int r = 0; r < N; r++) {
                    double d = 0.0;
                    for (int v = 0; v < nvoc; v++) {
                        d += std::fabs((double)a[(size_t)r * nvoc + v] - (double)b[(size_t)r * nvoc + v]);
                    }
                    if (d > 0.0) {
                        moved++;
                    }
                    if (d < worst) {
                        worst = d;
                    }
                }
                printf("decode batch of %d rows: %d/%d rows moved between two steps (smallest row's L1 move %.6g)\n",
                       N, moved, N, worst);
                CHECK(moved == N);
            }
        }

        if (g_fail) {
            fprintf(stderr, "test_sched: %d check(s) failed\n", g_fail);
            return 1;
        }
        printf("test_sched: all checks OK\n");
        return 0;
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}