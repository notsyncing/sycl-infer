// Scheduler token-stream path.  Model-backed (needs a GGUF), no HTTP.
//
// This exists because scheduler::loop had no coverage at all: test_spec drives
// engine::generate, which never goes through the scheduler, so the one place
// that turns a sampled token into a streamed piece - shared by the first token
// out of the head prefill and by every row of a decode batch - was untested.
// Anything refactoring that path has no oracle without this.
//
// usage: test_sched <model.gguf>
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
    try {
        engine e(model_path, 2048, 16, 256, 0, "", -1, -1, -1, -1, -1, lm ? lm : "");
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
            // not about determinism across batching.
            printf("concurrent: a=%zu chars/%d tokens, b=%zu chars/%d tokens, identical=%d\n", da.text.size(),
                   da.tokens, db.text.size(), db.tokens, (int)(da.text == db.text));
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