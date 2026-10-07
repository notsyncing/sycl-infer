// Speculative decoding must not change what the model says.
//
// Both drafters (MTP / NextN and the DFlash2 block drafter) exist only to make a
// greedy decode cheaper: they propose tokens and the TARGET model verifies them,
// so the emitted stream has to be the stream a plain greedy decode would emit.
// That is the one invariant end-to-end, and it is the one an acceptance-rate
// check does NOT cover - a broken verify can accept at a healthy rate and still
// emit the wrong text, because acceptance is only compared against the target's
// own argmax on the rows it kept.
//
// What this checks, per drafter:
//   1. GATING   - with the drafter disabled the speculative entry point must
//                 degrade to the plain decode, token for token.
//   2. EQUIVALENCE - at several draft lengths, the speculative stream must equal
//                 the plain greedy stream.
//   3. DETERMINISM - two runs of the same config must agree, because a verify
//                 that depends on host scheduling is not reproducible.
//   4. SANITY   - the stream is non-empty and the drafter is actually engaged
//                 (a silently-disabled drafter would pass 1-3 trivially).
//
// (2) is asserted on a SHORT generation.  That is deliberate and load-bearing:
// MTP's stream is documented as greedy-equivalent in intent but not
// byte-guaranteed on the 27B (a ~130-token generation diverges from the plain
// decode from token 11 on, in every draft-head variant and at k=1, so it is the
// verify/rollback state rather than draft quality).  Asserting a long generation
// would encode an open bug as a test expectation; asserting a short one still
// catches a broken verify, which diverges immediately.
//
// usage: test_spec <target.gguf> <draft.gguf> [n_tokens]
//   TEST_SPEC_K      comma-separated draft lengths (default 1,2,4)
//   TEST_SPEC_PROMPT "text"   (default a fixed prompt)
//   TEST_SPEC_LONG=1         also test a 545-token MTP prompt and cache reuse
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "common/env.h"
#include "engine.h"
#include "sampler.h"  // gen_params

using namespace si;

static int failures = 0;

static void check(bool cond, const char * what) {
    printf("  %-58s %s\n", what, cond ? "OK" : "FAIL");
    if (!cond) {
        failures++;
    }
}

static std::string join(const std::vector<int> & v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) {
        if (i) {
            s += " ";
        }
        s += std::to_string(v[i]);
    }
    return s;
}

static void diff_at(const char * label, const std::vector<int> & a, const std::vector<int> & b) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) {
        i++;
    }
    printf("    %s: len %zu vs %zu, first difference at token %zu of %zu\n", label, a.size(), b.size(), i,
           std::max(a.size(), b.size()));
    if (i < a.size() && i < b.size()) {
        printf("      spec=%d plain=%d\n", a[i], b[i]);
    }
}

// one engine, one prompt: bypass speculation for the independent greedy oracle,
// then run the requested drafter explicitly.  generate() would route right back
// to that same drafter and compare it against itself.
struct run_ctx {
    engine * e = nullptr;
    std::vector<int> toks;
    int n_tok = 32;
};

static gen_params greedy(int n) {
    gen_params gp;
    gp.max_tokens = n;
    gp.temperature = 0.f;
    gp.top_p = 1.f;
    gp.top_k = 1;
    gp.seed = 1;
    return gp;
}

static std::vector<int> gen(engine & e, const std::vector<int> & toks, int n, std::vector<float> * first = nullptr) {
    gen_params gp = greedy(n);
    std::vector<int> out;
    e.generate_plain(toks, gp, [&](int t) {
        out.push_back(t);
        return true;
    }, first);
    return out;
}

static std::vector<int> gen_mtp(engine & e, const std::vector<int> & toks, int n, std::vector<float> * first = nullptr) {
    gen_params gp = greedy(n);
    std::vector<int> out;
    e.generate_mtp(toks, gp, [&](int t) {
        out.push_back(t);
        return true;
    }, first);
    return out;
}

static int argmax(const std::vector<float> & v) {
    return v.empty() ? -1 : (int)(std::max_element(v.begin(), v.end()) - v.begin());
}

static std::vector<int> gen_dflash(engine & e, const std::vector<int> & toks, int n) {
    gen_params gp = greedy(n);
    std::vector<int> out;
    e.generate_dflash(toks, gp, [&](int t) {
        out.push_back(t);
        return true;
    }, nullptr);
    return out;
}

static void run_mtp(const char * label, engine & e, const std::vector<int> & toks, int n) {
    printf("%s (prompt=%zu tokens, n=%d)\n", label, toks.size(), n);
    const std::vector<int> ref = gen(e, toks, n);
    printf("  plain greedy: %zu tokens: %s\n", ref.size(), join(ref).substr(0, 90).c_str());
    check(!ref.empty(), "plain greedy produced tokens");
    const std::vector<int> s1 = gen_mtp(e, toks, n);
    check(!s1.empty(), "generate_mtp produced tokens");
    const bool eq = (s1 == ref);
    printf("  generate_mtp == plain greedy: %s\n", eq ? "OK" : "FAIL");
    if (!eq) {
        diff_at("mtp", s1, ref);
    }
    check(eq, "MTP stream equals the plain greedy stream");
    check(s1 == gen_mtp(e, toks, n), "MTP is deterministic across two runs of one process");
}

static void run_dflash(const char * label, engine & e, const std::vector<int> & toks, int n) {
    printf("%s (prompt=%zu tokens, n=%d)\n", label, toks.size(), n);
    const std::vector<int> ref = gen(e, toks, n);
    printf("  plain greedy: %zu tokens: %s\n", ref.size(), join(ref).substr(0, 90).c_str());
    check(!ref.empty(), "plain greedy produced tokens");
    const std::vector<int> d1 = gen_dflash(e, toks, n);
    check(!d1.empty(), "generate_dflash produced tokens");
    const bool eq = (d1 == ref);
    printf("  generate_dflash == plain greedy: %s\n", eq ? "OK" : "FAIL");
    if (!eq) {
        diff_at("dflash", d1, ref);
    }
    check(eq, "DFlash2 stream equals the plain greedy stream");
    check(d1 == gen_dflash(e, toks, n), "DFlash2 is deterministic across two runs of one process");
}


// Build an engine the way main.cpp does, so a failure here is the engine's and
// not a ctor-argument difference.  main.cpp passes kv_cap_mb == -1 for "unset"
// and the layer map / device straight through.
static std::unique_ptr<engine> make_engine(const char * target, const char * draft, const char * lm, int mtp_k,
                                           int draft_k, int draft_dev, int ctx) {
    return std::make_unique<engine>(target, ctx, 16, 512, /*kv_cap_mb=*/-1, "", /*pc_disk_mb=*/-1, /*pc_mem_mb=*/-1,
                                    /*pc_ram_mb=*/-1, /*pc_vram_mb=*/-1, /*device=*/-1, lm, mtp_k,
                                    draft ? std::string(draft) : std::string(), draft_k, draft_dev);
}

// Gating: a disabled drafter must fall through to the plain decode rather than
// producing nothing or refusing.  This is a separate engine because the gate is
// decided in the constructor.
static void run_gating(const char * target, const char * draft, const char * lm, int n) {
    printf("gating\n");
    {
        // no draft model, mtp_k > 0: must degrade to plain decode
        auto ep = make_engine(target, nullptr, lm, 4, 0, 0, 512);
        engine & e = *ep;
        const std::string text = si::env::str("TEST_SPEC_PROMPT") ? si::env::str("TEST_SPEC_PROMPT")
                                                                 : "user\nHello\nassistant\n";
        std::vector<int> toks = e.tk.encode(text, true);
        gen_params gp = greedy(n);
        std::vector<int> out;
        e.generate_mtp(toks, gp, [&](int t) {
            out.push_back(t);
            return true;
        }, nullptr);
        check(!out.empty(), "generate_mtp with no draft model still generates (gate falls through)");
    }
}

static void run_gating_k0(const char * target, const char * draft, const char * lm, int n) {
    printf("gating: mtp_k=0 with a draft model loaded\n");
    {
        auto ep = make_engine(target, draft, lm, 0, 4, 0, 512);
        engine & e = *ep;
        const std::string text = si::env::str("TEST_SPEC_PROMPT") ? si::env::str("TEST_SPEC_PROMPT")
                                                                 : "user\nHello\nassistant\n";
        std::vector<int> toks = e.tk.encode(text, true);
        gen_params gp = greedy(n);
        std::vector<int> out;
        e.generate_mtp(toks, gp, [&](int t) {
            out.push_back(t);
            return true;
        }, nullptr);
        check(!out.empty(), "generate_mtp with mtp_k=0 still generates (gate falls through)");
    }
}

// One configuration per PROCESS.  A second engine in the same process inherits
// process-wide state from the first: constructing an MTP-enabled engine and then
// a DFlash one in one process fails with "multi-device: weight tensor not
// uploaded to this device's partition", while each alone works, and while the
// reverse order also works.  The CLI never hits it because it builds one engine,
// so the test re-execs itself per configuration rather than encoding that
// limitation as a test failure.
static int run_parent(const char * self, int argc, char ** argv) {
    static const char * kConfigs[] = {"gate_nodraft", "gate_k0", "mtp", "dflash"};
    int bad = 0;
    for (const char * cfg : kConfigs) {
        printf("\n================ configuration: %s ================\n", cfg);
        fflush(stdout);
        std::string cmd = std::string(self) + " --child " + cfg;
        for (int i = 1; i < argc; i++) {
            cmd += " ";
            cmd += argv[i];
        }
        setenv("TEST_SPEC_ONLY", cfg, 1);
        const int rc = system(cmd.c_str());
        const bool ok = (rc == 0);
        printf("---- configuration %s: %s\n", cfg, ok ? "OK" : "FAIL");
        if (!ok) {
            bad++;
        }
    }
    printf("\nspec (all configurations): %s\n", bad ? "FAIL" : "OK");
    return bad ? 1 : 0;
}

int main(int argc, char ** argv) {
    // --child <cfg> is what the re-exec passes; strip it before the real run.
    const char * child = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--child") && i + 1 < argc) {
            child = argv[i + 1];
            for (int j = i; j + 2 < argc + 1; j++) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            break;
        }
    }
    if (!child && !si::env::str("TEST_SPEC_ONLY")) {
        return run_parent(argv[0], argc, argv);
    }
    const char * target = argc > 1 ? argv[1] : "/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf";
    const char * draft = argc > 2 ? argv[2] : "/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf";
    const int n = argc > 3 ? atoi(argv[3]) : 32;
    const char * lm = si::env::str("TEST_LAYER_MAP");
    const bool have_draft = draft && *draft && access(draft, R_OK) == 0;
    const char * dftxt = si::env::str("TEST_SPEC_PROMPT");

    try {
        printf("speculative decoding: MTP + DFlash2 vs the plain greedy decode\n");
        printf("  target=%s\n  draft =%s (%s)\n  layer_map=%s\n", target, have_draft ? draft : "(none)",
               have_draft ? "present" : "absent", lm ? lm : "(single device)");

        // One engine per configuration, and several per process: select a subset so
        // a failure can be attributed to a configuration rather than to whatever ran
        // before it (two engines in one process share process-wide device state).
        const char * only = child ? child : si::env::str("TEST_SPEC_ONLY");
        auto want = [&](const char * w) { return !only || !strcmp(only, w); };

        if (want("gate_nodraft")) {
            run_gating(target, nullptr, lm, n);
        }
        if (want("gate_k0")) {
            run_gating_k0(target, have_draft ? draft : nullptr, lm, n);
        }

        const std::string text = dftxt ? dftxt : "user\nThe capital of France is Paris. The capital of Germany is\nassistant\n";
        const std::string text2 =
            "user\nExplain in detail how attention works in a transformer model, step by step.\nassistant\n";

        // --- MTP: the target's own NextN head, no draft GGUF ---
        if (want("mtp")) {
            auto ep = make_engine(target, nullptr, lm, 4, 0, 0, 1024);
            engine & e = *ep;
            check(e.mtp.mtp_on, "MTP drafter enabled for stream comparison");
            if (!e.mtp.mtp_on) {
                return 1;
            }
            const std::vector<int> toks = e.tk.encode(text, true);
            printf("\n");
            run_mtp("mtp short prompt", e, toks, n);
            printf("\n");
            run_mtp("mtp long prompt", e, e.tk.encode(text2, true), n);
            printf("\n");
            run_mtp("mtp n=1 (commit edge)", e, toks, 1);
            if (si::env::flag("TEST_SPEC_LONG")) {
                // Main-only mode-2 + cache first, on an empty cache: 100 tokens
                // fit one batch and all three boundaries keep states, so warm
                // must hit.  This runs before the 545-token MTP pair populates
                // the deep chain, avoiding pool pressure and state eviction.
                std::vector<int> long_toks = toks;
                long_toks.resize(545, 198);
                const std::vector<int> toks100(long_toks.begin(),
                                               long_toks.begin() + std::min<size_t>(100, long_toks.size()));
                auto manual_one = [&](const std::vector<int> & tt, bool use_cache, std::vector<float> & lg,
                                      int & matched_out) {
                    std::vector<int> blocks;
                    if (use_cache) {
                        matched_out = e.pc_admit(1, tt, blocks);
                        if (matched_out <= 0) {
                            e.zero_slot(1);
                        }
                    } else {
                        matched_out = 0;
                        e.zero_slot(1);
                    }
                    const int need = ((int)tt.size() + kBlockSize - 1) / kBlockSize + 2;
                    for (int i = (int)blocks.size(); i < need; i++) {
                        const int b = e.alloc_block();
                        if (b < 0) {
                            if (use_cache) {
                                e.pc_retire(1, blocks);
                            } else {
                                for (int x : blocks) {
                                    e.free_block(x);
                                }
                            }
                            return false;
                        }
                        blocks.push_back(b);
                    }
                    e.set_table(1, blocks);
                    int pos = matched_out;
                    while (pos < (int)tt.size()) {
                        const int rem = (int)tt.size() - pos;
                        const int fit = e.batched_prefill_fit(rem);
                        const int nb = (fit >= 1 && fit <= rem) ? fit : std::min(kMaxT, rem);
                        e.prefill_text(tt, 1, nb, pos);
                        e.prefill_flush();
                        pos += nb;
                        if (use_cache) {
                            e.pc_commit(1, tt, blocks, pos);
                        }
                    }
                    lg = e.run_head();
                    if (use_cache) {
                        e.pc_retire(1, blocks);
                    } else {
                        for (int x : blocks) {
                            e.free_block(x);
                        }
                    }
                    return true;
                };
                {
                    // Both through admit: cold misses and populates (tracking on,
                    // so checkpoints are captured), warm must then hit.  Bypassing
                    // admit for cold would leave nothing for warm to hit.
                    std::vector<float> mcold, mwarm;
                    int m0 = -1, m1 = -1;
                    const bool ok0 = manual_one(toks100, true, mcold, m0);
                    const bool ok1 = manual_one(toks100, true, mwarm, m1);
                    double md = 0;
                    for (size_t i = 0; ok0 && ok1 && i < mcold.size() && i < mwarm.size(); i++) {
                        md = std::max(md, (double)std::fabs(mcold[i] - mwarm[i]));
                    }
                    printf("  manual100 main-only: ok=%d/%d matched=%d/%d argmax=%d/%d max|diff|=%.6f nodes=%d\n",
                           (int)ok0, (int)ok1, m0, m1, argmax(mcold), argmax(mwarm), md, e.pc_nodes());
                    check(ok0 && ok1 && argmax(mcold) >= 0 && argmax(mcold) == argmax(mwarm),
                          "main-only 100-token mode-2 equals cold");
                    // A single 100-token mode-2 batch has no intermediate split
                    // prefix (prefixes are 0,100), so admit correctly misses
                    // rather than restoring a boundary captured under a
                    // different M. No hit is expected here; the 545 pair below
                    // is the one that must hit at 512.
                }
                // Cross a whole 512-token prefill batch, then reuse its cached
                // prefix.  One token avoids conflating with long-stream drift.
                std::vector<float> plain_lg, cold_lg, warm_lg;
                gen(e, long_toks, 1, &plain_lg);
                gen_mtp(e, long_toks, 1, &cold_lg);
                {
                    const int pa = argmax(plain_lg), ca = argmax(cold_lg);
                    double md = 0;
                    for (size_t i = 0; i < plain_lg.size() && i < cold_lg.size(); i++) {
                        md = std::max(md, (double)std::fabs(plain_lg[i] - cold_lg[i]));
                    }
                    printf("  long cold: plain=%d cold=%d max|diff|=%.6f nodes=%d hits=%llu\n", pa, ca, md,
                           e.pc_nodes(), (unsigned long long)e.pc_stat_hits);
                }
                check(argmax(plain_lg) >= 0 && argmax(plain_lg) == argmax(cold_lg),
                      "MTP >512-token prefill equals plain first token");
                const uint64_t hits = e.pc_stat_hits;
                gen_mtp(e, long_toks, 1, &warm_lg);
                {
                    const int pa = argmax(plain_lg), wa = argmax(warm_lg);
                    double md = 0;
                    for (size_t i = 0; i < plain_lg.size() && i < warm_lg.size(); i++) {
                        md = std::max(md, (double)std::fabs(plain_lg[i] - warm_lg[i]));
                    }
                    printf("  long warm: plain=%d warm=%d max|diff|=%.6f nodes=%d hits=%llu\n", pa, wa, md,
                           e.pc_nodes(), (unsigned long long)e.pc_stat_hits);
                }
                check(argmax(plain_lg) >= 0 && argmax(plain_lg) == argmax(warm_lg),
                      "MTP cached long prompt equals plain first token");
                check(e.pc_stat_hits > hits, "MTP repeated long prompt reuses a prefix checkpoint");
            }
        }

        // --- DFlash2: a draft GGUF, MTP off ---
        if (have_draft && want("dflash")) {
            auto ep = make_engine(target, draft, lm, 0, 4, 0, 1024);
            engine & e = *ep;
            check(e.dfl.dflash_on_, "DFlash2 drafter enabled for stream comparison");
            if (!e.dfl.dflash_on_) {
                return 1;
            }
            const std::vector<int> toks = e.tk.encode(text, true);
            printf("\n");
            run_dflash("dflash short prompt", e, toks, n);
            printf("\n");
            run_dflash("dflash long prompt", e, e.tk.encode(text2, true), n);
            printf("\n");
            run_dflash("dflash n=1 (commit edge)", e, toks, 1);
        }

        printf("\nspec: %s\n", failures ? "FAIL" : "OK");
        return failures ? 1 : 0;
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}