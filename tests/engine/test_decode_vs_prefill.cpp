// Decode/prefill consistency: a causal model's logits for the next position must
// not depend on how the sequence got there, so the token a one-token *decode*
// step predicts must equal the token a *re-prefill* of the same sequence
// predicts.  This is the only check that exercises the single-token path (its
// paged-KV write, its attention and the recurrent-state update) against the
// prefill reference: a prefill-only comparison (test_forward, test_gpu_vs_ref)
// is blind to a decode-only bug, e.g. a KV layout or head mapping that only the
// decode kernels consume.
//
// usage: test_decode_vs_prefill <model.gguf> [tokens...]
//
// With no token arguments it runs a *matrix* of prompt lengths in one engine
// (one model load, N cheap checks).  The default prompt was 9 tokens, which
// cannot cross any of the boundaries below, so a bug that only appears once the
// KV spans several blocks, the prefill spans several mode-2 batches, or the
// attention passes the oneDNN key threshold was invisible here.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "engine.h"
#include "common/env.h"

using namespace si;

static int argmax(const std::vector<float> & v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); i++) {
        if (v[i] > v[best]) {
            best = (int)i;
        }
    }
    return best;
}

// The gap between the reference's top-2 logits.  A decode/prefill argmax
// disagreement means one of two very different things, and the margin is what
// tells them apart: a large margin means the decode step read different
// information (a real bug), while a margin near zero is a tie that any
// numerical difference flips - which is a different (and much less alarming)
// statement, so the number has to be printed either way.
static float top2_margin(const std::vector<float> & v, int best) {
    float second = -1e30f;
    for (size_t i = 0; i < v.size(); i++) {
        if ((int)i != best && v[i] > second) {
            second = v[i];
        }
    }
    return v[best] - second;
}

// Prompt lengths chosen to straddle each boundary rather than to be round
// numbers:
//   9    the original default
//   31/33  a 32-token KV block boundary (kBlockSize)
//   63/65  a kMaxT=32 prefill-chunk boundary (mode 1 -> mode 2)
//   512   a mode-2 batch multiple, and already past the 2048-key threshold
//   2048/2064  PF_ATTN_XMX_MIN's 2048 keys: the tail attention takes the
//          classic kernel at 2048 and the oneDNN int8 path at 2064, so a
//          disagreement between the two shows up as a decode/prefill split
static const int kLens[] = {9, 31, 33, 63, 65, 512, 2048, 2064};

// TEST_DVP_LENS overrides the matrix with a comma-separated list, so a failing
// length can be bisected without rebuilding.  Absent = the list above.
static std::vector<int> lens_to_run() {
    std::vector<int> out;
    const char * e = si::env::str("TEST_DVP_LENS");
    if (!e || !*e) {
        out.assign(std::begin(kLens), std::end(kLens));
        return out;
    }
    for (const char * p = e; *p;) {
        char * end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) {
            break;
        }
        out.push_back((int)v);
        p = (*end == ',') ? end + 1 : end;
        if (!*p) {
            break;
        }
    }
    return out;
}

// Above this reference top-2 gap, an argmax disagreement is a hard failure even
// under a layer split; below it the reference is simply ambiguous and the two
// paths can order near-equal candidates either way.
//
// Calibrated from measurements on the 0.8B (0-11/12-23 split) and the 27B
// (0-31/32-63): under a split the decode and prefill paths differ by roughly
// 0.5-1.3 logits at long context, and *the prefill's own argmax moves with the
// prefill batch shape* (PF_PFB_MAX_M=480/448/384 give 198, while 256/128/64/32
// and the uncapped 512-row batch give a different token for the same 513 tokens)
// - so the prefill is not a single-valued oracle there.  The decode path, by
// contrast, is stable: it answered 198 in every one of those configurations.
// Where the reference is confident the two agree exactly - single-device runs
// pass every length with margins 5-11.
static const float kConfidentMargin = 2.0f;

struct case_result {
    bool have_seq = false;
    int gen0 = -1, decode = -1, ref = -1;
    float margin = 0.f;
};

// One greedy two-step generation plus one re-prefill, returning both argmaxes.
// got[0] comes from the head prefill, got[1] from decoding got[0] one token at
// a time; the reference re-prefills head+got[0] and takes its argmax.
static case_result measure(engine & e, const std::vector<int> & toks) {
    case_result r;
    gen_params gp;
    gp.max_tokens = 2;
    gp.temperature = 0.f;
    gp.top_p = 1.f;
    gp.top_k = 1;
    // The engine does not emit a control token through the callback, so a
    // prompt whose top-1 is EOS yields zero tokens - which is correct behaviour
    // and would make this case fail for a reason unrelated to what is being
    // tested.  The prompts are degenerate (a handful of ids cycled N times) and
    // on the 27B some lengths do predict <|im_end|> immediately.  Stopping is
    // not what this test measures; sample EOS as an ordinary token instead.
    gp.ignore_eos = true;
    std::vector<int> got;
    e.generate(toks, gp, [&](int t) {
        got.push_back(t);
        return true;
    });
    if (got.size() < 2) {
        return r;
    }
    std::vector<int> seq = toks;
    seq.push_back(got[0]);
    const std::vector<float> ref_logits = e.eval(seq);
    r.have_seq = true;
    r.gen0 = got[0];
    r.decode = got[1];
    r.ref = argmax(ref_logits);
    r.margin = top2_margin(ref_logits, r.ref);
    return r;
}

static bool one_case(engine & e, const std::vector<int> & toks, bool split) {
    const case_result r = measure(e, toks);
    if (!r.have_seq) {
        printf("decode-vs-prefill: prompt=%zu FAIL (no tokens generated)\n", toks.size());
        return false;
    }
    const char * tag = split ? "split " : "";
    if (r.decode == r.ref) {
        printf("decode-vs-prefill: prompt=%zu gen0=%d gen1(decode)=%d ref(prefill)=%d margin=%.6f %sOK\n",
               toks.size(), r.gen0, r.decode, r.ref, r.margin, tag);
        return true;
    }
    // A disagreement.  Under a layer split at long context the *oracle itself*
    // is not reproducible - the 27B reference at 2064 alternated between
    // argmax 271 / margin 0.15 and argmax 198 / margin 3.63 across identical
    // runs - so one run cannot tell "the decode step is wrong" apart from "the
    // reference moved".  Re-measure: only a *reproducible* disagreement counts,
    // and a reproducible one under a confident reference is a real bug.
    const case_result r2 = measure(e, toks);
    // Reproducibility is about the *discrete* outcome only.  The margin jitters
    // in its last decimals run to run (measured 0.4822 vs 0.4879, 1.2710 vs
    // 1.2715 with identical argmaxes), so requiring the margin to match would
    // label ordinary noise as an unstable oracle.  A loose tolerance still
    // catches the reference actually moving to a different confidence class.
    const bool reproduced = r2.have_seq && r2.gen0 == r.gen0 && r2.decode == r.decode && r2.ref == r.ref &&
                            std::fabs(r2.margin - r.margin) < 0.05f;
    if (!split) {
        // single-device is exact and stable, so nothing else is acceptable
        printf("decode-vs-prefill: prompt=%zu gen0=%d gen1(decode)=%d ref(prefill)=%d margin=%.6f %sMISMATCH\n",
               toks.size(), r.gen0, r.decode, r.ref, r.margin, tag);
        return false;
    }
    if (!reproduced) {
        printf("decode-vs-prefill: prompt=%zu gen0=%d gen1(decode)=%d ref(prefill)=%d margin=%.6f %sUNSTABLE"
               " (rerun: gen1=%d ref=%d margin=%.6f)\n",
               toks.size(), r.gen0, r.decode, r.ref, r.margin, tag, r2.decode, r2.ref, r2.margin);
        return true;
    }
    if (r.margin < kConfidentMargin) {
        printf("decode-vs-prefill: prompt=%zu gen0=%d gen1(decode)=%d ref(prefill)=%d margin=%.6f %sNEAR-TIE\n",
               toks.size(), r.gen0, r.decode, r.ref, r.margin, tag);
        return true;
    }
    printf("decode-vs-prefill: prompt=%zu gen0=%d gen1(decode)=%d ref(prefill)=%d margin=%.6f %sMISMATCH"
           " (reproducible, confident reference)\n",
           toks.size(), r.gen0, r.decode, r.ref, r.margin, tag);
    return false;
}

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    const char * lm = si::env::str("TEST_LAYER_MAP");
    const bool split = lm && *lm;
    try {
        std::vector<int> explicit_toks;
        for (int i = 2; i < argc; i++) {
            explicit_toks.push_back(atoi(argv[i]));
        }
        // max_seq must exceed the longest case plus the two generated tokens;
        // n_blocks (512 * kBlockSize = 16384) already covers the KV
        engine_config ec;
        ec.model_path = model_path;
        // the length matrix goes past 2048, so the context must fit the longest
        ec.max_seq = 4096;
        ec.n_blocks = 512;
        ec.layer_map = lm ? lm : "";
        // TEST_MAX_SLOTS exercises --max-slots: the recurrent state is allocated
        // per slot and every [layer][slot] stride is built from it, so a wrong
        // stride shows up here as a decode-vs-prefill mismatch (or a crash) and
        // nowhere else.  Unset keeps the kMaxB default.
        if (const char * ms = si::env::str("TEST_MAX_SLOTS")) {
            ec.max_slots = atoi(ms);
        }
        if (const char * yf = si::env::str("TEST_YARN_FACTOR")) {
            // YaRN on, with an explicit factor so the test does not depend on the
            // model's declared context length
            ec.yarn = true;
            ec.yarn_factor = (float)atof(yf);
            ec.yarn_orig_ctx = 4096;
        }
        engine e(ec);
        if (!explicit_toks.empty()) {
            return one_case(e, explicit_toks, split) ? 0 : 1;
        }
        const std::vector<int> base = e.tk.encode("user\nHello\nassistant\n", /*parse_special=*/true);
        if (base.empty()) {
            fprintf(stderr, "error: empty base prompt\n");
            return 1;
        }
        bool all_ok = true;
        const std::vector<int> lens = lens_to_run();
        const size_t n_cases = lens.size();
        for (size_t i = 0; i < n_cases; i++) {
            const int n = lens[i];
            if (n + 2 > e.max_seq) {
                printf("decode-vs-prefill: prompt=%d SKIP (max_seq=%d)\n", n, e.max_seq);
                continue;
            }
            // Cycle the base ids, rotated per length so the cases do not share
            // a prefix: a shared prefix would let one case's cached blocks
            // mask another case's failure.
            std::vector<int> toks;
            toks.reserve((size_t)n);
            for (int j = 0; j < n; j++) {
                toks.push_back(base[(size_t)((j + i * 3) % base.size())]);
            }
            all_ok &= one_case(e, toks, split);
        }
        if (!all_ok) {
            printf("decode-vs-prefill: FAILED (of %zu length cases)\n", n_cases);
            return 1;
        }
        printf("decode-vs-prefill: all %zu length cases OK\n", n_cases);
        return 0;
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}