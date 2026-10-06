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
#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    const char * lm = si::env::str("TEST_LAYER_MAP");
    try {
        engine e(model_path, 512, 16, 512, 0, "", -1, -1, -1, -1, -1, lm ? lm : "");
        std::vector<int> toks;
        for (int i = 2; i < argc; i++) {
            toks.push_back(atoi(argv[i]));
        }
        if (toks.empty()) {
            const std::string text = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
            toks = e.tk.encode(text, /*parse_special=*/true);
        }

        // greedy two-step generation from `head`: got[0] comes from the head
        // prefill, got[1] from decoding got[0] one token at a time
        gen_params gp;
        gp.max_tokens = 2;
        gp.temperature = 0.f;
        gp.top_p = 1.f;
        gp.top_k = 1;
        std::vector<int> got;
        e.generate(toks, gp, [&](int t) {
            got.push_back(t);
            return true;
        });
        if (got.size() < 2) {
            printf("decode-vs-prefill: FAIL (only %zu tokens generated)\n", got.size());
            return 1;
        }

        // re-prefill the same sequence (head + first generated token) and take
        // its argmax: it must agree with what the decode step predicted
        std::vector<int> seq = toks;
        seq.push_back(got[0]);
        const int t_ref = argmax(e.eval(seq));
        const bool ok = (got[1] == t_ref);
        printf("decode-vs-prefill: prompt=%zu gen0=%d gen1(decode)=%d ref(prefill)=%d %s\n", toks.size(), got[0],
               got[1], t_ref, ok ? "OK" : "MISMATCH");
        return ok ? 0 : 1;
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
