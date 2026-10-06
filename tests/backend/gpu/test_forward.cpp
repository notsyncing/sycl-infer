#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "engine.h"
#include "common/env.h"

using namespace si;

// usage: test_forward <model.gguf> [tokens...]
//   without tokens: run a fixed prompt and print the top-5 tokens + logits checksum
//   with tokens: print the top-10 logits for the last position (for llama.cpp comparison)
int main(int argc, char ** argv) {
    if (!si::env::flag("TEST_DP4A")) {
        setenv("PF_DP4A", "0", 1); // strict tests validate the fp32 path
    }

    const char * model_path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    try {
        // TEST_LAYER_MAP / TEST_DEVICE let this test exercise a large model
        // split across devices (e.g. TEST_DEVICE=cpu for the host reference)
        const char * lm = si::env::str("TEST_LAYER_MAP");
        const char * dv = si::env::str("TEST_DEVICE");
        const int dev = (dv && strcmp(dv, "cpu") == 0) ? 1 : -1;
        engine e(model_path, 512, 16, 512, 0, "", -1, -1, -1, -1, dev, lm ? lm : "");
        std::vector<int> toks;
        for (int i = 2; i < argc; i++) {
            toks.push_back(atoi(argv[i]));
        }
        if (toks.empty()) {
            const std::string text = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
            toks = e.tk.encode(text);
            printf("prompt tokens (%zu):", toks.size());
            for (int t : toks) {
                printf(" %d", t);
            }
            printf("\n");
        }
        auto logits = e.eval(toks);
        // top 10
        std::vector<int> idx(logits.size());
        for (size_t i = 0; i < idx.size(); i++) {
            idx[i] = (int)i;
        }
        std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        double sum = 0;
        for (float v : logits) {
            sum += v;
        }
        printf("n_vocab=%zu sum=%.4f\n", logits.size(), sum);
        for (int i = 0; i < 10; i++) {
            printf("  top%d: id=%-7d logit=%.6f\n", i, idx[i], logits[idx[i]]);
        }
        printf("last_id=%d last_logit=%.6f\n", toks.back(), logits[toks.back()]);
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
