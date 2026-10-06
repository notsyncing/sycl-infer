// Decides whether the u4 weight path is *more accurate* than int8 by comparing
// both GPU results against the fp32 CPU reference (which dequantizes the GGUF
// weights exactly).  Run once with PF_W4=0 and once with PF_W4=1.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cpu_ref.h"
#include "engine.h"
#include "common/env.h"

using namespace si;

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf";
    std::vector<int> toks;
    for (int i = 2; i < argc; i++) {
        toks.push_back(atoi(argv[i]));
    }
    if (toks.empty()) {
        toks = {248045, 846, 198, 9419, 248046, 198}; // short ChatML-ish prompt
    }
    const char * w4 = si::env::str("PF_W4");
    printf("PF_W4=%s  tokens=%zu\n", w4 ? w4 : "(unset)", toks.size());

    const char * lm = si::env::str("TEST_LAYER_MAP");
    engine e(model_path, 2048, 16, 512, 0, "", -1, -1, -1, -1, -1, lm ? lm : "");
    auto gpu = e.eval(toks);
    // PF_DUMP_LOGITS=<path>: write the engine's full logit vector so it can be
    // diffed against an independent implementation (ll_logits) offline.
    if (const char * lp = si::env::str("PF_DUMP_LOGITS")) {
        if (FILE * fp = fopen(lp, "wb")) {
            fwrite(gpu.data(), sizeof(float), gpu.size(), fp);
            fclose(fp);
            fprintf(stderr, "[logits] wrote %zu values to %s\n", gpu.size(), lp);
        }
    }

    cpu_ref ref(e.m, 512);
    ref.forward(toks);
    ref.head();

    // metrics against the fp32 reference
    double maxd = 0, sumd = 0;
    int arg_gpu = 0, arg_ref = 0;
    for (size_t i = 0; i < gpu.size() && i < ref.logits.size(); i++) {
        const double d = std::fabs((double)gpu[i] - (double)ref.logits[i]);
        maxd = std::max(maxd, d);
        sumd += d;
        if (gpu[i] > gpu[arg_gpu]) {
            arg_gpu = (int)i;
        }
        if (ref.logits[i] > ref.logits[arg_ref]) {
            arg_ref = (int)i;
        }
    }
    const double n = (double)std::min(gpu.size(), ref.logits.size());
    printf("vs fp32 cpu ref: max|diff|=%.4f  mean|diff|=%.4f  argmax gpu=%d ref=%d %s\n", maxd, sumd / n, arg_gpu,
           arg_ref, arg_gpu == arg_ref ? "SAME" : "DIFFERENT");
    printf("  gpu[argmax]=%.4f  ref[argmax]=%.4f\n", (double)gpu[arg_gpu], (double)ref.logits[arg_ref]);
    return 0;
}
