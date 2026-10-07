// engine (GPU) end-to-end vs cpu reference
#include <cmath>
#include <cstdio>
#include <vector>

#include "engine.h"
#include "cpu_ref.h"

using namespace si;

int main(int argc, char ** argv) {
    setenv("PF_DP4A", "0", 1); // strict tests validate the fp32 path

    const char * model_path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    std::vector<int> toks;
    for (int i = 2; i < argc; i++) {
        toks.push_back(atoi(argv[i]));
    }
    if (toks.empty()) {
        toks = {248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198};
    }

    engine e(model_path, 512);
    auto gpu = e.eval(toks);

    cpu_ref ref(e.m, 512);
    ref.forward(toks);
    ref.head();

    if (gpu.empty() || gpu.size() != ref.logits.size()) {
        fprintf(stderr, "FAIL: logits size mismatch/empty: gpu=%zu ref=%zu\n", gpu.size(), ref.logits.size());
        return 1;
    }
    size_t n_nonfinite = 0;
    for (size_t i = 0; i < gpu.size(); i++) {
        if (!std::isfinite(gpu[i]) || !std::isfinite(ref.logits[i])) {
            n_nonfinite++;
        }
    }
    if (n_nonfinite != 0) {
        fprintf(stderr, "FAIL: %zu nonfinite logit value(s) (gpu or ref)\n", n_nonfinite);
        return 1;
    }

    double maxd = 0, maxv = 0;
    int argmax_gpu = 0, argmax_ref = 0;
    for (size_t i = 0; i < gpu.size(); i++) {
        double d = std::fabs((double)gpu[i] - ref.logits[i]);
        if (d > maxd) {
            maxd = d;
        }
        if (std::fabs((double)ref.logits[i]) > maxv) {
            maxv = std::fabs((double)ref.logits[i]);
        }
        if (gpu[i] > gpu[argmax_gpu]) {
            argmax_gpu = (int)i;
        }
        if (ref.logits[i] > ref.logits[argmax_ref]) {
            argmax_ref = (int)i;
        }
    }
    printf("logits: max|diff|=%.6f (max|ref|=%.2f) argmax gpu=%d ref=%d %s\n", maxd, maxv, argmax_gpu, argmax_ref,
           argmax_gpu == argmax_ref ? "SAME" : "DIFFERENT");
    return argmax_gpu == argmax_ref ? 0 : 1;
}
