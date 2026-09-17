#include <algorithm>
#include <cstdio>
#include <vector>

#include "model.h"

#include "cpu_ref.h"

using namespace si;

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    std::vector<int> toks;
    for (int i = 2; i < argc; i++) toks.push_back(atoi(argv[i]));
    if (toks.empty()) toks = {9419}; // "Hello"

    model m;
    m.load(model_path);
    cpu_ref ref(m, 128);
    ref.forward(toks);
    {
        // compare with llama.cpp result_norm (post output_norm hidden state)
        std::vector<float> hn(m.hp.n_embd);
        cpu_ref::rmsnorm(&ref.x[(size_t)(ref.n_tokens-1)*m.hp.n_embd], m.output_norm, hn.data(), m.hp.n_embd, m.hp.rms_eps);
        double sum=0, ss=0;
        for (float v : hn) { sum += v; ss += (double)v*v; }
        printf("cpuref hidden: sum=%.6f sumsq=%.6f\n", sum, ss);
        for (int i=0;i<8;i++) printf("  h[%d]=%.6f\n", i, hn[i]);
    }
    ref.head();
    printf("cpu ref: n_vocab=%d\n", m.hp.n_vocab);
    std::vector<int> idx(ref.logits.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(),
                      [&](int a, int b) { return ref.logits[a] > ref.logits[b]; });
    for (int i = 0; i < 10; i++)
        printf("  top%d id=%d logit=%.6f\n", i, idx[i], ref.logits[idx[i]]);
    double sum = 0;
    for (float v : ref.logits) sum += v;
    printf("  sum=%.4f\n", sum);
    return 0;
}
