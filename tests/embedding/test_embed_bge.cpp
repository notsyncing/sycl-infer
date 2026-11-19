// Model-backed embedding integration test.  Uses the BGE-M3 GGUF as argv[1]
// (or the conventional path) and checks that the encoder produces normalised,
// semantically meaningful vectors on the selected backend.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "embedding/embedding_engine.h"

using namespace si;

namespace {
int failures = 0;
void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}
} // namespace

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "/path/to/bge-m3-q8_0.gguf";
    const char * dev = std::getenv("TEST_EMBED_DEVICE");
    embedding::engine_config cfg;
    cfg.model_path = path;
    cfg.device = dev ? dev : "cpu";
    cfg.max_seq = 64;
    try {
        embedding::engine e(cfg);
        check(e.dim() == 1024, "BGE-M3 embedding width is 1024");
        const std::vector<std::string> texts = {
            "The capital of France is Paris.",
            "France's capital city is Paris.",
            "The quick brown fox jumps over the lazy dog.",
        };
        const auto v = e.embed(texts);
        check(v.size() == texts.size(), "one vector per input");
        if (v.size() != texts.size()) return 1;
        for (size_t i = 0; i < v.size(); ++i) {
            check(v[i].size() == (size_t)e.dim(), "vector width");
            double n = 0;
            for (float f : v[i]) n += (double)f * f;
            check(std::fabs(std::sqrt(n) - 1.0) < 1e-3, "L2-normalised");
        }
        const double related = cosine(v[0], v[1]);
        const double unrelated = cosine(v[0], v[2]);
        printf("[%s] sim(related)=%.4f sim(unrelated)=%.4f\n",
               e.device().c_str(), related, unrelated);
        check(related > 0.75, "paraphrases are similar");
        check(related - unrelated > 0.25, "unrelated text is clearly further away");

        // Multilingual regression: BGE-M3's tokenizer is SentencePiece (t5),
        // not BPE.  A spurious <unk>, a duplicated EOS (SEP == EOS), a
        // transposed embedding table or a wrong attention output permutation
        // all leave English paraphrases plausible while collapsing these
        // Chinese pairs (observed 0.523 / 0.111 against llama.cpp's 0.89 /
        // 0.64 on an intermediate build).
        const std::vector<std::string> zh = {"法国首都", "巴黎首都", "猫", "喵喵定义"};
        const auto zv = e.embed(zh);
        check(zv.size() == zh.size(), "one vector per Chinese input");
        if (zv.size() == zh.size()) {
            const double zh_capital = cosine(zv[0], zv[1]);
            const double zh_cat = cosine(zv[2], zv[3]);
            printf("[%s] sim(法国首都,巴黎首都)=%.4f sim(猫,喵喵定义)=%.4f\n",
                   e.device().c_str(), zh_capital, zh_cat);
            check(zh_capital > 0.80, "Chinese paraphrases match llama.cpp quality");
            check(zh_cat > 0.50, "Chinese definition pair matches llama.cpp quality");
        }
    } catch (const std::exception & ex) {
        fprintf(stderr, "FAIL: exception: %s\n", ex.what());
        return 1;
    }
    if (failures) {
        fprintf(stderr, "test_embed_bge: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_embed_bge (%s): OK\n", dev ? dev : "cpu");
    return 0;
}
