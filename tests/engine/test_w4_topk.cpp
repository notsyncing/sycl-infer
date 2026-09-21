// Prints the top-12 prefill logits for a chat prompt, so the int8 and u4 weight
// paths can be compared on the actual candidates (e.g. is "France" vs "Based" a
// near-tie flip, or a real gap?).  Run once with PF_W4=0 and once with PF_W4=1.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "chat.h"
#include "engine.h"

using namespace si;

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf";
    const char * text = argc > 2 ? argv[2] : "What is the capital of France? Answer in one word.";
    const char * lm = getenv("TEST_LAYER_MAP");
    engine e(model_path, 2048, 16, 512, 0, "", -1, -1, -1, -1, -1, lm ? lm : "");

    std::vector<chat_msg> msgs{chat_msg("user", text)};
    const std::string rendered = render_chat(e.m.chat_template, msgs, true, false);
    std::vector<int> toks = e.tk.encode(rendered, true);
    const char * w4 = getenv("PF_W4");
    printf("PF_W4=%s  prompt_tokens=%zu\n", w4 ? w4 : "(unset)", toks.size());

    std::vector<float> lg = e.eval(toks);
    // PF_DUMP_LOGITS=<path>: write the full logit vector so the int8 / u4 / fp32
    // paths can be compared against each other offline.
    if (const char * lp = getenv("PF_DUMP_LOGITS")) {
        if (FILE * fp = fopen(lp, "wb")) {
            fwrite(lg.data(), sizeof(float), lg.size(), fp);
            fclose(fp);
            fprintf(stderr, "[logits] wrote %zu values to %s\n", lg.size(), lp);
        }
    }
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); i++) {
        idx[i] = (int)i;
    }
    const int K = 12;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    for (int i = 0; i < K; i++) {
        printf("  #%2d id=%-7d logit=%9.4f  '%s'\n", i, idx[i], (double)lg[idx[i]], e.tk.decode({idx[i]}).c_str());
    }
    return 0;
}
