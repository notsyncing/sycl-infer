// GPU end-to-end test of the RAM tier of the prefix cache (needs the GPU and
// the reference model, no disk directory): a prefix evicted from VRAM must be
// preserved in host RAM and promoted back with identical logits.  A fresh
// engine (empty RAM, no disk) must not see the old entries.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <unistd.h>

#include "engine.h"

using namespace si;

static int g_fail = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                            \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

// mirror of the scheduler's chunked prefill for a single sequence in slot 0
static std::vector<float> run_prompt(engine & e, const std::vector<int> & prompt, int & matched_out) {
    const int slot = 0;
    std::vector<int> blocks;
    const int matched = e.pc_admit(slot, prompt, blocks);
    matched_out = matched;
    if (matched <= 0) {
        e.zero_slot(slot);
    }
    const int first_end = std::min<int>(matched + kMaxT, (int)prompt.size());
    const int need = (std::max(first_end, 1) + kBlockSize - 1) / kBlockSize;
    for (int i = (int)blocks.size(); i < need; i++) {
        const int b = e.alloc_block();
        if (b < 0) {
            return {};
        }
        blocks.push_back(b);
    }
    e.set_table(slot, blocks);
    int pos = matched;
    while (pos < (int)prompt.size()) {
        const int n = std::min<int>(kMaxT, (int)prompt.size() - pos);
        const int need2 = (pos + n + kBlockSize - 1) / kBlockSize;
        while ((int)blocks.size() < need2) {
            const int b = e.alloc_block();
            if (b < 0) {
                break;
            }
            blocks.push_back(b);
            e.set_table(slot, blocks);
        }
        if ((int)blocks.size() * kBlockSize < pos + n) {
            return {};
        }
        const bool last = (pos + n >= (int)prompt.size());
        e.prefill_chunk(prompt, pos, n, slot, last);
        pos += n;
        e.pc_commit(slot, prompt, blocks, pos);
    }
    std::vector<float> logits(e.m.hp.n_vocab);
    e.fetch_logits(0, logits.data());
    e.pc_retire(slot, blocks);
    return logits;
}

int main(int argc, char ** argv) {
    setenv("PF_DP4A", "0", 1); // deterministic fp32 weight path
    setenv("PF_PC_STATES", "2", 1);
    setenv("PF_PC_RAM_MB", "512", 1);
    const char * model_path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";

    const std::string prompt_text = "<|im_start|>user\nThe quick brown fox jumps over the lazy dog.";
    try {
        std::vector<int> prompt;
        {
            engine e(model_path, 512, 16, 16, -1, /*pc_dir=*/"", -1, -1);
            CHECK(e.pc_on());
            CHECK(e.pcr_enabled);  // RAM tier on by default
            CHECK(!e.pcd_enabled); // no disk directory

            prompt = e.tk.encode(prompt_text);
            while ((int)prompt.size() < 100) {
                prompt.push_back(198);
            }
            prompt.resize(100);

            int matched_cold = -1;
            const std::vector<float> cold = run_prompt(e, prompt, matched_cold);
            CHECK(!cold.empty());
            CHECK(matched_cold == 0);

            // force the VRAM nodes into RAM by exhausting the block pool
            std::vector<int> tmp;
            for (;;) {
                const int b = e.alloc_block();
                if (b < 0) {
                    break;
                }
                tmp.push_back(b);
            }
            CHECK(!tmp.empty());
            for (int b : tmp) {
                e.free_block(b);
            }
            CHECK(e.pc_nodes() == 0);
            CHECK(e.pc_ram_records() > 0);
            CHECK(e.pc_disk_records() == 0);

            int matched_warm = -1;
            const std::vector<float> warm = run_prompt(e, prompt, matched_warm);
            CHECK(!warm.empty());
            CHECK(matched_warm == 96); // resumed from RAM

            double maxd = 0;
            int amax_cold = 0, amax_warm = 0;
            for (size_t i = 0; i < cold.size() && i < warm.size(); i++) {
                maxd = std::max(maxd, (double)std::fabs(cold[i] - warm[i]));
                if (cold[i] > cold[amax_cold]) {
                    amax_cold = (int)i;
                }
                if (warm[i] > warm[amax_warm]) {
                    amax_warm = (int)i;
                }
            }
            printf("pc_ram_gpu: matched=%d logits max|diff|=%.6f argmax %d/%d %s\n", matched_warm, maxd, amax_cold,
                   amax_warm, amax_cold == amax_warm ? "SAME" : "DIFFERENT");
            CHECK(amax_cold == amax_warm);
            CHECK(maxd == 0.0);
        }
        // RAM is process-local: a fresh engine starts with nothing cached
        engine e2(model_path, 512, 16, 16, -1, "", -1, -1);
        CHECK(e2.pc_ram_records() == 0);
        int matched = -1;
        run_prompt(e2, prompt, matched);
        CHECK(matched == 0);
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }

    if (g_fail) {
        fprintf(stderr, "pc_ram_gpu: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("pc_ram_gpu: all checks OK\n");
    return 0;
}
