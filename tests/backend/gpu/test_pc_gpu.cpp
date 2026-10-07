// GPU end-to-end test of the disk tier of the prefix cache (needs the GPU and
// the reference model).  The same prompt is prefilled twice: the first pass
// captures checkpoints and then every cache node is pushed to disk; the second
// pass must resume from the promoted checkpoint and produce the same logits.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
static std::vector<float> run_prompt(engine & e, const std::vector<int> & prompt, int & matched_out,
                                     std::vector<int> * blocks_out = nullptr) {
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
    if (blocks_out) {
        *blocks_out = blocks;
    }
    e.pc_retire(slot, blocks);
    return logits;
}

int main(int argc, char ** argv) {
    setenv("PF_DP4A", "0", 1); // deterministic fp32 weight path
    setenv("PF_PC_STATES", "2", 1);
    setenv("PF_PC_RAM_MB", "0", 1); // this test exercises the disk tier directly
    const char * model_path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";

    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("si_pc_gpu_" + std::to_string((long)getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);

    const std::string prompt_text = "<|im_start|>user\nThe quick brown fox jumps over the lazy dog.";
    try {
        // Keep the most recently appended node pinned, then evict the older
        // (non-last) node.  Swap-removal must update the moved node's block
        // reverse index as well as its hash and checkpoint owner indexes.
        {
            engine idx(model_path, 512, 16, 16, -1, "", -1, -1);
            std::vector<int> a(33, 198), b(33, 198);
            a[0] = 846;
            b[0] = 9419;
            int matched = -1;
            CHECK(!run_prompt(idx, a, matched).empty());
            CHECK(matched == 0);
            CHECK(!run_prompt(idx, b, matched).empty());
            CHECK(matched == 0);
            CHECK(idx.pc_nodes() == 2);
            CHECK(idx.pc_indices_valid());
            std::vector<int> pinned;
            CHECK(idx.pc_admit(1, b, pinned) == 32);
            CHECK(pinned.size() == 1);
            std::vector<int> busy;
            while (idx.pool_free_blocks() > 0) {
                const int block = idx.alloc_block();
                if (block < 0) {
                    break;
                }
                busy.push_back(block);
            }
            const int evicted = idx.alloc_block();
            CHECK(evicted >= 0);
            CHECK(idx.pc_nodes() == 1);
            if (!idx.pc_indices_valid()) {
                fprintf(stderr, "FAIL: non-last cache eviction left an invalid block index\n");
                return 1; // do not pass a stale index through pc_retire
            }
            idx.pc_retire(1, pinned);
            std::vector<int> again;
            CHECK(idx.pc_admit(2, b, again) == 32);
            idx.pc_retire(2, again);
            for (int block : busy) {
                idx.free_block(block);
            }
            if (evicted >= 0) {
                idx.free_block(evicted);
            }
        }
        // 64 MB disk (one state checkpoint is ~19 MB); the block pool rounds
        // its reserve up to the 2 MB mapping granule
        std::vector<int> prompt;
        {
        engine e(model_path, 512, 16, 16, -1, dir.string(), 64, -1);
        CHECK(e.pc_on());
        CHECK(e.pcd_enabled);

        // 100-token prompt: 3 full blocks (boundaries 32/64/96) + a 4-token tail
        prompt = e.tk.encode(prompt_text);
        while ((int)prompt.size() < 100) {
            prompt.push_back(198); // newline, any valid id works here
        }
        prompt.resize(100);

        int matched_cold = -1;
        std::vector<int> cold_blocks;
        const std::vector<float> cold = run_prompt(e, prompt, matched_cold, &cold_blocks);
        CHECK(!cold.empty());
        CHECK(matched_cold == 0);

        // snapshot the layer-0 K of the cached blocks before they are spilled
        const size_t per_block = (size_t)e.m.hp.n_head_kv * kBlockSize * e.m.hp.head_dim;
        auto read_block = [&](int b) {
            std::vector<float> v(per_block);
            e.kv_read_vec(0, (size_t)b * per_block, v.data(), (int)per_block);
            return v;
        };
        std::vector<std::vector<float>> cold_kv;
        for (int i = 0; i < 3; i++) {
            cold_kv.push_back(cold_blocks[i] < 0 ? std::vector<float>() : read_block(cold_blocks[i]));
        }

        // exhaust the block pool, forcing every VRAM node down to disk
        auto evict_all = [&]() {
            std::vector<int> tmp;
            for (;;) {
                const int b = e.alloc_block();
                if (b < 0) {
                    break;
                }
                tmp.push_back(b);
            }
            for (int b : tmp) {
                e.free_block(b);
            }
            return (int)tmp.size();
        };
        CHECK(evict_all() > 0);
        CHECK(e.pc_nodes() == 0); // VRAM empty -> the next admit must hit disk
        CHECK(e.pc_disk_records() == 3);

        int matched_warm = -1;
        std::vector<int> warm_blocks;
        const std::vector<float> warm = run_prompt(e, prompt, matched_warm, &warm_blocks);
        CHECK(!warm.empty());
        CHECK(matched_warm == 96); // resumed from the deepest disk checkpoint
        CHECK(e.pc_disk_records() == 0); // disk -> VRAM is a move
        for (int i = 0; i < 3; i++) {
            if (warm_blocks[i] < 0 || cold_kv[i].empty()) {
                continue;
            }
            const std::vector<float> now = read_block(warm_blocks[i]);
            double d = 0;
            for (size_t k = 0; k < now.size(); k++) {
                d = std::max(d, (double)std::fabs(now[k] - cold_kv[i][k]));
            }
            CHECK(d == 0.0); // the KV blob must round-trip bit-exactly
        }

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
        printf("pc_gpu: matched=%d logits max|diff|=%.6f argmax %d/%d %s\n", matched_warm, maxd, amax_cold, amax_warm,
               amax_cold == amax_warm ? "SAME" : "DIFFERENT");
        CHECK(amax_cold == amax_warm);
        CHECK(maxd == 0.0); // resumed state + KV must reproduce the cold logits
        // push them back to disk so the restart below has records to load
        CHECK(evict_all() > 0);
        CHECK(e.pc_disk_records() == 3);
        } // engine e
        // a fresh engine must rebuild the index from the directory at startup
        // and resume the same prefix from disk
        engine e2(model_path, 512, 16, 16, -1, dir.string(), 64, -1);
        CHECK(e2.pc_disk_records() == 3);
        int matched_restart = -1;
        const std::vector<float> restart = run_prompt(e2, prompt, matched_restart);
        CHECK(!restart.empty());
        CHECK(matched_restart == 96);
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        fs::remove_all(dir, ec);
        return 1;
    }

    fs::remove_all(dir, ec);
    if (g_fail) {
        fprintf(stderr, "pc_gpu: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("pc_gpu: all checks OK\n");
    return 0;
}
