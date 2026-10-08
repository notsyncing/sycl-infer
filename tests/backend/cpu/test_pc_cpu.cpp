// CPU end-to-end test of the disk tier of the prefix cache and of paged
// attention on the host backend (same flow as test_pc_gpu, with --device cpu).
// The same prompt is prefilled twice: the first pass captures checkpoints and
// then every cache node is pushed to disk; the second pass must resume from the
// promoted checkpoint, restore the split paged KV and reproduce the logits.
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

// Exhaust the block pool, which is what forces every resident prefix-cache node
// down a tier (the allocator's fallback path evicts LRU-first).  Returns how many
// blocks it managed to take.
static int evict_all(engine & e) {
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
}

// ---- chunk-boundary independence (no prefix cache anywhere in this) ----------
//
// The registered known failure compares a cold pass against a warm one, and the
// two do not prefill the same way: cold walks the prompt in kMaxT pieces
// (32+32+32+4), warm restores 96 tokens and prefills only the 4-token tail.  So
// a cold/warm difference is only evidence about the *cache* if prefill itself is
// independent of where the chunk boundaries fall - and for a GDN model that is a
// real question, not a formality: the recurrence is carried across chunks in
// state, and the fused mode-2 GDN call has already had one bug where the kernel
// read the chunk length from the wrong place and only the first row advanced.
//
// This runs the same tokens through the same engine and the same slot several
// times with different boundaries and no cache at all, so a difference here
// exonerates the cache and puts the fault in the prefill path.
static std::vector<float> run_layout(engine & e, const std::vector<int> & prompt,
                                     const std::vector<int> & bounds) {
    const int slot = 0;
    e.zero_slot(slot);
    std::vector<int> blocks;
    const int need = ((int)prompt.size() + kBlockSize - 1) / kBlockSize;
    for (int i = 0; i < need; i++) {
        const int b = e.alloc_block();
        if (b < 0) {
            return {};
        }
        blocks.push_back(b);
    }
    e.set_table(slot, blocks);
    int pos = 0;
    for (size_t i = 0; i + 1 < bounds.size(); i++) {
        const int n = bounds[i + 1] - bounds[i];
        if (n <= 0 || n > kMaxT) {
            return {}; // not a legal chunk for this engine
        }
        e.prefill_chunk(prompt, pos, n, slot, i + 2 == bounds.size());
        pos = bounds[i + 1];
    }
    std::vector<float> logits(e.m.hp.n_vocab);
    e.fetch_logits(0, logits.data());
    return logits;
}

static void compare_layouts(const char * tag, const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        printf("  chunk-layout %s: SKIP (empty result)\n", tag);
        return;
    }
    double maxd = 0;
    int ia = 0, ib = 0;
    for (size_t i = 0; i < a.size(); i++) {
        maxd = std::max(maxd, (double)std::fabs(a[i] - b[i]));
        if (a[i] > a[ia]) {
            ia = (int)i;
        }
        if (b[i] > b[ib]) {
            ib = (int)i;
        }
    }
    printf("  chunk-layout %-18s max|diff|=%.9g argmax %d/%d %s\n", tag, maxd, ia, ib, ia == ib ? "SAME" : "DIFFERENT");
}

int main(int argc, char ** argv) {
    setenv("PF_DP4A", "0", 1); // deterministic fp32 weight path (CPU default)
    setenv("PF_PC_STATES", "2", 1);
    setenv("PF_PC_RAM_MB", "0", 1); // this test exercises the disk tier directly
    const char * model_path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";

    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("si_pc_cpu_" + std::to_string((long)getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);

    const std::string prompt_text = "<|im_start|>user\nThe quick brown fox jumps over the lazy dog.";
    try {
        // 64 MB disk (one state checkpoint is ~19 MB); the block pool rounds
        // its reserve up to the 2 MB mapping granule
        std::vector<int> prompt;
        {
        engine_config ec;
        ec.model_path = model_path;
        ec.max_seq = 512;
        ec.n_blocks = 16;
        ec.kv_cap_mb = -1;
        ec.pc_dir = dir.string();
        ec.pc_disk_mb = 64;
        ec.device = 1; // the CPU partition under test
        engine e(ec);
        CHECK(e.pc_on());
        CHECK(e.pcd_enabled);

        // 100-token prompt: 3 full blocks (boundaries 32/64/96) + a 4-token tail
        prompt = e.tk.encode(prompt_text);
        while ((int)prompt.size() < 100) {
            prompt.push_back(198); // newline, any valid id works here
        }
        prompt.resize(100);

        // Chunk-boundary independence, before anything touches the cache.  A
        // (what the cold pass below does) vs B (same tokens, boundaries moved).
        // If these differ, the cold/warm comparison later in this file is
        // measuring the prefill path and not the prefix cache at all.
        printf("chunk-boundary independence (no prefix cache)\n");
        const std::vector<float> lay_a = run_layout(e, prompt, {0, 32, 64, 96, 100});
        const std::vector<float> lay_b = run_layout(e, prompt, {0, 32, 64, 80, 96, 100});
        const std::vector<float> lay_c = run_layout(e, prompt, {0, 20, 52, 84, 100});
        compare_layouts("A 32+32+32+4", lay_a, lay_b);   // A vs B
        compare_layouts("A 32+32+32+4", lay_a, lay_c);   // A vs C
        compare_layouts("B 32+32+16+16+4", lay_b, lay_c); // B vs C
        CHECK(!lay_a.empty());

        int matched_cold = -1;
        std::vector<int> cold_blocks;
        const std::vector<float> cold = run_prompt(e, prompt, matched_cold, &cold_blocks);
        CHECK(!cold.empty());
        CHECK(matched_cold == 0);

        // snapshot the KV of the cached blocks before they are spilled.
        //
        // Read *every* attention layer and both K and V.  The first version of
        // this read layer 0's K only, and used to conclude "KV round-trips
        // exactly" from that - while PF_PC_VERIFY, which compares the whole blob
        // (all layers, K + V, both scale planes), reported a mismatch at byte 0
        // on every single promote.  A narrow check reports "fine" whenever the
        // corruption happens to miss the bytes it looks at.
        const size_t per_block = (size_t)e.m.hp.n_head_kv * kBlockSize * e.m.hp.head_dim;
        auto read_block = [&](int b, int layer, int which) {
            std::vector<float> v(per_block);
            e.kv_read_vec(which, e.kv_block_elem_off(which, layer, b), v.data(), (int)per_block);
            return v;
        };
        // [block][K|V][layer] -> that unit's values
        std::vector<std::vector<std::vector<std::vector<float>>>> cold_kv;
        for (int i = 0; i < 3; i++) {
            cold_kv.push_back({}); // [block]
            for (int which = 0; which < 2; which++) {
                cold_kv.back().push_back({}); // [K or V]
                for (int l = 0; l < e.n_attn_layers(); l++) {
                    cold_kv.back()[which].push_back(cold_blocks[i] < 0
                                                        ? std::vector<float>()
                                                        : read_block(cold_blocks[i], l, which));
                }
            }
        }

        // exhaust the block pool, forcing every VRAM node down to disk
        CHECK(evict_all(e) > 0);
        CHECK(e.pc_nodes() == 0); // VRAM empty -> the next admit must hit disk
        CHECK(e.pc_disk_records() == 3);

        int matched_warm = -1;
        std::vector<int> warm_blocks;
        const std::vector<float> warm = run_prompt(e, prompt, matched_warm, &warm_blocks);
        CHECK(!warm.empty());
        CHECK(matched_warm == 96); // resumed from the deepest disk checkpoint
        CHECK(e.pc_disk_records() == 0); // disk -> VRAM is a move
        // One line per (block, K/V, layer): which of them came back different.
        int n_units = 0, n_units_bad = 0;
        size_t worst_any = 0;
        double worst_any_d = 0;
        for (int i = 0; i < 3; i++) {
            if (warm_blocks[i] < 0) {
                continue;
            }
            for (int which = 0; which < 2; which++) {
                for (int l = 0; l < e.n_attn_layers(); l++) {
                    const std::vector<float> & before = cold_kv[i][which][l];
                    if (before.empty()) {
                        continue;
                    }
                    const std::vector<float> now = read_block(warm_blocks[i], l, which);
                    double d = 0;
                    size_t at = 0, n_diff = 0, first = now.size(), n_warm_zero = 0, n_cold_zero = 0;
                    for (size_t k = 0; k < now.size() && k < before.size(); k++) {
                        if (now[k] == before[k]) {
                            continue;
                        }
                        n_diff++;
                        first = std::min(first, k);
                        // all-zero on the warm side points at the *scale* plane being
                        // lost (every value in the block scales to 0); a scattered
                        // non-zero pattern points at the data plane instead
                        n_warm_zero += (now[k] == 0.0f) ? 1 : 0;
                        n_cold_zero += (before[k] == 0.0f) ? 1 : 0;
                        const double diff = std::fabs((double)now[k] - (double)before[k]);
                        if (diff > d) {
                            d = diff;
                            at = k;
                        }
                    }
                    n_units++;
                    const bool bad = (d != 0.0);
                    n_units_bad += bad ? 1 : 0;
                    if (d > worst_any_d) {
                        worst_any_d = d;
                        worst_any = at;
                    }
                    printf("  blk%d %s L%-2d cold_blk=%d warm_blk=%d n_diff=%zu/%zu (warm_zero=%zu cold_zero=%zu) "
                           "first=%zu max|d|=%.9g at %zu (cold=%.9g warm=%.9g) %s\n",
                           i, which ? "V" : "K", l, cold_blocks[i], warm_blocks[i], n_diff, now.size(), n_warm_zero,
                           n_cold_zero, first, d, at, (double)before[at], (double)now[at], bad ? "DIFFERS" : "identical");
                }
            }
        }
        // The whole KV of every cached block, every layer, K and V, must come
        // back bit-exact.  A per-layer-K-only check used to pass here while the
        // blob was wrong from its first byte.
        printf("  KV units compared: %d, differing: %d, worst max|d|=%.9g at element %zu of its unit\n", n_units, n_units_bad,
               worst_any_d, worst_any);
        CHECK(n_units > 0);
        CHECK(n_units_bad == 0);

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
        printf("pc_cpu: matched=%d logits max|diff|=%.6f argmax %d/%d %s\n", matched_warm, maxd, amax_cold, amax_warm,
               amax_cold == amax_warm ? "SAME" : "DIFFERENT");
        CHECK(amax_cold == amax_warm);
        CHECK(maxd == 0.0); // resumed state + KV must reproduce the cold logits
        // push them back to disk so the restart below has records to load
        CHECK(evict_all(e) > 0);
        CHECK(e.pc_disk_records() == 3);
        } // engine e
        // a fresh engine must rebuild the index from the directory at startup
        // and resume the same prefix from disk
        engine_config ec2;
        ec2.model_path = model_path;
        ec2.max_seq = 512;
        ec2.n_blocks = 16;
        ec2.kv_cap_mb = -1;
        ec2.pc_dir = dir.string();
        ec2.pc_disk_mb = 64;
        ec2.device = 1;
        engine e2(ec2);
        CHECK(e2.pc_disk_records() == 3);
        int matched_restart = -1;
        const std::vector<float> restart = run_prompt(e2, prompt, matched_restart);
        CHECK(!restart.empty());
        CHECK(matched_restart == 96);
    // ---- the RAM tier: the third demotion path, and the only one this test
        // never exercised (PF_PC_RAM_MB=0 until now, so pc_promote_ram never ran).
        //
        // One record is ~0.2 MB of blob plus ~19.3 MB of recurrent state, so a
        // 21 MB budget holds exactly one and spills the rest - which is the point:
        // a single run then exercises pc_promote_ram *and* pc_promote_disk, so the
        // round-trip guards cover both paths.  A budget that held everything (or
        // nothing) would cover only one.
        {
            const fs::path ram_dir = dir / "ram";
            fs::remove_all(ram_dir, ec);
            engine_config ec3;
            ec3.model_path = model_path;
            ec3.max_seq = 512;
            ec3.n_blocks = 16;
            ec3.kv_cap_mb = -1;
            ec3.pc_dir = ram_dir.string();
            ec3.pc_disk_mb = 64;
            ec3.pc_ram_mb = 21; // ~1 record: RAM takes one, disk gets the overflow
            ec3.device = 1;
            engine e3(ec3);
            CHECK(e3.pcr_enabled);
            CHECK(e3.pcd_enabled);

            std::vector<int> p3 = e3.tk.encode(prompt_text);
            while ((int)p3.size() < 100) {
                p3.push_back(198);
            }
            p3.resize(100);

            int m3 = -1;
            CHECK(!run_prompt(e3, p3, m3).empty());
            CHECK(m3 == 0); // cold: nothing cached yet

            CHECK(evict_all(e3) > 0);
            printf("  RAM tier after eviction: %zu record(s) in RAM, %zu on disk\n", e3.pc_ram_records(),
                   e3.pc_disk_records());
            // both tiers must hold something, or this phase only covered one path
            CHECK(e3.pc_ram_records() > 0);
            CHECK(e3.pc_disk_records() > 0);

            const uint64_t ram_loads0 = e3.pc_stat_ram_loads;
            int m3w = -1;
            const std::vector<float> warm3 = run_prompt(e3, p3, m3w);
            printf("  RAM-tier warm resume: matched=%d, ram loads=%llu\n", m3w,
                   (unsigned long long)(e3.pc_stat_ram_loads - ram_loads0));
            CHECK(!warm3.empty());
            CHECK(m3w > 0); // resumed from a lower tier rather than recomputing
            // the whole point of the phase: the RAM record really was loaded
            CHECK(e3.pc_stat_ram_loads > ram_loads0);
        }
        fs::remove_all(dir / "ram", ec);
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        fs::remove_all(dir, ec);
        return 1;
    }

    fs::remove_all(dir, ec);
    if (g_fail) {
        fprintf(stderr, "pc_cpu: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("pc_cpu: all checks OK\n");
    return 0;
}
