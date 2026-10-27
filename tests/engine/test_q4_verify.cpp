// Is the MTP verify affordable on qwen4exp?
//
// The verify is a single-row forward of n consecutive tokens followed by the
// LM head.  With n < kMaxT that is exactly what prefill_text does on this
// architecture (q4_prefill_text -> q4_forward(1, n, /*with_head=*/true)), so the
// shape can be priced through the public API: no drafter, no engine change, and
// no private q4_forward call.
//
// A ~96-token warm prompt is prefilled once; that forward also calibrates the
// per-expert LRU bank, so the numbers below are steady-state bank traffic.  Then
// for n = 1..6 the test prefills n continuation tokens at the live absolute
// position and times prefill_text + sync_all.  The whole sweep runs twice: the
// second pass is warm, and the first is printed too rather than averaged in, so
// the cold and warm costs are distinguishable.
//
// Gate (warm pass, 2x A770, --moe-vram-mb 17500): n=3 >= 190 ms or n=5 >= 230 ms
// kills the MTP verify as a win.  The limits are printed, not asserted - this
// test reports the cost and exits 0.
//
// usage: test_q4_verify <model.gguf>
//
// Env:
//   TEST_LAYER_MAP      layer map (default 0-23:gpu.0,24-47:gpu.1)
//   TEST_MOE_VRAM_MB    expert bank budget (default 17500)
//   PF_KV_TYPE          default f16 (set here when unset; the run command sets it)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "engine.h"

using namespace si;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// `src` repeated and truncated to exactly `n` tokens.  Callers check src is
// non-empty first: repeating nothing would spin for ever.
static std::vector<int> repeat_to(const std::vector<int> & src, int n) {
    std::vector<int> out;
    out.reserve((size_t)n);
    while ((int)out.size() < n) {
        for (int id : src) {
            out.push_back(id);
            if ((int)out.size() == n) {
                break;
            }
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    const char * model_path =
        argc > 1 ? argv[1] : "/path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
    // The engine reads this once, in its constructor; the run command sets it
    // too, but a bare invocation of this test should still measure the
    // recommended f16 KV.
    if (!si::env::str("PF_KV_TYPE")) {
        setenv("PF_KV_TYPE", "f16", 1);
    }
    const char * lm = si::env::str("TEST_LAYER_MAP");
    const char * mv = si::env::str("TEST_MOE_VRAM_MB");
    try {
        engine_config ec;
        ec.model_path = model_path;
        ec.max_seq = 2048;
        ec.n_blocks = 64;
        ec.max_slots = 1; // the verify is a single-sequence, single-slot shape
        ec.layer_map = lm ? lm : "0-23:gpu.0,24-47:gpu.1";
        ec.moe_vram_mb = mv ? (size_t)strtoull(mv, nullptr, 10) : (size_t)17500;
        engine e(ec);

        const int NWARM = 96; // 3 full kMaxT rows: one mode-2 forward
        const int NCALL = 6;  // n = 1..6 drafted-token verify widths
        const int per_pass = NCALL * (NCALL + 1) / 2;

        const std::vector<int> para = e.tk.encode(
            "Quantized inference on an integrated GPU is memory bound: every weight byte is read once per token, so "
            "a decode step does one multiply-add per weight byte and the cache hierarchy cannot hide the streaming "
            "cost. A prefill reuses each weight across a tile of token rows instead, which raises the arithmetic "
            "intensity by the tile width and moves the bottleneck to the matrix units.",
            /*parse_special=*/true);
        if (para.empty()) {
            fprintf(stderr, "test_q4_verify: the warm paragraph tokenised to nothing\n");
            return 1;
        }
        const std::vector<int> prompt = repeat_to(para, NWARM);

        // prefill_chunk and q4_prefill_text index the token vector by ABSOLUTE
        // position (`toks[start + i]`), so the vector has to span every position
        // the two sweeps touch; indices below NWARM are the prompt and are never
        // read again.  The continuation is a fixed phrase, repeated to fit.
        const int span = NWARM + 2 * per_pass;
        std::vector<int> cont = prompt;
        const std::vector<int> phrase =
            e.tk.encode(" The verify costs one weight pass plus one extra row per drafted token.", true);
        if (phrase.empty()) {
            fprintf(stderr, "test_q4_verify: the continuation phrase tokenised to nothing\n");
            return 1;
        }
        while ((int)cont.size() < span) {
            cont.insert(cont.end(), phrase.begin(), phrase.end());
        }
        cont.resize((size_t)span);

        // Four blocks (128 positions) would stop covering the second sweep at
        // position 128; an out-of-range table entry reads block 0, which is
        // wrong KV rather than a crash and would perturb the expert routing the
        // timing depends on.  Size the table for every position both sweeps
        // touch, plus one spare block.
        const int nblk = (span + kBlockSize - 1) / kBlockSize + 1;
        e.reset_state();
        std::vector<int> blocks;
        blocks.reserve((size_t)nblk);
        for (int i = 0; i < nblk; i++) {
            const int b = e.alloc_block();
            if (b < 0) {
                fprintf(stderr, "test_q4_verify: out of KV blocks at %d of %d\n", i, nblk);
                return 1;
            }
            blocks.push_back(b);
        }
        e.set_table(0, blocks);

        const double tw0 = now_ms();
        e.prefill_text(prompt, 0, NWARM, 0);
        e.sync_all();
        fprintf(stderr, "[q4verify] warm prompt %d tokens (bank calibration) in %.1f ms\n", NWARM, now_ms() - tw0);

        // cold = first sight of each width, warm = the repeat.  Positions keep
        // advancing, so the warm sweep sees a longer KV and the same bank state.
        double cold[NCALL + 1] = {};
        double warm[NCALL + 1] = {};
        int pos = NWARM;
        for (int pass = 0; pass < 2; pass++) {
            double * row = pass == 0 ? cold : warm;
            for (int n = 1; n <= NCALL; n++) {
                const double t0 = now_ms();
                e.prefill_text(cont, 0, n, pos);
                e.sync_all();
                row[n] = now_ms() - t0;
                fprintf(stderr, "[q4verify] %s n=%d pos=%d ms=%.2f\n", pass == 0 ? "cold" : "warm", n, pos, row[n]);
                pos += n;
            }
        }

        int bad = 0;
        for (int n = 1; n <= NCALL; n++) {
            if (!std::isfinite(cold[n]) || cold[n] <= 0.0) {
                fprintf(stderr, "FAIL: cold n=%d is not a finite positive time\n", n);
                bad++;
            }
            if (!std::isfinite(warm[n]) || warm[n] <= 0.0) {
                fprintf(stderr, "FAIL: warm n=%d is not a finite positive time\n", n);
                bad++;
            }
        }

        printf("[q4verify] one row, n consecutive tokens, LM head included (2x A770 gate)\n");
        printf("[q4verify] n     ");
        for (int n = 1; n <= NCALL; n++) {
            printf("%8d", n);
        }
        printf("\n[q4verify] cold  ");
        for (int n = 1; n <= NCALL; n++) {
            printf("%8.2f", cold[n]);
        }
        printf("\n[q4verify] warm  ");
        for (int n = 1; n <= NCALL; n++) {
            printf("%8.2f", warm[n]);
        }
        printf("\n[q4verify] gate: warm n=3 %.2f ms (kill at >= 190), warm n=5 %.2f ms (kill at >= 230)\n", warm[3],
               warm[5]);

        if (bad) {
            fprintf(stderr, "test_q4_verify: %d timing check(s) failed\n", bad);
            return 1;
        }
        printf("test_q4_verify: all timings finite and positive\n");
        return 0;
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
