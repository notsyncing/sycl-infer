// qwen4exp 2-row batched decode: does it pay?
//
// The q4 branch already exists - decode_batch sets step_info's rows and calls
// q4_forward(0, n_rows, true) - but nothing measured it.  This test feeds the
// same teacher-forced token stream to both rows, once decoded one row at a
// time and once decoded as a 2-row batch, and compares the logits step by
// step.  Batching changes the fp accumulation order, so equality is not the
// oracle; a stale row is: a row the forward never computed does not MOVE
// between two steps, and it is not close to its own baseline.
//
// usage: test_q4_batch <model.gguf>
//
// Env:
//   TEST_LAYER_MAP      layer map (default 0-23:gpu.0,24-47:gpu.1)
//   TEST_MOE_VRAM_MB    expert bank budget (default 17500)
//   TEST_Q4_STREAM_N    teacher-forced tokens to feed (default 14; 2 warmup)
//   PF_KV_TYPE          default f16 (set here when unset)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// log-softmax in double: the logits are ~250k wide and a stable reduction is
// the difference between a KL and a -inf.
static void softmax_logp(const float * x, int n, std::vector<double> & logp) {
    double mx = -INFINITY;
    for (int i = 0; i < n; i++) {
        if ((double)x[i] > mx) {
            mx = (double)x[i];
        }
    }
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        s += std::exp((double)x[i] - mx);
    }
    const double lse = mx + std::log(s);
    for (int i = 0; i < n; i++) {
        logp[(size_t)i] = (double)x[i] - lse;
    }
}

static double kl_div(const std::vector<double> & a, const std::vector<double> & b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        s += std::exp(a[i]) * (a[i] - b[i]);
    }
    return s;
}

struct cmp_stats {
    int top1 = 0;
    double kl_mean = 0.0;
    double kl_max = 0.0;
    double max_abs = 0.0;
    int nonfinite = 0;
};

static cmp_stats compare_rows(const std::vector<float> & base, const std::vector<float> & bat, int nvoc, int first,
                              int last) {
    cmp_stats st;
    std::vector<double> la((size_t)nvoc), lb((size_t)nvoc);
    for (int s = first; s < last; s++) {
        const float * x = base.data() + (size_t)s * nvoc;
        const float * y = bat.data() + (size_t)s * nvoc;
        int ai = 0, bi = 0;
        double md = 0.0;
        for (int v = 0; v < nvoc; v++) {
            if (!std::isfinite(x[v])) {
                st.nonfinite++;
            }
            if (!std::isfinite(y[v])) {
                st.nonfinite++;
            }
            if (x[v] > x[ai]) {
                ai = v;
            }
            if (y[v] > y[bi]) {
                bi = v;
            }
            const double d = std::fabs((double)x[v] - (double)y[v]);
            if (d > md) {
                md = d;
            }
        }
        if (ai == bi) {
            st.top1++;
        }
        softmax_logp(x, nvoc, la);
        softmax_logp(y, nvoc, lb);
        const double k = kl_div(la, lb);
        st.kl_mean += k;
        st.kl_max = std::max(st.kl_max, k);
        st.max_abs = std::max(st.max_abs, md);
    }
    st.kl_mean /= (double)std::max(1, last - first);
    return st;
}

// Number of consecutive step pairs whose logits differ.  A row that the
// forward never wrote stays bit-identical for ever; a real row changes with
// the position and the KV.
static int moved_pairs(const std::vector<float> & logits, int nvoc, int first, int last) {
    int moved = 0;
    for (int s = first; s + 1 < last; s++) {
        const float * a = logits.data() + (size_t)s * nvoc;
        const float * b = logits.data() + (size_t)(s + 1) * nvoc;
        double l1 = 0.0;
        for (int v = 0; v < nvoc; v++) {
            l1 += std::fabs((double)a[v] - (double)b[v]);
        }
        if (l1 > 0.0) {
            moved++;
        }
    }
    return moved;
}

// Are the two rows distinct at the same step?  A row that is a copy of the
// other (or a row the forward never wrote, if the other row is also stale)
// fails this.
static int distinct_steps(const std::vector<float> & a, const std::vector<float> & b, int nvoc, int first, int last) {
    int n = 0;
    for (int s = first; s < last; s++) {
        const float * x = a.data() + (size_t)s * nvoc;
        const float * y = b.data() + (size_t)s * nvoc;
        bool diff = false;
        for (int v = 0; v < nvoc && !diff; v++) {
            diff = x[v] != y[v];
        }
        if (diff) {
            n++;
        }
    }
    return n;
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
        ec.n_blocks = 512;
        ec.max_slots = 2;
        ec.layer_map = lm ? lm : "0-23:gpu.0,24-47:gpu.1";
        ec.moe_vram_mb = mv ? (size_t)strtoull(mv, nullptr, 10) : (size_t)17500;
        engine e(ec);

        const int nvoc = e.m.hp.n_vocab;
        const std::vector<int> p0 = e.tk.encode("The capital of France is", /*parse_special=*/true);
        const std::vector<int> p1 = e.tk.encode("The TCP/IP protocol is", /*parse_special=*/true);
        CHECK(!p0.empty());
        CHECK(!p1.empty());

        // One teacher-forced continuation, truncated.  Both the baseline and
        // the batch feed exactly these ids at exactly these positions, so the
        // only difference between the two runs is how many rows the forward
        // computes at once.
        std::vector<int> stream =
            e.tk.encode(" and the answer is a number that we can use to check the result of the calculation.",
                        /*parse_special=*/true);
        const char * nse = si::env::str("TEST_Q4_STREAM_N");
        int N = nse ? atoi(nse) : 14;
        if (N > (int)stream.size()) {
            N = (int)stream.size();
        }
        if (N < 4) {
            fprintf(stderr, "test_q4_batch: the continuation tokenised to %zu tokens; need at least 4\n",
                    stream.size());
            return 1;
        }
        stream.resize((size_t)N);
        const int WARM = 2; // steps not timed, in every run
        const int measured = N - WARM;

        // Four blocks each = 128 positions, enough for either prompt plus the
        // stream.  Both slots must have their own table row.
        std::vector<int> blocks0, blocks1;
        for (int i = 0; i < 4; i++) {
            const int b = e.alloc_block();
            CHECK(b >= 0);
            if (b >= 0) {
                blocks0.push_back(b);
            }
        }
        for (int i = 0; i < 4; i++) {
            const int b = e.alloc_block();
            CHECK(b >= 0);
            if (b >= 0) {
                blocks1.push_back(b);
            }
        }
        CHECK(blocks0.size() == 4 && blocks1.size() == 4);
        if (blocks0.size() != 4 || blocks1.size() != 4) {
            return 1;
        }
        auto arm_tables = [&] {
            e.set_table(0, blocks0);
            e.set_table(1, blocks1);
        };
        auto prefill_both = [&] {
            e.reset_state();
            arm_tables();
            e.prefill_text(p0, 0, (int)p0.size(), 0);
            e.prefill_text(p1, 1, (int)p1.size(), 0);
        };

        fprintf(stderr, "[q4batch] n_vocab=%d prompts=%zu/%zu stream=%d warm=%d measured=%d\n", nvoc, p0.size(),
                p1.size(), N, WARM, measured);

        // A preliminary 2-row round warms the expert LRU before either measured
        // run, so the baseline is not paying the cold-bank cost the batch would
        // not.  It uses the first tokens of the same stream and is discarded;
        // the state is reset and both prompts re-prefilled below.
        fprintf(stderr, "[q4batch] phase=warmup start\n");
        prefill_both();
        for (int s = 0; s < N && s < 8; s++) {
            int32_t toks[2] = {stream[(size_t)s], stream[(size_t)s]};
            int32_t poss[2] = {(int32_t)p0.size() + s, (int32_t)p1.size() + s};
            int32_t slots[2] = {0, 1};
            e.decode_batch(toks, poss, slots, 2);
        }
        fprintf(stderr, "[q4batch] phase=warmup end\n");

        // ---- baseline: each sequence alone, one row per forward ----
        prefill_both();
        std::vector<float> base0((size_t)N * nvoc), base1((size_t)N * nvoc);
        double t0 = 0.0, t1 = 0.0;
        fprintf(stderr, "[q4batch] phase=baseline_row0 start\n");
        for (int s = 0; s < N; s++) {
            int32_t tok = stream[(size_t)s];
            int32_t pos = (int32_t)p0.size() + s;
            int32_t slot = 0;
            const double a = now_ms();
            e.decode_batch(&tok, &pos, &slot, 1);
            e.fetch_logits(0, base0.data() + (size_t)s * nvoc);
            const double b = now_ms();
            if (s >= WARM) {
                t0 += b - a;
            }
        }
        fprintf(stderr, "[q4batch] phase=baseline_row0 end\n");

        fprintf(stderr, "[q4batch] phase=baseline_row1 start\n");
        for (int s = 0; s < N; s++) {
            int32_t tok = stream[(size_t)s];
            int32_t pos = (int32_t)p1.size() + s;
            int32_t slot = 1;
            const double a = now_ms();
            e.decode_batch(&tok, &pos, &slot, 1);
            e.fetch_logits(0, base1.data() + (size_t)s * nvoc);
            const double b = now_ms();
            if (s >= WARM) {
                t1 += b - a;
            }
        }
        fprintf(stderr, "[q4batch] phase=baseline_row1 end\n");

        // ---- batch: both rows in one forward, same stream, same positions ----
        prefill_both();
        std::vector<float> bat0((size_t)N * nvoc), bat1((size_t)N * nvoc);
        double t2 = 0.0;
        fprintf(stderr, "[q4batch] phase=batch start\n");
        for (int s = 0; s < N; s++) {
            int32_t toks[2] = {stream[(size_t)s], stream[(size_t)s]};
            int32_t poss[2] = {(int32_t)p0.size() + s, (int32_t)p1.size() + s};
            int32_t slots[2] = {0, 1};
            const double a = now_ms();
            e.decode_batch(toks, poss, slots, 2);
            e.fetch_logits(0, bat0.data() + (size_t)s * nvoc);
            e.fetch_logits(1, bat1.data() + (size_t)s * nvoc);
            const double b = now_ms();
            if (s >= WARM) {
                t2 += b - a;
            }
        }
        fprintf(stderr, "[q4batch] phase=batch end\n");

        // ---- comparison ----
        const cmp_stats s0 = compare_rows(base0, bat0, nvoc, WARM, N);
        const cmp_stats s1 = compare_rows(base1, bat1, nvoc, WARM, N);
        // Crossed comparison: row 0's batch logits against row 1's baseline.
        // The matched pair must be the closer one, or the rows are swapped.
        const cmp_stats x0 = compare_rows(base1, bat0, nvoc, WARM, N);
        const cmp_stats x1 = compare_rows(base0, bat1, nvoc, WARM, N);

        const int mv_base0 = moved_pairs(base0, nvoc, WARM, N);
        const int mv_base1 = moved_pairs(base1, nvoc, WARM, N);
        const int mv_bat0 = moved_pairs(bat0, nvoc, WARM, N);
        const int mv_bat1 = moved_pairs(bat1, nvoc, WARM, N);
        const int dist = distinct_steps(bat0, bat1, nvoc, WARM, N);

        const double base_steps_ms = (t0 + t1) / (double)measured;
        const double batch_steps_ms = t2 / (double)measured;
        const double base_tok_s = (2.0 * measured) / ((t0 + t1) / 1000.0);
        const double batch_tok_s = (2.0 * measured) / (t2 / 1000.0);
        const double speedup = (t0 + t1) / std::max(1e-9, t2);

        printf("[q4batch] baseline row0 %.2f ms/step, row1 %.2f ms/step | combined %.2f ms/step, "
               "%.2f tok/s (%.3f s for %d tokens)\n",
               t0 / measured, t1 / measured, base_steps_ms, base_tok_s, (t0 + t1) / 1000.0, 2 * measured);
        printf("[q4batch] batch    2 rows  %.2f ms/step, %.2f tok/s (%.3f s for %d tokens), speedup %.3fx\n",
               batch_steps_ms, batch_tok_s, t2 / 1000.0, 2 * measured, speedup);
        printf("[q4batch] row0 vs its baseline: top1 %d/%d, KL mean %.4g max %.4g, max|dlogit| %.4g, nonfinite %d\n",
               s0.top1, measured, s0.kl_mean, s0.kl_max, s0.max_abs, s0.nonfinite);
        printf("[q4batch] row1 vs its baseline: top1 %d/%d, KL mean %.4g max %.4g, max|dlogit| %.4g, nonfinite %d\n",
               s1.top1, measured, s1.kl_mean, s1.kl_max, s1.max_abs, s1.nonfinite);
        printf("[q4batch] crossed row0-vs-base1 KL mean %.4g, row1-vs-base0 KL mean %.4g\n", x0.kl_mean, x1.kl_mean);
        printf("[q4batch] movement: baseline %d/%d + %d/%d, batch %d/%d + %d/%d step pairs; "
               "batch rows distinct in %d/%d steps\n",
               mv_base0, measured - 1, mv_base1, measured - 1, mv_bat0, measured - 1, mv_bat1, measured - 1, dist,
               measured);

        // Hard checks: the batched rows must be real.  No NaN, each row moves
        // between two steps, and the two rows are not the same vector (a row
        // that was never written would duplicate the other, or be constant).
        CHECK(s0.nonfinite == 0);
        CHECK(s1.nonfinite == 0);
        CHECK(mv_bat0 >= 1);
        CHECK(mv_bat1 >= 1);
        CHECK(dist == measured);
        // 2/2 rows finite is exactly the two nonfinite checks above.  A top-1
        // flip is allowed (batching changes the fp accumulation order), and
        // the KL is reported rather than asserted for the same reason; a stale
        // row shows up as no movement, not as a large KL alone.

        if (g_fail) {
            fprintf(stderr, "test_q4_batch: %d check(s) failed\n", g_fail);
            return 1;
        }
        printf("test_q4_batch: all checks OK\n");
        return 0;
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
