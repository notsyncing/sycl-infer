// yarn: the YaRN per-pair frequency table (src/common/yarn.h).
//
// What is worth pinning here, in order of how quietly it could be wrong:
//
//  1. **factor == 1 reproduces the plain table bit for bit.**  The default-off
//     path does not use a table at all (the kernels keep the original exp2), so
//     what has to hold is that *enabling* the feature cannot silently perturb the
//     frequencies - a factor-1 table must be the plain one, not a near miss.
//  2. **The ramp's two anchors.**  Below the first anchor every pair keeps the
//     trained frequency *bit for bit*; above the second every pair is stretched
//     by exactly 1/factor.  Both are asserted against the ramp anchors
//     themselves, not "roughly the first N pairs".
//  3. **Monotonicity.**  A valid rope needs non-increasing frequencies.  A mask
//     blended the other way round yields a table that is not a rope at all, and
//     no end-to-end test would call that out - it just produces fluent garbage.
//  4. **mscale**, including the clamp at 1 and the explicit override.
//
// No model and no GPU: the table is a host function of (orig_ctx, factor, base,
// n_rot), which is what makes it cheap to test exhaustively.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/rope.h"
#include "common/yarn.h"

static int failures = 0;

static void check(bool ok, const char * what) {
    printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) {
        failures++;
    }
}

static bool bit_eq(float a, float b) {
    return memcmp(&a, &b, sizeof(a)) == 0;
}

// The two shipped geometries, read out of the GGUFs:
//   0.8B: rope.dimension_count 128, rope.freq_base 1e6
//   27B : rope.dimension_count  64, rope.freq_base 1e7
// The ramp's behaviour is a function of (n_rot, base, trained ctx) and not of the
// model, so both are worth pinning - in particular they have a different number of
// pairs (64 vs 32), which changes where the correction indices land.
static int run_case(const char * name, int n_rot, float base) {
    const int n_pairs = n_rot / 2;
    printf("\n== %s: n_rot=%d (%d pairs) base=%.0f ==\n", name, n_rot, n_pairs, base);

    // The plain table and the ramped one both come out of the *same* function, so
    // comparing them isolates the ramp from the seed loop's exp2 (see check 4).
    si::yarn_params off;
    off.on = false;
    float ms_off = 0.0f;
    const std::vector<float> t_off = si::yarn_build(off, n_rot, base, &ms_off);

    // ---- 1. off and factor-1 paths -------------------------------------------
    check(t_off.size() == (size_t)n_pairs, "the plain table has n_rot/2 entries");
    check(ms_off == 1.0f, "yarn.on=false gives mscale 1");
    {
        si::yarn_params p;
        p.on = true;
        p.factor = 1.0f;
        p.orig_ctx = 32768;
        float ms = 0.0f;
        const std::vector<float> t = si::yarn_build(p, n_rot, base, &ms);
        bool exact = ms == 1.0f && t.size() == t_off.size();
        for (int i = 0; i < n_pairs && exact; i++) {
            exact = bit_eq(t[(size_t)i], t_off[(size_t)i]);
        }
        check(exact, "factor=1 is bit-identical to the plain table (ramp skipped)");
    }
    {
        si::yarn_params p;
        p.on = true;
        p.factor = 0.5f; // shrink, not extend: must be a no-op, not a reversal
        p.orig_ctx = 32768;
        float ms = 0.0f;
        const std::vector<float> t = si::yarn_build(p, n_rot, base, &ms);
        bool exact = ms == 1.0f;
        for (int i = 0; i < n_pairs && exact; i++) {
            exact = bit_eq(t[(size_t)i], t_off[(size_t)i]);
        }
        check(exact, "factor<1 (shrink) is a no-op, not a reversal");
    }

    // ---- 2. the 27B at factor 32 ---------------------------------------------
    si::yarn_params p;
    p.on = true;
    p.factor = 32.0f;
    p.orig_ctx = 32768;
    float ms = 0.0f;
    const std::vector<float> t = si::yarn_build(p, n_rot, base, &ms);
    check(t.size() == (size_t)n_pairs, "the ramped table has n_rot/2 entries");

    // Recompute the anchors so the checks below are exact rather than "roughly
    // the first N pairs".  Same arithmetic as yarn_build(), deliberately.
    const float dim = (float)n_rot;
    float lo = si::yarn_correction_dim(p.beta_fast, dim, base, (float)p.orig_ctx);
    float hi = si::yarn_correction_dim(p.beta_slow, dim, base, (float)p.orig_ctx);
    if (lo > hi) {
        std::swap(lo, hi);
    }
    const float delta = (hi - lo) / dim;
    lo += delta;
    hi += delta;
    printf("  (ramp anchors: lo=%.2f hi=%.2f, of %d pairs; beta %.0f/%.0f)\n", lo, hi, n_pairs, p.beta_fast,
           p.beta_slow);

    {
        bool untouched_exact = true;
        int n_untouched = 0;
        for (int i = 0; i < n_pairs; i++) {
            if ((float)i <= lo) {
                n_untouched++;
                if (!bit_eq(t[(size_t)i], t_off[(size_t)i])) {
                    untouched_exact = false;
                    printf("    i=%d ramped=%.9g plain=%.9g\n", i, t[(size_t)i], t_off[(size_t)i]);
                }
            }
        }
        check(n_untouched > 10, "the ramp starts well past the high-frequency head");
        check(untouched_exact, "every pair below the first anchor is bit-identical to plain");
    }
    {
        bool stretched = true;
        int n_str = 0;
        for (int i = 0; i < n_pairs; i++) {
            if ((float)i >= hi) {
                n_str++;
                const float want = t_off[(size_t)i] / p.factor;
                if (std::fabs(t[(size_t)i] - want) > std::fabs(want) * 1e-6f) {
                    stretched = false;
                    printf("    i=%d got=%.9g want=%.9g\n", i, t[(size_t)i], want);
                }
            }
        }
        check(n_str > 10, "the stretched tail is substantial, not a couple of pairs");
        check(stretched, "every pair past the second anchor is stretched by exactly 1/factor");
    }
    {
        bool mono = true;
        for (int i = 1; i < n_pairs; i++) {
            if (t[(size_t)i] > t[(size_t)i - 1]) {
                mono = false;
                printf("    i=%d %.9g > %.9g\n", i, t[(size_t)i], t[(size_t)i - 1]);
            }
        }
        check(mono, "frequency is non-increasing in the pair index");
    }
    {
        bool in_range = true;
        for (int i = 0; i < n_pairs; i++) {
            const float pl = t_off[(size_t)i];
            if (t[(size_t)i] > pl * 1.00001f || t[(size_t)i] < pl / 32.0f * 0.99999f) {
                in_range = false;
                printf("    i=%d table=%.9g plain=%.9g\n", i, t[(size_t)i], pl);
            }
        }
        check(in_range, "every frequency lies within [plain/factor, plain]");
    }

    // ---- 3. mscale ------------------------------------------------------------
    const float want_ms = 1.0f + 0.1f * std::log10(32.0f);
    check(std::fabs(ms - want_ms) < 1e-6f, "mscale defaults to 1 + 0.1*log10(factor)");
    p.attn_factor = 1.25f;
    float ms2 = 0.0f;
    si::yarn_build(p, n_rot, base, &ms2);
    check(std::fabs(ms2 - 1.25f) < 1e-6f, "an explicit attn_factor overrides the derivation");
    si::yarn_params q = p;
    q.attn_factor = 0.5f;
    q.factor = 2.0f;
    float ms3 = 0.0f;
    si::yarn_build(q, n_rot, base, &ms3);
    check(ms3 == 1.0f, "a sub-1 attn_factor is clamped to 1 (never shrink the logits)");

    // ---- 4. the seed tracks rope_theta to within an ulp -----------------------
    // Not bit-exact by construction: the seed loop is vectorizable and an inlined
    // vector exp2 may differ from libm's exp2f by an ulp.  Irrelevant to the
    // feature (it is a frequency, not an accumulation) but it is why check 1
    // compares against the plain *table* and this one only bounds the error.
    {
        double worst = 0.0;
        for (int i = 0; i < n_pairs; i++) {
            const float pl = si::rope_theta(1.0f, i, n_rot, std::log2(base));
            const float rel = std::fabs((double)t_off[(size_t)i] - (double)pl) / (double)pl;
            worst = rel > worst ? rel : worst;
        }
        printf("  (worst relative deviation of the plain table from rope_theta: %.3g)\n", worst);
        check(worst < 1e-6, "the plain table tracks rope_theta's frequency to <1e-6 relative");
    }

    // ---- 5. degenerate inputs --------------------------------------------------
    float ms5 = -1.0f;
    si::yarn_build(off, 0, base, &ms5);
    check(ms5 == 1.0f, "n_rot=0 is handled (mscale 1)");
    si::yarn_params small;
    small.on = true;
    small.factor = 32.0f;
    small.orig_ctx = 4096; // far below the longest wavelength
    float ms6 = 0.0f;
    const std::vector<float> t6 = si::yarn_build(small, n_rot, base, &ms6);
    bool finite = t6.size() == (size_t)n_pairs;
    for (int i = 0; i < n_pairs && finite; i++) {
        finite = std::isfinite(t6[(size_t)i]) && t6[(size_t)i] > 0.0f;
    }
    check(finite, "orig_ctx below every wavelength still yields finite positive frequencies");

    // A larger factor on the same window must stretch at least as many pairs.
    {
        float m1 = 0.0f, m2 = 0.0f;
        si::yarn_params a = p;
        a.attn_factor = 0.0f;
        a.factor = 4.0f;
        si::yarn_params b = a;
        b.factor = 32.0f;
        const std::vector<float> ta = si::yarn_build(a, n_rot, base, &m1);
        const std::vector<float> tb = si::yarn_build(b, n_rot, base, &m2);
        bool ge = true;
        for (int i = 0; i < n_pairs; i++) {
            if (tb[(size_t)i] > ta[(size_t)i] * 1.000001f) {
                ge = false;
            }
        }
        check(ge, "a larger factor never raises a frequency (monotone in factor)");
        check(m2 > m1, "mscale grows with the factor");
    }

    return 0;
}

int main() {
    run_case("0.8B", 128, 1000000.0f);
    run_case("27B", 64, 10000000.0f);
    printf(failures ? "\n%d check(s) FAILED\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}