// Unit tests for the DFlash2 block drafter's device kernels, against host
// references.  These exist because of the two bug classes that actually cost
// time in this subsystem, neither of which an end-to-end check localizes:
//
//   * the conv's two coefficient tensors have MIRRORED axes (base is
//     channel-innermost, dynamic is token-innermost).  When only one was right
//     the anchor row - which only ever uses tap 0 - still matched the reference
//     at cos 0.994, so an aggregate accuracy check passed and only the mask rows
//     were wrong.  A per-row check catches it immediately.
//   * the top-k had its per-lane lists in SLM with a 16-way bank conflict, and
//     was rewritten into a two-stage slice+merge.  A merge that reorders ties,
//     drops a candidate, or reads a stale partial changes the SELECTOR's
//     lattice walk, which changes the emitted token stream with no local error.
//
// Ties resolve to the LOWEST token id in both paths, and the ggml (1,0) swap of
// the top two entries is part of the model's definition (the lattice walk reads
// candidate k as the successor of candidate k), so the reference reproduces both.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <random>
#include <vector>

#include "kernels.h"

using namespace si;

static int failures = 0;
static int checks = 0;

static void ok(bool cond, const char * what) {
    checks++;
    if (!cond) {
        failures++;
        printf("  FAIL %s\n", what);
    }
}

// ---------------------------------------------------------------------------
// top-k reference
//
// Mirrors what a single-pass reduction produces: K rounds of "take the global
// max, emit it, remove it", ties going to the lowest id.  Then the ggml (1,0)
// swap.  Independent of how the device splits the row.
static void topk_ref(const float * row, int n, int K, std::vector<int32_t> & ids, std::vector<float> & vals) {
    std::vector<int32_t> left(n);
    for (int i = 0; i < n; i++) {
        left[(size_t)i] = i;
    }
    ids.assign((size_t)K, 0);
    vals.assign((size_t)K, 0.f);
    for (int k = 0; k < K; k++) {
        int best = -1;
        if (left.empty()) {
            break;  // n < K: the device writes INT_MAX sentinels here
        }
        for (size_t j = 0; j < left.size(); j++) {
            const int32_t i = left[j];
            if (best < 0) {
                best = i;
                continue;
            }
            // strictly greater, or equal with a lower id
            if (row[(size_t)i] > row[(size_t)best] ||
                (row[(size_t)i] == row[(size_t)best] && i < best)) {
                best = i;
            }
        }
        ids[(size_t)k] = best;
        vals[(size_t)k] = row[(size_t)best];
        left.erase(std::find(left.begin(), left.end(), best));
    }
    if (K > 1) {
        std::swap(ids[0], ids[1]);
        std::swap(vals[0], vals[1]);
    }
}

// n must be < K in some cases: a lane list can hold fewer than K entries and the
// device writes kNone (-1 index) sentinels.  The reference must agree on those.
static void test_topk(sycl::queue & q, const char * label, int n, int K, int M, int S, bool dup) {
    std::mt19937 rng(1234u + (unsigned)n * 7u + (unsigned)K * 13u + (unsigned)S * 29u + (dup ? 1u : 0u));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> h((size_t)M * n);
    for (size_t i = 0; i < h.size(); i++) {
        h[i] = nd(rng);
    }
    if (dup) {
        // force many exact ties so the lowest-id rule is actually exercised
        for (size_t i = 0; i < h.size(); i++) {
            h[i] = (float)((int)(h[i] * 8.0f)) / 8.0f;
        }
    }

    float * d_logits = sycl::malloc_device<float>(h.size(), q);
    int32_t * d_ids = sycl::malloc_device<int32_t>((size_t)M * K, q);
    float * d_vals = sycl::malloc_device<float>((size_t)M * K, q);
    int32_t * d_pids = nullptr;
    float * d_pvals = nullptr;
    if (S > 1) {
        d_pids = sycl::malloc_device<int32_t>((size_t)M * S * K, q);
        d_pvals = sycl::malloc_device<float>((size_t)M * S * K, q);
    }
    q.memcpy(d_logits, h.data(), h.size() * 4).wait();

    // sentinel: a rejected K must leave the caller's buffers untouched, and
    // malloc_device does not zero, so the expectation has to be written in first
    {
        const int32_t sent_i = -12345;
        const float sent_f = -12345.f;
        std::vector<int32_t> si((size_t)M * K, sent_i);
        std::vector<float> sv((size_t)M * K, sent_f);
        q.memcpy(d_ids, si.data(), si.size() * 4);
        q.memcpy(d_vals, sv.data(), sv.size() * 4).wait();
    }
    df_topk_launch(q, d_logits, n, d_ids, d_vals, M, K, d_pids, d_pvals, S);
    q.wait();

    std::vector<int32_t> ids((size_t)M * K);
    std::vector<float> vals((size_t)M * K);
    q.memcpy(ids.data(), d_ids, ids.size() * 4);
    q.memcpy(vals.data(), d_vals, vals.size() * 4).wait();

    if (K > 30) {
        // documented no-op: the launcher must ignore it rather than fail to launch
        std::vector<int32_t> z((size_t)M * K);
        std::vector<float> zv((size_t)M * K);
        q.memcpy(z.data(), d_ids, z.size() * 4);
        q.memcpy(zv.data(), d_vals, zv.size() * 4).wait();
        bool untouched = true;
        for (size_t i = 0; i < z.size(); i++) {
            if (z[i] != -12345 || zv[i] != -12345.f) untouched = false;
        }
        printf("  topk %-34s n=%-7d K=%-3d M=%d S=%-3d -> %s\n", label, n, K, M, S,
               untouched ? "OK (ignored)" : "FAIL (wrote output)");
        ok(untouched, "K > 30 must be ignored, not launched");
        checks++;
        q.wait();
        sycl::free(d_logits, q);
        sycl::free(d_ids, q);
        sycl::free(d_vals, q);
        if (d_pids) sycl::free(d_pids, q);
        if (d_pvals) sycl::free(d_pvals, q);
        return;
    }
    int bad_id = 0, bad_val = 0;
    for (int r = 0; r < M; r++) {
        std::vector<int32_t> rid;
        std::vector<float> rval;
        topk_ref(h.data() + (size_t)r * n, n, K, rid, rval);
        for (int k = 0; k < K; k++) {
            if (ids[(size_t)r * K + k] != rid[(size_t)k]) bad_id++;
            if (std::fabs(vals[(size_t)r * K + k] - rval[(size_t)k]) > 1e-6f) bad_val++;
        }
    }
    printf("  topk %-34s n=%-7d K=%-3d M=%d S=%-3d%s -> %s\n", label, n, K, M, S, dup ? " dup" : "",
           (bad_id || bad_val) ? "MISMATCH" : "OK");
    if (bad_id || bad_val) {
        failures++;
        printf("    ids differ %d, vals differ %d (of %d)\n", bad_id, bad_val, M * K);
        std::vector<int32_t> rid;
        std::vector<float> rval;
        topk_ref(h.data(), n, K, rid, rval);
        printf("    row0 dev:");
        for (int k = 0; k < K; k++) printf(" %d", ids[(size_t)k]);
        printf("\n    row0 ref:");
        for (int k = 0; k < K; k++) printf(" %d", rid[(size_t)k]);
        printf("\n");
    }
    checks++;

    // the one-slice and multi-slice paths must agree with each other exactly:
    // that is the invariant the two-stage rewrite has to preserve
    if (S > 1 && n <= 65536) {
        int32_t * o_ids = sycl::malloc_device<int32_t>((size_t)M * K, q);
        float * o_vals = sycl::malloc_device<float>((size_t)M * K, q);
        df_topk_launch(q, d_logits, n, o_ids, o_vals, M, K, nullptr, nullptr, 1);
        q.wait();
        std::vector<int32_t> o((size_t)M * K);
        std::vector<float> ov((size_t)M * K);
        q.memcpy(o.data(), o_ids, o.size() * 4);
        q.memcpy(ov.data(), o_vals, ov.size() * 4).wait();
        const bool same = (o == ids) && (ov == vals);
        printf("    sliced == single-pass: %s\n", same ? "OK" : "MISMATCH");
        ok(same, "sliced topk must equal the single-pass path");
        q.wait();
        sycl::free(o_ids, q);
        sycl::free(o_vals, q);
    }

    q.wait();
    sycl::free(d_logits, q);
    sycl::free(d_ids, q);
    sycl::free(d_vals, q);
    if (d_pids) sycl::free(d_pids, q);
    if (d_pvals) sycl::free(d_pvals, q);
}

// ---------------------------------------------------------------------------
// conv reference, with the mirrored axes spelled out
//
//   base[side][group_size][n_groups][conv_k]  -> (c,t) = c + width*t
//   dyn [n_groups][conv_k][2][n_rows]         -> (g,t,side) =
//                                                  g + n_groups*t + side*n_groups*conv_k
//
// The two are NOT transposes of each other; treating dyn as channel-innermost
// (or base as token-innermost) is the bug this test exists to catch.  Getting
// base wrong is invisible on the anchor row (tap 0 only), so every case here
// checks ALL taps.
static void conv_ref(const std::vector<float> & x, const std::vector<float> & dyn,
                     const std::vector<float> & base, const std::vector<float> & resid, int n_rows, int width,
                     int conv_k, int conv_group, int dyn_proj, int side, std::vector<float> & out) {
    const int n_groups = width / conv_group;
    const bool has_resid = !resid.empty();
    out.assign((size_t)n_rows * width, 0.f);
    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < width; c++) {
            const int g = c / conv_group;
            float acc = 0.f;
            // row r's window: tap 0 is row r, tap t is row r-t.  Row 0 therefore
            // uses only tap 0, which is exactly why a wrong base axis stayed
            // invisible on the anchor row.
            for (int t = 0; t < conv_k; t++) {
                const int sr = r - t;
                if (sr < 0) {
                    continue;
                }
                const float b = base[((size_t)side * width * conv_k) + (size_t)c + (size_t)width * t];
                const float d = dyn[(size_t)r * dyn_proj + g + (size_t)n_groups * t +
                                    (size_t)side * n_groups * conv_k];
                // additive fusion: w = base + dyn, then w * x.  Multiplying the two
                // coefficients instead is a plausible-looking reference that fails
                // every element.
                acc += (b + d) * x[(size_t)sr * width + c];
            }
            if (has_resid) {
                acc += resid[(size_t)r * width + c];
            }
            out[(size_t)r * width + c] = acc;
        }
    }
}

int main() {
    try {
        sycl::queue q{sycl::gpu_selector_v};
        printf("dflash kernels: topk + conv vs host reference\n");

        // --- top-k ---
        // S=1 is the original single-workgroup-per-row path; S>1 is the
        // two-stage slice+merge.  They must agree with the reference and with
        // each other for every shape.
        test_topk(q, "real n_vocab", 248320, 16, 6, 8, false);
        test_topk(q, "heavy ties", 65536, 16, 6, 8, true);
        test_topk(q, "single-pass", 65536, 16, 6, 1, false);
        test_topk(q, "many slices", 65536, 16, 6, 64, false);
        test_topk(q, "slices > elements", 4096, 16, 4, 128, false);
        test_topk(q, "small n", 512, 16, 3, 8, false);
        test_topk(q, "n barely above K", 33, 16, 3, 8, false);
        test_topk(q, "K=1", 4096, 1, 4, 8, false);
        test_topk(q, "K=30 (SLM max)", 8192, 30, 2, 8, true);
        test_topk(q, "K=16 (model)", 8192, 16, 6, 8, false);
        // K=31 and K=32 must be REJECTED, not launched.  The budget is
        // 8*(256*K+1) + 8*256 <= 65536, i.e. K <= 30; the original guard said
        // K <= 32 with WG*K+2 lists and no scratch in the budget, so K=31 (65544
        // bytes) and K=32 (65552) both failed to LAUNCH and killed the process with
        // UR_RESULT_ERROR_OUT_OF_RESOURCES - a config error reported as a resource
        // error, which is why it needed a test to find.
        test_topk(q, "K=31 (rejected)", 8192, 31, 2, 8, false);
        test_topk(q, "K=32 (rejected)", 8192, 32, 2, 8, false);
        test_topk(q, "M=1", 20000, 16, 1, 8, false);
        test_topk(q, "prime n, ties", 9973, 16, 5, 7, true);

        // --- conv ---
        struct cshape {
            int width, conv_k, conv_group, n_rows, dyn_proj;
        };
        const cshape cases[] = {
            {5120, 2, 16, 6, 1280},  // the real 27B-DFlash2 geometry
            {5120, 2, 16, 1, 1280},  // the anchor row alone: the case that hid
                                     // the base-axis bug (tap 0 only)
            {64, 3, 8, 5, 0},        // dyn_proj derived below
            {32, 4, 32, 3, 0},       // conv_group == width: one group
            {16, 2, 16, 4, 0},       // same
            {96, 5, 1, 2, 0},        // conv_group 1: depthwise, widest kernel
        };
        int cc = 0;
        for (const cshape & cs : cases) {
            const int n_groups = cs.width / cs.conv_group;
            const int dyn_proj = cs.dyn_proj ? cs.dyn_proj : 2 * cs.conv_k * n_groups;
            const size_t xn = (size_t)cs.n_rows * cs.width;
            const size_t bn = (size_t)2 * cs.width * cs.conv_k;
            const size_t dn = (size_t)cs.n_rows * dyn_proj;

            std::mt19937 rng(99u + (unsigned)cc);
            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<float> hx(xn), hbase(bn), hdyn(dn), hres(xn);
            for (auto & v : hx) v = nd(rng);
            for (auto & v : hbase) v = nd(rng);
            for (auto & v : hdyn) v = nd(rng);
            for (auto & v : hres) v = nd(rng);
            ++cc;

            float * d_x = sycl::malloc_device<float>(xn, q);
            float * d_base = sycl::malloc_device<float>(bn, q);
            float * d_dyn = sycl::malloc_device<float>(dn, q);
            float * d_res = sycl::malloc_device<float>(xn, q);
            float * d_out = sycl::malloc_device<float>(xn, q);
            q.memcpy(d_x, hx.data(), xn * 4);
            q.memcpy(d_base, hbase.data(), bn * 4);
            q.memcpy(d_dyn, hdyn.data(), dn * 4);
            q.memcpy(d_res, hres.data(), xn * 4).wait();

            // both sides, with and without the residual add
            for (int side = 0; side < 2; side++) {
                for (int use_res = 0; use_res < 2; use_res++) {
                    std::vector<float> ref;
                    conv_ref(hx, hdyn, hbase, use_res ? hres : std::vector<float>(), cs.n_rows, cs.width,
                             cs.conv_k, cs.conv_group, dyn_proj, side, ref);
                    q.memset(d_out, 0, xn * 4);
                    df_conv_launch(q, d_x, d_dyn, d_base, d_out, use_res ? d_res : nullptr, cs.n_rows, cs.width,
                                   cs.width, cs.conv_k, cs.conv_group, dyn_proj, side);
                    q.wait();
                    std::vector<float> got(xn);
                    q.memcpy(got.data(), d_out, xn * 4).wait();
                    double worst = 0;
                    int nbad = 0;
                    for (size_t i = 0; i < xn; i++) {
                        const double e = std::fabs((double)got[i] - ref[i]) /
                                         (std::fabs((double)ref[i]) + 1e-3);
                        if (e > worst) worst = e;
                        if (e > 2e-2) nbad++;
                    }
                    const char * tag = "conv";
                    printf("  %s w=%-5d k=%d g=%-3d rows=%d side=%d res=%d -> %s (worst rel %.2e%s)\n", tag,
                           cs.width, cs.conv_k, cs.conv_group, cs.n_rows, side, use_res,
                           nbad ? "MISMATCH" : "OK", worst, nbad ? "" : "");
                    checks++;
                    if (nbad) {
                        failures++;
                        printf("    %d/%zu elements off\n", nbad, xn);
                    }
                }
            }

            sycl::free(d_x, q);
            sycl::free(d_base, q);
            sycl::free(d_dyn, q);
            sycl::free(d_res, q);
            sycl::free(d_out, q);
        }

        printf("dflash kernels: %d checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}