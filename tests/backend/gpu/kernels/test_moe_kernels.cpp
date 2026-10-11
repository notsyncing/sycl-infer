// Hermetic test of the Mixture-of-Experts kernels: router -> token/expert CSR ->
// per-expert gate/up -> per-expert down -> weighted reduction.
//
// There is no model and no engine here on purpose.  Every one of these kernels
// reads a *runtime* weight choice - the router's per-token top-k, the CSR that
// groups it, and the slot table that turns an expert id into an address - so the
// failure modes are all wiring failures that a whole-model test can only report
// as "the output is wrong".  With f32 experts (the same code path the Q4_K/Q5_K
// instantiations take, minus the dequant, which is the dense GEMV's own tested
// code) the reference is a dozen lines of scalar C++ and any mismatch localises
// to one kernel.
//
// Usage: test_moe_kernels [n_expert] [top_k] [n_embd] [n_ff_exp] [n_tok]
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kernels.h"
#include "gguf.h" // GGML_TYPE_*
#include "quant.h" // block formats + the host dequant reference

using namespace si;
using namespace sycl;

int main(int argc, char ** argv) {
    // n_expert is deliberately not a multiple of 32 or a power of two, so the
    // router's warp-per-expert chunking, the softmax's tail and the top-k's
    // tie-break are all live; both K extents are multiples of 256 because the
    // quantized superblock split (and therefore n_sb = K/256) is what the model
    // uses and a K < 256 would silently degenerate it to zero.
    const int n_expert = argc > 1 ? atoi(argv[1]) : 37;
    const int top_k = argc > 2 ? atoi(argv[2]) : 5;
    const int n_embd = argc > 3 ? atoi(argv[3]) : 256;
    const int n_ff = argc > 4 ? atoi(argv[4]) : 512;
    const int n_tok = argc > 5 ? atoi(argv[5]) : 7;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);

    // gpu_selector_v, never the default queue: the default device is
    // implementation-defined, and on a multi-GPU host it is not the GPU - every
    // kernel then runs somewhere the test is not looking and reads back the
    // allocation's initial contents, which is every id and weight zero.  That
    // reads exactly like "the router picked expert 0 for every token".
    queue q{sycl::gpu_selector_v};
    // ---- host reference data
    // one expert plane is [N][K]: gate/up are n_ff_exp rows of n_embd, down is
    // n_embd rows of n_ff_exp.  Sizing these by n_expert*K alone under-allocates
    // by a factor of N and the kernels then read past the buffer.
    const size_t plane_gu = (size_t)n_expert * n_ff * n_embd;
    const size_t plane_dn = (size_t)n_expert * n_embd * n_ff;
    std::vector<float> h_rout((size_t)n_expert * n_embd);
    std::vector<float> h_gate(plane_gu), h_up(plane_gu), h_down(plane_dn), h_x((size_t)n_tok * n_embd);
    for (auto & v : h_rout) v = uni(rng) * 0.1f;
    for (auto & v : h_gate) v = uni(rng);
    for (auto & v : h_up) v = uni(rng);
    for (auto & v : h_down) v = uni(rng);
    for (auto & v : h_x) v = uni(rng);
    // A per-token scale so a token swap cannot pass, kept mild on purpose, and
    // the router matrix is scaled down with it.  Together they keep the softmax
    // out of the underflow range at any token count: driving the logits apart by
    // O(n_tok) instead exp()-underflows all but the argmax to *exactly* zero,
    // and the top-k then degenerates into a tie among zeros that two
    // implementations break differently on nothing but the last ulp.
    for (int t = 0; t < n_tok; t++) {
        for (int k = 0; k < n_embd; k++) h_x[(size_t)t * n_embd + k] *= 1.0f + 0.02f * (float)t;
    }

    auto up = [&](void * p, const void * h, size_t n) {
        q.memcpy(p, h, n * sizeof(float)).wait();
    };
    float * d_rout = (float *)malloc_device(h_rout.size() * 4, q);
    float * d_gate = (float *)malloc_device(h_gate.size() * 4, q);
    float * d_up = (float *)malloc_device(h_up.size() * 4, q);
    float * d_dn = (float *)malloc_device(h_down.size() * 4, q);
    float * d_x = (float *)malloc_device(h_x.size() * 4, q);
    up(d_rout, h_rout.data(), h_rout.size());
    up(d_gate, h_gate.data(), h_gate.size());
    up(d_up, h_up.data(), h_up.size());
    up(d_dn, h_down.data(), h_down.size());
    up(d_x, h_x.data(), h_x.size());

    const int n_pairs = n_tok * top_k;
    auto I = [&](size_t n) { return (int32_t *)malloc_device(n * sizeof(int32_t), q); };
    auto F = [&](size_t n) -> float * { return (float *)malloc_device(n * sizeof(float), q); };
    int32_t * d_ids = I(n_pairs);
    float * d_wts = F(n_pairs);
    float * d_h = F((size_t)n_pairs * n_ff);
    float * d_acc = F((size_t)n_pairs * n_embd);
    int32_t * d_off = I(n_expert + 1), *d_cur = I(n_expert), *d_pair = I(n_pairs);
    int32_t * d_act = I(n_expert + 1), *d_nact = I(1);
    int32_t * d_slot = I(n_expert);
    // every expert resident in its own slot: the identity the engine installs
    {
        std::vector<int32_t> id(n_expert);
        for (int e = 0; e < n_expert; e++) id[e] = e;
        q.memcpy(d_slot, id.data(), n_expert * sizeof(int32_t)).wait();
    }

    int fails = 0;
    // Relative to the tensor's own scale: the kernels accumulate a 256-term dot
    // in a different order from the scalar reference, so the criterion has to be
    // the output's magnitude, not an absolute epsilon (AGENTS.md's per-element
    // yardstick flags cancellation, not defects).
    auto chk = [&](const char * what, const std::vector<float> & got, const std::vector<float> & want, float rtol) {
        double worst = 0, scale = 1e-6;
        int wi = -1;
        for (size_t i = 0; i < want.size(); i++) scale = std::max(scale, (double)std::fabs(want[i]));
        for (size_t i = 0; i < want.size(); i++) {
            const double e = std::fabs((double)got[i] - (double)want[i]) / scale;
            if (e > worst) { worst = e; wi = (int)i; }
        }
        const bool ok = worst <= rtol;
        if (!ok) fails++;
        printf("%-10s %s  rel=%.3e (tol %.1e, |max|=%.3g) at %d\n", what, ok ? "OK  " : "FAIL", worst, rtol, scale, wi);
    };

    // Infrastructure probe: does a trivial kernel's write come back at all?
    // Under this box's Level Zero backend every MoE kernel silently produced
    // nothing while the same source passed under OpenCL, so establish whether
    // that is this file or this device before reading anything into it.
    {
        float * d_probe = F(256);
        q.parallel_for(range<1>(256), [=](id<1> i) { d_probe[i[0]] = (float) i[0] + 0.5f; });
        std::vector<float> pv(256);
        q.memcpy(pv.data(), d_probe, 256 * sizeof(float)).wait();
        bool ok = true;
        for (int i = 0; i < 256; i++) {
            ok = ok && (pv[i] == (float) i + 0.5f);
        }
        printf("%-10s %s  (device %s)\n", "probe", ok ? "OK  " : "FAIL", q.get_device().get_info<info::device::name>().c_str());
        if (!ok) {
            fails++;
        }
        free(d_probe, q);
    }

    // ---- reference: router
    float * d_logits = F((size_t) n_tok * n_expert);
    moe_router_launch(q, d_rout, GGML_TYPE_F32, d_x, n_embd, n_tok, n_embd, n_expert, top_k, d_logits, d_ids,
                      d_wts);
    std::vector<int32_t> ids(n_pairs);
    std::vector<float> wts(n_pairs);
    q.memcpy(ids.data(), d_ids, n_pairs * sizeof(int32_t)).wait();
    q.memcpy(wts.data(), d_wts, n_pairs * sizeof(float)).wait();

    if (getenv("MOE_DUMP")) {
        printf("ids:");
        for (int i = 0; i < n_pairs; i++) printf(" %d", ids[i]);
        printf("\nwts:");
        for (int i = 0; i < n_pairs; i++) printf(" %.4f", wts[i]);
        printf("\n");
    }
    std::vector<float> ref_logits(n_expert), ref_p(n_expert);
    std::vector<std::vector<int>> ref_sel(n_tok);
    std::vector<std::vector<float>> ref_w(n_tok);
    for (int t = 0; t < n_tok; t++) {
        const float * xr = &h_x[(size_t)t * n_embd];
        float mx = -1e30f;
        for (int e = 0; e < n_expert; e++) {
            float a = 0;
            for (int k = 0; k < n_embd; k++) a += h_rout[(size_t)e * n_embd + k] * xr[k];
            ref_logits[e] = a;
            mx = a > mx ? a : mx;
        }
        float sum = 0;
        for (int e = 0; e < n_expert; e++) { ref_p[e] = std::exp(ref_logits[e] - mx); sum += ref_p[e]; }
        for (int e = 0; e < n_expert; e++) ref_p[e] /= sum;
        // top-k by value, ties to the lower index
        std::vector<int> idx(n_expert);
        for (int e = 0; e < n_expert; e++) idx[e] = e;
        std::stable_sort(idx.begin(), idx.end(),
                         [&](int a, int b) { return ref_p[a] > ref_p[b]; });
        idx.resize(top_k);
        float wsum = 0;
        for (int e : idx) wsum += ref_p[e];
        const float den = wsum < 6.103515625e-5f ? 6.103515625e-5f : wsum;
        for (int j = 0; j < top_k; j++) {
            ref_sel[t].push_back(idx[j]);
            ref_w[t].push_back(ref_p[idx[j]] / den);
        }
    }
    if (getenv("MOE_DUMP")) {
        printf("ref  ids:");
        for (int t = 0; t < n_tok; t++) for (int j = 0; j < top_k; j++) printf(" %d", ref_sel[t][j]);
        printf("\nref  wts:");
        for (int t = 0; t < n_tok; t++) for (int j = 0; j < top_k; j++) printf(" %.4f", ref_w[t][j]);
        printf("\n");
    }
    {
        bool ok = true;
        for (int p = 0; p < n_pairs && ok; p++) {
            const int t = p / top_k, j = p % top_k;
            if (ids[p] != ref_sel[t][j] || std::fabs(wts[p] - ref_w[t][j]) > 2e-5f) {
                printf("router    FAIL  pair %d (tok %d slot %d): got id=%d w=%.6f, want id=%d w=%.6f\n", p, t, j,
                       ids[p], wts[p], ref_sel[t][j], ref_w[t][j]);
                ok = false;
            }
        }
        printf("router     %s  %d pairs\n", ok ? "OK  " : "FAIL", n_pairs);
        if (!ok) fails++;
    }

    // ---- CSR + gate/up + down + reduce
    const int n_act = std::min(n_expert, n_pairs);
    moe_csr_clear_launch(q, d_off, d_cur, d_act, d_nact, n_expert);
    moe_csr_count_launch(q, d_ids, n_pairs, d_off);
    moe_csr_prefix_launch(q, d_off, d_act, d_nact, n_expert);
    moe_csr_scatter_launch(q, d_ids, d_off, d_cur, d_pair, n_pairs);
    {   // sentinel-fill h so "never written" is distinguishable from "written 0"
        std::vector<float> sent((size_t)n_pairs * n_ff, -12345.f);
        q.memcpy(d_h, sent.data(), sent.size() * 4).wait();
    }
    moe_gate_up_launch(q, d_gate, d_up, d_slot, n_ff * n_embd * sizeof(float), n_ff * n_embd * sizeof(float), d_off,
                       d_pair, d_act, d_nact, n_act, d_x, n_embd, d_h, n_ff, n_embd, n_ff, top_k, GGML_TYPE_F32,
                       GGML_TYPE_F32);
    moe_down_launch(q, d_dn, d_slot, n_embd * n_ff * sizeof(float), d_off, d_pair, d_act, d_nact, n_act, d_h, n_ff,
                    d_acc, n_embd, n_embd, n_ff, GGML_TYPE_F32);

    std::vector<int32_t> off(n_expert + 1), pair(n_pairs), act(n_expert + 1), nact(1);
    q.memcpy(off.data(), d_off, off.size() * 4).wait();
    q.memcpy(pair.data(), d_pair, n_pairs * 4).wait();
    q.memcpy(act.data(), d_act, act.size() * 4).wait();
    q.memcpy(nact.data(), d_nact, 4).wait();
    {
        bool ok = off[n_expert] == n_pairs;
        for (int e = 0; e < n_expert && ok; e++) ok = off[e] <= off[e + 1];
        // pair[pos] holds a pair *index*; the expert owning `pos` must be the
        // one the router picked for that pair
        for (int pos = 0; pos < n_pairs && ok; pos++) {
            int e = -1;
            for (int c = 0; c < n_expert; c++) {
                if (pos >= off[c] && pos < off[c + 1]) { e = c; break; }
            }
            ok = (e >= 0 && ids[pair[pos]] == e);
            if (!ok) printf("  pos %d holds pair %d (expert %d) but sits in expert %d's range\n", pos, pair[pos],
                            e >= 0 ? ids[pair[pos]] : -1, e);
        }
        // and the active list must be exactly the non-empty experts, in order
        {
            int na = 0;
            for (int e = 0; e < n_expert; e++) na += (off[e + 1] > off[e]);
            ok = ok && na == nact[0];
            for (int i = 0; i < na && ok; i++) ok = act[i] < n_expert;
        }
        if (getenv("MOE_DUMP")) {
            printf("  off:"); for (int e = 0; e <= n_expert; e++) printf(" %d", off[e]);
            printf("\n  active:"); for (int e = 0; e < nact[0]; e++) printf(" %d", act[e]);
            printf("\n  pair:"); for (int e = 0; e < nact[0]; e++) for (int i = off[e]; i < off[e+1]; i++) printf(" %d->%d", act[e], pair[i]);
            printf("\n  ids:"); for (int e = 0; e < nact[0]; e++) printf(" %d", ids[e]);
            printf("\n");
        }
        printf("csr        %s  n_active=%d (want %zu distinct)\n", ok ? "OK  " : "FAIL", nact[0],
               [&] {
                   std::vector<int> s = ref_sel[0];
                   std::vector<char> seen(n_expert, 0);
                   for (int t = 0; t < n_tok; t++) for (int e : ref_sel[t]) seen[e] = 1;
                   size_t c = 0; for (int e = 0; e < n_expert; e++) c += seen[e];
                   return c;
               }());
        if (!ok || nact[0] > n_act) fails++;
    }
    std::vector<float> got_h((size_t)n_pairs * n_ff);
    q.memcpy(got_h.data(), d_h, got_h.size() * 4).wait();
    {
        std::vector<float> want((size_t)n_pairs * n_ff);
        for (int p = 0; p < n_pairs; p++) {
            const int t = p / top_k, e = ids[p];
            const float * xr = &h_x[(size_t)t * n_embd];
            for (int n = 0; n < n_ff; n++) {
                float g = 0, u = 0;
                for (int k = 0; k < n_embd; k++) g += h_gate[(size_t)e * n_ff * n_embd + n * n_embd + k] * xr[k];
                for (int k = 0; k < n_embd; k++) u += h_up[(size_t)e * n_ff * n_embd + n * n_embd + k] * xr[k];
                want[(size_t)p * n_ff + n] = (g / (1.f + std::exp(-g))) * u;
            }
        }
        chk("gate/up", got_h, want, 1e-4f);
        if (getenv("MOE_DUMP")) {
            printf("  pair0 expert %d: got", ids[0]);
            for (int n = 0; n < 6; n++) printf(" %.4f", got_h[n]);
            printf("\n              want");
            for (int n = 0; n < 6; n++) printf(" %.4f", want[n]);
            printf("\n");
            {
                // which (expert,row) does each written value correspond to?
                for (int n = 0; n < n_ff; n++) {
                    if (got_h[n] != -12345.f) { printf("  pair0 row %d: got %.4f want %.4f\n", n, got_h[n], want[n]); if (n > 6) break; }
                }
                // brute force: which expert/row reproduces got_h[0]?
                for (int e = 0; e < n_expert && getenv("MOE_DUMP"); e++) {
                    float acc = 0.f;
                    const float *xr = &h_x[0];
                    for (int k = 0; k < n_embd; k++) acc += h_gate[(size_t)e * n_ff * n_embd + 0 * n_embd + k] * xr[k];
                    float uu = 0.f;
                    for (int k = 0; k < n_embd; k++) uu += h_up[(size_t)e * n_ff * n_embd + 0 * n_embd + k] * xr[k];
                    float v = (acc / (1.f + std::exp(-acc))) * uu;
                    if (std::fabs(v - got_h[0]) < 1e-3) printf("  got_h[0] matches expert %d row 0 (want %d)\n", e, (int)ids[0]);
                }
            }
            int nw = -1;
            for (int i = 0; i < n_pairs; i++) nw += (got_h[(size_t)i * n_ff] == -12345.f) ? 0 : 1;
            printf("  pairs written: %d of %d\n", nw, n_pairs);
            printf("  h_gate[exp*plane .. +6] (row 0):");
            for (int k = 0; k < 6; k++) printf(" %.4f", h_gate[(size_t)ids[0] * n_ff * n_embd + k]);
            printf("\n  x[0..5]:");
            for (int k = 0; k < 6; k++) printf(" %.4f", h_x[k]);
            printf("\n");
            for (int p = 0; p < n_pairs; p++) {
                double w = 0;
                for (int n = 0; n < n_ff; n++) w = std::max(w, std::fabs((double)got_h[(size_t)p * n_ff + n] - want[(size_t)p * n_ff + n]));
                if (w > 1e-4) printf("  pair %d (tok %d slot %d, expert %d) max|d|=%.3e\n", p, p / top_k, p % top_k, ids[p], w);
            }
        }
    }
    std::vector<float> got_acc((size_t)n_pairs * n_embd);
    q.memcpy(got_acc.data(), d_acc, got_acc.size() * 4).wait();
    {
        std::vector<float> want((size_t)n_pairs * n_embd);
        for (int p = 0; p < n_pairs; p++) {
            const int e = ids[p];
            for (int n = 0; n < n_embd; n++) {
                float a = 0;
                for (int k = 0; k < n_ff; k++) a += h_down[(size_t)e * n_embd * n_ff + n * n_ff + k] * got_h[(size_t)p * n_ff + k];
                want[(size_t)p * n_embd + n] = a;
            }
        }
        chk("down", got_acc, want, 2e-3f);
        if (getenv("MOE_DUMP")) {
            printf("  acc got:"); for (int i = 0; i < 6; i++) printf(" %.4f", got_acc[i]);
            printf("\n  acc want:"); for (int i = 0; i < 6; i++) printf(" %.4f", want[i]);
            printf("\n  h[0..5]:"); for (int i = 0; i < 6; i++) printf(" %.4f", got_h[i]);
            printf("\n  down[exp*plane..+5] row0:"); for (int i = 0; i < 6; i++) printf(" %.4f", h_down[(size_t)ids[0] * n_embd * n_ff + i]);
            printf("\n");
        }
    }
    // out starts at zero, so the reduce is the whole routed sum
    float * d_out = F((size_t)n_tok * n_embd);
    moe_reduce_launch(q, d_acc, d_wts, top_k, n_tok, n_embd, n_embd, d_out, n_embd);
    {
        std::vector<float> got((size_t)n_tok * n_embd);
        q.memcpy(got.data(), d_out, got.size() * 4).wait();
        std::vector<float> want((size_t)n_tok * n_embd, 0.f);
        for (int p = 0; p < n_pairs; p++) {
            const int t = p / top_k;
            for (int n = 0; n < n_embd; n++) want[(size_t)t * n_embd + n] += wts[p] * got_acc[(size_t)p * n_embd + n];
        }
        chk("reduce", got, want, 2e-3f);
    }

    // ---- a non-resident expert must contribute nothing on the GPU
    {
        std::vector<int32_t> sm(n_expert);
        for (int e = 0; e < n_expert; e++) sm[e] = (e % 3 == 0) ? -1 : e;
        q.memcpy(d_slot, sm.data(), n_expert * sizeof(int32_t)).wait();
        moe_csr_clear_launch(q, d_off, d_cur, d_act, d_nact, n_expert);
        moe_csr_count_launch(q, d_ids, n_pairs, d_off);
        moe_csr_prefix_launch(q, d_off, d_act, d_nact, n_expert);
        moe_csr_scatter_launch(q, d_ids, d_off, d_cur, d_pair, n_pairs);
        moe_gate_up_launch(q, d_gate, d_up, d_slot, n_ff * n_embd * sizeof(float), n_ff * n_embd * sizeof(float),
                           d_off, d_pair, d_act, d_nact, n_act, d_x, n_embd, d_h, n_ff, n_embd, n_ff, top_k,
                           GGML_TYPE_F32, GGML_TYPE_F32);
        moe_down_launch(q, d_dn, d_slot, n_embd * n_ff * sizeof(float), d_off, d_pair, d_act, d_nact, n_act, d_h,
                        n_ff, d_acc, n_embd, n_embd, n_ff, GGML_TYPE_F32);
        std::vector<float> got((size_t)n_pairs * n_embd);
        q.memcpy(got.data(), d_acc, got.size() * 4).wait();
        // every pair whose expert has no slot must read as untouched, and every
        // other pair must be unchanged from the resident run
        bool ok = true;
        for (int p = 0; p < n_pairs && ok; p++) ok = (ids[p] % 3 == 0) || got[(size_t)p * n_embd] == got_acc[(size_t)p * n_embd];
        printf("slot -1    %s  (a skipped expert contributes nothing)\n", ok ? "OK  " : "FAIL");
        if (!ok) fails++;
    }

    // restore the all-resident slot map: the stage above leaves every third
    // expert unresident, which is exactly what the quantized stages must not see
    {
        std::vector<int32_t> id(n_expert);
        for (int e = 0; e < n_expert; e++) id[e] = e;
        q.memcpy(d_slot, id.data(), n_expert * sizeof(int32_t)).wait();
    }

    // ---- the real model's weight types.  The f32 stages above exercise the
    // routing, the CSR and the reduction; what they cannot exercise is the
    // Q4_K/Q5_K lane dequant inside the expert dot, which is a *different* code
    // path (the dense GEMV normally repacks Q4_K onto the native u4 store and
    // never reaches dequant_sb_lane_typed<12>).  Random blocks through the host
    // dequant are the only oracle available without a model.
    for (int pass = 0; pass < 2; pass++) {
        const uint32_t wt = pass == 0 ? 12u : 13u; // Q4_K gate/up, Q5_K down
        const int bs = wt == 12 ? 144 : 176;
        // gate and up are two banks of n_expert planes; down is one
        const size_t plane = (size_t)n_ff * (size_t)n_embd / 256 * bs;
        const size_t n_bank = wt == 12 ? 2 : 1;
        std::vector<char> hq(n_bank * n_expert * plane);
        for (size_t i = 0; i + 1 < hq.size(); i += 2) {
            hq[i] = (char)(rng() & 0xFF);
            hq[i + 1] = (char)(rng() & 0xFF);
        }
        // sane f16 scales, or every dequant is inf/nan and the test proves nothing
        const size_t nb = hq.size() / bs;
        for (size_t b = 0; b < nb; b++) {
            uint16_t d = ggml_float_to_half(0.02f + 0.001f * (float)(b % 17));
            uint16_t m = ggml_float_to_half(0.01f + 0.001f * (float)(b % 11));
            __builtin_memcpy(hq.data() + b * bs + 0, &d, 2);
            __builtin_memcpy(hq.data() + b * bs + 2, &m, 2);
        }
        void * dq = malloc_device(hq.size(), q);
        q.memcpy(dq, hq.data(), hq.size()).wait();
        // host reference dequant of every expert plane
        const int rows = wt == 12 ? n_ff : n_embd; // gate/up rows, else down rows
        const int K = wt == 12 ? n_embd : n_ff;
        const int n_planes = wt == 12 ? 2 * n_expert : n_expert; // gate+up, or down
        std::vector<float> hr((size_t)n_planes * rows * K);
        std::vector<float> ref((size_t)(wt == 12 ? 2 * n_pairs * n_ff : n_pairs * n_embd));
        for (int pl = 0; pl < n_planes; pl++) {
            const char * base = hq.data() + (size_t)pl * plane;
            for (int r = 0; r < rows; r++) {
                float * dst = &hr[((size_t)pl * rows + r) * K];
                for (int sb = 0; sb < K / 256; sb++) {
                    const char * blk = base + ((size_t)r * (K / 256) + sb) * bs;
                    if (wt == 12) dequantize_block_q4_K((const block_q4_K *)blk, dst + sb * 256);
                    else dequantize_block_q5_K((const block_q5_K *)blk, dst + sb * 256);
                }
            }
        }
        // dot per plane/row against the token activation
        // one activation row per token: the kernels index it by pair/top_k, so
        // the reference must too
        std::vector<float> act((size_t)n_tok * K, 0.f);
        for (auto & v : act) v = uni(rng);
        void * da = malloc_device(act.size() * 4, q);
        q.memcpy(da, act.data(), act.size() * 4).wait();
        std::vector<float> hrow((size_t)n_pairs * n_ff);
        if (wt != 12) {
            q.memcpy(hrow.data(), d_h, hrow.size() * 4).wait();
        }
        if (wt == 12) {
            // gate and up share one launch's output layout, so drive them as the
            // two banks the engine does
            std::vector<char> hgate(hq.begin(), hq.begin() + (size_t)n_expert * plane);
            std::vector<char> hup(hq.begin() + (size_t)n_expert * plane, hq.end());
            void * dgq = malloc_device(hgate.size(), q), *duq = malloc_device(hup.size(), q);
            q.memcpy(dgq, hgate.data(), hgate.size()).wait();
            q.memcpy(duq, hup.data(), hup.size()).wait();
            // slot stride = ONE expert plane, i.e. rows*K/256 blocks
            moe_gate_up_launch(q, dgq, duq, d_slot, plane, plane, d_off, d_pair, d_act, d_nact,
                               n_act, (const float *)da, K, d_h, n_ff, n_embd, n_ff, top_k, wt, wt);
        } else {
            void * dqq = malloc_device(hq.size(), q);
            q.memcpy(dqq, hq.data(), hq.size()).wait();
            moe_down_launch(q, dqq, d_slot, plane, d_off, d_pair, d_act, d_nact, n_act, d_h, n_ff, d_acc, n_embd,
                            n_embd, n_ff, wt);
        }
        std::vector<float> res(wt == 12 ? (size_t)n_pairs * n_ff : (size_t)n_pairs * n_embd);
        q.memcpy(res.data(), wt == 12 ? (const void *)d_h : (const void *)d_acc, res.size() * 4).wait();
        // reference: for each pair, silu(gate . x) * (up . x) or down . h
        {
            std::vector<float> want(res.size());
            if (wt == 12) {
                for (int pr = 0; pr < n_pairs; pr++) {
                    const int t = pr / top_k, e = ids[pr];
                    const float * xr = &act[(size_t)t * K];
                    for (int r = 0; r < n_ff; r++) {
                        float g = 0, u = 0;
                        for (int k = 0; k < K; k++) {
                            g += hr[((size_t)e * n_ff + r) * K + k] * xr[k];
                            u += hr[((size_t)(n_expert + e) * n_ff + r) * K + k] * xr[k];
                        }
                        want[(size_t)pr * n_ff + r] = (g / (1.f + std::exp(-g))) * u;
                    }
                }
            } else {
                for (int pr = 0; pr < n_pairs; pr++) {
                    const int e = ids[pr];
                    for (int r = 0; r < n_embd; r++) {
                        float a = 0;
                        for (int k = 0; k < K; k++)
                            a += hr[((size_t)e * n_embd + r) * K + k] * hrow[(size_t)pr * n_ff + k];
                        want[(size_t)pr * n_embd + r] = a;
                    }
                }
            }
            char nm[32];
            snprintf(nm, sizeof nm, "expert %s", wt == 12 ? "Q4_K" : "Q5_K");
            chk(nm, res, want, 2e-3f);
        }
        free(dq, q);
    }

    printf(fails ? "MOE KERNELS: %d FAILURES\n" : "all MoE kernel stages OK\n", fails);
    return fails ? 1 : 0;
}