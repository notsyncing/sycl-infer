// ---------------------------------------------------------------------------
// Host MoE kernels for the CPU partition.
//
// A routed-MoE model has to run on the CPU backend two ways: an all-`cpu`
// --device cpu run, and a --layer-map whose last partition is `cpu` (the
// hybrid GPU+CPU split, where the CPU takes the tail layers).  Both go through
// record_forward's MoE branch, which used to call the SYCL launchers
// unconditionally and died at startup on `No kernel named moe_router_logits`
// for the CPU device.
//
// These mirror src/backend/gpu/kernels/moe.cpp step for step, because the
// hybrid case runs *both* paths in one forward and a discrepancy between them
// would show up as a layer-number-dependent error rather than a crash:
//
//   router    logits = gate_inp @ x, softmax, top-k by *rank counting* so ties
//             break to the lower index, then renormalise by
//             max(sum, 6.103515625e-5f) - the same clamp llama.cpp uses
//   experts   silu(gate) * up, then down.  The weights are NOT applied here:
//             the reduction multiplies, exactly like moe_reduce_launch.
//   epilogue  out += sigmoid(gate_inp_shexp @ x) * shared
//
// The expert dot products accumulate in double, matching test_moe_ref's
// reference implementation rather than the device's fp32; that difference is
// the ~1% routed-energy gap measured between a resident and a spilled pair.
// ---------------------------------------------------------------------------

#include <atomic>
#include <chrono>
#include <cstdlib>

#include "common.h"

#include <algorithm>
#include <cmath>

namespace si {

// ---- the activation a quantized row element -----------------------------------
static inline float moe_rd_elem(uint32_t type, const char * base, size_t off, int i) {
    switch (type) {
    case GGML_TYPE_F32: return ((const float *) base)[i];
    case GGML_TYPE_F16: return ggml_half_to_float(((const uint16_t *) base)[i]);
    case GGML_TYPE_BF16: {
        uint32_t b = ((const uint16_t *) base)[i];
        b <<= 16;
        float f;
        std::memcpy(&f, &b, 4);
        return f;
    }
    default: (void) off; return 0.f;
    }
}

// dequantize expert `e`'s row `r` of `t` into `dst`.
// `plane_bytes` is one *expert's* width, so the row's address is the tensor base
// plus the expert's plane plus the row.  Forgetting the plane evaluates every
// expert against expert 0's weights - the read is in range and the answer is
// plausible, which is why that mistake survived a passing gen.
static inline void moe_dequant_expert_row(const cpu_moe_wt & t, size_t e, int r, float * dst, int K) {
    const char * p = t.data + e * t.plane_bytes + (size_t) r * quant_row_bytes(t.type, K);
    dequantize_row(t.type, p, dst, K);
}

// ---- the router ----------------------------------------------------------------
// logits = gate_inp @ x, softmax, top-k, renormalise.
// `logits` is scratch of n_tok*n_expert floats (it may be null if the caller only
// wants ids/wts and already has its own scratch; we always need it, so it is not
// optional here).
void cpu_moe_route(const float * x, int x_stride, const void * gate_inp, uint32_t gp_type, int n_tok, int n_embd,
                   int n_expert, int top_k, float * logits, int32_t * ids, float * wts, bool fast_f32) {
    // The logits are per (token, expert) independent, and the F32 router (what
    // every Qwen MoE uses) is a plain dot: 512 rows of 2560.  This used to be a
    // serial double loop, which cost 217 ms/token on the 512-expert model.
    par(n_tok * n_expert, [&](int j) {
        const int t = j / n_expert;
        const int e = j % n_expert;
        const float * xr = x + (size_t)t * x_stride;
        const float * w = (const float *)gate_inp + (size_t)e * n_embd;
        if (fast_f32 && gp_type == GGML_TYPE_F32) {
            logits[(size_t)t * n_expert + e] = dot_f32(xr, w, n_embd);
        } else if (gp_type == GGML_TYPE_F32) {
            double a = 0;
            for (int k = 0; k < n_embd; k++) {
                a += (double)w[k] * (double)xr[k];
            }
            logits[(size_t)t * n_expert + e] = (float)a;
        } else {
            const char * wb = (const char *)gate_inp + (size_t)e * (size_t)n_embd * 2;
            double a = 0;
            for (int k = 0; k < n_embd; k++) {
                a += (double)moe_rd_elem(gp_type, wb, 0, k) * (double)xr[k];
            }
            logits[(size_t)t * n_expert + e] = (float)a;
        }
    });
    par(n_tok, [&](int t) {
        float * p = logits + (size_t)t * n_expert;
        float mx = -3.402823466e38f;
        for (int e = 0; e < n_expert; e++) {
            mx = std::fmax(mx, p[e]);
        }
        float sum = 0.f;
        for (int e = 0; e < n_expert; e++) {
            p[e] = std::exp(p[e] - mx);
            sum += p[e];
        }
        const float inv = 1.f / sum;
        for (int e = 0; e < n_expert; e++) {
            p[e] *= inv;
        }
        int32_t * ir = ids + (size_t)t * top_k;
        float * wr = wts + (size_t)t * top_k;
        float wsum = 0.f;
        for (int e = 0; e < n_expert; e++) {
            const float pe = p[e];
            int rank = 0;
            for (int f = 0; f < n_expert; f++) {
                const float pf = p[f];
                if (pf > pe || (pf == pe && f < e)) {
                    rank++;
                }
            }
            if (rank < top_k) {
                ir[rank] = e;
                wr[rank] = pe;
                wsum += pe;
            }
        }
        const float den = std::fmax(wsum, 6.103515625e-5f);
        for (int j = 0; j < top_k; j++) {
            wr[j] /= den;
        }
    });
}

// ---- grouped int8 expert path (qwen4exp) ---------------------------------------
//
// The per-pair path below re-reads an expert's weights for every pair that
// selected it.  At a 512-token prefill that is ~54 reads of each expert, i.e.
// ~73 GB of extra traffic per layer; the grouped path reads each expert's
// gate/up/down once for all of its pairs.  Gate/up use the K-quant int8 GEMM
// (Q4_K); down uses the symmetric int8 store built at startup because the
// Q5_1 down bank is not a K-quant block and K=640 is not a 256 multiple.
static void moe_experts_grouped_i8(const cpu_moe & M, const float * x, int x_stride, int n_tok, int top_k,
                                   const int32_t * ids, float * acc, int acc_stride) {
    const int K = M.n_embd, n_ff = M.n_ff_exp, Kd = n_ff, E = M.n_expert;
    const int np = n_tok * top_k;
    const int G = K / 32, Gd = Kd / 32;
    std::vector<int> cnt, off, cur, pairs;
    std::vector<float> x8m;
    std::vector<int32_t> xq8m;
    std::vector<int8_t> x8, x8h;
    std::vector<float> hmeta, hbuf;
    std::vector<int32_t> hsum;
    static std::atomic<int> grp_calls{0};
    const bool grp_t = grp_calls.fetch_add(1) == 0;
    auto ms_now = [] {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    const double tg0 = grp_t ? ms_now() : 0.0;
    if (std::getenv("PF_Q4_GRPDBG")) {
        fprintf(stderr, "[grp] enter n_tok=%d np=%d\n", n_tok, np);
    }
    cnt.assign((size_t)E, 0);
    off.assign((size_t)E + 1, 0);
    pairs.resize((size_t)np);
    for (int pr = 0; pr < np; pr++) {
        if (ids[pr] < 0 || ids[pr] >= E) {
            fprintf(stderr, "[moe] bad expert id %d at pair %d (E=%d)\n", ids[pr], pr, E);
            return;
        }
        cnt[(size_t)ids[pr]]++;
    }
    for (int e = 0; e < E; e++) {
        off[(size_t)e + 1] = off[(size_t)e] + cnt[(size_t)e];
    }
    cur = off;
    for (int pr = 0; pr < np; pr++) {
        pairs[(size_t)cur[(size_t)ids[pr]]++] = pr;
    }
    int maxP = 0, maxE = -1;
    for (int e = 0; e < E; e++) {
        const int P = off[(size_t)e + 1] - off[(size_t)e];
        if (P > maxP) {
            maxP = P;
            maxE = e;
        }
    }
    if (std::getenv("PF_Q4_GRPDBG")) {
        fprintf(stderr, "[grp] maxP=%d e=%d\n", maxP, maxE);
    }
    x8.resize((size_t)K * n_tok);
    x8m.resize((size_t)G * n_tok * 2);
    xq8m.resize((size_t)G * n_tok * 2);
    cpu_xq(x, nullptr, x_stride, 0, x8.data(), x8m.data(), xq8m.data(), nullptr, n_tok, K);
    const double tx1 = grp_t ? ms_now() : 0.0;
    if (std::getenv("PF_Q4_GRPDBG")) {
        fprintf(stderr, "[grp] xq done\n");
    }
    hbuf.assign((size_t)np * n_ff, 0.f);
    if (isa().qgemv_sb == nullptr) {
        return;
    }
    // One job per expert: the expert's weights are read once and its pairs are
    // dotted serially.  The earlier shape (a parallel GEMM inside a serial
    // expert loop) cost ~74k pool wakeups per prefill and measured 8 GB/s.
    static std::atomic<int> grp_done{0};
    grp_done.store(0);
    par(E, [&](int e) {
        const int b = off[(size_t)e], P = off[(size_t)e + 1] - b;
        const int gd = grp_done.fetch_add(1);
        if (std::getenv("PF_Q4_GRPDBG") && (e == 0 || (gd % 64) == 0)) {
            fprintf(stderr, "[grp] expert e=%d P=%d done=%d\n", e, P, gd);
        }
        if (P <= 0) {
            return;
        }
        const int * pl = pairs.data() + b;
        static std::atomic<int> first_pos{0};
        static std::atomic<int> first_two{0};
        const bool fp = std::getenv("PF_Q4_GRPDBG") && first_pos.fetch_add(1) == 0;
        const bool fp2 = std::getenv("PF_Q4_GRPDBG") && P >= 2 && first_two.fetch_add(1) == 0;
        if (fp2) {
            fprintf(stderr, "[grp] first P>=2 e=%d P=%d\n", e, P);
        }
        if (fp) {
            fprintf(stderr, "[grp] first P>0 e=%d P=%d\n", e, P);
        }
        thread_local std::vector<int8_t> x8e;
        thread_local std::vector<float> xme, gbuf, ubuf;
        thread_local std::vector<int32_t> xqe;
        // The K-quant int8 GEMM tiles tokens in groups of 32 and indexes
        // g*TB + t; TB must be a multiple of 32 or the last tile reads past
        // the group.  Pad the per-expert pair block to 32.
        const int PP = (P + 31) & ~31;
        x8e.resize((size_t)K * PP);
        xme.resize((size_t)G * PP * 2);
        xqe.resize((size_t)G * PP * 2);
        gbuf.resize((size_t)n_ff * PP);
        ubuf.resize((size_t)n_ff * PP);
        for (int v = P; v < PP; v++) {
            for (int g = 0; g < G; g++) {
                const size_t d2 = (size_t)g * PP + v;
                std::memset(x8e.data() + d2 * 32, 0, 32);
                xme[d2 * 2] = 0.f;
                xme[d2 * 2 + 1] = 0.f;
                xqe[d2 * 2] = 0;
                xqe[d2 * 2 + 1] = 0;
            }
        }
        for (int v = 0; v < P; v++) {
            const int tok = pl[v] / top_k;
            for (int g = 0; g < G; g++) {
                const size_t dst = (size_t)g * PP + v;
                const size_t src = (size_t)g * n_tok + tok;
                std::memcpy(x8e.data() + dst * 32, x8.data() + src * 32, 32);
                xme[dst * 2] = x8m[src * 2];
                xme[dst * 2 + 1] = x8m[src * 2 + 1];
                xqe[dst * 2] = xq8m[src * 2];
                xqe[dst * 2 + 1] = xq8m[src * 2 + 1];
            }
        }
        const char * gp = (const char *)M.gate_exps.data + (size_t)e * M.gate_exps.plane_bytes;
        const char * up = (const char *)M.up_exps.data + (size_t)e * M.up_exps.plane_bytes;
        if (fp) {
            fprintf(stderr, "[grp] first gemm start\n");
        }
        cpu_i8_gemm_batch_serial(12, gp, x8e.data(), xme.data(), xqe.data(), K, n_ff, PP, gbuf.data(), n_ff);
        if (fp2) {
            fprintf(stderr, "[grp] P2 gate done\n");
        }
        if (fp) {
            fprintf(stderr, "[grp] first gemm gate done\n");
        }
        cpu_i8_gemm_batch_serial(12, up, x8e.data(), xme.data(), xqe.data(), K, n_ff, PP, ubuf.data(), n_ff);
        if (fp2) {
            fprintf(stderr, "[grp] P2 up done\n");
        }
        if (fp) {
            fprintf(stderr, "[grp] first gemm up done\n");
        }
        for (int v = 0; v < P; v++) {
            for (int i = 0; i < n_ff; i++) {
                hbuf[(size_t)pl[v] * n_ff + i] = cpu_silu(gbuf[(size_t)v * n_ff + i]) * ubuf[(size_t)v * n_ff + i];
            }
        }
        if (fp2) {
            fprintf(stderr, "[grp] P2 h done\n");
        }
        if (fp) {
            fprintf(stderr, "[grp] first h done\n");
        }
    });
    const double tg1 = grp_t ? ms_now() : 0.0;
    if (std::getenv("PF_Q4_GRPDBG")) {
        fprintf(stderr, "[moe] grouped pass1 done n_tok=%d np=%d\n", n_tok, np);
    }
    if (!M.down_i8 || !M.down_sc || M.down_K != Kd || M.down_N != K) {
        fprintf(stderr, "[moe] grouped down store missing/mismatched (%p %p K=%d N=%d)\n", (void *)M.down_i8,
                (void *)M.down_sc, M.down_K, M.down_N);
        return;
    }
    x8h.resize((size_t)Kd * np);
    hmeta.resize((size_t)Gd * np * 2);
    hsum.resize((size_t)Gd * np * 2);
    cpu_xq(hbuf.data(), nullptr, Kd, 0, x8h.data(), hmeta.data(), hsum.data(), nullptr, np, Kd);
    const double th1 = grp_t ? ms_now() : 0.0;
    if (std::getenv("PF_Q4_GRPDBG")) {
        fprintf(stderr, "[moe] grouped pass2 start\n");
    }
    par(E, [&](int e) {
        const int b = off[(size_t)e], P = off[(size_t)e + 1] - b;
        if (P <= 0) {
            return;
        }
        if (M.down_type == 7 && M.down_raw && M.down_row_bytes) {
            cpu_q5_1_down_grouped_serial((const char *)M.down_raw + (size_t)e * M.down_exps.plane_bytes,
                                         M.down_row_bytes, x8h.data(), hmeta.data(), hsum.data(), np, P, M.down_N,
                                         M.down_K, acc, acc_stride, pairs.data() + b);
        } else {
        cpu_q4_down_grouped_serial(M.down_i8 + (size_t)e * M.down_N * M.down_K,
                                   M.down_sc + (size_t)e * M.down_N * (M.down_K / 32), x8h.data(), hmeta.data(),
                                   hsum.data(), np, P, M.down_N, M.down_K, acc, acc_stride, pairs.data() + b);
        }
    });
    if (grp_t && std::getenv("PF_Q4_GRPDBG")) {
        const double te = ms_now();
        fprintf(stderr, "[grp] one call: xq=%.0f gate=%.0f hxq=%.0f down=%.0f ms\n", tx1 - tg0, tg1 - tx1,
                th1 - tg1, te - th1);
    }
}

// ---- the routed experts --------------------------------------------------------
// Fills acc[pr*acc_stride + r] for every (token, expert) pair, skipping rows
// whose expert is not resident when `slots` is non-null.  The weights are NOT
// applied: the reduction multiplies, so this is the unweighted expert output.
void cpu_moe_experts(const cpu_moe & M, const float * x, int x_stride, int n_tok, int top_k, const int32_t * ids,
                     const int * slots, float * acc, int acc_stride, bool fast_f32) {
    if (fast_f32 && !slots && M.down_i8 && n_tok >= 8 && M.gate_exps.type == 12 && M.up_exps.type == 12) {
        moe_experts_grouped_i8(M, x, x_stride, n_tok, top_k, ids, acc, acc_stride);
        return;
    }
    const int n_ff = M.n_ff_exp;
    const int K = M.n_embd;
    const int Kd = n_ff;
    const int np = n_tok * top_k;
    // Per-pair activated intermediate.  Phase 1 and phase 2 are separate
    // parallel passes: a pair's down projection needs all n_ff values, so it
    // cannot be fused with its gate/up rows without a barrier.
    std::vector<float> h((size_t)np * n_ff);
    // Blocks of 8 output rows: the activation (or h) is dequantized once per
    // job instead of once per row, and 8 rows is enough work to amortise it.
    constexpr int B = 8;
    const int nb1 = (n_ff + B - 1) / B;
    par(np * nb1, [&](int j) {
        const int pr = j / nb1;
        const int i0 = (j % nb1) * B;
        const int t = pr / top_k;
        const int e = ids[pr];
        if (e < 0 || e >= M.n_expert || (slots && slots[e] < 0)) {
            for (int i = i0; i < i0 + B && i < n_ff; i++) {
                h[(size_t)pr * n_ff + i] = 0.f;
            }
            return;
        }
        thread_local std::vector<float> xr, rowbuf;
        xr.resize((size_t)K);
        rowbuf.resize((size_t)K);
        const float * xs = x + (size_t)t * x_stride;
        for (int k = 0; k < K; k++) {
            xr[(size_t)k] = xs[k];
        }
        const size_t eo = (size_t)e;
        for (int i = i0; i < i0 + B && i < n_ff; i++) {
            moe_dequant_expert_row(M.gate_exps, eo, i, rowbuf.data(), K);
            float ag = fast_f32 ? dot_f32(rowbuf.data(), xr.data(), K) : (float)0.0;
            if (!fast_f32) {
                double a = 0;
                for (int k = 0; k < K; k++) {
                    a += (double)rowbuf[(size_t)k] * xr[(size_t)k];
                }
                ag = (float)a;
            }
            moe_dequant_expert_row(M.up_exps, eo, i, rowbuf.data(), K);
            float au = fast_f32 ? dot_f32(rowbuf.data(), xr.data(), K) : (float)0.0;
            if (!fast_f32) {
                double a = 0;
                for (int k = 0; k < K; k++) {
                    a += (double)rowbuf[(size_t)k] * xr[(size_t)k];
                }
                au = (float)a;
            }
            h[(size_t)pr * n_ff + i] = cpu_silu(ag) * au;
        }
    });
    const int nb2 = (K + B - 1) / B;
    par(np * nb2, [&](int j) {
        const int pr = j / nb2;
        const int r0 = (j % nb2) * B;
        const int e = ids[pr];
        if (e < 0 || e >= M.n_expert || (slots && slots[e] < 0)) {
            for (int r = r0; r < r0 + B && r < K; r++) {
                acc[(size_t)pr * acc_stride + r] = 0.f;
            }
            return;
        }
        thread_local std::vector<float> hr, rowbuf;
        hr.resize((size_t)Kd);
        rowbuf.resize((size_t)Kd);
        for (int k = 0; k < Kd; k++) {
            hr[(size_t)k] = h[(size_t)pr * n_ff + k];
        }
        const size_t eo = (size_t)e;
        for (int r = r0; r < r0 + B && r < K; r++) {
            moe_dequant_expert_row(M.down_exps, eo, r, rowbuf.data(), Kd);
            float ad = fast_f32 ? dot_f32(rowbuf.data(), hr.data(), Kd) : (float)0.0;
            if (!fast_f32) {
                double a = 0;
                for (int k = 0; k < Kd; k++) {
                    a += (double)rowbuf[(size_t)k] * hr[(size_t)k];
                }
                ad = (float)a;
            }
            acc[(size_t)pr * acc_stride + r] = ad;
        }
    });
}

// ---- the weighted reduction -----------------------------------------------------
void cpu_moe_reduce(const float * acc, const float * wts, int top_k, int n_tok, int n_embd, int acc_stride, float * out,
                    int out_stride) {
    par(n_tok * n_embd, [&](int i) {
        const int t = i / n_embd;
        const int e = i - t * n_embd;
        double a = 0;
        for (int j = 0; j < top_k; j++) {
            const size_t pr = (size_t) t * top_k + j;
            a += (double) wts[pr] * acc[pr * (size_t) acc_stride + e];
        }
        out[(size_t) t * out_stride + e] += (float) a;
    });
}

// ---- the shared expert's sigmoid gate, and adding both halves ------------------
void cpu_moe_epilogue(const float * x, int x_stride, const void * gate, uint32_t gtype, int n_tok, int n_embd,
                      const float * shared, int sh_stride, float * out, int out_stride) {
    par(n_tok, [&](int t) {
        const float * xr = x + (size_t) t * x_stride;
        float g = 1.f;
        if (gate) {
            double a = 0;
            for (int k = 0; k < n_embd; k++) {
                a += (double) moe_rd_elem(gtype, (const char *) gate, 0, k) * (double) xr[k];
            }
            g = cpu_sigmoid((float) a);
        }
        for (int e = 0; e < n_embd; e++) {
            out[(size_t) t * out_stride + e] += g * shared[(size_t) t * sh_stride + e];
        }
    });
}

} // namespace si
