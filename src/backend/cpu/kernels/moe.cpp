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
                   int n_expert, int top_k, float * logits, int32_t * ids, float * wts) {
    // phase 1: the logits.  One row per token; the row's dot product is the
    // only expensive part (n_embd), and the rows are independent.
    par(n_tok, [&](int t) {
        const float * xr = x + (size_t) t * x_stride;
        float * p = logits + (size_t) t * n_expert;
        for (int e = 0; e < n_expert; e++) {
            const char * w = (const char *) gate_inp + (size_t) e * (size_t) n_embd * (gp_type == GGML_TYPE_F32 ? 4 : 2);
            double a = 0;
            for (int k = 0; k < n_embd; k++) {
                a += (double) moe_rd_elem(gp_type, w, 0, k) * (double) xr[k];
            }
            p[e] = (float) a;
        }
        // softmax over this token's row
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
        // top-k by rank counting: rank(e) = #{f : p[f] > p[e] || (p[f]==p[e] && f<e)},
        // so ties break to the lower index and the order is deterministic
        // without a sort (the device does the same, and the CPU rank loop is
        // 256*256 per token - cheap against the expert dots that follow).
        int32_t * ir = ids + (size_t) t * top_k;
        float * wr = wts + (size_t) t * top_k;
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

// ---- the routed experts --------------------------------------------------------
// Fills acc[pr*acc_stride + r] for every (token, expert) pair, skipping rows
// whose expert is not resident when `slots` is non-null.  The weights are NOT
// applied: the reduction multiplies, so this is the unweighted expert output.
void cpu_moe_experts(const cpu_moe & M, const float * x, int x_stride, int n_tok, int top_k, const int32_t * ids,
                     const int * slots, float * acc, int acc_stride) {
    const int n_ff = M.n_ff_exp;
    const int K = M.n_embd;
    const int Kd = n_ff;
    std::vector<float> xr((size_t) K), g((size_t) n_ff), u((size_t) n_ff), h((size_t) n_ff), rowbuf((size_t) K);
    // gate and up share the activation, so both rows are dequantized against the
    // same x: hoist the activation copy out of the per-row loop.
    for (int pr = 0; pr < n_tok * top_k; pr++) {
        const int t = pr / top_k;
        const int e = ids[pr];
        if (e < 0 || e >= M.n_expert) {
            for (int r = 0; r < K; r++) {
                acc[(size_t) pr * acc_stride + r] = 0.f;
            }
            continue;
        }
        if (slots && slots[e] < 0) {
            // non-resident: the device half skips it, the engine's host half
            // fills it.  A *partition* that is not where the split runs must
            // still answer correctly, so mirror the skip and leave zero.
            for (int r = 0; r < K; r++) {
                acc[(size_t) pr * acc_stride + r] = 0.f;
            }
            continue;
        }
        for (int k = 0; k < K; k++) {
            xr[k] = x[(size_t) t * x_stride + k];
        }
        const size_t eo = (size_t) e;
        for (int i = 0; i < n_ff; i++) {
            moe_dequant_expert_row(M.gate_exps, eo, i, rowbuf.data(), K);
            double ag = 0;
            for (int k = 0; k < K; k++) {
                ag += (double) rowbuf[k] * xr[k];
            }
            g[i] = (float) ag;
            moe_dequant_expert_row(M.up_exps, eo, i, rowbuf.data(), K);
            double au = 0;
            for (int k = 0; k < K; k++) {
                au += (double) rowbuf[k] * xr[k];
            }
            u[i] = (float) au;
            h[i] = cpu_silu(g[i]) * u[i];
        }
        for (int r = 0; r < K; r++) {
            moe_dequant_expert_row(M.down_exps, eo, r, rowbuf.data(), Kd);
            double ad = 0;
            for (int k = 0; k < Kd; k++) {
                ad += (double) rowbuf[k] * h[k];
            }
            acc[(size_t) pr * acc_stride + r] = (float) ad;
        }
    }
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
