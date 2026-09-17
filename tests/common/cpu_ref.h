#pragma once
// CPU reference implementation of the Qwen3.5-0.8B forward pass (single sequence,
// arbitrary token count) used to validate the SYCL kernels stage by stage.
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "kernels.h"
#include "model.h"
#include "quant.h"

namespace si {

struct cpu_ref {
    const model & m;
    const hparams & hp;
    int max_seq;

    std::vector<float> x, xn, qkv, z, beta, alpha, conv_out, attn_pre, attn_merged;
    std::vector<float> qbuf, kb, vb, attn_out, ffn, logits;
    std::vector<float> kcache, vcache;
    std::vector<int> attn_slot;
    std::vector<float> gdn_state, conv_state;
    int n_tokens = 0;
    bool record = false;
    bool dbg_attn = false;
    struct snap_t { std::string name; std::vector<float> data; };
    std::vector<snap_t> snaps;
    void snap(const char * name, const float * p, size_t n) {
        if (record) snaps.push_back({name, std::vector<float>(p, p + n)});
    }

    cpu_ref(const model & mm, int max_seq_ = 512) : m(mm), hp(mm.hp), max_seq(max_seq_) {
        const int T = kMaxT;
        x.resize((size_t) T * hp.n_embd);
        xn.resize((size_t) T * hp.n_embd);
        qkv.resize((size_t) T * 3 * hp.d_inner);
        z.resize((size_t) T * hp.d_inner);
        beta.resize((size_t) T * hp.dt_rank);
        alpha.resize((size_t) T * hp.dt_rank);
        conv_out.resize((size_t) T * 3 * hp.d_inner);
        attn_pre.resize((size_t) T * hp.d_inner);
        attn_merged.resize((size_t) T * hp.d_inner);
        qbuf.resize((size_t) T * hp.n_head * 2 * hp.head_dim);
        kb.resize((size_t) T * hp.n_head_kv * hp.head_dim);
        vb.resize((size_t) T * hp.n_head_kv * hp.head_dim);
        attn_out.resize((size_t) T * hp.n_head * hp.head_dim);
        ffn.resize((size_t) T * 2 * hp.n_ff);
        logits.resize(hp.n_vocab);
        int n_gdn = 0, n_attn = 0;
        attn_slot.assign(hp.n_layer, -1);
        for (int il = 0; il < hp.n_layer; il++) {
            if (hp.is_recr(il)) n_gdn++;
            else attn_slot[il] = n_attn++;
        }
        kcache.resize((size_t) n_attn * hp.n_head_kv * max_seq * hp.head_dim, 0.f);
        vcache.resize((size_t) n_attn * hp.n_head_kv * max_seq * hp.head_dim, 0.f);
        gdn_state.assign((size_t) n_gdn * hp.dt_rank * hp.d_state * hp.d_state, 0.f);
        conv_state.assign((size_t) n_gdn * 3 * 3 * hp.d_inner, 0.f);
    }

    static float silu(float v) { return v / (1.f + std::exp(-v)); }

    static void rmsnorm(const float * xx, const float * w, float * out, int n, float eps) {
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double) xx[i] * xx[i];
        const float scale = 1.0f / std::sqrt((float) (ss / n) + eps);
        for (int i = 0; i < n; i++) out[i] = xx[i] * scale * w[i];
    }

    // out[r] = dot(row_r(W), x)
    void matvec(const wt & W, const float * xx, float * out) const {
        std::vector<float> row(W.K);
        const size_t rb = quant_row_bytes(W.type, W.K);
        for (int r = 0; r < W.N; r++) {
            dequantize_row(W.type, (const char *) W.data + (size_t) r * rb, row.data(), W.K);
            double acc = 0;
            for (int k = 0; k < W.K; k++) acc += (double) row[k] * xx[k];
            out[r] = (float) acc;
        }
    }

    void conv_layer(int gidx, const layer_t & L, int t, const float * in, float * out) {
        const int cd = 3 * hp.d_inner;
        float * st = conv_state.data() + (size_t) gidx * 3 * cd;
        const float * cw = L.ssm_conv1d;
        for (int c = 0; c < cd; c++) {
            float v = cw[3 * cd + c] * in[c];
            for (int i = 0; i < 3; i++) v += cw[(size_t) i * cd + c] * st[(size_t) i * cd + c];
            out[c] = silu(v);
        }
        // l2 norm over q and k groups
        for (int grp = 0; grp < 2 * hp.n_group; grp++) {
            const int cb = grp * hp.d_state;
            double ss = 0;
            for (int i = 0; i < hp.d_state; i++) ss += (double) out[cb + i] * out[cb + i];
            const float sc = 1.0f / std::fmax(std::sqrt((float) ss), hp.rms_eps);
            for (int i = 0; i < hp.d_state; i++) out[cb + i] *= sc;
        }
        // update state (oldest first)
        for (int c = 0; c < cd; c++) {
            float v2 = in[c];
            float v1 = (t >= 1) ? (st + 3 * cd - cd)[c] * 0 + v2 * 0 + 0 : 0; // placeholder
            (void) v1;
        }
    }

    void forward(const std::vector<int> & tokens) {
        const int n = (int) tokens.size();
        n_tokens = n;
        for (int t = 0; t < n; t++) {
            int gidx = 0;
            const float * ein = nullptr;
            (void) ein;
            // embedding
            {
                const size_t rb = quant_row_bytes(m.tok_embd.type, hp.n_embd);
                dequantize_row(m.tok_embd.type, (const char *) m.tok_embd.data + (size_t) tokens[t] * rb,
                               x.data() + (size_t) t * hp.n_embd, hp.n_embd);
            }
            for (int il = 0; il < hp.n_layer; il++) {
                const layer_t & L = m.layers[il];
                float * xt = x.data() + (size_t) t * hp.n_embd;
                float * xnt = xn.data() + (size_t) t * hp.n_embd;
                rmsnorm(xt, L.attn_norm, xnt, hp.n_embd, hp.rms_eps);
                if (il == 0) {
                    snap("model.input_embed", xt, hp.n_embd);
                    snap("attn_norm-0", xnt, hp.n_embd);
                }
                if (L.recurrent) {
                    float * qkvt = qkv.data() + (size_t) t * 3 * hp.d_inner;
                    float * zt = z.data() + (size_t) t * hp.d_inner;
                    float * bt = beta.data() + (size_t) t * hp.dt_rank;
                    float * at = alpha.data() + (size_t) t * hp.dt_rank;
                    matvec(L.wqkv, xnt, qkvt);
                    matvec(L.wgate, xnt, zt);
                    matvec(L.ssm_beta, xnt, bt);
                    matvec(L.ssm_alpha, xnt, at);
                    if (il == 0) {
                        snap("raw_wqkv-0", qkvt, 3 * hp.d_inner);
                        snap("raw_wgate-0", zt, hp.d_inner);
                    }
                    if (il == 0) snap("linear_attn_qkv_mixed-0", qkvt, 3 * hp.d_inner);
                    if (il == 0) snap("beta-0", bt, hp.dt_rank);
                    for (int i = 0; i < hp.dt_rank; i++) {
                        bt[i] = 1.f / (1.f + std::exp(-bt[i]));
                        at[i] = std::log(1.f + std::exp(at[i] + L.ssm_dt[i]));
                    }
                    if (il == 0) {
                        snap("a_softplus-0", at, hp.dt_rank);
                        snap("z-0", zt, hp.d_inner);
                    }
                    // conv
                    const int cd = 3 * hp.d_inner;
                    float * st = conv_state.data() + (size_t) gidx * 3 * cd;
                    float * co = conv_out.data() + (size_t) t * cd;
                    for (int c = 0; c < cd; c++) {
                        float v = L.ssm_conv1d[3 + 4 * c] * qkvt[c];
                        for (int i = 0; i < 3; i++)
                            v += L.ssm_conv1d[i + 4 * c] * st[(size_t) i * cd + c];
                        co[c] = silu(v);
                    }
                    if (il == 0) snap("conv_output_silu-0", co, cd);
                    for (int grp = 0; grp < 2 * hp.n_group; grp++) {
                        const int cb = grp * hp.d_state;
                        double ss = 0;
                        for (int i = 0; i < hp.d_state; i++) ss += (double) co[cb + i] * co[cb + i];
                        const float sc = 1.0f / std::fmax(std::sqrt((float) ss), hp.rms_eps);
                        for (int i = 0; i < hp.d_state; i++) co[cb + i] *= sc;
                    }
                    if (il == 0) {
                        snap("q_conv_predelta-0", co, hp.n_group * hp.d_state);
                        snap("k_conv_predelta-0", co + hp.n_group * hp.d_state, hp.n_group * hp.d_state);
                        snap("v_conv_predelta-0", co + 2 * hp.n_group * hp.d_state, hp.n_group * hp.d_state);
                    }
                    // state shift: st[0]=x[t-3], st[1]=x[t-2], st[2]=x[t-1]
                    float * st0 = st, *st1 = st + cd, *st2 = st + 2 * cd;
                    std::memcpy(st0, st1, cd * 4);
                    std::memcpy(st1, st2, cd * 4);
                    std::memcpy(st2, qkvt, cd * 4);
                    // gdn recurrence
                    const int H = hp.dt_rank, D = hp.d_state;
                    float * statep = gdn_state.data() + (size_t) gidx * H * D * D;
                    float * ap = attn_pre.data() + (size_t) t * hp.d_inner;
                    for (int h = 0; h < H; h++) {
                        float * S = statep + (size_t) h * D * D; // S[col][row]: col = v dim, row = k dim
                        const float g = std::exp(at[h] * L.ssm_a[h]);
                        const float bv = bt[h];
                        const float * qv = co + h * D;
                        const float * kv = co + H * D + h * D;
                        const float * vv = co + 2 * H * D + h * D;
                        for (int col = 0; col < D; col++) {
                            float kvv = 0;
                            for (int i = 0; i < D; i++) kvv += S[col * D + i] * kv[i];
                            const float delta = (vv[col] - g * kvv) * bv;
                            float atn = 0;
                            for (int i = 0; i < D; i++) {
                                S[col * D + i] = g * S[col * D + i] + kv[i] * delta;
                                atn += S[col * D + i] * qv[i];
                            }
                            ap[h * D + col] = atn / std::sqrt((float) D);
                        }
                    }
                    if (il == 0) snap("final_output_pre-0", ap, hp.d_inner);
                    // gated norm
                    float * am = attn_merged.data() + (size_t) t * hp.d_inner;
                    for (int h = 0; h < H; h++) {
                        const float * o = ap + h * D;
                        double ss = 0;
                        for (int i = 0; i < D; i++) ss += (double) o[i] * o[i];
                        const float sc = 1.0f / std::sqrt((float) (ss / D) + hp.rms_eps);
                        for (int i = 0; i < D; i++)
                            am[h * D + i] = o[i] * sc * L.ssm_norm[i] * silu(zt[h * D + i]);
                    }
                    if (il == 0) snap("final_output-0", am, hp.d_inner);
                    // ssm_out + residual
                    std::vector<float> tmp(hp.n_embd);
                    matvec(L.ssm_out, am, tmp.data());
                    if (il == 0) snap("linear_attn_out-0", tmp.data(), hp.n_embd);
                    for (int i = 0; i < hp.n_embd; i++) xt[i] = tmp[i] + xt[i];
                    if (il == 0) snap("attn_residual-0", xt, hp.n_embd);
                    gidx++;
                } else {
                    float * kcache_l = kcache.data() + (size_t) attn_slot[il] * hp.n_head_kv * max_seq * hp.head_dim;
                    float * vcache_l = vcache.data() + (size_t) attn_slot[il] * hp.n_head_kv * max_seq * hp.head_dim;
                    if (il == 3) snap("attn_norm-3", xnt, hp.n_embd);
                    float * qbt = qbuf.data() + (size_t) t * hp.n_head * 2 * hp.head_dim;
                    float * kbt = kb.data() + (size_t) t * hp.n_head_kv * hp.head_dim;
                    float * vbt = vb.data() + (size_t) t * hp.n_head_kv * hp.head_dim;
                    matvec(L.wq, xnt, qbt);
                    matvec(L.wk, xnt, kbt);
                    matvec(L.wv, xnt, vbt);
                    if (il == 3) {
                        snap("raw_wq-3", qbt, hp.n_head * 2 * hp.head_dim);
                        snap("raw_wk-3", kbt, hp.n_head_kv * hp.head_dim);
                        snap("raw_wv-3", vbt, hp.n_head_kv * hp.head_dim);
                        snap("attn_merged_xn-3", xnt, hp.n_embd);
                    }
                    if (il == 3) {
                        snap("Vcur-3", vbt, hp.n_head_kv * hp.head_dim);
                    }
                    // q/k norm + rope, write cache
                    for (int h = 0; h < hp.n_head; h++) {
                        float * qh = qbt + h * 2 * hp.head_dim;
                        rmsnorm_t(qh, L.q_norm, hp.head_dim, hp.rms_eps);
                        rope(qh, t, hp);
                    }
                    for (int h = 0; h < hp.n_head_kv; h++) {
                        float * kh = kbt + h * hp.head_dim;
                        rmsnorm_t(kh, L.k_norm, hp.head_dim, hp.rms_eps);
                        rope(kh, t, hp);
                        std::memcpy(&kcache_l[((size_t) h * max_seq + t) * hp.head_dim], kh, hp.head_dim * 4);
                        std::memcpy(&vcache_l[((size_t) h * max_seq + t) * hp.head_dim],
                                    vbt + h * hp.head_dim, hp.head_dim * 4);
                    }
                    if (il == 3) {
                        snap("q_all-3", qbt, hp.n_head * 2 * hp.head_dim);
                        snap("k_all-3", kbt, hp.n_head_kv * hp.head_dim);
                        snap("v_all-3", vbt, hp.n_head_kv * hp.head_dim);
                    }
                    // attention (single query at position t, causal over [0,t])
                    float * ao = attn_out.data() + (size_t) t * hp.n_head * hp.head_dim;
                    const float scale = 1.0f / std::sqrt((float) hp.head_dim);
                    for (int h = 0; h < hp.n_head; h++) {
                        const float * qv = qbt + h * 2 * hp.head_dim;
                        const int kvh = h * hp.n_head_kv / hp.n_head;
                        std::vector<float> sc(t + 1);
                        float m = -INFINITY;
                        for (int k2 = 0; k2 <= t; k2++) {
                            const float * kp = &kcache_l[((size_t) kvh * max_seq + k2) * hp.head_dim];
                            double d = 0;
                            for (int i = 0; i < hp.head_dim; i++) d += (double) qv[i] * kp[i];
                            sc[k2] = (float) d * scale;
                            m = std::max(m, sc[k2]);
                        }
                        double sum = 0;
                        for (int k2 = 0; k2 <= t; k2++) { sc[k2] = std::exp(sc[k2] - m); sum += sc[k2]; }
                        if (dbg_attn && il == 3 && t == 1 && h == 0) {
                            printf("REF attn t=1 h=0 weights:");
                            for (int k2 = 0; k2 <= t; k2++) printf(" %.4f", sc[k2] / sum);
                            const float * kk0 = &kcache_l[(size_t) kvh * max_seq * hp.head_dim];
                            const float * kk1 = kk0 + hp.head_dim;
                            printf("\n  REF q[0..2]=%.5f %.5f %.5f\n  REF k0[0..2]=%.5f %.5f %.5f\n  REF k1[0..2]=%.5f %.5f %.5f\n",
                                   qv[0], qv[1], qv[2], kk0[0], kk0[1], kk0[2], kk1[0], kk1[1], kk1[2]);
                        }
                        float * o = ao + h * hp.head_dim;
                        for (int i = 0; i < hp.head_dim; i++) {
                            double acc = 0;
                            for (int k2 = 0; k2 <= t; k2++)
                                acc += sc[k2] * vcache_l[((size_t) kvh * max_seq + k2) * hp.head_dim + i];
                            o[i] = (float) (acc / sum);
                        }
                    }
                    if (il == 3) snap("attn_raw-3", ao, hp.n_head * hp.head_dim);
                    for (int h = 0; h < hp.n_head; h++) {
                        const float * gatev = qbt + h * 2 * hp.head_dim + hp.head_dim;
                        float * o = ao + h * hp.head_dim;
                        for (int i = 0; i < hp.head_dim; i++)
                            o[i] *= 1.f / (1.f + std::exp(-gatev[i]));
                    }
                    if (il == 3) snap("attn_pregate-3", ao, hp.n_head * hp.head_dim);
                    std::vector<float> tmp(hp.n_embd);
                    matvec(L.wo, ao, tmp.data());
                    if (il == 3) snap("attn_output-3", tmp.data(), hp.n_embd);
                    for (int i = 0; i < hp.n_embd; i++) xt[i] = tmp[i] + xt[i];
                }
                // post attn norm + ffn
                rmsnorm(xt, L.post_attn_norm, xnt, hp.n_embd, hp.rms_eps);
                if (il == 0) snap("attn_post_norm-0", xnt, hp.n_embd);
                float * ft = ffn.data() + (size_t) t * 2 * hp.n_ff;
                matvec(L.ffn_gate, xnt, ft);
                matvec(L.ffn_up, xnt, ft + hp.n_ff);
                for (int i = 0; i < hp.n_ff; i++) ft[i] = silu(ft[i]) * ft[hp.n_ff + i];
                std::vector<float> tmp(hp.n_embd);
                matvec(L.ffn_down, ft, tmp.data());
                if (il == 0) snap("ffn_out-0", tmp.data(), hp.n_embd);
                for (int i = 0; i < hp.n_embd; i++) xt[i] = tmp[i] + xt[i];
                snap("l_out", xt, hp.n_embd);
            }
        }
    }

    static void rmsnorm_t(float * xx, const float * w, int n, float eps) {
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double) xx[i] * xx[i];
        const float scale = 1.0f / std::sqrt((float) (ss / n) + eps);
        for (int i = 0; i < n; i++) xx[i] *= scale * w[i];
    }

    static void rope(float * v, int pos, const hparams & hp) {
        const int half = hp.n_rot / 2;
        for (int i = 0; i < half; i++) {
            const float ang = pos * std::pow(hp.rope_base, -2.0f * i / hp.n_rot);
            const float c = std::cos(ang), s = std::sin(ang);
            const float x0 = v[i], x1 = v[i + half];
            v[i] = x0 * c - x1 * s;
            v[i + half] = x0 * s + x1 * c;
        }
    }

    // final norm + head for the last token
    void head() {
        const int t = n_tokens - 1;
        std::vector<float> hn(hp.n_embd);
        rmsnorm(&x[(size_t) t * hp.n_embd], m.output_norm, hn.data(), hp.n_embd, hp.rms_eps);
        matvec(m.tok_embd, hn.data(), logits.data());
    }
};

} // namespace si
