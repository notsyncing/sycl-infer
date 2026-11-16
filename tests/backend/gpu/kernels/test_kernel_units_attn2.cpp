// Model-free unit tests for the fused GDN variants, the DFlash draft kernels,
// and the oneDNN int8 prefill attention.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "dnnl_gemm.h"
#include "kernels.h"
#include "kv_type.h"
#include "quant.h"

using namespace si;
using namespace sycl;

static int g_fail = 0;
static void fail(const char * name, const char * what) {
    std::printf("%-34s FAIL (%s)\n", name, what);
    g_fail++;
}
static void check_f32(const char * name, const std::vector<float> & got, const std::vector<float> & exp, double tol,
                      double floor = 1.0) {
    if (got.size() != exp.size()) return fail(name, "size");
    double maxd = 0, maxv = 0;
    for (size_t i = 0; i < got.size(); i++) {
        maxd = std::max(maxd, (double)std::fabs(got[i] - exp[i]));
        maxv = std::max(maxv, (double)std::fabs(exp[i]));
    }
    const bool ok = maxd <= tol * std::max(floor, maxv);
    std::printf("%-34s n=%-6zu max|exp|=%.5f max|diff|=%.3e %s\n", name, got.size(), maxv, maxd, ok ? "OK" : "FAIL");
    if (!ok) g_fail++;
}
static float sigmoid_(float v) { return 1.0f / (1.0f + std::exp(-v)); }
static float rnd(int i, float scale = 2.0f) {
    const unsigned h = (unsigned)(i * 2654435761u + 1013904223u);
    return ((float)((h >> 8) & 0xffff) / 32768.0f - 1.0f) * scale;
}
template <typename T> static T * dev(queue & q, size_t n) { return malloc_device<T>(n, q); }
template <typename T> static void up(queue & q, T * d, const std::vector<T> & h) {
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
}
template <typename T> static void dn(queue & q, std::vector<T> & h, const T * d) {
    q.memcpy(h.data(), d, h.size() * sizeof(T)).wait();
}

// ---------------------------------------------------------------------------
static void test_gdn_conv(queue & q) {
    const int conv_dim = 384, hk = 128, kernel = 4, tpb = 32, slot = 1, n_slots = 2;
    const float eps = 1e-6f;
    std::vector<float> raw((size_t)tpb * conv_dim), w((size_t)kernel * conv_dim);
    for (size_t i = 0; i < raw.size(); i++) raw[i] = rnd((int)i);
    for (size_t i = 0; i < w.size(); i++) w[i] = rnd((int)i + 300) * 0.1f;
    std::vector<float> st0((size_t)n_slots * 3 * conv_dim), st1 = st0;
    for (size_t i = 0; i < st0.size(); i++) st0[i] = st1[i] = rnd((int)i + 700) * 0.1f;
    step_info info{};
    info.n_rows = 1;
    info.n_real = 1;
    info.tpb = tpb;
    info.pos[0] = 5;
    info.slot[0] = slot;
    info.active[0] = 1;
    info.n_real_row[0] = 1;
    float * d_raw = dev<float>(q, raw.size());
    float * dw = dev<float>(q, w.size());
    float * d_st0 = dev<float>(q, st0.size()), *d_st1 = dev<float>(q, st1.size());
    float * d_o0 = dev<float>(q, raw.size()), *d_o1 = dev<float>(q, raw.size());
    step_info * d_info = dev<step_info>(q, 1);
    up(q, d_raw, raw); up(q, dw, w); up(q, d_st0, st0); up(q, d_st1, st1); up(q, d_info, {info});
    gdn_conv_launch(q, d_raw, d_st0, dw, d_o0, d_info, conv_dim, kernel, hk, 1, eps, 1, 1, 0, tpb, false, false, {});
    conv_l2_launch(q, d_raw, d_st1, dw, d_o1, d_info, conv_dim, kernel, hk, 1, eps, 1, 1, 0, tpb, false);
    std::vector<float> o0(raw.size()), o1(raw.size());
    dn(q, o0, d_o0);
    dn(q, o1, d_o1);
    check_f32("gdn_conv vs conv_l2", o0, o1, 1e-6, 1e-3);

    // the folded state update (one token)
    std::vector<float> es = st0;
    const float * cs = st0.data() + (size_t)slot * 3 * conv_dim;
    float * esw = es.data() + (size_t)slot * 3 * conv_dim;
    for (int i = 0; i < conv_dim; i++) {
        esw[i] = cs[conv_dim + i];
        esw[conv_dim + i] = cs[2 * conv_dim + i];
        esw[2 * conv_dim + i] = raw[i];
    }
    gdn_conv_launch(q, d_raw, d_st0, dw, d_o0, d_info, conv_dim, kernel, hk, 1, eps, 1, 1, 0, tpb, false, true, {});
    std::vector<float> gst(st0.size());
    dn(q, gst, d_st0);
    check_f32("gdn_conv state update", gst, es, 0.0);
    free(d_raw, q); free(dw, q); free(d_st0, q); free(d_st1, q); free(d_o0, q); free(d_o1, q); free(d_info, q);
}

// ---------------------------------------------------------------------------
static void test_gdn_head(queue & q) {
    const int hd = 128, n_head = 2, n_kv = 1, conv_dim = 2 * n_kv * hd + n_head * hd, tpb = 32, slot = 0, n_slots = 1;
    const int n_heads_hd = n_head * hd;
    const float scale = 1.0f / std::sqrt((float)hd), eps = 1e-6f;
    std::vector<float> raw((size_t)tpb * conv_dim), cor((size_t)tpb * conv_dim);
    std::vector<float> alpha((size_t)tpb * n_head), beta((size_t)tpb * n_head), z((size_t)tpb * n_heads_hd);
    std::vector<float> dt(n_head), ssm(n_head), nw(hd), st0((size_t)n_slots * n_head * hd * hd);
    for (size_t i = 0; i < raw.size(); i++) raw[i] = rnd((int)i);
    for (size_t i = 0; i < cor.size(); i++) cor[i] = rnd((int)i + 200);
    for (size_t i = 0; i < alpha.size(); i++) alpha[i] = rnd((int)i + 400) * 0.1f;
    for (size_t i = 0; i < beta.size(); i++) beta[i] = rnd((int)i + 600) * 0.1f;
    for (size_t i = 0; i < z.size(); i++) z[i] = rnd((int)i + 800);
    for (int i = 0; i < n_head; i++) {
        dt[i] = rnd(i + 1000) * 0.1f;
        ssm[i] = -0.1f - 0.05f * i;
    }
    for (int i = 0; i < hd; i++) nw[i] = 0.5f + 0.1f * (i % 5);
    for (size_t i = 0; i < st0.size(); i++) st0[i] = rnd((int)i + 1200) * 0.01f;
    std::vector<float> cst((size_t)n_slots * 3 * conv_dim), cst_ref = cst;
    for (size_t i = 0; i < cst.size(); i++) cst[i] = cst_ref[i] = rnd((int)i + 1500) * 0.1f;

    step_info info{};
    info.n_rows = 1;
    info.n_real = 1;
    info.tpb = tpb;
    info.pos[0] = 0;
    info.slot[0] = slot;
    info.active[0] = 1;
    info.n_real_row[0] = 1;

    float * d_raw = dev<float>(q, raw.size()), *d_cor = dev<float>(q, cor.size());
    float * d_alpha = dev<float>(q, alpha.size()), *d_beta = dev<float>(q, beta.size());
    float * d_dt = dev<float>(q, n_head), *d_ssm = dev<float>(q, n_head), *d_nw = dev<float>(q, hd);
    float * d_z = dev<float>(q, z.size()), *d_st = dev<float>(q, st0.size()), *d_st_ref = dev<float>(q, st0.size());
    float * d_attn = dev<float>(q, (size_t)tpb * n_heads_hd), *d_attn_ref = dev<float>(q, (size_t)tpb * n_heads_hd);
    float * d_merged = dev<float>(q, (size_t)tpb * n_heads_hd), *d_merged_ref = dev<float>(q, (size_t)tpb * n_heads_hd);
    float * d_cst = dev<float>(q, cst.size()), *d_cst_ref = dev<float>(q, cst.size());
    step_info * d_info = dev<step_info>(q, 1);
    up(q, d_raw, raw); up(q, d_cor, cor); up(q, d_alpha, alpha); up(q, d_beta, beta); up(q, d_dt, dt);
    up(q, d_ssm, ssm); up(q, d_nw, nw); up(q, d_z, z); up(q, d_st, st0); up(q, d_st_ref, st0);
    up(q, d_cst, cst); up(q, d_cst_ref, cst_ref); up(q, d_info, {info});

    gdn_head_launch(q, d_raw, d_cst, 4, d_cor, d_alpha, d_dt, d_ssm, d_beta, d_st, d_attn, d_z, d_nw, d_merged,
                    d_info, hd, n_kv, n_head, conv_dim, scale, 1, 0, tpb, 1, {}, true, true, eps, false, 8);
    gdn_launch(q, d_cor, d_alpha, d_dt, d_ssm, d_beta, d_st_ref, d_attn_ref, d_info, hd, n_kv, n_head, conv_dim,
               scale, n_slots, 1, 0, tpb, 1);
    gated_norm_launch(q, d_attn_ref, d_z, d_nw, d_merged_ref, d_info, n_head, hd, eps, 1, 1, 0, false);
    conv_state_update_launch(q, d_raw, d_cst_ref, d_info, conv_dim, 4, 1, 0, tpb, 1, false);

    std::vector<float> attn((size_t)tpb * n_heads_hd), attn_ref(attn.size());
    std::vector<float> merged((size_t)tpb * n_heads_hd), merged_ref(merged.size());
    std::vector<float> gst(st0.size()), gst_ref(st0.size()), gc(cst.size()), gc_ref(cst.size());
    dn(q, attn, d_attn); dn(q, attn_ref, d_attn_ref);
    dn(q, merged, d_merged); dn(q, merged_ref, d_merged_ref);
    dn(q, gst, d_st); dn(q, gst_ref, d_st_ref);
    dn(q, gc, d_cst); dn(q, gc_ref, d_cst_ref);
    // Direct scalar reference for the n_real == 1, C == 4 mapping (the shape
    // both launches use here): state row col0+c, all 128 columns.
    std::vector<float> esca((size_t)tpb * n_heads_hd, 0.f), escs = st0;
    for (int h = 0; h < n_head; h++) {
        const int qk_head = h % n_kv;
        const float A = ssm[h];
        const float bt = sigmoid_(beta[h]);
        const float g = std::exp(A * std::log(1.f + std::exp(alpha[h] + dt[h])));
        for (int col0 = 0; col0 < hd; col0 += 4) {
            for (int c = 0; c < 4; c++) {
                const int row = col0 + c;
                float * sr = &escs[((size_t)h * hd + row) * hd];
                const float * kr = &cor[(size_t)n_kv * hd + (size_t)qk_head * hd]; // k_off = n_kv*hd
                const float * qr = &cor[0];
                double dot = 0, qd = 0;
                for (int j = 0; j < hd; j++) dot += (double)sr[j] * kr[j];
                const float del = (cor[(size_t)2 * n_kv * hd + (size_t)h * hd + row] - g * (float)dot) * bt;
                for (int j = 0; j < hd; j++) sr[j] = g * sr[j] + kr[j] * del;
                for (int j = 0; j < hd; j++) qd += (double)sr[j] * qr[j];
                esca[(size_t)h * hd + row] = (float)qd * scale;
            }
        }
    }
    check_f32("gdn_launch vs scalar", attn_ref, esca, 1e-5, 1e-3);
    check_f32("gdn_head attn", attn, attn_ref, 1e-6, 1e-3);
    check_f32("gdn_head merged", merged, merged_ref, 1e-6, 1e-3);
    check_f32("gdn_head state", gst, gst_ref, 1e-6, 1e-3);
    check_f32("gdn_head conv state", gc, gc_ref, 1e-6, 1e-3);

    free(d_raw, q); free(d_cor, q); free(d_alpha, q); free(d_beta, q); free(d_dt, q); free(d_ssm, q);
    free(d_nw, q); free(d_z, q); free(d_st, q); free(d_st_ref, q); free(d_attn, q); free(d_attn_ref, q);
    free(d_merged, q); free(d_merged_ref, q); free(d_cst, q); free(d_cst_ref, q); free(d_info, q);
}

// ---------------------------------------------------------------------------
static void test_df_rope(queue & q) {
    const int n_rows = 2, n_head = 2, n_kv = 1, hd = 32, n_rot = 32, ring = 4, row_stride = n_head * hd;
    const float base = 10000.f, eps = 1e-6f;
    std::vector<float> qv((size_t)n_rows * row_stride), k((size_t)n_rows * row_stride), v((size_t)n_rows * row_stride);
    std::vector<float> qn(hd), kn(hd);
    for (size_t i = 0; i < qv.size(); i++) qv[i] = rnd((int)i);
    for (size_t i = 0; i < k.size(); i++) k[i] = rnd((int)i + 300);
    for (size_t i = 0; i < v.size(); i++) v[i] = rnd((int)i + 600);
    for (int i = 0; i < hd; i++) {
        qn[i] = 0.5f + 0.1f * (i % 3);
        kn[i] = 0.7f + 0.05f * (i % 5);
    }
    std::vector<float> kp((size_t)ring * n_kv * hd, 0.f), vp((size_t)ring * n_kv * hd, 0.f);
    std::vector<int32_t> pos{5, 7};
    float * d_q = dev<float>(q, qv.size()), *d_k = dev<float>(q, k.size()), *d_v = dev<float>(q, v.size());
    float * d_qn = dev<float>(q, hd), *d_kn = dev<float>(q, hd);
    float * d_kp = dev<float>(q, kp.size()), *d_vp = dev<float>(q, vp.size());
    int32_t * d_pos = dev<int32_t>(q, n_rows);
    up(q, d_q, qv); up(q, d_k, k); up(q, d_v, v); up(q, d_qn, qn); up(q, d_kn, kn); up(q, d_kp, kp); up(q, d_vp, vp);
    up(q, d_pos, pos);
    df_qknorm_rope_store_launch(q, d_q, d_k, d_v, d_qn, d_kn, d_kp, d_vp, d_pos, n_rows, n_head, n_kv, hd, n_rot, base,
                                eps, ring, 4, true, row_stride, true);
    std::vector<float> gq(qv.size()), gk(k.size()), gv(v.size()), gkp(kp.size()), gvp(vp.size());
    dn(q, gq, d_q); dn(q, gk, d_k); dn(q, gv, d_v); dn(q, gkp, d_kp); dn(q, gvp, d_vp);
    std::vector<float> eq = qv, ek = k, ev = v, ekp(kp), evp(vp);
    auto rope = [&](float * h, int p) {
        const int half = n_rot / 2;
        for (int i = 0; i < half; i++) {
            const float ang = (float)p * std::exp2(-2.f * i / (float)n_rot * std::log2(base));
            const float c = std::cos(ang), sn = std::sin(ang);
            const float x0 = h[i], x1 = h[i + half];
            h[i] = x0 * c - x1 * sn;
            h[i + half] = x0 * sn + x1 * c;
        }
    };
    for (int r = 0; r < n_rows; r++) {
        for (int h = 0; h < n_head; h++) {
            float * p = &eq[(size_t)r * row_stride + (size_t)h * hd];
            double ss = 0;
            for (int i = 0; i < hd; i++) ss += (double)p[i] * p[i];
            const float inv = 1.0f / std::sqrt((float)(ss / hd) + eps);
            for (int i = 0; i < hd; i++) p[i] *= inv * qn[i];
            rope(p, pos[r]);
        }
        for (int h = 0; h < n_kv; h++) {
            float * p = &ek[(size_t)r * row_stride + (size_t)h * hd];
            double ss = 0;
            for (int i = 0; i < hd; i++) ss += (double)p[i] * p[i];
            const float inv = 1.0f / std::sqrt((float)(ss / hd) + eps);
            for (int i = 0; i < hd; i++) p[i] *= inv * kn[i];
            rope(p, pos[r]);
            const size_t dst = ((size_t)(pos[r] % ring) * n_kv + h) * hd;
            for (int i = 0; i < hd; i++) ekp[dst + i] = p[i];
            const float * src = &v[(size_t)r * row_stride + (size_t)h * hd];
            for (int i = 0; i < hd; i++) evp[dst + i] = src[i];
        }
    }
    check_f32("df_qknorm_rope q", gq, eq, 1e-5, 1e-3);
    check_f32("df_qknorm_rope k", gk, ek, 1e-5, 1e-3);
    check_f32("df_qknorm_rope kpool", gkp, ekp, 1e-5, 1e-3);
    check_f32("df_qknorm_rope vpool", gvp, evp, 0.0);
    (void)gv; (void)ev;
    free(d_q, q); free(d_k, q); free(d_v, q); free(d_qn, q); free(d_kn, q); free(d_kp, q); free(d_vp, q);
    free(d_pos, q);
}

static void test_df_attn(queue & q) {
    const int n_rows = 2, n_blk = 3, n_head = 2, n_kv = 1, hd = 128, ring = 8, swa = 4, n_splits = 2;
    const int row_stride = n_head * hd, pstride = 2 + hd;
    const float scale = 1.0f / std::sqrt((float)hd);
    std::vector<float> qbuf((size_t)n_rows * row_stride), kp((size_t)ring * n_kv * hd), vp(kp.size());
    for (size_t i = 0; i < qbuf.size(); i++) qbuf[i] = rnd((int)i);
    for (size_t i = 0; i < kp.size(); i++) {
        kp[i] = rnd((int)i + 300);
        vp[i] = rnd((int)i + 600);
    }
    std::vector<int32_t> pos{5};
    const int pos0 = pos[0];
    float * d_q = dev<float>(q, qbuf.size()), *d_kp = dev<float>(q, kp.size()), *d_vp = dev<float>(q, vp.size());
    float * d_part = dev<float>(q, (size_t)n_rows * n_head * n_splits * pstride);
    int32_t * d_pos = dev<int32_t>(q, 1);
    up(q, d_q, qbuf); up(q, d_kp, kp); up(q, d_vp, vp); up(q, d_pos, pos);
    df_attn_launch(q, d_q, d_kp, d_vp, d_part, d_pos, n_rows, n_blk, n_head, n_kv, hd, ring, swa, n_splits, scale, 4,
                   row_stride);
    std::vector<float> part((size_t)n_rows * n_head * n_splits * pstride);
    dn(q, part, d_part);
    std::vector<float> epart(part.size(), 0.f);
    for (int r = 0; r < n_rows; r++) {
        const int p = pos0 + r;
        const int hi = pos0 + n_blk - 1;
        const int lo = std::max(0, p - swa + 1);
        const int nk = hi - lo + 1;
        const int per = (nk + n_splits - 1) / n_splits;
        for (int h = 0; h < n_head; h++) {
            const int kh = h / (n_head / n_kv);
            for (int s = 0; s < n_splits; s++) {
                const int k0 = lo + s * per, k1 = std::min(hi + 1, k0 + per);
                float mx = -INFINITY, ls = 0;
                std::vector<float> acc(hd, 0.f);
                for (int kk = k0; kk < k1; kk++) {
                    const int cell = (kk % ring) * n_kv + kh;
                    float dot = 0;
                    for (int d = 0; d < hd; d++) dot += qbuf[(size_t)r * row_stride + (size_t)h * hd + d] * kp[(size_t)cell * hd + d];
                    dot *= scale;
                    const float nm = std::max(mx, dot);
                    const float corr = std::exp(mx - nm), pw = std::exp(dot - nm);
                    ls = ls * corr + pw;
                    for (int d = 0; d < hd; d++) acc[d] = acc[d] * corr + pw * vp[(size_t)cell * hd + d];
                    mx = nm;
                }
                float * o = &epart[(((size_t)r * n_head + h) * n_splits + s) * pstride];
                o[0] = ls > 0 ? mx : -INFINITY;
                o[1] = ls;
                for (int d = 0; d < hd; d++) o[2 + d] = acc[d];
            }
        }
    }
    check_f32("df_attn partials", part, epart, 1e-5, 1e-3);

    // combine, with and without a sink
    std::vector<float> sinks{n_head, 0.f};
    float * d_out = dev<float>(q, (size_t)n_rows * n_head * hd);
    float * d_sinks = dev<float>(q, n_head);
    up(q, d_sinks, sinks);
    df_attn_combine_launch(q, d_part, d_out, n_rows, n_head, hd, n_splits, d_sinks);
    std::vector<float> out((size_t)n_rows * n_head * hd), eout(out.size());
    dn(q, out, d_out);
    for (int g = 0; g < n_rows * n_head; g++) {
        const float * base = &epart[(size_t)g * n_splits * pstride];
        float mx = -INFINITY;
        for (int s = 0; s < n_splits; s++) mx = std::max(mx, base[(size_t)s * pstride]);
        float ls = 0;
        std::vector<float> acc(hd, 0.f);
        for (int s = 0; s < n_splits; s++) {
            const float * pp = base + (size_t)s * pstride;
            const float wgt = pp[1] > 0 ? std::exp(pp[0] - mx) : 0.f;
            ls += pp[1] * wgt;
            for (int d = 0; d < hd; d++) acc[d] += pp[2 + d] * wgt;
        }
        const float h0 = sinks[g / n_rows];
        const float sink = 1.f / (1.f + std::exp(h0 - mx - std::log(ls)));
        for (int d = 0; d < hd; d++) eout[(size_t)g * hd + d] = (ls > 0 ? acc[d] / ls : 0.f) * sink;
    }
    check_f32("df_attn_combine", out, eout, 1e-6, 1e-3);
    free(d_q, q); free(d_kp, q); free(d_vp, q); free(d_part, q); free(d_pos, q); free(d_out, q); free(d_sinks, q);
}

// ---------------------------------------------------------------------------
static void test_df_sel(queue & q) {
    const int K = 16, rank = 256, n_blk = 3, n_vocab = 8;
    const int row = K + K * K;
    std::vector<int32_t> ids((size_t)n_blk * K);
    for (size_t i = 0; i < ids.size(); i++) ids[i] = (int32_t)((i * 3) % n_vocab);
    std::vector<float> vals((size_t)n_blk * K);
    for (size_t i = 0; i < vals.size(); i++) vals[i] = rnd((int)i) * 0.1f;
    std::vector<float> gate((size_t)n_blk * rank);
    for (size_t i = 0; i < gate.size(); i++) gate[i] = 0.1f + 0.001f * (float)(i % 17);
    std::vector<uint8_t> pv((size_t)n_vocab * (rank / 2)), nv(pv.size());
    std::vector<uint16_t> ps((size_t)(rank / 32) * n_vocab), ns(ps.size()), po(ps.size()), no(ps.size());
    for (size_t i = 0; i < pv.size(); i++) {
        pv[i] = (uint8_t)(i * 37 + 11);
        nv[i] = (uint8_t)(i * 53 + 7);
    }
    for (size_t i = 0; i < ps.size(); i++) {
        ps[i] = ggml_float_to_half(0.01f * (float)(i % 5 + 1));
        ns[i] = ggml_float_to_half(0.02f * (float)(i % 3 + 1));
        po[i] = ggml_float_to_half(-0.01f * (float)(i % 4));
        no[i] = ggml_float_to_half(-0.005f * (float)(i % 2));
    }
    const int32_t anchor = 3;
    float * d_lat = dev<float>(q, (size_t)n_blk * row);
    int32_t * d_ids = dev<int32_t>(q, ids.size());
    float * d_vals = dev<float>(q, vals.size()), *d_gate = dev<float>(q, gate.size());
    uint8_t * d_pv = dev<uint8_t>(q, pv.size()), *d_nv = dev<uint8_t>(q, nv.size());
    uint16_t * d_ps = dev<uint16_t>(q, ps.size()), *d_ns = dev<uint16_t>(q, ns.size());
    uint16_t * d_po = dev<uint16_t>(q, po.size()), *d_no = dev<uint16_t>(q, no.size());
    up(q, d_lat, std::vector<float>((size_t)n_blk * row, -1.f));
    up(q, d_ids, ids); up(q, d_vals, vals); up(q, d_gate, gate);
    up(q, d_pv, pv); up(q, d_nv, nv); up(q, d_ps, ps); up(q, d_ns, ns); up(q, d_po, po); up(q, d_no, no);
    df_sel_launch(q, d_ids, d_vals, d_gate, d_pv, d_ps, d_po, d_nv, d_ns, d_no, d_lat, anchor, n_blk, n_vocab, rank, K);
    std::vector<float> lat((size_t)n_blk * row);
    dn(q, lat, d_lat);

    auto val = [&](const std::vector<uint8_t> & v, const std::vector<uint16_t> & s, const std::vector<uint16_t> & o,
                   int id, int d) {
        const float st = ggml_half_to_float(s[(size_t)(d / 32) * n_vocab + id]);
        const float of = ggml_half_to_float(o[(size_t)(d / 32) * n_vocab + id]);
        const uint8_t by = v[(size_t)id * (rank / 2) + (d >> 1)];
        const int qv = (d & 1) ? (by >> 4) : (by & 0xF);
        return st * (float)qv + of;
    };
    bool ok = true;
    for (int i = 1; i < n_blk; i++) {
        for (int c = 0; c < K; c++) {
            if (lat[(size_t)i * row + c] != (float)ids[(size_t)i * K + c]) ok = false;
        }
        for (int p = 0; p < K; p++) {
            const int32_t ia = (i == 1) ? anchor : ids[(size_t)(i - 1) * K + p];
            for (int c = 0; c < K; c++) {
                const int32_t ib = ids[(size_t)i * K + c];
                double s = 0;
                for (int d = 0; d < rank; d++) {
                    const float a = (ia >= 0 && ia < n_vocab)
                                        ? val(pv, ps, po, ia, d) * gate[(size_t)i * rank + (d & ~3)]
                                        : 0.f;
                    const float b = (ib >= 0 && ib < n_vocab) ? val(nv, ns, no, ib, d) : 0.f;
                    s += (double)a * b;
                }
                s += vals[(size_t)i * K + c];
                const float got = lat[(size_t)i * row + K + p * K + c];
                if (std::fabs(got - (float)s) > 1e-5f * std::max(1.f, std::fabs((float)s))) ok = false;
            }
        }
    }
    std::printf("%-34s %s\n", "df_sel_launch", ok ? "OK" : "FAIL");
    if (!ok) g_fail++;
    free(d_lat, q); free(d_ids, q); free(d_vals, q); free(d_gate, q); free(d_pv, q); free(d_nv, q);
    free(d_ps, q); free(d_ns, q); free(d_po, q); free(d_no, q);
}

// ---------------------------------------------------------------------------
static void test_attn_xmx(queue & q) {
    const int n_head = 2, n_kv = 1, hd = 256, n_rows = 1, n_real = 2, tpb = 32, max_blocks = 2;
    const int pos = 62; // 64 keys
    const float scale = 1.0f / std::sqrt((float)hd);
    dnnl_gemm dg(q);
    dnnl_register_queue(q, &dg);
    kv_dtype_set_kv(kv_dtype_t::i8, kv_dtype_t::i8);
    // i8 pool: [block][kv head][token][hd] + a f16 scale per 32 values
    const size_t pool = (size_t)max_blocks * n_kv * 32 * hd;
    std::vector<int8_t> kp(pool), vp(pool);
    std::vector<uint16_t> ksc(pool / 32), vsc(pool / 32);
    for (size_t i = 0; i < pool; i++) {
        kp[i] = (int8_t)((int)(i * 7) % 255 - 127);
        vp[i] = (int8_t)((int)(i * 11) % 255 - 127);
    }
    for (size_t i = 0; i < ksc.size(); i++) {
        ksc[i] = ggml_float_to_half(0.01f);
        vsc[i] = ggml_float_to_half(0.01f);
    }
    std::vector<float> qbuf((size_t)n_real * (n_head * 2 * hd));
    for (size_t i = 0; i < qbuf.size(); i++) qbuf[i] = rnd((int)i);
    step_info info{};
    info.n_rows = n_rows;
    info.n_real = n_real;
    info.tpb = tpb;
    info.pos[0] = pos;
    info.slot[0] = 0;
    info.active[0] = 1;
    info.n_real_row[0] = n_real;
    int8_t * d_kp = dev<int8_t>(q, pool), *d_vp = dev<int8_t>(q, pool);
    uint16_t * d_ksc = dev<uint16_t>(q, ksc.size()), *d_vsc = dev<uint16_t>(q, vsc.size());
    float * d_q = dev<float>(q, qbuf.size());
    float * d_o = dev<float>(q, (size_t)n_real * n_head * hd);
    float * d_c = dev<float>(q, (size_t)n_real * n_head * hd);
    int32_t * d_tab = dev<int32_t>(q, max_blocks);
    // attn_xmx reads info->active/row_nr on the HOST to size its query tile,
    // so step_info is host USM here (the engine allocates it that way).
    step_info * d_info = malloc_host<step_info>(1, q);
    *d_info = info;
    up(q, d_kp, kp); up(q, d_vp, vp); up(q, d_ksc, ksc); up(q, d_vsc, vsc); up(q, d_q, qbuf);
    up(q, d_tab, std::vector<int32_t>{0, 1});
    float * d_p = dev<float>(q, (size_t)n_real * n_head * 1 * (2 + hd));
    const bool ok = attn_xmx_launch(q, d_q, nullptr, d_kp, d_vp, d_p, d_tab, n_head, n_kv, hd, 1, d_info, scale,
                                    max_blocks, n_rows, n_real, d_o, d_ksc, d_vsc);
    if (!ok) {
        fail("attn_xmx_launch", "declined for an in-scope i8 call");
    } else {
        attn_launch(q, d_q, nullptr, d_kp, d_vp, d_c, d_tab, n_head, n_kv, hd, 1, d_info, scale, max_blocks, n_rows,
                    n_real, d_c, -1, d_ksc, d_vsc);
        std::vector<float> g((size_t)n_real * n_head * hd), e(g.size());
        dn(q, g, d_o);
        dn(q, e, d_c);
        check_f32("attn_xmx vs classic", g, e, 1e-2, 1e-2);
    }
    free(d_kp, q); free(d_vp, q); free(d_ksc, q); free(d_vsc, q); free(d_q, q); free(d_o, q); free(d_c, q);
    free(d_tab, q); free(d_info, q); free(d_p, q);
    dnnl_register_queue(q, nullptr);
}

int main() {
    setenv("PF_GDN_COLS", "8", 1);
    setenv("PF_GDN_COLS_MIN", "1", 1);
    setenv("PF_GDN_VEC", "0", 1);
    setenv("PF_ATTN_XMX_MIN", "1", 1);
    try {
        queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
        std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        test_gdn_conv(q);
        test_gdn_head(q);
        test_df_rope(q);
        test_df_attn(q);
        test_df_sel(q);
        test_attn_xmx(q);
    } catch (const sycl::exception & e) {
        std::printf("SYCL exception: %s\n", e.what());
        return 1;
    }
    if (g_fail) {
        std::printf("test_kernel_units_attn2: FAILURES: %d\n", g_fail);
        return 1;
    }
    std::printf("test_kernel_units_attn2: all units OK\n");
    return 0;
}
