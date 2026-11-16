// Hermetic unit tests for the CPU attention/convolutions/RoPE kernels.
//
// The references are scalar loops written here; the paged layout and the
// storage dtypes come from the same public helpers the kernels use, so a
// failure names either the kernel's arithmetic or its indexing, never the
// harness's idea of the layout.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "kernels/common.h"
#include "cpu_types.h"
#include "common/rope.h"

using namespace si;

static int g_fail = 0;

static void fail(const char * name, const char * what) {
    std::printf("%-30s FAIL (%s)\n", name, what);
    g_fail++;
}

static void check_f32(const char * name, const std::vector<float> & got, const std::vector<float> & exp, double tol,
                      double floor = 1.0) {
    if (got.size() != exp.size()) {
        fail(name, "size");
        return;
    }
    double maxd = 0, maxv = 0;
    for (size_t i = 0; i < got.size(); i++) {
        maxd = std::max(maxd, (double)std::fabs(got[i] - exp[i]));
        maxv = std::max(maxv, (double)std::fabs(exp[i]));
    }
    const bool ok = maxd <= tol * std::max(floor, maxv);
    std::printf("%-30s n=%-6zu max|exp|=%.5f max|diff|=%.3e %s\n", name, got.size(), maxv, maxd, ok ? "OK" : "FAIL");
    if (!ok) g_fail++;
}

static float rnd(int i, float scale = 2.0f) {
    const unsigned h = (unsigned)(i * 2654435761u + 1013904223u);
    return ((float)((h >> 8) & 0xffff) / 32768.0f - 1.0f) * scale;
}
static float sigmoid_(float v) {
    return 1.0f / (1.0f + std::exp(-v));
}
static float silu_(float v) {
    return v / (1.0f + std::exp(-v));
}

// ---------------------------------------------------------------------------
// conv_l2 + conv_state_update.  Each call is compared with a scalar loop that
// reads the same ring of taps; the state update is then checked separately.
static void test_conv() {
    const int conv_dim = 6, group_dim = 3, kernel_size = 4, n_k_heads = 1;
    const int tpb = 32, n_real = 3, n_rows = 1, slot = 1, n_slots = 2;
    const float eps = 1e-6f;
    // Two rows of raw data so a cross-row tap can look back into row 0.
    std::vector<float> raw((size_t)2 * tpb * conv_dim);
    for (size_t i = 0; i < raw.size(); i++) raw[i] = rnd((int)i);
    std::vector<float> cw((size_t)kernel_size * conv_dim);
    for (size_t i = 0; i < cw.size(); i++) cw[i] = rnd((int)i + 300) * 0.3f;

    cpu_step_info info{};
    info.n_rows = 1;
    info.n_real = n_real;
    info.tpb = tpb;
    info.pos[0] = 0;
    info.slot[0] = slot;
    info.active[0] = 1;

    for (int cross = 0; cross < 2; cross++) {
        std::vector<float> state((size_t)n_slots * 3 * conv_dim);
        // seed the window so the negative-position taps are live
        for (size_t i = 0; i < state.size(); i++) state[i] = rnd((int)i + 700) * 0.1f;
        std::vector<float> out((size_t)tpb * conv_dim, -9.f), exp((size_t)tpb * conv_dim, -9.f);
        cpu_conv_l2(raw.data(), state.data(), cw.data(), out.data(), &info, conv_dim, kernel_size, group_dim,
                    n_k_heads, eps, n_rows, n_real, 0, tpb, cross != 0);
        const float * cstate = state.data() + (size_t)slot * 3 * conv_dim;
        for (int t = 0; t < n_real; t++) {
            for (int ch = 0; ch < conv_dim; ch++) {
                float v = 0;
                for (int i = 0; i < 4; i++) {
                    const float wv = cw[(size_t)i + (size_t)kernel_size * ch];
                    const int p = t + i - (kernel_size - 1);
                    if (p >= 0) {
                        const int q = cross ? p : p;
                        v += wv * raw[(size_t)q * conv_dim + ch];
                    } else {
                        v += wv * cstate[(size_t)(p + kernel_size - 1) * conv_dim + ch];
                    }
                }
                exp[(size_t)t * conv_dim + ch] = silu_(v);
            }
            // the first 2*n_k_heads groups are L2-normalised
            const int n_groups = conv_dim / group_dim;
            for (int grp = 0; grp < n_groups && grp < 2 * n_k_heads; grp++) {
                double ss = 0;
                for (int lane = 0; lane < group_dim; lane++) {
                    const float v = exp[(size_t)t * conv_dim + grp * group_dim + lane];
                    ss += (double)v * v;
                }
                const float inv = 1.0f / std::fmax(std::sqrt((float)ss), eps);
                for (int lane = 0; lane < group_dim; lane++) {
                    exp[(size_t)t * conv_dim + grp * group_dim + lane] *= inv;
                }
            }
        }
        check_f32(cross ? "cpu_conv_l2 cross_row" : "cpu_conv_l2", out, exp, 1e-5, 1e-3);
    }

    // conv_state_update: the last three tokens' raw values, per slot.
    std::vector<float> state((size_t)n_slots * 3 * conv_dim, 0.f);
    cpu_conv_state_update(raw.data(), state.data(), &info, conv_dim, kernel_size, n_rows, 0, tpb, -1, false, {});
    std::vector<float> exp((size_t)n_slots * 3 * conv_dim, 0.f);
    for (int i = 0; i < conv_dim; i++) {
        exp[(size_t)slot * 3 * conv_dim + (size_t)0 * conv_dim + i] = raw[(size_t)(n_real - 3) * conv_dim + i];
        exp[(size_t)slot * 3 * conv_dim + (size_t)1 * conv_dim + i] = raw[(size_t)(n_real - 2) * conv_dim + i];
        exp[(size_t)slot * 3 * conv_dim + (size_t)2 * conv_dim + i] = raw[(size_t)(n_real - 1) * conv_dim + i];
    }
    check_f32("cpu_conv_state_update", state, exp, 0.0);
}

// ---------------------------------------------------------------------------
// paged attention.  f32 KV first (the exact reference), then i8 (the scale
// plane).  Both the fused n_splits==1 form and the split+combine form.
static void test_attn() {
    const int n_head = 2, n_head_kv = 1, head_dim = 4, tpb = 32, n_rows = 1, n_real = 1;
    const int max_blocks = 2, pos = 2; // 3 keys, straddling a block? no: block 0
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int qstride = n_head * 2 * head_dim;
    const int kvstride = n_head_kv * head_dim;

    std::vector<float> qbuf((size_t)tpb * qstride), kbuf((size_t)tpb * kvstride), vbuf((size_t)tpb * kvstride);
    for (size_t i = 0; i < qbuf.size(); i++) qbuf[i] = rnd((int)i);
    for (size_t i = 0; i < kbuf.size(); i++) kbuf[i] = rnd((int)i + 1000);
    for (size_t i = 0; i < vbuf.size(); i++) vbuf[i] = rnd((int)i + 2000);
    int32_t tables[2] = {0, 1};

    cpu_step_info info{};
    info.n_rows = n_rows;
    info.n_real = n_real;
    info.tpb = tpb;
    info.pos[0] = pos;
    info.slot[0] = 0;
    info.active[0] = 1;

    // ---- f32 pool
    const size_t pool_f = (size_t)kCpuBlk * n_head_kv * head_dim * 3;
    std::vector<float> kp(pool_f), vp(pool_f);
    for (int kk = 0; kk <= pos; kk++) {
        const int kb = tables[kk / kCpuBlk], ko = kk % kCpuBlk;
        for (int h = 0; h < n_head_kv; h++) {
            const size_t unit = (size_t)kb * n_head_kv + h;
            for (int d = 0; d < head_dim; d++) {
                kp[kv_row_off(unit, ko, head_dim) + d] = rnd(5000 + kk * 97 + h * 31 + d);
                vp[kv_row_off(unit, ko, head_dim) + d] = rnd(7000 + kk * 53 + h * 17 + d);
            }
        }
    }
    // expected per head, f32
    std::vector<float> exp((size_t)tpb * n_head * head_dim, -9.f);
    for (int h = 0; h < n_head; h++) {
        const int kvh = (h * n_head_kv) / n_head;
        std::vector<float> sc(pos + 1);
        float mx = -INFINITY;
        for (int kk = 0; kk <= pos; kk++) {
            const int kb = tables[kk / kCpuBlk], ko = kk % kCpuBlk;
            const size_t off = kv_row_off((size_t)kb * n_head_kv + kvh, ko, head_dim);
            float dot = 0;
            for (int d = 0; d < head_dim; d++) dot += qbuf[(size_t)h * 2 * head_dim + d] * kp[off + d];
            sc[kk] = dot * scale;
            mx = std::max(mx, sc[kk]);
        }
        float l = 0;
        std::vector<float> acc(head_dim, 0.f);
        for (int kk = 0; kk <= pos; kk++) {
            const float e = std::exp(sc[kk] - mx);
            l += e;
            const int kb = tables[kk / kCpuBlk], ko = kk % kCpuBlk;
            const size_t off = kv_row_off((size_t)kb * n_head_kv + kvh, ko, head_dim);
            for (int d = 0; d < head_dim; d++) acc[d] += e * vp[off + d];
        }
        for (int d = 0; d < head_dim; d++) {
            exp[(size_t)h * head_dim + d] = (acc[d] / l) * sigmoid_(qbuf[(size_t)h * 2 * head_dim + head_dim + d]);
        }
    }

    std::vector<float> out((size_t)tpb * n_head * head_dim, -9.f);
    cpu_attn(qbuf.data(), nullptr, kp.data(), vp.data(), nullptr, tables, n_head, n_head_kv, head_dim, 1, &info,
             scale, max_blocks, n_rows, n_real, out.data(), cpu_kv_dtype::f32, cpu_kv_dtype::f32, nullptr, nullptr);
    std::vector<float> got(out.begin(), out.begin() + (size_t)n_head * head_dim);
    std::vector<float> ex(exp.begin(), exp.begin() + (size_t)n_head * head_dim);
    check_f32("cpu_attn f32 fused", got, ex, 1e-5, 1e-3);

    // split == 2, then combine
    const int n_splits = 2, pstride = 2 + head_dim;
    std::vector<float> part((size_t)tpb * n_head * n_splits * pstride, 0.f);
    cpu_attn(qbuf.data(), nullptr, kp.data(), vp.data(), part.data(), tables, n_head, n_head_kv, head_dim, n_splits,
             &info, scale, max_blocks, n_rows, n_real, nullptr, cpu_kv_dtype::f32, cpu_kv_dtype::f32, nullptr, nullptr);
    std::vector<float> outc((size_t)tpb * n_head * head_dim, -9.f);
    cpu_attn_combine(part.data(), qbuf.data(), outc.data(), &info, n_head, head_dim, n_splits, n_rows, n_real);
    std::vector<float> gotc(outc.begin(), outc.begin() + (size_t)n_head * head_dim);
    check_f32("cpu_attn f32 split+combine", gotc, ex, 1e-5, 1e-3);

    // ---- i8 pool, head_dim 32 (one scale group)
    const int hd8 = 32;
    const int qstride8 = n_head * 2 * hd8;
    const int kvstride8 = n_head_kv * hd8;
    std::vector<float> q8((size_t)tpb * qstride8), k8f((size_t)tpb * kvstride8), v8f((size_t)tpb * kvstride8);
    for (size_t i = 0; i < q8.size(); i++) q8[i] = rnd((int)i + 11000);
    for (size_t i = 0; i < k8f.size(); i++) k8f[i] = rnd((int)i + 12000);
    for (size_t i = 0; i < v8f.size(); i++) v8f[i] = rnd((int)i + 13000);
    std::vector<int8_t> k8(pool_f * 4), v8(pool_f * 4);
    std::vector<uint16_t> ksc(pool_f / kCpuI8Q * 4), vsc(pool_f / kCpuI8Q * 4);
    auto quant = [](float v, float amax) {
        const float sc = amax > 0 ? amax / 127.0f : 1.0f;
        return (int8_t)std::max(-127.0f, std::min(127.0f, std::round(v / sc)));
    };
    for (int kk = 0; kk <= pos; kk++) {
        const int kb = tables[kk / kCpuBlk], ko = kk % kCpuBlk;
        for (int h = 0; h < n_head_kv; h++) {
            const size_t unit = (size_t)kb * n_head_kv + h;
            float kmax = 0, vmax = 0;
            for (int d = 0; d < hd8; d++) {
                kmax = std::max(kmax, std::fabs(k8f[(size_t)kk * hd8 + d]));
                vmax = std::max(vmax, std::fabs(v8f[(size_t)kk * hd8 + d]));
            }
            const float ks = kmax > 0 ? kmax / 127.0f : 1.0f;
            const float vs = vmax > 0 ? vmax / 127.0f : 1.0f;
            for (int d = 0; d < hd8; d++) {
                k8[kv_row_off(unit, ko, hd8) + d] = quant(k8f[(size_t)kk * hd8 + d], kmax);
                v8[kv_row_off(unit, ko, hd8) + d] = quant(v8f[(size_t)kk * hd8 + d], vmax);
            }
            ksc[kv_scale_off(unit, ko, hd8)] = ggml_float_to_half(ks);
            vsc[kv_scale_off(unit, ko, hd8)] = ggml_float_to_half(vs);
        }
    }
    std::vector<float> exp8((size_t)n_head * hd8, 0.f);
    for (int h = 0; h < n_head; h++) {
        const int kvh = (h * n_head_kv) / n_head;
        std::vector<float> sc(pos + 1);
        float mx = -INFINITY;
        for (int kk = 0; kk <= pos; kk++) {
            const int kb = tables[kk / kCpuBlk], ko = kk % kCpuBlk;
            const size_t unit = (size_t)kb * n_head_kv + kvh;
            const float s = ggml_half_to_float(ksc[kv_scale_off(unit, ko, hd8)]);
            float dot = 0;
            for (int d = 0; d < hd8; d++) dot += q8[(size_t)h * 2 * hd8 + d] * (float)k8[kv_row_off(unit, ko, hd8) + d] * s;
            sc[kk] = dot * scale;
            mx = std::max(mx, sc[kk]);
        }
        float l = 0;
        std::vector<float> acc(hd8, 0.f);
        for (int kk = 0; kk <= pos; kk++) {
            const float e = std::exp(sc[kk] - mx);
            l += e;
            const int kb = tables[kk / kCpuBlk], ko = kk % kCpuBlk;
            const size_t unit = (size_t)kb * n_head_kv + kvh;
            const float s = ggml_half_to_float(vsc[kv_scale_off(unit, ko, hd8)]);
            for (int d = 0; d < hd8; d++) acc[d] += e * (float)v8[kv_row_off(unit, ko, hd8) + d] * s;
        }
        for (int d = 0; d < hd8; d++) {
            exp8[(size_t)h * hd8 + d] = (acc[d] / l) * sigmoid_(q8[(size_t)h * 2 * hd8 + hd8 + d]);
        }
    }
    std::vector<float> out8((size_t)tpb * n_head * hd8, -9.f);
    cpu_attn(q8.data(), nullptr, k8.data(), v8.data(), nullptr, tables, n_head, n_head_kv, hd8, 1, &info, scale,
             max_blocks, n_rows, n_real, out8.data(), cpu_kv_dtype::i8, cpu_kv_dtype::i8, ksc.data(), vsc.data());
    std::vector<float> got8(out8.begin(), out8.begin() + (size_t)n_head * hd8);
    check_f32("cpu_attn i8 fused", got8, exp8, 1e-5, 1e-3);
}

// ---------------------------------------------------------------------------
// qk_norm_rope: per-head RMSNorm, bias, RoPE and the paged KV store.
static void test_qk_norm_rope() {
    const int n_head = 2, n_head_kv = 1, head_dim = 4, n_rot = 4, tpb = 32, n_real = 2, max_blocks = 2;
    const float rope_base = 10000.f, eps = 1e-6f;
    const int qstride = n_head * 2 * head_dim, kvstride = n_head_kv * head_dim;
    std::vector<float> q((size_t)tpb * qstride), k((size_t)tpb * kvstride), v((size_t)tpb * kvstride);
    std::vector<float> qn(head_dim), kn(head_dim), qb(n_head * head_dim), kbias(n_head_kv * head_dim),
        vbias(n_head_kv * head_dim);
    for (size_t i = 0; i < q.size(); i++) q[i] = rnd((int)i);
    for (size_t i = 0; i < k.size(); i++) k[i] = rnd((int)i + 500);
    for (size_t i = 0; i < v.size(); i++) v[i] = rnd((int)i + 900);
    for (int d = 0; d < head_dim; d++) {
        qn[d] = 0.5f + 0.1f * d;
        kn[d] = 0.7f + 0.05f * d;
        qb[d] = rnd(200 + d) * 0.1f;
        kbias[d] = rnd(300 + d) * 0.1f;
        vbias[d] = rnd(400 + d) * 0.1f;
        qb[head_dim + d] = rnd(250 + d) * 0.1f;
    }
    int32_t tables[2] = {0, 1};
    cpu_step_info info{};
    info.n_rows = 1;
    info.n_real = n_real;
    info.tpb = tpb;
    info.pos[0] = 30; // second token lands at block 1, offset 0
    info.slot[0] = 0;
    info.active[0] = 1;

    const size_t pool = (size_t)2 * kCpuBlk * kvstride;
    std::vector<float> kp(pool), vp(pool);
    std::vector<float> qg = q, kg = k, vg = v;
    cpu_qk_norm_rope(qg.data(), kg.data(), vg.data(), qn.data(), kn.data(), kp.data(), vp.data(), tables, &info, n_head,
                     n_head_kv, head_dim, n_rot, rope_base, nullptr, 1.0f, eps, max_blocks, 1, n_real,
                     cpu_kv_dtype::f32, cpu_kv_dtype::f32, nullptr, nullptr, qb.data(), kbias.data(), vbias.data(), 2);

    // scalar reference: bias, RMSNorm, rope, store
    std::vector<float> eq = q, ek = k, ev = v, ekp(pool, 0.f), evp(pool, 0.f);
    for (int t = 0; t < n_real; t++) {
        const int pos = info.pos[0] + t;
        for (int h = 0; h < n_head; h++) {
            float * qh = &eq[(size_t)t * qstride + (size_t)h * 2 * head_dim];
            for (int d = 0; d < head_dim; d++) qh[d] += qb[(size_t)h * head_dim + d];
            double ss = 0;
            for (int d = 0; d < head_dim; d++) ss += (double)qh[d] * qh[d];
            const float inv = 1.0f / std::sqrt((float)(ss / head_dim) + eps);
            for (int d = 0; d < head_dim; d++) qh[d] *= inv * qn[d];
            rope_apply(qh, n_rot, rope_base, (float)pos, nullptr, 1.0f);
        }
        for (int h = 0; h < n_head_kv; h++) {
            float * kh = &ek[(size_t)t * kvstride + (size_t)h * head_dim];
            for (int d = 0; d < head_dim; d++) kh[d] += kbias[d];
            double ss = 0;
            for (int d = 0; d < head_dim; d++) ss += (double)kh[d] * kh[d];
            const float inv = 1.0f / std::sqrt((float)(ss / head_dim) + eps);
            for (int d = 0; d < head_dim; d++) kh[d] *= inv * kn[d];
            rope_apply(kh, n_rot, rope_base, (float)pos, nullptr, 1.0f);
            const int kb = tables[pos / kCpuBlk], ko = pos % kCpuBlk;
            const size_t off = kv_row_off((size_t)kb * n_head_kv + h, ko, head_dim);
            for (int d = 0; d < head_dim; d++) ekp[off + d] = kh[d];

            float * vh = &ev[(size_t)t * kvstride + (size_t)h * head_dim];
            for (int d = 0; d < head_dim; d++) {
                vh[d] += vbias[d];
                evp[off + d] = vh[d];
            }
        }
    }
    std::vector<float> gq(qg.begin(), qg.begin() + (size_t)n_real * qstride);
    std::vector<float> eqv(eq.begin(), eq.begin() + (size_t)n_real * qstride);
    check_f32("cpu_qk_norm_rope q", gq, eqv, 1e-5, 1e-3);
    check_f32("cpu_qk_norm_rope k", std::vector<float>(kg.begin(), kg.begin() + (size_t)n_real * kvstride),
              std::vector<float>(ek.begin(), ek.begin() + (size_t)n_real * kvstride), 1e-5, 1e-3);
    check_f32("cpu_qk_norm_rope kpool", kp, ekp, 1e-5, 1e-3);
    check_f32("cpu_qk_norm_rope vpool", vp, evp, 1e-5, 1e-3);

    // i8 store: one 32-value group, checked against the same quantizer.
    const int hd = 32;
    const int qs = n_head * 2 * hd, kvs = n_head_kv * hd;
    std::vector<float> qi((size_t)tpb * qs), ki((size_t)tpb * kvs), vi((size_t)tpb * kvs);
    for (size_t i = 0; i < qi.size(); i++) qi[i] = rnd((int)i + 17000);
    for (size_t i = 0; i < ki.size(); i++) ki[i] = rnd((int)i + 18000);
    for (size_t i = 0; i < vi.size(); i++) vi[i] = rnd((int)i + 19000);
    const size_t pool_i8 = (size_t)2 * kCpuBlk * kvs;
    std::vector<int8_t> kpi(pool_i8), vpi(pool_i8);
    std::vector<uint16_t> ksc(pool_i8), vsc(pool_i8);
    cpu_qk_norm_rope(qi.data(), ki.data(), vi.data(), nullptr, nullptr, kpi.data(), vpi.data(), tables, &info, n_head,
                     n_head_kv, hd, hd, rope_base, nullptr, 1.0f, eps, max_blocks, 1, n_real, cpu_kv_dtype::i8,
                     cpu_kv_dtype::i8, ksc.data(), vsc.data(), nullptr, nullptr, nullptr, 2);
    bool i8_ok = true;
    for (int t = 0; t < n_real; t++) {
        const int pos = info.pos[0] + t;
        for (int h = 0; h < n_head_kv; h++) {
            float maxk = 0, maxv = 0;
            for (int d = 0; d < hd; d++) {
                maxk = std::max(maxk, std::fabs(ki[(size_t)t * kvs + (size_t)h * hd + d]));
                maxv = std::max(maxv, std::fabs(vi[(size_t)t * kvs + (size_t)h * hd + d]));
            }
            const int kb = tables[pos / kCpuBlk], ko = pos % kCpuBlk;
            const size_t unit = (size_t)kb * n_head_kv + h;
            const float ks = ggml_half_to_float(ksc[kv_scale_off(unit, ko, hd)]);
            const float vs = ggml_half_to_float(vsc[kv_scale_off(unit, ko, hd)]);
            const float eks = maxk > 0 ? maxk / 127.0f : 1.0f;
            const float evs = maxv > 0 ? maxv / 127.0f : 1.0f;
            const float eks_h = ggml_half_to_float(ggml_float_to_half(eks));
            const float evs_h = ggml_half_to_float(ggml_float_to_half(evs));
            i8_ok = i8_ok && std::fabs(ks - eks_h) <= 1e-6f * std::max(1.f, eks_h);
            i8_ok = i8_ok && std::fabs(vs - evs_h) <= 1e-6f * std::max(1.f, evs_h);
            const size_t off = kv_row_off(unit, ko, hd);
            for (int d = 0; d < hd; d++) {
                const int q = (int)std::max(-127.f, std::min(127.f, std::round(ki[(size_t)t * kvs + d] / eks)));
                const int qv = (int)std::max(-127.f, std::min(127.f, std::round(vi[(size_t)t * kvs + d] / evs)));
                i8_ok = i8_ok && kpi[off + d] == (int8_t)q && vpi[off + d] == (int8_t)qv;
            }
        }
    }
    std::printf("%-30s %s\n", "cpu_qk_norm_rope i8 store", i8_ok ? "OK" : "FAIL");
    if (!i8_ok) g_fail++;
}

int main() {
    std::printf("%-30s %s\n", "cpu_pc_guard", cpu_pc_guard ? "on" : "off");
    test_conv();
    test_attn();
    test_qk_norm_rope();
    if (g_fail) {
        std::printf("test_cpu_attn_units: FAILURES: %d\n", g_fail);
        return 1;
    }
    std::printf("test_cpu_attn_units: all units OK\n");
    return 0;
}
