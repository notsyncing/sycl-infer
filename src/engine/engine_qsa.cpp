// QSA (query-side sparse attention) for qwen4exp.
//
// Up to `indexer_top_k` keys the reference's selector covers every visible
// key, so dense attention is exactly equivalent; beyond it the model attends
// to the top `top_k` keys pooled in blocks of `compress_ratio` plus the
// incomplete tail.  This implementation keeps the selector and the attention
// on the host:
//   * the indexer weights are small and already in the host mmap;
//   * full-attention layers keep a host mirror of their own K/V (allocated
//     lazily per layer and slot), so the selected keys need no device gather;
//   * the result is written back to the layer's ATTN_OUT buffer and the
//     existing wo GEMV runs on the device.
// Decode adds ~10-15 ms/token of host work; prefill pays it once per chunk.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <vector>

#include "engine.h"
#include "backend/cpu/kernels/common.h"
#include "backend/cpu/cpu_types.h"

namespace si {

namespace {

void head_norm(float * x, const float * w, int n, float eps) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    const float s = 1.f / std::sqrt(ss / (float)n + eps);
    for (int i = 0; i < n; i++) {
        x[i] = x[i] * s * w[i];
    }
}

void head_rope(float * x, int n_rot, float base, int pos) {
    rope_apply(x, n_rot, base, (float)pos);
}

// The reference keeps the indexer and main KV in the model's cache type (f16
// here), so every value it scores/attends to has been rounded to f16.  The host
// mirrors have to round too, or the f32 values drift by ~1e-3 and the top-k ties
// (rectified scores of exactly 0) pick different pools.
inline float f16r(float x) {
    return ggml_half_to_float(ggml_float_to_half(x));
}

} // namespace

bool engine::qsa_attention(int a, const layer_t & L, int dev, int n_tok, const step_info * inf, const float * mixed,
                           float * qbuf, const float * kbuf, const float * vbuf, float * attn_out) {
    if (!qsa_on_ || !L.idx.ok() || a < 0 || a >= (int)qsa_cache_.size()) {
        return false;
    }
    const hparams & hp = m.hp;
    const int hd = hp.head_dim, nh = hp.n_head, nkv = hp.n_head_kv;
    const int id = hp.idx_head_dim, inh = hp.idx_n_head;
    const int kvd = nkv * hd;
    const int gqa = nh / nkv;
    const int n_pool_max = hp.idx_top_k / (hp.idx_kpool > 0 ? hp.idx_kpool : 4);
    const int stride = hd * 2;

    auto is_cpu = [&](int d) { return multi_dev ? (dev_kind_[(size_t)d] == 1) : cpu_mode; };
    auto to_host = [&](void * dst, const void * src, size_t bytes) {
        if (is_cpu(dev)) {
            std::memcpy(dst, src, bytes);
        } else {
            dev_queue(dev).memcpy(dst, src, bytes).wait();
        }
    };
    auto from_host = [&](void * dst, const void * src, size_t bytes) {
        if (is_cpu(dev)) {
            std::memcpy(dst, src, bytes);
        } else {
            dev_queue(dev).memcpy(dst, src, bytes).wait();
        }
    };

    static const bool qdbg = std::getenv("PF_Q4_QSADBG") != nullptr;
    if (qdbg) {
        fprintf(stderr, "[qsa] enter a=%d n_tok=%d pos0=%d tpb=%d rows=%d\n", a, n_tok, inf->pos[0], inf->tpb,
                inf->n_rows);
    }
    const int n_embd = hp.n_embd;
    std::vector<float> mh((size_t)n_tok * n_embd);
    std::vector<float> qh((size_t)n_tok * nh * stride);
    std::vector<float> kh((size_t)n_tok * kvd);
    std::vector<float> vh((size_t)n_tok * kvd);
    to_host(mh.data(), mixed, mh.size() * sizeof(float));
    to_host(qh.data(), qbuf, qh.size() * sizeof(float));
    to_host(kh.data(), kbuf, kh.size() * sizeof(float));
    to_host(vh.data(), vbuf, vh.size() * sizeof(float));

    // Indexer projections (BF16 weights, host GEMV) for the whole batch.
    std::vector<float> iq((size_t)n_tok * inh * id);
    std::vector<float> ik((size_t)n_tok * id);
    cpu_q4_up_gemv(L.idx.q_proj.data, L.idx.q_proj.type, mh.data(), n_embd, iq.data(), inh * id, n_tok, n_embd,
                   inh * id);
    cpu_q4_up_gemv(L.idx.k_proj.data, L.idx.k_proj.type, mh.data(), n_embd, ik.data(), id, n_tok, n_embd, id);

    // Per row: normalise/RoPE the indexer keys, maintain the ring/pool and the
    // K/V mirror, and remember whether any row crosses the top-k budget.
    bool need = false;
    std::vector<float> qpos((size_t)n_tok * inh * id);
    for (int i = 0; i < n_tok; i++) {
        const int row = inf->tpb == 1 ? i : i / inf->tpb;
        const int t = inf->tpb == 1 ? 0 : i % inf->tpb;
        const int pos = inf->pos[row] + t;
        const int slot = inf->slot[row];
        float * qr = iq.data() + (size_t)i * inh * id;
        for (int h = 0; h < inh; h++) {
            head_norm(qr + h * id, L.idx.q_norm, id, hp.rms_eps);
            head_rope(qr + h * id, hp.n_rot, hp.rope_base, pos);
        }
        std::memcpy(qpos.data() + (size_t)i * inh * id, qr, (size_t)inh * id * sizeof(float));
        // The cache stores the *raw* indexer key: the reference pools the raw
        // keys of a block and only then applies k_norm + RoPE to the mean.
        // Normalising/rotating each member first and pooling those gives a
        // different (and wrong) selector score.
        float * kr = ik.data() + (size_t)i * id;

        if (slot < 0 || slot >= (int)qsa_cache_[(size_t)a].size()) {
            continue;
        }
        auto & sp = qsa_cache_[(size_t)a][(size_t)slot];
        if (!sp) {
            sp = std::make_unique<qsa_slot_state>();
            sp->k.assign((size_t)qsa_max_seq_ * kvd, 0.f);
            sp->v.assign((size_t)qsa_max_seq_ * kvd, 0.f);
            sp->pool.assign(((size_t)qsa_max_seq_ / 4 + 1) * id, 0.f);
            sp->ring.assign((size_t)4 * id, 0.f);
            sp->n_pool = 0;
        }
        sp->n_pool = std::max(sp->n_pool, pos / 4); // complete groups before pos
        float * ring = sp->ring.data() + (size_t)(pos % 4) * id;
        for (int d = 0; d < id; d++) {
            ring[d] = f16r(kr[d]);
        }
        if (pos % 4 == 3) {
            float * pl = sp->pool.data() + (size_t)(pos / 4) * id;
            for (int d = 0; d < id; d++) {
                pl[d] = 0.25f * (sp->ring[(size_t)0 * id + d] + sp->ring[(size_t)1 * id + d]
                                 + sp->ring[(size_t)2 * id + d] + sp->ring[(size_t)3 * id + d]);
            }
            head_norm(pl, L.idx.k_norm, id, hp.rms_eps);
            head_rope(pl, hp.n_rot, hp.rope_base, pos - 3);
            for (int d = 0; d < id; d++) {
                pl[d] = f16r(pl[d]);
            }
            sp->n_pool = std::max(sp->n_pool, pos / 4 + 1);
        }
        {
            const float * ks = kh.data() + (size_t)i * kvd;
            const float * vs = vh.data() + (size_t)i * kvd;
            float * kd = sp->k.data() + (size_t)pos * kvd;
            float * vd = sp->v.data() + (size_t)pos * kvd;
            for (int d = 0; d < kvd; d++) {
                kd[d] = f16r(ks[d]);
                vd[d] = f16r(vs[d]);
            }
        }
        if (pos + 1 > hp.idx_top_k) {
            need = true;
        }
    }
    if (qdbg) {
        fprintf(stderr, "[qsa] caches updated need=%d\n", (int)need);
    }
    if (a == 0) {
        if (const char * qp = std::getenv("PF_QSA_Q")) {
            if (FILE * fp = fopen(qp, "wb")) {
                fwrite(qpos.data() + (size_t)(n_tok - 1) * inh * id, sizeof(float), (size_t)inh * id, fp);
                fclose(fp);
            }
        }
        if (const char * mp = std::getenv("PF_QSA_MIX")) {
            if (FILE * fp = fopen(mp, "wb")) {
                fwrite(mh.data() + (size_t)(n_tok - 1) * n_embd, sizeof(float), (size_t)n_embd, fp);
                fclose(fp);
            }
        }
    }
    if (!need) {
        return false; // dense attention is exactly equivalent here
    }

    if (const char * kp = std::getenv("PF_QSA_POOLK")) {
        if (a < (int)qsa_cache_.size() && !qsa_cache_[(size_t)a].empty() && qsa_cache_[(size_t)a][0]) {
            if (FILE * fp = fopen(kp, "wb")) {
                fwrite(qsa_cache_[(size_t)a][0]->pool.data(), sizeof(float), (size_t)id, fp);
                fclose(fp);
            }
        }
    }
    std::vector<float> out((size_t)n_tok * nh * hd, 0.f);
    for (int i = 0; i < n_tok; i++) {
        const int row = inf->tpb == 1 ? i : i / inf->tpb;
        const int t = inf->tpb == 1 ? 0 : i % inf->tpb;
        const int pos = inf->pos[row] + t;
        const int slot = inf->slot[row];
        if (slot < 0 || slot >= (int)qsa_cache_[(size_t)a].size() || !qsa_cache_[(size_t)a][(size_t)slot]) {
            continue;
        }
        const qsa_slot_state & sp = *qsa_cache_[(size_t)a][(size_t)slot];
        const int n_vis = pos + 1;
        const int n_pool_vis = n_vis / 4;
        std::vector<int> sel;
        if (n_pool_vis <= n_pool_max) {
            sel.resize((size_t)n_vis);
            for (int p = 0; p < n_vis; p++) {
                sel[(size_t)p] = p;
            }
        } else {
            // The reference runs ggml_top_k, whose CPU implementation is
            // std::partial_sort over the score with `a > b`.  Reproduce it with
            // the same algorithm and comparator so equal (rectified-to-zero)
            // scores select the same pools; a priority queue with a lower-index
            // tie-break picks a different subset.
            const float * qr = qpos.data() + (size_t)i * inh * id;
            const float inv = 1.0f / std::sqrt((float)id);
            std::vector<float> scores((size_t)n_pool_vis);
            for (int p = 0; p < n_pool_vis; p++) {
                const float * pk = sp.pool.data() + (size_t)p * id;
                float s = 0.f;
                for (int h = 0; h < inh; h++) {
                    const float d = dot_f32(qr + h * id, pk, id);
                    s += d > 0.f ? d : 0.f;
                }
                scores[(size_t)p] = s * inv;
            }
            if (a == 0 && i == n_tok - 1) {
                if (const char * sp_path = std::getenv("PF_QSA_SCORE")) {
                    if (FILE * fp = fopen(sp_path, "wb")) {
                        fwrite(scores.data(), sizeof(float), scores.size(), fp);
                        fclose(fp);
                    }
                }
            }
            std::vector<int> idx((size_t)n_pool_vis);
            for (int p = 0; p < n_pool_vis; p++) {
                idx[(size_t)p] = p;
            }
            const int n_top = std::min(n_pool_vis, n_pool_max);
            std::partial_sort(idx.begin(), idx.begin() + n_top, idx.end(),
                              [&](int a, int b) { return scores[(size_t)a] > scores[(size_t)b]; });
            std::vector<int> pools(idx.begin(), idx.begin() + n_top);
            if (a == 0 && i == n_tok - 1) {
                if (const char * pp = std::getenv("PF_QSA_POOLS")) {
                    if (FILE * fp = fopen(pp, "wb")) {
                        fwrite(pools.data(), sizeof(int), pools.size(), fp);
                        fclose(fp);
                    }
                }
            }
            std::sort(pools.begin(), pools.end());
            sel.reserve(pools.size() * 4 + 3);
            for (int p : pools) {
                for (int j = 0; j < 4; j++) {
                    sel.push_back(p * 4 + j);
                }
            }
            for (int p = n_pool_vis * 4; p <= pos; p++) {
                sel.push_back(p);
            }
        }
        const int nsel = (int)sel.size();
        if (qdbg) {
            fprintf(stderr, "[qsa] row=%d pos=%d slot=%d nsel=%d\n", i, pos, slot, nsel);
        }
        const int kvh_stride = kvd;
        // one job per query head; each is ~nsel*2*head_dim MAC
        par(nh, [&](int h) {
            const int kvh = h / gqa;
            const float * q = qh.data() + (size_t)i * nh * stride + (size_t)h * stride;
            std::vector<float> sc((size_t)nsel);
            float mx = -3.402823466e38f;
            for (int j = 0; j < nsel; j++) {
                const float * kp = sp.k.data() + (size_t)sel[(size_t)j] * kvh_stride + (size_t)kvh * hd;
                const float s = dot_f32(q, kp, hd) * hp.attn_scale;
                sc[(size_t)j] = s;
                mx = std::max(mx, s);
            }
            float sum = 0.f;
            for (int j = 0; j < nsel; j++) {
                sc[(size_t)j] = std::exp(sc[(size_t)j] - mx);
                sum += sc[(size_t)j];
            }
            const float inv = 1.f / sum;
            float * o = out.data() + (size_t)i * nh * hd + (size_t)h * hd;
            for (int d = 0; d < hd; d++) {
                o[d] = 0.f;
            }
            for (int j = 0; j < nsel; j++) {
                const float w = sc[(size_t)j] * inv;
                const float * vp = sp.v.data() + (size_t)sel[(size_t)j] * kvh_stride + (size_t)kvh * hd;
                for (int d = 0; d < hd; d++) {
                    o[d] += w * vp[d];
                }
            }
            const float * gate = q + hd;
            for (int d = 0; d < hd; d++) {
                o[d] *= 1.f / (1.f + std::exp(-gate[d]));
            }
        });
    }
    if (qdbg) {
        fprintf(stderr, "[qsa] attention done, writing out\n");
    }
    from_host(attn_out, out.data(), out.size() * sizeof(float));
    if (qdbg) {
        fprintf(stderr, "[qsa] out written\n");
    }
    return true;
}

} // namespace si
