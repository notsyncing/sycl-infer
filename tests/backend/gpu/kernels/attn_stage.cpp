// attn_launch / attn_combine_launch vs the CPU reference (layer 3), plus the
// grouped decode attention (PF_DEC_GROUP): its partials must be bit-identical to
// the classic one-warp-per-(head, split) kernel, for the decode (n_real == 1)
// and the prefill (n_real == T) shapes.
#include <cstring>
#include <vector>

#include "stage_tests.h"

using namespace si;

void stage_attn(stage_env & env) {
    const hparams & hp = env.e.m.hp;
    const layer_t & L3 = env.e.m.layers[3];
    const int T = env.T;

    // set up the q/k/v projections and the paged KV cache
    std::vector<float> wq = env.get("raw_wq-3"), wk = env.get("raw_wk-3"), wv = env.get("raw_wv-3");
    env.e.q.memcpy(env.e.d_qbuf, wq.data(), wq.size() * 4);
    env.e.q.memcpy(env.e.d_kbuf, wk.data(), wk.size() * 4);
    env.e.q.memcpy(env.e.d_vbuf, wv.data(), wv.size() * 4);
    env.set_info(T, 0, 1);
    std::vector<int> blocks;
    const int need = (T + kBlockSize - 1) / kBlockSize;
    for (int i = 0; i < need; i++) {
        blocks.push_back(env.e.alloc_block());
    }
    env.e.set_table(0, blocks);
    qk_norm_rope_launch(env.e.q, env.e.d_qbuf, env.e.d_kbuf, env.e.d_vbuf, env.e.m.dev_f32(L3.q_norm),
                        env.e.m.dev_f32(L3.k_norm), env.e.d_kpool, env.e.d_vpool, env.e.d_tables, env.e.d_info,
                        hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_rot, hp.rope_base, /*rope_freqs=*/nullptr,
                        /*rope_mscale=*/1.0f, hp.rms_eps, env.e.max_blocks, 1, T, env.e.d_kscales, env.e.d_vscales);

    // classic split attention + combine
    attn_launch(env.e.q, env.e.d_qbuf, env.e.d_qbuf, env.e.d_kpool, env.e.d_vpool, env.e.d_partials, env.e.d_tables,
                hp.n_head, hp.n_head_kv, hp.head_dim, 16, env.e.d_info, hp.attn_scale, env.e.max_blocks, 1, T, nullptr,
                -1, env.e.d_kscales, env.e.d_vscales);
    attn_combine_launch(env.e.q, env.e.d_partials, env.e.d_qbuf, env.e.d_attn_out, env.e.d_info, hp.n_head, hp.head_dim,
                        16, 1, T);
    std::vector<float> hb((size_t)T * hp.n_head * hp.head_dim);
    env.e.q.memcpy(hb.data(), env.e.d_attn_out, hb.size() * 4).wait();
    auto attn_tol_of = [](kv_dtype_t t) {
        return t == kv_dtype_t::f32 ? 2e-3 : t == kv_dtype_t::i4 ? 3.0e-1 : t == kv_dtype_t::i8 ? 3e-2 : 5e-3;
    };
    const double attn_tol = std::max(attn_tol_of(kv_k_dtype()), attn_tol_of(kv_v_dtype()));
    env.cmp("attn_gated-3", hb, env.get("attn_pregate-3"), attn_tol);

    // grouped vs classic partials
    const int nsp = 4;
    auto check = [&](const char * name, int nreal, int pos0) {
        env.e.d_info->n_rows = 1;
        env.e.d_info->n_real = nreal;
        env.e.d_info->pos[0] = pos0;
        env.e.d_info->slot[0] = 1;
        env.e.d_info->active[0] = 1;
        const size_t npart = (size_t)nreal * hp.n_head * nsp * (2 + hp.head_dim);
        std::vector<float> p_cls(npart), p_grp(npart);
        auto run = [&](int g) {
            attn_launch(env.e.q, env.e.d_qbuf, env.e.d_qbuf, env.e.d_kpool, env.e.d_vpool, env.e.d_partials,
                        env.e.d_tables, hp.n_head, hp.n_head_kv, hp.head_dim, nsp, env.e.d_info, hp.attn_scale,
                        env.e.max_blocks, 1, nreal, nullptr, g);
            env.e.q.wait();
        };
        run(0);
        env.e.q.memcpy(p_cls.data(), env.e.d_partials, npart * 4).wait();
        run(1);
        env.e.q.memcpy(p_grp.data(), env.e.d_partials, npart * 4).wait();
        const bool same = std::memcmp(p_cls.data(), p_grp.data(), npart * 4) == 0;
        printf("%-24s %s\n", name, same ? "OK (partials bit-identical)" : "FAIL");
        if (!same) {
            double md = 0;
            size_t at = 0;
            for (size_t i = 0; i < npart; i++) {
                const double d = std::fabs((double)p_cls[i] - p_grp[i]);
                if (d > md) {
                    md = d;
                    at = i;
                }
            }
            printf("   max|diff|=%.6g at %zu (classic %.6g group %.6g)\n", md, at, (double)p_cls[at],
                   (double)p_grp[at]);
            env.fails++;
        }
    };
    check("attn_group-decode", 1, T - 1);
    check("attn_group-prefill", T, 0);
}
