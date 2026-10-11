// qk_norm_rope_launch: q/k RMSNorm + RoPE, V copy, and the paged KV cache
// contents, vs the CPU reference (layer 3).
#include <vector>

#include "stage_tests.h"

using namespace si;

void stage_qk_norm_rope(stage_env & env) {
    const hparams & hp = env.e.m.hp;
    const layer_t & L3 = env.e.m.layers[3];
    const int T = env.T;
    // layer 3 is the first full-attention layer in both the dense and the
    // MoE reference models, i.e. global attention index 0
    void * kp, * vp;
    const void * ksc, * vsc;
    env.kvp(0, kp, vp, ksc, vsc);

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

    qk_norm_rope_launch(env.e.q, env.e.d_qbuf, env.e.d_kbuf, env.e.d_vbuf, env.wf32(L3.q_norm),
                        env.wf32(L3.k_norm), kp, vp, env.e.d_tables, env.e.d_info,
                        hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_rot, hp.rope_base, /*rope_freqs=*/nullptr,
                        /*rope_mscale=*/1.0f, hp.rms_eps, env.e.max_blocks, 1, T, ksc, vsc);

    std::vector<float> hb;
    hb.resize(wq.size());
    env.e.q.memcpy(hb.data(), env.e.d_qbuf, hb.size() * 4).wait();
    env.cmp("qkq_norm_rope-3", hb, env.get("q_all-3"));
    auto tk = env.get("k_all-3");
    hb.resize(tk.size());
    env.e.q.memcpy(hb.data(), env.e.d_kbuf, hb.size() * 4).wait();
    env.cmp("k_norm_rope-3", hb, tk);
    auto tv = env.get("v_all-3");
    hb.resize(tv.size());
    env.e.q.memcpy(hb.data(), env.e.d_vbuf, hb.size() * 4).wait();
    env.cmp("v_copy-3", hb, tv);

    // verify the paged KV cache contents
    std::vector<float> kc((size_t)hp.n_head_kv * T * hp.head_dim), vc(kc.size());
    for (int t = 0; t < T; t++) {
        const int blk = blocks[t / kBlockSize];
        const int off = t % kBlockSize;
        for (int h = 0; h < hp.n_head_kv; h++) {
            const size_t src = ((size_t)blk * hp.n_head_kv + h) * kBlockSize + off;
            env.e.kv_read_vec(0, 0, src * hp.head_dim, &kc[((size_t)t * hp.n_head_kv + h) * hp.head_dim], hp.head_dim);
            env.e.kv_read_vec(0, 1, src * hp.head_dim, &vc[((size_t)t * hp.n_head_kv + h) * hp.head_dim], hp.head_dim);
        }
    }
    // the pool may store bf16/f16/i8/i4 (PF_KV_TYPE / --kv-type K:V): the
    // round-trip error is bounded by the storage precision, not by the kernel
    auto tol_of = [](kv_dtype_t t) {
        return t == kv_dtype_t::f32 ? 2e-3 : t == kv_dtype_t::i4 ? 3.0e-1 : t == kv_dtype_t::i8 ? 2e-2 : 5e-3;
    };
    const double kv_tol = std::max(tol_of(kv_k_dtype()), tol_of(kv_v_dtype()));
    env.cmp("kcache-3", kc, tk, kv_tol);
    env.cmp("vcache-3", vc, tv, kv_tol);
}
