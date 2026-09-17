// gdn_launch (GDN recurrence) vs the CPU reference (layer 0).  The conv output
// and the raw beta/alpha projections are prepared from the reference, matching
// what the engine feeds the kernel.
#include <cmath>
#include <vector>

#include "stage_tests.h"

using namespace si;

void stage_gdn(stage_env & env) {
    const hparams & hp = env.e.m.hp;
    const layer_t & L0 = env.e.m.layers[0];
    const int T = env.T;
    const int cd = 3 * hp.d_inner;

    // conv output is the GDN input
    auto raw = env.get("raw_wqkv-0");
    env.set_info(T, 0, 1);
    env.e.q.memcpy(env.e.d_qkv, raw.data(), raw.size() * 4);
    env.e.q.memset(env.e.d_conv_state, 0, (size_t) kMaxB * (hp.conv_k - 1) * cd * 4);
    conv_l2_launch(env.e.q, env.e.d_qkv, env.e.d_conv_state, env.e.m.dev_f32(L0.ssm_conv1d),
                   env.e.d_conv_out, env.e.d_info, cd, hp.conv_k, hp.d_state, hp.n_group,
                   hp.rms_eps, 1, T);

    // raw beta/alpha projections from the reference (the kernel applies
    // sigmoid / softplus)
    std::vector<float> bt(T * hp.dt_rank), at(T * hp.dt_rank);
    {
        std::vector<float> xnt = env.get("attn_norm-0");
        for (int t = 0; t < T; t++) {
            std::vector<float> tmp(hp.dt_rank);
            env.ref.matvec(L0.ssm_beta, &xnt[(size_t) t * hp.n_embd], tmp.data());
            for (int i = 0; i < hp.dt_rank; i++) bt[t * hp.dt_rank + i] = tmp[i];
            env.ref.matvec(L0.ssm_alpha, &xnt[(size_t) t * hp.n_embd], tmp.data());
            for (int i = 0; i < hp.dt_rank; i++) at[t * hp.dt_rank + i] = tmp[i];
        }
    }
    env.e.q.memcpy(env.e.d_beta, bt.data(), bt.size() * 4);
    env.e.q.memcpy(env.e.d_alpha, at.data(), at.size() * 4);
    env.e.q.memset(env.e.d_gdn_state, 0, (size_t) kMaxB * hp.dt_rank * hp.d_state * hp.d_state * 4);

    gdn_launch(env.e.q, env.e.d_conv_out, env.e.d_alpha, env.e.m.dev_f32(L0.ssm_dt),
               env.e.m.dev_f32(L0.ssm_a), env.e.d_beta, env.e.d_gdn_state, env.e.d_attn_pre,
               env.e.d_info, hp.d_state, hp.dt_rank, cd,
               1.0f / std::sqrt((float) hp.d_state), kMaxB, 1);
    std::vector<float> hb((size_t) T * hp.d_inner);
    env.e.q.memcpy(hb.data(), env.e.d_attn_pre, hb.size() * 4).wait();
    env.cmp("gdn-0", hb, env.get("final_output_pre-0"), 2e-3);
}
