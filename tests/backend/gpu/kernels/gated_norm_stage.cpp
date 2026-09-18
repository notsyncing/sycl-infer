// gated_norm_launch vs the CPU reference (layer 0): RMSNorm over the GDN heads,
// weighted and gated by silu(z).
#include <vector>

#include "stage_tests.h"

using namespace si;

void stage_gated_norm(stage_env & env) {
    const hparams & hp = env.e.m.hp;
    const layer_t & L0 = env.e.m.layers[0];
    const int T = env.T;

    auto ap = env.get("final_output_pre-0");
    env.set_info(T, 0, 1);
    env.e.q.memcpy(env.e.d_attn_pre, ap.data(), ap.size() * 4);
    auto z = env.get("z-0");
    env.e.q.memcpy(env.e.d_z, z.data(), z.size() * 4);

    gated_norm_launch(env.e.q, env.e.d_attn_pre, env.e.d_z, env.e.m.dev_f32(L0.ssm_norm), env.e.d_attn_merged,
                      env.e.d_info, hp.dt_rank, hp.d_state, hp.rms_eps, 1, T);
    std::vector<float> hb((size_t)T * hp.d_inner);
    env.e.q.memcpy(hb.data(), env.e.d_attn_merged, hb.size() * 4).wait();
    env.cmp("gated_norm-0", hb, env.get("final_output-0"), 2e-3);
}
