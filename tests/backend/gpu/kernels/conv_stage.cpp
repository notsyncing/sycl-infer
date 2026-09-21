// conv_l2_launch (depthwise conv + L2 norm) and conv_state_update_launch vs the
// CPU reference (layer 0).
#include <cstring>
#include <vector>

#include "stage_tests.h"

using namespace si;

void stage_conv(stage_env & env) {
    const hparams & hp = env.e.m.hp;
    const layer_t & L0 = env.e.m.layers[0];
    const int T = env.T;
    const int cd = hp.qkv_dim();
    const int qk = hp.n_group * hp.d_state;
    const int vd = hp.dt_rank * hp.d_state;

    auto raw = env.get("raw_wqkv-0");
    env.set_info(T, 0, 1);
    env.e.q.memcpy(env.e.d_qkv, raw.data(), raw.size() * 4);
    env.e.q.memset(env.e.d_conv_state, 0, (size_t)kMaxB * (hp.conv_k - 1) * cd * 4);
    conv_l2_launch(env.e.q, env.e.d_qkv, env.e.d_conv_state, env.e.m.dev_f32(L0.ssm_conv1d), env.e.d_conv_out,
                   env.e.d_info, cd, hp.conv_k, hp.d_state, hp.n_group, hp.rms_eps, 1, T);
    {
        auto qq = env.get("q_conv_predelta-0"), kk = env.get("k_conv_predelta-0"), vv = env.get("v_conv_predelta-0");
        std::vector<float> expect((size_t)T * cd);
        for (int t = 0; t < T; t++) {
            std::memcpy(&expect[(size_t)t * cd], &qq[(size_t)t * qk], qk * 4);
            std::memcpy(&expect[(size_t)t * cd + qk], &kk[(size_t)t * qk], qk * 4);
            std::memcpy(&expect[(size_t)t * cd + 2 * qk], &vv[(size_t)t * vd], vd * 4);
        }
        std::vector<float> hb(raw.size());
        env.e.q.memcpy(hb.data(), env.e.d_conv_out, hb.size() * 4).wait();
        env.cmp("conv_l2-0", hb, expect, 2e-3);
    }

    // state update: st[0..2] must be the last three raw tokens
    conv_state_update_launch(env.e.q, env.e.d_qkv, env.e.d_conv_state, env.e.d_info, cd, hp.conv_k, 1);
    std::vector<float> cs((size_t)(hp.conv_k - 1) * cd);
    env.e.q.memcpy(cs.data(), env.e.d_conv_state + (size_t)env.e.d_info->slot[0] * (size_t)3 * cd, cs.size() * 4)
        .wait();
    std::vector<float> expect_cs(cs.size());
    for (int j = 0; j < 3; j++) {
        std::memcpy(&expect_cs[(size_t)j * cd], &raw[(size_t)(T - 3 + j) * cd], cd * 4);
    }
    env.cmp("conv_state-0", cs, expect_cs);
}
