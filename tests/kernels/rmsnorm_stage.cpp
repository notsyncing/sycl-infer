// rmsnorm_launch vs the CPU reference (layer 0 attn_norm over all tokens).
#include <cstring>
#include <vector>

#include "stage_tests.h"

using namespace si;

void stage_rmsnorm(stage_env & env) {
    const hparams & hp = env.e.m.hp;
    std::vector<float> x((size_t) env.T * hp.n_embd), expect(x.size()), hb;
    std::memcpy(x.data(), env.get("model.input_embed").data(), x.size() * 4);
    std::memcpy(expect.data(), env.get("attn_norm-0").data(), expect.size() * 4);
    env.e.q.memcpy(env.e.d_x, x.data(), x.size() * 4);
    rmsnorm_launch(env.e.q, env.e.d_x, env.e.m.dev_f32(env.e.m.layers[0].attn_norm),
                   env.e.d_xnorm, env.T, hp.n_embd, hp.rms_eps);
    hb.resize(x.size());
    env.e.q.memcpy(hb.data(), env.e.d_xnorm, hb.size() * 4).wait();
    env.cmp("rmsnorm-0", hb, expect);
}
