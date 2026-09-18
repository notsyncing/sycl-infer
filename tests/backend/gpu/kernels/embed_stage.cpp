// embed_launch vs the CPU reference (token ids -> embedding rows).
#include <vector>

#include "stage_tests.h"

using namespace si;

void stage_embed(stage_env & env) {
    const hparams & hp = env.e.m.hp;
    env.set_info(env.T, 0, 1);
    embed_launch(env.e.q, env.e.m.dev_ptr(env.e.m.tok_embd.data), env.e.m.tok_embd.type, env.e.d_info, env.e.d_x,
                 hp.n_embd, env.e.m.tok_embd_row_bytes);
    std::vector<float> hb((size_t)env.T * hp.n_embd);
    env.e.q.memcpy(hb.data(), env.e.d_x, hb.size() * 4).wait();
    env.cmp("embed", hb, env.get("model.input_embed"));
}
