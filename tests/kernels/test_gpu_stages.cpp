// Stage-by-stage GPU kernel validation against the CPU reference.
//
// One source file per kernel lives next to this runner (see the *_stage.cpp
// siblings, mirroring src/kernels/); each prepares its own inputs from the
// reference snapshots and compares back against them.
#include "stage_tests.h"

using namespace si;

int main(int argc, char ** argv) {
    setenv("PF_DP4A", "0", 1); // strict tests validate the fp32 path

    stage_env env(stage_arg_model(argc, argv), stage_arg_tokens(argc, argv));
    stage_rmsnorm(env);
    stage_embed(env);
    stage_qk_norm_rope(env);
    stage_attn(env);
    stage_conv(env);
    stage_gdn(env);
    stage_gated_norm(env);
    return env.finish("test_gpu_stages");
}
