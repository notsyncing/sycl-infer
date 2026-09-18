#pragma once
// Declarations of the per-kernel stage tests, one source file per kernel
// (mirroring src/backend/gpu/kernels/).  test_gpu_stages.cpp runs them in order.
#include "stage_test.h"

void stage_rmsnorm(si::stage_env & env);
void stage_embed(si::stage_env & env);
void stage_qk_norm_rope(si::stage_env & env);
void stage_attn(si::stage_env & env);
void stage_conv(si::stage_env & env);
void stage_gdn(si::stage_env & env);
void stage_gated_norm(si::stage_env & env);
