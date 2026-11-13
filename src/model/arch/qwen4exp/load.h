#pragma once
namespace si {
struct model;
struct model_profile;

const model_profile & qwen4exp_profile();

namespace arch {
void load_qwen4exp(model & m);
} // namespace arch
} // namespace si
