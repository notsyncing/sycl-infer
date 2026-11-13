#pragma once
namespace si {
struct model;
struct model_profile;

const model_profile & deepseek41_profile();

namespace arch {
void load_deepseek41(model & m);
} // namespace arch
} // namespace si
