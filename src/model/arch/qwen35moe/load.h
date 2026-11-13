#pragma once
namespace si {
struct model;
struct model_profile;

const model_profile & qwen35moe_profile();

namespace arch {
void load_qwen35moe(model & m);
} // namespace arch
} // namespace si
