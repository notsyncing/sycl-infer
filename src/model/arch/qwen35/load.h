#pragma once
namespace si {
struct model;
struct model_profile;

const model_profile & qwen35_profile();

namespace arch {
void load_qwen35(model & m);
} // namespace arch
} // namespace si
