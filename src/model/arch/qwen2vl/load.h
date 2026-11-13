#pragma once
namespace si {
struct model;
struct model_profile;

const model_profile & qwen2vl_profile();

namespace arch {
void load_qwen2vl(model & m);
} // namespace arch
} // namespace si
