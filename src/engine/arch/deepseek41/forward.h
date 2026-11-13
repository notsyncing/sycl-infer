#pragma once
// DeepSeek-V4.1-Flash's model-owned forward entry points.  These used to be
// declared in generic engine.h; they are architecture API, so they live with
// the forward implementations (forward_host.cpp / forward_gpu.cpp) and only the
// architecture's own code includes this header.
#include "engine.h"

namespace si {

// The host forward (arch/deepseek41/forward_host.cpp): owns its per-sequence
// window / compressed caches and its Engram history, so it does not use the
// paged KV pool or the seg_plan machinery.
std::vector<int> ds41_generate(model & m, tokenizer & tk, const std::vector<int> & prompt, const mm_prompt * mm,
                               const gen_params & gp, const std::function<bool(int)> & cb,
                               std::vector<float> * first_logits = nullptr, size_t * cached_tokens = nullptr);

// The GPU forward (arch/deepseek41/forward_gpu.cpp).  The host forward above is
// the oracle.  When the run is on GPUs this state owns the device weight views
// (through engine::weight_maps_), the per-layer window / group-shared
// compressed caches, the routed-expert bank and the per-token scratch, and
// drives every dense GEMV, the mHC, the MLA attention, the compressor/indexer
// and the routed/shared experts on the card.
ds41_gpu_state * ds41_gpu_create(engine & e);
void ds41_gpu_destroy(ds41_gpu_state * st);
std::vector<int> ds41_gpu_generate(model & m, tokenizer & tk, const std::vector<int> & prompt, const mm_prompt * mm,
                                   const gen_params & gp, const std::function<bool(int)> & cb,
                                   std::vector<float> * first_logits, size_t * cached_tokens,
                                   ds41_gpu_state & st);

} // namespace si
