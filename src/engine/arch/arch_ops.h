#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
#include "model_profile.h"

namespace si {
struct engine;
struct engine_config;
struct gen_params;
struct mm_prompt;

// The per-architecture hook implementations live in registry_internal.h, which
// only registry.cpp and the per-arch runtime.cpp include.  The public seam keeps
// just this forward declaration so it names no architecture.
struct arch_ops_registry;

// The architecture dispatch table: the single seam for the lifecycle,
// capability, request-path and prefix-cache decisions that are a property of the
// forward kind rather than of the generic engine.  A row is selected once per
// engine from `model_profile::fwd`; `generic_plan` has no row (arch_ops_for
// returns null) and runs the shared seg_plan/record_forward machinery.
//
// A null hook means "the generic engine owns this": decode_batch/run_head/
// prefill_text are the generic plan, the pc_* hooks mean the checkpoint blob has
// no architecture-owned tail, and so on.  A caller therefore tests the hook, not
// the model's profile.
struct arch_ops {
    model_profile::fwd_kind kind;
    bool owns_kv_pool = false;      // the forward owns its KV caches (no paged pool)
    bool owns_request_path = false; // generate() handles the whole request
    bool md_dnnl = true;            // false: setup_md_dnnl declines
    // The forward owns enough of the step that the generic paths must yield to
    // it: a greedy decode may take its token from the device argmax.  (Whether a
    // media prompt goes through the model's own chunker is declared on the
    // model profile as `spec.mm_owned_prefill`, not here.)
    bool device_argmax = false;
    // The routed experts and their tables are read straight out of the mmap on
    // the host, so their pages must stay host-resident after the weights are
    // uploaded (release_host_weight_pages' keep-set).
    bool host_keep_routed_experts = false;
    // True when setup_weights registered architecture-owned dense oneDNN
    // stores.  A converted tensor skips its raw upload and the generic
    // weight-path report names the stores; null means the architecture has none
    // (and reads as false).
    bool (*dense_store)(const engine &) = nullptr;
    // ctor config resolution that is a property of the architecture (a model
    // profile's budget / tail-skip defaults), run once after the profile is known.
    void (*configure)(engine &, const engine_config &) = nullptr;
    void (*setup)(engine &) = nullptr;                       // constructor: build_graphs / model-owned allocation / device state
    void (*prefault)(engine &) = nullptr;                    // model-owned page prefault
    void (*setup_weights)(engine &, bool has_gpu) = nullptr; // model-owned dense stores
    void (*destroy)(engine &) = nullptr;                     // model-owned teardown / device state
    // The forward's own per-slot state reset (beyond the generic recurrent
    // state) and its end-of-run routing diagnostic.
    void (*reset_extra)(engine &) = nullptr;                 // model-owned per-slot history reset
    void (*slot_zero)(engine &, int slot) = nullptr;
    void (*hist_report)(engine &) = nullptr;
    std::vector<int> (*generate)(engine &, const std::vector<int> &, const mm_prompt *, const gen_params &,
                                 const std::function<bool(int)> &, std::vector<float> *, size_t *) = nullptr;

    // ---- forward dispatch -------------------------------------------------
    void (*prefill_text)(engine &, const std::vector<int> &, int slot, int n, int start) = nullptr;
    void (*prefill_mm)(engine &, const mm_prompt &, int slot) = nullptr;
    void (*decode_batch)(engine &, const int32_t *, const int32_t *, const int32_t *, int n_rows) = nullptr;
    std::vector<float> (*run_head)(engine &) = nullptr;

    // ---- prefix-cache checkpoint extras -----------------------------------
    // The generic checkpoint pool carries only the GDN/conv state; a forward
    // whose resume point also needs host state appends it to the blob through
    // these.  `depth_blocks` is the node's depth in kBlockSize blocks.
    size_t (*pc_extra_floats)(engine &, int depth_blocks) = nullptr;
    void (*pc_serialize)(engine &, float * out, int st, int depth_blocks) = nullptr;
    void (*pc_deserialize)(engine &, const float * in, int st, int depth_blocks) = nullptr;
    void (*pc_restore)(engine &, int slot, int st, int depth_blocks) = nullptr;
    void (*pc_clear)(engine &, int st) = nullptr;
    void (*pc_resume_prefixes)(engine &, std::vector<int> & prefixes, int L) = nullptr;
    // True when only a batch that ENDS on a block boundary may capture (the
    // architecture's chunker resumes exactly at a cold chunk boundary).
    bool (*pc_block_aligned_capture)(engine &) = nullptr;
};

const arch_ops * arch_ops_for(model_profile::fwd_kind);

} // namespace si
