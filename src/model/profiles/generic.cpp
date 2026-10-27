// Model profiles: the generic Qwen3.5 family (qwen35 and its MoE variant).
//
// One const per registry key, because model::load asserts `prof->key` equals the
// GGUF's exact `general.architecture`: a shared "generic" key could not pass that
// check, and a second lookup table would be one more thing to drift.  The two
// rows carry the same policy/weights/capabilities by design; the qwen35moe row
// only differs in its identity and in the MoE note in the provenance.
//
// Everything here is the engine's existing behaviour at eb19490, stated instead
// of implied: the shared seg_plan/record_forward forward, chunked prefill,
// prefix cache, no auto layer map, the single-blob upload, and MTP/DFlash/MM
// wired.  No qwen4exp-specific geometry is inherited.
#include "model_profile.h"

namespace si {

extern const model_profile kQwen35Profile = {
    .key = "qwen35",
    .name = "Qwen3.5 hybrid GDN + full attention",
    .provenance =
        "Architecture facts from src/model/qwen35.cpp and the 0.8B/27B GGUF metadata. The policy is "
        "the engine default measured in AGENTS.md (shared seg_plan/record_forward, chunked prefill, "
        "prefix cache on, no auto layer map, SIn/oneDNN weight stores). No qwen4exp geometry is "
        "inherited: the 0.8B is the tied-head reference and the 27B is the untied split reference.",

    .fwd = model_profile::fwd_kind::generic_plan,

    .residual = {.hyper = false, .streams = 0, .low_rank = 0},
    .extras = {.ple = false, .qsa = false, .qsa_top_k = 0, .qsa_pool = 0, .idx_heads = 0, .idx_head_dim = 0},

    .policy = {
        .max_decode_rows = 0, // the engine ceiling (kMaxB)
        .prefill = model_profile::prefill_kind::chunk32,
        .prefix_cache = true,
        .auto_layer_map = false,
        .md_splits = false,
        .engram_ram_mb_default = 0, // no PLE module
    },

    .rep = {
        .sin_w8 = true,
        .dnnl_int8 = true,
        .raw_gemv_decode = false,
        .split_per_tensor_upload = false,
    },

    .spec = {
        .mtp = true,
        .dflash = true,
        .multimodal = true,
        .why = "",
    },

    .moe = model_profile::moe_owner::engine_cache,
};

extern const model_profile kQwen35MoeProfile = {
    .key = "qwen35moe",
    .name = "Qwen3.6 35B-A3B hybrid MoE + NextN",
    .provenance =
        "Architecture facts from src/model/qwen35moe.cpp and the 35B-A3B GGUF metadata. The routed "
        "experts use the generic three-tier moe_cache (AGENTS.md's ncmoe measured at 48-66% RAM hit "
        "rate), not a forward-owned bank. The rest of the policy is identical to qwen35 because the "
        "dense half is the same hybrid trunk; test_model_profile asserts the two rows agree on "
        "everything except identity and the MoE note.",

    .fwd = model_profile::fwd_kind::generic_plan,

    .residual = {.hyper = false, .streams = 0, .low_rank = 0},
    .extras = {.ple = false, .qsa = false, .qsa_top_k = 0, .qsa_pool = 0, .idx_heads = 0, .idx_head_dim = 0},

    .policy = {
        .max_decode_rows = 0,
        .prefill = model_profile::prefill_kind::chunk32,
        .prefix_cache = true,
        .auto_layer_map = false,
        .md_splits = false,
        .engram_ram_mb_default = 0,
    },

    .rep = {
        .sin_w8 = true,
        .dnnl_int8 = true,
        .raw_gemv_decode = false,
        .split_per_tensor_upload = false,
    },

    .spec = {
        .mtp = true,
        .dflash = true,
        .multimodal = true,
        .why = "",
    },

    .moe = model_profile::moe_owner::engine_cache,
};

} // namespace si
