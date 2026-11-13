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
#include "load.h"
#include "model_profile.h"

namespace si {

namespace {

const model_profile kQwen35Profile = {
    .key = "qwen35",
    .name = "Qwen3.5 hybrid GDN + full attention",
    .provenance =
        "Architecture facts from src/model/qwen35.cpp and the 0.8B/27B GGUF metadata. The policy is "
        "the engine default measured in AGENTS.md (shared seg_plan/record_forward, chunked prefill, "
        "prefix cache on, no auto layer map, SIn/oneDNN weight stores). No qwen4exp geometry is "
        "inherited: the 0.8B is the tied-head reference and the 27B is the untied split reference.",

    .fwd = model_profile::fwd_kind::generic_plan,
    .vision = model_profile::vision_kind::qwen3vl,

    .residual = {.hyper = false, .streams = 0, .low_rank = 0},
    .extras = {.ple = false, .qsa = false, .qsa_top_k = 0, .qsa_pool = 0, .idx_heads = 0, .idx_head_dim = 0},

    .policy = {
        .max_decode_rows = 0, // the engine ceiling (kMaxB)
        .prefill = model_profile::prefill_kind::chunk32,
        .prefix_cache = true,
        .auto_layer_map = false,
        .auto_layer_map_one_gpu = false,
        // MEASURED on the 35B / Arc 140V (2026-11-01): the generic multi-device
        // path pins dec_splits to 1, so a single-token decode runs one warp per
        // (token, head) over the whole KV - 0.07 T-MAC/s.  With the derived
        // split (24 on this card) the 3.2k-token decode is 34.2 -> 29.9
        // ms/token and the prefill is neutral.  The old pin existed for the
        // multi-device split bug (fixed by the nat staging index); the
        // decode-vs-prefill oracle is run with the split on.
        .md_splits = true,
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
        .mm_owned_prefill = false,
        .why = "",
    },

    .moe = model_profile::moe_owner::engine_cache,
};

} // namespace

const model_profile & qwen35_profile() { return kQwen35Profile; }

} // namespace si
