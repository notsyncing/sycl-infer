// Model profile: the Qwen3.6 35B-A3B Mixture-of-Experts variant of the
// Qwen3.5 hybrid trunk.  Split out of the shared generic profile file so each
// architecture owns its own translation unit; the const is unchanged.
#include "load.h"
#include "model_profile.h"

namespace si {

namespace {

const model_profile kQwen35MoeProfile = {
    .key = "qwen35moe",
    .name = "Qwen3.6 35B-A3B hybrid MoE + NextN",
    .provenance =
        "Architecture facts from src/model/qwen35moe.cpp and the 35B-A3B GGUF metadata. The routed "
        "experts use the generic three-tier moe_cache (AGENTS.md's ncmoe measured at 48-66% RAM hit "
        "rate), not a forward-owned bank. The rest of the policy is identical to qwen35 because the "
        "dense half is the same hybrid trunk; test_model_profile asserts the two rows agree on "
        "everything except identity and the MoE note.",

    .fwd = model_profile::fwd_kind::generic_plan,
    .vision = model_profile::vision_kind::qwen3vl,

    .residual = {.hyper = false, .streams = 0, .low_rank = 0},
    .extras = {.ple = false, .qsa = false, .qsa_top_k = 0, .qsa_pool = 0, .idx_heads = 0, .idx_head_dim = 0},

    .policy = {
        .max_decode_rows = 0,
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
        .mm_owned_prefill = false,
        .why = "",
    },

    .moe = model_profile::moe_owner::engine_cache,
};

} // namespace

const model_profile & qwen35moe_profile() { return kQwen35MoeProfile; }

} // namespace si
