// Model profile: DeepSeek-V4.1-Flash ("deepseek41").
//
// A 40-layer, all-MoE decoder with a very different forward from every other
// architecture here:
//   * MLA attention: q is a 1280-wide low-rank projection expanded to 64 heads
//     of 512, and the single KV head is a 512-wide latent used as both K and V
//     (K-only cache; the output is de-roped and projected through 8 groups of
//     `wo_a` then `wo_b`).
//   * Compressed sparse attention: each layer keeps a 128-token sliding window
//     and reads a *group-shared* compressed KV cache.  Layers 2/8/14/20 own the
//     caches (ratio 2/2/2/1); later layers read the owner's rows.  A per-layer
//     DSA indexer selects the top 512 compressed rows, with a two-level
//     candidate stage at layer 20.
//   * 4-stream hyper-connections (Sinkhorn mixing) with a lagged pre-mix, and
//     Engram on layers 1 and 14 (an n-gram-hash gather from a 110 GiB table).
//   * 384 routed experts (top-6, sqrt-softplus gating) plus an ungated shared
//     expert, both with a one-sided SwiGLU clamp of 10.
//
// The forward is model-owned (arch/deepseek41/forward_host.cpp): the generic seg_plan cannot
// express the cross-layer compressed caches, the indexer or the lagged mHC.
//
// Provenance: geometry is the reference GGUF's metadata
// (DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS, 6 shards + Engram shard):
// 64 heads / 1 KV head, key_value_length 512, rope.dimension_count 64,
// q_lora_rank 1280, output_group_count 8 / output_lora_rank 1024,
// compress_ratios [0,0,2x18,1x20], kv_source_layers [2,8,14,20],
// indexer.source_layers [2,8,14,20,24,28,32,36], indexer 32x128 top_k 512,
// candidate_source_layer 20 / 2048 blocks x 8, hyper_connection 4 streams /
// 24 mix outputs / sinkhorn 20 / eps 1e-6, expert_count 384 / used 6 /
// shared 1, expert_feed_forward_length 2304, expert_weights_scale 1.5,
// expert_gating_func 4 (sqrt-softplus), swiglu clamp 10, ple.layers [1,14],
// ngram 4 / heads_per_ngram 8 / 256-wide rows, engram.pad_token_id 2.  The
// reference math is DeepSeek's inference/model.py as ported by Strata's
// tools/dsv41/ref.py; the llama.cpp `deepseek41` implementation agrees apart
// from the KV activation rounding (FP8/FP4 there, f16 here).
//
// NOT here: any VRAM budget, the expert bank size, the KV type or the layer
// map.  Those are properties of a run, not of the architecture.
#include "load.h"
#include "model_profile.h"

namespace si {

namespace {

const model_profile kDeepseek41Profile = {
    .key = "deepseek41",
    .name = "DeepSeek-V4.1-Flash (MLA + DSA + mHC + Engram)",
    .provenance =
        "Reference GGUF metadata (DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS): 64 heads / 1 KV head, "
        "kv 512, rope 64 of 512 (offset 448), q_lora 1280, out 8x1024, compress [0,0,2x18,1x20], "
        "kv sources [2,8,14,20], index sources [2,8,14,20,24,28,32,36], indexer 32x128 top_k 512, "
        "candidate layer 20 / 2048 blocks x 8, hyper-connection 4 streams / 24 / 20 iters / 1e-6, "
        "384 experts top-6 + 1 shared, ffn 2304, scale 1.5, sqrt-softplus gating, swiglu clamp 10, "
        "Engram layers [1,14] (ngram 4, 8 heads per ngram, 256-wide). Reference math: DeepSeek "
        "inference/model.py via Strata tools/dsv41/ref.py; llama.cpp src/models/deepseek41.cpp. "
        "No throughput measurement yet on this platform.",

    .fwd = model_profile::fwd_kind::deepseek41,
    .vision = model_profile::vision_kind::deepseek41v,

    // mHC: 4 streams and 24 mix outputs ((2+4)*4); `low_rank` is that width.
    .residual = {.hyper = true, .streams = 4, .low_rank = 24},
    .extras = {
        .ple = true,
        .qsa = true, // DSA: the lightning indexer + compressed-row top-k
        .qsa_top_k = 512,
        .qsa_pool = 2,
        .idx_heads = 32,
        .idx_head_dim = 128,
    },

    .policy = {
        // The compressed caches are per-sequence streams and the model is
        // weight-bound (155 GiB of text weights against 2x16 GiB VRAM), so one
        // sequence is the honest unit until a multi-stream cache exists.
        .max_decode_rows = 1,
        .prefill = model_profile::prefill_kind::whole_prompt,
        .prefix_cache = true,
        .auto_layer_map = true,
        .auto_layer_map_one_gpu = false,
        .md_splits = true,
        // The Engram table is 110 GiB; 8 GiB of direct-mapped host rows is the
        // default working set (hit rate ~ cap/rows), overridable per run.
        .engram_ram_mb_default = 8192,
        .keep_file_cache = true,
    },

    .rep = {
        // IQ2_XXS (routed experts) and IQ4_NL (Engram) are not SIn- or
        // oneDNN-int8-convertible; the dense half is Q8_0/Q5_K/F32/BF16.
        .sin_w8 = false,
        .dnnl_int8 = false,
        .raw_gemv_decode = true,
        .split_per_tensor_upload = true, // 6-shard split GGUF
    },

    .spec = {
        .mtp = false,
        .dflash = false,
        .multimodal = true,
        .mm_owned_prefill = false,
        .why =
            "no NextN block is bundled in the trunk shards (the DSpark drafter is a separate GGUF and "
            "would need its own forward); multimodal is the deepseek41v mmproj (32-block vision tower, "
            "3x3 merge projector, img_start/img_end/newline sentinels)",
    },

    .moe = model_profile::moe_owner::runtime_bank,
};

} // namespace

const model_profile & deepseek41_profile() { return kDeepseek41Profile; }

} // namespace si
