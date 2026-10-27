// Model profile: Qwen3.8-Flash-Next ("qwen4exp").
//
// The hybrid GDN/full-attention trunk plus hyper-connections, PLE and QSA, and a
// per-expert LRU bank.  Its own forward walk (engine_q4exp.cpp) means the
// generic seg_plan/record_forward mechanisms cannot represent it yet, so the
// capabilities below are declared unsupported with the reason the engine prints
// when a user asks for them -- not silently left to fail.
//
// Provenance: the geometry is the reference GGUF's metadata
// (Qwen3.8-Flash-Next-UD-Q4_K_XL, 4 shards): hyper_connection.count=4 and
// low_rank=320; attention.indexer.head_count=4, key_length=128, top_k=2048 and
// compress_ratios=4 (docs/design/18-qwen4exp.md's QSA table).  The policy and the
// weight representation are the measured values in that document and AGENTS.md:
// 1 decode row (2 rows = 0.85x, 208.1 vs 176.5 ms/step on 2x A770 at the 17500 MB
// bank), whole-prompt server prefill (2.2x: 36.8-41.7 s vs 83.0-92.6 s on a
// 2412-token prompt), no prefix cache (the PLE history and QSA mirror are not in
// the checkpoint blob), the engram cache's 4096 MB default, and the raw-GGUF
// decode because Q8_0/Q5_1/BF16 do not convert to SIn or oneDNN int8.
//
// NOT here: any per-layer expert plane size, the bank budget, the KV type or the
// layer map.  Those are properties of a specific file/run (the file mixes Q4_K,
// Q5_K, Q5_1 and Q8_0 down projections, so even the strides are per layer).
#include "model_profile.h"

namespace si {

extern const model_profile kQwen4ExpProfile = {
    .key = "qwen4exp",
    .name = "Qwen3.8-Flash-Next (HC + PLE + QSA)",
    .provenance =
        "Reference GGUF metadata: hyper_connection 4 streams / low_rank 320, indexer 4 heads x 128, "
        "top_k 2048 over pools of 4. Policy and weight values measured in docs/design/18-qwen4exp.md "
        "and AGENTS.md on 2x A770 (17500 MB bank, f16 KV): one decode row 0.85x for two, whole-prompt "
        "prefill 2.2x, no prefix cache (PLE/QSA state not checkpointed), 4096 MB engram default, raw "
        "GGUF decode (Q8_0/Q5_1/BF16 are not SIn/oneDNN-convertible).",

    .fwd = model_profile::fwd_kind::hc_ple_qsa,

    .residual = {.hyper = true, .streams = 4, .low_rank = 320},
    .extras = {
        .ple = true,
        .qsa = true,
        .qsa_top_k = 2048,
        .qsa_pool = 4,
        .idx_heads = 4,
        .idx_head_dim = 128,
    },

    .policy = {
        .max_decode_rows = 1,
        .prefill = model_profile::prefill_kind::whole_prompt,
        .prefix_cache = false,
        .auto_layer_map = true,
        .md_splits = true,
        .engram_ram_mb_default = 4096,
    },

    .rep = {
        .sin_w8 = false,
        .dnnl_int8 = false,
        .raw_gemv_decode = true,
        .split_per_tensor_upload = true,
    },

    .spec = {
        .mtp = false,
        .dflash = false,
        // The text tower can consume image/audio rows, but the server's
        // whole-prompt q4 prefill cannot carry per-chunk M-RoPE and the PLE
        // image positions are not wired: declared unsupported until that hook
        // exists rather than enabled and wrong.
        .multimodal = false,
        .why =
            "qwen4exp's HC/PLE/QSA state is not in the checkpoint/verify/blob hooks: no NextN block in "
            "the trunk file, the draft/verify needs per-token GDN/conv snapshots q4_forward does not "
            "produce, and the multimodal path needs per-chunk M-RoPE the whole-prompt prefill cannot "
            "carry",
    },

    .moe = model_profile::moe_owner::runtime_bank,
};

} // namespace si
