// Model profile: Qwen3.8-Flash-Next ("qwen4exp").
//
// The hybrid GDN/full-attention trunk plus hyper-connections, PLE and QSA, and a
// per-expert LRU bank.  Its own forward walk (arch/qwen4exp/forward.cpp) means the
// generic seg_plan/record_forward mechanisms cannot represent it, so the
// prefix cache and multimodal paths have their own hooks inside that walk:
//   * the checkpoint blob carries the GDN/conv state (the generic pool) plus
//     the PLE rolling history and the QSA indexer pool/ring (host state), and
//     the QSA host mirror is rebuilt from the restored KV blocks;
//   * the vision tower is the same qwen3vl mmproj the 27B uses
//     (`clip.projector_type = qwen3vl_merger`), and the model-owned chunker
//     (`spec.mm_owned_prefill`) slices M-RoPE and img_row per chunk.
//
// Provenance: the geometry is the reference GGUF's metadata
// (Qwen3.8-Flash-Next-UD-Q4_K_XL, 4 shards): hyper_connection.count=4 and
// low_rank=320; attention.indexer.head_count=4, key_length=128, top_k=2048 and
// compress_ratios=4 (docs/design/18-qwen4exp.md's QSA table).  The policy and the
// weight representation are the measured values in that document and AGENTS.md:
// 1 decode row (2 rows = 0.85x, 208.1 vs 176.5 ms/step on 2x A770 at the 17500 MB
// bank), whole-prompt server prefill (2.2x: 36.8-41.7 s vs 83.0-92.6 s on a
// 2412-token prompt), the qwen3vl vision tower (`clip.projector_type =
// qwen3vl_merger`, projection_dim 2560 = text n_embd), the prefix cache with
// PLE/QSA state in the checkpoint, the engram cache's 4096 MB default, and the
// raw-GGUF decode because Q8_0/Q5_1/BF16 do not convert to SIn or oneDNN int8.
//
// NOT here: any per-layer expert plane size, the bank budget, the KV type or the
// layer map.  Those are properties of a specific file/run (the file mixes Q4_K,
// Q5_K, Q5_1 and Q8_0 down projections, so even the strides are per layer).
#include "load.h"
#include "model_profile.h"

namespace si {

namespace {

const model_profile kQwen4ExpProfile = {
    .key = "qwen4exp",
    .name = "Qwen3.8-Flash-Next (HC + PLE + QSA)",
    .provenance =
        "Reference GGUF metadata: hyper_connection 4 streams / low_rank 320, indexer 4 heads x 128, "
        "top_k 2048 over pools of 4. Policy and weight values measured in docs/design/18-qwen4exp.md "
        "and AGENTS.md on 2x A770 (17500 MB bank, f16 KV): one decode row 0.85x for two, whole-prompt "
        "prefill 2.2x, prefix cache with PLE/QSA checkpoints, the qwen3vl mmproj (projection_dim 2560), "
        "4096 MB engram default, raw GGUF decode (Q8_0/Q5_1/BF16 are not SIn/oneDNN-convertible).",

    .fwd = model_profile::fwd_kind::hc_ple_qsa,
    .vision = model_profile::vision_kind::qwen3vl,

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
        .prefix_cache = true,
        .auto_layer_map = true,
        .auto_layer_map_one_gpu = false,
        .md_splits = true,
        .engram_ram_mb_default = 4096,
        .keep_file_cache = true,
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
        .multimodal = true,
        .mm_owned_prefill = true,
        .why =
            "qwen4exp's HC/PLE/QSA state has no draft/verify hooks and the trunk file carries no NextN "
            "block, so MTP/DFlash are declared unsupported; the model-owned chunker carries the "
            "per-chunk M-RoPE/img_row a multimodal prompt needs",
    },

    .moe = model_profile::moe_owner::runtime_bank,
};

} // namespace

const model_profile & qwen4exp_profile() { return kQwen4ExpProfile; }

} // namespace si
