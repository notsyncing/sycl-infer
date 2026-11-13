// Model profile: Qwen2-VL (MinerU2.5-Pro-2605-1.2B).
//
// The text trunk is the engine's generic seg_plan/record_forward path: a plain
// dense decoder where every layer is full GQA attention plus a SwiGLU FFN.
// Nothing here uses the Qwen3.5-only machinery (GDN recurrence, attention
// output gate, QK RMSNorm, NextN/MTP, DFlash), and the multimodal tower is a
// separate mmproj consumed by the MinerU path in src/mm/.
//
// Weight representation: the file is Q5_0/Q8_0/Q4_K/Q6_K.  Q5_0 is not a
// K-quant and the SIn packer does not accept it, so sin_w8 is false; the oneDNN
// grouped-scale int8 conversion (which reads the GGUF blocks through
// dequantize_row) handles every type, so dnnl_int8 is true.  The prefix cache
// is disabled because the MinerU prompts are per-page crops and never repeat,
// and because the no-GDN checkpoint has no state to restore.
#include "load.h"
#include "model_profile.h"

namespace si {

namespace {

const model_profile kQwen2VlProfile = {
    .key = "qwen2vl",
    .name = "Qwen2-VL dense decoder (MinerU2.5)",
    .provenance =
        "Geometry from the MinerU2.5-Pro-2605-1.2B GGUF metadata: 24 layers, n_embd 896, "
        "ffn 4864, 14 heads / 2 kv heads (head_dim 64), rope sections [8,12,12,0] at base 1e6, "
        "full attention on every layer.  Weight types in the mradermacher Q4_K_M file are "
        "Q5_0/Q8_0/Q4_K/Q6_K (K=896 is not a multiple of 256, so ggml uses Q5_0/Q8_0 there). "
        "Policy: the shared generic plan, chunked prefill, oneDNN int8 weights, no SIn packer, "
        "no prefix cache (per-crop prompts), no NextN block and no DFlash draft.",

    .fwd = model_profile::fwd_kind::generic_plan,
    .vision = model_profile::vision_kind::qwen2vl,

    .residual = {.hyper = false, .streams = 0, .low_rank = 0},
    .extras = {.ple = false, .qsa = false, .qsa_top_k = 0, .qsa_pool = 0, .idx_heads = 0, .idx_head_dim = 0},

    .policy = {
        .max_decode_rows = 0, // the engine ceiling (kMaxB)
        .prefill = model_profile::prefill_kind::chunk32,
        .prefix_cache = false,
        .auto_layer_map = true,
        .auto_layer_map_one_gpu = true,
        .md_splits = false,
        .engram_ram_mb_default = 0,
    },

    .rep = {
        .sin_w8 = false,             // Q5_0 is not SIn-packable; oneDNN reads it
        .dnnl_int8 = true,
        .raw_gemv_decode = false,
        .f32_gemm = false, // PF_F32=1 is the f32 accuracy A/B
        .split_per_tensor_upload = false,
    },

    .spec = {
        .mtp = false,
        .dflash = false,
        .multimodal = true,
        .mm_owned_prefill = false,
        .why = "the file has no NextN block and no DFlash draft",
    },

    .moe = model_profile::moe_owner::engine_cache,
};

} // namespace

const model_profile & qwen2vl_profile() { return kQwen2VlProfile; }

} // namespace si
