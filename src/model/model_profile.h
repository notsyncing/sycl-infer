#pragma once
// ---------------------------------------------------------------------------
// Model profiles: the declarative, per-architecture description of what a model
// *is* and which engine mechanisms it is allowed to use.
//
// WHY THIS EXISTS
// ---------------
// qwen4exp differs from the generic Qwen3.5 stack in ways that are not device
// tuning: it walks its own forward pass (hyper-connections, PLE, QSA, a
// per-expert LRU bank), it cannot use the prefix cache or MTP yet, it wants one
// decode row and a whole-prompt server prefill, and its file cannot be converted
// to the SIn/oneDNN weight stores.  Those decisions used to be per-model flags
// scattered through engine.cpp and scheduler.cpp.  A model profile
// puts them in one auditable place, the same way a device profile does for the
// GPU.
//
// THE BOUNDARY -- MODEL VS DEVICE
// -------------------------------
// A field belongs here when it is a property of the *model file / architecture*:
// the forward kind, HC/PLE/QSA geometry, whether the prefix cache can represent
// its state, which weight representations its file can use, whether MTP/DFlash/MM
// are wired, and which owner runs its routed experts.
//
// A field does NOT belong here when it is:
//   * a per-card tuning value (splits, work-group widths, u4/k5/cb4 defaults,
//     occupancy) - that is src/device/<profile>.cpp;
//   * a per-layer expert plane size or any VRAM/memory budget (`sg/su/sd`, the
//     bank budget, `--moe-vram-mb`, `--ctx`, KV type) - those are properties of
//     the specific file and the run, not of the architecture, and a profile that
//     carried them would silently serve the wrong strides for another file.
//
// The static profile is data.  Anything that needs per-engine state stays in the
// engine; selection is by the GGUF's exact `general.architecture` key (the
// loader registry is the single table), never by a fuzzy name matcher.
//
// An environment variable still overrides a profile *default* (PF_PREFIX_CACHE,
// PF_MD_SPLITS, --engram-ram-mb, ...): the profile is what the engine does
// unasked, the env/CLI is the A/B.
// ---------------------------------------------------------------------------
#include <cstddef>

namespace si {

struct model_profile {
    // How the forward pass is executed.  `generic_plan` is the shared
    // seg_plan/record_forward path; `hc_ple_qsa` is the qwen4exp walk
    // (arch/qwen4exp/forward.cpp), which cannot be expressed as the generic plan.
    enum class fwd_kind { generic_plan, hc_ple_qsa, deepseek41 };

    // Which vision tower an mmproj for this architecture is.  The loader
    // (src/mm/vision.cpp) is one module that implements every variant; this
    // field selects the per-architecture differences (fused vs separate q/k/v,
    // position embeddings, FFN/merger activation, head width) and is
    // cross-checked against the mmproj's own `clip.projector_type`.
    enum class vision_kind { none, qwen3vl, qwen2vl, deepseek41v };

    // How the scheduler feeds a prompt to prefill.  `chunk32` is the shared
    // chunker (kMaxT chunks, the batched mode-2 path); `whole_prompt` asks
    // prefill_text for the whole remaining prompt in one call, because the
    // model's forward has no generic mode-2 variant and chunks internally.
    enum class prefill_kind { chunk32, whole_prompt };

    // Who owns the routed experts.  `engine_cache` is the generic three-tier
    // moe_cache; `runtime_bank` is a forward-specific per-expert LRU bank;
    // `host_only` computes every routed pair on the host.
    enum class moe_owner { engine_cache, runtime_bank, host_only };

    const char * key = "";        // exact `general.architecture` value
    const char * name = "";       // human name, for the [model] banner
    const char * provenance = ""; // how these values were obtained

    fwd_kind fwd = fwd_kind::generic_plan;
    // Which vision tower this architecture's mmproj uses; `none` means the
    // architecture has no multimodal path at all.
    vision_kind vision = vision_kind::none;

    // Hyper-connections (qwen4exp): the residual is `streams` parallel copies of
    // the hidden state, mixed through a grouped RMSNorm and a `low_rank` gate.
    struct residual_t {
        bool hyper = false;
        int streams = 0;
        int low_rank = 0;
    } residual;

    // Per-layer extras: PLE (the n-gram-hash embedding table) and QSA (the
    // indexer plus its pooled top-k selector).
    struct extras_t {
        bool ple = false;
        bool qsa = false;
        int qsa_top_k = 0;   // keys the selector keeps per query
        int qsa_pool = 0;    // compress ratio (keys per pool), 0 = dense
        int idx_heads = 0;   // indexer query heads
        int idx_head_dim = 0;
    } extras;

    // Engine policy.  These are what the engine does unasked; an env var or a
    // CLI flag still overrides them for an A/B.
    struct policy_t {
        // Rows one decode forward may carry.  0 = the engine ceiling (kMaxB).
        // qwen4exp is 1: two rows select ~2x the distinct experts, so the
        // routed weight read doubles and batching measured 0.85x.
        int max_decode_rows = 0;
        prefill_kind prefill = prefill_kind::chunk32;
        // Can the checkpoint/blob represent this model's per-slot state?
        bool prefix_cache = true;
        // Synthesise a layer map when none was given (a split GGUF cannot use
        // the single-blob upload, so its per-tensor upload needs the multi-
        // device path even on one GPU).
        bool auto_layer_map = false;
        // Keep the synthesised map on a single GPU even when several are
        // present.  Right-sized for a small model: layer-pipeline parallelism
        // adds a handoff between the halves without reducing a batch-1 decode,
        // and it keeps the vision tower and the text trunk on one card.
        bool auto_layer_map_one_gpu = false;
        // Keep the attention K-split on a multi-device run.  The generic
        // multi-device path pins n_splits/dec_splits to 1; a model whose
        // forward derives its own split (qwen4exp) sets this.
        bool md_splits = false;
        // Default for the PLE engram row cache (`--engram-ram-mb`).  Only read
        // by a forward that has PLE; 0 = no cache.
        size_t engram_ram_mb_default = 0;
        // Keep the GGUF page cache across `page_out_host` (`PF_Q4_KEEPCACHE`).
        // A model whose startup copies tens of GiB into a pinned expert source
        // reads the file more than once and would otherwise pay the disk pass
        // on every start.  The PTEs are still dropped, so RSS is unchanged.
        bool keep_file_cache = false;
    } policy;

    // Weight representation the file can actually use.  These are capabilities,
    // not preferences: a Q8_0/Q5_1/BF16 mix cannot enter the SIn packer or the
    // oneDNN grouped-scale converter at all.
    struct rep_t {
        bool sin_w8 = true;               // build SIn int8 copies (dp4a)
        bool dnnl_int8 = true;            // convertible to oneDNN int8 weights
        bool raw_gemv_decode = false; // decode reads the raw GGUF blocks
        // Dequantize the decoder's linears to f32 and run the GEMMs on the
        // generic f32 kernel instead of the int8 oneDNN path.  A small
        // fine-tuned model can be sensitive to the int8 activation
        // quantization (Qwen2-VL/MinerU is), and f32 costs only ~15% of a
        // 1.2B decode's time.  Default off for every other architecture.
        bool f32_gemm = false;
        bool split_per_tensor_upload = false; // needs the per-tensor upload path
    } rep;

    // Mechanisms this architecture can use.  A false value is a declared
    // capability, not a silent fallback: the engine prints `why` once when a
    // user asks for the mechanism.
    struct spec_t {
        bool mtp = true;
        bool dflash = true;
        bool multimodal = true;
        // A multimodal prompt must go through the model's own forward chunker.
        // The generic mm driver feeds one kMaxT row per prefill_chunk, which is
        // exactly the path a model-owned forward (qwen4exp: hyper-connections,
        // PLE, QSA and the per-expert bank) bypasses; its whole-prompt
        // batching is where those live.  This is the declarative capability the
        // loader used to publish separately from the profile.
        bool mm_owned_prefill = false;
        const char * why = "";
    } spec;

    moe_owner moe = moe_owner::engine_cache;
};

// The name of a forward kind, for the PF_MODEL_INFO banner.  It lives here (not
// in the generic model.cpp) so the generic registry does not have to name an
// architecture to report it; the enum itself is the architecture vocabulary.
inline const char * fwd_kind_name(model_profile::fwd_kind k) {
    switch (k) {
    case model_profile::fwd_kind::hc_ple_qsa:
        return "hc_ple_qsa";
    case model_profile::fwd_kind::deepseek41:
        return "deepseek41";
    default:
        return "generic_plan";
    }
}

// One profile per `general.architecture` value, owned by that architecture's
// `src/model/arch/<arch>/profile.cpp` and published through its `load.h` as
// `<arch>_profile()`.  The internal table in `src/model/arch/registry.cpp` points
// at those; a new architecture needs both a loader and a profile, and
// model::load refuses a row whose key does not match the loader.

} // namespace si
