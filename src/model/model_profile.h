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
// to the SIn/oneDNN weight stores.  Those decisions used to be `m.q4exp`
// ternaries scattered through engine.cpp and scheduler.cpp.  A model profile
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
    // (engine_q4exp.cpp), which cannot be expressed as the generic plan.
    enum class fwd_kind { generic_plan, hc_ple_qsa };

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
        // Keep the attention K-split on a multi-device run.  The generic
        // multi-device path pins n_splits/dec_splits to 1; a model whose
        // forward derives its own split (qwen4exp) sets this.
        bool md_splits = false;
        // Default for the PLE engram row cache (`--engram-ram-mb`).  Only read
        // by a forward that has PLE; 0 = no cache.
        size_t engram_ram_mb_default = 0;
    } policy;

    // Weight representation the file can actually use.  These are capabilities,
    // not preferences: a Q8_0/Q5_1/BF16 mix cannot enter the SIn packer or the
    // oneDNN grouped-scale converter at all.
    struct rep_t {
        bool sin_w8 = true;               // build SIn int8 copies (dp4a)
        bool dnnl_int8 = true;            // convertible to oneDNN int8 weights
        bool raw_gemv_decode = false;     // decode reads the raw GGUF blocks
        bool split_per_tensor_upload = false; // needs the per-tensor upload path
    } rep;

    // Mechanisms this architecture can use.  A false value is a declared
    // capability, not a silent fallback: the engine prints `why` once when a
    // user asks for the mechanism.
    struct spec_t {
        bool mtp = true;
        bool dflash = true;
        bool multimodal = true;
        const char * why = "";
    } spec;

    moe_owner moe = moe_owner::engine_cache;
};

// One profile per `general.architecture` value.  The registry rows in
// src/model/model.cpp point at these; a new architecture needs both a loader and
// a profile, and model::load refuses a row whose key does not match the loader.
extern const model_profile kQwen35Profile;    // qwen35
extern const model_profile kQwen35MoeProfile; // qwen35moe
extern const model_profile kQwen4ExpProfile;  // qwen4exp

} // namespace si
