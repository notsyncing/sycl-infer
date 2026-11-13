#pragma once
// DeepSeek-V4.1-Flash ("deepseek41") per-layer and model-level tensor views.
// They live in `layer_t::extras` / `model::extras` (see
// arch/deepseek41/accessors.h); the scalar hparams are in
// arch/deepseek41/hparams.h.
#include <vector>

#include "model_types.h"

namespace si {

// DeepSeek-V4.1-Flash layer tensors.  The attention is MLA: `wq_a`/`wq_b` is a
// low-rank q, `wkv` is the single 512-wide latent used as both K and V, and the
// output is `wo_a` (8 groups of 1024) then `wo_b`.  `wo_a` is stored 3-D
// [K=4096][N=1024][E=8], so one group is a plain 2-D weight.
struct ds41_layer : arch_layer {
    ds41_layer() {
        k = kind::deepseek41;
    }
    wt wq_a, wq_b, wkv, wo_b;
    // attn_output_a is stored 2-D [K=4096][N=8192]; the eight output groups are
    // consecutive 1024-row blocks of N (the reference reshapes it to
    // [4096, 1024, 8] for the per-group GEMV).
    wt wo_a;
    const float * q_a_norm = nullptr;
    const float * kv_norm = nullptr;
    const float * attn_sinks = nullptr; // [n_head]
    // compressed-KV owner only (kv_src_of[il] == il)
    wt comp_wkv, comp_wgate; // comp_wgate absent at ratio 1
    const float * comp_norm = nullptr;
    wt idx_attn_k; // [512,128], owner only
    const float * idx_k_norm = nullptr;
    // index source only (topk_src_of[il] == il)
    wt idx_attn_q_b, idx_proj;
    // Engram layer only (1, 14)
    wt engram_kv; // [6144, 5*n_embd]
    const float * engram_gate = nullptr;
    // hyper-connections
    wt hc_attn_fn, hc_ffn_fn; // [hc*n_embd, (2+hc)*hc]
    const float * hc_attn_base = nullptr;
    const float * hc_attn_scale = nullptr;
    const float * hc_ffn_base = nullptr;
    const float * hc_ffn_scale = nullptr;
    // router biases (the `_vl` variant is used for image-span tokens)
    const float * exp_probs_b = nullptr;
    const float * exp_probs_b_vl = nullptr;

    // Defined in arch/deepseek41/load.cpp (it resolves the source arrays
    // through ds41_hp, so the definition needs the hparams header).
    void visit_weights(const hparams &hp, int il, const std::function<void(const void *)> &cb) const override;
};

struct ds41_model_extras : arch_model_extras {
    ds41_model_extras() {
        k = kind::deepseek41;
    }
    // The vocab -> n-gram-id map the Engram hash uses (a GGUF metadata array,
    // not a tensor).  Host-only.
    std::vector<int32_t> engram_token_map;
    // The Engram table (110 GiB): gathered on the host, never uploaded.
    wt ple_table;

    // The Engram table is gathered on the host on every run, so the flag does
    // not matter here.
    void visit_host_keep(bool, const std::function<void(const void *)> &cb) const override {
        cb(ple_table.data);
    }
};

} // namespace si
