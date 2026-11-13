#pragma once
// DeepSeek-V4.1-Flash ("deepseek41") architecture-specific hyper-parameters.
// Attached to the generic `hparams` through `hparams::arch` by the loader, so
// the generic struct carries no deepseek41 geometry; the forward reads it
// through `ds41_hp(...)`.
#include <cassert>
#include <cstdint>

#include "model.h"

namespace si {

// ---- DeepSeek-V4.1-Flash (`deepseek41`) --------------------------------
// MLA attention (512-wide latent, K == V), compressed sparse attention with a
// two-level DSA indexer, group-shared KV caches, 4-stream hyper-connections and
// Engram.  The per-layer source/ratio arrays are resolved at load time
// (`kv_src_of`/`topk_src_of`) so no use site re-derives the "greatest source
// <= il" rule, and the hash constants are stored per Engram layer.
struct ds41_hparams : arch_hparams {
    ds41_hparams() {
        k = kind::deepseek41;
    }

    // MLA attention's head is 512 wide, which the classic kernel's 256-wide
    // register array cannot serve; ds41 has its own attention kernels.
    bool wide_head_ok() const override {
        return true;
    }
    // MLA geometry checks (implemented in load.cpp) and the profile cross-check.
    void validate(const hparams & hp, const std::string & prefix) const override;
    void validate_profile(const hparams & hp, const model_profile & prof) const override;

    int q_lora_rank = 0;     // 1280
    int out_group_count = 0; // 8
    int out_lora_rank = 0;   // 1024
    int sliding_window = 0;  // 128 (includes self)
    int kv_dim = 0;          // 512 == key_length == value_length
    int rot_off = 0;         // kv_dim - n_rot (448): RoPE on the LAST n_rot dims
    float rope_base = 0.f;   // ratio-0 layers
    float compress_rope_base = 0.f; // ratio>0 layers (160000)
    float yarn_factor = 0.f;
    int yarn_orig = 0;
    float yarn_beta_fast = 0.f, yarn_beta_slow = 0.f;

    int idx_n_head = 0, idx_head_dim = 0, idx_top_k = 0;
    int cand_src = -1, cand_topk_blocks = 0, cand_block = 0;
    int n_idx_src = 0, idx_src[64] = {0};
    int topk_src_of[64] = {0}; // greatest index source <= il (-1 before the first)

    int n_kv_src = 0, kv_src[64] = {0};
    int kv_src_of[64] = {0}; // greatest KV source <= il (-1 before the first)
    int compress_ratio[64] = {0};

    int hc_mult = 0, hc_mix_dim = 0, hc_sinkhorn = 0;
    float hc_eps = 0.f;

    int n_expert_shared = 0;
    float expert_scale = 1.f;
    bool expert_norm = false;
    float swiglu_clamp[64] = {0.f};
    float swiglu_clamp_shexp[64] = {0.f};

    bool is_ple[64] = {false};
    int engram_index_of[64] = {0}; // number of Engram layers strictly before il
    int ple_ngram = 0, ple_heads_per_ngram = 0, ple_head_dim = 0, ple_n_heads = 0;
    uint64_t ple_mult[8] = {0};
    uint32_t ple_off[64] = {0}, ple_vocab[64] = {0};
    int engram_pad = 0;
    int image_token_id = -1;
};

inline ds41_hparams & ds41_hp(hparams & hp) {
    assert(hp.arch && hp.arch->k == arch_hparams::kind::deepseek41);
    return static_cast<ds41_hparams &>(*hp.arch);
}

inline const ds41_hparams & ds41_hp(const hparams & hp) {
    assert(hp.arch && hp.arch->k == arch_hparams::kind::deepseek41);
    return static_cast<const ds41_hparams &>(*hp.arch);
}

} // namespace si
