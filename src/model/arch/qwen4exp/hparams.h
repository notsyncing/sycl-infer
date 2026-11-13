#pragma once
// Qwen3.8-Flash-Next ("qwen4exp") architecture-specific hyper-parameters.
// Attached to the generic `hparams` through `hparams::arch` by the loader, so
// the generic struct carries no qwen4exp geometry.  Included only from q4 code
// (the loader, the forward walk and the QSA host path), which reads the fields
// through `q4_hp(...)`.
#include <cassert>

#include "model.h"

namespace si {

// ---- Qwen3.8-Flash-Next / qwen4exp -------------------------------------
// Hyper-connections replace every layer norm: the residual stream is `hc`
// parallel copies of the hidden state and each block mixes them through a
// low-rank gate.  PLE is an n-gram-hash embedding table on one layer; QSA is a
// sparse selector on the full-attention layers.  See arch/qwen4exp/load.cpp.
struct qwen4exp_hparams : arch_hparams {
    qwen4exp_hparams() {
        k = kind::qwen4exp;
    }

    // The profile cross-check (implemented in load.cpp).  The generic geometry
    // check already covers qwen4exp's head.
    void validate_profile(const hparams & hp, const model_profile & prof) const override;

    int hc_mult = 0;
    int hc_low_rank = 0;
    int idx_n_head = 0;
    int idx_head_dim = 0;
    int idx_top_k = 0;
    int idx_kpool = 0; // QSA compress ratio (0 = dense)
    int ple_layer = -1;
    int ple_n_heads = 0;
    int ple_head_dim = 0;
    int ple_ngram = 0;
    int ple_heads_per_ngram = 0;
    int ple_conv = 0;
    int ple_eos = 0;
    int ple_image = 0;
    int64_t ple_mult[8] = {0};
    uint32_t ple_off[64] = {0};
    uint32_t ple_vocab[64] = {0};
    bool gdn_sigmoid_gate = false; // qwen4exp's gated norm uses sigmoid, not silu
};

inline qwen4exp_hparams & q4_hp(hparams & hp) {
    assert(hp.arch && hp.arch->k == arch_hparams::kind::qwen4exp);
    return static_cast<qwen4exp_hparams &>(*hp.arch);
}

inline const qwen4exp_hparams & q4_hp(const hparams & hp) {
    assert(hp.arch && hp.arch->k == arch_hparams::kind::qwen4exp);
    return static_cast<const qwen4exp_hparams &>(*hp.arch);
}

} // namespace si
