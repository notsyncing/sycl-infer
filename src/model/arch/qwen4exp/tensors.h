#pragma once
// Qwen3.8-Flash-Next ("qwen4exp") per-layer and model-level tensor views: the
// hyper-connection mixers, the PLE module, the QSA indexer and the final HC
// head.  These live in `layer_t::extras` / `model::extras` (see
// arch/qwen4exp/accessors.h); the scalar hparams are in
// arch/qwen4exp/hparams.h.
#include "model_types.h"

namespace si {

// A hyper-connection module: grouped RMSNorm gamma [hc][n_embd], the low-rank
// gate down/up, and the per-stream scatter projection [hc].
struct hc_wt {
    const float * norm = nullptr;
    wt down, up, inject;
    bool ok() const {
        return norm && down.data && up.data && inject.data;
    }
};

// PLE (per-layer n-gram embedding) tensors on the one layer that has them.
struct ple_wt {
    wt key, value;
    const float * norm_key = nullptr;
    const float * norm_query = nullptr;
    const float * norm_conv = nullptr;
    const float * conv = nullptr; // depthwise taps [ple_conv][hc*n_embd]
    bool ok() const {
        return key.data && value.data && norm_key && norm_query && norm_conv && conv;
    }
};

// QSA (query-side sparse attention) indexer projections on a full-attention
// layer.  The stored keys are 128-wide, four indexer heads.
struct indexer_wt {
    wt q_proj, k_proj;
    const float * q_norm = nullptr;
    const float * k_norm = nullptr;
    bool ok() const {
        return q_proj.data && k_proj.data && q_norm && k_norm;
    }
};

struct qwen4exp_layer : arch_layer {
    qwen4exp_layer() {
        k = kind::qwen4exp;
    }
    hc_wt hc_attn, hc_ffn;
    ple_wt ple;
    indexer_wt idx;
    // The routed down projection converted once to a symmetric int8 store with
    // one scale per 32-value group (the Q5_1/Q8_0 mixed bank cannot be consumed
    // by the K-quant int8 GEMM otherwise).  Host-only.
    uint8_t * down_i8 = nullptr;
    float * down_sc = nullptr;
    int down_i8_K = 0, down_i8_N = 0, down_i8_E = 0;
    void visit_weights(const hparams &, int, const std::function<void(const void *)> &cb) const override;
};

struct qwen4exp_model_extras : arch_model_extras {
    qwen4exp_model_extras() {
        k = kind::qwen4exp;
    }
    hc_wt hc_head;
    // The PLE n-gram hash table (26.8 GiB in the reference Q4 file): host-only.
    wt ple_table;
    void visit_device_weights(const std::function<void(const void *)> &cb) const override;
    void visit_host_keep(bool cpu_partition, const std::function<void(const void *)> &cb) const override;
};

// The enumeration order matches this architecture's historical upload order
// (hc_attn, hc_ffn, PLE, indexer); order affects only which pointer is
// inserted first.
inline void qwen4exp_layer::visit_weights(const hparams &, int,
                                          const std::function<void(const void *)> &cb) const {
    cb(hc_attn.norm);
    cb(hc_attn.down.data);
    cb(hc_attn.up.data);
    cb(hc_attn.inject.data);
    cb(hc_ffn.norm);
    cb(hc_ffn.down.data);
    cb(hc_ffn.up.data);
    cb(hc_ffn.inject.data);
    if (ple.ok()) {
        cb(ple.key.data);
        cb(ple.value.data);
        cb(ple.norm_key);
        cb(ple.norm_query);
        cb(ple.norm_conv);
        cb(ple.conv);
    }
    if (idx.ok()) {
        cb(idx.q_proj.data);
        cb(idx.k_proj.data);
        cb(idx.q_norm);
        cb(idx.k_norm);
    }
}

inline void qwen4exp_model_extras::visit_device_weights(const std::function<void(const void *)> &cb) const {
    cb(hc_head.norm);
    cb(hc_head.down.data);
    cb(hc_head.up.data);
    // ple_table is intentionally NOT uploaded: 26.8 GiB, gathered on the host.
}

inline void qwen4exp_model_extras::visit_host_keep(bool cpu_partition,
                                                   const std::function<void(const void *)> &cb) const {
    // The HC head is uploaded to the device; only a CPU partition's host
    // kernels read the mmap copy.  The PLE table is gathered on the host on
    // every run, so it is always kept.
    if (cpu_partition) {
        cb(hc_head.norm);
        cb(hc_head.down.data);
        cb(hc_head.up.data);
    }
    cb(ple_table.data);
}

} // namespace si
