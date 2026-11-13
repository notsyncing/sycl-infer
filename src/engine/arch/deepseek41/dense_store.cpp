// DeepSeek-V4.1-Flash's dense linear stores, moved out of generic engine.cpp so
// the generic engine has no architecture-specific store setup.  The member
// declaration stays in engine.h (it is one architecture method, and the
// architecture-boundary gate excludes the header).
#include "engine.h"

#include "arch/deepseek41/accessors.h"
#include "arch/deepseek41/hparams.h"
#include "quant.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "common/env.h"

namespace si {

// DeepSeek-V4.1-Flash's dense linear stores (see engine.h).  Its GPU forward is
// per-token M=1 and reads every dense weight through ds41_gpu_state::gemv1, so
// the generic seg_plan stores never applied to it: Q5_K went through the raw
// fp32-dequant GEMV at 110 GB/s.  The k5 store keeps the native 5-bit values
// (no re-quantization, 0.75 B/w) and its M=1 GEMV measures 249 GB/s.  Q8_0 has
// no native store; PF_DS41_DENSE_Q8 registers the ordinary per-32-group int8
// conversion for it (measured worst per-row rel 2.2e-4, but neutral on time).
//
// The key is the host tensor pointer: a converted tensor skips its raw device
// upload, so engine::wkey(dev, t.data) returns the host pointer - and that is
// what the forward passes to gemv1.  PF_DS41_DENSE_STORE=0 restores the raw
// GEMV (the exact A/B: nothing is registered and the raw upload happens).
bool engine::setup_ds41_dense_store() {
    if (!dnnl_gemm_enabled(q)) {
        return false;
    }
    const char * es = si::env::str("PF_DS41_DENSE_STORE");
    if (es && atoi(es) == 0) {
        return false;
    }
    // Q8_0 has no native store; the ordinary per-32-group int8 conversion is
    // *not* the 0.98% generic re-quantization here: Q8_0 already carries one
    // f16 scale per 32 values, so the conversion keeps its own grid (measured
    // worst per-row rel 2.2e-4 against the same fp32 dot, below the k5 store's
    // 1.1e-3, because the activation quantization dominates).  It is an A/B and
    // **off** by default: the interleaved 3-rep cell measured decode 124.3
    // (k5 only) vs 125.0 (with the 80 int8 stores) ms/token, i.e. no time win
    // here - the Q8_0 projections are 372 MB of a 6.3 GB dense stream - so the
    // default keeps the raw Q8_0 GEMV and its exact weights.
    const bool add_q8 = si::env::flag("PF_DS41_DENSE_Q8");
    // Diagnostic bisection: PF_DS41_DENSE_ONLY=wq_b,wo_a,... registers only the
    // named groups (the others keep their raw upload and GEMV), so a store that
    // changes the stream can be isolated to one tensor class.
    const char * only = si::env::str("PF_DS41_DENSE_ONLY");
    auto want = [&](const char * tag) { return !only || std::strstr(only, tag) != nullptr; };
    const ds41_hparams & d = ds41_hp(m.hp);
    dnnl_dev_.clear();
    dnnl_dev_.resize(backends_.size());
    for (size_t dd = 0; dd < backends_.size(); dd++) {
        if (dev_kind_[dd] != 0 || !dev_queues_[dd]) {
            continue; // the ds41 GPU forward requires an all-GPU layer map
        }
        // decode-only: the forward quantizes one row per GEMV, so the
        // prefill-capacity scratch (360 MB/card) and the k5 prefill expansion
        // scratch (662 MB for the head alone) would be dead VRAM.
        dnnl_dev_[dd] = std::make_unique<dnnl_gemm>(*dev_queues_[dd], 2, 8192);
    }
    size_t n_k5 = 0, n_i8 = 0;
    uint64_t bytes = 0;
    // the stores are device memory this engine owns, so they must be in the
    // per-device accounting the control plane reports as free memory
    std::vector<uint64_t> dev_bytes(backends_.size(), 0);
    // `key` identifies the store and equals the pointer gemv1 receives;
    // `host_data` is where the packed rows start (the tensor base, or a group's
    // byte offset inside a stacked tensor such as wo_a).
    auto reg_k5 = [&](int dev, const void * key, const void * host_data, uint32_t type, int K, int N) {
        dnnl_gemm * D = dnnl_for(dev);
        if (!D || !key || !host_data || type != 13 || K <= 0 || N <= 0 || (K % 32) != 0) {
            return;
        }
        if (D->add_weight_k5(key, host_data, type, K, N)) {
            const uint64_t b =
                (uint64_t)((double)K * (double)N * 0.75) + 2ull * (uint64_t)(K / 32) * (uint64_t)N * 2;
            n_k5++;
            bytes += b;
            if ((size_t)dev < dev_bytes.size()) {
                dev_bytes[(size_t)dev] += b;
            }
        }
    };
    auto reg_i8 = [&](int dev, const wt & t) {
        dnnl_gemm * D = dnnl_for(dev);
        if (!D || !t.data || t.type != 8 || t.K <= 0 || t.N <= 0) {
            return;
        }
        if (D->add_weight(t.data, t.data, t.type, t.K, t.N, /*eager=*/false, /*lazy=*/true)) {
            const uint64_t b = (uint64_t)t.K * (uint64_t)t.N + 2ull * (uint64_t)(t.K / 32) * (uint64_t)t.N;
            n_i8++;
            bytes += b;
            if ((size_t)dev < dev_bytes.size()) {
                dev_bytes[(size_t)dev] += b;
            }
        }
    };
    for (int li = 0; li < m.hp.n_layer; li++) {
        const int dev = layer_dev_[(size_t)li];
        if (dev < 0 || (size_t)dev >= dev_kind_.size() || dev_kind_[(size_t)dev] != 0) {
            continue;
        }
        const layer_t & L = m.layers[(size_t)li];
        const ds41_layer & D = ds41_layer_of(L);
        if (want("wq_b")) {
            reg_k5(dev, D.wq_b.data, D.wq_b.data, D.wq_b.type, D.wq_b.K, D.wq_b.N);
        }
        if (want("wo_b")) {
            reg_k5(dev, D.wo_b.data, D.wo_b.data, D.wo_b.type, D.wo_b.K, D.wo_b.N);
        }
        // wo_a is one tensor holding out_group_count stacked groups; the GEMV
        // is called once per group with a byte offset into it, so each group is
        // its own store keyed at that offset.
        if (want("wo_a")) {
            const int out_g = d.out_group_count > 0 ? d.out_group_count : 1;
            const int out_rank = d.out_lora_rank > 0 ? d.out_lora_rank : 0;
            const int gsz = m.hp.n_head / out_g * d.kv_dim;
            const uint64_t row_b = (uint64_t)ggml_row_bytes(D.wo_a.type, (uint64_t)gsz);
            for (int g = 0; g < out_g; g++) {
                const char * key = (const char *)D.wo_a.data + (size_t)g * (size_t)out_rank * row_b;
                reg_k5(dev, key, key, D.wo_a.type, gsz, out_rank);
            }
        }
        if (want("ffn")) {
            reg_k5(dev, L.ffn_gate.data, L.ffn_gate.data, L.ffn_gate.type, L.ffn_gate.K, L.ffn_gate.N);
            reg_k5(dev, L.ffn_up.data, L.ffn_up.data, L.ffn_up.type, L.ffn_up.K, L.ffn_up.N);
            reg_k5(dev, L.ffn_down.data, L.ffn_down.data, L.ffn_down.type, L.ffn_down.K, L.ffn_down.N);
        }
        if (want("engram") && D.engram_kv.data) {
            reg_k5(dev, D.engram_kv.data, D.engram_kv.data, D.engram_kv.type, D.engram_kv.K, D.engram_kv.N);
        }
        if (add_q8) {
            reg_i8(dev, D.wq_a);
            reg_i8(dev, D.wkv);
        }
    }
    // The LM head runs on the primary device.  A tied head shares token_embd's
    // bytes, which the embed kernel reads as raw GGUF rows on the host, so it
    // cannot be replaced by a store.
    if (want("head") && m.output.data && m.output.data != m.tok_embd.data) {
        reg_k5(0, m.output.data, m.output.data, m.output.type, m.output.K, m.output.N);
    }
    if (n_k5 + n_i8 > 0) {
        if (dev_used_bytes_.size() < dev_bytes.size()) {
            dev_used_bytes_.resize(dev_bytes.size(), 0);
        }
        for (size_t d = 0; d < dev_bytes.size(); d++) {
            dev_used_bytes_[d] += dev_bytes[d];
        }
    }
    ds41x_.dense_store = (n_k5 + n_i8) > 0;
    if (ds41x_.dense_store) {
        fprintf(stderr,
                "[ds41] dense stores: %zu k5 + %zu int8 oneDNN tensors, %.0f MB%s "
                "(PF_DS41_DENSE_STORE=0 restores the raw GEMV)\n",
                n_k5, n_i8, (double)bytes / 1048576.0, add_q8 ? "" : " (Q8_0 kept raw)");
    } else {
        dnnl_dev_.clear();
    }
    return ds41x_.dense_store;
}

} // namespace si
