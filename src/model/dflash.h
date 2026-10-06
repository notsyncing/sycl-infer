// DFlash / DFlash2 draft model (llama.cpp architecture "dflash").
//
// A DFlash drafter is a small *block-diffusion* model trained against one
// specific target.  It is not a second language model: it has no embeddings and
// no LM head of its own (the GGUF ships none), it reads the target's hidden
// states at a handful of layers, and it turns them into a whole block of draft
// candidates in a single forward pass.  Reference: llama.cpp `src/models/dflash.cpp`
// + `common/speculative.cpp::common_speculative_impl_draft_dflash`.
//
//   features   the target's hidden state *input* at layers `dflash.target_layers`
//              (Qwen3.8-27B: [6, 20, 34, 48, 62]), concatenated -> [5*n_embd]
//   fc         projects them to the draft width, then `enc.output_norm`
//   injection  per draft layer, wk/wv on that projected state give the draft's
//              own K/V for every *committed* token, which is how the drafter sees
//              the conversation without running the target's layers
//   block      one forward over [anchor, MASK x (block_size-1)] with NON-causal
//              attention, emitting a candidate distribution per position
//   selector   (DFlash2) top-k per position plus pairwise transition scores; the
//              host walks one coherent path through that lattice
//
// The draft's vocabulary, embeddings and LM head are the *target's* (shared via
// llama.cpp's ctx_other), so this file only binds the draft's own tensors.
#pragma once

#include "model.h" // wt, hparams-free tensor view, bind_tensor/bind_f32

#include <string>
#include <unordered_map>
#include <vector>

namespace si {

struct dflash_hp {
    int n_layer = 0, n_embd = 0, n_ff = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    int n_vocab = 0; // the shared (target) vocabulary
    int n_ctx = 0;
    float rope_base = 0.f, rms_eps = 0.f;
    // dflash.block_size: the trained block width.  The draft input is
    // [anchor, MASK x (block_size-1)], so one cycle proposes block_size-1 tokens.
    int block_size = 0;
    // DFlash2 additions: the grouped dynamic depthwise convolution and the
    // candidate selector lattice.
    int conv_k = 0, conv_group = 0;
    int sel_rank = 0, sel_top_k = 0;
    int swa = 0; // attention.sliding_window (0 = attend to everything)
    int mask_id = 0;
    int n_tgt_layer = 0;              // dflash.target_layers.size()
    int tgt_layer[16] = {0};          // ... and the ids
    int n_feat = 0;                   // n_tgt_layer * target n_embd (fc's K)
    bool is_dflash2 = false;          // the selector tensors are present
    int conv_proj = 0;                // 2 * conv_k * (n_embd / conv_group)
    int n_groups = 0;                 // n_embd / conv_group
};

struct dflash_layer_t {
    const float * attn_norm = nullptr;
    const float * q_norm = nullptr;
    const float * k_norm = nullptr;
    const float * ffn_norm = nullptr;
    // DFlash2: static kernel + dynamic (input-projected) delta, [n_embd][conv_k][2]
    // (side 0 = the sublayer's input, side 1 = its output)
    const float * attn_conv_base = nullptr;
    const float * ffn_conv_base = nullptr;
    // Attention sinks, [n_head]: an extra key per query head whose score is this
    // logit and whose value is 0, so it only inflates the softmax denominator.
    // llama.cpp passes it to build_attn (src/models/dflash.cpp:799) and ggml
    // applies it in ggml_soft_max_add_sinks; omitting it scales every head's
    // output up by (z + exp(sink - max)) / z.
    const float * attn_sinks = nullptr;
    wt attn_conv_proj, ffn_conv_proj;
    wt wq, wk, wv, wo;
    wt ffn_gate, ffn_up, ffn_down;
};

struct dflash_model {
    gguf_file gguf;
    dflash_hp hp;
    wt fc;                  // [n_feat, n_embd]
    const float * enc_norm = nullptr;  // after fc
    const float * output_norm = nullptr; // final decoder norm
    wt sel_hidden;          // [n_embd, rank]
    wt sel_prev;            // [rank, n_vocab]
    wt sel_next;            // [rank, n_vocab]
    std::vector<dflash_layer_t> layers;

    // device USM pointers for the *unquantized* tensors (norms, conv bases).
    // Every quantized tensor lives on the device through its native/int8 store
    // (registered in the device's dnnl_gemm table, keyed by the host pointer),
    // so nothing else needs a raw device copy.
    std::unordered_map<const void *, void *> dev;
    void * dev_reserved = nullptr;

    void load(const std::string & path, int target_n_embd, int target_n_vocab);
    // upload the f32 tensors to `q`'s device
    void upload_f32(sycl::queue & q);
    const void * dev_ptr(const void * host) const {
        auto it = dev.find(host);
        return it == dev.end() ? host : it->second;
    }
    const float * dev_f32(const float * p) const {
        return (const float *)dev_ptr(p);
    }
    size_t bytes() const;
};

} // namespace si
