#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <sycl/sycl.hpp> // IWYU pragma: keep

#include "gguf.h"
#include "w8.h"

namespace si {

struct wt { // weight tensor view
    const void * data = nullptr;
    uint32_t type = 0;
    int32_t K = 0; // input dim (dims[0])
    int32_t N = 0; // output rows (product of dims[1:])
};

struct hparams {
    int n_layer = 0, n_embd = 0, n_ff = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    int n_vocab = 0;
    // `<arch>.context_length`: the window the model was *trained* on.  YaRN's
    // factor is target/trained, so it needs this (0 when the key is absent).
    int n_ctx = 0;
    float rope_base = 0.f, rms_eps = 0.f, attn_scale = 0.f;
    int d_state = 0, n_group = 0, dt_rank = 0, d_inner = 0, conv_k = 0;
    int full_attn_interval = 4;
    // number of MTP (NextN) layers shipped after the main blocks
    // (`qwen35.nextn_predict_layers`); their weights live in blk.<n_layer>.*
    int n_mtp = 0;
    // M-RoPE pair counts per section (t, h, w, e); all-zero = plain RoPE
    int rope_sections[4] = {0, 0, 0, 0};

    bool is_recr(int il) const {
        return (il + 1) % full_attn_interval != 0;
    }

    // GDN qkv projection/output width: q and k each carry `n_group` heads of
    // `d_state`, v carries `dt_rank` heads of `d_state` (= d_inner).  The
    // reference 0.8B has n_group == dt_rank, which is where the historical
    // 3*d_inner shortcut came from; larger models (Qwen3.8-27B: 16 vs 48)
    // separate the key and value head counts.
    int qkv_dim() const {
        return 2 * n_group * d_state + dt_rank * d_state;
    }
};

struct layer_t {
    bool recurrent = false;
    // One slot of the canonical per-layer weight enumeration: the weight view
    // and its SIn int8 copy (when the caller tracks them).
    struct wl {
        const wt * w;
        w8t * w8; // null when the caller does not track w8t pairs
    };
    // Fill out[] (must hold at least 7 entries) with this layer's quantized
    // weight slots in canonical order: FFN trio, then the GDN block
    // (wqkv/wgate/ssm_out) or the attention block (wq/wk/wv/wo), based on
    // `recurrent`.  ssm_beta/ssm_alpha are enumerated separately by the md
    // paths that convert them and have no SIn copy.  Returns the count.
    int lay_wts(wl out[], bool with_w8t = true) {
        int n = 0;
        out[n++] = {&ffn_gate, with_w8t ? &ffn_gate8 : nullptr};
        out[n++] = {&ffn_up, with_w8t ? &ffn_up8 : nullptr};
        out[n++] = {&ffn_down, with_w8t ? &ffn_down8 : nullptr};
        if (recurrent) {
            out[n++] = {&wqkv, with_w8t ? &wqkv8 : nullptr};
            out[n++] = {&wgate, with_w8t ? &wgate8 : nullptr};
            out[n++] = {&ssm_out, with_w8t ? &ssm_out8 : nullptr};
        } else {
            out[n++] = {&wq, with_w8t ? &wq8 : nullptr};
            out[n++] = {&wk, with_w8t ? &wk8 : nullptr};
            out[n++] = {&wv, with_w8t ? &wv8 : nullptr};
            out[n++] = {&wo, with_w8t ? &wo8 : nullptr};
        }
        return n;
    }
    const float * attn_norm = nullptr;
    const float * post_attn_norm = nullptr;
    wt ffn_gate, ffn_up, ffn_down;
    // attention layers
    wt wq, wk, wv, wo;
    const float * q_norm = nullptr;
    const float * k_norm = nullptr;
    // gdn layers
    wt wqkv, wgate, ssm_beta, ssm_alpha, ssm_out;
    const float * ssm_a = nullptr;
    const float * ssm_dt = nullptr;
    const float * ssm_norm = nullptr;
    const float * ssm_conv1d = nullptr;
    // SI8 (int8 + DP4A) copies of the quantized weights
    w8t ffn_gate8, ffn_up8, ffn_down8;
    w8t wq8, wk8, wv8, wo8;
    w8t wqkv8, wgate8, ssm_out8;
};

// Multi-token prediction (NextN) layer: one full-attention decoder block whose
// input is `eh_proj(concat(enorm(emb(t_p)), hnorm(h_{p-1})))` instead of the
// previous layer's output, followed by a shared output norm + LM head.  Used as
// a speculative-draft head: its logits at MTP row p predict token p+1 while the
// main model's hidden at p-1 conditions it.
struct mtp_layer_t {
    const float * attn_norm = nullptr;
    const float * post_attn_norm = nullptr;
    wt wq, wk, wv, wo;
    const float * q_norm = nullptr;
    const float * k_norm = nullptr;
    wt ffn_gate, ffn_up, ffn_down;
    // NextN extras
    wt eh_proj;                  // [2*n_embd][K=2*n_embd] -> n_embd
    const float * enorm = nullptr;
    const float * hnorm = nullptr;
    const float * shared_head_norm = nullptr;
    wt shared_head;              // optional; empty -> model::output
    // per-device SIn int8 copies (GPU decode path)
    w8t wq8, wk8, wv8, wo8, ffn_gate8, ffn_up8, ffn_down8, eh_proj8;
};

// Reject a model whose geometry the kernels do not implement, before anything
// is uploaded or allocated.  Several kernels hardcode a head_dim (the generic
// attention partial stride is literally 256) and would otherwise overrun their
// buffers rather than merely lose accuracy.  Throws std::runtime_error naming
// the first violated constraint.  Called by model::load for every architecture,
// so a new loader inherits it; exposed so a test can drive it without a GGUF.
void validate_hparams(const hparams & hp, const std::string & arch);

struct model {
    gguf_file gguf;
    hparams hp;
    // tokenizer.chat_template (Jinja); empty when the GGUF does not carry one
    std::string chat_template;
    std::vector<layer_t> layers;
    std::vector<int> gdn_layer_index; // layer id -> sequential index among GDN layers (-1)
    wt tok_embd;
    // LM-head weight: `output.weight` when the GGUF is untied, else tok_embd
    wt output;
    const float * output_norm = nullptr;
    size_t tok_embd_row_bytes = 0;
    w8t output8;

    // MTP (NextN) draft layer; has_mtp is false when the GGUF ships none
    mtp_layer_t mtp;
    bool has_mtp = false;

    // device copy of all weights (one blob)
    void * dev_weights = nullptr;
    size_t dev_weights_size = 0;

    void load(const std::string & path);
    // upload the whole mmap to device USM; `host` keeps the weights in the
    // mmap and only makes dev_ptr() an identity (CPU backend)
    void upload(sycl::queue & q, bool host = false);
    // build / free the SI8 int8 copies of the quantized weight tensors (DP4A path)
    void build_w8(sycl::queue & q, bool host = false);
    // multi-device: build the SIn copy of a single tensor on `q`'s device
    // (a non-K-quant tensor yields an empty w8t)
    void build_w8_one(sycl::queue & q, const wt & t, w8t & out);
    void free_w8(sycl::queue & q);
    // Drop this process's resident pages backing [p, p+len) of the GGUF mmap.
    // The MAP_PRIVATE mapping stays valid, so a later host read simply re-faults
    // from the file; used once a tensor has been copied to a device so the
    // process RSS does not keep the whole (16 GB for the 27B) file resident.
    void page_out_host(const void * p, size_t len);
    // byte length of a bound tensor (row bytes * rows)
    static size_t tensor_bytes(const wt & t) {
        return ggml_row_bytes(t.type, (uint64_t)t.K * (uint64_t)t.N);
    }
    void page_out_tensor(const wt & t) {
        page_out_host(t.data, tensor_bytes(t));
    }
    const void * dev_ptr(const void * host_ptr) const {
        if (!dev_weights) {
            return host_ptr; // CPU backend: weights stay in the mmap
        }
        return (const char *)dev_weights + ((const char *)host_ptr - (const char *)gguf.map_base);
    }
    const float * dev_f32(const float * p) const {
        return (const float *)dev_ptr(p);
    }
};

// Maximum context length advertised by the model's GGUF metadata
// (`<general.architecture>.context_length`); 0 when the key is absent.  Reads
// only metadata, so it can size the engine before the model is constructed.
int model_context_length(const std::string & path);

} // namespace si
