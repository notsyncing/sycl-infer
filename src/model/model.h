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
    float rope_base = 0.f, rms_eps = 0.f, attn_scale = 0.f;
    int d_state = 0, n_group = 0, dt_rank = 0, d_inner = 0, conv_k = 0;
    int full_attn_interval = 4;
    // M-RoPE pair counts per section (t, h, w, e); all-zero = plain RoPE
    int rope_sections[4] = {0, 0, 0, 0};

    bool is_recr(int il) const {
        return (il + 1) % full_attn_interval != 0;
    }
};

struct layer_t {
    bool recurrent = false;
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

struct model {
    gguf_file gguf;
    hparams hp;
    // tokenizer.chat_template (Jinja); empty when the GGUF does not carry one
    std::string chat_template;
    std::vector<layer_t> layers;
    std::vector<int> gdn_layer_index; // layer id -> sequential index among GDN layers (-1)
    wt tok_embd;
    const float * output_norm = nullptr;
    size_t tok_embd_row_bytes = 0;
    w8t tok_embd8;

    // device copy of all weights (one blob)
    void * dev_weights = nullptr;
    size_t dev_weights_size = 0;

    void load(const std::string & path);
    // upload the whole mmap to device USM; `host` keeps the weights in the
    // mmap and only makes dev_ptr() an identity (CPU backend)
    void upload(sycl::queue & q, bool host = false);
    // build / free the SI8 int8 copies of the quantized weight tensors (DP4A path)
    void build_w8(sycl::queue & q, bool host = false);
    void free_w8(sycl::queue & q);
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
