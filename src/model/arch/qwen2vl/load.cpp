// Qwen2-VL text trunk (MinerU2.5-Pro-2605-1.2B is a Qwen2-VL decoder).
//
// This is a plain dense decoder: every layer is full grouped-query attention
// followed by a SwiGLU FFN, with RMSNorm before each.  It has none of the
// Qwen3.5 machinery (no GDN recurrence, no attention output gate, no QK norm,
// no NextN block), and its rows are not K-quant multiples (n_embd 896), so the
// file uses Q5_0/Q8_0 for those tensors and the loader only has to bind the
// ordinary per-layer weights.  The multimodal tower is a separate mmproj and
// lives in src/mm/.
#include <cmath>
#include <string>

#include "arch_util.h"
#include "load.h"
#include "model.h"

namespace si {
namespace arch {

void load_qwen2vl(model & m) {
    const gguf_file & f = m.gguf;
    const std::string arch = "qwen2vl";
    auto key = [&](const char * k) { return arch + "." + k; };

    auto & hp = m.hp;
    hp.n_layer = (int)f.get_u32(key("block_count"));
    hp.n_embd = (int)f.get_u32(key("embedding_length"));
    hp.n_ff = (int)f.get_u32(key("feed_forward_length"));
    hp.n_head = (int)f.get_u32(key("attention.head_count"));
    hp.n_head_kv = (int)f.get_u32(key("attention.head_count_kv"));
    // Qwen2-VL omits key_length: head_dim is exactly n_embd/n_head.
    hp.head_dim = (int)f.get_u32(key("attention.key_length"), (uint32_t)(hp.n_embd / hp.n_head));
    hp.n_rot = (int)f.get_u32(key("rope.dimension_count"), (uint32_t)hp.head_dim);
    hp.n_ctx = (int)f.get_u32(key("context_length"), 0);
    hp.rope_base = f.get_f32(key("rope.freq_base"), 1000000.f);
    hp.rms_eps = f.get_f32(key("attention.layer_norm_rms_epsilon"), 1e-6f);
    // every layer is attention; is_recr() divides by this
    hp.full_attn_interval = 1;
    hp.attn_scale = 1.0f / std::sqrt((float)hp.head_dim);
    hp.attn_gate = false; // no q+gate fused projection and no sigmoid gate
    if (const gguf_kv * sec = f.meta(key("rope.dimension_sections"))) {
        for (int i = 0; i < 4 && i < (int)sec->arr.size(); i++) {
            hp.rope_sections[i] = sec->arr[i].as_i32();
        }
    }

    fprintf(stderr, "[qwen2vl] n_embd=%d n_ff=%d n_head=%d n_head_kv=%d head_dim=%d n_rot=%d sections=[%d,%d,%d,%d]\n",
            hp.n_embd, hp.n_ff, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_rot, hp.rope_sections[0],
            hp.rope_sections[1], hp.rope_sections[2], hp.rope_sections[3]);
    const auto * toks = f.meta("tokenizer.ggml.tokens");
    hp.n_vocab = toks ? (int)toks->arr.size() : 0;

    m.tok_embd = bind_tensor(f, "token_embd.weight");
    if (f.find("output.weight")) {
        m.output = bind_tensor(f, "output.weight");
    } else {
        m.output = m.tok_embd; // tied embeddings (the MinerU file is tied)
    }
    m.output_norm = bind_f32(f, "output_norm.weight");
    m.tok_embd_row_bytes = ggml_row_bytes(m.tok_embd.type, hp.n_embd);

    m.layers.resize(hp.n_layer);
    m.gdn_layer_index.assign(hp.n_layer, -1);
    for (int il = 0; il < hp.n_layer; il++) {
        layer_t & L = m.layers[il];
        const std::string pre = "blk." + std::to_string(il) + ".";
        L.recurrent = false;
        L.attn_norm = bind_f32(f, pre + "attn_norm.weight");
        // Qwen2 names the second RMSNorm `ffn_norm`; the engine's generic plan
        // calls its slot post_attn_norm.
        L.post_attn_norm = bind_f32(f, pre + "ffn_norm.weight");
        L.ffn_gate = bind_tensor(f, pre + "ffn_gate.weight");
        L.ffn_up = bind_tensor(f, pre + "ffn_up.weight");
        L.ffn_down = bind_tensor(f, pre + "ffn_down.weight");
        L.wq = bind_tensor(f, pre + "attn_q.weight");
        L.wk = bind_tensor(f, pre + "attn_k.weight");
        L.wv = bind_tensor(f, pre + "attn_v.weight");
        L.wo = bind_tensor(f, pre + "attn_output.weight");
        // Qwen2 carries q/k/v projection biases and no QK RMSNorm.
        L.q_bias = f.find(pre + "attn_q.bias") ? bind_f32(f, pre + "attn_q.bias") : nullptr;
        L.k_bias = f.find(pre + "attn_k.bias") ? bind_f32(f, pre + "attn_k.bias") : nullptr;
        L.v_bias = f.find(pre + "attn_v.bias") ? bind_f32(f, pre + "attn_v.bias") : nullptr;
        L.q_norm = nullptr;
        L.k_norm = nullptr;
    }
}

} // namespace arch
} // namespace si
