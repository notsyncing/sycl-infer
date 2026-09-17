// Qwen3.5 architecture: a hybrid stack of GDN (recurrent) and full-attention
// layers.  Every `full_attention_interval`-th layer is a grouped-query
// attention layer, the rest run the gated delta-net recurrence.
#include <cmath>
#include <string>
#include <vector>

#include "model_arch.h"

namespace si {
namespace arch {

void load_qwen35(model & m) {
    const gguf_file & f = m.gguf;
    const std::string arch = "qwen35";
    auto key = [&](const char * k) { return arch + "." + k; };

    auto & hp = m.hp;
    hp.n_layer = (int) f.get_u32(key("block_count"));
    hp.n_embd = (int) f.get_u32(key("embedding_length"));
    hp.n_ff = (int) f.get_u32(key("feed_forward_length"));
    hp.n_head = (int) f.get_u32(key("attention.head_count"));
    hp.n_head_kv = (int) f.get_u32(key("attention.head_count_kv"));
    hp.head_dim = (int) f.get_u32(key("attention.key_length"));
    hp.n_rot = (int) f.get_u32(key("rope.dimension_count"));
    hp.rope_base = f.get_f32(key("rope.freq_base"), 10000.f);
    hp.rms_eps = f.get_f32(key("attention.layer_norm_rms_epsilon"), 1e-6f);
    hp.d_state = (int) f.get_u32(key("ssm.state_size"));
    hp.n_group = (int) f.get_u32(key("ssm.group_count"));
    hp.dt_rank = (int) f.get_u32(key("ssm.time_step_rank"));
    hp.d_inner = (int) f.get_u32(key("ssm.inner_size"));
    hp.conv_k = (int) f.get_u32(key("ssm.conv_kernel"));
    hp.full_attn_interval = (int) f.get_u32(key("full_attention_interval"), 4);
    hp.attn_scale = 1.0f / std::sqrt((float) hp.head_dim);

    const auto * toks = f.meta("tokenizer.ggml.tokens");
    hp.n_vocab = toks ? (int) toks->arr.size() : 0;

    m.tok_embd = bind_tensor(f, "token_embd.weight");
    m.output_norm = bind_f32(f, "output_norm.weight");
    m.tok_embd_row_bytes = ggml_row_bytes(m.tok_embd.type, hp.n_embd);

    m.layers.resize(hp.n_layer);
    m.gdn_layer_index.assign(hp.n_layer, -1);
    int gdn_count = 0;
    for (int il = 0; il < hp.n_layer; il++) {
        layer_t & L = m.layers[il];
        const std::string pre = "blk." + std::to_string(il) + ".";
        L.recurrent = hp.is_recr(il);
        L.attn_norm = bind_f32(f, pre + "attn_norm.weight");
        L.post_attn_norm = bind_f32(f, pre + "post_attention_norm.weight");
        L.ffn_gate = bind_tensor(f, pre + "ffn_gate.weight");
        L.ffn_up = bind_tensor(f, pre + "ffn_up.weight");
        L.ffn_down = bind_tensor(f, pre + "ffn_down.weight");
        if (L.recurrent) {
            m.gdn_layer_index[il] = gdn_count++;
            L.wqkv = bind_tensor(f, pre + "attn_qkv.weight");
            L.wgate = bind_tensor(f, pre + "attn_gate.weight");
            L.ssm_beta = bind_tensor(f, pre + "ssm_beta.weight");
            L.ssm_alpha = bind_tensor(f, pre + "ssm_alpha.weight");
            L.ssm_out = bind_tensor(f, pre + "ssm_out.weight");
            L.ssm_a = bind_f32(f, pre + "ssm_a");
            L.ssm_dt = bind_f32(f, pre + "ssm_dt.bias");
            L.ssm_norm = bind_f32(f, pre + "ssm_norm.weight");
            L.ssm_conv1d = bind_f32(f, pre + "ssm_conv1d.weight");
        } else {
            L.wq = bind_tensor(f, pre + "attn_q.weight");
            L.wk = bind_tensor(f, pre + "attn_k.weight");
            L.wv = bind_tensor(f, pre + "attn_v.weight");
            L.wo = bind_tensor(f, pre + "attn_output.weight");
            L.q_norm = bind_f32(f, pre + "attn_q_norm.weight");
            L.k_norm = bind_f32(f, pre + "attn_k_norm.weight");
        }
    }
}

} // namespace arch
} // namespace si
