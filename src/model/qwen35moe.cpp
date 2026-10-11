// Qwen3.6 "35B-A3B" architecture: the Qwen3.5 hybrid GDN/attention trunk with a
// Mixture-of-Experts FFN in every layer, plus a NextN (MTP) block whose FFN is
// MoE as well.
//
// The FFN is the sum of two independent branches over the *same* post-attention
// normed hidden state:
//   routed:   gate_inp -> softmax over n_expert -> top-k -> sum(w * FFN_e(x))
//   shared:   FFN_shexp(x) * sigmoid(gate_inp_shexp . x)
// with the residual added once afterwards.  `ffn_gate/up/down_shexp` is an
// ordinary dense FFN of width n_ff_shexp, so it is bound into the layer's
// existing `ffn_gate/up/down` slots and reuses every dense path unchanged; the
// routed half lives in `layer_t::moe` and is the only genuinely new work.
#include <cmath>
#include <string>

#include "model_arch.h"

namespace si {
namespace arch {

namespace {

// A 3-D routed-expert tensor [K][N][E].  `wt` cannot express it: its N is the
// product of dims[1:], which for an expert tensor is N*E, and the stride
// between experts (nb[2]) is not derivable from the element count once the
// blocks are packed.
moe_wt bind_moe(const gguf_file & f, const std::string & name) {
    const gguf_tensor_info * ti = f.find(name);
    if (!ti) {
        throw std::runtime_error("missing tensor: " + name);
    }
    if (ti->dims.size() != 3) {
        throw std::runtime_error("expected a 3-D expert tensor: " + name);
    }
    moe_wt t;
    t.data = ti->data;
    t.type = ti->type;
    t.K = (int32_t)ti->dims[0];
    t.N = (int32_t)ti->dims[1];
    t.E = (int32_t)ti->dims[2];
    // ggml stores the tensors row-contiguous (nb[0] = type size, nb[1] =
    // row_size(ne[0]), nb[2] = row_size(ne[0])*ne[1]), so the plane stride is
    // the size of one [K][N] plane.
    t.plane_bytes = ggml_row_bytes(ti->type, ti->dims[0] * ti->dims[1]);
    return t;
}

} // namespace

void load_qwen35moe(model & m) {
    const gguf_file & f = m.gguf;
    const std::string arch = "qwen35moe";
    auto key = [&](const char * k) { return arch + "." + k; };

    auto & hp = m.hp;
    const uint32_t block_count = f.get_u32(key("block_count"));
    const uint32_t n_layer_nextn = f.get_u32(key("nextn_predict_layers"), 0);
    hp.n_layer = (int)(block_count - n_layer_nextn);
    hp.n_mtp = (int)n_layer_nextn;
    hp.n_embd = (int)f.get_u32(key("embedding_length"));
    hp.n_head = (int)f.get_u32(key("attention.head_count"));
    hp.n_head_kv = (int)f.get_u32(key("attention.head_count_kv"));
    hp.head_dim = (int)f.get_u32(key("attention.key_length"));
    hp.n_rot = (int)f.get_u32(key("rope.dimension_count"));
    hp.n_ctx = (int)f.get_u32(key("context_length"), 0);
    hp.rope_base = f.get_f32(key("rope.freq_base"), 10000.f);
    hp.rms_eps = f.get_f32(key("attention.layer_norm_rms_epsilon"), 1e-6f);
    hp.d_state = (int)f.get_u32(key("ssm.state_size"));
    hp.n_group = (int)f.get_u32(key("ssm.group_count"));
    hp.dt_rank = (int)f.get_u32(key("ssm.time_step_rank"));
    hp.d_inner = (int)f.get_u32(key("ssm.inner_size"));
    hp.conv_k = (int)f.get_u32(key("ssm.conv_kernel"));
    hp.full_attn_interval = (int)f.get_u32(key("full_attention_interval"), 4);
    hp.attn_scale = 1.0f / std::sqrt((float)hp.head_dim);
    if (const gguf_kv * sec = f.meta(key("rope.dimension_sections"))) {
        for (int i = 0; i < 4 && i < (int)sec->arr.size(); i++) {
            hp.rope_sections[i] = sec->arr[i].as_i32();
        }
    }

    hp.n_expert = (int)f.get_u32(key("expert_count"));
    hp.n_expert_used = (int)f.get_u32(key("expert_used_count"));
    hp.n_ff_exp = (int)f.get_u32(key("expert_feed_forward_length"));
    hp.n_ff_shexp = (int)f.get_u32(key("expert_shared_feed_forward_length"));
    hp.expert_weights_scale = f.get_f32(key("expert_weights_scale"), 1.0f);
    // The shared expert is the dense half of the FFN, so `n_ff` - the width
    // every dense GEMV call is sized from - is its width.  A model that has no
    // shared expert keeps the ordinary dense `feed_forward_length`.
    hp.n_ff = hp.n_ff_shexp ? hp.n_ff_shexp : (int)f.get_u32(key("feed_forward_length"), 0);

    const auto * toks = f.meta("tokenizer.ggml.tokens");
    hp.n_vocab = toks ? (int)toks->arr.size() : 0;

    m.tok_embd = bind_tensor(f, "token_embd.weight");
    if (f.find("output.weight")) {
        m.output = bind_tensor(f, "output.weight");
    } else {
        m.output = m.tok_embd; // tied embeddings
    }
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
        // shared expert -> the layer's ordinary dense FFN slots
        L.ffn_gate = bind_tensor(f, pre + "ffn_gate_shexp.weight");
        L.ffn_up = bind_tensor(f, pre + "ffn_up_shexp.weight");
        L.ffn_down = bind_tensor(f, pre + "ffn_down_shexp.weight");
        // routed experts
        L.moe.gate_exps = bind_moe(f, pre + "ffn_gate_exps.weight");
        L.moe.up_exps = bind_moe(f, pre + "ffn_up_exps.weight");
        L.moe.down_exps = bind_moe(f, pre + "ffn_down_exps.weight");
        const gguf_tensor_info * gi = f.find(pre + "ffn_gate_inp.weight");
        if (!gi) {
            throw std::runtime_error("missing tensor: " + pre + "ffn_gate_inp.weight");
        }
        L.moe.gate_inp = gi->data;
        L.moe.gate_inp_type = gi->type;
        const gguf_tensor_info * gs = f.find(pre + "ffn_gate_inp_shexp.weight");
        if (gs) {
            L.moe.gate_inp_shexp = gs->data;
            L.moe.gate_inp_shexp_type = gs->type;
        }
    }

    if (hp.n_mtp > 0) {
        const std::string pre = "blk." + std::to_string(hp.n_layer) + ".";
        mtp_layer_t & M = m.mtp;
        M.attn_norm = bind_f32(f, pre + "attn_norm.weight");
        M.post_attn_norm = bind_f32(f, pre + "post_attention_norm.weight");
        M.wq = bind_tensor(f, pre + "attn_q.weight");
        M.wk = bind_tensor(f, pre + "attn_k.weight");
        M.wv = bind_tensor(f, pre + "attn_v.weight");
        M.wo = bind_tensor(f, pre + "attn_output.weight");
        M.q_norm = bind_f32(f, pre + "attn_q_norm.weight");
        M.k_norm = bind_f32(f, pre + "attn_k_norm.weight");
        // the NextN block's FFN is MoE too, same split as a trunk layer
        M.ffn_gate = bind_tensor(f, pre + "ffn_gate_shexp.weight");
        M.ffn_up = bind_tensor(f, pre + "ffn_up_shexp.weight");
        M.ffn_down = bind_tensor(f, pre + "ffn_down_shexp.weight");
        M.moe.gate_exps = bind_moe(f, pre + "ffn_gate_exps.weight");
        M.moe.up_exps = bind_moe(f, pre + "ffn_up_exps.weight");
        M.moe.down_exps = bind_moe(f, pre + "ffn_down_exps.weight");
        const gguf_tensor_info * gi = f.find(pre + "ffn_gate_inp.weight");
        if (gi) {
            M.moe.gate_inp = gi->data;
            M.moe.gate_inp_type = gi->type;
        }
        const gguf_tensor_info * gs = f.find(pre + "ffn_gate_inp_shexp.weight");
        if (gs) {
            M.moe.gate_inp_shexp = gs->data;
            M.moe.gate_inp_shexp_type = gs->type;
        }
        M.eh_proj = bind_tensor(f, pre + "nextn.eh_proj.weight");
        M.enorm = bind_f32(f, pre + "nextn.enorm.weight");
        M.hnorm = bind_f32(f, pre + "nextn.hnorm.weight");
        if (f.find(pre + "nextn.shared_head_norm.weight")) {
            M.shared_head_norm = bind_f32(f, pre + "nextn.shared_head_norm.weight");
        }
        if (f.find(pre + "nextn.shared_head_head.weight")) {
            M.shared_head = bind_tensor(f, pre + "nextn.shared_head_head.weight");
        }
        m.has_mtp = true;
    }
}

} // namespace arch
} // namespace si