// Qwen3.8-Flash-Next ("qwen4exp"): the Qwen3.5 hybrid GDN/full-attention trunk
// with three additions:
//
//  * hyper-connections: there is no per-layer residual norm.  The residual is
//    `hc` parallel streams [n_embd, hc]; each block mixes them through a
//    grouped RMSNorm, a low-rank gate and a per-stream scatter projection.
//  * PLE: one layer gathers 16 n-gram-hash rows from a huge shared embedding
//    table and adds a gated, depthwise-convolved value into the residual.
//  * QSA: the full-attention layers keep a 128-wide indexer key per token and
//    attend only to the top `indexer_top_k` keys, pooled in blocks of
//    `compress_ratio`, plus the incomplete tail.
//
// The token mixer and the MoE FFN are the same shapes as qwen35moe's; only the
// gated normalization of the GDN differs (sigmoid, not silu).
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "model_arch.h"

namespace si {
namespace arch {

namespace {

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
    t.plane_bytes = ggml_row_bytes(ti->type, ti->dims[0] * ti->dims[1]);
    return t;
}

const gguf_kv * need_meta(const gguf_file & f, const std::string & key) {
    const gguf_kv * v = f.meta(key);
    if (!v) {
        throw std::runtime_error("missing metadata: " + key);
    }
    return v;
}

} // namespace

void load_qwen4exp(model & m) {
    const gguf_file & f = m.gguf;
    const std::string arch = "qwen4exp";
    auto key = [&](const char * k) { return arch + "." + k; };

    hparams & hp = m.hp;
    hp.q4exp = true;
    const uint32_t block_count = f.get_u32(key("block_count"));
    hp.n_layer = (int)block_count; // no NextN block in the trunk file
    hp.n_mtp = 0;
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
    hp.gdn_sigmoid_gate = true;
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
    hp.n_ff = hp.n_ff_shexp ? hp.n_ff_shexp : (int)f.get_u32(key("feed_forward_length"), 0);

    hp.hc_mult = (int)f.get_u32(key("hyper_connection.count"));
    hp.hc_low_rank = (int)f.get_u32(key("hyper_connection.low_rank"));
    hp.idx_n_head = (int)f.get_u32(key("attention.indexer.head_count"));
    hp.idx_head_dim = (int)f.get_u32(key("attention.indexer.key_length"));
    hp.idx_top_k = (int)f.get_u32(key("attention.indexer.top_k"));
    // The QSA layers must share one compress ratio; the reference requires the
    // budget to be an exact multiple of it.
    hp.idx_kpool = 0;
    if (const gguf_kv * cr = f.meta(key("attention.compress_ratios"))) {
        for (const gguf_kv & e : cr->arr) {
            const uint32_t r = e.as_u32();
            if (r == 0) {
                continue;
            }
            if (hp.idx_kpool != 0 && (uint32_t)hp.idx_kpool != r) {
                throw std::runtime_error("qwen4exp: QSA layers use different compress ratios");
            }
            hp.idx_kpool = (int)r;
        }
    }
    if (hp.idx_kpool > 1 && hp.idx_top_k % hp.idx_kpool != 0) {
        throw std::runtime_error("qwen4exp: indexer top_k is not a multiple of the compress ratio");
    }

    hp.ple_layer = -1;
    if (const gguf_kv * pl = f.meta(key("ple.layers"))) {
        if (!pl->arr.empty()) {
            hp.ple_layer = pl->arr[0].as_i32();
        }
    }
    if (hp.ple_layer >= 0) {
        hp.ple_ngram = (int)f.get_u32(key("ple.ngram_size"));
        hp.ple_heads_per_ngram = (int)f.get_u32(key("ple.heads_per_ngram"));
        hp.ple_conv = (int)f.get_u32(key("ple.conv_kernel"));
        hp.ple_eos = (int)f.get_u32(key("ple.eos_token_id"));
        hp.ple_image = (int)f.get_u32(key("ple.image_token_id"), hp.ple_eos);
        hp.ple_head_dim = (int)f.get_u32(key("embedding_length_per_layer_input"));
        hp.ple_n_heads = (hp.ple_ngram - 1) * hp.ple_heads_per_ngram;
        if (hp.ple_n_heads <= 0 || hp.ple_n_heads > 64 || hp.ple_head_dim <= 0) {
            throw std::runtime_error("qwen4exp: unsupported PLE head geometry");
        }
        const gguf_kv * mult = need_meta(f, key("ple.layer_multipliers"));
        const gguf_kv * off = need_meta(f, key("ple.head_offsets"));
        const gguf_kv * voc = need_meta(f, key("ple.head_vocab_sizes"));
        if ((int)mult->arr.size() < hp.ple_ngram || (int)off->arr.size() < hp.ple_n_heads
            || (int)voc->arr.size() < hp.ple_n_heads) {
            throw std::runtime_error("qwen4exp: PLE hash constant arrays are too short");
        }
        for (int i = 0; i < hp.ple_ngram; i++) {
            hp.ple_mult[i] = (int64_t)mult->arr[(size_t)i].u64;
        }
        for (int i = 0; i < hp.ple_n_heads; i++) {
            hp.ple_off[i] = off->arr[(size_t)i].as_u32();
            hp.ple_vocab[i] = voc->arr[(size_t)i].as_u32();
        }
    }

    const auto * toks = f.meta("tokenizer.ggml.tokens");
    hp.n_vocab = toks ? (int)toks->arr.size() : 0;

    m.q4exp = true;
    m.tok_embd = bind_tensor(f, "token_embd.weight");
    if (f.find("output.weight")) {
        m.output = bind_tensor(f, "output.weight");
    } else {
        m.output = m.tok_embd;
    }
    m.output_norm = nullptr; // the final hyper-connection mixer carries the norm
    m.tok_embd_row_bytes = ggml_row_bytes(m.tok_embd.type, hp.n_embd);

    m.hc_head.norm = bind_f32(f, "output_hc_norm.weight");
    m.hc_head.down = bind_tensor(f, "output_hc_down.weight");
    m.hc_head.up = bind_tensor(f, "output_hc_up.weight");
    if (hp.ple_layer >= 0) {
        m.ple_table = bind_tensor(f, "per_layer_token_embd.weight");
    }

    m.layers.resize(hp.n_layer);
    m.gdn_layer_index.assign(hp.n_layer, -1);
    int gdn_count = 0;
    for (int il = 0; il < hp.n_layer; il++) {
        layer_t & L = m.layers[il];
        const std::string pre = "blk." + std::to_string(il) + ".";
        L.recurrent = hp.is_recr(il);
        L.hc_attn.norm = bind_f32(f, pre + "hc_attn_norm.weight");
        L.hc_attn.down = bind_tensor(f, pre + "hc_attn_down.weight");
        L.hc_attn.up = bind_tensor(f, pre + "hc_attn_up.weight");
        L.hc_attn.inject = bind_tensor(f, pre + "hc_attn_inject.weight");
        L.hc_ffn.norm = bind_f32(f, pre + "hc_ffn_norm.weight");
        L.hc_ffn.down = bind_tensor(f, pre + "hc_ffn_down.weight");
        L.hc_ffn.up = bind_tensor(f, pre + "hc_ffn_up.weight");
        L.hc_ffn.inject = bind_tensor(f, pre + "hc_ffn_inject.weight");

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
            L.idx.q_proj = bind_tensor(f, pre + "indexer.q_proj.weight");
            L.idx.k_proj = bind_tensor(f, pre + "indexer.k_proj.weight");
            L.idx.q_norm = bind_f32(f, pre + "indexer.q_norm.weight");
            L.idx.k_norm = bind_f32(f, pre + "indexer.k_norm.weight");
        }

        if (il == hp.ple_layer) {
            L.ple.key = bind_tensor(f, pre + "ple_key.weight");
            L.ple.value = bind_tensor(f, pre + "ple_value.weight");
            L.ple.norm_key = bind_f32(f, pre + "ple_norm_key.weight");
            L.ple.norm_query = bind_f32(f, pre + "ple_norm_query.weight");
            L.ple.norm_conv = bind_f32(f, pre + "ple_norm_conv.weight");
            L.ple.conv = bind_f32(f, pre + "ple_conv1d.weight");
        }

        // shared expert -> the ordinary dense FFN slots, as in qwen35moe
        L.ffn_gate = bind_tensor(f, pre + "ffn_gate_shexp.weight");
        L.ffn_up = bind_tensor(f, pre + "ffn_up_shexp.weight");
        L.ffn_down = bind_tensor(f, pre + "ffn_down_shexp.weight");
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
        if (!gs) {
            throw std::runtime_error("missing tensor: " + pre + "ffn_gate_inp_shexp.weight");
        }
        L.moe.gate_inp_shexp = gs->data;
        L.moe.gate_inp_shexp_type = gs->type;
    }
}

} // namespace arch
} // namespace si
