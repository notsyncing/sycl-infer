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
#include <memory>
#include <stdexcept>
#include <string>

#include "arch_util.h"
#include "arch/qwen4exp/accessors.h"
#include "arch/qwen4exp/hparams.h"
#include "load.h"
#include "model.h"

namespace si {
namespace arch {

void load_qwen4exp(model & m) {
    const gguf_file & f = m.gguf;
    const std::string arch = "qwen4exp";
    auto key = [&](const char * k) { return arch + "." + k; };

    hparams & hp = m.hp;
    auto qh_p = std::make_shared<qwen4exp_hparams>();
    qwen4exp_hparams & qh = *qh_p;
    hp.arch = qh_p;
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
    qh.gdn_sigmoid_gate = true;
    if (const gguf_kv * sec = f.meta(key("rope.dimension_sections"))) {
        for (int i = 0; i < 4 && i < (int)sec->arr.size(); i++) {
            hp.rope_sections[i] = sec->arr[i].as_i32();
        }
    }
    // Qwen3.5/3.6's text model is Qwen3-VL-style: interleaved M-RoPE.
    hp.mrope_interleaved = 1;

    hp.n_expert = (int)f.get_u32(key("expert_count"));
    hp.n_expert_used = (int)f.get_u32(key("expert_used_count"));
    hp.n_ff_exp = (int)f.get_u32(key("expert_feed_forward_length"));
    hp.n_ff_shexp = (int)f.get_u32(key("expert_shared_feed_forward_length"));
    hp.expert_weights_scale = f.get_f32(key("expert_weights_scale"), 1.0f);
    hp.n_ff = hp.n_ff_shexp ? hp.n_ff_shexp : (int)f.get_u32(key("feed_forward_length"), 0);

    qh.hc_mult = (int)f.get_u32(key("hyper_connection.count"));
    qh.hc_low_rank = (int)f.get_u32(key("hyper_connection.low_rank"));
    qh.idx_n_head = (int)f.get_u32(key("attention.indexer.head_count"));
    qh.idx_head_dim = (int)f.get_u32(key("attention.indexer.key_length"));
    qh.idx_top_k = (int)f.get_u32(key("attention.indexer.top_k"));
    // The QSA layers must share one compress ratio; the reference requires the
    // budget to be an exact multiple of it.
    qh.idx_kpool = 0;
    if (const gguf_kv * cr = f.meta(key("attention.compress_ratios"))) {
        for (const gguf_kv & e : cr->arr) {
            const uint32_t r = e.as_u32();
            if (r == 0) {
                continue;
            }
            if (qh.idx_kpool != 0 && (uint32_t)qh.idx_kpool != r) {
                throw std::runtime_error("qwen4exp: QSA layers use different compress ratios");
            }
            qh.idx_kpool = (int)r;
        }
    }
    if (qh.idx_kpool > 1 && qh.idx_top_k % qh.idx_kpool != 0) {
        throw std::runtime_error("qwen4exp: indexer top_k is not a multiple of the compress ratio");
    }

    qh.ple_layer = -1;
    if (const gguf_kv * pl = f.meta(key("ple.layers"))) {
        if (!pl->arr.empty()) {
            qh.ple_layer = pl->arr[0].as_i32();
        }
    }
    if (qh.ple_layer >= 0) {
        qh.ple_ngram = (int)f.get_u32(key("ple.ngram_size"));
        qh.ple_heads_per_ngram = (int)f.get_u32(key("ple.heads_per_ngram"));
        qh.ple_conv = (int)f.get_u32(key("ple.conv_kernel"));
        qh.ple_eos = (int)f.get_u32(key("ple.eos_token_id"));
        qh.ple_image = (int)f.get_u32(key("ple.image_token_id"), qh.ple_eos);
        qh.ple_head_dim = (int)f.get_u32(key("embedding_length_per_layer_input"));
        qh.ple_n_heads = (qh.ple_ngram - 1) * qh.ple_heads_per_ngram;
        if (qh.ple_n_heads <= 0 || qh.ple_n_heads > 64 || qh.ple_head_dim <= 0) {
            throw std::runtime_error("qwen4exp: unsupported PLE head geometry");
        }
        const gguf_kv * mult = require_meta(f, key("ple.layer_multipliers"));
        const gguf_kv * off = require_meta(f, key("ple.head_offsets"));
        const gguf_kv * voc = require_meta(f, key("ple.head_vocab_sizes"));
        if ((int)mult->arr.size() < qh.ple_ngram || (int)off->arr.size() < qh.ple_n_heads
            || (int)voc->arr.size() < qh.ple_n_heads) {
            throw std::runtime_error("qwen4exp: PLE hash constant arrays are too short");
        }
        for (int i = 0; i < qh.ple_ngram; i++) {
            qh.ple_mult[i] = (int64_t)mult->arr[(size_t)i].u64;
        }
        for (int i = 0; i < qh.ple_n_heads; i++) {
            qh.ple_off[i] = off->arr[(size_t)i].as_u32();
            qh.ple_vocab[i] = voc->arr[(size_t)i].as_u32();
        }
    }

    const auto * toks = f.meta("tokenizer.ggml.tokens");
    hp.n_vocab = toks ? (int)toks->arr.size() : 0;

    m.tok_embd = bind_tensor(f, "token_embd.weight");
    if (f.find("output.weight")) {
        m.output = bind_tensor(f, "output.weight");
    } else {
        m.output = m.tok_embd;
    }
    m.output_norm = nullptr; // the final hyper-connection mixer carries the norm
    m.tok_embd_row_bytes = ggml_row_bytes(m.tok_embd.type, hp.n_embd);

    m.extras = std::make_unique<qwen4exp_model_extras>();
    qwen4exp_model_extras & MX = q4_model(m);
    MX.hc_head.norm = bind_f32(f, "output_hc_norm.weight");
    MX.hc_head.down = bind_tensor(f, "output_hc_down.weight");
    MX.hc_head.up = bind_tensor(f, "output_hc_up.weight");
    if (qh.ple_layer >= 0) {
        MX.ple_table = bind_tensor(f, "per_layer_token_embd.weight");
    }

    m.layers.resize(hp.n_layer);
    m.gdn_layer_index.assign(hp.n_layer, -1);
    int gdn_count = 0;
    for (int il = 0; il < hp.n_layer; il++) {
        layer_t & L = m.layers[il];
        const std::string pre = "blk." + std::to_string(il) + ".";
        L.recurrent = hp.is_recr(il);
        L.extras = std::make_unique<qwen4exp_layer>();
        qwen4exp_layer & X = q4_layer(L);
        X.hc_attn.norm = bind_f32(f, pre + "hc_attn_norm.weight");
        X.hc_attn.down = bind_tensor(f, pre + "hc_attn_down.weight");
        X.hc_attn.up = bind_tensor(f, pre + "hc_attn_up.weight");
        X.hc_attn.inject = bind_tensor(f, pre + "hc_attn_inject.weight");
        X.hc_ffn.norm = bind_f32(f, pre + "hc_ffn_norm.weight");
        X.hc_ffn.down = bind_tensor(f, pre + "hc_ffn_down.weight");
        X.hc_ffn.up = bind_tensor(f, pre + "hc_ffn_up.weight");
        X.hc_ffn.inject = bind_tensor(f, pre + "hc_ffn_inject.weight");

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
            X.idx.q_proj = bind_tensor(f, pre + "indexer.q_proj.weight");
            X.idx.k_proj = bind_tensor(f, pre + "indexer.k_proj.weight");
            X.idx.q_norm = bind_f32(f, pre + "indexer.q_norm.weight");
            X.idx.k_norm = bind_f32(f, pre + "indexer.k_norm.weight");
        }

        if (il == qh.ple_layer) {
            X.ple.key = bind_tensor(f, pre + "ple_key.weight");
            X.ple.value = bind_tensor(f, pre + "ple_value.weight");
            X.ple.norm_key = bind_f32(f, pre + "ple_norm_key.weight");
            X.ple.norm_query = bind_f32(f, pre + "ple_norm_query.weight");
            X.ple.norm_conv = bind_f32(f, pre + "ple_norm_conv.weight");
            X.ple.conv = bind_f32(f, pre + "ple_conv1d.weight");
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

// Loader geometry vs the declarative profile (the generic validate_model_profile
// used to carry this inline for qwen4exp).
void qwen4exp_hparams::validate_profile(const hparams & hp, const model_profile & prof) const {
    (void)hp;
    const qwen4exp_hparams & qh = *this;
    auto expect = [&](const char * what, long got, long want) {
        if (got != want) {
            throw std::runtime_error("model profile " + std::string(prof.key) + ": " + what + " is " +
                                     std::to_string(got) + " in the GGUF but " + std::to_string(want) + " in the profile");
        }
    };
    expect("hyper-connection streams", qh.hc_mult, prof.residual.streams);
    expect("hyper-connection low rank", qh.hc_low_rank, prof.residual.low_rank);
    expect("QSA top-k", qh.idx_top_k, prof.extras.qsa_top_k);
    expect("QSA pool", qh.idx_kpool, prof.extras.qsa_pool);
    expect("QSA indexer heads", qh.idx_n_head, prof.extras.idx_heads);
    expect("QSA indexer head_dim", qh.idx_head_dim, prof.extras.idx_head_dim);
}

} // namespace si
