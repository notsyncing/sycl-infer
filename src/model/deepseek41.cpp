// DeepSeek-V4.1-Flash ("deepseek41") loader.
//
// Binds the MLA attention (low-rank q, 512-wide K=V latent, 8-group output),
// the group-shared compressed-KV owners and their DSA indexers, the 4-stream
// hyper-connection mixers, the 384-expert MoE with its ungated shared expert,
// and the two Engram layers.  See src/model/profiles/deepseek41.cpp for why the
// forward is model-owned and reports/ds41_forward.md for the reference math.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "model_arch.h"

namespace si {
namespace arch {

namespace {

moe_wt bind_moe3(const gguf_file & f, const std::string & name) {
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

void append_unique(std::vector<int> & v, int x) {
    for (int y : v) {
        if (y == x) {
            return;
        }
    }
    v.push_back(x);
}

} // namespace

void load_deepseek41(model & m) {
    const gguf_file & f = m.gguf;
    auto key = [](const char * k) { return std::string("deepseek41.") + k; };
    auto arr = [&](const char * k) -> const std::vector<gguf_kv> * {
        const gguf_kv * v = f.meta(key(k));
        return (v && v->is_arr()) ? &v->arr : nullptr;
    };

    hparams & hp = m.hp;
    hp.deepseek41 = true;
    m.ds41 = true;

    hp.n_layer = (int)f.get_u32(key("block_count"));
    hp.n_mtp = 0;
    hp.n_embd = (int)f.get_u32(key("embedding_length"));
    hp.n_head = (int)f.get_u32(key("attention.head_count"));
    hp.n_head_kv = (int)f.get_u32(key("attention.head_count_kv"));
    hp.head_dim = (int)f.get_u32(key("attention.key_length"));
    hp.n_rot = (int)f.get_u32(key("rope.dimension_count"));
    hp.n_ctx = (int)f.get_u32(key("context_length"), 0);
    hp.rope_base = f.get_f32(key("rope.freq_base"), 10000.f);
    hp.rms_eps = f.get_f32(key("attention.layer_norm_rms_epsilon"), 1e-20f);
    hp.full_attn_interval = 1; // every layer is attention; there is no GDN half
    hp.attn_scale = 1.0f / std::sqrt((float)hp.head_dim);
    hp.attn_gate = false;
    hp.mrope_interleaved = 0;

    ds41_hparams & ds = hp.ds;
    ds.q_lora_rank = (int)f.get_u32(key("attention.q_lora_rank"));
    ds.out_group_count = (int)f.get_u32(key("attention.output_group_count"));
    ds.out_lora_rank = (int)f.get_u32(key("attention.output_lora_rank"));
    ds.sliding_window = (int)f.get_u32(key("attention.sliding_window"));
    ds.kv_dim = hp.head_dim;
    ds.rot_off = ds.kv_dim - hp.n_rot;
    ds.rope_base = hp.rope_base;
    ds.compress_rope_base = f.get_f32(key("attention.compress_rope_freq_base"), 0.f);
    ds.yarn_factor = f.get_f32(key("rope.scaling.factor"), 0.f);
    ds.yarn_orig = (int)f.get_u32(key("rope.scaling.original_context_length"), 0);
    ds.yarn_beta_fast = f.get_f32(key("rope.scaling.yarn_beta_fast"), 0.f);
    ds.yarn_beta_slow = f.get_f32(key("rope.scaling.yarn_beta_slow"), 0.f);

    ds.idx_n_head = (int)f.get_u32(key("attention.indexer.head_count"));
    ds.idx_head_dim = (int)f.get_u32(key("attention.indexer.key_length"));
    ds.idx_top_k = (int)f.get_u32(key("attention.indexer.top_k"));
    ds.cand_src = (int)f.get_u32(key("attention.indexer.candidate_source_layer"), 0xFFFFFFFFu);
    ds.cand_topk_blocks = (int)f.get_u32(key("attention.indexer.candidate_top_k_blocks"));
    ds.cand_block = (int)f.get_u32(key("attention.indexer.candidate_block_size"));

    ds.hc_mult = (int)f.get_u32(key("hyper_connection.count"));
    ds.hc_sinkhorn = (int)f.get_u32(key("hyper_connection.sinkhorn_iterations"));
    ds.hc_eps = f.get_f32(key("hyper_connection.epsilon"), 1e-6f);
    ds.hc_mix_dim = (2 + ds.hc_mult) * ds.hc_mult;

    ds.n_expert_shared = (int)f.get_u32(key("expert_shared_count"));
    ds.expert_scale = f.get_f32(key("expert_weights_scale"), 1.f);
    ds.expert_norm = f.get_u32(key("expert_weights_norm"), 1) != 0;

    hp.n_expert = (int)f.get_u32(key("expert_count"));
    hp.n_expert_used = (int)f.get_u32(key("expert_used_count"));
    hp.n_ff_exp = (int)f.get_u32(key("expert_feed_forward_length"));
    hp.expert_weights_scale = ds.expert_scale;
    hp.n_ff_shexp = hp.n_ff_exp;
    hp.n_ff = hp.n_ff_shexp;

    // Per-layer arrays.
    if (const auto * a = arr("swiglu_clamp_exp")) {
        for (int il = 0; il < hp.n_layer && il < (int)a->size(); il++) {
            ds.swiglu_clamp[il] = (*a)[(size_t)il].as_f32();
        }
    }
    if (const auto * a = arr("swiglu_clamp_shexp")) {
        for (int il = 0; il < hp.n_layer && il < (int)a->size(); il++) {
            ds.swiglu_clamp_shexp[il] = (*a)[(size_t)il].as_f32();
        }
    } else {
        for (int il = 0; il < hp.n_layer; il++) {
            ds.swiglu_clamp_shexp[il] = ds.swiglu_clamp[il];
        }
    }
    if (const auto * a = arr("attention.compress_ratios")) {
        for (int il = 0; il < hp.n_layer && il < (int)a->size(); il++) {
            const int r = (int)(*a)[(size_t)il].as_u32();
            if (r < 0 || r > 2) {
                throw std::runtime_error("deepseek41: unsupported compress ratio " + std::to_string(r));
            }
            ds.compress_ratio[il] = r;
        }
    }

    // KV / index sources and the "greatest source <= il" resolution.  Layers
    // before the first source are window-only; every later layer reads the
    // owner's compressed rows even when its own ratio is 0.
    std::vector<int> kv_src, idx_src;
    if (const auto * a = arr("attention.kv_source_layers")) {
        for (const gguf_kv & e : *a) {
            const int v = (int)e.as_u32();
            if (v >= 0 && v < hp.n_layer) {
                append_unique(kv_src, v);
            }
        }
    }
    if (const auto * a = arr("attention.indexer.source_layers")) {
        for (const gguf_kv & e : *a) {
            const int v = (int)e.as_u32();
            if (v >= 0 && v < hp.n_layer) {
                append_unique(idx_src, v);
            }
        }
    }
    ds.n_kv_src = (int)kv_src.size();
    ds.n_idx_src = (int)idx_src.size();
    for (int i = 0; i < ds.n_kv_src && i < 64; i++) {
        ds.kv_src[i] = kv_src[(size_t)i];
    }
    for (int i = 0; i < ds.n_idx_src && i < 64; i++) {
        ds.idx_src[i] = idx_src[(size_t)i];
    }
    {
        int last_kv = -1, last_idx = -1;
        for (int il = 0; il < hp.n_layer; il++) {
            for (int s : kv_src) {
                if (s == il) {
                    last_kv = il;
                }
            }
            for (int s : idx_src) {
                if (s == il) {
                    last_idx = il;
                }
            }
            ds.kv_src_of[il] = last_kv;
            ds.topk_src_of[il] = last_idx;
        }
        // A compressed layer must read a source of the same ratio; that is
        // what makes the group-shared cache unambiguous.
        for (int il = 0; il < hp.n_layer; il++) {
            const int r = ds.compress_ratio[il];
            if (r == 0) {
                continue;
            }
            if (last_kv < 0) {
                throw std::runtime_error("deepseek41: compressed layer " + std::to_string(il) +
                                         " has no preceding KV source");
            }
            const int src_r = ds.compress_ratio[ds.kv_src_of[il]];
            if (src_r != r) {
                throw std::runtime_error("deepseek41: layer " + std::to_string(il) + " ratio " +
                                         std::to_string(r) + " reads source " +
                                         std::to_string(ds.kv_src_of[il]) + " ratio " +
                                         std::to_string(src_r));
            }
        }
        bool cand_ok = ds.cand_src >= 0;
        for (int s : idx_src) {
            if (s == ds.cand_src) {
                cand_ok = true;
            }
        }
        if (!cand_ok) {
            throw std::runtime_error("deepseek41: candidate_source_layer is not an index source");
        }
    }

    // Engram / PLE.
    std::vector<int> ple_layers;
    if (const auto * a = arr("ple.layers")) {
        for (const gguf_kv & e : *a) {
            const int v = (int)e.as_u32();
            if (v >= 0 && v < hp.n_layer) {
                append_unique(ple_layers, v);
            }
        }
    }
    if (!ple_layers.empty()) {
        ds.ple_ngram = (int)f.get_u32(key("ple.ngram_size"));
        ds.ple_heads_per_ngram = (int)f.get_u32(key("ple.heads_per_ngram"));
        ds.ple_head_dim = (int)f.get_u32(key("embedding_length_per_layer_input"));
        ds.ple_n_heads = (ds.ple_ngram - 1) * ds.ple_heads_per_ngram;
        if (ds.ple_n_heads <= 0 || ds.ple_n_heads > 64 || ds.ple_head_dim <= 0) {
            throw std::runtime_error("deepseek41: unsupported Engram geometry");
        }
        const auto * mult = arr("ple.layer_multipliers");
        const auto * off = arr("ple.head_offsets");
        const auto * voc = arr("ple.head_vocab_sizes");
        if (!mult || !off || !voc || (int)mult->size() < ds.ple_ngram * (int)ple_layers.size()
            || (int)off->size() < ds.ple_n_heads * (int)ple_layers.size()
            || (int)voc->size() < ds.ple_n_heads * (int)ple_layers.size()) {
            throw std::runtime_error("deepseek41: Engram hash constant arrays are too short");
        }
        for (size_t i = 0; i < mult->size() && i < 8; i++) {
            ds.ple_mult[i] = (*mult)[i].u64;
        }
        for (int e = 0; e < (int)ple_layers.size(); e++) {
            for (int c = 0; c < ds.ple_n_heads; c++) {
                const size_t j = (size_t)e * ds.ple_n_heads + (size_t)c;
                ds.ple_off[j] = (*off)[j].as_u32();
                ds.ple_vocab[j] = (*voc)[j].as_u32();
            }
        }
        ds.engram_pad = (int)f.get_u32(key("engram.pad_token_id"));
        int e = 0;
        for (int il = 0; il < hp.n_layer; il++) {
            ds.engram_index_of[il] = e;
            if (il == ple_layers[(size_t)std::min<size_t>((size_t)e, ple_layers.size() - 1)]) {
                ds.is_ple[il] = true;
                e++;
            }
        }
    }

    hp.n_vocab = 0;
    if (const auto * toks = f.meta("tokenizer.ggml.tokens")) {
        hp.n_vocab = (int)toks->arr.size();
    }

    // The image marker is a tokenizer id; the mm planner needs it.
    if (const auto * toks = f.meta("tokenizer.ggml.tokens")) {
        for (int i = 0; i < (int)toks->arr.size(); i++) {
            if (toks->arr[(size_t)i].str == "<｜deepseek_image｜>") {
                ds.image_token_id = i;
                break;
            }
        }
    }

    if (const auto * tm = f.meta(key("engram.token_map"))) {
        m.engram_token_map.resize(tm->arr.size());
        for (size_t i = 0; i < tm->arr.size(); i++) {
            m.engram_token_map[i] = tm->arr[i].as_i32();
        }
    }

    // ---- globals --------------------------------------------------------
    m.tok_embd = bind_tensor(f, "token_embd.weight");
    m.output = bind_tensor(f, "output.weight"); // untied on this model
    m.output_norm = bind_f32(f, "output_norm.weight");
    m.tok_embd_row_bytes = ggml_row_bytes(m.tok_embd.type, hp.n_embd);
    if (!ple_layers.empty()) {
        m.ple_table = bind_tensor(f, "per_layer_token_embd.weight");
    }

    // ---- layers ---------------------------------------------------------
    m.layers.resize(hp.n_layer);
    m.gdn_layer_index.assign(hp.n_layer, -1);
    for (int il = 0; il < hp.n_layer; il++) {
        layer_t & L = m.layers[il];
        const std::string pre = "blk." + std::to_string(il) + ".";
        L.recurrent = false;

        L.attn_norm = bind_f32(f, pre + "attn_norm.weight");
        L.post_attn_norm = bind_f32(f, pre + "ffn_norm.weight");

        L.ds.wq_a = bind_tensor(f, pre + "attn_q_a.weight");
        L.ds.q_a_norm = bind_f32(f, pre + "attn_q_a_norm.weight");
        L.ds.wq_b = bind_tensor(f, pre + "attn_q_b.weight");
        L.ds.wkv = bind_tensor(f, pre + "attn_kv.weight");
        L.ds.kv_norm = bind_f32(f, pre + "attn_kv_a_norm.weight");
        L.ds.wo_a = bind_tensor(f, pre + "attn_output_a.weight");
        L.ds.wo_b = bind_tensor(f, pre + "attn_output_b.weight");
        L.ds.attn_sinks = bind_f32(f, pre + "attn_sinks.weight");

        L.ds.hc_attn_fn = bind_tensor(f, pre + "hc_attn_fn.weight");
        L.ds.hc_attn_base = bind_f32(f, pre + "hc_attn_base.weight");
        L.ds.hc_attn_scale = bind_f32(f, pre + "hc_attn_scale.weight");
        L.ds.hc_ffn_fn = bind_tensor(f, pre + "hc_ffn_fn.weight");
        L.ds.hc_ffn_base = bind_f32(f, pre + "hc_ffn_base.weight");
        L.ds.hc_ffn_scale = bind_f32(f, pre + "hc_ffn_scale.weight");

        // shared expert -> the ordinary dense FFN slots; the routed half is
        // `moe`.  The shared expert is added ungated (no gate_inp_shexp).
        L.ffn_gate = bind_tensor(f, pre + "ffn_gate_shexp.weight");
        L.ffn_up = bind_tensor(f, pre + "ffn_up_shexp.weight");
        L.ffn_down = bind_tensor(f, pre + "ffn_down_shexp.weight");
        L.moe.gate_exps = bind_moe3(f, pre + "ffn_gate_exps.weight");
        L.moe.up_exps = bind_moe3(f, pre + "ffn_up_exps.weight");
        L.moe.down_exps = bind_moe3(f, pre + "ffn_down_exps.weight");
        {
            const gguf_tensor_info * gi = f.find(pre + "ffn_gate_inp.weight");
            if (!gi) {
                throw std::runtime_error("missing tensor: " + pre + "ffn_gate_inp.weight");
            }
            L.moe.gate_inp = gi->data;
            L.moe.gate_inp_type = gi->type;
        }
        if (const gguf_tensor_info * b = f.find(pre + "exp_probs_b.bias")) {
            L.ds.exp_probs_b = (const float *)b->data;
        }
        if (const gguf_tensor_info * b = f.find(pre + "exp_probs_b_vl.bias")) {
            L.ds.exp_probs_b_vl = (const float *)b->data;
        }

        if (ds.kv_src_of[il] == il) {
            L.ds.comp_wkv = bind_tensor(f, pre + "attn_compressor_kv.weight");
            L.ds.comp_norm = bind_f32(f, pre + "attn_compressor_norm.weight");
            if (ds.compress_ratio[il] > 1) {
                L.ds.comp_wgate = bind_tensor(f, pre + "attn_compressor_gate.weight");
            }
            L.ds.idx_attn_k = bind_tensor(f, pre + "indexer.attn_k.weight");
            L.ds.idx_k_norm = bind_f32(f, pre + "indexer.k_norm.weight");
        }
        if (ds.topk_src_of[il] == il) {
            L.ds.idx_attn_q_b = bind_tensor(f, pre + "indexer.attn_q_b.weight");
            L.ds.idx_proj = bind_tensor(f, pre + "indexer.proj.weight");
        }
        if (ds.is_ple[il]) {
            L.ds.engram_kv = bind_tensor(f, pre + "engram_kv.weight");
            L.ds.engram_gate = bind_f32(f, pre + "engram_gate.weight");
        }
    }
}

} // namespace arch
} // namespace si
