// DFlash / DFlash2 draft-model loader.  See dflash.h for what the architecture is.
#include "dflash.h"

#include "model_arch.h"

#include <cstdio>
#include <stdexcept>

#include "common/env.h"

namespace si {

namespace {

int arr_i32(const gguf_kv * kv, int i) {
    if (!kv || !kv->is_arr() || (int)kv->arr.size() <= i) {
        throw std::runtime_error("dflash: metadata array too short");
    }
    return (int)kv->arr[(size_t)i].i64;
}

} // namespace

void dflash_model::load(const std::string & path, int target_n_embd, int target_n_vocab) {
    gguf.load(path);
    const char * arch_name = "dflash";
    if (const std::string * a = gguf.get_str("general.architecture")) {
        arch_name = a->c_str();
    }
    if (std::string(arch_name) != "dflash") {
        throw std::runtime_error("dflash: not a dflash GGUF (architecture = " + std::string(arch_name) + ")");
    }
    hp.n_layer = (int)gguf.get_u32("dflash.block_count", 0);
    hp.n_embd = (int)gguf.get_u32("dflash.embedding_length", 0);
    hp.n_ff = (int)gguf.get_u32("dflash.feed_forward_length", 0);
    hp.n_head = (int)gguf.get_u32("dflash.attention.head_count", 0);
    hp.n_head_kv = (int)gguf.get_u32("dflash.attention.head_count_kv", 0);
    hp.head_dim = (int)gguf.get_u32("dflash.attention.key_length", 0);
    hp.rope_base = gguf.get_f32("dflash.rope.freq_base", 1000000.f);
    hp.rms_eps = gguf.get_f32("dflash.attention.layer_norm_rms_epsilon", 1e-5f);
    hp.n_ctx = (int)gguf.get_u32("dflash.context_length", 0);
    hp.swa = (int)gguf.get_u32("dflash.attention.sliding_window", 0);
    hp.block_size = (int)gguf.get_u32("dflash.block_size", 0);
    hp.conv_k = (int)gguf.get_u32("dflash.conv_kernel_size", 0);
    hp.conv_group = (int)gguf.get_u32("dflash.conv_group_size", 0);
    hp.sel_rank = (int)gguf.get_u32("dflash.selector_rank", 0);
    hp.sel_top_k = (int)gguf.get_u32("dflash.selector_top_k", 0);
    // The block's filler tokens are the *draft* vocab's mask id, which for this
    // GGUF family is tokenizer.ggml.mask_token_id (248070 for the 27B DFlash2) -
    // llama.cpp reads the same key via llama_vocab_mask().
    hp.mask_id = (int)gguf.get_u32("tokenizer.ggml.mask_token_id", 0);
    if (hp.mask_id <= 0) {
        throw std::runtime_error("dflash: tokenizer.ggml.mask_token_id missing from the draft GGUF");
    }
    // The draft's rope is a degenerate M-RoPE (`dimension_sections =
    // [n_rot, 0, 0, 0]`), so only section 0 rotates.  ggml's rope asserts
    // `n_dims == ne0/2` for MROPE and `rotate_pairs` then pairs NEOX-style over
    // the first n_dims dims, i.e. the section list counts *dims*, not pairs:
    // head_dim 128 with sections [64,0,0,0] rotates dims 0..63 only, with pair
    // m at base^(-2m/64).  Taking sections[0]*2 rotated all 128 dims with a 2x
    // too shallow exponent, which silently gave every head the wrong attention
    // scores (the draft still ran, but its layer hidden drifted ~8x by layer 1
    // and its candidate sets were unrelated to llama.cpp's).
    // The draft's rope rotates the WHOLE head_dim.  dimension_sections is
    // [64, 0, 0, 0], and reading sections[0] as the rotated width gives 64 - which
    // costs a fixed ~0.7 % on every attention score and compounds to cos 0.79 by
    // layer 4.  Measured against llama.cpp's DFLASH_REF_CUR tap at pos0=204:
    //
    //     n_rot  32       64       96       128
    //     row 0  0.9930   0.9932   0.9947   0.999991
    //     row 1  0.9901   0.9932   0.9916   0.999982
    //     row 2  0.9846   0.9889   0.9895   0.999979
    //
    // so ggml's MROPE section list is not a count of rotated dims the way the
    // assertion in the old comment assumed.
    hp.n_rot = hp.head_dim;
    if (const gguf_kv * sec = gguf.meta("dflash.rope.dimension_sections")) {
        if (sec->is_arr() && !sec->arr.empty()) {
            fprintf(stderr, "[dflash] rope: head_dim=%d dimension_sections[0]=%d -> rotating %d dims\n", hp.head_dim,
                    (int)sec->arr[0].i64, hp.n_rot);
        }
    }
    if (const char * e = si::env::str("PF_DFLASH_NROT")) {
        hp.n_rot = atoi(e);
    }
    const gguf_kv * tl = gguf.meta("dflash.target_layers");
    if (!tl || !tl->is_arr() || tl->arr.empty()) {
        throw std::runtime_error("dflash: missing dflash.target_layers");
    }
    hp.n_tgt_layer = std::min<int>((int)tl->arr.size(), (int)(sizeof(hp.tgt_layer) / sizeof(hp.tgt_layer[0])));
    for (int i = 0; i < hp.n_tgt_layer; i++) {
        hp.tgt_layer[i] = arr_i32(tl, i);
    }
    hp.n_feat = hp.n_tgt_layer * target_n_embd;
    hp.n_vocab = target_n_vocab;
    if (hp.n_layer <= 0 || hp.n_embd <= 0 || hp.n_head <= 0 || hp.head_dim <= 0 || hp.block_size <= 0) {
        throw std::runtime_error("dflash: incomplete hparams");
    }
    if (hp.n_embd % hp.n_head || (hp.n_head % hp.n_head_kv)) {
        throw std::runtime_error("dflash: bad head geometry");
    }

    fc = arch::bind_tensor(gguf, "fc.weight", 0xFFFFFFFF);
    if (fc.K != hp.n_feat || fc.N != hp.n_embd) {
        throw std::runtime_error("dflash: fc.weight does not match target_layers x target n_embd");
    }
    enc_norm = arch::bind_f32(gguf, "enc.output_norm.weight");
    output_norm = arch::bind_f32(gguf, "output_norm.weight");

    const bool has_sel = gguf.find("selector_hidden.weight") != nullptr;
    hp.is_dflash2 = has_sel;
    if (has_sel) {
        if (hp.sel_rank <= 0 || hp.sel_top_k <= 0 || hp.conv_k <= 0 || hp.conv_group <= 0) {
            throw std::runtime_error("dflash2: conv/selector metadata missing");
        }
        if (hp.n_embd % hp.conv_group) {
            throw std::runtime_error("dflash2: n_embd not divisible by conv_group_size");
        }
        if (hp.n_embd < hp.sel_top_k * (hp.sel_top_k + 1)) {
            throw std::runtime_error("dflash2: n_embd too small for the selector lattice");
        }
        sel_hidden = arch::bind_tensor(gguf, "selector_hidden.weight", 0xFFFFFFFF);
        sel_prev = arch::bind_tensor(gguf, "selector_predecessor.weight", 0xFFFFFFFF);
        sel_next = arch::bind_tensor(gguf, "selector_successor.weight", 0xFFFFFFFF);
        if (sel_hidden.K != hp.n_embd || sel_hidden.N != hp.sel_rank || sel_prev.K != hp.sel_rank
            || sel_next.K != hp.sel_rank) {
            throw std::runtime_error("dflash2: selector geometry mismatch");
        }
        hp.n_groups = hp.n_embd / hp.conv_group;
        hp.conv_proj = 2 * hp.conv_k * hp.n_groups;
    }

    layers.resize((size_t)hp.n_layer);
    for (int i = 0; i < hp.n_layer; i++) {
        dflash_layer_t & L = layers[(size_t)i];
        const std::string p = "blk." + std::to_string(i) + ".";
        L.attn_norm = arch::bind_f32(gguf, p + "attn_norm.weight");
        L.q_norm = arch::bind_f32(gguf, p + "attn_q_norm.weight");
        L.k_norm = arch::bind_f32(gguf, p + "attn_k_norm.weight");
        L.ffn_norm = arch::bind_f32(gguf, p + "ffn_norm.weight");
        L.wq = arch::bind_tensor(gguf, p + "attn_q.weight", 0xFFFFFFFF);
        L.wk = arch::bind_tensor(gguf, p + "attn_k.weight", 0xFFFFFFFF);
        L.wv = arch::bind_tensor(gguf, p + "attn_v.weight", 0xFFFFFFFF);
        L.wo = arch::bind_tensor(gguf, p + "attn_output.weight", 0xFFFFFFFF);
        L.ffn_gate = arch::bind_tensor(gguf, p + "ffn_gate.weight", 0xFFFFFFFF);
        L.ffn_up = arch::bind_tensor(gguf, p + "ffn_up.weight", 0xFFFFFFFF);
        L.ffn_down = arch::bind_tensor(gguf, p + "ffn_down.weight", 0xFFFFFFFF);
        if (L.wq.K != hp.n_embd || L.wq.N != hp.n_head * hp.head_dim || L.wk.N != hp.n_head_kv * hp.head_dim
            || L.wv.N != hp.n_head_kv * hp.head_dim || L.wo.K != hp.n_head * hp.head_dim || L.wo.N != hp.n_embd) {
            throw std::runtime_error("dflash: blk." + std::to_string(i) + " attention geometry mismatch");
        }
        if (has_sel) {
            // llama.cpp names this LLM_TENSOR_ATTN_SINKS -> "blk.N.attn_sinks.weight";
            // accept the bare spelling too rather than hard-failing the whole drafter.
            // Optional: this GGUF has no attn_sinks, and llama.cpp creates the tensor
            // with TENSOR_NOT_REQUIRED, so a null sink is the correct state here.  When
            // one IS present it has to be applied - ggml's ggml_soft_max_add_sinks
            // gives every head's output an extra exp(sink) in the denominator - so it
            // is wired up rather than assumed absent.
            L.attn_sinks = gguf.find(p + "attn_sinks.weight") ? arch::bind_f32(gguf, p + "attn_sinks.weight")
                                                             : nullptr;
            L.attn_conv_base = arch::bind_f32(gguf, p + "attn_conv_base");
            L.ffn_conv_base = arch::bind_f32(gguf, p + "ffn_conv_base");
            L.attn_conv_proj = arch::bind_tensor(gguf, p + "attn_conv_proj.weight", 0xFFFFFFFF);
            L.ffn_conv_proj = arch::bind_tensor(gguf, p + "ffn_conv_proj.weight", 0xFFFFFFFF);
            if (L.attn_conv_proj.K != hp.n_embd || L.attn_conv_proj.N != hp.conv_proj) {
                throw std::runtime_error("dflash2: blk." + std::to_string(i) + " conv_proj geometry mismatch");
            }
        }
    }
}

void dflash_model::upload_f32(sycl::queue & q) {
    auto up = [&](const void * p, size_t bytes) {
        if (!p || dev.count(p)) {
            return;
        }
        void * g = sycl::malloc_device(bytes, q);
        if (!g) {
            throw std::runtime_error("dflash: device allocation failed");
        }
        q.memcpy(g, p, bytes).wait();
        dev[p] = g;
    };
    const size_t en = (size_t)hp.n_embd * 4;
    up(enc_norm, en);
    up(output_norm, en);
    const size_t cb = (size_t)hp.n_embd * hp.conv_k * 2 * 4;
    for (int i = 0; i < hp.n_layer; i++) {
        const dflash_layer_t & L = layers[(size_t)i];
        up(L.attn_norm, en);
        up(L.ffn_norm, en);
        up(L.q_norm, (size_t)hp.head_dim * 4);
        up(L.k_norm, (size_t)hp.head_dim * 4);
        if (L.attn_conv_base) {
            up(L.attn_conv_base, cb);
            up(L.ffn_conv_base, cb);
            // sinks is one float per query head, not a conv-base-sized tensor.
            up(L.attn_sinks, (size_t)hp.n_head * 4);
        }
    }
}

size_t dflash_model::bytes() const {
    size_t n = 0;
    auto add = [&](const wt & t) {
        if (t.data) {
            n += (size_t)t.N * ggml_row_bytes(t.type, (uint64_t)t.K);
        }
    };
    add(fc);
    add(sel_hidden);
    add(sel_prev);
    add(sel_next);
    for (const auto & L : layers) {
        // PF_W4_INFO-style per-tensor dump is on whenever PF_W4_INFO is set
        if (si::env::flag("PF_W4_INFO")) {
            const char * tyn[9] = {"wq", "wk", "wv", "wo", "fgate", "fup", "fdown", "acproj", "fcproj"};
            const wt * tw[9] = {&L.wq, &L.wk, &L.wv, &L.wo, &L.ffn_gate, &L.ffn_up, &L.ffn_down, &L.attn_conv_proj,
                                &L.ffn_conv_proj};
            for (int i = 0; i < 9; i++) {
                fprintf(stderr, "[dflash] %-7s type=%d K=%d N=%d\n", tyn[i], (int)tw[i]->type, tw[i]->K, tw[i]->N);
            }
        }
        add(L.wq);
        add(L.wk);
        add(L.wv);
        add(L.wo);
        add(L.ffn_gate);
        add(L.ffn_up);
        add(L.ffn_down);
        add(L.attn_conv_proj);
        add(L.ffn_conv_proj);
    }
    return n;
}

} // namespace si
