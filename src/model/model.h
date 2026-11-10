#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <sycl/sycl.hpp> // IWYU pragma: keep

#include "gguf.h"
#include "model_profile.h"
#include "w8.h"

namespace si {

struct wt { // weight tensor view
    const void * data = nullptr;
    uint32_t type = 0;
    int32_t K = 0; // input dim (dims[0])
    int32_t N = 0; // output rows (product of dims[1:])
};

// One Mixture-of-Experts weight: the GGUF stores the routed experts as a 3-D
// tensor [K][N][E], so a single expert is exactly a 2-D weight of the same
// (K, N) as any other linear - `plane(e)` is `data + e*plane_bytes`, which is
// why every existing GEMV path takes an expert unchanged.  Keeping the expert
// count and the plane stride here (rather than a std::vector of `wt`) is what
// lets one expert be handed to the GPU as an ordinary base pointer while the
// expert *index* stays a runtime value the kernels read from device memory.
struct moe_wt {
    const void * data = nullptr;
    uint32_t type = 0;
    int32_t K = 0;
    int32_t N = 0;
    int32_t E = 0;
    size_t plane_bytes = 0;

    bool ok() const {
        return data != nullptr && E > 0;
    }
    const void * plane(int e) const {
        return (const char *)data + (size_t)e * plane_bytes;
    }
    size_t total_bytes() const {
        return plane_bytes * (size_t)E;
    }
};

// ---- DeepSeek-V4.1-Flash (`deepseek41`) --------------------------------
// MLA attention (512-wide latent, K == V), compressed sparse attention with a
// two-level DSA indexer, group-shared KV caches, 4-stream hyper-connections and
// Engram.  The per-layer source/ratio arrays are resolved at load time
// (`kv_src_of`/`topk_src_of`) so no use site re-derives the "greatest source
// <= il" rule, and the hash constants are stored per Engram layer.
struct ds41_hparams {
    int q_lora_rank = 0;     // 1280
    int out_group_count = 0; // 8
    int out_lora_rank = 0;   // 1024
    int sliding_window = 0;  // 128 (includes self)
    int kv_dim = 0;          // 512 == key_length == value_length
    int rot_off = 0;         // kv_dim - n_rot (448): RoPE on the LAST n_rot dims
    float rope_base = 0.f;   // ratio-0 layers
    float compress_rope_base = 0.f; // ratio>0 layers (160000)
    float yarn_factor = 0.f;
    int yarn_orig = 0;
    float yarn_beta_fast = 0.f, yarn_beta_slow = 0.f;

    int idx_n_head = 0, idx_head_dim = 0, idx_top_k = 0;
    int cand_src = -1, cand_topk_blocks = 0, cand_block = 0;
    int n_idx_src = 0, idx_src[64] = {0};
    int topk_src_of[64] = {0}; // greatest index source <= il (-1 before the first)

    int n_kv_src = 0, kv_src[64] = {0};
    int kv_src_of[64] = {0}; // greatest KV source <= il (-1 before the first)
    int compress_ratio[64] = {0};

    int hc_mult = 0, hc_mix_dim = 0, hc_sinkhorn = 0;
    float hc_eps = 0.f;

    int n_expert_shared = 0;
    float expert_scale = 1.f;
    bool expert_norm = false;
    float swiglu_clamp[64] = {0.f};
    float swiglu_clamp_shexp[64] = {0.f};

    bool is_ple[64] = {false};
    int engram_index_of[64] = {0}; // number of Engram layers strictly before il
    int ple_ngram = 0, ple_heads_per_ngram = 0, ple_head_dim = 0, ple_n_heads = 0;
    uint64_t ple_mult[8] = {0};
    uint32_t ple_off[64] = {0}, ple_vocab[64] = {0};
    int engram_pad = 0;
    int image_token_id = -1;
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
    // Qwen3.5-family text models (qwen35, qwen35moe, qwen4exp) use Qwen3-VL's
    // *interleaved* M-RoPE: sector = pair % sect_dims and the temporal/row/col
    // axis cycles by sector % 3.  Qwen2-VL uses blocked sections instead.
    // The two are indistinguishable from `rope_sections` alone on the 27B/35B
    // (sections sum to n_rot/2 there too), so the layout is an explicit
    // architecture fact rather than a derived one.
    int mrope_interleaved = 0;

    // ---- Mixture of Experts (`qwen35moe` and friends) ----------------------
    // n_expert == 0 means the FFN is dense and n_ff is its width.  When it is
    // not, n_ff holds the *shared* expert's width (n_ff_shexp), because the
    // shared expert is the dense half of the FFN and reuses every dense path;
    // the routed half runs through the MoE kernels with n_ff_exp per expert.
    int n_expert = 0;
    int n_expert_used = 0;
    int n_ff_exp = 0;
    int n_ff_shexp = 0;
    float expert_weights_scale = 1.0f;

    // ---- Qwen3.8-Flash-Next / qwen4exp -------------------------------------
    // Hyper-connections replace every layer norm: the residual stream is `hc`
    // parallel copies of the hidden state and each block mixes them through a
    // low-rank gate.  PLE is an n-gram-hash embedding table on one layer; QSA
    // is a sparse selector on the full-attention layers.  See qwen4exp.cpp.
    bool q4exp = false;
    int hc_mult = 0;
    int hc_low_rank = 0;
    int idx_n_head = 0;
    int idx_head_dim = 0;
    int idx_top_k = 0;
    int idx_kpool = 0; // QSA compress ratio (0 = dense)
    int ple_layer = -1;
    int ple_n_heads = 0;
    int ple_head_dim = 0;
    int ple_ngram = 0;
    int ple_heads_per_ngram = 0;
    int ple_conv = 0;
    int ple_eos = 0;
    int ple_image = 0;
    int64_t ple_mult[8] = {0};
    uint32_t ple_off[64] = {0};
    uint32_t ple_vocab[64] = {0};
    bool gdn_sigmoid_gate = false; // qwen4exp's gated norm uses sigmoid, not silu
    // Qwen3.5 fuses the attention output gate into attn_q (q and gate in one
    // projection).  Qwen2/Qwen2-VL do not: their q projection is exactly
    // n_head*head_dim wide and the attention output has no sigmoid gate.
    bool attn_gate = true;

    // ---- DeepSeek-V4.1-Flash (`deepseek41`) ------------------------------
    bool deepseek41 = false;
    ds41_hparams ds;

    bool is_moe() const {
        return n_expert > 0;
    }

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

// The routed half of a Mixture-of-Experts FFN.  Empty (`ok()` false) for a
// dense layer.  `gate_exps`/`up_exps`/`down_exps` are the GGUF's [K][N][E]
// tensors; `gate_inp` is the router and `gate_inp_shexp` the shared expert's
// per-token sigmoid gate.  The router's element type is the GGUF's (the trunk
// layers ship f32 and the NextN block bf16), so it travels with the pointer
// rather than being assumed.
struct moe_t {
    moe_wt gate_exps, up_exps, down_exps;
    const void * gate_inp = nullptr;
    uint32_t gate_inp_type = 0;
    const void * gate_inp_shexp = nullptr;
    uint32_t gate_inp_shexp_type = 0;
    // qwen4exp: the down projection converted once to a symmetric int8 store
    // with one scale per 32-value group (the Q5_1/Q8_0 mixed bank cannot be
    // consumed by the K-quant int8 GEMM otherwise).  Host-only.
    uint8_t * down_i8 = nullptr;
    float * down_sc = nullptr;
    int down_i8_K = 0, down_i8_N = 0, down_i8_E = 0;

    bool ok() const {
        return gate_exps.ok();
    }
    size_t bytes() const {
        return gate_exps.total_bytes() + up_exps.total_bytes() + down_exps.total_bytes();
    }
};

// A hyper-connection module: grouped RMSNorm gamma [hc][n_embd], the low-rank
// gate down/up, and the per-stream scatter projection [hc].
struct hc_wt {
    const float * norm = nullptr;
    wt down, up, inject;
    bool ok() const {
        return norm && down.data && up.data && inject.data;
    }
};

// PLE (per-layer n-gram embedding) tensors on the one layer that has them.
struct ple_wt {
    wt key, value;
    const float * norm_key = nullptr;
    const float * norm_query = nullptr;
    const float * norm_conv = nullptr;
    const float * conv = nullptr; // depthwise taps [ple_conv][hc*n_embd]
    bool ok() const {
        return key.data && value.data && norm_key && norm_query && norm_conv && conv;
    }
};

// QSA (query-side sparse attention) indexer projections on a full-attention
// layer.  The stored keys are 128-wide, four indexer heads.
struct indexer_wt {
    wt q_proj, k_proj;
    const float * q_norm = nullptr;
    const float * k_norm = nullptr;
    bool ok() const {
        return q_proj.data && k_proj.data && q_norm && k_norm;
    }
};

// DeepSeek-V4.1-Flash layer tensors.  The attention is MLA: `wq_a`/`wq_b` is a
// low-rank q, `wkv` is the single 512-wide latent used as both K and V, and the
// output is `wo_a` (8 groups of 1024) then `wo_b`.  `wo_a` is stored 3-D
// [K=4096][N=1024][E=8], so one group is a plain 2-D weight.
struct ds41_layer {
    wt wq_a, wq_b, wkv, wo_b;
    // attn_output_a is stored 2-D [K=4096][N=8192]; the eight output groups are
    // consecutive 1024-row blocks of N (the reference reshapes it to
    // [4096, 1024, 8] for the per-group GEMV).
    wt wo_a;
    const float * q_a_norm = nullptr;
    const float * kv_norm = nullptr;
    const float * attn_sinks = nullptr; // [n_head]
    // compressed-KV owner only (kv_src_of[il] == il)
    wt comp_wkv, comp_wgate; // comp_wgate absent at ratio 1
    const float * comp_norm = nullptr;
    wt idx_attn_k; // [512,128], owner only
    const float * idx_k_norm = nullptr;
    // index source only (topk_src_of[il] == il)
    wt idx_attn_q_b, idx_proj;
    // Engram layer only (1, 14)
    wt engram_kv; // [6144, 5*n_embd]
    const float * engram_gate = nullptr;
    // hyper-connections
    wt hc_attn_fn, hc_ffn_fn; // [hc*n_embd, (2+hc)*hc]
    const float * hc_attn_base = nullptr;
    const float * hc_attn_scale = nullptr;
    const float * hc_ffn_base = nullptr;
    const float * hc_ffn_scale = nullptr;
    // router biases (the `_vl` variant is used for image-span tokens)
    const float * exp_probs_b = nullptr;
    const float * exp_probs_b_vl = nullptr;
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
    // qwen4exp: hyper-connection mixers before the token mixer and before the
    // FFN, plus the optional PLE module and the full-attention indexer.
    hc_wt hc_attn, hc_ffn;
    ple_wt ple;
    indexer_wt idx;
    // deepseek41 (MLA + compressor + indexer + Engram + mHC)
    ds41_layer ds;
    const float * attn_norm = nullptr;
    const float * post_attn_norm = nullptr;
    wt ffn_gate, ffn_up, ffn_down;
    // Mixture-of-Experts half of the FFN.  Empty for a dense layer; when set,
    // `ffn_gate/ffn_up/ffn_down` above hold the *shared* expert (a plain dense
    // FFN of width n_ff_shexp) and `moe` holds the routed experts.
    moe_t moe;
    // attention layers
    wt wq, wk, wv, wo;
    const float * q_norm = nullptr;
    const float * k_norm = nullptr;
    // optional projection biases (Qwen2/Qwen2-VL carry q/k/v bias, no qk norm)
    const float * q_bias = nullptr;
    const float * k_bias = nullptr;
    const float * v_bias = nullptr;
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
    moe_t moe;
    // NextN extras
    wt eh_proj;                  // [2*n_embd][K=2*n_embd] -> n_embd
    const float * enorm = nullptr;
    const float * hnorm = nullptr;
    const float * shared_head_norm = nullptr;
    wt shared_head;              // optional; empty -> model::output
    // per-device SIn int8 copies (GPU decode path)
    w8t wq8, wk8, wv8, wo8, ffn_gate8, ffn_up8, ffn_down8, eh_proj8;
};

// Capabilities of an architecture's own forward that the generic driver has to
// know about.  Set by the loader, read by the engine - a capability rather than
// a model-name test, so a second model with the same geometry inherits it.
struct arch_profile {
    // A multimodal prompt must go through the model's own forward chunker.  The
    // generic mm driver feeds one kMaxT row per prefill_chunk, which is exactly
    // the path a model-owned forward (qwen4exp: hyper-connections, PLE, QSA and
    // the per-expert bank) bypasses; its whole-prompt batching is where those
    // live.
    bool mm_model_prefill = false;
};

// Reject a model whose geometry the kernels do not implement, before anything
// is uploaded or allocated.  Several kernels hardcode a head_dim (the generic
// attention partial stride is literally 256) and would otherwise overrun their
// buffers rather than merely lose accuracy.  Throws std::runtime_error naming
// the first violated constraint.  Called by model::load for every architecture,
// so a new loader inherits it; exposed so a test can drive it without a GGUF.
void validate_hparams(const hparams & hp, const std::string & arch);

// Cross-check a loader's GGUF-derived geometry against the architecture's
// declarative profile.  A mismatch is a loader/profile drift (or a file that
// looks like the architecture but is not), and the two views must not diverge
// silently.  Exposed so a test can drive it without a GGUF.
void validate_model_profile(const hparams & hp, const model_profile & prof, bool split_file);

struct model {
    gguf_file gguf;
    hparams hp;
    // The architecture's declarative profile, set by model::load from the
    // registry row that matched general.architecture.  Never null after a
    // successful load; the row's key is asserted against the GGUF's exact key.
    const model_profile * prof = nullptr;
    // what this architecture's own forward can do (see arch_profile)
    arch_profile profile;
    // tokenizer.chat_template (Jinja); empty when the GGUF does not carry one
    std::string chat_template;
    std::vector<layer_t> layers;
    std::vector<int> gdn_layer_index; // layer id -> sequential index among GDN layers (-1)
    wt tok_embd;
    // LM-head weight: `output.weight` when the GGUF is untied, else tok_embd
    wt output;
    const float * output_norm = nullptr;
    // qwen4exp: the final hyper-connection mixer is the output norm, and this
    // is the n-gram hash table (host-only: 26.8 GiB in the reference Q4 file).
    hc_wt hc_head;
    wt ple_table;
    bool q4exp = false;
    // deepseek41: the Engram table (110 GiB in the reference file) and the
    // vocab -> n-gram-id map the hash uses (a GGUF metadata array, not a
    // tensor).  Both are host-only; the forward reads rows on demand.
    bool ds41 = false;
    std::vector<int32_t> engram_token_map;
    // Keep the GGUF shards' page cache when page_out_host() releases the PTEs.
    // MADV_DONTNEED alone drops only this process's private mappings, so the
    // file pages stay cached; the explicit evictor is the POSIX_FADV_DONTNEED
    // that page_out_host issues afterwards.  qwen4exp keeps the file warm so a
    // restart does not re-read 111.5 GB from disk (PF_Q4_KEEPCACHE).
    bool keep_file_cache = false;
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
