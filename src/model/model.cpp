#include "model.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "common/env.h"
#include "arch/registry.h"
#include "arch_util.h"
#include "model_profile.h"

namespace si {

namespace arch {

wt bind_tensor(const gguf_file & f, const std::string & name, uint32_t expect_type) {
    const gguf_tensor_info * ti = f.find(name);
    if (!ti) {
        throw std::runtime_error("missing tensor: " + name);
    }
    if (expect_type != 0xFFFFFFFF && ti->type != expect_type) {
        throw std::runtime_error("unexpected type for " + name);
    }
    wt t;
    t.data = ti->data;
    t.type = ti->type;
    t.K = (int32_t)ti->dims[0];
    t.N = (int32_t)ti->n_rows();
    return t;
}

const float * bind_f32(const gguf_file & f, const std::string & name) {
    const gguf_tensor_info * ti = f.find(name);
    if (!ti) {
        throw std::runtime_error("missing tensor: " + name);
    }
    if (ti->type != GGML_TYPE_F32) {
        throw std::runtime_error("expected f32 for " + name);
    }
    return (const float *)ti->data;
}

} // namespace arch

namespace {

const char * prefill_kind_name(model_profile::prefill_kind k) {
    return k == model_profile::prefill_kind::whole_prompt ? "whole_prompt" : "chunk32";
}

const char * moe_owner_name(model_profile::moe_owner m) {
    switch (m) {
    case model_profile::moe_owner::runtime_bank:
        return "runtime_bank";
    case model_profile::moe_owner::host_only:
        return "host_only";
    default:
        return "engine_cache";
    }
}

const char * yn(bool v) {
    return v ? "yes" : "no";
}

// PF_MODEL_INFO: the resolved profile and its provenance, in the same spirit as
// PF_DEVICE_INFO.  Printed by model::load, so it reports what the loader really
// selected rather than what a name suggests.
void report(const model_profile & p) {
    if (!si::env::flag("PF_MODEL_INFO")) {
        return;
    }
    fprintf(stderr,
            "[model] profile: %s (%s)\n"
            "[model]   forward: %s, moe owner: %s\n"
            "[model]   residual: hyper-connections %s (%d streams, low rank %d)\n"
            "[model]   extras: ple %s, qsa %s (top_k %d, pool %d, idx %dx%d)\n"
            "[model]   policy: max_decode_rows %d, prefill %s, prefix_cache %s, auto_layer_map %s, "
            "md_splits %s, engram %zu MiB\n"
            "[model]   weights: sin_w8 %s, dnnl_int8 %s, raw_gemv_decode %s, split_per_tensor_upload %s\n"
            "[model]   capabilities: mtp %s, dflash %s, multimodal %s%s%s%s\n"
            "[model]   provenance: %s\n",
            p.key, p.name, fwd_kind_name(p.fwd), moe_owner_name(p.moe), yn(p.residual.hyper),
            p.residual.streams, p.residual.low_rank, yn(p.extras.ple), yn(p.extras.qsa), p.extras.qsa_top_k,
            p.extras.qsa_pool, p.extras.idx_heads, p.extras.idx_head_dim, p.policy.max_decode_rows,
            prefill_kind_name(p.policy.prefill), yn(p.policy.prefix_cache), yn(p.policy.auto_layer_map),
            yn(p.policy.md_splits), p.policy.engram_ram_mb_default, yn(p.rep.sin_w8), yn(p.rep.dnnl_int8),
            yn(p.rep.raw_gemv_decode), yn(p.rep.split_per_tensor_upload), yn(p.spec.mtp), yn(p.spec.dflash),
            yn(p.spec.multimodal), (p.spec.why && *p.spec.why) ? " (" : "", (p.spec.why && *p.spec.why) ? p.spec.why : "",
            (p.spec.why && *p.spec.why) ? ")" : "", p.provenance);
}

} // namespace

// Geometry the kernels hardcode.  A model that does not match these is not
// "slower" or "approximate" - several kernels would read or write out of
// bounds, so the only correct behaviour is to refuse the file at load time.
// Called from model::load for every architecture, so a new loader inherits it.
void validate_hparams(const hparams & hp, const std::string & arch) {
    const std::string p = arch + ": ";
    // The classic attention kernel maps one 32-lane sub-group over the head
    // (`lane + 32*i`, i < head_dim/32), so any multiple of 32 is exact.  The
    // specialised kernels (oneDNN attention, flash, grouped decode) are
    // guarded on head_dim == 256 and fall back to the classic one, which is
    // what Qwen2-VL's 64-wide heads take.
    if (hp.head_dim < 32 || hp.head_dim % 32 != 0) {
        throw std::runtime_error(p + "head_dim " + std::to_string(hp.head_dim) +
                                 " must be a positive multiple of 32 (the attention sub-group width)");
    }
    if (hp.head_dim > 256 && !(hp.arch && hp.arch->wide_head_ok())) {
        // partials are `2 + head_dim` per (token, head, split); the allocation
        // follows head_dim, but the classic kernel's register array is sized
        // for 256 and the KV layouts assume at most one 32-lane block per byte.
        // An architecture with a wider head declares it through wide_head_ok().
        throw std::runtime_error(p + "head_dim " + std::to_string(hp.head_dim) + " exceeds the 256 ceiling");
    }
    // Architecture-specific geometry (ds41's MLA latent, ...) is declared by the
    // architecture's arch_hparams so this generic check never names it.
    if (hp.arch) {
        hp.arch->validate(hp, p);
    }
    if (hp.n_head <= 0 || hp.n_head_kv <= 0 || hp.n_head % hp.n_head_kv != 0) {
        // GQA is expanded as kvh = h * n_head_kv / n_head and the grouped/flash
        // kernels divide by n_head / n_head_kv.
        throw std::runtime_error(p + "unsupported head geometry: n_head=" + std::to_string(hp.n_head) +
                                 " n_head_kv=" + std::to_string(hp.n_head_kv) +
                                 " (need n_head % n_head_kv == 0)");
    }
    if (hp.n_layer <= 0 || hp.n_embd <= 0 || hp.n_vocab <= 0) {
        throw std::runtime_error(p + "incomplete hparams (n_layer=" + std::to_string(hp.n_layer) +
                                 " n_embd=" + std::to_string(hp.n_embd) +
                                 " n_vocab=" + std::to_string(hp.n_vocab) + ")");
    }
    if (hp.full_attn_interval <= 0) {
        throw std::runtime_error(p + "full_attn_interval must be positive (is_recr divides by it)");
    }
    // Every layer that is not full attention is a GDN block, and its geometry
    // comes from d_state / n_group / dt_rank (see hparams::qkv_dim).  A layer
    // with a zero extent would size the conv and the recurrence to nothing.
    if (hp.is_recr(0)) {
        if (hp.d_state <= 0 || hp.n_group <= 0 || hp.dt_rank <= 0 || hp.conv_k <= 0) {
            throw std::runtime_error(p + "recurrent layers need d_state/n_group/dt_rank/conv_k > 0 (got " +
                                     std::to_string(hp.d_state) + "/" + std::to_string(hp.n_group) + "/" +
                                     std::to_string(hp.dt_rank) + "/" + std::to_string(hp.conv_k) + ")");
        }
    }
    // MoE: the routed experts' K is n_embd and their N is n_ff_exp; the shared
    // expert's width is n_ff (it is bound into the dense FFN slots, so the
    // dense paths above already constrain it).  n_expert_used is bounded by
    // kMaxT because the routed intermediate is materialised as
    // [n_tokens][n_expert_used][2*n_ff_exp] and n_tokens is at most
    // kMaxB*kMaxT wide with every slot live.
    if (hp.is_moe()) {
        if (hp.n_expert_used <= 0 || hp.n_expert_used > hp.n_expert) {
            throw std::runtime_error(p + "n_expert_used " + std::to_string(hp.n_expert_used) +
                                     " outside [1, n_expert=" + std::to_string(hp.n_expert) + "]");
        }
        if (hp.n_ff_exp <= 0) {
            throw std::runtime_error(p + "MoE layers need expert_feed_forward_length > 0");
        }
        if (hp.n_expert > 4096) {
            // the router's per-token top-k runs in one work-group over the
            // expert logits, so the id/weight tables are bounded; beyond this a
            // model would need a multi-block top-k.
            throw std::runtime_error(p + "n_expert " + std::to_string(hp.n_expert) + " exceeds 4096");
        }
    }
}

// ---------------------------------------------------------------------------
// Generic model lifecycle.  Architecture-specific hparams and tensor bindings
// are delegated to the loader selected by `general.architecture`.
void validate_model_profile(const hparams & hp, const model_profile & prof, bool split_file) {
    // The model-owned forward's geometry is declared in the profile and read
    // from the GGUF metadata by the loader.  They are two views of the same
    // facts, so refuse a file whose geometry contradicts the architecture's
    // declaration instead of letting one view silently win at a use site.  The
    // check itself is the architecture's (arch_hparams::validate_profile), so
    // this generic function never names an architecture.
    if (hp.arch) {
        hp.arch->validate_profile(hp, prof);
    }
    // A split GGUF cannot use the single-blob upload; the profile has to
    // declare the per-tensor path, otherwise the split check in
    // weight_store::upload() is a surprise at the first layer-map-less run.
    if (split_file && !prof.rep.split_per_tensor_upload) {
        throw std::runtime_error("model profile " + std::string(prof.key) +
                                 ": the file is split but the profile does not declare the per-tensor upload");
    }
}

void model::load(const std::string & path) {
    gguf.load(path);
    if (const std::string * ct = gguf.get_str("tokenizer.chat_template")) {
        chat_template = *ct;
    }
    const std::string * arch_p = gguf.get_str("general.architecture");
    const std::string name = arch_p ? *arch_p : "";
    const arch::loader * a = arch::find(name);
    if (!a) {
        throw std::runtime_error("unsupported architecture: " + name);
    }
    if (!a->prof) {
        throw std::runtime_error("architecture " + name + " has no model profile");
    }
    if (name != a->prof->key) {
        // Exact-key selection, no fuzzy match: a profile is a policy, and the
        // wrong one would silently enable/disable engine mechanisms.
        throw std::runtime_error("model profile key mismatch: registry row " + name + " carries profile \"" +
                                 a->prof->key + "\"");
    }
    prof = a->prof;
    a->load(*this);
    validate_hparams(hp, name);
    validate_model_profile(hp, *prof, gguf.is_split());
    report(*prof);
}

int model_context_length(const std::string & path) {
    gguf_file f;
    f.load(path);
    const std::string * arch = f.get_str("general.architecture");
    if (!arch) {
        return 0;
    }
    return (int)f.get_u32(*arch + ".context_length", 0);
}

// MADV_DONTNEED over the page-aligned range covering [p, p+len) clipped to the
// mapping.  Only *resident* pages drop out of the process RSS; the mapping and
// its file contents are untouched, so this is safe even if a host reader shows
// up later (it just faults the page back in).
void model::page_out_host(const void * p, size_t len) {
    if (!gguf.map_base || !p || len == 0) {
        return;
    }
    const uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE);
    const uintptr_t base = (uintptr_t)gguf.map_base;
    const uintptr_t end = base + gguf.map_size;
    uintptr_t a = (uintptr_t)p & ~(pg - 1);
    uintptr_t b = ((uintptr_t)p + len + pg - 1) & ~(pg - 1);
    if (a < base) {
        a = base;
    }
    if (b > end) {
        b = end;
    }
    if (b > a) {
        madvise((void *)a, (size_t)(b - a), MADV_DONTNEED);
        // Now that the PTEs are gone, let the kernel drop the page cache too so
        // the memory is actually freed rather than just unmapped from the
        // process (a still-mapped CPU-partition range is skipped by the kernel).
        // keep_file_cache trades that RSS for a warm restart: the pages stay in
        // the page cache and the next process maps them without disk I/O.
        if (!keep_file_cache) {
            gguf.drop_cache((size_t)(a - base), (size_t)(b - a));
        }
    }
}

} // namespace si
