#include "model.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "common/env.h"
#include "model_arch.h"
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

// Supported architectures.  Add one loader per model type; the loaders live in
// their own translation units (e.g. qwen35.cpp).  Every row carries its
// declarative profile, and model::load asserts the profile's key is the row's
// exact general.architecture value: a row without a matching profile is a load
// error, not a silent fallback to another architecture's policy.
static const loader kLoaders[] = {
    {"qwen35", &load_qwen35, &kQwen35Profile},
    {"qwen35moe", &load_qwen35moe, &kQwen35MoeProfile},
    {"qwen4exp", &load_qwen4exp, &kQwen4ExpProfile},
};

const loader * find(const std::string & name) {
    for (const loader & l : kLoaders) {
        if (name == l.name) {
            return &l;
        }
    }
    return nullptr;
}

const loader * registry(size_t & n) {
    n = sizeof(kLoaders) / sizeof(kLoaders[0]);
    return kLoaders;
}

} // namespace arch

namespace {

const char * fwd_kind_name(model_profile::fwd_kind k) {
    return k == model_profile::fwd_kind::hc_ple_qsa ? "hc_ple_qsa" : "generic_plan";
}

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
            "[model]   capabilities: mtp %s, dflash %s, multimodal %s%s%s\n"
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
    // attn.cpp's generic (non-specialised) attention kernel sizes its per-split
    // partials with a *literal* `constexpr int HD = 256`, not with head_dim, so
    // any other head_dim makes it write 2 + 256 floats into a stride of
    // 2 + head_dim and overrun the partials buffer.  The special-cased kernels
    // (oneDNN attention, flash, grouped) all guard on head_dim == HD and simply
    // fall through to that same generic one, so there is no supported shape
    // other than 256.
    if (hp.head_dim != 256) {
        throw std::runtime_error(p + "unsupported head_dim " + std::to_string(hp.head_dim) +
                                 " (the attention kernels only support 256)");
    }
    if (hp.head_dim % 32 != 0) {
        // qk_norm_rope's RMSNorm loop is `head_dim / 32` over a 32-lane
        // sub-group; a non-multiple silently leaves the tail unnormalized.
        throw std::runtime_error(p + "head_dim must be a multiple of 32 (the QK-norm sub-group width)");
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
    // M0: m.q4exp (and hparams::q4exp) must agree with the declarative forward
    // kind.  The field stays until the forward is delegated to a runtime object,
    // so this catches a registry/profile/loader drift instead of letting the
    // engine take the generic path on a hyper-connection model (or vice versa).
    const bool hc_fwd = prof->fwd == model_profile::fwd_kind::hc_ple_qsa;
    if (m.q4exp != hc_fwd || hp.q4exp != hc_fwd) {
        throw std::runtime_error("model profile " + std::string(prof->key) +
                                 ": forward kind disagrees with the loader (m.q4exp=" +
                                 (m.q4exp ? "true" : "false") + ", hp.q4exp=" + (hp.q4exp ? "true" : "false") + ")");
    }
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

void model::upload(sycl::queue & q, bool host) {
    if (host) {
        dev_weights = nullptr;
        dev_weights_size = 0;
        return;
    }
    // A split GGUF is mapped as several files inside one virtual reservation;
    // copying the whole reservation would upload its padding and needs the
    // entire model in device memory.  Large split models take the per-tensor
    // upload path instead (the multi-device/layer-map one).
    if (gguf.is_split()) {
        throw std::runtime_error("single-blob upload is not supported for a split GGUF: use --layer-map");
    }
    dev_weights_size = gguf.map_size;
    dev_weights = sycl::malloc_device(dev_weights_size, q);
    if (!dev_weights) {
        throw std::runtime_error("device allocation failed");
    }
    q.memcpy(dev_weights, gguf.map_base, dev_weights_size).wait();
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
        // now that the PTEs are gone, let the kernel drop the page cache too so
        // the memory is actually freed rather than just unmapped from the
        // process (a still-mapped CPU-partition range is skipped by the kernel)
        gguf.drop_cache((size_t)(a - base), (size_t)(b - a));
    }
}

} // namespace si
