#pragma once
// Generic weight PODs shared by the model header and the per-architecture
// tensor headers.  They live here (rather than in model.h) so an architecture's
// tensors.h can describe its own layer structs without including the whole
// model, which would otherwise form an include cycle.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace si {

struct hparams; // model.h
struct model_profile; // model_profile.h

// Per-layer extras of an architecture (qwen4exp's HC/PLE/QSA, deepseek41's
// MLA/mHC/compressor/indexer/Engram).  `layer_t::extras` is a pimpl slot: null
// for every generic architecture, one of these for a model-owned forward.  The
// generic weight-enumeration loops call `visit_weights`, so they never name an
// architecture's tensors.
struct arch_layer {
    enum class kind { none, qwen4exp, deepseek41 };
    kind k = kind::none;
    virtual ~arch_layer() = default;
    // Enumerate this layer's extra device weights (the generic upload/keep
    // loops call this).  `il` is the layer index, `hp` the model hparams (a
    // deepseek41 layer resolves its source arrays through `ds41_hp(hp)`).
    virtual void visit_weights(const hparams &hp, int il,
                               const std::function<void(const void *)> &cb) const {
        (void)hp;
        (void)il;
        (void)cb;
    }
};

// Model-level extras (the LM-head mixer and the host-only weight tables).  The
// device-upload set and the host-keep set differ: qwen4exp uploads the HC head
// but keeps the 26.8 GiB PLE table host-resident, and deepseek41 uploads none
// of these because its forward gathers the Engram table on the host.
struct arch_model_extras {
    enum class kind { none, qwen4exp, deepseek41 };
    kind k = kind::none;
    virtual ~arch_model_extras() = default;
    virtual void visit_device_weights(const std::function<void(const void *)> &cb) const {
        (void)cb;
    }
    // `cpu_partition` is true when the model's globals run on a CPU partition:
    // an architecture whose extra weights are GPU-uploaded but host-read by a
    // CPU backend (qwen4exp's HC head) keeps them only in that case.
    virtual void visit_host_keep(bool cpu_partition, const std::function<void(const void *)> &cb) const {
        (void)cpu_partition;
        (void)cb;
    }
};

// Base of an architecture's extra hyper-parameters.  The generic `hparams`
// carries only the facts every architecture shares; a model with its own
// geometry (qwen4exp's HC/PLE/QSA, deepseek41's MLA/mHC/Engram) attaches one
// of these through `hparams::arch`, and its own code reads it through the
// per-architecture inline accessor (`q4_hp`, `ds41_hp`).  `k` is the runtime
// tag the accessors assert, so a wrong-kind cast fails loudly instead of
// silently reading another architecture's layout.
struct arch_hparams {
    enum class kind { none, qwen4exp, deepseek41 };
    kind k = kind::none;
    virtual ~arch_hparams() = default;
    // A wide (>256) head has its own attention kernels (ds41's 512-wide MLA).
    // The generic geometry check refuses anything above 256 unless the
    // architecture opts in here.
    virtual bool wide_head_ok() const {
        return false;
    }
    // Architecture-specific geometry the generic kernels do not implement.
    // `prefix` is the loader's "<arch>: " message prefix.  Throws
    // std::runtime_error naming the first violated constraint.
    virtual void validate(const hparams & hp, const std::string & prefix) const {
        (void)hp;
        (void)prefix;
    }
    // Cross-check the loader's GGUF-derived geometry against the architecture's
    // declarative profile.  Throws std::runtime_error on a mismatch.
    virtual void validate_profile(const hparams & hp, const model_profile & prof) const {
        (void)hp;
        (void)prof;
    }
};

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

} // namespace si
