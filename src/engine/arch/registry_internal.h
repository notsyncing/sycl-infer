#pragma once
#include "arch/arch_ops.h"

namespace si {

// The per-architecture runtime implementation registry: the static hooks the
// kArchOps[] rows in registry.cpp bind, implemented one TU per model-owned
// forward in engine/arch/<name>/runtime.cpp.  This is the *internal* half of
// the arch seam and is deliberately not part of the public interface: only
// registry.cpp and those runtime TUs may include it (ci/check_arch_boundaries.py
// enforces that).  arch_ops.h keeps only `struct arch_ops` + `arch_ops_for` and
// a forward declaration of this type, so the public seam names no architecture.
struct arch_ops_registry {
    static void q4_configure(engine &, const engine_config &);
    static void q4_destroy(engine &);
    static void q4_reset_extra(engine &);
    static void q4_setup(engine &);
    static void q4_prefault(engine &);
    static void q4_slot_zero(engine &, int);
    static void q4_hist_report(engine &);
    static void q4_prefill_text(engine &, const std::vector<int> &, int, int, int);
    static void q4_prefill_mm(engine &, const mm_prompt &, int);
    static void q4_decode_batch(engine &, const int32_t *, const int32_t *, const int32_t *, int);
    static std::vector<float> q4_run_head(engine &);
    static size_t q4_pc_extra_floats(engine &, int);
    static void q4_pc_serialize(engine &, float *, int, int);
    static void q4_pc_deserialize(engine &, const float *, int, int);
    static void q4_pc_restore(engine &, int, int, int);
    static void q4_pc_clear(engine &, int);
    static void q4_pc_resume_prefixes(engine &, std::vector<int> &, int);
    static bool q4_pc_block_aligned_capture(engine &);
    static void ds41_setup(engine &);
    static void ds41_setup_weights(engine &, bool has_gpu);
    static bool ds41_dense_store(const engine &);
    static void ds41_destroy(engine &);
    static std::vector<int> ds41_generate(engine &, const std::vector<int> &, const mm_prompt *, const gen_params &,
                                          const std::function<bool(int)> &, std::vector<float> *, size_t *);
};

} // namespace si
