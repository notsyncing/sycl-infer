// Hermetic test for the architecture dispatch table (no GPU, no model).
//
// The table is the single seam that keeps generic engine/model code from
// naming an architecture: a model-owned forward is a row of function pointers
// and capability flags, and `arch_ops_for(generic_plan) == nullptr` is what
// selects the shared seg_plan path.  This pins the rows' shape so a hook
// cannot silently disappear from one architecture's entry.  It never calls a
// function pointer - only checks that it is non-null or that a flag is set.
#include <cstdio>

#include "arch/arch_ops.h"

using namespace si;

static int g_fail = 0;

#define CHECK(cond, what)                                                                                              \
    do {                                                                                                               \
        if (cond) {                                                                                                    \
            printf("  ok       %s\n", what);                                                                           \
        } else {                                                                                                       \
            fprintf(stderr, "FAIL %s\n", what);                                                                        \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

int main() {
    // The shared generic plan has no row: null is the selector.
    CHECK(arch_ops_for(model_profile::fwd_kind::generic_plan) == nullptr,
          "generic_plan has no arch_ops row");

    // ---- qwen4exp (hc_ple_qsa) --------------------------------------------
    const arch_ops * q4 = arch_ops_for(model_profile::fwd_kind::hc_ple_qsa);
    CHECK(q4 != nullptr, "hc_ple_qsa row exists");
    if (q4) {
        CHECK(q4->kind == model_profile::fwd_kind::hc_ple_qsa, "hc_ple_qsa row kind");
        CHECK(q4->device_argmax, "hc_ple_qsa device_argmax");
        CHECK(q4->configure != nullptr, "hc_ple_qsa configure");
        CHECK(q4->destroy != nullptr, "hc_ple_qsa destroy");
        CHECK(q4->reset_extra != nullptr, "hc_ple_qsa reset_extra");
        CHECK(q4->setup != nullptr, "hc_ple_qsa setup");
        CHECK(q4->prefault != nullptr, "hc_ple_qsa prefault");
        CHECK(q4->slot_zero != nullptr, "hc_ple_qsa slot_zero");
        CHECK(!q4->host_keep_routed_experts, "hc_ple_qsa does not keep routed expert pages");
        // q4_destroy runs the routing diagnostic itself, so the generic
        // post-destroy hist_report call must stay null (the no-double-print
        // invariant), and q4 has no architecture-owned dense stores.
        CHECK(q4->hist_report == nullptr, "hc_ple_qsa has no post-destroy hist_report");
        CHECK(q4->dense_store == nullptr, "hc_ple_qsa has no dense_store hook");
        CHECK(q4->prefill_text != nullptr, "hc_ple_qsa prefill_text");
        CHECK(q4->prefill_mm != nullptr, "hc_ple_qsa prefill_mm");
        CHECK(q4->decode_batch != nullptr, "hc_ple_qsa decode_batch");
        CHECK(q4->run_head != nullptr, "hc_ple_qsa run_head");
        CHECK(q4->pc_extra_floats != nullptr, "hc_ple_qsa pc_extra_floats");
        CHECK(q4->pc_serialize != nullptr, "hc_ple_qsa pc_serialize");
        CHECK(q4->pc_deserialize != nullptr, "hc_ple_qsa pc_deserialize");
        CHECK(q4->pc_restore != nullptr, "hc_ple_qsa pc_restore");
        CHECK(q4->pc_clear != nullptr, "hc_ple_qsa pc_clear");
        CHECK(q4->pc_resume_prefixes != nullptr, "hc_ple_qsa pc_resume_prefixes");
        CHECK(q4->pc_block_aligned_capture != nullptr, "hc_ple_qsa pc_block_aligned_capture");
    }

    // ---- deepseek41 -------------------------------------------------------
    const arch_ops * ds = arch_ops_for(model_profile::fwd_kind::deepseek41);
    CHECK(ds != nullptr, "deepseek41 row exists");
    if (ds) {
        CHECK(ds->kind == model_profile::fwd_kind::deepseek41, "deepseek41 row kind");
        CHECK(ds->owns_kv_pool, "deepseek41 owns_kv_pool");
        CHECK(ds->owns_request_path, "deepseek41 owns_request_path");
        CHECK(ds->host_keep_routed_experts, "deepseek41 keeps routed expert pages host-resident");
        CHECK(ds->dense_store != nullptr, "deepseek41 has a dense_store hook");
        CHECK(!ds->md_dnnl, "deepseek41 declines the generic md oneDNN partition");
        CHECK(ds->generate != nullptr, "deepseek41 generate");
        CHECK(ds->setup != nullptr, "deepseek41 setup");
        CHECK(ds->setup_weights != nullptr, "deepseek41 setup_weights");
        CHECK(ds->destroy != nullptr, "deepseek41 destroy");
    }

    if (g_fail) {
        fprintf(stderr, "test_arch_ops: %d failure(s)\n", g_fail);
        return 1;
    }
    printf("test_arch_ops: all checks OK\n");
    return 0;
}
