// The architecture dispatch table.  One row per model-owned forward; the
// shared generic plan has no entry, so `arch_ops_for` returning null is what
// selects build_graphs()/record_forward for every generic architecture.
#include "arch/arch_ops.h"
#include "arch/registry_internal.h"

namespace si {

namespace {

const arch_ops kArchOps[] = {
    {
        .kind = model_profile::fwd_kind::hc_ple_qsa,
        .owns_kv_pool = false,
        .owns_request_path = false,
        .md_dnnl = true,
        .device_argmax = true,
        .host_keep_routed_experts = false,
        .dense_store = nullptr,
        .configure = &arch_ops_registry::q4_configure,
        .setup = &arch_ops_registry::q4_setup,
        .prefault = &arch_ops_registry::q4_prefault,
        .setup_weights = nullptr,
        .destroy = &arch_ops_registry::q4_destroy,
        .reset_extra = &arch_ops_registry::q4_reset_extra,
        .slot_zero = &arch_ops_registry::q4_slot_zero,
        // q4_destroy runs the routing diagnostic itself, before the engram
        // print and the frees, so the historical print order is preserved; the
        // generic post-destroy hist_report call must not print it a second time.
        .hist_report = nullptr,
        .generate = nullptr,
        .prefill_text = &arch_ops_registry::q4_prefill_text,
        .prefill_mm = &arch_ops_registry::q4_prefill_mm,
        .decode_batch = &arch_ops_registry::q4_decode_batch,
        .run_head = &arch_ops_registry::q4_run_head,
        .pc_extra_floats = &arch_ops_registry::q4_pc_extra_floats,
        .pc_serialize = &arch_ops_registry::q4_pc_serialize,
        .pc_deserialize = &arch_ops_registry::q4_pc_deserialize,
        .pc_restore = &arch_ops_registry::q4_pc_restore,
        .pc_clear = &arch_ops_registry::q4_pc_clear,
        .pc_resume_prefixes = &arch_ops_registry::q4_pc_resume_prefixes,
        .pc_block_aligned_capture = &arch_ops_registry::q4_pc_block_aligned_capture,
    },
    {
        .kind = model_profile::fwd_kind::deepseek41,
        .owns_kv_pool = true,
        .owns_request_path = true,
        .md_dnnl = false,
        .device_argmax = false,
        .host_keep_routed_experts = true,
        .dense_store = &arch_ops_registry::ds41_dense_store,
        .configure = nullptr,
        .setup = &arch_ops_registry::ds41_setup,
        .prefault = nullptr,
        .setup_weights = &arch_ops_registry::ds41_setup_weights,
        .destroy = &arch_ops_registry::ds41_destroy,
        .reset_extra = nullptr,
        .slot_zero = nullptr,
        .hist_report = nullptr,
        .generate = &arch_ops_registry::ds41_generate,
        .prefill_text = nullptr,
        .prefill_mm = nullptr,
        .decode_batch = nullptr,
        .run_head = nullptr,
        .pc_extra_floats = nullptr,
        .pc_serialize = nullptr,
        .pc_deserialize = nullptr,
        .pc_restore = nullptr,
        .pc_clear = nullptr,
        .pc_resume_prefixes = nullptr,
        .pc_block_aligned_capture = nullptr,
    },
};

} // namespace

const arch_ops * arch_ops_for(model_profile::fwd_kind kind) {
    if (kind == model_profile::fwd_kind::generic_plan) {
        return nullptr;
    }
    for (const arch_ops & ops : kArchOps) {
        if (ops.kind == kind) {
            return &ops;
        }
    }
    return nullptr;
}

} // namespace si
