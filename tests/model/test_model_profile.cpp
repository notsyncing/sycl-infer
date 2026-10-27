// Model profile registry and values.  Hermetic: no GGUF, no GPU.
//
// Two things are asserted.  First, the loader registry and the profile table
// cannot drift: every `kLoaders` row must carry a non-null profile whose key is
// exactly the row's `general.architecture`, because model::load selects the
// policy by that key and asserts the same thing.  Second, the qwen4exp values the
// engine's policy branches now read are pinned here (one decode row, prefix cache
// off, HC 4/320, QSA top_k 2048, raw-GGUF decode, runtime-owned expert bank), so
// a refactor cannot silently turn one of them back into a generic default.
//
// The generic qwen35 / qwen35moe rows are also compared field by field: they are
// separate consts (the exact-key assert forbids a shared "generic" key) but must
// keep identical policy, or a change to one silently changes only one model.
#include <cstdio>
#include <cstring>
#include <string>

#include "model.h"
#include "model_arch.h"
#include "model_profile.h"

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

static bool same_policy(const model_profile & a, const model_profile & b) {
    return a.policy.max_decode_rows == b.policy.max_decode_rows && a.policy.prefill == b.policy.prefill
           && a.policy.prefix_cache == b.policy.prefix_cache && a.policy.auto_layer_map == b.policy.auto_layer_map
           && a.policy.md_splits == b.policy.md_splits
           && a.policy.engram_ram_mb_default == b.policy.engram_ram_mb_default;
}

static bool same_rep(const model_profile & a, const model_profile & b) {
    return a.rep.sin_w8 == b.rep.sin_w8 && a.rep.dnnl_int8 == b.rep.dnnl_int8
           && a.rep.raw_gemv_decode == b.rep.raw_gemv_decode
           && a.rep.split_per_tensor_upload == b.rep.split_per_tensor_upload;
}

static bool same_spec(const model_profile & a, const model_profile & b) {
    return a.spec.mtp == b.spec.mtp && a.spec.dflash == b.spec.dflash && a.spec.multimodal == b.spec.multimodal
           && std::strcmp(a.spec.why, b.spec.why) == 0;
}

static bool same_extras(const model_profile & a, const model_profile & b) {
    return a.extras.ple == b.extras.ple && a.extras.qsa == b.extras.qsa && a.extras.qsa_top_k == b.extras.qsa_top_k
           && a.extras.qsa_pool == b.extras.qsa_pool && a.extras.idx_heads == b.extras.idx_heads
           && a.extras.idx_head_dim == b.extras.idx_head_dim;
}

int main() {
    printf("model profile registry\n");
    size_t n = 0;
    const arch::loader * rows = arch::registry(n);
    CHECK(rows != nullptr && n >= 3u, "the loader registry is present");
    for (size_t i = 0; i < n; i++) {
        const arch::loader & row = rows[i];
        const std::string what = "registry row " + std::string(row.name ? row.name : "(null)");
        CHECK(row.name != nullptr && *row.name != '\0', (what + " has a name").c_str());
        CHECK(row.load != nullptr, (what + " has a loader").c_str());
        CHECK(row.prof != nullptr, (what + " has a profile").c_str());
        if (!row.prof) {
            continue;
        }
        CHECK(std::strcmp(row.prof->key, row.name) == 0, (what + " profile key equals the architecture key").c_str());
        CHECK(row.prof->name != nullptr && *row.prof->name != '\0', (what + " profile has a human name").c_str());
        CHECK(row.prof->provenance != nullptr && *row.prof->provenance != '\0',
              (what + " profile states its provenance").c_str());
        CHECK(arch::find(row.name) == &row, (what + " is what arch::find returns").c_str());
        CHECK(row.prof->policy.max_decode_rows >= 0, (what + " has a non-negative decode row cap").c_str());
        if (row.prof->residual.hyper) {
            CHECK(row.prof->residual.streams > 0 && row.prof->residual.low_rank > 0,
                  (what + " hyper-connection geometry is positive").c_str());
        }
        if (row.prof->extras.qsa) {
            CHECK(row.prof->extras.qsa_top_k > 0 && row.prof->extras.idx_heads > 0
                      && row.prof->extras.idx_head_dim > 0,
                  (what + " QSA geometry is positive").c_str());
        }
        // A profile that declares a mechanism unsupported says why; a supported
        // one need not, but a null string would crash the report either way.
        CHECK(row.prof->spec.why != nullptr, (what + " capability reason is a string").c_str());
    }

    printf("qwen4exp profile values\n");
    const model_profile & q4 = kQwen4ExpProfile;
    CHECK(q4.fwd == model_profile::fwd_kind::hc_ple_qsa, "forward kind is hc_ple_qsa");
    CHECK(q4.policy.max_decode_rows == 1, "max_decode_rows is 1 (2 rows measured 0.85x)");
    CHECK(!q4.policy.prefix_cache, "prefix cache is off (PLE/QSA state is not in the blob)");
    CHECK(q4.policy.prefill == model_profile::prefill_kind::whole_prompt, "prefill is whole_prompt");
    CHECK(q4.policy.auto_layer_map, "auto layer map is on (split GGUF needs per-tensor upload)");
    CHECK(q4.policy.md_splits, "md_splits keeps the attention K-split");
    CHECK(q4.policy.engram_ram_mb_default == 4096u, "engram default is 4096 MiB");
    CHECK(q4.residual.hyper && q4.residual.streams == 4 && q4.residual.low_rank == 320,
          "hyper-connections are 4 streams / low_rank 320");
    CHECK(q4.extras.ple, "PLE is declared");
    CHECK(q4.extras.qsa && q4.extras.qsa_top_k == 2048, "QSA top_k is 2048");
    CHECK(q4.extras.qsa_pool == 4, "QSA pool (compress ratio) is 4");
    CHECK(q4.extras.idx_heads == 4 && q4.extras.idx_head_dim == 128, "indexer is 4 heads x 128");
    CHECK(q4.rep.raw_gemv_decode && !q4.rep.sin_w8 && !q4.rep.dnnl_int8,
          "weights are raw GGUF, not SIn/oneDNN int8");
    CHECK(q4.rep.split_per_tensor_upload, "split GGUF uses the per-tensor upload");
    CHECK(q4.moe == model_profile::moe_owner::runtime_bank, "routed experts are owned by the runtime bank");
    CHECK(!q4.spec.mtp && !q4.spec.dflash && !q4.spec.multimodal, "MTP/DFlash/MM are declared unsupported");
    CHECK(std::strlen(q4.spec.why) > 0u, "the unsupported capabilities carry a reason");

    printf("generic profiles\n");
    const model_profile & g35 = kQwen35Profile;
    const model_profile & gmoe = kQwen35MoeProfile;
    CHECK(std::strcmp(g35.key, "qwen35") == 0, "qwen35 profile key");
    CHECK(std::strcmp(gmoe.key, "qwen35moe") == 0, "qwen35moe profile key");
    CHECK(g35.fwd == model_profile::fwd_kind::generic_plan && gmoe.fwd == model_profile::fwd_kind::generic_plan,
          "both generic profiles use the shared forward");
    CHECK(g35.policy.max_decode_rows == 0 && gmoe.policy.max_decode_rows == 0,
          "generic decode row cap defers to the engine ceiling");
    CHECK(g35.policy.prefix_cache && gmoe.policy.prefix_cache, "generic prefix cache is on");
    CHECK(g35.policy.prefill == model_profile::prefill_kind::chunk32
              && gmoe.policy.prefill == model_profile::prefill_kind::chunk32,
          "generic prefill is chunked");
    CHECK(same_policy(g35, gmoe), "qwen35/qwen35moe policies are identical");
    CHECK(same_rep(g35, gmoe), "qwen35/qwen35moe weight representations are identical");
    CHECK(same_spec(g35, gmoe), "qwen35/qwen35moe capabilities are identical");
    CHECK(same_extras(g35, gmoe), "qwen35/qwen35moe extras are identical");
    CHECK(g35.residual.hyper == gmoe.residual.hyper && g35.residual.streams == gmoe.residual.streams
              && g35.residual.low_rank == gmoe.residual.low_rank,
          "qwen35/qwen35moe residual geometry is identical");
    CHECK(g35.moe == model_profile::moe_owner::engine_cache
              && gmoe.moe == model_profile::moe_owner::engine_cache,
          "generic MoE goes through the engine cache");
    CHECK(g35.rep.sin_w8 && g35.rep.dnnl_int8 && !g35.rep.raw_gemv_decode && !g35.rep.split_per_tensor_upload,
          "generic weights use the SIn/oneDNN stores and the single-blob upload");
    CHECK(g35.spec.mtp && g35.spec.dflash && g35.spec.multimodal, "generic MTP/DFlash/MM are wired");

    // The loader's registry rows must point at these exact objects, so the
    // profile report cannot print one object while the engine uses another.
    {
        const arch::loader * a = arch::find("qwen4exp");
        CHECK(a != nullptr && a->prof == &kQwen4ExpProfile, "qwen4exp row points at kQwen4ExpProfile");
    }
    {
        const arch::loader * a = arch::find("qwen35");
        CHECK(a != nullptr && a->prof == &kQwen35Profile, "qwen35 row points at kQwen35Profile");
    }
    {
        const arch::loader * a = arch::find("qwen35moe");
        CHECK(a != nullptr && a->prof == &kQwen35MoeProfile, "qwen35moe row points at kQwen35MoeProfile");
    }

    // Be honest about what a profile is: a static declaration.  A model object
    // only gets one from model::load, so a consumer cannot read a stale pointer
    // from a default-constructed model.
    {
        model m;
        CHECK(m.prof == nullptr, "a default-constructed model has no profile");
    }

    if (g_fail) {
        fprintf(stderr, "test_model_profile: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_model_profile: all checks OK\n");
    return 0;
}
