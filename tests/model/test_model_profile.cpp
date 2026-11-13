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
#include "arch/deepseek41/load.h"
#include "arch/qwen2vl/load.h"
#include "arch/qwen35/load.h"
#include "arch/qwen35moe/load.h"
#include "arch/qwen4exp/hparams.h"
#include "arch/qwen4exp/load.h"
#include "arch/registry.h"
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
           && a.policy.auto_layer_map_one_gpu == b.policy.auto_layer_map_one_gpu
           && a.policy.md_splits == b.policy.md_splits
           && a.policy.engram_ram_mb_default == b.policy.engram_ram_mb_default
           && a.policy.keep_file_cache == b.policy.keep_file_cache;
}

static bool same_rep(const model_profile & a, const model_profile & b) {
    return a.rep.sin_w8 == b.rep.sin_w8 && a.rep.dnnl_int8 == b.rep.dnnl_int8
           && a.rep.raw_gemv_decode == b.rep.raw_gemv_decode && a.rep.f32_gemm == b.rep.f32_gemm
           && a.rep.split_per_tensor_upload == b.rep.split_per_tensor_upload;
}

static bool same_spec(const model_profile & a, const model_profile & b) {
    return a.spec.mtp == b.spec.mtp && a.spec.dflash == b.spec.dflash && a.spec.multimodal == b.spec.multimodal
           && a.spec.mm_owned_prefill == b.spec.mm_owned_prefill && std::strcmp(a.spec.why, b.spec.why) == 0;
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
        // The vision tower kind and the multimodal capability must agree: a
        // model that cannot consume media declares `none`, and every model that
        // can names the tower its mmproj must be.
        CHECK(row.prof->spec.multimodal == (row.prof->vision != model_profile::vision_kind::none),
              (what + " vision tower kind agrees with spec.multimodal").c_str());
    }

    printf("qwen4exp profile values\n");
    const model_profile & q4 = qwen4exp_profile();
    CHECK(q4.fwd == model_profile::fwd_kind::hc_ple_qsa, "forward kind is hc_ple_qsa");
    CHECK(q4.policy.max_decode_rows == 1, "max_decode_rows is 1 (2 rows measured 0.85x)");
    CHECK(q4.policy.prefix_cache, "prefix cache is on (PLE/QSA state is checkpointed)");
    CHECK(q4.policy.prefill == model_profile::prefill_kind::whole_prompt, "prefill is whole_prompt");
    CHECK(q4.policy.auto_layer_map, "auto layer map is on (split GGUF needs per-tensor upload)");
    CHECK(q4.policy.md_splits, "md_splits keeps the attention K-split");
    CHECK(q4.policy.engram_ram_mb_default == 4096u, "engram default is 4096 MiB");
    CHECK(q4.policy.keep_file_cache, "shard page cache is kept across page_out_host");
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
    CHECK(q4.vision == model_profile::vision_kind::qwen3vl,
          "qwen4exp uses the qwen3vl mmproj tower");
    CHECK(!q4.spec.mtp && !q4.spec.dflash && q4.spec.multimodal, "MTP/DFlash off, multimodal on");
    CHECK(q4.spec.mm_owned_prefill, "qwen4exp owns the multimodal prefill chunker");
    CHECK(std::strlen(q4.spec.why) > 0u, "the unsupported capabilities carry a reason");

    printf("generic profiles\n");
    const model_profile & g35 = qwen35_profile();
    const model_profile & gmoe = qwen35moe_profile();
    CHECK(std::strcmp(g35.key, "qwen35") == 0, "qwen35 profile key");
    CHECK(std::strcmp(gmoe.key, "qwen35moe") == 0, "qwen35moe profile key");
    CHECK(g35.fwd == model_profile::fwd_kind::generic_plan && gmoe.fwd == model_profile::fwd_kind::generic_plan,
          "both generic profiles use the shared forward");
    CHECK(g35.policy.max_decode_rows == 0 && gmoe.policy.max_decode_rows == 0,
          "generic decode row cap defers to the engine ceiling");
    CHECK(g35.policy.prefix_cache && gmoe.policy.prefix_cache, "generic prefix cache is on");
    CHECK(!g35.policy.keep_file_cache && !gmoe.policy.keep_file_cache,
          "generic models evict the GGUF page cache (no second read)");
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
    CHECK(!g35.spec.mm_owned_prefill && !gmoe.spec.mm_owned_prefill,
          "generic profiles use the generic multimodal chunk loop");
    CHECK(g35.vision == model_profile::vision_kind::qwen3vl
              && gmoe.vision == model_profile::vision_kind::qwen3vl,
          "generic mmproj files use the qwen3vl tower");

    printf("qwen2vl profile values\n");
    const model_profile & q2 = qwen2vl_profile();
    CHECK(q2.fwd == model_profile::fwd_kind::generic_plan, "qwen2vl uses the shared forward");
    CHECK(q2.vision == model_profile::vision_kind::qwen2vl, "qwen2vl mmproj uses the qwen2vl tower");
    CHECK(q2.spec.multimodal && !q2.spec.mtp && !q2.spec.dflash, "qwen2vl declares MM only");
    CHECK(!q2.spec.mm_owned_prefill, "qwen2vl uses the generic multimodal chunk loop");
    CHECK(!q2.policy.prefix_cache && q2.policy.auto_layer_map && q2.policy.auto_layer_map_one_gpu,
          "qwen2vl takes the per-tensor upload and disables the prefix cache");

    printf("deepseek41 profile values\n");
    const model_profile & ds = deepseek41_profile();
    CHECK(ds.fwd == model_profile::fwd_kind::deepseek41, "deepseek41 uses its own forward");
    CHECK(ds.vision == model_profile::vision_kind::deepseek41v,
          "deepseek41 mmproj uses the deepseek41v tower");
    CHECK(ds.spec.multimodal && !ds.spec.mtp && !ds.spec.dflash, "deepseek41 declares MM only");
    CHECK(!ds.spec.mm_owned_prefill, "deepseek41 does not own a generic mm prefill chunker");
    CHECK(ds.policy.max_decode_rows == 1, "deepseek41 caps a decode at one row");
    CHECK(ds.policy.prefix_cache, "deepseek41 checkpoints its window + compressed state");
    CHECK(ds.policy.prefill == model_profile::prefill_kind::whole_prompt, "deepseek41 prefill is whole_prompt");
    CHECK(ds.policy.auto_layer_map && ds.policy.md_splits, "deepseek41 takes the per-tensor map and keeps splits");
    CHECK(ds.policy.engram_ram_mb_default == 8192u, "deepseek41 engram default is 8192 MiB");
    CHECK(ds.policy.keep_file_cache, "deepseek41 keeps the shard page cache");
    CHECK(ds.residual.hyper && ds.residual.streams == 4 && ds.residual.low_rank == 24,
          "deepseek41 mHC is 4 streams / 24 mix outputs");
    CHECK(ds.extras.ple, "deepseek41 Engram is declared");
    CHECK(ds.extras.qsa && ds.extras.qsa_top_k == 512, "deepseek41 DSA top_k is 512");
    CHECK(ds.extras.idx_heads == 32 && ds.extras.idx_head_dim == 128, "indexer is 32 heads x 128");
    CHECK(ds.rep.raw_gemv_decode && !ds.rep.sin_w8 && !ds.rep.dnnl_int8,
          "deepseek41 weights are raw GGUF (IQ2_XXS/IQ2_XS/IQ2_S are not SIn/oneDNN)");
    CHECK(ds.rep.split_per_tensor_upload, "deepseek41 split GGUF uses the per-tensor upload");
    CHECK(ds.moe == model_profile::moe_owner::runtime_bank, "deepseek41 routed experts are runtime-owned");

    // The loader's registry rows must point at these exact objects, so the
    // profile report cannot print one object while the engine uses another.
    {
        const arch::loader * a = arch::find("deepseek41");
        CHECK(a != nullptr && a->prof == &deepseek41_profile(), "deepseek41 row points at deepseek41_profile()");
    }
    {
        const arch::loader * a = arch::find("qwen4exp");
        CHECK(a != nullptr && a->prof == &qwen4exp_profile(), "qwen4exp row points at qwen4exp_profile()");
    }
    {
        const arch::loader * a = arch::find("qwen35");
        CHECK(a != nullptr && a->prof == &qwen35_profile(), "qwen35 row points at qwen35_profile()");
    }
    {
        const arch::loader * a = arch::find("qwen35moe");
        CHECK(a != nullptr && a->prof == &qwen35moe_profile(), "qwen35moe row points at qwen35moe_profile()");
    }
    {
        const arch::loader * a = arch::find("qwen2vl");
        CHECK(a != nullptr && a->prof == &qwen2vl_profile(), "qwen2vl row points at qwen2vl_profile()");
    }

    printf("profile geometry validation\n");
    {
        hparams hp{};
        auto qh = std::make_shared<qwen4exp_hparams>();
        hp.arch = qh;
        qh->hc_mult = 4;
        qh->hc_low_rank = 320;
        qh->idx_top_k = 2048;
        qh->idx_kpool = 4;
        qh->idx_n_head = 4;
        qh->idx_head_dim = 128;
        bool ok = true;
        try {
            validate_model_profile(hp, qwen4exp_profile(), /*split_file=*/true);
        } catch (const std::exception & e) {
            ok = false;
            fprintf(stderr, "FAIL matching qwen4exp geometry rejected: %s\n", e.what());
        }
        CHECK(ok, "matching qwen4exp geometry is accepted");

        qh->hc_mult = 3;
        bool threw = false;
        try {
            validate_model_profile(hp, qwen4exp_profile(), false);
        } catch (const std::exception &) {
            threw = true;
        }
        CHECK(threw, "a hyper-connection stream mismatch is rejected");

        hparams generic{};
        bool generic_ok = true;
        try {
            validate_model_profile(generic, qwen35_profile(), false);
        } catch (...) {
            generic_ok = false;
        }
        CHECK(generic_ok, "the generic plan has no geometry constraints");

        bool split_threw = false;
        try {
            validate_model_profile(generic, qwen35_profile(), /*split_file=*/true);
        } catch (const std::exception &) {
            split_threw = true;
        }
        CHECK(split_threw, "a split file without split_per_tensor_upload is rejected");
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
