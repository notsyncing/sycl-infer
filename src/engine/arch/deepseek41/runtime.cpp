// DeepSeek-V4.1-Flash's arch_ops hooks (see arch/arch_ops.h).  These are the
// lifecycle and request-path decisions that used to live as an engine-level
// architecture branch in engine.cpp/server.cpp; the forward implementations
// themselves stay in arch/deepseek41/forward_host.cpp and forward_gpu.cpp.
#include <stdexcept>

#include "engine.h"
#include "arch/registry_internal.h"
#include "arch/deepseek41/forward.h"

namespace si {

void arch_ops_registry::ds41_setup(engine & e) {
    if (!e.cpu_mode) {
        // The GPU forward is the default for a ds41 run on a GPU; PF_DS41_GPU=0
        // keeps the validated host forward for an A/B.  The weights were
        // uploaded by setup_multi_device -> upload_device_weights above; this
        // allocates the ds41-owned caches, the expert bank and the scratch.
        const char * eg = si::env::str("PF_DS41_GPU");
        const bool gpu_default = !(eg && atoi(eg) == 0);
        if (gpu_default && !e.multi_dev) {
            fprintf(stderr, "[ds41] --device gpu without a layer map: keeping the host forward "
                            "(pass --layer-map 0-%d:gpu.0)\n", e.m.hp.n_layer - 1);
        } else if (gpu_default) {
            bool any_cpu = false;
            for (int d : e.dev_kind_) {
                any_cpu = any_cpu || d != 0;
            }
            if (any_cpu) {
                fprintf(stderr, "[ds41] the GPU forward needs an all-GPU --layer-map; "
                                "keeping the host forward\n");
            } else {
                e.ds41x_.gpu = ds41_gpu_create(e);
            }
        }
    }
}

void arch_ops_registry::ds41_setup_weights(engine & e, bool has_gpu) {
    e.ds41x_.dense_store = has_gpu && e.setup_ds41_dense_store();
}

bool arch_ops_registry::ds41_dense_store(const engine & e) {
    return e.ds41x_.dense_store;
}

void arch_ops_registry::ds41_destroy(engine & e) {
    if (e.ds41x_.gpu) {
        ds41_gpu_destroy(e.ds41x_.gpu);
        e.ds41x_.gpu = nullptr;
    }
}

std::vector<int> arch_ops_registry::ds41_generate(engine & e, const std::vector<int> & prompt, const mm_prompt * mm,
                                                   const gen_params & gp, const std::function<bool(int)> & cb,
                                                   std::vector<float> * first_logits, size_t * cached_tokens) {
    size_t cached = 0;
    std::vector<int> out;
    auto run_fwd = [&](const mm_prompt * mmp) {
        if (e.ds41x_.gpu) {
            return ds41_gpu_generate(e.m, e.tk, prompt, mmp, gp, cb, first_logits, &cached, *e.ds41x_.gpu);
        }
        return si::ds41_generate(e.m, e.tk, prompt, mmp, gp, cb, first_logits, &cached);
    };
    if (!mm) {
        out = run_fwd(nullptr);
    } else {
        if (!mm->has_encoded_embd()) {
            throw std::runtime_error("multimodal: the prompt has media rows but no encoder output "
                                     "(image/video need --mmproj)");
        }
        // The ds41 forward reads the image embeddings from host memory.  The
        // device-built prompt keeps them in `d_embd`; copy once here.
        mm_prompt local;
        const mm_prompt * mmp = mm;
        if (mm->embd.empty() && mm->d_embd && mm->n_img > 0) {
            local = *mm;
            // `n_img` counts media blocks, not embedding rows: an image
            // contributes many rows.  `n_rows` is the buffer's first dimension.
            const size_t rows = (size_t)(mm->n_rows > 0 ? mm->n_rows : (int)mm->img_row.size());
            local.embd.resize(rows * (size_t)e.m.hp.n_embd);
            e.q.memcpy(local.embd.data(), mm->d_embd, local.embd.size() * sizeof(float)).wait();
            local.d_embd = nullptr;
            mmp = &local;
        }
        out = run_fwd(mmp);
    }
    if (cached_tokens) {
        *cached_tokens = cached;
    }
    return out;
}

} // namespace si
