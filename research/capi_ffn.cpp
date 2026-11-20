// C ABI implementation of the research FFN capture API, kept under research/
// so it does not mix with the inference C API in src/capi/.  It lives in the
// same libsycl_infer shared object (the version script still exports only
// syi_*), but the source and the public header are research-only.
#include <sycl_infer/sycl_infer_research.h>

#include "capi_internal.h"
#include "ffn_capture.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

extern "C" {

void syi_ffn_options_init(syi_ffn_options *o) {
    if (!o) {
        return;
    }
    std::memset(o, 0, sizeof(*o));
    o->struct_size = sizeof(*o);
    o->abi_version = SYI_ABI_VERSION;
    o->layer = -1;
    o->token_slot = -1;
}

syi_status syi_model_capture_ffn(syi_model *model, const syi_ffn_options *options,
                                 const char *const *texts, size_t n_texts,
                                 float **out_data, size_t *out_n_texts,
                                 size_t *out_n_layers, size_t *out_n_ff) {
    if (!model || !model->st || !model->st->e || !texts || !out_data || !out_n_texts ||
        !out_n_layers || !out_n_ff) {
        return si::capi::fail(SYI_INVALID_ARGUMENT, "invalid capture_ffn arguments");
    }
    *out_data = nullptr;
    *out_n_texts = 0;
    *out_n_layers = 0;
    *out_n_ff = 0;
    if (!options || options->struct_size < sizeof(syi_ffn_options)) {
        return si::capi::fail(SYI_INVALID_ARGUMENT, "syi_ffn_options.struct_size is too small");
    }
    if (options->abi_version != 0 && options->abi_version != SYI_ABI_VERSION) {
        return si::capi::fail(SYI_UNSUPPORTED, "unsupported syi_ffn_options.abi_version");
    }
    if (n_texts == 0) {
        return SYI_OK;
    }

    try {
        si::engine *e = model->st->e.get();
        const int n_layer = e->m.hp.n_layer;
        const int n_ff = e->m.hp.n_ff;
        const int layer = options->layer;
        if (layer < -1 || layer >= n_layer) {
            return si::capi::fail(SYI_INVALID_ARGUMENT, "capture_ffn: layer out of range");
        }
        const int token_slot = options->token_slot;
        if (token_slot < -1) {
            return si::capi::fail(SYI_INVALID_ARGUMENT, "capture_ffn: token_slot out of range");
        }

        std::vector<std::vector<int> > prompts;
        prompts.reserve(n_texts);
        for (size_t i = 0; i < n_texts; ++i) {
            if (!texts[i]) {
                return si::capi::fail(SYI_INVALID_ARGUMENT, "capture_ffn: null text");
            }
            std::vector<int> ids = e->tk.encode(texts[i], /*parse_special=*/false);
            if (ids.empty()) {
                return si::capi::fail(SYI_INVALID_ARGUMENT, "capture_ffn: empty tokenization");
            }
            prompts.push_back(std::move(ids));
        }

        // token_slot == -1 resolves to each prompt's last real token; the
        // research helper takes an absolute slot, so use the longest prompt's
        // last index (all prompts are validated to fit one chunk there too).
        int abs_slot = token_slot;
        if (token_slot == -1) {
            size_t maxlen = 0;
            for (const auto & p : prompts) {
                maxlen = std::max(maxlen, p.size());
            }
            abs_slot = (int)maxlen - 1;
        }

        std::vector<float> out;
        if (si::research::capture_ffn_activations(*e, prompts, layer, abs_slot, out) != 0) {
            return si::capi::fail(SYI_INVALID_ARGUMENT,
                                  "capture_ffn: prompt exceeds the prefill chunk or bad slot");
        }

        const size_t n_layers = layer < 0 ? (size_t)n_layer : 1;
        const size_t expected = n_texts * n_layers * (size_t)2 * n_ff;
        if (out.size() != expected) {
            return si::capi::fail(SYI_INTERNAL, "capture_ffn: unexpected activation size");
        }
        float *buf = (float *)std::malloc(expected * sizeof(float));
        if (!buf) {
            return si::capi::fail(SYI_OUT_OF_MEMORY, "capture_ffn allocation failed");
        }
        std::memcpy(buf, out.data(), expected * sizeof(float));
        *out_data = buf;
        *out_n_texts = n_texts;
        *out_n_layers = n_layers;
        *out_n_ff = (size_t)n_ff;
        return SYI_OK;
    } catch (const std::exception & ex) {
        return si::capi::fail(SYI_BACKEND, ex.what());
    }
}

} // extern "C"
