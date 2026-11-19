#include "capi_internal.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "embedding/embedding_engine.h"

namespace {

struct embedding_model {
    std::unique_ptr<si::embedding::engine> eng;
    int dim = 0;
    std::string device;
};

} // namespace

struct syi_embedding_model {
    std::unique_ptr<embedding_model> impl;
};

extern "C" {

void syi_embedding_options_init(syi_embedding_options *o) {
    if (!o) return;
    std::memset(o, 0, sizeof(*o));
    o->struct_size = sizeof(*o);
    o->abi_version = SYI_ABI_VERSION;
    o->max_seq = 0;
}

syi_status syi_embedding_model_create(const syi_embedding_options *options,
                                      syi_embedding_model **out_model) {
    if (!options || !out_model) return si::capi::fail(SYI_INVALID_ARGUMENT, "null argument");
    if (options->struct_size && options->struct_size < sizeof(*options))
        return si::capi::fail(SYI_INVALID_ARGUMENT, "syi_embedding_options is too small");
    if (!options->model_path) return si::capi::fail(SYI_INVALID_ARGUMENT, "model_path is required");
    try {
        si::embedding::engine_config cfg;
        cfg.model_path = options->model_path;
        cfg.device = options->device ? options->device : "auto";
        cfg.max_seq = options->max_seq > 0 ? options->max_seq : 512;
        auto e = std::make_unique<si::embedding::engine>(cfg);
        auto m = std::make_unique<syi_embedding_model>();
        m->impl = std::make_unique<embedding_model>();
        m->impl->dim = e->dim();
        m->impl->device = e->device();
        m->impl->eng = std::move(e);
        *out_model = m.release();
        return SYI_OK;
    } catch (const std::exception & ex) {
        return si::capi::fail(SYI_MODEL_LOAD, ex.what());
    }
}

void syi_embedding_model_destroy(syi_embedding_model *model) {
    delete model;
}

int32_t syi_embedding_dim(const syi_embedding_model *model) {
    return model && model->impl ? model->impl->dim : 0;
}

const char *syi_embedding_device(const syi_embedding_model *model) {
    return model && model->impl ? model->impl->device.c_str() : "";
}

syi_status syi_model_embed(syi_embedding_model *model,
                           const char *const *inputs, size_t n_inputs,
                           float **out_data, size_t *out_n, size_t *out_dim) {
    if (!model || !model->impl || !inputs || !out_data || !out_n || !out_dim)
        return si::capi::fail(SYI_INVALID_ARGUMENT, "null argument");
    *out_data = nullptr;
    *out_n = 0;
    *out_dim = 0;
    try {
        std::vector<std::string> texts;
        texts.reserve(n_inputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            if (!inputs[i]) return si::capi::fail(SYI_INVALID_ARGUMENT, "null input string");
            texts.emplace_back(inputs[i]);
        }
        const auto vecs = model->impl->eng->embed(texts);
        if (vecs.empty()) return SYI_OK;
        const size_t dim = vecs[0].size();
        float *buf = (float *)std::malloc(vecs.size() * dim * sizeof(float));
        if (!buf) return si::capi::fail(SYI_OUT_OF_MEMORY, "embedding allocation failed");
        for (size_t i = 0; i < vecs.size(); ++i)
            std::memcpy(buf + i * dim, vecs[i].data(), dim * sizeof(float));
        *out_data = buf;
        *out_n = vecs.size();
        *out_dim = dim;
        return SYI_OK;
    } catch (const std::exception & ex) {
        return si::capi::fail(SYI_BACKEND, ex.what());
    }
}

} // extern "C"
