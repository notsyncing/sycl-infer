// Two-layer research API: FFN activation patterns and their [0,1] similarity.
// Pure host-side math over the activation tensor from syi_model_capture_ffn;
// no engine/model handle is involved, so this stays fully inside research/.
//
// Method (matching the measurements kept under research/): per layer, each
// neuron is z-scored across the batch of texts, then each text's top-K neuron
// ids are collected.  Pairwise overlap is the shared fraction of those id
// sets; its enrichment over the chance level K/dim is mapped to
// score = clip(1 - 1/enrichment, 0, 1), averaged over the selected layers.
#include <sycl_infer/sycl_infer_research.h>

#include "capi_internal.h"
#include "ffn_pattern_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

extern "C" {

void syi_ffn_pattern_options_init(syi_ffn_pattern_options *o) {
    if (!o) {
        return;
    }
    std::memset(o, 0, sizeof(*o));
    o->struct_size = sizeof(*o);
    o->abi_version = SYI_ABI_VERSION;
    o->top_k = 0;   // resolved to dim/28
    o->half = 0;    // gate half
    o->zscore = 1;  // on by default
}

void syi_ffn_similarity_options_init(syi_ffn_similarity_options *o) {
    if (!o) {
        return;
    }
    std::memset(o, 0, sizeof(*o));
    o->struct_size = sizeof(*o);
    o->abi_version = SYI_ABI_VERSION;
    o->layers = nullptr;
    o->n_layers = 0;
    o->chance_baseline = 0.0; // theoretical K/dim
}

syi_status syi_ffn_pattern_create(const float *activations, size_t n_texts, size_t n_layers,
                                  size_t n_ff, const syi_ffn_pattern_options *options,
                                  syi_ffn_pattern **out_pattern) {
    if (!out_pattern) {
        return si::capi::fail(SYI_INVALID_ARGUMENT, "null out_pattern");
    }
    *out_pattern = nullptr;
    if (!activations || n_texts == 0 || n_layers == 0 || n_ff == 0 || !options ||
        options->struct_size < sizeof(syi_ffn_pattern_options)) {
        return si::capi::fail(SYI_INVALID_ARGUMENT, "invalid pattern_create arguments");
    }
    if (options->abi_version != 0 && options->abi_version != SYI_ABI_VERSION) {
        return si::capi::fail(SYI_UNSUPPORTED, "unsupported syi_ffn_pattern_options.abi_version");
    }
    if (options->half < 0 || options->half > 2) {
        return si::capi::fail(SYI_INVALID_ARGUMENT, "pattern_create: half out of range");
    }

    try {
        const size_t dim = options->half == 2 ? 2 * n_ff : n_ff;
        const size_t src_stride = 2 * n_ff; // activation rows are [gate|up]
        const size_t src_off = options->half == 1 ? n_ff : 0;
        const size_t layer_stride = n_texts * src_stride;

        int top_k = options->top_k > 0 ? options->top_k : (int)(dim / 28);
        if (top_k < 1) {
            top_k = 1;
        }
        if ((size_t)top_k > dim) {
            top_k = (int)dim;
        }

        auto *p = new syi_ffn_pattern();
        p->n_texts = n_texts;
        p->n_layers = n_layers;
        p->top_k = (size_t)top_k;
        p->dim = dim;
        p->has_z = options->zscore != 0;
        p->ids.assign(n_texts * n_layers * (size_t)top_k, 0);
        if (p->has_z) {
            p->z.assign(n_texts * n_layers * (size_t)top_k, 0.0f);
        }

        // For each layer, z-score each neuron across texts into a [texts][dim]
        // matrix, then pick each text's top-K neurons (ties -> lower id).
        std::vector<float> norm(n_texts * dim);
        for (size_t l = 0; l < n_layers; ++l) {
            const float *layer_base = activations + l * layer_stride;
            for (size_t d = 0; d < dim; ++d) {
                double mean = 0.0, ss = 0.0;
                const bool z = p->has_z && n_texts > 1;
                if (z) {
                    for (size_t t = 0; t < n_texts; ++t) {
                        mean += layer_base[t * src_stride + src_off + d];
                    }
                    mean /= n_texts;
                    for (size_t t = 0; t < n_texts; ++t) {
                        const double v = layer_base[t * src_stride + src_off + d] - mean;
                        ss += v * v;
                    }
                }
                const double sd = z ? std::sqrt(ss / n_texts) + 1e-6 : 1.0;
                for (size_t t = 0; t < n_texts; ++t) {
                    norm[t * dim + d] = (float)((layer_base[t * src_stride + src_off + d] - mean) / sd);
                }
            }
            for (size_t t = 0; t < n_texts; ++t) {
                const float *row = norm.data() + t * dim;
                const size_t base = (t * n_layers + l) * (size_t)top_k;
                std::vector<int> idx(dim);
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                                  [&](int a, int b) {
                                      if (row[a] != row[b]) {
                                          return row[a] > row[b];
                                      }
                                      return a < b;
                                  });
                for (int k = 0; k < top_k; ++k) {
                    p->ids[base + k] = (int32_t)idx[k];
                    if (p->has_z) {
                        p->z[base + k] = row[idx[k]];
                    }
                }
            }
        }

        *out_pattern = p;
        return SYI_OK;
    } catch (const std::exception & ex) {
        return si::capi::fail(SYI_BACKEND, ex.what());
    }
}

void syi_ffn_pattern_destroy(syi_ffn_pattern *pattern) {
    delete pattern;
}

syi_status syi_ffn_pattern_top(const syi_ffn_pattern *pattern,
                               int32_t **out_ids, float **out_z,
                               size_t *out_n_texts, size_t *out_n_layers,
                               size_t *out_top_k, size_t *out_dim) {
    if (!pattern) {
        return si::capi::fail(SYI_INVALID_ARGUMENT, "null pattern");
    }
    if (out_ids) {
        const size_t n = pattern->ids.size();
        int32_t *buf = (int32_t *)std::malloc(n * sizeof(int32_t));
        if (!buf) {
            return si::capi::fail(SYI_OUT_OF_MEMORY, "pattern_top allocation failed");
        }
        std::memcpy(buf, pattern->ids.data(), n * sizeof(int32_t));
        *out_ids = buf;
    }
    if (out_z) {
        const size_t n = pattern->z.size();
        float *buf = (float *)std::malloc(std::max(n, (size_t)1) * sizeof(float));
        if (!buf) {
            return si::capi::fail(SYI_OUT_OF_MEMORY, "pattern_top allocation failed");
        }
        if (n) {
            std::memcpy(buf, pattern->z.data(), n * sizeof(float));
        }
        *out_z = buf;
    }
    if (out_n_texts) {
        *out_n_texts = pattern->n_texts;
    }
    if (out_n_layers) {
        *out_n_layers = pattern->n_layers;
    }
    if (out_top_k) {
        *out_top_k = pattern->top_k;
    }
    if (out_dim) {
        *out_dim = pattern->dim;
    }
    return SYI_OK;
}

syi_status syi_ffn_similarity(const syi_ffn_pattern *pattern,
                               const syi_ffn_similarity_options *options,
                               double **out_matrix, size_t *out_n) {
    if (!pattern || !out_matrix || !out_n) {
        return si::capi::fail(SYI_INVALID_ARGUMENT, "invalid similarity arguments");
    }
    *out_matrix = nullptr;
    *out_n = 0;
    if (!options || options->struct_size < sizeof(syi_ffn_similarity_options)) {
        return si::capi::fail(SYI_INVALID_ARGUMENT, "syi_ffn_similarity_options.struct_size is too small");
    }
    if (options->abi_version != 0 && options->abi_version != SYI_ABI_VERSION) {
        return si::capi::fail(SYI_UNSUPPORTED, "unsupported syi_ffn_similarity_options.abi_version");
    }

    try {
        const size_t n = pattern->n_texts;
        const size_t K = pattern->top_k;
        const double chance = options->chance_baseline > 0.0
                                  ? options->chance_baseline
                                  : (double)K / (double)pattern->dim;
        if (chance <= 0.0) {
            return si::capi::fail(SYI_INVALID_ARGUMENT, "similarity: invalid chance baseline");
        }

        std::vector<size_t> layers;
        if (options->layers && options->n_layers) {
            for (size_t i = 0; i < options->n_layers; ++i) {
                if (options->layers[i] < 0 || (size_t)options->layers[i] >= pattern->n_layers) {
                    return si::capi::fail(SYI_INVALID_ARGUMENT, "similarity: layer out of range");
                }
                layers.push_back((size_t)options->layers[i]);
            }
        } else {
            for (size_t l = 0; l < pattern->n_layers; ++l) {
                layers.push_back(l);
            }
        }
        if (layers.empty()) {
            return si::capi::fail(SYI_INVALID_ARGUMENT, "similarity: no layers selected");
        }

        std::vector<double> accum(n * n, 0.0);
        const size_t row_items = K;
        for (size_t li : layers) {
            // sorted id sets, one per text, at this layer
            std::vector<std::vector<int32_t> > sets(n, std::vector<int32_t>(row_items));
            for (size_t t = 0; t < n; ++t) {
                const int32_t *src = pattern->ids.data() + (t * pattern->n_layers + li) * K;
                std::memcpy(sets[t].data(), src, row_items * sizeof(int32_t));
                std::sort(sets[t].begin(), sets[t].end());
            }
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i; j < n; ++j) {
                    double ov;
                    if (i == j) {
                        ov = 1.0;
                    } else {
                        size_t c = 0, a = 0, b = 0;
                        while (a < row_items && b < row_items) {
                            if (sets[i][a] == sets[j][b]) {
                                ++c;
                                ++a;
                                ++b;
                            } else if (sets[i][a] < sets[j][b]) {
                                ++a;
                            } else {
                                ++b;
                            }
                        }
                        ov = (double)c / (double)row_items;
                    }
                    const double e = ov / chance;
                    double s = e > 1.0 ? 1.0 - 1.0 / e : 0.0;
                    accum[i * n + j] += s;
                    if (j != i) {
                        accum[j * n + i] += s;
                    }
                }
            }
        }

        const double inv = 1.0 / (double)layers.size();
        double *mat = (double *)std::malloc(n * n * sizeof(double));
        if (!mat) {
            return si::capi::fail(SYI_OUT_OF_MEMORY, "similarity allocation failed");
        }
        for (size_t i = 0; i < n * n; ++i) {
            double s = accum[i] * inv;
            mat[i] = s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s);
        }
        for (size_t i = 0; i < n; ++i) {
            mat[i * n + i] = 1.0; // self-similarity is exactly 1 by definition
        }
        *out_matrix = mat;
        *out_n = n;
        return SYI_OK;
    } catch (const std::exception & ex) {
        return si::capi::fail(SYI_BACKEND, ex.what());
    }
}

} // extern "C"
