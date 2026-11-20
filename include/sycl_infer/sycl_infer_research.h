/* sycl_infer_research.h - research-only C API for sycl-infer.
 *
 * This header is deliberately kept out of sycl_infer.h: the features here are
 * interpretability/research probes, not part of the inference API.  They link
 * into the same libsycl_infer shared object (only syi_* symbols are exported)
 * but have their own include so a normal inference consumer never sees them.
 *
 * Include sycl_infer.h first (or this header includes it for you).
 */
#ifndef SYCL_INFER_RESEARCH_H
#define SYCL_INFER_RESEARCH_H

#include <sycl_infer/sycl_infer.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------- FFN captures
 * In-process twin of PF_DUMP_FFN: read the FFN [gate|up] activations of one
 * token for one or more texts at a transformer layer (the substrate used by
 * concept-neuron / shared-activation analysis).  This only works with a
 * generative model handle (syi_model), not the BERT embedding handle.
 *
 * Each text is encoded verbatim (no chat template) and prefilled in one chunk;
 * inputs longer than the model's prefill chunk (32 tokens) are rejected.
 * `layer` is in [0, n_layer) or -1 for every layer.  `token_slot` is the
 * zero-based token index whose activation is read (-1 = the last real token).
 *
 * On success *out_data holds n_texts * n_layers * 2 * n_ff floats, row-major
 * as [text][layer][gate_half(n_ff) || up_half(n_ff)], to be released with
 * syi_free().  Not thread-safe against concurrent generation on the same
 * model handle: the caller must not submit requests during a capture.
 */
typedef struct syi_ffn_options {
    size_t struct_size;    /* = sizeof(syi_ffn_options) */
    uint32_t abi_version;  /* = SYI_ABI_VERSION */
    int32_t layer;         /* 0..n_layer-1, or -1 = all layers */
    int32_t token_slot;    /* 0..len-1, or -1 = last real token */
} syi_ffn_options;

SYI_API void syi_ffn_options_init(syi_ffn_options *options);
SYI_API syi_status syi_model_capture_ffn(syi_model *model, const syi_ffn_options *options,
                                         const char *const *texts, size_t n_texts,
                                         float **out_data, size_t *out_n_texts,
                                         size_t *out_n_layers, size_t *out_n_ff);

/* ------------------------------------------------ FFN patterns + similarity
 * Two-layer research API for the "shared activation" analysis, built entirely
 * on the activation tensor produced by syi_model_capture_ffn (no model handle
 * needed here).  Layer 1 extracts each input's top-K most-active FFN neurons
 * per layer; layer 2 turns those patterns into a [0,1] similarity matrix.
 *
 * The method (see the measurement notes under research/): per layer, each
 * neuron is z-scored across the batch of texts (zscore=1, the default), then
 * each text's top-K neuron ids are collected.  Pairwise overlap is the shared
 * fraction of those id sets; its enrichment over the chance level K/dim is
 * mapped to score = clip(1 - 1/enrichment, 0, 1), then averaged over the
 * selected layers.  Diagonal is 1.  Larger batches give a more stable z-score;
 * with a single text the raw activation values are used.
 */

/* ---- layer 1: pattern extraction ---- */
typedef struct syi_ffn_pattern_options {
    size_t struct_size;   /* = sizeof(syi_ffn_pattern_options) */
    uint32_t abi_version; /* = SYI_ABI_VERSION */
    int32_t top_k;        /* neurons per text/layer; <=0 = dim/28 (~128 for 3584) */
    int32_t half;         /* 0 = gate half (default), 1 = up half, 2 = both (dim=2*n_ff) */
    int32_t zscore;       /* 1 (default) = z-score each dim across texts; 0 = raw */
} syi_ffn_pattern_options;

typedef struct syi_ffn_pattern syi_ffn_pattern;

SYI_API void syi_ffn_pattern_options_init(syi_ffn_pattern_options *options);
/* `activations` is [n_texts][n_layers][2*n_ff] as returned by capture_ffn. */
SYI_API syi_status syi_ffn_pattern_create(const float *activations, size_t n_texts, size_t n_layers,
                                          size_t n_ff, const syi_ffn_pattern_options *options,
                                          syi_ffn_pattern **out_pattern);
SYI_API void syi_ffn_pattern_destroy(syi_ffn_pattern *pattern);
/* Export the top-neuron ids (and optionally their z values) as flat arrays
 * [n_texts][n_layers][top_k].  Either out_ids/out_z may be NULL.  Allocated
 * with syi_free(). */
SYI_API syi_status syi_ffn_pattern_top(const syi_ffn_pattern *pattern,
                                        int32_t **out_ids, float **out_z,
                                        size_t *out_n_texts, size_t *out_n_layers,
                                        size_t *out_top_k, size_t *out_dim);

/* ---- layer 2: 0-1 similarity from a pattern ---- */
typedef struct syi_ffn_similarity_options {
    size_t struct_size;   /* = sizeof(syi_ffn_similarity_options) */
    uint32_t abi_version; /* = SYI_ABI_VERSION */
    const int32_t *layers; /* layer indices to use, NULL = all; each in [0,n_layers) */
    size_t n_layers;       /* number of entries in `layers` (ignored if layers==NULL) */
    double chance_baseline;/* <=0 = theoretical K/dim (default) */
} syi_ffn_similarity_options;

SYI_API void syi_ffn_similarity_options_init(syi_ffn_similarity_options *options);
/* `out_matrix` is a newly allocated n_texts*n_texts double array (row-major),
 * release with syi_free(); score[i*n+j] is in [0,1]. */
SYI_API syi_status syi_ffn_similarity(const syi_ffn_pattern *pattern,
                                       const syi_ffn_similarity_options *options,
                                       double **out_matrix, size_t *out_n);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SYCL_INFER_RESEARCH_H */
