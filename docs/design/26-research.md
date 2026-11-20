# Research-only FFN capture API (`sycl_infer_research.h`)

This describes the interpretability C API added under `research/`.  It is
**deliberately not part of the inference API**: a normal consumer includes only
`<sycl_infer/sycl_infer.h>` and never sees these symbols.  The header states the
isolation explicitly, the implementation lives outside `src/`, and the test
(`research/test_capi_ffn.cpp`) lives there too.  It links into the same
`libsycl_infer` shared object, so only the include is separate.

## The three layers

| layer | C entry points | needs a model handle? |
|---|---|---|
| capture | `syi_model_capture_ffn` | yes — a generative `syi_model` (a BERT embedding handle is rejected) |
| pattern | `syi_ffn_pattern_create` / `syi_ffn_pattern_top` / `syi_ffn_pattern_destroy` | no — pure host-side math over a captured tensor |
| similarity | `syi_ffn_similarity` | no — turns a pattern into an `[n_texts][n_texts]` `double` matrix |

`include/sycl_infer/sycl_infer_research.h` declares each options struct with a
`struct_size` and `abi_version` and provides an `*_options_init` function, so a
caller compiled against a different struct layout fails with `SYI_UNSUPPORTED`
rather than reading past the struct.  Every library allocation is released with
`syi_free`.

The Python twin is `sycl_infer.research` (`capture_ffn`, `FFNPattern`,
`similarity`, `concept_similarity`).  It is a separate module: importing
`sycl_infer` does not import it.

## Capture (`research/ffn_capture.cpp`)

`syi_model_capture_ffn` is the in-process twin of the `PF_DUMP_FFN` diagnostic.
Each input text is encoded verbatim (no chat template) and prefilled in one
chunk; a text longer than the model's prefill chunk (32 tokens) is rejected.
`options.layer` selects `[0, n_layer)` or `-1` for every layer, and
`token_slot` selects a zero-based token index or `-1` for the last real token.

The result is row-major `[text][layer][gate half (n_ff) || up half (n_ff)]`,
`n_texts * n_layers * 2 * n_ff` floats.  Engine support is deliberately small:
four state fields, one copy in the layer loop, and a forced direct replay
(no recorded command graph) while the capture runs.  A capture is **not**
thread-safe against generation on the same model handle — submit nothing during
it.

## Pattern and similarity (`research/ffn_pattern.cpp`)

The method, per layer:

1. z-score each FFN neuron across the batch of texts (`zscore=1`, the default;
   with a single text the raw activation values are used);
2. take each text's top-`K` neuron ids (`K` defaults to `dim/28`, about 128 for
   `dim = 3584`; `half` selects the gate half, up half, or both);
3. pairwise overlap is the shared fraction of two texts' id sets; its
   enrichment over the chance level `K/dim` maps to
   `score = clip(1 - 1/enrichment, 0, 1)`;
4. the score is averaged over the selected layers.  The diagonal is exactly 1.

A larger batch gives a more stable z-score.  `syi_ffn_pattern_top` can export
the raw ids and z values as flat `[text][layer][K]` arrays, so the similarity
step can be re-run with a different layer subset or chance baseline without a
second capture.

## Verification

`research/test_capi_ffn.cpp` (model-backed, CPU, 0.8B) checks: geometry,
determinism (two captures are byte-identical), finite values, top-id export,
and that semantically near words score higher than far words.  It is kept next
to the research sources in CMake's `test_capi_ffn` target, not under `tests/`,
to keep the research path isolated from the product test surface.

## Invariants

* Do not move the header's contents into `sycl_infer.h` — that would make a
  research probe part of the supported inference ABI.
* Do not add a research entry point to the Python `__init__` re-exports; callers
  import `sycl_infer.research` explicitly.
* The capture must use the same gate/up packing order as the engine's FFN
  output; reordering it silently invalidates any comparison against
  `PF_DUMP_FFN`.
