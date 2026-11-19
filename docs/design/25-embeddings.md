# Embedding models (GGUF BERT encoders such as BGE-M3)

The autoregressive engine is not the right shape for an embedding model: a
BERT encoder is bidirectional, has no LM head, no KV cache and no sampler.  A
GGUF whose `general.architecture == bert` is therefore routed to a separate,
small subsystem before `engine` is constructed.  One model file serves all
three backends:

```
GGUF (bert) --bert_model.cpp--> in-memory onnxpb::model_proto
                                    |-- onnx::executor (CPU / GPU queue)
                                    `-- si::npu::executor (Level Zero NPU graph)
```

## Files

| concern | file |
|---|---|
| SentencePiece (t5 / UGM) tokenizer | `src/embedding/sp_tokenizer.{h,cpp}` |
| GGUF loader + ONNX graph builder | `src/embedding/bert_model.{h,cpp}` |
| backend-agnostic encode/pool/normalise | `src/embedding/embedding_engine.{h,cpp}` |
| OpenAI `/v1/embeddings` server | `src/embedding/embedding_http.{h,cpp}` |
| C ABI `syi_model_embed` | `include/sycl_infer/sycl_infer.h`, `src/capi/capi_embedding.cpp` |
| tests | `tests/embedding/test_embed_tokenizer.cpp` (hermetic), `test_embed_bge.cpp` (model-backed) |

## CLI / API

```bash
# auto-detects the bert architecture; --device is cpu|gpu|auto|npu
sycl-infer --model /path/to/bge-m3-q8_0.gguf --device cpu --embed-max-seq 128 embed --prompt "hello"
sycl-infer --model /path/to/bge-m3-q8_0.gguf --device npu --embed-max-seq 128 serve   # /v1/embeddings
```

`--embed-max-seq` (default 512) fixes the sequence axis of the graph.  Inputs
are truncated or right-padded to it; an attention mask of `0`/`-1e9` keeps the
padding out of softmax.  NPU graphs need static shapes, so the same fixed axis
is used on every backend; batch is dynamic for the ONNX executor and pinned to
one for NPU (the NPU executor loops over rows itself).

## The GGUF → ONNX lowering

`bert_model::build_onnx` dequantizes all Q8_0 weights to f32 and emits:

* `Gather(token_table, input_ids) + Gather(position_table, position_ids)` →
  `LayerNorm` (decomposed, see below) — **embedding tables are copied, not
  transposed**: GGUF stores them `[dim, rows]` with `dim` fastest, which is
  already row-major `[rows, dim]` once dequantized.  Matrix weights *are*
  transposed (`out[k*N+n] = d[n*K+k]`) for ONNX's `[K,N]` MatMul layout.
* Per layer: Q/K/V `MatMul`+bias → reshape/transpose to `[B,nh,S,hd]` →
  `QKᵀ/√hd` + mask → `Softmax` → `PV` → transpose/reshape back → output
  `MatMul`+bias → residual → `LayerNorm` → FFN up + GELU → down → residual →
  `LayerNorm`.  The output transpose is `{0,2,1,3}`; `{0,2,3,1}` swaps heads
  and head-dims and silently destroys the embedding.
* Pooling from `bert.pooling_type`: `2` = CLS (BGE-M3), `1` = mean; then L2
  normalisation (`ReduceSum(x²) + 1e-12`, `Sqrt`, `Div`).

**LayerNorm is decomposed** into `ReduceMean`/`Sub`/`Mul`/`Sqrt`/`Div`/`Add`.
The NPU IR frontend maps ONNX `LayerNormalization` to the IR `LayerNorm` layer,
which the Lunar Lake driver rejects under opset1
(`Opset opset1 doesn't contain the operation with type: LayerNorm`).  The
decomposition is numerically identical and uses only opset1 primitives.

GELU is expressed with `Erf` rather than the ONNX `Gelu` op for the same
reason.  All three backends therefore run the same graph and the same
arithmetic order.

## Tokenizer

BGE-M3's GGUF declares `tokenizer.ggml.model = t5`: a Unigram (SentencePiece)
model with an NFKC normaliser compiled into `precompiled_charsmap` (XOR-
compressed double array).  The engine's BPE tokenizer cannot encode it, so
`sp_tokenizer` ports llama.cpp's UGM path: XCDA prefix normalisation with
whitespace escaped to U+2581, then a Viterbi unigram decode over the token
trie with the GGUF `scores`.

Two invariants, both pinned by `test_embed_tokenizer`:
* the `dp[0]` sentinel carries no token — backtracking must stop *before*
  emitting it, or every input starts with a spurious `<unk>`;
* only BOS and EOS are appended.  BGE-M3's `add_sep_token` metadata is true but
  `seperator_token_id == eos_token_id`, so adding SEP duplicates `</s>`.

## Verification (2026-11-19, BGE-M3 Q8_0, `--embed-max-seq 32`)

An independent NumPy forward over the same GGUF weights is the oracle (it
catches layout and transpose mistakes that a token-stream test cannot see).

| backend | sim(paraphrase) | sim(unrelated) | cos vs NumPy oracle |
|---|---:|---:|---:|
| CPU (`onnx::executor`) | 0.9796 | 0.3668 | 1.000000 |
| GPU (Iris Xe, `onnx::executor`) | 0.9796 | 0.3668 | 1.000000 |
| NPU (Lunar Lake, Level Zero graph) | – | – | 0.999992 |

The NPU delta is its executor's f32→FP16 conversion; max|diff| 5.9e-4, mean
9.9e-5, norm 0.99984.  NPU first-token wall is ~12 s including graph
construction and driver compile; CPU/GPU ~20 s including the one-time f32
dequantization of 567 M weights and graph build.  Subsequent requests on a
running server reuse the graph.

## Memory

The GGUF maps ~634 MB; the ONNX graph dequantizes all 567 M weights to f32
(~2.3 GB) and the executor keeps a host copy plus a device copy.  Peak RSS for
one process is therefore ~7.7 GB (measured, CPU and GPU).  The in-memory proto
is released as soon as the graph/IR is built, which saves one further f32 copy.
Keep `--embed-max-seq` modest: the weights dominate, but attention intermediates
are O(S²) per layer.

## What not to change

* Do not register `bert` in `src/model/arch/registry.cpp`: `model::load` would
  try to build an autoregressive plan and KV pool for an encoder.  Architecture
  detection is `embedding::is_embedding_model()` in the CLI.
* Do not add `LayerNormalization`/`Gelu` back into the graph for the NPU path;
  they are the two ops the driver rejects.
* Do not transpose embedding tables; only matrix weights need the layout flip.
* `--ctx` is ignored for embedding models: an encoder's graph axis is
  `--embed-max-seq`.  Raising it to the GGUF's 8192 without measuring is not
  free — the attention intermediates are O(S²) per layer.
