# sycl-infer — a general-purpose SYCL LLM inference engine

> **Experiment project to explore whether AI can be used to create an inference engine.**

sycl-infer is a from-scratch C++17/SYCL inference engine for quantized GGUF
models on Intel GPUs or the host CPU. It loads GGUF files directly (single file
or split shards), serves them through an OpenAI-compatible HTTP API or a CLI,
and selects the model through a **pluggable architecture registry**. The same
tree also runs BERT-style embedding models and hosts an ONNX model service for
document pipelines.

The reference model is Qwen3.5-0.8B (Q4_K_M) on Iris Xe-LP; throughput is
measured on Qwen3.8-27B and Qwen3.6-35B-A3B on two Arc A770. Every number, with
its conditions, is in **[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)**.

> **This project is entirely AI-generated.** All of it — kernels, engine,
> server, tests, documentation — was written by AI coding agents. The author
> knows nothing about GPU optimization: the occupancy cliffs, the
> bytes-per-weight trade-offs and the arithmetic-intensity ceilings this design
> is built around were found by measurement, not by expertise. Re-measure before
> trusting a number anywhere else — the tuning constants are per-GPU profiles
> precisely because they do not transfer.

## Features

- **GGUF models** — memory-mapped load, single device blob, no conversion step;
  Q4_K / Q5_K / Q6_K / Q3_K / Q5_1 / IQ4_XS / IQ4_NL / IQ3_S / Q8_0 / F16 /
  BF16 / F32 dequantized inside the kernels. Native lossless stores for Q4_K
  (u4), Q5_K (nibble + bit planes) and IQ4 (codebook 4-bit) cut per-token weight
  traffic; `PF_W4=0` / `PF_K5=0` / `PF_CB4=0` restore the int8 path.
- **Pluggable architectures** — `general.architecture` selects a loader and a
  model profile under `src/model/arch/<arch>/`; a model-owned forward adds
  `src/engine/arch/<arch>/` and one `kArchOps[]` row. See
  [Design 24](docs/design/24-arch-modules.md).
- **Paged KV cache + continuous batching** — 32-token blocks, chunked prefill,
  batched decode (16 concurrent sequences by default; the compile-time ceiling
  is 32, `--max-slots` is the policy that sizes the recurrent state).
- **Cross-request prefix cache** — hashed 32-token blocks with recurrent-state
  checkpoints and three LRU tiers: VRAM → host RAM → disk.
- **Compute paths** — in-house int8 DP4A/SIn kernels, an integer int8 GEMV
  straight from the GGUF blocks on CPU, optional oneDNN int8 GEMM for prefill,
  oneDNN int8 matmul prefill attention, and recorded SYCL command graphs for
  decode. KV type `i8` (default), `i4`, `bf16`, `f16`, `f32`, or `--kv-type K:V`
  to size the two caches independently.
- **Devices** — `--device cpu|gpu|auto`; the CPU backend dispatches AVX2 /
  AVX-VNNI / AVX-512 at run time. `--layer-map` places contiguous layer ranges
  on different devices (pipeline parallel), each with its own KV. Per-card
  tuning lives in `src/device/profiles/<card>.cpp`; an unknown card warns.
- **OpenAI-compatible API** — streaming and non-streaming chat/completions,
  reasoning content, tool calls, logprobs/`logit_bias`, JSON-schema guided
  decoding, `n` choices, `/v1/models`, and usage with prefix-cache hit counts.
- **Multimodal** — images (clip mmproj), video (sampled frames) and audio
  (separate AuT tower), freely mixed in one prompt; 4-section M-RoPE positions,
  media tokens bypass the prefix cache. Remote media URLs are off by default.
- **Speculative decoding** — `--mtp N` uses the model's own NextN head and
  `--spec-type dflash2` uses a separate block drafter; both are single-request
  greedy paths and their speedup depends entirely on prompt acceptance.
- **Embedding models** — GGUF BERT encoders (e.g. BGE-M3) on CPU, GPU or NPU,
  with a SentencePiece tokenizer, `/v1/embeddings`, an `embed` CLI command and
  a C ABI handle. See [Embedding models](#embedding-models).
- **Router and ONNX service** — one worker per model behind a single endpoint,
  plus a separate service for the document pipeline's ONNX models. See
  [Multiple models: the router](#multiple-models-the-router) and
  [The document pipeline](#the-document-pipeline-onnx-model-service).

## Supported models

| model | `general.architecture` | notes |
|---|---|---|
| Qwen3.5-0.8B (Q4_K_M) | `qwen35` | hybrid Gated DeltaNet + full attention; the test-suite reference |
| Qwen3.8-27B (UD-Q4_K_M) | `qwen35` | dense, 64 layers; needs the two-card split |
| Qwen3.6-35B-A3B (UD-Q4_K_M) | `qwen35moe` | routed MoE, 256 experts / 8 active, 40 layers; needs the two-card split |
| Qwen3.8-Flash-Next (UD-Q4_K_XL, 4 shards) | `qwen4exp` | hyper-connections + n-gram PLE + 512-expert MoE + QSA. Recommended: `--layer-map 0-19:gpu.0,20-47:gpu.1 --moe-vram-mb 17500 --ctx 2048 --kv-type f16`. [Design 18](docs/design/18-qwen4exp.md) |
| DeepSeek-V4.1-Flash (IQ2_XS, 6+Engram shards) | `deepseek41` | 40-layer all-MoE: MLA (K==V 512-wide latent), group-shared compressed KV, two-level DSA, 4-stream mHC, Engram. Needs `--device gpu --layer-map ... --moe-vram-mb 16000 --ctx 512 --max-slots 1`. [Design 23](docs/design/23-deepseek41.md) |
| MinerU2.5-Pro-2605-1.2B | `qwen2vl` | dense Qwen2-VL (24 layers, head_dim 64) behind the standalone `sycl-infer-mineru` client; needs its mmproj. [Design 21](docs/design/21-mineru-qwen2vl.md) |
| BGE-M3 (GGUF BERT) | `bert` | bidirectional encoder, routed away from `engine`; see [Embedding models](#embedding-models) |

File format: GGUF, single file or split shards merged into one virtual mapping.
Architecture detection is exact (`general.architecture`), validated against the
profile at load. Adding an architecture starts with
`src/model/arch/<arch>/{load,profile}.cpp` — see
[`AGENTS.md`](AGENTS.md#add-a-model-architecture) and
[Design 24](docs/design/24-arch-modules.md).

## Requirements

- Linux, C++17, CMake ≥ 3.20
- Intel oneAPI DPC++ compiler (`icpx`), tested with oneAPI 2026.2
- oneDNN 2026.0 (optional prefill path; default `/opt/intel/oneapi/dnnl/2026.0`,
  override with `-DDNNL_ROOT=`; 2026.1 also works)
- An Intel GPU with SYCL support (developed on Iris Xe-LP; 27B-class models on
  two Arc A770). The CPU backend (`--device cpu`) needs no GPU. NPU execution
  of ONNX/embedding graphs needs a Lunar Lake NPU with the Level Zero graph
  extension; no OpenVINO is linked or called.

## Build

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build
cmake --build build -j
```

AOT-compile the device image with `-DSYCL_INFER_AOT=ON`. The default target is
`adl-p`; `-DSYCL_INFER_AOT_DEVICE=acm-g10` selects another (it must agree with
the profile the binary picks at run time). `-DSYCL_INFER_AOT_JOBS=N` bounds the
parallel IGC translations.

The router builds on its own, without SYCL, oneDNN or even `icpx`:

```bash
cmake -S . -B build-router-only -DSI_ROUTER_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-router-only -j
```

## Usage

```bash
# OpenAI-compatible HTTP server (binds 127.0.0.1:8080 by default)
./build/sycl-infer --model /path/to/model.gguf serve

# one-shot generation
./build/sycl-infer --model /path/to/model.gguf gen --prompt "Hello" --max-tokens 128

# image prompt (needs the matching mmproj GGUF)
./build/sycl-infer --model /path/to/model.gguf --mmproj /path/to/mmproj.gguf \
    gen --image photo.png --prompt "Describe this image."

# 27B on two GPUs
./build/sycl-infer --model /path/to/Qwen3.8-27B-UD-Q4_K_M.gguf --ctx 4096 \
    --layer-map 0-31:gpu.0,32-63:gpu.1 gen --prompt "Hello"
```

Common flags:

| flag | default | meaning |
|---|---|---|
| `--model <gguf>` | 0.8B path | model file |
| `--device cpu\|gpu\|auto\|npu` | auto (gpu) | compute backend; `npu` for ONNX/embedding graphs |
| `--ctx N` \| `full` | 20480 | max sequence length; `full` reads the GGUF's context length |
| `--blocks N` | 512 | KV blocks committed at startup (32 tokens each); grows lazily |
| `--kv-cap-mb N` | auto | cap for the grown KV pool; also caps the three prefix-cache tiers |
| `--kv-type T` \| `K:V` | `i8` | `i4` / `i8` / `bf16` / `f16` / `f32`, or independent K and V |
| `--max-slots N` | 16 | concurrent sequences; sizes the recurrent state |
| `--layer-map L:dev,...` | – | `0-31:gpu.0,32-63:gpu.1`, or `cpu` ranges |
| `--mmproj` / `--audio-mmproj <gguf>` | – | vision / audio towers |
| `--image` / `--video` / `--audio <file>` | – | `gen`: attach media (repeatable, mixable) |
| `--moe-resident N` / `--moe-vram-mb MB` / `--moe-ram-mb MB` | 0 / 0 / 0 | routed-MoE expert placement and caches |
| `--engram-ram-mb MB` | 4096 | qwen4exp PLE row cache in host RAM |
| `--q4-tail-skip N` | 0 | qwen4exp expert tail-skip (`0` = exact) |
| `--spec-type none\|mtp\|dflash2` | none | speculative path |
| `--spec-draft-model <gguf>` / `--spec-draft-n-max N` | – / 5 | drafter GGUF and tokens per cycle |
| `--mtp [N]` / `--mtp-device N` | 4 / 0 | alias for the MTP path |
| `--pc-vram-mb` / `--pc-ram-mb` / `--pc-dir` / `--pc-disk-mb` | – / 512 / – / 1024 | prefix-cache tiers |
| `--host H` / `--port N` | 127.0.0.1 / 8080 | server bind address and port (no authentication) |
| `--control-port` / `--control-host` / `--control-token[-file]` | off / 127.0.0.1 / – | JSON-RPC control plane used by the router |
| `--yarn` / `--yarn-factor F` | off / derive | YaRN context extension; factor defaults to `--ctx / <arch>.context_length` |
| `--max-tokens` / `--temp` / `--top-p` / `--top-k` | 256 / 0.7 / 0.95 / 40 | `gen` sampling |
| `--raw` / `--thinking` | off | verbatim prompt / reasoning flag |
| `-h` / `--help` | – | print the built-in flag list |

`--mtp` needs a GGUF with `blk.<n>.nextn.*` **and** a multi-device oneDNN int8
partition; otherwise the engine prints one `[mtp]` line and decodes plainly.
`--spec-type dflash2` needs a matching draft GGUF. Both are greedy-only and do
not compose with batching, so the server keeps them off by default.

HTTP endpoints:

- `POST /v1/chat/completions`, `POST /v1/completions`
- `POST /v1/embeddings` (only on an embedding server, see below)
- `GET /v1/models`, `GET /v1/models/{id}`, `GET /health`

## Embedding models

A GGUF with `general.architecture == bert` (e.g. BGE-M3 Q8_0) is detected before
`engine` is constructed: a bidirectional encoder has no LM head, KV cache or
sampler. The file is lowered once to an in-memory ONNX graph; the same graph
runs on the CPU/GPU ONNX executor or on the NPU executor, so there are no
per-backend forward kernels to keep in sync.

```bash
# embed command (auto-detects bert; --device cpu|gpu|npu)
./build/sycl-infer --model /path/to/bge-m3-q8_0.gguf --device cpu \
    --embed-max-seq 128 embed --prompt "hello"

# /v1/embeddings server on the NPU
./build/sycl-infer --model /path/to/bge-m3-q8_0.gguf --device npu serve
```

`--embed-max-seq` (default 512) fixes the graph's sequence axis; inputs are
truncated or right-padded. BGE-M3 uses a SentencePiece Unigram tokenizer
(`tokenizer.ggml.model = t5`), not the engine's BPE. The graph stays in opset1
primitives (LayerNorm and GELU are decomposed) because the Lunar Lake NPU
driver rejects the higher-level operators. Output vectors are CLS-pooled and
L2-normalised. CPU/GPU match an independent NumPy oracle at cos 1.000000, NPU
at 0.999992 (FP16). See [Design 25](docs/design/25-embeddings.md).

## Embedding the engine: C API, Python, ONNX

- **C ABI** (`libsycl_infer`) — opaque handles, async submit + pull streaming,
  automatic batching, cancellation and timeouts. Includes a separate
  `syi_embedding_model` handle with `syi_model_embed`. See
  [`docs/guides/c-api.md`](docs/guides/c-api.md) and
  [Design 19](docs/design/19-c-api.md).
- **Python** (`sycl_infer`) — a ctypes binding over the C ABI (v0.1) with a
  wheel that vendors the SYCL/oneDNN runtimes. See
  [`docs/guides/python.md`](docs/guides/python.md) and
  [Design 20](docs/design/20-python-binding.md).
- **ONNX** (`libsycl_infer_onnx`) — the document models in-process:
  `syi_onnx_create` / `syi_onnx_run_wire`, one shared queue per device. The
  `device` argument accepts `GPU`, `GPU.N`, `CPU`, `AUTO`, `NPU` and
  `map:<file>` (per-model device routing). See
  [`docs/guides/onnx-inprocess.md`](docs/guides/onnx-inprocess.md).

## Multiple models: the router

`sycl-infer-router` serves several models behind one OpenAI-compatible
endpoint: one worker process per model, chosen by the request's `model` field.
It links neither SYCL nor oneDNN, needs no GPU to build, and supervises ONNX
workers (`"backend": "onnx"`) as well as LLM workers.

```bash
./build-router-only/sycl-infer-router --config router.json --check
./build-router-only/sycl-infer-router --config router.json
```

The config is JSON with comments. The router starts and watches workers, admits
each model against the memory the worker *reports* it uses (not the driver's
free-memory figure), proxies the OpenAI API including SSE and cancellation, and
does not retry a worker that exits 78 (`EX_CONFIG`). Remote workers connect
instead of starting. Containers: `docker/Dockerfile.router`,
`Dockerfile.worker`, `Dockerfile.all`, `compose.yaml`. See
[Design 16](docs/design/16-router.md).

## The document pipeline: ONNX model service

`sycl-infer-onnx` runs the document pipeline's ONNX models — layout detector,
OCR detector and recogniser, table classifiers and structurers, formula
recogniser. Its runtime is this project's own: protobuf reader, graph IR with
independent transform passes, operator library, SYCL kernels and a memory
planning executor. Every kernel takes its shapes as runtime arguments, so there
is no per-shape kernel JIT; a non-AOT build still pays the SYCL runtime's own
first-call JIT (~1.2–2.6 s), which `-DSYCL_INFER_AOT=ON` removes.

```bash
cmake -S . -B build -DSI_ONNX=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sycl-infer-onnx -j
./build/sycl-infer-onnx --config onnx_models.json --port 8890
```

The client (`rapid_doc/model/http_infer.py`) speaks a binary protocol and
returns numpy arrays. The router supports `"backend": "onnx"` with a binary
`POST /v1/onnx/run` proxy. All twelve pipeline models load; the seven covered
by the captured corpus are exact or within float32 accumulation noise, and the
formula recogniser is integer-equal.

The NPU path runs fixed-shape stages (OCR recogniser, classifiers) on the Lunar
Lake NPU through the Level Zero graph extension, without OpenVINO. Precision-
sensitive layout and dynamic-shape detector stages must stay on GPU/CPU.
Details, measurements and the validated real-PDF configuration are in
[Design 17](docs/design/17-onnx-service.md) and
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Performance

Measured on two Arc A770 16 GB, AOT Release, i8 KV, prefix cache off. The full
tables, harnesses and commands are in **[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)**.

| model | placement | prefill | decode, one stream |
|---|---|---:|---:|
| Qwen3.5-0.8B | 1 card | 6 336 tok/s (512) | 78.7 tok/s |
| Qwen3.8-27B | 2 cards | 507 tok/s (512) | 13.4 tok/s |
| Qwen3.6-35B-A3B | 2 cards | 779 tok/s (512) | 26.4 tok/s |
| Qwen3.8-Flash-Next | 2 cards, LRU bank, pinned source | 55.9 tok/s (3 660) | 11.3 tok/s |

Decode scales with context depth (`ms/step = 37.8 + 0.202 µs × depth` for the
35B) and aggregates with concurrency (N=1/2/4/8/16/32 → 26.4/35.7/60.9/101.1/
122.1/168.3 tok/s on the 35B). Batched qwen4exp decode is correct but 0.85×, so
the scheduler serialises its sequences. MTP and DFlash2 buy single-request
latency only when the prompt supports the draft; on a story opener DFlash2 is
*slower* than no drafter. Acceptance, not engine work, is the variable.

The 35B also runs single-device on an Arc 140V (Lunar Lake): 641 tok/s prefill
and ~30 tok/s decode; its shared-memory capacity, expert-bank governor and
safe context maxima are in [`AGENTS.md`](AGENTS.md) (*Long context on the Arc
140V*).

## Configuration

Runtime tuning is via environment variables; the most useful:

| variable | default | meaning |
|---|---|---|
| `PF_CTX` / `PF_KV_CAP_MB` / `PF_KV_TYPE` | 20480 / auto / `i8` | defaults for `--ctx` / `--kv-cap-mb` / `--kv-type` |
| `PF_DEVICE` / `PF_CPU_THREADS` / `PF_CPU_ISA` | gpu / physical cores / auto | backend and CPU kernel variant |
| `PF_PREFIX_CACHE` / `PF_PC_STATES` / `PF_PC_DIR` | on / 8 / – | prefix cache and its tiers |
| `PF_DP4A` / `PF_GEMM_DNNL` / `PF_ATTN_XMX` / `PF_W4` / `PF_K5` / `PF_CB4` | on | compute-path switches; `0` selects the slower reference path |
| `PF_MTP` / `PF_DFLASH_NMAX` | – / 5 | draft length overrides |
| `PF_DEVICE_PROFILE` / `PF_DEVICE_INFO` / `PF_MODEL_INFO` | – | pin or dump the resolved profiles |
| `PF_MM_URL_FETCH` / `PF_MM_URL_ALLOW_PRIVATE` | off | allow remote media URLs (the latter also private hosts) |
| `PF_YARN` / `PF_YARN_FACTOR` | off / derive | YaRN context extension |
| `PF_NPU_STATIC_DIMS` / `PF_NPU_INPUT_SHAPES` / `PF_NPU_IR_FILE` | – | NPU graph shapes and pre-generated IR |

The API has no authentication: the bind address (`--host`) and
`PF_MM_URL_FETCH` are the security boundary. The full knob list and invariants
are in [`AGENTS.md`](AGENTS.md).

## Tests

```bash
source /opt/intel/oneapi/setvars.sh
cmake --build build -j
ctest --test-dir build -L hermetic --output-on-failure   # no GPU, no model
./build/test_model_profile   # registry <-> model profile keys
./build/test_dev_profile     # device-profile values and fallback
./build/test_kv_type ./build/test_f16conv ./build/test_yarn
./build/test_embed_tokenizer # hermetic SentencePiece tokenizer
./build/test_sched           # scheduler + 20-row batched decode (GPU)
./build/test_decode_vs_prefill  # single-token decode == re-prefill (GPU)
./build/test_spec            # MTP / DFlash2 streams == greedy (GPU, 27B)
```

GPU/model tests are opt-in: configure with `-DSI_REGISTER_GPU_TESTS=ON` and the
model paths, then `ctest --test-dir build -L gpu`. The ONNX NPU test
(`test_onnx_npu`) needs an NPU machine. `ctest -N` lists all names and labels.

## Credits

sycl-infer is an independent implementation, but stands on several shoulders:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT) — the GGUF
  container and K-quant layouts, byte-level BPE behaviour, vendored Unicode
  tables and the NextN/MTP reference semantics.
- **[oneDNN](https://github.com/oneapi-src/oneDNN)** (Apache-2.0) — the int8
  matmul primitives behind the prefill GEMM and attention, plus the DPC++ toolchain.
- **[OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)** —
  the prefill reference point (with the build and quantization caveats recorded
  in the design docs).
- **llama-benchy** — the benchmark harness for the depth cells (its own `pp512`
  column is an artefact and is not quoted).
- **[Strata](https://github.com/Niko1221/Strata)** (MIT) — the CUDA reference
  for Qwen3.8-Flash-Next, whose tail-skip and per-expert cache ideas informed
  the qwen4exp work.

## License

MIT — see [`LICENSE`](LICENSE). Bundled third-party code in `third_party/`
retains its own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
