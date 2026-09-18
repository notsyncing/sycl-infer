# sycl-infer — a general-purpose SYCL LLM inference engine

sycl-infer is a from-scratch C++17/SYCL inference engine for quantized LLM
models on Intel GPUs or the host CPU.  It loads GGUF models directly, serves
them through an OpenAI-compatible HTTP API or a CLI, and is built around a
**pluggable architecture registry** so new model types can be added without
touching the engine or the kernel library.

The reference model used for development, validation and benchmarks is
Qwen3.5-0.8B (Q4_K_M) on Intel Iris Xe-LP; the engine itself is not tied to it.

## Features

* **GGUF models** — memory-mapped load, one device blob, no conversion step.
  Weight types Q4_K / Q5_K / Q6_K / Q8_0 / F32 are dequantized on the fly inside
  the kernels.
* **Pluggable architectures** — the model type is read from
  `general.architecture` and dispatched to a per-architecture loader (one source
  file plus one registry entry); `qwen35` (hybrid Gated DeltaNet + full
  attention) is implemented today.
* **Paged KV cache + continuous batching** — 32-token physical blocks with
  per-sequence block tables; chunked prefill and batched decode (up to 16
  concurrent sequences).
* **Cross-request prefix cache** — hashed 32-token blocks with recurrent-state
  checkpoints, so a shared prompt prefix is prefilled only once.
* **Quantized compute** — in-house int8 DP4A/SIn kernels on the GPU and an
  integer int8 GEMV straight from the GGUF blocks on the CPU, plus an optional
  oneDNN int8 GEMM path for prefill; KV cache stored as int8 (default), bf16,
  f16 or f32.
* **Device selection** — `--device cpu|gpu|auto`; the CPU backend has its own
  AVX2 / AVX-VNNI / AVX-512 kernels, picks the variant at run time, and keeps
  the paged KV in host RAM.
* **Multi-device (pipeline parallel)** — `--layer-map 0-11:gpu,12-23:cpu` places
  contiguous layer ranges on devices; each device owns the paged KV of its
  attention layers.
* **SYCL command graphs** — the full forward step is captured per shape and
  replayed, keeping per-token launch overhead minimal (GPU only).
* **OpenAI-compatible API** — streaming and non-streaming chat/completions with
  temperature / top-k / top-p / min-p / penalties, `logit_bias`,
  `logprobs`/`top_logprobs`, `reasoning_content` (thinking models),
  function/tool calling (`tools`, `tool_calls`, `tool` results), `n` choices,
  `echo`/`suffix`/`best_of` and `/v1/models`; `usage` reports prefix-cache
  `cached_tokens` and `reasoning_tokens` details.
* **Multimodal (vision) input** — Qwen3.5 image input via a `clip` mmproj GGUF:
  the vision encoder runs on the GPU (`src/backend/gpu/kernels/vit.cpp`), the CLI takes
  `--mmproj` + `--image`, and the server accepts OpenAI `image_url` content
  parts (base64 `data:` URLs or remote `http(s)://` URLs, bounded to 10 s / 10 MB).

## Supported models

| | |
|---|---|
| file format | GGUF (memory-mapped, single file) |
| weight quantization | Q4_K, Q5_K, Q6_K, Q8_0, F32 |
| architecture registry | selected via `general.architecture`; `qwen35` implemented, others plug in via `src/model/model_arch.h` + a `kLoaders[]` entry |
| KV cache dtype | `i8` (default), `bf16`, `f16`, `f32` |
| tokenizer | GPT-2 byte-level BPE from the GGUF vocabulary + special tokens / chat template |

Adding an architecture is a single new file under `src/model/` — see
[`AGENTS.md`](AGENTS.md#add-a-model-architecture).

## Requirements

* Linux, C++17, CMake ≥ 3.20
* Intel oneAPI DPC++ compiler (`icpx`) — tested with oneAPI 2026.1
* oneDNN 2026.0 (optional prefill path; `/opt/intel/oneapi/dnnl/2026.0`,
  override with `-DDNNL_ROOT=`)
* An Intel GPU with SYCL support (developed against Iris Xe-LP).  The CPU
  backend (`--device cpu`) runs without a GPU, using the AVX2 / AVX-VNNI /
  AVX-512 kernel variants selected at run time.

## Build

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build
cmake --build build -j
```

Optional: AOT-compile the device image with `-DSYCL_INFER_AOT=ON` (defaults to
`adl-p`; edit `CMakeLists.txt` for another target).

## Usage

```bash
# OpenAI-compatible HTTP server (default port 8080)
./build/sycl-infer --model /path/to/model.gguf serve

# one-shot CLI generation
./build/sycl-infer --model /path/to/model.gguf gen \
    --prompt "Hello" --max-tokens 128 --temp 0.7

# multimodal: describe an image (needs the matching mmproj GGUF)
./build/sycl-infer --model /path/to/model.gguf --mmproj /path/to/mmproj.gguf \
    gen --image photo.png --prompt "Describe this image."
```

Common flags:

| flag | default | meaning |
|---|---|---|
| `--model <gguf>` | – | model file (required) |
| `--mmproj <gguf>` | – | vision projector; required for `--image` and server image input |
| `--image <file>` | – | `gen`: attach an image (repeatable) |
| `--ctx N` | 20480 | max sequence length in tokens |
| `--blocks N` | 512 | KV blocks committed at startup (32 tokens each) |
| `--kv-cap-mb N` | auto | upper bound for the dynamically grown KV pool; also caps the sum of the three cache tiers (see below) |
| `--device cpu\|gpu\|auto` | auto (gpu) | compute backend; `auto` reads `PF_DEVICE` |
| `--cpu-threads N` | physical cores | CPU backend worker threads (`0` = auto) |
| `--layer-map L:dev,...` | – | pipeline-parallel layer placement, e.g. `0-11:gpu,12-23:cpu` |
| `--pc-vram-mb N` | – | VRAM (device) prefix-cache budget, converted to a checkpoint count |
| `--pc-ram-mb N` | 512 | host-RAM prefix-cache budget (`0` disables the RAM tier) |
| `--pc-dir DIR` | – | enable the disk prefix-cache tier in `DIR` (model-scoped) |
| `--pc-disk-mb N` | 1024 | disk prefix-cache budget (`0` = unbounded) |
| `--pc-mem-mb N` | – | alias for `--pc-vram-mb` (or set `PF_PC_STATES` directly) |
| `--host H` / `--port N` | 0.0.0.0 / 8080 | server bind address |
| `--max-tokens` / `--temp` / `--top-p` / `--top-k` | 256 / 0.7 / 0.95 / 40 | `gen` sampling |
| `--raw` | off | `gen`: send the prompt verbatim (no chat template) |

HTTP endpoints:

* `POST /v1/chat/completions` — OpenAI chat API (streaming + non-streaming;
  `stop`, `ignore_eos`, `stream_options.include_usage`, `n`, `reasoning_content`,
  `tools`/`tool_choice` + `tool_calls`/`tool` results, `logit_bias`, `logprobs`/
  `top_logprobs`, image parts)
* `POST /v1/completions` — raw text completion (string/string[]/token-id prompt,
  `n`, `echo`, `suffix`, `best_of`, `logit_bias`, `logprobs`,
  `stream_options.include_usage`)
* `GET /v1/models` + `GET /v1/models/{id}`, `GET /health`

### Prefix cache

The cross-request prompt prefix cache has three LRU tiers:

| tier | holds | configured by |
|---|---|---|
| VRAM | KV block + recurrent-state checkpoint in device memory | `--pc-vram-mb` / `PF_PC_STATES` |
| RAM | serialized record in host memory | `--pc-ram-mb` |
| disk | record file in `--pc-dir` (model-scoped) | `--pc-disk-mb` |

A lookup walks VRAM → RAM → disk and promotes the hit back to VRAM; an eviction
demotes in the opposite order, **VRAM → RAM → disk → dropped**.  A promotion is a
move, so a record lives in exactly one tier.  The RAM tier is process-local; the
disk tier is read at startup.  On a graceful shutdown (Ctrl-C / SIGTERM, or a
normal `gen` exit) the RAM records and the resident VRAM nodes are flushed to
disk when a directory is configured.

The three budget flags are one KV budget: when `--kv-cap-mb` / `PF_KV_CAP_MB` is
given, the sum of the three tiers is clamped to it (shrinking disk first, then
RAM, then the VRAM checkpoint count).  When it is not given, the configured three
tiers define the budget themselves.  The device KV pool reservation is always
made large enough to hold the VRAM tier.

## Configuration

Runtime behavior is controlled by environment variables.  The most useful ones:

| variable | default | meaning |
|---|---|---|
| `PF_CTX` | 20480 | default for `--ctx` |
| `PF_KV_CAP_MB` | auto | default for `--kv-cap-mb` |
| `PF_DEVICE` | `gpu` | default for `--device` (`cpu` / `gpu`) |
| `PF_CPU_THREADS` | physical cores | CPU backend worker threads (`--cpu-threads` overrides) |
| `PF_CPU_ISA` | auto | force a CPU kernel variant: `scalar` / `avx2` / `avx512` / `avxvnni` |
| `PF_KV_TYPE` | `i8` | KV storage: `i8` / `bf16` / `f16` / `f32` (`PF_KV_F32=1` = f32) |
| `PF_PREFIX_CACHE` | on | `0` disables the cross-request prefix cache |
| `PF_PC_STATES` | 8 | VRAM recurrent-state checkpoints kept |
| `PF_PC_VRAM_MB` | – | VRAM budget (derives `PF_PC_STATES`); `PF_PC_MEM_MB` is an alias |
| `PF_PC_RAM_MB` | 512 | host RAM tier budget (VRAM evictions land here first) |
| `PF_PC_DIR` | – | enable the disk tier and keep its records here (model-scoped) |
| `PF_PC_DISK_MB` | 1024 | disk budget; LRU records beyond it are dropped (`0` = unbounded) |
| `PF_GEMM_DNNL` | on | `0` disables oneDNN int8 GEMM and uses the DP4A path |
| `PF_DP4A` | on | `0` forces the fp32 path |
| `PF_DP4A_DEC` | on | `0` forces fp32 decode |

The full tuning/diagnostics knob list and internals are documented in
[`AGENTS.md`](AGENTS.md).

## Tests

```bash
source /opt/intel/oneapi/setvars.sh
cmake --build build -j
./build/test_tokenizer    # tokenizer round-trips (CPU)
./build/test_multimodal   # image preprocessing, vision encoder (host + device), positions (CPU+GPU)
./build/test_gpu_stages   # every kernel vs the CPU reference (GPU)
./build/test_gemv         # per-tensor GEMV vs CPU dequant reference (GPU)
./build/test_dp4a         # SIn repack + DP4A GEMM vs CPU reference (GPU)
./build/test_gpu_vs_ref   # end-to-end logits vs CPU reference (GPU)
./build/test_forward      # end-to-end logits / top-k (GPU)
./build/test_pc_disk      # disk prefix-cache format / LRU / reopen (CPU)
./build/test_pc_ram       # RAM prefix-cache tier / LRU (CPU)
./build/test_cpu_gemv     # CPU fp32 + int8 GEMV/RMSNorm vs dequant reference (CPU)
./build/test_pc_cpu       # host-backend paged attention + prefix-cache round-trip (CPU)
./build/test_pc_gpu       # prefix-cache disk spill + promote round-trip (GPU)
./build/test_pc_ram_gpu   # prefix-cache VRAM->RAM->VRAM round-trip (GPU)
./build/test_cpuref       # CPU reference head output (CPU)
./build/test_compare      # CPU reference vs llama.cpp dumps (CPU)
```

Each test takes an optional model path as its first argument and otherwise falls
back to a built-in default; the GPU tests need the model and a working device.

## Notes

* Batched decode is functionally correct but currently slower per step than
  batch-1 for batches > 1; prefill and decode are memory-bandwidth bound on
  integrated GPUs.

## License

sycl-infer is released under the MIT License — see [`LICENSE`](LICENSE).

Bundled third-party code (`third_party/`) retains its own license: cpp-httplib,
nlohmann/json, minja and the llama.cpp Unicode tables are all MIT-licensed.  See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the per-component
inventory and full license texts.
