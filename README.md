# sycl-infer — a general-purpose SYCL LLM inference engine

sycl-infer is a from-scratch C++17/SYCL inference engine for quantized LLM
models on Intel GPUs.  It loads GGUF models directly, serves them through an
OpenAI-compatible HTTP API or a CLI, and is built around a **pluggable
architecture registry** so new model types can be added without touching the
engine or the kernel library.

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
* **Quantized compute** — in-house int8 DP4A/SIn kernels, plus an optional
  oneDNN int8 GEMM path for prefill; KV cache stored as int8 (default), bf16,
  f16 or f32.
* **SYCL command graphs** — the full forward step is captured per shape and
  replayed, keeping per-token launch overhead minimal.
* **OpenAI-compatible API** — streaming and non-streaming chat/completions with
  temperature / top-k / top-p / min-p / penalties.

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
* An Intel GPU with SYCL support (developed against Iris Xe-LP)

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
```

Common flags:

| flag | default | meaning |
|---|---|---|
| `--model <gguf>` | – | model file (required) |
| `--ctx N` | 20480 | max sequence length in tokens |
| `--blocks N` | 512 | KV blocks committed at startup (32 tokens each) |
| `--kv-cap-mb N` | auto | upper bound for the dynamically grown KV pool |
| `--host H` / `--port N` | 0.0.0.0 / 8080 | server bind address |
| `--max-tokens` / `--temp` / `--top-p` / `--top-k` | 256 / 0.7 / 0.95 / 40 | `gen` sampling |
| `--raw` | off | `gen`: send the prompt verbatim (no chat template) |

HTTP endpoints:

* `POST /v1/chat/completions` — OpenAI chat API (streaming + non-streaming;
  `stop`, `ignore_eos`, `stream_options.include_usage`)
* `POST /v1/completions` — raw text completion
* `GET /v1/models`, `GET /health`

## Configuration

Runtime behavior is controlled by environment variables.  The most useful ones:

| variable | default | meaning |
|---|---|---|
| `PF_CTX` | 20480 | default for `--ctx` |
| `PF_KV_CAP_MB` | auto | default for `--kv-cap-mb` |
| `PF_KV_TYPE` | `i8` | KV storage: `i8` / `bf16` / `f16` / `f32` (`PF_KV_F32=1` = f32) |
| `PF_PREFIX_CACHE` | on | `0` disables the cross-request prefix cache |
| `PF_PC_STATES` | 8 | recurrent-state checkpoints kept |
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
./build/test_gpu_stages   # every kernel vs the CPU reference (GPU)
./build/test_gemv         # per-tensor GEMV vs CPU dequant reference (GPU)
./build/test_dp4a         # SIn repack + DP4A GEMM vs CPU reference (GPU)
./build/test_gpu_vs_ref   # end-to-end logits vs CPU reference (GPU)
./build/test_forward      # end-to-end logits / top-k (GPU)
./build/test_cpuref       # CPU reference head output (CPU)
./build/test_compare      # CPU reference vs llama.cpp dumps (CPU)
```

Each test takes an optional model path as its first argument and otherwise falls
back to a built-in default; the GPU tests need the model and a working device.

## Notes

* Batched decode is functionally correct but currently slower per step than
  batch-1 for batches > 1; prefill and decode are memory-bandwidth bound on
  integrated GPUs.
