# sycl-infer — a general-purpose SYCL LLM inference engine

> **Experiment project to explore if AI can be used to create an inference engine.**

sycl-infer is a from-scratch C++17/SYCL inference engine for quantized LLM
models on Intel GPUs or the host CPU.  It loads GGUF models directly, serves
them through an OpenAI-compatible HTTP API or a CLI, and is built around a
**pluggable architecture registry** so new model types can be added without
touching the engine or the kernel library.

The reference model used for development and validation is Qwen3.5-0.8B (Q4_K_M)
on Intel Iris Xe-LP.  The throughput numbers in this README are the 27B-class
Qwen3.8-27B-UD-Q4_K_M and the routed-MoE Qwen3.6-35B-A3B-UD-Q4_K_M on two Arc
A770; the engine itself is tied to neither — the model type is selected through
the architecture registry.  Every number, with the conditions it was measured
under, is in **[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)**.

> **This project is entirely AI-generated.**  All of it — kernels, engine,
> server, tests, documentation — was written by AI coding agents.  The author
> knows nothing about GPU optimization: the occupancy cliffs, the
> bytes-per-weight trade-offs and the arithmetic-intensity ceilings this design
> is built around were found by measurement on the hardware named above, not by
> expertise.  Read the code with that in mind, and re-measure before trusting a
> number anywhere else — the tuning constants are per-GPU profiles precisely
> because they do not transfer.

## Features

* **GGUF models** — memory-mapped load, one device blob, no conversion step.
  Weight types Q4_K / Q5_K / Q6_K / Q3_K / IQ4_XS / IQ4_NL / IQ3_S / Q8_0 / F32
  are dequantized on the fly inside the kernels.
* **Pluggable architectures** — the model type is read from
  `general.architecture` and dispatched to a per-architecture loader (one source
  file plus one registry entry); `qwen35` (hybrid Gated DeltaNet + full
  attention) is implemented today.
* **Paged KV cache + continuous batching** — 32-token physical blocks with
  per-sequence block tables; chunked prefill and batched decode (up to 16
  concurrent sequences by default, 32 with `--max-slots`).  Batched decode
  measures 13.4 → 18.9 → 34.4 → **62.5 tok/s aggregate** for 1/2/4/8
  concurrent greedy requests on the 27B and 26.4 → 35.7 → 60.9 → **101.1
  tok/s** on the 35B A3B, because every extra row amortises the same weight
  pass.
* **Cross-request prefix cache** — hashed 32-token blocks with recurrent-state
  checkpoints, so a shared prompt prefix is prefilled only once.  Three LRU
  tiers: device (VRAM) → host RAM → disk.
* **Several models at once** — `sycl-infer-router` runs one worker process per
  model behind a single OpenAI-compatible endpoint, with a per-model VRAM
  budget that is *measured* (the worker publishes its own device footprint) and
  a control plane for the model's identity, its memory and a graceful stop.
  The same router also supervises **ONNX** workers (`"backend": "onnx"`) and
  proxies their binary `/v1/onnx/run` endpoint.
  See [Multiple models: the router](#multiple-models-the-router).
* **Quantized compute** — in-house int8 DP4A/SIn kernels on the GPU and an
  integer int8 GEMV straight from the GGUF blocks on the CPU, plus an optional
  oneDNN int8 GEMM path for prefill; KV cache stored as int8 (default), 4-bit,
  bf16, f16 or f32, and `--kv-type K:V` sizes the K and V caches independently
  (e.g. `--kv-type i4:i8` = 4-bit keys, 8-bit values, 24 KB/token against i8's
  32 at near-i8 accuracy).
* **Native 4-bit (u4) weights** — Q4_K keeps the GGUF's own `(q, step, offset)`
  grid, which is lossless (0.077% vs 0.98% relative L2 for the int8 conversion)
  and 1.6x smaller, so a 27B-class Q4_K model fits two 16 GB cards.  `PF_W4=0`
  restores the pure int8 path.
* **Native 5-bit (Q5_K)** — Q5_K's grid is `q5 = lo4 | (hi1 << 4)` with a
  per-32 `(step, offset)` pair, so the engine keeps the 4-bit nibble plane plus
  a separate 1-bit plane (0.75 B/weight against int8's 1.125) and recombines
  them with one OR in the decode GEMV.  Q5_K is 35% of a UD-Q4_K_M model's
  weights, so this is the largest single per-token byte saving available, and
  the native values are kept exactly.  Worth ~+6% tg128 and 2.1 GB/card of
  device memory; `PF_K5=0` disables it.
* **Codebook 4-bit (IQ4_XS / IQ4_NL)** — these are a 16-entry int8 codebook times
  a per-32 f16 scale, so the engine stores the 4-bit indices plus that scale
  (0.5625 B/weight instead of int8's 1.0625) and expands the table in the decode
  GEMV; prefill expands each tensor to int8 in a small reused scratch.  The
  native values are kept exactly.  Worth ~+5% tg128 and 2.5 GB/card of device
  memory; `PF_CB4=0` disables it.
* **Qwen3.8-Flash-Next (`qwen4exp`)** — a 104 GiB, 4-shard hybrid with
  hyper-connections, a per-layer n-gram embedding (PLE), QSA sparse attention
  and 512 routed experts.  The dense trunk is ~5.3 GiB; the 71.7 GiB expert
  bank and 26.8 GiB PLE table stay in host RAM, with a 16.5 GiB per-expert LRU
  VRAM bank (121 slots x 47 layers, including the four Q8_0-down layers), a
  pinned 69.8 GiB expert source (`PF_Q4_PIN_SRC`) and a host-RAM cache for PLE
  rows (`--engram-ram-mb`).  Written up in [Design 18](docs/design/18-qwen4exp.md).
* **Device selection** — `--device cpu|gpu|auto`; the CPU backend has its own
  AVX2 / AVX-VNNI / AVX-512 kernels, picks the variant at run time, and keeps
  the paged KV in host RAM.
* **Multi-device (pipeline parallel)** — `--layer-map 0-11:gpu,12-23:cpu` places
  contiguous layer ranges on devices; each device owns the paged KV of its
  attention layers.  A 27B-class model needs this (it does not fit on one card);
  see [Multi-GPU](#multi-gpu-pipeline-parallel) below.
* **SYCL command graphs** — the full forward step is captured per shape and
  replayed, keeping per-token launch overhead minimal (GPU only).
* **OpenAI-compatible API** — streaming and non-streaming chat/completions with
  temperature / top-k / top-p / min-p / penalties, `logit_bias`,
  `logprobs`/`top_logprobs`, `reasoning_content` (thinking models),
  function/tool calling (`tools`, `tool_calls`, `tool` results), `n` choices,
  `echo`/`suffix`/`best_of` and `/v1/models`; `usage` reports prefix-cache
  `cached_tokens` and `reasoning_tokens` details.
* **Multimodal input** — Qwen3.5 images through a `clip` mmproj GGUF, plus
  video (uniformly sampled frames) and audio (a separate AuT-style tower GGUF),
  freely mixed in one prompt; the vision and audio encoders run on the GPU
  (`src/backend/gpu/kernels/vit.cpp`, `at.cpp`).  The CLI takes `--mmproj`
  `--image` / `--video` / `--audio` + `--audio-mmproj`, and the server accepts
  `image_url` / `video_url` / `input_audio` content parts (base64 `data:` URLs or
  remote `http(s)://` URLs, bounded to 10 s / 10 MB).  Media tokens carry 4-section
  M-RoPE positions and bypass the prefix cache.
* **MTP speculative decoding (`--mtp N`)** — drafts up to N tokens with the
  model's own NextN head and verifies them in one batched forward.  Measured
  **1.35x - 1.9x** single-request decode on the 27B depending on how much the
  prompt's own distribution supports the draft (see
  [Performance](#performance)); the emitted stream is greedy-equivalent.
  Requires a GGUF with a `blk.<n>.nextn.*` head *and* a
  multi-device oneDNN int8 partition, and it does not compose with batching, so
  the server keeps it off by default.
* **DFlash2 block drafter (`--spec-type dflash2`)** — a second speculative
  drafter, structurally different from MTP: the draft GGUF's own 5-layer head runs
  **one non-causal forward** over `[anchor, MASK x (n_max)]` and a selector scores
  the top-K candidate sets per block position plus every ordered pair, so the host
  walks one coherent path through the lattice instead of re-drawing per token.
  Measured **1.3x-3.1x** single-request decode on the 27B at `n_max=5` (21.8-77.0
  vs 67-71 ms/token plain), the spread being *acceptance* and nothing else: 4.25
  drafts/cycle on a code prompt, 2.19 on a technical one, and 0.49 on a story opener
  where the drafter is **slower than no drafter at all** (0.92x).  Check acceptance
  per workload before enabling it.  Needs a
  matching draft GGUF via `--spec-draft-model`.  **Note:** those numbers are
  from 2026-10-09 and the path does not run in the current build — the verify's
  USM copy segfaults inside the driver (see
  [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md#62-dflash2--the-block-drafter)).
  See [Design 14](docs/design/14-dflash2.md).
* **Per-GPU tuning profiles** — every launch constant that was measured on a
  particular card lives in that card's profile (`src/device/profiles/<card>.cpp`,
  selected from the reported device name), so an unknown GPU warns loudly instead
  of silently inheriting another card's tuning.

## Supported models

The architecture is read from the GGUF's `general.architecture` and dispatched
to a per-architecture loader; the engine itself is not tied to any model family.

| model | `general.architecture` | status | notes |
|---|---|---|---|
| Qwen3.5-0.8B (Q4_K_M) | `qwen35` | supported | hybrid Gated DeltaNet + full attention; the reference model for the test suite |
| Qwen3.8-27B (UD-Q4_K_M) | `qwen35` | supported | dense, 64 layers; 15.4 GiB of weights, so it needs the two-card split |
| Qwen3.6-35B-A3B (UD-Q4_K_M) | `qwen35moe` | supported | routed MoE, 256 experts / 8 active, 40 layers; 21.1 GiB of weights, so it needs the two-card split too |
| Qwen3.8-Flash-Next (UD-Q4_K_XL, 4 shards) | `qwen4exp` | **supported** | hyper-connections + n-gram PLE + 512-expert MoE + QSA.  The routed bank and PLE table are host-resident (with a pinned 69.8 GiB source mirror); the dense trunk plus a 16.5 GiB per-expert LRU bank (47 layers, Q8_0-down layers included) run on the GPUs.  Recommended: `--layer-map 0-23:gpu.0,24-47:gpu.1 --moe-vram-mb 17500 --ctx 2048 --kv-type f16`.  See [Design 18](docs/design/18-qwen4exp.md) |

| | |
|---|---|
| file format | GGUF (memory-mapped; single file **or split shards**, merged into one virtual mapping) |
| weight quantization | Q4_K, Q5_K, Q6_K, Q3_K, Q5_1, IQ4_XS, IQ4_NL, IQ3_S, Q8_0, F16, BF16, F32; Q4_K uses the native 4-bit (u4) path by default |
| architecture registry | selected via `general.architecture`; `qwen35`, `qwen35moe` (routed MoE) and `qwen4exp` (Qwen3.8-Flash-Next) implemented, others plug in via `src/model/model_arch.h` + a `kLoaders[]` entry |
| expert placement | routed-MoE models can keep part of the experts on the GPU and compute the rest on the host per step (llama.cpp's ncmoe), with a disk → host-RAM → VRAM three-tier expert cache (`--moe-resident` / `--moe-vram-mb` / `--moe-ram-mb`) |
| KV cache dtype | `i8` (default), `i4`, `bf16`, `f16`, `f32`; `--kv-type K:V` sizes K and V independently (`i4:i8`) |
| tokenizer | GPT-2 byte-level BPE from the GGUF vocabulary + special tokens / chat template |

Adding an architecture is a single new file under `src/model/` — see
[`AGENTS.md`](AGENTS.md#add-a-model-architecture).

### Best measured performance, at context depth 0

Two Arc A770, AOT Release, i8 KV, prefix cache off; the full tables — every
context depth, every batch size, both speculative paths — are in
**[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)**.

| model | placement | prefill (512 tokens) | decode, one stream | decode, 32 concurrent streams |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | 1 card | **6 336 tok/s** | **78.7 tok/s** (12.7 ms/token) | – |
| Qwen3.8-27B | 2 cards | **507 tok/s** | **13.4 tok/s** (74.6 ms/token) | **76.5 tok/s** |
| Qwen3.6-35B-A3B | 2 cards | **779 tok/s** | **26.4 tok/s** (37.9 ms/token) | **168 tok/s** |
| Qwen3.8-Flash-Next | 2 cards, LRU expert bank, pinned source | **47-56 tok/s** (~11.4 s cold / 9.6 s warm) | **10.7 tok/s** (~93.3 ms/token) | – |

The first two columns are the *isolated* prefill of a 512-token prompt and a
single-stream decode, measured in-process; the last column is the aggregate
throughput of 32 concurrent streams, which is what a server delivers.  The
qwen4exp row uses i8 KV for its prefill measurement and f16 for decode; both
types work.  Through the server the 27B's 512-token prompt takes **1.46 s to
its first token** (351 tok/s): the difference is the HTTP round trip, the chat
template, the lazily grown KV pool and the first token's own head — see
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Requirements

* Linux, C++17, CMake ≥ 3.20
* Intel oneAPI DPC++ compiler (`icpx`) — tested with oneAPI 2026.1
* oneDNN 2026.0 (optional prefill path; `/opt/intel/oneapi/dnnl/2026.0`,
  override with `-DDNNL_ROOT=`)
* An Intel GPU with SYCL support (developed against Iris Xe-LP; 27B-class models
  were run on two Arc A770).  The CPU backend (`--device cpu`) runs without a
  GPU, using the AVX2 / AVX-VNNI / AVX-512 kernel variants selected at run time.
* Two or more Intel GPUs for the pipeline-parallel split (`--layer-map`, see
  [Multi-GPU](#multi-gpu-pipeline-parallel)) - `sycl-ls` lists what is visible.

## Build

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build
cmake --build build -j
```

Optional: AOT-compile the device image with `-DSYCL_INFER_AOT=ON` (defaults to
`adl-p`; edit `CMakeLists.txt` for another target).

The **router** (the multi-model front end) builds on its own, without SYCL,
oneDNN or even `icpx`:

```bash
cmake -S . -B build-router-only -DSI_ROUTER_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-router-only -j
```

That is the container image's build path, and it is what proves the router
needs neither a GPU nor the oneAPI toolchain.

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

# a 27B-class model does not fit on one card: split the layers over two GPUs
./build/sycl-infer --model /path/to/Qwen3.8-27B-UD-Q4_K_M.gguf --ctx 4096 \
    --layer-map 0-31:gpu.0,32-63:gpu.1 gen --prompt "Hello"

# Qwen3.8-Flash-Next: all-GPU trunk + per-expert LRU, host-resident experts/PLE
./build/sycl-infer \
    --model /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    --layer-map 0-23:gpu.0,24-47:gpu.1 --moe-vram-mb 17500 \
    --ctx 2048 --kv-type f16 gen --prompt "Hello"
```

Common flags:

| flag | default | meaning |
|---|---|---|
| `--model <gguf>` | – | model file (required) |
| `--mmproj <gguf>` | – | vision projector; required for `--image` / `--video` |
| `--audio-mmproj <gguf>` | – | audio tower; required for `--audio` |
| `--image` / `--video` / `--audio <file>` | – | `gen`: attach media (each repeatable, freely mixable) |
| `--max-video-frames N` / `--max-video-side N` | 16 / 768 | frames sampled per video, and the frame-side cap |
| `--ctx N` \| `full` | 20480 | max sequence length in tokens; `full` takes `context_length` from the GGUF |
| `--blocks N` | 512 | KV blocks committed at startup (32 tokens each); grows lazily |
| `--kv-cap-mb N` | auto | upper bound for the dynamically grown KV pool; also caps the sum of the three cache tiers (see below) |
| `--kv-type T` \| `K:V` | `i8` | KV storage: `i4` / `i8` / `bf16` / `f16` / `f32`, or K and V sized independently (`i4:i8`) |
| `--device cpu\|gpu\|auto` | auto (gpu) | compute backend; `auto` reads `PF_DEVICE` |
| `--cpu-threads N` | physical cores | CPU backend worker threads (`0` = auto) |
| `--layer-map L:dev,...` | – | pipeline-parallel layer placement, e.g. `0-31:gpu.0,32-63:gpu.1`; qwen4exp recommends `0-23:gpu.0,24-47:gpu.1` |
| `--moe-resident N` | 0 | routed-MoE: experts kept on the GPU per layer (`0` = all resident) |
| `--moe-vram-mb MB` | 0 | routed-MoE: device budget for the expert bank; qwen4exp's per-expert LRU uses **17500** |
| `--moe-ram-mb MB` | 0 | routed-MoE: host budget for the expert-block tier (`0` = off); qwen4exp keeps its experts host-resident and does not use this |
| `--engram-ram-mb MB` | 4096 | qwen4exp: PLE row cache in host RAM (`0` disables) |
| `--spec-type T` | `none` | speculative decoding: `none` / `mtp` / `dflash2` |
| `--spec-draft-model <gguf>` | – | DFlash2 drafter GGUF (`--spec-type dflash2`); it reads the target's hidden states at the layers its `dflash.target_layers` names and proposes up to `dflash.block_size`-1 tokens |
| `--spec-draft-n-max N` | 5 | draft tokens per cycle (DFlash2); the measured optimum |
| `--spec-draft-device N` | 0 | device partition the draft runs on |
| `--mtp [N]` | 4 | alias for `--spec-type mtp --spec-draft-n-max N` |
| `--mtp-device N` | 0 | alias for `--spec-draft-device` |
| `--pc-vram-mb N` | – | VRAM (device) prefix-cache budget, converted to a checkpoint count |
| `--pc-ram-mb N` | 512 | host-RAM prefix-cache budget (`0` disables the RAM tier) |
| `--pc-dir DIR` | – | enable the disk prefix-cache tier in `DIR` (model-scoped) |
| `--pc-disk-mb N` | 1024 | disk prefix-cache budget (`0` = unbounded) |
| `--pc-mem-mb N` | – | alias for `--pc-vram-mb` (or set `PF_PC_STATES` directly) |
| `--host H` / `--port N` | 0.0.0.0 / 8080 | server bind address |
| `--max-tokens` / `--temp` / `--top-p` / `--top-k` | 256 / 0.7 / 0.95 / 40 | `gen` sampling |
| `--raw` | off | `gen`: send the prompt verbatim (no chat template) |
| `--thinking` | off | `gen`: set the chat template `enable_thinking` (reasoning on); `--enable-thinking` is an accepted alias; the server takes this from each request instead |
| `-h` / `--help` | – | print the built-in flag list and exit 0 |

`--mtp` is hard-gated: it needs a GGUF that bundles the NextN head
(`blk.<n>.nextn.*`) *and* a multi-device oneDNN int8 partition (`--layer-map` over
GPUs with `PF_DP4A` on).  If either is missing the engine prints one `[mtp]` line
and stays on the plain decode; the draft length is also capped at 12.

`--spec-type dflash2` needs a matching draft GGUF (`--spec-draft-model`) that
carries the block-drafter tensors (`dflash.block_size`, `dflash.target_layers`);
without one the engine stays on the plain decode.  It has no multi-device
partition requirement — the draft runs on `--spec-draft-device`.

HTTP endpoints:

* `POST /v1/chat/completions` — OpenAI chat API (streaming + non-streaming;
  `stop`, `ignore_eos`, `stream_options.include_usage`, `n`, `reasoning_content`
  (`chat_template_kwargs.enable_thinking` / `enable_thinking` / `thinking` /
  `reasoning_effort`),
  `tools`/`tool_choice` + `tool_calls`/`tool` results, `logit_bias`, `logprobs`/
  `top_logprobs`, image parts)
* `POST /v1/completions` — raw text completion (string/string[]/token-id prompt,
  `n`, `echo`, `suffix`, `best_of`, `logit_bias`, `logprobs`,
  `stream_options.include_usage`)
* `GET /v1/models` + `GET /v1/models/{id}`, `GET /health`

### Embedding the engine: C API and Python

The engine can also be linked directly into a product, without HTTP or a
subprocess:

* **C ABI** (`libsycl_infer`) — opaque model/request handles, async submit +
  pull streaming, automatic batching with a configurable admission window and
  a FIFO queue, per-request cancellation/timeout, and thread-safe submission
  from many host threads.  See the design in
  [`docs/design/19-c-api.md`](docs/design/19-c-api.md) and the usage guide in
  [`docs/guides/c-api.md`](docs/guides/c-api.md).
* **Python** (`sycl_infer`) — a pybind11 binding with blocking, streaming and
  batch APIs, GIL release for threaded workloads, and a wheel that vendors the
  SYCL/oneDNN/oneAPI runtime.  See
  [`docs/design/20-python-binding.md`](docs/design/20-python-binding.md) and
  [`docs/guides/python.md`](docs/guides/python.md).

Both are a thin layer over the same `engine`/`scheduler`: the C API uses the
existing continuous-batching scheduler, so concurrent requests share one weight
pass while keeping separate KV slots, sampler state, stop strings and
cancellation.

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

### Expert cache and the ncmoe split (routed-MoE models)

A MoE model's routed experts dominate its weight bytes (18.2 GiB of the 35B A3B),
and a token only ever activates `n_expert_used` of `n_expert`.  The engine can keep
part of them on the GPU and compute the rest on the host each step, the way
llama.cpp's ncmoe does, and keep them cached across three tiers:

```bash
# keep ~55 experts/layer on the GPU (a 100 MB bank), cache 150 MB of expert
# blocks on the host, and leave the rest on disk
./build/sycl-infer --model Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --layer-map 0-19:gpu.0,20-39:gpu.1 \
    --moe-vram-mb 100 --moe-ram-mb 150 gen --prompt "Q: What is 2+2? A:" --max-tokens 8

[moe] expert cache: 55 of 256 experts resident per layer, RAM tier 150 MB
[moe] expert cache: calibration done, 2255 of 10496 (layer, expert) pairs resident on the device
[moe] expert cache: bank 55/256 resident per layer (calibrated 2255 pairs), RAM tier 149.8 MB hits=2559 misses=2738 (48.3%)
```

* The resident set is **measured, not indexed**: one request's own routing fills a
  frequency table and the bank is frozen from it, so `--moe-vram-mb` is a byte
  budget rather than a count.
* The host tier holds the selected experts' GGUF blocks, keyed by (layer, expert)
  with an LRU, so the host half stops re-faulting pages that
  `release_host_weight_pages()` deliberately dropped.
* Correctness is oracle-checked, not eyeballed: with the split armed the logits are
  **bit-identical** to a fully-resident run (`max|diff|=0.125359`, argmax SAME), and
  `test_decode_vs_prefill` is 8/8.  Arming the split disables the decode command
  graph in all three places one is replayed - a recorded graph cannot hold a host
  pass whose expert set is chosen per step.
* Limit: with the split armed MTP is held at `k=1` (the verify's later rows are
  verified at that depth only).  Use a fully-resident expert bank for `--mtp 4`.

### Multi-GPU (pipeline parallel)

`--layer-map` splits the model into contiguous layer ranges and runs each range
on its own device.  Every device holds only its own layers' weights and owns the
paged KV of the attention layers in its range; at each range boundary the hidden
state is handed to the next device.  This is what makes a 27B-class Q4_K model
fit on two 16 GB cards.

```
--layer-map <begin-end>:<device>[,<begin-end>:<device>...]
    <device> = gpu | gpu.N | cpu
```

* `begin-end` is an **inclusive** layer range; entries are applied in order and
  must tile `[0, n_layer)` with no gaps (otherwise startup fails with
  `--layer-map must cover every layer without gaps` / `... must end at the model's
  layer count`).
* `gpu` is the first enumerated GPU, `gpu.N` the N-th one in
  `sycl::device::get_devices(sycl::info::device_type::gpu)` order (the order
  `sycl-ls` prints).  `cpu` (or `host`) puts that range on the AVX host backend,
  so GPU + CPU splits work too.
* Backend 0 - the first GPU in the map, or the CPU when the map has no GPU - runs
  the global tensors and the LM head.  `--device` is not needed: the map fully
  determines placement.  Distinct GPUs in the map share one SYCL context, so the
  host-USM activations stay valid across devices.

```bash
# 27B on two Arc GPUs: layers 0-31 on the first, 32-63 on the second
./build/sycl-infer --model Qwen3.8-27B-UD-Q4_K_M.gguf --ctx 4096 \
    --layer-map 0-31:gpu.0,32-63:gpu.1 gen --prompt "Hello"

# split a small model between one GPU and the CPU
./build/sycl-infer --model model.gguf --layer-map 0-11:gpu,12-23:cpu serve
```

The startup log reports what a map resolved to - the quickest way to check the
wiring:

```
[dev] profile=arc_a770  (Intel Arc A770 (DG2))
[dev] backend 0 -> gpu.0 (Intel(R) Arc(TM) A770 Graphics)
[dev] backend 1 -> gpu.1 (Intel(R) Arc(TM) A770 Graphics)
[dev] device 0 weights: 59 u4, ... k5, ... codebook, ... int8, ... MiB on device
[dev] multi-device weight path: oneDNN int8 matmul (jit:gemm:any)
[dev] multi-device layer map: 0-31:gpu.0,32-63:gpu.1 (2 backends)
[md] decode command graphs: 2 phase(s) over 2 backend(s)
```

(the per-store counts and MiB are elided here; they are printed per device and
name which native stores that partition got)

Notes:

* every device needs room for its share of the weights **plus its own paged KV**
  for the attention layers it holds, so VRAM grows with `--ctx` on each device
  (for the 27B above: ~12.4 GB of weights and ~17 KB per token per card);
* each device records its own SYCL command graph for the decode step and the two
  are replayed with the hidden-state handoff between them; the per-device weight
  path (oneDNN int8 matmul / DP4A SIn / native u4 / fp32) is selected
  automatically;
* continuous batching, the prefix cache and the OpenAI API work unchanged.

## Multiple models: the router

`sycl-infer-router` is a separate front end that serves **several models at
once**: one worker process per model, one OpenAI-compatible endpoint in front
of them.  The request's `model` field picks the worker.  LLM workers serve
the OpenAI API; ONNX workers (`"backend": "onnx"`) serve the binary
`/v1/onnx/run` API, which the router proxies from the same process.

It is a plain host program - it links neither SYCL nor oneDNN - so its container
is ~130 MB, it needs no GPU to build, and its own tests are hermetic.

```bash
cmake -S . -B build-router-only -DSI_ROUTER_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-router-only -j$(nproc)

./build-router-only/sycl-infer-router --config router.json --check   # validate
./build-router-only/sycl-infer-router --config router.json          # serve
```

```jsonc
{
  // JSON with comments: `//` and `/* */` are stripped by the parser.
  "router": { "listen": "0.0.0.0:8080", "device_vram_mb": 15541 },
  "models": [
    {
      "id": "qwen35-0.8b",                        // the client's `model` field
      "path": "/models/Qwen3.5-0.8B-Q4_K_M.gguf",
      "http": "127.0.0.1:18081",                  // the worker's OpenAI port
      "control": "127.0.0.1:18082",               // its control plane (RPC)
      "control_token_file": "/run/secrets/ctl",   // the control plane's secret
      "args": ["--ctx", "20480"],
      "device": { "group": "gpu.0", "filter": "level_zero:0", "count": 1 },
      "memory": {
        "vram_budget_mb": 4096,                    // the router's admission budget
        "kv_cap_mb": 2048,                         // -> --kv-cap-mb
        "max_slots": 8,                            // -> --max-slots
        "prefix_cache_vram_mb": 512                // -> --pc-vram-mb
      }
    }
  ]
}
```

What it does:

* **one worker process per model**, started, watched and restarted by the
  router; the models of one device start one at a time, in configuration
  order, so two models never fight over the same card's memory;
* **per-model VRAM budget**: the router admits a model against what the card
  has *measured* free (the worker publishes its own device footprint over the
  control plane) and refuses one that does not fit - with the measured
  figure, not a guess:
  `the device group 'gpu.0' has 9874 MB free, and this model needs 12288 MB`;
* **the OpenAI API unchanged**: `/v1/chat/completions`, `/v1/completions`,
  `/v1/models`, streaming, `reasoning_content`, tool calls, logprobs - the
  router proxies them, it does not reimplement them;
* **a client that disconnects cancels the generation**, across the extra hop;
* **a worker that cannot start is not retried**: the worker exits with 78
  (`EX_CONFIG`) and says so; the same attempt would fail the same way;
* **workers on other machines**: a model with a `remote` entry is connected to
  instead of started.

The VRAM figure the router admits against is the **model's own accounting**
(`total - what this model put on the card`), not the driver's: the Level Zero
backend either does not implement the free-memory aspect or reports a number
that does not count the SYCL allocations (it said 13 GB free on a card holding
9 GB of weights).  The model-side figure agrees with the hand-computed
capacity table in AGENTS.md.

The worker's control plane is optional but recommended:

```bash
./build/sycl-infer --model model.gguf --port 8081 --control-port 8082 \
    --control-token-file /run/secrets/control-token serve
```

It adds a JSON-RPC channel (model identity, device memory, graceful stop).
It takes a shared secret, presented once in the handshake; use the **file**
form so the secret never appears on a command line.  Without a secret the
control plane is open, so it must stay on a private network.

The router presents the same secret (`models[].control_token` /
`models[].control_token_file`); if it does not match, the router says so and
falls back to the OpenAI API - the model still serves, it just has no device
report.

Containers: `docker/Dockerfile.router` (the router alone), `docker/Dockerfile.worker`
(a worker), `docker/Dockerfile.all` (both, for the "the router starts the
workers" topology) and `docker/compose.yaml`.  The full design is in
[docs/design/16-router.md](docs/design/16-router.md).

## The document pipeline: the ONNX model service

`sycl-infer-onnx` is the other half of the engine: a service that runs the
**ONNX** models a document pipeline needs - the layout detector, the OCR
detector and recogniser, the table classifiers and structurers, the formula
recogniser.  The runtime is this project's own: an ONNX protobuf reader, a
graph IR with a fixed-point pipeline of independent transform passes
(`src/model/onnx/transforms/`), an operator library, a SYCL kernel library and
an executor that plans the memory - **all ahead-of-time compiled**, so there
is no per-shape kernel compilation and no runtime JIT.

The source follows the project's layering rather than a private tree:
`src/model/onnx/` is the graph and its transforms, `src/backend/onnx/` is the
operator/kernel/executor backend, `src/server/onnx/` is the service and
`src/common/onnx_wire.h` is the binary protocol.  It is still a separate
binary because it is a different workload, and because it can run without the
LLM engine.

```bash
cmake -S . -B build -DSI_ONNX=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sycl-infer-onnx -j$(nproc)

./build/sycl-infer-onnx --config onnx_models.json --port 8890
```

The client speaks a small binary protocol over HTTP and returns numpy arrays
(the document pipeline's `rapid_doc/model/http_infer.py`), so each model can be
swapped for the service without touching the code that calls it.

The router supports the ONNX backend as `"backend": "onnx"`: one
`sycl-infer-onnx` worker per model, the same `--device cpu|gpu|auto` and
`SYCL_DEVICE_FILTER` card pinning as the LLM workers, and a binary
`POST /v1/onnx/run` proxy.  See
[`docs/design/16-router.md`](docs/design/16-router.md) and the example in
[`docker/router.example.json`](docker/router.example.json).

**The state**: all twelve of the pipeline's models load, and the eight that
have been verified are **either exact or within the float32 accumulation** -
three of the captured corpus's seven are integer-exact, four differ in one or
two elements of millions by one unit, and the formula recogniser (five real
formula crops) is **integer-equal** to ONNXRuntime.  (The formula model is not
in the captured corpus: the captured pages have no formulas.)

**The performance** (one Arc A770, one 800x800 page, the layout detector):
**129-141 ms of service call**, of which the convolutions are 93-103 ms, the
top-k **4**, the matmuls 9 (bias and the fused activation included) and the
Concats 5.5.  The mask-to-box head - six reductions and ten 48 MB intermediates in
the model as exported - is one 0.2 ms scan now.  The service's call is
129-141 ms at batch 1 and 146 ms per page at batch 2; the pipeline drops the
48 MB mask output (`drop_outputs`), so it never crosses the wire.  The full
table, the conditions and the dead ends are in
[`docs/design/17-onnx-service.md`](docs/design/17-onnx-service.md).

## Performance

Measured on **two Arc A770 16 GB** (i9-10940X, driver 12.55.8), an AOT Release
build, i8 KV, the prefix cache off.  Every number here is in
**[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)** together with the harness, the
exact command and the date; this section is the summary and the caveats.

### At a glance, at context depth 0

| model | placement | prefill (512 tokens) | decode, one stream | decode, 32 streams |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | 1 card | **6 336 tok/s** | **78.7 tok/s** (12.7 ms/token) | – |
| Qwen3.8-27B | 2 cards | **507 tok/s** | **13.4 tok/s** (74.6 ms/token) | **76.5 tok/s** |
| Qwen3.6-35B-A3B | 2 cards | **779 tok/s** | **26.4 tok/s** (37.9 ms/token) | **168 tok/s** |
| Qwen3.8-Flash-Next | 2 cards, LRU expert bank, pinned source | **47-56 tok/s** (~11.4 s cold / 9.6 s warm) | **10.7 tok/s** (~93.3 ms/token) | – |

**Qwen3.8-Flash-Next is the exception in both columns**: its file is 104 GiB,
of which the 512-expert routed bank is 71.7 GiB and the n-gram PLE table 26.8
GiB; neither fits on two 16 GB cards.  The ~5.3 GiB dense trunk and a per-expert
LRU expert bank (`--moe-vram-mb 17500`) live on the two cards: 121 slots in each
of 47 layers, including the four Q8_0-down layers 4, 30, 46 and 47.  The expert
planes are pinned in host RAM once (`PF_Q4_PIN_SRC`, 69.8 GiB) so the uploads
read them directly.  Only layer 2's routed experts stay on the host (Q5_K
gate/up + Q8_0 down); its decode-only bank (`PF_Q4_BANK_Q5K=1`, 48 layers,
17.0 GiB) measured 107.2 ms against 93.3, so the flag stays off.  The
recommended run points the loader at the first shard:

```bash
./build/sycl-infer \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --layer-map 0-23:gpu.0,24-47:gpu.1 --moe-vram-mb 17500 \
  --ctx 2048 --kv-type f16 gen --prompt "The capital of France is"
```

The 512-token prefill is **~11.4 s cold / ~9.6 s warm (47-56 tok/s)**; the
cold number includes the bank's calibration.  Decode settles at **~93.3
ms/token (10.7 tok/s)** after ~16 tokens; a 64-token generation is ~15.6 s wall
including the warm prefill.  17 000 MB gives 118 slots and 94.0 ms; 18 000
loads (125 slots, 17.0 GiB) but is not faster.  `PF_Q4_PIN_SRC=0` (staging) and
`PF_Q4_BANK_Q5K=1` (layer 2, 48 layers, 17.0 GiB, 107.2 ms, so off) are the
A/Bs.
The 512-token prefill is the archive i8 cell and the decode is f16; the
deep-context cells are f16 (both KV types work).  `--device cpu` and a plain
`--device gpu` also work, but are slower.

**Known limits**: batched q4exp decode is **correct but 0.85x** (a 2-row forward
touches ~20 distinct experts per layer instead of ~10), so the scheduler caps it
at one row and serialises sequences; MTP is **no-go** for this build
(`test_q4_verify` puts the k+1 verify at 1.9-2.2x the kill line), and the separate
draft GGUF is not wired.  QSA is implemented: up to 2 048 keys it is
dense-equivalent, above that the host selector picks the top 2 048 pooled keys
and the device attends to those.  Full attribution, deep-context numbers and
dead ends: [Design 18](docs/design/18-qwen4exp.md),
[PERFORMANCE §4.4](docs/PERFORMANCE.md#44-qwen38-flash-next), [`AGENTS.md`](AGENTS.md).

The prefill column is the isolated 512-token chunk, measured in-process; the
decode columns are the aggregate over N concurrent streams, i.e. what a server
delivers to N clients.  **The routed-MoE model decodes faster than the dense one
although it reads more weights per token** — 8 of its 256 experts are active per
token, so a decode step touches a fraction of its 21 GiB.

### By context depth

| depth | 27B | 35B A3B |
|---:|---:|---:|
| 512 | 13.41 tok/s (74.6 ms/step) | 26.38 tok/s (37.9 ms/step) |
| 16 384 | 12.26 tok/s (81.6 ms/step) | 24.54 tok/s (40.8 ms/step) |
| 65 536 | 9.70 tok/s (103.1 ms/step) | 20.20 tok/s (49.5 ms/step) |
| 131 072 | – | 15.54 tok/s (64.3 ms/step) |

* **The decode's cost is linear in the depth, with no knee.**  The 27B's step
  is `74.3 + 0.439 us * depth` ms and the 35B's is `37.8 + 0.202 us * depth`
  ms: the constant is the weight pass (the whole step with an empty KV) and
  the slope is the attention over the KV.  The 27B's slope is 2.2× the 35B's
  because its 16 attention layers have head_dim 256 over 4 kv heads, so each
  kv head is re-read six times per layer.
* **The prefill's average rate peaks at ~16k of prompt and falls off after
  it** — the prompt attends to its own KV: 476 tok/s for a 384-token prompt
  and 984 for a 16 256-token one on the 27B; 745 and 1 477 on the 35B.

### By batch size, at depth 512

| N | 27B | 35B A3B |
|---:|---:|---:|
| 1 | 13.41 tok/s | 26.38 tok/s |
| 2 | 18.93 tok/s | 35.70 tok/s |
| 4 | 34.36 tok/s | 60.91 tok/s |
| 8 | **62.48 tok/s** | 101.12 tok/s |
| 16 | **45.20 tok/s** | 122.05 tok/s |
| 32 | 76.45 tok/s | **168.33 tok/s** |

* **Every extra row amortises the same weight pass, so the aggregate grows —
  but not always.**  The 27B's aggregate *drops 28 %* from 8 rows to 16: above
  13 rows the dense layers leave the native u4 GEMV (`nat_gemm_launch`, which
  refuses M > 13) for oneDNN's int8 matmul, and the step cost jumps from
  128 ms to 354 ms.  The 35B crosses the same boundary far more gently,
  because only its dense layers are affected.
* **The prefill saturates at 16 rows and the decode does not**: on the 35B,
  +2.5 % of prefill for twice the concurrency against +36 % of decode.

### Speculative decoding

`--mtp` (the model's own NextN head) and `--spec-type dflash2` (a separate
block drafter) cut the per-token cost of a **single-request** decode.  Both are
gated: MTP needs a GGUF with `blk.<n>.nextn.*` and a multi-device oneDNN int8
partition, DFlash2 needs a matching draft GGUF.

**Acceptance, not the engine, is what moves the speedup.**  The cycle cost is
nearly the same across prompts; what changes is how many drafted tokens the
head gets right, and the ceiling on the speedup is `1 + acceptance`.  On the
story prompt the DFlash2 drafter accepts 0.49 drafts per cycle and is
**slower than not drafting at all** — so measure the acceptance on your own
prompts before enabling either path.  Neither composes with batching (the
server keeps them off by default), and the full tables are in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md#6-speculative-decoding).

| path (27B, one stream) | technical | story | code |
|---|---:|---:|---:|
| plain decode | 69.7 ms/token | 69.7 | 69.7 |
| MTP, k=4 | 39.3 ms/token (1.78x) | 52.6 (1.32x) | 36.5 (1.91x) |
| DFlash2, n_max=5 | 35.8 ms/token (1.88x) | 77.0 (**0.92x**) | 21.8 (3.08x) |

The MTP row is from 2026-10-20; the DFlash2 row is from 2026-10-09, and the
DFlash2 path **does not run in the current build** (the verify's USM copy
segfaults inside the driver) — see
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md#62-dflash2--the-block-drafter)
before trusting a number from it.

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
| `PF_ATTN_XMX` | on | prefill attention through oneDNN int8 matmuls; `0` restores the classic kernel (faster below ~2048 keys) |
| `PF_DP4A` | on | `0` forces the fp32 path |
| `PF_DP4A_DEC` | on | `0` forces fp32 decode |
| `PF_MOE_RESIDENT` | 0 | experts kept on the GPU per layer (`--moe-resident`); `0` = all resident |
| `PF_MOE_VRAM_MB` | – | device budget for the expert bank (`--moe-vram-mb`); the bank holds the measured hot set |
| `PF_MOE_RAM_MB` | – | host budget for the expert-block tier (`--moe-ram-mb`); `0` = off |
| `PF_W4` | on | native 4-bit (u4) weights for Q4_K; `0` falls back to pure int8 |
| `PF_CB4` | on | store IQ4_XS/IQ4_NL as native codebook 4-bit; `0` keeps int8 |
| `PF_K5` | on | store Q5_K as the native 5-bit (nibble + bit) planes; `0` keeps int8 |
| `PF_MTP` | – | draft length for `--mtp` (overrides the flag) |
| `PF_MTP_SERVER` | off | `1` routes greedy server requests into the single-sequence MTP loop instead of the batching scheduler |
| `PF_DFLASH_NMAX` | 5 | draft tokens per cycle for `--spec-type dflash2` (overrides `--spec-draft-n-max`) |
| `PF_DFLASH_TIME` | off | DFlash2 cycle phase breakdown in ms |
| `PF_DFLASH_SEGTIME` | off | device-side breakdown of the draft block forward (needs `PF_DFLASH_DEBUG`-free; see `AGENTS.md`) |
| `PF_DFLASH_SLICES` | 8 | top-k slices per row; the measured optimum, *not* the occupancy-maximising value |
| `PF_DEVICE_PROFILE` / `PF_DEVICE_INFO` | – | pin a tuning profile / dump the resolved one with its provenance |

The full tuning/diagnostics knob list and internals are documented in
[`AGENTS.md`](AGENTS.md).

## Tests

```bash
source /opt/intel/oneapi/setvars.sh
cmake --build build -j
./build/test_tokenizer    # tokenizer round-trips (CPU)
./build/test_chat_template # GGUF chat template vs reference Jinja2 output (CPU)
./build/test_response_parser # reasoning_content / tool_call splitter (CPU)
./build/test_sampler      # logit_bias + logprob reporting (CPU)
./build/test_multimodal   # image preprocessing, vision encoder (host + device), positions (CPU+GPU)
./build/test_gpu_stages   # every kernel vs the CPU reference (GPU)
./build/test_gemv         # per-tensor GEMV vs CPU dequant reference (GPU)
./build/test_dp4a         # SIn repack + DP4A GEMM vs CPU reference (GPU)
./build/test_gpu_vs_ref   # end-to-end logits vs CPU reference (GPU)
./build/test_forward      # end-to-end logits / top-k (GPU)
./build/test_decode_vs_prefill  # single-token decode == re-prefill (GPU)
./build/test_sched        # the scheduler's token-stream path + a 20-row batched decode (GPU)
./build/test_spec         # the MTP and DFlash2 streams == the plain greedy decode (GPU, 27B)
./build/test_dflash_kernels # the DFlash2 top-k and conv vs the host references (GPU)
./build/test_w4*          # native-width u4 packing / GEMM / vs int8 (CPU+GPU)
./build/test_k5_gemv      # native 5-bit (Q5_K) store GEMV vs a host reference (GPU, 27B)
./build/test_w4_vs_cpuref # u4 end-to-end logits vs the fp32 reference (GPU, 27B)
./build/test_pc_disk      # disk prefix-cache format / LRU / reopen (CPU)
./build/test_pc_ram       # RAM prefix-cache tier / LRU (CPU)
./build/test_cpu_gemv     # CPU fp32 + int8 GEMV/RMSNorm vs dequant reference (CPU)
./build/test_pc_cpu       # host-backend paged attention + prefix-cache round-trip (CPU)
./build/test_pc_gpu       # prefix-cache disk spill + promote round-trip (GPU)
./build/test_pc_ram_gpu   # prefix-cache VRAM->RAM->VRAM round-trip (GPU)
./build/test_cpuref       # CPU reference head output (CPU)
./build/test_compare      # CPU reference vs llama.cpp dumps (CPU)
./build/test_control      # the worker's control plane: the protocol + its auth (CPU)
```

Each test takes an optional model path as its first argument and otherwise falls
back to a built-in default; the GPU tests need the model and a working device.
The weight-path audits (`test_w4*`, `test_k5_gemv`, `test_w4_vs_cpuref`,
`test_w4_topk`, `test_iq_dequant`, `test_quant_audit`, `test_gemv_stride`) default
to the 27B instead.

## Notes

* Batched decode is worth using: on the 27B, 1/2/4/8 concurrent greedy requests
  measure 13.4 / 18.9 / 34.4 / 62.5 tok/s aggregate, and on the 35B A3B
  26.4 / 35.7 / 60.9 / 101.1, because every extra row amortises the same
  weight pass.  On integrated GPUs both prefill and decode are
  memory-bandwidth bound, which is what the host backend is there for.
* **Batching is not bit-exact against a single request.**  At temperature 0 a
  prompt fanned out to N concurrent requests returns coherent, deterministic text
  that can differ from the single-request result: a sequence that joins a batch
  changes the order of the fp accumulation in the batched GEMMs, which flips the
  argmax at near-ties (measured: 1 of N rows byte-exact, the rest flipping at the
  same tie).  The set of outputs is stable across repeated runs.
* Speculative decoding (`--mtp`) trades that batching win away — see
  [Performance](#performance) for when it pays.
* Multimodal requests bypass the prefix cache, and a media block consumes
  `max(nx, ny)` M-RoPE positions while contributing `4·nx·ny` (image) or
  `T·nx·ny` (video) tokens, so a prompt's token count and its position count
  differ.
* `PF_W4=0` / `PF_K5=0` / `PF_CB4=0` trade prefill speed for decode speed and
  memory: the native stores are lossless for Q4_K, IQ4_XS/IQ4_NL and Q5_K, but
  prefill expands them back to int8 in a reused scratch first.  `PF_K5` is the
  clearest trade in the set — measured -15..-20 % pp512 for +6 % tg128 and
  2.1 GB per card.

## Credits

sycl-infer is an independent implementation, but it stands on five shoulders:

* **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT) — the GGUF
  container format and the K-quant block layouts that `src/common/quant.h`
  dequantizes on the fly, the byte-level BPE tokenizer behaviour, the vendored
  Unicode tables (`third_party/unicode*.{h,cpp}`), and the reference semantics
  the NextN/MTP speculative path follows (the draft row's right-shift pairing and
  the dry verify with a recurrent-state rewind).
* **[OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)**
  (Apache-2.0) — the reference point for prefill throughput on Intel GPUs, with two
  caveats worth stating: the 640 / 362 tok/s figures were measured on **OpenVINO
  Model Server built from the 2026.4.0 git tree**, not on a release, because
  Qwen3.8-27B support does not exist in 2026.3.1; and they were measured on an
  **int4-quantized** copy of the model, not on the Q4_K_M this engine runs.
* **[oneDNN](https://github.com/oneapi-src/oneDNN)** (Apache-2.0) — the int8
  matmul primitives behind both the prefill GEMM (`PF_GEMM_DNNL`) and the prefill
  attention (`PF_ATTN_XMX`), plus the DPC++/SYCL compiler and runtime the engine
  is built with.
* **llama-benchy** — the OpenAI-server benchmark harness the prefill and decode
  rates in the 16k/64k depth cells of the 2026-10-09 pass were measured with.
  Its own `pp512` column is an artefact (it divides the prompt length by the
  time to the first streamed SSE chunk, which the server sends *before* the
  prefill runs) and is not quoted here — see
  [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md#what-the-server-actually-delivers).
* **[Strata](https://github.com/Niko1221/Strata)** (MIT) — the reference
  CUDA-path engine for Qwen3.8-Flash-Next; its tail-skip, per-expert cache, MTP
  and batched-decode ideas informed the qwen4exp work.  The comparisons and the
  dead ends are in [Design 18](docs/design/18-qwen4exp.md).

## License

sycl-infer is released under the MIT License — see [`LICENSE`](LICENSE).

Bundled third-party code (`third_party/`) retains its own license: cpp-httplib,
nlohmann/json, minja and the llama.cpp Unicode tables are all MIT-licensed.  See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the per-component
inventory and full license texts.
