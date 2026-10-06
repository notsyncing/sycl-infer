# sycl-infer — a general-purpose SYCL LLM inference engine

> **Experiment project to explore if AI can be used to create an inference engine.**

sycl-infer is a from-scratch C++17/SYCL inference engine for quantized LLM
models on Intel GPUs or the host CPU.  It loads GGUF models directly, serves
them through an OpenAI-compatible HTTP API or a CLI, and is built around a
**pluggable architecture registry** so new model types can be added without
touching the engine or the kernel library.

The reference model used for development and validation is Qwen3.5-0.8B (Q4_K_M)
on Intel Iris Xe-LP; the throughput numbers below are the 27B-class
Qwen3.8-27B-UD-Q4_K_M on two Arc A770.  The engine itself is tied to neither — the
model type is selected through the architecture registry.

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
  concurrent sequences).  Batched decode measures 14.3 → 24.5 → 40.1 → **55.0
  tok/s aggregate** for 1/2/4/8 concurrent greedy requests on the 27B (3.85x at
  8), because every extra row amortises the same weight pass.
* **Cross-request prefix cache** — hashed 32-token blocks with recurrent-state
  checkpoints, so a shared prompt prefix is prefilled only once.  Three LRU
  tiers: device (VRAM) → host RAM → disk.
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
  **1.4x - 2.5x** single-request decode on the 27B depending on how much the
  prompt's own distribution supports the draft (see
  [Performance](#performance-27b-on-two-arc-a770)); the emitted stream is
  greedy-equivalent.  Requires a GGUF with a `blk.<n>.nextn.*` head *and* a
  multi-device oneDNN int8 partition, and it does not compose with batching, so
  the server keeps it off by default.
* **DFlash2 block drafter (`--spec-type dflash2`)** — a second speculative
  drafter, structurally different from MTP: the draft GGUF's own 5-layer head runs
  **one non-causal forward** over `[anchor, MASK x (n_max)]` and a selector scores
  the top-K candidate sets per block position plus every ordered pair, so the host
  walks one coherent path through the lattice instead of re-drawing per token.
  Measured **2.17x** single-request decode on the 27B (31.5 vs ~68 ms/token at
  `n_max=5`), with acceptance matching llama.cpp at the same `n_max`.  Needs a
  matching draft GGUF via `--spec-draft-model`.  See
  [Design 14](docs/design/14-dflash2.md).
* **Per-GPU tuning profiles** — every launch constant that was measured on a
  particular card lives in that card's profile (`src/device/profiles/<card>.cpp`,
  selected from the reported device name), so an unknown GPU warns loudly instead
  of silently inheriting another card's tuning.

## Supported models

| | |
|---|---|
| file format | GGUF (memory-mapped, single file) |
| weight quantization | Q4_K, Q5_K, Q6_K, Q3_K, IQ4_XS, IQ4_NL, IQ3_S, Q8_0, F32; Q4_K uses the native 4-bit (u4) path by default |
| architecture registry | selected via `general.architecture`; `qwen35` implemented, others plug in via `src/model/model_arch.h` + a `kLoaders[]` entry |
| KV cache dtype | `i8` (default), `i4`, `bf16`, `f16`, `f32`; `--kv-type K:V` sizes K and V independently (`i4:i8`) |
| tokenizer | GPT-2 byte-level BPE from the GGUF vocabulary + special tokens / chat template |

Adding an architecture is a single new file under `src/model/` — see
[`AGENTS.md`](AGENTS.md#add-a-model-architecture).

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
| `--layer-map L:dev,...` | – | pipeline-parallel layer placement, e.g. `0-31:gpu.0,32-63:gpu.1` |
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

## Performance (27B on two Arc A770)

Measured configuration: `Qwen3.8-27B-UD-Q4_K_M` (64 layers, 16 full-attention,
`n_head` 24 / `n_head_kv` 4, `head_dim` 256), two Arc A770 16 GB,
`--layer-map 0-31:gpu.0,32-63:gpu.1`, i8 KV, AOT Release build, ctx 131072,
`PF_PREFIX_CACHE=0`, driven by `llama-benchy` 0.4.0
(`--pp 512 --tg 128 --depth {0,16384,65536} --exact-tg`, 2-3 runs) against
the OpenAI server.

| context depth | prefill pp512 | decode tg128 |
|---|---:|---:|
| 0 | **2 870 tok/s** | **14.6 tok/s** |
| 16 384 | **911 tok/s** | **13.4 tok/s** |
| 65 536 | **540 tok/s** | **10.4 tok/s** |

* `pp512` is the rate of the *marginal* 512-token prefill chunk at that depth,
  i.e. what a long prompt actually experiences (at depth 0 it is the 512-token
  prompt itself, 178-187 ms).  The cold full-prompt time-to-first-token of those
  runs was 1.38 s / 18.6 s / 87.3 s at depth 0 / 16k / 60k.
* `tg128` is decode throughput over exactly 128 generated tokens for one request.
* The 16k and 64k prefill rates are above the OpenVINO reference numbers for this
  model on this hardware — 640 and 362 tok/s.  Two caveats on that comparison, both
  in OpenVINO's favour or against it, so read them with the numbers: they were
  measured on **OpenVINO Model Server built from the 2026.4.0 git tree** (the
  released 2026.3.1 predates Qwen3.8-27B support and cannot load the model at all),
  and on an **int4-quantized** copy of the model rather than the Q4_K_M used here —
  comparable in bytes-per-weight, but a different quantization, so it is a
  throughput reference point, not a like-for-like accuracy comparison.  Getting
  above those targets took two changes in the prefill path: the KV gather's
  block-max reduction spread over 64 work-groups instead of one, and the per-row
  index arrays built by a device kernel instead of three host→device copies per
  kv head.
* Where the remaining time goes: a decode step at 64k depth is 95 ms against an
  ~88 ms floor (27 ms attention + 62 ms of weight stream at the measured
  400 GB/s read ceiling), and a 512-token prefill chunk at 64k is 833 ms of which
  ~230 ms is attention and ~373 ms dense GEMM.

### MTP speculative decoding (`--mtp`)

Single-request greedy decode, 128 tokens per cell, one process per cell, same
box; plain decode on the same build is **69.0 ms/token**:

| prompt | draft `k` | acceptance | MTP ms/token | speedup | ceiling (1 + acc) |
|---|---:|---:|---:|---:|---:|
| code continuation | 4 | 2.78 | 27.8 | **2.49x** | 3.78x |
| technical explanation | 4 | 2.25 | 32.3 | **2.14x** | 3.25x |
| story opener | 4 | 1.18 | 48.3 | 1.43x | 2.18x |

Three things worth knowing before enabling it:

* **the speedup is acceptance-bound, not engine-bound** — the cycle costs ~15 ms
  over a plain decode on all three prompts, so what varies is how many drafted
  tokens the model's own NextN head gets right (the per-depth acceptance decays
  geometrically at ~0.7 with no step-0 collapse).  A prompt with `acc ≈ 1.2`
  cannot be pushed past 1.4x by any engine change;
* **it does not compose with batching** — MTP spends `k+1 = 5` verify rows per
  `acc+1` emitted tokens (≈2.0 rows/token against the plain decode's 1.0) while
  batching amortises the same weight pass for free, so MTP only wins at
  concurrency ≤ 2.  The server therefore defaults to the scheduler: 8 concurrent
  128-token greedy requests measure 55.0 tok/s through it against 17.5 tok/s with
  the single-sequence MTP loop (`PF_MTP_SERVER=1` restores the MTP routing);
* **it is a short-context win** — at 128k depth the 7-row verify costs 1390
  ms/cycle against 181 ms for a plain decode, because the extra rows multiply the
  attention over a 131k-key KV.

The spread between the three prompts is entirely the acceptance, not the engine:
`X` (the cycle-only cost over a plain decode) is ~15 ms on all of them, so the
only thing that moves the speedup is how many drafted tokens the NextN head gets
right — `reach[j]` (the fraction of cycles accepting at least `j` drafts) decays
geometrically at p ≈ 0.7 with no step-0 collapse, i.e. the draft's hidden state
is right and the decay is the model's own accuracy.

Two knobs that look like levers and are not: the draft's precision is free
(`PF_MTP_LAYER_EXACT=1` runs its GEMMs on the exact fp32 dequant reference and
the acceptance is identical), and neither is its bytes (`PF_MTP_HEAD_W2=1`, a 2-bit
head store at 0.375 B/weight, is a wash: -2.6 ms/cycle for -2 % acceptance,
because that GEMV is not bandwidth-bound).

### DFlash2 block drafter (`--spec-type dflash2`)

Single-request greedy decode on the same box and model.  `n_max` is the number of
draft tokens per cycle (`acc` is accepted drafts per cycle, so the ceiling on
speedup is `1 + acc`); each cell is one process per configuration, and the plain
decode on the same build is **~68 ms/token**:

| `n_max` | 1 | 2 | 3 | 4 | **5** | 6 |
|---|---:|---:|---:|---:|---:|---:|
| acc | 0.98 | 1.44 | 2.18 | 2.54 | **2.90** | 2.90 |
| ms/token | 48.6 | 41.9 | 34.1 | 32.5 | **31.5** | 33.3 |

**`n_max=5` is the shipped default: 2.90 accepted drafts per cycle, 31.5 ms/token,
2.17x over the plain decode.**  Acceptance against llama.cpp at the same `n_max`
(short prompt, mean accepted tokens per cycle):

| generated | 32 | 116 | 227 |
|---|---:|---:|---:|
| llama.cpp | 4.43 | 3.96 | 4.13 |
| this engine | 4.75 | 3.90 | 3.92 |

On a 204-token prompt both engines drop (llama.cpp 0.239 / mean 2.07, this engine
0.81 / 1.81), so the long-context gap is ~12 %, not a structural difference.

Where the 122 ms cycle goes at `n_max=5`: `verify=90.9`, `draft=26.9`,
`inject=1.5`, `emit=1.5`, `rollback=1.0`.  **The verify is already at its
structural floor** — a `n_max` sweep prices it at 72.2 ms fixed plus ~5.5 ms per
extra row, which the weight-stream model predicts to 0.1 ms — and it runs on the
recorded command graphs (378 of them), not the direct replay that cost 210 ms on
the MTP path.  The remaining levers are fewer weight bytes and a faster dp4a, and
both are already measured and rejected on accuracy grounds.  Details in
[Design 14 §7](docs/design/14-dflash2.md).

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
```

Each test takes an optional model path as its first argument and otherwise falls
back to a built-in default; the GPU tests need the model and a working device.
The weight-path audits (`test_w4*`, `test_k5_gemv`, `test_w4_vs_cpuref`,
`test_w4_topk`, `test_iq_dequant`, `test_quant_audit`, `test_gemv_stride`) default
to the 27B instead.

## Notes

* Batched decode is worth using: on the 27B, 1/2/4/8 concurrent greedy requests
  measure 14.3 / 24.5 / 40.1 / 55.0 tok/s aggregate, because every extra row
  amortises the same weight pass.  On integrated GPUs both prefill and decode are
  memory-bandwidth bound, which is what the host backend is there for.
* **Batching is not bit-exact against a single request.**  At temperature 0 a
  prompt fanned out to N concurrent requests returns coherent, deterministic text
  that can differ from the single-request result: a sequence that joins a batch
  changes the order of the fp accumulation in the batched GEMMs, which flips the
  argmax at near-ties (measured: 1 of N rows byte-exact, the rest flipping at the
  same tie).  The set of outputs is stable across repeated runs.
* Speculative decoding (`--mtp`) trades that batching win away — see
  [Performance](#performance-27b-on-two-arc-a770) for when it pays.
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

sycl-infer is an independent implementation, but it stands on four shoulders:

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
* **llama-benchy** — the OpenAI-server benchmark harness every throughput number
  in this README was measured with.

## License

sycl-infer is released under the MIT License — see [`LICENSE`](LICENSE).

Bundled third-party code (`third_party/`) retains its own license: cpp-httplib,
nlohmann/json, minja and the llama.cpp Unicode tables are all MIT-licensed.  See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the per-component
inventory and full license texts.
