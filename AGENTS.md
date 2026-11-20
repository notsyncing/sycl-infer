# AGENTS.md

Guidance for AI coding agents working on **sycl-infer**.

This file has three jobs, in this order:

1. **The reference** — what the engine is, how to build/test/extend it, the conventions, the environment variables.
2. **The measurement archive** — the conclusions a change must be read against, each with the condition it was measured under. The full tables live in `docs/PERFORMANCE.md`; this file keeps the *conclusions*, not a second copy.
3. **The invariants** — the rules that are not obvious from the code, and what breaks when they are broken. Every one was broken once, or fails silently.

**`reports/` is not part of this repository.** It is in `.gitignore` and stays that way: no committed document may link to it — not this file, `README.md`, nor `docs/`. A `reports/` reference in a committed doc is a dead link for everyone who clones the tree.

## Contents

| | |
|---|---|
| [What this is](#what-this-is) | the engine in one paragraph |
| [Documentation](#documentation) | the design-doc index |
| [Build](#build) / [Run](#run) | toolchain, AOT, CLI |
| [Testing](#testing) | hermetic layer, GPU layer, five mandatory tests, kernel coverage gates |
| [Repository layout](#repository-layout) | where each module lives |
| [How to extend](#how-to-extend) | a model, an embedding model, a pipeline, a kernel, multimodal |
| [Major subsystems](#major-subsystems) | qwen4exp, deepseek41, qwen2vl, MTP, DFlash2, ONNX service |
| [Device profiles](#device-profiles) / [Model profiles](#model-profiles) | the two policy layers and their boundary |
| [Conventions](#conventions) | style and repo rules |
| [Environment variables](#environment-variables) | every `PF_*` knob in condensed form |
| [Measured results and dead ends](#measured-results-and-dead-ends) | what not to retry |
| [Invariants and gotchas](#invariants-and-gotchas) | rules that fail silently |
| [Quick verification](#quick-verification-before-finishing-a-change) | the pre-commit run |

## What this is

A from-scratch C++17 + SYCL LLM inference engine for quantized GGUF models on
Intel GPUs, with a host CPU backend (`--device cpu`, AVX2 / AVX-VNNI / AVX-512
selected at run time) and a separate path for GGUF BERT embedding models
(`src/embedding/`, CPU / GPU / NPU through the ONNX executor). The reference
model is Qwen3.5-0.8B (Q4_K_M) on Iris Xe-LP, but the engine is
architecture-agnostic: types are selected through `src/model/arch/registry.h`
and the internal `kLoaders[]` table in `src/model/arch/registry.cpp`. It
contains a GGUF loader, a quantized SYCL kernel library, a paged-KV
continuous-batching engine with a reusable prefix cache, an OpenAI-compatible
HTTP server, a CLI, a router for multiple models, and an ONNX model service.

The GPU path prefers a Level Zero device and otherwise falls back to
`sycl::gpu_selector_v` (`src/device/device_select.h`); an explicit
`ONEAPI_DEVICE_SELECTOR` still wins. This is not cosmetic: on a host exposing
the same card through OpenCL and Level Zero, `gpu_selector_v` picks OpenCL,
which has no SYCL Graph extension, and the decode exits with a graph error.
Dense (or hybrid) models can be split by layer with `--layer-map` (pipeline
parallel); each device owns the paged KV of the attention layers it computes.

## Documentation

`docs/` is the source of truth for *why*; the index is
[`docs/README.md`](docs/README.md). Read the relevant doc before changing a
subsystem.

| doc | subject |
|---|---|
| `docs/architecture.md` | module map, runtime object model, data flow, threading, invariants |
| `docs/design/01-model-loading.md` | GGUF parser, mmap, registry, tensor binding, dev_ptr, CLI flags |
| `docs/design/02-quantization.md` | ggml K-quants, SIn int8, u4/k5/cb4 stores, `xq`, DP4A, oneDNN |
| `docs/design/03-kernels.md` | every kernel: layout, geometry, tuning, how to add one |
| `docs/design/04-engine.md` | `seg_plan`, `build_plan`, `record_forward`, graphs, prefill/decode, `step_info` |
| `docs/design/05-kv-cache.md` | paged KV, virtual-USM block pool, KV types |
| `docs/design/06-prefix-cache.md` | chained hashes, recurrent checkpoints, VRAM/RAM/disk tiers |
| `docs/design/07-sampler.md` / `08-tokenizer.md` | sampling; byte-level BPE and chat templates |
| `docs/design/09-server.md` | OpenAI HTTP API, scheduler, SSE, stop strings, field matrix |
| `docs/design/10-multimodal.md` / `13-audio-video.md` / `21-mineru-qwen2vl.md` | image / audio / video towers and MinerU |
| `docs/design/11-qwen35-model.md` / `18-qwen4exp.md` / `23-deepseek41.md` / `24-arch-modules.md` | the model architectures and the arch contract |
| `docs/design/12-build-and-testing.md` | build targets, test matrix, verification, debugging |
| `docs/design/14-dflash2.md` / `15-moe-ncmoe.md` / `16-router.md` | drafter, routed experts, multi-model front end |
| `docs/design/17-onnx-service.md` | ONNX graph, transforms, operators, executor, NPU path, wire protocol |
| `docs/design/19-c-api.md` / `20-python-binding.md` / `22-pipelines.md` | embedded ABI, Python wheel, request pipelines |
| `docs/design/25-embeddings.md` / `26-research.md` | GGUF BERT encoders; isolated FFN research C API |
| `docs/PERFORMANCE.md` | **the measured throughput table** with hardware, build, harness, command |

## Build

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build
cmake --build build -j$(nproc)
```

* `icpx` is required; CMake sets `CMAKE_CXX_COMPILER` to it **before** `project()`. Setting it afterwards leaves `g++` cached and the first `-fsycl` fails.
* oneDNN is expected at `/opt/intel/oneapi/dnnl/2026.0` (`-DDNNL_ROOT=` overrides). The binary's RUNPATH points at that removed path; without `setvars.sh` the 3.13.2 library from `dnnl/2026.1/lib` is not found at all. `PF_ONNX_*` numbers taken before the library change are stale.
* The Docker base is `intel/oneapi-toolkit` (2026 rename of `oneapi-basekit`); the 2026.2 tag is `2026.2.0-devel-ubuntu26.04`. `ocloc` comes from `ppa:kobuk-team/intel-graphics`, not the toolkit image. The 2026.2 runtime makes `enable_profiling` cost 24x per launch; that queue property is now opt-in.
* Debug builds force `-O2` for device code — at `-O0` IGC's image becomes huge and translation looks like a hang.
* Source files are listed explicitly in `CMakeLists.txt` (no globbing). Add new `.cpp` files there.
* **Watch the memory.** A 20-way parallel Debug SYCL build OOM-kills on a 30 GB box; use `-j4`. A killed build looks like a compile error with no message.

### AOT builds

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSYCL_INFER_AOT=ON -DSYCL_INFER_AOT_JOBS=6
cmake --build build --target sycl-infer -j$(nproc)
```

* Device lowering (`llvm-foreach` → `ocloc`) runs at **final link**, never at `-c`; `make -j` does not parallelise it. `-DSYCL_INFER_AOT_JOBS=N` sets `-fsycl-max-parallel-link-jobs=N`; lower it on memory-constrained hosts. `-fsycl-device-code-split=per_kernel` lets images be skipped individually.
* **AOT flags and the library you link must agree.** Linking the AOT `libsycl_infer_core.a` into a JIT binary crashes in `DeviceKernelInfo::setCompileTimeInfoIfNeeded`. Build hand-written tools against a JIT archive (`build-opencode-jit/libsycl_infer_core.a`). See *Benchmarking*.
* An Xe2 AOT target (`lnl`/`bmg`) must omit `nat_gemm`'s SG=8 image (`SI_NO_SG8`); ocloc rejects `reqd_sub_group_size(8)` there.
* Use Release: Debug `-g` makes each device image several times larger to lower.
* The `ocloc` compiler cache does not work on the distro driver. The runtime JIT cache (`SYCL_CACHE_PERSISTENT=1`) crashes every GPU test binary on the 2x A770 box; use AOT there.

## Run

Binary startup needs the oneAPI runtime libraries; `source /opt/intel/oneapi/setvars.sh`
handles it, or:

```bash
export LD_LIBRARY_PATH=/opt/intel/oneapi/2026.2/lib:/opt/intel/oneapi/compiler/2026.2/lib:$LD_LIBRARY_PATH
```

```bash
./build/sycl-infer --model /path/to/Qwen3.5-0.8B-Q4_K_M.gguf serve
./build/sycl-infer --model ... gen --prompt "Hello" --max-tokens 32
./build/sycl-infer --model /path/to/bge-m3.gguf --device cpu embed --prompt "hello"
```

The router is a separate binary with no SYCL/oneDNN dependency:
`cmake -S . -B build-router-only -DSI_ROUTER_ONLY=ON ...`, then
`./build-router-only/sycl-infer-router --config router.json [--check]`.

`--kv-type K:V` sizes K and V independently. **The attention error is dominated by V, not K**: on the 27B, `i4:i8` measures mean|diff| 0.061 vs i8's 0.035 and i4's 0.183. But `i4:i8` is a *capacity* setting, not a correctness guarantee — a KG-extraction prompt returned `{"facts":[]}` under `i4:i8` where `i8` returned facts; the i4 K Hadamard rotation (`PF_KV_ROT`, default on for GPU i4) fixes some prompts but not that one. Keep `i8` for greedy extraction. TurboQuant/QJL at 3-4 bits are still 13.5x worse than i8; do not implement them for this.

`--max-slots N` bounds concurrent sequences and sizes the **recurrent** state (1.125 GiB/card at the default 16 on the 27B) — the cheapest context headroom there is. It is not the row ceiling: `kMaxB` (32) is the compile-time ceiling, `--max-slots` (16) is policy.

`--yarn` enables NTK-by-parts context extension, **default off**; a factor <= 1 prints one line and disables itself. It extends the window but does not save KV memory and zero-shot costs quality. `--ctx 1048576 --yarn` on a 32k-trained model derives factor 32.

## Testing

Tests default to the 0.8B model and need the GPU. Strict kernel tests set `PF_DP4A=0`. The exceptions (`test_k5_gemv`, `test_quant_audit`, `test_w4*`, `test_gemv_stride`, `test_iq_dequant`, `test_27b_prefill`, `test_spec`) default to the 27B as `argv[1]`. `test_q4_batch`/`test_q4_verify` need the 4-shard qwen4exp GGUF and an AOT build.

```bash
ctest --test-dir build -L hermetic --output-on-failure        # no GPU, no model
ctest --test-dir build -L gpu --output-on-failure             # -DSI_REGISTER_GPU_TESTS=ON + model path
```

Useful hermetic tests: `test_tokenizer`, `test_chat_template`, `test_response_parser`, `test_sse_cancel`, `test_sampler`, `test_guided`, `test_model_profile`, `test_dev_profile`, `test_hparams`, `test_f16conv`, `test_yarn`, `test_kv_type`, `test_quant_synth`, `test_w8_layout`, `test_chat_builtin`, `test_request_parse`/`test_response_json`/`test_control`, `test_url_fetch_*`, `test_router_*`, `test_embed_tokenizer`, `test_arch_ops`, `test_mm_step`, `test_quant_synth`.

**Five tests are mandatory after touching a kernel or the engine**:
`test_gpu_stages`, `test_gpu_vs_ref`, `test_forward`, `test_decode_vs_prefill`, `test_sched`.
The first three are prefill-only and blind to decode bugs; **any single-token change must run `test_decode_vs_prefill`**. `test_sched` is the only batched-decode coverage — its last section decodes 20 rows and requires every row's logits to move between steps.

### Kernel unit tests (mandatory)

Every kernel entry point has a model-free unit test that is part of the change that adds or changes the kernel. An end-to-end run cannot attribute a moved logit to one of ~500 launchers, and a stage test needs a model and a forward.

| launcher tree | declaration | test location |
|---|---|---|
| `src/backend/gpu/kernels/*.cpp` | `src/backend/gpu/kernels/kernels.h` | `tests/backend/gpu/kernels/test_*_units.cpp` |
| `src/backend/cpu/kernels/*.cpp` | `src/backend/cpu/cpu_types.h` | `tests/backend/cpu/kernels/test_cpu_*_units.cpp` (hermetic) |

Reference ladder: (1) hermetic scalar reference, (2) CPU twin that itself has a scalar test, (3) exactness/property for data movement. Use a fixed seed, tail shapes, an inactive row/slot, a partial tile; assert value **and** size and print `max|diff|` against `max|expected|`; use the kernel's own tolerance (bit-exact for copy/pack, documented quantized error for int8/u4). `ci/kernel_test_map.txt` names the test for every declaration; `ci/check_kernel_coverage.py` (the `kernel_coverage` ctest) fails when a row is missing, names a non-existent test, or an `!integration` row has no reason. `python3 ci/check_kernel_coverage.py` before finishing a kernel change. `ci/check_arch_boundaries.py` enforces the model/engine boundary.

## Repository layout

```
src/common/     quant.h, w8.cpp, w4.cpp, dp4a.h, cpu_isa.*, jsonrpc.h, exit_codes.h,
                onnx_wire.*, onnx_activation.h, onnx_types.h
src/device/     device_profile.h, device_select.h, device_registry.cpp, profiles/<card>.cpp
src/backend/    backend.h, dnnl_gemm.*, weight_store.{h,cpp} (the single device weight store;
                model.h includes no SYCL), gpu/gpu_backend.cpp, cpu/cpu_backend.cpp
src/backend/gpu/kernels/  kernels.h, kernel_utils.h, kv_type.*, one .cpp per kernel
src/backend/cpu/kernels/  one .cpp per host kernel (compiled -fno-sycl, ISA variants)
src/backend/npu/          Level Zero NPU executor + IR frontend + constant folding
src/model/      gguf.*, model.*, model_types.h, arch_util.h, arch/registry.*,
                model_profile.h, tokenizer.*, dflash.*, onnx/ (graph, transforms)
src/model/arch/<arch>/    load.{h,cpp}, profile.cpp (+ tensors/accessors/hparams for owned forwards)
src/embedding/  GGUF BERT encoders: sp_tokenizer.*, bert_model.*, embedding_engine.*, embedding_http.*
src/mm/         vision/image/video/audio/audio_model/multimodal.{h,cpp}
src/engine/     engine.*, engine_types.h, graph.cpp, engine_kvpool.cpp,
                engine_prefix_cache.cpp, moe_cache.*, pc_ram.*, pc_disk.*, sampler.*,
                guided.*; spec/ (mtp, dflash, state headers); arch/<arch>/ (model-owned forwards)
src/server/     chat.*, chat_template.*, response_parser.*, sse.h, scheduler.*, server.*,
                control.cpp, pipeline_http.*, onnx/
src/main.cpp    CLI (serve/gen/embed); MinerU lives in tools/mineru/
src/pipeline/   request-level multi-step orchestration (registry empty by default)
src/router/     sycl-infer-router: config, registry, rpc_client, supervisor, proxy
src/capi/       C ABI implementation + thread-local kv_scope (embedding in capi_embedding.cpp)
src/onnx_api/   libsycl_infer_onnx: graph/executor + onnxwire
include/sycl_infer/  sycl_infer.h, onnx.h, sycl_infer_research.h (research only)
research/       isolated FFN capture / pattern / similarity C API + its own test
python/         ctypes binding, wheel packaging, research.py (explicit import only)
tests/          mirrors src/; tests/backend, tests/model, tests/server, tests/engine, tests/mm,
                tests/router, tests/onnx, tests/embedding
ci/             check_arch_boundaries.py, check_kernel_coverage.py + kernel_test_map.txt, Dockerfiles
tools/mineru/   standalone sycl-infer-mineru client (not a server pipeline)
third_party/    httplib, json, minja, unicode tables, stb, level_zero_npu (do not edit)
dev/            gitignored one-file harnesses and probes; they rot silently
```

## How to extend

### Add a model architecture

**Declare architecture-specific facts; never hardcode them at the use site.** Enumerate what can differ (RoPE layout, M-RoPE axis mapping, image floors, attention geometry, pad tokens, KV/recurrent layout) and give each a name in `model_profile` (policy/capabilities) or loader-filled `hparams` (file/geometry facts). This is not style: Qwen3.5's `rope.dimension_sections = [11,11,10,0]` sums to `n_rot/2` exactly like Qwen2-VL's *blocked* sections, so a `sum == n_rot/2` predicate sent 35B image prompts through the wrong M-RoPE (`WORLD` → `VOWEL`). `hparams::mrope_interleaved` is now a loader fact carried into `step_info`.

1. Create `src/model/arch/<arch>/load.cpp` (`si::arch::load_<arch>`) and `profile.cpp` (complete `model_profile`, `key` byte-identical to `general.architecture`), declared in `load.h`. Use `arch::bind_tensor` / `bind_f32` from `arch_util.h`.
2. Register the row in the **internal** `kLoaders[]` in `src/model/arch/registry.cpp`. The public `registry.h` and `model_profile.h` need no change.
3. A generic-plan architecture stops there (`arch_ops_for(generic_plan)` is null). A model-owned forward also needs `src/model/arch/<arch>/{hparams,tensors,accessors}.h`, `src/engine/arch/<arch>/{forward*,runtime.cpp,state.h}` and a `kArchOps[]` row in `src/engine/arch/registry.cpp`. Declare forward entry points in a local `forward.h`, never `engine.h`. See design 24.
4. Add every `.cpp` to `CMakeLists.txt`. Unknown architecture still throws `unsupported architecture: <name>`.
5. `validate_hparams` runs after every loader — a memory-safety guard, not precision: `attn.cpp`'s generic kernel needs `head_dim` a multiple of 32 and <= 256 or it overruns its register array. Wide heads opt in via `arch_hparams::wide_head_ok`.
6. Pin the profile in `test_model_profile`, a model-owned row in `test_arch_ops`, keep `check_arch_boundaries.py` clean. The qwen35/qwen35moe provenance strings intentionally cite the pre-move loader paths — do not "fix" them.

### Add an embedding model

GGUF `general.architecture == bert` is **not** loaded through `model::load`: a bidirectional encoder has no LM head, KV pool or sampler. The CLI detects it with `embedding::is_embedding_model()` before `engine` is built. One GGUF is lowered once to an in-memory ONNX graph shared by CPU/GPU and NPU. **Embedding tables are copied, not transposed**; matrix weights are transposed; the attention output transpose is `{0,2,1,3}`. The graph must stay in opset1 primitives: Lunar Lake rejects IR `LayerNorm` and ONNX `Gelu`, so both are decomposed. BGE-M3's tokenizer is SentencePiece Unigram (`t5`), not BPE. The graph axis is `--embed-max-seq`, not `--ctx`. See design 25 and `docs/guides/c-api.md`.

### Add a pipeline

Admission rule: only when it needs in-process engine/prefix/verify state, has a measured server-side win, or shares the server's concurrency/cancel/resource account. A client-side "call the model several times" workflow belongs in `tools/`. Implement the `pipeline` interface (`src/pipeline/pipeline.h`), keep prompts/grammars in TU without `engine`/SYCL, register at `serve()` startup, expose only `GET /v1/pipelines` and `POST /v1/pipelines/{id}/run`. MinerU is deliberately **not** a pipeline. See design 22.

### Add a kernel

1. Add `src/backend/gpu/kernels/<name>.cpp` with the launcher and kernel lambdas in the same TU; include `"kernels.h"` and `"kernel_utils.h"`; device helpers live in `si::kd` in `kernel_utils.h`. Do not put kernel bodies in headers.
2. Declare the launcher in `kernels.h`, add the `.cpp` to CMake, add a model-free unit test and a row in `ci/kernel_test_map.txt`, run `ci/check_kernel_coverage.py`.
3. A tuned constant measured on one card belongs in that card's device profile, read through `si::dev::for_queue(q)` — never a literal in the launcher.
4. CPU kernels add a `.cpp` in `src/backend/cpu/kernels/`, a declaration in `cpu_types.h`, forwarding in `cpu_backend.cpp`, and **both** the CMake source list and `SI_CPU_SOURCES` (`-fno-sycl`). Use `par(...)` and `isa().dot_f32` / `dot_i8`.

### Add a kernel stage test

Add `tests/backend/gpu/kernels/<kernel>_stage.cpp` defining `void stage_<kernel>(si::stage_env&)`, declare it in `stage_tests.h`, call it from `test_gpu_stages.cpp`, add the file to that test's CMake source list.

### Multimodal input

The vision encoder is driven by a separate `clip` mmproj GGUF (`--mmproj`). One `vision_model` serves qwen3vl and qwen2vl; `model_profile::vision` selects the tower and the mmproj's `clip.projector_type` is cross-checked. Image preprocessing does Qwen-VL smart resize, Pillow-compatible bicubic resample and CHW normalization. Both host and device encoders must agree to ~2e-4 (`test_multimodal`). The vision linears run on oneDNN int8 by default for qwen2vl/deepseek41v, **off for qwen3vl** (2.17x faster but cos 0.9976 and drops page OCR). The 80-wide attention default is the bit-exact ILP shape `BQ=48,BK=32,WG=1024,SG16,RQ2/RQP2` (`PF_VIT_ATTN=70`). Multimodal prefill uses chunk-batched mode 2 (`PF_MM_BATCH_PREFILL`, off on the pipelined two-GPU path). The M=1 decode fuses activation quantization into the int8 GEMV group (`i8_grp_quant_gemv_launch`). An image consumes `max(nx,ny)` positions but contributes `4*nx*ny` tokens, and media tokens never reach `tok_embd` (the embed kernel copies `img_embd[img_row[t]]`).

Audio: WAV native or ffmpeg fallback → log-mel (16 kHz, 25/10 ms, 128 bins) → AuT-style tower from a **separate** `--audio-mmproj`. Its int5 linears stay on `vit_gemm` by default (`PF_AUDIO_DNNL` is opt-in: int8 moves the host/device error to 7.2e-2 against a 1.7e-2 gate). Video: uniform frame sampling (max 16 frames, side 768), M-RoPE `(frame,row,col)`. Mixed media is assembled by `mm_build_prompt_mixed_device`; media paths bypass the prefix cache and use single-sequence `generate_mm`.

## Major subsystems

### qwen4exp (`Qwen3.8-Flash-Next`)

104 GiB, 4 shards, 48 layers (36 GDN + 12 full attention), `n_embd` 2560; hyper-connections (4 streams, low rank 320), PLE on layer 1 (16 n-gram hash rows from a 26.8 GiB IQ4_NL table), QSA on the 12 full-attention layers (top 2048 of pooled-4 indexer keys), 512 routed experts top-10 + shared. The split GGUF is mapped into one reservation with `MAP_FIXED` per shard; the single-blob upload refuses it, so qwen4exp always uses per-tensor upload.

**Policy is the model profile** (`fwd = hc_ple_qsa`, `max_decode_rows = 1`, whole-prompt prefill, prefix cache on, `spec.multimodal = true` with `vision = qwen3vl`, `spec.mtp = false`, `moe = runtime_bank`). The `--layer-map` is synthesised when absent. Scheduler caps decode at one row and serialises sequences. MTP is gated with a printed reason.

**Placement**: dense trunk + per-expert LRU bank on the GPUs; PLE and non-resident experts in host RAM. `--moe-vram-mb 17500` → 121 slots/layer (the measured knee; 16000/17000 are worse, 18000 is not faster). The bank is per device (`PF_Q4_PERDEV_R`, default on). Expert uploads use a 32-slot pinned ring; the 69.8 GiB pinned source is filled with parallel `pread` (8 workers, 10.49 GB/s warm) and `PF_Q4_KEEPCACHE` keeps the page cache. Startup is 28 s warm / 109 s cold, peak RSS 5.8 GiB. Prefill is two-pass (a 512-token chunk selects up to 328 experts against R=111); the next batch's uploads are prefetched. `moe_reduce_launch` accumulates, so the banked branch must zero `b.block`. qwen4exp's own prefix-cache checkpoint adds PLE history and QSA pool/ring state; it resumes at q4 chunk boundaries.

**Measured (2x A770, recommended config)**: warm 3660-token prefill 65.4 s (55.9 tok/s), decode 88.2 ms/token at depth 512, 110.2 at 3.6k, 119.7 at 14.6k. `--q4-tail-skip` defaults to 0 (exact); N=7 measures 86.9 ms with a measured near-tie quality gate. 2-row batched decode is correct but 0.85x; MTP is a **no-go** (`test_q4_verify` n=1/3/5 = 257/416/497 ms against kill lines 190/230). QSA is implemented (host selector, device/paged KV gather; device indexer on by default); at `--ctx 2048` QSA is off entirely.

**Bring-up defects that fail silently**: Q5_1 nibbles are interleaved halves (`qs[i%16] >> 4*(i/16)`), not consecutive pairs; K=320 (HC up) and K=640 (shared down) overflow the generic 256-superblock GEMV; `ple_conv1d` is `[kernel][channel]`; the mode-1 prefill tail must take the head from the last token; the host activation quantizer rounds its stored scale to f16.

See design 18 for the full measurement log and the long dead-end list (host-spill split, whole-layer bank, full host mirror, frozen CPU misses, native 4-bit expert bank, Q5_1 panel repack — all measured and rejected).

### deepseek41 (`DeepSeek-V4.1-Flash`)

40-layer all-MoE, 64 query / **1** KV head, K==V 512-wide latent, 128-token sliding window, compressed KV owners `[2,8,14,20]`, DSA indexer 32x128 top 512, mHC 4 streams / 24 mix outputs / Sinkhorn 20, 384 experts top-6 + shared, Engram on layers 1 and 14, RMSNorm eps 1e-20.

Facts that fail silently: RoPE rotates the **last** 64 of 512 dims; K and V are the same latent and the attention output is de-roped before `wo_a`; compressed rows are group-shared (only owners write); candidate blocks only at layer 20; mHC's attention pre-mix is lagged one sublayer; router uses `sqrt(softplus)` with bias affecting selection only and weights `score/(Σ+1e-20)*1.5`; SwiGLU clamp is one-sided; the Engram table (110 GiB) is never uploaded; IQ3_XXS is a real type (layer 23); the GGUF chat transition is `<|Assistant|>` + `<think>`/`</think>`; `mm_prompt::n_rows` counts embedding rows.

Measured (2x A770): routed experts split across both cards (`PF_DS41_SPLIT`) following the link asymmetry, with a 4-way spin-waited staging crew (condvar wakes cost more than the copy); the generic KV pool is **not** allocated for ds41 (`[mem] kv=1`). Decode 353 → 124 ms/token with native Q5_K oneDNN stores; prefill 180 → 133. The store is exact in weights but not activation; the 32-token greedy text is identical on natural prompts, a forced stream diverges at a near-tie. See design 23.

### MinerU (`qwen2vl`)

24-layer dense Qwen2-VL, head_dim 64, no attention gate. Two load-bearing bugs: the q projection's head stride follows the presence of the attention gate (`qgate = gate ? 2 : 1`; a fixed `2*head_dim` makes every head read the next head's q, while a single-token teacher-forced result is still right), and K=896 is not a 256 multiple so those tensors are Q5_0/Q8_0 and must take the per-tensor/multi-device path. On one A770 the whole 1036x1036 page is 10.99/11.23 s warm with byte-identical output. MinerU image/chart extraction is always on.

### MTP / NextN

`--mtp N` drafts N tokens with the model's NextN head and verifies in one batched forward. Hard gates: the GGUF must bundle `blk.<n>.nextn.*` and the run must be a multi-device oneDNN int8 partition; otherwise one `[mtp]` line and plain decode. The draft consumes the trunk hidden *before* `output_norm`; the verify is dry (`step_info::mtp_dry`) with GDN/conv snapshots in `d_mtp_hist_`. The prefix cache commits after each completed prefill batch including the MTP layer's KV writes.

Measured 27B/2x A770: acceptance is the entire story; cycle cost is constant (~106 ms with adapt off) across prompts and speedup spans 1.33x–1.96x. MTP does not compose with batching (at S=8 plain batching wins 0.72x), so the server keeps it off by default (`PF_MTP_SERVER=1` restores it). It is a short-context win: at 128k the verify costs 1390 ms/cycle. Emitted stream is *greedy-equivalent in intent* but not byte-guaranteed (one ~130-token prompt diverges from token 11). The recurrent-state stride bug (`kMaxB` vs `state_slots_`) caused all three historical USM-copy segfaults and is fixed. Do not retry: 2-bit draft/head, candidate-restricted head, head split across cards, merging commit/rollback barriers.

### DFlash2 (`--spec-type dflash2`)

A 5-layer draft head runs one non-causal forward over `[anchor, MASK x n_max]` in a private K/V ring and a selector scores top-K candidate sets per block position plus ordered pairs. Model: block 8, `n_max` 5, selector rank 256/top 16. Acceptance is prompt-dependent: at n_max=5, code 4.25 → 21.8 ms/token (3.08x), technical 2.19 → 35.8 (1.88x), story 0.49 → 77.0 (**0.92x, slower than no drafter**). The emitted stream is byte-identical to plain greedy; `test_spec` asserts stream equality, not acceptance. The verify uses `mtp_verify`'s recorded graphs. Load-bearing bugs found by diffing against llama.cpp: `wk`/`wv` were never projected; the conv composed two coefficient tensors with mirrored axes; the draft's rope rotated 64 dims instead of 128. The top-k was 6 workgroups on a 512-EU GPU; slicing into 8 partials per row took 2.06 → 1.39 ms (S=8 is the optimum, not the occupancy maximum). **`PF_DFLASH_OPTIME` must not be read as device time** — A/B subtraction is not attribution on an in-order queue; the device timeline's `sum ≈ devspan` is the check.

### ONNX model service

Runs the document pipeline's layout/OCR/formula/table models. No OpenVINO/ONNX Runtime: a protobuf reader, graph IR with fixed-point transforms, operator library, SYCL kernels and a memory-planning executor. Every kernel takes shapes at runtime — the *graph* JIT is absent, the SYCL runtime JIT still applies to non-AOT builds. Inference is f32: bf16 made the layout detector 17% wrong. All twelve models load; seven covered by the corpus are exact or within f32 noise, the formula recogniser is integer-exact.

Key invariants: `q_copy_dev` from pageable memory is synchronous on this backend, so uploads are staged through pinned USM; a host value moved into the operator loop as a temporary is a use-after-free. The layout mask head is one `MaskBox` scan (9.5 → 0.2 ms) whose output keeps the leading batch axis `[B,300]`. oneDNN's matmul *bias* primitive is 4.5x slower, so bias stays in `bias_act_launch`. `Resize`'s source index must be integer arithmetic (a float reciprocal made 33/45 tables drop a leading character). Buffer borrowing is disabled in `Loop` bodies; carried values are device-resident, body inputs pinned, and a produced carried value gets a dedicated device buffer with its shape/count/dtype copied. Model results: end to end 865-893 ms/page on the bank report; the in-process pipeline is 0.996 s/page over 957 pages. VRAM: layout executor 443 MiB at N=1, ~0.42 GiB per image, N=24 fits. The NPU path (`src/backend/npu/`) is a separate Level Zero graph-extension executor: fixed-shape stages (OCR recogniser, classifiers) run there, precision-sensitive layout and dynamic-shape detectors do not. See design 17 and `docs/guides/onnx-inprocess.md`.

## Device profiles

**A constant measured on a particular GPU lives in that GPU's profile, not as a literal at its use site.** `src/device/profiles/<card>.cpp` is a complete implementation (values, key, matcher); `device_registry.cpp` holds `kDevices[]`; `device_profile.h` holds the struct and API. What belongs: occupancy, SLM budget, split heuristics, per-kernel shapes, vision tile, DFlash slices, i4 K rotation, and the *default* of measured on/off switches. What does not: model/run/host properties (`PF_Q4_TAIL_SKIP`, `--moe-vram-mb`, `--cpu-threads`, TB buckets, prefill chunk, QSA host flags), which would make a value look measured on a card it never was. A field marked INHERITED was not re-measured; say so.

Launchers use `si::dev::for_queue(q)` and `wg_clamped_for_queue(q, want)`, never `active()`/`wg_clamped()` — those resolve once from the first GPU, so a heterogeneous `--layer-map` would mis-tune the second card and could launch a work-group it cannot accept (A770 1024 threads vs Iris Xe 512). The lookup costs 0.039 us; an uncached `get_devices()` in `rmsnorm_launch` cost 3.2 ms/launch = 210 ms in one MTP verify. Do not put a profile read behind `static const = []{ ... active() ... }()`; use `profile_int`/`profile_flag`. `profile_int` folds env `0` into "unset", so a field whose `0` is a real value (`moe.cfg_dec`, split count 0) must read the raw env with a `-1` sentinel. `shape.rmsnorm_wg` is a hardware boundary; an unknown card warns with `key = unknown`. `PF_DEVICE_PROFILE` pins a profile, `-DSYCL_INFER_AOT_PROFILE=` bakes one; `PF_DEVICE_INFO=1` dumps it. Work-group widths must be template parameters. Write every field with a designated initializer. A layer map may mix profiles — this is supported, not an error. `CMakeLists.txt`'s `SYCL_INFER_AOT_DEVICE` is card-specific for the same reason.

## Model profiles

**A property of the model file or architecture lives in that architecture's profile, not in a model-flag ternary at the use site.** `model_profile.h` declares it; `src/model/arch/<arch>/profile.cpp` implements it completely with a `provenance` string. Geometry/file facts (RoPE layout, image floors, pad tokens, KV layout) go in loader-filled `hparams`, not profile policy. What belongs: forward kind, `policy.*` (max_decode_rows, prefill, prefix_cache, auto_layer_map, md_splits, engram default, keep_file_cache), `rep.*`, `moe` owner, `spec.*` capabilities with one `why`, and HC/PLE/QSA geometry. What does not: per-card numbers, or per-file/run facts such as expert plane sizes, bank budget, KV type, layer map, TB buckets, prefill chunk.

Exact-key detection: every `kLoaders[]` row carries a `const model_profile*`, and `model::load` refuses a row without one and refuses a key that is not byte-identical to `general.architecture`. An env var/CLI flag overrides a profile *default*, never a capability: `policy.prefix_cache == false` cannot be turned on by `PF_PREFIX_CACHE=1`, and `spec.mtp == false` prints `[mtp] <key>: <why> - MTP disabled` and stays off. There is deliberately no `PF_MODEL_PROFILE` pin. Runtime dispatch goes through `arch_ops_for(...)` (`engine::aops_`), not through the profile. `PF_MODEL_INFO=1` prints the resolved profile. Pin values in `test_model_profile` / `test_arch_ops`.

## Conventions

* C++17, `namespace si`, 4-space indent, Allman braces. Types/functions `snake_case`, constants `kFoo`, device pointers `d_foo`, host copies `h_foo`.
* Comments explain *why* (often a measurement) — keep them when refactoring.
* Reuse or extend existing engine machinery before building a parallel path; a new subsystem needs a measured improvement recorded in `docs/PERFORMANCE.md`. The q4 CPU-miss path (`PF_Q4_CPU_MISS`, 1.49x slower) is the cautionary example.
* Keep CMake source lists updated; no globbing. Do not edit `third_party/`; one documented exception is a 2-line `is undefined` patch in `third_party/minja/minja.hpp`.
* `Diagnostics.UnusedIncludes: Strict` must stay clean; check with `clang-tidy -p build -checks='-*,misc-include-cleaner' <file>`.
* **Never write a real IP address, host name, SSH account or private home path into a doc, comment, source file or commit message.** Use `bench-host`, `proxy.internal:8118`, `/path/to/<file>.gguf`. RFC1918 *ranges* used by the SSRF classifier are constants and must stay.
* Do not commit unless explicitly asked.

## Environment variables

Unset/`0` = default unless noted. These select different kernel variants, so a performance number must state the env used. The full list, including the many one-off diagnostics, is in the original subsystem docs; the load-bearing ones:

**Model / memory**
`PF_CTX`, `PF_KV_CAP_MB`, `PF_KV_GROW` (default 64 blocks), `PF_MAX_SLOTS` (default 16; sizes recurrent state), `PF_KV_TYPE` (`i4|i8|bf16|f16|f32`, `K:V` mixing; `--kv-type` overrides), `PF_KV_ROT` (i4 K Hadamard rotation; on by default for GPU i4 via `attn.kv_rot`, opt-in on CPU; `pc_disk` v2 records skip v1), `PF_SI4`, `PF_META`.

**Compute / attention**
`PF_F32`, `PF_VISION_F32`, `PF_DEVICE` (`cpu|gpu`), `PF_CPU_ISA` (`scalar|avx2|avx512|avxvnni`), `PF_CPU_THREADS`, `PF_DP4A` / `PF_DP4A_DEC` (default on), `PF_W4` / `PF_K5` / `PF_CB4` (native stores, default on; `PF_W4_K5` and `PF_W4_ALL` are lossy and off), `PF_W4_GROUP` (32/64/128; 64/128 flip near-ties), `PF_HEAD_U4` (default on), `PF_GEMM_DNNL`, `PF_ATTN_XMX` (default on above 2048 keys), `PF_ATTN_XMX_MIN` (2048), `PF_ATTN_SPLIT_KEYS` (512), `PF_DEC_SPLIT` (from device profile; 160 on a 512-EU A770 for 27B's 24 heads — past 8 warps/EU the time steps up), `PF_DEC_GROUP`, `PF_ATTN_STREAM=1` (data race), `PF_QUEUE_PROF` (off by default on 2026.2), `PF_GDN_COLS` / `_WG` / `_VEC` / `_FUSE` (default fused mode 2; float4 only when `n_real < 2`), `PF_NAT` (0 disables `nat_gemm`).

**Prefix cache / server**
`PF_PREFIX_CACHE` (on), `PF_PC_STATES` (8), `PF_PC_VRAM_MB` (`PF_PC_MEM_MB` alias), `PF_PC_RAM_MB` (512), `PF_PC_DIR`, `PF_PC_DISK_MB` (1024), `PF_MM_URL_FETCH` (off), `PF_MM_URL_ALLOW_PRIVATE` (off), `PF_AV_FFMPEG` (`ffmpeg`), `PF_YARN` / `PF_YARN_FACTOR` / `_ORIG_CTX` / `_BETA_FAST` / `_BETA_SLOW` / `_ATTN_FACTOR`.

**ONNX path**
`PF_ONNX_CONV_DNNL` (1 = second sighting of a key and `N >= 8`; 0 = SYCL always), `PF_ONNX_DNNL=0`, `PF_ONNX_MATMUL1X1=0`, `PF_ONNX_MM_DNNL_MIN`, `PF_ONNX_DNNL_BF16=1` (A/B, not default). Stream ordering: `conv_dnnl.cpp` waits the queue before `execute` and the stream after it; `exec.cpp` waits the queue per operator when oneDNN is enabled plus once before readback. `SI_ONNX_NO_OPBARRIER=1` removes that wait, `SI_ONNX_OPBARRIER_ALL=1` forces it with oneDNN off, `SI_ONNX_END_WAIT=1` adds an end-of-call wait — all A/Bs. Skipping them produced all-NaN logits in the recogniser.

**NPU device** (no `PF_*` knob): `device: "npu"` in a model's config, `--model id=path:npu`, or `syi_onnx_create(..., "npu", ...)` selects the independent Level Zero graph path; `"map:<file>"` routes each ONNX basename to a device. Static shapes: `PF_NPU_STATIC_DIMS`, `PF_NPU_INPUT_SHAPES`, `PF_NPU_IR_FILE`; `PF_NPU_BUILD_FLAGS`, `PF_NPU_DBG`, `PF_NPU_IRDBG`, `PF_NPU_SHAPE_MAP`, `PF_NPU_NO_BN_FOLD`, `PF_NPU_NO_BN_BIAS`, `PF_NPU_PAD_ZERO`, `PF_NPU_STRIDED`, `PF_NPU_SLICE_SAFE` are development switches. No OpenVINO is linked.

**MoE experts**
`PF_MOE_RESIDENT`, `PF_MOE_VRAM_MB`, `PF_MOE_RAM_MB`, `PF_MOE_DP4A` / `_DEC` (default from profile, on), `PF_MOE_CFG` / `_DOWN` / `_DEC` (tile case; `_DEC` 0 = auto: M=1 single decode, M=2 batched), `PF_MOE_FLAT` / `_DEC`, `PF_MOE_PERSIST` (exact but 3% slower — off), `PF_MOE_NPB` (6-7% then occupancy loss), `PF_MOE_HIST`, `PF_MOE_REFDUMP_ALL` (layer-0-only dumps are a verifier that agrees with the bug). `PF_MOE_BANK_AUTO` / `_RESERVE_MB` / `_DOWN_I8` / `_U4BANK` (Arc 140V governor).

**Diagnostics**
`PF_GUIDED_DEBUG`, `PF_MODEL_INFO`, `PF_MEM_REPORT` (deterministic own accounting; the driver's `free_memory` is not an oracle), `PF_NOGRAPH`, `PF_PROF`, `PF_HOSTPROF` (host submission cost without device waits), `PF_TIME`, `PF_QUEUE_PROF`, `PF_DUMP_LAYERS` (a round trip *inside* the forward that can supply a missing sync), `PF_DUMP_PROMPT`, `PF_DUMP_GEN`, `PF_DUMP_KV`, `PF_DUMP_DEC_LOGITS` (tests only). The many `PF_MTP_*`, `PF_DFLASH_*`, `PF_Q4_*`, `PF_QSA_*` switches are A/Bs and dumps named in the subsystem docs above; `PF_Q4_TAIL_SKIP` defaults to 0, `PF_QSA_DEVATTN`/`_IDX_DEV`/`_EARLY`/`_PF_DEV` are on, `PF_QSA_SEL_DEV` is off.

## Measured results and dead ends

A throughput number belongs in `docs/PERFORMANCE.md`, not only here. When a change moves a number: update that table with model/layer split/KV type/harness/command/date; update this file only if the *shape* changed; fix every quote of the old number in the same change.

**35B A3B on Arc 140V** (single device, `PF_W4_ALL=1`): prefill 641 tok/s at 3.4k, decode ~30 tok/s; decode is 70% weight stream at 87-92 GB/s of 107.4. The dense u4 store is quality-neutral there. The expert banks are a prefill/decode trade; the memory governor sheds the down bank first.

**Routed experts**: the grouped dp4a GEMM over native GGUF blocks is default for prefill and decode. The experts are 18.22 GiB; decode reads `8/256` per token per layer at 16-26 GB/s against a 376 GB/s card because the K-quant layout gives 32 B per load instruction instead of 128. The decode's tile is a *shape*: M=1 for one token, M=2 for batched — the prefill's M=8 is seven eighths empty on decode. `dec_step` must mean `mode == 0`, not `n_tok <= 1`. Dead ends: dp4a rewriting the fp32 path (13x slower), SLM row staging, pair-tiling >8, transposing the weights (2.6x worse and 18 GiB more), persistent work-stealing (exact, 3% slower), DPAS (loses at the MoE's register ceiling).

**Batched decode**: run exactly `n_rows` rows — the old bucket set `{1,2,4,8,16}` rounded up, so 20 rows silently computed 16 and the sampler read stale logits. The native GEMM's M<=13 boundary was a real cliff on the 2026-10-20 build; `nat_gemm_big` now covers M=14..32 for u4/int8, so that curve is unmeasured on current code. The host is not the bottleneck; the sampler's host argmax costs ~0.45 ms/row and is a real product cost.

**Attention**: prefill uses oneDNN int8 matmuls (8.3/14.4 T-MAC/s), the fastest thing there. The gather's block-max reduction must be spread over `kGatherRed = 64` work-groups (single group: 18.1 of 37.8 ms). Decode attention is **occupancy** bound, not bandwidth: `n_head * n_splits` warps, 8/EU, with a cliff at 8.25 warps/EU (27B: 1.70 ms at 160 splits, 2.68 at 176). Do not share K/V loads across query heads (5x slower), unroll the key loop, or widen a lane to 16 dims (128 registers and spill). `PF_DEC_GROUP` is unavailable on models with 8:1 GQA. Whether the 8x GQA re-read hits L2 is **not measured** — the next step there is a counter, not a kernel.

**Long context**: decode is `37.8 ms + 0.202 us * depth` for the 35B, no knee; at 256k, 43% is attention. Prefill peaks at ~16k and falls after. At 131k the routed experts are 62% of prefill and the oneDNN attention is worth 10x. Capacity arithmetic, `--max-slots` headroom, 1M/4-card conclusion and the PCIe measurements are in the tables.

### How to measure here

Everything below, and the README's throughput table, was measured on `bench-host`
(two Arc A770 16 GB, `--layer-map 0-31:gpu.0,32-63:gpu.1`, i8 KV, AOT Release,
ctx 131072, `PF_PREFIX_CACHE=0`). Five things about that setup silently produce a
wrong number rather than an error:

* **Verify the remote runs the code you think it is.** The box is a source
  snapshot without git, synced file by file, so it drifts: `md5sum -c` a manifest
  of `src`/`tests`/`CMakeLists.txt` before trusting a number, and `touch` the
  sources after syncing (`rsync` preserves mtimes, so make otherwise skips the
  recompile and you read a stale binary's output).
* **Do not pin OpenCL to "make the selector deterministic".** `gpu_selector_v`
  used to pick this box's OpenCL GPU, where the decode graph does not exist
  (exit 78). `src/device/device_select.h` now prefers Level Zero; an explicit
  `ONEAPI_DEVICE_SELECTOR` still wins. `OPENCL` as a workaround measures a path
  the engine cannot run.
* **`llama-benchy` needs `--tokenizer Qwen/Qwen3.8-27B` and only `https_proxy`.**
  The reported model id is not a Hub repo id — asking for it 401s and burns
  minutes with the GPU idle. Never set `http_proxy`: the endpoint is
  `http://127.0.0.1:<port>` and a proxy there measures an error page. Use
  `PYTHONUNBUFFERED=1` and `--emit-progress` too.
* **Its `pp512` column is wrong at depth > 0 and must not be quoted.** It divides
  the prompt length by the time to the first SSE chunk, which the server sends
  *before* prefill (73 830 tok/s at 16k). The only surviving number at depth > 0
  is `prompt_tokens / e2e_ttft`; at depth 0 the isolated prefill is `est_ppt`.
* **A wall-clock plain-decode baseline includes model load.** The 27B takes ~80 s
  to load, so timing `gen` end to end gives ~690 ms/token (a 10x error). Instead
  difference two runs: `wall(N+128) - wall(N)` cancels the load, page-out and
  prefill. Differencing beats subtracting a 1-token run because it keeps the
  load jitter out rather than dividing it by the token count.

Two more traps: **one process per cell, heavy jobs serial** — a prefix-cache hit
inside one process moves acceptance enough to change the ranking, and two GPUs
jobs stacked drop a card with `UR_RESULT_ERROR_DEVICE_LOST` that looks like a code
regression; `set -u`/`set -e` must come **after** `source setvars.sh`, and a
backgrounded server needs `setsid`. **Do not merge a diagnostic stream into the
generated text** (`2>&1` interleaves timing lines mid-token and breaks a byte
comparison), and remember a speedup is a ratio — both sides need the same shape
(graphed decode against graphed decode).

An oracle ladder from token hash (too weak) to per-layer fingerprint (worse than
useless — it supplies a missing sync) is how these bugs were chased. An
instrument that changes the schedule moves the bug. A/B subtraction is not
attribution on an in-order queue; check a breakdown against an independent total
(`devspan`). The box drifts ~13% between days — compare cells in one run and
quote best-of.

`dev/` harnesses are gitignored, so they rot silently. `dev/bench_mtp.cpp` once
stopped compiling after the MTP state moved to `spec/mtp_state.h`; the bare
`engine::mtp_k` that remains is the *config* field, so `e.mtp_k = k` silently
became a no-op that would have measured a plain decode as speculative. They are
one-file tools compiled by hand against a **current JIT** archive
(`-Lbuild-opencode-jit`; linking the AOT library dies in the runtime — see *AOT
builds*), re-lowering the device image at each link.

## Invariants and gotchas

* **A kernel without a unit test is not finished.** `ci/kernel_test_map.txt` + `ci/check_kernel_coverage.py` are the mechanical gate; the `ggml_float_to_half` subnormal bug and the Q6_K 8-lane expert index bug both changed numbers no aggregate could see.
* **A generic engine/model file must not name an architecture, a per-model state member or a model-owned `fwd_kind`.** Reach architecture behaviour only through `model_profile`, `arch_ops`, `arch_hparams`/`arch_layer`/`arch_model_extras` virtuals, or the architecture's own code. `ci/check_arch_boundaries.py` enforces it.
* **`model.h` describes the file, not the device representation.** No `<sycl/sycl.hpp>`, `sycl::`, `dev_weights`, `upload()`, `build_w8` or `dev_ptr` — those belong to `src/backend/weight_store.*`. The gate also fails on a reintroduced `src/model/model_w8.cpp`.
* **The single raw device weight store is shared by the media towers and the drafter** via `upload_contiguous`/`upload_spans`; their headers must not declare `dev_weights`, `dev_weights_size` or `dev_reserved`.
* **A profiling-enabled queue is not free on 2026.2**: 0.133 ms vs 0.0055 ms per launch (24x), 109 vs 700 tok/s for 35B prefill. Created only for device-timeline diagnostics; `PF_QUEUE_PROF=1/0` overrides.
* **Each child process has exactly one owner and one `waitpid`.** Two waiters race; the loser sees `ECHILD` and SIGKILLs a worker mid-flush. `test_router_control`'s graceful-shutdown case is the regression.
* **A worker that cannot start exits 78 (`EX_CONFIG`) and is not retried**; otherwise phase decides (death before ready = startup, after = crash).
* **Count VRAM where it is allocated**: Level Zero's free-memory aspect either is missing or ignores SYCL allocations. The control plane reports `total - used` from the engine's own accounting.
* **`compute_backend::synchronize()` must actually wait.** The CPU override was empty (`-fno-sycl`, no queue) and turned ~24 call sites into a heap use-after-free; it was only visible in `test_pc_cpu` because the corruption was in a later copy, not the KV one. An override with no effect must assert, not be empty.
* **Heterogeneous `--layer-map` is supported.** `active()`/`wg_clamped()` resolve once from the first GPU; use `for_queue`/`wg_clamped_for_queue`. Do not put a profile read behind a `static const` lambda, and never query the device from a launcher (3.2 ms/launch uncached). `wg_clamped` guards the hardware work-group boundary (A770 1024 vs Iris Xe 512).
* **"The projector loaded" is not "the projector can run".** `vision_model::runnable()` must be extended with every new geometry constraint, or the failure comes after 23 checks have passed.
* **The GGUF mmap is paged out after weights reach a device.** `madvise(MADV_DONTNEED)` + `posix_fadvise` keep the mapping valid but any later host read re-faults from disk. CPU partitions stay in the keep set.
* **A converted tensor must have a device copy or a plan `xq` entry, never neither** — otherwise the GPU reads a released host VA and the run dies with `DEVICE_LOST`.
* **Graph capture freezes host values.** Per-step state must be read from host-USM `step_info` inside the kernel, never on the host. `record_forward`'s call order must stay in sync with `build_plan`. A recorded graph cannot contain a host pass (`moe_host_split()` has three replay sites), and the host must rebind device pointers on entry.
* **The recurrent state's slot axis is `state_slots_`, not `kMaxB`** (they differ by default: 16 vs 32). Mixing them is a silent out-of-bounds write into a USM pool. `zero_slot`, `pc_admit`, `pc_capture_begin`, `pc_commit` bound-check against `state_slots_`; the MTP verify sites do too.
* **A prefix-cache checkpoint is resumable only at a completed 32-token block**, and the capture takes the deepest one even if the batch ends mid-block (`cap_b = (L-1)/32` on the prompt's last batch); `pos0 % 32 == 0` stays a hard prerequisite. `kPcMapLen` is a ring indexed by `absolute_boundary % kPcMapLen`, not a depth cap.
* **YaRN's two blends are written in difference form and are exact at their anchors**; the convex form makes the untouched head a 1-ulp ramp and `test_yarn` fails.
* **A plan is a device-pointer snapshot.** The head is built after `bind_acts(0)`; `cur_dev` must be reset to 0 before the head so `gemv_at` reaches the primary device's oneDNN table (12.3 vs 3.4 ms/token). An untied head's key is the host pointer, a tied head's is the uploaded device pointer. `handoff_x` copies only live rows.
* **A partial `record_forward` starts its call cursor at the phase's first layer** (`seg_plan::layer_c0`); restarting at 0 makes later partitions read the first layer's call metadata.
* **Multi-device decode is graphed per contiguous device run, for `n_rows == 1` only** — oneDNN's int8 matmul cannot be recorded into a SYCL graph.
* **GDN head pairing is modulo, attention GQA is blocked** — do not unify them (they agree only on 0.8B). `qkv_dim()` is not `3*d_inner` and `m.output` is not always `m.tok_embd`.
* **"XMX" here names oneDNN's int8 matmul, not necessarily the XMX unit.** oneDNN 3.11.4 had no XMX kernel; 3.13.2 does. Hand-written DPAS was implemented, exact and 37x slower in a real GEMM — do not re-derive it.
* **A SYCL stream built from an in-order queue is not ordered against that queue's submissions.** Move attention matmuls to `p->st` (`PF_ATTN_STREAM=1` reproduces the bug); the ONNX path hit the same class independently (see *Environment variables*).
* **oneDNN eligibility is per call and all-or-nothing** — one f32 tensor takes the whole call to the fp32 GEMV (12.3 vs 3.4 ms/token). A partial mode-2 batch requires oneDNN. An all-`cpu` `--layer-map` must select the CPU queue.
* **The load path is lazy and staged.** Text primitives are built on first use (`PF_DNNL_LAZY`, default on for a multi-device partition with no MTP/DFlash; M=1 is still primed because it is recorded into the decode graph). The int8 converter uses a vectorised round (no `lround`; `PF_DNNL_QCHK=1` recomputes one reference row per tensor) and two pinned staging slots so a tensor's H2D overlaps the next tensor's dequantization (`PF_DNNL_NOSTAGE=1` / `PF_DNNL_SYNCCOPY=1` are the A/Bs). The tied-embedding copy stays on the CPU. `PF_LOAD_TIME=1` prints the split.
* **The fully-fused mode-2 GDN call used to process only the first chunk row** (fixed). `gdn_kernel`/`gdn_f4_kernel` ignored `nreal_arg` and used `row_nr(info, rr)`, so only row 0 got the recurrence and every later chunk row read a stale state; the CPU twin already honoured it. The scheduler also lets mode-2 prefill the tail (`>= kMaxT` instead of `>= 2*kMaxT`): a mode-1 chunk costs ~0.26 ms of handoff/sync per token slot on 2x A770, and a 553-token prompt went 327 → 399 tok/s. Mode-2 still costs ~0.26 s of fixed weight-stream per forward, so 512 tokens is the efficient unit.
* **A routed-MoE model's prefill is still the largest open item.** At 131k the routed experts are 62% of prefill and the oneDNN attention is worth 10x; do not "fix" it with a parallel path that has no measured win.
* **`ggml_float_to_half` flushed the f16 subnormal range to zero**, destroying weight-store scales (0.205% of columns in one 27B tensor at 240% relative L2). `test_f16conv` pins the bit patterns and the range property.
* **A criterion built on per-element relative error flags cancellation, not defects.** Judge survivors against the exact fp32 dot and `sum_k |w_k x_k|`; `PF_W4_NOCORR=1` is the red half.
* **Any activation feeding a native-store GEMV must be quantized with `do_split=true`.** `quantize()` now returns an `si::act_view` and every GEMM requires it, so forgetting the gate does not compile. A stale view returns `nullptr`.
* **The two drafters share exactly one piece of device state**, `verify_scratch` (argmax id + value). Allocate it through `mtp.verify.ensure(q)`; `test_spec` runs one drafter per process and cannot catch a null value side.
* **The scheduler used to serialise every request** (mutex held across engine calls). Releasing it gives 14.3 → 55.0 tok/s at N=8. Concurrency is deterministic but not bit-exact against a single request; do not assert greedy equality across a batch. Cross-sequence *prefill* batching corrupts and stays disabled.
* **`test_sched` is the only batched-decode test**, with a 20-row oracle — it has been red and is a verifier, not a decoration.
* **The host pass that fills a sparse buffer must run before its consumer** (`moe_down_launch` returns on `slot < 0`; stale rows were double-counted). A per-layer diagnostic that reports only layer 0 agrees with that bug; use `PF_MOE_REFDUMP_ALL`.
* **A handoff to a CPU partition must be waited** — CPU kernels are host `par()` workers with nothing to order them against the queue. Symptom: the GPU layer bit-identical, the next layer's MoE input 63% too large, prefill fine, first token wrong.
* **A hand-coded router that loops `n_expert` inside one work-item is a parallelism bug**: 4.4 s of a 4.6 s decode. One item per (token, expert) → 48 ms. K-parallel reduction changes its fp summation order and moves MTP at k>=2.
* **A MoE model's expert strides are per layer, not per tensor** (the 35B has Q4_K gate/up, Q5_K down and Q6_K on 3 layers); one `plane_bytes` reads wrong weights on exactly those layers. A Q6_K 32-value group is NOT 32 bytes and `moe_grp_impl` is an 8-lane split — the 32-lane index dropped half the superblock and cost mean 0.85 nat.
* **The GGUF's expert tensors are three separate tensors**; a packed `[gate][up][down]` cache needs different base pointers from the mmap path.
* **A page-release sweep that asks a component about itself must run after that component has decided** (`setup_moe_slots()` runs inside `alloc_buffers()`, before `release_host_weight_pages()`).
* **MTP's draft layer does not share the trunk's activation buffers** (`d_mtp_xnorm`); every launcher boundary taking "the activation" must be told which one.
* **A routed MoE model runs on the CPU backend too**; the MoE branch dispatches on the partition's kind, not `multi_dev`, and `src/backend/cpu/kernels/moe.cpp` mirrors every kernel step for step.
* **The HTTP API has no authentication.** `--host` defaults to loopback and `http(s)://` media fetch needs `PF_MM_URL_FETCH=1`. `url_fetch.cpp` requires every resolved address to be public and walks redirects by hand (httplib's `set_follow_location` re-resolves without a hook). Resolve, do not string-match (decimal/octal loopback, IPv4-mapped IPv6); a hostile DNS can still win the re-resolve race.
* **httplib signals client disconnect in one place**: `ContentProviderResourceReleaser`'s `bool` is false on a write failure *and* when the peer vanished between items. `serve_sse_chunked` is the wiring; `test_sse_cancel` drives a real TCP RST. The cancel must reach the producers (callback return is the only in-forward cancellation point).
* **Multimodal M-RoPE layout is a loader fact, not a derived one** (`hparams::mrope_interleaved`); blocked and interleaved sections can have the same sum. The Qwen3-VL preprocessing floor is `shortest_edge = 65536` (`vision_hparams::min_pixels`). Decode keeps `info->pos` = token count and carries the RoPE position in `step_info::mrope` with `mrope_on`.
* **ONNX invariants** (full detail and the oracle in design 17): a subgraph may read the enclosing graph's values (outer scope), copied once **per call** and kept alive until the copy; a `Loop` body's carried values are read next iteration, so per-op buffer borrowing is disabled there and body inputs are pinned; a `CumSum` keeps the axis (it is not a Reduce); a `Loop`'s trip count is optional (omitted = -1, not "unlimited"); a boolean is a float 0/1 on device and one byte on the wire; an `If` branch may declare an output produced by the enclosing scope (the formula recogniser's empty then-branch), and a constant inside a branch must run as an operator; `ScatterElements` updates have either the indices' or the data's shape; `GatherND` addresses with a flat trailing run; `TopK`'s `largest` is a runtime input for the layout decoder; `Split`'s sizes are an input since opset 13; `HardSwish` has no attributes and `HardSigmoid` is alpha=0.2 at every opset; a broadcast flag must say *which* operand varies, and the op is not commutative; `ConvTranspose` has `output_padding` and `output_shape`; `Tile`'s input can be int64; `Slice` keeps its rank; `Concat` promotes rank-0; `Gemm` has a bias; activation parameters travel by value; an integer value that is device-reduced must be reduced in its own type; a host-only integer value has no device copy; an operator's output type is not always its input's; an unimplemented operator in a folded subgraph is an error, not an empty output; a dead graph output still holds a buffer; a `TopK` quadratic in k looks like a hang (60 s/page); a `Cast` writes the output's width; a rank-0 value has no axis; `Loop` outputs are the final values and scans, the condition is an input; the guard band and per-op wait were 2.7x the formula model's device time and are off by default.

## Quick verification before finishing a change

```bash
cmake --build build -j$(nproc)          # warning-free except linker notes
ctest --test-dir build -L hermetic --output-on-failure
python3 ci/check_arch_boundaries.py
python3 ci/check_kernel_coverage.py     # or ctest -R kernel_coverage
ctest --test-dir build -R kernel_coverage --output-on-failure
for t in test_cpu_elementwise_units test_cpu_attn_units test_cpu_gemv_units \
         test_cpu_moe_qsa_units test_cpu_q4_units; do ./build/$t; done
./build/test_kernel_units ./build/test_kernel_units_native
./build/test_kernel_units_moe_qsa ./build/test_kernel_units_moe2 ./build/test_kernel_units_attn2
./build/test_gpu_stages ./build/test_gpu_vs_ref ./build/test_forward
./build/test_decode_vs_prefill ./build/test_sched
./build/test_capi <model.gguf>
./build/test_onnx_api <model.onnx> GPU
./build/test_embed_tokenizer
./build/test_embed_bge <bge-m3.gguf>
./build/test_capi_ffn <model.gguf>
./build/test_onnx_npu <model.onnx>      # NPU machine only
python3 -m sycl_infer.doctor
clang-tidy -p build --extra-arg=-I/opt/intel/oneapi/compiler/2026.2/include \
    -checks='-*,misc-include-cleaner' <changed files>
```

Router-only changes (no GPU, no model, no SYCL):

```bash
cmake --build build --target sycl-infer-router test_router_config test_router_proxy \
    test_router_supervisor test_router_control -j$(nproc)
ctest --test-dir build -R router --output-on-failure
ldd build/sycl-infer-router | grep -c sycl    # must be 0
cmake -S . -B build-router-only -DSI_ROUTER_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-router-only -j$(nproc)
```

For the 27B (needs the layer split): `TEST_LAYER_MAP=0-31:gpu.0,32-63:gpu.1`
with `test_w4_vs_cpuref` and `test_decode_vs_prefill`. `test_decode_vs_prefill`
also has `TEST_MAX_SLOTS=1` (must be numerically invisible) and
`TEST_YARN_FACTOR=4` (margins move, argmax does not) hooks, plus
`TEST_LAYER_MAP=0-11:gpu.0,12-23:cpu` for the tied-head hybrid path. Run these
on an A770, not a laptop iGPU: a Debug build pays a multi-minute JIT per
process there. `rsync -a` preserves source mtimes — `touch` sources first.
