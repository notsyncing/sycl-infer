# AGENTS.md

Guidance for AI coding agents working on **sycl-infer**.

## What this is

A general-purpose from-scratch C++17 + SYCL LLM inference engine for quantized
GGUF models on Intel GPUs.  The reference model used for development and testing
is **Qwen3.5-0.8B (Q4_K_M)** on Iris Xe-LP, but the engine is architecture-
agnostic: model types are selected through the registry in
`src/model/model_arch.h` / `model.cpp`.  It contains: a GGUF loader, a quantized
SYCL kernel library, a paged-KV continuous-batching engine with a reusable prefix
cache, and an OpenAI-compatible HTTP server plus a CLI.

The GPU path must keep working on `sycl::gpu_selector_v` and remains the default.
A host CPU backend (`src/backend/cpu`, selected with `--device cpu`) implements the same
forward pass on the host and picks AVX2 / AVX-VNNI / AVX-512 at run time.  Dense
(or hybrid) models can also be split across devices by layer with `--layer-map`
(pipeline parallel: each device runs its contiguous layer range and hands the
hidden state to the next), in which case each device owns the paged KV of the
attention layers it computes.

## Documentation

Detailed design docs live in [`docs/`](docs/README.md); read the relevant one
before changing a subsystem.  They are the source of truth for *why* things are
built the way they are; this file is the quick-reference for conventions and
gotchas.

* [`docs/architecture.md`](docs/architecture.md) — overall architecture, module
  map, runtime object model, data flow, threading, device memory, global
  invariants, extension points.
* [`docs/design/01-model-loading.md`](docs/design/01-model-loading.md) — GGUF
  parser, mmap, architecture registry, tensor binding, single-blob device
  upload, `dev_ptr`, CLI flags.
* [`docs/design/02-quantization.md`](docs/design/02-quantization.md) — ggml
  K-quant layouts, SIn int8 weight format, DP4A math, `xq` activation
  quantization, the oneDNN path and `PF_*` quant switches.
* [`docs/design/03-kernels.md`](docs/design/03-kernels.md) — every SYCL kernel:
  layouts, launch geometry, tuning knobs, how to add one.
* [`docs/design/04-engine.md`](docs/design/04-engine.md) — `seg_plan`,
  `build_plan`, `record_forward`, command graphs, prefill/decode modes,
  `step_info` and the graph-freeze invariant.
* [`docs/design/05-kv-cache.md`](docs/design/05-kv-cache.md) — paged KV, the
  virtual-USM dynamic block pool, block allocator, KV storage types.
* [`docs/design/06-prefix-cache.md`](docs/design/06-prefix-cache.md) — chained
  hashes, recurrent-state checkpoints, VRAM/RAM/disk tiers, budgets, flush.
* [`docs/design/07-sampler.md`](docs/design/07-sampler.md) — sampling.
* [`docs/design/08-tokenizer.md`](docs/design/08-tokenizer.md) — byte-level BPE
  and chat templates.
* [`docs/design/09-server.md`](docs/design/09-server.md) — OpenAI HTTP API,
  continuous-batching scheduler, SSE, stop strings.
* [`docs/design/10-multimodal.md`](docs/design/10-multimodal.md) — image
  preprocessing, vision encoder (host + device), M-RoPE assembly.
* [`docs/design/11-qwen35-model.md`](docs/design/11-qwen35-model.md) — the
  Qwen3.5 hybrid GDN + full-attention model.
* [`docs/design/12-build-and-testing.md`](docs/design/12-build-and-testing.md) —
  build targets, test matrix, verification flow.

## Build

```bash
source /opt/intel/oneapi/setvars.sh     # icpx, oneDNN, SYCL runtime on PATH/LD_LIBRARY_PATH
cmake -S . -B build
cmake --build build -j$(nproc)
```

* `icpx` is required; CMake sets `CMAKE_CXX_COMPILER` to it and links with `-fsycl`.
* oneDNN is expected at `/opt/intel/oneapi/dnnl/2026.0` (`-DDNNL_ROOT=` to override).
* `-DSYCL_INFER_AOT=ON` AOT-compiles the device image for `adl-p`.
* **Debug builds force `-O2` for device code** (see CMakeLists comments).  At
  `-O0` the IGC device image of `src/backend/gpu/kernels/*` becomes huge and its
  translation takes minutes (looks like a hang) — do not remove that.
* Source files are listed explicitly in `CMakeLists.txt` (no globbing).  Add new
  `.cpp` files there.

## Run

Binary/test startup needs the oneAPI runtime libraries.  `source
/opt/intel/oneapi/setvars.sh` handles it; the minimal override used in CI-like
runs is:

```bash
export LD_LIBRARY_PATH=/opt/intel/oneapi/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:$LD_LIBRARY_PATH
```

```bash
./build/sycl-infer --model /path/to/Qwen3.5-0.8B-Q4_K_M.gguf serve
./build/sycl-infer --model ... gen --prompt "Hello" --max-tokens 32
```

`main.cpp` documents the flags (`--model --ctx --blocks --kv-cap-mb --kv-type
--port --host --device --cpu-threads --layer-map --mmproj`, and for `gen`:
`--prompt --max-tokens --temp --top-p --top-k --raw --image`).  `--device cpu|gpu|auto`
selects the compute backend; `--cpu-threads N` (or `PF_CPU_THREADS`) sets the
host worker count;
`--layer-map 0-11:gpu,12-23:cpu` places closed layer ranges on devices (each
range must cover the layer list without gaps).

## Testing

The tests default to `/path/to/Qwen3.5-0.8B-Q4_K_M.gguf` and require the
GPU + that model.  Strict kernel/end-to-end tests set `PF_DP4A=0` (fp32 path).

```bash
./build/test_tokenizer     # tokenizer round-trips (CPU only, no GPU work)
./build/test_chat_template # GGUF chat template vs reference Jinja2 output (CPU only)
./build/test_multimodal    # image preprocessing, vision encoder (host + device), positions (CPU+GPU)
./build/test_compare       # CPU reference vs llama.cpp dumps (CPU only)
./build/test_cpuref        # CPU reference head (CPU only)
./build/test_gpu_stages    # every kernel vs the CPU reference (GPU)
./build/test_gemv          # GEMV vs CPU dequant reference (GPU)
./build/test_dp4a          # SIn repack + DP4A GEMM vs CPU reference (GPU)
./build/test_gpu_vs_ref    # end-to-end logits vs CPU reference (GPU)
./build/test_forward       # end-to-end logits / top-k (GPU)
```

Always run at least `test_gpu_stages`, `test_gpu_vs_ref` and `test_forward`
after touching kernels/engine, and confirm they still pass (they print
`all stages OK`, `argmax ... SAME`, a stable `last_id`).

## Repository layout

```
src/common/     quant.h (ggml block formats + host dequant), w8.{h,cpp} (SIn
                int8 weight format/repack), dp4a.h (portable dp4a helper),
                cpu_isa.{h,cpp} (host CPU feature detection + ISA dispatch)
src/backend/    backend.h (compute_backend abstraction), dnnl_gemm.{h,cpp}
                (optional oneDNN int8 matmul, PF_GEMM_DNNL),
                gpu/gpu_backend.cpp (forwards to the SYCL kernels),
                cpu/cpu_backend.cpp (converts the PODs, calls the host kernels),
                cpu/cpu_types.h (SYCL-free mirror structs + launch decls)
src/backend/cpu/kernels/    one .cpp per host kernel (compiled without -fsycl,
                AVX2 / AVX-VNNI / AVX-512 target variants): common (pool + ISA
                dispatch + dequant/RMSNorm helpers), rmsnorm, embed, copy_row,
                gemv, qk_norm_rope, attn, conv, gdn, gated_norm, xq, dp4a,
                i8 (integer GEMV straight from the GGUF blocks)
src/backend/gpu/kernels/    kernels.h (public launch API + step_info/gemv_seg), kernel_utils.h
                (shared device helpers in namespace si::kd), kv_type.{h,cpp},
                and one .cpp per kernel: rmsnorm, embed, copy_row, gemv,
                qk_norm_rope, attn, conv, gdn, gated_norm, xq, dp4a_gemv,
                dp4a_gemm (+ dp4a_common for the shared split-K workspace),
                vit (vision encoder: GEMM, LayerNorm, GELU/bias, 2D RoPE,
                bidirectional attention)
src/model/      gguf.{h,cpp}, model.{h,cpp} (generic load/upload + bind helpers),
                model_arch.h (architecture registry), qwen35.cpp, model_w8.cpp,
                tokenizer.{h,cpp}
src/mm/         image.{h,cpp} (decode + qwen smart-resize/normalize/patchify),
                vision.{h,cpp} (mmproj loader + host vision encoder),
                multimodal.{h,cpp} (prompt expansion + M-RoPE positions)
src/engine/     engine.{h,cpp} (orchestration), engine_graph.cpp (seg_plan,
                record_forward, build_graphs), engine_kvpool.cpp (dynamic KV
                pool), engine_prefix_cache.cpp (VRAM tier + tier demotion),
                pc_ram.{h,cpp} (host-RAM tier: LRU record store),
                pc_disk.{h,cpp} (disk tier: record format, index, LRU),
                sampler.{h,cpp}
src/server/     chat.{h,cpp} (render_chat + built-in ChatML fallback),
                chat_template.{h,cpp} (minja Jinja wrapper for the GGUF
                tokenizer.chat_template), response_parser.{h,cpp} (streaming
                reasoning_content / content / tool_calls split), chat_util.h,
                scheduler.{h,cpp}, server.{h,cpp} (OpenAI chat/completions +
                /v1/models)
src/main.cpp    CLI
tests/common/   cpu_ref.h (CPU reference forward), stage_test.h (stage harness)
tests/backend/gpu/kernels/  test_gemv.cpp, test_dp4a_gemm.cpp, test_gpu_stages.cpp
                + <kernel>_stage.cpp (one per GPU kernel)
tests/backend/gpu/  test_forward.cpp, test_gpu_vs_ref.cpp,
                test_pc_gpu.cpp (disk spill + promote round-trip),
                test_pc_ram_gpu.cpp (VRAM->RAM->VRAM round-trip)
tests/backend/cpu/  test_cpuref.cpp, test_pc_cpu.cpp (paged attention + disk
                tier on the host backend), test_pc_disk.cpp / test_pc_ram.cpp
                (tier stores)
tests/backend/cpu/kernels/  test_cpu_gemv.cpp (dequant GEMV + RMSNorm vs the
                host reference; run PF_CPU_ISA=scalar|avx2|avx512 to pin a variant)
tests/model/    test_tokenizer.cpp, test_compare.cpp, test_chat_template.cpp
tests/server/   test_response_parser.cpp (reasoning_content/tool_call splitter)
tests/engine/   test_sampler.cpp (logit_bias + logprob reporting)
tests/mm/       test_multimodal.cpp (preprocessing, vision encoder, positions)
third_party/    httplib.h, json.hpp, minja/ (Jinja chat template engine, MIT),
                unicode tables (vendored llama.cpp MIT), stb/stb_image.h
                (public-domain image decode)
```

## How to extend

### Add a model architecture

1. Create `src/model/<arch>.cpp` defining `si::arch::load_<arch>(model & m)` that
   fills `m.hp` and binds the layer tensors (copy the structure of
   `qwen35.cpp`; use `arch::bind_tensor` / `arch::bind_f32`).
2. Declare it in `src/model/model_arch.h` and register it in the `kLoaders[]`
   table in `src/model/model.cpp` (`general.architecture` value → loader).
3. Add the file to `CMakeLists.txt`.  `model::load` dispatches automatically; an
   unknown architecture still throws `unsupported architecture: <name>`.

See [`docs/design/01-model-loading.md`](docs/design/01-model-loading.md) and
[`docs/design/11-qwen35-model.md`](docs/design/11-qwen35-model.md) for the full
loading design.

### Add a kernel

1. Add `src/backend/gpu/kernels/<name>.cpp` that defines the launcher, declares the kernel
   lambdas in the same TU, and includes `"kernels.h"` + `"kernel_utils.h"`.
   The usual preamble is:
   ```cpp
   #include "kernels.h"
   #include "kernel_utils.h"
   namespace si {
   using namespace sycl;
   using namespace si::kd;   // only if you use the shared helpers
   ...
   }
   ```
2. Declare the launch function in `src/backend/gpu/kernels/kernels.h`.
3. Add the `.cpp` to `CMakeLists.txt`.
4. Add a stage test next to the others (see below).

Shared device helpers (dequantization, KV element access, sub-group reductions,
SIn/DP4A expansion, `gemm_ws`) live in `src/backend/gpu/kernels/kernel_utils.h` under
`si::kd`.  Device kernels must be compiled into the TU that uses them — do not
put kernel bodies in headers.

### Add a CPU kernel

1. Add `src/backend/cpu/kernels/<name>.cpp` including `"common.h"` (the shared
   pool, ISA dispatch and dequant/RMSNorm helpers) and define the `cpu_<name>`
   launcher; declare it in `src/backend/cpu/cpu_types.h` and forward it from
   `src/backend/cpu/cpu_backend.cpp`.
2. Add the `.cpp` to `CMakeLists.txt` **and** to the `SI_CPU_SOURCES` list so it
   is compiled with `-fno-sycl` (the CPU kernel directory must stay outside the
   SYCL device pass).
3. Use `par(...)` for parallelism and `isa().dot_f32` / `isa().dot_i8` for the
   ISA-dispatched inner products; structural loops can be plain scalar.

See [`docs/design/03-kernels.md`](docs/design/03-kernels.md) for the kernel
library layout and per-kernel launch geometry.

### Add a kernel stage test

1. Add `tests/backend/gpu/kernels/<kernel>_stage.cpp` defining
   `void stage_<kernel>(si::stage_env & env)` and comparing against `env.get(...)`
   snapshots with `env.cmp(...)` (see `tests/common/stage_test.h`).
2. Declare it in `tests/backend/gpu/kernels/stage_tests.h`.
3. Call it in `tests/backend/gpu/kernels/test_gpu_stages.cpp`.
4. Add the file to the `test_gpu_stages` source list in `CMakeLists.txt`.

### Multimodal (vision) input

The Qwen3.5 vision encoder lives in `src/mm/` and is driven by a separate
`clip` GGUF (`Qwen3.5-0.8B-mmproj-BF16.gguf`), loaded with `--mmproj`:

* `mm_image_preprocess` does the Qwen-VL smart resize (aspect preserving, sides
  aligned to `patch_size*merge`), a Pillow-compatible bicubic resample, and
  `(x-mean)/std` normalization into plane-major CHW f32.
* `vision_model::encode_host` is the reference forward and
  `vision_model::encode_device` the SYCL one (`src/backend/gpu/kernels/vit.cpp`): summed
  16x16 conv patch embedding (GEMM), 2x2 spatial-merge reorder, learned position
  embeddings, 12 LayerNorm + fused-QKV + GELU-MLP blocks with 2D vision RoPE and
  bidirectional attention (online softmax, tiled through SLM), then the
  `qwen3vl_merger` (concat 4 patches - a stride reinterpretation - -> mm.0 ->
  GELU -> mm.2).  Both paths agree to ~2e-4; `test_multimodal` checks that.
* The mmproj stores the linears as BF16 and the norms/biases/patch/position
  tensors as F32; `upload` copies the whole GGUF blob once and `dev_ptr` maps a
  host tensor pointer into it.  `encode_device` runs on the queue it is given,
  which must be **in-order** (the vision stages are dependent kernel launches);
  the engine's queue is.  Device scratch grows to the largest image seen.
* The vision tower's own RoPE is the 4-section `VISION` mode (pairs 0..15 use
  the patch row, 16..31 the column, exponent restarting per section) - do not
  confuse it with the text model's *interleaved* M-RoPE.
* `kMaxImgTokens` (1024) bounds one image's merged tokens and `kMaxImgPatches`
  (4x that) the ViT patch tokens; `mm_image_preprocess` caps `max_pixels` to
  match, and `encode_device` grows its scratch buffers to the largest image seen.
* `mm_build_prompt` computes the merged embeddings on the host;
  `mm_build_prompt_device` runs the tower on the GPU into the engine's
  `d_img_embd`.  Both return the same token/position layout: each `<|image_pad|>`
  expands to `n_out` merged tokens, `img_row` maps them to embedding rows, and
  the 4-section M-RoPE positions are `(base, base+row, base+col)` with the image
  consuming `max(nx, ny)` positions.

CLI: `gen --image FILE` (repeatable).  Server: an OpenAI `image_url` content
part holding a base64 `data:` URL or an `http(s)://` URL (`/v1/chat/completions`),
streaming and not.  Remote fetches are bounded (10 s, 10 MB), need OpenSSL at
build time for HTTPS, and can be disabled with `PF_MM_URL_FETCH=0`.

See [`docs/design/10-multimodal.md`](docs/design/10-multimodal.md) and
[`docs/design/03-kernels.md`](docs/design/03-kernels.md) for the full design.

### OpenAI-compatible API

`GET /v1/models` lists the model (id from the GGUF `general.name`) and
`GET /v1/models/{id}` retrieves it.  `POST /v1/chat/completions` accepts
`messages` (string or `[{type:text|image_url}]` parts), `tools`/`tool_choice`,
assistant messages with `reasoning_content` and `tool_calls`, and `tool` role
messages with `tool_call_id`/`name`; it returns `reasoning_content` (split at
`</think>`) and parsed `tool_calls` (finish_reason `"tool_calls"`), streaming or
not, for `n` choices.  `logit_bias` and `logprobs`/`top_logprobs` are supported
(chat `logprobs.content[]`, completions legacy arrays).  `POST /v1/completions`
accepts string / string[] / token id / token-id-array `prompt` with `n`, `echo`,
`suffix` and `best_of` (non-streaming, scored by token logprob).  `usage`
reports `prompt_tokens_details.cached_tokens` (the prefix cache match count,
`sequence::reused`), DeepSeek-style `prompt_cache_hit_tokens`/
`prompt_cache_miss_tokens`, and `completion_tokens_details.reasoning_tokens`.
Reasoning and tool markup are split by `src/server/response_parser.cpp`
(unit-tested by `test_response_parser`); `docs/design/09-server.md` is the source
of truth, including its §8 field-by-field support matrix (what is implemented,
partial or not).

## Conventions

* C++17, `namespace si`, 4-space indent, Allman braces.  Types/functions are
  `snake_case`, constants `kFoo`, device pointers `d_foo`, host copies `h_foo`.
* Comments explain *why* (often a measurement) — keep them when refactoring.
* The `src/` and `tests/` trees are parallel: one file per kernel/kernel-stage.
* Keep CMake source lists updated; there is no globbing.
* Do not edit `third_party/`; suppress warnings from vendored code via
  `set_source_files_properties(... COMPILE_OPTIONS ...)` in CMake.
* The compile database is `build/compile_commands.json`; `.clangd` adds the SYCL
  include dir and drops `-fsycl` so upstream clangd parses as host.  Lint policy:
  `Diagnostics.UnusedIncludes: Strict` must stay clean — check with
  `clang-tidy -p build -checks='-*,misc-include-cleaner' <file>`.
* Do not commit unless explicitly asked.

## Environment variables

Tuning/diagnostic knobs (unset/`0` = default).  Changing them selects different
kernel variants, so performance numbers must state the env used.

**Model / memory**
`PF_CTX`, `PF_KV_CAP_MB`, `PF_KV_GROW` (pool growth step, default 64 blocks),
`PF_KV_TYPE` (`i4`|`int4`|`i8`|`bf16`|`f16`|`f32`, default `i8`; `--kv-type`
overrides it), `PF_KV_F32`, `PF_KV_BF16`,
`PF_SI4` (re-quantize weights to 4-bit SIn), `PF_META` (fp32 side scales).

**Compute path**
`PF_DEVICE` (`cpu`/`gpu`, default `gpu`), `PF_CPU_ISA`
(`scalar`|`avx2`|`avx512`|`avxvnni`, forces a CPU kernel variant),
`PF_CPU_THREADS` (CPU backend worker threads, default = physical cores, else hardware concurrency),
`PF_DP4A` (int8 GEMM, default on: the GPU packs SIn w8 copies, the CPU's
integer kernel reads the GGUF blocks directly and does not build w8),
`PF_DP4A_DEC` (int8 decode, default on),
`PF_GEMM_DNNL` (oneDNN prefill GEMM, default on), `PF_DNNL_NOWARM`, `PF_DNNL_TIME`.

**Attention**
`PF_ATTN_SPLIT`, `PF_ATTN_SPLIT_KEYS` (default 512), `PF_ATTN_VEC` (default on),
`PF_ATTN_FUSE` (default on), `PF_DEC_SPLIT` (default 64, cap 256),
`PF_DEC_GROUP` (grouped decode attention, default off).

**GEMV / GEMM tuning**
`PF_GEMV_SPLIT`, `GEMV_DEC_VEC`, `GEMV_VEC12`, `GEMV_VEC13`, `GEMV_CFG1/8/16/32`,
`PF_GEMM_ROW`, `PF_GEMM_TILE`, `PF_GEMM_SPLIT`, `PF_GEMM_WG`, `PF_GEMM_SG`,
`PF_GEMM_XSLM`, `PF_GEMM_ARCH`, `PF_MT_R`/`PF_MT_R2`, `PF_MT_TB`, `PF_MT_WG`,
`PF_MT_WG2`, `PF_MT_PF`, `PF_MT_SLM`.

**GDN**
`PF_GDN_COLS` (state columns/warp, default 2), `PF_GDN_WG` (warps/WG, default 8),
`PF_GDN_VEC` (float4 path, default on), `PF_GDN_FUSE` (1/2).

**Prefix cache**
Three LRU tiers; an eviction demotes VRAM -> RAM -> disk -> dropped and a
lookup promotes the other way (a promotion is a move, so a record lives in one
tier).  `PF_PREFIX_CACHE` (default on), `PF_PC_STATES` (default 8),
`PF_PC_DEBUG`, `PF_PC_VRAM_MB` (VRAM budget; derives `PF_PC_STATES` from the
per-node bytes; `PF_PC_MEM_MB` is an alias), `PF_PC_RAM_MB` (host-RAM budget,
default 512, `0` disables the tier), `PF_PC_DIR` (enable the disk tier and its
base directory; records live in a model-fingerprint subdirectory and are read
at startup), `PF_PC_DISK_MB` (disk budget, default 1024, `0` = unbounded).  An explicit
`PF_KV_CAP_MB` also caps the sum of the three tier budgets (shrinking disk,
then RAM, then the VRAM checkpoint count); without it the three configured
tiers are the budget.  The shutdown path (`~engine` / SIGINT-SIGTERM handled in
`serve`) flushes RAM and the resident VRAM nodes into the disk tier.

**Server**
`PF_MM_URL_FETCH` (`0` rejects remote `http(s)://` `image_url` parts; base64
`data:` URLs still work).

**Diagnostics**
`PF_NOGRAPH` (replay kernels directly), `PF_PROF` (with `PF_NOGRAPH`),
`PF_TIME`, `PF_DBG_MID` (stop after embedding/norm/attn/ffn), `PF_DBG_GEMV`,
`PF_DBG_MT`, `PF_DBG_PFB`, `PF_GDN_DBG`, `PF_SRV_TIME`, `PF_CHAT_TMPL_DEBUG`
(log why a GGUF chat template fell back to the built-in renderer), `SCHED_DEBUG`,
`STOP_AFTER_LAYER`, `PF_ABL_NOATTN`, `PF_ABL_NOGDN`.

## Invariants and gotchas

* **Graph capture**: a value read on the host during `record_forward` is frozen
  into the graph.  Per-step state (positions, token ids, `n_real`, active rows)
  must be read from the host-USM `step_info` *inside* the kernel body, never on
  the host.  `engine::record_forward`'s call order must stay in sync with
  `engine::build_plan` (the plan encodes the segment/call layout it replays).
* **oneDNN cannot be recorded** into a SYCL graph; the `PF_GEMM_DNNL` path
  replays mode-2 prefill directly (`prefill_batch`).
* Do not include `sycl/ext/oneapi/dot_product.hpp` from several TUs (its
  functions are not `inline` in this toolchain) — use `src/common/dp4a.h`.
* `kMaxT` = max prefill chunk (32), `kMaxB` = max batched sequences (16),
  `kBlockSize` = KV block size (32), `kMaxSplits`/`kMaxDecSplits` bound the
  attention split buffers (`src/backend/gpu/kernels/kernels.h`).  Buffer sizes are sized
  from these constants; raising them affects device memory.
* The int8 KV layout is `[block][kv head][token][head_dim]` data plus a separate
  fp16 scale plane; both pool and scales are advanced in bytes
  (`engine::kv_layer_stride`, `kv_scale_stride`).
* `ld.bfd` warnings about `libsvml.so`/`libimf.so`/`libintlc.so.5` needed by
  `libdnnl.so` are benign (resolved from the oneAPI runtime path at run time).
* **Multimodal**: an image consumes `max(nx, ny)` positions, not one per token,
  and the text model uses *interleaved* M-RoPE (`rope.dimension_sections`, e.g.
  `[11,11,10,0]`) where the pair index picks the temporal/row/col position
  (`src/backend/gpu/kernels/qk_norm_rope.cpp`).  Image tokens never reach `tok_embd`: the
  embed kernel copies `step_info::img_embd[img_row[t]]` instead.  The
  multimodal path bypasses the prefix cache and uses the single-sequence
  `engine::generate_mm`; `kMaxImgTokens` bounds one image's merged tokens.
  Because an image consumes `max(nx, ny)` positions for `4*nx*ny` tokens, the
  KV slot index (token count) and the RoPE position diverge: decode keeps
  `info->pos` = token count and carries the running RoPE position in
  `step_info::mrope` with `mrope_on` set.

## Quick verification before finishing a change

```bash
cmake --build build -j$(nproc)          # must be warning-free (except the linker notes)
./build/test_gpu_stages                 # all stages OK
./build/test_gpu_vs_ref                 # argmax SAME
./build/test_forward                    # stable last_id
clang-tidy -p build -checks='-*,misc-include-cleaner' <changed files>
```

See [`docs/design/12-build-and-testing.md`](docs/design/12-build-and-testing.md)
for the full build/test matrix and the `docs/` index at
[`docs/README.md`](docs/README.md).
