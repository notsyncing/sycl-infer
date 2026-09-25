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
* [`docs/design/13-audio-video.md`](docs/design/13-audio-video.md) — audio/video
  decode, the AuT audio tower, video/audio M-RoPE, mixed-prompt assembly.
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
* `-DSYCL_INFER_AOT=ON` AOT-compiles the device image for `adl-p`
  (`-DSYCL_INFER_AOT_DEVICE=` overrides the target).  The device lowering
  (`llvm-foreach -> ocloc`/IGC) only runs at the **final link**, never at `-c`,
  so `make -j` does not parallelise it; by default it is one serial `ocloc`
  process over every device image and takes minutes per kernel.  See the AOT
  notes below.
* **Debug builds force `-O2` for device code** (see CMakeLists comments).  At
  `-O0` the IGC device image of `src/backend/gpu/kernels/*` becomes huge and its
  translation takes minutes (looks like a hang) — do not remove that.
* Source files are listed explicitly in `CMakeLists.txt` (no globbing).  Add new
  `.cpp` files there.

### AOT builds (`SYCL_INFER_AOT`)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSYCL_INFER_AOT=ON -DSYCL_INFER_AOT_JOBS=6
cmake --build build --target sycl-infer -j$(nproc)   # only the binary, not every test
```

* AOT lowering is the expensive step: `icpx` emits one device image per kernel,
  then `llvm-foreach` runs `ocloc` (IGC) on each.  It happens at the **final
  link of each executable**, so the ~20 test binaries each re-lower the device
  code they pull in.  Build a single target (`--target sycl-infer`) unless the
  tests really need AOT.
* The post-link device step is **serial by default** (a single `ocloc`
  process); that is not a `make -j` problem.  `CMakeLists.txt` now passes
  `-fsycl-max-parallel-link-jobs=N`, so `-DSYCL_INFER_AOT_JOBS=N` runs N IGC
  translations at once (default `nproc`).  RAM scales with N, lower it on a
  memory-constrained host.
* `CMakeLists.txt` also passes `-fsycl-device-code-split=per_kernel`: the
  default (`auto`) merges a whole module into one image, which serialises the
  lowering; per-kernel images can run in parallel and be skipped individually.
* Use `-DCMAKE_BUILD_TYPE=Release`.  A Debug build's `-g` adds device debug
  info that makes each image several times larger and slower to lower (measured
  ~2.3x and 4.4x the binary size on one GEMV TU); the forced `-O2` does not
  remove it.
* The `ocloc` compiler cache is **not usable** here: `-allow_caching` /
  `-cache_dir` (and `NEO_CACHE_DIR`) create the directory but write no entries
  and give no speedup on the distro `intel-ocloc` 26.27.1.  The working
  alternative is the runtime JIT cache (`SYCL_CACHE_PERSISTENT=1`,
  `SYCL_CACHE_DIR=...`, entries under `~/.cache/neo_compiler_cache/*.l0_cache`),
  which caches JIT-compiled kernels across runs and avoids the AOT build
  entirely; `ccache` does not help because device lowering is a link step.

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
--port --host --device --cpu-threads --layer-map --mmproj --audio-mmproj`, and
for `gen`:
`--prompt --max-tokens --temp --top-p --top-k --raw --thinking --image --video
--audio --max-video-frames --max-video-side`).  `--device cpu|gpu|auto`
selects the compute backend; `--cpu-threads N` (or `PF_CPU_THREADS`) sets the
host worker count;
`--layer-map 0-11:gpu,12-23:cpu` places closed layer ranges on devices (each
range must cover the layer list without gaps).

`--kv-type K:V` sizes the K and V caches independently (only the scale-carrying
i4/i8 may be mixed).  The attention error is dominated by V, not K: on the 27B
`--kv-type i4:i8` (4-bit keys, 8-bit values) measures mean|diff| 0.061 against
the fp32 CPU reference, versus 0.035 for i8 and 0.183 for i4, at 24 KB/token
(i8 is 32, i4 is 16) - so it fits a 262144-token context on 2x A770 with
near-i8 accuracy.  `i8:i4` is the mirror image and is *worse* than i4/i4
(mean|diff| 1.49), confirming V is the sensitive side.  See
[`reports/turboquant_and_perf.md`](reports/turboquant_and_perf.md).

## Testing

The tests default to `/path/to/Qwen3.5-0.8B-Q4_K_M.gguf` and require the
GPU + that model.  Strict kernel/end-to-end tests set `PF_DP4A=0` (fp32 path).

```bash
./build/test_tokenizer     # tokenizer round-trips (CPU only, no GPU work)
./build/test_chat_template # GGUF chat template vs reference Jinja2 output (CPU only)
./build/test_multimodal    # image/video prompt, audio decode+mels, vision + audio
                              # encoders (host + device), positions (CPU+GPU)
./build/test_compare       # CPU reference vs llama.cpp dumps (CPU only)
./build/test_cpuref        # CPU reference head (CPU only)
./build/test_cpu_gdn       # cpu_gdn with an *asymmetric* head count (CPU only)
./build/test_cpu_gemv      # host GEMV/RMSNorm vs dequant reference (CPU only)
./build/test_gpu_stages    # every kernel vs the CPU reference (GPU)
./build/test_gemv          # GEMV vs CPU dequant reference (GPU)
./build/test_dp4a          # SIn repack + DP4A GEMM vs CPU reference (GPU)
./build/test_iq_dequant    # IQ*/Q3_K GPU dequant vs host reference (GPU, 27B)
./build/test_quant_audit   # int8(oneDNN) GEMM loss per ggml type (GPU)
./build/test_w4*           # native-width u4 packing / GEMM / vs int8 (CPU+GPU)
./build/test_gpu_vs_ref    # end-to-end logits vs CPU reference (GPU)
./build/test_forward       # end-to-end logits / top-k (GPU)
./build/test_decode_vs_prefill  # single-token decode == re-prefill (GPU)
```

Always run at least `test_gpu_stages`, `test_gpu_vs_ref` and `test_forward`
after touching kernels/engine, and confirm they still pass (they print
`all stages OK`, `argmax ... SAME`, a stable `last_id`).  **Any change that can
affect the single-token path must also run `test_decode_vs_prefill`**: the three
tests above are prefill-only and are blind to a decode-only bug (that is how the
fixed head-segment binding bug stayed invisible).

## Repository layout

```
src/common/     quant.h (ggml block formats + host dequant), w8.{h,cpp} (SIn
                int8 weight format/repack), w4.{h,cpp} (native-width u4 packing
                for Q4_K: u4 plane + separate f16 scale/off planes; and the
                codebook 4-bit store for IQ4_XS/IQ4_NL: 4-bit index plane in the
                interleaved nibble order + per-32 f16 scale, `cb4t`/`cb4_pack`),
                dp4a.h (portable dp4a helper),
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
                w4_gemv (u4/int8/codebook decode GEMV: g-major SLM-staged 4-bit
                GEMV, LUT-expanding codebook GEMV and its prefill expansion, plus
                the opt-in batched u4 GEMM `w4_gemm_launch`), mtp (mtp_concat /
                mtp_capture for the NextN draft head),
                 vit (vision encoder: GEMM, LayerNorm, GELU/bias, 2D RoPE,
                 bidirectional attention), at (audio tower helpers: at_conv1d,
                 at_rope1d), attn_xmx (oneDNN int8 XMX prefill attention, see
                 PF_ATTN_XMX below)
src/model/      gguf.{h,cpp}, model.{h,cpp} (generic load/upload + bind helpers),
                model_arch.h (architecture registry), qwen35.cpp, model_w8.cpp,
                tokenizer.{h,cpp}
src/mm/         image.{h,cpp} (decode + qwen smart-resize/normalize/patchify),
                vision.{h,cpp} (mmproj loader + host vision encoder),
                video.{h,cpp} (video decode + uniform frame sampling),
                audio.{h,cpp} (audio decode: WAV native / ffmpeg fallback +
                log-mel), audio_model.{h,cpp} (AuT audio tower: host + device),
                multimodal.{h,cpp} (prompt expansion + M-RoPE positions)
src/engine/     engine.{h,cpp} (orchestration), engine_mtp.cpp (MTP plan/forward/
                verify/rollback + the speculative loop), engine_graph.cpp (seg_plan,
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
                + <kernel>_stage.cpp (one per GPU kernel), plus the weight-path
                audits: test_w4_gemm.cpp, test_w4_vs_i8.cpp, test_gemv_stride.cpp,
                test_iq_dequant.cpp, test_quant_audit.cpp
tests/backend/gpu/  test_forward.cpp, test_gpu_vs_ref.cpp,
                test_pc_gpu.cpp (disk spill + promote round-trip),
                test_pc_ram_gpu.cpp (VRAM->RAM->VRAM round-trip)
tests/backend/cpu/  test_cpuref.cpp, test_pc_cpu.cpp (paged attention + disk
                tier on the host backend), test_pc_disk.cpp / test_pc_ram.cpp
                (tier stores)
tests/backend/cpu/kernels/  test_cpu_gemv.cpp (dequant GEMV + RMSNorm vs the
                host reference; run PF_CPU_ISA=scalar|avx2|avx512 to pin a variant),
                test_cpu_gdn.cpp (cpu_gdn with an asymmetric head count)
tests/model/    test_tokenizer.cpp, test_compare.cpp, test_chat_template.cpp,
                test_w4.cpp (u4 packing round-trip)
tests/server/   test_response_parser.cpp (reasoning_content/tool_call splitter)
tests/engine/   test_sampler.cpp (logit_bias + logprob reporting),
                test_decode_vs_prefill.cpp (single-token decode vs re-prefill),
                test_w4_vs_cpuref.cpp / test_w4_topk.cpp (u4 logits vs the fp32
                reference; TEST_LAYER_MAP runs the 27B split across devices),
                test_27b_prefill.cpp (27B prefill bring-up probe)
tests/mm/       test_multimodal.cpp (preprocessing, vision encoder, video/audio
                prompts, audio kernels, positions)
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

See [`docs/design/10-multimodal.md`](docs/design/10-multimodal.md) and
[`docs/design/03-kernels.md`](docs/design/03-kernels.md) for the full design.

### Audio + video input

On top of the vision tower there are two more modalities, mixed freely in one
request (see [`docs/design/13-audio-video.md`](docs/design/13-audio-video.md)):

* **Video** (`src/mm/video.{h,cpp}`, `--video` / `video_url`): decoded into
  uniformly sampled frames (`mm_video_subsample`, at most `max_frames`=16,
  frame side capped by `max_side`=768, ffmpeg fallback for MP4/etc.), each frame
  preprocessed like an image; a `T`-frame video contributes `T·nx·ny` tokens and
  consumes `max(out_w,out_h)` positions (M-RoPE (frame, row, col)).
* **Audio** (`src/mm/audio.{h,cpp}` + `audio_model.{h,cpp}`, `--audio` /
  `input_audio` / `audio_url`): WAV decoded natively, other containers through
  the ffmpeg CLI (in-memory server data spills to a temp file), then a
  log-mel spectrogram (16 kHz, 25 ms/10 ms, 128 bins).  The **audio mmproj must
  be loaded separately** (`--audio-mmproj`, a second GGUF with `audio.` metadata
  and `a.*` tensors); without it every audio request fails with a clear error.
  The AuT-style tower is conv1->GELU->conv2(GELU, /2)->learned positions->N
  vision-style blocks->post_ln->optional projection, emitting one embedding per
  remaining frame (= one token, one M-RoPE position each); `out_width` must equal
  text `n_embd`.
* Mixed requests are assembled by `mm_build_prompt_mixed_device`
  (`multimodal.cpp`): `mm_media_ref order` lists blocks in placeholder order,
  each media is encoded into its slice of `d_img_embd`, total rows capped at
  `kMaxImgTokens`.  The vision+audio device kernels reuse `vit_*` plus
  `at_conv1d`/`at_rope1d` (`src/backend/gpu/kernels/at.cpp`).
* Server API: `image_url` / `video_url` content parts (base64 `data:` or
  http(s), gated by `PF_MM_URL_FETCH`) and `input_audio`/`audio_url`;
  `chat_part` (`src/server/chat.h`) carries a `kind` tag and the renderers emit
  `<|image_pad|>` / `<|video_pad|>` / `<|audio_pad|>` placeholders per kind.
* Like images, image/video/audio embeddings never reach `tok_embd` and the
  multimodal path bypasses the prefix cache.

CLI: `gen --image FILE` / `--video FILE` / `--audio FILE` (each repeatable, all
mixable), `--audio-mmproj`, `--max-video-frames`, `--max-video-side`.  Server:
OpenAI-style `image_url`/`video_url`/`input_audio`/`audio_url` parts in
`/v1/chat/completions`, streaming and not.

### Multi-token prediction (MTP / NextN) speculative decoding

`--mtp N` drafts up to `N` tokens with the model's own NextN head and verifies
them against the target in one batched forward, emitting exactly the tokens a
plain greedy decode would.  Requires a GGUF that bundles the head: the 27B
reference model has `qwen35.nextn_predict_layers == 1` with `blk.<n_layer>.nextn.*`
(`eh_proj`, `enorm`, `hnorm`, `attn_norm`, q/k/v/wo/ffn, optional
`shared_head_norm`/`shared_head_head`); the 0.8B has none, so `--mtp` there is a
no-op.  The MTP layer is a full-attention Qwen3.5 block, so it owns one extra
attention KV slice (`attn_layers() - 1`) in the same paged pool and is counted in
`--kv-cap-mb` and by all three prefix-cache tiers.

Semantics, all required for the emitted stream to stay bit-equal to a plain
greedy decode:

* the head consumes the trunk hidden **before** `output_norm` (`t_h_pre_norm`)
  and applies its own `enorm`/`hnorm`; a draft row pairs `emb(t_p)` with
  `h_{p-1}`, matching llama.cpp's right-shift;
* the verify is one batched forward of `[last_committed, draft0..draft_{k-1}]`
  at consecutive positions, and acceptance is `target_argmax(row i) == draft[i]`
  plus a bonus token from row `j`;
* the verify is **dry** (`step_info::mtp_dry`): it computes the forward but does
  not write the GDN/conv state, and it snapshots the per-token state into
  `d_mtp_hist_` so the commit rewinds the recurrent state to the last accepted
  row (llama.cpp's `n_rs_seq`); the conv window is rebuilt from the raw taps
  saved in `d_mtp_qsave_`;
* the prefix cache is supported: `generate_mtp` calls `pc_admit` for the prompt
  (which restores the matched chain's KV *and* recurrent state) and `pc_commit`
  after the prefill.

The path is opt-in (`--mtp 0`/absent is the plain decode, byte-identical output)
and currently gated behind `PF_MTP_EXPERIMENTAL=1` because it is **not yet a
speedup**: at k=6 it measures ~67-73 ms/token against the plain decode's
50.6 ms/token.  The gap is entirely the verify's batch GEMM - the engine's
small-M batched path runs oneDNN's `jit:gemm:any` at ~63 GB/s/card while the
decode's own u4/cb4/k5 GEMVs stream the same weights at ~220-250 GB/s/card, i.e.
one verify costs ~4 decode passes while the drafts only buy ~2.6 accepted
tokens/cycle.  `--mtp N` also disables the q5/cb4 native stores for the
speculative path (they would expand to int8 on every verify pass).

Reaching >1x needs a batched native-layout GEMM at the decode's per-byte rate.
Measured so far on the 27B: the weight stream itself is nearly free (removing
the per-`(row,group)` activation/scale loads from the batched kernel reaches
219 GB/s = the GEMV's own rate); the hand-written batched u4 kernels in
`w4_gemv.cpp` are bit-exact against the M=1 GEMV but reach only 43 GB/s at M=5
(accumulator spilling was the first 3x, fixed by expanding the row bodies with
`if constexpr`; every column-tiled variant then collapses ~10x), and forcing
oneDNN onto a blocked int8 weight layout makes it pick `ocl:ref:any` rather than
an XMX kernel - so neither the XMX route nor the current hand-written kernels
beat `jit:gemm:any` on this stack yet.

Diagnostics (all env-gated, `0`/unset = off unless noted): `PF_MTP` (draft
length, `--mtp` overrides), `PF_MTP_DEV`, `PF_MTP_TIME` (per-phase cycle ms),
`PF_MTP_DEBUG`/`PF_MTP_DUMP`, `PF_MTP_VERIFY_PAD`/`PF_MTP_VERIFY_M32` (pad the
verify's GEMM M), `PF_MTP_VERIFYN`, `PF_MTP_DECCHK` (verify row 0 vs a plain
decode of the same token), `PF_MTP_LSTAT`, `PF_MTP_DECODE_H`, `PF_MTP_NOACCEPT`,
`PF_MTP_NOMTPFWD`, `PF_MTP_NORB`, `PF_MTP_DBG_RB`.  Batch-GEMM knobs:
`PF_W4_GEMM_MAXM` (default 1 = the oneDNN matmul; 2..8 routes the u4 tensors to
`w4_gemm_launch`), `PF_W4_GEMM_U` (accumulator sets), `PF_W4_GEMM_TN`
(0 = row-expanded, 2/3/4 = the tiled experiments), `PF_DNNL_BLOCKED`.

### OpenAI-compatible API

`GET /v1/models` lists the model (id from the GGUF `general.name`) and
`GET /v1/models/{id}` retrieves it.  `POST /v1/chat/completions` accepts
`messages` (string or
`[{type:text|image_url|video_url|input_audio|audio_url}]` parts),
`tools`/`tool_choice`,
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
  `set_source_files_properties(... COMPILE_OPTIONS ...)` in CMake.  One
  documented exception: `third_party/minja/minja.hpp` carries a 2-line local
  patch adding Jinja's `is undefined` test (`is defined` was implemented,
  `undefined` was not).  Our copy is byte-identical to upstream HEAD, which has
  the same gap, and the Qwen3.8-27B chat template opens with
  `{%- if enable_thinking is undefined or ... %}` - without the test the whole
  template throws and the built-in ChatML fallback produces a prompt the model
  was not trained on (chat answers degrade to a single token).
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
overrides it; `K:V` sizes K and V independently, e.g. `i4:i8`), `PF_KV_F32`,
`PF_KV_BF16`,
`PF_SI4` (re-quantize weights to 4-bit SIn), `PF_META` (fp32 side scales).

**Compute path**
`PF_DEVICE` (`cpu`/`gpu`, default `gpu`), `PF_CPU_ISA`
(`scalar`|`avx2`|`avx512`|`avxvnni`, forces a CPU kernel variant),
`PF_CPU_THREADS` (CPU backend worker threads, default = physical cores, else hardware concurrency),
`PF_DP4A` (int8 GEMM, default on: the GPU packs SIn w8 copies, the CPU's
integer kernel reads the GGUF blocks directly and does not build w8),
`PF_DP4A_DEC` (int8 decode, default on), `PF_W4` (native-width u4 weights for
Q4_K, default on; `PF_W4=0` restores pure int8 - the per-32-group int8 weight
scales always apply),
`PF_GEMM_DNNL` (oneDNN prefill GEMM, default on), `PF_DNNL_NOWARM`, `PF_DNNL_TIME`.

**Attention**
`PF_ATTN_SPLIT`, `PF_ATTN_SPLIT_KEYS` (default 512), `PF_ATTN_VEC` (default on),
`PF_ATTN_FUSE` (default on), `PF_DEC_SPLIT` (default 64, cap 256),
`PF_DEC_GROUP` (grouped decode attention, default off), `PF_ATTN_XMX` (oneDNN
int8 XMX prefill attention, **default on**; `0` restores the classic kernel),
`PF_ATTN_XMX_MIN` (minimum key count for XMX, default 2048 - below it the
classic kernel is faster), `PF_ATTN_WAIT` (force a per-attention-matmul oneDNN
stream wait; default off, the in-order queue already orders them).

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
`PF_MM_URL_FETCH` (`0` rejects remote `http(s)://` `image_url`/`video_url`/`audio_url`
parts; base64 `data:` URLs still work), `PF_AV_FFMPEG` (audio/video decode CLI
path, default `ffmpeg`).

**Diagnostics**
`PF_NOGRAPH` (replay kernels directly), `PF_PROF` (with `PF_NOGRAPH`),
`PF_TIME`, `PF_DBG_MID` (stop after embedding/norm/attn/ffn), `PF_DBG_GEMV`,
`PF_DBG_MT`, `PF_DBG_PFB`, `PF_GDN_DBG`, `PF_SRV_TIME`,
`PF_CHAT_TMPL_DEBUG` (log why a GGUF chat template fell back to the built-in
renderer), `SCHED_DEBUG`, `STOP_AFTER_LAYER`, `PF_ABL_NOATTN`, `PF_ABL_NOGDN`,
`PF_DUMP_LAYERS` (per-layer hidden-state fingerprints; `layerlast` = the last
real token slot; `PF_DUMP_RAW=<prefix>` writes the whole activation vector so a
decode run can be diffed element-wise against a prefill),
`PF_DUMP_LOGITS`/`PF_DUMP_DEC_LOGITS=<path>` (sampler logits; the latter per
step), `PF_DUMP_PROMPT` (the exact ids - and, for a chat prompt, the rendered
text - the model is conditioned on), `PF_DUMP_GEN` (the decode loop's sampled id
and stop decisions), `PF_ROWACT` (restore the now-unused per-row activation
quantizer for A/B; it is dead because oneDNN reads the per-32-group form, and
cost ~13 ms/token in a 27B multi-device decode), `PF_DUMP_KV=<dir>` (append the
live tokens' post-norm+RoPE f32 K and V of every full-attention layer to
`<dir>/{k,v}_LL.bin`, for offline KV-quantization studies; skipped while a
command graph is being recorded, so it only fires on the direct prefill paths), `PF_W4_RB` (decode GEMV rows
per workgroup, default 16), `PF_CB4` (`0` keeps IQ4_XS/IQ4_NL on the int8
conversion instead of the native codebook store), `PF_K5` (`0` keeps Q5_K on
the int8 conversion instead of the native 5-bit store; the prefill cost below is
what it buys), `PF_K5_NOCORR` (drop the prefill offset correction - diagnostic
bisection for the k5 prefill cost), `PF_MD_GRAPH_DEV` (record the multi-device decode
command graph only for device N; `99` = none, for A/B against the direct
replay), `PF_PROF_ALL` (with `PF_PROF`: dump every call group's ms/step instead
of the top 8), `PF_PFB_MAX_M` (optional cap on the multi-device mode-2 prefill
batch; `0` = no cap, the default — the cap is no longer needed, see the
fused-GDN mode-2 fix below).

**Weight representation**
`PF_W4` (native u4 for Q4_K, default on), `PF_CB4` (store IQ4_XS/IQ4_NL as native 4-bit codebook indices + a per-32 f16
scale, 0.5625 B/weight and lossless, decoded by a LUT-expanding GEMV; prefill
expands each tensor to int8 in a reused scratch - see
[`reports/tg128_20tps_evaluation.md`](reports/tg128_20tps_evaluation.md)),
`PF_K5` (store Q5_K as the native 5-bit grid: 4-bit nibble plane + 1-bit plane
+ per-(g,n) f16 step/offset, 0.75 B/weight and lossless, recombined with one OR
in the decode GEMV; prefill expands q5 to int8 in the shared scratch and runs
the grouped-scale int8 primitive with the u4 offset-correction epilogue - i.e.
it costs ~1.6 B/weight of *serial* prefill traffic per pass, measured -15..-20%
pp512 for +6% tg128 and -2.1 GB/card; see
[`reports/tg128_20tps_evaluation.md`](reports/tg128_20tps_evaluation.md)),
`PF_W4_ALL` (`1` re-quantizes every
other type onto the same 4-bit grid: 16.0 vs 24.5 GB read per 27B decode token,
measured tg128 12.6 -> 16.2 t/s but -20% prefill and ~8x weight error vs fp32,
see [`reports/tg128_20tps_evaluation.md`](reports/tg128_20tps_evaluation.md)),
`PF_SI4` (SIn 4-bit), `PF_META`.

## Invariants and gotchas

* **The GGUF mmap is paged out of the host after the weights reach a device.**
  `model::page_out_host` does `madvise(MADV_DONTNEED)` + `posix_fadvise` on the
  tensor's file range; `setup_md_dnnl`'s `add` releases each tensor right after
  its conversion and `upload_device_weights` right after its raw copy (the final
  `engine::release_host_weight_pages` sweeps the rest).  The mapping stays valid
  (device pointers use `map_base` only for arithmetic), but **any new host read
  of a weight after `engine` construction re-faults from disk** - correctness is
  preserved, throughput is not.  CPU partitions keep their pages (their kernels
  read the mmap directly), so a hybrid map's CPU tensors must stay in the
  `keep` set of `release_host_weight_pages` - including the globals
  (`tok_embd` / `output` / `output_norm`) when backend 0 is a CPU partition (an
  all-`cpu` map, where they are never uploaded).
* **Graph capture**: a value read on the host during `record_forward` is frozen
  into the graph.  Per-step state (positions, token ids, `n_real`, active rows)
  must be read from the host-USM `step_info` *inside* the kernel body, never on
  the host.  `engine::record_forward`'s call order must stay in sync with
  `engine::build_plan` (the plan encodes the segment/call layout it replays).
* **A plan is a device-pointer snapshot**.  `build_plan` writes the *then
  current* `d_x*` member pointers into each `gemv_seg`, and multi-device
  `bind_acts(dev)` only rebinds those members (it never rewrites a built plan).
  So anything pinned to the primary device (today: the LM head) must be built
  *after* `bind_acts(0)`, otherwise it captures the last layer's device buffers
  and the batch-1 decode head reads a buffer no kernel wrote for that step -
  while the prefill (which goes through `run_head()`) stays correct.
* **The multi-device LM head must reach the primary device's oneDNN table, and
  `cur_dev` must be reset to 0 before it.**  `gemv_at` reads `dnnl_for(cur_dev)`
  and `cur_dev` is the last layer's device after the layer loop, so it is reset
  to 0 at the final `handoff_x`, and the head's plan call gets an `xq` entry
  (without one `gemv_at` cannot take its `dnnl_call` branch).  Miss any of these
  and the head silently falls through to the fp32 dequant GEMV: 12.3 vs 3.4
  ms/token.  The head's oneDNN key is the host pointer for an **untied** head
  (`output.weight` exists; `setup_md_dnnl` converts it before the upload, so the
  raw copy is skipped and `wkey` returns the host key) but the **uploaded device
  pointer** for a **tied** head (0.8B: `m.output == m.tok_embd`), which must keep
  its raw GGUF rows for the embed kernel and is therefore converted after the
  upload - see `setup_multi_device`.
* **`handoff_x` copies only the live rows** (`nrows * nreal`, the activation
  layout is `[token][n_embd]`).  The old full `kMaxB*kMaxT` copy moved 10.5 MB
  per handoff - twice per token - for a single-token decode.
* **A partial (per-device) `record_forward` must start its call cursor at the
  phase's first layer.**  `ci` indexes the plan's *global* `call_tb`/`call_xq`/
  `call_group_*` arrays, so `build_md_dec_graphs` uses `seg_plan::layer_c0`
  (layer -> first call, plus a final entry for the head) to seed it.  Restarting
  at 0 makes the later partition read the first layer's call metadata (activation
  pointer, K) while executing its own segments: `dnnl_call` goes false, the group
  falls back to the fp32 `gemv_group`, and the layers silently write nothing
  (one repeated token, ~6x slower from the NaN-heavy hidden state).
* **Multi-device decode is graphed per partition** (`build_md_dec_graphs`, one
  graph per contiguous device run, replayed with the host handoff between them;
  `PF_MD_GRAPH_DEV=N` limits it to device N for A/B).  It is worth ~1-2 ms/token
  only, so per-kernel dispatch is *not* the layer GEMV's main inefficiency.
* **GDN head pairing is modulo, attention GQA is blocked** - do not unify them.
  `gdn.cpp` pairs value head `h` with q/k head `h % n_group` (the reference
  tiles q/k with `ggml_repeat_4d`, whose repetition is modulo/interleaved),
  while `attn.cpp` expands GQA with `kvh = h*n_head_kv/n_head` (HF
  `repeat_kv`).  They agree only when the head counts are equal, which is
  exactly the 0.8B reference model (`n_group == dt_rank == 16`), so a mistake
  here is invisible on 0.8B and fatal on the 27B (16 vs 48).
* **`qkv_dim()` is not `3*d_inner`** and `m.output` is not always
  `m.tok_embd`: both assumptions hold on the 0.8B only.  Use `hp.qkv_dim()`
  for the GDN qkv/conv width and `m.output` for the LM head.
* **oneDNN cannot be recorded** into a SYCL graph; the `PF_GEMM_DNNL` path
  replays mode-2 prefill directly (`prefill_batch`).
* **XMX prefill attention (`attn_xmx.cpp`, `PF_ATTN_XMX`, default on above 2048
  keys)**: QK^T and PV run as plain oneDNN int8 matmuls over the paged KV
  gathered into a contiguous scratch.  Three things are load-bearing:
  (1) a plain int8 matmul sums over k, so a per-key scale cannot be applied
  after it - K and V each carry ONE block-wide scale, folded in at gather time;
  (2) the query tile stacks every HPG query head sharing a kv head, so the
  matmul width is `HPG * tokens` and the KV is read once per kv head, not once
  per query head; (3) the matmuls always run at the FIXED width `kBlk = 2048`
  (zero-padded), because oneDNN primitive creation is ~15 ms per new shape and
  a per-chunk `N` made prims never reusable - that alone was a 12x regression.
  The attention scratch is per-queue (`xmx_get` keys on the queue address): a
  function-local static was shared across the two `--layer-map` devices and the
  cross-device USM access cost another ~12x.  The matmuls use a dedicated
  oneDNN stream (`st_a`) on the same in-order queue so waiting on attention
  does not drain the dense GEMMs queued on the main stream (a queue-wide wait
  serialised the whole prefill).  The softmax is one work-group per query row
  with a single vectorized `exp` pass.  The per-row index arrays (`orow` /
  `oh` / `olim`) are built by a device kernel from `info`, **not** by host
  `q.memcpy`: three host->device copies per kv head kept the host pinned to the
  GPU (and raced with the reused host vector) and cost ~2x.  Measured marginals
  on the deepest full 512-token mode-2 chunk (27B / 2x A770, i8 KV,
  `PF_PREFIX_CACHE=0`, `--layer-map`): pp512@16k 296 -> 911 t/s, pp512@64k 93 ->
  540 t/s (both above the OpenVINO targets of 640 / 362).  Below the key
  threshold the classic kernel still wins (4k: 718 vs 649 t/s), hence the gate.
  Scope: i8 KV, head_dim 256, `n_real > 1`; other cases fall back.
* **`st_a` is an XMX-only stream.**  The dense `PF_GEMM_DNNL` GEMMs still use
  `p->st`; `attn_qk`/`attn_pv` submit to `p->st_a` and wait only that stream.
  Do not move the dense path onto `st_a`.
* **A partial mode-2 batch requires the oneDNN weight path.**
  `batched_prefill_fit` only allows a last row with `n_real_row < kMaxT` when
  `use_dnnl` or a multi-device GPU partition has oneDNN (`dnnl_any_dev()`); the
  md_int8 fallback's dp4a GEMM is only correct for a full `kMaxT` row, so it
  uses the recorded multiples-of-kMaxT variants plus the chunked tail.  Letting
  md_int8 take a partial batch silently corrupts the hidden state (a 2-GPU
  md_int8 decode-vs-prefill mismatch).  Likewise an **all-`cpu` `--layer-map`**
  must select the CPU queue (`resolve_device`), not the default GPU one.
* **The fully-fused mode-2 GDN call used to process only the first chunk row**
  (fixed).  `record_forward` mode 2 (chunk-batched prefill) runs the GDN
  recurrence once over the whole batch (`PF_GDN_FUSE=2`) with `n_rows=1` and
  `tpb_arg = nreal_arg = <total tokens>`, but the GPU `gdn_kernel` /
  `gdn_f4_kernel` ignored `nreal_arg` and used `row_nr(info, rr)`, so only row 0
  (`kMaxT` tokens) got the recurrence and every later chunk row read a stale
  state.  The CPU `cpu_gdn` already honoured `nreal_arg`, so a GPU+CPU hybrid map
  behaved differently.  It hit *single-device* mode-2 as well; it was invisible
  only because the multi-device cap (`PF_PFB_MAX_M`, old default `kMaxT`) forced
  the scheduler onto the mode-1 chunked path, and because `dev/cmp_pfb` compares
  hidden-state cosines (0.998) rather than the first token.  Symptom: a prompt of
  >=64 tokens produced wrong logits (63 was correct — the scheduler used mode 1
  for `rem < 2*kMaxT`).  Fixed in `gdn.cpp` (both kernels now use
  `nreal_arg > 0 ? nreal_arg : row_nr(info, rr)`), the cap default is removed,
  and `PF_GDN_FUSE=1` is a working fallback.  Measured on 27B / 2x A770: pp512
  91 -> 566 t/s at depth 0 and 63 -> 329 t/s at 4k; 2+2 answers correctly at
  614/1396 tokens.  The `PF_PFB_MAX_M=32` mode-1 fallback remains available for
  A/B.
* **The scheduler now lets mode-2 prefill the tail** (`>= kMaxT` instead of
  `>= 2*kMaxT`).  A mode-1 chunk costs ~0.26 ms of handoff/sync *per token
  slot* on 2x A770 (a 41-token tail was ~0.34 s), so a 553-token prompt used to
  pay a full extra mode-2-sized forward.  Correctness is unchanged because the
  fused-GDN bug above is fixed; measured cold pp512 of a 553-token prompt went
  327 -> 399 t/s.  Mode-2 still costs ~0.26 s of *fixed* weight-stream per
  forward, so a 512-token batch is the efficient unit.
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
  embed kernel copies `step_info::img_embd[img_row[t]]` instead.  Video blocks add
  a temporal axis (M-RoPE `(frame,row,col)`, `n_pos = max(out_w,out_h)`) and audio
  blocks are a pure time stream (`(pos,0,0)`, `n_pos = n_out`); both reuse the same
  `img_embd` rows and `img_row` mapping.  The
  multimodal path bypasses the prefix cache and uses the single-sequence
  `engine::generate_mm`; `kMaxImgTokens` bounds one request's merged tokens.
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
./build/test_decode_vs_prefill          # decode == re-prefill (any single-token change)
clang-tidy -p build -checks='-*,misc-include-cleaner' <changed files>
```

For the 27B (needs the layer split - it does not fit on one card), also run:

```bash
M=/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
TEST_LAYER_MAP=0-31:gpu.0,32-63:gpu.1 ./build/test_w4_vs_cpuref      # argmax SAME
TEST_LAYER_MAP=0-31:gpu.0,32-63:gpu.1 ./build/test_decode_vs_prefill "$M"  # OK
```

The 0.8B is the **tied-embedding** (`output.weight` absent) reference, and the
only model that exercises the tied multi-device LM head; check it with a split
(and a `cpu` partition for the hybrid path):

```bash
TEST_LAYER_MAP=0-11:gpu.0,12-23:gpu.1 ./build/test_decode_vs_prefill  # OK
TEST_LAYER_MAP=0-11:gpu.0,12-23:cpu   ./build/test_decode_vs_prefill  # OK
```

`test_decode_vs_prefill` defaults to the 0.8B, so it needs the model path
explicitly; `test_w4_vs_cpuref` defaults to the 27B.  The multi-device decode
cost breakdown (and the 20 tps feasibility analysis) is in
[`reports/tg128_20tps_evaluation.md`](reports/tg128_20tps_evaluation.md), with
the `STOP_AFTER_LAYER` sweep and the decode-GEMV microbenchmarks in `dev/`.

If a build seems to ignore your edit, remember `rsync -a` preserves source
mtimes: `find src tests -name '*.cpp' -o -name '*.h' | xargs touch` first.

See [`docs/design/12-build-and-testing.md`](docs/design/12-build-and-testing.md)
for the full build/test matrix and the `docs/` index at
[`docs/README.md`](docs/README.md).
