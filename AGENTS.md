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

**`reports/` is not part of this repository.**  It is in `.gitignore` and stays
that way: nothing under it is ever committed, so **no document here may link to
it** — not this file, not `README.md`, not `docs/`.  A `reports/` reference in a
committed doc is a dead link for everyone who clones the tree.  Every number worth
keeping is quoted inline below (with the measurement it came from), and a
measurement that only exists in a write-up has to be summarised here before it can
be relied on.

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

* **The AOT flags and the library you link must agree.**  Linking the AOT
  `libsycl_infer_core.a` (built with `-DSYCL_INFER_AOT=ON`, device images baked for
  `acm-g10`) into a hand-built **JIT** binary compiled without them crashes inside
  the SYCL runtime, not at link: `SIGSEGV` in
  `DeviceKernelInfo::setCompileTimeInfoIfNeeded`, reached from
  `dnnl_gemm::warmup` → `act_quant_launch` while JIT-compiling a kernel whose image
  came from the AOT archive.  The engine is fine — the AOT `build/sycl-infer` runs
  the same path to completion — so this reads as an engine bug and is not one.  Build
  a JIT tool against `build-opencode-jit/libsycl_infer_core.a` (`-Lbuild-opencode-jit`).
  See *Benchmarking* for the hand-built tools.
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
  **On the 2x A770 box that cache is a trap** (measured 2026-10-03, driver as
  shipped): with `SYCL_CACHE_PERSISTENT=1` *every* GPU test binary dies with
  `SIGSEGV` a few seconds in - right after the KV-pool init line, i.e. inside
  the first JIT-compiled kernel - while the same binary with the cache unset
  passes (`test_gpu_stages`: "all stages OK").  A brand-new `SYCL_CACHE_DIR`
  fails the same way, so it is not a poisoned directory: the cache itself
  crashes on that driver.  Use the AOT build there (or leave the cache off).

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
--port --host --device --cpu-threads --layer-map --mmproj --mtp --mtp-device
--audio-mmproj`, the three prefix-cache budgets `--pc-vram-mb`/`--pc-mem-mb`
(alias) `--pc-ram-mb`/`--pc-dir`+`--pc-disk-mb`, and
for `gen`:
`--prompt --max-tokens --temp --top-p --top-k --raw --thinking --image --video
--audio --max-video-frames --max-video-side`).  `--device cpu|gpu|auto`
selects the compute backend; `--cpu-threads N` (or `PF_CPU_THREADS`) sets the
host worker count;
`--layer-map 0-11:gpu,12-23:cpu` places closed **inclusive** `begin-end` ranges
on a device (`gpu`, `gpu.N`, or `cpu`/`host`/`1`); the ranges must tile
`[0, n_layer)` in order, with no gaps;
`--mtp [N]` turns the speculative draft on (the length is optional - a bare
`--mtp` is the measured optimum k=4 and never swallows the next flag) and
`--mtp-device N` picks the partition the draft layer runs on (default 0).  Both
are hard-gated: no `blk.<n>.nextn.*` in the GGUF, or no multi-device oneDNN
int8 partition, prints one `[mtp]` line and falls back to the plain decode (see
the MTP section).

`--kv-type K:V` sizes the K and V caches independently (only the scale-carrying
i4/i8 may be mixed).  The attention error is dominated by V, not K: on the 27B
`--kv-type i4:i8` (4-bit keys, 8-bit values) measures mean|diff| 0.061 against
the fp32 CPU reference, versus 0.035 for i8 and 0.183 for i4, at 24 KB/token
(i8 is 32, i4 is 16) - so it fits a 262144-token context on 2x A770 with
near-i8 accuracy.  `i8:i4` is the mirror image and is *worse* than i4/i4
(mean|diff| 1.49), confirming V is the sensitive side.

## Testing

The tests default to `/path/to/Qwen3.5-0.8B-Q4_K_M.gguf` and require the
GPU + that model.  Strict kernel/end-to-end tests set `PF_DP4A=0` (fp32 path).
The exceptions - `test_k5_gemv`, `test_quant_audit`, `test_w4_gemm`,
`test_w4_vs_i8`, `test_gemv_stride`, `test_iq_dequant`, `test_w4_vs_cpuref`,
`test_w4_topk`, `test_27b_prefill`, `test_spec` - default to the 27B at
`/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf` and take it as
`argv[1]`.

```bash
./build/test_tokenizer     # tokenizer round-trips (CPU only, no GPU work)
./build/test_chat_template # GGUF chat template vs reference Jinja2 output (CPU only)
./build/test_response_parser # reasoning_content / tool_call splitter (CPU only)
./build/test_sse_cancel     # SSE client-disconnect cancels generation (real socket,
                              # real httplib, no model needed)
./build/test_sampler       # logit_bias + logprob reporting (CPU only)
./build/test_sched         # scheduler streaming path: greedy, logprobs, max_tokens,
                           # ignore_eos=>"length", cancel, empty prompt,
                           # concurrent sequences (model-backed, no HTTP)
./build/test_multimodal    # image/video prompt, audio decode+mels, vision + audio
                              # encoders (host + device), positions (CPU+GPU)
./build/test_multimodal --video-only  # model-free AVI/odd-size ffmpeg + vision width guard
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
./build/test_k5_gemv       # native 5-bit (Q5_K) store GEMV vs a host reference
                              # (GPU, 27B)
./build/test_gpu_vs_ref    # end-to-end logits vs CPU reference (GPU)
./build/test_forward       # end-to-end logits / top-k (GPU)
./build/test_decode_vs_prefill  # single-token decode == re-prefill, over a prompt
                              # length matrix (GPU; the lengths straddle a
                              # 32-token KV block, a kMaxT prefill chunk, and
                              # the 2048-key oneDNN attention threshold - the
                              # old 9-token default could reach none of them).
                              # Single-device: strict equality, passes every
                              # length.  Under --layer-map: graded OK /
                              # NEAR-TIE / UNSTABLE, because there the *prefill*
                              # is the unstable side (PF_PFB_MAX_M=480/448/384
                              # makes it agree with a decode argmax that is
                              # constant at 198; 256/128/64/32 do not) and the
                              # 27B reference moves between runs.  A reproducible
                              # disagreement under a confident reference still
                              # fails.  TEST_DVP_LENS=... overrides the list.
./build/test_dflash_kernels # DFlash2 top-k + conv vs host references (GPU)
./build/test_spec          # MTP + DFlash2 stream == the plain greedy decode (27B)
```

Always run at least `test_gpu_stages`, `test_gpu_vs_ref` and `test_forward`
after touching kernels/engine, and confirm they still pass (they print
`all stages OK`, `argmax ... SAME`, and the default prompt's predicted argmax).  **Any change that can
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
src/device/     device_profile.h (the `profile` struct + the registry API),
                device_registry.cpp (the `kDevices[]` list, name -> profile
                selection, `PF_DEVICE_INFO` report, the cached `wg_clamped`),
                profiles/<card>.cpp (one card's key, values and matcher each:
                arc_a770, iris_xe) — see "Device profiles" below
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
                 w4_gemv (every native weight store's GEMV: u4/k5/codebook/2-bit
                 g-major SLM-staged decode GEMV, the LUT-expanding codebook GEMV
                 and its prefill expansion, the batched nat_gemm_launch (M <= 13),
                 the opt-in batched u4 GEMM `w4_gemm_launch`, and the fused narrow
                 int8 call group `i8_grp_gemv_rows_multi_launch`), mtp (mtp_concat /
                 mtp_capture for the NextN draft head), mtp_argmax (the verify's
                 accept argmax plus the opt-in gathered draft head: mtp_cand_launch
                 / mtp_gather_launch / mtp_gather_argmax_launch),
                  vit (vision encoder: GEMM, LayerNorm, GELU/bias, 2D RoPE,
                  bidirectional attention), at (audio tower helpers: at_conv1d,
                  at_rope1d), attn_xmx (oneDNN int8 *matmul* prefill attention -
                  NOT the XMX unit, see the "XMX is a name, not a unit" note
                  below; PF_ATTN_XMX below)
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
                sse.h (SSE queue/session + disconnect cancellation, kept out
                of server.cpp so it is testable without a model),
                scheduler.{h,cpp}, server.{h,cpp} (OpenAI chat/completions +
                /v1/models)
src/main.cpp    CLI
tests/common/   cpu_ref.h (CPU reference forward), stage_test.h (stage harness)
tests/backend/gpu/kernels/  test_gemv.cpp, test_dp4a_gemm.cpp, test_gpu_stages.cpp
                + <kernel>_stage.cpp (one per GPU kernel), plus the weight-path
                audits: test_w4_gemm.cpp, test_k5_gemv.cpp, test_w4_vs_i8.cpp,
                test_gemv_stride.cpp, test_iq_dequant.cpp, test_quant_audit.cpp
tests/backend/gpu/  test_forward.cpp, test_gpu_vs_ref.cpp,
                test_pc_gpu.cpp (disk spill + promote round-trip),
                test_pc_ram_gpu.cpp (VRAM->RAM->VRAM round-trip)
tests/backend/cpu/  test_cpuref.cpp, test_pc_cpu.cpp (paged attention + disk
                tier on the host backend; **known failure**, registered as
                `test_pc_cpu_known_fail` with a CTest `WILL_FAIL` waiver - a
                CPU-partition prefix-cache warm resume gives argmax
                248046/198, max|diff| 3.2-3.8; do not call it passing),
                test_pc_disk.cpp / test_pc_ram.cpp
                (tier stores)
tests/backend/cpu/kernels/  test_cpu_gemv.cpp (dequant GEMV + RMSNorm vs the
                host reference; run PF_CPU_ISA=scalar|avx2|avx512 to pin a variant),
                test_cpu_gdn.cpp (cpu_gdn with an asymmetric head count),
                test_dflash_kernels.cpp (DFlash2 top-k + conv vs host references)
tests/model/    test_tokenizer.cpp, test_compare.cpp, test_chat_template.cpp,
                test_w4.cpp (u4 packing round-trip)
tests/server/   test_response_parser.cpp (reasoning_content/tool_call splitter),
                test_sse_cancel.cpp (SSE disconnect cancellation: real httplib +
                real socket reset, no model)
tests/engine/   test_sampler.cpp (logit_bias + logprob reporting),
                test_decode_vs_prefill.cpp (single-token decode vs re-prefill),
                test_spec.cpp (MTP + DFlash2 stream == the plain greedy decode;
                one engine per process, re-execing itself per configuration),
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
4. `validate_hparams` runs after every loader, so a geometry the kernels do not
   implement is refused at load time, before any device allocation.  This is a
   memory-safety guard, not a precision one: `attn.cpp`'s generic attention sizes
   each split's partials with a literal `constexpr int HD = 256` while indexing
   them with `pstride = 2 + head_dim`, so `head_dim != 256` overruns the buffer
   instead of merely losing accuracy - and every specialised kernel falls back to
   that same one.  Also checked: `head_dim % 32`, the GQA ratio, the basic
   extents, `full_attn_interval`, and the GDN geometry.  If your architecture
   needs a geometry outside these, extend the check *and* the kernel together.

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
put kernel bodies in headers.  **A tuned constant that was measured on one GPU
does not belong in the launcher** - add a field to that card's profile
(`src/device/profiles/<card>.cpp`) and read it through `si::dev::active()`, so the
other card and an unknown one can differ; see "Device profiles" below.

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

### Benchmarking on the 2x A770 box

Everything in the two drafter sections below, and the throughput table in
[README.md](README.md), was measured on `bench-host` (two Arc A770 16 GB,
`--layer-map 0-31:gpu.0,32-63:gpu.1`, i8 KV, AOT Release, ctx 131072,
`PF_PREFIX_CACHE=0`).  Five things about that setup are not obvious and each one
silently produces a wrong number rather than an error.

* **Verify the remote is running the code you think it is.**  The box is a source
  snapshot without git, synced file-by-file, so it drifts: a benchmark run against a
  stale `server.cpp` measures the previous revision.  `md5sum -c` a manifest of
  `src`/`tests`/`CMakeLists.txt` over ssh before trusting any number, and rebuild
  after syncing (`find src tests -name '*.cpp' -o -name '*.h' | xargs touch`, because
  rsync preserves mtimes and make will otherwise skip the recompile — which once
  had me reading a stale binary's output as a result).
* **`llama-benchy` needs `--tokenizer Qwen/Qwen3.8-27B`, and only `https_proxy`.**
  The model *id* the server reports is `Qwen3.8-27B`, but that is not a Hub repo id;
  asking for it 401s and then burns minutes in a retry loop with the GPU idle.  Set
  `https_proxy=http://proxy.internal:8118` for the tokenizer fetch and **do not set
  `http_proxy`** — the endpoint under test is `http://127.0.0.1:<port>` and a proxy
  in front of a loopback request is how a local benchmark ends up measuring an
  error page.  Also `PYTHONUNBUFFERED=1` and `--emit-progress <path.jsonl>`, because
  with stdout redirected python block-buffers and a multi-depth run looks hung.
* **`llama-benchy`'s own `pp512` column is wrong at depth > 0 and must not be
  quoted.**  It reports 73 830 tok/s at 16k and 235 563 tok/s at 64k because it
  divides the prompt length by `ttfr` — the time to the first streamed SSE chunk,
  which the server sends *before* the prefill runs.  On one request `ttfr` (0.26 s)
  and `e2e_ttft` (19.5 s) differ by 75x.  Its `est_ppt` column is not the answer
  either: 234 ms at 16k and 281 ms at 64k, essentially flat, when a real 512-token
  chunk over a 64k KV cannot cost less than one over a 16k KV.  **The only
  prefill number that survives at depth > 0 is `prompt_tokens / e2e_ttft`**, and at
  depth 0 the isolated prefill is the `est_ppt` cell (nothing else in flight).
* **A plain-decode baseline measured as wall clock includes the model load.**  The
  27B takes ~80 s to load, so timing a 128-token `gen` end to end gives ~690
  ms/token — a 10x error that still "looks like a number".  Difference two runs
  instead: `wall(N+128) - wall(N)` is exactly 128 decode tokens with the load, the
  `madvise(MADV_DONTNEED)` page-out and the prefill shared.  (Differencing beats
  subtracting a 1-token run: it keeps load-time jitter out instead of dividing it by
  the token count.)  The baseline itself is prompt-independent within noise —
  67.2 / 67.3 / 70.7 ms/token on three prompts.
* **One process per cell, and run the heavy things serially.**  A prefix-cache hit
  inside one process moves the acceptance enough to change the ranking (1.9 cold vs
  2.8 warm), and stacking two GPU jobs on this box makes a card drop out with
  `UR_RESULT_ERROR_DEVICE_LOST`, which looks exactly like a code regression.  Also
  `set -u` / `set -e` must come **after** `source /opt/intel/oneapi/setvars.sh`,
  which dereferences unset variables, and a backgrounded server needs `setsid` (not
  just `nohup`) or the ssh session's process-group teardown SIGTERMs it — the log then
  shows a clean `[srv] shutting down` and you lose a minute of model loading.

Two measurement traps that cost real time and are worth stating once:

* **Do not merge a diagnostic stream into the generated text.**  `2>&1` interleaves
  `[dflash] time:` lines *into* the output mid-token, which made a byte-comparison
  report a spurious DIFFERS.  Separate the streams when comparing streams.
* **A speedup is a ratio, so both sides need the same shape.**  Measuring the
  baseline through the server and the drafter through the CLI compares a graphed
  decode against an ungraphed one.

The `dev/` harnesses are **gitignored**, so they rot silently and nothing fails until
you run one.  `dev/bench_mtp.cpp` (the one-process-per-cell MTP sweep) had stopped
compiling after the MTP state moved into `mspec_state` — and note the trap: the bare
`engine::mtp_k` that remains is the *config* field the constructor takes, so
`e.mtp_k = k` silently became a no-op that would have measured a plain decode and
reported it as speculative.  They are one-file tools compiled by hand, each of which
re-lowers its device image at link:

```bash
icpx -fsycl -std=c++17 -O2 dev/<tool>.cpp -o dev/<tool> \
  -Isrc -Isrc/common -Isrc/backend -Isrc/backend/cpu -Isrc/backend/gpu/kernels \
  -Isrc/model -Isrc/mm -Isrc/engine -Isrc/server -Ithird_party \
  -I/opt/intel/oneapi/dnnl/2026.0/include \
  -Lbuild-opencode-jit -lsycl_infer_core -ldnnl      # -L must match the tool's flags
```

`-Lbuild-opencode-jit` because the command above has no AOT flags; link the AOT
library instead and it dies in the runtime (see *AOT builds*).  The library must also
be **current**, or the link fails on whatever launcher the tool was last changed to
call.  `dev/bench_mtp.cpp` takes
`<gguf> <layer-map> <tokens> <k,k,...> [nprompt] [do_plain]`, and its prompt array
indexes 0 = technical explanation, 1 = story opener, 2 = code continuation.  Run it
with `PF_MTP_TIME=1`, or you get ms/token with no acceptance column — which is the
one number that explains the result.

### Multi-token prediction (MTP / NextN) speculative decoding

`--mtp N` drafts up to `N` tokens with the model's own NextN head and verifies them
against the target in one batched forward, emitting the tokens a plain greedy decode
would.  Two hard requirements, both of which downgrade silently to a plain decode:

* the GGUF must bundle the head — the 27B reference has
  `qwen35.nextn_predict_layers == 1` with `blk.<n_layer>.nextn.*` (`eh_proj`,
  `enorm`, `hnorm`, `attn_norm`, q/k/v/wo/ffn, optional
  `shared_head_norm`/`shared_head_head`); the 0.8B has none;
* the run must be a **multi-device oneDNN int8 partition** (`multi_dev && md_xmx`),
  i.e. a `--layer-map` whose GPU partitions have `setup_md_dnnl()` weights
  (`PF_DP4A` on).  Otherwise `engine`'s constructor prints one `[mtp]` line and
  disables it.  So a single-device `--mtp 4` is a no-op on the 0.8B *and* on a 27B
  that fits one card.

The draft length is clamped to 12 (`n = k+1 <= kMaxB`, the `d_logits` / `step_info`
bound).  The MTP layer is a full-attention Qwen3.5 block, so it owns one extra
attention KV slice (`attn_layers() - 1`) in the same paged pool and is counted in
`--kv-cap-mb` and by all three prefix-cache tiers.

**Semantics that keep the stream equivalent to a plain greedy decode:**

* the head consumes the trunk hidden **before** `output_norm` (`t_h_pre_norm`) and
  applies its own `enorm`/`hnorm`; a draft row pairs `emb(t_p)` with `h_{p-1}`,
  matching llama.cpp's right-shift;
* the verify is one batched forward of `[last_committed, draft0..draft_{k-1}]` at
  consecutive positions; acceptance is `target_argmax(row i) == draft[i]` plus a
  bonus token from row `j`;
* the verify is **dry** (`step_info::mtp_dry`): it computes the forward but does not
  write the GDN/conv state, and snapshots the per-token state into `d_mtp_hist_` so
  the commit rewinds the recurrence to the last accepted row (llama.cpp's
  `n_rs_seq`); the conv window is rebuilt from the raw taps in `d_mtp_qsave_`;
* the prefix cache is supported: `generate_mtp` calls `pc_admit` for the prompt
  (restoring the matched chain's KV *and* recurrent state), then commits **after
  each completed prefill batch and its MTP-layer KV writes**.  The order is
  load-bearing — `pc_commit` advances `pc_slot_[0].registered`, which the next
  batch's `pc_capture_begin` checks against `pos0`, so committing before a batch
  captures nothing and committing only after a multi-batch prompt discards the
  earlier pending checkpoints.  Measured on the 27B / 2x A770 with
  `--mtp 4 gen --temp 0` (~90-token prompt): moving the commit after the prefill
  moved the exit stats from `nodes=2 (with state 0) ... captured=0` to
  `nodes=2 (with state 2) ... captured=2`.  A greedy CLI `gen` routed through
  `generate_mtp` reaches this path, as does the server's `mtp_direct`
  (`PF_MTP_SERVER=1`); ordinary `gen` does not use the prefix cache.

**Caveat (measured, unresolved):** the emitted stream is not *byte*-guaranteed
against a plain decode.  A ~130-token 27B generation on one prompt diverges from
token 11 on, in both draft-head variants and with `PF_MTP_NOACCEPT=1`,
`PF_MTP_NORB=1` and `k=1` — so it is the verify/rollback state, not the draft
quality.  Other prompts are byte-identical.  Treat it as greedy-equivalent in
*intent*; `test_spec` is the oracle that would catch a change here.

#### Measured cost model (27B / 2x A770, 2026-10-09)

Single-request greedy decode, 128 tokens, **one process per cell** (a prefix-cache
hit inside one process moves the acceptance — 1.9 cold vs 2.8 warm — so an
in-process sweep compares states that did not start alike).  Plain decode on the
same build and prompt: **67-70 ms/token**, measured by differencing two generation
lengths so the 27B load cancels (see *Benchmarking* below).  `acc` is the engine's
own number from `PF_MTP_TIME=1`, not inferred from ms/token.

| prompt | `PF_MTP_ADAPT` | acc | cycle | ms/token | speedup |
|---|---|---:|---:|---:|---:|
| code continuation | on (default) | 2.08 | 99.0 | 36.4 | 1.84x |
| technical explanation | on (default) | 1.84 | 100.5 | 39.2 | 1.72x |
| story opener | on (default) | 0.84 | 88.8 | 52.5 | 1.35x |
| code continuation | off | 2.50 | 106.0 | 34.3 | 1.96x |
| technical explanation | off | 2.33 | 105.8 | 35.4 | 1.90x |
| story opener | off | 1.18 | 105.7 | 53.2 | 1.33x |

**The cycle is constant and the speedup is not.**  With adapt off, `draft` is
17.5 ms, `verify` 87.0-87.2 ms and `commit`+`rollback` 1.2 ms on *all three*
prompts — ~106 ms every time — while the speedup spans 1.33x-1.96x.  Nothing about
the engine differs between those cells; only how many of the 4 drafted tokens the
NextN head gets right.  So **acceptance is the entire story**, and it is a property
of the prompt, not of the engine.

`PF_MTP_ADAPT` (default on) shrinks the draft length by one on a cycle that accepts
nothing and grows it back on a full-acceptance cycle, never changing the emitted
stream (acceptance ignores `k`).  **Its sign is prompt-dependent, so measure both
settings.**  It buys a cheaper cycle when acceptance collapses (the story prompt:
88.8 ms instead of 105.7, because fewer rows are verified) but pays 5-6 ms/token on
the two high-acceptance prompts (39.2 vs 35.4, 36.4 vs 34.3) and is a wash on the
third (52.5 vs 53.2).  An earlier version of this file claimed it as an
unconditional win from one prompt pair; that was over-generalised.

**Acceptance is bounded by the text, so the ceiling is ~2-3x and no kernel work
moves it.**  `speedup = n * T_plain / (T_plain + X)` with `n = 1 + acc` and `X` the
cycle-only cost.  Both paths are memory bound and both read the same ~16.3 GB of
weights per unit of work, so the only lever is tokens per weight pass, and 4 greedy
drafts from a 27B carry ~3.1-3.3 accepted tokens.  `X` is ~38 ms here (106 - 67),
of which the extra verify rows are ~4 ms each.

**MTP does not compose with batching, so the server defaults to the scheduler.**
A batched decode step measures `59.0 ms + 10.7 ms per row` (fitted over N=1/2/4/8 at
70.0/81.6/99.7/145.5 ms), and MTP spends `k+1 = 5` verify rows per `acc+1` emitted
tokens — at the measured acc 1.49 that is **2.0 rows per token against the plain
decode's 1.0**.  Both pay the same per-row cost, so for a fixed row budget plain
batching wins as soon as the 59 ms weight pass is amortised: modelled
`S=1/2/4/8 -> plain 69.7/40.2/25.5/18.1 ms/token vs MTP 50.4/36.0/28.8/25.1`, i.e.
MTP wins only at `S<=2` and loses 0.89x at S=4 to 0.72x at S=8.  `server.cpp`'s
`mtp_direct` therefore does not route greedy requests into the single-sequence loop
by default — doing so took the engine for the whole generation and serialised
everything (8 concurrent greedy 128-token requests: 17.5 tok/s with MTP vs 55.0
through the scheduler).  `PF_MTP_SERVER=1` restores it for a strictly
single-request workload.  MTP's remaining value is single-request latency, and it
is strongly prompt-dependent.

**MTP is a short-context win only.**  At 128k depth the 7-row verify costs
1390 ms/cycle against 181 ms for a plain decode: MTP buys a ~3.5x weight
amortization with ~8 decode-equivalents of attention.  On the server the same
prompt goes 68.6 → 45.5 ms/token (1.51x).  Two bugs made it far worse before and
are worth not reintroducing:

* the draft's attention used the *prefill* split ceiling `n_splits`, which the
  `--layer-map` path pins to **1** (`PF_MD_SPLITS`); one warp per head over the
  whole KV made a 1-row draft pass cost 408 ms at 128k.  `mtp_forward` now derives
  its own split count from the KV length (~512 keys per split, capped by
  `PF_MTP_SPLITS`, default `kMaxDecSplits`) and allocates its own partials.  The
  draft phase went 2451 → 35.7 ms/cycle at 128k.  Note a *fixed* large split grid
  degrades short-context acceptance (1.27 vs 2.29 per cycle on a 320-token prompt),
  so derive it, do not hardcode it;
* the prompt prefill used to walk in `kMaxT` chunks, paying a `prefill_flush()` plus
  a host-blocking `memcpy`/`wait` of `d_mtp_hprev` per 32 tokens, which serialised
  the 3-phase pipeline (e2e_ttft at 131k 1475 vs 311 s).  Fixed: one
  `prefill_text()` batch of up to `batched_prefill_fit()` tokens, then the MTP layer
  over its captured hidden in `kMaxT` pieces.

#### The verify's kernel, and the ceiling that bounds it

`nat_gemm_launch` (`w4_gemv.cpp`) is a batched native-width GEMM (u4/k5/cb4/grouped
int8) for M = 2..13 that stages the K-tile's activations in SLM once per workgroup,
so the weight stream is read once for all M rows; it runs the whole
scale/offset/residual epilogue in the same kernel and takes over `gemm_w4`/`gemm`
for M <= 13.  That took the verify from 196 to 147 ms and let `--mtp N` keep the
native q5/cb4 stores (it no longer forces `PF_CB4=0`/`PF_K5=0`).

**The largest single win was a loop-order change, and the reason is invisible in the
source.**  With the **c-outer / m-inner** order (loop over the C columns a sub-group
owns, then over the M rows) IGC scalarises the whole inner loop — the IGC asm
contains **zero `dp4a`** and the kernel runs at 106 GB/s.  Reordering to **m-outer /
c-inner** (load one row's activation group once, then run the 8 `dp4a` for every
column against it) makes IGC emit 336 `dp4a` and the kernel jumps to **221 GB/s**
(u4, M=7, K=5120, N=17408: 0.421 → 0.200 ms, 2.08x; k5 0.342 → 0.237, cb4 0.380 →
0.258, int8 0.370 → 0.276).  The winning shape is `C=2, KT=32, SG=8, TX=128,
TREE=1, VECST=1, VECA=1`, all bit-exact against the M=1 GEMVs.  Also load-bearing:
SIMD8 sub-groups (a SIMD16/SIMD32 float costs 2/4 GRFs, so the M*C accumulator set
spilled 18-23 KB/thread to scratch); staging the activation and the scale/offset
tiles with 16-byte copies rather than byte loops; and the SLM activation stride —
48 bytes pushes M >= 8 over the register cliff, so `APAD` is 32 except for u4 at
M >= 12.  C=4/8 in any order, a larger `TX`, and the c-outer order all lose.

**`nat_gemm` is at a local optimum at M ~ 7 — do not attempt a fourth
restructure.**  Three changes touching SLM, registers and SLM footprint were
implemented and verified bit-exact, and all three lost on the 27B / 2x A770 verify
at M=6 (baseline 91.6 verify / 28.1 draft): SLM double buffer 117.1/30.2, register
prefetch 107.6/31.4, `NAT_DIRECT=1` (read the activation from global, shrink
`act_s` to 1 byte) 96.3/31.7.  **The kernel is occupancy-bound, and occupancy IS the
latency hiding** — many concurrent warps cover one warp's load latency — so spending
SLM or registers on single-warp overlap buys much less than spending them on more
concurrency.  The register form is also redundant: the `gt` loop already carries
`#pragma unroll 2`, so the compiler already hoists iteration *i+1*'s loads across
iteration *i*'s dp4a.  And the activation staging is **not** overhead: `nat_gemm`
reaches 223 GB/s at M=7 where the M=1 GEMV reaches 294 on the same weights, which
looks like `act_s`'s ~7 KB, but staging turns 24 dependent global loads per thread
into 24 register reads.  After the reorder the kernel is within ~1.3-1.5x of the M=1
rate (u4 223 / k5 188 / cb4 173 / int8 323 GB/s against 294/249/261/364); on *true*
per-format traffic k5 already matches u4's ~280 GB/s while reading 20 % more bytes,
and cb4 is decode-issue bound on its codebook LUT (a 256-entry uint32 table, a
pre-biased codebook and pre-shifted LUTs all measured worse).

**The arithmetic-intensity ceiling.**  Measured on one A770: stream rate ~376 GB/s
(`bw_probe`); dp4a ~4.9 T-MAC/s with register-resident operands
(`dev/bench_dp4a_peak.cpp`: 3.50 plain, 4.92 with u4 nibble decode, **78.8** for
DPAS u4 K=64); fp32 FMA peak ~5.0 T-FMA/s (`dev/alu_probe2.cpp` — the older
`alu_probe.cpp` is dead, the compiler folds its loop and it reports 1.5 PFLOP/s).
So the memory system's intensity is **~13 MAC/byte** against a tensor-core card's
20-30x, and **dp4a is not a hardware unit on Xe-LP** — it sits at 70-98 % of the
vector ALU, so DPAS really is a separate ~16x unit.

The critical batch size follows: with the u4 store's 0.625 B/weight and the rate the
*real* kernel achieves (**7.1-7.6 T-MAC/s**, from `dev/bench_wpass.cpp` replaying the
real 497-call list: `53.93 ms` of streaming `+ (M-1) x 3.67 ms` of dp4a),
`M* ~= 0.625 x 7.6e12 / 405e9 ~= 12`.  Below `M*` extra rows are nearly free; above
it each costs real time.  The equivalent DPAS crossover is ~122, so an XMX/DPAS unit
only wins for **M in [12, 122]** — and `kMaxB = 16` caps every batched forward at
M = 16, the first shape inside that window.  **Take `M*` from `bench_wpass`, not from
`bench_dp4a_peak`**: the latter understates the real kernel by 1.4-2x, and an `M*`
derived from it is too small.  (The older 6.4 figure came from the 4.9 T-MAC/s
number and is wrong.)

**XMX / DPAS was re-implemented and rejected on measured grounds — do not re-derive.**
The earlier rejection rested on "DPAS K is fixed at 32 = our group width", which is
too narrow: the native `esimd::dpas<8,8,int32_t,int32_t,uint32_t,uint32_t,u4,s8,64,
32,64>` takes a **u4** operand with **K=64**, i.e. two 32-wide groups chained without
a drain, and it is bit-exact for a per-32 u4 stream (0/64 and 0/200 mismatches).
Raw throughput with register-resident operands is **254 T-MAC/s** (u4), and `s8`
(K=32) is 16x slower at 16.2 — the Xe XMX apparently lacks a native signed-8 path.
It still loses by a wide margin in a real GEMM.  `dev/bench_dpas_gemm.cpp` does the
right thing — a weight plane pre-reordered once at load into the DPAS B layout, an
SLM-staged A tile, ESIMD `gather`, `block_store` out — and is **numerically exact**
(rel 5.26e-07 at the verify shape, all 2560 dpas values exact across all 160 groups)
and still **7.44 ms against `nat_gemm`'s 0.200 ms, 37x slower**.  Where it goes is
measured: K-split regresses exactly linearly (1/2/4/8/16/32 → 7.69/14.80/28.92/57.10/
112.82/231.73 ms), so the cost is proportional to the **work-group count** — building
the two 64/32-element cross-lane **offset vectors** (a runtime element write or a
union'd array write into a cross-lane simd is serialised through local memory) and
spilling the `NT x 8 x 8` accumulators.  `NT` sweep 1/2/4/8 → 13.27/9.16/**7.44**/8.93.
Three ways to build those offsets that **do not work**: an initialiser list
(`esimd::simd<int,64>{0,4,8,...}`) has no matching constructor at either length; the
`VS`-blocked `gather<VS>(p, simd<OffsetT, N/VS>)` overloads require a `simd_view`
pass-through argument; and there is no `(start, step)` / iota constructor and no
identity-`offsets` helper (the `offsets` in `memory.hpp` are gather *masks*).
Other measured constraints: `esimd::gather` needs **compile-time** offsets;
cross-lane vectors assume a **32-thread block** (an 8-warp workgroup produces zeros,
`W=1` is required); `block_store` is block-cooperative so guarding it to one warp
stores nothing; and **read results back with `block_store` to global, never through
SLM** — a `copy_to(local_accessor)` plus a scalar SLM loop returns zero even for a
plain scalar round trip in the 32-lane configuration, which is what made every
earlier probe (including the previous session's 0.06 T-MAC/s) look like it was
computing nothing.  `src/common/dp4a.h` remains the pattern to imitate.

Cost/benefit, since the hardware verdict above is only half the story: on this
model only ~3.5 ms of a 14.25 ms marginal GEMM cost is u4/k5 dp4a issue; 3.84 ms is
the 96 tiny `ssm_alpha`/`ssm_beta` GEMVs (2 workgroups each — launch latency, which
no arithmetic unit fixes) and 4.2 ms is the cb4 codebook LUT, which **cannot** be a
DPAS u4 operand at all (16 arbitrary int8 values vs values 0-15).  So a DPAS verify
kernel is worth ~3 ms of a ~106 ms cycle here.

**`nat_gemm_launch` cannot serve prefill, and that is a hard limit, not a missing
instantiation.**  `nat_gemm_impl` stages the call's whole 32-group activation tile
in SLM (`M x 1024` bytes), which with `APAD=32` caps it at M ~ 56 rows and with the
`APAD=48` u4 prefers at M >= 12, M ~ 37 — against DG2's 64 KB of SLM per workgroup.
Prefill's GEMM M is the *total token count* (`tbm = (mode == 2 && !single) ? rows :
tb`), so a 512-token prompt is **one** mode-2 forward at M=512 and no prefill call
lands inside the M <= 13 window.  On the main configuration prefill is always mode 2
and mode 1's M=kMaxT=32 chunks are never used.

**What prefill actually costs** (27B / 2x A770, `llama-benchy 0.4.0 --pp 512 --tg
128 --depth 0 --runs 3 --exact-tg`, `PF_PREFIX_CACHE=0`): **pp512 2 390 tok/s**,
tg128 14.05 tok/s, cold full-prompt TTFT 1.44 s.  512 tokens in 214 ms against
24.5 GB of *int8* weight bytes is ~114 GB/s effective, i.e. ~35 % of the card's read
ceiling, so prefill is not purely weight-bound; oneDNN's grouped-scale int8 matmul
is doing real work here.  Replaying the same 497-call list with the production dp4a
launchers would cost ~1.93 s at M=512, i.e. **oneDNN is ~11x better than dp4a at
this M** — which is why prefill needs a *blocked large-M* kernel and neither
`nat_gemm` (M-capped) nor dp4a is the answer.  At depth 16k/64k the whole-prompt
rates are 865 / 560 tok/s.  See *Benchmarking* below for why the harness's own
`pp512` column cannot be quoted.

**The verify's own floor.**  A `k` sweep prices it at ~one plain decode's weight
pass plus ~4 ms per extra row (the verify is 83-87 ms at k=4 across all three
prompts here), and it runs on `build_md_verify_graphs`' recorded command graphs, not
the direct replay.  Two guards earn their keep: `dnnl_capture_guard()` throws from
every oneDNN `prim.execute` while a declared-all-SYCL capture is in progress (the
verify's GEMMs all take `nat_gemm_launch`, but an unservable shape would fall back
to oneDNN and silently record a pass that is missing work), and `vf_graph_usable()`
declines the graph once the context outgrows `xmx_min_keys`, where the recorded
attention split count would be stale.

**The accept argmax is on the device because the host version was 24 % of the
cycle.**  `mtp_argmax_launch` was first written as `parallel_for(range<1>(M))` — one
work-item per row — so a 5-row verify meant five GPU threads each scanning 248320
floats serially: **33.7 ms/cycle**, and since `mtp_verify` ends with `sync_all()`
that is real GPU occupancy, not a hidden sync.  It is now one 256-lane work-group per
row with a strided coalesced walk and an SLM tree reduction, breaking ties to the
*lowest* index (as does the per-lane scan) to stay bit-identical to the host's
`if (v[i] > best)`: **33.7 → 1.2 ms/cycle (28x)**, verified by `PF_MTP_AMCHK`.
`compute_backend::mtp_argmax` is now the only entry point (`mtp_argmax.cpp` on the
GPU backend, a host loop in `cpu_backend.cpp` for an all-`cpu` run), so the device
path is what the verify uses and **`PF_MTP_ARGMAX_CPU` no longer exists**.

#### Dead ends (all measured — do not retry)

* **Fewer bits in the draft.**  `PF_MTP_LAYER_EXACT=1` (exact fp32 dequant GEMVs)
  gives an *identical* acceptance to int8, so draft precision is free — which means
  every "fewer bits" attempt can only lose the head's discrete argmax decisions, and
  all of them did.  `PF_MTP_LAYER_W4` (default on, u4 linears, 2.06 → 0.875 B/w,
  -1.4 ms/cycle) is accuracy-neutral.  `PF_MTP_LAYER_W2` (acc 2.14 → 1.75 for
  -0.5 ms) and `PF_MTP_HEAD_W2` (-2.6 ms/cycle for -2 % acceptance) are rejected:
  lossy is free at the *readout* but compounds in the *recurrence*.  A 2-bit head
  is also a wash because that GEMV is not bandwidth-bound (270 vs 314 GB/s), so only
  1.4x of the 1.6x byte cut is realised.  Note the first `PF_MTP_LAYER_W4`
  measurement showed acc collapsing to 0.11 — that was the `do_split` bug below, not
  the precision.
* **A candidate-restricted draft head** (`PF_MTP_CAND`, default off and it should
  stay off).  `mtp_gather_launch` evaluates the head on a few hundred gathered rows
  instead of all 248320 (0.25 MB vs 794 MB), is numerically exact, and took the
  draft from 19.1 to 10.9 ms/cycle — but the candidate set contains the next step's
  argmax only 25/17/7 % of the time at margin 8, and 81/77/83 % even at 16384 rows
  (7 % of the vocab).  The distribution moves too far in one token.  Net 43.4 →
  42.9 ms/token on one prompt, 48.4 → 50.4 on another.
* **Splitting the draft head readout across both cards** (`PF_MTP_HEAD_SPLIT`,
  default off): deterministically 17.2 → 13.1 ms/cycle and an end-to-end wash,
  because halving N changes the u4 GEMV's decomposition and flips the draft's
  argmax on a near-tie (p1 bit-identical over 128 tokens, p0 -6 %, p2 -8 %
  acceptance).
* **Merging the commit and rollback syncs.**  `commit` measured 4.0 ms and looked
  like pure overhead, but it is one device-wide barrier per cycle: merging it into
  the rollback's just moves the cost (commit 4.0 → 0.3, rb 1.5 → 4.0, net unchanged
  at 36.2 vs 36.4 ms/token).  **The barrier itself is the cost.**  For the same
  reason the rollback's own barrier is kept: removing it takes the phase from 3.9 to
  0.9 ms/cycle and the wall does not move (8.74 → 8.76 s for 192 tokens), because
  the freed time reappears in the draft's first wait.

#### Two bugs that look like tuning

* **Any activation feeding a native-store GEMV must be quantized with
  `do_split=true`.**  `eh_proj`'s activation was not, so the u4 path read the
  *previous* call's even/odd planes and gave acc 0.08 (garbage draft) until it was
  fixed.  `quantize()` now returns an `si::act_view` (`{m, k, split, gen}`) and every
  GEMM entry point *requires* it, so "forgot the gate" no longer compiles, and a
  stale view returns `nullptr` from the `act_*` accessors instead of handing back
  whatever is in the scratch.
* **Never put a device query in a kernel launcher.**  The device-profile refactor
  introduced `rmsnorm_launch` calling an uncached `sycl::device::get_devices()` per
  launch — 3.2 ms each, 210 ms of host time in one verify pass (65 rmsnorm calls).
  It was invisible in the plain decode because that is a *recorded* command graph
  (captured once at startup) while the verify is a direct replay; fixing it took the
  verify 221 → 85 ms.  See *Device profiles*.

Diagnostics, all env-gated (`0`/unset = off unless noted): `PF_MTP` (draft length,
`--mtp` overrides), `PF_MTP_LAYER_W4`, `PF_MTP_ADAPT` (`0` fixes `k`),
`PF_MTP_AMCHK`, `PF_MTP_DEV`, `PF_MTP_TIME` (per-phase cycle ms **and `acc`** — the
acceptance column any benchmark needs), `PF_MTP_SUBMIT`, `PF_MTP_DEBUG`/`PF_MTP_DUMP`,
`PF_MTP_VERIFY_PAD`/`PF_MTP_VERIFY_M32`, `PF_MTP_VERIFYN`, `PF_MTP_DECCHK`,
`PF_MTP_LSTAT`, `PF_MTP_DECODE_H`, `PF_MTP_NOACCEPT`, `PF_MTP_NOMTPFWD`,
`PF_MTP_NORB`, `PF_MTP_NORBSYNC`, `PF_MTP_FORCE_INT8`, `PF_MTP_STEPS=1` (per-depth
acceptance `reach[j]`, which distinguishes a geometric decay from a draft-state bug),
`PF_MTP_DSTEP=1`, `PF_MTP_LAYER_EXACT=1`, `PF_MTP_LAYER_W2`/`_W2_CALL`,
`PF_MTP_LAYER_W4_CALL=<ci>` (put only that call's tensors on the u4 grid — 0 qkv,
1 wo, 2 ffn gate/up, 3 ffn down, -1 eh_proj; the bisection knob that localised the
`do_split` bug), `PF_MTP_HEAD_SPLIT`(+`_DEBUG`), `PF_MTP_HEAD_W2`, `PF_MTP_CAND*`
(`PF_MTP_CAND`/`_CANDM`/`_CANDSRC`/`_CANDDBG`/`_CANDV`).  Batch-GEMM knobs:
`PF_MTP_SPLITS` (the draft's key-split cap, default `kMaxDecSplits`; `1` restores the
old single-split draft), `PF_W4_GEMM_MAXM`/`_U`/`_TN`, `PF_DNNL_BLOCKED`,
`PF_NOFUSE`, `PF_FUSEDBG`.
`dev/bench_mtp.cpp` is the one-process-per-cell sweep; `dev/bench_natgemm.cpp`,
`dev/bench_wpass.cpp` (+ `dev/wpass_list.py`) and `dev/bench_dpas_gemm.cpp` are the
kernel probes — see *Benchmarking* for how to build and run them.

### DFlash2 block drafter (`--spec-type dflash2`)

A second drafter, structurally different from MTP: the draft GGUF's own 5-layer
NextN-style head runs **one non-causal forward over `[anchor, MASK x (n_max)]`**
inside a private K/V ring, and a selector scores the top-K candidate sets per block
position plus every ordered pair, so the host walks one coherent path through the
lattice instead of re-drawing per token.  Model `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`,
5 layers, `n_embd` 5120, 32/8 heads, `head_dim` 128, block 8, `swa` 2048, mask id
248070, selector rank 256 / top 16, target layers `[6,20,34,48,62]`.  Files:
`src/model/dflash.{h,cpp}`, `src/backend/gpu/kernels/dflash.cpp`,
`src/engine/engine_dflash.cpp`, plus the capture hook in `engine_graph.cpp`.

#### Acceptance is the whole story, and it is prompt-dependent

`n_max` is the draft tokens per cycle; `acc` is accepted drafts per cycle, so the
ceiling on speedup is `1 + acc`.  Each cell is its own process.  Code-continuation
prompt, plain decode **67.2 ms/token** (see *Benchmarking* for how the baseline is
measured without the 27B load swamping it):

| `n_max` | 1 | 2 | 3 | 4 | **5** | 6 |
|---|---:|---:|---:|---:|---:|---:|
| acc | 0.92 | 1.80 | 2.84 | 3.46 | **4.25** | 5.05 |
| ms/token | 46.1 | 33.8 | 26.1 | 24.0 | **21.8** | 20.1 |
| verify ms | 74.3 | 78.4 | 82.3 | 87.1 | **92.4** | 97.4 |

**But that is one prompt, and the spread across prompts is 8x.**  At `n_max=5`:

| prompt | acc | ms/token | vs its own plain baseline |
|---|---:|---:|---:|
| code continuation | 4.25 | 21.8 | **3.08x** |
| technical explanation | 2.19 | 35.8 | 1.88x |
| story opener | 0.49 | 77.0 | **0.92x — slower than no drafter** |

At acc 0.49 each cycle emits 1.49 tokens for a 114 ms cycle, which is strictly worse
than the 70.7 ms plain step, and **no `n_max` rescues it** — the whole sweep above is
worse in ms/token on the story prompt than 21.8, and raising `n_max` makes the cycle
grow faster than acceptance improves.  So acceptance has to be checked per workload
before enabling this, and a single speedup number is not a property of the drafter.
This is the same lesson as MTP's, and it is the one most likely to be got wrong: both
drafters' published speedups are single-prompt numbers.

The emitted stream is **byte-identical to a plain greedy decode's** (verified: same
md5 over 128 tokens with diagnostics on stderr), which is the correctness oracle —
`test_spec` asserts it as stream equality against an explicitly non-speculative
`generate_plain`, not acceptance, because a broken verify can accept at a perfectly
healthy rate and still emit the wrong text.

An earlier table in this file recorded acc 0.98/1.44/2.18/2.54/2.90/2.90 and
31.5 ms/token at `n_max=5`, against the 0.92/1.80/2.84/3.46/4.25/5.05 and 21.8 ms
measured now.  The **cycle cost is unchanged** (`verify` 92.4 vs 90.9 ms; both fit
"one plain decode's weight pass plus ~4-5 ms per row"), so the difference is
acceptance, not engine work, and the drafter path and the shared per-queue profile
resolution under it have changed since.  The cause was not bisected; what is
established is that the text is unchanged.

#### Where the cycle goes, and its floor

At `n_max=5`: `verify=92.4`, `draft=18.3`, `emit=1.5`, `inject=1.1`,
`rollback=1.0` → 114.3 ms.  A least-squares fit of the verify column prices it at
**69.1 ms fixed + 4.64 ms per extra draft row** (endpoints predict 73.7 / 96.9 ms
against 74.3 / 97.4 measured).  The fixed part is one plain decode's weight pass,
which is the structural floor; the two remaining levers are fewer weight bytes and a
faster dp4a, and both are already priced and rejected (`PF_W4_K5` -5.7 % at 4.60 %
mean weight error, `PF_W4_ALL` -29 % bytes at ~8x error, DPAS 16-24x the issue rate
and 37x slower in a real GEMM).  **Do not re-derive either.**

The verify runs on `mtp_verify`'s recorded command graphs, not the direct replay —
the thing that cost 210 ms on the MTP path is not happening here, and
`PF_DFLASH_VFCHK` prints `vf_dec_ok`, the row count and the graph count (1, 6, 378)
so you can confirm it rather than assume it.

#### Three bugs that were load-bearing, and none was findable from the inside

Every host check passed throughout, because each check restates the bug:

* the block forward projected `wq` but **not `wk`/`wv`**, so the attention read K and
  V out of whatever the injection path last left in the fused buffer.  The injection
  writes the committed positions, so the ring held valid K/V for those and an anchor
  row (which attends mostly to committed keys) still tracked llama.cpp at cos 0.994,
  while the block's own rows attended to stale K/V.  **"Anchor close, mask rows off"
  is the signature.**  0.03 → 0.48 acceptance.
* the conv composes two coefficient tensors with **mirrored axes**: base's side slice
  is reshaped to `[group_size, n_groups, kernel_size]` (channel innermost, so
  `(c,t)` is `c + width*t`) while the dynamic is `[n_groups, kernel, 2, tokens]`
  (so `(g,t,side)` is `g + n_groups*t + side*n_groups*kernel`).  Both were wrong;
  fixing base first took the anchor row — which only ever uses tap 0 — to cos
  0.999997, which is what isolated the second error to the tap/side coefficients.
* the draft's rope **rotates the whole `head_dim`**.  `dimension_sections` is
  `[64,0,0,0]` and reading `sections[0]` as the rotated width gives 64; the
  reference rotates 128.  That is a fixed ~0.7 % on every attention score, invisible
  in any aggregate, and compounds to cos 0.79 by layer 4.  Measured against
  `DFLASH_REF_CUR` at pos0=204: n_rot 32/64/96/128 → 0.9930 / 0.9932 / 0.9947 /
  **0.999991**.

`attn_sinks` is wired up (ggml's `ggml_soft_max_add_sinks` semantics: one extra key
per query head, score `sinks[h]`, value 0, so the whole effect is a factor
`z / (z + exp(sink - max))`) but **this GGUF carries no such tensor** — 15 `blk.N.*`
tensors, `attn_sinks` not among them, and llama.cpp creates it `TENSOR_NOT_REQUIRED`.
So it is a no-op here; it is wired rather than assumed absent because a draft that
*does* carry sinks would be wrong on every head.

#### The one genuinely slow kernel is the draft's top-k, and it is 100x off the floor

The readout does a top-16 over `[M=6][n_vocab=248320]` logits — 5.96 MB — and
measured **2.06 ms**, about 2.8 GB/s where the floor is ~300 GB/s.  It is not the
obvious suspect: transposing the per-lane top-K SLM lists from `[tid][k]` to `[k][tid]`
removes a real 16-way bank conflict (stride-16 floats put all 256 lanes on two banks,
hit on every one of ~970 element compares *and* 16x per reduction round) and changes
nothing, 2.06 → 2.11 ms.  It was **six workgroups on a 512-EU GPU**: one workgroup
per row is all the parallelism `M=6` allows.  `df_topk_launch` now reduces `S` slices
per row into `[M][S][K]` partials and merges them with one more workgroup per row;
`PF_DFLASH_SLICES` overrides `S`.

Two things about that implementation are worth not re-deriving.  The slices must be
**one launch** covering `M*S` workgroups with the bounds computed from the group
index — the first version submitted one kernel per slice so the bounds could be
literal, which serialized the work and measured **8.0 ms, 4x worse**.  And the
optimum is **not** the occupancy-maximising value: `S=1/8/32/64/128/256` measures
`2.09/1.39/1.52/1.69/1.70/1.67 ms`, so `S=8` ships.  Past 8 the merge and its `S*K`
candidates cost more than the parallelism buys, which is the tell that the kernel is
**latency-bound on its per-lane SLM insertion chain**, not bandwidth-bound — 1.39 ms
for 5.96 MB is still ~4 GB/s.  A real rewrite would be a per-lane threshold in a
register plus a warp-ballot merge, not more slices.  Net: topk 2.06 → 1.39 ms, draft
28.0 → 26.9 ms, cycle 122.9 → 121.8 ms, output byte-identical.

The draft's readout tail also looked like 5.7 ms of host work and is 1.8 ms of device
work (`PF_DFLASH_TOPTIME` vs `PF_DFLASH_DTTAIL`); the lattice walk itself is free
(0.001 ms), so the whole tail is the two kernels and the blocking wait.

#### A lesson worth more than the numbers: A/B subtraction is not attribution here

**`PF_DFLASH_OPTIME` (per-op, inserted by LINE NUMBER — every text-anchored insert
landed in the injection path, which shares this code) must NOT be read as device
time.**  It attributed 16.8 of 20.8 ms to `norm + attn_conv_proj + attn conv0`, which
is why the FFN looked free; and `gate+up` was reported reading 167 MB of weights in
0.2 ms, which is impossible on an in-order queue — it submits, and the device time is
charged to whoever waits next.  Pricing the same phases with no-op knobs
(`PF_DFLASH_NOGEMM` / `PF_DFLASH_NOCONV`, both produce wrong results and exist only
for this) shows the smear directly: everything 20.8 ms, convs removed 17.8, GEMMs
removed 14.0, both removed 12.1 — and the removal does not land where the table says.
With both removed, `attn_norm`, `attn_conv_proj`, `wo` and `ffn_down` all report
**0.0**, not because they are free but because the heavy kernels are gone and nothing
blocks.  **So the three numbers do not add up, and A/B subtraction is not a valid
attribution on an in-order queue**; the only sound figure is the wall clock.

The device-side table meant to replace it had three bugs of its own: the segment
accumulation ran **once after the layer loop**, so it differenced the *last* layer's
markers; the markers are **not in index order** (program order is `0, 6, 5, 1, 3, 4`),
so an index-order loop differenced two early markers against two late ones and
printed **-3.60 ms**; and `seg_e[0]` sat **outside** the loop while every other marker
was inside it, so segment 0 measured "before layer 0 → layer 4's norm" — that single
line produced "the RMSNorm is 82 %".  Five RMSNorms moving 614 KB take **0.36 ms**,
0.07 ms per layer.

Fixed, the device timeline accounts for itself and the profile is flat:

    anorm=0.36  acproj=1.71  aconv0=2.27  attn=7.05          (qkv+rope+attn+wo+conv1)
    ffn: norm+cproj+cconv=1.89  gate_up=2.75  down+cconv=4.33
    sum=20.35   devspan=20.37   hostwall=20.70 ms

`sum ≈ devspan` to 0.1 % is the check that matters: `devspan` is the first layer's
start barrier to the last layer's end barrier, so **the forward is device-bound**
(20.37 of 20.70 ms), not submission-bound — which A/B subtraction could never have
told you.  There is no 82 % hotspot: the largest single item is the attention path
at 35 %, then `ffn_down + conv1` at 21 %.

**The lesson is the one this file already records elsewhere: a broken verifier is
worse than none.**  Here the verifier was self-consistent enough to look
authoritative — plausible positive numbers, one segment dominating, and a
ready-made physical story attached to it ("0.037 GB/s is impossible, so look at the
host path").  Every one of those numbers was a mis-differenced timestamp.
**A timing breakdown must be checked against an independent total before it is
believed**; `devspan` exists for exactly that, and `sum ≈ devspan` should be the
first thing printed rather than the last thing added.

#### Comparing against llama.cpp

Acceptance against llama.cpp at the same `n_max` (short prompt, mean accepted tokens
per cycle), from the pass that introduced the drafter — **not re-measured
2026-10-09**, because it needs patched llama.cpp probes and this box's
`~/llama.cpp/build/bin` has no DFlash tool:

| generated | 32 | 116 | 227 |
|---|---:|---:|---:|
| llama.cpp | 4.43 | 3.96 | 4.13 |
| this engine | 4.75 | 3.90 | 3.92 |

On a 204-token prompt both engines drop (llama.cpp 0.239 / mean 2.07, this engine
0.81 / 1.81), so the long-context gap is ~12 %, not the 5x an earlier broken state
suggested.

**Compare element-wise, never by fingerprint.**  `DFLASH_REF_{NOISE,CUR,FFNIN,
FFNCONV,FFNOUT,XNORM,DYN,HIDDEN,LAYER}` publish a tensor at a layer chosen by
`DFLASH_REF_IL=<il>` (default 0), and `DFLASH_REF_BIN=<path>` writes it raw; on this
side `PF_DFLASH_BIN=<path>` does the same, with `PF_DFLASH_SIGALL` adding every layer
and `PF_DFLASH_{SIG,GEMMCHK,PROJCHK,CONVCHK,CONVCHKALL,BASECHK,ROWRMS,EMBCHK,INJCHK}`
selecting probes.  `rms`/`max`/`first`/`top-6` say how big a divergence is but never
where, and the anchor row carries an order of magnitude more energy than the mask
rows, so a whole-batch aggregate is **not comparable with llama.cpp's per-position
print at all** — an aggregate that said the `wo` output was 30 % low was comparing
two different things.  llama.cpp's copy-out is fixed at `n_embd` wide, so a narrower
tensor has to be padded on its side (`DFLASH_REF_DYN` does this).

**A diagnostic whose identity depends on when it fired needs that context in its
name.**  Four confident wrong answers came from this, and they cost more time than
the bugs did: reading the f16 ring as f32 (1e12 rms garbage and a 104/205 "coverage
gap"); reading the layer output from `d_df_h` instead of `d_df_c`, which fabricated a
clean-looking error-accumulation curve; naming dumps by layer alone, so a later block
overwrote an earlier one and two readings of one layer disagreed by exactly one layer
(which produced a very convincing "layer 4 reads the wrong tensor" — the layer index
was being set in a *different* function, by a `replace(...,1)` that matched the first
of two identical `const dflash_layer_t &` declarations, the same mistake that
segfaulted a probe); and `df_conv_check` keeping the pre-fix index derivation, so
after the conv axes were corrected it reported the difference between two formulas
while presenting itself as a verification of the conv, and reported host 0.805 vs dev
1.545 at layer 0.

### What the new DFlash2 tests caught

`tests/backend/gpu/kernels/test_dflash_kernels.cpp` (top-k + conv vs host
references) and `tests/engine/test_spec.cpp` (both drafters vs the plain greedy
decode) each found a real bug that nothing else could reach.  All three are the
shape AGENTS.md keeps warning about: **a plausible-looking thing that is wrong,
and that nothing in the product path happens to exercise.**  (These are *not* the
three forward bugs above — those were found by diffing against llama.cpp.)

1. **`df_conv_launch` had a latent launch-geometry bug.**  It used a fixed
   256-thread local size, so SYCL rejected any `n_rows*width` that was not a
   multiple of 256 with *"Non-uniform work-groups are not supported"*.  The 27B
   shapes are multiples by luck (6*5120 = 120*256, 6*4096 = 96*256), which is why
   it never fired in the model - the kernel's own comment even says *"width is not
   a multiple of the 256-thread group"* while the launch assumed it was.  The
   kernel has no barriers and no local memory, so a work-group size buys nothing;
   it is now a plain `range<1>`.

2. **`df_topk_launch`'s `K` guard admitted a launch that could not run.**  The
   per-lane SLM lists are two `256*K+1` accessors plus two `256`-element reduction
   scratch arrays, so the real budget is `8*(256*K+1) + 8*256 <= 65536`, i.e.
   **`K <= 30`**.  The guard said `K <= 32` with `WG*K+2` lists and no scratch in
   the budget, so K=31 (65544 bytes) and K=32 (65552) both **failed to launch**
   with `UR_RESULT_ERROR_OUT_OF_RESOURCES` - a config error reported as a
   resource error, which killed the process rather than degrading.  The cap is now
   `static_assert`ed from the budget so it cannot drift.  DFlash2's `sel_top_k` is
   16, so no real model reaches it; the test asserts the out-of-range K is
   *ignored and leaves the caller's buffers untouched*.

3. **`generate_mtp` had no gate on `mtp_on`.**  The constructor clears `mtp_k`
   (hence `mtp_on`) whenever a gate fails - no NextN head, no multi-device oneDNN
   int8 partition - and leaves every `d_mtp_*` buffer null, but `generate_mtp`
   walked into the speculative loop regardless.  It surfaced from inside the
   verify's plan build as *"multi-device: weight tensor not uploaded to this
   device's partition"*, i.e. as a **weight-partition bug** when the actual fault
   was calling the wrong entry point.  It now falls through to `generate`, the
   way `generate_dflash` already gates on `dfm_`.  This is the kind of thing a
   test harness or a second-model server hits immediately and the CLI never does,
   because `main.cpp` builds one engine and routes by `--spec-type`.

Two things `test_spec` does that are worth stating.  It asserts **stream equality
against an explicitly non-speculative `generate_plain` decode**, not acceptance: a
broken verify can accept at a perfectly healthy rate and still emit the wrong text,
because acceptance only compares against the target's own argmax on the rows it
kept.  And it runs **one engine per process** (re-execing itself per
configuration), because a second engine in the same process inherits state from the
first - an MTP-enabled engine followed by a DFlash one fails at construction, while
each alone and the reverse order both work.  The CLI never hits that because it
builds one engine; the test would otherwise have encoded the limitation as a
failure.  It now also covers the two-engines-in-one-process case explicitly
(`TEST_SPEC_ONLY=two_engines`), because that is a real configuration and the
limitation should be documented rather than dodged.

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

## Device profiles: where a GPU-specific number belongs

**Rule: a constant that was measured on a particular GPU lives in that GPU's
device profile, not as a literal at its use site.**  The use site asks for it by
name.

    src/device/profiles/<card>.cpp  one card's COMPLETE implementation: its
                                    values, its key, and its own matcher
    src/device/device_registry.cpp  the kDevices[] list of those entries, and
                                    the selection/reporting that reads them
    src/device/device_profile.h     the `profile` struct and the API

* **What belongs there**: occupancy (`occ.warps_per_eu_x2`, which sets the decode
  K-split), the SLM budget the staging GEMVs size against, the split heuristics
  that fill the machine (`split.gemv_rows`, `split.gemm_rows`, `split.max`),
  per-kernel shapes (`shape.*`, `attn.xmx_gather_red`), and the *default* for any
  on/off feature whose value was chosen by measurement (`wt.w4`, `wt.k5`,
  `wt.cb4`, `attn.xmx`, `attn.dec_group`, ...).
* **What stays a literal**: hardware facts SYCL already reports and that cannot
  change with tuning — they are queried, with the profile as the fallback.  And
  anything a *model* dictates rather than a card (`kBlockSize`, `kI8Q`,
  `kMaxT`/`kMaxB`, `head_dim`).
* **An env var still overrides** the profile: the profile is the default, the env
  var is the A/B.  Keep that ordering (`env ? atoi(env) : profile`), and keep the
  profile value the one that is *measured* — an env default that was never
  measured is a bug.
* **Both parts are measured, not one assumed.**  `dec_group` is off on the A770
  because the grouped decode kernel is 5x slower there, and off on the Iris Xe
  because the *classic* kernel is faster there — same value, opposite reasons.
  That is the case the split exists for.
* **Say which numbers are inherited.**  A field marked INHERITED is a value
  carried across from the other card because the two share an architecture (both
  are Xe-LP, so the 8-warp/EU sub-group lattice is common) but which was *not*
  re-measured.  Unmeasured is a liability; write it down rather than implying it
  was checked.  The Iris Xe profile currently marks most of its tuning inherited:
  the decode-split microbenchmark on an integrated part returned 2.1-4.4 ms for
  one configuration across three repeats, i.e. the noise exceeded the effect, so
  the sweep has to be redone on an idle box before those values are trusted.
* **The registry holds no knowledge of any card.**  Detection is a function
  pointer: each `<card>.cpp` owns its own `arc_a770_matches(name)` /
  `iris_xe_matches(name)`, and `for_name()` just asks each registered card in
  order.  So "how do we recognise an A770" is answered in the A770's file, and
  adding a card never means editing a central `if` on device names.  Order is
  match order (first match wins), so a broad matcher goes after a specific one --
  the Iris Xe matcher matches only "Iris", deliberately not "Xe", because a
  future Xe part has different tunings and must get its own file.
* **Never query the device from a kernel launcher.**  `rmsnorm_launch` calls
  `si::dev::wg_clamped()` on *every* launch, so that function's
  `sycl::device::get_devices()` + `get_info<max_work_group_size>()` must stay
  cached (it is, in a function-local static).  Uncached it cost **3.2 ms per
  launch** = 210 ms of host time in one MTP verify pass (65 rmsnorm calls), and
  it was invisible in the plain decode because the decode is a recorded command
  graph (captured once at startup) while the verify is a direct replay.  The
  profile refactor introduced it; the measurement is the 3.2 ms and 210 ms above.
* **Write every field with a designated initializer.**  `.key = ...`,
  `.shape = {.rmsnorm_wg = ...}`.  The build uses `-Wall -Wextra`, so an omitted
  field warns rather than defaulting silently, and a field can be inserted or
  reordered without shifting every value after it.  Not theoretical: an earlier
  draft had `rmsnorm_wg` under `attn_vec` and every later field printed one slot
  off.
* **An unknown card must be loud.**  `for_name()` falls back to a profile whose
  key is `unknown`, a struct copy of `kDevices[0]`'s profile built once in a
  function-local static, so it cannot drift from the row it copies, and `active()`
  prints a WARNING naming where to add a
  card rather than silently inheriting the last card's tuning.  `PF_DEVICE_PROFILE`
  pins a profile regardless of the reported name (for re-measuring one card's
  curve on another), and `-DSYCL_INFER_AOT_PROFILE=<key>` bakes it into an AOT
  binary (`SI_FORCE_DEVICE_PROFILE`, which `PF_DEVICE_PROFILE` overrides);
  `PF_DEVICE_INFO=1` dumps the resolved profile and its provenance at
  startup.
* **`shape.rmsnorm_wg` is the hardware boundary to watch.**  It is one of the few places
  the two parts differ as *hardware* rather than as tuning: the A770 accepts a
  1024-thread work-group and the Iris Xe only 512, and an unconditional
  1024-thread kernel does not launch on the latter at all.  `wg_clamped()` exists
  so a card with no profile gets a smaller kernel instead of a launch failure.
  Work-group widths must be *template* parameters (a SYCL kernel cannot capture a
  runtime-initialised global), which also lets the strided load loops unroll.
* **A layer map may split across GPUs that resolve to different profiles; each
  device gets its own.**  Launchers must use `si::dev::for_queue(q)` and
  `si::dev::wg_clamped_for_queue(q, want)`, never `active()` / `wg_clamped()`:
  those resolve once per process from the *first* GPU, so a heterogeneous split
  would mis-tune the second card *and* could launch a work-group it cannot accept
  (A770 1024 threads vs Iris Xe 512) - a launch failure, not a slowdown.
  The lookup keys on `sycl::device` equality (handle compare), and the name and
  work-group limit are queried once per device, so it costs **0.039 us** per
  launch (measured, `test_dev_resolve`) against 3.2 ms for the old
  `get_devices()`-per-call version.  For the same reason, do **not** put a
  profile read behind `static const = []{ ... active() ... }()`: that pins the
  whole process to whichever card resolved first.  Use `si::dev::profile_int` /
  `profile_flag`, which cache the (device-independent) env once and read the
  default per call.  Four such statics existed and had to be converted - see
  `docs/design/12-build-and-testing.md` §3.4.1 for the list and for the A/B that
  keeps `rmsnorm`'s per-call lookup honest (the DFlash2 draft block forward had
  16.4 of its 20.5 ms inside five attn_norm launches, which is why that one was
  a static in the first place; `PF_DFLASH_SEGTIME` shows `anorm` unchanged at
  0.30 ms).  `test_dev_profile` covers the homogeneity *reporting* with synthetic
  names and no GPU; `test_dev_resolve` covers the cache itself.
* `CMakeLists.txt`'s `SYCL_INFER_AOT_DEVICE` is the ocloc target and is
  device-specific for the same reason: it must agree with the profile the binary
  will select at runtime.

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
`PF_ATTN_FUSE` (default on), `PF_DEC_SPLIT` (default
**`(warps_per_eu/2) * compute_units / n_head`** from the device profile's
`occ.warps_per_eu_x2`, rounded down to a multiple of 8 and floored at 8 — 160 on a
512-EU A770 for the 27B's 24 query heads; cap `kMaxDecSplits` = 512 — decode
K-split).  The decode attention is **occupancy** bound, not bandwidth
bound: `n_head * n_splits` 32-thread warps per layer, 8 per EU, and past 8
warps/EU the excess is not co-resident and the time steps *up*.  Measured one
layer, 27B shape, i8 KV, 64k depth, attn+combine ms: nsp=64 3.05, 128 2.18,
**160 1.70**, 168 1.74, 176 **2.68** (the cliff, at 8.25 warps/EU).  End to end
that is tg128@64k 8.26 -> 10.34 t/s and @16k 12.37 -> 13.35, with batched
decode neutral at c8.  Three consequences worth not re-deriving, all measured: do
**not** "fix" the 6x GQA K/V re-read by sharing loads across query heads (5x
*SLOWER* — the re-read is free from L2 and the lost warps are not); do not
unroll the key loop for MLP (KU=2..8 all slower; IGC already pipelines it); and
do not widen a lane to 16 dims for 16-byte loads (`attn_dec16_kernel`: 4.45 vs
1.55 ms/layer — 16 accumulators + 16 query + 16 K + 16 V floats per lane makes IGC
allocate 128 registers and spill; the report's §4.3 has two bugs in that kernel
that both measured *faster* while being wrong, so read it before trusting a
timing).  The 8-warps-per-EU wave is architectural, so no register trick buys
more warps),
`PF_DEC_GROUP` (grouped decode attention, default off), `PF_ATTN_XMX` (oneDNN
int8 *matmul* prefill attention, **default on**; `0` restores the classic kernel),
`PF_ATTN_XMX_MIN` (minimum key count for the oneDNN matmul path, default 2048 - below it the
classic kernel is faster), `PF_ATTN_WAIT` (force a per-attention-matmul oneDNN
stream wait; default off, the in-order queue already orders them).

**GEMV / GEMM tuning**
`PF_GEMV_SPLIT`, `GEMV_DEC_VEC`, `GEMV_VEC12`, `GEMV_VEC13`,
`PF_GEMM_ROW`, `PF_GEMM_TILE`, `PF_GEMM_SPLIT`, `PF_GEMM_WG`, `PF_GEMM_SG`,
`PF_GEMM_XSLM`, `PF_GEMM_ARCH`, `PF_MT_R`/`PF_MT_R2`, `PF_MT_TB`, `PF_MT_WG`,
`PF_MT_WG2`, `PF_MT_PF`, `PF_MT_SLM`.
`PF_NAT` (`0` disables `nat_gemm_launch` and falls back to the oneDNN grouped-scale
matmul + epilogue).  The two software-pipelining forms that used to sit in this list
(`PF_NATPF` / `PF_NATPIPE`) were **removed**, not just defaulted off — the shipped
schedule is the one that measured fastest, so a knob to re-enable a measured loss is
a trap.  A real rewrite would have to beat the occupancy argument in *MTP* first.

**GDN**
`PF_GDN_COLS` (state columns/warp, default 2), `PF_GDN_WG` (warps/WG, default 8),
`PF_GDN_VEC` (unset = **float4 only for `n_real < 2`**, 0 = always the scalar
kernel, 1 = always float4; the float4 state slice costs 4x the registers, so it
wins by 0.25 ms at one token per row and loses by 1.46x from two, 1.81x at 32 -
the MTP verify's `n_real = k+1` is in the losing half and gains 2.4 ms from the
scalar kernel), `PF_GDN_FUSE` (1/2),
`PF_GDN_COLS_MIN` (the `n_real` at which column batching starts, default 8 =
unchanged; the MTP verify's `n_real = k+1 = 5` is below it and every
`PF_GDN_COLS>1` variant is 1-1.7 ms *slower* there, so the gate is right).
The GDN is the MTP verify's second-largest block
(11.24 ms of 95, `PF_PROF` mode 2) and is sub-group-reduction-latency bound: one
warp owns one state row and spends two `sg_sum` per token to advance 8 MACs.

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
`PF_MM_URL_FETCH` (`1` allows remote `http(s)://` `image_url`/`video_url`/`audio_url`
parts, **default off**; base64 `data:` URLs always work),
`PF_MM_URL_ALLOW_PRIVATE` (`1` lets an enabled fetch reach loopback/private
addresses - an internal media server, and SSRF exposure again),
`PF_AV_FFMPEG` (audio/video decode CLI path, default `ffmpeg`).

**Diagnostics**
`PF_NOGRAPH` (replay kernels directly), `PF_PROF` (with `PF_NOGRAPH`),
`PF_TIME`, `PF_DBG_MID` (stop after embedding/norm/attn/ffn), `PF_DBG_GEMV`,
`PF_DBG_MT`, `PF_DBG_PFB`, `PF_GDN_DBG`, `PF_SRV_TIME`,
`PF_HOSTPROF=1` (the `PF_PROF` block breakdown *without* the per-stamp device
wait, so it measures host submission cost - the only way to tell a slow kernel
from a slow launch; this is what found the `wg_clamped` regression below),
`PF_MTP_STEPS=1` (per-depth MTP acceptance: reach[j] = the fraction of cycles
that accepted >= j drafts, which is how one tells a geometric decay from a
draft-state bug), `PF_MTP_DSTEP=1` (per-stage time of one draft step),
`PF_MTP_LAYER_EXACT=1` (the draft's MTP layer on the exact fp32 dequant GEMV -
it now needs `build_mtp_plan`'s raw device copy to work in multi-device mode, and
it prices the precision half: acc is *identical* to int8, so draft precision is
free), `PF_MTP_LAYER_W4` (**default on**: the draft's MTP-layer linears also get
a native per-32 u4 store, 2.06 -> 0.875 B/w, -1.4 ms/cycle, accuracy-neutral;
it is *added* to the int8 store because the MTP prefill at M = kMaxT still needs
int8, and only M == 1 dispatches native), `PF_MTP_LAYER_W2` / `PF_MTP_LAYER_W2_CALL`
(a 2-bit MTP-layer store: acc 2.14 -> 1.75 for -0.5 ms, **rejected** - lossy is
free at the *readout*, lossy compounds in the *recurrence*), `PF_MTP_HEAD_SPLIT`
(default **off**; 1 splits the draft's 795 MB head readout across both cards -
device 1 takes the upper half of the rows and writes into device 0's logits row,
so the argmax stays one scan on device 0.  Draft 17.2 -> 13.1 ms/cycle
deterministically and end-to-end a wash, because halving N changes the u4 GEMV's
decomposition and flips the draft's argmax on a near-tie: p1 bit-identical over
128 tokens, p0 -6 % and p2 -8 % acceptance), `PF_MTP_HEAD_SPLIT_DEBUG` (prints
the device and host argmax of the split row per step), `PF_MTP_HEAD_W2=1` /
`PF_W2_PAIR=1` / `PF_W2_INFO=1` / `PF_W2_DEBUG=1`
(the 2-bit draft head store, measured a wash: -2.6 ms/cycle for -2 % acceptance;
the per-step head time is inside `PF_MTP_DSTEP`, there is no separate head knob),
`PF_CHAT_TMPL_DEBUG` (log why a GGUF chat template fell back to the built-in
renderer), `SCHED_DEBUG`, `STOP_AFTER_LAYER`, `PF_ABL_NOATTN`, `PF_ABL_NOGDN`,
`PF_DUMP_LAYERS` (per-layer hidden-state fingerprints; `layerlast` = the last
real token slot; `PF_DUMP_RAW=<prefix>` writes the whole activation vector so a
decode run can be diffed element-wise against a prefill),
`PF_DUMP_LOGITS=<path>` (full logit vector, **tests only** - `test_w4_topk` /
`test_w4_vs_cpuref`) / `PF_DUMP_DEC_LOGITS=<path>` (sampler logits, per step),
`PF_DUMP_PROMPT` (the exact ids - and, for a chat prompt, the rendered
text - the model is conditioned on), `PF_DUMP_GEN` (the decode loop's sampled id
and stop decisions), `PF_ROWACT` (restore the now-unused per-row activation
quantizer for A/B; it is dead because oneDNN reads the per-32-group form, and
cost ~13 ms/token in a 27B multi-device decode), `PF_LAUNCHCNT` (tally one `record_forward` pass's SYCL submissions by kind and
count which are *adjacent-fusable*; the verify is 812 - 257 xq, 395 GEMV segments,
64 rmsnorm, 96 GDN-family - of which only **75 (9 %) are fusable**, because every
xq reads a different source and so sits on the critical dependency chain),
`PF_MTP_SUBMIT` (the
verify's host submit time vs its wall - 86.6 of 90.2 ms, i.e. the in-order queue
paces the host, so a recorded verify graph would only buy the decode graph's
measured 1-2 ms), `PF_XMX_BREAKDOWN` (per-stage wall clock inside the prefill
attention block loop - gather / QK / softmax / PV; the only way to attribute the
*mode-2* 512-token prefill, because `PF_PROF` needs `PF_NOGRAPH` and that forces
the mode-1 32-token path; every stage is measured with a `q.wait()`, so the
absolute values are inflated by the serialisation and only the shares are
meaningful), `PF_XMX_TIME` (whole-call wall clock, same caveat),
`PF_XMX_DBG` (print `M`, `max_nkv` and the per-block widths of one call),
`PF_XMX_GRED=N` (gather stage-1 work-group count, default 64), `PF_XMX_TAILBLK=0`
(keep the last block at the full `kBlk`), `PF_DUMP_KV=<dir>` (append the
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

**Internal A/B knobs** (same env-gated rule, one line each because they are
diagnostics rather than tuning): `PF_W4_INFO` (why each u4 weight was rejected or
registered, with its B/weight), `PF_W4_DEBUG` (which store each GEMV segment
takes), `PF_W4_NOCORR` / `PF_I8_NOGEMV` / `PF_W4_NOGEMV` / `PF_ABL_NOGEMV` (drop
one branch of the oneDNN int8 path, to bisect which half is wrong), `PF_DEC_R`
(dp4a decode GEMV rows per work-group: 1/2/4/8), `PF_ATTN_FLASH` (unfused tiled
prefill attention; `PF_ATTN_DBG` traces it), `PF_DUMP_SEGS=<layer>` (per-segment
hidden fingerprints inside one layer, `-1` = all), `PF_CPU_DBG` (trace the host
rmsnorm), `PF_AV_FFPROBE` (the ffprobe binary video decode uses; `PF_AV_FFMPEG`
is ffmpeg's), `PF_PF_PIPE=0` (disable the 3-phase multi-device prefill pipeline),
`PF_NO_PFB_PARTIAL` (mode-2 batches must be multiples of `kMaxT` again - the
pre-partial-batch behaviour, for A/B).  The MTP debug dumps follow the same
pattern and are one-off: `PF_MTP_DIAG`, `PF_MTP_VPROBE`, `PF_MTP_HPROBE`,
`PF_MTP_HVEC`, `PF_MTP_MEM`, `PF_MTP_HDUMPS`, `PF_MTP_HOSTCMP`, `PF_MTP_INFOCHK`,
`PF_MTP_SNAP1`/`PF_MTP_SNAPCHK`/`PF_MTP_SNAPDUMP`,
`PF_MTP_STATECHK`/`PF_MTP_STATEDUMP`/`PF_MTP_STATESEQ`,
`PF_MTP_EXACT_CALL`/`PF_MTP_EXACT_DEBUG`, `PF_MTP_AUTOTEST`/`PF_MTP_DRAFTTEST`,
`PF_MTP_LAYER_W4_CALL=<ci>` (put only call `ci`'s MTP-layer tensors on the u4
grid — 0 qkv, 1 wo, 2 ffn gate/up, 3 ffn down, -1 eh_proj; the bisection knob
that localised the `do_split` bug).

**DFlash2 diagnostics** (all default off; see the DFlash2 section above for what
each one is compared against)
`PF_DFLASH_DEBUG` (master switch for the probes below), `PF_DFLASH_VFCHK` (is the
verify using `mtp_verify`'s recorded command graphs, and with how many rows and
graphs - the direct replay cost 210 ms on the MTP path, so this is the first
question), `PF_DFLASH_TOPTIME` (device-side split of the draft's readout tail into
topk / selector-hidden / selector, against the host wall), `PF_DFLASH_DTTAIL` (the
same tail host-side: block vs blocking copy vs the lattice walk),
`PF_DFLASH_SLICES` (top-k slices per row; default 8, the measured optimum, and
*not* the occupancy-maximising value), `PF_DFLASH_SEGTIME` (device-side
segment timing, all 5 layers: `anorm/acproj/aconv0/attn | ffn_norm+cproj+cconv /
gate_up / down+cconv`, plus `sum` against `devspan` and the host wall - the last
of which is the check that makes the rest trustworthy; needs the queue's
enable_profiling, which the engine now sets),
`PF_DFLASH_TIME` (per-phase
cycle ms), `PF_DFLASH_NMAX` (draft length; default 5), `PF_DFLASH_SIG` /
`PF_DFLASH_BIN` / `PF_DFLASH_SIGALL` (per-position fingerprints and full-tensor dumps
for the element-wise diff against llama.cpp; filenames carry layer *and* block, see
the note above), `PF_DFLASH_BTIME` (draft block phase split),
`PF_DFLASH_CONVCHK` / `PF_DFLASH_CONVCHKALL`, `PF_DFLASH_PROJCHK` /
`PF_DFLASH_PROJCHKALL`, `PF_DFLASH_BASECHK` (the conv base tensors are byte-identical
device-side), `PF_DFLASH_GEMMCHK`, `PF_DFLASH_ROWRMS`, `PF_DFLASH_EMBCHK`,
`PF_DFLASH_INJP`, `PF_DFLASH_INJTRACE` (ring V readback + coverage),
`PF_DFLASH_ATTNCHK`, `PF_DFLASH_LAYER_W4` (int8 instead of the native u4 store for
the draft layers - an A/B, costs 2 %), `PF_DFLASH_ROWED` (one row at a time),
`PF_DFLASH_SPLITQ` (split activation quantisation - **this one is load-bearing**,
`0` costs 28.0 -> 43.0 ms), `PF_DFLASH_NOCOMMIT` (drop the injected keys from the
draft's attention), `PF_DFLASH_NROT`, `PF_DFLASH_HEADCHK` (exact fp32 reference for
the draft head's logits).

**Weight representation**
`PF_W4` (native u4 for Q4_K, default on), `PF_CB4` (store IQ4_XS/IQ4_NL as native 4-bit codebook indices + a per-32 f16
scale, 0.5625 B/weight and lossless, decoded by a LUT-expanding GEMV; prefill
expands each tensor to int8 in a reused scratch),
`PF_W4_K5` (`1` puts **only** Q5_K onto the per-32 4-bit grid instead of the
native 5-bit store: 0.625 B/weight against k5's 0.75, and Q5_K is 30 % of the 27B
weight pass, so the device weights go 16.90 -> 16.06 GiB and the MTP verify
91.5 -> 86.3 ms (-5.7 %, deterministic over two repeats); **default off**,
because `test_w4`'s round-trip rel L2 goes 0.077 % -> **4.60 % mean / 10.83 %
worst**, the same error class as `PF_W4_ALL`, and a 160-token greedy generation
stays coherent but diverges at word 37 - an accuracy decision, not an engine fix),
`PF_K5` (store Q5_K as the native 5-bit grid: 4-bit nibble plane + 1-bit plane
+ per-(g,n) f16 step/offset, 0.75 B/weight and lossless, recombined with one OR
in the decode GEMV; prefill expands q5 to int8 in the shared scratch and runs
the grouped-scale int8 primitive with the u4 offset-correction epilogue - i.e.
it costs ~1.6 B/weight of *serial* prefill traffic per pass, measured -15..-20%
pp512 for +6% tg128 and -2.1 GB/card),
`PF_W4_ALL` (`1` re-quantizes every
other type onto the same 4-bit grid: 16.0 vs 24.5 GB read per 27B decode token,
measured tg128 12.6 -> 16.2 t/s but -20% prefill and ~8x weight error vs fp32),
`PF_SI4` (SIn 4-bit), `PF_META`.

## Invariants and gotchas

* **`compute_backend::synchronize()` must actually wait, and the CPU backend's used
  to be an empty override — which was a heap use-after-free, not a slow path.**
  `cpu_backend.cpp` is compiled with `-fno-sycl` (it is in `SI_CPU_SOURCES`, outside
  the device pass), so it has no queue to wait on and the override compiled fine
  while doing nothing.  `engine::sync_all()` has ~24 call sites, so on a CPU
  partition **every one of them was a no-op**.  The prefix cache then handed a
  temporary `std::vector` to an asynchronous `queue::memcpy` and returned — the
  vector was freed while the copy was still in flight.  It surfaced only as
  `test_pc_cpu` failing, and the diagnostic trail is worth remembering: the KV
  comparison passed (36/36 blocks identical) while the *logits* differed, because
  the corruption was in a later host→device copy, not the KV one.  The fix is a
  waiter handed in at construction — single device `[this]{ q.wait(); }`,
  multi-device `[this, i]{ dev_queue(i).wait(); }`, and **`dev_queues_` cannot be
  used directly for the CPU index because a CPU partition's entry is null and it
  segfaults**.  **Generalisation: an override that exists only because the base
  class declares it is a silent no-op waiting to happen — if a virtual has no effect
  in one backend, assert it, do not leave it empty.**  A narrow test will not find
  this class of bug: `test_pc_cpu` originally compared only three layer-0 K blocks,
  and widening it to 36 units across all block-offset phases is what finally
  localised the fault.  See `docs/design/06-prefix-cache.md`.

* **A heterogeneous `--layer-map` is supported, not an error.**  A layer map may span
  GPUs that resolve to different `si::dev::profile`s (e.g. A770 + Iris Xe).  Each
  device gets its own profile resolved per queue, and the engine reports the mix at
  startup under `PF_DEVICE_INFO=1` rather than refusing.  This matters because
  launchers must use `si::dev::for_queue(q)` and `wg_clamped_for_queue(q, want)` — a
  single process-wide `active()` would mis-tune the second card *and* could launch a
  work-group it cannot accept (A770 accepts 1024 threads, Iris Xe only 512), which
  is a launch failure, not a slowdown.  The lookup keys on `sycl::device` equality
  and costs 0.039 us per launch.

* **"The projector loaded" is not "the projector can run".**  `vision_model` (and the
  audio tower) will happily load a GGUF whose geometry the kernels do not implement
  — the remote mmproj files all have `head_dim 72` where only 64 is implemented, or a
  `proj_dim` that does not match the text width — and then abort on the first
  forward, *after* the model is loaded and after 23 checks have already passed.
  `vision_model::runnable()` exists so a caller can ask that question up front and
  report "loads but cannot run" instead.  **If you add a geometry constraint, add it
  to `runnable()` as well as to the forward's check**, or the distinction is
  invisible.

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
* **"XMX" here is a name for oneDNN's int8 matmul, not the XMX unit.**
  `attn_xmx.cpp` / `PF_ATTN_XMX` / `PF_ATTN_XMX_MIN` all date back to "use
  oneDNN's int8 matmul for prefill attention"; the unit was never involved.
  Measured: `ONEDNN_VERBOSE=2` names the impl `jit:gemm:any` for every matmul
  the engine runs (a mode-2 prefill is 1154 of them, attention included), and in
  this oneDNN 3.11.4 (`libdnnl.so.3.11`, Level Zero, DG2) `strings | grep -c
  xmx` is 0 - all 498 `dpas` hits are `DUMMY_DPAS_*` JIT-generator placeholders,
  so **no XMX GEMM kernel is compiled in at all**.  Do not read a win from
  `PF_ATTN_XMX` as evidence about the XMX unit.  The consequence: routing the
  *verify* onto
  the oneDNN int8 primitive (`PF_NAT=0 PF_MTP_FORCE_INT8=1`) measures 2.05x
  *slower* (verify 91.6 -> 188.7 ms, 158 GB/s over 24.5 GB of int8 weights vs
  the native stores' 272-325).
* **oneDNN int8 prefill attention (`attn_xmx.cpp`, `PF_ATTN_XMX`, default on above 2048
  keys)**: QK^T and PV run as plain oneDNN int8 matmuls over the paged KV
  gathered into a contiguous scratch.  Four things are load-bearing:
  (1) a plain int8 matmul sums over k, so a per-key scale cannot be applied
  after it - K and V each carry ONE block-wide scale, folded in at gather time;
  (2) the query tile stacks every HPG query head sharing a kv head, so the
  matmul width is `HPG * tokens` and the KV is read once per kv head, not once
  per query head; (3) the matmuls always run at a FIXED width `kBlk = 8192`
  (zero-padded), because oneDNN primitive creation is ~15 ms per new shape and
  a per-chunk `N` made prims never reusable - that alone was a 12x regression;
  (4) **the block-max reduction in the gather must be spread over many
  work-groups, never one** - a single 256-thread group reducing a whole 8192-key
  block was 18.1 ms of the 37.8 ms one attention layer costs at 64k depth
  (the QK matmul 6.7, the softmax 6.1, the PV matmul 4.3), i.e. 23 % of the
  whole 512-token prefill chunk.  It is now `kGatherRed = 64` work-groups
  reducing a slice each into a private slot, then one small kernel folds the
  partials; `fmax` is exact and associative so the block scale, and the whole
  output, is bit-identical.  Measured: 18.1 -> 9.1 ms, marginal 512-token
  chunk at 64k **975 -> 833 ms (-15 %)**.  `PF_XMX_BREAKDOWN=1` prints the
  per-stage split and is the only way to attribute the mode-2 512-token
  prefill (`PF_PROF` needs `PF_NOGRAPH`, which forces the mode-1 32-token
  path); `PF_XMX_GRED` / `PF_XMX_TAILBLK` are the A/B knobs.  Note two
  *negative* results here, all worth not re-deriving: tiling M so the 101 MB
  score matrix fits the 16 MB L2 is +53 % (L2 caps the tile *area*, so the
  oneDNN execute count goes up ~7x either way at ~35 us each); the per-block
  padding the fixed width leaves behind is free (trimmed anyway, measured neutral
  at 16k and 64k); and computing the block max from the **scale plane alone** —
  exact, since every non-zero 32-wide group quantizes its extreme to |k| = 127, so
  max|k*scale| == 127 * max(scale[]) — cuts that pass's traffic 16x and measures
  **neutral** (833 -> 830 ms), because the gather is bound by the page-table
  lookup latency, not by bandwidth.  That is why parallelising it (§3.2, 15%) beat
  shrinking it.  The identity needs an all-zero group to store scale 0 rather
  than 1 (or it is wrong by up to 127x) plus an `i8_quant` zero-scale guard
  (0/0 is NaN); both are reverted, since a KV-format change is not worth 0.3 %.
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
* **`st_a` is the oneDNN-int8-matmul-only stream.**  The dense `PF_GEMM_DNNL` GEMMs still use
  `p->st`; `attn_qk`/`attn_pv` submit to `p->st_a` and wait only that stream.
  Do not move the dense path onto `st_a`.
* **`ggml_float_to_half` used to flush the whole f16 subnormal range to zero,
  which silently destroyed weight-store scales (`src/common/quant.h`).**  The
  f16 subnormals represent `[2^-24, 2^-14)` with ten mantissa bits; the old code
  had `if (exp <= 0) return sign;`, i.e. it threw away a thousand-fold range.  A
  Q4_K super-block whose `d` is *itself* subnormal then produced `d*sc` below
  `2^-14`, so `pack_q4_K` wrote a **step plane of identically zero** and the u4
  reconstruction of that column collapsed onto the constant `-dmin*m`.
  Measured on the 27B: `blk.16.attn_qkv.weight` has `d = 4.47e-6` (f16 `0x004b`)
  in the offending super-block, and **21 of its 10240 columns (0.205%) sat at up
  to 240% relative L2** against the exact GGUF dequant while every neighbouring
  column measured the expected 0.08% (the f16 rounding of the two constants).
  `blk.1`/`blk.2`/`blk.4` of the same tensor family had **zero** such columns, so
  it is data-dependent and invisible in any aggregate.  The same flush reached
  `w8.cpp`'s int8 per-group scales, the k5/cb4 and 2-bit packs, and the KV
  scales.  `test_f16conv` (hermetic, no model) pins the bit patterns; the old
  implementation fails it 68 ways, and it also asserts the *range* property
  ("nothing in `[2^-24, 2^-14)` may encode as zero") rather than sampled points,
  because a sampled test misses a boundary that moves.  **Lesson: the converter
  kept truncating (not round-to-nearest) so that nothing on the normal range
  moved - a correctness fix here did not need to be a numerics change.**
* **A test criterion built on a per-element *relative* yardstick flags
  cancellation, not defects.**  A dot product of K=17408 terms has an absolute
  error set by the terms (`|dy| ~ eps * sum_k |w_k x_k|`), so an output that
  nearly cancels to a small fraction of the tensor's max still differs between
  two *correct* weight stores by their full ~1%.  `test_w4_vs_i8`'s original
  `|dy| > 0.05 * max(1, |y|)` flagged 1 element in 163840 on one tensor and 2 in
  163840 on another while the tensor-level figure stayed at 1.6% / 0.9%.  It now
  screens against the *tensor's* output scale and adjudicates the survivors (and
  each tensor's worst element) against the exact fp32 dot product of the same
  quantized activation, judged on `|dy|` vs `sum_k |w_k x_k|`.  That criterion is
  what found the `ggml_float_to_half` defect above, so the rewrite is load-bearing
  rather than cosmetic - and `PF_W4_NOCORR=1` (drop the u4 zero-point
  correction) is its red half: `diff/accum` 0.10-0.15 against ~0.01 for a correct
  pair of stores, so the 0.05 threshold is not a loosened bar.
* **`nat_gemm_launch`'s int8 (FMT 3) and cb4 (FMT 2) paths need the
  decode/verify activation form, not just "an activation".**  `gemm()`'s int8
  nat call is gated on the activation's split form (like the u4 one): a
  mode-1/mode-2 prefill quantizes with `do_split=false`, so the grouped views it
  would read are the *previous* call's.  Missing that gate silently corrupted
  the 0.8B multi-device decode-vs-prefill (`gen0`/`gen1`/`ref` 220/220/567
  against 248068/271/271) while every single-device test passed, and it also
  made `PF_MD_SPLITS=1` look broken.  `PF_NAT=0` is the bisection knob; the u4
  path (FMT 0) already had the gate.  **This class of bug can no longer be
  written**: `quantize()` returns a `si::act_view` (`{m, k, split, gen}`) and
  every GEMM entry point *requires* it, so "forgot the gate" is a signature that
  does not compile, and a view that went stale is one integer compare
  (`view_is_current`) rather than a re-derived `M`/`K` check.  `split` is a
  field of the view, so a GEMM can no longer read the even/odd planes off the
  object and be wrong about whether they are there.  A stale view returns
  `nullptr` from the `act_*` accessors (they used to hand back whatever was in
  the scratch), so a caller that forgets to re-quantize now faults loudly instead
  of reading the previous call's data - that is how the one real instance of
  this shape was found, in `test_w4_gemm`.
* **The two drafters share exactly one piece of device state, and it is allocated
  as a pair.**  `engine::verify_scratch` owns the verify cycle's argmax staging
  (`d_argmax` + `d_argval`); every other drafter buffer is MTP-only or lives in
  the nested `dflash_state`.  The pair matters because MTP's candidate readout
  (`PF_MTP_CAND`) needs the *value* side and decides that from its own
  `mtp_cand_cap_` - so a value side that could be absent while the id side
  existed would let it write through a null pointer.  Each drafter used to carry
  its own `if (!d_argmax)` block and **only MTP's allocated both**, which is a bug
  rather than a style note: had DFlash run first in a process that then ran MTP
  with `PF_MTP_CAND=1`, the write would have gone through null.  `test_spec`
  never caught it because it deliberately runs **one drafter per process** (a
  second engine in the same process inherits state from the first).  Call
  `mtp.verify.ensure(q)`; do not allocate these buffers at a call site.
* **The MTP draft's attention split is `PF_MTP_SPLITS`, independent of
  `PF_MD_SPLITS`.**  The `--layer-map` path pins `n_splits`/`dec_splits` to 1
  (its split path is opt-in, and with the nat bug above fixed it verifies clean
  again), so the draft derives `ceil((pos0+n)/512)` clamped to `PF_MTP_SPLITS`
  and uses its own partials buffer - see the MTP section.
* **The scheduler used to serialise every request; releasing the sequence
  mutex fixes it (measured).**  `scheduler::loop` held the sequence mutex `m`
  across every engine call (`prefill_batch` ~300 ms, `decode_batch` ~65 ms) and
  immediately re-locked it, so `scheduler::submit()` was starved: each request
  was admitted only after the previous generation had *finished*, and both
  cross-sequence batching paths were dead code (8 concurrent requests finished
  9, 18, 27, ... 71 s apart, aggregate 14.5 tok/s = exactly one request's rate,
  with 188/188 decode passes at `nb=1 active=1`).  The engine calls now run with
  `m` **released** (a `std::unique_lock`; `e.mtx` still serialises the engine),
  which lets admission and the pre-existing multi-row decode batch.  Measured on
  the 27B (temperature 0, `ignore_eos`, 128 tokens each): N=1/2/4/8 -> 14.3 /
  24.5 / 40.1 / **55.0 tok/s**, i.e. **3.85x aggregate at N=8**.
  **Concurrency costs bit-exactness**: the same temperature-0 prompt fanned out
  to N concurrent requests returns coherent, *deterministic* text that can
  differ from the single-request result - the flip happens at a near-tie
  argmax, once a sequence joins a batch (measured: 1 of N rows is byte-exact,
  the rest flip at the same "attention mechanism" / "**Scaled Dot-Product
  Attention**" tie ~20 tokens in; the multiset of outputs is identical across
  repeated runs, so it is accumulation-order numerics, not a race).  A sequence
  admitted first and decoded alone keeps the exact early tokens; batching only
  changes the *order* of the fp accumulation in the batched GEMMs.  That also
  means a test must **not** assert two concurrent greedy runs of one prompt agree:
  `test_sched` checks they both finish and honour `max_tokens`, nothing more.
* **The scheduler's token-output path had no coverage at all until `test_sched`.**
  `test_spec` drives `engine::generate`, which never enters `scheduler::loop`, so
  the one place that turns a sampled token into a streamed piece - duplicated
  between the head prefill's first token and every decode row - was untested, and
  a refactor of it had no oracle.  `test_sched` (model-backed, no HTTP) covers:
  a greedy run, the logprobs variant (pieces carry an id *and* a logprob),
  `max_tokens`, an `ignore_eos` run that must report `finish_reason == "length"`,
  cancellation via `sequence::cancel` (the SSE disconnect path), an empty prompt
  (must retire rather than wedge), and two concurrent sequences.
  It earned its keep immediately: factoring the duplication into
  `sample_and_emit` initially returned one bool meaning both "was EOS" and
  "should retire", and the decode path's `&& !eos` reason test then reported
  `"stop"` where it must report `"length"` - a user-visible `finish_reason`
  regression.  The helper returns a `sample_outcome { eos, done }` for exactly
  that reason.
* **Cross-sequence prefill batching is NOT enabled** (it corrupts).  Packing
  several prompts into one mode-2 forward (row-major over the concatenated
  tokens, per-row `slot`/`pos`/`n_real_row`, one `prefill_text` per batch) was
  implemented and produced *garbage* - degenerate continuations, not plausible
  alternatives: `1000000000...`, `| 10 | 10 | 10 |` - even with the decode batch
  capped at 1, and `step_info` already carries per-row `slot`/`pos`/`active`, so
  the fault is in something reading plan-time state rather than `info`.
  Prefill therefore stays one sequence at a time (each ~300 ms); the 3.85x above
  is decode batching alone.  Fixing it is worth doing - at N=8 the serialised
  prefills are ~2.4 s of the 18.6 s wall - but it needs the same
  "N concurrent identical requests must return byte-identical output" harness,
  run with the decode batch capped at 1 to isolate it.
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
* **The HTTP API has no authentication, so the bind address and the remote-media
  fetch are the security boundary.**  Both default to closed:
  `--host` is `127.0.0.1` (pass `0.0.0.0` to expose, behind a reverse proxy),
  and `image_url`/`video_url`/`audio_url` `http(s)://` fetches need
  `PF_MM_URL_FETCH=1`.  With it on, `src/server/url_fetch.cpp` requires every
  address the hostname resolves to to be publicly routable
  (`PF_MM_URL_ALLOW_PRIVATE=1` lifts that for an internal media server) and
  walks redirects **by hand** - httplib's `set_follow_location` re-resolves and
  re-connects without a hook, so a validated URL becomes an arbitrary one,
  which is the usual bypass.  Two traps the test pins: resolve rather than
  string-match (decimal `2130706433` and octal `0177.0.0.1` are loopback), and
  classify an IPv4-mapped IPv6 by its embedded address (`::ffff:127.0.0.1`).
  Residual and *not* claimed fixed: the address is checked once and httplib
  resolves again to connect, so a hostile DNS server can still win that race.
* **httplib signals a client disconnect in exactly one place, and it is easy to
  miss.**  `ContentProviderResourceReleaser`'s `bool` is false both when a write
  fails and when the peer hung up - `content_provider_success_` is set only if
  `write_content_with_provider` returned true, and that loop's
  `is_peer_alive()` pre-check is what catches a client that vanished *between*
  two items (no `sink.write` ever returns false in that case).  A streaming
  handler that ignores it (and `sink.write`'s own return) runs the whole
  generation against a dead socket while the worker thread sits in the
  releaser's `join()` - and **the emitted text is identical either way**, so no
  generation test can catch it.  `serve_sse_chunked` (`src/server/sse.h`) is the
  production wiring and `test_sse_cancel` drives it over a real socket with a
  TCP RST; the cancel has to reach the producers too, because the engine
  callback's return value is the only cancellation point inside a forward, so
  a cancelled generation still finishes the decode step it is in (~65 ms).
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
./build/test_forward                    # default prompt predicted_argmax=248068
./build/test_decode_vs_prefill          # decode == re-prefill (any single-token change)
clang-tidy -p build --extra-arg=-I/opt/intel/oneapi/compiler/2026.1/include -checks='-*,misc-include-cleaner' <changed files>
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
cost breakdown and the `STOP_AFTER_LAYER` sweep are in `dev/`, with the
decode-GEMV microbenchmarks next to them.

If a build seems to ignore your edit, remember `rsync -a` preserves source
mtimes: `find src tests -name '*.cpp' -o -name '*.h' | xargs touch` first.

See [`docs/design/12-build-and-testing.md`](docs/design/12-build-and-testing.md)
for the full build/test matrix and the `docs/` index at
[`docs/README.md`](docs/README.md).
