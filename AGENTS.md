# AGENTS.md

Guidance for AI coding agents working on **sycl-infer**.

This file has three jobs and they are in this order:

1. **The reference**: what the engine is, how to build it, how to test it, how to
   extend it, the conventions, the environment variables.
2. **The measurement archive**: the numbers that a future change must be read
   against, each with the condition it was measured under.  A measurement that
   only exists in a write-up has to be summarised here before it can be relied
   on — and the *conclusion* is what is kept here, not the hunt that produced
   it.
3. **The invariants**: the rules that are not obvious from the code, and what
   breaks when they are broken.

**`reports/` is not part of this repository.**  It is in `.gitignore` and stays
that way: nothing under it is ever committed, so **no document here may link to
it** — not this file, not `README.md`, not `docs/`.  A `reports/` reference in a
committed doc is a dead link for everyone who clones the tree.

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
built the way they are; this file is the quick-reference for conventions,
measurements and gotchas.

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
* [`docs/design/13-audio-video.md`](docs/design/13-audio-video.md) — audio/video
  decode, the AuT audio tower, video/audio M-RoPE, mixed-prompt assembly.
* [`docs/design/14-dflash2.md`](docs/design/14-dflash2.md) — the DFlash2 block
  drafter.
* [`docs/design/15-moe-ncmoe.md`](docs/design/15-moe-ncmoe.md) — the MoE
  architecture, the routed experts' GEMM, the ncmoe split and the three-tier
  expert cache.
* [`docs/design/16-router.md`](docs/design/16-router.md) — the multi-model
  router: one worker process per model, a JSON config, the JSON-RPC control
  plane, the HTTP/SSE data-plane proxy, per-model VRAM budget admission, remote
  workers and Docker deployment.
* [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — **the measured throughput
  table**: prefill and decode by context depth and batch size, per model and
  per speculative path, each with the hardware, the build, the harness and the
  command that reproduces it.  A performance number belongs there (see *Measured
  results* below).

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
* **Watch the memory when you build.**  The device pass is the expensive one:
  a Debug build's `-g` makes each device image several times larger, and 20
  parallel `icpx` invocations of a SYCL TU is a multi-GB spike per process.  On a
  30 GB box, `-j$(nproc)` on a Debug SYCL build gets OOM-killed; use `-j4` and
  check `free -g` between targets.  A killed build looks like a compile error
  with no message, so a silent failure is usually the OOM killer, not the code.

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

The **router** (`sycl-infer-router`, a separate binary that needs neither
SYCL nor oneDNN) serves several models behind one endpoint; it starts one
worker process per model and proxies the OpenAI API to them:

```bash
cmake -S . -B build-router-only -DSI_ROUTER_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-router-only -j
./build-router-only/sycl-infer-router --config router.json --check   # validate
./build-router-only/sycl-infer-router --config router.json          # serve
```

Its config is JSON **with comments**; see
[docs/design/16-router.md](docs/design/16-router.md).

`main.cpp` documents the flags (`--model --ctx --blocks --kv-cap-mb --kv-type
--port --host --device --cpu-threads --layer-map --mmproj --mtp --mtp-device
--audio-mmproj`, the three prefix-cache budgets `--pc-vram-mb`/`--pc-mem-mb`
(alias) `--pc-ram-mb`/`--pc-dir`+`--pc-disk-mb`, the control plane
`--control-port`/`--control-host`/`--control-token`/`--control-token-file`, and
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

`--max-slots N` bounds the concurrent sequences, which is what sizes the
**recurrent** state; see *Long context: what actually fits* below for why that
is the cheapest context headroom there is (1.125 GiB per card at the default 16).
**It is not the engine's row ceiling**: `kMaxB` (32) is the hard ceiling and is
compile-time, `--max-slots` (default 16) is the policy.  Raising the ceiling must
not raise the default, or the recurrent state doubles.

`--yarn` enables YaRN context extension (NTK-by-parts, `src/common/yarn.h`) and
`--yarn-factor F` / `--yarn-orig-ctx N` / `--yarn-beta-fast F` /
`--yarn-beta-slow S` / `--yarn-attn-factor A` override what it derives.
**Default off**, and a factor <= 1 prints one `[yarn]` line and disables itself
rather than running with a no-op table.  With no `--yarn-factor` the factor is
`--ctx / <arch>.context_length`, so `--ctx 1048576 --yarn` on a 32k-trained
model is factor 32.  The engine prints what the ramp actually did:

```
[yarn] on: factor=32.000 (max_seq=327680, trained ctx=262144) base=10000000 n_rot=64 \
        beta=32/1 mscale=1.1505 pairs: 10 stretched, 7 ramped, 15 untouched
```

(verbatim from the 27B; the three buckets come from the analytic ramp mask, not a
magnitude threshold, so they do **not** change with the factor - only the depth
of the stretch does.  15 untouched of 32 pairs is `yarn_correction_dim(32, 64,
1e7, 262144) = 14.24` rounded up, i.e. every pair whose period is shorter than
the trained window keeps the trained frequency *bit for bit*.)

Two caveats that are not optional.  **YaRN is a scaling method**: the paper's
gains come from a small amount of long-context fine-tuning on top, so used
zero-shot it extends the window but costs quality relative to the trained one -
which is why the flag is opt-in and not folded into `--ctx`.  And it does **not**
save memory: 1M tokens still need their KV resident (see the capacity table), so
`--yarn` and `--max-slots`/`--kv-type` are independent levers.

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
                           # concurrent sequences, and a twenty-row batched
                           # decode whose oracle is "every row's logits moved
                           # between two steps" (model-backed, no HTTP)
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

**Five of these are mandatory after touching a kernel or the engine**:
`test_gpu_stages`, `test_gpu_vs_ref`, `test_forward`, `test_decode_vs_prefill`
and `test_sched`.  The first three are prefill-only and are blind to a
decode-only bug (that is how the fixed head-segment binding bug stayed
invisible), so **any change that can affect the single-token path must run
`test_decode_vs_prefill`**; and `test_sched` is the only coverage of a
**batched** decode — a single-token check cannot see "the engine computed only
some of the rows".

## Repository layout

```
src/common/     quant.h (ggml block formats + host dequant), w8.{h,cpp} (SIn
                int8 weight format/repack), w4.{h,cpp} (native-width u4 packing
                for Q4_K: u4 plane + separate f16 scale/off planes; and the
                codebook 4-bit store for IQ4_XS/IQ4_NL: 4-bit index plane in the
                interleaved nibble order + per-32 f16 scale, `cb4t`/`cb4_pack`),
                dp4a.h (portable dp4a helper),
                cpu_isa.{h,cpp} (host CPU feature detection + ISA dispatch),
                jsonrpc.h (the control plane's wire protocol: JSON-RPC 2.0
                over TCP, NDJSON framing; shared by the worker's server and
                the router's client),
                exit_codes.h (the worker's exit codes: 78 = a startup
                failure, do not retry)
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
                i8 (integer GEMV straight from the GGUF blocks), moe (the host
                half of a routed MoE layer)
src/backend/gpu/kernels/    kernels.h (public launch API + step_info/gemv_seg), kernel_utils.h
                (shared device helpers in namespace si::kd), kv_type.{h,cpp},
                and one .cpp per kernel: rmsnorm, embed, copy_row, gemv,
                qk_norm_rope, attn, conv, gdn, gated_norm, xq, dp4a_gemv,
                dp4a_gemm (+ dp4a_common for the shared split-K workspace),
                 w4_gemv (every native weight store's GEMV: u4/k5/codebook/2-bit
                 g-major SLM-staged decode GEMV, the LUT-expanding codebook GEMV
                 and its prefill expansion, the batched nat_gemm_launch (M <= 13),
                 the opt-in batched u4 GEMM `w4_gemm_launch`, and the fused narrow
                 int8 call group `i8_grp_gemv_rows_multi_launch`),
                 moe (the routed experts: the router, the CSR, the grouped dp4a
                 expert GEMM and its fp32 fallback, the reduce), dflash (the
                 DFlash2 block drafter's kernels), mtp (mtp_concat /
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
                verify/rollback + the speculative loop), engine_dflash.cpp (the
                DFlash2 drafter), engine_graph.cpp (seg_plan,
                record_forward, build_graphs), engine_kvpool.cpp (dynamic KV
                pool), engine_prefix_cache.cpp (VRAM tier + tier demotion),
                moe_cache.{h,cpp} (the three-tier expert cache),
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
                test_sched.cpp (the scheduler's token-stream path, and the
                batched-decode row coverage),
                test_spec.cpp (MTP + DFlash2 stream == the plain greedy decode;
                one engine per process, re-execing itself per configuration),
                test_w4_vs_cpuref.cpp / test_w4_topk.cpp (u4 logits vs the fp32
                reference; TEST_LAYER_MAP runs the 27B split across devices),
                test_27b_prefill.cpp (27B prefill bring-up probe)
tests/mm/       test_multimodal.cpp (preprocessing, vision encoder, video/audio
                prompts, audio kernels, positions)
src/router/     the multi-model front end (a SEPARATE binary, `sycl-infer-router`:
                config.cpp (JSON with comments), registry.cpp (the model table and
                its state machine), rpc_client.cpp (the control-plane client),
                supervisor.cpp (spawn/monitor/restart a worker per model, the
                VRAM ledger), proxy.cpp (the OpenAI HTTP/SSE proxy)).  It is
                built with `-fno-sycl` and links neither SYCL nor oneDNN, so it
                needs no GPU - see docs/design/16-router.md
src/common/     ... quant.h/w8/w4/dp4a/cpu_isa as above, plus jsonrpc.h (the
                control plane's wire protocol: JSON-RPC 2.0 over TCP, NDJSON
                framing; shared by the worker's server and the router's
                client so the two cannot drift) and exit_codes.h (78 = a
                startup failure)
src/server/     ... control.cpp: the worker's control plane (`--control-port`),
                the channel the router uses for the model's identity, its device
                memory and a graceful stop
tests/router/   test_router_config.cpp (the config parser + the validation
                rules), test_router_proxy.cpp (routing, the error shapes, the SSE
                passthrough, the cancel chain across the extra hop),
                test_router_supervisor.cpp (spawn/ready/restart, the per-device
                start gate), test_router_control.cpp (the control-plane client,
                the empirical VRAM admission).  All four are hermetic: no GPU,
                no model
docker/          Dockerfile.router (the router alone, built with
                -DSI_ROUTER_ONLY=ON so it needs no oneAPI toolchain),
                Dockerfile.worker, Dockerfile.all (both binaries, for the
                "the router starts the workers" topology), compose.yaml,
                router.example.json
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

### Long context: what actually fits on 2x A770

Everything below is measured, not derived, and the measurements are on the
`bench-host` box (`ssh bench-host`, i9-10940X, 247 GB RAM, two Arc
A770, driver 12.55.8).  The card's usable VRAM is **15.11 GiB**, not the 16 GB on
the box (16 GB = 14.90 GiB; `sycl::info::device::global_mem_size` reports
15.11 GiB, so the 15.11 figure is the one to budget against).  `lspci -vv` needs
root there, so link width is inferred from measured bandwidth instead (below).

Startup weight report, `--layer-map 0-31:gpu.0,32-63:gpu.1`:

```
[dev] device 0 weights: 59 u4, 43 k5, 69 codebook, 78 int8, 9072.1 MiB
[dev] device 1 weights: 44 u4, 88 k5, 55 codebook, 61 int8, 7824.5 MiB
```

The 1248 MiB of imbalance **is** the int8 LM head (248320x5120 at 1.0625 B/w),
pinned to backend 0 (`engine_graph.cpp` resets `cur_dev = 0` before the head
because `gemv_at` reads `dnnl_for(cur_dev)`).

Per-card VRAM budget, card 0 (card 1 has the same KV and 1248 MiB more room):

| item | MiB | source |
|---|---:|---|
| usable VRAM | 15 541 | `global_mem_size` |
| weights (incl. the int8 head) | 9 072 | startup report |
| activations + recurrent state | 1 484 | `alloc_act_set`; **1 125 of it is `gdn_state`** |
| oneDNN + attn_xmx scratch | 318 | `dnnl_gemm.cpp`, `attn_xmx.cpp` |
| **left for KV** | **4 667** | |

That is why a 256k context "does not fit" at `i8` and why `--max-slots` exists:
the KV at 256k/`i8` is 4.36 GiB/card against 4.46 GiB of headroom - it fits by
0.10 GiB, and **one extra allocation fails the whole load**
(`dev_alloc_on` throws `device alloc failed`).  Both configs do allocate today
(`--ctx 262144` exits 0 at `i8` and `i4:i8`; `kv_cap=8192 blocks` = 8192 MB /
6144 MB model-wide).

**The table is from the `kMaxB = 16` ceiling and was re-verified on the 32
ceiling** (the activation rows double with it): 262144 at `i8` with 16 slots, at
`i8` with 1 slot and at `i4:i4` with 16 slots all still allocate, and 327680 at
`i8` with 16 slots still fails - **the ceiling did not move the boundary.**

**`--max-slots` is the cheapest headroom there is.**  `gdn_state` is
`state_slots_ * n_gdn_local * (dt_rank*d_state*d_state)` f32 = 16 x 24 x 48*128*128 x 4
= **1.125 GiB per card**, and a single-stream long-context run uses one slot.
Every `[layer][slot]` stride is built from `state_slots_` (the clamped
`max_slots`), *not* `kMaxB` - the two differ from this change onward and mixing
them is a silent out-of-bounds write into a USM pool, not a slowdown.  What else
keeps addressing rows by `kMaxB` (`step_info`, the block table, `d_logits`,
`d_partials_dec`) is per-*sequence*, not per-state, and is small.

| card 0 | KV budget | i8:i8 | i4:i8 | i4:i4 |
|---|---:|---:|---:|---:|
| as shipped (16 slots) | 4.46 GiB | 275k | 360k | 519k |
| `--max-slots 1` | 5.51 GiB | 339k | 443k | 640k |

Measured A/B on the box, `--ctx 327680 --kv-type i8` (320k needs 5.45 GiB/card,
i.e. above the 4.46 GiB budget and below the 5.51 GiB one):

```
$ sycl-infer --model ... --layer-map 0-31:gpu.0,32-63:gpu.1 --ctx 327680 --kv-type i8 gen ...
error: level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)
$ ... --ctx 327680 --kv-type i8 --max-slots 1 gen ...
The capital of France is **Paris**.            exit=0
```

So `--max-slots 1` is not a projection: it converts a context that does not
allocate into one that does.

**1M is not reachable on 2x16 GB at >=4-bit KV, and the weights are why.**  1M
tokens need, per card (8 of the 16 attention layers), 16.6 GiB at `i8:i8`,
12.7 at `i4:i8`, **8.80 at `i4:i4`** - against 6.14 GiB of budget even after
`--max-slots 1`.  Confirmed by running it: `--ctx 1048576 --kv-type i4:i4 --max-slots 1` on the
2-card box prints `error: device alloc failed`.  It takes **four cards**
(`--layer-map 0-15:gpu.0,16-31:gpu.1,32-47:gpu.2,48-63:gpu.3`): weights/card
drop to ~5.7 GiB and the budget rises to ~8.9 GiB, so `i4:i4` fits with 0.09 GiB
spare.
Decode does not get slower with more cards - the weight pass is the same total
bytes and each card reads at full speed - but the KV budget per card is what
buys the context.  2-bit KV would fit 2 cards and is rejected on accuracy (the
measured `i8:i4` mean|diff| is already 1.49 against i8's 0.035, and V is the
sensitive side).

**Prefill, not memory, is the wall for 1M.**  Per 512-token chunk the cost is
`414 ms + 6.86 us * depth` (measured: 635 ms at pos 32k, 765 ms at pos 51k, so
the fixed half - weights, GDN, head, handoff - is 54% of a chunk at 51k).  A
51 499-token prompt prefilled in 62.3 s over 101 chunks (617 ms/chunk), and
decode ran 89 ms/token (11.2 t/s).  Extrapolating the quadratic: **256k ~ 15-16
min, 512k ~ 51 min, 1M ~ 3.1-3.4 h.**  So a 1M context is a *batch* capability,
not a serving one, regardless of how the memory is arranged.

#### PCIe, measured - and why streaming the KV cannot fix this

Kernel-free probe (`/tmp/h2d{,2,3}.cpp` on the box, `icpx -fsycl`; no device
code, so it also avoids that box's persistent-JIT-cache SIGSEGV).  2 GiB pinned
transfers, best of 3, order-independent:

| | H2D | D2H |
|---|---:|---:|
| dev0 (19:00.0) | **5.18 GB/s** | 5.68 GB/s |
| dev1 (67:00.0) | **9.98 GB/s** | 10.78 GB/s |
| pageable (dev0/dev1) | 3.70 / 5.30 | - |
| dev1->dev0 peer D2D | 5.23 GB/s | |
| **both H2D concurrently** | **10.37 GB/s aggregate** (5.19 each) | |

The two cards are on **different PCIe generations** - dev1 exceeds the Gen3 x8
line rate (7.877 GB/s), dev0 is 66% of it - so any plan that streams must assume
**5.2 GB/s per card**, not 10.  The ratio is flat from 4 MiB to 2 GiB (1.8-1.93x)
and chunking does not help (64 MiB -> 2 GiB chunks all measure 5.14-5.19 GB/s),
so "stream in small pieces to improve overlap" does not exist.  Peer D2D equals
host staging, so `handoff_x`'s host-staged path loses nothing; and at 9.4 MB the
mode-2 handoff copy is 1.97 ms on dev0, i.e. the 37 ms of host blocking
`PF_HOSTPROF` attributes to it is the `qf.wait()`, not the transfer.

The consequence is arithmetic, and it differs between prefill and decode because
their arithmetic intensities differ by 512x (one chunk is 512 query rows per KV
byte read; one decode token is one):

| | KV read : compute | best case from perfect overlap |
|---|---:|---:|
| decode (1 query/KV byte) | 15.8 : 1 | **5.8%** of the time saved |
| prefill (512 queries/KV byte) | 0.98 : 1 | ~26% on an over-capacity context |

So streaming a KV from host RAM is a **RAG-shaped** tool, not a serving one:
its cost is `generated_tokens x bytes / 5.2 GB/s` per request (256k at `i4:i8`:
0.57 s for 1 token, 18 s for 32, 73 s for 128), while the same context resident
in VRAM decodes at 62 ms/token.  Anything that re-reads the whole KV every token
- which is what decode is - divides by 67 against the 405 GB/s VRAM read ceiling,
and no amount of overlap recovers more than the 5.8% above.

#### Two changes deliberately **not** made

* **Splitting the LM head across cards** (`--head-split`) would free 644 MiB on
  card 0 (+14% context at `i4:i4`, and the difference between "1M on four cards
  fits by 0.09 GiB" and "fits comfortably").  It needs the head's rows 0..N/2 and
  N/2..N as **two** plan calls with two oneDNN keys, and a plan is a
  device-pointer snapshot: the head call's `x`, `out` and oneDNN key are all
  bound to backend 0 in `build_plan`/`gemv_at`.  Getting it wrong does not slow
  anything down, it reads a buffer no kernel wrote for that step - the exact
  class of bug the multi-device notes below describe.  Do it as its own change
  with a test asserting split-vs-unsplit logits are bit-identical.
* **Halving the KV scale plane is doable and the accuracy cost is now measured,
  not estimated** - so what is left is only the payoff.  The plane is 1 f16 per
  32-dim group (`kI8Q`), i.e. 12.5% of an i4 payload and 6.25% of an i8 one.
  Measured on a real post-RoPE dump (`PF_DUMP_KV`, 0.8B, layers 3/7/11, 48
  tokens) against the engine's own quantizer (`scale = amax/p`, p=127 / 7), as
  relative RMS reconstruction error vs the current G=32/f16 format:

  | variant | plane | i8 | i4 |
  |---|---|---:|---:|
  | per-64 groups, f16 scales | -50% | **1.182x** | **1.176x** |
  | per-32 groups, uint8 code + f16 base per (32-token block, group) | -47% | **1.011x** | **1.000x** |

  So the group-width variant costs **+18% RMS** and the uint8-scale variant is
  **free** - and the reason is quantitative, not luck: a uint8 code resolves the
  per-group scale to 1/255 = 0.4%, which is *below* the i8 step (1/254 of amax)
  and ~17x below the i4 step, so it is hidden under the quantization it sits on.
  The 18% is format-independent because the quantizer is scale-invariant: the
  error goes as amax/p whatever p is, so widening the group hurts i4 and i8
  equally - it is a *group* penalty, not a *bit-width* penalty.  Both numbers are
  reconstruction RMS, not the end-to-end attention-output mean|diff| recorded
  above, and both are from the 0.8B: the uint8 variant's cost depends on how
  similar the per-group maxima are inside a 32-token block, which is a property
  of the model and would want re-measuring on the 27B before shipping.

  The reasons it is still skipped are therefore only: **+1.5% context at 256k,
  +6% at 1M** (1M needs four cards regardless, at a 3.4 h prefill), against
  `kI8Q` being a layout constant in 12 sites spanning the pool strides, the
  prefix-cache blob, `kv_row_scales` and both attention readers plus the CPU
  twins.  Note what is **not** a reason: the on-disk prefix-cache format does
  **not** break - `pc_disk.cpp`'s header carries a `kVersion` and `read_header`
  returns 1 for any mismatch, which the scan keeps on disk, counts in the budget
  and never parses, so a layout change costs a cold cache and nothing else.

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

**The verify's USM copies are where the speculative paths die, and they die in
the driver, not in the engine.**  Two failures, both measured 2026-10-20 on the
2x A770 box and both silent until the first speculative cycle:

* **The DFlash2 drafter segfaults, and its only working configuration is the
  broken one.**  The drafter requires the draft on device 0 ("[dflash] needs a
  multi-device oneDNN GPU partition with draft on device 0 - disabled"), and
  with the draft on device 0 it dies with SIGSEGV *inside the Level Zero
  driver*, in `urEnqueueUSMMemcpy` reached from `engine::mtp_verify` (the
  DFlash2's verify reuses the MTP's verify machinery).  It reproduces with a
  32-token generation and a 45-token prompt, and it is not a VRAM shortage:
  `PF_MAX_SLOTS` 1 / 4 / 8 / 16 all crash, and card 0 holds 9 072 MiB of
  weights plus the 1 050 MiB draft.  **So the DFlash2 path does not run at all
  in this build** — and the numbers in the table below are the 2026-10-09 pass,
  the last one where it did.  (Beware the measurement trap: with
  `--spec-draft-device 1` the drafter is *disabled* and the run is a plain
  decode, so it still exits 0 and still produces text — at the plain decode's
  rate, which is how a "the drafter works if you put it on the other device"
  reading gets formed.)
* **The 35B A3B's MTP does not run at all.**  Its GGUF bundles the NextN head,
  the engine's gate passes and the verify's command graphs are recorded
  ("[mtp] verify command graphs: 3 phase(s), 5 rows"), and the plain decode is
  fine (38.7 ms/token) - but the first speculative cycle fails: with the
  default 16 slots (and with 4) the driver reports
  `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, and with `PF_MAX_SLOTS=1` it
  segfaults in the same place as the DFlash2 crash.  The MTP layer's own KV
  slice, the draft's ring and the verify's partials all come out of a card the
  routed experts have already filled.

A USM copy that dies in the driver is a *pointer* or an *ordering* bug, not a
performance one, and the engine's own gates do not catch either: the plan
builds, the graphs are recorded and the plain decode is fine.  **So the MTP
path is a 27B feature in this build, and the DFlash2 path needs a fix.**

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
Raw throughput with register-resident operands (`dev/probe_dpas_occ`): the XMX has
**no K=64 int8 form**, only K=32, so the int8 path delivers half the MACs per
instruction at the same instruction rate - `u4 x s8` (K=64) measures 7.7 T-MAC/s and
`s8 x s8` (K=32) **3.8**, both at 1.9 G-dpas/s.  That is **2x, not 16x**, and it is the
*K ratio*, not a missing capability.  (An earlier revision of this file recorded 254
T-MAC/s for u4 and a 16x ratio; both came from a probe whose u4 arm and dp4a's measured
instruction rate disagree by 10x, and are superseded.)
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

**What prefill actually costs** (27B / 2x A770).  The **in-process mode-2
prefill of 512 tokens is 1.010 s = 507 tok/s** (`dev/bench_pf`, 5 reps, the
mean 504), and the server's cold time-to-first-*content* for a 512-token prompt
is **1.459 s = 351 tok/s** (measured 2026-10-20: the first SSE *chunk* arrives
at 0.150 s, the first content token at 1.459 s).  The ~0.45 s between them is
the server's own overhead: the KV pool grows lazily and its first request pays
for the reallocation, on top of the HTTP, the chat template, the tokenization
and the first token's head.

**`llama-benchy`'s `pp512` column says 2 390 tok/s (214 ms) and that number is
an artefact** - it is the prompt length divided by the time to the first
*streamed SSE chunk*, which the server sends **before** the prefill runs.  The
same trap the depth>0 cells fall into applies at depth 0; the AGENTS.md's own
"the harness's `pp512` column cannot be quoted" note was right and the file
quoted it anyway.  **It is not physically possible**: 512 tokens over 25e9
active parameters is 25.6 TFLOP, and two A770s at the measured dp4a rate
(7.1-7.6 T-MAC/s each) need ~850 ms for the compute alone, so 214 ms is off by
4x.  The measured 1.01 s sits at the compute bound.

Replaying the same 497-call list with the production dp4a launchers would cost
~1.93 s at M=512, i.e. **oneDNN is ~11x better than dp4a at this M** - which is
why prefill needs a *blocked large-M* kernel and neither `nat_gemm` (M-capped)
nor dp4a is the answer.  At depth 16k/64k the whole-prompt rates are 865 / 560
tok/s (llama-benchy, a prompt *on top of* a context of that depth) against the
harness's 984 / 882 (a prompt of that length from an empty KV) - the two are
different shapes, not different engines.  See *Benchmarking* below, and
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for the table.

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
* **`kMaxB` is asked about often, and the answer is "neither a CLI flag nor a
  profile field" - for two separate reasons that have to be kept apart.**  A
  *profile* field is a category error: it is for a number whose measured optimum
  differs per card, and the concurrency ceiling is a capacity policy - putting it
  in the profile would make the same binary's VRAM footprint change with the
  card, which is the hardest class of bug to chase.  A *CLI flag* is
  infeasible: `step_info` is a POD with `int32_t pos[kMaxB]`,
  `tokens[kMaxB*kMaxT]` and `mrope[4*kMaxB*kMaxT]`, plus `h_argmax[kMaxB]` and
  `std::array<pc_slot, kMaxB>`, so the row count is compile-time by
  construction.  **The user-facing knob already exists and is `--max-slots`**
  (how many sequences the engine will actually run, and how much recurrent
  state it reserves); `kMaxB` is only the *ceiling* those buffers are dimensioned
  for.  The one thing `--max-slots` does **not** shrink is the row buffers
  (`d_logits`, `d_partials_dec`, the activation rows), which are `kMaxB`-sized
  regardless - that is the real gap, and it is a refactor, not a knob.
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
* **Never write a real IP address, host name, SSH/host account, or private home
  path into a doc, a comment, a source file, or a commit message.**  This repo is
  pushed to a public remote, so anything committed is public.  Use neutral
  placeholders instead: `bench-host` for a machine, `proxy.internal:8118` for a
  proxy, `/path/to/<file>.gguf` for a model path.  A measurement is still
  reportable without its location — "the 2x A770 box, i9-10940X" carries all the
  hardware detail, and the address carries none of the value.  This includes the
  obvious ones: an RFC1918 literal in a benchmark note, a `ssh user@host` line, and
  a `$HOME`-derived path baked into a test's default argument.  Note the trap in
  both directions: RFC1918 / IANA special-use *ranges* (`192.0.0.0/24`,
  `192.168.1.1` as an RFC1918 test vector) are constants the
  `url_fetch` SSRF classifier is built on and must stay; the rule is about *this
  infrastructure's* addresses, not about the protocol.
* Do not commit unless explicitly asked.

## Environment variables

Tuning/diagnostic knobs (unset/`0` = default).  Changing them selects different
kernel variants, so performance numbers must state the env used.

**Model / memory**
`PF_CTX`, `PF_KV_CAP_MB`, `PF_KV_GROW` (pool growth step, default 64 blocks),
`PF_MAX_SLOTS` (concurrent sequences; sizes the recurrent state, default
**16** - 1.125 GiB per card on the 27B, so this is the cheapest context
headroom there is; the ceiling is `kMaxB`=32),
`PF_KV_TYPE` (`i4`|`int4`|`i8`|`bf16`|`f16`|`f32`, default `i8`; `--kv-type`
overrides it; `K:V` sizes K and V independently, e.g. `i4:i8`), `PF_KV_F32`,
`PF_KV_BF16`,
`PF_SI4` (re-quantize weights to 4-bit SIn), `PF_META` (fp32 side scales).

**Compute path**
`PF_YARN` (YaRN context extension, **default off**; `--yarn`), `PF_YARN_FACTOR`
(the extension factor; 0/absent derives `--ctx / <arch>.context_length`),
`PF_YARN_ORIG_CTX` (override the trained context the factor is measured
against), `PF_YARN_BETA_FAST`/`PF_YARN_BETA_SLOW` (ramp cutoffs in rotations,
default 32/1), `PF_YARN_ATTN_FACTOR` (override the derived `1 + 0.1*log10(f)`),
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
classic kernel is faster), `PF_ATTN_STREAM=1` (put the attention matmuls back on
the separate oneDNN stream `st_a`, which is a **data race** - see the note on
`st_a` in `dnnl_gemm::impl`; unset/0 keeps them on `st` with everything else),
`PF_ATTN_WAIT` (force a blocking per-attention-matmul stream wait; default off).

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
`PF_XMX_GRED=N` (gather stage-1 work-group count, default 64), `PF_XMX_GQ=N`
(gather stage-2 lanes per key row inside a fixed 64-thread work-group;
1 is the old one-thread-per-row shape with the 16-byte accesses kept, so the
sweep separates the stripe width from the access width), `PF_XMX_TAILBLK=0`
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

**Expert split / ncmoe / three-tier expert cache** (`--moe-resident N`,
`--moe-vram-mb MB`, `--moe-ram-mb MB`; routed-MoE models only, see
[`docs/design/15-moe-ncmoe.md`](docs/design/15-moe-ncmoe.md))
`PF_MOE_RESIDENT` (experts kept on the device per layer, 0 = every expert
resident, the default and what a model that fits does anyway),
`PF_MOE_VRAM_MB` (device budget for the expert bank - authoritative over the
count, and the bank's *content* is the measured hot set from `moe_calibrate()`,
not an index range), `PF_MOE_RAM_MB` (host budget for the RAM tier of expert
blocks, an LRU keyed by (layer, expert); 0 = off. Measured 48-66 % hit rate at
150 MB on the 35B A3B). Diagnostics: `PF_MOE_CPUDBG` (per-layer
`spilled of n_pairs`), `PF_MOE_REFDUMP` (per-layer routed fingerprint
sum/sumsq/max), `PF_MOE_REFDUMP_ALL` (every layer instead of layer 0 -
**the only switch worth trusting**: a layer-0-only dump cannot see the layers
whose quantization type differs), `PF_MOE_REFDUMP_LAYER=<L>`,
`PF_MOE_HIST` (the routed-pair histogram: pairs per active expert, the mean, and
how many experts fit one row tile - see "The routed-pair histogram" below),
`TEST_MOE_RESIDENT` runs the stage tests through the split.

**Expert GEMM selection** (routed-MoE models only): `PF_MOE_DP4A` (**default
on** - the grouped dp4a expert GEMM; `=0` restores the fp32 dequant experts,
measured 248.2 vs 851.8 t/s pp512 on the 35B A3B with an identical argmax, see
*Measured results*), `PF_MOE_CFG` / `PF_MOE_CFG_DOWN` (tile shape `M x C x NSG` of
`moe_grp_impl`, plus the `HOIST` A/B in cases 16-21, selected **per projection** -
gate+up is `FUSE==2` and reads `PF_MOE_CFG`, down is `FUSE==1` and reads
`PF_MOE_CFG_DOWN`; the defaults are `M=8 C=2 NSG=16` for both, and
`dev/bench_grp` is the sweep - cases 40-57 are the register-budget sweep).  Note that
`dev/grp_min` drives the **down** path, so exactness runs must set
`PF_MOE_CFG_DOWN` - that is how a staged kernel under test once read as exact.
`PF_MOE_FLAT` / `PF_MOE_FLAT_DOWN` (put the row tile in the grid - one
work-group = one tile - instead of looping it inside an (expert, column block)
work-group; the trade is not uniform, see *Measured results*),
`PF_MOE_PERSIST` (replace the grid with `2 * compute_units` work-groups that
work-steal `(expert, row tile, column block)` triples off an atomic cursor, with the
item total summed on the device; **exact at every configuration, and 3% SLOWER -
refuted, see *Measured results*, so leave it off**), `PF_MOE_PERSIST_WG` (its grid,
default `2 * compute_units`), `PF_MOE_GRPDBG` prints the resolved tile and grid and is the assertion on the
tile selection path; `PF_DNNL_FPMATH` (widen the oneDNN int8/u4 matmul's accumulation to `fpmath_mode::any,true`; a guard against the K >= 5376 NaN the A770 library records, **default off** because no model here reaches it and it measures neutral), `PF_MOE_DP4A_DEC` (run the grouped dp4a expert GEMM on a DECODE step too;
**default ON**, because the decode now has its own tile - the old "decode is the
wrong shape for it" verdict was measured with the PREFILL's M=8 tile and a
batched decode was never even on the decode branch, see *Decode: the grouped
expert GEMM had the PREFILL's tile* section under *Measured results*; `=0`
restores the fp32 dequant experts),
`PF_MOE_CFG_DEC` (the DECODE's tile, 0 = auto: one row per expert for a
single-token decode and two for a batched one, because the weight re-read factor
is `ceil(p_e/M)`; a non-zero value forces that tile for both projections),
`PF_MOE_GRP_MIN_TOK` (prefill chunk size below which the
grouped kernel declines, default 8), `PF_MOE_DQ2` (halves the dequant
arithmetic with every load unchanged - **wrong results by design**, and the
experiment that bounds the f16 idea at 18%), `PF_MOE_NOACTQ` skips `moe_actq_launch` (wrong results -
diagnostic only; measured at 1.3 ms of the 501 ms gate+up phase, i.e. **the
activation quantizer is not part of the gap**).

## Measured results and dead ends

**A throughput number belongs in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md),
not only here.**  This file holds the *conclusions* a change must be read
against; `docs/PERFORMANCE.md` holds the measured table (prefill and decode, by
context depth and batch size, per model and per speculative path) with the
conditions each number was taken under.  The rule for a change that moves or adds
a number:

1. **Update the affected table in `docs/PERFORMANCE.md`** — with the model, the
   layer split, the KV type, the harness, the exact command and the date.  A
   throughput number without its conditions is not a number: the same binary
   measures 13 % apart on different days, and a different `PF_*` switch selects
   a different kernel.
2. **Update the conclusion here only if the shape changed** — a new knee, a new
   floor, a different bottleneck.  The raw numbers live in the table; this file
   says what they mean.
3. **Fix every quote of the old number in the same change** — the README, a
   design doc, a comment.  A number that is correct in one file and stale in
   another is worse than no number, because the reader cannot tell which is
   which.

The reason the table is a separate document: the numbers change with every
kernel, but the *shape* of the curves does not, and a future change has to be
read against both.  A measurement that exists only in a commit message, a chat
log or a local file is not in the record - if it is worth quoting, it goes in
the table.

Everything in this section is a **conclusion with the number it came from**, not
the hunt that produced it.  The hunt is in the commit history; what a future
change needs is the conclusion, so that it does not have to be re-derived.

Measurements are on the 2x A770 box (two Arc A770, driver 12.55.8) unless
stated.  The 35B A3B is `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` with
`--layer-map 0-19:gpu.0,20-39:gpu.1`; the 27B is `Qwen3.8-27B-UD-Q4_K_M.gguf`
with `--layer-map 0-31:gpu.0,32-63:gpu.1`.

### The routed experts: the state, and the wall they are against

**The state.**  The grouped dp4a expert GEMM over the **native GGUF blocks**
ships and is the default for the prefill (`PF_MOE_DP4A`) and, since the decode
got its own tile, for the decode too (`PF_MOE_DP4A_DEC`).  It quantizes the
activation to int8 with per-32-group scales and multiplies with dp4a directly on
the Q4_K nibbles - a 32-value group is 32 bytes of nibble, i.e. exactly two dp4a
operands, so there is no expansion pass and no extra weight storage.  The
prefill went **248.2 -> 851.8 t/s at pp512** with an identical argmax; the
current numbers are **pp512 778.3 t/s (argmax 11171)**, **pp16384 1485 t/s
(1204)**, **pp65536 1402 t/s (4779)**.

**The wall.**  The experts are 18.22 GiB of GGUF blocks, and the decode reads
`8/256` of that bank per token per layer.  It reads it at **16-26 GB/s against a
376 GB/s card**, and the reason is the **format**: the GGUF K-quant layout
interleaves by value, so a lane's four values of a 32-value group are four
nibbles inside **one byte** - there is nothing contiguous for it to load, and the
warp gets **32 B per load instruction** where int8 gets 128 B.  The four-step
decomposition of the decode's expert time: the weight arithmetic is 5.7 ms
(12%), the activation loads 0.7 ms (1.5%), and **83% is reading the weights and
nothing else**.  The dense int8 GEMV on the same device memory gets 132 GB/s
with 128 B per load, so the gap is 5x and it is a property of the on-disk
format, not of the kernel: no tile shape and no split count touches it.

**Reading the four bytes at once does not help.**  The obvious narrowing of the
access is to read a lane's four bytes into registers and take both nibbles from
each: bit-exact, and **measurably no change at all** (the load-only floor is
38.8 ms against 38.6).  **The compiler was already CSE-ing it.**  So
"instructions per byte" is not the lever either.

**What is left, stated as a hypothesis and not as a measurement.**  The fp32
expert kernel's grid is `n_act x n_ff_exp` = 4096 warps, each reading a
*different* output row, i.e. a different superblock 1152 bytes further along -
4096 concurrent read streams on a 1152-byte stride, the shape that thrashes DRAM
row buffers (and exactly what the dense int8 GEMV does not do, since each warp
sweeps its own row sequentially).  Testing it means giving each warp a block of
consecutive output rows instead of one, which cuts the stream count by the same
factor.  **`PF_MOE_NPB` is that experiment** (one warp walks NPB adjacent
output rows): it is worth **6-7%** and then the occupancy loss takes over hard
(NPB=16 is 22-26% worse than NPB=1), so the stream count is real and it is not
the 5x.

#### The routed experts: the dead ends, each with its number

* **The dp4a rewrite of the *fp32* path** (the original diagnosis, and the
  reason the grouped path exists) is **13x slower** than the fp32 kernel it
  replaces.  The mechanism is the group scales, not dp4a: with `xq_launch`'s int8
  activations the dot is `scale * (A * sum(nib*q) + B * sum(q))`, so every
  32-value group costs two extra global loads and three dependent float ops
  *per (row, pair, group)*, and those loads are re-issued for every one of the
  expert's ~512 output rows because they are pair-indexed, not row-indexed.  The
  grouped path fixes it by staging the activation once per tile.
* **SLM row staging**: 139 vs 148 t/s.  The reuse is real but one sub-group per
  work-group throws away 8x the occupancy.
* **Pair-tiled dequant reuse** (`kMoEPB` 16/32): worse - register spill.
* **Row-blocking on its own** (`kMoER`): gate_up 1170 -> 3430 ms.  The
  dequant-to-FMA ratio is 1/(2*kMoEPB) and *independent* of kMoER, but without
  the pair tiling it collapses to 1:1 and the dequant is 10 FMAs' worth of ALU
  per FMA.  Row-blocking only pays if the pair tiling is kept, and then
  `ag[kMoER][kMoEPB] + au[kMoER][kMoEPB]` spills.
* **The SLM activation tile** (`kMoEPB x kMoEKTC` floats = 32 KB, shared by the
  work-group's eight warps): **2084 vs 1170 ms and the logits are garbage**.
  **The reason not to retry it**: the router's one-work-group-per-token version
  - logits in SLM, softmax and top-k after a barrier - "never took effect" on
  the A770's Level Zero backend while passing under OpenCL and on the Iris Xe.
  That is this backend's codegen for SLM-plus-barrier shapes, and a barrier per
  K-chunk is exactly that shape.
* **The `kMoEPB` sweep** (8/12/16/20/24/28/32/40/48/64) all agrees with the
  wall clock, and **8 is the optimum**: the 8-tile's 2x dequant redundancy is
  cheaper than the registers that would remove it.  **Reverting the sweep back
  to 8 is load-bearing** - do not "fix" the roofline gap by widening the tile.
* **The constraint is the register budget `P = M*FUSE*C`,** not the
  dp4a-per-load ratio `M*FUSE*C/(M + FUSE*C)`.  Every P=32 config lands at
  6.0-6.1 ms, P=64 at 7.0-7.3, P=128 at 20-51.  The sweep refutes the ratio
  story: the optimum's ratio is 2.67, *below* the M=16 tile's 3.20.
* **Transposing the weights to raise M** is the wrong trade twice over.  The
  coalescing probe (same bytes, same instructions, same grid, only the lane
  mapping changed): KAXIS 194.8 GB/s against NAXIS 74.7 at C=1 - **2.6x worse**,
  and C=1 is exactly the configuration the transpose needs.  And it cannot be
  materialised: a transposed copy is the same 18 GiB, the decode GEMV still
  needs the row-major original, and a per-layer transposed scratch would cost
  ~104 ms per card to build.
* **The persistent work-stealing expert GEMM** (`PF_MOE_PERSIST`, default off)
  is **exact at every configuration and 3% slower**.  The premise it was built
  on is wrong: `PF_MOE_FLAT=1` already puts one work-group per (row tile,
  column block), so the imbalance is *across* work-groups and the hardware
  balances 12288 of them - there is no serialisation to remove.
* **The row tile in the grid** (`PF_MOE_FLAT`) is +2% on gate and -11% on down.
* **DPAS/XMX for the expert GEMM** loses at the MoE's own M: 7.52 against 6.93
  ms (gate/up) and 7.74 against 3.80 (down).  The DPAS accumulator is 32x denser
  than dp4a's, but the register ceiling is the same - the sweep's cliff tracks
  `NT*R` (the accumulator in GRF/lane) exactly as tightly as it tracks R, and
  the practical ceiling lands at 16-32 GRF/lane either way.  **DPAS does not get
  more registers; it just packs them differently.**
* **The oneDNN grouped matmul is not in the public 3.11 API** - the headers
  carry only `dnnl_graph_op_matmul` and the plain `dnnl::matmul`; the grouped
  `dnnl::matmul` that OpenVINO's MoE PRs use lives in their fork.
* **A u4 store for the experts is memory-blocked**: 0.625 B/w x 18.22/0.5625 =
  10.1 GiB per card *on top of* the 9.1 GiB raw upload the decode path still
  reads, against 15.11 GiB usable (confirmed `UR_RESULT_ERROR_DEVICE_LOST`).
  And `add_weight_w4` builds a prefill primitive ladder per weight, so 256
  experts x 3 tensors x 40 layers = 30780 ladders at ~15 ms of primitive
  creation each is not a load time.  **So that route is a full
  weight-representation swap** - replace the raw upload with the native store and
  route the decode through the native GEMV too - not a prefill-side change.
  Two details worth not re-deriving: the stores must be registered *after* the
  raw upload, because the upload skips any tensor a oneDNN table already holds
  and the MoE decode path still reads these tensors' GGUF blocks (registering
  first strands it on an unuploaded pointer); and `nat_gemm` rejects M > 13
  while a 512-token prefill routes ~16 pairs per expert, so the pairs must be
  tiled either way.

#### The routed experts: the decode, and the two shape bugs that were in it

**The decode's tile is a SHAPE, not a tuning knob.**  `M` is the number of
(token, expert) pairs a work-group handles, so the weight re-read factor is
`ceil(p_e/M)`.  The prefill routes ~54 pairs per expert and M=8 is right; a
decode step routes **one** pair per selected expert, so the prefill's M=8 tile is
**seven eighths empty**.  The decode therefore has its own tile: **M=1** for a
single-token decode (every selected expert gets exactly one pair, so a wider
tile is pure waste) and **M=2** for a batched one (some experts get two or
three, where a one-row tile re-reads their weights once per extra pair).
`PF_MOE_CFG_DEC` (0 = auto) selects it.

**And "decode" had to mean the mode, not the token count.**  The selection was
`dec_step = (n_tok <= 1)`, and a *batched* decode has `n_tok = rows = N > 1` -
so every batched decode step was treated as a prefill and ran the prefill's
M=8 tile.  Only a one-row decode ever reached the decode branch, which is why
the old measurement ("the grouped path is 1.38x slower on decode") was taken
where the tile was worst.  It is now `mode == 0`.

**The decode step, 35B A3B / 2x A770** (`dev/bench_conc`, one process per
cell; the `fp32 experts` column is the fp32 dequant expert GEMM, the `dp4a`
columns are the grouped dp4a path with the given tile):

| N | fp32 experts | dp4a M=8 | dp4a M=1 | dp4a M=2 | shipped |
|---|---:|---:|---:|---:|---:|
| 1 | 38.98 | 53.95 | **37.69** | 41.50 | **37.93** |
| 2 | 60.29 | 60.32 | 60.34 | 60.38 | **56.06** |
| 4 | 80.21 | 81.27 | 81.05 | 81.44 | **66.19** |
| 8 | 87.62 | 87.94 | 79.62 | **78.62** | **78.54** |
| 16 | 135.84 | 137.43 | 136.01 | **130.50** | **129.63** |
| 32 | 191.96 | 193.32 | 212.77 | **191.64** | **191.37** |

**The decode now quantizes its activation like the prefill does**, so the
decode-vs-prefill oracle compares two paths that agree on the activation
quantization (27B split, 3/3) and the prefill oracles are unchanged.  The
decode's **token stream moves** with it (a near-tie can flip) - the same class of
change the file already records for batching.  The MTP and DFlash verifies are
**mode 2** (a batch of k+1 rows), so they keep the prefill's tile and are
unaffected.

### The batched decode: the row count must be exact, and it is not a rounding

**The decode runs exactly `n_rows` rows.**  There is one plan bucket per row
count (1..kMaxB) and the decode picks `tb == n_rows`; `n_rows > kMaxB` throws
rather than truncating.

**The old bucket set was `{1,2,4,8,16}` and the batch was rounded UP**, which
was two bugs at once:

1. Nine concurrent sequences ran a **sixteen-row** forward - seven rows of dead
   work, paid for on every step.  That was the 38 ms cliff at N=9 (125.60
   against 87.62 ms/step) and it grew with the batch.
2. **More than sixteen sequences were never computed at all.**
   `buckets_.back()` is tb=16, so a batch of twenty rows ran a sixteen-row
   forward: rows 16.. of `d_logits` were never written, kept the previous step's
   values, and the sampler read them.  No error, no NaN, just plausible text -
   and the concurrency table was measured on that engine.  **A performance table
   measured on an engine that is computing the wrong thing is worse than no
   table**, and the only reason this one looked plausible is that it was smooth.

**The dense model's batch curve has a CLIFF at the native GEMM's M ≤ 13
boundary, and it is worth 2.5x.**  The 27B's aggregate throughput *falls* from
8 rows to 16: 62.53 tok/s (127.9 ms/step) at N=8 against **45.22 tok/s (353.8
ms/step)** at N=16.  The cause is `nat_gemm_launch`, which refuses M > 13: above
13 rows the dense layers leave the native u4 path for oneDNN's int8 matmul.
`PF_NAT=0` (which forces the oneDNN path at every M) prices it exactly:
**315.2 ms/step at N=8** against 127.9 with the native path, and **353.9 vs
353.8 ms/step at N=16** - i.e. at 16 rows the native path is already not being
used.  The MoE model crosses the same boundary far more gently, because only
its dense layers are affected.  **This is the same M > 13 boundary the native
GEMM's extension was measured against, and on a dense model it is not a
rounding detail but a 28 % loss of aggregate throughput.**

**The concurrency curve, on the fixed engine** (35B A3B, depth 512, 128 decode
steps, one process per cell; the full table with the prefill column and the
conditions is in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)):

| N | decode (aggregate) | ms/step |
|---:|---:|---:|
| 1 | 26.38 tok/s | 37.91 |
| 2 | 35.70 tok/s | 56.03 |
| 4 | 60.91 tok/s | 65.67 |
| 8 | 101.12 tok/s | 79.11 |
| 16 | 122.05 tok/s | 131.09 |
| 32 | 168.33 tok/s | 190.11 |

**pp saturates at 16** (+2.4% for twice the concurrency) and **tg does not stop
at 32** (+41.5% over the same doubling, a marginal row of 3.86 ms against
6.39 ms/row over the 8-to-16 segment).  The prefill's share of that saturation
is the mode-2 batch: `batched_prefill_fit` caps a batch at `kMaxB * kMaxT`
tokens, and 16 -> 32 doubled it from 512 to 1024, which is worth **+4.3%** on
its own at depth 16384 (isolated with `PF_PFB_MAX_M`, interleaved
512/1024/512/1024, argmax 1204 in all four).

**The non-monotonicity in the 5..12 range was a misattribution, and it is
recorded as one.**  On the engine with the row count fixed but the tile still
the prefill's, N=8 was **20 ms cheaper than N=7** (87.8 against 108.6
ms/step), reproducibly, and two independent instruments put it in the routed
experts' decode path: the per-bucket profile (the experts' share 61.6 / 68.8 /
50.4 / 51.8 ms at N=6 / 7 / 8 / 9) and the `PF_ABL_NOMOE=1` ablation (38.9 /
40.3 / 39.2 / 46.1 ms - the anomaly gone).  The reading was "the experts' decode
cost is a function of WHICH experts the routing picks, not of how many rows
there are - a routing lottery".  **It was the M=8 tile's interaction with the
routing**: the re-read factor is `ceil(p_e/M)`, so on a decode step every
expert with fewer than eight pairs still paid for eight rows of dp4a, and how
many pairs an expert gets depends on the routing.  With the decode's own tile the
range is flat and monotone: **75.5 / 78.7 / 78.7 / 87.9 ms/step at N=6 / 7 /
8 / 9** (i.e. -24.6% / -27.5% / -10.4% / -9.0% against the row-count-only
engine).  **The generalisable part: a shape mismatch between a kernel's tile and
the shape it is actually given can look like anything**, and it survived four
interleaved measurements and a two-instrument attribution because every
instrument agreed on *where* the time was and none of them asked whether the
tile was the right shape.

**The native GEMM's M > 13 boundary is where it should be, not a leftover.**
`nat_gemm_launch` rejects `M > 13` and `M` is a template parameter, so extending
it to 32 looked like a small change.  It is not: the native path's rate falls
monotonically with M (u4 91.8 GB/s at M=7, 60.0 at M=8, 51.6 at M=13, **36.0 at
M=16**, 7.0 at M=32) while oneDNN's int8 matmul holds ~110 GB/s at the LM head's
shape - **the two cross exactly at the existing boundary**.  The extension was
implemented, verified exact at every M and every format (`ok=1 maxdiff=0.0000`),
and measured **2-6x slower per byte**; N=32 went 191.96 -> 210.34 ms/step.
**A switch that is exact and slower is still a switch that should not exist**,
so it was reverted and only the measurement kept.

**The host is not the bottleneck, and the harness is inside the timed region.**
`PF_HOSTPROF` at N=16 measures the host at **111.48 ms against the device's
122.89 ms**.  The 12.9 ms between the profile's total and the wall clock is
mostly `bench_conc`'s own per-row logits fetch and host argmax
(**~0.45 ms per row**: 1 MB D2H plus 248320 comparisons), so the engine's own
marginal row is 3.41 ms and not 3.86.  That closes the "record a command graph
for the batched multi-device decode" idea - worth at most the 12.9 ms, and
unavailable above thirteen rows anyway, because oneDNN's int8 matmul cannot be
recorded into a SYCL command graph and that is the dense layers' path at M > 13.

**The sampler is host-side, and that is a real product cost**, not just a
harness artefact: the scheduler fetches `d_logits` row by row and argmaxes it on
the host (~0.45 ms per row per step; 14.4 ms per step at N=32).  A device-side
argmax already exists for the MTP path (`mtp_argmax_launch`, 28x faster than
the host's) and the server does not use it.  It cannot simply be copied: MTP
needs only the argmax, while the `logprobs` report needs the selected token's
value, so the host still has to fetch that one.

### The attention: what is at its floor and what is not

**The prefill attention runs on oneDNN's int8 matmuls** (`attn_xmx`), and those
matmuls are **the fastest thing in the prefill**: 8.3 T-MAC/s for QK and 14.4 for
PV.  Nothing beats them here, and the things that look like they might (DPAS,
fused softmax) all lose - see the dead ends below.

**Four things about `attn_xmx` are load-bearing:**

1. A plain int8 matmul sums over k, so a per-key scale cannot be applied after
   it - K and V each carry **one block-wide scale**, folded in at gather time.
2. The query tile stacks every HPG query head sharing a kv head, so the matmul
   width is `HPG * tokens` and the KV is read once per kv head, not once per
   query head.
3. The matmuls always run at a **fixed width `kBlk = 8192`** (zero-padded),
   because oneDNN primitive creation is ~15 ms per new shape and a per-chunk `N`
   made the primitives never reusable - that alone was a 12x regression.
4. The block-max reduction in the gather must be spread over many work-groups,
   never one: a single 256-thread group reducing a whole 8192-key block was
   18.1 ms of the 37.8 ms one attention layer costs at 64k depth.  It is now
   `kGatherRed = 64` work-groups reducing a slice each into a private slot,
   then one small kernel folds the partials.  `fmax` is exact and associative,
   so the block scale - and the whole output - is bit-identical.  Measured:
   18.1 -> 9.1 ms, and the marginal 512-token chunk at 64k **975 -> 833 ms
   (-15%)**.

**The gather's second stage was the one thing not at a machine limit, and it is
fixed.**  It requantizes the paged K/V into the contiguous scratch the matmuls
read, and it was one work-item per key, each walking 256 elements a byte at a
time: ~386 us per iteration against a ~23 us bandwidth floor (17x).  The fix is
`PF_XMX_GQ` lanes per key row, each lane owning whole 32-value scale groups so
the int8 accesses are 16-byte vectors: **0.902 ms -> 0.771 ms per call (4.5x)**
at GQ=8, and the end-to-end effect is depth-local (the gather runs once per
(kv head, block chunk), so its share grows with the key count): pp512 flat,
pp16384 1548.8 -> 1568.9, pp65536 1389.2 -> 1437.4.  **Vectorising the
accesses was 2/3 of the win and the parallelism the other 1/3** - the opposite of
what the shape invited ("8192 work-items is 256 warps, so half the machine
idles").  **The generalisable half: count the memory *operations* before
blaming the memory *system*.**

**Two negative results in the same area.**  Tiling M so the 101 MB score matrix
fits the 16 MB L2 is **+53%** (the L2 caps the tile *area*, so the oneDNN execute
count goes up ~7x either way at ~35 us each).  And computing the block max from
the **scale plane alone** - exact, since every non-zero 32-wide group quantizes
its extreme to |k| = 127, so `max|k*scale| == 127 * max(scale[])` - cuts that
pass's traffic 16x and measures **neutral** (833 -> 830 ms), because the gather
is bound by the page-table lookup latency, not by bandwidth.  (It needs an
all-zero group to store scale 0 rather than 1, plus an `i8_quant` zero-scale
guard; both are reverted, since a KV-format change is not worth 0.3%.)

**After the gather fix the attention is at its floor.**  The per-call split at
the deepest 64k chunk: qk 4.346 ms (37%), softmax 3.873 (33%), pv 2.501 (21%),
gather 0.771 (6%).  **58% of it is two oneDNN int8 matmuls already running at
8-14 T-MAC/s**, faster than dp4a's real kernel (7.1-7.6) and 3.8x faster than
DPAS.  There is nothing left to reclaim there short of changing the two-pass
structure - and **the softmax is already near its own floor**: it reads the
int32 score block and writes the u8 P operand, `M x cols x 5` bytes = 698 MB per
call at 64k, which at 3.873 ms is **180 GB/s, 48% of the card**.  Fusing it into
PV is **worse**, not better: it would save the 1-of-5 write of P but hand PV a
**fp32** operand instead of a u8 one, raising its read traffic 4x against the
stage that is currently the cheapest thing in the call.  **That is the whole
reason the path is split in two**, and it is a bandwidth fact, not a scheduling
choice.

**The decode attention is a different animal, and it is occupancy-bound, not
bandwidth-bound.**  `n_head * n_splits` 32-thread warps per layer, 8 per EU, and
past 8 warps/EU the excess is not co-resident and the time steps *up*.  The
split count is therefore a per-model optimum, not a constant:

| | 160 | **240** | 256 (profile) | 320 | 384 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| 35B A3B, 64k ms/tok | 59.6 | **58.4** | 59.6 | 64.1 | 62.7 | 62.5 |
| 27B, 64k ms/layer | 1.70 | - | - | **2.68** at 176 | | |

(The 27B's cliff is at 176 splits = 8.25 warps/EU.  The 35B's optimum is *fewer*
warps, which is the same "more warps is worse" shape the experts have.)

**Three consequences worth not re-deriving, all measured:**  do **not** "fix"
the GQA K/V re-read by sharing loads across query heads (5x **slower** - the
re-read is free from L2 and the lost warps are not); do not unroll the key loop
for MLP (KU=2..8 all slower; IGC already pipelines it); and do not widen a lane
to 16 dims for 16-byte loads (`attn_dec16_kernel`: 4.45 vs 1.55 ms/layer - 16
accumulators + 16 query + 16 K + 16 V floats per lane makes IGC allocate 128
registers and spill; the report's §4.3 has two bugs in that kernel that both
measured *faster* while being wrong, so read it before trusting a timing).
**The 8-warps-per-EU wave is architectural, so no register trick buys more
warps.**

**The GQA sharing is not available on this model at all.**  The grouped decode
attention (`PF_DEC_GROUP`) is exactly the sharing kernel - "one warp owns all
HPG heads of a kv head: K/V are loaded once per key and reused from registers
for every head" - and it is the one thing that could have taken the 8x K/V
re-read out of the decode attention at 256k.  But it requires the **256-wide
head** (`constexpr int HD = 256`) and the 35B A3B has head_dim 128, so
`PF_DEC_GROUP=1` silently falls back to the classic kernel: **50.0 vs 49.9
ms/tok at 64k and 90.0 vs 90.1 at 256k**, identical because the knob does
nothing.  The file's older "5x slower on the A770" note was measured on a model
whose classic kernel has 4 query heads per kv head; on one with 8 the grouped
kernel would have 8x fewer warps, which is the wrong direction for a
latency-bound kernel.

**And the number that would decide whether anything is left there is not
measured.**  The decode attention moves 51.9 ms for 20 layers at 256k, i.e.
2.6 ms per layer for a 32.8 MB kv head: **12.6 GB/s of useful traffic, or
100 GB/s if the 8x GQA re-read misses the L2.**  Which of the two it is decides
the whole question - if the redundancy is served by the L2 there is no traffic
to save and the attention is simply latency-bound; if it is not, the sharing is
worth up to 8x on 51.9 ms.  The 8 warps of one split are `n_splits` apart in the
grid index, so they are co-resident (and the L2 able to serve the redundancy)
only by accident of the scheduler, and 32.8 MB per layer against an 8 MB L2 is
borderline either way.  **A measurement that cannot tell 12.6 from 100 GB/s
cannot price the fix**, so the next step there is a counter, not a kernel.

### Long context: the capacity, and where the time goes at depth

**The capacity table (2x A770, card 0; the 1248 MiB of imbalance is the int8 LM
head, pinned to backend 0):**

| item | MiB | source |
|---|---:|---|
| usable VRAM | 15 541 | `global_mem_size` |
| weights (incl. the int8 head) | 9 072 | startup report |
| activations + recurrent state | 1 484 | `alloc_act_set`; **1 125 of it is `gdn_state`** |
| oneDNN + attn_xmx scratch | 318 | `dnnl_gemm.cpp`, `attn_xmx.cpp` |
| **left for KV** | **4 667** | |

So the KV at 256k/`i8` is 4.36 GiB/card against 4.46 GiB of headroom - it fits
by 0.10 GiB, and **one extra allocation fails the whole load**
(`dev_alloc_on` throws `device alloc failed`).  **`--max-slots` is the cheapest
headroom there is**: `gdn_state` is `state_slots_ * n_gdn_local *
(dt_rank*d_state*d_state)` f32 = 1.125 GiB per card at 16 slots, and a
single-stream long-context run uses one slot.

| card 0 | KV budget | i8:i8 | i4:i8 | i4:i4 |
|---|---:|---:|---:|---:|
| as shipped (16 slots) | 4.46 GiB | 275k | 360k | 519k |
| `--max-slots 1` | 5.51 GiB | 339k | 443k | 640k |

Measured A/B on the box, `--ctx 327680 --kv-type i8` (320k needs 5.45 GiB/card,
i.e. above the 4.46 GiB budget and below the 5.51 GiB one):

```
$ sycl-infer --model ... --layer-map 0-31:gpu.0,32-63:gpu.1 --ctx 327680 --kv-type i8 gen ...
error: level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)
$ ... --ctx 327680 --kv-type i8 --max-slots 1 gen ...
The capital of France is **Paris**.            exit=0
```

So `--max-slots 1` is not a projection: it converts a context that does not
allocate into one that does.  **The table was re-verified on the `kMaxB = 32`
ceiling** (the activation rows double with the ceiling): 262144 at `i8` with 16
slots, at `i8` with 1 slot and at `i4:i4` with 16 slots all still allocate, and
327680 at `i8` with 16 slots still fails - **the ceiling did not move the
boundary.**

**1M is not reachable on 2x16 GB at >=4-bit KV, and the weights are why.**  1M
tokens need, per card (8 of the 16 attention layers), 16.6 GiB at `i8:i8`, 12.7 at
`i4:i8`, **8.80 at `i4:i4`** - against 6.14 GiB of budget even after
`--max-slots 1`.  It takes **four cards** (`--layer-map 0-15:gpu.0,16-31:gpu.1,
32-47:gpu.2,48-63:gpu.3`): weights/card drop to ~5.7 GiB and the budget rises
to ~8.9 GiB, so `i4:i4` fits with 0.09 GiB spare.  Decode does not get slower
with more cards - the weight pass is the same total bytes and each card reads
at full speed - but the KV budget per card is what buys the context.

**The decode's cost is linear in depth and has no knee** (N=1, 128 decode steps
at the depth, best of 2, depth 512 run first and last as the drift control):

| depth | prefill rate | tg (tok/s) | ms/step |
|---:|---:|---:|---:|
| 512 | 745 | 26.38 | 37.91 |
| 4096 | 1 269 | 25.91 | 38.59 |
| **16384** | **1 477** | 24.54 | 40.76 |
| 65536 | 1 399 | 20.20 | 49.50 |
| 131072 | 1 242 | 15.54 | 64.34 |
| 262144 | – | 11.02 | 90.74 |

(the 512..131072 rows are the 2026-10-20 sweep in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md), whose `pp` column is the average
rate of a whole prompt of `depth - 128` tokens; the 262144 row is the earlier
measurement, where the prompt was 262144 tokens)

`ms/step = 37.8 + 0.202 us * depth` predicts 37.90 / 38.63 / 41.11 / 51.03 /
64.27 against the measured 37.91 / 38.59 / 40.76 / 49.50 / 64.34 - **the 37.8 ms
floor is the weight pass** (the whole N=1 decode step at depth 0) and everything
above it is attention, linear in depth with nothing to bend.  **At 256k, 43% of
the step is attention and 57% is the weight pass, and only the first grows with
context.**

**The prefill turns over at 16k and is strictly worse beyond it**: the average
rate of a whole prompt peaks at ~16k (1 477 tok/s for a 16 256-token prompt)
and falls to 1 399 at 64k and 1 242 at 128k, while **690.3 (-51.2%) at 256k**
(the earlier measurement).  So the prefill's best rate is at ~16k tokens of
context and the whole "2000 t/s" framing should be stated at a depth - a
384-token prompt gives 745 tok/s and a 16 256-token one gives 1 477, a 1.98x
spread from the context alone.

**Where the 131k prefill's time goes** (`dev/bench_conc` with a 131072-token
prompt, so the KV pool is sized right; the `PF_ABL_*` arms are wrong-answer-by-
design and these are **timing cells, not correctness ones**):

| arm | pp (t/s) | the prefill | share |
|---|---:|---:|---:|
| default | ~1148 | 114.2 s | - |
| `PF_ABL_NOMOE=1` (no routed experts) | 3044.8 | 43.0 s | **the experts = 62%** |
| `PF_ABL_NOGDN=1` (no GDN) | 1350.7 | 97.0 s | the GDN = 15% |
| `PF_ATTN_XMX=0` (the classic attention) | **103.9** | **1261.4 s** | the oneDNN int8 attention is worth **10x** |

**So the long-context prefill is the routed experts again** - the same wall as the
batched decode - and the oneDNN int8 attention is worth an order of magnitude at
this depth, which is why the turn-over at 16k is not the attention's fault but
the point where the experts' depth-independent share stops shrinking.

### How to measure here: the traps that cost the most time

* **The oracle ladder, and each rung's failure.**  A **token-stream hash** is
  too weak (the sampled output survived a 1.9-max logit difference).  An **`l1`
  norm** is not an oracle either - it is one number and says nothing about
  *where* the difference is.  A **per-layer fingerprint** is *worse than both*:
  `PF_DUMP_LAYERS` does a host->device->host round trip **inside** the forward,
  so it *supplies a missing synchronisation* - and a diagnostic that supplies the
  missing sync will reliably "fix" a missing-sync bug.  **That is how the whole
  `attn_xmx` locus was invented**: the fingerprint said the divergence was in
  the attention, and it was in fact the KV pool's *write* side.
* **An instrument that changes the schedule moves the bug it measures.**  The
  per-call checksum (`PF_ACTQ_SHAPE`) added a third `int64_t` per slot to
  `PF_KVSTORE_SUM` and the divergence moved from "the prefill's first qkv GEMM"
  to "the decode step".  Both readings were true of their own build, and
  neither was true of the shipped one.  **The only sound instrument is one that
  provably cannot change the schedule** - for a kernel, that means not touching
  it - or, failing that, an oracle read **strictly after** the region under test
  has completed and been waited on.
* **A/B subtraction is not attribution on an in-order queue.**  The DFlash2
  per-op profiler charged 16.8 of 20.8 ms to "norm + attn_conv_proj + attn
  conv0" and reported the FFN as free; with no-op ablations the removal landed
  in the wrong place, and with the heavy kernels removed the remaining ops
  reported **0.0 ms** - not because they were free but because nothing blocked.
  **A timing breakdown has to be checked against an independent total before it
  is believed** (the DFlash2 timeline's `devspan` is that check: `sum ~= devspan`
  should be printed first, not summed last).
* **The box drifts.**  The same binary and the same cell measured 41.2 ms and
  then 46.5 ms for tg128 at depth 512, ~13%, hours apart.  **Only compare cells
  measured in the same run**, and **"best" is the statistic** - a single 65536-
  token prefill is ~47 s and four reps of the shipped build measured a best of
  1437.4 against a mean of 1156.1.  Quote best-of with that caveat, never the
  mean, and never a single rep.
* **The harness served a stale binary for several rounds.**  `dev/build_*.sh`
  did `set -euo pipefail` and then sourced `setvars.sh`, which dereferences
  unset variables and returns non-zero - so the script died *inside* the source,
  the tool was never relinked, and the **old binary ran and measured the
  previous `libsycl_infer_core.a`**.  Nothing in the output looks wrong;
  `bench_pf` printed a plausible t/s.  `dev/build_tool.sh` now sources with
  `-e`/`-u` off and asserts the binary's mtime against the library, so a failed
  relink is a hard error.  **In a harness whose whole job is to measure the
  current code, "the tool still runs" is not evidence that it is current.**
* **A 64k cell must be measured against a correctly sized pool.**  Every depth
  cell above 16k was once run against a truncated block table, and the argmax
  those runs reported was reading device memory at a negative offset.  The
  throughput numbers were unaffected (the same number of keys were processed)
  and **the only reason it was caught is that the whole output was diffed, not
  the number being watched** - four targeted ablations all came back "still
  churns" and none of them was pointed at the pool.
* **A pool sized from one sequence's length is not a pool.**  `bench_pf` and
  `bench_conc` set `cfg.max_seq` from one sequence's token count, so
  `kv_cap_mb = -1` ("reserve what `max_seq` needs") reserved the pool for one
  slot: N <= 16 fitted inside the 512-block default by luck and N=20 died at slot
  18 with "KV pool exhausted".  `max_seq` and `n_blocks` now scale with N, and
  `alloc_block()`'s -1 is still checked.
* **One process per cell, and the heavy things serially.**  A prefix-cache hit
  inside one process moves the acceptance enough to change the ranking (1.9
  cold vs 2.8 warm), and stacking two GPU jobs on the box makes a card drop out
  with `UR_RESULT_ERROR_DEVICE_LOST`, which looks exactly like a code
  regression.  `set -u` / `set -e` must come **after**
  `source /opt/intel/oneapi/setvars.sh`, and a backgrounded server needs
  `setsid` (not just `nohup`).
* **The prefill's nondeterminism is closed as not-a-fix, and it is priced.**  It
  needs `attn_xmx`, it is invisible to the argmax and to the token stream, and
  its run-to-run spread (max 1.9 / rms 0.37 at 16k) is **the same order as the
  int8 attention's own approximation** (1.68 / 0.34), so the jitter is not
  separable from the approximation it replaces.  The oneDNN axis is an
  exhaustive negative result (host threading, accumulation precision, warmup,
  uninitialised memory, every ordering edge), and the remaining suspect is a
  **per-layer mixed quantisation** (the KV store's own scales), not oneDNN.
  `PF_DUMP_LAYERS` is not a valid arbiter for it - see the oracle ladder above.

## Invariants and gotchas

The rules that are not obvious from the code, and what breaks when they are
broken.  Each one is a thing that *was* broken, or that fails silently.

* **Each child process has exactly one owner, and the owner is the only one
  that waits for it.**  The router's supervisor thread `waitpid`s its own
  worker; nothing else may `waitpid` the same pid.  Two `waitpid`s on one pid
  is a race: the loser sees `ECHILD`, reads it as "the child is gone", and
  SIGKILLs a worker that is still shutting down gracefully - which loses the
  prefix cache flush, silently.  This was measured: the worker's log stopped
  at "server listening" with no "[srv] shutting down" line.  The regression
  case is `test_router_control.cpp`'s `test_graceful_shutdown`: the fake
  worker deliberately outlives the shutdown request by two seconds (standing
  in for the flush) and the test asserts it was allowed to finish.

* **A worker that cannot start says so with exit code 78, and the router
  does not retry it.**  78 is `sysexits.h`'s `EX_CONFIG` ("the process cannot
  start because of its configuration or its environment"): the model does not
  load, the card is out of memory, the arguments are wrong - the same attempt
  would fail the same way, so retrying only fills the log.  The router's
  rule is: exit 78 -> not retryable; otherwise the *phase* decides (a death
  before the worker was ready is a startup failure, a death after it is a
  crash and is retried).  The phase rule is the fallback for a worker that
  dies without the code (the OOM killer, for one).  78 is also what an
  external supervisor (systemd, docker, k8s) sees, so a restart policy can
  act on it.

* **A child's VRAM footprint has to be counted where it is allocated.**  The
  router's admission decision is the *measured* free memory, and the Level
  Zero backend does not implement the free-memory aspect - worse, where it
  does report one it does NOT count the SYCL allocations (it said 13 GB free
  on a card holding 9 GB of weights).  So the engine counts its own device
  allocations (`dev_alloc_on` plus the oneDNN partition's `weight_bytes()`,
  which allocates the weight stores itself), and the control plane reports
  `total - used` as the free memory.  That figure matches the hand-computed
  capacity table in this file (card 0 of the 27B: 9072 weights + 1484 state +
  1024 KV = 11501 used, 3973 free of 15474).

* **`compute_backend::synchronize()` must actually wait, and the CPU backend's
  used to be an empty override** - which was a heap use-after-free, not a slow
  path.  `cpu_backend.cpp` is compiled with `-fno-sycl` (it is in
  `SI_CPU_SOURCES`, outside the device pass), so it has no queue to wait on and
  the override compiled fine while doing nothing.  `engine::sync_all()` has ~24
  call sites, so on a CPU partition every one of them was a no-op, and the
  prefix cache handed a temporary `std::vector` to an asynchronous
  `queue::memcpy` and returned - the vector was freed while the copy was still
  in flight.  It surfaced only as `test_pc_cpu` failing, and the diagnostic
  trail is worth remembering: the KV comparison passed (36/36 blocks identical)
  while the *logits* differed, because the corruption was in a later host->device
  copy, not the KV one.  The fix is a waiter handed in at construction - single
  device `[this]{ q.wait(); }`, multi-device `[this, i]{ dev_queue(i).wait(); }`
  - and **`dev_queues_` cannot be used directly for the CPU index** because a
  CPU partition's entry is null and it segfaults.  **Generalisation: an
  override that exists only because the base class declares it is a silent
  no-op waiting to happen - if a virtual has no effect in one backend, assert
  it, do not leave it empty.**  A narrow test will not find this class of bug:
  `test_pc_cpu` originally compared only three layer-0 K blocks, and widening
  it to 36 units across all block-offset phases is what finally localised the
  fault.

* **A heterogeneous `--layer-map` is supported, not an error.**  A layer map may
  span GPUs that resolve to different `si::dev::profile`s (e.g. A770 + Iris Xe).
  Each device gets its own profile resolved per queue, and the engine reports the
  mix at startup under `PF_DEVICE_INFO=1` rather than refusing.  This matters
  because launchers must use `si::dev::for_queue(q)` and
  `wg_clamped_for_queue(q, want)`, never `active()` / `wg_clamped()`: those
  resolve once per process from the *first* GPU, so a heterogeneous split would
  mis-tune the second card *and* could launch a work-group it cannot accept
  (A770 accepts 1024 threads, Iris Xe only 512), which is a launch failure, not
  a slowdown.  The lookup keys on `sycl::device` equality and costs 0.039 us per
  launch (measured, `test_dev_resolve`) against 3.2 ms for the old
  `get_devices()`-per-call version.  For the same reason, do **not** put a
  profile read behind `static const = []{ ... active(); ... }()`: that pins the
  whole process to whichever card resolved first.  Use
  `si::dev::profile_int` / `profile_flag`, which cache the (device-independent)
  env once and read the default per call.  Four such statics existed and had to
  be converted - see `docs/design/12-build-and-testing.md` §3.4.1 for the list
  and for the A/B that keeps `rmsnorm`'s per-call lookup honest (the DFlash2
  draft block forward had 16.4 of its 20.5 ms inside five attn_norm launches,
  which is why that one was a static in the first place).

* **"The projector loaded" is not "the projector can run".**  `vision_model` (and
  the audio tower) will happily load a GGUF whose geometry the kernels do not
  implement - the remote mmproj files all have `head_dim 72` where only 64 is
  implemented, or a `proj_dim` that does not match the text width - and then
  abort on the first forward, *after* the model is loaded and after 23 checks
  have already passed.  `vision_model::runnable()` exists so a caller can ask
  that question up front and report "loads but cannot run" instead.  **If you
  add a geometry constraint, add it to `runnable()` as well as to the forward's
  check**, or the distinction is invisible.

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

* **The recurrent state's slot axis is `state_slots_`, not `kMaxB`.**  Every
  `[layer][slot][...]` stride into `d_gdn_state`/`d_conv_state` is built from
  `state_slots_` (the clamped `engine_config::max_slots`, default 16), and the
  allocation is `state_slots_ * n_gdn_local * per_slot`.  The two differ
  whenever `max_slots != kMaxB` - **and they differ by default now**, because
  the ceiling is 32 and the default is 16 - and mixing them is a silent
  out-of-bounds write into a USM pool, not a slowdown.  `zero_slot`, `pc_admit`,
  `pc_capture_begin` and `pc_commit` all bound-check against `state_slots_`,
  and `scheduler::loop`'s slot search only offers slots below
  `engine::max_slots`, so the index can never legitimately reach past the
  arrays.  Everything else in the engine still addresses *rows* by `kMaxB`
  (`step_info`, the block table, `d_logits`, `d_partials_dec`) because those are
  per-sequence, not per-state, and are small.

* **YaRN's two blends are written to be exact at their anchors, and that is not
  cosmetic.**  `yarn_build` combines three sequences (`interp`, `extrap`, and a
  ramp mask) and both blends are deliberately written in difference form
  (`blended + (extrap - blended) * (1 - m)`) rather than the equivalent convex
  form (`a*m + b*(1-m)`).  Each is exact at one anchor and carries an ulp of
  the *larger* term at the other: the convex form makes the untouched high-
  frequency head a 1-ulp ramp, and the wrong difference form makes the fully-
  stretched tail wrong by an ulp of `extrap` - which, at factor 32, is ~2e-6
  relative instead of 0.  `test_yarn` asserts both anchors bit-exactly against
  the plain table, so a "simplification" to the convex form fails there rather
  than in a 1M-token generation.

* **A plan is a device-pointer snapshot**.  `build_plan` writes the *then
  current* `d_x*` member pointers into each `gemv_seg`, and multi-device
  `bind_acts(dev)` only rebinds those members (it never rewrites a built plan).
  So anything pinned to the primary device (today: the LM head) must be built
  *after* `bind_acts(0)`, otherwise it captures the last layer's device buffers
  and the batch-1 decode head reads a buffer no kernel wrote for that step -
  while the prefill (which goes through `run_head()`) stays correct.

* **The multi-device LM head must reach the primary device's oneDNN table, and
  `cur_dev` must be reset to 0 before it.**  `gemv_at` reads
  `dnnl_for(cur_dev)` and `cur_dev` is the last layer's device after the layer
  loop, so it is reset to 0 at the final `handoff_x`, and the head's plan call
  gets an `xq` entry (without one `gemv_at` cannot take its `dnnl_call`
  branch).  Miss any of these and the head silently falls through to the fp32
  dequant GEMV: **12.3 vs 3.4 ms/token** (27B, 2x A770 measured).  The head's
  oneDNN key is the host pointer for an **untied** head (`output.weight`
  exists; `setup_md_dnnl` converts it before the upload, so the raw copy is
  skipped and `wkey` returns the host key) but the **uploaded device pointer**
  for a **tied** head (0.8B: `m.output == m.tok_embd`), which must keep its raw
  GGUF rows for the embed kernel and is therefore converted after the upload -
  see `setup_multi_device`.

* **`handoff_x` copies only the live rows** (`nrows * nreal`, the activation
  layout is `[token][n_embd]`), so copy the whole `kMaxB*kMaxT` block and a
  single-token decode pays 10.5 MB per handoff, twice.

* **A partial (per-device) `record_forward` must start its call cursor at the
  phase's first layer.**  `ci` indexes the plan **global** `calls[]`, so a
  phase must start at its first layer's call index - `seg_plan::layer_c0`
  (layer -> first call, with a final entry for the head).  Restarting at 0
  makes the later partition read the first layer's call metadata (activation
  pointer, K) while executing its own segments: `dnnl_call` goes false, the
  group falls back to the fp32 `gemv_group`, and the layers silently write
  nothing (one repeated token, ~6x slower from the NaN-heavy hidden state).

* **Multi-device decode is graphed per partition** (`build_md_dec_graphs`, one
  graph per contiguous device run), replayed with the host handoff between them;
  `PF_MD_GRAPH_DEV=N` limits it to device N for A/B against the direct replay.
  It is worth ~1-2 ms/token only, so per-kernel dispatch is *not* the layer
  GEMV's main inefficiency.  **These graphs serve `n_rows == 1` only** - the
  batched decode has no graph, because oneDNN's int8 matmul (the dense layers'
  path above M=13) cannot be recorded into a SYCL command graph.

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
  `PF_ATTN_XMX` as evidence about the XMX unit.

* **oneDNN int8 prefill attention (`attn_xmx.cpp`, `PF_ATTN_XMX`, default on
  above 2048 keys)**: QK^T and PV run as plain oneDNN int8 matmuls over the
  paged KV gathered into a contiguous scratch.  Four things are load-bearing: a
  plain int8 matmul sums over k, so a per-key scale cannot be applied after it
  - K and V each carry ONE block-wide scale, folded in at gather time; the query
  tile stacks every HPG query head sharing a kv head, so the matmul width is
  `HPG * tokens` and the KV is read once per kv head, not once per query head;
  the matmuls always run at a FIXED width `kBlk = 8192` (zero-padded), because
  oneDNN primitive creation is ~15 ms per new shape and a per-chunk `N` made
  prims never reusable - that alone was a 12x regression; and the block-max
  reduction in the gather must be spread over many work-groups, never one (see
  the measured-results section).  Scope: i8 KV only, head_dim 256, `n_real > 1`;
  other cases fall back.

* **A SYCL stream built from an in-order queue is NOT ordered against that
  queue's own submissions, and the attention matmuls were on the wrong side of
  that.**  `attn_qk`/`attn_pv` used to execute on a second oneDNN stream
  `st_a`, created by the same `sycl_interop::make_stream(p->eng, q)` call as
  `p->st`, on the stated theory that "it wraps the same in-order queue, so the
  matmul is ordered after the Q-quantize / K-gather kernels without an explicit
  wait".  **That inference is a category error: a stream is its own ordering
  domain.**  SYCL orders work *within* a stream; two streams are unordered
  regardless of how the queue was created.  In `attn_xmx`'s per-block-chunk loop
  the next iteration's `xmx_gather` (main stream) overwrote the `b.k`/`b.v` rows
  the current iteration's `attn_pv` (`st_a`) was still reading.  The fix is one
  line - the matmuls run on `p->st`, where every dense GEMM and every SYCL
  kernel already lives.  `PF_ATTN_STREAM=1` reproduces the split for A/B.
  **Generalisation: "same queue" is not an ordering guarantee, and a comment
  asserting one is worse than no comment.**

* **`ggml_float_to_half` used to flush the whole f16 subnormal range to zero,
  which silently destroyed weight-store scales (`src/common/quant.h`).**  The f16
  subnormals represent `[2^-24, 2^-14)` with ten mantissa bits; the old code
  had `if (exp <= 0) return sign;`, i.e. it threw away a thousand-fold range.  A
  Q4_K super-block whose `d` is *itself* subnormal then produced `d*sc` below
  `2^-14`, so `pack_q4_K` wrote a **step plane of identically zero** and the u4
  reconstruction of that column collapsed onto the constant `-dmin*m`.  Measured
  on the 27B: `blk.16.attn_qkv.weight` has `d = 4.47e-6` (f16 `0x004b`) in the
  offending super-block, and **21 of its 10240 columns (0.205%) sat at up to
  240% relative L2** against the exact GGUF dequant while every neighbouring
  column measured the expected 0.08% (the f16 rounding of the two constants).
  `blk.1`/`blk.2`/`blk.4` of the same tensor family had **zero** such columns,
  so it is data-dependent and invisible in any aggregate.  The same flush reached
  `w8.cpp`'s int8 per-group scales, the k5/cb4 and 2-bit packs, and the KV
  scales.  `test_f16conv` (hermetic, no model) pins the bit patterns; the old
  implementation fails it 68 ways, and it also asserts the *range* property
  ("nothing in `[2^-24, 2^-14)` may encode as zero") rather than sampled
  points, because a sampled test misses a boundary that moves.  **Lesson: the
  converter kept truncating (not round-to-nearest) so that nothing on the normal
  range moved - a correctness fix here did not need to be a numerics change.**

* **A test criterion built on a per-element *relative* yardstick flags
  cancellation, not defects.**  A dot product of K=17408 terms has an absolute
  error set by the terms (`|dy| ~ eps * sum_k |w_k x_k|`), so an output that
  nearly cancels to a small fraction of the tensor's max still differs between
  two *correct* weight stores by their full ~1%.  `test_w4_vs_i8`'s original
  `|dy| > 0.05 * max(1, |y|)` flagged 1 element in 163840 on one tensor and 2 in
  163840 on another while the tensor-level figure stayed at 1.6% / 0.9%.  It now
  screens against the *tensor's* output scale and adjudicates the survivors (and
  each tensor's worst element) against the exact fp32 dot product of the same
  quantized activation, judged on `|dy|` vs `sum_k |w_k x_k|`.  That criterion
  is what found the `ggml_float_to_half` defect above, so the rewrite is
  load-bearing rather than cosmetic - and `PF_W4_NOCORR=1` (drop the u4
  zero-point correction) is its red half: `diff/accum` 0.10-0.15 against ~0.01
  for a correct pair of stores, so the 0.05 threshold is not a loosened bar.

* **`nat_gemm_launch`'s int8 (FMT 3) and cb4 (FMT 2) paths need the
  decode/verify activation form, not just "an activation".**  `gemm()`'s int8
  nat call is gated on the activation's split form (like the u4 one): a mode-1/
  mode-2 prefill quantizes with `do_split=false`, so the grouped views it would
  read are the *previous* call's.  Missing that gate silently corrupted the
  0.8B multi-device decode-vs-prefill (`gen0`/`gen1`/`ref` 220/220/567 against
  248068/271/271) while every single-device test passed, and it also made
  `PF_MD_SPLITS=1` look broken.  `PF_NAT=0` is the bisection knob; the u4 path
  (FMT 0) already had the gate.  **This class of bug can no longer be
  written**: `quantize()` returns an `si::act_view` (`{m, k, split, gen}`) and
  every GEMM entry point *requires* it, so "forgot the gate" is a signature
  that does not compile, and a view that went stale is one integer compare
  (`view_is_current`) rather than a re-derived `M`/`K` check.  `split` is a
  field of the view, so a GEMM can no longer read the even/odd planes off the
  object and be wrong about whether they are there.  A stale view returns
  `nullptr` from the `act_*` accessors (they used to hand back whatever was in
  the scratch), so a caller that forgets to re-quantize now faults loudly
  instead of reading the previous call's data - that is how the one real
  instance of this shape was found, in `test_w4_gemm`.

* **The two drafters share exactly one piece of device state, and it is
  allocated as a pair.**  `engine::verify_scratch` owns the verify cycle's
  argmax staging (`d_argmax` + `d_argval`); every other drafter buffer is
  MTP-only or lives in the nested `dflash_state`.  The pair matters because
  MTP's candidate readout (`PF_MTP_CAND`) needs the *value* side and decides
  that from its own `mtp_cand_cap_` - so a value side that could be absent
  while the id side existed would let it write through a null pointer.  Each
  drafter used to carry its own `if (!d_argmax)` block and **only MTP's
  allocated both**, which is a bug rather than a style note: had DFlash run
  first in a process that then ran MTP with `PF_MTP_CAND=1`, the write would
  have gone through null.  `test_spec` never caught it because it deliberately
  runs **one drafter per process** (a second engine in the same process inherits
  state from the first).  Call `mtp.verify.ensure(q)`; do not allocate these
  buffers at a call site.

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
  9, 18, 27, ... 71 s apart, aggregate 14.5 tok/s = exactly one request's
  rate, with 188/188 decode passes at `nb=1 active=1`).  The engine calls now
  run with `m` **released** (a `std::unique_lock`; `e.mtx` still serialises
  the engine), which lets admission and the pre-existing multi-row decode
  batch.  Measured on the 27B (temperature 0, `ignore_eos`, 128 tokens each):
  N=1/2/4/8 -> 14.3 / 24.5 / 40.1 / **55.0 tok/s**, i.e. **3.85x aggregate at
  N=8**.

* **Concurrency costs bit-exactness**: the same temperature-0 prompt fanned out
  to N concurrent requests returns coherent, *deterministic* text that can
  differ from the single-request result - the flip happens at a near-tie
  argmax, once a sequence joins a batch (measured: 1 of N rows is byte-exact,
  the rest flip at the same "attention mechanism" / "**Scaled Dot-Product
  Attention**" tie ~20 tokens in; the multiset of outputs is identical across
  repeated runs, so it is accumulation-order numerics, not a race).  A sequence
  admitted first and decoded alone keeps the exact early tokens; batching only
  changes the *order* of the fp accumulation in the batched GEMMs.  That also
  means a test must **not** assert two concurrent greedy runs of one prompt
  agree: `test_sched` checks they both finish and honour `max_tokens`,
  nothing more.

* **The scheduler's token-output path had no coverage at all until
  `test_sched`.**  `test_spec` drives `engine::generate`, which never enters
  `scheduler::loop`, so the one place that turns a sampled token into a
  streamed piece - duplicated between the head prefill's first token and every
  decode row - was untested, and a refactor of it had no oracle.  `test_sched`
  (model-backed, no HTTP) covers: a greedy run, the logprobs variant (pieces
  carry an id and a logprob), `max_tokens`, an `ignore_eos` run that must
  report `"length"`, cancellation via `sequence::cancel` (the SSE disconnect
  path), an empty prompt (must retire rather than wedge), two concurrent
  sequences, and **a twenty-row batched decode**.  It earned its keep
  immediately: factoring the duplication into `sample_and_emit` initially
  returned one bool meaning both "was EOS" and "should retire", and the decode
  path's `&& !eos` reason test then reported `"stop"` where it must report
  `"length"` - a user-visible `finish_reason` regression.  The helper returns a
  `sample_outcome { eos, done }` for exactly that reason.

* **Cross-sequence prefill batching is NOT enabled** (it corrupts).  Packing
  several prompts into one mode-2 forward (row-major over the concatenated
  tokens, per-row `slot`/`pos`/`n_real_row`, one `prefill_text` per batch) was
  implemented and produced *garbage* - degenerate continuations, not plausible
  alternatives: `1000000000...`, `| 10 | 10 | 10 |` - even with the decode
  batch capped at 1, and `step_info` already carries per-row
  `slot`/`pos`/`active`, so the fault is in something reading plan-time state
  rather than `info`.  Prefill therefore stays one sequence at a time (each
  ~300 ms); the 3.85x above is decode batching alone.  Fixing it is worth
  doing - at N=8 the serialised prefills are ~2.4 s of the 18.6 s wall - but it
  needs the same "N concurrent identical requests must return byte-identical
  output" harness, run with the decode batch capped at 1 to isolate it.

* **A partial mode-2 batch requires the oneDNN weight path.**
  `batched_prefill_fit` only allows a last row with `n_real_row < kMaxT` when
  `use_dnnl` or a multi-device GPU partition has oneDNN (`dnnl_any_dev()`); the
  md_int8 fallback's dp4a GEMM (and its per-device w8/x8 layout) is only
  correct for a full `kMaxT` row, and letting it take a partial batch
  **silently corrupts** the hidden state (a 2-GPU md_int8 decode-vs-prefill
  mismatch).  Likewise an **all-`cpu` `--layer-map`** must select the CPU queue
  (`resolve_device`), not the default GPU one.

* **The fully-fused mode-2 GDN call used to process only the first chunk row**
  (fixed).  `record_forward` mode 2 (chunk-batched prefill) runs the GDN
  recurrence once over the whole batch (`PF_GDN_FUSE=2`) with `n_rows=1` and
  `tpb_arg = nreal_arg = <total tokens>`, but the GPU `gdn_kernel` /
  `gdn_f4_kernel` ignored `nreal_arg` and used `row_nr(info, rr)`, so only row 0
  (`kMaxT` tokens) got the recurrence and every later chunk row read a stale
  state.  The CPU `cpu_gdn` already honoured `nreal_arg`, so a GPU+CPU hybrid
  map behaved differently.  It hit *single-device* mode-2 as well; it was
  invisible only because the multi-device cap (`PF_PFB_MAX_M`, old default
  `kMaxT`) forced the scheduler onto the mode-1 chunked path, and because
  `dev/cmp_pfb` compares hidden-state cosines (0.998) rather than the first
  token.  Symptom: a prompt of >=64 tokens produced wrong logits (63 was
  correct - the scheduler used mode 1 for `rem < 2*kMaxT`).  Fixed in `gdn.cpp`
  (both kernels now use `nreal_arg > 0 ? nreal_arg : row_nr(info, rr)`), the
  cap default is removed, and `PF_GDN_FUSE=1` is a working fallback.

* **The scheduler now lets mode-2 prefill the tail** (`>= kMaxT` instead of
  `>= 2*kMaxT`).  A mode-1 chunk costs ~0.26 ms of handoff/sync *per token
  slot* on 2x A770 (a 41-token tail was ~0.34 s), so a 553-token prompt used to
  pay a full extra mode-2-sized forward.  Correctness is unchanged because the
  fused-GDN bug above is fixed; measured cold pp512 of a 553-token prompt went
  327 -> 399 t/s.  Mode-2 still costs ~0.26 s of *fixed* weight-stream per
  forward, so a 512-token batch is the efficient unit.

* Do not include `sycl/ext/oneapi/dot_product.hpp` from several TUs (its
  functions are not `inline` in this toolchain) — use `src/common/dp4a.h`.

* `kMaxT` = max prefill chunk (32), `kMaxB` = max batched sequences (**32**,
  the compile-time ceiling; `--max-slots` (default 16) is the policy),
  `kBlockSize` = KV block size (32), `kMaxSplits`/`kMaxDecSplits` bound the
  attention split buffers (`src/backend/gpu/kernels/kernels.h`).  Buffer sizes are
  sized from these constants; raising them affects device memory.

* The int8 KV layout is `[block][kv head][token][head_dim]` data plus a separate
  fp16 scale plane; both pool and scales are advanced in bytes
  (`engine::kv_layer_stride`, `kv_scale_stride`).

* `ld.bfd` warnings about `libsvml.so`/`libimf.so`/`libintlc.so.5` needed by
  `libdnnl.so` are benign (resolved from the oneAPI runtime path at run time).

* **The HTTP API has no authentication, so the bind address and the remote-media
  fetch are the security boundary.**  Both default to closed: `--host` is
  `127.0.0.1` (pass `0.0.0.0` to expose, behind a reverse proxy), and
  `image_url`/`video_url`/`audio_url` `http(s)://` fetches need
  `PF_MM_URL_FETCH=1`.  With it on, `src/server/url_fetch.cpp` requires every
  address the hostname resolves to to be publicly routable
  (`PF_MM_URL_ALLOW_PRIVATE=1` lifts that for an internal media server, and
  SSRF exposure again) and walks redirects **by hand** - httplib's
  `set_follow_location` re-resolves and re-connects without a hook, so a
  validated URL becomes an arbitrary one, which is the usual bypass.  Two traps
  the test pins: resolve rather than string-match (decimal `2130706433` and
  octal `0177.0.0.1` are loopback), and classify an IPv4-mapped IPv6 by its
  embedded address (`::ffff:127.0.0.1`).  Residual and *not* claimed fixed:
  the address is checked once and httplib resolves again to connect, so a
  hostile DNS server can still win that race.

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
  callback's return value is the only cancellation point inside a forward, so a
  cancelled generation still finishes the decode step it is in (~65 ms).

* **Multimodal**: an image consumes `max(nx, ny)` positions, not one per token,
  and the text model uses **interleaved** M-RoPE (`rope.dimension_sections`, e.g.
  `[11,11,10,0]`) where the pair index picks the temporal/row/col position
  (`src/backend/gpu/kernels/qk_norm_rope.cpp`).  Image tokens never reach
  `tok_embd`: the embed kernel copies `step_info::img_embd[img_row[t]]` instead.
  Video blocks add a temporal axis (M-RoPE `(frame,row,col)`) and audio blocks
  are a pure time stream (`(pos,0,0)`, `n_pos = n_out`); both reuse the same
  `img_embd` rows and `img_row` mapping.  The multimodal path bypasses the
  prefix cache and uses the single-sequence `engine::generate_mm`; `kMaxImgTokens`
  bounds one request's merged tokens.  Because an image consumes `max(nx, ny)`
  positions for `4*nx*ny` tokens, the KV slot index (token count) and the RoPE
  position diverge: decode keeps `info->pos` = token count and carries the
  running RoPE position in `step_info::mrope` with `mrope_on` set.

* **A MoE model's expert strides are per *layer*, not per tensor, and the file
  decides.**  The 35B A3B has Q4_K gate/up everywhere but Q5_K down for 37 layers
  and **Q6_K down for 34/38/39**.  One `plane_bytes` per tensor family reads
  the wrong weights on exactly those layers - it loads, it answers, and it is
  wrong in a place no aggregate catches.  `moe_slot_{g,u,d}b[]` are indexed by
  layer for this reason; say the layer count and the distinct values out loud at
  startup so the mixed case cannot be silent.

* **oneDNN eligibility is per *call* and all-or-nothing - one f32 tensor takes
  the whole call down with it.**  GDN's `ssm_beta`/`ssm_alpha` are f32; sharing
  a plan call with the convertible `wqkv`/`wgate` made the entire call fall back
  to the fp32 dequant GEMV.  That is 12.3 vs 3.4 ms/token disappearing for a
  shape reason, not a numerics one, and it looks exactly like "the kernels got
  slower".

* **A host pass that fills a sparse buffer has to run *before* its consumer,
  not after.**  `moe_down_launch` returns on `slot < 0`, so `d_moe_acc`'s
  non-resident rows still hold the previous layer's output, and `moe_reduce_launch`
  weights **every** row.  Adding the host contribution after the reduction
  therefore double-counts a stale row and then adds the right answer on top.  The
  signature is unmistakable and was read as a dequant bug for hours: **layer 0
  exact because the buffer is still zero, 2-4x too much routed energy from
  layer 1 on**, worst relative sumsq 3.29 -> 0.0088 once the order is right.
  Per-layer fingerprints, not a layer-0-only dump, are what showed it.

* **A per-layer diagnostic that reports one layer is a verifier that agrees
  with the bug.**  `PF_MOE_REFDUMP` dumped layer 0 only, so the mixed-
  quantization defect above and the stale-accumulator defect here both looked
  "fine" at layer 0 while every later layer was wrong.  It now dumps all 40 with
  `PF_MOE_REFDUMP_ALL`, and each layer's name carries its index because two
  dumps of the same name are indistinguishable.  **A timing or fingerprint
  breakdown has to be checkable against an independent total before it is
  believed** - this is the third time that rule has paid for itself here.

* **The GGUF's expert tensors are three *separate* tensors, so a packed cache
  entry changes where `up` starts.**  `gate_exps.data + e*plane_bytes +
  plane_bytes` is expert e+1's gate plane, not expert e's up plane.  A RAM tier
  that packs `[gate][up][down]` per expert therefore needs different base
  pointers from the mmap path; giving the mmap path the packed ones evaluated
  every expert against expert 0's weights.  Symptom: the split runs fine with
  the RAM tier off and emits garbage with it on.

* **A command graph cannot contain a host pass, and every replay site has to
  know it.**  The ncmoe host half has blocking transfers and an expert set
  chosen per step.  There are **three** places a forward is replayed from a
  graph (plain decode, multi-device decode, MTP verify) and each one was missed
  in turn; a recorded graph does not merely skip the pass, it silently drops
  every expert without a device slot.  Grep for `moe_host_split()` - it is the
  whole rule.

* **A page-release sweep that asks a component about itself has to run *after*
  that component has decided.**  `release_host_weight_pages()` sweeps from
  `alloc_buffers()`, which is where `setup_moe_slots()` also runs - **earlier**.
  The keep-set asked `moe_host_split()` and got `false` because the cache had not
  been initialised, so the expert pages were `MADV_DONTNEED`d and then re-faulted
  one row at a time by the host half.  It is not incorrect, so nothing failed;
  it is unobservable without measuring the RSS release.  `release 21603 -> 2396 MB`
  is the number that says the keep-set actually applied.

* **MTP's draft layer does not share the trunk's activation buffers.**  The draft
  runs out of `d_mtp_xnorm` (and writes `d_mtp_x`), so a host pass that hardcodes
  `d_xnorm` evaluates its experts against *another layer's* activation.  Every
  kernel-launcher boundary that takes "the activation" needs the caller to say
  which one - `moe_cpu_experts` now takes the buffer as an argument for this.

* **A routed MoE model runs on the CPU backend, and the host half has to mirror
  the device kernels exactly.**  `record_forward`'s MoE branch is dispatched on
  the *partition's kind* (`dev_kind_[dev] == 1`), not on `multi_dev`: an
  all-`cpu` run and a `--layer-map` with a `cpu` tail both take it, and calling
  the SYCL launchers unconditionally is a startup failure ("No kernel named
  moe_router_logits"), not a slow path.  `src/backend/cpu/kernels/moe.cpp` mirrors
  `moe_router_logits`/`moe_gate_up`/`moe_down`/`moe_reduce`/`moe_gate_add` step
  for step - same rank-counting top-k (ties to the lower index, no sort), same
  `max(sum, 6.103515625e-5f)` renormalisation, and the experts stay *unweighted*
  because the reduction multiplies.  `PF_MOE_CPUDBG` prints one line per layer:
  `dev / il / n_tok / ids0 / sum_w / rms_x / rms_acc / nan`.

* **A handoff to a CPU partition has to be waited, because that partition's
  kernels are not on the queue.**  `handoff_x` ended with
  `dev_queue(to).memcpy(as_[to].x, buf, bytes)` - *unwaited*.  The comment above
  it says the destination copies before its own kernels because it is an in-order
  queue, which holds for SYCL kernels enqueued there and not for a CPU partition,
  whose kernels are host `par()` workers with nothing to order them against.  A
  GPU+CPU `--layer-map` therefore raced the copy against the first host read of
  the receiving layers' activations.  The signature is the one to remember,
  because it reads exactly like a kernel bug: the GPU partition's layer was
  **bit-identical** in decode and the next layer's MoE input was 63% too large,
  and only the *decode* handoff showed it (the multi-device prefill runs
  pipelined with explicit flushes), so the prefill matched and the first
  generated token was already wrong.  `gpu+gpu` and `cpu+cpu` were both
  correct, which is what ruled out the host kernels.  A diagnostic that
  dereferences `d_xnorm` on the host is a SEGV here for the same reason - on a
  GPU partition it is a *device* pointer - and it must be read through the queue
  and skipped while a graph is recorded.

* **A recorded decode graph has the previous pass's device bindings baked into
  it** - the host side must rebind on entry.

* **A diagnostic that loops `n_expert` times *inside one work-item* is a
  parallelism bug, not a cost bug.**  `moe_router_topk` computed each expert's
  rank by scanning all `n_expert` experts to count how many beat it - O(n^2),
  which is fine - but it launched `range<1>(n_tok)`, so a **decode's 65536
  comparisons ran in a single thread**, twice (ids and weights).  Ranks are a
  permutation of 0..n_expert-1 because ties break on the index, so the loop
  parallelises across experts with no coordination at all.  Measured on the 35B
  A3B the router - 0.5 M-MAC of real work - was **4.4 s of a 4.6 s decode**.
  Fixed: one work-item per (token, expert).  `router=4393 -> 48 ms`, and the
  same prompt's decode went 2.13 -> 21.58 t/s.

* **K-parallel reduction changes the router's fp summation order, and MTP at
  k>=2 is sensitive to it.**  Same measurement on the 35B, `gen` of "The capital
  of France is": mtp 0/1 unaffected, mtp 2 and mtp 4 change their output.  The
  perturbation is one ulp in the router logits, which only moves *marginal*
  draft decisions.  It is shipped because the decode is 10x and the prefill 1.6x
  faster, and because MTP was already documented as not byte-guaranteed - but it
  is a real quality trade at k>=2, not a free win.

* **A routed-MoE model's prefill is still the largest open item.**  The grouped
  dp4a path took pp512 from 248.2 to 851.8 t/s, but the routed half is still
  ~3x short of what the same model's dense layers do, and the remaining levers
  are listed in the measured-results section (the tile's register budget, the
  weight representation, the roofline gap).
## Quick verification before finishing a change

```bash
cmake --build build -j$(nproc)          # must be warning-free (except the linker notes)
./build/test_gpu_stages                 # all stages OK
./build/test_gpu_vs_ref                 # argmax SAME
./build/test_forward                    # default prompt predicted_argmax=248068
./build/test_decode_vs_prefill          # decode == re-prefill (any single-token change)
./build/test_sched                      # the batched decode, incl. the 20-row case
clang-tidy -p build --extra-arg=-I/opt/intel/oneapi/compiler/2026.1/include -checks='-*,misc-include-cleaner' <changed files>
```

If the change touches `src/router/`, the router's own tests are the ones that
matter, and they need **no GPU and no model**:

```bash
cmake --build build --target sycl-infer-router test_router_config test_router_proxy test_router_supervisor test_router_control -j$(nproc)
ctest --test-dir build -R router --output-on-failure      # ~15 s
ldd build/sycl-infer-router | grep -c sycl                 # must be 0
```

The whole router can also be built **without SYCL at all** - that is the
container image's build path, and it is what proves the router needs neither a
GPU nor the oneAPI toolchain:

```bash
cmake -S . -B build-router-only -DSI_ROUTER_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-router-only -j$(nproc)
ctest --test-dir build-router-only --output-on-failure    # the 4 router tests
```

`test_sched` is the one that covers a **batched** decode, and its last section
decodes twenty rows and requires every row's logits to move between two steps -
the oracle for "the engine computed every row".  It has been seen red (on the
engine that quantised the row count), so it is a verifier and not a decoration.

For the 27B (needs the layer split - it does not fit on one card), also run:

```bash
M=/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
TEST_LAYER_MAP=0-31:gpu.0,32-63:gpu.1 ./build/test_w4_vs_cpuref      # argmax SAME
TEST_LAYER_MAP=0-31:gpu.0,32-63:gpu.1 ./build/test_decode_vs_prefill "$M"  # OK
```

`test_decode_vs_prefill` has three env hooks for exactly this kind of change, and
they are the only coverage `--max-slots` and `--yarn` have on the decode path
(it is the one test that exercises a per-*slot* stride and a rope table at the
same time):

```bash
M=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf
TEST_MAX_SLOTS=1            ./build/test_decode_vs_prefill "$M"  # same argmaxs, less VRAM
TEST_YARN_FACTOR=4.0        ./build/test_decode_vs_prefill "$M"  # table live, margins move
TEST_LAYER_MAP=0-11:gpu.0,12-23:gpu.1 TEST_MAX_SLOTS=1 ./build/test_decode_vs_prefill "$M"
```

Run all four on the **A770** box rather than a laptop iGPU: a Debug build pays a
one-time multi-minute JIT of the whole device image per process (no persistent
cache, see *AOT builds*), so one 3-length case takes ~13 min on an Iris Xe and
seconds on an A770.  Measured on the A770 (0.8B, `TEST_DVP_LENS=96,1056,2080`),
all three lengths OK in every configuration:

| config | result |
|---|---|
| single device, default | 3/3 OK |
| single device, `TEST_MAX_SLOTS=1` | 3/3 OK, **argmaxs identical to default** |
| single device, `TEST_YARN_FACTOR=4` | 3/3 OK, same argmaxs, margins differ |
| `0-11:gpu.0,12-23:gpu.1`, default | 3/3 split OK |
| the same + `TEST_MAX_SLOTS=1` | 3/3 split OK, **argmaxs identical** |
| the same + `TEST_YARN_FACTOR=4` | 3/3 split OK, margins differ |

That table is the reason to run it: `--max-slots` must be numerically invisible
(it only changes an allocation) and `--yarn` must be numerically *visible* (its
margins move) while leaving the greedy argmax alone at these lengths.

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
