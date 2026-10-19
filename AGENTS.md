* **The routed experts do not use the int8 GEMM path, and that is the whole gap.**
  **Status: half of it is closed.**  A grouped dp4a expert GEMM over the *native*
  GGUF blocks now ships and is the default (`PF_MOE_DP4A`), taking pp512 on the
  35B A3B from **248.2 to 851.8 t/s** with an identical argmax - the rest of this
  note (the 2000 t/s projection) is the remaining half, and the mechanics and the
  two register-locality bugs that had to be fixed first are in *"The routed-expert
  prefill"* below.  The original diagnosis follows.
  `setup_md_dnnl` converts the *dense* tensors to SIn int8 with `build_w8_one`
  and the plan takes the oneDNN branch for them.  The MoE `addp`-uploads its
  expert tensors raw, so 18.22 GiB - 90% of a MoE prefill - runs the fp32
  dequant kernel while 2 GiB of dense weights runs int8.  Measured on the 35B
  A3B, same prompt, warm, one config apart:

      dense only, int8 (PF_ABL_NOMOE)            2610.2 t/s   (0.169 s)
      dense only, fp32 (PF_ABL_NOMOE PF_DP4A=0)    94.1 t/s   (4.689 s)

  **27.7x for the same work.**  Projected against the current 239 t/s (of which
  the MoE is 90%), moving the experts onto int8 lands at ~2000 t/s, which is
  where a 3B-active model belongs next to the 27B's 2390.

  Why the obvious routes are closed, so none of them has to be re-derived:
  * `nat_gemm_launch` (the batched native-width GEMM, 221 GB/s at M=7) **rejects
    M > 13**, and a 452-token prefill routes 3616 pairs over 256 experts, so M
    per expert is ~14 - just outside.  Its oneDNN path would serve it, but ...
  * u4 stores for the experts need 0.625 B/w x 18.22 GiB/0.5625 = 10.1 GiB per
    card *on top of* the 9.1 GiB raw upload the decode path still reads, against
    15.11 GiB usable.  And `add_weight_w4` builds a prefill primitive ladder per
    weight, so 256 experts x 3 tensors x 40 layers = 30780 ladders at ~15 ms of
    primitive creation each is not a load time.
  * So the shape is: **expand the active experts to int8 in a reused scratch**
    (~768 MB per device, one layer at a time) and drive it from the existing
    activation view.  The scratch makes the weight pointer identical for every
    layer, so one oneDNN primitive per (M, K, N=512) shape is reused across all
    80 calls instead of 30780 - the same trick `attn_qk`'s fixed `kBlk = 8192`
    uses.  The gather of each expert's routed rows into a contiguous
    [p_e, K] tile, and the scatter back, is ~585 MB of traffic per prefill
    against a ~25 ms DRAM floor, so it is free.

  ### How OpenVINO structures it (looked up, and it matches the shape above)

  The Intel GPU plugin's MoE prefill is six ops: `moe_mask_gen` (per-expert
  token lists) -> `moe_gather` (gather the routed tokens into per-expert
  *contiguous* tiles) -> `moe_gemm_up` + bias -> `swiglu` -> `moe_gemm_down` +
  bias -> `moe_scatter_reduce` (weighted scatter back).  They state the reason
  for two micro-GEMM paths explicitly: **prefill is compute-bound, decode is
  memory-bound**, so they are different kernels.  Newer builds route the gather
  through oneDNN's *grouped* matmul (`dnnl::matmul` grouped) with an
  expert-major packed layout, and they warn that the token sort is
  single-threaded and "a clear bottleneck" for long sequences.

  So the structure is the same one this file already says is needed.  What is
  worth adding from their PRs: the gather makes each expert's activation tile
  contiguous, which is what lets the micro-GEMM tile (rows x K) and reuse it
  across the expert's output rows.

  ### Why the register blocking that should do it does not, measured

  `x`'s row for a routed pair was re-read once per output row, which is 15 GB of
  L2 traffic per layer against a 302 MB weight stream.  Ablating it (pin every
  pair to token row 0, leaving the weight stream byte-identical) takes gate_up
  1083 -> 591 ms, so the re-read is 45% of that kernel.

  **The gate_up ablations, in one place.**  Same 452-token prefill, each variant
  measured with the other costs still present:

  | variant | gate_up | cost removed |
  |---|---:|---:|
  | as shipped | 1909 ms | - |
  | no dequant ALU (`PF_ABL_NODEQ`) | 1054 | 855 ms, 45% |
  | no activation re-read (`PF_ABL_NOL2`) | 1632 | 277 ms, 15% |
  | both | **413** | 1496 ms, 78% |

  With both removed the residue is 413 ms against a ~60 ms fp32-FMA floor and a
  ~30 ms DRAM floor, i.e. still **5-7x off the roofline**, so this kernel is
  latency/issue-bound and not roofline-bound.  The dequant is the bigger half and
  it is ~3 ALU ops per value (`d * sc[b] * nib - dmin * mm[b]`, with the 6-bit
  scale unpacking counted), which is why the int8/dp4a rewrite is the real lever:
  the Q4_K nibble *is already an integer*, so feeding it to dp4a and applying
  the per-32 group scales afterwards looks like it cuts those ~3 ops to ~1.

  **It does not work, and the reason is worth having before re-deriving it.**
  Built and measured end to end on the 35B A3B: the dp4a kernel is *correct* (same
  argmax, same text) and **13x slower** than the fp32 kernel it replaces, and the
  mechanism is the group scales, not dp4a.  With `xq_launch`'s int8 activations the
  dot is

      sum(w*x) = scale * (A * sum(nib*q) + B * sum(q))

  so every 32-value group costs two extra global loads (`xmeta`'s scale and
  `xsumq`'s two half-sums) and three dependent float ops *per (row, pair, group)*
  - 64 groups a row - and those loads are re-issued for every one of the expert's
  ~512 output rows because they are pair-indexed, not row-indexed.  That is the
  activation re-read one level down, and it costs more than the dequant it removes.
  The lane split has to change too (a warp reading all 128 bytes of qs in one
  instruction needs each lane to own 8 consecutive elements, which is not the
  32-lane x 8 split the nibbles were unpacked for).  A faster version has to hoist
  the group scales into registers before the row loop, which makes the row loop
  not-innermost, and that is what spills the accumulators.

  The obvious fix is kMoER rows per warp, which is free on the ALU side (the
  dequant-to-FMA ratio is 1/(2*kMoEPB) and *independent* of kMoER).  It measures
  **worse**: 1170 -> 1801 ms at kMoER=4 and 1742 ms at kMoER=2, because
  `ag[kMoER][kMoEPB] + au[kMoER][kMoEPB]` is 2*R*PB float registers and both
  spill, and the spill traffic costs more than the re-read it removes.  The
  [R][PB] accumulator array is the wall, and it is the same wall for down.
  Getting both reuses at once needs the activation tile in **SLM** rather than
  registers - which is what their micro-GEMM does (BM x BK tile staged once,
  reused across BN output columns).  Attempted: stage one tile of
  `kMoEPB x kMoEKTC` floats (8 x 1024 x 4 B = 32 KB) in SLM, shared by the
  work-group's eight warps because they all serve the same expert and hence the
  same routed pairs, with the per-pair accumulators left at `[kMoEPB]`.  It is
  **both wrong and slower** (2084 vs 1170 ms, and the logits are garbage), and
  the reason not to retry it is already in this file: the router's one-work-
  group-per-token version - logits in SLM, softmax and top-k after a barrier -
  "never took effect" on the A770's Level Zero backend while passing under
  OpenCL and on the Iris Xe.  That is this backend's codegen for
  SLM-plus-barrier shapes, not the algorithm, and a barrier per K-chunk is
  exactly that shape.  So the OpenVINO micro-GEMM structure is not reachable
  through SLM here, and a GEMM rewrite that avoids barriers (or one that is
  routed through oneDNN's grouped matmul, which is a library call and not our
  codegen) is the only way to get both reuses.

  What is *not* the problem, each measured and reverted: SLM row staging (139
  vs 148 t/s - the reuse is real but one sub-group per work-group throws away
  8x the occupancy), pair-tiled dequant reuse `kMoEPB` 16/32 (worse, register
  spill), and row-blocking `kMoER` on its own (gate_up 1170 -> 3430 ms; the
  dequant-to-FMA ratio is 1/(2*kMoEPB) and *independent* of kMoER, but without
  the pair tiling it collapses to 1:1 and the dequant is 10 FMAs' worth of ALU
  per FMA).  Row-blocking only pays if the pair tiling is kept, and then the
  [kMoER][kMoEPB] accumulators spill.  `PF_MOE_TIME` is the tool for this; the
  router was a red herring for two whole attempts and was 46% of the prefill.

  **`PF_MOE_TIME` measures one kernel in isolation and the whole prefill
  disagrees - trust the wall clock.**  Raising `kMoEPB` 8 -> 40 cuts the *measured*
  gate_up 1909 -> 1570 ms, because the dequant then runs once instead of once per
  8-pair tile.  End to end the same change takes the prefill **239 -> 146 t/s**:
  `ag[kMoEPB] + au[kMoEPB]` is 2*PB registers, and at PB=40 that is 80 of them,
  so the occupancy loss hits every other kernel in the forward too.  A sweep of
  PB = 8/12/16/20/24/28/32/40/48/64 all agreed with the wall clock in the end:
  8 is the optimum, and the 8-tile's 2x dequant redundancy is cheaper than the
  registers that would remove it.  Reverting the PB sweep back to 8 is
  load-bearing - do not "fix" the 7x roofline gap by widening the tile.

  **OneDNN's grouped matmul is not in the public 3.11 API, and its substitute is
  memory-blocked.**  Checked: the headers carry only `dnnl_graph_op_matmul` and
  the plain `dnnl::matmul`; `DNNL_ARG_HINT_MAX_GROUP_SIZE` and the grouped
  `dnnl::matmul` that OpenVINO's MoE PRs use live in their *fork*.  The
  fixed-shape reuse the int8 entry already does (one primitive per (M,K,N) shape,
  which is `attn_qk`'s fixed `kBlk = 8192` trick) does still apply here.

  The other route, `nat_gemm_launch` (the batched native-width GEMM, 221 GB/s at
  M=7), needs no oneDNN primitive at all and *works* with the experts: packing
  each layer's expert bank as one `[n_expert*n_ff][K]` tensor is 3 `add_weight_w4`
  calls a layer, and `gemv_only=true` skips the prefill ladder.  It registers
  (40 u4 + 20 k5 per card) and is then **memory-blocked**: u4 is 0.625 B/w
  against raw Q4_K's 0.5625, so keeping both costs 9.1 + 10.1 = **19.2 GiB per
  card against 15.11 GiB usable** - confirmed, `UR_RESULT_ERROR_DEVICE_LOST`.
  Two details worth not re-deriving: the stores must be registered *after* the raw
  upload, because the upload skips any tensor a oneDNN table already holds and the
  MoE decode path still reads these tensors' GGUF blocks (registering first
  strands it on an unuploaded pointer); and `nat_gemm` rejects M > 13 while a
  452-token prefill routes 3616 pairs over 256 experts (~14 each), so the pairs
  must be tiled either way.  So that route is a full weight-representation swap -
  replace the raw upload with the native store and route decode through the
  native GEMV too - not a prefill-side change.

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

`--max-slots N` bounds the concurrent sequences, which is what sizes the
**recurrent** state; see *Long context: what actually fits* below for why that
is the cheapest context headroom there is (1.125 GiB per card at the default 16).

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

**`--max-slots` is the cheapest headroom there is.**  `gdn_state` is
`kMaxB * n_gdn_local * (dt_rank*d_state*d_state)` f32 = 16 x 24 x 48*128*128 x 4
= **1.125 GiB per card**, and a single-stream long-context run uses one slot.
Every `[layer][slot]` stride is built from `state_slots_` (the clamped
`max_slots`), *not* `kMaxB` - the two differ from this change onward and mixing
them is a silent out-of-bounds write into a USM pool, not a slowdown.  What else
keeps addressing rows by `kMaxB` (`step_info`, the block table, `d_logits`,
`d_partials_dec`) is per-*sequence*, not per-state, and is small.

| card 0 | KV budget | i8:i8 | i4:i8 | i4:i4 |
|---|---:|---:|---:|---:|
| as shipped (`kMaxB=16`) | 4.46 GiB | 275k | 360k | 519k |
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
`kMaxB`=16 - 1.125 GiB per card on the 27B, so this is the cheapest context
headroom there is),
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
the section below), `PF_MOE_CFG` / `PF_MOE_CFG_DOWN` (tile shape `M x C x NSG` of
`moe_grp_impl`, plus the `HOIST` A/B in cases 16-21, selected **per projection** -
gate+up is `FUSE==2` and reads `PF_MOE_CFG`, down is `FUSE==1` and reads
`PF_MOE_CFG_DOWN`; the defaults are `M=8 C=2 NSG=16` for both, and
`dev/bench_grp` is the sweep - cases 40-57 are the register-budget sweep).  Note that
`dev/grp_min` drives the **down** path, so exactness runs must set
`PF_MOE_CFG_DOWN` - that is how a staged kernel under test once read as exact.
`PF_MOE_FLAT` / `PF_MOE_FLAT_DOWN` (put the row tile in the grid - one
work-group = one tile - instead of looping it inside an (expert, column block)
work-group; the trade is not uniform, see the section below),
`PF_MOE_PERSIST` (replace the grid with `2 * compute_units` work-groups that
work-steal `(expert, row tile, column block)` triples off an atomic cursor, with the
item total summed on the device; **exact at every configuration, and 3% SLOWER -
refuted, see the section below, so leave it off**), `PF_MOE_PERSIST_WG` (its grid,
default `2 * compute_units`), `PF_MOE_GRPDBG` prints the resolved tile and grid and is the assertion on the
tile selection path; `PF_DNNL_FPMATH` (widen the oneDNN int8/u4 matmul's accumulation to `fpmath_mode::any,true`; a guard against the K >= 5376 NaN the A770 library records, **default off** because no model here reaches it and it measures neutral), `PF_MOE_DP4A_DEC=1` (run the grouped dp4a expert GEMM on a DECODE step too;
default off because decode is the wrong shape for it and the fp32 dequant path
is 1.29-1.38x faster at all three depths - see *Decode: the grouped expert GEMM
is prefill-shaped*), `PF_MOE_GRP_MIN_TOK` (prefill chunk size below which the
grouped kernel declines, default 8), `PF_MOE_DQ2` (halves the dequant
arithmetic with every load unchanged - **wrong results by design**, and the
experiment that bounds the f16 idea at 18%), `PF_MOE_NOACTQ` skips `moe_actq_launch` (wrong results -
diagnostic only; measured at 1.3 ms of the 501 ms gate+up phase, i.e. **the
activation quantizer is not part of the gap**).

## The routed-expert prefill: where the 8x actually is

**The 35B A3B prefills at 246 t/s and 90% of it is the routed experts.**  Measured
on the 2x A770 box, `--layer-map 0-19:gpu.0,20-39:gpu.1`, 512-token prefill,
`PF_MOE_TIME=1 PF_NOGRAPH=1` (`dev/pf.sh` is the harness; one process per cell):

```
[moe] phases: router=45.8 csr=6.3 gate_up=1489.7 down=367.1 reduce=5.0
              epilogue=0.0 sum=1913.9 ms   calls=40 n_tok=512 n_pairs=4096
[pf]  pp512  best 2.138 s (239.4 t/s)      with PF_NOGRAPH; 246 t/s without
```

and the same prefill with `PF_ABL_NOMOE=1` is **3081 t/s (0.166 s)**.  So the
budget for a 2000 t/s pp512 (0.256 s) is **~90 ms for the whole routed half**,
against 1914 ms today.  That is the whole problem, and it is not the arithmetic:
the routed MACs per prefill are 4.6e11, which is 30 ms at the machine's measured
7.6 T-MAC/s, and the expert weights are 18.1 GiB, which is 26 ms at the card's
376 GB/s.  **The roofline is ~30 ms and the work is ~64x that.**

Three facts that shape any fix, each measured:

* **The router is a GEMM that is not tiled, and it costs 46 ms** - 2.4% of the
  prefill, ~1.5% of the 90 ms budget.  `moe_router_logits` is one sub-group per
  (token, expert), so each of the 512 tokens re-reads the whole 256x2048 f32
  router weight: 1 GB of (mostly L2) reads per layer, 40 GB per prefill.  Tiled
  as a real `[M, n_embd] x [n_embd, n_expert]` GEMM it reads the weight once per
  M-tile and should cost well under 5 ms.

* **The routed-pair distribution is extremely skewed, and the row tile has to
  cover it.**  `PF_MOE_HIST` on the 35B at 512 tokens: **133 active experts of
  256, mean 31 pairs, max 124**, with 65 of the 133 at <=16 pairs, 95 at <=32,
  121 at <=64.  A kernel that tiles the rows by M re-reads an expert's weight
  `ceil(p_e/M)` times, so the weight stream is read **2.26x at M=16, 1.45x at
  M=32, 1.15x at M=64, 1.0x at M=128**.  Any expert GEMM here trades tile size
  against a weight re-read factor, and the factor is not uniform across M.

* **`nat_gemm_launch` cannot serve this shape at all**, which is worth stating
  because it looks like the obvious reuse: at the gate/up geometry (K=2048,
  N=512) it launches `N/(NSG*C) = 16` work-groups, and `dev/moe_probe` measures
  it at **5-8 GB/s** - 512 columns is simply not enough work-groups for a
  512-EU card.  It is the right kernel for `N=17408` (1088 work-groups) and the
  wrong one here, whatever the M limit says.

**A grouped dp4a expert GEMM is implemented, is numerically exact, and **ships,
default on** (`PF_MOE_DP4A`, set `=0` for the old fp32 path;
`src/backend/gpu/kernels/moe.cpp`, the "grouped dp4a expert GEMM" section).  It
quantizes the layer activation to
int8-with-per-32-group-scale, then runs dp4a against the *native GGUF* expert
blocks - Q4_K's 32-value group is 32 bytes of nibbles, which is already two dp4a
operands, so there is no expansion pass and no extra weight storage.  One 8-lane
slice takes one 32-value scale group (lane l owns values 4l..4l+3 = exactly one
dp4a operand), so no lane repeats another's work and the only cross-lane step is
one `sg_sum<8>` per output.  The asymmetric zero point costs one extra dp4a:
taking the activation's group sum over the lane's OWN four values
(`dp4a_s8u8(x, 0x01010101, 0)` - a multiply by one is a dp4a) makes the
correction per-lane, so it needs no broadcast and no divide-by-lane-count, and it
is correct for Q6_K's 16-value scale group as well as Q4_K/Q5_K's 32-value one.

It is **exact**: `dev/grp_min` compares 16 rows x 512 columns against the host
GGUF dequant and reports **0/8192 mismatches**, and that is the shipped default's
own configuration.  It is **3.4x the prefill**: pp512 on the 35B A3B / 2x A770
goes **248.2 -> 851.8 t/s** with a bit-identical argmax (11171; 4821 in the same
table below is pre-router-fix and the two are different quantities), pp64 188.7 ->
334.8 t/s (argmax 3790), and `PF_MOE_DP4A=0` reproduces the old number exactly.

| | fp32 dequant (`PF_MOE_DP4A=0`) | grouped dp4a (default) |
|---|---:|---:|
| pp512, 35B A3B, 2x A770 | 248.2 t/s | **851.8 t/s** |
| router / csr / gate_up / down / reduce | 43.8 / 5.6 / 1299 / 574 / 4.8 ms | 21.3 / 6.1 / **272.8** / **147.4** / 5.1 ms |
| kernel, gate+up (K=2048 N=512 Q4_K) | - | 6.03 ms, 58.7 GB/s |
| kernel, down (K=512 N=2048 Q5_K) | - | 3.39 ms, 63.7 GB/s |

**It is still ~3x short of the 2000 t/s budget** (routed half 449 ms against
~90 ms; see the budget arithmetic above), so the rest of this section is the
remaining gap rather than the remaining win.

Four correctness bugs, each of which looks like something else.  All four were
found by
`dev/grp_min`, a 170-line exactness check, and each one *ran*:

* **`nd_range<1>(G, L)` takes a TOTAL work-item count, not a work-group count.**
  `(ngrid, TX)` therefore launches `ngrid/TX` work-groups - one here.  The kernel
  then reads the same 32 KB from L1 for every column block (every column block
  hits the same activation rows), which is why the first working version
  measured *faster than the bandwidth roofline allows* (2.7 TB/s): it was one
  work-group doing 1/128th of the work.  The tell is that the output is
  plausible and partial.
* **A GGUF group index is row-global, and the offsets inside a block are not.**
  `qs[32*(g>>1) + 4*lane]` and `get_scale_min_k4(g, ...)` are right for the
  first superblock and wrong for every group after it - the first eighth of the
  dot product comes out exact and the rest is the next rows' bytes, a ~3% error.
  It must be `g & 7`.
* **`sg_sum<N>` hardcodes a 32-lane mask ladder** (`permute_group_by_xor` with
  masks 16, 8, 4, 2, 1).  On an 8-lane sub-group the first two masks are out of
  range, which is undefined behaviour; the visible effect was 99% of the
  work-groups producing no output at all.  Use `sg_sum<8>` or
  `reduce_over_group`.
* **The CSR's pair list is built by an atomic scatter, so an expert's rows are a
  permutation of its range, not `pair[i0+m] == pair[i0]+m`.**  Assuming contiguity
  misplaces every row of every expert that selected more than one pair, and the
  store is the only place that can tell.

And two properties of this backend that constrain the next attempt, both
measured:

* **`local_accessor` over ~34 KB per work-group is rejected**: 16384 bytes runs,
  34816 fails with `UR_RESULT_ERROR_OUT_OF_RESOURCES`, 65536 fails too.  The
  activation tile for one expert at M=16, K=2048 is 32 KB + 2 KB of scales, i.e.
  *just over* the limit, so SLM staging of the tile has to chunk K or shrink M -
  it cannot simply be switched on.
* **An empty kernel is not the launch cost, and neither is a 128-float frame.**
  `dev/bench_grp` measures `0.006 ms` at 16 work-groups and `0.083 ms` at 65536,
  so the grid size is free; the same 16384-work-group grid with this kernel's body
  cost 26 ms, which read as "the body/register frame".  It was neither.  A
  `dev/probe_frame.cpp` sweep (frames of 8/32/64/128 floats, and 32/64 unrolled
  ALU iterations) is **0.020 ms at 16384 work-groups for every one of them**, so
  neither the private frame nor the unrolled code size explains it.

**The 26 ms was a broken ablation, and a broken ablation is worse than none** —
the same failure mode as the `PF_DFLASH_OPTIME` misdifferencing further up this
file.  `PF_MOE_ABL`'s "immediate return" was `if (PF_MOE_RDBG && lid == 0)
{ ...; return; }`, so **127 of 128 lanes fell straight through into the real
body** and the measurement was of the body, not of a return.  The condition has
to hold for every lane; `PF_MOE_GRPDBG2=1` alone is 0.01 ms.  Two rules, both
worth keeping: an ablation that short-circuits a *subset* of the work-items is
measuring the complement, and **a stub that is faster than the real kernel means
the stub is wrong** — here it "proved" a 26 ms launch cost that did not exist.

**What it was, once the ablation was real: a runtime-indexed array, which SYCL
places in private (scratch) memory.**  `moe_grp_impl`'s row loop is bounded by
the *template* `M` and carries `if (m >= np) break;`.  The `break` stops the
unroller, so `m` is a runtime value at the point `acc[m][f][c]` is addressed, so
`acc` is not register-allocable, so **every one of the ~450 inner iterations is a
scratch load plus a scratch store instead of an FMA.**  Predicating instead -
`if (m >= np) continue;` - lets the unroller fold `m` to a constant and `acc`
becomes `M*FUSE*C` registers.  **Measured 51.6 -> 6.3 ms on the `down`
projection, 8x, from one keyword**, and the price is the dp4a for rows the expert
does not have, which `PF_MOE_HIST` puts at ~14%.  The same trap was hit twice in
this kernel: the superblock's `uint8_t sc8[FUSE][C][8]`, written by
`moe_sb_scale` *through a pointer parameter*, cannot be promoted no matter how it
is unrolled, so the eight group scales are now recomputed per group from the
block (2 extra L1 hits) and only `d`/`dmin` are hoisted, as scalars.

  **Generalisation, and it is not MoE-specific: on this backend an array indexed
  by anything the compiler cannot fold to a constant is private memory, and
  private memory is ~8x slower than registers here, silently.**  IGC says
  "compiled SIMD32 allocated 128 regs and spilled around 4" for a kernel that is
  100% scratch-resident, so **the spill report does not see it either** - the
  frame is not a spill, it never became an allocation.  A loop over a
  compile-time bound must be *predicated*, never `break`-ed.

**And the second, independent half: the row loop was inside the group loop, so
`pair[i0+m]` and the `top_k` integer division were re-executed for every
32-value group.**  They are per (tile, row), not per (tile, row, group), and the
inner body has only `FUSE*C` dp4a to hide them behind.  Hoisting them into a
`int trowv[M]` above the K loop (unrolled, so registers) took gate+up 8.94 ->
6.93 ms and down 6.03 -> 4.11 ms.

### SLM activation staging: tried twice.  Both times the "fast" version was wrong

The obvious next step is to stage the activation tile in SLM, because all `NSG`
sub-groups of a work-group load the *same* `M x K` bytes from global - 16x more
load requests than the tile has bytes.  It was implemented, and it measured
**4.9x faster: 1.82 ms vs 8.94**.  Both halves of that were false, and so was a
second reading of it, and the shape of the mistake is the part worth keeping:

* **it was wrong** - 7680/8192 mismatches, exactly one column block in sixteen;
* **the exactness check was silently testing a different kernel.**  The tile
  config is selected *per projection* (`FUSE == 2 ? cfg_gate : cfg_down`), so
  `dev/grp_min`, which drives `moe_grp_down_launch`, reads `PF_MOE_CFG_DOWN` and
  ignored `PF_MOE_CFG`.  Every "it passes" reading came from the un-staged path;
* **the 4.9x was a bug** - the K-chunk loop was bounded by `KC` instead of `K`,
  so at `KC=256` it processed 1/8 of the work.

Redone with exactness as the *gate* rather than a footnote, the picture is
cleaner and the negative is real (`PF_MOE_CFG`/`_CFG_DOWN` 30 and 32, both
0/8192):

| cfg | tile | exact? | gate ms | down ms |
|---|---|---|---:|---:|
| 0 | `M=16 C=2 NSG=16 KC=0` (shipped) | **yes** | **7.07** | **4.12** |
| 30 | `M=32 C=1 NSG=32 KC=512` | yes | 9.00 | 4.89 |
| 32 | `M=32 C=1 NSG=16 KC=256` | yes | 9.50 | 5.43 |
| 31 | `M=32 C=1 NSG=32 KC=128` | **8192/8192 wrong** | 1.15 | 1.08 |
| 34 | `M=64 C=1 NSG=32 KC=128` | **8192/8192 wrong** | 1.29 | 1.33 |

**The exact staged forms are 1.3-1.4x slower than reading global**, and `KC=128`
is the same "6x faster" artifact all over again - it computes almost nothing.
The `M=16` staged forms are wrong in exactly one row (512/8192 is one row of 512
columns) and `M=8` in two, so correctness at all requires `M=32`, which spills.
Cases 31/34 and the `M=16` ones are **deleted, not left as switches**: a knob
that returns wrong results 6x faster is the trap this file keeps warning about,
and it is indistinguishable from a win unless someone runs the exactness check.

Two things that had been believed on the strength of the broken version are now
corrected.  *SLM + a barrier per K-chunk is not pathological on this backend* -
`dev/slm_probe` isolates it and puts **stage+barrier at 0.012 ms of a 0.779 ms
pipeline (1.5%)**, with `full` (0.779) even ahead of `compute-only` (0.889), so
the earlier "a barrier per K-chunk is a shape this backend generates badly" is
false for the GEMM case.  *The activation re-read was never the bottleneck* -
`PF_MOE_ABL=8` (no weight reads, no activation reads, no dp4a) was within 1% of
the full kernel, which said so before this experiment and was ignored.  The
real cost of staging here is the `M=32` it forces, not the barrier.

### DPAS/XMX for the expert GEMM: re-measured at the MoE's M, and it still loses

Worth re-measuring because **the earlier rejection does not transfer**:
`dev/bench_dpas_gemm.cpp` rejected DPAS at **M=8** (the MTP verify shape), which
is *below* the arithmetic-intensity crossover - the memory system delivers
~13 MAC/byte and dp4a reaches ~18 there, so at M=8 neither path is
instruction-bound and a faster instruction cannot pay for itself.  The routed
experts are 1..124 pairs with a mean of 31, squarely inside the window where the
handover note says an XMX/DPAS unit should win.

`dev/probe_dpas_moe.cpp` reuses that probe's proven dpas form and operand layouts
and sweeps the MoE shapes with the expert in the grid (so the 128 MiB weight
stream comes from DRAM rather than L2 - one expert's 590 KB plane is L2-resident
and measuring it measures cache bandwidth, which is how the first attempt at this
got a nonsense 1 TB/s).  Reported per call, i.e. scaled by `sum_e ceil(p_e/M)`
over the measured histogram, so it is directly comparable with `dev/bench_grp`:

| projection | dp4a (shipped) | best DPAS |
|---|---:|---:|
| gate/up, K=2048 N=512 | **6.93** | 7.52 (NT=2, R=4, M=32) |
| down, K=512 N=2048 | **3.80** | 7.74 |

**1.08x and 2.0x slower.**  And the full `NT x R` sweep kills the idea that M
could be grown to pay for it:

```
NT=1: R=1 18.5  R=2 18.2  R=4 13.5  R=8 21.5
NT=2: R=1 13.9  R=2 10.4  R=4  7.5  R=8 29.6
NT=4: R=1 10.6  R=2  7.9  R=4 16.0  R=8 79.6
NT=8: R=1 10.8  R=2 20.3  R=4 53.0  R=8 111.8
```

M rises from 8 to 64 and gets better up to ~32 and then collapses.  The
tempting argument was that a DPAS 8x8 result is `simd<int32_t,64>` = **2 GRF per
lane** for 64 outputs while dp4a's `acc[16][FUSE][C]` is **64 GRF per lane** for
the same 64 - a 32x denser accumulator, so M would grow for free and take the
weight re-read factor from 2.26 to 1.05.  **The density claim is right and the
conclusion is wrong**: the sweep's cliff tracks `NT*R` (the accumulator in GRF/lane)
exactly as tightly as it tracks R, and the practical ceiling lands at 16-32
GRF/lane either way.  There is one register file, and DPAS does not get more of
it - it just packs it differently, which does not raise the ceiling.

Two structural notes that make the comparison bound rather than exact, both in
DPAS's favour, so the loss is not overstated: the probe's epilogue uses the same
`dsc` for the scale and the zero point, where the real Q4_K path needs both
(so the real one loads more), and it is a performance probe with no exactness
check.  The `u4` weight plane is also assumed pre-reordered into
`[group][n-tile][32 dwords]`, which is a one-time load-time cost - the 18 GiB wall
AGENTS.md already records for every other expert-store change.

### "Transpose the matrix to raise M": measured, and it is the wrong trade

`M` is the routed-pairs-per-tile and it is the one knob that reduces the weight
re-read factor (`ceil(p_e/M)`: 2.26 at M=16, 1.45 at M=32).  The register wall is
`M*FUSE*C` floats, so the obvious fix is to transpose the expert weights from
`[n][k]` to `[k][n]`, which lets one lane own one output column: `acc[M][FUSE][1]`
instead of `acc[M][FUSE][C]`, half the registers, no cross-lane reduction at the
end, and therefore room for `M=32`.  **It is not worth it, for two independent
reasons, both measured.**

*The weight read gets much worse.*  Row-major in k is exactly what makes the
current read coalesced: lane `l` takes k-values `[4l, 4l+3)` of **one** column, so
8 lanes read 32 *contiguous* bytes.  Transposing puts the 8 lanes on 8 different
rows, `row_bytes` (1152 B) apart.  `dev/probe_coalesce.cpp` holds the byte count,
the instruction count and the grid fixed and changes only that mapping:

| lane mapping | C=1 | C=2 | C=4 |
|---|---:|---:|---:|
| KAXIS (today: 32 contiguous B per lane-group) | **194.8 GB/s** | **157.4** | **126.1** |
| NAXIS (transposed: 8 strided loads) | 74.7 GB/s | 109.9 | 113.4 |

**2.6x worse at C=1** - which is precisely the configuration the transpose needs,
since C=1 is how it halves the accumulator.  The transpose buys a 1.56x smaller
weight-re-read factor (M=32 vs M=16) and pays 2.6x on the read: **net ~1.7x
worse**, before any cost of materialising the layout.

*And it cannot be materialised anyway.*  A transposed copy of the same bits is
the same 18 GiB, and the decode GEMV still needs the row-major original, so both
would have to be resident - the identical wall that already rejected the int8
expansion scratch and the u4 store.  A per-layer transposed scratch (486 MB) would
itself cost ~104 ms per card to build, more than the entire 90 ms MoE budget.

**But the deeper answer is that a transpose could not have raised M anyway.**
`M` loses on a different axis, and the formula says which:

    memory ops per (superblock, group) = M (activation) + FUSE*C (weight)
    dp4a per (superblock, group)        = M * FUSE * C
    dp4a per memory op                  = M*FUSE*C / (M + FUSE*C)

At `M=16, FUSE=1, C=1` that is **0.94**; at `M=32` it is **0.97**.  **Once
`M >= FUSE*C` the ratio is saturated** - the weight side is already fed (one
weight load feeds `M` dp4a) and the whole cost sits on the activation side (one
load feeds `FUSE*C` dp4a).  So widening `M` buys less weight traffic and *no*
extra dp4a density, which is the only thing that would pay for the registers.
The transpose's whole appeal was that it frees registers to widen the one knob
that is already saturated.

Measured, on `down` (FUSE=1, so the accumulator is `M*C` and M has its best
chance), all numerically exact at 0/8192:

| M | C | NSG | down (ms) | weight traffic | spills? |
|---|---|---|---:|---:|---|
| **16** | 1 | 32 | **3.80** | 1.00 | no |
| 24 | 1 | 32 | 5.49 | 0.70 | yes |
| 32 | 1 | 32 | 5.92 | 0.64 | yes |
| 16 | 1 | 16 | 3.80 | 1.00 | no |

**M=32 moves 1.56x less weight and is 1.55x slower.**  And "spills?" is not the
reason - that was the obvious guess and it is wrong.  `PF_MOE_CFG_DOWN` cases
16-21 re-run the same M values with the hoisted row dropped (`HOIST=0`, which
frees `M` integer registers and *does* remove `M=32`/`M=24` from the IGC spill
report, leaving only M=48/64):

| M | HOIST | down (ms) | spills? |
|---|---|---:|---|
| 16 | 1 | **3.80** | no |
| 16 | 0 | 4.02 | no |
| 24 | 0 | 4.18 | **no** (spill gone, still slower) |
| 32 | 0 | 6.37 | **no** (spill gone, still 1.68x slower) |
| 48 | 0 | 6.44 | yes |
| 64 | 0 | 12.04 | yes |

So the register pressure is real but it is not what caps M: **with the spill
eliminated M=32 is still 1.68x slower than M=16.**  What M buys is weight
bandwidth, and the kernel is not weight-bandwidth-bound - it is bound by the
dp4a-per-load ratio, which M cannot move.  **The lever that would move it is
`FUSE*C` (more columns/banks per activation load), not M** - which is exactly why
gate+up ships at `FUSE=2, C=2` and is the faster of the two projections.

`down` shipped at `M=16, C=1, NSG=32` on the reasoning that its accumulator
count is half and occupancy beats dp4a-per-activation-load there.  **Both
projections later turned out to want `M=8, C=2, NSG=16`**, and the reason is
register budget rather than any of the ratios argued for in this section - see
*What is left*.  The isolated measurements that motivated `C=1` were right about
that tile and wrong about the neighbourhood: 3.80 vs 4.13 ms for `C=2` at `M=16`,
against 3.38 ms for `C=2` at `M=8`.

### The bench harness silently served a stale binary for several rounds

Worth its own entry because it produced a wrong headline number twice in this
pass.  `dev/build_*.sh` did `set -euo pipefail` and then
`source /opt/intel/oneapi/setvars.sh`.  **`setvars.sh` dereferences unset
variables and returns non-zero**, so the script died *inside the source*, the
tool was never relinked, and the script's `|| echo "LINK FAIL"` fired - but the
**old binary still ran and measured the previous `libsycl_infer_core.a`**.
Nothing about the output looks wrong; `bench_pf` printed a plausible t/s.

`dev/build_tool.sh` now sources with `-e` *and* `-u` both off and asserts the
binary's mtime is newer than the library, so a failed relink is a hard error.
The general rule is the one AGENTS.md already states twice (for `rsync -a`
mtimes and for AOT/JIT flag agreement): **in a harness whose whole job is to
measure the current code, "the tool still runs" is not evidence that it is
current** - assert it.

**Context depth: the routed experts do not get less important, and the 2000 t/s
target was defined on the worst cell.**  Measured on the 35B A3B / 2x A770 with
`dev/pf.sh`, one process per cell (`NT` is the *total* prefill length, so
`NT=16384` is 32 mode-2 chunks at an average depth of 8k):

| cell | t/s | | routed experts removed |
|---|---:|---|---:|
| pp512 (depth ~0) | 851.8 | | - |
| pp4096 | 1376.7 | | - |
| **pp16384** | **1548.8** | | 4483.0 |
| **pp65536** | **1389.2** | | 3308.2 |

**Depth makes the *rate* worse than 16k, and both are far better than depth 0**
- the depth-0 cell carries the pipeline fill and the lowest arithmetic intensity,
  so pp512's 851.8 t/s is the floor of this model, not its typical number.  Anyone
  quoting "the 35B A3B prefills at N t/s" has to say which cell.

The routed half is **depth-independent**: `PF_MOE_TIME` measures 465.6 ms for a
512-token prefill and 455 ms per 512-token chunk at 16k and 448 ms at 64k, so it
scales with tokens and not with context.  Attention does scale, but from a much
smaller base.  `PF_XMX_BREAKDOWN` (the only attribution of the mode-2 chunk that
`PF_PROF` cannot do, because `PF_PROF` needs `PF_NOGRAPH` and that forces the
32-token path):

| cell | attention total | of the whole prefill |
|---|---:|---:|
| 16k | 1711 ms of 10860 | **15.8%** |
| 64k | 11094 ms of 48069 | **23.1%** |

So at 64k the routed experts are ~55%, attention ~23%, GDN ~10%
(`PF_ABL_NOGDN` 1518.5 vs 1363.5).  **Two things follow that are worth more than
any kernel work.**  *A DPAS attention targets at most 23% of the 64k prefill* -
the MoE's M is set by the 512 tokens in a chunk and does not move with depth, so
depth does not open the expert-GEMM window DPAS needs.  And *`PF_MOE_TIME` is an
upper bound, not a share*: at 64k its serialised total is 57.4 s against a
48.1 s wall clock, because its per-phase waits serialise a run that is otherwise
pipelined across the two cards.  **A measurement that sums to more than the wall
clock is measuring the instrumentation** - attribute by ablation
(`PF_ABL_NOMOE`, `PF_ABL_NOGDN`) or not at all.  (`PF_ABL_NOATTN` fails outright
at both depths.)

There *is* a real opportunity in that 23%, and it is not the DPAS unit's fault:
`attn_xmx`'s QK/PV are oneDNN int8 matmuls, and AGENTS.md records that this
oneDNN 3.11.4 resolves every one of them to `jit:gemm:any` with **no XMX kernel
compiled in at all** (all 498 `dpas` hits are `DUMMY_DPAS_*` JIT placeholders).
So the generic SIMD path is doing the attention, which is the one thing DPAS is
actually good at.  The obstacle is the same one that made
`dev/bench_dpas_gemm` 37x slower than `nat_gemm`: the cross-lane offset vectors,
not the arithmetic.  At depth the shapes are friendlier than that probe's
(M = 512 x HPG against a 64k-long K), so it is worth a re-measurement before it
is worth writing - but it is ~23%, against the routed experts' ~55%.

### DPAS at the attention shape: the s8 path does not exist on this part

The decision above rested on a previously-recorded number, so it was re-measured
directly: `dev/probe_dpas_rate.cpp` is a register-resident issue-throughput probe
(loop-constant operands, no gather, no memory, no offset vectors).  It is
deliberately *not* `dev/bench_dpas_gemm`, whose 37x loss is the offset vectors
and says nothing about the unit.

**The first version of that probe was wrong twice, and both errors produced a
plausible-looking number.**  It used the two-argument `dpas(B, A)` overload for
s8, which the header documents as `Result = A x B` — it does **not** accumulate
into C — so the loop body was a function of constants and the compiler hoisted
it: **2166 T-MAC/s**, eight times the u4 peak it was supposed to be compared
against.  And with only four chains it measured the dpas *latency* (one issue per
~6 cycles per warp), not its throughput.  Fixed to eight independent chains
through the three-argument accumulating form, with both arms measured the same
way on the same build:

| form | ms | G-instr/s | T-MAC/s |
|---|---:|---:|---:|
| `u4 x s8` (K=64) | 32.855 | 2.0 | **8.2** |
| `s8 x s8` (K=32) | 32.858 | 2.0 | **4.1** |

**The occupancy sweep rules out occupancy as the missing factor.**  `dev/probe_dpas_occ`
sweeps 4096 / 16384 / 65536 sub-groups at a fixed total instruction count and the `u4`
arm takes 34.886 / 34.886 / 41.894 ms - exactly proportional, so the unit is already
saturated at 8 warps/EU; 16 accumulator chains are *slower* than 8 (1.7 vs 1.9 G-instr/s).
And IGC is not scalarising: the `u4 K=64` arm does **2x the MACs of the `s8 K=32` arm in
bit-identical time**, which a scalarised version could not (it is the same signature that
exposed the zero-`dp4a` codegen in `nat_gemm`).  The absolute rate is still only ~1/10 of
dp4a's instruction rate (19.1 G-warp-instr/s at 8 MACs/lane against 1.9 G/s at 4096), so
3.8 T-MAC/s is a **lower bound** and not the unit's best case; resolving that needs
`ocloc disasm` on a device `.bin`, which this toolchain does not expose (`icpx` hides
`--emit-spirv` and `-fsycl-dump-device-code`).

**That lower bound is not the argument, because the argument is already built.**
`dev/bench_dpas_gemm` is a real GEMM on this unit: 7.44 ms against `nat_gemm`'s 0.200,
with the cost in cross-lane offset vectors and accumulator spills - neither of which
more peak would remove.  The unit's best case therefore does not decide the attention
question; the real GEMM already did.  What survives is that on this backend DPAS's int8
form delivers **3.8 T-MAC/s**, against

| | T-MAC/s |
|---|---|
| oneDNN int8 attention, QK (estimated from `PF_XMX_BREAKDOWN`) | ~8 |
| dp4a, the real kernel's rate | 7.1-7.6 |
| fp32 FMA peak | 5.0 |
| **DPAS s8 (K=32)** | **3.8** |

**So DPAS's int8 path is 2x slower than the oneDNN path already in use and 2x slower
than dp4a**, and dequantising the KV to f16 does not rescue it either - fp32 FMA peaks
at 5.0 and f16 buys about 2x for a conversion pass, against dp4a's 7.1-7.6 with no pass
at all.  **DPAS for attention is closed.**  It was only ever aimed at 10% of the 64k
prefill anyway, and AGENTS.md's own accounting says the routed experts are 55% of that
prefill and depth-independent - so the depth question is not an XMX question.

**What the attention actually runs at, and why there is no XMX headroom there.**
`PF_XMX_BREAKDOWN` at 64k, deepest call (1216 calls, 68158 cols/call): its absolute ms
are inflated ~2x because every stage is bracketed by a `q.wait()`, so only the
shares are meaningful - but the *rates* derived from them are ratios of work to
time inside one call and survive that:

| stage | ms/call | share of attention | share of the 64k prefill | rate |
|---|---:|---:|---:|---:|
| routed experts | - | - | **55%** | depth-independent |
| qk (oneDNN int8) | 4.324 | 27.9% | 6.4% | **8.3 T-MAC/s** |
| softmax | 3.876 | 25.0% | 5.8% | **180 GB/s** (48% of 376) |
| **gather** | **3.472** | **22.4%** | **5.2%** | **10.9 GB/s** |
| pv (oneDNN int8) | 2.494 | 16.1% | 3.7% | **14.4 T-MAC/s** |
| GDN | - | - | 10% | - |

(QK is `M x cols x head_dim` = 35.8 G-MAC at 2048 stacked query rows; PV is the
same MAC count reading a u8 operand instead of int32, which is why it is the
faster of the two.)

**So the matmuls are already the fastest thing in the prefill** - 8.3 and 14.4
T-MAC/s against dp4a's real-kernel 7.1-7.6, fp32 FMA's 5.0 and DPAS s8's 3.8.
There is no unused XMX capacity to reclaim: a DPAS attention would replace a
**14.4 T-MAC/s** PV with a **3.8 T-MAC/s** one, and even an infinitely fast DPAS
caps out at the 44% of attention that is matmul, i.e. **10% of the 64k prefill**.

**The softmax is already near its own floor, and it is a structural one.** It
reads the int32 score block and writes the u8 P operand: `M x cols x 5` bytes =
698 MB per call at 64k, which at 3.876 ms is 180 GB/s - 48% of the card.  The
only way below it is fusing softmax into PV (flash attention), and that trade is
*worse*, not better: it would save the 1-of-5 write of P but hand PV a **fp32**
operand instead of a u8 one, raising its read traffic 4x against the stage that
is currently the cheapest thing in the call.  **That is the whole reason the
path is split in two**, and it is a bandwidth fact, not a scheduling choice.

**The gather is the outlier, and by a factor of 16 to 35.**  At 10.9 GB/s it runs
at 1/35 of the bandwidth roofline while the softmax beside it runs at 48% of it,
and the reason is structural rather than arithmetic - see *The gather's second
stage* below.

### Decode: the grouped expert GEMM is prefill-shaped, and decode takes fp32

`tg128` on the 35B A3B / 2x A770 (`--layer-map 0-19:gpu.0,20-39:gpu.1`), measured
with `dev/bench_tg`, which prefills `depth` tokens and then times ONLY the decode
loop - so the depth is the variable and the prefill is not in the number.  It also
prints the generated token ids and a hash of them, which is a far better oracle
than one argmax.

| depth | before | after (decode on fp32) | |
|---|---:|---:|---:|
| 512 | 56.7 ms (17.6 t/s) | **41.2 ms (24.3 t/s)** | **1.38x** |
| 16384 | 59.6 (16.8) | **44.2 (22.6)** | **1.35x** |
| 65536 | 68.5 (14.6) | **53.0 (18.9)** | **1.29x** |

A decode step is `n_tok = 1` with `top_k = 8`, so every selected expert
contributes exactly **one** row and the kernel's `M = 8` tile is 7/8 empty.  The
grouped dp4a kernel is therefore 1.38x SLOWER on decode than the fp32 dequant
path it replaced, and `PF_MOE_DP4A_DEC` (default 0) now chooses per mode instead
of per process.  **Prefill is untouched**: pp512 / pp16384 / pp65536 stay at
860.8 / 1565.4 / 1437.0 t/s with argmax 11171 / 1204 / 4779, and the MoE exactness
oracle stays 0/8192.

**The decode budget, by ablation** (`PF_ABL_*`, one process per cell).  Note that
`PF_ABL_NOATTN` crashes at both depths, so attention is the difference between the
two "no experts" rows:

| | depth 512 | depth 65536 |
|---|---:|---:|
| routed experts | 39.6 ms | 39.5 ms |
| attention | ~0 | 12.1 ms |
| GDN | 1.7 | 1.7 |
| dense GEMVs + head + norms | 15.4 | 15.4 |

**The routed experts are 70% of a depth-0 decode step and 58% of a 64k one, and
they are depth-independent.**  They read `8/256` of an 18.22 GiB bank = 630 MB per
token, which at 376 GB/s is a **1.68 ms** floor, so they run ~15x off their own
bandwidth roofline.

**Two hypotheses about why, both measured, and the second killed a planned
rewrite.**

*The obvious one - a starved grid - is wrong.*  The grid is `n_act x N/RB` =
`8 x 16` = 128 work-groups on 512 EUs, which reads like half the machine idle.
Halving RB through the existing cfg cases doubles the work-group count and changes
**nothing** (56.8 -> 56.6 ms), while RB = 8 (512 work-groups) is 81.1.  The reason
is that `RB = NSG * C`, so the **total thread count `n_act * N * 8 / C` is
invariant under NSG** - the sweep traded work-group size for count and netted zero.
M is not in that formula either (each thread holds all M rows in registers), so no
tile shape changes the occupancy at all.

*And the occupancy curve then runs backwards.*  C is the only tile parameter that
does change the thread count, and it is anti-correlated with speed:

| cfg | C | warps/EU | ms/tok |
|---|---:|---:|---:|
| 45 | 4 | 0.5 | **44.7** |
| 43 (was shipped) | 2 | 1 | 56.7 |
| 44 | 1 | 2 | **81.4** |

**Fewer warps is 1.82x faster across a 4x swing in warp count**, i.e. steeper than
linear.  What the shape actually rewards is arithmetic per load -
`M*FUSE*C / (M + FUSE*C)` is 4.0 / 2.67 / 1.6 at C = 4 / 2 / 1 - so **a split-K
rewrite, whose entire mechanism is more warps, was pointed the wrong way and was
never written.**  The same ratio explains why the grouped kernel loses to the
fp32 one in decode while winning 3.4x in prefill: the two read very different
amounts of weight per lane (128 values per superblock against 16).

**And the decode stream was not reproducible — cause found, it was not in this
kernel at all.**  Five runs of one binary at depth 512 gave five different token
hashes at bit-identical timings.  It was **not** the `kv_cap_mb` harness bug
(`bench_tg` sets it), not the uninitialised-partial path, and not the MoE CSR's
atomic scatter.  It took a separate investigation and the answer was
**`moe_router_logits_tiled`'s missing `reqd_sub_group_size(32)`** — see the
router section below for the full account.  Two things about the hunt are worth
keeping, because both are instrument faults rather than engine faults:

* **The hash was not an oracle until the filler was made varied.**  `bench_tg`
  originally cycled ONE sentence, and the model fell into a fixed periodic
  continuation: `PF_ABL_NOMOE=1` — which removes the entire MoE FFN — produced
  **the same hash as the unmodified default**.  A bisection built on that would
  have "found" nothing and looked rigorous doing it.  27 distinct passages fixed
  it, after which every ablation moved the hash and the default varied 13 times
  out of 13.
* **The per-rep hashes are not a determinism oracle either, for a subtler
  reason.**  The prefill runs ONCE before the rep loop and each rep decodes 128
  further tokens, so rep 1's first token is read at position `depth` against
  rep 0's 128 tokens of KV.  Two reps of one process differ **by construction**;
  treating that as engine-state nondeterminism was wrong.  Only cross-process
  comparison of the *first* rep means anything.

The measurement that actually localized it was the cheapest one in the whole
hunt: print the prefill argmax **and its top1/top2 margin**, from three
processes.  The argmax alone reads as "flipped a near-tie"; the margin read
**0.04 to 1.28 across runs**, which says the whole logit vector is moving, and
that is a different bug class than a tie-break.  `PF_ABL_NOMOE=1` then pinned it
to the MoE path, and full-precision top-1/top-2 **values** separated "upstream
noise amplified by discrete routing" from "something reads memory it never
wrote".  After the fix, three processes give one hash.

**`PF_MOE_DQ2` answers "would dequantising to f16 be faster", and the answer is
bounded at 18%.**  A Q4_K/Q5_K value is `d * sc[b] * nib - dmin * mm[b]` - two
multiplies and a subtract per value - and f16's entire contribution is doing those
in half precision, with no precision question at all because `d` and `dmin` are
already f16 in the block.  `PF_MOE_DQ2=1` removes the two multiplies and changes
no load, which is an upper bound on the f16 gain:

| depth | real | DQ2 | |
|---|---:|---:|---:|
| 512 | 46.5 ms | **38.1** | **-18%** |
| 65536 | 58.2 | **49.8** | -14% |

So the dequant arithmetic is a real 18% and **fp16 can take no more than that,
and less** (the f32<->f16 conversions cost instructions of their own).  The other
82% is still unattributed, and it is not bandwidth (the reads are coalesced: a
warp reads 8 contiguous 128-byte runs per superblock) and not occupancy (the fp32
kernel runs 4096 warps = 8 per EU, the measured ceiling).  `PF_MOE_DQ2` is
wrong-answering by design and must never be combined with a correctness cell.

**Where the decode time actually goes: a four-point decomposition.**
`PF_MOE_DQ2` (drop the weight arithmetic, keep every load) and `PF_MOE_NOACT` (drop
the activation load and FMA operand, keep the weight loads and their arithmetic)
are single-variable, so the four combinations account for the kernel's time:

| cell | weight loads | weight ALU | activation loads | FMA | depth 512 | depth 65536 |
|---|:-:|:-:|:-:|:-:|---:|---:|
| real | yes | yes | yes | yes | 46.5 ms | 58.2 ms |
| `PF_MOE_DQ2` | yes | no | yes | yes | 40.8 | 52.4 |
| `PF_MOE_NOACT` | yes | yes | no | yes | 45.8 | 57.5 |
| both | yes | no | no | yes | **38.6** | **50.4** |

So the weight arithmetic is **5.7 ms (12%)**, the activation loads **0.7 ms
(1.5%)**, and **83% of the decode step is reading the 630 MB of expert weights and
nothing else.**  (The activation re-read looked like the suspect - a lane reads 32
bytes of activation against 4 bytes of weight per superblock, and 512 warps share
one 2 KB row - but it is 1.5%, so the L2 absorbs it completely.)

**And the reason is bytes per load instruction, which two independent negatives
both point at.**

*The occupancy curve runs backwards for the experts too.*  Already above: C = 4 /
2 / 1 gives 0.5 / 1 / 2 warps per EU and 44.7 / 56.7 / 81.4 ms.

*The decode attention's split count has an interior optimum for the same reason.*
This model has **8 query heads**, not the 27B's 24, so `n_head * n_splits` is the
whole grid and the profile's `(warps_per_eu/2) * CUs / n_head` resolves to 256
rather than the 160 that AGENTS.md records for the 27B.  Swept properly (the
earlier sweep stopped at 240, which is why it looked monotonic):

| `PF_DEC_SPLIT` | 160 | **240** | 256 | 320 | 384 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| 64k ms/tok | 59.6 | **58.4** | 59.6 | 64.1 | 62.7 | 62.5 |

**More warps is worse**, exactly as for the experts, and flat-to-worse at 16k and
depth 0.  So the 11.7 ms of 64k attention is not an occupancy problem either.

*What the two paths share is the width of one load.*  Measured, per warp per load
instruction:

| path | access | bytes | measured |
|---|---|---:|---:|
| dense int8 GEMV | `w[n*K + k]`, 4 consecutive bytes per lane | **128 B** | **132 GB/s** |
| raw GGUF expert, fp32 decode | `qs[32*(b>>1) + lane]`, **one byte per lane** | **32 B** | **26 GB/s** |

**The GGUF K-quant layout interleaves by value, so a lane's 4 values of a 32-value
group are 4 nibbles inside ONE byte - there is nothing contiguous for it to load,
and the warp gets 32 bytes per instruction where the int8 path gets 128.**  That
is the 4x, and the measured gap is 5x.  It is a property of the on-disk format, not
of the kernel, which is why no tile shape and no split count touches it.

**And the obvious narrowing of that is wrong too, which is worth as much as the
right answer.**  `qs[32 * (b >> 1) + lane]` is the same address for `b = 0` and
`b = 1` - the loop takes the low nibble of a byte and then the high nibble of the
same byte - so the source issues eight load instructions for the four bytes a lane
owns, half of them redundant.  Reading the four bytes into registers first and
taking both nibbles from each is bit-exact (`dev/grp_min` 0/8192 either way, and
pp512/16k/64k argmax 11171/1204/4779 unchanged) and **measures no change at all**:
the load-only floor is 38.8 ms against 38.6.  **The compiler was already CSE-ing
it.**  So "instructions per byte" is not the lever either, and the decomposition
above stands with that removed.

**That hypothesis, tested: partly right, and it is worth 6-7%.**  `PF_MOE_NPB`
makes one warp walk NPB *adjacent* output rows sequentially, reusing a single
accumulator set (holding NPB rows of `ag[]`/`au[]` at once would be NPB*kMoEPB*2
floats and blow the register budget).  Each warp's NPB separate 1152-byte runs
become one sequential run, and the grid drops from `n_act x rows` to
`n_act x ceil(rows/NPB)`, so the stream count falls by NPB and the occupancy falls
with it.  All cells in ONE run, since the box drifts:

| `PF_MOE_NPB` | 1 | 2 | **4** | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| depth 512 | 41.0 | **37.9** | 38.1 | 40.9 | 52.2 |
| depth 16384 | 43.8 | **40.8** | 41.2 | 43.9 | 55.1 |
| depth 65536 | 52.7 | **49.6** | 50.0 | 52.8 | 64.1 |

**Re-swept after the router fix**, all fifteen cells in ONE run because the box
drifts ~13%.  The first sweep was taken with the broken router, so its absolute
values are superseded - the shape was identical, which is all this knob needed.

**Dividing the stream count buys 6-7% and then the occupancy loss takes over,
hard** - 16 is 22-26% worse than 1.  So the stream count is real and it is not
the 5.5x: something else still accounts for most of the gap to the dense path.
`dev/grp_min` is **0/8192 at every NPB**, so the row-boundary handling is right.
**NPB=2 is nominally ahead of the shipped NPB=4 by 0.5-1% at all three depths**,
which is inside this box's run-to-run spread; the default stays at 4 because that
is the value that was swept for exactness and shipped, and a sub-noise difference
is not a reason to move a default.  Anyone re-sweeping should say which they
measured.

**Cumulative plain decode, all three changes together** - the fp32 decode expert
path, NPB=4, and the correct router, which is itself worth something because
every number quoted above it was measured with a router that was selecting a
wrong expert set:

| depth | start of this work | now | |
|---|---:|---:|---:|
| 512 | 56.7 ms (17.6 t/s) | **38.2 ms (26.2 t/s)** | **1.48x** |
| 16384 | 59.6 (16.8) | **41.2 (24.3)** | **1.45x** |
| 65536 | 68.5 (14.6) | **50.0 (20.0)** | **1.37x** |

**What is left, stated as a hypothesis and not as a measurement.**  The fp32 expert
kernel's grid is `n_act x n_ff_exp` = **4096 warps, each reading a DIFFERENT output
row**, i.e. a different superblock `8 * 144 = 1152` bytes further along.  That is
**4096 concurrent read streams on a 1152-byte stride** - the shape that thrashes
DRAM row buffers, and it is exactly what the dense int8 GEMV does not do (each warp
sweeps its own row sequentially).  It is consistent with the 5.5x gap to the dense
path on the same device memory.  **Testing it means giving each warp a block of
consecutive output rows instead of one, which cuts the stream count by the same
factor - a real kernel change, not a knob, and not attempted here.**

**The box drifts.**  The same binary and the same cell measured 41.2 ms and then
46.5 ms for tg128 at depth 512, ~13%, hours apart.  **Only compare cells measured
in the same run**, which is why the DQ2 and occupancy numbers above are each
paired against a baseline from their own run.

### The gather's second stage: one thread per key, 256 scalar bytes at a time

`xmx_gather`'s pass 2 requantizes the paged K/V into the contiguous scratch the
oneDNN matmuls read.  It was `parallel_for(range<1>(blk))` - **one work-item per
key** - and each work-item walked `kD = 256` elements one byte at a time.  Its
sibling in the same function, pass 1 (the block-max reduction), had already been
fixed for the same class of reason (1 work-group -> 64, 18.1 -> 9.1 ms per layer
at 64k); pass 2 had not.

Per key the stage moves 256 (K) + 256 (V) + 32 (two f16 scale planes, 8 groups
each) in and 512 out, so a full 8192-key iteration is ~8.7 MB - **~23 us at the
card's 376 GB/s**.  It measured **~386 us per iteration** (3472 us per call at
64k, 9 iterations), i.e. **17x its own bandwidth floor**, while the softmax
beside it sat at 48% of that same floor and the two matmuls at 8.3 and
14.4 T-MAC/s.  **It was the only stage in the whole prefill not already at a
machine limit.**  The fix is `PF_XMX_GQ` lanes per key row,
each lane owning whole 32-value scale groups so the int8 accesses are 16-byte
vectors:

| `PF_XMX_GQ` | gather, deepest 64k call | vs before |
|---|---:|---:|
| 1 (old shape, **16-byte accesses added**) | **0.902** | **3.85x** |
| 2 | 0.818 | 4.25x |
| 4 | 0.776 | 4.47x |
| **8** | **0.801 -> 0.771** | **4.50x** |

(measured in two builds: the first let the work-group width be `qg` itself,
which reached 1.158 / 0.963 / 0.853 / 0.801, and the shipped one holds it at 64
and reaches 0.902 / 0.818 / 0.776 / **0.771**.  End to end the two are
indistinguishable - 848.6 / 1568.9 / 1437.4 against 848.9 / 1565.0 / 1436.8 - so
the constant width is taken for the structure, not the 4%.)

**Vectorising the accesses was 2/3 of the win and the parallelism was the other
1/3**, which is the opposite of the hypothesis the shape invited.  The reasoning
that pointed at parallelism - "8192 work-items is 256 warps on a 512-EU card, so
half the machine idles" - was arithmetically right and practically beside the
point: what each of those threads did was ~1000 *scalar* memory operations for
one key row (256 elements x 4 loads/stores), and widening them is what the
hardware was actually waiting on.  **The generalisable half: count the memory
*operations* before blaming the memory *system*.**  `probe_coalesce` reached the
same verdict from the other end for the expert GEMM.

`PF_XMX_GRED` (pass 1's slice count) is already at its optimum: 32 is 4% faster
than the default 64 but 128 / 256 / 512 / 1024 all lose monotonically, so 64
stays.

End to end, and the gain is **depth-local as it should be** - the gather runs
once per (kv head, block chunk), so its share grows with the key count:

| cell | before | after |
|---|---:|---:|
| pp512 | 851.8 | 848.6 (flat: mean 834.0 vs 837.2) |
| pp16384 | 1548.8 | **1568.9** |
| pp65536 | 1389.2 | **1437.4** |

**The 64k cell's spread is large and only best-of is meaningful**: one
65536-token prefill is ~47 s, and four reps of the shipped build measured a best
of 1437.4 against a mean of 1156.1.  Quote best-of with that caveat, never the
mean, and never a single rep.

**After this, the attention is at its floor.**  The per-call split at the deepest
64k chunk is now qk 4.346 (37%), softmax 3.873 (33%), pv 2.501 (21%), gather 0.771
(6%) - so **58% of it is two oneDNN int8 matmuls already running at 8-14 T-MAC/s**,
which is faster than dp4a's real kernel and 3.8x faster than DPAS.  There is
nothing left to reclaim there short of changing the two-pass structure, and the
softmax analysis above says that trade is negative.

For scale, `PF_ATTN_XMX=0` (the classic kernel) measures **242.4 t/s at 64k
against the oneDNN path's 1437.4 - 5.9x** - which is also the size of the
`--kv-type i4:i8` cliff below, since any KV type other than i8 falls out of this
path.  (That cell's argmax is 32288, not 4779: a different attention kernel is a
different summation order, so it is a timing cell only and not a correctness
gate.)

**Two bugs this change introduced and fixed, both in the pad-row cold path.**
The zero-fill must stride by the *row-local* lane (`gl`), not by the work-group's
linear id: one work-group spans `QG_WG/qg` rows, so `lr` would give each row only
the bytes its own lane slot landed on - with qg=8 that is 4 of 256, leaving 252
bytes of the pad row holding whatever the previous chunk wrote.  And the scale
must stay inside the multiply: `k[c] * ksc[g] / ksr` is the original fp32
association, and hoisting `ksc[g]/ksr` out is a *different* expression and a
different bit pattern, so the requantized block would stop being bit-identical.

**`PF_XMX_BREAKDOWN`'s argmax is not a usable correctness oracle.**  Across
configurations that differ only in a work-group shape it reported 4779, 279, 3069,
44896, 304 and 32288, while **every uninstrumented run is 4779 at 64k** (and 1204
at 16k, 4821 at pp512), including before and after this change.  So inserting the
instrumentation's `q.wait()` calls changes the answer - most likely a
stream-ordering interaction between the gather/softmax (main stream) and the
matmuls (`st_a`), which this file already flags as delicate.  **The gate for
anything in this area has to be the uninstrumented end-to-end argmax.**  The
breakdown's *timing* is still sound, because the hazard changes which bytes are
read, not how long the stage takes.

**The 64k argmax is not an oracle at any depth, and that is a property of the
model, not of this change.**  Five prefills of one binary in five processes gave
13775 / 13775 / 264 / 44896 / 4779, and the classic kernel (`PF_ATTN_XMX=0`)
produced 264 too.  Use **pp512 (11171; 4821 before the router fix) and 16k (1204)**,
which are stable, as the
correctness gate for anything here.

### The depth cells were run against a 512-block KV pool

**Every depth cell above 16384 was measured against a truncated block table, and
the argmax those runs reported was reading device memory at a negative offset.**
`dev/bench_pf` never set `cfg.kv_cap_mb`, whose default of 0 the engine reads as
"unbounded, fall back to `n_blocks`" - and `n_blocks` is the *initial committed*
count, which that tool sets to 512.  So the pool was **512 blocks = 16384 tokens of
KV no matter what `ntok` was**.  The CLI sets `-1` ("auto: reserve what `max_seq`
needs") when `--kv-cap-mb` is absent; the tool did not, and nobody noticed because
the pool printed itself in every run:

    [kv] exit: pool=512 blocks (160 MB), cap=512 (160 MB), in_use=512, peak=512, grows=32 shrinks=0 (fixed)

Above 16384 `alloc_block()` returns **-1**, the -1 went into the block table
unchecked, and `attn_xmx`'s `kv_row_data(kp_base, kb * n_head_kv + kvh, ...)`
indexed the pool at block -1.  32768 needs 1028 blocks and 65536 needs 2052, so
**more than half the keys read garbage**, and which garbage depends on the
allocator's state - hence the argmax churn.  16384 needs **516** and was short by
**4**, so its apparent stability was luck: the corruption was confined to the last
128 tokens and happened not to move the argmax.  Even 16384 was being measured on
a broken table.

With `cfg.kv_cap_mb = -1` and a loud check on `alloc_block()`'s return, **every
depth is now bit-reproducible** (three separate processes each):

| NT | before (truncated pool) | after |
|---|---|---|
| 16384 | 1204 / 1204 / 1204 (4 blocks short) | **1204 / 1204 / 1204** |
| 32768 | 4821 / 4821 / **13775** | **4821 / 4821 / 4821** |
| 49152 | 8427 / 10993 / 44896 | **44896 / 44896 / 44896** |
| 65536 | 13775 / 264 / 44896 / 4779 / 11 | **4779 / 4779 / 4779** |

44896 and 4779 were already appearing in the churn: they were the *correct* answers,
and everything else was the out-of-bounds read.

**The throughput numbers were never affected, and that is the only reason the
performance work stands.**  The gather and both matmuls process the same number of
keys whether the block index behind a key is valid or not, and the region read was
mapped device memory (an unmapped one would have faulted), so the traffic and the
latency are the same.  Re-measured on the correctly sized pool: pp512 848.3
(was 852.0), pp16384 1566.7 (was 1574.1), pp65536 1437.0 (was 1438.7) - all flat.
**So a correctness gate can be void across a whole measurement campaign while the
performance conclusions are all still correct, and nothing but a check would have
told the two apart.**

Three things generalise, and all three cost more than they should have:

* **Diff the whole output, not the number you are watching.**  The argmax moved and
  `[kv] exit: pool=512` had been printed in every single run.  Four targeted
  ablations (`PF_NOGRAPH=1`, `PF_PF_PIPE=0`, `PF_ABL_NOMOE=1`, `PF_ATTN_XMX=0`) all
  came back "still churns" and none of them was pointed at the pool; the unfiltered
  diff would have.
* **"The harness is correct" is an assumption, not a default.**  `bench_pf`
  duplicated the engine's construction by hand and missed one field that `main.cpp`
  sets from the CLI's own defaults.  **Anything a CLI derives from a user-visible
  default has to be re-derived in every non-CLI entry point**, and the check that
  catches it is the cheapest one available: *does this call return a value I am not
  checking?*  `alloc_block()` returning -1 is an ordinary, documented failure - it
  is the caller's job to handle it, and `bench_pf` did not.
* **A negative index that reaches a kernel is an engine bug too.**  Every current
  caller handles it, but nothing in the attention's pool lookup rejects one, so a
  single missed check anywhere becomes a silent multi-minute corruption instead of
  an error.  Worth a guard at `kv_row_data`; not done here, and recorded here so it
  is not rediscovered as "the 64k argmax is flaky".

**`--kv-type i4:i8` is 5x slower at 64k, and the reason is not that i4 is bad.**
Measured: `i8` 1509.9/1363.1 t/s at 16k/64k, `i4:i8` 744.6/261.9, `i4:i4`
796.8/287.4 - and `PF_ATTN_XMX=0` (the oneDNN attention off) measures 780.8/256.9.
**The i4 numbers *are* the classic-kernel numbers**: `attn_xmx`'s documented
scope is i8 KV only, and any other KV type falls back.  So i4 is not a knob, it
is a port - one that halves the KV bytes the gather and both matmuls stream at
depth, which is the one branch of the three with a plausible payoff.  It is
"write it", not "measure it".

### The row tile in the grid: measured, worth 2% on gate and nothing on down

`moe_grp_impl` loops an expert's `ceil(p/M)` row tiles INSIDE a work-group, so a
work-group's cost is proportional to that expert's pair count - 1..32 tiles on the
real routing.  Lifting the tile into the grid (one work-group = one tile) is the
only thing that moves the critical path, and a host simulation of greedy
list-scheduling over 512 slots (`PF_MOE_HIST`, which now prints it) puts the bound
at **in-order 34 tile-units, LPT 34, perfect 17** - i.e. up to 2x on the expert
GEMMs.  **LPT ordering is worth 0%** (34 -> 34): the single 32-tile work-group *is*
the critical path and it is the same work-group in every order, so "sort the
experts" is a non-starter and only the grid shape can help.  That negative is why
no sort was written.

What the grid shape actually delivers, all four combinations exact at 0/8192:

| `PF_MOE_FLAT` (gate) | `PF_MOE_FLAT_DOWN` | pp512 | pp64 |
|---|---|---:|---:|
| 1 | 1 | 777.3 | 266.4 |
| **1** | **0** (shipped) | **862.4** | **334.4** |
| 0 | 0 | 844.8 | - |

**Gate +2%, down -11%.**  The 2x the simulation promised does not arrive, and
the reason is what the flattening adds: the grid is sized by the upper bound
`sum_e ceil(p_e/M) <= n_pairs/M + n_expert` (the exact count lives on the device
and the grid must be a host constant), so about a third of the work-groups scan
off the end of the CSR and return, every work-group now pays a `n_expert`-entry
scan to find its expert, and `down`'s grid is `N/RB = 64` column blocks against
gate's 16, so its surplus is 4x larger in absolute work-groups against a kernel
that is only half as long.  Net: a small win where the per-tile work dominates
and a loss where the fixed cost per work-group does.

The flat form locates its expert with a **linear scan of `off[]`**, not a
prebuilt tile list: `n_expert` is a few hundred, the array is 1 KB and
L1-resident, and it means no scratch to allocate, no extra kernel to launch, and
no way for `dev/grp_min` to exercise a different path from the engine.

**`dev/bench_grp`'s synthetic histogram was wrong and the whole M sweep was tuned
against it.**  It carried 133 experts with mean 31 and max 124 pairs; the measured
routing is **71 active, mean 54.6, max 256, with four experts at 256**, which
puts the weight re-read factor at **3.90** at M=16 rather than the 2.26 the old
array implied.  That is a much heavier tail, and it is why the shape the
benchmark picked and the shape the engine sees disagreed by 15% for so long.  The
array is now the measured one, which is also why the kernel numbers in the table
above are not comparable with the ones recorded before this change.

### What is left

**First, a correction: the shipped tile was simply the wrong shape, and finding
that out was worth more than any of the four restructurings below.**  An earlier
version of this section argued that the kernel's binding constraint was the
dp4a-per-load ratio `M*FUSE*C/(M + FUSE*C)` and therefore that `M` had to go up.
**The sweep refutes it.**  What binds is the register budget `P = M*FUSE*C`, and
at a given `P` the optimum sits where `M` is *small*, not large:

| cfg | M | C | NSG | P(gate) | gate ms | P(down) | down ms |
|---|---|---|---|---:|---:|---:|---:|
| 0 (shipped before) | 16 | 2 | 16 | 64 | 7.05 | 32 | 4.12 |
| 4 | 16 | 1 | 32 | 32 | 7.29 | 16 | 3.76 |
| **43 (shipped now)** | **8** | **2** | **16** | **32** | **6.01** | **16** | **3.38** |
| 52 | 12 | 2 | 16 | 48 | 6.78 | 24 | 3.61 |
| 55 | 16 | 2 | 8 | 64 | 6.96 | 32 | 4.15 |
| 56 | 24 | 2 | 16 | 96 | 11.43 | 48 | 4.72 |

All exact at 0/8192.  Every `P=32` config lands at 6.0-6.1 ms, `P=64` at
7.0-7.3, `P=128` at 20-51 - **P is monotone and it is the whole story.**  At
equal `P`, `M=8/C=2` (6.01 ms) beats `M=4/C=4` (10.0) because `M=4` doubles the
weight re-read - **and that is affordable**, because the kernel sustains only
50-68 GB/s of unique weight against a 376 GB/s card.  **Bandwidth is not the
scarce resource here; occupancy is.**  `M=8` reads the weight 1.87x more than
`M=16` and is 17% faster, which is the cleanest single demonstration in this
file that the "reduce the weight traffic" framing was backwards here.

The four restructurings all failed, and the sweep says they should have:
a weight transpose (2.6x worse reads, `dev/probe_coalesce`), DPAS (1.08-2.0x
slower, and its 32x denser accumulator does not raise the register ceiling),
SLM activation staging (1.3-1.4x slower, and it forces the M=32 that spills),
and a wider `M` (P grows with it).  What is genuinely left is to raise `FUSE*C`
without raising `P`, which no tile shape can do - it means keeping the
accumulator narrow in *time* rather than in space: a persistent kernel that walks
the column blocks with the activation already in registers.

**And the router WAS a GEMM tiled as a GEMV - which turned out to be 2.1x faster
and wrong, and is now deleted.**  `moe_router_logits_tiled` tiled one 32-lane
sub-group over an `MT tokens x EB experts` block instead of one per (token,
expert), reading the activation `n_expert/EB` times and the router weight
`n_tok/MT` times; measured **45.5 -> 20.5 ms**.  It is **deleted, not left behind
`PF_MOE_ROUTER`**, and the GEMV form is again the only router.  Two independent
defects, and the second is the one that makes this worth writing down:

* **It raced, so the engine's output varied between processes.**  **13 runs of one
  binary produced 13 different prefill argmaxes**, with the top1-top2 margin
  ranging **0.04 to 1.28** - not a tie-break flip, the whole logit vector moving.
  The expert GEMMs were proved innocent first: `dev/grp_min` now relaunches the
  same kernel into the same output six times and reports **0 of 65536 outputs
  differ**, at every tile config and at both decode shapes (`P=1`).
* **The cause is pinned, the mechanism is NOT, and the obvious mechanism is
  wrong.**  Adding `[[sycl::reqd_sub_group_size(32)]]` to that one lambda makes it
  4/4 bit-reproducible.  A three-point isolation in a single tree - HEAD (GEMV)
  3/3 identical, tiled **with** the attribute 4/4 identical, tiled **without** it
  4/4 all different with `l1` spanning 14.38 to 15.51 - attributes the
  nondeterminism to that one line beyond reasonable doubt.

  **But the explanation that first seemed obvious is refuted by direct
  measurement.**  The hypothesis was "a 32-thread work-group is not necessarily
  one 32-lane sub-group, so several lane-0s raced on the same `logits[t][e]`".
  `dev/sgprobe.cpp` measures it, and on **both** the A770 and the Iris Xe:

      sub_group_sizes: 8 16 32
      wg=32  no attribute : sub_group_range=32  n_sub_groups=1
      wg=32  reqd(32)      : sub_group_range=32  n_sub_groups=1

  A 32-thread work-group is one 32-lane sub-group either way, and the attribute
  does not change it (on a 16-thread group `reqd_sub_group_size(32)` is silently
  ignored).  **So the attribute is doing something to the lowering, not to the
  work-group partition, and what exactly is not established.**  The plausible
  remaining story is the one this file already documents elsewhere - `acc[MT][EB]`
  is 64 floats plus 16 temporaries, which is where IGC stops allocating registers,
  and a `sg_sum` over a scratch-resident value may be lowered through a shared
  buffer that the collective does not order.  **That is a guess**; it predicts
  that `MT=EB=4` (16 accumulators, comfortably in registers) should be
  deterministic without the attribute, and nobody has run that.  Do not record
  either the refuted hypothesis or the guess as the mechanism - the attribution is
  solid, the mechanism is open.

* **Pinning the attribute fixed the race and NOT the answer.**  With it the tiled
  form is 4/4 bit-reproducible, and still computes a different logit vector:
  pp512's argmax is **11171** against the established **4821**, and it is
  **slower** - 751.0 vs 772.7 t/s prefill and **49.1 vs 38.1 ms/token** on a
  depth-0 decode (1.29x).  The 2.1x was real; it was buying a wrong answer.  The
  MoE's discrete top-8 selection is what turns one ulp into a different logit
  vector rather than a smaller error, which is also why the top1/top2 margin is
  the useful signal and the argmax alone is not.  **So restoring the attribute
  would not have been a fix** - only deleting the kernel was.

**How it survived a round that was explicitly looking for it.**  `dev/grp_rout`
reported that all five tiled shapes gave "bit-identical logits and 0/4096 ids
different" against the GEMV formulation.  **That check passed while the engine
produced a different answer on every single run**, so it was testing the wrong
thing - the shapes were compared in one process, where a race tends to resolve
the same way, and the GEMV-vs-host disagreement it *did* report was filed as
"probably a pre-existing defect in the reference" and not acted on.  The
generalisation is the one this file already states twice, in a new place: **a
verifier that cannot fail is worse than none**, and the specific shape here is
that **correctness was checked against another device kernel, never against the
one number the product is judged on.**  The check that found it in one run was
`dev/bench_tg` printing the prefill argmax and its top1/top2 margin, once per
step, from three processes - and the check that localized it was the margin
being 25x different between runs, which says "the logits moved" rather than
"an argmax flipped".

**Still open, and not settled by any of this:** `dev/grp_rout` also compares the
GEMV path against a host f32 softmax+top-k on random input, and they disagree
(`max|dp| = 1.0`, expert sets differ for 512 of 512 tokens).  The GEMV form is
the one that reproduces the long-established, bit-reproducible oracles
(4821 / 1204 / 4779) that every prefill measurement in this file rests on, so the
weight of evidence is that it is the trusted path - but that is an inference,
not a proof, and settling it needs a reference the model was actually trained
against (llama.cpp's `ffn_gate_inp` selection on the same activations) rather
than a host reimplementation.  `MT=8 EB=4`'s `wts` differing from the GEMV by
exactly 1.0 with identical ids went with the deleted code and is no longer
reproducible.

**The persistent work-stealing expert GEMM was implemented and is 3% SLOWER, and
the reason is that the imbalance it was built to remove does not exist.**
`PF_MOE_PERSIST=1` (default off, exact at every configuration - `dev/grp_min`
0/8192 up to 0/262144) replaces the grid with `2 * compute_units` work-groups that
pull `(expert, row tile, column block)` triples off an atomic cursor, where the
item total `sum_e ceil(p_e/M) * ncb` is summed **on the device** because the host
cannot know it.  A/B, one run, argmax as the gate:

| | static (`PF_MOE_FLAT=1`) | persistent |
|---|---:|---:|
| pp512 | **778.3** t/s, argmax 4821 | 749.9, argmax 4821 |
| NT=16384 | **1417.6**, argmax 1204 | 1381.1, argmax 1204 |
| NT=65536 | **1315.9**, argmax 4779 | 1288.8, argmax 4779 |
| tg128 512 / 16k / 64k | 38.1 / 41.1 / 50.0 ms | 38.2 / 41.2 / 50.0 ms |

`PF_MOE_PERSIST_WG` = 256 / 512 / 1024 / 2048 / 4096 gives 615.5 / 748.4 / 741.7 /
749.4 / 740.6 t/s - flat above 512 and worse below, so there is no configuration
that wins.  **Do not pursue this further.**  The premise was that 71 active experts
with 1..256 pairs each (mean 54.6) means a 32x tail imbalance, because one
work-group walks 32 row tiles for a 256-pair expert while another walks 1.  That
premise is wrong in the shipped configuration: **`PF_MOE_FLAT=1` already puts one
work-group per (row tile, column block)**, so the imbalance is ACROSS work-groups,
which the hardware scheduler balances by running 12288 of them - there is no
serialisation left to remove.  This file's own host simulation predicted "in-order
34 tile-units, perfect 17, i.e. up to 2x" and that prediction is wrong for the same
reason: it assumed in-order scheduling.  (It is not wrong about `PF_MOE_FLAT`
itself, which measured +2% on gate and -11% on down for a different reason - a
host-sized grid that over-provisions.)

**And the measurement that nearly shipped it was fake, which is the part worth
having.**  The first version drew ONCE per work-group and returned, measured
**2.85x on the prefill, with all three argmax oracles unchanged**, and I did not
believe it.  It should not have: a real prefill has ~9000 work items against a
grid of ~1024, so that version **silently dropped ~90% of the work and still
produced the right argmax**.  The argmax is a weak oracle for precisely this class
of change - this file records the prefill argmax sitting at 4821 while the router
was selecting a completely wrong expert set - and the standing rule is that a stub
which beats the real kernel means the stub is wrong.  What caught it was arithmetic
(item total vs grid size), and what then caught the *real* bug in the same kernel
was that raising `PF_MOE_PERSIST_WG` from 64 to 4096 changed nothing: the cursor
was being drawn **per lane**, so the 128 lanes of a work-group each took a
different item and `sg_sum` summed eight unrelated computations.  A work item must
be drawn by `lid == 0` and broadcast through SLM, with the early-out placed AFTER
the barrier so every lane reaches it - the same shape as this file's recorded
"sg_sum is a sub-group collective, so it has to be called by ALL 32 lanes".

**OPEN, and NOT the block allocator after all: a >16k PREFILL is not
reproducible, low-rate, and the cause is not found.**  Diffing the WHOLE output of
six runs at depth 16384 (not just the hash - that is the method that found the
router bug) put the divergence in the prefill itself: the prefill argmax **agrees**
at 1056 in every run while its **margin does not**, 8.51988 vs 7.68071 and l1
23.2787991 vs 22.1598723 - a ~5% difference in the logits, not a tie-break.  Six
runs give three distinct token-stream hashes; another six give three again.  Depth
512 is 3/3 identical, depth 65536 gave the same hash twice.

Ruled out, each by measurement:

* **The block allocator.**  The `[kv] exit:` line is **identical in all six runs**
  (`pool=528 blocks (165 MB), cap=644, in_use=524, peak=524, grows=33`), so the
  growth itself is deterministic.
* **The 3-phase multi-device pipeline.**  It is the one structural thing that only
  activates with more than one chunk (16k is 32 chunks, 512 is 1, so every earlier
  probe of it was a no-op), and it makes things worse - default 4 runs give **4
  distinct hashes**.  But `PF_PF_PIPE=0` gives 3 of 4 equal and `PF_NOGRAPH=1`
  gives 3 of 4 equal, so it is not the sole cause.
* **Pre-reserving the KV cap.**  `PF_KV_CAP_MB=2048` gave 4/4 identical and
  `=512` gave three distinct, but plain auto also gave 4/4 in that batch while
  being 3-distinct in the previous one.  **Underpowered**: the deviation rate is
  low enough that 4 samples per cell cannot separate the arms.
* **The heap overflow below.**  Fixing it changed nothing, which is why it is
  recorded as a separate bug rather than as this one's cause.

The shape - a dominant mode plus occasional outliers, prefill-local, and needing
more than one chunk - is a race, and it is not any of the above.

**Both per-layer diagnostics were aborting the whole run, and the cause is worth
having: `q.memcpy(...).wait()` on a queue whose next work is a RECORDED COMMAND
GRAPH throws "wait method cannot be used for an event associated with a command
graph".**  The existing `!g_capturing_` guard does not cover it, because the
multi-device decode graph is recorded once at startup and *replayed* afterwards, so
only the recording pass is skipped and every later forward still hits it.  The
result was `terminate called` / core dump - the worst failure mode for a
diagnostic, since it reads as a code fault.  Both now go through `dev_read_back()`,
which reports once and skips, and **both work with `PF_NOGRAPH=1`**, which is the
whole point: the per-layer fingerprints are the sensitive oracle and the token hash
is not (next paragraph).

**With the fingerprints working, the picture changed: the threshold is between
depth 512 and 2048, not at 16k, and the token hash was hiding almost all of it.**
`PF_NOGRAPH=1 PF_DUMP_LAYERS=1`, three runs per depth, counting differing
`[dump]` lines:

| depth | dump lines | differing | distinct token hashes |
|---|---:|---:|---:|
| 128 | 124 | **0** | 1 |
| 512 | 124 | **0** | 1 |
| 2048 | 370 | 64 | 1 |
| 8192 | 1354 | 937 | 1 |
| 16384 | 2666 | 2142 | 1 |

**2666 fingerprint lines with 2142 differing, and one token hash.**  So the engine's
hidden state is wrong in the large majority of layers on most runs while the sampled
argmax usually survives it - which is exactly the "dominant mode plus rare
outliers" shape that made this look like a flaky decode.  **A token-stream hash is
not a correctness oracle for a forward pass; a per-layer fingerprint is.**  Any
conclusion in this file that rests on "the hash was the same" is weaker than it
looks, and the earlier "depth 512 is 3/3 identical" was a statement about the
oracle, not about the engine.

What the fingerprints add, beyond the threshold:

* **It needs more than one chunk.**  512 tokens is ONE `prefill_batch` and is
  bit-identical; 2048 is four and diverges.  So the trigger is "a second chunk",
  not "a big context" - which is why every earlier probe of the multi-chunk
  machinery looked innocent when it was run at depth 512.
* **First divergence is `layerlast il=4 dev=0`** - layer 4, device 0, and only the
  LAST-REAL-TOKEN row, while that same layer's main `layer` row is still identical.
  It then amplifies down the stack (sum 1.3906 -> 2.2100 by layer 11).
* **It survives everything thrown at it**, each by measurement, on the sensitive
  oracle: `PF_PF_PIPE=0` (2136/2132 differing), `PF_NO_PFB_PARTIAL=1` (2140/2124),
  `PF_MOE_DP4A=0` - the fp32 expert path, which does not use the CSR pair order at
  all - (2151/2147), `PF_MOE_NPB=1` (2118/2128), and with the MoE host pass forced
  off (2141/2140).  **Not the pipeline, not the partial-batch path, not the MoE.**

**The threshold is NOT "more than one chunk" - it is `attn_xmx` engaging at all.
The locus is the oneDNN int8 prefill attention, and nothing else.**  The attention is
the one stage the first sweep never touched (it had covered the pipeline, the
partial-batch path and the MoE), and separating it on the fingerprint oracle is
decisive.  Depth 2048, three runs each, differing `[dump]` lines:

| configuration | differing |
|---|---:|
| **`PF_ATTN_XMX=0`** (classic kernel) | **0, 0** |
| baseline (`attn_xmx`) | 72, 72 |
| `PF_ATTN_FLASH=1` | 72, 64 |
| `PF_ATTN_SPLIT=1` | 64, 64 |
| `PF_ATTN_SPLIT_KEYS=1` | 56, 56 |
| `PF_XMX_GQ=1` (gather lane count) | 72, 72 |
| `PF_XMX_GRED=8` (block-max slices) | 93, 85 |
| `PF_ATTN_WAIT=1` (force a blocking wait per matmul) | 72, 97 |
| `PF_ATTN_STREAM=1` (restore the second stream `st_a`) | 64, 64 |
| `PF_GDN_VEC=0` / `PF_GDN_FUSE=1` / `PF_GDN_COLS=1` | 64, 72 / 72, 56 / 72, 72 |

**Only turning `attn_xmx` off makes it bit-reproducible.**  That also corrects the
chunk story: depth 512 has fewer than `PF_ATTN_XMX_MIN = 2048` keys, so `attn_xmx`
never engages there and the cell is deterministic for a reason that has nothing to
do with chunking.  512 being one `prefill_batch` was a coincidence.

**Ruled out inside `attn_xmx`, each by measurement:**

* **Not a stream-ordering race.**  This is the one this file already suspected -
  "`PF_XMX_BREAKDOWN`'s argmax is not a usable correctness oracle ... inserting the
  instrumentation's `q.wait()` calls changes the answer - most likely a
  stream-ordering interaction between the gather/softmax (main stream) and the
  matmuls".  The suspicion was right about the signature and **wrong about the
  cause**: `PF_ATTN_WAIT=1` forces a blocking per-matmul wait and does not fix it,
  and restoring the split stream does not either.
* **Not oneDNN picking a different implementation.**  `ONEDNN_VERBOSE=1` gives
  `jit:gemm:any` with 1203 instances on every run, and the sorted impl set is
  md5-identical across four runs; `ONEDNN_PRIMITIVE_CACHE_CAPACITY=1` changes
  nothing.
* **Not the gather's tiling or the block-max reduction's slice count**
  (`PF_XMX_GQ`, `PF_XMX_GRED`), and not the split heuristics.

So inside `attn_xmx` the implementation, the shapes, the tiling and the ordering are
all deterministic and the result is not.  **Read the file and eliminated the
history-dependent-stread candidates by inspection** - each of these is the obvious
place for a reused `malloc_device` scratch to leak the previous call's bytes, and
each is in fact sound:

* **The pad-row zero-fill.**  `gather` pass 2 does `if (row >= nk) { for (c = gl;
  c < kD; c += qg) gkr[c] = gvr[c] = 0; }` - strided by the ROW-LOCAL lane `gl`
  and covering all of `kD`, which is the fix this file already records ("must stride
  by the row-local lane, not the work-group's linear id ... leaving 252 bytes of
  the pad row holding whatever the previous chunk wrote").  Full coverage, and the
  comment at line 271 states the reasoning: "a zero V row contributes nothing to
  PV".
* **The stale `P` beyond `nk`.**  The softmax only masks `[lim, nk)`
  (`for (int k = lim + tid; k < nk; k += T)`), so `b.p[nk .. blk)` is NOT rewritten
  this call - and `b.p` is `uint8_t`, so `0 * anything = 0` exactly, and the `V`
  rows there ARE zeroed by the gather.  Harmless.
* **`[blk, kBlk)` at all.**  `attn_pv(M, blk, b.p, b.v, b.pv)` uses **`blk`**, not
  `kBlk`, so PV never reads past the gathered width even though QK's shape is the
  fixed one.  The tail-block width (`xmx_tail_blk`) shrinking `blk` therefore cannot
  leak in through PV.
* **The block-max partials.**  Pass 1 writes `part[0 .. nslice)` and
  `part[nslice .. 2*nslice)`; the fold reads exactly those ranges
  (`for (s = 0; s < nslice; s++)`), so a smaller `nslice` never reads a stale slot.
  `PF_XMX_GRED` changing the slice count measures neutral anyway.

**And the ORDERING hypothesis is now disproven, which is the most useful thing in
this entry - it was this file's standing explanation.**  `PF_ATTN_QWAIT=1` (new,
default off) waits on `p->q`, the **SYCL queue every surrounding kernel is
submitted to**, immediately before `prim.execute`; `PF_ATTN_WAIT=1` waits on the
interop stream the primitive goes to.  Depth 2048, three runs each, differing
`[dump]` lines:

| | differing |
|---|---:|
| baseline | 48, 48 |
| `PF_ATTN_WAIT=1` (wait on the interop **stream**) | 72, 97 |
| **`PF_ATTN_QWAIT=1` (wait on the SYCL **queue**)** | **72, 48** |
| `PF_ATTN_XMX=0` | 0, 0 |

**Neither wait helps.**  A wait on the queue is ordered against every SYCL
submission that precedes it, so if the matmul were racing the gather this would
have to fix it.  **So `attn_xmx`'s nondeterminism is not a submission-order race**,
and the long-standing note here - "inserting the instrumentation's `q.wait()` calls
changes the answer - most likely a stream-ordering interaction between the
gather/softmax and the matmuls" - is **wrong about the cause** even though it was
right about the signature.  What made that note look like an ordering problem is
still unexplained, and that is the loose end worth having: something in this path
makes the result depend on whether a wait was inserted, and a wait is not what is
varying between processes.

Incidentally the queue wait is **free** - 772.4 vs 772.5 t/s at pp512, identical
argmax - so `PF_ATTN_QWAIT` is kept as a diagnostic rather than removed, but it is
not a fix and the default stays off.

**Two real defects found in `attn_xmx` by reading it, one of which is fixed.  Neither
is this bug, and that is worth knowing because both were high-confidence guesses.**

* **A negative `lim` in the softmax was an out-of-bounds write, on both sides.**
  `lim = min(nk, olim[i] - k0)` goes **negative** for any query row that sits before
  the current key block's first key - which is every row of every earlier chunk
  once there is more than one block, so from `max_nkv > 8192` up.  It then reaches
  the masked loop `for (k = lim + tid; k < nk; k += T) prow[k] = 0;` as a negative
  start, `k < nk` is true, and the store lands **before the row**: row 0 writes in
  front of the `p8` allocation and row `i` writes into row `i-1`'s live `P` values,
  which the row `i-1` work-group is filling concurrently.  So it is undefined
  behaviour *and* a cross-row race.  **Fixed** by clamping `lim` to `[0, nk]`, which
  is also the semantically correct value: a row that sees no key in this block has
  no visible key in it, so the whole block is masked.  **It is provably a no-op on
  the arithmetic** - pass 1 and pass 2 already skipped a negative `lim` - and the
  argmax oracles are unchanged (4821 / 1204 / 4779), which is what confirms it.
* **The capacity check was never needed, and the arithmetic that suggested it was
  wrong.**  `M = HPG * sum(row_nr)` against `m_cap = HPG * kMaxT * kMaxB` looked like
  a 2x overflow at depth 2048, and it is the only way `M` can overflow.  Measured
  (`PF_XMX_DBG` now prints `HPG`, `n_rows`, `sum_row_nr`, `cap`, `M`):
  **`HPG = 8`, `n_rows = 16 = kMaxB`, `sum_row_nr = 512 = kMaxT*kMaxB`, `cap = 4096`,
  `M = 4096`** - a full batch, filling the scratch exactly, with `n_rows` inside
  `kMaxB` so `step_info`'s own `[kMaxB]` row arrays are safe.  Note this **corrects
  the model note's `8 heads / 2 kv (HPG 4)`**: it is `HPG = 8`, so `n_head_kv` is 1.
* **The remaining three functions in the file are sound.**  `xmx_quant_q` (one
  work-item per row, row-local `amax`, resets `acc`/`m`/`l` in the same launch),
  `xmx_pv_acc` (one work-item per `(row, dim)`, reads `b.pv` and writes `b.acc`) and
  `xmx_output` (`pstride = 2 + head_dim`, split 0 real and every other split
  explicitly zeroed, which is why `n_splits > 1` is redundant rather than wrong).
  At the call site `n_wg` is computed but unused for this path, and `n_splits` only
  reaches `xmx_output`.

**The ordering hypothesis is now disproven in BOTH directions, and padding is
innocent.**  `PF_ATTN_QWAIT=1` was extended to wait on `p->q` after `prim.execute`
as well as before - before, because the gather -> matmul edge was the one the old
`st_a` note blamed, and after, because `xmx_softmax` reads `b.qk` and
`xmx_pv_acc` reads `b.pv` on the queue immediately after the primitives wrote
them on the interop stream, and nothing covers that edge.  Depth 2048 and 16384,
three runs each, differing `[dump]` lines:

| | 2048 | 16384 |
|---|---:|---:|
| baseline | 72, 64 | 2136, 2142 |
| `PF_ATTN_QWAIT=1` (queue wait on **both** edges) | 72, 64 | 2134, 2136 |
| `PF_XMX_TAILBLK=0` (6144 padded columns at 2048) | 64, 40 | - |
| `PF_ATTN_XMX=0` | 0, 0 | 0, 0 |

**No wait on either edge changes anything, and 6144 extra padded columns changes
nothing.**  `TAILBLK` is the arm that isolates the padded region - it holds `nk`
fixed and changes only `blk`, at the one depth where there is no padding to begin
with - and it moves the count within the same run's noise.  So: not a submission
race on either ordering domain, and not the padded columns.

**And the argmax is stable in every single arm** (628 at depth 2048, 1056 at 16384,
including the `PF_ATTN_XMX=0` control at 16384) while the per-layer fingerprints
move.  That asymmetry is the sharpest remaining clue and it cuts against the
byte-corruption reading this whole section assumes: a genuine race usually takes the
argmax with it, as the deleted `moe_router_logits_tiled` did.  Either the
divergence is confined to rows the argmax cannot see (the dumps' first divergence
is `layerlast il=4 dev=0`, the last-real-token row only), or what moves is not the
engine's data but the **fingerprint readback itself**.  The second has not been
tested at all, and it is where the next look should be: `dbg_dump_fp` /
`dev_read_back` issue a host -> device -> host round trip per layer inside a
multi-device forward, and a diagnostic that races the forward it is measuring
produces exactly this signature - a stable argmax with unstable per-layer
fingerprints, sensitive to whether a wait was inserted.

### THE MEASUREMENT WAS WRONG.  `PF_DUMP_LAYERS` is not a valid arbiter, and it invented the whole `attn_xmx` locus

**Everything above this line rests on an oracle that cannot arbitrate, and the
conclusion it produced - "the locus is `attn_xmx`" - is WRONG.**  It is retracted
here rather than left to be re-derived.

`PF_DUMP_LAYERS` and `PF_DUMP_RAW` round-trip host -> device -> host **inside** the
forward, once or eight times per layer.  `dev_read_back` is `q.memcpy(...).wait()`
on the owning device's in-order queue, so it is correctly ordered *as a read* - and
that is exactly the problem: **it inserts an ordering edge between consecutive
layers that the forward itself does not have.**  A diagnostic that supplies the
missing synchronisation will reliably "fix" a missing-synchronisation bug.  That is
the whole explanation for this file's oldest unsolved note, the one that has sat
here for months: *"inserting the instrumentation's `q.wait()` calls changes the
answer - most likely a stream-ordering interaction"*.  It was not a stream
interaction.  **The instrumentation was the ordering edge.**

**A valid oracle, and it is free.**  `dev/bench_tg.cpp` now takes the full post-prefill
logit vector with `PF_DUMP_PF_LOGITS=<prefix>` (and already printed post-prefill `l1`,
which costs one host scan of an already-fetched buffer).  `bench_tg` reads it at
`e.fetch_logits(0, ...)` - **after `prefill_flush()` and `sync_all()`** - so every
device wait is behind the prefill and **nothing inside the forward is perturbed**.
Spread of `l1` over N processes is the nondeterminism measure; full-vector diff is
the detail.

**With that oracle the prefill is genuinely, badly nondeterministic** (depth 16384,
`--layer-map 0-19:gpu.0,20-39:gpu.1`, prompt hash and KV pool state identical across
runs, and the *only* differing lines in the whole output are the two `[trace]` lines):

| depth | post-prefill `l1` over 3 processes | full logit vector |
|---|---|---|
| 2048 | 25.4147797 / 25.4286461 / 25.5590782 | - |
| 16384 | 22.3595371 / 22.9385338 / 23.4675236 | **all 248320 elements differ**, max abs 1.86 / 2.08, rel 8.3e-2 / 9.3e-2 |

and the **argmax is stable throughout** (1056 prefill, 440 first decode, in every
run).  A whole logit vector shifting a few percent while its argmax survives is not
what a corrupted read looks like - it is a systematic numerical difference, and it is
**amplified by the MoE** (whose discrete top-8 turns a small difference into a
different expert set) rather than caused by it.

**And `PF_ATTN_XMX=0` is NOT reproducible.**  Under the valid oracle it still spreads
**0.37%**.  The earlier "depth 16384 `PF_ATTN_XMX=0` -> 0 differing" was the
perturbing oracle lying.  Measured spread of post-prefill `l1` over 3 processes,
depth 16384:

| arm | spread | |
|---|---:|---|
| **`PF_GEMM_DNNL=0` (no oneDNN at all)** | **0** | **REPRODUCIBLE, twice** |
| `PF_ABL_NOMOE=1` | 0.0655937 | 0.24% - the MoE amplifies ~12x |
| `PF_ATTN_XMX=0` | 0.0832653 | 0.37% - attn_xmx amplifies ~8x |
| default | 0.668642 | 2.91% |
| `PF_XMX_POISON=1` | 0.563503 | 2.48% |
| `PF_ABL_NOGDN=1` | 0.36235 | 4.24% |
| `PF_MOE_DP4A=0` | 0.486687 | 1.76% |
| `PF_W4=0` | 0.645958 | 2.82% |
| `ONEDNN_MAX_CPU_ISA=1` | - | 2.64% |
| `ONEDNN_MAX_CPU_ISA=0` (**no threading at all**) | - | **5.38%, worse** |
| `PF_DNNL_QWAIT=1` (queue **and** stream wait after every oneDNN execute) | - | 1.46% vs 1.38% default |

**So: oneDNN is NECESSARY and the mechanism is still not found.**  Not its host
threading (disabling it is worse), not submission ordering (every edge, both
domains), not uninitialised scratch (poison), not padding, not the negative `lim`
(found and fixed above, demonstrably not it).  `PF_DNNL_QWAIT` (renamed from
`PF_ATTN_QWAIT`, and now covering all seven executes) is the general form of the
ordering test and it is the one that finally covered the dense GEMM -> SYCL kernel
edge; it does not help.

**What the element-wise dumps add, now that they can be trusted at all.**  Holding
the diagnostic configuration FIXED (the divergence point moves when the dump arity
changes - 2 waits/layer gave `il04`, 8 waits/layer gave `il12`, same binary
otherwise), the first differing layer is born from **bit-identical** inputs: every
one of the 2048 elements of the layer's output differs, at a relative 2.5e-4,
growing to ~1e-1 by the end.  Both the first token row (`w0`, position 0, which sees
exactly one key) and the last real token (`w1`) start diverging at the *same* layer,
so it is not row-local, and it is born inside a layer rather than accumulated into
one.  Only chunks with `max_nkv >= 2048` ever differ - which is the `attn_xmx`
threshold, but that is where the *amplification* is, not where the difference is
born.

**The methodological rule this earns, and it is the generalisation of three separate
mistakes in this file:** *an oracle that synchronises is not an oracle for a
missing-synchronisation bug.*  The token hash was rejected earlier as too weak; the
per-layer fingerprint is rejected now as actively misleading.  What is left is an
oracle read strictly after the region under test has completed and been waited on.

### The oneDNN axis is now an exhaustive negative result, and that is the finding

`PF_GEMM_DNNL=0` is still the ONLY arm that reproduces (spread 0, three separate
times), so **oneDNN's presence is necessary**.  It does not follow that oneDNN's
GEMM is at fault, and by now everything reachable from the descriptor has been
measured.  Spread of the post-prefill `l1`, depth 16384, non-perturbing oracle:

| hypothesis | arm | result |
|---|---|---|
| oneDNN host threading | `ONEDNN_MAX_CPU_ISA=1` / `=0` | 2.64% / **5.38%, worse** |
| uninitialised memory, `attn_xmx` | `PF_XMX_POISON=1` (all 15 buffers, 0x7F) | 2.48% |
| uninitialised memory, the whole oneDNN path | `PF_DNNL_POISON=1` (**every** `dev_alloc` buffer) | 2.22% |
| ordering, attention edges | `PF_ATTN_QWAIT` before and after both primitives | no effect |
| ordering, dense GEMM -> next SYCL kernel | `p->q.wait()` + `p->st.wait()` after all 7 executes | 1.46% vs 1.38% |
| ordering, quantizer -> dense GEMM | `p->q.wait()` **before** all 7 executes | 5.26% vs 2.49% |
| accumulation precision | `PF_DNNL_FPMATH=2` = `fpmath_mode::strict`, exact s32 | 1.44% |
| accumulation precision, the other way | `PF_DNNL_FPMATH=1` = `any,true` (fp32) | 3.75% |
| warmup's leftover work | `PF_DNNL_NOWARM=1` | 6.75%, worse |
| stale descriptor pointers | read the code: every long-lived buffer is allocated in `dnnl_gemm`'s ctor and **never reallocated** (`dev_erase` has no caller) | dead |
| the MoE's GEMM | `PF_MOE_DP4A=0` (fp32 dequant, no oneDNN) | 1.76% |
| the native weight stores | `PF_W4=0` | 2.82% |

**So the difference is inside oneDNN's own GPU kernel at run time, past everything
the descriptor exposes.**  Three specific things are worth writing down because they
are wrong in this file now:

* **`ONEDNN_MAX_CPU_ISA` is an ISA knob, not a threading knob.**  The rows above
  therefore do **not** refute host threading - nothing here ever disabled it.  For a
  SYCL-interop stream the thread count comes from the queue, and this engine never
  sets it.  **The one mechanism in the whole table that has not actually been tested
  is multi-threaded oneDNN**, and it remains the single most plausible explanation
  left: an int8 GEMM split across threads whose partial sums are combined in a
  runtime-dependent order is exactly a "consistent 1-5% spread with a stable argmax".
  Settling it needs the queue's oneDNN stream configured for one thread, not an env
  var on the library.
* **`fpmath_mode::strict` not reproducing it is informative.**  AGENTS.md's note that
  an A770 u4 matmul "overflows to NaN at K >= 5376 under narrow fp16-class
  accumulation" was being read as evidence the accumulation was inexact, and it was
  used to motivate this test.  Forcing strict changes the value (l1 mean 23.20 vs
  22.97) and does not remove the spread, so that note is about the *library's*
  default, not about where the nondeterminism comes from.
* **`PF_GEMM_DNNL=0` does not isolate the matmul.**  It also swaps the activation
  quantizer, the grouped-scale layout and the epilogue for the dp4a branch, and it
  makes `attn_xmx` decline (`dnnl_for_queue` returns null).  So "oneDNN is
  necessary" is a statement about a whole branch, not about `prim.execute`, and the
  four env knobs that exist to separate those pieces do not cover the dense GEMM's
  *input* side.  **The untested seam is the activation quantizer
  (`act_quant_grp_launch`), which is pure SYCL and is the only computation in that
  branch that is not oneDNN's** - it is a per-32-group amax reduction feeding the
  GEMM's source operand, and the engine's `act_view` staleness check (`gen`, `M`,
  `K`) is host-side, so it cannot see the device scratch being overwritten by a
  different call with the same shape.

### Host threading is closed too, and for a better reason than a knob

The table above needed a correction and then a real answer.  `ONEDNN_MAX_CPU_ISA`
reports `onednn_verbose,v1,info,cpu,runtime:DPC++,nthr:28`, i.e. oneDNN *is* using
28 host threads - so the ISA rows never disabled threading and did not refute it.
But threading cannot be the mechanism here, for a reason that needs no knob at all:

**every one of these is an int8 x int8 -> s32 matmul, so the accumulation is exact
integer addition.**  `fpmath_mode` only applies to bf16/fp16 *input* data types and
does not apply to an int8 source with an s32 destination at all - which is also why
`PF_DNNL_FPMATH=2` (`strict`) changing the value but not the spread is uninformative
rather than evidence.  Splitting K across threads and combining **exact integer**
partials in a runtime-dependent order is bit-for-bit identical.  No thread count can
produce a 2% difference in a sum that cannot round.

**And oneDNN's own decisions are identical run to run.**  Two runs at depth 2048 with
`ONEDNN_VERBOSE=2`, timings stripped, diffed line by line: **1572 verbose lines, 0
differing** - the same implementation selection, the same descriptors, the same
kernel and nested-primitive cache hits - while the engine's own post-prefill `l1` in
those same two runs was **25.2284012 vs 25.6039982**.  So the nondeterminism is not
inside oneDNN's execution.

What that does *not* exclude is that oneDNN is **handed different data**: verbose logs
descriptors, not contents.  That leaves the activation quantizer
(`act_quant_grp_launch`) as the untested seam - it is pure SYCL, the only computation
in the oneDNN branch that is not oneDNN's, and it feeds the GEMM's source operand.
Read, it is deterministic in itself: one 32-lane work-group per (row, group), the
value never leaves its register between the max pass and the store, the scale is
`fmax/127` (exact and associative) and the group sum is `plus<int>` (exact).  Its
*inputs* are the activation and `up`, the gate projection, and `up` is written by a
oneDNN GEMM on `p->st` - which makes the GEMM -> quantizer edge the one to look at,
and `PF_DNNL_QWAIT` already waits on it without effect.  **The engine's
`act_view` staleness check is host-side (`gen`, `M`, `K`) and cannot see the device
scratch being overwritten by a different call with the same shape** - that remains
the structural weakness on this seam.

**One latent defect found while reading it, not the cause.**  The kernel writes the
group scale from `lane == 0` only:
`if (lane == 0) sasc[...] = bits;` - while **every** lane quantized its own value
with `mx` from `reduce_over_group(it.get_sub_group(), ...)`.  Those agree only if the
sub-group size equals `kW4Group`.  On this part a 32-thread work-group is one
32-lane sub-group (`dev/sgprobe.cpp` measures `n_sub_groups=1` at `wg=32`), so it is
correct today - but it is an unstated precondition, and the same shape silently
produces a wrong-but-plausible scale on any part whose sub-group is 16.  AGENTS.md's
own rule applies: an `sg_sum`/collective's width is a hardware fact, so the store
should be from the lane whose reduction it recorded, or the two should be tied
together explicitly.

### A non-perturbing per-call checksum retires oneDNN as the locus

Every attempt above measured the nondeterminism from outside the branch, and every
one of them said "somewhere in oneDNN".  The instrument that settles it is
`PF_ACTQ_SUM`, and the design constraint is the whole point: **the checksum is
accumulated on the device and read exactly once, in `~dnnl_gemm`.**  A host read-back
inside the forward is precisely what made the per-layer fingerprint oracle useless
(this entry's own first half), so `act_quant_grp_launch` takes `int64_t * sumq,
sums, int slot` and, inside its existing `if (lane == 0)` block, adds
`reduce_over_group(q8, plus<int>)` and the scale bits through 64-bit relaxed atomics.
Both are exact and order-independent, so **the checksum is itself deterministic given
the data** - it can only move if the data moves.

It moves.  Three processes, depth 16384, 2643 quantizer calls each (identical count,
so the schedule is the same):

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| post-prefill `l1` | 22.5222721 | 23.1787815 | 22.6040802 |
| activation checksum | `37b52703ed68e724` | `0621ba1658433ad4` | `d14e6d02d94590e3` |

**So oneDNN is being handed different bytes.**  The quantizer is deterministic in
itself, so its *input* - the fp32 activation, and `up`, the gate projection - already
differs.  **oneDNN computes faithfully from whatever it is given, and is not the
locus.**  `ONEDNN_VERBOSE` could never have shown this: it logs descriptors, not
contents.

**Per-slot diffing then localises it precisely.**  `PF_ACTQ_SHAPE` adds each call's
`M`/`K`, and with two `dnnl_gemm` instances (one per `--layer-map` device) each
printing its own slot block, depth 2048 gives:

| device block | calls | `M==1` | `M>1` | differing | first differing `M>1` call |
|---|---:|---:|---:|---:|---|
| 0 (layers 0-19) | 402 | 161 | 241 | 100 (48 / 52) | **slot 350**, `qsum` -422904 -> -422947 |
| 1 (layers 20-39) | 400 | 161 | 239 | 160 (80 / 80) | **slot 320**, `qsum` -117931 -> -117492 |

`M == 1` is the **multi-device decode command graph being recorded at startup** (40
layers x 2 calls) plus the decode step at the end; `M > 1` is the prefill.  And
**48 of the 161 startup `M == 1` calls already differ**, so the divergence exists
*before the prefill runs* - which is consistent with the earlier element-wise finding
that chunks with `max_nkv < 2048` are bit-identical while later ones are not, and
inconsistent with a story where the prefill's own arithmetic invents it.

The size is what makes this tractable.  Slot 350 is `M=512, K=2048`: **1,048,576
values, of which 43 differ, each by exactly one int8 step** (`dqsum = -43` against a
population of 1,048,576; the scale plane's `dssum` moves by 27,462 out of 225
million).  **That is a handful of activation values sitting within an ulp of a
quantisation boundary** - the smallest difference this whole investigation has ever
produced, and the first one that is localisable to a call index.

**What this is not.**  It is not a near-tie in the MoE router: flipping one of the
top-8 experts changes the layer's output by far more than 43 values in a million.
It is not `attn_xmx` either - that only amplifies, and `PF_ATTN_XMX=0` still spreads
0.37%.  The instrument is what matters for the next step: **a per-call,
non-perturbing, order-independent checksum of the quantized activation is a
bisection instrument that needs no further instrumentation** - diff the per-slot
lines between two processes and the first differing slot is the call whose *input*
already differed.  Mapping that slot index to (chunk, layer, call) is the one cheap
step left, and it is the step that turns "189 calls into the prefill" into a layer
number.

#### Where the first differing call actually is

`PF_ACTQ_SHAPE` prints each call's `M`/`K`, and under `0-19:gpu.0,20-39:gpu.1` there
are **20 layers per device and exactly 3 quantized calls per layer**, so the `M > 1`
(prefill) calls are `4 chunks x 60 per chunk` and a slot index divides into
`(chunk, layer, position)`.  Depth 2048, three processes, `PF_DNNL_NOWARM=1` so the
startup decode-graph recording is out of the slot range:

| device block | first differing `M > 1` call | chunk | global layer | that call's `(M,K)` |
|---|---|---:|---:|---|
| 0 (layers 0-19) | `#173` (slot 334) | **3** | **17** | `(512, 4096)` |
| 1 (layers 20-39) | `#159` (slot 320) | **3** | **33** | `(512, 512)` |

**Both devices first differ in chunk 3**, and chunk 3 is exactly the chunk where
`max_nkv` reaches 2048 and `attn_xmx` engages (chunk 0/1/2 have `max_nkv` 512/1024/
1536, below `PF_ATTN_XMX_MIN`).  That is independent confirmation, from a
non-perturbing instrument, of what the element-wise dumps showed - the first three
chunks are bit-identical and the fourth is not.  It is *not* confirmation that
`attn_xmx` causes it: `PF_ATTN_XMX=0` still spreads 0.37%, so this is where the
difference is **visible**, and the honest reading is that whatever amplifies there
is downstream of a difference that is already present upstream.

**How far this can be trusted.**  The layer index rests on all 20 layers making
exactly 3 quantized calls, which the 100%-matching `(M,K)` periodicity supports.  The
position *within* a layer does not: the best `(M,K)` period found is 4 against 3 calls
per layer, i.e. the `K` sequence is degenerate and two of a layer's three calls share a
`K`.  **So read "chunk 3, layer ~17 / ~33" and not "position 2 of 3".**  Disambiguating
needs a per-call identifier the engine supplies, and `dnnl_gemm::quantize()` has no
weight `key` to carry one - threading it would touch every caller in the engine,
`engine_mtp.cpp` and `engine_dflash.cpp`.

#### The arm bisection, and it reverses the "amplifier only" reading

With the per-slot checksum the arms can finally be compared on *where* they first
differ rather than on a downstream logit norm, and the result reverses the reading
above.  Depth 2048, three processes each, `PF_DNNL_NOWARM=1 PF_ACTQ_SUM PF_ACTQ_SHAPE`:

| arm | first differing `M > 1` quantized activation |
|---|---|
| default | chunk 3, layer 17 (dev 0) / layer 13 (dev 1) |
| **`PF_ATTN_XMX=0`** | **REPRODUCIBLE - 0 of 241 and 0 of 239 calls differ** |
| `PF_ABL_NOGDN=1` | chunk 3, layer 17 / 13 |
| `PF_ABL_NOMOE=1` | chunk 3, layer 17 / 13 |
| `PF_MOE_DP4A=0` | chunk 3, layer 17 / 13 |
| `PF_W4=0` | chunk 4, layer 2 / chunk 3, layer 13 |

**`PF_ATTN_XMX=0` makes the quantized activation bit-reproducible.**  The earlier
"it still spreads 0.37% of `l1`" was measured on the logit norm, which sums every
chunk and every downstream amplification; at the quantized-activation level the
classic attention is exactly reproducible.  **`attn_xmx` creates the difference, it
is not only an amplifier** - and every other arm lands on the *same* (chunk, layer),
which says the MoE, the GDN, the expert GEMM and the weight stores are all innocent of
creating it.

The position is the mechanism.  Chunk 3 layers 0-16 are clean and layer 17 is not, so
this is not "one layer goes wrong" but **a per-layer difference small enough to stay
invisible for sixteen layers and then cross a rounding boundary**.  That fits the
measured size exactly: **43 of 1,048,576 int8 values, each off by exactly one step.**
One step is `round()` in `xmx_softmax`'s requantization -
`prow[k] = round(fmin(255, e * 255))` where `e = exp(qk*sc - blo)` and
`sc = qsc[i] * bs[0] * attn_scale`.  `qk` is an exact int32 matmul result, so the only
inputs to that `exp` that can move are `qsc[i]` (from `xmx_quant_q`'s row amax),
`bs[0]`/`bs[1]` (one scale per 8192-key block, from the gather's two-stage `fmax`),
and `attn_scale` - and a one-ulp move in any of them flips a `round()` for only the
handful of keys whose `e * 255` sits near an integer.  **A uniform scale difference
would move every element; a ulp difference in the exponent moves a few `round()`
results.  The measurement says the latter.**

**So the next single measurement is small and it splits the remaining space in two:**
device-side checksums, per gather call, of the **gathered K/V bytes** and of
**`bs[0]`/`bs[1]` separately**.  If the K/V bytes move, the paged pool's contents are
changing; if only the scales move, the gather read identical bytes and the block-max
reduction is what differs.  `PF_ACTQ_SUM` is the template - same rule (accumulate on
the device, read once), same reason (a host read-back inside the forward is what broke
the fingerprint oracle).  It is *not* implemented: threading a per-call slot through
`xmx_gather` into the `fold` and `pass 2` is a mechanical change that was started and
then reverted rather than left half-applied.

#### The K/V bytes change and the block-max reduction does not

`PF_XMX_SUM` is the same device-accumulate / read-once-at-exit instrument applied to
the gather, and it splits the two remaining candidates by construction: `kv` hashes
the gathered K/V **bytes**, `bs` hashes `bs[0]`/`bs[1]` **separately**.  Depth 2048,
three processes:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| gathered K/V bytes | `645ac384178c6ed6` | `7ef16a9ea40406a4` | `f8f29d6a0e7fddd4` |
| block scales `bs[0]`/`bs[1]` | `3f21ef1ab7506363` | `3f21ef1ab7506363` | `3f21ef1ab7506363` |

**18 of the 20 gather calls have different K/V bytes, and ZERO have a different block
scale** - `bs0`/`bs1` are bit-identical at every call in every run (slot 2's values are
literally `1033011200` / `1015226368` in all three).

**So the paged pool's contents change between processes, and the two-stage `fmax`
reduction over those contents is exactly deterministic.**  The reduction is exonerated
along with oneDNN, and the difference is on the **write** side of the KV pool, not the
read side and not the scale derivation.  That is also consistent with everything above:
a K/V byte differing by one quantization step is exactly a "few values off by one
step" signature, and it explains the ulp-in-the-exponent mechanism without needing the
scale to move at all - `qk` is an int32 dot of two int8 blocks, so one changed byte
moves one score by a whole step.

`calls=20` is chunk 3's ten full-attention layers (`full_attn_interval=4`, so il03,
07, ... 39 - and chunk 3 is the only prefill chunk where `attn_xmx` engages) followed
by the decode step's ten.  **The first differing gather is slot 2, i.e. il11.**

**And the writer is now identified.**  `attn_xmx` only ever *reads* the pool - all ten
of its `kv_row_data` uses are `const int8_t *`.  The writer is
**`qk_norm_rope.cpp`**: it takes `kpool`/`vpool`/`kscales`/`vscales` and dispatches on
`kv_dtype_t`, and for `i8` it launches with both scale planes, i.e. it does the q/k
norm, the RoPE, **and** the per-32-group i8 quantization of the KV it stores.

**The next single measurement is therefore the same instrument one level down**:
per-call device-side checksums of the K/V bytes and the two scale planes *as
`qk_norm_rope` stores them*.  If they already differ on the chunk-0 store then the
divergence is born in the store and the read path was never involved; if they match
until chunk 3, then something between the chunks changes them.  Note the tension to
resolve: the store runs on **both** the xmx and the classic path, and
`PF_ATTN_XMX=0` is bit-reproducible, so the store cannot be unconditionally the
source - which means either its input (the activation) already differs in the default
arm, or the xmx path perturbs something the later store reads.

#### The store's input already differs in the FIRST prefill chunk - and the tension resolves

`PF_KVSTORE_SUM` folds the post-norm, post-RoPE K and V into a per-call device checksum
**inside the existing kernel** (no extra launch, so no new ordering edge - both of
which this investigation has shown can move the answer).  It prints each call's extents
so the slot index is decodable, which immediately paid for itself: the earlier reading
of the `PF_XMX_SUM` result assumed slots 0-19 were the startup decode-graph recording,
and that had never been checked.

Depth 2048, three processes, `PF_KVSTORE_SUM=1`:

| arm | calls | differing | first differing |
|---|---:|---:|---|
| default | 100 | 9 | slots 2,4,6,8 (`nrows=1 nreal=1`) then **10,12,14,16,18** (`nrows=16 nreal=32`) |
| `PF_ATTN_XMX=0` | 100 | **0** | REPRODUCIBLE |

`100 = 20 + 4 chunks x 20 layers`: twenty `M == 1` calls are the startup decode-graph
recording plus its replays, and the rest are the prefill.  **The `M == 1` slots are
downstream and prove nothing** - a replayed graph re-reads the buffers at replay time,
so those slots inherit the prefill's difference.  **The signal is slots 10+.**  Those
are the `M == 512` prefill calls, and **the very first one already differs.**

**And that is the narrowest this has been.**  `PF_ACTQ_SUM` established that the
prefill's *quantized GEMM inputs* are bit-identical in chunks 0, 1 and 2.  This
establishes that the *K/V actually stored* are not.  So the difference is created
**between the quantized GEMM's input and the stored K/V** - i.e. in the qkv GEMM's
output path or in the norm + RoPE that `qk_norm_rope` applies in place before
quantizing.  An int32 accumulator converted to fp32 by a per-group-scaled epilogue is
the obvious place for an order-dependent fp32 accumulation, and it is downstream of
every GEMM *input* this investigation has been able to see.

**This also resolves the tension recorded above.**  The store runs on both the xmx and
the classic path, and `PF_ATTN_XMX=0` is bit-reproducible, so the store could not be
unconditionally the source.  It is not: the store's *input* is what differs, and it
differs only when `attn_xmx` is enabled - so `attn_xmx` changes what the qkv path is
fed, not what the store does.

#### The norm + RoPE is exonerated, the refutations survive a sharper oracle, and the
#### prefill's dense GEMM does not go through oneDNN at all

`PF_KVSTORE_SUM` was split into a **pre** hash (the qkv GEMM's fp32 output, taken at
the top of the kernel before its in-place norm and RoPE touch anything) and a **post**
hash, in the same slot pair, so one measurement separates the GEMM's epilogue from the
norm + RoPE.  Depth 2048, three processes:

| arm | pre differs | post differs |
|---|---:|---:|
| default | 9 | 9 |
| `PF_ATTN_XMX=0` | **0** | **0** |

**Pre and post differ at exactly the same slots.**  The norm + RoPE are exonerated: the
K/V already differ when `qk_norm_rope` receives them, i.e. in the qkv GEMM's fp32
output.

**The refutations were re-run with this oracle rather than `l1`**, because every one of
them was originally measured with an oracle this entry has since shown cannot arbitrate.
They all hold: `PF_DNNL_QWAIT=1`, `PF_DNNL_FPMATH=2`, `PF_XMX_POISON=1`,
`PF_DNNL_NOWARM=1` and `PF_W4=0` all leave pre/post differing at 9 slots.  Only
`PF_ATTN_XMX=0` is reproducible.

**And then the accumulator probe answered a question this entry has had open for
several rounds, by coming back empty.**  `PF_DNNL_ACC_SUM` hashes `p->acc` right after
oneDNN's `prim.execute` at all three dense sites (the k5 prefill site and both
`w.wmem` sites).  It reports **`calls=0`** - at depth 2048, **not one of them fires.**

**So the prefill's dense GEMM does not use oneDNN's matmul at all.**  It is a SYCL
kernel (`w4_gemm_launch` or `nat_gemm_launch`, both pure SYCL, both reached before the
oneDNN branch).  Two consequences:

* **`PF_GEMM_DNNL=0` is fully explained by `attn_xmx` declining.**  It returns false from
  `dnnl_for_queue(q) == nullptr`, so the classic attention is used - and that is the
  whole of its reproducibility.  Every "oneDNN is necessary" reading built on that arm
  was wrong in the same way as the earlier `PF_ATTN_XMX=0` reading: the arm changes more
  than the thing it names.
* **The remaining site is the prefill's SYCL GEMM or its epilogue**, writing `kbuf`.
  Given the prefill's chunk-0 qkv already differs while `attn_xmx` has not yet run
  even once (it first engages in chunk 3), the coupling is not a side effect of
  `attn_xmx`'s execution - it is that `attn_xmx_enabled` changes which branch the plan
  takes.

#### The wall: a schedule-sensitive bug cannot be attributed with an instrument that
#### perturbs the schedule

The previous commit's attribution - "the prefill's first qkv GEMM" - is **not stable**,
and the way it failed is the most useful thing in this whole entry.

Adding a `pos` field to `PF_KVSTORE_SUM` (a third int64 per slot, written by the
device, purely so a slot index could be decoded into a chunk) changed nothing about what
the probe does - two sub-group reductions and two atomics inside `qk_norm_rope` - and
the answer moved completely:

| build | the calls that differ |
|---|---|
| previous | the prefill's first M == 512 calls, `nrows=16 nreal=32` |
| with the `pos` field | **only `nrows=1 nreal=1` at `pos=2048`** - the decode step |

Both are true.  `l1` (the non-perturbing oracle) still reports the prefill's output
differing, and the prefill's stored K/V are now bit-identical, so the two statements
are only consistent if **the instrument moved the divergence**.

**So: an instrument that perturbs the schedule cannot attribute a schedule-sensitive
bug to a call site.**  Every per-call reading in this entry is a statement about the
*instrumented* build, not the shipped one - including, in principle, every refutation,
because an arm that reduces the perturbation makes the symptom disappear, which is
observationally identical to fixing it.  This is the same lesson as the fingerprint
oracle, one level deeper: that oracle *supplied a missing synchronisation*, this class
*changes a schedule*, and both produce confident, plausible, wrong answers.

What survives, because it holds in every build and under both oracles:

* **`PF_ATTN_XMX=0` is bit-reproducible; the default is not.**  Every arm, every
  oracle, every build.
* **`pre` and `post` agree** on which calls differ, in every build: the norm + RoPE
  add nothing.
* **The prefill's dense GEMM does not use oneDNN's `prim.execute`** - `calls=0` at all
  three dense sites.  That is a structural fact, not a timing-sensitive one, and it is
  what makes `PF_GEMM_DNNL=0`'s reproducibility entirely a story about
  `attn_xmx` declining.
* **oneDNN's own decisions are identical run to run** (1572 verbose lines, 0 differing),
  and its arithmetic is exact integer, so it cannot be the source on its own.

**What it would take to go further.**  Not another probe inside the forward - every one
of those is now known to move the thing it measures.  Either (a) an instrument that
provably cannot change the schedule, which for a kernel means *not touching it*, and
the only such thing left is a checksum computed by an **already-existing** kernel on
data it already touches, or (b) accept the symptom and characterise it externally: it
is stable in argmax, it is 1-5% on the logit vector, and it only appears once
`attn_xmx` engages at 2048 keys.

#### The profile on this build, and the router is the tractable target

Re-profiled after every change this entry describes (NT=16384, 35B A3B / 2x A770):

| | t/s | wall | share of the prefill |
|---|---:|---:|---:|
| baseline | 1420.1 | 11.537 s | - |
| `PF_ABL_NOMOE=1` | 5056.6 | 3.240 s | **routed experts = 71.9%** |
| `PF_ABL_NOGDN=1` | 1548.0 | 10.584 s | GDN = 8.3% |
| `PF_ATTN_XMX=0` | 750.7 | 21.825 s | attention ~15%; `attn_xmx` is worth 1.89x |

`PF_MOE_TIME`'s phase split (absolute values inflated ~1.4x by its own serialisation,
shares usable): `router=1358.7 csr=119.2 gate_up=9223.4 down=5314.6 reduce=140.2`
over `calls=1280` - so `gate_up` is 57% of the MoE and `down` 33%, and **the router is
1.06 ms per call**.

**The router is ~6% of the prefill and runs ~30x off its own roofline**: 512 tokens x
256 experts x 2048 = 268 M-MAC, and the engine's measured dp4a rate is 7.6 T-MAC/s, so
it should be ~35 us against a measured 1.06 ms.  It is also the only sizeable item in
the MoE that has **not** already resisted four restructurings.  So it is the tractable
target, and this entry's next line of work.

**A working version of the tiling was built, measured, and then deleted** - and the
deletion is the finding.  One expert per work-group with `MT` tokens, lanes keeping the
same `k = 32*b + lane` mapping and the same `sg_sum`, so it is arithmetically identical
to the GEMV form by construction:

| MT | router phase | pp16384 |
|---|---:|---:|
| 1 (shipped) | 1711.2 ms | 1391.8 t/s |
| **8** | **1159.1 ms** | **1422.7 t/s (+2.2%)** |
| 16 | 1331.3 ms | 1413.3 t/s |

**and it was not exact.**  `PF_MOE_ROUTER_AB` - a device-side A/B against the GEMV form,
because "bit-identical" is a claim a benchmark cannot check - reported **131072 of
131072 logits differing, max abs 9.77, and `max|tiled| = 11` where the GEMV form's max is
6.204**.  The cause is the shape this file warns about twice: the token loop was bounded
by a **runtime** `nm` (the tail-tile count), so `acc[m]` was indexed by a
non-constant, which on this backend places `acc` in **private/scratch memory** instead of
registers.  The fix is the prescribed one - **predicate** the store rather than bound the
loop with a runtime value, keeping `m` a compile-time constant - and that was not done
here.

**Two of this file's own lessons fired in the making of that probe, both worth
recording:**

* **The A/B's control was not the same source.**  It reported 131072-of-131072 differing
  *for every MT including MT=1*, with identical magnitudes - which is impossible for four
  different computations and was the tell.  MT=1 had been routed to `moe_router_logits_tiled<GP,1>`
  rather than to `moe_router_logits<GP>` itself.  With MT=1 routed to the GEMV form the
  control reports **0 of 131072, max|d| = 0**, and the tiled form is then genuinely
  broken rather than the harness.  **A control that is not the same code path as the thing
  it controls cannot fail.**
* **A count of "differ" without the magnitude is uninformative.**  The first version
  printed only a count, which cannot distinguish a 1-ulp difference from a structurally
  wrong logit - and only the second is capable of moving the MoE's discrete top-8.  The
  count alone would have been filed as "a rounding difference".

**The prescribed fix did NOT work, and the diagnosis behind it was wrong.**  The runtime
`nm` bound is real and is the documented private-memory shape, but it is **not** the
correctness cause: rewriting the tiled form to unroll the full compile-time `MT` and
predicate the tail's store - clamping the tail's activation address so every load stays
in bounds, and leaving `sg_sum` in the unrolled loop so the collective stays uniform -
produced a **byte-identical wrong answer**: `max|d| = 9.77069`, `max|tiled| = 11`, the
same for MT=4, 8 and 16 and the same as the runtime-bound version.  Two independent
implementations of the tiling converge on the same wrong value, so the cause is
systematic and **still unidentified**.  Deleted again; the +2.2% is real but its
correctness is not established, and this file's rule is explicit that a switch returning
wrong results faster is worse than no switch.

**THE ROUTER WAS DROPPING HALF OF EVERY DOT PRODUCT, AND THE "STILL OPEN" ENTRY ABOVE WAS THIS BUG.**  Found by building two closed-form probes into `dev/grp_rout` and then chasing them properly.  `moe_router_logits` launches `nd_range<1>(..., 32)` - a work-group of **exactly 32 threads** - with **no `[[sycl::reqd_sub_group_size(32)]]`** on the lambda, and then reduces with `sg_sum<32>`, whose mask ladder is `permute_group_by_xor` with masks 16, 8, 4, 2, 1.  On this backend that first mask-16 step reads only lanes 0-15, so **lane 0 stored a sum over half the K range**: every router logit lost the contribution of the upper 16 lanes.  Adding the attribute makes `dev/grp_rout` exact at **0 of 131072**.

| probe | what it pins | before | after |
|---|---|---|---|
| `ones` (hx = hw = 1) | logit == n_embd exactly | 0 of 131072 off | 0 of 131072 off |
| `eye` (hw[e][k] = (k==e), hx[t][k] = 1+0.25k) | logit == 1+0.25e, no accumulation at all | **32768 of 131072 wrong** | **0 of 131072** |
| random, exact within-token logit differences over the top-8 | magnitude | **64.895** | **2.3e-05** |
| random, expert sets | selection | **512 of 512 tokens differ** | **0** |

**`ones` mode was blind and that is the lesson.**  With every logit equal, any error that hits all experts equally leaves the softmax uniform, so a coverage bug of *any* size is invisible.  The probe that found it is the one whose correct answer is a **distinctive per-expert constant with no accumulation**, so the wrong ones cannot cancel: the failure was exactly the experts with `e mod 32 >= 16`, each reading logit **0** instead of `1+0.25e`, which is the signature of lane 0 not seeing lanes 16-31.  **A reference whose every answer is identical cannot find a bug whose every answer is identical.**

**This corrects a recorded conclusion in this file, in the device-profile section.**  `dev/sgprobe.cpp` measured `wg=32, no attribute -> sub_group_range=32, n_sub_groups=1` on both the A770 and the Iris Xe, and the tiled-router entry below used that to reject "a 32-thread work-group is two 16-lane sub-groups" as the cause of *its* nondeterminism.  The measurement did not transfer to this kernel.  **The rule that does hold is narrower and is about the work-group width, not the device:** a work-group of **exactly 32 threads** is the ambiguous case for the sub-group partition, while a multi-warp one is not - which is why `moe_grp_impl`, 71.9% of the prefill, gets away with the same omission.  Its `kMoEWG = kMoERG * 32` is 512 or 1024 threads, so its 32-lane sub-groups are unambiguous and `dev/grp_min`'s 0/8192 is real.  A sweep of the whole kernel tree for "32-thread work-group + hardcoded `sg_sum<32>` + no declaration" now finds exactly one site, the router, which is fixed.  `reduce_over_group` queries its sub-group at run time and is not affected.

**And it moves an oracle, which is the part that matters for everything above.**  pp512's argmax goes **4821 -> 11171** (4 of 4 separate processes), while pp16384 (1204), pp65536 (4779), `test_decode_vs_prefill` (3/3) and `test_w4_vs_cpuref` (SAME) are unchanged.  So **4821 was an artefact of the broken router**, and **11171 - the value this file twice called "the tiled router's known-bad signature" - is the correct one.**  It was the correct answer all along and the bug was what prevented it from being produced.  Every number in this entry that cites 4821 as an oracle is measuring the broken kernel; the throughput figures are unaffected (a systematically halved logit is still a deterministic function of the same inputs, and the sweep, not the argmax, is the statistic).  **The stable oracles are now pp512 11171, pp16384 1204, pp65536 4779.**

**And a correction to something this entry did twice.**  `dev/bench_tg` and
`dev/bench_pf` build **different prompts**, so their prefill argmaxes are different
quantities and must never be compared: at depth 512 `bench_tg` reports 8307 and
`bench_pf` reports 4821, and at 16384 they report 1056 and 1204.  Recognising the tiled
router's "known-bad signature" as 11171 was a **cross-harness comparison**, and the
"stale binary" story below was built on the same conflation.  The stale binary itself
was real - `bench_pf` reported 11171 before it was rebuilt and 4821 after - but the two
harnesses have never agreed on the number, and neither is the other's oracle.

**And a stale-binary scare that turned out to be exactly this file's own rule, one level
deeper.**  The gates immediately after the revert reported **pp512 argmax 11171** - the
tiled router's known-bad signature - twice, on runs that "passed" every other cell.  It
was not a regression: `dev/final.sh` **only ran, never built**, and `dev/pf.sh` executes
`dev/bench_pf`, which is a *different binary* from `build/sycl-infer`.  Rebuilding only
the latter left the prefill gate verifying a router that was no longer in the tree.  The
script now rebuilds both and **asserts each binary's mtime against
`libsycl_infer_core.a`**, because a gate that cannot tell a stale binary from a current
one is not a gate - which is what this file already says about `dev/build_*.sh`, and the
same failure one layer up.

**The tiling is not left behind as a switch.**  A knob that returns wrong results 2.2%
faster is exactly the trap this file keeps warning about, and it is indistinguishable
from a win unless someone runs the exactness check.  Reverted; the profile above and the
prescribed fix are what is worth keeping.

#### What it actually costs, measured against the approximation already being made

This is the number that matters for deciding whether to keep working on it.  Depth
16384, 35B A3B / 2x A770, post-prefill logits at the non-perturbing read point - four
processes with `attn_xmx` on and two with it off:

| | max abs diff | rms diff |
|---|---:|---:|
| run to run, `attn_xmx` ON (worst of 6 pairs) | **1.9016** | **0.3712** |
| run to run, `attn_xmx` OFF (the control) | 0.2580 | 0.0448 |
| **`attn_xmx` vs classic, i.e. the approximation the engine already makes** | **1.6813** | **0.3416** |

**The run-to-run jitter is 113% of the approximation on max and 109% on rms - the same
order of magnitude, not an order less.**  So the honest characterisation is not "int8
prefill attention is a few percent less reproducible": it is **that the run-to-run
spread is indistinguishable from the difference between the int8 attention path and the
classic one.**  Enabling `attn_xmx` multiplies a smaller, always-present
nondeterminism by about 7-8x (0.0448 rms -> 0.3712); `attn_xmx` itself is what raises it
to the scale of the approximation it replaces.

What it does **not** cost, all measured:

* **The sampled output: nothing.**  argmax 1056 in every one of the six runs at depth
  16384 and 628 in every run at depth 2048, and the generated token hash was identical
  across processes at both depths.
* **Performance: nothing.**  1313.9 t/s pp65536 and 1419.7 t/s pp16384, unchanged by
  every probe, and no arm has ever crashed a real run.

What it **does** cost is **reproducibility, and therefore the ability to see a real
regression**: the noise floor on the post-prefill logits is ~1.7 max / ~0.37 rms at depth
16384, so nothing smaller than that can be distinguished from run-to-run variation -
which is exactly why this bug survived a long search.  Depth matters: at 2048 the
spread is ~0.57% of `l1` against ~5% at 16384, roughly 10x.

`PF_ATTN_XMX=0` is a usable *reproducibility mode* (7x less noise) but it is **not a
fix**: the control's own two runs differ by max 0.258 / rms 0.0448, so there is a
smaller nondeterminism underneath that this investigation has not attributed either.

**Honest position: the divergence needs `attn_xmx` enabled, is invisible to the argmax
and to the token stream, and its run-to-run spread is the same size as the int8
approximation it replaces.**  It is not fixed; the remaining work is not more
instrumentation, and the cost/benefit of continuing should be judged against the table
above rather than against "the prefill is nondeterministic".  Ruled out by
measurement:
the perturbing oracle itself, `attn_xmx`-specificity, uninitialised memory in both
`attn_xmx` and the whole oneDNN path, submission ordering on every edge and both
domains, accumulation precision, warmup, the padded columns, and the negative `lim`.
Ruled out by reading the code: `attn_xmx`'s pad fill, stale `P` beyond `nk`,
`[blk, kBlk)`, the block-max partials, the scratch capacity (`HPG = 8`,
`n_rows = 16 = kMaxB`, `sum_row_nr = 512 = kMaxT*kMaxB`, `cap = M = 4096` - a full
batch, filling the scratch exactly), `xmx_quant_q` / `xmx_pv_acc` /
`xmx_output`, and every stale-descriptor-pointer route.

**Fixed on the way, and a real bug in its own right: the KV pool's host bookkeeping
was sized by `n_blocks` instead of `pool_cap`, which is a heap out-of-bounds read
AND write the moment the virtual pool grows.**  A block index legitimately reaches
`pool_cap - 1` - `kv_grow()` hands out indices up to `pool_blocks - 1` - while
`n_blocks` is only the initial committed count.  All three setup sites now size
`block_used_` / `block_extent_` / `pc_block_node_` to `pool_cap`.  The vulnerable
site is the virtual pool with an automatic cap, which is the default path for any
long-context run (`kv_cap_mb = -1`).  Two symptoms of the same oversight:  `alloc_block()`
indexed both arrays with no bound check and wrote past the end, and `free_block()`
*did* have a bound check - so a grown block was **never returned to the pool**, which
is why `in_use=524` of `pool=528` looked plausible rather than alarming.
`alloc_block()` now throws on an out-of-range index instead of `continue`-ing past
it, because the silent skip is what let this read as "decode is flaky at 16k".

Separately, `dev/bench_grp` says 6.0/3.4 ms per call while the engine measures
6.9/3.6 ms (274.9/143.9 ms over 40 calls), and **that gap is work-group tail
imbalance, not the kernel**: `PF_MOE_HIST`'s distribution runs from 1 to 124 pairs
per active expert, so one tile's work-group runs 1 tile and the next runs 8, and
the engine's real routing is less smooth than the benchmark's synthetic one.  With
`M=8` there are ~560 tiles instead of 300 and the imbalance is worse, which is the
price paid for the 17%; a grouped-M formulation (16 rows for the 95 experts with
p<=32, 8 for the rest) is what would recover it.

### What an external A770 kernel library did and did not settle

Reviewed the A770/DG2 fork of Intel's `omni_xpu_kernel` (the PyTorch/XRT kernel
collection) against this engine.  Most of it does not transfer, and the parts that
do are narrower than they look.  The one substantive result replaces a recorded
blocker, and it is below.

**DPAS operand delivery: the recorded blocker is real, and there is a 6x better
way to pay it.**  This file closes DPAS partly because `dev/bench_dpas_gemm`
measured 7.44 ms against `nat_gemm`'s 0.200 and attributed the cost to "building
the two 64/32-element cross-lane offset vectors ... serialised through local
memory".  That fork's DPAS flash attention never builds an offset vector: it does
`block_load` (lane-contiguous, 16-byte aligned) -> `slm_block_store` ->
`barrier` -> `slm_block_load` **at a permuted address**, i.e. it moves the
cross-lane permutation into the SLM *address*, which each lane computes for
itself.  `dev/dpas_slm` measures the three delivery paths with identical DPAS
count and identical accumulator, on the A770, oneAPI 2026.1:

| B-operand delivery | 512 WG | 2048 WG | 8192 WG |
|---|---:|---:|---:|
| A  `sv::gather` + runtime offset vector | 0.71 | 0.76 | 0.77 T-MAC/s |
| B  `slm_block_store` / `slm_block_load` | 4.60 | 4.85 | 4.92 |
| C  already in DPAS order (no transform) | 13.29 | 14.26 | **14.46** |

Three things follow. **The recorded 18x penalty reproduces** (A vs C, constant
across grid sizes - consistent with "proportional to the WORK-GROUP COUNT").
**The fork's recipe is worth 6x** (0.77 -> 4.9). **And it is still not free**:
B sits ~2.9x above C.  So the honest statement is that DPAS on this part is not
closed because the unit is slow - at 14.5 T-MAC/s arm C matches oneDNN's PV rate
and beats dp4a - but because getting operands into DPAS order costs 3x through
SLM or 18x through offset vectors, and only a **load-time pre-pack** reaches the
rate, which the 18 GiB weight wall blocks for weights and does not for
activations.  `slm_block_store` / `slm_block_load` are ESIMD built-ins
(`sycl/ext/intel/esimd/memory.hpp`) and the SLM comes from `sv::slm_init<N>()`, not
`local_accessor` - which is also the escape from this file's recorded
`copy_to(local_accessor)`-returns-zero trap.

**DPAS availability is PER MICROARCHITECTURE, and it is now a device-profile
field.**  **The A770 is Xe-HPG (DG2) and HAS the XMX/DPAS unit; the Iris Xe is
Xe-LP and does not.**  `dev/dpas_compat`: on the A770 `dpas<8,8,float>` (fp16 x
fp16, fp32 accumulate, i.e. the fork's form) **and**
`dpas<8,8,int32_t,...,u4,s8,64,32,64>` (the form this file's own probes used) both
JIT fine; on the Iris Xe **both fail**, the second with

    llvm.genx.dpas2.v64f32.v64f32.v64i32.v64i32(...):
        Intrinsic is not supported by the <XeLP> platform

(`block_load<uint32_t,64>(p, overaligned_tag<16>)` fails the same way, its
`lsc.load.ugm.v16i32` lowering).  **So a DPAS kernel here does not run slowly on
an Iris Xe - it does not BUILD**, and since the reference model runs on an Iris Xe
while the AOT target is `acm-g10`, an ungated DPAS kernel is a portability
landmine.  `dev/devinfo` shows the split is ADVERTISED, which is what makes this a
query rather than a name match:

| | `cl_intel_subgroup_matrix_multiply_accumulate` | `..._split_matrix_multiply_accumulate` |
|---|:-:|:-:|
| A770 (Xe-HPG) | yes | yes |
| Iris Xe (Xe-LP) | no | no |

**`si::dev::has_dpas_for_queue(q)` reads it from `info::device::extensions`** and
`profile::hw.has_dpas` is only the fallback, which is the order this file's device
profile section asks for (hardware facts are queried).  It is per-queue, not
per-process, because a `--layer-map` may span the two architectures.
`PF_DEVICE_INFO=1` prints `xmx(dpas)=1` on the A770.  `extensions` is deprecated in
SYCL 2020 in favour of `aspects`, but aspects describes memory/scalar/vector
capabilities and does **not** carry the matrix unit, so the deprecation is
suppressed locally with that reason.

**This file's own first inference on this was wrong, and the error is worth
recording**: the JIT message says `<XeLP>` and I read it as "both cards are
Xe-LP".  They are not - Xe-LP is the Iris Xe.  When a compiler message names a
platform, read it as naming *the device that produced it*, which here was the
only device I had run the probe on locally.

**Does not transfer, each for a stated reason.**  ESIMD int8 quantization (nothing
in the library computes a per-32 group amax; `xq.cpp`'s group-major layout plus
its f32 scale plane is strictly more expressive - and they record that replacing a
scalar sub-group design with an ESIMD SLM kernel **lost 3.7x** to an IGC
register-spill penalty).  GDN / gated delta (strictly worse: global state
read-modify-written twice per token, a 7-barrier ladder instead of `sg_sum`,
16x less parallelism - it is a *corroboration* of this file's
"sub-group-reduction-latency bound" diagnosis, not a source).  GGUF dequant (**no
Q5_K at all** - it has Q4_0/Q4_1/Q8_0/Q4_K/Q6_K only, so half this model's expert
bank has no counterpart; and it always materialises to float, i.e. exactly the
3.56x expansion the scratch/weight-store work here rejected).  Its K-quant
intermediates are computed **in fp16** (`gguf_dequant.cpp:298-311`), which is a
precision regression against `quant.h`'s fp32 and must not be copied.  Head-blocked
RoPE and fused rms+rope are **BMG-gated**, so A770 runs a weaker 6-barrier version
that re-reads the row from global.

**Measured, not assumed: the fpmath guard is free, and the other two are not worth
building.**  (1) It records that an A770 **u4 oneDNN matmul overflows to NaN when
K >= 5376** with narrow fp16-class accumulation, fixed by
`attr.set_fpmath_mode(fpmath_mode::any, true)`, and `dnnl_gemm.cpp` set **no**
fpmath_mode at all.  `PF_DNNL_FPMATH` now wraps both attr sites, **default off**.
Measured A/B, same run, argmax as the gate:

| | default | fpmath=1 |
|---|---:|---:|
| pp512 | 775.0 t/s, argmax 4821 | 774.5, argmax **4821** |
| NT=16384 | 1414.5, argmax 1204 | 1410.8, argmax **1204** |
| NT=65536 | 1312.8, argmax 4779 | 1306.2, argmax **4779** |
| tg128 depth 512 | 38.1 ms, hash 8787d83 | 38.3 ms, **same hash** |
| tg128 depth 64k | 49.9 ms, hash 82df23a | 50.2 ms, **same hash** |

**Neutral everywhere and bit-identical**, which is what a guard should cost.  (The
64k *means* were 52.0 vs 74.9 ms - that is four reps of drift on this box, not an
effect; `best` is the statistic, per the benchmarking rules in this file.)  **The
NaN itself is still the library's claim and has NOT been reproduced here** - no
model in reach reaches K >= 5376, since this engine's largest oneDNN int8 matmul K
is the 27B's 5120 and the 35B A3B's 2048.  So the knob is shipped OFF and is a
guard for the next model up, not a fix for an observed bug.

(2) Its RoPE carries a **precomputed 2x2 rotation matrix per pair** instead of
computing `exp2` per lane per call, and (3) its reduction is **one barrier, then
every work-item reloads all partials as a vector and reduces redundantly**
(`norm.cpp:1134`) instead of a tree.  **Neither is worth building, and the price is
measured rather than argued.**  `PF_PROF`'s norm aggregate - `rmsnorm` plus
`qk_norm_rope` together - is **1.07 ms of a 48.65 ms prefill step (2.2%)**, 1.09 of
49.59 on a decode step (2.2%), and 1.85 of 75.71 in the per-step decode accounting
(2.4%).  Both ideas attack a fraction of 2.2%, and the rotation table is not free
in memory: a `[pos][pair][2][2]` f16 table is **134 MB at ctx 131072 and n_rot
128**, i.e. it would spend 4% of a card's VRAM to remove an `exp2` from 2.2% of a
step.  Build the persistent-tile expert GEMM instead; that is where the 55% is.

**Two of my own suspicions were raised, tested with a control, and refuted - the
control is the part worth keeping.**  I thought `dequant_sb_lane_impl` applied the
wrong per-32 group to half of every lane's values.  A host-only replay said
240 of 256 differed - and the CONTROL arm, the grouped mapping that `dev/grp_min`
verifies at 0/8192 on device, also said 219/256 differed, which made the test, not
the kernel, the suspect.  Redone with the correct 8-lane grouped convention and a
random scale plane, the control came out 0/256 and so did the consumer convention
that `moe_dot`/`gemv.cpp` actually use: **`w[b]` is group `b`'s value `lane`, and
`sc[b]` is right.**  Two lessons, both already in this file in other forms: a
failing control invalidates the whole probe, and *do not build an arm from your own
algebra when the codebase has a convention you have not read* - I constructed a
placement (`64*(b>>1)+2*lane+(b&1)`) that no caller uses.  Separately, the
suggestion to hoist `d*sc[b]` out of that loop does not apply: a lane's 8 values
are in 8 **different** groups, each used once, so there is no reuse to exploit and
the 3-multiply form is already minimal.  `PF_MOE_DQ2`'s 12% is irreducible
arithmetic, not missed CSE.

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
* **The recurrent state's slot axis is `state_slots_`, not `kMaxB`.**  Every
  `[layer][slot][...]` stride into `d_gdn_state`/`d_conv_state` (device or host)
  is built from `state_slots_`, the clamped `engine_config::max_slots`
  (`--max-slots`, default `kMaxB`), and the allocation is
  `state_slots_ * n_gdn_local * per_slot`.  The two differ as soon as anyone
  passes `--max-slots`, and mixing them writes past the end of a USM pool rather
  than slowing anything down - which is the failure mode this file keeps
  repeating.  `zero_slot`, `pc_admit`, `pc_capture_begin` and `pc_commit` all
  bound-check against `state_slots_`, and `scheduler::loop`'s slot search only
  offers slots below `engine::max_slots`, so the index can never legitimately
  reach past the arrays.  Everything else in the engine still addresses *rows* by
  `kMaxB` (`step_info`, the block table, `d_logits`, `d_partials_dec`) because
  those are per-sequence, not per-state, and are small.

* **YaRN's two blends are written to be exact at their anchors, and that is not
  cosmetic.**  `yarn_build` combines three sequences (`interp`, `extrap`, and a
  ramp mask) and both blends are deliberately written in difference form
  (`blended + (extrap - blended) * (1 - m)`) rather than the equivalent convex
  form (`a*m + b*(1-m)`).  Each is exact at one anchor and carries an ulp of the
  *larger* term at the other: the convex form makes the untouched high-frequency
  head a 1-ulp ramp, and the wrong difference form makes the fully-stretched tail
  wrong by an ulp of `extrap` - which, at factor 32, is ~2e-6 relative instead
  of 0.  `test_yarn` asserts both anchors bit-exactly against the plain table, so
  a "simplification" to the convex form fails there rather than in a 1M-token
  generation.

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
  cross-device USM access cost another ~12x.  The matmuls run on `p->st` with the
  rest of the forward (see the `st_a` entry below - they used to be on their own
  stream and that was a race).  The softmax is one work-group per query row
  with a single vectorized `exp` pass.  The per-row index arrays (`orow` /
  `oh` / `olim`) are built by a device kernel from `info`, **not** by host
  `q.memcpy`: three host->device copies per kv head kept the host pinned to the
  GPU (and raced with the reused host vector) and cost ~2x.  Measured marginals
  on the deepest full 512-token mode-2 chunk (27B / 2x A770, i8 KV,
  `PF_PREFIX_CACHE=0`, `--layer-map`): pp512@16k 296 -> 911 t/s, pp512@64k 93 ->
  540 t/s (both above the OpenVINO targets of 640 / 362).  Below the key
  threshold the classic kernel still wins (4k: 718 vs 649 t/s), hence the gate.
  Scope: i8 KV, head_dim 256, `n_real > 1`; other cases fall back.
* **A SYCL stream built from an in-order queue is NOT ordered against that queue's
  own submissions, and the attention matmuls were on the wrong side of that.**
  `attn_qk`/`attn_pv` used to execute on a second oneDNN stream `st_a`, created by
  the same `sycl_interop::make_stream(p->eng, q)` call as `p->st`, on the stated
  theory that "it wraps the same in-order queue, so the matmul is ordered after the
  Q-quantize / K-gather kernels without an explicit wait".  **That inference is a
  category error: a stream is its own ordering domain.**  SYCL orders work *within*
  a stream; two streams are unordered regardless of how the queue was created.  So
  in `attn_xmx`'s per-block-chunk loop the next iteration's `xmx_gather` (main
  stream) overwrote the `b.k`/`b.v` rows the current iteration's `attn_pv` (`st_a`)
  was still reading, and `attn_qk` could read K before the gather wrote it.
  **This is NOT the cause of the 64k argmax churn, and getting that wrong burned
  three wrong hypotheses in a row.**  Five 65536-token prefills of one binary in
  five separate processes gave argmax **13775, 13775, 264, 44896, 4779** with timing
  stable to 0.2%, and `PF_NOGRAPH=1`, `PF_PF_PIPE=0` and `PF_ABL_NOMOE=1` all
  still churned - which rules out the recorded graph, the 3-phase pipeline and the
  routed experts - while `PF_ATTN_XMX=0`, a path that never touches `st_a`,
  produced 264 as well, which rules out this.  **The real cause is `dev/bench_pf`,
  and it was printed in every single run**: see *The depth cells were run against a
  512-block KV pool* below.
  In order the three guesses were: (a) this race, killed by `PF_ATTN_XMX=0`;
  (b) the MoE CSR's atomic scatter, killed by reasoning rather than by a run - the
  grouped GEMM accumulates over k for each row independently, so a permutation of
  an expert's rows changes no output; (c) a near-tie amplified by exposure (128
  chunks against 16k's 32), killed by a *third* run at 32768 giving 13775 when the
  first two had both given 4821 - so there was no threshold either.  **What found
  it was not an ablation at all: it was diffing the engine's entire printed output
  between two runs of the same depth.**  The argmax moved, and the only line with
  anything to say was `[kv] exit: pool=512 ... cap=512`, which had been there the
  whole time.

  * The fix is one line anyway - run the attention matmuls on `p->st`, where every
  dense GEMM and every SYCL kernel already lives and where ordering is known to
  work.  Nothing inside `attn_xmx` waits in the shipped path, so the split's
  original motive - an attention sync draining the dense GEMMs - never arose.  It
  removes an unsound assumption; it is **not** claimed to fix the argmax churn,
  which is not its symptom.  The general primitive would have been
  `ext_oneapi_submit_barrier()` events in both directions; that is unavailable
  here because `sycl_interop::stream` exposes only a blocking `wait()` and cannot
  produce a barrier event, and this SYCL's `sycl::stream` has no `(queue)`
  constructor at all.  `PF_ATTN_STREAM=1` reproduces the split for A/B.
  * `PF_XMX_BREAKDOWN` is still unusable as an oracle at any depth (4779 / 279 /
  3069 / 44896 / 304 / 32288 across work-group shapes), for the separate and
  sufficient reason that its `q.wait()` calls change the ordering.
  **Generalisation: "same queue" is not an ordering guarantee, and a comment
  asserting one is worse than no comment** - this one had been read as settled for
  the lifetime of the feature.  **And the meta-lesson: I reached for the most
  recently touched subsystem to explain a nondeterminism, and the control that
  killed it was one env var on a different path.**
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

* **A MoE model's expert strides are per *layer*, not per tensor, and the file
  decides.**  The 35B A3B has Q4_K gate/up everywhere but Q5_K down for 37 layers
  and **Q6_K down for 34/38/39**.  One `plane_bytes` per tensor family reads the
  wrong weights on exactly those layers - it loads, it answers, and it is wrong in
  a place no aggregate catches.  `moe_slot_{g,u,d}b[]` are indexed by layer for
  this reason; say the layer count and the distinct values out loud at startup so
  the mixed case cannot be silent.

* **oneDNN eligibility is per *call* and all-or-nothing - one f32 tensor takes
  the whole plan call down with it.**  GDN's `ssm_beta`/`ssm_alpha` are f32; sharing
  a plan call with the convertible `wqkv`/`wgate` made the entire call fall back to
  the fp32 dequant GEMV.  That is 12.3 vs 3.4 ms/token disappearing for a shape
  reason, not a numerics one, and it looks exactly like "the kernels got slower".

* **A host pass that fills a sparse buffer has to run *before* its consumer, not
  after.**  `moe_down_launch` returns on `slot < 0`, so `d_moe_acc`'s non-resident
  rows still hold the previous layer's output, and `moe_reduce_launch` weights
  **every** row.  Adding the host contribution after the reduction therefore
  double-counts a stale row and then adds the right answer on top.  The signature
  is unmistakable and was read as a dequant bug for hours: **layer 0 exact because
  the buffer is still zero, 2-4x too much routed energy from layer 1 on**, worst
  relative sumsq 3.29 -> 0.0088 once the order is right.  Per-layer fingerprints,
  not a layer-0-only dump, are what showed it - see the next entry.

* **A per-layer diagnostic that reports one layer is a verifier that agrees with
  the bug.**  `PF_MOE_REFDUMP` dumped layer 0 only, so the mixed-quantization
  defect above and the stale-accumulator defect here both looked "fine" at layer 0
  while every later layer was wrong.  It now dumps all 40 with
  `PF_MOE_REFDUMP_ALL`, and each layer's name carries its index because two dumps
  of the same name are indistinguishable.  **A timing or fingerprint breakdown has
  to be checkable against an independent total before it is believed** - this is
  the third time that rule has paid for itself in this repo.

* **The GGUF's expert tensors are three *separate* tensors, so a packed cache
  entry changes where `up` starts.**  `gate_exps.data + e*plane_bytes + plane_bytes`
  is expert e+1's gate plane, not expert e's up plane.  A RAM tier that packs
  `[gate][up][down]` per expert therefore needs different base pointers from the
  mmap path; giving the mmap path the packed ones evaluated every expert against
  expert 0's weights.  Symptom: the split runs fine with the RAM tier off and
  emits garbage with it on.

* **A command graph cannot contain a host pass, and every replay site has to know
  it.**  The ncmoe host half has blocking transfers and an expert set chosen per
  step.  There are **three** places a forward is replayed from a graph (plain
  decode, multi-device decode, MTP verify) and each one was missed in turn; a
  recorded graph does not merely skip the pass, it silently drops every expert
  without a device slot.  Grep for `moe_host_split()` - it is the whole rule.

* **A page-release sweep that asks a component about itself has to run *after*
  that component has decided.**  `release_host_weight_pages()` sweeps from
  `alloc_buffers()`, which is where `setup_moe_slots()` also runs - **earlier**.
  The keep-set asked `moe_host_split()` and got `false` because the cache had not
  been initialised, so the expert pages were `MADV_DONTNEED`d and then re-faulted
  one row at a time by the host half.  It is not incorrect, so nothing failed; it
  is unobservable without measuring the RSS release.  `release 21603 -> 2396 MB`
  is the number that says the keep-set actually applied.

* **MTP's draft layer does not share the trunk's activation buffers.**  The draft
  runs out of `d_mtp_xnorm` (and writes `d_mtp_x`), so a host pass that hardcodes
  `d_xnorm` evaluates its experts against *another layer's* activation.  Every
  kernel-launcher boundary that takes "the activation" needs the caller to say
  which one - `moe_cpu_experts` now takes the buffer as an argument for this.

* **A routed MoE model runs on the CPU backend, and the host half has to mirror
  the device kernels exactly.**  `record_forward`'s MoE branch is dispatched on
  the *partition's kind* (`dev_kind_[dev] == 1`), not on `multi_dev`: an all-`cpu`
  run and a `--layer-map` with a `cpu` tail both take it, and calling the SYCL
  launchers unconditionally is a startup failure ("No kernel named
  moe_router_logits"), not a slow path.  `src/backend/cpu/kernels/moe.cpp` mirrors
  `moe_router_launch`/`moe_gate_up`/`moe_down`/`moe_reduce`/`moe_gate_add` step for
  step - same rank-counting top-k (ties to the lower index, no sort), same
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
  and only the *decode* handoff showed it (the multi-device prefill runs pipelined
  with explicit flushes), so the prefill matched and the first generated token was
  already wrong.  `gpu+gpu` and `cpu+cpu` were both correct, which is what ruled
  out the host kernels.  A diagnostic that dereferences `d_xnorm` on the host is a
  SEGV here for the same reason - on a GPU partition it is a *device* pointer -
  and it must be read through the queue and skipped while a graph is recorded.

* **A recorded decode graph has the previous pass's device bindings baked into
  nothing - the host side must rebind on entry.**

* **A diagnostic that loops `n_expert` times *inside one work-item* is a
  parallelism bug, not a cost bug.**  `moe_router_topk` computed each expert's rank
  by scanning all `n_expert` experts to count how many beat it - O(n_expert^2),
  which is fine - but it launched `range<1>(n_tok)`, so a **decode's 65536
  comparisons ran in a single thread**, twice over (ids and weights).  Ranks are a
  permutation of 0..n_expert-1 because ties break on the index, so the loop
  parallelises across experts with no coordination at all.  Measured on the 35B
  A3B the router - 0.5 M-MAC of real work - was **4.4 s of a 4.6 s decode**.
  Fixed: one work-item per (token, expert).  `router=4393 -> 48 ms`, and the same
  prompt's decode went 2.13 -> 21.58 t/s.

  The lesson generalises: **measure before restructuring.**  The routed *experts*
  were blamed for the prefill's 94% for two whole attempts (SLM row staging, then a
  pair-tiled dequant reuse) and both were worth <15% - the router was 46% of it and
  the experts' shape was only an issue because they were never the top entry.
  `PF_MOE_TIME` exists for this now.

* **K-parallel reduction changes the router's fp summation order, and MTP at
  k>=2 is sensitive to it.**  Same measurement on the 35B, `gen` of "The capital
  of France is":

  | | mtp 0 | mtp 1 | mtp 2 | mtp 4 |
  |---|---|---|---|---|
  | before | `**Paris**.` | `**Paris**.` | `** **Paris*` | `**Paris**.` |
  | after | `**Paris**.` | `**Paris**.` | `The capital ` | `The capital capital of's capital of ` |

  `mtp 0`/`1` are unaffected and `test_gpu_vs_ref` still reports argmax SAME; the
  perturbation is one ulp in the router logits, which only moves *marginal* draft
  decisions.  It is shipped because the decode is 10x and the prefill 1.6x faster,
  and because MTP was already documented as not byte-guaranteed - but it is a real
  quality trade at k>=2, not a free win, and `test_spec` (the oracle) fails on this
  box for unrelated pre-existing reasons so it could not arbitrate.

* **Open (large): a routed-MoE model's prefill is ~17x off what its own dense
  layers do.**  Measured on the 35B A3B / 2x A770 / `--layer-map
  0-19:gpu.0,20-39:gpu.1`, one 441-token prompt through the server: 148.8 t/s
  whole-model, and **2605 t/s with `PF_ABL_NOMOE=1`** - the routed experts are 94%
  of the prefill and removing them is a 17.5x speedup.  The arithmetic and the
  bandwidth are both ~200x away from that, so it is the kernel shape and not the
  work: one warp per (active expert, output row), looping over that expert's
  routed pairs with `moe_dot` re-reading and re-dequantizing the weight row
  **per pair**.  A 512-token prefill routes 4096 pairs over 256 experts, so each
  row is dequantized ~14 times over, and the dequant is a dozen integer ops per
  value against one FMA of useful work.
  * Staging the dequantized row in SLM (one sub-group per work-group, so the
    staged row owns its own 16 KB) removed the redundancy and changed the wall
    time by nothing - 139 vs 148 t/s - because it also cut occupancy from eight
    warps per work-group to one.  Reverted; the occupancy/dequant trade is not
    the answer either way.
  * One real bug was found on the way and is worth not re-making: `sg_sum` is a
    sub-group collective, so it has to be called by **all 32 lanes**.  Moving it
    inside `if (lane == 0)` still compiled, still produced plausible text, and was
    wrong.
  * What is left is a proper grouped-GEMM shape: one work-group per active expert,
    tiling (rows x the expert's tokens) so both the weights and the activations
    are staged once and reused - the structure the dense prefill path already has
    through oneDNN, which is what makes the 27B's 512-token prefill 2.4k t/s.
  * Reference points for the same model on comparable hardware: llama.cpp Vulkan
    4x A770 `qwen35moe Q4_K` pp512 242.5 / tg128 49.2; 4x A770 Q8_0 pp512 497 /
    tg128 17.6; llama.cpp CPU-only on this box pp512 80.8 / tg128 7.4 at depth 0,
    36.5 / 2.0 at depth 64k.  (Post-8xxx builds regressed MoE PP on Vulkan, so a
    pre-regression build is the fair comparison.)
  `record_forward` only rebinds at
  a partition boundary (`dev != prev_dev`), so a direct-replay pass entered with
  the MTP draft's bindings still installed silently read and wrote the *draft
  device's* buffers for the first partition.  Graph-replayed passes are immune
  (their pointers are baked in), which is why this only ever showed up with
  ncmoe disabling them.

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
