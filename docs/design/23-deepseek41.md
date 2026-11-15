# DeepSeek-V4.1-Flash (`deepseek41`)

DeepSeek-V4.1-Flash is a 40-layer, all-MoE decoder whose forward differs from
every other architecture here.  This file is the source-of-truth summary of
what the engine implements, which files own each part, and the facts that fail
silently if changed.  The reference math is DeepSeek's `inference/model.py`, as
ported by the torch oracle used during bring-up; the GGUF metadata is the file.

## Geometry (reference file `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS`)

| item | value |
|---|---|
| layers / width / vocab | 40 / 5120 / 129280 |
| attention | 64 query heads, **1** KV head, key = value = 512, `rope.dimension_count = 64` |
| RoPE | last 64 of the 512 dims; **layer-dependent base** |
| q | low-rank `wq_a 5120→1280`, RMSNorm, `wq_b 1280→32768` |
| output | `wo_a` 8 groups × (4096→1024), concat 8192, `wo_b 8192→5120` |
| window | 128 tokens (includes self), all layers |
| compressed KV | owners `[2,8,14,20]`; `compress_ratios = [0,0, 2×18, 1×20, 0×3]` |
| DSA indexer | 32 heads × 128, per-query top 512 rows; sources `[2,8,14,20,24,28,32,36]` |
| candidate stage | layer 20 only: block max (8 rows), newest block pinned, top 2048 blocks |
| mHC | 4 streams, 24 mix outputs, Sinkhorn 20 iterations, eps 1e-6, lagged pre-mix |
| MoE | 384 routed experts, top-6, FFN 2304; 1 shared expert; `expert_weights_scale 1.5` |
| gating | `sqrt(softplus(logits))`; bias added to **selection only**; `w/(Σw+1e-20)·1.5` |
| SwiGLU | one-sided clamp: `min(gate,10)`, `clamp(up,-10,10)` |
| Engram | layers 1 and 14; n-gram 4, 8 heads per n-gram (24 columns), 256-wide rows |
| RMSNorm eps | **1e-20** everywhere |
| vision | separate mmproj, `clip.projector_type = deepseek41v`, projection 5120 |

Layer 0/1 have ratio 0 (window only).  Layers 2–19 have ratio 2 and read the
cache of the most recent owner `≤ il` (2, 8, 14).  Layers 20–39 have ratio 1
and read layer 20's per-token cache.  A layer with ratio 0 after a source still
reads that source's rows; only the layers before the first owner are
window-only.

## Files

| file | role |
|---|---|
| `src/model/arch/deepseek41/load.cpp` | loader; binds MLA, compressor, indexer, mHC, MoE, Engram; fills `hparams::arch` (a `ds41_hparams`) |
| `src/model/arch/deepseek41/profile.cpp` | model profile: `fwd = deepseek41`, `vision = deepseek41v`, policy |
| `src/model/arch/deepseek41/hparams.h` | `ds41_hparams : arch_hparams` (sources resolved to `kv_src_of`/`topk_src_of`), attached as `hparams::arch` and read through `ds41_hp(hp)` |
| `src/model/arch/deepseek41/tensors.h` | `ds41_layer` (per-layer MLA / compressor / indexer / Engram / mHC tensor views) and `ds41_model_extras` (Engram token map + table), held in `layer_t::extras` / `model::extras` |
| `src/model/arch/deepseek41/accessors.h` | `ds41_layer_of(L)` / `ds41_model(m)` pimpl accessors |
| `src/model/model.h` | `hparams::arch` (base `arch_hparams`) and the generic `layer_t::extras` / `model::extras` slots |
| `src/engine/arch/deepseek41/forward_host.cpp` | validated host forward (mHC, MLA, compressed KV, DSA, Engram, MoE), incremental `DecState`, prefix cache |
| `src/engine/arch/deepseek41/forward_gpu.cpp` | the 2×A770 GPU forward (same semantics, device kernels) |
| `src/engine/arch/deepseek41/forward.h` | the forward entry points (`ds41_generate`, `ds41_gpu_*`); includes them only from the architecture's own TUs |
| `src/engine/arch/deepseek41/runtime.cpp` | the `arch_ops` hooks (setup, setup_weights, dense_store, destroy, generate) |
| `src/engine/arch/deepseek41/state.h` | the composed runtime state (GPU handle + dense-store flag) |
| `src/engine/arch/deepseek41/dense_store.cpp` | the per-device oneDNN k5/int8 dense stores (`setup_ds41_dense_store`) |
| `src/engine/arch/deepseek41/prefix_cache.h` | chained-hash longest-prefix state cache |
| `src/backend/gpu/kernels/ds41_iq2.{h,cpp}` | raw IQ2_XXS/XS/S expert GEMV (no expansion pass) |
| `src/backend/gpu/kernels/ds41_attn.{h,cpp}` | MLA attention, compressor, RoPE, fp4, indexer score |
| `src/backend/gpu/kernels/ds41_hc.{h,cpp}` | mHC mixes/pre/post |
| `src/backend/gpu/kernels/ds41_ops.{h,cpp}` | fp8 round, swiglu clamp, router, select, candidates, gather, window write, Engram apply |
| `src/mm/vision.cpp` | `deepseek41v` tower + 3×3 merge projector + sentinels |
| `src/server/chat.cpp` | `render_chat_deepseek` structural fallback (minja cannot parse the template) |

## Forward

The forward is **model-owned**: it does not use `seg_plan` or the paged KV pool.
Its `arch_ops` row (`src/engine/arch/registry.cpp`, hooks in
`src/engine/arch/deepseek41/runtime.cpp`) declares `owns_kv_pool` and
`owns_request_path`; the engine's lifecycle sites ask `engine::owns_kv_pool()` /
`owns_request_path()`, and `generate_impl` hands the request to
`arch_ops::generate` (`ds41_generate` on the CPU or the GPU path).

Per layer, in order:

1. **Engram** (layers 1, 14): 24 columns from a rolling xor of the last 4 mapped
   token ids (`row = rolling % vocab[col] + off[col]`), 24 rows of 256 gathered
   from the 110 GiB IQ4_NL table, `engram_kv` 6144→25600, four keys + one value,
   signed-sqrt gate, added to all four streams.  Skipped for image rows.
2. **mHC attn**: `mixes = (flat·fnᵀ)·rsqrt(mean(flat²)+1e-20)`; pre/post/C as
   above; Sinkhorn = softmax over the 4 destinations, `+eps`, one column pass,
   then 19×(row,column), every denominator `+eps`.  The attention input uses
   the **previous** sublayer's pre-mix (layer 0 is stream 0 verbatim).
3. **Attention**: `qr = rms(x·wq_aᵀ)`, `q = qr·wq_bᵀ` (64×512); `kv = rms(x·wkvᵀ)`
   (one 512-wide head); RoPE on the last 64 dims.  Keys = the 128-token window
   plus the selected compressed rows; per-head sink is an extra value-0 key;
   `V = K`; output de-roped, `wo_a` groups, `wo_b`.
4. **Compressed row** (owners only): ratio 2 = per-feature softmax over the 2
   tokens' compressor scores, weighted sum of the two `wkv` rows, RMSNorm,
   RoPE at the **block start**, append `[K512 | indexK128]`; ratio 1 = the
   owner's own latent at the token.
5. **DSA**: index q from `qr` (32×128, RoPE on its last 64), index k from the
   **pre-RoPE** latent (128), `score = Σ_h relu(q·k)·w`, `w = x·projᵀ/√(128·32)`;
   top 512 rows; layer 20 additionally keeps 2048 candidate blocks of 8 and
   later index sources restrict to them.  Selection is reused from the previous
   index source otherwise.
6. **mHC ffn** (uses this layer's attn pre-mix), RMSNorm, MoE, mHC post; save
   the ffn pre-mix for the next layer.
7. Final `rms(hc_pre(h, last ffn pre), output_norm)`, untied head.

RoPE: ratio-0 layers use the model base with no scaling; ratio>0 layers use
`compress_rope_freq_base` + YaRN (factor 16, orig 65536, β 32/1) with the net
mscale cancelled to 1.0.  RMSNorm is always `1e-20`.

## Host implementation and measurements

`src/engine/arch/deepseek41/forward_host.cpp` is a scalar/AVX f32 implementation: it dequantizes the dense
weights once into RAM (routed experts are streamed from the mmap), keeps a
`DecState` per sequence (40 window rings of 128×512, four compressed caches of
640-wide rows, compressor rings, Engram history), and decodes one token at a
time.  Measured on the 2×A770 box, CPU backend:

* steady decode **~0.93 s/token** (naive whole-sequence recompute: 2.87 s/token);
* the raw completion `The capital of France is` -> ` Paris.` (ids 11111, 16, 1);
* a confident prompt (`The quick brown fox jumps over the lazy`) -> ` dog.` + EOS,
  identical to llama.cpp;
* chat `2+2` -> `4`; image chat (thinking off) -> `HELLO 123`; thinking on ->
  coherent reasoning.

The golden prompt's second token differs from llama.cpp (a near-tie between two
"." tokens and EOS); it is a numerics boundary, not a forward error.  All GGUF
dequantizers (IQ2_XXS/XS/S, IQ3_XXS, IQ4_NL, Q5_K, Q8_0) are bit-identical to
llama.cpp on the real tensors; IQ3_XXS was missing at first and its stale buffer
was masking the issue.

### Prefix cache

`src/engine/arch/deepseek41/prefix_cache.h` keys a chained FNV-1a hash per token and stores
`DecState` snapshots at the prompt end and every 16 tokens, with a 512 MiB
LRU-by-bytes budget.  A shared prefix resumes and only the suffix is
forwarded.  `ds41_generate` reports the reused count; the server maps it to
`usage.prompt_tokens_details.cached_tokens` and `prompt_cache_hit_tokens`.
Measured 2.7× on a 32-token reused prefix, with bit-identical logits.  This is
the **host** forward's cache: it snapshots `DecState`, which contains the
window, the compressed caches, the compressor rings and the Engram history, and
restores it for a reused prefix.  The GPU forward keeps the same state in
device buffers and currently does not snapshot it into this cache (the bank
stays warm, but a repeated prompt re-prefills); a device->host snapshot is a
clear extension point.

### Multimodal

`deepseek41v`: 32 blocks of RMSNorm, fused QKV + bias, 16 heads × 64, 2D vision
RoPE, bidirectional attention, SwiGLU, final `post_ln`; a 3×3 merge with the
patch grid zero-padded to multiples of 3, `mm.1` 9216→5120 + erf GELU, `mm.2`
5120→5120; then `[img_start] (row cells + [image_newline])* [img_end]`, i.e.
`N = ceil(H/42)·(ceil(W/42)+1)+2` tokens.  The text side feeds the rows at
plain 1-D positions, bypasses `tok_embd`, skips Engram, and uses
`exp_probs_b_vl` for the router.  The tower agrees with llama.cpp clip at
cos 0.9998.

The tower's linears default to the text model's oneDNN int8 matmul
(`PF_VISION_DNNL`, **on for `deepseek41v`**) when the tower is encoded on a
**GPU queue**: a 640x320 page (202 tokens) measures **17.6 s** on the bf16
`vit_gemm` path against **~1.7 s** on the int8 path, cos 0.9996 vs clip, and
the image chat still answers `HELLO 123` (`docs/PERFORMANCE.md` §5.5).  A
tower-level A/B on one A770 (Release JIT, 324 patches) re-measures **1305 ->
146 ms** with every linear registered (`[vtime] dnnl vision: on`, 490.9 MiB)
and cos 0.9969 / rel-L2 7.8 % against the fp32 host reference on a synthetic
gradient image.  All of the tower's linears (qkv/out/gate/up/down and
`mm.0`/`mm.2`) satisfy the K % 32 rule, so the compact upload is available
here.  `PF_VISION_DNNL=0` restores `vit_gemm`.

**The CLI's deployed `--device cpu` path is not one of those GPU-queue cells.**
With a CPU text queue the vision model is uploaded and encoded on that CPU
queue, so the oneDNN instance is not created (the int8 path there is oneDNN's
CPU engine, which measured slower than `vit_gemm`): it prints `[vision]
PF_VISION_DNNL needs a GPU queue; using vit_gemm` and runs the 18.3 s CPU
`vit_gemm` tower.  A full-page 640x320 run on 2026-11-15 (`--device cpu
--cpu-threads 28`) measured **728 s wall** (dominated by the 265 GiB text +
Engram load and the CPU prefill) and still answered `HELLO 123`; the
`PF_VISION_DNNL=1` cell reaches the same output through the fallback.  The
17.6 -> ~1.7 s cell is the standalone GPU-queue tower measurement, not the
CPU-text CLI path.

## GPU kernels (validated microbenchmarks, one A770)

| kernel | shape | time | accuracy vs host |
|---|---|---:|---|
| IQ2 GEMV | K=5120,N=2304, M=1 | 37–42 µs (73–91 GB/s) | max abs 1.5e-6 |
| IQ2 GEMV | K=2304,N=5120, M=1 | 38 µs (99 GB/s) | max abs 7.2e-7 |
| IQ2 GEMV | M=4 | 73–81 µs (2.0×/row) | same |
| mHC | K=20480,N=24, 20 iters | 0.045 ms | ≤1e-6 |
| MLA | 64×512, 128 window + 512 selected | 0.60 ms/layer | 3.1e-7 |
| compressor | ratio 2, 8 groups | 0.051 ms | 9.5e-7 |
| indexer score | 512 rows, 32×128 | 0.014 ms | 1.5e-5 |

The IQ2 GEMV is codebook/issue-bound, not DRAM-bound; the coalesced
value-per-lane mapping is 3–6× slower and is not shipped.  `sycl::exp2(int)` is
not exact on this device/JIT; fp4 powers of two are built from float bits.

The GPU engine path is `src/engine/arch/deepseek41/forward_gpu.cpp`: dense weights are uploaded per
tensor and split 0-19/20-39 (5.74 GiB), the routed experts live in a ds41-owned
LRU bank backed by the mmap, and `--device gpu` is selected with a layer map;
`PF_DS41_GPU=0` keeps the host forward.  The ds41 GPU path still skips the
generic seg_plan oneDNN partition (`setup_md_dnnl` declines for it), but since
2026-11-11 the *dense* linears use their own per-device `dnnl_gemm` stores -
k5 for Q5_K - see **Dense oneDNN stores** below.  Q8_0, BF16 and the routed
experts stay on the raw path.

### Expert split, copy crew, KV skip (design)

The 2026-11-07 measurement basis is in [§History](#history-superseded-measurements);
the design it introduced is current.  Measured on 2x A770, ctx 512,
`--moe-vram-mb 16000`, greedy, one process per cell: the 32-token greedy text
is **byte-identical** to the unsplit path.

* **Expert split across cards** (`PF_DS41_SPLIT`, default on for 2 GPUs).
  `banks[li*ndev + c]`: ids below `0.34 * E` live on card 0, the rest on card 1
  (`PF_DS41_SPLIT_RATIO`, the measured 5.18:9.98 GB/s link ratio), so one
  layer's misses use both links at once.  The budget split is separate
  (0.47/0.53, `PF_DS41_SPLIT_BUDGET_RATIO` or `PF_DS41_BANK0_MB`/
  `PF_DS41_BANK1_MB`) because card 0 also holds the dense MLA weights, the LM
  head and three of the four compressed caches.  Each card's budget is reduced
  by its window/compressed-cache bytes plus 512 MiB, so 512..262144 all
  allocate at 16000 MB.  The owner computes its experts into per-`j` slots
  while the peer does its own, the peer's slots come back as one 120 KB copy,
  and the six `yadd`s are replayed in top-k order - the same dots and the same
  add sequence, hence bit-identical output.  The router stays on the owner
  (its weight is uploaded only there).
* **Spin-waited copy crew** (`ds41_copy_crew`, 3 helpers + the caller).  The
  per-miss 9.2 MB mmap->pinned memcpy was 72 ms/token unsplit and 119 ms split
  on one host thread; four copiers splitting each plane in half take it to
  22/35 ms, and the staging ring is 8 slots (a layer selects up to 6 experts).
  Condvars were the problem (a wake per 0.3 ms copy cost ~0.5 ms/miss); the
  handshake is a spin generation counter.
* **The generic paged-KV pool is skipped** for ds41.  The forward owns the
  window and compressed caches, but the multi-device `kv_setup` re-derived
  `n_blocks` from `pool_cap` and committed 10.6 KB/token/card (1.28 GB/card at
  ctx 128k) - that was what made the split bank fault at depth.  `kv=1` now.

The prefill MoE uploads are ~42 misses/token, about one per layer, so each
9.2 MB H2D is exposed by the layer dependency; batching tokens does not help
because 384 experts with top-6 and strong adjacent-token locality give a
~29-expert union per layer over 8 tokens (most already resident), i.e. ~1 pair
per expert.  Collapsing a miss's three plane copies into one padded slot is
neutral (measured, reverted).  Text-only 18000 MB is the recommended budget
(16000 keeps room for the vision tower).  The `nohead`
ablation's 60 ms was a workload change (stale logits stop routing from moving
and cut 6.7 GB of misses), not the head (4.1 ms at 110 GB/s).  The dense Q5_K decode GEMV ran at 110 GB/s and was the largest remaining item;
it now uses the native k5 store (**Dense oneDNN stores** below), which took
decode to 124 ms/token and prefill to 133.  Bank hit saturates at ~60 %
(R 20-24 vs 384 experts).

### Dense oneDNN stores (2026-11-11)

The walk is per-token M=1, so the generic seg_plan stores never applied to it
and every dense GEMV read the raw GGUF type through
`gemv_dec_vec_kernel<13>` at 110 GB/s.  `setup_ds41_dense_store()`
(`engine.cpp`, called from `setup_multi_device` before the upload) now creates
a **decode-only** `dnnl_gemm(q, 2, 8192)` per GPU partition and registers:

* **k5** (`add_weight_k5`, exact native Q5_K values, 0.75 B/w) for `wq_b`,
  `wo_a` (one store per group - see the invariant), `wo_b`, the shared
  `ffn_gate/up/down`, `engram_kv` and the untied LM head;
* **int8** (`add_weight`) for Q8_0 (`wq_a`, `wkv`) behind `PF_DS41_DENSE_Q8=1`;
  it measured neutral (124.3 vs 125.0 ms/token) and is off by default.

`gemv1` (`src/engine/arch/deepseek41/forward_gpu.cpp`) takes `wkey(dev, t.data)` - the device pointer
for an uploaded tensor, the host pointer (the store's key) for a converted one
- and, when the store exists, runs `quantize()` + `gemm()` instead of
`gemv_group_launch`.  A converted tensor skips its raw device upload via the
existing oneDNN skip rule, now also gated on `ds41x_.dense_store`.

**Why the stores are decode-only.** `dnnl_gemm`'s prefill scratch is
`cap_M=1024, cap_K=32768` (360 MB/card) and `add_weight_k5` allocates a `K*N`
int8 expansion scratch for the M>1 path - 662 MB for the head alone.  With
`cap_m=2` the row scratch is ~1 MB, and `impl::decode_only` makes `add_weight_k5`
skip the expansion allocation (gemm()'s M>1 branches then decline).  The whole
store set costs +1.2 GB across both cards (+9% over raw Q5_K), and the bank
slots are unchanged.

Measured (2x A770, JIT `build-ds41-jit`, ctx 512, `--moe-vram-mb 18000`,
greedy, interleaved 3 reps, 80-token prompt + 64 **fixed** ids so both arms
compute the same tokens):

| cell | raw (`PF_DS41_DENSE_STORE=0`) | k5 store | delta |
|---|---:|---:|---:|
| decode, ctx 512 (ms/token) | 171.9 | **124.3** | -27.7% |
| prefill, 320-token prompt (ms/token) | 180.3 | **133.2** | -26.1% |
| decode, 3.6k depth (ms/token, 1 rep) | 208.0 | **186.2** | -10.5% |
| k5 + Q8_0 int8, decode ctx 512 | - | 125.0 | Q8 neutral |
| phase split, prefill 320 (`PF_DS41_TIME=2`) | attn 87.1 + MoE 87.8 | attn **49.9** + MoE 85.6 | dense -37 |
| phase split, decode (steady) | attn 84.5 + MoE 103.3 | attn **50.1** + MoE 97.7 | - |

Startup pays for the pack: engine ctor **5.0 -> 16.2 s** (5.9 GB of k5 packing,
single-threaded; the raw upload drops from 5.9 GB to 1.2 GB), and the CLI
reference cell's wall goes 9.0 -> 20.3 s.  `PF_DS41_DENSE_STORE=0` reproduces
the pre-change stream exactly (same md5).

**Exactness.** The k5 *weights* are exact, but `gemm()`'s M=1 branch requires
the per-32-group int8 activation form, so the dot is `quantize(x)` x native
Q5_K, not the raw path's fp32 activation.  Worst per-row relative error against
the exact fp32 dot: wq_b 1.1e-3, wo_a 5.5e-4, wo_b 3.9e-4, ffn_gate 4.7e-4,
ffn_down 8.8e-4, head 4.9e-4, engram 4.5e-4.  The reference cell's 32-token
greedy text is identical (` Paris.` in 3x3 paired runs and 4 earlier), and
`The Eiffel Tower is located in` / `def fibonacci(n):` are identical too, but a
forced 64-id stream diverges at a documented near-tie: the first 9 ids match,
then step 9 has token 455 vs EOS within **0.079** (raw) / **0.308** (store) and
the store chooses EOS.  So the store is text-equivalent on natural prompts, not
byte-identical on a forced stream; use `PF_DS41_DENSE_STORE=0` when exactness
is the requirement.  `PF_DS41_DENSE_ONLY=<wq_b|wo_a|wo_b|ffn|engram|head>` is the
bisection knob.

**Phase 2 (batched dense prefill) is not reachable from this walk.** The
per-token projections are ~24 ms of the 133 ms prefill (5.9 GB at 249 GB/s);
the MoE is 86 (57%) and the MLA core ~25.  Batching the dense projections
requires a layer-major chunked forward: the projection output is consumed
immediately by per-token state (the window ring at `pos`, the compressed-row
append with its block-start RoPE, the indexer pool over the pre-norm keys, the
lagged mHC pre-mix, the Engram history), so a chunked prefill is a second full
forward implementation.  The k5 store also has no M>1 path without the
expansion scratch (above), and a full-size instance does not fit at 18000 MB
(536 MB unallocated on card 0).  The measured ceiling from the dense alone,
even if it were free, is 133 -> ~108-115 ms/token (1.16-1.23x); a realistic
GEMV saving is ~20 ms -> ~113 ms/token.  Re-measured on the current HEAD
(2026-11-14) the numbers move but the shape does not: see **Batched prefill +
oneDNN dense** below for the split, the per-shape GEMM microbench and the
minimal layer-major design.

**XMX / DPAS (2026-11-08 study).**  The A770 is Xe-HPG (DG2) and does have the
unit; oneDNN 3.13.2 selects `jit:xe_hp:gemm:any` and reaches 41-44 T-MAC/s plain
s8 / 22-30 with our grouped scales at M=512 (89 TOPS, 34 % of the 262 TOPS
spec).  Hand-written ESIMD DPAS is not competitive (0.18-0.21 T-MAC/s; an 8-warp
DPAS work-group hangs in the IGC/AOT toolchain), so the actionable route is
oneDNN's systolic GEMM for the **dense prefill projections at M=chunk**.  That
needs the batched prefill first (this walk is per-token M=1) and an int8 weight
store: s8 is +45 % bytes, u4 is -9 % but 4.6 % lossy, and the 15 GiB cards cannot
absorb +3.1 GiB without cutting the expert bank.  Estimated prefill 180 ->
~150-160 (1.13-1.2x) for the dense alone.  The MoE is blocked by its IQ2
codebook format (s8 +3.8x, u4 +1.9x, which would cut the bank slots that
dominate the token); decode at M=1 gains nothing from DPAS (bandwidth-bound,
and oneDNN's 266-280 GB/s M=1 path needs an int8 store).

### Batched prefill + oneDNN dense: re-measured split, microbench and design (2026-11-14)

This section re-measures the "Phase 2" question on the current HEAD (the dense
k5 stores from 2026-11-11 are in) and prices the batched-dense alternative per
shape.  Everything below is measured; the design at the end is not implemented
and is labelled as such.

**Conditions.**  `bench-host` (2x Arc A770 16 GB), Release **JIT** `build/`,
`--device gpu --layer-map 0-19:gpu.0,20-39:gpu.1 --ctx 512 --max-slots 1`,
greedy, one process per cell, model pages in the host page cache (a *second*
process -- the first pays ~60 s of expert-plane page faults and is not a
throughput number).  Prefill cells use `--max-tokens 0` (prefill only),
`PF_DS41_TIME=2 PF_DS41_STATS=1`.  A phase's `attn` includes the dense
projections inside `attention()`; the shared expert's FFN is inside `moe`.
`head` runs once per prompt, so it is ~0 there.

| cell | total ms/tok | attn | MoE | bank hit |
|---|---:|---:|---:|---:|
| 12-token prompt, 18000 MB, **first** forward | 262.1 | 80.2 | 172.7 | 58.0% |
| 12-token prompt, 18000 MB, steady (repeat 2) | ~200 | 26.5 | 165.6 | 58.0% |
| 234-token prompt, 18000 MB | **172.0** | 33.8-35.4 | 130.2-131.7 | 70.3% |
| 234-token prompt, 16000 MB | 183.8 | 35.8 | 140.5 | 67.1% |
| 80-token prompt (decode bench, 18000 MB) | 121.7 | - | - | - |

The old `attn 49.9 + MoE 85.6` cell does not reproduce as a *pair*: `attn` is
now **better** (33-36 vs 49.9 -- the k5 stores are in) and the MoE is **worse**
(130-140 vs 85.6) because it is driven by the bank hit rate, which is a
property of the prompt's routing (70.3% here vs the doc's earlier cell) and of
the budget (140.5 at 16000 vs 130.2 at 18000, i.e. 2 GB of bank is worth
~10 ms/token).  The conclusion that matters is unchanged: the **MoE is 75-83%
of the prefill** and the **dense projections are ~29 ms/token**.  Decode on the
same build and config is **136.1 ms/token** (80-token prompt + 64 forward-only
tokens, differenced; the `--moe-vram-mb 18000` bank), not the 271-276 the
2026-11-07 paragraph records -- that number predates the dense k5 stores and a
warm page cache.  The k5-store table's 124.3 is 9% lower than this 136.1; the
box drifts ~13% between days (see AGENTS.md) and this tree carries unrelated
uncommitted `engine.cpp`/mm work, so read the two as the same cell within that
spread.

**Per-shape oneDNN microbench** (`dev/ds41_dense_mbench`, layer 0's real
tensors, one A770, reps=12, warm; `cur` = what the walk pays today, i.e. M
calls to the M=1 path: the k5 store's native GEMV for Q5_K, the raw GGUF GEMV
for Q8_0.  `i8` = the plain per-32-group int8 store.  `k5` = the k5 store's
M>1 path (per-call `k5_expand_launch` + the offset epilogue).  `nat` = the
M<=13 `nat_gemm` path on the native 5-bit planes, no expansion):

| tensor (K->N) | cur M=1 | nat M=8 | k5 M=32 | i8 M=32 | i8 M=64 |
|---|---:|---:|---:|---:|---:|
| wq_a 5120->1280 (Q8_0) | 0.1168 | - | - | 0.1609 | 0.1514 |
| wq_b 1280->32768 | 0.1250 | 0.1474 | 0.5764 | 0.1772 | 0.1943 |
| wkv 5120->512 (Q8_0) | 0.0235 | - | - | 0.0921 | 0.0969 |
| wo_a 4096->1024, per group | 0.0274 | 0.0659 | 0.3450 | 0.0776 | 0.0817 |
| wo_b 8192->5120 | 0.1231 | 0.1555 | 0.8880 | 0.2881 | 0.2971 |
| ffn_gate 5120->2304 | 0.0382 | 0.0835 | 0.4376 | 0.0953 | 0.1010 |
| ffn_up 5120->2304 | 0.0356 | 0.0779 | 0.4399 | 0.0938 | 0.1007 |
| ffn_down 2304->5120 | 0.0388 | 0.0503 | 0.3195 | 0.0896 | 0.0940 |
| head 5120->129280 | 1.2686 | 1.4917 | 7.0180 | 2.6596 | 2.7080 |
| engram_kv 6144->25600 | 0.3313 | 0.4085 | 1.9528 | 0.7232 | 0.7373 |

**Numerics.**  The batched result is compared row by row against the *same
store's* M=1 GEMV (k5 vs k5, int8 vs int8) on the same activation rows:
`max|d|` is 4e-5..4e-4 with a worst relative error of **5e-6..9e-6** for every
shape, including the head (0.55 GB) and the engram row (0.13 GB).  The
difference is accumulation order only; a cross-store comparison measures the
int8 re-quantization instead (1e-2) and must not be used as the batched path's
error.  The `nat` column is a different kernel (`nat_gemm` on the native
planes): it is the MTP verify's path and is documented bit-exact against the
M=1 GEMVs in `w4_gemv.cpp`; it was not re-verified in this pass.

**What the dense costs per prefill token** -- these columns are **sums of the
measured per-call times above** (40 layers; `wo_a` is 8 calls), not an in-situ
phase split; the in-situ `attn` column is the independent check.  The head is
excluded because it already runs once per prompt.  The Engram
projection is 2 layers, so it adds 0.66 ms/token to the `cur` column and
0.05 ms at M=32; the predicted-prefill table below counts it.

| route | per-layer | per chunk | per token |
|---|---:|---:|---:|
| current, per-token M=1 | 0.720 | - | **28.8** ms |
| nat, M=8, native k5 (no new store) | 1.291 | 51.6 ms | **6.45** ms |
| k5 store, M=32 (expansion per call) | 5.674 | 227 ms | **7.09** ms |
| int8 store, M=32 | 1.618 | 64.7 ms | **2.02** ms |
| int8 store, M=64 | 1.689 | 67.6 ms | **1.06** ms |

Two things the table says that a "batching is free" estimate misses.  The
**k5 store's M>1 path is not usable as-is**: the per-call `k5_expand_launch`
adds ~0.2-0.9 ms, which is 3.5x the int8 store at the same M and makes
wo_a's 8 small calls dominate (0.345 ms each for 3.5 MiB).  And the **per-call
floor matters at M<=8**: 400 dense calls per chunk pay ~0.01-0.05 ms each
regardless of size, which is why `nat` at M=8 (6.45 ms/token) is no better
than the k5 expansion at M=32 (7.09) even though it reads fewer bytes.

**VRAM.**  `PF_DS41_DENSE_Q8=1` is byte-neutral: the int8 store (390 MB over
40 layers) replaces the raw Q8_0 upload (372 MB), and the engine's total reads
5916 -> 6288 MB only because its report counts stores, not uploads.  A full swap of
the 523 k5 stores to int8 is 5916 -> 8381 MB, and the Q8_0 stores replace the
372 MB raw upload with 390 MB, so the device total goes **6288 -> 8771 MB
(+2.4 GB, +39%)**.  Keeping both stores is +8.4 GB, which does not fit.

**Predicted prefill** (dense replaced, everything else unchanged):

| cell | now | dense -> int8 M=32 | dense -> nat M=8 |
|---|---:|---:|---:|
| 80-token prompt | 121.7 | 94.3 (1.29x) | 98.8 (1.23x) |
| 234-token, 18000 MB | 172.0 | 144.6 (1.19x) | 149.1 (1.15x) |
| 234-token, 16000 MB | 183.8 | 156.4 (1.18x) | 160.9 (1.14x) |

The +2.4 GB comes out of the expert bank (2 GB of budget is ~10 ms/token at
this depth), so a realistic net is **1.15-1.18x**, not 1.19-1.29x; the nat
route needs no new store and is the low-risk variant at **1.14-1.23x**.

**What can be batched, and what must stay per token.**  Reading
`forward_gpu.cpp`:

*Batchable* (all elementwise or a pure GEMM over the chunk's rows):
the mHC pre/post mixes and `hc_pre` (`ds41_hc_*` take one row; loop them or
add a rows form), `rms` (`rms_rows` already exists), and every dense
projection: `wq_a`, `wq_b`, `wkv`, the 8 `wo_a` groups (`quantize`'s
`x_stride` = H*D, group offset `g*gsz`, `out_stride` = `out_g*out_rank`),
`wo_b`, the shared `ffn_gate/up/down` and `engram_kv`.

*Must stay per token (or sequential within the layer)*: the RoPE/fp8/window
write (`pos`-dependent but only elementwise -- it can also be batched with a
per-row `pos`), the compressor (`push_kvsrc`/`append_src`: the ring and the
block-start RoPE accumulate across tokens, and the `idx_attn_k` GEMV is per
block), the indexer and its host top-k (`index_select` does a D2H and a
`partial_sort` per token), the MLA attention (`ds41_mla_attn_launch` reads a
per-token window and a per-token selected row set), `route_one` (a 384-float
D2H and a host top-k per token), the routed-expert GEMVs (per-expert IQ2 at
M=1-pair), and `engram_apply`'s host n-gram hash/gather (`ple_table` is 110
GiB and never uploaded).

*Per-token state that a layer-major walk must carry as an array*: `pos`
(passed as an argument instead of the member), the Engram `hist[8]`/`hist_n`,
`cand`/`cand_nb`/`has_cand` (layer 20 writes them and layers 21+ of the *same
token* read them -- a shared member would make token 0 read token M-1's
candidates), and `sel_k`/`sel_dev`.  `S.clen`/`S.ring_n` are per *source*
(one layer) and stay sequential: a layer-major walk appends all M tokens of
the owner layer before a consumer layer runs, and `index_select` already
bounds `clen` by `(pos+1)/ratio`, so a longer cache is harmless.  The prefix
cache commits at block boundaries and needs the same per-token `pos`.

**Minimal correct change** (`forward_chunk(const int * toks, int M, ...)`,
used by the prefill loop; `forward_token` unchanged for decode and the tail):

1. Prefill prologue on the host: map tokens to the Engram n-gram rows
   (`rows[t][ncol]`), the per-token `pos` and `img_row` slice, and the mHC
   `pre_prev` init.
2. Chunk buffers on each device: `h[M][hc][dim]`, `x[M][dim]`, `y[M][dim]`,
   `qr[M][q_lora]`, `qb[M][H*D]`, `ob[M][H*D]`, `og[M][out_g*out_rank]`,
   `kvrow[M][D]`, `pre[M][hc]` -- ~12 MB per device at M=32.
3. Layer-major loop: mHC+norm over M rows -> ONE batched call per dense
   projection (`dnnl_gemm::quantize(x, up, x_stride, ..., M, K)` +
   `gemm(..., M, K, out, out_stride)`) -> per-token rope/window/compressor/
   indexer/MLA -> batched `wo_a`(x8)/`wo_b` -> per-token mHC post -> batched
   ffn -> per-token router/routed experts -> per-token mHC post.  On the k5
   store this also needs the `K*N` int8 expansion scratch (42 MB for the
   widest dense tensor, `wq_b`), which the decode-only instance does not
   allocate.
4. `gemv1` needs a general `(M, x_stride, out_stride)` entry: its oneDNN
   branch currently gates on `out_stride == 0 || out_stride == N`, which the
   8-group `wo_a` call violates.
5. A `dnnl_gemm` instance for the batched path must size `cap_M` from
   `M*N`, not from M: the accumulator cap is `cap_M*32768` int32, so the head
   (`N=129280`) refuses M=32 on the `cap_m=64` instance the decode uses and
   `gemm()` returns false (a silent 0.015 ms "call", which is how this was
   found).

**Why it was not implemented.**  It is a second loop over `n_layer`
(~400-600 lines) and it cannot be exercised by any independent slice: the
input to layer L for tokens 1..M-1 only exists once the layer-major order is
in place, so "batch one layer's FFN" has nothing to run on.  It also cannot be
accepted on stream equality: the batched GEMM reorders the k-sum, the router's
top-6 is a discrete function of the hidden state, and the measured 5e-6 row
error is the same class that already moved the k5 store's forced-id stream at
a near-tie (`Dense oneDNN stores` above).  The oracle has to be logits /
hidden-state tolerance plus the reference prompts' text, as for the k5 store --
not the argmax.  Given the 1.15-1.18x ceiling on a phase that is 17-24% dense,
the honest recommendation is: ship the int8 store swap only if the decode A/B
stays neutral (the in-situ `PF_DS41_DENSE_Q8` A/B measured **136.1 vs 135.1 /
137.3 / 137.0 ms/token**, i.e. neutral, despite a 2.7 ms/token per-call saving
on wq_a -- so a store's isolated GEMV win does not survive the pipeline), and
treat the layer-major walk as a prefill-throughput project, not a quick win.
**No batched-prefill code is in the tree.**


## Invariants and gotchas

* **RoPE rotates the LAST 64 of 512 dims**, not the first.
* **K == V** and the cache is K-only; the output is de-roped.
* **Compressed rows are group-shared**: only 4 owners write; consumers read the
  owner's rows even when their own ratio is 0.
* **The index key comes from the pre-RoPE latent**; the compressed row is
  RoPE'd at the block start, not the token's position.
* **Candidate blocks only at layer 20**, with the newest partial block pinned.
* **ratio-0 layers must not get the compress base or YaRN.**
* **The mHC pre-mix is lagged by one sublayer**; the final norm uses the last
  layer's ffn pre-mix.
* **Router bias affects selection only, not the weights**; `sqrt(softplus)`, not
  softmax; `w/(Σ+1e-20)`, not the usual 6.1e-5 clamp.
* **SwiGLU clamp is one-sided** on the gate.
* **Engram's table is 110 GiB**: never upload it; read rows on demand, skip it
  for image rows.
* **The chat transition is `<|Assistant|>` + `<think>` / `</think>`**; a bare
  assistant marker leaves chat-mode image prompts with an unclosed think block
  and the model answers EOS.
* **A stacked weight's group store must be keyed *and packed* at the group
  offset.**  `wo_a` is one `[8*1024, 4096]` tensor whose GEMV call is made per
  group with a byte offset into it, so `setup_ds41_dense_store` registers eight
  k5 stores at `base + g*out_rank*row_bytes`.  Passing the group key but packing
  from the tensor base gave every group group 0's weights: the model stayed
  fluent-looking but degenerated within a few tokens, and only the per-group A/B
  (`PF_DS41_DENSE_ONLY=wo_a`) localized it.  Each of the six other store classes
  alone was coherent.
* **Anything that skips a raw upload must be resolved through `wkey`, not
  `wptr`.**  A converted tensor has no `weight_maps_` entry, so `wptr` throws
  ("weight tensor not uploaded to this device's partition") while `wkey`
  returns the host pointer - which is the store's key.  All dense `gemv1` call
  sites use `wk()` for exactly that reason; the expert bank pointers stay raw.
* **The generic multi-device oneDNN branch must exclude ds41.**  The
  `else if (multi_dev && !md_int8 && !md_xmx && ...)` block in the engine
  constructor re-creates `dnnl_dev_` for every non-seg_plan model; without the
  `!ds41x_.dense_store` guard it replaced the decode-only instances and then
  `wptr()`-ed the shared-expert tensors whose raw upload the store had skipped.
* **`mm_prompt::n_rows` is the embedding-row count**, not `n_img`: an image
  contributes many rows.

## Tests

Hermetic: `test_model_profile`, `test_hparams`, `test_mm_step`,
`test_chat_builtin`, `test_chat_deepseek`, `test_ds41_pc`.
GPU stage tests/probes: `dev/ds41_iq2_probe`, `dev/ds41_attn_probe`,
`dev/ds41_ops_probe`, `dev/ds41_vision`.
Model-backed: `dev/ds41_gen` (golden ids), `dev/test_ds41_pc` (prefix reuse),
`dev/bench_ds41` (prefill/decode).

## History: superseded measurements

These cells are **not current state**; they are kept for reading a change
against.  The current numbers are in `docs/PERFORMANCE.md` §5.5 and in
"Dense oneDNN stores" above.

* **2026-11-07, pre-dense-store**: decode **353 -> 276 ms/token (3.62 tok/s)**,
  prefill **165-180 ms/token** at 16000 MB.  Same-stream phase split
  (`PF_DS41_TIME=2`, per-phase waits): 275 ms = attention 90 + MoE 173
  (~140 H2D) + head 2.2 + ~10 other; prefill at 16000: 183.7 = attention 68 +
  MoE 107 (~65 H2D) + other 9.  A 320-token prompt: 164.2 ms/token (hit
  88.0 %) and 156.4 (6.39 tok/s, hit 90.7 %) at 18000 MB.  The dense Q5_K
  decode GEMV ran at 110 GB/s and was the largest remaining item; the k5
  stores took decode to 124 ms/token and prefill to 133.
* **2026-11-14 re-measure note**: the old `attn 49.9 + MoE 85.6` pair does not
  reproduce as a pair; `attn` is better (33-36) and the MoE worse (130-140)
  because the bank hit rate is prompt-dependent.  The conclusions that survive
  are current: the MoE is 75-83 % of the prefill and the dense projections are
  ~29 ms/token.
