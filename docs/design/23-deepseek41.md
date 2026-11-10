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
| `src/model/deepseek41.cpp` | loader; binds MLA, compressor, indexer, mHC, MoE, Engram; fills `hparams::ds` |
| `src/model/profiles/deepseek41.cpp` | model profile: `fwd = deepseek41`, `vision = deepseek41v`, policy |
| `src/model/model.h` | `ds41_hparams` (sources resolved to `kv_src_of`/`topk_src_of`), `ds41_layer`, `engram_token_map` |
| `src/engine/engine_ds41.cpp` | validated host forward (mHC, MLA, compressed KV, DSA, Engram, MoE), incremental `DecState`, prefix cache |
| `src/engine/ds41_prefix_cache.h` | chained-hash longest-prefix state cache |
| `src/backend/gpu/kernels/ds41_iq2.{h,cpp}` | raw IQ2_XXS/XS/S expert GEMV (no expansion pass) |
| `src/backend/gpu/kernels/ds41_attn.{h,cpp}` | MLA attention, compressor, RoPE, fp4, indexer score |
| `src/backend/gpu/kernels/ds41_hc.{h,cpp}` | mHC mixes/pre/post |
| `src/backend/gpu/kernels/ds41_ops.{h,cpp}` | fp8 round, swiglu clamp, router, select, candidates, gather, window write, Engram apply |
| `src/mm/vision.cpp` | `deepseek41v` tower + 3×3 merge projector + sentinels |
| `src/server/chat.cpp` | `render_chat_deepseek` structural fallback (minja cannot parse the template) |

## Forward

The forward is **model-owned**: it does not use `seg_plan` or the paged KV pool.
`engine::ds41_fwd()` is the predicate; `generate_impl` hands the request to
`ds41_generate` (CPU) / the GPU path.

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

`engine_ds41.cpp` is a scalar/AVX f32 implementation: it dequantizes the dense
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

`ds41_prefix_cache.h` keys a chained FNV-1a hash per token and stores
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

The GPU engine path is `engine_ds41_gpu.cpp`: dense weights are uploaded per
tensor and split 0-19/20-39 (5.74 GiB), the routed experts live in a ds41-owned
LRU bank backed by the mmap, and `--device gpu` is selected with a layer map;
`PF_DS41_GPU=0` keeps the host forward.  The ds41 GPU path deliberately skips
the oneDNN int8 partition and the native u4/k5 stores (it reads raw GGUF
types), which saves ~1.6 GiB/card.

### Performance pass (2026-11-07)

Measured on 2x A770, ctx 512, `--moe-vram-mb 16000`, greedy, one process per
cell: **decode 353 -> 276 ms/token (3.62 tok/s)**, **prefill 165-180 ms/token**,
the 32-token greedy text **byte-identical** to the unsplit path.

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

Same-stream phase split (`PF_DS41_TIME=2`, per-phase waits): 275 ms =
attention 90 + MoE 173 (~140 H2D) + head 2.2 + ~10 other.  Prefill at 16000:
183.7 = attention 68 + MoE 107 (~65 H2D) + other 9; a 320-token prompt is
164.2 ms/token (hit 88.0 %) and **156.4 (6.39 tok/s, hit 90.7 %) at 18000 MB**.
The prefill MoE uploads are ~42 misses/token, about one per layer, so each
9.2 MB H2D is exposed by the layer dependency; batching tokens does not help
because 384 experts with top-6 and strong adjacent-token locality give a
~29-expert union per layer over 8 tokens (most already resident), i.e. ~1 pair
per expert.  Collapsing a miss's three plane copies into one padded slot is
neutral (measured, reverted).  Text-only 18000 MB is the recommended budget
(16000 keeps room for the vision tower).  The `nohead`
ablation's 60 ms was a workload change (stale logits stop routing from moving
and cut 6.7 GB of misses), not the head (4.1 ms at 110 GB/s).  The dense Q5_K
decode GEMV at 110 GB/s is the largest remaining item: the native u4 store
runs at 223-294 GB/s and is *smaller* (0.625 vs 0.6875 B/w) but lossy (4.6 %
weight error), so it is an accuracy decision; a staging prefetch would recover
~20-30 ms.  Bank hit saturates at ~60 % (R 20-24 vs 384 experts).

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
* **`mm_prompt::n_rows` is the embedding-row count**, not `n_img`: an image
  contributes many rows.

## Tests

Hermetic: `test_model_profile`, `test_hparams`, `test_mm_step`,
`test_chat_builtin`, `test_chat_deepseek`, `test_ds41_pc`.
GPU stage tests/probes: `dev/ds41_iq2_probe`, `dev/ds41_attn_probe`,
`dev/ds41_ops_probe`, `dev/ds41_vision`.
Model-backed: `dev/ds41_gen` (golden ids), `dev/test_ds41_pc` (prefix reuse),
`dev/bench_ds41` (prefill/decode).
