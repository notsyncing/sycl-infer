# PERFORMANCE.md — measured throughput

This is the project's throughput table: prefill and decode by context depth and
by batch size, per model and per speculative path, on one fixed piece of
hardware.

Every number here carries the conditions it was taken under, because a
throughput number without its conditions is not a number: a different `PF_*`
switch selects a different kernel, and the same binary measures ~13 % apart on
different days.  The tables are therefore written with the harness, the model,
the layer split and the date in the caption, and the commands that reproduce
them are in [§8](#8-reproducing).

**Adding or changing a measurement?**  Update the table it belongs to **and**
every sentence that quotes it (the README, a design doc, a code comment) in the
same change - the rule is in
[`../AGENTS.md`](../AGENTS.md#measured-results-and-dead-ends).

## 1. The measurement conditions

| | |
|---|---|
| hardware | 2 × Intel Arc A770 16 GB (DG2), i9-10940X, 247 GB host RAM |
| driver | 12.55.8, as shipped on the box |
| build | AOT Release (`-DSYCL_INFER_AOT=ON`, the device image baked for `acm-g10`), `-O2` device code |
| device profile | `arc_a770` on both cards (`PF_DEVICE_INFO=1` prints the resolved profile and its provenance) |
| KV cache | `i8` (the default); `--kv-type` not set |
| prefix cache | **off** (`PF_PREFIX_CACHE=0`) - a prefix-cache hit inside one process moves the numbers |
| weights | each model's own UD-Q4_K_M: Q4_K on the native u4 grid, Q5_K on the native 5-bit grid, IQ4_XS/IQ4_NL on the codebook grid |
| batch | the engine's own continuous batching, one sequence per row |

### The harnesses

The measurement tools live in `dev/`, which is **not part of the repository**
(it is gitignored), so they are described here rather than linked:

| harness | what it measures | command |
|---|---|---|
| `dev/bench_pf` | the **isolated prefill** of N tokens - one forward, no decode | `./dev/bench_pf <gguf> <ntok> <reps> <layer-map>` |
| `dev/bench_tg` | the **decode** at a context depth: prefill to `depth`, then `ntok` greedy steps | `./dev/bench_tg <gguf> <depth> <ntok> <reps> <layer-map>` |
| `dev/bench_conc` | **N concurrent rows** at a depth: N prefills, then N rows of batched decode | `./dev/bench_conc <gguf> <layer-map> <N> <prompt> <gen> <reps>` |
| `dev/bench_mtp` | the **MTP** path: the plain decode and the speculative decode from one engine, one load | `./dev/bench_mtp <gguf> <layer-map> <tokens> <k,k,...> <nprompt>` |

Every cell is **one process per configuration**.  A prefix-cache hit inside one
process moves the numbers (the speculative acceptance most of all), and
stacking two GPU jobs on the box makes a card drop out
(`UR_RESULT_ERROR_DEVICE_LOST`), which looks exactly like a code regression.

### How to read the columns

* **depth** is the KV depth at the first decode step (the prompt plus the
  generated tokens).  The harness is called with `prompt = depth - gen`, so the
  depth is exact rather than "the prompt length".
* **pp** is the prefill rate in tokens/s: the whole prompt over the time to the
  first token.
* **tg** is the decode rate in tokens/s, **aggregate over the N rows** - N rows
  each producing one token per step is N tokens per step, so the aggregate is
  what a server would deliver to N clients.
* **ms/step** is the wall time of one batched decode step (all N rows).
* **"best" is the statistic, never the mean.**  The box drifts ~13 % over
  hours, so only cells measured in the same run may be compared with each
  other, and a single representative is not a measurement.
* **The harness's own cost is inside the timed region**: it fetches each row's
  logits and argmaxes them on the host (~0.45 ms per row).  The engine's own
  marginal row is therefore ~3.4 ms where the table shows ~3.9 ms.  The
  sampler is host-side in the server too, so this is a real product cost and
  not only a harness artefact.
* The prefill rate at a depth is *not* the rate of an isolated chunk: a prompt
  of D tokens at depth D pays the attention over its own KV, so the longer the
  prompt the lower the average.  The isolated 512-token chunk (`bench_pf`) is
  the marginal cost of one more chunk and is the number to compare against
  another build's.

### What the server actually delivers

The harnesses measure the engine; a user measures the server.  The server's
cold time-to-first-**content** token — the HTTP round trip, the chat template,
the tokenization, the lazily grown KV pool, the prefill, the LM head and the
sampler — measured 2026-10-20 with a streaming chat completion:

| prompt | 27B | 35B A3B |
|---:|---:|---:|
| 512 tokens | 1.459 s = **351 tok/s** | 0.972 s = **527 tok/s** |
| 4 096 tokens | 5.668 s = 723 tok/s | 3.849 s = 1 064 tok/s |
| 16 384 tokens | 20.243 s = 809 tok/s | 13.550 s = 1 209 tok/s |

**The first SSE *chunk* arrives 0.085-0.156 s after the request — before the
prefill has run at all.**  A benchmark that divides the prompt length by the
time-to-first-chunk therefore reports ~3 400 tok/s for a 512-token prompt that
actually takes 1.46 s, and the same trap produces the 73 830 tok/s and 235 563
tok/s figures that the depth>0 cells of the OpenVINO comparison are built on.
The in-process harness (1.010 s for the same 512-token prompt) is the number
to compare builds with; the server's is the number a user feels.

## 2. The models

| model | file | architecture | placement |
|---|---|---|---|
| Qwen3.5-0.8B | `Qwen3.5-0.8B-Q4_K_M.gguf` | `qwen35` - hybrid Gated DeltaNet + full attention | one card |
| Qwen3.8-27B | `Qwen3.8-27B-UD-Q4_K_M.gguf` | `qwen35` - dense | two cards: `--layer-map 0-31:gpu.0,32-63:gpu.1` |
| Qwen3.6-35B-A3B | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `qwen35moe` - routed MoE, 256 experts, 8 active | two cards: `--layer-map 0-19:gpu.0,20-39:gpu.1` |

The 27B and the 35B do not fit on one 16 GB card, so the two-card split is not
an optimisation for them - it is the only way to run them at all.

## 3. Qwen3.5-0.8B — the reference model

One card (the model fits an A770 with room to spare), `qwen35`: 24 layers,
16 Gated DeltaNet + 8 full attention, head_dim 128.  This is the model the
test suite runs against; the numbers are here as a scale reference.

| measurement | harness | result |
|---|---|---|
| the isolated 512-token prefill | `bench_pf`, 5 reps | **6 336 tok/s** (0.081 s; the mean over 5 reps is 6 260), argmax 4821 |
| the decode at depth 512 | `bench_tg`, 128 tokens, 3 reps | **78.7 tok/s** (12.7 ms/token) |

## 4. Qwen3.6-35B-A3B — the routed-MoE model

`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, `qwen35moe`: 40 layers, 256 experts with 8
active per token, 21.1 GiB of weights, `--layer-map 0-19:gpu.0,20-39:gpu.1`.
The GGUF bundles the model's own NextN head (`blk.40.nextn.*`), so the engine
*builds* an MTP plan for it — but the speculative path itself does not run on
this model in the current build; see [§6.1](#61-mtp--the-models-own-nextn-head).

### 4.1 By context depth, one stream

`dev/bench_conc <gguf> 0-19:gpu.0,20-39:gpu.1 1 <depth-128> 128 <reps>`, one
process per cell, 2026-10-20.

| depth | prompt | prefill | prefill rate | decode | ms/step |
|---:|---:|---:|---:|---:|---:|
| 512 | 384 | 0.515 s | 745 tok/s | 26.38 tok/s | 37.91 |
| 4 096 | 3 968 | 3.127 s | 1 269 tok/s | 25.91 tok/s | 38.59 |
| 16 384 | 16 256 | 11.008 s | **1 477 tok/s** | 24.54 tok/s | 40.76 |
| 65 536 | 65 408 | 46.764 s | 1 399 tok/s | 20.20 tok/s | 49.50 |
| 131 072 | 130 944 | 105.466 s | 1 242 tok/s | 15.54 tok/s | 64.34 |

The **isolated 512-token chunk** from an empty KV (`bench_pf`, 5 reps) is
**779 tok/s** (0.657 s, the mean 768) with argmax 11171 — the same measurement
the engine's regression oracle uses.

* **The decode is linear in depth and has no knee.**  37.9 ms/step at depth
  512 and 64.3 ms at 131 072: about **0.20 µs per token of context on top of a
  ~37.8 ms floor**, and that floor is the weight pass — the whole step with an
  empty KV.  Only the attention grows with the context.
* **The prefill's rate peaks at ~16k of prompt and falls off after it.**  The
  rate *rises* with the prompt length first (745 tok/s for 384 tokens →
  1 477 for 16 256) because the fixed per-forward cost is amortised over more
  tokens, and then falls (1 399 at 65 536, 1 242 at 131 072) as the prompt
  attends to its own KV.  **A "prefill tok/s" without the prompt length is
  not a number.**

### 4.2 By batch size, at depth 512

`dev/bench_conc <gguf> 0-19:gpu.0,20-39:gpu.1 <N> 384 128 3`, one process per
cell, 2026-10-20.

| N | prefill | decode (aggregate) | ms/step | marginal row |
|---:|---:|---:|---:|---:|
| 1 | 745.1 tok/s | 26.38 tok/s | 37.91 | – |
| 2 | 993.1 tok/s | 35.70 tok/s | 56.03 | 18.1 ms |
| 4 | 1 184.2 tok/s | 60.91 tok/s | 65.67 | 4.8 ms |
| 8 | 1 309.9 tok/s | 101.12 tok/s | 79.11 | 3.4 ms |
| 16 | 1 382.9 tok/s | 122.05 tok/s | 131.09 | 6.5 ms |
| 32 | 1 417.4 tok/s | **168.33 tok/s** | 190.11 | 3.7 ms |

* **The aggregate grows with the batch and the marginal row does not get
  cheaper.**  The second row costs 18.1 ms (it pays for the batched GEMMs'
  first departure from the single-row shape), rows 3-4 cost 4.8 ms each,
  rows 5-8 cost 3.4 ms each, rows 9-16 cost 6.5 ms each and rows 17-32 cost
  3.7 ms each.  The 9-to-16 segment is the expensive one, and it is the
  routed experts' decode path: the weight re-read factor is `ceil(pairs/M)`
  per expert, so the rows that land on the same experts pay for their weights
  again.
* **The prefill saturates at 16 rows and the decode does not.**  1 382.9 →
  1 417.4 tok/s (+2.5 %) for twice the concurrency, against +36 % for the
  decode.  The prefill's mode-2 batch is capped at `kMaxB * kMaxT` tokens, so
  beyond 16 rows the prefill's batch stops growing.

### 4.3 Batched at depth 16 384

| N | prefill | decode (aggregate) | ms/step |
|---:|---:|---:|---:|
| 8 | 1 429.8 tok/s (8 × 16 256 tok, 90.955 s) | 76.42 tok/s | 104.68 |

Against the same batch at depth 512 (101.12 tok/s, 79.11 ms/step), the batched
decode is **24 % slower** at 16k of context: eight rows' worth of attention
over a 16k KV is what the extra 25.6 ms/step buys.

### 4.4 Qwen3.8-Flash-Next — the 104 GiB hybrid with a per-expert LRU

`Qwen3.8-Flash-Next-UD-Q4_K_XL` (4 shards, `qwen4exp`), AOT Release, 2×Arc A770,
2026-10-26.  The routed bank (71.7 GiB) and the PLE table (26.8 GiB) are
host-resident; the dense trunk plus a 16.5 GiB per-expert LRU slice of the
experts are on the cards.  Prefill numbers used
`--ctx 4096 --kv-type i8`; the recommended decode uses `--ctx 2048 --kv-type f16`
(both types work).

| placement | 512-token prefill | decode, one stream | ms/token |
|---|---:|---:|---:|
| host experts, bank off | 48.7 s (10.5 tok/s) | ~2 tok/s | ~500 |
| whole-layer bank, 18 000 MB (12/48) | 45.5 s (11.25 tok/s) | ~2 tok/s | ~470 |
| full host mirror, GPU reads over PCIe (Strata-style) | 14.1 s (544 tokens) | 5.6 tok/s steady | ~178 |
| per-expert LRU, all-GPU, 16 000 MB, pinned source | ~12.0 s (45 tok/s) | 9.9 tok/s steady | ~101 |
| **per-expert LRU, all-GPU, 17 500 MB, pinned source** | **~11.4 s cold / 9.6 s warm (47-56 tok/s)** | **10.7 tok/s steady** | **~93.3** |
| per-expert LRU, all-GPU, 18 000 MB, pinned source | – | 10.7 tok/s steady | ~93.3 |

The per-expert LRU is the default.  The recommended map is **all-GPU**
(`0-23:gpu.0,24-47:gpu.1`) with `--moe-vram-mb 17500`: 121 slots per layer in
47 layers (16.5 GiB), including the four Q8_0-down layers 4, 30, 46 and 47.
The expert planes are pinned in host RAM (`PF_Q4_PIN_SRC`, 69.8 GiB, one-time
copy) so the uploads read them directly.  Only layer 2 stays on the host
(Q5_K gate/up + Q8_0 down); a decode-only bank for it
(`PF_Q4_BANK_Q5K=1`, 48 layers, 17.0 GiB) measured **107.2 ms** against 93.3,
so it stays off.  A cold first prefill is **11.4 s** (the bank calibrates on
it); the second prefill is **9.6 s**, the steady number.  Decode reaches
**~93.3 ms/token (10.7 tok/s)** after ~16 tokens of LRU warm-up; a 64-token
generation is ~15.6 s wall including the warm prefill.  17 000 gives 118 slots
and 94.0 ms; 18 000 loads (125 slots, 17.0 GiB) but is not faster than 17 500;
the 17 500/121-slot point is the knee.

What produced the row, in the order the costs appeared:

* **Two-pass prefill.**  A 512-token chunk selects up to 328 distinct experts in
  a layer against R=121, so the old single pass evicted an active expert and the
  grouped kernels silently skipped its pairs.  The active set is now split into
  batches of `<= R`, each with its own slot map and uploads; every
  `(token, expert)` pair is computed exactly once.  The token-row activation is
  re-quantized at the start of every batch because `moe_actq_launch(A.h, ...)`
  overwrites the `A.xq/asa/xs` that gate/up reads.  Prefill is byte-identical
  across chunk sizes 64..1024.
* **Tail-skip** (`PF_Q4_TAIL_SKIP=7`, Strata's CUDA-path idea, opt-in).  On
  the 96-token TCP cell at 17500: **93.6 -> 86.9 ms** (-7.2%), last-token
  upload **56.2 -> 32.2 MiB** (-43%); rank 8 is 90.0 ms.  The output changes
  and still needs a KL/top-1 gate, so the default is off.
* **CPU misses with GPU-first overlap** (`PF_Q4_CPU_MISS`, Strata's CUDA
  `kind = -1` idea) was implemented and **rejected** (2026-10-27, the same
  96-token TCP cell, `--ctx 2048 --kv-type f16`, one process per cell; the
  last-32 wall is `wall(96) - wall(64)` so the load, the prefill and the first
  64 steps cancel; the baseline's 99.3 here is the archive's 93.6 at today's
  ~6% box drift):

  | cell | wall 96 | wall 64 | last-32 ms/token |
  |---|---:|---:|---:|
  | bank, default CPU threads (14) | 13.673 s | 10.497 s | **99.3** |
  | bank + `PF_Q4_TAIL_SKIP=7` | 11.996 s | 9.013 s | **93.2** |
  | CPU miss, `PF_CPU_THREADS=28`, warm 16 | 16.165 s | 11.426 s | 148.1 |
  | CPU miss, `PF_CPU_THREADS=28`, warm 64 | 13.923 s | 10.235 s | 115.3 |
  | CPU miss, default CPU threads | 22.827 s | 15.560 s | 227.1 |
  | CPU miss + tail-7, 28 threads | 13.766 s | 9.856 s | 122.2 |

  The host path is numerically sound: one-layer probe KL mean 1.2e-4 / max
  2.0e-3 / top-1 40/40, full cell KL 1.9e-4 / max 6.4e-3 / top-1 96/96,
  `PF_Q4_FFCMP` cos 0.999988.  It loses because the frozen resident set misses
  156/460 pairs/token against 48.8 for the adaptive LRU, and a host miss
  (0.58 ms at 28 threads) is not cheaper than the 0.52 ms H2D it replaces; see
  [`design/18-qwen4exp.md`](design/18-qwen4exp.md) §6.1.
* **Decayed heat** (`PF_Q4_HEAT=1`) is a loss: 105.7 ms and 156 MiB of upload
  against 93.6 ms / 56 MiB; leave it off.
* **Copy/compute overlap** (`PF_Q4_PF_PIPE`, default on).  The workspace is
  double-buffered and the next batch's expert planes are prefetched on a
  per-device in-order copy queue while the current batch's GEMMs run; the
  kernels wait on the copy queue's event.  Measured on the 544-token prompt:
  MoE 13.44 -> 12.85 s in the first A/B (13.39 -> 13.34 s on a repeat, where
  the box's ~4 % drift exceeds the effect; the staging copy+wait itself is
  0.6 s lower); the stream is identical, and a 576-token prefill still answers
  **108**.  The
  exact next layer is not prefetchable (its router needs the current layer's
  output); the prefetch is the next expert batch.  That measurement was before
  the pinned source and the Q8_0-down bank; together they remove the staging
  copy and four of the five host layers.  The bank-budget sweep that followed
  found the decode knee at 121 slots / 17 500 MB (101.2 -> 93.3 ms); 18 000
  loads but is not faster.
* **The staging memcpy is gone and the four Q8_0-down layers are on the card**
  (`PF_Q4_PIN_SRC`, default on).  Each banked layer's expert planes are copied
  once into pinned host memory (63.0 GiB for the 43 Q5_1-down layers; 69.8 GiB
  for all 47) and the H2D reads them directly: the 4.6 s of per-prefill staging
  memcpy becomes 0.55 ms of map copies.  `moe_down_q8_0_launch` is the Q8_0
  counterpart of the Q5_1 kernel (34-byte blocks; signed weights biased by 128
  inside dp4a, the `-128*sum(xq)` correction on one lane per group), which is
  what lets layers 4, 30, 46 and 47 bank.  Measured on the 544-token prompt:
  13.44 -> 12.31 s with the pinned source at 43 layers, then 10.89 s MoE /
  11.99 s prefill with the four extra layers; decode 101.2 ms/token against
  103.7 with the staging and 121 on the 43-layer Q5_1 bank.  `PF_Q4_PIN_SRC=0`
  restores the staging path.
* **The LRU eviction was not an LRU.**  Both loops stopped at the first
  non-active slot, so slots 0..~10 absorbed every miss.  Fixing it cut misses
  **95 -> 58** and uploads **280 -> 169 MB/token**.
* **Expert uploads go through a 32-slot pinned staging ring per device**, with
  asynchronous copies; the pageable `q.memcpy` of the GGUF mmap blocked the host
  until the queue drained and ran at ~2 GB/s.
* **Q5_1 down has a dedicated dp4a kernel** (`moe_down_q5_1_launch`, 32-value
  groups, `NPB=8`); the `m·xs` term is per group (carried on `lane%8 == 0`), not
  per lane.  GPU-vs-host cosine 0.99998.
* **Q8_0 down has its own dp4a kernel** (`moe_down_q8_0_launch`): 34-byte
  blocks, signed weights biased by 128 (`q ^ 0x80`) and the `-128*sum(xq)`
  correction on one lane per group.  Layer 4's FFCMP cosine is 0.999999.  The generic down also needed its
  K%256 row stride fixed: `moe_down_impl` used `n_ff_exp/256` (2 for K=640), so
  every row after the first read the wrong weights.  With `ceil(K/256)`, the
  real `quant_row_bytes` stride and a 256-byte guard, GPU-vs-host cosine is
  0.99995.
* **Q8_0 GEMV.**  `gemv_dec_q8_kernel` (four int8 per lane via two uint16 loads)
  and `gemv_dec_q8_split_kernel<WPR=8>` for rows <= 512 took HC down
  0.21-0.32 -> **0.042-0.046 ms**, ssm_out/wo 0.27 -> **0.099**, and the LM head
  7.89 -> **2.6**.  Do not use an 8-byte uint2 load: the AOT `acm-g10` lowering
  widens it and the prefill logits decorrelate.
* **The HC inject F32** is a 256-thread workgroup per `(token, stream)` (strided
  float4, four accumulators, warp + SLM reduce): **1.93 ms/token** over the 96
  calls, replacing the serial one-lane-per-row dot.
* **The router is one fused softmax+top-k kernel**, bit-identical
  (`PF_MOE_RT_CHK`) and **0.43 -> 0.07 ms/layer**.  GDN beta/alpha is F32 and
  uses a `WPR=8` split-K GEMV: **0.104 -> 0.031 ms/call**.
* **The GEMV segment descriptor is a 4096-slot ring** with an asynchronous H2D,
  not a per-call blocking copy: **1.3%** steady, md5 identical.

Correctness on this build: `12 x 9` -> **108**, the TCP/UDP prompt is coherent,
the 27B `test_decode_vs_prefill` oracle is **8/8 OK**, and `PF_Q4_FFCMP` gives
cosine **0.99998** against the host routed FFN.  QSA is implemented (host
selector + host K/V mirror); measured against llama.cpp layer 3 at 2 062
tokens, q/mixed cos 0.9999 and indexer score corr 0.9994.  The top-512 sets
differ by one boundary pool (fp accumulation order), so bit-exactness is not
claimed.  See [`design/18-qwen4exp.md`](design/18-qwen4exp.md) and §4.4.1 for
the host QSA cost and the AVX-512 result.

#### 4.4.1 The host QSA path

`PF_Q4_TIME=1 PF_QSA_TIME=1`, `--moe-vram-mb 17500 --ctx 4096 --kv-type f16`,
one process per cell, both AOT Release on 2xArc A770.  The short cell is the
96-token TCP prompt; the sparse cell is a 2 055-token TCP prompt (two
1024-token prefill chunks plus a 7-token tail) followed by 8 greedy decode
tokens, so the selector and the attention are live.

**`--ctx 2048` does not run QSA at all** (`qsa_on_` needs `max_seq >
idx_top_k`, both 2 048); the QSA-off row is the same prompt on another process
and is the baseline the +3.6 ms of true wall is measured against.

| short cell (96-token decode) | `[q4]` sum | true wall | QSA call |
|---|---:|---:|---:|
| QSA off (`--ctx 2048`) | 93.8 | 97.5 | - |
| QSA on, four waits per layer (first measurement) | 106.4 | - | 26.8 |
| QSA on, four copies -> one wait per layer | 100.8 | - | 21.5 |
| QSA on, deferred update, scalar | 97.3 | - | 6.1 |
| QSA on, deferred update, AVX2 | 97.3 | - | 6.1 |
| **QSA on, deferred update, AVX-512** | **97.4** | **101.1** | **6.0** |
| QSA on, `PF_QSA_DEFER=0`, AVX-512 | 99.5 | 103.2 | 20.2 |

The deferred call is `copy=5.25` (the single end-of-forward event wait),
`gemv=0.77/0.75/0.71` (bf16 indexer projections; the q projection - 80% of the
rows - is skipped while no selector runs) and `upd=0.05` for
scalar/AVX2/AVX-512; the host work is hidden under the LM head, so the vector
width is invisible end to end.  The win is the copy restructure: the four
blocking `memcpy().wait()` calls per layer drained the in-order queue 12 times
a token while the host MoE otherwise overlaps the device.  The `[q4]` sum stops
before the LM head and the deferred flush, so it is not a token time for the
deferred rows; **the true wall is 101.1 vs 103.2 ms/token**, i.e. the deferral
is worth 2.0 ms and QSA costs +3.6 ms over the QSA-off 97.5.

Sparse path (2 055 tokens of context):

| per token, 12 full-attention layers | scalar | AVX2 | AVX-512 |
|---|---:|---:|---:|
| decode, ms/token | 203.6 | 183.8 | **180.0** |
| sparse attention, ms | 39.3 | 19.2 | **16.0** |
| selector (512 pools), ms | 1.73 | 1.24 | 1.30 |
| D2H copy wait, ms | 32.2 | 32.1 | 32.2 |
| crossing prefill chunk, attention ms | 208.0 | 79.3 | **68.2** |
| 1024-token dense prefill, indexer GEMV ms | 91.5 | 55.4 | **44.9** |

The prefill's QSA copies are the remaining item there: 1024 tokens x
(2560 + 2x512) floats x 12 layers is 176 MB per chunk, a 1.19-1.24 s D2H at
~10.8 GB/s, and the projection+update after it are 0.1/0.05 s.  The 1024-token
dense chunk's whole QSA cost is ~1.32-1.37 s of the ~11.4 s prefill (the
crossing 7-token chunk is 236-383 ms, attention included).  A dense prefill
*tail* of up to 32 tokens is deferred like a decode: the 96-token cell's
29-token first prefill's QSA call measured **479 ms before / 62-65 ms after**
(422 -> 9-10 ms of copy wait); the one-time mirror allocation (49 ms) is what is
left of the update.

The sparse rows are `[q4]` sums (QSA is inline there, so the LM head is the
only excluded part; the true wall is ~3.7 ms higher).  The AVX-512 result is
exact in the parts that matter: against the scalar path
the selector scores have cosine **1.000000**, the **same 512 pools** are
selected (512/512), the layer-0 `mixed` dump is **bit-identical** and the
indexer q max|diff| is **1.2e-6**.  The selector's dots are
`cpu_qsa_score_pools` (serial: ~2.7 us/layer); the attention is the fused
per-head q.k + vector-exp softmax + vector V accumulation in
`src/backend/cpu/kernels/qsa.cpp`.  The gated result and the `wo` GEMV are
unchanged.  The 96-token cell's stream is unchanged by construction: it is
dense-equivalent, so `qsa_attention` returns false and no QSA output reaches
the layer.

```bash
./build/sycl-infer \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --layer-map 0-23:gpu.0,24-47:gpu.1 --moe-vram-mb 17500 \
  --ctx 4096 --kv-type f16 gen --prompt "<TCP prompt>" --temp 0 --max-tokens 96
# A/B the QSA kernels only; PF_CPU_ISA=scalar also scalarises the host MoE
PF_QSA_ISA=scalar|avx2|avx512 PF_QSA_TIME=1
```

**Dead ends** (each measured).  A **host-spill split** never converged: 5 498 of
5 957 slots changed on the next calibration, because routing is a discrete
function of the hidden state and a one-ulp difference flips the top-10 boundary.
The **whole-layer bank** is the 45.5 s baseline, not the answer: 12 of 48 layers
cover only ~25 % of the traffic.  An earlier **39-layer map** was measured
through the single-pass overflow bug and is superseded, as are all GPU-expert
numbers taken before the K%256 down fix.  **18 000/20 000 MB** fail to load and
16 500-17 500 give no decode win.  **Parallel staging** delays the first DMA
(the floor is 170.5 MiB at 5.9 GB/s), a **pinned mirror** cannot beat the same
H2D, **LFU** is neutral at 72 tokens and worse at 200, and a **preloaded slot
map** is churned by the prefill.  **Native Q5_1 down on the host** is slower
than the generic path.  **Layer 2's Q5_K gate/up banking**
(`PF_Q4_BANK_Q5K=1`) costs 6 % prefill and is decode-neutral.  The generic
three-tier `moe_cache` is inactive on qwen4exp (`host_only`, resident=0); the
q4 per-expert LRU is a separate path.  The server's q4 `prefill_batch` has no
recorded variant, and **batched q4exp decode is unmeasured**.

```bash
./build/sycl-infer \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --layer-map 0-23:gpu.0,24-47:gpu.1 --moe-vram-mb 17500 \
  --ctx 2048 --kv-type f16 gen --prompt "The capital of France is" --max-tokens 64 --temp 0
```

## 5. Qwen3.8-27B — the dense model

`Qwen3.8-27B-UD-Q4_K_M.gguf`, `qwen35`: 64 layers, 16 full-attention (24 query
heads over 4 kv heads, head_dim 256) and the rest Gated DeltaNet, 15.4 GiB of
weights, `--layer-map 0-31:gpu.0,32-63:gpu.1`.

### 5.1 By context depth, one stream

`dev/bench_conc <gguf> 0-31:gpu.0,32-63:gpu.1 1 <depth-128> 128 <reps>`, one
process per cell, 2026-10-20.

| depth | prompt | prefill | prefill rate | decode | ms/step |
|---:|---:|---:|---:|---:|---:|
| 512 | 384 | 0.806 s | 476 tok/s | 13.41 tok/s | 74.55 |
| 16 384 | 16 256 | 16.522 s | **984 tok/s** | 12.26 tok/s | 81.55 |
| 65 536 | 65 408 | 74.154 s | 882 tok/s | 9.70 tok/s | 103.10 |

The **isolated 512-token prefill** from an empty KV (`bench_pf`, 5 reps) is
**507 tok/s** (1.010 s, the mean 504), argmax 4821.

* **The decode is linear in depth, with a slope 2.2x the 35B's.**
  `ms/step = 74.3 + 0.439 us * depth` predicts 81.52 at 16 384 against the
  measured 81.55, and 103.10 at 65 536 against the measured 103.10.  The floor
  is the weight pass (74.3 ms — the 27B reads 15.4 GiB per token) and the
  slope is the attention: the 27B's 16 attention layers have head_dim 256 over
  4 kv heads, so the decode's kv-head read is re-read 6 times per layer.
* **The prefill's shape is the same as the 35B's**: the average rate of a whole
  prompt rises with the prompt length (476 tok/s for 384 tokens → 984 for
  16 256) and then falls (882 at 65 408), peaking at ~16k.

### 5.2 By batch size, at depth 512

`dev/bench_conc <gguf> 0-31:gpu.0,32-63:gpu.1 <N> 384 128 3`, one process per
cell, 2026-10-20.

| N | prefill | decode (aggregate) | ms/step | per row |
|---:|---:|---:|---:|---:|
| 1 | 476.3 tok/s | 13.41 tok/s | 74.55 | 74.6 ms |
| 2 | 628.5 tok/s | 18.93 tok/s | 105.66 | 52.8 ms |
| 4 | 748.7 tok/s | 34.36 tok/s | 116.42 | 29.1 ms |
| 8 | 826.8 tok/s | **62.48 tok/s** | 128.05 | 16.0 ms |
| 16 | 872.7 tok/s | **45.20 tok/s** | 353.98 | 22.1 ms |
| 32 | 898.3 tok/s | 76.45 tok/s | 418.56 | 13.1 ms |

* **The aggregate is not monotone: it drops 28 % from 8 rows to 16.**  The step
  cost jumps from 128 ms to 354 ms for twice the rows.  The cause is the
  native GEMM's **M ≤ 13 boundary** (`nat_gemm_launch`): above 13 rows the
  dense layers leave the native u4 path for oneDNN's int8 matmul, which is a
  different cost shape and cannot be recorded into a SYCL command graph.
  This is the same boundary the 35B's measurement found on the *prefill* side,
  and on a dense model it is worth far more than on the MoE.
* The per-row cost falls all the way down to 13.1 ms at 32 rows — the weight
  pass is amortised over 32 rows and the attention over the KV is shared.

### 5.3 Batched at depth 16 384

| N | prefill | decode (aggregate) | ms/step |
|---:|---:|---:|---:|
| 8 | 823.7 tok/s (8 × 16 256 tok, 157.889 s) | 41.07 tok/s | 194.80 |

Against the same batch at depth 512 (62.48 tok/s, 128.05 ms/step), the batched
decode is **34 % slower** at 16k of context — and it is 52 % slower than the
*single-stream* decode at 16k (12.26 tok/s, 81.55 ms/step), i.e. at 16k of
context the eight rows have stopped paying for each other: the attention over
the 16k KV dominates and does not amortise over the batch the way the weight
pass does.

## 6. Speculative decoding

Both speculative paths are **single-request** paths: they take the whole engine
for the generation, and they do not compose with batching — the server keeps
them off by default (`PF_MTP_SERVER=1` restores the MTP one for a strictly
single-request workload).

### 6.1 MTP — the model's own NextN head

`dev/bench_mtp <gguf> <layer-map> 128 4 3`: one engine, one load, the plain
decode and the speculative decode over three prompts (technical / story / code),
128 tokens each, 2026-10-20.  The plain decode is prompt-independent within
noise (69.70 ms/token here; 67.2 / 67.3 / 70.7 on the same three prompts in the
2026-10-09 pass), so it is quoted once.

| prompt | plain | MTP k=4 | speedup | acceptance |
|---|---:|---:|---:|---:|
| technical explanation | 69.70 ms/token | **39.25 ms/token** | 1.78x | 1.84 |
| story opener | 69.70 ms/token | **52.63 ms/token** | 1.32x | 0.84 |
| code continuation | 69.70 ms/token | **36.46 ms/token** | 1.91x | 2.08 |

(the acceptance column is the engine's own `acc` (`PF_MTP_TIME=1`): 2.08 for the
code prompt is this pass's number, and the other two are the 2026-10-09 values,
which the fresh code number reproduces to the decimal)

**Acceptance, not the engine, is what moves the speedup.**  The cycle is
nearly constant across the three prompts — `draft` 13.9-16.2 ms, `verify`
82.9-84.1 ms, `commit` + `rollback` 1.2 ms, ~106 ms every time — while the
speedup spans 1.32x-1.91x.  Nothing about the engine differs between those
cells; only how many of the four drafted tokens the NextN head got right.  The
ceiling on the speedup is `1 + acceptance`, so a drafter that accepts 2.08
drafts per cycle cannot beat 3.08x no matter how cheap the cycle gets.

> **Known issue, measured 2026-10-20: MTP does not work on the 35B A3B.**  Its
> GGUF bundles the NextN head (`blk.40.nextn.*`), the engine's gate passes and
> the verify's command graphs are built ("[mtp] verify command graphs: 3
> phase(s), 5 rows"), and the *plain* decode is fine (38.7 ms/token, matching
> the table above) — but the first speculative cycle fails: with the default
> 16 slots the driver reports `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, and with
> `PF_MAX_SLOTS=1` it **segfaults in the same place as the DFlash2 crash
> below** (a USM `memcpy` inside the Level Zero driver, reached from
> `engine::mtp_verify`).  So MTP is a 27B feature in this build, and the
> routed-MoE model's MTP numbers do not exist yet.

### 6.2 DFlash2 — the block drafter

A second drafter with a different structure: the draft GGUF's own 5-layer head
runs **one non-causal forward** over `[anchor, MASK x (n_max)]` in a private
K/V ring, and a selector scores the top-K candidate sets per block position
plus every ordered pair, so the host walks one coherent path through the
lattice instead of re-drawing per token.  It needs a matching draft GGUF
(`--spec-draft-model`).

The numbers below are from the 2026-10-09 pass (the DFlash2 CLI path is
currently the one that segfaults, see the note at the end of this section);
`n_max` is the draft tokens per cycle and the ceiling on the speedup is
`1 + acceptance`:

| `n_max` | 1 | 2 | 3 | 4 | **5** | 6 |
|---|---:|---:|---:|---:|---:|---:|
| acceptance | 0.92 | 1.80 | 2.84 | 3.46 | **4.25** | 5.05 |
| ms/token | 46.1 | 33.8 | 26.1 | 24.0 | **21.8** | 20.1 |
| verify ms | 74.3 | 78.4 | 82.3 | 87.1 | **92.4** | 97.4 |

`n_max=5` is the shipped default, and the *prompt* is the headline:

| prompt at `n_max=5` | acceptance | ms/token | vs its own plain baseline |
|---|---:|---:|---:|
| code continuation | 4.25 | 21.8 | **3.08x** |
| technical explanation | 2.19 | 35.8 | 1.88x |
| story opener | 0.49 | 77.0 | **0.92x — slower than no drafter** |

At an acceptance of 0.49 each cycle emits 1.49 tokens for a 114 ms cycle, which
is strictly worse than the 70 ms plain step, and **no `n_max` rescues it** —
the whole sweep above is worse in ms/token than 21.8 on the story prompt.
Acceptance has to be checked per workload before enabling this, and a single
speedup figure for the drafter is not a property of the drafter.

> **Known issue, measured 2026-10-20: the DFlash2 path does not run at all in
> the current build.**  The drafter requires the draft on device 0 ("[dflash]
> needs a multi-device oneDNN GPU partition with draft on device 0 - disabled"),
> and with the draft on device 0 it dies with SIGSEGV *inside the Level Zero
> driver*, in a USM `memcpy` reached from `engine::mtp_verify` — the DFlash2's
> verify reuses the MTP's verify machinery.  It reproduces with a 32-token
> generation and a 45-token prompt, and it is **not** a VRAM shortage on the
> default configuration (`PF_MAX_SLOTS` 1 / 4 / 8 / 16 all crash, and the 27B's
> card 0 holds 9 072 MiB of weights plus the 1 050 MiB draft).
>
> **The measurement trap**: with `--spec-draft-device 1` the drafter is
> *disabled* and the run is a plain decode — it still exits 0 and still
> produces text, at the plain decode's rate.  That is how a "put the draft on
> the other device and it works" reading gets formed; it does not work, it
> stops drafting.  The numbers in this subsection are therefore the
> 2026-10-09 pass, the last one where the path ran.

## 7. The ONNX document pipeline

A different workload from everything above: the document pipeline's ONNX
models (a layout detector, a text detector and recogniser, two table
structurers, a formula recogniser), run by `sycl-infer-onnx` on one device.
The numbers are the service's own operator timing (`SI_ONNX_TIME=1`), and the
conditions are: **one Arc A770 16 GB**, a JIT Release build
(`-DSI_ONNX=ON`), one 800x800 page, a warm service (2026-10-23).

### 7.1 The layout detector, by operator

| what | time |
|---|---:|
| the whole model | **129-141 ms** |
| the `Conv`s (118 of them, oneDNN) | 93 ms |
| the `TopK` (300 of 13 125, three calls) | **4 ms** |
| the `MatMul`s (123, plus the bias/activation epilogue) | 9 ms |
| the `Concat`s (140 of them) | 5.5 ms |
| the `LayerNormalization`s | 3.3 ms |
| the `Transpose`s | 3 ms |
| the `Add`s | 1.9 ms |
| the mask-to-box head (one `MaskBox` scan) | 0.2 ms |
| the `GatherND` (12 million elements) | 0.7 ms |
| the first call, with the runtime's JIT and weight upload | ~1.2-1.4 s |

The service's call is **129-141 ms at batch 1** (146 ms per page at batch 2).
The pipeline's config drops the 48 MB mask output (`drop_outputs`), so it
never crosses the wire; the executor now holds 443 MiB at batch 1 instead of
911 MiB, because the mask branch is one scan and the reshapes borrow their
input buffers.  The full table and the dead ends are in
[`design/17-onnx-service.md`](design/17-onnx-service.md).

### 7.2 The formula recogniser

The only model here whose loop runs the body (1 742 operators) per iteration:
**145 ms per iteration**, against ONNXRuntime's 12.4 on the same input.  Its
output is integer-equal to ONNXRuntime's on five real formula crops.

### 7.3 What the ONNX path taught, in one line each

* A selection that is quadratic in `k` looks like a hang: the layout model's
  `TopK` was 60 seconds a page before the sorted-insertion rewrite (10.5 ms);
  a 64-bit rank-key radix select now does the three layout calls in 4 ms.
* A device wait per operator costs the pipeline its depth: the guard band and
  the per-operator wait were 2.7x the formula model's device time.
* A value that no operator of a graph reads still has to stay alive when a
  subgraph reads it (the outer scope), and the copy has to set the tensor's
  own device pointer.
* A `CumSum` is not a `Reduce`; an `If` branch may declare a constant as its
  output; and an operator inside a branch must not be skipped as "constant".

## 8. What the curves mean


**A "prefill tok/s" without the prompt length is not a number.**  Two
quantities share the name: the *isolated chunk* (what `bench_pf` measures —
the marginal cost of one more 512-token chunk) and the *average rate of a whole
prompt* (what a user waits for).  The chunk rate rises with the depth (the
attention over the existing KV is cheap next to the weight stream) while the
prompt's average rate falls (the prompt attends to its own KV).  The average
rate peaks at ~16k of prompt on both models and falls off after it.

**The decode's cost is linear in the depth, with no knee.**
`ms/step = a + b * depth`, where `a` is the weight pass (the whole step with an
empty KV) and `b` is the attention over the KV:

| | floor `a` | slope `b` |
|---|---:|---:|
| 27B | 74.3 ms | 0.439 µs/token |
| 35B A3B | 37.8 ms | 0.202 µs/token |

The 27B's slope is 2.2x the 35B's because its 16 attention layers have
head_dim 256 over 4 kv heads (each kv head is read six times per layer), while
the 35B's 20 attention layers have head_dim 128 over 8 kv heads.

**Batching amortises the weight pass, and the marginal row gets dearer.**  The
aggregate throughput grows with the batch; the per-row cost falls at first (the
weight pass is shared) and then rises again (the batched GEMMs and the routed
experts' weight re-reads).  On the 35B A3B the marginal row costs 18.1 ms for
the second row, 4.8 ms for rows 3-4, 3.4 ms for rows 5-8, 6.5 ms for rows
9-16 and 3.7 ms for rows 17-32.

**The prefill saturates with the batch and the decode does not.**  The prefill's
mode-2 batch is capped at `kMaxB * kMaxT` tokens, so beyond 16 rows it stops
growing (+2.5 % for twice the concurrency on the 35B) while the decode's
aggregate grows 36 % over the same doubling.

**The dense model's aggregate is not monotone in the batch size.**  The 27B
loses 28 % of its aggregate throughput going from 8 rows to 16, because above
13 rows the dense layers leave the native u4 GEMV (`nat_gemm_launch` refuses
M > 13) for oneDNN's int8 matmul.  The same boundary on the 35B A3B costs far
less, because only its dense layers are affected — and it is the *prefill* side
of the boundary that hurt there, not the decode.

**The routed-MoE model decodes faster than the dense one** although it reads
more weights: 8 of its 256 experts are active per token, so a decode step
touches a fraction of its 21 GiB, while the 27B's decode reads all of its
15.4 GiB every token.  The MoE pays for that on the *prefill*, where every
expert of every pair is touched.

**The speculative paths trade a cycle for acceptance.**  The cycle cost is
nearly constant across prompts; what changes is how many drafted tokens the
head gets right, and the ceiling on the speedup is `1 + acceptance`.  A single
speedup figure for a drafter is not a property of the drafter — it is a
property of the prompt.

## 9. Reproducing

The harnesses live in `dev/`, which is gitignored, so they are not part of the
repository; the commands below are what produced the tables above.

```bash
source /opt/intel/oneapi/setvars.sh
export LD_LIBRARY_PATH=/opt/intel/oneapi/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:$LD_LIBRARY_PATH
export PF_PREFIX_CACHE=0

M35=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
M27=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf
LM35=0-19:gpu.0,20-39:gpu.1
LM27=0-31:gpu.0,32-63:gpu.1
GEN=128

# the isolated prefill of N tokens (an empty KV, one forward)
./dev/bench_pf "$M35" 512 5 "$LM35"

# one stream at a context depth: the prompt is depth-GEN, so the depth is exact
./dev/bench_conc "$M35" "$LM35" 1 $((16384-GEN)) $GEN 2

# N concurrent rows at a depth
./dev/bench_conc "$M35" "$LM35" 8 $((512-GEN)) $GEN 3

# the MTP path: the plain decode and the speculative decode from one load
PF_MTP_TIME=1 ./dev/bench_mtp "$M35" "$LM35" 128 4 3
```

**One process per cell**, and never two GPU jobs at once: a prefix-cache hit
inside one process moves the numbers, and two stacked GPU jobs make a card drop
out with `UR_RESULT_ERROR_DEVICE_LOST`, which looks exactly like a regression.
Only compare cells measured in the same run — the box drifts ~13 % over hours.

The build these numbers come from is an AOT Release build
(`-DSYCL_INFER_AOT=ON`, the device image baked for `acm-g10`); see
[`../AGENTS.md`](../AGENTS.md#aot-builds-sycl_infer_aot) for the AOT notes and
the JIT-cache trap on this driver.

