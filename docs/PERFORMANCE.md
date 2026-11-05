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
| Qwen3.8-Flash-Next | `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf` | `qwen4exp` - HC + PLE + QSA, 512 routed experts (top-10) | two cards; host-resident expert bank (§4.4) |
| MinerU2.5-Pro-2605-1.2B | `MinerU2.5-Pro-2605-1.2B-*.gguf` + mmproj | `qwen2vl` - dense Qwen2-VL decoder | one card (§7.4) |

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
The GGUF bundles the model's own NextN head (`blk.40.nextn.*`).  The
speculative path now runs after the 2026-11-05 recurrent-state fix (an 8-token
`--mtp 4 --max-slots 4` run is coherent), but no MTP throughput cells have been
measured for this model; see [§6.1](#61-mtp--the-models-own-nextn-head).

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

### 4.4 Qwen3.8-Flash-Next

`Qwen3.8-Flash-Next-UD-Q4_K_XL` (4 shards, `qwen4exp`): 48 layers, 512 routed
experts (top-10), a 5.3 GiB dense trunk and a 71.7 GiB host-resident expert bank
behind a per-expert LRU on the cards.  **Current configuration** (AOT Release,
2x Arc A770): `--layer-map 0-19:gpu.0,20-47:gpu.1 --moe-vram-mb 17500
--cpu-threads 28 --kv-type f16 --max-slots 16`, tail-skip 0 (the exact expert
set), `PF_QSA_IDX_DEV=1`, `PF_QSA_PF_DEV=1`, `PF_QSA_SEL_DEV=0`,
`PF_Q4_GEMV8_SPLIT_ROWS`/`WPR` at the profile's 512/8, `PF_HEAD_U4=1` store but
`PF_Q4_HEAD_W4=0` (the raw Q8_0 head read), and `PF_PREFIX_CACHE=0`.  Device 0
is the x8 slot, so the 20/28 layer split gives the x16 card more layers; the
17 500 MB bank is per device (152 slots/layer on device 0, 104 on device 1).
One process per cell; the box drifts ~3 % between runs, so only cells measured
in the same run are compared.

**Startup and footprint.**  Warm restart to first token: **28.0-28.7 s**.  A
cold start with the four shard page caches dropped
(`posix_fadvise(DONTNEED)` after `sync`): **109.4 s**.  Peak process RSS is
**5.8 GiB** and the shards stay in the page cache (**~99 GiB**), because the
pinned expert source is locked physical memory.  The 69.8 GiB pinned source is
filled in **7.1 s** warm by parallel `pread` over 8 workers (**10.49 GB/s**);
the prefault touches 1.9 GiB and skips the other **70.3 GiB** that is already
pinned or device-resident.  The cold floor is the drive (~1.0-1.15 GB/s
sequential), not the engine.  The routed-down int8 host store is built only for
the layers that do not run in the bank (0.9 GiB); the 71.7 GiB routed bank stays
host-resident.

**Column definitions.** `pp` is `prompt tokens / warm prefill`, from the
engine's own `[q4] fwd n=<rows>` prefill rows (a repeat in the same process;
the first prefill of a process also calibrates the expert bank and is 2-3 s
slower).  `ms/step` is the mean wall clock of one decode forward, i.e. the
engine's `[q4] fwd n=1` line, and `tg = 1000 / ms/step`.  The first decode
step after a prefill is 1.5-2x slower while the bank warms, so a short
generation reads high; the cells below are steady-state values.

#### 4.4.1 By context depth, one stream

| depth | prompt | pp | tg | ms/step | conditions |
|---:|---:|---:|---:|---:|---|
| 512 | 468-token prose | **51.7 tok/s** (9.05 s warm) | **11.34 tok/s** | **88.2** | `--ctx 2048`, 95 decode steps, raw head (`PF_Q4_HEAD_W4=0`); pp at `--ctx 4096` |
| 3 660 | 3 660-token prose | **55.9 tok/s** (65.4 s warm) | 9.07 tok/s | 110.2 | `--ctx 4096`, QSA sparse prefill on device (`PF_QSA_PF_DEV=1`) |
| 14 607 | 14 607-token repeated text | – | 8.35 tok/s | 119.7 | `--ctx 32768`, sparse QSA decode |

The 14.6k prompt is **synthetic** (a repeated phrase, because a 14.6k-token
prose prompt exceeds the 128 KB single-argument limit), so its routing is
degenerate and it is not directly comparable to a document.  The 512 cell is
the dense QSA path (the selector is dense-equivalent under 2 048 keys and its
deferred update is hidden under the LM head); the 3.6k and 14.6k cells take the
sparse path, which selects the top 2 048 pooled keys and attends to those on the
device.  The 3.6k decode includes the device indexer projection
(`PF_QSA_IDX_DEV=1`, -3.3 ms/token against the host projection at this depth);
the prefill row is the 3,660-token cell and the 468-token cell is 9.05 s
(51.7 tok/s), so neither is an isolated 512-token chunk.

**Numerics.**  Tail-skip 0 is the exact expert set.  The two QSA device paths
are numerically reordered: the sparse prefill attention can flip a near-tie on
a full prompt (the dense <=2,048-key path is byte-identical), and the indexer is
a device dot (its selected pools and 32-token stream are identical on the
measured cells).  `PF_QSA_PF_DEV=0` / `PF_QSA_IDX_DEV=0` restore the host paths.

#### 4.4.2 Batching

qwen4exp decodes **one row per forward**; the scheduler caps it at one row and
runs its sequences one at a time.  A 2-row forward is correct but **0.85x** the
throughput of decoding the two sequences serially, because two prompts select
~20 distinct experts per layer against ~10 each and the routed-expert weight
read doubles.  The 2-row form is therefore not used, and there is no batched
q4 path at depth.  MTP is gated off for this model (the model profile declares
it unsupported), so there is no speculative path to measure.

#### 4.4.3 Opt-in settings (non-default)

| setting | measured | note |
|---|---|---|
| `PF_Q4_GEMV8_SPLIT_ROWS=4096` (with `PF_Q4_GEMV8_WPR=8`) | depth 512, 96-token greedy: 84.8-85.1 vs 88.6-88.7 ms/step | -3.8 ms (4.2 %); changes the sum order and flips a near-tie (KL 4.9e-5 at step 12) |
| `PF_Q4_HEAD_W4=1` | same 96-token cell: head span 1.885 -> 1.006 ms; decode -0.879 ms/token | reads the native-u4 head store; lossy (KL 0.0165 at step 12); the first decode step pays ~12 ms |
| `PF_QSA_SEL_DEV=1` | 3.6k: +3.1 ms/step; 14.6k: a wash | correct (512/512 pools, identical 32-token stream); its device work contends with the routed experts |
| `PF_Q4_Q5SB=1` | Q5_1 down: 0.1797 -> 0.2073 ms/layer | byte-neutral 128-value panel repack; bit-identical but slower |
| `--q4-tail-skip 7` | 3.6k: 105.8 ms/step (9.45 tok/s) | drops a miss whose best batch rank is >= 7; approximate (can flip a near-tie) |

The default is 0 (the exact expert set); `--q4-tail-skip N` / `PF_Q4_TAIL_SKIP=N`
opts in (1..9 meaningful).  Every row in this table was measured on the same
A/B binary as the default cell it is compared with.

Detailed logs - the QSA stages, the 2-row breakdown, tail-skip, the MTP no-go,
the prefill ceiling, the per-stage budget and the dead ends - are in
[`design/18-qwen4exp.md`](design/18-qwen4exp.md) §4, §6, §7 and §9.

### 4.5 On the Arc 140V (Lunar Lake), one card

`--layer-map 0-39:gpu.0`, i8 KV, `PF_W4_GROUP=32 PF_W4_ALL=1
PF_PREFIX_CACHE=0`, u4 dense + u4 LM head + the resident u4 gate/up bank and the
down int8 bank (the `PF_MOE_BANK_AUTO` governor keeps both only when they fit;
see [AGENTS.md](../AGENTS.md) *Long context on the Arc 140V*).  Prefill is the
`e2e_ttft` of a prose prompt minus the first token; decode is the SSE
chunk-to-chunk rate, 32-vs-288 differencing, temperature 0, 2026-11.

| cell | prefill | rate | decode | vs OVMS |
|---|---:|---:|---:|---:|
| 3.4k prompt | 5.35 s (best of 3, warm) | **641 tok/s** | ~30 tok/s | 1.17x / 0.76x |
| 11.5k | 21.3 s | ~540 tok/s | ~23 tok/s | 1.14x / 0.68x |
| 24k | 51.6 s | ~466 tok/s | ~19 tok/s | 1.17x / 0.70x |

The 11.5k/24k prefill cells predate the `moe.gu_tile=32`/`dn_tile=96` sweep
(-4.6% at 3.4k), so the shipped build is a little faster than the column.  OVMS
(OpenVINO GPU plugin, int4 group-64 everywhere, FP32 MatMuls) is **~550 tok/s
at 3.4k** / 473 @11.5k / 398 @24k: the prefill is **1.14-1.23x** the reference
engine and the decode is **0.68-0.76x**, because our per-token stream is
2.07 GB of mostly 0.625 B/w u4 against its ~0.54 B/w int4 plus a smaller
non-stream share.  Decode's own split: dense 1.20 GB (13 ms), routed bank
0.87 GB (11 ms), GDN 2.0 ms (state floor), router 1.1, attention 0.7, norm 0.6
and ~4-5 ms host/launch/sampling - 70% of the step is the weight stream at
87-92 GB/s of the 107.4 GB/s measured peak.

Context capacity on this part is in
[AGENTS.md](../AGENTS.md) *Long context on the Arc 140V* (the device's 28.12
GiB is not enforced by Level Zero; 448k at i8 / 896k at i4 are the safe
maxima, 262k the trained window).

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

> **Fixed 2026-11-05: the MTP/DFlash2 USM-copy crashes were one recurrent-state
> stride bug, not driver bugs.**  `mtp_verify`/`mtp_rollback` addressed the
> GDN/conv state with `kMaxB` as the slot stride while the buffers are allocated
> with `state_slots_` (`--max-slots`, default 16).  Raising the compile-time
> ceiling from 16 to 32 made every GDN layer after the first read past its
> allocation; the Level Zero USM copy then faulted at enqueue.  The five sites
> now use `state_slots_`.  After the fix, the 27B MTP `test_spec` passes
> (stream-equal and deterministic), DFlash2 `test_spec` passes, and a 35B A3B
> `--mtp 4 --max-slots 4` run emits coherent text.  The 2026-10-09 throughput
> cells below were not re-measured; they are the last measured pass, not a
> statement that the path is broken.

### 6.2 DFlash2 — the block drafter

A second drafter with a different structure: the draft GGUF's own 5-layer head
runs **one non-causal forward** over `[anchor, MASK x (n_max)]` in a private
K/V ring, and a selector scores the top-K candidate sets per block position
plus every ordered pair, so the host walks one coherent path through the
lattice instead of re-drawing per token.  It needs a matching draft GGUF
(`--spec-draft-model`).

The numbers below are the **last measured pass (2026-10-09)** and are kept as the
record of what the drafter costs.  The path crashed on HEAD from the
recurrent-state stride bug above and runs again after the 2026-11-05 fix, but
these cells have not been re-measured.
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

> **Fixed 2026-11-05: the DFlash2 path runs again.**  Its verify reuses
> `engine::mtp_verify`, and the SIGSEGV was the same `kMaxB` vs `state_slots_`
> recurrent-state stride bug (see §6.1).  `test_spec` (DFlash config) now
> passes: stream-equal to the plain greedy decode on both prompts and
> deterministic.  The numbers in this subsection are still the 2026-10-09
> pass; they were not re-measured after the fix.
>
> **The measurement trap remains**: with `--spec-draft-device 1` the drafter is
> *disabled* and the run is a plain decode — it exits 0 and produces text at the
> plain decode's rate.  That is not a working speculative path.

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

### 7.4 The MinerU vision tower and two-step page extraction

The other document workload: MinerU2.5-Pro-2605-1.2B (`qwen2vl`: a Qwen2-VL
dense decoder behind a 32-layer, 1280-wide vision tower with 80-wide heads),
run as the two-step extraction (`sycl-infer-mineru`, the standalone
`tools/mineru/` reference client).  Conditions: **one Arc A770 16 GB**,
AOT Release (`acm-g10`), clocks fixed at **2400 MHz**
(`xpu-smi config -d 0 --standby never`), a quiet window, one 1036x1036 page
(`layout_original_image_01.png`, 21 blocks), greedy, `--max-tokens 1024` per
call (256 truncates the layout decode to 11 blocks), one process per rep,
2026-11-02.  The page md5 was re-baselined after the 2026-11 sampler
presence/frequency penalty-sign fix; `--max-tokens 1024` and the AOT/JIT builds
agree.  Warm page cache unless stated: a
concurrent GPU job time-slices the card and inflates a run up to 50x.

| phase | warm |
|---|---:|
| startup + text model load | **1.65 s** |
| layout vision encode (5476 patches -> 1369 tokens) | **2.49 s** |
| - attention (32 layers) | 2.10 s |
| - GEMM + other | 0.39 s |
| layout text prefill (1394 prompt tokens) | **0.26 s** |
| layout decode (463 tokens) | 3.16 s |
| 21 content blocks (vision + text, summed) | **2.77 s** (vision 1.31, text 1.93) |
| page text decode (676 tokens) | **4.60 s** (6.81 ms/token) |
| **page wall** | **10.99 / 11.23 s** (11.08/11.12/11.28) |
| cold file cache | **11.89 / 12.28 s** |

The output is the reference output at every step: page md5
`c79d2d222d6d90763f0d25c5ffb5269a`, 21 blocks; text-only `gen` md5
`bfc55c543af1d1aecae5da0b1f9b99e5`; vision self-check cos 0.999104.  The
per-stage numbers come from `PF_MINERU_TIME=1` (prep / vision / prefill /
decode / total per image) and `PF_VISION_TIME=1` (a `q.wait()` per vision
stage); both are diagnostics and the default path never waits there.

Against llama.cpp's SYCL backend (`llama-mtmd-cli`, same page/prompt/model,
2026-11-02).  That build has **no `llama-server`**, so the pipeline runs one
process per image, and each process pays ~7.3 s of model+mmproj init and ~6 s
of SYCL JIT (its persistent JIT cache SIGSEGVs on this driver: verified, three
runs, all `SIGSEGV`):

| | this engine | llama.cpp |
|---|---:|---:|
| layout step, one layout-only process (wall) | **9.03 s** (8.91/9.15) | 28.2 s |
| the same, each process's model load removed | **~7.4 s** | 20.9 s |
| whole page, 21 blocks (wall, one process vs 22) | **11.0 s** | 398.2 s |

So the like-for-like single-request ratio is **3.1x** as run (2.8x excluding
the model load each engine pays).  The llama.cpp cells were measured once (its
binary and per-process init/JIT are unchanged by the engine's later passes);
the full-page ratio is dominated by the CLI's per-image init/JIT, not by its
kernels - quote the layout-step figure (2.8-3.1x) whenever a conservative
number is wanted.

**Where the remaining time is, and the ceiling.**  The per-phase floor with the
current exact-order math is about **5.0 s one-shot** (load ~1.0, attention
0.87 traffic / 0.55 compute, decode 0.9-1.5, vision GEMM 0.75, prefill 0.12,
content ~0.5), or ~3.7 s with the process reused.  The measured 11 s is the
**bit-exact** ceiling; the gap to 5 s needs numerical reordering or batching:

* **decode 4.60 s vs 0.9-1.5 s floor**: 676 tokens x 0.52 GB/token at 76-88
  GB/s.  The narrow N=896 projections expose ~896 warps (~112 of 512 EUs),
  which is the shape's ceiling; split-K/reordered reductions change the token
  stream, and oneDNN M=1 cannot enter the recorded decode graph.
* **attention 2.10 s vs 0.87 s traffic floor**: the exact-order ceiling.  f32
  oneDNN QK/PV reaches 1.59 s but is not the reference summation order and is
  run-to-run nondeterministic; a block-level check found identical content on
  the layout page (1/5 runs moved one text box by 0.001) but **different image
  captions and recognized text on the table page** (e.g. `2020年/2021年` vs
  `29/2020`, `铜箔/HM9000` vs `钢带/HMMHPL`, 2-5 of 14 blocks across runs).
  int8 oneDNN QK/PV (1.38 s) changed the page md5 and was nondeterministic.
  Both stay opt-in/off.
* **content vision 1.31 s**: 21 different small M shapes each create their own
  oneDNN primitives; batching the crops or an M-bucket ladder is the next
  bit-exact step, worth ~0.5-1 s.
* **load 1.65 s vs ~1.0 s**: the remaining int8 conversion (~0.85 s total); a
  preconverted cache was rejected (unversioned key surface / wrong-output
  risk).  `PF_I8_SPLITK` (split-K for the narrow int8 decode GEMV) is off: it
  was bit-exact on both tested pages (2/2 each) but is not guaranteed by
  construction.  `i8_grp_gemv_rows_multi` was rejected for this
  page: its 1-ulp difference from the single-call GEMV changes the stream.

New knobs for this path: `PF_VIT_ATTN=70` (ILP default; 0 = the 32/32/512
fallback, 71/72/73/76 = the other ILP shapes, 51 = f32 oneDNN opt-in),
`PF_DNNL_LAZY`, `PF_VISION_COMPACT`, `PF_VISION_DEFER_QKV`, `PF_LOAD_TIME`,
`PF_DNNL_QCHK`, `PF_I8_SPLITK` (off).  The history of how the page reached
this point, including the rejected and opt-in paths, is in
[`design/21-mineru-qwen2vl.md`](design/21-mineru-qwen2vl.md).

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

**qwen4exp is the exception: its decode steps up at the sparse-QSA boundary and
grows with depth.**  The 512-depth cell (QSA off/dense) is 88.2 ms; the sparse
path is 110.2 ms at 3.6k and 119.7 ms at 14.6k.  The selector scores every
pooled key before choosing the top 2 048, so its cost grows with the key count
while the device attention stays fixed at 2 048 rows.  Prefill is 51.7 tok/s at
468 tokens and 55.9 tok/s at 3,660 tokens on the current build.

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
export LD_LIBRARY_PATH=/opt/intel/oneapi/2026.2/lib:/opt/intel/oneapi/compiler/2026.2/lib:$LD_LIBRARY_PATH
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

# qwen4exp: one server process, then one prompt per depth.  pp is the sum of
# the request's [q4] fwd n>1 lines, tg the mean of its n=1 lines.
M4=/path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
PF_Q4_TIME=1 PF_PREFIX_CACHE=0 ./build/sycl-infer \
  --model "$M4" --layer-map 0-19:gpu.0,20-47:gpu.1 --moe-vram-mb 17500 \
  --cpu-threads 28 --ctx 32768 --kv-type f16 --q4-tail-skip 0 --port 18082 serve
python3 /tmp/sse_client.py 18082 "$(cat 16k.txt)" 10
```

**One process per cell**, and never two GPU jobs at once: a prefix-cache hit
inside one process moves the numbers, and two stacked GPU jobs make a card drop
out with `UR_RESULT_ERROR_DEVICE_LOST`, which looks exactly like a regression.
Only compare cells measured in the same run — the box drifts ~13 % over hours.

The build these numbers come from is an AOT Release build
(`-DSYCL_INFER_AOT=ON`, the device image baked for `acm-g10`); see
[`../AGENTS.md`](../AGENTS.md#aot-builds-sycl_infer_aot) for the AOT notes and
the JIT-cache trap on this driver.

