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

## 7. What the curves mean


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

## 8. Reproducing

The harnesses live in `dev/`, which is gitignored, so they are not part of the
repository; the commands below are what produced the tables above.

```bash
source /opt/intel/oneapi/setvars.sh
export LD_LIBRARY_PATH=/opt/intel/oneapi/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:$LD_LIBRARY_PATH
export PF_PREFIX_CACHE=0

M35=/data/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
M27=/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
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

