# 18 — Qwen3.8-Flash-Next (`qwen4exp`)

`Qwen3.8-Flash-Next-UD-Q4_K_XL` is the Qwen3.5 hybrid GDN/full-attention
trunk with hyper-connections, a per-layer n-gram embedding (PLE), query-side
sparse attention (QSA) and a 512-expert routed FFN.  It is the largest model
the engine runs: 104 GiB across four shards, of which 71.7 GiB is the routed
expert bank and 26.8 GiB the PLE table.  The dense trunk is only ~5.3 GiB, so
the model is host-RAM bound, not VRAM bound.

## 1. Geometry

| item | value |
|---|---|
| layers | 48 (36 Gated DeltaNet + 12 full attention, every 4th) |
| `n_embd` | 2560 |
| hyper-connections | `hc = 4` streams, low-rank gate 320 |
| routed experts | 512, top-10, FFN width 640, plus a shared expert |
| PLE | one layer, 16 n-gram-hash rows of a 160-wide table (IQ4_NL) |
| QSA | 128-wide indexer key, pools of 4, top 2 048 keys |
| GGUF | 4 shards, UD-Q4_K_XL, `general.architecture = qwen4exp` |

Hyper-connections replace every layer norm.  The residual is `hc` parallel
streams `[n_embd][hc]`; a block mixes them through a grouped RMSNorm, the
320-wide low-rank gate and a per-stream scatter projection
(`hc_*_norm/down/up/inject`; the inject projection is `[hc_dim, hc]`).

## 2. Split GGUF loading

`gguf_file::load` recognises the `-NNNNN-of-MMMMM.gguf` suffix, opens every
sibling, and validates `split.no` against the file index and `split.count`
against the file names.  It then reserves one contiguous
`PROT_NONE | MAP_NORESERVE` range and maps each shard into it with
`MAP_FIXED` at a page-aligned slot.  The consequence is that
`tensor.data - map_base` remains a valid offset across the whole multi-file
model, so `dev_ptr`, page-out and the per-tensor upload path need no special
case.  The gaps between shards are padding, not file bytes.

`is_split()` is the predicate.  The single-blob upload **refuses** a split
file because its `memcpy` would read the `PROT_NONE` gaps; qwen4exp therefore
always takes the per-tensor (multi-device) upload, even when the layer map is
one GPU.  `drop_cache` translates a reservation-relative range into each
shard's own file range before `posix_fadvise`.

## 3. Forward path and placement

`q4_forward` walks the 48 layers itself.  HC/PLE do not fit the generic
`seg_plan` (one GEMV list per layer plus fixed epilogues), but the forward
reuses the existing GDN, attention, GEMV and paged-KV machinery.  The plan
builders, recorded decode graphs, prefix cache, MTP and the multimodal path
are deliberately skipped; `q4exp` forces `pc_enabled = false`.  The server's
q4 `prefill_batch` has no recorded variant and batched q4exp decode is
unmeasured.

The dense trunk is uploaded to the GPUs named by `--layer-map`; every routed
pair and the PLE module run on the host.  A `--layer-map` is synthesised when
none is given (two GPUs split the trunk evenly), but the recommended run is
explicit:

```bash
./build/sycl-infer \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --layer-map 0-23:gpu.0,24-47:gpu.1 --moe-vram-mb 17500 \
  --ctx 2048 --kv-type f16 gen --prompt "..."
```

Prefill measurements used `--ctx 4096 --kv-type i8`; both KV types work.

At startup the routed down bank is converted once to a symmetric int8 store
(per-32 scale): Q5_1's 32-value block is not a K-quant superblock and K=640 is
not a multiple of 256.  The host-resident expert, PLE and indexer pages are
then prefaulted (`PF_Q4_PREFAULT=0` skips it); a cold 4 KB fault per page
costs hundreds of ms/token at this model's expert traffic.

## 4. QSA

The selector and the attention over the selected keys run on the host
(`engine_qsa.cpp`).  Full-attention layers keep a host mirror of their K/V, so
no device gather is needed; the result is written back for the existing `wo`
GEMV.  The cache stores the **raw** indexer key (the reference pools raw keys
before `k_norm`+RoPE), the mirror rounds to f16 like the reference's cache, and
the top-2048 selection is a `std::partial_sort` with ggml's comparator.

Up to 2 048 keys the reference's selection covers every key, so the dense path
is exactly equivalent.  Above it the sparse path runs and appends the
incomplete tail.  Measured against llama.cpp layer 3 at 2 062 tokens: q/mixed
cos 0.9999, indexer score corr 0.9994.  The top-512 sets differ by one
boundary pool (fp accumulation order), so bit-exactness is not claimed.
`PF_Q4_QSADBG` and the `PF_QSA_*` dump knobs are diagnostics.

**The recommended `--ctx 2048` decode does not run QSA**: `qsa_on_` requires
`max_seq > idx_top_k` and both are 2 048, so the 93.3 ms/token cell has no QSA
cost.  Measured at `--ctx 4096` (same weights, bank and TCP prompt), the host
call was **26.8 ms/token** and **23.7 ms of it was the four blocking D2H copies
per layer** (`memcpy().wait()` drains the in-order queue while the host MoE
would otherwise overlap the device), not the dots: indexer BF16 projection
3.0, mirror/pool update 0.08.  The fixes, in order, are in
`engine_qsa.cpp`/`qsa.cpp`:

* enqueue the four copies and wait once: QSA **21.5 ms**, decode 106.4 ->
  **100.8** (both `[q4]` sums; the deferred rows below also give the true
  forward wall, because a deferred update moves work past the `[q4]` print);
* while `pos < top_k` (the selector is dense-equivalent, the common decode),
  defer the whole update: the copies are enqueued per layer and the last
  layer's event is waited **after the LM head has been enqueued**, so the
  projection and the mirror update run while the head is on the device.  QSA
  **6.0 ms of call time**, decode **97.4 on the `[q4]` sum and 101.1 ms of
  true forward wall** (93.8 / 97.5 with QSA off); `PF_QSA_DEFER=0` costs
  **2.0 ms/token** (wall 103.2).  The same path defers a prefill tail
  of up to `kMaxB` tokens: the 96-token cell's 29-token first prefill's QSA
  call goes 479 -> **62-65 ms** (copy 422 -> 9-10); larger chunks stay inline
  because their copies are bandwidth-bound (1.19 s per 1024 tokens);
* batch 8 output rows per job in the BF16 indexer GEMV and skip the q
  projection when no selector runs (80% of the rows): the dense 1024-token
  prefill chunk's projection goes **560 -> 91.5 / 55.4 / 44.9 ms**
  (scalar / AVX2 / AVX-512).

**AVX-512 pays only in the sparse path.**  Above 2 048 keys the selector and
the attention have to stay inline before `wo`, so the copies keep one wait per
layer (**32.2 ms/token** at 2 055) and the attention is the largest remaining
item: **39.3 ms scalar, 19.2 AVX2, 16.0 AVX-512** per token over the 12
full-attention layers (2.45x), and the prefill chunk that crosses the budget
goes 208.0 -> 68.2 ms of attention.  The selector (512 pools) is serial on
purpose: ~2.7 us/layer is below one thread-pool wake.  Against the scalar path
the AVX-512 run has score cosine **1.000000**, the **same 512 selected pools**
(512/512), a **bit-identical** layer-0 `mixed` dump and indexer q max|diff|
**1.2e-6**.  In the 96-token cell AVX-512/AVX2/scalar are indistinguishable end
to end (within a +-0.2 ms run spread) because the host work is hidden under the device tail;
the copy restructure is the win there, not the vector width.  The three forms
live in `src/backend/cpu/kernels/qsa.cpp`; `PF_QSA_ISA` isolates them
(`PF_CPU_ISA` also scalarises the MoE/HC/PLE and is not an A/B of this path).

## 5. PLE and the engram cache

The PLE layer hashes the token and its 16 n-gram context into 16 rows of a
160-wide table, projects them (`ple_key`/`ple_value`), gates them and applies a
depthwise convolution into the residual.  The host module is
`cpu_q4_ple_gate` + `cpu_q4_ple_apply`; `ple_conv1d` is `[kernel][channel]`
with `nb0 = kernel`, so tap `k` of channel `c` is `w[k + kern*c]`.

`--engram-ram-mb` (default 4096) is a direct-mapped host-RAM cache of already
dequantized PLE rows.  The capacity is a power of two, the slot is
`row & (cap - 1)`, and the tag is the row id; a miss dequantizes from the GGUF
mmap, which is the disk tier.  `--engram-ram-mb 0` disables it.

## 6. Expert bank: per-expert LRU and two-pass prefill

The generic three-tier `moe_cache` is **inactive** on qwen4exp:
`engine.cpp` sets `moe_host_only_` and `host_only = true`, so the cache
computes `total = 0`, and its device path lives in `record_forward`'s
`seg_plan` branch while q4exp uses its own layer walk.  The q4 path allocates
its own banks:

* `--moe-vram-mb 17500` gives **121 slots per layer in 47 layers, 16.5 GiB**,
  including the four Q8_0-down layers 4, 30, 46 and 47 (the dedicated
  `moe_down_q8_0_launch`).  Only layer 2 stays on the host (Q5_K gate/up +
  Q8_0 down); its decode-only bank (`PF_Q4_BANK_Q5K=1`, 48 layers, 17.0 GiB)
  measured 107.2 ms against 93.3, so it stays off.  16000 gives 111
  slots/15.1 GiB and 101.2 ms; 17000 118/16.1 and 94.0; 18000 125/17.0 and is
  not faster than 17500.  The 17 500/121-slot point is the knee.
* The resident set is calibrated from the first prefill's routing and then
  maintained by an LRU.  The old eviction stopped at the first non-active slot
  instead of the least-recently-used one; fixing it cut misses **95 -> 58**
  and uploads **280 -> 169 MB/token**.
* The banked layers' expert planes are copied once into pinned host memory
  (`PF_Q4_PIN_SRC`, default on; 63.0 GiB for the 43 Q5_1-down layers, 69.8 GiB
  for all 47) and the H2D reads them directly, so there is no per-prefill
  staging memcpy.  With `PF_Q4_PIN_SRC=0` uploads go through the 32-slot pinned
  staging ring; a pageable `q.memcpy` from the GGUF mmap blocks the host until
  the queue drains and runs at ~2 GB/s (`PF_Q4_STAGE=0` restores it).
* A 512-token prefill selects up to **328 distinct experts per layer** against
  R=111, so the active set is split into batches of `<= R`, each with its own
  slot-map row and uploads.  Every `(token, expert)` pair is computed once.
  The token-row activation is **re-quantized at the start of every batch**,
  because `moe_actq_launch(A.h, ...)` overwrites the `A.xq/asa/xs` that
  gate/up reads.  `PF_Q4_PF_ONEPASS=1` restores the incorrect single pass.
  The batches are double-buffered and the next batch's expert planes are
  prefetched on a per-device in-order copy queue while the current batch's
  GEMMs run (`PF_Q4_PF_PIPE`, default on; `0` serial A/B); the kernels wait on
  that queue's event.  The exact next *layer* cannot be prefetched - its
  router needs the current layer's output - so the prefetch is the next
  expert batch.  Measured 544 tokens: MoE 13.44 -> 12.85 s in the first A/B
  (13.39 -> 13.34 s on a repeat, where the ~4 % drift exceeds the effect); the
  stream is identical.  That was before the pinned source and the Q8_0-down
  bank; together they remove the staging copy and four of the five host layers.
* With the pinned source, the 544-token prefill is 13.44 -> 12.31 s at 43
  layers, then 10.89 s MoE / 11.99 s prefill at 47 layers; the decode is 101.2
  ms/token against 103.7 with the staging and 121 on the 43-layer Q5_1 bank.
  `PF_Q4_PIN_SRC=0` is the A/B.

Three kernels carry the bank's arithmetic:

* `moe_down_q5_1_launch` is the dedicated Q5_1 dp4a down (32-value groups,
  `NPB=8`).  The `m·xs` term is per group, carried on `lane%8 == 0` only; the
  first version counted it on every lane and was 8x wrong.
* `moe_down_q8_0_launch` is the Q8_0 counterpart: 34-byte blocks, signed
  weights biased by 128 inside dp4a (`q ^ 0x80`) and the `-128*sum(xq)` group
  correction on one lane; layer 4's FFCMP cosine is 0.999999.  It is what lets
  the four Q4_K/Q8_0-down layers bank.
* The generic down must accept K not a multiple of 256.  `moe_down_impl` used
  `n_ff_exp/256` as the row stride (2 for K=640) and read the wrong weights;
  the fix is `ceil(K/256)`, the real `quant_row_bytes` stride, masking past K,
  and a 256-byte guard on the bank.

### 6.1 CPU misses with GPU-first overlap (`PF_Q4_CPU_MISS`) — measured, rejected

Strata's CUDA path computes a non-resident expert on the host (`kind = -1` in
`expert_source.cpp`) instead of copying it.  The q4 port is behind
`PF_Q4_CPU_MISS` (default off): on a decode miss the pair is *not* uploaded;
the host computes it from the pinned source (the GGUF pages of a GPU layer are
released after the bank upload, so the mmap must not be read), stages the rows
in a pinned buffer and enqueues them into `A.acc` before `moe_reduce`.  The
resident grouped kernels are submitted first, so the host arithmetic runs under
them.  The host kernel matches the device's arithmetic: the activation goes
through the same per-32 amax/127 quantization (f16-rounded scale, the rule
`moe_actq_launch` writes), gate/up use the CPU's AVX-512-VNNI Q4_K row dot, and
down reads the native Q5_1/Q8_0 blocks (row-parallel; a row is 20 groups, so
the parallelism is over row chunks, not rows).  `=2` selects the existing f32
dequant path as the A/B.

The arithmetic is right and the performance is not.  One-layer probe (layer 0,
after a 16-token warm-up), vs the all-GPU bank: KL mean **1.2e-4**, max
2.0e-3, top-1 **40/40** (the first 16 steps, before the split starts, are
bit-identical).  That layer alone is a wash: frozen it misses 5.25/token
against 3.03 adaptive, but the host costs 2.52 ms and the layer wall is 2.69 ms
against the baseline's 3.1 ms - a per-miss win, erased at model scale by the
3.2x miss count.  The full cell is KL mean **1.9e-4**, max 6.4e-3, top-1
**96/96**, and `PF_Q4_FFCMP` reports layer 0 cos **0.999988**.  The last-32
wall is **148.1 ms against 99.3** (28 CPU threads; 227.1 at the default 14),
and 122.2 against 93.2 with `PF_Q4_TAIL_SKIP=7`.  The blocker is the resident
set, not the dot: the bank's hit rate comes from continuous LRU admission.
Frozen at decode step 16 (`PF_Q4_CPU_MISS_WARM`, default 16), it misses **156
of 460 pairs/token against 48.8** for the adaptive LRU, and a host miss costs
**0.58 ms in situ** at 28 threads (0.25-0.3 isolated) against the **0.52 ms
H2D it replaces**; the resident chain is only ~0.45 ms/layer, so there is
little to hide it under.  A 64-step warm-up narrows the loss to 115.3 vs
99.3 (MoE 94.7 vs 80.9 ms) but does not close it.  It cannot pay without admission, and admission puts
the upload bytes back.  `PF_Q4_CPU_MISS_LAYER=<il>` and
`PF_Q4_CPU_MISS_TIME` are the probe; the flag stays off.  The full table is in
[`../PERFORMANCE.md`](../PERFORMANCE.md) §4.4.

## 7. Measured (2x Arc A770, AOT Release)

| configuration | 512-token prefill | decode |
|---|---:|---:|
| host experts, bank off | 48.7 s (10.5 tok/s) | ~500 ms/token (~2 tok/s) |
| whole-layer bank, 18000 MB | 45.5 s (11.25 tok/s) | ~470 ms/token (~2 tok/s) |
| full host mirror, GPU reads over PCIe (Strata-style) | 14.1 s (544 tokens) | 178 ms/token |
| per-expert LRU, all-GPU, 16000 MB, pinned source | ~12.0 s (45 tok/s) | ~101 ms/token (9.9 tok/s) |
| **per-expert LRU, all-GPU, 17500 MB, pinned source** | **~11.4 s cold / 9.6 s warm (47-56 tok/s)** | **~93.3 ms/token (10.7 tok/s)** |
| per-expert LRU, all-GPU, 18000 MB, pinned source | – | ~93.3 ms/token (10.7 tok/s) |

The pinned expert source is 69.8 GiB of host RAM copied once at startup.  A
cold first prefill is **11.4 s** (the bank calibrates on it); the second/steady
prefill is **~9.6 s (56 tok/s)**.  Decode reaches **~93.3 ms/token
(10.7 tok/s)** after ~16 tokens of LRU warm-up; a 64-token generation is
~15.6 s wall including the warm prefill.  18 000 loads but is not faster than
17 500.  Correctness:
`12 x 9` -> 108, the TCP/UDP prompt is coherent, the 27B
`test_decode_vs_prefill` oracle is 8/8 OK, and `PF_Q4_FFCMP` gives cosine
0.99998 against the host routed FFN (layer 4's Q8_0 down 0.999999, layer 2's
Q5_K/Q8_0 0.99995).

## 8. Flags

| flag / env | default | meaning |
|---|---|---|
| `--moe-vram-mb MB` | 0 | per-expert LRU budget; 0 = all experts on the host |
| `--engram-ram-mb MB` | 4096 | PLE row cache in host RAM; 0 disables |
| `--moe-ram-mb MB` | 0 | generic ncmoe host tier; not used by qwen4exp |
| `PF_Q4_BANK=0` / `PF_Q4_BANK_LAYER=1` | on / off | all-host path / whole-layer bank (A/B) |
| `PF_Q4_BANK_Q5K=1` | off | layer 2 bank, decode-only (48 layers, 107.2 ms vs 93.3, off) |
| `PF_Q4_PIN_SRC=0` | on | stage from the mmap instead of the pinned expert source |
| `PF_Q4_TAIL_SKIP=N` | 0 | drop a miss whose best batch rank is >= N (7: 86.9 ms, 32.2 MiB vs 93.6/56.2) |
| `PF_Q4_CPU_MISS=1` | off | decode CPU miss for non-resident experts, int8 host path (`2` = f32 A/B); **148.1 vs 99.3 ms, rejected** (§6.1) |
| `PF_Q4_CPU_MISS_WARM=N` / `_LAYER=il` / `_TIME` | 16 / all / off | decode steps with the LRU before the split freezes / one-layer probe / per-layer host+GPU print |
| `PF_Q4_CMDBG=1` | off | per-call sub-phase ms inside the host miss expert |
| `PF_Q4_HEAT=1` | off | decayed-heat victim (105.7 ms / 156 MiB; slower, A/B only) |
| `PF_Q4_PF_ONEPASS=1` / `PF_Q4_PF_STATS=1` | off | old single pass / two-pass stats |
| `PF_Q4_PF_PIPE=0` | on | serial copies (A/B for the two-pass prefetch) |
| `PF_Q4_LRU_OLD=1` | off | pre-fix eviction (A/B only, not an LRU) |
| `PF_Q4_STAGE=0` | on | pageable expert uploads instead of the pinned ring |
| `PF_Q4_INJECT=0` / `PF_Q4_GEMV8=0` / `PF_Q4_GEMVSEG=0` | on | generic HC inject / Q8_0 GEMV / blocking segment copy |
| `PF_Q4_F32_SPLIT=0` | on | generic GDN beta/alpha GEMV |
| `PF_QSA_TIME=1` | off | QSA host split per forward (`copy/gemv/upd/sel/attn`) |
| `PF_QSA_ISA=scalar\|avx2\|avx512` | auto | force only the QSA host kernels' ISA |
| `PF_QSA_DEFER=0` | on | inline update instead of the end-of-forward deferred update |
| `PF_Q4_HIST=1`, `PF_Q4_TIME`, `PF_Q4_PHASE`, `PF_Q4_HCTIME`, `PF_Q4_HOT_TIME` | off | routing histogram / phase attribution |
| `PF_Q4_PREFAULT=0` / `PF_Q4_NOPLE=1` / `PF_Q4_FFCMP=<layer>` | on / off / off | skip prefault / skip PLE / routed-FFN host oracle |

The full knob list (including the `PF_QSA_*` dumps) is in
[`../../AGENTS.md`](../../AGENTS.md#environment-variables).

## 9. Dead ends

* **Strata-style full host mirror.**  `PF_Q4_MIRROR=1 --moe-vram-mb 0` keeps
  all 71.7 GiB in pinned host memory and lets the GPU read it over PCIe
  (4.28 GB/s).  On the 544-token prompt that is **14.1 s prefill / 178 ms
  decode**, against the bank's 11.4/9.6 s and 93.3 ms.  The stream re-reads
  the whole working set every prefill/token; the bank's H2D + VRAM is the
  right tier.  The useful part is the pinned mirror as the H2D source
  (`PF_Q4_PIN_SRC`).
* **Host-spill split.**  A static top-R set with the non-resident pairs on the
  host never converged: 5 498 of 5 957 slots changed on the next calibration.
  Routing is a discrete function of the hidden state, so a one-ulp difference
  flips the top-10 boundary; the all-GPU upload-on-miss path avoids the split.
* **Whole-layer bank.**  45.5 s, because 12 of 48 layers cover only ~25% of
  the traffic.  It is the A/B baseline (`PF_Q4_BANK_LAYER=1`).
* **Bigger banks and other transports.**  18000/20000 MB fail to load;
  16500-17500 give no decode win.  Parallel staging delays the first DMA (the
  floor is 170.5 MiB at 5.9 GB/s), a pinned mirror cannot beat the same H2D,
  LFU is neutral at 72 tokens and worse at 200, and a preloaded slot map is
  churned by the prefill.  Native Q5_1 down on the host is slower than the
  generic path.
* **Layer 2 Q5_K banking** (`PF_Q4_BANK_Q5K=1`) is a 6% prefill loss and
  decode-neutral; the Q8_0-down layers stay on the host.
