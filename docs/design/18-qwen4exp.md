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
| QSA | 128-wide indexer key, `compress_ratios = 4` pools (the code reads `hp.idx_kpool`, not a literal 4), top 2 048 keys |
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
builders, recorded decode graphs and the prefix cache are deliberately skipped;
`q4exp` forces `pc_enabled = false` while the policy still reads `m.q4exp`
(the model profile below is M0+M1; M2 delegates the walk to a runtime object).
MTP is declared unsupported and gated with a printed reason; DFlash has no q4
draft/verify path in this build; multimodal is routed through the model's own
chunker (§3.1).  The server's
q4 `prefill_batch` has no recorded variant, and batched q4exp decode is now
**measured and rejected**: two rows select ~20 distinct experts per layer
against ~10 each, so the routed-expert weight read doubles and the pair is
0.85x the throughput of decoding the two sequences one at a time.  The
scheduler caps qwen4exp at one row and serialises its sequences; see §7.1.
The scheduler's generic `prefill_batch` is not q4; it now routes qwen4exp prefill
through `prefill_text` for the whole remaining prompt (one scheduler chunk;
`q4_prefill_text` chunks internally at `kMaxB*kMaxT`).  A 2412-token server
request prefills in **41.7 s** (57.8 tok/s, one chunk) against **92.6 s** for
the old 32-token `prefill_chunk` workaround (**2.2x**), output md5 and sampled
ids matching the CLI; `PF_PFB_MAX_M=16` is no longer needed.  The 32-token vs
1024-token chunking shifts the QSA selector pools slightly (score cosine
0.9999945, max 0.0699, a few boundary pools), the known prefill accumulation
order; generated content matched.

The dense trunk is uploaded to the GPUs named by `--layer-map`; the PLE module
and the non-resident expert source stay in host RAM, and a routed pair with a
bank slot is computed on the card that owns it (a miss is uploaded from the
pinned source).  A `--layer-map` is synthesised when
none is given (two GPUs split the trunk evenly), but the recommended run is
explicit:

```bash
./build/sycl-infer \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --layer-map 0-23:gpu.0,24-47:gpu.1 --moe-vram-mb 17500 --cpu-threads 28 \
  --ctx 2048 --kv-type f16 gen --prompt "..."
```

Prefill measurements used `--ctx 4096 --kv-type i8`; both KV types work.

At startup the routed down bank is converted once to a symmetric int8 store
(per-32 scale): Q5_1's 32-value block is not a K-quant superblock and K=640 is
not a multiple of 256.  The host-resident expert, PLE and indexer pages are
then prefaulted (`PF_Q4_PREFAULT=0` skips it); a cold 4 KB fault per page
costs hundreds of ms/token at this model's expert traffic.

### 3.1 Model profile (M0+M1)

The architecture's policy is declarative: `src/model/model_profile.h` plus
`src/model/profiles/qwen4exp.cpp` (the registry rows in `model.cpp` point at the
const, and `model::load` requires the profile's key to be the GGUF's exact
`general.architecture`).  For qwen4exp:

| field | value |
|---|---|
| `fwd` | `hc_ple_qsa` (the model-owned walk; `generic_plan` is the shared `seg_plan`) |
| `residual` | hyper-connections, 4 streams, low rank 320 |
| `extras` | PLE on; QSA on, top_k 2048, pool 4, indexer 4 heads x 128 |
| `policy` | `max_decode_rows = 1`, prefill `whole_prompt`, prefix cache **no**, auto layer map **yes**, md_splits **yes**, engram default 4096 MiB |
| `rep` | `sin_w8 = false`, `dnnl_int8 = false`, `raw_gemv_decode = true`, `split_per_tensor_upload = true` |
| `spec` | `mtp`/`dflash` declared false with one `why` (no NextN block in the trunk, and the walk does not produce the per-token GDN/conv snapshots the verify needs); multimodal is a *capability* of the model-owned chunker (the loader's `mm_model_prefill`), while the `spec.multimodal` field is the declarative report |
| `moe` | `runtime_bank` (not the generic three-tier cache) |

The engine stores `mp_ = m.prof` and every policy branch reads it:
`moe != engine_cache` turns the generic cache off (qwen4exp's bank owns every
slot), `policy.prefix_cache` is a capability (`PF_PREFIX_CACHE=1` cannot enable
it - the checkpoint cannot represent PLE/QSA state), `policy.auto_layer_map`
synthesises a map when none is given, `policy.md_splits` keeps the multi-device
K-split on, `policy.engram_ram_mb_default` is the unset `--engram-ram-mb`
default, `rep.sin_w8` gates the `md_int8` path, `spec.mtp` prints the declared
reason and disables MTP before the `has_mtp` check, and the scheduler reads
`policy.max_decode_rows`/`prefill` (one row, whole prompt).  The generic
qwen35/qwen35moe profiles are `generic_plan`, engine-cache, chunk32 prefill,
prefix cache on, so their behaviour is unchanged.  `PF_MODEL_INFO=1` prints the
profile, policy, weights, capabilities and provenance; `test_model_profile`
pins the values.  The forward dispatch still reads `m.q4exp` (asserted equal to
`fwd == hc_ple_qsa`), which is the M0+M1 line: the policy is declarative, the
walk is not yet a runtime object.

The profile is not a tuning store: per-card constants remain in the device
profile, and per-file/run values (expert plane strides, bank budget, KV type,
layer map, `TB` buckets, prefill chunk) stay arguments.  A model profile and a
device profile answer different questions: "what is this model and what may it
use" versus "how does this card run it".

### 3.2 Multimodal and the shared chunker

The text GGUF has no vision/audio tensors (the towers are separate mmproj
GGUFs, as for every architecture).  `load_qwen4exp` sets
`arch_profile.mm_model_prefill`, and `generate_impl` then calls `q4_prefill_mm`
for the whole expanded prompt instead of one `kMaxT` row per `prefill_chunk`
(the generic loop would re-upload experts every 32 tokens and bypass the
HC/PLE/QSA state).  `q4_prefill_common` is the shared chunker: full `kMaxT` rows
use mode 2, the tail uses mode 1, and each chunk slices
`mm->mrope`/`mm->img_row` with `mm_write_step`, so an image block may span rows
without shifting a position or an embedding row.  With `mm == nullptr` it is the
old text path (same layout/positions), so text output is bit-identical.  The
decode loop sets `mm_decode_rope_` so `q4_forward`'s `step_info` reset does not
clear the per-step M-RoPE row; text-only runs and `reset_state` clear it.
`generate_mm` throws on media rows with no encoder output
(`has_encoded_embd`), which is the usual missing `--mmproj`/`--audio-mmproj`.
At startup the engine prints one `[mm]` line when the tokenizer has none of
`<|image_pad|>`/`<|video_pad|>`/`<|audio_pad|>`, so a media request fails at
prompt expansion with a clear message instead of a forward on an impossible
prompt.  `test_mm_step` (hermetic) pins the slice, a chunk boundary inside an
image block, and the error paths.

### 3.3 Device-profile provenance (M0)

The qwen4exp-specific per-card numbers no longer live at the call sites:
`moe.cfg_gate`/`cfg_down`/`cfg_dec`, `moe.flat`/`flat_dec`,
`moe.dp4a`/`dp4a_dec`, `moe.upload_ring`, `moe.pf_wmin` and
`shape.w4_rows_per_wg`/`q8_rows`/`f32_rows` are fields in
`src/device/profiles/arc_a770.cpp`.  The A770 values are 43/43/0 (0 = decode
auto), 1/0, 1/1, 32, 24, 16, 512 and 128; the Iris Xe file carries the same
values but marks the whole `moe` group and the three shape fields **INHERITED /
UNMEASURED** - the expert tile's register constraint is shared, but the
bandwidth/occupancy balance that picked M=8 and the row thresholds that scale
with the EU count were swept only on the 512-EU card.  `PF_DEVICE_INFO=1`
prints the resolved groups.  The env remains the A/B; a field whose 0 is a
measured value (`cfg_dec` auto, `PF_Q4_F32_SPLIT=0`) reads the raw env with a
`-1` sentinel because `profile_int` folds env 0 into "unset".

## 4. QSA

The selector runs on the host (`engine_qsa.cpp`).  The attention over the
selected keys runs on the host for CPU, non-f16 KV, a non-256 head and the
`PF_QSA_DEVATTN=0` / `PF_QSA_PF_DEV=0` A/Bs, and on the device for the f16 GPU
sparse paths (one token via Stage 2, a whole prefill chunk via Stage 3 below).
The host path keeps a mirror of each full-attention layer's K/V, so it needs no
device gather; either path writes the result back for the existing `wo` GEMV.
The cache stores the **raw** indexer key (the reference pools raw keys before
`k_norm`+RoPE), the mirror rounds to f16 like the reference's cache, and the
top-2048 selection is a `std::partial_sort` with ggml's comparator.

Up to 2 048 keys the reference's selection covers every key, so the dense path
is exactly equivalent.  Above it the sparse path runs and appends the
incomplete tail.  Measured against llama.cpp layer 3 at 2 062 tokens: q/mixed
cos 0.9999, indexer score corr 0.9994.  The top-512 sets differ by one
boundary pool (fp accumulation order), so bit-exactness is not claimed.
`PF_Q4_QSADBG` and the `PF_QSA_*` dump knobs are diagnostics.

**The pool width is `hp.idx_kpool`, not a literal 4.**  `qsa_pool_div()` returns
`idx_kpool > 1 ? idx_kpool : 4` and every pool expression uses it: the pool
array and per-slot ring, `pos / kp`, `pos % kp`, the pooled key's RoPE position
`pos - (kp - 1)`, `n_pool_max = idx_top_k / kp`, the selected range `p*kp + j`,
and the incomplete tail.  The old `/4` arithmetic was invisible on this file
(its `compress_ratios` is 4) but a model that pools 8 would have selected the
wrong key set while still running.  `kp = 4` is bit-identical to the old code:
`1/4` is exact and the four-term sum keeps its order.  The guard is
`assert(kp >= 2)` plus a grep guard - no `/ 4`, `% 4` or `* 4 +` pool
arithmetic may remain in `engine_qsa.cpp`.

**The recommended `--ctx 2048` decode does not run QSA**: `qsa_on_` requires
`max_seq > idx_top_k` and both are 2 048, so the 93.3 ms/token cell has no QSA
cost.  Measured at `--ctx 4096` (same weights, bank and TCP prompt), the host
call was **26.8 ms/token** and **23.7 ms of it was the four blocking D2H waits
per layer** (`memcpy().wait()` drains the in-order queue while the host MoE
would otherwise overlap the device), not the dots: indexer BF16 projection
3.0, mirror/pool update 0.08.  The payload is 744 KiB/token; the wait is mostly
the layer's own producers.  The fixes, in order, are in
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
item:

| per token, 12 full-attention layers | scalar | AVX2 | AVX-512 |
|---|---:|---:|---:|
| sparse decode, ms/token | 203.6 | 183.8 | **180.0** |
| sparse attention, ms | 39.3 | 19.2 | **16.0** |
| selector (512 pools), ms | 1.73 | 1.24 | 1.30 |
| D2H copy wait, ms | 32.2 | 32.1 | 32.2 |
| crossing prefill chunk, attention ms | 208.0 | 79.3 | **68.2** |
| 1024-token dense prefill, indexer GEMV ms | 91.5 | 55.4 | **44.9** |

The selector (512 pools) is serial on
purpose: ~2.7 us/layer is below one thread-pool wake.  Against the scalar path
the AVX-512 run has score cosine **1.000000**, the **same 512 selected pools**
(512/512), a **bit-identical** layer-0 `mixed` dump and indexer q max|diff|
**1.2e-6**.  In the 96-token cell AVX-512/AVX2/scalar are indistinguishable end
to end (within a +-0.2 ms run spread) because the host work is hidden under the device tail;
the copy restructure is the win there, not the vector width.  The three forms
live in `src/backend/cpu/kernels/qsa.cpp`; `PF_QSA_ISA` isolates them.

**The 32.2 ms above is mostly a producer drain, not copy bandwidth.**  The
four D2H copies are 10,240 + 49,152 + 2*2,048 = 63,488 B per layer, i.e.
**744 KiB per token** over the 12 full-attention layers.  The host cannot run
the sparse attention before `qk_norm_rope` has produced Q/K/V and cannot run
`wo` before the attention, so the wait is the layer's own GEMV/rope/FFN, not a
PCIe limit; a second copy queue does not remove the dependency.  The low-risk
wins are pinned per-layer staging, splitting the `mixed` copy from the selector
(the selector only needs `mixed` and the indexer q), storing the host K/V
mirror as f16 (it is already f16-rounded; attention 16.0 -> ~9-11 ms), and
putting the deferred dense-prefill K/V copies on a per-device copy queue
(~1.2 s per 1024-token chunk).  Removing the 16 ms host sparse attention needs
the device gather + `attn` path (Stage 2) with the selector left host-side;
that changes the accumulation order and needs a sparse-stream oracle.
(`PF_CPU_ISA` also scalarises the MoE/HC/PLE and is not an A/B of this path).

**Stage 1 (2026-10-27).**  The low-risk items are implemented in
`engine_qsa.cpp`/`qsa.cpp`: per-layer pinned USM staging for the four D2H
buffers and `attn_out` (a pageable `std::vector` D2H is synchronous, so even the
deferred copies blocked the host), an f16 host K/V mirror (`PF_QSA_F16MIR`,
default on for AVX2/AVX-512 and off for scalar, where f16 is 4.6x slower), and
`PF_QSA_COPYQ` (default on) putting the dense big-chunk K/V copies on a
per-device in-order copy queue with a guard before the next attention layer.
The `--ctx 4096` numbers: short dense QSA call 7.17 -> **1.67 ms**; 1024-token
dense chunk 1434 -> **56 ms steady** (first chunk 279-322 including the mirror
allocation; ~1.38 s saved per chunk); sparse decode attention 13.0 -> **8.0 ms**;
crossing-128 attention 842 -> **552 ms**; output md5 and selector pools
byte-identical.

| short cell (96-token decode, `--ctx 4096`) | `[q4]` sum | true wall | QSA call |
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
scalar/AVX2/AVX-512.  The 93.3 ms `--ctx 2048` cell is unchanged (QSA off); at
`--ctx 4096` the dense-equivalent update is deferred under the LM head, so the
96-token cell also stays at 93.4 ms.  A
pre-existing lifecycle bug was found and fixed: `qsa_cache_` was not cleared by
`engine::zero_slot`, so a server reusing a slot above `idx_top_k` could read the
previous sequence's mirror.  `engine::qsa_reset_slot` now flushes, clears the
pending entries and destroys the per-slot mirror/pool from `zero_slot` and
`reset_state`; QSA-off is a no-op.  Verified on the q4 server with three
sequential slot-0 requests: B1 == B2, A differs, 12 layers cleared, B1/B2
selector dumps byte-identical; the QSA-off and short QSA-on md5s are unchanged.
A prefix-cache resume at position > 0 would need saved QSA state (qwen4exp has
the prefix cache off).  The q4 server now uses the whole-prompt q4 prefill
(no `PF_PFB_MAX_M`; 2.2x; §7.1).

**Stage 2 (`PF_QSA_DEVATTN=1`, default on for f16 GPU sparse decode).**  The host selector/indexer/raw-key
pool is untouched; for sparse decode (`need`, n_tok==1, GPU, f16 KV, head_dim
256) the selected rows are gathered into a compact f16 pool and the existing
device `be.attn`/`attn_combine` runs with a device scratch `step_info`;
`attn_combine` applies the QBUF sigmoid gate once and `wo` follows.  Measured on
2412-token sparse decode: QSA call **42.31 -> 35.29 ms**, decode wall
**164.97 -> 154.01 ms** (-6.6%; median 162.42 -> 153.32), QSA `attn`
**7.99 -> 3.04 ms**, `copy` unchanged 26.7; the 32-token stream is identical,
the 16-token logits agree 16/16 top-1 (max KL 0.00391), the selector score/pools
are byte-identical, and the one-layer ATTN_OUT/`wo` cos is 0.999999919 /
0.999999788.  `--ctx 2048` QSA-off and `--ctx 4096` short md5s are unchanged.
Caveats: device tokens skip the main K/V mirror update, so a post-decode sparse
prefill on the same slot would need a mirror rebuild (the q4 prompt->decode
flows never do; `zero_slot` clears); `head_dim != 256`, i8 and CPU use the host
Stage 1 path (sparse prefill has its own device path, Stage 3 below); the
first device token allocates ~17 MB/device (~9.8 ms).
Default on for f16 GPU sparse decode on prompt->decode flows; `PF_QSA_DEVATTN=0`
keeps the host Stage 1 path.

**Stage 3: sparse prefill on the device (`PF_QSA_PF_DEV`, default on).**  A
sparse prefill chunk (`n_tok > 1`, `pos > idx_top_k`) used to run the host
attention over the mirror once per token - 4.6 s of QSA attention on the
1024-token crossing chunk and 2.3 s on the 576-token tail of the 3660-token
prompt below.  The selector is still the host one (same `std::partial_sort`
comparator, now run with `par(n_tok)` over independent rows into per-layer
pinned buffers), so the selected sets are bit-identical.  `qsa_devattn_prefill_run`
copies the ids and counts once per layer and then walks `cap_wave` rows per
wave:

* `qsa_sinfo_multi_launch` makes each row a one-token sequence whose position
  is its selected-key count and whose slot is its row index;
* `qsa_gather_rows_launch` compacts each row's selected paged-KV rows into its
  own slice of the compact pool (row `r` owns blocks
  `[r*cap_blocks, (r+1)*cap_blocks)`; the block table is the identity with a
  `cap_blocks` stride);
* the ordinary `attn`/`attn_combine` run over it with `partials_dec` and the
  decode split count (`PF_QSA_PF_NSP` may lower it).

One slot per chunk is required (a mixed-slot chunk falls back to the host
path).  The gate is GPU + f16 KV + `head_dim == 256` + `n_tok <= kMaxB*kMaxT`;
dense forwards (`pos <= idx_top_k`) never enter it.  `PF_QSA_PF_WAVE` (default
8) sets the compact-pool rows; its cost is
`wave * 65 * n_head_kv * 32 * head_dim * 2 * 2 B` of VRAM (77 MB/device at 8)
plus the per-layer pinned selection staging (8.5 MB/layer at `kMaxB*kMaxT`).

Measured 2026-11-03 on 2x A770, the 3660-token `/tmp/prompt4k.txt` cell
(`--ctx 4096 --kv-type f16`, `--moe-vram-mb 17500`, 28 threads, tail-skip 0,
one process per cell, two interleaved reps; the chunk wall is the `[qsa] fwd
wall` sum, not `[q4] rest`, which excludes async device work):

| | `PF_QSA_PF_DEV=0` | `=1` |
|---|---:|---:|
| prefill, sum of chunk walls | 75.47 s | **67.83 s** |
| post-load process wall | 80.88 s | 72.87 s |
| chunk 3 (1024, sparse) wall | 24.13 s | 19.05 s |
| chunk 4 (576, sparse) wall | 14.82 s | 12.26 s |
| QSA attention, chunk 3 / chunk 4 | 4.64 / 2.30 s | **0.046 / 0.026 s** |
| QSA selector, chunk 3 | 1.50 s | 0.13 s |
| decode median | 131-149 ms | **113 ms** |

The prefill gain is 7.6 s (-10%) and the QSA call itself falls to ~1/100; what
remains of the crossing chunk is the gather's D2D traffic (each row re-gathers
its ~2 080 keys) and the existing paged attention.  Decode is unchanged (its
113 ms matches the cell without the new path; the 131-149 ms spread of the
host-prefill cells is the expert bank state the different prefill leaves).

**Correctness and the f16 rounding caveat.**  On a 2560-token prompt the only
sparse chunk is the final 512-token one, so chunks 1-2 take the same dense path
in both cells: `PF_QSA_SCORE`/`POOLS`/`MIX` are byte-identical, and the layer-0
attention over all 512 rows matches the host attention to cosine
**0.99999976** (mean 1.0, rel-RMS max **1.2e-6**) - but only when both read the
same f16 K/V.  The shipped host mirror uses `ggml_float_to_half`, which
**truncates** the mantissa, while the device KV store casts to `sycl::half`
(round-to-nearest); that one-ulp disagreement is ~6e-4 relative RMS and is
about half of the final-logit difference on that prompt (KL 0.20 with the
truncating mirror, 0.045 with `PF_QSA_F16RND=1`, which makes the host mirror
round and is a diagnostic only).  On the full 3660-token prompt the first-token
logits land on a near-tie - 760 vs 27775, margins 0.13 / 0.19, KL 0.0127 - so
the continuation differs; that is the same accumulation-order sensitivity the
Stage 2 decode already has.  A dense 512-token prompt's stdout is md5-identical
between the two cells.  `PF_QSA_ATTN_DUMP_ALL=1` writes every attention row (the
per-row oracle used above); `PF_QSA_F16RND` is not the shipped host path.

## 5. PLE and the engram cache

The PLE layer hashes the token and its 16 n-gram context into 16 rows of a
160-wide table, projects them (`ple_key`/`ple_value`), gates them and applies a
depthwise convolution into the residual.  The host module is
`cpu_q4_ple_gate` + `cpu_q4_ple_apply`; `ple_conv1d` is `[kernel][channel]`
with `nb0 = kernel`, so tap `k` of channel `c` is `w[k + kern*c]`.

`--engram-ram-mb` (unset = the model profile's `policy.engram_ram_mb_default`,
4096 MiB; `0` disables) is a direct-mapped host-RAM cache of already
dequantized PLE rows.  The capacity is a power of two, the slot is
`row & (cap - 1)`, and the tag is the row id; a miss dequantizes from the GGUF
mmap, which is the disk tier.

### 5.1 What the PLE block actually costs (2026-11-01)

`PF_Q4_PLEFAULT=1` splits the block per token (`[q4ple] idx/gather/proj/d2h/
gate/apply/h2d`).  On the recommended config it was **3.94 ms**, and only
0.075 ms of that was the gather:

* **The engram cache's 1.5 % hit rate is capacity, not a bug.**
  `per_layer_token_embd` is IQ4_NL, **320,001,536 rows x 90 B = 28.8 GB**, and a
  token touches 16 rows = **1.44 KB**.  The index is `row & (cap-1)` with a
  default cap of 4,194,304 slots = **1.31 % of the rows**, and
  `--engram-ram-mb 16384` measures 4.7 % - i.e. `hit ~= cap/rows` exactly.
  There is no temporal reuse to exploit and the gather is 0.075 ms, so
  prefaulting 26.8 GiB (`PF_Q4_PLE_PREFAULT`) is not worth it.
* **`apply` was 1.90 ms against ~0.2 ms of work**, because each of its four
  10 240-element loops paid a `par()` wake + rendezvous of the 28-thread pool
  (**~0.5 ms per call** at this size; a pure 10240-float copy measured
  0.55 ms).  Regions whose **work** is <= `PF_Q4_PLE_SER_MAX` (65536 elements)
  now run on the calling thread: apply 1.90 -> 0.45 ms, block 3.94 -> 2.50 ms,
  **-1.2 ms/token, byte-identical**.  A prefill's 5.2 M-element regions still
  pool - the threshold is on the work, not on the job count, or loop 2 (only
  `n_tok*hc` jobs of `n_embd` elements each) would go serial at n_tok=512.
* **`proj` was 1.61 ms**: two host GEMVs over 34.8 MB at 21 GB/s.  The weights
  are already in the layer's VRAM (`upload_device_weights` covers
  `L.ple.key/value`), so `PF_Q4_PLEGPU=1` runs them with the existing Q8_0
  decode GEMV: **0.190 ms of device time = 183 GB/s**, 0 extra MiB, no extra
  D2H (the `to_host` the gate/apply path already did becomes the wait).  It is
  **-1.60 ms/token** and **opt-in**, because the device dot reassociates and
  the stream diverges at output token 38 on a genuine near-tie (the top two
  logits are 0.031 apart in both configs).

Measurement trap worth knowing: with a **cold** PLE page cache the gather is
10-12 ms/token (16 major faults), which swamps the 1.6 ms win - so both arms of
that A/B must run inside one lock, back to back.


## 6. Expert bank: per-expert LRU and two-pass prefill

The generic three-tier `moe_cache` is **inactive** on qwen4exp: the model
profile says `moe = runtime_bank`, so `engine.cpp` sets `moe_host_only_` and
`host_only = true`, the cache computes `total = 0`, and its device path lives
in `record_forward`'s `seg_plan` branch while q4exp uses its own layer walk.
The q4 path allocates its own banks:

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

The bank is the shipped default and it is output-correct: `PF_Q4_FFCMP` gives
cosine 0.999999 for a banked layer.  The earlier "do not ship it enabled"
reading came from the stale `b.block` accumulator (the reduce accumulates) and
the K%256 down-row stride, both fixed above, not from the bank; `PF_Q4_BANK=0`
is the all-host A/B.  The bank's per-card knobs are the device profile's
`moe.*` fields (§3.3); `--moe-vram-mb` remains a run argument, not a profile
value.

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
and 122.2 against 93.2 with `PF_Q4_TAIL_SKIP=7` (tail7, no longer the default).
The blocker is the resident
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

### 6.2 Placement and transfer details (2026-11-01)

* **R is per device.**  The budget is split evenly, but the two devices do not
  carry the same number of layers: on `0-19:gpu.0,20-47:gpu.1` device 0 owns 20
  and device 1 owns 28, so one `R` = the minimum left ~2.8 GiB of device 0's
  VRAM unallocated (104 slots/layer for both).  Per-device gives device 0 ~152
  and device 1 ~104 - fewer misses on the **slow** (x8, 5.17 GB/s) card at no
  cost on the fast one.  `q4_lru_age_`/`q4_lru_heat_` are now variable-stride,
  indexed through `q4_r_off_`; indexing them as `il * R` is wrong.
  `PF_Q4_PERDEV_R=0` restores the single-R behaviour.
* **The decode's expert H2D rides the per-device copy queue**, and the down
  plane is issued last and waited for only just before `down`, so ~1.2 MB of
  DMA runs under the gate/up GEMM.  Safe only because the router-ids `qd.wait()`
  drains the compute queue first - the previous layer's kernels must not still
  be reading a slot the copy overwrites.  Every path must still **upload** the
  down plane even when the split wait is off (`PF_Q4_DOWN_PREF=0`) or the copy
  queue does not exist (`PF_Q4_STAGE=0`); only the wait point may vary.
* **The routed-down int8 host store is built only for layers that do not run in
  the bank.**  It used to be built for all 48 layers at startup: **42.2 GiB and
  112.8 s** for a store only the non-banked layers (one, layer 2) and the
  `PF_Q4_FFCMP`/`PF_Q4_CPU_MISS` diagnostics read.  Selective: **1 layer,
  0.9 GiB, 0.5 s**; whole-run wall 200 -> 56 s, resident set 122 -> 80 GiB.
  `PF_Q4_FFCMP` forces the full build, so the oracle's reference is unchanged.
* **The non-bank layer's shared expert used to be discarded.**  The non-bank
  path ended with an **unconditional** `from_host(dev, b.block, q4_host_out_)`;
  on the `sh_gpu` path that copies a stale host scratch over what
  `moe_gate_add_launch` just wrote, and `PF_Q4_DUMP=1` showed layer 2's
  `ffn_out` **bit-identical to its `moe_routed`**.  Removed.  This block's
  routed residual is computed on the *host*, so unlike the bank path it needs
  no `to_host` - adding one would overwrite it.

``moe_down_q5_1_launch`` now runs **NPB=1** by default (bit-identical, down
kernel 0.1875 -> 0.1674 ms/layer, decode 90.71 -> 89.41 ms/token, identical
md5).  The old NPB=8 came from a probe whose 12.29 MB working set was replayed
against a 16 MB L2 - see `docs/design/12-build-and-testing.md` §6.1.


## 7. Measured (2x Arc A770, AOT Release)

| configuration | 512-token prefill | decode |
|---|---:|---:|
| host experts, bank off | 48.7 s (10.5 tok/s) | ~500 ms/token (~2 tok/s) |
| whole-layer bank, 18000 MB | 45.5 s (11.25 tok/s) | ~470 ms/token (~2 tok/s) |
| full host mirror, GPU reads over PCIe (Strata-style) | 14.1 s (544 tokens) | 178 ms/token |
| per-expert LRU, all-GPU, 16000 MB, pinned source | ~12.0 s (45 tok/s) | ~101 ms/token (9.9 tok/s) |
| **per-expert LRU, all-GPU, 17500 MB, pinned source** | **~11.4 s cold / 9.6 s warm (47-56 tok/s)** | **~93.3 ms/token (10.7 tok/s)** |
| per-expert LRU, all-GPU, 18000 MB, pinned source | – | ~93.3 ms/token (10.7 tok/s) |

The decode row is the 2026-10-27 configuration; §7.2 has the per-stage budget
of one decode step on the current build.

The pinned expert source is 69.8 GiB of host RAM copied once at startup.  A
cold first prefill is **11.4 s** (the bank calibrates on it); the second/steady
prefill is **~9.6 s (56 tok/s)**.  Decode reaches **~93.3 ms/token
(10.7 tok/s)** after ~16 tokens of LRU warm-up; a 64-token generation is
~15.6 s wall including the warm prefill.  18 000 loads but is not faster than
17 500.  Correctness:
`12 x 9` -> 108, the TCP/UDP prompt is coherent, the 27B
`test_decode_vs_prefill` oracle is 8/8 OK, and `PF_Q4_FFCMP` gives cosine
0.99998 against the host routed FFN (layer 4's Q8_0 down 0.999999, layer 2's
Q5_K/Q8_0 0.99995).  The profile/device/chunker changes add three hermetic
checks: `test_model_profile` (registry x exact profile keys and the qwen4exp
values), `test_mm_step` (the shared M-RoPE/image-row slice) and the extended
`test_dev_profile` (`moe.*`/`shape.*` on the A770, the Iris inheritance and the
unknown fallback).

### 7.1 Measurement log: batching, tail-skip, MTP and the prefill ceiling

Unless a cell says otherwise: AOT Release, 2x Arc A770, `--layer-map
0-23:gpu.0,24-47:gpu.1`, `--moe-vram-mb 17500`, `--cpu-threads 28`,
`--ctx 2048 --kv-type f16`, `PF_PREFIX_CACHE=0`, one process per cell,
2026-10-27.  The deep-context cells use f16 KV; the 512-token prefill archive
is i8 and its decode is f16 (a 2 426-token f16 prefill measures 53.4 tok/s, so
the KV type is not the difference).  [`../PERFORMANCE.md`](../PERFORMANCE.md)
§4.4 is the summary.

#### 7.1.1 Two-row decode: correct, but 0.85x

The q4 branch of `engine::decode_batch` sets one `step_info` row per sequence
and calls `q4_forward(0, n_rows, true)`.  `test_q4_batch` feeds both rows the
same teacher-forced stream, once decoded alone and once in one 2-row forward:

| | ms/step | tok/s | MoE | rest | upload |
|---|---:|---:|---:|---:|---:|
| row 0 alone | 95.96 | - | 80.08 | 12.51 | 132.5 MiB/step |
| row 1 alone | 80.55 | - | 63.88 | 11.36 | 61.3 MiB/step |
| **two alone (combined)** | **176.50** | **11.33** | - | - | 193.8 MiB/2 tok |
| **2-row batch** | **208.06** | **9.61** | 172.16 | 16.79 | **44.8 MiB/2 tok** |

The batch's MoE is **1.196x** the sum of the two singles: two prompts select
~20 distinct experts per layer against ~10 each, so the routed-expert weight
read doubles.  The dense trunk does amortise (batch `rest` 0.703x), but saves
only ~7 ms against ~28 ms lost in the experts.  The batch uploads **2.7x fewer
MiB/token** and is still slower, so it is weight-read bound, not H2D.  The LM
head runs at `TB=8` for two rows (the `q4_gemv_tb` bucket), so ~19 ms of the
residual is dead head work; removing it still leaves 0.91-0.94x.  Both rows are
numerically real and matched to their own baseline: top-1 **12/12** each, KL
mean **0.026** / **0.008**, max|dlogit| 2.28 / 1.16, no NaN, both rows moved in
11/11 step pairs.  A later run on the merged build measured 205.11 ms for the
batch against 169.58 for the two singles (**0.827x**).

| q4 server, two concurrent 32-token requests | aggregate | notes |
|---|---:|---|
| serialized one-row cap (shipped) | **5.40 tok/s** | req0 8.61 s, req1 11.84 s |
| serialized one-row cap + 28 threads | 5.57 tok/s | req0 8.31 s, req1 11.50 s; single 6.19 tok/s |
| round-robin one-row cap (tried, removed) | 4.33 tok/s | 14.79 s wall; two live sequences thrash the shared expert bank |
| one request at a time | 6.09 tok/s | 5.25 s for 32 tokens |

The shipped policy is **one row per forward**; the scheduler serialises
qwen4exp's sequences.  `PF_SCHED_MAX_DECODE` overrides the cap for tests.

#### 7.1.2 CPU threads and tail-skip

The default `PF_CPU_THREADS` is the physical-core count (14 on this 28-thread
box), but the host-side MoE/PLE/HC work scales with the logical count.  On the
96-token TCP prompt:

| threads | tail-skip | decode mean | wall (96 tok) | note |
|---:|---:|---:|---:|---|
| 14 | 0 | 120.7 ms | 20.95 s | default |
| 28 | 0 | 109.6 ms | 18.08 s | md5 identical; `rest` 23.22 -> 13.67 ms |
| 14 | 7 | 100.1 ms | 16.49 s | |
| **28** | **7** | **93.3 ms** | **15.52 s** | recommended; 10.7 tok/s |

The 93.3 ms cell is 28 threads + tail7 (no longer the default); the default-14
tail7 cell is 100.1 ms.  The server gains much less (5.40 -> 5.57 tok/s
aggregate; a single request 6.09 -> 6.19) because its q4 path is dominated by
the prefill and HTTP/sample path, and it still pays the one-row schedule.

Tail-skip (`PF_Q4_TAIL_SKIP=N` / `--q4-tail-skip N`, default 0 = the exact
expert set) drops a miss whose best rank over the batch is >= N: it stays
`sm[e] = -1`, contributes zero and is neither copied nor computed.  The
measurements below were taken at the old default, N=7, and are kept as
historical results.  It is a real approximation, gated on two prompts:

| prompt | top-1 agreement | first diff | KL at first diff | decode mean | wall |
|---|---:|---:|---:|---:|---:|
| short `The TCP/IP protocol is` | 28/29 steps | 28 | 0.029 | - | 14.35 -> **11.99 s** |
| 96-token TCP | 15/16 steps | 15 | 0.037 | 120.7 -> **100.1 ms** | 20.95 -> **16.49 s** |
| 96-token TCP, 28 threads, tail0 | - | - | - | 109.6 ms | 18.08 s |
| 96-token TCP, 28 threads, tail7 (no longer the default) | same stream as the 14-thread tail7 row | - | - | **93.3 ms** | **15.52 s** |

The first divergence was a near-tie in both (e.g. base 21.58 vs 21.07 with the
other candidate at 21.0); step-0 prefill logits had KL 0.019-0.021 with the
same top-1, and both outputs remained coherent.  The archive's 96-token cell
measured 93.6 -> **86.9 ms** (-7.2%), last-token upload **56.2 -> 32.2 MiB**
(-43%); rank 8 is 90.0 ms.  The default is now 0 (the exact expert set);
`--q4-tail-skip N` / `PF_Q4_TAIL_SKIP=N` opts back into the approximation, and
a performance-first decision accepts the near-tie flips when it is enabled.

#### 7.1.3 MTP verify gate: NO-GO

`test_q4_verify` prefills a 96-token warm prompt (bank calibration ~6.2 s), then
times a one-slot, n-consecutive-token `prefill_text` suffix (the existing mode-1
verify shape, LM head included) for n=1..6.

| run | n=1 | n=2 | n=3 | n=4 | n=5 | n=6 |
|---|---:|---:|---:|---:|---:|---:|
| warm, first gate | 257.8 | 318.0 | 416.4 | 427.7 | 496.8 | 496.9 |
| warm, merged build (tail7+28) | **234.2** | 295.5 | **375.6** | 392.1 | **434.1** | 462.4 |
| cold, merged build | 352.0 | 344.1 | 358.5 | 401.7 | 442.9 | 457.5 |

The kill thresholds were warm n=3 < 190 ms and n=5 < 230 ms; the measured
values are 1.9-2.2x over.  Even n=1 is 2.5x a plain token (93.3 ms), so the
existing mode-1 forward is not the verify shape.

> **MTP q4 result: NO-GO (2026-10-27).**  The verify gate (tail7, no longer the
> default, +28, warm)
> measured n=1/3/5 = **224.5 / 367.7 / 420.6 ms** (reproduced 368.3 / 418.7)
> against the kill lines 190 / 230; `PF_Q4_TIME` puts **207/331/403 ms** of the
> 232/368/447 ms forward in the FFN with only 150/436/576 MiB H2D, so the cost
> is the routed-expert compute/bank, not PCIe.  Each extra row is ~35-40 ms, the
> k+1 forward is 3.5-4.5x a plain token, and k=2/k=4 stay at 0.72x-1.0x even at
> the best measured acceptance; no k pays.  **No MTP code was added**; the
> draft semantics/state plan are design notes only.  The gate exposed two real
> q4 issues: `pos_of()` returned `pos[0]` for a multi-token mode-1 row (PLE
> positions after the first tail token; fixed to `pos[0]+i`, output
> byte-identical because PLE uses position differences), and
> `q4_hot_calibrate()` was dead code (wired at the first prefill behind
> `PF_Q4_CAL`, default off: +0.9-1.27 s wall, no decode-rate change).  The
> earlier 202/61 ms cold-bank claim was a load-time subtraction artifact; the
> actual first-32/steady is **94.4/94.1 ms**.

#### 7.1.4 The prefill ceiling

The 9.6 s warm 544-token prefill is ~4.3-4.9 s of expert upload (31-32 GiB at
6.9-7.9 GB/s), ~3.5-4.2 s of device MoE and ~1.1 s of non-MoE.  Host-spilling
the overflow experts has a ~0.25 ms/expert break-even against the existing
0.32-2.4 ms/expert path (no-go as a default), and a 2-chunk layer pipeline adds
~45% expert copies (+14.5 GiB, +1.9-2.4 s), so it lands at ~9.7 s, not 7.35 s.
A layer-map rebalance to the faster card drops R from 121 to ~103 and costs
~10% decode for ~0.2 s prefill.  **74 tok/s is not reachable on two cards**; the
warm prefill stays ~56 tok/s.

The server now uses the same whole-prompt `prefill_text`: 2,412 tokens in
41.7 s (57.8 tok/s) one chunk, against 92.6 s for the old 32-token
`prefill_chunk` workaround (2.2x), output md5/ids matching the CLI.  The
32-token vs 1024-token chunking shifts the QSA selector pools slightly (score
cosine 0.9999945, max 0.0699, a few boundary pools) - the known prefill
accumulation-order sensitivity; the generated content matched.

#### 7.1.5 Deep context (2026-10-27)

Conditions: the recommended 28 threads and tail7 (no longer the default), f16
KV, `--ctx 4096` for the
512 cell, `--ctx 32768` for 2,426 and 16,445, and `--ctx 65536` for the 48k
cell.  The 48k cell uses `--moe-vram-mb 16000` (111 slots/layer, 15.1 GiB)
because 17 500 MB does not allocate at that context; the other cells use
17 500 (121 slots, 16.5 GiB).  Prefill is the sum of the request's `[q4] fwd`
calls; decode is the mean of the `n=1` calls.

| depth | prompt | prefill | pp | tg | ms/step | decode steps |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 512 | 9.6 s (warm) | 53 tok/s | 10.70 tok/s | 93.3 | 95 |
| 2 426 | 2 426 | 45.4 s / 45.1 s (two reps) | 53.4 / 53.8 tok/s | 7.28 / 7.42 tok/s | 137.3 / 134.7 | 9 |
| 16 445 | 16 445 | 398.4 s | 41.3 tok/s | 8.24 tok/s | 121.3 | 9 |
| 48 139 | 48 139 | 1 343.8 s | 35.8 tok/s | 6.51 tok/s | 153.6 | 7 |

Raw runs:

* **512**: the 96-token TCP decode at `--ctx 4096` measured 93.56 ms over 95
  `n=1` calls (first 32 94.42, last 32 92.03), i.e. 10.70 tok/s.  The prefill
  (9.6 s warm for 512-544 tokens) is the archive i8 cell; the decode is f16.
  QSA is dense-equivalent at this length and its deferred update is hidden under
  the LM head.
* **2 426**: two reps of a 2×1024 + 352 + 26 prompt at `--ctx 32768`: prefill
  45.442 / 45.125 s, decode 137.29 / 134.70 ms.  This is the first sparse-QSA
  cell.  It is slower than the 16k cell, which is not a depth effect (both read
  2 048 keys); it was not bisected and is treated as routing/cache variance.
* **16 445**: a 16×1024 + 32 + 29 prompt at `--ctx 32768`, bank 17 500.  The
  `[q4] fwd` prefill sum is 396.66 s; the first content token arrived at
  398.37 s (the head and the first sample).  The 9 decode steps are
  `[143.8, 108.9, 134.2, 142.1, 119.7, 117.1, 119.7, 100.4, 106.0]` ms.
* **48 139**: a 47×1024 + 11 prompt at `--ctx 65536`, bank 16 000.  The 47
  prefill chunks sum to 1,343.79 s.  The 7 decode steps are
  `[204.2, 151.2, 127.9, 132.8, 136.3, 177.8, 144.7]` ms; the first step is
  cold and the spread is the bank/routing, so 153.6 ms is a sample, not a
  steady state.

**The sparse QSA path is not depth-flat.**  The selector scores every pooled
key before choosing the top 2 048, so its host cost grows with the key count
even though the device attention reads only 2 048 rows.  That is why 16k is
faster than 2.4k (bank/routing) but 48k is slower than 16k (selector): the
decode is 121 ms at 16k and 154 ms at 48k.  The prefill falls monotonically
with prompt length (53 -> 41 -> 36 tok/s) because each chunk attends to all the
KV before it.

### 7.2 The 2026-11-01 pass: H2D ceilings, the expert GEMV, the dense floor

Unless stated otherwise: 2x Arc A770, AOT Release, `--layer-map
0-19:gpu.0,20-47:gpu.1`, `--moe-vram-mb 17500`, `--cpu-threads 28`,
`--ctx 2048 --kv-type f16`, 48-token greedy decode, one process per cell.
The box drifts ~3 % between runs; the cells below are alternating or
best-of-three.

| | before | after |
|---|---:|---:|
| whole-run wall (48 tokens) | 7.350 s | **6.42 s** (-12.7 %) |
| startup to first token | ~200 s | **56 s** |
| resident set | ~122 GiB | **80 GiB** |
| bank slots/layer | 121 (24/24 map) | 152 / 104 per device |

Per-token stage budget measured with ablations (`PF_Q4_ABL_NOGEMV` /
`PF_Q4_ABL_NOMIX`) and barrier-pair segment timing (`PF_Q4_QSA_TIME`-style,
`PF_Q4_KTIME`, `PF_Q4_SEGTIME`):

| line | bytes/token | measured | floor | verdict |
|---|---:|---:|---:|---|
| dense GEMV stream (`pre`) | 3559.9 MB | 13.40 ms (266 GB/s) | ~12.5 (shape) | at floor |
| GDN + attention cores | - | 1.52 ms | - | small |
| PLE block (layer 1) | 34.8 MB + 16 rows | 3.94 -> **2.50 ms** | ~1.1 | was off; fixed |
| routed gate/up | 18.4 MB/layer | 0.115 ms (160 GB/s) | - | at floor |
| routed Q5_1 down | 12.3 MB/layer | 0.1875 -> **0.1674 ms** | ~79 GB/s | at floor |
| router GEMV `ffn_gate_inp` | 251.6 MB | 286 GB/s | 376 | 75 % |
| router fused selection | 2 KB | 0.0422 -> **0.0377 ms/layer** | - | latency-bound |
| LM head | 675 MB | 1.85 ms (364 GB/s) | 1.80 | at floor |

The dense part is **Q8_0/F32, not u4**: `rep.raw_gemv_decode` means every dense
linear is read in its raw GGUF type at M=1, so the stream is 3559.9 MB and not
5.2 GiB, and the M=1 rate is 294-364 GB/s rather than the 376 GB/s streaming
peak.  Against a shape-priced ~12.5 ms floor, 13.40 ms is 84 % - the earlier
"1.45x off a 14 ms floor" was comparing the whole `pre` against the GEMV-only
part and took the floor at the streaming peak.

**The x8 slot's H2D is at its protocol ceiling, and it is 64 B, not 128 B.**
Pinned bulk copy on the x8 card (root port `16:02.0`): H2D **5.186 GB/s**,
D2H **5.685 GB/s**, against a 7.877 GB/s encoded line rate.  Every element
advertises `MaxPayload 128` (the A770 and the retimer *capability* is 128, so
256 is impossible), and the root complex's **RCB is 64 B** (`LnkCtl` bit 3 = 0,
read-only), so a read completion carries 64 B: the TLP ceilings are **5.73
(writes) / 6.00 (completions)**.  D2H is at 99 % of its ceiling and H2D at
86 %; the last 14 % does not respond to 1/2/4/8 queues and threads, a device
kernel instead of `queue::memcpy`, OpenCL instead of Level Zero, or 2 MiB
alignment.  The config bits that would help (`RCB`, `MRRS`) are `EPERM` under
Secure Boot + `lockdown=[integrity]`.


What moved, and why:

* the routed-down int8 host store is built only for the non-banked layers
  (42.2 GiB / 112.8 s -> 0.9 GiB / 0.5 s) - most of the startup win;
* `R` is per device, so the 20/28 layer map stops wasting device 0's VRAM;
* the decode's expert H2D rides a per-device copy queue with the down plane
  waited for after gate/up, and `moe_down_q5_1` runs NPB=1
  (0.1875 -> 0.1674 ms/layer, bit-identical);
* the PLE host block's four `par()` calls were a 28-thread rendezvous each
  (-1.2 ms/token), and `PF_Q4_PLEGPU` moves its 34.8 MB projection to the card
  for another -1.60 ms/token (opt-in: not byte-identical);
* the fused router selection batches its SLM loads (-0.22 ms/token,
  bit-identical).

**What did not move, and why it cannot easily.**  The dense non-expert stream is
at its M=1 floor (3559.9 MB/token in 13.40 ms = 266 GB/s, LM head 364 GB/s =
97 % of the streaming rate), and the routed Q5_1 down is at the 24-byte-block
load-shape ceiling rather than a bandwidth one.  The x8 card's pinned H2D is
**5.186 GB/s** against a 6.00 GB/s 64-byte-completion ceiling - a protocol limit
(MaxPayload 128 is a hardware cap and the root complex's RCB is 64 B), not an
engine defect.  The candidate large change - a byte-neutral 128-value panel
repack of the Q5_1 down plane - was implemented and measured: it is
bit-identical and **slower** (0.1797 -> 0.2073 ms/layer), because bit-identity
pins the 4-byte-per-lane `qs` load shape; §9.1 has the numbers.

#### Cold start: the pinned-source read was one serial memcpy (2026-11-04)

The 69.8 GiB pinned expert source was populated by one `std::memcpy` per expert
plane (141 calls, one host thread) out of the GGUF mmap.  On a cold page cache
that single fault path is the whole disk pass and it dominates startup, and
`page_out_host` then evicted the shards with `POSIX_FADV_DONTNEED`, so the next
start paid it again.  Three changes, all default on:

* `PF_Q4_PREAD=1` reads each plane **directly from its shard fd** into the
  pinned destination (`pread`), split over `PF_Q4_PREAD_WORKERS` (8) workers.
  The split GGUF already knows each tensor's shard: `tensor.data - shard.base`
  is the file offset, and a range that does not lie wholly inside one shard
  falls back to `memcpy` (the reservation's padding is `PROT_NONE` and is never
  read).  `=0` restores the serial memcpy.
* `PF_Q4_PREFAULT` now **skips** a banked layer's expert planes (they live in
  the pinned source, not the mmap) and the GPU-resident shared/router tensors;
  it keeps the non-banked host layer, the PLE table (opt-in) and the host QSA
  indexer.  `PF_Q4_PREFAULT_ALL=1` restores the old every-tensor touch.
* `PF_Q4_KEEPCACHE=1` (qwen4exp default) drops the PTEs but **not the page
  cache** in `page_out_host`, so a restart maps the shards instead of reading
  them.  The expert pages were already kept by the `!moe_resident` keep-set;
  this is what keeps the ~5 GB of dense weights warm across runs (~14 s).

Same box, same binary family, one process per cell, 2x A770,
`--layer-map 0-19:gpu.0,20-47:gpu.1 --moe-vram-mb 17500 --cpu-threads 28
--ctx 2048 --kv-type f16`, cold simulated with `posix_fadvise(DONTNEED)` on the
four shards after `sync`:

| arm | cold wall | pin phase (cold) | prefault (cold) | warm wall | peak RSS |
|---|---:|---:|---:|---:|---:|
| HEAD (serial memcpy, touch every tensor, evict) | **284.6 s** | ~247 s | 3.0 s | **56.4 s** | 76.1 GiB |
| pread only | 109.7 s | 64.9 s | 2.1 s | – | 76.1 GiB |
| pread + prefault skip | 114.6 s | 67.0 s | 1.4 s | 43.1 s* | 5.8 GiB |
| **pread + skip + keep cache** | **109.4-114.6 s** | **65.1-67.0 s** | **1.3-1.7 s** | **28.0-28.7 s** | **5.8 GiB** |

\* the next run after a `PF_Q4_KEEPCACHE=0` run; a second keep-cache run is
28.7 s.  The 56.4 s HEAD warm is the documented 52-55 s cell; the new warm is
not slower.

The cold floor is the **disk, not the engine**: this NVMe reads ~1.0 GB/s cold
sequentially, and 8 and 16 pread workers both measure 1.13-1.16 GB/s aggregate
(69.8 GiB / 1.15 GB/s ≈ 60 s).  One outlier cold run measured 142 s for the
pin phase under a transient; the two repeats are 65-67 s.  The old target of
25-35 s would need ~2.5 GB/s and is not reachable on this drive.

**Exactness.** Against a pristine HEAD build of the same commit (not just
knobs-off in the same binary): a 96-token greedy decode is **bit-identical at
every step** (`PF_DUMP_DEC_LOGITS`, 96/96 files) and the stdout md5 is
`e18cd2d7b663815ed68b7e1be1cdf695`; a 4212-token prefill's post-prefill logits
are bit-identical (`4940...` on the earlier build, `1efcbcd8...` on HEAD and
both new arms) with stdout `04399da0...`.  Within the new binary, pread-vs-memcpy
and keep-vs-evict produce byte-identical logits.  Steady-state decode is
unchanged (mean `[q4] fwd n=1` 97-112 ms/token, within run noise) and 4k prefill
per-chunk times are unchanged.

### 7.3 Depth cells re-baselined, and the one-card experiment (2026-11-01)

Both cards, `--layer-map 0-19:gpu.0,20-47:gpu.1`, f16 KV, 28 threads, tail 7
(no longer the default),
`PF_PREFIX_CACHE=0`, one process per cell.  `pp` is prompt tokens over the
warm prefill (the `[q4] fwd n=<rows>` rows); `ms/step` is the mean of the
`[q4] fwd n=1` forward, skipping the first three steps while the bank warms.

| depth | prompt | pp | ms/step | tg | steps |
|---:|---:|---:|---:|---:|---:|
| 512 | 537 | 50.7 tok/s (10.6 s) | 93.6 | 10.68 tok/s | 23 |
| 16 037 | 16 037 | 48.9 tok/s (327.7 s) | 108.5 | 9.22 tok/s | 23 |
| 48 222 | 48 222 | 42.8 tok/s (1 126.8 s) | 108.5 | 9.21 tok/s | 11 |

The 48k prompt is a **repeated one-token word**: a 48k-token prose prompt is
~210 KB and a single argv entry is capped at 128 KB, so that cell's routing is
degenerate and its decode is not comparable to a document.  The 512 cell
reproduces in two consecutive runs (93.6 both); the prefill at 537 tokens
varies 49-51 tok/s and the same cell read 101.2 ms/step in a 31-step run, so
the bank's warm-up is the spread, not the engine.

**One card (the x16 slot).**  Hiding the x8 card and putting all 48 layers on
the x16 card (`ONEAPI_DEVICE_SELECTOR=level_zero:1`, `--layer-map 0-47:gpu.0`)
fits **5 250 MB of dense weights plus only a 4 000 MB bank = 28 slots/layer**
in 15.5 GiB; 5 000 MB fails to allocate, so VRAM is the constraint, not the
link.  The single card's H2D runs at 9.96 GB/s against the dual pair's ~5.17 on
half the traffic, but a quarter of the slots cancels it:

| depth | dual | one card (x16) |
|---|---:|---:|
| 512, 537 tok | **93.6 ms/step** | 101.6 (best of 3: 93.5) |
| 16 037 | 108.5 | **106.8** |
| 48 222 | **108.5** | does not fit (1.1 GB KV + bank) |

So the dual card stays the configuration to run: one card is a wash at 16k,
slower at 512, and cannot hold a 48k context.


## 8. Flags

| flag / env | default | meaning |
|---|---|---|
| `PF_MODEL_INFO=1` | off | print the resolved model profile (policy/weights/capabilities) and its provenance at load |
| `--moe-vram-mb MB` | 0 | per-expert LRU budget; 0 = all experts on the host |
| `--engram-ram-mb MB` | profile (4096) | PLE row cache in host RAM; unset = `policy.engram_ram_mb_default`, 0 disables |
| `--moe-ram-mb MB` | 0 | generic ncmoe host tier; not used by qwen4exp |
| `PF_Q4_BANK=0` / `PF_Q4_BANK_LAYER=1` | on / off | all-host path / whole-layer bank (A/B) |
| `PF_Q4_BANK_Q5K=1` | off | layer 2 bank, decode-only (48 layers, 107.2 ms vs 93.3, off) |
| `PF_Q4_PIN_SRC=0` | on | stage from the mmap instead of the pinned expert source |
| `PF_Q4_PREAD=0` / `PF_Q4_PREAD_WORKERS=N` | on / 8 | read the pinned expert source with parallel `pread` from each shard fd; `=0` restores the serial memcpy; 8 and 16 workers both measure the drive's ~1.15 GB/s ceiling |
| `PF_Q4_PREFAULT_ALL=1` | off | touch every tensor in the prefault again (the old behaviour); the default skips a banked layer's experts and the GPU-resident shared/router tensors |
| `PF_Q4_KEEPCACHE=0` | on (qwen4exp) | `page_out_host` drops the PTEs but keeps the shard page cache, so a restart is warm; `=0` restores the `POSIX_FADV_DONTNEED` eviction |
| `PF_Q4_TAIL_SKIP=N` | 0 | drop a miss whose best batch rank is >= N (default 0 = exact; 1..9 meaningful; `--q4-tail-skip` overrides; the 86.9 ms, 32.2 MiB vs 93.6/56.2 and the 2026-10-27 gate - KL 0.029/0.037 at near-tie divergences - were measured at N=7) |
| `--q4-tail-skip N` | 0 | CLI form of the above: default 0 = exact, 1..9 meaningful, env `PF_Q4_TAIL_SKIP` |
| `PF_Q4_CPU_MISS=1` | off | decode CPU miss for non-resident experts, int8 host path (`2` = f32 A/B); **148.1 vs 99.3 ms, rejected** (§6.1) |
| `PF_Q4_CPU_MISS_WARM=N` / `_LAYER=il` / `_TIME` | 16 / all / off | decode steps with the LRU before the split freezes / one-layer probe / per-layer host+GPU print |
| `PF_Q4_CMDBG=1` | off | per-call sub-phase ms inside the host miss expert |
| `PF_Q4_HEAT=1` | off | decayed-heat victim (105.7 ms / 156 MiB; slower, A/B only) |
| `PF_Q4_PF_ONEPASS=1` / `PF_Q4_PF_STATS=1` | off | old single pass / two-pass stats |
| `PF_Q4_PF_PIPE=0` | on | serial copies (A/B for the two-pass prefetch) |
| `PF_Q4_PF_WMIN` | profile `moe.pf_wmin` (24) | minimum two-pass workspace; `-1` = profile |
| `PF_Q4_LRU_OLD=1` | off | pre-fix eviction (A/B only, not an LRU) |
| `PF_Q4_STAGE=0` | on | pageable expert uploads instead of the pinned ring (ring = profile `moe.upload_ring`, 32) |
| `PF_Q4_INJECT=0` / `PF_Q4_GEMV8=0` / `PF_Q4_GEMVSEG=0` | on | generic HC inject / Q8_0 GEMV / blocking segment copy |
| `PF_Q4_F32_SPLIT=0` | profile `split.max` (8) | F32 GDN beta/alpha GEMV; `-1` = profile, 0 = off |
| `PF_MOE_CFG` / `_DOWN` / `_DEC` | profile `moe.cfg_*` (43/43/0) | grouped expert GEMM tile case; `_DEC` 0 = auto |
| `PF_MOE_FLAT` / `_DEC` | profile `moe.flat`/`flat_dec` (1/0) | row tile in the grid |
| `PF_MOE_DP4A` / `_DEC` | profile `moe.dp4a`/`dp4a_dec` (1/1) | grouped dp4a expert GEMM |
| `PF_QSA_TIME=1` | off | QSA host split per forward (`copy/gemv/upd/sel/attn`) |
| `PF_QSA_ISA=scalar\|avx2\|avx512` | auto | force only the QSA host kernels' ISA |
| `PF_QSA_DEFER=0` | on | inline update instead of the end-of-forward deferred update |
| `PF_QSA_F16MIR=0` | on (AVX2/AVX-512) | f32 host K/V mirror; scalar defaults to f32 (f16 scalar is 4.6x slower) |
| `PF_QSA_COPYQ=0` | on | inline dense-big-chunk K/V wait instead of the per-device copy queue |
| `PF_QSA_DEVATTN=1` | on | Stage 2 device sparse attention for f16 GPU decode; host selector unchanged; `=0` host Stage 1; see §7.1 |
| `PF_Q4_HIST=1`, `PF_Q4_TIME`, `PF_Q4_PHASE`, `PF_Q4_HCTIME`, `PF_Q4_HOT_TIME` | off | routing histogram / phase attribution |
| `PF_Q4_PREFAULT=0` / `PF_Q4_NOPLE=1` / `PF_Q4_FFCMP=<layer>` | on / off / off | skip prefault / skip PLE / routed-FFN host oracle |

| `PF_Q4_PERDEV_R=0` | on | per-device bank slots/layer (device 0's share otherwise sits unused) |
| `PF_Q4_DOWN_PREF=0` | on | issue the down plane last and wait for it just before `down`; `0` = one wait for all three planes |
| `PF_Q4_DOWN_I8_ALL=1` | off | build the routed-down int8 store for every layer (42.2 GiB / ~110 s) instead of only the non-banked ones |
| `PF_Q4_Q5NPB=8` | 1 | rows per warp for `moe_down_q5_1_launch`; 1 and 8 are the only instantiations |
| `PF_Q4_Q5_LANE=1` | off | one lane per 32-value block in that kernel (+1.15-1.25x, **not bit-identical**) |
| `PF_Q4_SHFUSE=0` | on | unfused shared expert (quantize + one gemm per segment, 5 launches) instead of the fused 2-launch path |
| `PF_Q4_SHOVER=0` | on | run the shared GEMVs after `moe_reduce` instead of under the routed-expert H2D wait |
| `PF_Q4_SHPREF=0` | on | plain strided `moe_gate_add` loops instead of the 8-pair register prefetch |
| `PF_Q4_PLE_SER_MAX` | 65536 | run a `cpu_q4_ple_apply` region on the calling thread when its WORK is below this |
| `PF_Q4_PLEGPU=1` | off | device PLE projections (-1.60 ms/token, 183 GB/s; **not byte-identical**) |
| `PF_MOE_RT_FAST=0` | on | batched SLM loads in the fused router selection (bit-identical, -0.22 ms/token) |
| `PF_Q4_KTIME=1` / `PF_Q4_SEGTIME=1` / `PF_Q4_SHTIME=1` | off | device-side kernel/segment timing from barrier pairs |
| `PF_Q4_PLEFAULT=1` / `PF_Q4_PLEAPPLY_TIME=1` / `PF_Q4_PLEGPU_TIME=1` | off | PLE stage splits |
| `PF_MOE_RT_ABL` / `PF_Q4_ABL_NOGEMV` / `PF_Q4_ABL_NOMIX` | 0 | phase ablations for attribution; **wrong output by design** |

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
* **Bigger banks and other transports.**  20000 MB fails to load; 18000 loads
  (125 slots, 17.0 GiB) but is **not** faster than 17500, and 16000 gives 111
  slots / 101.2 ms against 17500's 121 / 93.3 ms - the 17 500/121-slot point is
  the knee.  Parallel staging delays the first DMA (the floor is 170.5 MiB at
  5.9 GB/s), a pinned mirror cannot beat the same H2D, LFU is neutral at 72
  tokens and worse at 200, and a preloaded slot map is churned by the prefill.
  Native Q5_1 down on the host is slower than the generic path.
* **Layer 2 Q5_K banking** (`PF_Q4_BANK_Q5K=1`) is a decode loss (107.2 vs
  93.3 ms) and stays off; the four Q8_0-down layers 4/30/46/47 **are** banked
  through `moe_down_q8_0_launch`, and the native Q5_1/Q8_0 kernels mean those
  layers do not use the generic int8 conversion.
* **2-row batched decode** is 0.85-0.88x (208.1 vs 176.5 ms/step) because the
  routed-expert weight read doubles; the scheduler caps q4 at one row and
  serialises its sequences.  Round-robin (the first attempt at fairness) is
  worse (4.33 vs 5.40 tok/s aggregate) because two live sequences thrash the
  shared bank.
* **MTP** is no-go for this build: the existing one-slot mode-1 forward costs
  257.8/416.4/496.8 ms at n=1/3/5 against 190/230 kill thresholds.  A new
  one-slot k+1-row verify mode would be required, and the 2-row result makes
  the likely 3/5-row cost too high to repay.
* **A 2-chunk prefill layer pipeline** adds ~45% expert copies (+14.5 GiB) and
  lands at ~9.7 s, so it does not reach 7.35 s; a layer-map rebalance drops R
  to ~103 and costs ~10% decode.  The 74 tok/s target is out of reach on two
  cards.
* **Tail-skip** works (120.7 -> 100.1 ms decode on the 96-token TCP prompt,
  KL 0.029-0.037 at near-tie divergences; the numbers were taken at N=7).  It
  is an expert-drop but the default is now **0 (exact)**;
  `--q4-tail-skip N` / `PF_Q4_TAIL_SKIP=N` opts in; the gate is in
  §7.1.

### 9.1 Dead ends from the 2026-11-01 pass

* **A native 4-bit expert bank.**  Designed, measured, rejected.  The u4 store
  is `0.625 B/w`, so gate+up+down is **3,072,000 B/expert - byte-identical to
  the raw GGUF** (`2 x 0.5625 + 0.75`): *no* extra slots, and it is lossy
  (~4.4 % rel-L2) where the raw Q4_K/Q5_1 planes are exact.  A lossless
  `Q5_1 -> k5t` repack is byte-neutral (0.75 B/w) and would only buy the
  kernel's load shape.  Do not re-derive it.
* **The Q5_1 down's 64-73 GB/s is the 24-byte block's load shape.**  Cold
  sweep (`dev/bench_q5cold.cpp`: 256-slot / 300 MB bank, 25 disjoint 10-slot
  subsets, barrier-pair device time): pure `uint4` read **363.5 GB/s** (the
  ceiling), Q4_K gate/up grouped **171**, Q4_K down grouped **178**, Q5_1
  one-lane **98-101**, Q5_1 SG8 NPB=1 **81-85**, SG8 NPB=8 **73.8**, SLM
  row-staged SG8 **52** (slower - the per-row barrier costs more than the
  coalescing buys).  A Q4_K 144-byte superblock feeds a warp 128 B of
  contiguous `qs` per instruction; Q5_1's 16-byte `qs` per block either needs 8
  lanes with duplicate words (SG8) or a 20/32-lane gather (one-lane).  The slot
  scatter is a no-op (73.8 adjacent = spread), the down-plane DMA wait is 3 %
  of the composite `dn`, and the ratio is preserved AOT vs JIT.
* **The byte-neutral 128-value panel repack was implemented, is bit-exact, and
  is SLOWER (`PF_Q4_Q5SB`, default off; measured 2026-11-03).**  The hypothesis
  above - that a superblock repack would reach the grouped kernel's rate - was
  tested rather than assumed.  `q5_1_sb_pack` rewrites each 640-wide down row
  (20 x 24 B blocks) into 5 x 96 B panels (qs[64] in the Q4_K split-half nibble
  order, qh[16], meta[16]) and `moe_down_q5_1_sb_launch` reads it with the
  *same* lane/group/pair map and FMA order.  It is byte-neutral (0.75 B/w, no
  extra GiB), the 43 repacked layers cost 25.2 GiB and 3.4-4.1 s of startup,
  and the result is **bit-identical**: 96 greedy steps produced the same text
  md5 and the same per-step `PF_DUMP_DEC_LOGITS` md5, `PF_Q4_Q5NPB` 1 and 8 are
  identical, and prefill is unchanged (70.35 vs 70.51 s on the 3.2 k-token
  prompt).  **But the down kernel is 0.1797 -> 0.2073 ms/layer (median, 4 128
  samples/layer), i.e. ~70 -> ~61 GB/s.**  The repack cannot change the `qs`
  issue shape: bit-identity fixes one 32-value group per lane and 4 B of `qs`
  per load, so the warp still issues the same four 4-byte loads per group
  slice; what does change is `qh`/`meta`, which the 24-byte block layout served
  as per-block broadcasts and the panel serves as wider, repeatedly-read
  slices.  The form that actually reaches ~100 GB/s is the 16-byte-per-lane
  `moe_row_gemv_impl` (measured 56 -> 100 GB/s), and that necessarily changes
  the reduction/decomposition - not bit-identical.  So the repack is kept as
  the opt-in A/B (`PF_Q4_Q5SB=1`; default 0, `--q4-tail-skip` and the bank
  budget remain the levers), and **the earlier "only route" claim above is
  superseded**.
* **The shared expert must stay on oneDNN at M=1.**  Replacing its three Q8_0
  matmuls with the raw GEMV loses **1.3-1.8 ms/token**: oneDNN is already at
  86-88 GiB/s at this shape, the raw gate/up matches it (91) but the swiglu
  **down** falls to the generic `lane=k` scalar dot at **33 GiB/s** because
  K=640 is not a multiple of 256.  **The launch count, not the math, is what
  costs at M=1, and it is now measured** (`PF_Q4_SHTIME`, barrier pairs over
  48 layers on the §7.2 cell): the shared GEMVs are **5.13 ms/token** fused
  (`PF_Q4_SHFUSE`: `gemm_i8_quant_group` does gate+up in one launch and down
  in one, bit-identical to quantize + one gemm per segment) against **6.74
  ms** unfused, and `moe_gate_add` is **1.64 ms** with `PF_Q4_SHPREF`
  (prefetch eight gate/mixed and shared/out pairs, same add order) against
  **3.67 ms** without.  The earlier "6.3 ms shared FFN, ~3.6 ms of it
  `moe_gate_add`" was the pre-prefetch number and its 3.6 ms was mostly the
  launch gap.  `PF_Q4_SHOVER` submits the shared GEMVs before the routed
  expert's `pre_ev.wait()` and leaves the gate add after `moe_reduce` (decode
  M=1 only).  All three knobs default on, are bit-identical (96-step
  `PF_DUMP_DEC_LOGITS` md5 and the prefill logits match with each on and
  off), and together are worth **2.85 ms/token** (89.67 vs 92.52).
* **Router restructurings that lost** (all bit-identical): SLM max merge
  +65 %, 64-bit retire mask +29 %, full-lane rank scan +12 %.  The GEMV is
  already fully split over N (512 work-groups, one warp/EU) at 286 GB/s, so
  there is nothing to split; a K-split would flip near-ties for ~5.6 us/layer.
  The remaining 39 us/layer is one work-group's serial selection on a 2 KB
  problem - eliminating it entirely is worth only ~1.3 ms/token.
* **Increasing the engram cache.**  `hit ~= cap/rows` exactly, the gather is
  0.075 ms, and the table is 28.8 GB; `PF_Q4_PLE_PREFAULT` costs 26.8 GiB to
  save <= 0.08 ms.
