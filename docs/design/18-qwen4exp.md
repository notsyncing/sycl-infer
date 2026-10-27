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
item: **39.3 ms scalar, 19.2 AVX2, 16.0 AVX-512** per token over the 12
full-attention layers (2.45x), and the prefill chunk that crosses the budget
goes 208.0 -> 68.2 ms of attention.  The selector (512 pools) is serial on
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
byte-identical.  The 93.3 ms `--ctx 2048` cell is unchanged (QSA off).  A
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
flows never do; `zero_slot` clears); `head_dim != 256`, i8, CPU and prefill use
the host Stage 1 path; the first device token allocates ~17 MB/device (~9.8 ms).
Default on for f16 GPU sparse decode on prompt->decode flows; `PF_QSA_DEVATTN=0`
keeps the host Stage 1 path.

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
Q5_K/Q8_0 0.99995).  The profile/device/chunker changes add three hermetic
checks: `test_model_profile` (registry x exact profile keys and the qwen4exp
values), `test_mm_step` (the shared M-RoPE/image-row slice) and the extended
`test_dev_profile` (`moe.*`/`shape.*` on the A770, the Iris inheritance and the
unknown fallback).

### 7.1 Batching, tail-skip, MTP and the prefill ceiling (2026-10-27)

Full tables and commands are in [`../PERFORMANCE.md`](../PERFORMANCE.md) §4.4.2.

* **2-row decode:** 208.1 ms/step against 176.5 for the same two sequences one
  at a time (**0.85x**; run 1 0.881x).  Both rows are real and match their own
  baseline (top-1 12/12, KL 0.026/0.008), so the path is correct; it loses
  because two prompts select ~20 distinct experts per layer against ~10 each
  and the routed-expert weight read doubles.  The batch uploads 2.7x fewer
  MiB/token, so it is not H2D-bound.  The scheduler now caps qwen4exp at one
  row and **serialises** its sequences; round-robin was 4.33 tok/s aggregate
  against 5.40 serialized (the shared bank thrashes), and the one-row cap is
  the shipped policy.  `test_q4_batch` is the regression.
* **28 CPU threads is a free win.**  The default is the physical-core count
  (14 here), but the host MoE/PLE/HC work scales with the logical count:
  decode mean 120.7 -> 109.6 ms, wall 20.95 -> 18.08 s, output md5 identical;
  with tail7, 100.1 -> 93.3 ms and 16.49 -> 15.52 s.  The recommended q4 run
  sets `--cpu-threads 28`; the server gains much less (5.40 -> 5.57 tok/s)
  because it is prefill/HTTP-bound.  The archive 93.3 ms cell is default-14.
* **Tail-skip:** the gate is in §4.4.2.  Two prompts, top-1 28/29 and 15/16,
  first divergence always a near-tie, KL 0.029/0.037, coherent output; decode
  mean 120.7 -> 100.1 ms.  It is **default 7**; `PF_Q4_TAIL_SKIP=0` restores
  the exact set; it is a real expert-drop, which the performance-first decision
  accepts.
* **MTP verify:** `test_q4_verify` times the existing one-slot mode-1 suffix on
  a warm bank: n=1/3/5 = 257.8/416.4/496.8 ms against the kill thresholds
  190/230.  Even n=1 is 2.7x a plain token, so the existing forward is not the
  verify shape; MTP is **no-go** for this build without a new one-slot
  k+1-row mode (and the 2-row batch result makes that unlikely to pay).

> **MTP q4 result: PENDING.**  <!-- MTP-Q4-PLACEHOLDER: the parent fills this
> after the MTP q4 job.  The bullet above is the pre-MTP-job conclusion and
> must not be read as the final ranking. -->
> Replace this note with the measured one-slot k+1-row verify result when the
> MTP job reports; until then the numbers in §7.1 and §9 are the old forward's.

* **Prefill:** the 9.6 s warm 544-token prefill is ~4.3-4.9 s upload (31-32 GiB
  at 6.9-7.9 GB/s), ~3.5-4.2 s device MoE and ~1.1 s non-MoE.  Host-spilling
  the overflow experts has a ~0.25 ms/expert break-even against the existing
  0.32-2.4 ms/expert path; a 2-chunk layer pipeline adds ~45% copies (+14.5 GiB)
  and lands at ~9.7 s; a layer-map rebalance drops R to ~103 and costs ~10%
  decode.  **74 tok/s is not reachable on two cards**; the warm prefill stays
  ~56 tok/s.  The q4 server now uses the same whole-prompt `prefill_text`:
  2412 tokens in 41.7 s (57.8 tok/s) one chunk, against 92.6 s for the old
  32-token `prefill_chunk` workaround (2.2x), output md5/ids matching the CLI.

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
| `PF_Q4_TAIL_SKIP=N` | 7 | drop a miss whose best batch rank is >= N (7 = default; `0` = exact; 86.9 ms, 32.2 MiB vs 93.6/56.2; gate 2026-10-27: KL 0.029/0.037 at near-tie divergences) |
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
  KL 0.029-0.037 at near-tie divergences).  It is an expert-drop but is now
  **default 7**; `PF_Q4_TAIL_SKIP=0` restores the exact set; the gate is in
  §7.1/PERFORMANCE §4.4.2.
