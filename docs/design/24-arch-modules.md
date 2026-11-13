# Design 24: architecture modules, registries and the boundary gate

This document is the contract between the two halves of the code base:

* `src/model/` — **what the file is**: the GGUF reader, the loader, the
  architecture's tensor views, its extra hyper-parameters and its declarative
  policy.
* `src/engine/` — **how it runs**: buffers, plans, graphs, the KV pool, the
  prefix cache, the request path, and (for an architecture that owns its
  forward) the forward itself.

The only dependency arrow is `engine -> model`.  A model file, a loader or a
model profile never includes `engine.h`; a generic engine file never includes an
architecture header and never names an architecture.  The direction is enforced
by the gate described in §8, not by convention.

Related documents: [01-model-loading.md](01-model-loading.md) (loading and the
profile), [04-engine.md](04-engine.md) (`seg_plan` / `record_forward`),
[12-build-and-testing.md](12-build-and-testing.md) (test registration).

---

## 1. One architecture, one directory in each layer

```
src/model/arch/<arch>/
    load.cpp        si::arch::load_<arch>(model &)          # required
    load.h          load_<arch>() + <arch>_profile()        # required
    profile.cpp     file-local model_profile + <arch>_profile()  # required
    hparams.h       <arch>_hparams : arch_hparams           # model-owned only
    tensors.h       <arch>_layer / <arch>_model_extras      # model-owned only
    accessors.h     <arch>_hp / <arch>_layer_of / <arch>_model  # model-owned only
```

```
src/model/arch/registry.h       public: arch::loader + find/registry, no arch names
src/model/arch/registry.cpp     internal hub: kLoaders[] + find/registry, includes
                                each arch/<arch>/load.h
src/model/arch_util.h           bind_tensor/bind_f32 declarations + bind_moe/require_meta
```

```
src/engine/arch/<arch>/
    runtime.cpp     the arch_ops_registry hooks (registry_internal.h)  # required for a row
    forward.cpp     (or forward_host.cpp / forward_gpu.cpp)  # model-owned only
    state.h         the composed runtime state               # model-owned only
    qsa_state.h     a sub-state composed into state.h        # optional
    dense_store.cpp / prefix_cache.h / forward.h             # as needed
```

* A **generic-plan** architecture (today `qwen35`, `qwen35moe`, `qwen2vl`) has
  only the model side.  Its profile says `fwd = generic_plan`, the engine
  resolves no `arch_ops` row, and the shared `seg_plan` / `record_forward`
  machinery runs it.  It does not need a `src/engine/arch/<arch>/` directory.
* A **model-owned** architecture (today `qwen4exp`, `deepseek41`) has both
  sides.  Its profile names a model-owned `fwd_kind`, the engine resolves the
  matching `arch_ops` row, and `engine/arch/<arch>/` owns everything specific to
  that forward: the entry points, the state, and the hooks.
* `qwen35` and `qwen35moe` are two model modules with separate `kLoaders[]`
  rows and separate profile consts even though their engine behaviour is the
  same.  The registry key is exact, so they cannot share one "generic" row.

The state composition rule is one-way: `engine` holds the per-architecture state
members (`q4x_`, `ds41x_`) and starts the constructor's composition, but only
`engine/arch/<arch>/` code reads their fields.  A generic engine translation
unit reaches them through an `arch_ops` hook, never by name.

---

## 2. The two registries

### 2.1 `kLoaders[]` — the model side

`src/model/arch/registry.h` is the **public** seam and declares `arch::loader`:

```cpp
struct loader {
    const char * name;            // exact general.architecture value
    void (*load)(model &);        // fill hparams + bind tensors
    const model_profile * prof;   // key must equal name
};
```

It names no architecture.  `src/model/arch/registry.cpp` is the **internal hub**:
it owns the explicit `kLoaders[]` table, `arch::find` and `arch::registry`, and
includes every `arch/<arch>/load.h` (each of which declares that loader and its
`<arch>_profile()` accessor).  The table is explicit rather than self-registered
because the core is a static library, so a registration from an unreferenced
object would be dropped at link time.  A row without a profile, a profile whose
`key` differs from the GGUF's `general.architecture`, or an unknown architecture
is a load-time error (`unsupported architecture: <name>`,
`architecture ... has no model profile`, `model profile key mismatch: ...`).
There is deliberately no fuzzy match and no `PF_MODEL_PROFILE` pin: the file
states its own architecture.

The shared binding helpers (`arch::bind_tensor`, `arch::bind_f32` — definitions
in `model.cpp` — plus the header-only `bind_moe` / `require_meta`) live in
`src/model/arch_util.h`, so a loader includes `load.h` + `arch_util.h` and never
the registry table.

### 2.2 `kArchOps[]` — the engine side

`src/engine/arch/arch_ops.h` declares `struct arch_ops`: a row of function
pointers and capability bits, selected once per engine from
`model_profile::fwd`.  `arch_ops_for()` returns `nullptr` for `generic_plan` and
for an unknown kind, and **null is the selector** for "the shared generic engine
owns this" (there is no `generic_plan` field on the struct).

The seam is split into a public and an internal half:

* **`src/engine/arch/arch_ops.h` is public** (`engine.h` includes it): only
  `struct arch_ops`, `arch_ops_for()` and a forward declaration of
  `arch_ops_registry`.  It names no architecture, and
  `ci/check_arch_boundaries.py` scans it like the other generic files.
* **`src/engine/arch/registry_internal.h` is internal**: it defines
  `arch_ops_registry`, whose static members are the per-arch hook
  implementations.  Only `registry.cpp` and the per-arch `runtime.cpp` include
  it, which the gate enforces; the public header must never include it.

The row groups:

| group | examples | null means |
|---|---|---|
| lifecycle | `configure`, `setup`, `prefault`, `setup_weights`, `destroy`, `reset_extra`, `slot_zero`, `hist_report` | generic lifecycle |
| capabilities | `owns_kv_pool`, `owns_request_path`, `md_dnnl`, `device_argmax`, `host_keep_routed_experts`, `dense_store` | the generic behaviour |
| request path | `generate` | `generate_impl` runs the generic path |
| forward | `prefill_text`, `prefill_mm`, `decode_batch`, `run_head` | the generic `seg_plan` |
| prefix cache | `pc_extra_floats`, `pc_serialize`, `pc_deserialize`, `pc_restore`, `pc_clear`, `pc_resume_prefixes`, `pc_block_aligned_capture` | no architecture-owned checkpoint tail |

Hooks are implemented as static members of `arch_ops_registry` (declared in
`registry_internal.h`) in `src/engine/arch/<arch>/runtime.cpp` (which is a
friend of `engine`), and the row in `src/engine/arch/registry.cpp` binds them.
A generic file tests the hook, never the model name or the forward kind:

```cpp
bool owns_kv_pool() const       { return aops_ && aops_->owns_kv_pool; }
bool arch_dense_store() const   { return aops_ && aops_->dense_store && aops_->dense_store(*this); }
```

---

## 3. The data model

### 3.1 `arch_hparams` — the architecture's geometry

The generic `hparams` (`src/model/model.h`) carries only facts every
architecture shares.  A model-owned architecture attaches one derived
`arch_hparams` through `hparams::arch`; the generic struct carries no
architecture-specific field.

```cpp
struct arch_hparams {
    enum class kind { none, qwen4exp, deepseek41 };
    kind k;
    virtual bool wide_head_ok() const;                                  // default false
    virtual void validate(const hparams &, const std::string & prefix) const;
    virtual void validate_profile(const hparams &, const model_profile &) const;
};
```

* `wide_head_ok()` lets an architecture whose attention head is wider than the
  classic kernel's 256-value register ceiling opt in (ds41's 512-wide MLA).
* `validate()` is the architecture's own geometry check; it runs after the
  generic `validate_hparams`, with a `"<arch>: "` message prefix.
* `validate_profile()` cross-checks the loader's GGUF-derived geometry against
  the declarative profile, so a file that contradicts its architecture's policy
  is refused at load time.
* The accessor (`q4_hp`, `ds41_hp`) asserts the runtime `kind` tag, so a
  wrong-kind cast fails loudly rather than reading another layout.

### 3.2 `arch_layer` — per-layer extra weights

`layer_t::extras` is a `std::unique_ptr<arch_layer>` (null for a generic
architecture).  The only thing the generic upload and host-keep loops call is

```cpp
virtual void visit_weights(const hparams & hp, int il,
                           const std::function<void(const void *)> & cb) const;
```

so a generic file never names a per-architecture tensor.  The architecture's
own code reads fields through `q4_layer(L)` / `ds41_layer_of(L)`.

### 3.3 `arch_model_extras` — model-level extras

`model::extras` is a `std::unique_ptr<arch_model_extras>`.  It separates the
device set from the host-keep set:

```cpp
virtual void visit_device_weights(const std::function<void(const void *)> & cb) const;
virtual void visit_host_keep(const std::function<void(const void *)> & cb) const;
```

qwen4exp uploads the HC head and keeps the 26.8 GiB PLE table host-resident;
deepseek41 uploads none of these and gathers the Engram table on the host.

### 3.4 The rule

**A generic file never names an architecture.**  `engine.cpp`, `graph.cpp`,
`engine_prefix_cache.cpp`, `engine_kvpool.cpp`, `server.cpp`, `model.cpp` and
`model.h` reach an architecture only through:

* a `model_profile` field (declarative policy),
* an `arch_ops` hook (runtime seam),
* an `arch_hparams` / `arch_layer` / `arch_model_extras` virtual, or
* the per-architecture accessor, called from that architecture's own code.

If a generic file needs a new architecture fact, add it to one of those four
places in the same change.

---

## 4. `model_profile`: the declarative policy

`src/model/model_profile.h` is the declaration; each architecture's
`profile.cpp` is its complete const.  It says what the model **is** and what it
**may use**:

| group | fields |
|---|---|
| identity | `key`, `name`, `provenance` |
| forward | `fwd` (`generic_plan` / `hc_ple_qsa` / `deepseek41`), `vision` |
| geometry | `residual` (HC streams / low rank), `extras` (PLE, QSA) |
| policy | `max_decode_rows`, `prefill`, `prefix_cache`, `auto_layer_map`, `md_splits`, `engram_ram_mb_default`, `keep_file_cache` |
| representation | `rep.sin_w8`, `rep.dnnl_int8`, `rep.raw_gemv_decode`, `rep.split_per_tensor_upload` |
| capabilities | `spec.mtp` / `dflash` / `multimodal` + `why`, `spec.mm_owned_prefill` |
| experts | `moe` owner (engine cache / runtime bank / host only) |

`spec.mm_owned_prefill` is a forward capability, not a runtime hook: it says a
media prompt must go through the model's own chunker instead of the generic
one-row-at-a-time loop (true for qwen4exp).

The profile is **not** the runtime seam.  The profile is read by name at a few
policy sites; the engine's per-request dispatch (forward entry points, state
reset, prefix-cache extras, capabilities such as `device_argmax` and
`dense_store`) is the `arch_ops` row.  That split is deliberate: a new policy
value does not need a function pointer, and a new hook does not need a profile
field.

Per-card tuning never belongs in either: splits, work-group widths and tiles are
in the device profile (`src/device/profiles/<card>.cpp`).  Neither do the
properties of a specific file or run: expert plane sizes, bank budgets, KV type,
layer map, token-batch buckets and prefill chunk size.

---

## 5. Adding a generic-plan architecture

1. `src/model/arch/<arch>/load.cpp` — `si::arch::load_<arch>(model &)` fills
   `m.hp` and binds the layer tensors with `arch::bind_tensor` / `arch::bind_f32`
   from `arch_util.h`.
2. `src/model/arch/<arch>/profile.cpp` — one complete `model_profile` const
   (file-local) plus `<arch>_profile()`, with `fwd = generic_plan`, `key`
   exactly the GGUF's `general.architecture`.  `src/model/arch/<arch>/load.h`
   declares the loader and that accessor.
3. Include the new `arch/<arch>/load.h` in **`src/model/arch/registry.cpp`** and
   add one row to `kLoaders[]` there.  The public `arch/registry.h` and
   `model_profile.h` do not change.
4. Add the two `.cpp` files to `CMakeLists.txt`.  No engine file changes and no
   `src/engine/arch/<arch>/` directory: `arch_ops_for(generic_plan)` is null.
5. If a fact differs from the generic path, add a named field to
   `model_profile`, `hparams`, or (for geometry) an `arch_hparams` — never a
   ternary at the use site.
6. Pin the profile in `tests/model/test_model_profile.cpp`.

`qwen2vl` is the example: a generic-plan architecture whose only engine-visible
differences are profile fields (`auto_layer_map`, representation) and
loader-filled `hparams`.

## 6. Adding a model-owned forward architecture

Do everything in §5, then:

7. `src/model/arch/<arch>/hparams.h` — `<arch>_hparams : arch_hparams` with
   `wide_head_ok` / `validate` / `validate_profile`, plus an accessor that
   asserts the `kind` tag.  The loader attaches it as `hparams::arch`.
8. `src/model/arch/<arch>/tensors.h` — `<arch>_layer : arch_layer` and
   `<arch>_model_extras : arch_model_extras`; implement `visit_weights`,
   `visit_device_weights`, `visit_host_keep`.
9. `src/model/arch/<arch>/accessors.h` — the checked accessors the
   architecture's code uses.
10. `src/engine/arch/<arch>/state.h` (and any sub-state such as
    `qsa_state.h`) — the composed runtime state, included by `engine.h` only.
11. `src/engine/arch/<arch>/forward*.cpp` — the forward entry points; declare
    them in a local `forward.h`, never in `engine.h`.
12. `src/engine/arch/<arch>/runtime.cpp` — the `arch_ops_registry` static
    hooks.  Add the `kArchOps[]` row in `src/engine/arch/registry.cpp` with the
    capabilities and every hook the forward needs; leave a hook null when the
    generic engine owns it.
13. Add every new `.cpp` to `CMakeLists.txt`, extend
    `tests/engine/test_arch_ops.cpp` to pin the new row, and keep
    `python3 ci/check_arch_boundaries.py` clean.

The `arch_ops` row is the *only* place the engine learns that the forward is
different.  A new model-owned architecture must not add an `if (fwd == ...)` to
a generic file.

---

## 7. Where each fact lives

| fact | home | example |
|---|---|---|
| file/geometry the loader reads | loader-filled `hparams` / `arch_hparams` | `mrope_interleaved`, ds41 `kv_src_of` |
| architecture policy and capabilities | `model_profile` | `max_decode_rows`, `spec.mm_owned_prefill` |
| per-card tuning | device profile | `shape.rmsnorm_wg`, `moe.cfg_gate` |
| runtime seam / architecture-owned state | `arch_ops` + `engine/arch/<arch>/` | `dense_store`, `pc_extra_floats` |
| per-layer / model-level extra weights | `arch_layer` / `arch_model_extras` | qwen4exp HC/PLE, ds41 Engram |

---

## 8. Enforcement

Two hermetic checks run in the default `ctest -L hermetic` set:

* **`test_arch_ops`** (`tests/engine/test_arch_ops.cpp`) pins the registry
  shape without calling a hook: `arch_ops_for(generic_plan) == nullptr`; every
  qwen4exp hook is bound and `hist_report == nullptr` (q4's teardown prints the
  routing diagnostic itself, so the generic post-destroy call must not print it
  twice); ds41 declares `owns_kv_pool`, `owns_request_path`,
  `host_keep_routed_experts`, `!md_dnnl`, `generate/setup/setup_weights/destroy`
  and `dense_store != nullptr`.
* **`arch_boundaries`** (`ci/check_arch_boundaries.py`) scans the ten generic
  files — `engine.cpp`, `graph.cpp`, `engine_prefix_cache.cpp`,
  `engine_kvpool.cpp`, `server.cpp`, `model.cpp`, `model.h`,
  `model/arch/registry.h`, `model/arch_util.h`, `engine/arch/arch_ops.h` —
  after stripping comments and string/char literals (so a registry key or a doc
  comment is not a hit).  It fails on an architecture name, `q4_*` / `ds41_*`
  identifiers, the per-model state members `q4x_` / `ds41x_`, a model-owned
  `model_profile::fwd_kind::(hc_ple_qsa|deepseek41)` branch, `arch_profile`, an
  `#include "arch/<name>/..."` or `"engine/arch/..."` (only the public
  `arch/registry.h` is allowed), and (for `model.h`) the architecture field
  names.

```bash
python3 ci/check_arch_boundaries.py     # arch_boundaries: OK (10 generic files, ...)
./build/test_arch_ops                   # pins the rows
```

A violation is a design error, not a style nit: it means generic code can no
longer be read or tested without the architecture it names.  The gate is
deliberately small (seven files) and deliberately literal — comments and strings
are erased first, so the checker cannot be satisfied by prose.
