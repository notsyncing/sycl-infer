#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <sycl/sycl.hpp>

#include "kernels.h"

namespace si {

struct layer_t; // model.h; only a pointer is held here

// QSA runtime state: the host-side indexer cache, the deferred per-layer
// update, and the Stage-1/2/3 device scratch.  The engine's QSA member
// functions drive it; the data lives here so the model-owned forward owns it.
struct qsa_state {
    // Allocated lazily per (attention layer, slot).
    struct slot_state {
        // Only the selector's indexer pool/ring is per-sequence host state.  The
        // main K/V is read straight from the paged pool (and dequantized) by the
        // host sparse attention, so there is no host K/V mirror: it used to cost
        // max_seq*n_head_kv*head_dim*2 bytes per attention layer per slot (48 MB
        // at 4k, 3.2 GB at 256k) and made a prefix-cache resume impossible for a
        // quantized KV type (the mirror holds pre-quantization values).
        std::vector<float> pool; // [(max_seq/kp+1)][idx_head_dim]
        std::vector<float> ring; // [kp][idx_head_dim]
        int n_pool = 0;
    };
    std::vector<std::vector<std::unique_ptr<slot_state>>> cache;
    bool on = false;
    int max_seq = 0;
    // Deferred QSA update (decode while the selector is dense-equivalent).
    // The D2H copies are enqueued per layer and waited once at the end of the
    // forward, so the host MoE and the device no longer serialise 12 times per
    // token; the indexer projection and the mirror/pool update then run while
    // the LM head is on the device.  Only the per-token update is deferred -
    // the selector and the sparse attention must stay inline before `wo`.
    struct pending_t {
        const layer_t * L = nullptr;
        int a = -1;
        int n_tok = 0;
        int dev = -1;
        bool active = false;
        bool has_ev = false;
        sycl::event ev;
        std::vector<int> pos, slot;
        // The forward's step_info, for the prefix cache's per-block checkpoints
        // (the deferred update runs after the head, inside the same forward).
        const step_info * inf = nullptr;
        float * mixed = nullptr; // per-layer pinned staging
    };
    std::vector<pending_t> pending;
    // PF_QSA_EARLY (default on): enqueue the per-layer `mixed` D2H right after
    // hc_mix, i.e. before the QKV GEMVs, and consume its event in
    // qsa_attention instead of draining the in-order queue there.  `mixed` is
    // produced by hc_mix and the QKV/rope chain never touches it, so this is a
    // scheduling change only: the copy source/destination and the host update
    // read the same bytes.  `PF_QSA_EARLY=0` restores the old placement (copy
    // inside qsa_attention) and the blocking `sel` H2D wait for the A/B.
    bool early = true;
    struct early_copy {
        bool valid = false;
        int dev = -1;
        int kind = 0; // 0 = `mixed` D2H, 1 = device-indexer iq/ik D2H
        sycl::event ev;
    };
    std::vector<early_copy> early_ev;
    // PF_QSA_IDX_DEV (default on after validation): project the indexer on the
    // device from the live `mixed` activation and D2H only the 2.5 KB the host
    // selector/pool update reads.  Falls back to qsa_early_mixed() for every
    // shape it does not serve (non-BF16 indexer, CPU, dense-equivalent token).
    bool idx_dev = true;
    struct idx_dev_buf_t {
        int dev = -1;
        float * q = nullptr; // [cap][idx_n_head*idx_head_dim]
        float * k = nullptr; // [cap][idx_head_dim]
        int cap = 0;
    };
    std::vector<idx_dev_buf_t> idx_dev_buf;
    // PF_QSA_SEL_DEV (step 3, opt-in): the raw-key ring and pool live on the
    // card per (attention layer, slot), the selector runs there, and Stage 3's
    // gather consumes the device selection directly.  Measured a wash at
    // 14.6k and 3.2 ms/token slower at 3.6k (the device chain competes with the
    // MoE), so it defaults off; `=0` keeps the host ring/pool/selector (the
    // step-2 path).
    bool sel_dev = false;
    struct idx_slot_dev {
        int dev = -1;
        float * pool = nullptr; // [(max_seq/kp)+1][idx_head_dim]
        float * ring = nullptr; // [kp][idx_head_dim]
        bool ready = false;     // host pool/ring handed off
    };
    std::vector<std::vector<std::unique_ptr<idx_slot_dev>>> idx_slot;
    // Per-device pinned handoff staging for the host -> device pool/ring copy
    // that happens once per (layer, slot) at the first sparse decode token.
    struct handoff_t {
        int dev = -1;
        float * stage = nullptr;
        size_t cap = 0;
        sycl::event ev;
        bool ev_valid = false;
    };
    std::vector<handoff_t> handoff;
    // Per-attention-layer pinned staging for the D2H inputs (mixed/q/k/v) and
    // the sparse attention's H2D output.  Allocated lazily and grown to the
    // largest chunk seen; a pageable std::vector makes the copy synchronous
    // and the q4 box's in-order queue drain was the cost.
    struct stage_t {
        float * mixed = nullptr;
        float * iq = nullptr; // PF_QSA_IDX_DEV: normed+RoPEd q, [cap_i][inh*id]
        float * ik = nullptr; // PF_QSA_IDX_DEV: raw indexer key, [cap_i][id]
        size_t cap = 0;     // tokens: mixed
        size_t cap_i = 0;   // tokens: iq/ik
        int dev = -1;
        // Host sparse attention's scratch.  The selected rows are gathered from
        // the paged pool (dequantized) into a compact f32 buffer; a prefill chunk
        // gathers the union of its rows' selections once per layer.
        std::vector<float> gk, gv; // compact rows: [npos][n_head_kv*head_dim]
        // Pinned per-block staging for the gather's D2H copies (K/V data and,
        // for i8/i4, their scale planes).  Grown on demand; freed with the
        // rest of the per-layer staging in ~engine.
        uint8_t * sbk = nullptr, * sbv = nullptr, * sbsk = nullptr, * sbsv = nullptr;
        size_t cap_sbk = 0, cap_sbv = 0, cap_sbsk = 0, cap_sbsv = 0;
        std::vector<int32_t> gpos;  // union of selected positions (ascending)
        std::vector<uint8_t> gused; // per-position membership bitmap
        std::vector<int32_t> gmap;  // position -> compact row (-1 = absent)
        std::vector<int32_t> gblk;  // unique blocks in gpos
        std::vector<int32_t> grange; // [start,end) in gpos per gblk entry
        std::vector<int32_t> gstage; // staging block index per gblk entry
        std::vector<int32_t> grun_start, grun_len; // coalesced block runs
        std::vector<int32_t> gsel;  // one row's mapped selection
        std::vector<int32_t> gsel_all, goff, gcnt; // every row's selection
    };
    std::vector<stage_t> stage;
    // The host sparse path's Q and output staging is shared across the
    // attention layers of one device: only one layer runs inline at a time, and
    // per-layer copies cost ~30 MB x 12 of pinned USM (which the Level Zero
    // driver charges against the card) at a 1024-token chunk.
    struct host_stage_t {
        float * q = nullptr;
        float * out = nullptr;
        size_t cap_q = 0, cap_out = 0;
        int dev = -1;
        // The H2D of the host attention output is asynchronous; the buffer is
        // shared by the device's attention layers, so the next layer's memset
        // must not overwrite it before the copy has read it.
        sycl::event out_ev;
        bool out_pending = false;
    };
    std::vector<host_stage_t> host_stage;
    bool devattn = true; // PF_QSA_DEVATTN (default on, Stage 2; =0 keeps host)
    bool pf_dev = true;  // PF_QSA_PF_DEV: device sparse *prefill* (see docs 18 §4)
    int pf_wave = 8;     // PF_QSA_PF_WAVE: prefill rows per compact pool
    // PF_QSA_DEVATTN scratch.  Per device: the compact f16 K/V pool (the same
    // [block][kv head][kBlockSize token][head_dim] layout as the paged pool)
    // and its sequential table + selected-index buffer.  Per attention layer:
    // a device step_info (filled by qsa_sinfo_launch) and the pinned host
    // `sel` staging, which is per layer so a later layer's selector cannot
    // overwrite it before its H2D copy has completed.
    struct dev_buf {
        int dev = -1;
        sycl::half * ck = nullptr;
        sycl::half * cv = nullptr;
        int32_t * dsel = nullptr;  // [kMaxB*kMaxT][cap_keys]
        int32_t * dcounts = nullptr; // [kMaxB*kMaxT] selected-key count per row
        int32_t * tab = nullptr; // identity compact block table (device)
        int32_t * htab = nullptr; // pinned staging for the one-time table copy
        float * scores = nullptr; // PF_QSA_SEL_DEV: per-pool selector scores [max_seq/kp+1]
        int32_t * dpools = nullptr; // PF_QSA_SEL_DEV: selected pool ids
        int cap_keys = 0;
        int cap_blocks = 0;
        int cap_wave = 0; // prefill rows per compact pool
        int cap_pool = 0; // scores/dpools capacity
    };
    std::vector<dev_buf> dev;
    struct dev_aux {
        int dev = -1;
        int32_t * hsel = nullptr; // pinned staging, [kMaxB*kMaxT][cap_keys]
        int32_t * hcnt = nullptr; // pinned staging, [kMaxB*kMaxT]
        step_info * sinfo = nullptr; // device, one wave of rows
        // PF_QSA_EARLY decode path: the `sel` H2D is issued without waiting, so
        // its pinned source must not be overwritten while the copy is in
        // flight.  Two cap_keys-wide buffers are round-robined; before writing
        // one, the copy that last read it (two calls ago) is waited for.  The
        // prefill staging above is separate, so the two paths cannot collide.
        int32_t * dsel_stage[2] = {nullptr, nullptr};
        sycl::event dsel_ev[2];
        bool dsel_ev_valid[2] = {false, false};
        int dsel_par = 0;
    };
    std::vector<dev_aux> aux;
    // PF_QSA_TIME: the host QSA split, accumulated over one forward.  `copy`
    // is the D2H waits, `gemv` the indexer projections (q is skipped when no
    // selector runs), `upd` the per-token norm/RoPE/pool/mirror update, `sel`
    // the selector and `attn` the sparse attention.  `rows`/`nsel` size the
    // selector and the attention.
    double copy_ms = 0.0;
    double gemv_ms = 0.0;
    double upd_ms = 0.0;
    double sel_ms = 0.0;
    double attn_ms = 0.0;
    long rows = 0;
    long nsel = 0;
};

} // namespace si
