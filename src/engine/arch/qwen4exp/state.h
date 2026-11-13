#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "kernels.h"
#include "qsa_state.h"

namespace si {

// qwen4exp (hyper-connections + PLE) state: everything the model's own forward
// owns.  The engine composes one of these (`engine::q4x_`); the member
// functions that drive it stay on `engine`.
struct qwen4exp_state {
    // Per-device scratch for the trunk.  The residual is [token][hc][n_embd];
    // `xn`, `gate`, `mixed`, `block` are the per-layer temporaries.  The PLE
    // module and the routed MoE run on the host (their tables and expert bank
    // are host-resident), so their buffers are engine-wide, not per device.
    struct q4_bufs {
        float * res = nullptr;
        float * xn = nullptr;
        float * gate = nullptr;
        float * mixed = nullptr;
        float * block = nullptr;
        float * lo = nullptr;
        float * inj = nullptr;
        gemv_seg * segs = nullptr;      // device (or host, for a cpu partition)
        gemv_seg * host_segs = nullptr; // staging ring for the async copy
        int segs_cap = 0;
        int seg_used = 0;  // next ring slot
        int seg_alloc = 0; // allocations since the ring last wrapped (drained reset)
        float * router_logits = nullptr;
        int32_t * router_ids = nullptr;
        float * router_wts = nullptr;
    };
    std::vector<q4_bufs> q4b_;
    float * q4_host_res_ = nullptr; // host staging for a partition handoff
    float * q4_host_mix_ = nullptr;
    float * q4_host_out_ = nullptr;
    float * q4_moe_logits_ = nullptr;
    float * q4_moe_wts_ = nullptr;
    float * q4_moe_acc_ = nullptr;
    // PF_Q4_CPU_MISS: the host-computed miss pair rows, pinned so the H2D into
    // the device A.acc is an enqueue rather than a pageable blocking copy.
    float * q4_miss_acc_ = nullptr;
    float * q4_moe_sh_ = nullptr;
    float * q4_moe_shout_ = nullptr;
    int32_t * q4_moe_ids_ = nullptr;
    // routing histogram (PF_Q4_HIST): (layer, expert) pair counts, prefill and
    // decode kept apart.  Diagnostic only - it is never read by a kernel.
    // Per-expert hot bank: R slots per GPU layer, calibrated from the first
    // prefill's routing (PF_Q4_HOT, the default when --moe-vram-mb > 0).
    // PF_Q4_BANK_LAYER=1 keeps the older whole-layer bank for A/B.
    bool q4_hot_ = false;
    bool q4_hot_ready_ = false;
    bool q4_cal_done_ = false; // the first prefill's hot set has been frozen
    int q4_hot_r_ = 0;
    std::vector<uint32_t> q4_cal_;
    size_t q4_cal_pairs_ = 0;
    std::vector<int32_t> q4_sm_host_;
    // Pinned staging ring for the LRU expert uploads.  A q.memcpy whose source
    // is pageable blocks the host until the in-order queue drains (measured
    // 1.7 ms for 1 MB behind a kernel against 0.016 ms from pinned USM), and
    // the LRU's source is the GGUF mmap: PF_Q4_HOT_TIME priced the uploads at
    // 1.43 ms/expert, 112 ms of a decode token.  Staging keeps the ordering
    // (the device copy is still enqueued before the kernels that read the
    // bank) and takes the host out of the wait.
    std::vector<char *> q4_up_stage_;    // per device: pinned ring base
    std::vector<sycl::event> q4_up_ev_;  // [dev * q4_up_slots_ + slot]
    std::vector<unsigned char> q4_up_live_;
    std::vector<int> q4_up_next_;        // per device: next ring slot
    size_t q4_up_stride_ = 0;            // bytes per ring slot (max plane)
    int q4_up_slots_ = 0;
    // PF_Q4_UPSTAT: the upload split.  `copy` is the host memcpy into the ring,
    // `enq` the device-copy enqueues, `wait` the ring-slot event waits, and
    // `bytes` what went through the ring, so a `copy` that does not scale with
    // threads can be told from one that is already at the DMA floor.
    double q4_up_wait_ms_ = 0.0;
    double q4_up_copy_ms_ = 0.0;
    double q4_up_enq_ms_ = 0.0;
    double q4_up_dma_ms_ = 0.0;          // PF_Q4_UPDMA: the H2D floor, measured
    size_t q4_up_bytes_ = 0;
    size_t q4_up_calls_ = 0;
    // PF_Q4_UPSTAT: the device copies' own event intervals, so the printed rate
    // is the DMA rate and not the host memcpy's.  Before this the line divided
    // bytes by the pageable->pinned host memcpy time, which is zero on the
    // shipped pinned path and printed either 0.00 or absurd values (17389 GB/s)
    // depending on whether a ring slot happened to be reused in that step.
    std::vector<sycl::event> q4_up_prof_;
    std::vector<std::unique_ptr<sycl::queue>> q4_up_q_; // per device, in-order copy queue
    std::vector<sycl::event> q4_up_last_ev_;            // last async copy this batch
    std::vector<unsigned char> q4_up_last_live_;
    std::vector<int> R_cur_;           // per device: slots per layer
    std::vector<size_t> q4_r_off_;     // [n_layer+1] offsets into the LRU arrays
    // [layer][R_l] with R_l from that layer's device; q4_r_off_ gives the
    // per-layer offset, so index these through it and never as il * R.
    std::vector<uint32_t> q4_lru_age_; // the per-slot LRU clock
    std::vector<float> q4_lru_heat_;   // decayed routing heat (PF_Q4_HEAT)
    uint32_t q4_lru_tick_ = 0;
    bool q4_hotchk_done_ = false;
    // Decode forwards seen by the q4 path (PF_Q4_CPU_MISS's warm-up gate).
    uint32_t q4_dec_step_ = 0;
    float * q4_spill_acc_ = nullptr;
    std::vector<uint64_t> q4_hist_pre_, q4_hist_dec_;
    size_t q4_hist_pre_pairs_ = 0, q4_hist_dec_pairs_ = 0;
    float * q4_ple_hist_ = nullptr;
    int32_t * q4_ple_nhist_ = nullptr;
    float * q4_ple_emb_ = nullptr;
    float * q4_ple_key_ = nullptr;
    float * q4_ple_val_ = nullptr;
    float * q4_ple_gate_ = nullptr;
    float * q4_ple_a_ = nullptr;
    float * q4_ple_b_ = nullptr;
    int32_t * q4_ple_prev_ = nullptr; // [slot][ple_ngram-1] last tokens before the batch
    std::vector<void *> q4_owned_; // host buffers freed in ~engine
    // full-layer VRAM expert banks (all 512 experts of a layer); `--moe-vram-mb`
    // is the per-device byte budget.  A layer with a bank runs its whole MoE on
    // the GPU instead of the host.
    struct q4_bank_t {
        void * g = nullptr;
        void * u = nullptr;
        void * d = nullptr;
        size_t sg = 0, su = 0, sd = 0;
        int r = 0; // slots this layer's device could afford
        int32_t * sm = nullptr;  // slot map (decode / first prefill batch)
        int32_t * sm2 = nullptr; // alternate map for the two-pass prefetch
        void * hg = nullptr;     // pinned host copy of this layer's expert planes
        void * hu = nullptr;
        void * hd = nullptr;
        bool q5sb = false;       // hd holds the 128-value Q5_1 down panel repack
    };
    std::vector<q4_bank_t> q4_bank_layer_;
    std::vector<void *> q4_host_usm_; // pinned expert mirrors, freed with sycl::free
    std::vector<std::pair<void *, int>> q4_pin_src_; // pinned expert source + device, freed with sycl::free
    bool q4_mirror_ = false;
    // Resolved once in the constructor: CLI/engine_config > PF_Q4_TAIL_SKIP >
    // 0 (exact expert set).  0 means no tail-skip.
    int q4_tail_skip_ = 0;
    size_t engram_ram_mb_ = 0;
    size_t engram_cap_ = 0; // power of two
    std::vector<int32_t> engram_tag_;
    std::vector<float> engram_row_;
    size_t engram_hits_ = 0, engram_misses_ = 0;
    // ---- qwen4exp prefix-cache extras -----------------------------------
    // The generic checkpoint pool (`d_pc_states`) holds only the GDN/conv
    // state, which a kernel can snapshot at a block boundary.  qwen4exp also
    // needs two pieces of *host* state:
    //   * the PLE rolling history (the last (conv_k-1)*ngram normalised rows)
    //     plus the previous-token ring; and
    //   * the QSA indexer pool/ring (the raw indexer keys, which are not in the
    //     paged KV and cannot be recomputed without the prefix hidden states).
    // Both are indexed by the same checkpoint slot as `d_pc_states`; the QSA
    // host mirror is *not* stored - it is the same K/V the paged pool holds and
    // is rebuilt from the restored blocks when the host path needs it.
    struct q4_pc_ple_state {
        std::vector<float> hist;   // hist_rows*hc_dim, right-aligned
        std::vector<int32_t> prev; // ngram-1 previous token ids
        int32_t nhist = 0;
        bool valid = false;
    };
    std::vector<q4_pc_ple_state> q4_pc_ple_;
    struct q4_pc_qsa_state {
        std::vector<std::vector<float>> pool; // [attn layer][n_pool*idx_head_dim]
        std::vector<std::vector<float>> ring; // [attn layer][kp*idx_head_dim]
        bool valid = false;
    };
    std::vector<q4_pc_qsa_state> q4_pc_qsa_;
    // QSA runtime state (host indexer cache, deferred update, device scratch).
    // The member functions that drive it stay on `engine`; the data lives here
    // because it is qwen4exp-only (see arch/qwen4exp/qsa_state.h).
    qsa_state qsa;
};

} // namespace si
