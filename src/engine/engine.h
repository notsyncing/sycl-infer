#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>                               // IWYU pragma: keep
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp> // IWYU pragma: keep

#include "dnnl_gemm.h"
#include "backend.h"
#include "kernels.h"
#include "model.h"
#include "multimodal.h"
#include "pc_disk.h"
#include "pc_ram.h"
#include "sampler.h"
#include "tokenizer.h"

namespace si {

namespace sx = sycl::ext::oneapi::experimental;

// plan of GEMV calls for one full forward pass (must match record order)
struct seg_plan {
    std::vector<gemv_seg> segs;
    std::vector<size_t> call_offsets;
    std::vector<int> call_counts;
    std::vector<int> call_total_rows;
    std::vector<int> call_tb;
    std::vector<int> call_nsb;
    struct group_t {
        uint32_t type;
        int off;
        int n;
        int rows;
    };
    std::vector<group_t> groups;
    std::vector<int> call_group_begin;
    std::vector<int> call_group_count;
    // first call index of each layer, plus a final entry for the LM head call.
    // The call-group metadata arrays are plan-global and indexed by call order,
    // so a partial (per-device) replay must start its call cursor at the phase's
    // first layer instead of at 0.
    std::vector<int> layer_c0;
    // DP4A path: activation quantization needed before each call
    struct xq_t {
        const float * x = nullptr;
        const float * up = nullptr;
        int x_stride = 0;
        int up_stride = 0;
        int K = 0;
    };
    std::vector<xq_t> call_xq; // one entry per call (empty = activations already fp32)
    // false for the non-final prefill variant: record_forward must not replay a
    // "head" call then (there is none, and the last FFN call must not repeat)
    bool has_head = true;
    void set_xq(const float * x, const float * up, int xs, int us, int K) {
        call_xq.back() = {x, up, xs, us, K};
    }
    void begin_call(int tb, int nsb) {
        call_offsets.push_back(segs.size());
        call_counts.push_back(0);
        call_total_rows.push_back(0);
        call_tb.push_back(tb);
        call_nsb.push_back(nsb);
        call_xq.push_back({});
    }
    void add(const gemv_seg & s) {
        segs.push_back(s);
        call_counts.back()++;
        call_total_rows.back() += s.n_rows;
    }
    // ffn_down's activation is silu(gate)*up.  The quantizer path applies that
    // from call_xq.up; every other consumer of these segments (the fp32 GEMV
    // and the oneDNN fallback) reads the raw gate from seg.x and must apply it
    // itself, which it does when act_up is set.  Patch the segments added by
    // the current call, preserving any per-slice x offset.
    void set_act_up(const float * up_base, const float * x_base) {
        for (size_t i = (size_t)call_offsets.back(); i < segs.size(); i++) {
            segs[i].act_up = up_base + (segs[i].x - x_base);
        }
    }
    void finalize() {
        for (size_t ci = 0; ci < call_offsets.size(); ci++) {
            const int off = (int)call_offsets[ci];
            const int n = call_counts[ci];
            std::stable_sort(segs.begin() + off, segs.begin() + off + n,
                             [](const gemv_seg & a, const gemv_seg & b) {
                                 return (a.dev < b.dev) || (a.dev == b.dev && a.type < b.type);
                             });
            call_group_begin.push_back((int)groups.size());
            int i = 0;
            while (i < n) {
                const uint32_t t = segs[off + i].type;
                int j = i, rows = 0;
                while (j < n && segs[off + j].type == t) {
                    rows += segs[off + j].n_rows;
                    j++;
                }
                groups.push_back({t, off + i, j - i, rows});
                i = j;
            }
            call_group_count.push_back((int)groups.size() - call_group_begin.back());
        }
    }
};

struct engine {
    model m;
    tokenizer tk;
    int device_req = -1; // resolved device request (0 = gpu, 1 = cpu)
    // A pipeline-parallel run over several GPUs shares one SYCL context: USM
    // host allocations are context-scoped, so without this the host-USM
    // activations are not valid on the second device (intermittent faults).
    std::shared_ptr<sycl::context> md_ctx_;
    sycl::queue q;
    // compute backend: one for the single-device path; multi-device runs keep
    // one backend per layer partition (see layer_dev_ / backends_)
    device_kind dev_kind = device_kind::gpu;
    bool cpu_mode = false; // single-device CPU compute (no command graphs)
    std::unique_ptr<compute_backend> be;
    compute_backend & backend() {
        return multi_dev ? *backends_[0] : *be;
    }

    // ---- multi-device layer placement (dense models) --------------------
    // One backend per distinct device named in the layer map; each attention
    // layer's paged KV lives on the device that computes it, so the pool is
    // split per device (same global block ids, one pool per device).  For a
    // mixed CPU/GPU map the activations stay in host USM so the boundary
    // hidden state is shared without an explicit copy.
    bool multi_dev = false;
    bool host_act = false; // engine scratch allocated as host USM
    std::vector<std::unique_ptr<compute_backend>> backends_;
    std::vector<std::unique_ptr<sycl::queue>> owned_queues_; // extra GPU queues we create
    std::vector<sycl::queue *> dev_queues_;                  // GPU backend queues (may alias q)
    // per backend: host tensor data ptr -> device memory copy.  Only the
    // tensors of the layers placed on that device (plus the global tok_embd /
    // output_norm, which run on the primary device) are uploaded, so the GPU
    // holds just its partition of the weights, not a whole-GGUF copy.  Empty
    // for a backend that reads the host mmap directly (CPU).
    std::vector<std::unordered_map<const void *, void *>> weight_maps_;
    // multi-device decode: per-backend map of the SIn (w8) copies of the GPU
    // partitions' layer weights (host tensor pointer -> w8t on that device,
    // allocated on the device's queue).  The single-sequence decode plan then
    // runs the GPU segments on dp4a_gemv (like single-device) and the CPU ones
    // on i8_gemv; empty vector when int8 decode is disabled (PF_DP4A /
    // PF_DP4A_DEC=0).
    bool md_int8 = false;
    // multi-device XMX path: oneDNN int8 weights built per device (used when
    // the device supports the int8 matmul primitive); falls back to the w8/dp4a
    // path when oneDNN is unavailable for the device (PF_GEMM_DNNL=0)
    bool md_xmx = false;
    bool setup_md_dnnl();
    std::vector<std::unordered_map<const void *, w8t>> w8_dev_;
    std::vector<int> layer_dev_;                           // layer -> backend index
    std::vector<int> layer_attn_local_;                    // layer -> local attention index (-1)
    std::vector<int> dev_kind_;                            // backend -> 0 gpu / 1 cpu
    std::vector<void *> dev_kpool_, dev_vpool_, dev_kscales_, dev_vscales_;
    compute_backend * cur_be = nullptr; // backend of the layer being recorded
    compute_backend & be_of(int layer) {
        return multi_dev ? *backends_[(size_t)layer_dev_[layer]] : *be;
    }
    void sync_all();
    const void * wptr(int dev, const void * host) const;
    // Resolve a weight key the way build_plan's wkey does: the uploaded device
    // pointer when the tensor was copied to that partition, otherwise the host
    // pointer - which is the tensor's oneDNN key when it was converted (the
    // upload skips the raw copy of every converted tensor).
    const void * wkey(int dev, const void * host) const {
        if (!host) {
            return nullptr;
        }
        if (!multi_dev) {
            return wptr(dev, host);
        }
        const auto & mp = weight_maps_[(size_t)dev];
        auto it = mp.find(host);
        return it != mp.end() ? it->second : host;
    }
    const float * wf32(int dev, const float * host) const;
    int max_seq;                 // max tokens per sequence (limited by the block pool)
    int n_splits;                // prefill K-split ceiling (PF_ATTN_SPLIT overrides)
    int dec_splits = kMaxSplits; // decode K-split (PF_DEC_SPLIT overrides)
    int ffn_stride;              // 2 * n_ff
    int n_blocks;                // reserved KV blocks (virtual address range size)
    int max_blocks;              // block table size per sequence
    size_t kv_layer_stride;      // K pool bytes per attention layer
    size_t kv_v_layer_stride = 0; // V pool bytes per attention layer (== kv_layer_stride
                                  // unless --kv-type K:V mixes i4 with i8)

    // activation buffers, [kMaxRows][...]
    float * d_x = nullptr;
    float * d_xnorm = nullptr;
    float * d_qkv = nullptr;
    float * d_z = nullptr;
    float * d_beta = nullptr;
    float * d_alpha = nullptr;
    float * d_conv_out = nullptr;
    float * d_attn_pre = nullptr;
    float * d_attn_merged = nullptr;
    float * d_qbuf = nullptr;
    float * d_kbuf = nullptr;
    float * d_vbuf = nullptr;
    float * d_attn_out = nullptr;
    float * d_ffn = nullptr;
    float * d_partials = nullptr;     // prefill partials [R][n_head][n_splits][2+head_dim]
    float * d_partials_dec = nullptr; // decode partials [kMaxB][n_head][dec_splits][2+head_dim]
    float * d_logits = nullptr;       // [kMaxB][vocab]
    float * d_last_hidden = nullptr;
    float * d_img_embd = nullptr; // [kMaxImgTokens][n_embd] vision embeddings

    // paged KV cache + tables.  The pools are raw byte storage whose element
    // type is kv_dtype() (f32/bf16/f16/i8/i4, see kernels.h); sizes are in
    // elements of that type and all pointers are advanced in bytes.
    void * d_kpool = nullptr; // [n_attn][n_blocks][n_head_kv][kBlockSize][head_dim]
    void * d_vpool = nullptr;
    // int8 only: per-32 fp16 scale planes, same [attn][block][kv head][token][group]
    // indexing, allocated with the same block count as the pools
    void * d_kscales = nullptr;
    void * d_vscales = nullptr;
    size_t kv_scale_stride = 0;   // bytes per attention layer in a scale plane (K)
    size_t kv_v_scale_stride = 0; // ... and V (== kv_scale_stride unless mixed)
    int32_t * d_tables = nullptr; // [kMaxB][max_blocks]
    std::vector<int32_t> h_tables;

    // recurrent states, one slot per sequence
    float * d_gdn_state = nullptr;  // [kMaxB][n_gdn][dt_rank][d_state][d_state]
    float * d_conv_state = nullptr; // [kMaxB][n_gdn][3][conv_dim]

    // ---------------- dynamic KV block pool ------------------------------
    // The pool address range is reserved once with virtual USM (so the recorded
    // graphs keep their base pointers) and the physical memory is committed in
    // extents as the pool grows.  Blocks are handed out lowest-first so a
    // completely free top extent can be unmapped again.
    int pool_blocks = 0;  // blocks backed by physical memory right now
    int pool_cap = 0;     // reservation size in blocks (>= pool_blocks)
    int pool_initial = 0; // blocks committed at startup (--blocks)
    int pool_chunk = 64;  // blocks committed per growth step
    uint64_t pool_peak = 0, pool_grows = 0, pool_shrinks = 0;
    bool kv_virtual = false; // false: plain fixed malloc_device pool (fallback)
    bool kv_initializing = false;
    bool kv_map2 = false; // extents are 2 MB aligned (Level Zero requirement)
    uintptr_t kv_vbase = 0, kv_vbase_v = 0;
    size_t kv_reserve_bytes = 0;
    size_t kv_reserve_bytes_v = 0;

    step_info * d_info = nullptr; // host USM
    // multi-device prefill pipeline: a second step_info so the chunk on device 0
    // and the previous chunk on device 1 can be in flight together
    step_info * d_info2_ = nullptr;
    // pipeline state: the chunk whose device-0 phase has been enqueued but whose
    // device-1 phase has not
    bool pf_pipe_pending_ = false;
    int pf_pipe_parity_ = 0;
    int pf_pipe_rows_ = 0;
    bool pf_pipe_head_ = false;
    // per-chunk step_info used by pc_capture_begin/pc_commit during the pipeline
    step_info * pf_info_ = nullptr;

    // SI8 DP4A scratch: quantized activations of the current call (kMaxT rows)
    int8_t * d_x8 = nullptr;
    sycl::float2 * d_xmeta = nullptr; // activation scales (fp32)
    int32_t * d_xsumq = nullptr;

    // Pipeline parallel: one activation set per backend, allocated in that
    // device's own USM.  The GEMMs then read activations (especially the huge
    // per-group x8 scratch) from VRAM instead of re-reading host-USM over PCIe.
    // `as_` is empty on the single-device path, which keeps the members above.
    struct act_set {
        float * x = nullptr, *xnorm = nullptr, *qkv = nullptr, *z = nullptr, *beta = nullptr, *alpha = nullptr;
        float * conv_out = nullptr, *attn_pre = nullptr, *attn_merged = nullptr, *qbuf = nullptr, *kbuf = nullptr;
        float * vbuf = nullptr, *attn_out = nullptr, *ffn = nullptr, *partials = nullptr, *partials_dec = nullptr;
        int8_t * x8 = nullptr;
        sycl::float2 * xmeta = nullptr;
        int32_t * xsumq = nullptr;
        // this partition's recurrent state (its own GDN layers only; the state
        // never crosses a partition boundary, only the hidden activation does)
        float * gdn_state = nullptr;
        float * conv_state = nullptr;
    };
    std::vector<act_set> as_;
    // global layer -> index of its GDN layer within its own partition
    std::vector<int> layer_gdn_local_;
    std::vector<int> n_gdn_dev_;
    act_set alloc_act_set(int dev);
    // point the d_* members at backend `dev`'s buffers (no-op single-device)
    void bind_acts(int dev);
    // hand the hidden state from backend `from` to `to` (device-to-device via a
    // host staging buffer, blocking on `from`).  `rows` is the number of flat
    // token rows the forward actually uses - the activation layout is
    // [token][n_embd], so copy only those instead of the full kMaxB*kMaxT
    // staging buffer (single-token decode needs 1 row, not 512).
    void handoff_x(int from, int to, size_t rows, void * staging = nullptr);
    void * h_handoff = nullptr;
    // second staging buffer so the pipelined prefill can enqueue the device-1
    // copy of chunk i while the host stages chunk i+1
    void * h_handoff2_ = nullptr;

    gemv_seg * d_segs_dec = nullptr;
    gemv_seg * d_segs_dec8 = nullptr; // CPU SI8 decode plan (no command graph)
    gemv_seg * d_segs_pf = nullptr;
    gemv_seg * d_segs_pf8 = nullptr;
    gemv_seg * d_segs_aux = nullptr;
    // CPU/multi-device prefill: one plan per token count T in {8,16,24,32} so a
    // short prompt does not compute a full kMaxT chunk.  `plan_pf_`/graph-based
    // plans above stay for the GPU path; these are the host-side replays.
    static constexpr int kPfSlice = 8;
    static constexpr int kPfSlots = kMaxT / kPfSlice;
    seg_plan plan_pf_slot[kPfSlots];    // fp32, with head (final chunk / generate)
    seg_plan plan_pf_nh_slot[kPfSlots]; // fp32, no head (scheduler non-final chunk)
    gemv_seg * d_segs_pf_slot[kPfSlots] = {};
    gemv_seg * d_segs_pf_nh_slot[kPfSlots] = {};
    // decode graphs bucketed by batch size (1, 2, 4, 8, 16)
    struct dec_bucket {
        int tb;
        seg_plan plan;
        gemv_seg * d_segs;
        std::unique_ptr<sx::command_graph<sx::graph_state::modifiable>> g;
        std::unique_ptr<sx::command_graph<sx::graph_state::executable>> e;
    };
    std::vector<dec_bucket> buckets_;

    float * h_logits = nullptr; // [kMaxB][vocab]

    std::unique_ptr<sx::command_graph<sx::graph_state::modifiable>> g_dec, g_pf;
    // prefill graph using the SI8/DP4A int8 path
    std::unique_ptr<sx::command_graph<sx::graph_state::modifiable>> g_pf8;
    std::unique_ptr<sx::command_graph<sx::graph_state::executable>> e_pf8;
    // prefills that are not the last prompt chunk do not need the LM head
    std::unique_ptr<sx::command_graph<sx::graph_state::executable>> e_pf8_nh;
    // CPU backend: prefill plans (final / non-final chunk) and their host
    // segment copies; recorded graphs above are unused on the CPU path
    seg_plan plan_pf8_nh_;
    gemv_seg * d_segs_pf8_nh = nullptr;
    // decode (batch-1) graph on the SI8/DP4A path
    std::unique_ptr<sx::command_graph<sx::graph_state::modifiable>> g_dec8;
    std::unique_ptr<sx::command_graph<sx::graph_state::executable>> e_dec8;

    // chunk-batched prefill (mode 2): GEMMs of all chunks batched per layer,
    // chunk-major row segments so one tensor's weights stay L2-hot.
    // One recorded graph per batch size (the chunk-row loops and the M-tiled
    // token count are baked into the recording, so each size needs its own).
    gemv_seg * d_segs_pfb = nullptr; // [kMaxB][plan.segs.size()] row-offset copies
    struct pfb_variant {
        int ntok = 0; // tokens covered (multiple of kMaxT)
        std::unique_ptr<sx::command_graph<sx::graph_state::modifiable>> g;
        std::unique_ptr<sx::command_graph<sx::graph_state::executable>> e;
    };
    std::vector<pfb_variant> pfb_vars_;
    seg_plan plan_pfb_;
    // Largest chunk-batched size to use for `rem` pending prompt tokens (0 =
    // none).  The dp4a path replays a recorded graph, so it is limited to the
    // recorded variants (one graph per batch size).  PF_GEMM_DNNL replays mode
    // 2 directly, where every multiple of kMaxT up to kMaxB*kMaxT is a valid
    // batch size, so it prefills the whole remainder in one forward (the old
    // 256+128+25 split re-read all weights three times for a 409-token prompt).
    int batched_prefill_fit(int rem) const {
        if (cpu_mode) {
            return 0; // CPU uses the chunked prefill path
        }
        // single-GPU oneDNN and multi-device with a oneDNN GPU partition both
        // replay mode 2 directly, where every batch size up to kMaxB*kMaxT is
        // valid (the last row may be partial, see step_info::n_real_row).
        // The md_int8 fallback must NOT take partial batches: its dp4a GEMM
        // (and the per-device w8/x8 layout) is only correct for a full
        // kMaxT-token row, so a partial mode-2 batch silently corrupts the
        // hidden state (visible as a decode-vs-prefill mismatch on 2-GPU
        // md_int8).  Fall through to the recorded multiples-of-kMaxT variants
        // and the chunked path for the tail, exactly like the fp32 path.
        if (use_dnnl || (multi_dev && dnnl_any_dev())) {
            // direct replay: any batch size up to kMaxB*kMaxT is valid (the
            // last row may be partial, see step_info::n_real_row)
            if (getenv("PF_NO_PFB_PARTIAL")) { // A/B: old multiple-of-kMaxT only
                int n = (rem / kMaxT) * kMaxT;
                if (n > kMaxB * kMaxT) {
                    n = kMaxB * kMaxT;
                }
                return n >= 2 * kMaxT ? n : 0;
            }
            // PF_PFB_MAX_M: optional cap on the mode-2 batch (A/B; 0 = no cap)
            //
            // The old default capped the multi-device batch at kMaxT because a
            // batch of M >= 2*kMaxT produced wrong logits.  The real cause was
            // the fully-fused GDN kernel (PF_GDN_FUSE=2) ignoring `nreal_arg`:
            // the fused mode-2 call passes one row of T tokens, but the kernel
            // used row_nr(info, 0) = kMaxT, so only the first chunk row got the
            // recurrence and every later row saw a stale state.  gdn.cpp now
            // honors nreal_arg (like cpu_gdn), so mode-2 is correct at any M
            // and the cap is gone.  Keeping it at kMaxT cost ~7x on multi-device
            // pp512 (mode-1 chunked, ~91 t/s vs ~660 t/s mode-2 on 2x A770).
            const char * em = getenv("PF_PFB_MAX_M");
            const int mcap = em ? atoi(em) : 0;
            if (multi_dev && mcap > 0) {
                return std::min(rem, mcap);
            }
            return std::min(rem, kMaxB * kMaxT);
        }
        int best = 0;
        for (const auto & v : pfb_vars_) {
            if (v.ntok <= rem && v.ntok > best) {
                best = v.ntok;
            }
        }
        return best;
    }
    bool pf8 = false;
    bool pf8_dec = false; // SI8 decode (needs packed weights to win; off by default)
    // oneDNN int8 matmul for the chunk-batched (mode 2) prefill
    // (PF_GEMM_DNNL, default on); null when disabled with PF_GEMM_DNNL=0
    std::unique_ptr<dnnl_gemm> dnnl;
    bool use_dnnl = false;
    // multi-device: one dnnl_gemm per GPU backend, each bound to that device's
    // queue and holding only the layers that device computes (CPU layers keep
    // reading the GGUF blocks; there are no SIn w8 copies on this path, so the
    // weight keys are the fp32 device-weight pointers the plan segments carry
    // in gemv_seg::w).  Null entries for CPU backends and when PF_GEMM_DNNL=0.
    std::vector<std::unique_ptr<dnnl_gemm>> dnnl_dev_;
    // true when at least one GPU backend owns a dnnl_gemm: multi-device prefill
    // then runs every GPU layer on oneDNN via the full-chunk pf8-style plan
    bool dnnl_any_dev() const {
        return std::any_of(dnnl_dev_.begin(), dnnl_dev_.end(),
                           [](const std::unique_ptr<dnnl_gemm> & d) { return d != nullptr; });
    }
    // the dnnl_gemm bound to backend `dev` (the single-device instance, or the
    // device's own in multi-device mode; null on CPU backends)
    dnnl_gemm * dnnl_for(int dev) const {
        if (!multi_dev) {
            return dnnl.get();
        }
        return (size_t)dev < dnnl_dev_.size() ? dnnl_dev_[(size_t)dev].get() : nullptr;
    }
    std::unique_ptr<sx::command_graph<sx::graph_state::executable>> e_dec, e_pf;

    seg_plan plan_dec_, plan_pf_, plan_pf8_, plan_dec8_;

    // ------------------------------------------------------------------
    // MTP (NextN) speculative draft head.  Enabled when the GGUF ships a
    // blk.<n_layer> NextN block and the caller asked for it (--mtp N).  The MTP
    // layer is a full-attention qwen35 block that runs on the primary device and
    // owns its own paged KV slice (attention layer index attn_layers()-1), so it
    // shares the block table, the prefix cache and the KV storage type.
    // ------------------------------------------------------------------
    bool mtp_on = false;
    int mtp_dev = 0; // backend index the MTP layer runs on (its KV follows)
    int mtp_k = 0;      // draft predictions per verify cycle
    int mtp_nsnap = 0;  // per-token recurrent-state snapshots kept (k+1)
    int mtp_attn_local_ = -1; // MTP layer index inside the primary device's pool
    size_t mtp_hist_floats = 0; // per device: nsnap * n_local_gdn * gdn_per
    std::vector<float *> d_mtp_hist_;    // [dev] per-token GDN state snapshots
    std::vector<float *> d_mtp_convsave_; // [dev] conv rows (1,2) before the verify
    float * d_mtp_cat = nullptr;       // [R][2*n_embd] concat(enorm(emb), hnorm(h))
    float * d_mtp_x = nullptr;         // [R][n_embd]
    float * d_mtp_xnorm = nullptr;     // [R][n_embd]
    float * d_mtp_qbuf = nullptr;      // [R][n_head*2*head_dim]
    float * d_mtp_kbuf = nullptr;      // [R][n_head_kv*head_dim]
    float * d_mtp_vbuf = nullptr;      // [R][n_head_kv*head_dim]
    float * d_mtp_attn_out = nullptr;  // [R][n_head*head_dim]
    float * d_mtp_ffn = nullptr;       // [R][2*n_ff]
    float * d_mtp_hnorm = nullptr;
    float * d_mtp_raw = nullptr; // MTP hidden before shared_head_norm (AR seed)     // [R][n_embd] shared-head-normalized hidden
    float * d_mtp_main_h = nullptr;    // [kMaxB*kMaxT][n_embd] main hidden capture
    std::vector<float> h_save_;        // PF_MTP_STATECHK: pre-verify recurrent state
    // MTP rollback by recurrence replay: the verify saves each GDN layer's
    // per-token (conv_out, alpha, beta) plus the pre-verify GDN state, and the
    // commit restores the state and replays those tokens.
    std::vector<float *> d_mtp_rin_;
    std::vector<float *> d_mtp_qsave_; // per-layer raw GDN conv taps of the verify rows    // [dev] ng * mtp_nsnap * (conv_dim + 2*dt_rank)
    std::vector<float *> d_mtp_ssave_;  // [dev] ng * gdn_per
    step_info * d_mtp_rinfo = nullptr;  // step_info for the replay launches
    void mtp_rollback(int j);
    float * d_mtp_hnorm0 = nullptr;    // primary-device copy feeding the shared LM head
    float * d_mtp_hprev = nullptr;     // [kMaxB][n_embd] main hidden before the row
    step_info * d_mtp_info = nullptr;  // host USM step_info for the MTP layer
    seg_plan plan_vf_;                 // verify forward: mode 2 with a batched head
    gemv_seg * d_segs_vf = nullptr;
    seg_plan plan_mtp_;                // the MTP layer's own calls
    gemv_seg * d_segs_mtp = nullptr;
    // PF_MTP_LAYER_EXACT: same segments with `w` pointing at the raw GGUF bytes
    gemv_seg * d_segs_mtp_exact = nullptr;
    // PF_MTP_HEAD_W4: a draft-only u4 copy of the LM head, registered under a
    // private key (the target keeps its exact int8 head, so only the drafts -
    // and hence acceptance, never the emitted stream - can move).
    float * d_mtp_partials = nullptr; // the MTP's own attention partials (see mtp_splits)
    int32_t h_argmax[kMaxB] = {0};    // host mirror of the verify's per-row argmax
    bool mtp_head_w4_ = false;
    // PF_MTP_LAYER_W2: the MTP layer also has a 2-bit copy (0.375 B/weight).
    bool mtp_layer_w2_ = false;
    // 2-bit copy of the draft LM head (0.375 B/weight against the u4 copy's
    // 0.625): the draft reads the whole head once per drafted token, so this is
    // the single largest byte item in the speculative cycle.  The draft only
    // needs its argmax, and its weights are already re-quantized, so the extra
    // step is a trade in acceptance for bytes: 0.375 B/weight is exact enough
    // for an argmax (39.8 % relative L2) but costs ~2 % acceptance, which
    // cancels the saving.  Off with PF_MTP_HEAD_W2=0.
    bool mtp_head_w2_ = false;
    char mtp_head_w2_key_[1] = {0};
    bool mtp_layer_w4_ = false; // the MTP layer's own linears are GEMV-only u4 too
    char mtp_head_w4_key_[1] = {0};
    // The MTP draft is a single-token decode over the MTP layer's KV, so it gets
    // the decode's key-parallel split count - *not* n_splits, which the
    // --layer-map path pins to 1 (its split path is opt-in) and which left a
    // 1-row attention with n_head workgroups each looping the whole KV: 2451 vs
    // 183 ms/cycle at 128k.  PF_MTP_SPLITS overrides (1 = the old behaviour).
    int mtp_splits = kMaxDecSplits;
    void build_mtp_plan();
    void mtp_gemv(int ci, int M, int step = -1);
    // ---- MTP draft head (see mtp_argmax.cpp) -------------------------------
    // d_mtp_tok[i] is the token draft step i chose (the head's argmax, computed
    // on the device), read back by the next step's concat, so the chain needs no
    // host round-trip.  With PF_MTP_CAND the head is evaluated on a candidate set
    // (mtp_cand_launch / mtp_gather_launch) instead of all 248320 rows - measured
    // a net loss, so it is off by default (25/17/7 % hit rate at margin 8).
    int32_t * d_mtp_cand_ = nullptr;  // [mtp_cand_cap] candidate token ids (device 0)
    float * d_mtp_cvals_ = nullptr;   // [mtp_cand_cap] their head values
    int32_t * d_mtp_tok_ = nullptr;   // [kMaxT] per-step draft token ids (device 0, next to the head)
    // Head readout split across both cards (PF_MTP_HEAD_SPLIT=1, **default off**).
    // out[n] = w[n].h is independent per output row, so the draft's 795 MB u4
    // stream (3.0 ms of device time, 75 % of a draft step) can be halved by
    // giving the second device the upper row range while it is otherwise idle:
    // each side runs the identical GEMV over its own rows into the *same*
    // device-0 logits row, so the argmax stays one scan on the owning card.
    // Draft 17.2 -> 13.1 ms/cycle, and end to end a wash: halving N changes the
    // u4 GEMV's decomposition, so the draft's argmax flips on a near-tie now and
    // then (p0 -6 %, p2 -8 % acceptance).
    bool mtp_head_split_ = false;
    int mtp_head_half_ = 0;      // rows [0, half) on device 0, [half, n_vocab) on device 1
    char mtp_head_w4lo_key_[1] = {0};
    char mtp_head_w4b_key_[1] = {0};
    float * d_mtp_hnorm1 = nullptr;   // device-1 copy of the draft hidden
    float * h_head_stage = nullptr;   // host staging for the 20 KB activation copy
    float * d_mtp_cval1_ = nullptr;   // scalar: the chosen candidate's logit (PF_MTP_CANDV)
    int32_t * d_mtp_tokx_ = nullptr;  // 4-byte staging slot for a non-zero --mtp-device
    float * d_mtp_amv_ = nullptr;     // scalar: the seed distribution's argmax value
    int mtp_cand_src_ = 1;            // 1 = seed from the draft's own step-0 readout
    int mtp_cand_cap_ = 0;            // 0 = off (the full head readout)
    float mtp_cand_margin_ = 8.0f;
    bool mtp_cand_ok_ = false;        // the head's int8 view exists on device 0
    const int8_t * mtp_head_w8_ = nullptr;
    const uint16_t * mtp_head_wsc_ = nullptr;
    int mtp_head_rows_ = 0;
    // run the MTP layer over `n` tokens at positions pos0.. (mode 1 layout:
    // row 0, n <= kMaxT).  `h` is the main model's hidden [token][n_embd] for the
    // same tokens (h_{-1} comes from hprev), extra_writes the KV and the head.
    void mtp_forward(const int32_t * toks, const float * h, const float * hprev, int n, int slot, int pos0,
                     bool with_head, const int32_t * tok_dev = nullptr, int step = -1);
    // verify forward: main model over `n` tokens at pos0..; logits land in
    // d_logits rows 0..n-1.  Also fills d_mtp_hprev with the hidden at each row.
    void mtp_verify(const std::vector<int> & toks, int n, int slot, int pos0);
    std::vector<int> generate_mtp(const std::vector<int> & prompt, const gen_params & gp,
                                  const std::function<bool(int)> & cb, std::vector<float> * first_logits);

    // fp32 (scale, min) side arrays for Q4_K/Q5_K, keyed by the tensor's host data pointer
    std::unordered_map<const void *, sycl::float2 *> meta32_;
    bool use_meta32 = false;
    void build_meta32(sycl::queue & q);
    const sycl::float2 * meta32_of(const void * data) const {
        auto it = meta32_.find(data);
        return it == meta32_.end() ? nullptr : it->second;
    }

    std::mutex mtx;

    engine(const std::string & model_path, int max_seq = 8192, int n_splits = 16, int n_blocks = 512,
           int kv_cap_mb = 0, const std::string & pc_dir = "", int pc_disk_mb = -1, int pc_mem_mb = -1,
           int pc_ram_mb = -1, int pc_vram_mb = -1, int device = -1, const std::string & layer_map = "",
           int mtp_k = 0);
    ~engine();

    void reset_state();

    // ---- both entry points assume the caller holds `mtx` ----
    // process `n` tokens of one sequence starting at `start` (n <= kMaxT); the
    // sequence's KV blocks must already be assigned in its table row `slot`
    void prefill_chunk(const std::vector<int> & toks, int start, int n, int slot, bool with_head = true);
    // Prefill a whole text prompt with the fastest available path: the
    // chunk-batched mode 2 for every full kMaxT multiple it supports and the
    // chunked mode 1 only for the <kMaxT tail (or when no batched variant
    // exists, e.g. the CPU backend).  The sequence's KV blocks must already be
    // assigned in table row `slot`.
    void prefill_text(const std::vector<int> & toks, int slot, int n);
    // chunk-batched prefill: one row per kMaxT chunk, all chunks of one prompt
    // in a single forward (GEMMs segment-major, weights L2-hot).  Replayed
    // directly (mode 2): recorded graphs on single-device dp4a, oneDNN direct on
    // single-GPU PF_GEMM_DNNL, per-device oneDNN + i8_gemm on the multi-device
    // GPU/CPU partitions (batched_prefill_fit() enables it there).
    void prefill_batch(const std::vector<int> & toks, int start, int n, int slot, int pos0);
    // Multi-device prefill pipeline helpers (see prefill_batch).
    void pf_pipe_finish(bool head);
    void prefill_flush();
    static void setup_pf_info(step_info * inf, const std::vector<int> & toks, int start, int n, int slot, int pos0);
    // one decode step for up to kMaxB sequences; returns the logits rows
    void decode_batch(const int32_t * tokens, const int32_t * poss, const int32_t * slots, int n_rows);
    // copy the logits of row r to the host
    void fetch_logits(int row, float * out);
    // run the prefill forward without a graph (diagnostics / fallback)
    void forward_plain_pf();
    void forward_plain_pf8();
    void forward_plain_dec(int rows);

    // ---- simple single-sequence API (kept for tests / CLI) ----
    void reset_single();
    std::vector<float> eval(const std::vector<int> & tokens);
    std::vector<int> generate(const std::vector<int> & prompt, const gen_params & gp,
                              const std::function<bool(int)> & cb, std::vector<float> * first_logits = nullptr);
    // multimodal variant: `p` carries the expanded tokens, the vision
    // embeddings and the M-RoPE positions of the prompt
    std::vector<int> generate_mm(const mm_prompt & p, const gen_params & gp, const std::function<bool(int)> & cb,
                                 std::vector<float> * first_logits = nullptr);
    bool is_eos(int tok) const {
        return tok == tk.eos_id || tok == tk.eot_id;
    }

    // zero the recurrent state (GDN + conv) of one slot (used when admitting a sequence)
    void zero_slot(int slot);

    // block allocator (host side).  alloc_block() first evicts prefix-cache
    // nodes and then commits more pool memory (up to pool_cap) before failing.
    int alloc_block();
    void free_block(int b);
    bool has_free_block() const {
        return !free_blocks_.empty();
    }
    void set_table(int slot, const std::vector<int> & blocks);

    // block pool accounting (host side, informational)
    int pool_free_blocks() const {
        return (int)free_blocks_.size();
    }
    int pool_used_blocks() const {
        return pool_blocks - (int)free_blocks_.size();
    }
    void pool_print(const char * tag) const;
    // bytes of one block in one attention layer, per side (element-size aware)
    size_t kv_block_bytes() const;
    size_t kv_v_block_bytes() const;
    size_t kv_elem_bytes() const {
        return (size_t)kv_dtype_bytes(kv_k_dtype());
    }
    // committed and reserved K+V bytes over all attention layers
    size_t kv_bytes_total() const {
        return (kv_block_bytes() + kv_v_block_bytes()) * (size_t)attn_layers() * pool_blocks;
    }
    size_t kv_bytes_cap() const {
        return (kv_block_bytes() + kv_v_block_bytes()) * (size_t)attn_layers() * pool_cap;
    }
    // copy `n` elements of one KV vector out of the pool into fp32 (converts
    // the storage type; used by the stage tests)
    void kv_read_vec(int which, size_t elem_off, float * dst, int n);

    // ---- prefix cache (PF_PREFIX_CACHE=1) -------------------------------
    // A cache node is one full 32-token block of a prefilled prompt, keyed by a
    // chained hash of the parent node and the block's token ids.  Each node may
    // carry a checkpoint of the recurrent state (GDN + conv, ~19 MB) at its
    // boundary; without it a node is useless for this hybrid model, so only
    // nodes whose checkpoint exists are matched on admit.
    // All entry points assume the caller holds `mtx`.
    bool pc_enabled = false;
    int pc_max_states = 0; // checkpoint budget (nodes <= pc_max_states)
    uint64_t pc_stat_hits = 0, pc_stat_misses = 0, pc_stat_tokens = 0;
    uint64_t pc_stat_states = 0, pc_stat_evict_nodes = 0, pc_stat_evict_states = 0;
    // ---- disk tier (PF_PC_DIR / --pc-dir) --------------------------------
    // A lookup walks the tiers in order VRAM -> RAM -> disk; an eviction
    // demotes the node the other way (VRAM -> RAM -> disk -> dropped).  The
    // RAM and disk tiers keep the serialized record and are LRU bounded; VRAM
    // is bounded by the checkpoint pool plus the KV block pool.
    std::string pc_dir;       // disk base directory (empty = no disk tier)
    size_t pc_vram_bytes = 0; // VRAM budget the checkpoint pool was sized from
    size_t pc_disk_bytes = 0; // disk budget (0 = unbounded once enabled)
    size_t pc_ram_bytes = 0;  // host-RAM budget (0 = no RAM tier)
    bool pcd_enabled = false; // pc_dir set and the store opened
    bool pcr_enabled = false; // RAM tier open
    std::unique_ptr<pc_disk_store> pcd;
    std::unique_ptr<pc_ram_store> pcr;
    uint64_t pc_stat_spills = 0, pc_stat_loads = 0;                     // to/from disk
    uint64_t pc_stat_ram_stores = 0, pc_stat_ram_loads = 0, pc_stat_ram_spills = 0;
    // open the model-scoped directory and load the record index (no data)
    void pc_disk_init();
    void pc_ram_init();
    // on graceful shutdown move every resident VRAM node and all RAM records
    // into the disk tier (no-op without a disk directory)
    void pc_flush_to_disk();
    // match `prompt` against the cache, restore the recurrent state and append
    // the matched blocks to `blocks`; returns the matched token count (0 if the
    // slot must be zeroed by the caller)
    int pc_admit(int slot, const std::vector<int> & prompt, std::vector<int> & blocks);
    // reserve checkpoint slots for the block boundaries this forward completes
    // (called from prefill_chunk/prefill_batch); the conv/GDN kernels write the
    // snapshots, pc_commit links them to the cache nodes afterwards.
    // `tok_off` is where token position `pos0` lives in `toks`.
    void pc_capture_begin(int slot, const std::vector<int> & toks, int tok_off, int pos0, int n);
    // register the blocks of the just-prefilled prompt [0, done) as cache nodes
    // and attach the checkpoint slots captured by the forward
    void pc_commit(int slot, const std::vector<int> & toks, const std::vector<int> & blocks, int done);
    // release a retiring sequence's blocks: cached nodes lose a reference, the
    // rest go back to the pool
    void pc_retire(int slot, const std::vector<int> & blocks);
    void pc_print_stats(const char * tag) const;
    bool pc_on() const {
        return pc_enabled;
    }
    int pc_nodes() const {
        return (int)pc_nodes_.size();
    }
    size_t pc_disk_records() const {
        return (pcd_enabled && pcd) ? pcd->count() : 0;
    }
    size_t pc_ram_records() const {
        return (pcr_enabled && pcr) ? pcr->count() : 0;
    }

private:
    std::priority_queue<int, std::vector<int>, std::greater<int>> free_blocks_;
    void alloc_buffers();
    void build_graphs();
    void build_plans();
    // allocate `bytes` on the active backend's memory (device USM for GPU, host
    // USM for the CPU backend) so the same engine scratch works for both
    void * alloc_bytes(size_t bytes);
    // allocate on a specific backend's memory (device USM for that GPU, host USM
    // for the CPU backend); used by the multi-device KV pools
    void * dev_alloc_on(int dev, size_t bytes);
    void setup_multi_device(const std::string & layer_map);
    // multi-device: upload only the tensors of the layers placed on `dev` (plus
    // the global tensors that always run on the primary device) into that
    // backend's weight_maps_ entry, so the GPU memory holds just its partition
    void upload_device_weights(int dev);
    // after every weight tensor has been copied/converted to its device, drop
    // the GGUF mmap's host-resident pages from this process (the mapping stays
    // valid; CPU partitions keep theirs because their kernels read the mmap)
    void release_host_weight_pages();
    template <typename T> T * alloc_elems(size_t n) {
        return (T *)alloc_bytes(n * sizeof(T));
    }
    seg_plan build_plan(int T, int tb, bool head_batched, bool use_w8 = false, bool with_head = true);

    // ---- multi-device decode command graphs --------------------------------
    // The multi-device path replays record_forward directly, so a single-token
    // decode pays the per-kernel dispatch latency of ~700 submissions per step
    // (measured ~9 ms of a 62 ms step).  The single-device path hides that
    // behind a SYCL command graph; this does the same per partition: one graph
    // per contiguous device run of the layer loop, plus the embedding on the
    // first phase and the output norm + LM head on a final primary-device
    // phase, replayed in order with the existing host-staged handoff between
    // them.  Only the batch-1 decode plan is graphed; batched decode and
    // prefill keep the direct replay.
    struct md_phase {
        int dev = 0;
        int l0 = 0, l1 = 0; // layer range [l0, l1)
        bool embed = false; // embedding first (primary device)
        bool head = false;  // output norm + LM head after (primary device)
    };
    struct md_cmd_graph {
        md_phase ph;
        std::unique_ptr<sx::command_graph<sx::graph_state::modifiable>> g;
        std::unique_ptr<sx::command_graph<sx::graph_state::executable>> e;
    };
    std::vector<md_cmd_graph> md_dec_;
    bool md_dec_ok = false;
    // The MTP verify's own command graphs.  Same phase split as the decode
    // (one graph per contiguous device run, host handoff between), but the
    // recorded pass is the mode-2 batch over k+1 rows: a fixed shape, so it is
    // recorded once and replayed every cycle.  See build_md_verify_graphs().
    std::vector<md_cmd_graph> vf_dec_;
    bool vf_dec_ok = false;
    int vf_dec_rows = 0;

    // Multi-device prefill pipeline: the layer split into contiguous device runs
    // (device 0 with the embedding, device 1, device 0 with the head).  When the
    // map is exactly this two-GPU-partition shape, prefill_batch overlaps chunk
    // i+1 on device 0 with chunk i on device 1 instead of running them strictly
    // one after the other (T0+T1 -> max(T0,T1)).
    std::vector<md_phase> md_pf_phases_;
    bool pf_pipe_ok_ = false;
    int pf_pipe_count_ = 0; // parity source for the double-buffered step_info
    void build_md_dec_graphs();
    void replay_md_dec_graphs();
    void build_md_verify_graphs();
    void replay_md_verify_graphs();
    // true when the recorded verify graphs are usable for a `rows`-row pass at
    // `pos0`; false falls back to the direct replay.
    bool vf_graph_usable(int rows, int pos0) const;

    void record_forward(int mode, const seg_plan & plan, gemv_seg * d_segs, int rows, gemv_seg * d_segs_rows = nullptr,
                        int at_nsp_hint = 0, const md_phase * ph = nullptr, const step_info * info = nullptr);

    std::vector<float> run_head();

    // shared body of generate/generate_mm (mm == nullptr for text-only prompts)
    std::vector<int> generate_impl(const std::vector<int> & prompt, const mm_prompt * mm, const gen_params & gp,
                                   const std::function<bool(int)> & cb, std::vector<float> * first_logits);

    // dynamic pool internals
    struct kv_extent {
        int first = 0; // first block id covered
        int count = 0; // blocks covered
        size_t map_bytes = 0;   // K mapping size
        size_t map_v_bytes = 0; // V mapping size (== map_bytes unless --kv-type K:V mixed)
        // one physical allocation per (attention layer, K/V): a physical_mem
        // object can only be mapped once
        std::vector<std::unique_ptr<sx::physical_mem>> phys_k, phys_v;
    };
    std::vector<kv_extent> kv_extents_;
    std::vector<int> extent_used_;    // blocks in use per extent
    std::vector<uint8_t> block_used_; // [n_blocks] 1 = allocated
    std::vector<int> block_extent_;   // [n_blocks] extent index or -1
    int kv_align_blocks = 32;         // blocks per 2 MB mapping granule
    void kv_setup(int n_attn, int initial_blocks);
    bool kv_grow(int want);
    void kv_shrink();
    void kv_release_pool();
    void pool_note_peak() {
        if ((uint64_t)pool_used_blocks() > pool_peak) {
            pool_peak = pool_used_blocks();
        }
    }
    int attn_layers() const;
    // base pointers of global attention layer `a`'s K/V storage (+ int8 scale
    // planes, null when the kv type has none): the single pool on the normal
    // path, the owning device's pool in multi-device mode (block ids are global,
    // so `block` offsets within each device's local layer stride are the same)
    void kv_layer_ptrs(int a, const char *& kp, const char *& vp, const char *& ksc, const char *& vsc) const;
    // backend index that owns global attention layer `a` (0 when single-device)
    int attn_dev(int a) const;
    // queue whose device owns backend `d` (falls back to the primary queue for
    // CPU partitions and the single-device path)
    sycl::queue & dev_queue(int d) {
        if (multi_dev && d >= 0 && d < (int)dev_queues_.size() && dev_queues_[(size_t)d] && dev_kind_[(size_t)d] == 0) {
            return *dev_queues_[(size_t)d];
        }
        return q;
    }

    // prefix cache internals
    struct pc_node {
        uint64_t hash = 0;
        int32_t block = -1;
        int32_t refcount = 0; // live sequences holding this block
        uint64_t lru = 0;
        int32_t state = -1;       // checkpoint slot or -1
        int32_t depth = 0;        // blocks in this node's hash chain (1 = root)
        int32_t toks[kBlockSize]; // token ids (exact verification on lookup)
    };
    std::vector<pc_node> pc_nodes_;
    std::unordered_map<uint64_t, int32_t> pc_map_; // chained hash -> node
    std::vector<int32_t> pc_block_node_;           // block id -> node or -1
    uint64_t pc_clock = 0;
    float * d_pc_states = nullptr; // [pc_max_states][pc_state_floats]
    size_t pc_state_floats = 0;
    std::vector<int32_t> pc_state_owner; // checkpoint slot -> node
    std::vector<int32_t> pc_state_free;
    std::vector<uint64_t> pc_state_stamp;
    // checkpoint slots reserved for the *current* forward, linked to nodes by
    // pc_commit: {node hash, checkpoint slot}
    std::vector<std::pair<uint64_t, int32_t>> pc_pending_;
    struct pc_slot {
        uint64_t chain = 0;     // chained hash through the last registered block
        int32_t registered = 0; // number of leading blocks that are cache nodes
        bool tracking = false;  // pc_admit ran for this slot: capture snapshots
    };
    std::array<pc_slot, kMaxB> pc_slot_;

    static uint64_t pc_hash_block(uint64_t chain, const int32_t * toks);
    int pc_find(uint64_t chain, const int32_t * toks) const; // node index or -1
    int pc_add_node(uint64_t chain, const int32_t * toks, int block, int depth);
    void pc_restore_state(int slot, int st);
    int pc_state_take();           // free or LRU-evicted checkpoint slot, -1 if none
    void pc_state_drop(int st);    // drop the checkpoint of its owner node
    void pc_state_release(int st); // unowned reserved slot back to the free list
    int pc_evict_lru();            // evict one unreferenced node; 1 = evicted
    void pc_evict_node(int ni);
    // disk tier helpers
    size_t pc_block_blob_bytes() const; // serialized K/V(+scales) of one block
    void pc_serialize_block(int block, std::vector<uint8_t> & blob);
    void pc_deserialize_block(const uint8_t * blob, int block);
    void pc_serialize_state(int st, std::vector<float> & out);
    void pc_deserialize_state(const float * in, int st);
    // demote a node to the next enabled tier (RAM, else disk); returns true
    // when the record is stored in a lower tier
    bool pc_demote_node(int ni);
    // write a node straight to disk, bypassing the RAM tier (flush path)
    bool pc_node_to_disk(int ni);
    void pc_ram_to_disk(pc_ram_entry e);
    // promote a serialized record into a fresh VRAM node; returns the node or -1
    int pc_promote_bytes(const pc_disk_meta & meta, const uint8_t * blob, const float * state);
    int pc_promote_ram(uint64_t hash, const int32_t * toks);
    int pc_promote_disk(pc_disk_meta meta);
    // attach a checkpoint that only a lower tier holds to an existing VRAM node
    bool pc_attach_state_from_lower(int ni, uint64_t hash, const int32_t * toks);
    size_t gdn_per_slot() const;
    size_t conv_per_slot() const;
};

} // namespace si
