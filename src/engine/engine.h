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
#include "kernels.h"
#include "model.h"
#include "multimodal.h"
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
    // DP4A path: activation quantization needed before each call
    struct xq_t {
        const float * x = nullptr;
        const float * up = nullptr;
        int x_stride = 0;
        int up_stride = 0;
        int K = 0;
    };
    std::vector<xq_t> call_xq; // one entry per call (empty = activations already fp32)
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
    void finalize() {
        for (size_t ci = 0; ci < call_offsets.size(); ci++) {
            const int off = (int)call_offsets[ci];
            const int n = call_counts[ci];
            std::stable_sort(segs.begin() + off, segs.begin() + off + n,
                             [](const gemv_seg & a, const gemv_seg & b) { return a.type < b.type; });
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
    sycl::queue q;
    int max_seq;                 // max tokens per sequence (limited by the block pool)
    int n_splits;                // prefill K-split ceiling (PF_ATTN_SPLIT overrides)
    int dec_splits = kMaxSplits; // decode K-split (PF_DEC_SPLIT overrides)
    int ffn_stride;              // 2 * n_ff
    int n_blocks;                // reserved KV blocks (virtual address range size)
    int max_blocks;              // block table size per sequence
    size_t kv_layer_stride;

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
    // type is kv_dtype() (f32/bf16/f16, see kernels.h); sizes are in elements
    // of that type and all pointers are advanced in bytes.
    void * d_kpool = nullptr; // [n_attn][n_blocks][n_head_kv][kBlockSize][head_dim]
    void * d_vpool = nullptr;
    // int8 only: per-32 fp16 scale planes, same [attn][block][kv head][token][group]
    // indexing, allocated with the same block count as the pools
    void * d_kscales = nullptr;
    void * d_vscales = nullptr;
    size_t kv_scale_stride = 0;   // bytes per attention layer in a scale plane
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

    step_info * d_info = nullptr; // host USM

    // SI8 DP4A scratch: quantized activations of the current call (kMaxT rows)
    int8_t * d_x8 = nullptr;
    sycl::float2 * d_xmeta = nullptr; // activation scales (fp32)
    int32_t * d_xsumq = nullptr;

    gemv_seg * d_segs_dec = nullptr;
    gemv_seg * d_segs_pf = nullptr;
    gemv_seg * d_segs_pf8 = nullptr;
    gemv_seg * d_segs_aux = nullptr;
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
        if (use_dnnl) {
            int n = (rem / kMaxT) * kMaxT;
            if (n > kMaxB * kMaxT) {
                n = kMaxB * kMaxT;
            }
            return n >= 2 * kMaxT ? n : 0;
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
    std::unique_ptr<sx::command_graph<sx::graph_state::executable>> e_dec, e_pf;

    seg_plan plan_dec_, plan_pf_, plan_pf8_, plan_dec8_;

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
           int kv_cap_mb = 0);
    ~engine();

    void reset_state();

    // ---- both entry points assume the caller holds `mtx` ----
    // process `n` tokens of one sequence starting at `start` (n <= kMaxT); the
    // sequence's KV blocks must already be assigned in its table row `slot`
    void prefill_chunk(const std::vector<int> & toks, int start, int n, int slot, bool with_head = true);
    // experimental (dev/bench_pfb*): chunk-batched prefill, one row per kMaxT
    // chunk.  Correct only via the direct (PF_NOGRAPH) path and currently
    // slower than the chunked path - kept as the basis for an M-tiled GEMM.
    void prefill_batch(const std::vector<int> & toks, int start, int n, int slot, int pos0);
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
    // K bytes of one block in one attention layer (element-size aware)
    size_t kv_block_bytes() const;
    size_t kv_elem_bytes() const {
        return (size_t)kv_dtype_bytes(kv_dtype());
    }
    // committed and reserved K+V bytes over all attention layers
    size_t kv_bytes_total() const {
        return kv_block_bytes() * (size_t)attn_layers() * 2 * pool_blocks;
    }
    size_t kv_bytes_cap() const {
        return kv_block_bytes() * (size_t)attn_layers() * 2 * pool_cap;
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

private:
    std::priority_queue<int, std::vector<int>, std::greater<int>> free_blocks_;
    void alloc_buffers();
    void build_graphs();
    seg_plan build_plan(int T, int tb, bool head_batched, bool use_w8 = false, bool with_head = true) const;
    void record_forward(int mode, const seg_plan & plan, gemv_seg * d_segs, int rows, gemv_seg * d_segs_rows = nullptr,
                        int at_nsp_hint = 0);

    std::vector<float> run_head();

    // shared body of generate/generate_mm (mm == nullptr for text-only prompts)
    std::vector<int> generate_impl(const std::vector<int> & prompt, const mm_prompt * mm, const gen_params & gp,
                                   const std::function<bool(int)> & cb, std::vector<float> * first_logits);

    // dynamic pool internals
    struct kv_extent {
        int first = 0; // first block id covered
        int count = 0; // blocks covered
        size_t map_bytes = 0;
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
    size_t gdn_per_slot() const;
    size_t conv_per_slot() const;
};

} // namespace si
