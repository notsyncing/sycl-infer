// Expert residency and caching for the routed MoE half.
//
// The routed experts are the bulk of a MoE model's weight bytes (18 GiB of the
// 35B A3B) and a request only ever touches n_expert_used of n_expert per token,
// so which experts are resident is the whole memory story.  The device kernels
// already read a per-(layer, expert) *slot table* and skip a slot of -1, so
// everything here is about deciding that table and moving planes between tiers.
//
// Three tiers, cheapest to fill first:
//
//   disk  the GGUF mmap, always present, no budget - the GGUF blocks are the
//         ground truth and a miss here is a page fault, not a read
//   RAM   an explicit host-side cache of *decoded* expert weights, so the hot
//         experts stop re-dequantizing from blocks on every token
//   VRAM  the device bank: one slot per resident expert, shared by every layer
//         whose plane size is the same
//
// A residency set smaller than n_expert per layer is also what makes the
// ncmoe-style split work: the resident set is the *fixed* GPU half, and the
// pairs whose expert is not resident are the *variable* half, which the engine
// computes on the host.
#pragma once

#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "env.h"
#include "gguf.h"
#include "model.h"

namespace si {

struct moe_cache_cfg {
    // experts kept on the device per layer (0 = every expert resident, the
    // default and the current behaviour)
    int resident_per_layer = 0;
    // device budget for the expert bank; 0 = derived from resident_per_layer
    size_t vram_mb = 0;
    // host budget for the decoded-expert tier; 0 = off
    size_t ram_mb = 0;
    // keep the decoded copy in RAM for experts that were evicted (the disk tier
    // is always there underneath, so this only has to be set if the RAM tier is
    // actually wanted)
    bool ram_tier = false;

    static moe_cache_cfg from_env(int resident_per_layer);
};

class moe_cache {
public:
    void init(const model & m, const moe_cache_cfg & cfg, int n_slots_needed);

    bool active() const {
        return cfg_.resident_per_layer > 0 || cfg_.vram_mb > 0 || cfg_.ram_tier;
    }

    // Per layer: which experts are resident and in which device slot.  Rows of
    // the slot table handed to the kernels are [n_layer+1][n_expert] so the
    // NextN block is just the last one.
    void slotmap(const std::vector<moe_t> & layers, std::vector<int32_t> & out) const;

    // Record that `expert` of `layer` was selected this step.  The residency set
    // is frozen once `frozen_` is set, so the bank does not change under a
    // command graph.
    void note(int layer, int expert);
    void freeze() {
        frozen_ = true;
    }
    bool frozen() const {
        return frozen_;
    }
    // true when every expert of every layer is resident, i.e. the host half of
    // the split has nothing to do
    bool all_resident() const {
        return resident_ >= n_expert_;
    }
    int resident() const {
        return resident_;
    }
    int n_expert() const {
        return n_expert_;
    }

    // How many (layer, expert) pairs the RAM tier is currently holding.
    size_t ram_bytes() const {
        return ram_bytes_;
    }

    // Tier 1: the expert's GGUF blocks, cached on the host so the ncmoe host half
    // (and any later upload) do not have to re-fault them out of a page that
    // release_host_weight_pages() deliberately dropped.  One entry is one
    // expert's three planes, so the unit of eviction is a whole expert.
    // Returns nullptr on a miss, in which case the caller reads the mmap and
    // (if it wants the entry kept) calls ram_put().
    const char * ram(const model & m, const moe_t & M, int layer, int expert) const;
    void ram_put(const model & m, const moe_t & M, int layer, int expert);
    // (layer, expert) pairs the current slot table marks resident: the size of the
    // VRAM bank actually handed to the kernels, not the size it was configured with
    size_t resident_pairs() const {
        // resident_ is per layer, and the slot table's last row is the NextN block
        return (size_t) resident_ * (freq_.size() / (size_t) n_expert_);
    }

    size_t ram_hits() const {
        return ram_hits_;
    }
    size_t ram_misses() const {
        return ram_misses_;
    }

    const moe_cache_cfg & cfg() const {
        return cfg_;
    }

private:
    moe_cache_cfg cfg_{};
    int n_layer_ = 0;
    int n_expert_ = 0;
    int resident_ = 0; // experts per layer held on the device
    bool frozen_ = false;
    // layer*n_expert + expert -> selection count since the last rebuild
    std::vector<uint32_t> freq_;
    // device slot per (layer, expert), or -1
    std::vector<int32_t> slot_;
    size_t ram_bytes_ = 0;
    mutable size_t ram_hits_ = 0, ram_misses_ = 0; // a lookup must not mutate the policy
    // one entry per (layer, expert): the three planes as raw GGUF bytes.  Held
    // as offsets into a single arena so a eviction is a pop, not a free per plane.
    struct ram_entry {
        uint64_t key = 0;
        std::vector<uint8_t> bytes;
        uint32_t used = 0;
    };
    std::list<ram_entry> ram_lru_;
    std::unordered_map<uint64_t, std::list<ram_entry>::iterator> ram_idx_;
};

} // namespace si