#include "engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

size_t engine::gdn_per_slot() const {
    const hparams & hp = m.hp;
    return (size_t)hp.dt_rank * hp.d_state * hp.d_state;
}

size_t engine::conv_per_slot() const {
    const hparams & hp = m.hp;
    return (size_t)(hp.conv_k - 1) * 3 * hp.d_inner;
}

// FNV-1a over the block's token ids, chained onto the parent hash
uint64_t engine::pc_hash_block(uint64_t chain, const int32_t * toks) {
    uint64_t h = chain ? chain : 0xcbf29ce484222325ull;
    for (int i = 0; i < kBlockSize; i++) {
        const uint32_t v = (uint32_t)toks[i];
        for (int b = 0; b < 4; b++) {
            h ^= (v >> (8 * b)) & 0xffu;
            h *= 0x100000001b3ull;
        }
    }
    return h;
}

int engine::pc_find(uint64_t chain, const int32_t * toks) const {
    auto it = pc_map_.find(chain);
    if (it == pc_map_.end()) {
        return -1;
    }
    if (std::memcmp(pc_nodes_[it->second].toks, toks, sizeof(pc_node::toks)) != 0) {
        return -1;
    }
    return it->second;
}

int engine::pc_add_node(uint64_t chain, const int32_t * toks, int block, int depth) {
    if (pc_map_.count(chain)) {
        return -1;
    }
    if (block < 0 || block >= (int)pc_block_node_.size()) {
        return -1;
    }
    if (pc_block_node_[block] >= 0) {
        return -1;
    }
    pc_node nd;
    nd.hash = chain;
    nd.block = block;
    nd.refcount = 1; // the creating sequence holds the block
    nd.lru = ++pc_clock;
    nd.depth = depth;
    std::memcpy(nd.toks, toks, sizeof(nd.toks));
    const int ni = (int)pc_nodes_.size();
    pc_nodes_.push_back(nd);
    pc_map_[chain] = ni;
    pc_block_node_[block] = ni;
    return ni;
}

void engine::pc_restore_state(int slot, int st) {
    const hparams & hp = m.hp;
    const size_t gp = gdn_per_slot(), cp = conv_per_slot(), per = gp + cp;
    const float * src = d_pc_states + (size_t)st * pc_state_floats;
    int gi = 0;
    for (int il = 0; il < hp.n_layer; il++) {
        if (!hp.is_recr(il)) {
            continue;
        }
        q.memcpy(d_gdn_state + ((size_t)gi * kMaxB + slot) * gp, src + (size_t)gi * per, gp * 4);
        q.memcpy(d_conv_state + ((size_t)gi * kMaxB + slot) * cp, src + (size_t)gi * per + gp, cp * 4);
        gi++;
    }
}

int engine::pc_state_take() {
    if (pc_max_states <= 0 || !d_pc_states) {
        return -1;
    }
    if (!pc_state_free.empty()) {
        const int st = pc_state_free.back();
        pc_state_free.pop_back();
        return st;
    }
    // Budget exhausted: evict the least recently used committed checkpoint (a
    // fresh capture is the most recent by construction, so a new forward
    // displaces the oldest reuse point).  A slot reserved for the current
    // forward has no owner yet and is never evicted.
    int best = -1;
    uint64_t best_stamp = UINT64_MAX;
    for (int st = 0; st < pc_max_states; st++) {
        const int ni = pc_state_owner[st];
        if (ni < 0 || ni >= (int)pc_nodes_.size()) {
            continue;
        }
        if (pc_state_stamp[st] < best_stamp) {
            best_stamp = pc_state_stamp[st];
            best = st;
        }
    }
    if (best < 0) {
        return -1;
    }
    const int ni = pc_state_owner[best];
    pc_nodes_[ni].state = -1;
    pc_state_owner[best] = -1;
    pc_stat_evict_states++;
    return best;
}

void engine::pc_state_drop(int st) {
    if (st < 0 || st >= pc_max_states) {
        return;
    }
    const int ni = pc_state_owner[st];
    if (ni >= 0 && ni < (int)pc_nodes_.size()) {
        pc_nodes_[ni].state = -1;
    }
    pc_state_owner[st] = -1;
    pc_state_free.push_back(st);
}

// a slot reserved by pc_capture_begin whose forward never reached pc_commit
void engine::pc_state_release(int st) {
    if (st < 0 || st >= pc_max_states) {
        return;
    }
    if (pc_state_owner[st] >= 0) {
        pc_state_drop(st);
        return;
    }
    pc_state_free.push_back(st);
}

void engine::pc_evict_node(int ni) {
    const pc_node nd = pc_nodes_[ni];
    if (nd.state >= 0) {
        pc_state_drop(nd.state);
    }
    if (nd.block >= 0 && nd.block < (int)pc_block_node_.size()) {
        pc_block_node_[nd.block] = -1;
    }
    pc_map_.erase(nd.hash);
    const int last = (int)pc_nodes_.size() - 1;
    if (ni != last) {
        pc_nodes_[ni] = pc_nodes_[last];
        pc_map_[pc_nodes_[ni].hash] = ni;
        if (pc_nodes_[ni].state >= 0) {
            pc_state_owner[pc_nodes_[ni].state] = ni;
        }
    }
    pc_nodes_.pop_back();
    free_block(nd.block);
}

int engine::pc_evict_lru() {
    int best = -1;
    uint64_t best_stamp = UINT64_MAX;
    for (size_t i = 0; i < pc_nodes_.size(); i++) {
        const pc_node & nd = pc_nodes_[i];
        if (nd.refcount > 0 || nd.block < 0) {
            continue;
        }
        if (nd.lru < best_stamp) {
            best_stamp = nd.lru;
            best = (int)i;
        }
    }
    if (best < 0) {
        return 0;
    }
    pc_evict_node(best);
    pc_stat_evict_nodes++;
    return 1;
}

int engine::pc_admit(int slot, const std::vector<int> & prompt, std::vector<int> & blocks) {
    static const bool pcdbg = getenv("PF_PC_DEBUG") != nullptr;
    if (slot >= 0 && slot < kMaxB) {
        pc_slot_[slot] = {};
    }
    if (!pc_enabled) {
        return 0;
    }
    pc_slot_[slot].tracking = true; // prefill of this slot captures checkpoints
    const int L = (int)prompt.size();
    // keep the last partial block (and at least one token) for the prefill:
    // the first sampled token needs the logits of a real forward pass
    const int max_full = ((L - 1) / kBlockSize) * kBlockSize;
    int matched = 0, state = -1;
    uint64_t chain = 0;
    for (int i = 0; i * kBlockSize < max_full; i++) {
        const int32_t * tk = prompt.data() + (size_t)i * kBlockSize;
        const uint64_t h = pc_hash_block(chain, tk);
        const int ni = pc_find(h, tk);
        if (pcdbg) {
            fprintf(stderr, "[pcdbg] admit L=%d blk=%d h=%llx ni=%d\n", L, i, (unsigned long long)h, ni);
        }
        if (ni < 0) {
            break;
        }
        chain = h;
        // a checkpoint on this node is a valid resume point after i+1 blocks;
        // nodes without one (state budget) just cannot be resumed *there*, so
        // remember the deepest one that can
        if (pc_nodes_[ni].state >= 0) {
            matched = (i + 1) * kBlockSize;
            state = pc_nodes_[ni].state;
        }
    }
    if (matched <= 0 || state < 0) {
        pc_stat_misses++;
        return 0;
    }
    uint64_t c = 0;
    for (int i = 0; i < matched / kBlockSize; i++) {
        const int32_t * tk = prompt.data() + (size_t)i * kBlockSize;
        c = pc_hash_block(c, tk);
        const int ni = pc_find(c, tk);
        pc_nodes_[ni].refcount++;
        pc_nodes_[ni].lru = ++pc_clock;
        blocks.push_back(pc_nodes_[ni].block);
    }
    pc_restore_state(slot, state);
    if (state >= 0) {
        pc_state_stamp[state] = pc_clock;
    }
    pc_slot_[slot].chain = chain;
    pc_slot_[slot].registered = matched / kBlockSize;
    pc_stat_hits++;
    pc_stat_tokens += matched;
    return matched;
}

// Reserve checkpoint slots for the block boundaries this forward will complete
// and fill the kernel snapshot map.  Called from prefill_chunk/prefill_batch
// before the forward; the map is read by the conv/GDN kernels, which write the
// state at each marked boundary straight into the checkpoint pool.  pc_commit
// links the slots to the cache nodes afterwards.
void engine::pc_capture_begin(int slot, const std::vector<int> & toks, int tok_off, int pos0, int n) {
    // a previous forward's pending reservations (no commit) must not leak
    for (auto & p : pc_pending_) {
        pc_state_release(p.second);
    }
    pc_pending_.clear();
    d_info->pc_active = 0;
    if (!pc_enabled || slot < 0 || slot >= kMaxB) {
        return;
    }
    pc_slot & ps = pc_slot_[slot];
    if (!ps.tracking) {
        return;
    }
    if (n <= 0 || pos0 < 0 || pos0 % kBlockSize != 0) {
        return;
    }
    // every preceding block must already be registered: the chained hash (and
    // thus the node identity) is only known from a contiguous chain
    if ((int)ps.registered * kBlockSize != pos0) {
        return;
    }
    const int first = pos0 / kBlockSize;      // first block index of this forward
    const int last = (pos0 + n) / kBlockSize; // boundaries first+1 .. last
    if (last <= first || last >= kPcMapLen) {
        return;
    }
    for (int b = first + 1; b <= last; b++) {
        d_info->pc_row_slot[b] = -1;
    }

    // find the boundaries whose node cannot be resumed yet
    struct cand {
        int b;
        uint64_t h;
    };
    const int cap = std::min<int>(kMaxB * kMaxT / kBlockSize + 1, kPcMapLen / 4);
    std::vector<cand> need;
    need.reserve(std::min(last - first, cap));
    uint64_t chain = ps.chain;
    for (int b = first + 1; b <= last; b++) {
        const int32_t * tk = toks.data() + (size_t)tok_off + (size_t)(b - 1 - first) * kBlockSize;
        chain = pc_hash_block(chain, tk);
        const int ni = pc_find(chain, tk);
        if (ni < 0 || pc_nodes_[ni].state < 0) {
            need.push_back({b, chain});
        }
    }
    const int nneed = (int)need.size();
    if (nneed == 0) {
        return;
    }
    // Deterministic and bounded: keep the deepest min(need, budget) boundaries
    // of this forward, because a later request resumes at the deepest matching
    // boundary.  pc_state_take() evicts the LRU checkpoint when the pool is
    // full; iterate shallow->deep so the deepest slot is the most recent.
    const int nsel = std::min(nneed, std::max(pc_max_states, 0));
    for (int k = nneed - nsel; k < nneed; k++) {
        const int st = pc_state_take();
        if (st < 0) {
            break;
        }
        pc_state_stamp[st] = ++pc_clock;
        d_info->pc_row_slot[need[k].b] = st;
        pc_pending_.push_back({need[k].h, st});
    }
    if (pc_pending_.empty()) {
        return;
    }
    d_info->pc_base = d_pc_states;
    d_info->pc_stride = (int32_t)pc_state_floats;
    d_info->pc_active = 1;
}

void engine::pc_commit(int slot, const std::vector<int> & toks, const std::vector<int> & blocks, int done) {
    static const bool pcdbg = getenv("PF_PC_DEBUG") != nullptr;
    d_info->pc_active = 0; // the captured forward is over
    if (!pc_enabled || slot < 0 || slot >= kMaxB) {
        for (auto & p : pc_pending_) {
            pc_state_release(p.second);
        }
        pc_pending_.clear();
        return;
    }
    pc_slot & ps = pc_slot_[slot];
    const int full = done / kBlockSize;
    for (int i = ps.registered; i < full && i < (int)blocks.size(); i++) {
        const int32_t * tk = toks.data() + (size_t)i * kBlockSize;
        const uint64_t h = pc_hash_block(ps.chain, tk);
        const int found = pc_find(h, tk);
        if (pcdbg) {
            fprintf(stderr, "[pcdbg] commit slot=%d blk=%d done=%d h=%llx found=%d\n", slot, i, done,
                    (unsigned long long)h, found);
        }
        if (found < 0) {
            pc_add_node(h, tk, blocks[i], i + 1);
        }
        ps.chain = h;
        ps.registered = i + 1;
    }
    // attach the checkpoints captured by this forward
    for (auto & p : pc_pending_) {
        auto it = pc_map_.find(p.first);
        const int ni = it == pc_map_.end() ? -1 : it->second;
        if (ni >= 0 && pc_nodes_[ni].state < 0) {
            pc_nodes_[ni].state = p.second;
            pc_state_owner[p.second] = ni;
            pc_state_stamp[p.second] = ++pc_clock;
            pc_nodes_[ni].lru = pc_clock;
            pc_stat_states++;
        } else {
            pc_state_release(p.second);
        }
    }
    pc_pending_.clear();
}

void engine::pc_retire(int slot, const std::vector<int> & blocks) {
    if (slot >= 0 && slot < kMaxB) {
        pc_slot_[slot] = {};
    }
    for (int b : blocks) {
        const int ni = (b >= 0 && b < (int)pc_block_node_.size()) ? pc_block_node_[b] : -1;
        if (ni >= 0 && pc_nodes_[ni].block == b) {
            if (--pc_nodes_[ni].refcount <= 0) {
                pc_nodes_[ni].refcount = 0;
                pc_nodes_[ni].lru = ++pc_clock; // now evictable
            }
        } else {
            free_block(b);
        }
    }
}

void engine::pc_print_stats(const char * tag) const {
    if (!pc_enabled) {
        return;
    }
    int with_state = 0;
    uint64_t refs = 0;
    for (const auto & nd : pc_nodes_) {
        if (nd.state >= 0) {
            with_state++;
        }
        refs += (uint64_t)std::max(nd.refcount, 0);
    }
    const double state_mb = (double)pc_state_floats * 4.0 / (1024.0 * 1024.0);
    const uint64_t lookups = pc_stat_hits + pc_stat_misses;
    fprintf(stderr,
            "[pc] %s: nodes=%zu (with state %d), refs=%llu, hits=%llu/%llu (%.0f%%), reused=%llu tok, "
            "captured=%llu, evicted nodes=%llu states=%llu, checkpoint=%.1f MB x %d = %.0f MB\n",
            tag, pc_nodes_.size(), with_state, (unsigned long long)refs, (unsigned long long)pc_stat_hits,
            (unsigned long long)lookups, lookups ? 100.0 * (double)pc_stat_hits / (double)lookups : 0.0,
            (unsigned long long)pc_stat_tokens, (unsigned long long)pc_stat_states,
            (unsigned long long)pc_stat_evict_nodes, (unsigned long long)pc_stat_evict_states, state_mb, pc_max_states,
            state_mb * pc_max_states);
}

} // namespace si
