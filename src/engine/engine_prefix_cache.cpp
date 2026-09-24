#include "engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

static_assert(kPcBlockToks == kBlockSize, "prefix-cache block size changed");

size_t engine::gdn_per_slot() const {
    const hparams & hp = m.hp;
    return (size_t)hp.dt_rank * hp.d_state * hp.d_state;
}

size_t engine::conv_per_slot() const {
    const hparams & hp = m.hp;
    return (size_t)(hp.conv_k - 1) * hp.qkv_dim();
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

// ---- disk tier ------------------------------------------------------------
// Open <dir>/<model-fingerprint> and rebuild the record index from the file
// headers (metadata only; block/state bytes are read on demand).  The
// fingerprint covers the model identity and every shape that decides the
// serialized layout, so two models can share one directory safely.
void engine::pc_disk_init() {
    uint64_t h = 0xcbf29ce484222325ull;
    auto mix = [&](const void * p, size_t n) {
        const uint8_t * b = (const uint8_t *)p;
        for (size_t i = 0; i < n; i++) {
            h ^= b[i];
            h *= 0x100000001b3ull;
        }
    };
    auto mix_u64 = [&](uint64_t v) { mix(&v, sizeof(v)); };
    auto mix_str = [&](const std::string & s) {
        mix(s.data(), s.size());
        mix_u64(0x9e3779b97f4a7c15ull);
    };
    const hparams & hp = m.hp;
    mix_str("sycl-infer-pc");
    mix_u64(1); // on-disk generation (bump with a format change)
    mix_u64((uint64_t)kv_k_dtype());
    mix_u64((uint64_t)kv_v_dtype());
    mix_u64((uint64_t)kBlockSize);
    for (int v : {hp.n_layer, hp.n_embd, hp.n_ff, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_rot, hp.n_vocab,
                  hp.d_state, hp.n_group, hp.dt_rank, hp.d_inner, hp.conv_k, hp.full_attn_interval}) {
        mix_u64((uint64_t)(uint32_t)v);
    }
    for (int s : hp.rope_sections) {
        mix_u64((uint64_t)(uint32_t)s);
    }
    mix_u64((uint64_t)m.gguf.map_size);
    if (const std::string * s = m.gguf.get_str("general.architecture")) {
        mix_str(*s);
    }
    if (const std::string * s = m.gguf.get_str("general.name")) {
        mix_str(*s);
    }
    mix_u64(m.gguf.get_u32("general.file_type", 0));
    char fp[24];
    std::snprintf(fp, sizeof(fp), "%016llx", (unsigned long long)h);
    std::string dir = pc_dir;
    if (!dir.empty() && dir.back() != '/') {
        dir += '/';
    }
    dir += fp;
    pcd = std::make_unique<pc_disk_store>();
    if (!pcd->open(dir, pc_disk_bytes)) {
        pcd.reset();
        pcd_enabled = false;
        return;
    }
    pcd_enabled = pcd->active();
    if (pcd_enabled) {
        fprintf(stderr, "[pcd] disk prefix cache %s: %zu records, %.0f MB budget\n", dir.c_str(), pcd->count(),
                pc_disk_bytes ? (double)pc_disk_bytes / (1024.0 * 1024.0) : 0.0);
    }
}

void engine::pc_ram_init() {
    pcr = std::make_unique<pc_ram_store>();
    pcr->open(pc_ram_bytes);
    pcr_enabled = pcr->active();
    if (pcr_enabled) {
        fprintf(stderr, "[pcr] RAM prefix cache: %.0f MB budget\n", (double)pc_ram_bytes / (1024.0 * 1024.0));
    }
}

size_t engine::pc_block_blob_bytes() const {
    const size_t kb = kv_block_bytes();
    const size_t vb = kv_v_block_bytes();
    size_t per = kb + vb; // K + V data of one attention layer
    if (kv_dtype_has_scales(kv_k_dtype()) || kv_dtype_has_scales(kv_v_dtype())) {
        const size_t sb = (size_t)m.hp.n_head_kv * kBlockSize * (m.hp.head_dim / kI8Q) * sizeof(sycl::half);
        per += 2 * sb; // K + V scale planes
    }
    return per * (size_t)attn_layers();
}

void engine::pc_serialize_block(int block, std::vector<uint8_t> & blob) {
    const int na = attn_layers();
    const size_t kb = kv_block_bytes();
    const size_t vb = kv_v_block_bytes();
    const size_t sb = (kv_dtype_has_scales(kv_k_dtype()) || kv_dtype_has_scales(kv_v_dtype()))
                          ? (size_t)m.hp.n_head_kv * kBlockSize * (m.hp.head_dim / kI8Q) * sizeof(sycl::half)
                          : 0;
    blob.resize((kb + vb + 2 * sb) * (size_t)na);
    size_t off = 0;
    for (int l = 0; l < na; l++) {
        const char * kp, * vp, * ksc, * vsc;
        kv_layer_ptrs(l, kp, vp, ksc, vsc);
        // multi-device: the layer's pool may live on another device's queue
        sycl::queue & qd = dev_queue(attn_dev(l));
        qd.memcpy(blob.data() + off, kp + (size_t)block * kb, kb);
        off += kb;
        qd.memcpy(blob.data() + off, vp + (size_t)block * vb, vb);
        off += vb;
        if (sb) {
            qd.memcpy(blob.data() + off, ksc + (size_t)block * sb, sb);
            off += sb;
            qd.memcpy(blob.data() + off, vsc + (size_t)block * sb, sb);
            off += sb;
        }
    }
    sync_all();
}

void engine::pc_deserialize_block(const uint8_t * blob, int block) {
    const int na = attn_layers();
    const size_t kb = kv_block_bytes();
    const size_t vb = kv_v_block_bytes();
    const size_t sb = (kv_dtype_has_scales(kv_k_dtype()) || kv_dtype_has_scales(kv_v_dtype()))
                          ? (size_t)m.hp.n_head_kv * kBlockSize * (m.hp.head_dim / kI8Q) * sizeof(sycl::half)
                          : 0;
    size_t off = 0;
    for (int l = 0; l < na; l++) {
        const char * kp, * vp, * ksc, * vsc;
        kv_layer_ptrs(l, kp, vp, ksc, vsc);
        sycl::queue & qd = dev_queue(attn_dev(l));
        qd.memcpy((void *)(kp + (size_t)block * kb), blob + off, kb);
        off += kb;
        qd.memcpy((void *)(vp + (size_t)block * vb), blob + off, vb);
        off += vb;
        if (sb) {
            qd.memcpy((void *)(ksc + (size_t)block * sb), blob + off, sb);
            off += sb;
            qd.memcpy((void *)(vsc + (size_t)block * sb), blob + off, sb);
            off += sb;
        }
    }
    // the caller's blob is a temporary: the copies must complete before it dies
    sync_all();
}

void engine::pc_serialize_state(int st, std::vector<float> & out) {
    out.resize(pc_state_floats);
    q.memcpy(out.data(), d_pc_states + (size_t)st * pc_state_floats, pc_state_floats * sizeof(float)).wait();
}

void engine::pc_deserialize_state(const float * in, int st) {
    q.memcpy(d_pc_states + (size_t)st * pc_state_floats, in, pc_state_floats * sizeof(float));
    // the caller's state is a temporary: the copy must complete before it dies
    q.wait();
}

// Write a node straight to disk, bypassing the RAM tier (used by the shutdown
// flush).  Skips the copy when disk already holds a record at least as complete.
bool engine::pc_node_to_disk(int ni) {
    if (!pcd_enabled || !pcd || ni < 0 || ni >= (int)pc_nodes_.size()) {
        return false;
    }
    const pc_node & nd = pc_nodes_[ni];
    if (nd.block < 0) {
        return false;
    }
    const bool want_state = nd.state >= 0;
    if (const pc_disk_meta * e = pcd->find(nd.hash, nd.toks)) {
        if (!want_state || e->has_state) {
            pcd->touch(nd.hash);
            return true;
        }
    }
    std::vector<uint8_t> blob;
    pc_serialize_block(nd.block, blob);
    std::vector<float> state;
    if (want_state) {
        pc_serialize_state(nd.state, state);
    }
    const bool ok = pcd->store(nd.hash, nd.depth, nd.toks, blob.data(), blob.size(),
                               want_state ? state.data() : nullptr, want_state ? state.size() * sizeof(float) : 0);
    if (ok) {
        pc_stat_spills++;
    }
    return ok;
}

// Graceful-shutdown flush: move all RAM records and every resident VRAM node
// into the disk tier so a restart can resume them.  No-op without a disk dir.
void engine::pc_flush_to_disk() {
    // pools exist: single pool on the normal path, per-device pools on the
    // multi-device path
    const bool pools_ready =
        multi_dev ? std::any_of(dev_kpool_.begin(), dev_kpool_.end(), [](void * p) { return p != nullptr; })
                  : d_kpool != nullptr;
    if (!pc_enabled || !pcd_enabled || !pcd || !pools_ready) {
        return;
    }
    if (pcr_enabled && pcr) {
        std::vector<pc_ram_entry> all;
        pcr->drain(all);
        for (auto & e : all) {
            pc_ram_to_disk(std::move(e)); // RAM -> disk
        }
    }
    // oldest first, so if the disk budget evicts during the flush it drops the
    // least recently used prefixes rather than the ones just written
    std::vector<std::pair<uint64_t, int>> order;
    order.reserve(pc_nodes_.size());
    for (size_t i = 0; i < pc_nodes_.size(); i++) {
        order.push_back({pc_nodes_[i].lru, (int)i});
    }
    std::sort(order.begin(), order.end());
    for (const auto & p : order) {
        pc_node_to_disk(p.second);
    }
    fprintf(stderr, "[pcd] flushed on exit: %zu nodes, disk now %zu records (%.1f MB)\n", pc_nodes_.size(),
            pcd->count(), (double)pcd->bytes() / (1024.0 * 1024.0));
}

// Demote a VRAM node to the next enabled tier.  The record is serialized once
// and offered to RAM, whose own LRU overflow is written to disk; without a RAM
// tier it goes straight to disk.  A node already present in a lower tier (with
// at least as much data) is left alone, which makes evicting a promoted node
// free.
bool engine::pc_demote_node(int ni) {
    if (!pcr_enabled && !pcd_enabled) {
        return false;
    }
    const pc_node & nd = pc_nodes_[ni];
    if (nd.block < 0) {
        return false;
    }
    const bool want_state = nd.state >= 0;
    if (pcr_enabled) {
        if (const pc_ram_entry * e = pcr->find(nd.hash, nd.toks)) {
            if (!want_state || e->meta.has_state) {
                pcr->touch(nd.hash);
                return true;
            }
        }
    }
    if (pcd_enabled) {
        if (const pc_disk_meta * e = pcd->find(nd.hash, nd.toks)) {
            if (!want_state || e->has_state) {
                pcd->touch(nd.hash);
                return true;
            }
        }
    }
    pc_ram_entry e;
    e.meta.hash = nd.hash;
    e.meta.depth = nd.depth;
    e.meta.has_state = want_state;
    std::memcpy(e.meta.toks, nd.toks, sizeof(e.meta.toks));
    pc_serialize_block(nd.block, e.blob);
    if (want_state) {
        pc_serialize_state(nd.state, e.state);
    }
    if (pcr_enabled && pcr->put(std::move(e))) {
        pc_stat_ram_stores++;
        std::vector<pc_ram_entry> over;
        pcr->enforce_budget(over);
        for (auto & x : over) {
            pc_ram_to_disk(std::move(x)); // RAM -> disk -> (disk LRU) drop
        }
        return true;
    }
    // no RAM tier, or the record exceeds the whole RAM budget: write it to disk
    const bool ok =
        pcd_enabled && pcd->store(e.meta.hash, e.meta.depth, e.meta.toks, e.blob.data(), e.blob.size(),
                                  want_state ? e.state.data() : nullptr, want_state ? e.state.size() * sizeof(float) : 0);
    if (ok) {
        pc_stat_spills++;
    }
    return ok;
}

// RAM overflow is written to disk; with no disk tier it is dropped.
void engine::pc_ram_to_disk(pc_ram_entry e) {
    if (!pcd_enabled) {
        return;
    }
    const bool ok = pcd->store(e.meta.hash, e.meta.depth, e.meta.toks, e.blob.data(), e.blob.size(),
                               e.meta.has_state ? e.state.data() : nullptr,
                               e.meta.has_state ? e.state.size() * sizeof(float) : 0);
    if (ok) {
        pc_stat_ram_spills++;
    }
}

// Create a fresh VRAM node from an already-loaded serialized record.  The host
// buffers must stay alive across the call (the device copies are synchronous).
int engine::pc_promote_bytes(const pc_disk_meta & meta, const uint8_t * blob, const float * state) {
    if (meta.blob_bytes != (int32_t)pc_block_blob_bytes() ||
        (meta.has_state && meta.state_bytes != (int32_t)(pc_state_floats * sizeof(float)))) {
        return -1; // layout changed: the record is not loadable
    }
    const int block = alloc_block();
    if (block < 0) {
        return -1;
    }
    pc_deserialize_block(blob, block);
    int st = -1;
    if (meta.has_state && pc_state_floats > 0) {
        st = pc_state_take();
        if (st < 0) {
            free_block(block); // caller keeps the lower-tier record for a retry
            return -1;
        }
        pc_deserialize_state(state, st);
    }
    const int ni = pc_add_node(meta.hash, meta.toks, block, meta.depth);
    if (ni < 0) {
        if (st >= 0) {
            pc_state_release(st);
        }
        free_block(block);
        return -1;
    }
    if (st >= 0) {
        pc_nodes_[ni].state = st;
        pc_state_owner[st] = ni;
        pc_state_stamp[st] = ++pc_clock;
    }
    pc_nodes_[ni].lru = ++pc_clock;
    return ni;
}

int engine::pc_promote_ram(uint64_t hash, const int32_t * toks) {
    if (!pcr_enabled || !pcr) {
        return -1;
    }
    pc_ram_entry e;
    if (!pcr->take(hash, e)) {
        return -1;
    }
    if (toks && std::memcmp(e.meta.toks, toks, sizeof(e.meta.toks)) != 0) {
        pcr->put(std::move(e));
        return -1;
    }
    const int ni = pc_promote_bytes(e.meta, e.blob.data(), e.state.empty() ? nullptr : e.state.data());
    if (ni >= 0) {
        pc_stat_ram_loads++;
    } else {
        pcr->put(std::move(e)); // failed promotion must not lose the record
    }
    return ni;
}

int engine::pc_promote_disk(pc_disk_meta meta) {
    if (!pcd_enabled || !pcd) {
        return -1;
    }
    std::vector<uint8_t> blob((size_t)std::max(meta.blob_bytes, 0));
    std::vector<float> state(meta.has_state ? pc_state_floats : 0);
    if (!pcd->load(meta, blob.empty() ? nullptr : blob.data(), state.empty() ? nullptr : state.data())) {
        pcd->erase(meta.hash);
        return -1;
    }
    const int ni = pc_promote_bytes(meta, blob.data(), state.empty() ? nullptr : state.data());
    if (ni >= 0) {
        pcd->erase(meta.hash); // moved into VRAM
        pc_stat_loads++;
    }
    return ni;
}

// Give a resident node the checkpoint that only a lower tier holds (the VRAM
// checkpoint pool dropped it earlier).  Returns false when no resume point can
// be recovered.
bool engine::pc_attach_state_from_lower(int ni, uint64_t hash, const int32_t * toks) {
    if (pc_nodes_[ni].state >= 0) {
        return true;
    }
    if (pcr_enabled && pcr) {
        pc_ram_entry e;
        if (pcr->take(hash, e)) {
            if (e.meta.has_state && e.meta.state_bytes == (int32_t)(pc_state_floats * sizeof(float))) {
                const int st = pc_state_take();
                if (st >= 0) {
                    pc_deserialize_state(e.state.data(), st);
                    pc_nodes_[ni].state = st;
                    pc_state_owner[st] = ni;
                    pc_state_stamp[st] = ++pc_clock;
                    pc_nodes_[ni].lru = ++pc_clock;
                    pc_stat_ram_loads++;
                    return true;
                }
            }
            pcr->put(std::move(e)); // unusable here: keep it for a future lookup
        }
    }
    if (pcd_enabled && pcd) {
        const pc_disk_meta * e = pcd->find(hash, toks);
        if (e && e->has_state && e->state_bytes == (int32_t)(pc_state_floats * sizeof(float))) {
            std::vector<float> buf(pc_state_floats);
            if (pcd->load(*e, nullptr, buf.data())) {
                const int st = pc_state_take();
                if (st >= 0) {
                    pc_deserialize_state(buf.data(), st);
                    pc_nodes_[ni].state = st;
                    pc_state_owner[st] = ni;
                    pc_state_stamp[st] = ++pc_clock;
                    pc_nodes_[ni].lru = ++pc_clock;
                    pc_stat_loads++;
                    return true;
                }
            } else {
                pcd->erase(hash);
            }
        }
    }
    return false;
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
    for (int il = 0; il < hp.n_layer; il++) {
        if (!hp.is_recr(il)) {
            continue;
        }
        const int gi = m.gdn_layer_index[il];
        if (multi_dev) {
            // device-resident per-partition state: copy into the owning device
            // (host d_pc_states -> device USM), then wait before the forward
            const int dev = layer_dev_[(size_t)il];
            const int gl = layer_gdn_local_[(size_t)il];
            sycl::queue & qd = dev_queue(dev);
            qd.memcpy(as_[(size_t)dev].gdn_state + ((size_t)gl * kMaxB + slot) * gp, src + (size_t)gi * per, gp * 4);
            qd.memcpy(as_[(size_t)dev].conv_state + ((size_t)gl * kMaxB + slot) * cp, src + (size_t)gi * per + gp,
                      cp * 4);
            continue;
        }
        q.memcpy(d_gdn_state + ((size_t)gi * kMaxB + slot) * gp, src + (size_t)gi * per, gp * 4);
        q.memcpy(d_conv_state + ((size_t)gi * kMaxB + slot) * cp, src + (size_t)gi * per + gp, cp * 4);
    }
    if (multi_dev) {
        sync_all();
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
    if ((pcr_enabled || pcd_enabled) && pc_nodes_[ni].refcount <= 0) {
        // With a lower tier the whole node moves out: its checkpoint survives
        // there, so the slot can be handed out without losing the resume point.
        pc_demote_node(ni);
        pc_nodes_[ni].state = -1; // detach first: pc_evict_node must not free
        pc_state_owner[best] = -1; // the slot we are about to return
        pc_evict_node(ni);
        pc_stat_evict_nodes++;
        pc_stat_evict_states++;
        return best;
    }
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
    if (pcr_enabled || pcd_enabled) {
        pc_demote_node(best); // move the node's block/checkpoint down a tier first
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
    const int nblk = max_full / kBlockSize;
    if (nblk <= 0) {
        pc_stat_misses++;
        return 0;
    }
    auto tok_at = [&](int i) { return prompt.data() + (size_t)i * kBlockSize; };

    // Pass 1: walk the chained hashes.  A block resolves to a VRAM node, a RAM
    // record, a disk record, or breaks the chain; remember the deepest boundary
    // whose recurrent checkpoint is available in any tier, since that is where
    // the request can resume.
    std::vector<uint64_t> hashes((size_t)nblk);
    uint64_t chain = 0;
    int avail = 0, deep = 0;
    for (int i = 0; i < nblk; i++) {
        const int32_t * tk = tok_at(i);
        const uint64_t h = pc_hash_block(chain, tk);
        hashes[i] = h;
        const int ni = pc_find(h, tk);
        const pc_ram_entry * re = (ni < 0 && pcr_enabled) ? pcr->find(h, tk) : nullptr;
        const pc_disk_meta * de = (ni < 0 && !re && pcd_enabled) ? pcd->find(h, tk) : nullptr;
        if (pcdbg) {
            fprintf(stderr, "[pcdbg] admit L=%d blk=%d h=%llx vram=%d ram=%d disk=%d\n", L, i, (unsigned long long)h,
                    ni, re ? 1 : 0, de ? 1 : 0);
        }
        if (ni < 0 && !re && !de) {
            break;
        }
        avail = i + 1;
        bool has_state = ni >= 0 && pc_nodes_[ni].state >= 0;
        if (!has_state && re) {
            has_state = re->meta.has_state;
        }
        if (!has_state && pcd_enabled) {
            const pc_disk_meta * e = de ? de : pcd->find(h, tk);
            has_state = e && e->has_state;
        }
        if (has_state) {
            deep = i + 1;
        }
        chain = h;
    }
    if (avail <= 0 || deep <= 0) {
        pc_stat_misses++;
        return 0;
    }

    auto unpin = [&](int n) {
        for (int i = 0; i < n; i++) {
            const int ni = pc_find(hashes[i], tok_at(i));
            if (ni >= 0 && pc_nodes_[ni].refcount > 0) {
                pc_nodes_[ni].refcount--;
            }
        }
        blocks.clear();
    };

    // Pass 2: pin every block up to the deepest resumable boundary, promoting
    // lower-tier records into VRAM on the way.  Promotions may demote
    // unreferenced nodes to make room.
    int reach = 0;
    bool failed = false;
    for (int i = 0; i < deep; i++) {
        const int32_t * tk = tok_at(i);
        int ni = pc_find(hashes[i], tk);
        if (ni >= 0) {
            pc_nodes_[ni].refcount++;
            pc_nodes_[ni].lru = ++pc_clock;
        } else {
            ni = pc_promote_ram(hashes[i], tk);
            if (ni < 0 && pcd_enabled) {
                const pc_disk_meta * de = pcd->find(hashes[i], tk);
                if (de) {
                    ni = pc_promote_disk(*de); // by value: promotion resizes the index
                }
            }
            if (ni < 0) {
                failed = true;
                break;
            }
        }
        blocks.push_back(pc_nodes_[ni].block);
        reach++;
    }
    auto tier_has_state = [&](int b) {
        const int32_t * tk = tok_at(b - 1);
        const int ni = pc_find(hashes[b - 1], tk);
        if (ni >= 0 && pc_nodes_[ni].state >= 0) {
            return true;
        }
        if (pcr_enabled) {
            const pc_ram_entry * e = pcr->find(hashes[b - 1], tk);
            if (e && e->meta.has_state) {
                return true;
            }
        }
        if (pcd_enabled) {
            const pc_disk_meta * e = pcd->find(hashes[b - 1], tk);
            if (e && e->has_state) {
                return true;
            }
        }
        return false;
    };
    if (failed) {
        // fall back to the deepest resumable boundary we actually reached
        deep = 0;
        for (int b = reach; b >= 1; b--) {
            if (tier_has_state(b)) {
                deep = b;
                break;
            }
        }
    }
    for (int i = deep; i < reach; i++) {
        const int ni = pc_find(hashes[i], tok_at(i));
        if (ni >= 0 && pc_nodes_[ni].refcount > 0) {
            pc_nodes_[ni].refcount--;
        }
    }
    blocks.resize((size_t)deep);
    if (deep <= 0) {
        pc_stat_misses++;
        return 0;
    }

    // The boundary node may be resident but without its checkpoint (the VRAM
    // pool dropped it); pull the state back from RAM or disk.
    const int32_t * btk = tok_at(deep - 1);
    const int bni = pc_find(hashes[deep - 1], btk);
    if (bni < 0 || !pc_attach_state_from_lower(bni, hashes[deep - 1], btk)) {
        unpin(deep);
        pc_stat_misses++;
        return 0;
    }
    pc_restore_state(slot, pc_nodes_[bni].state);
    if (pcr_enabled) {
        pcr->touch(hashes[deep - 1]);
    }
    if (pcd_enabled) {
        pcd->touch(hashes[deep - 1]);
    }
    pc_slot_[slot].chain = hashes[deep - 1];
    pc_slot_[slot].registered = deep;
    pc_stat_hits++;
    pc_stat_tokens += (uint64_t)deep * kBlockSize;
    return deep * kBlockSize;
}

// Reserve checkpoint slots for the block boundaries this forward will complete
// and fill the kernel snapshot map.  Called from prefill_chunk/prefill_batch
// before the forward; the map is read by the conv/GDN kernels, which write the
// state at each marked boundary straight into the checkpoint pool.  pc_commit
// links the slots to the cache nodes afterwards.
void engine::pc_capture_begin(int slot, const std::vector<int> & toks, int tok_off, int pos0, int n) {
    step_info * inf = pf_info_ ? pf_info_ : d_info;
    // a previous forward's pending reservations (no commit) must not leak
    for (auto & p : pc_pending_) {
        pc_state_release(p.second);
    }
    pc_pending_.clear();
    inf->pc_active = 0;
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
        inf->pc_row_slot[b] = -1;
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
        inf->pc_row_slot[need[k].b] = st;
        pc_pending_.push_back({need[k].h, st});
    }
    if (pc_pending_.empty()) {
        return;
    }
    inf->pc_base = d_pc_states;
    inf->pc_stride = (int32_t)pc_state_floats;
    inf->pc_active = 1;
}

void engine::pc_commit(int slot, const std::vector<int> & toks, const std::vector<int> & blocks, int done) {
    step_info * inf = pf_info_ ? pf_info_ : d_info;
    static const bool pcdbg = getenv("PF_PC_DEBUG") != nullptr;
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
            "captured=%llu, evicted nodes=%llu states=%llu, checkpoint=%.1f MB x %d = %.0f MB",
            tag, pc_nodes_.size(), with_state, (unsigned long long)refs, (unsigned long long)pc_stat_hits,
            (unsigned long long)lookups, lookups ? 100.0 * (double)pc_stat_hits / (double)lookups : 0.0,
            (unsigned long long)pc_stat_tokens, (unsigned long long)pc_stat_states,
            (unsigned long long)pc_stat_evict_nodes, (unsigned long long)pc_stat_evict_states, state_mb, pc_max_states,
            state_mb * pc_max_states);
    if (pcr_enabled) {
        fprintf(stderr, ", ram stores=%llu loads=%llu spills=%llu records=%zu", (unsigned long long)pc_stat_ram_stores,
                (unsigned long long)pc_stat_ram_loads, (unsigned long long)pc_stat_ram_spills, pcr->count());
    }
    if (pcd_enabled) {
        fprintf(stderr, ", disk spills=%llu loads=%llu records=%zu", (unsigned long long)pc_stat_spills,
                (unsigned long long)pc_stat_loads, pcd->count());
    }
    fprintf(stderr, "\n");
}

} // namespace si
