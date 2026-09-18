#include "pc_ram.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace si {

void pc_ram_store::open(size_t max_bytes) {
    max_bytes_ = max_bytes;
    bytes_ = 0;
    lru_clock_ = 0;
    active_ = true;
    index_.clear();
}

const pc_ram_entry * pc_ram_store::find(uint64_t hash, const int32_t * toks) const {
    auto it = index_.find(hash);
    if (it == index_.end()) {
        return nullptr;
    }
    if (toks && std::memcmp(it->second.meta.toks, toks, sizeof(it->second.meta.toks)) != 0) {
        return nullptr; // hash collision: the tokens decide
    }
    return &it->second;
}

bool pc_ram_store::put(pc_ram_entry && e) {
    if (!active_) {
        return false;
    }
    const size_t sz = entry_bytes(e);
    if (max_bytes_ && sz > max_bytes_) {
        return false; // cannot ever fit
    }
    e.meta.blob_bytes = (int32_t)e.blob.size();
    e.meta.state_bytes = (int32_t)(e.state.size() * sizeof(float));
    e.meta.file_bytes = sz;
    e.meta.lru = ++lru_clock_;
    e.meta.name.clear();
    auto it = index_.find(e.meta.hash);
    if (it != index_.end()) {
        bytes_ -= entry_bytes(it->second);
        index_.erase(it);
    }
    index_[e.meta.hash] = std::move(e);
    bytes_ += sz;
    return true;
}

bool pc_ram_store::take(uint64_t hash, pc_ram_entry & out) {
    auto it = index_.find(hash);
    if (it == index_.end()) {
        return false;
    }
    bytes_ -= entry_bytes(it->second);
    out = std::move(it->second);
    index_.erase(it);
    return true;
}

bool pc_ram_store::erase(uint64_t hash) {
    auto it = index_.find(hash);
    if (it == index_.end()) {
        return false;
    }
    bytes_ -= entry_bytes(it->second);
    index_.erase(it);
    return true;
}

void pc_ram_store::touch(uint64_t hash) {
    auto it = index_.find(hash);
    if (it != index_.end()) {
        it->second.meta.lru = ++lru_clock_;
    }
}

void pc_ram_store::enforce_budget(std::vector<pc_ram_entry> & evicted) {
    if (max_bytes_ == 0) {
        return;
    }
    while (bytes_ > max_bytes_ && !index_.empty()) {
        auto best = index_.begin();
        for (auto it = index_.begin(); it != index_.end(); ++it) {
            if (it->second.meta.lru < best->second.meta.lru) {
                best = it;
            }
        }
        bytes_ -= entry_bytes(best->second);
        evicted.push_back(std::move(best->second));
        index_.erase(best);
    }
}

void pc_ram_store::drain(std::vector<pc_ram_entry> & out) {
    out.reserve(out.size() + index_.size());
    for (auto & kv : index_) {
        out.push_back(std::move(kv.second));
    }
    index_.clear();
    bytes_ = 0;
}

void pc_ram_store::print_stats(const char * tag) const {
    if (!active_) {
        return;
    }
    fprintf(stderr, "[pcr] %s: %zu records (%.1f MB / %.1f MB)\n", tag, index_.size(),
            (double)bytes_ / (1024.0 * 1024.0), max_bytes_ ? (double)max_bytes_ / (1024.0 * 1024.0) : 0.0);
}

} // namespace si
