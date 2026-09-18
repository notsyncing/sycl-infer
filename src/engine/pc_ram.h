#pragma once
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "pc_disk.h" // pc_disk_meta, kPcBlockToks

namespace si {

// One node record held in host memory: the same fields a disk record stores,
// already serialized (see pc_disk.h for the layout semantics).
struct pc_ram_entry {
    pc_disk_meta meta; // hash/depth/tokens/has_state (sizes are refreshed by put)
    std::vector<uint8_t> blob;
    std::vector<float> state;
};

// Host-RAM tier of the prefix cache.  It keeps the serialized node records that
// were evicted from VRAM, with an LRU byte budget.  Overflow is handed back to
// the caller (which writes it to the disk tier), so this store does no I/O and
// can be unit-tested on its own.
struct pc_ram_store {
    // max_bytes == 0 means unbounded
    void open(size_t max_bytes);
    bool active() const {
        return active_;
    }
    // lookup by node hash; the token ids are verified so a hash collision is
    // rejected.  The returned pointer is invalidated by put()/erase()/take().
    const pc_ram_entry * find(uint64_t hash, const int32_t * toks) const;
    // keep a record (replacing any entry with the same hash); returns false when
    // the record is larger than the whole budget and cannot be cached
    bool put(pc_ram_entry && e);
    // remove and return an entry (used to promote it back to VRAM)
    bool take(uint64_t hash, pc_ram_entry & out);
    bool erase(uint64_t hash);
    void touch(uint64_t hash);
    // move every entry over budget into `evicted`, least recently used first
    void enforce_budget(std::vector<pc_ram_entry> & evicted);
    // move every entry out (shutdown flush)
    void drain(std::vector<pc_ram_entry> & out);
    size_t bytes() const {
        return bytes_;
    }
    size_t count() const {
        return index_.size();
    }
    size_t budget() const {
        return max_bytes_;
    }
    void print_stats(const char * tag) const;

private:
    static size_t entry_bytes(const pc_ram_entry & e) {
        return e.blob.size() + e.state.size() * sizeof(float);
    }
    std::unordered_map<uint64_t, pc_ram_entry> index_;
    size_t max_bytes_ = 0;
    size_t bytes_ = 0;
    uint64_t lru_clock_ = 0;
    bool active_ = false;
};

} // namespace si
