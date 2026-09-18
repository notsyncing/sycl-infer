#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace si {

// token ids per prefix-cache block (= kBlockSize, kept local so this header
// does not pull in the SYCL kernel headers); pc_disk.cpp static_asserts it
constexpr int kPcBlockToks = 32;

// Metadata of one node record on disk.  `blob` is the serialized K/V (and, for
// the int8 KV type, the fp16 scale planes) of the node's 32-token block over
// every attention layer; the optional `state` is the recurrent (GDN + conv)
// checkpoint captured at the block's boundary.
struct pc_disk_meta {
    uint64_t hash = 0;      // chained hash of the node
    int32_t depth = 0;      // blocks in the hash chain (1 = root)
    bool has_state = false; // the record carries a state checkpoint
    int32_t blob_bytes = 0; // serialized block bytes
    int32_t state_bytes = 0;
    int32_t toks[kPcBlockToks] = {};
    size_t file_bytes = 0; // size of the record file
    uint64_t lru = 0;      // LRU stamp (seeded from the file mtime at open)
    std::string name;      // file name inside the directory
};

// Disk tier of the prefix cache: one immutable file per cache node, keyed by
// the node's chained hash.  The directory is model-specific (the engine passes
// <base>/<model-fingerprint>), files are written atomically (temp + rename),
// and the directory is rescanned at startup, so no separate index is needed.
//
// On-disk format: a fixed little-endian header (magic "PCD1", version,
// geometry sizes, hash, depth, tokens) followed by the block blob and the
// optional state.  A record whose version the reader does not understand is
// skipped, never deleted, so a cache written by a newer engine survives an
// older binary; records are only evicted by LRU.
struct pc_disk_store {
    // create/open `dir`; max_bytes == 0 means unbounded.  Returns true when the
    // directory is usable.  Unknown-format files are counted but never touched.
    bool open(const std::string & dir, size_t max_bytes);
    bool active() const {
        return active_;
    }
    const pc_disk_meta * find(uint64_t hash, const int32_t * toks) const;
    // read the blob (may be null to skip it) and the state (null to skip) of an
    // indexed record; returns false on an I/O or geometry mismatch
    bool load(const pc_disk_meta & m, void * blob, float * state) const;
    bool store(uint64_t hash, int32_t depth, const int32_t * toks, const void * blob, size_t blob_bytes,
               const float * state, size_t state_bytes);
    bool erase(uint64_t hash);
    void touch(uint64_t hash);
    size_t bytes() const {
        return bytes_;
    }
    size_t count() const {
        return index_.size();
    }
    const std::string & dir() const {
        return dir_;
    }
    void print_stats(const char * tag) const;

private:
    // 0 = parsed, 1 = well-formed but unknown version (leave it alone),
    // 2 = corrupt/unreadable (safe to delete)
    int read_header(const std::string & path, pc_disk_meta & m) const;
    bool write_record(const pc_disk_meta & m, const void * blob, const float * state);
    void enforce_budget();
    std::string path_of(const std::string & name) const;

    std::string dir_;
    size_t max_bytes_ = 0;
    size_t bytes_ = 0;        // bytes held by indexed records
    size_t opaque_bytes_ = 0; // files from a newer, unknown format (never deleted)
    uint64_t lru_clock_ = 0;
    bool active_ = false;
    std::unordered_map<uint64_t, pc_disk_meta> index_;
};

} // namespace si
