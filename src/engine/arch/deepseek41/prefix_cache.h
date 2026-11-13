// arch/deepseek41/prefix_cache.h - the bookkeeping core of the model-owned ds41 prefix
// cache, factored out of arch/deepseek41/forward_host.cpp so it can be tested without a model.
//
// The engine's tiered prefix cache is keyed on the paged KV pool, which the
// ds41 forward does not use: its reusable state is the sliding-window ring,
// the compressed KV rows, the DSA candidate mask and the Engram history.  That
// state is snapshotted as an opaque Payload (DecState + logits in the engine).
//
// Prefix identity is a chained FNV-1a over the token ids: h[i] identifies
// tokens[0..i], so a snapshot at any consumed-token count can be found by one
// hash update per prompt token and a map probe.  Lookups therefore cost
// O(prompt) and never copy a payload - the caller copies the one entry it
// resumes from.
#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace si {

inline uint64_t ds41_pc_seed() {
    return 1469598103934665603ull;
}

inline uint64_t ds41_pc_step(uint64_t h, int tok) {
    h ^= (uint64_t)(uint32_t)tok;
    h *= 1099511628211ull;
    return h;
}

template <class Payload> class ds41_prefix_cache {
public:
    struct entry {
        int consumed = 0;   // prompt tokens represented by payload
        uint64_t stamp = 0; // LRU clock
        size_t bytes = 0;
        Payload payload{};
    };
    struct hit {
        bool found = false;
        uint64_t hash = 0;
        int consumed = 0;
    };

    explicit ds41_prefix_cache(size_t cap = (size_t)512 << 20) : cap_(cap), max_entry_(cap / 4) {
    }

    size_t cap_bytes() const {
        return cap_;
    }
    // A state larger than this is skipped: one snapshot must not crowd out
    // every prefix in the budget.
    size_t max_entry_bytes() const {
        return max_entry_;
    }
    size_t bytes() const {
        return bytes_;
    }
    size_t size() const {
        return map_.size();
    }
    void clear() {
        map_.clear();
        bytes_ = 0;
    }

    // The longest stored prefix of `tokens`.  The chain identifies the whole
    // prefix, so an entry for a shorter prompt is found at its own position;
    // the returned `consumed` is the number of tokens to resume past.
    hit longest(const std::vector<int> & tokens) const {
        hit best;
        uint64_t h = ds41_pc_seed();
        for (size_t i = 0; i < tokens.size(); ++i) {
            h = ds41_pc_step(h, tokens[i]);
            auto it = map_.find(h);
            if (it == map_.end()) {
                continue;
            }
            const int consumed = (int)(i + 1);
            if (it->second.consumed == consumed && consumed > best.consumed) {
                best.found = true;
                best.hash = h;
                best.consumed = consumed;
            }
        }
        return best;
    }

    const entry * peek(uint64_t hash) const {
        auto it = map_.find(hash);
        return it == map_.end() ? nullptr : &it->second;
    }

    // Fetch a snapshot and mark it recently used.  The pointer is valid until
    // the next insert/erase.
    entry * find(uint64_t hash) {
        auto it = map_.find(hash);
        if (it == map_.end()) {
            return nullptr;
        }
        it->second.stamp = ++tick_;
        return &it->second;
    }

    void touch(uint64_t hash) {
        auto it = map_.find(hash);
        if (it != map_.end()) {
            it->second.stamp = ++tick_;
        }
    }

    // Insert (or replace) one snapshot, evicting least-recently-used entries by
    // bytes until it fits.  Returns false when `bytes` alone exceeds cap/4, the
    // whole cap, or the cap is zero.
    bool insert(uint64_t hash, int consumed, Payload payload, size_t bytes) {
        if (cap_ == 0 || bytes > max_entry_ || bytes > cap_) {
            return false;
        }
        auto it = map_.find(hash);
        if (it != map_.end()) {
            bytes_ -= it->second.bytes;
            map_.erase(it);
        }
        while (!map_.empty() && bytes_ + bytes > cap_) {
            auto old = map_.begin();
            for (auto j = map_.begin(); j != map_.end(); ++j) {
                if (j->second.stamp < old->second.stamp) {
                    old = j;
                }
            }
            bytes_ -= old->second.bytes;
            map_.erase(old);
        }
        entry e;
        e.consumed = consumed;
        e.stamp = ++tick_;
        e.bytes = bytes;
        e.payload = std::move(payload);
        bytes_ += e.bytes;
        map_.emplace(hash, std::move(e));
        return true;
    }

private:
    size_t cap_;
    size_t max_entry_;
    size_t bytes_ = 0;
    uint64_t tick_ = 0;
    std::unordered_map<uint64_t, entry> map_;
};

} // namespace si
