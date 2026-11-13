// Hermetic test for the ds41 prefix-cache bookkeeping (no GGUF, no GPU).
//
// The engine's real payload is a DecState+logits (arch/deepseek41/forward_host.cpp); here an int
// stands in for it, which is enough to pin the three properties the cache must
// have: the chained hash identifies a whole prefix, `longest` returns the
// greatest stored consumed count for a shared prefix, and eviction is LRU by
// bytes with a cap/4 per-entry ceiling.
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "engine/arch/deepseek41/prefix_cache.h"

using namespace si;

static int g_fail = 0;

#define CHECK(cond, what)                                                                                              \
    do {                                                                                                               \
        if (cond) {                                                                                                    \
            printf("  ok       %s\n", what);                                                                           \
        } else {                                                                                                       \
            fprintf(stderr, "FAIL %s\n", what);                                                                        \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

static uint64_t chain_at(const std::vector<int> & tokens, size_t n) {
    uint64_t h = ds41_pc_seed();
    for (size_t i = 0; i < n; i++) {
        h = ds41_pc_step(h, tokens[i]);
    }
    return h;
}

static std::vector<int> make_tokens(size_t n) {
    std::vector<int> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = (int)(i * 7 + 3);
    }
    return v;
}

int main() {
    printf("ds41 prefix cache\n");

    const std::vector<int> base = make_tokens(48);

    // A chained hash depends on the whole prefix, not just the last token.
    CHECK(ds41_pc_step(ds41_pc_seed(), 5) == ds41_pc_step(ds41_pc_seed(), 5), "the same token hashes identically");
    CHECK(ds41_pc_step(ds41_pc_seed(), 5) != ds41_pc_step(ds41_pc_seed(), 6), "a different token hashes differently");
    CHECK(chain_at(base, 8) == chain_at(base, 8), "the chain is deterministic");
    CHECK(chain_at(base, 8) != chain_at(base, 9), "a longer prefix has a different chain");
    {
        std::vector<int> other = base;
        other[7] ^= 1;
        CHECK(chain_at(base, 8) != chain_at(other, 8), "changing a prefix token changes the chain");
        CHECK(chain_at(base, 7) == chain_at(other, 7), "a later change does not affect an earlier chain");
    }

    // Longest-prefix lookup across snapshots at several lengths.
    ds41_prefix_cache<int> cache(1u << 20);
    const size_t marks[] = {8, 16, 24, 40};
    for (size_t m : marks) {
        cache.insert(chain_at(base, m), (int)m, (int)m * 10, 16);
    }
    for (size_t n : {0u, 1u, 7u, 8u, 9u, 15u, 16u, 17u, 23u, 24u, 39u, 40u, 41u, 48u}) {
        std::vector<int> q(base.begin(), base.begin() + n);
        const auto hit = cache.longest(q);
        size_t want = 0;
        for (size_t m : marks) {
            if (m <= n) {
                want = m;
            }
        }
        const std::string what = "longest(" + std::to_string(n) + ") == " + std::to_string(want);
        CHECK((want == 0 ? !hit.found : (hit.found && (size_t)hit.consumed == want)), what.c_str());
    }
    // A different suffix after a shared prefix still matches the prefix.
    {
        std::vector<int> q(base.begin(), base.begin() + 30);
        q.push_back(999999);
        const auto hit = cache.longest(q);
        CHECK(hit.found && hit.consumed == 24, "a differing suffix keeps the longest shared prefix (24)");
    }
    // An entry whose recorded consumed count disagrees with its position is
    // ignored (a chained-hash collision must not resume from the wrong place).
    {
        ds41_prefix_cache<int> c2(1u << 20);
        c2.insert(chain_at(base, 8), 7, 1, 16);
        CHECK(!c2.longest(std::vector<int>(base.begin(), base.begin() + 8)).found,
              "a mismatched consumed count is not a hit");
    }

    // Inserting the same hash replaces the payload and does not double-count.
    {
        ds41_prefix_cache<int> c3(1u << 20);
        const uint64_t h = chain_at(base, 8);
        c3.insert(h, 8, 1, 100);
        const size_t bytes1 = c3.bytes();
        c3.insert(h, 8, 2, 120);
        CHECK(c3.size() == 1 && c3.bytes() == 120 && bytes1 == 100, "replacing a hash updates its byte accounting");
        const auto * e = c3.peek(h);
        CHECK(e && e->payload == 2, "replacing a hash installs the new payload");
    }

    // cap/4 per-entry ceiling: a state that big is skipped, not stored.
    {
        ds41_prefix_cache<int> c4(1024);
        CHECK(c4.max_entry_bytes() == 256, "the per-entry ceiling is cap/4");
        CHECK(!c4.insert(chain_at(base, 4), 4, 1, 257), "an entry above cap/4 is rejected");
        CHECK(c4.size() == 0 && c4.bytes() == 0, "a rejected entry leaves the cache untouched");
        CHECK(c4.insert(chain_at(base, 4), 4, 1, 256), "an entry at exactly cap/4 is accepted");
    }

    // LRU by bytes: with cap 4096 and 512-byte entries, the ninth insert evicts
    // the least recently used one, not an arbitrary one.
    {
        ds41_prefix_cache<int> c5(4096);
        std::vector<uint64_t> hs;
        for (int i = 0; i < 8; i++) {
            const uint64_t h = chain_at(make_tokens((size_t)i + 1), (size_t)i + 1) ^ (uint64_t)i;
            hs.push_back(h);
            c5.insert(h, i + 1, i, 512);
        }
        CHECK(c5.size() == 8 && c5.bytes() == 4096, "eight 512-byte entries fill the 4096-byte cap");
        // Touch entry 0, then add a ninth: the untouched entry 1 must go.
        c5.find(hs[0]);
        const uint64_t h9 = chain_at(base, 9) ^ 0xabcdefull;
        c5.insert(h9, 9, 9, 512);
        CHECK(c5.size() == 8 && c5.bytes() == 4096, "the ninth entry keeps the byte total at the cap");
        CHECK(c5.peek(hs[0]) != nullptr, "the touched entry survives eviction");
        CHECK(c5.peek(hs[1]) == nullptr, "the least-recently-used entry is evicted");
        CHECK(c5.peek(h9) != nullptr, "the new entry is present");
    }

    // A zero cap disables the cache rather than storing unbounded.
    {
        ds41_prefix_cache<int> c6(0);
        CHECK(!c6.insert(chain_at(base, 4), 4, 1, 0), "a zero cap rejects every insert");
        CHECK(!c6.longest(std::vector<int>(base.begin(), base.begin() + 4)).found, "a zero-cap cache never hits");
    }

    if (g_fail) {
        fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    printf("all ds41 prefix-cache checks passed\n");
    return 0;
}
