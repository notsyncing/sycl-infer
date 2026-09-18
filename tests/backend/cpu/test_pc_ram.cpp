// Host-only unit test for the RAM tier of the prefix cache (pc_ram_store):
// put/find/take, token verification, LRU budget overflow and oversized records.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "pc_disk.h"
#include "pc_ram.h"

using namespace si;

static int g_fail = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                            \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

static pc_ram_entry make(uint64_t hash, int seed, size_t state_n) {
    pc_ram_entry e;
    e.meta.hash = hash;
    e.meta.depth = seed;
    e.meta.has_state = state_n > 0;
    for (int i = 0; i < kPcBlockToks; i++) {
        e.meta.toks[i] = seed * 1000 + i;
    }
    e.blob.resize(64);
    for (size_t i = 0; i < e.blob.size(); i++) {
        e.blob[i] = (uint8_t)(i + seed);
    }
    e.state.assign(state_n, (float)seed);
    return e;
}

int main() {
    {
        pc_ram_store s;
        s.open(2 * 64); // room for two state-less records
        CHECK(s.active());
        CHECK(s.budget() == 2 * 64);
        CHECK(s.put(make(0x11, 1, 0)));
        CHECK(s.put(make(0x22, 2, 0)));
        CHECK(s.count() == 2);
        CHECK(s.bytes() == 128);
        // token ids guard the hash
        int32_t other[kPcBlockToks];
        std::memcpy(other, make(0x11, 1, 0).meta.toks, sizeof(other));
        other[0]++;
        CHECK(s.find(0x11, other) == nullptr);
        // a third record evicts the oldest, and the caller gets it back
        CHECK(s.put(make(0x33, 3, 0)));
        std::vector<pc_ram_entry> over;
        s.enforce_budget(over);
        CHECK(over.size() == 1);
        CHECK(over[0].meta.hash == 0x11);
        CHECK(s.count() == 2);
        CHECK(s.bytes() == 128);
        CHECK(s.find(0x11, nullptr) == nullptr);
        CHECK(s.find(0x22, nullptr) != nullptr);
        // take removes and returns a record
        pc_ram_entry taken;
        CHECK(s.take(0x22, taken));
        CHECK(taken.meta.hash == 0x22);
        CHECK(s.find(0x22, nullptr) == nullptr);
        // a record larger than the whole budget is rejected
        pc_ram_entry big = make(0x66, 6, 0);
        big.blob.resize(200);
        CHECK(!s.put(std::move(big)));
        // a stateful record that fits is accepted
        CHECK(s.put(make(0x44, 4, 1)));
        CHECK(s.find(0x44, nullptr) != nullptr);
        pc_ram_store u;
        u.open(0); // unbounded
        CHECK(u.put(make(0x55, 5, 64)));
        CHECK(u.count() == 1);
    }
    if (g_fail) {
        fprintf(stderr, "pc_ram: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("pc_ram: all checks OK\n");
    return 0;
}
