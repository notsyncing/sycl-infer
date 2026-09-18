// Host-only unit test for the disk tier of the prefix cache (no GPU / model):
// format round-trip, token verification, LRU budget, persistence across a
// reopen, and graceful handling of unknown/corrupt records.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

#include "pc_disk.h"

using namespace si;

static int g_fail = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                            \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

static void fill_toks(int32_t * toks, int seed) {
    for (int i = 0; i < kPcBlockToks; i++) {
        toks[i] = seed * 1000 + i;
    }
}

static std::vector<uint8_t> make_blob(int seed) {
    std::vector<uint8_t> b(64);
    for (size_t i = 0; i < b.size(); i++) {
        b[i] = (uint8_t)(i * 7 + seed);
    }
    return b;
}

static std::vector<float> make_state(int seed, size_t n) {
    std::vector<float> s(n);
    for (size_t i = 0; i < n; i++) {
        s[i] = (float)seed + (float)i * 0.5f;
    }
    return s;
}

static bool store_one(pc_disk_store & s, uint64_t h, int depth, int seed, size_t state_n) {
    int32_t toks[kPcBlockToks];
    fill_toks(toks, seed);
    auto blob = make_blob(seed);
    auto state = make_state(seed, state_n);
    return s.store(h, depth, toks, blob.data(), blob.size(), state_n ? state.data() : nullptr,
                   state_n ? state.size() * sizeof(float) : 0);
}

static void check_record(const pc_disk_store & s, uint64_t h, int depth, int seed, size_t state_n) {
    int32_t toks[kPcBlockToks];
    fill_toks(toks, seed);
    const pc_disk_meta * m = s.find(h, toks);
    CHECK(m != nullptr);
    if (!m) {
        return;
    }
    CHECK(m->depth == depth);
    CHECK(m->has_state == (state_n > 0));
    CHECK(m->blob_bytes == (int32_t)make_blob(seed).size());
    CHECK(m->state_bytes == (int32_t)(state_n * sizeof(float)));
    std::vector<uint8_t> blob(m->blob_bytes);
    std::vector<float> state(state_n);
    CHECK(s.load(*m, blob.data(), state_n ? state.data() : nullptr));
    CHECK(blob == make_blob(seed));
    if (state_n) {
        CHECK(state == make_state(seed, state_n));
        // a state-only read (blob skipped) must return the same checkpoint
        std::vector<float> state2(state_n, 0.f);
        CHECK(s.load(*m, nullptr, state2.data()));
        CHECK(state2 == make_state(seed, state_n));
    }
    // a different token block must not match the same hash
    int32_t other[kPcBlockToks];
    fill_toks(other, seed + 1);
    CHECK(s.find(h, other) == nullptr);
}

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("si_pc_disk_" + std::to_string((long)getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);

    const size_t rec = 168 + 64; // on-disk header + blob of a state-less record

    // ---- round-trip and LRU budget (two state-less records fit) -----------
    {
        pc_disk_store s;
        CHECK(s.open(dir.string(), rec * 2));
        CHECK(s.active());
        CHECK(store_one(s, 0x1111, 1, 1, 0));
        CHECK(store_one(s, 0x2222, 2, 2, 0));
        check_record(s, 0x1111, 1, 1, 0);
        check_record(s, 0x2222, 2, 2, 0);
        CHECK(store_one(s, 0x3333, 3, 3, 0));
        CHECK(s.count() == 2);
        CHECK(s.find(0x1111, nullptr) == nullptr);
        CHECK(s.find(0x2222, nullptr) != nullptr);
        CHECK(s.find(0x3333, nullptr) != nullptr);
        s.touch(0x2222); // 0x2222 survives, 0x3333 becomes the LRU
        CHECK(store_one(s, 0x4444, 4, 4, 0));
        CHECK(s.find(0x2222, nullptr) != nullptr);
        CHECK(s.find(0x3333, nullptr) == nullptr);
        s.print_stats("test");
    }
    // ---- persistence: reopen restores the index from the headers ----------
    {
        pc_disk_store s;
        CHECK(s.open(dir.string(), 0)); // unbounded for this phase
        CHECK(s.count() == 2);
        check_record(s, 0x2222, 2, 2, 0);
        check_record(s, 0x4444, 4, 4, 0);
        // overwrite with a checkpoint, then add another stateful record
        CHECK(store_one(s, 0x4444, 4, 4, 8));
        check_record(s, 0x4444, 4, 4, 8);
        CHECK(store_one(s, 0x5555, 5, 5, 16));
        check_record(s, 0x5555, 5, 5, 16);
        CHECK(s.count() == 3);
    }
    {
        pc_disk_store s;
        CHECK(s.open(dir.string(), 0));
        CHECK(s.count() == 3);
        check_record(s, 0x2222, 2, 2, 0);
        check_record(s, 0x4444, 4, 4, 8);
        check_record(s, 0x5555, 5, 5, 16);
    }
    // ---- an unknown format version is kept, skipped, never deleted --------
    {
        const fs::path f = dir / "00000000deadbeef.pcn";
        std::vector<uint8_t> h(168, 0);
        std::memcpy(h.data(), "PCD1", 4);
        h[4] = 99; // version > current
        std::ofstream o(f, std::ios::binary);
        o.write((const char *)h.data(), (std::streamsize)h.size());
        o.close();
        pc_disk_store s;
        CHECK(s.open(dir.string(), 0));
        CHECK(s.find(0xdeadbeef, nullptr) == nullptr);
        CHECK(fs::exists(f));
        CHECK(s.count() == 3);
        fs::remove(f);
    }
    // ---- a corrupt record is discarded at open ---------------------------
    {
        const fs::path f = dir / "00000000c0ffee00.pcn";
        std::ofstream o(f, std::ios::binary);
        o.write("junk", 4);
        o.close();
        pc_disk_store s;
        CHECK(s.open(dir.string(), 0));
        CHECK(!fs::exists(f));
        CHECK(s.count() == 3);
    }

    fs::remove_all(dir, ec);
    if (g_fail) {
        fprintf(stderr, "pc_disk: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("pc_disk: all checks OK\n");
    return 0;
}
