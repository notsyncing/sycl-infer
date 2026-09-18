#include "pc_disk.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <system_error>

namespace si {

// ---- on-disk header -------------------------------------------------------
// Written with explicit little-endian integers so the file does not depend on
// the host endianness.  Anything a reader does not recognise is skipped.
static constexpr char kMagic[4] = {'P', 'C', 'D', '1'};
static constexpr uint32_t kVersion = 1;
static constexpr uint32_t kEndianMarker = 0x01020304u;
static constexpr uint32_t kFlagHasState = 1u;

// on big-endian hosts the stored little-endian value must be byte-swapped
static uint32_t bswap_if_be(uint32_t v) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return __builtin_bswap32(v);
#else
    return v;
#endif
}
static void wr_le32(uint8_t * p, uint32_t v) {
    v = bswap_if_be(v);
    std::memcpy(p, &v, 4);
}
static uint32_t rd_le32(const uint8_t * p) {
    uint32_t v = 0;
    std::memcpy(&v, p, 4);
    return bswap_if_be(v);
}
static void wr_le64(uint8_t * p, uint64_t v) {
    wr_le32(p, (uint32_t)(v & 0xffffffffu));
    wr_le32(p + 4, (uint32_t)(v >> 32));
}
static uint64_t rd_le64(const uint8_t * p) {
    return (uint64_t)rd_le32(p) | ((uint64_t)rd_le32(p + 4) << 32);
}

// magic(4) version(4) header_bytes(4) endian(4) flags(4) hash(8) depth(4)
// blob_bytes(4) state_bytes(4) toks(32*4) = 168
static constexpr uint32_t kHeaderBytes = 4 + 4 + 4 + 4 + 4 + 8 + 4 + 4 + 4 + kPcBlockToks * 4;

static std::string hex_name(uint64_t hash) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016llx.pcn", (unsigned long long)hash);
    return buf;
}

std::string pc_disk_store::path_of(const std::string & name) const {
    if (dir_.empty()) {
        return name;
    }
    std::string p = dir_;
    if (p.back() != '/') {
        p += '/';
    }
    return p + name;
}

bool pc_disk_store::open(const std::string & dir, size_t max_bytes) {
    dir_ = dir;
    max_bytes_ = max_bytes;
    bytes_ = opaque_bytes_ = 0;
    lru_clock_ = 0;
    active_ = false;
    index_.clear();
    if (dir_.empty()) {
        return false;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    if (ec) {
        fprintf(stderr, "[pcd] cannot create %s: %s\n", dir_.c_str(), ec.message().c_str());
        return false;
    }
    size_t n_ok = 0, n_bad = 0, n_unknown = 0;
    for (const auto & de : std::filesystem::directory_iterator(dir_, ec)) {
        if (ec) {
            break;
        }
        if (!de.is_regular_file(ec)) {
            continue;
        }
        const std::string name = de.path().filename().string();
        if (name.size() < 4 || name.compare(name.size() - 4, 4, ".pcn") != 0) {
            continue;
        }
        const std::string path = de.path().string();
        const size_t fbytes = (size_t)de.file_size(ec);
        pc_disk_meta m;
        const int st = read_header(path, m);
        if (st == 2) {
            // unreadable or truncated: drop it (a real record is written to a
            // temp file first and only then renamed, so it is never torn)
            std::filesystem::remove(path, ec);
            n_bad++;
            continue;
        }
        if (st == 1) {
            // written by a newer engine: keep it, but do not let it shrink the
            // budget we can manage (it is never evicted)
            opaque_bytes_ += fbytes;
            n_unknown++;
            continue;
        }
        if (m.depth <= 0) {
            std::filesystem::remove(path, ec);
            n_bad++;
            continue;
        }
        m.name = name;
        m.file_bytes = fbytes;
        // seed the LRU order from the file mtime so the newest survive a restart
        const auto ft = de.last_write_time(ec);
        if (!ec) {
            m.lru = (uint64_t)ft.time_since_epoch().count();
        }
        lru_clock_ = std::max(lru_clock_, m.lru);
        index_[m.hash] = m;
        bytes_ += fbytes;
        n_ok++;
    }
    active_ = true;
    if (n_ok || n_bad || n_unknown) {
        fprintf(stderr, "[pcd] %s: %zu records (%.1f MB), %zu discarded, %zu unknown%s\n", dir_.c_str(), n_ok,
                (double)bytes_ / (1024.0 * 1024.0), n_bad, n_unknown, max_bytes_ ? "" : " (unbounded)");
    }
    return true;
}

int pc_disk_store::read_header(const std::string & path, pc_disk_meta & m) const {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return 2;
    }
    uint8_t h[kHeaderBytes];
    f.read((char *)h, sizeof(h));
    if (f.gcount() != (std::streamsize)sizeof(h)) {
        return 2;
    }
    if (std::memcmp(h, kMagic, 4) != 0) {
        return 2;
    }
    const uint32_t version = rd_le32(h + 4);
    if (version != kVersion) {
        return 1; // unknown format: treated as opaque, not parsed
    }
    if (rd_le32(h + 8) != kHeaderBytes || rd_le32(h + 12) != kEndianMarker) {
        return 2;
    }
    const uint32_t flags = rd_le32(h + 16);
    m.hash = rd_le64(h + 20);
    m.depth = (int32_t)rd_le32(h + 28);
    m.blob_bytes = (int32_t)rd_le32(h + 32);
    m.state_bytes = (int32_t)rd_le32(h + 36);
    m.has_state = (flags & kFlagHasState) != 0;
    for (int i = 0; i < kPcBlockToks; i++) {
        m.toks[i] = (int32_t)rd_le32(h + 40 + (size_t)i * 4);
    }
    if (m.blob_bytes < 0 || m.state_bytes < 0) {
        return 2;
    }
    return 0;
}

const pc_disk_meta * pc_disk_store::find(uint64_t hash, const int32_t * toks) const {
    auto it = index_.find(hash);
    if (it == index_.end()) {
        return nullptr;
    }
    if (toks && std::memcmp(it->second.toks, toks, sizeof(it->second.toks)) != 0) {
        return nullptr; // hash collision: the tokens decide
    }
    return &it->second;
}

bool pc_disk_store::load(const pc_disk_meta & m, void * blob, float * state) const {
    std::ifstream f(path_of(m.name), std::ios::binary);
    if (!f) {
        return false;
    }
    uint8_t h[kHeaderBytes];
    f.read((char *)h, sizeof(h));
    if (f.gcount() != (std::streamsize)sizeof(h) || std::memcmp(h, kMagic, 4) != 0) {
        return false;
    }
    if (rd_le32(h + 4) != kVersion) {
        return false;
    }
    if (blob && m.blob_bytes > 0) {
        f.read((char *)blob, m.blob_bytes);
        if (f.gcount() != (std::streamsize)m.blob_bytes) {
            return false;
        }
    } else if (m.blob_bytes > 0) {
        f.seekg(m.blob_bytes, std::ios::cur); // state-only read: skip the blob
    }
    if (state && m.has_state && m.state_bytes > 0) {
        f.read((char *)state, m.state_bytes);
        if (f.gcount() != (std::streamsize)m.state_bytes) {
            return false;
        }
    }
    return true;
}

bool pc_disk_store::write_record(const pc_disk_meta & m, const void * blob, const float * state) {
    uint8_t h[kHeaderBytes];
    std::memcpy(h, kMagic, 4);
    wr_le32(h + 4, kVersion);
    wr_le32(h + 8, kHeaderBytes);
    wr_le32(h + 12, kEndianMarker);
    wr_le32(h + 16, m.has_state ? kFlagHasState : 0u);
    wr_le64(h + 20, m.hash);
    wr_le32(h + 28, (uint32_t)m.depth);
    wr_le32(h + 32, (uint32_t)m.blob_bytes);
    wr_le32(h + 36, (uint32_t)m.state_bytes);
    for (int i = 0; i < kPcBlockToks; i++) {
        wr_le32(h + 40 + (size_t)i * 4, (uint32_t)m.toks[i]);
    }
    const std::string path = path_of(m.name);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            return false;
        }
        f.write((const char *)h, sizeof(h));
        if (m.blob_bytes > 0 && blob) {
            f.write((const char *)blob, m.blob_bytes);
        }
        if (m.has_state && m.state_bytes > 0 && state) {
            f.write((const char *)state, m.state_bytes);
        }
        f.flush();
        if (!f) {
            std::remove(tmp.c_str());
            return false;
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

bool pc_disk_store::store(uint64_t hash, int32_t depth, const int32_t * toks, const void * blob, size_t blob_bytes,
                          const float * state, size_t state_bytes) {
    if (!active_) {
        return false;
    }
    pc_disk_meta m;
    m.hash = hash;
    m.depth = depth;
    m.name = hex_name(hash);
    m.has_state = state != nullptr && state_bytes > 0;
    m.blob_bytes = (int32_t)blob_bytes;
    m.state_bytes = m.has_state ? (int32_t)state_bytes : 0;
    if (toks) {
        std::memcpy(m.toks, toks, sizeof(m.toks));
    }
    if (!write_record(m, blob, state)) {
        return false;
    }
    std::error_code ec;
    m.file_bytes = (size_t)std::filesystem::file_size(path_of(m.name), ec);
    if (ec) {
        m.file_bytes = (size_t)kHeaderBytes + blob_bytes + m.state_bytes;
    }
    auto it = index_.find(hash);
    if (it != index_.end()) {
        bytes_ -= it->second.file_bytes;
    }
    m.lru = ++lru_clock_;
    index_[hash] = m;
    bytes_ += m.file_bytes;
    enforce_budget();
    return true;
}

bool pc_disk_store::erase(uint64_t hash) {
    auto it = index_.find(hash);
    if (it == index_.end()) {
        return false;
    }
    std::error_code ec;
    std::filesystem::remove(path_of(it->second.name), ec);
    bytes_ -= it->second.file_bytes;
    index_.erase(it);
    return true;
}

void pc_disk_store::touch(uint64_t hash) {
    auto it = index_.find(hash);
    if (it != index_.end()) {
        it->second.lru = ++lru_clock_;
    }
}

void pc_disk_store::enforce_budget() {
    if (max_bytes_ == 0) {
        return; // unbounded
    }
    while (bytes_ + opaque_bytes_ > max_bytes_ && !index_.empty()) {
        auto best = index_.begin();
        for (auto it = index_.begin(); it != index_.end(); ++it) {
            if (it->second.lru < best->second.lru) {
                best = it;
            }
        }
        std::error_code ec;
        std::filesystem::remove(path_of(best->second.name), ec);
        bytes_ -= best->second.file_bytes;
        index_.erase(best);
    }
}

void pc_disk_store::print_stats(const char * tag) const {
    if (!active_) {
        return;
    }
    fprintf(stderr, "[pcd] %s: %zu records (%.1f MB / %.1f MB) in %s\n", tag, index_.size(),
            (double)bytes_ / (1024.0 * 1024.0), max_bytes_ ? (double)max_bytes_ / (1024.0 * 1024.0) : 0.0,
            dir_.c_str());
}

} // namespace si
