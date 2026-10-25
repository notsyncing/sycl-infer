#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

const char * ggml_type_name(uint32_t t) {
    switch (t) {
    case GGML_TYPE_F32: return "F32";
    case GGML_TYPE_F16: return "F16";
    case GGML_TYPE_Q4_0: return "Q4_0";
    case GGML_TYPE_Q4_1: return "Q4_1";
    case GGML_TYPE_Q5_0: return "Q5_0";
    case GGML_TYPE_Q5_1: return "Q5_1";
    case GGML_TYPE_Q8_0: return "Q8_0";
    case GGML_TYPE_Q8_1: return "Q8_1";
    case GGML_TYPE_Q2_K: return "Q2_K";
    case GGML_TYPE_Q3_K: return "Q3_K";
    case GGML_TYPE_Q4_K: return "Q4_K";
    case GGML_TYPE_Q5_K: return "Q5_K";
    case GGML_TYPE_Q6_K: return "Q6_K";
    case GGML_TYPE_Q8_K: return "Q8_K";
    case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
    case GGML_TYPE_IQ4_NL: return "IQ4_NL";
    case GGML_TYPE_IQ3_S: return "IQ3_S";
    case GGML_TYPE_IQ4_XS: return "IQ4_XS";
    case GGML_TYPE_BF16: return "BF16";
    default: return "?";
    }
}

size_t ggml_blck_size(uint32_t t) {
    switch (t) {
    case GGML_TYPE_F32:
    case GGML_TYPE_F16:
    case GGML_TYPE_BF16: return 1;
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1: return 32;
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q8_K:
    case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ4_XS: return 256;
    case GGML_TYPE_IQ4_NL: return 32;
    default: throw std::runtime_error("unsupported ggml type");
    }
}

size_t ggml_type_size(uint32_t t) {
    switch (t) {
    case GGML_TYPE_F32: return 4;
    case GGML_TYPE_F16: return 2;
    case GGML_TYPE_BF16: return 2;
    case GGML_TYPE_Q4_0: return 2 + 16; // d + qs
    case GGML_TYPE_Q4_1: return 4 + 16; // dm + qs
    case GGML_TYPE_Q5_0: return 2 + 4 + 16;
    case GGML_TYPE_Q5_1: return 4 + 4 + 16;
    case GGML_TYPE_Q8_0: return 2 + 32;
    case GGML_TYPE_Q8_1: return 4 + 32;
    case GGML_TYPE_Q2_K: return 4 + 16 + 64;
    case GGML_TYPE_Q3_K: return 2 + 32 + 64 + 12;
    case GGML_TYPE_Q4_K: return 4 + 12 + 128;
    case GGML_TYPE_Q5_K: return 4 + 12 + 32 + 128;
    case GGML_TYPE_Q6_K: return 2 + 16 + 128 + 64;
    case GGML_TYPE_Q8_K: return 4 + 256 + 32;
    case GGML_TYPE_IQ3_XXS: return 2 + 32 + 64; // 2 + QK_K/8 + QK_K/4
    case GGML_TYPE_IQ4_NL: return 2 + 16; // 2 + 16
    case GGML_TYPE_IQ3_S: return 2 + 64 + 32 + 8 + 4; // 2 + QK_K/4 + QK_K/8 + QK_K/32 + 4
    case GGML_TYPE_IQ4_XS: return 2 + 2 + 128 + 4; // 2 + 2 + QK_K/2 + QK_K/64
    default: throw std::runtime_error("unsupported ggml type size");
    }
}

size_t ggml_row_bytes(uint32_t t, uint64_t n_elems) {
    size_t bs = ggml_blck_size(t);
    if (n_elems % bs != 0) {
        throw std::runtime_error("row size not divisible by block size");
    }
    return (size_t)(n_elems / bs) * ggml_type_size(t);
}

namespace {

struct reader {
    const uint8_t * p;
    const uint8_t * end;

    void need(size_t n) const {
        if (p + n > end) {
            throw std::runtime_error("gguf: unexpected end of file");
        }
    }
    template <typename T> T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string read_str() {
        uint64_t n = read<uint64_t>();
        need(n);
        std::string s((const char *)p, n);
        p += n;
        return s;
    }
    gguf_kv read_value(uint32_t type) {
        gguf_kv v;
        v.type = type;
        switch (type) {
        case 0:
            v.u64 = read<uint8_t>();
            v.i64 = (int64_t)v.u64;
            break;
        case 1:
            v.i64 = read<int8_t>();
            v.u64 = (uint64_t)v.i64;
            break;
        case 2:
            v.u64 = read<uint16_t>();
            v.i64 = (int64_t)v.u64;
            break;
        case 3:
            v.i64 = read<int16_t>();
            v.u64 = (uint64_t)v.i64;
            break;
        case 4:
            v.u64 = read<uint32_t>();
            v.i64 = (int64_t)v.u64;
            break;
        case 5:
            v.i64 = read<int32_t>();
            v.u64 = (uint64_t)v.i64;
            break;
        case 6: v.f64 = read<float>(); break;
        case 7:
            v.u64 = read<uint8_t>();
            v.i64 = (int64_t)v.u64;
            break;
        case 8: v.str = read_str(); break;
        case 9: {
            uint32_t et = read<uint32_t>();
            uint64_t n = read<uint64_t>();
            v.arr.reserve(std::min<uint64_t>(n, 1u << 20));
            for (uint64_t i = 0; i < n; i++) {
                v.arr.push_back(read_value(et));
            }
            break;
        }
        case 10:
            v.u64 = read<uint64_t>();
            v.i64 = (int64_t)v.u64;
            break;
        case 11:
            v.i64 = read<int64_t>();
            v.u64 = (uint64_t)v.i64;
            break;
        case 12: v.f64 = read<double>(); break;
        default: throw std::runtime_error("gguf: unknown kv type");
        }
        return v;
    }
};

} // namespace

gguf_file::~gguf_file() {
    if (map_base) {
        // one range: for a split file this also unmaps the per-shard MAP_FIXED
        // sub-mappings placed inside the reservation
        munmap(map_base, map_size);
    }
    for (const shard_t & s : shards) {
        if (s.fd >= 0) {
            close(s.fd);
        }
    }
    if (shards.empty() && fd >= 0) {
        close(fd);
    }
}

void gguf_file::drop_cache(size_t off, size_t len) const {
    if (len == 0) {
        return;
    }
    if (!is_split()) {
        if (fd >= 0) {
            posix_fadvise(fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);
        }
        return;
    }
    // Translate the reservation-relative range into each shard's own file range.
    // The caller's range is page-aligned and may span the padding between two
    // shards; clip it to every shard it touches.
    const uint64_t a = off;
    const uint64_t b = off + len;
    for (const shard_t & s : shards) {
        const uint64_t s0 = (uint64_t)((const uint8_t *)s.base - (const uint8_t *)map_base);
        const uint64_t s1 = s0 + s.size;
        const uint64_t lo = std::max<uint64_t>(a, s0);
        const uint64_t hi = std::min<uint64_t>(b, s1);
        if (hi > lo) {
            posix_fadvise(s.fd, (off_t)(lo - s0), (off_t)(hi - lo), POSIX_FADV_DONTNEED);
        }
    }
}

namespace {

struct parsed_header {
    uint32_t version = 0;
    size_t data_offset = 0;
    std::map<std::string, gguf_kv> kv;
    std::vector<gguf_tensor_info> tensors;
};

// Parse the header block of one already-mapped shard.  `base` is that shard's
// own mapping: tensor data pointers are left unset here and rebased by load().
parsed_header parse_header(const uint8_t * base, size_t size) {
    reader r{base, base + size};
    parsed_header h;
    uint32_t magic = r.read<uint32_t>();
    if (magic != 0x46554747) { // 'GGUF' little endian
        throw std::runtime_error("not a gguf file");
    }
    h.version = r.read<uint32_t>();
    uint64_t n_tensors = r.read<uint64_t>();
    uint64_t n_kv = r.read<uint64_t>();

    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = r.read_str();
        uint32_t t = r.read<uint32_t>();
        h.kv[key] = r.read_value(t);
    }

    h.tensors.resize(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor_info & ti = h.tensors[i];
        ti.name = r.read_str();
        uint32_t nd = r.read<uint32_t>();
        ti.dims.resize(nd);
        for (uint32_t d = 0; d < nd; d++) {
            ti.dims[d] = r.read<uint64_t>();
        }
        ti.type = r.read<uint32_t>();
        ti.offset = r.read<uint64_t>();
    }

    auto u32_of = [&h](const std::string & key, uint32_t def) {
        auto it = h.kv.find(key);
        return it == h.kv.end() ? def : it->second.as_u32();
    };
    const uint32_t alignment = u32_of("general.alignment", 32);
    const size_t pos = (size_t)(r.p - base);
    h.data_offset = (pos + alignment - 1) / alignment * alignment;
    if (h.data_offset > size) {
        throw std::runtime_error("gguf: bad data offset");
    }
    return h;
}

int open_shard(const std::string & path, size_t & size) {
    int f = open(path.c_str(), O_RDONLY);
    if (f < 0) {
        throw std::runtime_error("cannot open " + path);
    }
    struct stat st;
    if (fstat(f, &st) != 0) {
        close(f);
        throw std::runtime_error("fstat failed");
    }
    size = (size_t)st.st_size;
    return f;
}

// If `path` names one shard of a split GGUF ("...-00002-of-00004.gguf"), return
// the sibling paths in index order.  The count is taken from the file name; the
// metadata's split.count is only used to validate it.
std::vector<std::string> shard_paths(const std::string & path) {
    const std::string suffix = ".gguf";
    const size_t of = path.rfind("-of-");
    if (of == std::string::npos || path.size() < of + 4 + 5 + suffix.size()
        || path.compare(path.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return {path};
    }
    const size_t dash = path.rfind('-', of - 1);
    if (dash == std::string::npos) {
        return {path};
    }
    const std::string idx_s = path.substr(dash + 1, of - dash - 1);
    const std::string cnt_s = path.substr(of + 4, path.size() - suffix.size() - (of + 4));
    if (idx_s.size() != 5 || cnt_s.size() != 5) {
        return {path};
    }
    for (char c : idx_s + cnt_s) {
        if (c < '0' || c > '9') {
            return {path};
        }
    }
    const int count = atoi(cnt_s.c_str());
    if (count <= 1) {
        return {path};
    }
    std::vector<std::string> out;
    out.reserve((size_t)count);
    for (int i = 1; i <= count; i++) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%05d", i);
        out.push_back(path.substr(0, dash + 1) + buf + "-of-" + cnt_s + suffix);
    }
    return out;
}

} // namespace

void gguf_file::load(const std::string & path) {
    std::vector<std::string> paths = shard_paths(path);
    if (paths.size() == 1) {
        size_t size = 0;
        int f = open_shard(path, size);
        map_size = size;
        map_base = mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, f, 0);
        if (map_base == MAP_FAILED) {
            close(f);
            throw std::runtime_error("mmap failed");
        }
        fd = f;
        shard_t s;
        s.fd = f;
        s.base = map_base;
        s.size = map_size;
        parsed_header h = parse_header((const uint8_t *)map_base, map_size);
        version = h.version;
        kv = std::move(h.kv);
        tensors = std::move(h.tensors);
        data_offset = h.data_offset;
        s.data_offset = data_offset;
        shards.push_back(s);
        split_count = 1;
        split_no = 0;
        for (size_t i = 0; i < tensors.size(); i++) {
            tensor_index[tensors[i].name] = i;
        }
        for (auto & ti : tensors) {
            ti.data = (const uint8_t *)map_base + data_offset + ti.offset;
        }
        return;
    }

    // Split file: reserve one contiguous range and place every shard inside it
    // at a page-aligned slot, so a tensor's address minus map_base is a stable
    // offset through the whole (multi-file) model.
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    struct pending {
        int f = -1;
        size_t size = 0;
        parsed_header h;
    };
    std::vector<pending> pend(paths.size());
    size_t total = 0;
    for (size_t i = 0; i < paths.size(); i++) {
        pend[i].f = open_shard(paths[i], pend[i].size);
        // Parse the header through a temporary mapping: the metadata is a few MB
        // and this keeps the parse independent of the placement arithmetic.
        void * tmp = mmap(nullptr, pend[i].size, PROT_READ, MAP_PRIVATE, pend[i].f, 0);
        if (tmp == MAP_FAILED) {
            close(pend[i].f);
            throw std::runtime_error("mmap failed");
        }
        try {
            pend[i].h = parse_header((const uint8_t *)tmp, pend[i].size);
        } catch (...) {
            munmap(tmp, pend[i].size);
            close(pend[i].f);
            throw;
        }
        munmap(tmp, pend[i].size);
        total += (pend[i].size + page - 1) / page * page;
        if (i == 0) {
            version = pend[i].h.version;
            split_count = pend[i].h.kv.count("split.count") ? (int)pend[i].h.kv["split.count"].as_u32() : (int)paths.size();
            split_no = pend[i].h.kv.count("split.no") ? (int)pend[i].h.kv["split.no"].as_u32() : 0;
        } else {
            const int no = pend[i].h.kv.count("split.no") ? (int)pend[i].h.kv["split.no"].as_u32() : (int)i;
            if (no != (int)i) {
                throw std::runtime_error("gguf: shard " + paths[i] + " has split.no " + std::to_string(no));
            }
        }
        if (pend[i].h.kv.count("split.count") && (int)pend[i].h.kv["split.count"].as_u32() != (int)paths.size()) {
            throw std::runtime_error("gguf: split.count does not match the shard file names");
        }
        for (auto & e : pend[i].h.kv) {
            if (!kv.count(e.first)) {
                kv[e.first] = e.second;
            }
        }
    }

    map_size = total;
    map_base = mmap(nullptr, total, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (map_base == MAP_FAILED) {
        throw std::runtime_error("gguf: cannot reserve the split mapping");
    }
    size_t off = 0;
    for (size_t i = 0; i < paths.size(); i++) {
        const size_t slot = (pend[i].size + page - 1) / page * page;
        void * at = mmap((uint8_t *)map_base + off, slot, PROT_READ, MAP_PRIVATE | MAP_FIXED, pend[i].f, 0);
        if (at == MAP_FAILED) {
            throw std::runtime_error("gguf: cannot map shard " + paths[i]);
        }
        shard_t s;
        s.fd = pend[i].f;
        s.base = at;
        s.size = pend[i].size;
        s.data_offset = pend[i].h.data_offset;
        shards.push_back(s);
        for (auto & t : pend[i].h.tensors) {
            gguf_tensor_info copy = t;
            copy.data = (const uint8_t *)at + pend[i].h.data_offset + copy.offset;
            tensor_index[copy.name] = tensors.size();
            tensors.push_back(std::move(copy));
        }
        off += slot;
    }
    data_offset = shards[0].data_offset;
    fd = shards[0].fd;
}
