#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
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
    case GGML_TYPE_Q8_K: return 256;
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
        munmap(map_base, map_size);
    }
}

void gguf_file::load(const std::string & path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("cannot open " + path);
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        throw std::runtime_error("fstat failed");
    }
    map_size = (size_t)st.st_size;
    map_base = mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map_base == MAP_FAILED) {
        throw std::runtime_error("mmap failed");
    }

    reader r{(const uint8_t *)map_base, (const uint8_t *)map_base + map_size};
    uint32_t magic = r.read<uint32_t>();
    if (magic != 0x46554747) { // 'GGUF' little endian
        throw std::runtime_error("not a gguf file");
    }
    version = r.read<uint32_t>();
    uint64_t n_tensors = r.read<uint64_t>();
    uint64_t n_kv = r.read<uint64_t>();

    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = r.read_str();
        uint32_t t = r.read<uint32_t>();
        kv[key] = r.read_value(t);
    }

    tensors.resize(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor_info & ti = tensors[i];
        ti.name = r.read_str();
        uint32_t nd = r.read<uint32_t>();
        ti.dims.resize(nd);
        for (uint32_t d = 0; d < nd; d++) {
            ti.dims[d] = r.read<uint64_t>();
        }
        ti.type = r.read<uint32_t>();
        ti.offset = r.read<uint64_t>();
        tensor_index[ti.name] = i;
    }

    uint32_t alignment = get_u32("general.alignment", 32);
    size_t pos = (size_t)(r.p - (const uint8_t *)map_base);
    data_offset = (pos + alignment - 1) / alignment * alignment;
    if (data_offset > map_size) {
        throw std::runtime_error("gguf: bad data offset");
    }

    for (auto & ti : tensors) {
        ti.data = (const uint8_t *)map_base + data_offset + ti.offset;
    }
}
