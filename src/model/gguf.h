#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

enum ggml_type : uint32_t {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_BF16 = 30,
    GGML_TYPE_COUNT,
};

const char * ggml_type_name(uint32_t t);

// bytes required for `n` elements of type t (n must be a multiple of block size)
size_t ggml_type_size(uint32_t t); // bytes per block
size_t ggml_blck_size(uint32_t t); // elements per block
size_t ggml_row_bytes(uint32_t t, uint64_t n_elems);

struct gguf_kv {
    uint32_t type = 0; // raw gguf type id
    uint64_t u64 = 0;
    int64_t i64 = 0;
    double f64 = 0;
    std::string str;
    std::vector<gguf_kv> arr;

    uint32_t as_u32() const {
        return (uint32_t)u64;
    }
    int32_t as_i32() const {
        return (int32_t)i64;
    }
    float as_f32() const {
        return (float)f64;
    }
    bool is_arr() const {
        return type == 9;
    }
};

struct gguf_tensor_info {
    std::string name;
    std::vector<uint64_t> dims; // dims[0] = innermost (ggml ne[0])
    uint32_t type = 0;
    uint64_t offset = 0; // relative to data base
    const void * data = nullptr;

    uint64_t n_elements() const {
        uint64_t n = 1;
        for (auto d : dims) {
            n *= d;
        }
        return n;
    }
    // rows = product of dims[1:], row length = dims[0]
    uint64_t n_rows() const {
        uint64_t n = 1;
        for (size_t i = 1; i < dims.size(); i++) {
            n *= dims[i];
        }
        return n;
    }
    size_t nbytes() const {
        return ggml_row_bytes(type, n_elements());
    }
};

struct gguf_file {
    void * map_base = nullptr;
    size_t map_size = 0;
    size_t data_offset = 0;
    std::vector<gguf_tensor_info> tensors;
    std::map<std::string, size_t> tensor_index;
    std::map<std::string, gguf_kv> kv;
    uint32_t version = 0;

    ~gguf_file();
    void load(const std::string & path);

    const gguf_tensor_info * find(const std::string & name) const {
        auto it = tensor_index.find(name);
        return it == tensor_index.end() ? nullptr : &tensors[it->second];
    }
    const gguf_kv * meta(const std::string & key) const {
        auto it = kv.find(key);
        return it == kv.end() ? nullptr : &it->second;
    }
    uint32_t get_u32(const std::string & key, uint32_t def = 0) const {
        auto * v = meta(key);
        return v ? v->as_u32() : def;
    }
    float get_f32(const std::string & key, float def = 0.f) const {
        auto * v = meta(key);
        return v ? v->as_f32() : def;
    }
    const std::string * get_str(const std::string & key) const {
        auto * v = meta(key);
        return v ? &v->str : nullptr;
    }
};
