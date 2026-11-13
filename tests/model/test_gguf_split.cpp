// Split-GGUF loading, including a metadata-only shard.  Hermetic: writes its own
// files, needs no GGUF and no GPU.
//
// The real trigger is Qwen3.8-Flash-Next's first shard: n_tensors == 0, and its
// header ends exactly at EOF, so the alignment padding lands one byte range past
// the file.  parse_header() rejected that as "bad data offset" and the whole
// model failed to load (exit 78).  A metadata-only shard has no data section, so
// the offset is clamped into range; a shard that *claims* tensors must still be
// rejected, which the second case here asserts.
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    if (ok) {
        printf("  [%s] OK\n", what);
    } else {
        g_fail++;
        printf("  [%s] FAILED\n", what);
    }
}

namespace {

void put_u32(std::vector<uint8_t> & v, uint32_t x) {
    const uint8_t * p = reinterpret_cast<const uint8_t *>(&x);
    v.insert(v.end(), p, p + sizeof(x));
}
void put_u64(std::vector<uint8_t> & v, uint64_t x) {
    const uint8_t * p = reinterpret_cast<const uint8_t *>(&x);
    v.insert(v.end(), p, p + sizeof(x));
}
void put_str(std::vector<uint8_t> & v, const std::string & s) {
    put_u64(v, s.size());
    v.insert(v.end(), s.begin(), s.end());
}

// A GGUF header with `n_tensors` zero-filled tensor infos and no kv pairs.
// The caller decides what (if anything) follows the header.
std::vector<uint8_t> header(uint64_t n_tensors) {
    std::vector<uint8_t> v;
    put_u32(v, 0x46554747); // 'GGUF'
    put_u32(v, 3);          // version
    put_u64(v, n_tensors);
    put_u64(v, 0); // n_kv
    for (uint64_t i = 0; i < n_tensors; i++) {
        put_str(v, "t");
        put_u32(v, 1);       // n_dims
        put_u64(v, 4);       // dims[0] (F32, 4 elements)
        put_u32(v, GGML_TYPE_F32);
        put_u64(v, 0);       // offset within the data section
    }
    return v;
}

bool write_file(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }
    f.write(reinterpret_cast<const char *>(bytes.data()), (std::streamsize)bytes.size());
    return (bool)f;
}

std::string shard_name(const std::string & dir, int idx, int count) {
    char buf[64];
    snprintf(buf, sizeof(buf), "/model-%05d-of-%05d.gguf", idx, count);
    return dir + buf;
}

} // namespace

int main() {
    char tmpl[] = "/tmp/gguf_split_XXXXXX";
    const char * tmp = mkdtemp(tmpl);
    if (!tmp) {
        printf("test_gguf_split: mkdtemp failed\n");
        return 1;
    }
    const std::string dir = tmp;
    const std::string p1 = shard_name(dir, 1, 2);
    const std::string p2 = shard_name(dir, 2, 2);

    // ---- case 1: metadata-only shard 0, one tensor in shard 1.
    // Shard 0's header is 24 bytes and the file is exactly 24 bytes long, so
    // the 32-byte alignment pushes data_offset past EOF - the Qwen3.8 case.
    {
        std::vector<uint8_t> h = header(0);
        check(h.size() == 24, "metadata-only header is 24 bytes (ends at EOF)");
        check(write_file(p1, h), "wrote shard 1 (metadata-only)");

        std::vector<uint8_t> data = header(1); // 57 bytes
        const size_t aligned = (data.size() + 31) / 32 * 32;
        check(aligned == 64, "tensor shard data offset is 64");
        data.resize(aligned, 0);
        const float vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        const uint8_t * fp = reinterpret_cast<const uint8_t *>(vals);
        data.insert(data.end(), fp, fp + sizeof(vals));
        check(write_file(p2, data), "wrote shard 2 (one F32 tensor)");

        gguf_file f;
        bool threw = false;
        try {
            f.load(p1);
        } catch (const std::exception & e) {
            threw = true;
            printf("  load threw: %s\n", e.what());
        }
        check(!threw, "metadata-only shard loads");
        if (!threw) {
            check(f.is_split(), "recognised as split");
            check(f.shards.size() == 2, "two shards mapped");
            check(f.split_count == 2, "split_count == 2");

            const gguf_tensor_info * t = f.find("t");
            check(t != nullptr, "tensor from shard 2 is present");
            if (t) {
                check(t->type == GGML_TYPE_F32, "tensor type is F32");
                check(t->n_elements() == 4, "tensor has 4 elements");
                check(t->data != nullptr, "tensor data pointer is set");
                const uint8_t * d = reinterpret_cast<const uint8_t *>(t->data);
                const uint8_t * s2 = reinterpret_cast<const uint8_t *>(f.shards[1].base);
                check(d >= s2 && d + 16 <= s2 + f.shards[1].size,
                      "tensor data lies inside shard 2");
                float got[4] = {};
                if (t->data) {
                    memcpy(got, t->data, sizeof(got));
                }
                bool same = true;
                for (int i = 0; i < 4; i++) {
                    same = same && got[i] == vals[i];
                }
                check(same, "tensor data reads back the written values");
            }
        }
    }

    // ---- case 2: a shard that claims a tensor but has no room for its data
    // must still fail.  Its header also ends exactly at EOF.
    {
        std::vector<uint8_t> bad = header(1); // 57 bytes, aligned 64 > size
        check(write_file(p1, bad), "wrote a tensor-carrying shard with no data");
        std::vector<uint8_t> unused = header(0);
        check(write_file(p2, unused), "wrote its sibling shard");

        gguf_file f;
        bool threw = false;
        try {
            f.load(p1);
        } catch (const std::exception &) {
            threw = true;
        }
        check(threw, "tensor-carrying shard with no data still throws");
    }

    remove(p1.c_str());
    remove(p2.c_str());
    rmdir(dir.c_str());

    if (g_fail) {
        printf("test_gguf_split: FAILURES: %d\n", g_fail);
        return 1;
    }
    printf("test_gguf_split: OK\n");
    return 0;
}
