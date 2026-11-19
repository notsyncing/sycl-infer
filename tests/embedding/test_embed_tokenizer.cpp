#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "embedding/sp_tokenizer.h"
#include "gguf.h"

namespace {

gguf_kv arr_string(const std::vector<std::string> & v) {
    gguf_kv k;
    k.type = 9; // GGUF_TYPE_ARRAY
    for (const auto & s : v) {
        gguf_kv e;
        e.type = 8;
        e.str = s;
        k.arr.push_back(e);
    }
    return k;
}

gguf_kv arr_f32(const std::vector<float> & v) {
    gguf_kv k;
    k.type = 9;
    for (float f : v) {
        gguf_kv e;
        e.type = 6;
        e.f64 = f;
        k.arr.push_back(e);
    }
    return k;
}

gguf_kv arr_u32(const std::vector<uint32_t> & v) {
    gguf_kv k;
    k.type = 9;
    for (uint32_t x : v) {
        gguf_kv e;
        e.type = 4;
        e.u64 = x;
        k.arr.push_back(e);
    }
    return k;
}

gguf_kv str_kv(const std::string & s) {
    gguf_kv k;
    k.type = 8;
    k.str = s;
    return k;
}

gguf_kv u32_kv(uint32_t v) {
    gguf_kv k;
    k.type = 4;
    k.u64 = v;
    return k;
}

int failures = 0;
void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

} // namespace

int main() {
    // A tiny SentencePiece (t5) vocabulary.  No precompiled charsmap: the
    // normaliser still escapes whitespace when add_space_prefix is set.
    gguf_file f;
    f.kv["tokenizer.ggml.model"] = str_kv("t5");
    f.kv["tokenizer.ggml.tokens"] = arr_string(
        {"<s>", "<pad>", "</s>", "<unk>", "▁", "▁the", "▁cat", "▁sat", "▁on", "▁mat", "."});
    f.kv["tokenizer.ggml.scores"] = arr_f32(
        {0.f, 0.f, 0.f, 0.f, -1.f, -2.f, -2.f, -2.f, -2.f, -2.f, -1.f});
    f.kv["tokenizer.ggml.token_type"] = arr_u32({3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1});
    f.kv["tokenizer.ggml.bos_token_id"] = u32_kv(0);
    f.kv["tokenizer.ggml.eos_token_id"] = u32_kv(2);
    f.kv["tokenizer.ggml.unknown_token_id"] = u32_kv(3);
    f.kv["tokenizer.ggml.padding_token_id"] = u32_kv(1);
    f.kv["tokenizer.ggml.add_bos_token"] = u32_kv(1);
    f.kv["tokenizer.ggml.add_eos_token"] = u32_kv(1);
    f.kv["tokenizer.ggml.add_sep_token"] = u32_kv(1);

    si::embedding::sp_tokenizer tok;
    check(tok.load(f), "load a t5 tokenizer");
    check(tok.valid(), "tokenizer is valid");
    check(tok.bos == 0 && tok.eos == 2 && tok.unk == 3 && tok.pad == 1, "special ids");

    const auto ids = tok.encode("the cat sat");
    // The dp[0] sentinel must NOT be emitted as an extra <unk> (the bug this
    // test pins), and SEP must not duplicate EOS.
    const std::vector<int> want = {0, 5, 6, 7, 2};
    check(ids == want, "encode('the cat sat') == BOS + 3 words + EOS");
    if (ids != want) {
        fprintf(stderr, "  got:");
        for (int id : ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }

    const auto long_ids = tok.encode("the cat sat on the mat.");
    check(long_ids.size() == 9, "encode a longer sentence");
    int n_unk = 0, n_eos = 0;
    for (int id : long_ids) {
        if (id == tok.unk) ++n_unk;
        if (id == tok.eos) ++n_eos;
    }
    check(n_unk == 0, "no unknown tokens for an in-vocab sentence");
    check(n_eos == 1, "exactly one EOS, no duplicated SEP");

    if (failures) {
        fprintf(stderr, "test_embed_tokenizer: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_embed_tokenizer: OK\n");
    return 0;
}
