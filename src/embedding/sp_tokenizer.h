#pragma once
// SentencePiece (Unigram / "t5" / UGM) tokenizer for GGUF embedding models.
//
// BGE-M3's GGUF declares `tokenizer.ggml.model = t5`: a Unigram model with
// SentencePiece's NFKC-style normalization compiled into a
// `precompiled_charsmap` (an XOR-compressed double array plus replacement
// strings).  The engine's BPE tokenizer cannot encode it, so embedding models
// get their own small, self-contained tokenizer here.  Nothing in this file
// depends on SYCL or on the LLM engine.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gguf.h"

namespace si {
namespace embedding {

struct sp_tokenizer {
    std::vector<std::string> pieces;
    std::vector<float> scores_;
    std::vector<int> types_; // GGUF token_type: 2/3/4 are special
    int bos = -1, eos = -1, unk = -1, pad = -1, sep = -1, mask = -1;
    bool add_bos = false, add_eos = false, add_sep = false;
    bool escape_whitespace = true, add_space_prefix = true, remove_extra_ws = true;
    bool treat_ws_as_suffix = false;

    // Encode raw text to token ids (with the model's BOS/EOS/SEP convention).
    std::vector<int> encode(const std::string & text) const;

    // Load the tokenizer arrays and charsmap from a GGUF file.  Throws on a
    // malformed charsmap.  Returns false when the file is not a SentencePiece
    // model (so a caller can fall back to the BPE tokenizer).
    bool load(const gguf_file & f);

    bool valid() const { return !pieces.empty(); }

private:
    struct trie_node {
        std::map<unsigned char, std::unique_ptr<trie_node>> next;
        int value = -1;
    };
    std::unique_ptr<trie_node> root_;
    float min_score_ = 1e30f;
    float unknown_score_ = -1e10f;
    float max_score_ = -1e30f;

    // The precompiled charsmap: first u32 is the XCDA byte count, then the
    // packed entries, then NUL-terminated replacement strings.
    std::vector<uint8_t> charsmap_;
    const uint32_t * xcda_ = nullptr;
    size_t xcda_words_ = 0;
    const char * replacements_ = nullptr;
    size_t replacements_size_ = 0;

    void normalize(const std::string & in, std::string & out) const;
    bool normalize_prefix(const std::string & in, size_t off,
                          const char *& rep, size_t & rep_len, size_t & consumed) const;
};

} // namespace embedding
} // namespace si
