#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"

struct pair_hash {
    size_t operator()(const std::pair<std::string, std::string> & p) const {
        return std::hash<std::string>()(p.first) * 1000003u ^ std::hash<std::string>()(p.second);
    }
};

struct tokenizer {
    std::vector<std::string> token_text; // raw text as stored in gguf
    std::vector<int> token_type;         // gguf token type
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> bpe_rank;
    std::vector<int> special_ids; // sorted by text length desc
    int n_vocab = 0;
    int eos_id = -1;   // <|im_end|>
    int eot_id = -1;   // <|endoftext|>
    int pad_id = -1;
    int im_start_id = -1, im_end_id = -1, think_id = -1, endthink_id = -1;

    void load(const gguf_file & f);

    // Tokenize with optional special-token parsing and BOS handling
    std::vector<int> encode(const std::string & text, bool parse_special = true) const;

    // Decode id -> raw bytes
    std::string decode(const std::vector<int> & ids) const;
    std::string token_piece(int id) const;
};
