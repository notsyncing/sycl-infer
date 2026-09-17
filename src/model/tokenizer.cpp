#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <stdexcept>

#include "unicode.h"

namespace {

std::string byte_decode(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        size_t len = unicode_len_utf8(text[i]);
        if (i + len > text.size()) {
            out += text[i];
            i++;
            continue;
        }
        std::string cp = text.substr(i, len);
        uint8_t b = unicode_utf8_to_byte(cp);
        out += (char)b;
        i += len;
    }
    return out;
}

} // namespace

void tokenizer::load(const gguf_file & f) {
    auto * toks = f.meta("tokenizer.ggml.tokens");
    auto * types = f.meta("tokenizer.ggml.token_type");
    auto * merges = f.meta("tokenizer.ggml.merges");
    if (!toks || !toks->is_arr()) {
        throw std::runtime_error("gguf: missing tokenizer tokens");
    }

    n_vocab = (int)toks->arr.size();
    token_text.resize(n_vocab);
    token_type.assign(n_vocab, 1);
    for (int i = 0; i < n_vocab; i++) {
        token_text[i] = toks->arr[i].str;
        token_to_id[token_text[i]] = i;
    }
    if (types && types->is_arr()) {
        for (int i = 0; i < n_vocab && i < (int)types->arr.size(); i++) {
            token_type[i] = (int)types->arr[i].u64;
        }
    }

    if (merges && merges->is_arr()) {
        int n = (int)merges->arr.size();
        for (int i = 0; i < n; i++) {
            const std::string & word = merges->arr[i].str;
            size_t pos = word.find(' ', 1);
            if (pos == std::string::npos) {
                continue;
            }
            bpe_rank.emplace(std::make_pair(word.substr(0, pos), word.substr(pos + 1)), i);
        }
    }

    for (int i = 0; i < n_vocab; i++) {
        if (token_type[i] == 3 || token_type[i] == 4 || token_type[i] == 2) {
            special_ids.push_back(i);
        }
    }
    std::sort(special_ids.begin(), special_ids.end(),
              [&](int a, int b) { return token_text[a].size() > token_text[b].size(); });

    auto tid = [&](const char * s) {
        auto it = token_to_id.find(s);
        return it == token_to_id.end() ? -1 : it->second;
    };
    eos_id = tid("<|im_end|>");
    eot_id = tid("<|endoftext|>");
    pad_id = tid("<|vision_pad|>");
    im_start_id = tid("<|im_start|>");
    im_end_id = tid("<|im_end|>");
    think_id = tid("<think>");
    endthink_id = tid("</think>");
    if (eos_id < 0) {
        eos_id = f.get_u32("tokenizer.ggml.eos_token_id", 0);
    }
    if (eot_id < 0) {
        eot_id = eos_id;
    }
}

namespace {

std::vector<std::string> bpe_one(const tokenizer & tk, const std::string & word) {
    // symbols are byte-encoded utf8 chunks
    std::vector<std::string> syms;
    size_t off = 0;
    while (off < word.size()) {
        size_t len = std::min(word.size() - off, unicode_len_utf8(word[off]));
        syms.push_back(word.substr(off, len));
        off += len;
    }
    if (syms.empty()) {
        return syms;
    }

    while (syms.size() > 1) {
        int best = -1;
        int best_rank = INT_MAX;
        for (size_t i = 0; i + 1 < syms.size(); i++) {
            auto it = tk.bpe_rank.find(std::make_pair(syms[i], syms[i + 1]));
            if (it != tk.bpe_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best = (int)i;
            }
        }
        if (best < 0) {
            break;
        }
        syms[best] += syms[best + 1];
        syms.erase(syms.begin() + best + 1);
    }
    return syms;
}

void encode_plain(const tokenizer & tk, const std::string & raw_text, std::vector<int> & out) {
    static const std::vector<std::string> regex_exprs = {
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| "
        "?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };
    auto words = unicode_regex_split(raw_text, regex_exprs, /*byte_encode=*/true);
    for (const auto & word : words) {
        auto syms = bpe_one(tk, word);
        for (const auto & s : syms) {
            auto it = tk.token_to_id.find(s);
            if (it != tk.token_to_id.end()) {
                out.push_back(it->second);
            } else {
                for (size_t i = 0; i < s.size();) {
                    size_t len = std::min(s.size() - i, unicode_len_utf8(s[i]));
                    auto it2 = tk.token_to_id.find(s.substr(i, len));
                    if (it2 != tk.token_to_id.end()) {
                        out.push_back(it2->second);
                    }
                    i += len;
                }
            }
        }
    }
}

} // namespace

std::vector<int> tokenizer::encode(const std::string & text, bool parse_special) const {
    std::vector<int> out;
    if (!parse_special || special_ids.empty()) {
        encode_plain(*this, text, out);
        return out;
    }

    // split on special tokens (longest first)
    size_t pos = 0;
    std::string pending;
    auto flush = [&]() {
        if (!pending.empty()) {
            encode_plain(*this, pending, out);
            pending.clear();
        }
    };
    while (pos < text.size()) {
        int matched = -1;
        size_t match_len = 0;
        for (int id : special_ids) {
            const std::string & s = token_text[id];
            if (s.empty()) {
                continue;
            }
            if (text.compare(pos, s.size(), s) == 0) {
                matched = id;
                match_len = s.size();
                break;
            }
        }
        if (matched >= 0) {
            flush();
            out.push_back(matched);
            pos += match_len;
        } else {
            pending += text[pos];
            pos++;
        }
    }
    flush();
    return out;
}

std::string tokenizer::token_piece(int id) const {
    if (id < 0 || id >= n_vocab) {
        return "";
    }
    int type = token_type[id];
    if (type == 3 || type == 4 || type == 2) {
        return token_text[id];
    }
    return byte_decode(token_text[id]);
}

std::string tokenizer::decode(const std::vector<int> & ids) const {
    std::string out;
    for (int id : ids) {
        out += token_piece(id);
    }
    return out;
}
