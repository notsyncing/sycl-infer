#include "sp_tokenizer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace si {
namespace embedding {

static size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

bool sp_tokenizer::load(const gguf_file & f) {
    const std::string * model = f.get_str("tokenizer.ggml.model");
    if (!model || (*model != "t5" && *model != "ugm" && *model != "sentencepiece")) {
        return false;
    }
    const gguf_kv * toks = f.meta("tokenizer.ggml.tokens");
    const gguf_kv * scores = f.meta("tokenizer.ggml.scores");
    if (!toks || !toks->is_arr() || toks->arr.empty()) return false;

    pieces.reserve(toks->arr.size());
    for (const auto & v : toks->arr) pieces.push_back(v.str);
    if (scores && scores->is_arr()) {
        scores_.reserve(scores->arr.size());
        for (const auto & v : scores->arr) scores_.push_back(v.as_f32());
    }
    if (scores_.size() != pieces.size()) scores_.assign(pieces.size(), 0.f);

    const gguf_kv * types = f.meta("tokenizer.ggml.token_type");
    if (types && types->is_arr()) {
        types_.reserve(types->arr.size());
        for (const auto & v : types->arr) types_.push_back((int)v.as_u32());
    }
    if (types_.size() != pieces.size()) types_.assign(pieces.size(), 1);

    auto id_of = [&](const char * s) -> int {
        for (size_t i = 0; i < pieces.size(); ++i) if (pieces[i] == s) return (int)i;
        return -1;
    };
    bos = id_of("<s>");
    eos = id_of("</s>");
    unk = f.get_u32("tokenizer.ggml.unknown_token_id", (uint32_t)id_of("<unk>"));
    pad = f.get_u32("tokenizer.ggml.padding_token_id", (uint32_t)id_of("<pad>"));
    sep = f.get_u32("tokenizer.ggml.seperator_token_id", (uint32_t)eos);
    mask = f.get_u32("tokenizer.ggml.mask_token_id", (uint32_t)id_of("<mask>"));
    add_bos = f.meta("tokenizer.ggml.add_bos_token") ? f.meta("tokenizer.ggml.add_bos_token")->as_u32() != 0 : true;
    add_eos = f.meta("tokenizer.ggml.add_eos_token") ? f.meta("tokenizer.ggml.add_eos_token")->as_u32() != 0 : true;
    add_sep = f.meta("tokenizer.ggml.add_sep_token") ? f.meta("tokenizer.ggml.add_sep_token")->as_u32() != 0 : true;
    escape_whitespace = true;

    // GGUF token types: 0 undefined, 1 normal, 2 unknown, 3 control,
    // 4 user-defined, 5 unused, 6 byte.  The Unigram matcher holds normal,
    // user-defined and unused pieces; control tokens are handled as specials.
    root_ = std::make_unique<trie_node>();
    for (size_t id = 0; id < pieces.size(); ++id) {
        const int t = types_[id];
        if (t == 1 && scores_[id] < min_score_) min_score_ = scores_[id];
        if (t == 1 || t == 4 || t == 5) {
            trie_node * n = root_.get();
            for (unsigned char c : pieces[id]) {
                auto it = n->next.find(c);
                if (it == n->next.end()) it = n->next.emplace(c, std::make_unique<trie_node>()).first;
                n = it->second.get();
            }
            n->value = (int)id;
            if (t == 4) scores_[id] = 0.f; // user-defined pieces are always preferred
        }
    }
    unknown_score_ = min_score_ - 10.f;

    const gguf_kv * cs = f.meta("tokenizer.ggml.precompiled_charsmap");
    if (cs && cs->is_arr() && cs->arr.size() > sizeof(uint32_t)) {
        charsmap_.resize(cs->arr.size());
        for (size_t i = 0; i < cs->arr.size(); ++i) charsmap_[i] = (uint8_t)cs->arr[i].as_u32();
        uint32_t words = 0;
        std::memcpy(&words, charsmap_.data(), sizeof(uint32_t));
        if (words > charsmap_.size() || sizeof(uint32_t) + words >= charsmap_.size()) {
            throw std::runtime_error("sp_tokenizer: bad precompiled_charsmap size");
        }
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        words = __builtin_bswap32(words);
#endif
        xcda_ = (const uint32_t *)(charsmap_.data() + sizeof(uint32_t));
        xcda_words_ = words / sizeof(uint32_t);
        replacements_ = (const char *)(charsmap_.data() + sizeof(uint32_t) + words);
        replacements_size_ = charsmap_.size() - sizeof(uint32_t) - words;
        for (size_t i = 0; i < xcda_words_; ++i) {
            const uint32_t * w = (const uint32_t *)(charsmap_.data() + sizeof(uint32_t)) + i;
            uint32_t v = *w;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            v = __builtin_bswap32(v);
#endif
            const_cast<uint32_t *>(w)[0] = v; // endian-correct in place
        }
    }
    return true;
}

// XOR-compressed compact double array (XCDA), as in SentencePiece.
struct xcda_view {
    const uint32_t * a;
    size_t n;
    uint32_t node(size_t i) const {
        if (i >= n) throw std::runtime_error("sp_tokenizer: XCDA index out of bounds");
        return a[i];
    }
    uint32_t base(size_t i) const {
        uint32_t p = node(i);
        return (p >> 10) << ((p & (1u << 9)) >> 6);
    }
    uint32_t lcheck(size_t i) const { return node(i) & ((1u << 31) | 0xff); }
    bool leaf(size_t i) const { return (node(i) >> 8) & 1; }
    uint32_t value(size_t i) const { return node(i) & ((1u << 31) - 1); }
};

bool sp_tokenizer::normalize_prefix(const std::string & in, size_t off,
                                    const char *& rep, size_t & rep_len, size_t & consumed) const {
    if (off >= in.size()) return false;

    if (xcda_ && xcda_words_) {
        xcda_view v{xcda_, xcda_words_};
        uint32_t idx = v.base(0);
        size_t best = 0, best_off = 0;
        for (size_t p = off; p < in.size(); ++p) {
            unsigned char c = (unsigned char)in[p];
            if (c == 0) break;
            idx ^= c;
            if (v.lcheck(idx) != c) break;
            bool is_leaf = v.leaf(idx);
            idx ^= v.base(idx);
            if (is_leaf) {
                best = p - off + 1;
                best_off = v.value(idx);
            }
        }
        if (best > 0) {
            if (best_off >= replacements_size_) throw std::runtime_error("sp_tokenizer: charsmap replacement out of bounds");
            rep = replacements_ + best_off;
            rep_len = strnlen(rep, replacements_size_ - best_off);
            consumed = best;
            return true;
        }
    }
    rep = in.data() + off;
    rep_len = utf8_len((unsigned char)in[off]);
    consumed = rep_len;
    return true;
}

void sp_tokenizer::normalize(const std::string & in, std::string & out) const {
    out.clear();
    out.reserve(in.size() * 3);
    const std::string space = escape_whitespace ? "\xE2\x96\x81" : " ";
    const bool prepend = add_space_prefix && !treat_ws_as_suffix;
    const bool append = add_space_prefix && treat_ws_as_suffix;
    bool prepended = false, in_word = false;
    for (size_t off = 0; off < in.size();) {
        const char * rep; size_t rep_len, consumed;
        if (!normalize_prefix(in, off, rep, rep_len, consumed)) break;
        for (size_t i = 0; i < rep_len; ++i) {
            char c = rep[i];
            if (c != ' ') {
                if (!in_word) {
                    in_word = true;
                    if ((prepend && !prepended) || remove_extra_ws) {
                        out.append(space);
                        prepended = true;
                    }
                }
                out.push_back(c);
            } else {
                in_word = false;
                if (!remove_extra_ws) out.append(space);
            }
        }
        off += consumed;
    }
    if (append) out.append(space);
}

std::vector<int> sp_tokenizer::encode(const std::string & text) const {
    std::vector<int> out;
    if (!valid()) return out;
    std::string norm;
    normalize(text, norm);
    const size_t n = norm.size();
    if (!n) {
        if (add_bos && bos >= 0) out.push_back(bos);
        if (add_eos && eos >= 0) out.push_back(eos);
        return out;
    }

    struct best { int id = -1; size_t from = 0; double score = -1e300; };
    std::vector<best> dp(n + 1);
    dp[0].id = unk;
    dp[0].from = 0;
    dp[0].score = 0.0;

    for (size_t pos = 0; pos < n;) {
        const double cur = dp[pos].score;
        const size_t cp_len = std::min(utf8_len((unsigned char)norm[pos]), n - pos);
        bool single_cp = false;
        const trie_node * node = root_.get();
        for (size_t p = pos; p < n && node; ++p) {
            auto it = node->next.find((unsigned char)norm[p]);
            node = (it == node->next.end()) ? nullptr : it->second.get();
            if (!node) break;
            if (node->value >= 0) {
                if (p - pos + 1 == cp_len) single_cp = true;
                const double s = cur + scores_[node->value];
                if (s > dp[p + 1].score) dp[p + 1] = {node->value, pos, s};
            }
        }
        if (!single_cp) {
            const double s = cur + unknown_score_;
            const size_t end = pos + cp_len;
            if (s > dp[end].score) dp[end] = {unk, pos, s};
        }
        pos += cp_len;
    }

    // Backtrack to (but not including) the dp[0] sentinel, which carries no
    // token.  Emitting it is what produced a spurious <unk> as the first
    // body token of every input.
    bool prev_unk = false;
    for (size_t p = n; p > 0;) {
        const best & b = dp[p];
        bool is_unk = b.id == unk;
        if (!(prev_unk && is_unk)) out.push_back(b.id);
        prev_unk = is_unk;
        if (b.from == 0) break;
        p = b.from;
    }
    std::reverse(out.begin(), out.end());

    std::vector<int> final;
    if (add_bos && bos >= 0) final.push_back(bos);
    final.insert(final.end(), out.begin(), out.end());
    // llama.cpp adds only BOS and EOS here.  The GGUF's `add_sep_token` is a
    // metadata artefact for BGE-M3 (`seperator_token_id == eos_token_id`);
    // adding SEP as well would duplicate </s> and shift every position.
    if (add_eos && eos >= 0) final.push_back(eos);
    return final;
}

} // namespace embedding
} // namespace si
