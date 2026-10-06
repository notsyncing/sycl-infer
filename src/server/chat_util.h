#pragma once
#include <cstddef>
#include <string>

namespace si {

// Buffers token pieces so only complete UTF-8 sequences are emitted.
struct utf8_stream_buffer {
    std::string pending;

    std::string push(const std::string & piece) {
        pending += piece;
        size_t i = 0, last_good = 0;
        while (i < pending.size()) {
            const unsigned char c = (unsigned char)pending[i];
            size_t len;
            if (c < 0x80) {
                len = 1;
            } else if ((c >> 5) == 0x6) {
                len = 2;
            } else if ((c >> 4) == 0xE) {
                len = 3;
            } else if ((c >> 3) == 0x1E) {
                len = 4;
            } else {
                i++;
                last_good = i;
                continue;
            }
            if (i + len > pending.size()) {
                break;
            }
            bool ok = true;
            for (size_t k = 1; k < len; k++) {
                if ((pending[i + k] & 0xC0) != 0x80) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                i++;
                last_good = i;
                continue;
            }
            i += len;
            last_good = i;
        }
        std::string out = pending.substr(0, last_good);
        pending.erase(0, last_good);
        return out;
    }
    std::string flush() {
        std::string out = pending;
        pending.clear();
        return out;
    }
};


#include <cctype>


// Text-normalization helpers shared by the chat renderers and the streaming
// response parser.
inline bool is_space(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

inline std::string trim_spaces(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && is_space(s[b])) {
        b++;
    }
    while (e > b && is_space(s[e - 1])) {
        e--;
    }
    return s.substr(b, e - b);
}

inline std::string lstrip_newlines(const std::string & s) {
    size_t b = 0;
    while (b < s.size() && (s[b] == '\n' || s[b] == '\r')) {
        b++;
    }
    return s.substr(b);
}

inline std::string rstrip_newlines(const std::string & s) {
    size_t e = s.size();
    while (e > 0 && (s[e - 1] == '\n' || s[e - 1] == '\r')) {
        e--;
    }
    return s.substr(0, e);
}

inline std::string lstrip_spaces(const std::string & s) {
    size_t b = 0;
    while (b < s.size() && is_space(s[b])) {
        b++;
    }
    return s.substr(b);
}

inline std::string rstrip_spaces(const std::string & s) {
    size_t e = s.size();
    while (e > 0 && is_space(s[e - 1])) {
        e--;
    }
    return s.substr(0, e);
}

} // namespace si
