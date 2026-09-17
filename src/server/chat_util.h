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

} // namespace si
