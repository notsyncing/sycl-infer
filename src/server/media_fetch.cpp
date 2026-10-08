#include "media_fetch.h"

#include "url_fetch.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace si {

namespace {

int b64_val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') {
        return c - 'A';
    }
    if (c >= 'a' && c <= 'z') {
        return c - 'a' + 26;
    }
    if (c >= '0' && c <= '9') {
        return c - '0' + 52;
    }
    if (c == '+') {
        return 62;
    }
    if (c == '/') {
        return 63;
    }
    return -1;
}

} // namespace

bool b64_decode(const std::string & s, std::vector<uint8_t> & out) {
    out.clear();
    int val = 0, bits = 0;
    for (unsigned char c : s) {
        if (c == '=') {
            break;
        }
        if (c == '\n' || c == '\r' || c == ' ') {
            continue;
        }
        const int v = b64_val(c);
        if (v < 0) {
            return false;
        }
        val = (val << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t)(val >> bits));
        }
    }
    return true;
}

// Resolve a media part to raw bytes.  `data:` URLs are decoded locally;
// http(s) URLs go through url_fetch, which enforces the PF_MM_URL_FETCH policy
// (off by default, and to publicly routable addresses only when on) and bounds
// the download so the endpoint cannot be used as an unbounded proxy;
// `input_audio`'s inline base64 `data` field is decoded directly.  Other schemes
// are rejected.
bool load_media_bytes(const media_part & part, std::vector<uint8_t> & bytes, std::string & err) {
    bytes.clear();
    if (!part.url.empty()) {
        const std::string & url = part.url;
        if (url.rfind("data:", 0) == 0) {
            const size_t comma = url.find(',');
            if (comma == std::string::npos) {
                err = "malformed data URL";
                return false;
            }
            if (url.substr(5, comma - 5).find(";base64") == std::string::npos) {
                err = "only base64 data URLs are supported";
                return false;
            }
            if (!b64_decode(url.substr(comma + 1), bytes)) {
                err = "invalid base64 media data";
                return false;
            }
            return true;
        }
        if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
            return url_fetch(url, bytes, err);
        }
        err = "unsupported media URL (expected data:, http:// or https://)";
        return false;
    }
    if (!part.data.empty()) {
        if (!b64_decode(part.data, bytes)) {
            err = "invalid base64 media data";
            return false;
        }
        return true;
    }
    err = "media part has neither url nor data";
    return false;
}

} // namespace si
