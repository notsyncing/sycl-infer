#pragma once
// Turning one chat/completions media part into bytes.
//
// Split out of the request parser because the transport handling is a separate
// concern with its own policy: a `data:` URL and an inline base64 blob are
// decoded here, an `http(s)://` URL goes through url_fetch (which is OFF unless
// PF_MM_URL_FETCH is set - an unauthenticated endpoint that will proxy arbitrary
// URLs is an SSRF hole, see url_fetch.h), and anything else is rejected by name.
// The parser only records *what* the part was; this decides whether the server
// will read it.
#include <cstdint>
#include <string>
#include <vector>

#include "chat.h" // chat_part_kind

namespace si {

// A media part pulled out of a chat message.  `kind` picks the decoder;
// `url`/`data`/`format` are the transport fields (data: URL, inline base64,
// or file bytes).
struct media_part {
    chat_part_kind kind = chat_part_kind::IMAGE;
    std::string url;
    std::string data;
    std::string format;
};

// base64 -> bytes.  Whitespace is skipped, anything else outside the alphabet is
// an error (so a truncated or mis-pasted blob is reported rather than silently
// decoded to the wrong length).
bool b64_decode(const std::string & s, std::vector<uint8_t> & out);

// Fill `bytes` from `part`, or set `err` to a message naming the reason.
bool load_media_bytes(const media_part & part, std::vector<uint8_t> & bytes, std::string & err);

} // namespace si
