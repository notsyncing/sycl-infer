#pragma once
// The one JSON serializer the server uses.
//
// It is not `j.dump()`: a stream's UTF-8 tail can be split mid-sequence, so the
// buffer normally holds valid text but a genuinely malformed sequence must not
// abort the whole server - those bytes are encoded as U+FFFD instead of throwing.
// Both the request and the response side need that, so it lives here rather than
// in whichever one happened to define it first.
#include <string>

#include "json.hpp"

namespace si {

using json = nlohmann::json;

inline std::string dump_json(const json & j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

} // namespace si
