#pragma once

// The control plane's wire protocol: JSON-RPC 2.0 over TCP, one JSON value
// per line (NDJSON).
//
// It is shared by the two ends - the worker's control server
// (`src/server/control.cpp`) and the router's client
// (`src/router/rpc_client.cpp`) - so the method names and the frame helpers
// cannot drift apart.  It is header-only on purpose: the router must not
// link the SYCL core, and the worker must not link the router.
//
// Why NDJSON: nlohmann's `dump()` escapes newlines inside strings, so a
// line is unambiguously one JSON value, and the traffic can be read with
// `nc` when something goes wrong.  Why JSON-RPC 2.0: it is a standard, so a
// remote worker written in another language can be dropped in later without
// inventing a second protocol.

#include "json.hpp"

#include <cerrno>
#include <cstdint>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace si {
namespace rpc {

// The protocol's JSON type.  It is nlohmann's, so both ends speak the same
// dialect without a conversion layer.
using json = nlohmann::json;

// Bumped when the shape changes; the router refuses a worker that speaks a
// different one rather than misreading its reports.
//
// Version 2: the handshake carries a token (`hello`'s params.token) and the
// connection is authenticated from then on.  A worker with no token
// configured keeps accepting an unauthenticated handshake (the loopback
// default), so the version bump is about the protocol's shape, not about
// making the token mandatory.
constexpr int kProtocolVersion = 2;

// The methods.
constexpr const char * kHello = "hello";
constexpr const char * kHealth = "health";
constexpr const char * kStats = "stats";
constexpr const char * kShutdown = "shutdown";
constexpr const char * kReload = "reload";

// JSON-RPC error codes.
constexpr int kParseError = -32700;
constexpr int kInvalidRequest = -32600;
constexpr int kMethodNotFound = -32601;
constexpr int kInvalidParams = -32602;
constexpr int kInternalError = -32603;
// The server's own range (-32000..-32099) for the authentication failure.
constexpr int kUnauthorized = -32001;

// Compare two secrets without leaking their length or content through the
// time: a token is a shared secret, and a router that connects to a
// worker's control plane is exactly the kind of peer that must not be able
// to learn it one byte at a time.
inline bool secret_equal(const std::string & a, const std::string & b) {
    if (a.size() != b.size()) {
        return false;
    }
    unsigned char diff = 0;
    for (size_t i = 0; i < a.size(); i++) {
        diff |= (unsigned char)(a[i] ^ b[i]);
    }
    return diff == 0;
}

// One request.
inline json make_request(int64_t id, const std::string & method, const json & params = json::object()) {
    json j;
    j["jsonrpc"] = "2.0";
    j["id"] = id;
    j["method"] = method;
    if (!params.is_null() && !(params.is_object() && params.empty())) {
        j["params"] = params;
    }
    return j;
}

// One successful reply.
inline json make_result(int64_t id, json result) {
    json j;
    j["jsonrpc"] = "2.0";
    j["id"] = id;
    j["result"] = std::move(result);
    return j;
}

// One failed reply.
inline json make_error(int64_t id, int code, const std::string & message) {
    json j;
    j["jsonrpc"] = "2.0";
    j["id"] = id;
    j["error"]["code"] = code;
    j["error"]["message"] = message;
    return j;
}

// A server-initiated event (no id).
inline json make_event(const char * name, json params = json::object()) {
    json j;
    j["jsonrpc"] = "2.0";
    j["method"] = "event";
    j["params"]["event"] = name;
    if (!params.is_null()) {
        j["params"]["data"] = std::move(params);
    }
    return j;
}

// The frame: one JSON value plus a newline.
inline std::string frame(const json & j) {
    return j.dump() + "\n";
}

// Write everything (a short frame can still be split by the kernel).
inline bool write_all(int fd, const std::string & s) {
    size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        off += (size_t)n;
    }
    return true;
}

// Read one line (up to the next '\n').  Returns false on EOF, on a timeout,
// or on a line that is longer than `max_bytes`.
//
// The leftover bytes stay in the reader: a frame can arrive split across
// several reads, and the bytes after the newline belong to the NEXT frame.
//
// The timeout matters on both ends: the router's client must not hang when a
// worker wedges, and the worker's server must not keep a connection open
// forever.
class line_reader {
public:
    bool next(int fd, std::string & out, int timeout_ms, size_t max_bytes = 1u << 20) {
        out.clear();
        for (;;) {
            const size_t nl = buf_.find('\n');
            if (nl != std::string::npos) {
                out = buf_.substr(0, nl);
                buf_.erase(0, nl + 1);
                return true;
            }
            if (buf_.size() > max_bytes) {
                return false;
            }
            struct pollfd pfd;
            pfd.fd = fd;
            pfd.events = POLLIN;
            pfd.revents = 0;
            const int pr = ::poll(&pfd, 1, timeout_ms);
            if (pr < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            if (pr == 0) {
                return false; // timed out
            }
            char tmp[4096];
            const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            if (n == 0) {
                return false; // the peer closed
            }
            buf_.append(tmp, (size_t)n);
        }
    }

private:
    std::string buf_;
};

} // namespace rpc
} // namespace si
