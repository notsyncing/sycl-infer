#pragma once

// The router's end of the control plane: a JSON-RPC 2.0 client over TCP.
//
// It is deliberately a separate program from the worker, so it cannot share
// the worker's code; the wire protocol is shared through
// `src/common/jsonrpc.h` instead.

#include "json.hpp"
#include "jsonrpc.h"

#include <cstdint>
#include <string>

namespace si {
namespace router {

using json = nlohmann::json;

class rpc_client {
public:
    // `token` is the control plane's shared secret: it is presented in the
    // handshake (and only there - the connection is authenticated from then
    // on).
    rpc_client(std::string host, int port, std::string token = std::string());
    ~rpc_client();

    // Open the connection.  Returns false (with a reason) when the worker is
    // not there: the router treats that as "no control plane" and falls back
    // to the HTTP API, so a worker without --control-port still works.
    bool connect(int timeout_ms, std::string & err);
    void close();
    bool connected() const {
        return fd_ >= 0;
    }

    // One request/response round trip.  `result` receives the "result"
    // object; on an error reply `err` receives the message and the call
    // returns false.
    bool call(const std::string & method, const json & params, json & result, std::string & err,
              int timeout_ms);

    const std::string & host() const {
        return host_;
    }
    int port() const {
        return port_;
    }

private:
    std::string host_;
    int port_;
    std::string token_;
    int fd_ = -1;
    int64_t next_id_ = 1;
    // The line reader keeps the bytes of a frame that arrived split across
    // two reads, so it has to live across the calls.
    rpc::line_reader reader_;
};

} // namespace router
} // namespace si
