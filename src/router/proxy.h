#pragma once

#include "registry.h"

#include <httplib.h>

namespace si {
namespace router {

// The router's HTTP face: it routes an OpenAI request to the worker that
// owns the requested model and proxies the answer.
//
// Two properties matter and are easy to get wrong:
//
//   * **SSE is forwarded chunk by chunk.**  The router never buffers a
//     stream: a token that the worker produced is a chunk the client sees
//     next, not something that waits for the whole answer.
//   * **A client that goes away cancels the generation.**  The chain is
//     client -> router (the chunked provider's write fails) -> router
//     aborts the upstream request (httplib's Client::stop) -> the worker's
//     socket closes -> the worker's write fails -> the worker cancels the
//     sequence.  That is the contract `src/server/sse.h` documents for the
//     single-hop case; here it has to hold across the extra hop.
class proxy {
public:
    proxy(registry & reg, const router_config & cfg);
    void install(httplib::Server & srv);

private:
    void handle_completions(const httplib::Request & req, httplib::Response & res, const char * path);
    void handle_models(const httplib::Request & req, httplib::Response & res);
    void handle_model_one(const httplib::Request & req, httplib::Response & res);
    void handle_health(const httplib::Request & req, httplib::Response & res);
    void handle_status(const httplib::Request & req, httplib::Response & res);
    bool authorized(const httplib::Request & req) const;
    void access_log(const std::string & method, const std::string & path, int status, const std::string & model,
                    double ms) const;

    registry & reg_;
    router_config cfg_;
};

} // namespace router
} // namespace si
