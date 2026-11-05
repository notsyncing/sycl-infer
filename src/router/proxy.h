#pragma once

#include "config.h"
#include "registry.h"

#include <httplib.h>

#include <memory>
#include <string>

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
//
// Pipelines are a separate routing table: the owning worker comes from the
// model's `pipelines` config list, the body is forwarded verbatim (no `model`
// rewrite), and the ids never enter /v1/models or model routing.
class proxy {
public:
    proxy(registry & reg, const router_config & cfg);
    void install(httplib::Server & srv);

private:
    void handle_completions(const httplib::Request & req, httplib::Response & res, const char * path);
    void handle_pipelines(const httplib::Request & req, httplib::Response & res);
    void handle_pipeline_run(const httplib::Request & req, httplib::Response & res);
    void forward(const httplib::Request & req, httplib::Response & res, const std::shared_ptr<model_slot> & slot,
                 const std::string & path, const std::string & body, const std::string & log_model);
    // The ONNX binary data plane: the model is the first field of the
    // request body, so the router can route it without decoding the tensors.
    void handle_onnx_run(const httplib::Request & req, httplib::Response & res);
    void handle_onnx_models(const httplib::Request & req, httplib::Response & res);
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
