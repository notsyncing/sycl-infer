#pragma once
// The ONNX model service: named ONNX models over HTTP.
//
// The endpoint is deliberately small and binary.  A document pipeline moves
// tensors, not JSON: one 800x800x3 float page is 7.7 MB, so a JSON/base64
// encoding would spend longer in the decoder than the model spends in the
// forward pass.
//
//   POST /v1/onnx/run     the wire protocol in wire.h (raw tensor payloads)
//   GET  /v1/onnx/models  the model table, with each model's ports
//   GET  /health          liveness, the device, and the per-model statistics
//
// The service is model-agnostic: the client names a model and sends its
// inputs.  That is what lets the RapidDoc pipeline keep its own
// post-processing (box filtering, CTC decoding, table HTML assembly) and
// change nothing but the place the inference happens.

#include <string>
#include <vector>

namespace si {

struct onnx_service_config {
    // Loopback by default: the API is unauthenticated, so a routable bind
    // address exposes the models to anyone who can reach the port.
    std::string host = "127.0.0.1";
    int port = 8890;
    int n_threads = 4;
    // The same names as the LLM worker: cpu, gpu, auto (uppercase and GPU.N
    // are accepted for the standalone configs).
    std::string device = "gpu";
    // Extra models, "id=path" or "id=path:device".
    std::vector<std::string> models;
    // A JSON config file:
    //   {"device": "gpu", "models": {"<id>": {"path": "...", "device": "cpu"}}}
    std::string config_path;
    // The router gives a worker one model and its control plane; the
    // standalone --model form accepts the same per-model options.
    std::vector<std::string> drop_outputs;
    bool dnnl = true;
    bool dnnl_set = false;
    std::string control_host = "127.0.0.1";
    int control_port = 0;
    std::string control_token;
    std::string control_token_file;
};

// Run the service until it is stopped.  Returns the process exit code.
int serve_onnx(const onnx_service_config & cfg);

} // namespace si
