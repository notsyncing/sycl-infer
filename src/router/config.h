#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace si {
namespace router {

// One model's configuration.
//
// The memory block is the operator's explicit VRAM budget for this model
// (docs/design/16-router.md §9).  `vram_budget_mb` is the router's admission
// budget - it is NOT a worker flag, because nothing below the driver can
// enforce it (Level Zero has no VRAM partitioning).  The three knobs next to
// it ARE worker flags, and they are where the actual cap comes from: the KV
// pool stops growing at its cap instead of failing, and the recurrent state
// shrinks with the slot count.
struct model_config {
    std::string id;         // the routing key: the client's `model` field
    // "llm" (the GGUF engine) or "onnx" (the document-pipeline runtime).
    std::string backend = "llm";
    std::string path;       // GGUF/ONNX path inside the worker's mount namespace
    std::string binary;     // worker executable (local models only)
    std::string http_host = "127.0.0.1";
    int http_port = 0;      // the worker's OpenAI HTTP port
    std::string control_host; // the worker's control plane (JSON-RPC over TCP)
    int control_port = 0;
    // The control plane's shared secret: the router presents it in the
    // handshake.  A file is the container-friendly way to keep it out of
    // the config (and out of the process's command line).
    std::string control_token;
    std::string control_token_file;
    bool remote = false;    // a remote worker: connect, do not spawn
    std::vector<std::string> args; // extra worker CLI args
    bool is_onnx() const { return backend == "onnx"; }
    std::vector<std::string> env;  // "K=V" pairs for the child

    // Device placement.  `device_group` is the budget grouping key (two models
    // in the same group are assumed to share one card); `device_filter` is
    // passed to the child as SYCL_DEVICE_FILTER, which is how a worker is
    // pinned to one card without touching the engine's device selection.
    // `device_count` is how many cards the model runs on: a model split with
    // --layer-map uses several, and its budget is the TOTAL over them.
    std::string device_group = "default";
    // The worker's own backend name: cpu|gpu|auto (the same --device values
    // as the LLM engine).  Empty for an LLM model means "leave it to the
    // engine"; an ONNX worker defaults to gpu.
    std::string device_kind;
    std::string device_filter;
    int device_count = 1;

    // ONNX-only: the model's outputs the pipeline never reads, and whether
    // the fixed-shape convolutions/matmuls use oneDNN.
    std::vector<std::string> drop_outputs;
    bool dnnl = true;
    bool dnnl_set = false;

    uint64_t vram_budget_mb = 0;
    int kv_cap_mb = -1;     // -> --kv-cap-mb
    int max_slots = -1;     // -> --max-slots
    int pc_vram_mb = -1;    // -> --pc-vram-mb
    std::string pc_dir;     // -> --pc-dir

    // Derived by the loader: the child's argv.
    std::vector<std::string> argv;

    std::string http_base() const {
        return "http://" + http_host + ":" + std::to_string(http_port);
    }
};

struct router_config {
    std::string listen_host = "127.0.0.1";
    int listen_port = 8080;
    bool access_log = false;
    std::vector<std::string> bearer_tokens;
    int request_timeout_s = 3600; // upstream data-plane timeout
    // How long a worker may take to load.  It has to cover the device
    // image's JIT on a card without an AOT build (minutes on an Iris Xe),
    // so it is generous by default; a worker that dies during startup is
    // NOT retried, so killing a slow-but-progressing load is expensive.
    int startup_timeout_s = 3600;
    int health_interval_s = 10;   // readiness / liveness poll period
    int health_failures = 3;      // consecutive failures before a restart
    int max_restarts = 3;         // restart attempts before giving up
    int restart_backoff_s = 2;    // first backoff; it doubles per attempt
    std::string log_dir = "logs";
    std::string binary;           // default LLM worker executable
    std::string binary_onnx = "sycl-infer-onnx"; // default ONNX worker executable
    uint64_t device_vram_mb = 0;  // declared usable VRAM per device group
    std::vector<std::string> default_env;
    std::vector<model_config> models;
};

// Parse and validate a configuration document.
//
// The document is JSON with comments: `//` line comments and `/* */` block
// comments are accepted, because a configuration file that a human edits
// needs them and JSON's own grammar has no room for them.  nlohmann's parser
// does the stripping (`ignore_comments`), so this costs nothing.
//
// Throws std::runtime_error with a human-readable message on any error: a
// configuration mistake must fail at startup, never on the first request.
router_config parse_config(const std::string & text);

router_config load_config_file(const std::string & path);

// The child's argv, derived from the model config.  Exposed because the
// argument list is part of the contract between the config and the worker,
// and the tests pin it.
std::vector<std::string> build_worker_argv(const model_config & m);

} // namespace router
} // namespace si
