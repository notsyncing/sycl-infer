// sycl-infer-onnx: the ONNX model service.
//
// A separate binary from the LLM engine on purpose: it needs neither the
// SYCL device pass nor oneDNN, it holds a different set of models on the
// device, and it can be built and run on a machine with no GPU at all (the
// CPU plugin).  The container for it is therefore tiny.
//
//   sycl-infer-onnx --config models.json
//   sycl-infer-onnx --model layout=/path/pp_doclayoutv3.onnx --device GPU
//
// See docs/design/17-onnx-service.md for the config format and the wire
// protocol.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "onnx/service.h"

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s [options]\n"
            "\n"
            "options:\n"
            "  --config <file.json>      the model table (see below)\n"
            "  --model id=path[:device]  add a model (repeatable)\n"
            "  --device GPU|CPU|AUTO     the default device (default GPU)\n"
            "  --host <addr>             bind address (default 127.0.0.1)\n"
            "  --port <n>                port (default 8890)\n"
            "  --threads <n>             HTTP worker threads (default 4)\n"
            "\n"
            "the config file:\n"
            "  {\n"
            "    \"device\": \"GPU\",\n"
            "    \"props\": { \"INFERENCE_PRECISION_HINT\": \"f32\" },\n"
            "    \"models\": {\n"
            "      \"layout\": { \"path\": \"/path/pp_doclayoutv3.onnx\" },\n"
            "      \"formula\": { \"path\": \"/path/pp_formulanet_plus_m.onnx\",\n"
            "                     \"device\": \"CPU\" }\n"
            "    }\n"
            "  }\n"
            "\n"
            "endpoints:\n"
            "  POST /v1/onnx/run     run a model (binary framing, see the docs)\n"
            "  GET  /v1/onnx/models  the model table\n"
            "  GET  /health          liveness and the per-model statistics\n",
            prog);
}

int main(int argc, char ** argv) {
    si::onnx_service_config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", what);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--config") {
            cfg.config_path = next("--config");
        } else if (a == "--model") {
            cfg.models.push_back(next("--model"));
        } else if (a == "--device") {
            cfg.device = next("--device");
        } else if (a == "--host") {
            cfg.host = next("--host");
        } else if (a == "--port") {
            cfg.port = atoi(next("--port").c_str());
        } else if (a == "--threads") {
            cfg.n_threads = atoi(next("--threads").c_str());
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown option '%s'\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }
    return si::serve_onnx(cfg);
}
