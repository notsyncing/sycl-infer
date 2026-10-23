// sycl-infer-onnx: the ONNX model service.
//
// A separate binary from the LLM engine on purpose: it needs neither the
// SYCL device pass nor oneDNN, it holds a different set of models on the
// device, and it can be built and run on a machine with no GPU at all (the
// CPU plugin).  The container for it is therefore tiny.
//
//   sycl-infer-onnx --config models.json
//   sycl-infer-onnx --model layout=/path/pp_doclayoutv3.onnx --device gpu
//
// See docs/design/17-onnx-service.md for the config format and the wire
// protocol.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "service.h"

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s [options]\n"
            "\n"
            "options:\n"
            "  --config <file.json>      the model table (see below)\n"
            "  --model id=path[:device]  add a model (repeatable)\n"
            "  --device cpu|gpu|auto     the compute device (default gpu; the\n"
            "                            same names as `sycl-infer --device`;\n"
            "                            SYCL_DEVICE_FILTER pins the card)\n"
            "  --host <addr>             bind address (default 127.0.0.1)\n"
            "  --port <n>                port (default 8890)\n"
            "  --threads <n>             HTTP worker threads (default 4)\n"
            "  --drop-outputs <name>     do not read this model output (repeatable)\n"
            "  --dnnl <0|1>              oneDNN f32 convolution/matmul (default 1)\n"
            "  --control-host <addr>     the router's control plane address\n"
            "  --control-port <n>        the control plane port (0 = off)\n"
            "  --control-token <s>       the control plane's shared secret\n"
            "  --control-token-file <f>  ... read it from a file\n"
            "\n"
            "the config file:\n"
            "  {\n"
            "    \"device\": \"gpu\",\n"
            "    \"models\": {\n"
            "      \"layout\": { \"path\": \"/path/pp_doclayoutv3.onnx\" },\n"
            "      \"formula\": { \"path\": \"/path/pp_formulanet_plus_m.onnx\",\n"
            "                     \"device\": \"cpu\" }\n"
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
        } else if (a == "--drop-outputs") {
            cfg.drop_outputs.push_back(next("--drop-outputs"));
        } else if (a == "--dnnl") {
            cfg.dnnl = atoi(next("--dnnl").c_str()) != 0;
            cfg.dnnl_set = true;
        } else if (a == "--control-host") {
            cfg.control_host = next("--control-host");
        } else if (a == "--control-port") {
            cfg.control_port = atoi(next("--control-port").c_str());
        } else if (a == "--control-token") {
            cfg.control_token = next("--control-token");
        } else if (a == "--control-token-file") {
            cfg.control_token_file = next("--control-token-file");
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
