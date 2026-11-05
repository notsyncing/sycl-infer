// sycl-infer-mineru: the standalone MinerU two-step reference client.
//
// It is deliberately NOT part of the server and NOT a pipeline: the steps use
// only the model's public image-input/text-output capability, so the server
// stays a pure OpenAI-compatible worker (docs/design/22-pipelines.md §2.1).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

#include "common/exit_codes.h"
#include "engine.h"
#include "layout.h"
#include "mineru.h"

using namespace si;

namespace {

void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s --model <gguf> --mmproj <mmproj.gguf> --image <page.png> [options]\n"
            "\n"
            "MinerU2.5 two-step document extraction (layout + per-block prompts) on a\n"
            "Qwen2-VL trunk.  It is a reference/test client, not a server pipeline.\n"
            "\n"
            "  --model <gguf>          text model (qwen2vl architecture)\n"
            "  --mmproj <gguf>         vision projector (required)\n"
            "  --image <file>          page image (required)\n"
            "  --ctx N                 context length (default 8192)\n"
            "  --device cpu|gpu|auto   compute backend (default auto)\n"
            "  --layer-map L:dev,...   layer placement, e.g. 0-11:gpu,12-23:cpu\n"
            "  --kv-type T             KV storage type: i4|i8|bf16|f16|f32 or K:V\n"
            "  --layout-side N         layout input side (default 1036)\n"
            "  --only-type T           debug: extract only one block type\n"
            "  --max-tokens N          per-call generation limit (default 4096)\n"
            "  --debug                  print the layout response, block geometry and\n"
            "                           the host/device vision self-check\n"
            "  -h, --help               this message\n",
            prog);
}

} // namespace

int main(int argc, char ** argv) {
    std::string model;
    std::string mmproj;
    std::string image;
    std::string layer_map;
    std::string kv_type;
    int ctx = 8192;
    int device = -1;
    mineru_options opt;
    bool help = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", a.c_str());
                return std::string();
            }
            return argv[++i];
        };
        if (a == "--model") {
            model = next();
        } else if (a == "--mmproj") {
            mmproj = next();
        } else if (a == "--image") {
            image = next();
        } else if (a == "--ctx") {
            ctx = std::max(1, std::atoi(next().c_str()));
        } else if (a == "--device") {
            const std::string v = next();
            if (v == "cpu" || v == "host") {
                device = 1;
            } else if (v == "gpu") {
                device = 0;
            } else if (v == "auto") {
                device = -1;
            } else {
                fprintf(stderr, "error: --device expects cpu|gpu|auto\n");
                return 1;
            }
        } else if (a == "--layer-map") {
            layer_map = next();
        } else if (a == "--kv-type") {
            kv_type = next();
        } else if (a == "--layout-side") {
            opt.layout_side = std::max(1, std::atoi(next().c_str()));
        } else if (a == "--only-type") {
            opt.only_type = next();
        } else if (a == "--max-tokens") {
            opt.max_tokens = std::max(1, std::atoi(next().c_str()));
        } else if (a == "--debug") {
            opt.debug = true;
        } else if (a == "-h" || a == "--help") {
            help = true;
        } else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "error: unknown option %s\n", a.c_str());
            usage(argv[0]);
            return 1;
        } else {
            fprintf(stderr, "error: unexpected argument '%s'\n", a.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    if (help) {
        usage(argv[0]);
        return 0;
    }
    if (image.empty()) {
        fprintf(stderr, "error: --image <page.png> is required\n");
        usage(argv[0]);
        return 1;
    }
    if (model.empty()) {
        fprintf(stderr, "error: --model <gguf> is required\n");
        return 1;
    }
    if (mmproj.empty()) {
        fprintf(stderr, "error: --mmproj <mmproj.gguf> is required\n");
        return 1;
    }
    if (!kv_type.empty()) {
        kv_dtype_t kt, vt;
        if (!kv_dtype_parse_pair(kv_type.c_str(), kt, vt)) {
            fprintf(stderr, "error: --kv-type expects i4|i8|bf16|f16|f32, or K:V\n");
            return 1;
        }
        kv_dtype_set_kv(kt, vt);
    }

    engine_config ec;
    ec.model_path = model;
    ec.max_seq = ctx;
    ec.n_splits = 16;
    ec.n_blocks = std::max(512, (ctx + kBlockSize - 1) / kBlockSize);
    ec.device = device;
    ec.layer_map = layer_map;
    try {
        engine e(ec);
        return run_mineru(e, mmproj, image, opt, stdout);
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        fprintf(stderr, "(exit %d: a startup failure - do not retry, the same attempt would fail the same way)\n",
                kExitStartupFailure);
        return kExitStartupFailure;
    }
}
