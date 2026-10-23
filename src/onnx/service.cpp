#include "service.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include "httplib.h"
#include "json.hpp"

#include "exec.h"
#include "graph.h"
#include "wire.h"

using json = nlohmann::json;

namespace si {

namespace {

// A page image is 7.7 MB and a batch of OCR crops is larger, so the default
// 8 MB payload cap is not enough.
constexpr size_t kMaxPayload = 512u * 1024u * 1024u;

std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// "id=path" or "id=path:device"
bool parse_model_spec(const std::string & spec, std::string & id, std::string & path,
                      std::string & device) {
    const size_t eq = spec.find('=');
    if (eq == std::string::npos || eq == 0) return false;
    id = spec.substr(0, eq);
    std::string rest = spec.substr(eq + 1);
    const size_t colon = rest.rfind(':');
    if (colon != std::string::npos && colon + 1 < rest.size()) {
        const std::string tail = rest.substr(colon + 1);
        if (tail == "GPU" || tail == "CPU" || tail == "AUTO" || tail.rfind("GPU.", 0) == 0) {
            device = tail;
            rest = rest.substr(0, colon);
        }
    }
    path = rest;
    return !path.empty();
}

// The SYCL device the models run on.
sycl::device pick_device(const std::string & name) {
    if (name == "CPU") return sycl::device{sycl::cpu_selector_v};
    if (name == "AUTO" || name.empty()) {
        if (sycl::device{sycl::gpu_selector_v}.is_gpu()) return sycl::device{sycl::gpu_selector_v};
        return sycl::device{sycl::cpu_selector_v};
    }
    if (name.rfind("GPU.", 0) == 0) {
        const int idx = atoi(name.c_str() + 4);
        int n = 0;
        for (const auto & d : sycl::device{sycl::gpu_selector_v}.get_devices()) {
            if (n++ == idx) return d;
        }
        throw std::runtime_error("there is no GPU with the index " + name);
    }
    if (name == "GPU") {
        if (!sycl::device{sycl::gpu_selector_v}.is_gpu()) {
            fprintf(stderr, "[onnx] no GPU is available; running on the CPU\n");
            return sycl::device{sycl::cpu_selector_v};
        }
        return sycl::device{sycl::gpu_selector_v};
    }
    throw std::runtime_error("unknown device '" + name + "'");
}

// One served model.
struct served {
    std::string id;
    std::string path;
    std::string device;
    onnx::exec_graph graph;
    std::unique_ptr<onnx::executor> ex;
    std::mutex mtx; // one inference at a time per model
    uint64_t calls = 0;
    double total_ms = 0.0;
    double last_ms = 0.0;
};

void send_binary(httplib::Response & res, const std::vector<uint8_t> & body) {
    res.set_content(reinterpret_cast<const char *>(body.data()), body.size(), "application/octet-stream");
}

} // namespace

int serve_onnx(const onnx_service_config & cfg) {
    // The models.
    std::vector<std::pair<std::string, std::string>> models; // id, path
    std::map<std::string, std::string> model_devices;
    std::string default_device = cfg.device;

    if (!cfg.config_path.empty()) {
        const json j = json::parse(read_file(cfg.config_path));
        if (j.contains("device")) default_device = j["device"].get<std::string>();
        if (j.contains("models")) {
            for (const auto & kv : j["models"].items()) {
                const json & m = kv.value();
                if (!m.contains("path")) throw std::runtime_error("model '" + kv.key() + "' has no path");
                models.emplace_back(kv.key(), m["path"].get<std::string>());
                if (m.contains("device")) model_devices[kv.key()] = m["device"].get<std::string>();
            }
        }
    }
    for (const std::string & spec : cfg.models) {
        std::string id, path, dev;
        if (!parse_model_spec(spec, id, path, dev)) {
            fprintf(stderr, "[onnx] bad --model spec '%s' (want id=path[:device])\n", spec.c_str());
            return 2;
        }
        models.emplace_back(id, path);
        if (!dev.empty()) model_devices[id] = dev;
    }
    if (models.empty()) {
        fprintf(stderr, "[onnx] no models: pass --config <file.json> or --model id=path\n");
        return 2;
    }

    // The queues: one per device, so the models on the GPU and the ones on
    // the CPU do not fight over the same queue.
    std::map<std::string, sycl::queue> queues;
    auto queue_for = [&](const std::string & dev) -> sycl::queue & {
        auto it = queues.find(dev);
        if (it != queues.end()) return it->second;
        const sycl::device d = pick_device(dev);
        queues.emplace(dev, sycl::queue(d, sycl::property::queue::in_order()));
        return queues[dev];
    };

    std::vector<std::unique_ptr<served>> served_models;
    for (const auto & m : models) {
        const std::string dev = model_devices.count(m.first) ? model_devices[m.first] : default_device;
        auto s = std::unique_ptr<served>(new served());
        s->id = m.first;
        s->path = m.second;
        s->device = dev;
        fprintf(stderr, "[onnx] loading '%s' on %s (%s)\n", s->id.c_str(), dev.c_str(), s->path.c_str());
        try {
            s->graph = onnx::build_graph(s->path, true);
            s->ex.reset(new onnx::executor(s->graph, queue_for(dev)));
        } catch (const std::exception & e) {
            // A model that cannot be loaded is a reported failure, not a
            // reason to take the whole service down: the other models are
            // still useful, and the client gets a clear error for this one.
            fprintf(stderr, "[onnx]   FAILED: %s\n", e.what());
            served_models.push_back(std::move(s));
            continue;
        }
        fprintf(stderr, "[onnx]   %s: inputs", s->id.c_str());
        for (const auto & t : s->graph.input_slots) {
            fprintf(stderr, " %s", s->graph.tensors[t].name.c_str());
        }
        fprintf(stderr, "\n[onnx]   %s: outputs", s->id.c_str());
        for (const auto & t : s->graph.output_slots) {
            fprintf(stderr, " %s", s->graph.tensors[t].name.c_str());
        }
        fprintf(stderr, "\n");
        served_models.push_back(std::move(s));
    }

    httplib::Server srv;
    srv.set_read_timeout(3600, 0);
    srv.set_write_timeout(3600, 0);
    srv.set_payload_max_length(kMaxPayload);
    srv.set_keep_alive_max_count(64);

    srv.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        json j;
        j["status"] = "ok";
        j["device"] = default_device;
        j["models"] = json::array();
        for (const auto & s : served_models) {
            json mj;
            mj["id"] = s->id;
            mj["device"] = s->device;
            mj["calls"] = s->calls;
            mj["total_ms"] = s->total_ms;
            mj["last_ms"] = s->last_ms;
            mj["loaded"] = s->ex != nullptr;
            // The device memory the model holds.  The models share one card,
            // so this is what decides how many of them fit - and a model
            // that fails with "device allocation failed" is a model whose
            // *neighbours* took the memory.
            if (s->ex) {
                mj["vram_bytes"] = s->ex->live_bytes() + s->ex->pooled_bytes();
                mj["vram_live_bytes"] = s->ex->live_bytes();
            }
            j["models"].push_back(mj);
        }
        res.set_content(j.dump(), "application/json");
    });

    srv.Get("/v1/onnx/models", [&](const httplib::Request &, httplib::Response & res) {
        json j;
        j["models"] = json::array();
        for (const auto & s : served_models) {
            json mj;
            mj["id"] = s->id;
            mj["path"] = s->path;
            mj["device"] = s->device;
            mj["loaded"] = s->ex != nullptr;
            json ins = json::array(), outs = json::array();
            for (const auto & t : s->graph.input_slots) {
                ins.push_back(s->graph.tensors[t].name);
            }
            for (const auto & t : s->graph.output_slots) {
                outs.push_back(s->graph.tensors[t].name);
            }
            mj["inputs"] = ins;
            mj["outputs"] = outs;
            // The metadata (the "character" list the OCR and table decoders
            // need).
            for (const auto & kv : s->graph.metadata) {
                if (kv.first == "character") mj["character"] = kv.second;
            }
            j["models"].push_back(mj);
        }
        res.set_content(j.dump(), "application/json");
    });

    srv.Post("/v1/onnx/run", [&](const httplib::Request & req, httplib::Response & res) {
        std::string model;
        std::vector<onnxwire::tensor> inputs;
        std::string err;
        const uint8_t * body = reinterpret_cast<const uint8_t *>(req.body.data());
        if (!onnxwire::decode_request(body, req.body.size(), model, inputs, err)) {
            send_binary(res, onnxwire::encode_response(1, "bad request: " + err, {}));
            return;
        }
        served * target = nullptr;
        for (const auto & s : served_models) {
            if (s->id == model) {
                target = s.get();
                break;
            }
        }
        if (!target) {
            send_binary(res, onnxwire::encode_response(2, "unknown model '" + model + "'", {}));
            return;
        }
        if (!target->ex) {
            send_binary(res, onnxwire::encode_response(2, "model '" + model + "' failed to load", {}));
            return;
        }
        try {
            std::lock_guard<std::mutex> lock(target->mtx);
            // The inputs, by name.
            std::vector<onnxwire::tensor> feeds = inputs;
            // The outputs: the graph's, by name.
            std::vector<std::pair<std::string, std::vector<uint8_t> &>> outs;
            std::vector<std::string> out_names;
            for (const auto & s : target->graph.output_slots) {
                out_names.push_back(target->graph.tensors[s].name);
            }
            std::vector<std::vector<uint8_t>> out_bufs(out_names.size());
            for (size_t i = 0; i < out_names.size(); ++i) outs.emplace_back(out_names[i], out_bufs[i]);
            if (getenv("SI_ONNX_LOOPDBG")) {
                fprintf(stderr, "[onnx] the service: the run of '%s' with %zu inputs\n",
                        target->id.c_str(), feeds.size());
            }
            target->ex->run(feeds, outs);
            // The results, in the wire's format.  The shape comes from the
            // executor: it is only known once the graph has run.
            std::vector<onnxwire::tensor> results;
            for (size_t i = 0; i < out_names.size(); ++i) {
                const onnx::tensor & gt = target->ex->output(i);
                onnxwire::tensor t;
                t.name = out_names[i];
                t.type = gt.dtype == onnxpb::FLOAT
                            ? onnxwire::dtype::f32
                            : (gt.dtype == onnxpb::INT64
                                   ? onnxwire::dtype::i64
                                   : (gt.dtype == onnxpb::INT32 ? onnxwire::dtype::i32
                                                                 : onnxwire::dtype::f32));
                t.shape.assign(gt.shape.begin(), gt.shape.end());
                t.data = out_bufs[i];
                results.push_back(std::move(t));
            }
            target->calls++;
            target->last_ms = target->ex->last_ms();
            target->total_ms += target->last_ms;
            send_binary(res, onnxwire::encode_response(0, "", results));
        } catch (const std::exception & e) {
            fprintf(stderr, "[onnx] the request failed: %s\n", e.what());
            send_binary(res, onnxwire::encode_response(3, e.what(), {}));
        }
    });

    srv.Options(".*", [](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });

    fprintf(stderr, "[onnx] listening on %s:%d (%zu models)\n", cfg.host.c_str(), cfg.port,
            served_models.size());
    if (!srv.listen(cfg.host.c_str(), cfg.port)) {
        fprintf(stderr, "[onnx] failed to listen on %s:%d\n", cfg.host.c_str(), cfg.port);
        return 1;
    }
    return 0;
}

} // namespace si
