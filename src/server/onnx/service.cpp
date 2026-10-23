#include "service.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <chrono>

#include <sycl/sycl.hpp>

#include "control.h"
#include "conv_dnnl.h"
#include "exec.h"
#include "graph.h"
#include "onnx_wire.h"

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
        if (tail == "GPU" || tail == "gpu" || tail == "CPU" || tail == "cpu" ||
            tail == "AUTO" || tail == "auto" || tail == "host" ||
            tail.rfind("GPU.", 0) == 0 || tail.rfind("gpu.", 0) == 0) {
            device = tail;
            rest = rest.substr(0, colon);
        }
    }
    path = rest;
    return !path.empty();
}

// The SYCL device the models run on.
sycl::device pick_device(const std::string & name) {
    if (name == "CPU" || name == "cpu" || name == "host") return sycl::device{sycl::cpu_selector_v};
    if (name == "AUTO" || name == "auto" || name.empty()) {
        if (sycl::device{sycl::gpu_selector_v}.is_gpu()) return sycl::device{sycl::gpu_selector_v};
        return sycl::device{sycl::cpu_selector_v};
    }
    if (name.rfind("GPU.", 0) == 0 || name.rfind("gpu.", 0) == 0) {
        const int idx = atoi(name.c_str() + 4);
        int n = 0;
        for (const auto & d : sycl::device{sycl::gpu_selector_v}.get_devices()) {
            if (n++ == idx) return d;
        }
        throw std::runtime_error("there is no GPU with the index " + name);
    }
    if (name == "GPU" || name == "gpu") {
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
    std::map<std::string, std::vector<std::string>> model_drops;
    std::map<std::string, bool> model_dnnl;
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
                if (m.contains("drop_outputs")) {
                    for (const auto & o : m["drop_outputs"]) {
                        model_drops[kv.key()].push_back(o.get<std::string>());
                    }
                }
                if (m.contains("dnnl")) model_dnnl[kv.key()] = m["dnnl"].get<bool>();
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

    // The standalone --model form has no per-model config block; its
    // --drop-outputs/--dnnl flags apply to every model it names.  The router
    // names exactly one.
    if (!cfg.drop_outputs.empty()) {
        for (const auto & m : models) model_drops[m.first] = cfg.drop_outputs;
    }
    if (cfg.dnnl_set) {
        for (const auto & m : models) model_dnnl[m.first] = cfg.dnnl;
    }

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
            if (auto it = model_drops.find(s->id); it != model_drops.end()) {
                s->ex->skip_outputs(it->second);
            }
            if (auto it = model_dnnl.find(s->id); it != model_dnnl.end() && !it->second) {
                s->ex->set_dnnl(false);
            }
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
        const auto th0 = std::chrono::steady_clock::now();
        std::string model;
        std::vector<onnxwire::tensor> inputs;
        std::string err;
        const uint8_t * body = reinterpret_cast<const uint8_t *>(req.body.data());
        if (!onnxwire::decode_request(body, req.body.size(), model, inputs, err)) {
            send_binary(res, onnxwire::encode_response(1, "bad request: " + err, {}));
            return;
        }
        const auto th1 = std::chrono::steady_clock::now();
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
            std::vector<bool> keep(out_names.size(), true);
            for (size_t i = 0; i < out_names.size(); ++i) {
                // An output the model config dropped is not read back by the
                // executor and is not put on the wire.
                keep[i] = !target->ex->output_skipped(i);
                if (keep[i]) outs.emplace_back(out_names[i], out_bufs[i]);
            }
            if (getenv("SI_ONNX_LOOPDBG")) {
                fprintf(stderr, "[onnx] the service: the run of '%s' with %zu inputs\n",
                        target->id.c_str(), feeds.size());
            }
            target->ex->run(feeds, outs);
            // The results, in the wire's format.  The shape comes from the
            // executor: it is only known once the graph has run.
            std::vector<onnxwire::tensor> results;
            for (size_t i = 0; i < out_names.size(); ++i) {
                if (!keep[i]) continue;
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
                t.data = std::move(out_bufs[i]);
                results.push_back(std::move(t));
            }
            target->calls++;
            target->last_ms = target->ex->last_ms();
            target->total_ms += target->last_ms;
            const auto th2 = std::chrono::steady_clock::now();
            std::vector<uint8_t> encoded = onnxwire::encode_response(0, "", results);
            const auto th3 = std::chrono::steady_clock::now();
            send_binary(res, encoded);
            const auto th4 = std::chrono::steady_clock::now();
            if (getenv("SI_ONNX_HOSTTIME")) {
                auto ms = [](std::chrono::steady_clock::time_point a,
                             std::chrono::steady_clock::time_point b) {
                    return std::chrono::duration<double, std::milli>(b - a).count();
                };
                fprintf(stderr,
                        "[onnx] http: '%s': the request %zu bytes, the decode %.1f ms, the run "
                        "%.1f ms, the encode %.1f (%.1f MB), the set_content %.1f\n",
                        target->id.c_str(), req.body.size(), ms(th0, th1), ms(th1, th2), ms(th2, th3),
                        encoded.size() / 1048576.0, ms(th3, th4));
            }
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

    // The router's control plane: the model identity, this worker's device
    // memory and a graceful stop, over the same JSON-RPC protocol as the LLM
    // worker (src/server/control.cpp).  Standalone runs leave it off.
    std::unique_ptr<si::control_server> control;
    const auto started = std::chrono::steady_clock::now();
    if (cfg.control_port > 0) {
        std::string token = cfg.control_token;
        if (!cfg.control_token_file.empty()) {
            std::ifstream f(cfg.control_token_file, std::ios::binary);
            if (!f) {
                fprintf(stderr, "[onnx] cannot read the control token file '%s'\n",
                        cfg.control_token_file.c_str());
                return 1;
            }
            std::getline(f, token);
            while (!token.empty() && (token.back() == '\n' || token.back() == '\r')) token.pop_back();
        }
        auto report = [&]() {
            si::control_report r;
            r.model_id = served_models.empty() ? std::string() : served_models.front()->id;
            r.max_seq = 0;
            r.n_vocab = 0;
            r.n_layer = 0;
            r.inflight = 0;
            r.uptime_s = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - started)
                             .count();
            uint64_t used = 0;
            for (const auto & s : served_models) {
                if (s->ex) used += s->ex->live_bytes() + s->ex->pooled_bytes();
            }
            size_t dw = 0, ds = 0;
            onnx::conv_dnnl_mem(&dw, &ds);
            used += dw + ds;
            if (!served_models.empty()) {
                const std::string & name = served_models.front()->device;
                try {
                    sycl::device d = pick_device(name);
                    si::control_report::device_t dev;
                    dev.index = (name.rfind("GPU.", 0) == 0 || name.rfind("gpu.", 0) == 0)
                                    ? atoi(name.c_str() + 4) : 0;
                    dev.name = d.get_info<sycl::info::device::name>();
                    dev.total_vram_bytes = d.get_info<sycl::info::device::global_mem_size>();
                    dev.used_bytes = used;
                    dev.free_vram_bytes = dev.total_vram_bytes > used
                                              ? dev.total_vram_bytes - used : 0;
                    r.devices.push_back(dev);
                } catch (...) {
                }
            }
            r.host_rss_bytes = si::host_rss_bytes();
            return r;
        };
        control = std::make_unique<si::control_server>(
            report, [&]() { srv.stop(); }, token);
        std::string err;
        if (!control->start(cfg.control_host, cfg.control_port, err)) {
            fprintf(stderr, "[onnx] control plane: %s\n", err.c_str());
            return 1;
        }
        fprintf(stderr, "[onnx] control plane on %s:%d%s\n", cfg.control_host.c_str(),
                control->port(), control->needs_token() ? " (token required)" : "");
    }

    fprintf(stderr, "[onnx] listening on %s:%d (%zu models)\n", cfg.host.c_str(), cfg.port,
            served_models.size());
    const bool listening = srv.listen(cfg.host.c_str(), cfg.port);
    if (control) control->stop();
    if (!listening) {
        fprintf(stderr, "[onnx] failed to listen on %s:%d\n", cfg.host.c_str(), cfg.port);
        return 1;
    }
    return 0;
}

} // namespace si
