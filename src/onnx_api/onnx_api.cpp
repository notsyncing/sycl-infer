// In-process access to the ONNX executor.
//
// The HTTP service and this API run the same graph/executor pair; this file
// only owns a model, a queue and a mutex, and speaks the already-defined
// onnxwire framing.  A host program (RapidDoc) links this instead of starting
// a server, so a document pipeline's tensors never leave the process.

#include <sycl_infer/onnx.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <map>
#include <ze_graph_ext.h>

#include "backend/npu/npu_executor.h"
#include "backend/npu/npu_runtime.h"
#include "device/device_select.h"
#include "exec.h"
#include "graph.h"
#include "json.hpp"
#if __has_include("onnx_wire.h")
#include "onnx_wire.h"
#define SYI_ONNX_NEW_WIRE 1
#else
#include "wire.h"
#define SYI_ONNX_NEW_WIRE 0
#endif

using json = nlohmann::json;

namespace si {
namespace {

thread_local std::string g_error;

void set_error(const std::string & msg) {
    g_error = msg;
}

std::string lower(std::string s) {
    for (char & c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

sycl::device pick_device(const std::string & name) {
    const std::string n = lower(name.empty() ? "auto" : name);
    if (n == "cpu" || n == "host") {
        return sycl::device{sycl::cpu_selector_v};
    }
    if (n == "auto") {
        const std::vector<sycl::device> gpus = si::dev::gpu_devices();
        if (!gpus.empty()) return gpus.front();
        return sycl::device{sycl::cpu_selector_v};
    }
    if (n.rfind("gpu.", 0) == 0) {
        const int idx = std::atoi(n.c_str() + 4);
        const std::vector<sycl::device> gpus = si::dev::gpu_devices();
        if (idx < 0 || idx >= (int)gpus.size()) {
            throw std::runtime_error("there is no GPU with index " + std::to_string(idx));
        }
        return gpus[(size_t)idx];
    }
    if (n == "gpu") {
        const std::vector<sycl::device> gpus = si::dev::gpu_devices();
        if (gpus.empty()) throw std::runtime_error("there is no GPU available");
        return gpus.front();
    }
    throw std::runtime_error("unknown device '" + name + "'");
}

std::vector<std::string> split_csv(const char * s) {
    std::vector<std::string> out;
    if (!s) return out;
    std::string cur;
    for (const char * p = s; *p; ++p) {
        if (*p == ',') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else if (*p != ' ' && *p != '\t') {
            cur.push_back(*p);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// One queue per device, shared by every model on it.  The service does the
// same: the executor and oneDNN's primitive cache are built around an in-order
// queue, and giving each model its own queue makes two models on one card
// race the same device state.
sycl::queue & queue_for(const std::string & device) {
    static std::mutex m;
    static std::map<std::string, std::unique_ptr<sycl::queue>> queues;
    std::lock_guard<std::mutex> lk(m);
    auto it = queues.find(device);
    if (it == queues.end()) {
        const sycl::device d = pick_device(device);
        it = queues.emplace(device, std::make_unique<sycl::queue>(
                                       d, sycl::property::queue::in_order())).first;
    }
    return *it->second;
}

onnxwire::dtype wire_type(int dtype) {
    switch (dtype) {
    case onnxpb::FLOAT: return onnxwire::dtype::f32;
    case onnxpb::FLOAT16: return onnxwire::dtype::f16;
    case onnxpb::INT64: return onnxwire::dtype::i64;
    case onnxpb::INT32: return onnxwire::dtype::i32;
    case onnxpb::INT8: return onnxwire::dtype::i8;
    case onnxpb::UINT8: return onnxwire::dtype::u8;
    case onnxpb::BOOL: return onnxwire::dtype::boolean;
    default: return onnxwire::dtype::f32;
    }
}

} // namespace
} // namespace si

struct syi_onnx_model {
    sycl::queue *q = nullptr;
    si::onnx::exec_graph graph;
    std::unique_ptr<si::onnx::executor> ex;
    std::unique_ptr<si::npu::executor> nx;
    std::mutex m;
    std::vector<std::string> last_names;
    std::vector<std::vector<int64_t>> last_shapes;
    std::vector<std::vector<uint8_t>> last_data;
    std::vector<int32_t> last_dtype;
    std::string id;
    std::string path;
    std::string device;
    std::map<std::string, int> nx_reduce;
};

namespace {
void maybe_trim(syi_onnx_model & m) {
    const char * e = std::getenv("SYCL_INFER_TRIM_MB");
    size_t mb = e && *e ? (size_t)std::strtoull(e, nullptr, 10) : 1024;
    if (mb == 0 || !m.ex) return;
    if (m.ex->pooled_bytes() > mb * 1024ull * 1024ull) m.ex->trim_pool();
}
} // namespace

extern "C" {

int syi_onnx_abi_version(void) {
    return SYI_ONNX_ABI_VERSION;
}

const char *syi_onnx_last_error(void) {
    return si::g_error.c_str();
}

int syi_onnx_create(const char *path, const char *id, const char *device,
                    const char *drop_outputs, int dnnl, syi_onnx_model **out_model) {
    if (!out_model) {
        si::set_error("out_model is null");
        return -1;
    }
    *out_model = nullptr;
    if (!path || !*path) {
        si::set_error("path is required");
        return -1;
    }
    try {
        auto m = std::make_unique<syi_onnx_model>();
        m->id = (id && *id) ? id : path;
        m->path = path;
        m->device = (device && *device) ? device : "GPU";
        std::string devlow = si::lower(m->device);
        if (devlow.rfind("map:", 0) == 0) {
            std::string mapfile = m->device.substr(4);
            std::ifstream mf(mapfile);
            if (!mf) {
                mapfile = si::lower(mapfile);
                mf.open(mapfile);
            }
            std::string line;
            const std::string base = m->path.substr(m->path.find_last_of('/') + 1);
            while (std::getline(mf, line)) {
                std::istringstream iss(line);
                std::string key, val;
                if (!(iss >> key >> val)) continue;
                if (si::lower(base).find(si::lower(key)) != std::string::npos) {
                    m->device = val;
                    devlow = si::lower(m->device);
                    break;
                }
            }
        }
        if (devlow.rfind("npu", 0) == 0) {
            auto colon = devlow.find(':');
            if (colon != std::string::npos) {
                std::string mapfile = m->device.substr(colon + 1);
                std::ifstream mf(mapfile);
                if (!mf) {
                    mapfile = si::lower(mapfile);
                    mf.open(mapfile);
                }
                std::string line;
                const std::string base = m->path.substr(m->path.find_last_of('/') + 1);
                while (std::getline(mf, line)) {
                    std::istringstream iss(line);
                    std::string key, val;
                    if (!(iss >> key >> val)) continue;
                    if (key == base || base.find(key) != std::string::npos) {
                        setenv("PF_NPU_IR_FILE", val.c_str(), 1);
                        break;
                    }
                }
            }
            const auto mp = si::onnxpb::parse_model(m->path);
            std::string nerr;
            m->nx = std::make_unique<si::npu::executor>(mp, nerr);
            unsetenv("PF_NPU_IR_FILE");
            if (!m->nx->valid()) throw std::runtime_error("NPU: " + nerr);
        } else {
            m->q = &si::queue_for(m->device);
            m->graph = si::onnx::build_graph(m->path, true);
            m->ex = std::make_unique<si::onnx::executor>(m->graph, *m->q);
#if SYI_ONNX_NEW_WIRE
            const auto drops = si::split_csv(drop_outputs);
            if (!drops.empty()) m->ex->skip_outputs(drops);
            m->ex->set_dnnl(dnnl != 0);
#else
            (void)drop_outputs;
            (void)dnnl;
#endif
        }
        *out_model = m.release();
        return 0;
    } catch (const std::bad_alloc &) {
        si::set_error("out of memory while loading " + std::string(path));
        return -1;
    } catch (const std::exception & e) {
        si::set_error(e.what());
        return -1;
    }
}

void syi_onnx_destroy(syi_onnx_model *model) {
    delete model;
}

int syi_onnx_info_json(const syi_onnx_model *model, char **out_json, size_t *out_len) {
    if (!model || !out_json || !out_len) {
        si::set_error("invalid info arguments");
        return -1;
    }
    try {
        std::lock_guard<std::mutex> lk(const_cast<syi_onnx_model *>(model)->m);
        json j;
        j["id"] = model->id;
        j["path"] = model->path;
        j["device"] = model->device;
        j["loaded"] = model->ex != nullptr || model->nx != nullptr;
        j["inputs"] = json::array();
        j["outputs"] = json::array();
        if (model->nx) {
            for (const auto & n : model->nx->ir().inputs) j["inputs"].push_back(n);
            for (const auto & n : model->nx->ir().outputs) j["outputs"].push_back(n);
            size_t bytes = 0;
            for (const auto & a : model->nx->args()) bytes += a.bytes;
            j["vram_bytes"] = (uint64_t)bytes;
        } else {
            for (int s : model->graph.input_slots) j["inputs"].push_back(model->graph.tensors[s].name);
            for (int s : model->graph.output_slots) j["outputs"].push_back(model->graph.tensors[s].name);
#if SYI_ONNX_NEW_WIRE
            if (model->ex) {
                j["vram_bytes"] = (uint64_t)(model->ex->live_bytes() + model->ex->pooled_bytes());
            }
#endif
        }
        auto it = model->graph.metadata.find("character");
        j["character"] = it == model->graph.metadata.end() ? std::string() : it->second;
        const std::string s = j.dump();
        auto *p = (char *)std::malloc(s.size() + 1);
        if (!p) {
            si::set_error("out of memory for info JSON");
            return -1;
        }
        std::memcpy(p, s.data(), s.size());
        p[s.size()] = 0;
        *out_json = p;
        *out_len = s.size();
        return 0;
    } catch (const std::exception & e) {
        si::set_error(e.what());
        return -1;
    }
}

int syi_onnx_run_wire(const syi_onnx_model *model, const uint8_t *request, size_t request_len,
                      uint8_t **out_response, size_t *out_len) {
    if (!out_response || !out_len) {
        si::set_error("out_response/out_len is null");
        return -1;
    }
    *out_response = nullptr;
    *out_len = 0;
    auto reply = [&](int status, const std::string & msg) {
        const std::vector<uint8_t> enc = si::onnxwire::encode_response(status, msg, {});
        auto *p = (uint8_t *)std::malloc(enc.size());
        if (!p) {
            si::set_error("out of memory for the response");
            return -1;
        }
        std::memcpy(p, enc.data(), enc.size());
        *out_response = p;
        *out_len = enc.size();
        return 0;
    };
    if (!model || (!model->ex && !model->nx)) {
        si::set_error("model is null or was not loaded");
        return reply(2, "model is null or was not loaded");
    }
    std::string want;
    std::vector<si::onnxwire::tensor> inputs;
    std::string err;
    if (!si::onnxwire::decode_request(request, request_len, want, inputs, err)) {
        return reply(1, "bad request: " + err);
    }
    if (!want.empty() && want != model->id) {
        return reply(2, "unknown model '" + want + "'");
    }
    try {
        std::lock_guard<std::mutex> lk(const_cast<syi_onnx_model *>(model)->m);
        std::vector<si::onnxwire::tensor> feeds = inputs;
        std::vector<std::string> out_names;
        if (model->nx) {
            out_names = model->nx->ir().outputs;
        } else {
            for (int s : model->graph.output_slots) out_names.push_back(model->graph.tensors[s].name);
        }
        if (model->ex && model->ex->has_output_reductions()) {
            si::set_error("this model has output reductions configured, and the wire API "
                          "returns whole tensors; use the direct API");
            return reply(2, "this model has output reductions configured; use the direct API");
        }
        std::vector<std::vector<uint8_t>> bufs(out_names.size());
        std::vector<bool> keep(out_names.size(), true);
        std::vector<std::pair<std::string, std::vector<uint8_t> &>> outs;
        for (size_t i = 0; i < out_names.size(); ++i) {
#if SYI_ONNX_NEW_WIRE
            keep[i] = model->nx || !model->ex->output_skipped(i);
#else
            keep[i] = true;
#endif
            if (keep[i]) outs.emplace_back(out_names[i], bufs[i]);
        }
        std::vector<si::onnxwire::tensor> results;
        if (model->nx) {
            model->nx->run(feeds, outs);
            for (size_t i = 0; i < out_names.size(); ++i) {
                if (!keep[i]) continue;
                si::npu::arg_desc ad;
                if (!model->nx->find_output(out_names[i], ad)) continue;
                si::onnxwire::tensor t;
                t.name = out_names[i];
                t.type = ad.precision == ZE_GRAPH_ARGUMENT_PRECISION_INT64
                             ? si::onnxwire::dtype::i64
                             : (ad.precision == ZE_GRAPH_ARGUMENT_PRECISION_INT32
                                    ? si::onnxwire::dtype::i32
                                    : si::onnxwire::dtype::f32);
                t.shape.assign(ad.dims.begin(), ad.dims.end());
                t.data = std::move(bufs[i]);
                results.push_back(std::move(t));
            }
        } else {
            model->ex->run(feeds, outs);
            for (size_t i = 0; i < out_names.size(); ++i) {
                if (!keep[i]) continue;
                const si::onnx::tensor & gt = model->ex->output(i);
                si::onnxwire::tensor t;
                t.name = out_names[i];
                t.type = si::wire_type(gt.dtype);
                t.shape.assign(gt.shape.begin(), gt.shape.end());
                t.data = std::move(bufs[i]);
                results.push_back(std::move(t));
            }
        }
        const std::vector<uint8_t> enc = si::onnxwire::encode_response(0, "", results);
        auto *p = (uint8_t *)std::malloc(enc.size());
        if (!p) {
            si::set_error("out of memory for the response");
            return -1;
        }
        std::memcpy(p, enc.data(), enc.size());
        *out_response = p;
        *out_len = enc.size();
        maybe_trim(*const_cast<syi_onnx_model *>(model));
        return 0;
    } catch (const std::exception & e) {
        si::set_error(e.what());
        return reply(3, e.what());
    }
}

int syi_onnx_run_direct(const syi_onnx_model *model, const syi_onnx_input *inputs,
                        size_t n_inputs, syi_onnx_output *outputs, size_t max_outputs,
                        size_t *out_count) {
    if (!model || (!model->ex && !model->nx) || !out_count || (!inputs && n_inputs) ||
        (!outputs && max_outputs)) {
        si::set_error("invalid direct-run arguments");
        return -1;
    }
    *out_count = 0;
    auto & m = *const_cast<syi_onnx_model *>(model);
    try {
        std::lock_guard<std::mutex> lk(m.m);
        std::vector<si::onnxwire::tensor> feeds;
        feeds.reserve(n_inputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            const syi_onnx_input & in = inputs[i];
            if (!in.name || !in.shape || (!in.data && in.nbytes) || in.ndim < 0) {
                si::set_error("invalid input tensor");
                return -1;
            }
            si::onnxwire::tensor t;
            t.name = in.name;
            t.type = (si::onnxwire::dtype)in.dtype;
            const size_t es = si::onnxwire::elem_size(t.type);
            if (!es) {
                si::set_error("invalid input dtype");
                return -1;
            }
            size_t want = es;
            for (int d = 0; d < in.ndim; ++d) {
                if (in.shape[d] < 0) {
                    si::set_error("negative input dimension");
                    return -1;
                }
                want *= (size_t)in.shape[d];
            }
            if (want != in.nbytes) {
                si::set_error("input nbytes does not match its dtype/shape");
                return -1;
            }
            t.shape.assign(in.shape, in.shape + in.ndim);
            t.data.resize(in.nbytes);
            if (in.nbytes) std::memcpy(t.data.data(), in.data, in.nbytes);
            feeds.push_back(std::move(t));
        }

        std::vector<std::string> names;
        if (m.nx) {
            names = m.nx->ir().outputs;
        } else {
            for (int slot : m.graph.output_slots) names.push_back(m.graph.tensors[slot].name);
        }
        std::vector<std::vector<uint8_t>> bufs(names.size());
        std::vector<bool> keep(names.size(), true);
        std::vector<std::pair<std::string, std::vector<uint8_t> &>> outs;
        for (size_t i = 0; i < names.size(); ++i) {
#if SYI_ONNX_NEW_WIRE
            keep[i] = m.nx || !m.ex->output_skipped(i);
#else
            keep[i] = true;
#endif
            if (keep[i]) outs.emplace_back(names[i], bufs[i]);
        }
        if (m.nx) m.nx->run(feeds, outs);
        else m.ex->run(feeds, outs);

        m.last_names.clear();
        m.last_shapes.clear();
        m.last_data.clear();
        m.last_dtype.clear();
        for (size_t i = 0; i < names.size(); ++i) {
            if (!keep[i]) continue;
            if (m.nx) {
                si::npu::arg_desc ad;
                if (!m.nx->find_output(names[i], ad)) continue;
                const size_t actual_elems = ad.bytes ? bufs[i].size() / ad.bytes : 1;
                if (actual_elems > 1 && !ad.dims.empty()) ad.dims[0] *= static_cast<int64_t>(actual_elems);
                auto rit = m.nx_reduce.find(names[i]);
                int rkind = rit == m.nx_reduce.end() ? 0 : rit->second;
                std::vector<uint8_t> data = std::move(bufs[i]);
                if (rkind == 1 && ad.precision == ZE_GRAPH_ARGUMENT_PRECISION_FP32 && !ad.dims.empty()) {
                    const size_t V = static_cast<size_t>(ad.dims.back());
                    if (V > 0 && data.size() >= V * 4) {
                        const size_t outer = data.size() / (V * 4);
                        std::vector<uint8_t> idx(outer * 8), mx(outer * 4);
                        const float * src = reinterpret_cast<const float *>(data.data());
                        int64_t * di = reinterpret_cast<int64_t *>(idx.data());
                        float * dm = reinterpret_cast<float *>(mx.data());
                        for (size_t q = 0; q < outer; ++q) {
                            int64_t best = 0;
                            float bv = src[q * V];
                            for (size_t k = 1; k < V; ++k) if (src[q * V + k] > bv) { bv = src[q * V + k]; best = k; }
                            di[q] = best;
                            dm[q] = bv;
                        }
                        std::vector<int64_t> shp(ad.dims.begin(), ad.dims.end() - 1);
                        m.last_names.push_back(names[i] + ":argmax");
                        m.last_shapes.push_back(shp);
                        m.last_dtype.push_back((int32_t)si::onnxwire::dtype::i64);
                        m.last_data.push_back(std::move(idx));
                        m.last_names.push_back(names[i] + ":max");
                        m.last_shapes.push_back(shp);
                        m.last_dtype.push_back((int32_t)si::onnxwire::dtype::f32);
                        m.last_data.push_back(std::move(mx));
                        continue;
                    }
                } else if (rkind == 2) {
                    const size_t n = data.size() / 4;
                    std::vector<uint8_t> b(n);
                    const float * src = reinterpret_cast<const float *>(data.data());
                    for (size_t q = 0; q < n; ++q) b[q] = static_cast<uint8_t>(std::max(0.f, std::min(255.f, src[q])));
                    m.last_names.push_back(names[i]);
                    m.last_shapes.emplace_back(ad.dims.begin(), ad.dims.end());
                    m.last_dtype.push_back((int32_t)si::onnxwire::dtype::u8);
                    m.last_data.push_back(std::move(b));
                    continue;
                }
                m.last_names.push_back(names[i]);
                m.last_shapes.emplace_back(ad.dims.begin(), ad.dims.end());
                m.last_dtype.push_back(ad.precision == ZE_GRAPH_ARGUMENT_PRECISION_INT64
                                           ? (int32_t)si::onnxwire::dtype::i64
                                           : (ad.precision == ZE_GRAPH_ARGUMENT_PRECISION_INT32
                                                  ? (int32_t)si::onnxwire::dtype::i32
                                                  : (int32_t)si::onnxwire::dtype::f32));
                m.last_data.push_back(std::move(data));
                continue;
            }
            const size_t nred = m.ex->reduced_count(i);
            if (nred) {
                for (size_t k = 0; k < nred; ++k) {
                    const auto & r = m.ex->reduced(i, k);
                    std::string nm = names[i];
                    if (nred > 1) nm += (k == 0 ? ":argmax" : ":max");
                    m.last_names.push_back(nm);
                    m.last_shapes.push_back(r.shape);
                    m.last_dtype.push_back((int32_t)si::wire_type(r.dtype));
                    // A copy, because the reduced buffers are the executor's
                    // and the returned view must not alias the graph's host
                    // tensors.  They are small: 16 bytes per row for an
                    // argmax, one byte per mask element.
                    //
                    // `wire_type` and the reduced dtype are the same
                    // numbering (both are the ONNX tensor types), so the
                    // Python side reads it with the code it already uses.
                    m.last_data.push_back(r.host);
                }
                continue;
            }
            const si::onnx::tensor & gt = m.ex->output(i);
            m.last_names.push_back(names[i]);
            m.last_shapes.emplace_back(gt.shape.begin(), gt.shape.end());
            m.last_dtype.push_back((int32_t)si::wire_type(gt.dtype));
            // The host bytes are copied into the executor's graph tensor by
            // the readback; hand that buffer over instead of copying it a
            // second time (a recogniser output is 115 MB and the layout
            // detector's masks are 768 MB a call).
            m.last_data.push_back(std::move(bufs[i]));
        }
        const size_t n = m.last_data.size();
        for (size_t i = 0; i < n && i < max_outputs; ++i) {
            outputs[i].name = m.last_names[i].c_str();
            outputs[i].dtype = m.last_dtype[i];
            outputs[i].ndim = (int32_t)m.last_shapes[i].size();
            outputs[i].shape = m.last_shapes[i].data();
            outputs[i].data = m.last_data[i].data();
            outputs[i].nbytes = m.last_data[i].size();
        }
        *out_count = n;
        if (n > max_outputs) {
            si::set_error("output buffer too small");
            return -1;
        }
        maybe_trim(*const_cast<syi_onnx_model *>(model));
        return 0;
    } catch (const std::bad_alloc &) {
        si::set_error("out of memory in the direct ONNX run");
        return -1;
    } catch (const std::exception & e) {
        si::set_error(e.what());
        return -1;
    }
}

int syi_onnx_set_output_mode(syi_onnx_model *model, const char *output, const char *mode) {
    if (!model || (!model->ex && !model->nx) || !output || !*output) {
        si::set_error("invalid output-mode arguments");
        return -1;
    }
    try {
        const std::string m = si::lower(mode ? mode : "none");
        int kind = si::onnx::executor::kReduceNone;
        if (m == "argmax" || m == "argmax_last") kind = si::onnx::executor::kReduceArgmax;
        else if (m == "u8" || m == "uint8") kind = si::onnx::executor::kReduceU8;
        else if (!m.empty() && m != "none") {
            si::set_error("unknown output mode '" + m + "' (argmax, u8 or none)");
            return -1;
        }
        std::lock_guard<std::mutex> lk(model->m);
        if (model->nx) {
            std::string name = output;
            if (name.empty() || (name[0] >= '0' && name[0] <= '9')) {
                size_t idx = static_cast<size_t>(std::strtoull(output, nullptr, 10));
                if (idx < model->nx->ir().outputs.size()) name = model->nx->ir().outputs[idx];
            }
            model->nx_reduce[name] = kind;
        } else {
            model->ex->set_output_reduce(output, kind);
        }
        return 0;
    } catch (const std::exception & e) {
        si::set_error(e.what());
        return -1;
    }
}

int syi_onnx_trim(syi_onnx_model *model) {
    if (!model || !model->ex || model->nx) return -1;
    try {
        std::lock_guard<std::mutex> lk(model->m);
        model->ex->trim_pool();
        return 0;
    } catch (const std::exception & e) {
        si::set_error(e.what());
        return -1;
    }
}

void syi_onnx_free(void *p) {
    std::free(p);
}

} // extern "C"
