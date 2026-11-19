#include "embedding_engine.h"
#include <cctype>
#include <cstdlib>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>

#include "backend/npu/npu_executor.h"
#include "backend/onnx/exec.h"
#include "device/device_select.h"
#include "model/onnx/graph.h"
#include "onnx_wire.h"

namespace si {
namespace embedding {

struct engine::impl {
    engine_config cfg;
    bert_model model;
    onnxpb::model_proto mp;
    bool npu = false;
    std::unique_ptr<sycl::queue> q;
    std::unique_ptr<onnx::executor> ex;
    std::unique_ptr<npu::executor> nx;
    int dim_ = 0;
    std::string device_name;
};

static bool is_npu_name(const std::string & d) {
    std::string l = d;
    for (char & c : l) c = (char)std::tolower((unsigned char)c);
    return l == "npu";
}

static sycl::device pick_device(const std::string & name) {
    std::string l = name;
    for (char & c : l) c = (char)std::tolower((unsigned char)c);
    if (l == "cpu" || l == "host") return sycl::device{sycl::cpu_selector_v};
    if (l == "auto") {
        const auto gpus = dev::gpu_devices();
        if (!gpus.empty()) return gpus.front();
        return sycl::device{sycl::cpu_selector_v};
    }
    if (l.rfind("gpu.", 0) == 0) {
        const int idx = std::atoi(l.c_str() + 4);
        const auto gpus = dev::gpu_devices();
        if (idx < 0 || idx >= (int)gpus.size()) throw std::runtime_error("no GPU with index " + std::to_string(idx));
        return gpus[(size_t)idx];
    }
    if (l == "gpu") {
        const auto gpus = dev::gpu_devices();
        if (gpus.empty()) throw std::runtime_error("no GPU available");
        return gpus.front();
    }
    throw std::runtime_error("unknown embedding device '" + name + "'");
}

engine::engine(const engine_config & cfg) : p_(std::make_unique<impl>()) {
    p_->cfg = cfg;
    std::string err;
    p_->model.load(cfg.model_path, err);
    p_->dim_ = p_->model.spec().n_embd;

    int max_seq = cfg.max_seq > 0 ? cfg.max_seq : 128;
    if (max_seq > p_->model.spec().n_ctx) max_seq = p_->model.spec().n_ctx;
    p_->cfg.max_seq = max_seq;

    p_->npu = is_npu_name(cfg.device);
    if (p_->npu) {
        p_->device_name = "npu";
        p_->model.build_onnx(max_seq, 1, p_->mp);
        p_->nx = std::make_unique<npu::executor>(p_->mp, err);
        if (!p_->nx->valid()) throw std::runtime_error("NPU: " + (err.empty() ? p_->nx->error() : err));
        // The constructor has serialised the IR (including the weights blob);
        // keeping the proto would hold a second full f32 copy of 567 M weights.
        p_->mp = onnxpb::model_proto{};
        p_->device_name = "npu";
    } else {
        const sycl::device d = pick_device(cfg.device);
        p_->device_name = d.is_gpu() ? "gpu" : "cpu";
        if (cfg.device.rfind("gpu.", 0) == 0) p_->device_name = cfg.device;
        p_->q = std::make_unique<sycl::queue>(d, sycl::property::queue::in_order());
        p_->model.build_onnx(max_seq, -1, p_->mp);
        onnx::exec_graph g = onnx::build_graph_from_graph(p_->mp.graph, p_->mp.opset, cfg.model_path);
        p_->ex = std::make_unique<onnx::executor>(std::move(g), *p_->q);
        // build_graph copies every initializer into the executable graph; the
        // proto's f32 weights are dead after this point.  Holding both cost
        // ~2.3 GB extra on a 567 M-parameter encoder.
        p_->mp = onnxpb::model_proto{};
    }
}

engine::~engine() = default;
int engine::dim() const { return p_->dim_; }
const std::string & engine::device() const { return p_->device_name; }
bool engine::is_npu() const { return p_->npu; }
const bert_spec & engine::spec() const { return p_->model.spec(); }
const sp_tokenizer & engine::tokenizer() const { return p_->model.tokenizer(); }

std::vector<std::vector<float>> engine::embed(const std::vector<std::string> & texts) {
    std::vector<std::vector<float>> out;
    if (texts.empty()) return out;
    const int S = p_->cfg.max_seq;
    const int H = p_->dim_;
    const auto & tok = p_->model.tokenizer();
    if (!tok.valid()) throw std::runtime_error("bert: tokenizer is not available");

    // On NPU the graph has a fixed batch of 1; feed one text at a time.
    const size_t chunk = p_->npu ? 1 : (size_t)std::max(1, p_->cfg.batch);
    for (size_t base = 0; base < texts.size(); base += chunk) {
        const size_t n = std::min(chunk, texts.size() - base);
        const int B = (int)n;
        std::vector<int64_t> ids((size_t)B * S, (int64_t)tok.pad);
        std::vector<int64_t> pos((size_t)B * S);
        std::vector<float> mask((size_t)B * S, 0.f);
        for (size_t r = 0; r < n; ++r) {
            std::vector<int> t = tok.encode(texts[base + r]);
            if (std::getenv("PF_EMBED_DEBUG")) {
                fprintf(stderr, "[embed] text=%s ids=", texts[base + r].c_str());
                for (int id : t) fprintf(stderr, "%d ", id);
                fprintf(stderr, "\n");
            }
            if ((int)t.size() > S) t.resize(S);
            for (size_t j = 0; j < t.size(); ++j) ids[r * S + j] = t[j];
            for (int j = 0; j < S; ++j) {
                pos[r * S + j] = j;
                mask[r * S + j] = (j < (int)t.size()) ? 0.f : -1e9f;
            }
        }

        onnxwire::tensor ti, tp, tm;
        ti.name = "input_ids"; ti.type = onnxwire::dtype::i64;
        ti.shape = {(uint64_t)B, (uint64_t)S};
        ti.data.resize(ids.size() * sizeof(int64_t));
        std::memcpy(ti.data.data(), ids.data(), ti.data.size());
        tp.name = "position_ids"; tp.type = onnxwire::dtype::i64;
        tp.shape = {(uint64_t)B, (uint64_t)S};
        tp.data.resize(pos.size() * sizeof(int64_t));
        std::memcpy(tp.data.data(), pos.data(), tp.data.size());
        tm.name = "attn_mask"; tm.type = onnxwire::dtype::f32;
        tm.shape = {(uint64_t)B, 1, 1, (uint64_t)S};
        tm.data.resize(mask.size() * sizeof(float));
        std::memcpy(tm.data.data(), mask.data(), tm.data.size());

        std::vector<uint8_t> result((size_t)B * H * sizeof(float));
        if (p_->nx) {
            p_->nx->run({ti, tp, tm}, {{"embedding", result}});
        } else {
            p_->ex->run({ti, tp, tm}, {{"embedding", result}});
        }
        const float * f = (const float *)result.data();
        for (int r = 0; r < B; ++r)
            out.emplace_back(f + (size_t)r * H, f + (size_t)(r + 1) * H);
    }
    return out;
}

} // namespace embedding
} // namespace si
