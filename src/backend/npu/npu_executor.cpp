#include "npu_executor.h"

#include "backend/npu/npu_runtime.h"
#include "common/quant.h"

#include <ze_graph_ext.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace si {
namespace npu {

struct executor::impl {
    graph_ptr g{nullptr, [](graph *) {}};
};

executor::executor(const onnxpb::model_proto & m, std::string & err) : p_(new impl()) {
    if (const char * irfile = std::getenv("PF_NPU_IR_FILE")) {
        std::ifstream xf(irfile, std::ios::binary);
        if (!xf) {
            error_ = std::string("cannot open NPU IR file: ") + irfile;
            return;
        }
        std::stringstream ss;
        ss << xf.rdbuf();
        ir_.xml = ss.str();
        const std::string wfile = std::string(irfile) + ".bin";
        std::ifstream wf(wfile, std::ios::binary);
        if (!wf) {
            error_ = "cannot open NPU IR weights: " + wfile;
            return;
        }
        ir_.weights.assign(std::istreambuf_iterator<char>(wf), std::istreambuf_iterator<char>());
        for (const auto & name : {"x", "pixel_values", "input"}) {
            if (ir_.xml.find("names=\"" + std::string(name) + "\"") != std::string::npos ||
                ir_.xml.find("name=\"" + std::string(name) + "\"") != std::string::npos) {
                ir_.inputs.push_back(name);
            }
        }
        size_t rp = 0;
        while ((rp = ir_.xml.find("type=\"Result\"", rp)) != std::string::npos) {
            size_t np = ir_.xml.find("names=\"", rp);
            if (np == std::string::npos || np > rp + 2000) break;
            np += 7;
            size_t ne = ir_.xml.find('"', np);
            if (ne != std::string::npos) ir_.outputs.push_back(ir_.xml.substr(np, ne - np));
            rp = ne;
        }
    } else if (!serialize_ir(m, ir_, err)) {
        error_ = err;
        return;
    }
    if (std::getenv("PF_NPU_DBG")) std::fprintf(stderr, "IR xml=%zu weights=%zu env=%s\n", ir_.xml.size(), ir_.weights.size(), std::getenv("PF_NPU_IR_FILE") ? "set":"unset");
    p_->g = compile_ir(ir_.xml, ir_.weights.data(), ir_.weights.size(), err);
    if (std::getenv("PF_NPU_DBG")) std::fprintf(stderr, "compile graph=%p err=%s\n", (void*)p_->g.get(), err.c_str());
    if (!p_->g) {
        error_ = err;
        return;
    }
    if (!query_args(p_->g.get(), args_, err)) {
        error_ = err;
        return;
    }
    for (size_t i = 0; i < args_.size(); ++i) {
        by_name_[args_[i].name] = i;
        void * p = alloc_host(std::max<size_t>(args_[i].bytes, 64));
        if (!p) {
            error_ = "NPU host buffer allocation failed for argument " + args_[i].name;
            return;
        }
        buffers_.push_back(p);
    }
    valid_ = true;
    if (std::getenv("PF_NPU_DBG")) std::fprintf(stderr, "NPU executor valid=%d err=%s args=%zu\n", (int)valid_, error_.c_str(), args_.size());
}

executor::~executor() {
    for (void * p : buffers_) free_host(p);
}

bool executor::find_output(const std::string & name, arg_desc & out) const {
    auto it = by_name_.find(name);
    if (it == by_name_.end() || args_[it->second].kind != 1) return false;
    out = args_[it->second];
    return true;
}

static void copy_as(const void * src, onnxwire::dtype sdt, void * dst, uint32_t prec, size_t elems) {
    if (prec == ZE_GRAPH_ARGUMENT_PRECISION_FP16) {
        uint16_t * d = static_cast<uint16_t *>(dst);
        if (sdt == onnxwire::dtype::f32) {
            const float * s = static_cast<const float *>(src);
            for (size_t i = 0; i < elems; ++i) d[i] = ggml_float_to_half(s[i]);
        } else if (sdt == onnxwire::dtype::f16) {
            std::memcpy(d, src, elems * 2);
        } else {
            for (size_t i = 0; i < elems; ++i) d[i] = 0;
        }
    } else if (prec == ZE_GRAPH_ARGUMENT_PRECISION_FP32 || prec == ZE_GRAPH_ARGUMENT_PRECISION_INT32 ||
               prec == ZE_GRAPH_ARGUMENT_PRECISION_UINT32) {
        std::memset(dst, 0, elems * 4);
        if (sdt == onnxwire::dtype::f32 || sdt == onnxwire::dtype::i32) {
            std::memcpy(dst, src, elems * 4);
        }
    } else if (prec == ZE_GRAPH_ARGUMENT_PRECISION_INT64 || prec == ZE_GRAPH_ARGUMENT_PRECISION_UINT64) {
        std::memset(dst, 0, elems * 8);
        if (sdt == onnxwire::dtype::i64) std::memcpy(dst, src, elems * 8);
    } else {
        std::memcpy(dst, src, elems);
    }
}

void executor::run(const std::vector<onnxwire::tensor> & inputs,
                   const std::vector<std::pair<std::string, std::vector<uint8_t> &>> & outputs) {
    if (!valid_) throw std::runtime_error("NPU executor is not valid: " + error_);
    struct input_ref {
        size_t idx = 0;
        const uint8_t * data = nullptr;
        size_t row_bytes = 0;
        size_t esz = 4;
        onnxwire::dtype type = onnxwire::dtype::f32;
    };
    std::vector<input_ref> inrefs;
    size_t batch = 1;
    for (const auto & t : inputs) {
        auto it = by_name_.find(t.name);
        if (it == by_name_.end()) {
            error_ = "NPU graph has no input named " + t.name;
            throw std::runtime_error(error_);
        }
        const arg_desc & a = args_[it->second];
        size_t esz = 4;
        if (a.precision == ZE_GRAPH_ARGUMENT_PRECISION_FP16) esz = 2;
        else if (a.precision == ZE_GRAPH_ARGUMENT_PRECISION_INT64 || a.precision == ZE_GRAPH_ARGUMENT_PRECISION_UINT64) esz = 8;
        else if (a.precision == ZE_GRAPH_ARGUMENT_PRECISION_INT8 || a.precision == ZE_GRAPH_ARGUMENT_PRECISION_UINT8) esz = 1;
        if (a.bytes == 0 || t.data.empty() || t.data.size() % a.bytes != 0 || t.data.size() / a.bytes > 64) {
            error_ = "NPU input " + t.name + " has " + std::to_string(t.data.size()) + " bytes, expected a multiple of " +
                     std::to_string(a.bytes);
            throw std::runtime_error(error_);
        }
        size_t ratio = t.data.size() / a.bytes;
        batch = std::max(batch, ratio);
        inrefs.push_back({it->second, t.data.data(), a.bytes, esz, t.type});
    }
    std::vector<const void *> ptrs(args_.size(), nullptr);
    for (size_t i = 0; i < args_.size(); ++i) ptrs[i] = buffers_[i];
    for (const auto & o : outputs) {
        if (!by_name_.count(o.first)) {
            error_ = "NPU graph has no output named " + o.first;
            throw std::runtime_error(error_);
        }
    }
    std::string err;
    for (size_t row = 0; row < batch; ++row) {
        for (const auto & ir : inrefs) {
            copy_as(ir.data + row * ir.row_bytes, ir.type, buffers_[ir.idx], args_[ir.idx].precision,
                    ir.row_bytes / ir.esz);
        }
        if (!npu::run(p_->g.get(), ptrs, err)) {
            error_ = err;
            throw std::runtime_error("NPU graph execution failed: " + err);
        }
        for (const auto & o : outputs) {
            const arg_desc & a = args_[by_name_.at(o.first)];
            std::vector<uint8_t> one;
            if (a.precision == ZE_GRAPH_ARGUMENT_PRECISION_FP16) {
                const uint16_t * src = static_cast<const uint16_t *>(buffers_[by_name_.at(o.first)]);
                size_t elems = a.bytes / 2;
                one.resize(elems * 4);
                float * dst = reinterpret_cast<float *>(one.data());
                for (size_t i = 0; i < elems; ++i) dst[i] = ggml_half_to_float(src[i]);
            } else {
                const uint8_t * src = static_cast<const uint8_t *>(buffers_[by_name_.at(o.first)]);
                one.assign(src, src + a.bytes);
            }
            if (row == 0) o.second.assign(one.begin(), one.end());
            else o.second.insert(o.second.end(), one.begin(), one.end());
        }
    }
}

}  // namespace npu
}  // namespace si
