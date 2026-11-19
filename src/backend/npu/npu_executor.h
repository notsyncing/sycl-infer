#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "backend/npu/npu_ir.h"
#include "backend/npu/npu_runtime.h"
#include "common/onnx_wire.h"
#include "model/onnx/proto.h"

namespace si {
namespace npu {

class executor {
public:
    explicit executor(const onnxpb::model_proto & m, std::string & err);
    ~executor();
    executor(const executor &) = delete;
    executor & operator=(const executor &) = delete;

    bool valid() const { return valid_; }
    const std::string & error() const { return error_; }
    const ir_result & ir() const { return ir_; }
    size_t arg_count() const { return args_.size(); }
    const std::vector<arg_desc> & args() const { return args_; }

    void run(const std::vector<onnxwire::tensor> & inputs,
             const std::vector<std::pair<std::string, std::vector<uint8_t> &>> & outputs);

    bool find_output(const std::string & name, arg_desc & out) const;

private:
    struct impl;
    std::unique_ptr<impl> p_;
    ir_result ir_;
    std::vector<arg_desc> args_;
    std::map<std::string, size_t> by_name_;
    std::vector<void *> buffers_;
    bool valid_ = false;
    std::string error_;
};

}  // namespace npu
}  // namespace si
