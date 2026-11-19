#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model/onnx/proto.h"

namespace si {
namespace npu {

struct ir_result {
    std::string xml;
    std::vector<uint8_t> weights;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

bool serialize_ir(const onnxpb::model_proto & m, ir_result & out, std::string & err);

}  // namespace npu
}  // namespace si
