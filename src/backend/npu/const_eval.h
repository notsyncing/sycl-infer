#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model/onnx/proto.h"

namespace si {
namespace npu {

struct const_tensor {
    int32_t dtype = onnxpb::FLOAT;
    std::vector<int64_t> dims;
    std::vector<uint8_t> raw;

    size_t count() const;
    int64_t elem_i(size_t i) const;
    float elem_f(size_t i) const;
};

class const_evaluator {
public:
    explicit const_evaluator(const onnxpb::graph_proto & g);

    bool eval(const std::string & name, const_tensor & out);

private:
    const onnxpb::graph_proto & g_;
    std::vector<const_tensor> cache_;
    std::vector<std::string> names_;
    std::vector<int> states_;

    int find(const std::string & name) const;
    bool eval_node(const onnxpb::node_proto & n, const_tensor & out, int depth);
};

}  // namespace npu
}  // namespace si
