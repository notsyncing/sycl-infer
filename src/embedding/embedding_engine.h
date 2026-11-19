#pragma once
// A tiny, backend-agnostic embedding engine.  One GGUF encoder model is
// lowered to an in-memory ONNX graph; the graph runs on the in-process ONNX
// executor (CPU or GPU queue) or on the NPU executor.  Text tokenisation and
// pooling live here so all three backends produce the same vector.

#include <memory>
#include <string>
#include <vector>

#include "bert_model.h"

namespace si {
namespace embedding {

struct engine_config {
    std::string model_path;
    std::string device = "auto"; // auto|cpu|host|gpu|gpu.N|npu
    int max_seq = 512;
    int batch = 16; // max texts per forward on CPU/GPU
};

class engine {
public:
    explicit engine(const engine_config & cfg);
    ~engine();
    engine(const engine &) = delete;
    engine & operator=(const engine &) = delete;

    std::vector<std::vector<float>> embed(const std::vector<std::string> & texts);
    int dim() const;
    const std::string & device() const;
    bool is_npu() const;
    const bert_spec & spec() const;
    const sp_tokenizer & tokenizer() const;

private:
    struct impl;
    std::unique_ptr<impl> p_;
};

} // namespace embedding
} // namespace si
