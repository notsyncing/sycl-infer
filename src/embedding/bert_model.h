#pragma once
// Embedding models (`general.architecture == bert`, e.g. BGE-M3) are loaded
// and served outside the autoregressive engine: they are bidirectional
// encoders with no KV cache and no LM head.  This file declares the small
// GGUF loader and the in-memory ONNX graph builder that feeds all three
// backends — the ONNX executor on CPU/GPU and npu::executor on NPU.

#include <memory>
#include <string>
#include <vector>

#include "gguf.h"
#include "proto.h"
#include "sp_tokenizer.h"

namespace si {
namespace embedding {

struct bert_spec {
    int n_layer = 0;
    int n_embd = 0;
    int n_ff = 0;
    int n_head = 0;
    int head_dim = 0;
    int n_ctx = 0;
    int pooling_type = 2; // GGUF: 0 none, 1 mean, 2 cls, 3 end
    float eps = 1e-5f;
    bool causal = false;
};

class bert_model {
public:
    void load(const std::string & path, std::string & err);

    const bert_spec & spec() const { return spec_; }
    const gguf_file & gguf() const { return gguf_; }
    sp_tokenizer & tokenizer() { return tok_; }
    const sp_tokenizer & tokenizer() const { return tok_; }
    const std::string & path() const { return path_; }

    // Build an ONNX ModelProto for the encoder.  All weights are dequantized
    // to f32; the graph uses only ops supported by both the in-process ONNX
    // executor and the NPU IR frontend.  `max_seq` fixes the sequence axis
    // (NPU graphs need static shapes); inputs shorter than it are padded.
    void build_onnx(int max_seq, int batch, onnxpb::model_proto & out) const;

private:
    std::string path_;
    gguf_file gguf_;
    bert_spec spec_;
    sp_tokenizer tok_;
};

// Cheap architecture peek used by the CLI to route a GGUF path to this
// subsystem instead of the autoregressive engine.
bool is_embedding_model(const std::string & path);

} // namespace embedding
} // namespace si
