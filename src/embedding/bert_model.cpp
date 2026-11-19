#include "bert_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "quant.h"

namespace si {
namespace embedding {
namespace {

using onnxpb::attribute;
using onnxpb::node_proto;
using onnxpb::tensor_proto;

const gguf_tensor_info * req(const gguf_file & f, const std::string & name) {
    const gguf_tensor_info * t = f.find(name);
    if (!t) throw std::runtime_error("bert: missing tensor " + name);
    return t;
}

std::vector<float> dequant(const gguf_tensor_info * t) {
    std::vector<float> out((size_t)t->n_elements());
    dequantize_row(t->type, t->data, out.data(), (int64_t)t->n_elements());
    return out;
}

// GGUF matrices are [K, N] with K (dims[0]) varying fastest; ONNX wants
// row-major [K, N].  Return the transposed row-major f32 payload.
std::vector<float> mat_f32(const gguf_tensor_info * t) {
    if (t->dims.size() != 2) throw std::runtime_error("bert: expected a 2-D matrix: " + t->name);
    const size_t K = (size_t)t->dims[0], N = (size_t)t->dims[1];
    const std::vector<float> d = dequant(t);
    std::vector<float> out(K * N);
    for (size_t j = 0; j < N; ++j)
        for (size_t i = 0; i < K; ++i)
            out[i * N + j] = d[j * K + i];
    return out;
}

struct builder {
    onnxpb::model_proto & m;
    explicit builder(onnxpb::model_proto & mm) : m(mm) {
        m.ir_version = 8;
        m.opset = 13;
        m.producer_name = "sycl-infer/bert-onnx";
        m.graph.name = "bert_encoder";
    }
    int seq = 0;

    void init_f32(const std::string & name, std::vector<int64_t> dims, std::vector<float> data) {
        tensor_proto t;
        t.name = name;
        t.data_type = onnxpb::FLOAT;
        t.dims = std::move(dims);
        const size_t n = data.empty() ? 1 : data.size();
        t.raw.resize(n * sizeof(float));
        std::memcpy(t.raw.data(), data.data(), n * sizeof(float));
        m.graph.initializer.push_back(std::move(t));
    }
    void init_i64(const std::string & name, std::vector<int64_t> dims, std::vector<int64_t> data) {
        tensor_proto t;
        t.name = name;
        t.data_type = onnxpb::INT64;
        t.dims = std::move(dims);
        const size_t n = data.empty() ? 1 : data.size();
        t.raw.resize(n * sizeof(int64_t));
        std::memcpy(t.raw.data(), data.data(), n * sizeof(int64_t));
        m.graph.initializer.push_back(std::move(t));
    }
    void scalar(const std::string & name, float v) { init_f32(name, {}, {v}); }
    void node(const std::string & op, std::vector<std::string> in, std::vector<std::string> out) {
        node_proto n;
        n.op_type = op;
        n.input = std::move(in);
        n.output = std::move(out);
        m.graph.node.push_back(std::move(n));
    }
    void attr(const std::string & op, const std::string & name, attribute a) {
        m.graph.node.back().name = op;
        m.graph.node.back().attribute.push_back(std::move(a));
    }
    void attr_i(const std::string & n, int64_t v) {
        attribute a; a.name = n; a.type_ = attribute::type::i; a.i = v;
        m.graph.node.back().attribute.push_back(std::move(a));
    }
    void attr_ints(const std::string & n, std::vector<int64_t> v) {
        attribute a; a.name = n; a.type_ = attribute::type::ints; a.ints = std::move(v);
        m.graph.node.back().attribute.push_back(std::move(a));
    }
    void attr_f(const std::string & n, float v) {
        attribute a; a.name = n; a.type_ = attribute::type::f; a.f = v;
        m.graph.node.back().attribute.push_back(std::move(a));
    }
    // LayerNorm as a composition of opset-1 primitives.  The NPU IR frontend
    // maps ONNX LayerNormalization to the IR `LayerNorm` layer, which the
    // Lunar Lake driver rejects under opset1 ("Opset opset1 doesn't contain
    // the operation with type: LayerNorm").  These six ops are all in opset1
    // and produce the same numbers on every backend.
    void layer_norm(const std::string & x, const std::string & gamma, const std::string & beta,
                    float eps, const std::string & out) {
        node("ReduceMean", {x}, {out + "_mean"});
        attr_ints("axes", {-1}); attr_i("keepdims", 1);
        node("Sub", {x, out + "_mean"}, {out + "_d"});
        node("Mul", {out + "_d", out + "_d"}, {out + "_d2"});
        node("ReduceMean", {out + "_d2"}, {out + "_var"});
        attr_ints("axes", {-1}); attr_i("keepdims", 1);
        scalar(out + "_eps", eps);
        node("Add", {out + "_var", out + "_eps"}, {out + "_ve"});
        node("Sqrt", {out + "_ve"}, {out + "_sd"});
        node("Div", {out + "_d", out + "_sd"}, {out + "_n"});
        node("Mul", {out + "_n", gamma}, {out + "_g"});
        node("Add", {out + "_g", beta}, {out});
    }

    void input(const std::string & name, int32_t dtype, std::vector<int64_t> shape) {
        onnxpb::value_info v;
        v.name = name; v.elem_type = dtype; v.shape = std::move(shape); v.has_shape = true;
        m.graph.input.push_back(std::move(v));
    }
    void output(const std::string & name, std::vector<int64_t> shape) {
        onnxpb::value_info v;
        v.name = name; v.elem_type = onnxpb::FLOAT; v.shape = std::move(shape); v.has_shape = true;
        m.graph.output.push_back(std::move(v));
    }

    std::string reshape_to(const std::string & x, const std::string & name,
                           std::vector<int64_t> dims) {
        init_i64(name + "_shape", {(int64_t)dims.size()}, dims);
        node("Reshape", {x, name + "_shape"}, {name});
        return name;
    }
};

} // namespace

bool is_embedding_model(const std::string & path) {
    try {
        gguf_file f;
        f.load(path);
        const std::string * a = f.get_str("general.architecture");
        return a && *a == "bert";
    } catch (...) {
        return false;
    }
}

void bert_model::load(const std::string & path, std::string & err) {
    try {
        path_ = path;
        gguf_.load(path);
        const std::string * arch = gguf_.get_str("general.architecture");
        if (!arch || *arch != "bert") throw std::runtime_error("not a bert embedding model");
        spec_.n_layer = gguf_.get_u32("bert.block_count");
        spec_.n_embd = gguf_.get_u32("bert.embedding_length");
        spec_.n_ff = gguf_.get_u32("bert.feed_forward_length");
        spec_.n_head = gguf_.get_u32("bert.attention.head_count");
        spec_.n_ctx = gguf_.get_u32("bert.context_length", 512);
        spec_.eps = gguf_.get_f32("bert.attention.layer_norm_epsilon", 1e-5f);
        spec_.pooling_type = (int)gguf_.get_u32("bert.pooling_type", 2);
        spec_.causal = gguf_.meta("bert.attention.causal")
                       && gguf_.meta("bert.attention.causal")->as_u32() != 0;
        spec_.head_dim = spec_.n_embd / spec_.n_head;
        if (spec_.n_layer <= 0 || spec_.n_embd <= 0 || spec_.n_head <= 0 ||
            spec_.n_embd % spec_.n_head != 0) {
            throw std::runtime_error("bert: invalid geometry");
        }
        tok_.load(gguf_);
    } catch (const std::exception & e) {
        err = e.what();
        throw;
    }
}

void bert_model::build_onnx(int max_seq, int batch, onnxpb::model_proto & out) const {
    if (max_seq <= 0 || max_seq > spec_.n_ctx) {
        throw std::runtime_error("bert: bad max_seq");
    }
    builder b(out);
    b.seq = max_seq;
    // The sequence axis is fixed (NPU graphs need static shapes); batch is
    // dynamic for the ONNX executor and pinned to 1 for NPU.
    b.input("input_ids", onnxpb::INT64, {batch, max_seq});
    b.input("position_ids", onnxpb::INT64, {batch, max_seq});
    b.input("attn_mask", onnxpb::FLOAT, {batch, 1, 1, max_seq});

    // Embeddings: token + position, then the embedding LayerNorm.
    // The GGUF embedding tables are stored [dim, rows] with `dim` varying
    // fastest, so their flat f32 payload is already row-major [rows, dim].
    // Re-transposing them here scrambles every token's embedding (the model
    // still emits plausible-looking vectors, with no semantic structure).
    const gguf_tensor_info * te = req(gguf_, "token_embd.weight");
    b.init_f32("tok_table", {(int64_t)te->dims[1], (int64_t)spec_.n_embd}, dequant(te));
    const gguf_tensor_info * pe = req(gguf_, "position_embd.weight");
    std::vector<float> pos = dequant(pe);
    pos.resize((size_t)max_seq * spec_.n_embd); // truncate to max_seq positions
    b.init_f32("pos_table", {max_seq, (int64_t)spec_.n_embd}, std::move(pos));
    b.node("Gather", {"tok_table", "input_ids"}, {"emb_tok"});
    b.node("Gather", {"pos_table", "position_ids"}, {"emb_pos"});
    b.node("Add", {"emb_tok", "emb_pos"}, {"emb_sum"});
    const gguf_tensor_info * nlw = req(gguf_, "token_embd_norm.weight");
    const gguf_tensor_info * nlb = req(gguf_, "token_embd_norm.bias");
    b.init_f32("tok_norm_w", {(int64_t)spec_.n_embd}, dequant(nlw));
    b.init_f32("tok_norm_b", {(int64_t)spec_.n_embd}, dequant(nlb));
    b.layer_norm("emb_sum", "tok_norm_w", "tok_norm_b", spec_.eps, "h");

    const int H = spec_.n_embd, F = spec_.n_ff, NH = spec_.n_head, HD = spec_.head_dim;
    b.scalar("attn_scale", 1.f / std::sqrt((float)HD));
    b.scalar("sqrt2", 1.4142135623730951f);
    b.scalar("half", 0.5f);
    b.scalar("one", 1.f);
    b.scalar("norm_eps", 1e-12f);

    std::string h = "h";
    for (int il = 0; il < spec_.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        auto W = [&](const char * n) { return req(gguf_, p + n); };
        // QKV
        const char * qkv[3] = {"attn_q", "attn_k", "attn_v"};
        std::string qkv_out[3];
        for (int k = 0; k < 3; ++k) {
            const std::string w = "L" + std::to_string(il) + "_" + qkv[k] + "_w";
            const std::string bb = "L" + std::to_string(il) + "_" + qkv[k] + "_b";
            b.init_f32(w, {(int64_t)H, (int64_t)H}, mat_f32(W((std::string(qkv[k]) + ".weight").c_str())));
            b.init_f32(bb, {(int64_t)H}, dequant(W((std::string(qkv[k]) + ".bias").c_str())));
            const std::string mm = "L" + std::to_string(il) + "_" + qkv[k] + "_mm";
            b.node("MatMul", {h, w}, {mm});
            qkv_out[k] = "L" + std::to_string(il) + "_" + qkv[k];
            b.node("Add", {mm, bb}, {qkv_out[k]});
        }
        const std::string q4 = b.reshape_to(qkv_out[0], "L" + std::to_string(il) + "_q4", {batch, max_seq, NH, HD});
        const std::string k4 = b.reshape_to(qkv_out[1], "L" + std::to_string(il) + "_k4", {batch, max_seq, NH, HD});
        const std::string v4 = b.reshape_to(qkv_out[2], "L" + std::to_string(il) + "_v4", {batch, max_seq, NH, HD});
        b.node("Transpose", {q4}, {q4 + "_t"});
        b.attr_ints("perm", {0, 2, 1, 3});
        b.node("Transpose", {k4}, {k4 + "_t"});
        b.attr_ints("perm", {0, 2, 3, 1});
        b.node("Transpose", {v4}, {v4 + "_t"});
        b.attr_ints("perm", {0, 2, 1, 3});
        const std::string scores = "L" + std::to_string(il) + "_scores";
        b.node("MatMul", {q4 + "_t", k4 + "_t"}, {scores});
        const std::string scaled = "L" + std::to_string(il) + "_scaled";
        b.node("Mul", {scores, "attn_scale"}, {scaled});
        const std::string masked = "L" + std::to_string(il) + "_masked";
        b.node("Add", {scaled, "attn_mask"}, {masked});
        const std::string probs = "L" + std::to_string(il) + "_probs";
        b.node("Softmax", {masked}, {probs});
        b.attr_i("axis", -1);
        const std::string ctx = "L" + std::to_string(il) + "_ctx";
        b.node("MatMul", {probs, v4 + "_t"}, {ctx});
        b.node("Transpose", {ctx}, {ctx + "_t"}); // [B,NH,S,HD] -> [B,S,NH,HD]
        b.attr_ints("perm", {0, 2, 1, 3});
        const std::string ctx2 = b.reshape_to(ctx + "_t", "L" + std::to_string(il) + "_ctx2", {batch, max_seq, H});

        const std::string wo_w = "L" + std::to_string(il) + "_wo_w", wo_b = "L" + std::to_string(il) + "_wo_b";
        b.init_f32(wo_w, {(int64_t)H, (int64_t)H}, mat_f32(W("attn_output.weight")));
        b.init_f32(wo_b, {(int64_t)H}, dequant(W("attn_output.bias")));
        const std::string wo_mm = "L" + std::to_string(il) + "_wo_mm", ao = "L" + std::to_string(il) + "_ao";
        b.node("MatMul", {ctx2, wo_w}, {wo_mm});
        b.node("Add", {wo_mm, wo_b}, {ao});
        const std::string res1 = "L" + std::to_string(il) + "_res1";
        b.node("Add", {h, ao}, {res1});
        const std::string awn = "L" + std::to_string(il) + "_awn", awb = "L" + std::to_string(il) + "_awb";
        b.init_f32(awn, {(int64_t)H}, dequant(W("attn_output_norm.weight")));
        b.init_f32(awb, {(int64_t)H}, dequant(W("attn_output_norm.bias")));
        const std::string h1 = "L" + std::to_string(il) + "_h1";
        b.layer_norm(res1, awn, awb, spec_.eps, h1);

        // FFN: up -> GELU (via Erf, supported on NPU) -> down.
        const std::string up_w = "L" + std::to_string(il) + "_up_w", up_b = "L" + std::to_string(il) + "_up_b";
        b.init_f32(up_w, {(int64_t)H, (int64_t)F}, mat_f32(W("ffn_up.weight")));
        b.init_f32(up_b, {(int64_t)F}, dequant(W("ffn_up.bias")));
        const std::string up_mm = "L" + std::to_string(il) + "_up_mm", up = "L" + std::to_string(il) + "_up";
        b.node("MatMul", {h1, up_w}, {up_mm});
        b.node("Add", {up_mm, up_b}, {up});
        const std::string us = "L" + std::to_string(il) + "_us", ue = "L" + std::to_string(il) + "_ue";
        const std::string up1 = "L" + std::to_string(il) + "_up1", gelu = "L" + std::to_string(il) + "_gelu";
        b.node("Div", {up, "sqrt2"}, {us});
        b.node("Erf", {us}, {ue});
        b.node("Add", {ue, "one"}, {up1});
        b.node("Mul", {up1, "half"}, {up1 + "_h"});
        b.node("Mul", {up, up1 + "_h"}, {gelu});
        const std::string dw_w = "L" + std::to_string(il) + "_dw_w", dw_b = "L" + std::to_string(il) + "_dw_b";
        b.init_f32(dw_w, {(int64_t)F, (int64_t)H}, mat_f32(W("ffn_down.weight")));
        b.init_f32(dw_b, {(int64_t)H}, dequant(W("ffn_down.bias")));
        const std::string dw_mm = "L" + std::to_string(il) + "_dw_mm", fd = "L" + std::to_string(il) + "_fd";
        b.node("MatMul", {gelu, dw_w}, {dw_mm});
        b.node("Add", {dw_mm, dw_b}, {fd});
        const std::string res2 = "L" + std::to_string(il) + "_res2";
        b.node("Add", {h1, fd}, {res2});
        const std::string lwn = "L" + std::to_string(il) + "_lwn", lwb = "L" + std::to_string(il) + "_lwb";
        b.init_f32(lwn, {(int64_t)H}, dequant(W("layer_output_norm.weight")));
        b.init_f32(lwb, {(int64_t)H}, dequant(W("layer_output_norm.bias")));
        h = "L" + std::to_string(il) + "_h2";
        b.layer_norm(res2, lwn, lwb, spec_.eps, h);
    }

    // Pooling.
    std::string pooled;
    if (spec_.pooling_type == 1) {
        b.node("ReduceMean", {h}, {"pooled"});
        b.attr_ints("axes", {1});
        b.attr_i("keepdims", 0);
        pooled = "pooled";
    } else {
        b.init_i64("cls_idx_v", {1}, {0});
        b.node("Gather", {h, "cls_idx_v"}, {"cls"});
        b.attr_i("axis", 1);
        pooled = b.reshape_to("cls", "cls_vec", {batch, H});
    }
    // L2 normalisation (sentence-transformers Normalize).
    b.node("Mul", {pooled, pooled}, {"sq"});
    b.node("ReduceSum", {"sq"}, {"ss"});
    b.attr_ints("axes", {-1});
    b.attr_i("keepdims", 1);
    b.node("Add", {"ss", "norm_eps"}, {"ss_e"});
    b.node("Sqrt", {"ss_e"}, {"norm"});
    b.node("Div", {pooled, "norm"}, {"embedding"});
    b.output("embedding", {batch, H});
}

} // namespace embedding
} // namespace si
