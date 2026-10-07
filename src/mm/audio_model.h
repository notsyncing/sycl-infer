#pragma once
// ---------------------------------------------------------------------------
// Qwen3.5 audio encoder ("AuT"-style tower, in the vision mmproj mold).
//
// The audio mmproj is a **separate GGUF** (selected with --audio, or the
// server's `audio_mmproj_path`); without its weights every audio request fails
// with a clear error.  The tower takes a log-mel spectrogram produced by
// src/mm/audio.cpp (16 kHz, 25 ms / 10 ms, 128 bins) and returns one embedding
// per remaining frame -- every embedding becomes one text token in the prompt.
//
// GGUF schema (metadata prefix `audio.`, tensors prefix `a.`):
//   audio.sample_rate / n_fft / hop_length / n_mel / f_min / f_max   mel geometry
//   audio.embedding_length           tower width E
//   audio.feed_forward_length / block_count / attention.head_count
//   audio.attention.layer_norm_epsilon / rope_theta
//   audio.position_embd_length       learned per-frame positions (cap)
//   audio.projection_dim             text width; 0 = tower width feeds the LLM
//   a.conv1.weight[3][1][n_mel][C]   a.conv1.bias[C]
//   a.conv2.weight[3][1][C][E]       a.conv2.bias[E]      (stride 2)
//   a.position_embd.weight[n_pos][E]
//   a.blk.{i}.attn_qkv.weight|bias, attn_out, ffn_up, ffn_down, ln1, ln2
//   a.post_ln.weight|bias
//   a.out.weight|bias                optional projection to the text width
//
// Conv stem: conv1 (k=3, stride 1, pad 1) -> GELU -> conv2 (k=3, stride 2,
// pad 1) -> GELU -> learned position embedding -> N transformer blocks
// (LayerNorm + fused QKV + 1D RoPE + bidirectional attention + FFN) ->
// post LayerNorm -> optional `a.out` projection.  n_out = ceil(n_frames/2).
// ---------------------------------------------------------------------------
#include <cstdint>
#include <string>
#include <vector>

#include <sycl/sycl.hpp> // IWYU pragma: keep

#include "gguf.h"
#include "vision.h"

namespace si {

struct audio_hparams {
    int sample_rate = 16000;
    int n_fft = 400;
    int hop = 160;
    int n_mel = 128;
    float f_min = 0.f;
    float f_max = 8000.f;
    int n_embd = 0;    // tower width E
    int n_ff = 0;      // feed-forward width
    int n_layer = 0;
    int n_head = 0;
    int head_dim = 0;  // fixed at 64 (shared attention kernels)
    int proj_dim = 0;  // 0 = none (tower width feeds the text model)
    int n_pos = 0;     // position embedding length
    float eps = 1e-6f;
    float rope_base = 10000.0f;
};

// A preprocessed log-mel spectrogram plus the geometry the encoder needs.
struct audio_input {
    const float * mel = nullptr; // [n_frames][n_mel], row t = frame t
    int n_frames = 0;            // pre-conv mel frames
    int n_out = 0;               // embeddings (ceil(n_frames / 2))
};

struct audio_model {
    gguf_file gguf;
    audio_hparams hp;
    // dequantized conv weights (the reference f32 path) and position embeddings
    std::vector<float> c1_w;   // [C][3*n_mel]
    std::vector<float> c1_b;   // [C]
    std::vector<float> c2_w;   // [E][3*C]
    std::vector<float> c2_b;   // [E]
    std::vector<float> pos_embd_host; // [n_pos][E]
    const float * pos_embd = nullptr;
    const float * post_ln = nullptr;
    const float * post_ln_b = nullptr;
    std::vector<vision_layer> layers; // same layout as the vision blocks
    vt out;                          // optional projection [proj_dim][E]
    const float * out_b = nullptr;

    void load(const std::string & path);

    static audio_input make_input(const audio_model & am, const float * mel, int n_frames);

    // Reference forward on the host: writes n_out * out_width embeddings
    // (out_width = proj_dim if set, else the tower width).
    void encode_host(const audio_input & in, std::vector<float> & out) const;

    // device copy of the weights (one blob)
    void * dev_weights = nullptr;
    size_t dev_weights_size = 0;
    // Queue every device allocation below was made on (see vision.h: same
    // one-queue-per-model rule, shared helpers in enc_common.h).
    sycl::queue * dev_q = nullptr;
    void upload(sycl::queue & q);
    // releases dev_weights + conv blob + scratch on dev_q (no-op when nothing
    // was ever uploaded).  Models are never copied, only referenced - and the
    // deleted copy ctor suppresses the implicit default ctor, so default it.
    audio_model() = default;
    ~audio_model();
    audio_model(const audio_model &) = delete;
    audio_model & operator=(const audio_model &) = delete;
    const void * dev_ptr(const void * host_ptr) const {
        return (const char *)dev_weights + ((const char *)host_ptr - (const char *)gguf.map_base);
    }

    // Audio forward on the device; `d_out` holds n_out * out_width floats.
    void encode_device(sycl::queue & q, const audio_input & in, float * d_out);

    // device scratch (owned by this model)
    void * d_cw = nullptr;
    float * d_mel = nullptr, *d_x = nullptr, *d_ln = nullptr, *d_qkv = nullptr;
    float * d_attn = nullptr, *d_ffn = nullptr, *d_pos = nullptr, *d_post = nullptr;
    int scratch_frames = 0;
};

int audio_out_width(const audio_model & am);

} // namespace si