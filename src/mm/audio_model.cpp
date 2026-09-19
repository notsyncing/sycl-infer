#include "audio_model.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "kernels.h"
#include "quant.h"

namespace si {

namespace {

vt bind_vt(const gguf_file & f, const std::string & name) {
    const gguf_tensor_info * ti = f.find(name);
    if (!ti) {
        throw std::runtime_error("audio: missing tensor: " + name);
    }
    vt t;
    t.data = ti->data;
    t.type = ti->type;
    t.K = (int)ti->dims[0];
    t.N = (int)ti->n_rows();
    return t;
}

const float * bind_f32_opt(const gguf_file & f, const std::string & name) {
    const gguf_tensor_info * ti = f.find(name);
    if (!ti) {
        return nullptr;
    }
    if (ti->type != GGML_TYPE_F32) {
        throw std::runtime_error("audio: expected f32 for " + name);
    }
    return (const float *)ti->data;
}

float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void layer_norm(const float * x, const float * w, const float * b, float * out, int n, float eps) {
    double mean = 0;
    for (int i = 0; i < n; i++) {
        mean += x[i];
    }
    mean /= n;
    double var = 0;
    for (int i = 0; i < n; i++) {
        const double d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    const float inv = 1.0f / std::sqrt((float)var + eps);
    for (int i = 0; i < n; i++) {
        const float v = (x[i] - (float)mean) * inv;
        out[i] = b ? v * w[i] + b[i] : v * w[i];
    }
}

void gelu_tanh(float * x, int n) {
    const float c = 0.7978845608028654f;
    for (int i = 0; i < n; i++) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
    }
}

void dequant_row(const vt & t, int row, float * dst) {
    const size_t off = (size_t)row * t.K;
    switch (t.type) {
    case GGML_TYPE_F32: std::memcpy(dst, (const float *)t.data + off, sizeof(float) * t.K); break;
    case GGML_TYPE_F16:
        for (int k = 0; k < t.K; k++) {
            dst[k] = ggml_half_to_float(((const uint16_t *)t.data)[off + k]);
        }
        break;
    case GGML_TYPE_BF16:
        for (int k = 0; k < t.K; k++) {
            dst[k] = bf16_to_f32(((const uint16_t *)t.data)[off + k]);
        }
        break;
    default: throw std::runtime_error("audio: unsupported weight type " + std::to_string(t.type));
    }
}

void matmul_all(const vt & w, const float * bias, const float * x, float * out, int ntok) {
    std::vector<float> row(w.K);
    for (int o = 0; o < w.N; o++) {
        dequant_row(w, o, row.data());
        const float b = bias ? bias[o] : 0.f;
        for (int t = 0; t < ntok; t++) {
            const float * xt = x + (size_t)t * w.K;
            float acc = b;
            for (int k = 0; k < w.K; k++) {
                acc += row[k] * xt[k];
            }
            out[(size_t)t * w.N + o] = acc;
        }
    }
}

// 1D conv: y[t][o] = b[o] + sum_tap sum_i w[o][tap*xin+i] * x[t*stride+tap-pad][i]
// (zero outside the input); y_frames is fixed by the caller.
void conv1d_all(const float * w, const float * b, const float * x, int x_frames, int x_in, int w_tap, int stride,
                int pad, float * out, int y_frames, int w_out) {
    for (int t = 0; t < y_frames; t++) {
        for (int o = 0; o < w_out; o++) {
            float acc = b ? b[o] : 0.f;
            const float * wr = w + (size_t)o * w_tap * x_in;
            for (int tap = 0; tap < w_tap; tap++) {
                const int row = t * stride + tap - pad;
                if (row < 0 || row >= x_frames) {
                    continue;
                }
                const float * xr = x + (size_t)row * x_in;
                for (int i = 0; i < x_in; i++) {
                    acc += wr[tap * x_in + i] * xr[i];
                }
            }
            out[(size_t)t * w_out + o] = acc;
        }
    }
}

} // namespace

int audio_out_width(const audio_model & am) {
    return am.hp.proj_dim > 0 ? am.hp.proj_dim : am.hp.n_embd;
}

void audio_model::load(const std::string & path) {
    gguf.load(path);
    const gguf_file & f = gguf;
    auto & hp = this->hp;

    hp.sample_rate = (int)f.get_u32("audio.sample_rate", 16000);
    hp.n_fft = (int)f.get_u32("audio.n_fft", 400);
    hp.hop = (int)f.get_u32("audio.hop_length", 160);
    hp.n_mel = (int)f.get_u32("audio.n_mel", 128);
    hp.f_min = f.get_f32("audio.f_min", 0.f);
    hp.f_max = f.get_f32("audio.f_max", 8000.f);
    hp.n_embd = (int)f.get_u32("audio.embedding_length");
    hp.n_ff = (int)f.get_u32("audio.feed_forward_length");
    hp.n_layer = (int)f.get_u32("audio.block_count");
    hp.n_head = (int)f.get_u32("audio.attention.head_count");
    hp.head_dim = hp.n_head > 0 ? hp.n_embd / hp.n_head : 0;
    hp.proj_dim = (int)f.get_u32("audio.projection_dim", 0);
    hp.n_pos = (int)f.get_u32("audio.position_embd_length", 0);
    hp.eps = f.get_f32("audio.attention.layer_norm_epsilon", 1e-6f);
    hp.rope_base = f.get_f32("audio.rope_theta", 10000.0f);
    if (hp.n_embd <= 0 || hp.n_ff <= 0 || hp.n_layer <= 0 || hp.n_head <= 0) {
        throw std::runtime_error("audio: incomplete metadata (is this an audio mmproj?)");
    }
    if (hp.head_dim != 64) {
        throw std::runtime_error("audio: unsupported head_dim " + std::to_string(hp.head_dim));
    }

    // conv stem.  The GGUF tensors are 4D [w][h][cin][cout] (ggml conv layout),
    // so the matmul view is out_ch rows of w*h*cin elements.
    auto conv_tensor = [&](const char * name, int & taps, int & in_ch, int & out_ch) {
        const gguf_tensor_info * ti = f.find(name);
        if (!ti) {
            throw std::runtime_error("audio: missing tensor: " + std::string(name));
        }
        if (ti->dims.size() != 4) {
            throw std::runtime_error("audio: expected a 4D conv weight");
        }
        taps = (int)ti->dims[0];
        in_ch = (int)ti->dims[2];
        out_ch = (int)ti->dims[3];
        vt t;
        t.data = ti->data;
        t.type = ti->type;
        t.K = (int)(ti->dims[0] * ti->dims[1] * ti->dims[2]);
        t.N = (int)ti->dims[3];
        return t;
    };
    int c1_tap, c1_in, C, c2_tap, c2_in, E;
    vt c1 = conv_tensor("a.conv1.weight", c1_tap, c1_in, C);
    vt c2 = conv_tensor("a.conv2.weight", c2_tap, c2_in, E);
    if (c1_tap != 3 || c1_in != hp.n_mel || c2_tap != 3 || c2_in != C || E != hp.n_embd) {
        throw std::runtime_error("audio: unexpected conv shapes");
    }
    c1_w.resize((size_t)C * c1.K);
    for (int o = 0; o < C; o++) {
        dequant_row(c1, o, c1_w.data() + (size_t)o * c1.K);
    }
    c1_b.clear();
    if (const float * p = bind_f32_opt(f, "a.conv1.bias")) {
        c1_b.assign(p, p + C);
    }
    c2_w.resize((size_t)E * c2.K);
    for (int o = 0; o < E; o++) {
        dequant_row(c2, o, c2_w.data() + (size_t)o * c2.K);
    }
    c2_b.clear();
    if (const float * p = bind_f32_opt(f, "a.conv2.bias")) {
        c2_b.assign(p, p + E);
    }

    // learned positional embeddings [n_pos][E] (dequantized to f32)
    vt pe = bind_vt(f, "a.position_embd.weight");
    if (hp.n_pos <= 0) {
        hp.n_pos = (int)pe.N;
    }
    // the embedding rows are the cap; an over-large metadata value must not
    // drive %-indexing past the tensor (or the awkward extra rows)
    hp.n_pos = std::min(hp.n_pos, (int)pe.N);
    if (pe.K != hp.n_embd) {
        throw std::runtime_error("audio: position embedding width mismatch");
    }
    pos_embd_host.resize((size_t)pe.N * pe.K);
    for (int s = 0; s < pe.N; s++) {
        for (int d = 0; d < pe.K; d++) {
            pos_embd_host[(size_t)s * pe.K + d] = vt_get(pe, s, d);
        }
    }
    pos_embd = pos_embd_host.data();

    post_ln = bind_f32_opt(f, "a.post_ln.weight");
    post_ln_b = bind_f32_opt(f, "a.post_ln.bias");

    layers.resize(hp.n_layer);
    for (int il = 0; il < hp.n_layer; il++) {
        const std::string pre = "a.blk." + std::to_string(il) + ".";
        vision_layer & L = layers[il];
        L.qkv = bind_vt(f, pre + "attn_qkv.weight");
        L.qkv_b = bind_f32_opt(f, pre + "attn_qkv.bias");
        L.out = bind_vt(f, pre + "attn_out.weight");
        L.out_b = bind_f32_opt(f, pre + "attn_out.bias");
        L.up = bind_vt(f, pre + "ffn_up.weight");
        L.up_b = bind_f32_opt(f, pre + "ffn_up.bias");
        L.down = bind_vt(f, pre + "ffn_down.weight");
        L.down_b = bind_f32_opt(f, pre + "ffn_down.bias");
        L.ln1 = bind_f32_opt(f, pre + "ln1.weight");
        L.ln1_b = bind_f32_opt(f, pre + "ln1.bias");
        L.ln2 = bind_f32_opt(f, pre + "ln2.weight");
        L.ln2_b = bind_f32_opt(f, pre + "ln2.bias");
    }
    out = bind_vt(f, "a.out.weight");
    out_b = bind_f32_opt(f, "a.out.bias");
}

audio_model::~audio_model() {}

void audio_model::upload(sycl::queue & q) {
    dev_weights_size = gguf.map_size;
    dev_weights = sycl::malloc_device(dev_weights_size, q);
    if (!dev_weights) {
        throw std::runtime_error("audio device allocation failed");
    }
    q.memcpy(dev_weights, gguf.map_base, dev_weights_size).wait();
}

audio_input audio_model::make_input(const audio_model & am, const float * mel, int n_frames) {
    audio_input in;
    in.mel = mel;
    in.n_frames = n_frames;
    in.n_out = (n_frames + 1) / 2; // conv2: k 3, stride 2, pad 1
    return in;
}

void audio_model::encode_host(const audio_input & in, std::vector<float> & out) const {
    const int E = hp.n_embd;
    const int ff = hp.n_ff;
    const int n_mel = hp.n_mel;
    const int n_frames = in.n_frames;
    const int n_out = in.n_out;
    const int C = (int)(c2_w.size() / ((size_t)E * 3)); // conv1 output channels

    // conv stem
    std::vector<float> xa((size_t)n_frames * C);
    conv1d_all(c1_w.data(), c1_b.data(), in.mel, n_frames, n_mel, 3, 1, 1, xa.data(), n_frames, C);
    gelu_tanh(xa.data(), n_frames * C);

    std::vector<float> x((size_t)n_out * E);
    conv1d_all(c2_w.data(), c2_b.data(), xa.data(), n_frames, C, 3, 2, 1, x.data(), n_out, E);
    gelu_tanh(x.data(), n_out * E);
    if (pos_embd && hp.n_pos > 0) {
        for (int t = 0; t < n_out; t++) {
            const float * pe = pos_embd + (size_t)(t % hp.n_pos) * E;
            float * xt = x.data() + (size_t)t * E;
            for (int d = 0; d < E; d++) {
                xt[d] += pe[d];
            }
        }
    }

    // ---- transformer blocks (bidirectional attention, 1D rope over frames) --
    const int HD = hp.head_dim;
    const int NH = hp.n_head;
    const float inv_freq_scale = std::pow(hp.rope_base, -2.0f / (float)HD);
    std::vector<float> qkv((size_t)n_out * 3 * E);
    std::vector<float> attn((size_t)n_out * E);
    std::vector<float> tmp((size_t)n_out * E);
    std::vector<float> upbuf((size_t)n_out * ff);
    std::vector<float> downbuf((size_t)n_out * E);
    std::vector<float> scores((size_t)n_out);

    for (int il = 0; il < hp.n_layer; il++) {
        const vision_layer & L = layers[il];
        for (int t = 0; t < n_out; t++) {
            layer_norm(x.data() + (size_t)t * E, L.ln1, L.ln1_b, tmp.data() + (size_t)t * E, E, hp.eps);
        }
        matmul_all(L.qkv, L.qkv_b, tmp.data(), qkv.data(), n_out);
        // 1D rope: every pair uses the frame index as its position
        for (int t = 0; t < n_out; t++) {
            float * qt = qkv.data() + (size_t)t * 3 * E;
            for (int h = 0; h < NH; h++) {
                float * q = qt + h * HD;
                float * k = qt + E + h * HD;
                for (int ic = 0; ic < HD / 2; ic++) {
                    const float theta = (float)t * std::pow(inv_freq_scale, (float)ic);
                    const float c = std::cos(theta), s = std::sin(theta);
                    const float q0 = q[ic], q1 = q[ic + HD / 2];
                    q[ic] = q0 * c - q1 * s;
                    q[ic + HD / 2] = q0 * s + q1 * c;
                    const float k0 = k[ic], k1 = k[ic + HD / 2];
                    k[ic] = k0 * c - k1 * s;
                    k[ic + HD / 2] = k0 * s + k1 * c;
                }
            }
        }
        const float scale = 1.0f / std::sqrt((float)HD);
        for (int h = 0; h < NH; h++) {
            for (int t0 = 0; t0 < n_out; t0++) {
                const float * q = qkv.data() + (size_t)t0 * 3 * E + h * HD;
                float mx = -INFINITY;
                for (int t1 = 0; t1 < n_out; t1++) {
                    const float * k = qkv.data() + (size_t)t1 * 3 * E + E + h * HD;
                    float acc = 0;
                    for (int d = 0; d < HD; d++) {
                        acc += q[d] * k[d];
                    }
                    acc *= scale;
                    scores[t1] = acc;
                    if (acc > mx) {
                        mx = acc;
                    }
                }
                double sum = 0;
                for (int t1 = 0; t1 < n_out; t1++) {
                    scores[t1] = std::exp(scores[t1] - mx);
                    sum += scores[t1];
                }
                const float inv = (float)(1.0 / sum);
                float * o = attn.data() + (size_t)t0 * E + h * HD;
                for (int d = 0; d < HD; d++) {
                    o[d] = 0;
                }
                for (int t1 = 0; t1 < n_out; t1++) {
                    const float p = scores[t1] * inv;
                    const float * v = qkv.data() + (size_t)t1 * 3 * E + 2 * E + h * HD;
                    for (int d = 0; d < HD; d++) {
                        o[d] += p * v[d];
                    }
                }
            }
        }
        matmul_all(L.out, L.out_b, attn.data(), tmp.data(), n_out);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] += tmp[i];
        }
        for (int t = 0; t < n_out; t++) {
            layer_norm(x.data() + (size_t)t * E, L.ln2, L.ln2_b, tmp.data() + (size_t)t * E, E, hp.eps);
        }
        matmul_all(L.up, L.up_b, tmp.data(), upbuf.data(), n_out);
        for (int t = 0; t < n_out; t++) {
            gelu_tanh(upbuf.data() + (size_t)t * ff, ff);
        }
        matmul_all(L.down, L.down_b, upbuf.data(), downbuf.data(), n_out);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] += downbuf[i];
        }
    }

    if (post_ln) {
        for (int t = 0; t < n_out; t++) {
            layer_norm(x.data() + (size_t)t * E, post_ln, post_ln_b, tmp.data() + (size_t)t * E, E, hp.eps);
        }
        x.swap(tmp);
    }

    // optional projection to the text width
    const int W = audio_out_width(*this);
    out.clear();
    out.resize((size_t)n_out * W);
    if (W == E) {
        std::memcpy(out.data(), x.data(), sizeof(float) * n_out * E);
    } else {
        matmul_all(this->out, out_b, x.data(), out.data(), n_out);
    }
}

// ---------------------------------------------------------------------------
// Device forward.  Conv stem via at_conv1d_launch, then the transformer blocks
// through the shared vision kernels (vit_gemm / layernorm / gelu / add_bias /
// attn) and the 1D audio rope via at_rope1d_launch.
// ---------------------------------------------------------------------------
void audio_model::encode_device(sycl::queue & q, const audio_input & in, float * d_out) {
    const int E = hp.n_embd;
    const int ff = hp.n_ff;
    const int HD = hp.head_dim;
    const int NH = hp.n_head;
    const int n_mel = hp.n_mel;
    const int n_frames = in.n_frames;
    const int n_out = in.n_out;
    const int C = (int)(c2_w.size() / ((size_t)E * 3));
    if (n_frames <= 0 || n_out <= 0) {
        throw std::runtime_error("audio: empty input");
    }
    if (n_frames > kMaxImgTokens * 2) {
        throw std::runtime_error("audio: input too long (" + std::to_string(n_frames) + " frames, max "
                                 + std::to_string(kMaxImgTokens * 2) + ")");
    }
    if (HD != 64) {
        throw std::runtime_error("audio: unsupported head_dim");
    }

    if (!dev_weights) {
        upload(q);
    }
    if (scratch_frames < n_frames) {
        auto fre = [&](void * p) {
            if (p) {
                sycl::free(p, q);
            }
        };
        fre(d_cw);
        fre(d_pos);
        fre(d_mel);
        fre(d_x);
        fre(d_ln);
        fre(d_qkv);
        fre(d_attn);
        fre(d_ffn);
        // one blob: c1_w, c1_b, c2_w, c2_b, position embeddings (all f32)
        const size_t c1_n = (size_t)C * 3 * n_mel;
        const size_t c2_n = (size_t)E * 3 * C;
        d_cw = sycl::malloc_device((c1_n + C + c2_n + E + (size_t)hp.n_pos * E) * 4, q);
        d_mel = (float *)sycl::malloc_device((size_t)n_frames * n_mel * 4, q);
        d_pos = (float *)sycl::malloc_device((size_t)n_frames * E * 4, q); // built per call
        d_x = (float *)sycl::malloc_device((size_t)std::max((size_t)n_frames * C, (size_t)n_out * E) * 4, q);
        d_ln = (float *)sycl::malloc_device((size_t)n_out * E * 4, q);
        d_qkv = (float *)sycl::malloc_device((size_t)n_out * 3 * E * 4, q);
        d_attn = (float *)sycl::malloc_device((size_t)n_out * E * 4, q);
        d_ffn = (float *)sycl::malloc_device((size_t)n_out * ff * 4, q);
        if (!d_cw || !d_mel || !d_pos || !d_x || !d_ln || !d_qkv || !d_attn || !d_ffn) {
            throw std::runtime_error("audio: device scratch allocation failed");
        }
        scratch_frames = n_frames;
    }

    float * d_c1 = (float *)d_cw;
    float * d_c1b = d_c1 + C * 3 * n_mel;
    float * d_c2 = d_c1b + C;
    float * d_c2b = d_c2 + E * 3 * C;
    float * d_pe = d_c2b + E;
    q.memcpy(d_c1, c1_w.data(), (size_t)C * 3 * n_mel * 4);
    if (!c1_b.empty()) {
        q.memcpy(d_c1b, c1_b.data(), (size_t)C * 4);
    }
    q.memcpy(d_c2, c2_w.data(), (size_t)E * 3 * C * 4);
    if (!c2_b.empty()) {
        q.memcpy(d_c2b, c2_b.data(), (size_t)E * 4);
    }
    q.memcpy(d_pe, pos_embd_host.data(), (size_t)hp.n_pos * E * 4);
    q.memcpy(d_mel, in.mel, (size_t)n_frames * n_mel * 4);

    // per-token positions (row t uses embedding t % n_pos), like the vision
    // path builds the permuted position rows on the host
    std::vector<float> pos((size_t)n_out * E, 0.f);
    if (pos_embd && hp.n_pos > 0) {
        for (int t = 0; t < n_out; t++) {
            std::memcpy(pos.data() + (size_t)t * E, pos_embd + (size_t)(t % hp.n_pos) * E, sizeof(float) * E);
        }
    }
    q.memcpy(d_pos, pos.data(), pos.size() * sizeof(float));

    // conv1 (stride 1) -> gelu -> conv2 (stride 2) -> gelu -> + position
    at_conv1d_launch(q, d_mel, n_frames, n_mel, 3, 1, 1, d_c1, c1_b.empty() ? nullptr : d_c1b, d_x, n_frames, C);
    vit_gelu_launch(q, d_x, (size_t)n_frames * C);
    at_conv1d_launch(q, d_x, n_frames, C, 3, 2, 1, d_c2, c2_b.empty() ? nullptr : d_c2b, d_ln, n_out, E);
    vit_gelu_launch(q, d_ln, (size_t)n_out * E);
    vit_add_launch(q, d_ln, E, d_pos, E, n_out, E);

    auto devf = [&](const float * p) { return (const float *)dev_ptr(p); };
    const float scale = 1.0f / std::sqrt((float)HD);
    for (int il = 0; il < hp.n_layer; il++) {
        const vision_layer & L = layers[il];
        vit_layernorm_launch(q, d_ln, E, devf(L.ln1), devf(L.ln1_b), d_attn, E, n_out, E, hp.eps);
        vit_gemm_launch(q, dev_ptr(L.qkv.data), L.qkv.type, 3 * E, E, d_attn, E, d_qkv, 3 * E, n_out, 1.f, nullptr);
        vit_add_bias_launch(q, d_qkv, 3 * E, devf(L.qkv_b), n_out, 3 * E);
        at_rope1d_launch(q, d_qkv, 3 * E, n_out, NH, HD, hp.rope_base);
        vit_attn_launch(q, d_qkv, 3 * E, d_x, E, n_out, NH, HD, scale);
        // x += out_b, then x += attn_out_proj (residual through the GEMM),
        // mirroring the vision encoder's block order
        vit_add_bias_launch(q, d_ln, E, devf(L.out_b), n_out, E);
        vit_gemm_launch(q, dev_ptr(L.out.data), L.out.type, E, E, d_x, E, d_ln, E, n_out, 1.f, d_ln);
        vit_layernorm_launch(q, d_ln, E, devf(L.ln2), devf(L.ln2_b), d_attn, E, n_out, E, hp.eps);
        vit_gemm_launch(q, dev_ptr(L.up.data), L.up.type, ff, E, d_attn, E, d_ffn, ff, n_out, 1.f, nullptr);
        vit_add_bias_launch(q, d_ffn, ff, devf(L.up_b), n_out, ff);
        vit_gelu_launch(q, d_ffn, (size_t)n_out * ff);
        vit_add_bias_launch(q, d_ln, E, devf(L.down_b), n_out, E);
        vit_gemm_launch(q, dev_ptr(L.down.data), L.down.type, E, ff, d_ffn, ff, d_ln, E, n_out, 1.f, d_ln);
    }

    if (post_ln) {
        vit_layernorm_launch(q, d_ln, E, devf(post_ln), devf(post_ln_b), d_x, E, n_out, E, hp.eps);
    } else {
        vit_copy_launch(q, d_ln, E, d_x, E, n_out, E);
    }
    q.wait();
    const int W = audio_out_width(*this);
    if (W == E) {
        vit_copy_launch(q, d_x, E, d_out, W, n_out, E);
    } else {
        vit_gemm_launch(q, dev_ptr(out.data), out.type, W, E, d_x, E, d_out, W, n_out, 1.f, nullptr);
        vit_add_bias_launch(q, d_out, W, devf(out_b), n_out, W);
    }
    q.wait();
}

} // namespace si