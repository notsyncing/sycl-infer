#include "vision.h"

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
        throw std::runtime_error("mmproj: missing tensor: " + name);
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
        throw std::runtime_error("mmproj: expected f32 for " + name);
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
    const float c = 0.7978845608028654f; // sqrt(2/pi)
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
    default: throw std::runtime_error("mmproj: unsupported weight type " + std::to_string(t.type));
    }
}

// out[t][o] = bias[o] + sum_k W[o][k] * x[t][k]; one weight row is dequantized
// once and reused for every token.
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

// bilinear sample of the learned position grid (align_corners) for channel d
void pos_sample(const float * pos, int n_side, int n_embd, int d, int pw, int ph, int x, int y, float & out) {
    if (pw == n_side && ph == n_side) {
        out = pos[((size_t)y * n_side + x) * n_embd + d];
        return;
    }
    const float sx = pw > 1 ? (float)x * (n_side - 1) / (pw - 1) : 0.f;
    const float sy = ph > 1 ? (float)y * (n_side - 1) / (ph - 1) : 0.f;
    const int x0 = (int)sx, y0 = (int)sy;
    const int x1 = x0 + 1 < n_side ? x0 + 1 : x0;
    const int y1 = y0 + 1 < n_side ? y0 + 1 : y0;
    const float fx = sx - x0, fy = sy - y0;
    const float v00 = pos[((size_t)y0 * n_side + x0) * n_embd + d];
    const float v01 = pos[((size_t)y0 * n_side + x1) * n_embd + d];
    const float v10 = pos[((size_t)y1 * n_side + x0) * n_embd + d];
    const float v11 = pos[((size_t)y1 * n_side + x1) * n_embd + d];
    out = (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy;
}

} // namespace

float vt_get(const vt & t, int row, int k) {
    const size_t idx = (size_t)row * t.K + k;
    switch (t.type) {
    case GGML_TYPE_F32: return ((const float *)t.data)[idx];
    case GGML_TYPE_F16: return ggml_half_to_float(((const uint16_t *)t.data)[idx]);
    case GGML_TYPE_BF16: return bf16_to_f32(((const uint16_t *)t.data)[idx]);
    default: throw std::runtime_error("mmproj: unsupported weight type " + std::to_string(t.type));
    }
}

void vision_model::load(const std::string & path) {
    gguf.load(path);
    const gguf_file & f = gguf;
    auto & hp = this->hp;

    hp.image_size = (int)f.get_u32("clip.vision.image_size");
    hp.patch_size = (int)f.get_u32("clip.vision.patch_size");
    hp.n_embd = (int)f.get_u32("clip.vision.embedding_length");
    hp.n_ff = (int)f.get_u32("clip.vision.feed_forward_length");
    hp.n_layer = (int)f.get_u32("clip.vision.block_count");
    hp.n_head = (int)f.get_u32("clip.vision.attention.head_count");
    hp.head_dim = hp.n_head > 0 ? hp.n_embd / hp.n_head : 0;
    hp.proj_dim = (int)f.get_u32("clip.vision.projection_dim");
    hp.merge = (int)f.get_u32("clip.vision.spatial_merge_size", 2);
    hp.eps = f.get_f32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
    hp.rope_base = f.get_f32("clip.vision.rope_theta", 10000.0f);
    if (const gguf_kv * mean = f.meta("clip.vision.image_mean")) {
        for (int i = 0; i < 3 && i < (int)mean->arr.size(); i++) {
            hp.mean[i] = mean->arr[i].as_f32();
        }
    }
    if (const gguf_kv * std = f.meta("clip.vision.image_std")) {
        for (int i = 0; i < 3 && i < (int)std->arr.size(); i++) {
            hp.std[i] = std->arr[i].as_f32();
        }
    }

    const int P = hp.patch_size;
    const int patch_elems = 3 * P * P;

    // patch conv: the reference sums two convs over the same input.  The weight
    // is a 4D conv tensor [kw, kh, in_ch, out_ch], so the matmul view is
    // out_ch rows of kw*kh*in_ch elements (row-major, matching ggml's layout).
    auto conv_weight = [&](const char * name) {
        const gguf_tensor_info * ti = f.find(name);
        if (!ti) {
            throw std::runtime_error("mmproj: missing tensor: " + std::string(name));
        }
        if (ti->dims.size() != 4) {
            throw std::runtime_error("mmproj: expected a 4D conv weight");
        }
        vt t;
        t.data = ti->data;
        t.type = ti->type;
        t.K = (int)(ti->dims[0] * ti->dims[1] * ti->dims[2]);
        t.N = (int)ti->dims[3];
        return t;
    };
    vt pw0 = conv_weight("v.patch_embd.weight");
    vt pw1 = conv_weight("v.patch_embd.weight.1");
    if (pw0.N != hp.n_embd || pw0.K != patch_elems) {
        throw std::runtime_error("mmproj: unexpected patch embedding shape");
    }
    patch_w.resize((size_t)hp.n_embd * patch_elems);
    for (int o = 0; o < hp.n_embd; o++) {
        for (int k = 0; k < patch_elems; k++) {
            float v = vt_get(pw0, o, k);
            if (pw1.data) {
                v += vt_get(pw1, o, k);
            }
            patch_w[(size_t)o * patch_elems + k] = v;
        }
    }
    patch_b = bind_f32_opt(f, "v.patch_embd.bias");

    // learned absolute position embeddings, transposed to [pos][n_embd]
    vt pe = bind_vt(f, "v.position_embd.weight");
    const int n_side = (int)std::lround(std::sqrt((double)pe.N));
    hp.n_pos_side = n_side;
    if (n_side * n_side != pe.N) {
        throw std::runtime_error("mmproj: position embedding not square");
    }
    pos_embd_host.resize((size_t)pe.N * pe.K);
    for (int s = 0; s < pe.N; s++) {
        for (int d = 0; d < pe.K; d++) {
            pos_embd_host[(size_t)s * pe.K + d] = vt_get(pe, s, d);
        }
    }
    pos_embd = pos_embd_host.data();

    post_ln = bind_f32_opt(f, "v.post_ln.weight");
    post_ln_b = bind_f32_opt(f, "v.post_ln.bias");

    layers.resize(hp.n_layer);
    for (int il = 0; il < hp.n_layer; il++) {
        const std::string pre = "v.blk." + std::to_string(il) + ".";
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
    mm0 = bind_vt(f, "mm.0.weight");
    mm0_b = bind_f32_opt(f, "mm.0.bias");
    mm2 = bind_vt(f, "mm.2.weight");
    mm2_b = bind_f32_opt(f, "mm.2.bias");
}

vision_model::~vision_model() {
    if (dev_weights) {
        // the queue that allocated it is gone by now; the caller owns it
    }
}

void vision_model::upload(sycl::queue & q) {
    dev_weights_size = gguf.map_size;
    dev_weights = sycl::malloc_device(dev_weights_size, q);
    if (!dev_weights) {
        throw std::runtime_error("mmproj device allocation failed");
    }
    q.memcpy(dev_weights, gguf.map_base, dev_weights_size).wait();
}

vision_input vision_model::make_input(const vision_model & vm, const mm_image & img) {
    vision_input in;
    in.chw = img.chw.data();
    in.width = img.width;
    in.height = img.height;
    in.pw = img.width / vm.hp.patch_size;
    in.ph = img.height / vm.hp.patch_size;
    in.n_patches = in.pw * in.ph;
    in.out_w = in.pw / vm.hp.merge;
    in.out_h = in.ph / vm.hp.merge;
    in.n_out = in.out_w * in.out_h;
    return in;
}

void vision_model::build_raw_patches(const vision_input & in, std::vector<float> & x) const {
    const int P = hp.patch_size;
    const int M = hp.merge;
    const int np = in.n_patches;
    const int patch_elems = 3 * P * P;
    x.assign((size_t)np * patch_elems, 0.f);
    int tok = 0;
    for (int my = 0; my < in.out_h; my++) {
        for (int mx = 0; mx < in.out_w; mx++) {
            for (int dy = 0; dy < M; dy++) {
                for (int dx = 0; dx < M; dx++) {
                    const int py = my * M + dy;
                    const int px = mx * M + dx;
                    float * xt = x.data() + (size_t)tok * patch_elems;
                    int i = 0;
                    for (int c = 0; c < 3; c++) {
                        const float * plane = in.chw + (size_t)c * in.width * in.height;
                        for (int kh = 0; kh < P; kh++) {
                            const float * row = plane + (size_t)(py * P + kh) * in.width + px * P;
                            for (int kw = 0; kw < P; kw++, i++) {
                                xt[i] = row[kw];
                            }
                        }
                    }
                    tok++;
                }
            }
        }
    }
}

void vision_model::build_pos_emb(const vision_input & in, std::vector<float> & out) const {
    const int E = hp.n_embd;
    const int M = hp.merge;
    const int n_side = hp.n_pos_side;
    const int np = in.n_patches;
    out.assign((size_t)np * E, 0.f);
    if (!pos_embd) {
        return;
    }
    int tok = 0;
    for (int my = 0; my < in.out_h; my++) {
        for (int mx = 0; mx < in.out_w; mx++) {
            for (int dy = 0; dy < M; dy++) {
                for (int dx = 0; dx < M; dx++) {
                    const int py = my * M + dy;
                    const int px = mx * M + dx;
                    float * xt = out.data() + (size_t)tok * E;
                    if (in.pw == n_side && in.ph == n_side) {
                        const float * pe = pos_embd + (size_t)(py * n_side + px) * E;
                        for (int d = 0; d < E; d++) {
                            xt[d] = pe[d];
                        }
                    } else {
                        for (int d = 0; d < E; d++) {
                            pos_sample(pos_embd, n_side, E, d, in.pw, in.ph, px, py, xt[d]);
                        }
                    }
                    tok++;
                }
            }
        }
    }
}

void vision_model::build_patch_input(const vision_input & in, std::vector<float> & x) const {
    const int P = hp.patch_size;
    const int E = hp.n_embd;
    const int M = hp.merge;
    const int patch_elems = 3 * P * P;
    const int n_side = hp.n_pos_side;
    const int np = in.n_patches;
    x.assign((size_t)np * E, 0.f);

    // patch embedding + learned position embeddings, in merged token order
    std::vector<float> pv(patch_elems);
    int tok = 0;
    for (int my = 0; my < in.out_h; my++) {
        for (int mx = 0; mx < in.out_w; mx++) {
            for (int dy = 0; dy < M; dy++) {
                for (int dx = 0; dx < M; dx++) {
                    const int py = my * M + dy;
                    const int px = mx * M + dx;
                    int i = 0;
                    for (int c = 0; c < 3; c++) {
                        const float * plane = in.chw + (size_t)c * in.width * in.height;
                        for (int kh = 0; kh < P; kh++) {
                            const float * row = plane + (size_t)(py * P + kh) * in.width + px * P;
                            for (int kw = 0; kw < P; kw++, i++) {
                                pv[i] = row[kw];
                            }
                        }
                    }
                    float * xt = x.data() + (size_t)tok * E;
                    for (int o = 0; o < E; o++) {
                        const float * w = patch_w.data() + (size_t)o * patch_elems;
                        float acc = patch_b ? patch_b[o] : 0.f;
                        for (int k = 0; k < patch_elems; k++) {
                            acc += w[k] * pv[k];
                        }
                        xt[o] = acc;
                    }
                    // the position embedding is permuted with the same reorder,
                    // so sample it at the original (py, px) patch coordinate
                    if (pos_embd) {
                        if (in.pw == n_side && in.ph == n_side) {
                            const float * pe = pos_embd + (size_t)(py * n_side + px) * E;
                            for (int d = 0; d < E; d++) {
                                xt[d] += pe[d];
                            }
                        } else {
                            for (int d = 0; d < E; d++) {
                                float v = 0;
                                pos_sample(pos_embd, n_side, E, d, in.pw, in.ph, px, py, v);
                                xt[d] += v;
                            }
                        }
                    }
                    tok++;
                }
            }
        }
    }
}

void vision_model::encode_host(const vision_input & in, std::vector<float> & out) const {
    const int E = hp.n_embd;
    const int M = hp.merge;
    const int ff = hp.n_ff;
    const int np = in.n_patches;

    std::vector<float> x;
    build_patch_input(in, x);

    // ---- transformer layers ----
    const int HD = hp.head_dim;
    const int NH = hp.n_head;
    // the same base the device path hands vit_rope_launch (hp.rope_base, from
    // clip.vision.rope_theta): hardcoding 10000 here made the reference diverge
    // from the GPU for any mmproj that sets a different theta
    const float rope_scale = std::pow(hp.rope_base, -2.0f / (float)(HD / 2));
    std::vector<float> qkv((size_t)np * 3 * E);
    std::vector<float> attn((size_t)np * E);
    std::vector<float> tmp((size_t)np * E);
    std::vector<float> upbuf((size_t)np * ff);
    std::vector<float> downbuf((size_t)np * E);
    std::vector<float> scores((size_t)np);

    for (int il = 0; il < hp.n_layer; il++) {
        const vision_layer & L = layers[il];
        // ln1
        for (int t = 0; t < np; t++) {
            layer_norm(x.data() + (size_t)t * E, L.ln1, L.ln1_b, tmp.data() + (size_t)t * E, E, hp.eps);
        }
        // fused qkv + rope on Q/K
        matmul_all(L.qkv, L.qkv_b, tmp.data(), qkv.data(), np);
        // token -> patch coordinates in the original grid (merged order)
        auto patch_xy = [&](int t, int & px, int & py) {
            const int m = t / (M * M);
            const int sub = t % (M * M);
            const int my = m / in.out_w, mx = m % in.out_w;
            px = mx * M + sub % M;
            py = my * M + sub / M;
        };
        for (int t = 0; t < np; t++) {
            int px, py;
            patch_xy(t, px, py);
            float * qt = qkv.data() + (size_t)t * 3 * E;
            for (int h = 0; h < NH; h++) {
                float * q = qt + h * HD;
                float * k = qt + E + h * HD;
                for (int ic = 0; ic < HD / 2; ic++) {
                    const int sec = ic / 16; // sections {16,16,16,16}, exponent resets
                    const int p = ic % 16;
                    const int pos = sec == 0 ? py : px;
                    const float theta = pos * std::pow(rope_scale, (float)p);
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
        // bidirectional attention per head
        const float scale = 1.0f / std::sqrt((float)HD);
        for (int h = 0; h < NH; h++) {
            for (int t0 = 0; t0 < np; t0++) {
                const float * q = qkv.data() + (size_t)t0 * 3 * E + h * HD;
                float mx = -INFINITY;
                for (int t1 = 0; t1 < np; t1++) {
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
                for (int t1 = 0; t1 < np; t1++) {
                    scores[t1] = std::exp(scores[t1] - mx);
                    sum += scores[t1];
                }
                const float inv = (float)(1.0 / sum);
                float * o = attn.data() + (size_t)t0 * E + h * HD;
                for (int d = 0; d < HD; d++) {
                    o[d] = 0;
                }
                for (int t1 = 0; t1 < np; t1++) {
                    const float p = scores[t1] * inv;
                    const float * v = qkv.data() + (size_t)t1 * 3 * E + 2 * E + h * HD;
                    for (int d = 0; d < HD; d++) {
                        o[d] += p * v[d];
                    }
                }
            }
        }
        // attn_out projection + residual
        matmul_all(L.out, L.out_b, attn.data(), tmp.data(), np);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] += tmp[i];
        }
        // ln2 + ffn (up -> gelu -> down) + residual
        for (int t = 0; t < np; t++) {
            layer_norm(x.data() + (size_t)t * E, L.ln2, L.ln2_b, tmp.data() + (size_t)t * E, E, hp.eps);
        }
        matmul_all(L.up, L.up_b, tmp.data(), upbuf.data(), np);
        for (int t = 0; t < np; t++) {
            gelu_tanh(upbuf.data() + (size_t)t * ff, ff);
        }
        matmul_all(L.down, L.down_b, upbuf.data(), downbuf.data(), np);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] += downbuf[i];
        }
    }

    if (post_ln) {
        for (int t = 0; t < np; t++) {
            layer_norm(x.data() + (size_t)t * E, post_ln, post_ln_b, tmp.data() + (size_t)t * E, E, hp.eps);
        }
        x.swap(tmp);
    }

    // ---- multimodal projector: concat 4 patches, mm.0 -> gelu -> mm.2 ----
    const int C = M * M;
    const int hidden = C * E;
    std::vector<float> fe((size_t)in.n_out * hidden);
    for (int m = 0; m < in.n_out; m++) {
        for (int i = 0; i < C; i++) {
            const float * src = x.data() + (size_t)(m * C + i) * E;
            std::memcpy(fe.data() + ((size_t)m * C + i) * E, src, sizeof(float) * E);
        }
    }
    std::vector<float> h1((size_t)in.n_out * mm0.N);
    matmul_all(mm0, mm0_b, fe.data(), h1.data(), in.n_out);
    for (int m = 0; m < in.n_out; m++) {
        gelu_tanh(h1.data() + (size_t)m * mm0.N, mm0.N);
    }
    out.assign((size_t)in.n_out * hp.proj_dim, 0.f);
    matmul_all(mm2, mm2_b, h1.data(), out.data(), in.n_out);
}

// ---------------------------------------------------------------------------
// Device forward.  Mirrors encode_host one-to-one: patch GEMM, then per layer
// LayerNorm -> fused QKV -> bias -> 2D RoPE -> attention -> output GEMM
// (residual) -> LayerNorm -> FFN up -> bias -> GELU -> FFN down (residual), then
// post LayerNorm and the 2x2 merger (concat is a stride reinterpretation).
// ---------------------------------------------------------------------------
void vision_model::encode_device(sycl::queue & q, const vision_input & in, float * d_out) {
    const int E = hp.n_embd;
    const int ff = hp.n_ff;
    const int HD = hp.head_dim;
    const int NH = hp.n_head;
    const int M = hp.merge;
    const int np = in.n_patches;
    const int n_out = in.n_out;
    const int patch_elems = 3 * hp.patch_size * hp.patch_size;
    const int C = M * M;
    const int hidden = C * E;
    if (np <= 0 || n_out <= 0) {
        throw std::runtime_error("vision: empty patch grid");
    }
    if (np > kMaxImgPatches) {
        throw std::runtime_error("vision: image needs " + std::to_string(np) + " patch tokens (max "
                                 + std::to_string(kMaxImgPatches) + ")");
    }
    if (HD != 64) {
        throw std::runtime_error("vision: unsupported head_dim");
    }

    if (!dev_weights) {
        upload(q);
    }
    if (!d_patch_w) {
        d_patch_w = sycl::malloc_device((size_t)E * patch_elems * sizeof(float), q);
        if (!d_patch_w) {
            throw std::runtime_error("vision: patch weight allocation failed");
        }
        q.memcpy(d_patch_w, patch_w.data(), (size_t)E * patch_elems * sizeof(float));
    }

    if (scratch_patches < np) {
        auto fre = [&](void * p) {
            if (p) {
                sycl::free(p, q);
            }
        };
        fre(d_patch_in);
        fre(d_x);
        fre(d_ln);
        fre(d_qkv);
        fre(d_attn);
        fre(d_ffn);
        fre(d_mm0);
        fre(d_pos);
        d_patch_in = (float *)sycl::malloc_device((size_t)np * patch_elems * 4, q);
        d_pos = (float *)sycl::malloc_device((size_t)np * E * 4, q);
        d_x = (float *)sycl::malloc_device((size_t)np * E * 4, q);
        d_ln = (float *)sycl::malloc_device((size_t)np * E * 4, q);
        d_qkv = (float *)sycl::malloc_device((size_t)np * 3 * E * 4, q);
        d_attn = (float *)sycl::malloc_device((size_t)np * E * 4, q);
        d_ffn = (float *)sycl::malloc_device((size_t)np * ff * 4, q);
        d_mm0 = (float *)sycl::malloc_device((size_t)n_out * hidden * 4, q);
        if (!d_patch_in || !d_pos || !d_x || !d_ln || !d_qkv || !d_attn || !d_ffn || !d_mm0) {
            throw std::runtime_error("vision: device scratch allocation failed");
        }
        scratch_patches = np;
    }

    auto devf = [&](const float * p) { return (const float *)dev_ptr(p); };

    std::vector<float> pin, pos;
    build_raw_patches(in, pin);
    build_pos_emb(in, pos);
    q.memcpy(d_patch_in, pin.data(), pin.size() * sizeof(float));
    q.memcpy(d_pos, pos.data(), pos.size() * sizeof(float));

    // patch embedding: out[N=E, K=3*P*P], then patch bias + position embedding
    vit_gemm_launch(q, d_patch_w, GGML_TYPE_F32, E, patch_elems, d_patch_in, patch_elems, d_x, E, np, 1.f, nullptr);
    if (patch_b) {
        vit_add_bias_launch(q, d_x, E, devf(patch_b), np, E);
    }
    vit_add_launch(q, d_x, E, d_pos, E, np, E);

    const float scale = 1.0f / std::sqrt((float)HD);
    for (int il = 0; il < hp.n_layer; il++) {
        const vision_layer & L = layers[il];
        vit_layernorm_launch(q, d_x, E, devf(L.ln1), devf(L.ln1_b), d_ln, E, np, E, hp.eps);
        vit_gemm_launch(q, dev_ptr(L.qkv.data), L.qkv.type, 3 * E, E, d_ln, E, d_qkv, 3 * E, np, 1.f, nullptr);
        vit_add_bias_launch(q, d_qkv, 3 * E, devf(L.qkv_b), np, 3 * E);
        vit_rope_launch(q, d_qkv, 3 * E, np, NH, HD, in.out_w, M, hp.rope_base);
        vit_attn_launch(q, d_qkv, 3 * E, d_attn, E, np, NH, HD, scale);
        // x += out_b, then x += attn_out_proj (residual through the GEMM)
        vit_add_bias_launch(q, d_x, E, devf(L.out_b), np, E);
        vit_gemm_launch(q, dev_ptr(L.out.data), L.out.type, E, E, d_attn, E, d_x, E, np, 1.f, d_x);
        vit_layernorm_launch(q, d_x, E, devf(L.ln2), devf(L.ln2_b), d_ln, E, np, E, hp.eps);
        vit_gemm_launch(q, dev_ptr(L.up.data), L.up.type, ff, E, d_ln, E, d_ffn, ff, np, 1.f, nullptr);
        vit_add_bias_launch(q, d_ffn, ff, devf(L.up_b), np, ff);
        vit_gelu_launch(q, d_ffn, np * ff);
        vit_add_bias_launch(q, d_x, E, devf(L.down_b), np, E);
        vit_gemm_launch(q, dev_ptr(L.down.data), L.down.type, E, ff, d_ffn, ff, d_x, E, np, 1.f, d_x);
    }

    vit_layernorm_launch(q, d_x, E, devf(post_ln), devf(post_ln_b), d_ln, E, np, E, hp.eps);

    // merger: the 2x2 concat is a stride reinterpretation of d_ln
    vit_gemm_launch(q, dev_ptr(mm0.data), mm0.type, mm0.N, hidden, d_ln, hidden, d_mm0, mm0.N, n_out, 1.f, nullptr);
    vit_add_bias_launch(q, d_mm0, mm0.N, devf(mm0_b), n_out, mm0.N);
    vit_gelu_launch(q, d_mm0, n_out * mm0.N);
    vit_gemm_launch(q, dev_ptr(mm2.data), mm2.type, hp.proj_dim, mm0.N, d_mm0, mm0.N, d_out, hp.proj_dim, n_out, 1.f,
                    nullptr);
    vit_add_bias_launch(q, d_out, hp.proj_dim, devf(mm2_b), n_out, hp.proj_dim);
    q.wait();
}

} // namespace si
