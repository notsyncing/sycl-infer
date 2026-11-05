#include "vision.h"

#include "env.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "kernels.h"
#include "quant.h"
#include "enc_common.h"
#include "dnnl_gemm.h"

#include <chrono>

namespace si {

namespace {

// PF_LOAD_TIME: the mmproj's load/upload phases.  The page's startup is a third
// of its wall clock, and the mmproj alone is about half of that - but "load"
// covers the GGUF parse, the qwen2vl qkv f32 re-layout, a 1.3 GB H2D and the
// host-side int8 conversion, which have completely different fixes.  Off by
// default; only clock reads.
struct vision_load_timer {
    bool on = si::env::flag("PF_LOAD_TIME");
    std::chrono::steady_clock::time_point first, last;
    vision_load_timer() : first(std::chrono::steady_clock::now()) {
        last = first;
    }
    void lap(const char * what) {
        if (!on) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "[load]   vision %-24s %8.1f ms\n", what,
                std::chrono::duration<double, std::milli>(now - last).count());
        last = now;
    }
    void total(const char * what) {
        if (!on) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "[load]   vision %-24s %8.1f ms\n", what,
                std::chrono::duration<double, std::milli>(now - first).count());
        last = now;
    }
};

// (the rest of the anonymous namespace)

// PF_VISION_TIME: wall-clock split of the vision forward.  The stamps are taken
// with q.wait() so the number is the device time of that stage (an in-order
// queue otherwise lets launches run ahead and every stamp reads ~0).  Only the
// diagnostic pays any cost; the default path never waits here.
struct vision_stage_timer {
    bool on = false;
    std::chrono::high_resolution_clock::time_point t;
    double patch = 0, ln1 = 0, qkv = 0, rope = 0, attn = 0, out = 0, ln2 = 0, up = 0, act = 0, down = 0,
           post_ln = 0, mm = 0, layers = 0;
    explicit vision_stage_timer(bool enabled) : on(enabled) {
        if (on) {
            t = std::chrono::high_resolution_clock::now();
        }
    }
    void lap(sycl::queue & q, double & acc) {
        if (!on) {
            return;
        }
        q.wait();
        auto now = std::chrono::high_resolution_clock::now();
        acc += std::chrono::duration<double, std::milli>(now - t).count();
        t = now;
    }
    void lap_layer(sycl::queue & q, double & acc) {
        if (!on) {
            return;
        }
        q.wait();
        auto now = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - t).count();
        acc += ms;
        layers += ms;
        t = now;
    }
};

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

void vision_model::load(const std::string & path, model_profile::vision_kind requested) {
    vision_load_timer lt;
    if (requested == model_profile::vision_kind::none) {
        throw std::runtime_error("mmproj: the model profile declares no vision tower");
    }
    kind = requested;
    gguf.load(path);
    lt.lap("gguf parse/mmap");
    const gguf_file & f = gguf;
    auto & hp = this->hp;

    // The mmproj states its own projector; it must agree with the text model's
    // profile, or a Qwen2-VL tower would run under Qwen3.5's layout rules.
    if (const std::string * pt = f.get_str("clip.projector_type")) {
        model_profile::vision_kind meta = model_profile::vision_kind::none;
        if (pt->find("qwen2vl") != std::string::npos) {
            meta = model_profile::vision_kind::qwen2vl;
        } else if (pt->find("qwen3vl") != std::string::npos || pt->find("qwen3") != std::string::npos) {
            meta = model_profile::vision_kind::qwen3vl;
        }
        if (meta != model_profile::vision_kind::none && meta != kind) {
            throw std::runtime_error("mmproj: projector type \"" + *pt +
                                     "\" disagrees with the model profile's vision tower");
        }
    }

    const bool q2 = kind == model_profile::vision_kind::qwen2vl;
    hp.image_size = (int)f.get_u32("clip.vision.image_size");
    hp.patch_size = (int)f.get_u32("clip.vision.patch_size", q2 ? 14u : 16u);
    hp.n_embd = (int)f.get_u32("clip.vision.embedding_length");
    hp.n_ff = (int)f.get_u32("clip.vision.feed_forward_length");
    hp.n_layer = (int)f.get_u32("clip.vision.block_count");
    hp.n_head = (int)f.get_u32("clip.vision.attention.head_count");
    hp.head_dim = hp.n_head > 0 ? hp.n_embd / hp.n_head : 0;
    hp.proj_dim = (int)f.get_u32("clip.vision.projection_dim");
    hp.merge = (int)f.get_u32("clip.vision.spatial_merge_size", 2);
    hp.eps = f.get_f32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
    // Qwen2-VL's vision RoPE is fixed at 1e4; its metadata may carry no theta.
    hp.rope_base = q2 ? 10000.0f : f.get_f32("clip.vision.rope_theta", 10000.0f);
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

    // patch conv: the reference sums two convs over the same input (Qwen2-VL's
    // temporal pair).  The weight is a 4D conv tensor [kw, kh, in_ch, out_ch],
    // so the matmul view is out_ch rows of kw*kh*in_ch elements (row-major,
    // matching ggml's layout; the same order for both towers).
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
    vt pw1{};
    if (f.find("v.patch_embd.weight.1")) {
        pw1 = conv_weight("v.patch_embd.weight.1");
    }
    if (pw0.N != hp.n_embd || pw0.K != patch_elems) {
        throw std::runtime_error("mmproj: unexpected patch embedding shape");
    }
    // The vision GEMM loads eight weights at a time with a 16-byte load, so the
    // row length must be a multiple of 8 (and 8 floats is 32 bytes, keeping
    // every row base 16-byte aligned).
    patch_K = (patch_elems + 7) & ~7;
    patch_w.assign((size_t)hp.n_embd * patch_K, 0.f);
    for (int o = 0; o < hp.n_embd; o++) {
        for (int k = 0; k < patch_elems; k++) {
            float v = vt_get(pw0, o, k);
            if (pw1.data) {
                v += vt_get(pw1, o, k);
            }
            patch_w[(size_t)o * patch_K + k] = v;
        }
    }
    patch_b = bind_f32_opt(f, "v.patch_embd.bias");

    // learned absolute position embeddings (qwen3vl only)
    hp.n_pos_side = 0;
    if (!q2) {
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
    }

    {
        const int patch_area = hp.patch_size * hp.patch_size * hp.merge * hp.merge;
        // Qwen3-VL's preprocessor_config.json: shortest_edge = 65536.  The
        // Qwen2-VL reference (MinerU) uses a much smaller floor and its verified
        // page path never reaches it, so keep its existing 8-token floor.
        hp.min_pixels = (kind == model_profile::vision_kind::qwen3vl ? 64 : 8) * patch_area;
    }

    post_ln = bind_f32_opt(f, "v.post_ln.weight");
    post_ln_b = bind_f32_opt(f, "v.post_ln.bias");
    if (!post_ln) {
        throw std::runtime_error("mmproj: missing v.post_ln.weight");
    }

    // Bind the block tensors first: the qwen2vl file has separate q/k/v, the
    // qwen3vl file one fused qkv.  Both end up in the same vision_layer view.
    struct raw_layer {
        vt qkv, q, k, v, out, up, down;
        const float * qkv_b = nullptr;
        const float * q_b = nullptr, * k_b = nullptr, * v_b = nullptr;
        const float * out_b = nullptr, * up_b = nullptr, * down_b = nullptr;
        const float * ln1 = nullptr, * ln1_b = nullptr, * ln2 = nullptr, * ln2_b = nullptr;
    };
    std::vector<raw_layer> raw((size_t)hp.n_layer);
    for (int il = 0; il < hp.n_layer; il++) {
        const std::string pre = "v.blk." + std::to_string(il) + ".";
        raw_layer & R = raw[(size_t)il];
        if (q2) {
            R.q = bind_vt(f, pre + "attn_q.weight");
            R.k = bind_vt(f, pre + "attn_k.weight");
            R.v = bind_vt(f, pre + "attn_v.weight");
            R.q_b = bind_f32_opt(f, pre + "attn_q.bias");
            R.k_b = bind_f32_opt(f, pre + "attn_k.bias");
            R.v_b = bind_f32_opt(f, pre + "attn_v.bias");
        } else {
            R.qkv = bind_vt(f, pre + "attn_qkv.weight");
            R.qkv_b = bind_f32_opt(f, pre + "attn_qkv.bias");
        }
        R.out = bind_vt(f, pre + "attn_out.weight");
        R.out_b = bind_f32_opt(f, pre + "attn_out.bias");
        R.up = bind_vt(f, pre + "ffn_up.weight");
        R.up_b = bind_f32_opt(f, pre + "ffn_up.bias");
        R.down = bind_vt(f, pre + "ffn_down.weight");
        R.down_b = bind_f32_opt(f, pre + "ffn_down.bias");
        R.ln1 = bind_f32_opt(f, pre + "ln1.weight");
        R.ln1_b = bind_f32_opt(f, pre + "ln1.bias");
        R.ln2 = bind_f32_opt(f, pre + "ln2.weight");
        R.ln2_b = bind_f32_opt(f, pre + "ln2.bias");
    }
    mm0 = bind_vt(f, "mm.0.weight");
    mm0_b = bind_f32_opt(f, "mm.0.bias");
    mm2 = bind_vt(f, "mm.2.weight");
    mm2_b = bind_f32_opt(f, "mm.2.bias");

    // The Qwen2-VL mmproj's `clip.vision.feed_forward_length` says 896 (the text
    // hidden size) but ffn_up is [1280, 5120].  Trust the tensor for every
    // tower; a wrong width silently truncates the FFN.
    if (!raw.empty() && raw[0].up.N > 0) {
        hp.n_ff = raw[0].up.N;
    }

    lt.lap("hparams/tensor bind");
    const int E = hp.n_embd;
    const size_t kOffNone = (size_t)-1;
    const bool f32_all = si::env::flag("PF_VISION_F32");
    // PF_VISION_DEFER_QKV=0 builds the combined q|k|v f32 side blob at load, the
    // pre-optimisation behaviour (A/B only: it costs ~0.67 s and 0.63 GB).
    const bool defer_qkv = si::env::i32("PF_VISION_DEFER_QKV", 1) != 0;
    auto tsize = [](const vt & t) { return (size_t)t.N * (size_t)t.K; };
    auto wants_f32 = [&](const vt & t) { return f32_all && t.data && t.type != GGML_TYPE_F32; };

    // Reserve the exact side-blob size first; every converted tensor and the
    // qwen2vl combined qkv+bias is appended to it.
    size_t total = 0;
    for (const raw_layer & R : raw) {
        if (q2) {
            total += 3 * (size_t)E * (size_t)E; // the combined q/k/v matrix; the bias is separate
        } else if (wants_f32(R.qkv)) {
            total += tsize(R.qkv);
        }
        for (const vt * t : {&R.out, &R.up, &R.down}) {
            if (wants_f32(*t)) {
                total += tsize(*t);
            }
        }
    }
    for (const vt * t : {&mm0, &mm2}) {
        if (wants_f32(*t)) {
            total += tsize(*t);
        }
    }
    w32.reserve(total);
    w32_bytes = total * sizeof(float);
    if (q2) {
        qkv_b32.assign((size_t)hp.n_layer * 3 * (size_t)E, 0.f);
    }
    // Offsets are assigned against a cursor, not against w32.size(): the
    // qwen2vl q/k/v matrix is 3E*E floats per layer (~629 MB on MinerU's
    // tower) and is only needed by the host reference forward and the
    // vit_gemm fallback, so it is filled on demand (ensure_w32) unless the
    // PF_VISION_F32 diagnostic is on.
    size_t cursor = 0;
    auto append_vt = [&](const vt & t, bool fill) {
        const size_t off = cursor;
        cursor += tsize(t);
        if (fill) {
            if (w32.size() < cursor) {
                w32.resize(cursor);
            }
            for (int r = 0; r < t.N; r++) {
                for (int k = 0; k < t.K; k++) {
                    w32[off + (size_t)r * t.K + k] = vt_get(t, r, k);
                }
            }
        }
        return off;
    };

    std::vector<size_t> qkv_off((size_t)hp.n_layer, kOffNone), qkv_b_off((size_t)hp.n_layer, kOffNone);
    std::vector<size_t> out_off((size_t)hp.n_layer, kOffNone), up_off((size_t)hp.n_layer, kOffNone);
    std::vector<size_t> down_off((size_t)hp.n_layer, kOffNone);
    size_t mm0_off = kOffNone, mm2_off = kOffNone;

    for (int il = 0; il < hp.n_layer; il++) {
        const raw_layer & R = raw[(size_t)il];
        if (q2) {
            // One contiguous [3E][E] f32 qkv plus a [3E] bias: the decoder path
            // then runs one fused projection exactly like qwen3vl, and the
            // separate-tensor layout never reaches a kernel.  Only the bias is
            // filled here; the matrix is the deferred side blob.
            const size_t off = append_vt(vt{nullptr, GGML_TYPE_F32, E, 3 * E}, false);
            auto copy = [&](const vt & t, int dst0) {
                if (t.K != E || t.N != E) {
                    throw std::runtime_error("mmproj: qwen2vl q/k/v is not [E][E]");
                }
                for (int r = 0; r < E; r++) {
                    for (int k = 0; k < E; k++) {
                        w32[off + (size_t)(dst0 + r) * E + k] = vt_get(t, r, k);
                    }
                }
            };
            if (f32_all || !defer_qkv) {
                if (w32.size() < off + 3 * (size_t)E * (size_t)E) {
                    w32.resize(off + 3 * (size_t)E * (size_t)E);
                }
                copy(R.q, 0);
                copy(R.k, E);
                copy(R.v, 2 * E);
            }
            host_qkv.push_back(qkv_src{R.q, R.k, R.v, off});
            qkv_off[(size_t)il] = off;
            const size_t boff = (size_t)il * 3 * (size_t)E;
            for (int i = 0; i < E; i++) {
                qkv_b32[boff + i] = R.q_b ? R.q_b[i] : 0.f;
                qkv_b32[boff + E + i] = R.k_b ? R.k_b[i] : 0.f;
                qkv_b32[boff + 2 * E + i] = R.v_b ? R.v_b[i] : 0.f;
            }
            qkv_b_off[(size_t)il] = boff;
        } else if (wants_f32(R.qkv)) {
            qkv_off[(size_t)il] = append_vt(R.qkv, true);
        }
        if (wants_f32(R.out)) {
            out_off[(size_t)il] = append_vt(R.out, true);
        }
        if (wants_f32(R.up)) {
            up_off[(size_t)il] = append_vt(R.up, true);
        }
        if (wants_f32(R.down)) {
            down_off[(size_t)il] = append_vt(R.down, true);
        }
    }
    if (wants_f32(mm0)) {
        mm0_off = append_vt(mm0, true);
    }
    if (wants_f32(mm2)) {
        mm2_off = append_vt(mm2, true);
    }
    if (cursor != total) {
        throw std::runtime_error("mmproj: internal f32 side-blob size mismatch");
    }
    w32_built = f32_all || !defer_qkv; // otherwise the deferred q/k/v matrix is filled on demand

    auto view = [&](size_t off, const vt & fallback) {
        return off == kOffNone ? fallback : vt{w32.data() + off, GGML_TYPE_F32, fallback.K, fallback.N};
    };

    lt.lap("f32 side blob (qkv relayout)");
    layers.resize((size_t)hp.n_layer);
    for (int il = 0; il < hp.n_layer; il++) {
        const raw_layer & R = raw[(size_t)il];
        vision_layer & L = layers[(size_t)il];
        if (q2) {
            L.qkv = vt{w32.data() + qkv_off[(size_t)il], GGML_TYPE_F32, E, 3 * E};
            L.qkv_b = qkv_b32.data() + qkv_b_off[(size_t)il];
        } else {
            L.qkv = view(qkv_off[(size_t)il], R.qkv);
            L.qkv_b = R.qkv_b;
        }
        L.out = view(out_off[(size_t)il], R.out);
        L.out_b = R.out_b;
        L.up = view(up_off[(size_t)il], R.up);
        L.up_b = R.up_b;
        L.down = view(down_off[(size_t)il], R.down);
        L.down_b = R.down_b;
        L.ln1 = R.ln1;
        L.ln1_b = R.ln1_b;
        L.ln2 = R.ln2;
        L.ln2_b = R.ln2_b;
    }
    mm0 = view(mm0_off, mm0);
    mm2 = view(mm2_off, mm2);
    lt.total("load total");
}

// The deferred qwen2vl q|k|v f32 matrix: only the host reference forward and
// the vit_gemm fallback read it, and both are cold paths (the page's encoder
// runs every linear on the oneDNN int8 matmul, which converts q/k/v from the
// GGUF's own f16 tensors).
void vision_model::ensure_w32() const {
    if (w32_built) {
        return;
    }
    const int E = hp.n_embd;
    w32.resize(w32_bytes / sizeof(float));
    for (const qkv_src & S : host_qkv) {
        auto copy = [&](const vt & t, int dst0) {
            for (int r = 0; r < E; r++) {
                for (int k = 0; k < E; k++) {
                    w32[S.off + (size_t)(dst0 + r) * E + k] = vt_get(t, r, k);
                }
            }
        };
        copy(S.q, 0);
        copy(S.k, E);
        copy(S.v, 2 * E);
    }
    w32_built = true;
}

vision_model::~vision_model() {
    free_dev_ptrs(dev_q,
                  {dev_weights, d_w32, d_qkv_b32, d_patch_w, d_patch_in, d_pos, d_x, d_ln, d_qkv, d_attn, d_ffn, d_mm0});
    delete gemm;
    gemm = nullptr;
}

// One vision linear on the oneDNN int8 matmul, chunked to the activation
// scratch's row cap (the text model's largest batch, kMaxB*kMaxT).  Returns
// false when the weight is not registered on this instance, so the caller can
// run the old vit_gemm for it and stay correct.
bool vision_model::linear_dnnl(sycl::queue & q, const void * key, const float * x, int x_stride, int T, int K,
                               float * out, int out_stride, const float * residual, float alpha, int N) {
    (void)q;
    (void)N;
    if (!gemm || !key || T <= 0 || K <= 0) {
        return false;
    }
    if (!gemm->has_weight(key)) {
        return false;
    }
    const int kChunk = kMaxB * kMaxT;
    for (int t = 0; t < T; t += kChunk) {
        const int m = std::min(kChunk, T - t);
        // do_split=false: these are prefill-shaped calls (v.split must stay
        // false or gemm() would take the decode/verify nat_gemm branch, whose
        // activation form this quantizer did not produce).
        const act_view v = gemm->quantize(x + (size_t)t * x_stride, nullptr, x_stride, x_stride, m, K, false);
        if (!v.valid()) {
            return false;
        }
        if (!gemm->gemm(v, key, residual ? residual + (size_t)t * out_stride : nullptr, alpha, m, K,
                        out + (size_t)t * out_stride, out_stride)) {
            return false;
        }
    }
    return true;
}

void vision_model::upload(sycl::queue & q) {
    vision_load_timer lt;
    pin_dev_queue(dev_q, q, "vision");
    if (dev_weights) {
        return;
    }
    const int E = hp.n_embd;

    // The vision linears on the text model's oneDNN int8 path.  vit_gemm tiles
    // M in 32-token groups and re-reads each weight block per tile; at 5476
    // patches that is ~172 passes over every linear.  The oneDNN matmul reads
    // each weight once per <=1024-row chunk instead.  Registration is lazy:
    // only the M shapes actually asked for get a primitive (a page's content
    // crops all have different patch counts), so this does not create a ladder
    // of a thousand primitives per tensor.
    //
    // Registration runs *before* the blob copy because it decides what the
    // blob has to contain: once every linear is on the int8 path, the only
    // mmproj tensors a device kernel reads are the norms/biases (3.7 MB of the
    // 1.32 GB file), so the 1.28 GB of f16 linears is neither copied nor kept
    // resident.
    //
    // Default on for the qwen2vl tower (verified: identical page md5, self-check
    // cos 0.9991, 5x on the layout image).  Off for qwen3vl: that tower is
    // BF16/head_dim 64-72 and there is no model here to verify it against, so
    // the flag is left as the opt-in until it is measured (PF_VISION_DNNL=1).
    bool all = false;
    // PF_VISION_COMPACT=0 restores the whole-file blob (fallback/debug A/B).
    const bool compact_ok = si::env::i32("PF_VISION_COMPACT", 1) != 0;
    const int dnnl_dflt = kind == model_profile::vision_kind::qwen2vl ? 1 : 0;
    if (si::env::i32("PF_VISION_DNNL", dnnl_dflt) != 0) {
        gemm = new dnnl_gemm(q);
        all = true;
        auto add = [&](const vt & w) {
            if (!w.data) {
                return;
            }
            if (!gemm->add_weight(w.data, w.data, w.type, w.K, w.N, /*eager=*/false)) {
                all = false;
            }
        };
        for (size_t il = 0; il < layers.size(); il++) {
            vision_layer & L = layers[il];
            bool qkv_ok = false;
            if (kind == model_profile::vision_kind::qwen2vl && il < host_qkv.size()) {
                // q/k/v are three tensors and the kernel consumes one fused
                // [3E][E] projection.  Convert them straight from the GGUF
                // rather than from the 3E*E f32 copy that exists only for the
                // host path.
                const qkv_src & S = host_qkv[il];
                const void * srcs[3] = {S.q.data, S.k.data, S.v.data};
                const int rows[3] = {E, E, E};
                if (S.q.type == S.k.type && S.q.type == S.v.type && S.q.K == E && S.q.N == E
                    && S.k.K == E && S.k.N == E && S.v.K == E && S.v.N == E) {
                    // lazy: the tower's M values are per-crop patch counts (every
                    // content block differs), so a ladder cannot predict them -
                    // this is the same trade the other vision linears already
                    // make with eager=false.
                    qkv_ok = gemm->add_weight_blocks(L.qkv.data, srcs, rows, 3, S.q.type, E, 3 * E, /*lazy=*/true);
                }
                if (!qkv_ok) {
                    // mixed types/shapes: build the f32 copy and take the old route
                    ensure_w32();
                }
            }
            if (!qkv_ok) {
                add(L.qkv);
            }
            add(L.out);
            add(L.up);
            add(L.down);
        }
        add(mm0);
        add(mm2);
        gemm->report_load("vision");
        lt.lap("int8 convert (add_weight)");
        if (si::env::flag("PF_VISION_TIME")) {
            fprintf(stderr, "[vtime] dnnl vision: %s, %.1f MiB int8 weights\n", all ? "on" : "partial",
                    (double)gemm->weight_bytes() / 1048576.0);
        }
    }

    // Weight blob.  Compact when every linear is on the int8 path; otherwise
    // the whole file is mapped 1:1 so a vit_gemm fallback can still read any
    // tensor at its GGUF offset.
    blob_spans.clear();
    if (all && compact_ok) {
        // the host-side tensor each linear was converted from
        std::vector<const void *> linear;
        for (size_t il = 0; il < layers.size(); il++) {
            if (kind == model_profile::vision_kind::qwen2vl && il < host_qkv.size()) {
                linear.push_back(host_qkv[il].q.data);
                linear.push_back(host_qkv[il].k.data);
                linear.push_back(host_qkv[il].v.data);
            }
        }
        for (const vision_layer & L : layers) {
            for (const vt * t : {&L.out, &L.up, &L.down}) {
                linear.push_back(t->data);
            }
            if (kind != model_profile::vision_kind::qwen2vl) {
                linear.push_back(L.qkv.data);
            }
        }
        linear.push_back(mm0.data);
        linear.push_back(mm2.data);
        auto is_linear = [&](const void * p) {
            for (const void * q2p : linear) {
                if (q2p == p) {
                    return true;
                }
            }
            return false;
        };
        size_t need = 0;
        for (const gguf_tensor_info & t : gguf.tensors) {
            if (!t.data || t.nbytes() == 0 || is_linear(t.data)) {
                continue;
            }
            blob_spans.push_back(blob_span{(const char *)t.data, (size_t)t.nbytes(), need});
            need += (size_t)t.nbytes();
        }
        std::sort(blob_spans.begin(), blob_spans.end(),
                  [](const blob_span & a, const blob_span & b) { return a.host < b.host; });
        dev_weights_size = need;
        dev_weights = sycl::malloc_device(dev_weights_size ? dev_weights_size : 1, q);
        if (!dev_weights) {
            throw std::runtime_error("mmproj device allocation failed");
        }
        for (const blob_span & sp : blob_spans) {
            q.memcpy((char *)dev_weights + sp.off, sp.host, sp.bytes);
        }
        q.wait();
    } else {
        dev_weights_size = gguf.map_size;
        dev_weights = sycl::malloc_device(dev_weights_size, q);
        if (!dev_weights) {
            throw std::runtime_error("mmproj device allocation failed");
        }
        q.memcpy(dev_weights, gguf.map_base, dev_weights_size).wait();
        // a fallback reads the qwen2vl q/k/v through w32, so that copy must
        // exist and be resident
        if (!host_qkv.empty()) {
            ensure_w32();
        }
        if (!w32.empty()) {
            d_w32 = sycl::malloc_device<float>(w32.size(), q);
            if (!d_w32) {
                throw std::runtime_error("mmproj side-blob device allocation failed");
            }
            q.memcpy(d_w32, w32.data(), w32.size() * sizeof(float)).wait();
            lt.lap("f32 side blob H2D");
        }
    }
    lt.lap("blob H2D");
    if (!qkv_b32.empty()) {
        d_qkv_b32 = sycl::malloc_device<float>(qkv_b32.size(), q);
        if (!d_qkv_b32) {
            throw std::runtime_error("mmproj qkv-bias allocation failed");
        }
        q.memcpy(d_qkv_b32, qkv_b32.data(), qkv_b32.size() * sizeof(float)).wait();
    }
    lt.total("upload total");
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
    x.assign((size_t)np * patch_K, 0.f);
    int tok = 0;
    for (int my = 0; my < in.out_h; my++) {
        for (int mx = 0; mx < in.out_w; mx++) {
            for (int dy = 0; dy < M; dy++) {
                for (int dx = 0; dx < M; dx++) {
                    const int py = my * M + dy;
                    const int px = mx * M + dx;
                    float * xt = x.data() + (size_t)tok * patch_K;
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
    const int n_side = hp.n_pos_side;
    const int np = in.n_patches;
    x.assign((size_t)np * E, 0.f);

    // patch embedding + learned position embeddings, in merged token order.
    // The row is `patch_K` wide (zero-padded past patch_elems), matching the
    // padded patch_w rows the device GEMM reads.
    std::vector<float> pv(patch_K, 0.f);
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
                        const float * w = patch_w.data() + (size_t)o * patch_K;
                        float acc = patch_b ? patch_b[o] : 0.f;
                        for (int k = 0; k < patch_K; k++) {
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

bool vision_model::runnable(std::string * why) const {
    auto fail = [&](const std::string & m) {
        if (why) {
            *why = m;
        }
        return false;
    };
    if (hp.n_layer <= 0 || hp.n_embd <= 0 || hp.n_head <= 0 || hp.n_ff <= 0 || hp.patch_size <= 0) {
        return fail("vision: incomplete tower geometry");
    }
    // The attention kernel is instantiated per head width.  qwen3vl uses 64/72;
    // qwen2vl's 80-wide head has its own correct BQ=16 instantiation.
    if (kind == model_profile::vision_kind::qwen2vl) {
        if (hp.head_dim != 80) {
            return fail("vision: qwen2vl implements only the 80-wide head (got " + std::to_string(hp.head_dim) + ")");
        }
        if (hp.patch_size != 14 || hp.merge != 2) {
            return fail("vision: qwen2vl implements only patch 14 / merge 2");
        }
    } else if (hp.head_dim != 64 && hp.head_dim != 72) {
        return fail("vision: unsupported head_dim " + std::to_string(hp.head_dim) + " (the encoder implements 64, 72 and 80)");
    }
    const int E = hp.n_embd;
    if ((int)layers.size() != hp.n_layer) {
        return fail("vision: missing block weights");
    }
    for (const vision_layer & L : layers) {
        if (!L.qkv.data || L.qkv.N != 3 * E || L.qkv.K != E || !L.out.data || !L.up.data || !L.down.data) {
            return fail("vision: a block is missing a weight or its qkv is not [3E][E]");
        }
    }
    if (mm0.K != hp.merge * hp.merge * E || mm2.K != mm0.N || mm2.N != hp.proj_dim) {
        return fail("vision: projector shape mismatch");
    }
    return true;
}

void vision_model::encode_host(const vision_input & in, std::vector<float> & out) const {
    // the qwen2vl q|k|v f32 matrix is deferred; the host reference forward reads it
    ensure_w32();
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

    const bool dbg_mm = si::env::flag("PF_MM_DIAG");
    auto fph = [&](const char * tag, const std::vector<float> & v, size_t n) {
        if (!dbg_mm) {
            return;
        }
        double sm = 0, sq = 0, mx = 0;
        for (size_t i = 0; i < n; i++) {
            sm += v[i];
            sq += (double) v[i] * v[i];
            mx = std::max(mx, std::fabs((double) v[i]));
        }
        fprintf(stderr, "[mm] host %-12s sum=%.5f sumsq=%.5f max=%.5f\n", tag, sm, sq, mx);
    };
    for (int il = 0; il < hp.n_layer; il++) {
        const vision_layer & L = layers[il];
        // ln1
        for (int t = 0; t < np; t++) {
            layer_norm(x.data() + (size_t)t * E, L.ln1, L.ln1_b, tmp.data() + (size_t)t * E, E, hp.eps);
        }
        if (il == 0) {
            fph("ln1", tmp, (size_t) np * E);
        }
        // fused qkv + rope on Q/K
        matmul_all(L.qkv, L.qkv_b, tmp.data(), qkv.data(), np);
        if (il == 0) {
            fph("qkv", qkv, (size_t) np * 3 * E);
        }
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
                    // four sections of HD/4 pairs each (so {16,16,16,16} at
                    // HD=64): the exponent restarts at each section boundary.
                    // Hardcoding 16 made this agree with vit_rope_launch only at
                    // HD=64 - every projector with a different head dim put the
                    // section split in a different place on the two sides, and
                    // the device looked wrong instead.
                    const int sec = ic / (HD / 4);
                    const int p = ic % (HD / 4);
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
        if (il == 0) {
            fph("qkv_rope", qkv, (size_t) np * 3 * E);
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
        if (il == 0) {
            fph("attn", attn, (size_t) np * E);
        }
        // attn_out projection + residual
        matmul_all(L.out, L.out_b, attn.data(), tmp.data(), np);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] += tmp[i];
        }
        if (il == 0) {
            fph("x_after_attn", x, (size_t) np * E);
        }
        // ln2 + ffn (up -> gelu -> down) + residual
        for (int t = 0; t < np; t++) {
            layer_norm(x.data() + (size_t)t * E, L.ln2, L.ln2_b, tmp.data() + (size_t)t * E, E, hp.eps);
        }
        if (il == 0) {
            fph("ln2", tmp, (size_t) np * E);
        }
        matmul_all(L.up, L.up_b, tmp.data(), upbuf.data(), np);
        for (int t = 0; t < np; t++) {
            if (kind == model_profile::vision_kind::qwen2vl) {
                float * v = upbuf.data() + (size_t)t * ff;
                for (int i = 0; i < ff; i++) {
                    v[i] = v[i] / (1.0f + std::exp(-1.702f * v[i]));
                }
            } else {
                gelu_tanh(upbuf.data() + (size_t)t * ff, ff);
            }
        }
        if (il == 0) {
            fph("ffn_up", upbuf, (size_t) np * ff);
        }
        matmul_all(L.down, L.down_b, upbuf.data(), downbuf.data(), np);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] += downbuf[i];
        }
        if (il == 0) {
            fph("x_after_ffn", x, (size_t) np * E);
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
        float * h = h1.data() + (size_t)m * mm0.N;
        if (kind == model_profile::vision_kind::qwen2vl) {
            for (int i = 0; i < mm0.N; i++) {
                h[i] = 0.5f * h[i] * (1.0f + std::erf(h[i] * 0.7071067811865476f));
            }
        } else {
            gelu_tanh(h, mm0.N);
        }
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
    pin_dev_queue(dev_q, q, "vision");
    const int E = hp.n_embd;
    const int ff = hp.n_ff;
    const int HD = hp.head_dim;
    const int NH = hp.n_head;
    const int M = hp.merge;
    const int np = in.n_patches;
    const int n_out = in.n_out;
    const int C = M * M;
    const int hidden = C * E;
    if (np <= 0 || n_out <= 0) {
        throw std::runtime_error("vision: empty patch grid");
    }
    if (np > kMaxImgPatches) {
        throw std::runtime_error("vision: image needs " + std::to_string(np) + " patch tokens (max "
                                 + std::to_string(kMaxImgPatches) + ")");
    }
    // ask runnable() rather than repeating HD != 64 here, so the query a caller
    // uses to decide "can I run this at all" and the throw cannot disagree
    if (!runnable()) {
        std::string why;
        runnable(&why);
        throw std::runtime_error(why);
    }

    if (!dev_weights) {
        upload(q);
    }
    if (!d_patch_w) {
        d_patch_w = sycl::malloc_device((size_t)E * patch_K * sizeof(float), q);
        if (!d_patch_w) {
            throw std::runtime_error("vision: patch weight allocation failed");
        }
        q.memcpy(d_patch_w, patch_w.data(), (size_t)E * patch_K * sizeof(float));
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
        d_patch_in = (float *)sycl::malloc_device((size_t)np * patch_K * 4, q);
        d_pos = (float *)sycl::malloc_device((size_t)np * E * 4, q);
        d_x = (float *)sycl::malloc_device((size_t)np * E * 4, q);
        d_ln = (float *)sycl::malloc_device((size_t)np * E * 4, q);
        d_qkv = (float *)sycl::malloc_device((size_t)np * 3 * E * 4, q);
        d_attn = (float *)sycl::malloc_device((size_t)np * E * 4, q);
        d_ffn = (float *)sycl::malloc_device((size_t)np * ff * 4, q);
        d_mm0 = (float *)sycl::malloc_device((size_t)n_out * hidden * 4, q);
        if (si::env::flag("PF_MM_DIAG")) {
            fprintf(stderr,
                    "[mm] scratch np=%d E=%d ff=%d hidden=%d n_out=%d patch_K=%d -> patch_in %.1f qkv %.1f "
                    "ffn %.1f mm0 %.1f MiB\n",
                    np, E, ff, hidden, n_out, patch_K, np * (double) patch_K * 4 / 1048576.0,
                    np * 3.0 * E * 4 / 1048576.0, np * (double) ff * 4 / 1048576.0,
                    n_out * (double) hidden * 4 / 1048576.0);
        }
        if (!d_patch_in || !d_pos || !d_x || !d_ln || !d_qkv || !d_attn || !d_ffn || !d_mm0) {
            throw std::runtime_error("vision: device scratch allocation failed");
        }
        scratch_patches = np;
    }

    auto devf = [&](const float * p) { return (const float *)dev_ptr(p); };

    vision_stage_timer vt(si::env::flag("PF_VISION_TIME"));
    const auto t_all0 = std::chrono::high_resolution_clock::now();

    std::vector<float> pin, pos;
    build_raw_patches(in, pin);
    q.memcpy(d_patch_in, pin.data(), pin.size() * sizeof(float));
    const bool has_pos = pos_embd != nullptr;
    if (has_pos) {
        build_pos_emb(in, pos);
        q.memcpy(d_pos, pos.data(), pos.size() * sizeof(float));
    }

    // patch embedding: out[N=E, K=patch_K], then patch bias + position embedding
    vit_gemm_launch(q, d_patch_w, GGML_TYPE_F32, E, patch_K, d_patch_in, patch_K, d_x, E, np, 1.f, nullptr);
    if (patch_b) {
        vit_add_bias_launch(q, d_x, E, devf(patch_b), np, E);
    }
    if (has_pos) {
        vit_add_launch(q, d_x, E, d_pos, E, np, E);
    }
    vt.lap(q, vt.patch);

    const float scale = 1.0f / std::sqrt((float)HD);
    const bool dbg_mm = si::env::flag("PF_MM_DIAG");
    auto fpd = [&](const char * tag, const void * p, size_t n) {
        if (!dbg_mm) {
            return;
        }
        std::vector<float> h(n);
        q.memcpy(h.data(), p, n * 4).wait();
        double sm = 0, sq = 0, mx = 0;
        for (float v : h) {
            sm += v;
            sq += (double) v * v;
            mx = std::max(mx, std::fabs((double) v));
        }
        fprintf(stderr, "[mm] dev %-12s sum=%.5f sumsq=%.5f max=%.5f\n", tag, sm, sq, mx);
    };
    for (int il = 0; il < hp.n_layer; il++) {
        const vision_layer & L = layers[il];
        vit_layernorm_launch(q, d_x, E, devf(L.ln1), devf(L.ln1_b), d_ln, E, np, E, hp.eps);
        vt.lap_layer(q, vt.ln1);
        if (il == 0) {
            fpd("ln1", d_ln, (size_t) np * E);
        }
        if (!linear_dnnl(q, L.qkv.data, d_ln, E, np, E, d_qkv, 3 * E, nullptr, 1.f, 3 * E)) {
            vit_gemm_launch(q, dev_ptr(L.qkv.data), L.qkv.type, 3 * E, E, d_ln, E, d_qkv, 3 * E, np, 1.f,
                            nullptr);
        }
        if (L.qkv_b) {
            vit_add_bias_launch(q, d_qkv, 3 * E, devf(L.qkv_b), np, 3 * E);
        }
        vt.lap_layer(q, vt.qkv);
        if (il == 0) {
            fpd("qkv", d_qkv, (size_t) np * 3 * E);
        }
        vit_rope_launch(q, d_qkv, 3 * E, np, NH, HD, in.out_w, M, hp.rope_base);
        vt.lap_layer(q, vt.rope);
        if (il == 0) {
            fpd("qkv_rope", d_qkv, (size_t) np * 3 * E);
        }
        vit_attn_launch(q, d_qkv, 3 * E, d_attn, E, np, NH, HD, scale);
        vt.lap_layer(q, vt.attn);
        if (il == 0) {
            fpd("attn", d_attn, (size_t) np * E);
        }
        // x += out_b, then x += attn_out_proj (residual through the GEMM)
        if (L.out_b) {
            vit_add_bias_launch(q, d_x, E, devf(L.out_b), np, E);
        }
        if (!linear_dnnl(q, L.out.data, d_attn, E, np, E, d_x, E, d_x, 1.f, E)) {
            vit_gemm_launch(q, dev_ptr(L.out.data), L.out.type, E, E, d_attn, E, d_x, E, np, 1.f, d_x);
        }
        vt.lap_layer(q, vt.out);
        if (il == 0) {
            fpd("x_after_attn", d_x, (size_t) np * E);
        }
        vit_layernorm_launch(q, d_x, E, devf(L.ln2), devf(L.ln2_b), d_ln, E, np, E, hp.eps);
        vt.lap_layer(q, vt.ln2);
        if (il == 0) {
            fpd("ln2", d_ln, (size_t) np * E);
        }
        if (!linear_dnnl(q, L.up.data, d_ln, E, np, E, d_ffn, ff, nullptr, 1.f, ff)) {
            vit_gemm_launch(q, dev_ptr(L.up.data), L.up.type, ff, E, d_ln, E, d_ffn, ff, np, 1.f, nullptr);
        }
        if (L.up_b) {
            vit_add_bias_launch(q, d_ffn, ff, devf(L.up_b), np, ff);
        }
        vt.lap_layer(q, vt.up);
        if (kind == model_profile::vision_kind::qwen2vl) {
            vit_quick_gelu_launch(q, d_ffn, np * ff);
        } else {
            vit_gelu_launch(q, d_ffn, np * ff);
        }
        vt.lap_layer(q, vt.act);
        if (il == 0) {
            fpd("ffn_up", d_ffn, (size_t) np * ff);
        }
        if (L.down_b) {
            vit_add_bias_launch(q, d_x, E, devf(L.down_b), np, E);
        }
        if (!linear_dnnl(q, L.down.data, d_ffn, ff, np, ff, d_x, E, d_x, 1.f, E)) {
            vit_gemm_launch(q, dev_ptr(L.down.data), L.down.type, E, ff, d_ffn, ff, d_x, E, np, 1.f, d_x);
        }
        vt.lap_layer(q, vt.down);
        if (il == 0) {
            fpd("x_after_ffn", d_x, (size_t) np * E);
        }
    }

    vit_layernorm_launch(q, d_x, E, devf(post_ln), post_ln_b ? devf(post_ln_b) : nullptr, d_ln, E, np, E, hp.eps);
    vt.lap(q, vt.post_ln);

    // merger: the 2x2 concat is a stride reinterpretation of d_ln
    if (!linear_dnnl(q, mm0.data, d_ln, hidden, n_out, hidden, d_mm0, mm0.N, nullptr, 1.f, mm0.N)) {
        vit_gemm_launch(q, dev_ptr(mm0.data), mm0.type, mm0.N, hidden, d_ln, hidden, d_mm0, mm0.N, n_out, 1.f,
                        nullptr);
    }
    if (mm0_b) {
        vit_add_bias_launch(q, d_mm0, mm0.N, devf(mm0_b), n_out, mm0.N);
    }
    if (kind == model_profile::vision_kind::qwen2vl) {
        vit_gelu_errf_launch(q, d_mm0, n_out * mm0.N);
    } else {
        vit_gelu_launch(q, d_mm0, n_out * mm0.N);
    }
    if (!linear_dnnl(q, mm2.data, d_mm0, mm0.N, n_out, mm0.N, d_out, hp.proj_dim, nullptr, 1.f, hp.proj_dim)) {
        vit_gemm_launch(q, dev_ptr(mm2.data), mm2.type, hp.proj_dim, mm0.N, d_mm0, mm0.N, d_out, hp.proj_dim, n_out,
                        1.f, nullptr);
    }
    if (mm2_b) {
        vit_add_bias_launch(q, d_out, hp.proj_dim, devf(mm2_b), n_out, hp.proj_dim);
    }
    vt.lap(q, vt.mm);
    q.wait();
    if (vt.on) {
        const double total =
            std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_all0).count();
        fprintf(stderr,
                "[vtime] np=%d n_out=%d L=%d: total=%.1f ms | patch=%.1f merger=%.1f post_ln=%.2f | "
                "layers=%.1f (ln1=%.1f qkv=%.1f rope=%.1f attn=%.1f out=%.1f ln2=%.1f up=%.1f act=%.1f down=%.1f)\n",
                np, n_out, hp.n_layer, total, vt.patch, vt.mm, vt.post_ln, vt.layers, vt.ln1, vt.qkv, vt.rope,
                vt.attn, vt.out, vt.ln2, vt.up, vt.act, vt.down);
    }
}

} // namespace si
