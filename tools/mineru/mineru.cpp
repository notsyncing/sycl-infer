#include "mineru.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include "engine/engine.h"
#include "kernels.h"
#include "layout.h"
#include "mm/image.h"
#include "mm/multimodal.h"
#include "mm/vision.h"

namespace si {

namespace {

const char * kSystemPrompt = "You are a helpful assistant.";

// ChatML as rendered by the model's own template: image first, then the text
// instruction (mineru-vl-utils' default `text_before_image=False`).
std::string render_prompt(const std::string & instruction) {
    return std::string("<|im_start|>system\n") + kSystemPrompt + "<|im_end|>\n" +
           "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + instruction +
           "<|im_end|>\n<|im_start|>assistant\n";
}

void sampling_for(const std::string & type, gen_params & gp) {
    gp.temperature = 0.f;
    gp.top_k = 1;
    gp.top_p = 1.f;
    gp.repeat_penalty = 1.f;
    gp.repeat_last_n = 512; // llama.cpp backend's repeat window
    gp.presence_penalty = 0.f;
    gp.frequency_penalty = 0.f;
    if (type == "[layout]") {
        return;
    }
    gp.presence_penalty = 1.0f;
    gp.frequency_penalty = type == "table" ? 0.005f : 0.05f;
}

// Crop a rectangle out of an interleaved RGB8 image.
std::vector<uint8_t> crop_rgb(const std::vector<uint8_t> & src, int w, int h, int x0, int y0, int x1, int y1) {
    x0 = std::max(0, std::min(x0, w - 1));
    y0 = std::max(0, std::min(y0, h - 1));
    x1 = std::max(x0 + 1, std::min(x1, w));
    y1 = std::max(y0 + 1, std::min(y1, h));
    const int cw = x1 - x0, ch = y1 - y0;
    std::vector<uint8_t> out((size_t)cw * ch * 3);
    for (int y = 0; y < ch; y++) {
        std::memcpy(out.data() + (size_t)y * cw * 3, src.data() + ((size_t)(y0 + y) * w + x0) * 3,
                    (size_t)cw * 3);
    }
    return out;
}

// PIL rotate(angle, expand=True): the angles are multiples of 90 and rotate
// counter-clockwise.
std::vector<uint8_t> rotate_rgb(const std::vector<uint8_t> & src, int w, int h, int angle, int & ow, int & oh) {
    angle = ((angle % 360) + 360) % 360;
    if (angle == 0) {
        ow = w;
        oh = h;
        return src;
    }
    const bool swap = angle == 90 || angle == 270;
    ow = swap ? h : w;
    oh = swap ? w : h;
    std::vector<uint8_t> out((size_t)ow * oh * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int nx = 0, ny = 0;
            if (angle == 90) {
                nx = y;
                ny = w - 1 - x;
            } else if (angle == 180) {
                nx = w - 1 - x;
                ny = h - 1 - y;
            } else {
                nx = h - 1 - y;
                ny = x;
            }
            std::memcpy(out.data() + ((size_t)ny * ow + nx) * 3, src.data() + ((size_t)y * w + x) * 3, 3);
        }
    }
    return out;
}

// mineru-vl-utils' resize_by_need: pad an extreme aspect ratio to 1:50 with
// white, then upscale anything whose short edge is below 28 px.
std::vector<uint8_t> resize_by_need(const std::vector<uint8_t> & src, int w, int h, int & ow, int & oh) {
    std::vector<uint8_t> cur = src;
    int cw = w, ch = h;
    const double ratio = (double)std::max(cw, ch) / (double)std::max(1, std::min(cw, ch));
    if (ratio > 50.0) {
        int nw = cw, nh = ch;
        if (cw > ch) {
            nh = (int)std::ceil(cw / 50.0);
        } else {
            nw = (int)std::ceil(ch / 50.0);
        }
        std::vector<uint8_t> padded((size_t)nw * nh * 3, 255);
        const int ox = (nw - cw) / 2, oy = (nh - ch) / 2;
        for (int y = 0; y < ch; y++) {
            std::memcpy(padded.data() + ((size_t)(oy + y) * nw + ox) * 3, cur.data() + (size_t)y * cw * 3,
                        (size_t)cw * 3);
        }
        cur.swap(padded);
        cw = nw;
        ch = nh;
    }
    if (std::min(cw, ch) < 28) {
        const double s = 28.0 / std::min(cw, ch);
        const int nw = (int)std::ceil(cw * s), nh = (int)std::ceil(ch * s);
        std::vector<uint8_t> up((size_t)nw * nh * 3);
        mm_image_resize_rgb(cur.data(), cw, ch, up.data(), nw, nh);
        cur.swap(up);
        cw = nw;
        ch = nh;
    }
    ow = cw;
    oh = ch;
    return cur;
}

struct model_bundle {
    vision_model vm;
    float * d_out = nullptr;
    sycl::queue * q = nullptr;
};

std::string ask(engine & e, model_bundle & mb, const std::vector<uint8_t> & rgb, int w, int h,
                const std::string & instruction, const std::string & type, int max_tokens, bool debug) {
    const bool mtime = si::env::flag("PF_MINERU_TIME");
    auto now_ms = [] {
        return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    };
    const double t_ask = mtime ? now_ms() : 0;
    image_preproc_cfg cfg;
    cfg.patch_size = mb.vm.hp.patch_size;
    cfg.merge = mb.vm.hp.merge;
    cfg.min_pixels = 50176;      // preprocessor_config.json
    cfg.max_pixels = 1605632;
    for (int c = 0; c < 3; c++) {
        cfg.mean[c] = mb.vm.hp.mean[c];
        cfg.std[c] = mb.vm.hp.std[c];
    }
    const mm_image img = mm_image_preprocess(rgb.data(), w, h, cfg);
    const vision_input vi = vision_model::make_input(mb.vm, img);
    const int ow = vi.out_w, oh = vi.out_h, n_out = vi.n_out;
    if (n_out <= 0 || n_out > kMaxImgTokens) {
        throw std::runtime_error("mineru: image token count outside the embedding budget");
    }
    // PF_MINERU_TIME: per-image/per-block wall clock.  "prep" is the host
    // preprocessing (crop/resize/normalize) that precedes the encoder; vision is
    // the encoder call itself; "prefill" runs to the first sampled token (the
    // text prompt forward plus the head) and "decode" is everything after it.
    const double t_start = mtime ? now_ms() : 0;
    mb.vm.encode_device(*mb.q, vi, mb.d_out);
    const double t_vision = mtime ? now_ms() : 0;
    if (si::env::flag("PF_MINERU_ZERO_EMBD")) {
        mb.q->memset(mb.d_out, 0, (size_t)n_out * (size_t)e.m.hp.n_embd * sizeof(float)).wait();
    }
    if (const char * dp = si::env::str("PF_MINERU_DUMP_EMBD")) {
        std::vector<float> emb((size_t)n_out * (size_t)e.m.hp.n_embd);
        mb.q->memcpy(emb.data(), mb.d_out, emb.size() * sizeof(float)).wait();
        if (FILE * fp = std::fopen(dp, "wb")) {
            const int32_t hdr[2] = {n_out, e.m.hp.n_embd};
            std::fwrite(hdr, sizeof(hdr), 1, fp);
            std::fwrite(emb.data(), sizeof(float), emb.size(), fp);
            std::fclose(fp);
        }
    }
    if (debug) {
        fprintf(stderr, "[mineru] %s image %dx%d -> %dx%d (%d tokens)\n", type.c_str(), img.width, img.height, ow, oh,
                n_out);
    }

    const std::string rendered = render_prompt(instruction);
    mm_block b;
    b.pad_tok = e.tk.token_to_id.at("<|image_pad|>");
    b.kind = MM_KIND_IMAGE;
    b.out_w = ow;
    b.out_h = oh;
    b.n_tok = n_out;
    b.n_pos = std::max(ow, oh);
    mm_prompt p = mm_expand_prompt(e.tk, rendered, {b}, n_out);
    if (debug) {
        const int n = (int)p.tokens.size();
        int first = -1;
        for (int i = 0; i < n; i++) {
            if (p.img_row[i] >= 0) {
                first = i;
                break;
            }
        }
        if (first >= 0) {
            fprintf(stderr, "[mineru] prompt %d tokens, first image token %d, mrope t/h/w = %d/%d/%d, "
                            "next text t=%d\n",
                    n, first, p.mrope[first], p.mrope[n + first], p.mrope[2 * n + first],
                    p.mrope[3 * n + first]);
        }
    }
    if (si::env::flag("PF_MINERU_NOMROPE")) {
        // A/B: plain 1-D positions (all three sections equal).  If this fixes a
        // garbage layout then the M-RoPE section mapping, not the vision tower,
        // is what is wrong.
        const int n = (int)p.tokens.size();
        for (int i = 0; i < n; i++) {
            p.mrope[(size_t)n + i] = p.mrope[(size_t)i];
            p.mrope[(size_t)2 * n + i] = p.mrope[(size_t)i];
            p.mrope[(size_t)3 * n + i] = p.mrope[(size_t)i];
        }
    }
    p.d_embd = mb.d_out;
    p.embd.clear();

    gen_params gp;
    gp.max_tokens = max_tokens;
    gp.seed = 0;
    sampling_for(type, gp);
    std::vector<int> ids;
    double t_first = 0;
    e.generate_mm(p, gp, [&](int tok) {
        if (mtime && t_first == 0) {
            t_first = now_ms();
        }
        ids.push_back(tok);
        return true;
    });
    if (mtime) {
        const double t_end = now_ms();
        const double prefill = t_first > 0 ? t_first - t_vision : t_end - t_vision;
        const double decode = t_first > 0 ? t_end - t_first : 0;
        fprintf(stderr,
                "[mineru-time] %-8s img=%dx%d n_out=%d prompt=%d gen=%zu | prep=%.1f vision=%.1f prefill=%.1f "
                "decode=%.1f total=%.1f ms\n",
                type.c_str(), img.width, img.height, n_out, (int)p.tokens.size(), ids.size(),
                t_start - t_ask, t_vision - t_start, prefill, decode, t_end - t_ask);
    }
    return e.tk.decode(ids);
}

} // namespace

int run_mineru(engine & e, const std::string & mmproj_path, const std::string & image_path,
               const mineru_options & opt, FILE * out) {
    std::vector<uint8_t> rgb;
    int w = 0, h = 0;
    std::string err;
    if (!mm_image_decode_file(image_path, rgb, w, h, &err)) {
        fprintf(stderr, "error: cannot decode image '%s': %s\n", image_path.c_str(), err.c_str());
        return 1;
    }
    if (!e.m.prof || e.m.prof->vision != model_profile::vision_kind::qwen2vl) {
        fprintf(stderr, "error: mineru needs the qwen2vl architecture (profile vision tower), got '%s'\n",
                e.m.prof ? e.m.prof->key : "no profile");
        return 1;
    }

    model_bundle mb;
    mb.q = &e.q;
    const model_profile::vision_kind vkind = e.m.prof->vision;
    const bool mtime = si::env::flag("PF_MINERU_TIME");
    const double t_load0 = mtime ? std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now().time_since_epoch())
                                       .count()
                                 : 0;
    mb.vm.load(mmproj_path, vkind);
    std::string why;
    if (!mb.vm.runnable(&why)) {
        fprintf(stderr, "error: mmproj loaded but cannot run: %s\n", why.c_str());
        return 1;
    }
    if (mb.vm.hp.proj_dim != e.m.hp.n_embd) {
        fprintf(stderr, "error: vision projector width %d != text n_embd %d\n", mb.vm.hp.proj_dim, e.m.hp.n_embd);
        return 1;
    }
    mb.vm.upload(e.q);
    if (mtime) {
        const double t1 = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now().time_since_epoch())
                              .count();
        fprintf(stderr, "[mineru-time] mmproj load+upload = %.0f ms (vision dnnl %s)\n", t1 - t_load0,
                mb.vm.gemm ? "on" : "off");
    }
    mb.d_out = sycl::malloc_device<float>((size_t)kMaxImgTokens * (size_t)e.m.hp.n_embd, e.q);
    if (!mb.d_out) {
        fprintf(stderr, "error: cannot allocate the image embedding buffer\n");
        return 1;
    }

    int rc = 0;
    try {
        // Debug self-check: the whole vision tower on a tiny crop, host vs
        // device.  A 56x56 crop is 16 patches / 4 merged tokens, so the host
        // reference is instant; a mismatch here means the tower (not the text
        // trunk) is why the layout comes back as numbers.
        if (opt.debug) {
            const char * ps = si::env::str("PF_MINERU_PROBE");
            const int want = ps ? atoi(ps) : 56;
            const int cw = std::min(want > 0 ? want : 56, w), chh = std::min(want > 0 ? want : 56, h);
            std::vector<uint8_t> probe = crop_rgb(rgb, w, h, (w - cw) / 2, (h - chh) / 2, (w - cw) / 2 + cw,
                                                 (h - chh) / 2 + chh);
            image_preproc_cfg cfg;
            cfg.patch_size = mb.vm.hp.patch_size;
            cfg.merge = mb.vm.hp.merge;
            cfg.min_pixels = 0;
            cfg.max_pixels = 0;
            for (int c = 0; c < 3; c++) {
                cfg.mean[c] = mb.vm.hp.mean[c];
                cfg.std[c] = mb.vm.hp.std[c];
            }
            const mm_image pim = mm_image_preprocess(probe.data(), cw, chh, cfg);
            const vision_input pvi = vision_model::make_input(mb.vm, pim);
            std::vector<float> hout;
            mb.vm.encode_host(pvi, hout);
            mb.vm.encode_device(e.q, pvi, mb.d_out);
            const int n = (int)hout.size();
            const int nd = mb.vm.hp.proj_dim > 0 ? n / mb.vm.hp.proj_dim : 0;
            std::vector<float> dout((size_t)n);
            e.q.memcpy(dout.data(), mb.d_out, dout.size() * sizeof(float)).wait();
            double md = 0, hmax = 0;
            int bad = 0;
            for (int i = 0; i < n; i++) {
                md = std::max(md, (double)std::fabs(hout[i] - dout[i]));
                hmax = std::max(hmax, (double)std::fabs(hout[i]));
                if (!std::isfinite(dout[i])) {
                    bad++;
                }
            }
            double dot = 0, hn = 0, dn = 0;
            for (int i = 0; i < n; i++) {
                dot += (double)hout[i] * dout[i];
                hn += (double)hout[i] * hout[i];
                dn += (double)dout[i] * dout[i];
            }
            fprintf(stderr,
                    "[mineru] vision self-check: %dx%d patches -> %d merged, host|max|=%.4g, max|diff|=%.4g, "
                    "nonfinite=%d, cos=%.6f, rms ratio=%.4f\n",
                    pim.width, pim.height, nd, hmax, md, bad, hn > 0 && dn > 0 ? dot / std::sqrt(hn * dn) : 0.0,
                    hn > 0 ? std::sqrt(dn / hn) : 0.0);
            fprintf(stderr, "[mineru] h[0..7]=");
            for (int i = 0; i < 8 && i < n; i++) {
                fprintf(stderr, " %.5g", hout[i]);
            }
            fprintf(stderr, "\n[mineru] d[0..7]=");
            for (int i = 0; i < 8 && i < n; i++) {
                fprintf(stderr, " %.5g", dout[i]);
            }
            fprintf(stderr, "\n");
        }

        // ---- step 1: layout on the squashed 1036x1036 page ----
        const int side = opt.layout_side;
        std::vector<uint8_t> page((size_t)side * side * 3);
        mm_image_resize_rgb(rgb.data(), w, h, page.data(), side, side);
        const std::string layout_raw = ask(e, mb, page, side, side, layout_prompt(), "[layout]", opt.max_tokens, opt.debug);
        if (opt.debug) {
            fprintf(stderr, "[mineru] layout raw:\n%s\n", layout_raw.c_str());
        }
        std::vector<mineru_block> blocks = mineru_parse_layout(layout_raw);

        // ---- step 2: content per block ----
        const char * skip_types[] = {"list", "equation_block", "image_block"};
        for (mineru_block & b : blocks) {
            bool skip = false;
            for (const char * s : skip_types) {
                if (b.type == s) {
                    skip = true;
                }
            }
            if (!opt.only_type.empty() && b.type != opt.only_type) {
                skip = true;
            }
            if (skip) {
                continue;
            }
            const int x0 = (int)std::floor(b.bbox[0] * w), y0 = (int)std::floor(b.bbox[1] * h);
            const int x1 = (int)std::ceil(b.bbox[2] * w), y1 = (int)std::ceil(b.bbox[3] * h);
            const int cx0 = std::max(0, std::min(x0, w - 1));
            const int cy0 = std::max(0, std::min(y0, h - 1));
            const int cx1 = std::max(cx0 + 1, std::min(x1, w));
            const int cy1 = std::max(cy0 + 1, std::min(y1, h));
            std::vector<uint8_t> crop = crop_rgb(rgb, w, h, cx0, cy0, cx1, cy1);
            int cw = cx1 - cx0;
            int ch = cy1 - cy0;
            if (b.angle != 0) {
                crop = rotate_rgb(crop, cw, ch, b.angle, cw, ch);
            }
            crop = resize_by_need(crop, cw, ch, cw, ch);
            b.content = ask(e, mb, crop, cw, ch, content_prompt(b.type), b.type, opt.max_tokens, opt.debug);
            if (opt.debug) {
                fprintf(stderr, "[mineru] %s %.3f %.3f %.3f %.3f -> %zu bytes\n", b.type.c_str(), b.bbox[0],
                        b.bbox[1], b.bbox[2], b.bbox[3], b.content.size());
            }
        }

        // ---- JSON ----
        auto esc = [](const std::string & s) {
            std::string o;
            for (char c : s) {
                switch (c) {
                case '"': o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\n': o += "\\n"; break;
                case '\r': o += "\\r"; break;
                case '\t': o += "\\t"; break;
                default:
                    if ((unsigned char)c < 0x20) {
                        char buf[8];
                        snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                        o += buf;
                    } else {
                        o += c;
                    }
                }
            }
            return o;
        };
        fputs("[\n", out);
        for (size_t i = 0; i < blocks.size(); i++) {
            const mineru_block & b = blocks[i];
            fprintf(out,
                    "  {\"type\":\"%s\",\"bbox\":[%.6f,%.6f,%.6f,%.6f],\"angle\":%d,\"merge_prev\":%s,"
                    "\"content\":\"%s\"}%s\n",
                    b.type.c_str(), b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3], b.angle,
                    b.merge_prev ? "true" : "false", esc(b.content).c_str(), i + 1 < blocks.size() ? "," : "");
        }
        fputs("]\n", out);
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: mineru: %s\n", ex.what());
        rc = 1;
    }
    sycl::free(mb.d_out, e.q);
    return rc;
}

} // namespace si
