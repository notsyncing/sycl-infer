#include "multimodal.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "kernels.h"
#include "tokenizer.h"
#include "vision.h"

namespace si {

namespace {

// Expand the image placeholders into merged tokens and assign the interleaved
// M-RoPE positions (section 0 = temporal, 1 = row, 2 = col).  An image of
// nx*ny merged tokens consumes max(nx, ny) positions.
mm_prompt build_meta(const tokenizer & tk, const std::string & rendered, const std::vector<vision_input> & vin,
                     const std::vector<int> & offset, int total_rows) {
    auto it = tk.token_to_id.find("<|image_pad|>");
    if (it == tk.token_to_id.end()) {
        throw std::runtime_error("mm: tokenizer has no <|image_pad|> token");
    }
    const int pad = it->second;
    const int n_img = (int)vin.size();

    const std::vector<int> toks = tk.encode(rendered, /*parse_special=*/true);
    if (std::count(toks.begin(), toks.end(), pad) != n_img) {
        throw std::runtime_error("mm: image placeholder count does not match image count");
    }

    mm_prompt p;
    p.n_img = n_img;
    std::vector<int32_t> pt, ph, pw;
    p.tokens.reserve(toks.size() + total_rows);
    p.img_row.reserve(toks.size() + total_rows);
    pt.reserve(toks.size() + total_rows);
    ph.reserve(toks.size() + total_rows);
    pw.reserve(toks.size() + total_rows);
    int pos = 0;
    int k = 0;
    for (size_t i = 0; i < toks.size(); i++) {
        if (toks[i] == pad && k < n_img) {
            const int nx = vin[k].out_w;
            const int base = pos;
            for (int t = 0; t < vin[k].n_out; t++) {
                p.tokens.push_back(pad);
                p.img_row.push_back(offset[k] + t);
                pt.push_back(base);
                ph.push_back(base + t / nx);
                pw.push_back(base + t % nx);
            }
            pos = base + std::max(vin[k].out_w, vin[k].out_h);
            k++;
        } else {
            p.tokens.push_back(toks[i]);
            p.img_row.push_back(-1);
            pt.push_back(pos);
            ph.push_back(pos);
            pw.push_back(pos);
            pos++;
        }
    }
    const int n = (int)p.tokens.size();
    p.mrope.resize((size_t)4 * n);
    for (int i = 0; i < n; i++) {
        p.mrope[(size_t)i] = pt[i];
        p.mrope[(size_t)n + i] = ph[i];
        p.mrope[(size_t)2 * n + i] = pw[i];
        p.mrope[(size_t)3 * n + i] = pt[i]; // last section is unused (0 pairs)
    }
    p.pos_after = pos;
    return p;
}

std::vector<vision_input> make_inputs(const vision_model & vm, const std::vector<mm_image> & images,
                                      std::vector<int> & offset, int & total_rows) {
    const int n_img = (int)images.size();
    std::vector<vision_input> vin(n_img);
    offset.assign(n_img, 0);
    total_rows = 0;
    for (int k = 0; k < n_img; k++) {
        vin[k] = vision_model::make_input(vm, images[k]);
        if (vin[k].n_out <= 0) {
            throw std::runtime_error("mm: image too small for the patch grid");
        }
        if (total_rows + vin[k].n_out > kMaxImgTokens) {
            throw std::runtime_error("mm: image tokens exceed the kMaxImgTokens budget");
        }
        offset[k] = total_rows;
        total_rows += vin[k].n_out;
    }
    return vin;
}

} // namespace

mm_prompt mm_build_prompt(const tokenizer & tk, const std::string & rendered, const std::vector<mm_image> & images,
                          const vision_model & vm, int n_embd) {
    std::vector<int> offset;
    int total_rows = 0;
    const std::vector<vision_input> vin = make_inputs(vm, images, offset, total_rows);

    mm_prompt p = build_meta(tk, rendered, vin, offset, total_rows);
    p.embd.resize((size_t)total_rows * n_embd);
    for (size_t k = 0; k < images.size(); k++) {
        std::vector<float> e;
        vm.encode_host(vin[k], e);
        std::memcpy(p.embd.data() + (size_t)offset[k] * n_embd, e.data(), e.size() * sizeof(float));
    }
    return p;
}

mm_prompt mm_build_prompt_device(vision_model & vm, sycl::queue & q, const tokenizer & tk, const std::string & rendered,
                                 const std::vector<mm_image> & images, int n_embd, float * d_out) {
    std::vector<int> offset;
    int total_rows = 0;
    const std::vector<vision_input> vin = make_inputs(vm, images, offset, total_rows);

    for (size_t k = 0; k < images.size(); k++) {
        vm.encode_device(q, vin[k], d_out + (size_t)offset[k] * n_embd);
    }

    mm_prompt p = build_meta(tk, rendered, vin, offset, total_rows);
    p.d_embd = d_out;
    return p;
}

} // namespace si
