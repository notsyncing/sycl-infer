#include "multimodal.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>

#include "audio_model.h"
#include "kernels.h"
#include "tokenizer.h"
#include "video.h"

namespace si {

namespace {

// The pad token of each kind; the tokenizer must know it (the model's vocab
// has all three, otherwise every multimodal request fails with a clear error).
const char * kPadName[3] = {"<|image_pad|>", "<|video_pad|>", "<|audio_pad|>"};

int pad_tok_of(const tokenizer & tk, int kind) {
    auto it = tk.token_to_id.find(kPadName[kind]);
    if (it == tk.token_to_id.end()) {
        throw std::runtime_error(std::string("mm: tokenizer has no ") + kPadName[kind] + " token");
    }
    return it->second;
}

} // namespace

// ---------------------------------------------------------------------------
// Generic expansion.  `rendered` must contain exactly as many occurrences of
// each pad token as the blocks carrying it, in the order the pads appear.
// Position layout (Qwen2.5-VL get_vision_position_ids, interval = 1):
//   token (frame i, row j, col k) -> (t,row,col) = (base+i, base+j, base+k),
//   the block consumes max(out_w, out_h) text positions.
// Audio blocks follow TM-RoPE: temporal advances one position per embedding
// while the row/col sections stay at 0, and the block consumes n_tok positions.
// ---------------------------------------------------------------------------
mm_prompt mm_expand_prompt(const tokenizer & tk, const std::string & rendered, const std::vector<mm_block> & blocks,
                           int total_rows) {
    // how many blocks carry each pad token
    std::map<int, int> want;
    for (const mm_block & b : blocks) {
        want[b.pad_tok]++;
    }

    const std::vector<int> toks = tk.encode(rendered, /*parse_special=*/true);
    std::map<int, int> have;
    for (int t : toks) {
        if (want.count(t)) {
            have[t]++;
        }
    }
    for (const auto & pr : want) {
        if (have[pr.first] != pr.second) {
            throw std::runtime_error("mm: <pad> placeholder count does not match block count");
        }
    }

    mm_prompt p;
    p.n_img = (int)blocks.size();
    // embedding row offset of each block (blocks fill the caller's embd buffer)
    std::vector<int> off(blocks.size());
    int rows = 0;
    for (size_t k = 0; k < blocks.size(); k++) {
        off[k] = rows;
        rows += blocks[k].n_tok;
    }
    if (rows != total_rows) {
        throw std::runtime_error("mm: block token count does not match the filled rows");
    }

    const int n_res = (int)toks.size() + total_rows;
    std::vector<int32_t> pt, ph, pw;
    p.tokens.reserve(n_res);
    p.img_row.reserve(n_res);
    pt.reserve(n_res);
    ph.reserve(n_res);
    pw.reserve(n_res);

    std::map<int, int> used; // next block index for each pad token
    int pos = 0, k = 0;
    for (int t : toks) {
        if (want.count(t)) {
            const int bi = used[t]++;
            const mm_block & b = blocks[bi];
            const int base = pos;
            const int gwh = b.out_w * b.out_h;
            for (int i = 0; i < b.n_tok; i++) {
                p.tokens.push_back(t);
                p.img_row.push_back(off[bi] + i);
                if (b.kind == MM_KIND_AUDIO) {
                    pt.push_back(base + i); // temporal advances per embedding
                    ph.push_back(0);
                    pw.push_back(0);
                } else if (b.kind == MM_KIND_VIDEO) {
                    pt.push_back(base + i / gwh);
                    ph.push_back(base + (i % gwh) / b.out_w);
                    pw.push_back(base + (i % gwh) % b.out_w);
                } else { // image: every token shares the temporal position
                    pt.push_back(base);
                    ph.push_back(base + i / b.out_w);
                    pw.push_back(base + i % b.out_w);
                }
            }
            k++;
            pos = base + b.n_pos;
        } else {
            p.tokens.push_back(t);
            p.img_row.push_back(-1);
            pt.push_back(pos);
            ph.push_back(pos);
            pw.push_back(pos);
            pos++;
        }
    }
    if (k != (int)blocks.size()) {
        throw std::runtime_error("mm: not all blocks consumed by the prompt");
    }

    const int n = (int)p.tokens.size();
    p.mrope.resize((size_t)4 * n);
    for (int i = 0; i < n; i++) {
        p.mrope[(size_t)i] = pt[i];
        p.mrope[(size_t)n + i] = ph[i];
        p.mrope[(size_t)2 * n + i] = pw[i];
        p.mrope[(size_t)3 * n + i] = pt[i]; // last section unused (0 pairs)
    }
    p.pos_after = pos;
    return p;
}

// ---------------------------------------------------------------------------
// Images.
// ---------------------------------------------------------------------------
static mm_prompt build_image_prompt(const tokenizer & tk, const std::string & rendered,
                                    const std::vector<mm_image> & images, const vision_model & vm, int n_embd) {
    const int n_img = (int)images.size();
    std::vector<mm_block> blocks(n_img);
    std::vector<vision_input> vin(n_img);
    std::vector<int> off(n_img);
    int total_rows = 0;
    for (int k = 0; k < n_img; k++) {
        vin[k] = vision_model::make_input(vm, images[k]);
        if (vin[k].n_out <= 0) {
            throw std::runtime_error("mm: image too small for the patch grid");
        }
        mm_block & b = blocks[k];
        b.pad_tok = pad_tok_of(tk, MM_KIND_IMAGE);
        b.kind = MM_KIND_IMAGE;
        b.out_w = vin[k].out_w;
        b.out_h = vin[k].out_h;
        b.n_tok = vin[k].n_out;
        b.n_pos = std::max(vin[k].out_w, vin[k].out_h);
        off[k] = total_rows;
        total_rows += vin[k].n_out;
    }
    if (total_rows > kMaxImgTokens) {
        throw std::runtime_error("mm: image tokens exceed the kMaxImgTokens budget");
    }

    mm_prompt p = mm_expand_prompt(tk, rendered, blocks, total_rows);
    p.embd.resize((size_t)total_rows * n_embd);
    std::vector<float> e;
    for (int k = 0; k < n_img; k++) {
        vm.encode_host(vin[k], e);
        std::memcpy(p.embd.data() + (size_t)off[k] * n_embd, e.data(), e.size() * sizeof(float));
    }
    return p;
}

mm_prompt mm_build_prompt(const tokenizer & tk, const std::string & rendered, const std::vector<mm_image> & images,
                          const vision_model & vm, int n_embd) {
    return build_image_prompt(tk, rendered, images, vm, n_embd);
}

mm_prompt mm_build_prompt_device(vision_model & vm, sycl::queue & q, const tokenizer & tk, const std::string & rendered,
                                 const std::vector<mm_image> & images, int n_embd, float * d_out) {
    const int n_img = (int)images.size();
    std::vector<mm_block> blocks(n_img);
    std::vector<vision_input> vin(n_img);
    std::vector<int> off(n_img);
    int total_rows = 0;
    for (int k = 0; k < n_img; k++) {
        vin[k] = vision_model::make_input(vm, images[k]);
        if (vin[k].n_out <= 0) {
            throw std::runtime_error("mm: image too small for the patch grid");
        }
        mm_block & b = blocks[k];
        b.pad_tok = pad_tok_of(tk, MM_KIND_IMAGE);
        b.kind = MM_KIND_IMAGE;
        b.out_w = vin[k].out_w;
        b.out_h = vin[k].out_h;
        b.n_tok = vin[k].n_out;
        b.n_pos = std::max(vin[k].out_w, vin[k].out_h);
        off[k] = total_rows;
        total_rows += vin[k].n_out;
    }
    if (total_rows > kMaxImgTokens) {
        throw std::runtime_error("mm: image tokens exceed the kMaxImgTokens budget");
    }
    for (int k = 0; k < n_img; k++) {
        vm.encode_device(q, vin[k], d_out + (size_t)off[k] * n_embd);
    }
    mm_prompt p = mm_expand_prompt(tk, rendered, blocks, total_rows);
    p.d_embd = d_out;
    return p;
}

// ---------------------------------------------------------------------------
// Video.  Each video is subsampled to `max_frames`, the selected frames go
// through the same preprocessing as images (smart resize + normalize), and the
// vision tower encodes every frame into the embedding rows of the prompt.
// ---------------------------------------------------------------------------
namespace {

struct vid_plan {
    std::vector<mm_block> blocks;
    std::vector<int> off;
    std::vector<std::vector<mm_image>> imgs; // preprocessed selected frames
    int total_rows = 0;
};

image_preproc_cfg vision_cfg(const vision_model & vm) {
    image_preproc_cfg cfg;
    cfg.patch_size = vm.hp.patch_size;
    cfg.merge = vm.hp.merge;
    const int patch_area = cfg.patch_size * cfg.patch_size * cfg.merge * cfg.merge;
    cfg.min_pixels = 8 * patch_area;
    cfg.max_pixels = kMaxImgTokens * patch_area;
    for (int c = 0; c < 3; c++) {
        cfg.mean[c] = vm.hp.mean[c];
        cfg.std[c] = vm.hp.std[c];
    }
    return cfg;
}

// subsample + preprocess; also lays out the blocks (grid derived from the
// preprocessed frames, so it holds whatever smart-resize produced)
vid_plan plan_videos(const tokenizer & tk, const std::vector<mm_video> & vids, const vision_model & vm, int max_frames) {
    vid_plan pl;
    pl.blocks.resize(vids.size());
    pl.off.resize(vids.size());
    pl.imgs.resize(vids.size());
    const image_preproc_cfg cfg = vision_cfg(vm);
    for (size_t v = 0; v < vids.size(); v++) {
        const mm_video & vid = vids[v];
        const int t = std::min((int)vid.frames.size(), max_frames);
        std::vector<const mm_video_frame *> sel;
        mm_video_subsample(vid.frames, t, sel);
        if (sel.empty()) {
            throw std::runtime_error("mm: video has no usable frames");
        }
        pl.imgs[v].reserve(sel.size());
        for (const mm_video_frame * f : sel) {
            pl.imgs[v].push_back(mm_image_preprocess(f->rgb.data(), f->width, f->height, cfg));
        }
        vision_input vi0 = vision_model::make_input(vm, pl.imgs[v][0]);
        const int n_out = vi0.n_out;
        if (n_out <= 0) {
            throw std::runtime_error("mm: video frame too small for the patch grid");
        }
        mm_block & b = pl.blocks[v];
        b.pad_tok = pad_tok_of(tk, MM_KIND_VIDEO);
        b.kind = MM_KIND_VIDEO;
        b.n_frames = (int)sel.size();
        b.out_w = vi0.out_w;
        b.out_h = vi0.out_h;
        b.n_tok = b.n_frames * n_out;
        b.n_pos = std::max(b.out_w, b.out_h);
        pl.off[v] = pl.total_rows;
        pl.total_rows += b.n_tok;
    }
    if (pl.total_rows > kMaxImgTokens) {
        throw std::runtime_error("mm: video tokens exceed the kMaxImgTokens budget");
    }
    return pl;
}

} // namespace

mm_prompt mm_build_prompt_video(const tokenizer & tk, const std::string & rendered, const std::vector<mm_video> & vids,
                                const vision_model & vm, int n_embd, int max_frames) {
    vid_plan pl = plan_videos(tk, vids, vm, max_frames);
    mm_prompt p = mm_expand_prompt(tk, rendered, pl.blocks, pl.total_rows);
    p.embd.resize((size_t)pl.total_rows * n_embd);
    std::vector<float> e;
    for (size_t v = 0; v < vids.size(); v++) {
        int row = pl.off[v];
        for (const mm_image & img : pl.imgs[v]) {
            vision_input vi = vision_model::make_input(vm, img);
            vm.encode_host(vi, e);
            std::memcpy(p.embd.data() + (size_t)row * n_embd, e.data(), e.size() * sizeof(float));
            row += vi.n_out;
        }
    }
    return p;
}

mm_prompt mm_build_prompt_video_device(vision_model & vm, sycl::queue & q, const tokenizer & tk,
                                       const std::string & rendered, const std::vector<mm_video> & vids, int n_embd,
                                       float * d_out, int max_frames) {
    vid_plan pl = plan_videos(tk, vids, vm, max_frames);
    for (size_t v = 0; v < vids.size(); v++) {
        int row = pl.off[v];
        for (const mm_image & img : pl.imgs[v]) {
            vision_input vi = vision_model::make_input(vm, img);
            vm.encode_device(q, vi, d_out + (size_t)row * n_embd);
            row += vi.n_out;
        }
    }
    mm_prompt p = mm_expand_prompt(tk, rendered, pl.blocks, pl.total_rows);
    p.d_embd = d_out;
    return p;
}

// ---------------------------------------------------------------------------
// Audio.
// ---------------------------------------------------------------------------
namespace {

// decode + resample + mel, then the audio tower input geometry
void audio_prepare(const mm_audio & src, const audio_preproc_cfg & cfg, const audio_model & am, audio_input & ai,
                   std::vector<float> & mel) {
    mm_audio a;
    mm_audio_resample(src, cfg.sample_rate, a);
    int n_frames = 0;
    mm_audio_preprocess(a, cfg, mel, n_frames);
    ai = audio_model::make_input(am, mel.data(), n_frames);
}

audio_preproc_cfg cfg_of(const audio_model & am) {
    audio_preproc_cfg cfg;
    cfg.sample_rate = am.hp.sample_rate;
    cfg.n_fft = am.hp.n_fft;
    cfg.hop = am.hp.hop;
    cfg.n_mel = am.hp.n_mel;
    cfg.f_min = am.hp.f_min;
    cfg.f_max = am.hp.f_max;
    return cfg;
}

} // namespace

mm_prompt mm_build_prompt_audio(const tokenizer & tk, const std::string & rendered, const std::vector<mm_audio> & auds,
                                const audio_model & am, int n_embd) {
    if (audio_out_width(am) != n_embd) {
        throw std::runtime_error("mm: audio tower output width != text n_embd");
    }
    const audio_preproc_cfg cfg = cfg_of(am);
    std::vector<mm_block> blocks(auds.size());
    std::vector<audio_input> ai(auds.size());
    std::vector<std::vector<float>> mels(auds.size());
    std::vector<int> off(auds.size());
    int total_rows = 0;
    for (size_t k = 0; k < auds.size(); k++) {
        audio_prepare(auds[k], cfg, am, ai[k], mels[k]);
        if (ai[k].n_out <= 0) {
            throw std::runtime_error("mm: audio produced no embeddings");
        }
        mm_block & b = blocks[k];
        b.pad_tok = pad_tok_of(tk, MM_KIND_AUDIO);
        b.kind = MM_KIND_AUDIO;
        b.n_tok = ai[k].n_out;
        b.n_pos = ai[k].n_out;
        off[k] = total_rows;
        total_rows += ai[k].n_out;
    }
    if (total_rows > kMaxImgTokens) {
        throw std::runtime_error("mm: audio tokens exceed the kMaxImgTokens budget");
    }

    mm_prompt p = mm_expand_prompt(tk, rendered, blocks, total_rows);
    p.embd.resize((size_t)total_rows * n_embd);
    for (size_t k = 0; k < auds.size(); k++) {
        std::vector<float> e;
        am.encode_host(ai[k], e);
        std::memcpy(p.embd.data() + (size_t)off[k] * n_embd, e.data(), e.size() * sizeof(float));
    }
    return p;
}

mm_prompt mm_build_prompt_audio_device(audio_model & am, sycl::queue & q, const tokenizer & tk,
                                       const std::string & rendered, const std::vector<mm_audio> & auds, int n_embd,
                                       float * d_out) {
    if (audio_out_width(am) != n_embd) {
        throw std::runtime_error("mm: audio tower output width != text n_embd");
    }
    const audio_preproc_cfg cfg = cfg_of(am);
    std::vector<mm_block> blocks(auds.size());
    std::vector<audio_input> ai(auds.size());
    std::vector<std::vector<float>> mels(auds.size());
    std::vector<int> off(auds.size());
    int total_rows = 0;
    for (size_t k = 0; k < auds.size(); k++) {
        audio_prepare(auds[k], cfg, am, ai[k], mels[k]);
        if (ai[k].n_out <= 0) {
            throw std::runtime_error("mm: audio produced no embeddings");
        }
        mm_block & b = blocks[k];
        b.pad_tok = pad_tok_of(tk, MM_KIND_AUDIO);
        b.kind = MM_KIND_AUDIO;
        b.n_tok = ai[k].n_out;
        b.n_pos = ai[k].n_out;
        off[k] = total_rows;
        total_rows += ai[k].n_out;
    }
    if (total_rows > kMaxImgTokens) {
        throw std::runtime_error("mm: audio tokens exceed the kMaxImgTokens budget");
    }

    for (size_t k = 0; k < auds.size(); k++) {
        am.encode_device(q, ai[k], d_out + (size_t)off[k] * n_embd);
    }

    mm_prompt p = mm_expand_prompt(tk, rendered, blocks, total_rows);
    p.d_embd = d_out;
    return p;
}

mm_prompt mm_build_prompt_mixed_device(vision_model & vm, audio_model & am, sycl::queue & q, const tokenizer & tk,
                                       const std::string & rendered, const std::vector<mm_image> & images,
                                       const std::vector<mm_video> & vids, const std::vector<mm_audio> & auds,
                                       const std::vector<mm_media_ref> & order, int n_embd, float * d_out,
                                       int max_video_frames) {
    // ---- plan every block in pad order (mixed image/video/audio) -----------
    const int E = n_embd;
    bool any_audio = false;
    std::vector<mm_block> blocks(order.size());
    std::vector<int> off(order.size());
    // per-kind pre-encoded data: video uses subsampled+preprocessed frames,
    // audio keeps the log-mel (the audio_input references it) for the encode pass
    std::vector<std::vector<mm_image>> vimgs;
    vimgs.resize(vids.size());
    struct anim_input {
        audio_input in;
        std::vector<float> mel;
    };
    std::vector<anim_input> ains;
    ains.resize(auds.size());

    const image_preproc_cfg vcfg = vision_cfg(vm);
    int rows = 0;
    for (size_t k = 0; k < order.size(); k++) {
        const mm_media_ref & mr = order[k];
        mm_block & b = blocks[k];
        const int idx = mr.idx;
        if (mr.kind == MM_KIND_IMAGE) {
            if (idx >= (int)images.size()) {
                throw std::runtime_error("mm: image index out of range");
            }
            vision_input vi = vision_model::make_input(vm, images[idx]);
            if (vi.n_out <= 0) {
                throw std::runtime_error("mm: image too small for the patch grid");
            }
            b.pad_tok = pad_tok_of(tk, MM_KIND_IMAGE);
            b.kind = MM_KIND_IMAGE;
            b.out_w = vi.out_w;
            b.out_h = vi.out_h;
            b.n_tok = vi.n_out;
            b.n_pos = std::max(vi.out_w, vi.out_h);
        } else if (mr.kind == MM_KIND_VIDEO) {
            if (idx >= (int)vids.size()) {
                throw std::runtime_error("mm: video index out of range");
            }
            // subsample + preprocess once; reuse the frame list for encode
            const mm_video & vid = vids[idx];
            const int t = std::min((int)vid.frames.size(), max_video_frames);
            std::vector<const mm_video_frame *> sel;
            mm_video_subsample(vid.frames, t, sel);
            if (sel.empty()) {
                throw std::runtime_error("mm: video has no usable frames");
            }
            std::vector<mm_image> & fimgs = vimgs[idx];
            fimgs.reserve(sel.size());
            for (const mm_video_frame * f : sel) {
                fimgs.push_back(mm_image_preprocess(f->rgb.data(), f->width, f->height, vcfg));
            }
            vision_input vi0 = vision_model::make_input(vm, fimgs[0]);
            if (vi0.n_out <= 0) {
                throw std::runtime_error("mm: video frame too small for the patch grid");
            }
            b.pad_tok = pad_tok_of(tk, MM_KIND_VIDEO);
            b.kind = MM_KIND_VIDEO;
            b.n_frames = (int)sel.size();
            b.out_w = vi0.out_w;
            b.out_h = vi0.out_h;
            b.n_tok = (int)fimgs.size() * vi0.n_out;
            b.n_pos = std::max(vi0.out_w, vi0.out_h);
        } else if (mr.kind == MM_KIND_AUDIO) {
            if (idx >= (int)auds.size()) {
                throw std::runtime_error("mm: audio index out of range");
            }
            any_audio = true;
            anim_input & ain = ains[idx];
            audio_prepare(auds[idx], cfg_of(am), am, ain.in, ain.mel);
            if (ain.in.n_out <= 0) {
                throw std::runtime_error("mm: audio produced no embeddings");
            }
            b.pad_tok = pad_tok_of(tk, MM_KIND_AUDIO);
            b.kind = MM_KIND_AUDIO;
            b.n_tok = ain.in.n_out;
            b.n_pos = ain.in.n_out;
        } else {
            throw std::runtime_error("mm: unknown media kind");
        }
        off[k] = rows;
        rows += b.n_tok;
    }
    if (any_audio && audio_out_width(am) != E) {
        throw std::runtime_error("mm: audio tower output width != text n_embd");
    }
    if (rows > kMaxImgTokens) {
        throw std::runtime_error("mm: media tokens exceed the kMaxImgTokens budget");
    }

    // ---- encode into d_out at each block offset -----------------------------
    for (size_t k = 0; k < order.size(); k++) {
        const mm_media_ref & mr = order[k];
        float * base = d_out + (size_t)off[k] * E;
        if (mr.kind == MM_KIND_IMAGE) {
            vision_input vi = vision_model::make_input(vm, images[mr.idx]);
            vm.encode_device(q, vi, base);
        } else if (mr.kind == MM_KIND_VIDEO) {
            const std::vector<mm_image> & fimgs = vimgs[mr.idx];
            float * p = base;
            for (const mm_image & img : fimgs) {
                vision_input vi = vision_model::make_input(vm, img);
                vm.encode_device(q, vi, p);
                p += vi.n_out * E;
            }
        } else if (mr.kind == MM_KIND_AUDIO) {
            am.encode_device(q, ains[mr.idx].in, base);
        }
    }

    mm_prompt p = mm_expand_prompt(tk, rendered, blocks, rows);
    p.d_embd = d_out;
    return p;
}

} // namespace si