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
static void check_vision_width(const vision_model & vm, int n_embd) {
    if (vm.hp.proj_dim != n_embd) {
        throw std::runtime_error("mm: vision projector output width != text n_embd");
    }
}

// Phase 1, images.  This loop used to be written out twice - once in the host
// builder and once, verbatim, in the device one - so the two copies could drift
// apart with nothing to notice.
mm_image_plan mm_plan_images(const tokenizer & tk, const std::vector<mm_image> & images, const vision_model & vm) {
    mm_image_plan pl;
    const int n_img = (int)images.size();
    pl.p.blocks.resize(n_img);
    pl.p.off.resize(n_img);
    pl.vin.resize(n_img);
    for (int k = 0; k < n_img; k++) {
        pl.vin[k] = vision_model::make_input(vm, images[k]);
        if (pl.vin[k].n_out <= 0) {
            throw std::runtime_error("mm: image too small for the patch grid");
        }
        mm_block & b = pl.p.blocks[k];
        b.pad_tok = pad_tok_of(tk, MM_KIND_IMAGE);
        b.kind = MM_KIND_IMAGE;
        b.out_w = pl.vin[k].out_w;
        b.out_h = pl.vin[k].out_h;
        b.n_tok = pl.vin[k].n_out;
        b.n_pos = std::max(pl.vin[k].out_w, pl.vin[k].out_h);
        pl.p.off[k] = pl.p.total_rows;
        pl.p.total_rows += b.n_tok;
    }
    if (pl.p.total_rows > kMaxImgTokens) {
        throw std::runtime_error("mm: image tokens exceed the kMaxImgTokens budget");
    }
    return pl;
}

mm_prompt mm_build_prompt(const tokenizer & tk, const std::string & rendered, const std::vector<mm_image> & images,
                          const vision_model & vm, int n_embd) {
    check_vision_width(vm, n_embd);
    const mm_image_plan pl = mm_plan_images(tk, images, vm);
    // phase 3 first, so phase 2 writes into the buffer the prompt already sized
    mm_prompt p = mm_expand_prompt(tk, rendered, pl.p.blocks, pl.p.total_rows);
    p.embd.resize((size_t)pl.p.total_rows * n_embd);
    // phase 2: host encode at the offsets phase 1 handed out
    std::vector<float> e;
    for (size_t k = 0; k < pl.p.blocks.size(); k++) {
        vm.encode_host(pl.vin[k], e);
        std::memcpy(p.embd.data() + (size_t)pl.p.off[k] * n_embd, e.data(), e.size() * sizeof(float));
    }
    return p;
}

mm_prompt mm_build_prompt_device(vision_model & vm, sycl::queue & q, const tokenizer & tk, const std::string & rendered,
                                 const std::vector<mm_image> & images, int n_embd, float * d_out) {
    check_vision_width(vm, n_embd);
    const mm_image_plan pl = mm_plan_images(tk, images, vm);
    // phase 2: encode at the offsets phase 1 handed out
    for (size_t k = 0; k < pl.p.blocks.size(); k++) {
        vm.encode_device(q, pl.vin[k], d_out + (size_t)pl.p.off[k] * n_embd);
    }
    // phase 3
    mm_prompt p = mm_expand_prompt(tk, rendered, pl.p.blocks, pl.p.total_rows);
    p.d_embd = d_out;
    return p;
}

// ---------------------------------------------------------------------------
// Video.  Each video is subsampled to `max_frames`, the selected frames go
// through the same preprocessing as images (smart resize + normalize), and the
// vision tower encodes every frame into the embedding rows of the prompt.
// ---------------------------------------------------------------------------
namespace {

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

} // namespace

// Phase 1, videos: subsample + preprocess, and lay out the blocks (the grid is
// derived from the *preprocessed* frames, so it holds whatever smart-resize
// produced).
mm_video_plan mm_plan_videos(const tokenizer & tk, const std::vector<mm_video> & vids, const vision_model & vm,
                             int max_frames) {
    mm_video_plan pl;
    pl.p.blocks.resize(vids.size());
    pl.p.off.resize(vids.size());
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
        mm_block & b = pl.p.blocks[v];
        b.pad_tok = pad_tok_of(tk, MM_KIND_VIDEO);
        b.kind = MM_KIND_VIDEO;
        b.n_frames = (int)sel.size();
        b.out_w = vi0.out_w;
        b.out_h = vi0.out_h;
        b.n_tok = b.n_frames * n_out;
        b.n_pos = std::max(b.out_w, b.out_h);
        pl.p.off[v] = pl.p.total_rows;
        pl.p.total_rows += b.n_tok;
    }
    if (pl.p.total_rows > kMaxImgTokens) {
        throw std::runtime_error("mm: video tokens exceed the kMaxImgTokens budget");
    }
    return pl;
}

mm_prompt mm_build_prompt_video(const tokenizer & tk, const std::string & rendered, const std::vector<mm_video> & vids,
                                const vision_model & vm, int n_embd, int max_frames) {
    check_vision_width(vm, n_embd);
    const mm_video_plan pl = mm_plan_videos(tk, vids, vm, max_frames);
    mm_prompt p = mm_expand_prompt(tk, rendered, pl.p.blocks, pl.p.total_rows);
    p.embd.resize((size_t)pl.p.total_rows * n_embd);
    std::vector<float> e;
    for (size_t v = 0; v < vids.size(); v++) {
        int row = pl.p.off[v];
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
    check_vision_width(vm, n_embd);
    const mm_video_plan pl = mm_plan_videos(tk, vids, vm, max_frames);
    for (size_t v = 0; v < vids.size(); v++) {
        int row = pl.p.off[v];
        for (const mm_image & img : pl.imgs[v]) {
            vision_input vi = vision_model::make_input(vm, img);
            vm.encode_device(q, vi, d_out + (size_t)row * n_embd);
            row += vi.n_out;
        }
    }
    mm_prompt p = mm_expand_prompt(tk, rendered, pl.p.blocks, pl.p.total_rows);
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

// Phase 1, audio.  Also written out twice before (host and device), and the
// pairing of `in` with `mel` is why items travel together: audio_input::mel is a
// borrowed pointer, so splitting them into two parallel vectors would leave a
// dangling one on any move.
mm_audio_plan mm_plan_audios(const tokenizer & tk, const std::vector<mm_audio> & auds, const audio_model & am) {
    const audio_preproc_cfg cfg = cfg_of(am);
    mm_audio_plan pl;
    pl.p.blocks.resize(auds.size());
    pl.p.off.resize(auds.size());
    pl.items.resize(auds.size());
    for (size_t k = 0; k < auds.size(); k++) {
        audio_prepare(auds[k], cfg, am, pl.items[k].in, pl.items[k].mel);
        if (pl.items[k].in.n_out <= 0) {
            throw std::runtime_error("mm: audio produced no embeddings");
        }
        mm_block & b = pl.p.blocks[k];
        b.pad_tok = pad_tok_of(tk, MM_KIND_AUDIO);
        b.kind = MM_KIND_AUDIO;
        b.n_tok = pl.items[k].in.n_out;
        b.n_pos = pl.items[k].in.n_out;
        pl.p.off[k] = pl.p.total_rows;
        pl.p.total_rows += b.n_tok;
    }
    if (pl.p.total_rows > kMaxImgTokens) {
        throw std::runtime_error("mm: audio tokens exceed the kMaxImgTokens budget");
    }
    return pl;
}

mm_prompt mm_build_prompt_audio(const tokenizer & tk, const std::string & rendered, const std::vector<mm_audio> & auds,
                                const audio_model & am, int n_embd) {
    if (audio_out_width(am) != n_embd) {
        throw std::runtime_error("mm: audio tower output width != text n_embd");
    }
    const mm_audio_plan pl = mm_plan_audios(tk, auds, am);

    // phase 3, then phase 2 into the buffer it sized
    mm_prompt p = mm_expand_prompt(tk, rendered, pl.p.blocks, pl.p.total_rows);
    p.embd.resize((size_t)pl.p.total_rows * n_embd);
    std::vector<float> e;
    for (size_t k = 0; k < pl.items.size(); k++) {
        am.encode_host(pl.items[k].in, e);
        std::memcpy(p.embd.data() + (size_t)pl.p.off[k] * n_embd, e.data(), e.size() * sizeof(float));
    }
    return p;
}

mm_prompt mm_build_prompt_audio_device(audio_model & am, sycl::queue & q, const tokenizer & tk,
                                       const std::string & rendered, const std::vector<mm_audio> & auds, int n_embd,
                                       float * d_out) {
    if (audio_out_width(am) != n_embd) {
        throw std::runtime_error("mm: audio tower output width != text n_embd");
    }
    const mm_audio_plan pl = mm_plan_audios(tk, auds, am);

    // phase 2: encode at the offsets phase 1 handed out
    for (size_t k = 0; k < pl.items.size(); k++) {
        am.encode_device(q, pl.items[k].in, d_out + (size_t)pl.p.off[k] * n_embd);
    }
    // phase 3
    mm_prompt p = mm_expand_prompt(tk, rendered, pl.p.blocks, pl.p.total_rows);
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
    bool any_image = false, any_video = false, any_audio = false;
    for (const mm_media_ref & mr : order) {
        any_image |= mr.kind == MM_KIND_IMAGE;
        any_video |= mr.kind == MM_KIND_VIDEO;
        any_audio |= mr.kind == MM_KIND_AUDIO;
    }
    // Tower geometry first: it is a property of the deployment, not of this
    // request, so it is the cheapest thing to rule out before any preprocessing.
    if (any_image || any_video) {
        check_vision_width(vm, E);
    }
    if (any_audio && audio_out_width(am) != E) {
        throw std::runtime_error("mm: audio tower output width != text n_embd");
    }
    std::vector<mm_block> blocks(order.size());
    std::vector<int> off(order.size());

    // Each kind is planned by its own planner, over the whole vector, so the grid
    // derivation, the video frame subsampling and the audio preparation exist
    // once.  What is left here is genuinely the mixed case's own job: laying the
    // per-kind blocks out in *placeholder* order and checking the budget over all
    // of them together (a per-kind subtotal can never exceed the total, so the
    // planners' own checks are the earlier of the two rejections).
    mm_image_plan ipl;
    mm_video_plan vpl;
    mm_audio_plan apl;
    if (any_image) {
        ipl = mm_plan_images(tk, images, vm);
    }
    if (any_video) {
        vpl = mm_plan_videos(tk, vids, vm, max_video_frames);
    }
    if (any_audio) {
        apl = mm_plan_audios(tk, auds, am);
    }

    int rows = 0;
    for (size_t k = 0; k < order.size(); k++) {
        const mm_media_ref & mr = order[k];
        mm_block & b = blocks[k];
        const int idx = mr.idx;
        // The block is whatever that kind's planner decided; this only checks
        // that the reference is in range and places it in placeholder order.
        if (mr.kind == MM_KIND_IMAGE) {
            if (idx >= (int)images.size()) {
                throw std::runtime_error("mm: image index out of range");
            }
            b = ipl.p.blocks[idx];
        } else if (mr.kind == MM_KIND_VIDEO) {
            if (idx >= (int)vids.size()) {
                throw std::runtime_error("mm: video index out of range");
            }
            b = vpl.p.blocks[idx];
        } else if (mr.kind == MM_KIND_AUDIO) {
            if (idx >= (int)auds.size()) {
                throw std::runtime_error("mm: audio index out of range");
            }
            any_audio = true;
            b = apl.p.blocks[idx];
        } else {
            throw std::runtime_error("mm: unknown media kind");
        }
        off[k] = rows;
        rows += b.n_tok;
    }
    if (rows > kMaxImgTokens) {
        throw std::runtime_error("mm: media tokens exceed the kMaxImgTokens budget");
    }

    // ---- encode into d_out at each block offset -----------------------------
    for (size_t k = 0; k < order.size(); k++) {
        const mm_media_ref & mr = order[k];
        float * base = d_out + (size_t)off[k] * E;
        if (mr.kind == MM_KIND_IMAGE) {
            // the plan already built this vision_input; the mixed path used to
            // call make_input a second time for the same image
            vm.encode_device(q, ipl.vin[mr.idx], base);
        } else if (mr.kind == MM_KIND_VIDEO) {
            const std::vector<mm_image> & fimgs = vpl.imgs[mr.idx];
            float * p = base;
            for (const mm_image & img : fimgs) {
                vision_input vi = vision_model::make_input(vm, img);
                vm.encode_device(q, vi, p);
                p += vi.n_out * E;
            }
        } else if (mr.kind == MM_KIND_AUDIO) {
            am.encode_device(q, apl.items[mr.idx].in, base);
        }
    }

    mm_prompt p = mm_expand_prompt(tk, rendered, blocks, rows);
    p.d_embd = d_out;
    return p;
}

} // namespace si