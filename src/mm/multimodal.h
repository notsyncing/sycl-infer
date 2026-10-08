#pragma once
// ---------------------------------------------------------------------------
// Multimodal prompt assembly: tokenize a chat-template rendering that contains
// image/video/audio placeholders, expand each `<|image_pad|>` / `<|video_pad|>`
// / `<|audio_pad|>` into its embedding rows and derive the M-RoPE positions the
// text model needs.
//
// The position layout follows the Qwen2.5-VL reference (get_rope_index /
// get_vision_position_ids):
//   * an image of nx*ny merged tokens consumes max(nx, ny) positions.  Every
//     image token shares the temporal position `base`, and uses `base + row` /
//     `base + col` for the row/col sections.  Text tokens continue one position
//     per token.
//   * a video of T frames x nx*ny merged tokens contributes T*nx*ny tokens and
//     still consumes max(nx, ny) positions.  Token (frame i, row j, col k) uses
//     (t,h,w) = (base+i, base+j, base+k).
//   * an audio block of n embeddings consumes one position per embedding and
//     keeps the row/col sections at 0 (TM-RoPE: temporal only).
// ---------------------------------------------------------------------------
#include <cstdint>
#include <string>
#include <vector>

#include "audio.h"
#include "audio_model.h"
#include "image.h"
#include "video.h"
#include "vision.h"

// `tokenizer` lives in the global namespace (see src/model/tokenizer.h)
struct tokenizer;

namespace si {

constexpr int MM_KIND_IMAGE = 0; // still/2D grid (temporal = 1)
constexpr int MM_KIND_VIDEO = 1; // 3D grid (temporal = frame index)
constexpr int MM_KIND_AUDIO = 2; // 1D temporal stream (row/col = 0)

// One placeholder region in the rendered prompt.  `n_tok` embedding rows are
// substituted for a single pad token; `n_pos` positions are consumed in text
// space.  Blocks must be listed in the order their pads appear in `rendered`.
struct mm_block {
    int pad_tok = 0;   // <|image_pad|> / <|video_pad|> / <|audio_pad|>
    int kind = MM_KIND_IMAGE;
    int n_frames = 1;  // video temporal frames (= 1 for images, unused for audio)
    int out_w = 1;     // merged grid width (image/video)
    int out_h = 1;     // merged grid height (image/video)
    int n_tok = 0;     // tokens / embedding rows contributed
    int n_pos = 0;     // positions consumed in text space
};

struct mm_prompt {
    std::vector<int> tokens;
    std::vector<int32_t> mrope;     // 4 * n, section-major ([s * n + i])
    std::vector<int32_t> img_row;   // n, -1 or a row in `embd`
    std::vector<float> embd;        // n_img * text n_embd (host path)
    const float * d_embd = nullptr; // device path: n_img * n_embd, owned by caller
    int n_img = 0;
    int pos_after = 0; // next decode position
    bool has_images() const {
        return n_img > 0;
    }
};

// ---------------------------------------------------------------------------
// Three-phase assembly.  Prompt building is always the same three steps:
//
//   1. plan   - decide what each placeholder contributes (grid, token count,
//               positions) and hand out its embedding-row offset, then check the
//               kMaxImgTokens budget.  Nothing has run yet, so an over-budget
//               request is refused before any tower work happens.
//   2. encode - run the towers into the caller's buffer at those offsets.
//   3. expand - mm_expand_prompt: tokens, M-RoPE positions, img_row mapping.
//
// The phases exist to be *shared*: the four planners used to inline the same
// accumulate-blocks / assign-offsets / check-budget loop, and the image planner
// was written out twice - once per host/device variant - with the two copies
// free to drift.  A plan is also the only part of this that is checkable without
// a GPU, so it is what the header exposes.
// ---------------------------------------------------------------------------

// Phase 1's output, shared by every media kind.
struct mm_plan {
    std::vector<mm_block> blocks; // in placeholder order
    std::vector<int> off;          // first embedding row of each block
    int total_rows = 0;            // == sum of blocks[].n_tok
};

// Images: `vin[k]` is what phase 2 encodes.
struct mm_image_plan {
    mm_plan p;
    std::vector<vision_input> vin;
};

// Videos: `imgs[v]` are the subsampled, preprocessed frames.  All frames of one
// video share a grid, so blocks[v].n_tok == frames * n_out.
struct mm_video_plan {
    mm_plan p;
    std::vector<std::vector<mm_image>> imgs;
};

// Audio: `items[k].in.mel` points *into* `items[k].mel`, so the two must travel
// together - which is why this is one struct and not two parallel vectors.
struct mm_audio_plan {
    struct item {
        audio_input in;
        std::vector<float> mel;
    };
    mm_plan p;
    std::vector<item> items;
};

// Each planner validates its tower's output width against the text model and
// throws on an over-budget or unencodable request.  `vm`/`am` must outlive the
// plan only through phase 2's encode calls.
mm_image_plan mm_plan_images(const tokenizer & tk, const std::vector<mm_image> & images, const vision_model & vm);
mm_video_plan mm_plan_videos(const tokenizer & tk, const std::vector<mm_video> & vids, const vision_model & vm,
                             int max_frames);
mm_audio_plan mm_plan_audios(const tokenizer & tk, const std::vector<mm_audio> & auds, const audio_model & am);

// Generic expansion: `rendered` must contain exactly as many of each pad token
// as blocks with that pad token; the blocks appear in the order the pads do.
// `total_rows` is the number of embedding rows the caller filled; the planners
// above have already checked it against kMaxImgTokens.
mm_prompt mm_expand_prompt(const tokenizer & tk, const std::string & rendered, const std::vector<mm_block> & blocks,
                           int total_rows);

// Image prompt (kept for reference): `rendered` must contain exactly one
// `<|image_pad|>` per entry of `images`.  `n_embd` is the text model width
// (the vision projection width).
mm_prompt mm_build_prompt(const tokenizer & tk, const std::string & rendered, const std::vector<mm_image> & images,
                          const vision_model & vm, int n_embd);

// Device variant: runs the vision tower on the GPU and writes the merged
// embeddings into `d_out` (n_img * n_embd floats).  `vm` must outlive the call.
mm_prompt mm_build_prompt_device(vision_model & vm, sycl::queue & q, const tokenizer & tk, const std::string & rendered,
                                 const std::vector<mm_image> & images, int n_embd, float * d_out);

// Video prompt: each `mm_video` decodes to up to `max_frames` sampled frames,
// every frame goes through the vision tower (all frames share one grid), and a
// single `<|video_pad|>` expands to T*nx*ny merged tokens with the (t,h,w)
// layout above.  `max_frames` is capped per video; the total token budget is
// kMaxImgTokens.
mm_prompt mm_build_prompt_video(const tokenizer & tk, const std::string & rendered, const std::vector<mm_video> & vids,
                                const vision_model & vm, int n_embd, int max_frames);
mm_prompt mm_build_prompt_video_device(vision_model & vm, sycl::queue & q, const tokenizer & tk,
                                       const std::string & rendered, const std::vector<mm_video> & vids, int n_embd,
                                       float * d_out, int max_frames);

// Audio prompt: each `mm_audio` encodes to n embedding rows via the audio
// tower; a single `<|audio_pad|>` expands to one token per embedding using the
// TM-RoPE layout (temporal advances one position per embedding, row/col stay 0).
mm_prompt mm_build_prompt_audio(const tokenizer & tk, const std::string & rendered, const std::vector<mm_audio> & auds,
                                const audio_model & am, int n_embd);
mm_prompt mm_build_prompt_audio_device(audio_model & am, sycl::queue & q, const tokenizer & tk,
                                       const std::string & rendered, const std::vector<mm_audio> & auds, int n_embd,
                                       float * d_out);

// One media block in a mixed request.  `order` enumerates the media in the
// order their placeholders appear in `rendered`; each entry picks one of the
// three parallel vectors.
struct mm_media_ref {
    int kind = MM_KIND_IMAGE;
    int idx = 0;
};

// Combined image+video+audio prompt for the device:
//  * blocks are laid out in `order` (one per placeholder in `rendered`),
//  * every media item is encoded into `d_out` at its block offset (images and
//    videos through the vision tower, audio through the audio tower),
//  * `audio_model` is required iff any order entry is MM_KIND_AUDIO.
// The three vectors must contain every index referenced by `order` - and, since
// the planners now run over whole vectors, `order` must reference *every* index
// too.  Both callers build `order` in lockstep with the vectors (each push is
// followed by an order entry), so the two conditions hold together; an unused
// trailing item would otherwise be planned (and could be rejected) without ever
// appearing in the prompt.
mm_prompt mm_build_prompt_mixed_device(vision_model & vm, audio_model & am, sycl::queue & q, const tokenizer & tk,
                                       const std::string & rendered, const std::vector<mm_image> & images,
                                       const std::vector<mm_video> & vids, const std::vector<mm_audio> & auds,
                                       const std::vector<mm_media_ref> & order, int n_embd, float * d_out,
                                       int max_video_frames);

} // namespace si