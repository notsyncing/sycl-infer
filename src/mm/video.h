#pragma once
// ---------------------------------------------------------------------------
// Host-side video decoding and frame sampling.
//
// Two backends:
//   * the built-in AVI demuxer (RIFF/AVI + MJPEG or raw BI_RGB/BI_RGB565 video
//     streams), decoded with stb_image — enough for the test suite and for
//     simple MJPEG clips with no external tools;
//   * the `ffmpeg` CLI subprocess for anything else (detected by extension /
//     header sniffing), which pipes raw rgb24 frames.  No new compile-time
//     dependency: ffmpeg is just a subprocess, exactly like llama.cpp's audio
//     input handling.
//
// Frame sampling is uniform over the timeline: the decoder selects up to
// `max_frames` frames spread across the whole clip.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <string>
#include <vector>

namespace si {

// One decoded frame, still at its original resolution (interleaved RGB8).
struct mm_video_frame {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb; // w*h*3
    double pts = 0;           // presentation time in seconds (best effort)
};

constexpr int kMaxVideoDecodeFrames = 128; // hard cap on decoded frames

// Decoded + uniformly sampled video.  `frames` never exceeds `fmt.max_frames`
// and each stays at native resolution (the vision preprocessor resizes).
struct mm_video {
    std::vector<mm_video_frame> frames;
    int width() const {
        return frames.empty() ? 0 : frames[0].width;
    }
    int height() const {
        return frames.empty() ? 0 : frames[0].height;
    }
};

struct mm_video_fmt {
    int max_frames = 16; // frames to sample uniformly over the timeline
    int max_side = 1024; // frames larger than this (before sampling) are rejected
};

// Decode a video into uniformly sampled frames.  Returns false + `err` on
// failure (including "no ffmpeg" for containers the built-in cannot read).
bool mm_video_decode_mem(const uint8_t * data, size_t len, const mm_video_fmt & fmt, mm_video & out,
                         std::string * err = nullptr);
bool mm_video_decode_file(const std::string & path, const mm_video_fmt & fmt, mm_video & out,
                          std::string * err = nullptr);

// Uniformly reduce `frames` to `n` entries (drops every stride-th frame from
// the start).  Keeps temporal coverage; used to fit a video into the token
// budget once the per-frame grid is known.
void mm_video_subsample(const std::vector<mm_video_frame> & src, int n, std::vector<const mm_video_frame *> & out);

} // namespace si