#pragma once
// ---------------------------------------------------------------------------
// Host-side audio decoding and log-mel spectrogram extraction.
//
// WAV files are decoded natively (PCM 8/16/24/32, IEEE float, unsigned 8); any
// other container (MP3, Opus, ...) is decoded by the `ffmpeg` CLI subprocess
// (settable with PF_AV_FFMPEG), exactly like llama.cpp's audio input handling.
// The mel geometry follows the Qwen3-Omni / AuT convention: 16 kHz mono,
// 25 ms window / 10 ms hop, 128 mel bins.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <string>
#include <vector>

namespace si {

// mono PCM, -1.0 .. 1.0
struct mm_audio {
    int sample_rate = 16000;
    std::vector<float> samples;
};

struct audio_preproc_cfg {
    int sample_rate = 16000;
    int n_fft = 400;  // 25 ms at 16 kHz
    int hop = 160;    // 10 ms at 16 kHz
    int n_mel = 128;  // mel bins
    float f_min = 0.f;
    float f_max = 8000.f; // Nyquist at 16 kHz
    float floor = 1e-8f;  // log(1+power/floor) squelch
    int max_seconds = 60; // hard cap on the decoded duration
};

// Decode a WAV/RIFF buffer into `mm_audio` (native sample rate kept).  Returns
// false + `err` when the buffer is not a readable WAV.
bool mm_audio_decode_wav(const uint8_t * data, size_t len, mm_audio & out, std::string * err = nullptr);

// Decode a WAV or wave64 buffer; anything else is rejected (the caller falls
// back to the ffmpeg subprocess).  Resamples to `cfg.sample_rate`.
bool mm_audio_decode_mem(const uint8_t * data, size_t len, const audio_preproc_cfg & cfg, mm_audio & out,
                         std::string * err = nullptr);
// Decode an audio file on disk through the ffmpeg CLI to 16 kHz mono f32.
bool mm_audio_decode_ffmpeg(const std::string & path, mm_audio & out, std::string * err = nullptr);
// Decode an in-memory buffer: WAV natively, anything else through a temporary
// file + the ffmpeg CLI (so server-side mp3/ogg `input_audio` data works).
bool mm_audio_decode_bytes(const uint8_t * data, size_t len, const audio_preproc_cfg & cfg, mm_audio & out,
                           std::string * err = nullptr);

// Linear-interpolation resample to a fixed rate.
void mm_audio_resample(const mm_audio & in, int out_rate, mm_audio & out);

// Log-mel spectrogram.  `mel` is [n_frames][n_mel], frame t = t*hop samples
// with a Hann window of n_fft; power is folded through an HTK mel filterbank,
// then log(1 + power/floor).  `mel` has `n_frames` rows.
void mm_audio_preprocess(const mm_audio & a, const audio_preproc_cfg & cfg, std::vector<float> & mel, int & n_frames);

} // namespace si