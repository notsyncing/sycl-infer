#include "audio.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/types.h>
#include <unistd.h>
#include "common/env.h"

namespace si {

namespace {

constexpr const char * kFfmpegErr = "/tmp/opencode/ffmpeg_sycl_infer_err.log";

const char * ffmpeg_cmd() {
    const char * v = si::env::str("PF_AV_FFMPEG");
    return (v && v[0]) ? v : "ffmpeg";
}

// ---------------------------------------------------------------------------
// WAV (RIFF) decoder.  Covers the canonical formats: PCM 8/16/24/32 and IEEE
// float 32/64, mono or stereo (downmixed); the WAVE_FORMAT_EXTENSIBLE subformat
// tag is read from the SubFormat GUID.
// ---------------------------------------------------------------------------
bool parse_wav(const uint8_t * data, size_t len, int & rate, int & ch, int & bits, int & fmt_tag,
               std::vector<uint8_t> & raw) {
    if (len < 44 || std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0) {
        return false;
    }
    size_t p = 12;
    bool have_fmt = false, have_data = false;
    while (p + 8 <= len) {
        const char * id = (const char *)(data + p);
        const uint32_t size = (uint32_t)data[p + 4] | ((uint32_t)data[p + 5] << 8) | ((uint32_t)data[p + 6] << 16)
                              | ((uint32_t)data[p + 7] << 24);
        p += 8;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (size < 16 || p + 16 > len) {
                return false;
            }
            fmt_tag = (uint16_t)(data[p] | (data[p + 1] << 8));
            ch = (uint16_t)(data[p + 2] | (data[p + 3] << 8));
            rate = (int)((uint32_t)data[p + 4] | ((uint32_t)data[p + 5] << 8) | ((uint32_t)data[p + 6] << 16)
                         | ((uint32_t)data[p + 7] << 24));
            bits = (uint16_t)(data[p + 14] | (data[p + 15] << 8));
            if (fmt_tag == 6 && size >= 40 && p + 40 <= len) {
                // WAVE_FORMAT_EXTENSIBLE: real format tag = first two bytes of the
                // SubFormat GUID (after cbSize, valid bits, channel mask)
                const uint16_t sub = (uint16_t)(data[p + 24] | (data[p + 25] << 8));
                if (sub == 1 || sub == 3) {
                    fmt_tag = sub;
                }
            }
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            raw.assign(data + p, data + p + std::min<size_t>(size, len - p));
            have_data = true;
        }
        p += size + (size & 1);
        if (have_fmt && have_data) {
            break;
        }
    }
    return have_fmt && have_data;
}

void convert_pcm(const uint8_t * raw, size_t nbytes, int fmt_tag, int bits, int ch, std::vector<float> & out) {
    const size_t frame = (size_t)ch * (bits / 8);
    if (frame == 0) {
        return;
    }
    out.reserve(nbytes / frame);
    for (size_t i = 0; i + frame <= nbytes; i += frame) {
        float l = 0, r = 0;
        if (fmt_tag == 3) { // IEEE float
            if (bits == 32) {
                const float * f = (const float *)(raw + i);
                l = f[0];
                r = ch > 1 ? f[1] : 0.f;
            } else if (bits == 64) {
                const double * d = (const double *)(raw + i);
                l = (float)d[0];
                r = ch > 1 ? (float)d[1] : 0.f;
            } else {
                continue;
            }
        } else if (fmt_tag == 1) { // PCM
            if (bits == 8) {
                l = ((int)raw[i] - 128) / 128.f;
                r = ch > 1 ? ((int)raw[i + 1] - 128) / 128.f : 0.f;
            } else if (bits == 16) {
                int vi = (int16_t)(raw[i] | (raw[i + 1] << 8));
                l = vi / 32768.f;
                if (ch > 1) {
                    vi = (int16_t)(raw[i + 2] | (raw[i + 3] << 8));
                    r = vi / 32768.f;
                }
            } else if (bits == 24) {
                uint32_t ui = (uint32_t)raw[i] | ((uint32_t)raw[i + 1] << 8) | ((uint32_t)raw[i + 2] << 16);
                if (ui & 0x800000u) {
                    ui |= 0xFF000000u;
                }
                l = (int32_t)ui / 8388608.f;
                if (ch > 1) {
                    ui = (uint32_t)raw[i + 3] | ((uint32_t)raw[i + 4] << 8) | ((uint32_t)raw[i + 5] << 16);
                    if (ui & 0x800000u) {
                        ui |= 0xFF000000u;
                    }
                    r = (int32_t)ui / 8388608.f;
                }
            } else if (bits == 32) {
                l = (int32_t)((uint32_t)raw[i] | ((uint32_t)raw[i + 1] << 8) | ((uint32_t)raw[i + 2] << 16)
                              | ((uint32_t)raw[i + 3] << 24))
                    / 2147483648.f;
                if (ch > 1) {
                    r = (int32_t)((uint32_t)raw[i + 4] | ((uint32_t)raw[i + 5] << 8) | ((uint32_t)raw[i + 6] << 16)
                                  | ((uint32_t)raw[i + 7] << 24))
                        / 2147483648.f;
                }
            } else {
                continue;
            }
        } else {
            continue; // ADPCM etc.: the caller falls back to ffmpeg
        }
        out.push_back(ch > 1 ? (l + r) * 0.5f : l);
    }
}

} // namespace

bool mm_audio_decode_wav(const uint8_t * data, size_t len, mm_audio & out, std::string * err) {
    int rate = 0, ch = 0, bits = 0, tag = 0;
    std::vector<uint8_t> raw;
    if (!parse_wav(data, len, rate, ch, bits, tag, raw)) {
        if (err) {
            *err = "wav: not a readable WAV";
        }
        return false;
    }
    if (tag != 1 && tag != 3) {
        if (err) {
            *err = "wav: unsupported format tag (use ffmpeg)";
        }
        return false;
    }
    if (bits != 8 && bits != 16 && bits != 24 && bits != 32 && bits != 64) {
        if (err) {
            *err = "wav: unsupported bit depth";
        }
        return false;
    }
    out.samples.clear();
    convert_pcm(raw.data(), raw.size(), tag, bits, ch, out.samples);
    if (out.samples.empty()) {
        if (err) {
            *err = "wav: no audio samples";
        }
        return false;
    }
    out.sample_rate = rate;
    return true;
}

bool mm_audio_decode_mem(const uint8_t * data, size_t len, const audio_preproc_cfg & cfg, mm_audio & out,
                         std::string * err) {
    if (!data || len < 12) {
        if (err) {
            *err = "empty audio data";
        }
        return false;
    }
    mm_audio raw;
    if (!mm_audio_decode_wav(data, len, raw, err)) {
        return false;
    }
    if (raw.sample_rate != cfg.sample_rate) {
        mm_audio_resample(raw, cfg.sample_rate, out);
    } else {
        out.sample_rate = raw.sample_rate;
        out.samples = std::move(raw.samples);
    }
    // hard duration cap (protects the mel step and the model's token budget)
    const size_t cap = (size_t)cfg.sample_rate * cfg.max_seconds;
    if (out.samples.size() > cap) {
        out.samples.resize(cap);
    }
    return true;
}

bool mm_audio_decode_ffmpeg(const std::string & path, mm_audio & out, std::string * err) {
    std::string cmd = std::string(ffmpeg_cmd()) + " -v error -i \"" + path + "\" -f f32le -ac 1 -ar 16000 2> "
                      + kFfmpegErr + " pipe:1";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) {
        if (err) {
            *err = "cannot start ffmpeg";
        }
        return false;
    }
    std::vector<float> samples;
    float buf[4096];
    size_t rd = 0;
    while ((rd = std::fread(buf, sizeof(float), 4096, p)) > 0) {
        samples.insert(samples.end(), buf, buf + rd);
    }
    const int status = pclose(p);
    if (samples.empty() || status != 0) {
        std::string tail;
        FILE * ef = std::fopen(kFfmpegErr, "r");
        if (ef) {
            char tb[512];
            while (std::fgets(tb, sizeof(tb), ef) && tail.size() < 600) {
                tail += tb;
            }
            std::fclose(ef);
        }
        if (err) {
            if (samples.empty() && status == 0) {
                *err = "audio produced no samples";
            } else {
                *err = "ffmpeg failed: " + tail;
            }
        }
        return false;
    }
    out.sample_rate = 16000;
    out.samples.swap(samples);
    for (size_t i = 0; i < out.samples.size(); i++) {
        float v = out.samples[i];
        if (!std::isfinite(v)) {
            v = 0.f;
        }
        out.samples[i] = v > 1.f ? 1.f : (v < -1.f ? -1.f : v);
    }
    return true;
}

bool mm_audio_decode_bytes(const uint8_t * data, size_t len, const audio_preproc_cfg & cfg, mm_audio & out,
                           std::string * err) {
    std::string wav_err;
    if (mm_audio_decode_mem(data, len, cfg, out, &wav_err)) {
        return true;
    }
    // Non-WAV (MP3/OGG/...): spill to a temp file for the ffmpeg CLI, which can
    // only read paths.  The extension is left .tmp so ffmpeg sniffs the magic.
    const char * tmpl = "/tmp/opencode/sycl_infer_audio_XXXXXX";
    std::vector<char> path(tmpl, tmpl + std::strlen(tmpl) + 1);
    const int fd = mkstemp(path.data());
    if (fd < 0) {
        if (err) {
            *err = "cannot create temp file for audio decode";
        }
        return false;
    }
    bool ok = true;
    size_t off = 0;
    while (off < len) {
        const ssize_t w = write(fd, data + off, len - off);
        if (w <= 0) {
            ok = false;
            break;
        }
        off += (size_t)w;
    }
    close(fd);
    if (!ok) {
        std::remove(path.data());
        if (err) {
            *err = "cannot write audio temp file";
        }
        return false;
    }
    const bool decoded = mm_audio_decode_ffmpeg(path.data(), out, err);
    std::remove(path.data());
    if (!decoded) {
        // keep both failure leaves: a corrupt WAV must not masquerade as an
        // ffmpeg error, nor the other way round
        if (err && *err != "") {
            *err += " (WAV decode also failed: " + wav_err + ")";
        }
        return false;
    }
    // the ffmpeg fallback decodes at 16 kHz; resample if the config differs
    if (out.sample_rate != cfg.sample_rate) {
        mm_audio raw = out;
        mm_audio_resample(raw, cfg.sample_rate, out);
    }
    // stay within the cap
    if ((int)out.samples.size() > cfg.sample_rate * cfg.max_seconds) {
        out.samples.resize((size_t)cfg.sample_rate * cfg.max_seconds);
    }
    return true;
}

void mm_audio_resample(const mm_audio & in, int out_rate, mm_audio & out) {
    out.sample_rate = out_rate;
    if (in.sample_rate <= 0 || in.sample_rate == out_rate || in.samples.empty()) {
        out.samples = in.samples;
        return;
    }
    const double ratio = (double)in.sample_rate / out_rate;
    const size_t n = (size_t)std::ceil(in.samples.size() / ratio);
    out.samples.clear();
    out.samples.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const double src = (i + 0.5) * ratio - 0.5;
        if (src <= 0) {
            out.samples.push_back(in.samples[0]);
            continue;
        }
        const size_t i0 = (size_t)std::floor(src);
        const size_t i1 = i0 + 1 < in.samples.size() ? i0 + 1 : i0;
        const float w = (float)(src - i0);
        out.samples.push_back(in.samples[i0] * (1.f - w) + in.samples[i1] * w);
    }
}

void mm_audio_preprocess(const mm_audio & a, const audio_preproc_cfg & cfg, std::vector<float> & mel, int & n_frames) {
    const int n_fft = cfg.n_fft;
    const int hop = cfg.hop;
    const int n_mel = cfg.n_mel;
    if (n_fft <= 0 || hop <= 0 || n_mel <= 0 || a.sample_rate != cfg.sample_rate) {
        throw std::runtime_error("audio: preprocessor expects " + std::to_string(cfg.sample_rate) + " Hz input");
    }
    const size_t ns = a.samples.size();
    const size_t n_f = std::max<size_t>(1, (ns >= (size_t)n_fft ? (ns - n_fft) / hop + 1 : ns / hop));
    const int n_bins = n_fft / 2 + 1;
    const int n_bins_lim = std::min(n_bins, (int)(cfg.f_max * (double)n_fft / cfg.sample_rate) + 1);

    // Hann window
    std::vector<float> win(n_fft);
    for (int i = 0; i < n_fft; i++) {
        win[i] = 0.5f * (1.f - std::cos(2.f * (float)M_PI * i / n_fft));
    }

    // HTK mel filterbank: triangular filters over a mel-spaced center grid
    const float mel_lo = 2595.f * std::log10(1.f + cfg.f_min / 700.f);
    const float mel_hi = 2595.f * std::log10(1.f + cfg.f_max / 700.f);
    std::vector<float> centers(n_mel + 2);
    for (int m = 0; m < n_mel + 2; m++) {
        const float hz = 700.f * (std::pow(10.f, (mel_lo + (mel_hi - mel_lo) * m / (n_mel + 1)) / 2595.f) - 1.f);
        centers[m] = hz * n_fft / cfg.sample_rate;
    }
    std::vector<float> fb((size_t)n_mel * n_bins_lim, 0.f);
    for (int m = 0; m < n_mel; m++) {
        for (int k = 0; k < n_bins_lim; k++) {
            if (k < centers[m] || k > centers[m + 2]) {
                continue;
            }
            const float w = k < centers[m + 1] ? (k - centers[m]) / (centers[m + 1] - centers[m])
                                               : (centers[m + 2] - k) / (centers[m + 2] - centers[m + 1]);
            if (w > 0) {
                fb[(size_t)m * n_bins_lim + k] = w;
            }
        }
    }

    // precomputed DFT twiddles for the bins, reused by every frame
    std::vector<float> tw((size_t)(n_bins_lim * n_fft) * 2, 0.f);
    for (int k = 0; k < n_bins_lim; k++) {
        const float ang = 2.f * (float)M_PI * k / n_fft;
        float * T = tw.data() + (size_t)k * n_fft * 2;
        for (int i = 0; i < n_fft; i++) {
            T[2 * i] = std::cos(ang * i);
            T[2 * i + 1] = std::sin(ang * i);
        }
    }

    mel.assign((size_t)n_f * n_mel, 0.f);
    std::vector<float> pow_bin(n_bins_lim);
    std::vector<float> winre(n_fft, 0.f);
    for (size_t t = 0; t < n_f; t++) {
        const size_t off = t * hop;
        for (int i = 0; i < n_fft && off + i < ns; i++) {
            winre[i] = a.samples[off + i] * win[i];
        }
        for (int k = 0; k < n_bins_lim; k++) {
            float r = 0.f, ii = 0.f;
            const float * T = tw.data() + (size_t)k * n_fft * 2;
            for (int i = 0; i < n_fft; i++) {
                r += winre[i] * T[2 * i];
                ii -= winre[i] * T[2 * i + 1];
            }
            pow_bin[k] = r * r + ii * ii;
        }
        float * row = mel.data() + (size_t)t * n_mel;
        for (int m = 0; m < n_mel; m++) {
            float acc = 0.f;
            const float * ff = fb.data() + (size_t)m * n_bins_lim;
            for (int k = 0; k < n_bins_lim; k++) {
                acc += ff[k] * pow_bin[k];
            }
            row[m] = std::log(1.f + acc / cfg.floor);
        }
    }
    n_frames = (int)n_f;
}

} // namespace si