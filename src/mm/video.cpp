#include "video.h"
#include "av_common.h"

#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define STBI_NO_STDIO
#define STBI_ONLY_JPEG
#include "stb/stb_image.h"
#include "common/env.h"

namespace si {

namespace {

// ---------------------------------------------------------------------------
// Minimal RIFF/AVI demuxer.  Covers the containers the reference clips use:
// a video stream of MJPEG (`biCompression == 'MJPG'`) or raw BI_RGB / 565 /
// 32-bit frames (`'00dc'` / `'00db'` chunks).  Returns only the frame offsets;
// the sampled frames are decoded with stb_image later.
// ---------------------------------------------------------------------------
struct avi_frame_loc {
    uint64_t off = 0;
    uint32_t len = 0;
    double pts = 0;
};

struct mem_reader {
    const uint8_t * d;
    size_t n;
    size_t p = 0;
};

uint32_t rd_u32(mem_reader & r) {
    if (r.p + 4 > r.n) {
        throw std::runtime_error("truncated AVI");
    }
    uint32_t v = (uint32_t)r.d[r.p] | ((uint32_t)r.d[r.p + 1] << 8) | ((uint32_t)r.d[r.p + 2] << 16)
                 | ((uint32_t)r.d[r.p + 3] << 24);
    r.p += 4;
    return v;
}

uint32_t rd_u16(mem_reader & r) {
    if (r.p + 2 > r.n) {
        throw std::runtime_error("truncated AVI");
    }
    uint32_t v = (uint32_t)r.d[r.p] | ((uint32_t)r.d[r.p + 1] << 8);
    r.p += 2;
    return v;
}

// a fourcc is printable ASCII (used to detect chunk alignment padding)
bool fourcc_ok(const uint8_t * c) {
    for (int i = 0; i < 4; i++) {
        if (c[i] < 0x20 || c[i] > 0x7e) {
            return false;
        }
    }
    return true;
}

// biCompression is read little-endian, so a fourcc literal has to be built the
// same way ('MJPG' as a multi-character literal is byte-swapped on x86)
constexpr uint32_t fcc(char a, char b, char c, char d) {
    return (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) | ((uint32_t)(uint8_t)c << 16)
           | ((uint32_t)(uint8_t)d << 24);
}

bool is_mjpeg(uint32_t comp) {
    return comp == fcc('M', 'J', 'P', 'G') || comp == fcc('m', 'j', 'p', 'g') || comp == fcc('M', 'J', 'P', 'D')
           || comp == fcc('m', 'j', 'p', 'd');
}

// One RIFF chunk header: `id` points at the fourcc in the buffer, `size` is
// the chunk's *data* size read from the four bytes after it, [body, end) is the
// data (end may lie past the parent's limit when the file is truncated or the
// size is bogus: callers must check).
struct riff_chunk {
    const uint8_t * id = nullptr;
    uint32_t size = 0;
    uint64_t body = 0;
    uint64_t end = 0;
};

// read the 8-byte header at r.p (fourcc, then size); false when fewer than 8
// bytes remain before `limit`.  Leaves r.p at the body.
bool next_chunk(mem_reader & r, size_t limit, riff_chunk & c) {
    if (limit > r.n || r.p > limit || limit - r.p < 8) {
        return false;
    }
    c.id = r.d + r.p;
    r.p += 4; // the fourcc comes first ...
    c.size = rd_u32(r); // ... then the 32-bit little-endian size
    c.body = r.p;
    c.end = c.body + (uint64_t)c.size;
    return true;
}

// start of the sibling after `c` (chunks are padded to an even length)
size_t chunk_next(const riff_chunk & c) {
    return (size_t)(c.end + (c.size & 1));
}

// Parse the hdrl list: finds the video BITMAPINFOHEADER (strf) and the first
// avih header (for the frame rate).
struct avi_header {
    int width = 0;
    int height = 0; // negative = top-down
    int bpp = 0;
    uint32_t compression = 0;
    double fps = 0;
    int stream = 0; // index of the video stream (its chunks are "<NN>dc"/"<NN>db")
};

// Walk the chunks in [r.p, limit) for the LIST whose type is `want`.  Returns
// the end of that list's data (clamped to `limit`, so a truncated file is still
// usable) with r.p just past the list type; throws when it is not found.
size_t seek_list(mem_reader & r, const char * want, size_t limit) {
    riff_chunk c;
    while (next_chunk(r, limit, c)) {
        if (std::memcmp(c.id, "LIST", 4) == 0 && c.size >= 4 && c.body + 4 <= limit
            && std::memcmp(r.d + c.body, want, 4) == 0) {
            r.p = (size_t)c.body + 4;
            return (size_t)std::min<uint64_t>(c.end, limit);
        }
        if (c.end > limit) {
            break; // a chunk that runs past the buffer cannot be skipped
        }
        r.p = chunk_next(c);
    }
    throw std::runtime_error("AVI has no " + std::string(want) + " list");
}

bool parse_avi_header(const uint8_t * data, size_t len, avi_header & h) {
    mem_reader r{data, len};
    try {
        if (len < 12 || std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "AVI ", 4) != 0) {
            return false;
        }
        r.p = 12;
        const size_t hdrl_end = seek_list(r, "hdrl", len);
        int strl_idx = 0;
        riff_chunk c;
        while (next_chunk(r, hdrl_end, c)) {
            if (c.end > hdrl_end) {
                break;
            }
            if (std::memcmp(c.id, "avih", 4) == 0 && c.size >= 4) {
                mem_reader s{data, (size_t)c.end, (size_t)c.body};
                const uint32_t uSec = rd_u32(s);
                if (uSec > 0) {
                    h.fps = 1e6 / uSec;
                }
            } else if (std::memcmp(c.id, "LIST", 4) == 0 && c.size >= 4 && std::memcmp(data + c.body, "strl", 4) == 0) {
                // nested strl list: strh tells the type, strf the format
                mem_reader sub{data, (size_t)c.end, (size_t)c.body + 4};
                bool is_vid = false;
                riff_chunk sc;
                while (next_chunk(sub, (size_t)c.end, sc)) {
                    if (sc.end > c.end) {
                        break;
                    }
                    if (std::memcmp(sc.id, "strh", 4) == 0 && sc.size >= 4) {
                        is_vid = std::memcmp(data + sc.body, "vids", 4) == 0;
                    } else if (is_vid && std::memcmp(sc.id, "strf", 4) == 0 && sc.size >= 40) {
                        mem_reader f{data, (size_t)sc.end, (size_t)sc.body};
                        rd_u32(f); // biSize
                        h.width = (int)rd_u32(f);
                        h.height = (int)rd_u32(f);
                        rd_u16(f); // planes
                        h.bpp = (int)rd_u16(f);
                        h.compression = rd_u32(f);
                        h.stream = strl_idx;
                        return true;
                    }
                    sub.p = chunk_next(sc);
                }
                strl_idx++;
            }
            r.p = chunk_next(c);
        }
    } catch (const std::exception &) {
        return false;
    }
    return false;
}

// scan the movi list and collect the video frame locations
bool collect_avi_frames(const uint8_t * data, size_t len, const avi_header & h, std::vector<avi_frame_loc> & frames,
                        size_t max_frames) {
    mem_reader r{data, len};
    try {
        r.p = 12;
        const size_t movi_end = seek_list(r, "movi", len);
        // chunk ids of the video stream are two decimal digits + "dc"/"db"
        const char s0 = (char)('0' + (h.stream / 10) % 10);
        const char s1 = (char)('0' + h.stream % 10);
        riff_chunk c;
        while (next_chunk(r, movi_end, c)) {
            if (std::memcmp(c.id, "LIST", 4) == 0) {
                // "rec " grouping lists: descend, the frames inside are ordinary chunks
                if (c.size < 4 || c.body + 4 > movi_end) {
                    break;
                }
                r.p = (size_t)c.body + 4;
                continue;
            }
            if (c.end > movi_end) {
                break; // truncated chunk: its data is not all there
            }
            const bool is_frame = c.id[0] == (uint8_t)s0 && c.id[1] == (uint8_t)s1 && c.id[2] == 'd'
                                  && (c.id[3] == 'c' || c.id[3] == 'b');
            if (is_frame) {
                avi_frame_loc l;
                l.off = c.body;
                l.len = c.size;
                l.pts = h.fps > 0 ? (double)frames.size() / h.fps : 0;
                frames.push_back(l);
                if (frames.size() >= max_frames) {
                    break;
                }
            }
            // even alignment; bump to 4 if the next fourcc is not printable
            const uint64_t padded = chunk_next(c);
            r.p = (size_t)padded;
            if (r.p >= movi_end) {
                break;
            }
            if (movi_end - r.p >= 4 && std::memcmp(r.d + r.p, "idx1", 4) == 0) {
                break; // index follows the movi list
            }
            if (movi_end - r.p >= 4 && !fourcc_ok(r.d + r.p)) {
                r.p = (size_t)((c.end + 3) & ~(uint64_t)3);
            }
        }
        return !frames.empty();
    } catch (const std::exception &) {
        return false;
    }
}

bool decode_one(const uint8_t * data, size_t len, const avi_frame_loc & loc, const avi_header & h,
                const mm_video_fmt & fmt, mm_video_frame & f) {
    if (loc.off > len || loc.len > len - loc.off) {
        return false;
    }
    const uint8_t * fr = data + loc.off;
    const uint32_t size = loc.len;
    if (is_mjpeg(h.compression)) {
        if (size == 0 || size > (uint32_t)INT_MAX) {
            return false;
        }
        int w = 0, hh = 0, comp = 0;
        stbi_uc * px = stbi_load_from_memory(fr, (int)size, &w, &hh, &comp, 3);
        if (!px) {
            return false;
        }
        if (w > fmt.max_side || hh > fmt.max_side) {
            stbi_image_free(px);
            return false;
        }
        f.width = w;
        f.height = hh;
        f.rgb.assign(px, px + (size_t)w * hh * 3);
        stbi_image_free(px);
        return true;
    }
    // raw frames: DIB rows are padded to a multiple of 4 bytes
    const int w = h.width, hh = h.height < 0 ? -h.height : h.height;
    const bool bottom_up = h.height > 0;
    if (w <= 0 || hh <= 0 || (h.bpp != 16 && h.bpp != 24 && h.bpp != 32)) {
        return false;
    }
    const size_t bpx = (size_t)h.bpp / 8;
    const size_t stride = ((size_t)w * h.bpp + 31) / 32 * 4;
    if ((size_t)size < stride * (size_t)hh) {
        return false;
    }
    f.width = w;
    f.height = hh;
    f.rgb.resize((size_t)w * hh * 3);
    // BI_RGB 16-bit is 5-5-5.  BI_BITFIELDS is left to ffmpeg because its
    // masks can vary and this parser does not read/validate them.
    for (int y = 0; y < hh; y++) {
        const uint8_t * row = fr + (size_t)(bottom_up ? hh - 1 - y : y) * stride;
        uint8_t * dst = f.rgb.data() + (size_t)y * w * 3;
        for (int x = 0; x < w; x++) {
            const uint8_t * px = row + (size_t)x * bpx;
            if (h.bpp == 24) {
                dst[3 * x + 0] = px[2]; // DIB pixels are BGR
                dst[3 * x + 1] = px[1];
                dst[3 * x + 2] = px[0];
            } else if (h.bpp == 16) {
                const uint16_t v = (uint16_t)px[0] | ((uint16_t)px[1] << 8);
                dst[3 * x + 0] = (uint8_t)(((v >> 10) & 0x1F) * 255 / 31);
                dst[3 * x + 1] = (uint8_t)(((v >> 5) & 0x1F) * 255 / 31);
                dst[3 * x + 2] = (uint8_t)((v & 0x1F) * 255 / 31);
            } else {
                dst[3 * x + 0] = px[2];
                dst[3 * x + 1] = px[1];
                dst[3 * x + 2] = px[0];
            }
        }
    }
    return true;
}

bool try_avi(const uint8_t * data, size_t len, const mm_video_fmt & fmt, mm_video & out, std::string * err) {
    avi_header h;
    if (!parse_avi_header(data, len, h) || h.width <= 0 || h.height == 0 || h.height == INT_MIN || h.bpp == 0) {
        return false;
    }
    if (fmt.max_frames <= 0) {
        if (err) {
            *err = "max_frames must be positive";
        }
        return false;
    }
    if (h.width > fmt.max_side || std::abs(h.height) > fmt.max_side) {
        if (err) {
            *err = "video frame too large";
        }
        return false; // recognized but rejected (ffmpeg's probe applies the same limit)
    }
    // MJPEG or uncompressed BI_RGB 16/24/32-bit; formats with explicit channel
    // masks (BI_BITFIELDS) need ffmpeg unless their masks are parsed and checked.
    const bool raw_ok = h.compression == 0 && (h.bpp == 16 || h.bpp == 24 || h.bpp == 32);
    if (!is_mjpeg(h.compression) && !raw_ok) {
        if (err) {
            *err = "unsupported AVI video codec";
        }
        return false;
    }
    std::vector<avi_frame_loc> locs;
    if (!collect_avi_frames(data, len, h, locs, kMaxVideoDecodeFrames)) {
        if (err) {
            *err = "AVI has no readable video frames";
        }
        return false;
    }
    // uniform sample down to max_frames
    const size_t max_frames = (size_t)std::min(fmt.max_frames, kMaxVideoDecodeFrames);
    const size_t step = (locs.size() + max_frames - 1) / max_frames;
    for (size_t i = 0; i < locs.size(); i += step) {
        mm_video_frame f;
        if (decode_one(data, len, locs[i], h, fmt, f)) {
            f.pts = locs[i].pts;
            out.frames.push_back(std::move(f));
        }
        if (out.frames.size() >= max_frames) {
            break;
        }
    }
    if (out.frames.empty()) {
        if (err) {
            *err = "failed to decode any AVI video frame";
        }
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// ffmpeg subprocess decode.  The clip is probed once for its geometry and
// duration, then piped as raw rgb24 with an `fps` filter that produces about
// `max_frames` uniformly spaced frames.
// ---------------------------------------------------------------------------
namespace {
// parse "Stream #0:0: Video: ... 854x480 [SAR 1:1 DAR 16:9]" and
// "Duration: 01:29:45.07" from `ffmpeg -i` stderr (ffprobe-less fallback)
void parse_ffmpeg_info(const std::string & text, int & w, int & h, double & dur) {
    size_t pos = 0;
    while ((pos = text.find("Stream #", pos)) != std::string::npos) {
        size_t end = text.find('\n', pos);
        const std::string line = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        pos = end;
        if (line.find(": Video:") == std::string::npos) {
            continue;
        }
        // the resolution is the last "<digits>x<digits>" token in the line
        size_t xs = line.rfind('x');
        if (xs == std::string::npos) {
            continue;
        }
        size_t lstart = xs;
        while (lstart > 0 && line[lstart - 1] >= '0' && line[lstart - 1] <= '9') {
            lstart--;
        }
        size_t hv = xs + 1;
        size_t endv = hv;
        while (endv < line.size() && line[endv] >= '0' && line[endv] <= '9') {
            endv++;
        }
        if (lstart < xs && endv > hv) {
            w = atoi(line.c_str() + lstart);
            h = atoi(line.c_str() + hv);
            break;
        }
    }
    size_t d = text.find("Duration:");
    if (d != std::string::npos) {
        d += 9;
        int hh = 0, mm = 0;
        double ss = 0;
        if (sscanf(text.c_str() + d, "%d:%d:%lf", &hh, &mm, &ss) == 3) {
            dur = hh * 3600 + mm * 60 + ss;
        }
    }
}
} // namespace

bool probe_video(const std::string & path, int & w, int & h, double & dur) {
    // ffprobe prints "width,height,duration" on one line with -of csv
    const std::string pcmd = std::string(av_ffprobe_cmd())
                             + " -v error -select_streams v:0 -show_entries stream=width,height:format=duration -of "
                               "csv=p=0 "
                             + av_shell_quote(path);
    FILE * p = popen(pcmd.c_str(), "r");
    if (p) {
        char line[256] = {0};
        if (fgets(line, sizeof(line), p)) {
            pclose(p);
            int fields = 0;
            char * it = line;
            char * save = nullptr;
            while (fields < 3) {
                char * tok = strtok_r(it, ",;\n", &save);
                it = nullptr;
                if (!tok) {
                    break;
                }
                if (fields == 0) {
                    w = atoi(tok);
                } else if (fields == 1) {
                    h = atoi(tok);
                } else {
                    dur = atof(tok);
                }
                fields++;
            }
            if (fields == 3 && w > 0 && h > 0) {
                return true;
            }
        } else {
            pclose(p);
        }
    }
    // no ffprobe: parse `ffmpeg -i` stderr
    const std::string fcmd = std::string(av_ffmpeg_cmd()) + " -v error -i " + av_shell_quote(path) + " 2>&1";
    FILE * fp = popen(fcmd.c_str(), "r");
    if (!fp) {
        return false;
    }
    std::string text;
    char buf[1024];
    size_t got;
    while ((got = fread(buf, 1, sizeof(buf), fp)) > 0) {
        text.append(buf, got);
    }
    pclose(fp);
    parse_ffmpeg_info(text, w, h, dur);
    return w > 0 && h > 0;
}

bool decode_ffmpeg(const std::string & path, const mm_video_fmt & fmt, mm_video & out, std::string * err) {
    if (fmt.max_frames <= 0) {
        if (err) {
            *err = "max_frames must be positive";
        }
        return false;
    }
    const int max_frames = std::min(fmt.max_frames, kMaxVideoDecodeFrames);
    int w = 0, h = 0;
    double dur = 0;
    if (!probe_video(path, w, h, dur) || (w > fmt.max_side || h > fmt.max_side)) {
        if (err) {
            *err = "cannot probe video with ffmpeg" + std::string(w > fmt.max_side || h > fmt.max_side
                                                                     ? " (frame too large)"
                                                                     : " (is ffmpeg installed?)");
        }
        return false;
    }
    // The filter below forces even sides, so the raw frames are not w x h: the
    // output geometry is computed here and passed to ffmpeg explicitly (a
    // 1-pixel side would otherwise truncate to 0 and fail), so the rawvideo
    // stream is read with exactly the size it was written with.
    const int ow = w < 2 ? 2 : (w & ~1);
    const int oh = h < 2 ? 2 : (h & ~1);
    if (ow > fmt.max_side || oh > fmt.max_side) {
        if (err) {
            *err = "cannot probe video with ffmpeg (frame too large)";
        }
        return false;
    }
    double fps = dur > 0 ? (double)max_frames / dur : 0;
    if (fps <= 0 || fps > 120) {
        if (fps > 120) {
            fps = 120;
        } else {
            fps = 10;
        }
    }
    char with_odd[64];
    std::snprintf(with_odd, sizeof(with_odd), "%.4g", fps);
    // -noautorotate: ffprobe reports the coded geometry, so a display-matrix
    // rotation (which would swap the output sides) must not be applied here
    av_error_log log;
    if (log.path.empty()) {
        if (err) {
            *err = "cannot create ffmpeg error log";
        }
        return false;
    }
    std::string cmd = std::string(av_ffmpeg_cmd())
                      + " -v error -noautorotate -i " + av_shell_quote(path)
                      + " -an -sn -vf \"fps=" + with_odd
                      + ",scale=" + std::to_string(ow) + ":" + std::to_string(oh)
                      + "\" -pix_fmt rgb24 -f rawvideo 2> " + av_shell_quote(log.path) + " pipe:1";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) {
        if (err) {
            *err = "cannot start ffmpeg";
        }
        return false;
    }
    const size_t frame = (size_t)ow * oh * 3;
    std::vector<mm_video_frame> all;
    all.reserve(max_frames);
    std::vector<uint8_t> buf(frame);
    // cap the received count so a frame-rate mismatch cannot blow memory
    while ((int)all.size() < max_frames * 2 && std::fread(buf.data(), 1, frame, p) == frame) {
        mm_video_frame f;
        f.width = ow;
        f.height = oh;
        f.rgb = buf;
        f.pts = 0;
        all.push_back(std::move(f));
    }
    const int status = pclose(p);
    if (all.empty()) {
        std::string tail = av_err_tail(log.path);
        if (err) {
            *err = "ffmpeg produced no frames" + (status != 0 ? (": " + tail) : std::string());
        }
        return false;
    }
    // the fps filter is approximate: subsample to exactly max_frames
    const int want = std::min(max_frames, (int)all.size());
    const size_t step = (all.size() + want - 1) / want;
    for (size_t i = 0; i < all.size() && (int)out.frames.size() < want; i += step) {
        out.frames.push_back(std::move(all[i]));
    }
    return !out.frames.empty();
}

} // namespace

void mm_video_subsample(const std::vector<mm_video_frame> & src, int n, std::vector<const mm_video_frame *> & out) {
    out.clear();
    if (src.empty() || n <= 0) {
        return;
    }
    if ((int)src.size() <= n) {
        out.reserve(src.size());
        for (const auto & f : src) {
            out.push_back(&f);
        }
        return;
    }
    const size_t step = (src.size() + n - 1) / n;
    for (size_t i = 0; i < src.size() && (int)out.size() < n; i += step) {
        out.push_back(&src[i]);
    }
}

bool mm_video_decode_mem(const uint8_t * data, size_t len, const mm_video_fmt & fmt, mm_video & out, std::string * err) {
    out.frames.clear();
    if (fmt.max_frames <= 0) {
        if (err) {
            *err = "max_frames must be positive";
        }
        return false;
    }
    if (!data || len < 12) {
        if (err) {
            *err = "empty video data";
        }
        return false;
    }
    // AVI/MJPEG is decoded natively; anything else needs the ffmpeg CLI
    if (std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "AVI ", 4) == 0 && try_avi(data, len, fmt, out, err)) {
        return true;
    }
    // not a decodable AVI (or failed): hand the bytes to ffmpeg via a tmp file
    std::string path;
    const int fd = av_make_temp_file("video", path);
    if (fd < 0) {
        if (err) {
            *err = "cannot write video temp file";
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
        std::remove(path.c_str());
        if (err) {
            *err = "cannot write video temp file";
        }
        return false;
    }
    const bool decoded = decode_ffmpeg(path, fmt, out, err);
    std::remove(path.c_str());
    return decoded;
}

bool mm_video_decode_file(const std::string & path, const mm_video_fmt & fmt, mm_video & out, std::string * err) {
    out.frames.clear();
    if (fmt.max_frames <= 0) {
        if (err) {
            *err = "max_frames must be positive";
        }
        return false;
    }
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) {
            *err = "cannot open " + path;
        }
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        std::fclose(f);
        if (err) {
            *err = "cannot size " + path;
        }
        return false;
    }
    std::vector<uint8_t> buf((size_t)sz);
    const size_t rd = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (rd != buf.size()) {
        if (err) {
            *err = "short read " + path;
        }
        return false;
    }
    if (buf.size() >= 12 && std::memcmp(buf.data(), "RIFF", 4) == 0 && std::memcmp(buf.data() + 8, "AVI ", 4) == 0
        && try_avi(buf.data(), buf.size(), fmt, out, err)) {
        return true;
    }
    return decode_ffmpeg(path, fmt, out, err);
}

} // namespace si