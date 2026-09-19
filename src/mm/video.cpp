#include "video.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define STBI_NO_STDIO
#define STBI_ONLY_JPEG
#include "stb/stb_image.h"

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

void rd_bytes(mem_reader & r, void * dst, size_t len) {
    if (r.p + len > r.n) {
        throw std::runtime_error("truncated AVI");
    }
    std::memcpy(dst, r.d + r.p, len);
    r.p += len;
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

// Parse the hdrl list: finds the video BITMAPINFOHEADER (strf) and the first
// avih header (for the frame rate).
struct avi_header {
    int width = 0;
    int height = 0; // negative = top-down
    int bpp = 0;
    uint32_t compression = 0;
    double fps = 0;
};

// walk the RIFF/LIST nesting; `want` is the fourcc of the LIST we are looking for
void seek_list(mem_reader & r, const char * want) {
    while (r.p + 8 <= r.n) {
        const char * id = (const char *)(r.d + r.p);
        const uint32_t size = rd_u32(r);
        if (std::memcmp(id, "LIST", 4) == 0) {
            if (r.p + 4 > r.n) {
                throw std::runtime_error("truncated AVI");
            }
            const char * type = (const char *)(r.d + r.p);
            r.p += 4;
            if (std::memcmp(type, want, 4) == 0) {
                return;
            }
            r.p += size - 4;
            if (size & 1) {
                r.p++;
            }
        } else {
            r.p += size;
            if (size & 1) {
                r.p++;
            }
        }
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
        seek_list(r, "hdrl");
        while (r.p + 8 <= r.n) {
            const char * id = (const char *)(r.d + r.p);
            const uint32_t size = rd_u32(r);
            if (std::memcmp(id, "avih", 4) == 0) {
                uint32_t uSec = rd_u32(r);
                if (uSec > 0) {
                    h.fps = 1e6 / uSec;
                }
                r.p += size - 4;
                if (size & 1) {
                    r.p++;
                }
            } else if (std::memcmp(id, "LIST", 4) == 0) {
                // nested strl list: strh tells the type, strf the format
                mem_reader sub = r;
                sub.p += 4; // list type
                const size_t end = r.p + (size_t)size - 4;
                bool is_vid = false;
                while (sub.p + 8 <= end) {
                    const char * sid = (const char *)(sub.d + sub.p);
                    const uint32_t ssz = rd_u32(sub);
                    if (std::memcmp(sid, "strh", 4) == 0 && ssz >= 12 && sub.p + 8 <= end) {
                        char fcc[5] = {0, 0, 0, 0, 0};
                        rd_bytes(sub, fcc, 4);
                        is_vid = std::memcmp(fcc, "vids", 4) == 0;
                        sub.p += ssz - 4;
                    } else if (is_vid && std::memcmp(sid, "strf", 4) == 0 && ssz >= 40) {
                        const uint32_t biSize = rd_u32(sub);
                        h.width = (int)rd_u32(sub);
                        h.height = (int)rd_u32(sub);
                        rd_u32(sub); // planes
                        h.bpp = (int)rd_u16(sub);
                        rd_u16(sub);
                        h.compression = rd_u32(sub);
                        sub.p += biSize - 24;
                        // full strh? strf only; we are done
                        return true;
                    } else {
                        sub.p += ssz;
                        if (ssz & 1) {
                            sub.p++;
                        }
                    }
                }
                r.p = end;
                if (size & 1) {
                    r.p++;
                }
            } else {
                r.p += size;
                if (size & 1) {
                    r.p++;
                }
            }
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
        seek_list(r, "hdrl");
        seek_list(r, "movi");
        const size_t movi_start = r.p;
        while (r.p + 8 <= len) {
            const uint8_t * id = data + r.p;
            const uint32_t size = rd_u32(r);
            const bool is_frame = std::memcmp(id, "00dc", 4) == 0 || std::memcmp(id, "00db", 4) == 0
                                  || std::memcmp(id, "01dc", 4) == 0 || std::memcmp(id, "01db", 4) == 0;
            if (is_frame) {
                avi_frame_loc l;
                l.off = r.p;
                l.len = size;
                l.pts = h.fps > 0 ? (double)frames.size() / h.fps : 0;
                frames.push_back(l);
                if (frames.size() >= max_frames) {
                    break;
                }
            }
            const uint64_t data_end = r.p + size;
            if (data_end > len) {
                break;
            }
            // even alignment; bump to 4 if the next fourcc is not printable
            r.p = data_end;
            if (data_end & 1) {
                r.p = data_end + 1;
            }
            if (r.p >= len) {
                break;
            }
            if (r.p + 8 <= len) {
                const char * nid = (const char *)(r.d + r.p);
                if (std::memcmp(nid, "idx1", 4) == 0) {
                    break; // index follows the movi list
                }
            }
            if (r.p + 4 <= len && !fourcc_ok(r.d + r.p)) {
                r.p = (data_end + 3) & ~(uint64_t)3;
            }
            if (r.p == (size_t)movi_start) {
                break; // no progress guard
            }
        }
        return !frames.empty();
    } catch (const std::exception &) {
        return false;
    }
}

bool decode_one(const uint8_t * data, size_t len, const avi_frame_loc & loc, const avi_header & h, mm_video_frame & f) {
    const uint8_t * fr = data + loc.off;
    const uint32_t size = loc.len;
    if (h.compression == 'MJPG' || h.compression == 'MJPD') {
        int w = 0, hh = 0, comp = 0;
        stbi_uc * px = stbi_load_from_memory(fr, (int)size, &w, &hh, &comp, 3);
        if (!px) {
            return false;
        }
        f.width = w;
        f.height = hh;
        f.rgb.assign(px, px + (size_t)w * hh * 3);
        stbi_image_free(px);
        return true;
    }
    // raw frames
    const int w = h.width, hh = h.height < 0 ? -h.height : h.height;
    const bool bottom_up = h.height > 0;
    f.width = w;
    f.height = hh;
    const size_t pix = (size_t)w * hh;
    if (h.bpp == 24) {
        if ((size_t)size < pix * 3) {
            return false;
        }
        f.rgb.assign(fr, fr + pix * 3);
    } else if (h.bpp == 16) {
        if ((size_t)size < pix * 2) {
            return false;
        }
        f.rgb.resize(pix * 3);
        for (size_t i = 0; i < pix; i++) {
            const uint16_t v = (uint16_t)fr[2 * i] | ((uint16_t)fr[2 * i + 1] << 8);
            f.rgb[3 * i + 0] = (uint8_t)(((v >> 11) & 0x1F) * 255 / 31);
            f.rgb[3 * i + 1] = (uint8_t)(((v >> 5) & 0x3F) * 255 / 63);
            f.rgb[3 * i + 2] = (uint8_t)((v & 0x1F) * 255 / 31);
        }
    } else if (h.bpp == 32) {
        if ((size_t)size < pix * 4) {
            return false;
        }
        f.rgb.resize(pix * 3);
        for (size_t i = 0; i < pix; i++) {
            f.rgb[3 * i + 0] = fr[4 * i + 2];
            f.rgb[3 * i + 1] = fr[4 * i + 1];
            f.rgb[3 * i + 2] = fr[4 * i + 0];
        }
    } else {
        return false;
    }
    if (bottom_up) {
        for (int y = 0; y < hh / 2; y++) {
            std::swap_ranges(f.rgb.begin() + (size_t)y * w * 3, f.rgb.begin() + (size_t)y * w * 3 + (size_t)w * 3,
                             f.rgb.begin() + (size_t)(hh - 1 - y) * w * 3);
        }
    }
    return true;
}

bool try_avi(const uint8_t * data, size_t len, const mm_video_fmt & fmt, mm_video & out, std::string * err) {
    avi_header h;
    if (!parse_avi_header(data, len, h) || h.width <= 0 || h.height == 0 || h.bpp == 0) {
        return false;
    }
    if (h.width > fmt.max_side || std::abs(h.height) > fmt.max_side) {
        if (err) {
            *err = "video frame too large";
        }
        return false; // recognized but rejected: do not fall back to ffmpeg
    }
    std::vector<avi_frame_loc> locs;
    if (!collect_avi_frames(data, len, h, locs, kMaxVideoDecodeFrames)) {
        if (err) {
            *err = "AVI has no readable video frames";
        }
        return false;
    }
    // uniform sample down to max_frames
    const size_t step = (locs.size() + fmt.max_frames - 1) / fmt.max_frames;
    for (size_t i = 0; i < locs.size(); i += step) {
        mm_video_frame f;
        if (decode_one(data, len, locs[i], h, f)) {
            f.pts = locs[i].pts;
            out.frames.push_back(std::move(f));
        }
        if ((int)out.frames.size() >= fmt.max_frames) {
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
const char * ffmpeg_cmd() {
    const char * v = getenv("PF_AV_FFMPEG");
    return (v && v[0]) ? v : "ffmpeg";
}

const char * ffprobe_cmd() {
    const char * v = getenv("PF_AV_FFPROBE");
    return (v && v[0]) ? v : "ffprobe";
}

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
    const std::string pcmd = std::string(ffprobe_cmd())
                             + " -v error -select_streams v:0 -show_entries stream=width,height:format=duration -of "
                               "csv=p=0 \""
                             + path + "\"";
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
    } else if (p) {
        pclose(p);
    }
    // no ffprobe: parse `ffmpeg -i` stderr
    const std::string fcmd = std::string(ffmpeg_cmd()) + " -v error -i \"" + path + "\" 2>&1";
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
    double fps = dur > 0 ? (double)fmt.max_frames / dur : 0;
    if (fps <= 0 || fps > 120) {
        if (fps > 120) {
            fps = 120;
        } else {
            fps = 10;
        }
    }
    char with_odd[64];
    std::snprintf(with_odd, sizeof(with_odd), "%.4g", fps);
    const std::string err_path = "/tmp/opencode/ffmpeg_sycl_infer_err.log";
    std::string cmd = std::string(ffmpeg_cmd())
                      + " -v error -i \"" + path
                      + "\" -an -sn -vf \"fps=" + with_odd
                      + ",scale=trunc(iw/2)*2:trunc(ih/2)*2\" -pix_fmt rgb24 -f rawvideo 2> /tmp/opencode/ffmpeg_sycl_infer_err.log "
                        "pipe:1";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) {
        if (err) {
            *err = "cannot start ffmpeg";
        }
        return false;
    }
    const size_t frame = (size_t)w * h * 3;
    std::vector<mm_video_frame> all;
    all.reserve(fmt.max_frames);
    std::vector<uint8_t> buf(frame);
    // cap the received count so a frame-rate mismatch cannot blow memory
    while ((int)all.size() < fmt.max_frames * 2 && std::fread(buf.data(), 1, frame, p) == frame) {
        mm_video_frame f;
        f.width = w;
        f.height = h;
        f.rgb = buf;
        f.pts = 0;
        all.push_back(std::move(f));
    }
    const int status = pclose(p);
    if (all.empty()) {
        std::string tail;
        FILE * ef = std::fopen("/tmp/opencode/ffmpeg_sycl_infer_err.log", "r");
        if (ef) {
            char tb[512];
            while (std::fgets(tb, sizeof(tb), ef)) {
                tail += tb;
                if (tail.size() > 600) {
                    break;
                }
            }
            std::fclose(ef);
        }
        if (err) {
            *err = "ffmpeg produced no frames" + (status != 0 ? (": " + tail) : std::string());
        }
        return false;
    }
    // the fps filter is approximate: subsample to exactly max_frames
    const int want = std::min(fmt.max_frames, (int)all.size());
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
    // not a decodable AVI (or failed): hand the bytes to ffmpeg
    std::string path = "/tmp/opencode/video_input_" + std::to_string((uintptr_t)data) + ".bin";
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        if (err) {
            *err = "cannot write video temp file";
        }
        return false;
    }
    std::fwrite(data, 1, len, f);
    std::fclose(f);
    const bool ok = decode_ffmpeg(path, fmt, out, err);
    std::remove(path.c_str());
    return ok;
}

bool mm_video_decode_file(const std::string & path, const mm_video_fmt & fmt, mm_video & out, std::string * err) {
    out.frames.clear();
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