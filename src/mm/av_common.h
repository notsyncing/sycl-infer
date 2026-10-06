#pragma once

#include <cstdio>
#include <string>

#include "common/env.h"

namespace si {

// ffmpeg/ffprobe selection.  Both decoder towers used to carry their own copy.
inline const char * av_ffmpeg_cmd() {
    const char * v = si::env::str("PF_AV_FFMPEG");
    return (v && v[0]) ? v : "ffmpeg";
}

inline const char * av_ffprobe_cmd() {
    const char * v = si::env::str("PF_AV_FFPROBE");
    return (v && v[0]) ? v : "ffprobe";
}

// the ffmpeg subprocess stderr is redirected to this shared log, whose tail
// each decode error wraps.
inline const char * av_ffmpeg_err_log() {
    return "/tmp/opencode/ffmpeg_sycl_infer_err.log";
}

inline std::string av_err_tail() {
    std::string tail;
    FILE * ef = std::fopen(av_ffmpeg_err_log(), "r");
    if (ef) {
        char tb[512];
        while (std::fgets(tb, sizeof(tb), ef) && tail.size() < 600) {
            tail += tb;
        }
        std::fclose(ef);
    }
    return tail;
}

} // namespace si
