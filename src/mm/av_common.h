#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

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

// Use the system's configured temporary directory rather than a directory
// specific to a developer's machine.  The caller owns and closes the fd.
inline int av_make_temp_file(const char * kind, std::string & path) {
    std::error_code ec;
    const auto dir = std::filesystem::temp_directory_path(ec);
    if (ec) {
        return -1;
    }
    const std::string pattern = (dir / (std::string("sycl_infer_") + kind + "_XXXXXX")).string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int fd = mkstemp(writable.data());
    if (fd >= 0) {
        path = writable.data();
    }
    return fd;
}

// Files passed to the shell-based ffmpeg CLI may contain quotes or shell
// metacharacters.  Shell-quote each path as one argument, including the temp
// stderr file (the executable override itself is an operator-controlled knob).
inline std::string av_shell_quote(const std::string & path) {
    std::string out = "'";
    for (char c : path) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

// Every decode gets its own stderr file: a shared path races when two HTTP
// requests decode concurrently and can attach the wrong request's error.
struct av_error_log {
    std::string path;
    av_error_log() {
        const int fd = av_make_temp_file("ffmpeg_err", path);
        if (fd >= 0) {
            close(fd);
        }
    }
    ~av_error_log() {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }
    av_error_log(const av_error_log &) = delete;
    av_error_log & operator=(const av_error_log &) = delete;
};

inline std::string av_err_tail(const std::string & path) {
    std::string tail;
    FILE * ef = std::fopen(path.c_str(), "r");
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
