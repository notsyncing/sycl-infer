#pragma once
#include <string>

#include "engine.h"

namespace si {

struct server_config {
    std::string model_id = "qwen3.5-0.8b";
    // loopback by default; the API is unauthenticated, so a routable bind
    // address exposes the engine to anyone who can reach the port
    std::string host = "127.0.0.1";
    int port = 8080;
    int n_threads = 4;
    // vision projector GGUF; empty disables image input
    std::string mmproj_path;
    // audio tower GGUF; empty disables audio input
    std::string audio_mmproj_path;
    // video encode limits
    int max_video_frames = 16;
    int max_video_side = 768;
};

int serve(engine & e, const server_config & cfg);

} // namespace si
