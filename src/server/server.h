#pragma once
#include <string>

#include "engine.h"

namespace si {

struct server_config {
    std::string model_id = "qwen3.5-0.8b";
    std::string host = "0.0.0.0";
    int port = 8080;
    int n_threads = 4;
};

int serve(engine & e, const server_config & cfg);

} // namespace si
