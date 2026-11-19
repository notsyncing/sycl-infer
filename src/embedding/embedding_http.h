#pragma once
// OpenAI-compatible embeddings endpoint for GGUF embedding models.  It is a
// peer of the autoregressive server, not a route inside it: an encoder has no
// engine/KV pool, so `main` picks this server when the GGUF architecture is
// `bert`.  The wire format follows /v1/embeddings (float arrays only).

#include <string>

#include "embedding_engine.h"

namespace si {
namespace embedding {

struct server_config {
    std::string model_id = "sycl-infer-embedding";
    std::string host = "127.0.0.1";
    int port = 8080;
};

int serve(engine & e, const server_config & cfg);

} // namespace embedding
} // namespace si
