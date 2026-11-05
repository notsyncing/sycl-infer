#pragma once
// The worker's pipeline endpoints and the model list they must not leak into.
//
// Factored out of server.cpp so both halves are testable with a fake registry
// and no engine: the handlers only need the registry, the model profile (for
// availability) and the model id.
#include <string>

namespace httplib {
class Server;
}

namespace si {

struct model_profile;
class pipeline_registry;

struct pipeline_http_config {
    std::string model_id;          // the id advertised by the GGUF general.name
    std::string fallback_model_id; // the CLI/server default id
    std::string model_entry_json;  // the /v1/models entry, already serialized
};

// GET /v1/models and GET /v1/models/{id}, unchanged from the pre-pipeline
// server: models only, never a pipeline id.
void register_model_list_http(httplib::Server & srv, const pipeline_http_config & cfg);

// GET /v1/pipelines and POST /v1/pipelines/{id}/run.
// `prof` may be null (no engine context); availability is then evaluated
// against a default model profile.
void register_pipeline_http(httplib::Server & srv, pipeline_registry & registry, const model_profile * prof,
                            const std::string & model_id);

// The body of GET /v1/pipelines.  Exposed for the pure-JSON test.
std::string pipelines_json(pipeline_registry & registry, const model_profile * prof);

} // namespace si
