#pragma once
// Request-level pipelines: multi-step orchestration that has to happen inside
// the worker process (shared KV/prefix state, a verify/rollback cycle, the
// engine's image-embedding buffer, ...) rather than as several round trips
// over the OpenAI API.
//
// This layer is deliberately independent of httplib and of src/server/chat.h:
// the HTTP front end converts its own request types into these neutral ones.
// It may depend on the engine's public value types (gen_params) and on the
// model profile, but not on an engine object - `available()` takes the profile
// directly, which is exactly the state the design says availability may read
// and is what lets the registry and its tests run without a GGUF or a device.
//
// There is deliberately no built-in pipeline.  MinerU is a client/reference
// tool under tools/mineru/, not a pipeline: its steps use only the public
// image-input/text-output capability.
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "model_profile.h"
#include "sampler.h" // gen_params

namespace si {

struct pipeline_desc {
    const char * id = "";
    const char * name = "";
    model_profile::vision_kind requires_vision = model_profile::vision_kind::none;
    bool accepts_image = false;
    bool accepts_video = false;
    bool accepts_audio = false;
    const char * output_formats = "json"; // comma separated: "json,markdown,text"
    const char * options_schema = "";     // optional JSON-schema text
};

enum class pipeline_part_kind { text, image, video, audio };

// One content part of a pipeline request.  Media is already downloaded by the
// server (or supplied inline); `data` holds the raw bytes when known and `url`
// keeps the transport string for a pipeline that wants to re-read it.
struct pipeline_part {
    pipeline_part_kind kind = pipeline_part_kind::text;
    std::string text;
    std::string data;
    std::string format;
    std::string url;
};

struct pipeline_message {
    std::string role;
    std::string content;
    std::vector<pipeline_part> parts;
};

struct pipeline_media {
    pipeline_part_kind kind = pipeline_part_kind::image;
    std::string data;
    std::string format;
    std::string url;
};

struct pipeline_request {
    std::string input_text;
    std::vector<pipeline_message> messages;
    std::vector<pipeline_media> media;
    gen_params params;
    std::string options_json;
    bool stream = false;

    bool has_image() const {
        for (const pipeline_media & m : media) {
            if (m.kind == pipeline_part_kind::image) {
                return true;
            }
        }
        return false;
    }
    bool has_video() const {
        for (const pipeline_media & m : media) {
            if (m.kind == pipeline_part_kind::video) {
                return true;
            }
        }
        return false;
    }
    bool has_audio() const {
        for (const pipeline_media & m : media) {
            if (m.kind == pipeline_part_kind::audio) {
                return true;
            }
        }
        return false;
    }
};

struct pipeline_usage {
    long long prompt_tokens = 0;
    long long completion_tokens = 0;
    long long reasoning_tokens = 0;
    long long cached_tokens = 0;
};

// The producer side of a run.  `on_text` is the content delta; `on_done` is
// called once with the final usage and finish reason.  `cancelled` is the
// optional disconnect/cancel probe a long pipeline should consult between
// steps (the HTTP layer sets it; a synchronous caller leaves it empty).
struct pipeline_sink {
    std::function<void(std::string_view)> on_text;
    std::function<void()> on_progress;
    std::function<void(const pipeline_usage &, std::string finish_reason)> on_done;
    std::function<bool()> cancelled;
};

class pipeline {
public:
    virtual ~pipeline() = default;
    virtual const pipeline_desc & desc() const = 0;
    virtual bool available(const model_profile & prof) const = 0;
    virtual void run(const pipeline_request & request, pipeline_sink & sink) = 0;
};

} // namespace si
