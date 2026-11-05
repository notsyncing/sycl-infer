#include "pipeline_http.h"

#include "chat.h"
#include "httplib.h"
#include "json.hpp"
#include "json_util.h"
#include "media_fetch.h"
#include "pipeline/pipeline.h"
#include "pipeline/registry.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace si {

namespace {

using json = nlohmann::json;

void cors(httplib::Response & res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Headers", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
}

void set_json(httplib::Response & res, int status, const json & j) {
    res.status = status;
    res.set_content(dump_json(j), "application/json");
}

void set_error(httplib::Response & res, int status, const std::string & msg, const std::string & type) {
    set_json(res, status, {{"error", {{"message", msg}, {"type", type}}}});
}

std::string gen_pipeline_id() {
    static std::atomic<uint64_t> counter{0};
    return "pipeline-" + std::to_string((uint64_t)time(nullptr)) + "-" + std::to_string(++counter);
}

std::string first_format(const pipeline_desc & d) {
    std::string s = d.output_formats ? d.output_formats : "";
    const size_t comma = s.find(',');
    if (comma != std::string::npos) {
        s.resize(comma);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    return s.empty() ? "json" : s;
}

json json_formats(const pipeline_desc & d) {
    json out = json::array();
    std::string s = d.output_formats ? d.output_formats : "";
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) {
            tok.erase(tok.begin());
        }
        while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t')) {
            tok.pop_back();
        }
        if (!tok.empty()) {
            out.push_back(tok);
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return out;
}

json json_inputs(const pipeline_desc & d) {
    json out = json::array({"text"});
    if (d.accepts_image) {
        out.push_back("image");
    }
    if (d.accepts_video) {
        out.push_back("video");
    }
    if (d.accepts_audio) {
        out.push_back("audio");
    }
    return out;
}

json usage_json(const pipeline_usage & u) {
    json j = {{"prompt_tokens", u.prompt_tokens},
              {"completion_tokens", u.completion_tokens},
              {"total_tokens", u.prompt_tokens + u.completion_tokens}};
    if (u.reasoning_tokens > 0) {
        j["completion_tokens_details"] = {{"reasoning_tokens", u.reasoning_tokens}};
    }
    if (u.cached_tokens > 0) {
        j["prompt_tokens_details"] = {{"cached_tokens", u.cached_tokens}};
    }
    return j;
}

std::string mime_from_data_url(const std::string & url) {
    if (url.rfind("data:", 0) != 0) {
        return std::string();
    }
    const size_t semi = url.find(';', 5);
    if (semi == std::string::npos || semi <= 5) {
        return std::string();
    }
    return url.substr(5, semi - 5);
}

bool load_image(const std::string & value, pipeline_media & out, std::string & err) {
    media_part mp;
    mp.kind = chat_part_kind::IMAGE;
    if (value.rfind("data:", 0) == 0 || value.rfind("http://", 0) == 0 || value.rfind("https://", 0) == 0) {
        mp.url = value;
    } else {
        mp.data = value; // inline base64
    }
    std::vector<uint8_t> bytes;
    if (!load_media_bytes(mp, bytes, err)) {
        err = "image: " + err;
        return false;
    }
    out.kind = pipeline_part_kind::image;
    out.url = value;
    out.format = mime_from_data_url(value);
    out.data.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    return true;
}

struct run_input {
    pipeline_request request;
    bool has_image = false;
    bool has_video = false;
    bool has_audio = false;
};

bool parse_run_body(const std::string & body, run_input & in, std::string & err) {
    json j;
    try {
        j = json::parse(body);
    } catch (...) {
        err = "invalid json";
        return false;
    }
    if (!j.is_object()) {
        err = "request body must be a JSON object";
        return false;
    }
    in.request.stream = j.value("stream", false);

    const auto input_it = j.find("input");
    if (input_it != j.end() && !input_it->is_null()) {
        if (!input_it->is_object()) {
            err = "'input' must be an object";
            return false;
        }
        const auto text_it = input_it->find("text");
        if (text_it != input_it->end() && !text_it->is_null()) {
            if (!text_it->is_string()) {
                err = "'input.text' must be a string";
                return false;
            }
            in.request.input_text = text_it->get<std::string>();
        }
        std::string image;
        const auto image_url_it = input_it->find("image_url");
        if (image_url_it != input_it->end() && !image_url_it->is_null()) {
            if (!image_url_it->is_string()) {
                err = "'input.image_url' must be a string";
                return false;
            }
            image = image_url_it->get<std::string>();
        } else {
            const auto image_it = input_it->find("image");
            if (image_it != input_it->end() && !image_it->is_null()) {
                if (!image_it->is_string()) {
                    err = "'input.image' must be a string";
                    return false;
                }
                image = image_it->get<std::string>();
            }
        }
        if (!image.empty()) {
            pipeline_media m;
            if (!load_image(image, m, err)) {
                return false;
            }
            in.request.media.push_back(std::move(m));
            in.has_image = true;
        }
    }

    const auto options_it = j.find("options");
    if (options_it != j.end() && !options_it->is_null()) {
        if (!options_it->is_object()) {
            err = "'options' must be an object";
            return false;
        }
        in.request.options_json = dump_json(*options_it);
    }

    if (!in.request.input_text.empty() || !in.request.media.empty()) {
        pipeline_message msg;
        msg.role = "user";
        msg.content = in.request.input_text;
        if (!in.request.input_text.empty()) {
            pipeline_part p;
            p.kind = pipeline_part_kind::text;
            p.text = in.request.input_text;
            msg.parts.push_back(std::move(p));
        }
        for (const pipeline_media & m : in.request.media) {
            pipeline_part p;
            p.kind = m.kind;
            p.data = m.data;
            p.format = m.format;
            p.url = m.url;
            msg.parts.push_back(std::move(p));
        }
        in.request.messages.push_back(std::move(msg));
    }
    return true;
}

// The SSE producer/consumer pair.  Kept local to this TU (like sse.h for chat)
// so the pipeline endpoints can be tested with a plain fake pipeline.
struct pipe_stream {
    mutable std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> items;
    bool done = false;
    bool cancelled = false;
    static constexpr size_t kMaxQueue = 4096;

    void push(std::string s) {
        {
            std::lock_guard<std::mutex> lk(m);
            if (cancelled) {
                return;
            }
            if (items.size() >= kMaxQueue) {
                cancelled = true;
                items.clear();
                cv.notify_all();
                return;
            }
            items.push_back(std::move(s));
        }
        cv.notify_one();
    }
    void finish() {
        {
            std::lock_guard<std::mutex> lk(m);
            done = true;
        }
        cv.notify_one();
    }
    void cancel() {
        {
            std::lock_guard<std::mutex> lk(m);
            cancelled = true;
            items.clear();
        }
        cv.notify_all();
    }
    bool is_cancelled() const {
        std::lock_guard<std::mutex> lk(m);
        return cancelled;
    }
    bool pop(std::string & out) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !items.empty() || done || cancelled; });
        if (cancelled || items.empty()) {
            return false;
        }
        out = std::move(items.front());
        items.pop_front();
        return true;
    }
};

void serve_pipeline_sse(httplib::Response & res, const std::shared_ptr<pipe_stream> & st,
                        const std::shared_ptr<std::thread> & producer) {
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_chunked_content_provider(
        "text/event-stream",
        [st](size_t, httplib::DataSink & sink) {
            std::string item;
            if (!st->pop(item)) {
                sink.done();
                return true;
            }
            if (!sink.write(item.data(), item.size())) {
                st->cancel();
                return false;
            }
            return true;
        },
        [st, producer](bool ok) {
            if (!ok) {
                st->cancel();
            }
            if (producer && producer->joinable()) {
                producer->join();
            }
        });
}

void handle_pipeline_run(const httplib::Request & req, httplib::Response & res, pipeline & p,
                         const std::string & pipeline_id, const std::string & model_id) {
    run_input in;
    std::string err;
    if (!parse_run_body(req.body, in, err)) {
        set_error(res, 400, err, "invalid_request_error");
        return;
    }
    const pipeline_desc & d = p.desc();
    if (in.has_image && !d.accepts_image) {
        set_error(res, 400, "pipeline '" + pipeline_id + "' does not accept image input", "invalid_request_error");
        return;
    }
    if (in.has_video && !d.accepts_video) {
        set_error(res, 400, "pipeline '" + pipeline_id + "' does not accept video input", "invalid_request_error");
        return;
    }
    if (in.has_audio && !d.accepts_audio) {
        set_error(res, 400, "pipeline '" + pipeline_id + "' does not accept audio input", "invalid_request_error");
        return;
    }
    if (!in.request.stream) {
        std::string content;
        pipeline_usage usage;
        std::string finish = "stop";
        pipeline_sink sink;
        sink.on_text = [&](std::string_view v) { content.append(v.data(), v.size()); };
        sink.on_done = [&](const pipeline_usage & u, std::string reason) {
            usage = u;
            finish = std::move(reason);
        };
        sink.cancelled = [] { return false; };
        try {
            p.run(in.request, sink);
        } catch (const std::exception & ex) {
            set_error(res, 500, std::string("pipeline failed: ") + ex.what(), "server_error");
            return;
        }
        set_json(res, 200,
                 {{"id", gen_pipeline_id()},
                  {"object", "pipeline.result"},
                  {"created", (uint64_t)time(nullptr)},
                  {"pipeline", pipeline_id},
                  {"model", model_id},
                  {"output", {{"format", first_format(d)}, {"content", content}}},
                  {"usage", usage_json(usage)},
                  {"finish_reason", finish}});
        return;
    }

    auto st = std::make_shared<pipe_stream>();
    auto done_sent = std::make_shared<std::atomic<bool>>(false);
    auto producer = std::make_shared<std::thread>([st, &p, in, pipeline_id, model_id, done_sent]() mutable {
        pipeline_sink sink;
        sink.on_text = [st, pipeline_id](std::string_view v) {
            if (v.empty()) {
                return;
            }
            json j = {{"object", "pipeline.delta"},
                      {"pipeline", pipeline_id},
                      {"delta", {{"content", std::string(v)}}}};
            st->push("data: " + dump_json(j) + "\n\n");
        };
        sink.on_progress = [] {};
        sink.on_done = [st, pipeline_id, done_sent](const pipeline_usage & u, std::string reason) {
            if (done_sent->exchange(true)) {
                return;
            }
            json j = {{"object", "pipeline.done"},
                      {"pipeline", pipeline_id},
                      {"finish_reason", reason},
                      {"usage", usage_json(u)}};
            st->push("data: " + dump_json(j) + "\n\n");
        };
        sink.cancelled = [st] { return st->is_cancelled(); };
        std::string err;
        try {
            p.run(in.request, sink);
        } catch (const std::exception & ex) {
            err = ex.what();
        } catch (...) {
            err = "unknown error";
        }
        if (st->is_cancelled()) {
            st->finish();
            return;
        }
        if (!err.empty()) {
            json j = {{"object", "pipeline.error"},
                      {"pipeline", pipeline_id},
                      {"error", {{"message", err}, {"type", "server_error"}}}};
            st->push("data: " + dump_json(j) + "\n\n");
        }
        if (!done_sent->exchange(true)) {
            json j = {{"object", "pipeline.done"},
                      {"pipeline", pipeline_id},
                      {"finish_reason", err.empty() ? "stop" : "error"},
                      {"usage", usage_json(pipeline_usage{})}};
            st->push("data: " + dump_json(j) + "\n\n");
        }
        st->push("data: [DONE]\n\n");
        st->finish();
    });

    serve_pipeline_sse(res, st, producer);
}

} // namespace

std::string pipelines_json(pipeline_registry & registry, const model_profile * prof) {
    static const model_profile kNoModel{};
    const model_profile & context = prof ? *prof : kNoModel;
    json data = json::array();
    for (const pipeline * p : registry.available(context)) {
        const pipeline_desc & d = p->desc();
        data.push_back({{"id", d.id},
                        {"object", "pipeline"},
                        {"name", d.name},
                        {"input", json_inputs(d)},
                        {"output", json_formats(d)}});
    }
    return dump_json({{"object", "list"}, {"data", data}});
}

void register_model_list_http(httplib::Server & srv, const pipeline_http_config & cfg) {
    srv.Get("/v1/models", [cfg](const httplib::Request &, httplib::Response & res) {
        cors(res);
        res.status = 200;
        res.set_content("{\"data\":[" + cfg.model_entry_json + "],\"object\":\"list\"}", "application/json");
    });
    srv.Get(R"(/v1/models/(.+))", [cfg](const httplib::Request & req, httplib::Response & res) {
        cors(res);
        const std::string id = req.matches[1];
        if (id != cfg.model_id && id != cfg.fallback_model_id) {
            set_error(res, 404, "model '" + id + "' not found", "invalid_request_error");
            return;
        }
        res.status = 200;
        res.set_content(cfg.model_entry_json, "application/json");
    });
}

void register_pipeline_http(httplib::Server & srv, pipeline_registry & registry, const model_profile * prof,
                            const std::string & model_id) {
    srv.Get("/v1/pipelines", [&registry, prof](const httplib::Request &, httplib::Response & res) {
        cors(res);
        res.status = 200;
        res.set_content(pipelines_json(registry, prof), "application/json");
    });

    srv.Post(R"(/v1/pipelines/([^/]+)/run)",
             [&registry, prof, model_id](const httplib::Request & req, httplib::Response & res) {
                 cors(res);
                 const std::string id = req.matches[1];
                 pipeline * p = registry.find(id);
                 static const model_profile kNoModel{};
                 const model_profile & context = prof ? *prof : kNoModel;
                 if (!p || !p->available(context)) {
                     set_error(res, 404, "pipeline '" + id + "' not found", "pipeline_not_found");
                     return;
                 }
                 handle_pipeline_run(req, res, *p, id, model_id);
             });
}

} // namespace si
