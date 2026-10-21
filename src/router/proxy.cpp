#include "proxy.h"

#include "config.h"
#include "httplib.h"
#include "json.hpp"
#include "registry.h"

#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <ratio>
#include <string>
#include <thread>
#include <utility>




namespace si {
namespace router {

using ojson = nlohmann::ordered_json;

namespace {

// The state shared between the proxy's upstream thread and httplib's writer
// thread (the one that runs the chunked provider).
struct proxy_stream {
    std::mutex m;
    std::condition_variable cv;

    bool headers_ready = false;
    int status = 0;
    httplib::Headers headers;

    std::deque<std::string> chunks;
    bool done = false;      // the upstream finished (or failed)
    bool cancelled = false; // the client went away
    std::string error;
    std::shared_ptr<httplib::Client> client;
};

// Hop-by-hop headers must not be forwarded (RFC 9110 7.6.1), and neither
// may the ones httplib computes itself.  The comparison is case-insensitive:
// the header names arrive as the peer sent them.
bool is_content_type(const std::string & name) {
    std::string lower;
    lower.reserve(name.size());
    for (char c : name) {
        lower.push_back((char)tolower((unsigned char)c));
    }
    return lower == "content-type";
}

bool is_hop_by_hop(const std::string & name) {
    static const char * kDrop[] = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                                   "te", "trailer", "transfer-encoding", "upgrade", "content-length", "host"};
    std::string lower;
    lower.reserve(name.size());
    for (char c : name) {
        lower.push_back((char)tolower((unsigned char)c));
    }
    for (const char * d : kDrop) {
        if (lower == d) {
            return true;
        }
    }
    return false;
}

void set_cors(httplib::Response & res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Headers", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
}

void set_error(httplib::Response & res, int status, const std::string & msg, const std::string & type) {
    ojson j;
    j["error"]["message"] = msg;
    j["error"]["type"] = type;
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

std::string body_model(const std::string & body) {
    try {
        const ojson j = ojson::parse(body);
        if (j.is_object()) {
            auto it = j.find("model");
            if (it != j.end() && it->is_string()) {
                return it->get<std::string>();
            }
        }
    } catch (...) {
    }
    return std::string();
}

bool body_is_streaming(const std::string & body) {
    try {
        const ojson j = ojson::parse(body);
        if (j.is_object()) {
            auto it = j.find("stream");
            if (it != j.end() && it->is_boolean()) {
                return it->get<bool>();
            }
        }
    } catch (...) {
    }
    return false;
}

} // namespace

proxy::proxy(registry & reg, const router_config & cfg) : reg_(reg), cfg_(cfg) {
}

bool proxy::authorized(const httplib::Request & req) const {
    if (cfg_.bearer_tokens.empty()) {
        return true;
    }
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) {
        return false;
    }
    const std::string prefix = "Bearer ";
    if (it->second.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    const std::string tok = it->second.substr(prefix.size());
    for (const std::string & t : cfg_.bearer_tokens) {
        if (t == tok) {
            return true;
        }
    }
    return false;
}

void proxy::access_log(const std::string & method, const std::string & path, int status, const std::string & model,
                       double ms) const {
    if (!cfg_.access_log) {
        return;
    }
    fprintf(stderr, "[router] %s %s -> %d model=%s %.1fms\n", method.c_str(), path.c_str(), status,
            model.empty() ? "-" : model.c_str(), ms);
    fflush(stderr);
}

void proxy::install(httplib::Server & srv) {
    srv.Options(".*", [](const httplib::Request &, httplib::Response & res) {
        set_cors(res);
        res.status = 204;
    });

    srv.Get("/health", [this](const httplib::Request &, httplib::Response & res) {
        // The liveness probe stays unauthenticated: a load balancer has to
        // be able to reach it without a token.
        ojson j;
        j["status"] = "ok";
        ojson models = ojson::object();
        bool all_up = true;
        for (const auto & slot : reg_.all()) {
            std::string st;
            {
                std::lock_guard<std::mutex> lk(slot->mtx);
                st = model_state_name(slot->state);
            }
            models[slot->cfg.id] = st;
            if (st != std::string("up")) {
                all_up = false;
            }
        }
        j["models"] = models;
        j["ready"] = all_up;
        res.set_content(j.dump(), "application/json");
    });

    srv.Get("/router/status", [this](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req)) {
            set_error(res, 401, "missing or invalid bearer token", "invalid_request_error");
            return;
        }
        ojson j;
        ojson models = ojson::object();
        // The budget view: per device group, what is committed against what
        // the operator declared.  The engine has no VRAM partitioning, so
        // this is the admission view, not an enforced limit.
        std::map<std::string, uint64_t> committed;
        for (const auto & slot : reg_.all()) {
            ojson e;
            {
                std::lock_guard<std::mutex> lk(slot->mtx);
                e["state"] = model_state_name(slot->state);
                e["pid"] = slot->pid;
                e["restarts"] = slot->restarts;
                e["last_error"] = slot->last_error;
                e["advertised_id"] = slot->advertised_id;
            }
            e["http"] = slot->http_base();
            e["remote"] = slot->cfg.remote;
            e["device_group"] = slot->cfg.device_group;
            e["vram_budget_mb"] = slot->cfg.vram_budget_mb;
            e["kv_cap_mb"] = slot->cfg.kv_cap_mb;
            e["max_slots"] = slot->cfg.max_slots;
            models[slot->cfg.id] = e;
            committed[slot->cfg.device_group] += slot->cfg.vram_budget_mb;
        }
        j["models"] = models;
        ojson groups = ojson::object();
        for (const auto & kv : committed) {
            ojson g;
            g["committed_mb"] = kv.second;
            g["declared_device_mb"] = cfg_.device_vram_mb;
            groups[kv.first] = g;
        }
        j["device_groups"] = groups;
        res.set_content(j.dump(), "application/json");
    });

    srv.Get("/v1/models", [this](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req)) {
            set_error(res, 401, "missing or invalid bearer token", "invalid_request_error");
            return;
        }
        set_cors(res);
        ojson data = ojson::array();
        for (const auto & slot : reg_.all()) {
            std::string advertised;
            {
                std::lock_guard<std::mutex> lk(slot->mtx);
                advertised = slot->advertised_json;
            }
            if (!advertised.empty()) {
                try {
                    data.push_back(ojson::parse(advertised));
                    continue;
                } catch (...) {
                }
            }
            // Not ready yet: advertise the router's own view so a client can
            // still discover the model.
            ojson e;
            e["id"] = slot->cfg.id;
            e["object"] = "model";
            e["owned_by"] = "local";
            data.push_back(e);
        }
        ojson j;
        j["object"] = "list";
        j["data"] = data;
        res.set_content(j.dump(), "application/json");
    });

    srv.Get(R"(/v1/models/(.+))", [this](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req)) {
            set_error(res, 401, "missing or invalid bearer token", "invalid_request_error");
            return;
        }
        set_cors(res);
        const std::string id = req.matches[1];
        const auto slot = reg_.find(id);
        if (!slot) {
            set_error(res, 404, "model '" + id + "' not found", "invalid_request_error");
            return;
        }
        std::string advertised;
        {
            std::lock_guard<std::mutex> lk(slot->mtx);
            advertised = slot->advertised_json;
        }
        if (!advertised.empty()) {
            res.set_content(advertised, "application/json");
            return;
        }
        ojson e;
        e["id"] = slot->cfg.id;
        e["object"] = "model";
        e["owned_by"] = "local";
        res.set_content(e.dump(), "application/json");
    });

    srv.Post("/v1/chat/completions",
             [this](const httplib::Request & req, httplib::Response & res) { handle_completions(req, res, "/v1/chat/completions"); });
    srv.Post("/v1/completions",
             [this](const httplib::Request & req, httplib::Response & res) { handle_completions(req, res, "/v1/completions"); });
}

void proxy::handle_completions(const httplib::Request & req, httplib::Response & res, const char * path) {
    const auto t0 = std::chrono::steady_clock::now();
    set_cors(res);

    if (!authorized(req)) {
        set_error(res, 401, "missing or invalid bearer token", "invalid_request_error");
        access_log(req.method, req.path, res.status, "", 0.0);
        return;
    }

    std::string model = body_model(req.body);
    std::shared_ptr<model_slot> slot;
    if (model.empty()) {
        // OpenAI requires `model`.  With exactly one model configured the
        // natural reading is "that one"; with several it is an error.
        if (reg_.all().size() == 1) {
            slot = reg_.all()[0];
            model = slot->cfg.id;
        } else {
            set_error(res, 400, "'model' is required", "invalid_request_error");
            access_log(req.method, req.path, res.status, model, 0.0);
            return;
        }
    } else {
        slot = reg_.find(model);
        if (!slot) {
            set_error(res, 404, "model '" + model + "' not found", "invalid_request_error");
            access_log(req.method, req.path, res.status, model, 0.0);
            return;
        }
    }

    {
        std::lock_guard<std::mutex> lk(slot->mtx);
        if (slot->state != model_state::up) {
            const std::string st = model_state_name(slot->state);
            set_error(res, 503, "model '" + model + "' is unavailable (state: " + st + ")", "server_error");
            access_log(req.method, req.path, res.status, model, 0.0);
            return;
        }
    }

    // The router is the authority on routing, so the request carries the
    // worker's own id from here on: the worker's /v1/models/{id} and the
    // router's alias then agree.
    std::string body = req.body;
    {
        std::string advertised;
        {
            std::lock_guard<std::mutex> lk(slot->mtx);
            advertised = slot->advertised_id;
        }
        if (!advertised.empty() && advertised != model) {
            try {
                ojson j = ojson::parse(body);
                if (j.is_object()) {
                    j["model"] = advertised;
                    body = j.dump();
                }
            } catch (...) {
                // A body the router cannot parse is forwarded unchanged and
                // the worker reports the error.
            }
        }
    }

    httplib::Headers headers;
    for (const auto & kv : req.headers) {
        if (!is_hop_by_hop(kv.first)) {
            headers.emplace(kv.first, kv.second);
        }
    }

    auto cli = std::make_shared<httplib::Client>(slot->cfg.http_host, slot->cfg.http_port);
    cli->set_read_timeout(cfg_.request_timeout_s, 0);
    cli->set_write_timeout(cfg_.request_timeout_s, 0);
    cli->set_connection_timeout(10, 0);
    cli->set_follow_location(false);

    if (!body_is_streaming(body)) {
        // Non-streaming: the whole answer fits in one response, so do it
        // synchronously and copy the response verbatim (Content-Length and
        // all) - the router must be transparent for this shape.
        auto up = cli->Post(path, headers, body, "application/json");
        if (!up) {
            set_error(res, 502, "the worker did not answer: " + httplib::to_string(up.error()), "server_error");
        } else {
            res.status = up->status;
            for (const auto & kv : up->headers) {
                if (!is_hop_by_hop(kv.first) && !is_content_type(kv.first)) {
                    res.set_header(kv.first, kv.second);
                }
            }
            const char * ct = httplib::detail::get_header_value(up->headers, "Content-Type", "application/json", 0);
            res.set_content(up->body, ct);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        access_log(req.method, req.path, res.status, model, ms);
        return;
    }

    // Streaming: the upstream runs on its own thread and the response body
    // is forwarded chunk by chunk through the provider.
    auto stream = std::make_shared<proxy_stream>();
    stream->client = cli;

    std::thread([stream, cli, path, headers, body]() {
        // Client::send() with a hand-built request, because the POST
        // overloads have no response-handler: the router needs the upstream's
        // status and headers BEFORE the body, so that a 503 from the worker
        // is forwarded as a 503 instead of being turned into a broken stream.
        httplib::Request rq;
        rq.method = "POST";
        rq.path = path;
        rq.headers = headers;
        rq.body = body;
        rq.set_header("Content-Type", "application/json");
        rq.response_handler = [stream](const httplib::Response & r) {
            std::lock_guard<std::mutex> lk(stream->m);
            stream->status = r.status;
            stream->headers = r.headers;
            stream->headers_ready = true;
            stream->cv.notify_all();
            return true;
        };
        rq.content_receiver = [stream](const char * data, size_t len, uint64_t, uint64_t) {
            std::lock_guard<std::mutex> lk(stream->m);
            if (stream->cancelled) {
                return false;
            }
            stream->chunks.emplace_back(data, len);
            stream->cv.notify_all();
            return true;
        };
        auto up = cli->send(rq);
        {
            std::lock_guard<std::mutex> lk(stream->m);
            if (!up) {
                stream->error = httplib::to_string(up.error());
            }
            stream->done = true;
            stream->cv.notify_all();
        }
    }).detach();

    // Wait for the upstream's response headers: the status decides whether
    // this is a stream to forward or an error to pass through, and sending
    // 200 before knowing it would turn a 503 into a broken stream.
    std::string ct = "text/event-stream";
    {
        std::unique_lock<std::mutex> lk(stream->m);
        while (!stream->headers_ready && !stream->done) {
            if (stream->cv.wait_for(lk, std::chrono::seconds(cfg_.request_timeout_s)) == std::cv_status::timeout) {
                break;
            }
        }
        if (!stream->headers_ready && stream->done) {
            set_error(res, 502, "the worker did not answer: " + stream->error, "server_error");
            access_log(req.method, req.path, res.status, model, 0.0);
            return;
        }
        if (!stream->headers_ready) {
            set_error(res, 504, "the worker did not send a response header in time", "server_error");
            access_log(req.method, req.path, res.status, model, 0.0);
            return;
        }
        res.status = stream->status;
        ct = httplib::detail::get_header_value(stream->headers, "Content-Type", "text/event-stream", 0);
        if (stream->status < 200 || stream->status >= 300) {
            // An error answer: collect it and serve it as a normal response.
            while (!stream->done) {
                stream->cv.wait_for(lk, std::chrono::seconds(cfg_.request_timeout_s));
            }
            std::string err_body;
            for (const std::string & c : stream->chunks) {
                err_body += c;
            }
            res.set_content(err_body, ct);
            access_log(req.method, req.path, res.status, model, 0.0);
            return;
        }
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
    }

    // The access log for a stream is written when the stream ends, not when
    // the headers go out: the interesting duration is the whole generation,
    // and logging at header time would report the time-to-first-token as if
    // it were the request's cost.
    const auto t_start = t0;
    const std::string log_model = model;
    // The method and the path are copied, not referenced: the releaser runs
    // when the response is destroyed, and the request object may already be
    // gone by then.
    const std::string log_method = req.method;
    const std::string log_path = req.path;

    // The releaser is the reliable "the client went away" signal: httplib
    // checks the peer BEFORE every provider call, so a client that vanished
    // while the provider was waiting is never seen by the provider at all -
    // the response is simply aborted and the provider is never called again.
    // The releaser runs when the response is destroyed and reports whether
    // the body was written, which is exactly the fact the cancel chain
    // needs.
    res.set_chunked_content_provider(
        ct,
        [stream](size_t, httplib::DataSink & sink) -> bool {
            std::unique_lock<std::mutex> lk(stream->m);
            for (;;) {
                if (stream->cancelled) {
                    return false;
                }
                if (!stream->chunks.empty()) {
                    std::string chunk = std::move(stream->chunks.front());
                    stream->chunks.pop_front();
                    lk.unlock();
                    const bool ok = sink.write(chunk.data(), chunk.size());
                    lk.lock();
                    if (!ok) {
                        // The client is gone.  Abort the upstream request so
                        // the worker's next write fails and it cancels the
                        // generation instead of running to max_tokens.
                        stream->cancelled = true;
                        lk.unlock();
                        if (stream->client) {
                            stream->client->stop();
                        }
                        return false;
                    }
                    return true;
                }
                if (stream->done) {
                    sink.done();
                    return true;
                }
                // Wait for the next chunk, but wake up periodically so a
                // client that vanished while nothing was being generated is
                // still noticed.
                stream->cv.wait_for(lk, std::chrono::seconds(1));
                if (!sink.is_writable()) {
                    stream->cancelled = true;
                    lk.unlock();
                    if (stream->client) {
                        stream->client->stop();
                    }
                    return false;
                }
            }
        },
        [stream, this, t_start, log_model, log_method, log_path](bool success) {
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
            access_log(log_method, log_path, success ? 200 : 499, log_model, ms);
            if (success) {
                return;
            }
            std::lock_guard<std::mutex> lk(stream->m);
            if (stream->cancelled) {
                return; // the provider already aborted it
            }
            stream->cancelled = true;
            if (stream->client) {
                stream->client->stop();
            }
        });

}

} // namespace router
} // namespace si
