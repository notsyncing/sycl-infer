#include "pipeline_http.h"

#include "httplib.h"
#include "json.hpp"
#include "pipeline/registry.h"

#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;
int g_fail = 0;

#define CHECK(cond, ...)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                                                       \
            fprintf(stderr, __VA_ARGS__);                                                                              \
            fprintf(stderr, "\n");                                                                                     \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

class fake_pipeline : public si::pipeline {
public:
    fake_pipeline(std::string id, bool accepts_image) : id_(std::move(id)), accepts_image_(accepts_image) {
        d_.id = id_.c_str();
        d_.name = "Fake pipeline";
        d_.accepts_image = accepts_image_;
        d_.output_formats = "json,text";
    }

    const si::pipeline_desc & desc() const override {
        return d_;
    }
    bool available(const si::model_profile &) const override {
        return true;
    }
    void run(const si::pipeline_request & req, si::pipeline_sink & sink) override {
        std::string content = "echo:" + req.input_text + ":" + req.options_json;
        if (!req.media.empty()) {
            content += ":image=" + std::to_string(req.media[0].data.size());
        }
        sink.on_text(content.substr(0, 5));
        sink.on_text(content.substr(5));
        sink.on_done({7, 2, 1, 3}, "stop");
    }

private:
    std::string id_;
    bool accepts_image_;
    si::pipeline_desc d_;
};

struct fixture {
    si::pipeline_registry reg;
    httplib::Server srv;
    std::thread th;
    int port = 0;

    void start() {
        auto echo = std::make_unique<fake_pipeline>("echo", true);
        auto text_only = std::make_unique<fake_pipeline>("text-only", false);
        CHECK(reg.add(std::move(echo)), "register echo");
        CHECK(reg.add(std::move(text_only)), "register text-only");

        si::pipeline_http_config cfg;
        cfg.model_id = "fake-model";
        cfg.fallback_model_id = "fake-model";
        cfg.model_entry_json = "{\"id\":\"fake-model\",\"object\":\"model\"}";
        si::register_model_list_http(srv, cfg);
        si::register_pipeline_http(srv, reg, nullptr, "fake-model");

        port = srv.bind_to_any_port("127.0.0.1");
        th = std::thread([this]() { srv.listen_after_bind(); });
    }

    void stop() {
        srv.stop();
        if (th.joinable()) {
            th.join();
        }
    }
};

httplib::Result get(int port, const std::string & path) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_read_timeout(10, 0);
    return cli.Get(path);
}

httplib::Result post(int port, const std::string & path, const std::string & body) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_read_timeout(10, 0);
    return cli.Post(path, body, "application/json");
}

void test_list() {
    fixture fx;
    fx.start();
    auto res = get(fx.port, "/v1/pipelines");
    CHECK(!!res, "the list must answer");
    if (res) {
        CHECK(res->status == 200, "list status: %d", res->status);
        CHECK(res->get_header_value("Access-Control-Allow-Origin") == "*", "CORS header");
        json j = json::parse(res->body);
        CHECK(j.value("object", "") == "list", "list object");
        CHECK(j["data"].is_array() && j["data"].size() == 2, "two pipelines: %s", res->body.c_str());
        if (j["data"].is_array() && !j["data"].empty()) {
            const json & p = j["data"][0];
            CHECK(p.value("id", "") == "echo", "id");
            CHECK(p.value("object", "") == "pipeline", "object");
            CHECK(p["input"].is_array(), "input array");
            CHECK(p["output"].is_array(), "output array");
            bool has_image = false;
            for (const auto & v : p["input"]) {
                has_image |= v == "image";
            }
            CHECK(has_image, "echo accepts image");
        }
    }
    fx.stop();
}

void test_nonstream() {
    fixture fx;
    fx.start();
    auto res = post(fx.port, "/v1/pipelines/echo/run",
                    "{\"input\":{\"text\":\"hi\"},\"options\":{\"x\":1},\"model\":\"not-a-model\"}");
    CHECK(!!res, "the run must answer");
    if (res) {
        CHECK(res->status == 200, "run status: %d", res->status);
        json j = json::parse(res->body);
        CHECK(j.value("object", "") == "pipeline.result", "result object");
        CHECK(j.value("pipeline", "") == "echo", "pipeline id");
        CHECK(j.value("model", "") == "fake-model", "model id");
        CHECK(j["output"].value("format", "") == "json", "output format");
        CHECK(j["output"].value("content", "") == "echo:hi:{\"x\":1}", "content: %s",
              j["output"].value("content", "").c_str());
        CHECK(j["usage"].value("prompt_tokens", 0) == 7, "prompt tokens");
        CHECK(j["usage"].value("completion_tokens", 0) == 2, "completion tokens");
        CHECK(j["usage"].value("total_tokens", 0) == 9, "total tokens");
        CHECK(j["usage"]["prompt_tokens_details"].value("cached_tokens", 0) == 3, "cached tokens");
        CHECK(j["usage"]["completion_tokens_details"].value("reasoning_tokens", 0) == 1, "reasoning tokens");
    }
    fx.stop();
}

void test_image_input() {
    fixture fx;
    fx.start();
    // "AAAA" is three zero bytes.
    auto res = post(fx.port, "/v1/pipelines/echo/run",
                    "{\"input\":{\"image\":\"data:image/png;base64,AAAA\"},\"options\":{}}");
    CHECK(!!res, "the image run must answer");
    if (res) {
        CHECK(res->status == 200, "image run status: %d", res->status);
        json j = json::parse(res->body);
        CHECK(j["output"].value("content", "") == "echo::{}:image=3", "image content: %s",
              j["output"].value("content", "").c_str());
    }
    fx.stop();
}

void test_sse() {
    fixture fx;
    fx.start();
    httplib::Client cli("127.0.0.1", fx.port);
    cli.set_read_timeout(10, 0);
    std::string all;
    auto res = cli.Post("/v1/pipelines/echo/run", httplib::Headers(),
                        "{\"input\":{\"text\":\"hi\"},\"stream\":true}", "application/json",
                        [&](const char * data, size_t len) {
                            all.append(data, len);
                            return true;
                        });
    CHECK(!!res, "the stream must answer");
    CHECK(all.find("pipeline.delta") != std::string::npos, "delta frame");
    CHECK(all.find("echo:") != std::string::npos, "delta content");
    CHECK(all.find("pipeline.done") != std::string::npos, "done frame");
    CHECK(all.find("[DONE]") != std::string::npos, "done sentinel");
    fx.stop();
}

void test_errors() {
    fixture fx;
    fx.start();
    auto res = post(fx.port, "/v1/pipelines/nope/run", "{}");
    CHECK(!!res, "unknown pipeline must answer");
    if (res) {
        CHECK(res->status == 404, "unknown status: %d", res->status);
        json j = json::parse(res->body);
        CHECK(j["error"].value("type", "") == "pipeline_not_found", "unknown type");
    }
    res = post(fx.port, "/v1/pipelines/echo/run", "not json");
    CHECK(!!res, "bad json must answer");
    if (res) {
        CHECK(res->status == 400, "bad json status: %d", res->status);
        json j = json::parse(res->body);
        CHECK(j["error"].value("type", "") == "invalid_request_error", "bad json type");
    }
    res = post(fx.port, "/v1/pipelines/text-only/run",
               "{\"input\":{\"image\":\"data:image/png;base64,AAAA\"}}");
    CHECK(!!res, "the descriptor refusal must answer");
    if (res) {
        CHECK(res->status == 400, "image refusal status: %d", res->status);
        json j = json::parse(res->body);
        CHECK(j["error"].value("type", "") == "invalid_request_error", "image refusal type");
    }
    fx.stop();
}

void test_models_do_not_include_pipelines() {
    fixture fx;
    fx.start();
    auto res = get(fx.port, "/v1/models");
    CHECK(!!res, "the model list must answer");
    if (res) {
        json j = json::parse(res->body);
        CHECK(j.value("object", "") == "list", "model list object");
        CHECK(j["data"].is_array() && j["data"].size() == 1, "one model");
        for (const auto & e : j["data"]) {
            CHECK(e.value("id", "") == "fake-model", "model id: %s", e.value("id", "").c_str());
            CHECK(e.value("id", "") != "echo", "a pipeline id must not appear in /v1/models");
        }
    }
    fx.stop();
}

} // namespace

int main() {
    test_list();
    test_nonstream();
    test_image_input();
    test_sse();
    test_errors();
    test_models_do_not_include_pipelines();
    if (g_fail) {
        fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_pipeline_route: OK\n");
    return 0;
}
