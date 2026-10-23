#include "proxy.h"

#include "config.h"
#include "registry.h"

#include <httplib.h>
#include <json.hpp>

#include "onnx_wire.h"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <ratio>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>



using namespace si::router;
using json = nlohmann::json;

namespace {

int g_fail = 0;

#define CHECK(cond, ...)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                                                      \
            fprintf(stderr, __VA_ARGS__);                                                                              \
            fprintf(stderr, "\n");                                                                                     \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

// ---------------------------------------------------------------- fake worker

struct fake_worker {
    httplib::Server srv;
    int port = 0;
    std::thread th;

    std::atomic<bool> client_disconnected{false};
    std::atomic<int> sse_chunks{0};
    std::string last_model;
    std::string last_onnx_model;
    std::mutex mtx;

    // How many chunks the stream carries and how long it waits between them.
    int chunks = 5;
    int chunk_delay_ms = 100;

    void start() {
        srv.Get("/health", [](const httplib::Request &, httplib::Response & res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });
        srv.Get("/v1/models", [](const httplib::Request &, httplib::Response & res) {
            res.set_content("{\"object\":\"list\",\"data\":[{\"id\":\"worker-model\",\"object\":\"model\"}]}",
                            "application/json");
        });
        srv.Post("/v1/onnx/run", [this](const httplib::Request & req, httplib::Response & res) {
            std::string model, err;
            si::onnxwire::peek_request_model(reinterpret_cast<const uint8_t *>(req.body.data()),
                                             req.body.size(), model, err);
            {
                std::lock_guard<std::mutex> lk(mtx);
                last_onnx_model = model;
            }
            const std::vector<uint8_t> out = si::onnxwire::encode_response(0, "", {});
            res.set_content(reinterpret_cast<const char *>(out.data()), out.size(),
                            "application/octet-stream");
        });
        srv.Post("/v1/chat/completions", [this](const httplib::Request & req, httplib::Response & res) {
            json body;
            try {
                body = json::parse(req.body);
            } catch (...) {
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                last_model = body.value("model", std::string());
            }
            if (body.value("stream", false)) {
                res.set_header("Content-Type", "text/event-stream");
                auto n = std::make_shared<int>(0);
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [this, n](size_t, httplib::DataSink & sink) -> bool {
                        if (*n >= chunks) {
                            sink.done();
                            return true;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(chunk_delay_ms));
                        const std::string payload = "data: chunk " + std::to_string(*n) + "\n\n";
                        (*n)++;
                        if (!sink.write(payload.data(), payload.size())) {
                            // The peer went away: this is the signal the
                            // worker would use to cancel the generation.
                            client_disconnected.store(true);
                            return false;
                        }
                        sse_chunks.fetch_add(1);
                        return true;
                    },
                    // The releaser is the reliable disconnect signal: httplib
                    // checks the peer BEFORE every provider call, so a peer
                    // that vanished between two items never makes
                    // sink.write() fail.  This is the same fact
                    // src/server/sse.h documents for the real worker.
                    [this](bool success) {
                        if (!success) {
                            client_disconnected.store(true);
                        }
                    });
                return;
            }
            json out;
            out["echo_model"] = last_model;
            out["ok"] = true;
            res.set_content(out.dump(), "application/json");
        });
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

// ------------------------------------------------------------------- the router

struct router_fixture {
    fake_worker worker;
    registry reg;
    std::shared_ptr<model_slot> slot;
    std::shared_ptr<model_slot> onnx_slot;
    // The proxy must outlive the server: httplib keeps the handlers, and the
    // handlers capture the proxy's `this`.
    std::unique_ptr<proxy> px;
    httplib::Server srv;
    int port = 0;
    std::thread th;

    void start() {
        worker.start();
        model_config m;
        m.id = "router-model";
        m.http_host = "127.0.0.1";
        m.http_port = worker.port;
        slot = reg.add(m);
        slot->advertised_id = "worker-model";
        slot->advertised_json = "{\"id\":\"worker-model\",\"object\":\"model\"}";
        slot->set_state(model_state::up);
        router_config cfg;
        px = std::make_unique<proxy>(reg, cfg);
        px->install(srv);
        port = srv.bind_to_any_port("127.0.0.1");
        th = std::thread([this]() { srv.listen_after_bind(); });
    }

    void stop() {
        srv.stop();
        if (th.joinable()) {
            th.join();
        }
        worker.stop();
    }
};

std::string post_json(int port, const std::string & path, const std::string & body) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_read_timeout(20, 0);
    auto res = cli.Post(path, body, "application/json");
    if (!res) {
        return std::string();
    }
    return res->body;
}

// -------------------------------------------------------------------- the tests

void test_non_streaming() {
    router_fixture fx;
    fx.start();

    const std::string body = post_json(fx.port, "/v1/chat/completions",
                                       "{\"model\":\"router-model\",\"messages\":[{\"role\":\"user\","
                                       "\"content\":\"hi\"}]}");
    CHECK(!body.empty(), "the router must answer");
    if (!body.empty()) {
        json j = json::parse(body);
        // The worker must have seen the WORKER's id, not the router's alias:
        // the router rewrites the model field so the worker's own
        // /v1/models/{id} stays consistent with what the client asked for.
        CHECK(j.value("echo_model", std::string()) == "worker-model",
              "the router must rewrite the model field, got '%s'", j.value("echo_model", "").c_str());
    }
    fx.stop();
}

void test_unknown_model() {
    router_fixture fx;
    fx.start();
    const std::string body = post_json(fx.port, "/v1/chat/completions", "{\"model\":\"nope\"}");
    CHECK(!body.empty(), "the router must answer");
    if (!body.empty()) {
        json j = json::parse(body);
        CHECK(j.contains("error"), "the answer must be an error object");
        CHECK(j["error"].value("type", "") == "invalid_request_error", "the error type must match the worker's shape");
        CHECK(j["error"].value("message", "").find("nope") != std::string::npos, "the message names the model");
    }
    fx.stop();
}

void test_model_down() {
    router_fixture fx;
    fx.start();
    fx.slot->set_state(model_state::failed, "the worker died");
    const std::string body = post_json(fx.port, "/v1/chat/completions", "{\"model\":\"router-model\"}");
    CHECK(!body.empty(), "the router must answer");
    if (!body.empty()) {
        json j = json::parse(body);
        CHECK(j.contains("error"), "the answer must be an error object");
        CHECK(j["error"].value("message", "").find("unavailable") != std::string::npos,
              "the message says the model is unavailable: %s", j["error"].value("message", "").c_str());
    }
    fx.stop();
}

void test_sse_is_not_buffered() {
    router_fixture fx;
    fx.worker.chunks = 5;
    fx.worker.chunk_delay_ms = 120;
    fx.start();

    httplib::Client cli("127.0.0.1", fx.port);
    cli.set_read_timeout(20, 0);
    std::vector<std::chrono::steady_clock::time_point> arrivals;
    std::string all;
    auto res = cli.Post("/v1/chat/completions", httplib::Headers(), "{\"model\":\"router-model\",\"stream\":true}",
                        "application/json",
                        [&](const char * data, size_t len) {
                            arrivals.push_back(std::chrono::steady_clock::now());
                            all.append(data, len);
                            return true;
                        });
    CHECK(!!res, "the streaming request must succeed");
    CHECK(arrivals.size() >= 4, "the client must see the chunks separately: %zu", arrivals.size());
    CHECK(all.find("chunk 4") != std::string::npos, "the whole stream must arrive");
    if (arrivals.size() >= 2) {
        const double span = std::chrono::duration<double, std::milli>(arrivals.back() - arrivals.front()).count();
        // The worker sleeps 120 ms between chunks, so five chunks take ~600 ms.
        // A router that buffered the answer would deliver them all at once.
        CHECK(span > 200.0, "the chunks must arrive spread out, not in one batch (span %.0f ms)", span);
    }
    fx.stop();
}

// A client that vanishes mid-stream must make the worker stop: the router
// has to abort the upstream request, and the worker has to see the socket go
// away.  A raw socket with SO_LINGER 0 gives an RST, which is the
// unambiguous "the client is gone" signal.
void test_cancel_propagates() {
    router_fixture fx;
    fx.worker.chunks = 50;
    fx.worker.chunk_delay_ms = 100;
    fx.start();

    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(fd >= 0, "socket");
    if (fd < 0) {
        fx.stop();
        return;
    }
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)fx.port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    const int rc = ::connect(fd, (struct sockaddr *)&addr, sizeof(addr));
    CHECK(rc == 0, "connect");
    if (rc != 0) {
        ::close(fd);
        fx.stop();
        return;
    }
    const std::string body = "{\"model\":\"router-model\",\"stream\":true}";
    const std::string req = "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                             "Content-Type: application/json\r\nContent-Length: " +
                             std::to_string(body.size()) + "\r\n\r\n" + body;
    ::send(fd, req.data(), req.size(), 0);

    // Read enough to be sure the stream is running.
    char buf[512];
    ::recv(fd, buf, sizeof(buf), 0);
    ::recv(fd, buf, sizeof(buf), 0);

    // Vanish: SO_LINGER with a zero timeout makes close() send an RST.
    struct linger lg;
    lg.l_onoff = 1;
    lg.l_linger = 0;
    ::setsockopt(fd, SOL_SOCKET, SO_LINGER, &lg, sizeof(lg));
    ::close(fd);

    // The worker must notice within a couple of chunks.
    bool disconnected = false;
    for (int i = 0; i < 50; i++) {
        if (fx.worker.client_disconnected.load()) {
            disconnected = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    CHECK(disconnected, "the worker must see the client go away (the router has to abort the upstream)");
    fx.stop();
}

void test_models_aggregate() {
    router_fixture fx;
    fx.start();
    httplib::Client cli("127.0.0.1", fx.port);
    auto res = cli.Get("/v1/models");
    CHECK(!!res, "the model list must answer");
    if (res) {
        json j = json::parse(res->body);
        CHECK(j.value("object", "") == "list", "the shape must be an OpenAI model list");
        CHECK(j["data"].is_array() && j["data"].size() == 1, "one model");
        if (j["data"].is_array() && !j["data"].empty()) {
            CHECK(j["data"][0].value("id", "") == "worker-model", "the worker's own id is advertised");
        }
    }
    fx.stop();
}

} // namespace


void test_onnx_binary_route() {
    router_fixture fx;
    fx.start();
    model_config om;
    om.id = "onnx-model";
    om.backend = "onnx";
    om.http_host = "127.0.0.1";
    om.http_port = fx.worker.port;
    fx.onnx_slot = fx.reg.add(om);
    fx.onnx_slot->set_state(model_state::up);
    const std::vector<uint8_t> req = si::onnxwire::encode_request("onnx-model", {});
    httplib::Client cli("127.0.0.1", fx.port);
    cli.set_read_timeout(10, 0);
    auto res = cli.Post("/v1/onnx/run", reinterpret_cast<const char *>(req.data()), req.size(),
                        "application/octet-stream");
    std::vector<si::onnxwire::tensor> outs;
    int32_t status = -1;
    std::string msg, err;
    if (res) {
        si::onnxwire::decode_response(reinterpret_cast<const uint8_t *>(res->body.data()),
                                      res->body.size(), status, msg, outs, err);
    }
    CHECK(res && res->status == 200, "the binary route must return HTTP 200");
    CHECK(status == 0, "the ONNX response status: %d (%s)", status, msg.c_str());
    {
        std::lock_guard<std::mutex> lk(fx.worker.mtx);
        CHECK(fx.worker.last_onnx_model == "onnx-model", "worker saw '%s'",
              fx.worker.last_onnx_model.c_str());
    }
    const std::vector<uint8_t> unknown = si::onnxwire::encode_request("no-such-model", {});
    res = cli.Post("/v1/onnx/run", reinterpret_cast<const char *>(unknown.data()), unknown.size(),
                   "application/octet-stream");
    if (res) {
        si::onnxwire::decode_response(reinterpret_cast<const uint8_t *>(res->body.data()),
                                      res->body.size(), status, msg, outs, err);
    }
    CHECK(res && status == 2, "an unknown ONNX model must be a wire error (status %d)", status);
    fx.stop();
}

int main() {
    test_non_streaming();
    test_unknown_model();
    test_model_down();
    test_sse_is_not_buffered();
    test_cancel_propagates();
    test_models_aggregate();
    if (g_fail) {
        fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_router_proxy: OK\n");
    return 0;
}
