// The worker's control plane: the JSON-RPC protocol, the reports and the
// shutdown hook.
//
// No GPU and no model: the control server takes its report from a callback,
// so it can be exercised on its own.

#include "control.h"

#include "jsonrpc.h"

#include <arpa/inet.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace si;
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

int connect_to(int port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (::connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

// One request, one reply.
json call(int fd, const json & req) {
    const std::string out = rpc::frame(req);
    if (::send(fd, out.data(), out.size(), MSG_NOSIGNAL) < 0) {
        return json();
    }
    std::string buf;
    char tmp[4096];
    for (;;) {
        const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) {
            return json();
        }
        buf.append(tmp, (size_t)n);
        const size_t nl = buf.find('\n');
        if (nl != std::string::npos) {
            try {
                return json::parse(buf.substr(0, nl));
            } catch (...) {
                return json();
            }
        }
    }
}

control_report sample_report() {
    control_report r;
    control_report::device_t d;
    d.index = 0;
    d.name = "fake-gpu";
    d.total_vram_bytes = 2000ull * 1024 * 1024;
    d.free_vram_bytes = 1000ull * 1024 * 1024;
    d.kv_bytes = 42;
    r.devices.push_back(d);
    r.kv_used_bytes = 42;
    r.kv_cap_bytes = 84;
    r.kv_pool_blocks = 10;
    r.kv_pool_cap = 20;
    r.kv_used_blocks = 3;
    r.pc_vram_bytes = 7;
    r.pc_ram_records = 1;
    r.pc_disk_records = 2;
    r.host_rss_bytes = 1234;
    r.inflight = 5;
    r.uptime_s = 42;
    r.model_id = "fake-model";
    r.max_seq = 4096;
    r.n_vocab = 1000;
    r.n_layer = 24;
    return r;
}

void test_protocol() {
    std::atomic<bool> terminated{false};
    control_server ctl([]() { return sample_report(); }, [&terminated]() { terminated.store(true); });
    std::string err;
    CHECK(ctl.start("127.0.0.1", 0, err), "the control server must start: %s", err.c_str());
    const int port = ctl.port();
    CHECK(port > 0, "the port must be known: %d", port);

    const int fd = connect_to(port);
    CHECK(fd >= 0, "connect");
    if (fd < 0) {
        ctl.stop();
        return;
    }

    // hello: the handshake, with the protocol version.
    json reply = call(fd, rpc::make_request(1, rpc::kHello, json{{"protocol", rpc::kProtocolVersion}}));
    CHECK(reply.value("id", (int64_t)0) == 1, "the reply must carry the id");
    CHECK(reply.contains("result"), "hello must succeed");
    if (reply.contains("result")) {
        const json & w = reply["result"]["worker"];
        CHECK(w["devices"].is_array() && w["devices"].size() == 1, "the report must carry the devices");
        CHECK(w["devices"][0].value("free_vram_bytes", (uint64_t)0) == 1000ull * 1024 * 1024,
              "the free VRAM must be reported");
        CHECK(w["devices"][0].value("name", std::string()) == "fake-gpu", "the device name must be reported");
        CHECK(w["kv"].value("used_blocks", 0) == 3, "the KV pool must be reported");
        CHECK(w["model"].value("id", std::string()) == "fake-model", "the model must be reported");
        CHECK(w["host"].value("rss_bytes", (uint64_t)0) == 1234, "the host RSS must be reported");
    }

    // stats: the same report without the handshake.
    reply = call(fd, rpc::make_request(2, rpc::kStats));
    CHECK(reply.contains("result"), "stats must succeed");
    if (reply.contains("result")) {
        CHECK(reply["result"].value("inflight", 0) == 5, "the inflight count must be reported");
    }

    // health
    reply = call(fd, rpc::make_request(3, rpc::kHealth));
    CHECK(reply.contains("result"), "health must succeed");
    if (reply.contains("result")) {
        CHECK(reply["result"].value("ready", false), "health must report ready");
    }

    // a wrong protocol version is refused, not misread.
    reply = call(fd, rpc::make_request(4, rpc::kHello, json{{"protocol", 999}}));
    CHECK(reply.contains("error"), "a wrong protocol version must be an error");
    if (reply.contains("error")) {
        CHECK(reply["error"].value("code", 0) == rpc::kInvalidParams, "the error code must be the params one");
    }

    // an unknown method
    reply = call(fd, rpc::make_request(5, "nonsense"));
    CHECK(reply.contains("error"), "an unknown method must be an error");
    if (reply.contains("error")) {
        CHECK(reply["error"].value("code", 0) == rpc::kMethodNotFound, "the error code must be the method one");
    }

    // garbage: a parse error, and the connection survives it.
    const std::string junk = "{not json\n";
    ::send(fd, junk.data(), junk.size(), MSG_NOSIGNAL);
    std::string buf;
    char tmp[4096];
    const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
    if (n > 0) {
        buf.assign(tmp, (size_t)n);
        const size_t nl = buf.find('\n');
        if (nl != std::string::npos) {
            try {
                const json e = json::parse(buf.substr(0, nl));
                CHECK(e.contains("error"), "garbage must produce an error reply");
            } catch (...) {
                CHECK(false, "the error reply must be JSON");
            }
        }
    }

    // shutdown: the hook must fire (the worker takes the SIGTERM path).
    reply = call(fd, rpc::make_request(6, rpc::kShutdown));
    CHECK(reply.contains("result"), "shutdown must succeed");
    CHECK(terminated.load(), "shutdown must ask the worker to unwind");

    ::close(fd);
    ctl.stop();
}

void test_authentication() {
    // A control server with a token: the handshake must present it, and every
    // other method needs the handshake first.
    std::atomic<bool> terminated{false};
    control_server ctl([]() { return sample_report(); }, [&terminated]() { terminated.store(true); }, "s3cret");
    std::string err;
    CHECK(ctl.start("127.0.0.1", 0, err), "the control server must start: %s", err.c_str());
    CHECK(ctl.needs_token(), "a token was configured");

    // 1. no token at all -> refused
    int fd = connect_to(ctl.port());
    CHECK(fd >= 0, "connect");
    json reply = call(fd, rpc::make_request(1, rpc::kHello, json{{"protocol", rpc::kProtocolVersion}}));
    CHECK(reply.contains("error"), "a handshake without a token must be refused");
    if (reply.contains("error")) {
        CHECK(reply["error"].value("code", 0) == rpc::kUnauthorized, "the error code must be the auth one");
    }
    // ... and the connection stays unauthenticated
    reply = call(fd, rpc::make_request(2, rpc::kStats));
    CHECK(reply.contains("error"), "an unauthenticated connection must not read the report");
    ::close(fd);

    // 2. the wrong token -> refused
    fd = connect_to(ctl.port());
    reply = call(fd, rpc::make_request(3, rpc::kHello,
                                       json{{"protocol", rpc::kProtocolVersion}, {"token", "wrong"}}));
    CHECK(reply.contains("error"), "a wrong token must be refused");
    ::close(fd);

    // 3. the right token -> accepted, and the connection is authenticated
    fd = connect_to(ctl.port());
    reply = call(fd, rpc::make_request(4, rpc::kHello,
                                       json{{"protocol", rpc::kProtocolVersion}, {"token", "s3cret"}}));
    CHECK(reply.contains("result"), "the right token must be accepted");
    reply = call(fd, rpc::make_request(5, rpc::kStats));
    CHECK(reply.contains("result"), "an authenticated connection must be able to read the report");
    reply = call(fd, rpc::make_request(6, rpc::kHealth));
    CHECK(reply.contains("result"), "an authenticated connection must be able to ask for health");
    ::close(fd);

    // 4. a fresh connection that skips the handshake -> refused
    fd = connect_to(ctl.port());
    reply = call(fd, rpc::make_request(7, rpc::kStats));
    CHECK(reply.contains("error"), "a connection that skips the handshake must be refused");
    ::close(fd);

    ctl.stop();
}

// Without a token the control plane is open (the loopback default).
void test_no_token_is_open() {
    std::atomic<bool> terminated{false};
    control_server ctl([]() { return sample_report(); }, [&terminated]() { terminated.store(true); });
    std::string err;
    CHECK(ctl.start("127.0.0.1", 0, err), "the control server must start: %s", err.c_str());
    CHECK(!ctl.needs_token(), "no token was configured");
    const int fd = connect_to(ctl.port());
    CHECK(fd >= 0, "connect");
    if (fd >= 0) {
        json reply = call(fd, rpc::make_request(1, rpc::kHello, json{{"protocol", rpc::kProtocolVersion}}));
        CHECK(reply.contains("result"), "the handshake must work without a token");
        reply = call(fd, rpc::make_request(2, rpc::kStats));
        CHECK(reply.contains("result"), "the report must be readable without a token");
        ::close(fd);
    }
    ctl.stop();
}

} // namespace

int main() {
    test_protocol();
    test_authentication();
    test_no_token_is_open();
    if (g_fail) {
        fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_control: OK\n");
    return 0;
}
