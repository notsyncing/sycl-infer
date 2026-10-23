#include "config.h"
#include "registry.h"
#include "rpc_client.h"
#include "supervisor.h"

#include "json.hpp"

#include <httplib.h>

#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>



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

// --------------------------------------------------------------- the fake worker

json report_json(long free_mb, long total_mb) {
    json r;
    json d;
    d["index"] = 0;
    d["name"] = "fake-gpu";
    d["total_vram_bytes"] = (uint64_t)total_mb * 1024 * 1024;
    d["free_vram_bytes"] = (uint64_t)free_mb * 1024 * 1024;
    d["kv_bytes"] = 0;
    r["devices"] = json::array({d});
    r["model"]["id"] = "rpc-model";
    r["model"]["n_ctx"] = 4096;
    r["inflight"] = 0;
    r["uptime_s"] = 1;
    return r;
}

// How the fake worker left: the file records "rpc" when it took the control
// plane's shutdown, "sigterm" when it took the signal.  A SIGKILL leaves
// nothing behind - which is exactly the failure this test is for: the router
// must NOT kill a worker that is shutting down gracefully (the real worker
// flushes its prefix cache on the way out).
void mark_exit(const char * how) {
    const char * path = std::getenv("TEST_ROUTER_EXIT_MARK");
    if (!path) {
        return;
    }
    // APPEND, never truncate: two writers (the control thread and the signal
    // handler) can run at the same time, and a truncating write would make
    // the file's content depend on their interleaving.
    const int fd = ::open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) {
        return;
    }
    const std::string line = std::string(how) + " pid=" + std::to_string((long)getpid()) + "\n";
    const ssize_t ignored = ::write(fd, line.data(), line.size());
    (void)ignored;
    ::close(fd);
}


// A minimal control plane: it answers `hello`, `health` and `stats` with a
// report that says the card has `free_mb` megabytes left of `total_mb`.
int run_fake_worker(int argc, char ** argv) {
    int http_port = 0;
    int control_port = 0;
    long free_mb = 600;
    long total_mb = 2000;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--port" && i + 1 < argc) {
            http_port = atoi(argv[i + 1]);
        } else if (a == "--control-port" && i + 1 < argc) {
            control_port = atoi(argv[i + 1]);
        } else if (a == "--free-mb" && i + 1 < argc) {
            free_mb = atol(argv[i + 1]);
        } else if (a == "--total-mb" && i + 1 < argc) {
            total_mb = atol(argv[i + 1]);
        }
    }
    if (const char * e = std::getenv("TEST_ROUTER_FAKE_FREE_MB")) {
        free_mb = atol(e);
    }

    // The HTTP face.
    httplib::Server srv;
    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    srv.Get("/v1/models", [](const httplib::Request &, httplib::Response & res) {
        // Deliberately NOT the id the control plane reports: the router must
        // prefer the control plane's answer.
        res.set_content("{\"object\":\"list\",\"data\":[{\"id\":\"http-model\",\"object\":\"model\"}]}",
                        "application/json");
    });

    mark_exit("started");
    // A SIGTERM is the fallback path (a worker without a control plane), so
    // it has to be distinguishable from the graceful RPC one.
    // The signal handler only sets a flag and the real work does on a normal
    // thread - which is exactly what the real worker does (its handler sets
    // an atomic and a watchdog unwinds the server).  Doing the work IN the
    // handler (sleeping, opening a file) is not async-signal-safe, and it
    // is not what the router is being tested against.
    static std::atomic<bool> g_term{false};
    ::signal(SIGTERM, [](int) { g_term.store(true); });
    std::thread([&]() {
        while (!g_term.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        mark_exit("sigterm-start");
        std::this_thread::sleep_for(std::chrono::seconds(2)); // the "flush"
        mark_exit("sigterm-done");
        ::_exit(0);
    }).detach();

    // The control plane.
    const int cfd = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    ::setsockopt(cfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in caddr;
    std::memset(&caddr, 0, sizeof(caddr));
    caddr.sin_family = AF_INET;
    caddr.sin_port = htons((uint16_t)control_port);
    caddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (::bind(cfd, (struct sockaddr *)&caddr, sizeof(caddr)) != 0 || ::listen(cfd, 4) != 0) {
        fprintf(stderr, "fake worker: cannot bind the control port %d\n", control_port);
        return 2;
    }

    std::thread ctl([cfd, free_mb, total_mb]() {
        for (;;) {
            const int fd = ::accept(cfd, nullptr, nullptr);
            if (fd < 0) {
                return;
            }
            std::string buf;
            bool exit_after_reply = false;
            for (;;) {
                char tmp[4096];
                const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
                if (n <= 0) {
                    break;
                }
                buf.append(tmp, (size_t)n);
                size_t nl;
                while ((nl = buf.find('\n')) != std::string::npos) {
                    const std::string line = buf.substr(0, nl);
                    buf.erase(0, nl + 1);
                    json reply;
                    try {
                        const json req = json::parse(line);
                        const int64_t id = req.value("id", (int64_t)0);
                        const std::string method = req.value("method", std::string());
                        json result;
                        if (method == "hello") {
                            result["protocol"] = 2;
                            result["worker"] = report_json(free_mb, total_mb);
                        } else if (method == "health") {
                            result["ready"] = true;
                            result["uptime_s"] = 1;
                            result["inflight"] = 0;
                        } else if (method == "stats") {
                            result = report_json(free_mb, total_mb);
                        } else if (method == "shutdown") {
                            // The graceful stop: answer first, then leave,
                            // like the real worker (which unwinds after the
                            // reply).
                            result["ok"] = true;
                            exit_after_reply = true;
                            mark_exit("rpc-start");
                        } else {
                            reply["jsonrpc"] = "2.0";
                            reply["id"] = id;
                            reply["error"]["code"] = -32601;
                            reply["error"]["message"] = "unknown method";
                        }
                        if (reply.is_null()) {
                            reply["jsonrpc"] = "2.0";
                            reply["id"] = id;
                            reply["result"] = result;
                        }
                    } catch (...) {
                        reply["jsonrpc"] = "2.0";
                        reply["id"] = 0;
                        reply["error"]["code"] = -32700;
                        reply["error"]["message"] = "parse error";
                    }
                    const std::string out = reply.dump() + "\n";
                    ::send(fd, out.data(), out.size(), MSG_NOSIGNAL);
                    if (exit_after_reply) {
                        ::close(fd);
                        // The real worker flushes its prefix cache on the way
                        // out; the router must not kill it while it does.
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                        mark_exit("rpc-done");
                        ::_exit(0);
                    }
                }
            }
            ::close(fd);
        }
    });

    if (!srv.bind_to_port("127.0.0.1", http_port)) {
        return 2;
    }
    srv.listen_after_bind();
    return 0;
}

// ------------------------------------------------------------------- the tests

std::string self_path() {
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return std::string();
    }
    buf[n] = '\0';
    return std::string(buf);
}

// A deterministic port per test, not an ephemeral one: bind-and-close (the
// usual trick) leaves a window in which another process takes the port, and
// a test that fails once in twenty is worse than no test.
int free_port() {
    static int next = 21500;
    return next++;
}

model_config make_model(const std::string & binary, int http_port, int control_port, const std::string & group) {
    model_config m;
    m.id = "m" + std::to_string(http_port);
    m.path = "/etc/hostname";
    m.binary = binary;
    m.http_host = "127.0.0.1";
    m.http_port = http_port;
    m.control_host = "127.0.0.1";
    m.control_port = control_port;
    m.device_group = group;
    m.argv = build_worker_argv(m);
    return m;
}

bool wait_for(const std::shared_ptr<model_slot> & slot, model_state want, int timeout_s) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(slot->mtx);
            if (slot->state == want) {
                return true;
            }
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// The client speaks the protocol: a handshake, an error reply, and a
// connection that is not there.
void test_rpc_client() {
    const int port = free_port();
    // A one-shot server that answers exactly one request.
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    ::bind(fd, (struct sockaddr *)&addr, sizeof(addr));
    ::listen(fd, 2);
    std::thread srv([fd]() {
        const int c = ::accept(fd, nullptr, nullptr);
        if (c < 0) {
            return;
        }
        char tmp[4096];
        const ssize_t n = ::recv(c, tmp, sizeof(tmp), 0);
        if (n > 0) {
            const std::string out = "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":true}}\n";
            ::send(c, out.data(), out.size(), MSG_NOSIGNAL);
        }
        ::close(c);
    });

    rpc_client cli("127.0.0.1", port);
    std::string err;
    CHECK(cli.connect(2000, err), "the client must connect: %s", err.c_str());
    json result;
    CHECK(cli.call("hello", json{{"protocol", 1}}, result, err, 2000), "the call must succeed: %s", err.c_str());
    CHECK(result.value("ok", false), "the result must come back");
    srv.join();
    ::close(fd);

    // Nothing is listening on this port: the call must fail cleanly.
    const int dead = free_port();
    rpc_client cli2("127.0.0.1", dead);
    CHECK(!cli2.connect(1000, err), "connecting to nothing must fail");
    CHECK(!cli2.call("hello", json::object(), result, err, 1000), "calling without a connection must fail");
}

// The router must take the model's identity from the control plane, not
// from the worker's HTTP model list.
void test_identity_comes_from_the_control_plane() {
    const int http_port = free_port();
    const int ctl_port = free_port();
    registry reg;
    auto slot = reg.add(make_model(self_path(), http_port, ctl_port, "gpu.0"));

    router_config cfg;
    cfg.log_dir = "/tmp/sycl-infer-router-test-log";
    cfg.startup_timeout_s = 30;

    setenv("TEST_ROUTER_FAKE_WORKER", "1", 1);
    supervisor sup(reg, cfg);
    sup.start();

    CHECK(wait_for(slot, model_state::up, 30), "the worker must become ready");
    {
        std::lock_guard<std::mutex> lk(slot->mtx);
        CHECK(slot->advertised_id == "rpc-model", "the identity must come from the control plane, got '%s'",
              slot->advertised_id.c_str());
    }
    sup.stop();
    unsetenv("TEST_ROUTER_FAKE_WORKER");
}

// The empirical admission: the second model of a group is refused when it
// does not fit in what the card has left.
void test_vram_admission() {
    const int http_a = free_port();
    const int ctl_a = free_port();
    const int http_b = free_port();
    const int ctl_b = free_port();
    registry reg;
    model_config a = make_model(self_path(), http_a, ctl_a, "gpu.0");
    a.id = "a";
    a.vram_budget_mb = 400; // fits in the 600 MB the card reports free
    auto slot_a = reg.add(a);
    model_config b = make_model(self_path(), http_b, ctl_b, "gpu.0");
    b.id = "b";
    b.vram_budget_mb = 900; // does NOT fit
    auto slot_b = reg.add(b);

    router_config cfg;
    cfg.log_dir = "/tmp/sycl-infer-router-test-log";
    cfg.startup_timeout_s = 30;

    setenv("TEST_ROUTER_FAKE_WORKER", "1", 1);
    setenv("TEST_ROUTER_FAKE_FREE_MB", "600", 1);

    supervisor sup(reg, cfg);
    sup.start();

    CHECK(wait_for(slot_a, model_state::up, 30), "the first model must become ready");
    CHECK(wait_for(slot_b, model_state::failed, 30), "the second model must be refused");
    {
        std::lock_guard<std::mutex> lk(slot_b->mtx);
        CHECK(slot_b->last_error.find("VRAM") != std::string::npos ||
                  slot_b->last_error.find("free") != std::string::npos,
              "the refusal must say why: %s", slot_b->last_error.c_str());
    }
    sup.stop();
    unsetenv("TEST_ROUTER_FAKE_WORKER");
    unsetenv("TEST_ROUTER_FAKE_FREE_MB");
}

// The router must stop a worker GRACEFULLY: the control plane's `shutdown`
// first (the worker flushes its prefix cache on the way out), and the signal
// only as a fallback.
//
// The regression this pins: the router's supervisor thread and the router's
// stop path both waitpid'ed on the SAME child.  The loser of that race sees
// ECHILD, reads it as "the child is gone" and SIGKILLs a worker that was
// still shutting down - so the prefix cache flush was silently lost.
void test_graceful_shutdown() {
    const int http_port = free_port();
    const int ctl_port = free_port();
    registry reg;
    auto slot = reg.add(make_model(self_path(), http_port, ctl_port, "gpu.0"));

    router_config cfg;
    cfg.log_dir = "/tmp/sycl-infer-router-test-log";
    cfg.startup_timeout_s = 30;

    const std::string mark = "/tmp/sycl-infer-router-exit-mark";
    ::remove(mark.c_str());
    setenv("TEST_ROUTER_FAKE_WORKER", "1", 1);
    setenv("TEST_ROUTER_EXIT_MARK", mark.c_str(), 1);

    supervisor sup(reg, cfg);
    sup.start();
    CHECK(wait_for(slot, model_state::up, 30), "the worker must become ready");

    sup.stop();

    // The marker says HOW the worker left.  A SIGKILL leaves nothing behind.
    std::string how;
    {
        std::FILE * f = std::fopen(mark.c_str(), "r");
        CHECK(f != nullptr, "the worker must have recorded how it exited (a SIGKILL records nothing)");
        if (f) {
            // The WHOLE file: it holds one line per step, and the line that
            // says the shutdown finished can be well past a fixed buffer.
            char buf[4096];
            const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
            how.assign(buf, n);
            std::fclose(f);
        }
    }
    // The marker must carry the END of the shutdown ("-done"): a worker that
    // was killed while it was flushing leaves only the "-start".
    CHECK(how.find("-done") != std::string::npos,
          "the worker must have finished its shutdown, not been killed: '%s'", how.c_str());

    unsetenv("TEST_ROUTER_FAKE_WORKER");
    unsetenv("TEST_ROUTER_EXIT_MARK");
    ::remove(mark.c_str());
}

} // namespace

int main(int argc, char ** argv) {
    if (std::getenv("TEST_ROUTER_FAKE_WORKER")) {
        return run_fake_worker(argc, argv);
    }
    test_rpc_client();
    test_identity_comes_from_the_control_plane();
    test_vram_admission();
    test_graceful_shutdown();
    if (g_fail) {
        fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_router_control: OK\n");
    return 0;
}
