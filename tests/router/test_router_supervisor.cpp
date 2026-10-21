#include "registry.h"

#include "config.h"
#include "supervisor.h"

#include <httplib.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <signal.h>
#include <stdlib.h>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unistd.h>



using namespace si::router;

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

// ------------------------------------------------------------- the fake worker

// TEST_ROUTER_FAKE_WORKER=1 turns this binary into a worker.  It takes the
// port from the `--port` flag the supervisor derives, and its behaviour from
// TEST_ROUTER_FAKE_MODE:
//
//   serve  a well-behaved worker: /health and /v1/models answer
//   fail   a worker that cannot start: it prints the engine's own startup
//          error and exits
//
// TEST_ROUTER_FAKE_DELAY (seconds) makes the worker slow to come up, which is
// how the start-gate test observes that a second model waits.
//
int run_fake_worker(int argc, char ** argv) {
    int port = 0;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[i + 1]);
        }
    }
    const char * mode = std::getenv("TEST_ROUTER_FAKE_MODE");
    const std::string m = mode ? mode : "serve";
    if (m == "fail") {
        fprintf(stderr, "error: device alloc failed\n");
        fflush(stderr);
        return 1;
    }
    if (m == "fail78") {
        // The worker's own startup-failure code: it says "do not retry"
        // instead of the router having to guess from the log text.
        fprintf(stderr, "error: the model does not load\n");
        fprintf(stderr, "(exit %d: a startup failure - do not retry)\n", 78);
        fflush(stderr);
        return 78;
    }
    const char * delay = std::getenv("TEST_ROUTER_FAKE_DELAY");
    if (delay && atoi(delay) > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(atoi(delay)));
    }
    httplib::Server srv;
    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    srv.Get("/v1/models", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"object\":\"list\",\"data\":[{\"id\":\"fake\",\"object\":\"model\"}]}", "application/json");
    });
    if (!srv.bind_to_port("127.0.0.1", port)) {
        fprintf(stderr, "fake worker: cannot bind %d\n", port);
        return 2;
    }
    srv.listen_after_bind();
    return 0;
}

// ------------------------------------------------------------------ the tests

std::string self_path() {
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return std::string();
    }
    buf[n] = '\0';
    return std::string(buf);
}

// A deterministic port per test (see the note in test_router_control.cpp:
// bind-and-close leaves a window in which another process takes the port).
int free_port() {
    static int next = 22500;
    return next++;
}

bool wait_for(const std::shared_ptr<model_slot> & slot, model_state want, int timeout_s) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    for (;;) {
        {
            // The lock is taken for the check only: holding it across the
            // sleep would starve the supervisor thread, which needs the same
            // mutex to publish every state change.
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

model_config make_model(const std::string & binary, int port) {
    model_config m;
    m.id = "w";
    m.path = "/etc/hostname"; // never opened by the fake worker
    m.binary = binary;
    m.http_host = "127.0.0.1";
    m.http_port = port;
    m.argv = build_worker_argv(m);
    return m;
}

void test_ready_and_restart() {
    const int port = free_port();
    registry reg;
    auto slot = reg.add(make_model(self_path(), port));

    router_config cfg;
    cfg.log_dir = "/tmp/sycl-infer-router-test-log";
    cfg.startup_timeout_s = 20;
    cfg.health_interval_s = 1;
    cfg.max_restarts = 3;
    cfg.restart_backoff_s = 1;

    setenv("TEST_ROUTER_FAKE_WORKER", "1", 1);
    setenv("TEST_ROUTER_FAKE_MODE", "serve", 1);

    supervisor sup(reg, cfg);
    sup.start();

    CHECK(wait_for(slot, model_state::up, 25), "the worker must become ready");
    {
        std::lock_guard<std::mutex> lk(slot->mtx);
        CHECK(slot->advertised_id == "fake", "the advertised id must be read from the worker: %s",
              slot->advertised_id.c_str());
    }

    // Kill it: the supervisor must notice and put a new one in its place.
    pid_t pid = -1;
    {
        std::lock_guard<std::mutex> lk(slot->mtx);
        pid = (pid_t)slot->pid;
    }
    CHECK(pid > 0, "the supervisor must own a child");
    if (pid > 0) {
        ::kill(pid, SIGKILL);
    }
    bool restarted = false;
    for (int i = 0; i < 100; i++) {
        {
            std::lock_guard<std::mutex> lk(slot->mtx);
            if (slot->restarts >= 1) {
                restarted = true;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    CHECK(restarted, "a dead worker must be restarted");
    CHECK(wait_for(slot, model_state::up, 25), "the replacement must become ready");

    sup.stop();
    unsetenv("TEST_ROUTER_FAKE_WORKER");
    unsetenv("TEST_ROUTER_FAKE_MODE");
}

// A worker that cannot start is not a crash: it is a configuration or
// environment problem, and retrying it would fail the same way.  So it must
// NOT be restarted.
void test_failed_start_is_not_retried() {
    const int port = free_port();
    registry reg;
    auto slot = reg.add(make_model(self_path(), port));

    router_config cfg;
    cfg.log_dir = "/tmp/sycl-infer-router-test-log";
    cfg.startup_timeout_s = 20;
    cfg.max_restarts = 3;
    cfg.restart_backoff_s = 1;

    setenv("TEST_ROUTER_FAKE_WORKER", "1", 1);
    setenv("TEST_ROUTER_FAKE_MODE", "fail", 1);

    supervisor sup(reg, cfg);
    sup.start();

    CHECK(wait_for(slot, model_state::failed, 25), "a worker that cannot start must end up failed");
    {
        std::lock_guard<std::mutex> lk(slot->mtx);
        CHECK(slot->restarts == 0, "a startup failure must not be retried (restarts=%d)", slot->restarts);
        CHECK(slot->last_error.find("device alloc failed") != std::string::npos,
              "the error must carry the worker's own message: %s", slot->last_error.c_str());
    }

    sup.stop();
    unsetenv("TEST_ROUTER_FAKE_WORKER");
    unsetenv("TEST_ROUTER_FAKE_MODE");
}

// Two models that share a device group must not load at the same time: they
// would compete for the same card's memory and the loser would die with
// "device alloc failed".  So the second model of a group must wait for the
// first one to be ready.
void test_same_group_starts_serially() {
    const int port_a = free_port();
    const int port_b = free_port();
    registry reg;
    model_config a = make_model(self_path(), port_a);
    a.id = "a";
    a.device_group = "gpu0";
    a.env.push_back("TEST_ROUTER_FAKE_DELAY=4"); // slow to come up
    auto slot_a = reg.add(a);
    model_config b = make_model(self_path(), port_b);
    b.id = "b";
    b.device_group = "gpu0"; // the same card
    auto slot_b = reg.add(b);

    router_config cfg;
    cfg.log_dir = "/tmp/sycl-infer-router-test-log";
    cfg.startup_timeout_s = 30;
    cfg.max_restarts = 1;
    cfg.restart_backoff_s = 1;

    setenv("TEST_ROUTER_FAKE_WORKER", "1", 1);
    setenv("TEST_ROUTER_FAKE_MODE", "serve", 1);
    unsetenv("TEST_ROUTER_FAKE_DELAY");

    supervisor sup(reg, cfg);
    sup.start();

    // While the first model is still loading, the second must not have a
    // process at all.
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    {
        std::lock_guard<std::mutex> lk(slot_b->mtx);
        CHECK(slot_b->pid <= 0, "the second model of a group must wait for the first (pid=%lld)",
              (long long)slot_b->pid);
    }
    {
        std::lock_guard<std::mutex> lk(slot_a->mtx);
        CHECK(slot_a->state == model_state::loading, "the first model is loading: %s",
              model_state_name(slot_a->state));
    }

    // Both come up, one after the other.
    CHECK(wait_for(slot_a, model_state::up, 30), "the first model must become ready");
    CHECK(wait_for(slot_b, model_state::up, 30), "the second model must become ready after it");

    sup.stop();
    unsetenv("TEST_ROUTER_FAKE_WORKER");
    unsetenv("TEST_ROUTER_FAKE_MODE");
    unsetenv("TEST_ROUTER_FAKE_DELAY");
}

// A worker that reports the startup-failure exit code must not be retried,
// and the reason must name the code: "the same attempt would fail the same
// way" is the worker's own statement, not the router's guess.
void test_startup_exit_code_is_not_retried() {
    const int port = free_port();
    registry reg;
    auto slot = reg.add(make_model(self_path(), port));

    router_config cfg;
    cfg.log_dir = "/tmp/sycl-infer-router-test-log";
    cfg.startup_timeout_s = 20;
    cfg.max_restarts = 3;
    cfg.restart_backoff_s = 1;

    setenv("TEST_ROUTER_FAKE_WORKER", "1", 1);
    setenv("TEST_ROUTER_FAKE_MODE", "fail78", 1);

    supervisor sup(reg, cfg);
    sup.start();

    CHECK(wait_for(slot, model_state::failed, 25), "the worker must end up failed");
    {
        std::lock_guard<std::mutex> lk(slot->mtx);
        CHECK(slot->restarts == 0, "a startup failure must not be retried (restarts=%d)", slot->restarts);
        CHECK(slot->last_error.find("78") != std::string::npos,
              "the reason must name the exit code: %s", slot->last_error.c_str());
    }

    sup.stop();
    unsetenv("TEST_ROUTER_FAKE_WORKER");
    unsetenv("TEST_ROUTER_FAKE_MODE");
}

} // namespace

int main(int argc, char ** argv) {
    if (std::getenv("TEST_ROUTER_FAKE_WORKER")) {
        return run_fake_worker(argc, argv);
    }
    test_ready_and_restart();
    test_failed_start_is_not_retried();
    test_startup_exit_code_is_not_retried();
    test_same_group_starts_serially();
    if (g_fail) {
        fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_router_supervisor: OK\n");
    return 0;
}
