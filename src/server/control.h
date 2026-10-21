#pragma once

// The worker's control plane: a JSON-RPC 2.0 server over TCP.
//
// It is what the router talks to when it needs more than the OpenAI HTTP
// API: the model's identity, the device memory the model occupies, the KV
// pool's state, and a graceful shutdown that flushes the prefix cache.
//
// The protocol is in `src/common/jsonrpc.h` (shared with the router's
// client).  See docs/design/16-router.md §5 and §7.

#include "json.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace si {

using json = nlohmann::json;

// What the worker publishes about itself.
struct control_report {
    // One entry per device the model runs on.
    struct device_t {
        int index = 0;                 // the physical GPU ordinal
        std::string name;
        uint64_t total_vram_bytes = 0;
        // What THIS model leaves on the card: the card's capacity minus what
        // the model put on it.  This is the number the router's admission
        // uses, because the driver's own figure does not count the SYCL
        // allocations on the Level Zero backend.
        uint64_t free_vram_bytes = 0;
        // The driver's figure, when it reports one.  Informational only.
        uint64_t driver_free_vram_bytes = 0;
        uint64_t used_bytes = 0;       // what this model occupies
        uint64_t kv_bytes = 0;         // the KV pool this device holds
    };
    std::vector<device_t> devices;

    uint64_t kv_used_bytes = 0;
    uint64_t kv_cap_bytes = 0;
    int kv_pool_blocks = 0;
    int kv_pool_cap = 0;
    int kv_used_blocks = 0;

    uint64_t pc_vram_bytes = 0;
    int pc_ram_records = 0;
    int pc_disk_records = 0;

    uint64_t host_rss_bytes = 0;
    int inflight = 0; // the sequences the scheduler is running
    int64_t uptime_s = 0;

    // The model's identity, as the OpenAI API advertises it.
    std::string model_id;
    int64_t max_seq = 0;
    int64_t n_vocab = 0;
    int64_t n_layer = 0;
};

// The JSON shape of a report.  Shared by the worker (which fills it) and the
// router (which reads it), so the two cannot drift.
json control_report_to_json(const control_report & r);

// The process's resident set size (from /proc/self/statm).  0 if it cannot
// be read.
uint64_t host_rss_bytes();

class control_server {
public:
    // `report` is called on the control thread when a `stats` or `hello`
    // arrives; `terminate` asks the HTTP server to unwind (the same path a
    // SIGTERM takes, so the prefix cache is flushed).
    //
    // `token` is the shared secret: when it is empty the control plane is
    // open (the loopback default); when it is set, the handshake must carry
    // it and every later request on that connection is authenticated.
    control_server(std::function<control_report()> report, std::function<void()> terminate,
                   std::string token = std::string());
    ~control_server();

    // Bind and listen.  `host` defaults to the loopback: the control plane
    // has no authentication unless a token is configured (docs/design/16-router.md
    // §5.4), so it must not be exposed by accident.
    bool start(const std::string & host, int port, std::string & err);
    void stop();
    int port() const {
        return port_;
    }
    // Whether a token is required.  The startup line says so, because
    // "the control plane is open" is the thing an operator has to know.
    bool needs_token() const {
        return !token_.empty();
    }

private:
    // One accepted connection.  The accept loop reaps the finished ones, so
    // a long-lived worker does not accumulate thread objects.
    struct conn {
        std::thread th;
        std::atomic<bool> done{false};
    };

    void accept_loop();
    void serve_conn(int fd);
    json dispatch(const json & req, bool & authenticated);

    std::function<control_report()> report_;
    std::function<void()> terminate_;
    std::string token_;

    int listen_fd_ = -1;
    int port_ = 0;
    std::atomic<bool> stopping_{false};
    std::thread accept_thread_;
    std::vector<std::shared_ptr<conn>> conns_;
    std::mutex conn_mtx_;
};

} // namespace si
