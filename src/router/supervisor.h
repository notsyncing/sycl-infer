#pragma once

#include "registry.h"
#include "rpc_client.h"

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace si {
namespace router {

// One device group's start gate: an ordered queue.  The models of a group
// start one at a time, in configuration order, so two models sharing a card
// never load at the same time (they would compete for the same card's
// initialisation memory) and the order is deterministic rather than a race
// between the supervisor's threads.
struct group_gate {
    std::mutex mtx;
    std::condition_variable cv;
    int next = 0;  // the index allowed to start next
    bool busy = false;
};

// Supervises one worker process per model.
//
// The supervisor is the only component that knows whether a worker exists:
// it spawns it, waits for it to become ready, watches it, restarts it when it
// dies, and stops it on shutdown.  The proxy only asks "is this model up?".
//
// Readiness is "the worker's HTTP port answers /health".  That is exact, not
// a heuristic: the worker constructs the engine (loading the GGUF, uploading
// the weights, sizing the KV pool) BEFORE it starts listening, so an
// answering port means the model is resident and serving.
class supervisor {
public:
    supervisor(registry & reg, const router_config & cfg);
    ~supervisor();

    void start();
    // Stop every worker (SIGTERM, then SIGKILL) and join the threads.
    void stop();

private:
    void run(std::shared_ptr<model_slot> slot);
    // Returns false when the child could not be started at all.
    bool spawn(const model_slot & slot, pid_t & pid, std::string & err);
    // Poll /health until it answers, the child dies, or the timeout expires.
    bool wait_ready(const std::shared_ptr<model_slot> & slot, pid_t pid, std::string & err);
    // The per-device-group start gate (see docs/design/16-router.md §9.2):
    // two models that share a card must not load at the same time, because
    // they would compete for the same card's initialisation memory.  The
    // first start of each model follows the configuration order; a RESTART
    // only has to wait for the group to be free.
    void group_enter(const std::string & group, int index, bool restart);
    void group_leave(const std::string & group, int index, bool restart);
    // The control plane: connect, and read the model's identity and the
    // device memory it occupies.  `first` asks for `hello` (the handshake),
    // otherwise `stats` (the periodic refresh).
    bool refresh_control(model_slot & slot, bool first, std::string & err);
    // The empirical VRAM admission: does this model still fit on the cards
    // its group has left?  (docs/design/16-router.md §9.1, layer ②)
    bool vram_admit(const model_slot & slot, std::string & why);
    void ledger_update(const model_slot & slot, const nlohmann::json & report);
    // Give a model's budget back: it is not resident any more.
    void ledger_release(const model_slot & slot);
    std::shared_ptr<rpc_client> control_client(const model_slot & slot);

    registry & reg_;
    router_config cfg_;
    std::atomic<bool> stopping_{false};
    std::vector<std::thread> threads_;
    std::mutex groups_mtx_;
    std::map<std::string, std::shared_ptr<group_gate>> groups_;

    // The VRAM ledger: per device group, what the cards have left and what
    // the group's models have already committed.
    struct ledger {
        uint64_t free_vram_bytes = 0;   // the sum over the group's devices
        uint64_t total_vram_bytes = 0;
        // What the group has committed PER CARD: a model's budget is its
        // total over the cards it runs on, so what one card carries is its
        // share (budget / device_count).
        uint64_t committed_share_bytes = 0;
        bool measured = false; // false = the cards did not report it
    };
    std::mutex ledger_mtx_;
    std::map<std::string, ledger> ledgers_;

    // The control-plane connections, by model id (the stop path needs them).
    std::mutex clients_mtx_;
    std::map<std::string, std::shared_ptr<rpc_client>> clients_;
};

// The tail of a worker's log file, for the error message and the status
// endpoint.  Exposed for the tests.
std::string read_log_tail(const std::string & path, size_t max_lines);

// The worker's executable: an explicit path, else PATH, else next to the
// router's own binary (the Docker layout puts both in the same directory).
std::string resolve_binary(const std::string & name);

} // namespace router
} // namespace si
