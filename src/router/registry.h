#pragma once

#include "config.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace si {
namespace router {

enum class model_state {
    starting,  // the process was just spawned
    loading,   // spawned, waiting for the model to load
    up,        // serving
    degraded,  // was up, the liveness probe failed
    restarting,// waiting out the restart backoff
    failed,    // gave up (a non-retryable error, or the restart budget ran out)
    stopping,  // asked to stop
};

const char * model_state_name(model_state s);

// One model: its configuration plus its live state.  The state is written by
// the supervisor and read by the proxy and the status endpoint, so it lives
// behind a mutex.
struct model_slot {
    model_config cfg;

    mutable std::mutex mtx;
    model_state state = model_state::starting;
    std::string last_error;
    int restarts = 0;
    long long pid = -1;
    long long ready_at = 0;  // epoch seconds, 0 while not up
    // What the worker advertised on /v1/models once it became ready.  The
    // router's aggregated /v1/models is built from these, and the proxy
    // rewrites the request's `model` field to the advertised id so the
    // worker's own /v1/models/{id} stays consistent.
    std::string advertised_id;
    std::string advertised_json;

    // Position within its device group, in configuration order.  The
    // supervisor starts a group's models in that order, one at a time, so
    // the order is deterministic rather than a race between the threads.
    int group_index = 0;

    std::string http_base() const {
        return cfg.http_base();
    }

    bool is_up() const {
        std::lock_guard<std::mutex> lk(mtx);
        return state == model_state::up;
    }

    void set_state(model_state s, const std::string & err = std::string()) {
        std::lock_guard<std::mutex> lk(mtx);
        state = s;
        if (!err.empty()) {
            last_error = err;
        }
    }

    std::string error() const {
        std::lock_guard<std::mutex> lk(mtx);
        return last_error;
    }
};

class registry {
public:
    std::shared_ptr<model_slot> add(model_config cfg);
    std::shared_ptr<model_slot> find(const std::string & id) const;
    const std::vector<std::shared_ptr<model_slot>> & all() const {
        return models_;
    }

private:
    std::vector<std::shared_ptr<model_slot>> models_;
};

} // namespace router
} // namespace si
