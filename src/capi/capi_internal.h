#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sycl_infer/sycl_infer.h>
#include "engine.h"
#include "scheduler.h"

namespace si::capi {

struct model_state;
struct request_state;

void set_last_error(const std::string & msg);
syi_status fail(syi_status status, const std::string & msg);

struct model_state : std::enable_shared_from_this<model_state> {
    std::unique_ptr<engine> e;
    std::unique_ptr<scheduler> sched;

    std::thread dispatcher;
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::shared_ptr<request_state>> pending;
    bool stopping = false;
    bool flush_window = false;
    bool dispatcher_done = false;
    bool accepting = true;
    bool scheduler_stopped = false;

    int batch_window_us = 2000;
    int max_batch_rows = 0;
    int max_queue_depth = 1024;
    int64_t max_queue_tokens = 4194304;
    int request_timeout_ms = 0;
    int max_slots = 16;
    int64_t queued_tokens = 0;

    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> admitted{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> cancelled{0};
    std::atomic<uint64_t> failed{0};
    std::atomic<uint64_t> generated_tokens{0};
    std::atomic<uint64_t> prefill_steps{0};
    std::atomic<uint64_t> decode_steps{0};

    // stable strings for syi_model_info
    std::string model_name;
    std::string architecture;
    std::string device_name;

    syi_log_fn log_fn = nullptr;
    void *log_user = nullptr;

    ~model_state();
    void start();
    void dispatcher_loop();
    syi_status submit_one(const std::shared_ptr<request_state> & r);
    syi_status shutdown(uint32_t timeout_ms);
    void stop_dispatcher();

    void log(int level, const std::string & msg) {
        if (log_fn) {
            log_fn(level, msg.c_str(), log_user);
        }
    }
};

struct request_state : std::enable_shared_from_this<request_state> {
    std::shared_ptr<model_state> model;
    mutable std::mutex m;
    std::condition_variable cv;
    std::shared_ptr<sequence> seq;
    syi_request_state state = SYI_REQUEST_CREATED;
    std::string prompt_text;
    std::vector<int> prompt_tokens;
    bool has_tokens = false;
    bool submitted = false;
    bool cancel_requested = false;
    bool timed_out = false;
    bool failed = false;
    bool finalized = false;
    bool counted = false;
    int64_t timeout_ms = 0;
    gen_params gp;
    std::vector<std::string> stops;
    std::string error;

    // last syi_request_next result, so the returned views stay valid until the
    // next call (the design's per-call lifetime)
    std::string last_text;
    std::vector<syi_logprob> last_top;
    std::vector<std::string> last_top_text;
    uint64_t next_index = 0;

    // lazily built result views
    std::vector<int32_t> result_ids;
    std::string result_text;
    std::string result_error;

    void set_error(const std::string & e) {
        std::lock_guard<std::mutex> lk(m);
        error = e;
        failed = true;
    }
    bool terminal() const {
        if (!seq) {
            return state == SYI_REQUEST_FINISHED || state == SYI_REQUEST_FAILED ||
                   state == SYI_REQUEST_CANCELLED;
        }
        return seq->finished.load(std::memory_order_acquire);
    }
    syi_finish_reason finish_reason() const {
        if (timed_out) {
            return SYI_FINISH_TIMEOUT;
        }
        if (failed) {
            return SYI_FINISH_ERROR;
        }
        if (cancel_requested) {
            return SYI_FINISH_CANCELLED;
        }
        if (!seq) {
            return state == SYI_REQUEST_CANCELLED ? SYI_FINISH_CANCELLED : SYI_FINISH_NONE;
        }
        if (seq->finish_reason == "error") {
            return SYI_FINISH_ERROR;
        }
        if (seq->finish_reason == "timeout") {
            return SYI_FINISH_TIMEOUT;
        }
        if (seq->finish_reason == "length") {
            return SYI_FINISH_LENGTH;
        }
        if (seq->finish_reason == "stop") {
            return SYI_FINISH_STOP;
        }
        return SYI_FINISH_NONE;
    }
};

} // namespace si::capi

/* The C ABI's opaque handles.  The public header forward-declares these names;
   the state objects are shared so a request can outlive the model handle. */
struct syi_model {
    std::shared_ptr<si::capi::model_state> st;
};

struct syi_request {
    std::shared_ptr<si::capi::request_state> st;
};
