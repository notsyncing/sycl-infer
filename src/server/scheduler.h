#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "engine.h"

namespace si {

struct sequence {
    int id = 0;
    int slot = -1;
    std::vector<int> prompt; // tokens to prefill
    int prompt_pos = 0;
    std::vector<int> blocks; // KV cache blocks
    std::vector<int> recent; // tokens for sampling penalties
    gen_params gp;
    sampler_state ss;
    bool admitted = false;
    std::atomic<bool> finished{false};
    std::string finish_reason = "stop";
    int n_generated = 0;
    // set of stop strings
    std::vector<std::string> stops;

    // One generated token: `text` is the UTF-8-safe piece (empty when the
    // token's bytes are still incomplete) and carries the sampled token id /
    // logprob when the request asked for logprobs or best_of scoring.
    struct token_out {
        std::string text;
        int id = -1;
        float logprob = 0.f;
        std::vector<std::pair<int, float>> top; // (token id, logprob)
    };

    // ---- embedded C API mode ------------------------------------------------
    // The C API asks the scheduler to keep the whole token stream on the
    // sequence (no bounded out_q) and to apply its stop strings here, so a
    // request can be submitted without a consumer draining it.  The HTTP
    // server leaves both false and keeps its existing bounded queue.
    bool collect_all = false;
    bool apply_stops = false;
    bool api_mode = false;
    std::vector<token_out> tokens; // collect_all: every sampled token
    size_t read_pos = 0;           // collect_all: next token syi_request_next returns
    std::string text;              // collect_all: emitted text, stops filtered
    std::string error;             // asynchronous failure message
    bool stop_hit = false;
    bool failed = false;
    bool timed_out = false;
    std::chrono::steady_clock::time_point t_first{};
    std::chrono::steady_clock::time_point t_done{};
    // Request-level deadline (0 = none).  Set before the sequence enters the
    // waiting queue, so the loop never races the submitter.
    int64_t timeout_ms = 0;
    std::chrono::steady_clock::time_point deadline{};

    // Stop filter: hold back max_stop_len-1 bytes so a stop string split across
    // token pieces is detected before its prefix is emitted.  Only used when
    // apply_stops is set (C API); the server filters after out_q as before.
    struct stop_filter {
        std::vector<std::string> stops;
        std::string buf;
        size_t keep = 0;
        bool stopped = false;
        void init(const std::vector<std::string> & s) {
            stops = s;
            keep = 0;
            for (const auto & x : s) {
                keep = std::max(keep, x.size());
            }
            keep = keep > 0 ? keep - 1 : 0;
        }
        std::string feed(const std::string & piece) {
            if (stopped) {
                return {};
            }
            buf += piece;
            size_t cut = std::string::npos;
            for (const auto & s : stops) {
                if (s.empty()) {
                    continue;
                }
                const size_t p = buf.find(s);
                if (p != std::string::npos) {
                    cut = std::min(cut, p);
                }
            }
            if (cut != std::string::npos) {
                std::string out = buf.substr(0, cut);
                buf.clear();
                stopped = true;
                return out;
            }
            if (buf.size() > keep) {
                std::string out = buf.substr(0, buf.size() - keep);
                buf.erase(0, buf.size() - keep);
                return out;
            }
            return {};
        }
        std::string flush() {
            std::string out = buf;
            buf.clear();
            return out;
        }
    };
    stop_filter sf;

    // PF_SRV_TIME diagnostics (unused/null on the default path)
    std::chrono::steady_clock::time_point t_submit;
    double wait_ms = 0;  // submit -> first prefill
    double admit_ms = 0; // slot zero + block table setup
    double pf_ms = 0;    // summed time of all prefill forwards
    int n_chunks = 0;
    int reused = 0; // prompt tokens served from the prefix cache

    // Set when the client is gone and this sequence should stop at the next
    // scheduler iteration: the loop retires it (freeing the slot and its KV
    // blocks) instead of decoding to max_tokens.  An atomic because the SSE
    // releaser sets it from the HTTP thread while the loop reads it.
    std::atomic<bool> cancelled{false};

    // token/text stream out
    mutable std::mutex m;
    std::condition_variable cv;
    std::deque<token_out> out_q;
    int prompt_tokens = 0;

    // Backlog cap for out_q.  A client that reads slower than the engine
    // decodes would otherwise grow this deque without bound - and the HTTP
    // writer is the only consumer, so nothing else ever fails.  The scheduler
    // loop is shared by every sequence, so it must NOT block here (one slow
    // client would freeze every other request); instead a full sequence is left
    // out of the decode batch until its consumer drains it, which bounds memory
    // without letting one client stall the others.  256 tokens is ~17 s of
    // decode on a 27B, far beyond any healthy TCP backlog.
    static constexpr size_t kMaxOutQueue = 256;
    // Hard ceiling enforced by push_token itself, twice the pause threshold.
    // The scheduler pausing a full sequence is what normally keeps out_q under
    // kMaxOutQueue, but that is a property of the *call site*; a new producer
    // that forgets the gate would otherwise reintroduce the unbounded buffer.
    // Refusing to grow past this point cancels the sequence instead of dropping
    // tokens, because a dropped token silently truncates the response.
    static constexpr size_t kHardOutCap = 2 * kMaxOutQueue;
    bool out_queue_full() const {
        if (collect_all) {
            return false;
        }
        std::lock_guard<std::mutex> lk(m);
        return out_q.size() >= kMaxOutQueue;
    }

    void cancel() {
        cancelled.store(true, std::memory_order_relaxed);
        // wake a producer blocked in pop_token
        cv.notify_all();
    }
    void push(std::string s) {
        token_out t;
        t.text = std::move(s);
        push_token(std::move(t));
    }
    void push_token(token_out t) {
        {
            std::lock_guard<std::mutex> lk(m);
            if (out_q.size() >= kHardOutCap) {
                // See kHardOutCap: end the request rather than truncate it or
                // grow forever.  The token is discarded; nobody can read it.
                cancelled.store(true, std::memory_order_relaxed);
                cv.notify_all();
                return;
            }
            out_q.push_back(std::move(t));
        }
        cv.notify_all();
    }
    bool pop(std::string & out) {
        token_out t;
        if (!pop_token(t)) {
            return false;
        }
        out = std::move(t.text);
        return true;
    }
    bool pop_token(token_out & out) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !out_q.empty() || finished || cancelled.load(std::memory_order_relaxed); });
        // drain nothing after a cancel: the client that wanted these is gone
        if (cancelled.load(std::memory_order_relaxed) || out_q.empty()) {
            return false;
        }
        out = std::move(out_q.front());
        out_q.pop_front();
        return true;
    }
};

// Continuous batching scheduler: admits sequences (tokenized prompts), runs
// chunked prefill and batched decode steps, and streams tokens to the callers.
struct scheduler {
    engine & e;
    std::mutex m;
    std::condition_variable cv; // wakes the loop on submit/shutdown instead of polling
    std::vector<std::shared_ptr<sequence>> waiting;
    std::vector<std::shared_ptr<sequence>> active;
    std::thread th;
    bool stopping = false;
    int next_id = 1;
    // Optional C-API cap on decode rows (0 = the model profile / kMaxB).
    int decode_row_cap = 0;

    scheduler(engine & eng) : e(eng) {
    }
    ~scheduler() {
        shutdown();
    }

    void start();
    void shutdown();
    // `timeout_ms` is a request-level deadline (0 = none).  It is stored on the
    // sequence before it is queued, so the loop cannot race the submitter.
    std::shared_ptr<sequence> submit(std::vector<int> prompt, const gen_params & gp,
                                     std::vector<std::string> stops, int64_t timeout_ms = 0,
                                     bool api_mode = false);

    // How many sequences are running.  The control plane publishes it, so it
    // has to be read under the same lock the loop uses.
    int inflight();
    int waiting_count();
    // Cancel every waiting/active sequence (used by the C API's model
    // shutdown).  The loop retires them and releases their slots.
    void cancel_all();

private:
    void loop();
    bool admit(std::shared_ptr<sequence> & s);
    void retire(std::shared_ptr<sequence> & s, const char * reason);
};

} // namespace si
