#pragma once
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
    bool finished = false;
    std::string finish_reason = "stop";
    int n_generated = 0;
    // set of stop strings
    std::vector<std::string> stops;

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

    // One generated token: `text` is the UTF-8-safe piece (empty when the
    // token's bytes are still incomplete) and carries the sampled token id /
    // logprob when the request asked for logprobs or best_of scoring.
    struct token_out {
        std::string text;
        int id = -1;
        float logprob = 0.f;
        std::vector<std::pair<int, float>> top; // (token id, logprob)
    };

    // token/text stream out
    std::mutex m;
    std::condition_variable cv;
    std::deque<token_out> out_q;
    int prompt_tokens = 0;

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

    scheduler(engine & eng) : e(eng) {
    }
    ~scheduler() {
        shutdown();
    }

    void start();
    void shutdown();
    std::shared_ptr<sequence> submit(std::vector<int> prompt, const gen_params & gp, std::vector<std::string> stops);

private:
    void loop();
    bool admit(std::shared_ptr<sequence> & s);
    void retire(std::shared_ptr<sequence> & s, const char * reason);
};

} // namespace si
