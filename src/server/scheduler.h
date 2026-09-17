#pragma once
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "chat_util.h"
#include "engine.h"

namespace si {

struct sequence {
    int id = 0;
    int slot = -1;
    std::vector<int> prompt;   // tokens to prefill
    int prompt_pos = 0;
    std::vector<int> blocks;   // KV cache blocks
    std::vector<int> recent;   // tokens for sampling penalties
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
    double wait_ms = 0;   // submit -> first prefill
    double admit_ms = 0;  // slot zero + block table setup
    double pf_ms = 0;     // summed time of all prefill forwards
    int n_chunks = 0;
    int reused = 0;       // prompt tokens served from the prefix cache

    // token/text stream out
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> text_out;
    int prompt_tokens = 0;

    void push(std::string s) {
        {
            std::lock_guard<std::mutex> lk(m);
            text_out.push_back(std::move(s));
        }
        cv.notify_all();
    }
    bool pop(std::string & out) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !text_out.empty() || finished; });
        if (text_out.empty()) return false;
        out = std::move(text_out.front());
        text_out.pop_front();
        return true;
    }
    bool pop_wait(std::string & out) { // blocking without spin for the server
        return pop(out);
    }
};

// Continuous batching scheduler: admits sequences (tokenized prompts), runs
// chunked prefill and batched decode steps, and streams tokens to the callers.
struct scheduler {
    engine & e;
    std::mutex m;
    std::condition_variable cv;  // wakes the loop on submit/shutdown instead of polling
    std::vector<std::shared_ptr<sequence>> waiting;
    std::vector<std::shared_ptr<sequence>> active;
    std::thread th;
    bool stopping = false;
    int next_id = 1;

    scheduler(engine & eng) : e(eng) {}
    ~scheduler() { shutdown(); }

    void start();
    void shutdown();
    std::shared_ptr<sequence> submit(std::vector<int> prompt, const gen_params & gp,
                                     std::vector<std::string> stops);

private:
    void loop();
    bool admit(std::shared_ptr<sequence> & s);
    void retire(std::shared_ptr<sequence> & s, const char * reason);
};

} // namespace si
