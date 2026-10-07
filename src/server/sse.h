#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "httplib.h"
#include "scheduler.h"

// Server-Sent Events transport: the item queue, the per-response session, and
// the chunked writer wiring.  Split out of server.cpp so the cancellation
// contract is unit-testable without a model - it is a pure producer/consumer
// plus httplib's write result, and getting it wrong is invisible in a
// generation test (the output looks identical; only the wasted work and the
// blocked worker thread differ).
//
// Cancellation has exactly one input: httplib's `ContentProviderResourceReleaser`
// bool, which is false both when a write fails and when the peer hung up.  It
// is the only disconnect signal httplib offers, and the previous code ignored
// it - a client that closed the tab left the sequence decoding to max_tokens
// while the HTTP worker thread sat in join().
namespace si {

// Items produced by the generation thread(s), consumed by the HTTP writer.
struct sse_queue {
    mutable std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> items;
    bool done = false;
    bool cancelled = false;

    void push(std::string s) {
        {
            std::lock_guard<std::mutex> lk(m);
            if (cancelled) {
                return; // nobody is reading; drop rather than grow
            }
            items.push_back(std::move(s));
        }
        cv.notify_one();
    }
    void finish() {
        {
            std::lock_guard<std::mutex> lk(m);
            done = true;
        }
        cv.notify_one();
    }
    // Disconnect: wake a writer blocked in pop() and drop what is still queued,
    // so neither end stays blocked on a socket nobody is reading.
    void cancel() {
        {
            std::lock_guard<std::mutex> lk(m);
            cancelled = true;
            items.clear();
        }
        cv.notify_all();
    }
    bool is_cancelled() const {
        std::lock_guard<std::mutex> lk(m);
        return cancelled;
    }
    // false once finished or cancelled
    bool pop(std::string & out) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !items.empty() || done || cancelled; });
        if (cancelled || items.empty()) {
            return false;
        }
        out = std::move(items.front());
        items.pop_front();
        return true;
    }
};

// One SSE response: the queue, the producer thread(s), and the shared cancel
// state.  Deliberately free of the JSON chunk helpers (file-local in
// server.cpp) - the optional usage chunk is emitted through usage_emitter.
struct sse_session {
    std::shared_ptr<sse_queue> q;
    std::vector<std::thread> ths;
    std::mutex m;
    int remaining = 0;
    long long prompt_tokens = 0;
    long long completion_tokens = 0;
    long long reasoning_tokens = 0;
    long long cached_tokens = 0;
    bool include_usage = false;
    bool chat = true;
    std::string id;
    std::string model;
    uint64_t created = 0;
    // Set by the server: emits the usage chunk that must precede [DONE].
    std::function<void(const sse_session &)> usage_emitter;

    // Set once the response is no longer being written.  Producers poll it at
    // every token boundary - inside a forward the callback return value is the
    // only cancellation point, so a cancelled generation still finishes the
    // decode step it is in (one step, ~65 ms on a 27B decode).
    std::atomic<bool> stop{false};
    // Scheduler-backed sequences of this response, so a cancel also retires
    // them: the slot and KV blocks go back at the next scheduler iteration
    // instead of after the rest of max_tokens.
    std::vector<std::weak_ptr<sequence>> seqs;

    void track(const std::shared_ptr<sequence> & s) {
        {
            std::lock_guard<std::mutex> lk(m);
            seqs.push_back(s);
        }
        // A cancel that landed before this sequence was submitted would
        // otherwise never reach it and it would run to max_tokens.
        if (stop.load(std::memory_order_relaxed)) {
            s->cancel();
        }
    }
    bool cancelled() const {
        return stop.load(std::memory_order_relaxed);
    }
    void cancel() {
        stop.store(true, std::memory_order_relaxed);
        q->cancel();
        std::vector<std::weak_ptr<sequence>> live;
        {
            std::lock_guard<std::mutex> lk(m);
            live.swap(seqs);
        }
        // outside the lock: sequence::cancel takes the sequence's own mutex
        for (auto & w : live) {
            if (auto s = w.lock()) {
                s->cancel();
            }
        }
    }
    void choice_done(long long pt, long long ct, long long rt = 0, long long cached = 0) {
        {
            std::lock_guard<std::mutex> lk(m);
            prompt_tokens += pt;
            completion_tokens += ct;
            reasoning_tokens += rt;
            cached_tokens += cached;
            if (--remaining > 0) {
                return;
            }
        }
        // Outside the lock: both pushes are dropped after a cancel, so the
        // final frames cost nothing on a disconnected client.
        if (include_usage && usage_emitter) {
            usage_emitter(*this);
        }
        q->push("data: [DONE]\n\n");
        q->finish();
    }
    void join() {
        for (auto & t : ths) {
            if (t.joinable()) {
                t.join();
            }
        }
    }
};

// Attach the session to the response.  The completion callback runs from
// ~Response with the stream's success flag: false means the peer went away or
// the write failed, and is the only disconnect signal available.
inline void serve_sse_chunked(httplib::Response & res, const std::shared_ptr<sse_session> & sc) {
    res.set_chunked_content_provider(
        "text/event-stream",
        [sc](size_t, httplib::DataSink & sink) {
            std::string item;
            if (!sc->q->pop(item)) {
                sink.done();
                return true;
            }
            // A false return is httplib learning the peer is gone.  Ignoring it
            // (as this used to) keeps the stream "successful" and lets the
            // generation run to completion against a dead socket.
            if (!sink.write(item.data(), item.size())) {
                sc->cancel();
                return false; // -> Error::Canceled -> releaser gets ok=false
            }
            return true;
        },
        [sc](bool ok) {
            if (!ok) {
                sc->cancel();
            }
            sc->join();
        });
}

} // namespace si
