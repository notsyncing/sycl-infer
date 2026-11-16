#pragma once

#include <atomic>
#include <chrono>
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
// Cancellation has exactly one input: the socket's state, as the writer sees
// it.  `sink.write` returning false is the direct signal; the periodic
// `sink.is_writable()` probe covers a long silent prefill; and httplib's
// `ContentProviderResourceReleaser` bool is the final backstop.  A client that
// closed the tab must not leave the sequence decoding to max_tokens while the
// HTTP worker thread sits in join().
namespace si {

// Items produced by the generation thread(s), consumed by the HTTP writer.
struct sse_queue {
    mutable std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> items;
    bool done = false;
    bool cancelled = false;
    // Set the first time push() refuses an item, i.e. the consumer fell behind
    // by more than the cap.  Observable so a test can assert the bound without
    // counting threads.
    bool overflowed = false;
    // Backlog cap.  Same reasoning as sequence::kMaxOutQueue: the writer is the
    // only consumer, so an unbounded queue is an unbounded buffer.  This path
    // (multimodal / MTP) runs on one producer thread that holds engine::mtx for
    // the whole forward, so blocking here would stall every other request -
    // and dropping an SSE event would silently corrupt the stream.  Neither is
    // acceptable, so exceeding the cap cancels the session instead: a client
    // that cannot drain 4096 events (minutes of decode) is treated exactly
    // like one that disconnected.  backpressure_cancelled() reports which.
    static constexpr size_t kMaxQueue = 4096;

    void push(std::string s) {
        {
            std::lock_guard<std::mutex> lk(m);
            if (cancelled) {
                return; // nobody is reading; drop rather than grow
            }
            if (items.size() >= kMaxQueue) {
                overflowed = true;
                cancelled = true;
                items.clear();
                cv.notify_all();
                return;
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
    // True when the queue hit kMaxQueue rather than a client disconnect.
    bool backpressure_cancelled() const {
        std::lock_guard<std::mutex> lk(m);
        return overflowed;
    }
    // false once finished or cancelled.  With `timeout_ms >= 0` the call also
    // returns false on a timeout, and `timed_out` says which: the SSE writer
    // uses that to probe the socket while a long prefill has produced no
    // tokens yet (httplib only checks the peer between provider calls).
    bool pop(std::string & out, int timeout_ms = -1, bool * timed_out = nullptr) {
        std::unique_lock<std::mutex> lk(m);
        const auto ready = [&] { return !items.empty() || done || cancelled; };
        bool have = true;
        if (timeout_ms < 0) {
            cv.wait(lk, ready);
        } else {
            have = cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
        }
        if (timed_out) {
            *timed_out = !have;
        }
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
        // The queue also cancels itself when its backlog cap is hit, so
        // producers must consult it: `stop` alone would miss the overflow and
        // keep generating for a client that never caught up.
        return stop.load(std::memory_order_relaxed) || q->is_cancelled();
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
            bool timed_out = false;
            // Wait with a timeout: a long prefill produces no items for
            // seconds or minutes, and httplib only checks the peer *between*
            // provider calls.  A blocking pop would therefore not notice a
            // client that vanished during the prefill until the first token -
            // the whole prompt would still be prefilled.  One probe a second
            // costs nothing and cancels the sequence at the next scheduler
            // chunk boundary (the in-flight forward is not interruptible).
            if (!sc->q->pop(item, 1000, &timed_out)) {
                if (timed_out) {
                    if (sink.is_writable && !sink.is_writable()) {
                        sc->cancel();
                        return false;
                    }
                    return true; // nothing to hand over yet; probe again
                }
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
