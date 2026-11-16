#pragma once

// Timestamped, asynchronous diagnostic logging.
//
// Every runtime log line goes to stderr.  `si::logf` formats the message on
// the CALLING thread (a format string and its varargs cannot outlive the
// call), stamps it with the current time and hands the finished bytes to one
// writer thread through a queue.  The caller never formats into a shared
// buffer, never waits for stderr and never serializes with another logging
// thread beyond the queue push itself - a deque append under a mutex held for
// a few nanoseconds, not the I/O.  The timestamp is taken at the log call and
// formatted by the writer, so it describes when the event happened even if the
// queue is momentarily behind.
//
// The writer is also where the line-start state lives, so the timestamp is
// added exactly once per physical line: a diagnostic sequence written as a
// header plus per-item fragments (`"[prompt ids]"`, then `" %d"` per token)
// is stamped once, not once per fragment.
//
// Limits, stated rather than hidden:
//   * the queue is bounded (kMaxQueued); if a producer outruns the writer the
//     new message is dropped (never the engine blocked on a log write) and the
//     count is reported on the next write;
//   * the queue is drained and the writer joined by an atexit handler, so a
//     normal exit (and an exit code 78 startup failure) flushes every pending
//     line; a SIGKILL naturally loses whatever was still queued;
//   * after the flush a late log call (a static destructor, say) writes
//     synchronously, so shutdown logging is not dropped.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace si {

namespace log_detail {

class async_logger {
public:
    struct record {
        std::string text;
        std::chrono::system_clock::time_point at;
    };

    async_logger() {
        try {
            th_ = std::thread(&async_logger::run, this);
        } catch (...) {
            // No thread available: degrade to synchronous output instead of
            // throwing from a log call.
            sync_only_ = true;
        }
    }

    async_logger(const async_logger &) = delete;
    async_logger & operator=(const async_logger &) = delete;

    void write(std::string msg, std::chrono::system_clock::time_point at) {
        if (!sync_only_) {
            std::unique_lock<std::mutex> lk(m_);
            if (!stopped_) {
                if (q_.size() >= kMaxQueued) {
                    dropped_.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                q_.push_back(record{std::move(msg), at});
                lk.unlock();
                cv_.notify_one();
                return;
            }
            lk.unlock();
        }
        // No writer (thread creation failed) or it has already been joined:
        // the process is exiting, so fall back to a direct write.
        std::lock_guard<std::mutex> lk(sync_mutex_);
        std::string out;
        append_message(out, record{std::move(msg), at});
        flush(out);
    }

    void stop_and_flush() {
        if (sync_only_) {
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            if (stopped_ || stopping_) {
                return;
            }
            stopping_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) {
            th_.join();
        }
        std::deque<record> rest;
        {
            std::lock_guard<std::mutex> lk(m_);
            stopped_ = true;
            rest.swap(q_);
        }
        if (!rest.empty()) {
            std::lock_guard<std::mutex> lk(sync_mutex_);
            std::string out;
            for (const record & r : rest) {
                append_message(out, r);
            }
            flush(out);
        }
    }

private:
    static constexpr size_t kMaxQueued = 65536;

    void run() {
        try {
            std::unique_lock<std::mutex> lk(m_);
            for (;;) {
                cv_.wait(lk, [this] { return !q_.empty() || stopping_; });
                if (q_.empty() && stopping_) {
                    return;
                }
                std::deque<record> batch;
                batch.swap(q_);
                lk.unlock();

                std::string out;
                for (const record & r : batch) {
                    append_message(out, r);
                }
                const std::size_t d = dropped_.exchange(0, std::memory_order_relaxed);
                if (d > 0) {
                    if (!line_start_) {
                        out.push_back('\n');
                        line_start_ = true;
                    }
                    append_message(out, record{"[log] dropped " + std::to_string(d) + " message(s): queue full\n",
                                               std::chrono::system_clock::now()});
                }
                flush(out);

                lk.lock();
            }
        } catch (...) {
            // A log writer must not terminate the process.  Whatever is left
            // in the queue is drained by stop_and_flush at exit; until then a
            // full queue drops new messages instead of blocking.
        }
    }

    void append_timestamp(std::string & out, std::chrono::system_clock::time_point at) {
        using namespace std::chrono;
        const std::time_t t = system_clock::to_time_t(at);
        const auto ms = duration_cast<milliseconds>(at.time_since_epoch()) % 1000;
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char buf[32];
        const std::size_t n = std::strftime(buf, sizeof buf, "%Y-%m-%d %H:%M:%S", &tm);
        out.push_back('[');
        out.append(buf, n);
        char suffix[8];
        std::snprintf(suffix, sizeof suffix, ".%03ld] ", (long)ms.count());
        out += suffix;
    }

    void append_message(std::string & out, const record & r) {
        if (r.text.empty()) {
            return;
        }
        if (line_start_) {
            append_timestamp(out, r.at);
        }
        out += r.text;
        line_start_ = r.text.back() == '\n';
    }

    void flush(const std::string & out) {
        if (!out.empty()) {
            std::fwrite(out.data(), 1, out.size(), stderr);
        }
    }

    std::mutex m_;
    std::condition_variable cv_;
    std::deque<record> q_;
    bool stopping_ = false; // set under m_, before the writer drains
    bool stopped_ = false;  // set under m_, after the writer is joined
    bool sync_only_ = false;
    bool line_start_ = true; // writer thread (and the post-join fallback)
    std::atomic<std::size_t> dropped_{0};
    std::mutex sync_mutex_; // post-join / no-thread writers
    std::thread th_;
};

inline async_logger *& instance() {
    static async_logger * p = nullptr;
    return p;
}

inline void at_exit() {
    if (async_logger * p = instance()) {
        p->stop_and_flush();
    }
}

inline async_logger & logger() {
    static async_logger * p = [] {
        auto * q = new async_logger(); // intentionally never deleted: static
                                        // destruction order must not free a
                                        // logger another destructor may call
        instance() = q;
        std::atexit(&at_exit);
        return q;
    }();
    return *p;
}

} // namespace log_detail

#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
inline void logf(const char * fmt, ...) {
    const auto at = std::chrono::system_clock::now();
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    char stackbuf[1024];
    try {
        const int n = std::vsnprintf(stackbuf, sizeof stackbuf, fmt, ap);
        if (n >= 0) {
            std::string msg;
            if ((std::size_t)n < sizeof stackbuf) {
                msg.assign(stackbuf, (std::size_t)n);
            } else {
                msg.resize((std::size_t)n + 1);
                std::vsnprintf(msg.data(), msg.size(), fmt, ap2);
                msg.resize((std::size_t)n);
            }
            log_detail::logger().write(std::move(msg), at);
        }
    } catch (...) {
        // A log call must never throw (it is used in error paths).
    }
    va_end(ap2);
    va_end(ap);
}

} // namespace si
