// SSE disconnect cancellation, without a model.
//
// The contract under test is "a client that goes away stops the generation".
// That was invisible to every generation test before: the emitted text is
// identical either way, and the only symptoms are a sequence that decodes to
// max_tokens against a dead socket and an httplib worker thread parked in
// join().  So this drives the real transport (sse_queue + sse_session +
// serve_sse_chunked from src/server/sse.h) over a real socket and kills the
// client mid-stream, then asserts the producer actually stopped.
//
// It also pins the pieces the transport relies on: sequence::cancel() wakes a
// blocked pop_token, a cancel that lands before submit still reaches the
// sequence, and post-cancel pushes are dropped instead of growing the queue.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "sse.h"

using namespace si;

static int g_fail = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                            \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

static bool wait_for(const std::function<bool()> & p, int ms = 3000) {
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < ms) {
        if (p()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return p();
}

// ---------------------------------------------------------------- unit pieces

// A cancel must wake a consumer blocked in pop_token rather than leaving it
// parked until the sequence happens to finish.  `exited` is only set once
// pop_token has returned false, so observing it still false is a race-free
// proof that the consumer really is blocked (nothing else can set it).
static void test_sequence_cancel_wakes_consumer() {
    auto s = std::make_shared<sequence>();
    std::atomic<int> got{0};
    std::atomic<bool> exited{false};
    std::thread th([&] {
        sequence::token_out o;
        while (s->pop_token(o)) {
            got++;
        }
        exited = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(!exited.load()); // blocked on the empty queue
    CHECK(got.load() == 0);
    s->cancel();
    CHECK(wait_for([&] { return exited.load(); }));
    th.join();
    CHECK(s->cancelled);
    // nothing queued after a cancel is drained
    s->push("b");
    sequence::token_out o;
    CHECK(!s->pop_token(o));
}

// The session must not miss a sequence registered after the cancel arrived.
static void test_track_after_cancel() {
    auto sc = std::make_shared<sse_session>();
    sc->q = std::make_shared<sse_queue>();
    sc->cancel();
    auto s = std::make_shared<sequence>();
    sc->track(s);
    CHECK(s->cancelled);
    CHECK(sc->cancelled());
    // a post-cancel push must not grow the queue (a runaway producer would
    // otherwise buffer a whole generation for nobody)
    sc->q->push("data: x\n\n");
    std::string item;
    CHECK(!sc->q->pop(item));
    CHECK(sc->q->is_cancelled());
}

// ---------------------------------------------------------------- end to end

// The writer is the only consumer of both queues, so neither may grow without
// bound when that consumer stalls.  sequence::out_q pauses the sequence;
// sse_queue cancels the session (dropping an event would corrupt the stream,
// and blocking would stall every request, since the mm/MTP producer holds
// engine::mtx for the whole forward).
static void test_backpressure_bounds() {
    // scheduler path: the queue stops growing, and the sequence reports full
    {
        auto s = std::make_shared<sequence>();
        CHECK(!s->out_queue_full());
        for (size_t i = 0; i < sequence::kMaxOutQueue; i++) {
            s->push("t");
            // the last push lands exactly on the cap, so "not full" only holds
            // while there is still room after it
            if (i + 1 < sequence::kMaxOutQueue) {
                CHECK(!s->out_queue_full());
            }
        }
        CHECK(s->out_queue_full()); // exactly at the cap, still bounded
        // The pause threshold is only *consulted* by the scheduler, so the queue
        // itself keeps accepting up to the hard cap - what it must never do is
        // grow without bound, and it must not drop tokens silently either.
        for (size_t i = sequence::kMaxOutQueue; i < sequence::kHardOutCap; i++) {
            s->push("t");
        }
        CHECK(s->out_q.size() == sequence::kHardOutCap);
        CHECK(!s->cancelled); // still healthy, the scheduler would be pausing
        // past the hard cap the sequence ends instead of truncating
        s->push("one too many");
        CHECK(s->out_q.size() == sequence::kHardOutCap);
        CHECK(s->cancelled);
        // draining makes room again, which is what resumes a paused sequence
        auto s2 = std::make_shared<sequence>();
        for (size_t i = 0; i < sequence::kMaxOutQueue; i++) {
            s2->push("t");
        }
        CHECK(s2->out_queue_full());
        sequence::token_out t;
        CHECK(s2->pop_token(t));
        CHECK(!s2->out_queue_full());
    }
    // streaming path: the queue cancels itself instead of buffering
    {
        sse_queue q;
        for (size_t i = 0; i < sse_queue::kMaxQueue; i++) {
            q.push("data: x\n\n");
            CHECK(!q.is_cancelled());
        }
        q.push("data: one too many\n\n");
        CHECK(q.backpressure_cancelled());
        CHECK(q.is_cancelled());
        // and the items were released, not kept
        std::string out;
        CHECK(!q.pop(out));
        // a session sharing that queue reports cancelled, so its producers stop
        auto sc = std::make_shared<sse_session>();
        sc->q = std::make_shared<sse_queue>();
        for (size_t i = 0; i <= sse_queue::kMaxQueue; i++) {
            sc->q->push("data: x\n\n");
        }
        CHECK(sc->cancelled());
    }
}

// Real httplib server, real chunked SSE response, real client socket that is
// closed with a TCP RST (SO_LINGER 0) in the middle of the stream.
static void test_disconnect_cancels_generation(int port) {
    auto sc = std::make_shared<sse_session>();
    sc->q = std::make_shared<sse_queue>();
    sc->remaining = 1;

    // Producer: stands in for a generation thread - one "token" per 2 ms, up
    // to 4000 of them, stopping only when it observes the cancel.  Without the
    // fix this runs to completion after the client is gone.
    std::atomic<int> produced{0};
    std::atomic<bool> saw_cancel{false};
    std::atomic<bool> joined{false};
    sc->ths.emplace_back([sc, &produced, &saw_cancel] {
        for (int i = 0; i < 4000; i++) {
            if (sc->cancelled()) {
                saw_cancel = true;
                break;
            }
            sc->q->push("data: {\"i\":" + std::to_string(i) + "}\n\n");
            produced++;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (!sc->cancelled()) {
            saw_cancel = true; // ran to the end without ever being cancelled
        }
        sc->choice_done(1, produced);
    });

    httplib::Server srv;
    srv.Post("/stream", [&](const httplib::Request &, httplib::Response & res) {
        res.set_chunked_content_provider("text/event-stream",
                                         [](size_t, httplib::DataSink & sink) {
                                             sink.write("data: x\n\n", 11);
                                             return true;
                                         });
    });
    // replace the handler with the production wiring
    srv.Post("/sse", [&](const httplib::Request &, httplib::Response & res) { serve_sse_chunked(res, sc); });
    std::thread sth([&] { srv.listen("127.0.0.1", port); });
    CHECK(wait_for([&] { return srv.is_running(); }));

    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(fd >= 0);
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    sa.sin_addr.s_addr = inet_addr("127.0.0.1");
    CHECK(::connect(fd, (sockaddr *)&sa, sizeof(sa)) == 0);
    const std::string req = "POST /sse HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n";
    CHECK(::send(fd, req.data(), req.size(), MSG_NOSIGNAL) == (ssize_t)req.size());

    // read until the first SSE data frame actually arrives (proves the stream
    // is live before we kill it)
    std::string buf;
    char tmp[4096];
    bool got_frame = false;
    const auto t0 = std::chrono::steady_clock::now();
    while (!got_frame && std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() <
                              3000) {
        ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) {
            break;
        }
        buf.append(tmp, (size_t)n);
        if (buf.find("data: {") != std::string::npos) {
            got_frame = true;
        }
    }
    CHECK(got_frame);
    CHECK(buf.find("200") != std::string::npos); // chunked response started

    // hard reset, the "user closed the tab" case
    const int produced_at_kill = produced.load();
    {
        linger lg{1, 0}; // RST rather than FIN
        ::setsockopt(fd, SOL_SOCKET, SO_LINGER, &lg, sizeof(lg));
    }
    ::close(fd);

    // the producer must observe the cancel promptly and the worker thread must
    // come back (join is inside the releaser, which runs on that thread)
    CHECK(wait_for([&] { return saw_cancel.load(); }));
    CHECK(wait_for([&] { return sc->cancelled(); }));
    CHECK(produced.load() < 4000); // stopped early, not ran to the end
    CHECK(produced.load() < produced_at_kill + 400);
    CHECK(saw_cancel && produced.load() > 0);

    srv.stop();
    sth.join();
    sc->join();
    joined = true;
    CHECK(joined);
}

int main(int argc, char ** argv) {
    const int port = argc > 1 ? atoi(argv[1]) : 18131;
    test_sequence_cancel_wakes_consumer();
    test_track_after_cancel();
    test_backpressure_bounds();
    test_disconnect_cancels_generation(port);
    if (g_fail) {
        fprintf(stderr, "test_sse_cancel: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_sse_cancel: all checks OK\n");
    return 0;
}
