#include "server.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "chat.h"
#include "httplib.h"
#include "json.hpp"
#include "scheduler.h"

using json = nlohmann::json;

namespace si {

namespace {

// Generated token pieces can split a multi-byte UTF-8 sequence; the stream
// buffer normally holds the tail back, but a genuinely malformed sequence must
// not abort the whole server.  Encode such bytes as U+FFFD instead of throwing.
std::string dump_json(const json & j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

struct sse_queue {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> items;
    bool done = false;

    void push(std::string s) {
        {
            std::lock_guard<std::mutex> lk(m);
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
    bool pop(std::string & out) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !items.empty() || done; });
        if (items.empty()) return false;
        out = std::move(items.front());
        items.pop_front();
        return true;
    }
};

gen_params parse_params(const json & body) {
    gen_params gp;
    auto getf = [&](const char * k, float d) {
        return body.contains(k) && body[k].is_number() ? body[k].get<float>() : d;
    };
    auto geti = [&](const char * k, int d) {
        return body.contains(k) && body[k].is_number() ? body[k].get<int>() : d;
    };
    if (body.contains("max_tokens")) gp.max_tokens = geti("max_tokens", 256);
    else if (body.contains("max_completion_tokens")) gp.max_tokens = geti("max_completion_tokens", 256);
    gp.temperature = getf("temperature", 1.0f);
    gp.top_p = getf("top_p", 0.95f);
    gp.top_k = geti("top_k", 40);
    gp.min_p = getf("min_p", 0.0f);
    gp.repeat_penalty = getf("repetition_penalty", getf("repeat_penalty", 1.0f));
    gp.repeat_last_n = geti("repeat_last_n", 64);
    gp.presence_penalty = getf("presence_penalty", 0.0f);
    gp.frequency_penalty = getf("frequency_penalty", 0.0f);
    gp.ignore_eos = body.value("ignore_eos", false);
    if (body.contains("seed") && body["seed"].is_number()) {
        long long s = body["seed"].get<long long>();
        gp.seed = (uint64_t) s;
    }
    if (gp.max_tokens <= 0) gp.max_tokens = 256;
    return gp;
}

std::vector<std::string> parse_stop(const json & body) {
    std::vector<std::string> stops;
    if (!body.contains("stop")) return stops;
    const json & s = body["stop"];
    if (s.is_string()) stops.push_back(s.get<std::string>());
    else if (s.is_array()) {
        for (auto & v : s)
            if (v.is_string()) stops.push_back(v.get<std::string>());
    }
    return stops;
}

std::vector<chat_msg> parse_messages(const json & body) {
    std::vector<chat_msg> msgs;
    if (!body.contains("messages") || !body["messages"].is_array()) return msgs;
    for (auto & m : body["messages"]) {
        chat_msg cm;
        cm.role = m.value("role", "user");
        if (m.contains("content")) {
            const json & c = m["content"];
            if (c.is_string()) {
                cm.content = c.get<std::string>();
            } else if (c.is_array()) {
                for (auto & part : c) {
                    if (part.is_object() && part.contains("text") && part["text"].is_string())
                        cm.content += part["text"].get<std::string>();
                }
            }
        }
        msgs.push_back(std::move(cm));
    }
    return msgs;
}

std::string make_chat_response(const std::string & id, const std::string & model,
                               const std::string & text, const std::string & finish,
                               int prompt_tokens, int completion_tokens, uint64_t created) {
    json resp = {
        {"id", id},
        {"object", "chat.completion"},
        {"created", created},
        {"model", model},
        {"choices", json::array({{
            {"index", 0},
            {"message", {{"role", "assistant"}, {"content", text}}},
            {"finish_reason", finish},
        }})},
        {"usage", {{"prompt_tokens", prompt_tokens},
                   {"completion_tokens", completion_tokens},
                   {"total_tokens", prompt_tokens + completion_tokens}}},
    };
    return dump_json(resp);
}

std::string make_completion_response(const std::string & id, const std::string & model,
                                     const std::string & text, const std::string & finish,
                                     int prompt_tokens, int completion_tokens, uint64_t created) {
    json resp = {
        {"id", id},
        {"object", "text_completion"},
        {"created", created},
        {"model", model},
        {"choices", json::array({{
            {"index", 0},
            {"text", text},
            {"finish_reason", finish},
        }})},
        {"usage", {{"prompt_tokens", prompt_tokens},
                   {"completion_tokens", completion_tokens},
                   {"total_tokens", prompt_tokens + completion_tokens}}},
    };
    return dump_json(resp);
}

std::string gen_id() {
    static std::atomic<uint64_t> counter{0};
    char buf[64];
    snprintf(buf, sizeof(buf), "chatcmpl-%llu", (unsigned long long) counter.fetch_add(1) + 1);
    return buf;
}

// A prompt that does not fit max_seq cannot be prefilled (its block table row
// only has max_seq/kBlockSize entries).  Reject it with an explicit HTTP 400
// instead of letting scheduler::admit retire the sequence silently (HTTP 200,
// empty stream, finish_reason=length, 0 tokens).
bool reject_too_long(const engine & e, size_t prompt_tokens, httplib::Response & res) {
    if ((int) prompt_tokens <= e.max_seq) return false;
    json j = {{"error",
               {{"message", "prompt length " + std::to_string(prompt_tokens) +
                                " tokens exceeds the server context (max_seq=" +
                                std::to_string(e.max_seq) + "); restart with --ctx " +
                                std::to_string(prompt_tokens) + " (or PF_CTX) or shorten the prompt"},
                {"type", "invalid_request_error"},
                {"code", "context_length_exceeded"}}},
              {"max_seq", e.max_seq},
              {"prompt_tokens", (int) prompt_tokens}};
    res.status = 400;
    res.set_content(dump_json(j), "application/json");
    fprintf(stderr, "[http] 400 context_length_exceeded: prompt=%zu tokens > max_seq=%d\n",
            prompt_tokens, e.max_seq);
    return true;
}

struct stream_ctx {
    std::shared_ptr<sse_queue> q;
    std::thread th;
};

// generate into an SSE stream from the scheduler; returns after starting the producer thread
void run_stream(scheduler & sched, engine & e, std::shared_ptr<stream_ctx> st,
                std::vector<int> prompt, const gen_params & gp,
                const std::vector<std::string> & stops,
                const std::string & model, const std::string & id, bool chat, uint64_t created,
                bool include_usage) {
    st->q = std::make_shared<sse_queue>();
    auto seq = sched.submit(std::move(prompt), gp, stops);
    st->th = std::thread([=]() {
        std::string text;
        bool stopped = false;
        auto * q = st->q.get();
        auto send_delta = [&](const std::string & piece) {
            json j;
            if (chat) {
                j = {{"id", id}, {"object", "chat.completion.chunk"}, {"created", created},
                     {"model", model},
                     {"choices", json::array({{{"index", 0},
                                               {"delta", {{"content", piece}}},
                                               {"finish_reason", nullptr}}})}};
            } else {
                j = {{"id", id}, {"object", "text_completion"}, {"created", created},
                     {"model", model},
                     {"choices", json::array({{{"index", 0}, {"text", piece},
                                               {"finish_reason", nullptr}}})}};
            }
            q->push("data: " + dump_json(j) + "\n\n");
        };
        std::string item;
        while (seq->pop(item)) {
            text += item;
            size_t cut = std::string::npos;
            for (const auto & s : seq->stops) {
                if (s.empty()) continue;
                size_t p = text.rfind(s);
                if (p != std::string::npos) cut = std::min(cut, p);
            }
            if (cut != std::string::npos) {
                std::string send = text.substr(0, cut);
                if (!send.empty()) send_delta(send);
                stopped = true;
                break;
            }
            if (!item.empty()) send_delta(item);
        }
        const std::string finish = stopped ? "stop" : seq->finish_reason;
        json finalj;
        if (chat) {
            finalj = {{"id", id}, {"object", "chat.completion.chunk"}, {"created", created},
                      {"model", model},
                      {"choices", json::array({{{"index", 0}, {"delta", json::object()},
                                                {"finish_reason", finish}}})}};
        } else {
            finalj = {{"id", id}, {"object", "text_completion"}, {"created", created},
                      {"model", model},
                      {"choices", json::array({{{"index", 0}, {"text", ""},
                                                {"finish_reason", finish}}})}};
        }
        q->push("data: " + dump_json(finalj) + "\n\n");
        if (include_usage) {
            json uj = {{"id", id}, {"object", "chat.completion.chunk"}, {"created", created},
                       {"model", model}, {"choices", json::array()},
                       {"usage", {{"prompt_tokens", seq->prompt_tokens},
                                  {"completion_tokens", seq->n_generated},
                                  {"total_tokens", seq->prompt_tokens + seq->n_generated}}}};
            q->push("data: " + dump_json(uj) + "\n\n");
        }
        q->push("data: [DONE]\n\n");
        q->finish();
    });
}

} // namespace

int serve(engine & e, const server_config & cfg) {
    scheduler sched(e);
    sched.start();
    httplib::Server srv;
    srv.set_read_timeout(3600, 0);
    srv.set_write_timeout(3600, 0);
    srv.set_payload_max_length(64 * 1024 * 1024);

    auto cors = [](httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    };
    srv.Options(".*", [=](const httplib::Request &, httplib::Response & res) {
        cors(res);
        res.status = 204;
    });

    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json j = {{"object", "list"},
                  {"data", json::array({{{"id", cfg.model_id},
                                         {"object", "model"},
                                         {"created", 1700000000},
                                         {"owned_by", "local"}}})}};
        res.set_content(dump_json(j), "application/json");
    });

    auto handle_chat = [&](const httplib::Request & req, httplib::Response & res) {
        cors(res);
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"invalid json\"}}", "application/json");
            return;
        }
        auto msgs = parse_messages(body);
        bool thinking = false;
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
            thinking = body["chat_template_kwargs"].value("enable_thinking", false);
        std::string text = render_chat(msgs, true, thinking);
        static const bool srv_t = getenv("PF_SRV_TIME") != nullptr;
        const auto t_tok0 = std::chrono::steady_clock::now();
        auto prompt = e.tk.encode(text);
        if (srv_t)
            fprintf(stderr, "[http] tokenize=%.2f ms chars=%zu tokens=%zu\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_tok0)
                        .count(),
                    text.size(), prompt.size());
        if (reject_too_long(e, prompt.size(), res)) return;
        gen_params gp = parse_params(body);
        auto stops = parse_stop(body);
        const bool stream = body.value("stream", false);
        const std::string id = gen_id();
        const uint64_t created = (uint64_t) time(nullptr);
        const std::string model = body.value("model", cfg.model_id);

        if (!stream) {
            auto seq = sched.submit(std::move(prompt), gp, stops);
            std::string out_text, item;
            bool stopped = false;
            while (seq->pop(item)) {
                out_text += item;
                for (const auto & s : seq->stops) {
                    if (s.empty()) continue;
                    size_t p = out_text.rfind(s);
                    if (p != std::string::npos) {
                        out_text.resize(p);
                        stopped = true;
                        break;
                    }
                }
                if (stopped) break;
            }
            const std::string finish = stopped ? "stop" : seq->finish_reason;
            res.set_content(make_chat_response(id, model, out_text, finish, seq->prompt_tokens,
                                               seq->n_generated, created),
                            "application/json");
            return;
        }

        auto st = std::make_shared<stream_ctx>();
        bool include_usage = false;
        if (body.contains("stream_options") && body["stream_options"].is_object())
            include_usage = body["stream_options"].value("include_usage", false);
        run_stream(sched, e, st, prompt, gp, stops, model, id, true, created, include_usage);
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider(
            "text/event-stream",
            [st](size_t, httplib::DataSink & sink) {
                std::string item;
                if (!st->q->pop(item)) {
                    sink.done();
                    return true;
                }
                sink.write(item.data(), item.size());
                return true;
            },
            [st](bool) {
                if (st->th.joinable()) st->th.join();
            });
    };
    srv.Post("/v1/chat/completions", handle_chat);

    auto handle_completion = [&](const httplib::Request & req, httplib::Response & res) {
        cors(res);
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"invalid json\"}}", "application/json");
            return;
        }
        std::vector<int> prompt;
        static const bool srv_t = getenv("PF_SRV_TIME") != nullptr;
        const auto t_tok0 = std::chrono::steady_clock::now();
        size_t prompt_chars = 0;
        if (body.contains("prompt")) {
            if (body["prompt"].is_string()) {
                const std::string & ps = body["prompt"].get<std::string>();
                prompt_chars = ps.size();
                prompt = e.tk.encode(ps);
            } else if (body["prompt"].is_array() && !body["prompt"].empty()) {
                const std::string & ps = body["prompt"][0].get<std::string>();
                prompt_chars = ps.size();
                prompt = e.tk.encode(ps);
            }
        }
        if (srv_t)
            fprintf(stderr, "[http] tokenize=%.2f ms chars=%zu tokens=%zu\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_tok0)
                        .count(),
                    prompt_chars, prompt.size());
        if (reject_too_long(e, prompt.size(), res)) return;
        gen_params gp = parse_params(body);
        auto stops = parse_stop(body);
        const bool stream = body.value("stream", false);
        const std::string id = gen_id();
        const uint64_t created = (uint64_t) time(nullptr);
        const std::string model = body.value("model", cfg.model_id);

        if (!stream) {
            auto seq = sched.submit(std::move(prompt), gp, stops);
            std::string out_text, item;
            bool stopped = false;
            while (seq->pop(item)) {
                out_text += item;
                for (const auto & s : seq->stops) {
                    if (s.empty()) continue;
                    size_t p = out_text.rfind(s);
                    if (p != std::string::npos) {
                        out_text.resize(p);
                        stopped = true;
                        break;
                    }
                }
                if (stopped) break;
            }
            const std::string finish = stopped ? "stop" : seq->finish_reason;
            res.set_content(make_completion_response(id, model, out_text, finish, seq->prompt_tokens,
                                                     seq->n_generated, created),
                            "application/json");
            return;
        }

        auto st = std::make_shared<stream_ctx>();
        bool include_usage = false;
        if (body.contains("stream_options") && body["stream_options"].is_object())
            include_usage = body["stream_options"].value("include_usage", false);
        run_stream(sched, e, st, prompt, gp, stops, model, id, false, created, include_usage);
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider(
            "text/event-stream",
            [st](size_t, httplib::DataSink & sink) {
                std::string item;
                if (!st->q->pop(item)) {
                    sink.done();
                    return true;
                }
                sink.write(item.data(), item.size());
                return true;
            },
            [st](bool) {
                if (st->th.joinable()) st->th.join();
            });
    };
    srv.Post("/v1/completions", handle_completion);

    printf("server listening on %s:%d\n", cfg.host.c_str(), cfg.port);
    fflush(stdout);
    if (!srv.listen(cfg.host.c_str(), cfg.port)) {
        fprintf(stderr, "failed to listen on %s:%d\n", cfg.host.c_str(), cfg.port);
        return 1;
    }
    return 0;
}

} // namespace si
