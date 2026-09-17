#include "server.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include "chat.h"
#include "chat_util.h"
#include "httplib.h"
#include "image.h"
#include "json.hpp"
#include "multimodal.h"
#include "scheduler.h"
#include "vision.h"

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
        if (items.empty()) {
            return false;
        }
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
    auto geti = [&](const char * k, int d) { return body.contains(k) && body[k].is_number() ? body[k].get<int>() : d; };
    if (body.contains("max_tokens")) {
        gp.max_tokens = geti("max_tokens", 256);
    } else if (body.contains("max_completion_tokens")) {
        gp.max_tokens = geti("max_completion_tokens", 256);
    }
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
        gp.seed = (uint64_t)s;
    }
    if (gp.max_tokens <= 0) {
        gp.max_tokens = 256;
    }
    return gp;
}

std::vector<std::string> parse_stop(const json & body) {
    std::vector<std::string> stops;
    if (!body.contains("stop")) {
        return stops;
    }
    const json & s = body["stop"];
    if (s.is_string()) {
        stops.push_back(s.get<std::string>());
    } else if (s.is_array()) {
        for (auto & v : s) {
            if (v.is_string()) {
                stops.push_back(v.get<std::string>());
            }
        }
    }
    return stops;
}

// ---------------------------------------------------------------- vision input
// Loaded once at startup; the vision forward runs on the host and reads only
// the (immutable) mmproj weights, so the mutex serializes preprocessing/encode
// against other concurrent multimodal requests.
struct mm_server {
    vision_model vm;
    image_preproc_cfg cfg;
    bool ready = false;
    std::mutex m;
};

int b64_val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') {
        return c - 'A';
    }
    if (c >= 'a' && c <= 'z') {
        return c - 'a' + 26;
    }
    if (c >= '0' && c <= '9') {
        return c - '0' + 52;
    }
    if (c == '+') {
        return 62;
    }
    if (c == '/') {
        return 63;
    }
    return -1;
}

bool b64_decode(const std::string & s, std::vector<uint8_t> & out) {
    out.clear();
    int val = 0, bits = 0;
    for (unsigned char c : s) {
        if (c == '=') {
            break;
        }
        if (c == '\n' || c == '\r' || c == ' ') {
            continue;
        }
        const int v = b64_val(c);
        if (v < 0) {
            return false;
        }
        val = (val << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t)(val >> bits));
        }
    }
    return true;
}

// Only base64 data URLs are accepted; fetching remote URLs would need an HTTP
// client and would expose the server as an SSRF proxy.
bool decode_image_url(const std::string & url, std::vector<uint8_t> & bytes, std::string & err) {
    if (url.rfind("data:", 0) != 0) {
        err = "unsupported image URL (only data: URLs are accepted)";
        return false;
    }
    const size_t comma = url.find(',');
    if (comma == std::string::npos) {
        err = "malformed data URL";
        return false;
    }
    if (url.substr(5, comma - 5).find(";base64") == std::string::npos) {
        err = "only base64 data URLs are supported";
        return false;
    }
    if (!b64_decode(url.substr(comma + 1), bytes)) {
        err = "invalid base64 image data";
        return false;
    }
    return true;
}

std::vector<chat_msg> parse_messages(const json & body, std::vector<std::string> & image_urls) {
    std::vector<chat_msg> msgs;
    if (!body.contains("messages") || !body["messages"].is_array()) {
        return msgs;
    }
    for (auto & m : body["messages"]) {
        chat_msg cm;
        cm.role = m.value("role", "user");
        bool has_image = false;
        if (m.contains("content")) {
            const json & c = m["content"];
            if (c.is_string()) {
                cm.content = c.get<std::string>();
            } else if (c.is_array()) {
                for (auto & part : c) {
                    if (!part.is_object()) {
                        continue;
                    }
                    const std::string type = part.value("type", "text");
                    if (type == "image_url" || type == "image") {
                        std::string url;
                        if (part.contains("image_url")) {
                            const json & iu = part["image_url"];
                            url = iu.is_string() ? iu.get<std::string>() : iu.value("url", std::string());
                        } else {
                            url = part.value("image", std::string());
                        }
                        image_urls.push_back(url);
                        cm.parts.push_back({true, ""});
                        has_image = true;
                    } else if (part.contains("text") && part["text"].is_string()) {
                        const std::string t = part["text"].get<std::string>();
                        cm.content += t;
                        cm.parts.push_back({false, t});
                    }
                }
            }
        }
        if (!has_image) {
            cm.parts.clear(); // text-only messages keep the plain path
        }
        msgs.push_back(std::move(cm));
    }
    return msgs;
}

std::string make_chat_response(const std::string & id, const std::string & model, const std::string & text,
                               const std::string & finish, int prompt_tokens, int completion_tokens, uint64_t created) {
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
        {"usage",
         {{"prompt_tokens", prompt_tokens},
          {"completion_tokens", completion_tokens},
          {"total_tokens", prompt_tokens + completion_tokens}}},
    };
    return dump_json(resp);
}

std::string make_completion_response(const std::string & id, const std::string & model, const std::string & text,
                                     const std::string & finish, int prompt_tokens, int completion_tokens,
                                     uint64_t created) {
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
        {"usage",
         {{"prompt_tokens", prompt_tokens},
          {"completion_tokens", completion_tokens},
          {"total_tokens", prompt_tokens + completion_tokens}}},
    };
    return dump_json(resp);
}

std::string gen_id() {
    static std::atomic<uint64_t> counter{0};
    char buf[64];
    snprintf(buf, sizeof(buf), "chatcmpl-%llu", (unsigned long long)counter.fetch_add(1) + 1);
    return buf;
}

// A prompt that does not fit max_seq cannot be prefilled (its block table row
// only has max_seq/kBlockSize entries).  Reject it with an explicit HTTP 400
// instead of letting scheduler::admit retire the sequence silently (HTTP 200,
// empty stream, finish_reason=length, 0 tokens).
bool reject_too_long(const engine & e, size_t prompt_tokens, httplib::Response & res) {
    if ((int)prompt_tokens <= e.max_seq) {
        return false;
    }
    json j = {
        {"error",
         {{"message", "prompt length " + std::to_string(prompt_tokens) + " tokens exceeds the server context (max_seq="
                          + std::to_string(e.max_seq) + "); restart with --ctx " + std::to_string(prompt_tokens)
                          + " (or PF_CTX) or shorten the prompt"},
          {"type", "invalid_request_error"},
          {"code", "context_length_exceeded"}}},
        {"max_seq", e.max_seq},
        {"prompt_tokens", (int)prompt_tokens}};
    res.status = 400;
    res.set_content(dump_json(j), "application/json");
    fprintf(stderr, "[http] 400 context_length_exceeded: prompt=%zu tokens > max_seq=%d\n", prompt_tokens, e.max_seq);
    return true;
}

struct stream_ctx {
    std::shared_ptr<sse_queue> q;
    std::thread th;
};

// generate into an SSE stream from the scheduler; returns after starting the producer thread
void run_stream(scheduler & sched, engine & e, std::shared_ptr<stream_ctx> st, std::vector<int> prompt,
                const gen_params & gp, const std::vector<std::string> & stops, const std::string & model,
                const std::string & id, bool chat, uint64_t created, bool include_usage) {
    st->q = std::make_shared<sse_queue>();
    auto seq = sched.submit(std::move(prompt), gp, stops);
    st->th = std::thread([=]() {
        std::string text;
        bool stopped = false;
        auto * q = st->q.get();
        auto send_delta = [&](const std::string & piece) {
            json j;
            if (chat) {
                j = {{"id", id},
                     {"object", "chat.completion.chunk"},
                     {"created", created},
                     {"model", model},
                     {"choices",
                      json::array({{{"index", 0}, {"delta", {{"content", piece}}}, {"finish_reason", nullptr}}})}};
            } else {
                j = {{"id", id},
                     {"object", "text_completion"},
                     {"created", created},
                     {"model", model},
                     {"choices", json::array({{{"index", 0}, {"text", piece}, {"finish_reason", nullptr}}})}};
            }
            q->push("data: " + dump_json(j) + "\n\n");
        };
        std::string item;
        while (seq->pop(item)) {
            text += item;
            size_t cut = std::string::npos;
            for (const auto & s : seq->stops) {
                if (s.empty()) {
                    continue;
                }
                size_t p = text.rfind(s);
                if (p != std::string::npos) {
                    cut = std::min(cut, p);
                }
            }
            if (cut != std::string::npos) {
                std::string send = text.substr(0, cut);
                if (!send.empty()) {
                    send_delta(send);
                }
                stopped = true;
                break;
            }
            if (!item.empty()) {
                send_delta(item);
            }
        }
        const std::string finish = stopped ? "stop" : seq->finish_reason;
        json finalj;
        if (chat) {
            finalj = {{"id", id},
                      {"object", "chat.completion.chunk"},
                      {"created", created},
                      {"model", model},
                      {"choices", json::array({{{"index", 0}, {"delta", json::object()}, {"finish_reason", finish}}})}};
        } else {
            finalj = {{"id", id},
                      {"object", "text_completion"},
                      {"created", created},
                      {"model", model},
                      {"choices", json::array({{{"index", 0}, {"text", ""}, {"finish_reason", finish}}})}};
        }
        q->push("data: " + dump_json(finalj) + "\n\n");
        if (include_usage) {
            json uj = {{"id", id},
                       {"object", "chat.completion.chunk"},
                       {"created", created},
                       {"model", model},
                       {"choices", json::array()},
                       {"usage",
                        {{"prompt_tokens", seq->prompt_tokens},
                         {"completion_tokens", seq->n_generated},
                         {"total_tokens", seq->prompt_tokens + seq->n_generated}}}};
            q->push("data: " + dump_json(uj) + "\n\n");
        }
        q->push("data: [DONE]\n\n");
        q->finish();
    });
}

} // namespace

struct mm_run_result {
    std::string text;
    std::string finish = "length";
    int n_gen = 0;
};

// Run a multimodal prompt on the single-sequence path, forwarding decoded text
// pieces to `on_piece` (return false to stop early).
void run_mm_generate(engine & e, const mm_prompt & mp, const gen_params & gp, const std::vector<std::string> & stops,
                     const std::function<bool(const std::string &)> & on_piece, mm_run_result & out) {
    utf8_stream_buffer ub;
    std::string text;
    bool stopped = false;
    auto cb = [&](int tok) -> bool {
        const std::string piece = ub.push(e.tk.token_piece(tok));
        if (piece.empty()) {
            return true;
        }
        text += piece;
        out.n_gen++;
        size_t cut = std::string::npos;
        for (const auto & s : stops) {
            if (s.empty()) {
                continue;
            }
            const size_t p = text.rfind(s);
            if (p != std::string::npos) {
                cut = std::min(cut, p);
            }
        }
        if (cut != std::string::npos) {
            const std::string send = text.substr(0, cut);
            if (!send.empty()) {
                on_piece(send);
            }
            stopped = true;
            return false;
        }
        return on_piece(piece);
    };
    e.generate_mm(mp, gp, cb);
    const std::string tail = ub.flush();
    if (!tail.empty() && !stopped) {
        on_piece(tail);
    }
    out.text = text;
    out.finish = stopped ? "stop" : "length";
}

int serve(engine & e, const server_config & cfg) {
    scheduler sched(e);
    sched.start();
    std::mutex mm_req;
    mm_server mm;
    if (!cfg.mmproj_path.empty()) {
        try {
            mm.vm.load(cfg.mmproj_path);
            mm.cfg.patch_size = mm.vm.hp.patch_size;
            mm.cfg.merge = mm.vm.hp.merge;
            const int patch_area = mm.cfg.patch_size * mm.cfg.patch_size * mm.cfg.merge * mm.cfg.merge;
            mm.cfg.min_pixels = 8 * patch_area;
            mm.cfg.max_pixels = kMaxImgTokens * patch_area;
            for (int c = 0; c < 3; c++) {
                mm.cfg.mean[c] = mm.vm.hp.mean[c];
                mm.cfg.std[c] = mm.vm.hp.std[c];
            }
            mm.ready = true;
            fprintf(stderr, "[mm] vision projector loaded: %s (%d layers, %dx%d patches, merge %d)\n",
                    cfg.mmproj_path.c_str(), mm.vm.hp.n_layer, mm.vm.hp.image_size, mm.vm.hp.image_size,
                    mm.vm.hp.merge);
        } catch (const std::exception & ex) {
            fprintf(stderr, "[mm] cannot load mmproj %s: %s (image input disabled)\n", cfg.mmproj_path.c_str(),
                    ex.what());
        }
    }
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
                  {"data",
                   json::array(
                       {{{"id", cfg.model_id}, {"object", "model"}, {"created", 1700000000}, {"owned_by", "local"}}})}};
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
        std::vector<std::string> image_urls;
        auto msgs = parse_messages(body, image_urls);
        bool thinking = false;
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
            thinking = body["chat_template_kwargs"].value("enable_thinking", false);
        }
        gen_params gp = parse_params(body);
        auto stops = parse_stop(body);
        const bool stream = body.value("stream", false);
        const std::string id = gen_id();
        const uint64_t created = (uint64_t)time(nullptr);
        const std::string model = body.value("model", cfg.model_id);

        if (!image_urls.empty()) {
            if (!mm.ready) {
                res.status = 400;
                res.set_content("{\"error\":{\"message\":\"image input requires --mmproj\"}}", "application/json");
                return;
            }
            // multimodal requests share the engine's image-embedding buffer, so
            // they are serialized for their whole lifetime (build + generate)
            auto req_lk = std::make_shared<std::unique_lock<std::mutex>>(mm_req);
            mm_prompt mp;
            try {
                std::vector<mm_image> imgs;
                for (const std::string & u : image_urls) {
                    std::vector<uint8_t> bytes, rgb;
                    std::string err;
                    int w = 0, h = 0;
                    if (!decode_image_url(u, bytes, err)
                        || !mm_image_decode_mem(bytes.data(), bytes.size(), rgb, w, h, &err)) {
                        throw std::runtime_error(err);
                    }
                    imgs.push_back(mm_image_preprocess(rgb.data(), w, h, mm.cfg));
                }
                const std::string rendered = render_chat(e.m.chat_template, msgs, true, thinking);
                mp = mm_build_prompt_device(mm.vm, e.q, e.tk, rendered, imgs, e.m.hp.n_embd, e.d_img_embd);
            } catch (const std::exception & ex) {
                res.status = 400;
                json j = {{"error", {{"message", ex.what()}, {"type", "invalid_request_error"}}}};
                res.set_content(dump_json(j), "application/json");
                return;
            }
            if (reject_too_long(e, mp.tokens.size(), res)) {
                return;
            }

            if (!stream) {
                mm_run_result out;
                run_mm_generate(e, mp, gp, stops, [](const std::string &) { return true; }, out);
                res.set_content(
                    make_chat_response(id, model, out.text, out.finish, (int)mp.tokens.size(), out.n_gen, created),
                    "application/json");
                return;
            }
            auto st = std::make_shared<stream_ctx>();
            st->q = std::make_shared<sse_queue>();
            bool include_usage = false;
            if (body.contains("stream_options") && body["stream_options"].is_object()) {
                include_usage = body["stream_options"].value("include_usage", false);
            }
            st->th = std::thread([&e, mp, gp, stops, id, model, created, st, include_usage, req_lk]() {
                auto * q = st->q.get();
                auto send_delta = [&](const std::string & piece) {
                    if (piece.empty()) {
                        return true;
                    }
                    json j = {
                        {"id", id},
                        {"object", "chat.completion.chunk"},
                        {"created", created},
                        {"model", model},
                        {"choices",
                         json::array({{{"index", 0}, {"delta", {{"content", piece}}}, {"finish_reason", nullptr}}})}};
                    q->push("data: " + dump_json(j) + "\n\n");
                    return true;
                };
                mm_run_result out;
                run_mm_generate(e, mp, gp, stops, send_delta, out);
                json finalj = {
                    {"id", id},
                    {"object", "chat.completion.chunk"},
                    {"created", created},
                    {"model", model},
                    {"choices",
                     json::array({{{"index", 0}, {"delta", json::object()}, {"finish_reason", out.finish}}})}};
                q->push("data: " + dump_json(finalj) + "\n\n");
                if (include_usage) {
                    json uj = {{"id", id},
                               {"object", "chat.completion.chunk"},
                               {"created", created},
                               {"model", model},
                               {"choices", json::array()},
                               {"usage",
                                {{"prompt_tokens", (int)mp.tokens.size()},
                                 {"completion_tokens", out.n_gen},
                                 {"total_tokens", (int)mp.tokens.size() + out.n_gen}}}};
                    q->push("data: " + dump_json(uj) + "\n\n");
                }
                q->push("data: [DONE]\n\n");
                q->finish();
            });
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
                    if (st->th.joinable()) {
                        st->th.join();
                    }
                });
            return;
        }

        std::string text = render_chat(e.m.chat_template, msgs, true, thinking);
        static const bool srv_t = getenv("PF_SRV_TIME") != nullptr;
        const auto t_tok0 = std::chrono::steady_clock::now();
        auto prompt = e.tk.encode(text);
        if (srv_t) {
            fprintf(stderr, "[http] tokenize=%.2f ms chars=%zu tokens=%zu\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_tok0).count(),
                    text.size(), prompt.size());
        }
        if (reject_too_long(e, prompt.size(), res)) {
            return;
        }

        if (!stream) {
            auto seq = sched.submit(std::move(prompt), gp, stops);
            std::string out_text, item;
            bool stopped = false;
            while (seq->pop(item)) {
                out_text += item;
                for (const auto & s : seq->stops) {
                    if (s.empty()) {
                        continue;
                    }
                    size_t p = out_text.rfind(s);
                    if (p != std::string::npos) {
                        out_text.resize(p);
                        stopped = true;
                        break;
                    }
                }
                if (stopped) {
                    break;
                }
            }
            const std::string finish = stopped ? "stop" : seq->finish_reason;
            res.set_content(
                make_chat_response(id, model, out_text, finish, seq->prompt_tokens, seq->n_generated, created),
                "application/json");
            return;
        }

        auto st = std::make_shared<stream_ctx>();
        bool include_usage = false;
        if (body.contains("stream_options") && body["stream_options"].is_object()) {
            include_usage = body["stream_options"].value("include_usage", false);
        }
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
                if (st->th.joinable()) {
                    st->th.join();
                }
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
        if (srv_t) {
            fprintf(stderr, "[http] tokenize=%.2f ms chars=%zu tokens=%zu\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_tok0).count(),
                    prompt_chars, prompt.size());
        }
        if (reject_too_long(e, prompt.size(), res)) {
            return;
        }
        gen_params gp = parse_params(body);
        auto stops = parse_stop(body);
        const bool stream = body.value("stream", false);
        const std::string id = gen_id();
        const uint64_t created = (uint64_t)time(nullptr);
        const std::string model = body.value("model", cfg.model_id);

        if (!stream) {
            auto seq = sched.submit(std::move(prompt), gp, stops);
            std::string out_text, item;
            bool stopped = false;
            while (seq->pop(item)) {
                out_text += item;
                for (const auto & s : seq->stops) {
                    if (s.empty()) {
                        continue;
                    }
                    size_t p = out_text.rfind(s);
                    if (p != std::string::npos) {
                        out_text.resize(p);
                        stopped = true;
                        break;
                    }
                }
                if (stopped) {
                    break;
                }
            }
            const std::string finish = stopped ? "stop" : seq->finish_reason;
            res.set_content(
                make_completion_response(id, model, out_text, finish, seq->prompt_tokens, seq->n_generated, created),
                "application/json");
            return;
        }

        auto st = std::make_shared<stream_ctx>();
        bool include_usage = false;
        if (body.contains("stream_options") && body["stream_options"].is_object()) {
            include_usage = body["stream_options"].value("include_usage", false);
        }
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
                if (st->th.joinable()) {
                    st->th.join();
                }
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
