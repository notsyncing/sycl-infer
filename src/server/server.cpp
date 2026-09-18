#include "server.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <ratio>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "chat.h"
#include "chat_util.h"
#include "engine.h"
#include "httplib.h"
#include "image.h"
#include "json.hpp"
#include "kernels.h"
#include "multimodal.h"
#include "response_parser.h"
#include "sampler.h"
#include "scheduler.h"
#include "vision.h"

using json = nlohmann::json;

namespace si {

namespace {

// SIGINT/SIGTERM only set this flag (async-signal-safe); a watchdog thread in
// serve() turns it into srv.stop() so the process unwinds normally (scheduler
// shutdown, then the engine destructor flushes the prefix cache).
std::atomic<bool> g_term_requested{false};
void on_term_signal(int) {
    g_term_requested.store(true);
}

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

// ------------------------------------------------------------- request parsing

constexpr int kMaxN = 16;      // per-request choices
constexpr int kMaxJobs = 64;   // prompt * n ceiling for one request
constexpr size_t kMaxStops = 16;

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
    // logit_bias: {"123": -100, ...} (token id -> additive bias, clamped)
    if (body.contains("logit_bias") && body["logit_bias"].is_object()) {
        for (auto it = body["logit_bias"].begin(); it != body["logit_bias"].end(); ++it) {
            int id = -1;
            try {
                id = std::stoi(it.key());
            } catch (...) {
                continue;
            }
            if (id < 0 || !it.value().is_number()) {
                continue;
            }
            const float b = std::max(-100.0f, std::min(100.0f, it.value().get<float>()));
            gp.logit_bias[id] = b;
        }
    }
    // logprobs: chat uses a bool (+ top_logprobs), completions an int count
    if (body.contains("logprobs")) {
        const json & lp = body["logprobs"];
        if (lp.is_boolean()) {
            gp.logprobs = lp.get<bool>();
        } else if (lp.is_number_integer()) {
            const int v = lp.get<int>();
            gp.logprobs = v > 0;
            gp.top_logprobs = std::min(20, std::max(0, v));
        }
    }
    if (body.contains("top_logprobs") && body["top_logprobs"].is_number_integer()) {
        gp.top_logprobs = std::min(20, std::max(0, body["top_logprobs"].get<int>()));
    }
    return gp;
}

int parse_n(const json & body) {
    int n = 1;
    if (body.contains("n") && body["n"].is_number()) {
        n = body["n"].get<int>();
    }
    if (n < 1) {
        n = 1;
    }
    if (n > kMaxN) {
        n = kMaxN;
    }
    return n;
}

// completions `best_of`: generate this many candidates and return the `n`
// highest-scoring ones (`best_of < n` is invalid, as in OpenAI).
int parse_best_of(const json & body, int n) {
    int best = n;
    if (body.contains("best_of") && body["best_of"].is_number_integer()) {
        best = body["best_of"].get<int>();
    }
    if (best < 1) {
        best = 1;
    }
    if (best > kMaxN) {
        best = kMaxN;
    }
    return best;
}

bool parse_include_usage(const json & body) {
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        return body["stream_options"].value("include_usage", false);
    }
    return false;
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
            if (v.is_string() && stops.size() < kMaxStops) {
                stops.push_back(v.get<std::string>());
            }
        }
    }
    return stops;
}

// `enable_thinking` for the chat template.  `chat_template_kwargs` wins (that is
// the llama.cpp convention), then the flat `enable_thinking`/`thinking` flags,
// then `reasoning_effort` (anything but "none" turns reasoning on).
bool parse_thinking(const json & body) {
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
        const json & k = body["chat_template_kwargs"];
        if (k.contains("enable_thinking") && k["enable_thinking"].is_boolean()) {
            return k["enable_thinking"].get<bool>();
        }
    }
    if (body.contains("enable_thinking") && body["enable_thinking"].is_boolean()) {
        return body["enable_thinking"].get<bool>();
    }
    if (body.contains("thinking") && body["thinking"].is_boolean()) {
        return body["thinking"].get<bool>();
    }
    if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string()) {
        return body["reasoning_effort"].get<std::string>() != "none";
    }
    return false;
}

// The OpenAI `tools` array serialized back to JSON, filtered by `tool_choice`.
// Returns "" when no tool is offered (so the template omits the tools block).
std::string parse_tools_json(const json & body) {
    if (!body.contains("tools") || !body["tools"].is_array() || body["tools"].empty()) {
        return "";
    }
    std::string choice = "auto";
    std::string fname;
    if (body.contains("tool_choice")) {
        const json & tc = body["tool_choice"];
        if (tc.is_string()) {
            choice = tc.get<std::string>();
        } else if (tc.is_object()) {
            choice = "function";
            if (tc.contains("function") && tc["function"].is_object()) {
                fname = tc["function"].value("name", std::string());
            } else if (tc.contains("name") && tc["name"].is_string()) {
                fname = tc["name"].get<std::string>();
            }
        }
    }
    if (choice == "none") {
        return "";
    }
    json out = json::array();
    for (const auto & t : body["tools"]) {
        if (choice == "function" && !fname.empty()) {
            if (!t.contains("function") || !t["function"].is_object()
                || t["function"].value("name", std::string()) != fname) {
                continue;
            }
        }
        out.push_back(t);
    }
    return out.empty() ? std::string() : dump_json(out);
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

// Split an absolute http(s) URL into the "scheme://host:port" base that
// httplib::Client expects and the request path.  Only http/https with an
// explicit host are accepted.
bool parse_http_url(const std::string & url, std::string & base, std::string & path, std::string & err) {
    const size_t scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
        err = "malformed image URL";
        return false;
    }
    const std::string scheme = url.substr(0, scheme_end);
    if (scheme != "http" && scheme != "https") {
        err = "unsupported image URL scheme: " + scheme;
        return false;
    }
    const size_t host_start = scheme_end + 3;
    const size_t path_start = url.find('/', host_start);
    std::string authority =
        (path_start == std::string::npos) ? url.substr(host_start) : url.substr(host_start, path_start - host_start);
    path = (path_start == std::string::npos) ? "/" : url.substr(path_start);
    const size_t at = authority.rfind('@'); // drop any userinfo
    if (at != std::string::npos) {
        authority = authority.substr(at + 1);
    }
    int port = (scheme == "https") ? 443 : 80;
    std::string host = authority;
    std::string port_s;
    if (!authority.empty() && authority[0] == '[') { // IPv6 literal
        const size_t br = authority.find(']');
        if (br == std::string::npos) {
            err = "malformed IPv6 host in image URL";
            return false;
        }
        host = authority.substr(0, br + 1);
        if (br + 1 < authority.size() && authority[br + 1] == ':') {
            port_s = authority.substr(br + 2);
        }
    } else {
        const size_t colon = authority.rfind(':');
        if (colon != std::string::npos) {
            host = authority.substr(0, colon);
            port_s = authority.substr(colon + 1);
        }
    }
    if (host.empty()) {
        err = "image URL has no host";
        return false;
    }
    if (!port_s.empty()) {
        if (port_s.find_first_not_of("0123456789") != std::string::npos) {
            err = "malformed port in image URL";
            return false;
        }
        port = std::atoi(port_s.c_str());
    }
    base = scheme + "://" + host + ":" + std::to_string(port);
    return true;
}

// Download a remote image.  The 10 s timeout and 10 MB cap keep a slow or
// hostile URL from stalling the handler thread or exhausting memory; the size
// is enforced by aborting the transfer once the limit is crossed.
bool fetch_http_image(const std::string & url, std::vector<uint8_t> & bytes, std::string & err) {
    std::string base, path;
    if (!parse_http_url(url, base, path, err)) {
        return false;
    }
    constexpr size_t kMaxBytes = 10 * 1024 * 1024;
    try {
        httplib::Client cli(base);
        cli.set_follow_location(true);
        cli.set_connection_timeout(10, 0);
        cli.set_read_timeout(10, 0);
        cli.set_write_timeout(10, 0);
        bool too_large = false;
        auto res = cli.Get(path, httplib::Headers{{"User-Agent", "sycl-infer"}},
                           [&](const char * data, size_t len) {
                               if (bytes.size() + len > kMaxBytes) {
                                   too_large = true;
                                   return false;
                               }
                               bytes.insert(bytes.end(), data, data + len);
                               return true;
                           });
        if (too_large) {
            err = "image URL exceeds the 10 MB limit";
            return false;
        }
        if (!res) {
            err = "cannot fetch image URL: " + httplib::to_string(res.error());
            return false;
        }
        if (res->status < 200 || res->status >= 300) {
            err = "image URL returned HTTP " + std::to_string(res->status);
            return false;
        }
    } catch (const std::exception & ex) {
        err = std::string("cannot fetch image URL: ") + ex.what();
        return false;
    }
    if (bytes.empty()) {
        err = "image URL returned no data";
        return false;
    }
    return true;
}

// Resolve an OpenAI `image_url` value to raw bytes.  `data:` URLs are decoded
// locally; http(s) URLs are downloaded (like llama.cpp's server), bounded so
// the endpoint cannot be used as an unbounded download proxy.  Other schemes
// are rejected.
bool load_image_url(const std::string & url, std::vector<uint8_t> & bytes, std::string & err) {
    bytes.clear();
    if (url.rfind("data:", 0) == 0) {
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
    if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
        // Remote fetches turn the server into a proxy, so allow deployments to
        // opt out (e.g. when exposed beyond localhost); on by default.
        static const bool remote_ok = [] {
            const char * v = getenv("PF_MM_URL_FETCH");
            return !(v != nullptr && v[0] == '0' && v[1] == '\0');
        }();
        if (!remote_ok) {
            err = "remote image URLs are disabled (PF_MM_URL_FETCH=0)";
            return false;
        }
        return fetch_http_image(url, bytes, err);
    }
    err = "unsupported image URL (expected data:, http:// or https://)";
    return false;
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
        if (m.contains("content") && !m["content"].is_null()) {
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
        if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
            cm.reasoning_content = m["reasoning_content"].get<std::string>();
        } else if (m.contains("reasoning") && m["reasoning"].is_string()) {
            cm.reasoning_content = m["reasoning"].get<std::string>();
        }
        if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
            for (auto & tc : m["tool_calls"]) {
                chat_tool_call call;
                call.id = tc.value("id", std::string());
                if (tc.contains("function") && tc["function"].is_object()) {
                    const json & fn = tc["function"];
                    call.name = fn.value("name", std::string());
                    if (fn.contains("arguments")) {
                        const json & a = fn["arguments"];
                        if (a.is_string()) {
                            call.arguments = a.get<std::string>();
                        } else if (!a.is_null()) {
                            call.arguments = dump_json(a);
                        }
                    }
                } else if (tc.contains("name")) {
                    call.name = tc.value("name", std::string());
                    if (tc.contains("arguments")) {
                        const json & a = tc["arguments"];
                        call.arguments = a.is_string() ? a.get<std::string>() : dump_json(a);
                    }
                }
                if (!call.name.empty()) {
                    cm.tool_calls.push_back(std::move(call));
                }
            }
        } else if (m.contains("function_call") && m["function_call"].is_object()) {
            const json & fn = m["function_call"];
            chat_tool_call call;
            call.name = fn.value("name", std::string());
            if (fn.contains("arguments")) {
                const json & a = fn["arguments"];
                call.arguments = a.is_string() ? a.get<std::string>() : dump_json(a);
            }
            if (!call.name.empty()) {
                cm.tool_calls.push_back(std::move(call));
            }
        }
        if (m.contains("tool_call_id") && m["tool_call_id"].is_string()) {
            cm.tool_call_id = m["tool_call_id"].get<std::string>();
        }
        if (m.contains("name") && m["name"].is_string()) {
            cm.name = m["name"].get<std::string>();
        }
        if (!has_image) {
            cm.parts.clear(); // text-only messages keep the plain path
        }
        msgs.push_back(std::move(cm));
    }
    return msgs;
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

// ----------------------------------------------------------------- generation

// Holds back up to (longest stop - 1) bytes so a stop string split across token
// pieces is still detected before its prefix is streamed to the client.
struct stop_filter {
    const std::vector<std::string> & stops;
    std::string buf;
    size_t keep = 0;
    bool stopped = false;

    explicit stop_filter(const std::vector<std::string> & s) : stops(s) {
        for (const std::string & x : stops) {
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
        for (const std::string & s : stops) {
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

struct lp_fragment {
    std::string text;
    float logprob = 0.f;
    std::vector<std::pair<int, float>> top;
};

json chat_logprobs_json(const std::vector<lp_fragment> & es, const tokenizer & tk);
json completion_logprobs_json(const std::vector<sequence::token_out> & toks, const tokenizer & tk);
json usage_json(long long prompt, long long completion, long long reasoning, long long cached);

struct choice_out {
    std::string content;
    std::string reasoning;
    std::vector<response_tool_call> tools;
    std::string finish = "stop";
    int n_gen = 0;
    int prompt_tokens = 0;
    int reasoning_tokens = 0; // generated tokens that made up reasoning_content
    int cached_tokens = 0;    // prompt tokens served from the prefix cache
    // logprobs / best_of scoring; empty unless the request asked for them
    std::vector<sequence::token_out> tokens;
    std::vector<lp_fragment> chat_lp;
    double score = 0.0;
};

gen_params gp_for_choice(const gen_params & gp, int c) {
    gen_params g = gp;
    if (gp.seed != 0) {
        g.seed = gp.seed + (uint64_t)c;
    }
    return g;
}

// Run one text sequence to completion, splitting the reply into reasoning /
// content / tool calls when `chat`.  For the plain completions endpoint the raw
// text lands in `out.content`.  When the request asks for logprobs (or best_of
// scoring) the per-token sampled details are collected too; otherwise the hot
// loop is exactly the pre-logprobs one.
void run_text_choice(scheduler & sched, const gen_params & gp, const std::vector<std::string> & stops,
                     std::vector<int> prompt, bool chat, bool thinking, bool parse_tools, choice_out & out) {
    auto seq = sched.submit(std::move(prompt), gp, stops);
    stop_filter sf(stops);
    const bool collect = gp.wants_logprobs();
    const bool want_lp = gp.logprobs;
    int cur_id = -1;
    float cur_lp = 0.f;
    std::vector<std::pair<int, float>> cur_top;
    bool cur_attached = true;
    response_parser parser(thinking, parse_tools, [&](const response_piece & p) {
        if (p.kind == response_piece_kind::reasoning) {
            out.reasoning_tokens++;
            return;
        }
        if (p.kind != response_piece_kind::content || !want_lp) {
            return;
        }
        lp_fragment f;
        f.text = p.text;
        if (!cur_attached && cur_id >= 0) {
            f.logprob = cur_lp;
            f.top = cur_top;
            cur_attached = true;
        }
        out.chat_lp.push_back(std::move(f));
    });
    sequence::token_out t;
    while (!sf.stopped && seq->pop_token(t)) {
        std::string emit = sf.feed(t.text);
        if (!emit.empty()) {
            if (chat) {
                if (want_lp) {
                    cur_id = t.id;
                    cur_lp = t.logprob;
                    cur_top = t.top;
                    cur_attached = false;
                }
                parser.feed(emit);
            } else {
                out.content += emit;
            }
        }
        if (collect) {
            if (t.id >= 0) {
                out.score += t.logprob;
            }
            out.tokens.push_back(std::move(t));
        }
    }
    if (!sf.stopped) {
        std::string tail = sf.flush();
        if (!tail.empty()) {
            if (chat) {
                parser.feed(tail);
            } else {
                out.content += tail;
            }
        }
    }
    if (chat) {
        parser.finish();
        out.reasoning = parser.reasoning();
        out.content = parser.content();
        out.tools = parser.tool_calls();
    }
    out.n_gen = seq->n_generated;
    out.prompt_tokens = seq->prompt_tokens;
    out.cached_tokens = seq->reused;
    if (chat && !out.tools.empty()) {
        out.finish = "tool_calls";
    } else {
        out.finish = sf.stopped ? "stop" : seq->finish_reason;
    }
}

void mm_piece(const std::string & piece, stop_filter & sf, response_parser * parser, choice_out & out, bool chat) {
    const std::string emit = sf.feed(piece);
    if (emit.empty()) {
        return;
    }
    if (chat) {
        const size_t before = parser->reasoning().size();
        parser->feed(emit);
        if (parser->reasoning().size() > before) {
            out.reasoning_tokens++;
        }
    } else {
        out.content += emit;
    }
}

// Run one multimodal prompt (single-sequence engine path, no scheduler).
void run_mm_choice(engine & e, const mm_prompt & mp, const gen_params & gp, const std::vector<std::string> & stops,
                   bool chat, bool thinking, bool parse_tools, choice_out & out) {
    stop_filter sf(stops);
    response_parser parser(thinking, parse_tools, [](const response_piece &) {});
    utf8_stream_buffer ub;
    auto cb = [&](int tok) -> bool {
        const std::string piece = ub.push(e.tk.token_piece(tok));
        if (piece.empty()) {
            return true;
        }
        out.n_gen++;
        mm_piece(piece, sf, &parser, out, chat);
        return !sf.stopped;
    };
    e.generate_mm(mp, gp, cb);
    const std::string tail = ub.flush();
    if (!tail.empty() && !sf.stopped) {
        mm_piece(tail, sf, &parser, out, chat);
    }
    if (!sf.stopped) {
        const std::string rest = sf.flush();
        if (!rest.empty()) {
            mm_piece(rest, sf, &parser, out, chat);
        }
    }
    if (chat) {
        parser.finish();
        out.reasoning = parser.reasoning();
        out.content = parser.content();
        out.tools = parser.tool_calls();
    }
    out.prompt_tokens = (int)mp.tokens.size();
    if (chat && !out.tools.empty()) {
        out.finish = "tool_calls";
    } else {
        out.finish = sf.stopped ? "stop" : "length";
    }
}

// --------------------------------------------------------------- SSE plumbing

json chat_chunk(const std::string & id, const std::string & model, uint64_t created, int index, const json & delta,
                const json & finish_reason, const json & logprobs = json()) {
    json ch = json::array();
    json choice = {{"index", index}, {"delta", delta}, {"finish_reason", finish_reason}};
    if (!logprobs.is_null()) {
        choice["logprobs"] = logprobs;
    }
    ch.push_back(std::move(choice));
    return {{"id", id},           {"object", "chat.completion.chunk"},
            {"created", created}, {"model", model},
            {"choices", ch}};
}

json text_chunk(const std::string & id, const std::string & model, uint64_t created, int index, const std::string & text,
                const json & finish_reason, const json & logprobs = json()) {
    json ch = json::array();
    json choice = {{"index", index}, {"text", text}, {"finish_reason", finish_reason}};
    if (!logprobs.is_null()) {
        choice["logprobs"] = logprobs;
    }
    ch.push_back(std::move(choice));
    return {{"id", id}, {"object", "text_completion"}, {"created", created}, {"model", model}, {"choices", ch}};
}

// One SSE stream shared by all choices of a request; the last choice to finish
// emits the usage chunk (optional) and the terminating [DONE].
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

    void choice_done(long long pt, long long ct, long long rt = 0, long long cached = 0) {
        std::lock_guard<std::mutex> lk(m);
        prompt_tokens += pt;
        completion_tokens += ct;
        reasoning_tokens += rt;
        cached_tokens += cached;
        if (--remaining > 0) {
            return;
        }
        if (include_usage) {
            json uj = {{"id", id},
                       {"object", chat ? "chat.completion.chunk" : "text_completion"},
                       {"created", created},
                       {"model", model},
                       {"choices", json::array()},
                       {"usage", usage_json(prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens)}};
            q->push("data: " + dump_json(uj) + "\n\n");
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

void sse_error(const std::shared_ptr<sse_session> & sc, int index, const char * msg) {
    json d = {{"error", {{"message", msg}, {"type", "server_error"}}}};
    if (sc->chat) {
        sc->q->push("data: " + dump_json(chat_chunk(sc->id, sc->model, sc->created, index, d, "stop")) + "\n\n");
    }
}

void stream_chat_text_choice(std::shared_ptr<sse_session> sc, scheduler & sched, int index, std::vector<int> prompt,
                             const gen_params & gp, const std::vector<std::string> & stops, bool thinking,
                             bool parse_tools) {
    try {
        auto seq = sched.submit(std::move(prompt), gp, stops);
        sc->q->push(
            "data: " + dump_json(chat_chunk(sc->id, sc->model, sc->created, index, {{"role", "assistant"}}, nullptr))
            + "\n\n");
        int tool_index = 0;
        long long reasoning_tokens = 0;
        const bool want_lp = gp.logprobs;
        int cur_id = -1;
        float cur_lp = 0.f;
        std::vector<std::pair<int, float>> cur_top;
        bool cur_attached = true;
        response_parser parser(thinking, parse_tools, [&](const response_piece & p) {
            json delta;
            json lpj;
            if (p.kind == response_piece_kind::reasoning) {
                reasoning_tokens++;
                delta = {{"reasoning_content", p.text}};
            } else if (p.kind == response_piece_kind::content) {
                delta = {{"content", p.text}};
                if (want_lp) {
                    lp_fragment f;
                    f.text = p.text;
                    if (!cur_attached && cur_id >= 0) {
                        f.logprob = cur_lp;
                        f.top = cur_top;
                        cur_attached = true;
                    }
                    lpj = chat_logprobs_json(std::vector<lp_fragment>{std::move(f)}, sched.e.tk);
                }
            } else {
                json call = {{"index", tool_index},
                             {"id", p.call.id},
                             {"type", "function"},
                             {"function", {{"name", p.call.name}, {"arguments", p.call.arguments}}}};
                delta = {{"tool_calls", json::array({call})}};
                tool_index++;
            }
            sc->q->push("data: "
                        + dump_json(chat_chunk(sc->id, sc->model, sc->created, index, delta, nullptr, lpj)) + "\n\n");
        });
        stop_filter sf(stops);
        sequence::token_out t;
        while (!sf.stopped && seq->pop_token(t)) {
            const std::string emit = sf.feed(t.text);
            if (!emit.empty()) {
                if (want_lp) {
                    cur_id = t.id;
                    cur_lp = t.logprob;
                    cur_top = t.top;
                    cur_attached = false;
                }
                parser.feed(emit);
            }
        }
        if (!sf.stopped) {
            const std::string tail = sf.flush();
            if (!tail.empty()) {
                parser.feed(tail);
            }
        }
        parser.finish();
        const std::string finish = sf.stopped ? "stop"
                                              : (parser.tool_calls().empty() ? seq->finish_reason : "tool_calls");
        sc->q->push("data: " + dump_json(chat_chunk(sc->id, sc->model, sc->created, index, json::object(), finish))
                    + "\n\n");
        sc->choice_done(seq->prompt_tokens, seq->n_generated, reasoning_tokens, seq->reused);
    } catch (const std::exception &) {
        sse_error(sc, index, "generation failed");
        sc->choice_done(0, 0);
    }
}

void stream_chat_mm_choices(std::shared_ptr<sse_session> sc, engine & e, const mm_prompt & mp, const gen_params & gp,
                            const std::vector<std::string> & stops, int n, bool thinking, bool parse_tools) {
    for (int c = 0; c < n; c++) {
        try {
            sc->q->push("data: "
                        + dump_json(chat_chunk(sc->id, sc->model, sc->created, c, {{"role", "assistant"}}, nullptr))
                        + "\n\n");
            int tool_index = 0;
            long long reasoning_tokens = 0;
            response_parser parser(thinking, parse_tools, [&](const response_piece & p) {
                json delta;
                if (p.kind == response_piece_kind::reasoning) {
                    reasoning_tokens++;
                    delta = {{"reasoning_content", p.text}};
                } else if (p.kind == response_piece_kind::content) {
                    delta = {{"content", p.text}};
                } else {
                    json call = {{"index", tool_index},
                                 {"id", p.call.id},
                                 {"type", "function"},
                                 {"function", {{"name", p.call.name}, {"arguments", p.call.arguments}}}};
                    delta = {{"tool_calls", json::array({call})}};
                    tool_index++;
                }
                sc->q->push("data: " + dump_json(chat_chunk(sc->id, sc->model, sc->created, c, delta, nullptr))
                            + "\n\n");
            });
            stop_filter sf(stops);
            utf8_stream_buffer ub;
            int n_gen = 0;
            auto cb = [&](int tok) -> bool {
                const std::string piece = ub.push(e.tk.token_piece(tok));
                if (piece.empty()) {
                    return true;
                }
                n_gen++;
                const std::string emit = sf.feed(piece);
                if (!emit.empty()) {
                    parser.feed(emit);
                }
                return !sf.stopped;
            };
            e.generate_mm(mp, gp_for_choice(gp, c), cb);
            const std::string tail = ub.flush();
            if (!tail.empty() && !sf.stopped) {
                const std::string emit = sf.feed(tail);
                if (!emit.empty()) {
                    parser.feed(emit);
                }
            }
            if (!sf.stopped) {
                const std::string rest = sf.flush();
                if (!rest.empty()) {
                    parser.feed(rest);
                }
            }
            parser.finish();
            const std::string finish =
                sf.stopped ? "stop" : (parser.tool_calls().empty() ? std::string("length") : std::string("tool_calls"));
            sc->q->push(
                "data: " + dump_json(chat_chunk(sc->id, sc->model, sc->created, c, json::object(), finish)) + "\n\n");
            sc->choice_done((int)mp.tokens.size(), n_gen, reasoning_tokens, 0);
        } catch (const std::exception &) {
            sse_error(sc, c, "multimodal generation failed");
            sc->choice_done(0, 0);
        }
    }
}

void stream_completion_text_choice(std::shared_ptr<sse_session> sc, scheduler & sched, int index,
                                   std::vector<int> prompt, const gen_params & gp,
                                   const std::vector<std::string> & stops, bool echo, const std::string & prompt_text,
                                   const std::string & suffix) {
    try {
        auto send = [&](const std::string & t) {
            if (t.empty()) {
                return;
            }
            sc->q->push("data: " + dump_json(text_chunk(sc->id, sc->model, sc->created, index, t, nullptr)) + "\n\n");
        };
        if (echo) {
            send(prompt_text);
        }
        auto seq = sched.submit(std::move(prompt), gp, stops);
        stop_filter sf(stops);
        sequence::token_out t;
        while (!sf.stopped && seq->pop_token(t)) {
            const std::string emit = sf.feed(t.text);
            json lpj;
            if (gp.logprobs && t.id >= 0) {
                lpj = completion_logprobs_json(std::vector<sequence::token_out>{t}, sched.e.tk);
            }
            if (!emit.empty() || !lpj.is_null()) {
                sc->q->push("data: "
                            + dump_json(text_chunk(sc->id, sc->model, sc->created, index, emit, nullptr, lpj))
                            + "\n\n");
            }
        }
        if (!sf.stopped) {
            send(sf.flush());
        }
        send(suffix);
        const std::string finish = sf.stopped ? "stop" : seq->finish_reason;
        sc->q->push("data: " + dump_json(text_chunk(sc->id, sc->model, sc->created, index, "", finish)) + "\n\n");
        sc->choice_done(seq->prompt_tokens, seq->n_generated, 0, seq->reused);
    } catch (const std::exception &) {
        sc->choice_done(0, 0);
    }
}

// ------------------------------------------------------------------- response

json tools_to_json(const std::vector<response_tool_call> & tools) {
    json arr = json::array();
    for (const auto & t : tools) {
        arr.push_back({{"id", t.id},
                       {"type", "function"},
                       {"function", {{"name", t.name}, {"arguments", t.arguments}}}});
    }
    return arr;
}

json bytes_json(const std::string & s) {
    json a = json::array();
    for (unsigned char c : s) {
        a.push_back((int)c);
    }
    return a;
}

// chat `logprobs`: one entry per content fragment (OpenAI `logprobs.content`).
json chat_logprobs_json(const std::vector<lp_fragment> & es, const tokenizer & tk) {
    json content = json::array();
    for (const lp_fragment & e : es) {
        json tops = json::array();
        for (const auto & p : e.top) {
            const std::string ts = tk.token_piece(p.first);
            tops.push_back({{"token", ts}, {"logprob", p.second}, {"bytes", bytes_json(ts)}});
        }
        content.push_back(
            {{"token", e.text}, {"logprob", e.logprob}, {"bytes", bytes_json(e.text)}, {"top_logprobs", tops}});
    }
    return {{"content", content}, {"refusal", nullptr}};
}

// completions `logprobs`: legacy parallel arrays over the generated tokens.
json completion_logprobs_json(const std::vector<sequence::token_out> & toks, const tokenizer & tk) {
    json tokens = json::array();
    json token_logprobs = json::array();
    json top_logprobs = json::array();
    json text_offset = json::array();
    size_t off = 0;
    for (const sequence::token_out & t : toks) {
        if (t.id < 0) {
            continue;
        }
        const std::string ts = tk.token_piece(t.id);
        tokens.push_back(ts);
        token_logprobs.push_back(t.logprob);
        json m = json::object();
        for (const auto & p : t.top) {
            m[tk.token_piece(p.first)] = p.second;
        }
        top_logprobs.push_back(std::move(m));
        text_offset.push_back((int)off);
        off += ts.size();
    }
    return {{"tokens", tokens},
            {"token_logprobs", token_logprobs},
            {"top_logprobs", top_logprobs},
            {"text_offset", text_offset}};
}

json chat_message_json(const choice_out & out, bool thinking) {
    json msg = {{"role", "assistant"}};
    if (!out.tools.empty()) {
        msg["content"] = out.content.empty() ? json(nullptr) : json(out.content);
        msg["tool_calls"] = tools_to_json(out.tools);
    } else {
        msg["content"] = out.content;
    }
    if (thinking && !out.reasoning.empty()) {
        msg["reasoning_content"] = out.reasoning;
    }
    return msg;
}

json usage_json(long long prompt, long long completion, long long reasoning, long long cached) {
    return {
        {"prompt_tokens", (int)prompt},
        {"completion_tokens", (int)completion},
        {"total_tokens", (int)(prompt + completion)},
        // OpenAI prompt caching detail ...
        {"prompt_tokens_details", {{"cached_tokens", (int)cached}, {"audio_tokens", 0}}},
        // ... and the DeepSeek-style top-level pair (same numbers)
        {"prompt_cache_hit_tokens", (int)cached},
        {"prompt_cache_miss_tokens", (int)(prompt - cached)},
        {"completion_tokens_details",
         {{"reasoning_tokens", (int)reasoning},
          {"audio_tokens", 0},
          {"accepted_prediction_tokens", 0},
          {"rejected_prediction_tokens", 0}}},
    };
}

std::string gen_id() {
    static std::atomic<uint64_t> counter{0};
    char buf[64];
    snprintf(buf, sizeof(buf), "chatcmpl-%llu", (unsigned long long)counter.fetch_add(1) + 1);
    return buf;
}

struct completion_prompts {
    std::vector<std::vector<int>> tokens;
    std::vector<std::string> texts;
};

// OpenAI `prompt` may be a string, an array of strings, an array of token ids,
// or an array of token-id arrays.  Each entry becomes one completion prompt.
completion_prompts parse_completion_prompts(engine & e, const json & body) {
    completion_prompts cp;
    auto add_str = [&](const std::string & s) {
        cp.tokens.push_back(e.tk.encode(s));
        cp.texts.push_back(s);
    };
    auto add_ids = [&](const json & a) {
        std::vector<int> ids;
        for (const auto & v : a) {
            if (v.is_number_integer()) {
                ids.push_back(v.get<int>());
            }
        }
        cp.tokens.push_back(ids);
        cp.texts.push_back(e.tk.decode(ids));
    };
    if (!body.contains("prompt") || body["prompt"].is_null()) {
        add_str("");
        return cp;
    }
    const json & p = body["prompt"];
    if (p.is_string()) {
        add_str(p.get<std::string>());
    } else if (p.is_array()) {
        if (p.empty()) {
            add_str("");
        } else if (p[0].is_string()) {
            for (const auto & v : p) {
                if (v.is_string()) {
                    add_str(v.get<std::string>());
                }
            }
        } else if (p[0].is_number_integer()) {
            add_ids(p);
        } else if (p[0].is_array()) {
            for (const auto & v : p) {
                if (v.is_array()) {
                    add_ids(v);
                }
            }
        } else {
            add_str("");
        }
    } else {
        add_str("");
    }
    if (cp.tokens.empty()) {
        add_str("");
    }
    return cp;
}

} // namespace

// Run a multimodal prompt on the single-sequence path, forwarding decoded text
// pieces to `on_piece` (return false to stop early).
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

    // model id / metadata advertised by /v1/models
    std::string model_id = cfg.model_id;
    if (const std::string * name = e.m.gguf.get_str("general.name"); name != nullptr && !name->empty()) {
        model_id = *name;
    }
    const uint64_t model_created = (uint64_t)time(nullptr);
    json model_entry = {{"id", model_id},
                        {"object", "model"},
                        {"created", model_created},
                        {"owned_by", "local"},
                        {"meta", {{"n_ctx", e.max_seq}, {"n_vocab", e.m.hp.n_vocab}, {"n_layer", e.m.hp.n_layer}}}};

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

    auto set_json = [](httplib::Response & res, int status, const json & j) {
        res.status = status;
        res.set_content(dump_json(j), "application/json");
    };
    auto bad_request = [&](httplib::Response & res, const std::string & msg, const std::string & type) {
        set_json(res, 400, {{"error", {{"message", msg}, {"type", type}}}});
    };

    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        cors(res);
        set_json(res, 200, {{"object", "list"}, {"data", json::array({model_entry})}});
    });
    srv.Get(R"(/v1/models/(.+))", [&](const httplib::Request & req, httplib::Response & res) {
        cors(res);
        const std::string id = req.matches[1];
        if (id != model_id && id != cfg.model_id) {
            set_json(res, 404,
                     {{"error", {{"message", "model '" + id + "' not found"}, {"type", "invalid_request_error"}}}});
            return;
        }
        set_json(res, 200, model_entry);
    });

    auto cors_sse = [](httplib::Response & res) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
    };
    auto serve_sse = [](httplib::Response & res, const std::shared_ptr<sse_session> & sc) {
        res.set_chunked_content_provider(
            "text/event-stream",
            [sc](size_t, httplib::DataSink & sink) {
                std::string item;
                if (!sc->q->pop(item)) {
                    sink.done();
                    return true;
                }
                sink.write(item.data(), item.size());
                return true;
            },
            [sc](bool) { sc->join(); });
    };

    auto handle_chat = [&](const httplib::Request & req, httplib::Response & res) {
        cors(res);
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            bad_request(res, "invalid json", "invalid_request_error");
            return;
        }
        std::vector<std::string> image_urls;
        auto msgs = parse_messages(body, image_urls);
        const bool thinking = parse_thinking(body);
        const std::string tools_json = parse_tools_json(body);
        const bool parse_tools = !tools_json.empty();
        gen_params gp = parse_params(body);
        auto stops = parse_stop(body);
        const int n = parse_n(body);
        const bool stream = body.value("stream", false);
        const bool include_usage = parse_include_usage(body);
        const std::string id = gen_id();
        const uint64_t created = (uint64_t)time(nullptr);
        const std::string model = body.value("model", model_id);

        if (!image_urls.empty()) {
            if (!mm.ready) {
                bad_request(res, "image input requires --mmproj", "invalid_request_error");
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
                    if (!load_image_url(u, bytes, err)
                        || !mm_image_decode_mem(bytes.data(), bytes.size(), rgb, w, h, &err)) {
                        throw std::runtime_error(err);
                    }
                    imgs.push_back(mm_image_preprocess(rgb.data(), w, h, mm.cfg));
                }
                const std::string rendered = render_chat(e.m.chat_template, msgs, true, thinking, tools_json);
                mp = mm_build_prompt_device(mm.vm, e.q, e.tk, rendered, imgs, e.m.hp.n_embd, e.d_img_embd);
            } catch (const std::exception & ex) {
                bad_request(res, ex.what(), "invalid_request_error");
                return;
            }
            if (reject_too_long(e, mp.tokens.size(), res)) {
                return;
            }

            if (!stream) {
                json choices = json::array();
                long long pu = 0, cu = 0, ru = 0, ca = 0;
                for (int c = 0; c < n; c++) {
                    choice_out out;
                    run_mm_choice(e, mp, gp_for_choice(gp, c), stops, true, thinking, parse_tools, out);
                    choices.push_back({{"index", c},
                                       {"message", chat_message_json(out, thinking)},
                                       {"finish_reason", out.finish},
                                       {"logprobs", nullptr}});
                    pu += out.prompt_tokens;
                    cu += out.n_gen;
                    ru += out.reasoning_tokens;
                    ca += out.cached_tokens;
                }
                set_json(res, 200,
                         {{"id", id},
                          {"object", "chat.completion"},
                          {"created", created},
                          {"model", model},
                          {"choices", choices},
                          {"usage", usage_json(pu, cu, ru, ca)}});
                return;
            }

            auto sc = std::make_shared<sse_session>();
            sc->q = std::make_shared<sse_queue>();
            sc->remaining = n;
            sc->include_usage = include_usage;
            sc->chat = true;
            sc->id = id;
            sc->model = model;
            sc->created = created;
            sc->ths.emplace_back([&e, mp, gp, stops, n, thinking, parse_tools, sc, req_lk]() {
                stream_chat_mm_choices(sc, e, mp, gp, stops, n, thinking, parse_tools);
            });
            cors_sse(res);
            serve_sse(res, sc);
            return;
        }

        std::string text = render_chat(e.m.chat_template, msgs, true, thinking, tools_json);
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
            json choices = json::array();
            long long pu = 0, cu = 0, ru = 0, ca = 0;
            for (int c = 0; c < n; c++) {
                choice_out out;
                run_text_choice(sched, gp_for_choice(gp, c), stops, prompt, true, thinking, parse_tools, out);
                json logprobs = gp.logprobs ? chat_logprobs_json(out.chat_lp, e.tk) : json();
                choices.push_back({{"index", c},
                                   {"message", chat_message_json(out, thinking)},
                                   {"finish_reason", out.finish},
                                   {"logprobs", logprobs}});
                pu += out.prompt_tokens;
                cu += out.n_gen;
                ru += out.reasoning_tokens;
                ca += out.cached_tokens;
            }
            set_json(res, 200,
                     {{"id", id},
                      {"object", "chat.completion"},
                      {"created", created},
                      {"model", model},
                      {"choices", choices},
                      {"usage", usage_json(pu, cu, ru, ca)}});
            return;
        }

        auto sc = std::make_shared<sse_session>();
        sc->q = std::make_shared<sse_queue>();
        sc->remaining = n;
        sc->include_usage = include_usage;
        sc->chat = true;
        sc->id = id;
        sc->model = model;
        sc->created = created;
        for (int c = 0; c < n; c++) {
            sc->ths.emplace_back([&sched, sc, c, prompt, gp, stops, thinking, parse_tools]() {
                stream_chat_text_choice(sc, sched, c, prompt, gp_for_choice(gp, c), stops, thinking, parse_tools);
            });
        }
        cors_sse(res);
        serve_sse(res, sc);
    };
    srv.Post("/v1/chat/completions", handle_chat);

    auto handle_completion = [&](const httplib::Request & req, httplib::Response & res) {
        cors(res);
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            bad_request(res, "invalid json", "invalid_request_error");
            return;
        }
        static const bool srv_t = getenv("PF_SRV_TIME") != nullptr;
        const auto t_tok0 = std::chrono::steady_clock::now();
        completion_prompts cp = parse_completion_prompts(e, body);
        if (srv_t) {
            size_t chars = 0;
            for (const std::string & t : cp.texts) {
                chars += t.size();
            }
            fprintf(stderr, "[http] tokenize=%.2f ms chars=%zu prompts=%zu\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_tok0).count(),
                    chars, cp.tokens.size());
        }
        gen_params gp = parse_params(body);
        auto stops = parse_stop(body);
        const int n = parse_n(body);
        const bool stream = body.value("stream", false);
        const bool include_usage = parse_include_usage(body);
        const bool echo = body.value("echo", false);
        const std::string suffix = body.value("suffix", std::string());
        const std::string id = gen_id();
        const uint64_t created = (uint64_t)time(nullptr);
        const std::string model = body.value("model", model_id);

        for (const std::vector<int> & p : cp.tokens) {
            if (reject_too_long(e, p.size(), res)) {
                return;
            }
        }
        const int best_of = parse_best_of(body, n);
        if (best_of < n) {
            bad_request(res, "best_of must be >= n", "invalid_request_error");
            return;
        }
        if (stream && best_of > 1) {
            bad_request(res, "best_of is not supported with stream", "invalid_request_error");
            return;
        }
        const long long jobs = (long long)cp.tokens.size() * (long long)std::max(n, best_of);
        if (jobs > kMaxJobs) {
            bad_request(res, "prompt count * n (or best_of) exceeds the server limit", "invalid_request_error");
            return;
        }

        if (!stream) {
            json choices = json::array();
            long long pu = 0, cu = 0, ca = 0;
            int index = 0;
            const bool score = best_of > n;
            for (size_t pi = 0; pi < cp.tokens.size(); pi++) {
                std::vector<choice_out> outs;
                const int gen = score ? best_of : n;
                for (int j = 0; j < gen; j++) {
                    gen_params g = gp_for_choice(gp, j);
                    if (score) {
                        g.need_score = true;
                    }
                    choice_out out;
                    run_text_choice(sched, g, stops, cp.tokens[pi], false, false, false, out);
                    outs.push_back(std::move(out));
                }
                if (score) {
                    std::stable_sort(outs.begin(), outs.end(),
                                     [](const choice_out & a, const choice_out & b) { return a.score > b.score; });
                    outs.resize(n);
                }
                for (int c = 0; c < n && c < (int)outs.size(); c++) {
                    choice_out & out = outs[c];
                    std::string text = (echo ? cp.texts[pi] : std::string()) + out.content;
                    text += suffix;
                    json lp = gp.logprobs ? completion_logprobs_json(out.tokens, e.tk) : json();
                    choices.push_back(
                        {{"index", index++}, {"text", text}, {"finish_reason", out.finish}, {"logprobs", lp}});
                    pu += out.prompt_tokens;
                    cu += out.n_gen;
                    ca += out.cached_tokens;
                }
            }
            set_json(res, 200,
                     {{"id", id},
                      {"object", "text_completion"},
                      {"created", created},
                      {"model", model},
                      {"choices", choices},
                      {"usage", usage_json(pu, cu, 0, ca)}});
            return;
        }

        auto sc = std::make_shared<sse_session>();
        sc->q = std::make_shared<sse_queue>();
        sc->remaining = (int)jobs;
        sc->include_usage = include_usage;
        sc->chat = false;
        sc->id = id;
        sc->model = model;
        sc->created = created;
        int index = 0;
        for (size_t pi = 0; pi < cp.tokens.size(); pi++) {
            for (int c = 0; c < n; c++) {
                sc->ths.emplace_back([&sched, sc, index, c, prompt = cp.tokens[pi], gp, stops, echo,
                                      ptext = cp.texts[pi], suffix]() {
                    stream_completion_text_choice(sc, sched, index, prompt, gp_for_choice(gp, c), stops, echo, ptext,
                                                  suffix);
                });
                index++;
            }
        }
        cors_sse(res);
        serve_sse(res, sc);
    };
    srv.Post("/v1/completions", handle_completion);

    printf("server listening on %s:%d\n", cfg.host.c_str(), cfg.port);
    fflush(stdout);
    std::signal(SIGINT, on_term_signal);
    std::signal(SIGTERM, on_term_signal);
    std::atomic<bool> listen_done{false};
    std::thread watchdog([&] {
        while (!g_term_requested.load() && !listen_done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (g_term_requested.load()) {
            srv.stop();
        }
    });
    const bool listening = srv.listen(cfg.host.c_str(), cfg.port);
    listen_done.store(true);
    watchdog.join();
    g_term_requested.store(false);
    if (!listening) {
        fprintf(stderr, "failed to listen on %s:%d\n", cfg.host.c_str(), cfg.port);
        return 1;
    }
    fprintf(stderr, "[srv] shutting down, flushing prefix cache\n");
    return 0;
}

} // namespace si
