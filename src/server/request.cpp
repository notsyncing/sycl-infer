#include "request.h"

// Included directly rather than through request.h: this file uses these symbols,
// and Diagnostics.UnusedIncludes: Strict wants the providing header named.
#include "chat.h"
#include "json_util.h"
#include "media_fetch.h"
#include "sampler.h"
#include "tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace si {

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

std::vector<chat_msg> parse_messages(const json & body, std::vector<media_part> & media) {
    std::vector<chat_msg> msgs;
    if (!body.contains("messages") || !body["messages"].is_array()) {
        return msgs;
    }
    for (auto & m : body["messages"]) {
        chat_msg cm;
        cm.role = m.value("role", "user");
        bool has_media = false;
        auto add_media = [&](chat_part_kind kind, const std::string & url, const std::string & data,
                             const std::string & format) {
            media_part mp;
            mp.kind = kind;
            mp.url = url;
            mp.data = data;
            mp.format = format;
            media.push_back(std::move(mp));
            chat_part cp;
            cp.kind = kind;
            cm.parts.push_back(std::move(cp));
        };
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
                        add_media(chat_part_kind::IMAGE, url, "", "");
                        has_media = true;
                    } else if (type == "video_url" || type == "video") {
                        std::string url;
                        if (part.contains("video_url")) {
                            const json & vu = part["video_url"];
                            url = vu.is_string() ? vu.get<std::string>() : vu.value("url", std::string());
                        } else {
                            url = part.value("video", std::string());
                        }
                        add_media(chat_part_kind::VIDEO, url, "", "");
                        has_media = true;
                    } else if (type == "input_audio") {
                        std::string data, format;
                        const json & a = part.value("input_audio", json::object());
                        if (a.is_object()) {
                            data = a.value("data", std::string());
                            format = a.value("format", std::string());
                        }
                        add_media(chat_part_kind::AUDIO, "", data, format);
                        has_media = true;
                    } else if (type == "audio_url") {
                        std::string url, format;
                        if (part.contains("audio_url")) {
                            const json & au = part["audio_url"];
                            url = au.is_string() ? au.get<std::string>() : au.value("url", std::string());
                            format = au.value("format", std::string());
                        }
                        add_media(chat_part_kind::AUDIO, url, "", format);
                        has_media = true;
                    } else if (part.contains("text") && part["text"].is_string()) {
                        const std::string t = part["text"].get<std::string>();
                        cm.content += t;
                        chat_part cp;
                        cp.kind = chat_part_kind::TEXT;
                        cp.text = t;
                        cm.parts.push_back(std::move(cp));
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
        if (!has_media) {
            cm.parts.clear(); // text-only messages keep the plain path
        }
        msgs.push_back(std::move(cm));
    }
    return msgs;
}

completion_prompts parse_completion_prompts(tokenizer & tk, const json & body) {
    completion_prompts cp;
    auto add_str = [&](const std::string & s) {
        cp.tokens.push_back(tk.encode(s));
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
        cp.texts.push_back(tk.decode(ids));
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

prompt_verdict check_prompt(int max_seq, size_t prompt_tokens) {
    prompt_verdict v;
    if (prompt_tokens == 0) {
        v.ok = false;
        json j = {{"error", {{"message", "prompt must contain at least one token"},
                              {"type", "invalid_request_error"}}}};
        v.body = dump_json(j);
        return v;
    }
    if ((int)prompt_tokens <= max_seq) {
        return v;
    }
    v.ok = false;
    v.too_long = true;
    json j = {
        {"error",
         {{"message", "prompt length " + std::to_string(prompt_tokens) + " tokens exceeds the server context (max_seq="
                           + std::to_string(max_seq) + "); restart with --ctx " + std::to_string(prompt_tokens)
                           + " (or PF_CTX) or shorten the prompt"},
          {"type", "invalid_request_error"},
          {"code", "context_length_exceeded"}}},
        {"max_seq", max_seq},
        {"prompt_tokens", (int)prompt_tokens}};
    v.body = dump_json(j);
    return v;
}

} // namespace si
