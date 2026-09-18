#include "response_parser.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

#include "json.hpp"

namespace si {

namespace {

using ojson = nlohmann::ordered_json;

const char kThinkOpen[] = "<think>";
const char kThinkClose[] = "</think>";
const char kToolOpen[] = "<tool_call>";
const char kToolClose[] = "</tool_call>";
constexpr size_t kThinkOpenLen = sizeof(kThinkOpen) - 1;
constexpr size_t kThinkCloseLen = sizeof(kThinkClose) - 1;
constexpr size_t kToolOpenLen = sizeof(kToolOpen) - 1;
constexpr size_t kToolCloseLen = sizeof(kToolClose) - 1;
constexpr size_t kParamCloseLen = sizeof("</parameter>") - 1;

bool is_space(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

std::string trim_spaces(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && is_space(s[b])) {
        b++;
    }
    while (e > b && is_space(s[e - 1])) {
        e--;
    }
    return s.substr(b, e - b);
}

std::string lstrip_newlines(const std::string & s) {
    size_t b = 0;
    while (b < s.size() && (s[b] == '\n' || s[b] == '\r')) {
        b++;
    }
    return s.substr(b);
}

std::string lstrip_spaces(const std::string & s) {
    size_t b = 0;
    while (b < s.size() && is_space(s[b])) {
        b++;
    }
    return s.substr(b);
}

std::string rstrip_spaces(const std::string & s) {
    size_t e = s.size();
    while (e > 0 && is_space(s[e - 1])) {
        e--;
    }
    return s.substr(0, e);
}

// Length of the whitespace run ending at `end` (used to hold back trailing
// whitespace so the reasoning text can be right-trimmed at `</think>` the way
// the chat template does).
size_t trailing_whitespace_run(const std::string & buf, size_t end) {
    size_t k = 0;
    while (k < end && is_space(buf[end - 1 - k])) {
        k++;
    }
    return k;
}

// Longest suffix of `buf` that is a proper prefix of `marker`.  Held back so a
// marker split across two token pieces is never emitted as content/reasoning.
size_t partial_marker_suffix(const std::string & buf, const char * marker, size_t mlen) {
    const size_t n = std::min(buf.size(), mlen - 1);
    for (size_t k = n; k > 0; --k) {
        if (buf.compare(buf.size() - k, k, marker, 0, k) == 0) {
            return k;
        }
    }
    return 0;
}

// A `<parameter>` value: the template writes the value verbatim between the
// opening tag's newline and the closing tag's newline, so strip exactly one
// newline on each side and then try to recover its JSON type (a mapping or
// sequence is serialized as JSON by the template; a scalar string is raw).
ojson parse_parameter_value(const std::string & raw) {
    std::string v = raw;
    if (!v.empty() && v[0] == '\n') {
        v.erase(0, 1);
    }
    if (!v.empty() && v[v.size() - 1] == '\n') {
        v.pop_back();
    }
    try {
        return ojson::parse(v);
    } catch (...) {
        return ojson(v);
    }
}

std::string make_tool_call_id() {
    static std::atomic<uint64_t> counter{0};
    const uint64_t n = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t t = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    char buf[32];
    snprintf(buf, sizeof(buf), "call_%08llx%04llx", (unsigned long long)t, (unsigned long long)(n & 0xffff));
    return buf;
}

} // namespace

bool parse_tool_block(const std::string & block, response_tool_call & out) {
    const size_t f = block.find("<function=");
    if (f != std::string::npos) {
        const size_t name_start = f + sizeof("<function=") - 1;
        const size_t name_end = block.find('>', name_start);
        if (name_end == std::string::npos) {
            return false;
        }
        out.name = trim_spaces(block.substr(name_start, name_end - name_start));
        if (out.name.empty()) {
            return false;
        }
        ojson args = ojson::object();
        size_t p = name_end + 1;
        while (true) {
            const size_t ps = block.find("<parameter=", p);
            if (ps == std::string::npos) {
                break;
            }
            const size_t key_start = ps + sizeof("<parameter=") - 1;
            const size_t key_end = block.find('>', key_start);
            if (key_end == std::string::npos) {
                break;
            }
            const std::string key = trim_spaces(block.substr(key_start, key_end - key_start));
            const size_t val_start = key_end + 1;
            const size_t val_end = block.find("</parameter>", val_start);
            if (val_end == std::string::npos) {
                break;
            }
            if (!key.empty()) {
                args[key] = parse_parameter_value(block.substr(val_start, val_end - val_start));
            }
            p = val_end + kParamCloseLen;
        }
        out.arguments = args.dump();
        return true;
    }

    // Fallback for templates that emit a JSON object body instead of the XML
    // form: {"name": "...", "arguments": {...}}.
    const size_t b = block.find('{');
    if (b != std::string::npos) {
        try {
            ojson j = ojson::parse(block.substr(b));
            if (j.contains("name") && j["name"].is_string()) {
                out.name = j["name"].get<std::string>();
                ojson args = ojson::object();
                if (j.contains("arguments")) {
                    if (j["arguments"].is_string()) {
                        try {
                            args = ojson::parse(j["arguments"].get<std::string>());
                        } catch (...) {
                        }
                    } else if (j["arguments"].is_object()) {
                        args = j["arguments"];
                    }
                } else if (j.contains("parameters") && j["parameters"].is_object()) {
                    args = j["parameters"];
                }
                out.arguments = args.dump();
                return !out.name.empty();
            }
        } catch (...) {
        }
    }
    return false;
}

response_parser::response_parser(bool thinking, bool parse_tools, emit_fn emit)
    : thinking_(thinking), parse_tools_(parse_tools), emit_(std::move(emit)) {
    st_ = thinking_ ? state::reasoning : state::content;
    strip_content_leading_ = thinking_;
}

void response_parser::push_reasoning(std::string text) {
    if (text.empty()) {
        return;
    }
    if (reason_start_) {
        if (text.size() >= kThinkOpenLen && text.compare(0, kThinkOpenLen, kThinkOpen) == 0) {
            text.erase(0, kThinkOpenLen);
        }
        text = lstrip_spaces(text);
        if (text.empty()) {
            return;
        }
        reason_start_ = false;
    }
    reasoning_ += text;
    response_piece p;
    p.kind = response_piece_kind::reasoning;
    p.text = text;
    emit_(p);
}

void response_parser::push_content(std::string text, bool before_tool) {
    if (text.empty()) {
        return;
    }
    if (strip_content_leading_) {
        text = lstrip_newlines(text);
        if (text.empty()) {
            return;
        }
        strip_content_leading_ = false;
    }
    if (before_tool) {
        text = rstrip_spaces(text);
        if (text.empty()) {
            return;
        }
    }
    content_ += text;
    response_piece p;
    p.kind = response_piece_kind::content;
    p.text = text;
    emit_(p);
}

void response_parser::process(bool final) {
    bool progress = true;
    while (progress) {
        progress = false;
        switch (st_) {
        case state::reasoning: {
            const size_t p = buf_.find(kThinkClose);
            if (p != std::string::npos) {
                push_reasoning(rstrip_spaces(buf_.substr(0, p)));
                buf_.erase(0, p + kThinkCloseLen);
                st_ = state::content;
                strip_content_leading_ = true;
                progress = true;
            } else if (final) {
                push_reasoning(rstrip_spaces(buf_));
                buf_.clear();
            } else {
                const size_t marker_keep = partial_marker_suffix(buf_, kThinkClose, kThinkCloseLen);
                const size_t base = buf_.size() - marker_keep;
                const size_t keep = marker_keep + trailing_whitespace_run(buf_, base);
                if (buf_.size() > keep) {
                    std::string seg = buf_.substr(0, buf_.size() - keep);
                    buf_.erase(0, buf_.size() - keep);
                    push_reasoning(std::move(seg));
                }
            }
            break;
        }
        case state::content: {
            if (!parse_tools_) {
                if (!buf_.empty()) {
                    push_content(std::move(buf_), false);
                    buf_.clear();
                }
                break;
            }
            const size_t p = buf_.find(kToolOpen);
            if (p != std::string::npos) {
                push_content(buf_.substr(0, p), true);
                buf_.erase(0, p + kToolOpenLen);
                st_ = state::tool;
                progress = true;
                break;
            }
            if (final) {
                push_content(std::move(buf_), false);
                buf_.clear();
                break;
            }
            const size_t keep = partial_marker_suffix(buf_, kToolOpen, kToolOpenLen);
            if (buf_.size() > keep) {
                std::string seg = buf_.substr(0, buf_.size() - keep);
                buf_.erase(0, buf_.size() - keep);
                push_content(std::move(seg), false);
            }
            break;
        }
        case state::tool: {
            const size_t p = buf_.find(kToolClose);
            if (p != std::string::npos) {
                std::string blk = buf_.substr(0, p);
                buf_.erase(0, p + kToolCloseLen);
                response_tool_call tc;
                if (parse_tool_block(blk, tc)) {
                    tc.id = make_tool_call_id();
                    tools_.push_back(tc);
                    response_piece piece;
                    piece.kind = response_piece_kind::tool_call;
                    piece.call = tc;
                    emit_(piece);
                } else {
                    push_content(std::string(kToolOpen) + blk + std::string(kToolClose), false);
                }
                st_ = state::content;
                strip_content_leading_ = true;
                progress = true;
            } else if (final) {
                response_tool_call tc;
                if (parse_tool_block(buf_, tc)) {
                    tc.id = make_tool_call_id();
                    tools_.push_back(tc);
                    response_piece piece;
                    piece.kind = response_piece_kind::tool_call;
                    piece.call = tc;
                    emit_(piece);
                } else if (!buf_.empty()) {
                    push_content(std::string(kToolOpen) + buf_, false);
                }
                buf_.clear();
                st_ = state::content;
                progress = true;
            }
            break;
        }
        }
    }
}

void response_parser::feed(const std::string & piece) {
    if (piece.empty()) {
        return;
    }
    buf_ += piece;
    process(false);
}

void response_parser::finish() {
    process(true);
    // whitespace-only content alongside tool calls is separator noise
    if (!tools_.empty() && trim_spaces(content_).empty()) {
        content_.clear();
    }
}

} // namespace si
