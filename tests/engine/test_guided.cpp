// Hermetic tests for JSON-schema guided decoding.  The vocabulary is synthetic:
// one token per byte plus a handful of multi-byte pieces, which lets the test
// exercise token boundaries inside keys, strings, numbers and escapes without a
// model or a GGUF.
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "guided.h"
#include "json.hpp"
#include "sampler.h"

using namespace si;
using json = nlohmann::json;

static int g_fail = 0;
static void check(bool ok, const char * name) {
    if (ok) {
        printf("  [%s] OK\n", name);
        return;
    }
    g_fail++;
    printf("  [%s] FAILED\n", name);
}

static constexpr int kEos = 256;

struct vocab {
    std::vector<std::string> pieces;
    vocab() {
        pieces.resize(257);
        for (int i = 0; i < 256; i++) {
            pieces[(size_t)i].assign(1, (char)i);
        }
        pieces[kEos].clear();
    }
    int find(const std::string & s) const {
        for (size_t i = 0; i < pieces.size(); i++) {
            if (pieces[i] == s) {
                return (int)i;
            }
        }
        return -1;
    }
    int add(const std::string & s) {
        pieces.push_back(s);
        return (int)pieces.size() - 1;
    }
};

static std::shared_ptr<const guided_constraint> make_gc(const vocab & v, const std::string & schema) {
    return make_guided_constraint(schema, v.pieces, {kEos});
}

// Greedy driver: at each step the next script token has the highest logit and
// EOS the second-highest; anything the constraint forbids is masked.  The
// first allowed token wins, so a script token that is not legal silently falls
// back to another legal one (and the resulting JSON is checked).
static std::string run_script(const std::shared_ptr<const guided_constraint> & gc, const vocab & v,
                              const std::vector<int> & script, int max_steps = 256) {
    auto st = make_guided_state(*gc);
    std::string text;
    std::vector<float> logits(v.pieces.size());
    for (int step = 0; step < max_steps; step++) {
        std::fill(logits.begin(), logits.end(), -1000.f);
        logits[kEos] = 50.f;
        if (step < (int)script.size()) {
            logits[(size_t)script[(size_t)step]] = 100.f;
        }
        guided_apply(*gc, *st, logits.data(), (int)logits.size());
        int best = 0;
        for (int i = 1; i < (int)logits.size(); i++) {
            if (logits[(size_t)i] > logits[(size_t)best]) {
                best = i;
            }
        }
        if (best == kEos) {
            break;
        }
        if (!std::isfinite(logits[(size_t)best])) {
            break;
        }
        text += v.pieces[(size_t)best];
        guided_commit(*gc, *st, best);
    }
    return text;
}

static std::vector<int> ids(const vocab & v, const std::vector<std::string> & pieces) {
    std::vector<int> out;
    for (const auto & s : pieces) {
        const int id = v.find(s);
        if (id >= 0) {
            out.push_back(id);
        }
    }
    return out;
}

static void test_object_and_required(const vocab & v) {
    const std::string schema = R"({
        "type":"object",
        "properties":{
            "name":{"type":"string"},
            "age":{"type":"integer","minimum":0}
        },
        "required":["name","age"],
        "additionalProperties":false
    })";
    auto gc = make_gc(v, schema);
    const std::vector<std::string> script = {"{", "\"", "name", "\"", ":", "\"", "Alice", "\"", ",",
                                             "\"", "age", "\"", ":", "42", "}"};
    const std::string text = run_script(gc, v, ids(v, script));
    check(text == R"({"name":"Alice","age":42})", "object: exact guided text");
    bool ok = true;
    try {
        ok = json::parse(text) == json::parse(R"({"name":"Alice","age":42})");
    } catch (...) {
        ok = false;
    }
    check(ok, "object: parses and matches");
}

static void test_key_filter(const vocab & v) {
    const std::string schema = R"({"type":"object","properties":{"name":{"type":"string"}},
                                   "required":["name"],"additionalProperties":false})";
    auto gc = make_gc(v, schema);
    auto st = make_guided_state(*gc);
    guided_commit(*gc, *st, v.find("{"));
    guided_commit(*gc, *st, v.find("\""));
    std::vector<float> logits(v.pieces.size(), 0.f);
    guided_apply(*gc, *st, logits.data(), (int)logits.size());
    check(!std::isfinite(logits[(size_t)v.find("b")]), "key: unknown first letter masked");
    check(std::isfinite(logits[(size_t)v.find("n")]), "key: declared first letter allowed");

    // End the key early: no property is complete yet, so the quote must be
    // rejected too.
    guided_commit(*gc, *st, v.find("n"));
    std::fill(logits.begin(), logits.end(), 0.f);
    guided_apply(*gc, *st, logits.data(), (int)logits.size());
    check(!std::isfinite(logits[(size_t)v.find("\"")]), "key: partial key cannot close");
}

static void test_required_blocks_empty_object(const vocab & v) {
    const std::string schema = R"({"type":"object","properties":{"a":{"type":"integer"}},
                                   "required":["a"],"additionalProperties":false})";
    auto gc = make_gc(v, schema);
    auto st = make_guided_state(*gc);
    guided_commit(*gc, *st, v.find("{"));
    std::vector<float> logits(v.pieces.size(), 0.f);
    guided_apply(*gc, *st, logits.data(), (int)logits.size());
    check(!std::isfinite(logits[(size_t)v.find("}")]), "required: '{}' rejected");
}

static void test_duplicate_key_and_forced_eos(const vocab & v) {
    // Duplicate: "a" is followed by another "a" while "b" is still missing, so
    // the comma is legal but the repeated key must not close.
    const std::string schema = R"({"type":"object","properties":{"a":{"type":"integer"},
                                   "b":{"type":"integer"}},"required":["a","b"],
                                   "additionalProperties":false})";
    auto gc = make_gc(v, schema);
    auto st = make_guided_state(*gc);
    for (const char * p : {"{", "\"", "a", "\"", ":", "1", ",", "\"", "a"}) {
        guided_commit(*gc, *st, v.find(p));
    }
    std::vector<float> logits(v.pieces.size(), 0.f);
    guided_apply(*gc, *st, logits.data(), (int)logits.size());
    check(!std::isfinite(logits[(size_t)v.find("\"")]), "object: duplicate key cannot close");

    // With every declared property seen, a comma cannot start another key
    // either: the only legal continuation is the closing brace.
    const std::string schema2 = R"({"type":"object","properties":{"a":{"type":"integer"}},
                                    "required":["a"],"additionalProperties":false})";
    auto gc2 = make_gc(v, schema2);
    auto st2 = make_guided_state(*gc2);
    for (const char * p : {"{", "\"", "a", "\"", ":", "1"}) {
        guided_commit(*gc2, *st2, v.find(p));
    }
    std::fill(logits.begin(), logits.end(), 0.f);
    guided_apply(*gc2, *st2, logits.data(), (int)logits.size());
    check(!std::isfinite(logits[(size_t)v.find(",")]), "object: comma blocked when no key remains");
    check(std::isfinite(logits[(size_t)v.find("}")]), "object: close still allowed when no key remains");

    // A completed document admits EOS and nothing else (no trailing-space
    // budget drain).
    auto gc3 = make_gc(v, "");
    auto st3 = make_guided_state(*gc3);
    guided_commit(*gc3, *st3, v.find("{"));
    guided_commit(*gc3, *st3, v.find("}"));
    check(guided_finished(*gc3, *st3), "json_object: completed document is finished");
    std::fill(logits.begin(), logits.end(), 0.f);
    guided_apply(*gc3, *st3, logits.data(), (int)logits.size());
    check(std::isfinite(logits[(size_t)kEos]), "json_object: EOS allowed at completion");
    check(!std::isfinite(logits[(size_t)v.find(" ")]), "json_object: no trailing-space tokens");
}

static void test_enum_shortest_form(const vocab & v) {
    const std::string schema = R"({"type":"object","properties":{
        "status":{"enum":["ok","failed"]}},"required":["status"],"additionalProperties":false})";
    auto gc = make_gc(v, schema);
    const std::string text = run_script(gc, v, ids(v, {"{", "\"", "status", "\"", ":", "\"", "failed", "\"", "}"}));
    check(text == R"({"status":"failed"})", "enum: value restricted to the declared spelling");

    auto st = make_guided_state(*gc);
    guided_commit(*gc, *st, v.find("{"));
    guided_commit(*gc, *st, v.find("\""));
    guided_commit(*gc, *st, v.find("status"));
    guided_commit(*gc, *st, v.find("\""));
    guided_commit(*gc, *st, v.find(":"));
    guided_commit(*gc, *st, v.find("\""));
    std::vector<float> logits(v.pieces.size(), 0.f);
    guided_apply(*gc, *st, logits.data(), (int)logits.size());
    check(std::isfinite(logits[(size_t)v.find("o")]), "enum: 'o' allowed");
    check(!std::isfinite(logits[(size_t)v.find("x")]), "enum: 'x' masked");
}

static void test_array_and_integer_range(const vocab & v) {
    const std::string schema = R"({"type":"array","items":{"type":"integer","minimum":2,"maximum":5},
                                   "minItems":1,"maxItems":3})";
    auto gc = make_gc(v, schema);
    const std::string text = run_script(gc, v, ids(v, {"[", "2", ",", "5", "]"}));
    check(text == "[2,5]", "array: items and bounds");

    // A first digit that cannot reach [2,5] must be masked, not accepted and
    // then dead-ended at the comma.
    auto st = make_guided_state(*gc);
    guided_commit(*gc, *st, v.find("["));
    std::vector<float> logits(v.pieces.size(), 0.f);
    guided_apply(*gc, *st, logits.data(), (int)logits.size());
    check(!std::isfinite(logits[(size_t)v.find("9")]), "integer: unreachable prefix masked");
    check(std::isfinite(logits[(size_t)v.find("2")]), "integer: reachable prefix allowed");
    check(!std::isfinite(logits[(size_t)v.find("A")]), "array: non-JSON token masked");

    // maxItems: after three items only ']' may follow.
    auto st2 = make_guided_state(*gc);
    for (const char * p : {"[", "2", ",", "2", ",", "2"}) {
        guided_commit(*gc, *st2, v.find(p));
    }
    std::fill(logits.begin(), logits.end(), 0.f);
    guided_apply(*gc, *st2, logits.data(), (int)logits.size());
    check(!std::isfinite(logits[(size_t)v.find(",")]), "array: maxItems blocks a fourth element");
    check(std::isfinite(logits[(size_t)v.find("]")]), "array: maxItems still allows close");
}

static void test_bounded_number(const vocab & v) {
    auto gc = make_gc(v, R"({"type":"number","maximum":1})");
    auto st = make_guided_state(*gc);
    std::vector<float> logits(v.pieces.size(), 0.f);
    guided_apply(*gc, *st, logits.data(), (int)logits.size());
    check(!std::isfinite(logits[(size_t)v.find("9")]), "number: leading 9 cannot reach max 1");
    check(std::isfinite(logits[(size_t)v.find("0")]), "number: leading 0 allowed");
    check(std::isfinite(logits[(size_t)v.find("-")]), "number: sign allowed");

    guided_commit(*gc, *st, v.find("0"));
    std::fill(logits.begin(), logits.end(), 0.f);
    guided_apply(*gc, *st, logits.data(), (int)logits.size());
    check(std::isfinite(logits[(size_t)v.find(".")]), "number: fraction allowed under a bound");
    check(!std::isfinite(logits[(size_t)v.find("e")]), "number: exponent masked under a bound");
    check(!std::isfinite(logits[(size_t)v.find("5")]), "number: no second integer digit after 0");

    for (const char * p : {".", "9", "5"}) {
        guided_commit(*gc, *st, v.find(p));
    }
    check(guided_finished(*gc, *st), "number: 0.95 completes within max 1");
}

static void test_string_length_and_escape(const vocab & v) {
    const std::string schema = R"({"type":"string","maxLength":3})";
    auto gc = make_gc(v, schema);
    auto st = make_guided_state(*gc);
    guided_commit(*gc, *st, v.find("\""));
    guided_commit(*gc, *st, v.find("a"));
    guided_commit(*gc, *st, v.find("b"));
    guided_commit(*gc, *st, v.find("c"));
    std::vector<float> logits(v.pieces.size(), 0.f);
    guided_apply(*gc, *st, logits.data(), (int)logits.size());
    check(!std::isfinite(logits[(size_t)v.find("d")]), "string: maxLength blocks a fourth char");
    check(std::isfinite(logits[(size_t)v.find("\"")]), "string: maxLength still allows close");

    // \uXXXX is five raw bytes but one decoded character.
    auto st2 = make_guided_state(*gc);
    guided_commit(*gc, *st2, v.find("\""));
    guided_commit(*gc, *st2, v.find("\\"));
    guided_commit(*gc, *st2, v.find("u"));
    for (const char * h : {"0", "0", "4", "1"}) {
        guided_commit(*gc, *st2, v.find(h));
    }
    guided_commit(*gc, *st2, v.find("a"));
    check(guided_finished(*gc, *st2) == false, "string: two decoded chars is not finished");
    guided_commit(*gc, *st2, v.find("\""));
    check(guided_finished(*gc, *st2), "string: escaped char counts once");
}

static void test_refs_and_anyof(const vocab & v) {
    const std::string schema = R"({
        "$defs":{"point":{"type":"object","properties":{"x":{"type":"integer"}},
                  "required":["x"],"additionalProperties":false}},
        "type":"array","items":{"$ref":"#/$defs/point"}
    })";
    auto gc = make_gc(v, schema);
    const std::string text = run_script(gc, v, ids(v, {"[", "{", "\"", "x", "\"", ":", "1", "}", "]"}));
    check(text == R"([{"x":1}])", "ref: $defs resolved");

    const std::string schema2 = R"({"anyOf":[{"type":"integer"},{"type":"string"}]})";
    auto gc2 = make_gc(v, schema2);
    const std::string t2 = run_script(gc2, v, ids(v, {"\"", "h", "i", "\""}));
    check(t2 == R"("hi")", "anyOf: string branch taken");
    const std::string t3 = run_script(gc2, v, ids(v, {"7"}));
    check(t3 == "7", "anyOf: number branch taken");
}

static void test_recursive_ref(const vocab & v) {
    const std::string schema = R"({"type":"object","properties":{"child":{"$ref":"#"}},
                                   "additionalProperties":false})";
    auto gc = make_gc(v, schema);
    const std::string text =
        run_script(gc, v, ids(v, {"{", "\"", "child", "\"", ":", "{", "\"", "child", "\"", ":", "{", "}", "}", "}"}));
    check(text == R"({"child":{"child":{}}})", "ref: recursive schema resolves");
}

static void test_generic_json_object(const vocab & v) {
    auto gc = make_gc(v, "");
    const std::string text = run_script(gc, v, ids(v, {"{", "}"}));
    check(text == "{}", "json_object: empty object accepted");
}

static void test_sampler_integration(const vocab & v) {
    auto gc = make_gc(v, R"({"type":"string"})");
    gen_params gp;
    gp.temperature = 0.f;
    gp.guided = gc;
    sampler_state ss;
    ss.seed(1);

    // The highest logit is 'A' (only legal once the string is open), so an
    // unconstrained sampler would emit it.  The first token must be the quote.
    std::vector<float> lg(v.pieces.size(), 0.f);
    lg[(size_t)v.find("A")] = 10.f;
    const int t0 = sample_token(lg.data(), (int)lg.size(), gp, {}, ss);
    check(t0 == v.find("\""), "sampler: guided mask forces the opening quote");

    std::vector<float> lg2(v.pieces.size(), 0.f);
    lg2[(size_t)v.find("p")] = 10.f;
    const int t1 = sample_token(lg2.data(), (int)lg2.size(), gp, {}, ss);
    check(t1 == v.find("p"), "sampler: legal byte sampled");

    // EOS is masked until the string is closed.
    std::vector<float> lg3(v.pieces.size(), 0.f);
    lg3[kEos] = 1000.f;
    const int t2 = sample_token(lg3.data(), (int)lg3.size(), gp, {}, ss);
    check(t2 != kEos, "sampler: EOS masked before the value is complete");
}

int main() {
    vocab v;
    // Multi-byte pieces used by the scripts.
    for (const char * s : {"name", "age", "Alice", "status", "failed", "42", "x", "p", "child"}) {
        if (v.find(s) < 0) {
            v.add(s);
        }
    }
    // Keep the EOS id stable: add() must not shadow it.
    check(v.pieces.size() > (size_t)kEos, "vocab: EOS index valid");

    test_object_and_required(v);
    test_key_filter(v);
    test_required_blocks_empty_object(v);
    test_duplicate_key_and_forced_eos(v);
    test_enum_shortest_form(v);
    test_array_and_integer_range(v);
    test_bounded_number(v);
    test_string_length_and_escape(v);
    test_refs_and_anyof(v);
    test_recursive_ref(v);
    test_generic_json_object(v);
    test_sampler_integration(v);
    if (g_fail == 0) {
        printf("guided: all checks OK\n");
    }
    return g_fail == 0 ? 0 : 1;
}
