#include "config.h"
#include "onnx_wire.h"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>


using namespace si::router;

namespace {

int g_fail = 0;

#define CHECK(cond, ...)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                                                      \
            fprintf(stderr, __VA_ARGS__);                                                                              \
            fprintf(stderr, "\n");                                                                                     \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

bool parse_fails(const std::string & text) {
    try {
        parse_config(text);
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

// A model file that exists and is readable on every Linux box.
const char * kModelPath = "/etc/hostname";

std::string one_model(const std::string & extra) {
    return std::string("{\"models\": [{\"id\": \"m1\", \"path\": \"") + kModelPath + "\", \"http\": \"127.0.0.1:18081\"" +
           extra + "}]}";
}

void test_comments() {
    const char * text = R"({
  // a line comment
  "router": {
    "listen": "127.0.0.1:9000"   /* a block comment */
  },
  /* a block comment
     spanning lines */
  "models": [
    { "id": "m1", "path": "/etc/hostname", "http": "127.0.0.1:18081" }  // trailing
  ]
})";
    router_config c;
    try {
        c = parse_config(text);
    } catch (const std::exception & ex) {
        CHECK(false, "a config with comments must parse: %s", ex.what());
        return;
    }
    CHECK(c.listen_port == 9000, "listen port: %d", c.listen_port);
    CHECK(c.models.size() == 1, "model count: %zu", c.models.size());
    CHECK(c.models[0].id == "m1", "model id: %s", c.models[0].id.c_str());
}

// A comment marker inside a string is data, not a comment.
void test_comment_in_string() {
    const std::string text = std::string("{\"models\": [{\"id\": \"a//b\", \"path\": \"") + kModelPath +
                            "\", \"http\": \"127.0.0.1:18081\"}]}";
    router_config c;
    try {
        c = parse_config(text);
    } catch (const std::exception & ex) {
        CHECK(false, "parse failed: %s", ex.what());
        return;
    }
    CHECK(c.models[0].id == "a//b", "the id must keep the '//': %s", c.models[0].id.c_str());
}

void test_derived_fields() {
    const char * text = R"({
  "router": { "listen": "0.0.0.0:8080", "request_timeout_s": 120, "device_vram_mb": 8192 },
  "defaults": { "env": { "PF_PREFIX_CACHE": "1" } },
  "models": [{
    "id": "m1", "path": "/etc/hostname", "http": "127.0.0.1:18081",
    "args": ["--ctx", "32768"],
    "device": { "group": "gpu0", "filter": "level_zero:0" },
    "memory": { "vram_budget_mb": 4096, "kv_cap_mb": 2048, "max_slots": 4,
                "prefix_cache_vram_mb": 512, "prefix_cache_dir": "/tmp/pc" }
  }]
})";
    router_config c;
    try {
        c = parse_config(text);
    } catch (const std::exception & ex) {
        CHECK(false, "parse failed: %s", ex.what());
        return;
    }
    const model_config & m = c.models[0];
    CHECK(m.vram_budget_mb == 4096, "budget: %llu", (unsigned long long)m.vram_budget_mb);
    CHECK(m.kv_cap_mb == 2048, "kv cap: %d", m.kv_cap_mb);
    CHECK(m.max_slots == 4, "max slots: %d", m.max_slots);
    CHECK(m.pc_vram_mb == 512, "pc vram: %d", m.pc_vram_mb);
    CHECK(m.pc_dir == "/tmp/pc", "pc dir: %s", m.pc_dir.c_str());
    CHECK(m.device_group == "gpu0", "device group: %s", m.device_group.c_str());
    CHECK(m.device_filter == "level_zero:0", "device filter: %s", m.device_filter.c_str());
    CHECK(c.request_timeout_s == 120, "request timeout: %d", c.request_timeout_s);
    CHECK(c.default_env.size() == 1 && c.default_env[0] == "PF_PREFIX_CACHE=1", "default env");

    // The derived command line: the memory knobs become the worker's flags.
    const std::vector<std::string> & a = m.argv;
    auto has = [&](const std::string & s) {
        for (const std::string & x : a) {
            if (x == s) {
                return true;
            }
        }
        return false;
    };
    CHECK(has("--model"), "argv must carry --model");
    CHECK(has("--kv-cap-mb"), "argv must carry --kv-cap-mb");
    CHECK(has("--max-slots"), "argv must carry --max-slots");
    CHECK(has("--pc-vram-mb"), "argv must carry --pc-vram-mb");
    CHECK(has("--pc-dir"), "argv must carry --pc-dir");
    CHECK(has("serve"), "argv must end with the serve command");
    CHECK(a.back() == "serve", "the command goes last: %s", a.back().c_str());
}


void test_onnx_backend() {
    const char * text = R"({
  "defaults": { "binary": "sycl-infer", "binary_onnx": "sycl-infer-onnx" },
  "models": [{
    "id": "layout", "backend": "onnx", "path": "/etc/hostname",
    "http": "127.0.0.1:18081", "control": "127.0.0.1:18082",
    "device": { "kind": "gpu", "group": "gpu0", "filter": "level_zero:0" },
    "onnx": { "drop_outputs": ["fetch_name_2"], "dnnl": false }
  }]
})";
    router_config c;
    try {
        c = parse_config(text);
    } catch (const std::exception & ex) {
        CHECK(false, "an ONNX model must parse: %s", ex.what());
        return;
    }
    const model_config & m = c.models[0];
    CHECK(m.backend == "onnx", "backend: %s", m.backend.c_str());
    CHECK(m.binary == "sycl-infer-onnx", "binary: %s", m.binary.c_str());
    CHECK(m.device_kind == "gpu", "device kind: %s", m.device_kind.c_str());
    CHECK(m.device_filter == "level_zero:0", "device filter: %s", m.device_filter.c_str());
    CHECK(m.drop_outputs.size() == 1 && m.drop_outputs[0] == "fetch_name_2",
          "drop_outputs");
    CHECK(m.dnnl_set && !m.dnnl, "dnnl must be carried through");
    auto has = [&](const std::string & s) {
        for (const std::string & x : m.argv) {
            if (x == s) return true;
        }
        return false;
    };
    CHECK(has("--model"), "ONNX argv must carry --model");
    CHECK(has("layout=/etc/hostname"), "ONNX argv must carry the id=path spec");
    CHECK(has("--device") && has("gpu"), "ONNX argv must carry the device kind");
    CHECK(has("--drop-outputs") && has("fetch_name_2"), "ONNX argv must carry drop_outputs");
    CHECK(has("--dnnl") && has("0"), "ONNX argv must carry dnnl=0");
    CHECK(!has("serve"), "an ONNX worker has no serve subcommand");
}

void test_onnx_wire_peek() {
    const std::vector<uint8_t> body = si::onnxwire::encode_request("layout", {});
    std::string model, err;
    CHECK(si::onnxwire::peek_request_model(body.data(), body.size(), model, err),
          "a valid ONNX request must expose its model: %s", err.c_str());
    CHECK(model == "layout", "model: %s", model.c_str());
    CHECK(!si::onnxwire::peek_request_model(body.data(), 5, model, err),
          "a truncated header must be rejected");
}

void test_validation() {
    // duplicate ids
    CHECK(parse_fails("{\"models\": ["
                       "{\"id\": \"a\", \"path\": \"/etc/hostname\", \"http\": \"127.0.0.1:18081\"},"
                       "{\"id\": \"a\", \"path\": \"/etc/hostname\", \"http\": \"127.0.0.1:18082\"}]}"),
          "duplicate ids must be rejected");
    // missing models
    CHECK(parse_fails("{}"), "a config without models must be rejected");
    CHECK(parse_fails("{\"models\": []}"), "an empty model list must be rejected");
    // a missing model file
    CHECK(parse_fails("{\"models\": [{\"id\": \"a\", \"path\": \"/nonexistent/model.gguf\", "
                       "\"http\": \"127.0.0.1:18081\"}]}"),
          "a missing model file must be rejected");
    // a port collision
    CHECK(parse_fails("{\"models\": ["
                       "{\"id\": \"a\", \"path\": \"/etc/hostname\", \"http\": \"127.0.0.1:18081\"},"
                       "{\"id\": \"b\", \"path\": \"/etc/hostname\", \"http\": \"127.0.0.1:18081\"}]}"),
          "two models on one port must be rejected");
    // a reserved argument
    CHECK(parse_fails(one_model(", \"args\": [\"--port\", \"9999\"]")),
          "the router's own flags must not be overridden in args");
    CHECK(parse_fails(one_model(", \"args\": [\"--model\", \"/other.gguf\"]")),
          "--model must not be overridden in args");
    // a malformed document
    CHECK(parse_fails("{"), "a malformed document must be rejected");
    CHECK(parse_fails("[]"), "the top level must be an object");
    // a bad endpoint
    CHECK(parse_fails("{\"models\": [{\"id\": \"a\", \"path\": \"/etc/hostname\", \"http\": \"nonsense\"}]}"),
          "a malformed endpoint must be rejected");
    // a zero budget is not a budget
    CHECK(parse_fails(one_model(", \"memory\": {\"vram_budget_mb\": 0, \"kv_cap_mb\": 0}")),
          "a zero kv cap must be rejected");
    // the budgets of one device group must fit the device
    const char * over = R"({
  "router": { "device_vram_mb": 4096 },
  "models": [
    { "id": "a", "path": "/etc/hostname", "http": "127.0.0.1:18081",
      "device": { "group": "gpu0" }, "memory": { "vram_budget_mb": 3000 } },
    { "id": "b", "path": "/etc/hostname", "http": "127.0.0.1:18082",
      "device": { "group": "gpu0" }, "memory": { "vram_budget_mb": 3000 } }
  ]
})";
    CHECK(parse_fails(over), "budgets that overflow the device must be rejected");
}

void test_remote() {
    const char * text = R"({
  "models": [
    { "id": "remote-1", "remote": { "http": "http://gpu-host-b:19081", "control": "gpu-host-b:19082" } }
  ]
})";
    router_config c;
    try {
        c = parse_config(text);
    } catch (const std::exception & ex) {
        CHECK(false, "a remote model must parse: %s", ex.what());
        return;
    }
    CHECK(c.models[0].remote, "the model must be marked remote");
    CHECK(c.models[0].http_host == "gpu-host-b", "host: %s", c.models[0].http_host.c_str());
    CHECK(c.models[0].http_port == 19081, "port: %d", c.models[0].http_port);
    CHECK(c.models[0].path.empty(), "a remote model has no local path");
}

void test_bearer_tokens() {
    const char * text = R"({
  "router": { "auth": { "bearer_tokens": ["sk-a", "sk-b"] } },
  "models": [ { "id": "m1", "path": "/etc/hostname", "http": "127.0.0.1:18081" } ]
})";
    router_config c = parse_config(text);
    CHECK(c.bearer_tokens.size() == 2, "token count: %zu", c.bearer_tokens.size());
}

} // namespace

int main() {
    test_comments();
    test_comment_in_string();
    test_derived_fields();
    test_validation();
    test_remote();
    test_bearer_tokens();
    test_onnx_backend();
    test_onnx_wire_peek();
    if (g_fail) {
        fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_router_config: OK\n");
    return 0;
}
