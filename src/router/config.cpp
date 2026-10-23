#include "config.h"

#include "json.hpp"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <ios>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>



namespace si {
namespace router {

using json = nlohmann::json;

namespace {

[[noreturn]] void cfg_err(const std::string & msg) {
    throw std::runtime_error("config: " + msg);
}

const json * member(const json & obj, const char * key) {
    auto it = obj.find(key);
    if (it == obj.end() || it->is_null()) {
        return nullptr;
    }
    return &(*it);
}

std::string get_str(const json & obj, const char * key, const std::string & def) {
    const json * v = member(obj, key);
    if (!v) {
        return def;
    }
    if (!v->is_string()) {
        cfg_err(std::string("'") + key + "' must be a string");
    }
    return v->get<std::string>();
}

std::string get_str_req(const json & obj, const char * key, const std::string & where) {
    const json * v = member(obj, key);
    if (!v) {
        cfg_err(where + ": missing required string '" + key + "'");
    }
    if (!v->is_string()) {
        cfg_err(where + ": '" + key + "' must be a string");
    }
    return v->get<std::string>();
}

int get_int(const json & obj, const char * key, int def) {
    const json * v = member(obj, key);
    if (!v) {
        return def;
    }
    if (!v->is_number_integer()) {
        cfg_err(std::string("'") + key + "' must be an integer");
    }
    return v->get<int>();
}

uint64_t get_u64(const json & obj, const char * key, uint64_t def) {
    const json * v = member(obj, key);
    if (!v) {
        return def;
    }
    if (!v->is_number_unsigned() && !(v->is_number_integer() && v->get<int64_t>() >= 0)) {
        cfg_err(std::string("'") + key + "' must be a non-negative integer");
    }
    return v->get<uint64_t>();
}

bool get_bool(const json & obj, const char * key, bool def) {
    const json * v = member(obj, key);
    if (!v) {
        return def;
    }
    if (!v->is_boolean()) {
        cfg_err(std::string("'") + key + "' must be a boolean");
    }
    return v->get<bool>();
}

// A list of strings.  A bare string is accepted as a one-element list,
// because `--args "--ctx 32768"` is a natural thing to write.
std::vector<std::string> get_str_list(const json & obj, const char * key) {
    std::vector<std::string> out;
    const json * v = member(obj, key);
    if (!v) {
        return out;
    }
    if (v->is_string()) {
        out.push_back(v->get<std::string>());
        return out;
    }
    if (!v->is_array()) {
        cfg_err(std::string("'") + key + "' must be an array of strings");
    }
    for (const auto & e : *v) {
        if (!e.is_string()) {
            cfg_err(std::string("'") + key + "' must contain only strings");
        }
        out.push_back(e.get<std::string>());
    }
    return out;
}

// An object of string -> string, flattened to "K=V".  Used for the child's
// environment: a map is the natural shape, and duplicates are an error
// rather than a last-one-wins surprise.
std::vector<std::string> get_kv_map(const json & obj, const char * key) {
    std::vector<std::string> out;
    const json * v = member(obj, key);
    if (!v) {
        return out;
    }
    if (!v->is_object()) {
        cfg_err(std::string("'") + key + "' must be an object of string -> string");
    }
    for (auto it = v->begin(); it != v->end(); ++it) {
        if (!it.value().is_string()) {
            cfg_err(std::string("'") + key + "." + it.key() + "' must be a string");
        }
        out.push_back(it.key() + "=" + it.value().get<std::string>());
    }
    return out;
}

// "host:port" -> (host, port)
void split_host_port(const std::string & s, const std::string & where, std::string & host, int & port) {
    const size_t colon = s.rfind(':');
    if (colon == std::string::npos || colon + 1 >= s.size()) {
        cfg_err(where + ": expected host:port, got '" + s + "'");
    }
    host = s.substr(0, colon);
    const std::string ps = s.substr(colon + 1);
    for (char c : ps) {
        if (c < '0' || c > '9') {
            cfg_err(where + ": '" + ps + "' is not a port number");
        }
    }
    port = atoi(ps.c_str());
    if (port <= 0 || port > 65535) {
        cfg_err(where + ": port " + ps + " out of range");
    }
    if (host.empty()) {
        cfg_err(where + ": empty host");
    }
}

bool file_readable(const std::string & p) {
    struct stat st;
    if (stat(p.c_str(), &st) != 0) {
        return false;
    }
    if (!S_ISREG(st.st_mode)) {
        return false;
    }
    return access(p.c_str(), R_OK) == 0;
}

} // namespace

std::vector<std::string> build_worker_argv(const model_config & m) {
    std::vector<std::string> a;
    // The control plane is common to both worker kinds.
    auto control = [&]() {
        if (m.control_port > 0 && !m.remote) {
            a.push_back("--control-port");
            a.push_back(std::to_string(m.control_port));
            if (!m.control_host.empty() && m.control_host != "127.0.0.1" &&
                m.control_host != "localhost") {
                a.push_back("--control-host");
                a.push_back(m.control_host);
            }
            if (!m.control_token_file.empty()) {
                a.push_back("--control-token-file");
                a.push_back(m.control_token_file);
            } else if (!m.control_token.empty()) {
                a.push_back("--control-token");
                a.push_back(m.control_token);
            }
        }
    };

    if (m.is_onnx()) {
        // One ONNX worker per model; its model id is the router's routing
        // key, so the binary wire request's `model` needs no rewrite.
        a.push_back("--model");
        a.push_back(m.id + "=" + m.path);
        if (!m.device_kind.empty()) {
            a.push_back("--device");
            a.push_back(m.device_kind);
        }
        a.push_back("--host");
        a.push_back(m.http_host);
        a.push_back("--port");
        a.push_back(std::to_string(m.http_port));
        control();
        for (const std::string & d : m.drop_outputs) {
            a.push_back("--drop-outputs");
            a.push_back(d);
        }
        if (m.dnnl_set && !m.dnnl) {
            a.push_back("--dnnl");
            a.push_back("0");
        }
        for (const std::string & s : m.args) {
            a.push_back(s);
        }
        return a;
    }

    a.push_back("--model");
    a.push_back(m.path);
    a.push_back("--host");
    a.push_back(m.http_host);
    a.push_back("--port");
    a.push_back(std::to_string(m.http_port));
    if (!m.device_kind.empty()) {
        a.push_back("--device");
        a.push_back(m.device_kind);
    }
    control();
    if (m.kv_cap_mb >= 0) {
        a.push_back("--kv-cap-mb");
        a.push_back(std::to_string(m.kv_cap_mb));
    }
    if (m.max_slots > 0) {
        a.push_back("--max-slots");
        a.push_back(std::to_string(m.max_slots));
    }
    if (m.pc_vram_mb >= 0) {
        a.push_back("--pc-vram-mb");
        a.push_back(std::to_string(m.pc_vram_mb));
    }
    if (!m.pc_dir.empty()) {
        a.push_back("--pc-dir");
        a.push_back(m.pc_dir);
    }
    for (const std::string & s : m.args) {
        a.push_back(s);
    }
    a.push_back("serve");
    return a;
}

router_config parse_config(const std::string & text) {
    json root;
    try {
        // ignore_comments = true: `//` and `/* */` are stripped by the
        // parser, so the config can carry the notes an operator needs.
        root = json::parse(text, nullptr, true, true);
    } catch (const std::exception & ex) {
        cfg_err(std::string("parse failed: ") + ex.what());
    }
    if (!root.is_object()) {
        cfg_err("the top level must be an object");
    }

    router_config cfg;

    // ---- router section ----------------------------------------------------
    const json * r = member(root, "router");
    json rsec = r ? *r : json::object();
    if (!rsec.is_object()) {
        cfg_err("'router' must be an object");
    }
    const std::string listen = get_str(rsec, "listen", "");
    if (!listen.empty()) {
        split_host_port(listen, "router.listen", cfg.listen_host, cfg.listen_port);
    }
    cfg.access_log = get_bool(rsec, "access_log", cfg.access_log);
    cfg.bearer_tokens = get_str_list(rsec, "bearer_tokens");
    const json * auth = member(rsec, "auth");
    if (auth) {
        if (!auth->is_object()) {
            cfg_err("'router.auth' must be an object");
        }
        // `bearer_tokens` under auth merges with the top-level spelling.
        for (const std::string & t : get_str_list(*auth, "bearer_tokens")) {
            cfg.bearer_tokens.push_back(t);
        }
    }
    cfg.request_timeout_s = get_int(rsec, "request_timeout_s", cfg.request_timeout_s);
    cfg.startup_timeout_s = get_int(rsec, "startup_timeout_s", cfg.startup_timeout_s);
    cfg.health_interval_s = get_int(rsec, "health_interval_s", cfg.health_interval_s);
    cfg.health_failures = get_int(rsec, "health_failures", cfg.health_failures);
    cfg.max_restarts = get_int(rsec, "max_restarts", cfg.max_restarts);
    cfg.restart_backoff_s = get_int(rsec, "restart_backoff_s", cfg.restart_backoff_s);
    cfg.log_dir = get_str(rsec, "log_dir", cfg.log_dir);
    cfg.binary = get_str(rsec, "binary", cfg.binary);
    cfg.binary_onnx = get_str(rsec, "binary_onnx", cfg.binary_onnx);
    cfg.device_vram_mb = get_u64(rsec, "device_vram_mb", cfg.device_vram_mb);

    // ---- defaults ----------------------------------------------------------
    const json * d = member(root, "defaults");
    json dsec = d ? *d : json::object();
    if (!dsec.is_object()) {
        cfg_err("'defaults' must be an object");
    }
    cfg.default_env = get_kv_map(dsec, "env");
    if (cfg.binary.empty()) {
        cfg.binary = get_str(dsec, "binary", cfg.binary);
    }
    cfg.binary_onnx = get_str(dsec, "binary_onnx", cfg.binary_onnx);

    // ---- models ------------------------------------------------------------
    const json * ms = member(root, "models");
    if (!ms) {
        cfg_err("missing required 'models'");
    }
    if (!ms->is_array()) {
        cfg_err("'models' must be an array");
    }
    if (ms->empty()) {
        cfg_err("'models' is empty");
    }

    std::set<std::string> ids;
    std::set<int> ports;
    for (const auto & e : *ms) {
        if (!e.is_object()) {
            cfg_err("each entry of 'models' must be an object");
        }
        const std::string where = "model[" + std::to_string(cfg.models.size()) + "]";
        model_config m;
        m.id = get_str_req(e, "id", where);
        if (m.id.empty()) {
            cfg_err(where + ": 'id' must not be empty");
        }
        if (m.id.find_first_of(" \t\n") != std::string::npos) {
            cfg_err(where + ": 'id' must not contain whitespace");
        }
        if (m.id.find('=') != std::string::npos) {
            cfg_err(where + ": 'id' must not contain '=' (it prefixes the ONNX model spec)");
        }
        if (!ids.insert(m.id).second) {
            cfg_err("duplicate model id '" + m.id + "'");
        }
        m.backend = get_str(e, "backend", get_str(dsec, "backend", "llm"));
        if (m.backend != "llm" && m.backend != "onnx") {
            cfg_err(where + ": 'backend' must be 'llm' or 'onnx'");
        }
        m.binary = get_str(e, "binary",
                           m.is_onnx() ? cfg.binary_onnx : cfg.binary);

        const json * rem = member(e, "remote");
        if (rem) {
            m.remote = true;
            if (!rem->is_object()) {
                cfg_err(where + ": 'remote' must be an object");
            }
            const std::string http = get_str_req(*rem, "http", where + ".remote");
            // The remote worker's HTTP base may carry a scheme; the router
            // only needs host and port for the proxy target.
            std::string h = http;
            const size_t sep = h.find("://");
            if (sep != std::string::npos) {
                h = h.substr(sep + 3);
            }
            split_host_port(h, where + ".remote.http", m.http_host, m.http_port);
        } else {
            m.path = get_str_req(e, "path", where);
            if (!file_readable(m.path)) {
                cfg_err(where + ": the model file '" + m.path +
                        "' is missing or unreadable - the router starts the workers, so it has to be able to see "
                        "the model itself");
            }
            const std::string http = get_str(e, "http", "127.0.0.1:0");
            split_host_port(http, where + ".http", m.http_host, m.http_port);
            if (m.http_port == 0) {
                cfg_err(where + ": 'http' must name a port");
            }
        }

        // The control plane (optional): the router uses it for the model's
        // identity, its device memory and a graceful stop.  Without it the
        // router falls back to the OpenAI API and a signal.
        {
            const std::string control = get_str(e, "control", "");
            if (!control.empty()) {
                split_host_port(control, where + ".control", m.control_host, m.control_port);
            }
            m.control_token = get_str(e, "control_token", "");
            m.control_token_file = get_str(e, "control_token_file", "");
            if (!m.control_token_file.empty()) {
                std::ifstream f(m.control_token_file, std::ios::binary);
                if (!f) {
                    cfg_err(where + ": cannot read the control token file '" + m.control_token_file + "'");
                }
                std::string tok;
                std::getline(f, tok);
                while (!tok.empty() && (tok.back() == '\n' || tok.back() == '\r')) {
                    tok.pop_back();
                }
                m.control_token = tok;
            }
        }

        m.args = get_str_list(e, "args");
        m.env = get_kv_map(e, "env");

        const json * dev = member(e, "device");
        if (dev) {
            if (!dev->is_object()) {
                cfg_err(where + ": 'device' must be an object");
            }
            m.device_group = get_str(*dev, "group", m.device_group);
            m.device_kind = get_str(*dev, "kind", m.device_kind);
            m.device_filter = get_str(*dev, "filter", m.device_filter);
            m.device_count = get_int(*dev, "count", m.device_count);
            if (m.device_count < 1) {
                cfg_err(where + ": 'device.count' must be at least 1");
            }
        }
        if (m.device_kind.empty() && m.is_onnx()) m.device_kind = "gpu";
        if (!m.device_kind.empty() && m.device_kind != "cpu" && m.device_kind != "gpu" &&
            m.device_kind != "auto" && m.device_kind != "host") {
            cfg_err(where + ": 'device.kind' must be cpu, gpu or auto");
        }
        if (m.is_onnx() && m.device_count != 1) {
            cfg_err(where + ": an ONNX model runs on one device (no layer map)");
        }

        const json * mem = member(e, "memory");
        if (mem) {
            if (!mem->is_object()) {
                cfg_err(where + ": 'memory' must be an object");
            }
            m.vram_budget_mb = get_u64(*mem, "vram_budget_mb", m.vram_budget_mb);
            m.kv_cap_mb = get_int(*mem, "kv_cap_mb", m.kv_cap_mb);
            m.max_slots = get_int(*mem, "max_slots", m.max_slots);
            m.pc_vram_mb = get_int(*mem, "prefix_cache_vram_mb", m.pc_vram_mb);
            m.pc_dir = get_str(*mem, "prefix_cache_dir", m.pc_dir);
        }

        const json * onnx = member(e, "onnx");
        if (onnx) {
            if (!onnx->is_object()) {
                cfg_err(where + ": 'onnx' must be an object");
            }
            m.drop_outputs = get_str_list(*onnx, "drop_outputs");
            if (member(*onnx, "dnnl")) {
                m.dnnl = get_bool(*onnx, "dnnl", m.dnnl);
                m.dnnl_set = true;
            }
        }

        // The router owns --model/--host/--port: an operator who sets them
        // in `args` would silently desynchronise the router's idea of the
        // worker's endpoint from the worker's actual one.
        static const char * kReserved[] = {"--model", "--host", "--port", "--control-port",
                                          "--control-host", "--control-token",
                                          "--control-token-file", "--device", "--drop-outputs",
                                          "--dnnl", "--config", "--layer-map"};
        for (const std::string & a : m.args) {
            for (const char * r : kReserved) {
                if (a == r) {
                    cfg_err(where + ": '" + r + "' is set by the router and must not appear in 'args'");
                }
            }
        }
        if (m.max_slots == 0) {
            cfg_err(where + ": 'max_slots' must be positive");
        }
        if (m.kv_cap_mb == 0) {
            cfg_err(where + ": 'kv_cap_mb' must be positive");
        }

        // Port conflicts: two workers on one port would silently shadow each
        // other, and the router's own port must not collide with a worker's.
        if (!m.remote) {
            if (!ports.insert(m.http_port).second) {
                cfg_err(where + ": http port " + std::to_string(m.http_port) + " is already used by another model");
            }
        }
        if (m.http_port == cfg.listen_port) {
            cfg_err(where + ": http port " + std::to_string(m.http_port) + " collides with the router's listen port");
        }
        if (m.control_port == cfg.listen_port) {
            cfg_err(where + ": control port " + std::to_string(m.control_port) +
                    " collides with the router's listen port");
        }
        if (m.control_port > 0 && m.control_port == m.http_port) {
            cfg_err(where + ": the control port and the HTTP port must differ");
        }

        m.argv = build_worker_argv(m);
        cfg.models.push_back(std::move(m));
    }

    // ---- cross-model checks ------------------------------------------------
    // The budget is per *device group*: two models in the same group are
    // assumed to share one card, and Level Zero does not partition VRAM
    // between them (docs/design/16-router.md §9.3), so their budgets have to
    // add up to less than the card.
    if (cfg.device_vram_mb > 0) {
        // A model's budget is its TOTAL over the cards it runs on, so what a
        // card has to carry is its share: budget / device_count.
        std::map<std::string, uint64_t> sum;
        for (const model_config & m : cfg.models) {
            sum[m.device_group] += m.vram_budget_mb / (uint64_t)m.device_count;
        }
        for (const auto & kv : sum) {
            if (kv.second > cfg.device_vram_mb) {
                cfg_err("device group '" + kv.first + "': the VRAM budgets add up to " +
                        std::to_string(kv.second) + " MB, over the declared " +
                        std::to_string(cfg.device_vram_mb) + " MB per device");
            }
        }
    }

    if (cfg.request_timeout_s <= 0 || cfg.startup_timeout_s <= 0 || cfg.health_interval_s <= 0 ||
        cfg.health_failures <= 0 || cfg.max_restarts < 0 || cfg.restart_backoff_s < 0) {
        cfg_err("the timeouts and the retry counts must be positive");
    }
    return cfg;
}

router_config load_config_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("config: cannot open '" + path + "'");
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    try {
        return parse_config(ss.str());
    } catch (const std::exception & ex) {
        throw std::runtime_error(std::string(ex.what()) + " (in " + path + ")");
    }
}

} // namespace router
} // namespace si
