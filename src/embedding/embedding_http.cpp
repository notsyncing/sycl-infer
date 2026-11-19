#include "embedding_http.h"

#include <cctype>
#include <csignal>
#include <iostream>
#include <mutex>
#include <vector>

#include "httplib.h"
#include "json.hpp"

namespace si {
namespace embedding {
namespace {

using json = nlohmann::json;

void set_json(httplib::Response & res, int code, const json & j) {
    res.status = code;
    res.set_content(j.dump(), "application/json");
}

bool accepted_model(const json & body, const std::string & id) {
    if (!body.contains("model") || !body["model"].is_string()) return true;
    return body["model"].get<std::string>() == id || body["model"].get<std::string>() == "sycl-infer";
}

bool parse_inputs(const json & body, std::vector<std::string> & texts, std::string & err) {
    if (!body.contains("input")) {
        err = "missing 'input'";
        return false;
    }
    const json & in = body["input"];
    if (in.is_string()) {
        texts.push_back(in.get<std::string>());
    } else if (in.is_array()) {
        for (const auto & v : in) {
            if (!v.is_string()) {
                err = "'input' array elements must be strings";
                return false;
            }
            texts.push_back(v.get<std::string>());
        }
    } else if (in.is_number_integer()) {
        texts.push_back(std::to_string(in.get<int64_t>()));
    } else {
        err = "'input' must be a string or an array of strings";
        return false;
    }
    if (texts.empty()) {
        err = "'input' is empty";
        return false;
    }
    return true;
}

} // namespace

int serve(engine & e, const server_config & cfg) {
    httplib::Server srv;
    const auto cors = [](httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    };
    srv.Options(".*", [&](const httplib::Request &, httplib::Response & res) {
        cors(res);
        res.status = 204;
    });
    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        set_json(res, 200, {{"status", "ok"}});
    });
    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        cors(res);
        set_json(res, 200, {{"object", "list"},
                            {"data", json::array({{{"id", cfg.model_id},
                                                   {"object", "model"},
                                                   {"owned_by", "sycl-infer"}}})}});
    });

    srv.Post("/v1/embeddings", [&](const httplib::Request & req, httplib::Response & res) {
        cors(res);
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & ex) {
            set_json(res, 400, {{"error", {{"message", std::string("invalid JSON: ") + ex.what()},
                                           {"type", "invalid_request_error"}}}});
            return;
        }
        if (!accepted_model(body, cfg.model_id)) {
            set_json(res, 404, {{"error", {{"message", "model not found"},
                                           {"type", "invalid_request_error"}}}});
            return;
        }
        std::vector<std::string> texts;
        std::string err;
        if (!parse_inputs(body, texts, err)) {
            set_json(res, 400, {{"error", {{"message", err}, {"type", "invalid_request_error"}}}});
            return;
        }
        if (body.contains("encoding_format") && body["encoding_format"].is_string() &&
            body["encoding_format"].get<std::string>() != "float") {
            set_json(res, 400, {{"error", {{"message", "only encoding_format=float is supported"},
                                           {"type", "invalid_request_error"}}}});
            return;
        }
        try {
            const auto vecs = e.embed(texts);
            json data = json::array();
            size_t prompt_tokens = 0;
            const auto & tk = e.tokenizer();
            for (size_t i = 0; i < vecs.size(); ++i) {
                prompt_tokens += tk.encode(texts[i]).size();
                data.push_back({{"object", "embedding"},
                                {"index", (int)i},
                                {"embedding", vecs[i]}});
            }
            set_json(res, 200, {{"object", "list"},
                                {"data", data},
                                {"model", cfg.model_id},
                                {"usage", {{"prompt_tokens", (int)prompt_tokens},
                                           {"total_tokens", (int)prompt_tokens}}}});
        } catch (const std::exception & ex) {
            set_json(res, 500, {{"error", {{"message", ex.what()}, {"type", "server_error"}}}});
        }
    });

    std::cout << "[embed] listening on " << cfg.host << ":" << cfg.port
              << " model=" << cfg.model_id << " device=" << e.device() << std::endl;
    srv.listen(cfg.host.c_str(), cfg.port);
    return 0;
}

} // namespace embedding
} // namespace si
