#include "config.h"
#include "proxy.h"
#include "registry.h"
#include "supervisor.h"

#include "httplib.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <exception>
#include <string>
#include <thread>



using namespace si::router;

namespace {

std::atomic<bool> g_term{false};
std::atomic<int> g_term_signal{0};

void on_term(int sig) {
    g_term.store(true);
    g_term_signal.store(sig);
}

void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s --config <router.json> [--check]\n"
            "\n"
            "  --config <path>   the router configuration (JSON, comments allowed)\n"
            "  --check           validate the configuration and exit\n"
            "\n"
            "The configuration lists every model: its GGUF path, its HTTP port,\n"
            "its memory budget and the extra worker arguments.  The router\n"
            "starts one `sycl-infer serve` process per model and proxies the\n"
            "OpenAI API to them.  See docs/design/16-router.md.\n"
            "\n"
            "A model entry:\n"
            "  id                  the name clients use in the `model` field\n"
            "  path                the GGUF (the router has to be able to see it:\n"
            "                      it starts the workers)\n"
            "  http                the worker's OpenAI port (host:port)\n"
            "  control             the worker's control plane (host:port): its\n"
            "                      identity, its device memory, a graceful stop\n"
            "  control_token[_file]  the control plane's shared secret\n"
            "  device              {group, filter, count}: the budget group, the\n"
            "                      SYCL_DEVICE_FILTER for the worker, how many\n"
            "                      cards it runs on\n"
            "  memory              {vram_budget_mb, kv_cap_mb, max_slots,\n"
            "                      prefix_cache_vram_mb, prefix_cache_dir}\n"
            "  args                extra worker arguments\n"
            "  remote              {http}: a worker on another machine (the\n"
            "                      router connects instead of starting it)\n"
            "\n"
            "The router's HTTP face: /v1/chat/completions, /v1/completions,\n"
            "/v1/models, /health and /router/status.  It has no authentication\n"
            "of its own unless `router.auth.bearer_tokens` is set, so bind it\n"
            "to the loopback and put a reverse proxy in front of it.\n",
            prog);
}

} // namespace

int main(int argc, char ** argv) {
    std::string config_path;
    bool check_only = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (a == "--check") {
            check_only = true;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            usage(argv[0]);
            return 1;
        }
    }
    if (config_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    router_config cfg;
    try {
        cfg = load_config_file(config_path);
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }

    if (check_only) {
        printf("config ok: %zu model(s)\n", cfg.models.size());
        for (const model_config & m : cfg.models) {
            printf("  %-24s %-28s budget=%llu MB group=%s%s\n", m.id.c_str(), m.http_base().c_str(),
                   (unsigned long long)m.vram_budget_mb, m.device_group.c_str(), m.remote ? " (remote)" : "");
        }
        return 0;
    }

    registry reg;
    for (const model_config & m : cfg.models) {
        reg.add(m);
    }

    fprintf(stderr, "[router] listening on %s:%d with %zu model(s)\n", cfg.listen_host.c_str(), cfg.listen_port,
            cfg.models.size());
    for (const auto & slot : reg.all()) {
        fprintf(stderr, "[router]   %-24s %-28s budget=%llu MB%s\n", slot->cfg.id.c_str(),
                slot->http_base().c_str(), (unsigned long long)slot->cfg.vram_budget_mb,
                slot->cfg.remote ? " (remote: connect only)" : "");
    }
    fflush(stderr);

    std::signal(SIGINT, on_term);
    std::signal(SIGTERM, on_term);

    httplib::Server srv;
    srv.set_read_timeout(cfg.request_timeout_s + 60, 0);
    srv.set_write_timeout(cfg.request_timeout_s + 60, 0);
    // The payload limit matches the worker's: a base64 image is 64 MiB.
    srv.set_payload_max_length(64 * 1024 * 1024);

    proxy px(reg, cfg);
    px.install(srv);

    supervisor sup(reg, cfg);
    sup.start();

    std::thread watchdog([&]() {
        while (!g_term.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        srv.stop();
    });

    const bool listening = srv.listen(cfg.listen_host.c_str(), cfg.listen_port);
    g_term.store(true);
    watchdog.join();
    if (!listening) {
        fprintf(stderr, "[router] failed to listen on %s:%d\n", cfg.listen_host.c_str(), cfg.listen_port);
    }
    fprintf(stderr, "[router] shutting down (signal %d)\n", g_term_signal.load());
    // Stop the workers last: they flush their prefix cache on the way out.
    sup.stop();
    return listening ? 0 : 1;
}
