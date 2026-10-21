#include "supervisor.h"

#include "config.h"
#include "exit_codes.h"
#include "httplib.h"
#include "json.hpp"
#include "jsonrpc.h"
#include "registry.h"
#include "rpc_client.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <ios>
#include <map>
#include <memory>
#include <mutex>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>






extern char **environ;

namespace si {
namespace router {

namespace {

std::atomic<long long> g_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// mkdir -p
bool make_dirs(const std::string & path) {
    if (path.empty()) {
        return true;
    }
    std::string cur;
    size_t i = 0;
    if (path[0] == '/') {
        cur = "/";
        i = 1;
    }
    while (i <= path.size()) {
        const size_t slash = path.find('/', i);
        const std::string part = path.substr(i, slash == std::string::npos ? std::string::npos : slash - i);
        if (!part.empty()) {
            if (cur.empty()) {
                cur = part;
            } else if (cur == "/") {
                cur += part;
            } else {
                cur += "/" + part;
            }
            if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
        if (slash == std::string::npos) {
            break;
        }
        i = slash + 1;
    }
    return true;
}

std::string dir_of(const std::string & path) {
    const size_t slash = path.rfind('/');
    if (slash == std::string::npos) {
        return ".";
    }
    if (slash == 0) {
        return "/";
    }
    return path.substr(0, slash);
}

// The oneAPI runtime's library directories.  The worker binary links
// libdnnl.so.3 and the SYCL runtime, and the dynamic loader needs their
// directories; the documented way to provide them is LD_LIBRARY_PATH (the
// AGENTS.md's "Run" section), but a router that starts the workers itself
// can do it for them instead of making the operator export it in every
// shell that launches the router.
//
// Two levels deep: the runtime is at <root>/<version>/lib, but oneDNN lives
// at <root>/dnnl/<version>/lib and the compiler at
// <root>/compiler/<version>/lib.
//
// Only ADDED to the inherited path, never replacing it.
std::string oneapi_runtime_paths() {
    static const std::string cached = []() {
        std::string out;
        const char * roots[] = {"/opt/intel/oneapi", "/opt/intel/oneapi/compiler"};
        auto add_lib = [&](const std::string & lib) {
            struct stat st;
            if (::stat(lib.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
                return;
            }
            if (out.find(lib) != std::string::npos) {
                return;
            }
            if (!out.empty()) {
                out += ":";
            }
            out += lib;
        };
        for (const char * root : roots) {
            DIR * d = ::opendir(root);
            if (!d) {
                continue;
            }
            while (struct dirent * e = ::readdir(d)) {
                if (e->d_name[0] == '.') {
                    continue;
                }
                const std::string ver = std::string(root) + "/" + e->d_name;
                add_lib(ver + "/lib");
                // one level deeper: dnnl/<version>/lib
                DIR * d2 = ::opendir(ver.c_str());
                if (!d2) {
                    continue;
                }
                while (struct dirent * e2 = ::readdir(d2)) {
                    if (e2->d_name[0] == '.') {
                        continue;
                    }
                    add_lib(ver + "/" + e2->d_name + "/lib");
                }
                ::closedir(d2);
            }
            ::closedir(d);
        }
        return out;
    }();
    return cached;
}

// The child's environment: the router's own, plus the configured additions,
// plus SYCL_DEVICE_FILTER when the model is pinned to a card.  Duplicates are
// resolved here, in the parent, so the child only has to execve.
std::vector<std::string> build_child_env(const router_config & cfg, const model_config & m) {
    std::vector<std::string> out;
    for (char **e = environ; e && *e; ++e) {
        out.push_back(*e);
    }
    auto set = [&](const std::string & kv) {
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) {
            return;
        }
        const std::string k = kv.substr(0, eq);
        for (size_t i = 0; i < out.size();) {
            if (out[i].compare(0, k.size(), k) == 0 && out[i].size() > k.size() && out[i][k.size()] == '=') {
                out.erase(out.begin() + (long)i);
                continue;
            }
            ++i;
        }
        out.push_back(kv);
    };
    for (const std::string & kv : cfg.default_env) {
        set(kv);
    }
    for (const std::string & kv : m.env) {
        set(kv);
    }
    if (!m.device_filter.empty()) {
        set("SYCL_DEVICE_FILTER=" + m.device_filter);
    }
    // The oneAPI runtime's libraries, appended to whatever the router
    // inherited (an explicit LD_LIBRARY_PATH in the config wins).
    const std::string runtime = oneapi_runtime_paths();
    if (!runtime.empty()) {
        std::string ld;
        for (const std::string & e : out) {
            if (e.compare(0, 12, "LD_LIBRARY_") == 0 && e.compare(0, 16, "LD_LIBRARY_PATH=") == 0) {
                ld = e.substr(16);
            }
        }
        if (ld.find("/opt/intel/oneapi") == std::string::npos) {
            set("LD_LIBRARY_PATH=" + (ld.empty() ? runtime : ld + ":" + runtime));
        }
    }
    return out;
}

} // namespace

std::string read_log_tail(const std::string & path, size_t max_lines) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return std::string();
    }
    const std::streamsize size = f.tellg();
    if (size <= 0) {
        return std::string();
    }
    // Read at most the last 64 KiB: enough for a stack of error lines
    // without pulling a whole log into memory.
    const std::streamsize want = size < 65536 ? size : 65536;
    f.seekg(size - want);
    std::string buf(want, '\0');
    f.read(&buf[0], want);
    buf.resize((size_t)f.gcount());

    std::vector<std::string> lines;
    std::string cur;
    for (char c : buf) {
        if (c == '\n') {
            lines.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        lines.push_back(cur);
    }
    if (lines.size() > max_lines) {
        lines.erase(lines.begin(), lines.begin() + (long)(lines.size() - max_lines));
    }
    std::string out;
    for (const std::string & l : lines) {
        out += l;
        out += "\n";
    }
    return out;
}

std::string resolve_binary(const std::string & name) {
    if (name.empty()) {
        return name;
    }
    if (name.find('/') != std::string::npos) {
        return name;
    }
    // Next to the router's own binary: the container layout puts both there.
    char self[4096];
    const ssize_t n = ::readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n > 0) {
        self[n] = '\0';
        const std::string cand = dir_of(self) + "/" + name;
        if (::access(cand.c_str(), X_OK) == 0) {
            return cand;
        }
    }
    const char * path = ::getenv("PATH");
    if (path) {
        std::string p(path);
        size_t i = 0;
        while (i <= p.size()) {
            const size_t colon = p.find(':', i);
            const std::string dir = p.substr(i, colon == std::string::npos ? std::string::npos : colon - i);
            if (!dir.empty()) {
                const std::string cand = dir + "/" + name;
                if (::access(cand.c_str(), X_OK) == 0) {
                    return cand;
                }
            }
            if (colon == std::string::npos) {
                break;
            }
            i = colon + 1;
        }
    }
    return name;
}

supervisor::supervisor(registry & reg, const router_config & cfg) : reg_(reg), cfg_(cfg) {
}

supervisor::~supervisor() {
    stop();
}

void supervisor::start() {
    if (!cfg_.log_dir.empty()) {
        make_dirs(cfg_.log_dir);
    }
    // Number the models within their device group, in configuration order:
    // that is the order the group gate releases them in.
    std::map<std::string, int> next_index;
    for (const auto & slot : reg_.all()) {
        if (slot->cfg.remote) {
            continue; // a remote worker does not use this host's memory
        }
        slot->group_index = next_index[slot->cfg.device_group]++;
    }
    for (const auto & slot : reg_.all()) {
        threads_.emplace_back([this, slot]() { run(slot); });
    }
}

void supervisor::stop() {
    if (stopping_.exchange(true)) {
        return;
    }
    // Ask every worker to go away first, so the prefix cache is flushed on the
    // worker's own shutdown path (the router must not SIGKILL a worker that
    // still has VRAM checkpoints to write).  The control plane's `shutdown`
    // takes the same path as a SIGTERM inside the worker; a worker without a
    // control plane gets the signal.
    {
        std::lock_guard<std::mutex> lk(clients_mtx_);
        for (const auto & kv : clients_) {
            if (!kv.second->connected()) {
                continue;
            }
            nlohmann::json result;
            std::string err;
            nlohmann::json params;
            params["timeout_s"] = 30;
            const bool ok = kv.second->call(rpc::kShutdown, params, result, err, 5000);
            fprintf(stderr, "[router] the control plane of '%s': shutdown %s%s%s\n", kv.first.c_str(),
                    ok ? "accepted" : "FAILED", ok ? "" : " - ", ok ? "" : err.c_str());
        }
    }
    for (const auto & slot : reg_.all()) {
        pid_t pid = -1;
        {
            std::lock_guard<std::mutex> lk(slot->mtx);
            pid = (pid_t)slot->pid;
            slot->state = model_state::stopping;
        }
        if (pid > 0) {
            ::kill(pid, SIGTERM);
            fprintf(stderr, "[router] SIGTERM -> the worker of '%s' (pid %d)\n", slot->cfg.id.c_str(), (int)pid);
        }
    }
    // The grace period belongs to the supervisor's own threads: each of them
    // waits for (and reaps) its child, so the router must not waitpid on the
    // same pid - two waitpids on one pid is a race, and the loser takes
    // ECHILD for "the child is gone" and kills a worker that was still
    // flushing its prefix cache.
    for (std::thread & t : threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    threads_.clear();
}

bool supervisor::spawn(const model_slot & slot, pid_t & pid, std::string & err) {
    const model_config & m = slot.cfg;
    const std::string binary = resolve_binary(m.binary.empty() ? std::string("sycl-infer") : m.binary);
    if (::access(binary.c_str(), X_OK) != 0) {
        err = "worker binary '" + binary + "' is not executable (looked next to the router and in PATH)";
        return false;
    }

    const std::string log_path =
        (cfg_.log_dir.empty() ? std::string(".") : cfg_.log_dir) + "/" + m.id + ".log";
    if (!cfg_.log_dir.empty() && !make_dirs(cfg_.log_dir)) {
        err = "cannot create the log directory '" + cfg_.log_dir + "'";
        return false;
    }
    const int fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) {
        err = "cannot open the log file '" + log_path + "': " + std::strerror(errno);
        return false;
    }

    // Everything the child touches between fork() and execve() must already
    // exist: after fork() in a threaded process only async-signal-safe calls
    // are legal, so the argv/envp vectors are built here, in the parent.
    //
    // argv[0] is the BINARY, not the first flag: the worker's own argument
    // loop starts at argv[1], and the dynamic loader prints argv[0] in its
    // "error while loading shared libraries" message - so a missing argv[0]
    // turns into a startup error that names "--model".
    std::vector<std::string> argv_store;
    argv_store.push_back(binary);
    for (const std::string & a : m.argv) {
        argv_store.push_back(a);
    }
    std::vector<char *> cargv;
    cargv.reserve(argv_store.size() + 1);
    for (std::string & s : argv_store) {
        cargv.push_back(const_cast<char *>(s.c_str()));
    }
    cargv.push_back(nullptr);

    const std::vector<std::string> env_store = build_child_env(cfg_, m);
    std::vector<char *> cenv;
    cenv.reserve(env_store.size() + 1);
    for (const std::string & s : env_store) {
        cenv.push_back(const_cast<char *>(s.c_str()));
    }
    cenv.push_back(nullptr);

    const pid_t p = ::fork();
    if (p < 0) {
        ::close(fd);
        err = std::string("fork failed: ") + std::strerror(errno);
        return false;
    }
    if (p == 0) {
        // child
        ::setsid();
        ::dup2(fd, STDOUT_FILENO);
        ::dup2(fd, STDERR_FILENO);
        if (fd > STDERR_FILENO) {
            ::close(fd);
        }
        ::execve(binary.c_str(), cargv.data(), cenv.data());
        const char msg[] = "router: exec failed\n";
        const ssize_t ignored = ::write(STDERR_FILENO, msg, sizeof(msg) - 1);
        (void)ignored;
        ::_exit(127);
    }
    ::close(fd);
    pid = p;
    return true;
}

namespace {

// Is the worker serving?  The port answering /health means the model is
// loaded: the worker listens only after the engine is constructed.
bool probe_health(const std::string & host, int port, int timeout_s) {
    httplib::Client cli(host, port);
    cli.set_read_timeout(timeout_s, 0);
    cli.set_write_timeout(timeout_s, 0);
    cli.set_connection_timeout(std::min(timeout_s, 2), 0);
    auto res = cli.Get("/health");
    return res && res->status == 200;
}

std::string fetch_advertised(const std::string & host, int port, std::string & id, int timeout_s) {
    httplib::Client cli(host, port);
    cli.set_read_timeout(timeout_s, 0);
    cli.set_write_timeout(timeout_s, 0);
    cli.set_connection_timeout(std::min(timeout_s, 2), 0);
    auto res = cli.Get("/v1/models");
    if (!res || res->status != 200) {
        return std::string();
    }
    // The worker answers with {"object":"list","data":[{...}]}.  Keep the
    // entry verbatim so the router can re-serve it without reformatting.
    const size_t k = res->body.find("\"data\"");
    if (k == std::string::npos) {
        return std::string();
    }
    const size_t open = res->body.find('[', k);
    const size_t close = res->body.find(']', open);
    if (open == std::string::npos || close == std::string::npos || close <= open) {
        return std::string();
    }
    const std::string arr = res->body.substr(open + 1, close - open - 1);
    const size_t b = arr.find('{');
    const size_t e = arr.rfind('}');
    if (b == std::string::npos || e == std::string::npos || e <= b) {
        return std::string();
    }
    const std::string entry = arr.substr(b, e - b + 1);
    const size_t idk = entry.find("\"id\"");
    if (idk != std::string::npos) {
        const size_t q1 = entry.find('"', entry.find(':', idk) + 1);
        if (q1 != std::string::npos) {
            const size_t q2 = entry.find('"', q1 + 1);
            if (q2 != std::string::npos) {
                id = entry.substr(q1 + 1, q2 - q1 - 1);
            }
        }
    }
    return entry;
}

} // namespace

bool supervisor::wait_ready(const std::shared_ptr<model_slot> & slot, pid_t pid, std::string & err) {
    const long long deadline = g_now_ms() + (long long)cfg_.startup_timeout_s * 1000;
    for (;;) {
        if (stopping_.load()) {
            err = "the router is shutting down";
            return false;
        }
        if (pid > 0) {
            int st = 0;
            const pid_t r = ::waitpid(pid, &st, WNOHANG);
            if (r == pid) {
                // The child died before it ever served.  Reap it and report.
                const std::string tail = read_log_tail((cfg_.log_dir.empty() ? "." : cfg_.log_dir) + "/" +
                                                          slot->cfg.id + ".log",
                                                      8);
                err = "the worker exited during startup";
                if (!tail.empty()) {
                    err += ": " + tail;
                }
                slot->set_state(model_state::failed, err);
                return false;
            }
        }
        if (probe_health(slot->cfg.http_host, slot->cfg.http_port, 2)) {
            return true;
        }
        if (g_now_ms() >= deadline) {
            err = "the worker did not become ready within " + std::to_string(cfg_.startup_timeout_s) +
                  "s (the first start also JITs the device image; an AOT build skips that)";
            return false;
        }
        sleep_ms(200);
    }
}

void supervisor::group_enter(const std::string & group, int index, bool restart) {
    std::shared_ptr<group_gate> g;
    {
        std::lock_guard<std::mutex> lk(groups_mtx_);
        auto it = groups_.find(group);
        if (it == groups_.end()) {
            it = groups_.emplace(group, std::make_shared<group_gate>()).first;
        }
        g = it->second;
    }
    std::unique_lock<std::mutex> lk(g->mtx);
    // Wait for this model's turn: the group's models start in configuration
    // order, one at a time.  (A plain "wait until free" would serialise them
    // but leave the ORDER to a race between the threads, which is exactly the
    // kind of non-determinism a deployment must not have.)  A restart has
    // already had its turn, so it only waits for the group to be free.
    while ((g->busy || (!restart && g->next != index)) && !stopping_.load()) {
        g->cv.wait_for(lk, std::chrono::milliseconds(50));
    }
    g->busy = true;
}

void supervisor::group_leave(const std::string & group, int index, bool restart) {
    std::shared_ptr<group_gate> g;
    {
        std::lock_guard<std::mutex> lk(groups_mtx_);
        auto it = groups_.find(group);
        if (it == groups_.end()) {
            return;
        }
        g = it->second;
    }
    {
        std::lock_guard<std::mutex> lk(g->mtx);
        g->busy = false;
        if (!restart) {
            g->next = index + 1;
        }
    }
    g->cv.notify_all();
}

std::shared_ptr<rpc_client> supervisor::control_client(const model_slot & slot) {
    std::lock_guard<std::mutex> lk(clients_mtx_);
    auto it = clients_.find(slot.cfg.id);
    if (it != clients_.end()) {
        return it->second;
    }
    auto cli = std::make_shared<rpc_client>(slot.cfg.control_host, slot.cfg.control_port, slot.cfg.control_token);
    clients_[slot.cfg.id] = cli;
    return cli;
}

// Read the worker's own report over the control plane.  `first` is the
// handshake (`hello`), which also carries the protocol version; the periodic
// refresh asks for `stats`.
bool supervisor::refresh_control(model_slot & slot, bool first, std::string & err) {
    if (slot.cfg.control_port <= 0) {
        err = "no control endpoint";
        return false;
    }
    auto cli = control_client(slot);
    nlohmann::json params;
    nlohmann::json result;
    // Retry a couple of times: a transient failure here is not fatal, but
    // LOSING the device report is - without it the router has no measured
    // free memory, so the next model of the group would be admitted against
    // the static budget instead (i.e. it would start and then die with
    // "device alloc failed").
    for (int attempt = 0; attempt < 3; attempt++) {
        if (!cli->connected()) {
            std::string cerr;
            if (!cli->connect(2000, cerr)) {
                err = cerr;
                sleep_ms(200);
                continue;
            }
        }
        if (cli->call(first ? rpc::kHello : rpc::kStats, params, result, err, 5000)) {
            break;
        }
        sleep_ms(200);
    }
    if (result.is_null()) {
        return false;
    }
    const nlohmann::json report = first ? (result.contains("worker") ? result["worker"] : nlohmann::json::object())
                                        : result;
    if (report.is_object()) {
        // The model's identity (what the OpenAI API advertises).
        if (report.contains("model") && report["model"].is_object()) {
            const nlohmann::json & mdl = report["model"];
            std::lock_guard<std::mutex> lk(slot.mtx);
            if (mdl.contains("id") && mdl["id"].is_string()) {
                slot.advertised_id = mdl["id"].get<std::string>();
            }
            slot.advertised_json = report["model"].dump();
        }
        ledger_update(slot, report);
    }
    return true;
}

// Fold a worker's device report into the group's ledger: how much the
// group's cards have left is what the NEXT model of the group has to fit
// into (docs/design/16-router.md §9.1, layer ②).
void supervisor::ledger_update(const model_slot & slot, const nlohmann::json & report) {
    if (!report.contains("devices") || !report["devices"].is_array()) {
        return;
    }
    uint64_t free_sum = 0;
    uint64_t total_sum = 0;
    bool known = false;
    for (const nlohmann::json & d : report["devices"]) {
        const uint64_t f = d.value("free_vram_bytes", (uint64_t)0);
        const uint64_t t = d.value("total_vram_bytes", (uint64_t)0);
        total_sum += t;
        if (f > 0) {
            free_sum += f;
            known = true;
        }
    }
    std::lock_guard<std::mutex> lk(ledger_mtx_);
    ledger & g = ledgers_[slot.cfg.device_group];
    g.total_vram_bytes = total_sum;
    if (known) {
        g.free_vram_bytes = free_sum;
        g.measured = true;
    }
}

// Give a model's budget back: it is not resident any more, so it must not
// keep blocking the next model of the group.
void supervisor::ledger_release(const model_slot & slot) {
    std::lock_guard<std::mutex> lk(ledger_mtx_);
    auto it = ledgers_.find(slot.cfg.device_group);
    if (it == ledgers_.end()) {
        return;
    }
    const uint64_t share = (slot.cfg.vram_budget_mb / (uint64_t)slot.cfg.device_count) * 1024ull * 1024ull;
    it->second.committed_share_bytes =
        it->second.committed_share_bytes > share ? it->second.committed_share_bytes - share : 0;
}

// Does the model still fit on the cards its group has left?  The budget is
// the model's total across its devices, and the group's free memory is the
// sum over the devices the group reported - so a multi-card model is
// compared against the cards' combined headroom.
bool supervisor::vram_admit(const model_slot & slot, std::string & why) {
    const uint64_t budget = slot.cfg.vram_budget_mb * 1024ull * 1024ull;
    if (budget == 0) {
        return true; // no budget declared: nothing to enforce
    }
    uint64_t free_bytes = 0;
    uint64_t committed_share = 0;
    bool measured = false;
    {
        std::lock_guard<std::mutex> lk(ledger_mtx_);
        auto it = ledgers_.find(slot.cfg.device_group);
        if (it != ledgers_.end()) {
            free_bytes = it->second.free_vram_bytes;
            committed_share = it->second.committed_share_bytes;
            measured = it->second.measured;
        }
    }
    // What ONE card of the group has to carry: the model's share of its
    // budget (its total over the cards it runs on).
    const uint64_t share = (slot.cfg.vram_budget_mb / (uint64_t)slot.cfg.device_count) * 1024ull * 1024ull;
    if (!measured) {
        // No card has reported yet (this is the first model of the group):
        // fall back to the declared capacity, if the operator gave one.
        const uint64_t cap = cfg_.device_vram_mb * 1024ull * 1024ull;
        if (cap > 0 && committed_share + share > cap) {
            why = "the device group '" + slot.cfg.device_group + "' has " + std::to_string(cap / (1024 * 1024)) +
                  " MB declared per card, and this model needs " + std::to_string(slot.cfg.vram_budget_mb / (uint64_t)slot.cfg.device_count) +
                  " MB on a card on top of " + std::to_string(committed_share / (1024 * 1024)) + " MB already committed";
            return false;
        }
        return true;
    }
    // The measured case: the card's free memory ALREADY excludes the models
    // that are resident, so this model's budget is compared against it
    // directly - adding the committed total again would count the resident
    // models twice.
    if (budget > free_bytes) {
        why = "the device group '" + slot.cfg.device_group + "' has " +
              std::to_string(free_bytes / (1024 * 1024)) + " MB free, and this model needs " +
              std::to_string(slot.cfg.vram_budget_mb) + " MB";
        return false;
    }
    return true;
}

void supervisor::run(std::shared_ptr<model_slot> slot) {
    const model_config & m = slot->cfg;
    const std::string log_path = (cfg_.log_dir.empty() ? "." : cfg_.log_dir) + "/" + m.id + ".log";
    int attempt = 0;

    for (;;) {
        if (stopping_.load()) {
            return;
        }
        slot->set_state(model_state::starting);

        pid_t pid = -1;
        std::string err;
        // The first pass through the loop is the model's FIRST start; the
        // later ones are restarts, which do not take a place in the group's
        // order again.
        const bool restart = (attempt > 0);
        if (!m.remote) {
            // One model at a time per device group: two models loading at the
            // same time would compete for the same card's memory, and the
            // loser would die with "device alloc failed".
            group_enter(m.device_group, slot->group_index, restart);
            if (stopping_.load()) {
                group_leave(m.device_group, slot->group_index, restart);
                return;
            }
            // The empirical VRAM admission: the model has to fit on the cards
            // its group has left (docs/design/16-router.md §9.1).
            {
                std::string why;
                if (!vram_admit(*slot, why)) {
                    group_leave(m.device_group, slot->group_index, restart);
                    slot->set_state(model_state::failed, "the VRAM budget refused the model: " + why);
                    fprintf(stderr, "[router] model '%s' was not started: %s\n", m.id.c_str(), why.c_str());
                    fflush(stderr);
                    return;
                }
            }
            if (!restart) {
                std::lock_guard<std::mutex> lk(ledger_mtx_);
                ledgers_[m.device_group].committed_share_bytes +=
                    (m.vram_budget_mb / (uint64_t)m.device_count) * 1024ull * 1024ull;
            }
            if (!spawn(*slot, pid, err)) {
                group_leave(m.device_group, slot->group_index, restart);
                ledger_release(*slot);
                slot->set_state(model_state::failed, err);
                // A spawn failure is an environment problem, not a crash:
                // there is nothing to retry.
                return;
            }
            {
                std::lock_guard<std::mutex> lk(slot->mtx);
                slot->pid = pid;
                slot->state = model_state::loading;
            }
            if (!wait_ready(slot, pid, err)) {
                group_leave(m.device_group, slot->group_index, restart);
                if (stopping_.load()) {
                    return;
                }
                // A startup death is a configuration or environment problem
                // (the model does not load, the card is out of memory), so it
                // is not retried: the same attempt would fail the same way.
                if (slot->error().empty() || slot->error() != err) {
                    slot->set_state(model_state::failed, err);
                }
                ledger_release(*slot);
                // The child may already have been reaped by wait_ready(), in
                // which case its pid could have been recycled by an unrelated
                // process: only signal it if it is still running.
                {
                    int st = 0;
                    const pid_t r = ::waitpid(pid, &st, WNOHANG);
                    if (r == 0) {
                        ::kill(pid, SIGKILL);
                        ::waitpid(pid, &st, 0);
                    }
                }
                {
                    std::lock_guard<std::mutex> lk(slot->mtx);
                    slot->pid = -1;
                }
                return;
            }
            // The model's identity and its footprint on the card.  This has
            // to happen BEFORE the gate is released: the next model of the
            // group is admitted against what this one leaves behind, so the
            // ledger must already know it.
            {
                std::lock_guard<std::mutex> lk(slot->mtx);
                slot->state = model_state::up;
                slot->ready_at = (long long)time(nullptr);
                slot->last_error.clear();
            }
            std::string cerr;
            if (m.control_port > 0) {
                if (!refresh_control(*slot, true, cerr)) {
                    fprintf(stderr, "[router] model '%s': the control plane is not usable (%s)\n", m.id.c_str(),
                            cerr.c_str());
                }
            } else {
                std::string id;
                const std::string entry = fetch_advertised(m.http_host, m.http_port, id, 5);
                std::lock_guard<std::mutex> lk(slot->mtx);
                if (!id.empty()) {
                    slot->advertised_id = id;
                }
                if (!entry.empty()) {
                    slot->advertised_json = entry;
                }
            }
            // The card now holds this model; the next model in the group may
            // start.
            group_leave(m.device_group, slot->group_index, restart);
        } else {
            // A remote worker: there is no process to spawn, only an endpoint
            // to wait for.
            const long long deadline = g_now_ms() + (long long)cfg_.startup_timeout_s * 1000;
            bool ready = false;
            while (g_now_ms() < deadline && !stopping_.load()) {
                if (probe_health(m.http_host, m.http_port, 2)) {
                    ready = true;
                    break;
                }
                sleep_ms(200);
            }
            if (!ready) {
                if (stopping_.load()) {
                    return;
                }
                slot->set_state(model_state::failed, "the remote worker did not become ready");
                return;
            }
        }

        // Ready.  (The state and the identity were published above, in the
        // branch that got here.)  Now watch it.
        fprintf(stderr, "[router] model '%s' is up (%s)\n", m.id.c_str(), slot->http_base().c_str());
        fflush(stderr);

        if (m.remote) {
            // Remote: there is no child to wait for, so the watch loop is a
            // liveness poll.  A failure degrades the model and retries.
            {
                std::lock_guard<std::mutex> lk(slot->mtx);
                slot->state = model_state::up;
                slot->ready_at = (long long)time(nullptr);
                slot->last_error.clear();
            }
            std::string cerr;
            if (m.control_port > 0) {
                if (!refresh_control(*slot, true, cerr)) {
                    fprintf(stderr, "[router] model '%s': the control plane is not usable (%s)\n", m.id.c_str(),
                            cerr.c_str());
                }
            } else {
                std::string id;
                const std::string entry = fetch_advertised(m.http_host, m.http_port, id, 5);
                std::lock_guard<std::mutex> lk(slot->mtx);
                if (!id.empty()) {
                    slot->advertised_id = id;
                }
                if (!entry.empty()) {
                    slot->advertised_json = entry;
                }
            }
            int fails = 0;
            while (!stopping_.load()) {
                sleep_ms((unsigned)cfg_.health_interval_s * 1000);
                if (stopping_.load()) {
                    break;
                }
                if (m.control_port > 0) {
                    // The control plane carries the liveness check and the
                    // device report in one round trip.
                    std::string cerr;
                    if (refresh_control(*slot, false, cerr)) {
                        fails = 0;
                        continue;
                    }
                } else if (probe_health(m.http_host, m.http_port, 2)) {
                    fails = 0;
                    continue;
                }
                if (++fails < cfg_.health_failures) {
                    continue;
                }
                slot->set_state(model_state::degraded, "the remote worker stopped answering /health");
                break;
            }
            if (stopping_.load()) {
                return;
            }
        } else {
            // Local: watch the child.  A crash is retryable (up to the
            // restart budget); a hang is caught by the liveness probe.
            int fails = 0;
            bool hung = false;
            long long next_probe = g_now_ms() + (long long)cfg_.health_interval_s * 1000;
            int status = 0;
            while (!stopping_.load()) {
                const pid_t r = ::waitpid(pid, &status, WNOHANG);
                if (r == pid) {
                    break;
                }
                if (r < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    break;
                }
                if (g_now_ms() >= next_probe) {
                    next_probe = g_now_ms() + (long long)cfg_.health_interval_s * 1000;
                    if (m.control_port > 0) {
                        // The control plane carries the liveness check AND
                        // the device report in one round trip.
                        std::string cerr;
                        if (refresh_control(*slot, false, cerr)) {
                            fails = 0;
                        } else if (++fails >= cfg_.health_failures) {
                            hung = true;
                            fprintf(stderr, "[router] the worker of '%s' is not answering (%d failures) -> replacing it\n",
                                    slot->cfg.id.c_str(), fails);
                            ::kill(pid, SIGTERM);
                            sleep_ms(500);
                            ::kill(pid, SIGKILL);
                            int st = 0;
                            ::waitpid(pid, &st, 0);
                            status = st;
                            break;
                        }
                    } else if (probe_health(m.http_host, m.http_port, 2)) {
                        fails = 0;
                    } else if (++fails >= cfg_.health_failures) {
                        // The worker is alive but not answering: treat it as
                        // hung and replace it.
                        hung = true;
                        ::kill(pid, SIGTERM);
                        sleep_ms(500);
                        ::kill(pid, SIGKILL);
                        int st = 0;
                        ::waitpid(pid, &st, 0);
                        status = st;
                        break;
                    }
                }
                sleep_ms(100);
            }
            if (stopping_.load()) {
                // The router asked us to stop.  Wait for the worker to finish
                // ITS OWN shutdown (the prefix cache flush) and only insist
                // after the grace period.
                //
                // This thread owns the child: it is the one that waits for it,
                // so it is the one that reaps it.  Two waitpids on one pid
                // (this thread and the router's stop path) is a race - the
                // router's would see ECHILD, take it for "the child is gone"
                // and SIGKILL a worker that was in the middle of flushing.
                if (pid > 0) {
                    const long long deadline = g_now_ms() + 10000;
                    int st = 0;
                    for (;;) {
                        const pid_t r = ::waitpid(pid, &st, WNOHANG);
                        if (r == pid) {
                            break; // it exited
                        }
                        if (r < 0) {
                            break; // already reaped
                        }
                        if (g_now_ms() >= deadline) {
                            fprintf(stderr, "[router] the worker of '%s' did not exit in 10s -> SIGKILL\n",
                                    slot->cfg.id.c_str());
                            ::kill(pid, SIGKILL);
                            ::waitpid(pid, &st, 0);
                            break;
                        }
                        sleep_ms(100);
                    }
                }
                return;
            }
            const std::string tail = read_log_tail(log_path, 8);
            std::string reason;
            bool startup_failure = false;
            if (hung) {
                reason = "the worker stopped answering /health";
            } else if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                reason = "the worker exited cleanly";
            } else if (WIFEXITED(status)) {
                reason = "the worker exited with status " + std::to_string(WEXITSTATUS(status));
                // The worker's own startup-failure code: it says so itself, so
                // the router does not have to guess from the log text.
                startup_failure = WEXITSTATUS(status) == kExitStartupFailure;
            } else if (WIFSIGNALED(status)) {
                reason = "the worker was killed by signal " + std::to_string(WTERMSIG(status));
            } else {
                reason = "the worker stopped";
            }
            if (!tail.empty()) {
                reason += ": " + tail;
            }
            {
                std::lock_guard<std::mutex> lk(slot->mtx);
                slot->pid = -1;
            }
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0 && !hung) {
                // A clean exit is somebody's decision, not a failure.
                slot->set_state(model_state::failed, reason);
                ledger_release(*slot);
                return;
            }
            if (startup_failure) {
                // The worker says it could not start: the same attempt would
                // fail the same way, so it is not retried.
                slot->set_state(model_state::failed,
                               reason + " (a startup failure - the same attempt would fail the same way)");
                ledger_release(*slot);
                return;
            }
            attempt++;
            if (attempt > cfg_.max_restarts) {
                slot->set_state(model_state::failed,
                               reason + " (restart budget of " + std::to_string(cfg_.max_restarts) + " exhausted)");
                ledger_release(*slot);
                return;
            }
            slot->set_state(model_state::restarting, reason);
            {
                std::lock_guard<std::mutex> lk(slot->mtx);
                slot->restarts = attempt;
            }
            int backoff = cfg_.restart_backoff_s;
            for (int i = 1; i < attempt && backoff < 60; i++) {
                backoff *= 2;
            }
            fprintf(stderr, "[router] model '%s' died (%s); restarting in %ds\n", m.id.c_str(), reason.c_str(),
                    backoff);
            fflush(stderr);
            for (int waited = 0; waited < backoff * 10 && !stopping_.load(); waited++) {
                sleep_ms(100);
            }
        }
    }
}

} // namespace router
} // namespace si
