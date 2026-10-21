#include "control.h"

#include "jsonrpc.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <memory>
#include <unistd.h>

namespace si {

json control_report_to_json(const control_report & r) {
    json j;
    json devs = json::array();
    for (const control_report::device_t & d : r.devices) {
        json dj;
        dj["index"] = d.index;
        dj["name"] = d.name;
        dj["total_vram_bytes"] = d.total_vram_bytes;
        dj["free_vram_bytes"] = d.free_vram_bytes;
        // The driver's own figure, kept separate: on the Level Zero backend it
        // does NOT count the SYCL allocations (it reported 13 GB free on a
        // card holding 9 GB of weights), so it must not drive the admission.
        dj["driver_free_vram_bytes"] = d.driver_free_vram_bytes;
        dj["used_bytes"] = d.used_bytes;
        dj["kv_bytes"] = d.kv_bytes;
        devs.push_back(dj);
    }
    j["devices"] = devs;
    j["kv"]["used_bytes"] = r.kv_used_bytes;
    j["kv"]["cap_bytes"] = r.kv_cap_bytes;
    j["kv"]["pool_blocks"] = r.kv_pool_blocks;
    j["kv"]["pool_cap"] = r.kv_pool_cap;
    j["kv"]["used_blocks"] = r.kv_used_blocks;
    j["prefix_cache"]["vram_bytes"] = r.pc_vram_bytes;
    j["prefix_cache"]["ram_records"] = r.pc_ram_records;
    j["prefix_cache"]["disk_records"] = r.pc_disk_records;
    j["host"]["rss_bytes"] = r.host_rss_bytes;
    j["inflight"] = r.inflight;
    j["uptime_s"] = r.uptime_s;
    json model;
    model["id"] = r.model_id;
    model["n_ctx"] = r.max_seq;
    model["n_vocab"] = r.n_vocab;
    model["n_layer"] = r.n_layer;
    j["model"] = model;
    return j;
}

uint64_t host_rss_bytes() {
    std::FILE * f = std::fopen("/proc/self/statm", "r");
    if (!f) {
        return 0;
    }
    unsigned long long total = 0, resident = 0;
    const int n = std::fscanf(f, "%llu %llu", &total, &resident);
    std::fclose(f);
    if (n != 2) {
        return 0;
    }
    const long page = ::sysconf(_SC_PAGESIZE);
    return (uint64_t)resident * (uint64_t)(page > 0 ? page : 4096);
}

control_server::control_server(std::function<control_report()> report, std::function<void()> terminate,
                             std::string token)
    : report_(std::move(report)), terminate_(std::move(terminate)), token_(std::move(token)) {
}

control_server::~control_server() {
    stop();
}

bool control_server::start(const std::string & host, int port, std::string & err) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        err = std::string("control: socket failed: ") + std::strerror(errno);
        return false;
    }
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (host == "0.0.0.0" || host.empty()) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        err = "control: '" + host + "' is not an IPv4 address";
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    if (::bind(listen_fd_, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        err = std::string("control: bind failed: ") + std::strerror(errno);
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    if (::listen(listen_fd_, 8) != 0) {
        err = std::string("control: listen failed: ") + std::strerror(errno);
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    // The port may have been 0 ("any"): read back what the kernel chose.
    struct sockaddr_in bound;
    socklen_t blen = sizeof(bound);
    if (::getsockname(listen_fd_, (struct sockaddr *)&bound, &blen) == 0) {
        port_ = ntohs(bound.sin_port);
    } else {
        port_ = port;
    }
    accept_thread_ = std::thread([this]() { accept_loop(); });
    return true;
}

void control_server::stop() {
    if (stopping_.exchange(true)) {
        return;
    }
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
    std::vector<std::shared_ptr<conn>> conns;
    {
        std::lock_guard<std::mutex> lk(conn_mtx_);
        conns.swap(conns_);
    }
    for (const auto & c : conns) {
        if (c->th.joinable()) {
            c->th.join();
        }
    }
}

void control_server::accept_loop() {
    while (!stopping_.load()) {
        struct pollfd pfd;
        pfd.fd = listen_fd_;
        pfd.events = POLLIN;
        pfd.revents = 0;
        const int pr = ::poll(&pfd, 1, 200);
        // Reap the connections that finished, so a long-lived worker does
        // not accumulate thread objects.
        {
            std::lock_guard<std::mutex> lk(conn_mtx_);
            for (size_t i = 0; i < conns_.size();) {
                if (conns_[i]->done.load()) {
                    if (conns_[i]->th.joinable()) {
                        conns_[i]->th.join();
                    }
                    conns_.erase(conns_.begin() + (long)i);
                    continue;
                }
                ++i;
            }
        }
        if (pr <= 0) {
            continue;
        }
        const int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR || errno == EAGAIN) {
                continue;
            }
            break;
        }
        int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        auto c = std::make_shared<conn>();
        c->th = std::thread([this, fd, c]() {
            serve_conn(fd);
            c->done.store(true);
        });
        std::lock_guard<std::mutex> lk(conn_mtx_);
        conns_.push_back(c);
    }
}

void control_server::serve_conn(int fd) {
    rpc::line_reader reader;
    // The connection is authenticated by the handshake; every later request
    // on it inherits that.
    bool authenticated = false;
    for (;;) {
        std::string line;
        // A short poll: the connection has to notice the shutdown promptly
        // (stop() joins these threads), and an idle connection must not hold
        // the shutdown up for a minute.
        if (!reader.next(fd, line, 1000)) {
            if (stopping_.load()) {
                break;
            }
            continue;
        }
        json reply;
        try {
            const json req = json::parse(line);
            reply = dispatch(req, authenticated);
        } catch (const std::exception & ex) {
            reply = rpc::make_error(0, rpc::kParseError, ex.what());
        }
        if (!rpc::write_all(fd, rpc::frame(reply))) {
            break;
        }
    }
    ::close(fd);
}

json control_server::dispatch(const json & req, bool & authenticated) {
    const int64_t id = req.value("id", (int64_t)0);
    const std::string method = req.value("method", std::string());
    if (method.empty()) {
        return rpc::make_error(id, rpc::kInvalidRequest, "no method");
    }
    if (method == rpc::kHello) {
        const int64_t proto = req["params"].value("protocol", (int64_t)0);
        if (proto != rpc::kProtocolVersion) {
            return rpc::make_error(id, rpc::kInvalidParams,
                                   "protocol version " + std::to_string(proto) + " != " +
                                       std::to_string(rpc::kProtocolVersion));
        }
        if (!token_.empty()) {
            // The handshake carries the secret; a wrong one leaves the
            // connection unauthenticated, so every later request fails too.
            const std::string got = req["params"].value("token", std::string());
            if (!rpc::secret_equal(got, token_)) {
                authenticated = false;
                return rpc::make_error(id, rpc::kUnauthorized, "bad or missing token");
            }
            authenticated = true;
        } else {
            authenticated = true;
        }
        json result;
        result["protocol"] = rpc::kProtocolVersion;
        result["worker"] = control_report_to_json(report_());
        return rpc::make_result(id, result);
    }
    if (!token_.empty() && !authenticated) {
        // Every other method needs the handshake first: an unauthenticated
        // peer must not be able to read the device report or stop the worker.
        return rpc::make_error(id, rpc::kUnauthorized, "the handshake must come first");
    }
    if (method == rpc::kHealth) {
        const control_report r = report_();
        json result;
        result["ready"] = true;
        result["uptime_s"] = r.uptime_s;
        result["inflight"] = r.inflight;
        return rpc::make_result(id, result);
    }
    if (method == rpc::kStats) {
        return rpc::make_result(id, control_report_to_json(report_()));
    }
    if (method == rpc::kShutdown) {
        // Take the same path a SIGTERM takes, so the prefix cache is flushed
        // on the way out.
        json result;
        result["ok"] = true;
        if (terminate_) {
            terminate_();
        }
        return rpc::make_result(id, result);
    }
    if (method == rpc::kReload) {
        return rpc::make_error(id, rpc::kInvalidParams, "reload is not implemented");
    }
    return rpc::make_error(id, rpc::kMethodNotFound, "unknown method '" + method + "'");
}

} // namespace si
