#include "rpc_client.h"

#include "json.hpp"
#include "jsonrpc.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>



namespace si {
namespace router {

using json = nlohmann::json;

rpc_client::rpc_client(std::string host, int port, std::string token)
    : host_(std::move(host)), port_(port), token_(std::move(token)) {
}

rpc_client::~rpc_client() {
    close();
}

void rpc_client::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    reader_ = rpc::line_reader();
}

bool rpc_client::connect(int timeout_ms, std::string & err) {
    close();
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        err = std::string("socket failed: ") + std::strerror(errno);
        return false;
    }
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port_);
    if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
        err = "'" + host_ + "' is not an IPv4 address";
        ::close(fd);
        return false;
    }
    // Connect with a timeout: a worker on another machine may be
    // unreachable, and the router must not block its own start-up on that.
    const int flags = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    int rc = ::connect(fd, (struct sockaddr *)&addr, sizeof(addr));
    if (rc != 0) {
        if (errno != EINPROGRESS) {
            err = std::string("connect failed: ") + std::strerror(errno);
            ::close(fd);
            return false;
        }
        struct pollfd pfd;
        pfd.fd = fd;
        pfd.events = POLLOUT;
        pfd.revents = 0;
        const int pr = ::poll(&pfd, 1, timeout_ms);
        if (pr <= 0) {
            err = "connect timed out";
            ::close(fd);
            return false;
        }
        int soerr = 0;
        socklen_t len = sizeof(soerr);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &len) != 0 || soerr != 0) {
            err = std::string("connect failed: ") + std::strerror(soerr ? soerr : errno);
            ::close(fd);
            return false;
        }
        ::fcntl(fd, F_SETFL, flags);
    }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    fd_ = fd;
    return true;
}

bool rpc_client::call(const std::string & method, const json & params, json & result, std::string & err,
                       int timeout_ms) {
    if (fd_ < 0) {
        err = "not connected";
        return false;
    }
    const int64_t id = next_id_++;
    json p = params;
    if (method == rpc::kHello) {
        // The handshake carries the secret (when there is one) and the
        // protocol version: a worker that speaks a different one is refused
        // rather than misread.
        p["protocol"] = rpc::kProtocolVersion;
        if (!token_.empty()) {
            p["token"] = token_;
        }
    }
    const std::string req = rpc::frame(rpc::make_request(id, method, p));
    if (!rpc::write_all(fd_, req)) {
        err = "the write failed: " + std::string(std::strerror(errno));
        close();
        return false;
    }
    // Read one line.  A worker may push an event before answering, so skip
    // the frames that are not our reply.
    for (int attempt = 0; attempt < 8; attempt++) {
        std::string line;
        if (!reader_.next(fd_, line, timeout_ms)) {
            err = "the worker did not answer";
            close();
            return false;
        }
        json reply;
        try {
            reply = json::parse(line);
        } catch (const std::exception & ex) {
            err = std::string("the worker's answer is not JSON: ") + ex.what();
            close();
            return false;
        }
        if (reply.value("id", (int64_t)-1) != id) {
            continue; // an event, or a reply to a call that timed out
        }
        if (reply.contains("error")) {
            err = reply["error"].value("message", std::string("the worker reported an error"));
            return false;
        }
        result = reply.value("result", json::object());
        return true;
    }
    err = "no reply from the worker";
    return false;
}

} // namespace router
} // namespace si
