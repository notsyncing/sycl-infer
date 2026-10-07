#include "url_fetch.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cstdlib>
#include <cstring>

#include "httplib.h"
#include "common/env.h"

namespace si {

namespace {

// ---------------------------------------------------------------- address bits

bool v4_is_public(uint32_t a, std::string & err) {
    const uint8_t b0 = (uint8_t)(a >> 24), b1 = (uint8_t)(a >> 16), b2 = (uint8_t)(a >> 8);
    if (a == 0) {
        err = "unspecified address";
        return false;
    }
    if (b0 == 127) {
        err = "loopback";
        return false;
    }
    if (b0 == 10) {
        err = "private (RFC1918)";
        return false;
    }
    if (b0 == 172 && b1 >= 16 && b1 <= 31) {
        err = "private (RFC1918)";
        return false;
    }
    if (b0 == 192 && b1 == 168) {
        err = "private (RFC1918)";
        return false;
    }
    if (b0 == 169 && b1 == 254) {
        err = "link-local (includes the cloud metadata address)";
        return false;
    }
    if (b0 == 100 && b1 >= 64 && b1 <= 127) {
        err = "CGNAT (RFC6598)";
        return false;
    }
    if (b0 >= 224) {
        err = b0 < 240 ? "multicast" : "reserved/broadcast";
        return false;
    }
    // special-use / documentation ranges: none of them is a media host, and
    // letting them through only helps an attacker probe them
    if (b0 == 192 && b1 == 0 && b2 == 0) { // 192.0.0.0/24 IETF protocol assignments
        err = "special-use (192.0.0.0/24)";
        return false;
    }
    if (b0 == 192 && b1 == 0 && b2 == 2) { // 192.0.2.0/24 TEST-NET-1
        err = "special-use (192.0.2.0/24)";
        return false;
    }
    if (b0 == 192 && b1 == 88 && b2 == 99) { // 192.88.99.0/24 6to4 anycast
        err = "special-use (192.88.99.0/24)";
        return false;
    }
    if (b0 == 198 && (b1 == 18 || b1 == 19)) { // 198.18.0.0/15 benchmarking
        err = "special-use (198.18.0.0/15)";
        return false;
    }
    if (b0 == 198 && b1 == 51 && b2 == 100) { // 198.51.100.0/24 TEST-NET-2
        err = "special-use (198.51.100.0/24)";
        return false;
    }
    if (b0 == 203 && b1 == 0 && b2 == 113) { // 203.0.113.0/24 TEST-NET-3
        err = "special-use (203.0.113.0/24)";
        return false;
    }
    return true;
}

bool v6_is_public(const uint8_t a[16], std::string & err) {
    // IPv4-mapped (::ffff:0:0/96) and IPv4-compatible: classify the embedded
    // address, otherwise ::ffff:127.0.0.1 walks straight past the IPv4 rules.
    if (IN6_IS_ADDR_V4MAPPED(a) || (a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0 && a[4] == 0 && a[5] == 0 &&
                                   a[6] == 0 && a[7] == 0 && a[8] == 0 && a[9] == 0 && a[10] == 0xff)) {
        uint32_t v = ((uint32_t)a[12] << 24) | ((uint32_t)a[13] << 16) | ((uint32_t)a[14] << 8) | (uint32_t)a[15];
        if (!v4_is_public(v, err)) {
            err = "IPv4-mapped " + err;
            return false;
        }
        return true;
    }
    bool all_zero = true;
    for (int i = 0; i < 16; i++) {
        if (a[i]) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        err = "unspecified address";
        return false;
    }
    if (a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0 && a[4] == 0 && a[5] == 0 && a[6] == 0 && a[7] == 0 &&
        a[8] == 0 && a[9] == 0 && a[10] == 0 && a[11] == 0 && a[12] == 0 && a[13] == 0 && a[14] == 0 && a[15] == 1) {
        err = "loopback";
        return false;
    }
    if ((a[0] & 0xFE) == 0xFC) { // fc00::/7 unique local
        err = "unique-local (fc00::/7)";
        return false;
    }
    if (a[0] == 0xFE && (a[1] & 0xC0) == 0x80) { // fe80::/10 link-local
        err = "link-local (fe80::/10)";
        return false;
    }
    if (a[0] == 0xFF) {
        err = "multicast";
        return false;
    }
    if (a[0] == 0x20 && (a[1] & 0x0F) == 0x02) { // 2002::/16 6to4 wrapping a v4 address
        uint32_t v = ((uint32_t)a[2] << 24) | ((uint32_t)a[3] << 16) | ((uint32_t)a[4] << 8) | (uint32_t)a[5];
        std::string ignored;
        if (!v4_is_public(v, ignored)) {
            err = "6to4 of a non-public IPv4 address";
            return false;
        }
    }
    return true;
}

std::string strip_ipv6_brackets(const std::string & h) {
    if (h.size() >= 2 && h.front() == '[' && h.back() == ']') {
        return h.substr(1, h.size() - 2);
    }
    return h;
}

// ---------------------------------------------------------------- URL pieces

// RFC 3986 relative resolution, restricted to what a Location header uses.
std::string resolve_path(const std::string & base_path, const std::string & rel) {
    if (rel.empty()) {
        return base_path;
    }
    if (rel[0] == '/') {
        return rel;
    }
    const size_t slash = base_path.rfind('/');
    const std::string dir = (slash == std::string::npos) ? std::string("/") : base_path.substr(0, slash + 1);
    return dir + rel;
}

} // namespace

// ------------------------------------------------------------------ public API

bool url_parse(const std::string & url, url_parts & out, std::string & err) {
    const size_t scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
        err = "malformed media URL";
        return false;
    }
    const std::string scheme = url.substr(0, scheme_end);
    if (scheme != "http" && scheme != "https") {
        err = "unsupported media URL scheme: " + scheme;
        return false;
    }
    const size_t host_start = scheme_end + 3;
    const size_t path_start = url.find('/', host_start);
    std::string authority =
        (path_start == std::string::npos) ? url.substr(host_start) : url.substr(host_start, path_start - host_start);
    out.path = (path_start == std::string::npos) ? "/" : url.substr(path_start);
    const size_t at = authority.rfind('@'); // drop any userinfo
    if (at != std::string::npos) {
        authority = authority.substr(at + 1);
    }
    int port = (scheme == "https") ? 443 : 80;
    std::string host = authority;
    std::string port_s;
    if (!authority.empty() && authority[0] == '[') { // IPv6 literal
        const size_t br = authority.find(']');
        if (br == std::string::npos) {
            err = "malformed IPv6 host in media URL";
            return false;
        }
        host = authority.substr(0, br + 1);
        if (br + 1 < authority.size() && authority[br + 1] == ':') {
            port_s = authority.substr(br + 2);
        }
    } else {
        const size_t colon = authority.rfind(':');
        if (colon != std::string::npos) {
            host = authority.substr(0, colon);
            port_s = authority.substr(colon + 1);
        }
    }
    host = strip_ipv6_brackets(host);
    if (host.empty()) {
        err = "media URL has no host";
        return false;
    }
    if (!port_s.empty()) {
        if (port_s.find_first_not_of("0123456789") != std::string::npos) {
            err = "malformed port in media URL";
            return false;
        }
        port = std::atoi(port_s.c_str());
        if (port <= 0 || port > 65535) {
            err = "media URL port out of range";
            return false;
        }
    }
    out.scheme = scheme;
    out.host = host;
    out.port = port;
    const bool v6 = host.find(':') != std::string::npos;
    out.base = scheme + "://" + (v6 ? "[" + host + "]" : host) + ":" + std::to_string(port);
    return true;
}

bool addr_is_public(const void * sockaddr_ptr, std::string & err) {
    const struct sockaddr * sa = (const struct sockaddr *)sockaddr_ptr;
    if (sa->sa_family == AF_INET) {
        const struct sockaddr_in * s4 = (const struct sockaddr_in *)sa;
        uint32_t v = ntohl(s4->sin_addr.s_addr);
        return v4_is_public(v, err);
    }
    if (sa->sa_family == AF_INET6) {
        const struct sockaddr_in6 * s6 = (const struct sockaddr_in6 *)sa;
        return v6_is_public(s6->sin6_addr.s6_addr, err);
    }
    err = "unsupported address family";
    return false;
}

bool host_is_public(const std::string & host, std::string & err) {
    // Resolve, never string-match: decimal ("2130706433") and octal
    // ("0177.0.0.1") loopback spellings reach 127.0.0.1 through getaddrinfo,
    // so they are caught here and would slip past any textual check.
    struct addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo * res = nullptr;
    const int rc = ::getaddrinfo(host.c_str(), nullptr, &hints, &res);
    if (rc != 0 || !res) {
        err = "cannot resolve host";
        if (res) {
            ::freeaddrinfo(res);
        }
        return false;
    }
    int n = 0;
    bool ok = true;
    for (struct addrinfo * it = res; it; it = it->ai_next) {
        std::string why;
        if (!it->ai_addr) {
            continue;
        }
        n++;
        if (!addr_is_public(it->ai_addr, why)) {
            ok = false;
            err = why;
            // keep scanning: report the first reason, but a name with several
            // A records must be rejected if *any* of them is non-public
        }
    }
    ::freeaddrinfo(res);
    if (n == 0) {
        err = "host resolved to no addresses";
        return false;
    }
    return ok;
}

bool resolve_redirect(const url_parts & from, const std::string & location, std::string & next, std::string & err) {
    if (location.empty()) {
        err = "redirect with an empty Location";
        return false;
    }
    if (location.find("://") != std::string::npos) {
        next = location;
    } else if (location.compare(0, 2, "//") == 0) {
        next = from.scheme + ":" + location; // scheme-relative
    } else {
        next = from.scheme + "://" + from.host + ":" + std::to_string(from.port) + resolve_path(from.path, location);
    }
    // normalize a duplicated slash so the emitted base matches httplib's
    url_parts probe;
    if (!url_parse(next, probe, err)) {
        return false;
    }
    if (from.scheme == "https" && probe.scheme == "http") {
        err = "refusing an https -> http redirect";
        return false;
    }
    next = probe.scheme + "://" + (probe.host.find(':') != std::string::npos ? "[" + probe.host + "]" : probe.host) +
           ":" + std::to_string(probe.port) + probe.path;
    return true;
}

url_fetch_policy url_fetch_mode() {
    static const url_fetch_policy mode = [] {
        const char * v = si::env::str("PF_MM_URL_FETCH");
        if (!v || std::atoi(v) == 0) {
            return url_fetch_policy::disabled;
        }
        const char * p = si::env::str("PF_MM_URL_ALLOW_PRIVATE");
        if (p && std::atoi(p) != 0) {
            return url_fetch_policy::allow_private;
        }
        return url_fetch_policy::public_only;
    }();
    return mode;
}

bool url_fetch(const std::string & url, std::vector<uint8_t> & bytes, std::string & err) {
    bytes.clear();
    const url_fetch_policy mode = url_fetch_mode();
    if (mode == url_fetch_policy::disabled) {
        err = "remote media URLs are disabled (set PF_MM_URL_FETCH=1 to allow)";
        return false;
    }
    constexpr size_t kMaxBytes = 10 * 1024 * 1024;
    constexpr int kMaxRedirects = 5;

    url_parts cur;
    if (!url_parse(url, cur, err)) {
        return false;
    }
    for (int hop = 0;; hop++) {
        // Re-validated per hop, not once for the original URL: that is the
        // whole point of not using set_follow_location.
        if (mode == url_fetch_policy::public_only) {
            std::string why;
            if (!host_is_public(cur.host, why)) {
                err = "refusing to fetch " + cur.host + ": " + why +
                      " (set PF_MM_URL_ALLOW_PRIVATE=1 to allow private addresses)";
                return false;
            }
        }
        httplib::Client cli(cur.base);
        cli.set_follow_location(false); // every hop goes through the checks above
        cli.set_connection_timeout(10, 0);
        cli.set_read_timeout(10, 0);
        cli.set_write_timeout(10, 0);
        const size_t before = bytes.size();
        bool too_large = false;
        auto res = cli.Get(cur.path, httplib::Headers{{"User-Agent", "sycl-infer"}},
                           [&](const char * data, size_t len) {
                               if (bytes.size() + len > kMaxBytes) {
                                   too_large = true;
                                   return false;
                               }
                               bytes.insert(bytes.end(), data, data + len);
                               return true;
                           });
        if (too_large) {
            err = "media URL exceeds the 10 MB limit";
            bytes.clear();
            return false;
        }
        if (!res) {
            err = "cannot fetch media URL: " + httplib::to_string(res.error());
            bytes.clear();
            return false;
        }
        if (res->status >= 300 && res->status < 400) {
            const std::string loc = res->get_header_value("Location");
            if (loc.empty()) {
                err = "media URL returned HTTP " + std::to_string(res->status) + " with no Location";
                bytes.clear();
                return false;
            }
            if (hop >= kMaxRedirects) {
                err = "too many redirects fetching media URL";
                bytes.clear();
                return false;
            }
            std::string next;
            if (!resolve_redirect(cur, loc, next, err)) {
                bytes.clear();
                return false;
            }
            url_parts np;
            if (!url_parse(next, np, err)) {
                bytes.clear();
                return false;
            }
            bytes.resize(before); // a redirect body is not payload
            cur = np;
            continue;
        }
        if (res->status < 200 || res->status >= 300) {
            err = "media URL returned HTTP " + std::to_string(res->status);
            bytes.clear();
            return false;
        }
        break;
    }
    if (bytes.empty()) {
        err = "media URL returned no data";
        return false;
    }
    return true;
}

} // namespace si