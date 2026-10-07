// Remote media URL policy (SSRF boundary).  No model, no GPU.
//
// The unit checks the address classifier exhaustively against the ranges that
// make this an SSRF primitive rather than a media downloader.  The integration
// checks use a real in-process httplib server: it listens on 127.0.0.1, which
// is loopback, so it is both the attacker-controlled endpoint and the
// redirect target - i.e. the whole "public URL that 302s to an internal
// address" chain is exercised without leaving the machine.
//
// The env is read through a function-local static (the same one production
// uses), so each policy needs its own process; the mode is chosen by argv.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"
#include "url_fetch.h"

using namespace si;

static int g_fail = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                            \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

static bool addr_str_public(const char * s, std::string & why) {
    struct sockaddr_in a;
    std::memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    if (::inet_pton(AF_INET, s, &a.sin_addr) != 1) {
        fprintf(stderr, "bad test address %s\n", s);
        g_fail++;
        return false;
    }
    return addr_is_public(&a, why);
}

static bool addr6_str_public(const char * s, std::string & why) {
    struct sockaddr_in6 a;
    std::memset(&a, 0, sizeof(a));
    a.sin6_family = AF_INET6;
    if (::inet_pton(AF_INET6, s, &a.sin6_addr.s6_addr) != 1) {
        fprintf(stderr, "bad test address %s\n", s);
        g_fail++;
        return false;
    }
    return addr_is_public(&a, why);
}

// Every address that must be refused, with the reason only checked as non-empty.
static void test_blocked_v4() {
    static const char * blocked[] = {
        "0.0.0.0",      // unspecified
        "127.0.0.1",    // loopback
        "127.1.2.3",    // loopback, not just .0.0.1
        "10.0.0.1",     // RFC1918
        "10.255.255.254",
        "172.16.0.1",   // RFC1918 lower edge
        "172.31.255.254",// RFC1918 upper edge
        "192.168.1.1",  // RFC1918
        "169.254.169.254", // cloud instance metadata
        "169.254.0.1",
        "100.64.0.1",   // CGNAT
        "100.127.255.254",
        "224.0.0.1",    // multicast
        "239.255.255.255",
        "240.0.0.1",    // reserved
        "255.255.255.255",
        "192.0.0.1",    // special use
        "192.0.2.1",    // TEST-NET-1
        "198.18.0.1",   // benchmarking
        "198.19.255.254",
        "198.51.100.1", // TEST-NET-2
        "203.0.113.1",  // TEST-NET-3
        "192.88.99.1",  // 6to4 anycast
    };
    for (const char * s : blocked) {
        std::string why;
        const bool ok = addr_str_public(s, why);
        if (ok || why.empty()) {
            fprintf(stderr, "FAIL %s should be blocked (why=%s)\n", s, why.c_str());
            g_fail++;
        }
    }
    // 172.15 / 172.32 are public, i.e. the RFC1918 edges are not over-blocked
    std::string why;
    CHECK(addr_str_public("172.15.0.1", why));
    CHECK(addr_str_public("172.32.0.1", why));
}

static void test_blocked_v6() {
    static const char * blocked[] = {
        "::",           // unspecified
        "::1",          // loopback
        "fe80::1",      // link-local
        "fc00::1",      // unique local
        "fd12:3456::1", // unique local
        "ff02::1",      // multicast
    };
    for (const char * s : blocked) {
        std::string why;
        const bool ok = addr6_str_public(s, why);
        if (ok || why.empty()) {
            fprintf(stderr, "FAIL %s should be blocked (why=%s)\n", s, why.c_str());
            g_fail++;
        }
    }
    // The bypass this specifically exists for: an IPv4-mapped loopback address
    // must be classified by the embedded IPv4, not accepted as "IPv6".
    std::string why;
    CHECK(!addr6_str_public("::ffff:127.0.0.1", why));
    CHECK(why.find("IPv4-mapped") != std::string::npos);
    CHECK(!addr6_str_public("::ffff:169.254.169.254", why));
    // ...and a mapped public address is still allowed
    CHECK(addr6_str_public("::ffff:93.184.216.34", why));
    // a real public v6 and a 6to4 of a public v4 pass
    CHECK(addr6_str_public("2606:2800:220:1:248:1893:25c8:1946", why));
    CHECK(addr6_str_public("2002:5db8:d822::1", why));
    CHECK(!addr6_str_public("2002:7f00:1::1", why)); // 6to4 of 127.0.0.1
}

// Resolve-based, so the decimal/octal loopback spellings cannot slip past.
static void test_host_resolution() {
    std::string why;
    CHECK(!host_is_public("127.0.0.1", why));
    CHECK(!host_is_public("localhost", why));
    CHECK(!host_is_public("::1", why));
    // "2130706433" is 127.0.0.1 in decimal and "0177.0.0.1" in octal; a
    // string-matching filter would call both public
    CHECK(!host_is_public("2130706433", why));
    CHECK(!host_is_public("0177.0.0.1", why));
    CHECK(!host_is_public("this-host-does-not-exist.invalid", why));
}

static void test_url_parse() {
    url_parts p;
    std::string err;
    CHECK(url_parse("https://example.com/a/b.png", p, err));
    CHECK(p.scheme == "https" && p.host == "example.com" && p.port == 443);
    CHECK(p.path == "/a/b.png");
    CHECK(p.base == "https://example.com:443");

    // host case is preserved (DNS is case-insensitive and getaddrinfo handles
    // it), so this only pins the port/path/default-path handling
    CHECK(url_parse("http://Example.COM:8080", p, err));
    CHECK(p.port == 8080 && p.path == "/" && p.base == "http://Example.COM:8080");

    // userinfo is dropped, not forwarded
    CHECK(url_parse("http://user:pw@host.tld/x", p, err));
    CHECK(p.host == "host.tld");

    // IPv6 literal: brackets stripped from host, kept in the httplib base
    CHECK(url_parse("http://[::1]:9000/y", p, err));
    CHECK(p.host == "::1" && p.port == 9000);
    CHECK(p.base == "http://[::1]:9000");

    // rejections
    CHECK(!url_parse("ftp://host/x", p, err));
    CHECK(!url_parse("file:///etc/passwd", p, err));
    CHECK(!url_parse("http://", p, err));
    CHECK(!url_parse("http://host:port/x", p, err));
    CHECK(!url_parse("no-scheme", p, err));
}

static void test_redirect_resolution() {
    url_parts from;
    std::string err, next;
    CHECK(url_parse("https://a.example/dir/page.html", from, err));

    CHECK(resolve_redirect(from, "https://b.example/x.png", next, err));
    CHECK(next == "https://b.example:443/x.png");

    // root-relative and path-relative both re-anchor on the right directory
    CHECK(resolve_redirect(from, "/img/x.png", next, err));
    CHECK(next == "https://a.example:443/img/x.png");
    CHECK(resolve_redirect(from, "x.png", next, err));
    CHECK(next == "https://a.example:443/dir/x.png");

    // scheme-relative inherits the scheme
    CHECK(resolve_redirect(from, "//cdn.example/x.png", next, err));
    CHECK(next == "https://cdn.example:443/x.png");

    // https -> http downgrade refused
    CHECK(!resolve_redirect(from, "http://b.example/x.png", next, err));
    CHECK(err.find("https -> http") != std::string::npos);
    CHECK(!resolve_redirect(from, "http://a.example/x.png", next, err));

    // http -> https is fine
    url_parts plain;
    CHECK(url_parse("http://a.example/dir/", plain, err));
    CHECK(resolve_redirect(plain, "https://a.example/x.png", next, err));

    CHECK(!resolve_redirect(from, "", next, err));
}

// A public-looking hostname is not testable offline, so the integration test
// uses the loopback server and asserts the policy it must refuse.  Mode comes
// from argv because url_fetch_mode() caches its answer in a static.
static void test_fetch_policy(int port) {
    httplib::Server srv;
    srv.Get("/ok.png", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content("PNGDATA", "image/png");
    });
    srv.Get("/redirect", [&](const httplib::Request &, httplib::Response & res) {
        res.status = 302;
        res.set_header("Location", "http://127.0.0.1:" + std::to_string(port) + "/secret");
    });
    srv.Get("/secret", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content("SECRET", "text/plain");
    });
    std::thread sth([&] { srv.listen("127.0.0.1", port); });
    for (int i = 0; i < 300 && !srv.is_running(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(srv.is_running());

    const std::string base = "http://127.0.0.1:" + std::to_string(port);
    const url_fetch_policy mode = url_fetch_mode();
    std::vector<uint8_t> bytes;
    std::string err;

    if (mode == url_fetch_policy::disabled) {
        // default: even a perfectly harmless loopback fetch is refused
        CHECK(!url_fetch(base + "/ok.png", bytes, err));
        CHECK(err.find("disabled") != std::string::npos);
        CHECK(bytes.empty());
    } else if (mode == url_fetch_policy::public_only) {
        CHECK(mode == url_fetch_policy::public_only);
        // a direct internal fetch is refused, and the refusal names the reason
        CHECK(!url_fetch(base + "/ok.png", bytes, err));
        CHECK(err.find("refusing to fetch 127.0.0.1") != std::string::npos);
        CHECK(err.find("loopback") != std::string::npos);
        CHECK(bytes.empty());
        // ...and so is the redirect chain, which is the reason redirects are
        // walked by hand: the first hop is the only one this check would miss
        CHECK(!url_fetch(base + "/redirect", bytes, err));
        CHECK(err.find("refusing to fetch") != std::string::npos);
        CHECK(bytes.empty());
        // the escape hatch is a separate env var, so it must not be implied
        CHECK(err.find("PF_MM_URL_ALLOW_PRIVATE=1") != std::string::npos);
    } else {
        // allow_private: the whole point is that it now works
        CHECK(url_fetch(base + "/ok.png", bytes, err));
        CHECK(std::string(bytes.begin(), bytes.end()) == "PNGDATA");
        CHECK(url_fetch(base + "/redirect", bytes, err));
        CHECK(std::string(bytes.begin(), bytes.end()) == "SECRET");
        CHECK(url_fetch(base + "/missing", bytes, err) == false);
        CHECK(err.find("HTTP 404") != std::string::npos);
    }

    srv.stop();
    sth.join();
}

int main(int argc, char ** argv) {
    const int port = argc > 1 ? std::atoi(argv[1]) : 18141;
    // url_fetch_mode() caches its answer in a function-local static, so the
    // three policies need three processes.  Clear the env first: an ambient
    // PF_MM_URL_FETCH must not decide what the "default" case measures.
    unsetenv("PF_MM_URL_FETCH");
    unsetenv("PF_MM_URL_ALLOW_PRIVATE");
    const std::string mode = argc > 2 ? argv[2] : "default";
    if (mode == "public" || mode == "private") {
        setenv("PF_MM_URL_FETCH", "1", 1);
    }
    if (mode == "private") {
        setenv("PF_MM_URL_ALLOW_PRIVATE", "1", 1);
    }
    printf("policy: %s\n", url_fetch_mode() == url_fetch_policy::disabled ? "disabled"
                                : url_fetch_mode() == url_fetch_policy::public_only ? "public_only"
                                                                                  : "allow_private");
    test_blocked_v4();
    test_blocked_v6();
    test_host_resolution();
    test_url_parse();
    test_redirect_resolution();
    test_fetch_policy(port);
    if (g_fail) {
        fprintf(stderr, "test_url_fetch: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_url_fetch: all checks OK\n");
    return 0;
}