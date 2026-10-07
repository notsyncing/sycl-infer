#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Remote media fetching (the http(s) form of image_url / video_url /
// audio_url / input_audio).  Its own translation unit because the policy is
// the security boundary and must be testable without a model - see
// tests/server/test_url_fetch.cpp.
//
// Threat model: this endpoint is unauthenticated and the server binds a routable
// address by default in some deployments, so "let the server GET this URL" is
// a server-side request forgery primitive against everything the server can
// reach but the caller cannot - cloud instance metadata, cluster-internal
// admin APIs, an unauthenticated inference port on the same host.  Two
// defenses, both opt-out-able because both are legitimate for an on-premise
// media server:
//
//   1. remote fetch is OFF unless PF_MM_URL_FETCH is explicitly set;
//   2. with it on, every address the hostname resolves to must be publicly
//      routable (PF_MM_URL_ALLOW_PRIVATE=1 lifts this for internal servers),
//      and redirects are followed by hand so each hop is re-checked -
//      httplib's own set_follow_location re-resolves and re-connects without
//      giving the caller a hook, which turns a validated URL into an arbitrary
//      one (the classic "public URL that 302s to 169.254.169.254").
//
// Known limit: the address is validated at check time and httplib resolves
// again when it connects, so a hostile DNS server can still win that race
// (DNS rebinding).  Closing it needs the connection pinned to the validated
// IP, which httplib offers no hook for.  With the default off, reaching it
// requires having opted in to remote fetch at all.
namespace si {

struct url_parts {
    std::string scheme; // "http" or "https"
    std::string host;   // no port, no brackets (IPv6 literal kept bare)
    std::string path;   // always starts with '/'
    int port = 0;
    std::string base;   // "scheme://host:port", what httplib::Client takes
};

// Absolute http(s) only; rejects other schemes, a missing host and a
// non-numeric port.  Any userinfo is dropped (it is never sent).
bool url_parse(const std::string & url, url_parts & out, std::string & err);

// True when the address is publicly routable.  Rejects loopback, private
// (RFC1918), CGNAT, link-local (which is where cloud metadata lives: the
// 169.254.169.254 well-known address), unspecified, multicast and the
// special-use / documentation ranges.  An IPv4-mapped IPv6 address is
// classified by its embedded IPv4, so ::ffff:127.0.0.1 is not a bypass.
bool addr_is_public(const void * sockaddr_ptr, std::string & err);

// Resolve `host` and require *every* returned address to be public, so a name
// with one public and one private A record cannot slip through.
bool host_is_public(const std::string & host, std::string & err);

// Resolve a Location header against the URL it came from (absolute,
// scheme-relative, root-relative or path-relative).  An https -> http
// downgrade is refused: it would silently drop the transport guarantee of a
// URL that was allowed on that basis.
bool resolve_redirect(const url_parts & from, const std::string & location, std::string & next, std::string & err);

enum class url_fetch_policy {
    disabled,     // PF_MM_URL_FETCH unset or 0 (default)
    public_only,  // PF_MM_URL_FETCH=1
    allow_private // PF_MM_URL_FETCH=1 PF_MM_URL_ALLOW_PRIVATE=1
};

url_fetch_policy url_fetch_mode();

// Download one URL, following up to 5 redirects and re-validating each hop.
// Bounded at 10 MB.  `err` receives a message safe to hand back to the client
// (it never contains fetched bytes).
bool url_fetch(const std::string & url, std::vector<uint8_t> & bytes, std::string & err);

} // namespace si