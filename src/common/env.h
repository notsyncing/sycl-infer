// Central place for reading the PF_* tuning/diagnostic environment variables.
//
// Two forms at each call site:
//   env::flag(name) / env::i32(name, dflt) / env::str(name)
//       read the variable on every call (cheap, cold paths);
//   SI_ENV_FLAG(name) / SI_ENV_I32(name, dflt)
//       cache the first read in a function-local static (one per call site).
//
// Every variable in this project is fixed for the lifetime of the process (the
// A/B sweeps run one process per config), so caching a site is behaviour-
// preserving wherever the site used to read a `static const bool x = ...`.
//
// Uniform "0/empty = off" rule:
//   flag        unset, "0", or "" -> false, anything else -> true
//   i32         unset or unparsable -> dflt, otherwise atoi
//   str         nullptr when unset, else the raw value (not owned)
#pragma once

#include <cstdlib>
#include <string>

namespace si {
namespace env {

inline bool flag(const char * name) {
    const char * e = std::getenv(name);
    return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}

inline const char * str(const char * name) {
    return std::getenv(name);
}

inline int i32(const char * name, int dflt) {
    const char * e = std::getenv(name);
    return e ? std::atoi(e) : dflt;
}

inline float f32(const char * name, float dflt) {
    const char * e = std::getenv(name);
    return e ? (float)std::atof(e) : dflt;
}

} // namespace env
} // namespace si

#define SI_ENV_FLAG(name) [] { static const bool v = ::si::env::flag(name); return v; }()
#define SI_ENV_I32(name, dflt) [] { static const int v = ::si::env::i32(name, dflt); return v; }()
