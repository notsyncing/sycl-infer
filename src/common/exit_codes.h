#pragma once

// The worker's exit codes.
//
// The router supervises the worker and has to decide whether a death is worth
// retrying.  "The same attempt would fail the same way" (a model that does
// not load, a card that is out of memory, a bad flag) is not, so the worker
// says so with a distinct exit code instead of the router having to guess
// from the log text.
//
// 78 is sysexits.h's EX_CONFIG: "the process cannot start because of its
// configuration or its environment".  It is the conventional value and it is
// far from 0 and 1, so it cannot collide with a normal failure.
namespace si {

// The worker could not start (the model does not load, the device is out of
// memory, the arguments are wrong).  Do not retry.
constexpr int kExitStartupFailure = 78;

// A runtime failure (the command itself failed).  Retryable.
constexpr int kExitRuntimeError = 1;

} // namespace si
