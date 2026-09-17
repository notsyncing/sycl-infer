#pragma once
// Shim so that vendored headers expecting <nlohmann/json.hpp> (minja) pick up
// the single-header nlohmann json already vendored at third_party/json.hpp.
#include "../json.hpp"
