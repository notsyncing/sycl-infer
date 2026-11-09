#include "registry.h"

#include "config.h"

#include <memory>
#include <mutex>
#include <string>
#include <utility>



namespace si {
namespace router {

const char * model_state_name(model_state s) {
    switch (s) {
    case model_state::starting:
        return "starting";
    case model_state::loading:
        return "loading";
    case model_state::up:
        return "up";
    case model_state::degraded:
        return "degraded";
    case model_state::restarting:
        return "restarting";
    case model_state::failed:
        return "failed";
    case model_state::stopping:
        return "stopping";
    }
    return "unknown";
}

std::shared_ptr<model_slot> registry::add(model_config cfg) {
    auto s = std::make_shared<model_slot>();
    s->cfg = std::move(cfg);
    models_.push_back(s);
    return s;
}

std::shared_ptr<model_slot> registry::find(const std::string & id) const {
    // The configured id is the routing key and the published one.  The worker's
    // own advertised id (its GGUF `general.name`) is accepted as a
    // compatibility alias: clients that discovered it from an older
    // /v1/models, or that hard-code it, still route.  The configured id wins
    // if the two ever collide.
    for (const auto & m : models_) {
        if (m->cfg.id == id) {
            return m;
        }
    }
    for (const auto & m : models_) {
        std::lock_guard<std::mutex> lk(m->mtx);
        if (!m->advertised_id.empty() && m->advertised_id == id) {
            return m;
        }
    }
    return nullptr;
}

} // namespace router
} // namespace si
