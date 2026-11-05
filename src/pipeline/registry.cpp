#include "registry.h"

#include "model_profile.h"
#include "pipeline.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace si {

bool pipeline_registry::add(std::unique_ptr<pipeline> p) {
    if (!p || p->desc().id == nullptr || p->desc().id[0] == '\0') {
        return false;
    }
    const std::string id = p->desc().id;
    if (find(id) != nullptr) {
        return false;
    }
    items_.push_back(std::move(p));
    return true;
}

pipeline * pipeline_registry::find(const std::string & id) const {
    for (const std::unique_ptr<pipeline> & p : items_) {
        if (p && id == p->desc().id) {
            return p.get();
        }
    }
    return nullptr;
}

std::vector<const pipeline *> pipeline_registry::available(const model_profile & prof) const {
    std::vector<const pipeline *> out;
    for (const std::unique_ptr<pipeline> & p : items_) {
        if (p && p->available(prof)) {
            out.push_back(p.get());
        }
    }
    return out;
}

} // namespace si
