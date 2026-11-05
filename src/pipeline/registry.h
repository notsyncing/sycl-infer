#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model_profile.h"
#include "pipeline.h"

namespace si {

// The pipeline table.  Registration happens once at serve() startup; the
// model's own `/v1/models` table and the router's model routing know nothing
// about it.
class pipeline_registry {
public:
    // Returns false (and keeps the first registration) for a duplicate or an
    // empty id.  A duplicate is a programming error, but making it observable
    // is what lets the hermetic test pin it.
    bool add(std::unique_ptr<pipeline> p);

    pipeline * find(const std::string & id) const;

    // Only the pipelines whose `available(prof)` is true, in registration
    // order.  The HTTP layer calls this with the engine's model profile (or a
    // default one when there is no model context).
    std::vector<const pipeline *> available(const model_profile & prof) const;

    const std::vector<std::unique_ptr<pipeline>> & all() const {
        return items_;
    }
    bool empty() const {
        return items_.empty();
    }

private:
    std::vector<std::unique_ptr<pipeline>> items_;
};

} // namespace si
