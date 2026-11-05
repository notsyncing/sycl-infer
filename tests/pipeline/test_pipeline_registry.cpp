#include "pipeline/registry.h"

#include <cstdio>
#include <memory>
#include <string>

namespace {

int g_fail = 0;

#define CHECK(cond, ...)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                                                       \
            fprintf(stderr, __VA_ARGS__);                                                                              \
            fprintf(stderr, "\n");                                                                                     \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

class fake_pipeline : public si::pipeline {
public:
    fake_pipeline(std::string id, std::string name, bool available)
        : id_(std::move(id)), name_(std::move(name)), available_(available) {
        d_.id = id_.c_str();
        d_.name = name_.c_str();
        d_.accepts_image = true;
        d_.output_formats = "json,text";
    }

    const si::pipeline_desc & desc() const override {
        return d_;
    }
    bool available(const si::model_profile &) const override {
        return available_;
    }
    void run(const si::pipeline_request &, si::pipeline_sink &) override {
        ++runs;
    }

    int runs = 0;

private:
    std::string id_;
    std::string name_;
    bool available_;
    si::pipeline_desc d_;
};

void test_empty() {
    si::pipeline_registry reg;
    CHECK(reg.empty(), "a new registry is empty");
    CHECK(reg.find("nope") == nullptr, "an empty registry finds nothing");
    CHECK(reg.available(si::model_profile{}).empty(), "an empty registry has no available pipeline");
    CHECK(reg.all().empty(), "all() is empty");
}

void test_add_find_and_filter() {
    si::pipeline_registry reg;
    auto on = std::make_unique<fake_pipeline>("on", "On", true);
    auto off = std::make_unique<fake_pipeline>("off", "Off", false);
    fake_pipeline * on_raw = on.get();
    CHECK(reg.add(std::move(on)), "the first add succeeds");
    CHECK(reg.add(std::move(off)), "the second add succeeds");
    CHECK(!reg.empty(), "the registry is not empty");
    CHECK(reg.all().size() == 2, "two entries");
    CHECK(reg.find("on") == on_raw, "find returns the registered instance");
    CHECK(reg.find("missing") == nullptr, "find misses an unknown id");

    const auto avail = reg.available(si::model_profile{});
    CHECK(avail.size() == 1, "only the available pipeline is listed: %zu", avail.size());
    if (!avail.empty()) {
        CHECK(std::string(avail[0]->desc().id) == "on", "available id: %s", avail[0]->desc().id);
    }
}

void test_duplicate() {
    si::pipeline_registry reg;
    auto first = std::make_unique<fake_pipeline>("dup", "First", true);
    fake_pipeline * first_raw = first.get();
    CHECK(reg.add(std::move(first)), "the first registration");
    CHECK(!reg.add(std::make_unique<fake_pipeline>("dup", "Second", true)), "a duplicate id is refused");
    CHECK(reg.all().size() == 1, "the duplicate is not added");
    CHECK(reg.find("dup") == first_raw, "the first registration wins");
}

void test_empty_id() {
    si::pipeline_registry reg;
    CHECK(!reg.add(std::make_unique<fake_pipeline>("", "No id", true)), "an empty id is refused");
    CHECK(reg.empty(), "nothing was added");
}

} // namespace

int main() {
    test_empty();
    test_add_find_and_filter();
    test_duplicate();
    test_empty_id();
    if (g_fail) {
        fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_pipeline_registry: OK\n");
    return 0;
}
