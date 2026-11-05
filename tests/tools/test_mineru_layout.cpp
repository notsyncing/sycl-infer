#include "layout.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

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

bool near(float a, float b) {
    return std::fabs(a - b) < 1e-6f;
}

void test_prompts() {
    CHECK(si::layout_prompt() == "\nLayout Detection:", "layout prompt");
    CHECK(si::content_prompt("table") == "\nTable Recognition:", "table prompt");
    CHECK(si::content_prompt("equation") == "\nFormula Recognition:", "equation prompt");
    CHECK(si::content_prompt("image") == "\nImage Analysis:", "image prompt");
    CHECK(si::content_prompt("chart") == "\nImage Analysis:", "chart prompt");
    CHECK(si::content_prompt("text") == "\nText Recognition:", "text prompt");
}

void test_parser() {
    const std::string text =
        "<|box_start|>100 200 300 400<|box_end|><|ref_start|>text<|ref_end|>hello"
        "<|box_start|>10 20 30 40<|box_end|><|ref_start|>table<|ref_end|><|rotate_right|>cells";
    const std::vector<si::mineru_block> b = si::mineru_parse_layout(text);
    CHECK(b.size() == 2, "two blocks: %zu", b.size());
    if (b.size() == 2) {
        CHECK(b[0].type == "text", "type: %s", b[0].type.c_str());
        CHECK(near(b[0].bbox[0], 0.1f) && near(b[0].bbox[1], 0.2f) && near(b[0].bbox[2], 0.3f) &&
                  near(b[0].bbox[3], 0.4f),
              "bbox normalization");
        CHECK(b[0].angle == 0, "no rotation");
        CHECK(b[1].type == "table", "second type: %s", b[1].type.c_str());
        CHECK(b[1].angle == 90, "rotate_right: %d", b[1].angle);
    }
}

void test_rotation_and_merge() {
    const std::string text =
        "<|box_start|>0 0 10 10<|box_end|><|ref_start|>text<|ref_end|><|rotate_down|>txt_contd_tgt"
        "<|box_start|>0 0 10 10<|box_end|><|ref_start|>text<|ref_end|><|rotate_left|>other";
    const std::vector<si::mineru_block> b = si::mineru_parse_layout(text);
    CHECK(b.size() == 2, "two rotated blocks: %zu", b.size());
    if (b.size() == 2) {
        CHECK(b[0].angle == 180, "rotate_down: %d", b[0].angle);
        CHECK(b[0].merge_prev, "txt_contd_tgt sets merge_prev");
        CHECK(b[1].angle == 270, "rotate_left: %d", b[1].angle);
        CHECK(!b[1].merge_prev, "other text does not merge");
    }
}

void test_filtering() {
    const std::string text =
        "<|box_start|>0 0 10 10<|box_end|><|ref_start|>inline_formula<|ref_end|>x"
        "<|box_start|>0 0 10 10<|box_end|><|ref_start|>not_a_type<|ref_end|>x"
        "<|box_start|>1001 0 10 10<|box_end|><|ref_start|>text<|ref_end|>x"
        "<|box_start|>0 0 0 10<|box_end|><|ref_start|>text<|ref_end|>x"
        "<|box_start|>0 0 10 10<|box_end|><|ref_start|>unknown<|ref_end|>x";
    const std::vector<si::mineru_block> b = si::mineru_parse_layout(text);
    CHECK(b.size() == 1, "only the unknown->image block survives: %zu", b.size());
    if (!b.empty()) {
        CHECK(b[0].type == "image", "unknown maps to image: %s", b[0].type.c_str());
    }
}

void test_iou_dedup() {
    const std::string text =
        "<|box_start|>100 200 300 400<|box_end|><|ref_start|>text<|ref_end|>a"
        "<|box_start|>101 201 301 401<|box_end|><|ref_start|>text<|ref_end|>b"
        "<|box_start|>100 200 300 400<|box_end|><|ref_start|>text<|ref_end|><|rotate_right|>c";
    const std::vector<si::mineru_block> b = si::mineru_parse_layout(text);
    CHECK(b.size() == 2, "same type+angle within 0.9 IoU is suppressed; a different angle is kept: %zu", b.size());
}

void test_known_type_and_iou() {
    CHECK(si::known_type("text"), "text is known");
    CHECK(si::known_type("table"), "table is known");
    CHECK(si::known_type("unknown"), "unknown is known (mapped to image)");
    CHECK(!si::known_type("not_a_type"), "an unknown label is rejected");

    const float a[4] = {0, 0, 10, 10};
    const float b[4] = {0, 0, 10, 10};
    const float c[4] = {20, 20, 30, 30};
    const float d[4] = {5, 5, 15, 15};
    CHECK(near(si::iou(a, b), 1.0f), "identical boxes: %f", si::iou(a, b));
    CHECK(near(si::iou(a, c), 0.0f), "disjoint boxes: %f", si::iou(a, c));
    CHECK(near(si::iou(a, d), 25.0f / 175.0f), "quarter overlap: %f", si::iou(a, d));
}

} // namespace

int main() {
    test_prompts();
    test_parser();
    test_rotation_and_merge();
    test_filtering();
    test_iou_dedup();
    test_known_type_and_iou();
    if (g_fail) {
        fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_mineru_layout: OK\n");
    return 0;
}
