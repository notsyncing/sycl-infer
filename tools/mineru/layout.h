#pragma once
// The pure half of the MinerU two-step client: the upstream prompts, the
// layout block grammar and its geometry rules.  No engine, no SYCL, no HTTP,
// so it can be unit-tested on its own.
#include <string>
#include <vector>

namespace si {

struct mineru_block {
    std::string type;
    float bbox[4] = {0, 0, 0, 0};
    int angle = 0; // 0/90/180/270
    bool merge_prev = false;
    std::string content;
};

std::string layout_prompt();
std::string content_prompt(const std::string & type);

// Parse the model's `box_start/ref_start/rotate` grammar into blocks.
std::vector<mineru_block> mineru_parse_layout(const std::string & text);

bool known_type(const std::string & type);
float iou(const float a[4], const float b[4]);

} // namespace si
