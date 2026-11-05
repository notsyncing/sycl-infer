#include "layout.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>
#include <vector>

namespace si {

std::string layout_prompt() {
    return "\nLayout Detection:";
}

std::string content_prompt(const std::string & type) {
    if (type == "table") {
        return "\nTable Recognition:";
    }
    if (type == "equation") {
        return "\nFormula Recognition:";
    }
    if (type == "image" || type == "chart") {
        return "\nImage Analysis:";
    }
    return "\nText Recognition:";
}

namespace {

const std::regex & layout_re() {
    static const std::regex re(
        R"(<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(?:(<\|rotate_(?:up|right|down|left)\|>))?([\s\S]*?)(?=<\|box_start\|>|$))",
        std::regex::ECMAScript);
    return re;
}

} // namespace

float iou(const float a[4], const float b[4]) {
    const float x1 = std::max(a[0], b[0]), y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]), y2 = std::min(a[3], b[3]);
    if (x2 <= x1 || y2 <= y1) {
        return 0.f;
    }
    const float in = (x2 - x1) * (y2 - y1);
    const float aa = (a[2] - a[0]) * (a[3] - a[1]);
    const float bb = (b[2] - b[0]) * (b[3] - b[1]);
    return in / (aa + bb - in);
}

bool known_type(const std::string & t) {
    static const char * const kTypes[] = {
        "text", "title", "doc_title", "paragraph_title", "table", "equation", "formula_number", "code", "algorithm",
        "aside_text", "ref_text", "index", "phonetic", "list_item", "caption", "table_caption", "image_caption",
        "code_caption", "footnote", "table_footnote", "image_footnote", "header", "footer", "page_number",
        "page_footnote", "image", "chart", "list", "image_block", "equation_block", "unknown"};
    for (const char * k : kTypes) {
        if (t == k) {
            return true;
        }
    }
    return false;
}

std::vector<mineru_block> mineru_parse_layout(const std::string & text) {
    std::vector<mineru_block> blocks;
    for (std::sregex_iterator it(text.begin(), text.end(), layout_re()), end; it != end; ++it) {
        const std::smatch & m = *it;
        int c[4];
        bool bad = false;
        for (int i = 0; i < 4; i++) {
            c[i] = std::stoi(m[i + 1].str());
            if (c[i] < 0 || c[i] > 1000) {
                bad = true;
            }
        }
        if (bad) {
            continue;
        }
        int x1 = std::min(c[0], c[2]), x2 = std::max(c[0], c[2]);
        int y1 = std::min(c[1], c[3]), y2 = std::max(c[1], c[3]);
        if (x1 == x2 || y1 == y2) {
            continue;
        }
        std::string type = m[5].str();
        std::transform(type.begin(), type.end(), type.begin(), [](unsigned char ch) { return (char)std::tolower(ch); });
        if (type == "unknown") {
            type = "image";
        }
        if (type == "inline_formula" || !known_type(type)) {
            continue;
        }
        mineru_block b;
        b.type = type;
        b.bbox[0] = x1 / 1000.f;
        b.bbox[1] = y1 / 1000.f;
        b.bbox[2] = x2 / 1000.f;
        b.bbox[3] = y2 / 1000.f;
        const std::string tail = m[7].str();
        const std::string rot = m[6].str();
        if (rot == "<|rotate_right|>") {
            b.angle = 90;
        } else if (rot == "<|rotate_down|>") {
            b.angle = 180;
        } else if (rot == "<|rotate_left|>") {
            b.angle = 270;
        }
        b.merge_prev = type == "text" && tail.find("txt_contd_tgt") != std::string::npos;
        // IoU-suppress the same type at the same angle (first occurrence wins)
        bool dup = false;
        for (const mineru_block & k : blocks) {
            if (k.type == b.type && k.angle == b.angle && iou(k.bbox, b.bbox) > 0.9f) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            blocks.push_back(b);
        }
    }
    return blocks;
}

} // namespace si
