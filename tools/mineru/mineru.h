#pragma once
// ---------------------------------------------------------------------------
// MinerU2.5 two-step document extraction on top of the Qwen2-VL trunk.
//
// This mirrors opendatalab/mineru-vl-utils' MinerUClient.two_step_extract:
//   1. the whole page is resized to 1036x1036 and asked for a *layout* with
//      the prompt "\nLayout Detection:";
//   2. every returned block is cropped from the original page, optionally
//      rotated, and asked for its content with the type-specific prompt
//      ("\nText Recognition:", "\nTable Recognition:", "\nFormula
//      Recognition:", "\nImage Analysis:").  Image/chart blocks are always
//      extracted (there is no type filter for them).
// The prompts and the sampling parameters (greedy with presence/frequency
// penalties) come from that project; the block grammar and the JSON shape are
// the same, so existing MinerU consumers can read the output.
//
// This is a client/reference tool (sycl-infer-mineru), not a server pipeline:
// its steps need only the public image-input/text-output model capability.
// The pure prompt/grammar half lives in layout.h.
// ---------------------------------------------------------------------------
#include <cstdio>
#include <string>

namespace si {

struct engine;

struct mineru_options {
    int layout_side = 1036;
    int max_tokens = 4096;
    bool debug = false;
    std::string only_type; // debug: run only one block type (empty = all)
};

// Run the full two-step extraction.  Returns 0 on success.  The JSON array of
// blocks (type, bbox, angle, content) is written to `out`.
int run_mineru(engine & e, const std::string & mmproj_path, const std::string & image_path,
               const mineru_options & opt, FILE * out);

} // namespace si
