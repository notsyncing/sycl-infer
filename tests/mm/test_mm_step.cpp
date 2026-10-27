// Hermetic check of the per-token M-RoPE / image-row slice that both prefill
// paths use (the generic chunk loop and qwen4exp's own chunker).  No model, no
// GPU: the slice is a pure copy of an expanded prompt.
//
// The positions below encode the Qwen2.5-VL layout: a 2x2 merged image consumes
// max(2,2) = 2 text positions while contributing four tokens.  Token 1..4 are
// that image; the temporal position stays at its base and the row/col sections
// walk the grid.
#include <cstdio>
#include <stdexcept>

#include "multimodal.h"

namespace {

bool eq(const int32_t * got, const int32_t * want, int n, const char * what) {
    for (int i = 0; i < n; i++) {
        if (got[i] != want[i]) {
            std::fprintf(stderr, "mm step: %s[%d] = %d, want %d\n", what, i, (int)got[i], (int)want[i]);
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    si::mm_prompt p;
    p.tokens = {100, 200, 200, 200, 200, 300}; // text, <|image_pad|> x4, text
    p.n_img = 1;
    p.pos_after = 3;
    // section-major: [4][6]
    p.mrope = {
        0, 1, 1, 1, 1, 2, // temporal: image tokens share the block base
        0, 1, 1, 2, 2, 2, // row
        0, 1, 2, 1, 2, 2, // col
        0, 1, 1, 1, 1, 2, // unused 4th section == temporal
    };
    p.img_row = {-1, 0, 1, 2, 3, -1};

    // a prompt the planners built but phase 2 never filled must be refused by
    // generate_mm, not fed to the forward as uninitialised rows
    bool ok = !p.has_encoded_embd();
    p.embd.assign(4, 0.0f); // placeholder rows (the check only asks that they exist)
    ok = ok && p.has_encoded_embd();

    constexpr int kStride = 8;
    int32_t m[4 * kStride] = {};
    int32_t r[kStride] = {};

    // one whole chunk
    si::mm_write_step(p, 0, 6, kStride, m, r);
    const int32_t t0[] = {0, 1, 1, 1, 1, 2};
    const int32_t h0[] = {0, 1, 1, 2, 2, 2};
    const int32_t w0[] = {0, 1, 2, 1, 2, 2};
    const int32_t r0[] = {-1, 0, 1, 2, 3, -1};
    ok = ok && eq(m + 0 * kStride, t0, 6, "temporal") && eq(m + 1 * kStride, h0, 6, "row") &&
         eq(m + 2 * kStride, w0, 6, "col") && eq(m + 3 * kStride, t0, 6, "section3") &&
         eq(r, r0, 6, "img_row");

    // a chunk boundary inside the image block: the slice must stay per-token,
    // not restart at the block's first row
    si::mm_write_step(p, 3, 2, kStride, m, r);
    const int32_t t1[] = {1, 1};
    const int32_t h1[] = {2, 2};
    const int32_t w1[] = {1, 2};
    const int32_t r1[] = {2, 3};
    ok = ok && eq(m + 0 * kStride, t1, 2, "slice temporal") && eq(m + 1 * kStride, h1, 2, "slice row") &&
         eq(m + 2 * kStride, w1, 2, "slice col") && eq(m + 3 * kStride, t1, 2, "slice section3") &&
         eq(r, r1, 2, "slice img_row");

    // out-of-range slice and a too-small section stride must be errors, not
    // reads past the prompt (a chunker bug used to be exactly this)
    bool threw = false;
    try {
        si::mm_write_step(p, 5, 2, kStride, m, r);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    ok = ok && threw;
    threw = false;
    try {
        si::mm_write_step(p, 0, 4, 3, m, r);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    ok = ok && threw;

    if (!ok) {
        return 1;
    }
    std::printf("mm step: all slices OK\n");
    return 0;
}
