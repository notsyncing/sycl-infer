// Hermetic oracle for the grouped dp4a expert GEMM (`moe_grp_impl`, reached
// through `moe_grp_down_launch`) on a Q6_K bank.
//
// WHY THIS EXISTS.  `test_moe_kernels` covers the fp32 expert path and
// `test_moe_tiled` the tiled prefill GEMM; the grouped dp4a path - the one a
// quantized model actually runs - had no oracle at all, and that is how a real
// defect lived here: the Q6_K extractor (`moe_dp4a_w` / `moe_grp_sc`) assumed a
// 32-lane split (`lane << 2` over 128 bytes, `lane >> 4` for the 16-value scale)
// while the kernel hands it an 8-lane sub-group lane, and it ignored the
// superblock's second 128-value half (`ql+64` / `qh+32` / `scales+8` for groups
// 4..7).  On Qwen3.6-35B-A3B (Q6_K expert down on 3 of 40 layers) that measured
// a mean 0.85 / max 1.59 nat next-token logprob error against the CPU reference
// with exact weights - enough to drop a markdown `**` and to write `3` for `3D`
// while the prose stayed fluent, i.e. exactly the class an argmax/token-stream
// oracle cannot see.
//
// Q6_K is the awkward type: a 32-value group is 16 bytes of `ql` (one nibble
// per value, the high nibble belonging to the value 64 later) plus 8 bytes of
// `qh` (two bits per value) and TWO 16-value scales, and the superblock is two
// 128-value halves.  A zero-error comparison against `quant.h`'s own dequantizer
// is the only thing that distinguishes "the packing is right" from "the numbers
// look plausible".
//
// Oracle: every weight the kernel reads is dequantized by `quant.h`, multiplied
// by the int8 activation times the f16 per-32-group scale that
// `moe_actq_launch` itself produced, and summed in double for the exact
// (pair, row) the kernel writes.  The per-pair expert comes from the same CSR
// the kernel walks.
//
// Usage: test_moe_grp_q6 [n_expert] [n_pairs] [n_embd] [n_ff_exp]
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "kernels.h"
#include "quant.h" // block formats, ggml_float_to_half, dequantize_block_q6_K

using namespace si;
using namespace sycl;

namespace {

int g_fails = 0;

void fill_block_q6(block_q6_K & blk, std::mt19937 & rng) {
    // d is a power of two and the 16-value scales are small integers, so every
    // dequantized weight is exact in fp32 and the reference's only error is the
    // double-vs-fp32 accumulation order (gated below at 1e-5 relative).
    const uint16_t d = ggml_float_to_half(ldexpf(1.f, -10));
    std::memcpy(&blk.d, &d, sizeof(d));
    for (int i = 0; i < QK_K / 2; i++) {
        blk.ql[i] = (uint8_t)(rng() & 0xFF);
    }
    for (int i = 0; i < QK_K / 4; i++) {
        blk.qh[i] = (uint8_t)(rng() & 0xFF);
    }
    for (int i = 0; i < QK_K / 16; i++) {
        blk.scales[i] = (int8_t)(1 + (int)(rng() % 48));
    }
}

} // namespace

int main(int argc, char ** argv) {
    const int n_expert = argc > 1 ? atoi(argv[1]) : 4;
    const int n_pairs = argc > 2 ? atoi(argv[2]) : 3;
    const int n_embd = argc > 3 ? atoi(argv[3]) : 2048;  // N of the down projection
    const int n_ff = argc > 4 ? atoi(argv[4]) : 512;     // K of the down projection
    if (n_pairs > n_expert || n_pairs <= 0) {
        printf("test bug: n_pairs must be in 1..n_expert\n");
        return 1;
    }
    if ((n_ff % 256) != 0 || (n_embd % 32) != 0) {
        printf("test bug: K must be a multiple of 256 and N of 32\n");
        return 1;
    }

    queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};

    // ---- Q6_K expert bank: `n_expert` slots of `n_embd` rows of `n_ff` values.
    const int bs = (int)sizeof(block_q6_K);
    const int nsb = n_ff / 256;
    const size_t row_bytes = (size_t)nsb * (size_t)bs;
    const size_t slot_bytes = (size_t)n_embd * row_bytes + 64; // deliberately not tight
    std::vector<char> h_bank((size_t)n_expert * slot_bytes, 0);
    std::mt19937 rng(20261109);
    for (int e = 0; e < n_expert; e++) {
        for (int n = 0; n < n_embd; n++) {
            for (int sb = 0; sb < nsb; sb++) {
                char * p = h_bank.data() + (size_t)e * slot_bytes + (size_t)n * row_bytes + (size_t)sb * bs;
                fill_block_q6(*reinterpret_cast<block_q6_K *>(p), rng);
            }
        }
    }
    // The reference weights, straight out of quant.h.
    std::vector<float> h_w((size_t)n_expert * n_embd * n_ff, 0.f);
    {
        std::vector<float> tmp(256);
        for (int e = 0; e < n_expert; e++) {
            for (int n = 0; n < n_embd; n++) {
                for (int sb = 0; sb < nsb; sb++) {
                    const char * p =
                        h_bank.data() + (size_t)e * slot_bytes + (size_t)n * row_bytes + (size_t)sb * bs;
                    dequantize_block_q6_K(reinterpret_cast<const block_q6_K *>(p), tmp.data());
                    std::memcpy(&h_w[((size_t)e * n_embd + n) * n_ff + sb * 256], tmp.data(), 256 * sizeof(float));
                }
            }
        }
    }

    // ---- CSR: one pair per expert for the first `n_pairs` experts.
    // The grouped kernel dereferences `slot_map[e]` unconditionally (nullptr is
    // not "identity" here - it is a device read of address 0), so hand it one.
    std::vector<int32_t> h_off((size_t)n_expert + 1, 0), h_pair((size_t)n_pairs), h_active, h_nact(1);
    std::vector<int32_t> h_slot((size_t)n_expert);
    for (int e = 0; e < n_expert; e++) h_slot[(size_t)e] = e;
    for (int e = 0; e < n_expert; e++) {
        h_off[(size_t)e + 1] = h_off[(size_t)e] + (e < n_pairs ? 1 : 0);
        if (e < n_pairs) h_active.push_back(e);
    }
    for (int p = 0; p < n_pairs; p++) h_pair[(size_t)p] = p;
    h_nact[0] = n_pairs;

    // ---- activation rows (one per pair) and the engine's own quantizer.
    std::vector<float> h_x((size_t)n_pairs * n_ff);
    {
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        for (auto & v : h_x) v = uni(rng);
    }
    std::vector<int8_t> h_xq((size_t)n_pairs * n_ff);
    std::vector<uint16_t> h_asa((size_t)n_pairs * (n_ff / 32));
    std::vector<float> h_xs((size_t)n_pairs * (n_ff / 32));

    // ---- device buffers
    void * d_bank = malloc_device(h_bank.size(), q);
    q.memcpy(d_bank, h_bank.data(), h_bank.size()).wait();
    int8_t * d_xq = malloc_device<int8_t>(h_xq.size(), q);
    uint16_t * d_asa = malloc_device<uint16_t>(h_asa.size(), q);
    float * d_xs = malloc_device<float>(h_xs.size(), q);
    int32_t * d_off = malloc_device<int32_t>(h_off.size(), q);
    int32_t * d_pair = malloc_device<int32_t>(h_pair.size(), q);
    int32_t * d_active = malloc_device<int32_t>(h_active.size(), q);
    int32_t * d_nact = malloc_device<int32_t>(h_nact.size(), q);
    int32_t * d_slot = malloc_device<int32_t>(h_slot.size(), q);
    float * d_acc = malloc_device<float>((size_t)n_pairs * n_embd, q);
    float * d_x = malloc_device<float>(h_x.size(), q);
    q.memcpy(d_x, h_x.data(), h_x.size() * sizeof(float)).wait();
    q.memcpy(d_off, h_off.data(), h_off.size() * sizeof(int32_t)).wait();
    q.memcpy(d_pair, h_pair.data(), h_pair.size() * sizeof(int32_t)).wait();
    q.memcpy(d_active, h_active.data(), h_active.size() * sizeof(int32_t)).wait();
    q.memcpy(d_nact, h_nact.data(), h_nact.size() * sizeof(int32_t)).wait();
    q.memcpy(d_slot, h_slot.data(), h_slot.size() * sizeof(int32_t)).wait();
    q.memset(d_acc, 0, (size_t)n_pairs * n_embd * sizeof(float)).wait();

    moe_actq_launch(q, d_x, n_ff, n_pairs, n_ff, d_xq, d_asa, d_xs);
    q.wait();
    q.memcpy(h_xq.data(), d_xq, h_xq.size()).wait();
    q.memcpy(h_asa.data(), d_asa, h_asa.size() * sizeof(uint16_t)).wait();

    // ---- the kernel under test (Q6_K down: K = n_ff, N = n_embd, dec = false)
    const bool ok = moe_grp_down_launch(q, d_bank, d_slot, slot_bytes, d_off, d_pair, d_active, d_nact,
                                        n_pairs /*n_act_max*/, n_expert, n_pairs, d_xq, d_asa, d_xs, d_acc,
                                        n_embd /*acc_stride*/, n_embd, n_ff, 14, false);
    q.wait();
    if (!ok) {
        printf("grouped Q6_K down: FAIL  launcher declined (K=%d N=%d type=14)\n", n_ff, n_embd);
        return 1;
    }
    std::vector<float> h_acc((size_t)n_pairs * n_embd, 0.f);
    q.memcpy(h_acc.data(), d_acc, h_acc.size() * sizeof(float)).wait();

    // ---- reference in double, for the exact (pair, row) the kernel writes
    // The gate is |diff| against the ACCUMULATION magnitude (sum |a*w|), not
    // against |ref|: a dot product whose terms nearly cancel still differs by
    // its rounding error, which is what a per-element relative yardstick
    // measures (the same trap test_w4_vs_i8 records).  Measured here: the
    // fp32-vs-double order alone is 3.5e-6 absolute, and the pre-fix weight
    // extraction is O(10) - eight orders of magnitude apart.
    double max_abs = 0.0, max_q = 0.0;
    for (int p = 0; p < n_pairs; p++) {
        // the CSR says which expert owns pair p
        int e = -1;
        for (int x = 0; x < n_expert; x++) {
            if (p >= h_off[(size_t)x] && p < h_off[(size_t)x + 1]) e = x;
        }
        for (int n = 0; n < n_embd; n++) {
            double ref = 0.0, scale = 0.0;
            for (int k = 0; k < n_ff; k++) {
                const double a = (double)h_xq[(size_t)p * n_ff + k] *
                                 (double)ggml_half_to_float(h_asa[(size_t)p * (n_ff / 32) + k / 32]);
                const double w = (double)h_w[((size_t)e * n_embd + n) * n_ff + k];
                ref += a * w;
                scale += std::fabs(a * w);
            }
            const double got = h_acc[(size_t)p * n_embd + n];
            const double d = std::fabs(got - ref);
            max_abs = std::max(max_abs, d);
            max_q = std::max(max_q, d / std::max(scale, 1e-12));
        }
    }
    const bool pass = max_q <= 1e-6;
    printf("grouped Q6_K down: %s  max|diff|=%.3e  max|d|/sum|a*w|=%.3e  (%d experts, %d pairs, K=%d, N=%d)\n",
           pass ? "OK" : "FAIL", max_abs, max_q, n_expert, n_pairs, n_ff, n_embd);
    if (!pass) {
        printf("  a Q6_K group is 16 ql bytes + 8 qh bytes + two 16-value scales, and a\n"
               "  superblock is two 128-value halves: check moe_dp4a_w/moe_grp_sc's\n"
               "  half offset (ql+64/qh+32/scales+8) and the 8-lane scale selector.\n");
        g_fails++;
    }

    free(d_bank, q);
    free(d_xq, q);
    free(d_asa, q);
    free(d_xs, q);
    free(d_off, q);
    free(d_pair, q);
    free(d_active, q);
    free(d_nact, q);
    free(d_slot, q);
    free(d_acc, q);
    free(d_x, q);
    return g_fails == 0 ? 0 : 1;
}
