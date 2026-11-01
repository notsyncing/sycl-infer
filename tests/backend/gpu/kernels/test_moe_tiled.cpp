// Hermetic oracle for the tiled prefill expert GEMM (`moe_tiled.{h,cpp}`).
//
// Its only coverage used to be the 35B end-to-end prefill, which can only say
// "the text is wrong" - and the kernel is almost entirely wiring: it reads the
// native GGUF blocks where they lie (a CSR pair table picks the expert and the
// activation row), unpacks one 32-value group per k-step into SLM, and writes
// `pair[p] * out_stride + column`.  An unpack or an index mistake is invisible
// in an aggregate but is exactly what this test localises.
//
// Oracle: `quant.h`'s own dequantizer (the engine's block layout, and the
// packing the kernel hand-rolls in `q4k_scale_min`) dotted in double against
// the int8 activation times the f16 per-32-group scale that
// `moe_actq_launch` actually produced, for the exact (expert, pair, row) the
// kernel writes.
//
// Why the numbers are powers of two: the super-block d/dmin are drawn from
// {2^-10, 2^-11} and every activation 32-group is seeded with +-127 so
// `moe_actq_launch` resolves its scale to exactly 2^-10.  Every weight, every
// activation value and every partial sum is then exactly representable in
// fp32, so the double reference and the kernel dot agree to 0 and a nonzero
// error is an unpacking/indexing fault rather than the summation order of a
// K=2048 fp32 dot (a random f16 d/dmin alone puts ~2.5e-6 of pure accumulation
// noise on that comparison, which would make the 2e-6 gate meaningless).
//
// What is covered: Q4_K gate (top_k > 0, no gate), Q4_K up (top_k > 0, gate
// non-null, written in place), Q4_K down (top_k == 0) and Q5_K down; a CSR
// with an uneven pair split (40 / 4 / 4), an expert with zero pairs, an expert
// past `n_active[0]`, a non-resident (slot -1) expert, a non-identity
// slot_map, `pair[]` as a permutation, padded lanes past `np` (they must not
// write, checked with sentinel guard rows) and both tile shapes the audit
// flagged: a short tail (4 and 15 pairs) and two tiles (33, 40 pairs).
//
// Usage: ./test_moe_tiled            (add 'time' to repeat the T-MAC/s smoke)

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "kernels.h"
#include "moe_tiled.h"
#include "quant.h"

using namespace si;
using namespace sycl;

namespace {

constexpr float kSentinel = -12345.25f;
constexpr int kGuard = 8; // sentinel rows before and after the output
int g_fails = 0;

// The inverse of quant.h's get_scale_min_k4, for the eight 32-value sub-blocks
// of one super-block.  Packing here and decoding through quant.h is what keeps
// the test independent of the kernel's own six-bit unpacker.
void pack_scales(const int sc[8], const int mn[8], uint8_t q[12]) {
    for (int j = 0; j < 4; j++) {
        q[j + 0] = (uint8_t)((sc[j + 0] & 63) | ((sc[j + 4] >> 4) << 6));
        q[j + 4] = (uint8_t)((mn[j + 0] & 63) | ((mn[j + 4] >> 4) << 6));
        q[j + 8] = (uint8_t)((sc[j + 4] & 15) | ((mn[j + 4] & 15) << 4));
    }
}

// One expert bank: `n_slots` slots of `N` rows, each row `K/256` GGUF blocks.
// `slot_stride` is deliberately not `N * rb`, so a kernel that ignores it (or
// assumes slots are contiguous) is caught.
struct bank {
    uint32_t type = 12;
    int K = 0;
    int N = 0;
    int n_slots = 0;
    size_t rb = 0;           // bytes per output row (ggml_row_bytes)
    size_t slot_stride = 0;  // bytes per expert slot
    std::vector<char> host;
    std::vector<float> w; // [slot][row][K], quant.h's own dequantization
    void * dev = nullptr;

    float weight(int slot, int row, int k) const { return w[((size_t)slot * N + row) * K + k]; }
};

bank make_bank(queue & q, uint32_t type, int K, int N, int n_slots, uint32_t seed) {
    bank b;
    b.type = type;
    b.K = K;
    b.N = N;
    b.n_slots = n_slots;
    const int bs = (type == 12) ? (int)sizeof(block_q4_K) : (int)sizeof(block_q5_K);
    const int nsb = K / 256;
    b.rb = (size_t)nsb * bs;
    b.slot_stride = (size_t)N * b.rb + 64;
    b.host.assign((size_t)n_slots * b.slot_stride, 0);

    std::mt19937 rng(seed);
    const uint16_t dpow[2] = {ggml_float_to_half(ldexpf(1.f, -10)), ggml_float_to_half(ldexpf(1.f, -11))};
    int bad_pack = 0;
    for (int s = 0; s < n_slots; s++) {
        for (int r = 0; r < N; r++) {
            for (int sb = 0; sb < nsb; sb++) {
                char * blk = b.host.data() + (size_t)s * b.slot_stride + (size_t)r * b.rb + (size_t)sb * bs;
                const uint16_t d = dpow[rng() & 1], dm = dpow[rng() & 1];
                std::memcpy(blk + 0, &d, 2);
                std::memcpy(blk + 2, &dm, 2);
                int sc[8], mn[8];
                for (int j = 0; j < 8; j++) {
                    sc[j] = 1 + (int)(rng() % 63); // never 0: no all-zero group
                    mn[j] = (int)(rng() % 64);
                }
                uint8_t scales[12];
                pack_scales(sc, mn, scales);
                std::memcpy(blk + 4, scales, 12);
                for (int j = 0; j < 8; j++) {
                    uint8_t gs = 0, gm = 0;
                    get_scale_min_k4(j, scales, &gs, &gm);
                    if (gs != (uint8_t)sc[j] || gm != (uint8_t)mn[j]) bad_pack++;
                }
                const int qs_off = (type == 12) ? 16 : 48;
                for (int i = 0; i < 128; i++) blk[qs_off + i] = (char)(rng() & 0xFF);
                if (type == 13) {
                    for (int i = 0; i < 32; i++) blk[16 + i] = (char)(rng() & 0xFF);
                }
            }
        }
    }
    if (bad_pack) {
        printf("test bug: pack_scales/get_scale_min_k4 disagree on %d sub-blocks\n", bad_pack);
        g_fails++;
    }

    b.w.assign((size_t)n_slots * N * K, 0.f);
    std::vector<float> tmp(256);
    for (int s = 0; s < n_slots; s++) {
        for (int r = 0; r < N; r++) {
            for (int sb = 0; sb < nsb; sb++) {
                const char * blk =
                    b.host.data() + (size_t)s * b.slot_stride + (size_t)r * b.rb + (size_t)sb * bs;
                if (type == 12) {
                    dequantize_block_q4_K((const block_q4_K *)blk, tmp.data());
                } else {
                    dequantize_block_q5_K((const block_q5_K *)blk, tmp.data());
                }
                std::memcpy(&b.w[((size_t)s * N + r) * K + sb * 256], tmp.data(), 256 * sizeof(float));
            }
        }
    }
    b.dev = malloc_device(b.host.size(), q);
    q.memcpy(b.dev, b.host.data(), b.host.size()).wait();
    return b;
}

// int8 activation plus its f16 per-32-group scale, produced by the engine's
// own quantizer so the test does not restate its rounding.
struct activation {
    int n_rows = 0;
    int K = 0;
    std::vector<int8_t> xq;
    std::vector<uint16_t> asa;
    int8_t * d_xq = nullptr;
    uint16_t * d_asa = nullptr;
};

activation make_activation(queue & q, int n_rows, int K, uint32_t seed) {
    const float kQ = ldexpf(1.f, -10); // 2^-10
    std::mt19937 rng(seed);
    std::vector<float> x((size_t)n_rows * K);
    for (auto & v : x) v = (float)((int)(rng() % 255) - 127) * kQ;
    // Force the group maximum to exactly 127*2^-10, so moe_actq_launch's
    // `mx / 127` is exactly 2^-10 and the scale's f16 store is exact.
    for (int r = 0; r < n_rows; r++) {
        for (int g = 0; g < K / 32; g++) {
            const int at = g * 32 + (int)(rng() % 32);
            x[(size_t)r * K + at] = ((rng() & 1) ? 127.f : -127.f) * kQ;
        }
    }
    float * d_x = malloc_device<float>(x.size(), q);
    float * d_xs = malloc_device<float>((size_t)n_rows * (K / 32), q);
    q.memcpy(d_x, x.data(), x.size() * sizeof(float)).wait();

    activation a;
    a.n_rows = n_rows;
    a.K = K;
    a.xq.resize((size_t)n_rows * K);
    a.asa.resize((size_t)n_rows * (K / 32));
    a.d_xq = malloc_device<int8_t>(a.xq.size(), q);
    a.d_asa = malloc_device<uint16_t>(a.asa.size(), q);
    moe_actq_launch(q, d_x, K, n_rows, K, a.d_xq, a.d_asa, d_xs);
    q.memcpy(a.xq.data(), a.d_xq, a.xq.size()).wait();
    q.memcpy(a.asa.data(), a.d_asa, a.asa.size() * sizeof(uint16_t)).wait();
    free(d_x, q);
    free(d_xs, q);

    int bad = 0;
    for (uint16_t bits : a.asa) {
        if (ggml_half_to_float(bits) != kQ) bad++;
    }
    printf("activation %dx%d: %s (%d group scales != 2^-10)\n", n_rows, K, bad ? "FAIL" : "OK  ", bad);
    if (getenv("MOE_TILED_ROWDUMP")) {
        for (int r = 0; r < n_rows; r++) {
            long nz = 0;
            for (int k = 0; k < K; k++) nz += a.xq[(size_t)r * K + k] != 0;
            printf("  act row %2d: nz=%4ld xq0=%4d asa0=%.6g\n", r, nz, (int)a.xq[(size_t)r * K],
                   (double)ggml_half_to_float(a.asa[(size_t)r * (K / 32)]));
        }
    }
    if (bad) g_fails++;
    return a;
}

struct csr {
    int n_expert = 0;
    int n_pairs = 0;
    int n_active = 0;
    int n_act_max = 0;
    int top_k = 0;
    std::vector<int32_t> off, pair, active, slot_map;
};

struct dcsr {
    int32_t *off = nullptr, *pair = nullptr, *active = nullptr, *nact = nullptr, *slot = nullptr;
};

dcsr upload_csr(queue & q, const csr & c) {
    dcsr d;
    d.off = malloc_device<int32_t>(c.n_expert + 1, q);
    d.pair = malloc_device<int32_t>(c.n_pairs > 0 ? c.n_pairs : 1, q);
    d.active = malloc_device<int32_t>(c.n_act_max > 0 ? c.n_act_max : 1, q);
    d.nact = malloc_device<int32_t>(1, q);
    d.slot = malloc_device<int32_t>(c.n_expert, q);
    std::vector<int32_t> act((size_t)std::max(c.n_act_max, 1), 0);
    for (size_t i = 0; i < c.active.size(); i++) act[i] = c.active[i];
    const int32_t na = c.n_active;
    q.memcpy(d.off, c.off.data(), (c.n_expert + 1) * sizeof(int32_t));
    q.memcpy(d.pair, c.pair.data(), c.n_pairs * sizeof(int32_t));
    q.memcpy(d.active, act.data(), act.size() * sizeof(int32_t));
    q.memcpy(d.nact, &na, sizeof(int32_t));
    q.memcpy(d.slot, c.slot_map.data(), c.n_expert * sizeof(int32_t));
    q.wait();
    return d;
}

// `n_active` lets a case stop the active list early (the kernel's
// `ia >= n_active[0]` guard); `n_act_max` is the grid size, so it stays at the
// full non-empty expert count.
csr make_csr(const std::vector<int> & sizes, int top_k, std::vector<int32_t> slot_map, uint32_t seed,
             int n_active = -1) {
    csr c;
    c.n_expert = (int)sizes.size();
    c.top_k = top_k;
    c.slot_map = std::move(slot_map);
    c.off.assign((size_t)c.n_expert + 1, 0);
    for (int e = 0; e < c.n_expert; e++) c.off[e + 1] = c.off[e] + sizes[e];
    c.n_pairs = c.off[(size_t)c.n_expert];
    c.pair.resize((size_t)c.n_pairs);
    std::iota(c.pair.begin(), c.pair.end(), 0);
    // pair[] is a permutation: the CSR position and the output row are then
    // unrelated, which is what makes a `pair[]`-vs-`rows[]` mix-up visible.
    std::shuffle(c.pair.begin(), c.pair.end(), std::mt19937(seed));
    for (int e = 0; e < c.n_expert; e++) {
        if (sizes[e] > 0) c.active.push_back(e);
    }
    c.n_active = n_active < 0 ? (int)c.active.size() : n_active;
    c.n_act_max = (int)c.active.size();
    return c;
}

float silu_ref(float x) { return x / (1.0f + std::exp(-x)); }

// Launch one projection, then compare every row it owns against the scalar
// reference in double, every row it does not own against the pre-launch
// buffer, and every guard row against the sentinel.
void run_projection(const char * what, queue & q, const bank & b, const csr & c, const activation & act,
                    float * d_out_base, int out_stride, float * d_gate) {
    const size_t rows = (size_t)c.n_pairs + 2 * kGuard;
    const size_t nfloat = rows * (size_t)out_stride;
    std::vector<float> before(nfloat), after(nfloat);
    q.memcpy(before.data(), d_out_base, nfloat * sizeof(float)).wait();

    float * out = d_out_base + (size_t)kGuard * out_stride;
    dcsr d = upload_csr(q, c);
    const bool ok = moe_tiled_gemm_launch(q, b.dev, b.slot_stride, d.slot, d.off, d.pair, d.active, d.nact,
                                          c.n_act_max, act.d_xq, act.d_asa, out, out_stride, b.K, b.N, c.top_k,
                                          b.type, d_gate);
    q.wait();
    q.memcpy(after.data(), d_out_base, nfloat * sizeof(float)).wait();
    if (!ok) {
        printf("%-34s FAIL  launcher declined (K=%d N=%d type=%u)\n", what, b.K, b.N, b.type);
        g_fails++;
        return;
    }

    // Which rows must the kernel write, and what is the value?  The gate
    // (up's in-place operand) is read back from the device so the reference
    // sees the exact bytes the epilogue reads.
    const float * gate_host = nullptr;
    if (d_gate != nullptr) {
        if (d_gate != out) {
            printf("%-34s FAIL  test bug: gate must alias out (in-place)\n", what);
            g_fails++;
            return;
        }
        gate_host = before.data() + (size_t)kGuard * out_stride;
    }
    std::vector<char> written((size_t)c.n_pairs, 0);
    std::vector<float> want((size_t)c.n_pairs * out_stride, 0.f);
    const int gstride = b.K / 32;
    for (int ia = 0; ia < c.n_active; ia++) {
        const int e = c.active[(size_t)ia];
        const int slot = c.slot_map[(size_t)e];
        const int np = c.off[(size_t)e + 1] - c.off[(size_t)e];
        if (slot < 0 || np <= 0) continue; // the kernel retires these whole
        for (int pos = c.off[(size_t)e]; pos < c.off[(size_t)e + 1]; pos++) {
            const int pid = c.pair[(size_t)pos];
            const int prow = c.top_k > 0 ? pid / c.top_k : pid;
            written[(size_t)pid] = 1;
            for (int n = 0; n < b.N; n++) {
                double acc = 0.0;
                for (int k = 0; k < b.K; k++) {
                    const double a = (double)act.xq[(size_t)prow * b.K + k] *
                                     (double)ggml_half_to_float(act.asa[(size_t)prow * gstride + k / 32]);
                    acc += a * (double)b.weight(slot, n, k);
                }
                const size_t o = (size_t)pid * out_stride + n;
                want[o] = gate_host ? (float)(silu_ref(gate_host[o]) * acc) : (float)acc;
            }
        }
    }

    if (getenv("MOE_TILED_ROWDUMP")) {
        for (int pid = 0; pid < c.n_pairs; pid++) {
            printf("  rowdump %-30s pid %2d written %d want0 %11.5g got0 %11.5g\n", what, pid, (int)written[(size_t)pid],
                   (double)want[(size_t)pid * out_stride], (double)after[((size_t)kGuard + pid) * out_stride]);
        }
    }
    double worst = 0.0;
    int worst_pid = -1, worst_n = -1, pad_guard = 0, touched_unowned = -1;
    int n_bad_rows = 0;
    std::vector<int> bad_pids;
    for (int pid = 0; pid < c.n_pairs; pid++) {
        const float * got = &after[((size_t)kGuard + pid) * out_stride];
        const float * pre = &before[((size_t)kGuard + pid) * out_stride];
        if (!written[(size_t)pid]) {
            if (std::memcmp(got, pre, (size_t)out_stride * sizeof(float)) != 0 && touched_unowned < 0) {
                touched_unowned = pid;
            }
            continue;
        }
        double rowmax = 1e-30, rowbad = 0.0;
        for (int n = 0; n < out_stride; n++) rowmax = std::max(rowmax, std::fabs((double)want[(size_t)pid * out_stride + n]));
        for (int n = 0; n < out_stride; n++) {
            const double diff = std::fabs((double)got[n] - (double)want[(size_t)pid * out_stride + n]);
            const double r = diff / rowmax;
            rowbad = std::max(rowbad, r);
            if (r > worst) {
                worst = r;
                worst_pid = pid;
                worst_n = n;
            }
        }
        if (rowbad > 2e-6) {
            n_bad_rows++;
            bad_pids.push_back(pid);
        }
    }
    // Guard rows: a lane whose TM rows are all padding has rows[] == -1, and a
    // write at a negative pid lands here (out points kGuard rows in).
    for (int g = 0; g < kGuard; g++) {
        for (int side = 0; side < 2; side++) {
            const size_t row = side == 0 ? (size_t)g : (size_t)(kGuard + c.n_pairs + g);
            if (memcmp(&after[row * out_stride], &before[row * out_stride], (size_t)out_stride * sizeof(float)) != 0 &&
                pad_guard == 0) {
                pad_guard = side == 0 ? -(g + 1) : (g + 1);
            }
        }
    }

    const bool okc = worst <= 2e-6 && pad_guard == 0 && touched_unowned < 0;
    if (!okc) g_fails++;
    printf("%-34s %s  max|d|/rowmax=%.2e", what, okc ? "OK  " : "FAIL", worst);
    if (worst > 2e-6) printf(" (pair %d col %d)", worst_pid, worst_n);
    if (touched_unowned >= 0) printf(" [wrote unowned pair %d]", touched_unowned);
    if (pad_guard != 0) printf(" [pad wrote guard row %d]", pad_guard);
    printf("\n");
    if (worst > 2e-6) {
        printf("   bad rows %d:", n_bad_rows);
        for (int p : bad_pids) printf(" %d", p);
        printf("\n");
    }

    // When a row disagrees, ask *what* the kernel computed: try every expert
    // slot and every plausible activation row (pair id, pair/top_k, CSR
    // position, position/top_k) and report the one that reproduces the bytes.
    if (!okc && worst_pid >= 0) {
        const int pid = worst_pid;
        int pos = -1, own_e = -1;
        for (int e = 0; e < c.n_expert; e++) {
            for (int p = c.off[(size_t)e]; p < c.off[(size_t)e + 1]; p++) {
                if (c.pair[(size_t)p] == pid) {
                    pos = p;
                    own_e = e;
                }
            }
        }
        double rowmax = 1e-30;
        for (int n = 0; n < out_stride; n++)
            rowmax = std::max(rowmax, std::fabs((double)want[(size_t)pid * out_stride + n]));
        printf("   pair %d: expert %d pos %d prow %d rowmax %.4g  want[0..3] %.4g %.4g %.4g %.4g  got %.4g %.4g %.4g %.4g\n",
               pid, own_e, pos, c.top_k > 0 ? pid / c.top_k : pid, rowmax, (double)want[(size_t)pid * out_stride + 0],
               (double)want[(size_t)pid * out_stride + 1], (double)want[(size_t)pid * out_stride + 2],
               (double)want[(size_t)pid * out_stride + 3], (double)after[((size_t)kGuard + pid) * out_stride + 0],
               (double)after[((size_t)kGuard + pid) * out_stride + 1], (double)after[((size_t)kGuard + pid) * out_stride + 2],
               (double)after[((size_t)kGuard + pid) * out_stride + 3]);
        for (int b0 = 0; b0 < b.N; b0 += 256) {
            const int b1 = std::min(b0 + 256, b.N);
            double mx = 0.0;
            int cnt = 0;
            for (int n = b0; n < b1; n++) {
                const double r = std::fabs((double)after[((size_t)kGuard + pid) * out_stride + n] -
                                           (double)want[(size_t)pid * out_stride + n]) /
                                 rowmax;
                mx = std::max(mx, r);
                cnt += r > 2e-6;
            }
            printf("   cols %4d..%4d: max %.2e bad %d\n", b0, b1 - 1, mx, cnt);
        }
        {
            int bp = -1;
            double bv = 1e30;
            for (int pid2 = 0; pid2 < c.n_pairs; pid2++) {
                double mx = 0.0;
                for (int n = 0; n < out_stride; n++) {
                    mx = std::max(mx, std::fabs((double)after[((size_t)kGuard + pid) * out_stride + n] -
                                                (double)want[(size_t)pid2 * out_stride + n]) /
                                           rowmax);
                }
                if (mx < bv) {
                    bv = mx;
                    bp = pid2;
                }
            }
            printf("   best want row: %d -> %.2e\n", bp, bv);
        }
        double best = 1e30;
        int be = -1, bmode = -1;
        for (int e2 = 0; e2 < c.n_expert; e2++) {
            const int s2 = c.slot_map[(size_t)e2];
            if (s2 < 0) continue;
            for (int mode = 0; mode < 4; mode++) {
                const int pr2 = mode == 0 ? (c.top_k > 0 ? pid / c.top_k : pid)
                                : mode == 1 ? pid
                                : mode == 2 ? pos
                                            : (c.top_k > 0 ? pos / c.top_k : pos);
                if (pr2 < 0 || pr2 >= act.n_rows) continue;
                double mx = 0.0;
                for (int n = 0; n < out_stride; n++) {
                    double acc = 0.0;
                    for (int k = 0; k < b.K; k++) {
                        acc += (double)act.xq[(size_t)pr2 * b.K + k] *
                               (double)ggml_half_to_float(act.asa[(size_t)pr2 * gstride + k / 32]) *
                               (double)b.weight(s2, n, k);
                    }
                    const double v =
                        gate_host ? (double)(silu_ref(gate_host[(size_t)pid * out_stride + n]) * acc) : acc;
                    mx = std::max(mx, std::fabs((double)after[((size_t)kGuard + pid) * out_stride + n] - v) / rowmax);
                }
                if (mx < best) {
                    best = mx;
                    be = e2;
                    bmode = mode;
                }
            }
        }
        static const char * modes[] = {"pair/top_k", "pair", "pos", "pos/top_k"};
        printf("   best match: expert %d, prow=%s -> %.2e\n", be, bmode >= 0 ? modes[bmode] : "?", best);
    }
}

// T-MAC/s smoke: a large-enough grid that the number means something.  No
// reference here - correctness is the checks above.
void time_tiled(queue & q, const char * what, const bank & b, int n_expert, int pairs_per_expert,
                const activation & act, float * d_out, int out_stride, int top_k, int reps) {
    csr c;
    c.n_expert = n_expert;
    c.top_k = top_k;
    c.slot_map.resize((size_t)n_expert);
    for (int e = 0; e < n_expert; e++) c.slot_map[(size_t)e] = e % b.n_slots;
    c.off.assign((size_t)n_expert + 1, 0);
    for (int e = 0; e < n_expert; e++) c.off[(size_t)e + 1] = c.off[(size_t)e] + pairs_per_expert;
    c.n_pairs = c.off[(size_t)n_expert];
    c.pair.resize((size_t)c.n_pairs);
    std::iota(c.pair.begin(), c.pair.end(), 0);
    for (int e = 0; e < n_expert; e++) c.active.push_back(e);
    c.n_active = n_expert;
    c.n_act_max = n_expert;

    dcsr d = upload_csr(q, c);
    for (int i = 0; i < 3; i++) { // warm-up (JIT + caches)
        moe_tiled_gemm_launch(q, b.dev, b.slot_stride, d.slot, d.off, d.pair, d.active, d.nact, c.n_act_max, act.d_xq,
                              act.d_asa, d_out, out_stride, b.K, b.N, top_k, b.type, nullptr);
    }
    q.wait();
    // Pipelined: reps launches behind one wait, so the host submission latency
    // is amortised (the shape a real prefill batch sees).
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) {
        moe_tiled_gemm_launch(q, b.dev, b.slot_stride, d.slot, d.off, d.pair, d.active, d.nact, c.n_act_max, act.d_xq,
                              act.d_asa, d_out, out_stride, b.K, b.N, top_k, b.type, nullptr);
    }
    q.wait();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double macs = (double)c.n_pairs * b.N * b.K * reps;
    double best_ms = 1e30;
    for (int i = 0; i < 5; i++) { // one launch per wait: submission is inside the sample
        const auto s0 = std::chrono::steady_clock::now();
        moe_tiled_gemm_launch(q, b.dev, b.slot_stride, d.slot, d.off, d.pair, d.active, d.nact, c.n_act_max, act.d_xq,
                              act.d_asa, d_out, out_stride, b.K, b.N, top_k, b.type, nullptr);
        q.wait();
        best_ms = std::min(best_ms,
                           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count());
    }
    const double single_macs = (double)c.n_pairs * b.N * b.K;
    printf("%-34s %.2f T-MAC/s pipelined / %.2f single  (%d pairs x %d x %d, %d reps, %.1f ms)\n", what,
           macs / (ms * 1e9), single_macs / (best_ms * 1e9), c.n_pairs, b.N, b.K, reps, ms);
}

} // namespace

int main(int argc, char ** argv) {
    const bool do_time = argc > 1 && std::string(argv[1]) == "time";
    try {
        // In-order, like the engine's queues: every buffer here is filled by a
        // memcpy immediately before the launch that reads it, and SYCL 2020's
        // default is out-of-order (a race here silently corrupts the CSR and
        // the activation, which looks exactly like a kernel unpack bug).
        queue q{gpu_selector_v, property::queue::in_order()};
        const std::string dev = q.get_device().get_info<info::device::name>();
        printf("device: %s  (local mem %zu B, work-group %zu)\n", dev.c_str(),
               q.get_device().get_info<info::device::local_mem_size>(),
               q.get_device().get_info<info::device::max_work_group_size>());

        // n_embd = 2048, n_ff = 512, i.e. the gate/up shape is [n_ff][n_embd]
        // and the down shape is [n_embd][n_ff].
        const int K_gu = 2048, N_gu = 512;
        const int K_dn = 512, N_dn = 2048;
        const bank bank_gate = make_bank(q, 12, K_gu, N_gu, 3, 11);
        const bank bank_up = make_bank(q, 12, K_gu, N_gu, 3, 12);
        const bank bank_dn4 = make_bank(q, 12, K_dn, N_dn, 3, 13);
        const bank bank_dn5 = make_bank(q, 13, K_dn, N_dn, 3, 14);
        printf("banks: Q4_K %dx%d (%.2f MiB), Q5_K %dx%d (%.2f MiB)\n", N_gu, K_gu,
               bank_gate.host.size() / 1048576.0, N_dn, K_dn, bank_dn5.host.size() / 1048576.0);

        const int top_k = 2;
        const activation act_gu = make_activation(q, 48 / top_k, K_gu, 21); // 48 pairs, 24 tokens
        const activation act_dn = make_activation(q, 48, K_dn, 22);          // one row per pair

        // ---- the main CSR: 40 / 4 / 4 pairs, non-identity slots.
        // 40 is two tiles (32 + 8); 4 is a short tail.  pair[] is a
        // permutation, so the output row and the CSR position are unrelated.
        const csr c_main = make_csr({40, 4, 4}, top_k, {2, 0, 1}, 31);
        // The up projection reads a different bank through a different slot
        // map, so a slot mix-up between the two calls cannot cancel out.
        const csr c_up = make_csr({40, 4, 4}, top_k, {1, 2, 0}, 31);
        // The down projection indexes its activation by pair id (top_k == 0),
        // because the engine quantizes one row per routed pair for it.
        const csr c_down = make_csr({40, 4, 4}, 0, {2, 0, 1}, 31);

        const size_t rows = (size_t)c_main.n_pairs + 2 * kGuard;
        float * d_gu = malloc_device<float>(rows * (size_t)N_gu, q);
        float * d_dn = malloc_device<float>(rows * (size_t)N_dn, q);
        std::vector<float> sent_gu(rows * (size_t)N_gu, kSentinel);
        std::vector<float> sent_dn(rows * (size_t)N_dn, kSentinel);
        auto reset_gu = [&] { q.memcpy(d_gu, sent_gu.data(), sent_gu.size() * sizeof(float)).wait(); };
        auto reset_dn = [&] { q.memcpy(d_dn, sent_dn.data(), sent_dn.size() * sizeof(float)).wait(); };

        // gate into the up buffer's payload, then up in place over it.  The
        // sentinel is not reset between the two: the up epilogue must read the
        // gate output the gate launch just wrote.
        reset_gu();
        run_projection("gate Q4_K top_k>0", q, bank_gate, c_main, act_gu, d_gu, N_gu, nullptr);
        run_projection("up Q4_K top_k>0, gate, in place", q, bank_up, c_up, act_gu, d_gu, N_gu,
                       d_gu + (size_t)kGuard * N_gu);

        reset_dn();
        run_projection("down Q4_K top_k=0", q, bank_dn4, c_down, act_dn, d_dn, N_dn, nullptr);
        reset_dn();
        run_projection("down Q5_K top_k=0", q, bank_dn5, c_down, act_dn, d_dn, N_dn, nullptr);

        // A zero-pair expert (33 / 0 / 15: expert 0 is 32+1, expert 2 is a
        // 15-pair tail).  Expert 1 is still in `active[]`, so the kernel's
        // `np <= 0` retire path is the one under test.
        reset_dn();
        const csr c_zero = make_csr({33, 0, 15}, 0, {2, 0, 1}, 32);
        run_projection("down Q4_K 0-pair expert", q, bank_dn4, c_zero, act_dn, d_dn, N_dn, nullptr);

        // Only the first two experts are active: expert 2's four pairs must be
        // left alone even though the grid still has a slot for it.
        reset_dn();
        const csr c_cap = make_csr({40, 4, 4}, 0, {2, 0, 1}, 33, 2);
        run_projection("down Q4_K n_active cuts expert 2", q, bank_dn4, c_cap, act_dn, d_dn, N_dn, nullptr);

        // A non-resident expert (slot -1) contributes nothing.
        reset_dn();
        const csr c_nr = make_csr({40, 4, 4}, 0, {2, -1, 0}, 34);
        run_projection("down Q4_K slot -1 expert", q, bank_dn4, c_nr, act_dn, d_dn, N_dn, nullptr);

        // ---- T-MAC/s smoke check.  The grid here is larger than the correctness
        // cases (32 experts x 32 pairs, i.e. one full BM tile each, which is what
        // the real routing looks like), so the number is a kernel rate and not a
        // launch-latency measurement.
        {
            const int n_ex = 32, ppe = 32;
            const int t_reps = do_time ? 400 : 100; // 20 leaves the clock ramp in the sample
            activation t_gu = make_activation(q, (n_ex * ppe) / top_k, K_gu, 41);
            activation t_dn = make_activation(q, n_ex * ppe, K_dn, 42);
            float * d_tgu = malloc_device<float>((size_t)(n_ex * ppe) * N_gu, q);
            float * d_tdn = malloc_device<float>((size_t)(n_ex * ppe) * N_dn, q);
            time_tiled(q, "T-MAC/s gate Q4_K", bank_gate, n_ex, ppe, t_gu, d_tgu, N_gu, top_k, t_reps);
            time_tiled(q, "T-MAC/s down Q4_K", bank_dn4, n_ex, ppe, t_dn, d_tdn, N_dn, 0, t_reps);
            time_tiled(q, "T-MAC/s down Q5_K", bank_dn5, n_ex, ppe, t_dn, d_tdn, N_dn, 0, t_reps);
        }
    } catch (const std::exception & e) {
        printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
    if (g_fails) {
        printf("MOE TILED: %d FAILURES\n", g_fails);
    } else {
        printf("all moe_tiled stages OK\n");
    }
    return g_fails ? 1 : 0;
}
