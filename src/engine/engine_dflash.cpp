// DFlash / DFlash2 speculative decoding for the qwen35 family.
//
// The GGUF loaded with --spec-draft-model is a *block-diffusion drafter*: a 5-layer
// model that reads the target's hidden states at a few layers, keeps its own K/V
// ring for every committed token, and emits a whole block of draft candidates in
// one forward pass (see model/dflash.h).  One cycle of this file is
//
//   1. inject   the features the target captured for the newly committed tokens
//               (5 hidden states each) into the draft's K/V ring
//   2. draft    one batched forward over [anchor, MASK x (n_max)] at consecutive
//               positions -> the target head's logits -> top-k -> selector lattice
//               -> the host walks one coherent path through it (n_max tokens)
//   3. verify   one batched target forward over [anchor, draft0..draft_{n_max-1}]
//   4. accept   greedy prefix match, then the target's own token as the bonus
//   5. rollback the target's recurrent state to the last accepted row
//
// The verify / rollback / accept half is shared with MTP (mtp_verify,
// mtp_rollback), which is why the emitted stream is bit-identical to a plain
// greedy decode: the target's own token is always emitted last per cycle, so
// acceptance can only ever skip work.
//
// The draft side is GPU-only (it reads its weights through the native/int8
// stores in dnnl_gemm and runs the dflash.cpp kernels), so `--spec-type dflash2`
// needs a GPU partition with the oneDNN int8 table; anything else falls back to
// the plain decode with one [dflash] line.
#include "engine.h"
#include <tuple>
#include <algorithm>

#include "device/device_profile.h"
#include "model/dflash.h"
#include "quant.h"

#include <algorithm>
#include <chrono>
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "common/env.h"

namespace si {

namespace {

// First index of the maximum, ties going to the lowest index - the same rule the
// host sampler uses, so a tie cannot flip the emitted stream.
int argmax_f(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}

// nat_gemm serves M = 2..13; the injection runs over a whole prefill batch, so it
// is chunked to that width (the weight stream is re-read per chunk, which is
// 115 MB * ceil(rows/13) - see the cost note in setup_dflash).
// The draft's activation buffers are sized for dfl.df_block_ rows (the block forward),
// so the injection - which runs over *committed* tokens and can see more of them -
// must be chunked at that width, not at an arbitrary 13.
constexpr int kDfInjMax = 8;

// The current draft layer, so the probes can label a dump per layer: comparing
// each layer's output against llama.cpp's DFLASH_REF_LAYER=<il> tap is what says
// which layer a divergence appears in, instead of only seeing it in the sum.
static int df_sig_layer = -1;

static auto now_t = [] { return std::chrono::high_resolution_clock::now(); };
static auto ms_t = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };

// Device-side segment timing for the draft block forward.  A barrier orders with
// everything already queued on an in-order queue, so the interval between two
// barriers' command_start / command_end is exactly the device time of whatever was
// submitted between them.  This is the only sound attribution here: host-side timers
// charge an op with whichever one happens to wait when the queue fills, and no-op A/B
// cannot repair it (removing the heavy kernels just lets the queue drain with nothing
// blocking).  Needs the queue's enable_profiling property, which engine.cpp now sets.
static bool df_seg_on = false;
static double df_seg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
// device-side span of the whole layer stack (first layer's start barrier -> last
// layer's end barrier), against the host wall: the difference is host submission
// time the device timeline cannot see.
static sycl::event df_seg_dev0, df_seg_dev1;
static sycl::event ffn_e[2];
static sycl::event tt_e[4];
static const bool tt_dbg = si::env::flag("PF_DFLASH_TOPTIME");
static auto tt_now = [] { return std::chrono::high_resolution_clock::now(); };
static inline double df_ts_ms(const sycl::event & e, bool start) {
    const uint64_t t = start ? e.get_profiling_info<sycl::info::event_profiling::command_start>()
                             : e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return (double)t / 1e6;
}

// PF_DFLASH_OPTIME: per-op accumulation over one draft block forward.  Inserted by
// LINE NUMBER inside df_block's layer loop - text-anchored inserts kept landing in
bool df_dbg() {
    static const bool b = si::env::flag("PF_DFLASH_DEBUG");
    return b;
}

// PF_DFLASH_STAGE=1: ||x|| of each draft stage (a wrong draft is almost always
// a NaN, an explosion or a zero at one specific step, and this says which).
// Host reference for the grouped dynamic conv, gated on PF_DFLASH_CONVCHK.  It
// re-derives the conv from the same base/dynamic tensors the kernel reads, which is
// how the conv's static+dynamic split and its causal-within-block tap were checked
// against llama.cpp element by element.
void df_conv_check(sycl::queue & q, const float * in, const float * dyn, const float * base, const float * out,
                   const float * res, int M, int width, int conv_k, int conv_group, int conv_proj, int side,
                   const char * tag) {
    std::vector<float> hi((size_t)M * width), hd((size_t)M * conv_proj), ho((size_t)M * width);
    // base comes from dev_f32(), i.e. DEVICE USM, so it has to come back before the
    // host can index it - dereferencing it in place is what made this check read
    // nothing and silently "pass"
    std::vector<float> hb((size_t)width * conv_k * 2);
    q.memcpy(hi.data(), in, hi.size() * 4).wait();
    q.memcpy(hd.data(), dyn, hd.size() * 4).wait();
    q.memcpy(ho.data(), out, ho.size() * 4).wait();
    q.memcpy(hb.data(), base, hb.size() * 4).wait();
    double hs = 0, ds = 0;
    long n = 0;
    double worst = 0;
    int wi = -1, wc = -1;
    for (int i = 0; i < M; i++) {
        for (int c = 0; c < width; c++) {
            double acc = 0;
            for (int t = 0; t < conv_k && t <= i; t++) {
                // MUST match df_conv_launch exactly.  These two indices used to be
                // base[side][c*conv_k + t] and dyn[(g*conv_k + t)*2 + side], i.e. the
                // pre-fix derivation, and were never updated when the kernel's axes
                // were corrected - so this check spent that time reporting the
                // difference between two formulas while claiming to verify the conv.
                const size_t ngrp = (size_t)width / conv_group;
                const double b = hb[(size_t)width * conv_k * side + (size_t)c + (size_t)width * t];
                const double d = hd[(size_t)i * conv_proj + (size_t)(c / conv_group) + ngrp * (size_t)t
                                   + ngrp * conv_k * (size_t)side];
                acc += (b + d) * hi[(size_t)(i - t) * width + c];
            }
            if (res) {
                acc += res[(size_t)i * width + c];
            }
            const double g = ho[(size_t)i * width + c];
            const double e = std::fabs(acc - g);
            if (e > worst) {
                worst = e;
                wi = i;
                wc = c;
            }
            hs += acc * acc;
            ds += g * g;
            n++;
        }
    }
    // The conv term on its own, so it can be compared with llama.cpp when `res` is
    // the same buffer as `out` (the draft's side-1 conv): there the host reference
    // reads a residual the kernel has already overwritten, so its rms is ~2x the
    // device's and says nothing about the conv.
    double hs0 = 0;
    if (res) {
        for (int i = 0; i < M; i++) {
            for (int c = 0; c < width; c++) {
                double acc = 0;
                for (int t = 0; t < conv_k && t <= i; t++) {
                    const size_t ngrp = (size_t)width / conv_group;
                    const double b = hb[(size_t)width * conv_k * side + (size_t)c + (size_t)width * t];
                    const double d = hd[(size_t)i * conv_proj + (size_t)(c / conv_group) + ngrp * (size_t)t
                                       + ngrp * conv_k * (size_t)side];
                    acc += (b + d) * hi[(size_t)(i - t) * width + c];
                }
                hs0 += acc * acc;
            }
        }
        fprintf(stderr, "[dflash] convchk %s: conv-only rms=%.6f (host+res=%.6f, dev=%.6f)\n", tag,
                std::sqrt(hs0 / n), std::sqrt(hs / n), std::sqrt(ds / n));
    }
    fprintf(stderr, "[dflash] convchk %s: host rms=%.6f dev rms=%.6f rms err=%.6g worst=%.6g at row %d c%d\n", tag,
            std::sqrt(hs / n), std::sqrt(ds / n), std::sqrt(std::fabs(hs - ds) / n), worst, wi, wc);
}

void df_stage(sycl::queue & q, const char * tag, const float * p, size_t n) {
    static const bool on = si::env::flag("PF_DFLASH_STAGE");
    if (!on) {
        return;
    }
    std::vector<float> h(n);
    q.memcpy(h.data(), p, n * 4).wait();
    double s = 0.0;
    int bad = 0;
    double amax = 0;
    for (size_t i = 0; i < n; i++) {
        s += (double)h[i] * h[i];
        amax = std::max(amax, (double)std::fabs(h[i]));
        bad += !std::isfinite(h[i]);
    }
    fprintf(stderr, "[dfstage] %-14s n=%zu rms=%.5f max=%.4g nonfinite=%d first=%.6g\n", tag, n, std::sqrt(s / (double)n),
            amax, bad, h[0]);
}

// PF_DFLASH_RAWQK=1 leaves the draft's Q/K projection un-normed, so a host check
// can separate a wrong GEMM from a wrong norm/rope (both are Q4_K x f32)
static const bool df_dn_norm = !si::env::flag("PF_DFLASH_RAWQK");

// f16 ring reads must REINTERPRET the bit pattern.  sycl::half(uint16_t) is a
// numeric conversion, so every check that used it was reading the *integer* as a
// float - which is how a correct K of -0.348877 read back as 46485 and looked like
// a 35000-rms saturated ring.
[[maybe_unused]] static inline double f16_bits(uint16_t b) {
    sycl::half h;
    std::memcpy(&h, &b, sizeof(h));
    return (double)h;
}

// Host reference for the block attention, over the SAME ring and Q the kernel
// reads.  Everything upstream is now verified (injection, conv, q/k norm+rope,
// the ring stores), so this is the one link left: it compares the device's
// combine output against a plain softmax(QK^T/sqrt(d)) * V recomputed on the host,
// reading the f16 ring through f16_bits (a sycl::half(uint16_t) conversion would
// read the integer, not the bit pattern).
void df_attn_check(sycl::queue & q, const float * qbuf, int row_stride, int M, const void * kpool,
                   const void * vpool, const int32_t * pos, int n_head, int n_head_kv, int head_dim, int pos0,
                   int swa, int ring, float scale, const float * out) {
    const int nkv = n_head_kv * head_dim;
    std::vector<float> qv((size_t)M * row_stride);
    std::vector<float> ov((size_t)M * n_head * head_dim);
    std::vector<uint16_t> kr((size_t)ring * nkv), vr((size_t)ring * nkv);
    q.memcpy(qv.data(), qbuf, qv.size() * 4).wait();
    q.memcpy(ov.data(), out, ov.size() * 4).wait();
    q.memcpy(kr.data(), kpool, kr.size() * 2).wait();
    q.memcpy(vr.data(), vpool, vr.size() * 2).wait();
    for (int r2 = 0; r2 < M; r2++) {
        const int p = pos[r2];
        const int lo = std::max(0, p - swa + 1);
        const int hi = pos0 + M - 1;
        const int gqa = n_head / n_head_kv;
        double worst = 0, hs = 0, os = 0;
        for (int h2 = 0; h2 < n_head; h2++) {
            const int kh2 = h2 / gqa;
            std::vector<double> sc;
            double mx = -1e30;
            for (int kk = lo; kk <= hi; kk++) {
                double dot = 0;
                for (int d2 = 0; d2 < head_dim; d2++) {
                    dot += (double)qv[(size_t)r2 * row_stride + (size_t)h2 * head_dim + d2]
                           * f16_bits(kr[(size_t)(kk % ring) * nkv + (size_t)kh2 * head_dim + d2]);
                }
                dot *= scale;
                mx = std::max(mx, dot);
                sc.push_back(dot);
            }
            double sum = 0;
            for (size_t i2 = 0; i2 < sc.size(); i2++) {
                sc[i2] = std::exp(sc[i2] - mx);
                sum += sc[i2];
            }
            for (int d2 = 0; d2 < head_dim; d2++) {
                double a = 0;
                for (size_t i2 = 0; i2 < sc.size(); i2++) {
                    a += sc[i2] / sum * f16_bits(vr[(size_t)((lo + (int)i2) % ring) * nkv
                                                      + (size_t)kh2 * head_dim + d2]);
                }
                const double g = ov[(size_t)r2 * n_head * head_dim + (size_t)h2 * head_dim + d2];
                worst = std::max(worst, std::fabs(a - g));
                hs += a * a;
                os += g * g;
            }
        }
        const int nq = n_head * head_dim;
        fprintf(stderr, "[dflash] attnchk row %d (keys %d..%d): host rms=%.5f dev rms=%.5f maxdiff=%.4g\n", r2, lo, hi,
                std::sqrt(hs / nq), std::sqrt(os / nq), worst);
    }
}

// Per-row rms of a conv's output.  The reference's REF_* taps print ONE row at a
// time, and the anchor row (a real committed token) is an order of magnitude larger
// than the mask rows - so a whole-batch aggregate is not comparable with them.
void df_rowrms(sycl::queue & q, const float * p, int M, int width, const char * tag) {
    std::vector<float> v((size_t)M * width);
    q.memcpy(v.data(), p, v.size() * 4).wait();
    if (const char * bp = si::env::str("PF_DFLASH_BIN")) {
        // one file per stage: the four taps are all dumped in a single pass and the
        // reference writes a single tensor, so the tag has to be part of the name.
        std::string bp2 = std::string(bp) + "." + tag + ".bin";
        if (FILE * f = fopen(bp2.c_str(), "wb")) {
            fwrite(v.data(), 4, v.size(), f);
            fclose(f);
            fprintf(stderr, "[dflash-sig] %s written to %s (%zu bytes)\n", tag, bp2.c_str(), v.size() * 4);
        }
    skip_bin:;
    }
    fprintf(stderr, "[dflash] rowrms %s:", tag);
    for (int m = 0; m < M; m++) {
        double ss = 0, am = 0;
        for (int j = 0; j < width; j++) {
            ss += (double)v[(size_t)m * width + j] * v[(size_t)m * width + j];
            am = std::max(am, std::fabs((double)v[(size_t)m * width + j]));
        }
        fprintf(stderr, " r%d=%.4f/%.4g", m, std::sqrt(ss / width), am);
    }
    fprintf(stderr, "\n");
}

// Per-position rms / max / first for a whole device tensor, printed in the same
// shape as llama.cpp's "[dflash-ref] pos N hidden rms=... max=... first=..." so the
// two can be lined up stage by stage.  Whole-batch aggregates are NOT comparable:
// the anchor row is an order of magnitude larger than the mask rows, which is what
// sent me chasing a phantom for several rounds.
// One block counter shared by every probe, so two probes in the same block agree on
// the ".bN" in their filename.  It was a function-local static inside df_sig, which
// meant a probe outside df_sig could not label its file at all.
static int sig_block_ctr = 0;

int sig_block_now() {
    return sig_block_ctr;
}

void df_sig(sycl::queue & q, const float * p, int M, int width, const char * tag) {
    std::vector<float> v((size_t)M * width);
    q.memcpy(v.data(), p, v.size() * 4).wait();
    static const bool sig_all = si::env::flag("PF_DFLASH_SIGALL");
    // The reference snapshots only the FIRST draft block, so the dump must too or
    // the comparison lines up block n with block 0.
    static const bool bin_first_only = !si::env::flag("PF_DFLASH_BIN_ALL");
    static int bin_block = -1;
    // Count blocks, not calls: all five stages are tapped in one pass, so the stage
    // that comes FIRST in the layer is what advances the block index.
    if (strcmp(tag, "xnorm") == 0 && (df_sig_layer <= 0 || !sig_all)) {
        // xnorm at layer 0 is the first probe of a block
        static int bin_blocks = 0;
        bin_block = bin_first_only ? bin_blocks++ : 0;
    }
    // The block index belongs in the FILENAME, not just in a gate.  Gating on
    // "first block only" was not enough: the probe order within a layer means the
    // block counter advances after that layer's own probes have already run, so
    // later blocks' tensors landed in the previous layer's file.  Two independent
    // readings of the same layer disagreed by exactly one layer, which is what a
    // one-block offset looks like.  A name that carries the block cannot be
    // ambiguous.
    if (strcmp(tag, "xnorm") == 0 && df_sig_layer <= 0) {
        sig_block_ctr++;
    }
    const int sig_block = sig_block_ctr;
    // ALWAYS first-block-only, including under SIGALL: llama.cpp's DFLASH_REF_TAP
    // snapshots only the first draft block, so a later block's tensor silently
    // replaces the one being compared.  Reading a stale file is worse than having no
    // file - it is how "layer 4's input is zero" came to be believed.
    const bool bin_here = !bin_first_only || bin_block == 0;
    if (const char * bp = si::env::str("PF_DFLASH_BIN")) {
        // one file per stage: the four taps are all dumped in a single pass, so the
        // stage name has to be part of the filename.
        const std::string bp2 = std::string(bp) + "." + tag + (sig_all ? ".L" + std::to_string(df_sig_layer) : "")
                               + ".b" + std::to_string(sig_block) + ".bin";
        if (bin_here) {
            if (FILE * f = fopen(bp2.c_str(), "wb")) {
                fwrite(v.data(), 4, v.size(), f);
                fclose(f);
                fprintf(stderr, "[dflash-sig] %s written to %s (%zu bytes)\n", tag, bp2.c_str(), v.size() * 4);
            }
        }
    }
    for (int m = 0; m < M; m++) {
        double ss = 0, am = 0;
        for (int j = 0; j < width; j++) {
            const float x = v[(size_t)m * width + j];
            ss += (double)x * x;
            am = std::max(am, std::fabs((double)x));
        }
        fprintf(stderr, "[dflash-sig] %s pos %d rms=%.5f max=%.4g first=%.6g\n", tag, m, std::sqrt(ss / width), am,
                v[(size_t)m * width]);
    }
}

// Host reference for one draft GEMM: dequantize the GGUF weight a row at a time
// and dot it with the device's own activation.  Samples columns so a 248320-wide
// head is affordable.  `up` (the SwiGLU gate) is applied the way df_gemm's callers
// mean it: silu(x) * up.
void df_gemm_check(sycl::queue & q, const wt & w, const float * x_dev, int xs, float * out_dev, int os, int M,
                   const float * up, int us, const char * tag, int ncol) {
    std::vector<float> x((size_t)M * xs);
    std::vector<float> gr((size_t)w.K);
    std::vector<float> u;
    q.memcpy(x.data(), x_dev, x.size() * 4).wait();
    if (up) {
        u.resize((size_t)M * us);
        q.memcpy(u.data(), up, u.size() * 4).wait();
    }
    const int step = std::max(1, w.N / std::max(1, ncol));
    double worst = 0;
    int wc = -1;
    for (int m = 0; m < M; m++) {
        for (int c2 = 0; c2 < w.N; c2 += step) {
            dequantize_row(w.type, (const char *)w.data + (size_t)c2 * quant_row_bytes(w.type, w.K), gr.data(),
                           (int64_t)w.K);
            double acc = 0;
            for (int k2 = 0; k2 < w.K; k2++) {
                double xv = x[(size_t)m * xs + k2];
                if (up) {
                    xv = (xv / (1.0 + std::exp(-xv))) * u[(size_t)m * us + k2];
                }
                acc += (double)gr[k2] * xv;
            }
            float one = 0.f;
            q.memcpy(&one, out_dev + (size_t)m * os + c2, 4).wait();
            const double e = std::fabs(acc - one);
            if (e > worst) {
                worst = e;
                wc = c2;
            }
        }
    }
    fprintf(stderr, "[dflash] gemmchk %s: K=%d N=%d M=%d up=%d worst=%.6g at col %d\n", tag, w.K, w.N, M,
            up ? 1 : 0, worst, wc);
}

// Host reference for a conv dynamic-projection GEMM.  These two (attn_conv_proj,
// ffn_conv_proj) are the only weights in the draft's forward that had never been
// checked against an exact dequant, and they feed the conv coefficients directly -
// a wrong one scales the whole conv.  Sampled columns only: conv_proj is 1280 wide.
void df_proj_check(sycl::queue & q, const wt & w, const float * x_dev, int M, float * dyn_dev,
                   const char * tag) {
    std::vector<float> x((size_t)M * w.K);
    std::vector<float> gr((size_t)w.K);
    q.memcpy(x.data(), x_dev, x.size() * 4).wait();
    const int ngrp = (w.K + 31) / 32;
    for (int m = 0; m < M; m++) {
        for (int c2 = 0; c2 < w.N; c2 += 97) {
            dequantize_row(w.type, (const char *)w.data + (size_t)c2 * quant_row_bytes(w.type, w.K), gr.data(),
                           (int64_t)w.K);
            double a = 0;
            for (int k2 = 0; k2 < w.K; k2++) {
                a += (double)gr[k2] * x[(size_t)m * w.K + k2];
            }
            float one = 0.f;
            q.memcpy(&one, dyn_dev + (size_t)m * w.N + c2, 4).wait();
            if (m == 0 && c2 < 4 * 97) {
                fprintf(stderr, "[dflash] projchk %s col %d: host=%.6g dev=%.6g\n", tag, c2, a, one);
            }
        }
    }
    (void)ngrp;
}

static inline float silu_test(float x) {
    return x / (1.0f + sycl::exp(-x));
}
} // namespace

#define DFDBG(...)                                                                                                      \
    do {                                                                                                                 \
        if (si::env::flag("PF_DFLASH_DEBUG")) {                                                                                 \
            fprintf(stderr, __VA_ARGS__);                                                                                \
        }                                                                                                                \
    } while (0)

// ---------------------------------------------------------------------------
// Load the drafter, register its weights, size its buffers.
void engine::setup_dflash(const std::string & draft_path) {
    dfl.dflash_path_ = draft_path;
    // Draft weights use the primary GPU's oneDNN/native-store table.  A
    // single-device setup reaches this method before that table exists, and a
    // non-primary draft cannot use the shared LM head's primary-device logits.
    if (!multi_dev || !md_xmx || dfl.df_dev_ != 0 || !dnnl_for(0)) {
        fprintf(stderr, "[dflash] needs a multi-device oneDNN GPU partition with draft on device 0 - disabled\n");
        return;
    }
    auto dm = std::make_unique<dflash_model>();
    try {
        dm->load(draft_path, m.hp.n_embd, m.hp.n_vocab);
    } catch (const std::exception & e) {
        fprintf(stderr, "[dflash] draft model rejected: %s\n", e.what());
        return;
    }
    // The draft block itself fits the per-call token buffer, and the verifier
    // needs one logits row for the anchor plus each drafted token.
    if (dm->hp.block_size < 2 || dm->hp.block_size > kMaxT) {
        fprintf(stderr, "[dflash] draft block_size=%d is outside [2,%d] - disabled\n", dm->hp.block_size, kMaxT);
        return;
    }
    // every dflash.target_layers entry must be a real target layer *input*
    for (int i = 0; i < dm->hp.n_tgt_layer; i++) {
        const int il = dm->hp.tgt_layer[i];
        if (il < 0 || il >= m.hp.n_layer) {
            fprintf(stderr, "[dflash] draft model rejected: target_layers[%d] = %d is out of range [0, %d)\n", i, il,
                    m.hp.n_layer);
            return;
        }
    }
    dnnl_gemm * D = dnnl_for(dfl.df_dev_);
    sycl::queue & dq = dev_queue(dfl.df_dev_);
    dfl.dfm_ = std::move(dm);
    const dflash_hp & hp = dfl.dfm_->hp;
    dfl.dfm_->upload_f32(dq);
    // every f32 tensor must have a device copy: the kernels read them directly and
    // a host mmap pointer would be garbage on the device
    {
        int missing = 0;
        auto chk = [&](const void * p, const char * nm) {
            if (p == nullptr || dfl.dfm_->dev.find(p) == dfl.dfm_->dev.end()) {
                missing++;
                fprintf(stderr, "[dflash] missing device copy: %s\n", nm);
            }
        };
        chk(dfl.dfm_->enc_norm, "enc.output_norm");
    {
        // After an RMSNorm the output's rms is forced to be rms(weight), so this is
        // the number the injected rows must match - the reference's post-norm inp_g
        // measures 0.177, and a mismatch here is a real defect, not a probe artefact.
        auto frms2 = [](const float * p, int n) {
            double s2 = 0;
            for (int i = 0; i < n; i++) {
                s2 += (double)p[i] * p[i];
            }
            return std::sqrt(s2 / n);
        };
        fprintf(stderr, "[dflash] norm rms: enc=%.6f out=%.6f (n_embd=%d)\n", frms2(dfl.dfm_->enc_norm, hp.n_embd),
                frms2(dfl.dfm_->output_norm, hp.n_embd), hp.n_embd);
    }
    if (si::env::flag("PF_DFLASH_NDEVCHK")) {
        // Every host check so far computed its reference from the DEVICE's weight, so
        // a wrong device-side norm weight would be invisible to all of them.  Compare
        // the uploaded copies against the GGUF tensors directly.
        const float * hs[3] = {dfl.dfm_->enc_norm, dfl.dfm_->output_norm, dfl.dfm_->layers[0].k_norm};
        const char * nm[3] = {"enc_norm", "output_norm", "blk0.k_norm"};
        const int wn3[3] = {hp.n_embd, hp.n_embd, hp.head_dim};
        for (int t = 0; t < 3; t++) {
            const float * dp = dfl.dfm_->dev_f32(hs[t]);
            std::vector<float> dv((size_t)wn3[t]);
            if (dp == nullptr) {
                fprintf(stderr, "[dflash] ndevchk %s: dev_f32 is NULL\n", nm[t]);
                continue;
            }
            dev_queue(dfl.df_dev_).memcpy(dv.data(), dp, dv.size() * 4).wait();
            int nbad = 0;
            double hs3 = 0, ds3 = 0;
            for (int i = 0; i < wn3[t]; i++) {
                if (hs[t][i] != dv[i]) {
                    if (nbad < 3) {
                        fprintf(stderr, "   ndevchk %s[%d]: host=%.6f dev=%.6f\n", nm[t], i, hs[t][i], dv[i]);
                    }
                    nbad++;
                }
                hs3 += (double)hs[t][i] * hs[t][i];
                ds3 += (double)dv[i] * dv[i];
            }
            fprintf(stderr, "[dflash] ndevchk %s: %d/%d mismatches, host rms=%.6f dev rms=%.6f\n", nm[t], nbad, wn3[t],
                    std::sqrt(hs3 / wn3[t]), std::sqrt(ds3 / wn3[t]));
    }
    }
        chk(dfl.dfm_->output_norm, "output_norm");
        for (int i = 0; i < hp.n_layer; i++) {
            const dflash_layer_t & L = dfl.dfm_->layers[(size_t)i];
            chk(L.attn_norm, "attn_norm");
            chk(L.q_norm, "attn_q_norm");
            chk(L.k_norm, "attn_k_norm");
            chk(L.ffn_norm, "ffn_norm");
            chk(L.attn_conv_base, "attn_conv_base");
            chk(L.ffn_conv_base, "ffn_conv_base");
        }
        if (missing) {
            fprintf(stderr, "[dflash] draft model rejected: %d tensors lack a device copy\n", missing);
            return;
        }
    }

    // ---- weights: the same native/int8 stores the main model uses, keyed by
    // the draft GGUF's host pointers (which are unique per tensor) ----
    int n_u4 = 0, n_int8 = 0, n_fail = 0;
    // per card: the draft runs on dfl.df_dev_, so its store choice follows that
    // device's profile, not whichever GPU happened to be first
    const bool w4_on = si::dev::profile_flag(dq, "PF_W4", si::dev::for_queue(dq).wt.w4 != 0);
    // PF_DFLASH_WSCALE: multiply every drafter weight's step plane.  Diagnostic
    // for the "is the drafter's weight scale what a trained model expects"
    // question; 1 = the GGUF as shipped.
    const float wscale = si::env::str("PF_DFLASH_WSCALE") ? (float)atof(si::env::str("PF_DFLASH_WSCALE")) : 1.0f;

    auto add = [&](const wt & t, const char * nm, bool gather_only = false) {
                bool ok = false;
        bool ok4 = false;
        // PF_DFLASH_LAYER_W4=0 keeps the selector codebooks on u4 (df_sel_launch
        // gathers their raw u4 planes, so they have no int8 form) but sends every
        // *layer* tensor down add_weight's per-row int8 store instead.  That is the
        // only valid u4-vs-int8 A/B for the draft; PF_W4=0 disables the selector and
        // with it the whole draft.
        static const bool layer_w4 = [] {
            const char * e = si::env::str("PF_DFLASH_LAYER_W4");
            return e ? atoi(e) != 0 : true;
        }();
        if (w4_on && (layer_w4 || gather_only)) {
            const bool any = (t.type != 12);
            ok = D->add_weight_w4(t.data, t.data, t.type, t.K, t.N, /*any_type=*/any,
                                  /*gemv_only=*/gather_only, wscale);
            ok4 = ok;
            if (ok) {
                n_u4++;
            }
        }
        if (!ok) {
            ok = D->add_weight(wkey(dfl.df_dev_, t.data), t.data, t.type, t.K, t.N);
            if (ok) {
                n_int8++;
            } else {
                n_fail++;
                fprintf(stderr, "[dflash] no weight store for %s (K=%d N=%d type=%d)\n", nm, t.K, t.N, (int)t.type);
            }
        }
        return ok || ok4;
    };
    add(dfl.dfm_->fc, "fc");
    add(dfl.dfm_->sel_hidden, "selector_hidden");
    // the two selector codebooks are only ever read one row at a time by
    // df_sel_launch, so they need no oneDNN prefill primitive (a 248320-wide
    // tensor never gets one - the same reason the MTP draft head is GEMV-only)
    add(dfl.dfm_->sel_prev, "selector_predecessor", /*gather_only=*/true);
    add(dfl.dfm_->sel_next, "selector_successor", /*gather_only=*/true);
    for (int i = 0; i < hp.n_layer; i++) {
        const dflash_layer_t & L = dfl.dfm_->layers[(size_t)i];
        char nm[64];
        auto addl = [&](const wt & t, const char * suffix) {
            snprintf(nm, sizeof(nm), "blk.%d.%s", i, suffix);
            add(t, nm);
        };
        addl(L.wq, "attn_q");
        addl(L.wk, "attn_k");
        addl(L.wv, "attn_v");
        addl(L.wo, "attn_output");
        addl(L.ffn_gate, "ffn_gate");
        addl(L.ffn_up, "ffn_up");
        addl(L.ffn_down, "ffn_down");
        addl(L.attn_conv_proj, "attn_conv_proj");
        addl(L.ffn_conv_proj, "ffn_conv_proj");
    }
    // the draft reads the *target's* LM head once per block (0.625 vs 1.0625
    // B/weight for a u4 copy, the same trade PF_MTP_HEAD_W4 makes); reuse the MTP
    // draft's private copy when it exists so a second 0.8 GB is not allocated
    // PF_DFLASH_HEAD_W4=0 forces the exact int8 head.  The draft's whole job is the
    // top-1 of each slot, and a per-32 4-bit grid of the Q6_K head can reorder
    // candidates that the hidden state got right, which costs acceptance without
    // any other symptom.
    static const bool head_w4 = si::env::flag("PF_DFLASH_HEAD_W4");
    if (mtp.mtp_head_w4_ && head_w4) {
        dfl.df_head_key_ = mtp.mtp_head_w4_key_;
    } else if (w4_on && head_w4
               && D->add_weight_w4(dfl.df_head_key_store_, m.output.data, m.output.type, m.output.K, m.output.N,
                                   /*any_type=*/true)) {
        dfl.df_head_key_ = dfl.df_head_key_store_;
        dfl.df_head_w4_ = true;
    } else {
        dfl.df_head_key_ = wkey(dfl.df_dev_, m.output.data);
    }
    fprintf(stderr, "[dflash] weight stores: u4=%d int8=%d failed=%d\n", n_u4, n_int8, n_fail);
    if (n_fail > 0) {
        fprintf(stderr, "[dflash] %d draft tensors have no weight store - disabled\n", n_fail);
        return;
    }
                // ---- geometry ----
    dfl.df_block_ = hp.block_size;
    // The default is the measured optimum, not block_size-1: acceptance is flat
    // from 2 to 5 and each extra draft row costs a full extra pass through the
    // draft, so the optimum sits where acceptance stops improving.  Measured on the
    // 27B / 2x A770 at nmax 1/2/3/5/7 (96-token greedy generation):
    //
    //   k      1     2     3     5     7
    //   acc  0.45  0.88  0.88  0.88  0.75
    //   ms/t 65.3  53.9  56.7  64.6  80.3
    //
    // so k=2 wins and block_size-1 (7) is the worst of them.  That sweep was taken
    // with the rotary width wrong (see dflash.cpp), which capped acceptance at ~0.9
    // drafts per cycle and hid the whole k>2 range.  Re-measured after fixing it:
    //
    //     k      1     2     3     4     5     6
    //     acc  0.98  1.44  2.18  2.54  2.90  2.90
    //     ms/t 48.6  41.9  34.1  32.5  31.5  33.3
    //
    // so k=5 wins, at 31.5 ms/token against the ~68 ms/token plain decode.
    dfl.df_k_ = std::min({dfl.df_kmax_ > 0 ? dfl.df_kmax_ : 5, dfl.df_block_ - 1, kMaxB - 1});
        // The draft attends over [committed window, whole block]; the ring must hold
    // the window plus the block or a live committed cell would alias a block one.
    const int swa = hp.swa > 0 ? hp.swa : 2048;
    dfl.df_swa_ = swa;
    dfl.df_ring_ = ((swa + dfl.df_block_ + 31) / 32) * 32;
    static const int kvb = [] {
        const char * e = si::env::str("PF_DFLASH_KV");
        return e ? atoi(e) : 2;
    }();
    dfl.df_kv_bytes_ = kvb == 4 ? 4 : 2;
    dfl.df_splits_ = 8;
        const int M = dfl.df_block_;
    const int nh = hp.n_head * hp.head_dim;
    const int nkv = hp.n_head_kv * hp.head_dim;
    const size_t ring_stride = (size_t)dfl.df_ring_ * hp.n_head_kv * hp.head_dim;
    dfl.d_df_h = alloc_elems<float>((size_t)M * hp.n_embd);
    dfl.d_df_b = alloc_elems<float>((size_t)M * hp.n_embd);
    dfl.d_df_c = alloc_elems<float>((size_t)M * hp.n_embd);
    dfl.d_df_qkv = alloc_elems<float>((size_t)M * (nh + 2 * nkv));
    dfl.d_df_gu = alloc_elems<float>((size_t)M * 2 * hp.n_ff);
    dfl.d_df_dyn = alloc_elems<float>((size_t)M * std::max(1, hp.conv_proj));
    dfl.d_df_gate = alloc_elems<float>((size_t)M * std::max(1, hp.sel_rank));
    // the selector's transition lattice: K*K edges plus K node scores per row,
    // mirrored to the host so the walk can read it
    dfl.h_df_lattice.assign((size_t)M * (hp.sel_top_k + hp.sel_top_k * hp.sel_top_k), 0.f);
    // the head's per-position top-k ids and logits, which the selector walks
    dfl.d_df_ids = alloc_elems<int32_t>((size_t)M * hp.sel_top_k);
    dfl.d_df_vals = alloc_elems<float>((size_t)M * hp.sel_top_k);
    // topk slice partials: S slices per row.  Sized from the device's compute-unit
    // count so the slice pass fills the machine (M alone is 6 workgroups).
    // sized from *this* device's compute-unit count so the slice pass fills that
    // machine (M alone is 6 workgroups)
    const sycl::queue & dq_s = dev_queue(dfl.df_dev_);
    dfl.df_slices_ = (int)si::dev::for_queue(dq_s).hw.compute_units / std::max(1, M);
    dfl.df_slices_ = std::max(1, std::min(dfl.df_slices_, 256));
    if (const char * e = si::env::str("PF_DFLASH_SLICES")) {
        dfl.df_slices_ = atoi(e);
    } else {
        // measured optimum, and NOT the occupancy-maximising value: S=1/8/32/64/128/256
        // measures topk 2.09/1.39/1.52/1.69/1.70/1.67 ms.  Past S=8 the merge pass and
        // the S*K candidates it reduces cost more than the extra parallelism buys, so
        // the kernel is latency-bound on its per-lane SLM insertion chain, not on
        // bandwidth.  1.39 ms for 5.96 MB is still ~4 GB/s.
        dfl.df_slices_ = 8;
    }
    dfl.df_pcap_ = M * dfl.df_slices_ * hp.sel_top_k;
    dfl.d_df_pids = alloc_elems<int32_t>((size_t)dfl.df_pcap_);
    dfl.d_df_pvals = alloc_elems<float>((size_t)dfl.df_pcap_);
    dfl.d_df_lattice = alloc_elems<float>((size_t)M * (hp.sel_top_k + hp.sel_top_k * hp.sel_top_k));
    // The selector's two codebooks are read one row at a time by df_sel_launch, so
    // it wants their raw native u4 planes (a 248320-wide tensor gets no oneDNN
    // primitive - the same reason the MTP draft head is GEMV-only).
    {
        const uint8_t * pv = nullptr;
        const uint16_t * ps = nullptr;
        const uint16_t * po = nullptr;
        int wk = 0, wn = 0;
        if (D->w4_planes(dfl.dfm_->sel_prev.data, &pv, &ps, &po, &wk, &wn)) {
            dfl.df_sel_pv_ = pv;
            dfl.df_sel_ps_ = ps;
            dfl.df_sel_po_ = po;
        }
        const uint8_t * nv = nullptr;
        const uint16_t * ns = nullptr;
        const uint16_t * no = nullptr;
        if (D->w4_planes(dfl.dfm_->sel_next.data, &nv, &ns, &no, &wk, &wn)) {
            dfl.df_sel_nv_ = nv;
            dfl.df_sel_ns_ = ns;
            dfl.df_sel_no_ = no;
        }
        if (dfl.df_sel_pv_ == nullptr || dfl.df_sel_nv_ == nullptr) {
            fprintf(stderr, "[dflash] the DFlash2 selector codebooks have no u4 store - disabled\n");
            return;
        }
    }
    dfl.d_df_partials =
        alloc_elems<float>((size_t)M * hp.n_head * dfl.df_splits_ * (2 + hp.head_dim));
    dfl.d_df_feat_dev = alloc_elems<float>((size_t)kDfInjMax * hp.n_feat);
    // the feature capture: one interleaved [token][n_tgt_layer * n_embd] buffer in
    // HOST USM, because the captured layers can sit on different devices and the
    // two GPUs share no device memory (host USM is the engine's cross-device path
    // already - see handoff_x).  Every forward writes it; the injection reads it.
    dfl.df_cap_rows_ = kMaxB * kMaxT;
    // DEVICE USM, not host: the capture kernels write it from the target's
    // per-device queues and the injection reads it from the draft's, so it has to
    // be device-visible to both.  alloc_bytes only returns host memory under
    // cpu_mode/host_act, and the injection's host->device memcpy then read zeros.
    dfl.d_df_feat = alloc_elems<float>((size_t)dfl.df_cap_rows_ * hp.n_feat);
    dfl.d_df_kring = dev_alloc_on(dfl.df_dev_, ring_stride * (size_t)hp.n_layer * (size_t)dfl.df_kv_bytes_);
    dfl.d_df_vring = dev_alloc_on(dfl.df_dev_, ring_stride * (size_t)hp.n_layer * (size_t)dfl.df_kv_bytes_);
    dfl.d_df_pos = alloc_elems<int32_t>(kMaxT);
    dfl.h_df_pos.assign(kMaxT, 0);
    dfl.d_df_info = (step_info *)alloc_bytes(sizeof(step_info));
    memset(dfl.d_df_info, 0, sizeof(step_info));
        dfl.dflash_on_ = true;
    fprintf(stderr,
            "[dflash] %s: %d layers, n_embd %d, %d q heads / %d kv, head_dim %d, block %d (n_max %d), %s, "
            "conv k%d/g%d, selector rank %d top-%d, target layers [%s], device %d, %.2f GiB of GGUF weights\n",
            draft_path.c_str(), hp.n_layer, hp.n_embd, hp.n_head, hp.n_head_kv, hp.head_dim, dfl.df_block_, dfl.df_k_,
            hp.is_dflash2 ? "DFlash2" : "DFlash1", hp.conv_k, hp.conv_group, hp.sel_rank, hp.sel_top_k,
            [&] {
                static std::string s;
                s.clear();
                for (int i = 0; i < hp.n_tgt_layer; i++) {
                    s += (i ? "," : "") + std::to_string(hp.tgt_layer[i]);
                }
                return s.c_str();
            }(),
            dfl.df_dev_, (double)dfl.dfm_->bytes() / (1024.0 * 1024.0 * 1024.0));
        fprintf(stderr, "[dflash] draft LM head: %s (%d rows x %d, the target's), ring %d tokens, %s KV\n",
            dfl.df_head_w4_ ? "u4 copy" : "int8", m.output.N, m.output.K, dfl.df_ring_,
            dfl.df_kv_bytes_ == 4 ? "f32" : "f16");
}

// ---------------------------------------------------------------------------
// One draft GEMM: quantize the activation, then the first store that serves the
// shape (native u4/k5/cb4 at M>=2 via nat_gemm, else the grouped int8 matmul).
// `do_split` is always on: nat_gemm's u4/k5 paths read the even/odd activation
// planes, and without them they silently read the *previous* call's activation.
bool engine::df_gemm(const wt & w, const float * x, int xs, float * out, int os, const float * res, int M,
                     const float * up, int us) {
    dnnl_gemm * D = dnnl_for(dfl.df_dev_);
    // PF_DFLASH_NOGEMM=1 skips the draft's projections.  WRONG RESULTS; it is the
    // complement of PF_DFLASH_NOCONV, and together the two price the layer:
    // removing the convs took the five layers 20.8 -> 17.8 ms, so the convs are 3.0
    // and everything else is 17.8 - which for 1040 MB of u4 weights is 58 GB/s
    // against a 4.67 ms / 223 GB/s floor.
    if (si::env::flag("PF_DFLASH_NOGEMM")) {
        return true;
    }
    [[maybe_unused]] static const bool dsplit = [] {
        const char * e = si::env::str("PF_DFLASH_SPLITQ");
        return e ? atoi(e) != 0 : true;
    }();
    // PF_DFLASH_ROWED=1 runs the projection one row at a time.  M == 1 takes the
    // dedicated grouped-scale GEMV, M >= 2 the batched matmul, so this is both the
    // A/B that tells the two apart and the fallback if only the batched one is bad.
    static const bool rowed = [] {
        const char * e = si::env::str("PF_DFLASH_ROWED");
        return e && atoi(e) != 0;
    }();
    if (D == nullptr) {
        return false;
    }
    if (rowed && M > 1) {
        for (int m = 0; m < M; m++) {
            const float * xm = x ? x + (size_t)m * xs : nullptr;
            float * om = out + (size_t)m * os;
            if (!D->quantize(xm, up ? up + (size_t)m * us : nullptr, xs, us, 1, w.K, dsplit)) {
                return false;
            }
            if (!D->gemm_w4(w.data, res ? res + (size_t)m * os : nullptr, 1.0f, 1, w.K, om, os)
                && !D->gemm(w.data, res ? res + (size_t)m * os : nullptr, 1.0f, 1, w.K, om, os)) {
                return false;
            }
        }
        return true;
    }
    if (!D->quantize(x, up, xs, us, M, w.K, dsplit)) {
        return false;
    }
    if (D->gemm_w4(w.data, res, 1.0f, M, w.K, out, os)) {
        return true;
    }
    return D->gemm(w.data, res, 1.0f, M, w.K, out, os);
}

// ---------------------------------------------------------------------------
// Inject the target features of `M` committed tokens into the draft's K/V ring.
//
// The features are the interleaved [token][n_tgt_layer * n_embd] block the
// capture wrote; `feat_row` is the first token to inject.  fc fuses them to the
// draft width, and every draft layer's wk/wv turns that into its own K/V at the
// token's position - which is how the drafter "sees" the committed conversation.
void engine::df_inject(int M, int pos0, int feat_row) {
    const dflash_hp & hp = dfl.dfm_->hp;
    sycl::queue & dq = dev_queue(dfl.df_dev_);
    const int nkv = hp.n_head_kv * hp.head_dim;
    for (int off = 0; off < M; off += kDfInjMax) {
        const int n = std::min(kDfInjMax, M - off);
        const size_t bytes = (size_t)n * hp.n_feat * 4;
        // dfl.d_df_feat is written by capture kernels on the target's per-device
        // queues; wait for those before reading it here.
        dev_queue(dfl.df_dev_).wait();
        dq.memcpy(dfl.d_df_feat_dev, dfl.d_df_feat + (size_t)(feat_row + off) * hp.n_feat, bytes).wait();
                if (!df_gemm(dfl.dfm_->fc, dfl.d_df_feat_dev, hp.n_feat, dfl.d_df_b, hp.n_embd, nullptr, n)) {
            throw std::runtime_error("dflash: fc GEMM failed");
        }
        df_stage(dq, "fc out", dfl.d_df_b, (size_t)n * hp.n_embd);
        rmsnorm_launch(dq, dfl.d_df_b, dfl.dfm_->dev_f32(dfl.dfm_->enc_norm), dfl.d_df_c, n, hp.n_embd, hp.rms_eps);
        if (df_dbg() && si::env::flag("PF_DFLASH_NINV")) {
            dq.wait();
            std::vector<float> oc((size_t)n * hp.n_embd), xc((size_t)n * hp.n_embd);
            std::vector<float> wc2((size_t)hp.n_embd);
            std::memcpy(wc2.data(), dfl.dfm_->enc_norm, (size_t)hp.n_embd * 4);
            dq.memcpy(oc.data(), dfl.d_df_c, oc.size() * 4).wait();
            dq.memcpy(xc.data(), dfl.d_df_b, xc.size() * 4).wait();
            for (int r2 = 0; r2 < std::min(n, 2); r2++) {
                double xs3 = 0, os3 = 0, ws3 = 0;
                for (int k2 = 0; k2 < hp.n_embd; k2++) {
                    xs3 += (double)xc[(size_t)r2 * hp.n_embd + k2] * xc[(size_t)r2 * hp.n_embd + k2];
                    os3 += (double)oc[(size_t)r2 * hp.n_embd + k2] * oc[(size_t)r2 * hp.n_embd + k2];
                    ws3 += (double)wc2[k2] * wc2[k2];
                }
                fprintf(stderr, "[dflash] ninv row %d: in(fc) rms=%.5f  out rms=%.5f  |w| rms=%.5f  eps=%.6g "
                                "n=%d nrows=%d\n",
                        r2, std::sqrt(xs3 / hp.n_embd), std::sqrt(os3 / hp.n_embd), std::sqrt(ws3 / hp.n_embd),
                        hp.rms_eps, hp.n_embd, n);
            }
        }
        df_stage(dq, "injected inp_g", dfl.d_df_c, (size_t)n * hp.n_embd);
        if (si::env::flag("PF_DFLASH_INJP")) {
            // same shape as llama.cpp's DFLASH_REF_HIDDEN dump: row rms/max and the
            // top-8 channels by |value|, so the two can be diffed directly
            for (int r2 = 0; r2 < std::min(n, 2); r2++) {
                std::vector<float> v((size_t)hp.n_embd);
                dq.memcpy(v.data(), dfl.d_df_c + (size_t)r2 * hp.n_embd, v.size() * 4).wait();
                double ss = 0, am = 0;
                for (int j2 = 0; j2 < hp.n_embd; j2++) {
                    ss += (double)v[j2] * v[j2];
                    am = std::max(am, (double)std::fabs(v[j2]));
                }
                fprintf(stderr, "[dflash-inj] inp_g row %d rms=%.5f max=%.4g first=%.6g top:", r2,
                        std::sqrt(ss / hp.n_embd), am, v[0]);
                for (int c = 0; c < 8; c++) {
                    int bi = 0;
                    float bv = -1e30f;
                    for (int j2 = 0; j2 < hp.n_embd; j2++) {
                        if (std::fabs(v[j2]) > bv) {
                            bv = std::fabs(v[j2]);
                            bi = j2;
                        }
                        (void)j2;
                    }
                    fprintf(stderr, " %d=%+.4g", bi, v[bi]);
                    v[bi] = 0.f; // exclude, so the next pick is a different channel
                }
                fprintf(stderr, "\n");
            }
        }
                for (int i = 0; i < n; i++) {
            dfl.h_df_pos[i] = pos0 + off + i;
        }
        dq.memcpy(dfl.d_df_pos, dfl.h_df_pos.data(), (size_t)n * sizeof(int32_t)).wait();
        for (int il = 0; il < hp.n_layer; il++) {
            const dflash_layer_t & L = dfl.dfm_->layers[(size_t)il];
        df_sig_layer = il;
            const int nkv = hp.n_head_kv * hp.head_dim;
            // one fused q|k|v buffer: every segment's out is its row's start, so the
            // row stride is the whole buffer's width, not the segment's
            const int qkv_stride = hp.n_head * hp.head_dim + 2 * nkv;
            if (!df_gemm(L.wk, dfl.d_df_c, hp.n_embd, dfl.d_df_qkv + hp.n_head * hp.head_dim, qkv_stride, nullptr, n)
                || !df_gemm(L.wv, dfl.d_df_c, hp.n_embd, dfl.d_df_qkv + hp.n_head * hp.head_dim + nkv, qkv_stride, nullptr,
                            n)) {
                throw std::runtime_error("dflash: injection wk/wv GEMM failed");
            }
            char * kb = (char *)dfl.d_df_kring + (size_t)il * df_ring_stride_bytes();
            char * vb = (char *)dfl.d_df_vring + (size_t)il * df_ring_stride_bytes();
            df_qknorm_rope_store_launch(dq, dfl.d_df_qkv, dfl.d_df_qkv + hp.n_head * hp.head_dim,
                                        dfl.d_df_qkv + hp.n_head * hp.head_dim + nkv, dfl.dfm_->dev_f32(L.q_norm),
                                        dfl.dfm_->dev_f32(L.k_norm), kb, vb, dfl.d_df_pos, n, hp.n_head, hp.n_head_kv,
                                        hp.head_dim, hp.n_rot, hp.rope_base, hp.rms_eps, dfl.df_ring_, dfl.df_kv_bytes_,
                                        /*do_q=*/false, qkv_stride, df_dn_norm);
        }
    }
    if (si::env::flag("PF_DFLASH_SLOTRMS")) {
        // per-slot rms of the captured features: if the capture still aliases or
        // misses a slot the draft sees one layer's hidden state five times
        std::vector<float> all((size_t)std::max(M, 1) * hp.n_feat);
        dq.memcpy(all.data(), dfl.d_df_feat, all.size() * 4).wait();
        for (int r2 = 0; r2 < std::min(M, 2); r2++) {
            fprintf(stderr, "[dflash] slotrms row %d:", r2);
            for (int t2 = 0; t2 < hp.n_tgt_layer; t2++) {
                double ss = 0;
                for (int k2 = 0; k2 < hp.n_embd; k2++) {
                    const float v = all[(size_t)r2 * hp.n_feat + (size_t)t2 * hp.n_embd + k2];
                    ss += (double)v * v;
                }
                fprintf(stderr, " s%d=%.5f", t2, std::sqrt(ss / hp.n_embd));
            }
            fprintf(stderr, "\n");
        }
    }
    DFDBG("inject M=%d pos0=%d feat_row=%d\n", M, pos0, feat_row);
}

size_t engine::df_ring_stride_bytes() const {
    const dflash_hp & hp = dfl.dfm_->hp;
    return (size_t)dfl.df_ring_ * hp.n_head_kv * hp.head_dim * (size_t)dfl.df_kv_bytes_;
}

// ---------------------------------------------------------------------------
// The draft block: one batched forward over [anchor, MASK x (M-1)].
//
// The attention is non-causal inside the block, so this is a single M-row pass,
// not k autoregressive steps - that is the whole point of the block drafter.
// Afterwards the target's (shared) LM head gives the logits, the top-k kernel the
// candidate sets and df_sel_launch the transition lattice.
void engine::df_block(int pos0, const int32_t * toks, int M) {
    const dflash_hp & hp = dfl.dfm_->hp;
    sycl::queue & dq = dev_queue(dfl.df_dev_);
    const int nh = hp.n_head * hp.head_dim;
    const int nkv = hp.n_head_kv * hp.head_dim;
    // PF_DFLASH_ASCALE multiplies the attention scale.  The draft's scores come out
    // ~2.4 with the 1/sqrt(head_dim) llama.cpp uses, which softmaxes to a near-uniform
    // average over the keys and shrinks the attention output several-fold; this knob
    // is how that hypothesis gets tested rather than argued.
    static const float ascale_mul = [] {
        const char * e = si::env::str("PF_DFLASH_ASCALE");
        return e ? (float)atof(e) : 1.0f;
    }();
    const float attn_scale = ascale_mul / std::sqrt((float)hp.head_dim);
    step_info * inf = dfl.d_df_info;
    reset_step_info(inf);
    inf->n_rows = 1;
    inf->n_real = M;
    inf->tpb = M;
    inf->n_real_row[0] = M;
    inf->pos[0] = pos0;
    inf->slot[0] = 0;
    inf->active[0] = 1;
    inf->pc_active = 0;
    inf->mtp_dt = 0;
    inf->mtp_dry = 0;
    for (int i = 0; i < M; i++) {
        inf->tokens[i] = toks[i];
        inf->img_row[i] = -1;
        dfl.h_df_pos[i] = pos0 + i;
    }
    dq.memcpy(dfl.d_df_pos, dfl.h_df_pos.data(), (size_t)M * sizeof(int32_t)).wait();
    DFDBG("block M=%d pos0=%d tok0=%d\n", M, pos0, toks[0]);
    if (df_dbg() && si::env::flag("PF_DFLASH_PTRS")) {
        // The layer chain relies on dfl.d_df_h and dfl.d_df_c being two DISTINCT buffers that
        // the per-layer swap exchanges: the attn side-1 conv updates dfl.d_df_h in place
        // (out == res), the FFN's output conv writes dfl.d_df_c with dfl.d_df_h as the
        // residual, and the swap is what makes the layer output the next layer's
        // input.  If the two alias, the swap is a no-op and the FFN's output lands on
        // top of the residual it was supposed to be added to.
        fprintf(stderr, "[dflash] ptrs h=%p c=%p b=%p gu=%p\n", (void *)dfl.d_df_h, (void *)dfl.d_df_c, (void *)dfl.d_df_b,
                (void *)dfl.d_df_gu);
    }

    df_seg_on = si::env::flag("PF_DFLASH_SEGTIME");
    for (int i = 0; i < 8; i++) {
        df_seg[i] = 0;
    }
    const auto t_b0 = now_t();
    // 1. embeddings (the target's table; the draft GGUF ships none)
    embed_launch(dq, wptr(0, m.tok_embd.data), m.tok_embd.type, inf, dfl.d_df_h, hp.n_embd, m.tok_embd_row_bytes);
    df_stage(dq, "embed", dfl.d_df_h, (size_t)M * hp.n_embd);
    if (df_dbg() && si::env::flag("PF_DFLASH_EMBCHK")) {
        // The block's own embedding is the one input every host check above shares
        // without validating, so a wrong one makes the whole drafter consistently
        // wrong.  Compare row 0 against the target's table on the host.
        dq.wait();
        std::vector<float> ev((size_t)M * hp.n_embd), er((size_t)hp.n_embd);
        dq.memcpy(ev.data(), dfl.d_df_h, ev.size() * 4).wait();
        for (int r2 = 0; r2 < std::min(M, 2); r2++) {
            dequantize_row(m.tok_embd.type,
                           (const char *)m.tok_embd.data
                               + (size_t)toks[r2] * m.tok_embd_row_bytes,
                           er.data(), (int64_t)hp.n_embd);
            double es = 0, ds = 0, worst = 0;
            for (int k2 = 0; k2 < hp.n_embd; k2++) {
                es += (double)er[k2] * er[k2];
                ds += (double)ev[(size_t)r2 * hp.n_embd + k2] * ev[(size_t)r2 * hp.n_embd + k2];
                worst = std::max(worst, std::fabs((double)er[k2] - ev[(size_t)r2 * hp.n_embd + k2]));
            }
            fprintf(stderr, "[dflash] embchk row %d (tok %d): host rms=%.6f dev rms=%.6f worst=%.6g\n", r2, toks[r2],
                    std::sqrt(es / hp.n_embd), std::sqrt(ds / hp.n_embd), worst);
        }
    }
        // 2. the 5 draft layers
    sycl::event seg_e[8];
    sycl::event ffn_e[2];
    for (int il = 0; il < hp.n_layer; il++) {
        // per-LAYER, like every other marker below.  Leaving it outside the loop made
        // segment 0 measure "before layer 0 -> layer 4's norm", i.e. almost the whole
        // forward, and read as "the RMSNorm is 82% of the draft layer".
        if (df_seg_on) { seg_e[0] = dq.ext_oneapi_submit_barrier(); }
                const dflash_layer_t & L = dfl.dfm_->layers[(size_t)il];
        df_sig_layer = il;
        if (df_dbg() && si::env::flag("PF_DFLASH_SIGALL") && si::env::flag("PF_DFLASH_SIG")) {
            // what this layer actually reads.  rmsnorm of it is xnorm, and RMSNorm is
            // elementwise, so if this tensor is wrong the layer is wrong while every
            // probe downstream stays self-consistent.
            dq.wait();
            df_sig(dq, dfl.d_df_h, M, hp.n_embd, "layer_in");
        }
        rmsnorm_launch(dq, dfl.d_df_h, dfl.dfm_->dev_f32(L.attn_norm), dfl.d_df_b, M, hp.n_embd, hp.rms_eps);
        if (df_seg_on) { seg_e[6] = dq.ext_oneapi_submit_barrier(); }
        df_stage(dq, il == 0 ? "l0 xnorm" : "xnorm", dfl.d_df_b, (size_t)M * hp.n_embd);
        // DFlash2: dynamic depthwise conv in front of the attention projections.
        // llama.cpp applies the *attention* conv to build_attn's output, and
        // build_attn includes wo - so the conv sees the n_embd-wide wo result,
        // not the raw n_head*head_dim attention output, and it runs BEFORE the
        // residual add (which is why df_conv_launch takes the residual).
        static const bool nocv = si::env::flag("PF_DFLASH_NOCONV");
        const bool d2 = hp.is_dflash2 && !nocv;
        if (d2) {
            if (!df_gemm(L.attn_conv_proj, dfl.d_df_b, hp.n_embd, dfl.d_df_dyn, hp.conv_proj, nullptr, M)) {
                throw std::runtime_error("dflash: attn_conv_proj GEMM failed");
            }
            if (df_seg_on) { seg_e[5] = dq.ext_oneapi_submit_barrier(); }
            if (df_dbg() && si::env::flag("PF_DFLASH_BASECHK")) {
                // attn_conv_base is a raw f32 tensor copied straight from the GGUF, so
                // the device copy must be byte-identical.  Layer 4's conv takes an
                // input at cos 0.99 and produces 0.88 while layer 0's takes 1.0 and
                // produces 1.0 - same code, same formula - so one of the conv's other
                // two inputs differs, and this is the one that cannot differ by
                // arithmetic.
                dq.wait();
                const size_t nb4 = (size_t)hp.n_embd * hp.conv_k * 2;
                std::vector<float> db(nb4);
                dq.memcpy(db.data(), dfl.dfm_->dev_f32(L.attn_conv_base), nb4 * 4).wait();
                double worst = 0, rmsd = 0, hs = 0;
                for (size_t z = 0; z < nb4; z++) {
                    const double a = db[z], b = L.attn_conv_base[z];
                    worst = std::max(worst, std::fabs(a - b));
                    rmsd += (a - b) * (a - b);
                    hs += b * b;
                }
                fprintf(stderr, "[dflash] basechk attn blk.%d: worst=%.6g rmsdiff=%.6g hostrms=%.6f\n", il, worst,
                        std::sqrt(rmsd / nb4), std::sqrt(hs / nb4));
            }
            if (df_dbg() && (il == 0 || si::env::flag("PF_DFLASH_PROJCHKALL")) && si::env::flag("PF_DFLASH_PROJCHK")) {
                dq.wait();
                df_proj_check(dq, L.attn_conv_proj, dfl.d_df_b, M, dfl.d_df_dyn, "attn_proj");
            }
            if (df_dbg() && (il == 0 || si::env::flag("PF_DFLASH_SIGALL")) && si::env::flag("PF_DFLASH_SIG")) {
                dq.wait();
                df_sig(dq, dfl.d_df_b, M, hp.n_embd, "xnorm");
                if (const char * bp = si::env::str("PF_DFLASH_BIN")) {
                    // the conv's other two inputs, so the whole conv can be
                    // recomputed on the host and checked against llama.cpp's output
                    const std::string dp = std::string(bp) + ".attn_dyn.bin";
                    const std::string bp2 = std::string(bp) + ".attn_base.bin";
                    std::vector<float> dv((size_t)M * hp.conv_proj);
                    std::vector<float> bv((size_t)hp.n_embd * hp.conv_k * 2);
                    dq.memcpy(dv.data(), dfl.d_df_dyn, dv.size() * 4).wait();
                    dq.memcpy(bv.data(), dfl.dfm_->dev_f32(L.attn_conv_base), bv.size() * 4).wait();
                    if (FILE * f = fopen(dp.c_str(), "wb")) {
                        fwrite(dv.data(), 4, dv.size(), f);
                        fclose(f);
                    }
                    if (FILE * f = fopen(bp2.c_str(), "wb")) {
                        fwrite(bv.data(), 4, bv.size(), f);
                        fclose(f);
                    }
                    fprintf(stderr, "[dflash-sig] conv inputs written (dyn %zu, base %zu)\n", dv.size(), bv.size());
                }
            }
            if (df_dbg() && si::env::flag("PF_DFLASH_SIGALL") && si::env::flag("PF_DFLASH_BIN")) {
                // attn_dynamic is the one conv input with no exact host check: the
                // check shares this kernel's index assumption, so a permutation both
                // agree on is invisible to it.  Zero-pad to n_embd so it lines up with
                // llama.cpp's DFLASH_REF_DYN, whose copy-out is fixed at that width.
                dq.wait();
                const int cp = hp.conv_proj;
                // row by row: the device buffer is packed at cp, the file (like
                // llama.cpp's padded nextn copy-out) is at n_embd.  Copying it as one
                // block leaves every row past the first reading as zeros.
                std::vector<float> dv2((size_t)M * hp.n_embd, 0.f);
                std::vector<float> tmp((size_t)cp);
                for (int rr = 0; rr < M; rr++) {
                    dq.memcpy(tmp.data(), dfl.d_df_dyn + (size_t)rr * cp, (size_t)cp * 4).wait();
                    std::memcpy(dv2.data() + (size_t)rr * hp.n_embd, tmp.data(), (size_t)cp * 4);
                }
                const std::string dp2 = std::string(si::env::str("PF_DFLASH_BIN")) + ".dyn.L" + std::to_string(il) + ".b"
                                        + std::to_string(sig_block_now()) + ".bin";
                if (FILE * f = fopen(dp2.c_str(), "wb")) {
                    fwrite(dv2.data(), 4, dv2.size(), f);
                    fclose(f);
                }
            }
            df_conv_launch(dq, dfl.d_df_b, dfl.d_df_dyn, dfl.dfm_->dev_f32(L.attn_conv_base), dfl.d_df_c, nullptr, M, hp.n_embd,
                           hp.n_embd, hp.conv_k, hp.conv_group, hp.conv_proj, /*side=*/0);
            if (si::env::flag("PF_DFLASH_CONVCHK") && (il == 0 || si::env::flag("PF_DFLASH_CONVCHKALL"))) {
                df_conv_check(dq, dfl.d_df_b, dfl.d_df_dyn, dfl.dfm_->dev_f32(L.attn_conv_base), dfl.d_df_c, nullptr, M, hp.n_embd,
                              hp.conv_k, hp.conv_group, hp.conv_proj, 0, "attn side0");
            }
        } else {
            dq.memcpy(dfl.d_df_c, dfl.d_df_b, (size_t)M * hp.n_embd * 4).wait();
        }
        df_stage(dq, il == 0 ? "l0 attnconv" : "attnconv", dfl.d_df_c, (size_t)M * hp.n_embd);
        if (df_dbg() && (il == 0 || si::env::flag("PF_DFLASH_SIGALL")) && si::env::flag("PF_DFLASH_SIG")) {
            dq.wait();
            df_sig(dq, dfl.d_df_c, M, hp.n_embd, "noise_norm");
        }
        if (df_dbg() && il == 0 && si::env::flag("PF_DFLASH_ROWRMS")) {
            dq.wait();
            df_rowrms(dq, dfl.d_df_c, M, hp.n_embd, "attn_conv_out");
        }
        df_stage(dq, il == 0 ? "l0 attn_cbase" : "attn_cbase", dfl.dfm_->dev_f32(L.attn_conv_base),
                 (size_t)hp.n_embd * hp.conv_k * 2);
        // one fused q|k|v buffer: every segment's out is the *row* start, so the
        // row stride is the whole buffer's, not the segment's width
        const int qkv_stride = nh + 2 * nkv;
        // q, k and v all read the conv's output.  wq drives the softmax's peakedness:
        // with the Q segment left at zero every score is 0, the softmax becomes a
        // uniform average over all keys, and the attention output comes out several
        // times too small - which silently wrecks the whole draft.
        if (df_seg_on) { seg_e[1] = dq.ext_oneapi_submit_barrier(); }
        if (!df_gemm(L.wq, dfl.d_df_c, hp.n_embd, dfl.d_df_qkv, qkv_stride, nullptr, M)) {
            throw std::runtime_error("dflash: wq GEMM failed");
        }
        // K and V are projected from the SAME conv output.  They were missing here
        // entirely, so the block's attention read whatever the injection path last
        // left in the fused buffer: the ring's committed positions were valid (the
        // injection writes them), which is why an anchor row still tracked the
        // reference, but the block's own rows attended to stale K/V.  That asymmetry
        // - anchor close, mask rows off - is the signature this had all along.
        if (!df_gemm(L.wk, dfl.d_df_c, hp.n_embd, dfl.d_df_qkv + nh, qkv_stride, nullptr, M)
            || !df_gemm(L.wv, dfl.d_df_c, hp.n_embd, dfl.d_df_qkv + nh + nkv, qkv_stride, nullptr, M)) {
            throw std::runtime_error("dflash: wk/wv GEMM failed");
        }
        // PF_DFLASH_RAWQK=1 leaves the projection un-normed, so a host check can
        // separate a wrong GEMM from a wrong norm/rope (both are Q4_K x f32)
                // PF_DFLASH_NONORM=1: skip the per-head q/k rms-norm, to attribute a
        // mismatch between our K rms and llama.cpp's to the norm itself
        static const bool nonorm = si::env::flag("PF_DFLASH_NONORM");
        static const float k1w[1] = {1.0f};
        const float * qw = nonorm ? k1w : dfl.dfm_->dev_f32(L.q_norm);
        const float * kw = nonorm ? k1w : dfl.dfm_->dev_f32(L.k_norm);
        df_qknorm_rope_store_launch(dq, dfl.d_df_qkv, dfl.d_df_qkv + nh, dfl.d_df_qkv + nh + nkv, qw,
                                    kw, (char *)dfl.d_df_kring + (size_t)il * df_ring_stride_bytes(),
                                    (char *)dfl.d_df_vring + (size_t)il * df_ring_stride_bytes(), dfl.d_df_pos, M, hp.n_head,
                                    hp.n_head_kv, hp.head_dim, hp.n_rot, hp.rope_base, hp.rms_eps, dfl.df_ring_,
                                    dfl.df_kv_bytes_, /*do_q=*/true, qkv_stride, df_dn_norm);

        df_attn_launch(dq, dfl.d_df_qkv, (char *)dfl.d_df_kring + (size_t)il * df_ring_stride_bytes(),
                       (char *)dfl.d_df_vring + (size_t)il * df_ring_stride_bytes(), dfl.d_df_partials, dfl.d_df_pos, M, M,
                       hp.n_head, hp.n_head_kv, hp.head_dim, dfl.df_ring_, dfl.df_swa_, dfl.df_splits_, attn_scale,
                       dfl.df_kv_bytes_, qkv_stride);
        df_attn_combine_launch(dq, dfl.d_df_partials, dfl.d_df_b, M, hp.n_head, hp.head_dim, dfl.df_splits_,
                               dfl.dfm_->dev_f32(dfl.dfm_->layers[(size_t)il].attn_sinks));
        if (df_dbg() && il == 0 && si::env::flag("PF_DFLASH_ATTNCHK")) {
            dq.wait();
            std::vector<int32_t> pv((size_t)M);
            dq.memcpy(pv.data(), dfl.d_df_pos, pv.size() * 4).wait();
            df_attn_check(dq, dfl.d_df_qkv, qkv_stride, M,
                          (char *)dfl.d_df_kring + (size_t)il * df_ring_stride_bytes(),
                          (char *)dfl.d_df_vring + (size_t)il * df_ring_stride_bytes(), pv.data(), hp.n_head,
                          hp.n_head_kv, hp.head_dim, pos0, dfl.df_swa_, dfl.df_ring_, attn_scale, dfl.d_df_b);
        }
        if (df_dbg() && il == 0 && si::env::flag("PF_DFLASH_ROWRMS")) {
            dq.wait();
            df_rowrms(dq, dfl.d_df_b, M, hp.n_head * hp.head_dim, "attn_out");
        }

        // wo -> (DFlash2 conv) -> residual add
        if (d2) {
        if (df_seg_on) { seg_e[2] = dq.ext_oneapi_submit_barrier(); }
            if (!df_gemm(L.wo, dfl.d_df_b, nh, dfl.d_df_c, hp.n_embd, nullptr, M)) {
                throw std::runtime_error("dflash: wo GEMM failed");
            }
            // The conv's input is the *wo output* (dfl.d_df_c), not the normed
            // pre-attention activation (dfl.d_df_b), and llama.cpp's build_dflash2_conv
            // takes no residual at all - the sublayer's residual add is the separate
            // `ffn_inp = conv_out + inpL` that follows.  So the layer input (dfl.d_df_h)
            // is the residual here, and out == residual is safe because every work
            // item reads and writes exactly one element.
            df_stage(dq, il == 0 ? "l0 wo" : "wo", dfl.d_df_c, (size_t)M * hp.n_embd);
        if (df_dbg() && (il == 0 || si::env::flag("PF_DFLASH_SIGALL")) && si::env::flag("PF_DFLASH_SIG")) {
            dq.wait();
            df_sig(dq, dfl.d_df_c, M, hp.n_embd, "cur");
        }
            if (df_dbg() && si::env::flag("PF_DFLASH_GEMMCHK")) {
                dq.wait();
                char tg[32];
                snprintf(tg, sizeof(tg), "wo_l%d", il);
                df_gemm_check(dq, L.wo, dfl.d_df_b, nh, dfl.d_df_c, hp.n_embd, M, nullptr, 0, tg, 4);
            }
            if (df_dbg() && si::env::flag("PF_DFLASH_RING") && il == 0) {
                // per-position ring V rms around the block: if the committed
                // positions are empty the softmax averages them in as zeros and the
                // attention output comes out several times too small
                // the ring is f16, so copy it into a uint16 buffer and reinterpret
                // per element - reading it into a float buffer and casting would
                // pair up adjacent halves
                std::vector<uint16_t> vr((size_t)dfl.df_ring_ * hp.n_head_kv * hp.head_dim);
                dq.memcpy(vr.data(), (char *)dfl.d_df_vring + (size_t)il * df_ring_stride_bytes(),
                          vr.size() * 2).wait();
                const int pv = pos0 + M - 1;
                const int lo2 = std::max(0, pv - 6);
                // Q rms drives how peaked the softmax is: a near-uniform softmax
                // (too-flat scores) averages many V rows and shrinks the output
                {
                    std::vector<float> qv((size_t)M * (hp.n_head * hp.head_dim + 2 * hp.n_head_kv * hp.head_dim));
                    dq.memcpy(qv.data(), dfl.d_df_qkv, qv.size() * 4).wait();
                    double qs2 = 0, ks2 = 0;
                    const int nhq = hp.n_head * hp.head_dim;
                    for (int k2 = 0; k2 < nhq; k2++) {
                        qs2 += (double)qv[k2] * qv[k2];
                    }
                    for (int k2 = 0; k2 < hp.n_head_kv * hp.head_dim; k2++) {
                        const size_t cell = (size_t)(pv % dfl.df_ring_) * hp.n_head_kv * hp.head_dim + k2;
                        const double v = f16_bits(vr[cell]);
                        ks2 += v * v;
                    }
                    // K in the qkv buffer (post norm+rope, pre ring write) vs K in
                    // the ring: rms(k_norm weights) is the invariant both must have,
                    // so this says whether the reduction or the ring store is wrong
                    // a dedicated K-ring copy: the earlier probe reused the V ring
                    // buffer, so the "K(ring)" number it printed was V's
                    std::vector<uint16_t> krng((size_t)dfl.df_ring_ * hp.n_head_kv * hp.head_dim);
                    dq.memcpy(krng.data(), (char *)dfl.d_df_kring + (size_t)il * df_ring_stride_bytes(),
                              krng.size() * 2).wait();
                    double kring2 = 0;
                    for (int k2 = 0; k2 < hp.n_head_kv * hp.head_dim; k2++) {
                        const size_t cell = (size_t)(pv % dfl.df_ring_) * hp.n_head_kv * hp.head_dim + k2;
                        const double v = f16_bits(krng[cell]);
                        kring2 += v * v;
                    }
                    double kbuf2 = 0;
                    for (int k2 = 0; k2 < hp.n_head_kv * hp.head_dim; k2++) {
                        kbuf2 += (double)qv[nhq + k2] * qv[nhq + k2];
                    }
                    double wrms = 0;
                    {
                        std::vector<float> kn((size_t)hp.head_dim);
                        std::memcpy(kn.data(), dfl.dfm_->layers[0].k_norm, (size_t)hp.head_dim * 4);
                        for (int k2 = 0; k2 < hp.head_dim; k2++) {
                            wrms += (double)kn[k2] * kn[k2];
                        }
                        wrms = std::sqrt(wrms / hp.head_dim);
                    }
                    fprintf(stderr, "[dflash] qk il=0: Q rms=%.4f (q_norm rms=?)  K(buf) rms=%.4f  K(ring) rms=%.4f  "
                                    "k_norm rms=%.4f\n",
                            std::sqrt(qs2 / nhq), std::sqrt(kbuf2 / (hp.n_head_kv * hp.head_dim)),
                            std::sqrt(kring2 / (hp.n_head_kv * hp.head_dim)), wrms);
                }
                fprintf(stderr, "[dflash] ring il=0 V rms (pos %d..%d):", lo2, pv);
                for (int q2 = lo2; q2 <= pv; q2++) {
                    const size_t cell = (size_t)(q2 % dfl.df_ring_) * hp.n_head_kv * hp.head_dim;
                    double ss = 0;
                    for (int j2 = 0; j2 < hp.n_head_kv * hp.head_dim; j2++) {
                        const double v = f16_bits(vr[cell + j2]);
                        ss += v * v;
                    }
                    fprintf(stderr, " %d:%.2f", q2, std::sqrt(ss / (hp.n_head_kv * hp.head_dim)));
                }
                fprintf(stderr, "\n");
            }
            df_conv_launch(dq, dfl.d_df_c, dfl.d_df_dyn, dfl.dfm_->dev_f32(L.attn_conv_base), dfl.d_df_h, dfl.d_df_h, M, hp.n_embd,
                           hp.n_embd, hp.conv_k, hp.conv_group, hp.conv_proj, /*side=*/1);
            if (si::env::flag("PF_DFLASH_CONVCHK") && (il == 0 || si::env::flag("PF_DFLASH_CONVCHKALL"))) {
                df_conv_check(dq, dfl.d_df_c, dfl.d_df_dyn, dfl.dfm_->dev_f32(L.attn_conv_base), dfl.d_df_h, dfl.d_df_h, M, hp.n_embd,
                              hp.conv_k, hp.conv_group, hp.conv_proj, 1, "attn side1");
            }
        } else {
            df_add_launch(dq, dfl.d_df_b, dfl.d_df_h, dfl.d_df_c, M, hp.n_embd);
        }
        df_stage(dq, il == 0 ? "l0 post-attn" : "post-attn", dfl.d_df_h, (size_t)M * hp.n_embd);
        if (df_dbg() && (il == 0 || si::env::flag("PF_DFLASH_SIGALL")) && si::env::flag("PF_DFLASH_SIG")) {
            dq.wait();
            df_sig(dq, dfl.d_df_h, M, hp.n_embd, "ffn_inp");
        }
        if (df_dbg() && il == 0 && si::env::flag("PF_DFLASH_ROWRMS")) {
            dq.wait();
            // ffn_inp = attn_conv_out + layer input; the reference's REF_FFNIN
            // prints this tensor one row at a time
            df_rowrms(dq, dfl.d_df_h, M, hp.n_embd, "ffn_inp");
        }
        // FFN: norm -> conv -> gate/up -> down -> conv -> residual add
        if (df_seg_on) { seg_e[3] = dq.ext_oneapi_submit_barrier(); }
        rmsnorm_launch(dq, dfl.d_df_h, dfl.dfm_->dev_f32(L.ffn_norm), dfl.d_df_b, M, hp.n_embd, hp.rms_eps);
        if (d2) {
            if (!df_gemm(L.ffn_conv_proj, dfl.d_df_b, hp.n_embd, dfl.d_df_dyn, hp.conv_proj, nullptr, M)) {
                throw std::runtime_error("dflash: ffn_conv_proj GEMM failed");
            }
            if (df_dbg() && (il == 0 || si::env::flag("PF_DFLASH_PROJCHKALL")) && si::env::flag("PF_DFLASH_PROJCHK")) {
                dq.wait();
                df_proj_check(dq, L.ffn_conv_proj, dfl.d_df_b, M, dfl.d_df_dyn, "ffn_proj");
            }
            df_conv_launch(dq, dfl.d_df_b, dfl.d_df_dyn, dfl.dfm_->dev_f32(L.ffn_conv_base), dfl.d_df_c, nullptr, M, hp.n_embd,
                           hp.n_embd, hp.conv_k, hp.conv_group, hp.conv_proj, /*side=*/0);
            if (si::env::flag("PF_DFLASH_CONVCHK") && (il == 0 || si::env::flag("PF_DFLASH_CONVCHKALL"))) {
                df_conv_check(dq, dfl.d_df_b, dfl.d_df_dyn, dfl.dfm_->dev_f32(L.ffn_conv_base), dfl.d_df_c, nullptr, M, hp.n_embd,
                              hp.conv_k, hp.conv_group, hp.conv_proj, 0, "ffn side0");
            }
        } else {
            dq.memcpy(dfl.d_df_c, dfl.d_df_b, (size_t)M * hp.n_embd * 4).wait();
        }
        df_stage(dq, il == 0 ? "l0 ffn_xnorm" : "ffn_xnorm", dfl.d_df_c, (size_t)M * hp.n_embd);
        if (df_dbg() && (il == 0 || si::env::flag("PF_DFLASH_SIGALL")) && si::env::flag("PF_DFLASH_SIG")) {
            dq.wait();
            df_sig(dq, dfl.d_df_c, M, hp.n_embd, "ffn_conv_in");
        }
        if (df_dbg() && il == 0 && si::env::flag("PF_DFLASH_ROWRMS")) {
            dq.wait();
            df_rowrms(dq, dfl.d_df_c, M, hp.n_embd, "ffn_conv_in");
        }
        // gate and up in one fused [M][2*n_ff] buffer, laid out [gate | up] per row
        // so ffn_down below can read the up half as its own activation
        if (df_seg_on) { ffn_e[0] = dq.ext_oneapi_submit_barrier(); }
        if (!df_gemm(L.ffn_gate, dfl.d_df_c, hp.n_embd, dfl.d_df_gu, 2 * hp.n_ff, nullptr, M)
            || !df_gemm(L.ffn_up, dfl.d_df_c, hp.n_embd, dfl.d_df_gu + hp.n_ff, 2 * hp.n_ff, nullptr, M)) {
            throw std::runtime_error("dflash: ffn gate/up GEMM failed");
        }
        const bool dnchk = si::env::flag("PF_DFLASH_GEMMCHK") && il == 0;
        if (dnchk) {
            std::vector<float> gu((size_t)M * 2 * hp.n_ff);
            dq.memcpy(gu.data(), dfl.d_df_gu, gu.size() * 4).wait();
            double gn = 0, un = 0, am = 0;
            for (int m = 0; m < M; m++) {
                for (int k = 0; k < hp.n_ff; k++) {
                    const float g = gu[(size_t)m * 2 * hp.n_ff + k];
                    const float u2 = gu[(size_t)m * 2 * hp.n_ff + hp.n_ff + k];
                    gn += (double)g * g;
                    un += (double)u2 * u2;
                    am = std::max(am, (double)std::fabs(silu_test(g) * u2));
                }
            }
            fprintf(stderr, "[dflash] ffngu: |gate|=%.4f |up|=%.4f max|silu(g)*u|=%.4g\n",
                    std::sqrt(gn / (M * hp.n_ff)), std::sqrt(un / (M * hp.n_ff)), am);
        }
        std::vector<float> chk_gu;
        if (dnchk) {
            chk_gu.resize((size_t)M * 2 * hp.n_ff);
            dq.memcpy(chk_gu.data(), dfl.d_df_gu, chk_gu.size() * 4).wait();
        }
        static const bool nosilu = si::env::flag("PF_DFLASH_NOSILU");
        if (df_seg_on) { ffn_e[1] = dq.ext_oneapi_submit_barrier(); }
        if (!df_gemm(L.ffn_down, dfl.d_df_gu, 2 * hp.n_ff, dfl.d_df_b, hp.n_embd, nullptr, M,
                     nosilu ? nullptr : dfl.d_df_gu + hp.n_ff, 2 * hp.n_ff)) {
            throw std::runtime_error("dflash: ffn_down GEMM failed");
        }
        if (dnchk) {
            // the FFN activation's own dynamic range: an int8 per-32 quantizer
            // cannot represent a 100x outlier without wrecking the other 31 lanes
            {
                std::vector<float> chk((size_t)M * 2 * hp.n_ff);
                dq.memcpy(chk.data(), dfl.d_df_gu, chk.size() * 4).wait();
                for (int r2 = 0; r2 < M; r2++) {
                    const float * g2 = chk.data() + (size_t)r2 * 2 * hp.n_ff;
                    const float * u2 = g2 + hp.n_ff;
                    double sg = 0, mx2 = 0, su = 0, mu = 0;
                    for (int j2 = 0; j2 < hp.n_ff; j2++) {
                        sg += (double)g2[j2] * g2[j2];
                        su += (double)u2[j2] * u2[j2];
                        mx2 = std::max(mx2, (double)std::fabs(g2[j2]));
                        mu = std::max(mu, (double)std::fabs(u2[j2]));
                    }
                    fprintf(stderr, "[dflash] ffndyn row %d: gate rms=%.4g max=%.4g (%.0fx) up rms=%.4g max=%.4g\n",
                            r2, std::sqrt(sg / hp.n_ff), mx2, mx2 / (std::sqrt(sg / hp.n_ff) + 1e-30),
                            std::sqrt(su / hp.n_ff), mu);
                }
            }
            std::vector<float> dref((size_t)M * hp.n_embd), dr((size_t)M * hp.n_embd), wr((size_t)L.ffn_down.K);
            dq.memcpy(dr.data(), dfl.d_df_b, dr.size() * 4).wait();
            for (int n2 = 0; n2 < hp.n_embd; n2 += 53) {
                dequantize_row(L.ffn_down.type,
                               (const char *)L.ffn_down.data + (size_t)n2 * quant_row_bytes(L.ffn_down.type,
                                                                                              L.ffn_down.K),
                               wr.data(), (int64_t)L.ffn_down.K);
                for (int m = 0; m < M; m++) {
                    double acc = 0;
                    for (int k = 0; k < L.ffn_down.K; k++) {
                        const float g = chk_gu[(size_t)m * 2 * hp.n_ff + k];
                        acc += (double)wr[(size_t)k] * (double)(g / (1.0f + sycl::exp(-g))
                                                               * chk_gu[(size_t)m * 2 * hp.n_ff + hp.n_ff + k]);
                    }
                    dref[(size_t)m * hp.n_embd + n2] = (float)acc;
                }
            }
            // only every 53rd column was computed, so both sums must run over the
            // same sampled columns - summing the whole row deflates the host rms by
            // sqrt(53) and reads as a 7x "GEMM error" that does not exist
            double xr = 0, drr = 0;
            int cnt = 0;
            for (size_t i = 0; i < dref.size(); i++) {
                if (i % 53 != 0) {
                    continue;
                }
                xr += (double)dref[i] * dref[i];
                drr += (double)dr[i] * dr[i];
                cnt++;
            }
            fprintf(stderr, "[dflash] gemmchk ffn_down  host rms=%.4f dev rms=%.4f ratio=%.4f (n=%d)\n",
                    std::sqrt(xr / cnt), std::sqrt(drr / cnt), std::sqrt(drr / (xr + 1e-30)), cnt);

        }
        if (d2) {
            // the conv writes the new residual stream; dfl.d_df_h is both the input
            // (the residual term) and the destination
            df_stage(dq, il == 0 ? "l0 ffn_down" : "ffn_down", dfl.d_df_b, (size_t)M * hp.n_embd);
            df_conv_launch(dq, dfl.d_df_b, dfl.d_df_dyn, dfl.dfm_->dev_f32(L.ffn_conv_base), dfl.d_df_c, dfl.d_df_h, M, hp.n_embd,
                           hp.n_embd, hp.conv_k, hp.conv_group, hp.conv_proj, /*side=*/1);
            if (si::env::flag("PF_DFLASH_CONVCHK") && (il == 0 || si::env::flag("PF_DFLASH_CONVCHKALL"))) {
                df_conv_check(dq, dfl.d_df_b, dfl.d_df_dyn, dfl.dfm_->dev_f32(L.ffn_conv_base), dfl.d_df_c, dfl.d_df_h, M, hp.n_embd,
                              hp.conv_k, hp.conv_group, hp.conv_proj, 1, "ffn side1");
            }
        } else {
            df_add_launch(dq, dfl.d_df_b, dfl.d_df_h, dfl.d_df_c, M, hp.n_embd);
        }
        if (df_dbg() && si::env::flag("PF_DFLASH_SIGALL") && si::env::flag("PF_DFLASH_BIN")) {
            // dfl.d_df_c, NOT dfl.d_df_h: the FFN's output conv writes dfl.d_df_c with dfl.d_df_h as
            // the residual, so the layer output is dfl.d_df_c and dfl.d_df_h still holds
            // ffn_inp.  The swap below is what publishes it.
            dq.wait();
            df_sig_layer = il;
            df_sig(dq, dfl.d_df_c, M, hp.n_embd, "layer_out");
        }
        df_stage(dq, il == 0 ? "l0 layer_out" : "layer_out", dfl.d_df_h, (size_t)M * hp.n_embd);
        // The sublayer residual adds go into dfl.d_df_h (input + attention), but the
        // FFN's output conv writes dfl.d_df_c, so the swap is what makes the layer's
        // output the next layer's input - and it must happen on EVERY layer,
        // including the last, or the final norm below reads input+attention
        // instead of the last layer's output.
        if (df_seg_on) { seg_e[4] = dq.ext_oneapi_submit_barrier(); }
        if (df_seg_on) {
            // The markers are NOT in index order - program order is 0, 6, 5, 1, 3, 4 -
            // so they must be differenced in that order.  Indexing them 0..5
            // differenced two early markers against two late ones and printed
            // negative segment times.
            df_seg[0] += df_ts_ms(seg_e[6], false) - df_ts_ms(seg_e[0], true);
            df_seg[1] += df_ts_ms(seg_e[5], false) - df_ts_ms(seg_e[6], true);
            df_seg[2] += df_ts_ms(seg_e[1], false) - df_ts_ms(seg_e[5], true);
            df_seg[3] += df_ts_ms(seg_e[3], false) - df_ts_ms(seg_e[1], true);
            df_seg[5] += df_ts_ms(ffn_e[0], false) - df_ts_ms(seg_e[3], true);
            df_seg[6] += df_ts_ms(ffn_e[1], false) - df_ts_ms(ffn_e[0], true);
            df_seg[7] += df_ts_ms(seg_e[4], false) - df_ts_ms(ffn_e[1], true);
            if (il == 0) { df_seg_dev0 = seg_e[0]; }
            df_seg_dev1 = seg_e[4];
        }

        std::swap(dfl.d_df_h, dfl.d_df_c);
    }
    if (df_seg_on) {
        dq.wait();
        // The markers are NOT in index order - program order is 0, 6, 5, 1, 3 - and
        // two of them sit inside `if (d2)`, so they must be differenced in program
        // order and the FFN pair kept in its own array.  Getting this wrong made
        // segment 0 span "before layer 0 -> layer 4's norm" (i.e. the whole forward)
        // and print the RMSNorm as 82% of the draft layer.
        const double devspan = df_ts_ms(df_seg_dev1, false) - df_ts_ms(df_seg_dev0, true);
        const double sum = df_seg[0] + df_seg[1] + df_seg[2] + df_seg[3] + df_seg[5] + df_seg[6] + df_seg[7];
        fprintf(stderr,
                "[dflash-sg] anorm=%.2f acproj=%.2f aconv0=%.2f attn=%.2f | "
                "ffn_norm+cproj+cconv=%.2f gate_up=%.2f down+cconv=%.2f "
                "| sum=%.2f devspan=%.2f hostwall=%.2f ms\n",
                df_seg[0], df_seg[1], df_seg[2], df_seg[3], df_seg[5], df_seg[6], df_seg[7], sum, devspan,
                ms_t(t_b0, now_t()));
    }
    const auto t_b1 = now_t();
    // 3. final norm (also the selector's gate input), the shared LM head, the
    //    candidate sets and the transition lattice
    df_stage(dq, "pre_out_norm", dfl.d_df_h, (size_t)M * hp.n_embd);
    rmsnorm_launch(dq, dfl.d_df_h, dfl.dfm_->dev_f32(dfl.dfm_->output_norm), dfl.d_df_b, M, hp.n_embd, hp.rms_eps);
    if (df_dbg() && si::env::flag("PF_DFLASH_SIG") && si::env::flag("PF_DFLASH_BIN")) {
        // The final normed hidden, i.e. what the head and the selector's gate both
        // read.  Comparing it against llama.cpp's DFLASH_REF_HIDDEN tap is what says
        // whether a wrong top-k is accumulated layer error or a readout bug.
        dq.wait();
        std::string p2 = std::string(si::env::str("PF_DFLASH_BIN")) + ".final.bin";
        std::vector<float> fv((size_t)M * hp.n_embd);
        dq.memcpy(fv.data(), dfl.d_df_b, fv.size() * 4).wait();
        if (FILE * f = fopen(p2.c_str(), "wb")) {
            fwrite(fv.data(), 4, fv.size(), f);
            fclose(f);
        }
    }
    df_stage(dq, "out_norm", dfl.d_df_b, (size_t)M * hp.n_embd);
        if (!df_gemm_head(dfl.d_df_b, M)) {
        throw std::runtime_error("dflash: LM head GEMM failed");
    }
        if (!hp.is_dflash2) {
        return;
    }
    if (df_dbg() && si::env::flag("PF_DFLASH_VRING")) {
        // the block rows' own V, per row: if the mask rows' V is short the whole
        // attention output (and so the hidden) is, which is what a wrong V looks like
        std::vector<float> vr((size_t)dfl.df_ring_ * hp.n_head_kv * hp.head_dim);
        dev_queue(dfl.df_dev_).memcpy(vr.data(), dfl.d_df_vring, vr.size() * 4).wait();
        for (int r2 = 0; r2 < M; r2++) {
            double ss = 0;
            const size_t cell = (size_t)((pos0 + r2) % dfl.df_ring_) * hp.n_head_kv;
            for (int j2 = 0; j2 < hp.n_head_kv * hp.head_dim; j2++) {
                ss += (double)vr[cell * hp.head_dim + j2] * vr[cell * hp.head_dim + j2];
            }
            fprintf(stderr, "[dflash] vring row %d (pos %d) rms=%.4f top:", r2, pos0 + r2,
                    std::sqrt(ss / (hp.n_head_kv * hp.head_dim)));
            {
                // compare the top channels against llama.cpp's REF_V dump: same
                // indices and signs means the weights agree and only the scale is off
                std::vector<std::pair<float, int> > top;
                for (int j2 = 0; j2 < hp.n_head_kv * hp.head_dim; j2++) {
                    top.push_back(std::make_pair(vr[cell * hp.head_dim + j2], j2));
                }
                std::partial_sort(top.begin(), top.begin() + 6, top.end(),
                                  std::greater<std::pair<float, int> >());
                for (int k = 0; k < 6; k++) {
                    fprintf(stderr, " %d=%+.4g", top[k].second, top[k].first);
                }
            }
            fprintf(stderr, " per-head:");
            for (int kh2 = 0; kh2 < hp.n_head_kv; kh2++) {
                double s2 = 0;
                for (int d2 = 0; d2 < hp.head_dim; d2++) {
                    const double v = vr[cell * hp.head_dim + kh2 * hp.head_dim + d2];
                    s2 += v * v;
                }
                fprintf(stderr, " h%d=%.2f", kh2, std::sqrt(s2 / hp.head_dim));
            }
            fprintf(stderr, "\n");
        }
    }
    const int K = hp.sel_top_k;
    if (tt_dbg) { tt_e[0] = dq.ext_oneapi_submit_barrier(); }
    const auto tt0 = tt_now();
    df_topk_launch(dq, d_logits, m.hp.n_vocab, dfl.d_df_ids, dfl.d_df_vals, M, K, dfl.d_df_pids, dfl.d_df_pvals, dfl.df_slices_);
    const auto tt1 = tt_now();
    if (!df_gemm(dfl.dfm_->sel_hidden, dfl.d_df_b, hp.n_embd, dfl.d_df_gate, hp.sel_rank, nullptr, M)) {
        throw std::runtime_error("dflash: selector_hidden GEMM failed");
    }
    const auto tt2 = tt_now();
    if (tt_dbg) { tt_e[1] = dq.ext_oneapi_submit_barrier(); }
    if (df_dbg()) {
        // the lattice score is unary + <A[p]*gate(h_i), B[c]>; a collapsed gate
        // leaves the scores at the unary alone, which is what a small lattice means
        std::vector<float> gg((size_t)M * hp.sel_rank);
        std::vector<float> vv((size_t)M * K);
        dq.memcpy(gg.data(), dfl.d_df_gate, gg.size() * 4).wait();
        dq.memcpy(vv.data(), dfl.d_df_vals, vv.size() * 4).wait();
        double gs = 0, us = 0, us1 = 0;
        for (size_t i2 = 0; i2 < gg.size(); i2++) {
            gs += (double)gg[i2] * gg[i2];
        }
        for (size_t i2 = 0; i2 < vv.size(); i2++) {
            us += (double)vv[i2] * vv[i2];
        }
        for (int c2 = 0; c2 < K; c2++) {
            us1 += (double)vv[1 * K + c2] * vv[1 * K + c2];
        }
        fprintf(stderr, "[dflash] gate rms=%.5f unary rms=%.5f row1 rms=%.4f\n",
                std::sqrt(gs / gg.size()), std::sqrt(us / vv.size()), std::sqrt(us1 / K));
    }
    df_sel_launch(dq, dfl.d_df_ids, dfl.d_df_vals, dfl.d_df_gate, dfl.df_sel_pv_, dfl.df_sel_ps_, dfl.df_sel_po_, dfl.df_sel_nv_, dfl.df_sel_ns_,
                  dfl.df_sel_no_, dfl.d_df_lattice, toks[0], M, m.hp.n_vocab, hp.sel_rank, K);
    if (tt_dbg) { tt_e[2] = dq.ext_oneapi_submit_barrier(); }
    if (si::env::flag("PF_DFLASH_BTIME")) {
        const auto t_b4 = now_t();
        fprintf(stderr, "[dflash-bt] embed+layers=%.1f outnorm+head+topk+sel=%.1f | total=%.1f ms\n",
                ms_t(t_b0, t_b1), ms_t(t_b1, t_b4), ms_t(t_b0, t_b4));
    }
}

// The shared LM head: the draft's readout goes through the private u4 copy when
// there is one (0.625 vs 1.0625 B/weight, the same trade PF_MTP_HEAD_W4 makes).

bool engine::df_gemm_head(const float * x, int M) {
    dnnl_gemm * D = dnnl_for(dfl.df_dev_);
    const wt & head = m.output;
    if (!D->quantize(x, nullptr, m.hp.n_embd, 0, M, head.K, /*do_split=*/true)) {
        return false;
    }
    if (dfl.df_head_key_ && D->gemm_w4(dfl.df_head_key_, nullptr, 1.0f, M, head.K, d_logits, m.hp.n_vocab)) {
        return true;
    }
    if (D->gemm(wkey(dfl.df_dev_, head.data), nullptr, 1.0f, M, head.K, d_logits, m.hp.n_vocab)) {
        if (si::env::flag("PF_DFLASH_HEADCHK") && M <= 4) {
            // Compare against an exact fp32 dequant of the GGUF blocks: the head is
            // the one readout with no exact path (gemm_w4 / oneDNN int8 only), so
            // this is the only way to tell a flat or biased logit row from a
            // genuinely different hidden state.  Id 9564 sitting at the top of every
            // row of an otherwise flat distribution is exactly the shape a broken
            // weight row produces.
            sycl::queue & hq = dev_queue(dfl.df_dev_);
            hq.wait();
            const int K2 = head.K;
            const int nv = m.hp.n_vocab;
            std::vector<float> hx((size_t)K2), lg((size_t)nv), acc((size_t)nv), rw((size_t)K2);
            const int rb = quant_row_bytes(head.type, K2);
            for (int r = 0; r < M; r++) {
                hq.memcpy(hx.data(), dfl.d_df_b + (size_t)r * K2, (size_t)K2 * 4).wait();
                hq.memcpy(lg.data(), d_logits + (size_t)r * nv, (size_t)nv * 4).wait();
                for (int n = 0; n < nv; n++) {
                    dequantize_row(head.type, (const char *)head.data + (size_t)n * rb, rw.data(), K2);
                    float a = 0;
                    for (int k = 0; k < K2; k++) {
                        a += rw[k] * hx[k];
                    }
                    acc[n] = a;
                }
                int bi[5] = {0, 0, 0, 0, 0};
                float bv[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
                for (int n = 0; n < nv; n++) {
                    for (int c = 0; c < 5; c++) {
                        if (lg[n] > bv[c]) {
                            for (int c2 = 4; c2 > c; c2--) {
                                bv[c2] = bv[c2 - 1];
                                bi[c2] = bi[c2 - 1];
                            }
                            bv[c] = lg[n];
                            bi[c] = n;
                            break;
                        }
                    }
                }
                fprintf(stderr, "[dflash] headchk row %d dev:", r);
                for (int c = 0; c < 5; c++) {
                    fprintf(stderr, " %d(%.3f)", bi[c], bv[c]);
                }
                fprintf(stderr, "   exact:");
                for (int c = 0; c < 5; c++) {
                    fprintf(stderr, " %d(%.3f)", bi[c], acc[bi[c]]);
                }
                fprintf(stderr, "\n");
            }
        }
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// One draft step: the block forward plus the lattice walk.  Returns the block's
// tokens (cand[0] = the anchor).
std::vector<int> engine::df_draft(int pos0, int anchor_tok, int n_max) {
    const dflash_hp & hp = dfl.dfm_->hp;
    sycl::queue & dq = dev_queue(dfl.df_dev_);
    const int M = n_max + 1;
    std::vector<int32_t> toks((size_t)M);
    toks[0] = anchor_tok;
    for (int i = 1; i < M; i++) {
        toks[(size_t)i] = hp.mask_id;
    }
    static const bool dtt = si::env::flag("PF_DFLASH_DTTAIL");
    auto dtt_now = [] { return std::chrono::high_resolution_clock::now(); };
    const auto dtt0 = dtt_now();
    df_block(pos0, toks.data(), M);
    const auto dtt1 = dtt_now();
    std::vector<int> cand((size_t)M);
    cand[0] = anchor_tok;
    if (!hp.is_dflash2) {
        // DFlash1: every position's own argmax, one device argmax per row
        dq.wait();
        std::vector<float> row((size_t)m.hp.n_vocab);
        for (int i = 1; i < M; i++) {
            dq.memcpy(row.data(), d_logits + (size_t)i * m.hp.n_vocab, (size_t)m.hp.n_vocab * 4).wait();
            cand[(size_t)i] = argmax_f(row.data(), m.hp.n_vocab);
        }
        return cand;
    }
    const int K = hp.sel_top_k;
    const int row = K + K * K;
    const auto tt_d0 = tt_now();
    if (tt_dbg) { tt_e[3] = dq.ext_oneapi_submit_barrier(); }
    dq.memcpy(dfl.h_df_lattice.data(), dfl.d_df_lattice, (size_t)M * row * 4).wait();
    if (tt_dbg) {
        dq.wait();
        auto de = [&](const sycl::event & lo, const sycl::event & hi) {
            return (df_ts_ms(hi, false) - df_ts_ms(lo, true));
        };
        fprintf(stderr,
                "[dflash-tt] DEV topk=%.2f selhidden=%.2f sel=%.2f (total %.2f) | host_block=%.2f "
                "host_tail=%.2f ms\n",
                de(tt_e[0], tt_e[1]), de(tt_e[1], tt_e[2]), de(tt_e[2], tt_e[3]), de(tt_e[0], tt_e[3]),
                ms_t(tt_d0, tt_now()) - (de(tt_e[3], tt_e[3])), 0.0);
    }
    if (df_dbg()) {
        // localize a wrong draft: the head's top-k per position and the
        // selector's first lattice row
        std::vector<int32_t> kid((size_t)M * K);
        std::vector<float> kval((size_t)M * K);
        std::vector<float> hb((size_t)hp.n_embd * M);
        dq.memcpy(kid.data(), dfl.d_df_ids, kid.size() * 4).wait();
        dq.memcpy(kval.data(), dfl.d_df_vals, kval.size() * 4).wait();
        dq.memcpy(hb.data(), dfl.d_df_b, hb.size() * 4).wait();
        if (df_dbg()) {
            fprintf(stderr, "[dflash] topk raw (M=%d K=%d):", M, K);
            for (int i = 0; i < M; i++) {
                fprintf(stderr, "  r%d ids:", i);
                for (int c = 0; c < K; c++) {
                    fprintf(stderr, " %d", (int)kid[(size_t)i * K + c]);
                }
            }
            fprintf(stderr, "\n");
        }
        for (int i = 0; i < M; i++) {
            double n2 = 0;
            for (int j = 0; j < hp.n_embd; j++) {
                n2 += (double)hb[(size_t)i * hp.n_embd + j] * hb[(size_t)i * hp.n_embd + j];
            }
            if (i == 1) {
                // host-side top-5 of the draft head's logits: if this is sane
                // while df_topk_launch's output repeats one id, the top-k kernel
                // is at fault, not the logits
                std::vector<float> lg((size_t)m.hp.n_vocab);
                dev_queue(0).memcpy(lg.data(), d_logits + (size_t)1 * m.hp.n_vocab, lg.size() * 4).wait();
                int bi[5] = {0, 0, 0, 0, 0};
                float bv[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
                for (int j = 0; j < m.hp.n_vocab; j++) {
                    const float v = lg[(size_t)j];
                    for (int c = 0; c < 5; c++) {
                        if (v > bv[c]) {
                            for (int c2 = 4; c2 > c; c2--) {
                                bv[c2] = bv[c2 - 1];
                                bi[c2] = bi[c2 - 1];
                            }
                            bv[c] = v;
                            bi[c] = j;
                            break;
                        }
                    }
                }
                fprintf(stderr, "[dflash] host logits row1 top:");
                for (int c = 0; c < 5; c++) {
                    fprintf(stderr, " %d(%.4f)", bi[c], bv[c]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "[dflash] row %d |h|=%.4f top:", i, std::sqrt(n2));
            for (int c = 0; c < 5 && c < K; c++) {
                fprintf(stderr, " %d(%.3f)", (int)kid[(size_t)i * K + c], kval[(size_t)i * K + c]);
            }
            fprintf(stderr, "\n");
        }
        for (int pp = 1; pp < M; pp++) {
            const float * r = dfl.h_df_lattice.data() + (size_t)pp * row;
            fprintf(stderr, "[dflash-lat] pos %d ids:", pp);
            for (int c = 0; c < K; c++) {
                fprintf(stderr, " %.0f", r[c]);
            }
            fprintf(stderr, "  scores:");
            for (int c = 0; c < 4; c++) {
                fprintf(stderr, " %.4f", r[K + c]);
            }
            fprintf(stderr, "\n");
        }
    }
    const auto dtt2 = dtt_now();
    int pred = 0;
    for (int i = 1; i < M; i++) {
        const float * r = dfl.h_df_lattice.data() + (size_t)i * row;
        const float * sc = r + K + (size_t)pred * K;
        pred = argmax_f(sc, K);
        cand[(size_t)i] = (int)r[pred];
    }
    if (dtt) {
        auto d = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        fprintf(stderr, "[dflash-dt] block=%.2f tail_copy=%.2f walk=%.3f ms\n", d(dtt0, dtt1), d(dtt1, dtt2),
                d(dtt2, dtt_now()));
    }
    if (df_dbg()) {
        fprintf(stderr, "[dflash] block pos=%d anchor=%d cand:", pos0, anchor_tok);
        for (int i = 1; i < M; i++) {
            fprintf(stderr, " %d", cand[(size_t)i]);
        }
        fprintf(stderr, "\n");
    }
    return cand;
}

// ---------------------------------------------------------------------------
// Speculative generation with a DFlash/DFlash2 drafter (greedy).
std::vector<int> engine::generate_dflash(const std::vector<int> & prompt, const gen_params & gp,
                                         const std::function<bool(int)> & cb, std::vector<float> * first_logits) {
    if (!dfl.dflash_on_) {
        return generate(prompt, gp, cb, first_logits);
    }
    if (!gp.speculative_greedy()) {
        throw std::invalid_argument("DFlash2 requires greedy sampling without logit penalties or bias");
    }
    reset_single();
    const hparams & hp = m.hp;
    sampler_state ss;
    ss.seed(gp.seed ? gp.seed : std::random_device{}());
    std::vector<int> out;
    const int nprompt = (int)prompt.size();
    if (nprompt == 0) {
        return out;
    }
    // The drafter's K/V ring is seeded from the features the forward captures, so
    // a prefix-cache hit (which restores the target's KV but not the ring) would
    // leave the drafter blind: no cache admission here.
    std::vector<int> blocks;
    const int need = (nprompt + kBlockSize - 1) / kBlockSize + (dfl.df_k_ + 2 + kBlockSize - 1) / kBlockSize + 1;
    for (int i = 0; i < need; i++) {
        const int b = alloc_block();
        if (b < 0) {
            throw std::runtime_error("out of KV blocks");
        }
        blocks.push_back(b);
    }
    set_table(0, blocks);

    DFDBG("generate_dflash: prompt %d tokens, block %d, n_max %d\n", nprompt, dfl.df_block_, dfl.df_k_);
    // ---- prefill: every batch's captured features are injected right after the
    // forward that produced them ----
    for (int pos = 0; pos < nprompt;) {
        const int rem = nprompt - pos;
        const int fit = cpu_mode ? 0 : batched_prefill_fit(rem);
        const int nb = (fit >= 1 && fit <= rem) ? fit : std::min(kMaxT, rem);
        DFDBG("prefill pos=%d n=%d batched=%d\n", pos, nb, (int)(fit >= 1 && fit <= rem));
        if (fit >= 1 && fit <= rem) {
            prefill_batch(prompt, pos, nb, 0, pos);
        } else {
            prefill_chunk(prompt, pos, nb, 0);
        }
        prefill_flush();
        // the forward above captured this batch's features into dfl.d_df_feat
        sync_all();
        df_inject(nb, pos, /*feat_row=*/0);
        pos += nb;
    }
    std::vector<float> logits = run_head();
    if (first_logits) {
        *first_logits = logits;
    }
    int pos = nprompt;
    int tok = sample_token(logits.data(), hp.n_vocab, gp, out, ss);
    out.push_back(tok);
    if ((!gp.ignore_eos && is_eos(tok)) || !cb(tok)) {
        for (int b : blocks) {
            free_block(b);
        }
        return out;
    }
    // ---- speculative loop ----
    static const bool mt = si::env::flag("PF_MTP_TIME") || si::env::flag("PF_DFLASH_TIME");
    double t_draft = 0, t_verify = 0, t_rb = 0, t_emit = 0, t_inj = 0, t_cyc = 0;
    long t_acc = 0;
    int t_cycles = 0, t_tok = 0;
    auto now_t = [] { return std::chrono::high_resolution_clock::now(); };
    auto ms_t = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::vector<int> cand;
    static const bool steps_dbg = si::env::flag("PF_DFLASH_STEPS");
    long steps_cyc = 0, steps_reach[40] = {0};
    while ((int)out.size() < gp.max_tokens && pos < max_seq - 1) {
        const auto tc0 = now_t();
        // ---- draft ----
        cand = df_draft(pos, tok, dfl.df_k_);
        const auto tc1 = now_t();
        t_draft += ms_t(tc0, tc1);
        // ---- verify ----
        const int n_ver = (int)cand.size();
        mtp_verify(cand, n_ver, 0, pos);
        sync_all();
        const auto tc2 = now_t();
        t_verify += ms_t(tc1, tc2);
        if (si::env::flag("PF_DFLASH_VFCHK")) {
            fprintf(stderr, "[dflash-vf] pos=%d rows=%d vf_dec_ok=%d vf_rows=%d graphs=%d\n", pos,
                    n_ver + 1, (int)vf_dec_ok, vf_dec_rows,
                    (int)vf_dec_.size());
        }
        // ---- accept (greedy: the device argmax is exactly the greedy sample) ----
        int32_t * d_argmax = mtp.d_argmax_buf_;
        if (!d_argmax) {
            mtp.d_argmax_buf_ = sycl::malloc_device<int32_t>(kMaxB, dev_queue(0));
            d_argmax = mtp.d_argmax_buf_;
        }
        mtp_argmax_launch(dev_queue(0), d_logits, hp.n_vocab, d_argmax, nullptr, n_ver);
        dev_queue(0).memcpy(mtp.h_argmax, d_argmax, (size_t)n_ver * 4).wait();
        int j = 0;
        while (j < dfl.df_k_ && (int)mtp.h_argmax[j] == cand[(size_t)j + 1]) {
            j++;
        }
        if (df_dbg()) {
            fprintf(stderr, "[dflash] verify pos=%d anchor=%d cand:", pos, cand[0]);
            for (int i = 1; i < (int)cand.size(); i++) {
                fprintf(stderr, " %d", cand[(size_t)i]);
            }
            fprintf(stderr, "  target:");
            for (int i = 0; i < (int)cand.size(); i++) {
                fprintf(stderr, " %d", (int)mtp.h_argmax[i]);
            }
            fprintf(stderr, "  j=%d\n", j);
        }
        if (steps_dbg) {
            steps_cyc++;
            for (int s2 = 0; s2 <= j; s2++) {
                steps_reach[s2]++;
            }
            if (steps_cyc % 32 == 0) {
                fprintf(stderr, "[dflash] steps: cycles=%ld reach:", steps_cyc);
                for (int s2 = 0; s2 < dfl.df_k_; s2++) {
                    fprintf(stderr, " %ld%%", 100L * steps_reach[s2 + 1] / steps_cyc);
                }
                fprintf(stderr, "  (mean accepted %.2f)\n", (double)j / steps_cyc);
            }
        }
        for (int i = 1; i <= j; i++) {
            if ((int)out.size() >= gp.max_tokens) {
                for (int b : blocks) {
                    free_block(b);
                }
                return out;
            }
            out.push_back(cand[(size_t)i]);
            if ((!gp.ignore_eos && is_eos(cand[(size_t)i])) || !cb(cand[(size_t)i])) {
                for (int b : blocks) {
                    free_block(b);
                }
                return out;
            }
        }
        const int bonus = (int)mtp.h_argmax[j];
        if ((int)out.size() >= gp.max_tokens) {
            break;
        }
        tok = bonus;
        out.push_back(tok);
        if ((!gp.ignore_eos && is_eos(tok)) || !cb(tok)) {
            break;
        }
        const auto tc3 = now_t();
        t_emit += ms_t(tc2, tc3);
        // ---- commit + rollback: the recurrent state back to the last accepted
        // row, then the drafter's ring: inject the features the verify captured
        // for the newly committed rows 1..j+1 (row j+1 is the bonus token, which
        // is the next cycle's anchor) ----
        mtp_rollback(j);
        const auto tc4 = now_t();
        t_rb += ms_t(tc3, tc4);
        if (si::env::flag("PF_DFLASH_INJTRACE")) {
            // what the ring will hold after this cycle: the injected cells' V rms
            // next to the block's own, so a mis-placed injection shows up as a
            // discontinuity at the boundary
            // the ring may be f16 (dfl.df_kv_bytes_ == 2); reading it as f32 reports
            // 1e12 rms garbage and makes every coverage number meaningless
            const size_t rn = (size_t)dfl.df_ring_ * hp.n_head_kv * hp.head_dim;
            std::vector<float> vr(rn);
            std::vector<uint16_t> vh(rn);
            if (dfl.df_kv_bytes_ == 4) {
                dev_queue(dfl.df_dev_).memcpy(vr.data(), dfl.d_df_vring, rn * 4).wait();
            } else {
                dev_queue(dfl.df_dev_).memcpy(vh.data(), dfl.d_df_vring, rn * 2).wait();
                for (size_t z = 0; z < rn; z++) {
                    vr[z] = f16_bits(vh[z]);
                }
            }
            const int lo = pos > 2 ? pos - 2 : 0;
            fprintf(stderr, "[dflash] injtrace pos=%d j=%d anchor=%d bonus=%d ringV:", pos, j, cand[0], bonus);
            for (int p2 = lo; p2 <= pos + dfl.df_k_ + 1; p2++) {
                double ss = 0;
                const size_t cell = (size_t)(p2 % dfl.df_ring_) * hp.n_head_kv;
                for (int j2 = 0; j2 < hp.n_head_kv * hp.head_dim; j2++) {
                    ss += (double)vr[cell * hp.head_dim + j2] * vr[cell * hp.head_dim + j2];
                }
                fprintf(stderr, " %d:%.2f", p2, std::sqrt(ss / (hp.n_head_kv * hp.head_dim)));
            }
            fprintf(stderr, "\n");
            // Ring-wide coverage.  A mask row attends over every committed position,
            // so a shortfall here is a pure magnitude deficit on exactly the rows that
            // diverge from the reference, and it grows with the context - which is
            // what the acceptance-vs-length falloff looked like.
            {
                std::vector<double> rv((size_t)dfl.df_ring_);
                for (int p2 = 0; p2 < dfl.df_ring_; p2++) {
                    double ss = 0;
                    const size_t cell = (size_t)p2 * hp.n_head_kv;
                    for (int j2 = 0; j2 < hp.n_head_kv * hp.head_dim; j2++) {
                        ss += (double)vr[cell * hp.head_dim + j2] * vr[cell * hp.head_dim + j2];
                    }
                    rv[p2] = std::sqrt(ss / (hp.n_head_kv * hp.head_dim));
                }
                int nz = 0;
                double mn = 1e30, mx = 0, sum = 0;
                for (int p2 = 0; p2 < pos + 1 && p2 < dfl.df_ring_; p2++) {
                    if (rv[p2] > 1e-6) {
                        nz++;
                    }
                    mn = std::min(mn, rv[p2]);
                    mx = std::max(mx, rv[p2]);
                    sum += rv[p2];
                }
                fprintf(stderr, "[dflash] ringcov: positions 0..%d nonzero=%d/%d rms min=%.4g mean=%.4g max=%.4g\n",
                        pos, nz, std::min(pos + 1, dfl.df_ring_), mn, sum / std::max(1, std::min(pos + 1, dfl.df_ring_)), mx);
            }
        }
        if (j + 1 > 0) {
            df_inject(j + 1, pos + 1, /*feat_row=*/1);
        }
        const auto tc5 = now_t();
        t_inj += ms_t(tc4, tc5);
        pos += j + 1;
        const bool have_headroom = (int)out.size() >= gp.max_tokens || ensure_block_headroom(blocks, pos, dfl.df_k_ + 1);
        t_cyc += ms_t(tc0, tc5);
        t_cycles++;
        t_tok += j + 1;
        t_acc += j;
        if (mt && (t_cycles % 4 == 0)) {
            fprintf(stderr,
                    "[dflash] time: cycles=%d tok=%d acc=%.2f | draft=%.1f verify=%.1f rb=%.1f inject=%.1f "
                    "emit=%.1f cycle=%.1f ms | %.2f ms/token\n",
                    t_cycles, t_tok, (double)t_acc / t_cycles, t_draft / t_cycles, t_verify / t_cycles,
                    t_rb / t_cycles, t_inj / t_cycles, t_emit / t_cycles, t_cyc / t_cycles,
                    (t_draft + t_verify + t_rb + t_inj + t_emit) / t_tok);
        }
        if ((int)out.size() >= gp.max_tokens || !have_headroom) {
            break; // never draft against a block table without KV headroom
        }
    }
    for (int b : blocks) {
        free_block(b);
    }
    return out;
}

} // namespace si
