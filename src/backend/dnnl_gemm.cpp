#include "dnnl_gemm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <mutex>
#include <utility>
#include <vector>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include "kernels.h"
#include "device/device_profile.h"
#include "quant.h"
#include "w4.h"
#include "common/env.h"

namespace si {

namespace {

using namespace dnnl;

inline float silu_act(float x) {
    return x / (1.0f + sycl::exp(-x));
}

// M values of the recorded chunk-batched prefill variants (engine::build_graphs
// records nch = 2/4/8/16 chunk rows of kMaxT tokens).  Mode 1 uses kMaxT and
// mode 2 can use any multiple of kMaxT up to kMaxB*kMaxT, so every one of them
// gets its primitive created in add_weight() and executed once in warmup().

constexpr int kActW = 256;   // activation quantizer workgroup size
constexpr int kActMaxK = 32768; // max K the row quantizer handles (27B n_ff=17408)

// Quantize one call's activations: row-major [M][K] int8 with one symmetric
// scale per row (scale = max|row|/127, the same scheme as the dp4a xq_launch
// but per row instead of per 32 values - oneDNN's int8 matmul only takes one
// scale per dimension).  The row is read once (values stay in registers) and
// the row max is reduced across the workgroup.
void act_quant_launch(sycl::queue & q, const float * x, const float * up, int x_stride, int up_stride, int8_t * xq,
                      float * scales, int32_t * asum, int M, int K) {
    q.parallel_for(sycl::nd_range<1>((size_t)M * kActW, kActW), [=](sycl::nd_item<1> it) {
        const int m = (int)it.get_group(0);
        const int lane = (int)it.get_local_id(0);
        const float * xr = x + (size_t)m * x_stride;
        const float * ur = up ? up + (size_t)m * up_stride : nullptr;
        // two passes (max, then quantize): no per-thread value array, so K can
        // be arbitrarily large without register spilling.  The second pass also
        // totals the signed int8 values: the decode dp4a GEMV biases weights to
        // unsigned and corrects with 128*sum(a).
        float mx = 0.f;
        for (int k = lane; k < K; k += kActW) {
            float val = xr[k];
            if (ur) {
                val = silu_act(val) * ur[k];
            }
            mx = sycl::fmax(mx, sycl::fabs(val));
        }
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        const float scale = mx > 0.f ? mx / 127.0f : 1.0f;
        if (lane == 0) {
            scales[m] = scale;
        }
        const float inv = 1.0f / scale;
        int8_t * orow = xq + (size_t)m * K;
        int s = 0;
        for (int k = lane; k < K; k += kActW) {
            float val = xr[k];
            if (ur) {
                val = silu_act(val) * ur[k];
            }
            int qi = (int)sycl::round(val * inv);
            qi = sycl::max(-127, sycl::min(127, qi));
            orow[k] = (int8_t)qi;
            s += qi;
        }
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<int>());
        if (lane == 0 && asum) {
            asum[m] = s;
        }
    });
}

// Per-32-group activation quantization for the u4 path: each group of 32 values
// gets its own symmetric scale, matching oneDNN's grouped SRC scales (mask on
// K, groups {1,32}).  The per-row version above is still produced because one
// call can mix u4 and int8 segments, and int8's oneDNN matmul only takes one
// scale per row.  This is what removes the dominant quantization error: with a
// per-row scale the activation error (~0.9%) swamped the 4-bit weight error.
void act_quant_grp_launch(sycl::queue & q, const float * x, const float * up, int x_stride, int up_stride, int8_t * xq,
                          uint16_t * sasc, float * xs, uint16_t * xsr, int8_t * axe, int8_t * axo, bool do_split,
                          int M, int K, int64_t * sumq, int64_t * sums, int slot) {
    const int ng = K / kW4Group;
    const size_t kh = (size_t)K / 2;
    // One 32-lane workgroup per (row, group): each lane owns exactly one value,
    // so the row max is a sub-group reduction and the value never leaves a
    // register between the max pass and the store.  The previous form was one
    // *thread* per group (M*ng threads, i.e. 544 for K=17408) which was latency
    // bound: 12.6 us/call x 320 calls = 4 ms of a 27B decode step.
    //
    // The same launch also produces the two derived views the GEMVs need - the
    // group sums xs (one sub-group reduction of the just-computed quantized
    // values) and, when do_split, the even/odd k deinterleave of xq - so a call
    // costs one kernel instead of three (measured ~5 ms/token of launch
    // overhead in a 27B multi-device decode).
    q.parallel_for(sycl::nd_range<1>((size_t)M * ng * kW4Group, kW4Group), [=](sycl::nd_item<1> it) {
        const int gid = (int)it.get_group(0);
        const int m = gid / ng;
        const int g = gid % ng;
        const int lane = (int)it.get_local_id(0);
        const float * xr = x + (size_t)m * x_stride + (size_t)g * kW4Group;
        float v = xr[lane];
        if (up) {
            v = silu_act(v) * up[(size_t)m * up_stride + (size_t)g * kW4Group + lane];
        }
        const float mx = sycl::reduce_over_group(it.get_sub_group(), sycl::fabs(v), sycl::maximum<float>());
        const float sc = mx > 0.f ? mx / 127.0f : 1.0f;
        const float inv = 1.0f / sc;
        const int qi = (int)sycl::round(v * inv);
        const int8_t q8 = (int8_t)sycl::max(-127, sycl::min(127, qi));
        xq[(size_t)m * K + (size_t)g * kW4Group + lane] = q8;
        if (do_split) {
            // nibble-pair k with k+1 lives in one byte, so the u4 GEMV wants the
            // even/odd k planes: axe[j] = xq[2j], axo[j] = xq[2j+1]
            const size_t half = (size_t)m * kh + (size_t)g * (kW4Group / 2) + (size_t)(lane >> 1);
            if ((lane & 1) == 0) {
                axe[half] = q8;
            } else {
                axo[half] = q8;
            }
        }
        if (lane == 0) {
            sycl::half h(sc);
            uint16_t bits;
            __builtin_memcpy((void *)&bits, &h, sizeof(h));
            sasc[(size_t)m * ng + g] = bits;
            // PF_ACTQ_SUM: an order-independent INTEGER checksum of what this
            // kernel just wrote, so two processes can be compared WITHOUT a host
            // read-back inside the forward - a read-back is precisely what broke
            // the per-layer fingerprint oracle for this investigation, so the
            // checksum is accumulated on the device and read exactly once at
            // teardown.  `plus<int>` over the sub-group and a 64-bit atomic add are
            // both exact and order-independent, so the checksum is itself
            // deterministic given the data: it can only move if the DATA moves.
            // That is the whole point - it separates "oneDNN was handed different
            // bytes" from "oneDNN computed differently from the same bytes", which
            // oneDNN_VERBOSE cannot answer because verbose logs descriptors, not
            // contents.
            if (sumq) {
                const int gsum = sycl::reduce_over_group(it.get_sub_group(), (int)q8, sycl::plus<int>());
                sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                    aq(sumq[slot]);
                aq.fetch_add((int64_t)gsum);
                sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                    as(sums[slot]);
                as.fetch_add((int64_t)bits);
            }
        }
        if (xs || xsr) {
            // a sub-group collective, so every lane must execute it (a divergent
            // call is UB and produced garbage xs - the u4 correction term)
            const int s = sycl::reduce_over_group(it.get_sub_group(), (int)q8, sycl::plus<int>());
            if (lane == 0) {
                if (xs) {
                    xs[(size_t)m * ng + g] = (float)s;
                }
                if (xsr) {
                    // PF_W4_CORR_GEMM: the correction matmul's left operand.  A
                    // group's dequantized activation sum is sum_k x[m][k] =
                    // sum(q8) * scale = xs * sa, so this is exactly the product
                    // w4_epilogue_launch forms per (m,g) - using the *stored*
                    // f16 scale, which is what that kernel reads too.  Stored as
                    // f16 (the operand type), an ulp of a group of <= 127*32.
                    const float scf = (float)sycl::half(sc);
                    xsr[(size_t)m * ng + g] = sycl::bit_cast<uint16_t>(sycl::half((float)s * scf));
                }
            }
        }
    });
}

static inline float w4_f16_bits(uint16_t v) {
    sycl::half h;
    __builtin_memcpy((void *)&h, &v, sizeof(h));
    return (float)h;
}

// PF_W4_CORR_GEMM (default 1): run the u4 offset correction
//   corr[m][n] = sum_g off[g][n] * (xs[m][g] * sa[m][g])
// as a second oneDNN matmul  [M, K/32] f16 x [K/32, N] f16 --sum--> [M, N] f32
// instead of w4_epilogue_launch's scalar loop over g.  That loop was 73 % of
// the u4 prefill path (419 ms/chunk against 108 for the raw u4 matmul) because
// the correction is a rank-(K/32) GEMM being run one multiply-add at a time;
// the matmul version measures 164 ms/chunk on the same shapes.  The `off` plane
// is already [ng][N] f16 and `src[m][g]` is produced by the same
// act_quant_grp_launch that produces xs/asa, so this costs no extra launch and
// no extra weight traffic.  M == 1 never reaches here (decode is the u4 GEMV),
// so the decode path is untouched; any shape whose correction primitive does
// not validate keeps the scalar epilogue.
static bool w4_corr_gemm_env() {
    static const bool on = [] {
        const char * e = si::env::str("PF_W4_CORR_GEMM");
        return !(e && atoi(e) == 0);
    }();
    return on;
}

// out[m][n] = alpha*sx[m]*(acc4[m][n] + sum_g off[g][n]*xs[m][g]) + residual
// acc4 is already the step-scaled sum (oneDNN applied the grouped f16 scales).
// Register-tiled TM x TN per thread over a 64 x 64 workgroup tile: the offset
// plane is then read only M/64 times (a naive per-(m,n) loop over g rereads it
// once per output row and was ~45x too slow), and each xs value is reused TN
// times / each offset value TM times.
void w4_epilogue_launch(sycl::queue & q, const float * acc4, const float * xs, const uint16_t * off,
                        const uint16_t * sa, float * out, int out_stride, const float * residual, float alpha, int M,
                        int N, int NG, int OF) {
    constexpr int TM = 4, TN = 4, GX = 16, GY = 16, TX = GX * GY;
    const int mb = GY * TM, nb = GX * TN;
    const int mt = (M + mb - 1) / mb;
    const int nt = (N + nb - 1) / nb;
    q.parallel_for(sycl::nd_range<1>((size_t)mt * nt * TX, TX), [=](sycl::nd_item<1> it) {
        const int grp = (int)it.get_group(0);
        const int m0 = (grp % mt) * mb;
        const int n0 = (grp / mt) * nb;
        const int lid = (int)it.get_local_id(0);
        const int m = m0 + (lid / GX) * TM;
        const int n = n0 + (lid % GX) * TN;
        float c[TM][TN];
        for (int a = 0; a < TM; a++) {
            for (int b = 0; b < TN; b++) {
                c[a][b] = 0.f;
            }
        }
        if (off) {
            for (int g = 0; g < NG; g++) {
                // the group's activation scale belongs to this correction too;
                // the oneDNN term already has it applied via the SRC scales
                float xv[TM], wv[TN];
                for (int a = 0; a < TM; a++) {
                    xv[a] = (m + a < M)
                                ? xs[(size_t)(m + a) * NG + g] * w4_f16_bits(sa[(size_t)(m + a) * NG + g])
                                : 0.f;
                }
                for (int b = 0; b < TN; b++) {
                    wv[b] = (n + b < N) ? w4_f16_bits(off[(size_t)(g / OF) * N + n + b]) : 0.f;
                }
                for (int a = 0; a < TM; a++) {
                    for (int b = 0; b < TN; b++) {
                        c[a][b] += xv[a] * wv[b];
                    }
                }
            }
        }
        for (int a = 0; a < TM; a++) {
            if (m + a >= M) {
                continue;
            }
            for (int b = 0; b < TN; b++) {
                if (n + b >= N) {
                    continue;
                }
                const size_t o = (size_t)(m + a) * out_stride + n + b;
                float v = alpha * (acc4[(size_t)(m + a) * N + n + b] + c[a][b]);
                if (residual) {
                    v += residual[o];
                }
                out[o] = v;
            }
        }
    });
}

// PF_W4_GROUP > 32: rebin the correction source from the quantizer's per-32
// groups to the weight store's group, so the correction matmul's A operand has
// the same K as the per-G off plane.  F consecutive f16 entries are summed in
// f32 and written back as f16; with F == 1 the plan is skipped entirely.
static void w4_corr_group_launch(sycl::queue & q, const uint16_t * src32, uint16_t * dstG, int F, int M, int ng32) {
    const int ngG = ng32 / F;
    q.parallel_for(sycl::range<1>((size_t)M * (size_t)ngG), [=](sycl::id<1> i) {
        const int m = (int)(i / (size_t)ngG);
        const int g = (int)(i % (size_t)ngG);
        float s = 0.f;
        for (int j = 0; j < F; j++) {
            s += w4_f16_bits(src32[(size_t)m * ng32 + (size_t)g * F + j]);
        }
        sycl::half h(s);
        uint16_t bits;
        __builtin_memcpy((void *)&bits, &h, sizeof(h));
        dstG[(size_t)m * ngG + g] = bits;
    });
}

// out[m][n] = alpha*acc[m][n] + residual, where acc is the f32 accumulator the
// matmul produced with the grouped weight *and* activation scales already
// applied inside - so there is no per-row factor left for the epilogue.
void epilogue_f32_launch(sycl::queue & q, const float * acc, float * out, int out_stride, const float * residual,
                         float alpha, int M, int N) {
    q.parallel_for(sycl::range<1>((size_t)M * N), [=](sycl::id<1> i) {
        const int m = (int)(i / (size_t)N);
        const int n = (int)(i % (size_t)N);
        const size_t o = (size_t)m * out_stride + n;
        float v = alpha * acc[i];
        if (residual) {
            v += residual[o];
        }
        out[o] = v;
    });
}

// out[m][n] = alpha * sx[m]*sw[n]*acc[m][n] + (residual ? residual : 0)
template <int V>
void epilogue_vec_launch(sycl::queue & q, const int32_t * acc, const float * sx, const float * sw, float * out,
                         int out_stride, const float * residual, float alpha, int M, int N) {
    using Vf = sycl::vec<float, V>;
    const int nv = N / V;
    q.parallel_for(sycl::range<1>((size_t)M * nv), [=](sycl::id<1> i) {
        const int m = (int)(i / nv);
        const int n0 = (int)(i % nv) * V;
        const float a = alpha * sx[m];
        const int32_t * pa = acc + (size_t)m * N + n0;
        const size_t o = (size_t)m * out_stride + n0;
        Vf val;
        if (residual) {
            const Vf r = *reinterpret_cast<const Vf *>(residual + o);
            for (int j = 0; j < V; j++) {
                val[j] = a * sw[n0 + j] * (float)pa[j] + r[j];
            }
        } else {
            for (int j = 0; j < V; j++) {
                val[j] = a * sw[n0 + j] * (float)pa[j];
            }
        }
        *reinterpret_cast<Vf *>(out + o) = val;
    });
}

} // namespace

void act_quant_grp_run(sycl::queue & q, const float * x, int x_stride, int M, int K, int8_t * xq, uint16_t * asa,
                       float * xs) {
    // The same kernel quantize() runs; only its contiguous outputs are needed
    // (the split planes are for the dense u4 GEMV, not the expert bank).
    act_quant_grp_launch(q, x, nullptr, x_stride, x_stride, xq, asa, xs, nullptr, nullptr, nullptr, /*do_split=*/false,
                         M, K, nullptr, nullptr, 0);
}

static bool g_dnnl_capturing = false;
void dnnl_set_capturing(bool on) {
    g_dnnl_capturing = on;
}
void dnnl_capture_guard() {
    if (g_dnnl_capturing) {
        throw std::runtime_error("oneDNN primitive reached during SYCL graph capture");
    }
}

// Owning pointer for a *per-call* device temporary.  Long-lived allocations go
// through impl::dev_alloc / impl::owned_ instead; this exists for the few
// malloc/free pairs that live inside one function and must still be released
// when that function unwinds (a warmup execute can throw from
// dnnl_capture_guard).  sycl::free needs the allocating queue, hence the
// by-reference member instead of a function-pointer deleter.
template <typename T> struct qfree {
    T * p = nullptr;
    sycl::queue * q = nullptr;
    qfree() = default;
    qfree(T * p_, sycl::queue & q_) : p(p_), q(&q_) {}
    qfree(const qfree &) = delete;
    qfree & operator=(const qfree &) = delete;
    ~qfree() {
        if (p) {
            sycl::free(p, *q);
        }
    }
    T * get() const {
        return p;
    }
    explicit operator bool() const {
        return p != nullptr;
    }
};

bool dnnl_gemm_enabled(const sycl::queue & q) {
    // default ON (the mode-2 int8 GEMMs run on oneDNN); PF_GEMM_DNNL=0 forces
    // the dp4a chunk-batched path, e.g. for bit-exact dp4a validation.  The env
    // is device-independent (read once); the default is not, so it is read per
    // call from the queue's own card.
    static const int forced = [] {
        const char * e = si::env::str("PF_GEMM_DNNL");
        return e ? atoi(e) : -1;
    }();
    if (forced == 0) {
        return false;
    }
    return forced != 0 || si::dev::for_queue(q).wt.gemm_dnnl;
}

// PF_DNNL_QWAIT: wait on the SYCL QUEUE after EVERY oneDNN execute in this file.
//
// Every primitive here runs on an interop stream (`p->st` / `p->st_a`) while every
// consumer of its output - the next SYCL kernel, the epilogue, the quantizer - is
// submitted to `p->q`.  PF_ATTN_QWAIT only ever covered `attn_qk`/`attn_pv`, so the
// dense mode-2 GEMM -> next-SYCL-kernel edge was never ordered by anything the
// engine controls.  That edge is what the prefill's nondeterminism hangs on: the
// symptom is a consistent 1-5% spread with a STABLE argmax, which is what a kernel
// reading the PREVIOUS layer's value looks like (consecutive layers differ by a few
// percent, so a stale read is a few percent rather than garbage).  It also explains
// why every per-layer diagnostic seemed to stabilise the run - PF_DUMP_LAYERS
// inserts a q.wait() after each layer, which inserts exactly the missing edge.
//
// The oneDNN host-threading hypothesis is refuted by measurement: forcing
// ONEDNN_MAX_CPU_ISA=0 makes it WORSE (5.38% vs 3.04% default).  Nor is it
// attn_xmx-specific - with a non-perturbing oracle PF_ATTN_XMX=0 still spreads
// 0.37%, so "only PF_ATTN_XMX=0 is reproducible" was an artifact of the
// per-layer fingerprint oracle, which perturbs the very schedule it measures.
//
// DIAGNOSTIC ONLY, default off.  If this collapses the spread, the fix is to submit
// the primitive on the queue's own ordering domain (or ext_oneapi_submit_barrier
// between the stream and the queue), not to pay a host wait per GEMM.
//
// PF_ATTN_WAIT=1 calls `ast.wait()` on the interop stream the primitive is
// submitted to.  Every SYCL kernel around the matmul - the Q quantize, the K/V
// gather, the softmax - is submitted to the SYCL queue, not to that stream, so a
// wait on the stream is only ordered against work submitted to the same stream.  If
// the two are distinct ordering domains then PF_ATTN_WAIT cannot make the matmul
// see the gather, and the symptom is exactly the one this file already records for
// the old `st_a`: results that change when a `q.wait()` is inserted, i.e. when the
// instrumentation adds an ordering edge.  This waits on `p->q` itself, which orders
// against the SYCL submissions, and is the honest test of whether there is an
// ordering problem at all.  DIAGNOSTIC ONLY, default off: if this does turn out to
// be the fix, the real fix is to submit the primitive on the queue's own ordering
// domain rather than to pay a host wait per matmul.
// PF_DNNL_ACC_SUM: per-call device-side checksum of the int32 accumulator oneDNN
// just wrote, i.e. the GEMM's own output BEFORE the epilogue scales it.  Together
// with PF_ACTQ_SUM (the GEMM's input) and PF_KVSTORE_SUM's `pre` (the fp32 qkv
// output) this closes the last gap: input identical AND accumulator identical would
// leave the epilogue as the only possible site.
static bool acc_sum_env() {
    static const bool on = si::env::flag("PF_DNNL_ACC_SUM");
    return on;
}

static void acc_sum_launch(sycl::queue & q, const int32_t * acc, size_t n, int64_t * sum, int slot) {
    constexpr int W = 256;
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<int64_t, 1> red(W, h);
        h.parallel_for(sycl::nd_range<1>((size_t)n, W), [=](sycl::nd_item<1> it) {
            red[it.get_local_id(0)] = (int64_t)acc[it.get_global_id(0)];
            it.barrier();
            if (it.get_local_id(0) == 0) {
                int64_t t = 0;
                for (int k = 0; k < W; k++) {
                    t = t * 1000003 + red[k];
                }
                sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                    a(sum[slot]);
                a.fetch_add(t);
            }
        });
    });
}

static bool actq_sum_env() {
    static const bool on = si::env::flag("PF_ACTQ_SUM");
    return on;
}

static bool dnnl_poison_env() {
    static const bool on = si::env::flag("PF_DNNL_POISON");
    return on;
}

static bool dnnl_qwait_env() {
    static const bool on = si::env::flag("PF_ATTN_QWAIT");
    return on;
}

// ---------------------------------------------------------------------------
// PF_ATTN_STREAM=1 puts the attention matmuls back on the separate `st_a`, which
// is the racy arrangement; it exists only to re-measure the race.  Unset/0 runs
// them on `st` with everything else.  See the note on `st_a` in impl.

// PF_DNNL_FPMATH: widen the matmul's accumulation class.
//
// The A770 fork of intel/omni_xpu_kernel records (onednn_s8u4_gemm.cpp:113-116):
// "A770 u4 matmul: narrow (fp16-class) accumulation overflows to NaN when
// K >= 5376 (Qwen-style K=12288). fpmath_mode::any with the second arg true
// allows the bf16 accumulation the jit:gemm DPAS path needs - wide exponent, no
// overflow (recorded A770 fix)".  This engine sets NO fpmath_mode at all.
//
// It has never bitten here: the largest K in a oneDNN int8 matmul is the 27B's
// n_embd = 5120, and the 35B A3B's is 2048, both under 5376.  So this is a guard
// for the next model up, not a fix for a bug that has been observed - and the
// claim is the LIBRARY's, not a measurement in this file.  Hence the switch:
// off by default so nothing changes until it is measured both ways on real shapes.
static inline void dnnl_fpmath(primitive_attr & attr) {
    static const int on = si::env::i32("PF_DNNL_FPMATH", 0);
    // PF_DNNL_FPMATH=2 forces `strict`, i.e. exact s32 accumulation.
    //
    // This is the test of the last surviving hypothesis on the oneDNN axis.
    // PF_GEMM_DNNL=0 is the ONLY arm that makes the prefill reproducible, and it is
    // reproducible because the dp4a path accumulates in int32 EXACTLY - which says
    // oneDNN's own accumulation is where the run-to-run difference comes from.  If
    // so, the accumulation is not plain s32: this file sets no fpmath_mode at all,
    // so oneDNN picks its own, and its default on this library is a reduced-precision
    // class (which is also why the A770 "K >= 5376 overflows to NaN" note exists -
    // that note is evidence the accumulation is NOT exact, not a curiosity).
    //
    // Reduced precision plus a runtime-variable accumulation order is exactly the
    // observed signature: a consistent 1-5% spread with a STABLE argmax, which is
    // not what a race looks like and not what garbage looks like.  And it is why
    // nothing else moved the needle - not threads (ONEDNN_MAX_CPU_ISA is an ISA
    // knob, not a threading knob, so threading was never actually disabled),
    // not ordering (both domains, every edge), not uninitialised memory (every
    // dev_alloc'd buffer poisoned with 0x7F), not warmup.
    //
    // DIAGNOSTIC.  `strict` is the right answer for an int8 GEMM whose output is
    // s32, so if this collapses the spread the fix is to set it unconditionally
    // rather than to leave oneDNN's default in charge.
    if (on == 2) {
        attr.set_fpmath_mode(fpmath_mode::strict);
    } else if (on) {
        attr.set_fpmath_mode(fpmath_mode::any, true);
    }
}
static bool attn_stream_env() {
    static const bool v = [] {
        const char * e = si::env::str("PF_ATTN_STREAM");
        return e && atoi(e) != 0;
    }();
    return v;
}

struct dnnl_gemm::impl {
    sycl::queue q;
    engine eng;
    stream st;
    // The XMX attention matmuls USED to get a second stream (`st_a`) so that a
    // wait on attention would not drain the dense GEMMs queued on `st`.  That
    // reasoning was wrong about SYCL: a stream is its own ordering domain, so two
    // streams built from the same queue are NOT ordered with respect to each
    // other, however "in-order" the queue is.  The consequence was a real data
    // race in `attn_xmx`'s per-block loop - iteration i+1's K/V gather (main
    // stream) overwrites the `b.k`/`b.v` rows that iteration i's `attn_pv`
    // (st_a) is still reading, and `attn_qk` can read K before the gather wrote
    // it.  It shows up as a prefill argmax that is not reproducible: 4779 on most
    // 64k runs and 9966 on one, while 16k (2 block chunks per call against 8)
    // was stable across four runs - the window scales with the chunk count.
    //
    // `st_a` is kept only so PF_ATTN_STREAM=1 can re-measure the split; the
    // attention matmuls run on `st` by default, which is where the rest of the
    // engine already lives and where ordering is known to work.  Nothing waits
    // inside `attn_xmx` in the shipped path, so the original motive for the
    // split (attention syncs draining the dense GEMMs) does not arise there.
    stream st_a;
    bool attn_sep = false; // run the attention matmuls on `st_a`; see attn_stream_env()
    // activation scratch: row-major [M][K] int8 + one scale per row
    int8_t * ax = nullptr;
    float * axs = nullptr;
    int32_t * axsum = nullptr;
    int cap_M = 0, cap_K = 0; // scratch capacity
    int cur_M = 0, cur_K = 0; // currently quantized activations
    bool acts_valid = false;
    // Which quantize() last ran.  A view carries this back, so "is this the
    // activation the caller thinks it is?" is one integer compare - and, unlike
    // re-deriving M and K at every entry point, it also catches a view that went
    // stale because something else quantized in between without changing the
    // shape.
    unsigned act_gen = 0;
    // whether the current activations include the even/odd k split (axe/axo)
    // that the native batched u4/k5 GEMM reads; false for the plain prefill
    // quantizer (PF_GEMM_DNNL mode-2), which skips the extra stores.
    bool split_valid = false;
    // integer matmul output [M][N] s32
    int32_t * acc = nullptr;
    size_t acc_cap = 0;
    // 4-bit path: f32 matmul output [M][N] and the per-group activation sums
    float * accf = nullptr;
    float * xs = nullptr; // [M][K/32]
    // PF_W4_CORR_GEMM: f16 left operand of the correction matmul, [M][K/32].
    // xsr[m][g] = xs[m][g]*sa[m][g], written by the same act_quant_grp_launch
    // as xs/asa (lane 0 only, so the u4 path pays no extra launch).
    uint16_t * corrsrc = nullptr;
    // PF_W4_GROUP > 32: the same plane rebinned to the *weight* group (K/G
    // entries per row), so the correction matmul's B operand (the per-G off
    // plane) and its A operand have matching K.  Built by a small reduce right
    // after the quantizer.
    uint16_t * corrsrcG = nullptr;
    int8_t * axg = nullptr;    // per-32-group quantized activations [M][K]
    uint16_t * asa = nullptr;  // their per-(m,g) f16 scales, [M][K/32]
    int8_t * axe = nullptr;    // even/odd k split of axg, for the u4 decode GEMV
    int8_t * axo = nullptr;
    // PF_ACTQ_SUM: per-call device checksums of the quantized activation and its
    // scale plane, written by act_quant_grp_launch and read once in the destructor.
    static constexpr int kActSumSlots = 4096;
    int64_t * actsum_q = nullptr;
    int64_t * actsum_s = nullptr;
    int actsum_n = 0;
    static constexpr int kAccSlots = 8192;
    int64_t * accsum = nullptr;
    int accsum_n = 0;
    bool actsum_shape = si::env::flag("PF_ACTQ_SHAPE");
    bool has_w4 = false;       // any u4 weight registered -> produce axg/asa
    // PF_LOAD_TIME: where the registration time goes.  The three parts have
    // different fixes - the host dequant+requantize is CPU (vectorise/thread),
    // the copies are pageable H2D (stage/pin), the ladder is oneDNN primitive
    // creation (lazy) - and they look identical from outside.
    bool pf_on = si::env::flag("PF_LOAD_TIME");
    // PF_DNNL_QCHK=1: recompute one row per tensor with the reference
    // (lround) formula and report mismatches.  The vectorised loop replaces
    // std::lround (a libm call the compiler cannot vectorise) with
    // add-copysign-truncate, which is the same round-half-away-from-zero for
    // every value in [-127,127], but the claim is checked rather than assumed.
    bool pf_qchk = si::env::flag("PF_DNNL_QCHK");
    double pf_conv_ms = 0, pf_copy_ms = 0, pf_prim_ms = 0, pf_spawn_ms = 0;
    int pf_n = 0;
    long pf_qbad = 0;

    // Host staging for the int8 conversion.  The conversion output is copied to
    // the device once, so it does not need a zero-initialised std::vector per
    // tensor: a reusable *pinned* buffer removes the memset, the allocation and
    // the pageable-copy penalty (measured ~1.15x on H2D, plus the per-tensor
    // page faults).  Capped so an LM-head-sized tensor (a 27B's 1.27 GB) does
    // not pin that much host memory; anything larger keeps the vector path.
    static constexpr size_t kStageMax = 256u << 20;
    // Two slots so a tensor's H2D can run while the next tensor is being
    // dequantized (the queue is in-order, so the copies queue up behind each
    // other and the host never blocks).  A slot is only reused after its last
    // copy is waited on, which by then is free: a full conversion ran in
    // between.
    struct stage_set {
        int8_t * p = nullptr;
        size_t cap = 0;
        uint16_t * ps = nullptr;
        size_t pcap = 0;
        bool pending = false;
    } sets[2];
    int cur_set = 0;
    bool pf_stage = [] {
        const char * e = si::env::str("PF_DNNL_NOSTAGE");
        return !(e && atoi(e) != 0);
    }();
    // PF_DNNL_SYNCCOPY=1 restores the per-tensor blocking copy (A/B).
    bool pf_async = [] {
        const char * e = si::env::str("PF_DNNL_SYNCCOPY");
        return !(e && atoi(e) != 0);
    }();
    int stage_begin() {
        stage_set & s = sets[cur_set];
        if (s.pending) {
            q.wait();
            s.pending = false;
        }
        return cur_set;
    }
    int8_t * stage_i8(int slot, size_t n) {
        if (!pf_stage || n > kStageMax) {
            return nullptr;
        }
        stage_set & s = sets[slot];
        if (n > s.cap) {
            if (s.p) {
                q.wait();
                sycl::free(s.p, q);
            }
            s.p = sycl::malloc_host<int8_t>(n, q);
            s.cap = s.p ? n : 0;
        }
        return s.p;
    }
    uint16_t * stage_u16(int slot, size_t n) {
        if (!pf_stage || n * 2 > kStageMax) {
            return nullptr;
        }
        stage_set & s = sets[slot];
        if (n > s.pcap) {
            if (s.ps) {
                q.wait();
                sycl::free(s.ps, q);
            }
            s.ps = sycl::malloc_host<uint16_t>(n, q);
            s.pcap = s.ps ? n : 0;
        }
        return s.ps;
    }
    void stage_done(int slot) {
        sets[slot].pending = true;
        cur_set ^= 1;
    }


    // int8 weight with a PER-32-GROUP scale (was per-row).  Every native quant
    // type here carries per-32 scales, so re-quantizing onto one per-row scale
    // was the dominant weight error (measured ~1% for *every* type, including
    // Q8_0 - see test_quant_audit).  Grouped scales cut that to ~0.1% at the
    // cost of oneDNN's f32-dst kernel (~+20% prefill).
    struct w_entry {
        int K = 0, N = 0, ng = 0;
        int8_t * dev = nullptr;      // [N][K] row-major int8, quantized per group
        uint16_t * scales = nullptr; // [ng][N] f16 step (oneDNN grouped scale)
        memory wmem, scmem;
        const uint16_t * off = nullptr; // per-32 f16 offset coefficient (lossless store)
        memory offmem;
        bool offmem_ok = false;
        bool ok = false;
        bool dev_backed = false;     // the caller owns dev/scales (a bank)
    };
    std::unordered_map<const void *, w_entry> weights;

    // 4-bit weight: u4 values [N][K] (K inner, low nibble first) plus the
    // per-32-group f16 step plane (oneDNN grouped scales) and f16 offset plane
    // (the correction coefficient).  See common/w4.h.
    struct w4_entry {
        int K = 0, N = 0;
        int G = kW4Group; // scale/offset group width (PF_W4_GROUP)
        uint8_t * vals = nullptr;
        uint16_t * scales = nullptr;
        uint16_t * off = nullptr;
        // The three planes are the caller's memory (the routed experts' resident
        // bank), not this instance's allocation: weight_bytes() must not count
        // them again, since the engine's own ledger already holds the bank.
        bool dev_backed = false;
        memory wmem, scmem;
        // PF_W4_CORR_GEMM: the same `off` plane as the {ng,N} f16 B operand of
        // the correction matmul.  One wrapper per weight entry (the layout is
        // fixed by the entry's K,N), so the execute path allocates nothing.
        memory offmem;
        bool offmem_ok = false;
        bool ok = false;
    };
    std::unordered_map<const void *, w4_entry> w4weights;
    // 2-bit draft store: the same three planes as a w4_entry (no oneDNN memory
    // objects - only the M=1 w2 GEMV reads it).
    std::unordered_map<const void *, w4_entry> w2weights;

    // Native-width 5-bit (Q5_K): the u4 nibble plane, a 1-bit fifth-bit plane
    // and the same two per-(g,n) f16 planes the u4 path keeps.  Decode
    // recombines the planes in-kernel; prefill expands q5 to int8 into
    // cb4_scratch and runs the grouped-scale int8 primitive with the u4
    // offset-correction epilogue (off = -dmin*m).
    struct k5_entry {
        int K = 0, N = 0, ng = 0;
        uint8_t * vals = nullptr;    // [N][K/2] nibbles, interleaved
        uint8_t * hi = nullptr;      // [N][K/8] fifth bit, 4 bytes per 32-group
        uint16_t * scales = nullptr; // [ng][N] f16 step = d*sc
        uint16_t * off = nullptr;    // [ng][N] f16 additive constant = -dmin*m
        bool ok = false;
    };
    std::unordered_map<const void *, k5_entry> k5weights;
    uint32_t * bit_lut = nullptr; // 16 x (4 bits -> 4 bytes of 0/1), for k5_expand

    // Codebook 4-bit (IQ4_XS / IQ4_NL): the native nibble indices plus the
    // per-(g,n) f16 scale, i.e. w = scale[g][n] * kvalues_iq4nl[q].  Decode
    // expands the table in-kernel (cb4_gemv_launch); prefill expands the
    // indices to int8 into cb4_scratch and feeds the *existing* int8 primitive,
    // which re-reads the weight memory at every execute (verified).
    struct cb4_entry {
        int K = 0, N = 0, ng = 0;
        uint8_t * idx = nullptr;     // [N][K/2] nibble indices, native order
        uint16_t * scales = nullptr; // [ng][N] f16, oneDNN grouped scale layout
        bool ok = false;
    };
    std::unordered_map<const void *, cb4_entry> cb4weights;
    int8_t * lut = nullptr;          // kvalues_iq4nl on the device
    uint16_t * lut16 = nullptr;      // 256-entry byte-pair table over it (see cb4_gemv_launch)
    int8_t * cb4_scratch = nullptr;  // shared int8 expansion for prefill
    size_t cb4_scratch_cap = 0;

    // Ownership: every long-lived device allocation is registered here at
    // creation (dev_alloc) and unregistered on grow-realloc (dev_erase); the
    // destructor frees the registry, so a newly added store cannot leak by
    // forgetting the dtor the way cb4weights/LUTs once did.  Allocation timing
    // and counts are unchanged - only the manual field-by-field frees go away.
    // Short-lived per-call temporaries (dsc, dx/dsw/dout) stay manual: they
    // are malloc/free balanced inside one function.
    std::vector<void *> owned_;
    template <typename T> T * dev_alloc(size_t n) {
        T * ptr = sycl::malloc_device<T>(n, q);
        if (ptr) {
            owned_.push_back(ptr);
            // PF_DNNL_POISON=1: fill every long-lived device buffer at creation
            // with 0x7F instead of leaving malloc_device's uninitialised bytes.
            // This is the oneDNN path's analogue of PF_XMX_POISON, and it is the
            // right next probe because PF_GEMM_DNNL=0 is the ONLY arm that makes
            // the prefill reproducible - but that arm does not merely disable the
            // matmul, it swaps the whole branch (activation quantizer, grouped
            // scale layout, epilogue) for the dp4a one.  So "oneDNN is necessary"
            // does not yet mean "oneDNN's GEMM is at fault", and uninitialised
            // memory anywhere in that branch is still untested.
            //
            // 0x7F bytes are ~1.7e38 as float and a large positive int32, so a
            // read of a poisoned buffer is loud rather than quietly plausible.
            // Poisoning at creation is safe: every one of these buffers is meant
            // to be written before it is read.
            if (dnnl_poison_env()) {
                q.memset(ptr, 0x7F, (size_t)n * sizeof(T)).wait();
            }
        }
        return ptr;
    }
    int actq_slot() {
        if (!actsum_q || actsum_n >= kActSumSlots) {
            return -1;
        }
        return actsum_n++;
    }
    // PF_ACTQ_SHAPE: what M/K each quantized call actually had.  Without it a
    // checksum whose base is a few hundred cannot be read - a difference of 334
    // out of M*K int8 values means something quite different at M=1 than at
    // M=512, and the slot dump alone does not say which.
    void actq_note(int M, int K, const void * key) {
        if (actsum_shape) {
            // N identifies WHICH tensor, so the per-call M/K/N sequence is
            // periodic with the layer, and the first differing slot maps to a
            // (layer, position-in-layer) pair by arithmetic on the period - no
            // cross-process-stable identifier needed, which matters because the
            // weight KEY is a pointer and pointer values are not stable across
            // processes (ASLR).
            int N = -1;
            auto it = weights.find(key);
            if (it != weights.end()) {
                N = it->second.N;
            }
            fprintf(stderr, "[actqshape] %d M=%d K=%d N=%d groups=%d values=%d\n", actsum_n, M, K, N,
                    (M * (K / kW4Group)), M * K);
        }
    }
    void dev_erase(void * ptr) {
        if (!ptr) {
            return;
        }
        sycl::free(ptr, q);
        for (size_t i = 0; i < owned_.size(); i++) {
            if (owned_[i] == ptr) {
                owned_[i] = owned_.back();
                owned_.pop_back();
                return;
            }
        }
    }

    struct prim4_entry {
        matmul prim;
        memory src, dst, sscales;
        // PF_W4_CORR_GEMM: the correction matmul for the same (M, K, N), cached
        // with it.  `cok` is false when oneDNN will not validate the f16
        // correction shape (or M == 1, where the old epilogue is kept); the
        // execute then falls back to w4_epilogue_launch.
        matmul cprim;
        bool cok = false;
        memory csrc, cdst;
        // The same correction with the residual folded in as a binary post-op
        // and the sum post-op last (dst += corr + residual).  Created lazily -
        // only the shapes that actually carry a residual pay for it - and only
        // usable when out is the compact [M][N] f32 layout.
        matmul rprim;
        bool rcok = false, rfails = false;
        // CSR-order expert GEMMs: persistent memory objects whose handles are
        // re-pointed at the caller's slices per tile.  Creating a memory object
        // per tile cost more than the primitive (~10-20 us of a ~36 us call);
        // set_data_handle is a store.
        memory xsrc, xsscales, xcsrc, xdst;
        bool xmem = false, xcsrc_ok = false;
    };
    std::unordered_map<uint64_t, prim4_entry> prims4;

    struct prim_entry {
        matmul prim;
        memory src, dst, sscales; // sscales wraps p->asa (grouped SRC scales)
        // rank-(K/32) offset-correction matmul for a lossless int8 store
        matmul cprim;
        memory csrc;
        bool cok = false, cfails = false;
        // the same persistent-handle trick as prim4_entry
        memory xsrc, xsscales, xcsrc, xdst;
        bool xmem = false, xcsrc_ok = false;
    };
    std::unordered_map<uint64_t, prim_entry> prims;

    static uint64_t pkey(int M, int K, int N) {
        return ((uint64_t)M << 42) | ((uint64_t)K << 21) | (uint64_t)N;
    }

    // ---- XMX attention GEMMs (PF_ATTN_XMX) --------------------------------
    // Plain int8 matmuls (no scale attributes) over caller-owned USM buffers.
    // QK reads K stored [N, K] (per key row of head_dim), hence "ba".
    struct attn_prim {
        matmul prim;
        bool ok = false;
    };
    std::unordered_map<uint64_t, attn_prim> attn_qk_prims;
    std::unordered_map<uint64_t, attn_prim> attn_pv_prims;

    const attn_prim * attn_qk_prim(int M, int K, int N) {
        const uint64_t k = pkey(M, K, N);
        auto it = attn_qk_prims.find(k);
        if (it != attn_qk_prims.end()) {
            return &it->second;
        }
        attn_prim e;
        try {
            auto xmd = memory::desc({M, K}, memory::data_type::s8, memory::format_tag::ab);
            auto wmd = memory::desc({K, N}, memory::data_type::s8, memory::format_tag::ba);
            auto dmd = memory::desc({M, N}, memory::data_type::s32, memory::format_tag::ab);
            e.prim = matmul(matmul::primitive_desc(eng, xmd, wmd, dmd));
            e.ok = true;
        } catch (const std::exception & ex) {
            fprintf(stderr, "[dnnl] attn_qk_prim M=%d K=%d N=%d failed: %s\n", M, K, N, ex.what());
        }
        auto ins = attn_qk_prims.emplace(k, std::move(e));
        return &ins.first->second;
    }

    const attn_prim * attn_pv_prim(int M, int K, int N) {
        const uint64_t k = pkey(M, K, N);
        auto it = attn_pv_prims.find(k);
        if (it != attn_pv_prims.end()) {
            return &it->second;
        }
        attn_prim e;
        try {
            auto xmd = memory::desc({M, K}, memory::data_type::u8, memory::format_tag::ab);
            auto wmd = memory::desc({K, N}, memory::data_type::s8, memory::format_tag::ab);
            auto dmd = memory::desc({M, N}, memory::data_type::s32, memory::format_tag::ab);
            e.prim = matmul(matmul::primitive_desc(eng, xmd, wmd, dmd));
            e.ok = true;
        } catch (const std::exception & ex) {
            fprintf(stderr, "[dnnl] attn_pv_prim M=%d K=%d N=%d failed: %s\n", M, K, N, ex.what());
        }
        auto ins = attn_pv_prims.emplace(k, std::move(e));
        return &ins.first->second;
    }

    // 4-bit variant of make_prim: u4 weights, grouped f16 scales, f32 dst
    prim4_entry * make_prim4(int M, int K, int N) {
        auto it = prims4.find(pkey(M, K, N));
        if (it != prims4.end()) {
            return &it->second;
        }
        if ((size_t)M * N > acc_cap || M > cap_M || K > cap_K) {
            return nullptr;
        }
        auto xmd = memory::desc({M, K}, memory::data_type::s8, memory::format_tag::ab);
        auto wmd = memory::desc({K, N}, memory::data_type::u4, memory::format_tag::ba);
        auto dmd = memory::desc({M, N}, memory::data_type::f32, memory::format_tag::ab);
        primitive_attr attr;
        dnnl_fpmath(attr);
        // mask = dim 0 (K) zoomed by the store's group; layouts validated on
        // this GPU (dims {1, NG*N} f16, exact against the integer reference).
        // PF_W4_GROUP only widens the *weight* group: the activation scales stay
        // per 32 (grouped SRC scales below), because the quantizer's per-32
        // granularity is what keeps the u4 prefill's accuracy and the epilogue's
        // xs/asa planes unchanged.
        attr.set_scales(DNNL_ARG_WEIGHTS, 1, {si::w4_group(), 1}, memory::data_type::f16);
        // grouped SRC scales: mask on dim 1 (K) of src {M,K}, groups {1,32}
        attr.set_scales(DNNL_ARG_SRC, 2, {1, kW4Group}, memory::data_type::f16);
        prim4_entry e;
        e.prim = matmul(matmul::primitive_desc(eng, xmd, wmd, dmd, attr));
        // the u4 primitive consumes the *grouped* activation quantization
        // (p->axg), not the per-row one used by add_weight()/gemm()
        e.src = sycl_interop::make_memory(xmd, eng, sycl_interop::memory_kind::usm, (void *)axg);
        e.dst = sycl_interop::make_memory(dmd, eng, sycl_interop::memory_kind::usm, (void *)accf);
        e.sscales = sycl_interop::make_memory(
            memory::desc({M, K / kW4Group}, memory::data_type::f16, memory::format_tag::ab), eng,
            sycl_interop::memory_kind::usm, (void *)asa);
        // PF_W4_CORR_GEMM: the offset-correction matmul for this same shape.
        // f16 src is the [M][ng] group-sum-times-scale plane, f16 B is the
        // weight's own off plane (bound at execute time, per weight), and the
        // sum post-op accumulates into accf - the u4 matmul's destination -
        // so the whole correction needs no [M][N] scratch.  M == 1 is the
        // decode GEMV's shape, so it never gets one (keeps that path identical).
        if (M > 1 && w4_corr_gemm_env() && corrsrc) {
            // The B operand is the weight's per-G off plane, so the A operand
            // must be binned to the same group: corrsrcG holds the F f16 sums
            // of corrsrc's per-32 entries (PF_W4_GROUP > 32), corrsrc itself
            // when G == 32.
            const int wg = si::w4_group();
            const int ng = K / wg;
            const uint16_t * csrc = (wg > kW4Group) ? corrsrcG : corrsrc;
            auto xc = memory::desc({M, ng}, memory::data_type::f16, memory::format_tag::ab);
            auto wc = memory::desc({ng, N}, memory::data_type::f16, memory::format_tag::ab);
            primitive_attr cattr;
            post_ops cop;
            cop.append_sum(1.0f); // dst := 1*(dst - 0) + op(...) = accf + corr
            cattr.set_post_ops(cop);
            try {
                e.cprim = matmul(matmul::primitive_desc(eng, xc, wc, dmd, cattr));
                e.csrc = sycl_interop::make_memory(xc, eng, sycl_interop::memory_kind::usm, (void *)csrc);
                e.cdst = sycl_interop::make_memory(dmd, eng, sycl_interop::memory_kind::usm, (void *)accf);
                e.cok = true;
            } catch (const std::exception & ex) {
                static bool once = true;
                if (once) {
                    once = false;
                    fprintf(stderr, "[dnnl] w4 correction prim M=%d ng=%d N=%d failed, using the scalar epilogue: %s\n",
                            M, ng, N, ex.what());
                }
            }
        }
        auto ins = prims4.emplace(pkey(M, K, N), std::move(e));
        return &ins.first->second;
    }

    // create (and cache) the primitive for one shape + its src/dst memories
    prim_entry * make_prim(int M, int K, int N) {
        auto it = prims.find(pkey(M, K, N));
        if (it != prims.end()) {
            return &it->second;
        }
        if ((size_t)M * N > acc_cap) {
            return nullptr;
        }
        if (M > cap_M || K > cap_K) {
            return nullptr;
        }
        auto xmd = memory::desc({M, K}, memory::data_type::s8, memory::format_tag::ab);
        // PF_DNNL_BLOCKED: force a blocked (XMX-friendly) weight layout instead of
        // oneDNN's plain ba.  Diagnostic: the weight buffer is not reordered into
        // it, so results are wrong - the point is to see via DNNL_VERBOSE whether
        // oneDNN then selects an XMX kernel for this small-M int8 matmul.
        static const int blocked = [] {
            const char * e = si::env::str("PF_DNNL_BLOCKED");
            return e ? atoi(e) : 0;
        }();
        auto wmd = memory::desc({K, N}, memory::data_type::s8,
                                blocked == 1   ? memory::format_tag::Ab4a
                                : blocked == 2 ? memory::format_tag::Ab4a
                                : blocked == 3 ? memory::format_tag::Ab8a
                                               : memory::format_tag::ba);
        // f32 dst: oneDNN applies the grouped weight/activation scales in float,
        // so the accumulator can no longer stay int32
        auto dmd = memory::desc({M, N}, memory::data_type::f32, memory::format_tag::ab);
        primitive_attr attr;
        dnnl_fpmath(attr);
        attr.set_scales(DNNL_ARG_WEIGHTS, 1, {kW4Group, 1}, memory::data_type::f16);
        attr.set_scales(DNNL_ARG_SRC, 2, {1, kW4Group}, memory::data_type::f16);
        prim_entry e;
        e.prim = matmul(matmul::primitive_desc(eng, xmd, wmd, dmd, attr));
        // src must be the per-32-group quantized activation (axg); the per-row
        // form (ax/axs) stays for the dp4a/SIn paths
        e.src = sycl_interop::make_memory(xmd, eng, sycl_interop::memory_kind::usm, (void *)axg);
        e.dst = sycl_interop::make_memory(dmd, eng, sycl_interop::memory_kind::usm, (void *)accf);
        e.sscales = sycl_interop::make_memory(
            memory::desc({M, K / kW4Group}, memory::data_type::f16, memory::format_tag::ab), eng,
            sycl_interop::memory_kind::usm, (void *)asa);
        auto ins = prims.emplace(pkey(M, K, N), std::move(e));
        return &ins.first->second;
    }
};

dnnl_gemm::dnnl_gemm(sycl::queue & q) : p(new impl) {
    p->q = q;
    p->eng = sycl_interop::make_engine(q.get_device(), q.get_context());
    p->st = sycl_interop::make_stream(p->eng, q);
    p->st_a = sycl_interop::make_stream(p->eng, q);
    p->attn_sep = attn_stream_env();
    p->cap_M = kMaxB * kMaxT; // max rows of one chunk-batched prefill
    p->cap_K = kActMaxK;
    p->ax = p->dev_alloc<int8_t>((size_t)p->cap_M * p->cap_K);
    p->axs = p->dev_alloc<float>(p->cap_M);
    p->axsum = p->dev_alloc<int32_t>(p->cap_M);
    // widest N any converted tensor has (27B ffn_gate/up = 17408; rounded to a
    // power of two), so every (M,N) up to a full kMaxB*kMaxT prefill tile can
    // create its primitive
    p->acc_cap = (size_t)p->cap_M * 32768;
    p->acc = p->dev_alloc<int32_t>(p->acc_cap);
    // 4-bit path scratch: f32 accumulator over the same [M][N] range plus the
    // per-group activation sums (K/32 per row, at most 1024)
    p->accf = p->dev_alloc<float>(p->acc_cap);
    p->xs = p->dev_alloc<float>((size_t)p->cap_M * (p->cap_K / kW4Group));
    // PF_W4_CORR_GEMM's f16 source: M*K/32 elements, at most 1024*1024 = 2 MiB
    p->corrsrc = p->dev_alloc<uint16_t>((size_t)p->cap_M * (p->cap_K / kW4Group));
    p->corrsrcG = p->dev_alloc<uint16_t>((size_t)p->cap_M * (p->cap_K / si::w4_group()));
    p->axg = p->dev_alloc<int8_t>((size_t)p->cap_M * p->cap_K);
    p->asa = p->dev_alloc<uint16_t>((size_t)p->cap_M * (p->cap_K / kW4Group));
    p->axe = p->dev_alloc<int8_t>((size_t)p->cap_M * p->cap_K / 2);
    p->axo = p->dev_alloc<int8_t>((size_t)p->cap_M * p->cap_K / 2);
    if (actq_sum_env()) {
        p->actsum_q = p->dev_alloc<int64_t>(impl::kActSumSlots);
        p->actsum_s = p->dev_alloc<int64_t>(impl::kActSumSlots);
    }
    if (acc_sum_env()) {
        p->accsum = p->dev_alloc<int64_t>(impl::kAccSlots);
    }
    if (!p->ax || !p->axs || !p->acc || !p->accf || !p->xs || !p->corrsrc || !p->corrsrcG || !p->axg || !p->asa || !p->axe
        || !p->axo) {
        throw std::runtime_error("dnnl scratch alloc failed");
    }
}

dnnl_gemm::~dnnl_gemm() {
    // One read-back, here, after everything has been submitted and waited on by the
    // engine - never inside the forward.  Two processes printing the same sequence
    // here have handed oneDNN bit-identical bytes; a differing sequence means the
    // difference entered at or before this kernel.
    if (p->actsum_q) {
        std::vector<int64_t> hq((size_t)p->actsum_n), hs((size_t)p->actsum_n);
        if (p->actsum_n > 0) {
            p->q.memcpy(hq.data(), p->actsum_q, (size_t)p->actsum_n * 8).wait();
            p->q.memcpy(hs.data(), p->actsum_s, (size_t)p->actsum_n * 8).wait();
        }
        uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < p->actsum_n; i++) {
            for (int k = 0; k < 8; k++) {
                h = (h ^ (uint64_t)((hq[i] >> (k * 8)) & 0xFF)) * 1099511628211ULL;
                h = (h ^ (uint64_t)((hs[i] >> (k * 8)) & 0xFF)) * 1099511628211ULL;
            }
        }
        fprintf(stderr, "[actq] calls=%d checksum=%016llx\n", p->actsum_n, (unsigned long long)h);
        if (p->accsum && p->accsum_n > 0) {
            std::vector<int64_t> ha((size_t)p->accsum_n);
            p->q.memcpy(ha.data(), p->accsum, (size_t)p->accsum_n * 8).wait();
            fprintf(stderr, "[accsum] calls=%d\n", p->accsum_n);
            for (int i = 0; i < p->accsum_n; i++) {
                fprintf(stderr, "[accsumslot] %d %lld\n", i, (long long)ha[i]);
            }
        }
        // Per-slot values, so two runs can be diffed to find the FIRST call whose
        // input already differed.  That is a bisection needing no new
        // instrumentation: slot 0 differing means the divergence is born at the
        // very first quantized GEMM of the run.
        for (int i = 0; i < p->actsum_n; i++) {
            fprintf(stderr, "[actqslot] %d %lld %lld\n", i, (long long)hq[i], (long long)hs[i]);
        }
    }
    // Every long-lived allocation was registered in owned_ at creation;
    // freeing the registry (not a per-field list) is what keeps a newly
    // added store from leaking by forgetting this function.
    p->q.wait();
    for (void * ptr : p->owned_) {
        sycl::free(ptr, p->q);
    }
    for (auto & s : p->sets) {
        if (s.p) {
            sycl::free(s.p, p->q);
        }
        if (s.ps) {
            sycl::free(s.ps, p->q);
        }
    }
}

// Host-side conversion of one tensor: GGUF K-quant rows -> int8 + one
// symmetric scale per output row.  The dequantized values are exactly what the
// SIn layout holds (the packed values are copied verbatim from the GGUF file),
// so the values are the ones the dp4a path uses; only the representation
// changes (per-row int8 instead of per-32 packed 4/5/6-bit).  Rows are
// independent -> convert in parallel with one thread per row slab.
bool dnnl_gemm::add_weight(const void * key, const void * host_data, uint32_t ggml_type, int K, int N, bool eager,
                           bool lazy) {
    if (!key || !host_data) {
        return false;
    }
    if (K <= 0 || N <= 0 || (K % 32) != 0 || K > kActMaxK) {
        return false;
    }
    if (p->weights.count(key)) {
        return p->weights[key].ok;
    }
    const size_t row_bytes = quant_row_bytes(ggml_type, K);
    std::vector<const char *> rows((size_t)N);
    for (int r = 0; r < N; r++) {
        rows[(size_t)r] = (const char *)host_data + (size_t)r * row_bytes;
    }
    return add_weight_rows(key, rows.data(), ggml_type, K, N, eager, lazy);
}

bool dnnl_gemm::add_weight_blocks(const void * key, const void * const * srcs, const int * rows_per_src, int n_src,
                                  uint32_t ggml_type, int K, int N, bool lazy) {
    if (!key || !srcs || !rows_per_src || n_src <= 0) {
        return false;
    }
    if (K <= 0 || N <= 0 || (K % 32) != 0 || K > kActMaxK) {
        return false;
    }
    if (p->weights.count(key)) {
        return p->weights[key].ok;
    }
    int total_rows = 0;
    for (int s = 0; s < n_src; s++) {
        if (!srcs[s] || rows_per_src[s] < 0) {
            return false;
        }
        total_rows += rows_per_src[s];
    }
    if (total_rows != N) {
        return false;
    }
    const size_t row_bytes = quant_row_bytes(ggml_type, K);
    std::vector<const char *> rows((size_t)N);
    int r = 0;
    for (int s = 0; s < n_src; s++) {
        for (int j = 0; j < rows_per_src[s]; j++, r++) {
            rows[(size_t)r] = (const char *)srcs[s] + (size_t)j * row_bytes;
        }
    }
    return add_weight_rows(key, rows.data(), ggml_type, K, N, /*eager=*/true, lazy);
}

bool dnnl_gemm::add_weight_rows(const void * key, const char * const * rows, uint32_t ggml_type, int K, int N,
                                bool eager, bool lazy) {
    if (!key || !rows) {
        return false;
    }
    // any format quant.h can dequantize row-wise converts to an int8 oneDNN
    // weight: the K-quants and the IQ codebook formats, plus the float types the
    // vision mmproj uses for its linears (f16 / f32 / bf16).
    switch (ggml_type) {
    case 0:  // F32
    case 1:  // F16
    case 30: // BF16
    case 6:  // Q5_0 (used when the row length is not a multiple of 256)
    case 8:  // Q8_0
    case 11: // Q3_K
    case 12: // Q4_K
    case 13: // Q5_K
    case 14: // Q6_K
    case 20: // IQ4_NL
    case 21: // IQ3_S
    case 23: // IQ4_XS
        break;
    default:
        return false;
    }
    if (K <= 0 || N <= 0 || (K % 32) != 0 || K > kActMaxK) {
        return false;
    }
    if (p->weights.count(key)) {
        return p->weights[key].ok;
    }

    const auto pf_t0 = std::chrono::steady_clock::now();
    const size_t nvals = (size_t)N * K;
    const int ng = K / kW4Group;
    std::vector<int8_t> hw_vec;
    std::vector<uint16_t> hs_vec;
    const int stage_slot = p->stage_begin();
    int8_t * hw = p->stage_i8(stage_slot, nvals);
    uint16_t * hs = p->stage_u16(stage_slot, (size_t)ng * N);
    if (!hw) {
        hw_vec.resize(nvals);
        hw = hw_vec.data();
    }
    if (!hs) {
        hs_vec.resize((size_t)ng * N);
        hs = hs_vec.data();
    }
    const int nthr = (int)std::min<size_t>((size_t)N, std::max<size_t>(1, std::thread::hardware_concurrency()));
    std::vector<std::vector<float>> frows(nthr, std::vector<float>(K));
    std::vector<std::thread> th;
    th.reserve(nthr);
    const auto pf_s0 = std::chrono::steady_clock::now();
    for (int t = 0; t < nthr; t++) {
        th.emplace_back([&, t] {
        std::vector<float> & fr = frows[t];
        for (int r = t; r < N; r += nthr) {
            dequantize_row(ggml_type, rows[(size_t)r], fr.data(), K);
            int8_t * q8 = hw + (size_t)r * K;
            for (int g = 0; g < ng; g++) {
                float amax = 0.f;
                for (int k = g * kW4Group; k < (g + 1) * kW4Group; k++) {
                    amax = std::max(amax, std::fabs(fr[k]));
                }
                const float sc = amax > 0.f ? amax / 127.0f : 1.0f;
                sycl::half h(sc);
                uint16_t bits;
                std::memcpy(&bits, &h, sizeof(h));
                hs[(size_t)g * N + r] = bits;
                const float inv = 1.0f / sc;
                if (p->pf_qchk && r == 0) {
                    for (int k = g * kW4Group; k < (g + 1) * kW4Group; k++) {
                        const int ref = (int)std::lround(fr[k] * inv);
                        const float t0 = fr[k] * inv;
                        const int got = (int)(t0 + (t0 >= 0.f ? 0.5f : -0.5f));
                        if (ref != got) {
                            p->pf_qbad++;
                        }
                    }
                }
                // Round-half-away-from-zero without the libm call: the compiler
                // vectorises this (measured 4x on the row conversion) where
                // std::lround left the loop scalar.
                #pragma clang loop vectorize(enable)
                for (int k = g * kW4Group; k < (g + 1) * kW4Group; k++) {
                    const float t0 = fr[k] * inv;
                    int v = (int)(t0 + (t0 >= 0.f ? 0.5f : -0.5f));
                    q8[k] = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
                }
            }
        }
        });
    }
    const auto pf_s1 = std::chrono::steady_clock::now();
    for (auto & x : th) {
        x.join();
    }
    if (p->pf_on) {
        p->pf_spawn_ms += std::chrono::duration<double, std::milli>(pf_s1 - pf_s0).count();
    }
    if (p->pf_on) {
        p->pf_conv_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pf_t0).count();
    }
    const auto pf_t1 = std::chrono::steady_clock::now();

    impl::w_entry w;
    w.K = K;
    w.N = N;
    w.ng = ng;
    w.dev = p->dev_alloc<int8_t>(nvals);
    w.scales = p->dev_alloc<uint16_t>((size_t)ng * N);
    if (!w.dev || !w.scales) {
        p->dev_erase(w.dev);
        p->dev_erase(w.scales);
        return false;
    }
    if (hw == p->sets[stage_slot].p && hs == p->sets[stage_slot].ps) {
        // staged: enqueue and let the next tensor's conversion overlap the copy
        p->q.memcpy(w.dev, hw, nvals);
        p->q.memcpy(w.scales, hs, (size_t)ng * N * 2);
        if (p->pf_async) {
            p->stage_done(stage_slot);
        } else {
            p->q.wait();
            p->sets[stage_slot].pending = false;
            p->cur_set ^= 1;
        }
    } else {
        p->q.memcpy(w.dev, hw, nvals).wait();
        p->q.memcpy(w.scales, hs, (size_t)ng * N * 2).wait();
    }
    if (p->pf_on) {
        p->pf_copy_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pf_t1).count();
    }
    const auto pf_t2 = std::chrono::steady_clock::now();
    static const int blocked = [] {
        const char * e = si::env::str("PF_DNNL_BLOCKED");
        return e ? atoi(e) : 0;
    }();
    w.wmem = sycl_interop::make_memory(memory::desc({K, N}, memory::data_type::s8,
                                                    blocked == 1   ? memory::format_tag::Ab4a
                                                    : blocked == 2 ? memory::format_tag::Ab4a
                                                    : blocked == 3 ? memory::format_tag::Ab8a
                                                                   : memory::format_tag::ba),
                                       p->eng,
                                       sycl_interop::memory_kind::usm, (void *)w.dev);
    w.scmem = sycl_interop::make_memory(memory::desc({1, ng * N}, memory::data_type::f16, memory::format_tag::ab),
                                        p->eng, sycl_interop::memory_kind::usm, (void *)w.scales);

    // Eagerly create every primitive the prefill paths can select: mode 2 uses
    // any multiple of kMaxT up to kMaxB*kMaxT rows (batched_prefill_fit), mode 1
    // uses kMaxT.  If a shape cannot run on this GPU, the tensor is marked
    // unsupported now so the engine falls back to dp4a before quantizing.
    bool any = !eager || lazy; // lazy: primitives are created on the first gemm() for each shape
    for (int M = kMaxT; eager && !lazy && M <= p->cap_M; M += kMaxT) {
        if ((size_t)M * N > p->acc_cap) {
            continue;
        }
        try {
            if (p->make_prim(M, K, N)) {
                any = true;
            }
        } catch (const std::exception & ex) {
            static bool once = true;
            if (once) {
                once = false;
                fprintf(stderr, "[dnnl] make_prim M=%d K=%d N=%d failed: %s\n", M, K, N, ex.what());
            }
        }
    }
    if (lazy) {
        // the 1-row decode runs inside a recorded graph; prime that shape now
        try {
            p->make_prim(1, K, N);
        } catch (const std::exception &) {
            any = false;
        }
    }
    w.ok = any;
    if (p->pf_qchk) {
        static long total_bad = 0;
        static int nchk = 0;
        total_bad += p->pf_qbad;
        p->pf_qbad = 0;
        nchk++;
        fprintf(stderr, "[qchk] %3d K=%-5d N=%-6d mismatches(total)=%ld\n", nchk, K, N, total_bad);
    }
    if (p->pf_on) {
        p->pf_prim_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pf_t2).count();
        p->pf_n++;
    }
    auto ins = p->weights.emplace(key, std::move(w));
    return ins.first->second.ok;
}

bool dnnl_gemm::has_weight(const void * key) const {
    auto it = p->weights.find(key);
    return it != p->weights.end() && it->second.ok;
}

size_t dnnl_gemm::weight_bytes() const {
    size_t n = 0;
    for (const auto & kv : p->weights) {
        if (kv.second.dev_backed) {
            continue; // the caller owns the planes (the routed down bank)
        }
        n += (size_t)kv.second.K * kv.second.N + (size_t)kv.second.ng * kv.second.N * 2;
    }
    for (const auto & kv : p->w4weights) {
        if (kv.second.dev_backed) {
            continue; // the caller owns the planes (see w4_entry::dev_backed)
        }
        n += (size_t)kv.second.K * kv.second.N / 2 + (size_t)(kv.second.K / kv.second.G) * kv.second.N * 4;
    }
    for (const auto & kv : p->w2weights) {
        n += (size_t)kv.second.K * kv.second.N / 4 + (size_t)(kv.second.K / kW4Group) * kv.second.N * 4;
    }
    for (const auto & kv : p->cb4weights) {
        n += (size_t)kv.second.K * kv.second.N / 2 + (size_t)kv.second.ng * kv.second.N * 2;
    }
    for (const auto & kv : p->k5weights) {
        n += (size_t)kv.second.K * kv.second.N / 2 + (size_t)kv.second.K * kv.second.N / 8
             + (size_t)kv.second.ng * kv.second.N * 4;
    }
    n += p->cb4_scratch_cap;
    return n;
}

bool dnnl_gemm::has_weight_k5(const void * key) const {
    auto it = p->k5weights.find(key);
    return it != p->k5weights.end() && it->second.ok;
}

// The 16-entry table mapping 4 bits to 4 bytes of 0/1 (see k5_expand_launch).
bool dnnl_gemm::add_weight_k5(const void * key, const void * host_data, uint32_t ggml_type, int K, int N) {
    if (!key || !host_data || !si::k5_supported(ggml_type)) {
        return false;
    }
    if (K <= 0 || N <= 0 || (K % kW4Group) != 0 || K > kActMaxK) {
        return false;
    }
    auto found = p->k5weights.find(key);
    if (found != p->k5weights.end()) {
        return found->second.ok;
    }
    si::k5t b;
    if (!si::k5_pack(ggml_type, host_data, K, N, b)) {
        return false;
    }
    impl::k5_entry e;
    e.K = K;
    e.N = N;
    e.ng = K / kW4Group;
    e.vals = p->dev_alloc<uint8_t>(b.vals.size());
    e.hi = p->dev_alloc<uint8_t>(b.hi.size());
    e.scales = p->dev_alloc<uint16_t>(b.scale.size());
    e.off = p->dev_alloc<uint16_t>(b.off.size());
    if (!e.vals || !e.hi || !e.scales || !e.off) {
        return false;
    }
    // blocking: `b` is a local and the async copies would outlive it
    p->q.memcpy(e.vals, b.vals.data(), b.vals.size()).wait();
    p->q.memcpy(e.hi, b.hi.data(), b.hi.size()).wait();
    p->q.memcpy(e.scales, b.scale.data(), b.scale.size() * sizeof(uint16_t)).wait();
    p->q.memcpy(e.off, b.off.data(), b.off.size() * sizeof(uint16_t)).wait();
    if (!p->bit_lut) {
        // the same table the decode kernel builds in SLM
        uint32_t host[16];
        for (int i = 0; i < 16; i++) {
            host[i] = (uint32_t)((i >> 0) & 1) | ((uint32_t)((i >> 1) & 1) << 8) | ((uint32_t)((i >> 2) & 1) << 16)
                      | ((uint32_t)((i >> 3) & 1) << 24);
        }
        // On this OOM path the entry planes stay registered (never emplaced),
        // so the destructor still frees them - previously they leaked.
        p->bit_lut = p->dev_alloc<uint32_t>(16);
        if (!p->bit_lut) {
            return false;
        }
        p->q.memcpy(p->bit_lut, host, sizeof(host)).wait();
    }
    // the prefill expansion needs the int8-sized scratch (shared with cb4)
    const size_t need = (size_t)K * (size_t)N;
    if (need > p->cb4_scratch_cap) {
        p->dev_erase(p->cb4_scratch);
        p->cb4_scratch = nullptr;
        p->cb4_scratch_cap = 0;
        p->cb4_scratch = p->dev_alloc<int8_t>(need);
        if (!p->cb4_scratch) {
            return false;
        }
        p->cb4_scratch_cap = need;
    }
    e.ok = true;
    auto ins = p->k5weights.emplace(key, std::move(e));
    return ins.first->second.ok;
}

bool dnnl_gemm::has_weight_cb4(const void * key) const {
    auto it = p->cb4weights.find(key);
    return it != p->cb4weights.end() && it->second.ok;
}

bool dnnl_gemm::add_weight_cb4(const void * key, const void * host_data, uint32_t ggml_type, int K, int N) {
    if (!key || !host_data || !si::cb4_supported(ggml_type)) {
        return false;
    }
    if (K <= 0 || N <= 0 || (K % kW4Group) != 0 || K > kActMaxK) {
        return false;
    }
    auto found = p->cb4weights.find(key);
    if (found != p->cb4weights.end()) {
        return found->second.ok;
    }
    si::cb4t b;
    if (!si::cb4_pack(ggml_type, host_data, K, N, b)) {
        return false;
    }
    impl::cb4_entry e;
    e.K = K;
    e.N = N;
    e.ng = K / kW4Group;
    e.idx = p->dev_alloc<uint8_t>(b.idx.size());
    e.scales = p->dev_alloc<uint16_t>(b.scale.size());
    if (!e.idx || !e.scales) {
        p->dev_erase(e.idx);
        p->dev_erase(e.scales);
        return false;
    }
    p->q.memcpy(e.idx, b.idx.data(), b.idx.size()).wait();
    p->q.memcpy(e.scales, b.scale.data(), b.scale.size() * sizeof(uint16_t)).wait();
    if (!p->lut) {
        p->lut = p->dev_alloc<int8_t>(16);
        if (!p->lut) {
            return false;
        }
        p->q.memcpy(p->lut, kvalues_iq4nl, 16).wait();
        // the byte-pair table the kernels consume: one 256-entry lookup turns an
        // interleaved index byte (two nibbles) into two codebook values
        uint16_t h16[256];
        for (int b = 0; b < 256; b++) {
            h16[b] = (uint16_t)((uint8_t)kvalues_iq4nl[b & 0xF] | ((uint16_t)(uint8_t)kvalues_iq4nl[b >> 4] << 8));
        }
        p->lut16 = p->dev_alloc<uint16_t>(256);
        if (!p->lut16) {
            return false;
        }
        p->q.memcpy(p->lut16, h16, sizeof(h16)).wait();
    }
    // the prefill expansion needs the int8-sized scratch; grow it to the widest
    // converted tensor (the biggest is ffn_gate/up, ~95 MB, not the full model)
    const size_t need = (size_t)K * (size_t)N;
    if (need > p->cb4_scratch_cap) {
        p->dev_erase(p->cb4_scratch);
        p->cb4_scratch = nullptr;
        p->cb4_scratch_cap = 0;
        p->cb4_scratch = p->dev_alloc<int8_t>(need);
        if (!p->cb4_scratch) {
            return false;
        }
        p->cb4_scratch_cap = need;
    }
    e.ok = true;
    auto ins = p->cb4weights.emplace(key, std::move(e));
    return ins.first->second.ok;
}

// Convert one tensor to the 4-bit form (see common/w4.h): u4 values plus the
// per-32-group f16 step/offset planes.  Returns false for unsupported types,
// which keep their int8 conversion.
bool dnnl_gemm::add_weight_w4(const void * key, const void * host_data, uint32_t ggml_type, int K, int N,
                              bool any_type, bool gemv_only, float scale_mul, const char * cls, int layer,
                              bool lazy) {
    const bool w4info = si::env::flag("PF_W4_INFO");
    // PF_W4_ALL_ONLY: a set list turns the generic pack on for its classes and
    // off for everything else, so it overrides both w4_supported() and
    // any_type.  Unset = the original behaviour.
    const bool filtered = si::w4_only_set();
    const bool listed = filtered && si::w4_only_allows(cls, layer);
    if (!key || !host_data || (filtered ? !listed : !(any_type || si::w4_supported(ggml_type)))) {
        if (w4info) {
            fprintf(stderr, "[w4] reject key=%p: precheck type=%u any=%d cls=%s layer=%d listed=%d\n", key,
                    ggml_type, (int)any_type, cls ? cls : "-", layer, (int)listed);
        }
        return false;
    }
    const int G = si::w4_group();
    if (K <= 0 || N <= 0 || (K % G) != 0 || K > kActMaxK) {
        if (w4info) {
            fprintf(stderr, "[w4] reject key=%p: K=%d N=%d maxK=%d\n", key, K, N, (int)kActMaxK);
        }
        return false;
    }
    auto found = p->w4weights.find(key);
    if (found != p->w4weights.end()) {
        return found->second.ok;
    }
    const auto pf_t0 = std::chrono::steady_clock::now();
    si::w4t w;
    // A natively-representable type keeps its exact grid; every other type
    // (including a class that PF_W4_ALL_ONLY selected) takes the generic pack.
    // Unfiltered callers keep the original any_type/pack dispatch exactly.
    const bool native = (ggml_type == 12) || (ggml_type == 13 && si::w4_k5_only());
    const bool use_any = filtered ? !native : any_type;
    if (!(use_any ? si::w4_pack_any(ggml_type, host_data, K, N, w) : si::w4_pack(ggml_type, host_data, K, N, w))) {
        if (w4info) {
            fprintf(stderr, "[w4] reject key=%p: pack failed type=%u K=%d N=%d\n", key, ggml_type, K, N);
        }
        return false;
    }
    // scale_mul: diagnostic multiplier on the store's per-(g,n) f16 step plane
    // (used by the DFlash draft to test a weight-scale hypothesis; 1 = off)
    if (scale_mul != 1.0f) {
        for (size_t i = 0; i < w.scale.size(); i++) {
            uint16_t h;
            memcpy(&h, &w.scale[i], 2);
            float v = ggml_half_to_float(h) * scale_mul;
            h = ggml_float_to_half(v);
            memcpy(&w.scale[i], &h, 2);
        }
    }
    // The packed store's own group width (PF_W4_GROUP): its scale/off planes and
    // the oneDNN grouped-scale mask both follow it.
    const int ng = K / w.G;
    impl::w4_entry e;
    e.K = K;
    e.N = N;
    e.G = w.G;
    if (p->pf_on) {
        p->pf_conv_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pf_t0).count();
    }
    const auto pf_t1 = std::chrono::steady_clock::now();
    if (w4info) {
        fprintf(stderr, "[w4] register key=%p type=%u K=%d N=%d (%.4f B/weight)\n", key, ggml_type, K, N,
                (double)(w.vals.size() + w.scale.size() * 2 + w.off.size() * 2) / ((double)K * N));
    }
    e.vals = p->dev_alloc<uint8_t>(w.vals.size());
    e.scales = p->dev_alloc<uint16_t>(w.scale.size());
    e.off = p->dev_alloc<uint16_t>(w.off.size());
    if (!e.vals || !e.scales || !e.off) {
        p->dev_erase(e.vals);
        p->dev_erase(e.scales);
        p->dev_erase(e.off);
        return false;
    }
    p->q.memcpy(e.vals, w.vals.data(), w.vals.size()).wait();
    p->q.memcpy(e.scales, w.scale.data(), w.scale.size() * 2).wait();
    p->q.memcpy(e.off, w.off.data(), w.off.size() * 2).wait();
    if (p->pf_on) {
        p->pf_copy_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pf_t1).count();
    }
    const auto pf_t2 = std::chrono::steady_clock::now();
    e.wmem = sycl_interop::make_memory(memory::desc({K, N}, memory::data_type::u4, memory::format_tag::ba), p->eng,
                                       sycl_interop::memory_kind::usm, (void *)e.vals);
    e.scmem =
        sycl_interop::make_memory(memory::desc({1, ng * N}, memory::data_type::f16, memory::format_tag::ab), p->eng,
                                  sycl_interop::memory_kind::usm, (void *)e.scales);
    // PF_W4_CORR_GEMM: the same off plane as the [ng][N] f16 B operand of the
    // correction matmul.  Wrapped once here, so a GEMM execute only rebinds the
    // pointer's memory object into the cached primitive.
    e.offmem = sycl_interop::make_memory(memory::desc({ng, N}, memory::data_type::f16, memory::format_tag::ab),
                                         p->eng, sycl_interop::memory_kind::usm, (void *)e.off);
    e.offmem_ok = e.offmem.get(true) != nullptr;

    // Eagerly create the prefill primitives (same M ladder as add_weight): if no
    // shape can run on this GPU the tensor is marked unsupported and the engine
    // keeps the int8 conversion.
    // A tensor only becomes 4-bit if *every* M the engine can ask for has a
    // primitive.  Otherwise a later call would fail and the engine would fall
    // back to the fp32 path, which needs the raw device weight that converted
    // tensors never upload - that would read garbage.  cap_M = kMaxB*kMaxT is
    // the largest M any plan uses, so requiring it covers every call.
    //
    // `any_type` (the draft-only LM head and MTP layer tensors) is consumed by
    // the M=1 u4 GEMV and the M <= 13 native GEMM, neither of which touches the
    // oneDNN f32 accumulator scratch or needs a prim4 - and a 248k-wide head can
    // never have one, which is why the target head is int8.  Register it
    // GEMV-only.  `gemv_only` is the same thing for a *native* (Q4_K) tensor
    // whose prefill ladder cannot be built at this shape.
    bool any = any_type || gemv_only || lazy;
    const int ladder[] = {1, kMaxT, 2 * kMaxT, 3 * kMaxT, 4 * kMaxT, 8 * kMaxT, 16 * kMaxT};
    if (!(any_type || gemv_only || lazy)) {
        if ((size_t)p->cap_M * (size_t)N > p->acc_cap) {
            if (w4info) {
                // Say so: the packing succeeded and everything above was
                // allocated, and only this budget check keeps the tensor on
                // int8 - the LM head (248320 rows) is rejected here, so a
                // "register" line printed before it reads as a conversion that
                // never happens.
                fprintf(stderr, "[w4] reject key=%p type=%u K=%d N=%d: cap_M*N=%zu > acc_cap=%zu\n", key, ggml_type,
                        K, N, (size_t)p->cap_M * (size_t)N, p->acc_cap);
            }
            p->dev_erase(e.vals);
            p->dev_erase(e.scales);
            p->dev_erase(e.off);
            return false;
        }
        // M=1 is the decode shape (the 4-bit path has no dedicated GEMV yet, so the
        // oneDNN matmul runs at M=1); the rest is the prefill ladder.
        for (int M : ladder) {
            if (M > p->cap_M || (size_t)M * N > p->acc_cap) {
                continue;
            }
            try {
                if (p->make_prim4(M, K, N)) {
                    any = true;
                }
            } catch (const std::exception & ex) {
                static bool once = true;
                if (once) {
                    once = false;
                    fprintf(stderr, "[dnnl] make_prim4 M=%d K=%d N=%d failed: %s\n", M, K, N, ex.what());
                }
            }
        }
    }
    if (lazy && !any_type && !gemv_only) {
        // Only the 1-row shape can be reached inside a recorded command graph
        // (the multi-device decode).  Everything else is a direct replay, where
        // gemm_w4() creates the primitive on first use.
        try {
            p->make_prim4(1, K, N);
        } catch (const std::exception &) {
            any = false;
        }
    }
    e.ok = any;
    if (w4info) {
        fprintf(stderr, "[w4] %s key=%p type=%u K=%d N=%d (%.4f B/weight)\n", any ? "register" : "reject(no M ladder)",
                key, ggml_type, K, N,
                (double)(w.vals.size() + w.scale.size() * 2 + w.off.size() * 2) / ((double)K * N));
    }
    if (any) {
        p->has_w4 = true; // quantize() must now also produce the grouped form
    }
    if (p->pf_on) {
        p->pf_prim_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pf_t2).count();
        p->pf_n++;
    }
    auto ins = p->w4weights.emplace(key, std::move(e));
    return ins.first->second.ok;
}

// Register a u4 store whose three planes already live in device memory.  This is
// the routed experts' resident gate/up bank: the engine packs E experts into one
// `E * plane` device buffer and calls this once per expert with a pointer into
// it, so nothing is re-packed or copied here.  The oneDNN memories wrap the
// caller's pointers, and the primitive is created lazily by gemm_w4 (it is keyed
// by (M,K,N), which is the same for every expert of a tensor).
bool dnnl_gemm::add_weight_w4_dev(const void * key, const void * dev_vals, const void * dev_scale, const void * dev_off,
                                  int K, int N) {
    if (!key || !dev_vals || !dev_scale || !dev_off) {
        return false;
    }
    const int G = si::w4_group();
    if (K <= 0 || N <= 0 || (K % G) != 0 || K > kActMaxK) {
        return false;
    }
    auto found = p->w4weights.find(key);
    if (found != p->w4weights.end()) {
        return found->second.ok;
    }
    const int ng = K / G;
    impl::w4_entry e;
    e.K = K;
    e.N = N;
    e.G = G;
    e.vals = (uint8_t *)dev_vals;
    e.scales = (uint16_t *)dev_scale;
    e.off = (uint16_t *)dev_off;
    e.dev_backed = true;
    e.wmem = sycl_interop::make_memory(memory::desc({K, N}, memory::data_type::u4, memory::format_tag::ba), p->eng,
                                       sycl_interop::memory_kind::usm, (void *)e.vals);
    e.scmem =
        sycl_interop::make_memory(memory::desc({1, ng * N}, memory::data_type::f16, memory::format_tag::ab), p->eng,
                                  sycl_interop::memory_kind::usm, (void *)e.scales);
    e.offmem = sycl_interop::make_memory(memory::desc({ng, N}, memory::data_type::f16, memory::format_tag::ab),
                                         p->eng, sycl_interop::memory_kind::usm, (void *)e.off);
    e.offmem_ok = e.offmem.get(true) != nullptr;
    e.ok = true;
    p->has_w4 = true; // quantize() must produce the grouped form
    auto ins = p->w4weights.emplace(key, std::move(e));
    return ins.first->second.ok;
}

// Register a per-32-group int8 store whose two planes already live in device
// memory: the routed-expert down bank.  Layout matches add_weight()'s output
// exactly - values [N][K] int8 with an amax/127 scale per 32-value group, and a
// [ng][N] f16 step plane - so gemm()'s plain int8 branch consumes it with no
// other change.  The planes belong to the caller; weight_bytes() must not count
// them again (the engine's own ledger holds the bank).
bool dnnl_gemm::add_weight_i8_dev(const void * key, const void * dev_vals, const void * dev_scales,
                                  const void * dev_off, int K, int N) {
    if (!key || !dev_vals || !dev_scales || K <= 0 || N <= 0 || (K % kW4Group) != 0 || K > kActMaxK) {
        return false;
    }
    auto found = p->weights.find(key);
    if (found != p->weights.end()) {
        return found->second.ok;
    }
    const int ng = K / kW4Group;
    impl::w_entry e;
    e.K = K;
    e.N = N;
    e.ng = ng;
    e.dev = (int8_t *)dev_vals;
    e.scales = (uint16_t *)dev_scales;
    e.off = (const uint16_t *)dev_off;
    e.dev_backed = true;
    if (dev_off) {
        e.offmem = sycl_interop::make_memory(memory::desc({ng, N}, memory::data_type::f16, memory::format_tag::ab),
                                             p->eng, sycl_interop::memory_kind::usm, (void *)dev_off);
        e.offmem_ok = e.offmem.get(true) != nullptr;
    }
    e.wmem = sycl_interop::make_memory(memory::desc({K, N}, memory::data_type::s8, memory::format_tag::ba), p->eng,
                                       sycl_interop::memory_kind::usm, (void *)e.dev);
    e.scmem = sycl_interop::make_memory(memory::desc({1, ng * N}, memory::data_type::f16, memory::format_tag::ab),
                                        p->eng, sycl_interop::memory_kind::usm, (void *)e.scales);
    e.ok = true;
    auto ins = p->weights.emplace(key, std::move(e));
    return ins.first->second.ok;
}

bool dnnl_gemm::has_weight_w4(const void * key) const {
    auto it = p->w4weights.find(key);
    return it != p->w4weights.end() && it->second.ok;
}

// The u4 store's three device planes.  The DFlash2 candidate selector reads
// single per-token rows of its codebooks (one Q4_K block each), which no GEMM
// entry point can express, so it needs the planes themselves.
bool dnnl_gemm::w4_planes(const void * key, const uint8_t ** vals, const uint16_t ** scales, const uint16_t ** offs,
                          int * K, int * N) const {
    auto it = p->w4weights.find(key);
    if (it == p->w4weights.end() || !it->second.ok) {
        return false;
    }
    if (vals) {
        *vals = it->second.vals;
    }
    if (scales) {
        *scales = it->second.scales;
    }
    if (offs) {
        *offs = it->second.off;
    }
    if (K) {
        *K = it->second.K;
    }
    if (N) {
        *N = it->second.N;
    }
    return true;
}

// The 2-bit draft store.  No oneDNN memory object: the only consumer is the
// M=1 w2 GEMV, so the three device planes are all that is needed.
bool dnnl_gemm::add_weight_w2(const void * key, const void * host_data, uint32_t ggml_type, int K, int N) {
    if (!key || !host_data || K <= 0 || N <= 0 || (K % kW4Group) != 0) {
        return false;
    }
    auto found = p->w2weights.find(key);
    if (found != p->w2weights.end()) {
        return found->second.ok;
    }
    si::w2t w;
    if (!si::w2_pack_any(ggml_type, host_data, K, N, w)) {
        return false;
    }
    impl::w4_entry e;
    e.K = K;
    e.N = N;
    e.vals = p->dev_alloc<uint8_t>(w.vals.size());
    e.scales = p->dev_alloc<uint16_t>(w.scale.size());
    e.off = p->dev_alloc<uint16_t>(w.off.size());
    if (!e.vals || !e.scales || !e.off) {
        p->dev_erase(e.vals);
        p->dev_erase(e.scales);
        p->dev_erase(e.off);
        return false;
    }
    p->q.memcpy(e.vals, w.vals.data(), w.vals.size()).wait();
    p->q.memcpy(e.scales, w.scale.data(), w.scale.size() * 2).wait();
    p->q.memcpy(e.off, w.off.data(), w.off.size() * 2).wait();
    e.ok = true;
    if (si::env::flag("PF_W2_INFO")) {
        fprintf(stderr, "[w2] %s: K=%d N=%d  %.4f B/weight  rel L2 vs source = %.3f%%\n", (const char *)key, K, N,
                (double)(w.vals.size() + w.scale.size() * 2 + w.off.size() * 2) / ((double)K * N), 100.0 * w.rel_l2);
    }
    auto ins = p->w2weights.emplace(key, std::move(e));
    return ins.first->second.ok;
}

bool dnnl_gemm::has_weight_w2(const void * key) const {
    auto it = p->w2weights.find(key);
    return it != p->w2weights.end() && it->second.ok;
}

// out[n] = sum_g asa[g] * (scale[g][n] * qd[g] + off[g][n] * xs[g]), qd the
// int8 dot product of the group's quantized activation with the 2-bit levels.
bool dnnl_gemm::gemm_w2(const act_view & v, const void * key, const float * residual, float alpha, int M, int K, float * out,
                        int out_stride) {
    static const bool dbg = si::env::flag("PF_W2_DEBUG");
    auto it = p->w2weights.find(key);
    if (it == p->w2weights.end() || !it->second.ok || M != 1 || it->second.K != K) {
        if (dbg) {
            fprintf(stderr, "[w2] miss: entry=%d ok=%d M=%d K=%d (entry K=%d)\n",
                    (int)(it != p->w2weights.end()), (int)(it != p->w2weights.end() && it->second.ok), M, K,
                    it == p->w2weights.end() ? -1 : it->second.K);
        }
        return false;
    }
    if (!view_is_current(v) || M != v.m || K != v.k || !p->axg || !p->asa || !p->xs) {
        if (dbg) {
            fprintf(stderr, "[w2] miss: view gen=%u/%u curM=%d/%d curK=%d/%d axg=%p asa=%p xs=%p\n", v.gen,
                    p->act_gen, M, p->cur_M, K, p->cur_K, (void *)p->axg, (void *)p->asa, (void *)p->xs);
        }
        return false;
    }
    if (dbg) {
        static int once = 0;
        if (once++ < 4) {
            fprintf(stderr, "[w2] hit: M=%d K=%d N=%d\n", M, K, it->second.N);
        }
    }
    const impl::w4_entry & w = it->second;
    w2_gemv_launch(p->q, w.vals, w.scales, w.off, p->axg, p->asa, p->xs, out, out_stride, residual, alpha, K, w.N);
    return true;
}

// One execute per cached primitive with whatever is in the scratch buffers.
// The values are irrelevant (the int32 accumulator cannot overflow on 8-bit
// inputs); the point is that each primitive's GPU kernel is loaded once here
// instead of during the first request.  The activation quantizer and the
// epilogue are launched once too: their SYCL kernels would otherwise be
// JIT-compiled inside the first request.
void dnnl_gemm::report_load(const char * who) const {
    if (!p->pf_on) {
        return;
    }
    const double tot = p->pf_conv_ms + p->pf_copy_ms + p->pf_prim_ms;
    fprintf(stderr,
            "[load]   dnnl %-12s %3d tensors: conv %7.1f ms (spawn %.1f), copy %6.1f ms, prim %6.1f ms (%.1f)\n", who,
            p->pf_n, p->pf_conv_ms, p->pf_spawn_ms, p->pf_copy_ms, p->pf_prim_ms, tot);
}

int dnnl_gemm::warmup() {
    const bool tdbg = si::env::flag("PF_DNNL_TIME");
    const auto t0 = std::chrono::high_resolution_clock::now();
    int n = 0;
    for (auto & it : p->prims) {
        const uint64_t key = it.first;
        const int K = (int)((key >> 21) & 0x1FFFFF);
        const int N = (int)(key & 0x1FFFFF);
        const memory * wmem = nullptr;
        for (auto & w : p->weights) {
            if (w.second.ok && w.second.K == K && w.second.N == N) {
                wmem = &w.second.wmem;
                break;
            }
        }
        if (!wmem) {
            continue;
        }
        impl::prim_entry & pe = it.second;
        // dnnl_capture_guard() below throws when a oneDNN execute is reached
        // during a declared-all-SYCL capture; a manual free would then be
        // skipped and leak this dummy plane (ng*N f16, up to ~1 MB) for every
        // M in the ladder, i.e. exactly when the run is already failing.  The
        // guard is declared outside the try so unwinding still releases it,
        // and the alloc stays inside so the catch still swallows a throw.
        qfree<uint16_t> dsc;
        try {
            const auto t1 = std::chrono::high_resolution_clock::now();
            // the grouped scales must be supplied now; dummy planes are fine
            dsc.p    = sycl::malloc_device<uint16_t>((size_t)(K / kW4Group) * N, p->q);
            dsc.q    = &p->q;
            if (!dsc) {
                continue;
            }
            auto scmem = sycl_interop::make_memory(
                memory::desc({1, (memory::dim)((K / kW4Group) * N)}, memory::data_type::f16, memory::format_tag::ab),
                p->eng,
                sycl_interop::memory_kind::usm, (void *)dsc.get());
            dnnl_capture_guard();
    if (dnnl_qwait_env()) {
        p->q.wait();
    }
            pe.prim.execute(p->st, {{DNNL_ARG_SRC, pe.src},
                                    {DNNL_ARG_WEIGHTS, *wmem},
                                    {DNNL_ARG_DST, pe.dst},
                                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scmem},
                                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, pe.sscales}});
    if (dnnl_qwait_env()) {
        p->q.wait();
        // ALSO the stream's own wait.  A queue wait only orders against work
        // submitted to the queue, so if `p->st` is a distinct ordering domain
        // (it is an interop stream built from the queue) then waiting on `p->q`
        // does not wait for the primitive at all, and the whole ordering
        // hypothesis would have been tested against the wrong domain.  PF_ATTN_WAIT
        // did use the stream's wait, but only on the two attention primitives and
        // only under the perturbing fingerprint oracle, which this whole entry
        // shows is not a valid arbiter.  Both domains, one arm.
        p->st.wait();
    }
            if (tdbg) {
                p->q.wait();
                fprintf(
                    stderr, "[dnnl] warm M=%d K=%d N=%d: %.2f ms\n", (int)(key >> 42), K, N,
                    std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t1).count());
            }
        } catch (const std::exception &) {
            continue;
        }
        n++;
    }
    p->q.wait();
    if (tdbg) {
        fprintf(stderr, "[dnnl] warmup %d prims: %.1f ms\n", n,
                std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count());
    }
    // SYCL-side warmup: quantizer + vec4 epilogue on dummy data
    const int WM = kMaxT, WK = 1024, WN = 6144;
    float * dx = sycl::malloc_device<float>((size_t)WM * WK, p->q);
    float * dsw = sycl::malloc_device<float>((size_t)WN, p->q);
    float * dout = sycl::malloc_device<float>((size_t)WM * WN, p->q);
    if (dx && dsw && dout) {
        act_quant_launch(p->q, dx, nullptr, WK, WK, p->ax, p->axs, p->axsum, WM, WK);
        p->actq_note(WM, WK, nullptr);
    act_quant_grp_launch(p->q, dx, nullptr, WK, WK, p->axg, p->asa, p->xs, p->corrsrc, p->axe, p->axo, true, WM, WK,
                         p->actsum_q, p->actsum_s, p->actq_slot());
        epilogue_f32_launch(p->q, p->accf, dout, WN, nullptr, 1.f, WM, WN);
        p->q.wait();
    }
    if (dx) {
        sycl::free(dx, p->q);
    }
    if (dsw) {
        sycl::free(dsw, p->q);
    }
    if (dout) {
        sycl::free(dout, p->q);
    }
    return n;
}

act_view dnnl_gemm::quantize(const float * x, const float * up, int x_stride, int up_stride, int M, int K,
                             bool do_split) {
    act_view v;
    p->acts_valid = false;
    ++p->act_gen; // any view from before this call is now stale, valid or not
    if (!x || M <= 0 || M > p->cap_M || K <= 0 || K > p->cap_K) {
        return v;
    }
    // The per-row int8 form (ax/axs/axsum) is no longer consumed: both weight
    // paths read the per-32-group form (axg + asa for the oneDNN matmul's
    // grouped SRC scales, and axg/xs for the decode GEMVs).  The only caller of
    // act_data()/act_scales()/act_sum() is a dead branch (it also requires a
    // per-row weight scale, which weight_scales() no longer returns), so this
    // kernel was pure per-call latency: it runs one 256-thread workgroup over
    // the whole row and cost ~13 ms/token across the 320 calls of a 27B decode.
    // PF_ROWACT=1 restores it for A/B.
    static const bool row_act = [] {
        const char * e = si::env::str("PF_ROWACT");
        return e && atoi(e) != 0;
    }();
    if (row_act) {
        act_quant_launch(p->q, x, up, x_stride, up_stride, p->ax, p->axs, p->axsum, M, K);
    }
    // Both weight paths consume the per-32-group form now (oneDNN grouped SRC
    // scales and the grouped GEMVs); the per-row form stays for dp4a/SIn.  The
    // same launch produces the group sums (needed by the u4 GEMV and the u4
    // prefill epilogue) and, for decode, the even/odd k split the u4 GEMV
    // reads, so no separate view kernels are needed.
    p->actq_note(M, K, nullptr); // quantize() has no weight key; (M,K) is periodic per layer
    act_quant_grp_launch(p->q, x, up, x_stride, up_stride, p->axg, p->asa, p->xs, p->corrsrc, p->axe, p->axo,
                         do_split, M, K, p->actsum_q, p->actsum_s, p->actq_slot());
    // PF_W4_GROUP > 32: the correction matmul's A operand must be binned to the
    // weight store's group (its B is the per-G off plane).  One small kernel per
    // quantize; at G == 32 the per-32 plane is already the right shape.
    if (p->has_w4 && M > 1 && si::w4_group() > kW4Group) {
        w4_corr_group_launch(p->q, p->corrsrc, p->corrsrcG, si::w4_group() / kW4Group, M, K / kW4Group);
    }
    p->cur_M = M;
    p->cur_K = K;
    p->acts_valid = true;
    p->split_valid = do_split;
    v.m = M;
    v.k = K;
    v.split = do_split;
    v.gen = p->act_gen;
    return v;
}

act_view dnnl_gemm::quantize_gathered(const int8_t * xq, const uint16_t * asa, const float * xs,
                                      const int32_t * pair, int p0, int n, int top_k, int tile_m, int K) {
    act_view v;
    p->acts_valid = false;
    ++p->act_gen;
    // pair == nullptr: a linear gather (rows p0 .. p0+n-1), which the banked
    // prefill's down projection uses because its activation plane is already in
    // CSR order.
    if (!xq || !asa || !xs || n < 0 || n > tile_m || tile_m <= 0 || tile_m > p->cap_M || K <= 0 ||
        K > p->cap_K || K % 32 != 0 || (!pair && top_k != 0)) {
        return v;
    }
    static const bool abl_qg = [] {
        const char * e = si::env::str("PF_MOE_ABL_QG");
        return e && atoi(e) != 0;
    }();
    if (!abl_qg)
        moe_u4_gather_i8_launch(p->q, xq, asa, xs, pair, p0, n, tile_m, top_k, K, p->axg, p->axe, p->axo, p->asa,
                                p->xs, p->corrsrc);
    if (p->has_w4 && tile_m > 1 && si::w4_group() > kW4Group) {
        w4_corr_group_launch(p->q, p->corrsrc, p->corrsrcG, si::w4_group() / kW4Group, tile_m, K / kW4Group);
    }
    p->cur_M = tile_m;
    p->cur_K = K;
    p->acts_valid = true;
    // The gather also writes the even/odd k planes, so the native batched GEMM
    // (nat_gemm, extended to the prefill's 32-row tiles) can consume this view.
    p->split_valid = (p->axe != nullptr && p->axo != nullptr);
    v.m = tile_m;
    v.k = K;
    v.split = p->split_valid;
    v.gen = p->act_gen;
    return v;
}

bool dnnl_gemm::view_is_current(const act_view & v) const {
    return v.valid() && p->acts_valid && v.gen == p->act_gen;
}

bool dnnl_gemm::gemm(const act_view & v, const void * key, const float * residual, float alpha, int M, int K, float * out, int out_stride) {
    if (!view_is_current(v) || M != v.m || K != v.k) {
        return false;
    }
    if (auto kit = p->k5weights.find(key); kit != p->k5weights.end() && kit->second.ok) {
        impl::k5_entry & e = kit->second;
        if (e.K != K || !view_is_current(v) || M != v.m || K != v.k) {
            return false;
        }
        if (M == 1) {
            // decode: recombine the two 5-bit planes in-kernel (0.75 B/weight)
            k5_gemv_launch(p->q, e.vals, e.hi, e.scales, e.off, p->axe, p->axo, p->asa, p->xs, out, residual, alpha, K,
                           e.N);
            return true;
        }
        // batched verify: same native 5-bit stream (0.75 B/weight) for all M
        if (v.split && p->axe && p->axo && p->axg && p->asa && p->xs && p->bit_lut
            && nat_gemm_launch(p->q, 1, e.vals, e.hi, nullptr, e.scales, e.off, nullptr, p->bit_lut, p->axe, p->axo,
                               p->axg, p->asa, p->xs, out, out_stride, residual, alpha, M, K, e.N)) {
            return true;
        }
        // prefill: expand q5 to int8 into the shared scratch, then the ordinary
        // grouped-scale int8 primitive with the u4 offset-correction epilogue
        // (the step plane is already in the grouped [g][N] f16 layout oneDNN
        // wants, so only the values need materializing).
        if (!p->cb4_scratch || (size_t)K * (size_t)e.N > p->cb4_scratch_cap) {
            return false;
        }
        impl::prim_entry * pe = nullptr;
        try {
            pe = p->make_prim(M, K, e.N);
        } catch (const std::exception &) {
            return false;
        }
        if (!pe) {
            return false;
        }
        k5_expand_launch(p->q, e.vals, e.hi, p->bit_lut, p->cb4_scratch, K, e.N);
        auto wmem = sycl_interop::make_memory(
            memory::desc({K, e.N}, memory::data_type::s8, memory::format_tag::ba), p->eng,
            sycl_interop::memory_kind::usm, (void *)p->cb4_scratch);
        auto scmem = sycl_interop::make_memory(
            memory::desc({1, e.ng * e.N}, memory::data_type::f16, memory::format_tag::ab), p->eng,
            sycl_interop::memory_kind::usm, (void *)e.scales);
        dnnl_capture_guard();
    if (dnnl_qwait_env()) {
        p->q.wait();
    }
        pe->prim.execute(p->st, {{DNNL_ARG_SRC, pe->src},
                                 {DNNL_ARG_WEIGHTS, wmem},
                                 {DNNL_ARG_DST, pe->dst},
                                 {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scmem},
                                 {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, pe->sscales}});
        // PF_DNNL_ACC_SUM.  This is the k5 prefill path - the one the 35B A3B's qkv
        // actually takes.  The two `w.wmem` sites are the plain int8 and u4
        // variants and neither fires here, which is why the first run of this probe
        // reported calls=0.
        if (acc_sum_env() && p->accsum && p->accsum_n < impl::kAccSlots) {
            acc_sum_launch(p->q, p->acc, (size_t)M * e.N, p->accsum, p->accsum_n++);
        }
    if (dnnl_qwait_env()) {
        p->q.wait();
        // ALSO the stream's own wait.  A queue wait only orders against work
        // submitted to the queue, so if `p->st` is a distinct ordering domain
        // (it is an interop stream built from the queue) then waiting on `p->q`
        // does not wait for the primitive at all, and the whole ordering
        // hypothesis would have been tested against the wrong domain.  PF_ATTN_WAIT
        // did use the stream's wait, but only on the two attention primitives and
        // only under the perturbing fingerprint oracle, which this whole entry
        // shows is not a valid arbiter.  Both domains, one arm.
        p->st.wait();
    }
        // PF_K5_NOCORR (diagnostic): drop the offset correction, keeping only the
        // step-scaled matmul - tells whether the prefill cost is the expansion
        // kernel or the u4-style correction epilogue.
        static const bool nocorr = [] {
            const char * ev = si::env::str("PF_K5_NOCORR");
            return ev && atoi(ev) != 0;
        }();
        w4_epilogue_launch(p->q, p->accf, p->xs, nocorr ? nullptr : e.off, p->asa, out, out_stride, residual, alpha, M,
                           e.N, e.ng, 1);
        return true;
    }
    if (auto cit = p->cb4weights.find(key); cit != p->cb4weights.end() && cit->second.ok) {
        impl::cb4_entry & e = cit->second;
        if (e.K != K || !view_is_current(v) || M != v.m || K != v.k) {
            return false;
        }
        if (M == 1) {
            // decode: expand the codebook in-kernel (half the weight bytes)
            cb4_gemv_launch(p->q, e.idx, p->lut16, e.scales, p->axg, p->asa, p->xs, out, residual, alpha, K, e.N);
            return true;
        }
        // batched verify: the same index plane + in-kernel LUT, no int8 scratch
        // (split_valid as above: the same activation form as the decode/verify)
        if (v.split && p->axg && p->asa && p->xs && p->lut16
            && nat_gemm_launch(p->q, 2, e.idx, nullptr, nullptr, e.scales, nullptr, p->lut16, nullptr, nullptr,
                               nullptr, p->axg, p->asa, p->xs, out, out_stride, residual, alpha, M, K, e.N)) {
            return true;
        }
        // prefill: materialize this tensor's indices as int8 into the shared
        // scratch, then run the ordinary int8 primitive over it.  oneDNN reads
        // the weight memory at every execute (verified), so the scratch can be
        // reused by every codebook tensor in turn.
        if (!p->cb4_scratch || (size_t)K * (size_t)e.N > p->cb4_scratch_cap) {
            return false;
        }
        impl::prim_entry * pe = nullptr;
        try {
            pe = p->make_prim(M, K, e.N);
        } catch (const std::exception &) {
            return false;
        }
        if (!pe) {
            return false;
        }
        cb4_expand_launch(p->q, e.idx, p->lut16, p->cb4_scratch, K, e.N);
        auto wmem = sycl_interop::make_memory(
            memory::desc({K, e.N}, memory::data_type::s8, memory::format_tag::ba), p->eng,
            sycl_interop::memory_kind::usm, (void *)p->cb4_scratch);
        auto scmem = sycl_interop::make_memory(
            memory::desc({1, e.ng * e.N}, memory::data_type::f16, memory::format_tag::ab), p->eng,
            sycl_interop::memory_kind::usm, (void *)e.scales);
        dnnl_capture_guard();
    if (dnnl_qwait_env()) {
        p->q.wait();
    }
        pe->prim.execute(p->st, {{DNNL_ARG_SRC, pe->src},
                                 {DNNL_ARG_WEIGHTS, wmem},
                                 {DNNL_ARG_DST, pe->dst},
                                 {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scmem},
                                 {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, pe->sscales}});
    if (dnnl_qwait_env()) {
        p->q.wait();
        // ALSO the stream's own wait.  A queue wait only orders against work
        // submitted to the queue, so if `p->st` is a distinct ordering domain
        // (it is an interop stream built from the queue) then waiting on `p->q`
        // does not wait for the primitive at all, and the whole ordering
        // hypothesis would have been tested against the wrong domain.  PF_ATTN_WAIT
        // did use the stream's wait, but only on the two attention primitives and
        // only under the perturbing fingerprint oracle, which this whole entry
        // shows is not a valid arbiter.  Both domains, one arm.
        p->st.wait();
    }
        epilogue_f32_launch(p->q, p->accf, out, out_stride, residual, alpha, M, e.N);
        return true;
    }
    auto it = p->weights.find(key);
    if (it == p->weights.end() || !it->second.ok) {
        return false;
    }
    impl::w_entry & w = it->second;
    if (w.K != K) {
        return false;
    }
    // batched verify: grouped-scale int8 read once for all M rows (the oneDNN
    // primitive below is ~2-3x slower per byte for these small-M shapes).
    // split_valid gates it to the calls whose activation form this kernel is
    // built for (mode-0 decode / the mtp_dry verify): a mode-1/mode-2 prefill
    // quantizes with do_split=false and then reads *different* (stale) grouped
    // activation views - without the guard the 0.8B multi-device
    // decode-vs-prefill test fails with nat on and passes with PF_NAT=0.
    if (M >= 2 && v.split && p->axg && p->asa && p->xs) {
        if (nat_gemm_launch(p->q, 3, nullptr, nullptr, w.dev, w.scales, nullptr, nullptr, nullptr, nullptr, nullptr,
                            p->axg, p->asa, p->xs, out, out_stride, residual, alpha, M, K, w.N)) {
            return true;
        }
    }
    auto pit = p->prims.find(impl::pkey(M, K, w.N));
    if (pit == p->prims.end()) {
        impl::prim_entry * e = nullptr;
        try {
            e = p->make_prim(M, K, w.N);
        } catch (const std::exception &) {
            return false;
        }
        if (!e) {
            return false;
        }
        pit = p->prims.find(impl::pkey(M, K, w.N));
    }
    if (M == 1) {
        // decode: the oneDNN matmul is all per-call overhead at one row, so run
        // the dedicated grouped-scale int8 GEMV instead (PF_I8_NOGEMV=1 restores
        // the matmul, for A/B).
        static const bool no_gemv = [] {
            const char * e = si::env::str("PF_I8_NOGEMV");
            return e && atoi(e) != 0;
        }();
        if (!no_gemv) {
            i8_grp_gemv_launch(p->q, w.dev, w.scales, p->axg, p->asa, p->xs, out, residual, alpha, K, w.N);
            return true;
        }
    }
    impl::prim_entry & pe = pit->second;
    // A compact [M][N] destination with no residual and alpha == 1 can be the
    // primitive's dst directly: the f32 epilogue pass and its kernel launch
    // disappear.  The routed down bank uses this to write straight into its
    // CSR-ordered accumulator.
    const bool direct_i8 = out != nullptr && residual == nullptr && alpha == 1.0f && out_stride == w.N &&
                           ((uintptr_t)out % 64) == 0;
    memory dst_m = pe.dst;
    if (direct_i8) {
        dst_m = sycl_interop::make_memory(pe.dst.get_desc(), p->eng, sycl_interop::memory_kind::usm, (void *)out);
    }
    dnnl_capture_guard();
    if (dnnl_qwait_env()) {
        p->q.wait();
    }
    pe.prim.execute(p->st, {{DNNL_ARG_SRC, pe.src},
                            {DNNL_ARG_WEIGHTS, w.wmem},
                            {DNNL_ARG_DST, dst_m},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, w.scmem},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, pe.sscales}});
    if (acc_sum_env() && p->accsum && p->accsum_n < impl::kAccSlots) {
        acc_sum_launch(p->q, p->acc, (size_t)M * w.N, p->accsum, p->accsum_n++);
    }
    if (dnnl_qwait_env()) {
        p->q.wait();
        // ALSO the stream's own wait.  A queue wait only orders against work
        // submitted to the queue, so if `p->st` is a distinct ordering domain
        // (it is an interop stream built from the queue) then waiting on `p->q`
        // does not wait for the primitive at all, and the whole ordering
        // hypothesis would have been tested against the wrong domain.  PF_ATTN_WAIT
        // did use the stream's wait, but only on the two attention primitives and
        // only under the perturbing fingerprint oracle, which this whole entry
        // shows is not a valid arbiter.  Both domains, one arm.
        p->st.wait();
    }
    // both the weight and the activation scales are applied by the matmul now
    // A lossless store keeps the source's q and carries its zero point in the
    // off plane: accf already has the step (weights scales) and the activation
    // scale applied, so add sum_g off*xs*sa exactly as the u4 path does.  The
    // scalar epilogue over ng=16 groups is far too slow for N=2048 (measured
    // 32 s of the 3213-token prefill), so use the same rank-ng correction
    // matmul the u4 path uses, with this weight's off plane as its B operand.
    if (w.off) {
        if (!pit->second.cok && !pit->second.cfails) {
            try {
                const int ng32 = K / kW4Group;
                auto xc = memory::desc({M, ng32}, memory::data_type::f16, memory::format_tag::ab);
                auto wc = memory::desc({ng32, w.N}, memory::data_type::f16, memory::format_tag::ab);
                primitive_attr cattr;
                post_ops cop;
                cop.append_sum(1.0f);
                cattr.set_post_ops(cop);
                pit->second.cprim = matmul(matmul::primitive_desc(p->eng, xc, wc, pit->second.dst.get_desc(), cattr));
                pit->second.csrc =
                    sycl_interop::make_memory(xc, p->eng, sycl_interop::memory_kind::usm, (void *)p->corrsrc);
                pit->second.cok = true;
            } catch (const std::exception &) {
                pit->second.cfails = true;
            }
        }
        if (pit->second.cok && w.offmem_ok) {
            pit->second.cprim.execute(p->st, {{DNNL_ARG_SRC, pit->second.csrc},
                                              {DNNL_ARG_WEIGHTS, w.offmem},
                                              {DNNL_ARG_DST, dst_m}});
            if (!direct_i8) {
                epilogue_f32_launch(p->q, p->accf, out, out_stride, residual, alpha, M, w.N);
            }
        } else {
            w4_epilogue_launch(p->q, p->accf, p->xs, w.off, p->asa, out, out_stride, residual, alpha, M, w.N, w.ng, 1);
        }
    } else {
        if (!direct_i8) {
            epilogue_f32_launch(p->q, p->accf, out, out_stride, residual, alpha, M, w.N);
        }
    }
    return true;
}

// 4-bit prefill GEMM: out = alpha*sx[m]*(acc4 + sum_g off[g][n]*xs[m][g]) +
// residual.  acc4 comes from oneDNN with the grouped f16 step scales applied;
// the offset (zero-point) part is added by w4_epilogue_launch from the group
// sums of the same quantized activations.
bool dnnl_gemm::gemm_w4(const act_view & v, const void * key, const float * residual, float alpha, int M, int K, float * out,
                        int out_stride) {
    if (!view_is_current(v) || M != v.m || K != v.k) {
        return false;
    }
    auto it = p->w4weights.find(key);
    if (it == p->w4weights.end() || !it->second.ok) {
        return false;
    }
    impl::w4_entry & w = it->second;
    if (w.K != K) {
        return false;
    }
    // MTP speculative verify: read the 0.625 B/weight native u4 stream once for
    // all M rows at the card's read ceiling, instead of oneDNN's grouped
    // f32-dst matmul.  Needs the even/odd activation split (mtp_dry / decode).
    if (M >= 2 && v.split && p->axe && p->axo && p->axg && p->asa && p->xs) {
        if (nat_gemm_launch(p->q, 0, w.vals, nullptr, nullptr, w.scales, w.off, nullptr, nullptr, p->axe, p->axo,
                            p->axg, p->asa, p->xs, out, out_stride, residual, alpha, M, K, w.N)) {
            return true;
        }
    }
    static int n_fail = 0;
    if (!p->prims4.count(impl::pkey(M, K, w.N)) && M > p->cap_M) {
        if (n_fail++ < 5) {
            fprintf(stderr, "[dnnl] w4 gemm cannot run M=%d K=%d N=%d (no primitive)\n", M, K, w.N);
        }
        return false;
    }
    auto pit = p->prims4.find(impl::pkey(M, K, w.N));
    if (pit == p->prims4.end()) {
        impl::prim4_entry * e = nullptr;
        try {
            e = p->make_prim4(M, K, w.N);
        } catch (const std::exception &) {
            return false;
        }
        if (!e) {
            return false;
        }
        pit = p->prims4.find(impl::pkey(M, K, w.N));
    }
    static const bool w4dbg = [] {
        const char * e = si::env::str("PF_W4_DEBUG");
        return e && atoi(e) != 0;
    }();
    if (w4dbg) {
        static int nrep = 0;
        static long hist[4] = {0, 0, 0, 0};
        hist[M <= 1 ? 0 : (M <= 32 ? 1 : (M <= 64 ? 2 : 3))]++;
        if (nrep < 6) {
            nrep++;
            const bool has_axe = p->axe != nullptr && p->axo != nullptr;
            fprintf(stderr,
                    "[w4] gemm_w4 M=%d K=%d N=%d key=%p actsv=%d split=%d axe=%d/%d out=%p res=%p alpha=%.2f "
                    "(buckets: 1=%ld 32=%ld 64=%ld big=%ld)\n",
                    M, K, w.N, key, (int)p->acts_valid, (int)v.split, (int)(p->axe != nullptr), (int)has_axe,
                    (void *)out, (void *)residual, alpha, hist[0], hist[1], hist[2], hist[3]);
        }
    }
    static const bool no_gemv = [] {
        const char * e = si::env::str("PF_W4_NOGEMV");
        return e && atoi(e) != 0;
    }();
    if (M == 1) {
        // decode: one row, so the oneDNN matmul is all per-call overhead.  Run
        // the dedicated u4 GEMV instead (same formula, 16-byte nibble loads).
        if (!no_gemv) {
            w4_gemv_launch(p->q, w.vals, w.scales, w.off, p->axe, p->axo, p->asa, p->xs, out, out_stride, residual,
                           alpha, K, w.N);
            return true;
        }
    } else {
        // Small batch (MTP speculative verify, short prefill tail): keep the
        // native u4 weights and stream them once for all M rows.  oneDNN's
        // grouped int8 matmul reads a different (1.5x larger) weight set at
        // ~1/4 the rate, so this is the difference between the verify costing
        // one decode pass and costing four.  Opt-in via PF_W4_GEMM_MAXM until
        // it beats the matmul on a given shape.
        static const int gemm_max_m = [] {
            const char * e = si::env::str("PF_W4_GEMM_MAXM");
            return e ? atoi(e) : 1;
        }();
        // the kernel is instantiated for M <= 9 (the MTP verify range); larger
        // batches keep the oneDNN matmul
        if (M >= 2 && M <= 8 && M <= gemm_max_m && !no_gemv && p->axe && p->axo && p->asa && p->xs) {
            w4_gemm_launch(p->q, w.vals, w.scales, w.off, p->axe, p->axo, p->asa, p->xs, out, out_stride, residual,
                           alpha, M, K, w.N);
            return true;
        }
    }
    impl::prim4_entry & pe = pit->second;
    const int N = w.N;
    const int ng = K / w.G;
    // PF_W4_NOCORR: bisection knob - drop the offset (zero-point) correction and
    // keep only the oneDNN step-scaled matmul, to tell which half is wrong.
    static const bool nocorr = [] {
        const char * e = si::env::str("PF_W4_NOCORR");
        return e && atoi(e) != 0;
    }();
    // PF_W4_CORR_GEMM: the off-plane correction runs as a second oneDNN matmul
    // whose sum post-op accumulates into the same dst as the u4 matmul.  The
    // correction's A operand is binned to the store's group (corrsrcG when
    // PF_W4_GROUP > 32), so its K matches the per-G off plane for any G.
    const bool corr_gemm = !nocorr && w4_corr_gemm_env() && pe.cok && w.offmem_ok;
    // Without a residual (and with alpha == 1) the sum post-op already leaves
    // the finished result in the destination, so the remaining "epilogue" would
    // be a plain f32 copy: let the u4 matmul write `out` itself and the copy
    // disappears (that is the 2-gemm number the dense lab measures).  The same
    // shortcut applies with a residual by folding it into the correction as a
    // binary add post-op.  Both need `out` (and the residual) to be the compact
    // [M][N] f32 layout the primitive's dst descriptor expects (out_stride == N)
    // and USM pointers oneDNN may load from (64 B); otherwise the accf +
    // epilogue path below is kept.
    const bool compact_out = out != nullptr && out_stride == N && ((uintptr_t)out % 64) == 0;
    const bool compact_res = residual != nullptr && ((uintptr_t)residual % 64) == 0;
    const bool alpha1 = alpha == 1.0f;
    // The residual-fused form is only valid when `residual` is a *different*
    // buffer from `out`.  With residual == out (the engine's in-place residual
    // stream: attn_output "o" and GDN "ssm_out" are called that way) the u4
    // matmul writes `out` in place first, so the binary post-op would read the
    // u4 result as the residual: dst = u4 + (u4 + corr) instead of
    // residual + u4 + corr.  The residual is then ~2x the layer output, which
    // compounds to garbage over the 40 layers (measured: cos 0.998 against the
    // scalar epilogue per GEMM, but the token stream degrades immediately).
    // Nothing about this is shape-dependent: the lab reproduces it at every
    // (K,N), including the K=2048 ones, whenever it passes out == residual.
    const bool res_aliases_out = residual != nullptr && residual == out;
    bool res_fuse = false;
    if (corr_gemm && alpha1 && compact_out && compact_res && !res_aliases_out) {
        if (!pe.rcok && !pe.rfails) {
            try {
                auto xc = memory::desc({M, ng}, memory::data_type::f16, memory::format_tag::ab);
                auto wc = memory::desc({ng, N}, memory::data_type::f16, memory::format_tag::ab);
                primitive_attr rattr;
                post_ops rop;
                // binary add first, sum last - both are additions, so their
                // order (and thus the fp association) is the only difference
                // from the old (acc + corr) + residual.
                rop.append_binary(algorithm::binary_add, pe.dst.get_desc());
                rop.append_sum(1.0f);
                rattr.set_post_ops(rop);
                pe.rprim = matmul(matmul::primitive_desc(p->eng, xc, wc, pe.dst.get_desc(), rattr));
                pe.rcok = true;
            } catch (const std::exception & ex) {
                pe.rfails = true;
                static bool once = true;
                if (once) {
                    once = false;
                    fprintf(stderr, "[dnnl] w4 residual-fused correction M=%d ng=%d N=%d failed: %s\n", M, ng, N,
                            ex.what());
                }
            }
        }
        res_fuse = pe.rcok;
    }
    const bool direct = corr_gemm && residual == nullptr && alpha1 && compact_out;
    memory dst_m = pe.dst;
    if (direct || res_fuse) {
        dst_m = sycl_interop::make_memory(pe.dst.get_desc(), p->eng, sycl_interop::memory_kind::usm, (void *)out);
    }
    dnnl_capture_guard();
    if (dnnl_qwait_env()) {
        p->q.wait();
    }
    pe.prim.execute(p->st, {{DNNL_ARG_SRC, pe.src},
                            {DNNL_ARG_WEIGHTS, w.wmem},
                            {DNNL_ARG_DST, dst_m},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, w.scmem},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, pe.sscales}});
    if (acc_sum_env() && p->accsum && p->accsum_n < impl::kAccSlots) {
        acc_sum_launch(p->q, p->acc, (size_t)M * w.N, p->accsum, p->accsum_n++);
    }
    if (dnnl_qwait_env()) {
        p->q.wait();
        // ALSO the stream's own wait.  A queue wait only orders against work
        // submitted to the queue, so if `p->st` is a distinct ordering domain
        // (it is an interop stream built from the queue) then waiting on `p->q`
        // does not wait for the primitive at all, and the whole ordering
        // hypothesis would have been tested against the wrong domain.  PF_ATTN_WAIT
        // did use the stream's wait, but only on the two attention primitives and
        // only under the perturbing fingerprint oracle, which this whole entry
        // shows is not a valid arbiter.  Both domains, one arm.
        p->st.wait();
    }
    if (corr_gemm) {
        // PF_W4_CORR_GEMM: dst = u4 matmul (step-scaled) + off^T * (xs*sa),
        // the correction accumulated by oneDNN's sum post-op.  The source plane
        // (p->corrsrc) was written by the quantizer that produced xs/asa, so no
        // kernel and no [M][N] scratch is added.  In the non-direct case `dst`
        // is accf and the remaining pass is alpha/residual (the same one pass
        // the old epilogue did); in the direct case `dst` is `out` and there is
        // nothing left to run.
        dnnl_capture_guard();
        if (res_fuse) {
            memory rmem = sycl_interop::make_memory(pe.dst.get_desc(), p->eng, sycl_interop::memory_kind::usm,
                                                    (void *)const_cast<float *>(residual));
            pe.rprim.execute(p->st, {{DNNL_ARG_SRC, pe.csrc},
                                     {DNNL_ARG_WEIGHTS, w.offmem},
                                     {DNNL_ARG_DST, dst_m},
                                     {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, rmem}});
            return true;
        }
        pe.cprim.execute(p->st, {{DNNL_ARG_SRC, pe.csrc}, {DNNL_ARG_WEIGHTS, w.offmem}, {DNNL_ARG_DST, dst_m}});
        if (!direct) {
            epilogue_f32_launch(p->q, p->accf, out, out_stride, residual, alpha, M, N);
        }
        return true;
    }
    // xs/asa are the quantizer's per-32 planes (NG = K/32); the weight's off
    // plane is per G, so OF = G/32 consecutive activation groups share one entry.
    w4_epilogue_launch(p->q, p->accf, p->xs, nocorr ? nullptr : w.off, p->asa, out, out_stride, residual, alpha, M, N,
                       K / kW4Group, w.G / kW4Group);
    return true;
}

// CSR-order prefill expert GEMMs: the activation is already what quantize()
// would have produced (per-32-group int8, f16 scales, f16 xs*sa), and the
// caller's rows are the GEMM's rows in order - so every expert tile binds the
// slices directly.  The per-tile gather kernel this replaces was ~20 us of a
// ~39 us call on the 35B A3B.
bool dnnl_gemm::gemm_w4_csr(const void * key, const int8_t * xq, const uint16_t * asa, const float * xs,
                            const uint16_t * corr, int M, int K, float * out, int out_stride) {
    if (!key || !xq || !asa || !xs || M <= 0 || K <= 0 || M > p->cap_M) {
        return false;
    }
    auto it = p->w4weights.find(key);
    if (it == p->w4weights.end() || !it->second.ok) {
        return false;
    }
    impl::w4_entry & w = it->second;
    if (w.K != K) {
        return false;
    }
    auto pit = p->prims4.find(impl::pkey(M, K, w.N));
    if (pit == p->prims4.end()) {
        impl::prim4_entry * e = nullptr;
        try {
            e = p->make_prim4(M, K, w.N);
        } catch (const std::exception &) {
            return false;
        }
        if (!e) {
            return false;
        }
        pit = p->prims4.find(impl::pkey(M, K, w.N));
    }
    impl::prim4_entry & pe = pit->second;
    const int N = w.N;
    const bool direct = out != nullptr && out_stride == N && ((uintptr_t)out % 64) == 0;
    memory src = sycl_interop::make_memory(pe.src.get_desc(), p->eng, sycl_interop::memory_kind::usm, (void *)xq);
    memory ssc = sycl_interop::make_memory(pe.sscales.get_desc(), p->eng, sycl_interop::memory_kind::usm, (void *)asa);
    memory dst_m = direct ? sycl_interop::make_memory(pe.dst.get_desc(), p->eng, sycl_interop::memory_kind::usm,
                                                      (void *)out)
                          : pe.dst;
    dnnl_capture_guard();
    pe.prim.execute(p->st, {{DNNL_ARG_SRC, src},
                            {DNNL_ARG_WEIGHTS, w.wmem},
                            {DNNL_ARG_DST, dst_m},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, w.scmem},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, ssc}});
    if (corr && pe.cok) {
        memory cs = sycl_interop::make_memory(pe.csrc.get_desc(), p->eng, sycl_interop::memory_kind::usm, (void *)corr);
        pe.cprim.execute(p->st, {{DNNL_ARG_SRC, cs}, {DNNL_ARG_WEIGHTS, w.offmem}, {DNNL_ARG_DST, dst_m}});
        if (!direct) {
            epilogue_f32_launch(p->q, p->accf, out, out_stride, nullptr, 1.0f, M, N);
        }
        return true;
    }
    w4_epilogue_launch(p->q, direct ? out : p->accf, xs, corr ? w.off : nullptr, asa, out, out_stride, nullptr, 1.0f,
                       M, N, K / kW4Group, w.G / kW4Group);
    return true;
}

bool dnnl_gemm::gemm_i8_csr(const void * key, const int8_t * xq, const uint16_t * asa, const float * xs,
                            const uint16_t * corr, int M, int K, float * out, int out_stride) {
    if (!key || !xq || !asa || !xs || M <= 0 || K <= 0 || M > p->cap_M) {
        return false;
    }
    auto it = p->weights.find(key);
    if (it == p->weights.end() || !it->second.ok) {
        return false;
    }
    impl::w_entry & w = it->second;
    if (w.K != K) {
        return false;
    }
    auto pit = p->prims.find(impl::pkey(M, K, w.N));
    if (pit == p->prims.end()) {
        impl::prim_entry * e = nullptr;
        try {
            e = p->make_prim(M, K, w.N);
        } catch (const std::exception &) {
            return false;
        }
        if (!e) {
            return false;
        }
        pit = p->prims.find(impl::pkey(M, K, w.N));
    }
    impl::prim_entry & pe = pit->second;
    const bool direct = out != nullptr && out_stride == w.N && ((uintptr_t)out % 64) == 0;
    memory src = sycl_interop::make_memory(pe.src.get_desc(), p->eng, sycl_interop::memory_kind::usm, (void *)xq);
    memory ssc = sycl_interop::make_memory(pe.sscales.get_desc(), p->eng, sycl_interop::memory_kind::usm, (void *)asa);
    memory dst_m = direct ? sycl_interop::make_memory(pe.dst.get_desc(), p->eng, sycl_interop::memory_kind::usm,
                                                      (void *)out)
                          : pe.dst;
    dnnl_capture_guard();
    pe.prim.execute(p->st, {{DNNL_ARG_SRC, src},
                            {DNNL_ARG_WEIGHTS, w.wmem},
                            {DNNL_ARG_DST, dst_m},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, w.scmem},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, ssc}});
    if (w.off) {
        if (!pe.cok && !pe.cfails) {
            try {
                const int ng32 = K / kW4Group;
                auto xc = memory::desc({M, ng32}, memory::data_type::f16, memory::format_tag::ab);
                auto wc = memory::desc({ng32, w.N}, memory::data_type::f16, memory::format_tag::ab);
                primitive_attr cattr;
                post_ops cop;
                cop.append_sum(1.0f);
                cattr.set_post_ops(cop);
                pe.cprim = matmul(matmul::primitive_desc(p->eng, xc, wc, pe.dst.get_desc(), cattr));
                pe.csrc = sycl_interop::make_memory(xc, p->eng, sycl_interop::memory_kind::usm, (void *)p->corrsrc);
                pe.cok = true;
            } catch (const std::exception &) {
                pe.cfails = true;
            }
        }
        if (pe.cok && w.offmem_ok && corr) {
            memory cs = sycl_interop::make_memory(pe.csrc.get_desc(), p->eng, sycl_interop::memory_kind::usm,
                                                  (void *)corr);
            pe.cprim.execute(p->st, {{DNNL_ARG_SRC, cs}, {DNNL_ARG_WEIGHTS, w.offmem}, {DNNL_ARG_DST, dst_m}});
            if (!direct) {
                epilogue_f32_launch(p->q, p->accf, out, out_stride, nullptr, 1.0f, M, w.N);
            }
            return true;
        }
        static bool once_fail = true;
        if (once_fail) {
            once_fail = false;
            fprintf(stderr, "[dnnl] i8 csr correction unavailable M=%d K=%d N=%d cok=%d csrc=%d offmem=%d\n", M, K,
                    w.N, (int)pe.cok, (int)(corr != nullptr), (int)w.offmem_ok);
        }
        w4_epilogue_launch(p->q, direct ? out : p->accf, xs, w.off, asa, out, out_stride, nullptr, 1.0f, M, w.N, w.ng,
                           1);
        return true;
    }
    if (!direct) {
        epilogue_f32_launch(p->q, p->accf, out, out_stride, nullptr, 1.0f, M, w.N);
    }
    return true;
}

void dnnl_gemm::warm_i8(int K, int N, const int * Ms, int n) {
    for (int i = 0; i < n; i++) {
        const int M = Ms[i];
        if (M <= 0 || M > p->cap_M || (size_t)M * (size_t)N > p->acc_cap) {
            continue;
        }
        impl::prim_entry * pe = nullptr;
        try {
            pe = p->make_prim(M, K, N);
        } catch (const std::exception &) {
            continue;
        }
        if (!pe || pe->cok || pe->cfails) {
            continue;
        }
        try {
            const int ng32 = K / kW4Group;
            auto xc = memory::desc({M, ng32}, memory::data_type::f16, memory::format_tag::ab);
            auto wc = memory::desc({ng32, N}, memory::data_type::f16, memory::format_tag::ab);
            primitive_attr cattr;
            post_ops cop;
            cop.append_sum(1.0f);
            cattr.set_post_ops(cop);
            pe->cprim = matmul(matmul::primitive_desc(p->eng, xc, wc, pe->dst.get_desc(), cattr));
            pe->csrc = sycl_interop::make_memory(xc, p->eng, sycl_interop::memory_kind::usm, (void *)p->corrsrc);
            pe->cok = true;
        } catch (const std::exception &) {
            pe->cfails = true;
        }
    }
}

bool dnnl_gemm::gemm_i8_csr_nocorr(const void * key, const int8_t * xq, const uint16_t * asa, int M, int K,
                                    float * out, int out_stride) {
    if (!key || !xq || !asa || M <= 0 || K <= 0 || M > p->cap_M || out == nullptr) {
        return false;
    }
    auto it = p->weights.find(key);
    if (it == p->weights.end() || !it->second.ok) {
        return false;
    }
    impl::w_entry & w = it->second;
    if (w.K != K || out_stride != w.N || ((uintptr_t)out % 64) != 0) {
        return false;
    }
    auto pit = p->prims.find(impl::pkey(M, K, w.N));
    if (pit == p->prims.end()) {
        impl::prim_entry * e = nullptr;
        try {
            e = p->make_prim(M, K, w.N);
        } catch (const std::exception &) {
            return false;
        }
        if (!e) {
            return false;
        }
        pit = p->prims.find(impl::pkey(M, K, w.N));
    }
    impl::prim_entry & pe = pit->second;
    memory src = sycl_interop::make_memory(pe.src.get_desc(), p->eng, sycl_interop::memory_kind::usm, (void *)xq);
    memory ssc = sycl_interop::make_memory(pe.sscales.get_desc(), p->eng, sycl_interop::memory_kind::usm, (void *)asa);
    memory dst = sycl_interop::make_memory(pe.dst.get_desc(), p->eng, sycl_interop::memory_kind::usm, (void *)out);
    dnnl_capture_guard();
    pe.prim.execute(p->st, {{DNNL_ARG_SRC, src},
                            {DNNL_ARG_WEIGHTS, w.wmem},
                            {DNNL_ARG_DST, dst},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, w.scmem},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, ssc}});
    return true;
}

void dnnl_gemm::sync() {
    p->q.wait();
    p->st.wait();
}

bool dnnl_gemm::gemm_i8_quant_group(const float * x, const float * up, const void * const * keys, const int * n_rows,
                                   const float * const * outs, const float * const * residuals, const float * alphas,
                                   int n_segs, int K) {
    if (!x || n_segs < 1 || n_segs > 4 || K <= 0 || K % 32) {
        return false;
    }
    const int ng = K / 32;
    // SLM: the quantized activation + its f16 scales + group sums + RB rows of
    // weight scales, with the kernel's fixed RB = 16.
    const size_t slm = (size_t)K + (size_t)ng * 6 + (size_t)16 * (size_t)ng * 2;
    // Half the per-work-group budget: the tile must not cost a resident
    // work-group (the 896/4864-wide decoders use 2-11 KB; a 17k-wide K would
    // take 38 KB).  The shape that misses this keeps the separate quantizer.
    if (slm > (size_t)si::dev::for_queue(p->q).slm.budget_bytes / 2) {
        return false;
    }
    i8_grp_seg segs[4];
    int total = 0;
    for (int i = 0; i < n_segs; i++) {
        auto it = p->weights.find(keys[i]);
        if (it == p->weights.end() || !it->second.ok || !it->second.dev || !it->second.scales) {
            return false;
        }
        if (it->second.K != K || n_rows[i] <= 0) {
            return false;
        }
        total += n_rows[i];
        segs[i].w8 = it->second.dev;
        segs[i].wsc = it->second.scales;
        segs[i].n_rows = n_rows[i];
        segs[i].out = (float *)outs[i];
        segs[i].out_stride = n_rows[i];
        segs[i].alpha = alphas[i];
        segs[i].residual = residuals[i];
    }
    i8_grp_quant_gemv_launch(p->q, segs, n_segs, total, x, up, K);
    return true;
}

bool dnnl_gemm::gemm_w4_quant_group(const float * x, const float * up, const void * const * keys, const int * n_rows,
                                    const float * const * outs, const float * const * residuals,
                                    const float * alphas, int n_segs, int K) {
    if (!x || n_segs < 1 || n_segs > 4 || K <= 0 || K % 32) {
        return false;
    }
    const int G = si::w4_group();
    w4_grp_seg segs[4];
    int total = 0;
    for (int i = 0; i < n_segs; i++) {
        auto it = p->w4weights.find(keys[i]);
        if (it == p->w4weights.end() || !it->second.ok) {
            return false;
        }
        const impl::w4_entry & w = it->second;
        // the packed planes must match this call's K and the global group width
        // the fused kernel's staging index uses
        if (w.K != K || w.G != G || !w.vals || !w.scales || !w.off || n_rows[i] <= 0) {
            return false;
        }
        total += n_rows[i];
        segs[i].vals = w.vals;
        segs[i].scale = w.scales;
        segs[i].off = w.off;
        segs[i].n_rows = n_rows[i];
        segs[i].out = (float *)outs[i];
        segs[i].alpha = alphas[i];
        segs[i].residual = residuals[i];
    }
    return w4_grp_quant_gemv_launch(p->q, segs, n_segs, total, x, up, K);
}

const int8_t * dnnl_gemm::weight_data(const void * key) const {
    auto it = p->weights.find(key);
    return it == p->weights.end() ? nullptr : it->second.dev;
}

// Narrow-int8 group fusion.  kNarrowRows is where the per-launch latency stops
// mattering: at 27B sizes the GEMV itself (nat_gemm's per-call staging and
// dp4a) is what costs, and the batched path is better above it.
static const int kNarrowRows = 2048;

bool dnnl_gemm::gemm_i8_group(const act_view & v, const void * const * keys, const int * n_rows, const float * const * outs,
                              int out_stride, const float * const * residuals, const float * alphas, int n_segs,
                              int M, int K) {
    if (n_segs < 1 || n_segs > 4 || !view_is_current(v) || M != v.m || K != v.k || M < 1 || M > 13 || K <= 0 || K % 32) {
        return false;
    }
    i8_grp_seg segs[4];
    int total = 0;
    for (int i = 0; i < n_segs; i++) {
        auto it = p->weights.find(keys[i]);
        if (it == p->weights.end() || !it->second.ok || !it->second.dev || !it->second.scales) {
            return false; // not a grouped int8 weight
        }
        if (it->second.K != K) {
            return false;
        }
        if (n_rows[i] <= 0) {
            return false;
        }
        total += n_rows[i];
        if (total > kNarrowRows) {
            return false;
        }
        segs[i].w8 = it->second.dev;
        segs[i].wsc = it->second.scales;
        segs[i].n_rows = n_rows[i];
        segs[i].out = (float *)outs[i];
        segs[i].out_stride = out_stride;
        segs[i].alpha = alphas[i];
        segs[i].residual = residuals[i];
    }
    i8_grp_gemv_rows_multi_launch(p->q, segs, n_segs, total, p->axg, p->asa, p->xs, M, K);
    return true;
}

const uint16_t * dnnl_gemm::weight_group_scales(const void * key) const {
    auto it = p->weights.find(key);
    return it == p->weights.end() ? nullptr : it->second.scales;
}

const float * dnnl_gemm::weight_scales(const void * key) const {
    // The int8 weights now carry a per-32-group f16 scale plane (consumed by
    // oneDNN and by i8_grp_gemv_launch), not a per-row float array.  Returning
    // null makes the engine's decode fall through to dnnl_gemm::gemm, which
    // runs the grouped GEMV for M == 1.
    (void)key;
    return nullptr;
}

const int8_t * dnnl_gemm::act_grp_data(const act_view & v) const {
    return view_is_current(v) ? p->axg : nullptr;
}

bool dnnl_gemm::act_split(const act_view & v, const int8_t ** even, const int8_t ** odd) const {
    if (!view_is_current(v) || !v.split || !p->axe || !p->axo) {
        return false;
    }
    if (even) {
        *even = p->axe;
    }
    if (odd) {
        *odd = p->axo;
    }
    return true;
}

const uint16_t * dnnl_gemm::act_grp_scales(const act_view & v) const {
    return view_is_current(v) ? p->asa : nullptr;
}

const int8_t * dnnl_gemm::act_data(const act_view & v) const {
    return view_is_current(v) ? p->ax : nullptr;
}

const float * dnnl_gemm::act_group_sums(const act_view & v) const {
    return view_is_current(v) ? p->xs : nullptr;
}

const float * dnnl_gemm::act_scales(const act_view & v) const {
    return view_is_current(v) ? p->axs : nullptr;
}

const int32_t * dnnl_gemm::act_sum(const act_view & v) const {
    return view_is_current(v) ? p->axsum : nullptr;
}

// ---------------------------------------------------------------------------
// XMX attention GEMMs.  Operands and dst are the caller's USM buffers; only
// the primitive is cached.
// The attention matmuls run on `st`, the same oneDNN stream every dense GEMM and
// every SYCL kernel of the forward uses, so the following softmax is ordered after
// the matmul with no explicit wait and no queue drain.  They used to run on a
// second stream on the theory that a stream built from the same in-order queue is
// ordered against it; it is not, and that was a live data race - see the note on
// `st_a` in impl.  PF_ATTN_STREAM=1 reproduces it; PF_ATTN_WAIT=1 adds a blocking
// per-matmul wait for A/B.
static bool attn_wait_env() {
    static const bool v = [] {
        const char * e = si::env::str("PF_ATTN_WAIT");
        return e && atoi(e) != 0;
    }();
    return v;
}



bool dnnl_gemm::attn_qk(int M, int blk, const int8_t * q_kmajor, const int8_t * k_keymajor, int32_t * dst) {
    const auto * e = p->attn_qk_prim(M, /*K=*/256, blk);
    if (!e->ok) {
        return false;
    }
    // On `st` - the same stream every dense GEMM and every SYCL kernel uses - so
    // the matmul is ordered after the Q-quantize / K-gather that produced `k` and
    // before the softmax that reads `dst`, with no host-side wait and no queue
    // drain.  The separate `st_a` this used was NOT ordered against the main
    // stream despite wrapping the same queue; see the note on `st_a` in impl.
    stream & ast = p->attn_sep ? p->st_a : p->st;
    auto xm = sycl_interop::make_memory(memory::desc({M, 256}, memory::data_type::s8, memory::format_tag::ab), p->eng,
                                        sycl_interop::memory_kind::usm, (void *)q_kmajor);
    auto wm = sycl_interop::make_memory(memory::desc({256, blk}, memory::data_type::s8, memory::format_tag::ba), p->eng,
                                        sycl_interop::memory_kind::usm, (void *)k_keymajor);
    auto dm = sycl_interop::make_memory(memory::desc({M, blk}, memory::data_type::s32, memory::format_tag::ab), p->eng,
                                        sycl_interop::memory_kind::usm, (void *)dst);
    dnnl_capture_guard();
    if (dnnl_qwait_env()) {
        p->q.wait();
    }
    e->prim.execute(ast, {{DNNL_ARG_SRC, xm}, {DNNL_ARG_WEIGHTS, wm}, {DNNL_ARG_DST, dm}});
    if (dnnl_qwait_env()) {
        p->q.wait();
        // ALSO the stream's own wait.  A queue wait only orders against work
        // submitted to the queue, so if `p->st` is a distinct ordering domain
        // (it is an interop stream built from the queue) then waiting on `p->q`
        // does not wait for the primitive at all, and the whole ordering
        // hypothesis would have been tested against the wrong domain.  PF_ATTN_WAIT
        // did use the stream's wait, but only on the two attention primitives and
        // only under the perturbing fingerprint oracle, which this whole entry
        // shows is not a valid arbiter.  Both domains, one arm.
        p->st.wait();
    }
    if (dnnl_qwait_env()) {
        // The OTHER direction.  The wait above orders the gather -> matmul edge;
        // this one orders the matmul -> consumer edge, which is the one nothing
        // covers: `xmx_softmax` reads b.qk and `xmx_pv_acc` reads b.pv, both
        // submitted to `p->q`, right after these two primitives ran.  If the
        // interop stream and the queue are distinct ordering domains then a wait
        // in front of the matmul fixes nothing, because it is the consumer that
        // can run early.  Cheap to ask.
        p->q.wait();
    }
    if (attn_wait_env()) {
        ast.wait();
    }
    return true;
}

bool dnnl_gemm::attn_pv(int M, int blk, const uint8_t * p_major, const int8_t * v_keymajor, int32_t * dst) {
    const auto * e = p->attn_pv_prim(M, blk, /*N=*/256);
    if (!e->ok) {
        return false;
    }
    auto xm = sycl_interop::make_memory(memory::desc({M, blk}, memory::data_type::u8, memory::format_tag::ab), p->eng,
                                        sycl_interop::memory_kind::usm, (void *)p_major);
    auto wm = sycl_interop::make_memory(memory::desc({blk, 256}, memory::data_type::s8, memory::format_tag::ab), p->eng,
                                        sycl_interop::memory_kind::usm, (void *)v_keymajor);
    auto dm = sycl_interop::make_memory(memory::desc({M, 256}, memory::data_type::s32, memory::format_tag::ab), p->eng,
                                        sycl_interop::memory_kind::usm, (void *)dst);
    // same stream choice as attn_qk, and for the same reason - see impl's note on
    // `st_a`.  PV reads the `b.v` the gather wrote, and the next block chunk's
    // gather overwrites it, so an unordered stream here is a live hazard.
    stream & ast = p->attn_sep ? p->st_a : p->st;
    dnnl_capture_guard();
    if (dnnl_qwait_env()) {
        p->q.wait();
    }
    e->prim.execute(ast, {{DNNL_ARG_SRC, xm}, {DNNL_ARG_WEIGHTS, wm}, {DNNL_ARG_DST, dm}});
    if (dnnl_qwait_env()) {
        p->q.wait();
        // ALSO the stream's own wait.  A queue wait only orders against work
        // submitted to the queue, so if `p->st` is a distinct ordering domain
        // (it is an interop stream built from the queue) then waiting on `p->q`
        // does not wait for the primitive at all, and the whole ordering
        // hypothesis would have been tested against the wrong domain.  PF_ATTN_WAIT
        // did use the stream's wait, but only on the two attention primitives and
        // only under the perturbing fingerprint oracle, which this whole entry
        // shows is not a valid arbiter.  Both domains, one arm.
        p->st.wait();
    }
    if (dnnl_qwait_env()) {
        // The OTHER direction.  The wait above orders the gather -> matmul edge;
        // this one orders the matmul -> consumer edge, which is the one nothing
        // covers: `xmx_softmax` reads b.qk and `xmx_pv_acc` reads b.pv, both
        // submitted to `p->q`, right after these two primitives ran.  If the
        // interop stream and the queue are distinct ordering domains then a wait
        // in front of the matmul fixes nothing, because it is the consumer that
        // can run early.  Cheap to ask.
        p->q.wait();
    }
    if (attn_wait_env()) {
        ast.wait();
    }
    return true;
}

// ---------------------------------------------------------------------------
// Queue -> dnnl_gemm registry (see dnnl_gemm.h).  A small vector with a mutex;
// engines are constructed one at a time and the lookups are on the attention
// hot path, so a linear scan of a handful of entries is fine.
namespace {
struct qreg {
    std::mutex mtx;
    std::vector<std::pair<sycl::queue *, dnnl_gemm *>> v;
};
qreg & qregistry() {
    static qreg r;
    return r;
}
} // namespace

void dnnl_register_queue(sycl::queue & q, dnnl_gemm * g) {
    qreg & r = qregistry();
    std::lock_guard<std::mutex> lk(r.mtx);
    for (auto & e : r.v) {
        if (e.first == &q) {
            e.second = g;
            return;
        }
    }
    r.v.emplace_back(&q, g);
}

dnnl_gemm * dnnl_for_queue(sycl::queue & q) {
    qreg & r = qregistry();
    std::lock_guard<std::mutex> lk(r.mtx);
    for (auto & e : r.v) {
        if (e.first == &q) {
            return e.second;
        }
    }
    return nullptr;
}


} // namespace si
