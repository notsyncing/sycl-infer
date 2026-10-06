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
                          uint16_t * sasc, float * xs, int8_t * axe, int8_t * axo, bool do_split, int M, int K) {
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
        }
        if (xs) {
            // a sub-group collective, so every lane must execute it (a divergent
            // call is UB and produced garbage xs - the u4 correction term)
            const int s = sycl::reduce_over_group(it.get_sub_group(), (int)q8, sycl::plus<int>());
            if (lane == 0) {
                xs[(size_t)m * ng + g] = (float)s;
            }
        }
    });
}

static inline float w4_f16_bits(uint16_t v) {
    sycl::half h;
    __builtin_memcpy((void *)&h, &v, sizeof(h));
    return (float)h;
}

// out[m][n] = alpha*sx[m]*(acc4[m][n] + sum_g off[g][n]*xs[m][g]) + residual
// acc4 is already the step-scaled sum (oneDNN applied the grouped f16 scales).
// Register-tiled TM x TN per thread over a 64 x 64 workgroup tile: the offset
// plane is then read only M/64 times (a naive per-(m,n) loop over g rereads it
// once per output row and was ~45x too slow), and each xs value is reused TN
// times / each offset value TM times.
void w4_epilogue_launch(sycl::queue & q, const float * acc4, const float * xs, const uint16_t * off,
                        const uint16_t * sa, float * out, int out_stride, const float * residual, float alpha, int M,
                        int N, int NG) {
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
                    wv[b] = (n + b < N) ? w4_f16_bits(off[(size_t)g * N + n + b]) : 0.f;
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

void epilogue_scalar_launch(sycl::queue & q, const int32_t * acc, const float * sx, const float * sw, float * out,
                            int out_stride, const float * residual, float alpha, int M, int N) {
    q.parallel_for(sycl::range<1>((size_t)M * N), [=](sycl::id<1> i) {
        const int m = (int)(i / N);
        const int n = (int)(i % N);
        const size_t o = (size_t)m * out_stride + n;
        const float v = alpha * sx[m] * sw[n] * (float)acc[(size_t)m * N + n] + (residual ? residual[o] : 0.f);
        out[o] = v;
    });
}

} // namespace

static bool g_dnnl_capturing = false;
void dnnl_set_capturing(bool on) {
    g_dnnl_capturing = on;
}
void dnnl_capture_guard() {
    if (g_dnnl_capturing) {
        throw std::runtime_error("oneDNN primitive reached during SYCL graph capture");
    }
}

bool dnnl_gemm_enabled() {
    // default ON (the mode-2 int8 GEMMs run on oneDNN); PF_GEMM_DNNL=0 forces
    // the dp4a chunk-batched path, e.g. for bit-exact dp4a validation
    static const bool on = [] {
        const char * e = getenv("PF_GEMM_DNNL");
        return !(e && atoi(e) == 0) && si::dev::active().wt.gemm_dnnl;
    }();
    return on;
}

// ---------------------------------------------------------------------------
struct dnnl_gemm::impl {
    sycl::queue q;
    engine eng;
    stream st;
    // Separate stream for the XMX attention matmuls.  Both streams wrap the
    // same in-order SYCL queue, so submissions stay ordered, but waiting on
    // this stream no longer drains the dense GEMMs queued on `st` (which is
    // what made attention syncs cost tens of ms and serialised prefill).
    stream st_a;
    // activation scratch: row-major [M][K] int8 + one scale per row
    int8_t * ax = nullptr;
    float * axs = nullptr;
    int32_t * axsum = nullptr;
    int cap_M = 0, cap_K = 0; // scratch capacity
    int cur_M = 0, cur_K = 0; // currently quantized activations
    bool acts_valid = false;
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
    int8_t * axg = nullptr;    // per-32-group quantized activations [M][K]
    uint16_t * asa = nullptr;  // their per-(m,g) f16 scales, [M][K/32]
    int8_t * axe = nullptr;    // even/odd k split of axg, for the u4 decode GEMV
    int8_t * axo = nullptr;
    bool has_w4 = false;       // any u4 weight registered -> produce axg/asa

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
        bool ok = false;
    };
    std::unordered_map<const void *, w_entry> weights;

    // 4-bit weight: u4 values [N][K] (K inner, low nibble first) plus the
    // per-32-group f16 step plane (oneDNN grouped scales) and f16 offset plane
    // (the correction coefficient).  See common/w4.h.
    struct w4_entry {
        int K = 0, N = 0;
        uint8_t * vals = nullptr;
        uint16_t * scales = nullptr;
        uint16_t * off = nullptr;
        memory wmem, scmem;
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

    struct prim4_entry {
        matmul prim;
        memory src, dst, sscales;
    };
    std::unordered_map<uint64_t, prim4_entry> prims4;

    struct prim_entry {
        matmul prim;
        memory src, dst, sscales; // sscales wraps p->asa (grouped SRC scales)
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
        // mask = dim 0 (K) zoomed by the 32-value group; layouts validated on
        // this GPU (dims {1, NG*N} f16, exact against the integer reference)
        attr.set_scales(DNNL_ARG_WEIGHTS, 1, {kW4Group, 1}, memory::data_type::f16);
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
            const char * e = getenv("PF_DNNL_BLOCKED");
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
    p->cap_M = kMaxB * kMaxT; // max rows of one chunk-batched prefill
    p->cap_K = kActMaxK;
    p->ax = sycl::malloc_device<int8_t>((size_t)p->cap_M * p->cap_K, q);
    p->axs = sycl::malloc_device<float>(p->cap_M, q);
    p->axsum = sycl::malloc_device<int32_t>(p->cap_M, q);
    // widest N any converted tensor has (27B ffn_gate/up = 17408; rounded to a
    // power of two), so every (M,N) up to a full kMaxB*kMaxT prefill tile can
    // create its primitive
    p->acc_cap = (size_t)p->cap_M * 32768;
    p->acc = sycl::malloc_device<int32_t>(p->acc_cap, q);
    // 4-bit path scratch: f32 accumulator over the same [M][N] range plus the
    // per-group activation sums (K/32 per row, at most 1024)
    p->accf = sycl::malloc_device<float>(p->acc_cap, q);
    p->xs = sycl::malloc_device<float>((size_t)p->cap_M * (p->cap_K / kW4Group), q);
    p->axg = sycl::malloc_device<int8_t>((size_t)p->cap_M * p->cap_K, q);
    p->asa = sycl::malloc_device<uint16_t>((size_t)p->cap_M * (p->cap_K / kW4Group), q);
    p->axe = sycl::malloc_device<int8_t>((size_t)p->cap_M * p->cap_K / 2, q);
    p->axo = sycl::malloc_device<int8_t>((size_t)p->cap_M * p->cap_K / 2, q);
    if (!p->ax || !p->axs || !p->acc || !p->accf || !p->xs || !p->axg || !p->asa || !p->axe || !p->axo) {
        throw std::runtime_error("dnnl scratch alloc failed");
    }
}

dnnl_gemm::~dnnl_gemm() {
    for (auto & it : p->weights) {
        if (it.second.dev) {
            sycl::free(it.second.dev, p->q);
        }
        if (it.second.scales) {
            sycl::free(it.second.scales, p->q);
        }
    }
    if (p->ax) {
        sycl::free(p->ax, p->q);
    }
    if (p->axs) {
        sycl::free(p->axs, p->q);
    }
    if (p->axsum) {
        sycl::free(p->axsum, p->q);
    }
    if (p->acc) {
        sycl::free(p->acc, p->q);
    }
    if (p->accf) {
        sycl::free(p->accf, p->q);
    }
    if (p->xs) {
        sycl::free(p->xs, p->q);
    }
    if (p->axg) {
        sycl::free(p->axg, p->q);
    }
    if (p->asa) {
        sycl::free(p->asa, p->q);
    }
    if (p->axe) {
        sycl::free(p->axe, p->q);
    }
    if (p->axo) {
        sycl::free(p->axo, p->q);
    }
    for (auto * wm : {&p->w4weights, &p->w2weights}) {
        for (auto & it : *wm) {
            if (it.second.vals) {
                sycl::free(it.second.vals, p->q);
            }
            if (it.second.scales) {
                sycl::free(it.second.scales, p->q);
            }
            if (it.second.off) {
                sycl::free(it.second.off, p->q);
            }
        }
    }
    for (auto & it : p->k5weights) {
        if (it.second.vals) {
            sycl::free(it.second.vals, p->q);
        }
        if (it.second.hi) {
            sycl::free(it.second.hi, p->q);
        }
        if (it.second.scales) {
            sycl::free(it.second.scales, p->q);
        }
        if (it.second.off) {
            sycl::free(it.second.off, p->q);
        }
    }
}

// Host-side conversion of one tensor: GGUF K-quant rows -> int8 + one
// symmetric scale per output row.  The dequantized values are exactly what the
// SIn layout holds (the packed values are copied verbatim from the GGUF file),
// so the values are the ones the dp4a path uses; only the representation
// changes (per-row int8 instead of per-32 packed 4/5/6-bit).  Rows are
// independent -> convert in parallel with one thread per row slab.
bool dnnl_gemm::add_weight(const void * key, const void * host_data, uint32_t ggml_type, int K, int N) {
    if (!key || !host_data) {
        return false;
    }
    // any quantized format quant.h can dequantize row-wise (the K-quants, the
    // IQ codebook formats and Q8_0) converts to an int8 oneDNN weight
    switch (ggml_type) {
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

    const size_t nvals = (size_t)N * K;
    std::vector<int8_t> hw(nvals);
    const int ng = K / kW4Group;
    std::vector<uint16_t> hs((size_t)ng * N); // [g][n] f16 step
    const size_t row_bytes = quant_row_bytes(ggml_type, K);
    const int nthr = (int)std::min<size_t>((size_t)N, std::max<size_t>(1, std::thread::hardware_concurrency()));
    std::vector<std::vector<float>> frows(nthr, std::vector<float>(K));
    std::vector<std::thread> th;
    th.reserve(nthr);
    for (int t = 0; t < nthr; t++) {
        th.emplace_back([&, t] {
            std::vector<float> & fr = frows[t];
            for (int r = t; r < N; r += nthr) {
                dequantize_row(ggml_type, (const char *)host_data + (size_t)r * row_bytes, fr.data(), K);
                int8_t * q8 = hw.data() + (size_t)r * K;
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
                    for (int k = g * kW4Group; k < (g + 1) * kW4Group; k++) {
                        int v = (int)std::lround(fr[k] * inv);
                        q8[k] = (int8_t)std::max(-127, std::min(127, v));
                    }
                }
            }
        });
    }
    for (auto & x : th) {
        x.join();
    }

    impl::w_entry w;
    w.K = K;
    w.N = N;
    w.ng = ng;
    w.dev = sycl::malloc_device<int8_t>(nvals, p->q);
    w.scales = sycl::malloc_device<uint16_t>((size_t)ng * N, p->q);
    if (!w.dev || !w.scales) {
        if (w.dev) {
            sycl::free(w.dev, p->q);
        }
        if (w.scales) {
            sycl::free(w.scales, p->q);
        }
        return false;
    }
    p->q.memcpy(w.dev, hw.data(), nvals).wait();
    p->q.memcpy(w.scales, hs.data(), (size_t)ng * N * 2).wait();
    static const int blocked = [] {
        const char * e = getenv("PF_DNNL_BLOCKED");
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
    bool any = false;
    for (int M = kMaxT; M <= p->cap_M; M += kMaxT) {
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
    w.ok = any;
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
        n += (size_t)kv.second.K * kv.second.N + (size_t)kv.second.ng * kv.second.N * 2;
    }
    for (const auto & kv : p->w4weights) {
        n += (size_t)kv.second.K * kv.second.N / 2 + (size_t)(kv.second.K / kW4Group) * kv.second.N * 4;
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
    e.vals = sycl::malloc_device<uint8_t>(b.vals.size(), p->q);
    e.hi = sycl::malloc_device<uint8_t>(b.hi.size(), p->q);
    e.scales = sycl::malloc_device<uint16_t>(b.scale.size(), p->q);
    e.off = sycl::malloc_device<uint16_t>(b.off.size(), p->q);
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
        p->bit_lut = sycl::malloc_device<uint32_t>(16, p->q);
        if (!p->bit_lut) {
            return false;
        }
        p->q.memcpy(p->bit_lut, host, sizeof(host)).wait();
    }
    // the prefill expansion needs the int8-sized scratch (shared with cb4)
    const size_t need = (size_t)K * (size_t)N;
    if (need > p->cb4_scratch_cap) {
        if (p->cb4_scratch) {
            sycl::free(p->cb4_scratch, p->q);
            p->cb4_scratch = nullptr;
            p->cb4_scratch_cap = 0;
        }
        p->cb4_scratch = sycl::malloc_device<int8_t>(need, p->q);
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
    e.idx = sycl::malloc_device<uint8_t>(b.idx.size(), p->q);
    e.scales = sycl::malloc_device<uint16_t>(b.scale.size(), p->q);
    if (!e.idx || !e.scales) {
        if (e.idx) {
            sycl::free(e.idx, p->q);
        }
        if (e.scales) {
            sycl::free(e.scales, p->q);
        }
        return false;
    }
    p->q.memcpy(e.idx, b.idx.data(), b.idx.size()).wait();
    p->q.memcpy(e.scales, b.scale.data(), b.scale.size() * sizeof(uint16_t)).wait();
    if (!p->lut) {
        p->lut = sycl::malloc_device<int8_t>(16, p->q);
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
        p->lut16 = sycl::malloc_device<uint16_t>(256, p->q);
        if (!p->lut16) {
            return false;
        }
        p->q.memcpy(p->lut16, h16, sizeof(h16)).wait();
    }
    // the prefill expansion needs the int8-sized scratch; grow it to the widest
    // converted tensor (the biggest is ffn_gate/up, ~95 MB, not the full model)
    const size_t need = (size_t)K * (size_t)N;
    if (need > p->cb4_scratch_cap) {
        if (p->cb4_scratch) {
            sycl::free(p->cb4_scratch, p->q);
            p->cb4_scratch = nullptr;
            p->cb4_scratch_cap = 0;
        }
        p->cb4_scratch = sycl::malloc_device<int8_t>(need, p->q);
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
                              bool any_type, bool gemv_only, float scale_mul) {
    const bool w4info = getenv("PF_W4_INFO") != nullptr;
    if (!key || !host_data || !(any_type || si::w4_supported(ggml_type))) {
        if (w4info) {
            fprintf(stderr, "[w4] reject key=%p: precheck type=%u any=%d\n", key, ggml_type, (int)any_type);
        }
        return false;
    }
    if (K <= 0 || N <= 0 || (K % kW4Group) != 0 || K > kActMaxK) {
        if (w4info) {
            fprintf(stderr, "[w4] reject key=%p: K=%d N=%d maxK=%d\n", key, K, N, (int)kActMaxK);
        }
        return false;
    }
    auto found = p->w4weights.find(key);
    if (found != p->w4weights.end()) {
        return found->second.ok;
    }
    si::w4t w;
    if (!(any_type ? si::w4_pack_any(ggml_type, host_data, K, N, w) : si::w4_pack(ggml_type, host_data, K, N, w))) {
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
    const int ng = K / kW4Group;
    impl::w4_entry e;
    e.K = K;
    e.N = N;
    if (w4info) {
        fprintf(stderr, "[w4] register key=%p type=%u K=%d N=%d (%.4f B/weight)\n", key, ggml_type, K, N,
                (double)(w.vals.size() + w.scale.size() * 2 + w.off.size() * 2) / ((double)K * N));
    }
    e.vals = sycl::malloc_device<uint8_t>(w.vals.size(), p->q);
    e.scales = sycl::malloc_device<uint16_t>(w.scale.size(), p->q);
    e.off = sycl::malloc_device<uint16_t>(w.off.size(), p->q);
    if (!e.vals || !e.scales || !e.off) {
        if (e.vals) {
            sycl::free(e.vals, p->q);
        }
        if (e.scales) {
            sycl::free(e.scales, p->q);
        }
        if (e.off) {
            sycl::free(e.off, p->q);
        }
        return false;
    }
    p->q.memcpy(e.vals, w.vals.data(), w.vals.size()).wait();
    p->q.memcpy(e.scales, w.scale.data(), w.scale.size() * 2).wait();
    p->q.memcpy(e.off, w.off.data(), w.off.size() * 2).wait();
    e.wmem = sycl_interop::make_memory(memory::desc({K, N}, memory::data_type::u4, memory::format_tag::ba), p->eng,
                                       sycl_interop::memory_kind::usm, (void *)e.vals);
    e.scmem =
        sycl_interop::make_memory(memory::desc({1, ng * N}, memory::data_type::f16, memory::format_tag::ab), p->eng,
                                  sycl_interop::memory_kind::usm, (void *)e.scales);

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
    bool any = any_type || gemv_only;
    const int ladder[] = {1, kMaxT, 2 * kMaxT, 3 * kMaxT, 4 * kMaxT, 8 * kMaxT, 16 * kMaxT};
    if (!(any_type || gemv_only)) {
        if ((size_t)p->cap_M * (size_t)N > p->acc_cap) {
            sycl::free(e.vals, p->q);
            sycl::free(e.scales, p->q);
            sycl::free(e.off, p->q);
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
    e.ok = any;
    if (any) {
        p->has_w4 = true; // quantize() must now also produce the grouped form
    }
    auto ins = p->w4weights.emplace(key, std::move(e));
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
    e.vals = sycl::malloc_device<uint8_t>(w.vals.size(), p->q);
    e.scales = sycl::malloc_device<uint16_t>(w.scale.size(), p->q);
    e.off = sycl::malloc_device<uint16_t>(w.off.size(), p->q);
    if (!e.vals || !e.scales || !e.off) {
        if (e.vals) {
            sycl::free(e.vals, p->q);
        }
        if (e.scales) {
            sycl::free(e.scales, p->q);
        }
        if (e.off) {
            sycl::free(e.off, p->q);
        }
        return false;
    }
    p->q.memcpy(e.vals, w.vals.data(), w.vals.size()).wait();
    p->q.memcpy(e.scales, w.scale.data(), w.scale.size() * 2).wait();
    p->q.memcpy(e.off, w.off.data(), w.off.size() * 2).wait();
    e.ok = true;
    if (getenv("PF_W2_INFO")) {
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
bool dnnl_gemm::gemm_w2(const void * key, const float * residual, float alpha, int M, int K, float * out,
                        int out_stride) {
    static const bool dbg = getenv("PF_W2_DEBUG") != nullptr;
    auto it = p->w2weights.find(key);
    if (it == p->w2weights.end() || !it->second.ok || M != 1 || it->second.K != K) {
        if (dbg) {
            fprintf(stderr, "[w2] miss: entry=%d ok=%d M=%d K=%d (entry K=%d)\n",
                    (int)(it != p->w2weights.end()), (int)(it != p->w2weights.end() && it->second.ok), M, K,
                    it == p->w2weights.end() ? -1 : it->second.K);
        }
        return false;
    }
    if (!p->acts_valid || M != p->cur_M || K != p->cur_K || !p->axg || !p->asa || !p->xs) {
        if (dbg) {
            fprintf(stderr, "[w2] miss: acts_valid=%d curM=%d/%d curK=%d/%d axg=%p asa=%p xs=%p\n",
                    (int)p->acts_valid, M, p->cur_M, K, p->cur_K, (void *)p->axg, (void *)p->asa, (void *)p->xs);
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
int dnnl_gemm::warmup() {
    const bool tdbg = getenv("PF_DNNL_TIME") != nullptr;
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
        try {
            const auto t1 = std::chrono::high_resolution_clock::now();
            // the grouped scales must be supplied now; dummy planes are fine
            uint16_t * dsc = sycl::malloc_device<uint16_t>((size_t)(K / kW4Group) * N, p->q);
            if (!dsc) {
                continue;
            }
            auto scmem = sycl_interop::make_memory(
                memory::desc({1, (memory::dim)((K / kW4Group) * N)}, memory::data_type::f16, memory::format_tag::ab),
                p->eng,
                sycl_interop::memory_kind::usm, (void *)dsc);
            dnnl_capture_guard();
            pe.prim.execute(p->st, {{DNNL_ARG_SRC, pe.src},
                                    {DNNL_ARG_WEIGHTS, *wmem},
                                    {DNNL_ARG_DST, pe.dst},
                                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scmem},
                                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, pe.sscales}});
            sycl::free(dsc, p->q);
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
        act_quant_grp_launch(p->q, dx, nullptr, WK, WK, p->axg, p->asa, p->xs, p->axe, p->axo, true, WM, WK);
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

bool dnnl_gemm::quantize(const float * x, const float * up, int x_stride, int up_stride, int M, int K,
                         bool do_split) {
    p->acts_valid = false;
    if (!x || M <= 0 || M > p->cap_M || K <= 0 || K > p->cap_K) {
        return false;
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
        const char * e = getenv("PF_ROWACT");
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
    act_quant_grp_launch(p->q, x, up, x_stride, up_stride, p->axg, p->asa, p->xs, p->axe, p->axo, do_split, M, K);
    p->cur_M = M;
    p->cur_K = K;
    p->acts_valid = true;
    p->split_valid = do_split;
    return true;
}

bool dnnl_gemm::gemm(const void * key, const float * residual, float alpha, int M, int K, float * out, int out_stride) {
    if (!p->acts_valid || M != p->cur_M || K != p->cur_K) {
        return false;
    }
    if (auto kit = p->k5weights.find(key); kit != p->k5weights.end() && kit->second.ok) {
        impl::k5_entry & e = kit->second;
        if (e.K != K || !p->acts_valid || M != p->cur_M || K != p->cur_K) {
            return false;
        }
        if (M == 1) {
            // decode: recombine the two 5-bit planes in-kernel (0.75 B/weight)
            k5_gemv_launch(p->q, e.vals, e.hi, e.scales, e.off, p->axe, p->axo, p->asa, p->xs, out, residual, alpha, K,
                           e.N);
            return true;
        }
        // batched verify: same native 5-bit stream (0.75 B/weight) for all M
        if (p->split_valid && p->axe && p->axo && p->axg && p->asa && p->xs && p->bit_lut
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
        pe->prim.execute(p->st, {{DNNL_ARG_SRC, pe->src},
                                 {DNNL_ARG_WEIGHTS, wmem},
                                 {DNNL_ARG_DST, pe->dst},
                                 {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scmem},
                                 {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, pe->sscales}});
        // PF_K5_NOCORR (diagnostic): drop the offset correction, keeping only the
        // step-scaled matmul - tells whether the prefill cost is the expansion
        // kernel or the u4-style correction epilogue.
        static const bool nocorr = [] {
            const char * ev = getenv("PF_K5_NOCORR");
            return ev && atoi(ev) != 0;
        }();
        w4_epilogue_launch(p->q, p->accf, p->xs, nocorr ? nullptr : e.off, p->asa, out, out_stride, residual, alpha, M,
                           e.N, e.ng);
        return true;
    }
    if (auto cit = p->cb4weights.find(key); cit != p->cb4weights.end() && cit->second.ok) {
        impl::cb4_entry & e = cit->second;
        if (e.K != K || !p->acts_valid || M != p->cur_M || K != p->cur_K) {
            return false;
        }
        if (M == 1) {
            // decode: expand the codebook in-kernel (half the weight bytes)
            cb4_gemv_launch(p->q, e.idx, p->lut16, e.scales, p->axg, p->asa, p->xs, out, residual, alpha, K, e.N);
            return true;
        }
        // batched verify: the same index plane + in-kernel LUT, no int8 scratch
        // (split_valid as above: the same activation form as the decode/verify)
        if (p->split_valid && p->axg && p->asa && p->xs && p->lut16
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
        pe->prim.execute(p->st, {{DNNL_ARG_SRC, pe->src},
                                 {DNNL_ARG_WEIGHTS, wmem},
                                 {DNNL_ARG_DST, pe->dst},
                                 {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scmem},
                                 {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, pe->sscales}});
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
    if (M >= 2 && p->split_valid && p->axg && p->asa && p->xs) {
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
            const char * e = getenv("PF_I8_NOGEMV");
            return e && atoi(e) != 0;
        }();
        if (!no_gemv) {
            i8_grp_gemv_launch(p->q, w.dev, w.scales, p->axg, p->asa, p->xs, out, residual, alpha, K, w.N);
            return true;
        }
    }
    impl::prim_entry & pe = pit->second;
    dnnl_capture_guard();
    pe.prim.execute(p->st, {{DNNL_ARG_SRC, pe.src},
                            {DNNL_ARG_WEIGHTS, w.wmem},
                            {DNNL_ARG_DST, pe.dst},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, w.scmem},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, pe.sscales}});
    // both the weight and the activation scales are applied by the matmul now
    epilogue_f32_launch(p->q, p->accf, out, out_stride, residual, alpha, M, w.N);
    return true;
}

// 4-bit prefill GEMM: out = alpha*sx[m]*(acc4 + sum_g off[g][n]*xs[m][g]) +
// residual.  acc4 comes from oneDNN with the grouped f16 step scales applied;
// the offset (zero-point) part is added by w4_epilogue_launch from the group
// sums of the same quantized activations.
bool dnnl_gemm::gemm_w4(const void * key, const float * residual, float alpha, int M, int K, float * out,
                        int out_stride) {
    if (!p->acts_valid || M != p->cur_M || K != p->cur_K) {
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
    if (M >= 2 && p->split_valid && p->axe && p->axo && p->axg && p->asa && p->xs) {
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
        const char * e = getenv("PF_W4_DEBUG");
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
                    M, K, w.N, key, (int)p->acts_valid, (int)p->split_valid, (int)(p->axe != nullptr), (int)has_axe,
                    (void *)out, (void *)residual, alpha, hist[0], hist[1], hist[2], hist[3]);
        }
    }
    static const bool no_gemv = [] {
        const char * e = getenv("PF_W4_NOGEMV");
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
            const char * e = getenv("PF_W4_GEMM_MAXM");
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
    dnnl_capture_guard();
    pe.prim.execute(p->st, {{DNNL_ARG_SRC, pe.src},
                            {DNNL_ARG_WEIGHTS, w.wmem},
                            {DNNL_ARG_DST, pe.dst},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, w.scmem},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, pe.sscales}});
    const int N = w.N;
    const int ng = K / kW4Group;
    // PF_W4_NOCORR: bisection knob - drop the offset (zero-point) correction and
    // keep only the oneDNN step-scaled matmul, to tell which half is wrong.
    static const bool nocorr = [] {
        const char * e = getenv("PF_W4_NOCORR");
        return e && atoi(e) != 0;
    }();
    // xs was produced by quantize() (same launch as the grouped activation)
    w4_epilogue_launch(p->q, p->accf, p->xs, nocorr ? nullptr : w.off, p->asa, out, out_stride, residual, alpha, M, N,
                       ng);
    return true;
}

const int8_t * dnnl_gemm::weight_data(const void * key) const {
    auto it = p->weights.find(key);
    return it == p->weights.end() ? nullptr : it->second.dev;
}

// Narrow-int8 group fusion.  kNarrowRows is where the per-launch latency stops
// mattering: at 27B sizes the GEMV itself (nat_gemm's per-call staging and
// dp4a) is what costs, and the batched path is better above it.
static const int kNarrowRows = 2048;

bool dnnl_gemm::gemm_i8_group(const void * const * keys, const int * n_rows, const float * const * outs,
                              int out_stride, const float * const * residuals, const float * alphas, int n_segs,
                              int M, int K) {
    if (n_segs < 1 || n_segs > 4 || !p->acts_valid || M < 1 || M > 13 || K <= 0 || K % 32) {
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

const int8_t * dnnl_gemm::act_grp_data() const {
    return p->axg;
}

const uint16_t * dnnl_gemm::act_grp_scales() const {
    return p->asa;
}

const int8_t * dnnl_gemm::act_data() const {
    return p->acts_valid ? p->ax : nullptr;
}

const float * dnnl_gemm::act_group_sums() const {
    return p->acts_valid ? p->xs : nullptr;
}

const float * dnnl_gemm::act_scales() const {
    return p->acts_valid ? p->axs : nullptr;
}

const int32_t * dnnl_gemm::act_sum() const {
    return p->acts_valid ? p->axsum : nullptr;
}

// ---------------------------------------------------------------------------
// XMX attention GEMMs.  Operands and dst are the caller's USM buffers; only
// the primitive is cached.
// The attention matmuls are submitted to `st_a`, which wraps the same in-order
// queue as the softmax/accumulate kernels, so the following SYCL kernel is
// ordered after the matmul without an explicit wait.  PF_ATTN_WAIT=1 restores
// a per-matmul stream wait for A/B.
static bool attn_wait_env() {
    static const bool v = [] {
        const char * e = getenv("PF_ATTN_WAIT");
        return e && atoi(e) != 0;
    }();
    return v;
}

bool dnnl_gemm::attn_qk(int M, int blk, const int8_t * q_kmajor, const int8_t * k_keymajor, int32_t * dst) {
    const auto * e = p->attn_qk_prim(M, /*K=*/256, blk);
    if (!e->ok) {
        return false;
    }
    // `st_a` wraps the same in-order queue as the engine's SYCL work, so the
    // matmul is ordered after the Q-quantize / K-gather kernels without a
    // queue drain; waiting only `st_a` avoids draining the dense GEMMs queued
    // on the main stream (a queue-wide wait here serialised the whole prefill).
    auto xm = sycl_interop::make_memory(memory::desc({M, 256}, memory::data_type::s8, memory::format_tag::ab), p->eng,
                                        sycl_interop::memory_kind::usm, (void *)q_kmajor);
    auto wm = sycl_interop::make_memory(memory::desc({256, blk}, memory::data_type::s8, memory::format_tag::ba), p->eng,
                                        sycl_interop::memory_kind::usm, (void *)k_keymajor);
    auto dm = sycl_interop::make_memory(memory::desc({M, blk}, memory::data_type::s32, memory::format_tag::ab), p->eng,
                                        sycl_interop::memory_kind::usm, (void *)dst);
    dnnl_capture_guard();
    e->prim.execute(p->st_a, {{DNNL_ARG_SRC, xm}, {DNNL_ARG_WEIGHTS, wm}, {DNNL_ARG_DST, dm}});
    if (attn_wait_env()) {
        p->st_a.wait();
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
    dnnl_capture_guard();
    e->prim.execute(p->st_a, {{DNNL_ARG_SRC, xm}, {DNNL_ARG_WEIGHTS, wm}, {DNNL_ARG_DST, dm}});
    if (attn_wait_env()) {
        p->st_a.wait();
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
