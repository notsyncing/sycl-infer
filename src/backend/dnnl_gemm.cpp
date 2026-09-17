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
#include <vector>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include "kernels.h"
#include "quant.h"

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

constexpr int kActW = 256;  // activation quantizer workgroup size
constexpr int kActVPT = 16; // values per thread (K <= kActW * kActVPT)
constexpr int kActMaxK = kActW * kActVPT;

// Quantize one call's activations: row-major [M][K] int8 with one symmetric
// scale per row (scale = max|row|/127, the same scheme as the dp4a xq_launch
// but per row instead of per 32 values - oneDNN's int8 matmul only takes one
// scale per dimension).  The row is read once (values stay in registers) and
// the row max is reduced across the workgroup.
void act_quant_launch(sycl::queue & q, const float * x, const float * up, int x_stride, int up_stride, int8_t * xq,
                      float * scales, int M, int K) {
    q.parallel_for(sycl::nd_range<1>((size_t)M * kActW, kActW), [=](sycl::nd_item<1> it) {
        const int m = (int)it.get_group(0);
        const int lane = (int)it.get_local_id(0);
        const float * xr = x + (size_t)m * x_stride;
        const float * ur = up ? up + (size_t)m * up_stride : nullptr;
        float v[kActVPT];
        float mx = 0.f;
#pragma unroll
        for (int i = 0; i < kActVPT; i++) {
            const int k = lane + i * kActW;
            float val = 0.f;
            if (k < K) {
                val = xr[k];
                if (ur) {
                    val = silu_act(val) * ur[k];
                }
            }
            v[i] = val;
            mx = sycl::fmax(mx, sycl::fabs(val));
        }
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        const float scale = mx > 0.f ? mx / 127.0f : 1.0f;
        if (lane == 0) {
            scales[m] = scale;
        }
        int8_t * orow = xq + (size_t)m * K;
#pragma unroll
        for (int i = 0; i < kActVPT; i++) {
            const int k = lane + i * kActW;
            if (k < K) {
                int qi = (int)sycl::round(v[i] / scale);
                qi = sycl::max(-127, sycl::min(127, qi));
                orow[k] = (int8_t)qi;
            }
        }
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

bool dnnl_gemm_enabled() {
    // default ON (the mode-2 int8 GEMMs run on oneDNN); PF_GEMM_DNNL=0 forces
    // the dp4a chunk-batched path, e.g. for bit-exact dp4a validation
    static const bool on = [] {
        const char * e = getenv("PF_GEMM_DNNL");
        return !(e && atoi(e) == 0);
    }();
    return on;
}

// ---------------------------------------------------------------------------
struct dnnl_gemm::impl {
    sycl::queue q;
    engine eng;
    stream st;
    // activation scratch: row-major [M][K] int8 + one scale per row
    int8_t * ax = nullptr;
    float * axs = nullptr;
    int cap_M = 0, cap_K = 0; // scratch capacity
    int cur_M = 0, cur_K = 0; // currently quantized activations
    bool acts_valid = false;
    // integer matmul output [M][N] s32
    int32_t * acc = nullptr;
    size_t acc_cap = 0;

    struct w_entry {
        int K = 0, N = 0;
        int8_t * dev = nullptr;   // [N][K] row-major int8
        float * scales = nullptr; // [N]
        memory wmem;
        bool ok = false;
    };
    std::unordered_map<const void *, w_entry> weights;

    struct prim_entry {
        matmul prim;
        memory src, dst;
    };
    std::unordered_map<uint64_t, prim_entry> prims;

    static uint64_t pkey(int M, int K, int N) {
        return ((uint64_t)M << 42) | ((uint64_t)K << 21) | (uint64_t)N;
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
        auto wmd = memory::desc({K, N}, memory::data_type::s8, memory::format_tag::ba);
        auto dmd = memory::desc({M, N}, memory::data_type::s32, memory::format_tag::ab);
        prim_entry e;
        e.prim = matmul(matmul::primitive_desc(eng, xmd, wmd, dmd));
        e.src = sycl_interop::make_memory(xmd, eng, sycl_interop::memory_kind::usm, (void *)ax);
        e.dst = sycl_interop::make_memory(dmd, eng, sycl_interop::memory_kind::usm, (void *)acc);
        auto ins = prims.emplace(pkey(M, K, N), std::move(e));
        return &ins.first->second;
    }
};

dnnl_gemm::dnnl_gemm(sycl::queue & q) : p(new impl) {
    p->q = q;
    p->eng = sycl_interop::make_engine(q.get_device(), q.get_context());
    p->st = sycl_interop::make_stream(p->eng, q);
    p->cap_M = kMaxB * kMaxT; // max rows of one chunk-batched prefill
    p->cap_K = kActMaxK;
    p->ax = sycl::malloc_device<int8_t>((size_t)p->cap_M * p->cap_K, q);
    p->axs = sycl::malloc_device<float>(p->cap_M, q);
    p->acc_cap = (size_t)p->cap_M * 6144; // widest N in this model (wqkv)
    p->acc = sycl::malloc_device<int32_t>(p->acc_cap, q);
    if (!p->ax || !p->axs || !p->acc) {
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
    if (p->acc) {
        sycl::free(p->acc, p->q);
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
    if (ggml_type != 12 && ggml_type != 13 && ggml_type != 14) {
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
    std::vector<float> hs((size_t)N);
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
                float amax = 0.f;
                for (int k = 0; k < K; k++) {
                    amax = std::max(amax, std::fabs(fr[k]));
                }
                const float s = amax > 0.f ? amax / 127.0f : 1.0f;
                int8_t * q8 = hw.data() + (size_t)r * K;
                for (int k = 0; k < K; k++) {
                    int v = (int)std::lround(fr[k] / s);
                    v = std::max(-127, std::min(127, v));
                    q8[k] = (int8_t)v;
                }
                hs[(size_t)r] = s;
            }
        });
    }
    for (auto & x : th) {
        x.join();
    }

    impl::w_entry w;
    w.K = K;
    w.N = N;
    w.dev = sycl::malloc_device<int8_t>(nvals, p->q);
    w.scales = sycl::malloc_device<float>((size_t)N, p->q);
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
    p->q.memcpy(w.scales, hs.data(), (size_t)N * 4).wait();
    w.wmem = sycl_interop::make_memory(memory::desc({K, N}, memory::data_type::s8, memory::format_tag::ba), p->eng,
                                       sycl_interop::memory_kind::usm, (void *)w.dev);

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
        } catch (const std::exception &) {
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
            pe.prim.execute(p->st, {{DNNL_ARG_SRC, pe.src}, {DNNL_ARG_WEIGHTS, *wmem}, {DNNL_ARG_DST, pe.dst}});
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
        act_quant_launch(p->q, dx, nullptr, WK, WK, p->ax, p->axs, WM, WK);
        epilogue_vec_launch<4>(p->q, p->acc, p->axs, dsw, dout, WN, nullptr, 1.f, WM, WN);
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

bool dnnl_gemm::quantize(const float * x, const float * up, int x_stride, int up_stride, int M, int K) {
    p->acts_valid = false;
    if (!x || M <= 0 || M > p->cap_M || K <= 0 || K > p->cap_K) {
        return false;
    }
    act_quant_launch(p->q, x, up, x_stride, up_stride, p->ax, p->axs, M, K);
    p->cur_M = M;
    p->cur_K = K;
    p->acts_valid = true;
    return true;
}

bool dnnl_gemm::gemm(const void * key, const float * residual, float alpha, int M, int K, float * out, int out_stride) {
    if (!p->acts_valid || M != p->cur_M || K != p->cur_K) {
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
    pe.prim.execute(p->st, {{DNNL_ARG_SRC, pe.src}, {DNNL_ARG_WEIGHTS, w.wmem}, {DNNL_ARG_DST, pe.dst}});
    const int N = w.N;
    const bool aligned = ((N % 4) == 0) && ((out_stride % 4) == 0) && ((reinterpret_cast<uintptr_t>(out) % 16) == 0)
                         && (residual == nullptr || (reinterpret_cast<uintptr_t>(residual) % 16) == 0);
    if (aligned) {
        epilogue_vec_launch<4>(p->q, p->acc, p->axs, w.scales, out, out_stride, residual, alpha, M, N);
    } else {
        epilogue_scalar_launch(p->q, p->acc, p->axs, w.scales, out, out_stride, residual, alpha, M, N);
    }
    return true;
}

const int8_t * dnnl_gemm::weight_data(const void * key) const {
    auto it = p->weights.find(key);
    return it == p->weights.end() ? nullptr : it->second.dev;
}

const float * dnnl_gemm::weight_scales(const void * key) const {
    auto it = p->weights.find(key);
    return it == p->weights.end() ? nullptr : it->second.scales;
}

const int8_t * dnnl_gemm::act_data() const {
    return p->acts_valid ? p->ax : nullptr;
}

const float * dnnl_gemm::act_scales() const {
    return p->acts_valid ? p->axs : nullptr;
}

} // namespace si
