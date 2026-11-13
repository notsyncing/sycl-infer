// DeepSeek-V4.1-Flash mHC kernels.  See ds41_hc.h for the contract; the math
// mirrors arch/deepseek41/forward_host.cpp's hc_mixes/hc_pre/hc_post step for step.
#include "ds41_hc.h"

namespace si {

using namespace sycl;

namespace {

constexpr int kMixWG = 256;
constexpr int kMixNSub = kMixWG / 32;

inline float sigmoidd(float x) { return 1.f / (1.f + sycl::exp(-x)); }

// Sinkhorn with a compile-time hc: every index is constant-bounded, so the
// hc*hc matrix stays in registers.  With a runtime hc the compiler spills the
// array and the 20 serial normalisation passes become ~57 us for one token.
template <int HC>
inline void hc_sinkhorn_t(const float * mixes, const float * scale, const float * base, float * pre,
                          float * post, float * comb, int iters, float eps) {
    for (int s = 0; s < HC; s++) {
        pre[s] = sigmoidd(mixes[s] * scale[0] + base[s]) + eps;
        post[s] = 2.f * sigmoidd(mixes[HC + s] * scale[1] + base[HC + s]);
    }
    float C[HC * HC];
    for (int j = 0; j < HC; j++) {
        float m = -1e30f;
        for (int k = 0; k < HC; k++) {
            const int idx = 2 * HC + j * HC + k;
            C[j * HC + k] = mixes[idx] * scale[2] + base[idx];
            m = sycl::fmax(m, C[j * HC + k]);
        }
        float sum = 0.f;
        for (int k = 0; k < HC; k++) {
            C[j * HC + k] = sycl::exp(C[j * HC + k] - m);
            sum += C[j * HC + k];
        }
        for (int k = 0; k < HC; k++) {
            C[j * HC + k] = C[j * HC + k] / sum + eps;
        }
    }
    // Column (destination) normalisation first, then (iters-1) x (row, column).
    for (int pass = 0; pass < iters; pass++) {
        if (pass > 0) {
            for (int j = 0; j < HC; j++) {
                float s = eps;
                for (int k = 0; k < HC; k++) {
                    s += C[j * HC + k];
                }
                const float rs = 1.f / s;
                for (int k = 0; k < HC; k++) {
                    C[j * HC + k] *= rs;
                }
            }
        }
        for (int k = 0; k < HC; k++) {
            float s = eps;
            for (int j = 0; j < HC; j++) {
                s += C[j * HC + k];
            }
            const float rs = 1.f / s;
            for (int j = 0; j < HC; j++) {
                C[j * HC + k] *= rs;
            }
        }
    }
    for (int i = 0; i < HC * HC; i++) {
        comb[i] = C[i];
    }
}

// Fallback for a runtime hc (hc != 4).  Same arithmetic, slower.
inline void hc_sinkhorn_generic(const float * mixes, const float * scale, const float * base, float * pre,
                                float * post, float * comb, int hc, int iters, float eps) {
    for (int s = 0; s < hc; s++) {
        pre[s] = sigmoidd(mixes[s] * scale[0] + base[s]) + eps;
        post[s] = 2.f * sigmoidd(mixes[hc + s] * scale[1] + base[hc + s]);
    }
    float C[64];
    for (int j = 0; j < hc; j++) {
        float m = -1e30f;
        for (int k = 0; k < hc; k++) {
            const int idx = 2 * hc + j * hc + k;
            C[j * hc + k] = mixes[idx] * scale[2] + base[idx];
            m = sycl::fmax(m, C[j * hc + k]);
        }
        float sum = 0.f;
        for (int k = 0; k < hc; k++) {
            C[j * hc + k] = sycl::exp(C[j * hc + k] - m);
            sum += C[j * hc + k];
        }
        for (int k = 0; k < hc; k++) {
            C[j * hc + k] = C[j * hc + k] / sum + eps;
        }
    }
    for (int pass = 0; pass < iters; pass++) {
        if (pass > 0) {
            for (int j = 0; j < hc; j++) {
                float s = eps;
                for (int k = 0; k < hc; k++) {
                    s += C[j * hc + k];
                }
                const float rs = 1.f / s;
                for (int k = 0; k < hc; k++) {
                    C[j * hc + k] *= rs;
                }
            }
        }
        for (int k = 0; k < hc; k++) {
            float s = eps;
            for (int j = 0; j < hc; j++) {
                s += C[j * hc + k];
            }
            const float rs = 1.f / s;
            for (int j = 0; j < hc; j++) {
                C[j * hc + k] *= rs;
            }
        }
    }
    for (int i = 0; i < hc * hc; i++) {
        comb[i] = C[i];
    }
}

} // namespace

// Split-K across kDs41HcMixSplit workgroups per mix row: a single workgroup
// per row is only 192 warps for the whole [24,20480] GEMV and cannot hide the
// weight read's latency.  Each writes (sumsq, dot) for its K slice.
void ds41_hc_mixes_launch(queue & q, const float * flat, const float * fn, const float * scale,
                          const float * base, float * mixes, float * partials, float * pre, float * post,
                          float * comb, int hc, int d, int iters, float eps) {
    if (!flat || !fn || !scale || !base || !mixes || !partials || !pre || !post || !comb) {
        return;
    }
    if (hc <= 0 || hc > 8 || d <= 0 || iters < 1) {
        return;
    }
    const int fd = hc * d;
    const int nm = (2 + hc) * hc;

    q.submit([&](handler & h) {
        local_accessor<float, 1> part(2 * kMixNSub, h);
        h.parallel_for(nd_range<1>((size_t)nm * kDs41HcMixSplit * kMixWG, kMixWG),
                       [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int g = (int)it.get_group(0);
                           const int m = g / kDs41HcMixSplit;
                           const int sp = g % kDs41HcMixSplit;
                           const int tid = (int)it.get_local_id(0);
                           const int lane = tid % 32;
                           const int sub = tid / 32;

                           const int kb = (int)((long long)sp * fd / kDs41HcMixSplit);
                           const int ke = (int)((long long)(sp + 1) * fd / kDs41HcMixSplit);
                           float ss = 0.f;
                           float acc = 0.f;
                           const float * fr = fn + (size_t)m * (size_t)fd;
                           for (int i = kb + tid; i < ke; i += kMixWG) {
                               const float v = flat[i];
                               ss = sycl::fma(v, v, ss);
                               acc = sycl::fma(v, fr[i], acc);
                           }
                           const sub_group sg = it.get_sub_group();
                           for (int off = 16; off > 0; off >>= 1) {
                               ss += sycl::permute_group_by_xor(sg, ss, off);
                               acc += sycl::permute_group_by_xor(sg, acc, off);
                           }
                           if (lane == 0) {
                               part[sub] = ss;
                               part[kMixNSub + sub] = acc;
                           }
                           it.barrier();
                           if (tid == 0) {
                               float s = 0.f, a = 0.f;
                               for (int i = 0; i < kMixNSub; i++) {
                                   s += part[i];
                                   a += part[kMixNSub + i];
                               }
                               float * dst = partials + (size_t)(m * kDs41HcMixSplit + sp) * 2;
                               dst[0] = s;
                               dst[1] = a;
                           }
                       });
    });

    // Fold the K slices, form `mixes`, then the Sinkhorn pass.  One work-item
    // reproduces the host's sequential order.
    q.single_task([=]() {
        for (int m = 0; m < nm; m++) {
            float ss = 0.f, dot = 0.f;
            for (int sp = 0; sp < kDs41HcMixSplit; sp++) {
                const float * src = partials + (size_t)(m * kDs41HcMixSplit + sp) * 2;
                ss += src[0];
                dot += src[1];
            }
            const float r = 1.f / sycl::sqrt(ss / (float)fd + 1e-20f);
            mixes[m] = dot * r;
        }
        if (hc == 4) {
            hc_sinkhorn_t<4>(mixes, scale, base, pre, post, comb, iters, eps);
        } else {
            hc_sinkhorn_generic(mixes, scale, base, pre, post, comb, hc, iters, eps);
        }
    });
}

void ds41_hc_pre_launch(queue & q, const float * h, const float * mix, float * y, int hc, int d) {
    if (!h || !mix || !y || hc <= 0 || d <= 0) {
        return;
    }
    q.parallel_for(range<1>((size_t)d), [=](id<1> e) {
        float acc = 0.f;
        for (int s = 0; s < hc; s++) {
            acc = sycl::fma(mix[s], h[(size_t)s * (size_t)d + (size_t)e[0]], acc);
        }
        y[e[0]] = acc;
    });
}

void ds41_hc_post_launch(queue & q, const float * y, const float * res, const float * post,
                         const float * comb, float * out, int hc, int d) {
    if (!y || !res || !post || !comb || !out || hc <= 0 || d <= 0) {
        return;
    }
    q.parallel_for(range<2>((size_t)hc, (size_t)d), [=](id<2> idx) {
        const int k = (int)idx[0];
        const int e = (int)idx[1];
        float acc = post[k] * y[e];
        for (int j = 0; j < hc; j++) {
            acc = sycl::fma(comb[j * hc + k], res[(size_t)j * (size_t)d + (size_t)e], acc);
        }
        out[(size_t)k * (size_t)d + (size_t)e] = acc;
    });
}

} // namespace si
