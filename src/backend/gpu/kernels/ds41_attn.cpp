// DeepSeek-V4.1-Flash MLA attention, compressor, indexer, RoPE and fp4
// rounding.  The math mirrors engine_ds41.cpp (sparse_attn, compressor,
// indexer, rope_one, fp4_round_row); see ds41_attn.h for the contracts.
#include "ds41_attn.h"

#include <cstdint>
#include <cstring>

namespace si {

using namespace sycl;

namespace {

constexpr int kAttnWG = 512;
constexpr int kAttnNSub = kAttnWG / 32;

// Exact 2^e for the small integer exponents used below.  Built from the float
// bits rather than sycl::exp2 because the device lowering of exp2 on this
// backend is not exact enough for a quantization scale (measured ~3.7% off at
// 2^-6 in the first probe run, which is far outside any fast-math budget).
inline float pow2f(int e) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = (uint32_t)(e + 127) << 23;
    return v.f;
}

// ---------------------------------------------------------------- fp4 helpers
// E4M3FN nearest representable value (round to nearest even); the exponent is
// taken from the float bits because a := |v| is always normal here.
inline float e4m3_val(float v) {
    const bool neg = sycl::signbit(v);
    float a = sycl::fabs(v);
    if (!(a > 0.f)) {
        return neg ? -0.0f : 0.0f;
    }
    if (a >= 448.f) {
        return neg ? -448.f : 448.f;
    }
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = a;
    const uint32_t u = bits.u;
    const int E = (int)((u >> 23) & 0xffu) - 127;
    const float step = (E < -6) ? pow2f(-9) : pow2f(E - 3);
    const float q = a / step;
    float out = sycl::rint(q) * step;
    if (out > 448.f) {
        out = 448.f;
    }
    return neg ? -out : out;
}

constexpr float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

inline float e2m1_val(float v) {
    float a = sycl::fabs(v);
    if (a > 6.f) {
        a = 6.f;
    }
    int best = 0;
    float bd = 1e30f;
    for (int i = 0; i < 8; i++) {
        const float d = sycl::fabs(a - kE2M1[i]);
        if (d < bd) {
            bd = d;
            best = i;
        }
    }
    if (best < 7 && sycl::fabs(a - kE2M1[best + 1]) == bd && (best & 1)) {
        best++;
    }
    return sycl::copysign(kE2M1[best], v);
}

} // namespace

void ds41_mla_attn_launch(queue & q, const float * qv, const float * win, const int * win_pos,
                          int win_cap, const float * ckv, int n_sel, const float * sink, int n_head,
                          int head_dim, int pos, int n_win, int win_start, float scale,
                          const float * freq, int rot_off, int n_rot, float * out) {
    if (!qv || !sink || !out || n_head <= 0 || head_dim <= 0) {
        return;
    }
    if (head_dim > kAttnWG * 8) {
        return; // the per-thread accumulator array below is sized for this
    }
    if (n_win < 0) {
        n_win = 0;
    }
    if (n_sel < 0) {
        n_sel = 0;
    }
    if ((n_win > 0) && (!win || win_cap <= 0)) {
        return;
    }
    if ((n_sel > 0) && !ckv) {
        return;
    }
    const int nk = n_win + n_sel;
    if (nk > 8192) {
        return; // the per-key probability table is a 32 KB local-accessor cap
    }

    q.submit([&](handler & h) {
        local_accessor<float, 1> sc((size_t)nk + 1, h);
        h.parallel_for(nd_range<1>((size_t)n_head * kAttnWG, kAttnWG),
                       [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int head = (int)it.get_group(0);
                           const int tid = (int)it.get_local_id(0);
                           const int lane = tid % 32;
                           const int sub = tid / 32;
                           const float * qh = qv + (size_t)head * (size_t)head_dim;

                           // --- scores: one warp per key, lane-strided dot.
                           // The window and compressed loops are separate so
                           // the validity test and slot arithmetic stay out of
                           // the inner dimension loop.
                           const sub_group sg = it.get_sub_group();
                           for (int j = sub; j < n_win; j += kAttnNSub) {
                               const int wj = win_start + j;
                               int slot = wj % win_cap;
                               if (slot < 0) {
                                   slot += win_cap;
                               }
                               if (win_pos && win_pos[slot] != wj) {
                                   if (lane == 0) {
                                       sc[j] = -1e30f;
                                   }
                                   continue;
                               }
                               const float * kv = win + (size_t)slot * (size_t)head_dim;
                               float acc = 0.f;
                               for (int d = lane; d < head_dim; d += 32) {
                                   acc = sycl::fma(qh[d], kv[d], acc);
                               }
                               for (int off = 16; off > 0; off >>= 1) {
                                   acc += sycl::permute_group_by_xor(sg, acc, off);
                               }
                               if (lane == 0) {
                                   sc[j] = acc * scale;
                               }
                           }
                           for (int j = sub; j < n_sel; j += kAttnNSub) {
                               const float * kv = ckv + (size_t)j * (size_t)head_dim;
                               float acc = 0.f;
                               for (int d = lane; d < head_dim; d += 32) {
                                   acc = sycl::fma(qh[d], kv[d], acc);
                               }
                               for (int off = 16; off > 0; off >>= 1) {
                                   acc += sycl::permute_group_by_xor(sg, acc, off);
                               }
                               if (lane == 0) {
                                   sc[n_win + j] = acc * scale;
                               }
                           }
                           it.barrier();

                           // --- softmax (sink is an extra value-0 key); the
                           // sequential order matches the host reference.
                           if (tid == 0) {
                               float m = sink[head];
                               for (int i = 0; i < nk; i++) {
                                   m = sycl::fmax(m, sc[i]);
                               }
                               m = sycl::fmax(m, -1e30f);
                               float den = sycl::exp(sink[head] - m);
                               for (int i = 0; i < nk; i++) {
                                   const float p = sycl::exp(sc[i] - m);
                                   den += p;
                                   sc[i] = p;
                               }
                               sc[nk] = (den > 0.f) ? 1.f / den : 0.f;
                           }
                           it.barrier();
                           const float inv = sc[nk];

                           // --- output: out[d] = sum_i p_i * key_i[d].
                           // Every thread owns <= 8 dims, so the per-key
                           // pointer/slot/branch work happens once per key
                           // instead of once per (key, dim).
                           int dv[8];
                           float acc[8];
                           int nd = 0;
                           for (int d = tid; d < head_dim; d += kAttnWG) {
                               dv[nd] = d;
                               acc[nd] = 0.f;
                               nd++;
                           }
                           for (int j = 0; j < n_win; j++) {
                               const float p = sc[j];
                               if (p == 0.f) {
                                   continue;
                               }
                               const int wj = win_start + j;
                               int slot = wj % win_cap;
                               if (slot < 0) {
                                   slot += win_cap;
                               }
                               if (win_pos && win_pos[slot] != wj) {
                                   continue;
                               }
                               const float * kv = win + (size_t)slot * (size_t)head_dim;
                               for (int t = 0; t < nd; t++) {
                                   acc[t] = sycl::fma(p, kv[dv[t]], acc[t]);
                               }
                           }
                           for (int j = 0; j < n_sel; j++) {
                               const float p = sc[n_win + j];
                               if (p == 0.f) {
                                   continue;
                               }
                               const float * kv = ckv + (size_t)j * (size_t)head_dim;
                               for (int t = 0; t < nd; t++) {
                                   acc[t] = sycl::fma(p, kv[dv[t]], acc[t]);
                               }
                           }
                           for (int t = 0; t < nd; t++) {
                               out[(size_t)head * (size_t)head_dim + (size_t)dv[t]] = acc[t] * inv;
                           }
                           it.barrier();

                           // --- inverse RoPE on the last n_rot dims, in place
                           if (freq && n_rot > 0 && rot_off >= 0 && rot_off + n_rot <= head_dim) {
                               if (tid < n_rot / 2) {
                                   const float ang = (float)pos * freq[tid];
                                   const float c = sycl::cos(ang);
                                   float s = sycl::sin(ang);
                                   s = -s; // inverse
                                   float * p = out + (size_t)head * (size_t)head_dim +
                                               (size_t)rot_off + (size_t)(2 * tid);
                                   const float x0 = p[0];
                                   const float x1 = p[1];
                                   p[0] = x0 * c - x1 * s;
                                   p[1] = x0 * s + x1 * c;
                               }
                           }
                       });
    });
}

void ds41_compressor_launch(queue & q, const float * kvc, const float * gate, const float * norm,
                            float * out, int n_groups, int ratio, int d, float eps) {
    if (!kvc || !norm || !out || n_groups <= 0 || ratio <= 0 || d <= 0) {
        return;
    }
    if (ratio > 1 && !gate) {
        return;
    }
    // 1) per-column softmax-gated sum over the ratio rows of one group.
    q.parallel_for(range<2>((size_t)n_groups, (size_t)d), [=](id<2> idx) {
        const int g = (int)idx[0];
        const int e = (int)idx[1];
        const size_t base = (size_t)g * (size_t)ratio * (size_t)d + (size_t)e;
        if (ratio == 1) {
            out[(size_t)g * (size_t)d + (size_t)e] = kvc[base];
            return;
        }
        float m = -1e30f;
        for (int j = 0; j < ratio; j++) {
            m = sycl::fmax(m, gate[base + (size_t)j * (size_t)d]);
        }
        float sum = 0.f;
        for (int j = 0; j < ratio; j++) {
            sum += sycl::exp(gate[base + (size_t)j * (size_t)d] - m);
        }
        float acc = 0.f;
        for (int j = 0; j < ratio; j++) {
            const float p = sycl::exp(gate[base + (size_t)j * (size_t)d] - m) / sum;
            acc = sycl::fma(p, kvc[base + (size_t)j * (size_t)d], acc);
        }
        out[(size_t)g * (size_t)d + (size_t)e] = acc;
    });

    // 2) RMSNorm (float sum of squares; this device has no fp64 path).
    constexpr int WG = 256;
    q.submit([&](handler & h) {
        local_accessor<float, 1> rr(1, h);
        h.parallel_for(nd_range<1>((size_t)n_groups * WG, WG), [=](nd_item<1> it) {
            const int g = (int)it.get_group(0);
            const int tid = (int)it.get_local_id(0);
            float * row = out + (size_t)g * (size_t)d;
            if (tid == 0) {
                // The host uses a double accumulator here; this device has no
                // fp64 scalar path, so use a strided float sum.  Over 512
                // values the extra rounding is ~1e-7 relative, below the
                // 1e-6 comparison tolerance.
                float ss = 0.f;
                for (int i = 0; i < d; i++) {
                    ss = sycl::fma(row[i], row[i], ss);
                }
                rr[0] = 1.f / sycl::sqrt(ss / (float)d + eps);
            }
            it.barrier();
            const float r = rr[0];
            for (int i = tid; i < d; i += WG) {
                row[i] = row[i] * r * norm[i];
            }
        });
    });
}

void ds41_rope_launch(queue & q, float * v, int n_rows, int row_stride, int off, int rd, int pos0,
                      int pos_stride, const float * freq, bool inverse) {
    if (!v || !freq || n_rows <= 0 || rd <= 0 || rd % 2 != 0) {
        return;
    }
    if (off < 0 || off + rd > row_stride) {
        return;
    }
    q.parallel_for(range<2>((size_t)n_rows, (size_t)(rd / 2)), [=](id<2> idx) {
        const int r = (int)idx[0];
        const int j = (int)idx[1];
        const int pos = pos0 + r * pos_stride;
        const float ang = (float)pos * freq[j];
        const float c = sycl::cos(ang);
        float s = sycl::sin(ang);
        if (inverse) {
            s = -s;
        }
        float * p = v + (size_t)r * (size_t)row_stride + (size_t)off + (size_t)(2 * j);
        const float x0 = p[0];
        const float x1 = p[1];
        p[0] = x0 * c - x1 * s;
        p[1] = x0 * s + x1 * c;
    });
}

void ds41_fp4_round_launch(queue & q, float * v, int n, int block, bool e4m3_scale) {
    if (!v || n <= 0 || block <= 0) {
        return;
    }
    const int nblk = (n + block - 1) / block;
    q.parallel_for(range<1>((size_t)nblk), [=](id<1> ib) {
        const int b = (int)ib[0];
        const int i0 = b * block;
        const int m = (block < n - i0) ? block : (n - i0);
        float amax = 0.f;
        for (int i = 0; i < m; i++) {
            amax = sycl::fmax(amax, sycl::fabs(v[i0 + i]));
        }
        float s;
        if (e4m3_scale) {
            amax = sycl::fmax(amax, 6.f * pow2f(-9));
            s = e4m3_val(amax / 6.f);
        } else {
            amax = sycl::fmax(amax, 6.f * pow2f(-126));
            const float qv = amax / 6.f;
            union {
                float f;
                uint32_t u;
            } bits;
            bits.f = qv;
            const uint32_t u = bits.u;
            const int E = (int)((u >> 23) & 0xffu) - 127;
            const bool pow2 = (u & 0x007fffffu) == 0;
            s = pow2f(pow2 ? E : E + 1);
        }
        for (int i = 0; i < m; i++) {
            const float x = sycl::fmax(-6.f, sycl::fmin(6.f, v[i0 + i] / s));
            v[i0 + i] = e2m1_val(x) * s;
        }
    });
}

void ds41_indexer_score_launch(queue & q, const float * qi, const float * ik, const float * w,
                               float * score, int n_rows, int n_head, int hdim) {
    if (!qi || !ik || !w || !score || n_rows <= 0 || n_head <= 0 || hdim <= 0) {
        return;
    }
    const int nsub = kAttnNSub;
    q.submit([&](handler & h) {
        local_accessor<float, 1> part(nsub, h);
        h.parallel_for(nd_range<1>((size_t)n_rows * kAttnWG, kAttnWG),
                       [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int row = (int)it.get_group(0);
                           const int tid = (int)it.get_local_id(0);
                           const int lane = tid % 32;
                           const int sub = tid / 32;
                           const float * kr = ik + (size_t)row * (size_t)hdim;
                           float mine = 0.f;
                           for (int hh = sub; hh < n_head; hh += nsub) {
                               const float * qh = qi + (size_t)hh * (size_t)hdim;
                               float acc = 0.f;
                               for (int d = lane; d < hdim; d += 32) {
                                   acc = sycl::fma(qh[d], kr[d], acc);
                               }
                               const sub_group sg = it.get_sub_group();
                               for (int off = 16; off > 0; off >>= 1) {
                                   acc += sycl::permute_group_by_xor(sg, acc, off);
                               }
                               if (lane == 0 && acc > 0.f) {
                                   mine = sycl::fma(acc, w[hh], mine);
                               }
                           }
                           if (lane == 0) {
                               part[sub] = mine;
                           }
                           it.barrier();
                           if (tid == 0) {
                               float s = 0.f;
                               for (int i = 0; i < nsub; i++) {
                                   s += part[i];
                               }
                               score[row] = s;
                           }
                       });
    });
}

} // namespace si
