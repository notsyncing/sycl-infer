// Host kernels for the qwen4exp additions: hyper-connection mixing, the PLE
// n-gram module, and the QSA indexer's small scans.  Everything else in the
// architecture reuses the ordinary CPU GEMV / GDN / MoE kernels.
//
// The engine's q4 forward passes one flat token list: token i lives at buffer
// row `rows[i]` (the same row the other kernels use), with its own sequence
// slot and absolute position.  Prefill is one sequence and decode is one token
// per sequence, so both shapes fall out of the same list.
#include <immintrin.h>

#include <chrono>
#include <cstdio>

#include "common.h"
#include "common/cpu_isa.h"
#include "common/env.h"

namespace si {

namespace {

__attribute__((target("avx2"))) inline int32_t dot32_u8s8(const uint8_t * a, const int8_t * b) {
    const __m256i va = _mm256_loadu_si256((const __m256i *)a);
    const __m256i vb = _mm256_loadu_si256((const __m256i *)b);
    const __m256i p = _mm256_maddubs_epi16(va, vb);
    const __m256i s = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
    __m128i lo = _mm256_castsi256_si128(s);
    const __m128i hi = _mm256_extracti128_si256(s, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni"))) inline int32_t dot32_vnni(const uint8_t * a,
                                                                                       const int8_t * b) {
    const __m256i va = _mm256_loadu_si256((const __m256i *)a);
    const __m256i vb = _mm256_loadu_si256((const __m256i *)b);
    const __m256i v = _mm256_dpbusd_epi32(_mm256_setzero_si256(), va, vb);
    __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

inline int32_t dot32_dispatch(const uint8_t * a, const int8_t * b) {
    if (cpu_features_of().avx512vnni) {
        return dot32_vnni(a, b);
    }
    return isa().qgemv_sb ? dot32_u8s8(a, b) : isa().dot_i8(a, b, 32);
}

// Q8_0's weights are signed int8; dp4a is u8 x s8, so bias the weights by 128
// and take the whole-group activation sum back out - exactly what the device
// kernel does with `q ^ 0x80` and `-128 * sum(xq)`.  (AVX-VNNI-INT8's signed
// dpbssd would also work but is a different extension than AVX512-VNNI and is
// not on the reference CPU.)
__attribute__((target("avx2"))) inline int32_t dot32_s8(const int8_t * a, const int8_t * b, int32_t xsum) {
    const __m256i qa = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)a), _mm256_set1_epi8((char)0x80));
    alignas(32) uint8_t qu[32];
    _mm256_store_si256((__m256i *)qu, qa);
    return dot32_dispatch(qu, b) - 128 * xsum;
}

// The stream layout of one token is [hc][n_embd], contiguous.
inline int tok_row(const int32_t * rows, int i) {
    return rows ? rows[i] : i;
}

} // namespace

void cpu_q4_hc_norm(const float * x, const float * w, float * out, const int32_t * rows, int n_tok, int n_embd, int hc,
                    float eps) {
    const int total = n_tok * hc;
    par(total, [&](int j) {
        const int i = j / hc;
        const int s = j % hc;
        const size_t base = (size_t)tok_row(rows, i) * hc * n_embd;
        const float * src = x + base + (size_t)s * n_embd;
        float * dst = out + base + (size_t)s * n_embd;
        const float * g = w + (size_t)s * n_embd;
        rmsnorm_row(src, g, dst, n_embd, eps);
    });
}

void cpu_q4_hc_mix(const float * xn, const float * gate, float * mixed, const int32_t * rows, int n_tok, int n_embd,
                   int hc) {
    const int total = n_tok * n_embd;
    par(total, [&](int j) {
        const int i = j / n_embd;
        const int c = j % n_embd;
        const size_t base = (size_t)tok_row(rows, i) * hc * n_embd;
        float acc = 0.0f;
        for (int s = 0; s < hc; s++) {
            const float v = xn[base + (size_t)s * n_embd + c];
            const float g = gate[base + (size_t)s * n_embd + c];
            acc += v * cpu_sigmoid(g);
        }
        mixed[(size_t)tok_row(rows, i) * n_embd + c] = acc / (float)hc;
    });
}

// residual[s] += block_out * 2*sigmoid(inject[s] / hc)
void cpu_q4_hc_combine(float * residual, const float * block_out, const float * inject, const int32_t * rows, int n_tok,
                       int n_embd, int hc) {
    const int total = n_tok * hc * n_embd;
    par(total, [&](int j) {
        const int c = j % n_embd;
        const int s = (j / n_embd) % hc;
        const int i = j / (hc * n_embd);
        const size_t base = (size_t)tok_row(rows, i) * hc * n_embd;
        const float w2 = 2.0f * cpu_sigmoid(inject[(size_t)tok_row(rows, i) * hc + s] / (float)hc);
        residual[base + (size_t)s * n_embd + c] += block_out[(size_t)tok_row(rows, i) * n_embd + c] * w2;
    });
}

// x = silu(x * scale), in place.  The HC low-rank gate applies the 1/hc factor
// before the activation.
void cpu_q4_arb_gemv(const void * w, uint32_t type, const float * x, int xs, const float * up, float * out, int os,
                     int n_tok, int K, int N) {
    const int nb = K / 32;
    par(n_tok * N, [&](int j) {
        const int t = j / N;
        const int n = j % N;
        const float * xr = x + (size_t)t * xs;
        const float * ur = up ? up + (size_t)t * xs : nullptr;
        if (ur) {
            // silu(gate) * up: the activation is materialised per row because
            // the block loop below reads it with the same stride as x
        }
        float acc = 0.f;
        if (type == 8) {
            const unsigned char * row = (const unsigned char *)w + (size_t)n * nb * 34;
            for (int b = 0; b < nb; b++) {
                const auto * q = (const block_q8_0 *)(row + (size_t)b * 34);
                const float d = ggml_half_to_float(q->d);
                float part = 0.f;
                for (int i = 0; i < 32; i++) {
                    const float xv = xr[b * 32 + i];
                    part += (float)q->qs[i] * (ur ? cpu_silu(xv) * ur[b * 32 + i] : xv);
                }
                acc += d * part;
            }
        } else if (type == 0) {
            if (ur) {
                const float * wr = (const float *)w + (size_t)n * K;
                for (int i = 0; i < K; i++) {
                    acc += wr[i] * cpu_silu(xr[i]) * ur[i];
                }
            } else {
                acc = dot_f32(xr, (const float *)w + (size_t)n * K, K);
            }
        } else if (type == 30) {
            const uint16_t * rp = (const uint16_t *)w + (size_t)n * K;
            for (int k = 0; k < K; k++) {
                acc += ggml_bf16_to_float(rp[k]) * xr[k];
            }
        } else if (type == 7) {
            const unsigned char * row = (const unsigned char *)w + (size_t)n * nb * 24;
            for (int b = 0; b < nb; b++) {
                const auto * q = (const block_q5_1 *)(row + (size_t)b * 24);
                const float d = ggml_half_to_float(q->d), m = ggml_half_to_float(q->m);
                float part = 0.f;
                for (int i = 0; i < 32; i++) {
                    const int hi = (q->qh[i / 8] >> (i % 8)) & 1;
                    const int lo = (q->qs[i % 16] >> (4 * (i / 16))) & 0xF;
                    const float xv = xr[b * 32 + i];
                    part += (d * (float)(lo | (hi << 4)) + m) * (ur ? cpu_silu(xv) * ur[b * 32 + i] : xv);
                }
                acc += part;
            }
        }
        out[(size_t)t * os + n] = acc;
    });
}

void cpu_q4_up_gemv(const void * w, uint32_t type, const float * x, int xs, float * out, int os, int n_tok, int K,
                    int N) {
    cpu_q4_arb_gemv(w, type, x, xs, nullptr, out, os, n_tok, K, N);
}

void cpu_q4_convert_down(const void * src, uint32_t type, int K, int N, int E, size_t plane_bytes, uint8_t * dst,
                         float * sc) {
    const int G = K / 32;
    const size_t row_bytes = quant_row_bytes(type, K);
    par(E * N, [&](int j) {
        const int e = j / N;
        const int n = j % N;
        thread_local std::vector<float> v;
        v.resize((size_t)K);
        const char * p = (const char *)src + (size_t)e * plane_bytes + (size_t)n * row_bytes;
        dequantize_row(type, p, v.data(), K);
        uint8_t * drow = dst + (size_t)j * K;
        float * srow = sc + (size_t)j * G;
        for (int g = 0; g < G; g++) {
            float mx = 0.f;
            for (int i = 0; i < 32; i++) {
                mx = std::max(mx, std::fabs(v[(size_t)g * 32 + i]));
            }
            const float s = mx > 0.f ? mx / 127.0f : 1.0f;
            srow[g] = s;
            for (int i = 0; i < 32; i++) {
                const int q = (int)std::lround(v[(size_t)g * 32 + i] / s);
                drow[(size_t)g * 32 + i] = (uint8_t)(std::max(-127, std::min(127, q)) + 128);
            }
        }
    });
}

// Q5_1 -> 128-value panel repack for the routed down plane.
//
// WHY.  A Q5_1 block is 24 B per 32 values (d, m, a 4-byte high-bit plane and a
// 16-byte nibble plane), so an SG=8 kernel has to issue four loads per 24 B and
// the warp covers only four blocks per instruction group.  Re-laying the plane
// out as 128-value panels - which is the Q4_K superblock width the grouped
// gate/up and down kernels already stream well - lets a lane's four values of
// a group pair come from one naturally-aligned uint32, one uint16 of high bits
// and one uint32 of meta: the same 96 B per panel as four Q5_1 blocks, so the
// plane stays byte-identical in size (0.75 B/weight, no extra GiB).
//
// Panel = 4 consecutive 32-value groups:
//   qs[64]  : group pair p at qs + 32p; byte j is the low nibble of the even
//             group's element j and the high nibble of the odd group's, so a
//             lane reads uint32 at qs + 32p + 4*ch and gets both groups' four
//             values (low = even, high = odd).
//   qh[16]  : 8 uint16, lane ch reads qh + 2ch; its four bits per group are at
//             bit 4*sub, where sub = gr&3, and cover elements 4ch..4ch+3.
//   meta[16]: 4 x {f16 d, f16 m} at meta + 4*sub, read as one uint32.
// The raw f16 bits are copied, not converted: this is a repack, not a
// quantization.
void q5_1_sb_pack(const void * src, int K, int N, int E, void * dst) {
    const int ng = K / 32;
    if (ng <= 0 || (ng % 4) != 0) {
        return; // only whole 128-value panels; the caller gates on this too
    }
    const size_t row_bytes = (size_t)ng * sizeof(block_q5_1);
    const int npan = ng / 4;
    constexpr size_t kPanel = 96; // 64 qs + 16 qh + 16 meta
    const uint8_t * s = (const uint8_t *)src;
    uint8_t * d = (uint8_t *)dst;
    // A row is only 480 B, so one row per pool task is ~56M tasks per startup
    // and the pool's shared atomic cursor dominates (measured 23.7 s for the
    // 25.2 GiB).  Chunk rows instead: the work is a linear copy either way and
    // one task per 256 rows keeps the atomic count in the thousands.
    const int total = E * N;
    constexpr int kRowsPerTask = 256;
    const int ntasks = (total + kRowsPerTask - 1) / kRowsPerTask;
    par(ntasks, [&](int t) {
        const int j0 = t * kRowsPerTask;
        const int j1 = std::min(total, j0 + kRowsPerTask);
        for (int j = j0; j < j1; j++) {
        // plane_bytes == N * row_bytes for every routed expert tensor, so one
        // contiguous index addresses (expert, row).
        const uint8_t * sr = s + (size_t)j * row_bytes;
        uint8_t * dr = d + (size_t)j * row_bytes;
        for (int pn = 0; pn < npan; pn++) {
            const uint8_t * sp = sr + (size_t)pn * 4 * sizeof(block_q5_1);
            uint8_t * dp = dr + (size_t)pn * kPanel;
            uint8_t * dqs = dp;      // [64]
            uint8_t * dqh = dp + 64; // [16]
            uint8_t * dme = dp + 80; // [16]
            for (int p = 0; p < 2; p++) {
                const auto * be = (const block_q5_1 *)(sp + (size_t)(2 * p) * sizeof(block_q5_1));
                const auto * bo = (const block_q5_1 *)(sp + (size_t)(2 * p + 1) * sizeof(block_q5_1));
                for (int i = 0; i < 32; i++) {
                    const int ev = i < 16 ? (be->qs[i] & 0xF) : ((be->qs[i - 16] >> 4) & 0xF);
                    const int ov = i < 16 ? (bo->qs[i] & 0xF) : ((bo->qs[i - 16] >> 4) & 0xF);
                    dqs[32 * p + i] = (uint8_t)(ev | (ov << 4));
                }
            }
            uint16_t qh[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int sub = 0; sub < 4; sub++) {
                const auto * b = (const block_q5_1 *)(sp + (size_t)sub * sizeof(block_q5_1));
                uint32_t q = 0;
                std::memcpy(&q, b->qh, 4);
                for (int i = 0; i < 32; i++) {
                    if ((q >> i) & 1u) {
                        qh[i >> 2] |= (uint16_t)(1u << (4 * sub + (i & 3)));
                    }
                }
                std::memcpy(dme + 4 * sub, &b->d, 2);
                std::memcpy(dme + 4 * sub + 2, &b->m, 2);
            }
            std::memcpy(dqh, qh, sizeof(qh));
        }
        }
    });
}

void cpu_q4_down_grouped(const uint8_t * w, const float * sc, const int8_t * x8, const float * xmeta,
                         const int32_t * xsumq, int xTB, int P, int N, int K, float * acc, int acc_stride,
                         const int * pairs) {
    const int G = K / 32;
    const int TT = 64;
    par(N, [&](int row) {
        const uint8_t * wr = w + (size_t)row * K;
        const float * sr = sc + (size_t)row * G;
        for (int t0 = 0; t0 < P; t0 += TT) {
            const int pn = std::min(TT, P - t0);
            float a[TT];
            for (int v = 0; v < pn; v++) {
                a[v] = 0.f;
            }
            for (int g = 0; g < G; g++) {
                const uint8_t * qv = wr + (size_t)g * 32;
                const float sv = sr[g];
                for (int v = 0; v < pn; v++) {
                    const int pr = pairs ? pairs[t0 + v] : (t0 + v);
                    const size_t gi = (size_t)g * xTB + pr;
                    const int32_t qd = dot32_dispatch(qv, x8 + gi * 32);
                    const int32_t xs = xsumq[gi * 2] + xsumq[gi * 2 + 1];
                    a[v] += xmeta[gi * 2] * sv * (float)(qd - 128 * xs);
                }
            }
            for (int v = 0; v < pn; v++) {
                const int pr = pairs ? pairs[t0 + v] : (t0 + v);
                acc[(size_t)pr * acc_stride + row] = a[v];
            }
        }
    });
}



static bool q5_bmi2() {
    static const bool v = __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("avx2");
    return v;
}

__attribute__((target("avx2,bmi2"))) static inline void q5_build_fast(const uint8_t * b, uint8_t * q) {
    __m128i qsv = _mm_loadu_si128((const __m128i *)(b + 8));
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m128i lo0 = _mm_and_si128(qsv, m4);
    __m128i lo1 = _mm_and_si128(_mm_srli_epi16(qsv, 4), m4);
    uint32_t qh;
    memcpy(&qh, b + 4, 4);
    const uint64_t sp = 0x0101010101010101ull;
    uint64_t e0 = _pdep_u64((uint64_t)(qh & 0xFFu), sp);
    uint64_t e1 = _pdep_u64((uint64_t)((qh >> 8) & 0xFFu), sp);
    uint64_t e2 = _pdep_u64((uint64_t)((qh >> 16) & 0xFFu), sp);
    uint64_t e3 = _pdep_u64((uint64_t)((qh >> 24) & 0xFFu), sp);
    __m256i hi = _mm256_setr_epi64x((long long)e0, (long long)e1, (long long)e2, (long long)e3);
    hi = _mm256_slli_epi16(hi, 4);
    __m256i lo = _mm256_set_m128i(lo1, lo0);
    _mm256_storeu_si256((__m256i *)q, _mm256_or_si256(lo, hi));
}

// Native Q5_1 down: same math as the int8 store path, but reads the GGUF block
// (24 B / 32 values = 0.75 B/weight against 1.125) and reconstructs the q5
// values in a register.  The q build is hoisted out of the pair loop, so its
// 32 scalar ops are amortised over the tile's tokens.
//
// Factored into a per-row body so the serial form (the prefill's `par(expert)`
// path) and the row-parallel form (a decode CPU miss: one expert, so the
// parallelism has to be on the output rows) compute the identical arithmetic.
static void q5_1_down_row(const uint8_t * base, size_t row_bytes, const int8_t * x8, const float * xmeta,
                          const int32_t * xsumq, int xTB, int P, int K, int row, float * acc, int acc_stride,
                          const int * pairs, bool fastq) {
    const int G = K / 32;
    const int TT = 64;
    const uint8_t * wr = base + (size_t)row * row_bytes;
    for (int t0 = 0; t0 < P; t0 += TT) {
        const int pn = std::min(TT, P - t0);
        float a[TT];
        for (int v = 0; v < pn; v++) {
            a[v] = 0.f;
        }
        for (int g = 0; g < G; g++) {
            const uint8_t * b = wr + (size_t)g * 24;
            const float d = ggml_half_to_float((uint16_t)(b[0] | (b[1] << 8)));
            const float m = ggml_half_to_float((uint16_t)(b[2] | (b[3] << 8)));
            uint32_t qh;
            memcpy(&qh, b + 4, 4);
            uint8_t q[32];
            if (fastq) {
                q5_build_fast(b, q);
            } else {
                for (int i = 0; i < 32; i++) {
                    const int lo = (b[8 + (i % 16)] >> (4 * (i / 16))) & 0xF;
                    q[i] = (uint8_t)(lo | (((qh >> i) & 1u) << 4));
                }
            }
            for (int v = 0; v < pn; v++) {
                const int pr = pairs ? pairs[t0 + v] : (t0 + v);
                const size_t gi = (size_t)g * xTB + pr;
                const int32_t qd = dot32_dispatch(q, x8 + gi * 32);
                const int32_t xs = xsumq[gi * 2] + xsumq[gi * 2 + 1];
                a[v] += xmeta[gi * 2] * (d * (float)qd + m * (float)xs);
            }
        }
        for (int v = 0; v < pn; v++) {
            const int pr = pairs ? pairs[t0 + v] : (t0 + v);
            acc[(size_t)pr * acc_stride + row] = a[v];
        }
    }
}

void cpu_q5_1_down_grouped_serial(const void * w, size_t row_bytes, const int8_t * x8, const float * xmeta,
                                  const int32_t * xsumq, int xTB, int P, int N, int K, float * acc, int acc_stride,
                                  const int * pairs) {
    const uint8_t * base = (const uint8_t *)w;
    const bool fastq = q5_bmi2();
    for (int row = 0; row < N; row++) {
        q5_1_down_row(base, row_bytes, x8, xmeta, xsumq, xTB, P, K, row, acc, acc_stride, pairs, fastq);
    }
}

void cpu_q5_1_down_grouped(const void * w, size_t row_bytes, const int8_t * x8, const float * xmeta,
                           const int32_t * xsumq, int xTB, int P, int N, int K, float * acc, int acc_stride,
                           const int * pairs) {
    const uint8_t * base = (const uint8_t *)w;
    const bool fastq = q5_bmi2();
    // Row chunks, not rows: one row is 20 groups of work and the pool's
    // dispatch dominates it.  The arithmetic is per row and unchanged.
    const int nc = std::max(1, std::min(N, cpu_thread_count() * 4));
    par(nc, [&](int c) {
        const int r0 = (int)((long long)N * c / nc);
        const int r1 = (int)((long long)N * (c + 1) / nc);
        for (int row = r0; row < r1; row++) {
            q5_1_down_row(base, row_bytes, x8, xmeta, xsumq, xTB, P, K, row, acc, acc_stride, pairs, fastq);
        }
    });
}

// Native Q8_0 down: `{f16 d; int8 qs[32]}` (34 B per 32 values, no minimum).
// The device kernel biases the weights by 128 for dp4a and subtracts
// 128*sum(xq); the host reads the signed bytes directly, so the same value is
// `xmeta * d * sum(q*xq)` with no correction term.  qwen4exp's layers
// 4/30/46/47 carry this down type.
static void q8_0_down_row(const uint8_t * base, size_t row_bytes, const int8_t * x8, const float * xmeta,
                          const int32_t * xsumq, int xTB, int P, int K, int row, float * acc, int acc_stride,
                          const int * pairs) {
    const int G = K / 32;
    const uint8_t * wr = base + (size_t)row * row_bytes;
    for (int t0 = 0; t0 < P; t0 += 64) {
        const int pn = std::min(64, P - t0);
        float a[64];
        for (int v = 0; v < pn; v++) {
            a[v] = 0.f;
        }
        for (int g = 0; g < G; g++) {
            const uint8_t * b = wr + (size_t)g * 34;
            const float d = ggml_half_to_float((uint16_t)(b[0] | (b[1] << 8)));
            const int8_t * q = (const int8_t *)(b + 2);
            for (int v = 0; v < pn; v++) {
                const int pr = pairs ? pairs[t0 + v] : (t0 + v);
                const size_t gi = (size_t)g * xTB + pr;
                const int32_t xs = xsumq[gi * 2] + xsumq[gi * 2 + 1];
                const int32_t qd = dot32_s8(q, x8 + gi * 32, xs);
                a[v] += xmeta[gi * 2] * d * (float)qd;
            }
        }
        for (int v = 0; v < pn; v++) {
            const int pr = pairs ? pairs[t0 + v] : (t0 + v);
            acc[(size_t)pr * acc_stride + row] = a[v];
        }
    }
}

void cpu_q8_0_down_grouped(const void * w, size_t row_bytes, const int8_t * x8, const float * xmeta,
                           const int32_t * xsumq, int xTB, int P, int N, int K, float * acc, int acc_stride,
                           const int * pairs) {
    const uint8_t * base = (const uint8_t *)w;
    const int nc = std::max(1, std::min(N, cpu_thread_count() * 4));
    par(nc, [&](int c) {
        const int r0 = (int)((long long)N * c / nc);
        const int r1 = (int)((long long)N * (c + 1) / nc);
        for (int row = r0; row < r1; row++) {
            q8_0_down_row(base, row_bytes, x8, xmeta, xsumq, xTB, P, K, row, acc, acc_stride, pairs);
        }
    });
}

void cpu_q4_down_grouped_serial(const uint8_t * w, const float * sc, const int8_t * x8, const float * xmeta,
                                const int32_t * xsumq, int xTB, int P, int N, int K, float * acc, int acc_stride,
                                const int * pairs) {
    const int G = K / 32;
    const int TT = 64;
    for (int row = 0; row < N; row++) {
        const uint8_t * wr = w + (size_t)row * K;
        const float * sr = sc + (size_t)row * G;
        for (int t0 = 0; t0 < P; t0 += TT) {
            const int pn = std::min(TT, P - t0);
            float a[TT];
            for (int v = 0; v < pn; v++) {
                a[v] = 0.f;
            }
            for (int g = 0; g < G; g++) {
                const uint8_t * qv = wr + (size_t)g * 32;
                const float sv = sr[g];
                for (int v = 0; v < pn; v++) {
                    const int pr = pairs ? pairs[t0 + v] : (t0 + v);
                    const size_t gi = (size_t)g * xTB + pr;
                    const int32_t qd = dot32_dispatch(qv, x8 + gi * 32);
                    const int32_t xs = xsumq[gi * 2] + xsumq[gi * 2 + 1];
                    a[v] += xmeta[gi * 2] * sv * (float)(qd - 128 * xs);
                }
            }
            for (int v = 0; v < pn; v++) {
                const int pr = pairs ? pairs[t0 + v] : (t0 + v);
                acc[(size_t)pr * acc_stride + row] = a[v];
            }
        }
    }
}

void cpu_q4_silu_scaled(float * x, int n, float scale) {
    par(n, [&](int i) { x[i] = cpu_silu(x[i] * scale); });
}

// PLE gate: normed key . normed hidden, per stream.
//   s      = sum_i kn[i]*qn[i] / sqrt(n_embd)
//   gate   = sigmoid(sign(s) * sqrt(|s|))
void cpu_q4_ple_gate(const float * key, const float * hidden, const float * w_key, const float * w_query, float * gate,
                     const int32_t * rows, int n_tok, int n_embd, int hc, float eps) {
    const float inv_sqrt = 1.0f / std::sqrt((float)n_embd);
    const int total = n_tok * hc;
    par(total, [&](int j) {
        const int i = j / hc;
        const int s = j % hc;
        const size_t base = (size_t)tok_row(rows, i) * hc * n_embd + (size_t)s * n_embd;
        const float * k = key + base;
        const float * q = hidden + base;
        const float * wk = w_key + (size_t)s * n_embd;
        const float * wq = w_query + (size_t)s * n_embd;
        const float sk = sumsq_f32(k, n_embd);
        const float sq = sumsq_f32(q, n_embd);
        const float ik = 1.0f / std::sqrt(sk / n_embd + eps);
        const float iq = 1.0f / std::sqrt(sq / n_embd + eps);
        float acc = 0.0f;
        for (int d = 0; d < n_embd; d++) {
            acc += k[d] * wk[d] * ik * q[d] * wq[d] * iq;
        }
        const float sval = acc * inv_sqrt;
        const float mag = std::sqrt(std::max(1e-6f, std::fabs(sval)));
        gate[(size_t)tok_row(rows, i) * hc + s] = cpu_sigmoid(sval < 0.0f ? -mag : mag);
    });
}

// gated = value broadcast over the streams, scaled by each stream's gate, then
// the depthwise causal conv (dilation = ngram, kernel ple_conv) over the
// grouped-RMSNorm of `gated`.  Writes residual += gated + conv_out and appends
// the normalised rows to the per-slot history.
//
// `hist` is [slot][hist_rows][hc*n_embd] in chronological order; `hist_rows` is
// (kern-1)*ngram.  `n_hist[slot]` says how many of those rows are valid.
void cpu_q4_ple_apply(const float * value, const float * gate, const float * w_conv, const float * conv_w,
                      float * residual, float * gated, float * normed, float * hist, int32_t * n_hist,
                      const int32_t * rows, const int32_t * slots, const int32_t * poss, int n_tok, int n_embd, int hc,
                      int ngram, int kern, float eps) {
    const int hc_dim = hc * n_embd;
    const int hist_rows = (kern - 1) * ngram;
    // PF_Q4_PLEAPPLY_TIME: per-step wall clock of this host kernel.  `apply` is
    // 1.9 ms/token in the engine's PLE block, which is far more than its
    // 4 x 10 240-element loops should cost; this says which step pays.
    static const bool pat = si::env::flag("PF_Q4_PLEAPPLY_TIME");
    auto pnow = [] {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    const double pt0 = pat ? pnow() : 0.0;
    double pt1 = pt0, pt2 = pt0, pt3 = pt0;
    // The four loops here are 10 240 elements in a decode.  `par` wakes the
    // whole host pool and rendezvouses on it, and measured on the 28-thread
    // box that costs ~0.5 ms per call for this size - the whole 4-loop kernel
    // was 1.9 ms/token against ~0.2 ms of actual work.  Run small regions on
    // the calling thread; the prefill (n_tok * hc_dim = 256 K) still pools.
    // The threshold is on the WORK, not on the job count: loop 2 below has only
    // n_tok*hc jobs but each is a 2560-element RMSNorm, so a prefill's 2048 jobs
    // are 5.2 M elements and must stay pooled.
    static const int ple_ser_max = si::env::i32("PF_Q4_PLE_SER_MAX", 65536);
    auto papply = [&](int n, long long work, auto && fn) {
        if (work <= ple_ser_max) {
            for (int i = 0; i < n; i++) {
                fn(i);
            }
            return;
        }
        par(n, fn);
    };
    // PF_Q4_PLE_APPLY_IDIV: restructure loops 1 and 3 so the element index is
    // produced by nested (i, s, c) loops instead of `j % n_embd` / `j / hc_dim`
    // per element (two integer divisions on 20 480 elements per decode token,
    // ~0.3 ms).  Every element's arithmetic and the 4-tap conv order are
    // unchanged, and the papply work argument stays n_tok*hc_dim so the
    // serial/pooled decision is the same.
    static const bool ple_idiv = si::env::i32("PF_Q4_PLE_APPLY_IDIV", 1) != 0;
    // 1. gated value per stream
    const int total = n_tok * hc_dim;
    if (ple_idiv) {
        papply(n_tok * hc, (long long)total, [&](int js) {
            const int i = js / hc;
            const int s = js % hc;
            const int vr = tok_row(rows, i);
            const float g = gate[(size_t)vr * hc + s];
            const float * vp = value + (size_t)vr * n_embd;
            float * gp = gated + (size_t)vr * hc_dim + (size_t)s * n_embd;
            for (int c = 0; c < n_embd; c++) {
                gp[c] = vp[c] * g;
            }
        });
    } else {
        papply(total, (long long)total, [&](int j) {
            const int c = j % n_embd;
            const int s = (j / n_embd) % hc;
            const int i = j / hc_dim;
            gated[(size_t)tok_row(rows, i) * hc_dim + (size_t)s * n_embd + c] =
                value[(size_t)tok_row(rows, i) * n_embd + c] * gate[(size_t)tok_row(rows, i) * hc + s];
        });
    }
    if (pat) pt1 = pnow();
    // 2. grouped RMSNorm of the gated value
    papply(n_tok * hc, (long long)n_tok * hc_dim, [&](int j) {
        const int i = j / hc;
        const int s = j % hc;
        const size_t base = (size_t)tok_row(rows, i) * hc_dim + (size_t)s * n_embd;
        rmsnorm_row(gated + base, w_conv + (size_t)s * n_embd, normed + base, n_embd, eps);
    });
    if (pat) pt2 = pnow();
    // 3. conv + residual add.  group[i] = position of token i inside its slot's
    // token run, computed by a linear scan (n_tok is at most kMaxB*kMaxT=512 and
    // usually <= 32).
    std::vector<int> base_pos(n_tok, 0);
    for (int i = 0; i < n_tok; i++) {
        int g = 0;
        for (int j = 0; j < i; j++) {
            if (slots[j] == slots[i]) {
                g++;
            }
        }
        base_pos[i] = poss[i] - g; // position of this slot's first token in the call
    }
    if (ple_idiv) {
        papply(n_tok, (long long)n_tok * hc_dim, [&](int i) {
            const int slot = slots[i];
            const int base = base_pos[i];
            const size_t row = (size_t)tok_row(rows, i) * hc_dim;
            for (int c = 0; c < hc_dim; c++) {
                float acc = 0.0f;
                for (int k = 0; k < kern; k++) {
                    const int d = (kern - 1 - k) * ngram;
                    const int tp = poss[i] - d;
                    float xv;
                    if (tp >= base) {
                        const int gi = i - (poss[i] - tp);
                        xv = normed[(size_t)tok_row(rows, gi) * hc_dim + c];
                    } else {
                        const int hi = hist_rows + (tp - base);
                        xv = (hi >= 0 && hi < hist_rows && n_hist[slot] > hi)
                                 ? hist[((size_t)slot * hist_rows + hi) * hc_dim + c]
                                 : 0.0f;
                    }
                    // ple_conv1d is stored [kernel][channel] with nb0=kernel: the
                    // tap coefficient for channel c is conv_w[k + kern*c], not
                    // conv_w[k*hc_dim+c]
                    acc += conv_w[(size_t)k + (size_t)kern * c] * xv;
                }
                residual[row + c] += gated[row + c] + cpu_silu(acc);
            }
        });
    } else {
        papply(n_tok * hc_dim, (long long)n_tok * hc_dim, [&](int j) {
            const int c = j % hc_dim;
            const int i = j / hc_dim;
            const int slot = slots[i];
            const int base = base_pos[i];
            float acc = 0.0f;
            for (int k = 0; k < kern; k++) {
                const int d = (kern - 1 - k) * ngram;
                const int tp = poss[i] - d;
                float xv;
                if (tp >= base) {
                    const int gi = i - (poss[i] - tp);
                    xv = normed[(size_t)tok_row(rows, gi) * hc_dim + c];
                } else {
                    const int hi = hist_rows + (tp - base);
                    xv = (hi >= 0 && hi < hist_rows && n_hist[slot] > hi)
                             ? hist[((size_t)slot * hist_rows + hi) * hc_dim + c]
                             : 0.0f;
                }
                // ple_conv1d is stored [kernel][channel] with nb0=kernel: the tap
                // coefficient for channel c is conv_w[k + kern*c], not conv_w[k*hc_dim+c]
                acc += conv_w[(size_t)k + (size_t)kern * c] * xv;
            }
            residual[(size_t)tok_row(rows, i) * hc_dim + c] +=
                gated[(size_t)tok_row(rows, i) * hc_dim + c] + cpu_silu(acc);
        });
    }
    if (pat) pt3 = pnow();
    // 4. append the new normalised rows to each slot's history
    for (int i = 0; i < n_tok;) {
        const int slot = slots[i];
        int end = i;
        while (end < n_tok && slots[end] == slot) {
            end++;
        }
        const int cnt = end - i;
        float * h = hist + (size_t)slot * hist_rows * hc_dim;
        const int valid = n_hist[slot];
        // Right-aligned invariant: the last `valid` rows occupy
        // h[hist_rows-valid .. hist_rows-1], so a tap at position tp maps to
        // h[hist_rows + (tp - base)].  Append the newest `keep` rows at the end
        // and slide the surviving old ones left.
        const int keep = std::min(cnt, hist_rows);
        const int keep_old = std::min(valid, hist_rows - keep);
        const int old_dst = hist_rows - keep - keep_old;
        const int old_src = hist_rows - valid;
        if (keep_old > 0 && old_dst != old_src) {
            std::memmove(h + (size_t)old_dst * hc_dim, h + (size_t)old_src * hc_dim,
                         (size_t)keep_old * hc_dim * sizeof(float));
        }
        papply(keep * hc_dim, (long long)keep * hc_dim, [&](int j) {
            const int t = j / hc_dim;
            const int c = j % hc_dim;
            h[(size_t)(hist_rows - keep + t) * hc_dim + c] =
                normed[(size_t)tok_row(rows, i + cnt - keep + t) * hc_dim + c];
        });
        n_hist[slot] = keep_old + keep;
        i = end;
    }
    if (pat) {
        const double pe = pnow();
        fprintf(stderr, "[q4pa] n=%d gated=%.3f rms=%.3f conv=%.3f hist=%.3f total=%.3f ms\n", n_tok, pt1 - pt0,
                pt2 - pt1, pt3 - pt2, pe - pt3, pe - pt0);
    }
}

} // namespace si
