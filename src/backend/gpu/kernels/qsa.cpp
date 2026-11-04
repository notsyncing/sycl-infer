// QSA Stage 2 (PF_QSA_DEVATTN): device-side sparse attention helpers.
//
// The selector stays on the host.  These two kernels only bridge to the
// existing paged-attention kernels:
//   * qsa_gather_launch copies the selected key rows out of the paged f16 K/V
//     pool into a compact pool laid out exactly like the paged one
//     ([block][kv head][kBlockSize token][head_dim]), so `attn` can read it
//     with a sequential block table;
//   * qsa_sinfo_launch writes the scratch step_info that makes that compact
//     pool look like a one-row, nsel-position sequence.  It is a device kernel
//     because a host-USM step_info would be overwritten for the next token
//     before the previous attention kernel had read it.
#include <cassert>
#include <climits>

#include "kernels.h"
#include "kernel_utils.h"

namespace si {

using namespace sycl;

void qsa_sinfo_launch(queue & q, step_info * info, int nsel) {
    q.single_task([=]() {
        info->n_rows = 1;
        info->tpb = 1;
        info->n_real = 1;
        info->n_real_row[0] = 1;
        info->active[0] = 1;
        info->pos[0] = nsel - 1; // the compact pool is a sequence 0..nsel-1
        info->slot[0] = 0;       // compact_table has no per-slot stride
    });
}

// Sparse-prefill scratch step_info: `n_rows` rows, each one token, each at the
// position that makes the compact pool look like a `counts[r]`-key sequence.
// It is a device kernel for the same reason as qsa_sinfo_launch: the next wave
// overwrites the buffer while this wave's attention may still be reading it,
// and only the queue's ordering may order them.
void qsa_sinfo_multi_launch(queue & q, step_info * info, const int32_t * counts, int n_rows) {
    q.single_task([=]() {
        info->n_rows = n_rows;
        info->tpb = 1;
        info->n_real = 1;
        for (int r = 0; r < n_rows; r++) {
            info->n_real_row[r] = 1;
            info->active[r] = 1;
            info->pos[r] = counts[r] - 1; // compact positions 0..counts[r]-1
            info->slot[r] = r;            // row r's block base is r*cap_blocks
        }
    });
}

// Compact the selected rows of a *whole prefill wave* into one pool.  The
// destination uses the paged layout ([block][kv head][kBlockSize][head_dim])
// with row r owning blocks [r*cap_blocks, (r+1)*cap_blocks), so the caller's
// identity block table maps row r straight to them.  Rows past `counts[r]`
// are left alone: attn's position bound never reads them.
void qsa_gather_rows_launch(queue & q, const void * kpool, const void * vpool, const int32_t * src_table,
                            const int32_t * sel, const int32_t * counts, int row_stride, int cap_blocks, int n_rows,
                            int n_kv, int head_dim, int block_size, void * ck, void * cv) {
    assert(block_size == 32 && "QSA prefill gather assumes the engine's KV block size");
    assert(head_dim == 256 && "QSA prefill gather assumes Qwen3.5's 256-wide heads");
    assert(n_rows <= kMaxB && "QSA prefill wave exceeds the step_info row ceiling");
    if (n_rows <= 0 || n_kv <= 0 || head_dim <= 0 || block_size <= 0 || cap_blocks <= 0) {
        return;
    }
    const sycl::half * K = (const sycl::half *)kpool;
    const sycl::half * V = (const sycl::half *)vpool;
    sycl::half * CK = (sycl::half *)ck;
    sycl::half * CV = (sycl::half *)cv;
    const int keys_per_row = cap_blocks * block_size;
    q.parallel_for(range<1>((size_t)n_rows * (size_t)keys_per_row * (size_t)n_kv), [=](id<1> i) {
        const int r = (int)(i / (size_t)(keys_per_row * n_kv));
        const int rem = (int)(i % (size_t)(keys_per_row * n_kv));
        const int j = rem / n_kv;
        const int kvh = rem % n_kv;
        if (j >= counts[r]) {
            return;
        }
        const int sp = sel[(size_t)r * row_stride + j];
        const int sb = src_table[sp / block_size];
        const int so = sp % block_size;
        const int db = r * cap_blocks + j / block_size;
        const int dd = j % block_size;
        const size_t unit_src = (size_t)sb * n_kv + kvh;
        const size_t unit_dst = (size_t)db * n_kv + kvh;
        const sycl::half * sk = K + (unit_src * block_size + so) * head_dim;
        const sycl::half * sv = V + (unit_src * block_size + so) * head_dim;
        sycl::half * dk = CK + (unit_dst * block_size + dd) * head_dim;
        sycl::half * dv = CV + (unit_dst * block_size + dd) * head_dim;
        int d = 0;
        for (; d + 8 <= head_dim; d += 8) {
            *reinterpret_cast<sycl::vec<sycl::half, 8> *>(dk + d) =
                *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(sk + d);
            *reinterpret_cast<sycl::vec<sycl::half, 8> *>(dv + d) =
                *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(sv + d);
        }
        for (; d < head_dim; d++) {
            dk[d] = sk[d];
            dv[d] = sv[d];
        }
    });
}

// QSA device indexer: one token's indexer projection.  The host implementation
// (cpu_qsa_bf16_gemv + head_norm + head_rope) is the oracle; here each output
// row is one sub-group whose lanes walk the K dimension, then lane 0 writes the
// dot.  A second, tiny kernel applies q_norm and RoPE to the q rows in place.
// The dot reassociates (a sub-group reduction instead of AVX-512's four
// accumulators), so the selector scores can move by fp noise; PF_QSA_Q /
// PF_QSA_SCORE / PF_QSA_POOLS are the dump gates.
namespace {

inline float qsa_bf2f(uint16_t h) {
    const uint32_t b = (uint32_t)h << 16;
    float f;
    __builtin_memcpy(&f, &b, sizeof(f));
    return f;
}

} // namespace

void qsa_indexer_launch(queue & q, const float * x, const void * qw, const void * kw, int K, int inh, int id,
                        int n_rot, float rope_base, int pos, const float * q_norm, float * qout, float * kout,
                        float eps) {
    if (K <= 0 || inh <= 0 || id <= 0) {
        return;
    }
    const int qrows = inh * id;
    const int rows = qrows + id;
    const uint16_t * QW = (const uint16_t *)qw;
    const uint16_t * KW = (const uint16_t *)kw;
    q.parallel_for(nd_range<1>((size_t)rows * 32, 32), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int r = it.get_group(0);
        const auto sg = it.get_sub_group();
        const int lane = sg.get_local_id();
        const uint16_t * w = r < qrows ? QW + (size_t)r * K : KW + (size_t)(r - qrows) * K;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int k = lane;
        for (; k + 96 < K; k += 128) {
            a0 += qsa_bf2f(w[k]) * x[k];
            a1 += qsa_bf2f(w[k + 32]) * x[k + 32];
            a2 += qsa_bf2f(w[k + 64]) * x[k + 64];
            a3 += qsa_bf2f(w[k + 96]) * x[k + 96];
        }
        for (; k < K; k += 32) {
            a0 += qsa_bf2f(w[k]) * x[k];
        }
        const float s = reduce_over_group(sg, (a0 + a1) + (a2 + a3), sycl::plus<float>());
        if (lane == 0) {
            if (r < qrows) {
                qout[r] = s;
            } else {
                kout[r - qrows] = s;
            }
        }
    });
    // q_norm (RMS over the indexer head) + RoPE, both per head, in the host's
    // order.  id is 128 and inh is 4, so one work-item per head is ample.
    q.parallel_for(range<1>((size_t)inh), [=](sycl::id<1> h) {
        float * p = qout + (size_t)h * id;
        float ss = 0.f;
        for (int i = 0; i < id; i++) {
            ss += p[i] * p[i];
        }
        const float s = 1.f / sycl::sqrt(ss / (float)id + eps);
        for (int i = 0; i < id; i++) {
            p[i] = p[i] * s * q_norm[i];
        }
        const int half = n_rot / 2;
        const float log2_base = sycl::log2(rope_base);
        for (int i = 0; i < half; i++) {
            const float ang = (float)pos * sycl::exp2(-2.0f * i / (float)n_rot * log2_base);
            const float c = sycl::cos(ang), sn = sycl::sin(ang);
            const float x0 = p[i], x1 = p[i + half];
            p[i] = x0 * c - x1 * sn;
            p[i + half] = x0 * sn + x1 * c;
        }
    });
}

// ---------------------------------------------------------------- device selector
// The rank key: larger is a better pool.  Scores are non-negative (a sum of
// rectified dots), so their f32 bit patterns are monotonic; the low 32 bits are
// the inverted pool index, so an equal score prefers the lower index and every
// key is unique.  A radix select can therefore stop at an exact n_top-th key.
inline uint64_t qsa_rank_key(float v, int idx) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = v >= 0.f ? v : 0.f;
    return ((uint64_t)bits.u << 32) | (uint32_t)(0xffffffffu - (uint32_t)idx);
}

void qsa_idx_update_launch(queue & q, const float * kraw, int pos, float * ring, float * pool, const float * k_norm,
                           int id, int kp, int n_rot, float rope_base, float eps) {
    if (id <= 0 || id > 256 || kp <= 1) {
        return;
    }
    const int slot = pos % kp;
    const int pidx = pos / kp;
    q.submit([&](handler & h) {
        local_accessor<float, 1> pl((size_t)id, h);
        h.parallel_for(nd_range<1>((size_t)256, 256), [=](nd_item<1> it) {
            const int d = (int)it.get_local_id(0);
            if (d < id) {
                // The host ring holds f16-rounded f32 values; reproduce that
                // exact rounding so the pool average is bit-identical.
                ring[(size_t)slot * id + d] = ggml_half_to_float(ggml_float_to_half(kraw[d]));
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (d == 0 && slot == kp - 1) {
                const float inv_kp = 1.0f / (float)kp;
                for (int j = 0; j < id; j++) {
                    float acc = ring[j];
                    for (int t = 1; t < kp; t++) {
                        acc += ring[(size_t)t * id + j];
                    }
                    pl[j] = inv_kp * acc;
                }
                float ss = 0.f;
                for (int j = 0; j < id; j++) {
                    ss += pl[j] * pl[j];
                }
                const float s = 1.f / sycl::sqrt(ss / (float)id + eps);
                for (int j = 0; j < id; j++) {
                    pl[j] = pl[j] * s * k_norm[j];
                }
                const int half = n_rot / 2;
                const float log2_base = sycl::log2(rope_base);
                const int rpos = pos - (kp - 1);
                for (int i = 0; i < half; i++) {
                    const float ang = (float)rpos * sycl::exp2(-2.0f * i / (float)n_rot * log2_base);
                    const float c = sycl::cos(ang), sn = sycl::sin(ang);
                    const float x0 = pl[i], x1 = pl[i + half];
                    pl[i] = x0 * c - x1 * sn;
                    pl[i + half] = x0 * sn + x1 * c;
                }
                for (int j = 0; j < id; j++) {
                    pool[(size_t)pidx * id + j] = ggml_half_to_float(ggml_float_to_half(pl[j]));
                }
            }
        });
    });
}

void qsa_score_launch(queue & q, const float * qact, const float * pool, int n_pool, int id, int inh, float inv,
                      float * scores) {
    if (n_pool <= 0 || id <= 0 || inh <= 0) {
        return;
    }
    q.parallel_for(range<1>((size_t)n_pool), [=](sycl::id<1> pi) {
        const int p = (int)pi;
        const float * pl = pool + (size_t)p * id;
        float s = 0.f;
        for (int h = 0; h < inh; h++) {
            const float * qh = qact + (size_t)h * id;
            float d = 0.f;
            for (int k = 0; k < id; k++) {
                d += qh[k] * pl[k];
            }
            s += d > 0.f ? d : 0.f;
        }
        scores[p] = s * inv;
    });
}

// One work-group.  The radix pass is the ONNX TopK's: 8 key bytes, a 256-bin
// SLM histogram, keep the bucket that holds the n_top-th key.  The keys are
// unique, so the cutoff is exact.  A bitonic sort of the <=512 selected pools
// follows, by index, then the expansion into the host selector's `sel` order
// (pools ascending, kp keys each, then the incomplete tail).
void qsa_topk_launch(queue & q, const float * scores, int n_pool, int n_top, int kp, int n_vis, int32_t * pools,
                     int32_t * sel) {
    if (n_pool <= 0 || n_top <= 0 || n_top > 512 || kp <= 1) {
        return;
    }
    if (n_top >= n_pool) {
        q.parallel_for(range<1>((size_t)n_pool), [=](sycl::id<1> i) { pools[i] = (int32_t)i; });
    } else {
        constexpr int WG = 256;
        q.submit([&](handler & h) {
            local_accessor<uint32_t, 1> hist(256, h);
            local_accessor<uint64_t, 1> lprefix(1, h);
            local_accessor<int, 1> lrem(1, h);
            local_accessor<int, 1> lpos(1, h);
            h.parallel_for(nd_range<1>((size_t)WG, WG), [=](nd_item<1> it) {
                const int lane = (int)it.get_local_id(0);
                if (lane == 0) {
                    lprefix[0] = 0;
                    lrem[0] = n_top;
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int shift = 56; shift >= 0; shift -= 8) {
                    for (int i = lane; i < 256; i += WG) {
                        hist[i] = 0;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    const uint64_t mask = shift == 56 ? 0ULL : (~0ULL << (shift + 8));
                    const uint64_t pref = lprefix[0];
                    for (int c = lane; c < n_pool; c += WG) {
                        const uint64_t key = qsa_rank_key(scores[c], c);
                        if ((key & mask) != pref) {
                            continue;
                        }
                        const uint32_t b = (uint32_t)((key >> shift) & 0xff);
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            a(hist[b]);
                        a.fetch_add(1);
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if (lane == 0) {
                        const int rem = lrem[0];
                        int acc = 0, b = -1;
                        for (int xb = 255; xb >= 0; --xb) {
                            acc += (int)hist[xb];
                            if (acc >= rem) {
                                b = xb;
                                break;
                            }
                        }
                        if (b >= 0) {
                            lprefix[0] |= (uint64_t)(uint32_t)b << shift;
                            lrem[0] = rem - (acc - (int)hist[b]);
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                const uint64_t cutoff = lprefix[0];
                if (lane == 0) {
                    lpos[0] = 0;
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int c = lane; c < n_pool; c += WG) {
                    const uint64_t key = qsa_rank_key(scores[c], c);
                    if (key < cutoff) {
                        continue;
                    }
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        a(lpos[0]);
                    const int slot = a.fetch_add(1);
                    if (slot < n_top) {
                        pools[slot] = (int32_t)c;
                    }
                }
            });
        });
    }
    // Sort by index and expand.  n_top <= 512, so one element per lane; the
    // pad is INT_MAX.  The result is exactly the host's `sel`: complete pools
    // ascending, then the incomplete tail.
    q.submit([&](handler & h) {
        local_accessor<int32_t, 1> sk(512, h);
        h.parallel_for(nd_range<1>((size_t)512, 512), [=](nd_item<1> it) {
            const int i = (int)it.get_local_id(0);
            sk[i] = i < n_top ? pools[i] : INT_MAX;
            it.barrier(sycl::access::fence_space::local_space);
            for (int k = 2; k <= 512; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    const int partner = i ^ j;
                    if (partner > i) {
                        const bool asc = ((i & k) == 0);
                        const int32_t a = sk[i], b = sk[partner];
                        if (asc ? (a > b) : (a < b)) {
                            sk[i] = b;
                            sk[partner] = a;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
            }
            const int npool = n_vis / kp;
            const int nsel = n_top * kp + (n_vis - npool * kp);
            for (int j = i; j < nsel; j += 512) {
                sel[j] = j < n_top * kp ? sk[j / kp] * kp + (j % kp) : npool * kp + (j - n_top * kp);
            }
            if (i < n_top) {
                pools[i] = sk[i];
            }
        });
    });
}

void qsa_gather_launch(queue & q, const void * kpool, const void * vpool, const int32_t * tables,
                       const int32_t * sel, int nsel, int n_kv, int head_dim, int block_size, void * ck, void * cv) {
    if (nsel <= 0 || n_kv <= 0 || head_dim <= 0 || block_size <= 0) {
        return;
    }
    const sycl::half * K = (const sycl::half *)kpool;
    const sycl::half * V = (const sycl::half *)vpool;
    sycl::half * CK = (sycl::half *)ck;
    sycl::half * CV = (sycl::half *)cv;
    q.parallel_for(range<1>((size_t)nsel * (size_t)n_kv), [=](id<1> i) {
        const int j = (int)(i / (size_t)n_kv);
        const int kvh = (int)(i % (size_t)n_kv);
        const int sp = sel[j];
        const int sb = tables[sp / block_size];
        const int so = sp % block_size;
        const int db = j / block_size;
        const int dd = j % block_size;
        const size_t unit_src = (size_t)sb * n_kv + kvh;
        const size_t unit_dst = (size_t)db * n_kv + kvh;
        const sycl::half * sk = K + (unit_src * block_size + so) * head_dim;
        const sycl::half * sv = V + (unit_src * block_size + so) * head_dim;
        sycl::half * dk = CK + (unit_dst * block_size + dd) * head_dim;
        sycl::half * dv = CV + (unit_dst * block_size + dd) * head_dim;
        // 8 halves = 16 bytes; the rows are 16-byte aligned in both pools.
        int d = 0;
        for (; d + 8 <= head_dim; d += 8) {
            *reinterpret_cast<sycl::vec<sycl::half, 8> *>(dk + d) =
                *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(sk + d);
            *reinterpret_cast<sycl::vec<sycl::half, 8> *>(dv + d) =
                *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(sv + d);
        }
        for (; d < head_dim; d++) {
            dk[d] = sk[d];
            dv[d] = sv[d];
        }
    });
}

} // namespace si
