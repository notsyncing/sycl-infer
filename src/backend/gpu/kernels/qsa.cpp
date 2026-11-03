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

#include "kernels.h"

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
