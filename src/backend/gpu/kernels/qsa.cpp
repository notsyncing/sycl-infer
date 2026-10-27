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
