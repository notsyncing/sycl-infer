// CPU quantized GEMV/GEMM (mirror of src/backend/gpu/kernels/gemv.cpp).
//
// One job per (token, output row): the row's weight blocks are dequantized and
// dotted by gemv_row.  All segments of a call group are flattened into a single
// pool barrier (segments share the token count, only the row count differs), so
// a group of e.g. gate+up costs one parallel region instead of two.
// Segment output is token-major (out[t*out_stride + row]), like the SYCL kernel.
#include "common.h"

namespace si {

void cpu_gemv_group(uint32_t type, const cpu_gemv_seg * segs, int n_segs, int total_rows, int TB, int nsb,
                    int n_tok_blocks, int n_threads, int chunk) {
    (void)total_rows;
    (void)n_threads;
    (void)nsb;
    if (n_segs <= 0) {
        return;
    }
    const int ntok = (TB > 1 && n_tok_blocks > 0) ? TB * n_tok_blocks : TB;
    const int step = sb_step(type);
    const int bytes = sb_bytes(type);

    auto run_row = [&](const cpu_gemv_seg & sg, int t, int r) {
        const int K = sg.K;
        const int row_bytes = (K / step) * bytes;
        const char * W = (const char *)sg.w;
        const float * x = sg.x + (size_t)t * sg.x_stride;
        const float * up = sg.act_up ? sg.act_up + (size_t)t * sg.x_stride : nullptr;
        float * out = sg.out + (size_t)t * sg.out_stride;
        const float * res = sg.residual ? sg.residual + (size_t)t * sg.out_stride + r : nullptr;
        gemv_row(type, W + (size_t)r * row_bytes, x, up, K, sg.alpha, res, out + r);
    };

    // `chunk` > 1 turns each pool job into a contiguous run of flat indices.
    // Every row still goes through the same run_row(...) with the same
    // arguments, so the arithmetic and the output bytes are unchanged; the
    // only difference is that the pool's `next_` atomic is taken once per
    // chunk instead of once per row (12800 rows per PLE projection call).
    auto dispatch = [&](int njobs, auto && fn) {
        if (njobs <= 0) {
            return;
        }
        if (chunk <= 1) {
            par(njobs, fn);
            return;
        }
        const int nc = (njobs + chunk - 1) / chunk;
        par(nc, [&](int c) {
            const int j0 = c * chunk;
            const int j1 = j0 + chunk < njobs ? j0 + chunk : njobs;
            for (int j = j0; j < j1; j++) {
                fn(j);
            }
        });
    };

    if (n_segs > kCpuGemvMaxSegs) {
        // unlikely (groups are per type/nsb per layer): keep the old per-group
        // barriers rather than overflow the prefix table
        for (int s = 0; s < n_segs; s++) {
            dispatch(ntok * segs[s].n_rows,
                     [&](int j) { run_row(segs[s], j / segs[s].n_rows, j % segs[s].n_rows); });
        }
        return;
    }
    // flat job index: token-major, then the concatenated segment rows
    int pref[kCpuGemvMaxSegs];
    int total = 0;
    for (int s = 0; s < n_segs; s++) {
        pref[s] = total;
        total += segs[s].n_rows;
    }
    if (total <= 0 || ntok <= 0) {
        return;
    }
    dispatch(ntok * total, [&](int j) {
        const int t = j / total;
        const int rr = j % total;
        int s = 0;
        while (s + 1 < n_segs && rr >= pref[s + 1]) {
            s++;
        }
        run_row(segs[s], t, rr - pref[s]);
    });
}

} // namespace si
