#include "kernels.h"
#include "kernel_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace si {

using namespace sycl;
using namespace si::kd;

// ---------------------------------------------------------------------------
void embed_launch(queue & q, const void * table, uint32_t type, const step_info * info, float * out, int n_embd,
                  size_t row_bytes) {
    // n_embd is not necessarily a multiple of 256: Qwen2-VL's 896 is 3.5
    // superblocks, and truncating to 3 left the last 128 embedding values
    // unwritten.  The partial superblock is a prefix of the block types whose
    // group is 32 (Q8_0/Q5_0/Q5_1), so the guard below is all it needs.
    const int n_sb = (n_embd + 255) / 256;
    const int sb_bytes = superblock_bytes(type);
    // fixed grid (kMaxB*kMaxT rows): the row count is device-side state, so a
    // host-computed extent would be captured as zero in a command graph
    q.parallel_for(nd_range<1>((size_t)kMaxB * kMaxT * n_sb * 256, 256), [=](nd_item<1> it) {
        const int g = it.get_group(0);
        const int t = g / n_sb;
        const int sb = g % n_sb;
        const int tid = it.get_local_id(0);
        // per-row real count: a chunk-batched row may be partial
        const int row = t / info->tpb;
        if (row >= info->n_rows || (t - row * info->tpb) >= row_nr(info, row)) {
            return;
        }
        const int col = sb * 256 + tid;
        if (col >= n_embd) {
            return; // partial tail superblock
        }
        // multimodal: tokens whose embedding was produced by the vision encoder
        // are copied verbatim instead of looked up in the token table
        const int ir = info->img_row[t];
        if (info->img_embd && ir >= 0) {
            out[(size_t)t * n_embd + sb * 256 + tid] = info->img_embd[(size_t)ir * n_embd + sb * 256 + tid];
            return;
        }
        const int tok = info->tokens[t];
        const char * rowp = (const char *)table + (size_t)tok * row_bytes + (size_t)sb * sb_bytes;
        out[(size_t)t * n_embd + sb * 256 + tid] = dequant_elem_sb(type, rowp, tid);
    });
}

} // namespace si
