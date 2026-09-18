// CPU token embedding (mirror of src/backend/gpu/kernels/embed.cpp).  Image
// tokens are copied verbatim from the vision embeddings; everything else is
// dequantized from the row of the token table (quant.h host reference).
#include "common.h"

namespace si {

void cpu_embed(const void * table, uint32_t type, const cpu_step_info * info, float * out, int n_embd,
               size_t row_bytes) {
    const int ntok = info->n_rows * info->n_real;
    par(ntok, [&](int t) {
        const int ir = info->img_row[t];
        if (info->img_embd && ir >= 0) {
            std::memcpy(out + (size_t)t * n_embd, info->img_embd + (size_t)ir * n_embd, (size_t)n_embd * 4);
            return;
        }
        const int tok = info->tokens[t];
        dequantize_row(type, (const char *)table + (size_t)tok * row_bytes, out + (size_t)t * n_embd, n_embd);
    });
}

} // namespace si
