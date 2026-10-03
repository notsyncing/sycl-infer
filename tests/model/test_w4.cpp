// Verifies the w4 native-width packing:
//   1. round-trip: unpack(pack(Q4_K)) == dequantize_row(Q4_K) to within the
//      f16 metadata rounding (expected ~0.077% relative L2, against 0.98% for
//      the int8 conversion it replaces);
//   2. the prefill identity used by the oneDNN path:
//        sum_k w[k]*x[k] == sum_g step_g*sum_{k in g} q[k]*x[k]
//                          + sum_g offset_g*XS[g]
//      with XS[g] = sum_{k in g} x[k].  This is what the grouped-scale matmul
//      plus the correction term compute, so it proves the prefill math without
//      needing the GPU.
#include "gguf.h"
#include "quant.h"
#include "w4.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static float h16(uint16_t h) {
    return ggml_half_to_float(h);
}

int main(int argc, char ** argv) {
    // CPU-only packing round-trip: shape-independent, so it defaults to the 0.8B
    // like the other CPU tests (the 27B works too - pass it as argv[1]).
    const char * path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    gguf_file f;
    f.load(path);

    std::mt19937 rng(1234);
    std::vector<float> ref, got;
    double sum_rt = 0, worst_rt = 0;
    double sum_id = 0, worst_id = 0;
    int rows = 0, idrows = 0, tensors = 0;

    for (auto & t : f.tensors) {
        if (!si::w4_supported(t.type) || t.dims.empty() || t.dims[0] % QK_K != 0) {
            continue;
        }
        const int K = (int)t.dims[0];
        int N = 1;
        for (size_t i = 1; i < t.dims.size(); i++) {
            N *= (int)t.dims[i];
        }
        const char * base = (const char *)f.map_base + f.data_offset + t.offset;
        si::w4t w;
        if (!si::w4_pack(t.type, base, K, N, w)) {
            printf("pack failed on %s\n", t.name.c_str());
            return 1;
        }
        tensors++;
        const size_t rb = ggml_row_bytes(t.type, K);
        const int ng = K / si::kW4Group;
        for (int r = 0; r < N && r < 4; r++) {
            ref.assign(K, 0.f);
            dequantize_row(t.type, base + (size_t)r * rb, ref.data(), K);
            // ---- 1. round-trip
            got.assign(K, 0.f);
            for (int k = 0; k < K; k++) {
                const size_t idx = (size_t)r * K + k;
                const int q = (w.vals[idx >> 1] >> ((idx & 1) * 4)) & 0xF;
                const size_t gi = (size_t)(k / si::kW4Group) * N + r;
                got[k] = h16(w.off[gi]) + h16(w.scale[gi]) * (float)q;
            }
            double num = 0, den = 0;
            for (int k = 0; k < K; k++) {
                const double d = (double)got[k] - (double)ref[k];
                num += d * d;
                den += (double)ref[k] * (double)ref[k];
            }
            const double e = den > 0 ? std::sqrt(num / den) : 0.0;
            sum_rt += e;
            worst_rt = std::max(worst_rt, e);
            rows++;

            // ---- 2. prefill identity with a random int8 activation row
            if ((r % 2) == 0) {
                std::vector<int8_t> x((size_t)K);
                for (auto & v : x) {
                    v = (int8_t)((int)(rng() % 255) - 127);
                }
                double lhs = 0, l1 = 0, term1 = 0, corr = 0;
                for (int k = 0; k < K; k++) {
                    lhs += (double)ref[k] * (double)x[k];
                    l1 += std::fabs((double)ref[k] * (double)x[k]);
                }
                for (int g = 0; g < ng; g++) {
                    const size_t gi = (size_t)g * N + r;
                    const float step = h16(w.scale[gi]);
                    const float off = h16(w.off[gi]);
                    double qdot = 0, xs = 0;
                    for (int k = g * si::kW4Group; k < (g + 1) * si::kW4Group; k++) {
                        const size_t idx = (size_t)r * K + k;
                        const int q = (w.vals[idx >> 1] >> ((idx & 1) * 4)) & 0xF;
                        qdot += (double)q * (double)x[k];
                        xs += (double)x[k];
                    }
                    term1 += (double)step * qdot;
                    corr += (double)off * xs;
                }
                const double rhs = term1 + corr;
                // normalize by the L1 sum: the residual is the f16 metadata
                // rounding, and |sum ref*x| can cancel to near zero
                const double e = std::fabs(rhs - lhs) / std::fmax(1.0, l1);
                sum_id += e;
                worst_id = std::max(worst_id, e);
                idrows++;
            }
        }
    }

    printf("Q4_K tensors checked : %d\n", tensors);
    printf("round-trip rel L2    : mean=%.6f  worst=%.6f  (%d rows)\n", sum_rt / std::max(1, rows), worst_rt, rows);
    printf("prefill identity rel : mean=%.2e  worst=%.2e  (%d rows)\n", sum_id / std::max(1, idrows), worst_id, idrows);
    const bool ok = worst_rt < 0.002 && worst_id < 0.01;
    printf("%s\n", ok ? "w4 pack OK" : "w4 pack FAILED");
    return ok ? 0 : 1;
}
