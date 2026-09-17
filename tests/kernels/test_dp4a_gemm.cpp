// SIn format validation: GPU dp4a kernels vs a CPU reference built from the
// *same* quantized weights/activations (validates kernel arithmetic exactly),
// plus the total quantization error against the fp32 path (information only).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "gguf.h"
#include "kernels.h"
#include "model.h"
#include "quant.h"
#include "w8.h"

using namespace si;

static inline float h2f(uint16_t h) { return ggml_half_to_float(h); }

// mirror of the kernel-side group expansion, on the CPU
static void unpack_group(uint32_t type, const uint8_t * p, uint8_t * q) {
    const int G = w8_group_size(type);
    const int bits = type == 12 ? 4 : (type == 13 ? 5 : 6);
    for (int h = 0; h < G / 16; h++) {
        const uint8_t * ph = p + h * 8;
        uint8_t * qh = q + h * 16;
        for (int b = 0; b < 8; b++) {
            qh[b] = (uint8_t) (ph[b] & 0xF);
            qh[b + 8] = (uint8_t) (ph[b] >> 4);
        }
        if (bits == 5) {
            const int off = (G == 32) ? (16 + 2 * h) : 8;
            const uint32_t pl = (uint32_t) p[off] | ((uint32_t) p[off + 1] << 8);
            for (int i = 0; i < 16; i++) qh[i] |= (uint8_t) (((pl >> i) & 1) << 4);
        } else if (bits == 6) {
            const uint32_t pl = (uint32_t) p[8] | ((uint32_t) p[9] << 8) |
                                ((uint32_t) p[10] << 16) | ((uint32_t) p[11] << 24);
            for (int i = 0; i < 16; i++) qh[i] |= (uint8_t) (((pl >> (2 * i)) & 3) << 4);
        }
    }
}

static double ref_dot(uint32_t type, int K, const uint8_t * vals, const uint8_t * meta,
                      int melem, const int8_t * x8, const sycl::float2 * xm,
                      const int32_t * xs, int TB, int t, int row) {
    const int G = w8_group_size(type);
    const int GB = w8_group_bytes(type);
    const int MG = K / G;
    const int rb = row / kRB, ri = row % kRB;
    const uint8_t * wp = vals + ((size_t) rb * MG * kRB + ri) * GB;
    const char * mp = (const char *) meta + ((size_t) rb * MG * kRB + ri) * melem;
    double acc = 0;
    for (int g = 0; g < MG; g++) {
        uint8_t q[32];
        unpack_group(type, wp + (size_t) g * kRB * GB, q);
        float sw, mw;
        if (melem == 2) {   // Q6_K scale-only meta: min == 32*scale
            sw = h2f(((const uint16_t *) mp)[(size_t) g * kRB]);
            mw = 32.0f * sw;
        } else {
            const uint32_t m = ((const uint32_t *) mp)[(size_t) g * kRB];
            sw = h2f((uint16_t) (m & 0xFFFF));
            mw = h2f((uint16_t) (m >> 16));
        }
        const int gx = (G == 32) ? g : (g >> 1);
        const float sx = xm[(size_t) gx * TB + t].x();
        int32_t d = 0;
        for (int h = 0; h < G / 16; h++) {
            const int xo = (G == 32) ? h : (g & 1);
            const int8_t * xq = x8 + ((size_t) gx * TB + t) * 32 + xo * 16;
            for (int i = 0; i < 16; i++) d += (int32_t) q[h * 16 + i] * (int32_t) xq[i];
        }
        // both half sums arrive as one int2 per 32-value group
        const sycl::int2 sq = reinterpret_cast<const sycl::int2 *>(xs)[(size_t) gx * TB + t];
        const int32_t c = (G == 32) ? (sq.x() + sq.y()) : ((g & 1) ? sq.y() : sq.x());
        acc += (double) sx * ((double) sw * (double) d - (double) mw * (double) c);
    }
    return acc;
}

// same as run_tensor but with a residual aliased with the output (the engine's
// ssm_out/ffn_down pattern: out == residual == d_x)
static void run_tensor_res(sycl::queue & q, const gguf_tensor_info & ti, const char * name) {
    const int K = (int) ti.dims[0], N = (int) ti.n_rows();
    const int TB = 32;
    printf("test %-32s %-4s K=%-5d N=%-6d (residual) ... ", name, ggml_type_name(ti.type), K, N);

    const size_t vals_bytes = w8_vals_bytes(ti.type, K, N);
    const bool scale_only = w8_q6_scale_only_ok(ti.type, ti.data, K, N);
    const int melem = scale_only ? 2 : 4;
    const size_t meta_bytes = w8_meta_bytes(ti.type, K, N, scale_only);
    std::vector<uint8_t> hv(vals_bytes);
    std::vector<uint8_t> hm(meta_bytes);
    if (!w8_repack(ti.type, ti.data, K, N, hv.data(),
                   reinterpret_cast<uint32_t *>(hm.data()), scale_only)) {
        printf("repack failed\n");
        return;
    }

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.5f);
    std::vector<float> hx((size_t) TB * K);
    for (auto & v : hx) v = nd(rng);

    float * dx = sycl::malloc_device<float>((size_t) TB * K, q);
    int8_t * dx8 = sycl::malloc_device<int8_t>((size_t) TB * K, q);
    sycl::float2 * dxm = sycl::malloc_device<sycl::float2>((size_t) TB * (K / 32), q);
    int32_t * dxs = sycl::malloc_device<int32_t>((size_t) TB * (K / 16), q);
    float * dout = sycl::malloc_device<float>((size_t) TB * N, q);
    uint8_t * dv = sycl::malloc_device<uint8_t>(vals_bytes, q);
    uint8_t * dm = sycl::malloc_device<uint8_t>(meta_bytes, q);
    std::vector<float> hres((size_t) TB * N);
    for (size_t i = 0; i < hres.size(); i++) hres[i] = 0.25f * (float) ((i % 37) - 18);
    q.memcpy(dx, hx.data(), hx.size() * 4);
    q.memcpy(dout, hres.data(), hres.size() * 4);
    q.memcpy(dv, hv.data(), vals_bytes);
    q.memcpy(dm, hm.data(), meta_bytes).wait();

    step_info * info = sycl::malloc_host<step_info>(1, q);
    std::memset(info, 0, sizeof(step_info));
    info->n_real = TB;

    xq_launch(q, dx, nullptr, K, K, dx8, dxm, dxs, info, TB, K);
    w8t w;
    w.vals = dv;
    w.meta = reinterpret_cast<uint32_t *>(dm);
    w.K = K;
    w.N = N;
    w.type = w8_effective_type(ti.type);
    w.meta_elem = melem;
    dp4a_gemm_launch(q, w, dx8, dxm, dxs, dout, N, dout, 1.0f, TB);
    q.wait();

    std::vector<int8_t> hx8((size_t) TB * K);
    std::vector<sycl::float2> hxm((size_t) TB * (K / 32));
    std::vector<int32_t> hxs((size_t) TB * (K / 16));
    std::vector<float> hout((size_t) TB * N);
    q.memcpy(hx8.data(), dx8, hx8.size());
    q.memcpy(hxm.data(), dxm, hxm.size() * 8);
    q.memcpy(hxs.data(), dxs, hxs.size() * 4);
    q.memcpy(hout.data(), dout, hout.size() * 4).wait();

    const int dt = std::max(1, TB / 5);
    const int dr = std::max(1, N / 61);
    double max_rel = 0, max_abs = 0, ref_max = 0;
    for (int t = 0; t < TB; t += dt)
        for (int r = 0; r < N; r += dr) {
            const double ref = ref_dot(w8_effective_type(ti.type), K, hv.data(), hm.data(), melem,
                                       hx8.data(), hxm.data(), hxs.data(), TB, t, r) +
                               (double) hres[(size_t) t * N + r];
            const float got = hout[(size_t) t * N + r];
            const double ad = std::fabs((double) got - ref);
            const double rl = ad / std::max(1e-6, std::fabs(ref));
            if (rl > max_rel) max_rel = rl;
            if (ad > max_abs) max_abs = ad;
            if (std::fabs(ref) > ref_max) ref_max = std::fabs(ref);
        }
    printf("%s  kernel_vs_ref max_rel=%.2e max_abs=%.2e (|ref|max=%.1f)\n",
           max_rel < 1e-3 ? "OK  " : "FAIL", max_rel, max_abs, ref_max);
    sycl::free(dx, q); sycl::free(dx8, q); sycl::free(dxm, q); sycl::free(dxs, q);
    sycl::free(dout, q); sycl::free(dv, q); sycl::free(dm, q);
    sycl::free(info, q);
}

// M-tiled: 256 tokens in one dispatch (mode-2 chunk-batched prefill layout)
static void run_tensor_mt(sycl::queue & q, const gguf_tensor_info & ti, const char * name) {
    const int K = (int) ti.dims[0], N = (int) ti.n_rows();
    const int TB = 256;
    printf("test %-32s %-4s K=%-5d N=%-6d (M-tiled %d tok) ... ", name, ggml_type_name(ti.type), K, N, TB);
    const size_t vals_bytes = w8_vals_bytes(ti.type, K, N);
    const bool scale_only = w8_q6_scale_only_ok(ti.type, ti.data, K, N);
    const int melem = scale_only ? 2 : 4;
    const size_t meta_bytes = w8_meta_bytes(ti.type, K, N, scale_only);
    std::vector<uint8_t> hv(vals_bytes);
    std::vector<uint8_t> hm(meta_bytes);
    if (!w8_repack(ti.type, ti.data, K, N, hv.data(),
                   reinterpret_cast<uint32_t *>(hm.data()), scale_only)) { printf("repack failed\n"); return; }
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 1.5f);
    std::vector<float> hx((size_t) TB * K);
    for (auto & v : hx) v = nd(rng);
    float * dx = sycl::malloc_device<float>((size_t) TB * K, q);
    int8_t * dx8 = sycl::malloc_device<int8_t>((size_t) TB * K, q);
    sycl::float2 * dxm = sycl::malloc_device<sycl::float2>((size_t) TB * (K / 32), q);
    int32_t * dxs = sycl::malloc_device<int32_t>((size_t) TB * (K / 16), q);
    float * dout = sycl::malloc_device<float>((size_t) TB * N, q);
    uint8_t * dv = sycl::malloc_device<uint8_t>(vals_bytes, q);
    uint8_t * dm = sycl::malloc_device<uint8_t>(meta_bytes, q);
    q.memcpy(dx, hx.data(), hx.size() * 4);
    q.memcpy(dv, hv.data(), vals_bytes);
    q.memcpy(dm, hm.data(), meta_bytes).wait();
    step_info * info = sycl::malloc_host<step_info>(1, q);
    std::memset(info, 0, sizeof(step_info));
    info->n_real = 32;
    xq_launch(q, dx, nullptr, K, K, dx8, dxm, dxs, info, TB, K);
    w8t w;
    w.vals = dv; w.meta = reinterpret_cast<uint32_t *>(dm); w.K = K; w.N = N;
    w.type = w8_effective_type(ti.type); w.meta_elem = melem;
    dp4a_gemm_launch(q, w, dx8, dxm, dxs, dout, N, nullptr, 1.0f, TB);
    q.wait();
    double tms = 1e9;
    for (int i = 0; i < 5; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        dp4a_gemm_launch(q, w, dx8, dxm, dxs, dout, N, nullptr, 1.0f, TB);
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        tms = std::min(tms, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    const double wbytes_mt = (double) N * K * 1.25;
    printf("   [M-tiled] %8.3f ms  wbytes*8=%.2f MB -> %.2f GB/s-per-pass\n", tms,
           wbytes_mt * 8 / 1e6, wbytes_mt / tms / 1e6);
    std::vector<int8_t> hx8((size_t) TB * K);
    std::vector<sycl::float2> hxm((size_t) TB * (K / 32));
    std::vector<int32_t> hxs((size_t) TB * (K / 16));
    std::vector<float> hout((size_t) TB * N);
    q.memcpy(hx8.data(), dx8, hx8.size());
    q.memcpy(hxm.data(), dxm, hxm.size() * 8);
    q.memcpy(hxs.data(), dxs, hxs.size() * 4);
    q.memcpy(hout.data(), dout, hout.size() * 4).wait();
    // count how many tokens were quantized (xq must cover all 256)
    long nonzero_x8 = 0;
    for (size_t i = 0; i < hx8.size(); i++) if (hx8[i]) nonzero_x8++;
    double max_rel = 0, max_abs = 0;
    for (int t = 0; t < TB; t += 7)
        for (int r = 0; r < N; r += 61) {
            const double ref = ref_dot(w8_effective_type(ti.type), K, hv.data(), hm.data(), melem,
                                       hx8.data(), hxm.data(), hxs.data(), TB, t, r);
            const double ad = std::fabs((double) hout[(size_t) t * N + r] - ref);
            const double rl = ad / std::max(1e-6, std::fabs(ref));
            if (rl > max_rel) max_rel = rl;
            if (ad > max_abs) max_abs = ad;
        }
    printf("%s max_rel=%.2e max_abs=%.2e x8_nonzero=%ld/%zu\n", max_rel < 1e-3 ? "OK  " : "FAIL",
           max_rel, max_abs, nonzero_x8, hx8.size());
    sycl::free(dx, q); sycl::free(dx8, q); sycl::free(dxm, q); sycl::free(dxs, q);
    sycl::free(dout, q); sycl::free(dv, q); sycl::free(dm, q); sycl::free(info, q);
}

static void run_tensor(sycl::queue & q, const gguf_tensor_info & ti, const char * name) {
    const int K = (int) ti.dims[0], N = (int) ti.n_rows();
    const int TB = 32;
    printf("test %-32s %-4s K=%-5d N=%-6d ... ", name, ggml_type_name(ti.type), K, N);

    // host-side repack
    const size_t vals_bytes = w8_vals_bytes(ti.type, K, N);
    const bool scale_only = w8_q6_scale_only_ok(ti.type, ti.data, K, N);
    const int melem = scale_only ? 2 : 4;
    const size_t meta_bytes = w8_meta_bytes(ti.type, K, N, scale_only);
    std::vector<uint8_t> hv(vals_bytes);
    std::vector<uint8_t> hm(meta_bytes);
    if (!w8_repack(ti.type, ti.data, K, N, hv.data(),
                   reinterpret_cast<uint32_t *>(hm.data()), scale_only)) {
        printf("repack failed\n");
        return;
    }

    // inputs
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.5f);
    std::vector<float> hx((size_t) TB * K);
    for (auto & v : hx) v = nd(rng);

    const bool use_up = strstr(name, "ffn_down") != nullptr;
    std::vector<float> hup((size_t) TB * K);
    for (auto & v : hup) v = nd(rng) * 0.5f + 1.f;

    float * dx = sycl::malloc_device<float>((size_t) TB * K, q);
    float * dup = use_up ? sycl::malloc_device<float>((size_t) TB * K, q) : nullptr;
    int8_t * dx8 = sycl::malloc_device<int8_t>((size_t) TB * K, q);
    sycl::float2 * dxm = sycl::malloc_device<sycl::float2>((size_t) TB * (K / 32), q);
    int32_t * dxs = sycl::malloc_device<int32_t>((size_t) TB * (K / 16), q);
    float * dout = sycl::malloc_device<float>((size_t) TB * N, q);
    uint8_t * dv = sycl::malloc_device<uint8_t>(vals_bytes, q);
    uint8_t * dm = sycl::malloc_device<uint8_t>(meta_bytes, q);
    q.memcpy(dx, hx.data(), hx.size() * 4);
    if (use_up) q.memcpy(dup, hup.data(), hup.size() * 4);
    q.memcpy(dv, hv.data(), vals_bytes);
    q.memcpy(dm, hm.data(), meta_bytes).wait();

    step_info * info = sycl::malloc_host<step_info>(1, q);
    std::memset(info, 0, sizeof(step_info));
    info->n_real = TB;

    xq_launch(q, dx, dup, K, K, dx8, dxm, dxs, info, TB, K);
    w8t w;
    w.vals = dv;
    w.meta = reinterpret_cast<uint32_t *>(dm);
    w.K = K;
    w.N = N;
    w.type = w8_effective_type(ti.type);
    w.meta_elem = melem;
    dp4a_gemm_launch(q, w, dx8, dxm, dxs, dout, N, nullptr, 1.0f, TB);
    q.wait();

    // copy quantized activations and output back
    std::vector<int8_t> hx8((size_t) TB * K);
    std::vector<sycl::float2> hxm((size_t) TB * (K / 32));
    std::vector<int32_t> hxs((size_t) TB * (K / 16));
    std::vector<float> hout((size_t) TB * N);
    q.memcpy(hx8.data(), dx8, hx8.size());
    q.memcpy(hxm.data(), dxm, hxm.size() * 8);
    q.memcpy(hxs.data(), dxs, hxs.size() * 4);
    q.memcpy(hout.data(), dout, hout.size() * 4).wait();

    // kernel vs CPU reference over a sample
    const int dt = std::max(1, TB / 5);
    const int dr = std::max(1, N / 61);
    double max_rel = 0, max_abs = 0, ref_max = 0;
    for (int t = 0; t < TB; t += dt)
        for (int r = 0; r < N; r += dr) {
            const double ref = ref_dot(w8_effective_type(ti.type), K, hv.data(), hm.data(), melem, hx8.data(), hxm.data(),
                                       hxs.data(), TB, t, r);
            const float got = hout[(size_t) t * N + r];
            const double ad = std::fabs((double) got - ref);
            const double rl = ad / std::max(1e-6, std::fabs(ref));
            if (rl > max_rel) max_rel = rl;
            if (ad > max_abs) max_abs = ad;
            if (std::fabs(ref) > ref_max) ref_max = std::fabs(ref);
        }

    // quantization error vs the exact fp32 path (information only)
    double qmax_rel = 0, qsum = 0;
    long qn = 0;
    for (int t = 0; t < TB; t += dt)
        for (int r = 0; r < N; r += dr) {
            double fref = 0;
            std::vector<float> rowf(K);
            dequantize_row(ti.type, (const char *) ti.data + (size_t) r * quant_row_bytes(ti.type, K),
                           rowf.data(), K);
            const float * xr = hx.data() + (size_t) t * K;
            for (int k = 0; k < K; k++) {
                float xv = xr[k];
                if (use_up) xv = xv / (1.0f + std::exp(-xv)) * hup[(size_t) t * K + k];
                fref += (double) rowf[k] * xv;
            }
            const double ad = std::fabs((double) hout[(size_t) t * N + r] - fref);
            qmax_rel = std::max(qmax_rel, ad / std::max(1e-6, std::fabs(fref)));
            qsum += ad / std::max(1e-6, std::fabs(fref));
            qn++;
        }

    const bool ok = max_abs < 5e-3 * ref_max + 5e-3;
    printf("%s  kernel_vs_ref max_rel=%.2e max_abs=%.2e (|ref|max=%.1f) | quant_err mean_rel=%.2e max_rel=%.2e\n",
           ok ? "OK  " : "FAIL", max_rel, max_abs, ref_max, qsum / std::max(1L, qn), qmax_rel);

    // timing
    const int IT = 20;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < IT; i++) {
        xq_launch(q, dx, dup, K, K, dx8, dxm, dxs, info, TB, K);
        dp4a_gemm_launch(q, w, dx8, dxm, dxs, dout, N, nullptr, 1.0f, TB);
    }
    q.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / IT;
    const double bytes = (double) vals_bytes + (double) meta_bytes;
    printf("   [32 tok] %8.3f ms  wtraf %5.2f GB/s  %7.1f GF/s  %6.1f tok/s-equiv\n", ms,
           bytes / ms / 1e6, 2.0 * 32.0 * N * K / ms / 1e9, 32 * 1000.0 / ms);

    sycl::free(dx, q);
    if (dup) sycl::free(dup, q);
    sycl::free(dx8, q);
    sycl::free(dxm, q);
    sycl::free(dxs, q);
    sycl::free(dout, q);
    sycl::free(dv, q);
    sycl::free(dm, q);
    sycl::free(info, q);
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    sycl::queue q(sycl::gpu_selector_v);
    printf("dev: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    model m;
    m.load(path);
    run_tensor(q, *m.gguf.find("blk.0.ffn_gate.weight"), "blk.0.ffn_gate.weight");
    run_tensor(q, *m.gguf.find("blk.0.attn_qkv.weight"), "blk.0.attn_qkv.weight");
    run_tensor(q, *m.gguf.find("blk.0.ffn_down.weight"), "blk.0.ffn_down.weight");
    run_tensor(q, *m.gguf.find("blk.3.attn_q.weight"), "blk.3.attn_q.weight");
    run_tensor(q, *m.gguf.find("token_embd.weight"), "token_embd.weight");
    run_tensor_res(q, *m.gguf.find("blk.0.ssm_out.weight"), "blk.0.ssm_out.weight");
    run_tensor_res(q, *m.gguf.find("blk.0.attn_qkv.weight"), "blk.0.attn_qkv.weight");
    run_tensor_mt(q, *m.gguf.find("blk.0.ffn_gate.weight"), "blk.0.ffn_gate.weight");
    run_tensor_mt(q, *m.gguf.find("blk.0.attn_qkv.weight"), "blk.0.attn_qkv.weight");
    run_tensor_mt(q, *m.gguf.find("blk.0.ffn_down.weight"), "blk.0.ffn_down.weight");
    return 0;
}
