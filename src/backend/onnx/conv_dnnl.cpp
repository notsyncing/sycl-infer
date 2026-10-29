// The oneDNN convolution path of the ONNX runtime.
//
// The SYCL conv kernel is register-tiled and word-parallel, and measured
// against oneDNN on the layout model's own shapes it is 1.3-4.5x slower on
// the deep layers (which are 90% of the 157 GFLOP): 3x3 C256 100x100 is
// 11.8 GFLOP a call, and the model has three of them.  oneDNN's JIT chooses
// a blocked layout and a vectorised loop for the shape, and the device's
// XMX is not involved in either - this is the f32 path.
//
// The two constraints that make the integration small:
//
//   * The weights stay in the runtime's own layout (C, kh, kw, M): a
//     one-time reorder into the primitive's layout is cached per weight
//     pointer, because a per-execute reorder would read and write the whole
//     weight for every token.
//   * The epilogue (bias, the folded batch norm's scale/add, the activation)
//     is a separate small kernel over the output.  oneDNN's post-ops could
//     fuse it, but the output pass is ~0.05 ms on the biggest layer and a
//     separate kernel keeps the semantics identical to the SYCL conv.
#include "conv_dnnl.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

namespace si {
namespace onnx {

namespace {

using namespace dnnl;

struct dev_ctx {
    sycl::device dev;
    engine eng;
    stream st;
    // The one scratchpad, reused by every convolution on this device.  It
    // used to be one per (weight, input shape): the UNET alone has 66 convs
    // and the table pipeline presents a new input shape per table, so the
    // cached scratchpads grew to gigabytes.  The executes are serialised
    // (g_mtx, and the queue is in order), so one buffer is enough.
    void * scratch = nullptr;
    size_t scratch_sz = 0;
};

struct conv_entry {
    convolution_forward::primitive_desc pd;
    void * w_re = nullptr; // shared with the other shapes of the same weight
};

std::mutex g_mtx;
std::vector<std::unique_ptr<dev_ctx>> g_devs;
std::vector<std::unique_ptr<conv_entry>> g_cache;
// A parallel key list: the entry index is the cache index, and the key says
// which call it belongs to.
std::vector<std::string> g_keys;
// The reordered weights, keyed by the weight pointer and the *destination*
// layout: the primitive's weight layout does not depend on the batch, so the
// same reorder serves every input shape (and every batch size).
std::map<std::string, void *> g_wbuf;
// The batch-1 matmul primitives, by (M, K, N).
std::map<std::string, matmul::primitive_desc> g_mm;

thread_local bool g_enabled = true;
size_t g_wbytes = 0;
size_t g_sbytes = 0;

dev_ctx * ctx_for(sycl::queue & q) {
    for (auto & c : g_devs) {
        if (c->dev == q.get_device()) return c.get();
    }
    auto c = std::make_unique<dev_ctx>();
    c->dev = q.get_device();
    c->eng = sycl_interop::make_engine(q.get_device(), q.get_context());
    c->st = sycl_interop::make_stream(c->eng, q);
    dev_ctx * p = c.get();
    g_devs.push_back(std::move(c));
    return p;
}

std::string make_key(const void * w, int N, int C, int H, int W, int M, int kh, int kw, int sh,
                     int sw, int ph, int pw, int dh, int dw) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%p:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d", w, N, C, H, W, M, kh, kw,
             sh, sw, ph, pw, dh, dw);
    return std::string(buf);
}

} // namespace

void conv_dnnl_set_enabled(bool on) { g_enabled = on; }
bool conv_dnnl_enabled() { return g_enabled; }

bool matmul_dnnl_launch(sycl::queue & q, const float * a, const float * b, float * y, int64_t M,
                        int64_t K, int64_t N) {
    if (!g_enabled) return false;
    if (M <= 0 || K <= 0 || N <= 0) return false;
    // oneDNN's per-call cost (a mutex, three memory descriptors and a stream
    // submission) is ~30 us on these cards; for the table decoder's tiny
    // matrices the plain SYCL kernel is cheaper.  The layout model's 384M
    // MAC bmm and the convolutions are far above any threshold.  A minimum
    // op count of 0 keeps every matmul on oneDNN (the old behaviour).
    static const long long mm_min = [] {
        const char * e = getenv("PF_ONNX_MM_DNNL_MIN");
        return e ? atoll(e) : 0;
    }();
    if (M * K * N < mm_min) return false;
    std::lock_guard<std::mutex> lock(g_mtx);
    try {
        dev_ctx * ctx = ctx_for(q);
        char key[112];
        snprintf(key, sizeof(key), "%lld:%lld:%lld", (long long)M, (long long)K, (long long)N);
        auto it = g_mm.find(key);
        if (it == g_mm.end()) {
            auto src_md = memory::desc({M, K}, memory::data_type::f32, memory::format_tag::ab);
            auto w_md = memory::desc({K, N}, memory::data_type::f32, memory::format_tag::ab);
            auto dst_md = memory::desc({M, N}, memory::data_type::f32, memory::format_tag::ab);
            matmul::primitive_desc pd(ctx->eng, src_md, w_md, dst_md);
            it = g_mm.emplace(key, pd).first;
        }
        memory src_m(memory::desc({M, K}, memory::data_type::f32, memory::format_tag::ab), ctx->eng,
                     const_cast<float *>(a));
        memory w_m(memory::desc({K, N}, memory::data_type::f32, memory::format_tag::ab), ctx->eng,
                   const_cast<float *>(b));
        memory dst_m(memory::desc({M, N}, memory::data_type::f32, memory::format_tag::ab), ctx->eng,
                     y);
        matmul(it->second).execute(ctx->st, {{DNNL_ARG_SRC, src_m},
                                             {DNNL_ARG_WEIGHTS, w_m},
                                             {DNNL_ARG_DST, dst_m}});
    } catch (const std::exception & ex) {
        static bool said = false;
        if (!said) {
            fprintf(stderr, "[onnx] oneDNN matmul fell back for %lldx%lldx%lld: %s\n", (long long)M,
                    (long long)K, (long long)N, ex.what());
            said = true;
        }
        return false;
    }
    return true;
}

namespace {
// The transposed 1x1 weights, keyed by the weight pointer.  A weight lives
// for the process, so the entry does too (and the buffer is device memory
// owned by this cache, not by the executor's pool).
std::map<const void *, void *> g_wt;
} // namespace

bool conv1x1_matmul_launch(sycl::queue & q, const float * w, const float * x, float * y, int N,
                           int M, int C, int HW) {
    if (!g_enabled) return false;
    if (N <= 0 || M <= 0 || C <= 0 || HW <= 0) return false;
    // Only when the matmul is worth its per-image call: a few large images
    // amortise the primitive and the transposition, a batch of one does not.
    const float * wt = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        auto it = g_wt.find(w);
        if (it == g_wt.end()) {
            const size_t bytes = (size_t)M * C * sizeof(float);
            void * wt = sycl::malloc_device(bytes, q);
            if (!wt) return false;
            const float * src = w;
            float * dst = (float *)wt;
            const int mm = M, cc = C;
            q.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>((size_t)M * C), [=](sycl::item<1> it2) {
                    const size_t i = it2.get_id(0);
                    const int m = (int)(i / cc);
                    const int c = (int)(i % cc);
                    dst[i] = src[(size_t)c * mm + m];
                });
            });
            it = g_wt.emplace(w, wt).first;
            g_wbytes += bytes;
        }
        wt = (const float *)it->second;
    }
    for (int n = 0; n < N; ++n) {
        if (!matmul_dnnl_launch(q, wt, x + (size_t)n * C * HW, y + (size_t)n * M * HW, M, C, HW)) {
            return false;
        }
    }
    return true;
}

void conv_dnnl_mem(size_t * weights, size_t * scratch) {
    if (weights) *weights = g_wbytes;
    if (scratch) *scratch = g_sbytes;
}

bool conv_dnnl_launch(sycl::queue & q, const float * x, const float * w, float * y,
                      int N, int C, int H, int W, int M, int Ho, int Wo, int kh, int kw,
                      int sh, int sw, int ph, int pw, int dh, int dw) {
    if (!g_enabled) return false;
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || M <= 0 || Ho <= 0 || Wo <= 0) return false;
    if (kh <= 0 || kw <= 0) return false;

    const std::string key = make_key(w, N, C, H, W, M, kh, kw, sh, sw, ph, pw, dh, dw);
    std::lock_guard<std::mutex> lock(g_mtx);
    conv_entry * e = nullptr;
    for (size_t i = 0; i < g_keys.size(); ++i) {
        if (g_keys[i] == key) {
            e = g_cache[i].get();
            break;
        }
    }
    dev_ctx * ctx = nullptr;
    try {
        ctx = ctx_for(q);
        if (!e) {
            // The primitive: NCHW in and out, the weights' layout is chosen
            // by oneDNN (`any`), and the convolution is direct (the shapes
            // here are large enough that the direct algorithm is the right
            // one; winograd is not in this oneDNN's GPU JIT).
            memory::dims src_d{N, C, H, W};
            memory::dims dst_d{N, M, Ho, Wo};
            memory::dims w_d{M, C, kh, kw};
            auto src_md = memory::desc(src_d, memory::data_type::f32, memory::format_tag::nchw);
            auto dst_md = memory::desc(dst_d, memory::data_type::f32, memory::format_tag::nchw);
            auto w_any = memory::desc(w_d, memory::data_type::f32, memory::format_tag::any);
            // PF_ONNX_DNNL_BF16=1 asks oneDNN for bf16 math on the f32
            // descriptors (f32 accumulate): the XMX path.  It is an A/B for
            // the accuracy gate, not the default.
            primitive_attr pattr;
            const char * bf = getenv("PF_ONNX_DNNL_BF16");
            if (bf && atoi(bf) != 0) pattr.set_fpmath_mode(fpmath_mode::bf16);
            convolution_forward::primitive_desc pd(
                ctx->eng, prop_kind::forward_inference, algorithm::convolution_direct, src_md,
                w_any, dst_md, memory::dims{sh, sw}, memory::dims{dh, dw},
                memory::dims{ph, pw}, memory::dims{ph, pw}, pattr);

            auto entry = std::make_unique<conv_entry>(conv_entry{pd, nullptr});
            // The runtime's weight layout: (C, kh, kw, M) contiguous, which
            // as a function of (m, c, r, s) is
            //   c*kh*kw*M + r*kw*M + s*M + m,
            // i.e. strides (1, kh*kw*M, kw*M, M).  Dimensions are the
            // ONNX/oneDNN order (M, C, kh, kw).
            const memory::desc w_user(w_d, memory::data_type::f32,
                                      memory::dims{1, kh * kw * M, kw * M, M});
            // The reorder is shared by every shape of this weight: the key
            // is the weight pointer, its own geometry and the destination
            // layout - not the batch size or the spatial size.
            char wb[128];
            snprintf(wb, sizeof(wb), "%p:%d:%d:%d:%d:%d:%d", w, C, M, kh, kw, dh, dw);
            std::string wkey = wb;
            wkey += "|";
            for (int64_t d : pd.weights_desc().get_dims()) wkey += std::to_string(d) + ",";
            wkey += "|";
            for (int64_t s : pd.weights_desc().get_strides()) wkey += std::to_string(s) + ",";
            auto wit = g_wbuf.find(wkey);
            if (wit == g_wbuf.end()) {
                void * buf = sycl::malloc_device(pd.weights_desc().get_size(), q);
                if (!buf) return false;
                memory w_user_m(w_user, ctx->eng, const_cast<float *>(w));
                memory w_re_m(pd.weights_desc(), ctx->eng, buf);
                reorder(w_user_m, w_re_m).execute(ctx->st, w_user_m, w_re_m);
                wit = g_wbuf.emplace(wkey, buf).first;
                g_wbytes += pd.weights_desc().get_size();
            }
            entry->w_re = wit->second;
            e = entry.get();
            g_keys.push_back(key);
            g_cache.push_back(std::move(entry));
        } else {
            ctx = ctx_for(q);
        }

        memory src_m(memory::desc({N, C, H, W}, memory::data_type::f32, memory::format_tag::nchw),
                     ctx->eng, const_cast<float *>(x));
        memory dst_m(memory::desc({N, M, Ho, Wo}, memory::data_type::f32, memory::format_tag::nchw),
                     ctx->eng, y);
        memory w_re_m(e->pd.weights_desc(), ctx->eng, e->w_re);
        std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, src_m},
                                                {DNNL_ARG_WEIGHTS, w_re_m},
                                                {DNNL_ARG_DST, dst_m}};
        const size_t scratch_sz = e->pd.scratchpad_desc().get_size();
        if (scratch_sz) {
            if (scratch_sz > ctx->scratch_sz) {
                // A still-queued execute may hold the old buffer.
                if (ctx->scratch) {
                    q.wait();
                    sycl::free(ctx->scratch, q);
                }
                ctx->scratch = sycl::malloc_device(scratch_sz, q);
                ctx->scratch_sz = ctx->scratch ? scratch_sz : 0;
                if (!ctx->scratch) return false;
            }
            args.insert({DNNL_ARG_SCRATCHPAD,
                         memory(e->pd.scratchpad_desc(), ctx->eng, ctx->scratch)});
        }
        convolution_forward(e->pd).execute(ctx->st, args);
    } catch (const std::exception & ex) {
        // A shape or layout oneDNN refuses is not an error: the SYCL kernel
        // handles it.  Say it once, so a silent fallback is still visible in
        // the log.
        static bool said = false;
        if (!said) {
            fprintf(stderr, "[onnx] oneDNN conv fell back for C%d %dx%d -> M%d: %s\n", C, H, W, M,
                    ex.what());
            said = true;
        }
        return false;
    }
    return true;
}


} // namespace onnx
} // namespace si
