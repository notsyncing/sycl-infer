#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace si {

// ---------------------------------------------------------------------------
static bool g_capturing = false;
struct capture_guard {
    capture_guard() {
        g_capturing = true;
    }
    ~capture_guard() {
        g_capturing = false;
    }
};
static double prof_ci_t[256];
static long prof_ci_n[256];
static bool prof_on() {
    static bool p = getenv("PF_PROF") != nullptr && getenv("PF_NOGRAPH") != nullptr;
    return p && !g_capturing;
}
static double prof_acc[16];
static long prof_calls = 0;

// row-offset copy of a plan segment: all token-major pointers move by
// row * tpb, the quantized activation views by row * (K*32) bytes
static gemv_seg row_offset_seg(const gemv_seg & s, int r, int tpb) {
    gemv_seg c = s;
    c.dev = s.dev;
    const size_t off = (size_t)r * tpb;
    c.x = s.x + off * s.x_stride;
    if (s.act_up) {
        c.act_up = s.act_up + off * s.x_stride;
    }
    c.out = s.out + off * s.out_stride;
    if (s.residual) {
        c.residual = s.residual + off * s.out_stride;
    }
    if (s.w8.vals) {
        c.x8 = s.x8 + (size_t)r * (size_t)s.w8.K * 32;
        c.xmeta = s.xmeta + (size_t)r * (s.w8.K / 32);
        c.xsumq = s.xsumq + (size_t)r * (s.w8.K / 16);
    }
    return c;
}
// ---------------------------------------------------------------------------
seg_plan engine::build_plan(int T, int tb, bool head_batched, bool use_w8, bool with_head) {
    seg_plan plan;
    const hparams & hp = m.hp;
    const int n_slices = use_w8 ? 1 : (T + tb - 1) / tb;
    auto add_sliced = [&](gemv_seg s) {
        for (int sl = 0; sl < n_slices; sl++) {
            gemv_seg c = s;
            c.x = s.x + (size_t)sl * tb * s.x_stride;
            if (s.act_up) {
                c.act_up = s.act_up + (size_t)sl * tb * s.x_stride;
            }
            c.out = s.out + (size_t)sl * tb * s.out_stride;
            if (s.residual) {
                c.residual = s.residual + (size_t)sl * tb * s.out_stride;
            }
            plan.add(c);
        }
    };
    // SI8/int8 variant: one full-chunk segment (no 8-token slicing).  The GPU
    // uses the packed w8 copy; the CPU sets `i8` and reads the GGUF blocks
    // directly (so w/type/K/n_rows are the real tensor geometry).  In
    // multi-device mode each GPU segment takes the w8 copy of its own device's
    // partition (w8_dev_) when the int8 decode path is on - prefill then runs
    // oneDNN on it, single-token decode dp4a_gemv - and the CPU partitions set
    // `i8`; otherwise the segments stay shaped for the fp32/oneDNN paths.
    // weight key/pointer for a segment: the device copy when it was uploaded,
    // otherwise the host weight pointer, which is the oneDNN int8 weight key
    // (the converter skips the raw upload of every tensor it converted).
    // Resolve exactly like wptr(): a single-device run goes through the model's
    // one uploaded blob, and only multi-device keeps a per-device host->device
    // map.  (Indexing that empty map used to be out-of-bounds, which both
    // segfaulted the CPU backend and - with the fp32 path, where a segment's
    // `w` is dereferenced on the device - silently fed the kernels host
    // pointers, so the logits were garbage.)
    auto wkey = [&](int dev, const void * host) -> const void * {
        if (!host) {
            return nullptr;
        }
        if (!multi_dev) {
            return wptr(dev, host); // single device: the one uploaded blob
        }
        const auto & mp = weight_maps_[(size_t)dev];
        auto it = mp.find(host);
        return it != mp.end() ? it->second : host;
    };
    auto add8 = [&](int dev, const wt & w, const w8t & w8, const float * x, int xs, float * out, int os,
                    const float * res) {
        w8t w8v = w8;
        if (multi_dev && w8_dev_.size()) {
            auto it = w8_dev_[(size_t)dev].find(w.data);
            if (it != w8_dev_[(size_t)dev].end()) {
                w8v = it->second;
            }
        }
        gemv_seg c{};
        c.dev = dev;
        // with a SIn copy the raw device weight is not uploaded: key the
        // segment on the packed buffer (oneDNN/fallback paths use w8.vals too)
        c.w = w8v.vals ? (const void *)w8v.vals : wkey(dev, w.data);
        if (md_xmx && !c.w8.vals) {
            if (dnnl_gemm * D = dnnl_for(dev)) {
                c.wi8 = D->weight_data(c.w);
                c.wsc = D->weight_scales(c.w);
            }
        }
        c.type = w.type;
        c.K = w.K;
        c.n_rows = w.N;
        c.x = x;
        c.x_stride = xs;
        c.out = out;
        c.out_stride = os;
        c.residual = res;
        c.alpha = 1.0f;
        c.w8 = w8v;
        c.x8 = d_x8;
        c.xmeta = d_xmeta;
        c.xsumq = d_xsumq;
        // i8 only for CPU-partition layers: single-device CPU, or the CPU
        // backends of a multi-device run (dev_kind_ is empty when !multi_dev)
        c.i8 = use_w8 && (multi_dev ? dev_kind_[(size_t)dev] == 1 : cpu_mode);
        plan.add(c);
    };
    auto mk = [&](int dev, const wt & w, const float * x, int xs, float * out, int os, const float * res) {
        gemv_seg s{};
        s.dev = dev;
        s.w = wkey(dev, w.data);
        if (md_xmx) {
            // decode reads the oneDNN int8 buffer directly (M=1 has too much
            // per-call overhead through the primitive)
            if (dnnl_gemm * D = dnnl_for(dev)) {
                s.wi8 = D->weight_data(s.w);
                s.wsc = D->weight_scales(s.w);
            }
        }
        s.meta32 = meta32_of(w.data);
        s.type = w.type;
        s.K = w.K;
        s.n_rows = w.N;
        s.x = x;
        s.x_stride = xs;
        s.act_up = nullptr;
        s.out = out;
        s.out_stride = os;
        s.residual = res;
        s.alpha = 1.0f;
        return s;
    };
    for (int il = 0; il < hp.n_layer; il++) {
        const layer_t & L = m.layers[il];
        const int dev = multi_dev ? layer_dev_[il] : 0;
        bind_acts(dev); // per-device activations: the plan captures dev's buffers
        plan.layer_c0.push_back((int)plan.call_tb.size());
        if (L.recurrent) {
            plan.begin_call(tb, hp.n_embd / 256);
            if (use_w8) {
                add8(dev, L.wqkv, L.wqkv8, d_xnorm, hp.n_embd, d_qkv, hp.qkv_dim(), nullptr);
                add8(dev, L.wgate, L.wgate8, d_xnorm, hp.n_embd, d_z, hp.d_inner, nullptr);
                plan.set_xq(d_xnorm, nullptr, hp.n_embd, hp.n_embd, hp.n_embd);
                add_sliced(mk(dev, L.ssm_beta, d_xnorm, hp.n_embd, d_beta, hp.dt_rank, nullptr));
                add_sliced(mk(dev, L.ssm_alpha, d_xnorm, hp.n_embd, d_alpha, hp.dt_rank, nullptr));
            } else {
                add_sliced(mk(dev, L.wqkv, d_xnorm, hp.n_embd, d_qkv, hp.qkv_dim(), nullptr));
                add_sliced(mk(dev, L.wgate, d_xnorm, hp.n_embd, d_z, hp.d_inner, nullptr));
                add_sliced(mk(dev, L.ssm_beta, d_xnorm, hp.n_embd, d_beta, hp.dt_rank, nullptr));
                add_sliced(mk(dev, L.ssm_alpha, d_xnorm, hp.n_embd, d_alpha, hp.dt_rank, nullptr));
            }
            plan.begin_call(tb, hp.d_inner / 256);
            if (use_w8) {
                add8(dev, L.ssm_out, L.ssm_out8, d_attn_merged, hp.d_inner, d_x, hp.n_embd, d_x);
                plan.set_xq(d_attn_merged, nullptr, hp.d_inner, hp.d_inner, hp.d_inner);
            } else {
                add_sliced(mk(dev, L.ssm_out, d_attn_merged, hp.d_inner, d_x, hp.n_embd, d_x));
            }
        } else {
            plan.begin_call(tb, hp.n_embd / 256);
            if (use_w8) {
                add8(dev, L.wq, L.wq8, d_xnorm, hp.n_embd, d_qbuf, hp.n_head * 2 * hp.head_dim, nullptr);
                add8(dev, L.wk, L.wk8, d_xnorm, hp.n_embd, d_kbuf, hp.n_head_kv * hp.head_dim, nullptr);
                add8(dev, L.wv, L.wv8, d_xnorm, hp.n_embd, d_vbuf, hp.n_head_kv * hp.head_dim, nullptr);
                plan.set_xq(d_xnorm, nullptr, hp.n_embd, hp.n_embd, hp.n_embd);
            } else {
                add_sliced(mk(dev, L.wq, d_xnorm, hp.n_embd, d_qbuf, hp.n_head * 2 * hp.head_dim, nullptr));
                add_sliced(mk(dev, L.wk, d_xnorm, hp.n_embd, d_kbuf, hp.n_head_kv * hp.head_dim, nullptr));
                add_sliced(mk(dev, L.wv, d_xnorm, hp.n_embd, d_vbuf, hp.n_head_kv * hp.head_dim, nullptr));
            }
            plan.begin_call(tb, hp.n_head * hp.head_dim / 256);
            if (use_w8) {
                add8(dev, L.wo, L.wo8, d_attn_out, hp.n_head * hp.head_dim, d_x, hp.n_embd, d_x);
                plan.set_xq(d_attn_out, nullptr, hp.n_head * hp.head_dim, hp.n_head * hp.head_dim,
                            hp.n_head * hp.head_dim);
            } else {
                add_sliced(mk(dev, L.wo, d_attn_out, hp.n_head * hp.head_dim, d_x, hp.n_embd, d_x));
            }
        }
        plan.begin_call(tb, hp.n_embd / 256);
        if (use_w8) {
            add8(dev, L.ffn_gate, L.ffn_gate8, d_xnorm, hp.n_embd, d_ffn, ffn_stride, nullptr);
            add8(dev, L.ffn_up, L.ffn_up8, d_xnorm, hp.n_embd, d_ffn + hp.n_ff, ffn_stride, nullptr);
            plan.set_xq(d_xnorm, nullptr, hp.n_embd, hp.n_embd, hp.n_embd);
        } else {
            add_sliced(mk(dev, L.ffn_gate, d_xnorm, hp.n_embd, d_ffn, ffn_stride, nullptr));
            add_sliced(mk(dev, L.ffn_up, d_xnorm, hp.n_embd, d_ffn + hp.n_ff, ffn_stride, nullptr));
        }
        plan.begin_call(tb, hp.n_ff / 256);
        if (use_w8) {
            // ffn_down: activations are silu(gate)*up.  The quantizer applies it
            // from call_xq.up; the fp32 GEMV (and the oneDNN fallback) apply it
            // from the segment's act_up, so that must be set here as well -
            // otherwise the fp32 path multiplies by the raw gate.
            add8(dev, L.ffn_down, L.ffn_down8, d_ffn, ffn_stride, d_x, hp.n_embd, d_x);
            plan.set_xq(d_ffn, d_ffn + hp.n_ff, ffn_stride, ffn_stride, hp.n_ff);
            plan.set_act_up(d_ffn + hp.n_ff, d_ffn);
        } else {
            gemv_seg s = mk(dev, L.ffn_down, d_ffn, ffn_stride, d_x, hp.n_embd, d_x);
            s.act_up = d_ffn + hp.n_ff;
            add_sliced(s);
        }
    }
    // head: decode -> kMaxB rows of d_xnorm; prefill -> single row in d_last_hidden
    // (the DP4A path keeps the head on the fp32 GEMV: only one row is needed)
    // Skipped entirely for non-final prefill chunks (with_head == false).
    plan.has_head = with_head || head_batched;
    plan.layer_c0.push_back((int)plan.call_tb.size()); // the head call index
    if (with_head || head_batched) {
        // The head always runs on the primary backend, and bind_acts() only
        // rebinds the engine's members - the plan has already captured the
        // *last layer's* device buffers above (d_xnorm/d_x8/... belong to
        // layer_dev_[n_layer-1]).  Rebind to device 0 here so the head segment
        // captures the primary's buffers, which is where record_forward's
        // rmsnorm (after its own bind_acts(0)) writes them.  Without this the
        // batch-1 decode head read a buffer no kernel had written for that
        // step, so the decode's logits (and thus its sampling) were wrong
        // while the prefill - which goes through run_head() - stayed correct.
        bind_acts(0);
        plan.begin_call(head_batched ? tb : 1, hp.n_embd / 256);
        gemv_seg s{};
        s.dev = 0;
        // wkey, not wptr: `wkey` returns the (not-uploaded) host pointer for a
        // converted head, which is its oneDNN key - and gemv_at's oneDNN branch
        // now covers single-token calls in every mode, so the fp32 gemv_group
        // (which would dereference that pointer on the device) is never reached.
        // A head that could not be converted is still uploaded and wkey returns
        // its device pointer, keeping the fp32 fallback valid.
        s.w = wkey(0, m.output.data);
        s.type = m.output.type;
        s.K = hp.n_embd;
        s.n_rows = hp.n_vocab;
        s.x = head_batched ? d_xnorm : d_last_hidden;
        s.x_stride = hp.n_embd;
        s.act_up = nullptr;
        s.out = d_logits;
        s.out_stride = hp.n_vocab;
        s.residual = nullptr;
        s.alpha = 1.0f;
        if (use_w8 && head_batched) {
            // batch-1 decode: run the LM head (45% of the decode weights) on the
            // int8 GEMV path as well.  The head always runs on the primary
            // device (backend 0); multi-device uses that device's w8 copy.
            w8t t8 = m.output8;
            if (multi_dev && w8_dev_.size()) {
                auto it = w8_dev_[0].find(m.output.data);
                if (it != w8_dev_[0].end()) {
                    t8 = it->second;
                }
            }
            if (t8.vals) {
                s.w8 = t8;
                s.x8 = d_x8;
                s.xmeta = d_xmeta;
                s.xsumq = d_xsumq;
                plan.set_xq(d_xnorm, nullptr, hp.n_embd, hp.n_embd, hp.n_embd);
            }
        }
        // multi-device oneDNN weight path: without an xq entry gemv_at cannot
        // take the dnnl_call branch, so the head fell through to the fp32
        // dequant GEMV - 12.3 ms per token vs 3.7 ms for the int8 grouped GEMV
        // (measured on the 27B, 2x A770).  `head_batched` selects the activation
        // the segment points at: the batch-1 decode head reads d_xnorm, the
        // single-token prefill head reads d_last_hidden (see s.x above).
        if (md_xmx && !plan.call_xq.back().x) {
            plan.set_xq(head_batched ? d_xnorm : d_last_hidden, nullptr, hp.n_embd, hp.n_embd, hp.n_embd);
        }
        plan.add(s);
    }
    return plan;
}
// PF_DUMP_SEGS: fingerprint of one GEMV segment's whole field set plus its
// output row, so the int8 / u4 / fp32 weight paths can be diffed tensor by
// tensor and field by field inside a layer.  Diagnostic only.
static void dbg_dump_seg(sycl::queue & q, const gemv_seg & sg, int call, int tb, int nsb, int nch, int up_stride,
                         int xq_up_stride, int xq_x_stride) {
    const int n = std::min(sg.out_stride, 2048);
    static std::vector<float> h;
    if ((int)h.size() < n) {
        h.resize((size_t)n);
    }
    q.memcpy(h.data(), sg.out, (size_t)n * 4).wait();
    double s0 = 0, s1 = 0;
    float mx = 0;
    for (int i = 0; i < n; i++) {
        const double v = h[i];
        s0 += v;
        s1 += v * v;
        mx = sycl::fmax(mx, sycl::fabs((float)v));
    }
    fprintf(stderr,
            "[seg] call=%2d type=%-2u K=%-5d rows=%-5d x_stride=%-6d out_stride=%-6d alpha=%.2f res=%d "
            "up=%d up_stride=%-6d xq_xs=%-6d xq_us=%-6d tb=%-2d nsb=%-4d nch=%d | sum=%12.6f sumsq=%14.6f max=%.6f\n",
            call, sg.type, sg.K, sg.n_rows, sg.x_stride, sg.out_stride, (double)sg.alpha, sg.residual ? 1 : 0,
            sg.act_up ? 1 : 0, up_stride, xq_x_stride, xq_up_stride, tb, nsb, nch, s0, s1, mx);
}

// PF_DUMP_LAYERS: fingerprint of one device activation buffer, printed so two
// weight paths (int8 / u4 / fp32) can be compared layer by layer to see where
// they start to differ.  Diagnostic only, off unless PF_DUMP_LAYERS is set.
static void dbg_dump_fp(sycl::queue & q, const float * d, int n, const char * tag, int il, int dev) {
    static std::vector<float> h;
    if ((int)h.size() < n) {
        h.resize((size_t)n);
    }
    q.memcpy(h.data(), d, (size_t)n * 4).wait();
    double s0 = 0, s1 = 0;
    float mx = 0;
    int am = 0;
    for (int i = 0; i < n; i++) {
        const double v = h[i];
        s0 += v;
        s1 += v * v;
        if (std::fabs((float)v) > mx) {
            mx = std::fabs((float)v);
            am = i;
        }
    }
    fprintf(stderr, "[dump] %-5s il=%2d dev=%d sum=%.6f sumsq=%.6f max=%.6f@%d\n", tag, il, dev, s0, s1, mx, am);
}

// PF_DUMP_RAW=<prefix>: write the full activation vector of one layer to
// <prefix>.ilNN.w<which>.bin, so a decode and a prefill run can be diffed
// element by element (the printed sum/sumsq fingerprints cannot see a change of
// direction, only of magnitude).
static void dbg_dump_raw(sycl::queue & q, const float * d, int n, const char * prefix, int call, int il, int which) {
    static std::vector<float> h;
    if ((int)h.size() < n) {
        h.resize((size_t)n);
    }
    q.memcpy(h.data(), d, (size_t)n * 4).wait();
    char path[512];
    snprintf(path, sizeof(path), "%s.c%d.il%02d.w%d.bin", prefix, call, il, which);
    if (FILE * fp = fopen(path, "wb")) {
        fwrite(h.data(), 4, (size_t)n, fp);
        fclose(fp);
    }
}

void engine::record_forward(int mode, const seg_plan & plan, gemv_seg * d_segs, int rows, gemv_seg * d_segs_rows,
                            int at_nsp_hint, const md_phase * ph) {
    // mode 2: chunk-batched prefill. `rows` is the total token count, split into
    // rows/kMaxT chunk rows; GEMM calls are executed segment-major (all chunk
    // rows of one tensor back to back) so the weights stay L2-hot.
    const int NCH = (mode == 2) ? (rows + kMaxT - 1) / kMaxT : 1;
    const size_t NSEG = plan.segs.size();
    const hparams & hp = m.hp;
    // backend index of the layer currently processed (multi-device): the
    // gemv_at closure reads it to pick the device's oneDNN instance
    int cur_dev = 0;
    // Phased recording (multi-device decode graphs): one phase covers a
    // contiguous run of layers on one device, with no device switch inside it,
    // so the handoff (a host round trip) stays outside the graph.
    const bool phased = (ph != nullptr);
    int dbg_layer = -1; // current layer index for the PF_DUMP_SEGS diagnostic
    const bool prof = prof_on();
    auto tnow = [] { return std::chrono::high_resolution_clock::now(); };
    auto tms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const auto pt0 = tnow();
    double c_embed = 0, c_gemv = 0, c_head = 0, c_attn = 0, c_gdn = 0, c_norm = 0;
    static const bool dbg_dump = [] {
        const char * e = getenv("PF_DUMP_LAYERS");
        return e && atoi(e) != 0;
    }();
    // PF_DUMP_SEGS=<layer>: dump that layer's per-segment fields + output
    // fingerprint (-1 = every layer, noisy).  Diagnostic only.
    static const int dbg_segs = [] {
        const char * e = getenv("PF_DUMP_SEGS");
        return e ? atoi(e) : -2;
    }();
    double c_g8 = 0, c_gf = 0; // w8 GEMM vs fp32/side GEMV time inside gemv
    size_t ci = 0;
    static const bool dbg_pfb2 = getenv("PF_DBG_PFB") != nullptr;
    // Per-group timing under PF_PROF (PF_NOGRAPH mode only: waits serialize).
    // Multi-device runs each partition on its own queue, so the wait must target
    // the queue the current layer was enqueued on - waiting on `q` (the primary
    // queue, which is a *different* queue object in multi-device mode) returned
    // immediately and reported enqueue-only times.
    auto sync_cur = [&]() {
        if (multi_dev) {
            dev_queue(cur_dev).wait();
        } else {
            q.wait();
        }
    };
    auto stamp = [&](double & acc, const std::chrono::high_resolution_clock::time_point & a) {
        if (prof) {
            sync_cur();
            acc += tms(a, tnow());
        }
    };
    auto gemv_at = [&](size_t idx) {
        const int gb = plan.call_group_begin[idx];
        const int gc = plan.call_group_count[idx];
        if (dbg_pfb2) {
            const gemv_seg & s0 = plan.segs[plan.groups[gb].off];
            fprintf(stderr, "[gemv_at] idx=%zu groups=%d first_w8=%d mode=%d rows=%d", idx, gc, s0.w8.vals ? 1 : 0,
                    mode, rows);
            if (plan.call_tb[idx] == 1) {
                fprintf(stderr, " HEAD x=%p out=%p out_stride=%d n_rows=%d type=%u d_x=%p d_logits=%p",
                        (const void *)s0.x, (void *)s0.out, s0.out_stride, s0.n_rows, s0.type, (void *)d_x,
                        (void *)d_logits);
            }
            fprintf(stderr, "\n");
        }
        // DP4A path: quantize this call's activations once, then run the
        // int8 GEMMs (segments with a w8 view) and fp32 GEMVs (the rest)
        const int tb = plan.call_tb[idx];
        // the head call runs on a single token (d_last_hidden) - never batched
        const bool single = (tb == 1);
        const auto a_grp = tnow();
        const int nb = single ? 1 : NCH;
        // chunk-batched prefill: one M-tiled GEMM over all chunks' tokens.  The
        // activation layout is (g*tbm + t) with t the flat token index (chunk
        // rows are tpb=32 apart, so the tokens are contiguous).
        const int tbm = (mode == 2 && !single) ? rows : tb;
        // PF_GEMM_DNNL: run this call's w8 GEMMs on oneDNN.  Per call, not per
        // segment: the activation quantization is shared by all tensors of the
        // call, exactly like the dp4a xq_launch above.  Only take this route
        // when every w8 segment of the call has a converted weight (otherwise
        // the dp4a kernel below needs the x8 buffers quantized the usual way).
        // Mode 1 (chunked, incl. the <32-token prompt tail) also runs its GEMMs
        // on oneDNN, at M = kMaxT: the chunked path is otherwise ~100 ms for a
        // 25-token tail because every GEMM is a full-weight pass at M=32.
        // the dnnl_gemm bound to the backend computing the current call (the
        // single-device instance, or the device's own in multi-device mode)
        dnnl_gemm * D = dnnl_for(cur_dev);
        const bool dnnl_call = [&]() -> bool {
            // single-device decode keeps the dp4a GEMV; multi-device without a
            // SIn copy routes decode through oneDNN too (M=1, via the custom
            // i8_row_gemv), so all of the layer linears live in one int8
            // representation for both phases
            // `single` calls outside decode used to be excluded, which sent the
            // *prefill* LM head to the fp32 dequant gemv_group - the same slow
            // path the decode head had (10.1 vs 3.3 ms).  Now that the head's
            // weight is only a oneDNN entry (its raw device copy is skipped when
            // it was converted), that fallback would also dereference a host
            // pointer, so the oneDNN branch must cover it.  The remaining checks
            // still require an xq entry and a convertible weight per segment.
            if (!D || (mode == 0 && !multi_dev)) {
                return false;
            }
            if (idx >= plan.call_xq.size() || !plan.call_xq[idx].x) {
                return false;
            }
            const int gb0 = plan.call_group_begin[idx];
            const int gc0 = plan.call_group_count[idx];
            for (int g = 0; g < gc0; g++) {
                const seg_plan::group_t & gr0 = plan.groups[gb0 + g];
                for (int j = 0; j < gr0.n; j++) {
                    const gemv_seg & sj = plan.segs[gr0.off + j];
                    // key: the SIn w8 copy on the single-device path, the fp32
                    // device weight pointer (gemv_seg::w) in multi-device mode
                    const void * key = sj.w8.vals ? sj.w8.vals : sj.w;
                    const int kk = sj.w8.vals ? sj.w8.K : sj.K;
                    // multi-device has no w8 view: every segment of the call is
                    // routed to oneDNN, so all of them must be convertible
                    const bool need = sj.w8.vals || multi_dev;
                    if (need
                        && (!(D->has_weight(key) || D->has_weight_w4(key) || D->has_weight_cb4(key))
                            || kk != plan.call_xq[idx].K)) {
                        return false;
                    }
                }
            }
            return true;
        }();
        if (dnnl_call) {
            const seg_plan::xq_t & xq = plan.call_xq[idx];
            const auto a2 = tnow();
            // decode reads the even/odd split in the u4 GEMV; prefill does not
            D->quantize(xq.x, xq.up, xq.x_stride, xq.up_stride, tbm, xq.K, /*do_split=*/mode == 0);
            if (prof_on()) {
                sync_cur();
                prof_acc[5] += tms(a2, tnow());
            }
        } else if (idx < plan.call_xq.size() && plan.call_xq[idx].x) {
            const seg_plan::xq_t & xq = plan.call_xq[idx];
            const bool p2 = prof_on();
            const auto a2 = tnow();
            if (mode == 2 && !single) {
                cur_be->xq(xq.x, xq.up, xq.x_stride, xq.up_stride, d_x8, d_xmeta, d_xsumq, d_info, tbm, xq.K);
            } else {
                for (int r = 0; r < nb; r++) {
                    const float * xr = xq.x + (size_t)r * kMaxT * xq.x_stride;
                    const float * ur = xq.up ? xq.up + (size_t)r * kMaxT * xq.up_stride : nullptr;
                    // TB must be the call's token count (kMaxT for prefill chunks,
                    // the batch size for decode) - it sets the x8 group stride
                    cur_be->xq(xr, ur, xq.x_stride, xq.up_stride, d_x8 + (size_t)r * (size_t)xq.K * tb,
                              d_xmeta + (size_t)r * (xq.K / 32), d_xsumq + (size_t)r * (xq.K / 16), d_info, tb, xq.K);
                }
            }
            if (p2) {
                sync_cur();
                prof_acc[5] += tms(a2, tnow());
            }
        }
        for (int g = 0; g < gc; g++) {
            const seg_plan::group_t & gr = plan.groups[gb + g];
            const gemv_seg & s0 = plan.segs[gr.off];
            const auto a_g2 = tnow();
            if (s0.i8) {
                // CPU integer path: compute straight from the GGUF blocks
                if (mode != 0 && !single) {
                    for (int j = 0; j < gr.n; j++) {
                        cur_be->i8_gemm(plan.segs[gr.off + j], tbm);
                    }
                } else {
                    for (int r = 0; r < nb; r++) {
                        for (int j = 0; j < gr.n; j++) {
                            const gemv_seg sj =
                                (nb > 1) ? row_offset_seg(plan.segs[gr.off + j], r, kMaxT) : plan.segs[gr.off + j];
                            if (tb == 1) {
                                cur_be->i8_gemv(sj);
                            } else {
                                cur_be->i8_gemm(sj, tb);
                            }
                        }
                    }
                }
            } else if (s0.w8.vals) {
                // a group may hold several DP4A segments (same quant type, e.g.
                // wqkv+wgate or ffn_gate+ffn_up): each is a separate GEMM
                // (mode 1 chunked prefill uses the same path with tbm = kMaxT)
                if (mode != 0 && !single) {
                    for (int j = 0; j < gr.n; j++) {
                        const gemv_seg & sj = plan.segs[gr.off + j];
                        if (dnnl_call
                            && D->gemm(sj.w8.vals, sj.residual, sj.alpha, tbm, sj.w8.K, sj.out, sj.out_stride)) {
                            continue;
                        }
                        // fallback: an unsupported oneDNN shape (unexpected) needs
                        // the dp4a x8 quantization the dnnl_call branch skipped
                        if (dnnl_call) {
                            const seg_plan::xq_t & xq = plan.call_xq[idx];
                            cur_be->xq(xq.x, xq.up, xq.x_stride, xq.up_stride, d_x8, d_xmeta, d_xsumq, d_info, tbm,
                                      xq.K);
                        }
                        cur_be->dp4a_gemm(sj.w8, sj.x8, sj.xmeta, sj.xsumq, sj.out, sj.out_stride, sj.residual, sj.alpha, tbm);
                    }
                } else {
                    for (int r = 0; r < nb; r++) {
                        for (int j = 0; j < gr.n; j++) {
                            const gemv_seg sj =
                                (nb > 1) ? row_offset_seg(plan.segs[gr.off + j], r, kMaxT) : plan.segs[gr.off + j];
                            if (tb == 1) { // single-token decode: dedicated coalesced GEMV
                                cur_be->dp4a_gemv(sj.w8, sj.x8, sj.xmeta, sj.xsumq, sj.out, sj.residual, sj.alpha);
                            } else {
                                cur_be->dp4a_gemm(sj.w8, sj.x8, sj.xmeta, sj.xsumq, sj.out, sj.out_stride,
                                              sj.residual, sj.alpha, tb);
                            }
                        }
                    }
                }
            } else if (multi_dev && dnnl_call) {
                // multi-device GPU group: segments carry no SIn w8 view (the
                // w8 copies are not built), so run the prefill GEMMs on the
                // device's oneDNN instance, keyed by gemv_seg::w.  dnnl_call
                // verified every segment of the call is convertible; a gemm
                // failure (unexpected) falls the whole group back to fp32.
                bool dnnl_done = true;
                const int8_t * aq = D->act_data();
                const float * asc = D->act_scales();
                bool decoded = false;
                if (mode == 0 && aq && asc) {
                    // all segments of the call share the activation row: one
                    // GEMV launch covers the whole call
                    bool all = true;
                    int tot = 0;
                    for (int j = 0; j < gr.n; j++) {
                        const gemv_seg & sj = plan.segs[gr.off + j];
                        if (!sj.wi8 || !sj.wsc) {
                            all = false;
                            break;
                        }
                        tot += sj.n_rows;
                    }
                    if (all) {
                        static const bool nogemv = getenv("PF_ABL_NOGEMV") != nullptr;
                        if (!nogemv) {
                            cur_be->i8_row_gemv_multi(d_segs + gr.off, gr.n, tot, aq, asc, D->act_sum());
                        }
                        decoded = true;
                    }
                }
                if (!decoded) {
                    for (int j = 0; j < gr.n; j++) {
                        const gemv_seg & sj = plan.segs[gr.off + j];
                        // 4-bit weights first (u4 + grouped step scales + offset
                        // correction); gemm() returns false for those keys, and
                        // gemm_w4() returns false for int8 keys - so either
                        // order works, try the 4-bit one first.
                        static long c_w4 = 0, c_i8 = 0, c_fb = 0;
                        static const bool w4dbg = [] {
                            const char * e = getenv("PF_W4_DEBUG");
                            return e && atoi(e) != 0;
                        }();
                        if (D->gemm_w4(sj.w, sj.residual, sj.alpha, tbm, sj.K, sj.out, sj.out_stride)) {
                            c_w4++;
                            continue;
                        }
                        if (D->gemm(sj.w, sj.residual, sj.alpha, tbm, sj.K, sj.out, sj.out_stride)) {
                            c_i8++;
                            continue;
                        }
                        c_fb++;
                        static int nrep = 0;
                        if (w4dbg && nrep++ < 3) {
                            fprintf(stderr,
                                    "[w4] segment fell through: w4=%ld i8=%ld fb=%ld (mode=%d M=%d K=%d w4key=%d)\n",
                                    c_w4, c_i8, c_fb, mode, tbm, sj.K, (int)D->has_weight_w4(sj.w));
                        }
                        dnnl_done = false;
                        break;
                    }
                }
                if (!dnnl_done) {
                    cur_be->gemv_group(gr.type, d_segs + gr.off, gr.n, gr.rows, tb, plan.call_nsb[idx],
                                       (mode == 2 && !single) ? NCH : 0);
                }
            } else if (mode == 2 && !single) {
                // chunk-batched prefill: one dispatch over all chunk rows via
                // the token-block grid.  Per-row calls of these (small) fp32
                // tensors cost ~0.5 ms of dispatch tail each, so one call per
                // group is much cheaper than NCH calls.
                cur_be->gemv_group(gr.type, d_segs + gr.off, gr.n, gr.rows, tb, plan.call_nsb[idx], NCH);
            } else {
                for (int r = 0; r < nb; r++) {
                    const gemv_seg * segs_r = (nb > 1 && d_segs_rows) ? d_segs_rows + (size_t)r * NSEG : d_segs;
                    cur_be->gemv_group(gr.type, segs_r + gr.off, gr.n, gr.rows, tb, plan.call_nsb[idx], 0);
                }
            }
            if (dbg_segs != -2 && (dbg_segs < 0 || dbg_segs == dbg_layer)) {
                for (int j = 0; j < gr.n; j++) {
                    const gemv_seg & sj = plan.segs[gr.off + j];
                    const seg_plan::xq_t & xq0 = plan.call_xq[idx];
                    dbg_dump_seg(dev_queue(cur_dev), sj, idx, tb, plan.call_nsb[idx], NCH,
                                 sj.act_up ? sj.x_stride : 0, xq0.x_stride, xq0.up_stride);
                }
            }
            if (prof) {
                sync_cur();
                const double dt = tms(a_g2, tnow());
                if (s0.w8.vals || s0.i8 || (multi_dev && dnnl_call)) {
                    c_g8 += dt;
                } else {
                    c_gf += dt;
                }
            }
        }
        (void)a_grp;
    };
    auto gemv = [&]() {
        const auto a = tnow();
        gemv_at(ci);
        const int my_ci = (int)ci;
        ci++;
        if (prof) {
            sync_cur();
            const double dt = tms(a, tnow());
            c_gemv += dt;
            if (my_ci < 256) {
                prof_ci_t[my_ci] += dt;
                prof_ci_n[my_ci]++;
            }
        }
    };
    const float * out_norm = wf32(0, m.output_norm);
    int attn_idx = 0;
    const int T = (mode == 0) ? rows : (mode == 2 ? rows : kMaxT);
    // grid extents: decode = `rows` sequences x 1 token, prefill = 1 row x `rows`
    // tokens, chunk-batched prefill = NCH chunk rows x kMaxT tokens
    const int nrows = (mode == 0) ? rows : (mode == 2 ? NCH : 1);
    const int nreal = (mode == 0) ? 1 : (mode == 2 ? kMaxT : rows);
    // debug stops: 1 = after the first sub-layer of a layer, 2 = after post-attn
    // norm, 3 = after the first FFN GEMM, 4 = right after the embedding
    const int dbg_mid = [] {
        const char * e = getenv("PF_DBG_MID");
        return e ? atoi(e) : 0;
    }();

    if (!phased || ph->embed) {
        const auto a = tnow();
        // the plan builders leave the members bound to the last layer's device;
        // the embedding always runs on the primary device, so bind its buffers
        // before writing d_x (otherwise device 0 reads the other partition's USM)
        if (multi_dev) {
            bind_acts(0);
        }
        cur_be = &backend();
        cur_be->embed(wptr(0, m.tok_embd.data), m.tok_embd.type, d_info, d_x, hp.n_embd, m.tok_embd_row_bytes);
        if (prof) {
            sync_cur();
            c_embed += tms(a, tnow());
        }
        if (dbg_dump) {
            dbg_dump_fp(dev_queue(0), d_x, hp.n_embd, "embed", -1, 0);
        }
        if (dbg_mid == 4 && !phased) {
            return;
        }
    }

    const int stop_layer = [] {
        const char * e = getenv("STOP_AFTER_LAYER");
        return e ? atoi(e) : -1;
    }();
    int prev_dev = 0; // embedding runs on the primary device
    // The members must be bound to the phase's device before its kernels are
    // recorded.
    if (phased && multi_dev) {
        bind_acts(ph->dev);
        prev_dev = ph->dev;
        cur_dev = ph->dev; // gemv_at reads dnnl_for(cur_dev)
    }
    const int il_beg = phased ? ph->l0 : 0;
    const int il_end = phased ? ph->l1 : hp.n_layer;
    if (phased && (size_t)hp.n_layer + 1 <= plan.layer_c0.size()) {
        // a phase starts at its first layer's call, not at call 0
        const bool head_only = ph->head && ph->l0 == ph->l1;
        ci = (size_t)plan.layer_c0[head_only ? (size_t)hp.n_layer : (size_t)il_beg];
    }
    // debug: 1 = after the first sub-layer of a layer, 2 = after post-attn
    // norm, 3 = after the first FFN GEMM, 4 = right after the embedding
    static int dbg_call_no = -1;
    for (int il = il_beg; il < il_end; il++) {
        const layer_t & L = m.layers[il];
        // multi-device: pick this layer's backend; synchronize the previous
        // device at a partition boundary so its writes to the shared host-USM
        // activations are visible before the next device reads them
        compute_backend & LB = be_of(il);
        cur_be = &LB;
        const int dev = multi_dev ? layer_dev_[il] : 0;
        cur_dev = dev;
        dbg_layer = il;
        if (!phased && multi_dev && dev != prev_dev) {
            handoff_x(prev_dev, dev, (size_t)nrows * (size_t)nreal);
            bind_acts(dev);
            prev_dev = dev;
        }
        static const bool nogdn = [] {
            const char * e = getenv("PF_ABL_NOGDN");
            return e && atoi(e) != 0;
        }();
        static const bool noattn = [] {
            const char * e = getenv("PF_ABL_NOATTN");
            return e && atoi(e) != 0;
        }();
        const auto a_n = tnow();
        cur_be->rmsnorm(d_x, wf32(dev, L.attn_norm), d_xnorm, T, hp.n_embd, hp.rms_eps);
        stamp(c_norm, a_n);
        if (L.recurrent) {
            gemv();
            const size_t conv_per = (size_t)(hp.conv_k - 1) * hp.qkv_dim();      // per slot
            const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state; // per slot
            const int gi = m.gdn_layer_index[il];
            // multi-device: each partition owns only its own GDN layers' state
            // (local index); the state never crosses the boundary, only d_x does
            const int gl = multi_dev ? layer_gdn_local_[(size_t)il] : gi;
            float * cs = d_conv_state + (size_t)gl * kMaxB * conv_per; // [layer][slot][3][conv_dim]
            float * gs = d_gdn_state + (size_t)gl * kMaxB * gdn_per;   // [layer][slot][rank][S][S]
            // prefix cache: per-layer slice of a checkpoint slot (see pc_snap)
            pc_snap snap{};
            if (d_pc_states && pc_enabled) {
                snap.base = d_pc_states;
                snap.stride = (int64_t)pc_state_floats;
                snap.layer_off = (int64_t)gi * ((int64_t)gdn_per + (int64_t)conv_per);
                snap.gdn_per = (int32_t)gdn_per;
                snap.conv_per = (int32_t)conv_per;
            }
            // the GDN recurrence is sequential over tokens.  In chunk-batched
            // prefill the whole batch is *materialized* (qkv), so one call per
            // kernel can cover all chunk rows: the conv reads cross-row taps
            // from qkv_raw, the state update runs once for the last row, and
            // the recurrence walks all tokens in one kernel.  This removes the
            // per-row dependency chain (48 launches -> 4).
            const int gdn_rows = (mode == 2) ? NCH : 1;
            const int gdn_nr = (mode == 2) ? kMaxT : nreal;
            // PF_GDN_FUSE: 1 = fused conv + per-row gdn, 2 = fully fused (one
            // gdn call over the whole batch; default).  The per-row mode-2 loop
            // (0) does not survive graph recording (repeated identical kernels
            // with USM args are not replayed per node), so it is diagnosed only.
            static const int fuse_gdn = [] {
                const char * e = getenv("PF_GDN_FUSE");
                const int v = e ? atoi(e) : 2;
                return v == 1 ? 1 : 2;
            }();
            const auto a_g = tnow();
            auto a_sub = tnow();
            if (mode == 2 && fuse_gdn >= 1 && !nogdn) {
                cur_be->conv_l2(d_qkv, cs, wf32(dev, L.ssm_conv1d), d_conv_out, d_info, hp.qkv_dim(), hp.conv_k,
                               hp.d_state, hp.n_group, hp.rms_eps, NCH, kMaxT, 0, kMaxT, /*cross_row=*/true);
                stamp(prof_acc[11], a_sub);
                a_sub = tnow();
                cur_be->conv_state_update(d_qkv, cs, d_info, hp.qkv_dim(), hp.conv_k, NCH, 0, kMaxT, kMaxT,
                                         /*last_row_only=*/true, snap);
                stamp(prof_acc[12], a_sub);
                a_sub = tnow();
                static const bool tdbg = getenv("PF_TIME") != nullptr;
                if (fuse_gdn >= 2) {
                    if (tdbg && !g_capturing) {
                        q.wait();
                    }
                    const auto t0 = tnow();
                    cur_be->gdn(d_conv_out, d_alpha, wf32(dev, L.ssm_dt), wf32(dev, L.ssm_a), d_beta, gs, d_attn_pre,
                               d_info, hp.d_state, hp.n_group, hp.dt_rank, hp.qkv_dim(),
                               1.0f / std::sqrt((float)hp.d_state), kMaxB, 1, 0, T, T, snap);
                    if (tdbg && !g_capturing) {
                        q.wait();
                        fprintf(stderr, "[t] layer %d gdn=%.2f ms\n", il, tms(t0, tnow()));
                    }
                } else {
                    for (int r0 = 0; r0 < NCH; r0++) {
                        cur_be->gdn(d_conv_out, d_alpha, wf32(dev, L.ssm_dt), wf32(dev, L.ssm_a), d_beta, gs,
                                   d_attn_pre, d_info, hp.d_state, hp.n_group, hp.dt_rank, hp.qkv_dim(),
                                   1.0f / std::sqrt((float)hp.d_state), kMaxB, 1, r0, -1, -1, snap);
                    }
                }
                stamp(prof_acc[13], a_sub);
                a_sub = tnow();
            } else if (!nogdn) {
                static const bool dbg_g = getenv("PF_GDN_DBG") != nullptr;
                for (int r0 = 0; r0 < gdn_rows; r0++) {
                    const int rr0 = (mode == 2) ? r0 : 0;
                    const int rn = (mode == 2) ? 1 : nrows;
                    if (dbg_g) {
                        fprintf(stderr,
                                "[gdn] mode=%d r0=%d rr0=%d rn=%d gdn_nr=%d info_n_rows=%d "
                                "active=%d tpb=%d\n",
                                mode, r0, rr0, rn, gdn_nr, d_info->n_rows, d_info->active[rr0], d_info->tpb);
                    }
                    cur_be->conv_l2(d_qkv, cs, wf32(dev, L.ssm_conv1d), d_conv_out, d_info, hp.qkv_dim(), hp.conv_k,
                                   hp.d_state, hp.n_group, hp.rms_eps, rn, gdn_nr, rr0, -1, false);
                    cur_be->conv_state_update(d_qkv, cs, d_info, hp.qkv_dim(), hp.conv_k, rn, rr0, -1, -1, false,
                                             snap);
                    cur_be->gdn(d_conv_out, d_alpha, wf32(dev, L.ssm_dt), wf32(dev, L.ssm_a), d_beta, gs, d_attn_pre,
                               d_info, hp.d_state, hp.n_group, hp.dt_rank, hp.qkv_dim(),
                               1.0f / std::sqrt((float)hp.d_state), kMaxB, rn, rr0, -1, -1, snap);
                }
                stamp(prof_acc[13], a_sub);
                a_sub = tnow();
            }
            // gated_norm has no state dependency on the row order, so all chunk
            // rows run in one dispatch (batched it is ~4x cheaper than NCH calls)
            cur_be->gated_norm(d_attn_pre, d_z, wf32(dev, L.ssm_norm), d_attn_merged, d_info, hp.dt_rank, hp.d_state,
                              hp.rms_eps, mode == 2 ? NCH : nrows, gdn_nr, 0);
            stamp(prof_acc[14], a_sub);
            a_sub = tnow();
            stamp(c_gdn, a_g);
            gemv();
            if (dbg_mid == 1) {
                break;
            }
        } else {
            gemv();
            // pool layer stride is in bytes (kv_layer_stride, see kv_setup);
            // multi-device indexes the pool of the device that owns this layer
            // by its device-local attention-layer index
            void * kp;
            void * vp;
            const void * ksc0;
            const void * vsc0;
            if (multi_dev) {
                const int d = layer_dev_[il];
                const int la = layer_attn_local_[il];
                kp = (char *)dev_kpool_[(size_t)d] + (size_t)la * kv_layer_stride;
                vp = (char *)dev_vpool_[(size_t)d] + (size_t)la * kv_layer_stride;
                ksc0 = dev_kscales_[(size_t)d] ? (char *)dev_kscales_[(size_t)d] + (size_t)la * kv_scale_stride
                                               : nullptr;
                vsc0 = dev_vscales_[(size_t)d] ? (char *)dev_vscales_[(size_t)d] + (size_t)la * kv_scale_stride
                                               : nullptr;
            } else {
                kp = (char *)d_kpool + (size_t)attn_idx * kv_layer_stride;
                vp = (char *)d_vpool + (size_t)attn_idx * kv_layer_stride;
                ksc0 = d_kscales ? (char *)d_kscales + (size_t)attn_idx * kv_scale_stride : nullptr;
                vsc0 = d_vscales ? (char *)d_vscales + (size_t)attn_idx * kv_scale_stride : nullptr;
            }
            const auto a_a = tnow();
            // Attention split count for this forward.
            //  PF_ATTN_SPLIT=N           force N splits for prefill (1 = fused)
            //  PF_DEC_SPLIT=N            decode split count (default kMaxSplits)
            //  mode 2 direct (oneDNN)    splits ~= ceil(max n_kv / PF_ATTN_SPLIT_KEYS)
            //                            clamped to [1, n_splits]
            //  mode 2 recorded variants  split chosen at record time (hint)
            //  mode 1 (single-row)       n_splits
            static const int at_split = [] {
                const char * e = getenv("PF_ATTN_SPLIT");
                return e ? atoi(e) : 0;
            }();
            static const int at_split_keys = [] {
                const char * e = getenv("PF_ATTN_SPLIT_KEYS");
                const int v = e ? atoi(e) : 512;
                return v > 0 ? v : 512;
            }();
            // PF_ATTN_FUSE=0: keep the separate attn_combine kernel even when
            // n_splits == 1 (A/B knob; fusion is the default)
            static const bool at_fuse = [] {
                const char * e = getenv("PF_ATTN_FUSE");
                return !(e && atoi(e) == 0);
            }();
            int nsp;
            float * part;
            if (mode == 0) {
                nsp = dec_splits;
                part = d_partials_dec;
                if (at_split > 0) {
                    nsp = std::min(std::max(at_split, 1), dec_splits);
                }
            } else if (mode == 2) {
                nsp = n_splits;
                if (at_split > 0) {
                    nsp = at_split;
                } else if (at_nsp_hint > 0) {
                    nsp = at_nsp_hint; // recorded variant
                } else {
                    int max_nkv = 0;
                    for (int r = 0; r < nrows; r++) {
                        max_nkv = std::max(max_nkv, d_info->pos[r] + nreal);
                    }
                    nsp = (max_nkv + at_split_keys - 1) / at_split_keys;
                }
                nsp = std::min(std::max(nsp, 1), n_splits);
                part = d_partials;
            } else {
                nsp = at_split > 0 ? at_split : n_splits;
                nsp = std::min(std::max(nsp, 1), n_splits);
                part = d_partials;
            }
            if (noattn) {
                gemv();
                attn_idx++;
                if (dbg_mid == 1) {
                    break;
                }
                continue;
            }
            const void * ksc = ksc0;
            const void * vsc = vsc0;
            cur_be->qk_norm_rope(d_qbuf, d_kbuf, d_vbuf, wf32(dev, L.q_norm), wf32(dev, L.k_norm), kp, vp, d_tables,
                                d_info, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_rot, hp.rope_base, hp.rms_eps,
                                max_blocks, nrows, nreal, ksc, vsc);
            const bool at_fused = (nsp == 1 && at_fuse);
            cur_be->attn(d_qbuf, d_qbuf, kp, vp, part, d_tables, hp.n_head, hp.n_head_kv, hp.head_dim, nsp, d_info,
                        hp.attn_scale, max_blocks, nrows, nreal, at_fused ? d_attn_out : nullptr, -1, ksc, vsc);
            // n_splits == 1 (batched prefill): the attention kernel writes the
            // gated output directly and the combine kernel is skipped
            if (!at_fused) {
                cur_be->attn_combine(part, d_qbuf, d_attn_out, d_info, hp.n_head, hp.head_dim, nsp, nrows, nreal);
            }
            stamp(c_attn, a_a);
            gemv();
            attn_idx++;
            if (dbg_mid == 1) {
                break;
            }
        }
        const auto a_n2 = tnow();
        cur_be->rmsnorm(d_x, wf32(dev, L.post_attn_norm), d_xnorm, T, hp.n_embd, hp.rms_eps);
        stamp(c_norm, a_n2);
        if (dbg_mid == 2) {
            break;
        }
        gemv();
        if (dbg_mid == 3) {
            break;
        }
        gemv();
        // PF_DUMP_LAYERS: per-layer hidden-state fingerprint, to check that the
        // u4 and int8 paths diverge smoothly across layers (amplification)
        // rather than jumping at one layer (a wiring bug).
        if (dbg_dump) {
            if (il == 0) {
                dbg_call_no++;
                fprintf(stderr, "[dump] call=%d extents mode=%d rows=%d nrows=%d nreal=%d NCH=%d\n", dbg_call_no, mode,
                        rows, nrows, nreal, NCH);
            }
            dbg_dump_fp(dev_queue(cur_dev), d_x, hp.n_embd, "layer", il, cur_dev);
            // also fingerprint the *last real* token slot: mode 2 lays a chunk
            // out as [NCH][kMaxT][n_embd] with only d_info->n_real_row[r] slots
            // live, mode 0/1 keep their tokens in row 0 - the token a following
            // decode continues from must be fingerprinted on both sides
            const int last_slot = (mode == 2) ? (NCH - 1) * kMaxT + (int)d_info->n_real_row[NCH - 1] - 1
                                              : nreal - 1;
            if (last_slot > 0) {
                dbg_dump_fp(dev_queue(cur_dev), d_x + (size_t)last_slot * hp.n_embd, hp.n_embd, "layerlast", il,
                            cur_dev);
            }
            if (const char * rp = getenv("PF_DUMP_RAW")) {
                dbg_dump_raw(dev_queue(cur_dev), d_x, hp.n_embd, rp, dbg_call_no, il, 0);
                if (last_slot > 0) {
                    dbg_dump_raw(dev_queue(cur_dev), d_x + (size_t)last_slot * hp.n_embd, hp.n_embd, rp, dbg_call_no, il,
                                 1);
                }
            }
        }
        if (il == stop_layer) {
            break;
        }
    }

    const auto a_out = tnow();
    // The last partition wrote d_x on its own queue: wait for it before the
    // primary device reads the hidden state for the output norm / LM head.
    // Without this the head races the last device (the earlier pipeline only
    // synchronized when the device changed, i.e. never after the final range).
    if (!phased && multi_dev) {
        // move the final hidden state back to the primary device for the head
        handoff_x(prev_dev, 0, (size_t)nrows * (size_t)nreal);
        bind_acts(0);
        // the head is pinned to backend 0, and gemv_at reads dnnl_for(cur_dev),
        // so reset it here: after the loop cur_dev is the *last layer's* device,
        // whose oneDNN table has no entry for the primary's head weight and
        // would silently drop the head to the fp32 dequant GEMV.
        cur_dev = 0;
    }
    if (phased && !ph->head) {
        return; // this phase ends before the output norm / head
    }
    // global tensors (output norm, LM head) live on the primary device
    cur_be = &backend();
    cur_be->rmsnorm(d_x, out_norm, d_xnorm, T, hp.n_embd, hp.rms_eps);
    if (mode != 0) {
        cur_be->copy_row(d_xnorm, d_last_hidden, d_info, hp.n_embd, -1);
    }
    stamp(c_norm, a_out);
    if (plan.has_head) {
        const auto a = tnow();
        gemv_at(plan.call_offsets.size() - 1); // head is the last call group
        if (prof) {
            dev_queue(0).wait(); // the head is pinned to the primary device
            c_head += tms(a, tnow());
        }
    }
    if (prof) {
        dev_queue(0).wait();
        const double c_total = tms(pt0, tnow());
        prof_acc[0] += c_embed;
        prof_acc[2] += c_gemv + c_head;
        prof_acc[3] += c_total - c_embed - c_gemv - c_head - c_attn - c_gdn - c_norm;
        prof_acc[4] += c_total;
        prof_acc[6] += c_attn;
        prof_acc[7] += c_gdn;
        prof_acc[9] += 0;
        prof_acc[8] += c_norm;
        prof_calls++;
        const long every = (mode == 2) ? 1 : 8;
        if ((prof_calls % every) == 0) {
            printf("[prof] %s: gemv=%.2f xq=%.2f head=%.2f attn=%.2f gdn=%.2f norm=%.2f "
                   "other=%.2f total=%.2f ms (embed=%.2f)\n",
                   mode == 2 ? "batch" : "chunk", prof_acc[2] / every, prof_acc[5] / every, c_head, prof_acc[6] / every,
                   prof_acc[7] / every, prof_acc[8] / every, prof_acc[3] / every, prof_acc[4] / every,
                   prof_acc[0] / every);
            printf("      gemv split: w8=%.2f fp32/side=%.2f (incl. per-group sync)\n", c_g8, c_gf);
            printf("      gdn split: conv_l2=%.2f conv_state=%.2f gdn=%.2f gated=%.2f\n", prof_acc[11] / every,
                   prof_acc[12] / every, prof_acc[13] / every, prof_acc[14] / every);
            for (double & v : prof_acc) {
                v = 0;
            }
            c_g8 = c_gf = 0;
            prof_calls = 0;
            // PF_PROF_ALL: dump every call group (ms/step = ms/call * calls/step)
            static const bool prof_all = getenv("PF_PROF_ALL") != nullptr;
            int idx[256];
            for (int i = 0; i < 256; i++) {
                idx[i] = i;
            }
            std::sort(idx, idx + 256, [](int a, int b) { return prof_ci_t[a] > prof_ci_t[b]; });
            const int top = prof_all ? 256 : 8;
            double gemv_sum = 0;
            for (int k = 0; k < top; k++) {
                const int i = idx[k];
                if (prof_ci_t[i] <= 0 || prof_ci_n[i] == 0) {
                    continue;
                }
                const seg_plan::xq_t & xq = plan.call_xq[i];
                const gemv_seg & s0 = plan.segs[plan.groups[plan.call_group_begin[i]].off];
                const double per_step = prof_ci_t[i] / (double)every;
                gemv_sum += per_step;
                printf("   ci=%3d %7.3f ms/call  %5.1f calls/step  %6.2f ms/step  K=%-6d N=%-6d type=%-7s nr=%d xq=%d\n", i,
                       prof_ci_t[i] / prof_ci_n[i], (double)prof_ci_n[i] / (double)every, per_step, xq.K,
                       s0.w8.ok() ? s0.w8.N : s0.n_rows, ggml_type_name(s0.w8.ok() ? s0.w8.type : s0.type), s0.n_rows,
                       xq.x ? 1 : 0);
                prof_ci_t[i] = 0;
                prof_ci_n[i] = 0;
            }
            if (prof_all) {
                printf("   [sum of call groups = %.2f ms/step]\n", gemv_sum);
            }
        }
    }
}
void engine::build_plans() {
    const bool w8_skip_raw = multi_dev && md_int8;
    // CPU backend: build the same segment plans the GPU records, but keep them
    // host-side and replay them directly (no SYCL command graph can be encoded
    // by the host kernels).
    if (!w8_skip_raw) {
    plan_pf_ = build_plan(kMaxT, 8, false);
    plan_pf_.finalize();
    if (plan_pf_.segs.size() > 4096) {
        throw std::runtime_error("segment buffer too small");
    }
    q.memcpy(d_segs_pf, plan_pf_.segs.data(), plan_pf_.segs.size() * sizeof(gemv_seg)).wait();
    }
    // per-token-count prefill plans: the fp32 plan is sliced by kPfSlice, so T
    // rows are T/8 slices.  A prompt shorter than the chunk only pays for its
    // rounded-up token count instead of a full kMaxT chunk.
    for (int i = 0; !w8_skip_raw && i < kPfSlots; i++) {
        const int T = (i + 1) * kPfSlice;
        plan_pf_slot[i] = build_plan(T, kPfSlice, false);
        plan_pf_slot[i].finalize();
        if (plan_pf_slot[i].segs.size() > 2048) {
            throw std::runtime_error("segment buffer too small");
        }
        d_segs_pf_slot[i] = alloc_elems<gemv_seg>(plan_pf_slot[i].segs.size());
        q.memcpy(d_segs_pf_slot[i], plan_pf_slot[i].segs.data(), plan_pf_slot[i].segs.size() * sizeof(gemv_seg))
            .wait();
        plan_pf_nh_slot[i] = build_plan(T, kPfSlice, false, false, false);
        plan_pf_nh_slot[i].finalize();
        d_segs_pf_nh_slot[i] = alloc_elems<gemv_seg>(plan_pf_nh_slot[i].segs.size());
        q.memcpy(d_segs_pf_nh_slot[i], plan_pf_nh_slot[i].segs.data(),
                 plan_pf_nh_slot[i].segs.size() * sizeof(gemv_seg))
            .wait();
    }
    if (pf8 || multi_dev) {
        // multi-device has no recorded graphs and always replays these plans
        // directly, so build them even without oneDNN (the w8 copies may be
        // absent, in which case the fp32 GEMV path handles each segment)
        if (pf8 || multi_dev) {
            plan_pf8_ = build_plan(kMaxT, kMaxT, false, true, true);
            plan_pf8_.finalize();
            q.memcpy(d_segs_pf8, plan_pf8_.segs.data(), plan_pf8_.segs.size() * sizeof(gemv_seg)).wait();
            plan_pf8_nh_ = build_plan(kMaxT, kMaxT, false, true, false);
            plan_pf8_nh_.finalize();
            if (plan_pf8_nh_.segs.size() > 4096) {
                throw std::runtime_error("segment buffer too small");
            }
            d_segs_pf8_nh = alloc_elems<gemv_seg>(plan_pf8_nh_.segs.size());
            q.memcpy(d_segs_pf8_nh, plan_pf8_nh_.segs.data(), plan_pf8_nh_.segs.size() * sizeof(gemv_seg)).wait();
        }
        if (multi_dev) {
            // chunk-batched (mode 2) prefill: same plan shape as plan_pf8_ (one
            // row per kMaxT-token chunk), replayed directly with `rows` = total
            // tokens so all of one prompt's chunks run GEMMs segment-major /
            // weights L2-hot in a single forward.  d_segs_pfb holds per-chunk-row
            // row-offset copies like the single-device graph path.
            plan_pfb_ = build_plan(kMaxT, kMaxT, false, true, true);
            plan_pfb_.finalize();
            const size_t nseg = plan_pfb_.segs.size();
            if (nseg > 4096) {
                throw std::runtime_error("segment buffer too small");
            }
            d_segs_pfb = alloc_elems<gemv_seg>((size_t)kMaxB * nseg);
            std::vector<gemv_seg> h((size_t)kMaxB * nseg);
            for (int r = 0; r < kMaxB; r++) {
                for (size_t i = 0; i < nseg; i++) {
                    h[(size_t)r * nseg + i] = row_offset_seg(plan_pfb_.segs[i], r, kMaxT);
                }
            }
            q.memcpy(d_segs_pfb, h.data(), h.size() * sizeof(gemv_seg)).wait();
        }
        if (pf8_dec || (multi_dev && (md_int8 || md_xmx))) {
            // single-token decode on int8: GPU segments dp4a_gemv (per-device
            // w8), CPU segments i8_gemv - same plan shape both partitions
            plan_dec8_ = build_plan(1, 1, true, true, true);
            plan_dec8_.finalize();
            if (plan_dec8_.segs.size() > 1024) {
                throw std::runtime_error("segment buffer too small");
            }
            d_segs_dec8 = alloc_elems<gemv_seg>(plan_dec8_.segs.size());
            q.memcpy(d_segs_dec8, plan_dec8_.segs.data(), plan_dec8_.segs.size() * sizeof(gemv_seg)).wait();
        }
    }
    for (auto & b : buckets_) {
        b.plan = build_plan(b.tb, b.tb, true, multi_dev || md_int8);
        b.plan.finalize();
        if (b.plan.segs.size() > 1024) {
            throw std::runtime_error("segment buffer too small");
        }
        q.memcpy(b.d_segs, b.plan.segs.data(), b.plan.segs.size() * sizeof(gemv_seg)).wait();
    }
    // leave the members on the primary device for any runtime use
    if (multi_dev) {
        bind_acts(0);
    }
}

void engine::build_md_dec_graphs() {
    static const bool nog = getenv("PF_NOGRAPH") != nullptr;
    if (!multi_dev || !(md_int8 || md_xmx) || !d_segs_dec8 || m.hp.n_layer <= 0 || nog) {
        return; // PF_NOGRAPH=1 keeps the direct replay (diagnostics/PF_PROF)
    }
    // Split the layer loop into contiguous device runs.  The embedding always
    // runs on the primary device and the output norm + head after the loop do
    // too, so the first phase carries the embedding and the last one the head.
    std::vector<md_phase> phs;
    md_phase p{};
    p.dev = 0;
    p.embed = true;
    int il = 0;
    while (il < m.hp.n_layer) {
        const int d = layer_dev_[(size_t)il];
        int j = il;
        while (j < m.hp.n_layer && layer_dev_[(size_t)j] == d) {
            j++;
        }
        if (p.l0 == p.l1 && p.dev == d) {
            p.l1 = j; // the embedding-only phase continues into this run
        } else {
            phs.push_back(p);
            md_phase q{};
            q.dev = d;
            q.l0 = il;
            q.l1 = j;
            p = q;
        }
        il = j;
    }
    if (p.dev == 0) {
        p.head = true;
        phs.push_back(p);
    } else {
        phs.push_back(p);
        md_phase h{};
        h.dev = 0;
        h.head = true;
        phs.push_back(h);
    }

    // a CPU partition runs host code, not kernels on a queue, so it cannot be
    // recorded; fall back to the direct replay for any mixed map
    for (const md_phase & ph : phs) {
        if (ph.dev < 0 || (size_t)ph.dev >= dev_kind_.size() || dev_kind_[(size_t)ph.dev] != 0) {
            fprintf(stderr, "[md] decode command graphs: skipped (device %d is not a GPU partition)\n", ph.dev);
            return;
        }
    }

    // PF_MD_GRAPH_DEV: which devices get a recorded graph (default: all GPU
    // partitions).  Diagnostic/A-B knob while bringing the multi-device graphs
    // up - a phase without a graph falls back to the direct replay.
    static const int graph_dev = [] {
        const char * e = getenv("PF_MD_GRAPH_DEV");
        return e ? atoi(e) : -1; // -1 = every phase
    }();
    capture_guard cg;
    md_dec_.clear();
    md_dec_.reserve(phs.size());
    for (const md_phase & ph : phs) {
        md_cmd_graph mg;
        mg.ph = ph;
        if (graph_dev < 0 || ph.dev == graph_dev) {
            sycl::queue & qd = dev_queue(ph.dev);
            mg.g = std::make_unique<sx::command_graph<sx::graph_state::modifiable>>(qd.get_context(), qd.get_device());
            mg.g->begin_recording(qd);
            record_forward(0, plan_dec8_, d_segs_dec8, 1, nullptr, 0, &ph);
            mg.g->end_recording();
            mg.e = std::make_unique<sx::command_graph<sx::graph_state::executable>>(mg.g->finalize());
        }
        md_dec_.push_back(std::move(mg));
    }
    md_dec_ok = true;
    // leave the members on the primary device for any runtime use
    bind_acts(0);
    fprintf(stderr, "[md] decode command graphs: %zu phase(s) over %zu backend(s)\n", md_dec_.size(), backends_.size());
}

void engine::replay_md_dec_graphs() {
    for (size_t i = 0; i < md_dec_.size(); i++) {
        const md_phase & ph = md_dec_[i].ph;
        if (md_dec_[i].e) {
            dev_queue(ph.dev).ext_oneapi_graph(*md_dec_[i].e);
        } else {
            // no graph for this phase: replay it directly on its own queue
            record_forward(0, plan_dec8_, d_segs_dec8, 1, nullptr, 0, &ph);
        }
        if (i + 1 < md_dec_.size() && md_dec_[i + 1].ph.dev != ph.dev) {
            // hand the activation to the next phase's device; handoff_x waits on
            // the producing queue and stages through host memory (no P2P)
            handoff_x(ph.dev, md_dec_[i + 1].ph.dev, 1);
        }
    }
    // the head phase wrote d_logits on the primary device; rebind the members so
    // fetch_logits()/run_head() read the right buffers
    bind_acts(0);
}

void engine::build_graphs() {
    if (cpu_mode || multi_dev) {
        build_plans();
        if (multi_dev) {
            build_md_dec_graphs();
        }
        return;
    }
    capture_guard cg;
    plan_pf_ = build_plan(kMaxT, 8, false); // 1 row x kMaxT tokens, 8-token gemv slices
    plan_pf_.finalize();
    if (plan_pf_.segs.size() > 4096) {
        throw std::runtime_error("segment buffer too small");
    }
    q.memcpy(d_segs_pf, plan_pf_.segs.data(), plan_pf_.segs.size() * sizeof(gemv_seg)).wait();

    for (auto & b : buckets_) {
        b.plan = build_plan(b.tb, b.tb, true);
        b.plan.finalize();
        if (b.plan.segs.size() > 1024) {
            throw std::runtime_error("segment buffer too small");
        }
        q.memcpy(b.d_segs, b.plan.segs.data(), b.plan.segs.size() * sizeof(gemv_seg)).wait();
        b.g = std::make_unique<sx::command_graph<sx::graph_state::modifiable>>(q.get_context(), q.get_device());
        b.g->begin_recording(q);
        record_forward(0, b.plan, b.d_segs, b.tb);
        b.g->end_recording();
        b.e = std::make_unique<sx::command_graph<sx::graph_state::executable>>(b.g->finalize());
    }

    g_pf = std::make_unique<sx::command_graph<sx::graph_state::modifiable>>(q.get_context(), q.get_device());
    g_pf->begin_recording(q);
    record_forward(1, plan_pf_, d_segs_pf, kMaxT);
    g_pf->end_recording();
    e_pf = std::make_unique<sx::command_graph<sx::graph_state::executable>>(g_pf->finalize());

    // SI8/DP4A prefill graph (enabled with PF_DP4A=1)
    {
        if (pf8) {
            plan_pf8_ = build_plan(kMaxT, kMaxT, false, true, true);
            plan_pf8_.finalize();
            q.memcpy(d_segs_pf8, plan_pf8_.segs.data(), plan_pf8_.segs.size() * sizeof(gemv_seg)).wait();
            g_pf8 = std::make_unique<sx::command_graph<sx::graph_state::modifiable>>(q.get_context(), q.get_device());
            g_pf8->begin_recording(q);
            record_forward(1, plan_pf8_, d_segs_pf8, kMaxT);
            g_pf8->end_recording();
            e_pf8 = std::make_unique<sx::command_graph<sx::graph_state::executable>>(g_pf8->finalize());
            // second variant without the head (used for all but the last prompt chunk)
            // batch-1 decode graph on the int8 path (only when explicitly enabled:
            // the SI8 copies are ~2x the GGUF bytes for Q4_K/Q5_K, so the decode
            // GEMV wins only once the values are nibble-packed)
            if (pf8_dec) {
                plan_dec8_ = build_plan(1, 1, true, true, true);
                plan_dec8_.finalize();
                gemv_seg * d_segs_d8 = sycl::malloc_device<gemv_seg>(plan_dec8_.segs.size(), q);
                q.memcpy(d_segs_d8, plan_dec8_.segs.data(), plan_dec8_.segs.size() * sizeof(gemv_seg)).wait();
                g_dec8 =
                    std::make_unique<sx::command_graph<sx::graph_state::modifiable>>(q.get_context(), q.get_device());
                g_dec8->begin_recording(q);
                record_forward(0, plan_dec8_, d_segs_d8, 1);
                g_dec8->end_recording();
                e_dec8 = std::make_unique<sx::command_graph<sx::graph_state::executable>>(g_dec8->finalize());
                sycl::free(d_segs_d8, q);
            }
            // chunk-batched prefill: all chunks in one forward, GEMMs batched
            // across rows so a tensor's weights stay resident in L2.  The
            // chunk-row loops and the M-tiled token count are baked into the
            // recording, so record one graph per supported batch size.
            {
                plan_pfb_ = build_plan(kMaxT, kMaxT, false, true, true);
                plan_pfb_.finalize();
                const size_t nseg = plan_pfb_.segs.size();
                d_segs_pfb = sycl::malloc_device<gemv_seg>((size_t)kMaxB * nseg, q);
                // ONE copy for all rows: a per-row q.memcpy from the same host
                // vector races with the next row's writes (the copy is async),
                // which silently mixed the row offsets
                std::vector<gemv_seg> h((size_t)kMaxB * nseg);
                for (int r = 0; r < kMaxB; r++) {
                    for (size_t i = 0; i < nseg; i++) {
                        h[(size_t)r * nseg + i] = row_offset_seg(plan_pfb_.segs[i], r, kMaxT);
                    }
                }
                q.memcpy(d_segs_pfb, h.data(), h.size() * sizeof(gemv_seg)).wait();
                // prefill K-split: a recorded graph bakes its grid, so each
                // variant picks a split for its own chunk size (the direct
                // dnnl path derives it from the real n_kv instead).  Bigger
                // chunks amortize the combine cost over more tokens, so the
                // hint grows with ntok up to 4; the long-context win of a
                // wider split is left to the direct path.
                for (int nch : {2, 4, 8, 16}) {
                    pfb_variant v;
                    v.ntok = nch * kMaxT;
                    // PF_GEMM_DNNL runs mode 2 directly (oneDNN cannot be
                    // recorded); the variant only carries its token count so
                    // batched_prefill_fit() still selects the same sizes
                    if (!use_dnnl) {
                        const char * ese = getenv("PF_ATTN_SPLIT");
                        const int fixed = ese ? atoi(ese) : 0;
                        const int hint = fixed > 0 ? fixed : std::min({n_splits, 4, std::max(1, v.ntok / 128)});
                        v.g = std::make_unique<sx::command_graph<sx::graph_state::modifiable>>(q.get_context(),
                                                                                               q.get_device());
                        v.g->begin_recording(q);
                        record_forward(2, plan_pfb_, d_segs_pfb, v.ntok, d_segs_pfb, hint);
                        v.g->end_recording();
                        v.e = std::make_unique<sx::command_graph<sx::graph_state::executable>>(v.g->finalize());
                    }
                    pfb_vars_.push_back(std::move(v));
                }
            }
            seg_plan plan_pf8_nh = build_plan(kMaxT, kMaxT, false, true, false);
            plan_pf8_nh.finalize();
            gemv_seg * d_segs_nh = sycl::malloc_device<gemv_seg>(plan_pf8_nh.segs.size(), q);
            q.memcpy(d_segs_nh, plan_pf8_nh.segs.data(), plan_pf8_nh.segs.size() * sizeof(gemv_seg)).wait();
            auto g_nh =
                std::make_unique<sx::command_graph<sx::graph_state::modifiable>>(q.get_context(), q.get_device());
            g_nh->begin_recording(q);
            record_forward(1, plan_pf8_nh, d_segs_nh, kMaxT);
            g_nh->end_recording();
            e_pf8_nh = std::make_unique<sx::command_graph<sx::graph_state::executable>>(g_nh->finalize());
            sycl::free(d_segs_nh, q);
        }
    }
}

} // namespace si
