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
seg_plan engine::build_plan(int T, int tb, bool head_batched, bool use_w8, bool with_head) const {
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
    // directly (so w/type/K/n_rows are the real tensor geometry).
    const bool i8_mode = use_w8 && cpu_mode;
    auto add8 = [&](int dev, const wt & w, const w8t & w8, const float * x, int xs, float * out, int os,
                    const float * res) {
        gemv_seg c{};
        c.w = wptr(dev, w.data);
        c.type = w.type;
        c.K = w.K;
        c.n_rows = w.N;
        c.x = x;
        c.x_stride = xs;
        c.out = out;
        c.out_stride = os;
        c.residual = res;
        c.alpha = 1.0f;
        c.w8 = w8;
        c.x8 = d_x8;
        c.xmeta = d_xmeta;
        c.xsumq = d_xsumq;
        c.i8 = i8_mode;
        plan.add(c);
    };
    auto mk = [&](int dev, const wt & w, const float * x, int xs, float * out, int os, const float * res) {
        gemv_seg s{};
        s.w = wptr(dev, w.data);
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
        if (L.recurrent) {
            plan.begin_call(tb, hp.n_embd / 256);
            if (use_w8) {
                add8(dev, L.wqkv, L.wqkv8, d_xnorm, hp.n_embd, d_qkv, 3 * hp.d_inner, nullptr);
                add8(dev, L.wgate, L.wgate8, d_xnorm, hp.n_embd, d_z, hp.d_inner, nullptr);
                plan.set_xq(d_xnorm, nullptr, hp.n_embd, hp.n_embd, hp.n_embd);
                add_sliced(mk(dev, L.ssm_beta, d_xnorm, hp.n_embd, d_beta, hp.dt_rank, nullptr));
                add_sliced(mk(dev, L.ssm_alpha, d_xnorm, hp.n_embd, d_alpha, hp.dt_rank, nullptr));
            } else {
                add_sliced(mk(dev, L.wqkv, d_xnorm, hp.n_embd, d_qkv, 3 * hp.d_inner, nullptr));
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
            // ffn_down: activations are silu(gate)*up, applied during quantization
            add8(dev, L.ffn_down, L.ffn_down8, d_ffn, ffn_stride, d_x, hp.n_embd, d_x);
            plan.set_xq(d_ffn, d_ffn + hp.n_ff, ffn_stride, ffn_stride, hp.n_ff);
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
    if (with_head || head_batched) {
        plan.begin_call(head_batched ? tb : 1, hp.n_embd / 256);
        gemv_seg s{};
        s.w = wptr(0, m.tok_embd.data);
        s.type = m.tok_embd.type;
        s.K = hp.n_embd;
        s.n_rows = hp.n_vocab;
        s.x = head_batched ? d_xnorm : d_last_hidden;
        s.x_stride = hp.n_embd;
        s.act_up = nullptr;
        s.out = d_logits;
        s.out_stride = hp.n_vocab;
        s.residual = nullptr;
        s.alpha = 1.0f;
        if (use_w8 && head_batched && m.tok_embd8.vals) {
            // batch-1 decode: run the LM head (45% of the decode weights) on the
            // int8 GEMV path as well
            s.w8 = m.tok_embd8;
            s.x8 = d_x8;
            s.xmeta = d_xmeta;
            s.xsumq = d_xsumq;
            plan.set_xq(d_xnorm, nullptr, hp.n_embd, hp.n_embd, hp.n_embd);
        }
        plan.add(s);
    }
    return plan;
}
void engine::record_forward(int mode, const seg_plan & plan, gemv_seg * d_segs, int rows, gemv_seg * d_segs_rows,
                            int at_nsp_hint) {
    // mode 2: chunk-batched prefill. `rows` is the total token count, split into
    // rows/kMaxT chunk rows; GEMM calls are executed segment-major (all chunk
    // rows of one tensor back to back) so the weights stay L2-hot.
    const int NCH = (mode == 2) ? (rows + kMaxT - 1) / kMaxT : 1;
    const size_t NSEG = plan.segs.size();
    const hparams & hp = m.hp;
    const bool prof = prof_on();
    auto tnow = [] { return std::chrono::high_resolution_clock::now(); };
    auto tms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const auto pt0 = tnow();
    double c_embed = 0, c_gemv = 0, c_head = 0, c_attn = 0, c_gdn = 0, c_norm = 0;
    double c_g8 = 0, c_gf = 0; // w8 GEMM vs fp32/side GEMV time inside gemv
    size_t ci = 0;
    static const bool dbg_pfb2 = getenv("PF_DBG_PFB") != nullptr;
    // per-group timing under PF_PROF (PF_NOGRAPH mode only: waits serialize)
    auto stamp = [&](double & acc, const std::chrono::high_resolution_clock::time_point & a) {
        if (prof) {
            q.wait();
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
        const bool dnnl_call = [&]() -> bool {
            if (!use_dnnl || mode == 0 || single) {
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
                    if (sj.w8.vals && (!dnnl->has_weight(sj.w8.vals) || sj.w8.K != plan.call_xq[idx].K)) {
                        return false;
                    }
                }
            }
            return true;
        }();
        if (dnnl_call) {
            const seg_plan::xq_t & xq = plan.call_xq[idx];
            const auto a2 = tnow();
            dnnl->quantize(xq.x, xq.up, xq.x_stride, xq.up_stride, tbm, xq.K);
            if (prof_on()) {
                q.wait();
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
                q.wait();
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
                            && dnnl->gemm(sj.w8.vals, sj.residual, sj.alpha, tbm, sj.w8.K, sj.out, sj.out_stride)) {
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
            if (prof) {
                q.wait();
                const double dt = tms(a_g2, tnow());
                if (s0.w8.vals || s0.i8) {
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
            q.wait();
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

    {
        const auto a = tnow();
        cur_be = &backend();
        cur_be->embed(wptr(0, m.tok_embd.data), m.tok_embd.type, d_info, d_x, hp.n_embd, m.tok_embd_row_bytes);
        if (prof) {
            q.wait();
            c_embed += tms(a, tnow());
        }
        if (dbg_mid == 4) {
            return;
        }
    }

    const int stop_layer = [] {
        const char * e = getenv("STOP_AFTER_LAYER");
        return e ? atoi(e) : -1;
    }();
    int prev_dev = -1;
    // debug: 1 = after the first sub-layer of a layer, 2 = after post-attn
    // norm, 3 = after the first FFN GEMM, 4 = right after the embedding
    for (int il = 0; il < hp.n_layer; il++) {
        const layer_t & L = m.layers[il];
        // multi-device: pick this layer's backend; synchronize the previous
        // device at a partition boundary so its writes to the shared host-USM
        // activations are visible before the next device reads them
        compute_backend & LB = be_of(il);
        cur_be = &LB;
        const int dev = multi_dev ? layer_dev_[il] : 0;
        if (multi_dev && dev != prev_dev) {
            if (prev_dev >= 0) {
                backends_[(size_t)prev_dev]->synchronize();
            }
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
            const size_t conv_per = (size_t)(hp.conv_k - 1) * 3 * hp.d_inner;    // per slot
            const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state; // per slot
            const int gi = m.gdn_layer_index[il];
            float * cs = d_conv_state + (size_t)gi * kMaxB * conv_per; // [layer][slot][3][conv_dim]
            float * gs = d_gdn_state + (size_t)gi * kMaxB * gdn_per;   // [layer][slot][rank][S][S]
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
                cur_be->conv_l2(d_qkv, cs, wf32(dev, L.ssm_conv1d), d_conv_out, d_info, 3 * hp.d_inner, hp.conv_k,
                               hp.d_state, hp.n_group, hp.rms_eps, NCH, kMaxT, 0, kMaxT, /*cross_row=*/true);
                stamp(prof_acc[11], a_sub);
                a_sub = tnow();
                cur_be->conv_state_update(d_qkv, cs, d_info, 3 * hp.d_inner, hp.conv_k, NCH, 0, kMaxT, kMaxT,
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
                               d_info, hp.d_state, hp.dt_rank, 3 * hp.d_inner, 1.0f / std::sqrt((float)hp.d_state),
                               kMaxB, 1, 0, T, T, snap);
                    if (tdbg && !g_capturing) {
                        q.wait();
                        fprintf(stderr, "[t] layer %d gdn=%.2f ms\n", il, tms(t0, tnow()));
                    }
                } else {
                    for (int r0 = 0; r0 < NCH; r0++) {
                        cur_be->gdn(d_conv_out, d_alpha, wf32(dev, L.ssm_dt), wf32(dev, L.ssm_a), d_beta, gs,
                                   d_attn_pre, d_info, hp.d_state, hp.dt_rank, 3 * hp.d_inner,
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
                    cur_be->conv_l2(d_qkv, cs, wf32(dev, L.ssm_conv1d), d_conv_out, d_info, 3 * hp.d_inner, hp.conv_k,
                                   hp.d_state, hp.n_group, hp.rms_eps, rn, gdn_nr, rr0, -1, false);
                    cur_be->conv_state_update(d_qkv, cs, d_info, 3 * hp.d_inner, hp.conv_k, rn, rr0, -1, -1, false,
                                             snap);
                    cur_be->gdn(d_conv_out, d_alpha, wf32(dev, L.ssm_dt), wf32(dev, L.ssm_a), d_beta, gs, d_attn_pre,
                               d_info, hp.d_state, hp.dt_rank, 3 * hp.d_inner, 1.0f / std::sqrt((float)hp.d_state),
                               kMaxB, rn, rr0, -1, -1, snap);
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
        if (il == stop_layer) {
            break;
        }
    }

    const auto a_out = tnow();
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
            q.wait();
            c_head += tms(a, tnow());
        }
    }
    if (prof) {
        q.wait();
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
            // top call groups by accumulated time
            int idx[256];
            for (int i = 0; i < 256; i++) {
                idx[i] = i;
            }
            std::sort(idx, idx + 256, [](int a, int b) { return prof_ci_t[a] > prof_ci_t[b]; });
            for (int k = 0; k < 8; k++) {
                const int i = idx[k];
                if (prof_ci_t[i] <= 0 || prof_ci_n[i] == 0) {
                    continue;
                }
                const seg_plan::xq_t & xq = plan.call_xq[i];
                const gemv_seg & s0 = plan.segs[plan.groups[plan.call_group_begin[i]].off];
                printf("   ci=%2d %8.2f ms/call  n=%ld  K=%d N=%d type=%s nr=%d os=%d xq=%d\n", i,
                       prof_ci_t[i] / prof_ci_n[i], prof_ci_n[i], xq.K, s0.w8.ok() ? s0.w8.N : 0,
                       ggml_type_name(s0.w8.ok() ? s0.w8.type : s0.type), s0.n_rows, s0.out_stride, xq.x ? 1 : 0);
                prof_ci_t[i] = 0;
                prof_ci_n[i] = 0;
            }
        }
    }
}
void engine::build_plans() {
    // CPU backend: build the same segment plans the GPU records, but keep them
    // host-side and replay them directly (no SYCL command graph can be encoded
    // by the host kernels).
    plan_pf_ = build_plan(kMaxT, 8, false);
    plan_pf_.finalize();
    if (plan_pf_.segs.size() > 4096) {
        throw std::runtime_error("segment buffer too small");
    }
    q.memcpy(d_segs_pf, plan_pf_.segs.data(), plan_pf_.segs.size() * sizeof(gemv_seg)).wait();
    // per-token-count prefill plans: the fp32 plan is sliced by kPfSlice, so T
    // rows are T/8 slices.  A prompt shorter than the chunk only pays for its
    // rounded-up token count instead of a full kMaxT chunk.
    for (int i = 0; i < kPfSlots; i++) {
        const int T = (i + 1) * kPfSlice;
        plan_pf_slot[i] = build_plan(T, kPfSlice, false);
        plan_pf_slot[i].finalize();
        if (plan_pf_slot[i].segs.size() > 1024) {
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
    if (pf8) {
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
        if (pf8_dec) {
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
        b.plan = build_plan(b.tb, b.tb, true);
        b.plan.finalize();
        if (b.plan.segs.size() > 1024) {
            throw std::runtime_error("segment buffer too small");
        }
        q.memcpy(b.d_segs, b.plan.segs.data(), b.plan.segs.size() * sizeof(gemv_seg)).wait();
    }
}

void engine::build_graphs() {
    if (cpu_mode || multi_dev) {
        build_plans();
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
