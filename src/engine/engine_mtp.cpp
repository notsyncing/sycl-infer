// MTP (NextN) speculative decoding for the qwen35 family.
//
// The GGUF ships one extra full-attention block after the main layers
// (`blk.<n_layer>.nextn.*`).  It is a single-step draft head: given the main
// model's hidden state h_{p-1} and the token t_p it predicts t_{p+1}.  Running it
// k times autoregressively yields k draft tokens, which one batched forward of
// the main model then verifies (greedy acceptance).  The MTP layer owns its own
// paged KV slice (attention layer index attn_layers()-1) so it shares the block
// table, the KV storage type and the prefix cache.
//
// Alignment (matches llama.cpp graph_mtp and the training objective):
//   MTP row at token position p  <->  input (emb(t_p), h_{p-1})  ->  t_{p+1}
// so the first row of a cycle uses the main hidden at the last committed
// position, and every later draft row uses the MTP's own hidden.
#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>

namespace si {

namespace {
int argmax_f(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}
bool mtp_dbg() {
    static const bool b = getenv("PF_MTP_DEBUG") != nullptr;
    return b;
}
} // namespace

#define MTPDBG(...)                                                                                                    \
    do {                                                                                                               \
        if (mtp_dbg()) {                                                                                               \
            fprintf(stderr, "[mtp] " __VA_ARGS__);                                                                     \
        }                                                                                                              \
    } while (0)

// ---------------------------------------------------------------------------
// The MTP layer's call plan: 4 layer calls + the shared LM head.
void engine::build_mtp_plan() {
    const hparams & hp = m.hp;
    const mtp_layer_t & M = m.mtp;
    seg_plan & p = plan_mtp_;
    p = seg_plan{};
    dnnl_gemm * D = dnnl_for(mtp_dev);
    auto seg = [&](const wt & w, const float * x, int xs, float * out, int os, const float * res) {
        gemv_seg s{};
        s.dev = mtp_dev;
        s.w = multi_dev ? wkey(mtp_dev, w.data) : wptr(0, w.data);
        s.type = w.type;
        s.K = w.K;
        s.n_rows = w.N;
        s.x = x;
        s.x_stride = xs;
        s.out = out;
        s.out_stride = os;
        s.residual = res;
        s.alpha = 1.0f;
        if (D) {
            s.wi8 = D->weight_data(s.w);
            s.wsc = D->weight_scales(s.w);
        }
        return s;
    };
    // call 0: wq / wk / wv
    p.begin_call(kMaxT, hp.n_embd / 256);
    p.add(seg(M.wq, d_mtp_xnorm, hp.n_embd, d_mtp_qbuf, hp.n_head * 2 * hp.head_dim, nullptr));
    p.add(seg(M.wk, d_mtp_xnorm, hp.n_embd, d_mtp_kbuf, hp.n_head_kv * hp.head_dim, nullptr));
    p.add(seg(M.wv, d_mtp_xnorm, hp.n_embd, d_mtp_vbuf, hp.n_head_kv * hp.head_dim, nullptr));
    p.set_xq(d_mtp_xnorm, nullptr, hp.n_embd, hp.n_embd, hp.n_embd);
    // call 1: wo (+ residual)
    p.begin_call(kMaxT, hp.n_head * hp.head_dim / 256);
    p.add(seg(M.wo, d_mtp_attn_out, hp.n_head * hp.head_dim, d_mtp_x, hp.n_embd, d_mtp_x));
    p.set_xq(d_mtp_attn_out, nullptr, hp.n_head * hp.head_dim, hp.n_head * hp.head_dim, hp.n_head * hp.head_dim);
    // call 2: ffn_gate + ffn_up
    p.begin_call(kMaxT, hp.n_embd / 256);
    p.add(seg(M.ffn_gate, d_mtp_xnorm, hp.n_embd, d_mtp_ffn, ffn_stride, nullptr));
    p.add(seg(M.ffn_up, d_mtp_xnorm, hp.n_embd, d_mtp_ffn + hp.n_ff, ffn_stride, nullptr));
    p.set_xq(d_mtp_xnorm, nullptr, hp.n_embd, hp.n_embd, hp.n_embd);
    // call 3: ffn_down (silu gate) (+ residual)
    p.begin_call(kMaxT, hp.n_ff / 256);
    p.add(seg(M.ffn_down, d_mtp_ffn, ffn_stride, d_mtp_x, hp.n_embd, d_mtp_x));
    p.set_xq(d_mtp_ffn, d_mtp_ffn + hp.n_ff, ffn_stride, ffn_stride, hp.n_ff);
    p.set_act_up(d_mtp_ffn + hp.n_ff, d_mtp_ffn);
    // call 4: shared LM head over the MTP hidden
    p.begin_call(kMaxT, hp.n_embd / 256);
    {
        const wt & head = M.shared_head.data ? M.shared_head : m.output;
        gemv_seg s = seg(head, d_mtp_hnorm0, hp.n_embd, d_logits, hp.n_vocab, nullptr);
        s.dev = 0; // the shared LM head only exists on the primary device
        s.w = multi_dev ? wkey(0, head.data) : wptr(0, head.data);
        p.add(s);
        p.set_xq(d_mtp_hnorm0, nullptr, hp.n_embd, hp.n_embd, hp.n_embd);
    }
    p.finalize();
    p.has_head = false;
    if (d_segs_mtp == nullptr) {
        d_segs_mtp = alloc_elems<gemv_seg>(p.segs.size());
    }
    q.memcpy(d_segs_mtp, p.segs.data(), p.segs.size() * sizeof(gemv_seg)).wait();
}

// One call of the MTP plan: quantize its activations once, then run the GEMMs.
// OneDNN cannot be recorded and its M==1 path is the grouped GEMV the
// multi-device decode already uses per segment, so this mirrors that dispatch.
void engine::mtp_gemv(int ci, int M) {
    const seg_plan & p = plan_mtp_;
    dnnl_gemm * D = dnnl_for(ci == 4 ? 0 : mtp_dev);
    const seg_plan::xq_t & xq = p.call_xq[(size_t)ci];
    if (D == nullptr || xq.x == nullptr) {
        throw std::runtime_error("mtp: the oneDNN int8 weight path is required");
    }
    // Any single-token draft call can run from a u4 copy (the shared LM head
    // under its private key, the MTP layer's own linears under their own), and
    // the u4 GEMV reads the even/odd activation split - so *every* call that
    // could hit a u4 weight must produce it, not just the head's.  The u4
    // tensors themselves are found through gemm_w4's key lookup, so no special
    // dispatch is needed here beyond the split.
    const bool w4_draft = M == 1 && (mtp_head_w4_ || mtp_layer_w4_);
    D->quantize(xq.x, xq.up, xq.x_stride, xq.up_stride, M, xq.K, /*do_split=*/w4_draft);
    const int gb = p.call_group_begin[(size_t)ci];
    const int gc = p.call_group_count[(size_t)ci];
    for (int g = 0; g < gc; g++) {
        const seg_plan::group_t & gr = p.groups[(size_t)(gb + g)];
        for (int j = 0; j < gr.n; j++) {
            const gemv_seg & s = d_segs_mtp[gr.off + j];
            const void * wk = (ci == 4 && mtp_head_w4_) ? (const void *)mtp_head_w4_key_ : s.w;
            if (D->gemm_w4(wk, s.residual, s.alpha, M, s.K, s.out, s.out_stride)) {
                continue;
            }
            if (D->gemm(wk, s.residual, s.alpha, M, s.K, s.out, s.out_stride)) {
                continue;
            }
            throw std::runtime_error("mtp: oneDNN GEMM failed for a layer tensor");
        }
    }
}

// ---------------------------------------------------------------------------
// Run the MTP layer over `n` tokens (mode-1 layout: one row, n <= kMaxT) at
// positions pos0..pos0+n-1.  `h` is the main model's hidden [token][n_embd] for
// those same tokens (or null when n == 1 and hprev carries the hidden); token i
// takes h[i-1] and the first token of the row takes hprev.
void engine::mtp_forward(const int32_t * toks, const float * h, const float * hprev, int n, int slot, int pos0,
                         bool with_head) {
    const hparams & hp = m.hp;
    const mtp_layer_t & M = m.mtp;
    dnnl_gemm * D = dnnl_for(mtp_dev);
    compute_backend & be = multi_dev ? *backends_[(size_t)mtp_dev] : backend();

    step_info * inf = d_mtp_info;
    std::memset(inf, 0, sizeof(step_info));
    std::memcpy(inf->mrope_sections, hp.rope_sections, sizeof(hp.rope_sections));
    inf->n_rows = 1;
    inf->n_real = n;
    inf->tpb = kMaxT;
    inf->n_real_row[0] = n;
    inf->pos[0] = pos0;
    inf->slot[0] = slot;
    inf->active[0] = 1;
    inf->pc_active = 0;
    inf->mtp_dt = 0;
    for (int i = 0; i < n; i++) {
        inf->tokens[i] = toks[i];
        inf->img_row[i] = -1;
    }

    // 1. concat(enorm(emb(tok)), hnorm(h_prev))
    MTPDBG("forward n=%d pos=%d head=%d concat\n", n, pos0, (int)with_head);
    be.mtp_concat(wptr(0, m.tok_embd.data), m.tok_embd.type, m.tok_embd_row_bytes, wf32(mtp_dev, M.enorm),
                  wf32(mtp_dev, M.hnorm), h, hprev, inf, d_mtp_cat, hp.n_embd, hp.rms_eps);
    MTPDBG("forward eh_proj\n");
    // 2. eh_proj -> d_mtp_x
    {
        const void * ek = wkey(multi_dev ? mtp_dev : 0, M.eh_proj.data);
        D->quantize(d_mtp_cat, nullptr, 2 * hp.n_embd, 0, n, 2 * hp.n_embd, false);
        if (!(D->gemm_w4(ek, nullptr, 1.0f, n, 2 * hp.n_embd, d_mtp_x, hp.n_embd)
              || D->gemm(ek, nullptr, 1.0f, n, 2 * hp.n_embd, d_mtp_x, hp.n_embd))) {
            throw std::runtime_error("mtp: eh_proj GEMM failed");
        }
    }
    // 3. attention block
    MTPDBG("forward attn_norm\n");
    be.rmsnorm(d_mtp_x, wf32(mtp_dev, M.attn_norm), d_mtp_xnorm, n, hp.n_embd, hp.rms_eps);
    mtp_gemv(0, n);
    MTPDBG("forward qk_norm_rope\n");
    const char * kp;
    const char * vp;
    const char * ksc;
    const char * vsc;
    kv_layer_ptrs(attn_layers() - 1, kp, vp, ksc, vsc);
    if (mtp_dbg()) {
        static bool once = false;
        if (!once) {
            once = true;
            fprintf(stderr, "[mtp] attn_layers=%d mtp_local=%d multi_dev=%d\n", attn_layers(), mtp_attn_local_,
                    (int)multi_dev);
            for (int a = 0; a <= attn_layers() - 1; a++) {
                const char *p1;
                const char *p2;
                const char *p3;
                const char *p4;
                kv_layer_ptrs(a, p1, p2, p3, p4);
                fprintf(stderr, "[mtp]   attn a=%2d dev=%d kp=%p vp=%p\n", a, attn_dev(a), (void *)p1, (void *)p2);
            }
        }
    }
    // The draft is a *decode* (one row over the MTP layer's KV), so it wants key
    // parallelism: one warp per head looping the whole KV cost 2451 vs 183
    // ms/cycle at 128k.  The split count is derived from the KV length (the same
    // ~512 keys per split the prefill uses) rather than taken as the full decode
    // cap: at short context a fixed 256-split grid measurably degrades the
    // *draft* (acceptance 1.27 vs 2.29 per cycle on a 320-token prompt), while
    // the plain decode - which does use the full cap - has no acceptance to
    // lose.  A multi-row call (the prompt chunks) already has n*n_head
    // workgroups, so it keeps the fused single split; that is also why the
    // partials buffer only has to cover max(mtp_splits, kMaxT*n_head) entries.
    const int max_nkv = pos0 + n;
    const int nsp = std::min(std::max((max_nkv + 511) / 512, 1), mtp_splits);
    be.qk_norm_rope(d_mtp_qbuf, d_mtp_kbuf, d_mtp_vbuf, wf32(mtp_dev, M.q_norm), wf32(mtp_dev, M.k_norm), (void *)kp, (void *)vp,
                    d_tables, inf, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_rot, hp.rope_base, hp.rms_eps,
                    max_blocks, 1, n, ksc, vsc);
    const bool fused = (nsp == 1);
    be.attn(d_mtp_qbuf, d_mtp_qbuf, kp, vp, d_mtp_partials, d_tables, hp.n_head, hp.n_head_kv, hp.head_dim, nsp, inf,
            hp.attn_scale, max_blocks, 1, n, fused ? d_mtp_attn_out : nullptr, -1, ksc, vsc);
    if (!fused) {
        be.attn_combine(d_mtp_partials, d_mtp_qbuf, d_mtp_attn_out, inf, hp.n_head, hp.head_dim, nsp, 1, n);
    }
    MTPDBG("forward wo\n");
    mtp_gemv(1, n); // wo + residual
    // 4. FFN
    be.rmsnorm(d_mtp_x, wf32(mtp_dev, M.post_attn_norm), d_mtp_xnorm, n, hp.n_embd, hp.rms_eps);
    mtp_gemv(2, n);
    mtp_gemv(3, n);
    // 5. the AR draft chain seeds the next step with the MTP hidden *before*
    // the shared head norm (llama.cpp: the draft context's pre-norm hidden)
    be.copy_row(d_mtp_x, d_mtp_raw, inf, hp.n_embd, -1);
    be.rmsnorm(d_mtp_x, wf32(mtp_dev, M.shared_head_norm), d_mtp_hnorm, n, hp.n_embd, hp.rms_eps);
    if (with_head) {
        MTPDBG("forward head\n");
        if (d_mtp_hnorm0 != d_mtp_hnorm) {
            // the shared LM head lives on the primary device
            dev_queue(0).memcpy(d_mtp_hnorm0, d_mtp_hnorm, (size_t)n * hp.n_embd * 4).wait();
        }
        mtp_gemv(4, n);
    }
    MTPDBG("forward done\n");
}

// ---------------------------------------------------------------------------
// Verify forward: the main model over `n` tokens at positions pos0.., with the
// LM head batched so every row's logits land in d_logits.  Also captures the
// per-token recurrent state (mtp_hist) and saves the conv window.
void engine::mtp_verify(const std::vector<int> & toks, int n, int slot, int pos0) {
    const hparams & hp = m.hp;
    if (n > kMaxT) {
        throw std::runtime_error("mtp_verify: n > kMaxT");
    }
    if (n > kMaxB) {
        throw std::runtime_error("mtp_verify: too many logit rows");
    }
    // save the conv window (rows 1,2) of every GDN layer's slot before the
    // forward advances it; the rollback derives the new window from d_qkv.
    // conv state is [layer][slot][conv_k-1][conv_dim]
    const size_t conv_dim = (size_t)hp.qkv_dim();
    const size_t conv_per = (size_t)(hp.conv_k - 1) * conv_dim;
    const int ndev = (int)as_.size();
    for (int dev = 0; dev < ndev; dev++) {
        if (d_mtp_convsave_[(size_t)dev] == nullptr) {
            continue;
        }
        sycl::queue & qd = dev_queue(dev);
        float * cstate = as_[(size_t)dev].conv_state;
        for (int il = 0; il < hp.n_layer; il++) {
            if (!hp.is_recr(il) || layer_dev_[(size_t)il] != dev) {
                continue;
            }
            const int gl = layer_gdn_local_[(size_t)il];
            float * src = cstate + (size_t)gl * kMaxB * conv_per + conv_dim; // slot 0, row 1
            qd.memcpy(d_mtp_convsave_[(size_t)dev] + (size_t)gl * 2 * conv_dim, src, conv_dim * 4);
            qd.memcpy(d_mtp_convsave_[(size_t)dev] + ((size_t)gl * 2 + 1) * conv_dim, src + conv_dim, conv_dim * 4);
        }
    }
    if (getenv("PF_MTP_STATECHK") != nullptr && !d_mtp_hist_.empty()) {
        // snapshot the live recurrent state of (device 0, GDN layer 0) so the
        // rollback can be diffed against a plain-decode reference
        const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state;
        const size_t conv_per = (size_t)(hp.conv_k - 1) * hp.qkv_dim();
        h_save_.resize(gdn_per + conv_per);
        sycl::queue & qd0 = dev_queue(0);
        qd0.memcpy(h_save_.data(), as_[0].gdn_state, gdn_per * 4).wait();
        qd0.memcpy(h_save_.data() + gdn_per, as_[0].conv_state, conv_per * 4).wait();
        // the layer 0 slot-0 conv/GDN state starts at the base of the buffers
    }
    setup_pf_info(d_info, toks, 0, n, slot, pos0);
    // The verify is speculative: it must leave the live recurrent state and the
    // prefix cache untouched (mtp_dry), but it DOES write the per-token GDN
    // state snapshots (pc_active + mtp_dt): the commit rewinds the state to the
    // snapshot after the last accepted row (llama.cpp's n_rs_seq mechanism).
    // The conv window is rebuilt from the saved raw taps instead, because
    // conv_state_update only snapshots at 32-token boundaries.
    d_info->pc_active = (d_mtp_hist_.empty() || d_mtp_hist_[0] == nullptr) ? 0 : 1;
    d_info->mtp_dt = 1;
    d_info->mtp_dry = 1;
    for (int t = 0; t < mtp_nsnap && t < kPcMapLen; t++) {
        d_info->pc_row_slot[t] = t;
    }
    MTPDBG("verify record_forward n=%d pos=%d\n", n, pos0);
    if (getenv("PF_MTP_INFOCHK") != nullptr) {
        fprintf(stderr, "[mtp] infochk before rf: mtp_dt=%d pc_active=%d slot0=%d slot1=%d sizeof=%zu\n",
                d_info->mtp_dt, d_info->pc_active, d_info->pc_row_slot[0], d_info->pc_row_slot[1],
                sizeof(step_info));
    }
    // PF_MTP_VERIFY_PAD=<m>: run the verify's GEMMs at a padded M (the real
    // tokens are still n, via n_real_row).  oneDNN's chosen kernel for the SIn
    // int8 weights is a plain jit:gemm, whose sweet spot is a SIMD-aligned M;
    // PF_MTP_VERIFY_M32=1 is the equivalent pad-to-kMaxT shortcut.
    static const int v_pad = [] {
        const char * e = getenv("PF_MTP_VERIFY_PAD");
        if (e && atoi(e) > 0) {
            return atoi(e);
        }
        const char * m32 = getenv("PF_MTP_VERIFY_M32");
        return (m32 && atoi(m32) != 0) ? kMaxT : 0;
    }();
    const int rows_vf = v_pad > n ? v_pad : n;
    record_forward(2, plan_vf_, d_segs_vf, rows_vf, d_segs_vf);
    MTPDBG("verify record_forward returned, syncing\n");
    sync_all();
    if (const char * sd = getenv("PF_MTP_SNAPDUMP"); sd && pos0 == 0) {
        const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state;
        const size_t per = gdn_per + (size_t)(hp.conv_k - 1) * hp.qkv_dim();
        const int ng = n_gdn_dev_[0];
        FILE * f = fopen(sd, "wb");
        if (f) {
            std::vector<float> row(gdn_per);
            for (int t = 0; t < mtp_nsnap && t < n; t++) {
                dev_queue(0).memcpy(row.data(), d_mtp_hist_[0] + (size_t)t * ng * per, gdn_per * 4).wait();
                fwrite(row.data(), 4, gdn_per, f);
            }
            fclose(f);
            fprintf(stderr, "[mtp] snapdump %s n=%d\n", sd, n);
        }
    }
    if (getenv("PF_MTP_SNAPCHK") != nullptr && pos0 == 22) {
        fprintf(stderr, "[mtp] gdnmap:");
        for (int il = 0; il < std::min(8, hp.n_layer); il++) {
            fprintf(stderr, " il%d(dev%d,gl%d)", il, layer_dev_[(size_t)il],
                    hp.is_recr(il) ? layer_gdn_local_[(size_t)il] : -1);
        }
        fprintf(stderr, "  ng_dev=[%d,%d]\n", n_gdn_dev_[0], n_gdn_dev_.size() > 1 ? n_gdn_dev_[1] : -1);
    }
    if (getenv("PF_MTP_SNAPCHK") != nullptr && pos0 == 0) {
        const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state;
        for (int dev = 0; dev < (int)as_.size(); dev++) {
            if (d_mtp_hist_[(size_t)dev] == nullptr || n_gdn_dev_[(size_t)dev] == 0) {
                continue;
            }
            sycl::queue & qd = dev_queue(dev);
            const size_t per = gdn_per + (size_t)(hp.conv_k - 1) * hp.qkv_dim();
            const int ng = n_gdn_dev_[(size_t)dev];
            std::vector<float> s0(gdn_per), slast(gdn_per), live(gdn_per);
            qd.memcpy(s0.data(), d_mtp_hist_[(size_t)dev], gdn_per * 4).wait();
            qd.memcpy(slast.data(), d_mtp_hist_[(size_t)dev] + (size_t)(n - 1) * (size_t)ng * per, gdn_per * 4)
                .wait();
            const double dlast = [&] {
                double mx = 0;
                for (size_t i = 0; i < gdn_per; i++) {
                    mx = std::max(mx, (double)std::fabs(slast[i] - live[i]));
                }
                return mx;
            }();
            fprintf(stderr, "[mtp] snapchk dev=%d slot(last=%d)-vs-live maxdiff=%.6f\n", dev, n - 1, dlast);
            std::vector<float> probe(gdn_per);
            for (int t = 0; t < n && t < 8; t++) {
                qd.memcpy(probe.data(), d_mtp_hist_[(size_t)dev] + (size_t)t * (size_t)ng * per, gdn_per * 4).wait();
                size_t nz = 0;
                double mx = 0;
                for (size_t i = 0; i < gdn_per; i++) {
                    if (probe[i] != 0.f) {
                        nz++;
                    }
                    mx = std::max(mx, (double)std::fabs(probe[i]));
                }
                fprintf(stderr, "[mtp] snapchk dev=%d slot %d: nonzero=%zu max|.|=%.4f\n", dev, t, nz, mx);
            }
            qd.memcpy(live.data(), as_[(size_t)dev].gdn_state, gdn_per * 4).wait();
            double d = 0, nb = 0, n0 = 0;
            for (size_t i = 0; i < gdn_per; i++) {
                d = std::max(d, (double)std::fabs(s0[i] - live[i]));
                nb = std::max(nb, (double)std::fabs(live[i]));
                n0 = std::max(n0, (double)std::fabs(s0[i]));
            }
            fprintf(stderr, "[mtp] snapchk dev=%d slot0-vs-live maxdiff=%.6f |live|=%.4f |slot0|=%.4f\n", dev, d, nb,
                    n0);
        }
    }
    MTPDBG("verify synced\n");
    d_info->pc_active = 0;
    d_info->mtp_dt = 0;
    d_info->mtp_dry = 0;
}

// ---------------------------------------------------------------------------
// Roll the recurrent state back to "after j+1 tokens of the verify batch".
//   GDN: copy history slot j into the live state (per layer, device-local).
//   conv: rebuild the 3-row window from the verify's raw qkv rows.
void engine::mtp_rollback(int j) {
    const hparams & hp = m.hp;
    const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state;
    const size_t conv_dim = (size_t)hp.qkv_dim();
    const size_t conv_per = (size_t)(hp.conv_k - 1) * conv_dim;
    const int ndev = (int)as_.size();
    for (int dev = 0; dev < ndev; dev++) {
        if (d_mtp_hist_[(size_t)dev] == nullptr) {
            continue;
        }
        sycl::queue & qd = dev_queue(dev);
        const int ng = n_gdn_dev_[(size_t)dev];
        const size_t per = gdn_per + conv_per;
        const float * hist = d_mtp_hist_[(size_t)dev];
        float * save = d_mtp_convsave_[(size_t)dev];
        const float * qtap = d_mtp_qsave_[(size_t)dev];
        for (int il = 0; il < hp.n_layer; il++) {
            if (!hp.is_recr(il) || layer_dev_[(size_t)il] != dev) {
                continue;
            }
            const int gl = layer_gdn_local_[(size_t)il];
            // GDN state: the dry verify snapshotted the state after every token
            // into d_mtp_hist_[token]; row j is the last committed token, so its
            // snapshot is exactly the committed state.
            qd.memcpy(as_[(size_t)dev].gdn_state + (size_t)gl * kMaxB * gdn_per,
                      hist + (size_t)j * (size_t)ng * per + (size_t)gl * per, gdn_per * 4);
            // conv window of the last three committed tokens: rows j-2..j come
            // from the verify's own taps, older ones from the pre-verify window.
            float * cs = as_[(size_t)dev].conv_state + (size_t)gl * kMaxB * conv_per; // slot 0
            const float * v2 = qtap + (size_t)gl * mtp_nsnap * conv_dim + (size_t)j * conv_dim;
            const float * v1 = j >= 1 ? qtap + (size_t)gl * mtp_nsnap * conv_dim + (size_t)(j - 1) * conv_dim
                                      : save + (size_t)gl * 2 * conv_dim + conv_dim;
            const float * v0 = j >= 2 ? qtap + (size_t)gl * mtp_nsnap * conv_dim + (size_t)(j - 2) * conv_dim
                                      : (j == 1 ? save + (size_t)gl * 2 * conv_dim + conv_dim
                                                : save + (size_t)gl * 2 * conv_dim);
            qd.memcpy(cs, v0, conv_dim * 4);
            qd.memcpy(cs + conv_dim, v1, conv_dim * 4);
            qd.memcpy(cs + 2 * conv_dim, v2, conv_dim * 4);
        }
    }
}

// ---------------------------------------------------------------------------
// NOTE on the verify's accept test (PF_MTP_ARGMAX_CPU=1 for the old path): the
// per-row argmax runs on the device and copies back k+1 ints instead of
// (k+1)*n_vocab floats (~7 MB/cycle at k=6).  Measured: this does NOT make the
// engine faster - 200.2/199.8 s (host scan) against 199.8/200.1 s (device) for a
// 2000-token run, and user+sys CPU time is unchanged (~150 s both ways).  The
// transfer was never on the critical path: the verify's own sync dominates, and
// the copy overlaps the next cycle's draft.  It is kept because it is strictly
// less host work and bit-identical (ties resolve to the lowest index, matching
// the host `v[i] > v[best]` scan), but do not expect it to show up in t/s.
//
// Speculative generation loop (greedy).  Emits exactly the tokens a plain greedy
// decode would: the target's own sampled token is always the last one emitted
// per cycle, so acceptance can only ever skip work, never change the output.
std::vector<int> engine::generate_mtp(const std::vector<int> & prompt, const gen_params & gp,
                                      const std::function<bool(int)> & cb, std::vector<float> * first_logits) {
    reset_single();
    const hparams & hp = m.hp;
    static const bool dbg_gen = getenv("PF_DUMP_GEN") != nullptr;
    sampler_state ss;
    ss.seed(gp.seed ? gp.seed : std::random_device{}());
    std::vector<int> out;
    const int nprompt = (int)prompt.size();
    if (nprompt == 0) {
        return out;
    }
    dev_queue(0).memset(d_mtp_hprev, 0, (size_t)hp.n_embd * 4).wait();

    // KV blocks: reuse the deepest cached block chain of this prompt first
    // (pc_admit restores that node's KV and recurrent state and pins its
    // blocks), then allocate only what is still missing: the prompt tail, the
    // draft window and a block of slack.
    std::vector<int> blocks;
    const int matched = pc_admit(0, prompt, blocks);
    const int pretail = nprompt - matched;
    const int need = (pretail + kBlockSize - 1) / kBlockSize + (mtp_k + 2 + kBlockSize - 1) / kBlockSize + 1;
    for (int i = 0; i < need; i++) {
        int b = alloc_block();
        if (b < 0) {
            throw std::runtime_error("out of KV blocks");
        }
        blocks.push_back(b);
    }
    if (dbg_gen && matched > 0) {
        fprintf(stderr, "[mtp] prefix cache: reused %d prompt tokens\n", matched);
    }
    set_table(0, blocks);

    // prefill (main + MTP KV).  The MTP is driven chunk by chunk because the
    // draft head needs the main hidden of the previous token, which only exists
    // for the tokens already forwarded.
    if (getenv("PF_MTP_DECODE_H") != nullptr) {
        // Diagnostic / workaround: the mode-2 chunk forward leaves a correct
        // per-position hidden only for some rows, so build the draft head's h
        // rows from a one-token-at-a-time decode (the single-token path is the
        // one the whole engine is validated against).
        std::vector<float> htmp((size_t)nprompt * hp.n_embd);
        for (int i = 0; i < nprompt; i++) {
            int32_t tk = prompt[(size_t)i];
            int32_t pp = i;
            int32_t sl = 0;
            decode_batch(&tk, &pp, &sl, 1);
            // mode 0 leaves the single token's post-output-norm hidden in d_xnorm
            dev_queue(0).memcpy(htmp.data() + (size_t)i * hp.n_embd, d_xnorm, (size_t)hp.n_embd * 4).wait();
            if (const char * sq = getenv("PF_MTP_STATESEQ")) {
                const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state;
                std::vector<float> row(gdn_per);
                dev_queue(0).memcpy(row.data(), as_[0].gdn_state, gdn_per * 4).wait();
                FILE * f = fopen(sq, i == 0 ? "wb" : "ab");
                if (f) {
                    fwrite(row.data(), 4, gdn_per, f);
                    fclose(f);
                }
            }
            if (getenv("PF_MTP_HDUMPS")) {
                std::vector<float> lg((size_t)hp.n_vocab);
                fetch_logits(0, lg.data());
                const int am = argmax_f(lg.data(), hp.n_vocab);
                fprintf(stderr, "[mtp] dec %d argmax=%d\n", i, am);
            }
        }
        for (int i = 0; i < nprompt; i++) {
            dev_queue(0)
                .memcpy(d_mtp_main_h + (size_t)i * hp.n_embd, htmp.data() + (size_t)i * hp.n_embd,
                        (size_t)hp.n_embd * 4)
                .wait();
        }
    }
    // Register the prompt's checkpoints with the cache (same contract as the
    // scheduler: pc_commit after the prefill, before generation).
    if (nprompt > 0) {
        pc_commit(0, prompt, blocks, nprompt);
    }
    std::vector<int32_t> chunk;
    const bool decode_h = getenv("PF_MTP_DECODE_H") != nullptr;
    for (int pos = matched; decode_h ? false : pos < nprompt;) {
        // Take the largest prefill batch the plan supports (batched_prefill_fit,
        // up to kMaxB*kMaxT) and then walk *its captured hidden* in kMaxT-token
        // pieces for the MTP layer.  Doing kMaxT-token prefill batches instead
        // costs a prefill_flush() plus a host-blocking copy per 32 tokens, which
        // serialises the 3-phase multi-device pipeline: measured e2e_ttft at
        // 131k was 1475 s against 311 s for the same prompt through the
        // scheduler, and 64k 697 s against 117 s.
        const int rem = nprompt - pos;
        const int fit = batched_prefill_fit(rem);
        const int nb = (fit >= 1 && fit <= rem) ? fit : std::min(kMaxT, rem);
        if (dbg_gen) {
            fprintf(stderr, "[mtp] prefill batch pos=%d n=%d\n", pos, nb);
        }
        // mode 2 is the prefill path the multi-device engine actually uses; its
        // output norm leaves the per-position main hidden in d_xnorm
        prefill_text(prompt, pos, nb);
        // The multi-device prefill pipeline defers the last chunk's device-1 and
        // head phases to the next call (or to prefill_flush).  The per-position
        // main hidden (and the capture) is only written by that head phase, so
        // it must be flushed before the draft head reads it.
        prefill_flush();
        if (dbg_gen) {
            fprintf(stderr, "[mtp] prefill batch done\n");
        }
        // mtp_capture copies the batch's rows in token order, and mtp_concat
        // indexes the hidden by the *global* slot (r*tpb + t) with h_prev only
        // at slot 0, so a piece starting at `off` passes the capture base at
        // `off` and takes its preceding token (the same capture, unless this is
        // the batch's first piece) as h_prev.
        for (int off = 0; off < nb; off += kMaxT) {
            const int n = std::min(kMaxT, nb - off);
            chunk.assign(prompt.begin() + pos + off, prompt.begin() + pos + off + n);
            const float * h = d_mtp_main_h + (size_t)off * hp.n_embd;
            const float * hprev = off > 0 ? (d_mtp_main_h + (size_t)(off - 1) * hp.n_embd) : d_mtp_hprev;
            mtp_forward(chunk.data(), h, hprev, n, 0, pos + off, /*with_head=*/false);
            if (dbg_gen) {
                fprintf(stderr, "[mtp] mtp prefill piece off=%d n=%d\n", off, n);
            }
        }
        dev_queue(0).memcpy(d_mtp_hprev, d_mtp_main_h + (size_t)(nb - 1) * hp.n_embd, (size_t)hp.n_embd * 4).wait();
        pos += nb;
    }

    if (getenv("PF_MTP_HVEC") != nullptr) {
        sync_all();
        const int nr = 8;
        std::vector<float> hb((size_t)nr * hp.n_embd);
        q.memcpy(hb.data(), d_mtp_main_h, hb.size() * 4).wait();
        for (int i = 0; i < nr; i++) {
            double n = 0;
            for (int j = 0; j < hp.n_embd; j++) {
                n += (double)hb[(size_t)i * hp.n_embd + j] * hb[(size_t)i * hp.n_embd + j];
            }
            double dprev = 0;
            if (i > 0) {
                for (int j = 0; j < hp.n_embd; j++) {
                    dprev = std::max(dprev, (double)std::fabs(hb[(size_t)i * hp.n_embd + j]
                                                              - hb[(size_t)(i - 1) * hp.n_embd + j]));
                }
            }
            fprintf(stderr, "[mtp] hvec[%d] ||.|=%.4f d_prev=%.6f v=%.4f,%.4f,%.4f\n", i, std::sqrt(n), dprev,
                    hb[(size_t)i * hp.n_embd], hb[(size_t)i * hp.n_embd + 1], hb[(size_t)i * hp.n_embd + 2]);
        }
        {
            std::vector<float> lh((size_t)hp.n_embd);
            q.memcpy(lh.data(), d_last_hidden, (size_t)hp.n_embd * 4).wait();
            double d = 0;
            for (int j = 0; j < hp.n_embd; j++) {
                d = std::max(d, (double)std::fabs(lh[(size_t)j] - hb[(size_t)21 * hp.n_embd + j]));
            }
            fprintf(stderr, "[mtp] hvec d(last_hidden, capt[21])=%.6f\n", d);
        }
    }
    std::vector<float> logits = run_head();
    if (first_logits) {
        *first_logits = logits;
    }
    int pos = nprompt;
    int tok = sample_token(logits.data(), hp.n_vocab, gp, out, ss);
    if (getenv("PF_MTP_DIAG") != nullptr && nprompt >= 2) {
        // Run the MTP over the whole prompt (h=zeros, like the prefill pass) and
        // check every row's argmax against the actual next prompt token.  Row i
        // must predict prompt[i+1]; the last row must predict the main model's
        // first generated token.
        std::vector<float> h0((size_t)hp.n_embd, 0.f);
        const int dn = std::min(nprompt, kMaxB); // d_logits holds kMaxB rows
        float * d_h0 = (float *)dev_alloc_on(0, (size_t)hp.n_embd * 4);
        dev_queue(0).memcpy(d_h0, h0.data(), (size_t)hp.n_embd * 4).wait();
        mtp_forward(chunk.data(), d_mtp_main_h, d_h0, dn, 0, 0, /*with_head=*/true);
        sync_all();
        q.memcpy(h_logits, d_logits, (size_t)dn * hp.n_vocab * 4).wait();
        for (int i = 0; i < dn; i++) {
            const int want = (i + 1 < nprompt) ? prompt[(size_t)i + 1] : ((i < 32) ? tok : tok);
            const int got = argmax_f(h_logits + (size_t)i * hp.n_vocab, hp.n_vocab);
            fprintf(stderr, "[mtp] row %2d tok=%6d -> %6d want %6d %s\n", i, prompt[(size_t)i], got, want,
                    got == want ? "OK" : "XX");
        }
        {
            // Is d_xnorm row i really the hidden at position i?  Run the MAIN head
            // on each row: a correct row must predict prompt[i+1].
            std::vector<int> probe;
            for (int i = 0; i < 32; i++) { probe.push_back(i); }
            for (int i : probe) {
                q.memcpy(d_last_hidden, d_mtp_main_h + (size_t)i * hp.n_embd, (size_t)hp.n_embd * 4).wait();
                std::vector<float> lg2 = run_head();
                const int got = argmax_f(lg2.data(), hp.n_vocab);
                const int want = (i + 1 < nprompt) ? prompt[(size_t)i + 1] : tok;
                fprintf(stderr, "[mtp] mainhead(xnorm[%d]) -> %d want %d %s\n", i, got, want, got == want ? "OK" : "XX");
            }
        }
        {
            // norms of the layer's per-row activations: a row where the input h
            // or the concat/x collapses points at the broken stage
            const int nr = std::min(dn, 6);
            std::vector<float> buf((size_t)nr * hp.n_embd);
            std::vector<float> cat((size_t)nr * 2 * hp.n_embd);
            q.memcpy(buf.data(), d_xnorm, (size_t)nr * hp.n_embd * 4).wait();
            for (int i = 0; i < nr; i++) {
                double n = 0;
                for (int j = 0; j < hp.n_embd; j++) {
                    n += (double)buf[(size_t)i * hp.n_embd + j] * buf[(size_t)i * hp.n_embd + j];
                }
                fprintf(stderr, "[mtp]   ||xnorm[%d]||=%.4f\n", i, std::sqrt(n));
            }
            q.memcpy(cat.data(), d_mtp_cat, (size_t)nr * 2 * hp.n_embd * 4).wait();
            q.memcpy(buf.data(), d_mtp_x, (size_t)nr * hp.n_embd * 4).wait();
            std::vector<float> hn((size_t)nr * hp.n_embd);
            q.memcpy(hn.data(), d_mtp_hnorm, (size_t)nr * hp.n_embd * 4).wait();
            for (int i = 0; i < nr; i++) {
                double ne = 0, nh = 0, nx = 0, nhn = 0;
                for (int j = 0; j < hp.n_embd; j++) {
                    ne += (double)cat[(size_t)i * 2 * hp.n_embd + j] * cat[(size_t)i * 2 * hp.n_embd + j];
                    nh += (double)cat[(size_t)i * 2 * hp.n_embd + hp.n_embd + j]
                          * cat[(size_t)i * 2 * hp.n_embd + hp.n_embd + j];
                    nx += (double)buf[(size_t)i * hp.n_embd + j] * buf[(size_t)i * hp.n_embd + j];
                    nhn += (double)hn[(size_t)i * hp.n_embd + j] * hn[(size_t)i * hp.n_embd + j];
                }
                fprintf(stderr, "[mtp]   row %d ||e||=%.4f ||hh||=%.4f ||x||=%.4f ||hnorm||=%.4f\n", i,
                        std::sqrt(ne), std::sqrt(nh), std::sqrt(nx), std::sqrt(nhn));
            }
        }
        sycl::free(d_h0, dev_queue(0));
        if (const char * dp = getenv("PF_MTP_DUMP")) {
            // dump the prompt, the h rows fed to the MTP and its logits so a
            // host reference (Python) can reproduce the layer step by step
            FILE * f = fopen(dp, "wb");
            if (f) {
                // make sure every producing queue is idle, then read with the
                // primary device's queue (the same one the plan uses)
                sync_all();
                sycl::queue & qd = dev_queue(0);
                const int ni = nprompt;
                std::vector<float> hb((size_t)dn * hp.n_embd);
                q.memcpy(hb.data(), d_mtp_main_h, (size_t)dn * hp.n_embd * 4).wait();
                fwrite(&ni, 4, 1, f);
                fwrite(&dn, 4, 1, f);
                fwrite(&hp.n_embd, 4, 1, f);
                fwrite(&hp.n_vocab, 4, 1, f);
                fwrite(&hp.n_head, 4, 1, f);
                fwrite(&hp.n_head_kv, 4, 1, f);
                fwrite(&hp.head_dim, 4, 1, f);
                fwrite(prompt.data(), 4, prompt.size(), f);
                fwrite(hb.data(), 4, hb.size(), f);
                auto dumpf = [&](const float * src, size_t n) {
                    std::vector<float> tmp(n);
                    qd.memcpy(tmp.data(), src, n * 4).wait();
                    fwrite(tmp.data(), 4, n, f);
                };
                dumpf(d_mtp_cat, (size_t)dn * 2 * hp.n_embd);
                dumpf(d_mtp_qbuf, (size_t)dn * hp.n_head * 2 * hp.head_dim);
                dumpf(d_mtp_kbuf, (size_t)dn * hp.n_head_kv * hp.head_dim);
                dumpf(d_mtp_vbuf, (size_t)dn * hp.n_head_kv * hp.head_dim);
                dumpf(d_mtp_attn_out, (size_t)dn * hp.n_head * hp.head_dim);
                dumpf(d_mtp_x, (size_t)dn * hp.n_embd);
                dumpf(d_mtp_hnorm, (size_t)dn * hp.n_embd);
                fwrite(h_logits, 4, (size_t)dn * hp.n_vocab, f);
                fclose(f);
                fprintf(stderr, "[mtp] dumped %s (ni=%d dn=%d)\n", dp, ni, dn);
            }
        }
    }
    out.push_back(tok);
    if (getenv("PF_MTP_VPROBE") != nullptr) {
        // Run the MAIN model over the first 16 prompt tokens with the batched
        // head (the verify plan) and check each row against the next prompt
        // token.  This uses no MTP code and no capture: it tests whether the
        // mode-2 forward produces a correct hidden state for every position.
        std::vector<int> pv(prompt.begin(), prompt.begin() + std::min<int>(16, nprompt));
        mtp_verify(pv, (int)pv.size(), 0, 0);
        dev_queue(0).memcpy(h_logits, d_logits, pv.size() * (size_t)hp.n_vocab * 4).wait();
        for (size_t i = 0; i < pv.size(); i++) {
            const int got = argmax_f(h_logits + i * (size_t)hp.n_vocab, hp.n_vocab);
            const int want = (i + 1 < (size_t)nprompt) ? prompt[i + 1] : tok;
            fprintf(stderr, "[mtp] vprobe row %2zu tok=%6d -> %6d want %6d %s\n", i, prompt[i], got, want,
                    got == want ? "OK" : "XX");
        }
    }
    if (getenv("PF_MTP_HOSTCMP") != nullptr) {
        // d_mtp_main_h and d_last_hidden are HOST USM, so compare them directly
        // (no SYCL queue involved, no ordering question)
        const int rows[4] = {0, 1, nprompt - 2, nprompt - 1};
        for (int k = 0; k < 4; k++) {
            const int i = rows[k];
            if (i < 0 || i >= nprompt) continue;
            const float * a = d_mtp_main_h + (size_t)i * hp.n_embd;
            double nb = 0;
            for (int j = 0; j < hp.n_embd; j++) {
                nb = std::max(nb, (double)std::fabs(a[j]));
            }
            fprintf(stderr, "[mtp] hostcmp capt[%d] ||.|=%.4f v=%.4f,%.4f,%.4f\n", i, nb, a[0], a[1], a[2]);
        }
        {
            const float * a = d_mtp_main_h + (size_t)(nprompt - 1) * hp.n_embd;
            const float * b = d_last_hidden;
            double d = 0, nb = 0;
            for (int j = 0; j < hp.n_embd; j++) {
                d = std::max(d, (double)std::fabs(a[j] - b[j]));
                nb = std::max(nb, (double)std::fabs(b[j]));
            }
            fprintf(stderr, "[mtp] hostcmp capt[last] vs d_last_hidden: maxdiff=%.6f |last_hidden|=%.4f\n", d, nb);
        }
    }
    if (const char * sp = getenv("PF_MTP_STATEDUMP")) {
        // dump the post-prompt recurrent state of (device 0, GDN layer 0) so the
        // mode-2 prefill and the token-by-token decode paths can be compared
        const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state;
        const size_t conv_per = (size_t)(hp.conv_k - 1) * hp.qkv_dim();
        std::vector<float> base(gdn_per), cv(conv_per);
        dev_queue(0).memcpy(base.data(), as_[0].gdn_state, gdn_per * 4).wait();
        dev_queue(0).memcpy(cv.data(), as_[0].conv_state, conv_per * 4).wait();
        FILE * f = fopen(sp, "wb");
        if (f) {
            fwrite(&gdn_per, 8, 1, f);
            fwrite(&conv_per, 8, 1, f);
            fwrite(base.data(), 4, gdn_per, f);
            fwrite(cv.data(), 4, conv_per, f);
            fclose(f);
            fprintf(stderr, "[mtp] statedump %s (decode_h=%d)\n", sp, (int)decode_h);
        }
    }
    if (decode_h) {
        // the sequential decode already produced the main KV and state; only the
        // MTP's own KV for the prompt is missing
        chunk.assign(prompt.begin(), prompt.end());
        for (int pos = 0; pos < nprompt;) {
            const int n = std::min(kMaxT, nprompt - pos);
            mtp_forward(chunk.data() + pos, d_mtp_main_h + (size_t)pos * hp.n_embd, d_mtp_hprev, n, 0, pos,
                        /*with_head=*/false);
            dev_queue(0)
                .memcpy(d_mtp_hprev, d_mtp_main_h + (size_t)(pos + n - 1) * hp.n_embd, (size_t)hp.n_embd * 4)
                .wait();
            pos += n;
        }
    }
    if (getenv("PF_MTP_SNAPDUMP") != nullptr && nprompt >= mtp_nsnap) {
        // verify-like pass over exactly mtp_nsnap prompt tokens (a longer pass
        // would write past the pc_row_slot entries mtp_verify initialises)
        const int pn = getenv("PF_MTP_SNAP1") ? 1 : mtp_nsnap;
        std::vector<int> pv(prompt.begin(), prompt.begin() + pn);
        mtp_verify(pv, pn, 0, 0);
    }
    if (const char * hd = getenv("PF_MTP_HDUMPS")) {
        // d_mtp_main_h is host USM: dump it directly, before run_head or any
        // other call can touch the activation buffers
        FILE * f = fopen(hd, "wb");
        if (f) {
            int hdr[4] = {nprompt, hp.n_embd, hp.n_vocab, mtp_on ? 1 : 0};
            fwrite(hdr, 4, 4, f);
            fwrite(prompt.data(), 4, prompt.size(), f);
            fwrite(d_mtp_main_h, 4, (size_t)nprompt * hp.n_embd, f);
            fclose(f);
            fprintf(stderr, "[mtp] hdumps %s (nprompt=%d decode_h=%d)\n", hd, nprompt,
                    (int)(getenv("PF_MTP_DECODE_H") != nullptr));
        }
    }
    if (getenv("PF_MTP_DRAFTTEST") != nullptr) {
        // Emulate the first draft row *inside* a multi-token pass: row 22 gets
        // (emb(tok), h_21) exactly like the single-token draft call, but the
        // pass also carries rows 0..21.
        std::vector<int32_t> t23(prompt.begin(), prompt.end());
        t23.push_back(tok);
        std::vector<float> h0((size_t)hp.n_embd, 0.0f);
        float * d_h0 = (float *)dev_alloc_on(0, (size_t)hp.n_embd * 4);
        dev_queue(0).memcpy(d_h0, h0.data(), (size_t)hp.n_embd * 4).wait();
        mtp_forward(t23.data(), d_mtp_main_h, d_h0, (int)t23.size(), 0, 0, /*with_head=*/true);
        sync_all();
        dev_queue(0).memcpy(h_logits, d_logits + (size_t)(t23.size() - 1) * hp.n_vocab, (size_t)hp.n_vocab * 4).wait();
        fprintf(stderr, "[mtp] drafttest in-pass row %zu (tok=%d) -> %d  (single-token draft gives %d)\n",
                t23.size() - 1, tok, argmax_f(h_logits, hp.n_vocab),
                /* reported by the caller */ -1);
        // and the same row as a standalone single-token call
        const int32_t t22 = tok;
        mtp_forward(&t22, nullptr, d_mtp_hprev, 1, 0, nprompt, /*with_head=*/true);
        dev_queue(0).wait();
        dev_queue(0).memcpy(h_logits, d_logits, (size_t)hp.n_vocab * 4).wait();
        fprintf(stderr, "[mtp] drafttest single row %d (tok=%d) -> %d\n", nprompt, t22,
                argmax_f(h_logits, hp.n_vocab));
        sycl::free(d_h0, dev_queue(0));
    }
    if (getenv("PF_MTP_HPROBE") != nullptr) {
        // capt[nprompt-1] must reproduce the first sampled token; capt[i] must
        // predict the prompt token at i+1
        std::vector<int> rows = {0, 1, 2, nprompt - 2, nprompt - 1};
        for (int i : rows) {
            if (i < 0 || i >= nprompt) continue;
            q.memcpy(d_last_hidden, d_mtp_main_h + (size_t)i * hp.n_embd, (size_t)hp.n_embd * 4).wait();
            std::vector<float> lg2 = run_head();
            const int got = argmax_f(lg2.data(), hp.n_vocab);
            const int want = (i + 1 < nprompt) ? prompt[(size_t)i + 1] : tok;
            fprintf(stderr, "[mtp] hprobe capt[%d] -> %d want %d %s\n", i, got, want, got == want ? "OK" : "XX");
        }
    }
    if (dbg_gen) {
        fprintf(stderr, "[mtp] step=0 pos=%d id=%d eos=%d\n", pos, tok, (int)is_eos(tok));
    }
    if ((!gp.ignore_eos && is_eos(tok)) || !cb(tok)) {
        for (int b : blocks) {
            free_block(b);
        }
        return out;
    }

    // Adaptive draft length: with poor acceptance (an aligned/boilerplate
    // continuation, or an ignore_eos run) MTP is a net loss and usually falls
    // back to plain decoding, which is single-sequence and cannot share the
    // scheduler.  Shrinking k by one whenever a whole cycle accepts nothing and
    // growing it back on a full-acceptance cycle keeps the ceiling at mtp_k (so
    // the buffers still cover it) while cutting the wasted verify work.  The
    // emitted stream is unaffected: acceptance ignores k.
    static const bool adapt = [] {
        const char * e = getenv("PF_MTP_ADAPT");
        return e ? atoi(e) != 0 : true;
    }();
    int k = mtp_k;
    std::vector<int> cand((size_t)mtp_k + 1);
    static const bool mt = getenv("PF_MTP_TIME") != nullptr;
    double t_draft = 0, t_verify = 0, t_commit = 0, t_rb = 0, t_emit = 0, t_cb = 0, t_am = 0, t_cyc = 0;
    long t_acc = 0;
    int t_cycles = 0, t_tok = 0;
    auto now_t = [] { return std::chrono::high_resolution_clock::now(); };
    auto ms_t = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    while ((int)out.size() < gp.max_tokens && pos < max_seq - 1) {
        const auto tc0 = now_t();
        // ---- draft: k autoregressive MTP steps -----------------------------
        cand[0] = tok;
        const float * hprev = d_mtp_hprev;
        static const bool no_mtpfwd = getenv("PF_MTP_NOMTPFWD") != nullptr;
        for (int i = 0; !no_mtpfwd && i < k; i++) {
            const int32_t t = cand[(size_t)i];
            MTPDBG("draft %d\n", i);
            mtp_forward(&t, nullptr, hprev, 1, 0, pos + i, /*with_head=*/true);
            // the LM head runs on device 0 while the MTP layer may live on a
            // partition device: `q` is not ordered against dev_queue(0), so a
            // q.wait()/q.memcpy here reads the *previous* step's logits and the
            // draft chain degenerates into a stale-logits echo.  Only the MTP
            // device and device 0 produced anything this step, so synchronizing
            // just those two beats the multi-device sync_all's serial sweep of
            // every backend (this is k of the 9 syncs per cycle).
            if (multi_dev) {
                if (mtp_dev != 0) {
                    dev_queue(mtp_dev).wait();
                }
                dev_queue(0).wait();
            } else {
                backend().synchronize();
            }
            dev_queue(0).memcpy(h_logits, d_logits, (size_t)hp.n_vocab * 4).wait();
            cand[(size_t)i + 1] = argmax_f(h_logits, hp.n_vocab);
            static const bool lstat = getenv("PF_MTP_LSTAT") != nullptr;
            if (lstat) {
                double mx = -1e30, mn = 1e30, sum = 0;
                int nan = 0;
                for (int v = 0; v < hp.n_vocab; v++) {
                    const float x = h_logits[v];
                    if (!std::isfinite(x)) {
                        nan++;
                        continue;
                    }
                    mx = std::max(mx, (double)x);
                    mn = std::min(mn, (double)x);
                    sum += x;
                }
                int t3[3] = {-1, -1, -1};
                float b3[3] = {-1e30f, -1e30f, -1e30f};
                for (int v = 0; v < hp.n_vocab; v++) {
                    for (int q = 0; q < 3; q++) {
                        if (h_logits[v] > b3[q]) {
                            for (int w = 2; w > q; w--) {
                                b3[w] = b3[w - 1];
                                t3[w] = t3[w - 1];
                            }
                            b3[q] = h_logits[v];
                            t3[q] = v;
                            break;
                        }
                    }
                }
                fprintf(stderr, "[mtp] lstat step=%d pos=%d max=%.3f min=%.3f mean=%.4f nan=%d top=%d(%.2f) %d(%.2f) %d(%.2f)\n",
                        i, pos + i, mx, mn, sum / hp.n_vocab, nan, t3[0], b3[0], t3[1], b3[1], t3[2], b3[2]);
            }
            hprev = d_mtp_raw; // the raw MTP hidden at this row seeds the next
        }
        const auto tc1 = now_t();
        t_draft += ms_t(tc0, tc1);
        // ---- verify --------------------------------------------------------
        // PF_MTP_DECCHK: does the verify's own row 0 agree with a plain
        // single-token decode of the same token at the same position from the
        // same (saved) state?  Distinguishes "state is wrong" from "batched
        // verify forward is wrong".
        if (getenv("PF_MTP_DECCHK") != nullptr) {
            const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state;
            const size_t conv_per = (size_t)(hp.conv_k - 1) * hp.qkv_dim();
            std::vector<std::vector<float>> st_save(as_.size()), cv_save(as_.size());
            std::vector<size_t> nst(as_.size(), 0), ncv(as_.size(), 0);
            for (size_t d = 0; d < as_.size(); d++) {
                const size_t ng = (size_t)n_gdn_dev_[d];
                if (ng == 0) {
                    continue;
                }
                nst[d] = ng * kMaxB * gdn_per;
                ncv[d] = ng * kMaxB * conv_per;
                st_save[d].resize(nst[d]);
                cv_save[d].resize(ncv[d]);
                dev_queue((int)d).memcpy(st_save[d].data(), as_[d].gdn_state, nst[d] * 4).wait();
                dev_queue((int)d).memcpy(cv_save[d].data(), as_[d].conv_state, ncv[d] * 4).wait();
            }
            int32_t tt = cand[0], pp = pos, sl = 0;
            decode_batch(&tt, &pp, &sl, 1);
            q.wait();
            dev_queue(0).memcpy(h_logits, d_logits, (size_t)hp.n_vocab * 4).wait();
            const int plain = argmax_f(h_logits, hp.n_vocab);
            // diff the loop's state against the state the plain decode produced
            // (both are "after the token at `pos`", so they must be identical)
            double dg = 0, dc = 0, rg = 0, rc = 0;
            for (size_t d = 0; d < as_.size(); d++) {
                if (nst[d] == 0) {
                    continue;
                }
                std::vector<float> now(nst[d]);
                dev_queue((int)d).memcpy(now.data(), as_[d].gdn_state, nst[d] * 4).wait();
                for (size_t i = 0; i < nst[d]; i++) {
                    dg = std::max(dg, (double)std::fabs(now[i] - st_save[d][i]));
                    rg = std::max(rg, (double)std::fabs(st_save[d][i]));
                }
                std::vector<float> nowc(ncv[d]);
                dev_queue((int)d).memcpy(nowc.data(), as_[d].conv_state, ncv[d] * 4).wait();
                for (size_t i = 0; i < ncv[d]; i++) {
                    dc = std::max(dc, (double)std::fabs(nowc[i] - cv_save[d][i]));
                    rc = std::max(rc, (double)std::fabs(cv_save[d][i]));
                }
            }
            fprintf(stderr, "[mtp] statediff pos=%d gdn=%.3e (|.|=%.2f) conv=%.3e (|.|=%.2f)\n", pos, dg, rg, dc,
                    rc);
            for (size_t d = 0; d < as_.size(); d++) {
                if (nst[d] == 0) {
                    continue;
                }
                dev_queue((int)d).memcpy(as_[d].gdn_state, st_save[d].data(), nst[d] * 4).wait();
                dev_queue((int)d).memcpy(as_[d].conv_state, cv_save[d].data(), ncv[d] * 4).wait();
            }
            fprintf(stderr, "[mtp] decchk pos=%d tok=%d plain-decode=%d\n", pos, tt, plain);
        }
        MTPDBG("verify start\n");
        static const int vn_env = [] {
            const char * e = getenv("PF_MTP_VERIFYN");
            return e ? atoi(e) : 0;
        }();
        const int n_ver = (vn_env > 0 && vn_env < k + 1) ? vn_env : k + 1;
        mtp_verify(cand, n_ver, 0, pos);
        MTPDBG("verify done\n");
        sync_all();
        const auto tc2 = now_t();
        t_verify += ms_t(tc1, tc2);
        // The accept test needs only the per-row argmax, so it runs on the device
        // and copies back k+1 ints instead of (k+1)*n_vocab floats (~7 MB/cycle
        // at k=6, comparable to a whole draft pass).  The bonus token still goes
        // through sample_token, but the MTP loop is greedy-only and a greedy
        // sample_token is exactly the row argmax - so when nothing else can
        // perturb the choice the device indices serve both and the host copy
        // leaves the hot path entirely.
        const bool greedy = (gp.temperature <= 0.f || gp.top_k == 1);
        const bool no_penalty =
            gp.repeat_penalty == 1.0f && gp.presence_penalty == 0.f && gp.frequency_penalty == 0.f;
        const bool have_dev_argmax = greedy && no_penalty && gp.logit_bias.empty() && !mtp_dbg();
        // `dev_argmax` must be device memory: a kernel cannot write a host stack
        // array.
        static int32_t * d_argmax = nullptr;
        if (!d_argmax) {
            d_argmax = sycl::malloc_device<int32_t>(kMaxB, dev_queue(0));
        }
        const auto tam0 = now_t();
        if (have_dev_argmax) {
            backend().mtp_argmax(d_logits, hp.n_vocab, d_argmax, nullptr, k + 1);
            dev_queue(0).memcpy(h_argmax, d_argmax, (size_t)(k + 1) * 4).wait();
            if (getenv("PF_MTP_AMCHK") != nullptr) {
                dev_queue(0).memcpy(h_logits, d_logits, (size_t)(k + 1) * hp.n_vocab * 4).wait();
                for (int i = 0; i <= k; i++) {
                    const int h = argmax_f(h_logits + (size_t)i * hp.n_vocab, hp.n_vocab);
                    if (h != (int)h_argmax[i]) {
                        fprintf(stderr, "[mtp] amchk MISMATCH row=%d dev=%d host=%d\n", i, (int)h_argmax[i], h);
                    }
                }
            }
        } else {
            dev_queue(0).memcpy(h_logits, d_logits, (size_t)(k + 1) * hp.n_vocab * 4).wait();
        }
        t_am += ms_t(tam0, now_t());
        auto emit_cb = [&](int t) -> bool {
            const auto a = now_t();
            const bool r = cb(t);
            t_cb += ms_t(a, now_t());
            return r;
        };
        auto row_argmax = [&](int i) -> int {
            return have_dev_argmax ? (int)h_argmax[i] : argmax_f(h_logits + (size_t)i * hp.n_vocab, hp.n_vocab);
        };
        if (mtp_dbg()) {
            fprintf(stderr, "[mtp] pos=%d drafts:", pos);
            for (int i = 1; i <= k; i++) {
                fprintf(stderr, " %d", cand[(size_t)i]);
            }
            fprintf(stderr, "  target:");
            for (int i = 0; i < k; i++) {
                fprintf(stderr, " %d", row_argmax(i));
            }
            fprintf(stderr, "\n");
        }
        int j = 0;
        static const bool no_accept = getenv("PF_MTP_NOACCEPT") != nullptr;
        while (!no_accept && j < std::min(k, n_ver - 1)) {
            const int t = row_argmax(j);
            if (t != cand[(size_t)j + 1]) {
                break;
            }
            j++;
        }
        // adopt the main model's token at the first mismatch (or past the drafts)
        {
            const float * row = h_logits + (size_t)j * hp.n_vocab;
            int bonus = have_dev_argmax ? (int)h_argmax[j] : sample_token(row, hp.n_vocab, gp, out, ss);
            // emit the accepted drafts then the target's own token
            for (int i = 1; i <= j; i++) {
                if ((int)out.size() >= gp.max_tokens) {
                    for (int b : blocks) {
                        free_block(b);
                    }
                    return out;
                }
                out.push_back(cand[(size_t)i]);
                if (dbg_gen) {
                    fprintf(stderr, "[mtp] accept pos=%d id=%d\n", pos + i, cand[(size_t)i]);
                }
                if ((!gp.ignore_eos && is_eos(cand[(size_t)i])) || !emit_cb(cand[(size_t)i])) {
                    for (int b : blocks) {
                        free_block(b);
                    }
                    return out;
                }
            }
            if ((int)out.size() >= gp.max_tokens) {
                break;
            }
            tok = bonus;
            out.push_back(tok);
            if (dbg_gen) {
                fprintf(stderr, "[mtp] cycle pos=%d accepted=%d bonus=%d\n", pos, j, tok);
            }
            if ((!gp.ignore_eos && is_eos(tok)) || !emit_cb(tok)) {
                break;
            }
        }
        const auto tc3 = now_t();
        t_emit += ms_t(tc2, tc3); // accept + argmax + sampling + the host callback
        // ---- commit --------------------------------------------------------
        // Rebuild the MTP KV for the *committed* tokens: cand[0..j] plus the
        // target's own bonus token.  Rebuilding it from the raw drafts (as this
        // used to) would key every position from the first rejected draft on,
        // and since only the bonus is committed, EVERY later position would be
        // built from a token that was never generated.
        if (!no_mtpfwd) {
            std::vector<int32_t> comm(cand.begin(), cand.begin() + (size_t)j + 1);
            comm.push_back(tok);
            mtp_forward(comm.data(), d_mtp_main_h, d_mtp_hprev, j + 2, 0, pos, /*with_head=*/false);
        }
        // The commit and the rollback are back-to-back and both end in a
        // device-wide barrier, so the commit's own sync is pure latency: the
        // rollback's memcpys are queued behind the commit's kernels on the same
        // in-order queues, and one sync after both waits for everything.  (At
        // ~0.45 ms per sync_all x 2 devices this was ~0.9 ms of the 4.0 ms
        // "commit" phase; the phase split below keeps the commit timer
        // meaningful by charging the shared barrier to the rollback.)
        const auto tc4 = now_t();
        static const bool no_rb = getenv("PF_MTP_NORB") != nullptr;
        // The rollback only *enqueues* memcpys into the per-device in-order
        // queues, and nothing on the host reads the recurrent state, so the
        // device-wide barrier that used to follow it is pure latency: the next
        // device work (the draft, which reads that state) is already ordered
        // after the memcpys on the same queue.  PF_MTP_NORBSYNC=1 keeps the old
        // barrier for A/B.
        static const bool rb_sync = getenv("PF_MTP_NORBSYNC") != nullptr;
        if (!no_rb) {
            mtp_rollback(j);
            if (rb_sync) {
                sync_all();
            }
        } else {
            sync_all();
        }
        const auto tc5 = now_t();
        t_commit += ms_t(tc3, tc4);
        t_rb += ms_t(tc4, tc5);
        t_cyc += ms_t(tc0, now_t()); // whole loop body, from the draft's start
        t_cycles++;
        t_tok += j + 1;
        t_acc += j;
        if (adapt && !no_accept) {
            // a fully accepted cycle means the chain can still run
            if (j == k) {
                k = std::min(k + 1, mtp_k);
            } else if (j <= 1) {
                k = std::max(k - 1, 1);
            }
        }
        if (mt && (t_cycles % 4 == 0)) {
            fprintf(stderr,
                    "[mtp] time: cycles=%d tok=%d acc=%.2f | draft=%.1f verify=%.1f commit=%.1f rb=%.1f "
                    "ms/cycle | %.1f ms/token\n",
                    t_cycles, t_tok, (double)t_acc / t_cycles, t_draft / t_cycles, t_verify / t_cycles,
                    t_commit / t_cycles, t_rb / t_cycles, (t_draft + t_verify + t_commit + t_rb) / t_tok);
            fprintf(stderr, "[mtp] emit: %.1f ms/cycle  cb: %.1f  argmax-blk: %.1f\n", t_emit / t_cycles,
                    t_cb / t_cycles, t_am / t_cycles);
            fprintf(stderr, "[mtp] cycle-wall: %.1f ms (sum %.1f = draft+verify+commit+rb+emit), gap %.1f ms/cycle\n",
                    t_cyc / t_cycles, (t_draft + t_verify + t_commit + t_rb + t_emit) / t_cycles,
                    (t_cyc - (t_draft + t_verify + t_commit + t_rb + t_emit)) / t_cycles);
        }
        static const bool statechk = getenv("PF_MTP_STATECHK") != nullptr;
        if (statechk && !h_save_.empty()) {
            const size_t gdn_per = (size_t)hp.dt_rank * hp.d_state * hp.d_state;
            const size_t conv_per = (size_t)(hp.conv_k - 1) * hp.qkv_dim();
            const size_t tot = gdn_per + conv_per;
            std::vector<float> rb(tot);
            sycl::queue & qd0 = dev_queue(0);
            qd0.memcpy(rb.data(), as_[0].gdn_state, gdn_per * 4).wait();
            qd0.memcpy(rb.data() + gdn_per, as_[0].conv_state, conv_per * 4).wait();
            // reference: restore the pre-verify state and replay the committed
            // tokens one at a time on the validated single-token path
            qd0.memcpy(as_[0].gdn_state, h_save_.data(), gdn_per * 4).wait();
            qd0.memcpy(as_[0].conv_state, h_save_.data() + gdn_per, conv_per * 4).wait();
            for (int c = 0; c <= j; c++) {
                int32_t tt = (c == 0) ? cand[0] : cand[(size_t)c];
                int32_t pp = pos + c;
                int32_t sl = 0;
                decode_batch(&tt, &pp, &sl, 1);
            }
            std::vector<float> ref(tot);
            qd0.memcpy(ref.data(), as_[0].gdn_state, gdn_per * 4).wait();
            qd0.memcpy(ref.data() + gdn_per, as_[0].conv_state, conv_per * 4).wait();
            double d = 0, nb = 0;
            for (size_t i = 0; i < tot; i++) {
                d = std::max(d, (double)std::fabs(rb[i] - ref[i]));
                nb = std::max(nb, (double)std::fabs(ref[i]));
            }
            fprintf(stderr, "[mtp] statechk pos=%d j=%d rollback-vs-ref maxdiff=%.6f |ref|=%.4f\n", pos, j, d, nb);
            h_save_.clear();
        }
        if (getenv("PF_MTP_AUTOTEST") != nullptr) {
            // compare the verify's row-0 prediction against a plain single-token
            // decode from the same (rolled-back) state
            int32_t tt = tok, pp = pos, sl = 0;
            decode_batch(&tt, &pp, &sl, 1);
            q.wait();
            dev_queue(0).memcpy(h_logits, d_logits, (size_t)hp.n_vocab * 4).wait();
            int plain = argmax_f(h_logits, hp.n_vocab);
            fprintf(stderr, "[mtp] autotest pos=%d j=%d bonus=%d plain-decode=%d %s\n", pos, j, tok, plain,
                    plain == tok ? "SAME" : "DIFF");
            mtp_rollback(j);
        }
        // the main hidden at the last committed position seeds the next cycle
        dev_queue(0).memcpy(d_mtp_hprev, d_mtp_main_h + (size_t)j * hp.n_embd, (size_t)hp.n_embd * 4).wait();
        pos += j + 1;
        // grow the block table if the draft window could cross a block boundary
        if ((pos + mtp_k + 1 + kBlockSize - 1) / kBlockSize > (int)blocks.size()) {
            for (int kb = (int)blocks.size(); kb < (pos + mtp_k + 1 + kBlockSize - 1) / kBlockSize; kb++) {
                int b = alloc_block();
                if (b < 0) {
                    break;
                }
                blocks.push_back(b);
            }
            set_table(0, blocks);
        }
        if ((int)out.size() >= gp.max_tokens) {
            break;
        }
    }
    for (int b : blocks) {
        free_block(b);
    }
    return out;
}

} // namespace si
