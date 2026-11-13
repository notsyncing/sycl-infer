#pragma once
#include <cstdint>
#include <vector>

#include "engine_types.h"
#include "kernels.h"
#include "spec/verify_scratch.h"

namespace si {

struct mspec_state {
  bool mtp_on = false;
  int mtp_dev = 0; // backend index the MTP layer runs on (its KV follows)
  int mtp_k = 0;      // draft predictions per verify cycle
  int mtp_nsnap = 0;  // per-token recurrent-state snapshots kept (k+1)
  int mtp_attn_local_ = -1; // MTP layer index inside the primary device's pool
  std::vector<float *> d_mtp_hist_;    // [dev] per-token GDN state snapshots
  std::vector<float *> d_mtp_convsave_; // [dev] conv rows (1,2) before the verify
  float * d_mtp_cat = nullptr;       // [R][2*n_embd] concat(enorm(emb), hnorm(h))
  float * d_mtp_x = nullptr;         // [R][n_embd]
  float * d_mtp_xnorm = nullptr;     // [R][n_embd]
  float * d_mtp_qbuf = nullptr;      // [R][n_head*2*head_dim]
  float * d_mtp_kbuf = nullptr;      // [R][n_head_kv*head_dim]
  float * d_mtp_vbuf = nullptr;      // [R][n_head_kv*head_dim]
  float * d_mtp_attn_out = nullptr;  // [R][n_head*head_dim]
  float * d_mtp_ffn = nullptr;       // [R][2*n_ff]
  float * d_mtp_hnorm = nullptr;
  float * d_mtp_raw = nullptr; // MTP hidden before shared_head_norm (AR seed)     // [R][n_embd] shared-head-normalized hidden
  float * d_mtp_main_h = nullptr;    // [kMaxB*kMaxT][n_embd] main hidden capture
  std::vector<float> h_save_;        // PF_MTP_STATECHK: pre-verify recurrent state
  // MTP rollback by recurrence replay: the verify saves each GDN layer's
  // per-token (conv_out, alpha, beta) plus the pre-verify GDN state, and the
  // commit restores the state and replays those tokens.
  std::vector<float *> d_mtp_rin_;
  std::vector<float *> d_mtp_qsave_; // per-layer raw GDN conv taps of the verify rows    // [dev] ng * mtp_nsnap * (conv_dim + 2*dt_rank)
  std::vector<float *> d_mtp_ssave_;  // [dev] ng * gdn_per
  step_info * d_mtp_rinfo = nullptr;  // step_info for the replay launches
  float * d_mtp_hnorm0 = nullptr;    // primary-device copy feeding the shared LM head
  float * d_mtp_hprev = nullptr;     // [kMaxB][n_embd] main hidden before the row
  step_info * d_mtp_info = nullptr;  // host USM step_info for the MTP layer
  seg_plan plan_vf_;                 // verify forward: mode 2 with a batched head
  gemv_seg * d_segs_vf = nullptr;
  seg_plan plan_mtp_;                // the MTP layer's own calls
  gemv_seg * d_segs_mtp = nullptr;
  // which plan call holds the shared LM head.  It is 4 for a dense
  // draft layer and 6 once the NextN block's own MoE takes four calls,
  // so the verify and the readout ask the plan instead of hardcoding it.
  int head_call = 4;
  // PF_MTP_LAYER_EXACT: same segments with `w` pointing at the raw GGUF bytes
  gemv_seg * d_segs_mtp_exact = nullptr;
  // PF_MTP_HEAD_W4: a draft-only u4 copy of the LM head, registered under a
  // private key (the target keeps its exact int8 head, so only the drafts -
  // and hence acceptance, never the emitted stream - can move).
  float * d_mtp_partials = nullptr; // the MTP's own attention partials (see mtp_splits)
  int32_t h_argmax[kMaxB] = {0};    // host mirror of the verify's per-row argmax
  bool mtp_head_w4_ = false;
  // PF_MTP_LAYER_W2: the MTP layer also has a 2-bit copy (0.375 B/weight).
  bool mtp_layer_w2_ = false;
  // 2-bit copy of the draft LM head (0.375 B/weight against the u4 copy's
  // 0.625): the draft reads the whole head once per drafted token, so this is
  // the single largest byte item in the speculative cycle.  The draft only
  // needs its argmax, and its weights are already re-quantized, so the extra
  // step is a trade in acceptance for bytes: 0.375 B/weight is exact enough
  // for an argmax (39.8 % relative L2) but costs ~2 % acceptance, which
  // cancels the saving.  Off with PF_MTP_HEAD_W2=0.
  bool mtp_head_w2_ = false;
  char mtp_head_w2_key_[1] = {0};
  bool mtp_layer_w4_ = false; // the MTP layer's own linears are GEMV-only u4 too
  char mtp_head_w4_key_[1] = {0};
  // The MTP draft is a single-token decode over the MTP layer's KV, so it gets
  // the decode's key-parallel split count - *not* n_splits, which the
  // --layer-map path pins to 1 (its split path is opt-in) and which left a
  // 1-row attention with n_head workgroups each looping the whole KV: 2451 vs
  // 183 ms/cycle at 128k.  PF_MTP_SPLITS overrides (1 = the old behaviour).
  int mtp_splits = kMaxDecSplits;
  // The speculative draft's LM head: the MTP/NextN layer's shared head when
  // the GGUF ships one, else the model's output projection.
  // ---- MTP draft head (see mtp_argmax.cpp) -------------------------------
  // d_mtp_tok[i] is the token draft step i chose (the head's argmax, computed
  // on the device), read back by the next step's concat, so the chain needs no
  // host round-trip.  With PF_MTP_CAND the head is evaluated on a candidate set
  // (mtp_cand_launch / mtp_gather_launch) instead of all 248320 rows - measured
  // a net loss, so it is off by default (25/17/7 % hit rate at margin 8).
  int32_t * d_mtp_cand_ = nullptr;  // [mtp_cand_cap] candidate token ids (device 0)
  float * d_mtp_cvals_ = nullptr;   // [mtp_cand_cap] their head values
  int32_t * d_mtp_tok_ = nullptr;   // [kMaxT] per-step draft token ids (device 0, next to the head)
  // Head readout split across both cards (PF_MTP_HEAD_SPLIT=1, **default off**).
  // out[n] = w[n].h is independent per output row, so the draft's 795 MB u4
  // stream (3.0 ms of device time, 75 % of a draft step) can be halved by
  // giving the second device the upper row range while it is otherwise idle:
  // each side runs the identical GEMV over its own rows into the *same*
  // device-0 logits row, so the argmax stays one scan on the owning card.
  // Draft 17.2 -> 13.1 ms/cycle, and end to end a wash: halving N changes the
  // u4 GEMV's decomposition, so the draft's argmax flips on a near-tie now and
  // then (p0 -6 %, p2 -8 % acceptance).
  bool mtp_head_split_ = false;
  int mtp_head_half_ = 0;      // rows [0, half) on device 0, [half, n_vocab) on device 1
  char mtp_head_w4lo_key_[1] = {0};
  char mtp_head_w4b_key_[1] = {0};
  float * d_mtp_hnorm1 = nullptr;   // device-1 copy of the draft hidden
  float * h_head_stage = nullptr;   // host staging for the 20 KB activation copy
  float * d_mtp_cval1_ = nullptr;   // scalar: the chosen candidate's logit (PF_MTP_CANDV)
  float * d_mtp_amv_ = nullptr;     // scalar: the seed distribution's argmax value
  verify_scratch verify;             // the verify cycle's argmax staging
  int32_t * d_am2_ = nullptr;        // PF_MTP_CANDDBG diagnostic staging
  int mtp_cand_src_ = 1;            // 1 = seed from the draft's own step-0 readout
  int mtp_cand_cap_ = 0;            // 0 = off (the full head readout)
  float mtp_cand_margin_ = 8.0f;
  bool mtp_cand_ok_ = false;        // the head's int8 view exists on device 0
  const int8_t * mtp_head_w8_ = nullptr;
  const uint16_t * mtp_head_wsc_ = nullptr;
  int mtp_head_rows_ = 0;

};

} // namespace si
