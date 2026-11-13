#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dflash.h"
#include "engine_types.h"
#include "kernels.h"

namespace si {

struct dflash_state {
  bool dflash_on_ = false;
  std::string dflash_path_;
  std::unique_ptr<dflash_model> dfm_;
  int df_dev_ = 0;      // device the draft runs on (must be 0: the shared head)
  int df_kmax_ = 0;     // requested draft tokens per cycle (block_size-1 max)
  int df_k_ = 0;        // the clamped draft length actually used
  int df_block_ = 0;    // dflash.block_size
  int df_swa_ = 0;      // the draft's sliding window
  int df_ring_ = 0;     // K/V ring tokens per layer
  int df_kv_bytes_ = 2; // 2 = f16 ring, 4 = f32
  int df_splits_ = 8;   // draft attention key splits
  int df_cap_rows_ = 0;
  float * d_df_h = nullptr; // [block][n_embd] residual stream
  float * d_df_b = nullptr; // scratch [block][n_embd]
  float * d_df_c = nullptr; // scratch [block][n_embd]
  float * d_df_qkv = nullptr; // [block][(n_head + 2*n_head_kv)*head_dim]
  float * d_df_gu = nullptr; // [block][2*n_ff]
  float * d_df_dyn = nullptr; // [block][2*conv_k*n_groups]
  float * d_df_gate = nullptr; // [block][selector rank]
  float * d_df_partials = nullptr;
  float * d_df_feat = nullptr; // HOST USM [kMaxB*kMaxT][n_tgt_layer*n_embd]
  float * d_df_feat_dev = nullptr; // device staging for one injection chunk
  void * d_df_kring = nullptr; // [n_layer][ring][n_head_kv][head_dim]
  void * d_df_vring = nullptr;
  int32_t * d_df_pos = nullptr; // device: absolute position per row
  std::vector<int32_t> h_df_pos; // host mirror of d_df_pos
  int32_t * d_df_ids = nullptr; // [block][top_k]
  float * d_df_vals = nullptr;
  int32_t * d_df_pids = nullptr; // [block][topk slices][top_k] partials
  float * d_df_pvals = nullptr;
  int df_pcap_ = 0;
  int df_slices_ = 1;
  float * d_df_lattice = nullptr; // [block][top_k + top_k*top_k]
  std::vector<float> h_df_lattice;
  step_info * d_df_info = nullptr;
  const void * df_head_key_ = nullptr; // the store key for the shared LM head
  bool df_head_w4_ = false;
  char df_head_key_store_[1] = {0};
  const uint8_t * df_sel_pv_ = nullptr; // selector codebooks (u4 planes)
  const uint16_t * df_sel_ps_ = nullptr, * df_sel_po_ = nullptr;
  const uint8_t * df_sel_nv_ = nullptr;
  const uint16_t * df_sel_ns_ = nullptr, * df_sel_no_ = nullptr;
};

} // namespace si
