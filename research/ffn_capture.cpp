// FFN activation capture implementation, moved out of the inference engine.
// It uses only the engine's public fields/methods (m.hp, reset_single,
// alloc_block, set_table, prefill_chunk, free_block, reset_state, mtx, and the
// four documented capture hook fields).  The actual capture happens in the
// layer loop in src/engine/graph.cpp - the only hook that must stay in the
// main flow; see the comment there.
#include "ffn_capture.h"
#include "engine.h"

#include <mutex>

namespace si {
namespace research {

int capture_ffn_activations(engine & e, const std::vector<std::vector<int> > & prompts, int layer, int slot,
                            std::vector<float> & out) {
    const hparams & hp = e.m.hp;
    if (prompts.empty()) {
        out.clear();
        return 0;
    }
    if (layer < -1 || layer >= hp.n_layer) {
        return -1;
    }
    for (const auto & toks : prompts) {
        if (toks.empty()) {
            return -1;
        }
        const int n = (int)toks.size();
        if (n > kMaxT) {
            return -1; // capture intentionally covers a single prefill chunk
        }
        if (slot < 0 || slot >= n) {
            return -1;
        }
    }
    const int n_layers = layer < 0 ? hp.n_layer : 1;
    const size_t per = (size_t)2 * hp.n_ff;
    out.assign(prompts.size() * n_layers * per, 0.0f);

    std::lock_guard<std::mutex> lk(e.mtx);
    kv_scope kv_scope_(e.kv_k_, e.kv_v_);
    // Reuse slot 0 as scratch: no request is in flight while the C API holds
    // the handle (callers must not submit concurrently).
    for (size_t p = 0; p < prompts.size(); ++p) {
        const auto & toks = prompts[p];
        e.reset_single();
        const int need = ((int)toks.size() + kBlockSize - 1) / kBlockSize;
        std::vector<int> blocks;
        blocks.reserve(need);
        for (int i = 0; i < need; ++i) {
            int b = e.alloc_block();
            if (b < 0) {
                return -1;
            }
            blocks.push_back(b);
        }
        e.set_table(0, blocks);
        e.ffn_cap_out = out.data() + p * n_layers * per;
        e.ffn_cap_layer = layer;
        e.ffn_cap_slot = slot;
        e.capture_mode = true;
        try {
            e.prefill_chunk(toks, 0, (int)toks.size(), 0, /*with_head=*/false);
        } catch (...) {
            e.capture_mode = false;
            e.ffn_cap_out = nullptr;
            for (int b : blocks) {
                e.free_block(b);
            }
            throw;
        }
        e.capture_mode = false;
        e.ffn_cap_out = nullptr;
        for (int b : blocks) {
            e.free_block(b);
        }
    }
    e.reset_state();
    return 0;
}

} // namespace research
} // namespace si
