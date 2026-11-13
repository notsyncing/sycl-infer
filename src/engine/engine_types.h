#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "kernels.h"

namespace si {

// plan of GEMV calls for one full forward pass (must match record order)
struct seg_plan {
    std::vector<gemv_seg> segs;
    struct group_t {
        uint32_t type;
        int off;
        int n;
        int rows;
    };
    std::vector<group_t> groups;
    // first call index of each layer, plus a final entry for the LM head call.
    // The call metadata is plan-global and indexed by call order, so a partial
    // (per-device) replay must start its call cursor at the phase's first layer
    // instead of at 0.
    std::vector<int> layer_c0;
    // DP4A path: activation quantization needed before one call.  An empty `x`
    // means the call's activations are already fp32.
    struct xq_t {
        const float * x = nullptr;
        const float * up = nullptr;
        int x_stride = 0;
        int up_stride = 0;
        int K = 0;
    };
    // Everything record_forward needs about one call, in the order it must
    // replay them.
    //
    // These were eight parallel std::vectors indexed by call - call_offsets,
    // call_counts, call_total_rows, call_tb, call_nsb, call_xq,
    // call_group_begin, call_group_count - which made "call 3 exists but has no
    // xq entry" a representable state.  That state is what validate() spent
    // four size comparisons checking for, and the comment on begin_call ("pushes
    // all six per-call vectors together") was an admission that nothing else
    // held them together.  One array of records makes it unrepresentable, and a
    // bounds check on `calls` now covers every field at once instead of only the
    // one array a given site happened to read.
    struct call_t {
        int seg_off = 0;     // first index into `segs`
        int seg_count = 0;   // segments in this call
        int total_rows = 0;  // sum of this call's segment n_rows
        int tb = 0;          // token-block width (the batched GEMM's row origin)
        int nsb = 0;         // segments before the batch base
        int group_begin = 0; // first index into `groups`
        int group_count = 0; // groups in this call
        // What the call *is*.  A dense call replays its segments; a MoE call
        // has none and dispatches the MoE kernels instead, so the router's
        // output - which no host code has seen at record time - still lives in
        // the same recorded command graph.
        enum kind_t { dense = 0, moe_routed = 1, moe_epilogue = 2 };
        kind_t kind = dense;
        xq_t xq;
    };
    std::vector<call_t> calls;
    // false for the non-final prefill variant: record_forward must not replay a
    // "head" call then (there is none, and the last FFN call must not repeat)
    bool has_head = true;
    void set_xq(const float * x, const float * up, int xs, int us, int K) {
        calls.back().xq = {x, up, xs, us, K};
    }
    void begin_call(int tb, int nsb) {
        call_t c;
        c.seg_off = (int)segs.size();
        c.tb = tb;
        c.nsb = nsb;
        calls.push_back(c);
    }
    // open a MoE call: it advances the call cursor exactly like begin_call (so
    // the replay stays a straight walk over `calls`) but carries no segments.
    call_t & begin_moe_call(call_t::kind_t kind, int tb, int nsb) {
        begin_call(tb, nsb);
        calls.back().kind = kind;
        return calls.back();
    }
    void add(const gemv_seg & s) {
        segs.push_back(s);
        calls.back().seg_count++;
        calls.back().total_rows += s.n_rows;
    }
    // ffn_down's activation is silu(gate)*up.  The quantizer path applies that
    // from calls.back().xq.up; every other consumer of these segments (the fp32
    // GEMV and the oneDNN fallback) reads the raw gate from seg.x and must apply
    // it itself, which it does when act_up is set.  Patch the segments added by
    // the current call, preserving any per-slice x offset.
    void set_act_up(const float * up_base, const float * x_base) {
        for (size_t i = (size_t)calls.back().seg_off; i < segs.size(); i++) {
            segs[i].act_up = up_base + (segs[i].x - x_base);
        }
    }
    void finalize() {
        for (call_t & c : calls) {
            const int off = c.seg_off;
            const int n = c.seg_count;
            std::stable_sort(segs.begin() + off, segs.begin() + off + n,
                             [](const gemv_seg & a, const gemv_seg & b) {
                                 return (a.dev < b.dev) || (a.dev == b.dev && a.type < b.type);
                             });
            c.group_begin = (int)groups.size();
            int i = 0;
            while (i < n) {
                const uint32_t t = segs[off + i].type;
                int j = i, rows = 0;
                while (j < n && segs[off + j].type == t) {
                    rows += segs[off + j].n_rows;
                    j++;
                }
                groups.push_back({t, off + i, j - i, rows});
                i = j;
            }
            c.group_count = (int)groups.size() - c.group_begin;
        }
        // grouping continues on `type` only after a (dev,type) sort, so a group
        // spanning two devices would silently merge two backends' segments -
        // fail fast here instead of miswiring a future cross-device call.
        // n_layer < 0 skips the layer-count checks (plan_mtp_ has no layer_c0).
        validate(-1, "finalize");
    }
    // Structural self-check, defined in graph.cpp: 4 calls per layer, the
    // head call (if any) pinned to device 0, per-call segment and group ranges
    // inside their arrays, and device-homogeneous groups.  Throws naming the
    // first violation.  Pure integer compares; no behavior change when passing.
    // (It used to also check that the per-call vectors were the same length;
    // that is now the type's invariant rather than a runtime property.)
    void validate(int n_layer, const char * what) const;
};

} // namespace si
