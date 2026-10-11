#pragma once
// ---------------------------------------------------------------------------
// Shared harness for the per-kernel "stage" tests (tests/kernels/*_stage.cpp).
//
// Loads the model once, runs the CPU reference forward and exposes its
// snapshots, plus the info/comparison helpers the original monolithic
// test_gpu_stages.cpp carried.
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "cpu_ref.h"
#include "engine.h"

namespace si {

inline std::vector<int> stage_default_tokens() {
    return {248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198};
}

inline const char * stage_arg_model(int argc, char ** argv) {
    return argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
}

inline std::vector<int> stage_arg_tokens(int argc, char ** argv) {
    std::vector<int> t;
    for (int i = 2; i < argc; i++) {
        t.push_back(atoi(argv[i]));
    }
    return t.empty() ? stage_default_tokens() : t;
}

// The stage harness only ever needs a model and a context; everything else takes
// its default.  A named config reads better here than the two-argument call it
// replaced, and it keeps working if engine_config grows a field.
inline engine_config make_stage_config(const char * model_path, int max_seq) {
    engine_config ec;
    ec.model_path = model_path;
    ec.max_seq = max_seq;
    // TEST_LAYER_MAP: a model that needs a `--layer-map` split (the 35B A3B MoE
    // does not fit on one card) must still be stage-testable, so the split is
    // taken from the environment here rather than from a flag the harness has no
    // way to pass.  Without it the engine's device count is 1 and every expert
    // tensor is uploaded whole.
    if (const char * lm = getenv("TEST_LAYER_MAP")) {
        ec.layer_map = lm;
    }
    if (const char * mb = getenv("TEST_MAX_SLOTS")) {
        ec.max_slots = atoi(mb);
    }
    // TEST_MOE_RESIDENT: run with a partial expert bank so the ncmoe host half
    // is on the path this test is comparing (decode against re-prefill).
    if (const char * mr = getenv("TEST_MOE_RESIDENT")) {
        ec.moe_resident = atoi(mr);
    }
    return ec;
}

struct stage_env {
    engine e;
    cpu_ref ref;
    std::vector<int> toks;
    int T = 0;
    int fails = 0;
    int stage_dev_ = 0;   // device the activation pointers and queue are bound to
    std::map<std::string, std::vector<float>> snaps;

    stage_env(const char * model_path, const std::vector<int> & tokens, int max_seq = 512)
        : e(make_stage_config(model_path, max_seq)), ref(e.m, max_seq), toks(tokens), T((int)tokens.size()) {
        // Multi-device: build_plan's layer loop leaves the activation pointers
        // bound to the *last* layer's device, so a stage test that reuses them
        // (d_qkv, d_conv_state, d_gdn_state, ...) writes one device's scratch
        // through another device's queue -- which is silent corruption, not a
        // failure.  Every stage here drives layer 0, so bind its device and take
        // its queue with it.  Cost of finding out the hard way: three stages
        // reporting engine values of exactly 0.0 against a healthy reference.
        if (e.multi_dev) {
            int d = e.layer_dev_.empty() ? 0 : e.layer_dev_[0];
            stage_dev_ = d;
            e.bind_acts(d);
            e.q = e.dev_queue(d);
            std::vector<float> t4{1.f, 2.f, 3.f, 4.f}, o4(4);
            e.q.memcpy(e.d_x, t4.data(), 16).wait();
            e.q.memcpy(o4.data(), e.d_x, 16).wait();
            std::printf("[stage] dev %d d_x=%p roundtrip %.1f %.1f %.1f %.1f\n", d, (void *)e.d_x, o4[0], o4[1],
                        o4[2], o4[3]);
        }
        ref.dbg_attn = true;
        ref.record = true;
        ref.forward(toks);
        for (auto & s : ref.snaps) {
            auto & v = snaps[s.name];
            v.insert(v.end(), s.data.begin(), s.data.end());
        }
    }

    // Device pointer of a weight, as uploaded on the device this harness is
    // bound to.  model::dev_f32/dev_ptr cannot be used by a stage test: in
    // multi-device mode there is no whole-blob upload (dev_weights is null), so
    // dev_ptr returns the *host* mmap pointer and the kernel dereferences host
    // memory on the device.  That reads as 0 rather than faulting, so every
    // weighted stage came back as max|diff| == max|exp| - i.e. "the engine
    // computed nothing" - which is indistinguishable from a real numeric
    // mismatch.  wptr throws if the tensor is not in this partition, so a layer
    // map that puts a stage's layer on another device fails loudly.
    const void * wptr(const void * host) const {
        return e.wptr(stage_dev_, host);
    }
    const float * wf32(const float * p) const {
        return (const float *)wptr(p);
    }
    // K/V pool base for global attention layer `a`, resolved the way the
    // engine's own path resolves it: per layer, and per *device* in multi-device
    // mode (each partition owns the KV of the attention layers it computes).
    // The d_kpool/d_vpool members are the primary device's only, so a stage test
    // that wrote through them stored into a pool the device it was running on
    // never reads - while kv_read_vec, which does resolve per layer, then
    // compared a written-elsewhere pool against the reference.
    void kvp(int a, void *& kp, void *& vp, const void *& ksc, const void *& vsc) const {
        const char * a1 = nullptr, * a2 = nullptr, * a3 = nullptr, * a4 = nullptr;
        e.kv_layer_ptrs(a, a1, a2, a3, a4);
        kp = const_cast<void *>(static_cast<const void *>(a1));
        vp = const_cast<void *>(static_cast<const void *>(a2));
        ksc = a3;
        vsc = a4;
    }

    // reference snapshot (concatenation of all recorded tokens)
    std::vector<float> get(const std::string & name) const {
        auto it = snaps.find(name);
        if (it == snaps.end()) {
            printf("missing snapshot %s\n", name.c_str());
            return {};
        }
        return it->second;
    }

    // decode-style step_info for one row: n_real tokens from position pos0
    void set_info(int n_real, int pos0 = 0, int slot = 1) {
        e.d_info->n_rows = 1;
        e.d_info->tpb = kMaxT;
        e.d_info->n_real = n_real;
        e.d_info->pos[0] = pos0;
        e.d_info->slot[0] = slot;
        e.d_info->active[0] = 1;
        for (int i = 0; i < n_real && i < (int)toks.size(); i++) {
            e.d_info->tokens[i] = toks[i];
        }
    }

    void cmp(const char * name, const std::vector<float> & got, const std::vector<float> & expect, double tol = 2e-3) {
        if (got.size() != expect.size()) {
            printf("%-24s SIZE mismatch %zu vs %zu FAIL\n", name, got.size(), expect.size());
            fails++;
            return;
        }
        double maxd = 0, maxv = 0;
        for (size_t i = 0; i < got.size(); i++) {
            maxd = std::max(maxd, (double)std::fabs(got[i] - expect[i]));
            maxv = std::max(maxv, (double)std::fabs(expect[i]));
        }
        const bool ok = maxd <= tol * std::max(1.0, maxv);
        printf("%-24s n=%-7zu max|exp|=%.4f max|diff|=%.6f %s\n", name, got.size(), maxv, maxd, ok ? "OK" : "FAIL");
        if (!ok) {
            fails++;
        }
    }

    int finish(const char * what) const {
        printf(fails ? "%s: FAILURES: %d\n" : "%s: all stages OK\n", what, fails);
        return fails ? 1 : 0;
    }
};

} // namespace si
