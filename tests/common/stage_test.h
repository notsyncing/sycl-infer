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

struct stage_env {
    engine e;
    cpu_ref ref;
    std::vector<int> toks;
    int T = 0;
    int fails = 0;
    std::map<std::string, std::vector<float>> snaps;

    stage_env(const char * model_path, const std::vector<int> & tokens, int max_seq = 512)
        : e(model_path, max_seq), ref(e.m, max_seq), toks(tokens), T((int)tokens.size()) {
        ref.dbg_attn = true;
        ref.record = true;
        ref.forward(toks);
        for (auto & s : ref.snaps) {
            auto & v = snaps[s.name];
            v.insert(v.end(), s.data.begin(), s.data.end());
        }
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
