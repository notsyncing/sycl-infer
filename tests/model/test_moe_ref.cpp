// CPU-reference per-layer fingerprints for a model, for comparing against the
// engine's PF_DUMP_LAYERS output.  No engine, no GPU, no CUDA: it runs
// tests/common/cpu_ref's forward pass over the given token ids and prints one
// line per layer (sum / sumsq / max of the layer's output hidden state), which
// is the same triple PF_DUMP_LAYERS prints.
//
// That makes "which layer does the engine first diverge from the reference" a
// text diff instead of a debugging session: run this, run the engine with
// PF_DUMP_LAYERS=1, and compare the two columns.
//
// Usage: test_moe_ref <model.gguf> [token ids...]
//        TEST_TOKENS="1,2,3"   same, comma-separated
//        TEST_MAX_LAYER=N       stop the reference after layer N
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cpu_ref.h"
#include "env.h"
#include "gguf.h"
#include "model.h"

using namespace si;

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: test_moe_ref <model.gguf> [tokens...]\n");
        return 2;
    }
    model m;
    m.load(argv[1]);

    std::vector<int> toks;
    if (const char * t = getenv("TEST_TOKENS")) {
        std::string s(t);
        size_t p = 0;
        while (p < s.size()) {
            size_t c = s.find(',', p);
            if (c == std::string::npos) c = s.size();
            toks.push_back(atoi(s.substr(p, c - p).c_str()));
            p = c + 1;
        }
    } else {
        for (int i = 2; i < argc; i++) {
            toks.push_back(atoi(argv[i]));
        }
    }
    if (toks.empty()) {
        toks = {91, 108, 1939, 3838, 374, 279, 6722, 315, 12081, 97, 248045};
    }

    printf("[ref] %s: n_layer=%d n_embd=%d n_head=%d/%d head_dim=%d n_vocab=%d\n", argv[1], m.hp.n_layer,
           m.hp.n_embd, m.hp.n_head, m.hp.n_head_kv, m.hp.head_dim, m.hp.n_vocab);
    if (m.hp.is_moe()) {
        printf("[ref] MoE: %d experts, top-%d, expert_ff=%d, shared_ff=%d, w_scale=%.4g\n", m.hp.n_expert,
               m.hp.n_expert_used, m.hp.n_ff_exp, m.hp.n_ff_shexp, (double)m.hp.expert_weights_scale);
    }

    cpu_ref ref(m, 512);
    ref.abl_nomoe = getenv("TEST_ABL_NOMOE") != nullptr;
    ref.abl_noshexp = getenv("TEST_ABL_NOSHEXP") != nullptr;
    ref.record = getenv("TEST_NOREC") == nullptr;
    ref.dbg_attn = true;
    ref.forward(toks);

    // the named mid-layer snapshots first: they are what localises the
    // divergence to attention/GDN vs FFN at layer 0.  TEST_SNAPS selects the
    // list (comma-separated) so the same run can be diffed against llama.cpp's
    // cb() names for the same stages without a rebuild; the default keeps the
    // short list this test started with.
    std::vector<std::string> want_names;
    if (const char * sel = getenv("TEST_SNAPS")) {
        std::string s(sel);
        size_t p = 0;
        while (p < s.size()) {
            size_t c = s.find(',', p);
            if (c == std::string::npos) {
                c = s.size();
            }
            if (c > p) {
                want_names.push_back(s.substr(p, c - p));
            }
            p = c + 1;
        }
    } else {
        want_names = {"model.input_embed", "attn_post_norm-0", "ffn_out-0"};
    }
    for (const std::string & want : want_names) {
        // accumulate over every token, so the numbers are directly comparable
        // with the engine's batch-wide PF_MOE_REFDUMP / PF_DUMP_LAYERS output
        double sum = 0, sumsq = 0, mx = 0;
        size_t n = 0;
        for (const auto & s : ref.snaps) {
            if (s.name != want) {
                continue;
            }
            for (float v : s.data) {
                sum += v;
                sumsq += (double)v * v;
                mx = std::max(mx, std::fabs((double)v));
            }
            n += s.data.size();
        }
        if (n) {
            printf("[ref] %-24s sum=%.6f sumsq=%.6f max=%.6f n=%zu\n", want.c_str(), sum, sumsq, mx, n);
        }
    }

    // the router's chosen experts, printed as ints: they are snapshotted
    // through cpu_ref's float-only snap(), so the int32 bits have to be read
    // back rather than summed (as floats the denormals sum to zero)
    const std::vector<float> * last_ids = nullptr;
    for (const auto & s2 : ref.snaps) {
        if (s2.name == "moe_ids-0") {
            last_ids = &s2.data;
        }
    }
    if (last_ids) {
        {
            const std::vector<float> & s2 = *last_ids;
            const int k = (int) m.hp.n_expert_used;
            const size_t off = (s2.size() >= (size_t) k) ? s2.size() - (size_t) k : 0;
            printf("[ref] moe_ids-0 last:");
            for (int j = 0; j < k; j++) {
                int32_t v;
                std::memcpy(&v, &s2[off + (size_t) j], 4);
                printf(" %d", v);
            }
            printf("\n");
        }
    }

    // per-token rows of a snapshot: TEST_ROWS=1 prints token 0 and the last one,
    // because a fold over all tokens hides a single broken row behind 17 good
    // ones (the routed MoE output was 7x the reference while every per-token
    // diagnostic looked healthy)
    if (env::flag("TEST_TOKROW")) {
        for (const char * nm : {"attn_residual-0", "linear_attn_out-0"}) {
            int r = 0;
            for (const auto & s2 : ref.snaps) {
                if (s2.name != nm) {
                    continue;
                }
                const int tok = r++;
                const int nt = (int) toks.size();
                if (tok != 0 && tok != nt - 1) {
                    continue;
                }
                double sum = 0, sumsq = 0;
                for (float v : s2.data) {
                    sum += v;
                    sumsq += (double) v * v;
                }
                printf("[ref] %-18s tok=%2d sum=%.6f sumsq=%.6f\n", nm, tok, sum, sumsq);
            }
        }
    }

    const int max_layer = getenv("TEST_MAX_LAYER") ? atoi(getenv("TEST_MAX_LAYER")) : m.hp.n_layer;
    // "l_out" is snapshotted once per (token, layer).  ref.snaps is appended
    // token-major (the reference loops tokens outside layers), so the layer of
    // the r-th l_out snapshot is r % n_layer - grouping *consecutive* snapshots
    // instead folds n_layer different layers of one token and prints a number
    // that matches neither a layer nor a token.  Fold by that index so the
    // numbers line up with llama.cpp's per-layer residual dump.
    {
        const int nlayer = m.hp.n_layer;
        std::vector<double> sum((size_t) nlayer, 0.0), sumsq((size_t) nlayer, 0.0), mx((size_t) nlayer, 0.0);
        int r = 0;
        for (const auto & s : ref.snaps) {
            if (s.name != "l_out") {
                continue;
            }
            const int layer = r % nlayer;
            r++;
            if (layer < 0 || layer >= max_layer) {
                continue;
            }
            for (float v : s.data) {
                sum[layer] += v;
                sumsq[layer] += (double)v * v;
                mx[layer] = std::max(mx[layer], std::fabs((double)v));
            }
        }
        for (int il = 0; il < max_layer && il < nlayer; il++) {
            printf("[ref] layer il=%2d sum=%.6f sumsq=%.6f max=%.6f\n", il, sum[il], sumsq[il], mx[il]);
        }
    }

    // Per-*row* fingerprints, matching the engine's PF_DUMP_LAYERS lines: `layer`
    // is the first token's row and `layerlast` the last real token's slot, so
    // folding a layer over every row here would not compare against anything.
    if (env::flag("TEST_ROWS")) {
        const int nlayer = m.hp.n_layer;
        const int ntok = (int)toks.size();
        const int rows[2] = {0, ntok - 1};
        std::vector<std::vector<double>> rsum(2, std::vector<double>((size_t) nlayer, 0.0)),
            rsq(2, std::vector<double>((size_t) nlayer, 0.0)), rmx(2, std::vector<double>((size_t) nlayer, 0.0));
        int k = 0;
        for (const auto & s : ref.snaps) {
            if (s.name != "l_out") {
                continue;
            }
            const int layer = k % nlayer;
            const int tok = k / nlayer;
            k++;
            if (layer < 0 || layer >= max_layer) {
                continue;
            }
            for (int j = 0; j < 2; j++) {
                if (tok != rows[j]) {
                    continue;
                }
                for (float v : s.data) {
                    rsum[j][layer] += v;
                    rsq[j][layer] += (double)v * v;
                    rmx[j][layer] = std::max(rmx[j][layer], std::fabs((double)v));
                }
            }
        }
        // Keyed by TOKEN, not by the layer: the engine's PF_DUMP_LAYERS prints
        // `layer` = token 0's row and `layerlast` = the last real token's slot,
        // for every layer.  Folding a diagonal (layer i of token i) produces a
        // comparison that looks like a divergence at layer 0 and is not one.
        for (int j = 0; j < 2; j++) {
            for (int il = 0; il < max_layer && il < nlayer; il++) {
                printf("[ref] tok%-4s il=%2d sum=%.6f sumsq=%.6f max=%.6f\n", rows[j] == 0 ? "0" : "last", il,
                       rsum[j][il], rsq[j][il], rmx[j][il]);
            }
        }
    }

    return 0;
}