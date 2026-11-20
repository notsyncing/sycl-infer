// Standalone C ABI test for the research FFN capture API, two layers, and
// their 0-1 similarity.  Build against the C shared library; run with the
// Qwen3.5-0.8B GGUF as argv[1] (CPU backend, no GPU).
//
//   ./build-embed/test_capi_ffn /path/to/Qwen3.5-0.8B-Q4_K_M.gguf
//
// The 24-word batch is large enough for the cross-text z-score used by the
// similarity method to be stable (see research/ffn_pattern.cpp).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <sycl_infer/sycl_infer.h>
#include <sycl_infer/sycl_infer_research.h>

static int fail(const char *what, syi_status st) {
    std::fprintf(stderr, "%s: %s (%s)\n", what, syi_status_string(st), syi_last_error());
    return 1;
}

// top-k index set overlap of two rows (gate half only, first nf floats)
static double raw_overlap(const float *a, const float *b, int nf, int k) {
    std::vector<char> ma(nf, 0), mb(nf, 0);
    for (int t = 0; t < k; ++t) {
        int ba = -1, bb = -1;
        float va = -1e30f, vb = -1e30f;
        for (int i = 0; i < nf; ++i) {
            if (!ma[i] && a[i] > va) { va = a[i]; ba = i; }
            if (!mb[i] && b[i] > vb) { vb = b[i]; bb = i; }
        }
        ma[ba] = 1;
        mb[bb] = 1;
    }
    int c = 0;
    for (int i = 0; i < nf; ++i) c += ma[i] && mb[i];
    return (double)c / k;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    syi_model_options o;
    syi_model_options_init(&o);
    o.model_path = argv[1];
    o.device = SYI_DEVICE_CPU;
    o.max_seq = 128;
    o.max_slots = 1;
    o.n_blocks = 32;
    o.kv_cap_mb = 128;
    o.cpu_threads = 4;
    o.batch_window_us = 0;
    o.prefix_cache = 0;
    o.kv_type = SYI_KV_I8;
    syi_model *m = nullptr;
    syi_status st = syi_model_create(&o, &m);
    if (st != SYI_OK) return fail("create", st);

    syi_model_info info; syi_model_info_init(&info);
    if ((st = syi_model_get_info(m, &info)) != SYI_OK) return fail("info", st);
    std::printf("model: n_layer=%d n_embd=%d\n", info.n_layer, info.n_embd);

    // 0=苹果 1=梨 4=橘子 5=猫 6=狗 18=剪刀 19=白菜 20=石头
    const char *words[] = {
        "苹果", "梨", "香蕉", "葡萄", "橘子",
        "猫", "狗", "老虎", "兔子", "盒子",
        "汽车", "桌子", "椅子", "上", "下",
        "思想", "语言", "数字", "剪刀", "白菜",
        "石头", "茶", "咖啡", "牛奶",
    };
    const int n = (int)(sizeof(words) / sizeof(words[0]));
    const int I_APPLE = 0, I_PEAR = 1, I_CAT = 5, I_DOG = 6, I_SCISSOR = 18, I_CABBAGE = 19, I_STONE = 20;

    syi_ffn_options fo; syi_ffn_options_init(&fo);
    fo.layer = 5; fo.token_slot = -1;

    float *data1 = nullptr, *data2 = nullptr;
    size_t nt1 = 0, nl1 = 0, nf1 = 0, nt2 = 0, nl2 = 0, nf2 = 0;
    if ((st = syi_model_capture_ffn(m, &fo, words, n, &data1, &nt1, &nl1, &nf1)) != SYI_OK)
        return fail("capture_ffn#1", st);
    if ((st = syi_model_capture_ffn(m, &fo, words, n, &data2, &nt2, &nl2, &nf2)) != SYI_OK)
        return fail("capture_ffn#2", st);
    if (nt1 != (size_t)n || nl1 != 1 || nf1 <= 0) {
        std::fprintf(stderr, "bad geometry nt=%zu nl=%zu nf=%zu\n", nt1, nl1, nf1);
        return 1;
    }
    const size_t per = (size_t)nl1 * 2 * nf1;
    if (std::memcmp(data1, data2, nt1 * per * sizeof(float)) != 0) {
        std::fprintf(stderr, "capture is not deterministic\n");
        return 1;
    }
    for (size_t i = 0; i < nt1 * per; ++i) {
        if (!std::isfinite(data1[i])) { std::fprintf(stderr, "non-finite activation at %zu\n", i); return 1; }
    }

    const int K = 128;
    double near_pair = raw_overlap(data1 + I_APPLE * per, data1 + I_PEAR * per, (int)nf1, K);
    double far_pair = raw_overlap(data1 + I_SCISSOR * per, data1 + I_CABBAGE * per, (int)nf1, K);
    std::printf("raw overlap(top%d): 苹果/梨=%.3f  剪刀/白菜=%.3f\n", K, near_pair, far_pair);
    if (!(near_pair > far_pair)) {
        std::fprintf(stderr, "expected near pair to share more than far pair\n");
        return 1;
    }

    // all-layers capture sanity
    syi_ffn_options foa; syi_ffn_options_init(&foa);
    foa.layer = -1; foa.token_slot = 0;
    float *all = nullptr; size_t na = 0, nla = 0, nfa = 0;
    if ((st = syi_model_capture_ffn(m, &foa, words, 1, &all, &na, &nla, &nfa)) != SYI_OK)
        return fail("capture_ffn all-layers", st);
    if (na != 1 || nla != (size_t)info.n_layer || nfa != nf1) {
        std::fprintf(stderr, "bad all-layers geometry nt=%zu nl=%zu nf=%zu\n", na, nla, nfa);
        return 1;
    }
    syi_free(all);

    // ---- two layers: pattern extraction -> 0-1 similarity ----
    syi_ffn_pattern_options po;
    syi_ffn_pattern_options_init(&po);
    po.top_k = 128;
    po.half = 0;
    po.zscore = 1;
    syi_ffn_pattern *pat = nullptr;
    if ((st = syi_ffn_pattern_create(data1, nt1, nl1, nf1, &po, &pat)) != SYI_OK)
        return fail("pattern_create", st);

    int32_t *ids = nullptr;
    float *zvals = nullptr;
    size_t pn_texts = 0, pn_layers = 0, ptopk = 0, pdim = 0;
    if ((st = syi_ffn_pattern_top(pat, &ids, &zvals, &pn_texts, &pn_layers, &ptopk, &pdim)) != SYI_OK)
        return fail("pattern_top", st);
    if (pn_texts != (size_t)n || pn_layers != 1 || ptopk != 128 || pdim != nf1) {
        std::fprintf(stderr, "bad pattern shape nt=%zu nl=%zu k=%zu dim=%zu\n",
                     pn_texts, pn_layers, ptopk, pdim);
        return 1;
    }
    for (size_t i = 0; i < pn_texts * pn_layers * ptopk; ++i) {
        if (ids[i] < 0 || (size_t)ids[i] >= pdim) { std::fprintf(stderr, "top id out of range\n"); return 1; }
        if (!std::isfinite(zvals[i])) { std::fprintf(stderr, "non-finite z\n"); return 1; }
    }
    syi_free(ids);
    syi_free(zvals);

    syi_ffn_similarity_options so;
    syi_ffn_similarity_options_init(&so);
    double *mat = nullptr;
    size_t mn = 0;
    if ((st = syi_ffn_similarity(pat, &so, &mat, &mn)) != SYI_OK)
        return fail("similarity", st);
    if (mn != (size_t)n) { std::fprintf(stderr, "bad similarity n=%zu\n", mn); return 1; }
    if (std::fabs(mat[I_APPLE * n + I_APPLE] - 1.0) > 1e-12) { std::fprintf(stderr, "diag != 1\n"); return 1; }
    for (size_t i = 0; i < (size_t)n; ++i) {
        for (size_t j = 0; j < (size_t)n; ++j) {
            if (mat[i * n + j] < 0.0 || mat[i * n + j] > 1.0) { std::fprintf(stderr, "score out of [0,1]\n"); return 1; }
            if (std::fabs(mat[i * n + j] - mat[j * n + i]) > 1e-12) { std::fprintf(stderr, "not symmetric\n"); return 1; }
        }
    }
    double s_near = mat[I_APPLE * n + I_PEAR];
    double s_far = mat[I_SCISSOR * n + I_CABBAGE];
    std::printf("similarity: 苹果/梨=%.4f  猫/狗=%.4f  剪刀/白菜=%.4f  苹果/石头=%.4f\n",
                s_near, mat[I_CAT * n + I_DOG], s_far, mat[I_APPLE * n + I_STONE]);
    if (!(s_near > s_far)) { std::fprintf(stderr, "expected near score > far score\n"); return 1; }
    syi_free(mat);
    syi_ffn_pattern_destroy(pat);

    syi_free(data1);
    syi_free(data2);
    syi_model_destroy(m);
    std::printf("test_capi_ffn OK (n_ff=%zu, deterministic, pattern+similarity)\n", nf1);
    return 0;
}
