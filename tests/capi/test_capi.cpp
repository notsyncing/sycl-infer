// Hermetic API-shape smoke test; the model is passed as argv[1].
//
//   ./build-rel2/test_capi /path/to/Qwen3.5-0.8B-Q4_K_M.gguf
//
// It runs on the CPU backend so it needs no GPU and no layer map.  The model
// is small enough for a normal CI box; the test deliberately uses one slot and
// a 128-token context to keep its footprint low.
#include <cstdio>
#include <cstring>
#include <string>
#include <sycl_infer/sycl_infer.h>

static int fail(const char *what, syi_status st) {
    std::fprintf(stderr, "%s: %s (%s)\n", what, syi_status_string(st), syi_last_error());
    return 1;
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
    o.n_blocks = 16;
    o.kv_cap_mb = 64;
    o.cpu_threads = 2;
    o.batch_window_us = 0;
    o.prefix_cache = 0;
    o.kv_type = SYI_KV_I8;
    syi_model *m = nullptr;
    syi_status st = syi_model_create(&o, &m);
    if (st != SYI_OK) {
        return fail("create", st);
    }
    syi_model_info info;
    syi_model_info_init(&info);
    if ((st = syi_model_get_info(m, &info)) != SYI_OK) {
        return fail("info", st);
    }
    if (info.n_vocab <= 0 || info.max_seq != 128) {
        std::fprintf(stderr, "bad info: vocab=%d max_seq=%d\n", info.n_vocab, info.max_seq);
        return 1;
    }
    int32_t *ids = nullptr;
    size_t n = 0;
    if ((st = syi_model_tokenize(m, "Hello", 5, 1, &ids, &n)) != SYI_OK) {
        return fail("tokenize", st);
    }
    if (!n || !ids) {
        std::fprintf(stderr, "tokenize returned nothing\n");
        return 1;
    }
    syi_free(ids);
    char *txt = nullptr;
    size_t tn = 0;
    if ((st = syi_model_detokenize(m, nullptr, 0, &txt, &tn)) != SYI_OK) {
        return fail("detokenize-empty", st);
    }
    syi_free(txt);

    syi_generation_options g;
    syi_generation_options_init(&g);
    g.max_tokens = 8;
    g.temperature = 0.0f;
    g.top_k = 1;
    g.seed = 1;
    syi_request *r = nullptr;
    if ((st = syi_request_create(m, &g, &r)) != SYI_OK) {
        return fail("request_create", st);
    }
    st = syi_request_set_text(r, "Hello", 5);
    if (st == SYI_OK) {
        st = syi_request_submit(r);
    }
    if (st != SYI_OK) {
        return fail("submit", st);
    }
    int ntok = 0;
    for (;;) {
        syi_token t;
        syi_token_init(&t);
        st = syi_request_next(r, &t, SYI_TIMEOUT_INFINITE);
        if (st == SYI_END) {
            break;
        }
        if (st != SYI_OK) {
            return fail("next", st);
        }
        ntok++;
    }
    syi_result res;
    syi_result_init(&res);
    if ((st = syi_request_get_result(r, &res)) != SYI_OK) {
        return fail("result", st);
    }
    if (res.finish_reason != SYI_FINISH_LENGTH || res.completion_tokens != 8 || res.n_tokens != 8) {
        std::fprintf(stderr, "bad result: finish=%d completion=%llu tokens=%zu\n", res.finish_reason,
                     (unsigned long long)res.completion_tokens, res.n_tokens);
        return 1;
    }
    if (ntok != 8) {
        std::fprintf(stderr, "streamed %d tokens, expected 8\n", ntok);
        return 1;
    }
    syi_request_destroy(r);

    // Batched submission/wait: two requests, one slot, FIFO.
    g.max_tokens = 4;
    syi_request *batch[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i) {
        if ((st = syi_request_create(m, &g, &batch[i])) != SYI_OK) {
            return fail("batch_create", st);
        }
        const char *p = i == 0 ? "Hello" : "World";
        if ((st = syi_request_set_text(batch[i], p, std::strlen(p))) != SYI_OK) {
            return fail("batch_set", st);
        }
    }
    if ((st = syi_submit_batch(m, batch, 2)) != SYI_OK) {
        return fail("submit_batch", st);
    }
    if ((st = syi_wait_batch(batch, 2, SYI_TIMEOUT_INFINITE)) != SYI_OK) {
        return fail("wait_batch", st);
    }
    for (int i = 0; i < 2; ++i) {
        syi_result br;
        syi_result_init(&br);
        if ((st = syi_request_get_result(batch[i], &br)) != SYI_OK) {
            return fail("batch_result", st);
        }
        if (br.finish_reason != SYI_FINISH_LENGTH || br.completion_tokens != 4) {
            std::fprintf(stderr, "bad batch result %d: finish=%d completion=%llu\n", i, br.finish_reason,
                         (unsigned long long)br.completion_tokens);
            return 1;
        }
        syi_request_destroy(batch[i]);
    }
    syi_model_destroy(m);
    std::printf("test_capi OK (%d tokens, %llu prompt, batch 2/2)\n", ntok,
                (unsigned long long)res.prompt_tokens);
    return 0;
}
