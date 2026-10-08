// Load-time geometry validation (validate_hparams).  Hermetic: no GGUF, no GPU.
//
// The point of this is that an unsupported shape must be *refused*.  Several
// kernels hardcode a head_dim rather than reading it - attn.cpp's generic
// attention sizes each split's partials with a literal `constexpr int HD = 256`
// while indexing them with `pstride = 2 + head_dim`, so a head_dim of 128 makes
// it write 256 floats into a 130-float slot and overrun the buffer.  The
// special-cased kernels (oneDNN attention, flash, grouped) all guard on
// head_dim == HD and fall through to that same generic one, so no other value is
// a supported shape: the file must be rejected at load time.
//
// Each case below is a geometry a GGUF could plausibly carry.
#include <cstdio>
#include <stdexcept>
#include <string>

#include "model.h"

using namespace si;

static int g_fail = 0;

// A geometry both shipped models satisfy: 0.8B (n_head 16, kv 16) and the 27B
// (head_dim 256, GQA ratio 4 or 6) - see the test's comments below.
static hparams good() {
    hparams hp;
    hp.n_layer = 24;
    hp.n_embd = 2048;
    hp.n_ff = 6144;
    hp.n_head = 16;
    hp.n_head_kv = 16;
    hp.head_dim = 256;
    hp.n_vocab = 151936;
    hp.d_state = 128;
    hp.n_group = 16;
    hp.dt_rank = 16;
    hp.conv_k = 4;
    hp.full_attn_interval = 4;
    return hp;
}

static void accepts(const char * what, const hparams & hp) {
    try {
        validate_hparams(hp, "qwen35");
        printf("  ok       %s\n", what);
    } catch (const std::exception & ex) {
        fprintf(stderr, "FAIL %s should be accepted, got: %s\n", what, ex.what());
        g_fail++;
    }
}

static void rejects(const char * what, const hparams & hp, const char * expect_substr) {
    try {
        validate_hparams(hp, "qwen35");
    } catch (const std::exception & ex) {
        const std::string msg = ex.what();
        if (msg.find(expect_substr) == std::string::npos) {
            fprintf(stderr, "FAIL %s: message %s does not mention %s\n", what, msg.c_str(), expect_substr);
            g_fail++;
            return;
        }
        printf("  rejected %-34s -> %s\n", what, msg.c_str());
        return;
    }
    fprintf(stderr, "FAIL %s should be rejected (expected a message about %s)\n", what, expect_substr);
    g_fail++;
}

int main() {
    printf("validate_hparams\n");
    accepts("the 0.8B geometry", good());

    // GQA with a real ratio, as the 27B has
    {
        hparams hp = good();
        hp.n_head = 32;
        hp.n_head_kv = 8;
        accepts("GQA ratio 4", hp);
        hp.n_head_kv = 32;
        accepts("GQA ratio 1", hp);
    }
    // A model whose first layer is full attention has no GDN geometry to check
    {
        hparams hp = good();
        hp.full_attn_interval = 1; // is_recr(0) == false
        hp.d_state = hp.n_group = hp.dt_rank = hp.conv_k = 0;
        accepts("no recurrent layers", hp);
    }

    // The one that used to corrupt memory
    rejects("head_dim 128", [&] {
        hparams hp = good();
        hp.head_dim = 128;
        hp.n_embd = 2048;
        return hp;
    }(), "head_dim");
    rejects("head_dim 64", [&] {
        hparams hp = good();
        hp.head_dim = 64;
        return hp;
    }(), "head_dim");

    rejects("n_head not a multiple of n_head_kv", [&] {
        hparams hp = good();
        hp.n_head = 18;
        hp.n_head_kv = 16;
        return hp;
    }(), "head geometry");
    rejects("n_head_kv 0", [&] {
        hparams hp = good();
        hp.n_head_kv = 0;
        return hp;
    }(), "head geometry");
    rejects("n_head 0", [&] {
        hparams hp = good();
        hp.n_head = 0;
        return hp;
    }(), "head geometry");

    rejects("n_layer 0", [&] {
        hparams hp = good();
        hp.n_layer = 0;
        return hp;
    }(), "incomplete hparams");
    rejects("n_vocab 0", [&] {
        hparams hp = good();
        hp.n_vocab = 0;
        return hp;
    }(), "incomplete hparams");

    rejects("full_attn_interval 0", [&] {
        hparams hp = good();
        hp.full_attn_interval = 0;
        return hp;
    }(), "full_attn_interval");

    rejects("recurrent layer with d_state 0", [&] {
        hparams hp = good();
        hp.d_state = 0;
        return hp;
    }(), "recurrent layers");
    rejects("recurrent layer with conv_k 0", [&] {
        hparams hp = good();
        hp.conv_k = 0;
        return hp;
    }(), "recurrent layers");

    // The architecture name reaches the message, so an operator can tell which
    // loader produced it.
    try {
        hparams hp = good();
        hp.head_dim = 128;
        validate_hparams(hp, "somefuturearch");
    } catch (const std::exception & ex) {
        if (std::string(ex.what()).find("somefuturearch:") == std::string::npos) {
            fprintf(stderr, "FAIL the architecture name is missing from: %s\n", ex.what());
            g_fail++;
        }
    }

    if (g_fail) {
        fprintf(stderr, "test_hparams: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test_hparams: all checks OK\n");
    return 0;
}