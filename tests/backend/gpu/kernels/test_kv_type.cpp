// KV storage type: name parsing, mixed K:V validation, element geometry and
// the host accessor.  Hermetic: the functions under test touch no queue and no
// device, so this runs on any machine.
//
// Why: `--kv-type` is a user-facing string that selects a memory layout, and a
// wrong parse does not fail - it silently runs with the default.  The mixed
// K:V form is narrower than "any two names" (only the scale-carrying i4/i8 pair
// is implemented), so the rejection cases matter as much as the accepted ones.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "kv_type.h"

using namespace si;

static int g_fail = 0;

static void check(bool ok, const char * name) {
    if (ok) {
        printf("  [%s] OK\n", name);
        return;
    }
    g_fail++;
    printf("  [%s] FAILED\n", name);
}

static void parse_is(const char * spec, bool want_ok, kv_dtype_t want) {
    kv_dtype_t got = kv_dtype_t::bf16;
    const bool ok = kv_dtype_parse(spec, got);
    const std::string label = std::string("parse ") + (spec ? spec : "(null)");
    check(ok == want_ok && (!ok || got == want), label.c_str());
}

static void pair_is(const char * spec, bool want_ok, kv_dtype_t wk = kv_dtype_t::i8, kv_dtype_t wv = kv_dtype_t::i8) {
    kv_dtype_t k = kv_dtype_t::bf16, v = kv_dtype_t::bf16;
    const bool ok = kv_dtype_parse_pair(spec, k, v);
    const std::string label = std::string("pair ") + (spec ? spec : "(null)");
    check(ok == want_ok && (!ok || (k == wk && v == wv)), label.c_str());
}

int main() {
    printf("kv_dtype_parse\n");
    parse_is("f32", true, kv_dtype_t::f32);
    parse_is("fp32", true, kv_dtype_t::f32);
    parse_is("0", true, kv_dtype_t::f32);
    parse_is("bf16", true, kv_dtype_t::bf16);
    parse_is("f16", true, kv_dtype_t::f16);
    parse_is("fp16", true, kv_dtype_t::f16);
    parse_is("i8", true, kv_dtype_t::i8);
    parse_is("int8", true, kv_dtype_t::i8);
    parse_is("q8", true, kv_dtype_t::i8);
    parse_is("i4", true, kv_dtype_t::i4);
    parse_is("int4", true, kv_dtype_t::i4);
    parse_is("q4", true, kv_dtype_t::i4);
    parse_is("", false, kv_dtype_t::i8);
    parse_is("i7", false, kv_dtype_t::i8);
    parse_is("i8 ", false, kv_dtype_t::i8); // no trimming: a typo is not a type
    parse_is(nullptr, false, kv_dtype_t::i8);

    printf("kv_dtype_parse_pair\n");
    pair_is("i4:i8", true, kv_dtype_t::i4, kv_dtype_t::i8);
    pair_is("i8:i4", true, kv_dtype_t::i8, kv_dtype_t::i4);
    pair_is("i4:i4", true, kv_dtype_t::i4, kv_dtype_t::i4);
    pair_is("f16", true, kv_dtype_t::f16, kv_dtype_t::f16);
    // The kernels only implement the two scale-carrying types as a mix: a
    // different-width fp pair has no shared scale plane.
    pair_is("f16:f32", false);
    pair_is("i8:f16", false);
    pair_is("i8:", false);
    pair_is(":i8", false);
    pair_is("i8:i8:i8", false);
    pair_is("bogus:i8", false);
    pair_is(nullptr, false);

    printf("geometry\n");
    check(kv_dtype_bytes(kv_dtype_t::f32) == 4, "f32 bytes");
    check(kv_dtype_bytes(kv_dtype_t::bf16) == 2, "bf16 bytes");
    check(kv_dtype_bytes(kv_dtype_t::i8) == 1, "i8 bytes");
    // i4 keeps one byte per element in the accessor API but packs two values,
    // which is why bits (not bytes) is the size that halves.
    check(kv_dtype_bits(kv_dtype_t::i4) == 4, "i4 bits");
    check(kv_dtype_bits(kv_dtype_t::i8) == 8, "i8 bits");
    check(kv_dtype_row_bytes(kv_dtype_t::i4, 256) == 128, "i4 row bytes");
    check(kv_dtype_row_bytes(kv_dtype_t::i8, 256) == 256, "i8 row bytes");
    check(kv_dtype_row_bytes(kv_dtype_t::f16, 256) == 512, "f16 row bytes");
    check(kv_dtype_has_scales(kv_dtype_t::i4) && kv_dtype_has_scales(kv_dtype_t::i8), "i4/i8 carry scales");
    check(!kv_dtype_has_scales(kv_dtype_t::f16) && !kv_dtype_has_scales(kv_dtype_t::bf16) &&
              !kv_dtype_has_scales(kv_dtype_t::f32),
          "fp types carry no scale plane");

    printf("names\n");
    check(!strcmp(kv_dtype_name(kv_dtype_t::f32), "f32"), "name f32");
    check(!strcmp(kv_dtype_name(kv_dtype_t::f16), "f16"), "name f16");
    check(!strcmp(kv_dtype_name(kv_dtype_t::i8), "i8"), "name i8");
    check(!strcmp(kv_dtype_name(kv_dtype_t::i4), "i4"), "name i4");
    check(!strcmp(kv_dtype_name(kv_dtype_t::bf16), "bf16"), "name bf16");

    printf("host accessor\n");
    {
        std::vector<float> f = {1.5f, -2.25f};
        check(kv_ld_host(kv_dtype_t::f32, f.data(), 0) == 1.5f, "f32 element");
        check(kv_ld_host(kv_dtype_t::f32, f.data(), 1) == -2.25f, "f32 element 1");

        std::vector<sycl::half> h = {sycl::half(0.5f), sycl::half(-1.0f)};
        check(std::fabs(kv_ld_host(kv_dtype_t::f16, h.data(), 0) - 0.5f) < 1e-6f, "f16 element");
        check(std::fabs(kv_ld_host(kv_dtype_t::f16, h.data(), 1) + 1.0f) < 1e-6f, "f16 element 1");

        std::vector<sycl::ext::oneapi::bfloat16> b = {sycl::ext::oneapi::bfloat16(2.0f)};
        check(std::fabs(kv_ld_host(kv_dtype_t::bf16, b.data(), 0) - 2.0f) < 1e-3f, "bf16 element");

        std::vector<int8_t> q = {-128, -1, 0, 127};
        check(kv_ld_host(kv_dtype_t::i8, q.data(), 0) == -128.f, "i8 -128");
        check(kv_ld_host(kv_dtype_t::i8, q.data(), 3) == 127.f, "i8 127");

        // i4 is two's-complement nibbles, low nibble = even index: 0xF is -1,
        // 0x8 is -8, 0x0 is 0 and 0x7 is 7.
        std::vector<uint8_t> n = {0x8F, 0x70};
        check(kv_ld_host(kv_dtype_t::i4, n.data(), 0) == -1.f, "i4 low nibble 0xF is -1");
        check(kv_ld_host(kv_dtype_t::i4, n.data(), 1) == -8.f, "i4 high nibble 0x8 is -8");
        check(kv_ld_host(kv_dtype_t::i4, n.data(), 2) == 0.f, "i4 low nibble 0");
        check(kv_ld_host(kv_dtype_t::i4, n.data(), 3) == 7.f, "i4 high nibble 7");
    }

    printf("scoped override\n");
    {
        kv_dtype_set_kv(kv_dtype_t::i8, kv_dtype_t::i4);
        check(kv_k_dtype() == kv_dtype_t::i8 && kv_v_dtype() == kv_dtype_t::i4, "process-wide set");
        {
            kv_scope outer(kv_dtype_t::f32, kv_dtype_t::f16);
            check(kv_k_dtype() == kv_dtype_t::f32 && kv_v_dtype() == kv_dtype_t::f16, "outer scope");
            {
                kv_scope inner(kv_dtype_t::i4, kv_dtype_t::i8);
                check(kv_k_dtype() == kv_dtype_t::i4 && kv_v_dtype() == kv_dtype_t::i8, "nested scope");
            }
            check(kv_k_dtype() == kv_dtype_t::f32 && kv_v_dtype() == kv_dtype_t::f16, "nested scope restores outer");
        }
        check(kv_k_dtype() == kv_dtype_t::i8 && kv_v_dtype() == kv_dtype_t::i4,
              "scope restores the process-wide value");
        kv_dtype_set(kv_dtype_t::i8);
        check(kv_k_dtype() == kv_dtype_t::i8 && kv_v_dtype() == kv_dtype_t::i8, "single-type set");
    }

    printf(g_fail ? "kv_type test FAILED (%d)\n" : "kv_type test OK\n", g_fail);
    return g_fail ? 1 : 0;
}
