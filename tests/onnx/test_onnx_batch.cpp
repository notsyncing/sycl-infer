// Batch invariant of the integer Transpose, through the in-process ONNX ABI.
//
//   ./build/test_onnx_batch [model.onnx] [GPU|GPU.N|CPU]
//
// The layout detector's `orders` output is produced by an int64
// `Transpose([B,300], perm=[0,1])` and the executor used to peel the output
// coordinates in column-major order.  That is an identity for B=1 (every
// leading extent is 1) and duplicates/interleaves the elements for B>1, which
// scrambled the layout model's reading order for the second and later pages of
// a batch.  This test pins the semantics without the 130 MB model: row i of a
// batch-N run must equal the batch-1 run's row 0 for the identity transpose,
// and column i must equal its column 0 for the [1,0] transpose; the f32 path
// (a device kernel) is checked the same way.
#include <cstdio>
#include <cstring>
#include <string>
#include <sycl_infer/onnx.h>

namespace {

struct named_out {
    const char *name;
    const void *data;
    const int64_t *shape;
    int32_t ndim;
    int32_t dtype;
};

bool find(const syi_onnx_output * o, size_t n, const char *name, named_out & slot) {
    for (size_t i = 0; i < n; ++i) {
        if (std::strcmp(o[i].name, name) == 0) {
            slot = {o[i].name, o[i].data, o[i].shape, o[i].ndim, o[i].dtype};
            return true;
        }
    }
    return false;
}

bool run(syi_onnx_model * m, int n, int64_t * xi, float * xf, syi_onnx_output * outs,
         size_t cap, size_t & got) {
    int64_t sxi[2] = {n, 3}, sxf[2] = {n, 3};
    syi_onnx_input in[2] = {
        {"x", 2 /* i64 */, 2, sxi, xi, (size_t)n * 3 * 8},
        {"xf", 0 /* f32 */, 2, sxf, xf, (size_t)n * 3 * 4},
    };
    got = 0;
    const int st = syi_onnx_run_direct(m, in, 2, outs, cap, &got);
    if (st != 0) {
        std::fprintf(stderr, "run N=%d: %s\n", n, syi_onnx_last_error());
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    std::string path = argc > 1 ? argv[1] : std::string(SI_TEST_DATA_DIR) + "/transpose_int64.onnx";
    const char *device = argc > 2 ? argv[2] : "GPU";
    syi_onnx_model * m = nullptr;
    if (syi_onnx_create(path.c_str(), "batch", device, nullptr, 1, &m) != 0) {
        std::fprintf(stderr, "create %s: %s\n", path.c_str(), syi_onnx_last_error());
        return 1;
    }

    const int kN = 8;
    int64_t xi[kN * 3];
    float xf[kN * 3];
    for (int i = 0; i < kN; ++i) {
        for (int c = 0; c < 3; ++c) {
            xi[i * 3 + c] = 100 * i + 10 + c;
            xf[i * 3 + c] = 0.5f + 100.0f * i + c;
        }
    }

    syi_onnx_output one[8], many[8];
    size_t n_one = 0, n_many = 0;
    bool ok = run(m, 1, xi, xf, one, 8, n_one);
    named_out id1{}, sw1{}, sf1{};
    if (ok && !(find(one, n_one, "id", id1) && find(one, n_one, "swap", sw1) &&
                find(one, n_one, "swf", sf1))) {
        ok = false;
    }
    if (!ok || id1.dtype != 2 || sw1.dtype != 2 || sf1.dtype != 0) {
        std::fprintf(stderr, "the model's outputs are not the expected id/swap/swf int64/int64/f32\n");
        syi_onnx_destroy(m);
        return 1;
    }

    // N = 1: every permutation is its own inverse for a single row.
    {
        const int64_t * id = (const int64_t *)id1.data;
        const int64_t * sw = (const int64_t *)sw1.data;
        const float * sf = (const float *)sf1.data;
        for (int c = 0; c < 3; ++c) {
            if (id[c] != xi[c] || sw[c] != xi[c] || sf[c] != xf[c]) {
                std::fprintf(stderr, "N=1 mismatch at column %d\n", c);
                ok = false;
            }
        }
    }

    for (int n : {2, 3, kN}) {
        if (!run(m, n, xi, xf, many, 8, n_many)) { ok = false; break; }
        named_out id{}, sw{}, sf{};
        if (!find(many, n_many, "id", id) || !find(many, n_many, "swap", sw) ||
            !find(many, n_many, "swf", sf)) {
            std::fprintf(stderr, "N=%d: missing an output\n", n);
            ok = false;
            break;
        }
        const int64_t * idv = (const int64_t *)id.data;
        const int64_t * swv = (const int64_t *)sw.data;
        const float * sfv = (const float *)sf.data;
        for (int i = 0; i < n; ++i) {
            for (int c = 0; c < 3; ++c) {
                if (idv[i * 3 + c] != xi[i * 3 + c]) {
                    std::fprintf(stderr, "N=%d: identity transpose row %d col %d = %lld, want %lld\n",
                                 n, i, c, (long long)idv[i * 3 + c], (long long)xi[i * 3 + c]);
                    ok = false;
                }
                if (swv[c * n + i] != xi[i * 3 + c]) {
                    std::fprintf(stderr, "N=%d: swap transpose [%d,%d] = %lld, want %lld\n", n, c, i,
                                 (long long)swv[c * n + i], (long long)xi[i * 3 + c]);
                    ok = false;
                }
                if (sfv[c * n + i] != xf[i * 3 + c]) {
                    std::fprintf(stderr, "N=%d: f32 transpose [%d,%d] = %g, want %g\n", n, c, i,
                                 (double)sfv[c * n + i], (double)xf[i * 3 + c]);
                    ok = false;
                }
            }
        }
        if (!ok) break;
    }

    syi_onnx_destroy(m);
    if (!ok) return 1;
    std::printf("test_onnx_batch OK: int64/f32 Transpose is per-row at N=1,2,3,%d\n", kN);
    return 0;
}
