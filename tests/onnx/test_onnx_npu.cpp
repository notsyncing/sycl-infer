// Independent Intel NPU execution path for the ONNX service.  The tiny
// MatMul+Add+Relu fixture is checked in, so this needs no document model.
// A machine without the NPU Level Zero driver reports a skip and passes.

#include <sycl_infer/onnx.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef SI_TEST_DATA_DIR
#define SI_TEST_DATA_DIR "."
#endif

static bool close(float a, float b) { return std::fabs(a - b) <= 2e-3f * (1.0f + std::fabs(b)); }

int main(int argc, char ** argv) {
    const std::string path = argc > 1 ? argv[1] : std::string(SI_TEST_DATA_DIR) + "/tiny_matmul.onnx";
    syi_onnx_model * m = nullptr;
    int r = syi_onnx_create(path.c_str(), "tiny", "npu", nullptr, 0, &m);
    if (r != 0) {
        const std::string e = syi_onnx_last_error();
        if (e.find("no Level Zero NPU driver") != std::string::npos ||
            e.find("no Level Zero VPU device") != std::string::npos ||
            e.find("libze_loader") != std::string::npos) {
            std::printf("SKIP: no Intel NPU available (%s)\n", e.c_str());
            return 0;
        }
        std::printf("FAIL: syi_onnx_create: %s\n", e.c_str());
        return 1;
    }
    char * info = nullptr;
    size_t info_len = 0;
    r = syi_onnx_info_json(m, &info, &info_len);
    if (r != 0) {
        std::printf("FAIL: info: %s\n", syi_onnx_last_error());
        return 1;
    }
    std::printf("info: %.*s\n", static_cast<int>(info_len), info);
    syi_onnx_free(info);

    const float x[4] = {2.0f, 1.0f, 0.5f, 3.0f};
    const int64_t shape[2] = {1, 4};
    syi_onnx_input in;
    in.name = "x";
    in.dtype = 0;
    in.ndim = 2;
    in.shape = shape;
    in.data = x;
    in.nbytes = sizeof(x);
    syi_onnx_output out[4];
    size_t n = 0;
    r = syi_onnx_run_direct(m, &in, 1, out, 4, &n);
    if (r != 0 || n != 1) {
        std::printf("FAIL: run_direct: r=%d n=%zu error=%s\n", r, n, syi_onnx_last_error());
        return 1;
    }
    const float want[3] = {0.0f, 0.25f, 11.5f};
    if (out[0].nbytes != 12 || out[0].ndim != 2) {
        std::printf("FAIL: output shape: ndim=%d bytes=%zu\n", out[0].ndim, out[0].nbytes);
        return 1;
    }
    const float * got = static_cast<const float *>(out[0].data);
    for (int i = 0; i < 3; ++i) std::printf("y[%d] = %.5f (want %.5f)\n", i, got[i], want[i]);
    for (int i = 0; i < 3; ++i) {
        if (!close(got[i], want[i])) {
            std::printf("FAIL: NPU output differs at %d\n", i);
            return 1;
        }
    }
    syi_onnx_destroy(m);
    std::printf("OK\n");
    return 0;
}
