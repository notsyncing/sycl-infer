// In-process ONNX ABI smoke test: load a model, read its ports, destroy.
//
//   ./build-rel2/test_onnx_api /path/to/q_cls.onnx GPU
//
// The model is a classifier, so loading it is cheap; the run path is covered
// end to end by RapidDoc's pipeline.
#include <cstdio>
#include <cstring>
#include <sycl_infer/onnx.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.onnx> [GPU|CPU|GPU.N]\n", argv[0]);
        return 2;
    }
    const char *device = argc > 2 ? argv[2] : "GPU";
    syi_onnx_model *m = nullptr;
    if (syi_onnx_create(argv[1], "test", device, nullptr, 1, &m) != 0) {
        std::fprintf(stderr, "create: %s\n", syi_onnx_last_error());
        return 1;
    }
    char *info = nullptr;
    size_t n = 0;
    if (syi_onnx_info_json(m, &info, &n) != 0) {
        std::fprintf(stderr, "info: %s\n", syi_onnx_last_error());
        syi_onnx_destroy(m);
        return 1;
    }
    const bool ok = n > 0 && std::strstr(info, "\"inputs\"") && std::strstr(info, "\"outputs\"");
    std::printf("test_onnx_api OK: %zu bytes of info\n", n);
    syi_onnx_free(info);
    syi_onnx_destroy(m);
    return ok ? 0 : 1;
}
