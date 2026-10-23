#include "transform.h"

#include <cstdlib>

namespace si {
namespace onnx {
namespace transforms {

bool conv_epilogue(graph_build & g);
bool elementwise_activation(graph_build & g);
bool matmul_bias(graph_build & g);
bool mask_box(graph_build & g);

namespace {
const transform kPipeline[] = {
    {"conv_epilogue", conv_epilogue},
    {"elementwise_activation", elementwise_activation},
    {"matmul_bias", matmul_bias},
    {"mask_box", mask_box},
};
} // namespace

const transform * pipeline(size_t & count) {
    count = sizeof(kPipeline) / sizeof(kPipeline[0]);
    return kPipeline;
}

void run(graph_build & g) {
    if (!g.fuse) return;
    size_t n = 0;
    const transform * p = pipeline(n);
    for (int iteration = 0; iteration < 8; ++iteration) {
        bool changed = false;
        for (size_t i = 0; i < n; ++i) {
            changed = p[i].run(g) || changed;
        }
        if (!changed) break;
    }
}

} // namespace transforms
} // namespace onnx
} // namespace si
