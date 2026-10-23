#pragma once
// A graph transform is a function over graph_build.  The builder runs the
// registered list to a fixed point before it lowers the executor ops; see
// graph.cpp and the individual transforms/ files.

#include "graph_build.h"

namespace si {
namespace onnx {
namespace transforms {

using transform_fn = bool (*)(graph_build &);

struct transform {
    const char * name;
    transform_fn run;
};

// The ordered pipeline.  Each entry is defined in its own .cpp:
//   conv_epilogue.cpp        convolution + batch norm + activation
//   elementwise_activation.cpp  Add/Sub/Mul/Div + activation
//   matmul_bias.cpp          MatMul + constant bias (+ activation)
//   mask_box.cpp             the layout mask-to-box branch
const transform * pipeline(size_t & count);

// Run the pipeline until no pass changes the graph (bounded).  `SI_ONNX_NOFUSE`
// disables the whole pipeline; the graph remains loadable.
void run(graph_build & g);

} // namespace transforms
} // namespace onnx
} // namespace si
