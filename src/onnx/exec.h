#pragma once
// The ONNX executor: it runs an executable graph.
//
// The executor owns the device memory.  A buffer is reused once the value
// it holds is dead (its last consumer has run), so a graph with a thousand
// operators and a few hundred live tensors runs in the memory of the live
// set - not of the sum.
//
// The values that are known at build time (the weights, and the results of
// the constant sub-graphs) stay on the host and are uploaded when an
// operator needs them.  The constant sub-graphs are evaluated once, on the
// device, the first time the graph runs.

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "graph.h"
#include "wire.h"

namespace si {
namespace onnx {

class executor {
public:
    executor(exec_graph g, sycl::queue & q);
    ~executor();

    // Run the graph.  The inputs carry their shape (the graph's own input
    // shape is often dynamic); the outputs come back the same way.  A
    // tensor that is missing is an error.
    void run(const std::vector<onnxwire::tensor> & inputs,
             const std::vector<std::pair<std::string, std::vector<uint8_t> &>> & outputs);

    // The number of times the graph has run, and the last run's time.
    uint64_t calls() const { return calls_; }
    double last_ms() const { return last_ms_; }
    // The device memory the graph holds: the live allocations and the
    // buffers in the pool.  The service reports it, because a document
    // pipeline's models share one card and the biggest one decides how many
    // of them fit.
    size_t live_bytes() const { return live_bytes_calc(); }
    size_t pooled_bytes() const { return pooled_bytes_calc(); }

    const exec_graph & graph() const { return g_; }

    // A tag for the dump files.  A Loop's body is a graph of its own, and
    // it can hold a value with the same name as one in the enclosing
    // graph (an outer-scope read) - so the two dumps would write the same
    // file and the later one would win.
    void set_dump_tag(const std::string & tag) { dump_tag_ = tag; }

    // Copy the value of a slot of this graph into another executor's slot.
    //
    // This is the outer scope: ONNX lets a subgraph read the values of the
    // graph that holds it, and the formula recogniser's Loop body reads a
    // transpose that the enclosing graph computes.  The value is on the
    // device (a float), so it comes back to the host and goes out again.
    void copy_into(executor & dst, int dst_slot, int src_slot);

    // The graph's outputs, as they are after the last run: the shape and
    // the data.  The caller must read them from here and not from the
    // graph it handed over: the shapes are only known once the graph has
    // run.
    size_t num_outputs() const { return g_.output_slots.size(); }
    const tensor & output(size_t i) const { return g_.tensors[g_.output_slots[i]]; }

private:
    struct slot_state {
        // The device buffer (null when the value lives on the host).
        void * dev = nullptr;
        size_t bytes = 0;
        // Whether the value is a constant (known at build time).
        bool is_const = false;
        // The last operator that reads the value.
        int last_use = -1;
        bool live = false;
    };

    // The host copy of a slot.
    //
    // It is the *tensor's* host, not a second copy: the operators and the
    // shape functions read the tensor, and a second copy here is a second
    // thing to keep in step - and it was not.  The folded constants (the
    // values a Constant node produces) lived only in the slot, so the shape
    // functions read them as empty: a Slice's bounds came out as no bounds
    // at all, and the value it produced had the input's length instead of
    // the sliced one.
    std::vector<uint8_t> & host_of(int slot) { return g_.tensors[slot].host; }

    void * acquire(size_t bytes);
    void release(void * p, size_t bytes);
    // The device's free memory, for the error message of a failed allocation.
    size_t free_mem_bytes() const;
    // Make sure the value is on the device, uploading it if needed.
    const float * device_value(int slot);
    // The Loop's body: built on demand (the constructor builds it eagerly,
    // so the outer scope's liveness can be extended before the run).
    executor * loop_body(size_t op_index);
    // The Loop: the body's graph, run once per iteration.
    void run_loop(size_t op_index, exec_op & op);
    // Whether a slot is one of the graph's outputs (whose buffer stays alive
    // until the client has them).
    bool is_output(int slot) const;
    // The scratch buffer for the operators' small arrays.
    void * scratch();
    // The small device buffer (64 KB).
    void * scratch_small();
    // The op's inputs, uploaded.
    void prepare_inputs(exec_op & op);
    void fold_constants();
    // The convolution weights, transposed once the constants are known.
    void prepare_conv_weights();
    // The leak report (SI_ONNX_LEAK).
    void report_leak() const;

    exec_graph g_;
    sycl::queue & q_;
    std::vector<slot_state> state_;
    // The Loop bodies, built once.
    //
    // A Loop's body is a graph of its own: building it folds its constant
    // sub-graphs and transposes its convolution weights, which is the work of
    // a whole model.  Doing that once per *iteration* of the loop is not a
    // detail - the structure recogniser's loop runs 501 times, and the
    // rebuild dominates everything else.  The body is built the first time
    // the loop runs and kept.
    std::map<size_t, std::unique_ptr<executor>> loop_bodies_;
    // The tag in the dump file names (empty for the enclosing graph).
    std::string dump_tag_;
    // The free device buffers, by their *real* size.  The size a buffer is
    // released with is the size it was *asked* for, which is not the size it
    // has: the pool hands out the smallest buffer that fits, so a buffer
    // allocated for 8 MB can be released as 1 MB.  Keying the pool by the
    // requested size would then hand a 1 MB buffer to an operator that
    // asked for 4 MB, and the operator would write past the end of it.
    std::map<size_t, std::vector<void *>> free_;
    // The real size of every live allocation.
    std::map<void *, size_t> alloc_size_;
    // Every live allocation, so a double hand-out is caught where it
    // happens instead of as a wrong number much later.
    std::set<void *> live_;
    // The operator that wrote each slot, so a value written twice is caught.
    std::vector<int> producer_;
    // The live and the pooled device bytes, for the report.
    size_t live_bytes_calc() const;
    size_t pooled_bytes_calc() const;
    // The scratch buffer the operators use for their small arrays (the
    // strides, the indices).  It is a *device* buffer: a host array read
    // from a device kernel reads freed stack, and on a discrete GPU it
    // faults.
    void * ws_ = nullptr;
    size_t ws_bytes_ = 0;
    // The small device buffer the operators convert a value into.
    void * ws2_ = nullptr;
    size_t ws2_bytes_ = 0;
    // The constant values that have been evaluated.

    bool folded_ = false;
    // The branch of an If that does *not* run: its operators are in the op
    // list, between these two indices, and they are skipped.
    int skip_begin_ = -1;
    int skip_end_ = -1;
    uint64_t calls_ = 0;
    double last_ms_ = 0.0;
};

} // namespace onnx
} // namespace si
