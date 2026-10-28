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
#include "onnx_wire.h"

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
    // Drop the free device buffers (and wait for the queue first).  Long
    // multi-model sessions accumulate one pool entry per input shape; a
    // document pipeline calls this at a document boundary.
    void trim_pool();

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

    // The outputs the client never reads (a model config's `drop_outputs`):
    // the executor does not read them back to the host, and the service does
    // not put them on the wire.  The layout model's 300x200x200 mask is 48 MB
    // per page, and the pipeline's rect mode never looks at it.
    void skip_outputs(const std::vector<std::string> & names);
    bool output_skipped(size_t i) const;
    // Whether this model's convolutions may use oneDNN.  A model whose
    // input shape changes per request turns it off: the primitive cache is
    // per shape, and the table pipeline presents dozens of them.
    void set_dnnl(bool on) { dnnl_ = on; }

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
        // The device copy of a host-resident value, and whether it is
        // current.
        //
        // A value that lives on the host (an integer, a shape, a constant)
        // is read by the device through a mirror, and the write-back used to
        // throw that mirror away at every operator - so a value read by N
        // operators was uploaded N times, and a constant was uploaded again
        // on every call (3508 copies of 707 MB per layout call, which was
        // half of the call's time).  `host_gen` counts the writes of the
        // host bytes and `dev_gen` is the generation the mirror was made
        // from; they are equal while the device copy is current.
        uint64_t host_gen = 0;
        uint64_t dev_gen = 0;
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
    // How many times the pool had to ask the device for a new buffer (a
    // pool miss: the diagnostic SI_ONNX_HOSTTIME reports it).
    size_t malloc_count_ = 0;
    // The host->device uploads of the inputs, and their bytes.  Per run:
    // their total is what SI_ONNX_HOSTTIME reports, and a cumulative counter
    // made the second call look exactly like the first.
    size_t upload_count_ = 0;
    size_t upload_bytes_ = 0;
    // Whether the uploads of the last run may still be in the queue.  The
    // copies read the graph tensors' host bytes, and the next run overwrites
    // them (a graph input's bytes are replaced before the first operator), so
    // it waits once for the previous run's copies instead of waiting at every
    // upload - a wait per upload drains the whole in-order queue.
    bool uploads_pending_ = false;
    // SI_ONNX_MIRRORCHK: verify every reused mirror against its host bytes
    // before the consumer reads it.  The generations are what make the reuse
    // correct, and a missed bump is invisible in the output - so the
    // invariant is checked on demand rather than assumed.
    bool mirror_chk_ = false;
    bool dnnl_ = true;
    // The live device bytes and their high-water mark, maintained on
    // acquire/release (the pool's own total is the set of distinct buffers,
    // which is not the same thing as the peak concurrent need).
    size_t live_now_ = 0;
    size_t peak_live_ = 0;
    // The pinned (USM host) staging buffer an upload is copied through.  A
    // copy from pageable memory is synchronous on this backend: it blocks
    // until the queue drains, so a tiny value uploaded behind 80 ms of
    // kernels cost 80 ms.  The staging copy is asynchronous, and the buffer
    // is reused only after the queue has drained.
    static constexpr size_t kUpStageBytes = (size_t)8 << 20;
    void * up_stage_ = nullptr;
    size_t up_stage_cap_ = 0;
    size_t up_stage_off_ = 0;
    // Per-slot upload tallies for SI_ONNX_UPLOADDBG: which values are copied
    // again and again.  The sizes are counted only when the diagnostic is
    // armed, because the arrays are one entry per tensor and a model can
    // have thousands.
    bool upload_dbg_ = false;
    std::vector<uint32_t> upload_by_slot_;
    std::vector<size_t> upload_bytes_by_slot_;
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
    // The graph's output slots the client asked not to receive.
    std::set<int> skip_out_;
    // The operators a kept output needs, for the backward mark in
    // skip_outputs().  Empty when nothing is dropped (every op runs).
    std::vector<bool> live_ops_;
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
