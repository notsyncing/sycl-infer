#include "exec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "kernels.h"
#include "ops.h"
#include "conv_dnnl.h"

namespace si {
namespace onnx {

// The executor owns the device memory and the run loop.
//
// Three things it has to get right, and each of them was a bug:
//
//   * A value stays on the device.  The host copy is made only for the
//     values that need it (the constants, and the graph's outputs): a host
//     round trip per operator is the difference between a GPU runtime and a
//     slow one.
//   * A buffer goes back to the pool only when its value is dead.  Releasing
//     an output right after the operator that produced it hands the same
//     memory to the next operator, and the two values alias - which is how
//     a correct pipeline produces garbage without any visible error.
//   * The pool tracks the *real* size of a buffer, not the size it was
//     asked for: the pool hands out the smallest buffer that fits, so a
//     buffer allocated for 8 MB can be released as 1 MB.

namespace {

// Defined below: the integer operators that propagate a *shape* (a
// `Shape`'s output, a reshape target's arithmetic) rather than data.
bool shape_arith_op(const std::string & ty);

// The trace: the operator and its slot, for a run that has to be bisected.
bool trace_on() {
    static const bool on = getenv("SI_ONNX_TRACE") != nullptr;
    return on;
}

// Whether a value is true.
//
// **A BOOL is a float 0/1 in this runtime** (the comparison operators
// produce a float and the boolean operators work on floats), so the bytes
// have to be read as the value's *type*: reading the first byte of a float
// 1.0f reads 0x00 (it is little-endian) and the condition is false.
//
// (The Loop's condition is exactly this: the body computes it as a float
// 0/1, the write-back keeps it on the device because it is not an integer,
// and a host read of an empty host copy says "false" - so the loop's body
// never ran at all.)
bool value_is_true(const std::vector<uint8_t> & h, int32_t dtype) {
    if (h.empty()) return false;
    if (dtype == onnxpb::FLOAT) {
        float v = 0.0f;
        std::memcpy(&v, h.data(), std::min<size_t>(sizeof(v), h.size()));
        return v != 0.0f;
    }
    if (dtype == onnxpb::INT64) {
        int64_t v = 0;
        std::memcpy(&v, h.data(), std::min<size_t>(sizeof(v), h.size()));
        return v != 0;
    }
    if (dtype == onnxpb::INT32) {
        int32_t v = 0;
        std::memcpy(&v, h.data(), std::min<size_t>(sizeof(v), h.size()));
        return v != 0;
    }
    if (dtype == onnxpb::BOOL) {
        // A boolean is a float 0/1 in this runtime, and 1.0f's first byte is
        // 0x00 - so a byte test says "false" for a true value.
        float v = 0.0f;
        std::memcpy(&v, h.data(), std::min<size_t>(sizeof(v), h.size()));
        return v != 0.0f;
    }
    return h[0] != 0;
}

} // namespace

executor::executor(exec_graph g, sycl::queue & q) : g_(std::move(g)), q_(q) {
    state_.resize(g_.tensors.size());
    producer_.assign(g_.tensors.size(), -1);
    oc_.resize(g_.ops.size());
    for (size_t i = 0; i < g_.ops.size(); ++i) {
        oc_[i].entry = find_op(g_.ops[i].node->op_type);
        if (g_.ops[i].node->op_type == "Loop") has_loop_ = true;
    }
    upload_dbg_ = getenv("SI_ONNX_UPLOADDBG") != nullptr;
    mirror_chk_ = getenv("SI_ONNX_MIRRORCHK") != nullptr;
    // The per-operator diagnostics, read once: see exec.h's note on why a
    // `getenv` inside the op loop was a measurable cost of the Loop body.
    trace_ = getenv("SI_ONNX_TRACE") != nullptr;
    loop_dbg_ = getenv("SI_ONNX_LOOPDBG") != nullptr;
    shape_dbg_ = getenv("SI_ONNX_SHAPE") != nullptr;
    dump_dbg_ = getenv("SI_ONNX_DUMP") != nullptr;
    time_dbg_ = getenv("SI_ONNX_TIME") != nullptr;
    dump_file_dbg_ = getenv("SI_ONNX_DUMP_FILE") != nullptr;
    const_chk_ = getenv("SI_ONNX_CONSTCHK") != nullptr;
    if (dump_dbg_) dump_nvals_ = atoi(getenv("SI_ONNX_DUMP"));
    dump_file_dir_ = dump_file_dbg_ ? getenv("SI_ONNX_DUMP_FILE") : nullptr;
    if (const char * g = getenv("SI_ONNX_GUARD")) guard_ = (size_t)atoi(g);
    opstat_ = getenv("SI_ONNX_OPSTAT") != nullptr;
    if (const char * pm = getenv("PF_ONNX_POOL_MB")) {
        const long v = atol(pm);
        if (v > 0) pool_cap_ = (size_t)v << 20;
    }
    ophost_dbg_ = getenv("SI_ONNX_OPHOST") != nullptr;
    intdev_dbg_ = getenv("SI_ONNX_INTDEV") != nullptr;
    if (upload_dbg_) {
        upload_by_slot_.assign(g_.tensors.size(), 0);
        upload_bytes_by_slot_.assign(g_.tensors.size(), 0);
    }
    for (size_t i = 0; i < g_.tensors.size(); ++i) {
        // A constant is a constant even when it is *empty*: an empty tensor
        // is a value (a Reshape's target can be an empty shape, which ONNX
        // defines as "a scalar"), and treating it as "no value" makes the
        // operator that reads it fail with a message about a missing input.
        state_[i].is_const = g_.tensors[i].on_host;
    }
    // The last operator that reads each value: its buffer is free after
    // that operator.
    for (size_t i = 0; i < g_.ops.size(); ++i) {
        for (int s : g_.ops[i].in) {
            if (s >= 0 && s < (int)state_.size()) state_[s].last_use = (int)i;
        }
    }
    // A Loop's body reads the enclosing graph's values, and *no* operator of
    // this graph does - so without this the pool recycles them long before
    // the Loop runs, and the body computes with zeros.  The bodies are built
    // here, not on the first iteration, because the liveness has to be
    // extended *before* the operators run.
    for (size_t i = 0; i < g_.ops.size(); ++i) {
        if (g_.ops[i].node->op_type != "Loop") continue;
        const exec_graph & bg = loop_body(i)->graph();
        for (const auto & r : bg.outer_refs) {
            if (r.second >= 0 && r.second < (int)state_.size()) {
                state_[r.second].last_use = std::max(state_[r.second].last_use, (int)i);
            }
        }
    }
    // The shape arithmetic: a constant or a `Shape`'s output, propagated
    // through the integer metadata/arithmetic operators.  The formula
    // recogniser's reshape targets are built this way, and the values are a
    // few scalars; a token id (`ArgMax`) is not, and it must stay on the
    // device so the embedding gather that consumes it does not pull it back
    // (and drain the queue) every iteration.
    shape_derived_.assign(g_.tensors.size(), 0);
    for (size_t i = 0; i < g_.tensors.size(); ++i) {
        if (g_.tensors[i].on_host || state_[i].is_const) shape_derived_[i] = 1;
    }
    for (const exec_op & op : g_.ops) {
        const std::string & ty = op.node->op_type;
        if (ty == "Constant") {
            for (int s : op.out) {
                if (s >= 0 && s < (int)shape_derived_.size()) shape_derived_[s] = 1;
            }
            continue;
        }
        if (ty != "Shape" && !shape_arith_op(ty)) continue;
        bool all = !op.in.empty();
        for (int s : op.in) {
            if (s < 0) continue;
            if (s >= (int)shape_derived_.size() || !shape_derived_[s]) {
                all = false;
                break;
            }
        }
        if (!all && ty != "Shape") continue;
        for (int s : op.out) {
            if (s >= 0 && s < (int)shape_derived_.size()) shape_derived_[s] = 1;
        }
    }
}

// The outer scope: a value of this graph that a subgraph reads.
//
// The value is on the device (a float), so it comes back to the host and
// the destination uploads it again when its operator runs.  The shape and
// the type travel with it.
void executor::copy_into(executor & dst, int dst_slot, int src_slot) {
    const tensor & st = g_.tensors[src_slot];
    tensor & dt = dst.g_.tensors[dst_slot];
    dt.dtype = st.dtype;
    dt.shape = st.shape;
    dt.count = st.count;
    const size_t esz = runtime_elem_size(st.dtype);
    const size_t want = st.count * esz;
    // A float (or a boolean, which is a float 0/1) lives on the device, and
    // the copy is a *device* copy: a round trip through the host costs a
    // device wait, and the formula recogniser's body reads 816 values of
    // the enclosing graph.
    //
    // The queue is in order, so the copy is in place before the body's
    // operators run - and the source's buffer is released only after the
    // loop, so it is still valid.
    if (st.host.empty() && state_[src_slot].dev && want) {
        void * d = dst.acquire(want);
        if (!d) throw std::runtime_error("device allocation failed");
        q_.memcpy(d, state_[src_slot].dev, want);
        dt.host.clear();
        dt.on_host = false;
        dst.state_[dst_slot].is_const = false;
        dst.state_[dst_slot].bytes = want;
        if (dst.state_[dst_slot].dev) {
            dst.release(dst.state_[dst_slot].dev, dst.state_[dst_slot].bytes);
        }
        dst.state_[dst_slot].dev = d;
        // The *tensor's* device pointer as well: the body's operators read
        // their inputs through it, and it was still the old (empty) buffer -
        // so the body computed with zeros.
        dt.dev = d;
        // The value is device-authoritative now: there is no host copy for
        // the mirror to disagree with.
        ++dst.state_[dst_slot].host_gen;
        dst.state_[dst_slot].dev_gen = dst.state_[dst_slot].host_gen;
        // The copy has to live for the whole loop: the body reads it in every
        // iteration, and the pool would recycle it after the first one (the
        // body's own "last use" of it is inside one iteration).
        dst.state_[dst_slot].last_use = 1 << 30;
        return;
    }
    std::vector<uint8_t> bytes;
    if (!st.host.empty()) {
        bytes = st.host;
    } else if (state_[src_slot].dev) {
        bytes.resize(want);
        q_download(q_, state_[src_slot].dev, bytes.data(), bytes.size());
    }
    bytes.resize(want, 0);
    dt.host = bytes;
    dt.on_host = true;
    dst.state_[dst_slot].is_const = false;
    dst.state_[dst_slot].bytes = bytes.size();
    dst.state_[dst_slot].last_use = 1 << 30;
    // The host bytes are the value; any device copy the slot had is stale.
    ++dst.state_[dst_slot].host_gen;
    dst.state_[dst_slot].dev_gen = dst.state_[dst_slot].host_gen - 1;
    if (dst.state_[dst_slot].dev) {
        dst.release(dst.state_[dst_slot].dev, dst.state_[dst_slot].bytes);
        dst.state_[dst_slot].dev = nullptr;
    }
}

executor::~executor() {
    // An async upload may still be reading the pinned stage, and a device
    // buffer may still be in use: wait once before the memory goes back.
    if (uploads_pending_) {
        try {
            q_.wait();
        } catch (...) {
        }
        uploads_pending_ = false;
    }
    for (auto & kv : free_) {
        for (void * p : kv.second) {
            if (p) sycl::free(p, q_);
        }
    }
    for (void * p : free_later_) {
        if (p) sycl::free(p, q_);
    }
    for (auto & s : state_) {
        if (s.dev) sycl::free(s.dev, q_);
    }
    if (ws_) sycl::free(ws_, q_);
    if (ws2_) sycl::free(ws2_, q_);
    if (pv_ws_) sycl::free(pv_ws_, q_);
    if (up_stage_) sycl::free(up_stage_, q_);
}

size_t executor::live_bytes_calc() const {
    // The live bytes are *derived* from the live set, not counted: an
    // incremental count and the pool can disagree (a buffer released twice,
    // a release whose size was not the real one), and the report then says
    // the card holds gigabytes that it does not.
    size_t n = 0;
    for (void * p : live_) {
        auto it = alloc_size_.find(p);
        if (it != alloc_size_.end()) n += it->second;
    }
    // A Loop body is a separate executor with its own live set.  Its buffers
    // are just as much this graph's device memory, and `pool_debug` /
    // syi_onnx_info must report them; otherwise the memory report says a
    // formula run holds nothing while its body holds gigabytes.
    for (const auto & b : loop_bodies_) {
        if (b.second) n += b.second->live_bytes_calc();
    }
    return n;
}

size_t executor::pooled_bytes_calc() const {
    size_t n = 0;
    for (const auto & kv : free_) n += kv.first * kv.second.size();
    // The formula recogniser's Loop body is where the free pool actually
    // grows: one entry per distinct tensor size in a 1 700-operator body.
    // The size-based trim decision must see it, or a caller that only looks
    // at this graph's own (small) pool never trims the body at all - which
    // is exactly how the 12-document run reached OUT_OF_DEVICE_MEMORY while
    // `pooled_bytes()` reported a few megabytes.
    for (const auto & b : loop_bodies_) {
        if (b.second) n += b.second->pooled_bytes_calc();
    }
    return n;
}

size_t executor::free_mem_bytes() const {
    // The device does not report its free memory through the standard
    // queries, so the pool's own accounting is used: the live allocations
    // plus the ones the other models hold are not visible here, but the
    // number still says whether *this* graph is the one that ran out.
    size_t live = 0;
    for (const auto & kv : alloc_size_) live += kv.second;
    for (const auto & b : loop_bodies_) {
        if (!b.second) continue;
        for (const auto & kv : b.second->alloc_size_) live += kv.second;
    }
    size_t total = 0;
    try {
        total = q_.get_device().get_info<sycl::info::device::global_mem_size>();
    } catch (...) {
        return 0;
    }
    return total > live ? total - live : 0;
}

void * executor::acquire(size_t bytes) {
    if (bytes == 0) return nullptr;
    // **Round the allocation up.**  A Loop whose KV cache grows by one token
    // asks for a slightly larger buffer every iteration; at exact sizes the
    // pool's free list accumulates one buffer per step (12 GB over a
    // 2 560-step formula decode), and every whole-pool trim then frees
    // thousands of buffers.  Two coarse buckets (4 KiB below 64 KiB, 256 KiB
    // above) make consecutive steps reuse the same buffer; the fragmentation
    // is bounded by one bucket per *live* value.
    size_t alloc_bytes = bytes;
    if (bytes <= (size_t)65536) {
        alloc_bytes = (bytes + 4095) & ~(size_t)4095;
    } else {
        alloc_bytes = (bytes + 262143) & ~(size_t)262143;
    }
    if (alloc_bytes < bytes) alloc_bytes = bytes;
    // The best fit: the smallest free buffer that is big enough.
    for (auto it = free_.begin(); it != free_.end(); ++it) {
        if (it->first >= bytes && !it->second.empty()) {
            void * p = it->second.back();
            it->second.pop_back();
            pooled_now_ = pooled_now_ > it->first ? pooled_now_ - it->first : 0;
            // The buffer is live again.  (The old code only checked that it
            // was not live and never re-inserted it, so a reused buffer was
            // in neither set and the service's own memory report undercounted
            // every value that had gone through the pool once.)
            if (!live_.insert(p).second) {
                throw std::runtime_error("the memory pool handed out a live buffer twice");
            }
            live_now_ += it->first;
            if (live_now_ > peak_live_) peak_live_ = live_now_;
            // The buffer's *real* size is the key it is filed under, and the
            // record has to survive the round trip: the pool hands out the
            // smallest buffer that fits, so a buffer allocated for 8 MB can
            // be released as 1 MB, and keying the pool by the requested size
            // would then hand a 1 MB buffer to an operator that asked for
            // 4 MB - which writes past the end of it, and takes the device
            // with it.
            alloc_size_[p] = it->first;
            return p;
        }
    }
    void * p = sycl::malloc_device(alloc_bytes, q_);
    if (!p) {
        // The card refused an allocation.  The pooled buffers are free by
        // construction (nothing holds them), so dropping them - including
        // the ones a Loop body parked in its own pool - is always safe and
        // is the difference between a 500 and a slow iteration.  Wait first:
        // the queue is in order and a released buffer may still be read by
        // an already-submitted kernel.
        trim_pool();
        p = sycl::malloc_device(alloc_bytes, q_);
        if (p) {
            fprintf(stderr,
                    "[onnx] device allocation of %zu bytes failed; trimmed the pool (including "
                    "Loop bodies) and retried\n", bytes);
        } else {
            fprintf(stderr,
                    "[onnx] device allocation of %zu bytes failed after trimming: live=%zu "
                    "pooled=%zu free=%zu\n",
                    bytes, live_bytes_calc(), pooled_bytes_calc(), free_mem_bytes());
        }
    }
    if (p) {
        ++malloc_count_;
        alloc_size_[p] = alloc_bytes;
        if (!live_.insert(p).second) {
            throw std::runtime_error("the memory pool handed out a live buffer twice");
        }
        live_now_ += alloc_bytes;
        if (live_now_ > peak_live_) peak_live_ = live_now_;
    }
    return p;
}

void executor::trim_pool() {
    q_.wait();
    // A Loop body is a separate executor with its own pool.  The formula
    // recogniser's body runs hundreds of iterations; trimming only the outer
    // graph leaves that pool alive and it eventually exhausts the card.  Trim
    // nested bodies first, then this graph's own free list.
    for (auto & body : loop_bodies_) {
        if (body.second) body.second->trim_pool();
    }
    for (auto & kv : free_) {
        for (void * p : kv.second) {
            auto it = alloc_size_.find(p);
            if (it != alloc_size_.end()) {
                sycl::free(p, q_);
                alloc_size_.erase(it);
            }
        }
    }
    free_.clear();
    for (void * p : free_later_) {
        auto it = alloc_size_.find(p);
        if (it != alloc_size_.end()) {
            sycl::free(p, q_);
            alloc_size_.erase(it);
        }
    }
    free_later_.clear();
    pooled_now_ = 0;
}

void executor::release(void * p, size_t bytes) {
    (void)bytes;
    if (!p) return;
    if (loop_dbg_) {
        fprintf(stderr, "[onnx] the release: %p (%zu bytes)\n", p, bytes);
    }
    // The buffer goes back under its *real* size, not the size this caller
    // asked for.  The record stays: the buffer is handed out again later,
    // and the next release has to find the same real size.
    const size_t real = alloc_size_.count(p) ? alloc_size_[p] : bytes;
    // It must not be in the pool already.  A double release hands the same
    // buffer to two different values, and the second one overwrites the
    // first - which is how a constant's value became another constant's.
    //
    // `live_` is exactly the set of allocated, not-yet-released buffers
    // (`acquire` and the constant path insert into it), so one erase decides
    // it.  The old check scanned every free buffer, and a body iteration
    // releases hundreds of intermediates: the scan was a measurable part of
    // the loop's write-back.
    if (!live_.erase(p)) {
        throw std::runtime_error("a device buffer was released twice");
    }
    live_now_ = live_now_ > alloc_size_[p] ? live_now_ - alloc_size_[p] : 0;
    if (pooled_now_ + real > pool_cap_) {
        // Over the cap.  Do not grow the free list: the buffer is queued for
        // the run's end, where the queue is already drained.
        free_later_.push_back(p);
    } else {
        free_[real].push_back(p);
        pooled_now_ += real;
    }
}

// The scratch buffer for the operators' small arrays.
void * executor::scratch() {
    if (!ws_) {
        // 4 MiB: the top-k's index output (720 KB at 300 rows and k=300)
        // and the gather's small index array live here.  The top-k's
        // histogram and running prefix are local memory now.
        ws_bytes_ = 4 << 20;
        ws_ = sycl::malloc_device(ws_bytes_, q_);
    }
    return ws_;
}

void * executor::scratch_small() {
    if (!ws2_) {
        // 64 KB: the operators that convert a value and put it on the
        // device (a mixed-type elementwise) use this.
        ws2_ = sycl::malloc_device(ws2_bytes_ = 64 << 10, q_);
    }
    return ws2_;
}

void * executor::pv_scratch(size_t bytes) {
    if (bytes == 0) return nullptr;
    if (pv_ws_ && pv_ws_bytes_ >= bytes) return pv_ws_;
    // Growing replaces the buffer, so the queue has to be drained first:
    // a PV reduction may still be reading the old partials.
    if (pv_ws_) {
        q_.wait();
        sycl::free(pv_ws_, q_);
        pv_ws_ = nullptr;
        pv_ws_bytes_ = 0;
    }
    size_t cap = std::max<size_t>(bytes, 1u << 20);
    cap = (cap + 4095) & ~(size_t)4095;
    pv_ws_ = sycl::malloc_device(cap, q_);
    if (!pv_ws_) return nullptr;
    pv_ws_bytes_ = cap;
    return pv_ws_;
}

bool executor::is_output(int slot) const {
    for (int s : g_.output_slots) {
        if (s == slot) return true;
    }
    return false;
}

void executor::skip_outputs(const std::vector<std::string> & names) {
    skip_out_.clear();
    for (const std::string & n : names) {
        bool found = false;
        for (int s : g_.output_slots) {
            if (g_.tensors[s].name == n) {
                skip_out_.insert(s);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("the graph has no output called '" + n +
                                     "' (the model config's drop_outputs)");
        }
    }
    // The *computation* that only feeds a dropped output is dead: the layout
    // model's mask head is 300x200x200 of output and a whole decoder branch,
    // and nothing a kept output needs flows through it.  Mark the operators
    // that reach a kept output, walking backwards; `run()` skips the rest.
    // (The graph's operators are topologically ordered, so one reverse pass
    // is enough.)
    live_ops_.assign(g_.ops.size(), false);
    std::vector<bool> live_slot(g_.tensors.size(), false);
    for (int s : g_.output_slots) {
        if (!skip_out_.count(s)) live_slot[s] = true;
    }
    for (size_t i = g_.ops.size(); i-- > 0;) {
        const exec_op & op = g_.ops[i];
        bool needed = false;
        for (int s : op.out) {
            if (s >= 0 && s < (int)live_slot.size() && live_slot[s]) {
                needed = true;
                break;
            }
        }
        if (!needed) continue;
        live_ops_[i] = true;
        for (int s : op.in) {
            if (s >= 0 && s < (int)live_slot.size()) live_slot[s] = true;
        }
        // A Loop body reads the enclosing graph's values, and no operator of
        // this graph does.  A body that is built (the constructor builds
        // them) publishes those reads as its outer_refs.
        if (op.node->op_type == "Loop") {
            auto it = loop_bodies_.find(i);
            if (it != loop_bodies_.end()) {
                for (const auto & r : it->second->graph().outer_refs) {
                    if (r.second >= 0 && r.second < (int)live_slot.size()) live_slot[r.second] = true;
                }
            }
        }
    }
}

bool executor::output_skipped(size_t i) const {
    return i < g_.output_slots.size() && skip_out_.count(g_.output_slots[i]) != 0;
}

void executor::set_output_reduce(const std::string & name, int kind) {
    if (kind == kReduceNone) {
        out_reduce_.erase(name);
        return;
    }
    bool found = false;
    for (int s : g_.output_slots) {
        if (g_.tensors[s].name == name) {
            found = true;
            break;
        }
    }
    if (!found) {
        throw std::runtime_error("the graph has no output called '" + name +
                                 "' (the output reduction)");
    }
    out_reduce_[name] = kind;
}

// The integer values and the operators that can consume them on the device.
//
// An integer used to be host-authoritative by construction: every integer
// output was downloaded at its write-back, because the shape functions read
// their parameters there.  That is true for a *parameter* (a reshape target,
// the slice bounds, the split sizes) and false for *data* (the TopK indices
// and the index arithmetic built from them) - and the download waited for
// the whole in-order queue, so a 38 KB index array blocked on every
// convolution submitted before it.
//
// The rule now: an integer output stays on the device when the operator
// wrote it there, and the two lists below decide where a consumer reads it.
namespace {

bool int_dtype(int32_t d) { return d == onnxpb::INT64 || d == onnxpb::INT32; }

// An operator with a device path for an integer tensor.
bool int_dev_capable(const std::string & ty) {
    return ty == "Concat" || ty == "Slice" || ty == "Reshape" || ty == "Flatten" ||
           ty == "Squeeze" || ty == "Unsqueeze" || ty == "Identity" || ty == "Transpose" ||
           ty == "GatherND" || ty == "ScatterND" || ty == "Tile" || ty == "Expand" ||
           ty == "Cast" || ty == "Mod" || ty == "Max" || ty == "Min" || ty == "Add" ||
           ty == "Sub" || ty == "Mul" || ty == "ReduceMax" || ty == "ReduceMin" ||
           ty == "ReduceSum" || ty == "ReduceMean" || ty == "CumSum";
}

// An operator that reads the *values* of this input before it runs (its
// shape function does).  These are the parameters, and they are small - a
// reshape target, the bounds of a Slice, the repeats of a Tile, a TopK's k.
bool int_shape_param(const std::string & ty, size_t pos) {
    if (ty == "Reshape" || ty == "Expand" || ty == "Tile" || ty == "Split" || ty == "Squeeze" ||
        ty == "Unsqueeze" || ty == "TopK" || ty == "OneHot" || ty == "Shape") {
        return pos == 1;
    }
    if (ty == "Slice") return pos >= 1 && pos <= 4;
    if (ty == "ReduceMax" || ty == "ReduceMin" || ty == "ReduceSum" || ty == "ReduceMean" ||
        ty == "CumSum") {
        return pos == 1;
    }
    if (ty == "Range") return pos <= 2;
    if (ty == "ConstantOfShape") return pos == 0;
    if (ty == "Pad") return pos == 1;
    if (ty == "Resize") return pos >= 1 && pos <= 3;
    return false;
}

// An operator whose execution reads only the *shape* of an integer input (a
// Shape never reads its input's values, and downloading a whole index tensor
// for it was 38 KB and a full queue drain).
bool int_shape_only(const std::string & ty) { return ty == "Shape"; }

// The integer operators that propagate a *shape* without turning it into
// data.  A tensor that is derived from a `Shape` through only these stays
// shape arithmetic; anything else (a token id, a TopK index, a matmul) is
// data, and the host must not pull it across the bus to do a few scalar
// ops on it.
bool shape_arith_op(const std::string & ty) {
    return ty == "Shape" || ty == "Cast" || ty == "Concat" || ty == "Slice" ||
           ty == "Gather" || ty == "Reshape" || ty == "Flatten" || ty == "Squeeze" ||
           ty == "Unsqueeze" || ty == "Identity" || ty == "Transpose" || ty == "Mul" ||
           ty == "Add" || ty == "Sub" || ty == "Div" || ty == "Mod" || ty == "Max" ||
           ty == "Min" || ty == "Equal" || ty == "Greater" || ty == "Less" ||
           ty == "GreaterOrEqual" || ty == "LessOrEqual" || ty == "And" || ty == "Or" ||
           ty == "Not" || ty == "Where" || ty == "Expand" || ty == "Tile" || ty == "Range" ||
           ty == "ReduceMax" || ty == "ReduceMin" || ty == "ReduceSum" || ty == "ReduceMean" ||
           ty == "CumSum" || ty == "Split" || ty == "ConstantOfShape";
}

// The elementwise operators exec_elementwise can evaluate on the host
// (`ew_host_ok` in ops_exec.cpp).  Kept here as a list because the decision
// to skip the input uploads has to be made before the operator runs; see
// the call site.  Keep the two lists in sync.
bool host_elementwise_op(const std::string & ty) {
    return ty == "Add" || ty == "Sub" || ty == "Mul" || ty == "Div" || ty == "Pow" ||
           ty == "Neg" || ty == "Abs" || ty == "Floor" || ty == "Ceil" || ty == "Sign" ||
           ty == "Greater" || ty == "Less" || ty == "Equal" || ty == "GreaterOrEqual" ||
           ty == "LessOrEqual" || ty == "And" || ty == "Or" || ty == "Not" || ty == "Max" ||
           ty == "Min" || ty == "Mod";
}

} // namespace

void executor::materialize_int(int slot) {
    if (slot < 0 || slot >= (int)state_.size()) return;
    slot_state & st = state_[slot];
    tensor & t = g_.tensors[slot];
    if (!t.host.empty()) return;
    if (!st.dev || !st.bytes) return;
    if (intdev_dbg_) {
        fprintf(stderr, "[onnx] intdev: '%s' comes back to the host (%zu bytes, the dtype %d)\n",
                t.name.c_str(), st.bytes, (int)t.dtype);
    }
    t.host.resize(st.bytes);
    q_download(q_, st.dev, t.host.data(), t.host.size());
    t.on_host = true;
    // The device buffer is still the same value, so its mirror generation
    // stays current: a consumer that reads it on the device must not
    // re-upload it.
    st.dev_gen = st.host_gen;
}

size_t executor::reduced_count(size_t out_index) const {
    if (out_index >= reduced_.size()) return 0;
    return reduced_[out_index].size();
}

const executor::reduced_output & executor::reduced(size_t out_index, size_t k) const {
    static const reduced_output empty;
    if (out_index >= reduced_.size() || k >= reduced_[out_index].size()) return empty;
    return reduced_[out_index][k];
}

const float * executor::device_value(int slot) {
    slot_state & s = state_[slot];
    // A value with no host bytes is device-authoritative: the device buffer
    // *is* the value, whatever the generations say (they only track a
    // mirror of host bytes).
    if (s.dev && (s.dev_gen == s.host_gen || host_of(slot).empty())) {
        return (const float *)s.dev;
    }
    std::vector<uint8_t> & h = host_of(slot);
    if (h.empty()) {
        throw std::runtime_error("value " + g_.tensors[slot].name + " is empty");
    }
    if (s.dev && alloc_size_.count(s.dev) && alloc_size_[s.dev] >= h.size()) {
        q_upload(q_, h.data(), s.dev, h.size());
        s.dev_gen = s.host_gen;
        return (const float *)s.dev;
    }
    if (s.dev) {
        release(s.dev, s.bytes);
        s.dev = nullptr;
    }
    s.dev = acquire(h.size());
    if (!s.dev) throw std::runtime_error("device allocation failed");
    q_upload(q_, h.data(), s.dev, h.size());
    s.dev_gen = s.host_gen;
    return (const float *)s.dev;
}

// Put the inputs on the device.
//
// The device copy is a *mirror* of the host value, and it is kept while it
// is current: `host_gen` counts the writes of the host bytes and `dev_gen`
// is the write the mirror was made from.  The mirror used to be dropped at
// every operator (the write-back released a host-resident output's buffer),
// so each consumer uploaded the value again and every constant was uploaded
// once per call - 3508 copies of 707 MB on one layout call.  For a constant
// the mirror survives the call as well: it never changes.
void executor::prepare_inputs(exec_op & op) {
    // The timers belong to SI_ONNX_UPLOADDBG: an unconditional clock pair
    // per allocation and per upload is real time on a model that does
    // thousands of small uploads (the table decoder's Loop body).
    using tp = std::chrono::steady_clock::time_point;
    const auto tp0 = upload_dbg_ ? std::chrono::steady_clock::now() : tp{};
    size_t copies = 0;
    double t_alloc = 0.0, t_copy = 0.0;
    size_t copy_bytes = 0;
    for (size_t pos = 0; pos < op.in.size(); ++pos) {
        const int s = op.in[pos];
        if (s < 0 || s >= (int)state_.size()) continue;
        // The *parameters* a shape function reads are host-only: the device
        // kernel gets the derived geometry (strides, axis, counts), never the
        // parameter tensor.  Uploading them was one copy per operator for
        // every reshape target (the formula recogniser rebuilds 134 of them
        // per iteration), and nothing on the device read a byte of it.
        if (int_shape_param(op.node->op_type, pos)) continue;
        slot_state & st = state_[s];
        std::vector<uint8_t> & h = host_of(s);
        if (h.empty()) {
            // A device-authoritative value (a float an operator wrote on the
            // device) has no host copy: the device buffer *is* the value.
            if (st.dev) continue;
            // An empty constant (a zero-element tensor) is legal: the
            // operator has to know that it is empty.
            if (g_.tensors[s].count == 0) continue;
            throw std::runtime_error("the input '" + g_.tensors[s].name + "' has no value (the slot " +
                                     std::to_string(s) + ", the dtype " +
                                     std::to_string((int)g_.tensors[s].dtype) + ", the count " +
                                     std::to_string(g_.tensors[s].count) + ", the const " +
                                     std::to_string(st.is_const) + ", the dev " +
                                     std::to_string(st.dev != nullptr) + ")");
        }
        // The mirror is current: nothing to do.  This is the whole point of
        // the generations - a constant is uploaded once for the process, and
        // an integer that ten operators read is uploaded once per call.
        if (st.dev && st.dev_gen == st.host_gen) {
            // The invariant the generations encode is checked, not assumed:
            // a missed `host_gen` bump would make a consumer read the
            // previous value, and the model would still produce plausible
            // output.  Slow (a device wait per input), so it is opt-in.
            if (mirror_chk_) {
                auto it = alloc_size_.find(st.dev);
                if (it == alloc_size_.end() || it->second < h.size()) {
                    throw std::runtime_error("the mirror of '" + g_.tensors[s].name +
                                             "' is smaller than its host value (the slot " +
                                             std::to_string(s) + ")");
                }
                std::vector<uint8_t> now(h.size());
                if (!now.empty()) {
                    q_download(q_, st.dev, now.data(), now.size());
                }
                if (now != h) {
                    throw std::runtime_error("the mirror of '" + g_.tensors[s].name +
                                             "' is stale (the slot " + std::to_string(s) +
                                             ", the host generation " + std::to_string(st.host_gen) +
                                             ", the device generation " + std::to_string(st.dev_gen) +
                                             ")");
                }
            }
            continue;
        }
        // A buffer that is too small (a loop-carried value changed shape) is
        // replaced, never overrun.
        auto it = st.dev ? alloc_size_.find(st.dev) : alloc_size_.end();
        if (st.dev && (it == alloc_size_.end() || it->second < h.size())) {
            release(st.dev, st.bytes);
            st.dev = nullptr;
        }
        if (!st.dev) {
            const auto a0 = upload_dbg_ ? std::chrono::steady_clock::now() : tp{};
            if (st.is_const) {
                // A constant stays on the device for the process.  Taking it
                // from the pool means the best-fit can hand it a buffer far
                // larger than the value (a 4 KB bias was holding a 48 MB
                // block), and because a constant is never released that
                // block is lost to the activations for ever - 1.9 GB of the
                // layout model's steady footprint was exactly this.
                st.dev = sycl::malloc_device(h.size(), q_);
                if (!st.dev) {
                    trim_pool();
                    st.dev = sycl::malloc_device(h.size(), q_);
                }
                if (st.dev) {
                    alloc_size_[st.dev] = h.size();
                    if (!live_.insert(st.dev).second) {
                        throw std::runtime_error("the pool handed out a live buffer twice");
                    }
                    live_now_ += h.size();
                    if (live_now_ > peak_live_) peak_live_ = live_now_;
                }
            } else {
                st.dev = acquire(h.size());
            }
            if (upload_dbg_) {
                t_alloc += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - a0).count();
            }
            if (!st.dev) throw std::runtime_error("device allocation failed");
        }
        // The copy is *not* waited for.  The queue is in order, so the
        // operator's kernels see the bytes; the wait used to drain the queue
        // at every upload, which splits the pipeline into as many segments as
        // there are values - the layout model uploads 375 values per call and
        // spent 169 of its 329 ms here, on 7 MB.  The host bytes stay alive
        // and unmodified until the next run (a value is written once), and
        // `run()` waits for the previous run's copies before it overwrites
        // an input's host bytes.
        const auto c0 = upload_dbg_ ? std::chrono::steady_clock::now() : tp{};
        // A copy out of *pageable* host memory is synchronous on this backend:
        // it blocks until the queue drains, which is what made a 0-byte
        // upload take 82 ms in the middle of the layout model (the queue was
        // 82 ms deep).  Staging the bytes through pinned USM host memory
        // makes the copy asynchronous - the same bytes, just from memory the
        // device can read without a synchronous wrapper.
        bool staged = false;
        if (h.size() <= kUpStageBytes) {
            if (!up_stage_) {
                up_stage_cap_ = kUpStageBytes;
                up_stage_ = sycl::malloc_host(up_stage_cap_, q_);
            }
            if (up_stage_) {
                const size_t need = (h.size() + 63) & ~(size_t)63;
                if (up_stage_off_ + need > up_stage_cap_) {
                    // The staged copies still read the buffer: drain them
                    // before it is reused.  (Rare: the cap covers a whole
                    // call's uploads for these models.)
                    q_.wait();
                    up_stage_off_ = 0;
                }
                std::memcpy((uint8_t *)up_stage_ + up_stage_off_, h.data(), h.size());
                q_.memcpy(st.dev, (const uint8_t *)up_stage_ + up_stage_off_, h.size());
                up_stage_off_ += need;
                staged = true;
            }
        }
        if (!staged) q_.memcpy(st.dev, h.data(), h.size());
        if (upload_dbg_) {
            t_copy += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - c0).count();
        }
        copy_bytes += h.size();
        uploads_pending_ = true;
        st.dev_gen = st.host_gen;
        ++upload_count_;
        upload_bytes_ += h.size();
        ++copies;
        if (upload_dbg_ && s < (int)upload_by_slot_.size()) {
            ++upload_by_slot_[s];
            upload_bytes_by_slot_[s] += h.size();
        }
    }
    if (upload_dbg_) {
        const double dt = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - tp0).count();
        if (dt > 2.0) {
            fprintf(stderr,
                    "[onnx] prepare: %s %.1f ms (%zu inputs, %zu uploads, %.1f MB; the alloc "
                    "%.1f, the copy %.1f)\n",
                    op.node->op_type.c_str(), dt, op.in.size(), copies, copy_bytes / 1048576.0,
                    t_alloc, t_copy);
        }
    }
}

// The convolution weights that a Constant node produced are transposed after
// the constants are known.
void executor::prepare_conv_weights() {
    for (exec_op & op : g_.ops) {
        if (op.kind != op_kind::conv || op.weight_transposed) continue;
        if (op.in.size() < 2) continue;
        const int wi = op.in[1];
        if (wi < 0) continue;
        int64_t grp = 1;
        for (const onnxpb::attribute & a : op.node->attribute) {
            if (a.name == "group") grp = a.i;
        }
        if (host_of(wi).empty()) continue;
        tensor src = g_.tensors[wi];
        src.on_host = true;
        const tensor t = op.node->op_type == "Conv" ? transpose_conv_weight(src, grp)
                                                   : transpose_conv_transpose_weight(src, grp);
        g_.tensors[wi].host = t.host;
        g_.tensors[wi].shape = t.shape;
        g_.tensors[wi].count = t.count;
        g_.tensors[wi].on_host = true;
        // The weight's host bytes were replaced: a device mirror made from
        // the old ones (the folded path can have uploaded it) is stale.
        ++state_[wi].host_gen;
        state_[wi].dev_gen = state_[wi].host_gen - 1;
        op.weight_transposed = true;
    }
}

// The Loop: the body is a graph and it runs once per iteration.
//
// The loop-carried values and the scan outputs go through the host: there
// are few of them, and the shape of a scan output is only known once the
// loop has run (the number of the iterations is data).
//
// A carried value keeps its *type*: the structure recogniser's loop carries
// int32 and int64 values next to the float ones, and reading them all as
// float32 halves the int64 ones (8 bytes per element read as 4) and turns
// the int32 ones into garbage.
executor * executor::loop_body(size_t op_index) {
    auto it = loop_bodies_.find(op_index);
    if (it != loop_bodies_.end()) return it->second.get();
    const exec_op & op = g_.ops[op_index];
    const onnxpb::graph_proto * body = nullptr;
    for (const onnxpb::attribute & a : op.node->attribute) {
        if (a.name == "body" && a.g) body = a.g.get();
    }
    if (!body) throw std::runtime_error("the Loop has no body");
    // The body is built with *this* graph as its scope: ONNX lets a subgraph
    // read the values of the graph that holds it, and the formula
    // recogniser's Loop body reads a transpose that this graph computes.
    exec_graph bg = build_graph_from_graph(*body, g_.opset, "", false, &g_);
    auto owned = std::make_unique<executor>(std::move(bg), q_);
    owned->body_ = true;
    // The body's dump files are tagged: a value of the enclosing graph can
    // have the same name as one of the body's (an outer-scope read), and the
    // two dumps would write the same file.
    owned->set_dump_tag("body_");
    executor * p = owned.get();
    loop_bodies_[op_index] = std::move(owned);
    return p;
}

void executor::run_loop(size_t op_index, exec_op & op) {
    // The body.  It is built the first time the loop runs and kept: a
    // rebuild folds the body's constant sub-graphs and transposes its
    // convolution weights, which is the work of a whole model.
    executor * bexec = loop_body(op_index);
    const exec_graph & bg = bexec->graph();
    if (loop_dbg_) {
        fprintf(stderr, "[onnx] the loop: the body is built (%zu ops, %zu tensors)\n", bg.ops.size(),
                bg.tensors.size());
    }

    // The inputs: the maximum number of the iterations, the condition and
    // the loop-carried values.
    const std::vector<int> & in = op.in;
    if (in.size() < 2) throw std::runtime_error("the Loop has too few inputs");
    int64_t M = -1;
    {
        // The trip count: the first input.  **It is optional.**  ONNX says
        // an empty trip count means "no limit": the loop runs until the
        // condition is false.  (The formula recogniser's loop is that one,
        // and the slot is -1 - an omitted input - so reading it as a tensor
        // reads whatever is next to it in memory.)
        const int ms = in[0];
        if (ms >= 0) {
            if (ms >= (int)g_.tensors.size()) {
                throw std::runtime_error("the Loop's trip count is the slot " + std::to_string(ms) +
                                         ", which is not a tensor (there are " +
                                         std::to_string(g_.tensors.size()) + ")");
            }
            const std::vector<int64_t> v = read_i64(&g_.tensors[ms]);
            if (!v.empty()) M = v[0];
        }
    }
    bool cond = false;
    {
        // The condition is a value like any other: it is on the device (it is
        // a float 0/1, and the write-back keeps the floats there), so the host
        // copy is empty and it has to be read back.
        if (in[1] < 0 || in[1] >= (int)g_.tensors.size()) {
            throw std::runtime_error("the Loop's condition is the slot " + std::to_string(in[1]) +
                                     ", which is not a tensor (there are " +
                                     std::to_string(g_.tensors.size()) + ")");
        }
        if (state_[in[1]].bytes > (size_t)1 << 24) {
            throw std::runtime_error("the loop's condition ('" + g_.tensors[in[1]].name + "') holds " +
                                     std::to_string(state_[in[1]].bytes) + " bytes (the count " +
                                     std::to_string(g_.tensors[in[1]].count) + ", the dtype " +
                                     std::to_string((int)g_.tensors[in[1]].dtype) + ")");
        }
        std::vector<uint8_t> h = host_of(in[1]);
        if (h.empty() && state_[in[1]].dev) {
            const size_t bytes = state_[in[1]].bytes;
            h.resize(bytes);
            q_download(q_, state_[in[1]].dev, h.data(), bytes);
        }
        cond = value_is_true(h, g_.tensors[in[1]].dtype);
        if (loop_dbg_) {
            fprintf(stderr, "[onnx] the loop: the condition '%s': %zu bytes, the dtype %d, the value %d\n",
                    g_.tensors[in[1]].name.c_str(), h.size(), (int)g_.tensors[in[1]].dtype, (int)cond);
        }
    }
    // The loop-carried values: the data, the shape and the type.
    struct carried_value {
        std::vector<uint8_t> data;
        std::vector<int64_t> shape;
        int32_t dtype = onnxpb::FLOAT;
    };
    std::vector<carried_value> carried;
    for (size_t i = 2; i < in.size(); ++i) {
        carried_value cv;
        if (state_[in[i]].bytes > (size_t)1 << 24) {
            throw std::runtime_error("the loop's carried value " + std::to_string(i - 2) + " ('" +
                                     g_.tensors[in[i]].name + "') holds " +
                                     std::to_string(state_[in[i]].bytes) + " bytes (the count " +
                                     std::to_string(g_.tensors[in[i]].count) + ", the dtype " +
                                     std::to_string((int)g_.tensors[in[i]].dtype) + ")");
        }
        cv.data = host_of(in[i]);
        if (cv.data.empty() && state_[in[i]].dev) {
            // A float value lives on the device: the write-back clears the
            // host copy of everything that is not an integer, and the loop's
            // body needs the bytes.  (An empty feed is a body that computes
            // nothing: the wireless table structurer's decoder came out as
            // a scalar.)
            const size_t bytes = state_[in[i]].bytes;
            cv.data.resize(bytes);
            q_download(q_, state_[in[i]].dev, cv.data.data(), bytes);
        }
        cv.shape = g_.tensors[in[i]].shape;
        cv.dtype = g_.tensors[in[i]].dtype;
        carried.push_back(std::move(cv));
    }
    // The body's input names: the iteration number, the condition, and the
    // carried values.
    std::vector<std::string> body_in;
    for (int s : bg.input_slots) body_in.push_back(bg.tensors[s].name);
    if (body_in.size() < 2 + carried.size()) {
        throw std::runtime_error("the Loop's body does not take the carried values");
    }
    // The scans: the body's outputs after the condition and the carried
    // values.  Each one is a list of the iterations' values, and the shape
    // of an iteration comes from the body.
    std::vector<std::vector<std::vector<uint8_t>>> scans;
    std::vector<std::vector<int64_t>> scan_shape;

    // Which carried values the body actually *produces*, and which it only
    // passes through.
    //
    // A body output that no node writes is the body's input of the same name
    // returned unchanged (ONNX's pass-through).  Most of the table
    // structurer's 30 carried values are its decoder's weights: the body
    // reports them so they can be carried, but they never change.  Feeding
    // them again every iteration re-uploads 2.3 MB of weights and empties
    // their host copy on the read-back, which is why the "unchanged feed"
    // check in `run()` could never fire - the value it compares against had
    // been moved out.  A pass-through is therefore neither re-fed nor
    // downloaded: the body's input slot keeps the value (it is a graph
    // output, so the pool does not release it).
    std::vector<char> carried_produced(carried.size(), 1);
    for (size_t i = 0; i < carried.size() && i + 1 < bg.output_produced.size(); ++i) {
        carried_produced[i] = bg.output_produced[i + 1] ? 1 : 0;
    }
    // A produced carried value whose body output lives on the device is
    // chained to the next iteration by a device-to-device copy into the
    // body's input slot: the value never crosses the bus.  (The host route
    // reads it back and re-uploads it, which at a batch of 31 is 3.7 MB in
    // each direction, per iteration.)  `via_dev` says which carried values
    // take that route; a value whose output has no device buffer falls back
    // to the host route and is requested/read back as before.
    std::vector<char> via_dev(carried.size(), 0);
    // One dedicated device buffer per chained carried value.  It is *not*
    // from the executor's pool: the pool hands a released buffer straight
    // back out, and the copy into it is queued, so a pooled buffer can be
    // given to another value before the copy runs.
    std::vector<void *> chain_buf(carried.size(), nullptr);
    // Every body *input* is read on every iteration: the iteration number,
    // the condition, the carried values and the outer-scope copies.  The
    // pool would hand their buffers back after their last use inside
    // iteration 0 (a body input slot is a different tensor from the body
    // output that names it), so the value would be re-uploaded each time -
    // and the in-place borrow (which is enabled for a body, see
    // `build_graph_from_graph`) would take the buffer of a value the next
    // iteration still reads.  Pin all of them for the body's lifetime.
    for (int is : bg.input_slots) {
        if (is >= 0 && is < (int)bexec->state_.size()) bexec->state_[is].pinned = true;
    }
    // The body outputs the loop needs back after one iteration: the
    // condition, the carried values still on the host route, and the scans.
    // The pass-throughs are left in their slots, which is also what keeps
    // their host bytes for the next iteration's comparison.
    std::vector<size_t> want_out;
    auto build_want = [&]() {
        want_out.clear();
        want_out.push_back(0);
        for (size_t i = 0; i < carried.size(); ++i) {
            if (carried_produced[i] && !via_dev[i]) want_out.push_back(i + 1);
        }
        for (size_t i = 1 + carried.size(); i < bg.output_slots.size(); ++i) want_out.push_back(i);
    };
    build_want();
    // Chain one produced carried value from its body output to the body
    // input, on the device.  Returns false when the value has to take the
    // host route (no output buffer, or an input buffer that cannot be
    // allocated).  `PF_ONNX_LOOP_NODEV=1` restores the host route, for the
    // A/B.
    static const bool chain_on = getenv("PF_ONNX_LOOP_NODEV") == nullptr;
    auto chain_device = [&](size_t i) -> bool {
        if (!chain_on) return false;
        // An *integer* carried value stays on the host route **when it is
        // shape arithmetic**: the shape functions read those on the host and
        // they are a few bytes.  An integer *data* tensor is not shape
        // arithmetic - the formula recogniser carries a 50 000-element int
        // mask - and copying 200 KB to the host and re-uploading it every
        // iteration was 3.4 ms of the loop's 5.8 ms post-processing.
        if (carried[i].dtype == onnxpb::INT32 || carried[i].dtype == onnxpb::INT64) {
            const size_t oi = i + 1;
            if (oi >= bg.output_slots.size()) return false;
            const int os = bg.output_slots[oi];
            if (os >= 0 && os < (int)shape_derived_.size() && shape_derived_[os]) return false;
        }
        const size_t oi = i + 1;
        if (oi >= bg.output_slots.size() || 2 + i >= bg.input_slots.size()) return false;
        const int os = bg.output_slots[oi];
        const int is = bg.input_slots[2 + i];
        slot_state & ss = bexec->state_[os];
        // The output's device buffer must *be* the value.  A value the body
        // computed on the host has a stale device mirror (its generation is
        // one behind), and copying that would chain the previous iteration's
        // bytes; those values keep the host route.
        if (!ss.dev || ss.bytes == 0 || ss.dev_gen != ss.host_gen) return false;
        slot_state & ds = bexec->state_[is];
        const size_t bytes = ss.bytes;
        // The buffer is recorded in the executor's size map so the
        // destructor frees it; it is never put in the pool's free list.
        void * &cb = chain_buf[i];
        // The growth is geometric, not exact.  A carried KV cache grows by
        // one token every iteration, so an exact-size chain buffer meant a
        // `sycl::free` + `sycl::malloc_device` for it on *every* iteration
        // (~58 us each; 58 values = 3.4 ms of a 20 ms loop).  Rounding the
        // allocation up to 64 KiB and then doubling turns that into a
        // handful of reallocations over 2 500 steps.
        auto chain_cap = [](size_t want) {
            size_t cap = want + want / 2 + 65536;
            return cap < want ? want : cap;
        };
        if (!cb) {
            const size_t cap = chain_cap(bytes);
            cb = sycl::malloc_device(cap, bexec->q_);
            if (!cb) return false;
            bexec->alloc_size_[cb] = cap;
            bexec->live_.insert(cb);
        } else if (bexec->alloc_size_[cb] < bytes) {
            // The carried value outgrew the capacity: replace the chain
            // buffer.  The run has waited, so nothing is still reading the
            // old one.
            if (ds.dev == cb) ds.dev = nullptr;
            bexec->live_.erase(cb);
            bexec->alloc_size_.erase(cb);
            sycl::free(cb, bexec->q_);
            const size_t cap = chain_cap(bytes);
            cb = sycl::malloc_device(cap, bexec->q_);
            if (!cb) return false;
            bexec->alloc_size_[cb] = cap;
            bexec->live_.insert(cb);
        }
        if (ds.dev != cb) {
            // The slot's old buffer (a pool buffer) goes back *after* the
            // run has waited, so nothing is still reading it.  The chain
            // buffer itself is never returned to the pool: the pool would
            // hand it to another value while a queued copy still writes it.
            if (ds.dev && bexec->alloc_size_.count(ds.dev)) bexec->release(ds.dev, ds.bytes);
            ds.dev = cb;
        }
        // Queued, not waited: the body's first kernel for the next
        // iteration is on the same in-order queue.  The copies are batched
        // (see chain_flush): one memcpy call per value was 58 us of host
        // time, and this loop chains 58 values.
        bexec->q_.memcpy(cb, ss.dev, bytes);
        ds.bytes = bytes;
        ds.dev_gen = ds.host_gen;
        // The body input is the body *output* now, and its shape has to move
        // with the bytes.  A cached value grows every iteration (the formula
        // recogniser's KV caches go [1,16,0,32] -> [1,16,n,32]); the host
        // route carries the shape in the feed, but this route feeds nothing,
        // so leaving the input tensor's shape at the initial one made every
        // Concat see an empty cache.  The values were copied but the shape
        // said zero, so the model's KV cache never accumulated and its
        // reshape target stayed at the first iteration's shape - the
        // `[1,16,1,560] -> [16,1,1]` allocation that reported
        // OUT_OF_DEVICE_MEMORY.
        const tensor & ot = bexec->g_.tensors[os];
        tensor & it = bexec->g_.tensors[is];
        it.shape = ot.shape;
        it.count = ot.count;
        it.dtype = ot.dtype;
        it.dev = ds.dev;
        it.on_host = false;
        it.host.clear();
        return true;
    };

    if (loop_dbg_) {
        fprintf(stderr, "[onnx] the loop: %zu carried values\n", carried.size());
        fprintf(stderr, "[onnx] the loop: the trip count %lld, the condition %d\n", (long long)M,
                (int)cond);
        for (size_t i = 0; i < carried.size(); ++i) {
            fprintf(stderr, "   carried[%zu]: the shape [", i);
            for (int64_t d : carried[i].shape) fprintf(stderr, " %lld", (long long)d);
            fprintf(stderr, "] the bytes %zu\n", carried[i].data.size());
        }
        fprintf(stderr, "   the body's %zu inputs:", bg.input_slots.size());
        for (int s2 : bg.input_slots) {
            fprintf(stderr, " %s[", bg.tensors[s2].name.c_str());
            for (int64_t d : bg.tensors[s2].shape) fprintf(stderr, " %lld", (long long)d);
            fprintf(stderr, "]");
        }
        fprintf(stderr, "\n   the body's %zu outputs:", bg.output_slots.size());
        for (int s2 : bg.output_slots) {
            fprintf(stderr, " %s[", bg.tensors[s2].name.c_str());
            for (int64_t d : bg.tensors[s2].shape) fprintf(stderr, " %lld", (long long)d);
            fprintf(stderr, "]");
        }
        fprintf(stderr, "\n   the body's %zu ops:", bg.ops.size());
        for (size_t i = 0; i < bg.ops.size() && i < 60; ++i) {
            fprintf(stderr, " %zu:%s", i, bg.ops[i].node->op_type.c_str());
        }
        fprintf(stderr, "\n");
    }
    // The outer scope: the values the body reads from this graph.  This
    // graph computes them before the Loop, and the body's own copy is what
    // its operators read.
    //
    // They are copied **once**, before the loop: ONNX's outer scope is
    // read-only, so the body cannot change them, and the formula recogniser
    // reads 816 of them - a copy per iteration is 816 device copies per
    // iteration, and the copy is what the body's operators wait for.
    //
    // **They have to stay alive until here.**  The executor hands a value's
    // device buffer back to the pool after the last operator that reads it,
    // and an outer-scope value is read by *no* operator of this graph - only
    // by the body.  So the pool had already recycled the buffer, and the copy
    // read whatever the pool had put there: the body then computed the same
    // thing for every input, and the formula recogniser's output did not
    // depend on the image at all.
    for (const auto & r : bg.outer_refs) {
        state_[r.second].last_use = std::max(state_[r.second].last_use, (int)op_index);
    }
    if (loop_dbg_) {
        fprintf(stderr, "[onnx] the loop: %zu outer refs\n", bg.outer_refs.size());
        for (const auto & r : bg.outer_refs) {
            const tensor & src = g_.tensors[r.second];
            const float * v = src.count && state_[r.second].dev
                                  ? (const float *)state_[r.second].dev : nullptr;
            float first = 0.0f;
            if (v) q_download(q_, v, &first, 4);
            fprintf(stderr, "   '%s' <- '%s' (count %zu, dev %s, the first %g)\n",
                    bg.tensors[r.first].name.c_str(), src.name.c_str(), src.count,
                    state_[r.second].dev ? "yes" : "NULL", first);
        }
    }
    for (const auto & r : bg.outer_refs) {
        copy_into(*bexec, r.first, r.second);
        // An outer-scope copy is read by the body on every iteration, so it
        // is a body input in everything but name: pin it too (see the input
        // pinning above - the borrow must not take it).
        if (r.first >= 0 && r.first < (int)bexec->state_.size()) {
            bexec->state_[r.first].pinned = true;
        }
    }
    if (loop_dbg_) {
        for (const auto & r : bg.outer_refs) {
            const tensor & d = bexec->graph().tensors[r.first];
            float hv = -1.0f;
            if (!d.host.empty()) std::memcpy(&hv, d.host.data(), 4);
            fprintf(stderr, "[onnx] the copy: the slot %d '%s' on_host %d, the host bytes %zu, the first %g, dev %p\n",
                    r.first, d.name.c_str(), (int)d.on_host, d.host.size(), hv, d.dev);
        }
    }
    int64_t max_iter = M < 0 ? (int64_t)1 << 40 : M;
    // A diagnostic: the cap on the iterations.  A loop that does not
    // terminate runs its body hundreds of times, and the dump of one
    // iteration is all that is needed to see why.
    if (const char * lm = getenv("SI_ONNX_LOOPMAX")) {
        const int64_t v = atoll(lm);
        if (v > 0 && v < max_iter) max_iter = v;
    }
    std::map<std::string, double> op_ms0;
    std::map<std::string, int> op_n0;
    std::map<std::string, long> op_dl_base, op_up_base;
    long dl0 = g_q_downloads.load();
    double w0 = bexec->wait_ms_, w00 = bexec->wait0_ms_;
    double f0 = bexec->feed_ms_, r0 = bexec->rb_ms_;
    if (opstat_) {
        op_ms0 = bexec->op_ms_;
        op_n0 = bexec->op_n_;
        op_dl_base = bexec->op_dl_;
        op_up_base = bexec->op_up_;
    }
    int64_t iters_run = 0;
    int64_t next_trim_iter = 0;
    const bool looptime = getenv("SI_ONNX_LOOPTIME") != nullptr;
    static const char * timestep = getenv("SI_ONNX_TIMESTEP");
    double body_ms = 0.0, body_min = 1e30, body_max = 0.0;
    double t_feed = 0.0, t_post = 0.0, t_cond = 0.0, t_carry = 0.0, t_scan = 0.0, t_trim = 0.0;
    double t_chain = 0.0; long n_chain = 0; long n_trim = 0;
    for (int64_t iter = 0; iter < max_iter; ++iter) {
        if (!cond) break;
        ++iters_run;
        static const bool loopmem = getenv("SI_ONNX_LOOPMEM") != nullptr;
        if (loopmem) {
            size_t dw = 0, ds = 0;
            conv_dnnl_mem(&dw, &ds);
            fprintf(stderr,
                    "[onnx] loop iter=%lld body_live=%.1f MiB body_pooled=%.1f MiB "
                    "body_allocs=%zu dnnl_w=%.1f MiB dnnl_s=%.1f MiB\n",
                    (long long)iters_run, bexec->live_bytes_calc() / 1048576.0,
                    bexec->pooled_bytes_calc() / 1048576.0, bexec->alloc_size_.size(),
                    dw / 1048576.0, ds / 1048576.0);
        }
        // The body's inputs.
        const auto tf0 = looptime ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
        std::vector<onnxwire::tensor> feeds;
        {
            onnxwire::tensor t;
            t.name = body_in[0];
            t.type = onnxwire::dtype::i64;
            t.shape = {1};
            const int64_t v = iter;
            t.data.resize(8);
            std::memcpy(t.data.data(), &v, 8);
            feeds.push_back(t);
        }
        {
            onnxwire::tensor t;
            t.name = body_in[1];
            // The condition is a float 0/1 in this runtime (a BOOL is a
            // float 0/1), so it is fed as one.
            t.type = onnxwire::dtype::f32;
            t.shape = {1};
            const float v = cond ? 1.0f : 0.0f;
            t.data.resize(4);
            std::memcpy(t.data.data(), &v, 4);
            feeds.push_back(t);
        }
        for (size_t i = 0; i < carried.size(); ++i) {
            // A pass-through only needs its value on the first iteration:
            // after that the body's input slot still holds it (it is a
            // graph output, so it is never released), and re-feeding it
            // would re-upload the weights and empty the host copy the
            // executor's own unchanged-feed check compares against.
            if (iter > 0 && !carried_produced[i]) continue;
            // A produced value on the device route is already in the input
            // slot when this iteration starts.
            if (iter > 0 && via_dev[i]) continue;
            onnxwire::tensor t;
            t.name = body_in[2 + i];
            t.type = carried[i].dtype == onnxpb::INT64 ? onnxwire::dtype::i64
                     : carried[i].dtype == onnxpb::INT32 ? onnxwire::dtype::i32
                                                        : onnxwire::dtype::f32;
            for (int64_t d : carried[i].shape) t.shape.push_back((uint64_t)d);
            t.data = carried[i].data;
            feeds.push_back(t);
        }
        // The body's outputs.  The buffers have to be built *before* the run:
        // the executor fills the buffers the caller asks for, by name, and an
        // empty list means the body's outputs are never collected - and the
        // code below then reads past the end of an empty vector.  Only the
        // condition, the produced carried values and the scans are asked
        // for; a pass-through stays in its slot (and keeps its host bytes).
        std::vector<std::vector<uint8_t>> out_bufs(want_out.size());
        std::vector<std::pair<std::string, std::vector<uint8_t> &>> outs;
        for (size_t j = 0; j < want_out.size(); ++j) {
            outs.emplace_back(bg.tensors[bg.output_slots[want_out[j]]].name, out_bufs[j]);
        }
        const auto bt0 = looptime ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
        if (timestep) bexec->sample_time_ = (iter == atoll(timestep));
        if (looptime) {
            t_feed += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - tf0).count();
        }
        bexec->run(feeds, outs);
        bexec->sample_time_ = false;
        const auto tp0 = looptime ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
        if (looptime) {
            const double dt = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - bt0).count();
            body_ms += dt;
            body_min = std::min(body_min, dt);
            body_max = std::max(body_max, dt);
        }
        const auto tc0 = looptime ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
        // The condition.
        if (!out_bufs.empty()) {
            const int cs = bg.output_slots.empty() ? -1 : bg.output_slots[0];
            cond = value_is_true(out_bufs[0], cs >= 0 ? bg.tensors[cs].dtype : onnxpb::FLOAT);
            if (loop_dbg_ && (iter < 3 || iter % 64 == 0)) {
                fprintf(stderr, "[onnx] the loop: the iteration %lld, the condition %d\n",
                        (long long)iter, (int)cond);
            }
        }
        if (looptime) {
            t_cond += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - tc0).count();
        }
        const auto tca0 = looptime ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        // The produced carried values and the scans, through the same
        // body-output indices.
        size_t j = 1;
        for (size_t i = 0; i < carried.size(); ++i) {
            if (!carried_produced[i]) continue;
            if (!via_dev[i]) {
                if (j >= out_bufs.size()) break;
                carried[i].data = out_bufs[j];
                ++j;
            }
            // The shape comes from the body: it is the only place where the
            // value's shape is known (the outer graph's shape for a carried
            // value is the shape of the *initial* value, which a loop that
            // changes it invalidates).
            const int s = bg.output_slots.size() > i + 1 ? bg.output_slots[i + 1] : -1;
            if (s >= 0) carried[i].shape = bexec->output(i + 1).shape;
            // Chain it to the next iteration on the device.  The copy runs
            // on every iteration; `via_dev` only records that this value no
            // longer needs the host read-back and the host feed.
            const auto tch0 = looptime ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
            const bool chained = chain_device(i);
            if (looptime && chained) {
                t_chain += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - tch0).count();
                ++n_chain;
            }
            if (chained) {
                if (!via_dev[i]) {
                    via_dev[i] = 1;
                    build_want();
                }
            }
        }
        if (looptime) {
            t_carry += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - tca0).count();
        }
        const auto ts0 = looptime ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
        // The scans.
        for (size_t si = 1 + carried.size(); si < bg.output_slots.size(); ++si) {
            if (j >= out_bufs.size()) break;
            const size_t s = si - 1 - carried.size();
            if (scans.size() <= s) scans.resize(s + 1);
            scans[s].push_back(out_bufs[j]);
            if (s >= scan_shape.size()) scan_shape.resize(s + 1);
            const int os = bg.output_slots.size() > si ? bg.output_slots[si] : -1;
            if (os >= 0) scan_shape[s] = bexec->output(si).shape;
            ++j;
        }
        // A growing shape leaves one buffer per past size in the body's free
        // pool: the formula recogniser's KV caches go [1,16,n,32] and every
        // iteration allocates a slightly larger tile than the last, so the
        // free list grows by ~28 MB per iteration.  The whole model's pool
        // is trimmed at the end of a run, but this loop can run 2 500
        // iterations in one run and reaches 12 GB long before it ends - the
        // OUT_OF_DEVICE_MEMORY that killed the formula model on the long
        // report corpus.  Trim at a bound, not every iteration: the pool's
        // recent sizes are still reused, and the wait is free because the
        // body's own run has already waited.
        if (looptime) {
            t_scan += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - ts0).count();
        }
        const auto tt0 = looptime ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
        static const size_t loop_trim_mb = [] {
            const char * e = getenv("PF_ONNX_LOOP_TRIM_MB");
            // 2 GiB: the free pool accumulates one buffer per distinct
            // growing size, and a trim frees all of them (a 128 ms drain +
            // free measured at 2 500 steps).  A lower bound made the trim
            // run 61 times over a 2 560-step decode - 7.8 s of the loop -
            // for memory that is reusable until it is freed.
            return e && *e ? (size_t)std::strtoull(e, nullptr, 10) : (size_t)2048;
        }();
        if (loop_trim_mb && iter >= next_trim_iter &&
            bexec->pooled_bytes_calc() > loop_trim_mb * 1024ull * 1024ull) {
            // The trim drains the queue and frees every pooled buffer; doing
            // it as soon as the bound is crossed meant once per iteration
            // (the body re-releases its 1 700 intermediates every time), and
            // it measured 2.4 ms of the loop.  At most once per 32 steps
            // keeps the free list bounded without paying it every step.
            if (looptime) ++n_trim;
            bexec->trim_pool();
            next_trim_iter = iter + 128;
        }
        if (looptime) {
            const auto tt1 = std::chrono::steady_clock::now();
            t_trim += std::chrono::duration<double, std::milli>(tt1 - tt0).count();
            t_post += std::chrono::duration<double, std::milli>(tt1 - tp0).count();
        }
    }
    if (looptime && iters_run) {
        fprintf(stderr,
                "[onnx] loop time: %lld iterations, the body run avg %.3f ms (min %.3f max %.3f), "
                "the feeds avg %.3f ms, the post avg %.3f ms "
                "(cond %.3f, carried %.3f, scans %.3f, trim %.3f over %ld; chain %.3f over %ld)\n",
                (long long)iters_run, body_ms / (double)iters_run, body_min, body_max,
                t_feed / (double)iters_run, t_post / (double)iters_run,
                t_cond / (double)iters_run, t_carry / (double)iters_run,
                t_scan / (double)iters_run, t_trim / (double)iters_run, n_trim,
                n_chain ? t_chain / (double)n_chain : 0.0, n_chain);
    }
    if (opstat_) {
        struct row { double ms; int n; long dl; long up; std::string name; };
        std::vector<row> rows;
        double total = 0.0;
        int n_ops = 0;
        for (const auto & kv : bexec->op_ms_) {
            const double dt = kv.second - (op_ms0.count(kv.first) ? op_ms0[kv.first] : 0.0);
            const int n = bexec->op_n_[kv.first] - (op_n0.count(kv.first) ? op_n0[kv.first] : 0);
            if (n <= 0) continue;
            const long dl = bexec->op_dl_[kv.first] -
                            (op_dl_base.count(kv.first) ? op_dl_base[kv.first] : 0L);
            const long up = bexec->op_up_[kv.first] -
                            (op_up_base.count(kv.first) ? op_up_base[kv.first] : 0L);
            rows.push_back({dt, n, dl, up, kv.first});
            total += dt;
            n_ops += n;
        }
        std::sort(rows.begin(), rows.end(), [](const row & a, const row & b) { return a.ms > b.ms; });
        fprintf(stderr,
                "[onnx] loop opstat: %lld iterations, %d ops, %.1f ms (%.3f ms/iteration), "
                "%ld blocking downloads, the final wait %.1f ms (%.3f/iteration), "
                "the feed-upload wait %.1f ms, the feeds %.1f ms, the readback enqueue %.1f ms\n",
                (long long)iters_run, n_ops, total, iters_run ? total / iters_run : 0.0,
                g_q_downloads.load() - dl0, bexec->wait_ms_ - w0,
                iters_run ? (bexec->wait_ms_ - w0) / iters_run : 0.0,
                bexec->wait0_ms_ - w00, bexec->feed_ms_ - f0, bexec->rb_ms_ - r0);
        for (size_t i = 0; i < rows.size() && i < 24; ++i) {
            fprintf(stderr, "   %-18s %8.2f ms total %7.3f ms/iteration x%d, %ld dl, %ld up\n",
                    rows[i].name.c_str(), rows[i].ms,
                    iters_run ? rows[i].ms / iters_run : 0.0, rows[i].n,
                    rows[i].dl, rows[i].up);
        }
    }
    // The outer scope's values are dead now: the body has its own copies.
    // (They were kept alive above because the copy reads their buffers.)
    for (const auto & r : bg.outer_refs) {
        if (state_[r.second].dev && !is_output(r.second) && !state_[r.second].is_const) {
            release(state_[r.second].dev, state_[r.second].bytes);
            state_[r.second].dev = nullptr;
            g_.tensors[r.second].dev = nullptr;
        }
    }
    if (loop_dbg_) {
        fprintf(stderr, "[onnx] the loop: after the run, the body's outputs:");
        for (int s2 : bg.output_slots) {
            fprintf(stderr, " %s[", bg.tensors[s2].name.c_str());
            for (int64_t d : bg.tensors[s2].shape) fprintf(stderr, " %lld", (long long)d);
            fprintf(stderr, "]");
        }
        fprintf(stderr, "\n");
    }
    // The outputs: the final values of the loop-carried dependencies, and
    // then the scans (the iterations along a new axis).
    //
    // **The condition is not one of them.**  A Loop's outputs are the final
    // values and the scans; the condition is an *input* (the loop's second
    // one).  The body's outputs are different: they start with the
    // condition, because the body reports it to the loop.
    //
    // The number of the scans comes from the body's own output count (its
    // outputs after the condition and the carried values), and what is left
    // of the loop's outputs is the number of the final values.  A model may
    // output *fewer* carried values than the body carries - the body can
    // carry values the loop never reports.
    const size_t n_scan = bg.output_slots.size() > 1 + carried.size()
                              ? bg.output_slots.size() - 1 - carried.size()
                              : 0;
    const size_t n_out = op.out.size();
    if (n_out < n_scan) {
        throw std::runtime_error("the Loop has too few outputs");
    }
    const size_t n_final = n_out - n_scan;
    if (n_final > carried.size()) {
        throw std::runtime_error("the Loop has more outputs than the body carries (" +
                                 std::to_string(n_final) + " against " +
                                 std::to_string(carried.size()) + ")");
    }
    for (size_t k = 0; k < n_out; ++k) {
        const int s = op.out[k];
        slot_state & st = state_[s];
        std::vector<uint8_t> raw;
        std::vector<int64_t> shape;
        int32_t dtype = onnxpb::FLOAT;
        if (k < n_final) {
            // A loop-carried value.
            const carried_value & cv = carried[k];
            raw = cv.data;
            // A value the body chained on the device never came back to the
            // host; its last iteration's output is read back once, here.
            if (k < via_dev.size() && via_dev[k] && k + 1 < bg.output_slots.size()) {
                const int os = bg.output_slots[k + 1];
                slot_state & ss = bexec->state_[os];
                if (ss.dev && ss.bytes) {
                    raw.resize(ss.bytes);
                    q_download(q_, ss.dev, raw.data(), raw.size());
                }
            }
            shape = cv.shape;
            dtype = cv.dtype;
        } else {
            // A scan output: the iterations along a new axis.
            const size_t si = k - n_final;
            const std::vector<std::vector<uint8_t>> & sc = si < scans.size() ? scans[si]
                                                                             : std::vector<std::vector<uint8_t>>();
            const std::vector<int64_t> & one = si < scan_shape.size() ? scan_shape[si]
                                                                      : std::vector<int64_t>();
            // The element size of one iteration's value: the body's own
            // output type.  The scan's shape is the iteration's shape with
            // a new leading axis.
            size_t esz = 4;
            const int os = bg.output_slots.size() > 1 + carried.size() + si
                               ? bg.output_slots[1 + carried.size() + si]
                               : -1;
            if (os >= 0) esz = runtime_elem_size(bg.tensors[os].dtype);
            const size_t per_bytes = one.empty() ? 0 : esz;
            size_t per_elems = 0;
            for (int64_t d : one) per_elems *= (size_t)(d > 0 ? d : 1);
            raw.resize(per_bytes * sc.size());
            for (size_t i = 0; i < sc.size(); ++i) {
                const size_t n = std::min(per_bytes, sc[i].size());
                if (n) std::memcpy(raw.data() + i * per_bytes, sc[i].data(), n);
            }
            shape.assign(1, (int64_t)sc.size());
            for (int64_t d : one) shape.push_back(d);
            (void)per_elems;
            if (os >= 0) dtype = bg.tensors[os].dtype;
        }
        g_.tensors[s].host = raw;
        if (loop_dbg_) {
            fprintf(stderr, "[onnx] the loop: the output %zu '%s': %zu bytes, the shape [", k,
                    g_.tensors[s].name.c_str(), raw.size());
            for (int64_t d : shape) fprintf(stderr, " %lld", (long long)d);
            fprintf(stderr, "]\n");
        }
        st.is_const = false;
        st.bytes = raw.size();
        g_.tensors[s].count = raw.size() / runtime_elem_size(dtype);
        g_.tensors[s].shape = shape;
        g_.tensors[s].dtype = dtype;
        g_.tensors[s].on_host = true;
        // The host bytes are new (a carried value or a scan).  The slot's
        // device copy, if any, is a mirror of a *previous* iteration/call and
        // is stale - and it must go back to the pool, or it leaks.
        ++st.host_gen;
        st.dev_gen = st.host_gen - 1;
        if (st.dev) {
            release(st.dev, st.bytes);
            st.dev = nullptr;
        }
        g_.tensors[s].dev = nullptr;
    }
}

// The leak report: the slots that still hold a device buffer after the run,
// and the pool's own accounting.  A value that is dead but still holds its
// buffer is a leak, and the document pipeline runs the same models thousands
// of times, so a leak is a crash after a few pages.
void executor::report_leak() const {
    if (!getenv("SI_ONNX_LEAK")) return;
    size_t n = 0;
    for (size_t s = 0; s < state_.size(); ++s) {
        if (!state_[s].dev) continue;
        if (is_output((int)s)) continue;
        fprintf(stderr, "[onnx] leak: slot %zu '%s' %zu bytes, the last use %d\n", s,
                g_.tensors[s].name.c_str(), state_[s].bytes, state_[s].last_use);
        ++n;
    }
    size_t pool_buffers = 0, pool_bytes = 0;
    for (const auto & kv : free_) {
        pool_buffers += kv.second.size();
        pool_bytes += kv.first * kv.second.size();
    }
    size_t alloc_total = 0;
    for (const auto & kv : alloc_size_) alloc_total += kv.second;
    // The slots that share a device buffer: two values, one buffer.
    {
        std::map<void *, std::vector<size_t>> owners;
        for (size_t s = 0; s < state_.size(); ++s) {
            if (state_[s].dev) owners[state_[s].dev].push_back(s);
        }
        for (const auto & kv : owners) {
            if (kv.second.size() < 2) continue;
            fprintf(stderr, "[onnx] leak: the buffer %p is held by", kv.first);
            for (size_t s : kv.second) fprintf(stderr, " %s", g_.tensors[s].name.c_str());
            fprintf(stderr, "\n");
        }
    }
    fprintf(stderr,
            "[onnx] leak: %zu live slots, %zu live bytes, %zu pooled bytes, %zu live allocations\n",
            n, live_bytes_calc(), pooled_bytes_calc(), live_.size());
    fprintf(stderr,
            "[onnx] leak: the pool holds %zu buffers, %zu bytes; the allocations total %zu bytes "
            "over %zu entries\n",
            pool_buffers, pool_bytes, alloc_total, alloc_size_.size());
}

namespace {
// Restore the oneDNN switch when a nested executor (a Loop body) returns.
struct dnnl_scope {
    bool prev;
    explicit dnnl_scope(bool on) : prev(conv_dnnl_enabled()) { conv_dnnl_set_enabled(on); }
    ~dnnl_scope() { conv_dnnl_set_enabled(prev); }
};
} // namespace

void executor::run(const std::vector<onnxwire::tensor> & inputs,
                   const std::vector<std::pair<std::string, std::vector<uint8_t> &>> & outputs) {
    const auto t0 = std::chrono::steady_clock::now();
    dnnl_scope dnnl_on(dnnl_);
    // The previous run's uploads read this graph's host bytes asynchronously,
    // and the input assignments below overwrite them.  One wait here is what
    // lets `prepare_inputs` enqueue its copies without waiting at each one.
    if (uploads_pending_) {
        const auto w0 = std::chrono::steady_clock::now();
        q_.wait();
        wait0_ms_ += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - w0).count();
        uploads_pending_ = false;
        up_stage_off_ = 0;
    }
    // The uploads are counted per run: a cumulative counter made the second
    // call look exactly like the first, and the whole point of the
    // measurement is what one call pays.
    upload_count_ = 0;
    upload_bytes_ = 0;
    if (upload_dbg_) {
        std::fill(upload_by_slot_.begin(), upload_by_slot_.end(), 0);
        std::fill(upload_bytes_by_slot_.begin(), upload_bytes_by_slot_.end(), 0);
    }
    // SI_ONNX_HOSTTIME: where an operator's time goes.  The operator sum
    // (SI_ONNX_TIME) is only part of the service's call: the input uploads,
    // the output allocations, the write-back (which downloads every integer
    // output, i.e. a device wait each) and the shape functions are the rest.
    // This separates them, because a slow kernel and a slow host need
    // different fixes.
    const bool hosttime = getenv("SI_ONNX_HOSTTIME") != nullptr;
    double ht_prepare = 0.0, ht_submit = 0.0, ht_writeback = 0.0;
    double ht_upload = 0.0, ht_alloc = 0.0, ht_ctx = 0.0;
    size_t ht_n = 0, ht_n_upload = 0;
    if (loop_dbg_) {
        fprintf(stderr, "[onnx] the run: %zu inputs, %zu ops\n", inputs.size(), g_.ops.size());
    }
    // The inputs.  Their shape comes with them: the graph's own input shape
    // is often dynamic (a text crop, a page), and the buffers are sized
    // from what the client sent.
    const auto tf0 = std::chrono::steady_clock::now();
    for (const onnxwire::tensor & in : inputs) {
        auto it = std::find_if(g_.input_slots.begin(), g_.input_slots.end(),
                               [&](int s) { return g_.tensors[s].name == in.name; });
        if (it == g_.input_slots.end()) {
            throw std::runtime_error("the graph has no input called " + in.name);
        }
        slot_state & s = state_[*it];
        // A feed whose bytes are the ones already there does not need a new
        // upload.  A Loop re-feeds its carried values every iteration, and
        // most of them are the decoder's weights (the table structurer's
        // body carries 31 values across 27 iterations): re-uploading 3 MB of
        // weights per iteration is 80 MB of host-to-device traffic for a
        // graph whose actual work is a handful of small matmuls.  The device
        // mirror is only reused when it is current (`dev_gen == host_gen`),
        // which `device_value` checks.
        {
            std::vector<uint8_t> & th = g_.tensors[*it].host;
            if (in.type != onnxwire::dtype::boolean && th.size() == in.data.size() &&
                (th.empty() || std::memcmp(th.data(), in.data.data(), th.size()) == 0)) {
                auto & gt0 = g_.tensors[*it];
                gt0.on_host = true;
                s.is_const = false;
                s.bytes = th.size();
                gt0.shape.assign(in.shape.begin(), in.shape.end());
                const int32_t dt0 = in.type == onnxwire::dtype::f32      ? onnxpb::FLOAT
                                    : in.type == onnxwire::dtype::i64   ? onnxpb::INT64
                                    : in.type == onnxwire::dtype::i32   ? onnxpb::INT32
                                    : in.type == onnxwire::dtype::i8    ? onnxpb::INT8
                                    : in.type == onnxwire::dtype::u8    ? onnxpb::UINT8
                                    : in.type == onnxwire::dtype::boolean ? onnxpb::BOOL
                                                                          : onnxpb::FLOAT;
                gt0.dtype = dt0;
                int64_t n0 = 1;
                for (uint64_t d : in.shape) n0 *= (int64_t)d;
                const size_t esz0 = runtime_elem_size(dt0);
                gt0.count = in.shape.empty() ? in.data.size() / esz0 : (size_t)n0;
                continue;
            }
        }
        if (in.type == onnxwire::dtype::boolean) {
            // The wire's boolean is one byte per element; the runtime's is a
            // float 0/1 (the device holds float32, and a one-byte boolean on
            // the device is read as three bytes of whatever is next).
            std::vector<uint8_t> f(in.data.size() * 4, 0);
            for (size_t i = 0; i < in.data.size(); ++i) {
                const float v = in.data[i] ? 1.0f : 0.0f;
                std::memcpy(f.data() + i * 4, &v, 4);
            }
            g_.tensors[*it].host = std::move(f);
        } else {
            g_.tensors[*it].host = in.data;
        }
        g_.tensors[*it].on_host = true;
        s.is_const = false;
        s.bytes = g_.tensors[*it].host.size();
        // New host bytes: any device copy made from the previous call's
        // bytes is stale.  The buffer is kept as the upload target rather
        // than released - the very next operator needs it, and the
        // generations are what says its contents are not current.
        ++s.host_gen;
        s.dev_gen = s.host_gen - 1;
        g_.tensors[*it].shape.assign(in.shape.begin(), in.shape.end());
        // The type: the wire's own.  A boolean is one byte per element and
        // an int32 four, so the count cannot be a fixed divisor.
        const int32_t dt = in.type == onnxwire::dtype::f32      ? onnxpb::FLOAT
                           : in.type == onnxwire::dtype::i64   ? onnxpb::INT64
                           : in.type == onnxwire::dtype::i32   ? onnxpb::INT32
                           : in.type == onnxwire::dtype::i8    ? onnxpb::INT8
                           : in.type == onnxwire::dtype::u8    ? onnxpb::UINT8
                           : in.type == onnxwire::dtype::boolean ? onnxpb::BOOL
                                                                 : onnxpb::FLOAT;
        g_.tensors[*it].dtype = dt;
        int64_t n = 1;
        for (uint64_t d : in.shape) n *= (int64_t)d;
        const size_t esz = runtime_elem_size(dt);
        g_.tensors[*it].count = in.shape.empty() ? in.data.size() / esz : (size_t)n;
    }
    feed_ms_ += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tf0).count();
    // The producer of each slot, for this run: a value written twice in one
    // run is a bug, but a Loop's body runs once per iteration and writes the
    // same slots every time.
    std::fill(producer_.begin(), producer_.end(), -1);
    // The branch of an If that does not run: nothing is skipped until an If
    // says so.
    skip_begin_ = -1;
    skip_end_ = -1;
    // The constant sub-graphs: evaluated once.
    if (!folded_) {
        fold_constants();
        prepare_conv_weights();
        folded_ = true;
    }

    for (size_t i = 0; i < g_.ops.size(); ++i) {
        exec_op & op = g_.ops[i];
        // The branch of an If that does not run: its operators are in the
        // list, and they are skipped.
        if (skip_begin_ >= 0 && i >= (size_t)skip_begin_ && i < (size_t)skip_end_) {
            continue;
        }
        // An operator that only feeds a dropped output: the graph was pruned
        // for the outputs the client asked for, so it is not computed at all.
        if (!live_ops_.empty() && !live_ops_[i]) continue;
        if (op.node->op_type == "Constant") {
            // Its value was folded already - unless it is the branch of an
            // If, where only one of the two branches runs and the value has
            // to be produced by the selected one.
            if (!op.in_if_branch) continue;
        }
        if (op.node->op_type == "Loop") {
            // The Loop has its own path: the body's shape is not known
            // before the loop has run.
            run_loop(i, op);
            continue;
        }
        if (op.node->op_type == "If") {
            // The condition decides which of the two inlined branches runs.
            // The other one is skipped; its operators are the ones the
            // branch's range names.
            bool cond = false;
            if (!op.in.empty() && op.in[0] >= 0) {
                const int s = op.in[0];
                std::vector<uint8_t> h = host_of(s);
                if (h.empty() && state_[s].dev) {
                    h.resize(state_[s].bytes);
                    q_download(q_, state_[s].dev, h.data(), h.size());
                }
                cond = value_is_true(h, g_.tensors[s].dtype);
            }
            if (cond) {
                skip_begin_ = op.if_else_begin;
                skip_end_ = op.if_else_end;
            } else {
                skip_begin_ = op.if_then_begin;
                skip_end_ = op.if_then_end;
            }
            if (loop_dbg_) {
                fprintf(stderr, "[onnx] the if: the condition %d, the then [%d, %d), the else [%d, %d)\n",
                        (int)cond, op.if_then_begin, op.if_then_end, op.if_else_begin, op.if_else_end);
            }
            continue;
        }
        const op_entry * e = oc_[i].entry;
        if (!e) {
            throw std::runtime_error("the operator '" + op.node->op_type + "' is not implemented");
        }
        // An operator whose inputs are all constants was folded already.
        // **Not one inside an If's branch**: only the selected branch runs,
        // so its value depends on the condition - and the branch's
        // pass-through copy (an empty branch's constant output) has a
        // constant input too.  Skipping it left the If's slot marked as a
        // constant that was never computed, and every consumer of it was
        // skipped as well.
        bool all_const = !op.in.empty() && !op.in_if_branch;
        for (int s : op.in) {
            if (s < 0) continue;
            if (!state_[s].is_const) {
                all_const = false;
                break;
            }
        }
        if (all_const) {
            if (loop_dbg_) {
                fprintf(stderr, "[onnx] the op %zu %s is const (skipped)\n", i,
                        op.node->op_type.c_str());
            }
            // An operator the runtime does not implement is an error here
            // too, not a silent skip: the fold skipped it as well (it does
            // not know it either), so its output would be an *empty* value,
            // and a consumer that reads it (a loop-carried value) reads
            // garbage instead of failing.
            if (!e) {
                throw std::runtime_error("the operator '" + op.node->op_type +
                                         "' is not implemented (the operator " + std::to_string(i) + ")");
            }
            for (int s : op.out) {
                if (s >= 0) state_[s].is_const = true;
            }
            continue;
        }
        if (trace_on()) {
            fprintf(stderr, "[onnx] op %zu %s\n", i, op.node->op_type.c_str());
        }
        if (loop_dbg_) {
            fprintf(stderr, "[onnx] the op %zu %s runs\n", i, op.node->op_type.c_str());
        }
        // The timestamps are taken only when a diagnostic needs them: six
        // clock reads per operator is ~1 us, and the table structurer's Loop
        // runs 157 operators per iteration.
        using tp = std::chrono::steady_clock::time_point;
        const bool timed = hosttime || opstat_ || sample_time_;
        const auto th0 = timed ? std::chrono::steady_clock::now() : tp{};
        const long dl_op0 = opstat_ ? g_q_downloads.load() : 0;
        const size_t up_op0 = opstat_ ? upload_count_ : 0;
        auto th1 = th0, th2 = th0;
        op_ctx ctx(q_, *op.node);
        ctx.activation = op.activation;
        ctx.act_params = op.act_params;
        ctx.opset = g_.opset;
        ctx.workspace = scratch();
        ctx.workspace_bytes = ws_bytes_;
        ctx.scratch = scratch_small();
        ctx.scratch_bytes = ws2_bytes_;
        // The attention PV's partials live in their own buffer (the shared
        // workspace is also the top-k's and the gather's).  Allocate it only
        // for a model that has MatMuls; the capacity a call needs is checked
        // there and an oversized shape falls back to the naive path.
        if (op.node->op_type == "MatMul") {
            ctx.pv_workspace = pv_scratch(1u << 20);
            ctx.pv_workspace_bytes = pv_ws_bytes_;
        }
        // The inputs are uploaded *before* the context is filled: the
        // operators read the device pointers through the context's tensors,
        // and a pointer captured before the upload is null.
        //
        // The upload is inside the try: an upload that fails is a device
        // error, and it has to name the operator like every other one.  (The
        // upload was outside, and a device lost by the *previous* operator
        // surfaced here, without a name.)
        // Operators whose *inputs* the host reads directly do not need their
        // device mirrors refreshed.  The integer control flow of the table
        // decoder is built from such operators (Shape, Range, the int
        // Concat/Slice/Tile, the int elementwise ops), and each of their
        // inputs was copied to the device and then never read there - 47 of
        // the ~86 uploads one Loop iteration was paying.
        bool host_inputs_only = false;
        // An int64 comparison whose operands are a scalar or the same length
        // has a device kernel now (ew_i64_cmp_launch), including when its
        // result is a BOOL (a float 0/1).  It is not in `int_dev_capable`
        // because that list also promises an int32 and a broadcasting path;
        // this one is narrower, so it only turns off the host fast path and
        // the safety-net download for exactly the shapes the kernel handles.
        bool cmp64_dev = false;
        {
            const std::string & ty = op.node->op_type;
            if ((ty == "Equal" || ty == "Greater" || ty == "Less" || ty == "GreaterOrEqual" ||
                 ty == "LessOrEqual" || ty == "And" || ty == "Or" || ty == "Not") &&
                !op.in.empty() && op.in.size() <= 2) {
                cmp64_dev = true;
                int64_t counts[2] = {0, 0};
                int n_in = 0;
                for (int si : op.in) {
                    if (si < 0) {
                        cmp64_dev = false;
                        break;
                    }
                    if (g_.tensors[si].dtype != onnxpb::INT64) {
                        cmp64_dev = false;
                        break;
                    }
                    if (n_in < 2) counts[n_in] = (int64_t)g_.tensors[si].count;
                    ++n_in;
                }
                if (cmp64_dev && n_in == 2) {
                    cmp64_dev = counts[0] == counts[1] || counts[0] == 1 || counts[1] == 1;
                }
            }
        }
        if (!mirror_chk_) {
            const std::string & ty = op.node->op_type;
            if (ty == "Shape" || ty == "Range") {
                host_inputs_only = true;
            } else if (ty == "Concat" || ty == "Slice" || ty == "Tile" || ty == "Expand" ||
                       ty == "Gather" || ty == "Split" || ty == "Transpose" ||
                       ty == "GatherND" || ty == "ScatterND" || ty == "OneHot") {
                host_inputs_only = true;
                for (int si : op.in) {
                    if (si < 0) continue;
                    if (g_.tensors[si].dtype == onnxpb::FLOAT) {
                        host_inputs_only = false;
                        break;
                    }
                }
            } else if (op.kind == op_kind::elementwise) {
                // The host path is taken for an integer operand or an
                // integer result; all-integer inputs are always that path -
                // unless the op carries a fused activation, where the host
                // branch does not apply and the device path needs current
                // mirrors (the formula's outer graph adds two shape scalars
                // with such an epilogue; uploading them was cheaper than
                // reading a stale mirror and growing a Range to 4 GB).
                host_inputs_only =
                    op.activation.empty() || op.activation == "none";
                for (int si : op.in) {
                    if (si < 0) continue;
                    if (g_.tensors[si].dtype == onnxpb::FLOAT) {
                        host_inputs_only = false;
                        break;
                    }
                }
            } else if (ty == "ReduceMax" || ty == "ReduceMin" || ty == "ReduceSum" ||
                       ty == "ReduceMean") {
                const int si = op.in.empty() ? -1 : op.in[0];
                host_inputs_only = si >= 0 && si < (int)state_.size() && state_[si].bytes > 0 &&
                                   state_[si].bytes <= (size_t)8192 && g_.tensors[si].on_host &&
                                   (g_.tensors[si].dtype == onnxpb::INT32 ||
                                    g_.tensors[si].dtype == onnxpb::INT64 ||
                                    g_.tensors[si].dtype == onnxpb::BOOL);
            } else if (ty == "Cast" && !op.in.empty() && op.in[0] >= 0) {
                // The host cast path (see exec_cast): an integer target up to
                // 64K elements, or a small float target, from host bytes.
                const tensor & ti = g_.tensors[op.in[0]];
                int32_t to = onnxpb::FLOAT;
                for (const onnxpb::attribute & a : op.node->attribute) {
                    if (a.name == "to") to = (int32_t)a.i;
                }
                const bool int_target = to == onnxpb::INT64 || to == onnxpb::INT32;
                host_inputs_only = ti.on_host && !ti.host.empty() && ti.count > 0 &&
                                   (int_target ? ti.count <= (size_t)65536 : ti.count <= (size_t)8192);
            }
            // A device-authoritative integer input turns the host-only fast
            // path off: the operator's device path reads it through its
            // device pointer, and the host inputs alongside it have to be
            // uploaded for that path.
            if (host_inputs_only && (int_dev_capable(op.node->op_type) || cmp64_dev)) {
                for (int si : op.in) {
                    if (si < 0) continue;
                    // A device-authoritative integer input means the op's
                    // device path needs the *host* operands uploaded too (a
                    // comparison of a device token id with a host constant
                    // was reading the constant's stale mirror).  A host value
                    // whose mirror is merely behind is not this case: the
                    // op's host path will use the host bytes, and forcing
                    // prepare_inputs uploaded 162 shape tensors per formula
                    // iteration for nothing.
                    if (int_dtype(g_.tensors[si].dtype) && state_[si].dev && host_of(si).empty()) {
                        host_inputs_only = false;
                        break;
                    }
                }
            }
            // A metadata operator on a small host value copies it on the
            // host and never reads the device mirror; preparing (uploading)
            // its input was one copy per iteration of the formula's shape
            // scalars.
            if (!host_inputs_only && op.kind == op_kind::alias && !op.in.empty() &&
                op.in[0] >= 0 && g_.tensors[op.in[0]].on_host &&
                !g_.tensors[op.in[0]].host.empty() &&
                g_.tensors[op.in[0]].count <= (size_t)65536) {
                host_inputs_only = true;
            }
            // A small elementwise operator whose operands are already on the
            // host is computed there by exec_elementwise.  Preparing its
            // inputs uploaded them first and threw the copies away: the
            // formula recogniser's position-embedding scalars are a Cast of
            // an index times a constant, and both the inputs (12 x per step)
            // and the (host) result were copied for nothing.
            if (!host_inputs_only && op.kind == op_kind::elementwise &&
                (op.activation.empty() || op.activation == "none") &&
                host_elementwise_op(op.node->op_type)) {
                bool all_host = true;
                size_t max_in = 0;
                for (int si : op.in) {
                    if (si < 0) continue;
                    const tensor & ti = g_.tensors[si];
                    max_in = std::max(max_in, ti.count);
                    if (!ti.on_host || ti.host.empty()) {
                        all_host = false;
                        break;
                    }
                }
                if (all_host && max_in > 0 && max_in <= (size_t)8192) host_inputs_only = true;
            }
        }
        // A *small* integer chain is host arithmetic even when its first
        // value was produced on the device.
        //
        // The formula recogniser builds its reshape targets as
        // `Slice -> Cast -> Mul -> Reshape -> Cast -> Unsqueeze -> Concat`,
        // a handful of scalars that derive from the KV cache's current
        // length.  The first value is device-authoritative, so every
        // operator in the chain took its device path and the *Reshape* that
        // consumed the last one came back to the host for it - one blocking
        // queue drain per chain, ~48 per iteration, and each drain waits for
        // all the device work queued so far.  The arithmetic is a few
        // elements: the host does it for free and the result stays there, so
        // the whole chain after the first value never crosses the bus.
        //
        // The gate is the *shape of the data*, not the operator's name: only
        // integer tensors of at most 8 192 elements, and only for the
        // operators whose host path is the same arithmetic as the device's.
        // A large integer (the layout model's 12-million-element mask, a
        // TopK index plane) is untouched - the device is the right side for
        // it, and forcing it to the host would be the regression the
        // integer-chain work was written to avoid.
        if (!host_inputs_only && !mirror_chk_) {
            const std::string & ty = op.node->op_type;
            const bool shape_chain_op =
                ty == "Cast" || ty == "Concat" || ty == "Slice" || ty == "Reshape" ||
                ty == "Flatten" || ty == "Squeeze" || ty == "Unsqueeze" || ty == "Identity" ||
                ty == "Transpose" || ty == "Gather" || ty == "Shape" || ty == "Add" ||
                ty == "Sub" || ty == "Mul" || ty == "Div" || ty == "Mod" || ty == "Max" ||
                ty == "Min" || ty == "Equal" || ty == "Greater" || ty == "Less" ||
                ty == "GreaterOrEqual" || ty == "LessOrEqual" || ty == "And" || ty == "Or" ||
                ty == "Not" || ty == "Where";
            bool any_in = false, all_int_in = true, small = true, any_dev = false;
            bool any_data_dev = false;
            size_t max_in = 0;
            for (int si : op.in) {
                if (si < 0) continue;
                any_in = true;
                const tensor & ti = g_.tensors[si];
                if (!int_dtype(ti.dtype)) all_int_in = false;
                max_in = std::max(max_in, ti.count);
                if (state_[si].dev && host_of(si).empty()) {
                    any_dev = true;
                    if (si >= (int)shape_derived_.size() || !shape_derived_[si]) any_data_dev = true;
                }
            }
            if (!any_in || max_in > (size_t)8192) small = false;
            // A cast between a small integer and a small float is part of
            // the same chain (the model casts the KV length to the dtype its
            // arithmetic uses and back).
            bool cast_chain = false;
            if (ty == "Cast" && small && !op.in.empty() && op.in[0] >= 0) {
                int32_t to = onnxpb::FLOAT;
                for (const onnxpb::attribute & a : op.node->attribute) {
                    if (a.name == "to") to = (int32_t)a.i;
                }
                cast_chain = int_dtype(to) || int_dtype(g_.tensors[op.in[0]].dtype);
            }
            if (shape_chain_op && small && any_dev && !any_data_dev && (all_int_in || cast_chain)) {
                static const bool chain_dbg = getenv("SI_ONNX_CHAINDBG") != nullptr;
                const long chain_dl0 = chain_dbg ? g_q_downloads.load() : 0;
                if (chain_dbg) {
                    fprintf(stderr, "[onnx] chain: op %zu %s %s", i, op.node->op_type.c_str(),
                            op.node->name.c_str());
                    for (int si : op.in) {
                        if (si < 0) {
                            fprintf(stderr, " | -");
                            continue;
                        }
                        fprintf(stderr, " | '%s' d%d n%zu host%zu dev%d sder%d",
                                g_.tensors[si].name.c_str(), (int)g_.tensors[si].dtype,
                                g_.tensors[si].count, host_of(si).size(),
                                state_[si].dev ? 1 : 0,
                                si < (int)shape_derived_.size() ? shape_derived_[si] : -1);
                    }
                    fprintf(stderr, "\n");
                }
                for (int si : op.in) {
                    if (si < 0) continue;
                    if (state_[si].dev && host_of(si).empty()) materialize_int(si);
                }
                if (chain_dbg) {
                    fprintf(stderr, "[onnx] chain: op %zu %s downloads %ld\n", i,
                            op.node->op_type.c_str(), g_q_downloads.load() - chain_dl0);
                }
                // Every input is on the host now: the uploads the device
                // path would need are not just unnecessary, they are the
                // copies that started the chain on the device again.
                host_inputs_only = true;
            }
        }
        const auto tu0 = hosttime ? std::chrono::steady_clock::now() : tp{};
        try {
            if (!host_inputs_only) prepare_inputs(op);
        } catch (const std::exception & ex) {
            throw std::runtime_error("the operator " + std::to_string(i) + " (" +
                                     op.node->op_type + ") failed while preparing its inputs: " +
                                     ex.what());
        }
        const auto tu1 = hosttime ? std::chrono::steady_clock::now() : tp{};
        for (int s : op.in) {
            if (s < 0) continue;
            ctx.in.push_back(&g_.tensors[s]);
            ctx.in.back()->dev = state_[s].dev;
        }
        // The output tensors live in a vector that outlives the call: a
        // pointer to a stack temporary would dangle the moment the loop
        // iteration ends, and every write through it (the device pointer,
        // the shape, the count) would land on freed stack.
        std::vector<tensor> out_tensors;
        out_tensors.reserve(op.out.size());
        for (int s : op.out) {
            if (s < 0) continue;
            tensor t;
            t.name = g_.tensors[s].name;
            t.dtype = g_.tensors[s].dtype;
            out_tensors.push_back(t);
        }
        for (tensor & t : out_tensors) ctx.out.push_back(&t);

        // The output's type is the input's: the model does not carry the
        // types of the intermediates, and an operator that copies (the
        // Slice, the Concat, the Gather, the Reshape) keeps the type.  The
        // shape function may change it (the Shape and the ArgMax produce
        // int64).
        for (tensor * t : ctx.out) {
            if (!ctx.in.empty()) t->dtype = ctx.in[0]->dtype;
        }
        // The parameters the shape functions read are the exception to the
        // device residency: a reshape target, a Slice's bounds, a Tile's
        // repeats, a TopK's k.  They are small, and they are the *only*
        // integer values whose content is needed before the operator runs.
        // Materialising them here (and nothing else) keeps a device-only
        // index tensor from being pulled back just because an operator's
        // shape function happens to sit between the two.
        {
            const std::string & ty = op.node->op_type;
            for (size_t pos = 0; pos < op.in.size(); ++pos) {
                const int s = op.in[pos];
                if (s < 0) continue;
                const tensor & t = g_.tensors[s];
                if (t.host.empty() == false) continue;
                if (!int_shape_param(ty, pos)) continue;
                const bool int_param = int_dtype(t.dtype) || ty == "Resize";
                if (!int_param) continue;
                materialize_int(s);
            }
        }
        op_cache & oc = oc_[i];
        // The signature of a shape function: the input shapes, types and
        // counts, plus the bytes of the small integer/float inputs whose
        // *values* decide a shape (a reshape target, a slice's bounds, a
        // Resize's scales).  A hit reuses the last output shapes and skips
        // the function; an operator whose size is data-dependent is never
        // cached.
        std::vector<int64_t> sig;
        std::vector<uint8_t> params;
        bool sig_complete = true;
        if (!oc.shaped_from_size) {
            for (int s : op.in) {
                if (s < 0) continue;
                const tensor & t = g_.tensors[s];
                sig.push_back((int64_t)t.dtype);
                sig.push_back((int64_t)t.count);
                for (int64_t d : t.shape) sig.push_back(d);
                if (t.count <= 64) {
                    // A small input can decide the shape (a reshape target, a
                    // Slice bound, a Resize scale).  If its bytes are not on
                    // the host we cannot tell whether they changed, so this
                    // call's shapes are not cacheable - a stale shape is an
                    // out-of-bounds write, not a rounding error.
                    if (t.host.empty()) sig_complete = false;
                    else {
                        const size_t nb = std::min<size_t>(t.host.size(), (size_t)512);
                        params.insert(params.end(), t.host.begin(), t.host.begin() + nb);
                    }
                }
            }
        }
        bool cache_hit = oc.valid && !oc.shaped_from_size && oc.last_sig == sig &&
                         oc.last_params == params;
        std::vector<std::vector<int64_t>> & shapes = oc.shapes;
        bool have_shape = cache_hit;
        if (!cache_hit) shapes.clear();
        try {
            if (cache_hit) {
                // A shape function also sets an output type that differs
                // from its input's (Shape, ArgMax, Cast).  A hit has to
                // restore it, or the cached shape comes with the wrong
                // width.
                for (size_t k = 0; k < ctx.out.size() && k < oc.last_dtypes.size(); ++k) {
                    ctx.out[k]->dtype = oc.last_dtypes[k];
                }
            } else {
                have_shape = e->shape(ctx, shapes);
                if (!have_shape) {
                    // The output's shape depends on the data: the operator
                    // computes it (a Resize whose sizes are computed, a Range
                    // whose limit is the number of the boxes), and the result
                    // is not a pure function of the signature - never cache
                    // an operator that took this route.
                    const size_fn * sf = find_size_fn(op.node->op_type);
                    if (sf) have_shape = (*sf)(ctx, shapes);
                    oc.shaped_from_size = true;
                } else if (sig_complete) {
                    oc.last_sig = std::move(sig);
                    oc.last_params = std::move(params);
                    oc.last_dtypes.clear();
                    for (const tensor * t : ctx.out) oc.last_dtypes.push_back(t->dtype);
                    oc.valid = true;
                } else {
                    oc.valid = false;
                }
            }
        } catch (const std::exception & ex) {
            // The shape functions run *before* the operator, and their
            // failures used to reach the client as a bare "shapes do not
            // broadcast" - which says nothing about which operator or which
            // shapes.  This is the trace that finds the wrong one.
            std::string what = "the operator " + std::to_string(i) + " (" + op.node->op_type +
                               ") failed while planning its output: " + ex.what();
            for (size_t k = 0; k < ctx.in.size(); ++k) {
                what += " | in[" + std::to_string(k) + "] " + ctx.in[k]->name + " [";
                for (int64_t d : ctx.in[k]->shape) what += " " + std::to_string(d);
                what += "]";
            }
            throw std::runtime_error(what);
        }
        if (trace_ && op.node->op_type == "Reshape") {
            fprintf(stderr, "[onnx] the reshape's shape: %d", (int)have_shape);
            for (const auto & sh : shapes) {
                fprintf(stderr, " [");
                for (int64_t d : sh) fprintf(stderr, "%lld ", (long long)d);
                fprintf(stderr, "]");
            }
            fprintf(stderr, "\n");
        }
        for (size_t k = 0; k < ctx.out.size(); ++k) {
            if (have_shape && k < shapes.size()) {
                ctx.out[k]->shape = shapes[k];
                ctx.out[k]->count = (size_t)std::max<int64_t>(0, shape_count(shapes[k]));
            }
            if (!have_shape) {
                // The output's shape is neither known nor computable: that is
                // an error, not a zero-sized tensor.  (A count of 1 would
                // make the operator write past the end of its buffer.)
                throw std::runtime_error("the operator '" + op.node->op_type +
                                         "' does not know the shape of its output");
            }
            if (ctx.out[k]->count == 0) ctx.out[k]->count = 1;
            if (trace_) {
                for (int64_t d : ctx.out[k]->shape) {
                    if (d == 0) {
                        fprintf(stderr, "[onnx] op %zu %s: the output '%s' has a zero dim:", i,
                                op.node->op_type.c_str(), ctx.out[k]->name.c_str());
                        for (int64_t e : ctx.out[k]->shape) fprintf(stderr, " %lld", (long long)e);
                        fprintf(stderr, "\n");
                        break;
                    }
                }
            }
        }
        if (shape_dbg_) {
            // The shapes of every operator's inputs and outputs.  This is
            // the trace that finds a wrong shape: a value whose shape does
            // not match the operator that produced it.
            fprintf(stderr, "[onnx] op %zu %s\n", i, op.node->op_type.c_str());
            for (size_t k = 0; k < ctx.in.size(); ++k) {
                fprintf(stderr, "    in  %s: [", ctx.in[k]->name.c_str());
                for (int64_t d : ctx.in[k]->shape) fprintf(stderr, " %lld", (long long)d);
                fprintf(stderr, " ] count=%zu\n", ctx.in[k]->count);
            }
            for (size_t k = 0; k < ctx.out.size(); ++k) {
                fprintf(stderr, "    out %s: [", ctx.out[k]->name.c_str());
                for (int64_t d : ctx.out[k]->shape) fprintf(stderr, " %lld", (long long)d);
                fprintf(stderr, " ] count=%zu\n", ctx.out[k]->count);
            }
        }
        // The dump of the inputs happens *before* the operator runs: after it,
        // the input's buffer may have been overwritten (an alias), and the
        // dump would then show the new values and hide the fault.
        if (dump_dbg_) {
            // A diagnostic: the first elements of the inputs and the outputs,
            // so the host's view of a buffer can be compared with the
            // device's.  The format is parseable, so a script can compare
            // the runtime's intermediates with ONNXRuntime's.
            const size_t nvals = (size_t)dump_nvals_;
            for (size_t c = 0; c < ctx.in.size(); ++c) {
                const tensor * t = ctx.in[c];
                const size_t n = std::min<size_t>(t->count, std::min<size_t>(nvals, 64));
                // Where the value lives.  An integer lives on the host, and
                // so does a value that an operator computed on the host -
                // and then the device copy is stale.  (Reading the device
                // there is how the dump showed a comparison of two int64
                // masks as zero while the host had the answer.)
                const bool ints = t->dtype == onnxpb::INT64 || t->dtype == onnxpb::INT32;
                const size_t esz = ints ? (t->dtype == onnxpb::INT64 ? 8u : 4u) : 4u;
                const bool on_host = t->on_host && !t->host.empty();
                fprintf(stderr, "[onnx] op %zu %s in[%zu] %s:", i, op.node->op_type.c_str(), c,
                        t->name.c_str());
                if (on_host) {
                    const size_t have = t->host.size() / esz;
                    for (size_t b = 0; b < n && b < have; ++b) {
                        if (esz == 8) {
                            int64_t x = 0;
                            std::memcpy(&x, t->host.data() + b * 8, 8);
                            fprintf(stderr, " %lld", (long long)x);
                        } else if (ints) {
                            int32_t x = 0;
                            std::memcpy(&x, t->host.data() + b * 4, 4);
                            fprintf(stderr, " %d", x);
                        } else {
                            float x = 0.0f;
                            std::memcpy(&x, t->host.data() + b * 4, 4);
                            fprintf(stderr, " %.9g", x);
                        }
                    }
                } else if (t->dev && n) {
                    if (ints) {
                        uint8_t raw[64 * 8] = {0};
                        q_download(q_, t->dev, raw, n * esz);
                        for (size_t b = 0; b < n; ++b) {
                            if (esz == 8) {
                                int64_t x = 0;
                                std::memcpy(&x, raw + b * 8, 8);
                                fprintf(stderr, " %lld", (long long)x);
                            } else {
                                int32_t x = 0;
                                std::memcpy(&x, raw + b * 4, 4);
                                fprintf(stderr, " %d", x);
                            }
                        }
                    } else {
                        float v[64] = {0};
                        q_download(q_, t->dev, v, n * 4);
                        for (size_t b = 0; b < n; ++b) fprintf(stderr, " %.9g", v[b]);
                    }
                }
                fprintf(stderr, "  @%p\n", (const void *)t->dev);
            }
        }
        // The output buffers.  Each one gets a guard band behind it, so an
        // operator that writes past the end is caught here instead of
        // corrupting the heap much later.  The band is written *on the
        // device*: a host write to a device pointer faults on a discrete GPU
        // (and silently corrupts the heap on an integrated one).
        // The guard band behind every output: it catches a kernel that
        // writes past the end of its buffer.  Its size is configurable
        // because a *big* overflow (more than a few hundred bytes) lands in
        // the next allocation - and corrupts a constant.
        //
        // It is **off** by default, because it costs a device wait per
        // output and a wait per operator costs the pipeline its depth: on
        // the formula recogniser (thousands of small operators) the waits
        // were most of the runtime.  A debug run turns it on:
        // `SI_ONNX_GUARD=256`.
        const size_t kGuard = guard_;
        const auto ta0 = hosttime ? std::chrono::steady_clock::now() : tp{};
        std::vector<void *> out_dev(ctx.out.size(), nullptr);
        const size_t miss0 = malloc_count_;
        // In-place reuse.  A value-preserving metadata op (Reshape/Flatten/
        // Squeeze/Unsqueeze/Identity) and an elementwise op whose output has
        // its input's bytes can take over an input buffer that dies here
        // instead of allocating and copying one.  The input is the last use
        // of *this* operator, is not a graph output and is not a constant,
        // so the ownership transfer is safe; the op still runs for the
        // elementwise case (it reads and writes the same element).
        int borrow_slot = -1;
        bool skip_exec = false;
        if (kGuard == 0 && ctx.out.size() == 1 &&
            (ctx.out[0]->dtype == onnxpb::FLOAT || int_dtype(ctx.out[0]->dtype)) &&
            (op.kind == op_kind::alias || op.kind == op_kind::elementwise)) {
            const size_t out_bytes = ctx.out[0]->bytes();
            const size_t n_in = op.kind == op_kind::alias ? 1 : op.in.size();
            for (size_t k = 0; k < n_in; ++k) {
                const int s = op.in[k];
                if (s < 0 || s >= (int)state_.size()) continue;
                slot_state & cs = state_[s];
                if (!cs.dev || cs.is_const || cs.pinned || is_output(s) ||
                    cs.last_use != (int)i) continue;
                if (cs.bytes != out_bytes) continue;
                if (g_.tensors[s].dtype != ctx.out[0]->dtype) continue;
                // A value that has host bytes cannot be borrowed: the
                // ownership transfer moves the *device* buffer only, and the
                // output would lose the host copy.  That is exactly what
                // happened to the formula recogniser's shape scalars: a
                // `Reshape` of a host-resident scalar borrowed the scalar's
                // device mirror, skipped its own host copy, and the `Cast`
                // after it pulled the value back with a full queue drain.
                // (It was an integer-only guard before - the same bug for an
                // integer, but the float scalars took the same path.)
                if (!g_.tensors[s].host.empty()) continue;
                if (op.kind == op_kind::elementwise &&
                    g_.tensors[s].shape != ctx.out[0]->shape) {
                    continue;
                }
                if (op.kind == op_kind::alias && ctx.in[k]->count != ctx.out[0]->count) continue;
                borrow_slot = s;
                break;
            }
        }
        try {
            for (size_t k = 0; k < ctx.out.size(); ++k) {
                const size_t bytes = ctx.out[k]->bytes();
                if (k == 0 && borrow_slot >= 0) {
                    out_dev[k] = state_[borrow_slot].dev;
                    ctx.out[k]->dev = out_dev[k];
                    // The output owns the buffer now.  The *tensor* pointer
                    // stays valid for this operator's inputs; only the
                    // slot's ownership moves.
                    state_[borrow_slot].dev = nullptr;
                    skip_exec = op.kind == op_kind::alias;
                    continue;
                }
                out_dev[k] = acquire(bytes + kGuard);
                if (!out_dev[k]) {
                    // A failed allocation used to reach the operator as a null
                    // device pointer, and the operator's copy reported it as
                    // "NULL pointer argument in memory copy operation" - which
                    // says nothing about the allocation that failed.  Say it
                    // here, with the size and the device's free memory.
                    const size_t free_bytes = free_mem_bytes();
                    throw std::runtime_error(
                        "the device allocation of " + std::to_string(bytes) + " bytes failed (" +
                        std::to_string(free_bytes) + " bytes free) for the output '" +
                        ctx.out[k]->name + "' of the operator " + std::to_string(i) + " (" +
                        op.node->op_type + ")");
                }
                // The band is written on the device (a host write to a device
                // pointer faults on a discrete GPU), and *not* waited for:
                // the queue is in order, so the band is in place before the
                // operator's kernels run.
                if (kGuard) q_.memset((uint8_t *)out_dev[k] + bytes, 0xAB, kGuard);
                ctx.out[k]->dev = out_dev[k];
            }
            const auto ta1 = hosttime ? std::chrono::steady_clock::now() : tp{};
            if (hosttime) {
                ht_upload += std::chrono::duration<double, std::milli>(tu1 - tu0).count();
                ht_alloc += std::chrono::duration<double, std::milli>(ta1 - ta0).count();
                ht_ctx += std::chrono::duration<double, std::milli>(tu0 - th0).count();
                ht_n_upload += (malloc_count_ - miss0);
            }
            th1 = timed ? std::chrono::steady_clock::now() : tp{};
            const auto t_op = (time_dbg_ || sample_time_) ? std::chrono::steady_clock::now() : tp{};
            const auto t_sub = ophost_dbg_ ? std::chrono::steady_clock::now() : tp{};
            // The safety net: an operator with no device path for an integer
            // gets the host bytes its code expects.  The alternative is an
            // empty vector, which is not an error - every index comes out as
            // zero and the model still produces plausible text.  A Shape is
            // exempt (it reads only the shape), and a capable operator took
            // the device path above.
            if (!skip_exec && !(int_dev_capable(op.node->op_type) || cmp64_dev) &&
                !int_shape_only(op.node->op_type)) {
                for (int s : op.in) {
                    if (s < 0) continue;
                    if (int_dtype(g_.tensors[s].dtype) && state_[s].dev && host_of(s).empty()) {
                        if (intdev_dbg_) {
                            fprintf(stderr, "[onnx] intdev: the host path of %zu (%s) reads '%s'\n", i,
                                    op.node->op_type.c_str(), g_.tensors[s].name.c_str());
                        }
                        materialize_int(s);
                    }
                }
            }
            if (!skip_exec) e->exec(ctx);
            th2 = timed ? std::chrono::steady_clock::now() : tp{};
            if (ophost_dbg_) {
                // The time spent *submitting* the operator, with no wait: a
                // host operator that downloads a tensor shows up here, and
                // that is the only way to see the queue drains that hide
                // inside e->exec.
                fprintf(stderr, "[onnx] ophost: %zu %s %.3f\n", i,
                        op.node->op_type.c_str(),
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_sub).count());
            }
            // The wait: only the diagnostics need the operator's result (the
            // timing, the dump and the guard band's check).  The queue is in
            // order, so the next operator's kernels see this one's writes -
            // and a wait per operator costs the pipeline its depth, which
            // on a model with thousands of small operators is most of the
            // time.  (The host operators read the device themselves, and
            // `q_download` waits.)
            if (guard_ || time_dbg_ || sample_time_ || dump_dbg_ || dump_file_dbg_) {
                q_.wait();
            }
            if (time_dbg_ || sample_time_) {
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t_op).count();
                fprintf(stderr, "[onnx] time: op %zu %s %s %.3f ms\n", i,
                        op.node->op_type.c_str(), op.node->name.c_str(), ms);
            }
            if (dump_dbg_) {
                // The outputs are dumped *after* the operator: the input's
                // buffer may have been overwritten by it, and the output's
                // buffer does not exist before it.
                const size_t nvals = (size_t)dump_nvals_;
                for (size_t k = 0; k < ctx.out.size(); ++k) {
                    const size_t n = std::min<size_t>(ctx.out[k]->count, std::min<size_t>(nvals, 64));
                    // An operator that computes on the host (an integer
                    // value, a comparison of two integers) fills the *host*
                    // copy and leaves the device buffer alone: reading the
                    // device there shows whatever was in the pool.
                    const bool on_host = ctx.out[k]->on_host && !ctx.out[k]->host.empty();
                    // An integer value is read as an integer, in its own
                    // width: the device holds float32, and reading an int32
                    // buffer as int64 gives the bits of the next element too.
                    const bool ints = ctx.out[k]->dtype == onnxpb::INT64 ||
                                       ctx.out[k]->dtype == onnxpb::INT32;
                    if (ints) {
                        const size_t esz = ctx.out[k]->dtype == onnxpb::INT64 ? 8 : 4;
                        uint8_t raw[64 * 8] = {0};
                        if (n) {
                            if (on_host) {
                                std::memcpy(raw, ctx.out[k]->host.data(),
                                            std::min<size_t>(n * esz, ctx.out[k]->host.size()));
                            } else {
                                q_download(q_, ctx.out[k]->dev, raw, n * esz);
                            }
                        }
                        fprintf(stderr, "[onnx] op %zu %s out[%zu] %s [i%zu]:", i,
                                op.node->op_type.c_str(), k, ctx.out[k]->name.c_str(), esz * 8);
                        for (size_t b = 0; b < n; ++b) {
                            if (esz == 8) {
                                int64_t x = 0;
                                std::memcpy(&x, raw + b * 8, 8);
                                fprintf(stderr, " %lld", (long long)x);
                            } else {
                                int32_t x = 0;
                                std::memcpy(&x, raw + b * 4, 4);
                                fprintf(stderr, " %d", (int)x);
                            }
                        }
                        fprintf(stderr, "\n");
                        continue;
                    }
                    float v[64] = {0};
                    if (n) {
                        if (on_host) {
                            const size_t have = ctx.out[k]->host.size() / 4;
                            std::memcpy(v, ctx.out[k]->host.data(),
                                        std::min<size_t>(n, have) * 4);
                        } else {
                            q_download(q_, ctx.out[k]->dev, v, n * 4);
                        }
                    }
                    fprintf(stderr, "[onnx] op %zu %s out[%zu] %s:", i, op.node->op_type.c_str(), k,
                            ctx.out[k]->name.c_str());
                    for (size_t b = 0; b < n; ++b) fprintf(stderr, " %.9g", v[b]);
                    fprintf(stderr, "\n");
                }
            }
            if (dump_file_dbg_) {
                // The *whole* output, so the values can be compared with
                // ONNXRuntime's element by element.  The first few values
                // are not enough: a divergence past them is invisible.
                for (size_t k = 0; k < ctx.out.size(); ++k) {
                    const std::string p = std::string(dump_file_dir_) + "/" +
                                          dump_tag_ + std::to_string(i) + "_" +
                                          ctx.out[k]->name + ".bin";
                    const size_t n = ctx.out[k]->count;
                    const size_t esz = ctx.out[k]->dtype == onnxpb::INT64 ? 8 : 4;
                    std::vector<uint8_t> raw;
                    if (ctx.out[k]->dtype == onnxpb::INT64 || ctx.out[k]->dtype == onnxpb::INT32) {
                        // An integer value lives on the *host*: the operator
                        // that computed it on the host put it there, and the
                        // write-back (which copies it into the graph's
                        // tensor) has not run yet.  An operator that computed
                        // it on the *device* has no host copy yet - the
                        // device buffer is the value.
                        raw = ctx.out[k]->host;
                        if (raw.empty() && ctx.out[k]->dev && n) {
                            raw.assign(n * esz, 0);
                            q_download(q_, ctx.out[k]->dev, raw.data(), raw.size());
                        }
                    } else {
                        raw.assign(n * esz, 0);
                        if (n) q_download(q_, ctx.out[k]->dev, raw.data(), raw.size());
                    }
                    FILE * f = fopen(p.c_str(), "wb");
                    if (!f) continue;
                    int64_t shape[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (size_t d = 0; d < ctx.out[k]->shape.size() && d < 8; ++d) {
                        shape[d] = ctx.out[k]->shape[d];
                    }
                    fwrite(shape, sizeof(int64_t), 8, f);
                    const int64_t cnt = (int64_t)n;
                    fwrite(&cnt, sizeof(int64_t), 1, f);
                    const int64_t dt = ctx.out[k]->dtype;
                    fwrite(&dt, sizeof(int64_t), 1, f);
                    fwrite(raw.data(), 1, raw.size(), f);
                    fclose(f);
                }
            }
        } catch (const std::exception & ex) {
            // Name the operator: a bare SYCL error says nothing about where
            // it came from.
            std::string what = "the operator " + std::to_string(i) + " (" + op.node->op_type +
                               ") failed: " + ex.what();
            for (size_t k = 0; k < op.in.size(); ++k) {
                what += " | in[" + std::to_string(k) + "] ";
                if (op.in[k] < 0) {
                    what += "(none)";
                    continue;
                }
                what += g_.tensors[op.in[k]].name;
                what += " dtype=" + std::to_string((int)g_.tensors[op.in[k]].dtype);
                what += " count=" + std::to_string(g_.tensors[op.in[k]].count);
                what += state_[op.in[k]].dev ? " dev=yes" : " dev=NULL";
            }
            report_leak();
            throw std::runtime_error(what);
        }
        for (size_t k = 0; k < ctx.out.size(); ++k) {
            if (kGuard) {
                // The guard is read back with a copy, for the same reason.
                std::vector<uint8_t> guard(kGuard);
                q_download(q_, (const uint8_t *)ctx.out[k]->dev + ctx.out[k]->bytes(), guard.data(),
                           guard.size());
                for (size_t b = 0; b < guard.size(); ++b) {
                    if (guard[b] != 0xAB) {
                        fprintf(stderr,
                                "[onnx] op %zu %s wrote past the output '%s' (the count %zu, the bytes "
                                "%zu, the dtype %d, the guard[%zu] = 0x%02x)\n",
                                i, op.node->op_type.c_str(), ctx.out[k]->name.c_str(),
                                ctx.out[k]->count, ctx.out[k]->bytes(), (int)ctx.out[k]->dtype, b,
                                guard[b]);
                        break;
                    }
                }
            }
            const int s = op.out[k];
            slot_state & st = state_[s];
            // Where the value lives.
            //
            // A value that the operator wrote on the host stays there.  An
            // *integer* value lives on the host whatever the operator did:
            // the shape functions read it there, and they run before the
            // operator does - so a value that is only on the device is read
            // as empty, and the shape arithmetic downstream is wrong.
            //
            // Everything else stays on the device: a host copy of every
            // intermediate would be a round trip per operator.
            const bool int_out = ctx.out[k]->dtype == onnxpb::INT64 ||
                                 ctx.out[k]->dtype == onnxpb::INT32;
            // A graph output with a configured reduction keeps its value on
            // the device: the readback reduces it there and transfers only
            // the result.  The layout detector's instance mask is an int32
            // output, and the generic integer rule below would otherwise
            // download all 48 MB/page before the readback ever sees it.
            const bool reduced_out =
                is_output(s) && out_reduce_.count(g_.tensors[s].name) != 0;
            // An integer output whose operator wrote it on the device stays
            // there: the value is device-authoritative and `host` is empty.
            // The old rule made every integer host-authoritative and
            // downloaded it here, and the download waits for the whole
            // in-order queue - the layout model's three TopK outputs alone
            // blocked 1.3 s per batch-16 call behind convolutions the host
            // could have submitted past.  A consumer that needs the bytes
            // (a shape parameter, or an operator with no device path)
            // materialises them at its own point of use.
            const bool on_host = ctx.out[k]->on_host && !reduced_out;
            // A value is written once per run.  The check used to be in the
            // device-resident branch only, so a host value written twice
            // silently kept the last writer's bytes and the first writer's
            // consumer read them.
            if (producer_[s] >= 0) {
                throw std::runtime_error("the value '" + g_.tensors[s].name +
                                         "' is written twice (the operators " +
                                         std::to_string(producer_[s]) + " and " +
                                         std::to_string(i) + ": " +
                                         g_.ops[producer_[s]].node->op_type + " and " +
                                         op.node->op_type + ")");
            }
            producer_[s] = (int)i;
            if (ctx.out[k]->on_host) {
                g_.tensors[s].host = ctx.out[k]->host;
            } else {
                g_.tensors[s].host.clear();
            }
            if (intdev_dbg_ && int_out) {
                fprintf(stderr, "[onnx] intout: %s %s %s\n", on_host ? "host" : "dev",
                        op.node->op_type.c_str(), g_.tensors[s].name.c_str());
            }
            st.is_const = false;
            st.bytes = ctx.out[k]->bytes();
            g_.tensors[s].on_host = on_host;
            g_.tensors[s].count = ctx.out[k]->count;
            g_.tensors[s].shape = ctx.out[k]->shape;
            // The type too: the next operator reads its input's type, and an
            // int64 value (the dimensions) has to be recognised as one.
            g_.tensors[s].dtype = ctx.out[k]->dtype;
            // The host bytes (if any) are new: a device copy made from the
            // previous ones is stale until it is uploaded again.
            ++st.host_gen;
            if (on_host) {
                // Keep the operator's buffer as the device *mirror* instead
                // of releasing it.  An integer the device computed was
                // downloaded into the host copy, but the buffer still holds
                // the same value - so the next operator that reads it on the
                // device reuses it, where releasing the buffer made every
                // consumer upload it through a fresh allocation.  A
                // host-computed value's buffer was never written, so its
                // generation says it is stale and the next consumer uploads
                // into it.
                if (ctx.out[k]->dev) {
                    if (st.dev && st.dev != ctx.out[k]->dev) {
                        release(st.dev, st.bytes);
                    }
                    st.dev = ctx.out[k]->dev;
                    st.dev_gen = (int_out && !ctx.out[k]->on_host) ? st.host_gen : st.host_gen - 1;
                } else {
                    // The operator allocated no device buffer.  Keep the
                    // slot's old allocation as the upload target if it is
                    // large enough; a smaller one is replaced.
                    auto it = st.dev ? alloc_size_.find(st.dev) : alloc_size_.end();
                    if (st.dev && (it == alloc_size_.end() || it->second < ctx.out[k]->bytes())) {
                        release(st.dev, st.bytes);
                        st.dev = nullptr;
                    }
                    st.dev_gen = st.host_gen - 1;
                }
                g_.tensors[s].dev = st.dev;
            } else {
                // The device pointer has to go back into the graph's tensor:
                // the next operator reads its inputs through *that* pointer,
                // and the operator wrote through its own copy of it.
                // The slot's old buffer goes back to the pool (see the
                // folded-constants path above).
                if (st.dev && st.dev != ctx.out[k]->dev) {
                    release(st.dev, st.bytes);
                }
                g_.tensors[s].dev = ctx.out[k]->dev;
                st.dev = ctx.out[k]->dev;
                // The device buffer *is* the value, and there are no host
                // bytes for a mirror to disagree with.
                st.dev_gen = st.host_gen;
                if (trace_) {
                    fprintf(stderr, "[onnx] the op %zu %s writes the slot %d ('%s')\n", i,
                            op.node->op_type.c_str(), s, g_.tensors[s].name.c_str());
                }
            }
            if (trace_on() && op.node->op_type == "Concat") {
                fprintf(stderr,
                        "[onnx] the concat's output '%s': the dtype %d, the host %zu, on_host %d\n",
                        ctx.out[k]->name.c_str(), (int)ctx.out[k]->dtype, ctx.out[k]->host.size(),
                        (int)ctx.out[k]->on_host);
            }
        }
        if (hosttime) {
            const auto th3 = std::chrono::steady_clock::now();
            ht_prepare += std::chrono::duration<double, std::milli>(th1 - th0).count();
            ht_submit += std::chrono::duration<double, std::milli>(th2 - th1).count();
            ht_writeback += std::chrono::duration<double, std::milli>(th3 - th2).count();
            ht_n++;
        }
        if (opstat_) {
            const double dt = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - th0).count();
            std::string key = op.node->op_type;
            if (key == "Cast" && !op.in.empty() && op.in[0] >= 0) {
                const tensor & ti = g_.tensors[op.in[0]];
                key += "_" + std::to_string((int)ti.dtype) + "_" + std::to_string((int)ctx.out[0]->dtype) +
                       "_n" + std::to_string(ti.count) + (ti.on_host ? "_h" : "_d");
                if (ti.count <= 4) key += "(" + ti.name.substr(0, 18) + ")";
            }
            op_ms_[key] += dt;
            ++op_n_[key];
            op_dl_[key] += g_q_downloads.load() - dl_op0;
            op_up_[key] += (long)(upload_count_ - up_op0);
        }
        // The input buffers that are dead.  The graph's outputs are not:
        // they are what the client asked for, and they are read back at the
        // end of the run.
        for (int s : op.in) {
            if (s < 0) continue;
            // A constant never changes, so its device mirror is kept for the
            // process instead of being released and uploaded again on the
            // next operator (or the next call).
            if (state_[s].last_use == (int)i && state_[s].dev && !is_output(s) &&
                !state_[s].is_const && !state_[s].pinned) {
                if (trace_) {
                    fprintf(stderr, "[onnx] the release: op %zu releases '%s' (%zu bytes)\n", i,
                            g_.tensors[s].name.c_str(), state_[s].bytes);
                }
                release(state_[s].dev, state_[s].bytes);
                state_[s].dev = nullptr;
            }
        }
        if (const_chk_) {
            // The constants' device copies are checked after every operator:
            // one of them was overwritten (by a kernel that wrote past its
            // output, or by a buffer that two values shared), and the
            // operator that did it is the one that has to be fixed.
            for (size_t s = 0; s < state_.size(); ++s) {
                if (!state_[s].is_const || !state_[s].dev) continue;
                if (host_of((int)s).empty()) continue;
                std::vector<uint8_t> now(host_of((int)s).size());
                q_download(q_, state_[s].dev, now.data(), now.size());
                if (now != host_of((int)s)) {
                    throw std::runtime_error("the constant '" + g_.tensors[s].name +
                                             "' was overwritten by the operator " + std::to_string(i) +
                                             " (" + op.node->op_type + ")");
                }
            }
        }
    }

    // The graph's outputs are read back to the host: they are what the
    // client asked for.  A reduced output transfers only its reduction (see
    // set_output_reduce): the tensor itself never crosses the bus.
    //
    // **Only the outputs the caller asked for are copied.**  The formula
    // recogniser's Loop body declares 62 outputs, and 12 of them are the
    // growing KV caches (5.2 MB each): the body used to copy every one of
    // them to the host on every iteration (65 MB per iteration, 166 GB over
    // a 2 560-step decode) even though the loop only requested the
    // condition and a few small integer carried values.  The copies are
    // pageable, which on this backend is synchronous - so each one also
    // drained the in-order queue and serialised host dispatch against the
    // device.  `outputs` is the request list for this run; a body run asks
    // for the condition and the host-route carried values only (the
    // device-chained ones never cross the bus), which is the difference
    // between a body iteration dominated by PCIe and one dominated by its
    // kernels.
    reduced_.assign(g_.output_slots.size(), {});
    std::vector<char> want_out(g_.output_slots.size(), 0);
    for (const auto & o : outputs) {
        auto it = std::find_if(g_.output_slots.begin(), g_.output_slots.end(),
                               [&](int s) { return s >= 0 && g_.tensors[s].name == o.first; });
        if (it != g_.output_slots.end()) want_out[(size_t)(it - g_.output_slots.begin())] = 1;
    }
    const auto tr0 = std::chrono::steady_clock::now();
    for (size_t oi = 0; oi < g_.output_slots.size(); ++oi) {
        if (!want_out[oi]) continue;
        const int s = g_.output_slots[oi];
        if (s < 0 || s >= (int)state_.size()) continue;
        // An output the client asked not to receive is not read back: the
        // device buffer stays where the last operator left it, and the
        // 48 MB mask of the layout model never crosses the bus.
        if (skip_out_.count(s)) continue;
        slot_state & st = state_[s];
        const auto rit = out_reduce_.find(g_.tensors[s].name);
        if (rit != out_reduce_.end() && st.dev && !g_.tensors[s].on_host && st.bytes) {
            const std::vector<int64_t> & shp = g_.tensors[s].shape;
            const int64_t dt = g_.tensors[s].dtype;
            if (rit->second == kReduceArgmax && dt == onnxpb::FLOAT) {
                const int64_t V = shp.empty() ? 1 : shp.back();
                const size_t outer = V > 0 ? g_.tensors[s].count / (size_t)V : 0;
                if (V > 0 && outer > 0) {
                    void * d_idx = acquire(outer * 8);
                    void * d_max = acquire(outer * 4);
                    if (!d_idx || !d_max) {
                        if (d_idx) release(d_idx, outer * 8);
                        if (d_max) release(d_max, outer * 4);
                        throw std::runtime_error("the device allocation for the argmax of '" +
                                                 g_.tensors[s].name + "' failed");
                    }
                    argmax_max_launch(q_, (const float *)st.dev, (int64_t *)d_idx,
                                      (float *)d_max, (int64_t)outer, V);
                    reduced_output ri, rv;
                    ri.shape.assign(shp.begin(), shp.end() - 1);
                    ri.dtype = onnxpb::INT64;
                    ri.host.resize(outer * 8);
                    rv.shape = ri.shape;
                    rv.dtype = onnxpb::FLOAT;
                    rv.host.resize(outer * 4);
                    q_download(q_, d_idx, ri.host.data(), ri.host.size());
                    q_download(q_, d_max, rv.host.data(), rv.host.size());
                    release(d_idx, outer * 8);
                    release(d_max, outer * 4);
                    reduced_[oi].push_back(std::move(ri));
                    reduced_[oi].push_back(std::move(rv));
                    continue;
                }
            } else if (rit->second == kReduceU8) {
                const size_t n = g_.tensors[s].count;
                void * d = acquire(n ? n : 1);
                if (!d) {
                    throw std::runtime_error("the device allocation for the byte cast of '" +
                                             g_.tensors[s].name + "' failed");
                }
                const int in_kind = dt == onnxpb::INT64  ? kCastI64
                                    : dt == onnxpb::INT32 ? kCastI32
                                    : dt == onnxpb::BOOL ? kCastBool
                                                          : kCastF32;
                cast_launch(q_, st.dev, d, n, in_kind, kCastU8);
                reduced_output r;
                r.shape = shp;
                r.dtype = onnxpb::UINT8;
                r.host.resize(n);
                q_download(q_, d, r.host.data(), r.host.size());
                release(d, n ? n : 1);
                reduced_[oi].push_back(std::move(r));
                continue;
            }
        }
        if (st.dev && g_.tensors[s].host.empty()) {
            g_.tensors[s].host.resize(st.bytes);
            // Enqueue every output copy and wait once.  `q_download` waits on
            // the whole (in-order) queue per call, and a Loop-body run reads
            // back 31 carried values every iteration: 31 full drains for one
            // iteration's work, 27 times over, was most of the table
            // structurer's decoder.  The copies are ordered against the
            // producers by the queue itself, so one wait at the end is enough.
            q_.memcpy(g_.tensors[s].host.data(), st.dev, g_.tensors[s].host.size());
        }
    }
    rb_ms_ += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - tr0).count();
    {
        const auto w0 = std::chrono::steady_clock::now();
        q_.wait();
        wait_ms_ += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - w0).count();
    }
    // A Loop body's free pool is the loop's speed, so it is not trimmed
    // during the loop.  A top-level call that has finished, however, should
    // give the memory back: the document pipeline runs the layout, OCR and
    // table models on the same card, and the formula decoder's pool is the
    // one that grows to gigabytes.  The 512 MiB gate keeps short formulas
    // (most real crops) from paying a trim at all.
    if (!body_ && has_loop_ && pooled_bytes_calc() > (512ull << 20)) trim_pool();
    // The queue is drained: the buffers released past the pool cap are safe
    // to free now.  A few per run (the net growth), not a whole-pool trim.
    if (!free_later_.empty()) {
        for (void * p : free_later_) {
            auto it = alloc_size_.find(p);
            if (it != alloc_size_.end()) {
                sycl::free(p, q_);
                alloc_size_.erase(it);
            }
        }
        free_later_.clear();
    }

    report_leak();

    // The outputs.
    for (const auto & o : outputs) {
        auto it = std::find_if(g_.output_slots.begin(), g_.output_slots.end(),
                               [&](int s) { return g_.tensors[s].name == o.first; });
        if (it == g_.output_slots.end()) {
            throw std::runtime_error("the graph has no output called " + o.first);
        }
        o.second = std::move(g_.tensors[*it].host);
    }
    calls_++;
    if (upload_dbg_) {
        // Which values are copied to the device again and again: the totals
        // alone say *how much* a call uploads, not *what* is being repeated.
        std::vector<int> order;
        for (size_t s = 0; s < upload_by_slot_.size(); ++s) {
            if (upload_by_slot_[s]) order.push_back((int)s);
        }
        std::sort(order.begin(), order.end(), [this](int a, int b) {
            return upload_bytes_by_slot_[a] > upload_bytes_by_slot_[b];
        });
        static const int topn = [] {
            const char * e = getenv("SI_ONNX_UPLOADTOPN");
            return e && *e ? atoi(e) : 12;
        }();
        static const bool by_count = getenv("SI_ONNX_UPLOADCOUNT") != nullptr;
        if (by_count) {
            std::sort(order.begin(), order.end(), [this](int a, int b) {
                return upload_by_slot_[a] > upload_by_slot_[b];
            });
        }
        if (topn > 0 && order.size() > (size_t)topn) order.resize((size_t)topn);
        fprintf(stderr, "[onnx] uploads: %zu copies, %.2f MB", upload_count_,
                upload_bytes_ / 1048576.0);
        for (int s : order) {
            fprintf(stderr, " | '%s' %u x %.1f KB", g_.tensors[s].name.c_str(),
                    (unsigned)upload_by_slot_[s], upload_bytes_by_slot_[s] / 1024.0);
        }
        fprintf(stderr, "\n");
    }
    if (hosttime) {
        const double total = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr,
                "[onnx] host: %zu ops: prepare %.1f ms (the upload %.1f over %zu uploads "
                "(%.1f MB), the allocation %.1f, the rest %.1f; %zu pool misses), "
                "submit %.1f ms, write-back %.1f ms, the other %.1f ms of %.1f ms\n",
                ht_n, ht_prepare, ht_upload, upload_count_, upload_bytes_ / 1048576.0, ht_alloc,
                ht_ctx, ht_n_upload, ht_submit, ht_writeback,
                total - ht_prepare - ht_submit - ht_writeback, total);
    }
    if (getenv("SI_ONNX_MEMDBG")) {
        size_t total = 0;
        for (const auto & kv : alloc_size_) total += kv.second;
        // The buffers that are neither live nor in the free pool: they are
        // allocated (the driver sees them) but nothing can reuse them.
        std::set<void *> pooled;
        for (const auto & kv : free_) {
            for (void * p : kv.second) pooled.insert(p);
        }
        size_t lost = 0, lost_n = 0;
        std::map<size_t, size_t> lost_sz;
        for (const auto & kv : alloc_size_) {
            if (live_.count(kv.first) || pooled.count(kv.first)) continue;
            lost += kv.second;
            ++lost_n;
            ++lost_sz[kv.second];
        }
        fprintf(stderr,
                "[onnx] mem: %zu allocations, %zu bytes total (%zu mallocs), live %zu (peak %zu), "
                "pooled %zu, lost %zu in %zu buffers\n",
                alloc_size_.size(), total, malloc_count_, live_bytes_calc(), peak_live_,
                pooled_bytes_calc(), lost, lost_n);
        std::vector<std::pair<size_t, size_t>> pool_sizes;
        for (const auto & kv : free_) pool_sizes.push_back({kv.first, kv.second.size()});
        std::sort(pool_sizes.begin(), pool_sizes.end(),
                  [](auto & a, auto & b) { return a.first * a.second > b.first * b.second; });
        for (size_t i = 0; i < pool_sizes.size() && i < 8; ++i) {
            fprintf(stderr, "[onnx]   pool: %zu bytes x %zu\n", pool_sizes[i].first,
                    pool_sizes[i].second);
        }
        size_t dw = 0, ds = 0;
        conv_dnnl_mem(&dw, &ds);
        fprintf(stderr, "[onnx]   oneDNN: reordered weights %zu bytes, scratch %zu bytes\n", dw, ds);
        for (const auto & kv : lost_sz) {
            fprintf(stderr, "[onnx]   lost: %zu bytes x %zu\n", kv.first, kv.second);
        }
        // Which slot holds each lost buffer (the largest by bytes).
        std::vector<std::pair<size_t, int>> held;
        for (const auto & kv : alloc_size_) {
            if (live_.count(kv.first) || pooled.count(kv.first)) continue;
            int owner = -1;
            for (size_t s = 0; s < state_.size(); ++s) {
                if (state_[s].dev == kv.first) { owner = (int)s; break; }
            }
            held.push_back({kv.second, owner});
        }
        std::sort(held.begin(), held.end(), [](auto & a, auto & b) { return a.first > b.first; });
        for (size_t i = 0; i < held.size() && i < 12; ++i) {
            const int s = held[i].second;
            fprintf(stderr, "[onnx]   held %zu bytes: slot %d %s const %d last_use %d\n", held[i].first,
                    s, s >= 0 ? g_.tensors[s].name.c_str() : "(no slot)", s >= 0 ? (int)state_[s].is_const : -1,
                    s >= 0 ? state_[s].last_use : -1);
        }
    }
    last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace onnx
} // namespace si
