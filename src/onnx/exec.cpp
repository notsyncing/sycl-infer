#include "exec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "kernels.h"
#include "ops.h"

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
    if (dst.state_[dst_slot].dev) {
        dst.release(dst.state_[dst_slot].dev, dst.state_[dst_slot].bytes);
        dst.state_[dst_slot].dev = nullptr;
    }
}

executor::~executor() {    for (auto & kv : free_) {
        for (void * p : kv.second) {
            if (p) sycl::free(p, q_);
        }
    }
    for (auto & s : state_) {
        if (s.dev) sycl::free(s.dev, q_);
    }
    if (ws_) sycl::free(ws_, q_);
    if (ws2_) sycl::free(ws2_, q_);
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
    return n;
}

size_t executor::pooled_bytes_calc() const {
    size_t n = 0;
    for (const auto & kv : free_) n += kv.first * kv.second.size();
    return n;
}

size_t executor::free_mem_bytes() const {
    // The device does not report its free memory through the standard
    // queries, so the pool's own accounting is used: the live allocations
    // plus the ones the other models hold are not visible here, but the
    // number still says whether *this* graph is the one that ran out.
    size_t live = 0;
    for (const auto & kv : alloc_size_) live += kv.second;
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
    // The best fit: the smallest free buffer that is big enough.
    for (auto it = free_.begin(); it != free_.end(); ++it) {
        if (it->first >= bytes && !it->second.empty()) {
            void * p = it->second.back();
            it->second.pop_back();
            if (live_.count(p)) {
                throw std::runtime_error("the memory pool handed out a live buffer twice");
            }
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
    void * p = sycl::malloc_device(bytes, q_);
    if (p) {
        alloc_size_[p] = bytes;
        if (!live_.insert(p).second) {
            throw std::runtime_error("the memory pool handed out a live buffer twice");
        }
    }
    return p;
}

void executor::release(void * p, size_t bytes) {
    (void)bytes;
    if (!p) return;
    if (getenv("SI_ONNX_LOOPDBG")) {
        fprintf(stderr, "[onnx] the release: %p (%zu bytes)\n", p, bytes);
    }
    // The buffer goes back under its *real* size, not the size this caller
    // asked for.  The record stays: the buffer is handed out again later,
    // and the next release has to find the same real size.
    const size_t real = alloc_size_.count(p) ? alloc_size_[p] : bytes;
    // It must not be in the pool already.  A double release hands the same
    // buffer to two different values, and the second one overwrites the
    // first - which is how a constant's value became another constant's.
    for (const auto & kv : free_) {
        for (void * q : kv.second) {
            if (q == p) {
                throw std::runtime_error("a device buffer was released twice");
            }
        }
    }
    live_.erase(p);
    free_[real].push_back(p);
}

// The scratch buffer for the operators' small arrays.
void * executor::scratch() {
    if (!ws_) {
        // 4 MiB: the top-k's per-row sorted list (the values and the
        // indices) and the indices themselves live here.
        ws_bytes_ = 4 << 20;
        ws_ = sycl::malloc_device(ws_bytes_, q_);
    }
    return ws_;
}

void * executor::scratch_small() {
    if (!ws2_) {
        // 64 KB: the operators that convert a value and put it on the
        // device (a mixed-type elementwise) use this.
        ws2_bytes_ = 64 << 10;
        ws2_ = sycl::malloc_device(ws2_bytes_, q_);
    }
    return ws2_;
}

bool executor::is_output(int slot) const {
    for (int s : g_.output_slots) {
        if (s == slot) return true;
    }
    return false;
}

const float * executor::device_value(int slot) {
    slot_state & s = state_[slot];
    if (s.dev) return (const float *)s.dev;
    std::vector<uint8_t> & h = host_of(slot);
    if (h.empty()) {
        throw std::runtime_error("value " + g_.tensors[slot].name + " is empty");
    }
    s.dev = acquire(h.size());
    if (!s.dev) throw std::runtime_error("device allocation failed");
    q_upload(q_, h.data(), s.dev, h.size());
    return (const float *)s.dev;
}

void executor::prepare_inputs(exec_op & op) {
    for (int s : op.in) {
        if (s < 0 || s >= (int)state_.size()) continue;
        if (state_[s].dev) continue;
        std::vector<uint8_t> & h = host_of(s);
        if (h.empty()) {
            // An empty constant (a zero-element tensor) is legal: the
            // operator has to know that it is empty.
            if (g_.tensors[s].count == 0) continue;
            throw std::runtime_error("the input '" + g_.tensors[s].name + "' has no value (the slot " +
                                     std::to_string(s) + ", the dtype " +
                                     std::to_string((int)g_.tensors[s].dtype) + ", the count " +
                                     std::to_string(g_.tensors[s].count) + ", the const " +
                                     std::to_string(state_[s].is_const) + ", the dev " +
                                     std::to_string(state_[s].dev != nullptr) + ")");
        }
        state_[s].dev = acquire(h.size());
        if (!state_[s].dev) throw std::runtime_error("device allocation failed");
        q_upload(q_, h.data(), state_[s].dev, h.size());
    }
}

// The constant sub-graphs are evaluated once, on the device, the first time
// the graph runs; their results are kept as constants.
void executor::fold_constants() {
    for (size_t i = 0; i < g_.ops.size(); ++i) {
        exec_op & op = g_.ops[i];
        if (op.node->op_type == "Constant") {
            // A constant inside an If's branch is NOT folded: the two
            // branches write the *same* output slot (the If's output),
            // so a folded value is destroyed by the other branch's
            // operator the first time that branch runs - and the next
            // time *this* branch is selected, the slot holds nothing.
            //
            // The branch's constant therefore runs as an operator, in
            // the selected branch only.
            if (op.in_if_branch) continue;
            // The value is an attribute.
            for (const onnxpb::attribute & a : op.node->attribute) {
                if (a.name != "value" || a.type_ != onnxpb::attribute::type::t) continue;
                for (size_t k = 0; k < op.out.size(); ++k) {
                    slot_state & st = state_[op.out[k]];
                    // A boolean is a float 0/1 in this runtime, so its
                    // bytes are converted from the model's one-byte form.
                    g_.tensors[op.out[k]].host = a.t.data_type == onnxpb::BOOL
                                                    ? bool_to_runtime_bytes(a.t.raw)
                                                    : a.t.raw;
                    st.is_const = true;
                    g_.tensors[op.out[k]].on_host = true;
                    g_.tensors[op.out[k]].dtype = a.t.data_type;
                    g_.tensors[op.out[k]].shape = a.t.dims;
                    size_t n = 1;
                    for (int64_t d : a.t.dims) n *= (size_t)d;
                    g_.tensors[op.out[k]].count = n;
                    st.bytes = g_.tensors[op.out[k]].host.size();
                }
            }
            continue;
        }
        // An operator inside an If's branch: it is not folded, because only
        // one of the two branches runs.
        if (op.in_if_branch) continue;
        // An operator whose inputs are all constants.
        bool all_const = !op.in.empty();
        for (int s : op.in) {
            if (s < 0) continue;
            if (!state_[s].is_const) {
                all_const = false;
                break;
            }
        }
        if (!all_const) {
            if (getenv("SI_ONNX_LOOPDBG")) {
                fprintf(stderr, "[onnx] the fold: the op %zu %s is not const:", i,
                        op.node->op_type.c_str());
                for (int s : op.in) {
                    if (s < 0) continue;
                    fprintf(stderr, " '%s'(const=%d)", g_.tensors[s].name.c_str(),
                            (int)state_[s].is_const);
                }
                fprintf(stderr, "\n");
            }
            continue;
        }

        if (trace_on()) {
            fprintf(stderr, "[onnx] fold %zu %s\n", i, op.node->op_type.c_str());
        }
        op_ctx ctx(q_, *op.node);
        ctx.opset = g_.opset;
        std::vector<std::vector<int64_t>> shapes;
        for (int s : op.in) {
            ctx.in.push_back(&g_.tensors[s]);
        }
        // The output tensors live in a vector that outlives the call: a
        // pointer to a stack temporary would dangle the moment the loop
        // iteration ends, and every write through it (the device pointer, the
        // shape, the count) would land on freed stack.
        std::vector<tensor> out_tensors;
        out_tensors.reserve(op.out.size());
        for (size_t k = 0; k < op.out.size(); ++k) {
            tensor t;
            t.name = g_.tensors[op.out[k]].name;
            t.dtype = g_.tensors[op.out[k]].dtype;
            out_tensors.push_back(t);
        }
        for (tensor & t : out_tensors) ctx.out.push_back(&t);

        const op_entry * e = find_op(op.node->op_type);
        if (!e) continue;
        for (tensor * t : ctx.out) {
            if (!ctx.in.empty()) t->dtype = ctx.in[0]->dtype;
        }
        if (!e->shape(ctx, shapes)) {
            if (getenv("SI_ONNX_TRACE")) {
                fprintf(stderr, "[onnx] the fold: the op %zu %s has no shape\n", i,
                        op.node->op_type.c_str());
            }
            continue;
        }
        for (size_t k = 0; k < ctx.out.size() && k < shapes.size(); ++k) {
            ctx.out[k]->shape = shapes[k];
            ctx.out[k]->count = (size_t)std::max<int64_t>(0, shape_count(shapes[k]));
            // A count of zero is a count of one: the value is a scalar.  The
            // main loop says so; without it here a folded value has no bytes
            // at all, and the operator that reads it finds nothing.
            if (ctx.out[k]->count == 0) ctx.out[k]->count = 1;
        }
        // The device buffers for the inputs and the outputs.  The outputs get
        // a guard band like the main path's: an operator that writes past
        // its output corrupts whatever the pool handed it next - and in the
        // folder that is a *constant*, whose value then changes under the
        // model's feet.
        static const size_t kGuardFold =
            (size_t)atoi(getenv("SI_ONNX_GUARD") ? getenv("SI_ONNX_GUARD") : "256");
        std::vector<void *> in_dev(op.in.size(), nullptr);
        for (size_t k = 0; k < op.in.size(); ++k) {
            const int s = op.in[k];
            const size_t bytes = host_of(s).size();
            in_dev[k] = acquire(bytes);
            if (in_dev[k]) q_upload(q_, host_of(s).data(), in_dev[k], bytes);
        }
        std::vector<void *> out_dev(ctx.out.size(), nullptr);
        for (size_t k = 0; k < ctx.out.size(); ++k) {
            out_dev[k] = acquire(ctx.out[k]->bytes() + kGuardFold);
            if (out_dev[k]) {
                q_.memset((uint8_t *)out_dev[k] + ctx.out[k]->bytes(), 0xAB, kGuardFold).wait();
            }
            ctx.out[k]->dev = out_dev[k];
        }
        for (size_t k = 0; k < op.in.size(); ++k) {
            if (k < ctx.in.size() && in_dev[k]) ctx.in[k]->dev = in_dev[k];
        }
        try {
            e->exec(ctx);
            q_.wait();
            // The guard band: a folded operator that writes past its output
            // would corrupt the next allocation.
            for (size_t k = 0; k < ctx.out.size(); ++k) {
                if (!ctx.out[k]->dev) continue;
                std::vector<uint8_t> guard(kGuardFold);
                q_download(q_, (const uint8_t *)ctx.out[k]->dev + ctx.out[k]->bytes(), guard.data(),
                           guard.size());
                for (size_t b = 0; b < guard.size(); ++b) {
                    if (guard[b] != 0xAB) {
                        throw std::runtime_error("the folded operator " + std::to_string(i) + " (" +
                                                 op.node->op_type + ") wrote past the output '" +
                                                 ctx.out[k]->name + "'");
                    }
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
        // The result: a host copy, and the device buffer stays alive (it holds
        // the constant's value, and the pool would hand it to somebody
        // else).
        for (size_t k = 0; k < ctx.out.size(); ++k) {
            const int s = op.out[k];
            slot_state & st = state_[s];
            // A value is written once.  Two operators writing the same slot
            // means the second one overwrites the first's buffer, and the
            // first's consumer reads the second's value.
            if (producer_[s] >= 0) {
                throw std::runtime_error("the value '" + g_.tensors[s].name + "' is written twice (the " +
                                         "operators " + std::to_string(producer_[s]) + " and " +
                                         std::to_string(i) + ": " +
                                         g_.ops[producer_[s]].node->op_type + " and " +
                                         op.node->op_type + ")");
            }
            producer_[s] = (int)i;
            if (getenv("SI_ONNX_TRACE")) {
                fprintf(stderr, "[onnx] the op %zu %s writes the slot %d ('%s')\n", i,
                        op.node->op_type.c_str(), s, g_.tensors[s].name.c_str());
            }
            // The new buffer must not be another value's buffer: the pool
            // hands out a released buffer, and a value that is still live
            // must never be released.
            if (ctx.out[k]->dev) {
                for (size_t t = 0; t < state_.size(); ++t) {
                    if (t == (size_t)s || !state_[t].dev) continue;
                    if (state_[t].dev == ctx.out[k]->dev) {
                        throw std::runtime_error("the buffer of '" + g_.tensors[s].name +
                                                 "' is also the buffer of '" + g_.tensors[t].name +
                                                 "' (the operator " + std::to_string(i) + ")");
                    }
                }
            }
            const bool int_out = ctx.out[k]->dtype == onnxpb::INT64 ||
                                 ctx.out[k]->dtype == onnxpb::INT32;
            if (ctx.out[k]->on_host) {
                g_.tensors[s].host = ctx.out[k]->host;
            } else if (int_out) {
                // An integer value lives on the host: the shape functions
                // read it there.
                g_.tensors[s].host.resize(ctx.out[k]->bytes());
                if (!g_.tensors[s].host.empty()) {
                    q_download(q_, ctx.out[k]->dev, g_.tensors[s].host.data(), g_.tensors[s].host.size());
                }
            } else {
                g_.tensors[s].host.resize(ctx.out[k]->bytes());
                q_download(q_, ctx.out[k]->dev, g_.tensors[s].host.data(), g_.tensors[s].host.size());
            }
            st.is_const = true;
            st.bytes = ctx.out[k]->bytes();
            if (st.bytes > (size_t)1 << 30) {
                throw std::runtime_error("the folded operator " + std::to_string(i) + " (" +
                                         op.node->op_type + ") produced a value of " +
                                         std::to_string(st.bytes) + " bytes (the count " +
                                         std::to_string(ctx.out[k]->count) + ", the dtype " +
                                         std::to_string((int)ctx.out[k]->dtype) + ")");
            }
            g_.tensors[s].on_host = true;
            g_.tensors[s].count = ctx.out[k]->count;
            g_.tensors[s].shape = ctx.out[k]->shape;
            g_.tensors[s].dtype = ctx.out[k]->dtype;
            if (ctx.out[k]->on_host || int_out) {
                // The value is on the host (an integer, a shape): it has no
                // device copy, and the device pointer it had was never
                // written - keeping it would make the next operator read
                // uninitialised memory.
                if (ctx.out[k]->dev) {
                    release(ctx.out[k]->dev, ctx.out[k]->bytes());
                }
                st.dev = nullptr;
                g_.tensors[s].dev = nullptr;
            } else {
                // The slot's *old* buffer goes back to the pool: the value
                // it held is dead now.  The graph's outputs are never
                // released by the "dead value" rule, so without this each
                // call allocated a new output buffer and dropped the old
                // one - 48 MB per call for the layout model, and a card that
                // filled up after a few pages.
                if (st.dev && st.dev != ctx.out[k]->dev) {
                    release(st.dev, st.bytes);
                }
                g_.tensors[s].dev = ctx.out[k]->dev;
                st.dev = ctx.out[k]->dev;
                // The pool's record of the buffer's *real* size is the one
                // `acquire` wrote, and it includes the guard band.  Writing
                // the value's size over it would file the buffer under a
                // smaller key, and the pool would then hand it to an
                // operator that asked for more - which writes past the end
                // of it, and takes the device with it.
                live_.insert(ctx.out[k]->dev);
            }
        }
        // The inputs of the folded operator are free again.
        for (size_t k = 0; k < op.in.size(); ++k) {
            if (!in_dev[k]) continue;
            release(in_dev[k], host_of(op.in[k]).size());
            if (state_[op.in[k]].dev == in_dev[k]) state_[op.in[k]].dev = nullptr;
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
    if (getenv("SI_ONNX_LOOPDBG")) {
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
        if (getenv("SI_ONNX_LOOPDBG")) {
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

    if (getenv("SI_ONNX_LOOPDBG")) {
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
    if (getenv("SI_ONNX_LOOPDBG")) {
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
    }
    if (getenv("SI_ONNX_LOOPDBG")) {
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
    for (int64_t iter = 0; iter < max_iter; ++iter) {
        if (!cond) break;
        // The body's inputs.
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
        // code below then reads past the end of an empty vector.
        std::vector<std::vector<uint8_t>> out_bufs(bg.output_slots.size());
        std::vector<std::pair<std::string, std::vector<uint8_t> &>> outs;
        for (size_t i = 0; i < bg.output_slots.size(); ++i) {
            outs.emplace_back(bg.tensors[bg.output_slots[i]].name, out_bufs[i]);
        }
        bexec->run(feeds, outs);
        // The condition.
        if (!outs.empty()) {
            const int cs = bg.output_slots.empty() ? -1 : bg.output_slots[0];
            cond = value_is_true(outs[0].second, cs >= 0 ? bg.tensors[cs].dtype : onnxpb::FLOAT);
            if (getenv("SI_ONNX_LOOPDBG") && (iter < 3 || iter % 64 == 0)) {
                fprintf(stderr, "[onnx] the loop: the iteration %lld, the condition %d\n",
                        (long long)iter, (int)cond);
            }
        }
        // The carried values.
        for (size_t i = 0; i < carried.size() && i + 1 < outs.size(); ++i) {
            // A body output that no node produces is a pass-through of the
            // body's input of the same name: the value is unchanged.  (The
            // wireless table structurer's body carries an "assigned" counter
            // that it only reads, and ONNXRuntime returns it unchanged.)
            const size_t oi = i + 1;
            if (oi < bg.output_produced.size() && !bg.output_produced[oi]) continue;
            carried[i].data = outs[oi].second;
            // The shape comes from the body: it is the only place where the
            // value's shape is known (the outer graph's shape for a carried
            // value is the shape of the *initial* value, which a loop that
            // changes it invalidates).
            const int s = bg.output_slots.size() > i + 1 ? bg.output_slots[i + 1] : -1;
            if (s >= 0) carried[i].shape = bexec->output(i + 1).shape;
        }
        // The scans.
        for (size_t i = 1 + carried.size(); i < outs.size(); ++i) {
            const size_t s = i - 1 - carried.size();
            if (scans.size() <= s) scans.resize(s + 1);
            scans[s].push_back(outs[i].second);
            if (s >= scan_shape.size()) scan_shape.resize(s + 1);
            const int os = bg.output_slots.size() > i ? bg.output_slots[i] : -1;
            if (os >= 0) scan_shape[s] = bexec->output(i).shape;
        }
    }
    // The outer scope's values are dead now: the body has its own copies.
    // (They were kept alive above because the copy reads their buffers.)
    for (const auto & r : bg.outer_refs) {
        if (state_[r.second].dev && !is_output(r.second)) {
            release(state_[r.second].dev, state_[r.second].bytes);
            state_[r.second].dev = nullptr;
            g_.tensors[r.second].dev = nullptr;
        }
    }
    if (getenv("SI_ONNX_LOOPDBG")) {
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
        if (getenv("SI_ONNX_LOOPDBG")) {
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

void executor::run(const std::vector<onnxwire::tensor> & inputs,
                   const std::vector<std::pair<std::string, std::vector<uint8_t> &>> & outputs) {
    const auto t0 = std::chrono::steady_clock::now();
    if (getenv("SI_ONNX_LOOPDBG")) {
        fprintf(stderr, "[onnx] the run: %zu inputs, %zu ops\n", inputs.size(), g_.ops.size());
    }
    // The inputs.  Their shape comes with them: the graph's own input shape
    // is often dynamic (a text crop, a page), and the buffers are sized
    // from what the client sent.
    for (const onnxwire::tensor & in : inputs) {
        auto it = std::find_if(g_.input_slots.begin(), g_.input_slots.end(),
                               [&](int s) { return g_.tensors[s].name == in.name; });
        if (it == g_.input_slots.end()) {
            throw std::runtime_error("the graph has no input called " + in.name);
        }
        slot_state & s = state_[*it];
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
        if (s.dev) {
            release(s.dev, s.bytes);
            s.dev = nullptr;
        }
        s.bytes = g_.tensors[*it].host.size();
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
            if (getenv("SI_ONNX_LOOPDBG")) {
                fprintf(stderr, "[onnx] the if: the condition %d, the then [%d, %d), the else [%d, %d)\n",
                        (int)cond, op.if_then_begin, op.if_then_end, op.if_else_begin, op.if_else_end);
            }
            continue;
        }
        const op_entry * e = find_op(op.node->op_type);
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
            if (getenv("SI_ONNX_LOOPDBG")) {
                fprintf(stderr, "[onnx] the op %zu %s is const (skipped)\n", i,
                        op.node->op_type.c_str());
            }
            // An operator the runtime does not implement is an error here
            // too, not a silent skip: the fold skipped it as well (it does
            // not know it either), so its output would be an *empty* value,
            // and a consumer that reads it (a loop-carried value) reads
            // garbage instead of failing.
            if (!find_op(op.node->op_type)) {
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
        if (getenv("SI_ONNX_LOOPDBG")) {
            fprintf(stderr, "[onnx] the op %zu %s runs\n", i, op.node->op_type.c_str());
        }
        op_ctx ctx(q_, *op.node);
        ctx.activation = op.activation;
        ctx.act_params = op.act_params;
        ctx.opset = g_.opset;
        ctx.workspace = scratch();
        ctx.workspace_bytes = ws_bytes_;
        ctx.scratch = scratch_small();
        ctx.scratch_bytes = ws2_bytes_;
        // The inputs are uploaded *before* the context is filled: the
        // operators read the device pointers through the context's tensors,
        // and a pointer captured before the upload is null.
        //
        // The upload is inside the try: an upload that fails is a device
        // error, and it has to name the operator like every other one.  (The
        // upload was outside, and a device lost by the *previous* operator
        // surfaced here, without a name.)
        try {
            prepare_inputs(op);
        } catch (const std::exception & ex) {
            throw std::runtime_error("the operator " + std::to_string(i) + " (" +
                                     op.node->op_type + ") failed while preparing its inputs: " +
                                     ex.what());
        }
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
        std::vector<std::vector<int64_t>> shapes;
        bool have_shape = false;
        try {
            have_shape = e->shape(ctx, shapes);
            if (!have_shape) {
                // The output's shape depends on the data: the operator computes
                // it (a Resize whose sizes are computed, a Range whose limit is
                // the number of the boxes).
                const size_fn * sf = find_size_fn(op.node->op_type);
                if (sf) have_shape = (*sf)(ctx, shapes);
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
        if (getenv("SI_ONNX_TRACE") && op.node->op_type == "Reshape") {
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
            if (getenv("SI_ONNX_TRACE")) {
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
        if (getenv("SI_ONNX_SHAPE")) {
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
        if (getenv("SI_ONNX_DUMP")) {
            // A diagnostic: the first elements of the inputs and the outputs,
            // so the host's view of a buffer can be compared with the
            // device's.  The format is parseable, so a script can compare
            // the runtime's intermediates with ONNXRuntime's.
            const size_t nvals = (size_t)atoi(getenv("SI_ONNX_DUMP"));
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
        static const size_t kGuard = (size_t)atoi(getenv("SI_ONNX_GUARD") ? getenv("SI_ONNX_GUARD") : "0");
        std::vector<void *> out_dev(ctx.out.size(), nullptr);
        try {
            for (size_t k = 0; k < ctx.out.size(); ++k) {
                const size_t bytes = ctx.out[k]->bytes();
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
            const auto t_op = std::chrono::steady_clock::now();
            e->exec(ctx);
            // The wait: only the diagnostics need the operator's result (the
            // timing, the dump and the guard band's check).  The queue is in
            // order, so the next operator's kernels see this one's writes -
            // and a wait per operator costs the pipeline its depth, which
            // on a model with thousands of small operators is most of the
            // time.  (The host operators read the device themselves, and
            // `q_download` waits.)
            if (kGuard || getenv("SI_ONNX_TIME") || getenv("SI_ONNX_DUMP") ||
                getenv("SI_ONNX_DUMP_FILE")) {
                q_.wait();
            }
            if (getenv("SI_ONNX_TIME")) {
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t_op).count();
                fprintf(stderr, "[onnx] time: op %zu %s %.3f ms\n", i, op.node->op_type.c_str(), ms);
            }
            if (getenv("SI_ONNX_DUMP")) {
                // The outputs are dumped *after* the operator: the input's
                // buffer may have been overwritten by it, and the output's
                // buffer does not exist before it.
                const size_t nvals = (size_t)atoi(getenv("SI_ONNX_DUMP"));
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
            if (getenv("SI_ONNX_DUMP_FILE")) {
                // The *whole* output, so the values can be compared with
                // ONNXRuntime's element by element.  The first few values
                // are not enough: a divergence past them is invisible.
                for (size_t k = 0; k < ctx.out.size(); ++k) {
                    const std::string p = std::string(getenv("SI_ONNX_DUMP_FILE")) + "/" +
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
            if (ctx.out[k]->on_host) {
                g_.tensors[s].host = ctx.out[k]->host;
            } else if (int_out) {
                g_.tensors[s].host.resize(ctx.out[k]->bytes());
                if (!g_.tensors[s].host.empty()) {
                    q_download(q_, ctx.out[k]->dev, g_.tensors[s].host.data(), g_.tensors[s].host.size());
                }
            } else {
                g_.tensors[s].host.clear();
            }
            st.is_const = false;
            st.bytes = ctx.out[k]->bytes();
            g_.tensors[s].on_host = ctx.out[k]->on_host || int_out;
            g_.tensors[s].count = ctx.out[k]->count;
            g_.tensors[s].shape = ctx.out[k]->shape;
            // The type too: the next operator reads its input's type, and an
            // int64 value (the dimensions) has to be recognised as one.
            g_.tensors[s].dtype = ctx.out[k]->dtype;
            // A value that is on the host has no device copy.  The buffer
            // the executor allocated for it was never written (or is written
            // once more into the host copy), and keeping the pointer would
            // make the next operator read uninitialised memory: the pointer
            // says "there is a value here" and the value is not there.
            if (g_.tensors[s].on_host) {
                if (ctx.out[k]->dev) {
                    release(ctx.out[k]->dev, ctx.out[k]->bytes());
                }
                g_.tensors[s].dev = nullptr;
                st.dev = nullptr;
            } else {
                // The device pointer has to go back into the graph's tensor:
                // the next operator reads its inputs through *that* pointer,
                // and the operator wrote through its own copy of it.
                // The slot's old buffer goes back to the pool (see the
                // folded-constants path above).
                if (st.dev && st.dev != ctx.out[k]->dev) {
                    release(st.dev, st.bytes);
                }
                if (producer_[s] >= 0) {
                    throw std::runtime_error("the value '" + g_.tensors[s].name +
                                             "' is written twice (the operators " +
                                             std::to_string(producer_[s]) + " and " +
                                             std::to_string(i) + ": " +
                                             g_.ops[producer_[s]].node->op_type + " and " +
                                             op.node->op_type + ")");
                }
                producer_[s] = (int)i;
                g_.tensors[s].dev = ctx.out[k]->dev;
                st.dev = ctx.out[k]->dev;
                if (getenv("SI_ONNX_TRACE")) {
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
        // The input buffers that are dead.  The graph's outputs are not:
        // they are what the client asked for, and they are read back at the
        // end of the run.
        for (int s : op.in) {
            if (s < 0) continue;
            if (state_[s].last_use == (int)i && state_[s].dev && !is_output(s)) {
                if (getenv("SI_ONNX_TRACE")) {
                    fprintf(stderr, "[onnx] the release: op %zu releases '%s' (%zu bytes)\n", i,
                            g_.tensors[s].name.c_str(), state_[s].bytes);
                }
                release(state_[s].dev, state_[s].bytes);
                state_[s].dev = nullptr;
            }
        }
        if (getenv("SI_ONNX_CONSTCHK")) {
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
    // client asked for.
    for (int s : g_.output_slots) {
        if (s < 0 || s >= (int)state_.size()) continue;
        slot_state & st = state_[s];
        if (st.dev && g_.tensors[s].host.empty()) {
            g_.tensors[s].host.resize(st.bytes);
            q_download(q_, st.dev, g_.tensors[s].host.data(), g_.tensors[s].host.size());
        }
    }

    report_leak();

    // The outputs.
    for (const auto & o : outputs) {
        auto it = std::find_if(g_.output_slots.begin(), g_.output_slots.end(),
                               [&](int s) { return g_.tensors[s].name == o.first; });
        if (it == g_.output_slots.end()) {
            throw std::runtime_error("the graph has no output called " + o.first);
        }
        o.second = g_.tensors[*it].host;
    }
    calls_++;
    last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace onnx
} // namespace si
