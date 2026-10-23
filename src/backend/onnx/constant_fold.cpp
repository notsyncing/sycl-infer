#include "exec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "conv_dnnl.h"
#include "kernels.h"
#include "ops.h"

namespace si {
namespace onnx {

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

        if ((getenv("SI_ONNX_TRACE") != nullptr)) {
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
            // A folded value is a constant and is kept for the process: a
            // tight allocation, for the same reason as the constant mirrors
            // in prepare_inputs (a pooled buffer would be held for ever and
            // the pool's best-fit picks a big one).
            out_dev[k] = sycl::malloc_device(ctx.out[k]->bytes() + kGuardFold, q_);
            if (out_dev[k]) {
                alloc_size_[out_dev[k]] = ctx.out[k]->bytes() + kGuardFold;
                live_.insert(out_dev[k]);
                live_now_ += ctx.out[k]->bytes() + kGuardFold;
                if (live_now_ > peak_live_) peak_live_ = live_now_;
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
            // The host bytes changed.  A host-resident folded value has no
            // device copy (the buffer it was given was never written); a
            // device one was just downloaded into the host, so its mirror is
            // the buffer itself.
            ++st.host_gen;
            st.dev_gen = (ctx.out[k]->on_host || int_out) ? st.host_gen - 1 : st.host_gen;
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


} // namespace onnx
} // namespace si
