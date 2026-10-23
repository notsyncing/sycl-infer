# 17 - The ONNX model service

The document pipeline needs *graphs*, not a transformer: a layout detector, a
text detector, a text recogniser, a table segmenturer and a formula
recogniser.  Those models come as ONNX, and this is the subsystem that runs
them.

    src/model/onnx/
        proto.h/.cpp      the ONNX protobuf reader
        graph.h/.cpp      the graph builder, the lowering and the If inliner
        graph_build.h     the graph the transforms rewrite
        transforms/       one file per transform (conv_epilogue,
                          elementwise_activation, matmul_bias, mask_box,
                          transform_util, pipeline)
    src/backend/onnx/
        ops.h/.cpp        the operator interface and the shared helpers
        ops_shape.cpp     the shape functions (what the outputs look like)
        ops_exec.cpp      the operators themselves
        kernels.h/.cpp    the SYCL kernels
        exec.h/.cpp       the executor: the memory plan and the run loop
        constant_fold.cpp the device-executed constant folding pass
        conv_dnnl.h/.cpp  the oneDNN f32 convolution/matmul path
    src/server/onnx/
        service.h/.cpp    the HTTP service and its control plane
        main.cpp          the sycl-infer-onnx binary
    src/common/
        onnx_wire.h/.cpp  the binary wire protocol (also linked by the router)
        onnx_activation.h ONNX activation semantics shared by the graph and
                          the executor
        onnx_types.h      the boolean representation at the boundary

## The three measurements that shaped the design

All three are from the target machine (an Arc A770) and are the reason the
runtime looks the way it does.

**1. A graph runtime's GPU plugin compiles a new kernel set for every input
shape it sees, and that costs seconds of JIT.**  The same model, three
input shapes, the first inference at each:

| shape | first inference | the same shape, later |
|---|---:|---:|
| 704x960 | 8 934 ms | 47 ms |
| 736x960 | 2 970 ms | 46 ms |
| 768x960 | 2 938 ms | 42 ms |

A document pipeline's shapes vary with the data (the text crops, the table
images, the batch sizes), so a per-shape compilation is a cost paid per
document, per model.  A disk cache turns it into a one-off per machine (the
same three shapes then took 281 ms, 89 ms and 67 ms on the second run) -
still a cost.

**This runtime has no JIT anywhere.**  Every kernel takes its shapes as
*runtime* arguments, so one ahead-of-time compiled kernel handles every
shape the document produces.  It is the same property the LLM path has, and
the reason the engine has an AOT build at all.

**2. The GPU plugin's implicit bf16 conversion is wrong for a detector.**
The layout model's decoder amplifies the 0.6% bf16 noise into a 17% error,
which flips the class of every box, and the table segmenter's label map
changes with it.  Measured on the layout detector, the same input:

| precision | latency | the boxes against ONNXRuntime |
|---|---:|---|
| f32 | 160 ms/page | the class ids and the scores are **equal**; the max coordinate error is 27 |
| f16 / bf16 (the plugin's default) | 100 ms/page | the coordinates are wrong by 3 578, the mask plane is all zero, the class ids are permuted |

**The inference precision here is f32**, and it is not a tuning knob: a
detector's output is a decision, and the decision flips.

**3. The per-page budget is dominated by one model.**  Measured on the A770
(one model, one call):

| model | shape | f32 | bf16 |
|---|---|---:|---:|
| layout (PP-DocLayoutV3) | 800x800 | 160-171 ms | 100 ms |
| text detector (PP-OCRv6 det) | 960x704 | 47 ms | 13 ms |
| text recogniser (PP-OCRv6 rec) | 8 x 3 x 48 x 320 | 19 ms | 12 ms |
| table segmenter (UNet) | 1024x1024 | 55 ms | 19 ms |

The layout detector alone is most of a 200 ms page, which is the fact the
rest of the pipeline has to be measured against.

## The pipeline

    ONNX file
      -> proto            the protobuf reader (the ONNX schema, read directly)
      -> graph_build      the named values, constants, consumers and liveness
      -> transforms       a fixed-point pipeline of independent graph passes
                          (conv epilogue, elementwise activation, MatMul bias,
                          mask-to-box); each is one file under transforms/
      -> lowering         the executor op list (graph.cpp)
      -> ops/kernels      the shape functions, the operators and the SYCL
                          kernels (ahead-of-time compiled)
      -> exec             the memory plan and the run loop; the constant
                          sub-graphs are evaluated once, on the device
      -> service          the HTTP endpoint and the JSON-RPC control plane

The graph transforms are deliberately *not* branches in the builder:

* `graph_build.h` is the IR: nodes, named values, the constant analysis,
  consumers, dead nodes and the alias map.
* Every pass is a function over that IR with its own `.cpp` under
  `src/model/onnx/transforms/`: `conv_epilogue`, `elementwise_activation`,
  `matmul_bias`, `mask_box`.  `pipeline.cpp` runs them in order to a bounded
  fixed point; `SI_ONNX_NOFUSE=1` skips the whole pipeline.
* The builder lowers the settled graph to the executor's `exec_op` list.
  Constant folding is the exception: its sub-graphs are evaluated by the
  executor because they run operator kernels, so it lives in
  `src/backend/onnx/constant_fold.cpp` and runs once on the device.

Adding a transform means adding one file and one entry in `pipeline.cpp`; it
does not touch the loader or the executor.

### The protobuf reader (`proto.h`)

ONNX stores a model as a `ModelProto`, a protobuf message.  The reader
implements the parts of the wire format the schema needs: the varint, the
64-bit and the 32-bit types, and the length-delimited ones.  It is strict:
a malformed model produces an error with a message, never a wrong number.

Three things in the schema are easy to get wrong, and each one cost a
debugging session:

* **A repeated string field is not a repeated message.**  Reading the node's
  `input` list as a packed sub-message turns every name into an empty
  string; the node then has no operator and the graph silently does
  nothing.
* **A repeated numeric field is unpacked** (the schema is proto2), so every
  element is its own field.  The packed form is accepted too.
* **The `dims` of a tensor are a repeated field, not a message** - and a
  tensor whose `dims` are not read has an empty shape, which makes every
  loop that walks the shape walk off the end.

### The graph (`graph.cpp`)

Two optimisations, both of which are the reason a hand-written runtime can
be competitive:

* **Constant folding.**  A torch-exported model is full of constants: the
  table models carry 484 `Constant` nodes, and the layout model's shape
  arithmetic (78 `Shape`, 165 `Reshape`) collapses into constants.  The
  folding is a *device* pass: the constant sub-graphs are evaluated once, on
  the device, the first time the graph runs, and the results are kept as
  constants.
* **Operator fusion.**  The text detector is 83 convolutions followed by 88
  `Mul`/`Add` pairs (the batch norm that was folded into the weights) and 20
  activations.  Fused, that is 83 kernels instead of 200, and each
  convolution's epilogue does the scale, the bias and the activation in
  registers.  The rules are narrow and explicit:
  * a `BatchNormalization` folds into the preceding convolution (its scale
    becomes `gamma/sqrt(var+eps)` and its bias `beta - mean*scale`);
  * a `Mul`/`Add` with a constant operand folds into the convolution the
    same way (this is the batch norm the exporter left in the graph);
  * an activation folds into the convolution's epilogue;
  * the convolution's weight is transposed once, to `(C/g, kh, kw, M/g)`, so
    that the output channels of one `(c, kh, kw)` are contiguous and the
    kernel reads them as a run.

  A fused operator changes the *name* of the value it produces, so the
  consumers are rewritten through an alias map: without it the next
  convolution reads a value nobody produces.

### The operators (`ops.cpp`, `ops_shape.cpp`, `ops_exec.cpp`)

Every operator is a pair: a *shape* function (what the outputs look like,
given the inputs) and an *execute* function (the kernel).  The shape
function is what lets the graph be planned before it runs, and it is what
the constant folder uses to evaluate the constant sub-graphs.

The shape functions and the kernels must agree exactly: a shape function
that computes a *smaller* output than the kernel writes is a heap overflow,
and a guard band behind every output buffer turns that into a report naming
the operator instead of a crash much later.

### The executor (`exec.cpp`)

The executor owns the device memory.  A buffer is reused once the value it
holds is dead (its last consumer has run), so a graph with a thousand
operators and a few hundred live tensors runs in the memory of the live
set.

Two rules matter here:

* **The graph owns its nodes.**  The operators point at their node's
  attributes, so a node that outlived its owner is a dangling pointer.
* **A host value's device copy is a mirror, and the generations say whether
  it is current.**  `host_gen` counts the writes of the host bytes and
  `dev_gen` is the write the mirror was made from; equal means the device
  copy can be read without copying again.  The copy used to be thrown away
  at every operator, so every consumer uploaded again and every constant was
  uploaded again on every call: the layout model paid **3 508 copies of
  707 MB** per call, half of its time.

The upload path is the subtle part:

* **A copy out of pageable memory is synchronous on this backend.**  It
  blocks until the in-order queue drains, so a 0-byte upload enqueued behind
  80 ms of kernels cost 80 ms.  Uploads are therefore staged through a
  pinned USM host buffer (8 MB per executor), where the copy is
  asynchronous; the bytes are copied into the stage with `memcpy` first.
* **The staged bytes must stay alive until the copy runs.**  The stage is
  appended to and reused only after a queue drain; `run()` drains once
  before it overwrites the next call's input bytes, and a full stage drains
  and starts over.
* **A downloaded integer stays on the device as well.**  The shape
  functions need the host bytes, but a later consumer that reads the value
  on the device can use the buffer it was downloaded from - the write-back
  keeps it and marks the mirror current when the operator computed it on
  the device.
* **The generation pair is an invariant, so it can be checked.**
  `SI_ONNX_MIRRORCHK=1` downloads every reused mirror and compares it with
  the host bytes; a missed `host_gen` bump then fails loudly instead of
  producing plausible output from the previous step's value.

### The wire protocol (`wire.h`)

The service is model-agnostic: the client names a model and sends its
inputs.  The framing is binary, because a document pipeline moves tensors,
not JSON: one 800x800x3 float page is 7.7 MB, and a JSON/base64 encoding
spends longer in the decoder than the model spends in the forward pass.

    request   'S' 'I' 'O' 'N' | u32 version | u16 name_len | name | u32 n_in
              | per input:  u16 name_len | name | u8 dtype | u8 n_dim
                            | u32 dim[n_dim] | u64 n_bytes | payload
    response  'S' 'I' 'O' 'R' | i32 status | u32 msg_len | msg | u32 n_out
              | per output: u16 name_len | name | u8 dtype | u8 n_dim
                            | u32 dim[n_dim] | u64 n_bytes | payload

The input's *shape* travels with it: the graph's own input shape is often
dynamic (a text crop, a page), and the buffers are sized from what the
client sent.

### The service (`service.cpp`)

    POST /v1/onnx/run     run a model
    GET  /v1/onnx/models  the model table, with each model's ports
    GET  /health          liveness, the device, and the per-model statistics

One model per id, one device per model (`--device cpu|gpu|auto`, with the old `GPU.0`/`CPU` aliases still accepted), and
a model that cannot be loaded is a *reported* failure rather than a reason
to take the service down.

## What the runtime has to get right

Every one of these is a bug that was found and fixed, and each of them
looks like a *numerical* bug rather than a memory bug, which is what makes
them worth writing down.

* **An output's device pointer has to go back into the graph's tensor.**
  The operators write through their own copy of the tensor, and the next
  operator reads its inputs through the graph's copy.  Without the
  write-back the second operator reads a *stale* pointer - and the numbers
  are wrong with no visible error.
* **The output tensors must outlive the call.**  A `std::vector<tensor>`
  that holds the outputs, not a pointer to a stack temporary: the
  temporary dies at the end of the loop iteration and every write through
  the dangling pointer (the device pointer, the shape, the count) lands on
  freed stack.  The symptom was a count of `0xABABABABABABABAB` - the
  guard band's fill pattern.
* **An output's buffer is released only when the value is dead.**  Releasing
  it right after the operator that produced it hands the same memory to the
  next operator, and the two values alias.
* **The pool tracks the *real* size of a buffer**, not the size it was
  asked for: the pool hands out the smallest buffer that fits, so a buffer
  allocated for 8 MB can be released as 1 MB, and the next operator that
  asks for 4 MB gets a 1 MB buffer.  The pool also keeps the set of live
  allocations, so a double hand-out is an error where it happens.
* **The host/device copies are *not* all synchronous.**  The uploads are
  staged through pinned host memory and enqueued (see the executor above);
  a copy out of a temporary vector would still be a use-after-free, which is
  why the staging buffer, the per-run drain and the generation pair are one
  mechanism, not three.  The *downloads* stay synchronous: an integer is
  read on the host as soon as the operator returns.
* **The intermediate values stay on the device.**  The host copy is made
  only for the values that need it (the constants, and the graph's outputs):
  a round trip per operator is the difference between a GPU runtime and a
  slow one.
* **The guard band behind an output is written *on the device*.**  A host
  write to a device pointer faults on a discrete GPU and silently corrupts
  the heap on an integrated one.
* **`auto_pad` has to be read.**  The exporters write it as an int (the
  schema's form) *or* as a string ("SAME_UPPER"), and a model that says
  SAME_UPPER gets an output of `ceil(input/stride)` with the padding
  computed from it.  Reading the pads alone gives the wrong size - and a
  size that does not match the operator's, so the operator and the shape
  function disagree about the shape of the operator's own output.  In the
  text detector that showed up as a `Concat` whose two inputs had different
  spatial sizes.
* **The sigmoid is computed stably.**  The device's `exp` is a fast
  approximation; the naive `1/(1+exp(-x))` form is not accurate enough
  where it matters.

### The `Resize`, and the table accuracy

The OCR detector upsamples and downsamples its feature maps with ONNX
`Resize(mode=nearest, coordinate_transformation_mode=asymmetric,
nearest_mode=floor)`.  The source index is `floor(x_out * in / out)`, and it
was computed in `float`:

    iw = (int)((float)wo * W / (float)Wo)

The device compiler turns that division into a reciprocal multiply, and the
reciprocal is slightly **below** the true value: upsampling 30 -> 60, the
element at `wo = 2` came out as `floor(0.999992)` = 0 instead of 1, so the
first source pixel took **three** output columns and every later one shifted
by a pixel.  The detector's probability map was one pixel off, the text boxes
moved, and the PDF text spans clipped to those boxes lost their leading
character: `190,083,517` became `90,083,517`, `-77,332` became `77,332`.  It
was **33 of the 45 tables** on the reference document, and no aggregate
(detection count, CTC text, table grid) noticed.

The fix is integer arithmetic - `(wo * W) / Wo` - which is exact and
floor-correct for non-negative indices:

    const int ih = (int)(((int64_t)ho * H) / Ho);
    const int iw = (int)(((int64_t)wo * W) / Wo);

After it the detector's raw output is within **1.5e-5** of ONNXRuntime's and
its boxes are **identical**; the pipeline's digest is **byte-identical on
15 of 17 pages** (the other two differ by a single 0.1 px box coordinate),
and **all 45 table HTMLs match the local ONNXRuntime/OpenVINO reference
cell for cell**.  The comparison tools are `dev/cmp_det_raw.py` (one model's
input/raw output against the local engine), `dev/pipeline_digest.py` (the
whole document's content: every region's type, box and OCR text plus every
table's HTML) and `dev/cmp_tables.py` (the HTML as a cell grid, so a
formatting difference is not confused with a content difference).

## The integer values

The int32 and int64 values (the dimensions, the axes, the indices) are a
separate path, and they have to be: the *shape* functions run **before** the
operator does, and they read the dimensions from the host.

So an integer value stays on the host, the operators that produce one
(`Shape`, `Cast`, `ConstantOfShape`) write the host copy, and the operators
that consume one (`Slice`, `Concat`, `Gather`, `Transpose`, `Reshape`,
`Identity`) read it there.  The device holds floats; an integer tensor's
device buffer is never read.

The exception is the operators that *reduce* an integer: the layout
model's decoder reduces a mask of twelve million int32 elements, and the
host takes 300 milliseconds for it.  The reduction and the elementwise
family therefore have a kernel that reads the integers in their own
width - the value goes to the device for the arithmetic and comes back
to the host afterwards, like every other integer.

Three things follow from that, and each of them was a bug:

* **The type has to be propagated.**  The model does not carry the types of
  the intermediates, so the executor gives an operator's output the type of
  its first input - and the `Shape`, the `ArgMax` and the `Cast` change it.
  Without that, an int64 value is a float value, and every consumer reads
  the wrong number of bytes.
* **The models mix int32 and int64** for the same kind of value (the
  axes, the starts), so every integer read follows the tensor's type
  rather than assuming 8 bytes.
* **A value that is on the host has no device copy.**  Its device pointer
  was never written, so keeping it makes the next operator read
  uninitialised memory - a device fault on a discrete GPU and a heap
  corruption on an integrated one.

## The shape of an operator's output

Most operators' output shape is known before they run, and the shape
function says what it is.  Two cases are not:

* **The shape depends on the data** (a `Resize` whose sizes are computed,
  a `Range` whose limit is the number of the boxes).  These have a *size
  function*, which the executor calls when the shape function cannot
  answer.  The size function may download what it needs.
* **The operator removes axes** (the ONNX `Slice` *removes* the axes it
  slices, so its output's rank is lower than its input's).  A shape
  function that keeps the rank makes every consumer read the wrong number
  of bytes.

An operator whose output's shape is neither known nor computable is an
**error**, not a zero-sized tensor: a count of 1 makes the operator write
past the end of its buffer, and the guard band reports it as an overflow
rather than as the shape error it is.

## The state of it

Working: the protobuf reader, the graph and the fusion, the executor's
memory plan, the wire protocol, the HTTP service, and the operators
`Conv`, `ConvTranspose`, `MaxPool`, `AveragePool`, `GlobalAveragePool`,
`Resize`, `Reduce{Max,Min,Sum,Mean}`, `Concat`, `Softmax`,
`LayerNormalization`, `MatMul` (2-D and batched), `Gemm`, the elementwise
family, `Identity`, `Shape`, `Reshape`, `Flatten`, `Transpose`, `Gather`,
`Slice`, `TopK`, `ArgMax`, `Where`, `Squeeze`, `Unsqueeze`, `Cast`,
`Clip`, `ConstantOfShape`, `Range`, `Split`, `Tile`, `Expand`, `Not`,
`GridSample`, `GatherND`, `ScatterND`, `ScatterElements`, `Einsum`,
`Mod`, `BatchNormalization`, `OneHot`, `Loop` and `If`.

Two of them are the sub-graph operators, and they are where the
sub-graph rules live:

* **`If`** - its two branches are inlined into the enclosing graph's
  operator list and only the selected one runs.  The branches are *not*
  self-contained (the exporter's sub-graphs read the values around
  them), so they share the enclosing graph's slots, and a branch's own
  initializers come with it.
* **`Loop`** - the body is a graph of its own, built with the
  enclosing graph as its *scope*, so the body can read the values the
  enclosing graph computes (ONNX's outer scope).  The trip count is
  optional: an empty one means "no limit".

### The oracle, and what it says

The right way to find a divergence is to compare the *intermediates*, not
the output.  The model is rewritten so that the named intermediate values
are graph outputs too, both engines run it, and every output is compared.
The first difference in the topological order is where the runtime goes
wrong.

The tools are in `dev/`:

| script | what it does |
|---|---|
| `onnx_oracle.py` | runs the pipeline with ONNXRuntime, records every model's inputs and outputs, then replays them against the service and compares |
| `full_diff.py` | compares the service's *dumped* intermediates with ONNXRuntime's, element by element |
| `value_diff.py` | the same, from the `SI_ONNX_DUMP` trace |
| `shape_diff.py` | the same, for the shapes |
| `ort_shapes.py` | runs a model with ONNXRuntime and prints every intermediate's shape |
| `make_mini.py` | builds a two-node model, to isolate one operator |
| `compare_remote.sh` | the per-model comparison, one call at a time |

The comparison needs the fusion off (`SI_ONNX_NOFUSE=1`) to be about the
*operators*, and on to be about the *model*: the fused operators produce
the convolution's output (after the batch norm and the activation), not the
graph's intermediate.

**The text detector is exact.**  Against ONNXRuntime: the largest
difference is 1.9e-05 on a probability map, which is the float32
accumulation.

**The table classifiers are exact** (`q_cls`, the 2-class wired/wireless
classifier: 1.6e-06).

**The layout detector, the recogniser and the UNet differ from
ONNXRuntime by the float32 accumulation** - two elements of twelve
million differ, by one unit.  The wireless table model agrees on 97 of
its 98 class scores (0.026 apart).  **The formula model is exact too**:
on five real formula crops from a paper template, every output is
integer-equal to ONNXRuntime's, and its backbone matches to the last
value (793 intermediate values, no difference).  (It used to exit its `Loop` after one iteration - the
loop's condition is a boolean, and a boolean was read as a byte, so a
condition that was true read as false.)  It is also the slowest model
here: its loop re-runs the body (1 742 operators) per iteration, and we
spend 145 ms on an iteration where ONNXRuntime spends 12.4.

The layout detector's difference is a *permutation*: its anchor scores
are nearly equal, a 1e-05 difference in the matmul reorders them, and
the boxes come out in a different order (the same 300 boxes, the same
classes, the same scores - a different reading order).

### The rules the numbers taught

These are the bugs that the oracle found, and they are the ones that a
reader of the code is most likely to repeat.

* **A subgraph may read the enclosing graph's values.**  ONNX's "outer
  scope", and the exporter uses it: the formula recogniser's `Loop` body
  reads a transpose that the enclosing graph computes.  A body built on
  its own cannot resolve such a name, so the builder takes the enclosing
  graph as its scope and the executor copies the value in before every
  **call** - the value changes between the calls, so a copy taken once
  is stale.  It is copied once per call and not once per iteration
  (ONNX's outer scope is read-only, so the body cannot change it), and
  the copy is a *device* copy: a round trip through the host costs a
  device wait, and the formula recogniser's body reads 816 of those
  values.

  **Two things about that copy are not obvious, and both were bugs.**
  *The value has to stay alive until the copy*: the executor hands a
  value's device buffer back to the pool after the last operator that
  reads it, and an outer-scope value is read by *no* operator of the
  enclosing graph - only by the body.  The pool had therefore already
  recycled the buffer, and the copy read whatever the pool had put
  there: the body computed the same thing for every input, and the
  formula recogniser's output did not depend on the image at all.  And
  *the copy has to set the tensor's own device pointer*, not only the
  executor's slot: the body's operators read their inputs through the
  tensor, and it was still the old (empty) buffer - so the body
  computed with zeros.

* **An `If` branch may declare an output that none of its nodes produces**,
  and then the value is the enclosing scope's.  The formula recogniser's
  exit check is `If(all_found)` and its *then* branch is empty - it
  declares a constant (true) as its output.  The inliner has to copy that
  value into the If's slot, inside the branch's range; without the copy
  the slot keeps the other branch's zero and the loop never stops.  The
  copy in turn must not be skipped as a "constant input" - only the
  selected branch runs, so an operator inside a branch is never constant.

* **A `CumSum` is not a `Reduce`.**  It has no `keepdims` attribute and
  it does not remove the axis: the output has the input's shape, and
  the values are the cumulative sums along the axis.  Copying the
  `Reduce`'s shape function drops the axis, and the formula recogniser's
  mask of the classes seen so far collapses to one element - so the
  count of each class can never grow, the loop's exit condition can
  never become true, and the loop runs until the card's memory is gone.

* **An operator's output type is not always its input's.**  The
  `OneHot`'s output is the *values'* type (its indices are int64 and
  its vector is a float), the `ArgMax`'s and the `Range`'s are int64
  whatever the input is, and a `ConstantOfShape`'s is its `value`
  attribute's.  The executor's default - the output takes the input's
  type - is wrong for those, and the mismatch is silent: the operator
  writes the right bytes and the consumer reads them at the wrong
  width.  The wireless table structurer's one-hot was written as int64
  and concatenated with a float score, so the decoder's argmax picked a
  random column and its `Loop` never terminated.

* **The two branches of an `If` write the same output slot.**  So a
  constant folded in one of them is destroyed by the other branch's
  operator the first time that branch runs - and the next time *this*
  branch is selected, the slot holds nothing.  A constant inside a
  branch has to run as an operator, in the selected branch only.

* **A `Loop`'s trip count is optional**, and an empty one means "no
  limit".  The omitted input is the slot -1, so reading it as a tensor
  reads whatever is next to it in memory - and the trip count comes out
  as several exabytes, which is how the first version of the formula
  model died.

* **A boolean is one byte in the model and in the wire, and a float
  0/1 inside the runtime.**  The device holds float32, so a one-byte
  boolean on the device is three bytes of whatever is next; and 1.0f's
  first byte is 0x00, so a boolean that was *true* reads as false when
  it is tested as a byte.  A `Loop`'s condition is such a value, and its
  host copy is *cleared* by the write-back (a float lives on the
  device), so it has to be read back from the device - and read **as a
  float**.  That is why the wireless table structurer's loop ran 501
  times instead of 49, and why the formula recogniser's loop exited
  after one iteration.  **Every place that touches a boolean has to
  agree on the representation** - the element size, the host copy, the
  device buffer, the wire - and the boundary is where the model's
  one-byte form is converted.

* **A device wait per operator costs the pipeline its depth.**  The
  guard band behind every output (a diagnostic that catches an operator
  writing past the end of its buffer) was written *and read back* per
  output, and the operator was waited for after every operator: three
  device waits per output.  On the formula recogniser (1 742 operators
  per iteration) the waits were 2.7x the device time - and the model
  that runs the loop 2 561 times is exactly where that shows.  The band
  is off by default (`SI_ONNX_GUARD=<n>` turns it on), and the wait
  happens only when a diagnostic needs the result: the queue is in
  order, so the next operator's kernels see this one's writes anyway.

* **An integer value the device reduces has to be reduced on the
  device, in its own type.**  The layout model's decoder reduces a
  mask of twelve million int32 elements: the host takes 300
  milliseconds for it, the device 17.  The device's kernel reads
  floats, so an integer operand needs a typed kernel (the reduction
  and the elementwise operators have one) or the host path - and the
  host path is only *correct* for the few values, because the integers
  live on the host anyway (a condition is a scalar).  Reading an
  integer as a float reads its bits: that is how a comparison of two
  int64 masks came out right by accident (a small integer read as a
  float32 is a denormal, and the denormals keep the order), and how a
  division by an integer shape came out as garbage.

* **The `ScatterElements`' updates have either the indices' shape (one
  value per index) or the data's shape (a slice of the axis).**  The
  wireless table structurer uses the first, and reading it as the
  second writes nothing at all.

* **The `GatherND`'s trailing block is contiguous**: the index tuple
  addresses the data's first dims and the rest is one flat run, so the
  gather is `data[sum(index[d] * stride[d]) + t]`.  The
  `ScatterElements` is a different operator with a different
  convention, and mixing the two is how the layout's mask came out
  wrong.

* **An operator's parameter is not always an attribute.**  The `TopK`'s
  `largest` is an attribute, and the layout model's decoder asks for the
  *smallest* 300 of 13 125 scores.  Picking the largest instead reorders
  every anchor, and the boxes come out in the wrong places.  The `Split`'s
  sizes are an *input* since opset 13, and reading only the attribute
  divides the encoder's 13 125 positions into three equal thirds instead
  of the model's 10 000 / 2 500 / 625.
* **The `Slice` keeps its rank.**  Opset 10 and later: the sliced axes
  stay where they are, they are only shorter.  The gather has to follow
  the *output's* coordinates, and the source's stride has to include the
  slice's *start* - without it the second slice of a tensor reads the
  first slice's elements again.
* **The `Concat` promotes a rank-0 value.**  numpy's rule: an input is
  left-padded with 1s to the output's rank, and the output's rank is at
  least `axis + 1`.  Paddle's export builds vectors out of scalars, and
  without the promotion the vector comes out one element short - and the
  operator downstream (a `Resize`, whose sizes are the vector) reads a
  truncated size.
* **The `Gemm` has a bias.**  It is the third input, per output column.
  Without it the scores are off by a whole vector, and the class that
  wins flips.
* **The activation's parameters travel by value.**  A host pointer read
  from a device kernel is garbage: the convolution's epilogue read
  whatever the host's stack held, and the hard sigmoid came out as zero.
* **The integer values live on the host, and the host is not the
  device.**  An operator that produces an integer value and one that
  consumes it must both agree on where it is: the `Expand` of an integer
  (the layout model's decoder expands the arange into the box indices)
  read a host-only buffer through the device pointer, and the scatter
  then scattered garbage.
* **An integer value that is an *output* is on the device.**  The
  `GatherND`'s output is a float and the next operator is a kernel, so
  the value has to be uploaded; a host-only output is read as zeros.
* **The fusion's alias is a rename, and a rename breaks a consumer that
  reads both.**  `y = x * sigmoid(x)` reads the convolution's output
  *and* the activation's output; folding the activation renames the
  second to the first, and the consumer reads one value twice.
* **The device memory is the pool's, and the pool's keys are the real
  sizes.**  A buffer released under the size it was *asked* for is handed
  out to an operator that wants more, which writes past the end of it -
  and that is a device fault, not a wrong number.
* **A value that is dead is still holding a buffer.**  The graph's
  *outputs* are never released by the "dead value" rule, so every call
  allocated a new output buffer and dropped the old one: 48 MB a call for
  the layout model, and a card that filled up after a few pages.

* **A cast writes the *output's* width, not the input's.**  A cast from
  int64 to int64 that writes floats leaves the consumer reading the
  float's bits as an integer - and the consumer is usually a `Concat`
  that builds a shape, so the *next* operator reads a garbage size and
  the model dies with a dimension of 4.9e18.  The kernel therefore
  takes the input's kind and the output's kind, and not a boolean.

* **A rank-0 value has no axis.**  Every operator that indexes the
  shape (`shape[0]`, `shape.back()`, the axes of a `Slice`) has to
  handle it: a `Slice` of a scalar is the scalar, and `shape[0]` on an
  empty vector is a wild read.  The models produce rank-0 values (a
  `Gather` of one element, a loop-carried scalar), so this is a normal
  input and not an edge case.

* **A `Loop`'s outputs are the final values and the scans - the
  condition is an *input*.**  The *body*'s outputs are different: they
  start with the condition.  And the body's output buffers have to be
  built *before* the body runs: the executor fills the buffers the
  caller asks for, so an empty list means the body's outputs are never
  collected and the loop reads past the end of an empty vector.

* **An operator the runtime does not implement is an error, even in a
  folded sub-graph.**  The fold skips what it does not know, and the
  runtime skips a folded operator whose inputs are all constants - so
  its output is an *empty* value, and a consumer that reads it (a
  loop-carried value) reads garbage instead of failing.  The runtime
  now says so.

* **The `HardSwish` has no attributes at all, and the `HardSigmoid`'s
  formula never changed.**  ONNXRuntime rejects an `alpha` attribute on
  a `HardSwish` ("Unrecognized attribute: alpha for operator
  HardSwish"), so the form is fixed: `x * max(0, min(1, (x + 3) / 6))`.
  Using `alpha=0.2` - what most sources quote - is the difference
  between the Paddle classifier's scores and the reference's: every one
  of its six million hardswish values was wrong, and the recogniser was
  wrong with it.  The hard sigmoid is `max(0, min(1, alpha*x + beta))`
  with alpha=0.2 at *every* opset (measured with a one-node model at
  opset 11, 12 and 14), so the "legacy" form this runtime used to
  apply before opset 14 was wrong for every model that has a hard
  sigmoid and is older than that.

* **A broadcast flag has to say *which* operand is the broadcast.**  The
  layout model's decoder adds a scalar to a tensor, and the scalar is
  the *first* operand.  A flag that only says "one of them is a
  scalar" reads the wrong one: the result is a tensor of zeros where
  the reference has the `1e8` sentinel, and then every box is wrong.
  The operator is not commutative either (a subtraction and a division
  are not), so the operands cannot simply be swapped.

* **A `ConvTranspose` has an `output_padding`.**  It is "additional
  elements added to the side of the output shape", and without it the
  UNet's decoder comes out one pixel short in both directions - and a
  decoder that is one pixel short puts every box one pixel off.  The
  `output_shape` attribute, when it is there, replaces the geometry
  altogether.

* **An operator that writes float32 into an integer output leaves half
  of it unwritten.**  The `Tile`'s input can be an `int64` (the layout
  model's anchor indices are an `arange`), and a float kernel over an
  8-byte buffer writes every other element - the rest is whatever the
  pool handed out.  The `Tile` also has an `axis` attribute, and its
  repeats are an int32 in some models and an int64 in others.

### The performance

The layout detector is the dominant cost, and on the A770 it is (2026-10-23,
warm call; `SI_ONNX_TIME=1` waits per operator, so the shares are what the
breakdown is for; the whole-model row is the service's own `run` time):

| what | time |
|---|---:|
| the whole model, one 800x800 page (oneDNN `Conv`/`MatMul`, fused) | **129-141 ms** |
| the `Conv`s (118 of them, oneDNN) | 93 ms |
| the `TopK` (300 of 13 125, three calls) | **4 ms** |
| the `MatMul`s (123 of them, plus the bias/act epilogue) | **9 ms** |
| the `Concat`s (140 of them) | 5.5 ms |
| the `LayerNormalization`s | 3.3 ms |
| the `Transpose`s | 3 ms |
| the `Add`s | **1.9 ms** |
| the `Relu`s that are left | 2 ms |
| the mask-to-box head (was six reductions and ten 48 MB tensors) | **0.2 ms** |
| the first call, with the runtime's JIT and the weight upload | ~1.2-1.4 s |

**The MatMuls and the reductions were the next two after the convolutions.**
The batch-1 matmul (the decoder's `bmm`: `[1,300,32] x [1,32,40000]`) and the
2-D matmuls now run on oneDNN (`matmul_dnnl_launch`, plain layouts, the
primitive cached by `M:K:N`): **36 -> 8 ms**.  The reductions were one
work-item per row - the layout's mask max is **300 rows of 40 000 int
elements**, i.e. 300 threads - and a work-group per row (256 lanes, tree in
SLM) took them **29 -> ~1 ms**.  The UNET's `ArgMax` was a *host* loop (a
download and a serial scan, 14.6 ms/table); one work-item per output
position on the device took it to ~0.1 ms and the UNET's operator total
56.2 -> 41.3 ms.

**The formula recogniser's loop is a single-call model in this build.**  A
synthetic random 384x384 input can drive its `Loop` long enough to grow the
pool; a second call then fails with OOM or DEVICE_LOST **with the fusions
off as well**, so that is not a fusion regression.  A zero input is stable
(output sum 1853), the five real formula crops were integer-exact, and the
service keeps the model loaded; exercise it one crop per process until the
loop's memory is bounded.

**The mask head is now one scan.**  The decoder's mask-to-box branch is
`bmm [1,300,32] x [1,32,40000]` -> reshape -> `Greater` -> `Cast` ->
`Mul` by the row/column grids -> `Add` -> `Where` -> `Reshape` ->
`ReduceMax`/`ReduceMin`, and the model derives the four box extrema and the
"any mask" flag from it.  `graph.cpp` recognises that exact pattern
(including the `+0` splat and the `1e8` empty-mask sentinel), proves that
nothing outside the branch reads its intermediates, and replaces the whole
cone with one `MaskBox` kernel over the bmm: `mask_box_launch`.  It reads
the 48 MB bmm once and writes five `[B,300]` values.  The graph went from
ten concurrent 12M-element tensors to **one** (the bmm), and the operator
time from **9.5 ms to 0.2 ms**.  The output shape keeps every leading batch
axis (`[B,300,200,200] -> [B,300]`); an earlier `[1,B*300]` version was
correct only for batch 1 and broke the layout batch-2 pipeline.

**The smaller fusions.**  `MatMul` + a constant 1-D `Add` is one op: the
Add is removed and the bias is applied by `bias_act_launch` after the
oneDNN matmul.  The obvious alternative - oneDNN's matmul *bias* primitive
- was **4.5x slower** on these shapes (36 ms against 8 ms for the model's
123 matmuls), so the bias is a separate epilogue, like the convolution's.
An `Add`/`Mul`/`Div` followed by a single `Relu`/`Clip`/`Sigmoid` gets the
activation in its elementwise epilogue (the same `apply_act` the conv uses);
that removes 41 activation nodes.  `Reshape`/`Flatten`/`Squeeze`/
`Unsqueeze`/`Identity` borrow the input's buffer when it dies at that op,
instead of copying the tensor, and an elementwise op may reuse a
same-shape, same-type input buffer that dies there.  Both are disabled in a
`Loop` body: its carried values are read again on the next iteration, which
per-op liveness cannot see.  The aliasing and in-place reuse cut the
layout's executor footprint roughly in half (below).

**The convolutions run on oneDNN** (`src/backend/onnx/conv_dnnl.cpp`,
`PF_ONNX_DNNL=0` turns it off).  The runtime's weights stay in their own
(C, kh, kw, M) layout; the primitive's layout is reordered into once per
weight and cached, and the bias/scale/add/activation epilogue stays a
separate kernel so its semantics are the SYCL conv's exactly.  Measured on
the model's own shapes against the SYCL kernel:

| shape | SYCL | oneDNN f32 | oneDNN bf16 |
|---|---:|---:|---:|
| C32 400x400 -> M64 | 0.82 ms | 0.70 | 0.65 |
| C128 100x100 -> M256 | 0.73 | 0.56 | **0.16** |
| C96 64x240 -> M96 | 1.62 | 0.57 | **0.13** |
| C256 50x50 -> M512 | 1.31 | 0.20 | 0.17 |

bf16 reaches 9-19 TFLOP/s on the shapes where the XMX unit helps, but the
gain is shape-dependent (1.0x on several) and the layout detector's accuracy
is sensitive to bf16 (the engine's earlier all-bf16 layout produced a 17%
decoder error), so the bf16 path is a probe, not the default.  The cheap
variant - `PF_ONNX_DNNL_BF16=1`, oneDNN's `fpmath_mode::bf16` on the f32
descriptors - was tried on the real layout: the digest is **identical**
(boxes and tables unchanged), but the call is **not faster** (233 ms against
208-225 for f32; the probe's fast bf16 path used bf16 *memory descriptors*,
which would need an f32<->bf16 conversion pass per conv in a mixed graph).
So bf16 is rejected on speed here, not on accuracy.  The
deconvolution was tried the same way and **reverted**: its output was wrong
on the detector's 2x2 stride-2 transposed convolutions, and the table UNET's
transposed convolutions are only 26 ms of its forward.

**The uploads were 3 508 copies of 707 MB per call**, and they are now **366
copies of 7.4 MB**: the constant mirrors survive the call, the host values
keep their device copy between consumers, and the copy is asynchronous
(through pinned host memory) so it no longer drains the queue.  The mask
branch's 48 MB output is not transferred at all: `drop_outputs` in the model
config names it, the executor does not read it back, and the graph's
backward mark from the kept outputs skips the 28 operators that only feed
it.  On this document the layout postprocess only ever used rectangles, and
the pipeline's digest is **byte-identical with and without the mask**.

**The `TopK` was the first thing to fix and the reason is worth keeping:**
the obvious implementation (k passes over the columns, each picking the
best of the ones not yet picked) is O(k²·cols) because the "is it taken"
test is O(k) - 1.2 billion comparisons for the layout model, which is
**60 seconds for one page**.  Keeping the row's list sorted and inserting
into it is O(cols·k): 4 million, 59 ms.  **A selection that is quadratic
in k looks like a hang, not like a performance problem**, and it was
99.7% of the model's time.

**Those 59 ms were the insertion, not the comparisons.**  The list is
sorted, so the position is a binary search - but the *shift* is a chain
of dependent global reads and writes, and the chain is the cost.
Copying eight elements at a time took it to 10.5 ms.  **The copy has to
go from the top down**: a block copy from the bottom up is four times
faster and it is *wrong*, because each block overwrites the next one's
source - and the boxes come out in a different order, which is not a
rounding error but a wrong answer.

**The list still left one thread per row and an O(k) shift.**  A binary
heap per row made an accepted column O(log k) and a rejected one a single
comparison against the root; it took the three layout calls from 20 ms to
14 ms.  The remaining cost was occupancy and coalescing: 300 threads is
ten warps, and every thread reads its own 52 KB row.  The shipped version
uses a **64-bit rank key** (`monotonic float, inverted index`) and one
radix-select work-group per row: eight coalesced byte passes find the
exact k-th (value, index), one more pass gathers the k winners, and a
small sort puts them best-first.  The three calls are now **4 ms total
(1.2-1.4 ms each)**, and a probe against `std::sort` on quantised random
data is exact for both `largest=1` and `largest=0`.

**Two kernels were re-shaped by measurement.**  The `Concat` used to launch
one kernel per input over the *whole output*, and every element worked out
which input owned it: 140 concats were 39 ms.  Copying each input's own
elements (the input is contiguous, so its index is its source offset) took
it to 5.5 ms.  The elementwise broadcast did a full coordinate
decomposition per element, four 64-bit divisions for an 800x800 image
(`Div` alone was 14 ms); a 2D range over the continuous run where the
second operand does not vary removed the divisions, and the same pattern
fixed the reduction (four independent accumulator chains instead of one
dependent one and a `switch` in the inner loop).

**The JIT is the other 2.6 seconds**, and it is the reason the service is
built ahead of time like the rest of the engine: `SYCL_INFER_AOT=ON` with
`SYCL_INFER_AOT_DEVICE=acm-g10` for the A770.  The first call after a
fresh start goes from 3 274 ms to 686 ms.

**Three operators were 58% of the model and all three were on the
host.**  The `Einsum` is a batched matrix multiply (`bmd,bnd->bmn` - the
device's `matmul_launch` has the transposed second operand), the
`GatherND` is a strided copy, and the `TopK` is a selection.  The first
two became device kernels and the third was rewritten; that is the
686 ms -> 282 ms.

### The end to end

The document pipeline (the one the service is for), 17 pages of a Chinese
bank's annual report, one A770, the layout, OCR and table models over HTTP
(`dev/profile_http_pipeline.py`; the service is the JIT Release build,
`2026-10-23`):

| engine | wall | per page |
|---|---:|---:|
| this service, before the upload/mask work | 29.04 s | 1 708 ms/page |
| this service, before the fusions | 15.8-16.3 s | 930-957 ms/page |
| this service, fused, layout batched 2 | **14.71-15.18 s** | **865-893 ms/page** |
| the same build, layout batch 1 | 17.22 s | 1 013 ms/page |

The first document after a service start measures ~20-24 s: the JIT compiles
the kernels and oneDNN creates the layout's primitives on first use.  The
service is long-lived, so that is a startup cost, not a per-document one.

The 865-893 split is: the table stage 505-518 ms/page, the layout 168-184,
the PDF render 76 (see below), the OCR post-process 65 and the OCR 42; the
rest is pipeline overhead.  The layout stage's own service calls are **129-141
ms at batch 1 and 146 ms/page at batch 2** (a batch of two costs ~292-294 ms).

**The PDF render was one process for the whole document.**  `load_images_from_pdf`
spawns `ceil(pages / MINERU_PDF_MIN_PAGES_PER_RENDER_PROCESS)` workers capped by
`MINERU_PDF_MAX_RENDER_PROCESSES`, and the defaults were 30 pages per worker
and 3 workers - so 17 pages rendered serially.  With 4 pages per worker and
8 workers the render is **1 616 -> 1 287 ms** (95 -> 76 ms/page) on the 17
pages; the defaults in `pdf_image_tools.py` are that now, and the environment
still overrides them.

**The content is exact.**  The pipeline's digest - every region's type, its
box and OCR text, and every table's HTML - is **byte-identical to the local
ONNXRuntime/OpenVINO reference on 15-16 of the 17 pages** (with the layout
batch at 2 as well); the other page(s) differ by one box coordinate of 0.1 px,
and
**all 45 tables match cell for cell**.  That is the Resize fix above: before
it, 33 of the 45 tables had lost a leading character, and no aggregate
reported it.

### Device memory and batch sizes

The measurements are the driver's (`xpu-smi`) and the executor's own
accounting.  One A770 16 GB, JIT Release build.  The 12-model config with
the bank document at 17 pages holds **2 776 MiB** at layout batch 2 (it was
3 906 before the mask/alias work), of which the layout is 788 MiB.

**The constant-pool bug was one half of the old figure.**  The executor
uploaded its *constants* (the weights and the folded values) through the
activation pool, and the pool's best-fit handed a 4 KB bias a 48 MB block.
A constant is never released, so the block was lost to the activations for
the rest of the process: **362 constants held 1.95 GB** in the layout model
alone.  **Tight allocation for the constant mirrors - and for the folded
outputs, which are constants too - cut the layout's first call 2 424 ->
1 361 MiB, its steady footprint 3 178 -> 1 379, the service at layout batch
1 8 159 -> 3 260 MiB and at batch 4 15 882 -> 5 662 MiB.**

**The layout's mask head was the other half, and it is now fused.**  Before
the fusion the executor was 911 MB at N=1 (124 MB of weights and an **889 MB
peak live set**) plus 38 MB of oneDNN's reordered weights.  The peak was
**ten concurrent 12M-element (48 MB) tensors**: the mask was
`bmm [1,300,32] x [1,32,40000]` -> reshape -> greater -> cast -> mul by the
grids -> expand -> reduce, and the model derives the box extrema from the
mask's `ReduceMax`/`ReduceMin`, so `drop_outputs` removed the final 48 MB
*transfer* but not the computation.  The `MaskBox` scan below removes the
whole cone (one bmm read, five `[B,300]` values), and the reshape aliases
and in-place elementwise reuse remove the remaining copies: the executor is
now **443 MiB at N=1** (peak 465, live 148).  The bmm's 48 MB stays; the
ten intermediates do not.

The layout's own footprint, measured on a layout-only service (the
executor's live + pooled bytes):

| layout batch | executor MiB | live MiB | ms/image |
|---:|---:|---:|---:|
| 1 | 443 | 148 | 150.2 |
| 2 | 818 | 154 | 154.9 |
| 4 | 1 567 | 166 | 144.1 |
| 8 | 3 066 | 194 | 124.6 |
| 12 | 5 161 | 244 | **118.9** |
| 16 | 6 812 | 241 | 129.3 |
| 24 | 9 287 | 301 | 136.3 |

The per-image cost is now ~0.42 GiB, down from ~0.94 GiB.  Batch 24 fits; a
32-image call fails after the 24-image pool has grown.  The per-image
*time* optimum is 12 (118.9 ms), but the pipeline's latency optimum is
still 2: a batch of two costs 310 ms and keeps the document moving.

The oneDNN cache is still per (weight, input shape) - a **time** cost
(primitive creation per table shape), not the memory one.  The per-model
`"dnnl": false` keeps the table models on the SYCL kernel because that is
faster for them (batch-4 table stage 8 902 ms against 9 239 with oneDNN on);
the fixed-shape layout keeps oneDNN.

**Max batch per stage** (the non-layout rows are all models resident at their
normal batches; the layout row is a layout-only service after the mask-head
fusion.  "max" is extrapolated from the per-image cost and the first
failure):

| stage | probe input | cost/image | tested | max (memory) | time per image, N=1 -> best |
|---|---|---:|---:|---:|---|
| layout | 3x800x800 | ~0.42 GiB | 24 (9.3 GB) | ~24 (32 fails) | 150 -> 118.9 ms (N=12) |
| ocr_det | 3x960x960 | ~0.17 GiB | 32 (8.3 GB) | ~32 (64 crashes the service) | 121 -> 82 ms (N=32) |
| ocr_rec | 3x48x320 | ~18.7 MiB | 128 | ~600 | 34 -> 14.7 ms (N=128) |
| ocr_cls | 3x48x192 | ~2.3 MiB | 128 | ~5000 | 6.7 -> 0.5 ms (N=32) |
| table_cls | 3x224x224 | ~7.4 MiB | 64 | ~1600 | 8.5 -> 2.1 ms (N=16) |
| table_q_cls | 3x224x224 | ~13.6 MiB | 64 | ~900 | 15.2 -> 2.9 ms (N=16) |
| table_unet | 3x288x1024 | ~140 MiB | 32 (7.6 GB) | ~80 | 50 -> 58 ms (no gain) |
| table_slanet | 3x488x488 | ~50 MiB | 64 (6.4 GB) | ~240 | 193 -> 34.5 ms (N=64) |
| slanet1m | 3x488x488 | ~50 MiB | 16 | ~240 | 109 -> 28 ms (N=16) |
| seal_det | 3x960x960 | ~165 MiB | 32 | ~70 | 130 -> 85 ms |
| orientation | 3x224x224 | ~7.5 MiB | 128 | ~1600 | 8.6 -> 2.1 ms (N=16) |
| formula | 1x384x384 | ~210 MiB | 32 (10.5 GB) | ~55 | 274 -> 79 ms (N=32) |

**What to use:** layout **2** for latency (865-893 ms/page, 0.8 GiB) or 8-12 for
throughput (125/119 ms per image, 3.1/5.2 GiB), ocr_det 8-16, ocr_rec 64-128,
ocr_cls 32, table_cls / table_q_cls 16-32, table_unet **1** (compute-bound:
its per-image time is flat), table_slanet 8-16 (5.6x per image), seal_det
4-8, orientation 16, formula 8-16.  The pipeline already batches rec at up
to 128 and the classifiers at 32; the layout's `batch_num` and the table
det/unet batching are the two places with room.

The remaining floor is the table stage: the UNET's forward is 31-100 ms per
table (it scales with the table's aspect, and its input shape changes with
it), the per-table text detection is a second model, and the client-side
cell/HTML assembly is another ~50 ms.  Two attempts to remove the per-table
detector were **rejected on content**: the PDF's own span boxes (and its
line boxes) are not the detector's segmentation - a line box that crosses a
cell border merged the cell's label with the next cell's number
(`加权平均净资产收` + `益率（%）21` instead of `加权平均净资产收益率（%）`
and `21.02`), so the detector stays.  Grouping equal-shape tables into one
UNET batch was content-neutral but no faster (the model's cost is per image),
so it was reverted too.  The layout's convolutions and matmuls are on oneDNN
now.  The `TopK` is down to 4 ms of the 129-141 (the mask head is 0.2 ms);
the next lever is the table client/GPU overlap.  bf16 is not the lever: the f32
descriptor with `fpmath_mode::bf16` was digest-exact but not faster (233 ms
against 208-225), and the 1.0-4.3x probe used bf16 *memory descriptors*,
which need a conversion pass per conv in a mixed graph.

## Running under the router

The ONNX worker is now a first-class router backend.  There is no separate
per-process model table in that topology: the router starts **one
`sycl-infer-onnx` worker per ONNX model**, exactly as it starts one
`sycl-infer` per GGUF model, and the routing key is the model id in the
binary wire request.

The worker gets the same device specification as the LLM worker:

* `--device cpu|gpu|auto` (the `--device` names of `sycl-infer`), and
* `SYCL_DEVICE_FILTER` from the router's `device.filter`, which pins the
  card.

There is no layer map for ONNX (`device.count` must be 1).  The worker also
opens the router's JSON-RPC control plane (`--control-port` /
`--control-host` / `--control-token[-file]`), so the router gets the model
identity, the device's measured free VRAM and a graceful stop.  The data
plane is `POST /v1/onnx/run`: the router reads only the wire header's model
name, forwards the body unchanged, and returns the worker's wire response.
`GET /v1/onnx/models` is aggregated.  The router's `POST /v1/chat/completions`
does not route ONNX models, and the ONNX endpoint does not do OpenAI-style
`model` rewriting, SSE or retries.

A model entry is:

```json
{
  "id": "layout", "backend": "onnx",
  "path": "/models/pp_doclayoutv3.onnx",
  "http": "127.0.0.1:18085", "control": "127.0.0.1:18086",
  "device": { "kind": "gpu", "group": "gpu.0", "filter": "level_zero:0", "count": 1 },
  "memory": { "vram_budget_mb": 4096 },
  "onnx": { "drop_outputs": ["fetch_name_2"], "dnnl": true }
}
```

The ONNX and LLM workers can share a card only if their declared budgets
fit; the router's measured admission reads the ONNX control report just as
it reads the LLM one.  The combined Docker image ships both worker binaries;
the router-only image has no local ONNX worker (it can still connect to
remote workers through a `remote` entry).

## The environment switches

| switch | what it does |
|---|---|
| `SI_ONNX_TRACE=1` | the operators, one per line, and the fusion's decisions |
| `SI_ONNX_TIME=1` | the time of every operator, in milliseconds |
| `SI_ONNX_NOFUSE=1` | no fusion (the operators as the model has them) |
| `SI_ONNX_NOFUSE_ACT=1` | no *activation* fusion (the batch norm still folds) |
| `SI_ONNX_NOMASKFUSE=1` | keep the layout model's mask head as its ONNX operators (the A/B for `MaskBox`) |
| `SI_ONNX_NOINPLACE=1` | no buffer borrowing for Reshape/Identity/elementwise (kept as a correctness A/B; a `Loop` body never borrows) |
| `SI_ONNX_DUMP=<n>` | the first n values of every operator's inputs and outputs |
| `SI_ONNX_DUMP_FILE=<dir>` | the *whole* of every operator's output, as a file |
| `SI_ONNX_SHAPE=1` | the shape of every operator's inputs and outputs |
| `SI_ONNX_LEAK=1` | the device memory the graph holds, and the slots that hold it |
| `SI_ONNX_CONSTCHK=1` | the constants' device copies are checked after every operator |
| `SI_ONNX_GUARD=<n>` | the guard band behind every output, and its check (off by default: it costs a device wait per output) |
| `SI_ONNX_PROBE=1` | the convolution writes its pre-activation value |
| `SI_ONNX_ECHO=1` | the elementwise operators write their input |
| `SI_ONNX_LOOPDBG=1` | the `Loop`'s trip count, its condition and its iterations |
| `SI_ONNX_HOSTTIME=1` | where an operator's time goes: the input uploads, the output allocations, the submit, the write-back - and how much of the call is left; also the service's `[onnx] http:` line (the request decode, the run, the response encode and its copy into httplib) |
| `SI_ONNX_UPLOADDBG=1` | the uploads of one call: the totals, the top slots by bytes, and every operator whose input preparation took more than 2 ms |
| `SI_ONNX_MIRRORCHK=1` | verify every reused device mirror against its host bytes (the `host_gen`/`dev_gen` invariant); slow, because each check waits for the device |
| `SI_ONNX_MEMDBG=1` | per-run device allocations: the total, the live set, the pool, and the buffers that are held but were not counted (they are the reused pool buffers that a constant mirror kept) |
| `PF_ONNX_DNNL=0` | keep the convolutions on the SYCL kernel (the oneDNN f32 path is the default; a grouped/depthwise convolution always uses the SYCL one) |
| `SI_ONNX_LOOPMAX=<n>` | the cap on a `Loop`'s iterations (a loop that does not terminate runs its body hundreds of times, and one iteration is all that is needed to see why) |

The model config (`--config`, JSON with comments) takes `"drop_outputs": [...]`
per model: the named outputs are not read back to the host and not put on the
wire, and the operators that only feed them are skipped.  It is how the
layout model's 48 MB mask is dropped - and the postprocess falls back to
rectangles, so check the shapes your documents need before using it on
rotated or irregular layouts.

It also takes `"dnnl": false` per model: the convolutions use the SYCL kernel
instead of oneDNN.  The oneDNN primitive (and its JIT code) is cached per
input shape, so a model whose input size changes with every request - the
OCR detector, the table UNET - accumulates gigabytes over a document.  The
shipped config keeps oneDNN for the fixed-shape layout and turns it off for
the variable-shape models; the service held 15.9 GiB with it on everywhere
and 8.2 GiB with it off (layout batch 1).

## The client side

The client is `rapid_doc/model/http_infer.py` in the document pipeline's
tree: it speaks the protocol above and returns numpy arrays, so every
`InferSession` in the pipeline can be swapped for it without touching the
code that calls it.  The pipeline's post-processing (the box filtering,
the CTC decoding, the table HTML assembly) stays where it is, so only the place
the inference happens changes.
