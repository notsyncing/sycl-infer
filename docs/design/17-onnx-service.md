# 17 - The ONNX model service

The document pipeline needs *graphs*, not a transformer: a layout detector, a
text detector, a text recogniser, a table segmenturer and a formula
recogniser.  Those models come as ONNX, and this is the subsystem that runs
them.

    src/onnx/
        proto.h/.cpp      the ONNX protobuf reader
        graph.h/.cpp      the graph, the constant folding, the operator fusion
        ops.h/.cpp        the operator interface and the shared helpers
        ops_shape.cpp     the shape functions (what the outputs look like)
        ops_exec.cpp      the operators themselves
        kernels.h/.cpp    the SYCL kernels
        exec.h/.cpp       the executor: the memory plan and the run loop
        wire.h/.cpp       the wire protocol
        service.h/.cpp    the HTTP service
        main.cpp          the sycl-infer-onnx binary

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
      -> graph            the constant folding and the operator fusion
      -> ops              the shape functions and the operators
      -> kernels          the SYCL kernels (ahead-of-time compiled)
      -> exec             the memory plan and the run loop
      -> service          the HTTP endpoint

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

Two rules the copies follow:

* **The host/device copies are synchronous.**  A queued copy reads its
  source buffer later, and the host buffers here are temporaries (a vector
  in the operator's frame): an asynchronous copy reads them after they are
  destroyed, which is a use-after-free the heap notices much later.
* **The graph owns its nodes.**  The operators point at their node's
  attributes, so a node that outlived its owner is a dangling pointer.

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

One model per id, one device per model (`--device GPU`, `GPU.0`, `CPU`), and
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
* **The host/device copies are synchronous.**  A queued copy reads its
  source buffer later, and the host buffers here are temporaries (a vector
  in the operator's frame): an asynchronous copy reads them after they are
  destroyed.
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

The layout detector is the dominant cost, and on the A770 it is:

| what | time |
|---|---:|
| the whole model, one 800x800 page | **281 ms** |
| the `Conv`s (118 of them) | 107 ms |
| the `Concat`s (140 of them) | 39 ms |
| the `MatMul`s (123 of them) | 38 ms |
| the `ReduceMax`es (4 of them) | 17 ms |
| the `TopK` (300 of 13 125) | 16 ms |
| the `ReduceMin`s (2 of them) | 12 ms |
| the `GatherND` (12 million elements) | 0.7 ms |
| the first call, with the runtime's JIT | +2 589 ms |

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
bank's annual report, one A770, the layout and the OCR over HTTP:

| engine | wall | per page |
|---|---:|---:|
| ONNXRuntime, on the CPU | 15.09 s | 888 ms/page |
| this service | 16.78 s | 987 ms/page |

**The inference is 328 ms of the 987**: the layout 261 ms a call, the
detector 24 ms and the recogniser 54 ms.  The rest is the pipeline's own
PDF rendering, which is the same in both runs.

So the service is now *faster than ONNXRuntime on the CPU* on the
layout (261 ms against 360 ms) and slower on the page as a whole, and
the difference is the pipeline's overhead rather than the inference.
The target is 200 ms a page, which is 100 ms for the layout - and the
convolutions (110 ms for 118 of them) are the next thing, because they
are already at the card's speed.

## The environment switches

| switch | what it does |
|---|---|
| `SI_ONNX_TRACE=1` | the operators, one per line, and the fusion's decisions |
| `SI_ONNX_TIME=1` | the time of every operator, in milliseconds |
| `SI_ONNX_NOFUSE=1` | no fusion (the operators as the model has them) |
| `SI_ONNX_NOFUSE_ACT=1` | no *activation* fusion (the batch norm still folds) |
| `SI_ONNX_DUMP=<n>` | the first n values of every operator's inputs and outputs |
| `SI_ONNX_DUMP_FILE=<dir>` | the *whole* of every operator's output, as a file |
| `SI_ONNX_SHAPE=1` | the shape of every operator's inputs and outputs |
| `SI_ONNX_LEAK=1` | the device memory the graph holds, and the slots that hold it |
| `SI_ONNX_CONSTCHK=1` | the constants' device copies are checked after every operator |
| `SI_ONNX_GUARD=<n>` | the guard band behind every output, and its check (off by default: it costs a device wait per output) |
| `SI_ONNX_PROBE=1` | the convolution writes its pre-activation value |
| `SI_ONNX_ECHO=1` | the elementwise operators write their input |
| `SI_ONNX_LOOPDBG=1` | the `Loop`'s trip count, its condition and its iterations |
| `SI_ONNX_LOOPMAX=<n>` | the cap on a `Loop`'s iterations (a loop that does not terminate runs its body hundreds of times, and one iteration is all that is needed to see why) |

## The client side

The client is `rapid_doc/model/http_infer.py` in the document pipeline's
tree: it speaks the protocol above and returns numpy arrays, so every
`InferSession` in the pipeline can be swapped for it without touching the
code that calls it.  The pipeline's post-processing (the box filtering,
the CTC decoding, the table HTML assembly) stays where it is, so only the place
the inference happens changes.
