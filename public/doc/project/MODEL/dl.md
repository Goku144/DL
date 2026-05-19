# MODEL::DL

> Namespace: [MODEL](index.md)

`MODEL::DL` is the current concrete model orchestration class. The older
project overview described `MODEL` as future work, but the implementation now
contains a trainable convolutional digit classifier built directly on top of
`VIEW`, `HANDLER`, and `OPERATOR`.

`DL` owns the runtime handlers, dataset tensors, trainable parameters,
intermediate activations, gradients, loss storage, checkpointing logic, training
loop, and single-image estimate path.

## General Architecture

`MODEL::DL` turns the low-level runtime blocks into one complete learning
pipeline:

```text
CSV metadata
  -> File::readCsv
  -> labels copied to GPU

image paths
  -> File::readImages
  -> uint8 CPU image store
  -> File::copyImageToDevice
  -> F16 GPU image store

forward batch
  -> pullGpuImage
  -> Normalize
  -> Conv2DRelu
  -> Relu
  -> Pool
  -> MatrixMulBias
  -> Relu
  -> MatrixMulBias
  -> Softmax

training
  -> CrossEntropy
  -> backward operators
  -> SGD updates
  -> optional checkpoint

estimate
  -> optional checkpoint load
  -> single image load
  -> forward
  -> copy probabilities to CPU
  -> log class scores
```

The class is intentionally explicit. It does not introduce a generic layer
abstraction yet. Instead, it wires the existing operators directly.

## Shape And Parameter Constants

Implementation-local constants in `lib/src/MODEL/DL.cu` define the network:

| Constant | Meaning |
|---|---|
| `MODEL_CONV_FILTERS = 16` | Number of convolution output channels. |
| `MODEL_CONV_KERNEL = 3` | Convolution filter height/width. |
| `MODEL_POOL_WINDOW = 2` | Pooling window and downsample factor. |
| `MODEL_HIDDEN = 128` | Hidden feature count in the first linear layer. |
| `MODEL_OUTPUT_CLASSES = 16` | Softmax output columns. |
| `MODEL_REAL_CLASSES = 10` | Real digit classes reported to the user. |

The model trains with 16 output columns even though only classes `0..9` are
real digit classes. The extra columns are padding capacity for vectorized
16-wide loss logic.

## Owned State

`DL` owns all major runtime resources:

| Field Group | Meaning |
|---|---|
| `cpu`, `gpu`, `io`, `file`, `workspace` | Runtime handlers and execution resources. |
| `normalizeOp`, `convOp`, `relu0Op`, `poolOp`, `fc1Op`, `relu1Op`, `fc2Op`, `softmaxOp`, `lossOp`, `sgd*Op` | Reusable operator objects allocated once after tensors are bound. |
| `filePaths`, `Y` | Dataset path strings and labels. |
| `x` | Active input batch tensor. |
| `w0`, `b0`, `w1`, `b1`, `w2`, `b2` | Trainable parameters. |
| `z0`, `a0`, `p`, `z1`, `a1`, `z2`, `out` | Forward intermediate tensors. |
| `dz0`, `dw0`, `db0`, `da0`, `dp`, `dz1`, `dw1`, `db1`, `da1`, `dz2`, `dw2`, `db2` | Backward gradients. |
| `L` | Scalar FLOAT loss tensor. |
| `imageOffset`, `imageBatch`, `modImage` | Training batch cursor and dataset size. |

## Public Interface

```cpp
DL(size_t imageBatch, const char *csvPath = "public/target/meta/test.csv");
~DL();

size_t getImageBatch() const;
void train(size_t iterations, float learningRate = 0.01f, size_t checkpointEvery = 0, const char *checkpointPrefix = "public/checkpoints/dl");
void estimate(const char *imagePath, const char *checkpointPath = NULL);
```

## Private Interface

```cpp
void initializeOperators();
void destroyOperators();
void setActiveBatch(size_t batch);
void initializeParameters();
void saveCheckpoint(const char *path);
bool loadCheckpoint(const char *path);
void forward(size_t batchOffset, size_t batch);
void backward();
void update(float learningRate);
```

## File-Local Helper Functions

These helpers are not public API. They exist to keep the model implementation
compact.

### `static void setShape(VIEW::Math& math, int d0, int d1, int d2, int d3, int rank, VIEW::DType dtype)`

**Type:** File-local helper.

**Core Meaning:** Assign a complete `VIEW::Shape` to a tensor.

**Implementation Logic:** Builds a four-slot `dims` array, then calls
`math.getLayout().setShape(...)`. It changes tensor metadata only; it does not
allocate memory.

**Dependencies & Propagation:** Depends on `VIEW::Shape`. Every helper that
binds or relayouts tensors goes through this path, so bad dimensions propagate
to all later memory-size and operator-descriptor calculations.

**Edge Cases/Assumptions:** Existing memory is not resized here. If the tensor
is already bound, changing the shape can make the metadata disagree with the
allocated capacity.

### `static void bindGpu(HANDLER::IO *io, VIEW::Math& math, ...)`

**Type:** File-local helper.

**Core Meaning:** Shape a tensor and allocate GPU memory for it.

**Implementation Logic:** Calls `setShape(...)`, then `io->bindGpu(math)`.

**Dependencies & Propagation:** Depends on `HANDLER::IO` and `HANDLER::Cuda`.
The resulting pointer is later consumed by operators.

**Edge Cases/Assumptions:** Assumes `io` is non-null and has a GPU handler.
Errors are stored in `IO`, but this helper does not inspect them.

### `static void bindBoth(HANDLER::IO *io, VIEW::Math& math, ...)`

**Type:** File-local helper.

**Core Meaning:** Shape a tensor and allocate both CPU and GPU memory.

**Implementation Logic:** Calls `setShape(...)`, then `io->bind(math)`.

**Dependencies & Propagation:** Used for parameters and scalar loss where CPU
and GPU access are both needed.

**Edge Cases/Assumptions:** As with `bindGpu`, allocation errors are not handled
inside the helper.

### `static void relayout(VIEW::Math& math, int d0, int d1, int d2, int d3, int rank, VIEW::DType dtype)`

**Type:** File-local helper.

**Core Meaning:** Reinterpret an already-bound tensor with a new logical shape.

**Implementation Logic:** Calls `setShape(...)`, then updates `count` to
`d0 * stride0`.

**Dependencies & Propagation:** This is central to dynamic batch sizing and
flattening pooled feature maps before fully connected layers.

**Edge Cases/Assumptions:** It does not check that the new count fits the
original allocation. Correctness depends on the constructor binding buffers at
the maximum batch size.

### `static __global__ void initParamKernel(__half *dst, size_t count, float scale, unsigned int seed)`

**Type:** CUDA kernel.

**Core Meaning:** Deterministically initialize F16 parameter tensors.

**Implementation Logic:** For each element, derives a pseudo-random integer from
the element index and seed, maps part of it to roughly `[-1, 1]`, multiplies by
`scale`, and stores the result as `__half`.

**Dependencies & Propagation:** Used by `initializeParameters()` for weights.
The generated values determine the starting optimization state.

**Edge Cases/Assumptions:** This is not a statistical RNG with external state.
It is deterministic and index-based.

### `static __global__ void zeroHalfKernel(__half *dst, size_t count)`

**Type:** CUDA kernel.

**Core Meaning:** Fill an F16 tensor with zero.

**Implementation Logic:** Each thread writes one `__half(0)` if its index is in
range.

**Dependencies & Propagation:** Used for bias initialization.

**Edge Cases/Assumptions:** Assumes valid GPU pointer and count.

### `static __global__ void halfToFloatKernel(float *dst, const __half *src, size_t count)`

**Type:** CUDA kernel.

**Core Meaning:** Convert F16 values to FLOAT values.

**Implementation Logic:** Each thread reads one half and writes one float.

**Dependencies & Propagation:** Used in training report logic to inspect
probabilities on CPU after a device copy.

**Edge Cases/Assumptions:** Temporary float memory must be allocated by caller.

### `static void initParam(VIEW::Math& math, float scale, unsigned int seed)`

**Type:** File-local helper.

**Core Meaning:** Launch parameter initialization for one tensor.

**Implementation Logic:** Computes a 256-thread grid over `math.getCount()` and
launches `initParamKernel`.

**Dependencies & Propagation:** Called by `initializeParameters()`.

**Edge Cases/Assumptions:** Launches on the default stream rather than the
workspace stream.

### `static void zeroParam(VIEW::Math& math)`

**Type:** File-local helper.

**Core Meaning:** Launch zero fill for one F16 tensor.

**Implementation Logic:** Computes a 256-thread grid and launches
`zeroHalfKernel`.

**Dependencies & Propagation:** Called by `initializeParameters()`.

**Edge Cases/Assumptions:** Launches on the default stream.

### `static size_t tensorBytes(VIEW::Math& math)`

**Type:** File-local helper.

**Core Meaning:** Compute the unaligned serialized byte size of a tensor.

**Implementation Logic:** Returns `math.getCount() * math.getLayout().getDType()`.

**Dependencies & Propagation:** Defines checkpoint payload sizes.

**Edge Cases/Assumptions:** Uses logical count and dtype, not aligned
`math.getBytes()`.

### `static size_t checkpointBytes(...)`

**Type:** File-local helper.

**Core Meaning:** Compute the exact checkpoint payload size.

**Implementation Logic:** Adds `tensorBytes` for `w0`, `b0`, `w1`, `b1`, `w2`,
and `b2`.

**Dependencies & Propagation:** Used by both checkpoint save and load. The
checkpoint format is fixed by this order.

**Edge Cases/Assumptions:** No architecture metadata is stored in the file. The
current model shape must match the checkpoint.

### `static bool restoreTensor(HANDLER::IO *io, VIEW::Math& checkpoint, size_t& offset, VIEW::Math& math)`

**Type:** File-local helper.

**Core Meaning:** Restore one parameter tensor from a raw checkpoint buffer.

**Implementation Logic:** Checks that `offset + tensorBytes(math)` fits inside
the checkpoint tensor, copies the matching bytes into the parameter CPU buffer,
advances `offset`, then copies that CPU parameter to GPU.

**Dependencies & Propagation:** Used by `loadCheckpoint()` for every parameter.

**Edge Cases/Assumptions:** Assumes parameter tensors have CPU and GPU memory.
Failure leaves earlier restored parameters already modified.

## Operator Lifecycle

`DL` pre-creates its operator objects after all tensors are bound. This avoids
reconstructing cuDNN descriptors, cuBLASLt wrapper objects, loss objects, and SGD
wrappers every time `forward`, `backward`, `train`, or `update` runs.

The operators still observe the latest tensor shapes because they store
pointers to `VIEW::Math` objects. `setActiveBatch` and `relayout` mutate those
tensor layouts in place, so reused operators see the current batch shape when
their `forward` or `backward` methods set descriptors.

## Constructor And Destructor

### `MODEL::DL::DL(size_t imageBatch, const char *csvPath)`

**Type:** Constructor.

**Core Meaning:** Build a complete model runtime and load the initial dataset.

**Implementation Logic:**

1. Allocates `HANDLER::Cpu`, `HANDLER::Cuda`, `HANDLER::IO`, `HANDLER::File`,
   and `HANDLER::Workspace` using model default memory sizes.
2. Stores `imageBatch`, falling back to `CORE::MODEL_IMAGE_BATCH_DEFAULT` if the
   caller passes zero.
3. Reads the CSV into `filePaths` and `Y`.
4. Binds `Y` on GPU and copies labels from host to device.
5. Reads images listed in `filePaths` under `public/target/meta`, requesting one
   channel.
6. Converts loaded uint8 CPU images to F16 GPU image storage.
7. Sets `modImage` from label row count.
8. Reads image dimensions from `File`.
9. Computes convolution, pooling, and flattened feature sizes.
10. Binds every forward tensor, gradient tensor, parameter tensor, and loss
    tensor.
11. Calls `initializeOperators()` to allocate and attach reusable operators.
12. Calls `initializeParameters()`.

**Dependencies & Propagation:** This constructor wires the whole lower stack
together. `File` determines image shape, `IO` allocates all tensors, `Workspace`
provides operator resources, and all later `forward/backward/train/estimate`
paths depend on this setup.

**Edge Cases/Assumptions:** Many file and IO operations store errors but are not
checked immediately in the constructor. If CSV or image loading fails, later
shape and allocation work may be based on invalid or zero metadata.

### `MODEL::DL::~DL()`

**Type:** Destructor.

**Core Meaning:** Release owned runtime resources.

**Implementation Logic:** Calls `destroyOperators()`, then deletes `cpu`, `gpu`,
`io`, `file`, and `workspace`.

**Dependencies & Propagation:** The handler destructors release arena memory and
GPU resources.

**Edge Cases/Assumptions:** `Math` objects retain raw pointers but do not own
memory. After handler deletion, those pointers are invalid.

## Public Methods

### `size_t MODEL::DL::getImageBatch() const`

**Type:** Instance method.

**Core Meaning:** Report the configured training batch size.

**Implementation Logic:** Returns `this->imageBatch`.

**Dependencies & Propagation:** Used by callers that need to inspect model
batching.

**Edge Cases/Assumptions:** It returns configuration, not the size of the next
effective batch.

### `void MODEL::DL::train(size_t iterations, float learningRate, size_t checkpointEvery, const char *checkpointPrefix)`

**Type:** Instance method.

**Core Meaning:** Train the model with mini-batch SGD and optional checkpointing.

**Implementation Logic:**

1. Returns immediately if no images are loaded or `iterations == 0`.
2. For each iteration, chooses `batch = min(imageBatch, modImage)`.
3. Computes the dataset offset from `imageOffset`; wraps to zero if the batch
   would cross the dataset end.
4. Runs `forward(offset, batch)`.
5. Reuses `lossOp`, sets the target batch window, then computes loss and
   probability gradient.
6. Runs `backward()`.
7. Runs `update(learningRate)`.
8. Synchronizes the workspace stream.
9. On report intervals or the final iteration, copies probabilities to CPU
   float memory, copies loss to CPU, computes accuracy and average confidence
   across real classes `0..9`, and logs training status.
10. On checkpoint intervals, writes a checkpoint file named
    `checkpointPrefix_iter_<iteration>.bin`.
11. Advances `imageOffset`.

**Dependencies & Propagation:** This method coordinates the entire learning
graph. Changes to `forward`, `backward`, `CrossEntropy`, or `SGD` directly
change training semantics.

**Edge Cases/Assumptions:** Dataset traversal is sequential, not shuffled.
Accuracy ignores classes `10..15`. Reporting allocates temporary host/device
buffers. The checkpoint interval controls both reporting and saving behavior.

### `void MODEL::DL::estimate(const char *imagePath, const char *checkpointPath)`

**Type:** Instance method.

**Core Meaning:** Run single-image inference and log class probabilities.

**Implementation Logic:**

1. Returns with a warning if `imagePath` is null.
2. If `checkpointPath` is non-null, calls `loadCheckpoint(checkpointPath)`.
3. Builds a one-row CPU `VIEW::Math` containing the image path string.
4. Calls `File::readImages(pathMath, NULL, 1)` to load that path as a one-image
   dataset.
5. Calls `File::copyImageToDevice()`.
6. Runs `forward(0, 1)`.
7. Synchronizes the workspace stream.
8. Converts `out` from GPU F16 to CPU FLOAT.
9. Scans classes `0..9`, logs every probability, and logs the best class and
   confidence.

**Dependencies & Propagation:** Uses checkpoint restore, file image loading,
the normal forward graph, and IO conversion. It is the public inference path.

**Edge Cases/Assumptions:** `loadCheckpoint` return value is ignored. File
errors from `readImages` and `copyImageToDevice` are not checked before
`forward`, so a failed image load can leave stale image data active. Only real
classes `0..9` are reported even though softmax has 16 outputs.

## Private Methods

### `void MODEL::DL::initializeOperators()`

**Type:** Private instance method.

**Core Meaning:** Allocate every reusable operator once and attach its stable
tensor operands.

**Implementation Logic:** Creates `Normalize`, `Conv2DRelu`, two `Relu`
operators, `Pool`, two `MatrixMulBias` operators, `Softmax`, `CrossEntropy`, and
six `SGD` operators. It also sets static convolution/pooling configs and
attaches backward/gradient operands.

**Dependencies & Propagation:** Depends on all tensors already being shaped and
bound. The resulting operator pointers are used by `forward`, `backward`,
`train`, and `update`.

**Edge Cases/Assumptions:** Operators store tensor pointers, not copies. Tensor
layout changes made later by `relayout` are visible to the same operators.

### `void MODEL::DL::destroyOperators()`

**Type:** Private instance method.

**Core Meaning:** Release all reusable operator objects.

**Implementation Logic:** Deletes all SGD, loss, softmax, linear, activation,
pooling, convolution, and normalization operator pointers, then resets each
pointer to `NULL`.

**Dependencies & Propagation:** Called by the destructor before runtime handlers
are deleted.

**Edge Cases/Assumptions:** Deleting `NULL` is safe, so partially initialized
operator state can still be cleaned up.

### `void MODEL::DL::setActiveBatch(size_t batch)`

**Type:** Private instance method.

**Core Meaning:** Reinterpret all batch-dependent tensors for the active batch
size.

**Implementation Logic:** Derives image dimensions and pooled dimensions from
`File`, computes flattened pool size, then calls `relayout` on input,
activations, gradients, pooled tensors, logits, output probabilities, and loss
gradient tensors.

**Dependencies & Propagation:** Called at the start of `forward`. This controls
the shapes that later operators see.

**Edge Cases/Assumptions:** It changes metadata and count only. It assumes the
constructor allocated enough memory for the largest configured batch.

### `void MODEL::DL::initializeParameters()`

**Type:** Private instance method.

**Core Meaning:** Initialize trainable weights and biases.

**Implementation Logic:** Calls `initParam` on `w0`, `w1`, and `w2` with
different scales/seeds; calls `zeroParam` on all biases; synchronizes the device.

**Dependencies & Propagation:** Sets initial training state before any forward
pass.

**Edge Cases/Assumptions:** Uses deterministic index-based pseudo-random values.
The initialization kernels use the default stream.

### `void MODEL::DL::forward(size_t batchOffset, size_t batch)`

**Type:** Private instance method.

**Core Meaning:** Compute probabilities for a dataset batch.

**Implementation Logic:**

1. Calls `setActiveBatch(batch)`.
2. Pulls `batch` GPU images from `File` into `x`.
3. Normalizes `x` in place by dividing by 255.
4. Runs convolution plus bias into `z0`.
5. Runs ReLU into `a0`.
6. Runs max-pooling into `p`.
7. Relayouts `p` from `[N,C,H,W]` to `[N,flat]`.
8. Runs first linear layer into `z1`.
9. Runs ReLU into `a1`.
10. Runs final linear layer into `z2`.
11. Runs softmax into `out`.

**Dependencies & Propagation:** This method connects `File`, `Normalize`,
`Conv2DRelu`, `Relu`, `Pool`, `MatrixMulBias`, and `Softmax`.

**Edge Cases/Assumptions:** `Normalize` reads and writes `x` in place. Flattening
is a metadata reinterpretation, not a data copy. File pull errors are not checked
inside the method.

### `void MODEL::DL::backward()`

**Type:** Private instance method.

**Core Meaning:** Propagate gradients from output loss gradient back through all
trainable layers.

**Implementation Logic:**

1. Reads active batch from `out` layout.
2. Recomputes pool and flat dimensions.
3. Runs final linear backward to produce `da1`, `dw2`, `db2`.
4. Runs hidden ReLU backward to produce `dz1`.
5. Relayouts `p` and `dp` to `[N,flat]`.
6. Runs first linear backward to produce `dp`, `dw1`, `db1`.
7. Relayouts `p` and `dp` back to `[N,C,H,W]`.
8. Runs max-pool backward to produce `da0`.
9. Runs first ReLU backward to produce `dz0`.
10. Runs convolution backward to produce `dw0`, `db0`, and input gradient.

**Dependencies & Propagation:** Depends on forward intermediates being intact.
The gradients produced here are consumed by `update`.

**Edge Cases/Assumptions:** Uses `x` as both the original input and convolution
input gradient destination in `setGradOperand`; this overwrites the active input
gradient storage.

### `void MODEL::DL::update(float learningRate)`

**Type:** Private instance method.

**Core Meaning:** Apply one SGD step to every trainable tensor.

**Implementation Logic:** Reuses the six `OPERATOR::SGD` objects created during
`initializeOperators()`, then calls `update(learningRate)` on all of them.

**Dependencies & Propagation:** Mutates `w0`, `b0`, `w1`, `b1`, `w2`, and `b2`.
All future forward passes use the updated parameters.

**Edge Cases/Assumptions:** No gradient scaling beyond what `CrossEntropy`
already applies. No clipping or optimizer state.

### `void MODEL::DL::saveCheckpoint(const char *path)`

**Type:** Private instance method.

**Core Meaning:** Serialize the trainable parameters to a raw binary file.

**Implementation Logic:**

1. Ensures `public/checkpoints` exists.
2. Computes payload size from parameter logical counts and dtypes.
3. Allocates a CPU staging buffer.
4. Synchronizes the workspace stream.
5. Copies `w0`, `b0`, `w1`, `b1`, `w2`, and `b2` from GPU to the staging buffer
   in fixed order.
6. Opens the destination path with create/truncate mode.
7. Writes all bytes, closes, frees staging memory, and logs success.

**Dependencies & Propagation:** Checkpoints are later consumed by
`loadCheckpoint` and `estimate`.

**Edge Cases/Assumptions:** The file has no header, magic number, version,
shape metadata, or checksum. Restore correctness depends on exact model shape
and parameter order.

### `bool MODEL::DL::loadCheckpoint(const char *path)`

**Type:** Private instance method.

**Core Meaning:** Restore trainable parameters from a raw checkpoint file.

**Implementation Logic:**

1. Returns false if `path` is null.
2. Reads the file into a CPU `VIEW::Math` as CHAR data.
3. Checks file read errors.
4. Computes expected payload bytes from current parameter tensors.
5. Verifies checkpoint byte count matches expected bytes.
6. Restores `w0`, `b0`, `w1`, `b1`, `w2`, and `b2` in fixed order through
   `restoreTensor`.
7. Logs success or failure and returns the result.

**Dependencies & Propagation:** Changes model weights used by training and
estimate.

**Edge Cases/Assumptions:** Partial restore can occur before a later tensor
fails. Caller must check the returned bool; `estimate` currently does not.

## End-To-End Training Flow

```text
DL::train
  -> DL::forward
    -> File::pullGpuImage
    -> Normalize::normByScalar
    -> Conv2DRelu::forward
    -> Relu::forward
    -> Pool::maxForward
    -> MatrixMulBias::forward
    -> Relu::forward
    -> MatrixMulBias::forward
    -> Softmax::forward
  -> CrossEntropy::forwardBackward
  -> DL::backward
    -> MatrixMulBias::backward
    -> Relu::backward
    -> MatrixMulBias::backward
    -> Pool::maxBackward
    -> Relu::backward
    -> Conv2DRelu::backward
  -> DL::update
    -> SGD::update for all weights and biases
  -> optional DL::saveCheckpoint
```

## End-To-End Estimate Flow

```text
DL::estimate
  -> optional DL::loadCheckpoint
  -> create one-row path Math
  -> File::readImages
  -> File::copyImageToDevice
  -> DL::forward(0, 1)
  -> IO::copyHalfToCpuFloat
  -> log real class probabilities
```

## Important Behavioral Notes

- `MODEL::DL` is implemented in code even though older docs described `MODEL`
  as future work.
- The model uses 16 softmax columns and reports only 10 digit classes.
- Image tensors are one-channel F16 tensors normalized by `255.0`.
- `File::readImages` stores image data in planar NCHW order.
- Checkpoints are raw parameter bytes with no schema metadata.
- `estimate` should be used with valid image paths; otherwise stale loaded image
  state can be reused because file errors are not checked before forward.
