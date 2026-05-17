# OPERATOR

`public/inc/OPERATOR` and `lib/src/OPERATOR` define GPU operations.

Every operator follows the same pattern:

1. Create and bind tensors.
2. Put input data in CPU memory.
3. Copy inputs to GPU.
4. Create operator with `Workspace`.
5. Attach operands.
6. Call forward/backward/update.
7. Copy outputs back if needed.

## Operators

### `Normalize`

Computes:

```cpp
out = in / scalar
```

Input and output are F16 tensors. Count should be a multiple of 8.

### `Relu`

Forward:

```cpp
out = max(in, 0)
```

Backward:

```cpp
dIn = in > 0 ? dOut : 0
```

### `Conv2DRelu`

Current forward behavior is convolution plus bias:

```cpp
out = conv2d(in, weight) + bias
```

The class creates a ReLU descriptor, but does not currently call cuDNN activation in `forward()`.

Backward computes `dIn`, `dWeight`, and `dBias`.

### `Pool`

Max-pooling forward and backward using cuDNN.

### `Softmax`

Row softmax using cuDNN. Rank-2 tensors are treated as `[batch, classes]`.

### `CrossEntropy`

Consumes probabilities and integer labels. Writes scalar loss and `dProb`.

### `SGD`

In-place weight update:

```cpp
weight -= lr * grad
```

### `MatrixMulBias`

Linear layer:

```cpp
out = in * weight + bias
```

Shape contract:

```cpp
in     = [N, K]
weight = [K, M]
bias   = [M]
out    = [N, M]
```

Backward:

```cpp
dIn     = dOut * weight^T
dWeight = in^T * dOut
dBias   = sum(dOut over batch)
```

