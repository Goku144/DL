# Full Project Overview

This project is organized as a tiny neural-network execution stack. It is not a high-level framework yet; it is closer to a runtime core where tensors, memory, and operations are explicit.

The main idea is:

```cpp
VIEW::Math tensor;
HANDLER::IO io(cpu, gpu);
OPERATOR::Relu relu(workspace);
```

The user controls tensor shapes, memory binding, data movement, and operator calls.

## Layer 1: CORE

`CORE` is the shared base layer.

It defines:

- memory sizes
- alignment constants
- error enums
- log levels
- logging macros

The most important design choice here is alignment. Tensor buffers are aligned to 256 bytes so kernels can safely use vectorized memory access such as `uint4`.

Errors are grouped by owner:

- `CORE::errIO` for `HANDLER::IO`, `Cpu`, and `Cuda`
- `CORE::errFile` for `HANDLER::File`
- `CORE::errWorkspace` for `HANDLER::Workspace`

## Layer 2: VIEW

`VIEW` does not own global memory. It describes tensor layout and points to memory owned by handlers.

### `VIEW::Shape`

`Shape` stores:

- dimensions
- strides
- rank
- dtype

Example:

```cpp
int dims[VIEW::MAX_RANK] = {2, 16, 0, 0};
x.getLayout().setShape(dims, 2, VIEW::F16);
```

### `VIEW::Math`

`Math` stores:

- CPU pointer
- GPU pointer
- CPU/GPU offsets
- bytes
- count
- shape

It is the object passed to every operator.

Important rule:

```cpp
shape first, bind second, copy third, operate fourth
```

## Layer 3: HANDLER

Handlers own resources.

### `HANDLER::Cpu`

Owns a CPU memory arena. `IO::bindCpu()` asks it for tensor slices.

### `HANDLER::Cuda`

Owns a GPU memory arena. `IO::bindGpu()` asks it for tensor slices.

### `HANDLER::IO`

Owns no tensor memory itself. It binds tensors to the CPU/GPU arenas and copies data.

Common calls:

```cpp
io.bind(x);
io.copyHostToDevice(x);
io.copyDeviceToHost(x);
io.copyHalfToCpuFloat(xFloat, x);
```

The raw pointer overloads now copy only data. They do not change tensor shape.

### `HANDLER::Workspace`

Owns execution resources:

- CUDA stream
- cuDNN handle
- cuBLASLt handle
- scratch buffer

Operators use the same workspace so the project can coordinate stream execution.

### `HANDLER::File`

Loads raw files, CSV metadata, and images. It bridges dataset storage into `VIEW::Math`.

## Layer 4: OPERATOR

Operators read and write `VIEW::Math` GPU memory.

They do not allocate output tensors automatically. The caller must create, shape, and bind outputs before calling the operator.

### Operator Summary

| Operator | Forward | Backward / Update |
|---|---|---|
| `Normalize` | `out = in / scalar` | none |
| `Relu` | `out = max(in, 0)` | `dIn = gate(in) * dOut` |
| `Conv2DRelu` | currently `conv + bias` | `dIn`, `dWeight`, `dBias` |
| `Pool` | max-pool | max-pool backward |
| `Softmax` | probabilities | softmax backward |
| `CrossEntropy` | loss + `dProb` | fused in one call |
| `SGD` | none | `weight -= lr * grad` |
| `MatrixMulBias` | `out = in * weight + bias` | `dIn`, `dWeight`, `dBias` |

## How To Use An Operator

```cpp
VIEW::Math x, y;
int dims[VIEW::MAX_RANK] = {16, 0, 0, 0};

x.getLayout().setShape(dims, 1, VIEW::F16);
y.getLayout().setShape(dims, 1, VIEW::F16);

io.bind(x);
io.bind(y);

__half *xCpu = (__half *)x.getCpuPtr();
for(int i = 0; i < 16; i++) {
  xCpu[i] = __float2half((float)i);
}

io.copyHostToDevice(x);

OPERATOR::Relu relu(workspace, y, x);
relu.forward();

VIEW::Math yFloat;
io.copyHalfToCpuFloat(yFloat, y);
io.printData(yFloat);
```

## Function Declaration vs Function Use

Declarations live in `public/inc`.

Implementations live in `lib/src`.

Example:

- declaration: `public/inc/OPERATOR/Relu.hpp`
- implementation: `lib/src/OPERATOR/Relu.cu`
- use: `app/src/app.cu`

The headers describe what the class exposes. The `.cu` files hold kernels and library calls. The app test shows how they work together.

## Current Readiness Test

`app/src/app.cu` is a full readiness test for the operators. It creates small tensors, fills deterministic input data, runs every operator, copies results back, and compares against CPU reference values.

The test is intentionally direct. It is not a framework test runner. It is a living example of how to create handlers, bind tensors, use operators, and validate outputs.

