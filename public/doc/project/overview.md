# Full Project Overview

> **Reading Path**  
> Home: [Project Manual](index.md) | Previous: [Usage](usage.md) | Next: Finished

DL is a compact CUDA/C++ runtime for building deep-learning pieces manually. The code is split by responsibility rather than by high-level model concepts.

## The Hierarchy

```text
CORE
  vocabulary and rules

VIEW
  tensor descriptions and tensor handles

HANDLER
  memory ownership, file IO, and execution resources

OPERATOR
  GPU math operations

MODEL
  concrete model orchestration plus future abstraction layer
```

## Why This Shape

The project keeps memory explicit. That makes it easier to see what is on CPU, what is on GPU, and when copies happen.

The central object is:

```cpp
VIEW::Math
```

Every operator reads or writes `VIEW::Math`.

`VIEW::Math` does not allocate. It is connected to memory through:

```cpp
HANDLER::IO
```

Operators share execution state through:

```cpp
HANDLER::Workspace
```

## CORE In One Sentence

`CORE` defines constants, memory sizes, error enums, alignment helpers, and logging.

It should not depend on the rest of the project.

## VIEW In One Sentence

`VIEW` defines tensor metadata and tensor references.

`Shape` says what the tensor is.

`Math` says where the tensor data lives.

## HANDLER In One Sentence

`HANDLER` owns resources.

It owns the memory arenas, the copy logic, file/data loading, and GPU library handles.

## OPERATOR In One Sentence

`OPERATOR` performs math on tensors already prepared by `VIEW` and `HANDLER`.

It assumes:

- inputs contain valid GPU data
- outputs are already shaped and bound
- workspace has valid CUDA/cuDNN/cuBLASLt resources

## MODEL In One Sentence

`MODEL` currently contains `MODEL::DL`, which wraps the lower-level pieces into
a concrete trainable digit-classifier pipeline with dataset loading,
forward/backward propagation, SGD updates, checkpointing, and inference.

## Function Declaration vs Implementation

Headers:

```text
public/inc
```

Implementations:

```text
lib/src
```

Example:

```text
public/inc/OPERATOR/Relu.hpp
lib/src/OPERATOR/Relu.cu
```

The header tells you what the class exposes. The source file tells you how it works.

## The Most Common Mistake

Do not bind before shape:

```cpp
// wrong
io.bind(x);
x.getLayout().setShape(dims, 1, VIEW::F16);
```

Correct:

```cpp
x.getLayout().setShape(dims, 1, VIEW::F16);
io.bind(x);
```

Binding calculates memory size from the current shape.

## Current Operators

| Operator | Main Use |
|---|---|
| `Normalize` | scale input values |
| `Relu` | activation |
| `SGD` | in-place parameter update |
| `Softmax` | logits to probabilities |
| `CrossEntropy` | loss and probability gradient |
| `Pool` | max-pooling |
| `Conv2DRelu` | currently convolution plus bias |
| `MatrixMulBias` | linear layer math |

## What To Read For Each Question

| Question | Read |
|---|---|
| How do classes fit together? | [Architecture](architecture.md) |
| What are the error codes? | [CORE](CORE/index.md) |
| How do tensors work? | [VIEW](VIEW/index.md) |
| How does memory/copying work? | [HANDLER](HANDLER/index.md) |
| What does each operator do? | [OPERATOR](OPERATOR/index.md) |
| How does the current model work? | [MODEL](MODEL/index.md) |
| How do I write a tiny program? | [Usage](usage.md) |

---

> **End Of Guide**  
> Previous: [Usage](usage.md) | Back to: [Project Manual](index.md)
