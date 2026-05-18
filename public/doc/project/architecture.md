# Architecture

> **Reading Path**  
> Home: [Project Manual](index.md) | Previous: [Project Manual](index.md) | Next: [CORE](CORE/index.md)

This file explains how to orchestrate the classes together. It is the project-level map.

## High-Level Flow

The runtime is explicit. Nothing hidden creates tensors or memory for you.

```cpp
HANDLER::Cpu cpu(CORE::MEMORY_32_MB);
HANDLER::Cuda gpu(CORE::MEMORY_32_MB);
HANDLER::IO io(cpu, gpu);
HANDLER::File file;
HANDLER::Workspace workspace(io, file, CORE::MEMORY_32_MB);
```

Then tensors:

```cpp
VIEW::Math x, y;
int dims[VIEW::MAX_RANK] = {16, 0, 0, 0};
x.getLayout().setShape(dims, 1, VIEW::F16);
y.getLayout().setShape(dims, 1, VIEW::F16);
io.bind(x);
io.bind(y);
```

Then data:

```cpp
__half *xCpu = (__half *)x.getCpuPtr();
for(int i = 0; i < 16; i++)
  xCpu[i] = __float2half((float)i);
io.copyHostToDevice(x);
```

Then operation:

```cpp
OPERATOR::Relu relu(workspace, y, x);
relu.forward();
```

Then output inspection:

```cpp
VIEW::Math yFloat;
io.copyHalfToCpuFloat(yFloat, y);
io.printData(yFloat);
```

## Dependency Direction

The intended hierarchy is:

```text
CORE
  used by VIEW and HANDLER

VIEW
  uses CORE
  used by HANDLER and OPERATOR

HANDLER
  uses CORE and VIEW
  owns memory and execution resources

OPERATOR
  uses VIEW tensors and HANDLER::Workspace

MODEL
  future layer/model abstraction built on OPERATOR + HANDLER + VIEW
```

Operators should not allocate user tensors. They should operate on already-shaped, already-bound `VIEW::Math` objects.

## Object Ownership

`VIEW::Math` does not own memory. It only stores pointers and metadata.

`HANDLER::Cpu` owns CPU arena memory.

`HANDLER::Cuda` owns GPU arena memory.

`HANDLER::IO` binds a `VIEW::Math` object to arena slices owned by `Cpu` and `Cuda`.

`HANDLER::Workspace` owns execution resources: CUDA stream, cuDNN handle, cuBLASLt handle, and scratch memory.

`OPERATOR::*` classes own operator descriptors when needed, but they do not own the tensors passed into them.

## Error Flow

Handlers store error state:

```cpp
if(io.peekErr() != CORE::ioSuccess)
  io.info();
```

Use `peekErr()` to inspect without clearing.

Use `getErr()` to inspect and clear.

Operators currently mostly rely on CUDA/cuDNN/cuBLASLt return paths and stream synchronization. They do not all write into a shared project error enum.

## Shape And Binding Order

Correct:

```cpp
x.getLayout().setShape(dims, 1, VIEW::F16);
io.bind(x);
```

Wrong:

```cpp
io.bind(x);
x.getLayout().setShape(dims, 1, VIEW::F16);
```

Binding computes `count` and `bytes` from the current shape. If the shape is wrong when binding happens, the memory size is wrong.

## Copying Data

Use this when CPU and GPU memory already belong to the same tensor:

```cpp
io.copyHostToDevice(x);
```

Use raw pointer overloads when you have an existing CPU/GPU pointer:

```cpp
io.copyHostToDevice(x, rawHostPtr, n, VIEW::F16);
```

The raw pointer overloads copy data only. They should not change tensor shape.

## Where MODEL Will Fit

`MODEL` is planned as the future high-level layer/model API.

Expected responsibilities:

- own layer objects
- connect operator outputs to later inputs
- run forward propagation
- run backward propagation
- manage trainable parameters and gradients
- coordinate optimizers
- hide repetitive tensor wiring without hiding memory ownership completely

`MODEL` should be built after the lower layers are stable.

---

> **Continue Reading**  
> Previous: [Project Manual](index.md) | Next: [CORE](CORE/index.md)
