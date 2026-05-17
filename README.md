# DL

DL is a small CUDA/C++ neural-network runtime. It is built around explicit tensor views, arena-style CPU/GPU memory handlers, and GPU operators backed by CUDA kernels, cuDNN, and cuBLASLt.

The project is intentionally low-level: you create handlers, shape tensors, bind memory, copy data, attach tensors to operators, then launch operations.

## Current Status

The app test harness currently reports:

```text
Operator readiness: ready
```

Covered operators:

- `Normalize`
- `Relu`
- `SGD`
- `Softmax`
- `CrossEntropy`
- `Pool`
- `Conv2DRelu`
- `MatrixMulBias`

## Project Layout

```text
app/src/app.cu              Operator readiness test harness
public/inc/CORE             Shared enums, errors, alignment, logging
public/inc/VIEW             Tensor shape and tensor view classes
public/inc/HANDLER          CPU/GPU memory, IO, file, workspace handlers
public/inc/OPERATOR         Public operator class declarations
lib/src                     Implementations and CUDA kernels
public/doc/project          Project documentation
Makefile                    Build, run, dataset, and clean targets
```

## Architecture

The stack has four main layers.

### CORE

`CORE` contains common definitions:

- memory sizes such as `CORE::MEMORY_32_MB`
- alignment constants such as `CORE::ALIGNE_TO_256`
- error enums such as `CORE::errIO`
- logging helpers

### VIEW

`VIEW::Shape` stores tensor metadata: dimensions, strides, rank, and dtype.

`VIEW::Math` stores a tensor view: CPU pointer, GPU pointer, offsets, byte count, element count, and shape.

`VIEW::Math` does not allocate memory by itself.

### HANDLER

Handlers own resources:

- `HANDLER::Cpu`: CPU arena allocator
- `HANDLER::Cuda`: GPU arena allocator
- `HANDLER::IO`: tensor binding and memory copies
- `HANDLER::File`: raw file, CSV, and image loading
- `HANDLER::Workspace`: CUDA stream, cuDNN handle, cuBLASLt handle, scratch buffer

### OPERATOR

Operators read and write `VIEW::Math` GPU memory. Outputs must already exist, have shape, and be bound before launching an operator.

## Basic Usage

```cpp
HANDLER::Cpu cpu(CORE::MEMORY_32_MB);
HANDLER::Cuda gpu(CORE::MEMORY_32_MB);
HANDLER::IO io(cpu, gpu);
HANDLER::File file;
HANDLER::Workspace workspace(io, file, CORE::MEMORY_32_MB);

VIEW::Math x;
int dims[VIEW::MAX_RANK] = {16, 0, 0, 0};
x.getLayout().setShape(dims, 1, VIEW::F16);
io.bind(x);

__half *xCpu = (__half *)x.getCpuPtr();
for(int i = 0; i < 16; i++) {
  xCpu[i] = __float2half((float)i);
}

io.copyHostToDevice(x);
```

## Build And Test

Build the library:

```bash
make lib
```

Build and run the app readiness harness:

```bash
make app
```

If `nvcc` is installed but not on `PATH`, pass it explicitly:

```bash
make app NVCC=/usr/local/cuda/bin/nvcc
```

Clean generated build outputs:

```bash
make clean
```

## How To Search This Repo

Use `rg` first. It is faster and cleaner than recursive `grep`.

Find a class:

```bash
rg -n "class .*MatrixMulBias|MatrixMulBias" public/inc lib/src app/src
```

Find an operator implementation:

```bash
rg -n "void OPERATOR::Relu::forward|reluKernel" lib/src/OPERATOR
```

Find public declarations:

```bash
rg -n "class|void|getErr|setOperand|forward|backward" public/inc
```

Find all CUDA kernels:

```bash
rg -n "__global__" lib/src
```

Find memory binding and copy behavior:

```bash
rg -n "bind\\(|copyHostToDevice|copyDeviceToHost|copyHalfToCpuFloat" public/inc lib/src
```

Find shape usage:

```bash
rg -n "setShape|getDim|getStride|getCount|getBytes" public/inc lib/src app/src
```

List files:

```bash
rg --files
```

List only headers:

```bash
rg --files public/inc
```

List only operator source files:

```bash
rg --files lib/src/OPERATOR
```

## Documentation

Start here:

- [Project docs index](public/doc/project/index.md)
- [Full overview](public/doc/project/huge.md)
- [Core guide](public/doc/project/core.md)
- [View guide](public/doc/project/view.md)
- [Handler guide](public/doc/project/handler.md)
- [Operator guide](public/doc/project/operator.md)
- [Usage guide](public/doc/project/usage.md)
- [App test guide](public/doc/project/app-tests.md)

Public headers also contain Doxygen comments for classes and functions.

## What Should Stay In This README

Keep this README focused on fast orientation. It should answer:

- What is this project?
- What problem does it solve?
- How is the repo organized?
- How do I build it?
- How do I run the readiness tests?
- How do I search the code?
- Where are the deeper docs?
- What operators and subsystems currently exist?
- What known behavior matters, such as `Conv2DRelu` currently being convolution plus bias, not activation?

Detailed class-by-class explanation should live in `public/doc/project/*.md` and Doxygen comments, not only in this README.

## Important Notes

- Shape comes before binding.
- Binding gives a tensor CPU/GPU memory.
- Operators do not allocate outputs automatically.
- Raw pointer IO copy overloads copy data only and should not change tensor shape.
- Most custom F16 kernels assume element counts are multiples of 8 because they use `uint4` vectorized memory access.
- `Conv2DRelu` currently performs convolution plus bias. It creates a ReLU descriptor but does not call cuDNN activation in `forward()`.
