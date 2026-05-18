# DL

DL is a small CUDA/C++ neural-network runtime. It is built around explicit tensor views, arena-style CPU/GPU memory handlers, and GPU operators backed by CUDA kernels, cuDNN, and cuBLASLt.

The project is intentionally low-level: you create handlers, shape tensors, bind memory, copy data, attach tensors to operators, then launch operations.

## Principal Folders

```text
CORE/
  Shared constants, errors, alignment, and logging.

VIEW/
  Tensor shape metadata and tensor handles.

HANDLER/
  CPU/GPU memory arenas, IO, files, and execution workspace.

OPERATOR/
  GPU math operators.

MODEL/
  Future model/layer abstraction. Not implemented yet.
```

In the repo:

```text
public/inc/CORE             Public CORE headers
public/inc/VIEW             Public VIEW headers
public/inc/HANDLER          Public HANDLER headers
public/inc/OPERATOR         Public OPERATOR headers
lib/src                     Implementations and CUDA kernels
app/src/app.cu              User application / experiment entrypoint
public/doc/project          Project manual
Makefile                    Build, run, dataset, and clean targets
```

## Architecture In One Pass

`CORE` defines the vocabulary.

`VIEW` defines tensor metadata and tensor handles.

`HANDLER` owns memory and execution resources.

`OPERATOR` performs GPU work on already-bound `VIEW::Math` tensors.

`MODEL` will later orchestrate layers, parameters, forward propagation, backward propagation, and optimizers.

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

## Build

Build the library:

```bash
make lib
```

Build and run `app/src/app.cu`:

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

Use `rg` first.

Find a class:

```bash
rg -n "class .*MatrixMulBias|MatrixMulBias" public/inc lib/src app/src
```

Find public declarations:

```bash
rg -n "class|void|getErr|setOperand|forward|backward" public/inc
```

Find implementations:

```bash
rg -n "OPERATOR::Relu|reluKernel" lib/src/OPERATOR
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

## Documentation

Start here:

- [Project manual index](public/doc/project/index.md)
- [Architecture](public/doc/project/architecture.md)
- [CORE reference](public/doc/project/CORE/index.md)
- [VIEW reference](public/doc/project/VIEW/index.md)
- [HANDLER reference](public/doc/project/HANDLER/index.md)
- [OPERATOR reference](public/doc/project/OPERATOR/index.md)
- [MODEL plan](public/doc/project/MODEL/index.md)
- [Usage](public/doc/project/usage.md)
- [Full overview](public/doc/project/overview.md)

Public headers also contain Doxygen comments for classes and functions.

## Important Notes

- Shape comes before binding.
- Binding gives a tensor CPU/GPU memory.
- Operators do not allocate outputs automatically.
- Raw pointer IO copy overloads copy data only and should not change tensor shape.
- Most custom F16 kernels assume element counts are multiples of 8 because they use `uint4` vectorized memory access.
- `Conv2DRelu` currently performs convolution plus bias. It creates a ReLU descriptor but does not call cuDNN activation in `forward()`.
- `app/src/app.cu` is user application space. It can be used for experiments or tests, but it is not a core framework layer.
