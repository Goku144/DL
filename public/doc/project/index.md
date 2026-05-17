# Project Guide

This project is a small CUDA neural-network runtime built around four ideas:

- **CORE** defines common error codes, memory sizes, alignment, and logging.
- **VIEW** defines tensor metadata and tensor handles.
- **HANDLER** owns memory, file IO, CUDA/cuDNN/cuBLASLt handles, and data movement.
- **OPERATOR** implements GPU operations that read and write `VIEW::Math` tensors.

The usual flow is:

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
```

After shape and binding, fill CPU memory, copy to GPU, run an operator, and copy results back if needed.

```cpp
__half *xCpu = (__half *)x.getCpuPtr();
xCpu[0] = __float2half(1.0f);
io.copyHostToDevice(x);
```

Read the guides in this folder in this order:

1. [core.md](core.md)
2. [view.md](view.md)
3. [handler.md](handler.md)
4. [operator.md](operator.md)
5. [app-tests.md](app-tests.md)
6. [usage.md](usage.md)

