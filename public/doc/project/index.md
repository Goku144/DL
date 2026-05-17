# Project Guide

> **Reading Path**  
> Home: **Project Guide** | Previous: None | Next: [CORE](core.md)

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

## Reading Order

| Step | Guide | Purpose |
|---:|---|---|
| 1 | [CORE](core.md) | Shared errors, alignment, memory sizes, and logging. |
| 2 | [VIEW](view.md) | Tensor metadata and tensor handles. |
| 3 | [HANDLER](handler.md) | Memory arenas, IO, files, and execution workspace. |
| 4 | [OPERATOR](operator.md) | GPU operations and their tensor contracts. |
| 5 | [App Tests](app-tests.md) | The readiness harness and what it validates. |
| 6 | [Usage](usage.md) | The shortest practical recipe for using the runtime. |
| 7 | [Full Overview](huge.md) | The long-form explanation tying everything together. |

---

> **Continue Reading**  
> Next: [CORE](core.md)
