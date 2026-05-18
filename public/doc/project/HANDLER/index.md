# HANDLER Namespace

> **Reading Path**  
> Home: [Project Manual](../index.md) | Previous: [VIEW](../VIEW/index.md) | Next: [OPERATOR](../OPERATOR/index.md)

`HANDLER` owns runtime resources. `VIEW` describes tensors, but `HANDLER` allocates memory, binds tensors, copies data, loads files, and owns execution handles.

## Classes

| Class | File | Purpose |
|---|---|---|
| `HANDLER::Cpu` | [cpu.md](cpu.md) | CPU arena allocator. |
| `HANDLER::Cuda` | [cuda.md](cuda.md) | GPU arena allocator. |
| `HANDLER::IO` | [io.md](io.md) | Tensor binding and memory copy helper. |
| `HANDLER::File` | [file.md](file.md) | Raw file, CSV, and image loading helper. |
| `HANDLER::Workspace` | [workspace.md](workspace.md) | CUDA stream, cuDNN, cuBLASLt, and scratch workspace. |

## Namespace Role

`HANDLER` is the ownership layer:

- `Cpu` and `Cuda` own memory arenas.
- `IO` connects `VIEW::Math` objects to those arenas.
- `File` loads external data and uses `IO`.
- `Workspace` owns GPU execution resources shared by operators.

Typical construction:

```cpp
HANDLER::Cpu cpu(CORE::MEMORY_32_MB);
HANDLER::Cuda gpu(CORE::MEMORY_32_MB);
HANDLER::IO io(cpu, gpu);
HANDLER::File file;
HANDLER::Workspace workspace(io, file, CORE::MEMORY_32_MB);
```

---

> **Class Pages**  
> [Cpu](cpu.md) | [Cuda](cuda.md) | [IO](io.md) | [File](file.md) | [Workspace](workspace.md)

