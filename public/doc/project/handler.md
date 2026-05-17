# HANDLER

Navigation: Previous: [view.md](view.md) | Next: [operator.md](operator.md)

`public/inc/HANDLER` owns memory, IO, files, and GPU library handles.

## `HANDLER::Cpu`

Owns one aligned CPU arena. `allocate()` returns slices from that arena.

It does not free individual tensors. `reset()` moves the allocation offset back to zero.

## `HANDLER::Cuda`

Owns one CUDA device arena. `allocate()` returns aligned device slices.

Like `Cpu`, it is arena-based and does not free individual tensor slices.

## `HANDLER::IO`

`IO` binds `VIEW::Math` tensors and copies data.

Important functions:

- `bindCpu(math)`: allocates CPU memory
- `bindGpu(math)`: allocates GPU memory
- `bind(math)`: allocates both
- `copyHostToDevice(math)`: copies the tensor CPU buffer to its GPU buffer
- `copyDeviceToHost(math)`: copies the tensor GPU buffer to its CPU buffer
- `copyHalfToCpuFloat(dst, src)`: converts an F16 GPU tensor into a FLOAT CPU tensor
- `printData(math)`: prints CPU memory

The raw pointer copy overloads copy data only and do not change tensor shape.

## `HANDLER::Workspace`

Owns:

- CUDA stream
- cuDNN handle
- cuBLASLt handle
- scratch GPU buffer

Operators use `Workspace` to launch kernels and library calls on the same stream.

## `HANDLER::File`

Reads raw files, CSV metadata, and image batches. It can convert loaded uint8 image data into F16 GPU tensors.

Navigation: Previous: [view.md](view.md) | Next: [operator.md](operator.md)
