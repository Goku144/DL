# CORE State, Errors, And Helpers

> Namespace: [CORE](index.md)

Defined in:

```text
public/inc/CORE/State.hpp
```

## `CUDA_CPU`

```cpp
#define CUDA_CPU 0
```

Controls CPU allocation backend.

- `0`: `aligned_alloc`
- `1`: `cudaMallocHost`

## `CORE::Aligne`

```cpp
ALIGNE_TO_32
ALIGNE_TO_64
ALIGNE_TO_128
ALIGNE_TO_256
```

Alignment constants. Tensor memory generally uses `ALIGNE_TO_256`.

## `CORE::MemorySize`

```cpp
MEMORY_32_MB
MEMORY_64_MB
MEMORY_128_MB
MEMORY_256_MB
MEMORY_512_MB
MEMORY_1_GB
```

Convenient arena/workspace sizes.

## `CORE::errIO`

Used by `HANDLER::IO`, `HANDLER::Cpu`, and `HANDLER::Cuda`.

Important values:

- `ioSuccess`
- `ioErrCopyToHost`
- `ioErrCopyToDevice`
- `ioErrOutOfMemory`
- `ioErrOutOfBound`
- `ioErrNull`
- `ioErrInvalidValue`
- `ioErrInvalidState`

## `CORE::errFile`

Used by `HANDLER::File`.

Important values:

- `fileSuccess`
- `fileErrRead`
- `fileErrWrite`
- `fileErrReadCsv`
- `fileErrReadImg`
- `fileErrOpen`
- `fileErrClose`
- `fileErrNull`
- `fileErrInvalidState`
- `fileErrIO`

## `CORE::errWorkspace`

Used by `HANDLER::Workspace`.

Important values:

- `workspaceSuccess`
- `workspaceErrCudnnCreate`
- `workspaceErrCudnnDestroy`
- `workspaceErrCublasLtCreate`
- `workspaceErrCublasLtDestroy`
- `workspaceErrScratchAlloc`
- `workspaceErrScratchFree`
- `workspaceErrScratchOutOfBound`
- `workspaceErrNull`
- `workspaceErrStreamCreate`
- `workspaceErrStreamDestroy`
- `workspaceErrCudnnSetStream`

## `CORE::State`

Log severity:

```cpp
INFO
WARN
FATAL
```

## Logging Macros

```cpp
logInfo(file, line, fmt, ...)
logWarn(file, line, fmt, ...)
logFatal(file, line, fmt, ...)
```

All call:

```cpp
printState(level, file, line, fmt, ...)
```

## `printState`

```cpp
void printState(State level, const char *file, int line, const char *fmt, ...);
```

Prints a formatted project log message.

## `aligne`

```cpp
inline size_t aligne(size_t x, CORE::Aligne aligneTo);
```

Rounds `x` upward to the requested alignment.

## `ALIGNE`

```cpp
#define ALIGNE(x, aligneTo) aligne(x, aligneTo)
```

Macro wrapper used throughout the project.

