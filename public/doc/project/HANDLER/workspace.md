# HANDLER::Workspace

> Namespace: [HANDLER](index.md)

Shared execution context for operators.

## Fields

```cpp
HANDLER::IO *io;
HANDLER::File *file;
CORE::errWorkspace err;
cudnnHandle_t cudnnHandle;
cublasLtHandle_t cublasLtHandle;
cudaStream_t stream;
void *scratchPtr;
size_t scratchBytes;
```

## Constructor

```cpp
Workspace(HANDLER::IO& io, HANDLER::File& file, size_t scratchBytes);
```

Creates:

- CUDA stream
- cuDNN handle
- cuBLASLt handle
- scratch GPU memory

Also connects `file` to `io`.

## Destructor

```cpp
~Workspace();
```

Destroys scratch memory, cuBLASLt handle, cuDNN handle, and stream.

## Resource Getters

```cpp
HANDLER::IO& getIO();
HANDLER::File& getFile();
cudnnHandle_t getCudnnHandle();
cublasLtHandle_t getCublasLtHandle();
cudaStream_t getStream();
```

Return shared objects/handles used by operators.

## Scratch

```cpp
void *getScratch(size_t requiredBytes);
size_t getScratchBytes() const;
```

`getScratch` returns scratch memory if request fits.

Returns `NULL` if:

- request is zero
- scratch pointer is null
- request exceeds scratch capacity

## Error Functions

```cpp
CORE::errWorkspace getErr();
CORE::errWorkspace peekErr() const;
void clearErr();
void info(CORE::State level = CORE::WARN, const char *file = __FILE__, int line = __LINE__) const;
```

Same pattern as `IO` and `File`.

