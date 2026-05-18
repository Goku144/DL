# HANDLER::Cuda

> Namespace: [HANDLER](index.md)

GPU arena allocator.

## Fields

```cpp
void *data;
size_t offset;
size_t capacity;
```

## Constructor

```cpp
Cuda(size_t capacity = CORE::MEMORY_1_GB, const char* file = __FILE__, int line = __LINE__);
```

Allocates one CUDA device arena with `cudaMalloc`.

Failure:

- logs fatal if allocation fails

## Destructor

```cpp
~Cuda();
```

Frees the GPU arena with `cudaFree`.

## `getData`

Returns the base device pointer.

## `getOffset`

Returns current allocation offset.

## `getCapacity`

Returns arena capacity in bytes.

## `allocate`

```cpp
CORE::errIO allocate(void **gpuPtr, size_t& offset, size_t capacity);
```

Allocates an aligned slice from the GPU arena.

Returns:

- `CORE::ioSuccess`
- `CORE::ioErrNull`
- `CORE::ioErrOutOfBound`

## `reset`

Resets allocation offset to zero.

