# HANDLER::Cpu

> Namespace: [HANDLER](index.md)

CPU arena allocator.

## Fields

```cpp
void *data;
size_t offset;
size_t capacity;
```

## Constructor

```cpp
Cpu(size_t capacity = CORE::MEMORY_1_GB, const char* file = __FILE__, int line = __LINE__);
```

Allocates one CPU arena.

- `capacity`: bytes to allocate
- `file`, `line`: used for fatal allocation logs

Behavior:

- if `CORE::CUDA_CPU == 0`, uses `aligned_alloc`
- if `CORE::CUDA_CPU == 1`, uses `cudaMallocHost`
- stores arena capacity

Failure:

- logs fatal if allocation fails

## Destructor

```cpp
~Cpu();
```

Frees the CPU arena with `free` or `cudaFreeHost`.

## `getData`

```cpp
void *getData();
```

Returns the base CPU arena pointer.

## `getOffset`

```cpp
size_t getOffset();
```

Returns the current allocation offset.

## `getCapacity`

```cpp
size_t getCapacity();
```

Returns arena capacity in bytes.

## `allocate`

```cpp
CORE::errIO allocate(void **cpuPtr, size_t& offset, size_t capacity);
```

Allocates an aligned slice from the CPU arena.

- `cpuPtr`: receives slice pointer
- `offset`: receives slice offset
- `capacity`: requested bytes

Behavior:

- aligns request to `CORE::ALIGNE_TO_256`
- checks available space
- writes pointer and offset
- advances internal offset

Returns:

- `CORE::ioSuccess`
- `CORE::ioErrNull`
- `CORE::ioErrOutOfBound`

## `reset`

```cpp
void reset();
```

Resets allocation offset to zero.

Existing tensor pointers should be treated as stale if the arena is reused.

