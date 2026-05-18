# HANDLER::IO

> Namespace: [HANDLER](index.md)

`IO` binds `VIEW::Math` tensors to `Cpu`/`Cuda` arenas and copies data between host and device.

## Fields

```cpp
HANDLER::Cpu *handleCpu;
HANDLER::Cuda *handleGpu;
CORE::errIO err;
```

## Constructors

```cpp
IO(HANDLER::Cpu& handleCpu, HANDLER::Cuda& handleGpu);
IO(HANDLER::Cpu& handleCpu);
IO(HANDLER::Cuda& handleGpu);
```

The first is the normal CPU+GPU workflow. The others are for CPU-only or GPU-only handling.

## Error Functions

```cpp
CORE::errIO getErr();
CORE::errIO peekErr() const;
void clearErr();
```

- `getErr()` returns and clears the current error.
- `peekErr()` returns without clearing.
- `clearErr()` resets to `CORE::ioSuccess`.

## Handler Setters

```cpp
void setHandler(HANDLER::Cpu& handleCpu, HANDLER::Cuda& handleGpu);
void setHandler(HANDLER::Cpu& handleCpu);
void setHandler(HANDLER::Cuda& handleGpu);
```

Replace active memory arenas.

## Binding

### `bindCpu`

```cpp
void bindCpu(VIEW::Math& math);
```

Allocates CPU memory for `math`.

Required:

- shape must already be set

Behavior:

- computes `count`
- computes aligned `bytes`
- allocates from `Cpu`
- stores CPU pointer and offset in `math`

### `bindGpu`

Allocates GPU memory for `math`.

### `bind`

```cpp
void bind(VIEW::Math& math);
```

Calls `bindCpu(math)` and `bindGpu(math)`.

### `unbind`

```cpp
void unbind(VIEW::Math& math);
```

Clears tensor pointers, offsets, bytes, count, and shape.

It does not free arena slices individually.

## Copy Functions

Raw-pointer overloads copy data only and do not change shape.

Math-to-Math overloads also copy layout from source to destination.

### Host To Host

```cpp
void copyHostToHost(VIEW::Math& dstMath, VIEW::Math& srcMath);
void copyHostToHost(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype = VIEW::CHAR);
void copyHostToHost(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype = VIEW::CHAR);
```

Use for CPU memory copies.

### Host To Device

```cpp
void copyHostToDevice(VIEW::Math& math);
void copyHostToDevice(VIEW::Math& dstMath, VIEW::Math& srcMath);
void copyHostToDevice(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype = VIEW::CHAR);
void copyHostToDevice(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype = VIEW::CHAR);
```

Use to send CPU data to GPU.

Common:

```cpp
io.copyHostToDevice(x);
```

### Device To Host

```cpp
void copyDeviceToHost(VIEW::Math& math);
void copyDeviceToHost(VIEW::Math& dstMath, VIEW::Math& srcMath);
void copyDeviceToHost(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype = VIEW::CHAR);
void copyDeviceToHost(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype = VIEW::CHAR);
```

Use to bring GPU results back to CPU.

### Half To CPU Float

```cpp
void copyHalfToCpuFloat(VIEW::Math& dstMath, VIEW::Math& srcMath);
```

Converts a GPU F16 tensor to CPU FLOAT tensor.

Useful for readable printing:

```cpp
VIEW::Math outFloat;
io.copyHalfToCpuFloat(outFloat, out);
io.printData(outFloat);
```

### Device To Device

```cpp
void copyDeviceToDevice(VIEW::Math& dstMath, VIEW::Math& srcMath);
void copyDeviceToDevice(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype = VIEW::CHAR);
void copyDeviceToDevice(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype = VIEW::CHAR);
```

Use for GPU memory copies.

## Printing

```cpp
void printData(VIEW::Math& math, const char* file = __FILE__, int line = __LINE__) const;
```

Prints CPU memory using tensor shape.

## Error Printing

```cpp
void info(CORE::State level = CORE::WARN, const char *file = __FILE__, int line = __LINE__) const;
```

Prints current IO error.

