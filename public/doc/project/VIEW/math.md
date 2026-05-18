# VIEW::Math

> Namespace: [VIEW](index.md)

`Math` is the tensor handle passed to handlers and operators.

It does not own memory. It stores pointers into memory owned by `HANDLER::Cpu` and `HANDLER::Cuda`.

## Fields

```cpp
void *cpuPtr;
void *gpuPtr;
size_t cpuOffset;
size_t gpuOffset;
size_t bytes;
size_t count;
Shape layout;
```

## Constructors

```cpp
Math(VIEW::Shape& layout);
Math();
```

The layout constructor copies shape metadata.

## Destructor

```cpp
~Math();
```

Destroys the view only.

## Getters

```cpp
void *getCpuPtr() const;
void *getGpuPtr() const;
size_t getCpuOffset() const;
size_t getGpuOffset() const;
size_t getBytes() const;
size_t getCount() const;
VIEW::Shape& getLayout();
```

Return tensor pointers, allocation metadata, and shape.

## Setters

```cpp
void setCpuPtr(void *cpuPtr);
void setGpuPtr(void *gpuPtr);
void setCpuOffset(size_t cpuOffset);
void setGpuOffset(size_t gpuOffset);
void setBytes(size_t bytes);
void setCount(size_t count);
void setLayout(VIEW::Shape& layout);
```

Mostly used by `HANDLER::IO`.

## Indexed Data Helpers

```cpp
void *getCpuDataAt(int i = 0, int j = 0, int k = 0, int l = 0);
void *getGpuDataAt(int i = 0, int j = 0, int k = 0, int l = 0);
```

Return a pointer to an indexed CPU/GPU element.

## `info`

```cpp
void info(const char* file = __FILE__, int line = __LINE__) const;
```

Prints offsets, byte size, element count, and layout.

